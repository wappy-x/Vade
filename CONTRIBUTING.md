# Contributing to Vade

Vade is an inference engine for the NVIDIA DGX Spark / GB10, written in Rust
using only the standard library. Version 0.1.0 runs one model,
`nvidia/Qwen3.8-Flash-Next-FP4-Local`, as text only. It runs on a single Spark
out of the box and can spread across more Sparks when you link them. Drafting
is on by default: an extra model head guesses the next few tokens, and the
model checks them in one pass.

This page walks through checking a change on your own computer, the house
rules, what exactness asks of engine changes, and how to send a pull request.
[AGENTS.md](AGENTS.md) has the full rules and every CI command; read it before
you change code. The [glossary](docs/GLOSSARY.md) explains the terms used
here.

## Build and validate

You can run every host check on this page on macOS or Linux. You only need a
Spark for the device checks described under
[exactness](#exactness-and-evidence).

### Install the toolchains

[CI](.github/workflows/ci.yml) pins two Rust toolchains: 1.99.0 for the host
code and nightly-2026-10-01 for the GPU kernels. Install them once, while
you're online:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

To match CI, which denies warnings, run `export RUSTFLAGS="-D warnings"` in
your shell. You don't need GPU flags: the kernel builder adds its own device
flags.

### Check your change

There is no root Cargo workspace, so run these commands from `engine/`.

Build the kernels first, even on a Mac: the host checks fail if the compiled
kernels (PTX files) left from an earlier build are stale. The first build also
compiles Rust's core library for the GPU target (the build-std cache) and can
need downloads, so run it once while you're online. After that, everything
here works offline. [Validation](engine/docs/validation.md#on-any-computer-host-checks)
has the details.

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
python3 ../tools/check-doc-links.py
```

That block checks the CLI, the harness and the benchmark tools, runs the
ratchet test (explained under [rules](#rules-and-formatting)) and checks the
documentation links. `--locked` makes Cargo use the committed `Cargo.lock`
files as they are, the way CI does. If you changed the engine, check and test
the engine workspace too. Pass `--workspace` as CI does; a plain `cargo test`
only covers the default members and skips some crates. `--all-targets` doesn't
run doc tests, so CI runs them in a separate step, which is the last line here:

```sh
cargo +1.99.0 check --offline --locked --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --locked --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
cargo +1.99.0 test --offline --locked --doc --workspace --exclude spark-rust-kernels
```

Run the tests for the workspace you changed before you open a PR. The rest of
CI's host recipes are in [AGENTS.md](AGENTS.md) and
[engine validation](engine/docs/validation.md): the CUDA stub for Linux
machines without a GPU, the separate build of the TP (tensor parallelism)
kernels and the Linux cross-target checks.

These are host checks. Device-only tests are skipped when there is no GPU, so
a host PASS does not prove GPU correctness, exactness, memory use or
performance.

### Build the CLI

To build the `vade` command yourself, also from `engine/`:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path ../cli/Cargo.toml
```

The binary lands at `cli/target/release/vade`. Building it does not install or
start an engine. To install from source on a Spark, follow
[INSTALL.md](INSTALL.md).

## Rules and formatting

- **No third-party dependencies.** Use only the Rust standard library and the
  first-party path crates already here. Registry, git, external path, dev and
  build dependencies all count.
- **No new `unsafe`.** The unsafe code that already exists at the CUDA,
  kernel and transport boundaries doesn't make room for more. The ratchet test
  (below) doesn't count unsafe code, so check your diff for it yourself.
- **Model-generic behavior.** Derive it from the checkpoint configuration and
  tensor metadata, with no dispatch on the model name and no model-specific
  tuning. See [adding a model](docs/adding-a-model.md).
- **Formatting by hand.** The code is formatted densely on purpose, so match
  the surrounding style and don't run `cargo fmt`, `clippy --fix` or any
  formatting sweep. CI gates on neither the formatter nor clippy.
- **Small files.** Keep new Rust files to 1,500 lines at most.
- **No fixed Spark counts.** Read the number of Sparks from the topology.
  Never branch on a literal world size or rank count.
- **The ratchet only goes down.** The ratchet test counts third-party
  dependencies, Rust files over 1,500 lines and those fixed-count branches.
  It fails when a count rises above its ceiling. Don't raise the
  [ratchet ceilings](cli/ratchet-baseline.txt); when a count drops, lower its
  ceiling in the same commit.
- **Nothing private.** Never commit secrets, credentials, model weights,
  generated PTX or private machine data. Keep local results outside the
  tracked tree. Keep license notices intact. If you add non-code third-party
  material (allowed only when its license permits), list it in
  [NOTICE](NOTICE), or in [engine/NOTICE](engine/NOTICE) for files under
  `engine/`.

## Exactness and evidence

Exactness means **identical token IDs every run on every Spark with the same
setup**. A request must give the same result whether it runs alone, batched,
cached or chunked, and drafting ON must match drafting OFF. When a request
doesn't specify sampling, decoding is greedy (always the most likely next
token).

To keep that true, don't use floating-point atomics, reductions that depend
on arrival order, or reductions whose split order changes with batching,
sequence length or occupancy.

### Which changes need a Spark

Host-only CLI and documentation changes don't need a Spark. Changes to engine
arithmetic, scheduling, caching or transport must pass a gate on real hardware
before they merge: run the change at least three times, and compare the
results across at least two Sparks with the same setup. If you have one
Spark, run the repeats on it and say so in the PR. The change can't merge
until the comparison on a second Spark is done.

- If your change leaves the arithmetic order alone, it must keep main's
  results: the same token IDs.
- If you change the arithmetic order on purpose, put it behind an explicit
  switch, and bring the full exactness gate and a quality comparison against
  main.

### Tools for the checks

Two tools cover the checks today:

1. Run `vade verify` against a running engine. It compares repeated greedy
   token IDs with drafting OFF and ON, and saves a receipt (a record of what
   it checked).
2. Run `vade test determinism` for the batch, cache and chunk cases; the
   [gate guide](cli/docs/test.md) shows how. It runs saved test inputs
   (fixtures) through the older `flash-bench` benchmark, so a pass there
   doesn't yet prove that FP4-Local is qualified.

Neither one covers everything yet, so list in the PR which checks you ran and
which you didn't.

## Commits and pull requests

Make small commits with clear, imperative messages that say what changed and
why. Keep one change per PR, and save unrelated cleanup for another PR. Open
the PR against `main` at
[github.com/wappy-x/vade](https://github.com/wappy-x/vade).

In the PR, explain the problem, the resulting behavior, the commands you ran
and any limits of your validation. For engine changes, attach the receipts
from `vade verify` and `vade test determinism`.

If users would notice the change, add one plain line about it to the
Unreleased section of [CHANGELOG.md](CHANGELOG.md).

### How a pull request lands

1. **CI passes.** [CI](.github/workflows/ci.yml) runs the host checks above
   on every pull request against `main`. A red run blocks the change.
2. **A maintainer reviews it.** [CODEOWNERS](.github/CODEOWNERS) requests
   the review automatically. Expect questions about exactness, the house
   rules and anything the PR says it didn't check.
3. **Engine changes pass the device gate.** A change to engine arithmetic,
   scheduling, caching or transport also needs the runs on real Sparks
   described under [exactness](#exactness-and-evidence) before it can land.
4. **The maintainer applies it.** `main` here is published from the
   maintainers' development tree, one commit per update, so the merge
   button isn't used. The maintainer applies the accepted change there, CI
   runs again on the result, and the change reaches `main` in the next
   published update. The PR is then closed, not merged.

If you haven't measured performance, don't make a speed claim. A performance
claim needs reproducible receipts that record:

- the source revision and checkpoint identity
- the hardware and topology
- the request and sampling settings
- the drafting and cache modes
- repeated measurements and the baseline comparison

Strip machine details and secrets from them, and don't commit weights or raw
tensor dumps.
