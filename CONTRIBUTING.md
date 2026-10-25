# Contributing to Vade

Vade is an inference engine for the NVIDIA DGX Spark / GB10, written in Rust
using only the standard library. Version 0.1.0 targets
`nvidia/Qwen3.8-Flash-Next-FP4-Local`. It handles text only and runs on one
Spark by default; the project supports setups from one to N Sparks. Drafting
is on by default: an extra model head proposes tokens and the model verifies
them.

Read [AGENTS.md](AGENTS.md) before you change code.

## Build and validate

Use the pinned toolchains from [CI](.github/workflows/ci.yml). Install them once:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

There is no root Cargo workspace. Run the commands below from `engine/`, on
macOS or Linux. Build kernels first on either OS: a leftover PTX file
(compiled GPU kernel code) can fail a host check, even on a Mac. Set
`RUSTFLAGS` to `-D warnings` to match CI; the kernel builder adds its own
device flags. The first kernel build needs the Rust components and the
build-std cache (Rust's core library built for the GPU target) in place; see
[validation](engine/docs/validation.md).

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
python3 ../tools/check-doc-links.py
```

To build the CLI yourself, also from `engine/`:

```sh
cargo +1.99.0 build --release --offline --manifest-path ../cli/Cargo.toml
```

The binary lands at `cli/target/release/vade`. Building it does not install or
start an engine. To install from source on a Spark, follow
[INSTALL.md](INSTALL.md).

Run the tests for the workspace you changed before you open a PR. The full CI
host recipes, including the Linux CUDA stub, the separate TP kernel build and
the Linux cross-target checks, are in [AGENTS.md](AGENTS.md) and
[engine validation](engine/docs/validation.md). For the engine workspace, use
the same explicit package selection as CI, because default members leave
packages out:

```sh
cargo +1.99.0 check --offline --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

These are host checks. Device-only tests are skipped when there is no GPU, so
a host PASS does not prove GPU correctness, exactness, memory use or
performance.

## Rules and formatting

- Keep zero third-party dependencies: only the Rust standard library and the
  first-party path crates already here. Registry, git, external path, dev and
  build dependencies all count.
- Add no new `unsafe`. Existing unsafe code is not an exception.
- Keep behavior model-generic. Derive it from the checkpoint configuration and
  tensor metadata, with no dispatch on the model name and no model-specific
  tuning. See [adding a model](docs/adding-a-model.md).
- Match the surrounding source style by hand. The code is formatted densely on
  purpose, so don't run `cargo fmt`, `clippy --fix` or any formatting sweep.
  CI gates on neither the formatter nor clippy.
- Keep new Rust files to 1,500 lines at most. Avoid new branches on literal
  world or rank counts (code that checks for a fixed number of Sparks).
  Don't raise the [ratchet ceilings](cli/ratchet-baseline.txt); when a count
  drops, lower its ceiling in the same commit.
- Never commit secrets, credentials, model weights, generated PTX or private
  machine data. Keep local results outside the tracked tree. Keep licence
  notices intact, and document any permitted third-party non-code material.

## Exactness and evidence

Here a *gate* means a set of checks that must pass before a change merges.

Exactness means **identical token IDs every run on every Spark with the same
setup**. When a request doesn't specify sampling, decoding is greedy. A request
must give the same result whether it runs alone, batched, cached or chunked,
and drafting ON must match drafting OFF. Don't use floating-point atomics,
reductions that depend on arrival order, or reductions whose split order
changes with batching, sequence length or occupancy.

Changes to engine arithmetic, scheduling, caching or transport need validation
on Sparks before they merge: at least three repeats, compared across at least
two Sparks with the same setup. If you don't change the arithmetic order,
your change must keep main's results. A deliberate change to arithmetic order
needs an explicit switch, the full exactness gate and a quality comparison
against main. Host-only CLI and documentation changes don't need a Spark.

The [gate guide](cli/docs/test.md) describes the existing gate, which is
driven by fixtures (saved test inputs). That gate's legacy Flash benchmark
doesn't prove that FP4-Local is qualified.

`vade verify` needs a running engine; it compares repeated greedy token IDs
with drafting OFF and ON. Its receipt (the saved record of what the run
checked) doesn't cover every batch, cache and chunk invariant on its own.
Write down which checks ran and which are still untested.

## Commits and pull requests

Make small commits with clear, imperative messages that say what changed and
why. Keep one change per PR, and save unrelated cleanup for another PR. Open
the PR against `main` in the repository named by
[`VADE_REPO_URL`](cli/src/repository.rs). Explain the problem, the resulting
behavior, the commands you ran and any limits of your validation. For engine
changes, attach the required exactness receipts.

Performance claims need reproducible receipts: source revision, checkpoint
identity, hardware and topology, request and sampling settings, drafting and
cache modes, repeated measurements and the baseline comparison. Strip machine
details and secrets from them, and don't commit weights or raw tensor dumps. If
you haven't measured performance, don't make a speed claim.
