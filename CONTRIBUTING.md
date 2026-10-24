# Contributing to Vade

Read [AGENTS.md](AGENTS.md) before changing code. Vade is a standard-library-only
Rust inference engine for NVIDIA DGX Spark / GB10. The v0.1.0 release target is
`nvidia/Qwen3.8-Flash-Next-FP4-Local`, text-only, on one Spark by default, with
drafting ON. The project supports configurations from one to N Sparks.

## Build and validate

Use the pinned toolchains from [CI](.github/workflows/ci.yml). Install them once:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

There is no root Cargo workspace. Run the following from `engine/`, on macOS
or Linux. Build kernels first on either OS: an old PTX artifact can fail a host
check even on a Mac. Set `RUSTFLAGS` to `-D warnings` to match CI; the kernel
builder supplies its own device flags. The first kernel build needs the Rust
components and build-std cache available; see [validation](engine/docs/validation.md).

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
python3 ../tools/check-doc-links.py
```

To build the CLI locally, also from `engine/`:

```sh
cargo +1.99.0 build --release --offline --manifest-path ../cli/Cargo.toml
```

The binary is `cli/target/release/vade`. Building it does not install or launch
an engine. For source installation on a Spark, follow [INSTALL.md](INSTALL.md).

Run the affected workspace's tests before opening a PR. The complete CI host
recipes, including the Linux CUDA stub, separate TP kernel build, and Linux
cross-target checks, are in [AGENTS.md](AGENTS.md) and
[engine validation](engine/docs/validation.md). For the engine workspace, use
the explicit CI selection; default members omit packages:

```sh
cargo +1.99.0 check --offline --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

These are host checks. Device-only tests are skipped without a GPU; a host
PASS does not prove GPU correctness, exactness, memory use, or performance.

## Rules and formatting

- Keep zero third-party dependencies: only the Rust standard library and
  existing first-party path crates. Registry, git, external path, dev, and
  build dependencies count too.
- Add no new `unsafe`. Existing unsafe code does not grant an exception.
- Keep behavior model-generic: derive it from checkpoint configuration and
  tensor metadata, without model-name dispatch or model-specific tuning.
  See [adding a model](docs/adding-a-model.md).
- Preserve the surrounding source style manually. The code uses deliberate
  dense formatting; do not run `cargo fmt`, `clippy --fix`, or formatting
  sweeps. Neither formatter nor clippy is a CI gate.
- Keep new Rust files at most 1,500 lines and avoid new literal world/rank-count
  branches. Do not raise the [ratchet ceilings](cli/ratchet-baseline.txt);
  lower a ceiling in the same commit when its count drops.
- Never commit secrets, credentials, model weights, generated PTX, or private
  machine data. Keep local results outside the tracked tree. Preserve licence
  notices and document any permitted third-party non-code material.

## Exactness and evidence

A *gate* below is a set of checks that must pass before a change merges.

Exactness means **identical token IDs every run on every Spark with the same
setup**. Unspecified sampling means greedy decoding. Results must agree for
alone, batched, cached, and chunked execution, and drafting ON must equal OFF.
Do not use floating-point atomics, arrival-order reductions, or reductions
whose split order changes with batching, sequence length, or occupancy.

Engine arithmetic, scheduling, caching, and transport changes need Spark
validation before merge: at least three repeats and comparison across at least
two Sparks with the same setup. Preserve main's results when arithmetic order
is unchanged. An intentional arithmetic-order change needs an explicit switch,
the full exactness gate, and quality comparison against main. Host-only CLI or
documentation changes do not require a Spark merely to check documentation.

The [gate guide](cli/docs/test.md) describes the existing fixture-driven gate;
its legacy Flash benchmark is not proof that FP4-Local is qualified. The
`vade verify` command requires a running engine and compares repeated greedy
token IDs with drafting OFF and ON; its receipt alone does not cover every
batch/cache/chunk invariant. Record which checks ran and which remain untested.

## Commits and pull requests

Make small commits with clear imperative messages describing the change and
its purpose. Keep one change per PR; leave unrelated cleanup for another PR.
Open the PR against `main` in the repository named by
[`VADE_REPO_URL`](cli/src/repository.rs).
Explain the problem, resulting behavior, commands run, and any validation
limits. Attach the required exactness receipts for engine changes.

Performance claims need reproducible receipts: source revision, checkpoint
identity, hardware and topology, request and sampling settings, drafting and
cache modes, repeated measurements, and the baseline comparison. Sanitize
machine details and secrets; do not commit weights or raw tensor dumps. A
change without measured performance evidence should make no speed claim.
