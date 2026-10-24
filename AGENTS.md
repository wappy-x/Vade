# Working on Vade

Vade is a std-only Rust inference engine for NVIDIA DGX Spark / GB10. The
v0.1.0 release target is `nvidia/Qwen3.8-Flash-Next-FP4-Local`: text-only,
one Spark by default, with 1..N Sparks supported by the topology. Drafting is
ON by default; unspecified sampling means greedy decoding. The source
repository URL is defined by [`VADE_REPO_URL`](cli/src/repository.rs).

## House rules

- **Zero third-party dependencies.** Use the Rust standard library and existing
  first-party crates. Registry, git, off-tree path, build/dev dependencies and
  lockfile sources all count. Do not add a dependency to solve a small task.
- **No new unsafe.** Do not add unsafe blocks, functions, implementations or
  extern declarations. Existing CUDA/FFI/kernel/transport boundaries are not
  permission to expand them; preserve their safety contracts and SAFETY comments.
- **Model-generic code.** Derive execution from checkpoint configuration,
  tensor shapes/dtypes and safetensors metadata. No model-name dispatch for
  execution tuning, checkpoint-specific constants, prompt-specific shortcuts or
  model-specific performance tuning. Named download catalogs are metadata, not
  an exception for engine behavior.
- **Preserve source formatting.** The Rust uses deliberate dense formatting.
  Never run `cargo fmt` or clippy's automatic fix mode over this tree. Match the
  surrounding style by hand; no drive-by formatting. CI does not gate on fmt or
  clippy. Keep new Rust files at or below 1,500 lines.
- **Protect private material.** Never commit secrets, credentials, machine
  addresses, local config, model weights, generated PTX or build output. Keep
  private scratch data in ignored `.local/` or `STATUS.md`. Do not touch weights,
  SSH settings or another user's processes. Never kill by process name. Do not
  change git configuration, merge or push as part of a coding-agent task unless
  the task explicitly delegates that operation.
- **Stay in scope.** Prefer small commits and one change per PR. Do not operate
  on a Spark without explicit authorization. Documentation and host-only changes
  need no Spark; engine behavior changes need the device gate below.

## Exactness law and merge gate

Exactness means **identical token IDs every run on every Spark with the same
setup**, not merely similar text. Greedy decoding is the default when sampling
is unspecified. Engine arithmetic, scheduling, caching and transport changes
require at least three repeats and comparison across at least two Sparks before
merge. Require alone == batched == cached == chunked and drafting
(MTP) ON == OFF for the same request and settings.

Do not introduce floating-point atomics, arrival-order reductions, or reduction
splits that change with batch size, prompt length or occupancy. Reduction order
and tiling must follow fixed tensor dimensions. Changes that preserve arithmetic
order must match main's token IDs; an intentional order change needs an explicit
switch, a full exactness gate, reviewed golden updates and quality evidence
against main before becoming the default. Performance claims need reproducible
receipts with revision, hardware, checkpoint identity, settings and raw results.

Host checks below do not establish GPU exactness or speed. Device execution and
the running-server verification described in [CLI validation](cli/docs/test.md)
and [CLI verification](cli/README.md#verify-exactness) require an authorized Spark.
The historical determinism runner uses flash-bench; accepting world 1 does not
prove FP4-Local coverage. Do not substitute its FP8 fixtures for an FP4-Local
gate or claim an unrun gate passed.

## Repository map

There is **no root Cargo.toml**. `engine/Cargo.toml` has 13 members; the engine
tree has 15 packages after counting the separate benchmark and TP-kernel
workspaces. See the complete [package map](engine/crates/README.md).

| Path | Responsibility |
| --- | --- |
| `cli/` | Standalone `vade` lifecycle, launcher, verification and benchmark CLI; repository ratchet tests. |
| `engine/` | Checkpoint inspection, tokenizer, GPU kernels, execution, serving and multi-Spark transport workspace. |
| `engine/tools/bench/` | Separate benchmark workspace. |
| `engine/crates/tp/kernels/` | Separate nightly NVPTX workspace. |
| `harness/` | Standalone native agent harness and gateway. |
| `docs/` | Public architecture, decisions, model guide and release documentation. |
| `tools/` | Documentation link checker and CI support. |

## Local validation

Run this block from **`engine/`**, on macOS or Linux. Put rustup's bin directory
(normally `~/.cargo/bin`) on PATH. Use the pinned host toolchain 1.99.0 and kernel
toolchain nightly-2026-10-01 with rust-src, llvm-tools, llvm-bitcode-linker and
the nvptx64-nvidia-cuda target. CI provisions these with the following commands
(initial setup requires downloads; do not run during offline work):

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

Set RUSTFLAGS to `-D warnings`, as CI does. With
the toolchains and compiler dependencies already cached, set CARGO_NET_OFFLINE
to `true` and RUSTUP_AUTO_INSTALL to `0` for a fully offline check.

Build kernels **first on every OS**. Existing stale PTX fails host checks on a
Mac too. Kernel compilation needs no GPU; a first uncached toolchain/build-std
setup needs downloads and must be provisioned separately.

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
python3 ../tools/check-doc-links.py
```

### Engine and full CI checks

Continue from `engine/`. The TP-kernel build uses its own RUSTFLAGS:

```sh
RUSTFLAGS='-C target-cpu=sm_121 -C overflow-checks=off' \
  cargo +nightly-2026-10-01 build --offline -Z build-std=core --release \
  --target nvptx64-nvidia-cuda --manifest-path crates/tp/kernels/Cargo.toml
```

On **Linux hosts without a GPU only**, CI installs an abort-on-device-call CUDA
stub before host tests. From `engine/`:

```sh
python3 ../tools/ci-no-gpu.py
export LIBRARY_PATH="$PWD/target/ci-no-gpu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$PWD/target/ci-no-gpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

On macOS or the prepared Linux host, from `engine/`:

```sh
cargo +1.99.0 check --offline --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

Always include `--workspace --exclude spark-rust-kernels` for engine host
checks/tests: default-members omit packages, and the NVPTX crate builds separately.
GPU-only tests may be ignored on a host without CUDA; that is not a device pass.
For Linux-only engine code on a Mac with the aarch64-unknown-linux-gnu target
installed, also run from `engine/`:

```sh
cargo +1.99.0 check --offline --all-targets --workspace --exclude spark-rust-kernels --target aarch64-unknown-linux-gnu
```

For the remaining CI tests, run the first command separately from `cli/`; run
the second separately from each of `harness/` and `engine/tools/bench/`:

```sh
cargo +1.99.0 test --offline --all-targets -- --nocapture
cargo +1.99.0 test --offline --all-targets
```

From each of `cli/` and `harness/`, CI also checks the Linux target:

```sh
cargo +1.99.0 check --offline --all-targets --target aarch64-unknown-linux-gnu
```

The authoritative workflow is [.github/workflows/ci.yml](.github/workflows/ci.yml).
See [engine validation](engine/docs/validation.md) for host/device boundaries.

## Ratchet

[cli/tests/ratchet.rs](cli/tests/ratchet.rs) scans tracked and unignored source.
[cli/ratchet-baseline.txt](cli/ratchet-baseline.txt) is the authority for ceilings:
currently `third_party_dependencies=0`, `large_rs_files=3` (existing files over
1,500 lines), and `box_count_branches=97` (integer comparisons/match arms on
world/rank/box counts). Use topology data instead of adding hardcoded count
branches. Never raise a ceiling to make a change pass; lower the baseline in
the same commit when a count falls.

The unsafe ratchet is a no-growth rule too: no new unsafe sites in any crate.
The current test has only the three counters above and does **not** count unsafe
sites; review the diff explicitly. If the baseline gains per-crate `unsafe_sites`
counters, they are ceilings, not an allowance to add unsafe or move it between
crates. Do not claim automated enforcement that is absent from the test.

For the human workflow, read [CONTRIBUTING.md](CONTRIBUTING.md). For orientation,
see [architecture](docs/architecture.md), [decisions](docs/decisions/README.md),
[adding a model](docs/adding-a-model.md) and the [glossary](docs/GLOSSARY.md).
