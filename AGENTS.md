# Working on Vade

> [!NOTE]
> This file is for agents that change Vade's code. To install, start or use
> Vade for a person, or to connect Claude Code or Codex to it, read the
> [agent quickstart](docs/agents/quickstart.md) instead. On a Spark, the path
> is `curl -fsSL https://vade.lol/install.sh | sh`, `vade pull`, then `vade up`.

Vade is an inference engine for the NVIDIA DGX Spark / GB10. It is written in
Rust and uses only the standard library. This file gives the rules for a
change to Vade's code.

Version 0.1.0 targets `nvidia/Qwen3.8-Flash-Next-FP4-Local` and handles text
only. Main also serves `Qwen/Qwen3.8-Flash-Next-FP8` on 2 or 4 Sparks. Six
Sparks start, but exactness does not cover them yet. Vade runs on one Spark by
default, and the topology supports 1..N Sparks.

Drafting is on by default. An extra model head proposes tokens, and the model
verifies them. With FP4-Local, a request that does not say how to sample is
decoded greedily. With FP8, such a request is sampled with the checkpoint's
values ([FP8 defaults](docs/install-fp8.md#defaults-and-drafting)).

[`VADE_REPO_URL`](cli/src/repository.rs) defines the source repository URL.

## House rules

- **Language.** Write code in Rust with the standard library only. Use shell
  only where Rust cannot do the job. Do not use or require Python.
- **Writing.** Use ASD-STE100 Simplified Technical English and American
  spelling. Use short sentences, active voice and plain words. Give each
  sentence one main idea. Do not use em dashes or lab slang. Apply this rule
  to docs, help, errors, reports and messages. Every numerical claim must
  name its card, log or command output. Give raw values with any percentage.
- **Zero third-party dependencies.** Use the Rust standard library and the
  first-party crates already in the tree. Registry, git, off-tree path, build
  and dev dependencies all count, and so do lockfile sources. Don't add a
  dependency to solve a small task.
- **No new unsafe, with one exception.** Don't add unsafe blocks, functions,
  implementations or extern declarations. The existing CUDA, FFI, kernel and
  transport boundaries are not permission to grow them. Keep their safety
  contracts and SAFETY comments intact. The exception (2026-11-01): a kernel
  may add one new unsafe block, in the GPU crate (`engine/crates/gpu`) only.
  Write its safety contract in a comment directly above the block: what must
  be true for the call to be sound and why it is true there. A reviewer
  checks the contract before the merge. One block per kernel, never in any
  other crate.
- **Model-generic code.** Derive execution from the checkpoint configuration,
  tensor shapes and dtypes, and safetensors metadata. Don't dispatch on the
  model name for execution tuning, and don't add checkpoint-specific constants,
  prompt-specific shortcuts or model-specific performance tuning. Named
  download catalogs are metadata; they are not an exception for engine
  behavior.
- **Preserve source formatting.** The Rust is formatted densely on purpose.
  Never run `cargo fmt` or clippy's automatic fix mode over this tree. Match
  the surrounding style by hand, and don't reformat code you aren't changing.
  CI does not gate on fmt or clippy. Keep new Rust files at or below 1,500
  lines.
- **Protect private material.** Never commit secrets, credentials, machine
  addresses, local config, model weights, generated PTX or build output. Keep
  private scratch data in the ignored `.local/` or `STATUS.md`. Don't touch
  weights, SSH settings or another user's processes, and never kill a process
  by name. As a coding agent, don't change git configuration, merge or push
  unless the task explicitly hands you that job.
- **Stay in scope.** Prefer small commits, one change per PR. Don't operate on
  a Spark without explicit permission. Documentation and host-only changes
  (code that runs on the CPU, not the GPU) need no Spark. Changes to engine
  behavior need the device gate below: a set of checks on real Sparks that
  must pass before merge.

## Legacy code

The owner marked the parts below as legacy on 2026-11-10. Nobody works on
them now. They can come back later, or move to a legacy folder, or leave the
tree.

- **Model families:** dense Qwen3 and its mixture-of-experts form, Phi and
  Llama; the Qwen3.5, Qwen3.6 and Qwen3.8-27B hybrid; Gemma 4.
- **Server programs:** `spark-serve`, `tp-serve` and `next-http`.
- **CLI:** `vade run qwen3-8b`.

Current work is Qwen3.8 Flash-Next and the GLM text engine. The
[module map](engine/docs/module-map.md) follows one request through each.

Rules for legacy code:

- Do no new work on it: no features, no tuning and no new tests.
- No device gate runs on it. It can break on a Spark, and a release does not
  claim that it works.
- Keep it compiling, and keep its host tests passing, while it is in the
  tree. The host checks do not know that it is legacy.
- Do not build new code on it.
- If a change to shared code makes a legacy path fail to compile, make the
  smallest edit that compiles. If that edit is not small, stop and ask.

## Exactness law and merge gate

Exactness means **identical token IDs every run on every Spark with the same
setup**. The same request and settings must also give the same IDs across
supported Spark counts. Similar text is not proof. With FP4-Local, greedy
decoding is the default when a request leaves sampling unspecified. With FP8,
a request is greedy only when it sets `temperature` 0 and leaves the
penalties neutral. Before you merge a change to engine arithmetic, scheduling,
caching or transport, run it at least three times and compare the results
across at least two Sparks. One exception: a gated change rebased onto a
moved main can merge on its gate with an identical patch, disjoint files and
a passing pre-merge screen on the Sparks (see `CONTRIBUTING.md`, Exactness
and evidence). For the same request and settings, the answer must not depend
on how it ran. Alone, batched, cached and chunked runs must all match.
Drafting (multi-token prediction, MTP) ON and OFF must match too.

Don't introduce floating-point atomics, reductions that depend on arrival
order, or reduction splits that change with batch size, prompt length or
occupancy. Reduction order and tiling must follow fixed tensor dimensions. A
change that keeps the arithmetic order must match main's token IDs.

Use one execution path for each topology. The tensor split is the only
two-Spark path. Do not add switches or flags that select a different
arithmetic order. Do not keep an old path behind a switch. A proposed
arithmetic change needs the full exactness gate, a measurement card and
quality evidence against main. Review any updates to the goldens (the saved
reference answers). A change that alters token IDs needs the owner's
approval before merge.

Performance claims need reproducible receipts (the saved records of a run)
with revision, hardware, checkpoint identity, settings and raw results.

The host checks below don't establish GPU exactness or speed. Device runs need
an authorized Spark, and so does the running-server verification described in
[CLI validation](cli/docs/test.md) and
[CLI verification](cli/README.md#verify-exactness).

The older determinism runner, built on flash-bench, accepts world 1 (a
one-Spark run), but that doesn't prove it covers FP4-Local. Don't substitute
its FP8 fixtures for an FP4-Local gate, and never claim that a gate you didn't
run has passed.

## Repository map

There is **no root Cargo.toml**. `engine/Cargo.toml` has 13 members. With the
separate benchmark workspace and the separate TP (tensor parallelism) kernel
workspace, the engine tree has 15 packages. The
[package map](engine/crates/README.md) lists them all.

| Path | What lives there |
| --- | --- |
| `cli/` | The standalone `vade` CLI: lifecycle, launcher, verification and benchmarks. The repository ratchet tests live here too. |
| `engine/` | The engine workspace: checkpoint inspection, the tokenizer, GPU kernels, execution, serving and transport between Sparks. |
| `engine/tools/bench/` | Separate benchmark workspace. |
| `engine/crates/tp/kernels/` | A separate workspace for the GPU kernels, built with the nightly compiler for NVPTX (NVIDIA's GPU target). |
| `harness/` | Standalone native agent harness and gateway. |
| `docs/` | Public architecture, decisions, model guide and release documentation. |
| `tools/` | Documentation link checker and CI support. |

## Local validation

Local validation runs on macOS or Linux. Put rustup's bin directory
(normally `~/.cargo/bin`) on your PATH. You need the pinned host toolchain
1.99.0 and the kernel toolchain nightly-2026-10-01 with rust-src, llvm-tools,
llvm-bitcode-linker and the nvptx64-nvidia-cuda target. CI installs them with
the commands below. This first setup downloads files, so run it while you're
online:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

Set RUSTFLAGS to `-D warnings`, as CI does. Once the toolchains and compiler
dependencies are cached, you can set CARGO_NET_OFFLINE to `true` and
RUSTUP_AUTO_INSTALL to `0` for a fully offline check.

Build kernels **first on every OS**: stale PTX (compiled GPU kernel code) left
over from an earlier build fails host checks, even on a Mac. Compiling kernels
needs no GPU, but the first, uncached toolchain and build-std setup (building
Rust's core library for the GPU target) does need downloads, so set that up
separately while you're online. Then run this block from **`engine/`**:

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
cargo +1.99.0 run --offline --locked --manifest-path ../tools/Cargo.toml -p vade-tools -- check-doc-links
```

### Engine and full CI checks

Stay in `engine/`. The TP-kernel build uses its own RUSTFLAGS:

```sh
RUSTFLAGS='-C target-cpu=sm_121 -C overflow-checks=off' \
  cargo +nightly-2026-10-01 build --offline -Z build-std=core --release \
  --target nvptx64-nvidia-cuda --manifest-path crates/tp/kernels/Cargo.toml
```

On **Linux hosts without a GPU only**, CI installs a CUDA stub that aborts on
any device call before it runs the host tests. From `engine/`:

```sh
cargo +1.99.0 run --offline --locked --manifest-path ../tools/Cargo.toml -p vade-tools -- ci-no-gpu
export LIBRARY_PATH="$PWD/target/ci-no-gpu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$PWD/target/ci-no-gpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

On macOS, or on a Linux host prepared that way, from `engine/`:

```sh
cargo +1.99.0 check --offline --locked --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --locked --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

Always pass `--workspace --exclude spark-rust-kernels` for engine host checks
and tests: default-members leaves packages out, and the NVPTX crate builds
separately. GPU-only tests may be ignored on a host without CUDA, and that
doesn't count as a device pass. For Linux-only engine code on a Mac with the
aarch64-unknown-linux-gnu target installed, also run this from `engine/`:

```sh
cargo +1.99.0 check --offline --locked --all-targets --workspace --exclude spark-rust-kernels --target aarch64-unknown-linux-gnu
```

For the rest of the CI tests, run the first command below from `cli/`. Run the
second from `harness/`, then again from `engine/tools/bench/`:

```sh
cargo +1.99.0 test --offline --locked --all-targets -- --nocapture
cargo +1.99.0 test --offline --locked --all-targets
```

CI also checks the Linux target from both `cli/` and `harness/`:

```sh
cargo +1.99.0 check --offline --locked --all-targets --target aarch64-unknown-linux-gnu
```

The workflow [.github/workflows/ci.yml](.github/workflows/ci.yml) is the full
check list. The commands above omit doctests, Rustdoc and the tools test
suite; [docs/ci.md](docs/ci.md) gives those commands. Use `--locked` in host
Cargo commands to preserve the committed lockfiles. See
[engine validation](engine/docs/validation.md) for where host
checks end and device checks begin.

## Ratchet

The ratchet is a set of counts that may go down but never up.
[cli/tests/ratchet.rs](cli/tests/ratchet.rs) scans tracked and unignored source,
and [cli/ratchet-baseline.txt](cli/ratchet-baseline.txt) holds the ceilings.
Today they are:

- `third_party_dependencies=0`
- `large_rs_files=1`: existing files over 1,500 lines
- `box_count_branches=96`: integer comparisons and match arms on world, rank
  or box counts (code that checks for a fixed number of Sparks)

Use topology data instead of adding hardcoded count branches. Never raise a
ceiling to make a change pass. When a count falls, lower the baseline in the
same commit.

Unsafe follows the same no-growth rule: no new unsafe sites in any crate,
except one block per kernel in the GPU crate with a written safety contract
above it and a reviewer check (see "No new unsafe, with one exception"). The
current test has only the three counters above and does **not** count unsafe
sites, so review the diff for them yourself. If the baseline later gains
per-crate `unsafe_sites` counters, they are ceilings, not an allowance to add
unsafe or move it between crates. Don't claim automated enforcement that the
test doesn't have.

Two more checks are separate commands. They are not part of the ratchet
test, so run both before you ask for a merge. From the repository root:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-unsafe-sites
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-one-path
```

`check-unsafe-sites` compares each unsafe site with
[tools/unsafe-allowlist.txt](tools/unsafe-allowlist.txt). `check-one-path`
compares each environment read and each fixed box-count branch in engine
source with [tools/one-path-allowlist.txt](tools/one-path-allowlist.txt).
The tools test suite runs both. A new row in the unsafe allowlist needs a
ruling, and a new row in the one-path allowlist needs a review. Do not add a
row to make a check pass. [tools/README.md](tools/README.md) gives the rules
for each check.

For the human workflow, read [CONTRIBUTING.md](.github/CONTRIBUTING.md). To find your
way around, see [architecture](docs/architecture.md),
[decisions](docs/decisions/README.md), [adding a model](docs/adding-a-model.md)
and the [glossary](docs/GLOSSARY.md).
