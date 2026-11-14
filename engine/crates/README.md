# Engine packages

There are **16 packages under `engine/`**: the **14 members** explicitly listed
in [engine/Cargo.toml](../Cargo.toml), plus the independent `tools/bench` and
`crates/tp/kernels` workspaces. The repository also has separate CLI and harness
packages outside `engine/`.

| Package and source | Workspace | Role | Validation below |
| --- | --- | --- | --- |
| [`engine-base`](base/README.md) | engine | Bounded JSON metadata and shared admission diagnostics. | Host |
| [`engine-fabric`](fabric/README.md) | engine | Peer discovery, RDMA interfaces, topology and exchange primitives. | Host |
| [`engine-hash`](hash/README.md) | engine | SHA-256 and checkpoint file verification I/O. | Host |
| [`engine-model`](model/README.md) | engine | Checkpoint metadata, tensor validation, model plans and splits. | Host |
| [`spark-gpu-probe`](gpu/README.md) | engine | CUDA Driver API, GPU memory and kernel launch wrappers. | Host |
| [`spark-rust-kernels`](gpu/kernels/README.md) | engine | Rust GPU kernels compiled to NVPTX. | Shared kernels |
| [`spark-tc-kernels`](gpu/kernels-tc/src/lib.rs) | engine | Tensor-core kernel image for `sm_121a`; `cargo kernels --tc` builds it. | Shared kernels |
| [`spark-build-kernels`](gpu/build-kernels/README.md) | engine | Builds, copies and stamps shared kernel PTX. | Host / shared kernels |
| [`spark-engine`](engine/README.md) | engine | Model execution, sessions and KV caches. | Host |
| [`engine-tokenizer`](tokenizer/README.md) | engine | Tokenization, decoding and chat templates. | Host |
| [`spark-serve`](serve/README.md) | engine | HTTP serving, scheduling, sampling and prompt caching. | Host |
| [`spark-tp`](tp/README.md) | engine | Tensor parallelism, Flash-Next serving and multi-Spark launch tools. | Host |
| [`spark-seqpar`](seqpar/README.md) | engine | Sequence-parallel prefill, transport and control. | Host |
| [`vade-export`](../tools/export/README.md) | engine | Allowlisted source snapshots and export validation. | Host |
| [`vade-bench`](../tools/bench/README.md) | tools/bench | HTTP benchmark library and the separate `kernel-bench` binary. | Benchmark |
| [`spark-tp-kernels`](tp/kernels/README.md) | crates/tp/kernels | Optional TP projection kernels using the shared canonical multiply order. | TP kernels |

Run shared kernels first, from `engine/`, on macOS as well as Linux:

```sh
cargo +1.99.0 kernels
```

Host validation, from `engine/`:

```sh
cargo +1.99.0 check --offline --locked --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --locked --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

The workspace has only eight default members; a plain Cargo test omits six
members. Always use the workspace selection above. CI sets `RUSTFLAGS` to
`-D warnings`; Linux hosts without CUDA also need the fake-library setup in
[Validation](../docs/validation.md#on-any-computer-host-checks).

Benchmark validation, from `engine/tools/bench/`:

```sh
cargo +1.99.0 check --offline --locked --all-targets
cargo +1.99.0 test --offline --locked --all-targets
```

TP kernels, from `engine/` (compilation needs no GPU):

```sh
RUSTFLAGS='-C target-cpu=sm_121 -C overflow-checks=off' \
  cargo +nightly-2026-10-01 build --offline -Z build-std=core --release \
  --target nvptx64-nvidia-cuda --manifest-path crates/tp/kernels/Cargo.toml
```

These are the commands used by [CI](../../.github/workflows/ci.yml). Host checks
do not establish GPU exactness or performance; execution checks need a Spark.
See [AGENTS.md](../../AGENTS.md) for the house rules and pinned toolchains.

**Back:** [Engine](../README.md) · [Architecture](../docs/architecture.md)
