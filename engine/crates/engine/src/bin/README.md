# engine/crates/engine/src/bin

**Programs built from the `spark-engine` crate, one per file.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`batch-bench.rs`](batch-bench.rs) | The `batch-bench` program. |
| [`draft-generate.rs`](draft-generate.rs) | Reproducible plain/STRICT/FAST comparison, using the public engine APIs. |
| [`flash-bench.rs`](flash-bench.rs) | Resident Flash fixture runner. |
| [`generate-tokens.rs`](generate-tokens.rs) | Fixed-ID autoregressive runner. |
| [`hf-batch-check.rs`](hf-batch-check.rs) | Compare batched prefill and teacher-forced decode with supplied HF logits. |
| [`load-check.rs`](load-check.rs) | Real-engine startup and same-context unload/reload check. |
| [`flash-bench/`](flash-bench/README.md) | Folder: see its README. |

**Why a folder:** Cargo builds every file in a `bin/` folder as its own program.

`flash-bench` and `load-check` include the existing `seqpar/src/partition.rs`,
`control.rs` and `transport.rs` with `#[path]`. These modules still belong to
`spark-seqpar`; depending on that crate would create a cycle because it
already depends on `spark-engine`. Both binaries use `engine-fabric`
directly for discovery and Linux RDMA verbs.

**How it fits:** see the [`spark-engine` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
