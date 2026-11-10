# engine/crates/engine/src/bin/flash-bench

**Rust modules for `flash-bench`, part of the `spark-engine` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`gate.rs`](gate.rs) | The determinism matrix uses the engine's real prefill, batch and MTP paths. |
| [`server.rs`](server.rs) | Token and accounting determinism over the existing streaming HTTP protocol. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `flash-bench`.

**How it fits:** see the [`spark-engine` README](../../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../../docs/README.md)
