# engine/crates/model/tests

**Rust modules for `tests`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`checkpoint_io.rs`](checkpoint_io.rs) | Integration tests for this crate. |
| [`flash_mixed.rs`](flash_mixed.rs) | Exhaustive MXFP8 oracle, mixed config and optional full checkpoint metadata reconciliation. |
| [`flash_refcpu.rs`](flash_refcpu.rs) | Golden tests for the FP32 CPU qwen4_exp reference. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `tests`.

**How it fits:** see the [`engine-model` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
