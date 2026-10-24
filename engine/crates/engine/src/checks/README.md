# engine/crates/engine/src/checks

**Rust modules for `checks`, part of the `spark-engine` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`cache.rs`](cache.rs) | Actual-checkpoint cache equivalence gate, without a tokenizer or mock backend. |
| [`prefill.rs`](prefill.rs) | Actual-checkpoint equality of step-every-token and head-skipping prompt advance. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `checks`.

**How it fits:** see the [`spark-engine` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
