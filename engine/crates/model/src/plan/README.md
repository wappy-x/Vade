# engine/crates/model/src/plan

**Rust modules for `plan`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`config.rs`](config.rs) | Config adapters for checkpoint-derived geometry and supported semantics. |
| [`error.rs`](error.rs) | Typed failures while reading, validating and reconciling a model plan. |
| [`mod.rs`](mod.rs) | A checkpoint-derived, typed model plan, independent of loading and serving. |
| [`tensors.rs`](tensors.rs) | Expected tensor names, storage, roles and header reconciliation. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `plan`.

**How it fits:** see the [`engine-model` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
