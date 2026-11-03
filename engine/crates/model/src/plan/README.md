# engine/crates/model/src/plan

**Rust modules for `plan`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`config.rs`](config.rs) | Config adapters for checkpoint-derived geometry and supported semantics. |
| [`description.rs`](description.rs) | Host descriptions for operations that have no execution binding. |
| [`error.rs`](error.rs) | Typed failures while reading, validating and reconciling a model plan. |
| [`glm.rs`](glm.rs) | Translate the GLM text schema into operation data for host inspection. |
| [`glm_index.rs`](glm_index.rs) | Bind index names to tensor roles and scale components. |
| [`layer.rs`](layer.rs) | Logical decoder order, kernel targets, buffers, and tensor splits. |
| [`layer/`](layer/README.md) | Independent scalar forward and program replay. |
| [`mod.rs`](mod.rs) | A checkpoint-derived, typed model plan, independent of loading and serving. |
| [`profile.rs`](profile.rs) | Count logical compulsory reads from model geometry. |
| [`runtime.rs`](runtime.rs) | Hybrid runtime geometry and tensor roles, including draft layers. |
| [`storage.rs`](storage.rs) | Read mixed storage recipes without device admission. |
| [`tensors.rs`](tensors.rs) | Expected tensor names, storage, roles and header reconciliation. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `plan`.

**How it fits:** see the [`engine-model` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
