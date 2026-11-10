# engine/crates/model/tests

**Rust modules for `tests`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`checkpoint_io.rs`](checkpoint_io.rs) | Integration tests for this crate. |
| [`dsa_plan.rs`](dsa_plan.rs) | Synthetic headers connect DSA plan data to the host reference. |
| [`decoder_reference.rs`](decoder_reference.rs) | Causal stack composition, exact scalar state reuse, and rollback. |
| [`flash_mixed.rs`](flash_mixed.rs) | Exhaustive MXFP8 oracle, mixed config and optional full checkpoint metadata reconciliation. |
| [`flash_refcpu.rs`](flash_refcpu.rs) | Golden tests for the FP32 CPU qwen4_exp reference. |
| [`glm_loader.rs`](glm_loader.rs) | Deferred GLM metadata and tensor role checks. |
| [`glm_plan.rs`](glm_plan.rs) | Deferred config, storage and operation gap checks. |
| [`host_ops_legacy_parity.rs`](host_ops_legacy_parity.rs) | Config reference parity with existing host operations. |
| [`layer_binding.rs`](layer_binding.rs) | Deferred layer weight bindings and scalar replay checks. |
| [`layout_fixture.rs`](layout_fixture.rs) | Header fixture reports, metadata bounds and path checks. |
| [`reference_config_limits.rs`](reference_config_limits.rs) | Byte-boundary checks for the DSA, KDA, MLA and mHC config readers. |
| [`service_admission.rs`](service_admission.rs) | Deferred metadata and primitive targets cannot admit a resident service. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `tests`.

**How it fits:** see the [`engine-model` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
