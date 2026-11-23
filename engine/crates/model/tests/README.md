# engine/crates/model/tests

**Rust modules for `tests`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`capped_head_trace.rs`](capped_head_trace.rs) | Embedding norm, last norm, head and logit cap against a saved reference trace and the real stored weights. Ignored by default. |
| [`capped_moe_trace.rs`](capped_moe_trace.rs) | Routed expert block against a saved reference trace and the real stored weights. Ignored by default. |
| [`checkpoint_io.rs`](checkpoint_io.rs) | Integration tests for this crate. |
| [`dsa_plan.rs`](dsa_plan.rs) | Synthetic headers connect DSA plan data to the host reference. |
| [`decoder_reference.rs`](decoder_reference.rs) | Causal stack composition, exact scalar state reuse, and rollback. |
| [`expert_widen_dump.rs`](expert_widen_dump.rs) | Widened expert weights against the values that the reference model holds after its load. Ignored by default. |
| [`flash_mixed.rs`](flash_mixed.rs) | Exhaustive MXFP8 oracle, mixed config and optional full checkpoint metadata reconciliation. |
| [`flash_refcpu.rs`](flash_refcpu.rs) | Golden tests for the FP32 CPU qwen4_exp reference. |
| [`glm_loader.rs`](glm_loader.rs) | Deferred GLM metadata and tensor role checks. |
| [`glm_plan.rs`](glm_plan.rs) | Deferred config, storage and operation gap checks. |
| [`inkling_index.rs`](inkling_index.rs) | Tensor roles, shape and type checks and refusal texts for the looped text family. |
| [`host_ops_legacy_parity.rs`](host_ops_legacy_parity.rs) | Config reference parity with existing host operations. |
| [`layer_binding.rs`](layer_binding.rs) | Deferred layer weight bindings and scalar replay checks. |
| [`layout_fixture.rs`](layout_fixture.rs) | Header fixture reports, metadata bounds and path checks. |
| [`loop_load.rs`](loop_load.rs) | Load plan, state plan and byte ledger of the looped text family, on the made-up fixture of the tensor index. |
| [`loop_packed.rs`](loop_packed.rs) | Ignored check on the real weights: an expert product on the stored codes gives the words of the product on the widened matrix. |
| [`loop_host_real.rs`](loop_host_real.rs) | Ignored checks of the whole-model host form on the real weights: module dumps, layer trace, first token and answers of the reference code. |
| [`loop_plan.rs`](loop_plan.rs) | Looped text config, storage sidecar and refusal checks on a small made-up fixture. |
| [`reference_config_limits.rs`](reference_config_limits.rs) | Byte-boundary checks for the DSA, KDA, MLA and mHC config readers. |
| [`service_admission.rs`](service_admission.rs) | Deferred metadata and primitive targets cannot admit a resident service. |
| [`sink_attention_trace.rs`](sink_attention_trace.rs) | Sink attention and short convolutions against a saved reference trace and the real stored weights. |
| [`visit_loop_trace.rs`](visit_loop_trace.rs) | Ignored check of the block mix and the visit loop against a layer trace of the reference code. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `tests`.

**How it fits:** see the [`engine-model` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
