# Deferred GLM config plan fixture

This is synthetic test data. It contains no checkpoint payload.
The dimensions in `config.json` and `header.json` are small test values.
They are not observed checkpoint dimensions or dtypes.

The fixture uses field and tensor names from the public
[GLM FP4-Local config and index](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/f5a0baba30120da613e27973ee3535a725d4ece3).
It uses the header-only fixture method from the earlier layout audit.
It does not use that audit branch as a build dependency.

The shared expert role follows `Glm5NextTextMoE` in public Transformers
`src/transformers/models/glm5_next/modeling_glm5_next.py`. The local reference
was saved as `tf_modeling_glm5_next.py`. Its SHA256 is
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
This hash identifies the inspected bytes. It does not assert a Git revision.
The source copy and fetch receipt are not included in this repository.
Its shared width is `moe_intermediate_size * n_shared_experts`.
The shared output is added without a separate shared gate.

The FP8 scale layouts follow `KFp8Block128` in public vLLM
`vllm/model_executor/layers/quantization/modelopt.py`, revision
`355553ac22dca4080929ea8f88a446f15df1f3ba`.
The scale fixtures exercise its F32 grid and expanded grid forms.
The MXFP8 example exercises the current storage relation in
[`mixed.rs`](../../../src/flash/mixed.rs).

The appended MTP names remain opaque. The local public reference does not
define their operations. The tests do not assign them `mtp.*` aliases.
The CSF name also remains an explicit storage gap. The fixture does not
define its codec or component layout.

[`glm_plan.rs`](../../glm_plan.rs) builds a virtual safetensors reader from
the header. It checks that header inspection never reads payload bytes.
It checks accepted metadata, shared roles, scale pairing, bounds, and named
execution gaps. It does not establish execution support or GPU exactness.
