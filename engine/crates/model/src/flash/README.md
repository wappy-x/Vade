# engine/crates/model/src/flash

**Rust modules for `flash`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`manifest.rs`](manifest.rs) | Payload identity for NVFP4 loading. |
| [`mixed.rs`](mixed.rs) | Config-driven ModelOpt tensor storage, world-1 memory inventory and typed NVFP4/PLE hooks. |
| [`mxfp8.rs`](mxfp8.rs) | CPU E4M3FN/UE8M0 dequantization oracle, one scale per 32 values. |
| [`mod.rs`](mod.rs) | Qwen3.8-Flash-Next's observed checkpoint contract. |
| [`nvfp4.rs`](nvfp4.rs) | Official ModelOpt NVFP4: adjacent E2M1 values, E4M3 scales per 16 values and one FP32 multiplier per projection. |
| [`plan.rs`](plan.rs) | Text-only ownership in original tensor coordinates; payloads pack these ranges. |
| [`refcpu.rs`](refcpu.rs) | FP32 CPU reference for the qwen4_exp_text decoder (batch 1, token order). |
| [`shard.rs`](shard.rs) | World-1/2/4 mixed rank ownership and bounded map-checked payload reads. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `flash`.

Mixed checkpoints reuse the config-derived [`ModelPlan`](../plan/mod.rs): its
logical names, shapes and layer roles are reconciled with ModelOpt's `ignore`
patterns and `quantized_layers`. Vision tensors remain `Place::Unused`.
`Checkpoint::inspect` accepts the mixed headers; `mixed_plan(1)` exposes storage
and byte counts. `shard::ShardCheckpoint` adds whole-expert ownership plus MXFP8
head/channel selections at worlds 1/2/4, checked against the tensor map, index
and actual shard headers. Its bounded stream reads only resident local selections;
embedding and PLE rows stay on disk. It checks metadata, not payload hashes.
The existing GPU loader's mixed world-1 guard stays in place.

`mixed::Nvfp4` is the kernel handoff: packed values, logical dimensions, a global
FP32 scale, optional activation scale, and either direct E4M3 block scales or
opaque CSF fixed/exception tensors. CSF payload decoding belongs to the kernel
implementation. `ple_row` describes bounded packed and scale reads within one
of the config's shards; it does not perform disk I/O or retain expanded rows.
The world-1 raw weight loader preserves these payloads and exposes
`Weights::bind_nvfp4` for device addresses. The existing GPU executor refuses
mixed checkpoints until MXFP8 projections and CSF decoding are connected.

Run the [storage example](../../examples/README.md) with metadata only to print
all tensor classes and the memory table. The table uses current FP32 KV/index
storage, one session, optional MTP and a stated 4 GiB workspace reserve.
The reserve is a planning assumption; concurrent sessions, prefix snapshots and
MTP verification snapshots require additional memory.

**How it fits:** see the [`engine-model` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
