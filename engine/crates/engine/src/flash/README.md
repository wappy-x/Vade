# engine/crates/engine/src/flash

**Rust modules for `flash`, part of the `spark-engine` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`blocks.rs`](blocks.rs) | Four-stream residual connections and lexical projections. |
| [`candidate.rs`](candidate.rs) | Host reduction for vocabulary candidates and the sticky routing verdict. |
| [`gdn.rs`](gdn.rs) | Rank-local GatedDeltaNet: official BF16 projections, FP32 chunked prefill and cached recurrence. |
| [`head.rs`](head.rs) | Untied, vocabulary-sharded BF16 output projection with FP32 activations. |
| [`mod.rs`](mod.rs) | Flash Next text engine. |
| [`moe.rs`](moe.rs) | Tensor-parallel routed/shared MLP and its four-stream HC injection. |
| [`nvfp4.rs`](nvfp4.rs) | Borrowed device descriptors for the exact packed checkpoint layout. |
| [`qsa.rs`](qsa.rs) | Head-sharded indexed attention. |
| [`rows.rs`](rows.rs) | Rank-owned vocabulary and n-gram rows. |
| [`session.rs`](session.rs) | Batch-one text sessions with resident or layer-streamed weights and caches. |
| [`verification.rs`](verification.rs) | Verification metadata transitions, independent of GPU ownership. |
| [`weights.rs`](weights.rs) | Selected official checkpoint bytes in one allocation, with global coordinates. |
| [`session/`](session/README.md) | Folder: see its README. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `flash`.

**How it fits:** see the [`spark-engine` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)

Flash-Next row execution reads its controls once through `next_options::Options`.
Pooled decode embeds and reads PLE on device, returns compact greedy winners,
and queues output copies with the graph. The replaced host-fed decode path and
its two controls are removed after byte and timing qualification. Shared decode
scratch and graphs are reserved once per model, separately from each chat's
full-context KV, recurrent state and prompt scratch budget.
Prompt execution caches immutable row descriptors/files and uploads residual
and lexical arrays once per chunk. The replaced per-token descriptor/upload
path and its PREFILL_GAPS switch are removed after byte and timing qualification.
The native exact path remains the oracle.
