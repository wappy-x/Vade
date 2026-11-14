# engine/crates/gpu/src

**The Rust source of the `spark-gpu-probe` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`backend.rs`](backend.rs) | Scoped GB10 backend. |
| [`batch.rs`](batch.rs) | Batched forward launches; weight loading remains in backend.rs. |
| [`driver.rs`](driver.rs) | Minimal, manually declared NVIDIA CUDA Driver API ABI (not CUDA Runtime API). |
| [`fp4.rs`](fp4.rs) | Explicit W4A16 dispatch; the default FP8 path never selects these kernels. |
| [`fp4_abi.rs`](fp4_abi.rs) | Opt-in NVFP4 weights with BF16 operands. |
| [`fp8.rs`](fp8.rs) | Resident E4M3 projections use checkpoint-declared dynamic W8A8 math. |
| [`fp8_abi.rs`](fp8_abi.rs) | No summary comment yet; read the file. |
| [`gdn_params.rs`](gdn_params.rs) | Shared host/NVPTX row descriptor for canonical cached recurrence. |
| [`lib.rs`](lib.rs) | GB10 CUDA driver ownership, packed-weight loading and kernel dispatch. |
| [`load.rs`](load.rs) | Batched shard loading. |
| [`mhc_params.rs`](mhc_params.rs) | Config-derived FP32 Sinkhorn stream ABI and shape checks. |
| [`mhc_boundary.rs`](mhc_boundary.rs) | Explicit BF16 mHC boundaries in FP32 lanes and checked physical spans. |
| [`backend/mhc.rs`](backend/mhc.rs) | Owned mHC parameters, typed BF16 lanes, and synchronized device launches. |
| [`moe.rs`](moe.rs) | Device-resident expert selection over the model's existing weight arena. |
| [`multi.rs`](multi.rs) | Arena row views and queued reductions for independent-session batches. |
| [`projection_params.rs`](projection_params.rs) | Shared host/NVPTX ABI for independent BF16 decode projections. |
| [`ordered_projection.rs`](ordered_projection.rs) | Plain FP32 shape, layout, bounds, and scalar-order host reference. |
| [`ordered_projection_kernel.rs`](ordered_projection_kernel.rs) | Native plain FP32 projection with separate products and ascending adds. |
| [`router_abi.rs`](router_abi.rs) | Checked corrected-sigmoid routing and ungated expert-finish spans. |
| [`segments.rs`](segments.rs) | Fixed virtual KV addresses, append-only physical pages and shared prefixes. |
| [`teardown.rs`](teardown.rs) | Serving processes bound fatal cleanup even when the driver stops responding. |
| [`tp_payload.rs`](tp_payload.rs) | Fixed word offsets and extent checks for a two-vector mailbox message. |
| [`tp_payload_tests.rs`](tp_payload_tests.rs) | Compare packed and separate exchanges with fixed raw words. |
| [`bin/`](bin/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

## Text edges

`text_edges.rs` checks replicated embedding and stored MXFP8 head spans.
`text_reference.rs` checks stored BF16 rows and candidate head arithmetic on the host.
`backend/text_edges.rs` owns their device views and the stream mean launch.
See [the request contract](../../serve/TEXT-PROGRAM.md) for the proof limits.

**How it fits:** see the [`spark-gpu-probe` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
