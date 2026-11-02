# engine/crates/gpu/kernels/src

**The Rust source of the `spark-rust-kernels` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`attention.rs`](attention.rs) | Fused causal attention: BF16 Q/K/V and P, native FP32 tensor-core accumulation. |
| [`batch.rs`](batch.rs) | Causal batch kernels with FP32, BF16, or explicitly selected scaled I16 KV. |
| [`bf16.rs`](bf16.rs) | Software round-to-nearest-even conversion shared by host and GPU code. |
| [`canonical.rs`](canonical.rs) | Decode projections use four fixed K ranges and one balanced FP32 merge. |
| [`decode_attention.rs`](decode_attention.rs) | Grouped decode/verify attention: one K/V read, online softmax, bounded summaries. |
| [`flash_bf16.rs`](flash_bf16.rs) | Flash projections, using the dense engine's canonical BF16 MMA. |
| [`flash_gdn.rs`](flash_gdn.rs) | Head-sharded Flash GatedDeltaNet. |
| [`flash_gdn_chunk.rs`](flash_gdn_chunk.rs) | 64-token gated-delta chunks. |
| [`flash_moe.rs`](flash_moe.rs) | Flash routing and ordered reduction of rank-sharded expert partials. |
| [`flash_ops.rs`](flash_ops.rs) | Flash-Next embedding, four-stream hyper-connections and lexical residuals. |
| [`flash_qsa.rs`](flash_qsa.rs) | Head-sharded indexed attention. |
| [`fp4.rs`](fp4.rs) | NVFP4 primitives for GB10, opt-in through `cargo kernels --fp4`. |
| [`fp4_experts.rs`](fp4_experts.rs) | Resident NVFP4, register dequantization and canonical BF16 tensor cores. |
| [`fp8.rs`](fp8.rs) | Native E4M3 tensor-core projections for the checkpoint's 128x128 blocks. |
| [`fp8_accurate_decode.rs`](fp8_accurate_decode.rs) | BF16 gate/up decode with rank-sharded resident checkpoint FP8 weights. |
| [`fp8_accurate_gate.rs`](fp8_accurate_gate.rs) | Rank-sharded W8A8 gate/up; explicit legacy BF16 operands remain available. |
| [`fp8_bf16.rs`](fp8_bf16.rs) | Checkpoint E4M3 conversion and grouped BF16 tensor-core operands. |
| [`fp8_decode.rs`](fp8_decode.rs) | Decode directly from checkpoint bytes; split K only inside each CTA. |
| [`fp8_routing.rs`](fp8_routing.rs) | Device routing for the Flash-Next grouped expert FFN. |
| [`gemm.rs`](gemm.rs) | Tensor-core projections through LLVM intrinsics and small packing PTX blocks. |
| [`gemm_gate_packed.rs`](gemm_gate_packed.rs) | Larger gate/up tile retaining the native packed shared-memory layout. |
| [`gemm_prefill_wide.rs`](gemm_prefill_wide.rs) | Full-row prefill tiles: 256x128x64, two L2-only asynchronous stages. |
| [`gemm_shared.rs`](gemm_shared.rs) | Packed GEMM: on-chip asynchronous staging; 8-row packed or blocked 64-row weights. |
| [`layer.rs`](layer.rs) | Layer kernels compiled with the projection kernels for the same GPU target. |
| [`lib.rs`](lib.rs) | Genuine Rust-authored kernels compiled by the official rustc NVPTX backend. |
| [`math.rs`](math.rs) | BF16 bit conversion shared by Rust GPU kernels. |
| [`matvec.rs`](matvec.rs) | Single-row projections use the same native BF16 MMA order as verification. |
| [`moe.rs`](moe.rs) | Sparse expert decode: device routing and only selected BF16 projections. |
| [`moe_decode.rs`](moe_decode.rs) | Selected-expert decode over the resident MMA-packed BF16 weight arena. |
| [`multi.rs`](multi.rs) | Ragged Qwen3 GQA rows with exactly the serial reduction order. |
| [`ptx.rs`](ptx.rs) | Small instructions missing from Rust's NVPTX address-space interface. |
| [`verify_mma.rs`](verify_mma.rs) | Canonical K ranges run in parallel warps; their balanced merge stays on chip. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`spark-rust-kernels` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
