# engine/crates/engine/src

**The Rust source of the `spark-engine` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`context.rs`](context.rs) | Checked admission for FP32 key/value storage. |
| [`device.rs`](device.rs) | Persistent Qwen activations and selectable FP32/BF16 KV per session. |
| [`draft.rs`](draft.rs) | Interruptible speculative decoding. |
| [`gemma4.rs`](gemma4.rs) | Dense Gemma text cartridge; projections, normalization and attention are shared. |
| [`hybrid.rs`](hybrid.rs) | One-Spark dense hybrid text model. |
| [`hybrid_decode.rs`](hybrid_decode.rs) | Independent chats share one BF16 projection pass while retaining their own delta-net history/matrices and full-attention KV. |
| [`hybrid_moe.rs`](hybrid_moe.rs) | Routed hybrid MLP over views into the single immutable expert arena. |
| [`hybrid_mtp.rs`](hybrid_mtp.rs) | Shared-embedding, one-layer MTP at checkpoint precision. |
| [`hybrid_prefill.rs`](hybrid_prefill.rs) | Shared dense projections for independent prompts. |
| [`layer.rs`](layer.rs) | Qwen layer arithmetic shared by local execution and tensor-parallel ranks. |
| [`lib.rs`](lib.rs) | Owned GB10 engine: BF16 weights, device-resident FP32 activations and KV. |
| [`multi.rs`](multi.rs) | Shared BF16 tensor-core projections for independent decode and prompt rows. |
| [`qwen.rs`](qwen.rs) | Dense and expert Qwen3 execution with immutable BF16 GPU matrices and per-request KV. |
| [`sample.rs`](sample.rs) | Draws are keyed by seed and absolute position; replay consumes no RNG state. |
| [`bin/`](bin/README.md) | Folder: see its README. |
| [`checks/`](checks/README.md) | Folder: see its README. |
| [`flash/`](flash/README.md) | Folder: see its README. |
| [`gemma4/`](gemma4/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`spark-engine` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
