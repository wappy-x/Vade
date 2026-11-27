# engine/crates/model/src

**The Rust source of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`block_mix.rs`](block_mix.rs) | Fixed-order host block mix of a block residual. See its [contract](block_mix/README.md). |
| [`capped_head.rs`](capped_head.rs) | Host reference of an embedding with a weighted norm, a last norm, and a head with a soft cap on its logits. |
| [`capped_moe.rs`](capped_moe.rs) | Host reference of a routed expert block with soft-capped activation: router pick, one expert, fold. |
| [`cartridge.rs`](cartridge.rs) | Identify a cartridge using bounded metadata, before any tokenizer or GPU work. |
| [`csf.rs`](csf.rs) | Decode bounded CSF scale blocks on the CPU. |
| [`dsa.rs`](dsa.rs) | Host sparse indexer projections, compressed pools and exact token selection. See the [contract](dsa/README.md). |
| [`expert_widen.rs`](expert_widen.rs) | Expert weight decode for two stored formats, widened to BF16 at load. The one seam where expert numbers are made. |
| [`gemma4.rs`](gemma4.rs) | Gemma 4 dense text geometry; multimodal towers are excluded from residency. |
| [`gemma4_assistant.rs`](gemma4_assistant.rs) | Official 31B assistant: query-only decoder over the target's last local/global KV. |
| [`json.rs`](../../base/src/json.rs) | Small strict JSON reader for bounded model metadata. |
| [`kda.rs`](kda.rs) | Fixed-order host KDA reference. See its [source and state contract](kda/README.md). |
| [`layout.rs`](layout.rs) | Check a header fixture and print its written loader gap map. |
| [`lib.rs`](lib.rs) | Bounded model-file inspection and dense Qwen3 checkpoint validation. |
| [`host_ops/`](host_ops/README.md) | Fixed-order router and expert-join references. |
| [`llama.rs`](llama.rs) | Official Llama 3.1/3.2 BF16 dense checkpoint contract. |
| [`loop_host.rs`](loop_host.rs) | Whole-model host form of a looped decoder: token IDs in, logits and the greedy next token out. See its [contract](loop_host/README.md). |
| [`mhc.rs`](mhc.rs) | Fixed-order host Sinkhorn stream blocks and unweighted mean head. |
| [`mla.rs`](mla.rs) | Host f32 latent attention reference with config dimensions and a compressed KV cache. |
| [`mla/`](mla/README.md) | Reference contract, source evidence, and unit tests for latent attention. |
| [`norm_cap.rs`](norm_cap.rs) | Four small shared host functions: weighted norm, plain norm, length clip, soft cap. |
| [`phi4.rs`](phi4.rs) | Phi3ForCausalLM BF16 contract, read from the official Phi-4 checkpoint. |
| [`placement.rs`](placement.rs) | Pure ownership calculations in global model coordinates. |
| [`qwen3.rs`](qwen3.rs) | Shared dense geometry and Qwen3 BF16 checkpoint contracts. |
| [`qwen3_5.rs`](qwen3_5.rs) | Text-only Qwen3.5/3.6/3.8 dense and expert hybrid checkpoint contract. |
| [`qwen3_5_plan.rs`](qwen3_5_plan.rs) | Shared CLI and HTTP hybrid admission and compatibility geometry. |
| [`safetensors.rs`](safetensors.rs) | Header-only safetensors inspection. |
| [`shapes.rs`](shapes.rs) | Shapes and sizes shared by the model plan and the splitter . |
| [`short_conv.rs`](short_conv.rs) | Fixed-order host reference for short causal convolutions over the token axis. See its [contract](short_conv/README.md). |
| [`sink_attention.rs`](sink_attention.rs) | Fixed-order host reference for softmax attention with a sink, a relative position term, length clips, shared key and value heads and an output gate. See its [contract](sink_attention/README.md). |
| [`visit_loop.rs`](visit_loop.rs) | Host visit list, messages and message gates of a looped decoder. See its [contract](visit_loop/README.md). |
| [`bin/`](bin/README.md) | Folder: see its README. |
| [`split/`](split/README.md) | Folder: see its README. |
| [`flash/`](flash/README.md) | Folder: see its README. |
| [`plan/`](plan/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`engine-model` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
