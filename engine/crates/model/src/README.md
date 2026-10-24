# engine/crates/model/src

**The Rust source of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`cartridge.rs`](cartridge.rs) | Identify a cartridge using bounded metadata, before any tokenizer or GPU work. |
| [`gemma4.rs`](gemma4.rs) | Gemma 4 dense text geometry; multimodal towers are excluded from residency. |
| [`gemma4_assistant.rs`](gemma4_assistant.rs) | Official 31B assistant: query-only decoder over the target's last local/global KV. |
| [`json.rs`](../../base/src/json.rs) | Small strict JSON reader for bounded model metadata. |
| [`lib.rs`](lib.rs) | Bounded model-file inspection and dense Qwen3 checkpoint validation. |
| [`llama.rs`](llama.rs) | Official Llama 3.1/3.2 BF16 dense checkpoint contract. |
| [`phi4.rs`](phi4.rs) | Phi3ForCausalLM BF16 contract, read from the official Phi-4 checkpoint. |
| [`placement.rs`](placement.rs) | Pure ownership calculations in global model coordinates. |
| [`qwen3.rs`](qwen3.rs) | Shared dense geometry and Qwen3 BF16 checkpoint contracts. |
| [`qwen3_5.rs`](qwen3_5.rs) | Text-only Qwen3.5/3.6/3.8 dense and expert hybrid checkpoint contract. |
| [`qwen3_5_plan.rs`](qwen3_5_plan.rs) | Shared CLI and HTTP hybrid admission and compatibility geometry. |
| [`safetensors.rs`](safetensors.rs) | Header-only safetensors inspection. |
| [`shapes.rs`](shapes.rs) | Shapes and sizes shared by the model plan and the splitter . |
| [`bin/`](bin/README.md) | Folder: see its README. |
| [`split/`](split/README.md) | Folder: see its README. |
| [`flash/`](flash/README.md) | Folder: see its README. |
| [`plan/`](plan/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`engine-model` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
