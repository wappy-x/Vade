# crates/tokenizer/templates

**Chat templates: the rules for turning a conversation into one prompt string, per model family.**

| File | Model family |
| --- | --- |
| [`qwen3.jinja`](qwen3.jinja) | Qwen3 and Qwen3.8 |
| [`gemma4.jinja`](gemma4.jinja) | Gemma 4 |
| [`llama3.jinja`](llama3.jinja), [`llama31.jinja`](llama31.jinja) | Llama 3 and 3.1 |
| [`phi4.jinja`](phi4.jinja) | Phi-4 |

**Why:** a model answers best when its prompt is laid out exactly as it was trained.

**How it fits:** the tokenizer crate builds prompts in Rust and checks them against these files; see [`engine-tokenizer`](../README.md). Which files are copied from official models is recorded in [`NOTICE`](../../../NOTICE).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
