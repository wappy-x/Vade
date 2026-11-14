# engine/crates/serve/src

**The Rust source of the `spark-serve` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`anthropic.rs`](anthropic.rs) | Stateless Anthropic Messages translation into the shared Chat path. |
| [`auth.rs`](auth.rs) | Startup credentials and fixed-work request authentication, using std only. |
| [`budget.rs`](budget.rs) | Serving admission uses each request's prompt plus output limit. |
| [`cache.rs`](cache.rs) | Elastic token-prefix snapshots. |
| [`chat_output.rs`](chat_output.rs) | Incremental Qwen and Gemma chat markup decoding. |
| [`contract.rs`](contract.rs) | Additive, versioned engine contract shared by single- and multi-rank serving. |
| [`gemma4.rs`](gemma4.rs) | Text serving adapter over the shared BF16 Gemma cartridge. |
| [`generation.rs`](generation.rs) | No summary comment yet; read the file. |
| [`http.rs`](http.rs) | HTTP/1.1, one request per connection, Content-Length only. |
| [`hybrid.rs`](hybrid.rs) | Dense hybrid cartridge adapter for the existing cooperative scheduler. |
| [`lib.rs`](lib.rs) | Bounded HTTP JSON/SSE responses and cooperative single-engine-thread scheduling. |
| [`protocol.rs`](protocol.rs) | Chat-completions requests and JSON encoding. |
| [`qwen.rs`](qwen.rs) | Actual owned-engine adapter. |
| [`responses.rs`](responses.rs) | Stateless Responses wire translation. |
| [`scheduler.rs`](scheduler.rs) | One CUDA thread, bounded nonblocking sockets, priority between model steps. |
| [`tokenizer.rs`](tokenizer.rs) | Bridge to the authored tokenizer and exact checkpoint chat template. |
| [`tokenizer/`](tokenizer/README.md) | GLM text chat renderer and hand-derived golden tests. |
| [`usage.rs`](usage.rs) | Shared Chat/Responses accounting. |
| [`bin/`](bin/README.md) | Folder: see its README. |
| [`chat_output/`](chat_output/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

## Text program

`text_program.rs` binds checked text metadata to ordered request transactions.
[`text_checkpoint.rs`](text_checkpoint.rs) binds the retained checkpoint files
to the resident text edge layout and the existing file uploader.
See [the contract and remaining gates](../TEXT-PROGRAM.md).

**How it fits:** see the [`spark-serve` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
