# engine/crates/tokenizer/src

**The Rust source of the `engine-tokenizer` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`chat.rs`](chat.rs) | Qwen3 chat formatting authored from the official tokenizer configuration. |
| [`lib.rs`](lib.rs) | Owned shared BPE implementation for Qwen3 and Gemma 4 tokenizer artifacts. |
| [`unicode.rs`](unicode.rs) | Independently authored Unicode normalization and category lookup. |
| [`unicode_data.rs`](unicode_data.rs) | Generated Unicode character tables (see the Unicode notice in the crate). |
| [`chat/`](chat/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`engine-tokenizer` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
