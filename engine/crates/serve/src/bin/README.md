# engine/crates/serve/src/bin

**Programs built from the `spark-serve` crate, one per file.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`qwen-reference-check.rs`](qwen-reference-check.rs) | Compare real chat tokenization and teacher-forced logits with the one-time HF oracle. |
| [`serve-draft-check.rs`](serve-draft-check.rs) | Compare streamed IDs and text over HTTP, with cold prefill for every mode. |
| [`serve-http-check.rs`](serve-http-check.rs) | Focused actual-server check: branch equality, foreground latency, cancellation. |
| [`spark-serve.rs`](spark-serve.rs) | The `spark-serve` program. |
| [`text-check.rs`](text-check.rs) | Check pinned text vectors and original checkpoint embedding and head bytes. See the [probe contract](../../TEXT-CHECK.md). |
| [`spark-serve/`](spark-serve/README.md) | Folder: see its README. |

**Why a folder:** Cargo builds every file in a `bin/` folder as its own program.

**How it fits:** see the [`spark-serve` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
