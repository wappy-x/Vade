# engine/crates/serve/tests

**Rust modules for `tests`, part of the `spark-serve` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`chat_output.rs`](chat_output.rs) | Integration tests for this crate. |
| [`glm_text_contract.rs`](glm_text_contract.rs) | Host text vectors and complete EOS checks. See the [sources and proof limits](fixtures/glm-text/README.md). |
| [`text_checkpoint.rs`](text_checkpoint.rs) | Original tensor bytes and retained files for resident text edges. |
| [`support/glm_tokenizer_evidence.rs`](support/glm_tokenizer_evidence.rs) | Complete source row inventory and fixed publisher delimiter checks. See the [evidence note](fixtures/glm-tokenizer/README.md). |
| [`fixtures/glm-upstream/`](fixtures/glm-upstream/README.md) | Published external prompt vectors and their proof limits. |
| [`scheduler.rs`](scheduler.rs) | Integration tests for this crate. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `tests`.

**How it fits:** see the [`spark-serve` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
