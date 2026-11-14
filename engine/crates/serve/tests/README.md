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
| [`support/scheduler_engine.rs`](support/scheduler_engine.rs) | Shared host fake with history-based token IDs and ownership counts. |
| [`support/scheduler_model.rs`](support/scheduler_model.rs) | Bounded enumeration through the production scheduler. Built as scheduler unit tests. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `tests`.

The `scheduler_model_small_cases` test checks all ordered request lists within
its bound. Requests arrive at turn 0, 1 or 3. Prompt lengths are 1, 4 or 9.
The fake prompt slice has 4 tokens. Each request asks for 1, 2 or 3 output
tokens. Lists have at most 4 requests. Their total work must be at most 4,
where request work is `ceil(prompt / 4) + output - 1`. Requests with the same
arrival retain every admission order. Each list runs without cancellation,
then with one cancellation at each turn from that request's arrival through
turn 8. The test asserts 752 base lists and 18,232 scenarios. These counts
come from the generator in `scheduler_model_small_cases`. They do not depend
on observed completion turns. These constants are test limits, not engine settings.

The test feeds prepared requests into the real dispatch loop. Unix socket
pairs carry real response bytes. Test hooks advance prompt age and skip idle
parking. Scheduler time reads use a test clock that advances by one millisecond
per model turn. The product clock wrappers retain the original time reads.
The test uses no preparation worker, socket intake, prefix cache or draft.
The fake allocation pool funds two sessions. Session creation checks the pool,
and each observed reservation must match the fake allocation. Normal completion
must retire the session before release. Cancellation may release it directly.
It checks host scheduling and these fake allocation contracts. It gives no GPU
exactness evidence and does not prove fairness under sustained arrivals.

**How it fits:** see the [`spark-serve` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
