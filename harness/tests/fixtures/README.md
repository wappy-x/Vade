# harness/tests/fixtures

**Stand-ins and sample data used by the tests.**

| File | What it is |
| --- | --- |
| `codex_child.rs`, `hermes_child.rs` | First-party stand-ins that speak the Codex and Hermes protocols, so delegation is tested without installing them. |
| `codex-v2-items.jsonl` | Sample protocol items. |
| `suite-replay/` | Scrubbed suite script records for replay grading and error tests. No model or Spark ran. |
| `engine-warm-turn.json` | One warm turn in the engine's reply format for Chat, Responses and Messages, JSON and SSE: 960 of 1000 prompt tokens restored from the retained prefix cache. Assembled from the engine serializers, not recorded live. |
| `engine-capabilities-v1.json` | A sample engine capabilities reply (`vade.engine.v1`). |
| `openai-clients.json` | Requests from the openai Python client (captured), Continue and Open WebUI (from memory) and curl, with engine-format replies; see [OpenAI-compatible API](../../../docs/openai-api.md#how-this-page-is-tested). |
| `web-search/` | Live search pages recorded 2026-11-01, trimmed, with tokens replaced and no cookies (the challenge page keeps its fixed text): DuckDuckGo results, no-results and bot challenge, and Bing results. |
| `lock_spawn_probe.rs` | A standalone Mac diagnostic, not part of the test run. |

**How it fits:** read by the tests in [`tests/`](../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
