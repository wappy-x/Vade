# harness/tests/fixtures

**Stand-ins and sample data used by the tests.**

| File | What it is |
| --- | --- |
| `codex_child.rs`, `hermes_child.rs` | First-party stand-ins that speak the Codex and Hermes protocols, so delegation is tested without installing them. |
| `codex-v2-items.jsonl` | Sample protocol items. |
| `engine-capabilities-v1.json` | A sample engine capabilities reply (`vade.engine.v1`). |
| `lock_spawn_probe.rs` | A standalone Mac diagnostic, not part of the test run. |

**How it fits:** read by the tests in [`tests/`](../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
