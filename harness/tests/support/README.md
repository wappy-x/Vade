# harness/tests/support

**Shared helpers for the integration tests.**

| File | What it does |
| --- | --- |
| `cli.rs` | Runs the CLI with time limits. |
| `raw_ui.rs` | A shared macOS terminal (PTY) fixture for interface checks. |
| `children_contract.rs`, `children_native.rs` | Parent and child agent scenarios against loopback mocks. |

**How it fits:** used by the tests in [`tests/`](../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
