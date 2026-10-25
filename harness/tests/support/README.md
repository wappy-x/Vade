# harness/tests/support

**Shared helpers for the integration tests.**

| File | What it does |
| --- | --- |
| `cli.rs` | Runs the CLI with time limits. |
| `raw_ui.rs` | A shared macOS terminal (PTY) fixture for interface checks. |
| `coding.rs` | Temporary workspaces and a scripted chat-completions mock for the coding-agent tests. |
| `contract.rs` | Loopback engine fixtures shared by the contract, workload and CLI contract tests. |
| `children_contract.rs`, `children_native.rs` | Parent and child agent scenarios against loopback mocks. |

**How it fits:** used by the tests in [`tests/`](../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
