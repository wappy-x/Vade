# harness/tests

**Integration tests: the real command-line program against local mock servers.** No engine, model, GPU or network is needed.

Run them all with `cargo test --offline`.

| Item | What it covers |
| --- | --- |
| `agent_suite_fixtures.rs` | The agent suite fixtures are complete in Git: every input and fixture file that a task names exists and is tracked, and no suite file is untracked or ignored. |
| `coding_cli.rs` | The native coding agent in temporary workspaces: tools, approvals, cutoffs, resume and child grants. |
| `coding_context_cli.rs` | The native coding agent's context handling: budgets, compaction, journaled tool output, project rules, forks and continuation. |
| `contract_admission.rs` | Exact admission against loopback fixtures: affinity, cache counts, refusals and transport failures. |
| `contract_cli.rs` | The command-line program against the engine contract: context and cache display, compaction and resume, generation flags, cache permission and sampling policy. |
| `contract_mock.rs` | The engine contract, against loopback fixtures: negotiation, compaction, prompt resets and timing. |
| `delegated_cli.rs` | Handing turns to a first-party stand-in for an outside agent. |
| `gateway_cli.rs` | The gateway across repeated upstreams. |
| `kv_mock.rs` | The engine checkpoint wire contract, with a stateful mock. |
| `route_a_check.rs` | `scripts/route-a-check.sh` against command mocks: the private home, endpoint and CLI each step uses, model-path checks, launch and stop failures, concurrent runs and interrupts. |
| `setup_doctor.rs` | Client setup and the endpoint doctor. |
| `ui_mock.rs` | The terminal interface's lifecycle, against scripted sockets. |
| `workload_mock.rs` | The workload driver and its reports, against loopback fixtures. |
| [`agent-suite/`](agent-suite/README.md) | Fixtures for the agent task suite: 12 mixed tasks with exact checks. A suite runner drives them against a live server; `cargo test` checks only that their files are complete. |
| [`fixtures/`](fixtures/README.md) | Stand-in peers and sample data. |
| [`goldens/`](goldens/README.md) | Expected terminal screens. |
| [`support/`](support/README.md) | Shared test helpers. |

**Why:** every feature is checked end to end without outside services, so the tests run anywhere.

**How it fits:** the library code under test is in [`src/`](../src/README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
