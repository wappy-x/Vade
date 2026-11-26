# harness/tests

**Integration tests: the real command-line program against local mock servers.** No engine, model, GPU or network is needed.

Run them all with `cargo test --offline`.

| Item | What it covers |
| --- | --- |
| `agent_suite_fixtures.rs` | The agent suite fixtures are complete in Git: every input and fixture file that a task names exists and is tracked, and no suite file is untracked or ignored. |
| `suite_dry_run.rs` | `vade-harness suite`: the dry-run gate over the whole agent suite, gate failures on small suites in temporary Git checkouts, and a live run of three runs against a loopback model that streams token IDs. |
| `coding_cli.rs` | The native coding agent in temporary workspaces: tools, approvals, cutoffs, resume and child grants. |
| `coding_context_cli.rs` | The native coding agent's context handling: budgets, compaction, journaled tool output, project rules, forks and continuation. |
| `contract_admission.rs` | Exact admission against loopback fixtures: affinity, cache counts, refusals and transport failures. |
| `contract_cli.rs` | The command-line program against the engine contract: context and cache display, compaction and resume, generation flags, cache permission and sampling policy. |
| `contract_mock.rs` | The engine contract, against loopback fixtures: negotiation, compaction, prompt resets and timing. |
| `delegated_cli.rs` | Handing turns to a first-party stand-in for an outside agent. |
| `gateway_cli.rs` | The gateway across repeated upstreams. |
| `kv_mock.rs` | The engine checkpoint wire contract, with a stateful mock. |
| `mcp.rs` | The MCP client against a fake stdio server in the same test program: handshake, lists, calls, progress, time limits, malformed JSON, restarts, `vade mcp` and an agent turn that calls an MCP tool. |
| `route_a_check.rs` | `scripts/route-a-check.sh` against command mocks: the private home, endpoint and CLI each step uses, model-path checks, launch and stop failures, concurrent runs and interrupts. |
| `sessions_cli.rs` | Saved sessions: a named chat, the session list, a resume by name, the Markdown export and delete. |
| `setup_doctor.rs` | Client setup and the endpoint doctor. |
| `tool_rerun.rs` | An external tool vendor (a fake MCP server in the same test program) counts every call it gets: after a tool error or a server that dies during the call, the original call reached the vendor once, and a resume makes no call. |
| `ui_mock.rs` | The terminal interface's lifecycle, against scripted sockets. |
| `workload_mock.rs` | The workload driver and its reports, against loopback fixtures. |
| [`agent-suite/`](agent-suite/README.md) | Fixtures for the agent task suite: 22 mixed tasks with exact checks. `vade-harness suite` runs them against a live server; `cargo test` runs their dry run with no model. |
| [`fixtures/`](fixtures/README.md) | Stand-in peers and sample data. |
| [`goldens/`](goldens/README.md) | Expected terminal screens. |
| [`support/`](support/README.md) | Shared test helpers. |

**Why:** every feature is checked end to end without outside services, so the tests run anywhere.

**How it fits:** the library code under test is in [`src/`](../src/README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
