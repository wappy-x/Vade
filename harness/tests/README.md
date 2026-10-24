# harness/tests

**Integration tests: the real command-line program against local mock servers.** No engine, model, GPU or network is needed.

Run them all with `cargo test --offline`.

| Item | What it covers |
| --- | --- |
| `coding_cli.rs` | The native coding agent in temporary workspaces. |
| `contract_mock.rs` | The engine contract, against loopback fixtures. |
| `delegated_cli.rs` | Handing turns to a first-party stand-in for an outside agent. |
| `gateway_cli.rs` | The gateway across repeated upstreams. |
| `kv_mock.rs` | The engine checkpoint wire contract, with a stateful mock. |
| `setup_doctor.rs` | Client setup and the endpoint doctor. |
| `ui_mock.rs` | The terminal interface's lifecycle, against scripted sockets. |
| [`fixtures/`](fixtures/README.md) | Stand-in peers and sample data. |
| [`goldens/`](goldens/README.md) | Expected terminal screens. |
| [`support/`](support/README.md) | Shared test helpers. |

**Why:** every feature is checked end to end without outside services, so the tests run anywhere.

**How it fits:** the library code under test is in [`src/`](../src/README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
