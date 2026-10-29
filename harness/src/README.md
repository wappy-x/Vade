# harness/src

**The Rust source of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`agent.rs`](agent.rs) | Native, bounded tool-aware orchestration. |
| [`agent_wire.rs`](agent_wire.rs) | Lossless message storage and bounded chat-completion tool-call assembly. |
| [`app_server.rs`](app_server.rs) | Optional installed Codex app-server adapter, implemented with Rust std only. |
| [`artifacts.rs`](artifacts.rs) | Session-scoped, bounded output snapshots. |
| [`auth.rs`](auth.rs) | File credentials for every upstream route; never formatted or persisted. |
| [`branch.rs`](branch.rs) | Transcript branches share a workspace, not file snapshots. |
| [`coding_tools.rs`](coding_tools.rs) | Native coding adapter. |
| [`context.rs`](context.rs) | Explicit context accounting without pretending to own the model tokenizer. |
| [`contract.rs`](contract.rs) | Optional `vade.engine.v1` contract. |
| [`delegation.rs`](delegation.rs) | A delegated backend owns an entire external turn. |
| [`doctor.rs`](doctor.rs) | Bounded diagnostics against an explicitly selected HTTP endpoint. |
| [`edit.rs`](edit.rs) | Exact, bounded text patches. |
| [`gateway.rs`](gateway.rs) | Bounded loopback-only chat and Responses relay. |
| [`gateway_conformance.rs`](gateway_conformance.rs) | Local, first-party gateway conformance fixtures. |
| [`instructions.rs`](instructions.rs) | Project instruction files: one per directory, from the user level down to the workspace root, joined into one bounded prompt section. |
| [`interaction.rs`](interaction.rs) | Bounded UI lanes. |
| [`journal.rs`](journal.rs) | Private, locked, append-only native sessions. |
| [`json.rs`](json.rs) | First-party, bounded JSON parsing. |
| [`kv.rs`](kv.rs) | Optional Engine-owned KV hints. |
| [`lib.rs`](lib.rs) | Vade's terminal harness. |
| [`main.rs`](main.rs) | The program's entry point. |
| [`native_children.rs`](native_children.rs) | Native CLI ownership of the current/last activation's bounded children. |
| [`native_children_tests.rs`](native_children_tests.rs) | Tests for the modules in this folder. |
| [`project.rs`](project.rs) | Bounded project instructions and declarative skills. |
| [`prompt.rs`](prompt.rs) | Stable message ordering and exact contents; no hidden prompt rewriting. |
| [`protocol.rs`](protocol.rs) | Minimal OpenAI-compatible text chat wire types. |
| [`session.rs`](session.rs) | In-memory conversation and optional content-free, private session metadata. |
| [`setup.rs`](setup.rs) | Offline, original configuration templates based on the credited client docs. |
| [`shell.rs`](shell.rs) | Bounded macOS command execution. |
| [`terminal.rs`](terminal.rs) | A bounded terminal transcript and fixed input dock, with a line-mode fallback. |
| [`test_workspace.rs`](test_workspace.rs) | Temporary local fixtures only. |
| [`tools.rs`](tools.rs) | Explicit, bounded, read-only workspace operations. |
| [`transport.rs`](transport.rs) | Standard-library HTTP/1.1 transport for text chat and typed native agents. |
| [`wake.rs`](wake.rs) | Bounded channels that wake their receiving UI thread without a polling delay. |
| [`workload.rs`](workload.rs) | Product workload driver: concurrent native sessions, synthetic context only. |
| [`app_server/`](app_server/README.md) | Folder: see its README. |
| [`child_agents/`](child_agents/README.md) | Folder: see its README. |
| [`cli/`](cli/README.md) | Folder: see its README. |
| [`gateway/`](gateway/README.md) | Folder: see its README. |
| [`setup/`](setup/README.md) | Folder: see its README. |
| [`shell/`](shell/README.md) | Folder: see its README. |
| [`terminal/`](terminal/README.md) | Folder: see its README. |
| [`tools/`](tools/README.md) | Folder: see its README. |
| [`transport/`](transport/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`vade-harness` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
