# harness/src/child_agents

**Rust modules for `child_agents`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`http_tests.rs`](http_tests.rs) | First-party loopback mocks only. |
| [`mod.rs`](mod.rs) | Bounded, parent-owned native children. |
| [`native.rs`](native.rs) | Exact-admission HTTP child adapter. |
| [`native_tests.rs`](native_tests.rs) | Real native executors driven by deterministic model replies. |
| [`native_tools.rs`](native_tools.rs) | The real coding executor, with host-owned approvals and one private journal and build session per child. |
| [`parent.rs`](parent.rs) | No summary comment yet; read the file. |
| [`policy.rs`](policy.rs) | Capability masks and in-process write ownership, not an OS sandbox. |
| [`prefix.rs`](prefix.rs) | No summary comment yet; read the file. |
| [`tests.rs`](tests.rs) | Tests for the modules in this folder. |
| [`ui.rs`](ui.rs) | Bounded plain/TUI child list. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `child_agents`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
