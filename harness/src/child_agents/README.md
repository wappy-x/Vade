# harness/src/child_agents

**Rust modules for `child_agents`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`http_tests.rs`](http_tests.rs) | First-party loopback mocks only. |
| [`lifecycle_tests.rs`](lifecycle_tests.rs) | Child lifecycle: bounded concurrency, joined workers, cancellation from the parent or the UI, and lifecycle wakeups without polling. |
| [`mod.rs`](mod.rs) | Bounded, parent-owned native children. |
| [`native.rs`](native.rs) | Exact-admission HTTP child adapter. |
| [`native_tests.rs`](native_tests.rs) | Real native executors driven by deterministic model replies. |
| [`native_tools.rs`](native_tools.rs) | The real coding executor, with host-owned approvals and one private journal and build session per child. |
| [`parent.rs`](parent.rs) | Running a parent with children: the child_* tool schemas, their dispatch and joining uncollected children. |
| [`policy.rs`](policy.rs) | Capability masks and in-process write ownership, not an OS sandbox. |
| [`prefix.rs`](prefix.rs) | The frozen parent prefix a child starts from: rules and tool schemas, no transcript. |
| [`quota_tests.rs`](quota_tests.rs) | Child resource allocation and tool quota enforcement. |
| [`scope_tests.rs`](scope_tests.rs) | Write scopes and grants: read-only and non-recursive scopes, disjoint assignments, unsafe paths and cancellation in the middle of a write. |
| [`tests.rs`](tests.rs) | Tests for the modules in this folder. |
| [`ui.rs`](ui.rs) | Bounded plain/TUI child list. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `child_agents`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
