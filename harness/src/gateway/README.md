# harness/src/gateway

**Rust modules for `gateway`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`cluster_tests.rs`](cluster_tests.rs) | Three independent, first-party loopback endpoints with mock prefix caches. |
| [`routing.rs`](routing.rs) | Cache locality is a scheduling hint, never a cache-hit or security claim. |
| [`routing_tests.rs`](routing_tests.rs) | Tests for the modules in this folder. |
| [`system_turns.rs`](system_turns.rs) | Two client role values the engine refuses, relabeled so the request runs. |
| [`usage.rs`](usage.rs) | Bounded, best-effort telemetry tap. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `gateway`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
