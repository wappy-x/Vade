# harness/src/transport

**Rust modules for `transport`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`kv.rs`](kv.rs) | Native checkpoint lifecycle. |
| [`request.rs`](request.rs) | Pooled HTTP requests: the engine, native and gateway entry points over one policy. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `transport`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
