# engine/crates/gpu/src/bin

**Programs built from the `spark-gpu-probe` crate, one per file.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`load-range-check.rs`](load-range-check.rs) | Focused staged-loader checks: exact rows, uneven TP ranks, bounds, cleanup. |

**Why a folder:** Cargo builds every file in a `bin/` folder as its own program.

**How it fits:** see the [`spark-gpu-probe` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
