# engine/crates/model/src/split

**Rust modules for `split`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`error.rs`](error.rs) | Typed failures for logical splitting and memory budgets. |
| [`memory.rs`](memory.rs) | Checked resident weight, cache, state and workspace ledgers. |
| [`mod.rs`](mod.rs) | Pure logical ownership, tensor selections and budgets for a model shape. |
| [`owners.rs`](owners.rs) | Contiguous owners and role-based selections, including rotated expert blocks. |
| [`sums.rs`](sums.rs) | Rank-ordered K segments and decoder-layer collective schedules. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `split`.

**How it fits:** see the [`engine-model` README](../../README.md) for what the crate does and how it connects to the rest. The splitter reads shared [shapes](../shapes.rs) and emits logical units; it allocates no device memory and executes no sums.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
