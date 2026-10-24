# Splitter

`SplitPlan` computes logical tensor ownership, collective schedules and memory
ledgers from a reconciled model shape and a `SplitRequest`. It is integer
planning: it allocates no GPU memory and executes no collectives.

The [implementation](../engine/crates/model/src/split/mod.rs) is used by the
[typed hybrid loader](../engine/crates/engine/src/hybrid/forward.rs) and its
[distributed binding](../engine/crates/engine/src/hybrid/tp.rs).

## Inputs

`SplitRequest` supplies the rank count, per-rank usable memory and workspace
reserve, context and sequence capacity, cache/state element sizes, draft
retention and expert-row ownership selection. Limits come from
[shared shape definitions](../engine/crates/model/src/shapes.rs).

Pass a shape reconciled against checkpoint headers. A config-only shape has
no complete tensor inventory and cannot establish actual weight residency.

## Outputs

- `owners` assigns logical heads, channels, vocabulary rows and related ranges.
- Tensor selections describe whole, row-sliced, column-sliced or absent storage.
- `layers` records decoder-layer collectives in execution order.
- `ledgers` accounts for weights, cache, state and the caller's workspace reserve.

[Ownership](../engine/crates/model/src/split/owners.rs),
[collectives](../engine/crates/model/src/split/sums.rs) and
[memory accounting](../engine/crates/model/src/split/memory.rs) use checked
arithmetic and return typed errors for unsupported shapes or insufficient budgets.

## The fixed sum order

Reduction segments are contiguous and ordered by rank. The execution binding
must preserve that order. Planning is deterministic for identical inputs;
it does not by itself prove byte equality across different rank counts or
kernel implementations. Native comparisons must cover each supported topology.

## Limits

Workspace bounds come from the caller. The logical ledger is not a measurement
of total process or driver memory. A successful plan is not evidence that a
checkpoint is runnable or numerically qualified. Draft execution and other
runtime-specific operations need their own binding and admission checks.

## Run its tests

From `engine/`, with no model files required:

```sh
cargo test --offline -p engine-model --lib split::tests -- --nocapture
```

The tests cover ownership equivalence, ordered coverage, error handling and
budget boundaries. See [model planning](model-plan.md) and
[adding a model](adding-a-model.md) for the surrounding workflow.
