# Splitter

`SplitPlan` decides how a model is divided across Sparks. Given a checked
model shape and a `SplitRequest`, it works out three things: which Spark owns
which part of each tensor, which sums and gathers pass between Sparks and in
what order, and how much memory each Spark spends. It is integer arithmetic only: it allocates no GPU
memory and sends nothing between Sparks.

The [code](../engine/crates/model/src/split/mod.rs) is used by the
[typed hybrid loader](../engine/crates/engine/src/hybrid/forward.rs) and its
[multi-rank binding](../engine/crates/engine/src/hybrid/tp.rs).

## Inputs

A `SplitRequest` gives:

- the number of Sparks (ranks);
- each Spark's usable memory and the workspace it keeps back;
- the context length and the number of full-context sequences to fund;
- the bytes per cached element and per recurrent-state element;
- whether to keep the drafting tensors;
- whether expert down projections are split by output rows.

The limits come from the
[shared shape definitions](../engine/crates/model/src/shapes.rs).

Pass a shape that was checked against the checkpoint's headers. A shape built
from the config alone has no complete tensor list, so it can't say where each
weight really lives.

## Outputs

- `owners` gives each Spark its ranges of heads, channels, vocabulary rows and
  the like.
- Each tensor is kept whole, split by rows, split by columns, or not stored.
- `layers` lists each decoder layer's cross-Spark sums and gathers in
  execution order.
- `ledgers` counts the bytes for weights, cache, state and the workspace each
  Spark keeps back.

[Ownership](../engine/crates/model/src/split/owners.rs),
[sums](../engine/crates/model/src/split/sums.rs) and
[memory accounting](../engine/crates/model/src/split/memory.rs) use checked
arithmetic. An unsupported shape or a budget that doesn't fit returns a typed
error.

## The fixed sum order

Each cross-Spark sum is cut into contiguous segments, ordered by rank, and the
code that runs it must keep that order. The same inputs always give the same
plan. That alone doesn't make the bytes equal across different Spark counts or
kernels, so device comparisons must cover every supported layout.

## Limits

The workspace limits come from the caller. A ledger counts planned bytes; it
is not a measurement of what the process or the driver really uses. A plan
that fits doesn't show that a checkpoint runs or that its numbers are
qualified. Drafting and other runtime work need their own binding and
admission checks.

## Run its tests

From `engine/`, with no model files:

```sh
cargo +1.99.0 test --locked --offline -p engine-model --lib split::tests -- --nocapture
```

The tests cover equivalent ownership, ordered coverage, error handling and
budget limits. See the [model plan](model-plan.md) and
[adding a model](adding-a-model.md) for the steps around it.
