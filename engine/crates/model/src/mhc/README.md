# Host mHC reference

[`mhc.rs`](../mhc.rs) implements logical FP32 Sinkhorn stream operations.
The [math table and source receipt](../../MHC.md) define its formulas, tensor
layouts, arithmetic order, primitive reuse, and remaining device gates.
The [unit tests](tests.rs) include hand vectors, configured iterations,
residual matrix direction, raw stream collapse, mean head, repeated bits,
row partitions, shape checks, and finite overflow checks.

`Reference::new` accepts decoded tensors. Projection is row-major `[P,N*D]`,
where `P=N*(N+2)`. Base is `[P]`. Scale has pre, post, then comb entries.
`mix` returns pre, post, comb, and the collapsed block input.
The caller runs its block input norm and block, then calls `combine` with
the same raw residual. Comb uses `[source][destination]` order.
`Config::mean_head` averages final raw streams. Final RMS norm is a caller
operation. All source casts are identities under this FP32 contract.

`Config::from_json` reads a direct text config or its `text_config` object.
It requires the operation dimensions, iteration count, and both epsilon
fields. If `mhc` is present, it must be true. `from_description` binds the
loader's typed Sinkhorn description. Neither adapter reads a model name.
Deferred model admission stays closed. This module does not change the
current low-rank HC path.

The iteration count must not exceed `shapes::MAX_SINKHORN_ITERS`.
This is a host resource ceiling, not a source default or a convergence rule.
The [boundary tests](tests.rs) check the JSON parser and direct `Config`
validation. The typed-description adapter uses the same validation.
