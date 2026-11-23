# Host block mix reference

[`block_mix.rs`](../block_mix.rs) is a fixed-order f32 host reference for the
block mix of a block residual. It does not enable a model plan and it does not
change the serving path.

| File | What it does |
| --- | --- |
| [`tests.rs`](tests.rs) | Unit tests with hand values for the plain norm, the mix and the end mix. |

## What the operation is

A decoder with a block residual groups its layers in blocks. Inside a block
the stream holds only the sum that the layers of that block added. The stream
does not hold the input of the block. A layer reads the past through the block
mix: a softmax blend over the finished block vectors and the stream.

One mix has one learned query for each result. The attention mix of a layer
has one query, or three queries for the query, key and value inputs. The
expert mix has one query. The end mix before the last norm has one query.

## Formula and order

Inputs: the finished block vectors, oldest first, the stream, and the queries
as row-major `[rows][hidden]`.

1. Put the stream after the block vectors.
2. Norm each entry with the plain norm: `x * (1 / sqrt(mean(x^2) + 1e-6))`.
   There is no learned weight. The sum of squares uses the 32-lane tree below.
3. For each query row, each score is the dot product of the query and the
   normed entry, as a chain in rising index from zero. Then one product with
   the f32 value of `hidden^-0.5`.
4. Softmax over the entries: take the largest score out first, add the
   exponentials as a chain in entry order, divide each one by the sum. There
   is no sink and no position term.
5. The result is the sum of `weight * normed entry`, as a chain in entry
   order from zero.

The result is a blend of the normed entries, not of the raw entries. All
arithmetic is f32. A product and a sum are separate f32 operations, never a
fused multiply-add. The result stays f32: no step rounds to BF16.

The epsilon `1e-6` belongs to the operation. The source builds the plain norm
of the mix with its own default. It does not use the config field
`rms_norm_eps` there.

## Sum orders and scalar math

The orders are the engine's own, so that a device path can give the same bits.

- **Chain:** one chain in rising index from zero. This is the order of the
  plain row products ([`plain_bf16.rs`](../../../gpu/kernels/src/plain_bf16.rs)).
- **32-lane tree:** lane `i mod 32` adds the squares of its elements in rising
  index from zero. The lanes join at distances 16, 8, 4, 2 and 1. This is
  `square_sum` in [`causal_ops.rs`](../../../gpu/src/causal_ops.rs). The test
  `the_sum_of_squares_uses_the_32_lane_tree` pins it.

Scalar math is the shared file
[`fixed_math.rs`](../../../gpu/src/fixed_math.rs) and nothing else. The model
crate includes that file by path as `fixed_math`, and the device code builds
the same source. No file of this reference calls the platform's `exp`: it
does not give the device's bits and it can differ between hosts. The test
`the_softmax_uses_the_shared_exponential` pins the exponential and the order
of the softmax.

## Source

Transformers v5.24.0, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`,
`src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`:

- Lines 67-76: the plain norm.
- Lines 126 and 137-139: the score, the softmax and the blend.
- Lines 828-853: the mix. Line 1442: the end mix.
- Lines 1192-1203: the queries stay f32.

The source does not give the order of the additions inside its products. The
orders above are the engine's. They do not claim bit parity with the source
library.

## API

- `plain_norm(x, eps)` returns the normed vector.
- `mix(hidden, queries, blocks, stream)` returns the weights as
  `[query][entry]` and the results as `[query][hidden]`.
- `end_mix(hidden, output_query, blocks, stream)` returns the one result. The
  caller applies the last norm next.

The [visit loop reference](../visit_loop/README.md) uses this module for the
mixes of each visit and for the messages between passes.

## Proof and remaining device work

The [unit tests](tests.rs) use a hidden size of 4, so each normed test vector
and each softmax weight is a short fraction. They check one query, three
queries, the end mix over five entries, a zero entry, an empty block list,
the 32-lane tree, the shared exponential, repeated bits, a recomputation in
f64, and refused shapes and values.

No device code exists for this operation. A device path must use the same
orders and the same scalar math as this file. Host tests do not establish GPU
exactness or speed.

**Back:** [Model source](../README.md)
