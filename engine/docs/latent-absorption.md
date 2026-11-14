# Fixed-order latent absorption

This arithmetic path applies to admitted plain BF16 latent KV projections with
no rotary dimensions and an even latent rank. Admission uses tensor roles,
shapes, and dtypes. Other projections retain their existing arithmetic.
The path changes GLM reference answers. It requires the separate numerical
reset gates before release. Qwen reference answers must stay unchanged.

## SUM-ORDER v1

Each reduction uses 32 FP32 sums, initially positive zero. Term `j` enters
sum `(j / 8) % 32`. Each sum receives its terms in ascending `j` order.
Join the sums at distances 16, 8, 4, 2, and 1. At each distance `d`, sum
`i` adds sum `i+d` for `i < d`. Empty sums also take part in the joins.

Round every product before adding it. Each operation uses binary32 round
to nearest, ties to even, with gradual underflow. Do not use fused
multiply-add. NaN locations must agree; NaN payload bits are unspecified.
The order must not depend on Spark count, occupancy, batching, or chunking.
A change to this definition requires a new version and a new golden reset.

## Product grouping

Let `fold` mean SUM-ORDER v1. Checkpoint key and value weights and the
selected latent rows keep their existing BF16 operand bits. `fl` means
one FP32 rounding. `bf16` means the existing BF16 rounding operation.
In this path `bf16` keeps every NaN a NaN. A NaN input gives a quiet
NaN with the same sign. The shared rounding adds a carry that can reach the
sign bit, so the device's canonical NaN would become -0.0 and hide the NaN.
Finite and infinite words round as before.
For each whole attention head:

```
q_abs[j] = bf16(fold_d(fl(query[d] * W_K[d,j])))
score[t] = fl(fold_j(fl(q_abs[j] * latent[t,j])) * fl(1 / sqrt(key_dim)))
probability = existing softmax(score), with its existing BF16 rounding
u[j] = bf16(fold_t(fl(probability[t] * latent[t,j])))
out[d] = bf16(fold_j(fl(u[j] * W_V[d,j])))
```

`q_abs` and `u` round to BF16 once, where they are written. This matches
the rounding points of the absorbed (top-k MQA) path of the maker's BF16
stack, which never expands keys or values. Selection order stays unchanged. The loader
transposes key weight bits without arithmetic. Value weights keep their
checkpoint order. Each rank owns complete heads, so no reduction crosses
ranks. Prompt groups and scalar decode use the same operations per token.

Prompt groups execute five grouped operations. One tile kernel runs the four
fold steps. A block owns 32 rows of one operand and 64 rows of the other, and
a thread owns 2 x 4 dot products for the whole reduction. The block stages the
columns in shared memory in chunks of four sums, one 256-column period at a
time, and stages the next chunk while it multiplies the current one. Each sum
takes its columns in ascending order. Chunk `g` holds sums `g`, `g+16`, `g+8`,
and `g+24`. The chunks run in the order 0, 4, 2, 6, 1, 5, 3, 7. A thread then
needs three held values for each dot product to join the 32 sums exactly as
the joins at distances 16, 8, 4, 2, and 1 do. Scores and values read the
selected cache rows directly, in each token's original selection order. There
is no gathered cache copy for each token. The tile changes no product, sum
membership, join, or rounding boundary. Empty selections return positive
zero. The workspace charge covers the two FP32 latent planes for the largest
layer.

This changes product grouping and moves the BF16 rounding from expanded
keys and values to the absorbed query and the latent output. It is more than a change in the addition tree. The reset
receipt must name these changes.

## Decode step: no host read, one size a step

The decode path of an absorbed layer reads nothing on the host. The
selection count follows from the context (kept pools times the pool
size, plus the tail rows; `bounded_decoder::selected_count`). The device
count word is still checked: the Gather of the layer leaves a deferred
check of it (`selection_checks`), so a NaN score is refused as before.
The row identities are not read, because nothing in the absorbed path
uses them.

Scores and probabilities are heads times live rows words. Their storage
is heads times the layer's slot bound, and the value is a range of it.
So a decode step takes vectors of one size at every context, and the
decode reserve can reuse them. Kernels, shapes, and launch grids do not
depend on the storage size.

## Validation

`latent-absorb` is a benchmark binary under the `kernel-bench` feature.
It checks strip conformance, all five CPU/GPU planes, and real checkpoint
loader ownership for simulated 1-, 2-, and 4-rank layouts on one GPU.
It compares packed attention and absorption with an actual new-row
projection in the packed baseline. Synthetic activations and component
timings do not establish full decoder speed or quality.

The release gate must compare fresh loads, physical 1-, 2-, and 4-Spark
runs, prompt and decode forms, chunking, cache settings, and supported
MTP settings. Generate new goldens twice and require equal IDs and raw
outputs. Keep old and new outputs, first changed tokens, and checkpoint,
request, source, and compiler hashes. Check quality against the maker's
reference and report old/new answer agreement. Qwen must retain its
existing 72/72 gate. Missing gates do not count as passes.
