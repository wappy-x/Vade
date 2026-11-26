# Sink attention host reference

[`sink_attention.rs`](../sink_attention.rs) defines a fixed-order host
reference for one softmax attention layer with these parts: a sink, a
relative position term, length clips, key and value heads that several
query heads share, short convolutions on keys and values, and an output
gate. A layer attends to a sliding window or to the full prefix.
[`tests.rs`](tests.rs) checks it. This module does not admit a checkpoint,
load weights, or change serving.

## Public source

The operation source is the public Transformers release v5.24.0, commit
`eeee11099135592f1d03c2871fbd3d8132e5a4d7`, file
`src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`.
No source code is copied into this module.

| Source lines | Host operation |
| --- | --- |
| 67-76 | Plain norm of a value head |
| 99-142 | Shared heads, scores, the sink and the softmax |
| 176-236 | Learned scale, length clip and position scale |
| 239-270 | Relative position term |
| 458-543 | Query, key and value projections and the gated output projection |
| 546-727 | The layer with a sliding window and the layer with full reach |

The source runs this attention in its plain mode only, because of the sink
and the relative term.

## Config and weights

`Config::from_json` reads one layer kind from a text config or from its
`text_config` object. It does not read a model name and it has no default
sizes. The caller says if the layer has a window. The fields are:

| Config field | Use |
| --- | --- |
| `hidden_size`, `d_rel`, `sconv_kernel_size`, `rms_norm_eps` | Both kinds |
| `qk_l2_clip_threshold`, `rel_l2_clip_threshold` | Both kinds |
| `swa_num_attention_heads`, `swa_num_key_value_heads`, `swa_head_dim`, `sliding_window_size` | Window layer. The window is also the width of its relative table. |
| `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rel_extent`, `log_scaling_alpha`, `log_scaling_n_floor` | Full layer |
| `q_bias`, `o_bias`, `use_value_norm_no_gamma` | Checked if present. A projection bias or a weighted value norm is refused. |

`Reference::new` checks the length of each weight and that each value is
exact BF16. Fixed convolution taps are the one exception: they are f32.
Let `H` be `hidden_size`, `A` the query heads, `G` the key and value heads,
`D` the head size, `R` the relative width and `E` the relative table width.

| Weight | Shape |
| --- | --- |
| `q_proj`, `gate_proj` | `[A * D, H]` |
| `q_scale` | `[A, D]` |
| `k_proj`, `v_proj` | `[G * D, H]` |
| `k_conv`, `v_conv` | A [short convolution](../short_conv/README.md) over `G * D` channels |
| `rel_proj` | `[A * R, H]` |
| `rel_scale` | `[A, R]` |
| `rel_table` | `[R, E]`, column `d` for the key `d` tokens back |
| `sink_proj` | `[A, H]` |
| `o_proj` | `[H, A * D]` |

## One token

`step` takes three exact BF16 vectors: the query input, the key input and
the value input. The query input also feeds the relative term, the sink and
the output gate. For the token at position `p`, with 0 for the first token:

1. **Query.** Project, multiply by the learned scale, clip the length of
   each head at `qk_clip`, and round. A full layer then multiplies by the
   position scale and rounds again.
2. **Relative state.** The same steps with `rel_proj`, `rel_scale` and
   `rel_clip`.
3. **Key.** Project, convolve, clip the length of each head, and round.
4. **Value.** Project, convolve, apply the plain norm to each head, and round.
5. **Sink.** One projected number for each query head.
6. **Scores.** Query head `h` reads key and value head
   `h / (A / G)`. For a visible key at distance `d`, with 0 for the token
   itself: `a = (q . k) * D^(-1/2) + term(d)`. `term(d)` is the relative
   state times column `d` of the table if `d < E`, and 0 if not.
7. **Softmax.** `w = exp(a - m) / (sum of exp(a - m) + exp(sink - m))`,
   with `m` the largest of the scores and the sink. The sink takes weight
   and has no value.
8. **Values.** `o = sum of w * v` for each query head.
9. **Output.** `sigmoid(gate_proj * x) * o`, then `o_proj`.

The position scale is `1 + alpha * ln(max((p + 1) / floor, 1))`. The
division by the floor is a product with the f32 reciprocal of the floor,
made by `reciprocal` of [`short_conv.rs`](../short_conv.rs). The scale is
exactly 1 while that product is not above 1.

A window layer sees the token and the `window - 1` tokens before it. A full
layer sees every token. In a full layer only the nearest `E` keys have a
relative term.

The length clip is `x * min(1, T / (length(x) + 1e-6))`. `CLIP_EPSILON` is
that `1e-6`. The division is two rounded f32 steps: the reciprocal
`r = 1 / (length(x) + 1e-6)`, then the product `T * r`. The function
`number_over` holds this form. The source's run has this form for a number
divided by a 32-bit tensor. The receipt of that measurement is not in this
repository.

The plain norm is `x * 1 / sqrt(mean(x * x) + eps)` with no learned weight.

## Number contract

A value that the source holds as BF16 is an f32 here with exact BF16 bits.
All arithmetic is f32. Products and sums are separate operations.

| Step | Rounding to BF16 |
| --- | --- |
| Each projection (`q`, `k`, `v`, relative, sink, gate, output) | One time, after the sum |
| Query and relative state | After the clip. In a full layer also after the position scale. |
| Key clip, value norm, each convolution | One time |
| Relative state times a table column | One time |
| Score | Three times: after the dot product, after the scale, after the relative term |
| Softmax weight | One time, before it meets a value |
| Weighted value sum | One time |
| Gate times head output | One time |

The score scale is the f32 value of `D^(-1/2)`. It is not rounded to BF16.
`score_scale` is the one place that holds this form.

## Fixed sum orders

The source leaves the order of its sums to a library. This reference uses
the orders of the engine's device operations, so that a device run can give
the same words. No order depends on the place of a token in a buffer.

| Sum | Order |
| --- | --- |
| Projection, dot product, table column | One chain in ascending index from +0.0 |
| Squares of a length clip | `square_tree`: 32 lanes, lane `i mod 32` a chain, then a join at the distances 16, 8, 4, 2, 1 |
| Squares of the value norm | `norm_squares`: `square_tree` for each chunk of 128 values, then the chunk sums in 32 lanes and the same join |
| Softmax sum and weighted value sum | `distance_sum`, see below. The sink is added last. |

The distance order takes its terms by the distance `d` of the key from the
query token. Distances are cut into blocks of 1,024. In a block, lane
`d mod 32` is a chain in ascending `d` from +0.0, and the lanes join as
above. The block sums are one chain in ascending block number. The order is
a function of the distance and of the count of visible keys. The two trees
give the same value for a head of at most 128 values.

Every exponential, logarithm and BF16 round comes from the engine's shared
math file, [`fixed_math.rs`](../../../gpu/src/fixed_math.rs). The device
operations use the same file. No platform math library is called.

Two small functions here, `length_clip` and `plain_norm`, are marked TEMP.
They are local copies of shared functions that are not yet in this tree.

## State

`State` holds the finished keys and values of one layer visit for one
sequence: after the convolution, and after the clip or the norm. A window
layer keeps only the tokens that the next token can see. The two
convolution histories are in `VisitHistories` of
[`short_conv.rs`](../short_conv.rs). A layer that runs more than one time
for a token needs one `State` and one `VisitHistories` for each visit.

`step` and `scan` change no state if they return an error. `scan` runs a
token-major chunk with the step arithmetic, so a prompt in one piece and a
prompt in chunks give the same bits. `step_parts` also returns each stage
of a token, for a check of one stage at a time.

## Proof

The unit tests use small shapes and hand-worked values: a sequence start,
the window edge, a distance outside the relative table, the table column of
each distance, the position scale, the learned scale before the clip, a
zero gate and a closed gate, the three roundings of a score, the rounding
of the softmax weights, shared heads, the named sum orders, the config
reader, and refusals that change no state. The literal values come from an
exact-fraction calculator that rounds each f32 step one time. A second
check in the tests is a whole-sequence oracle that does not call the
reference. It runs 7 layer shapes, and compares one scan with a scan in
chunks. The calculator is not in this repository.

The ignored test
[`sink_attention_trace.rs`](../../tests/sink_attention_trace.rs) compares
this reference with a layer trace of the source code on the real weights of
a public checkpoint, for one window layer and one full layer. The weights
and the trace stay outside this repository, in the ignored folder
`.local/capped-reference/`. The test asks that at least 9 of 10 values of
the state after a visit equal the trace bit for bit. It also asks that each
opposite layout, and a run with no relative term or no sink, gives under 1
of 10. It cannot ask for equal bits: the source's library does not fix the
order of its sums. This file gives no measured numbers, because their
receipts are not in this repository.

## Device work left

A device path needs these operations with the same words as this
reference: the clip with a scale and a position factor, the relative table
product, both convolutions, the scores with three roundings, the softmax
with a sink in the distance order, and the gated value sum. It also needs a
key and value state for each layer visit, with a ring for window layers.
Host tests do not establish GPU exactness or speed. Device integration
needs the full authorized exactness gate.

**Back:** [Model source](../README.md)
