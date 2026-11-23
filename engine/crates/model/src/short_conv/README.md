# Short convolution host reference

[`short_conv.rs`](../short_conv.rs) defines a fixed-order host reference for
short causal convolutions over the token axis. [`tests.rs`](tests.rs) checks
it. This module does not admit a checkpoint, load weights, or change
serving. The caller supplies decoded weights and all sizes.

## Public source

The operation source is the public Transformers release v5.24.0, commit
`eeee11099135592f1d03c2871fbd3d8132e5a4d7`, file
`src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`.
No source code is copied into this module.

| Source lines | Host operation |
| --- | --- |
| 285-325 | The causal depthwise convolution calls, for one token and for a sequence |
| 328-393 | Fixed taps, the added input and the division by `sqrt(taps + 1)` |
| 396-455 | Taps made from the token, shared in a head, and the division by `sqrt(2 * taps + 1)` |
| 562-567, 868-873 | The four convolutions of a layer visit and their state index |

## The two kinds

Each convolution mixes the input of a token with the inputs of the
`taps - 1` tokens before it, one channel at a time. Tokens before the start
of the text count as zero.

**Fixed taps.** `FixedTaps` holds weights `[channel][tap]`. For channel `c`:

`y = (sum over taps of w[c][tap] * u(t - lag(tap))[c] + u(t)[c]) / sqrt(taps + 1)`

**Taps made from the token.** `TokenTaps` holds static taps `[tap][channel]`
and a projection `[heads * taps][source]`. The projection of the source
vector of the current token gives `taps` numbers for each head. The tap for
channel `c` in head `h` is `static[tap][c] + projected[h][tap]`. Then:

`y = (sum over taps of tap * u(t - lag(tap))[c] + u(t)[c]) / sqrt(2 * taps + 1)`

The taps of a token come from that token only. They weight the window of
inputs that ends at that token.

`Taps` is one of the two kinds, for a caller that takes the kind from the
tensors of a layer.

## Layout facts in one place

Each layout fact is one constant or one small function. A wrong one is a
change of one line.

| Name | Meaning | Status |
| --- | --- | --- |
| `FIXED_TAP_ORDER` | Stored tap 0 of a fixed row weights the oldest token. The last tap weights the current token. | Inferred from source lines 293-299 and 314-322. Checked by a run, see below. |
| `TOKEN_TAP_ORDER` | Static row 0 and projected tap 0 weight the current token. | Stated in source line 410. Checked by a run. |
| `projected_tap` | The projection gives `[head][tap]`: index `head * taps + tap`. | From source line 431. Checked by a run. |
| `reciprocal`, `factor` | A division by a number is a product with its reciprocal. The reciprocal is `1 / sqrt(terms)` in f64, rounded to f32 one time. A convolution multiplies by it one time. | Source lines 392 and 454 write a division. The form is measured on the source's GPU run, see below. |
| `Slot` | State index of a visit: key 0, value 1, after attention 2, after the experts 3. | Source lines 562-567 and 868-873. |

The ignored test
[`sink_attention_trace.rs`](../../tests/sink_attention_trace.rs) checks the
first three. It uses the real weights of a public checkpoint and a layer
trace of the source code. Both stay outside this repository, in the ignored
folder `.local/capped-reference/`. Each of the three opposite layouts must
differ from the trace in at least 9 of 10 values. This file gives no
measured numbers, because their receipts are not in this repository.

## Number contract

All arithmetic is f32. Products and sums are separate operations.

1. The sum of one channel is one chain from +0.0. It adds the tap products
   from the oldest token to the current token.
2. The input of the current token is added last.
3. The result is multiplied one time by the f32 reciprocal of the square
   root of the term count.
4. The product is rounded to BF16 one time, nearest and ties to even.

The source writes step 3 as a division of a 32-bit tensor by a number. Its
GPU run computes that as a product with the reciprocal, where the
reciprocal is made in 64-bit and rounded to 32-bit one time. A measurement
on that run found this form in 4,000,000 of 4,000,000 values for each of 4
divisors, and a true division in fewer. The function `reciprocal` holds the
form. The position scale of [`sink_attention.rs`](../sink_attention.rs)
uses the same function. The receipt of the measurement is not in this
repository.

The source leaves the order of the taps to its library. The order above is
the order of the engine's device operation for the same convolution. Fixed
taps are f32 values. Static taps, the tap projection and its source vector
must be exact BF16 values: their product is a BF16 product, an f32 chain in
ascending index that is rounded one time. An input that is not finite, or a
result that is not finite in BF16, returns an error.

The BF16 round comes from the engine's shared math file,
[`fixed_math.rs`](../../../gpu/src/fixed_math.rs). The device operations use
the same file. This module calls no platform math library.

## State

`History` holds the last `taps - 1` inputs of one convolution, oldest first.
The inputs are the vectors before the convolution. A new history holds
zeros. `apply` computes one token and does not change the history. `step`
computes one token and then records its input. An error leaves the history
unchanged.

`VisitHistories` holds the four histories of one layer visit for one
sequence. A layer that runs more than one time for a token needs one set
for each visit. A convolution keeps no other state.

In a batch of prompts of unequal length, the source sets the inputs of
padding positions to zero. This reference handles one sequence and has no
padding.

## Proof and device work left

The unit tests use small shapes and hand-worked values: the tap order of
both kinds, the sequence start, the single rounding, the reciprocal form, the
chain order, a zero projection, four separate histories, and refusals that
change no state. The literal values come from an exact-fraction calculator
that rounds each f32 step one time. The calculator is not in this
repository.

Host tests do not establish GPU exactness or speed. A device operation must
give the same words as this reference before it is used. The key and value
convolutions of a layer are used by [`sink_attention.rs`](../sink_attention.rs).

**Back:** [Model source](../README.md)
