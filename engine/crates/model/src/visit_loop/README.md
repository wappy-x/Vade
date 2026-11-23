# Host visit loop reference

[`visit_loop.rs`](../visit_loop.rs) is the host form of the visit driver of a
looped decoder. It holds the visit list, the state that each pass starts
from, the messages from the pass before and the gates that mix them in. It
does not enable a model plan and it does not change the serving path.

| File | What it does |
| --- | --- |
| [`tests.rs`](tests.rs) | Unit tests for the visit list, the messages, the gates and the driver. |

## Words

- **Layer:** one set of stored weights.
- **Visit:** one run of one layer for one token.
- **Pass:** one run of the loop layers. Each pass uses the same weights.
- **Block:** a group of `n_layers_per_attn_res_block` layers. The first layer
  of a block is a boundary.
- **Stream:** the vector that goes from one visit to the next.
- **Block list:** the finished block vectors of the token, oldest first.

## The visit list is data

`Config` reads these fields from the text config. Each field is required.

| Field | Meaning |
| --- | --- |
| `hidden_size` | Width of the stream. |
| `num_hidden_layers` | Count of stored layers. |
| `n_layers_per_attn_res_block` | Layers in one block. |
| `loop_config.core_start`, `core_end` | Stored layers `start..end` are the loop. |
| `loop_config.num_passes` | How many times the loop layers run. |
| `loop_config.gate_rank` | Inner width of a message gate. |
| `rms_norm_eps` | Epsilon of the plain norm inside a gate. |

`loop_config.is_varloop` must be false or absent. A model name does not select
behavior. `Schedule::new` makes the list from these numbers. No list is fixed
in the code.

Each `Visit` says what the visit runs, reads and writes:

| Field | Meaning |
| --- | --- |
| `layer` | Stored layer. It selects the weights. |
| `pass` | Pass of the loop, from 1. A layer outside the loop has pass 1. |
| `slot` | State slot for the keys, values and convolution tails of the visit. |
| `boundary` | The input of the visit joins the block list. |
| `stream` | The vector that the visit reads as its stream. |
| `blocks` | The block list before the visit. |
| `messages` | From pass 2: the loop layer number and the vectors that the messages read. |

A vector is named by a `Source`: the normed embedding, or the stream after an
earlier visit. So the list has no hidden state.

Rules of the list:

1. Layers before the loop run once.
2. Pass 1 goes on from them and has no messages.
3. Each later pass restarts from the stream and the block list that pass 1
   started from. It does not start from the end of the pass before.
4. The messages of a pass read the block outputs of the pass before: the
   block list entries that the pass made after its first one, then the stream
   that it ended with.
5. Layers after the loop go on from the last pass. So do the end mix inputs.
6. Slots: pass 1 uses the layer number. Pass `p` uses
   `layers + (p - 2) * loop layers + (layer - loop start)`. Each visit has its
   own slot.

Example from the test
`twenty_four_layers_with_three_passes_over_twelve_give_48_visits`: 24 layers,
blocks of 6, the loop on layers 6 to 17 and 3 passes give 48 visits. Visits 18
and 30 read the stream after visit 5. Pass 2 reads its messages from the
streams after visits 11 and 17. Pass 3 reads them from visits 23 and 29. The
end mix reads the embedding and the streams after visits 5, 35, 41 and 47.
`Schedule::peak_held` is 6 for that config: one token holds 6 vectors at most.

## One visit

`Layer` holds the stream arithmetic of one stored layer. The caller supplies
attention and the experts through the `Body` trait.

1. `Layer::attention_inputs`: the attention mix over the block list and the
   stream. One query gives one result for the query, key and value inputs.
   Three queries give one result each. The first stored layer has no mix: it
   reads the stream. With messages, each input then goes through its own gate.
2. `Body::attention`: the norms, attention and the convolution after it.
3. `Layer::expert_input`: in a boundary layer the stream joins the block list
   and the residual is the attention output alone. In other layers the
   residual is the stream plus the attention output. Then the expert mix over
   the block list and the residual, and the expert gate when there are messages.
4. `Body::experts`: the norm, the router, the experts and the convolution
   after them.
5. `visit_output`: the residual plus the expert output, rounded to BF16 once.

With no messages the gates are skipped. They are not fed zeros.

## Messages and gates

`messages` makes the four messages of one loop layer in the order query, key,
value, expert. A message is a [block mix](../block_mix/README.md) over the
block outputs of the pass before, with the message queries of that layer.

`Gate::apply` mixes a message into a value:

1. Plain norm of the value and of the message, with `rms_norm_eps`. Round
   each number to BF16. Put the two side by side.
2. Inner vector: for each row of the down weight, add `weight * input` in f32
   as a chain in rising input order from zero. Round the sum to BF16 once.
3. `forget = tanh(forget weight * inner)` and `gain = tanh(gain weight * inner)`,
   each an f32 dot product as a chain in rising order from zero.
4. Result: `(1 - forget) * value + gain * message`, in f32: two products,
   then one sum.

The sum of squares of the norms in step 1 has the order of the engine's
weighted norm kernel, which a device path runs here with no weight (`norm` in
[`causal_layer.rs`](../../../tp/src/causal_layer.rs)): chunks of 128 values
with 32 lanes each, the lanes joined, then the chunk sums in 32 lanes, joined
again. The test `the_gate_norm_sums_squares_in_chunks_of_128` pins it. The
plain norm of a mix and of a message has another order: the 32-lane tree of
the [block mix](../block_mix/README.md).

A product and a sum are separate f32 operations, never a fused multiply-add.
Scalar math (`tanh` and the BF16 round) is the shared file
[`fixed_math.rs`](../../../gpu/src/fixed_math.rs) and nothing else. The device
code builds the same source.

## Number types

| Value | Type |
| --- | --- |
| Stream after a visit, block list entries | BF16 values, held as f32 |
| Mix results, messages, gate results | f32 |
| Residual inside a visit | f32, not rounded |
| Gate inner input and inner vector | BF16 |
| Mix queries, message queries, forget and gain weights | f32 |
| Down weight of a gate | BF16 |

`Gate::new` refuses a down weight that does not hold BF16 values.

## Stored layout

These facts are for the code that loads the weights. This module takes the
logical layout.

- Message queries: the checkpoint stores two tensors of
  `[2 * loop layers][hidden]`. `merge_message_queries` puts them in the order
  `[loop layers][4][hidden]`.
- Gate weights: down is `[gate width][2 * hidden]`. Forget and gain are
  `[hidden][gate width]`.
- Attention queries: one row or three rows of `[hidden]`. Three rows are in
  the order query, key, value.

## Source

Transformers v5.24.0, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`, under
`src/transformers/`:

- `models/inkling2_loop/configuration_inkling2_loop.py`, lines 139-176: the
  loop fields and their checks.
- `models/inkling2_loop/modeling_inkling2_loop.py`: lines 79-96, the gate.
  Lines 145-173, the messages. Lines 876-910, 937-997, 1022-1062 and
  1091-1149, the place of the mixes and gates in a visit. Lines 1192-1203,
  the parts that stay f32. Lines 1350-1443, the visit order and the restart.
- `core_model_loading.py`, lines 823-849: the stored message query layout.

This module holds its own code and none of that source. The source does not
give the order of the additions inside its BF16 product. The orders of the
gate are the engine's. They do not claim bit parity with the source library.

## Proof and remaining device work

The [unit tests](tests.rs) check the visit list of two small configs and of
the 48-visit config, the config reader and its refusals, the vectors that the
driver hands to each visit, hand values for the messages and the gates, a
zero gate, both BF16 points of a gate, the sum order of the gate norm, the
first pass, passes 2 and 3, the block edge, and the driver against the same
visits written out one by one.

The ignored test [`visit_loop_trace.rs`](../../tests/visit_loop_trace.rs)
checks the visit list, the end mix and the expert input of each visit against
a layer trace of the reference code, with the stored weights. It needs the
model folder and the trace under `.local/visit-loop-trace/`, so it does not
run by default. Its expert input check is approximate: the trace does not
hold the residual of a visit, and the test makes it from rounded values.

Open points:

- No device code exists for the mixes, the messages or the gates.
- The typed model description does not carry a visit list yet.

Host tests do not establish GPU exactness or speed.

**Back:** [Model source](../README.md)
