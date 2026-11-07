# GLM prompt rows

The bounded GLM decoder reads a prompt in large groups, up to 256 rows of one
chat. All rows finish one layer before the next layer starts, so one weight read
serves many rows. [`Decoder::prefill_rows`](../crates/tp/src/bounded_decoder/rows.rs)
is that path. It keeps every row's scalar arithmetic: the same folds, the same
rounding boundaries and the same order of additions.

## What runs together

- Projections take token rows `[tokens,cols]` through one weight pass. NVFP4 and
  FP8 keep one accumulator per (weight row, token). MXFP8 uses the existing M
  kernels, which keep one element set, FMA chain and shuffle tree at every M.
- Experts: rows choosing one expert share its pass. One launch covers all
  selected experts of a layer, with one gather for each of the middle and down
  outputs. Gate and up meet in SwiGLU on each rank, before that gather.
- mHC, norms, the router and the expert fold run each row in the same launch.

## What stays in token order

- KDA state: one thread owns each value column and carries it through the tokens.
- Convolution history: each token reads the rows before it.
- Latent attention: each token selects from its own causal prefix.

A pool depends only on its own rows, so the last token's pools hold each earlier
token's pools as a prefix. A latent row's expansion depends only on its bits, so
it is computed once and later tokens read it. These are pure functions, and the
results have the same bits.

## Decode table

`expert_group` reads weight bytes through a table of decoded E2M1 and E4M3 values.
This has no decode branch, so lanes with different weight rows do not diverge.
The table holds the same bits as `e2m1` and `e4m3`, and a test checks each byte.

## Which path runs

Every served prompt token runs the row path, in groups of one or more rows. Decode
runs `step`: on the current kernels it is faster than a one-row group at every length
measured. Both give the same bits, and `--decode-rows` still compares them.

The server uses groups of `PROMPT_GROUP` when no other chat waits, and
`PROMPT_SHARED` rows when one does. A chat holds up to `RESERVED_ROWS` expanded rows, all charged at admission. Past
that, each group expands only the rows its tokens selected, in memory that lasts for
one layer. The head runs only on the last row of a served
prompt. The checker can run it on every row to compare prompt logits.

## Exact drafting

One chat can decode several tokens in one pass. `flash-serve --mtp 1..3` loads the
checkpoint's own predictor layer. A request then asks for it with `"draft":"strict"`
and `mtp_depth`. With `"draft":"plain"` the same server decodes one token per step.

- **The proposer.** The predictor layer follows the last decoder layer
  ([`proposer.rs`](../crates/tp/src/bounded_decoder/proposer.rs)). It takes the last
  hidden state of a kept row and the token chosen after it, and proposes up to
  three further tokens. It keeps its own cache of the rows it has seen, starting
  with the last prompt rows. A proposal can be wrong. It never changes a kept token.
- **The verify pass.** The next token and the proposals run as one group of rows
  through `prefill_rows`, with the head on every row
  ([`verify.rs`](../crates/tp/src/bounded_decoder/verify.rs)). Each row has the bits
  of a plain step. A proposal is kept only when it equals the token the row before
  it chose, so the kept tokens are the tokens of plain decoding.
- **Rollback.** Rows after the first wrong proposal are dropped. The pass keeps the
  inputs of each recurrent layer's state kernels, and the caches are appended in
  row order. So the state after the kept rows is built from the pass itself
  (`Decoder::prefix_state`), with no second pass.
- **Held rows.** A decoder with a proposer holds the expanded rows of the tokens it
  decoded, up to `RESERVED_ROWS`, so no pass expands a row twice. The held rows of
  a layer sit in one allocation with room behind them. A pass writes its rows
  there when no live state reads past the held rows. Otherwise it copies them into
  a new allocation. Admission charges the request's tokens and the room.
- **Choosing.** The server's draft chooser decides the depth of each cycle from this
  request's measured step time, pass time and kept proposals
  ([`row_split.rs`](../crates/tp/src/row_split.rs)). When a pass does not pay, the
  chat decodes plain steps.

A pass serves one chat. With more chats, each decodes plain rows.

## Memory

`group_workspace_bytes` gives the largest amount one group can hold, from the layer
shapes: all rows' fixed buffers, attention at this context, one layer's selected
rows and the exchange. It is charged once, since one execution runs at a time.
`expanded_bytes` is charged for each chat.

## Proof

`resident-decode-check` saves all generated logits and a hash of every prompt
row's logits. `compare` checks them against a run of the scalar path.
`resident-decode-check prompt-rows` checks each row kernel against the host code.
