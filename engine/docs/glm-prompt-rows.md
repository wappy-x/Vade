# GLM prompt rows

The bounded GLM decoder reads a prompt in large groups, up to 1024 rows of one
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

## Work behind a message

On two ranks a large group sends each rank's half of a result over the host link, and the
GPU runs nothing while the link carries it. For the routed gate and up of a group of 64
rows or more, the decoder queues the next work that does not read that result before the
link step starts: the shared expert's SwiGLU and this rank's rows of its down projection
(`Decoder::rows_ahead`, `Weights::apply_experts_open` and `apply_experts_close`). The
launches and their inputs are the same, so every row keeps its bits. The served counter is
`prompt_link_ahead`.

## What stays in token order

- KDA state: one thread owns each value column and carries it through the tokens.
- Convolution history: each token reads the rows before it.
- Latent attention: each token selects from its own causal prefix.

A pool depends only on its own rows, so the last token's pools hold each earlier
token's pools as a prefix. A latent row's expansion depends only on its bits, so
it is computed once and later tokens read it. These are pure functions, and the
results have the same bits.

A group that follows earlier rows folds only the pools it completes. Their rows
are one window that starts at a pool boundary: one fold for the window in a group
past 16 rows (`rows/pools.rs append_window`, counter `pool_window`), one fold a
pool in a smaller pass. Earlier pools are kept from the state. The first group of
a prompt folds its own rows.

## Decode table

`expert_group` reads weight bytes through a table of decoded E2M1 and E4M3 values.
This has no decode branch, so lanes with different weight rows do not diverge.
The table holds the same bits as `e2m1` and `e4m3`, and a test checks each byte.

## Which path runs

Every served prompt token runs the row path, in groups of one or more rows. Decode
runs `step`: on the current kernels it is faster than a one-row group at every length
measured. Both give the same bits, and `--decode-rows` still compares them.

When no other chat waits, the server uses the largest group its memory budget
funds: 1024, 512 or 256 rows. The ranks agree on it at startup
(`Factory::negotiate_prompt_group`), and startup prints it as `TEXT PROMPT GROUP`.
When another chat waits, the server uses `PROMPT_SHARED` rows.

A group has a fixed cost. It reads every expert that one of its rows routes to,
and from about 100 rows up that is most of the 288 experts of each routed layer.
It also pays each exchange and each read-back of a layer once. A larger group
pays this fixed cost for more rows. The bits of a row do not depend on the group.

 A chat holds up to `RESERVED_ROWS` expanded rows, all charged at admission. Past
that, each group expands only the rows its tokens selected, in memory that lasts for
one layer. The head runs only on the last row of a served
prompt. The checker can run it on every row to compare prompt logits.

## Exact drafting

One chat can decode several tokens in one pass. `flash-serve --mtp 1..7` loads the
checkpoint's own predictor layer for proposals up to that depth. A request then asks for it with `"draft":"strict"`
and `mtp_depth`. With `"draft":"plain"` the same server decodes one token per step.

- **The proposer.** The predictor layer follows the last decoder layer
  ([`proposer.rs`](../crates/tp/src/bounded_decoder/proposer.rs)). It takes the last
  hidden state of a kept row and the token chosen after it, and proposes up to
  seven further tokens (`PREDICTOR_DEPTH_MAX`), never more than the loaded depth. It keeps its own cache of the rows it has seen, starting
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
- **Deeper than three.** A request whose depth is above 3 starts each answer at
  depth 3. It goes to depth 5, and then to its whole depth, only while those
  proposals are kept, and it comes back when they are lost
  ([`deeper.rs`](../crates/tp/src/row_split/deeper.rs)). The rule counts kept
  proposals and holds no time, so the same request proposes the same depths every
  time. A request that names no draft mode leaves the depth to the service, up to
  the loaded depth. A request of depth 3 or less, and a server started with
  `--mtp 3` or less, propose as before.

A pass serves one chat, or two to three chats that draft together. Two to three strict
requests of depth 2 or more that are alone on the server each keep their own proposer. Their check rows run
in ONE shared pass in a scheduler tick
([`rows/shared.rs`](../crates/tp/src/bounded_decoder/rows/shared.rs), command 32 in
[`row_split/shared.rs`](../crates/tp/src/row_split/shared.rs)). The steps of a layer
that read weights only run once on the rows of all the chats. The steps that read a
chat's history run once for each chat, on that chat's rows, with its own positions
and caches. Each chat has its own verdict and its own rollback: one chat may keep
four tokens while the other keeps one. A row in a shared pass has the bits of the
same row in that chat's own pass. A pass holds 16 rows at most, kept rows included.
The pair decides for both chats
([`pair.rs`](../crates/tp/src/row_split/pair.rs)): both propose their whole depth in
a tick, or both ride the step of two chats. A pair starts with shared plain steps.
Each chat runs the pair trial: lookback probes that start eight steps into its
answer, five steps apart. It does not wait for the think block to end. Seven matched
tokens within four probes start a chat. The steps retain owned predictor inputs for
each chat. Both trials must pass before shared passes start. A failed trial keeps
both chats on shared plain steps, and each chat turns its predictor off when it
hands back to plain steps (command 31, word 2: it keeps the rows its prompt expanded,
because its drafting invoice stays). An output cap under 32 tokens cannot pay the probes back
and uses ordinary shared decoding without pair trial state. This cap guard does not
change a lone strict chat.

Once shared passes start, the pair compares each chat's kept tokens with the measured cost
of shared plain steps. If either chat stops paying, both leave the shared passes for the
rest of the pair. The served log records probe time with `SERVED PAIR PROBE`, trial
verdicts with `SERVED PAIR TRIAL`, each shared pass with `SERVED SHARED`, and one `SERVED PAIR` account for each pair. Three strict chats follow the same rule
as two: every chat's trial must pass, all propose in a tick or none does, and the chat that keeps the fewest rows
must still gain against the step of all of them. A chat that arrives starts a new trial for all. When a chat ends,
the chats that are left go on. With four or more chats, and for a request that
names no draft mode beside another chat, each chat decodes plain rows.

## Memory

`group_workspace_bytes` gives the largest amount one group can hold, from the layer
shapes: all rows' fixed buffers, attention at this context, one layer's selected
rows and the exchange. It is charged once, since one execution runs at a time.
The service charges it for the group size it negotiated. Past 256 rows it also
charges the recurrent state copies that can wait behind their sources
(`service_memory::workspace_for_prompt_group`).
`expanded_bytes` is charged for each chat.

## Proof

`resident-decode-check` saves all generated logits and a hash of every prompt
row's logits. `compare` checks them against a run of the scalar path.
`resident-decode-check prompt-rows` checks each row kernel against the host code.
