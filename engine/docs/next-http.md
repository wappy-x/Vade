# Flash-Next NVFP4 serving

Start at [INSTALL.md](../../INSTALL.md) for system and compiler prerequisites.
This page describes the native **FP4-Local** checkpoint route. The FP8
`vade up` deployment has different defaults.

## One Spark and two Sparks

`flash-serve --fp4-local` selects the native model. With one rank, a loopback
launch from a built checkout has this shape; set `PTX` and `MODEL` to the
verified artifact and checkpoint paths:

```sh
flash-serve 0 1 127.0.0.1:0 127.0.0.1:0 "$PTX" "$MODEL" 127.0.0.1:8080 0 --fp4-local
```

For two ranks, FP4-Local serves the [tensor split](#two-spark-tensor-split)
over the layer pipeline's links. It is the only two-Spark path. Launch each rank with its own control address and its peer's address, the same
fresh `TP_RUN_ID`, and the same `SEQPAR_CONTROL_SECRET`. Non-loopback control
requires that secret. Do not supply `--tensor-map` for this route. Three or
more ranks run the layer pipeline as a chain; the engine chooses each path itself.

Current defaults are defined by the
[launcher](../crates/tp/src/bin/flash-serve.rs),
[drafting settings](../crates/engine/src/flash/draft_stack.rs),
[cohort policy](../crates/serve/src/mtp_batch.rs) and
[prompt scheduling](../crates/engine/src/flash/next_rows.rs):

- Context defaults to 32768 and output is capped at 16384, subject to admission.
- Startup/request MTP depth defaults to 3. The resident draft ceiling is 7 on
  one Spark and on the pipeline. The adaptive chooser is enabled by default;
  request depth and resident ceiling are distinct. `--mtp 0` selects plain decoding.
- The launcher selects `Auto8` for one Spark, the two-Spark tensor split and
  four Sparks (the split in parts or the layer chain). This mode allows greedy drafting with one through
  eight active chats. Larger cohorts use pooled plain steps. The cost controller
  can also choose plain steps. With `VADE_MTP_C8=0`, the automatic table allows
  drafting through five chats. The three-Spark chain uses that table by default.
  `VADE_MTP_BATCH` can override the automatic table.
- The two-Spark tensor split and four Sparks (the split in parts or the layer
  chain) use a larger shared
  token-checking pass by default. Five through eight chats can share a pass of
  up to 32 rows.
  `VADE_MTP_COHORT=off` restores the 16-row controller unless
  `VADE_VERIFY_ROWS` overrides the row budget. One Spark defaults to 16 rows;
  `VADE_MTP_COHORT=wide` enables the larger shared pass.
- Local prompt tiles are automatic: shorter prompts use 4096 rows, while
  prompts ending at or beyond 8192 tokens use 8192 rows. The layer chain defaults
  to 2048-row tiles. The tensor split funds a 4096-row prompt workspace, as set
  by the [startup settings](../crates/tp/src/flash/settings.rs).
  `VADE_PROMPT_TILE` can select a fixed supported tile size.
- Prompt lookup is controlled separately by `VADE_MTP_LOOKUP`; startup with
  MTP disabled also disables lookup.
- On the pipeline, each rank allocates chat state (GDN and attention memory)
  for its own layers only, and funds every chat at the largest stage's share,
  so all ranks admit the same number of chats.

These are source-defined defaults, not performance measurements. Verify the
loaded checkpoint, effective settings and health identity before timing.
The [checkpoint and kernel reference](fp4-kernels.md) describes storage.

## Conversation memory reuse

Freed attention arenas and device vectors can serve later requests. Each
reserve keeps at most the larger of 2,000,000,000 bytes and the largest
conversation reservation times the largest observed cohort, with a minimum
cohort of four. This changes reuse capacity, not the startup session limit.
Memory pressure and the device budget can release reserve memory sooner.

A session can free ordinary buffers before it releases its attention arenas.
If the cached pressure reading refused reuse and accounted device memory has
fallen since that reading, the return path reads host memory again. It keeps
the same safety margin. A failed read keeps the refusal in effect.

Flash-Next serving also gives both reserves its session plan: the admitted
session limit. With every admitted session live, host memory can fall below
the margin while the engine stays inside its own budget. A freed buffer then
returns to the reserve, not to the device allocator, and the next session uses
it again without a new allocation. This exception applies to an ordinary buffer
only. It applies only while the buffers of that kind that the process holds,
live or spare, are at most the session limit times the largest session
reservation. Under pressure, a reserve still releases ordinary spares above
that bound, and segmented pages outside their own plan (see below). After 30 seconds without a request, each reserve
releases its spares when it reads pressure. Each reserve reads pressure
separately. If the first release clears the pressure, the second reserve
keeps its spares. Without pressure, spares inside the plan stay. A new request
ends the idle period before its admission can allocate memory or read
pressure. A worker Spark has no periodic idle check. It reads pressure when it
allocates. It also reads pressure when it returns a buffer after device bytes
fell under pressure, if the buffers held are outside the plan. It keeps spares
inside the plan.

Flash-Next serving admits a request by the KV pages it can use: its prompt,
its output allowance, the draft verify rows and 64 more rows, in whole 2 MiB
pages. Admission charges these bytes to the session pool when the request
starts. The engine maps the pages when the rows need them. Segmented KV pages
have their own plan: the session pool bytes. Under pressure, the KV reserve
keeps spare segmented pages while the segmented pages that the process holds,
live or spare, are at most that pool. Above it, pressure releases them. If the
device budget or the driver refuses a page during later growth, the growth
stops with a capacity error (`KV segment growth` in the log).

The `flashnext kv_arenas` log gives per-operation counts and bytes for ordinary attention
arena allocations, reuse, reserve returns, direct frees and reserve evictions.
Fresh session construction emits an `admit` line before `request_admitted`.
Cached-state restoration has no `admit` arena line. Its events still contribute
to the cumulative `kv_reserve` counters. A `retire` line precedes
`request_retired` for a normally completed live session. Direct frees and
reserve evictions have separate counters; neither means an arena is retained
by a live owner. These counters do not include segmented KV pages.
`request_admitted` also gives `admitted_at_ms`, measured from one monotonic
clock at the start of the serving loop. Subtract these timestamps to measure
the interval between admissions. Queue wait times have different start times.
A queue wait starts when the serving loop accepts the connection.

## Burst intake

A client that sends several chats at once connects them within a few
milliseconds. The request bytes of a chat can come later than its
connection. The first prompt pass waits for the rest of such a burst only
when no chat decodes and every job of the server is a new prompt that can
join a shared prompt pass. Such a prompt is not longer than the shared
prefill maximum (`VADE_BATCH_PREFILL_MAX_TOKENS` on Flash-Next, 512 tokens by
default), and it is not longer than the shared prefill limit or the engine
splits prompts into tiles. Alone, such a prompt can start on the serial
prompt path, and the first chat of a burst is often admitted alone, so the
wait holds the serial path too. A longer prompt, such as a cold 8K prompt on
Flash-Next, is not held.

The pass waits only while two or more new prompts wait, or while another
request is still arriving: a connection that has not sent its whole request,
a request in preparation or a request that waits for admission. An open
connection that has sent no bytes counts only until it is one window old, so
an idle preconnect does not hold every lone request. The pass does not wait
when no session slot is free. A lone request with nothing else arriving does
not wait. Token IDs do not change: a prompt gives the same IDs alone and in a
batch.

The window bounds the hold only: the hold ends when the oldest waiting
chat's connection is 10 ms old (the default). The first pass can wait longer
for the waits that apply without the hold: up to 100 ms after admission while
requests are in preparation, one loop turn when a connection has unread
request bytes, the 2 ms prompt age, and the admission of each chat that the
server has read. While only the hold keeps the pass back, the serving loop
sleeps for up to 1 ms per turn, and it does not sleep past the window.

`VADE_INTAKE_BURST_MS` sets the window in whole milliseconds, from 0 to 100.
0 starts the first pass as soon as a prompt is ready, as before. Other values
stop the server at startup. Each first prompt pass of an idle server logs one
`intake_burst` line: `chats` in the pass, `waited_ms` since the oldest chat's
connection, `held_ms` that the wait added, and `window_ms`. A late chat that
missed the pass shows as a `request_admitted` line with a long `queue_ms`
after an `intake_burst` line with fewer chats.

<a id="two-spark-tensor-split-opt-in"></a><a id="one-spark-and-the-two-spark-pipeline"></a>

## Two-Spark tensor split

World 2 FP4-Local serves a tensor split of the
[layer stack](../crates/engine/src/flash/next_tp.rs). It is the only two-Spark
path, and every proven level is the only code: no environment name selects or
tunes a level. Start both ranks with the same `TP_RUN_ID`
and `SEQPAR_CONTROL_SECRET`, each rank's own control address and its peer's:

```sh
TP_RUN_ID="$RUN" SEQPAR_CONTROL_SECRET="$SECRET" \
  flash-serve 0 2 "$RANK0_ADDR" "$RANK1_ADDR" "$PTX" "$MODEL" 127.0.0.1:8080 0 --fp4-local
TP_RUN_ID="$RUN" SEQPAR_CONTROL_SECRET="$SECRET" \
  flash-serve 1 2 "$RANK1_ADDR" "$RANK0_ADDR" "$PTX" "$MODEL" 127.0.0.1:8080 0 --fp4-local
```

The split needs RDMA. At world 2, flash-serve refuses `SEQPAR_RDMA=0` and
`--tensor-map`, and `vade up --sparks 2` refuses TCP. The serving receipt names
the path and why under `vade.numerics.path` and `vade.numerics.path_reason`.

It keeps the pipeline's links, recipe and session protocol, with rank 1 starting at
layer 2:

- Rank 0 runs the world-one engine on every layer, plus the head, sampling and
  drafting. Rank 1 mirrors every main-model pass from layer 2 on.
- On verify passes of at most 32 rows (the served 8-chat cohort tick verifies
  32 rows in one pass) and on other passes (decode and short prompt tiles) of
  at most 16 rows (wider prompt passes run the prompt expert kernels), both
  ranks split every routed expert by output columns with the unchanged NVFP4
  expert kernels: each rank computes half of every expert's gate/up columns and
  half of its down columns, at widths taken from the checkpoint configuration.
  The ranks swap the BF16 middle halves and then the finished MoE output halves
  as raw bits. No sum is split, and the router, routes and shared expert run on
  both ranks from the same bits, so every output element is world one's. Token
  IDs are identical to world one and to the pipeline.
- Before serving starts, the tensor split exchanges each rank's memory-admitted
  session count over the control link. Both ranks use the smaller count. The
  startup log records `rank0`, `rank1` and `chosen` counts in one
  `flashnext tp session_pool` line. Different free memory does not cause a
  refusal if both ranks can fund at least one session.
  Each rank sets a 600-second load deadline before it loads the model. The
  count exchange uses that deadline, including partial frames. It checks the
  stop flag while it waits. The short serving timeout does not end this wait
  while a healthy peer still loads.
- The exchanges stay inside the CUDA stream: a kernel writes this rank's half
  into a mapped mailbox slot and publishes its sequence word; the peer's kernel
  waits for the word (with a sleep backoff, bounded by the 10-second link timeout;
  a timeout fails the pass on both ranks) and copies the rows in. Split verify
  passes therefore keep their captured graphs. Rank 0's residual after layer 1
  reaches rank 1 the same way; wider passes send it with the request.
- The split also cuts the projections on the split layers (from layer 2). On
  passes of one chat at most 16 rows wide, each GDN layer's qkv and z
  in-projections, the out projection of every GDN and attention layer, and the
  attention layers' q/gate projection (by query-head halves; index, k and v
  stay whole) run by output rows: each rank runs its half of the rows with the
  unchanged row kernels (and the matching rows of the MXFP8 twin), and the
  halves cross as raw bits. On every split pass, the shared expert's down
  projection runs by output rows too (no extra message: each rank's MoE finish
  reads only its own columns of it). The tokens are the same.
- On passes of one chat (at most 16 rows), the split can run each GDN layer by
  heads instead of its in-projection halves: each rank projects the rows of its
  own heads, runs the conv, recurrence and output gate on a state that holds
  only its heads (in the same storage), and the ranks swap the gated rows (one
  message) before the split out projection. Before any other pass (several
  chats, a prompt wider than the mailbox, a pipeline tile) both ranks rebuild
  the whole state over the split's TCP link, so rank 0 switches a chat only
  when that pays: it times one rebuild once (a probe exchange of one chat's
  halves) and the passes run whole and by heads, and switches a chat once the
  time it would have saved since its last pass of another shape reaches one
  rebuild (the first chat switches at once, to measure). Rank 1 follows rank
  0's choice, sent with each pass. The tokens are the same.
- One-sided RDMA writes move each message: when the kernel publishes
  a message, a doorbell thread on its rank posts two chained writes on one queue
  pair, the published bytes straight into the peer's receive slot and then the
  sequence word the peer's kernel waits for. There are no host copies and no
  receive-side host work. The split's lockstep keeps a slot's
  payload from being rewritten while it is in use; the doorbell also posts a
  slot again only after the write completion of the message that used it
  before. A failure marks this rank's mailbox failed and, best effort, the
  peer's; the peer's takes otherwise stop at their timeout. A wire that cannot
  start stops both ranks before rank 0 serves. The takes poll with a 128 ns pause cap,
  the kernels copy four words at a time and a take uses at most 6 CTAs. While a
  pass runs the doorbell spins at most 1 millisecond between yields of its core.
  The doorbell blocks whenever every message the host has queued is posted and
  acknowledged (between passes, between requests, and while rank 1 idles); the
  next queued message wakes it. Only while a queued message is not yet published
  does it poll as above. A stop also ends the wire's startup key swap at once,
  even when the peer never answers. No environment name tunes the wire.
- The split splits the
  verify head of one chat's verify pass of 1 to 8 tokens by vocabulary halves.
  Both ranks hold the pass's final residual, mix it and project their half of
  the head rows with the same launches as the whole head, so every logit is the
  whole head's. When the bounded head is funded, each rank loads the screen and
  row norms of its own half only (rank 1 then holds about 0.33 GB more and rank 0
  as much less) and rescores exactly the rows of its half that can win. Rank 1 sends its best logit and token per row (12 bytes per
  row) through the mailbox; rank 0 keeps the larger logit, ties to the lower
  token ID, as the whole head's argmax does. Only greedy verification uses the
  halves: a sampled pass runs the whole head on rank 0 and drops rank 1's
  message. The verifier then holds no full logits; a request for logits or
  log-probabilities projects the whole head again from the kept inputs. Token
  IDs do not change. Drafts keep the whole draft head on rank 0.
- The plain greedy decode step (drafting OFF) splits its head the same way.
  Rank 0 asks for the split in the step's request to rank 1 when the step
  reads only the token IDs and its rows cross in the mailbox (1 to 16 chats).
  Each rank projects its half for each chat's last row with the same launches
  as the whole head. Rank 1 sends each chat's best logit and token (12 bytes
  per chat), and rank 0 keeps the larger logit, ties to the lower token ID. A
  sampled step, a prompt pass and a dump run the whole head on rank 0 as
  before, so they keep full logits. Each chat keeps its head input of the
  split step. A later request for that step's logits (for example when the
  chosen token is not in the tokenizer) projects the whole head again from it.
  Token IDs do not change.
- A prompt pass whose every chat spans at least half of the 2048-row pipeline tile
  (1024 rows) runs as a pipeline tile (rank 0 layers 0-23, rank 1 layers 24-47,
  exactly as the pipeline). Two Sparks use a 4096-row prompt workspace for the
  token split below, but the serve still hands other prompts to the engine in
  2048-row tiles (wider in the middle of a long prompt, below). The rest of
  such a prompt (its later, shorter passes up to the first token) also runs as
  pipeline tiles. Before the chat's next split pass, the ranks swap the session
  state the other did not compute (GDN state and history, QSA keys, values and
  index rows) as raw words, GPU to GPU through the split's mailbox in pieces of
  one slot; the TCP link carries only each layer's row count and vector sizes,
  which both ranks check. Shorter prompts run on both ranks. A failed swap stops
  the link and leaves the chats' sessions unusable (they are not reused).
- In such a prompt of eight prompt workspaces or more (32768 rows by default),
  a pass after a chat's first one takes the whole workspace (4096 rows by
  default) while more than the workspace plus 2048 rows remain; a new chat's
  first pass and the last rows of the prompt keep 2048-row tiles, so the
  pipeline fills and drains in short tiles. Every cut except the last stays a
  multiple of 64. The routed experts get twice the rows per expert in a wider
  pass, and each row and layer computes the same words, so token IDs do not
  change. A shorter prompt, or a workspace of 2048 rows or less, keeps 2048-row
  tiles: the first token comes after one rank's time for all rows plus its time
  for the largest tile, and a wider tile pays for itself only in a long prompt.
- A cold one-chat prompt of 2048 to 8192 rows (two prompt workspaces) runs as a
  token split instead ([`next_tp_seq.rs`](../crates/engine/src/flash/next_tp_seq.rs)).
  The serve hands the whole prompt in one call, and the engine runs it as two
  parts. First part, rows 0..P (P is half the prompt, rounded up to 64): rank 0
  runs layers 0-1 and posts the rows as a pipeline tile; rank 1 runs layers 2-47
  and, after each layer, puts that layer's state at row P in the mailbox. Second
  part, rows P..N: rank 0 runs every layer and, before each layer from 2, takes
  rank 1's state of that layer. These are the words a prefix-cache resume at P
  installs, so token IDs do not change. A tail under 64 rows also runs on rank 0.
  Before the chat's next pass, rank 0 sends rank 1 layers 2-47 one way, and
  decode runs on the split. If a different pass comes before the second part,
  rank 0 first takes and drops the rest of rank 1's state, so the link does not
  stop; that chat's later passes then fail with an error.
- A cold one-chat prompt past two prompt workspaces, of at least four 2048-row
  tiles, whose rows up to its last 64-row edge number at most 20416 (prompts up
  to 20480 tokens), runs as a layer-major transaction
  ([`next_split.rs`](../crates/engine/src/flash/next_split.rs)), with a prompt
  workspace of 2048 rows or more. The cut is half the rows, on a 64-row edge.
  Rank 0 runs layers 0-1 of every row and sends the rows before the cut to rank
  1 tile by tile; rank 1 runs layers 2-47 of those rows and, after each layer,
  puts that layer's state at the cut in the mailbox; rank 0 takes each layer's
  state before it runs that layer for the rows after the cut. Each row and layer
  computes the same words, so token IDs do not change. Before either starts,
  each rank maps the KV pages of every row it writes (rank 1 the rows before the
  cut, rank 0 every row, its drafter included) and funds its extra rows; if a
  rank cannot, the prompt keeps the pipeline tiles. The tail after the last 64-row edge (1 to 64 rows) runs as after the
  token split. `VADE_LONG_SPLIT=0` turns the transaction off;
  `VADE_LONG_SPLIT_CUT=<permille>` (1 to 999) places the cut at that share of
  the rows (a measurement setting). Longer prompts keep the pipeline tiles.
- Plain (drafting-off) steps use the session path instead of pooled decode.
- The serving metadata (`vade.numerics`) of two or more ranks names the path the
  engine chose in `"path"` and why in `"path_reason"`: `"tensor_split"` (FP4-Local
  on two Sparks, or on more Sparks over RDMA whose widths cut into whole parts and
  whose mailbox has room: four for the shipped checkpoint),
  `"layer_pipeline"` (FP4-Local on three or more without the split),
  `"rank_path"` (FP4-Local with `--tensor-map`) or `"ring"` (FP8: the tensor split
  does not serve FP8). The split adds `"tensor_split"` with its recipe:
  `"experts"` (`"output_columns"`), `"long_prompts"`, `"projection_level"`,
  `"wire"` and `"head"`. On two Sparks these are
  `"pipeline_tiles_then_state_handoff"`, `3`, `"write"` and `"vocabulary_halves"`;
  on more Sparks `"chain_tiles_then_state_gather"`, `2` (the linear-attention
  heads stay whole), `"relay"` and `"vocabulary_parts"`.

## Native rank backend

The following world-four backend is separate from the pipeline service. It no
longer serves two ranks: world 2 runs the tensor split, and `flash-serve`
refuses `--tensor-map` there. Its retained qualification is incomplete; do not
generalize the pipeline's behavior to it. Worlds expose the same lowercase
checkpoint-folder model ID, with world and topology as separate metadata.

`flash-serve` owns startup, authentication, checkpoint selection, request parsing,
templates, SSE and prompt/output admission for every world. Mixed FP4-Local
metadata selects world-one rows/MTP, the tensor split at world 2, or the world-4 native rank backend.
The rank-local arena, session schedule, scalar C1 and C2..C8 cohorts are retained
from the pinned world path. `next-tp` remains the native capture command.

Every rank needs the same fresh `TP_RUN_ID`, authenticated TP control secret and
production binary/PTX. Use the existing RDMA transport and `TP_TOPOLOGY=ring`
for the W2 or W4 ring. Set topology explicitly; W4 otherwise inherits the existing
path default. Invocation:

```text
flash-serve RANK WORLD LISTEN NEXT PTX MODEL 32768 BIND REQUESTS --tensor-map TENSOR_MAP --mtp 0 [--allow-remote] [--api-key-file PATH]
```

Only rank zero binds HTTP. `REQUESTS=0` serves continuously. World-2/4 context is
fixed at 32768; `FLASH_SERVE_OUTPUT_TOKENS` defaults to 16384 and accepts
1..=16384. Remote HTTP requires `--allow-remote` and an API key. Checkpoint,
build, PTX, world and topology metadata use the same loaders as world one.
Wait for checkpoint hashing to become ready before timing.

The native rank backend defaults to plain decoding and refuses positive
startup MTP depth. Its request `mtp_depth` must be zero. The one-Spark and
pipeline routes accept request depths 0..3, bounded by the resident model;
the adaptive policy described above may propose more tokens for eligible
requests. Invalid field types and unsupported depths fail before admission.
Other engines ignore this Next-only field in the translated API paths.

`next-http` is a compatibility launcher that execs the sibling `flash-serve`;
it has no request parser or serve loop. Its original positional form and
`NEXT_TP_HTTP=1` requirement remain; `OUTPUT` maps to
`FLASH_SERVE_OUTPUT_TOKENS`. It supplies context 32768, `--tensor-map` and
`--mtp 0`. Install both binaries together. Worlds 2/4 retain the pinned native
greedy defaults even when checkpoint metadata requests sampling. World one and
FP8 retain their checkpoint generation defaults. Qualification requests send
all greedy fields explicitly. This greedy-only rule binds this native rank
backend only. One Spark, the two-Spark tensor split, the split in parts and the chain
sample on rank 0 ([Sampling](../../docs/sampling.md)).

Worlds 2/4 refuse `--verify-full` and `FLASH_VERIFY_FULL=1` before rank startup:
the native shard loader does not implement full payload verification. Use world
one when full verification is required. Normal metadata/header verification and
rank artifact agreement remain enabled.

The shared OpenAI-compatible HTTP/JSON/SSE scheduler admits eight full-context
sessions, queues in FIFO order, and rotates one causal token per turn. Prompt
work also yields after one token. Cancellation and disconnect destruction send
CLOSE to every worker before a slot can be reused; new IDs never reuse state.
The adapter validates a whole cohort before any mutation.

Only penalty-free temperature-zero generation with plain drafting is accepted;
unsupported sampling/drafting fails before admission or a rank command. The
prompt adapter supplies finite selection rows from native greedy IDs. Cache
retention and MTP are disabled. A failed rank command poisons the controller.

Each rank must fund eight reservations: config-derived full-context KV and
recurrent bytes plus 512 MiB per session for M1 scratch and alignment. Startup
uses the lowest available rank budget, then caps device allocations to this
aggregate pool plus loaded weights. This conservative sizing is host-only;
actual GPU allocation and serving qualification remain outstanding.

## Replicated dense compute with world-two/four expert split

Set `VADE_FLASHNEXT_FP4_EP2=1` on **every** rank of `flash-serve` or `next-tp`.
The legacy switch name also enables W4 EP; no new arithmetic switch is needed.
Unset or `0` preserves the original layout and packet bytes. Other values and
worlds are rejected. Serving accepts only unset or `NEXT_TP_SPLIT=ep`; the
unqualified `ep2tp2` split is rejected before tokenizer, listener or GPU setup.
The world-one `flash-serve` Next adapters do not implement this layout.

For 512 routed experts, rank zero owns IDs 0..255 and rank one owns 256..511;
odd counts use contiguous balanced ranges, giving the extra expert to rank zero.
At W4 the ranges are 0..127, 128..255, 256..383 and 384..511. Remainders
go to the earliest ranks. Each expert's containers stay intact on one rank.
All packed values, scales and scalars follow their expert. Attention, GDN,
shared experts, router, HC and PLE projections replicate. Embedding/PLE disk
rows and the vocabulary head stay sharded. Startup funds the replicated caches.

Every rank sums its selected routed experts in ascending global ID, starting
from FP32 zero. One rank-major gather of H FP32 values implements the all-reduce:
copy rank zero, add ranks 1 through WORLD-1 in ascending rank order, then add
the replicated shared expert once. Every rank executes exactly that FP32 fold;
transport arrival order never determines arithmetic. This is one reduction per
layer at W2 and W4, including status and route agreement in the same gather.
Only then run HC/residual combination; no collective/normalization fusion.
The packet also carries one status float and the sorted top-k IDs/weights.
Local attention errors propagate through that same gather so peers fail together.
At H=2560/top-10, each contributed packet is 10,324 bytes/one collective,
versus 30,808 bytes/two collectives for the original native W2 adapter. These
figures exclude link framing and the per-token front/head/control packets.
W4 gathers three peer packets per rank (30,972 received payload bytes/layer);
ring forwarding and the two direct neighbor rails use the existing transport.
Startup discovery only reads interfaces and addresses; it never configures them.

Record weights, binary/PTX, topology and settings with every golden set.
Sum order can differ across these backends; same-setup repeatability alone does
not establish cross-topology equality. Compare the required topologies explicitly. `next-tp` case files
remain `NAME PROMPT_CSV EXPECTED_CSV`; expected IDs normally come from that
setup's frozen set. For a **comparison** with the one-Spark CPU reference, set
`NEXT_TP_CPU_REFERENCE=1` on every rank: receipts report agreement counts and
retain generated IDs, full trace and routes without gating on CPU token equality.
This comparison does not qualify determinism. Repeat each mode three times and
compare same-rank full `.f32`/`.routes` files and generated IDs byte for byte.
Rank-local vocabulary rows differ; compare each rank with its own prior capture.

For timing, start fresh paired `flash-serve` processes in alternating mode order
0,1,0,1,0,1 (same binaries/PTX, fresh `TP_RUN_ID`, ring, MTP off). After each
start run the existing client through the rank-zero loopback tunnel:

```sh
vade bench --url "$URL" --model flash-next-nvfp4 --prompts "$PROMPTS" \
  --concurrency 1,8 --max-tokens 256 --repeats 3 --output "$RECEIPT"
```

The client sends all greedy parameters explicitly. Retain effective sampling,
total/per-chat tok/s, bytes per rank, medians/ranges and identical-answer checks
from the private receipts. Overlapping timing ranges are a tie. GPU qualification
and the separate full-row determinism gate remain required before deployment.

## Integrated rank rows

W2/W4 use rank rows for cohorts of 2..=8 and the existing scalar path for one chat. `VADE_FLASHNEXT_WORLD_ROWS` is retired; it has no effect. The HTTP scheduler
admits eight foreground/background rows. Native capture applies the same rule
with `NEXT_TP_COHORT=1|8`. Each row retains ascending expert and rank folds.
Native GPU validation is required for each supported world and for comparisons
across worlds; host coverage alone is insufficient.

QSA/GDN layer objects and their plans stay with each chat, including across reset. Reset clears recurrence and cache cursors while retaining allocation
owners. Disk embedding/PLE descriptors and PLE global scale are cached once;
bounded reads use the same local ownership and CPU dequantization as before.
One expert cohort workspace is funded in each rank arena; per-chat KV, recurrence and conservative 512 MiB scratch reservations remain separate.
Full native captures use buffered writes, flushed with errors at every logits
boundary, to avoid one filesystem write per float.

This rank backend uses native M16 expert tensor cores. Multi-token prompt tiles, batched frontend uploads, device
frontend and per-width graphs are not ported: the native adapter's sharded
frontend and host rank collectives require a rank-aware runner before the
world-one pool can be reused. The M32 prefill TC kernel needs those prompt
chunks; no performance or GPU equivalence claim is made for this integration.
