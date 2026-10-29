# Changelog

This file lists the changes you would notice when you use Vade, newest first.
Changes that are merged but not yet released are under Unreleased.

## Unreleased

- Flash-Next on two Sparks admits more short chats at once. A chat now
  reserves only the KV pages its rows use. Before, every chat reserved the
  pages for the whole context (8,231,544,132 bytes at `--context 131072`,
  whatever the prompt length). The second Spark keeps the buffers of a chat
  that ends and gives them to the next chat in the slot. At
  `--context 131072`, all 16 chats of a 16-chat test (1,247 to 1,248 tokens
  each) started at once, against 2 on main `54392da`. Their median wait in the
  queue was 0.16 s for chats 1 to 9 and 1.02 s for chats 10 to 16, against
  44.8 s and 128.9 s. Two 100,378-token chats still start together, but they
  reach the first token later, because the engine now creates their KV pages
  when they start: 76.9 s and 91.7 s, against 74.8 s and 88.9 s (+2.9 % and
  +3.1 %). Each figure comes from one server start per build, in the
  `cap-c2b` capacity test of 2026-10-28: the byte count from the main
  server's `request_admitted` lines (`reserved_bytes`), and the chat counts,
  queue waits and first-token times from the test client's log of each build.
  Token IDs do not change.

- With drafting on, prompts longer than 32K tokens on two Sparks reach the
  first token sooner. The first Spark now keeps up to 131,072 prompt rows for
  the drafter (KEEP_ROWS in engine/crates/engine/src/flash/next_pp.rs) and
  reads them while it waits for the second Spark's last tile, as it already
  did below 32K rows. Before, past 32,768 rows it read them between its own
  prompt tiles and slowed both Sparks. While a long drafting prompt runs, the
  first Spark holds its kept rows in host memory until the drafter reads them;
  the second Spark returns each tile's rows with its acknowledgment and does
  not hold them. A chain of three or more Sparks keeps the old 32,768-row
  limit (CHAIN_KEEP_ROWS): its middle rank returns all kept rows in one reply,
  and the reply length is a 32-bit byte count. Token IDs and the drafter's
  history do not change.

- Long prompts move fewer bytes in the hyper-connection mix. On fused prompt
  tiles, the norm no longer writes the BF16 hi/lo copy of the four residual
  streams (4 x 2560 features x 2 planes x 2 bytes = 40,960 bytes per token);
  the HC down projection forms the same words from the residual rows instead.
  Token IDs do not change.

- When enabled, the Flash-Next prefix cache can restore a prompt on the tensor
  split. Each rank keeps and restores its own matching state. The layer chain
  still refuses the cache. A second question on a 100K-token chat restores when
  the front snapshot fits the free memory; at `--context 131072` the 6.2 GB
  snapshot does not fit the 5.3 to 6.0 GB that stays free, so that turn reads
  the prompt again. On two Sparks at main `9d2ec6c`, with the cache on, the
  second question on a 32,103-token chat gave its first token in 0.75 to
  0.77 s, against 9.98 to 10.01 s with the cache off. On a 100,364-token
  chat at `--context 110592` it took 1.60 to 1.62 s, against 31.79 to
  31.86 s. The first question takes longer with the cache on, because the
  engine saves the prompt state: the medians are 11.07 s against 10.38 s
  at 32K, and 33.07 s against 31.81 s at 100K. Each range covers three
  server starts. Token IDs do not change
  ([benchmarks](docs/benchmarks.md#the-prefix-cache-on-two-sparks)).

- When enabled, the Flash-Next prefix cache can use free memory that an empty
  spare-buffer reserve has not allocated. Cache retention still yields to live
  requests and physical memory pressure.

- Two Sparks read a prompt of 32K tokens or more in wider passes. The middle
  of the prompt now runs in passes of the whole prompt workspace (4096 rows by
  default) instead of 2048 rows, so the experts run fewer, larger batches. The
  start and the end of the prompt keep 2048-row passes. Shorter prompts read as
  before. Token IDs do not change.

- Two Sparks read a cold prompt of 2K to 8K tokens faster. Each Spark reads
  half of the prompt, and the answer stays the same, token for token. In the
  final 2026-10-27 cards, a cold 8K prompt gave its first token in 2.19 s
  with drafting ON and 2.17 s with drafting OFF, against 2.80 s and 2.77 s in
  that morning's cards.
  Longer prompts read as before. In exchange, two Sparks hold one chat fewer at
  a 32K context: 8 instead of 9.

- `vade doctor` now shows the binary version, four launch defaults, model file
  counts and sizes, and Spark listener ports before startup.

- `vade install` and `vade up` no longer refuse on a stale GPU utilization
  counter. A GPU counts as busy only when a compute app is listed or power
  draw is at least 30 W. A stale counter prints a notice and the command
  continues.

- Checkpoint indexes now reject fractional byte counts that rounded to integers.
  The agent harness now rejects replies with an empty finish reason.

- On two FP4-Local Sparks, a greedy decode step with drafting OFF now splits
  the output head by vocabulary halves across the two Sparks, as drafted
  verification already does. Each Spark computes half of the head. Token IDs
  do not change. Sampled steps and prompts still use the whole head.

- On two Sparks, a long prompt with drafting on no longer stops halfway. Past
  32768 prompt tokens, the first Spark used to wait for the second Spark to
  finish every queued part of the prompt. Now the drafter reads only the
  oldest returned rows that it needs, and the rest waits for the end of the
  prompt. The token IDs do not change.

### Repository tools

- The README states that CI runs locally. A new CI guide gives public host
  build and test commands and explains the private CI script's coverage.
- The security policy states which engine routes local gateway callers can
  reach. It lists access refusals and host tests for the gateway checks.
- The README states the engine's scope and limits. Contribution and security
  guides explain the proof and report steps. Issue forms ask for checkpoint
  details and token IDs. Agent rules now state the same proof and writing
  rules. The benchmark page adds one-Spark results and repeat commands.
- `vade-tools bench-docs <cards-dir> <repo-dir>` refreshes the marked benchmark
  tables from the newest measured main revision. It keeps the raw card values,
  measurement times and revision in both pages. It resolves drafting from the
  request and the recorded server default. An effective depth of 0 means OFF.
  It refuses an unknown drafting mode or depth.
- All repository tools now use Rust and the standard library. There are no
  Python files or Python setup requirements. The tools crate checks links and
  release bundles, tests shell examples, builds CI CUDA linkage stubs and draws
  the release figure. The model helper uses Rust validation and curl. API
  examples are available in shell and Rust. Interrupted model downloads release
  their lock after curl stops. Stalled transfers fail with a time limit. The
  shell API example reports stream errors and incomplete streams as failures.

### Engine

- **A chat alone drafts the same depths every run.** The depth chooser no
  longer reads the time of single passes. On FP4-Local it prices verify rows
  with its cost prior and a draft price. Each chat measures the draft price
  over its first 16 cycles, rounds it to a power of two and keeps it for the
  rest of the chat. The same request on the same setup now drafts the same
  depth in each cycle every run, so its speed varies less from one server
  start to the next. Token IDs do not change, because verification selects
  every token. The `draft_cost` server log line now also shows `plan` (a hash
  of the rows of each cycle) and `draft_price`.

- A one-token decode pass packs the shared expert's input during routing.
  This removes one GPU launch per layer. The math keeps the same order.

- **The tensor split is the only two-Spark path, and the engine chooses each
  path itself.** No `VADE_W2_*` switch remains: `VADE_W2_PATH`, `VADE_W2_TP`,
  `VADE_W2_TP_WIRE`, `VADE_W2_TP_PROJ`, `VADE_W2_TP_HEAD`, `VADE_W2_PP`,
  `VADE_W2_PP_SPLIT`, `VADE_W2_PP_MTP`, `VADE_W2_PP_ASYNC`, `VADE_W2_PP_RDMA`,
  `VADE_W2_PP_STAGE_STATE`, `VADE_W2_PP_TIMEOUT_MS`, `VADE_W2_PP_C8`, the
  `VADE_W2_TP_DOORBELL*`, `VADE_W2_TP_TAKE_*` and `VADE_W2_TP_WIDE` wire
  settings, `VADE_W2_CHUNK` and `VADE_W2_BW` no longer have any effect. Each
  setting keeps the value that was its default, so token IDs do not change.
  FP4-Local on two Sparks runs the tensor split at its proven levels (the write
  wire, projection level 3, the head split by vocabulary halves, verify passes
  of up to 32 rows). On three or more Sparks it runs the layer chain; with
  `--tensor-map` on four Sparks it runs the older rank path. FP8 keeps its
  ring. The serving receipt names the path and why in `vade.numerics.path` and
  `vade.numerics.path_reason`. The split
  needs RDMA: `vade up` refuses `SEQPAR_RDMA=0` on two FP4-Local Sparks
  before it contacts a Spark. FP8 on two Sparks keeps its ring and may use
  TCP. The pipeline wire follows `SEQPAR_RDMA`. For every group of Sparks
  that serves one model, `vade status` shows the path and why in the
  `Multi-Spark path` row, and `vade status --json` shows them in
  `multi_spark_path` (before: `two_spark_path`, for two FP4-Local Sparks
  only). For the split, both also show its levels.

- **Plain decoding on two Sparks now runs as captured CUDA graphs.** With
  drafting OFF, the tensor split's one-token step now captures layers 1 to 47
  in the same segments that drafting's verify pass uses. The GPU runs one
  segment while the host captures the next, so fewer launch gaps stay in the
  step. Each segment keeps the same arithmetic in the same order, and the
  exchanges between the Sparks keep their order. Token IDs do not change.
  There is no setting. A first step of each shape, a segment that the graph
  budget cannot fund, or a step that a decode profile reads back from runs
  eagerly as before. The pass-count engine log line `flash_next segment_graph`
  (before: `flash_next mtp_graph`) now shows `verify=true` or `verify=false`.
  The deferred-free log line keeps the `flash_next mtp_graph` prefix.

- **Two Sparks run the tensor split by default.** `vade up --sparks 2` now
  runs the tensor split, with every proven level on: both Sparks work on every
  token. You set nothing. Before, two Sparks ran the layer pipeline. In the
  2026-10-26 cards with drafting ON, one chat wrote prose at 54.2 tokens per
  second against 42.2 on the pipeline, code at 100.9 against 80.3 and
  structured output at 130.4 against 100.8; eight chats reached 220.0 against
  201.7. With drafting OFF, eight chats are slower on the split: 110.6 against
  137.0. The split needs RDMA. The serving receipt records the path and the
  split's levels under `path` and `tensor_split`. See the [benchmarks
  page](docs/benchmarks.md#current-2026-10-26-main-262b0c3-home-dgx-sparks).

- On two Sparks, a long prompt from a chat that drafts no longer stalls
  halfway. The first Spark now reads the second Spark's replies for each
  prompt tile as they arrive. Before, it read them only late, so both Sparks
  waited on each other for seconds in a 32K prompt. The tokens do not change.

- Before the two-Spark tensor split serves requests, both ranks agree on the
  smaller session count that they can fund. Different free memory on the two
  Sparks no longer causes a startup refusal when both can fund a session.
  The count exchange uses a 600-second load deadline. A worker no longer
  fails when the front takes longer than the serving timeout to load.

- Flash-Next sizes its freed-memory reserve for the largest observed chat
  cohort, with a minimum of four conversations. During a session release, it
  checks memory pressure again after earlier buffers free device memory.
  Attention arenas can then return to the reserve when the memory margin
  permits it. Logs count new allocations, reuse, returns and direct frees.
  Under host memory pressure, a freed buffer also returns to the reserve
  while the buffers of its kind stay inside the admitted session plan. Eight
  chats then reuse their attention arenas without new allocations. Spares
  above the plan are still released. After 30 seconds without a request, a
  reserve releases its spares if it still reads pressure. A new request ends
  the idle period before its admission can allocate.

- With the two-Spark tensor split, the last rows of a long
  prompt now run on the layer pipeline like the prompt's earlier tiles, so the
  first token no longer waits for the two Sparks to swap the prompt's layer
  state. The swap now runs before the chat's second token. The tokens do not
  change.
- With the two-Spark tensor split, the swap of a long
  prompt's layer state between the two Sparks now goes GPU to GPU through the
  split's RDMA mailbox, in pieces of one mailbox slot, instead of through host
  memory and TCP. The state words are copied as they are, so the tokens do not
  change.

The cards of October 28, 2026 after #139 measured main `104205a` on one pair
of DGX Sparks, with the recipe of that morning. Rates below are tokens per
second.

- With drafting ON, eight chats reached 229.6, against 217.5 in the morning
  card (+12.0, +5.5 %). Eight chats now reuse their attention arenas. With
  drafting OFF, eight chats reached 125.4 against 124.3 on the same pair.
- With drafting ON, one chat wrote prose at 53.8, code at 100.2 and
  structured output at 128.7. With drafting OFF, prose was 35.0. The other
  rates are within 1 % of the morning cards. The 32K, 64K and 128K
  first-token cells and the one-Spark prefix-cache cell did not change.
- Token IDs matched in 66 of 66 responses between the ON and OFF cards and
  against the morning cards. A drafting-ON card at main `8b7b866` (#140)
  matched them too, and its 32K and 64K prompts reached their first token
  0.8 % to 1.5 % sooner. See the
  [benchmarks page](docs/benchmarks.md#current-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks).

The morning cards of October 28, 2026 measured main `bb24e80` (#136) with the
same recipe and the same two pairs of DGX Sparks as the final cards of
October 27. Rates below are tokens per second.

- With drafting ON, one chat wrote prose at 54.3, code at 100.7 and structured
  output at 129.3. Eight chats reached 217.5. A cold 8K prompt gave its first
  token in 2.19 s.
- With drafting OFF, one chat wrote prose at 35.1 and code at 35.3, and eight
  chats reached 124.3. A cold 8K prompt took 2.17 s.
- Every rate is within 0.9 % of the final October 27 cards. Token IDs matched
  in 66 of 66 responses between the ON and OFF cards, and against the final
  October 27 cards.
- With drafting ON, a 64,217-token prompt gave its first token in 20.04 s,
  against 21.06 s in the final October 27 run, and a 129,220-token prompt in
  42.47 s. A 32,099-token prompt took 10.03 s.
- On one Spark with context 65536 and the prefix cache on, the second question
  on a 32,103-token chat reused 32,064 tokens and gave its first token in
  0.62 s. With the cache off it took 16.29 s. See the
  [benchmarks page](docs/benchmarks.md#previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks).

The final cards of October 27, 2026 measured main `09e1fc4` (#131) with the
FP4-Local checkpoint on two pairs of DGX Sparks, greedy decoding, thinking off
and no `VADE_W2_*` setting. Rates below are tokens per second.

- With drafting ON, one chat wrote prose at 54.1, code at 100.6 and structured
  output at 129.6. Eight chats reached 218.8. A cold 8K prompt gave its first
  token in 2.19 s.
- With drafting OFF, one chat wrote prose at 35.2 and code at 35.4, and eight
  chats reached 123.9. That morning, on the same pair, these were 29.7, 29.6
  and 110.8. A cold 8K prompt took 2.17 s.
- On a second start with context 69632 and drafting ON, a 32,099-token prompt
  gave its first token in 10.24 s and a 64,217-token prompt in 21.06 s.
- Token IDs matched in 66 of 66 responses between the ON and OFF cards, and
  against that morning's cards. See the
  [benchmarks page](docs/benchmarks.md#current-2026-10-27-final-main-09e1fc4-home-dgx-sparks).

Merges #105 through #110 were measured on October 26, 2026, with the FP4-Local
checkpoint on two DGX Sparks. These runs used greedy decoding, drafting on,
and thinking off. Rates below are tokens per second.

- **#110: Keep eight-chat verification split.** With the tensor split at
  `262b0c3`, verify steps of up to 32 rows split the routed experts between
  the Sparks. Other steps
  split experts only up to 16 rows. With projection level 3 and the split
  verify head enabled, eight code chats went from 173.127 to 219.097 in
  pooled measurements across two pairs, against `09a1184`. All 336 token-ID
  comparisons matched. A later RigMark card on the merged revision measured
  220.0 for eight chats. Both figures are aggregate throughput.
- **#109: Split more layer work between two Sparks.** At `09a1184`,
  projection level 2 splits output projections, attention query/gate
  projections, and the shared expert's down projection. Level 3 also divides
  linear-attention heads and their recurrent state on one-chat steps. A shorter first captured
  verify segment reduces the initial wait for GPU work. In two runs per
  revision on one pair, level 3 against level 1 on `dbc2385` changed prose
  from 48.104–48.244 to 51.079–51.243, code from 94.343–94.968 to
  97.711–97.775, and structured output from 121.116–121.417 to
  124.674–125.281. Both revisions kept the verify head unsplit. All 168
  token-ID comparisons matched.
- **#108: Share linear-attention input projections.** At `dbc2385`,
  projection level 1 divides the input projections of split layers between
  the Sparks when one chat runs. It exchanges both sets of output rows
  before waiting for either set. It also fixes weight-copy lookup and bounds
  the state transfer before allocation. In two runs per revision on one
  pair, against `41e3c22` with projection splitting off, prose changed from
  45.208–45.454 to 47.804–48.253, code from 91.190–91.347 to
  94.493–94.848, and structured output from 117.047–117.213 to
  119.726–121.516. Both revisions kept the verify head unsplit. All 168
  token-ID comparisons matched.
- **#107: Share token scoring between two Sparks.** At `41e3c22`, the
  split verify head scores half of the vocabulary on each Spark for one
  greedy chat. When bounds are available, each Spark screens its own half
  and fully scores the rows that can win. Sampled chats and drafts keep the
  whole head on the first Spark. Against `984147f` with the tensor split and
  RDMA write transfer,
  pooled measurements across two pairs changed prose from 45.617 to 47.091
  and structured output from 117.189 to 120.018. Code changed from 91.636
  to 92.053, a tie within the gate's tolerance. All 392 token-ID comparisons
  matched, including the extra eight-chat check.
- **#106: Let the transfer thread sleep.** At `984147f`, the RDMA write
  transfer thread sleeps when no exchange is in progress. During requests,
  it used 0.82 of a CPU core on the first Spark and 0.79–0.80 on the second,
  compared with 0.98 on each before. Throughput tied with `2ef7744`; all 48
  token-ID comparisons matched. A stop during startup also
  cancels the connection attempt if the other Spark does not answer.
- **#105: Check more draft tokens with the bounded head.** At `2ef7744`,
  the exact head that skips vocabulary rows that cannot win also covers
  verify steps of five to eight rows, up from two to four. Pooled
  measurements across two pairs, against `6f5388d` on the pipeline path,
  changed code from 78.689 to 80.375 and structured output from 98.215 to
  100.305. Prose changed from 42.008 to 42.118, a tie. All 336 token-ID
  comparisons and 79 exactness checks matched.
- On two Sparks with the FP8 checkpoint, the routed experts' decode tiles
  now keep two weight blocks in flight instead of one, so decoding is a little
  faster with the same tokens.
- On two Sparks with the FP8 checkpoint, one chat now picks how many tokens to
  draft on each step, up to seven, from how many drafts were accepted recently
  and how long each verify step took. Code and structured replies are much
  faster; prose stays as fast as before. The tokens do not change.
  `VADE_FLASH_MTP_DEPTH=1..7` sets the most it drafts; `VADE_MTP_ADAPT=0` or an
  explicit `--mtp` keeps the old fixed depth.
- On two Sparks with the FP8 checkpoint, drafts read a draft-only MXFP8 head
  copy, so each drafted token costs less: code and structured replies are
  about 9 percent faster, prose about 4 percent. Verification keeps the BF16
  head, so outputs are unchanged. Cost: rank 0 chat slots 19 -> 17, and
  +0.33 GB per rank. `VADE_FP8_DRAFT_MX8=0` turns it off.
- With the two-Spark tensor split, the write wire moves each exchange with
  one-sided RDMA writes straight from one GPU's mapped mailbox into the
  other's, with no host copies and no host work on the receiving side. It is
  now the split's only wire. The tokens are the same. Measured on two Sparks
  with a mailbox benchmark, one exchange: 17.6 to 10.3 microseconds at 10 KB,
  34.0 to 15.3 at 40 KB, 165.9 to 59.4 at 400 KB.
  One chat with drafting on, against the split on today's wire (main
  `62c9c83`): prose 44.76 to 45.68, code 87.14 to 91.48,
  structured 111.08 to 114.26 tokens per second, ids identical. The front
  Spark still waits on the other Spark's compute for most of each exchange.
- On one Spark, `VADE_MTP_COHORT=wide` lets five to eight chats draft three
  tokens each and check them in one 32-row step. Before, five chats proposed
  two tokens each and six to eight chats one each. Four or fewer chats keep
  the 16-row step. Measured on one Spark with eight code chats (`45b3950`
  against main `7e625e7`): 169.6 to 194.4 tokens per second, with the same
  tokens. Unset, nothing changes on one Spark. On two and four Sparks the
  layer pipeline now takes this step by default, with the cost controller
  opt-in (eight code chats on two Sparks, `277463d` against main `254fb86`:
  176.1 to 201.3 tokens per second, same tokens); `VADE_MTP_COHORT=off` on
  every Spark restores the previous step and controller.
- FP8 experts now launch fewer kernels per layer when a step has under 32
  tokens: the gate kernel also computes the activation and the quantized
  middle, and the routing kernel builds the tables for 2 to 31 tokens in one
  launch. The arithmetic order is unchanged, so the tokens are the same.
  Measured on two Sparks with the FP8 checkpoint against main: one chat 42.34
  to 42.77 tokens per second with drafting on and 25.78 to 25.79 with it off,
  eight chats 93.60 to 95.32. The expert weights already streamed near the
  memory limit, so the gain is small; `kernel-bench --fp8-moe` times each
  stage of the expert layer.
- Two Sparks can now split every decode step between them instead of passing
  it down the layer pipeline. This tensor split is now the only two-Spark path
  for FP4-Local. Both Sparks compute half of each routed expert
  and exchange the halves inside the GPU stream, so the tokens stay identical
  to one Spark. Long prompts run as pipeline tiles, and each chat then hands the
  other Spark the state it did not compute. Measured on two Sparks with drafting
  on, against the pipeline: prose 41.73 to 44.70, code 78.84 to 87.34,
  structured 97.99 to 111.41 tokens per second, one to four chats 4 to 9 %
  faster, 8 chats unchanged; ids identical on all 66 card requests. A cold 8K
  prompt takes 2.82 s instead of 2.79 s. Plain (drafting-off) decoding was about
  7 % slower in a smaller test: under the split it does not use pooled decode.
- On four Sparks, the layer pipeline now drafts for six to eight chats at
  once, under the same rules as two Sparks: each tick checks up to 16 drafted
  rows, and the engine switches a cohort to ordinary decoding when that is
  cheaper. Before, eight chats on four Sparks were always decoded plainly.
  Measured on four Sparks with eight code chats: 138.93 to 180.95 tokens per
  second, with the same tokens.
- On one Spark, when one chat runs alone, the drafting depth can now rise to 7
  instead of 5, as on the layer pipeline. Measured on one Spark: code 75.6 to
  81.6 tokens per second, structured output 88.3 to 99.7, prose unchanged, with
  the same tokens. Batches keep their depth. `VADE_FLASHNEXT_MTP_DEPTH=5`
  restores the previous ceiling.
- On the layer pipeline, the check of drafted tokens now replays each
  Spark's layers as captured GPU work, and when one chat runs alone the
  drafting depth can rise to 7 instead of 3 (the resident depth in the
  [glossary](docs/GLOSSARY.md)). Measured on two Sparks against depth 3:
  code +16 %, structured output +39 %, prose unchanged, with the same
  tokens. `VADE_FLASHNEXT_MTP_DEPTH=3` restores the previous ceiling, and
  `VADE_MTP_GRAPH=0` keeps every Spark's work eager. The 2026-10-25 two-
  and four-Spark cards on the [benchmarks page](docs/benchmarks.md#reading-these-cards)
  include it: with drafting on, one chat wrote code at 79.3 tokens per
  second on two Sparks, against 65.1 on the v0.1.0 pair.
- On the layer pipeline, the first Spark now projects the model's head for
  only the rows one greedy chat needs, as one Spark already does, instead
  of the full head. Measured on a pair, prose went from 41.26 to 42.60
  tokens per second with the same tokens. `VADE_HEAD_BOUNDS=0` turns it
  off.
- On the layer pipeline, each Spark now keeps chat state only for the layers
  it runs, which leaves room for more chats at once: 15 instead of 9 in our
  two-Spark test with a 109 GB memory ceiling (the number depends on the
  memory each Spark has free). In that test, sixteen chats sent at once
  finished 22 % faster with drafting on and 6 % faster with drafting off, with
  the same answers.

### CLI

- `sh scripts/install.sh` sets up the pinned Rust tools and source build on
  one Spark, checks an existing checkpoint, and prints recovery steps on failure.
- On a Spark, `vade install` now compiles the GPU code into Vade's private
  startup cache before you run `vade up`. It leaves checkpoint files alone.
- `vade install` from a controller now builds every Spark at the same time
  instead of one after another. It still prints each machine's output in
  order, and if a machine fails it stops with that machine's error, as before.
- `vade launch` no longer opens a chat when its input isn't a terminal, for
  example under `script` or with input from `/dev/null`. It prints the ready
  line and the chat command, exits successfully and leaves the engine running,
  so you can connect later with `vade chat`.
- When you run Vade by a path, such as `./vade` from the launcher bundle
  folder, the next-step hints from `vade launch` and `vade status` repeat that
  path (for example `./vade chat`) instead of a bare `vade` that your shell
  may not find.
- A build supervisor that crashed, was killed or was lost to a reboot no
  longer blocks every later `vade install`, `launch`, `up`, `update` and
  `down` with "Build supervisor was lost". Once its processes are gone,
  recovery closes the record and prints one line naming the job folder.
  While anything may still run it keeps the record and the error as
  before, and it still sends no signals.
- `vade test` help lists world 1, which the determinism test accepts, and
  `vade help` puts a blank line between the evaluation commands and the
  gate usage.
- Every `vade` help page has the same layout: a summary, usage, an aligned
  options table, examples and notes. `vade --help bench` and `vade -h bench`
  now print the bench help, as `vade help bench` does.
- Errors print as `error:` with what happened, then `hint:` with the next
  step. Launch, install, up, down and chat print their steps in one aligned
  column, and `vade status`, the bare `vade` menu and the launch plan print
  aligned rows with the next commands. Colour appears only on a terminal;
  `NO_COLOR` or `TERM=dumb` turns it off, and JSON output and receipts never
  use it.
- `vade up --dry-run` and `vade launch --dry-run` name the context and output
  limits they would use, and whether each one is explicit, saved or the
  default, instead of printing values such as `explicit context: None`.
- `vade doctor` prints one row per check (`ok`, `FAIL`, `skip` or `plan`,
  the check, what it found), the fix under each failed row, then one PASS or
  FAIL line with the counts. A failed check no longer stops the rest, and a
  check that an earlier failure hid is `skip`, not a guess. From a
  controller with no engine running, the engine row is `skip` and doctor
  exits 0 when the machine checks pass.
- `vade config` (new) names the setup the lifecycle commands will use
  (local, controller, legacy or none) and what it holds. An owned engine
  whose local config is gone is still the local setup, reported as an error.
- `--json` now works on `doctor`, `config`, `detect` and `version` as well as
  `status`: one object with a `schema` field, the same checks and exit code as
  without it, and failures inside the object.
- Help pages start with examples, and `vade --help` groups the commands.
  `vade verify` argument errors use the `error:`/`hint:` form on stderr, and
  verify prints progress on stderr while a request is still running.

### Harness

- New `vade-harness quality-card` measures answer quality and the cost of
  exactness. It runs one fixed set of 12 prompts (code, structured output,
  prose) against a served model and prints two cards: quality scores from
  exact checks only (code answers run under unit tests), and strict
  drafting against plain decoding with token ids compared and the verify
  cost read from the server draft receipts. See the
  [quality card guide](harness/docs/quality-card.md).
- Chat has two voices: your turns sit behind a `▌` bar on every row and the
  answer has none. A thought is one dim row (time, tokens, seconds) that
  Ctrl+T expands. One status strip above the `›` prompt shows the model,
  context used of the window, tokens in and out and tokens per second as raw
  numbers; an unknown count is left out, never shown as 0. Text wraps
  between words.
- Claude Code works through both gateways again. Claude Code 2.1.310 puts
  its environment notes in every request as a `system` role entry inside
  the messages, which the engine refuses with HTTP 400, so `claude -p`
  failed outright. On `/v1/messages` and `count_tokens`, both gateways
  now rewrite just those role values to `user`, the recovery Claude Code
  documents; every other byte is forwarded unchanged.
- `vade-harness setup claude` prints the gateway settings for Claude Code:
  the API-root base URL, a dummy bearer, every model alias and the
  subagents pointed at the loaded model, and betas, tool search and
  nonessential traffic off. `--dedicated-home` also writes a private
  Claude Code home with `CLAUDE_CONFIG_DIR`, the verified context and
  output limits (Claude Code otherwise assumes 32000 output tokens), read
  and search-only permissions, the tool rule and a completed onboarding
  flag. Checked with Claude Code 2.1.310: a plain answer and a real Read
  tool round trip.
- `vade-harness setup aider`, `setup continue` and `setup open-webui`
  print each client's documented settings for the engine or the gateway:
  aider's `OPENAI_API_BASE` and `openai/` model prefix, a Continue
  `config.yaml` for chat, edit and apply (no autocomplete, because the
  engine ignores FIM suffixes), and Open WebUI's `OPENAI_API_BASE_URL`
  with the Docker host and `--allow-host` note. These print or `--write`
  only; there are no dedicated homes for them.
- The built-in gateway (`vade-harness gateway`) relays `GET /v1/models`
  and `/models` to the upstream's model list, keeping the query, so Open
  WebUI, Continue and the OpenAI SDKs can list models before they chat.
  It used to answer 404. In cluster mode the list comes from one verified
  endpoint. `POST` gets 405 and `/v1/models/{id}` 404.
- Both gateways' own refusals (401, 403, 404, 413, busy, upstream
  unreachable) now carry an error type in the route's dialect:
  `{"type":"error","error":{...}}` on the Messages routes, as Claude Code
  and the Anthropic SDK expect, and the OpenAI shape with a type such as
  `rate_limit_error`, `overloaded` or `server_error` elsewhere. Upstream
  error bodies are still passed through byte for byte.
- `vade-harness gateway` relays `POST /v1/kv/release` (it answered 404),
  so a native session behind it can confirm a checkpoint release. With
  several upstreams, cancel and release no longer answer 503: they go, in
  configuration order, to every endpoint whose contract discovery has
  verified, and you get the first reply that reports cancelled or
  released, otherwise the first reply, or 502 when none answers. An
  endpoint that never verified, or that answered as foreign or with a
  changed contract, does not see the key until the verified contract
  answers again. One gap remains: no handle-based cluster affinity for
  requests that name a checkpoint handle.
- In cluster mode, `vade-harness gateway` sends cancel and release to
  every verified endpoint at once, under one 1.5-second deadline (or the
  client request's own, if sooner), instead of one after another with a
  30-second idle budget each. One stalled endpoint no longer keeps the
  others from hearing a cancel. `vade-gateway` still fans out one
  endpoint at a time.
- `vade-gateway` keeps listening after a failed accept, such as an
  aborted handshake or a full descriptor table. It drops only that
  connection and logs accept errors with a short backoff.
  `vade-harness gateway` does the same.
- When `vade-gateway` cannot record a request or write its statistics, it
  says what happened to the request (forwarded but not recorded, served
  without its statistics line, or recording incomplete) and names the
  folder or file to check, instead of a bare "recorder start failed".
  `demo-compaction` with an argument says to run it on its own.
- On Linux, including the Spark itself, the coding agent is no longer
  offered `workspace_shell`, which has no sandbox there and only ever
  returned `unsupported_platform`. The agent is told it cannot build, test
  or run anything, re-reads its edits and names the command for you to
  run, and `/status` says there is no command tool. macOS is unchanged,
  and sessions saved by an earlier Linux build still resume.
- `--children --child-shell NAME` is refused while the arguments are
  parsed, with a message that names the cause, where the shell is not
  offered (Linux, or `VADE_HARNESS_SHELL=off`), instead of failing later
  with "invalid child tool scope".

### Documentation

- The README opens with a short summary and separate quick starts for one
  Spark and for two or more, and the benchmarks page adds the 2026-10-25 two-
  and four-Spark cards.
- The README and the [benchmarks page](docs/benchmarks.md) show measured
  v0.1.0 results, and the receipts behind them are published under
  [bench/receipts/2026-10-24](bench/receipts/2026-10-24/README.md).
- The 2026-10-25 cards ran on main `6cb46bf` and `80ac2d4`. Each row names
  its result folder; those folders are not in the repository yet.
- The [architecture page](docs/architecture.md) follows one request from
  the gateway through the scheduler to the model, and explains the storage
  formats, the kernels, drafting, the layer pipeline and the prefix cache.
- A new [examples folder](examples/README.md) has short scripts, each shown
  with the output it printed on a Spark: curl against the OpenAI and
  Anthropic endpoints, a standard-library Python client, Claude Code and the
  Codex CLI pointed at Vade, a `vade bench` run that reproduces the published
  one-Spark number, and a two-Spark launch from a controller that is a dry
  run by default.
- The results figure compares one Spark with two, with drafting on. In the
  2026-10-24 runs, a second Spark answered a long prompt sooner but didn't make
  a single chat write faster. Eight chats at once vary from run to run, and the
  figure shows the range we measured: 130.6 to 173.9 tokens per second on six
  single Sparks and 167.6 to 168.7 on three pairs, from the same-day cards with
  drafting on. The headline is checked against all nine cards.
- [INSTALL](INSTALL.md) covers more situations: an engine that turns away an
  extra chat with HTTP 503, checking each replica with `vade verify --url`,
  keeping the launcher bundle outside your checkout, and what to do when
  `vade install` refuses a checkout with local changes.
- The guides were rewritten in plainer language. Commands, numbers and links
  didn't change.
- Two facts were corrected against the source: rule files are capped at
  16 KiB each, and the automatic output reservation includes the
  16,384-token cap.
- README, [INSTALL](INSTALL.md) and the one-Spark quickstart say what chat,
  status, sampling and the bundle really do, checked against the source:
  controller `vade chat` always opens the agent, which refuses a context
  below 8192, and only local chat falls back to direct chat; local
  `vade up` steps a refused default or saved context down as `vade launch`
  does; controller `vade status` prints no URL, since the API is on the
  local port; naming any sampling field fills the rest from the checkpoint
  defaults; and the two legacy routes are named, with `vade run` needing a
  fresh `VADE_HOME` after `vade launch`.
- [INSTALL](INSTALL.md) and the CLI guide say that a controller install
  needs `vade-prebuilt-v1`, not the launcher bundle. INSTALL and the
  [benchmarks page](docs/benchmarks.md) say how a `vade bench` with no
  flags differs from a RigMark card.
- The [security policy](SECURITY.md) describes the gateways'
  `VADE_GATEWAY_TOKEN`. It applies to the gateways you run by hand; the
  replica gateway behind `vade up --replicas` never takes a token.
- The harness guide and the invisible-gateway page list the complete
  gateway sampling policy again: every shared default with its exact
  value, every sampling field, and the null rule.
- Engine docs: a new [unsafe-code page](engine/docs/unsafe.md) names the
  four boundaries where the engine uses `unsafe` (kernels, the CUDA
  driver, launches, RDMA), gives a per-package inventory and the exact
  count of `SAFETY` notes, lists the lints that make `unsafe` a compile
  error elsewhere, says how to write a `SAFETY` note, and states plainly
  that the ratchet does not count unsafe sites today, so review is the
  check. A new [module map](engine/docs/module-map.md) follows one request
  through `flash-serve` and names the main modules per package and where
  the tests are. Both are linked from the engine docs index.
- Engine docs corrections: messages are capped at 4,096, and eight spare
  connections are kept for `/health` and `/v1/cancel` once requests fill
  the 32 (there is no single control connection); Flash Next is not a
  separate research engine, the v0.1.0 model runs in `flash-serve`,
  started by `vade launch` and `vade up`; and the architecture page says
  the layer pipeline's drafting ceiling is 7.
- Smaller fixes: the harness guide puts the source-built harness on PATH,
  the network guide's RDMA GID example runs as written, the fresh-install
  guide's legacy controller examples match current output, the
  adding-a-model anchors point at the moved CLI module docs, and the
  benchmarks page and CLI guide are clearer about bench `ignore_eos`,
  `--checkpoint-hash` and the controller config.

### Tools

- The chart tool `vade-tools benchmarks-chart` reads the release-card
  numbers from the committed receipts, reads and writes UTF-8, names a
  missing receipt folder, and stops if the cards stop supporting the
  figure's headline. The assets README lists both figures and gives the
  bar colours as hex values.
- `harness/scripts/route-a-check.sh` checks Route A on one Spark in one
  command: toolchain, CLI build, model, `vade launch`, one chat request,
  `vade verify` and `vade down`, each with PASS or FAIL and its seconds. It
  runs in a new private Vade home of its own, so it never touches `~/.vade`,
  and stops its engine also after a failed step or an interrupt. See
  [INSTALL](INSTALL.md#check-the-whole-route-in-one-go).

### Project

- Bug reports and feature requests now use issue forms. The bug form asks for
  the version, model, number of Sparks, how you run Vade, the exact command
  and the full error, plus `vade verify` output when answers are wrong.
- The security policy explains how to report a vulnerability privately, what
  it covers, and what Vade exposes on the network by default.
- [CONTRIBUTING](CONTRIBUTING.md) says how a pull request lands: CI, a
  maintainer's review (requested through CODEOWNERS), the device gate for
  engine changes, and the published update that carries it.
- Every public item in the CLI crate and the harness library has a doc
  comment (140 harness items had none), the CLI's `main.rs` maps its
  modules by area, and rustdoc builds without warnings.
- No CLI source file is over 1,200 lines any more, and every non-test
  harness source file is under 1,200 lines: tests moved beside their
  modules and the largest files were split along their seams, with no
  change in behaviour.
- New tests pin the CLI's `vade doctor`, `vade status`, `down`, `logs`,
  `detect` and `update` messages, its remaining argument errors and every
  state in which controller chat refuses to open; the engine's bounded
  JSON reader and admission refusal text, prompt partitions and loopback
  rank admission, the fabric peer layout and profiling wrapper, and the
  tokenizer against a real-format `tokenizer.json`; and the native
  `/status` tool list per platform. The harness tests no longer depend on
  an exported `VADE_GATEWAY_TOKEN`, the native tool tests run in both
  shell shapes, which fixes Linux CI, and the loopback admission tests
  retry a port lost to a parallel process.
- Clippy findings that pointed at real issues are fixed in the CLI and the
  harness, and the CLI has one SHA-256 implementation instead of two.

## v0.1.0 - 2026-10-24

The first public release. Vade runs a large language model on your own NVIDIA
DGX Spark and gives you three things: a coding agent in your terminal, a plain
chat, and an OpenAI-compatible API. It serves one model,
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (text only), on one Spark or on linked
Sparks. It is written in Rust using only the standard library, and it is built
so that the same request on the same setup returns the same token IDs on every
run; `vade verify` lets you check that on your own machine.

### Engine and serving

- An inference engine for the NVIDIA DGX Spark, written in Rust using only the
  standard library. The release path is the text-only Qwen Flash-Next
  FP4-Local model, on one Spark by default. Configured multi-Spark
  deployments run through the exact layer pipeline, which splits the model's
  layers across the Sparks.
- Drafting is on by default: an extra model head proposes tokens and the model
  verifies them. A request that sets no sampling field is decoded greedily, on
  any number of Sparks. A request that sets any sampling field starts from the
  checkpoint's own sampling defaults, and its values replace those defaults.
- Drafting on two Sparks. Both pipeline stages follow the same drafting
  policy for a cohort (a group of requests running at the same time), and the
  two-stage pipeline uses guarded cohort drafting, which switches a cohort
  back to plain decoding when drafting stops paying off.
- At startup, the pipeline refuses to run if its ranks (one process per
  Spark) have different artifacts, numerical settings or prompt options. It
  also checks the pipeline switches before it chooses a topology.
- `/health` and receipts report the drafting mode and depth the server
  actually ran, not the ones the request asked for.
- Release builds load verified precompiled kernels. Checkpoint hashing uses a
  safe streaming SHA-256 that does not fill the file cache.
- The startup memory check counts reclaimable page cache as free (Linux
  `MemAvailable`, capped by CUDA total), so it also works inside containers,
  where CUDA's own free-memory figure leaves the page cache out. A memory
  refusal names MemAvailable, the required bytes and the ceiling. Session
  pools hold back a 512 MiB reserve, so 8 and 16 concurrent chats at the
  default context have room to finish.
- The Flash-Next prefix cache is off by default. It stays off until tests show
  that serving with the cache is as fast as serving without it in every
  measured case. `VADE_FLASHNEXT_PREFIX_CACHE=1` turns it on, on one Spark
  only.

### Model and installation

- A pinned, verified FP4-Local publisher revision with a committed SHA256
  manifest of the complete snapshot, plus a one-line fetch helper that checks
  the download against that revision.
- A replay-tested local install on one Spark. The launcher reads the Spark's
  identity and resources, finds the models you already have, generates local
  defaults, and runs install, up and chat in one terminal.
- A prebuilt launcher bundle for Linux aarch64. A release workflow builds it on
  version tags, checks every manifest hash, and attaches the bundle and its
  `SHA256SUMS` to the tag's release. If a bundle is rejected, the launcher
  falls back to building from source.
- Guides for first-time readers. README, INSTALL and the first-run guide
  start with the one-Spark launcher, keep the source route and the bundle
  route apart, and name the supported model explicitly.
- A two-Spark route in INSTALL, next to the one-Spark default.
- A supported-systems table in INSTALL. The engine runs on DGX Spark / GB10
  under Linux `aarch64`. For Route B, where you drive the Spark from a
  second computer (the controller), the controller can be Linux `aarch64` or
  `x86_64`, or macOS. For Route A, where the CLI, engine and chat all run on
  the Spark, a Windows computer is supported as the SSH client, but not as a
  controller.
- Source installs also work from a downloaded source archive with no Git
  metadata. The launcher records the archive's stamped commit as the build id,
  or builds as `unknown` when the archive carries none.

### CLI, chat and gateway

- Controller commands for install, launch, up, status, logs, verify and
  shutdown, with bounded JSON status, specific errors that say what to do next,
  and help that stays usable in pipes.
- `vade up` on the controller and `vade launch` on the Spark start the same
  FP4-Local defaults (context 32768, output 8192). If the engine refuses the
  default or saved context for memory, `vade launch` steps it down through
  16384, 8192 and 4096 and saves the result once a start succeeds; that is the
  only case where it steps down. An explicit `--context` that the engine
  refuses is an error.
- `--context 262144` keeps the single-chat long-context preset, and
  `vade status` prints the chats the engine admitted (how many it allows to
  run at once at this context). A controller bench measures concurrent chats
  instead of one queued chat.
- `vade up --replicas N` (2..16, FP4-Local) starts an independent one-Spark
  server on each of N Sparks behind one local `vade-gateway`, which sends each
  request to the least-busy replica. `vade-harness vade-gateway` runs the same
  gateway by hand, with `--upstream` repeated.
- A machine's `ssh` entry can be a `user@host` destination as well as an
  alias, host name or IPv4 address.
- `vade verify` checks that repeated greedy runs against a running server
  return the same tokens, and finds the endpoint of the server it owns
  automatically.
- Chat history lives under the Vade home (`~/.vade/projects`), never in the
  folder you chat from. Set `VADE_CHAT_IN_PROJECT=1` to keep it in a
  git-ignored `./.vade` instead.
- Chat and coding-agent workflows: steer a chat while it runs, keep replies
  that were interrupted, manage sessions, and use an OpenAI-compatible HTTP
  gateway with credential handling and recorded replay.
- Both gateways refuse browser-driven requests by default. They refuse a
  request whose Host header is anything other than loopback with the listener
  port, and any request that carries an `Origin` header. You can allow a
  browser client on purpose with exact allow-lists and an optional bearer
  token.

### Benchmarks

- `vade bench` runs with no flags. It finds the recorded endpoint, reads the
  model from `/health`, uses a bundled offline prompt and measures the
  server's own drafting default.
- Drafting-aware receipts (the saved result records) that keep the requested
  drafting mode apart from the mode and depth the server reports, and withhold
  rates when the two disagree.
- A benchmark page with methods, results and limitations, linking every quoted
  source folder to its published, sanitized receipt.
- A results chart in the README and on the benchmark page, drawn by a
  standard-library script from the committed receipts, so every label is a
  value that a receipt prints.

### Project and community

- Agent rules in AGENTS.md and the human contribution process in
  CONTRIBUTING.md, with a package map, architecture notes and decisions.
- A security policy, Code of Conduct, issue and pull request templates, NOTICE
  and the bundled third-party license texts.

For measurements, recipes and their limitations, see
[benchmark methods and results](docs/benchmarks.md). See [NOTICE](NOTICE) for
bundled third-party material and [LICENSE](LICENSE) for Vade's Apache-2.0
license.
