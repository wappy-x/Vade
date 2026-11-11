# Flash-Next rows

`NextSession` has one execution path for a prompt tile or one token from each
of up to sixteen independent chats. On one Spark, unset `VADE_PROMPT_TILE`
allocates memory for an 8192-row prompt workspace. Automatic tiles use 4096 rows
for prompts ending below 8192 tokens. Prompts ending at or beyond that boundary
use 8192 rows. An explicit value
of `2048`, `4096` or `8192` selects a fixed tile size. See
[`prompt_tile` and `prompt_tile_for`](../crates/engine/src/flash/next_rows.rs).
[`set_chunk_size`](../crates/engine/src/flash/next_batched.rs) accepts 1 through
the session's admitted `chunk_limit`. The default limit comes from `prompt_tile()`;
`session_with_chunk` can set a smaller limit. This changes scheduling only.
Multi-token tiles start at
absolute positions divisible by 64; unaligned continuations use single rows. `step` is the same path with
one row, including a cold sequence. Sparse QSA is the default indexer.

The world-one `flash-serve` bootstrap selects mixed checkpoints before creating
rank transport. It uses `spark_serve::flashnext::FlashNext`, the existing HTTP
scheduler and sampler, automatic prompt tiles and up to sixteen memory-admitted chats.
Strict greedy MTP defaults to request depth 3 with prompt lookup; requests may
set `mtp_depth` to 0..=3. The resident ceiling is separate, as described below.
The [launcher](../crates/tp/src/bin/flash-serve.rs) accepts world sizes 1 through
64. FP4-Local uses the tensor split on two Sparks and the layer chain on three
or more. Those routes call the mixed loader as world one. The
[`vade up` limit](../../cli/src/config.rs) is 24 Sparks. Accepted launch sizes
do not establish device qualification. The
[startup settings](../crates/tp/src/flash/settings.rs) fund a 4096-row prompt
workspace for the tensor split and pin layer-chain tiles to 2048 rows by default.
Session reservations cover KV, recurrent state and draft state. The model funds
shared prompt scratch once. HTTP passes the admitted chunk to
`session_reservation_bytes_for_chunk` and `session_with_chunk`; see the
[row model](../crates/engine/src/flash/next_batched.rs) and
[HTTP adapter](../crates/serve/src/flashnext.rs). Sixteen
128K sessions are not assumed to fit; the scheduler admits within the pool.

Arithmetic and ownership:

- Mixed-checkpoint dense projections keep the same BF16 hi/lo operands,
  ascending K16 chains and four-range merge, including M=1. Small rows use
  M16/N32 tiles; grouped prompt projections from 64 rows use fused M64 or
  M128 scheduling. Variable M/N tiles preserve each output's K16 arithmetic. HC norm
  keeps its solo warp reductions at every row count. This replaces the old
  shape-dependent projection dispatch, so old-build logits require qualification.
- GDN projects the whole tile, convolves causal inputs against each sequence's
  prior history, and carries each recurrent column in token order. The shared
  `next_math::column` function handles cold rows and prompt chunks. No triangular
  chunk solve or chunk-local cumulative decay changes the recurrence order.
- QSA projects the whole tile and applies the existing causal sparse selection,
  score, softmax and value kernels. Each chat keeps its own past and KV/index
  history. Prompt spans share projections and execute attention in causal order;
  decode cohorts share capacity-bounded batch kernels. Custom Dense hooks retain dense attention.
- Routed experts use unchanged packed nibbles, expanded stored scales and the
  BF16 operand contract at every M. The router's fixed warp softmax/top-ten,
  compact expert bounds/inverse map, and common K16 grouped FFN run
  without host route readbacks. Small rows build routes in one CTA; prompt
  tiles from 64 rows use four stable cooperative phases, including per-expert
  CTAs. Expert rows are never padded in memory: GU
  gathers BF16 operands in registers, the middle/down arrays contain live rows,
  and only register tiles mask their tails. Each expert shares one weight sweep per M16 tile;
  gate/up and down use FAST_MOE's one ascending K16 chain, BF16 projection
  boundaries, BF16 SwiGLU with expf reconstruction, and ascending-ID FMA fold
  with FP32 route weights. Shared is added last with the same FMA contract,
  then the block output has FAST_MOE's BF16 boundary. Shared projections retain
  the common canonical high/residual arithmetic.
  This changes the earlier scalar expert arithmetic and requires model qualification.
- HC uses option A's common K16 down/up order at every M. Its up epilogue
  fuses the four stream sigmoid/mix operations. Activation, mix and injection
  share the frozen expf reconstruction at every M. There is no shape-dependent reduction or BF16 gate round trip.
- PLE hashing and convolution retain each chat's token order. Projections use
  the same canonical K16 contract in eager chunks and pooled decode.
- The head uses the first-party bandwidth study's K32 resident permutation,
  aligned 128-bit loads and two M8 accumulators sharing each weight fragment.
  Four fixed K ranges retain independent high/residual chains and the same
  balanced merge. Fused partial maxima preserve lowest-ID ties and reject
  nonfinite logits. Full logits remain available to the serving sampler.
  Eager prompt spans project only their last token; a decode step also computes
  the head for advance-only calls. Debug dumps use eager one-token slices.
- Validation precedes history changes. Execution failures invalidate every cohort
  participant until reset. Prompt advance invalidates old logits. Workspace may
  grow; it borrows dead expert-output storage for QSA projection rows.

Host tests exercise shared HC/expert/recurrent arithmetic, ordered BF16 MoE
finish, the head permutation/tie contract, sparse QSA visibility, slot admission,
dummy exclusion and workspace extents across chunk/batch boundaries.
They do not establish GPU instruction equivalence. Qualify full logits on the
same machine before treating this as byte-exact or claiming a speed improvement.

On a Linux/CUDA box with the official mixed checkpoint and matching built PTX:

```sh
FLASH_NVFP4=1 cargo run --release --offline -p spark-engine --example flashnext_rows -- \
  --model "$MODEL_DIR" --check --measure --prefill 8192 --decode 256
```

Run from `engine/` with the pinned toolchain. `--check` compares complete logits
at prompt chunk sizes 1/2/7/16/63/64/65/127/512 and every chat width 1..16, including
reversed chat order and a continuation after the prompt. `--measure` reports
cold-session 8K TTFT and C1/C8/C16 decode aggregate/per-chat rates, with model load
separately reported. These are native engine measurements, excluding HTTP.
Repeat in three fresh processes and report medians and ranges. Receipts contain
hashes only; they never print full logits. The existing full-model debug gate
continues to provide reset replay and optional CPU-stage comparison.


Decode slots and graphs

The serving Session API selects a model-owned graph for each padded width
1/2/4/8/16. A single uploaded `u32` slot vector maps the live rows to funded
GDN state/history, QSA KV/index histories and PLE convolution allocations.
The descriptor table has seventeen records and reserves slot 0. State is
allocated when a session is admitted; the table references those allocations,
without reserving sixteen full contexts. Retiring a session releases its slot;
a new occupant replaces its descriptor before a replay can select it.

Every mutation resolves its slot on device before dereferencing history.
Dummy rows skip GDN, PLE and QSA mutation, publish zero attention/core scratch,
and use expert ID `u32::MAX` with zero weight. Sticky routing status excludes
their expert tasks; the completion check examines live rows only. There are no
floating-point atomics. Integer routing histograms do not determine FP reduction
order. One mapped status check covers all 48 MoE layers at step completion.

Graph arguments name the pool table, slot vector, model weights and graph-owned
scratch. They retain no session-owned pointer constants. QSA grids and scratch
are sized by pool capacity, while descriptors carry each chat's actual past and
stride. Growing the pool capacity destroys its graphs before replacing scratch;
reset keeps state addresses and clears their histories and device position.
The graph is destroyed before its table, weights and scratch allocations.

Embedding/lexical I/O and uploads precede replay. Full-logit/result publication
and the existing sampler follow replay. On one Spark the graphs capture PLE and
all 48 layer traversals. Final mixer, head and device position commit end the
graphs at 16 rows or with no head bounds. With head bounds at up to 8 rows they
run behind the graphs as eager launches: a full cohort whose reader takes the ids
only uses the bounded head, and every other reader uses the whole head. On a
layer chain each rank's graphs capture its own layers, and the front runs mixer,
whole head and position commit in a third graph after the rows return. Custom Dense hooks and
debug dumps keep the same eager arithmetic. Prompt tiles still start only on
absolute 64-token boundaries; this path has no snapshot API.

Admission conservatively funds all five graph widths, grouped scratch, QSA
scratch, and the shared dense input arena. It may admit fewer long-context chats
than a tighter shared-pool estimate; it never assumes sixteen full contexts fit.
The native example's `--check` compares eager chunks with pooled single rows,
all widths including padding, slot reuse after retirement, and reversed order.
GPU execution is still required to establish full-logit equality and graph
capture/replay correctness; host tests and PTX compilation do not prove those.

The compact layout must pass complete C1/C2/C8/C16/reversed-cohort GPU bytes
against the padded K16 candidate, then improve C8 step time by at least 1%.
The preceding arithmetic-alignment revision retains that padded candidate for the separately owned A/B;
remove the compact-layout commit if the measured gain is smaller. Host checks
establish compact bounds, inverse-map ownership, tail masks and ordered finish
invariance; no Mac result establishes the C8 retention threshold.

Unmasked prompt expert prefill above sixteen rows always uses the M32/N128/K64
tensor-core kernels. Each M128 task executes four M32 planes with ascending K16
accumulation and unchanged BF16 boundaries. Small cohorts and masked decode keep
compact kernels. The tiled PLE schedule is unconditional.
`NextSession::prefill_batch` shares up to the funded chunk limit from at most eight live
prompts at a time; a sixteen-chat cohort uses two groups. Each chat retains its
own causal histories and final logits. Existing GDN/QSA/HC/PLE work is reused;
PLE reuses its existing 64-row gate/convolution path and canonical projections;
lexical positional reads run through the cached prompt reader.

The default prompt frontend independently caches immutable PLE/embedding
row descriptors and open checkpoint files at model load. Prompt tiles reuse
host staging and upload residual/lexical activations once each, before layer 0.
This removes per-token tensor-plan scans, file opens and activation uploads;
bounded positional reads and the existing CPU dequantizer remain. Pooled
single-token decode uses device tables; kernel arithmetic is unchanged.
At 512 rows it adds 26,214,400 host staging bytes and reserves a 32 MiB pinned
upload extent (replacing the initial 4 MiB extent if it has not already grown).
The tiled PLE schedule is independent of the expert dispatch. Integration
gates retain the native exact reference and frozen row-output goldens.

`flashnext_prefill --help` describes the native receipt example. Its optional
layer observer streams every residual output to new local files and disables
graph replay while observing; turn it off for timings. Full checkpoint bytes,
CPU greedy/normwise gates, physical register spills and TTFT still require GPU
qualification. The Mac PTX compiler alone proves no numerical or speed result.

## Integrated candidate selection

The exact `flash::next` reference and row product have separate entry points.
The three parsed controls are FAST_PROJ, FAST_FRONTEND and
FP4_EP2 (all prefixed `VADE_FLASHNEXT_`). Rows are the default HTTP service;
the BATCHED_ROWS switch and former M=1 service are deleted. Row execution
rejects FAST_PROJ/FAST_FRONTEND. Device row
frontend, compact readback, cached prompt uploads and pinned PLE staging are defaults.
Prompt BF16 input/route fusion is always selected above its 64-row threshold;
it composes with the existing expert TC tiles. Chunks accept 1 through the
admitted limit described above.

Unmasked expert prefill above sixteen rows always uses the GPU-qualified
M32/N128/K64 W4A16 tensor-core tile from 1f4d01d. There is no TC switch.
The replaced variable-M expert prefill kernels are removed; small cohorts
keep compact kernels and `flash::next` keeps the exact reference path.
Tiled PLE and wide expert prefill are always enabled. Set
`FLASH_NVFP4=1` for the mixed checkpoint; multi-rank HTTP uses `next-http`
with `NEXT_TP_HTTP=1`. FP8/35B/122B retain pinned main arithmetic.

After an execution error, reset is allowed only while every GDN/QSA owner
remains present. A missing owner permanently rejects reset and execution:
replace the session, whose replacement republishes the complete slot record.
No reset reconstructs storage behind captured addresses. Compact readback
uses the result source returned by execution, even with a changing indexer.

Exactness means identical bytes on the same machine and circumstances.
Rank-order expert reduction fixes bytes per world size; two- and four-Spark
runs use separate goldens. The cross-world test checks placement, token
agreement and a numerical tolerance, and makes no cross-world byte claim.

Mixed ModelOpt metadata alone permits 500,000 tensors. Nonmixed typed FP8
keeps main's 320,000 ceiling; the Flash FP8 index keeps its 200,000 limit.
Legacy nonmixed Flash NVFP4 retains main's 320,000 limit in both loader and receipt.

MTP retains one recurrent entry snapshot, projection tapes and small
PLE/hash/hidden prefixes. Verification scans every causal GDN row in one
pass. A partial accept rebuilds the accepted history from the tape in one
launch and, for a short window over a chunk end whose accept stops before
the chunk end, copies the accepted tail slots. Only
a long window over a chunk end replays the accepted GDN projection prefix.
`mtp_depth:0` requests plain. `--mtp` sets the request default, which is 3 for
one-Spark Next and the pipeline service. The
[draft settings](../crates/engine/src/flash/draft_stack.rs) keep depth 7 resident
by default on these routes. For a chat running alone, omitted request depth or
depth 3 allows the adaptive chooser to use that resident ceiling.
Request depths 1 and 2 keep their own limits. Cohorts keep the request depth,
subject to the row budget. `VADE_MTP_ADAPT=0` disables adaptive depth. See the
[request setup and depth selection](../crates/serve/src/flashnext.rs).
Native rank Next keeps plain decoding, as defined
in the [startup settings](../crates/tp/src/flash/settings.rs).

Prompt lookup uses exact 8-token keys to copy up to 15 proposed tokens, then
verifies every proposal against the target. A miss falls back to the configured
MTP depth. Its admission increment funds GDN projection tapes, small prefixes,
replacement verification logits and a bounded host index. The resident MTP
reservation already funds the single recurrent snapshot; there are no fifteen
full GDN snapshots. [`VADE_MTP_LOOKUP=0`](../crates/engine/src/flash/next_lookup.rs)
turns lookup off. Unset or any other value leaves it enabled. There is no separate
HTTP lookup selector. The integration measurement
wrapper retains an MTP-only oracle for byte and speed comparisons. Existing
`mtp_depth:0` / `draft:"plain"` controls select plain output.
The integration handoff qualifies the composed implementation through HTTP;
earlier isolated measurements do not replace that gate.

The [launcher cohort policy](../crates/serve/src/mtp_batch.rs) selects `Auto8`
on one Spark and on the two-Spark tensor split. Greedy cohorts of one through
eight chats can draft. The cost controller can choose plain steps. One Spark
defaults to a 16-row verify budget. The tensor split enables wide ticks by
default, so five through eight chats can share up to 32 verification rows.
`VADE_MTP_COHORT=off` restores its 16-row controller unless `VADE_VERIFY_ROWS`
overrides that budget. The [request parser](../crates/serve/src/protocol.rs)
allows lookup for greedy requests with drafting enabled, nonzero depth and
`draft` other than `"plain"`. Cohort size does not clear that request setting.
Every proposed token is target-verified. Verification scans GDN once for all
rows, including lookup rows. A rejected prefix keeps the verify rows and
rebuilds its history in one launch. Only long lookup windows over a chunk end replay
retained GDN projections.

The [HTTP scheduler](../crates/serve/src/scheduler/dispatch.rs) uses its configured
prompt chunk while other chats decode. Otherwise it can use the
[adapter's `prefill_chunk_size`](../crates/serve/src/flashnext.rs), which follows
the admitted workspace and the route's prompt policy.
The long-context C8 receipt measured chunk shrinking but supplied no measured
decode-latency reason to preserve it. Existing layer-boundary yielding remains.
The active cap is a ceiling: memory admission can queue peers (seven 32K chats
were measured with the prior pool); this change does not enlarge that pool.

`VADE_PROMPT_COLD8K_R2` is parsed once at startup: unset or `1` enables the
exact prompt kernel bundle, and `0` restores the preceding kernels and scratch
allocation. Other values are startup errors. The bundle hoists bounded dense
copy addresses for matrices with 8192–32768 rows, emits the existing BF16 expert operand from the HC epilogue,
and uses fixed routing segments with validated packed IDs. It retains every canonical reduction and rounding
boundary. Unsupported shapes use the preceding entries. The effective switch
value is included in pipeline rank identity checks.
