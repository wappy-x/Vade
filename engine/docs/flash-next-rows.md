# Flash-Next rows

`NextSession` has one execution path for a prompt tile or one token from each
of up to sixteen independent chats. Prompts default to 2048 tokens per tile;
`set_chunk_size(1..=2048)` changes scheduling only. Multi-token tiles start at
absolute positions divisible by 64; unaligned continuations use single rows. `step` is the same path with
one row, including a cold sequence. Sparse QSA is the default indexer.

The world-one `flash-serve` bootstrap selects mixed checkpoints before creating
rank transport. It uses `spark_serve::flashnext::FlashNext`, the existing HTTP
scheduler and sampler, 2048-token prompt slices and up to sixteen memory-admitted chats.
Strict greedy MTP defaults to k=3 with prompt lookup; requests may set `mtp_depth` to 0..=3.
World sizes above one are refused. Session reservations include
FP32 KV, recurrent matrices, funded prompt scratch, norms, head and PLE. Native
`session_reservation_bytes` covers all 2048 rows. HTTP uses
`session_reservation_bytes_for_chunk(context, 2048)` and `session_with_chunk`
to cap each session at its admitted 2048 rows. Sixteen
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
  Eager prompt spans project only their last token; the single decode graph also
  computes the head for advance-only calls. Debug dumps use eager one-token slices.
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
and the existing sampler follow replay. The graph captures PLE, all 48 layer
traversals, final mixer, head and device position commit. Custom Dense hooks and
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
`NextSession::prefill_batch` shares up to the funded chunk limit (at most 2048 rows) from at most eight live
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
it composes with the existing expert TC tiles. Chunks accept 1..=2048 rows.

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

MTP retains one recurrent entry snapshot, projection tapes and small PLE/hash/
hidden prefixes. Verification scans every causal GDN row in one pass. Rollback
selects the entry owner and replays only the accepted GDN projection prefix.
`mtp_depth:0` requests plain; k=1..3 selects the MTP proposal limit. The server
keeps depth3 resident. `--mtp` sets the request default; omitted world-one Next
uses k3, following the supplied lookup tuning report's best-default line. FP8
keeps k2 and rank Next keeps plain. The existing single-chat cohort gate applies
regardless of an explicit positive request depth.

Prompt lookup uses exact 8-token keys to copy up to 15 proposed tokens, then
verifies every proposal against the target. A miss falls back to the configured
MTP depth. Its admission increment funds GDN projection tapes, small prefixes,
replacement verification logits and a bounded host index. The resident MTP
reservation already funds the single recurrent snapshot; there are no fifteen
full GDN snapshots. There is no HTTP or environment lookup selector. The integration measurement
wrapper retains an MTP-only oracle for byte and speed comparisons. Existing
`mtp_depth:0` / `draft:"plain"` controls select plain output.
The integration handoff qualifies the composed implementation through HTTP;
earlier isolated measurements do not replace that gate.

World-one strict greedy serving defaults to MTP k3 plus prompt lookup with at
most 16 verification rows, only while exactly one chat is active. Every proposed
token is target-verified. Cohorts suspend both lookup and MTP and resume when one
chat remains. Verification scans GDN once for all rows, including lookup rows;
rejected prefixes replay only retained GDN projections. No arithmetic switch.

HTTP keeps its configured 2048-row prompt chunk when other chats decode.
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
