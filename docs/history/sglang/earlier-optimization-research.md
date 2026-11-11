# Earlier optimization research

Curated on 2026-10-12 from the September 24–28 audit and breakthrough dossiers.
These describe earlier SGLang deployments: official Qwen FP8 on three Sparks
and GLM on four. They are historical hypotheses and recorded corrections, not
current Vade measurements, qualification, or permission to change precision,
model, hardware, dependencies or defaults. The assigned current policy is
kept privately outside this repository.

## Measure the workload the user actually runs

The early scheduling review found admission decisions based on original prompt
length even when a prefix could be cached. Its short synthetic main-chat test
did not establish responsiveness after tens of thousands of history tokens.
The reusable experiment is a warmed long main conversation receiving a short
turn while background agents prefill fresh prompts. Record actual reusable
tokens, first useful output, stream pauses and background task completion;
matching text alone does not establish reusable hybrid recurrent state.

Cold reading, cached turns, raw generation, complete coding tasks and concurrent
throughput answer different questions. Fix prompts, output lengths, sampling,
thinking, cache state and token-count policy before comparing. Separate thinking
and visible tokens when that distinction affects the task. Check structured
tools, cancellation resource release, queue drain, cache eviction/resume and
idle-then-resume. A health endpoint or short math screen does not establish
these behaviors.

The early audit identified summed kernel times, prefill included in a purported
decode profile, rank-arrival delays counted as communication, and poorly aligned
cross-host clocks. These cannot establish a transport bottleneck. Profile warmed
decode/draft/verification at observed batch shapes; compare profiler-on and
profiler-off runs. Count overlapping work once and use client elapsed time for
the final rate. Performance screens select candidates; they do not replace the
current exactness, quality and repeatability gates.

## Startup: distinguish preparation from physical I/O

The retained September loading logs showed a main loading phase of several
minutes. Visiting all 131 checkpoint shards did not prove every byte was read
from storage on every rank: mapped slices, page-cache hits, ignored tensors,
sharding and postprocessing affect the work. Checkpoint size divided by loading
time was therefore not a physical disk-bandwidth measurement.

The pinned loader already used eight threads. The proposed investigation was
to measure phase timings, physical reads, CPU, memory pressure and first correct
answer; retain compatible compiler caches; and test bounded loader settings.
Prepared per-rank target and draft caches needed complete tensor/scale/alias/
auxiliary-state coverage, exact values/dtypes, layout-bound manifests and a real
reload. A rank can depend on files bearing another rank's label, so file naming
alone was not a sufficient copy plan. Preserve the official checkpoint as the
provenance and rollback source.

The later note recorded a modest first-answer startup improvement with compiler
cache reuse and shorter polling. It did not distinguish loader-thread changes
from noise, and the inspected prepared-weight loads had not established a speed
benefit. Do not turn the earlier recommendation into an achieved loading claim.

## Speculation: useful tokens and total cost must improve together

Let `A` be mean committed output tokens per round, including correction/bonus,
and `T` be the complete draft–verify–commit duration in milliseconds:

`tokens per second = 1000 * A / T`

Accepted continuation routes omit rejected speculative suffixes. Grouping those
routes into artificial verification blocks cannot prove actual expert traffic,
DRAM utilization or a hardware ceiling. Wider verification can read more
experts; additional positions are not free. A perfect-proposal replay is an
optimistic screen for its particular workload, rather than a bound on every
speculative method. Time candidate-shaped drafting separately and include
controlled wrong suffixes and actual verification routes.

Historical planning worksheets used workload-specific proxies and hypothetical
draft-cache geometry. Those calculations do not describe current Vade MTP.
A bounded draft allocator needs prefix-hit, eviction, rejected-state and
graph-safe allocation checks and must not truncate target context.


The ranked research paths were parallel proposal generation, bounded reuse of
rejected target features, native stochastic coupling, exact indexer sharding,
lossless representation inside consuming kernels, and measured host/graph-entry
work. Stop a candidate when acceptance gains fail to exceed its extra total
cost, memory headroom is inadequate, or correctness fails. Sampling fallback to
greedy was explicitly a failed sampled candidate. Fixed seeds alone do not prove
distributional correctness. A proposer checkpoint or licensing exception still
needed a separate owner decision; this import supplies no such authorization.

For indexer partitioning, a candidate outside a partition's top-k cannot reach
global top-k under the same total order. Floating-point order, tie-breaking,
causal/padding masks and index expansion still need equivalence checks. Lossless
weight storage also does not establish bit-identical arithmetic after fusion.
Small component savings cannot be multiplied as though they were independent.

## Attribution and retained evidence

The original dossiers credited community recipes for draft-depth control, host
synchronization, affinity and workload profiling. Their low-bit weights/KV,
different models/rank counts and estimated stream-overlap denominators were not
controlled comparisons with official FP8. No community implementation is copied
here. The research completion audit certified a bounded investigation, not a
cluster speed multiplier.

The dossiers' primary references remain leads: [Carryover Drafting](https://arxiv.org/abs/2609.14717),
[H-Spec](https://arxiv.org/abs/2609.24197), [Block Verification](https://arxiv.org/abs/2403.10444)
and [ZipServ](https://arxiv.org/abs/2603.17435). This curation did not rerun their
results or establish compatibility with Vade.

[Source hashes and selections](github-home-source-manifest.json) identify the
retained local originals. Full audits, journals, traces, overlays, raw results
and source snapshots stay in the private archive. The corrected
[historical promotion record](exactness-decision-2026-10-01.md) and
[protocol corrections](README.md) retain their own narrower evidence boundaries.
