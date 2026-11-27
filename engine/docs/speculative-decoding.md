# Speculative decoding

**Vade guesses the next few tokens cheaply, checks them all in one pass, and
keeps only what the model itself would have written.** For greedy requests
the answer is the same as without guessing.

Reading the weights is the slow part of writing an answer, and checking
several guesses costs about the same weight read as writing one token. Wrong
guesses still cost their share of the check, so Vade measures whether
guessing pays.

This page covers how it works, which models do it, and how to measure it,
for developers and anyone tuning speed. Vade's flags call it drafting.

<details>
<summary><b>Terms on this page</b>: target and draft model, guess, verification, row, acceptance, greedy, logits, cohort, MTP head, MMA, CUDA graph</summary>

<dl>
<dt>Token</dt>
<dd>The unit a model reads and writes, often part of a word.</dd>
<dt>Target model</dt>
<dd>The model that answers; its choices decide the output.</dd>
<dt>Draft model</dt>
<dd>A small official model that proposes tokens for the target.</dd>
<dt>Guess, proposal</dt>
<dd>A token suggested before the target has checked it.</dd>
<dt>Verification</dt>
<dd>Reading the guesses in one pass and deciding which to keep.</dd>
<dt>Row</dt>
<dd>One token position processed in a pass.</dd>
<dt>Acceptance</dt>
<dd>How many guesses the target keeps.</dd>
<dt>Greedy</dt>
<dd>Picking the top-scoring token; ties go to the lowest token ID.</dd>
<dt>Logits</dt>
<dd>The model's raw score for every possible next token.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Cohort</dt>
<dd>The group of chats that decode together in one step.</dd>
<dt>MTP head</dt>
<dd>Multi-token prediction layer in some checkpoints, for guessing.</dd>
<dt>MMA</dt>
<dd>The GPU's tensor-core matrix multiply-accumulate instruction.</dd>
<dt>CUDA graph</dt>
<dd>A recorded sequence of kernel launches, replayed with one call.</dd>
</dl>

KiB means 1,024 bytes. GiB means 1,073,741,824 bytes.

</details>

## How it works

**Guess, check, keep the part that matches, roll back the rest.**

1. **Guess.** A cheap source proposes a short chain of next tokens.
2. **Check.** The target reads its pending token plus the guesses in one
   pass, in its own canonical arithmetic.
3. **Keep.** Only the part that matches the target's own greedy choices
   commits. Everything after the first mismatch is dropped.
4. **Roll back.** The rejected tokens' KV cache entries are truncated before
   the next cycle. Every token Vade returns already has committed KV.

Output limits, stop tokens and explicit `plain` requests still apply.

<details>
<summary><b>Precision and exactness</b>: one arithmetic order; what stays opt-in</summary>

- **One arithmetic order.** Plain decoding, STRICT verification and shared
  draft-model decoding use the same native BF16 MMA order.
- **Checkpoint precision.** Qwen3 uses BF16 operands and a BF16 KV cache with
  FP32 sums, without splitting activations into high and residual parts.
  Anything below the checkpoint's precision is opt-in.
- **Flash projections.** Covered BF16 projections always use BF16 high and
  residual operands on tensor cores with FP32 sums, including greedy MTP
  verification. This is Flash's one projection mode. Repeatability is for the
  same machine, build and setup; answers may differ from older builds.
- **FAST.** Draft rows in the HTTP server use exact shared verification in
  both STRICT and FAST. The standalone FAST path uses the approximate
  `forward_batch` with a logit and probability guard plus exact repair. Its
  accepted approximate KV can drift over a long prefix. That trade-off stays
  opt-in.
- **Ties and sampling.** Greedy ties choose the lowest token ID. Seeded
  sampling is keyed by absolute position.
- **Official agreement.** Acceptance follows Vade's canonical target choice.
  [Validation](validation.md) records the remaining official-model
  disagreements.

</details>

## Modes

**The default, `auto`, falls back to plain decoding when guessing does not
pay. `fast` and `sampled` are explicit opt-ins.**

<dl>
<dt><code>auto</code> (default)</dt>
<dd>Strict verification, sized by measured cost and acceptance.</dd>
<dt><code>strict</code></dt>
<dd>Canonical target verification; draft sources depend on the model.</dd>
<dt><code>plain</code></dt>
<dd>Never guess.</dd>
<dt><code>fast</code></dt>
<dd>Experimental, with a numerical guard; can change the output.</dd>
<dt><code>sampled</code></dt>
<dd>The draft head samples its proposals from its filtered row. The target
keeps its own draw, so host tests give the same tokens as with drafting off.</dd>
</dl>

- A request selects a mode with the `"draft"` field.
  `spark-serve --draft MODE` sets the server default.
- As the server default, `--draft plain` also skips loading any draft model.
- `fast` is not part of the release checks, and `spark-serve --draft fast`
  is refused for Qwen3.8-27B and Gemma 4 31B.

### Flash MTP sampled verification

Flash keeps Strict as its default verifier. With MTP enabled, set
`FLASH_MTP_SAMPLED=1` on **every rank before launch**, or select
`"draft":"sampled"` in a request to either HTTP generation endpoint. Startup
refuses ranks that disagree on the environment setting. An explicit request
`"draft":"strict"` or `"draft":"plain"` overrides the launch default.
The environment flag selects verification; enable the head separately with
`flash-serve ... --mtp 3`.

For nonzero temperature, the MTP head proposes each token with the request's
own draw: the same seed, position and filters (temperature, top-k, top-p,
min-p and token restrictions) as the target, applied to the head's row. The
target then makes its own draw from each canonical row, as in Strict decoding.
It keeps a proposal only if its draw is the same token, and it stops at the
first one that is different. When the head's row equals the target's row,
every proposal survives (host test `sampled_drafts_commit_the_plain_ids`). The
speed of this mode on a Spark is not measured.

In host tests, the tokens are the same as with drafting off for the same
seed. They do not depend on the proposals, the row count or whether another
chat holds the drafter. A Spark run has not checked this mode. Temperature zero always selects Strict. Targets without filtered
draft distributions use single canonical rows. Strict and Fast behavior is
unchanged.

## Rules for each model

**The dense Qwen3 models draft for independent chats, Qwen3.8-27B and Gemma 4
31B for one active chat, and Qwen3-30B-A3B never.**

| Model | `auto` drafts for | Guesses from |
| --- | --- | --- |
| Qwen3-8B, Qwen3-32B | Independent chats | Prompt, `Qwen3-0.6B` |
| Qwen3-30B-A3B | Never | None |
| Qwen3.8-27B (BF16) | One active chat | `Qwen3.5-0.8B`, MTP |
| Qwen3.8-27B-FP8 | One active chat | `Qwen3.5-0.8B` |
| Gemma 4 31B | One active chat | 31B assistant |

Prompt means the conversation's own text; MTP is the checkpoint's built-in
multi-token prediction head.

A guess commits only after target verification.

<details>
<summary><b>Each model in detail</b>: when it drafts, and where its guesses come from</summary>

- **Qwen3-8B, Qwen3-32B.** Adaptive strict verification; long shared-prefix
  cohorts decode plainly. Guesses come from the conversation's own text, plus
  the official `Qwen3-0.6B` draft model when installed.
- **Qwen3-30B-A3B.** Always plain, even if draft weights are loaded:
  verification was slower than plain decoding on this expert model.
- **Qwen3.8-27B.** Larger cohorts decode plainly. Guesses come from the
  official `Qwen3.5-0.8B` sibling; without it, BF16 uses the checkpoint's
  built-in MTP head. The built-in FP8 MTP draft does not start yet.
- **Gemma 4 31B.** Larger cohorts decode plainly. Guesses come from Google's
  official 31B assistant, when installed.

</details>

<details>
<summary><b>Where guesses come from</b>: the conversation, recycled predictions, a draft model, recent answers</summary>

- **The conversation itself.** Earlier text in the request often repeats, as
  in code edits. Lookup needs no extra model.
- **Recycled predictions.** The target's own top choices from earlier rows,
  including rejected ones, are kept as future guesses.
- **A draft model.** A small official model with the same tokenizer guesses
  ahead. The target still checks every guess.
- **Recent answers.** The server keeps up to 16 recent committed
  conversations, at most 65,536 token IDs in all, to seed later requests with
  the same prompt. They are guesses only. By default they are scoped to the
  request's session capability; `--draft-global-bank` explicitly shares them
  across clients.

</details>

<details>
<summary><b>Automatic choices</b>: how far to guess, why long shared prompts decode plainly, default depths</summary>

**How far to guess**

- Each request's acceptance and the measured cost of the target and draft
  choose between one canonical row and a longer chain of guesses.
- Low acceptance falls back to one row, with an occasional extra guess to
  notice when guessing starts to pay again.
- In shared greedy cohorts, spare target rows go to draft depths already
  chosen by other chats. The largest remaining replies use those rows first.
  This adds no extra draft-model pass. When every chat selects one row, the
  batch stays plain.

**Long shared prompts decode plainly**

With prompt caching, the scheduler looks for at least 1,024 common prompt
tokens among chats in the same cache scope, including queued ones, before the
prefix owner starts decoding.

- When it finds them, those chats use plain decoding until the request finishes.
- Reason: verification would turn off cascade attention (reading a shared
  prefix once for many chats), and an empty draft would replay the cached
  prefix.
- Explicit draft modes override this choice. Short independent cohorts keep
  adaptive guessing.

**Default depth and row budgets.** These are the Qwen3 server defaults.

| Case | Draft depth K | Background rows |
| --- | ---: | ---: |
| Qwen3-8B, with a draft model | 2 | 48 |
| Qwen3-32B, with a draft model | 3 | 64 |
| Dense, without a draft model | Not used | 32 |
| Qwen3-30B-A3B | 8, unused | 32 |

- **The cases.** Qwen3-8B stands for a small dense target with a draft model.
  Qwen3-32B stands for a dense target reading at least 32 GiB of weights per
  pass, with a draft model. The expert model's depth of 8 is unused: the
  server never drafts for this model.
- With a draft model, row budgets are capped by active chat slots times K+1.
- `--draft-k K` sets the depth ceiling, from 1 to 8.
- Flags on the command line are explicit and stay fixed. A model swap
  recomputes the automatic defaults.

</details>

## Hybrid model drafts

**Qwen3.8-27B drafts for one active chat only, with the official
`Qwen3.5-0.8B` sibling or, on BF16, its own built-in MTP head.** It mixes ordinary
attention with recurrent linear attention; larger cohorts use shared plain
decoding.

<details>
<summary><b>Discovery, the built-in head, FP8, depth and verification</b></summary>

- **Discovery.** Automatic drafting looks for the official `Qwen3.5-0.8B`
  folder next to the model. It checks vocabulary, byte-pair merges and special
  tokens for compatibility first.
- **Built-in head.** Without that sibling, BF16 uses the checkpoint's built-in
  MTP head. `--mtp` selects it explicitly. `--draft plain` turns automatic
  drafting off.
- **FP8.** The FP8 checkpoint's built-in MTP draft does not start yet.
- **Depth.** Built-in hybrid MTP defaults to gated ceiling 3 at world 1, or
  fixed ceiling 2 at world > 1; copy stays enabled. `--draft-depth 0` selects adaptive depth
  (the typed MoE head caps it at 3); explicit depths 1 to 4 set its ceiling.
  `--draft-k K` overrides the default ceiling (built-in capacity is 4); if both
  flags are explicit, the smaller ceiling wins.
- **Verification.** The target checks proposals in its canonical arithmetic
  before committing any output.

Canonical target acceptance establishes the same output as plain decoding in
the measured cases. It does not establish that the MTP head matches the
maker's own forward pass.

</details>

## Flash-Next sampled MTP

`flash-serve --mtp 3` sets the request's default draft depth. It does not enable
sampled drafting. The [sampled drafting switch](../crates/serve/src/mtp_batch.rs)
defaults to off on one Spark and on the two-Spark tensor split.
`VADE_MTP_SAMPLED=1` allows a sampled chat to draft while it runs alone.
`VADE_MTP_SAMPLED=2` also allows sampled chats to draft in cohorts. The
[scheduler](../crates/serve/src/scheduler/dispatch.rs) keeps sampled cohort rows
plain unless that switch is `2`.

Greedy cohorts can draft by default on both routes. The launcher selects
`Auto8`, which allows drafting with one through eight active chats. The cost
controller can choose plain steps. On the tensor split, five through eight
chats can share a verify pass of up to 32 rows by default. See the
[cohort policy](../crates/serve/src/mtp_batch.rs) and
[row budget](../crates/serve/src/flashnext.rs).

For sampled drafting, the target verifier samples its own canonical logits
with the request's original seed, absolute position, and committed history.
A rejected guess consumes no random state. Draft argmax choices never replace
target samples. Remaining output and context bound the proposal count.
For a chat running alone, the adaptive chooser can exceed request depth 3,
up to the default resident ceiling of 7. Request depths 1 and 2 keep their limits;
cohorts keep the request depth, subject to the row budget. See
[depth selection](../crates/serve/src/flashnext.rs) and
[resident settings](../crates/engine/src/flash/draft_stack.rs).
`draft: "plain"` disables proposals for an individual request.
Sampling can lower draft acceptance enough to make plain decoding faster;
measure the intended workload with its actual sampling settings.

## Gemma assistant

**Gemma 4 31B drafts with Google's official 31B assistant, for one active
chat only.** [Validation](validation.md#qwen38-and-gemma-scope) states what
its gate covers.

<details>
<summary><b>Discovery, how it guesses, official weights, and the gate</b></summary>

- **Discovery.** Google's official 31B assistant is found in a sibling folder
  named after the target folder plus `-assistant`. `--draft-model PATH`
  selects it explicitly.
- **How it guesses.** It reads the target's last local and global KV, combines
  scaled embeddings with the normalized hidden state, and proposes tokens for
  strict target verification.
- **Official weights only.** It uses the full tied output head. It does not use
  a pruned vocabulary or extra trained weights from another project.
- **One chat.** Only one active chat drafts. Larger cohorts decode plainly.

The merged gate matched 96 saved official token IDs, matched plain decoding
on those 96 IDs with the assistant on, and matched 16 saved long-prompt IDs.
[Validation](validation.md#qwen38-and-gemma-scope) states the limits.

</details>

## Measure it

**The standalone `draft-generate` tool runs the drafter outside the server
and reports whether the token IDs matched, the speed and the memory.**

### Do it yourself

1. Set `MODEL_DIR` to the Qwen3 checkpoint folder, then build the kernels
   before the host programs.

   ```sh
   # Run from the checkout root.
   P=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
   M="$MODEL_DIR"
   cargo kernels
   cargo build --release --offline --workspace --exclude spark-rust-kernels
   ```

2. Build the prompt tool and write the prompts.

   ```sh
   rustc --edition=2024 -C panic=abort tools/draft-prompts.rs \
     -L dependency=target/release/deps \
     --extern engine_tokenizer=target/release/libengine_tokenizer.rlib \
     -o target/release/draft-prompts
   target/release/draft-prompts "$M" /tmp/vade-draft-prompts
   ```

3. Run the comparison. `SPARK_KV_BF16` is unset, so this run keeps an FP32 KV
   cache; the server's default is BF16.

   ```sh
   env -u DRAFT_COSTS -u SPARK_KV_BF16 DRAFT_SAVE=/tmp/vade-draft-ids \
     target/release/draft-generate "$P" "$M" /tmp/vade-draft-prompts \
     128 both 42 16 adaptive lookup strict bench
   ```

4. Run the existing gates. The second needs a running server at `$ADDRESS`.

   ```sh
   target/release/batch-bench "$P" "$M" 2048 0 verify
   target/release/serve-draft-check "$ADDRESS" /tmp/vade-draft-prompts 128
   ```

### Ask your coding agent

```text
Measure speculative decoding for the Qwen3 model in "$MODEL_DIR" on this DGX Spark,
following the "Measure it" section of docs/speculative-decoding.md. Stop any
running Vade server first so the GPU is idle. Run as the current user and
never use sudo. Build the kernels before the host programs. Report every
CURVE, RESULT and DRAFTER line, whether the token IDs matched, and the peak
memory bytes. Note that this standalone run uses an FP32 KV cache. Then run
the batch-bench verify gate, and the serve-draft-check gate against a
spark-serve started on a loopback address. Do not commit or push.
```

<details>
<summary><b>Read the output</b>: CURVE, RESULT and DRAFTER lines, saved IDs and memory scopes</summary>

**Setup**

- **Prompts.** They cover prose, code, JSON and quoted summaries. The
  `lookup` argument selects the combined drafter.
- **Calibration.** `strict bench` calibrates once after model load, using the
  first prompt's prefix. Every prompt and sampler reuses that curve.
  `DRAFT_COSTS` skips calibration with five comma-separated milliseconds for
  1, 2, 4, 8 and 16 rows.

**The lines**

- **`CURVE`** reports startup time separately, including the calibration
  prefix prefill.
- **`RESULT`** reports token equality, throughput, committed tokens per cycle
  and memory. It leaves startup calibration out of request decode time and
  throughput. Its setup-inclusive peak device bytes include calibration
  allocations.
- **`DRAFTER`** reports costs and per-source arrays in the order recycling,
  suffix, rejected tail. A source's committed tokens per cycle equal
  `(source_cycles + source_accepted) / source_cycles` when nonzero.

**Comparing runs**

- **Saved IDs** in `DRAFT_SAVE` allow a direct comparison between runs. A
  STRICT mismatch exits with failure.
- **Memory scopes.** `process_lifetime_peak_device_bytes` includes temporary
  shared workspaces and earlier cases in the same process. Host RSS and
  per-session reachable KV have different scopes; do not add them together.

Earlier combined results are kept in the historical
[FP32-cache measurements](history/fp32-cache-measurements.md), with the precision,
hardware and memory scope of each run.

</details>

## For developers

**Engine and scheduler detail, for people changing the code.** Each part
opens on click.

<details>
<summary><b>Engine API</b></summary>

**Creating and stepping**

- `draft::Speculator::new(prompt, prefill_logits, config, sampling)` creates
  request-owned state.
- `step(target, row_cap, stop)` returns one committed cycle. The row cap
  includes the already selected first token: 0 yields, 1 decodes serially,
  and larger caps verify a draft chain. STRICT calls `verify_batch`.
- `prepare_with_target` puts the target's pending token and K proposals into
  one canonical target batch. Only the prefix matching the target's own choices
  commits, including position-keyed samples when the target supports them.
- Rejected target and draft KV truncate together. Accepted bonus tokens catch
  up on the next cycle.

**Shared serving**

- For shared serving, `prepare_exact` or `prepare_exact_filtered` returns a
  `Proposal`. Its input rows can join unrelated sessions in `Multi`.
  `finish_exact` accepts full logits; `finish_greedy` accepts selected IDs
  for plain greedy sampling.
- The scheduler records the shared target call's time with
  `Proposal::record_target_ms`, excluding queueing and other preparation.
- Tokenizer filters apply to proposal construction and final sampling.

**Rollback and resume**

- A rejected suffix is truncated before the next cycle. Failed cycles roll
  back to their entry checkpoint.
- `from_selected` resumes at the committed prefix and pending token.
  Automatic plain mode caps an existing speculator at one row, keeping its
  state.

</details>

<details>
<summary><b>Draft models</b></summary>

**Choosing a draft model**

- **Discovery for dense targets.** Startup finds an official sibling
  `Qwen3-0.6B` whose tokenizer JSON, tokenizer config, vocabulary and merges
  are byte-identical to the target's. `--draft-model CHECKPOINT` overrides
  it; `--draft plain` skips loading it.
- **Expert targets.** Implicit drafting is off because verification was slower
  than plain decoding. The server's expert adapter never returns a draft
  target, even if explicit draft weights are loaded. The choice and reason are
  logged at startup.
- **Depth.** The BF16 draft adapts its proposal count to request-local
  acceptance and measured cost, following the target's pending token.

**Running it**

- **One wait per chain.** Draft chains keep selected IDs on the GPU and return
  all proposals after one synchronization. KV capacity is reserved before the
  chain starts.
- **Shared greedy drafts.** These batch each depth across ready chats, each
  with its own KV. A short bonus catch-up joins the first proposal's weight
  pass, and only its last row needs logits. Selected IDs stay on the GPU
  across depths, including when the cohort shrinks, and return after one wait.
- **Tied weights.** The small model's tied embedding and output weights share
  one packed allocation. Tokenizer file equality is required at load.
- **Sampling** keeps the existing exact lookup path.
- **Expert verification.** Expert models keep grouped chat batching. Explicit
  draft verification uses consecutive single-row calls to keep the gated
  expert arithmetic.

</details>

<details>
<summary><b>Arithmetic order</b></summary>

**The order**

- Single-row decode, plain shared rows and STRICT verification use one native
  BF16 MMA order, including the vocabulary head.
- The K=16 tiles are divided into four fixed contiguous ranges. Boundary `i`
  is `floor(i * tile_count / 4)`.
- Each range accumulates with native FP32 MMA. The final merge is
  `(range0 + range1) + (range2 + range3)`.
- Token and output tile shapes never change this order.

**Tiles and memory**

- Shared rows pack BF16 operands once into existing scratch, then use 16x32,
  16x64, 32x32 or 32x64 tiles. The wider tile assigns the four K ranges to
  separate warps and merges their FP32 partials in shared memory, with no
  global partial storage.
- Weights sit in contiguous 64-row by 64-column blocks. Decode staging stays
  pipelined across range boundaries; only the accumulators reset.
- Narrow O and down tiles use 18,432 shared-memory bytes; the 64-column tile
  uses 30,720. Q, K and V share one canonical decode launch, and packed
  activations use the multiprocessor's cache.
- Single-row decode replicates its A fragment and uses the same instructions.
  No weights are copied.
- Dense target and draft projections keep the gated BF16 tensor-core
  accumulation order for single and batched rows. The narrow packed
  matrix-vector entry point also has four tensor-core column ranges with
  32 KiB of FP32 partial sums; it does not replace the gated dense projections.

</details>

<details>
<summary><b>CUDA graphs</b></summary>

**Capture and replay**

- Plain BF16 decode and draft chains capture their complete kernel sequence
  through the CUDA driver API, then replay it on the existing stream.
- A 16-byte device input holds the first token and absolute position. Later
  draft embeddings read the preceding device argmax.
- Graphs keep the same attention partitions and reductions.
- Growth of the KV cache, attention scratch or argmax drops the affected graphs
  before replacement. Partition changes recapture the active width.
- Uploads and the single result read stay outside capture.

**Layout and tracing**

- Separate graph entry points keep ordinary kernels' original parameter
  layout, with positions passed by value.
- Graphs own no weight copies. Wider chat batches stay on the existing
  shared-weight path.
- GQA8 graph wrappers use the same grouped kernel and adaptive spans as
  multi-chat decode.
- Per-kernel profiling uses Nsight graph-node tracing. Backend statistics
  count graph captures and launches separately.

</details>

<details>
<summary><b>Attention for draft and verification rows</b></summary>

- The draft's two-query-head attention uses 64-key partitions and FP32 summary
  merges through 4,096 positions.
- Two-head BF16 draft rows up to 512 positions fuse scoring, full-row
  normalization, 64-key value partials and their ordered merge in 12 KiB of
  shared memory. Two 128-thread groups compute separate value chunks, then
  merge in ascending chunk order straight into the output. Longer rows keep
  the split kernels.
- Four- and eight-query-head attention shares each K/V load while keeping
  each head's canonical dot product, softmax, value sum and chunk merge order.
- With an FP32 KV cache, long-context verification shares K/V reads across up
  to 16 consecutive query rows. One warp per row computes two keys in parallel
  from shared 64-position K/V tiles. Vectorized scores and values keep the
  serial decoder's 256-position softmax order and split-summary merge. Groups
  split at cache and summary-stride boundaries. Dynamic shared memory is at
  most 98,304 bytes per block, with no extra global device scratch.

</details>

<details>
<summary><b>Lookup sources in detail</b></summary>

**Recycling**

- Keeps one raw top-one successor per vocabulary token. It is seeded from the
  final prefill row and updated from every decode and verification row,
  including rejected rows.
- The serial standalone fallback reads logits to warm the table without
  another model pass. The HTTP greedy ID-only path updates the observed
  successor.
- Short conditional successors also keep 2-, 4- and 8-token contexts from
  verified rows.
- Proposals prefer the longest matching context, then the raw token successor,
  until one is unknown or blocked by vocabulary filtering.
- For Qwen3-8B the table takes 607,744 bytes. It stores no logits or weights.

**Suffix lookup**

- Indexes suffix lengths 1, 2, 4, 8, 16 and 32, with the latest continuation
  for each. Longer suffixes tell long copies apart.
- Entries stay available for the whole request. New tokens are indexed at
  once, including unfinished historical tails.
- At most six entries per history token, with a 24-byte entry payload on
  64-bit hosts plus hash-table overhead.
- Suffix matches start with a stronger prior.

**Rejected tail**

- A third source shifts predictions beyond the accepted prefix into the next
  cycle.
- Lookup and rejected-tail candidates can extend through recycling.

**Recent answers**

- Requests seeded from recent answers are still checked in full:
  `cache:false` still recomputes prefill, and STRICT verifies every proposed
  token.
- When reporting speed, keep first cold requests separate from requests with
  learned continuations.

</details>

<details>
<summary><b>Cost model</b></summary>

**Estimates**

- Adaptive source and width selection estimates committed tokens per
  millisecond, including drafting and recycling time. It learns how often
  guesses survive, by source and depth.
- A measured GB10 curve supplies the starting estimate. Every useful cycle
  updates one moving-average scale shared by all widths, so unused widths
  follow context drift.
- All-zero costs select the context-adjusted GB10 prior. `gb10` restores the
  original short-context measurements. `at_context` applies the measured
  STRICT context slope.
- Requests do not force width-calibration cycles.

**Measuring and reporting**

- `CostCurve::measure(target, tokens, mode)` measures widths 1, 2, 4, 8 and
  16 after one warm-up at sixteen rows. The caller supplies sixteen tokens and
  reserves sixteen temporary KV rows. Every call starts at the same checkpoint
  and rolls back afterwards, including on errors. Timing excludes rollback.
- Store the curve in `Config::costs` and reuse it for the same target
  configuration at a similar context length. Useful generation calls refine
  the costs as context grows.
- `shared_decode` log lines report the actual cumulative target-pass and draft
  wall times. Request-local draft costs charge shared depth time for adaptive
  width selection.

</details>

<details>
<summary><b>First token and snapshots</b></summary>

- HTTP can send the first token straight from the prompt logits, before that
  token's KV commits. The next cycle commits it without sending it again.
- Prompt snapshots keep matching logits.
- Draft completion snapshots drop stale logits, so reusing an exact
  completion prefix recomputes its final row.
- KV snapshots share immutable physical prefixes and reserve private writable
  tails before an append, truncate or replay.

</details>

<details>
<summary><b>Prompt reading alongside decoding</b></summary>

**Chunks and shared passes**

- HTTP session prefill rounds each multiply input to BF16 once and adds up in
  FP32 ([engine](../crates/engine/README.md#prefill-details)).
- `--prefill-chunk` accepts 1 to 512. With the default, a lone long prompt
  uses 4,096-row chunks and 100 ms layer slices. When other work can run, new
  chunks go back to the configured size. A chunk already in flight keeps its
  shape until it commits.
- KV reservation uses separate 512-row physical blocks, so cache snapshots
  share complete pages.
- For Qwen3, prompts or cached suffixes of at most 256 new tokens share a
  tensor-core prefill pass after a 2 ms collection window. A pass holds at
  most 2,048 rows, or 256 while a foreground chat decodes.
- BF16 prompt attention keeps the session's arithmetic while projections
  share weight reads. A session slice in flight keeps its route until the
  chunk commits; a short remaining tail can then join a shared prompt pass.

**Time slices**

- Requests that depend on an owner's prefix let its wide chunks finish
  without pausing between layers.
- Background slices use 100 ms. Foreground slices with other runnable work use
  15 ms. Indivisible layer work and setup can exceed a slice budget.
- The selected foreground decoder runs every tick. Others run every 1 to 4
  ticks, by context length relative to the shortest ready decoder, with an
  8,192-token floor. Equal-length cohorts run every tick.
- Prompt tails and shared rows age after four ticks. Shared exact row budgets
  stay separate from sliced prefill.
- Decode and verification stay exact throughout.

</details>

**Where next:** [Server and API](server.md#speculative-decoding) ·
[Models](models.md#draft-models) · [Validation](validation.md)


## Hybrid copy and confidence proposals

Built-in hybrid MTP (including Qwen 35B/122B) defaults to ceiling 3 at world 1,
`HYBRID_MTP_COPY=1` and `HYBRID_MTP_GATE=1` (g3). At world > 1 the default
is ceiling 2, and the gate is bypassed regardless of its env setting to keep
collective counts rank-consistent; explicit depth env/flags still win. Copy tries an exact
eight-token suffix lookup before the MTP head. The gate stops deeper levels
using confidence and measured request-local costs. Set either variable to `0`
to disable it; the gate runs only at world 1. For fixed depth-2 (d2), use
`HYBRID_MTP_COPY=0 HYBRID_MTP_GATE=0` with
`--mtp --draft strict --draft-depth 2 --draft-k 2`. To disable both proposal
features, set both variables to `0`. Flash-Next MTP defaults are unchanged.
These flags change proposals only. Greedy output still
comes exclusively from the existing target verifier and prefix commit path.

The copy index contains the prompt and committed output tokens. Its query ends
with the already selected next token. A match proposes up to the configured
MTP depth (at most four) from the earlier continuation. It never indexes
unverified tokens or extends a copy beyond the known continuation. A missing
match falls back to the MTP head. A shorter history clears the index; a new
session or cache fork builds its own index from the supplied history.

The singleton-only gate multiplies each level's top-16 softmax share into cumulative confidence.
After three measured MTP cycles, it keeps drafting only while confidence squared
covers measured accepted tokens per millisecond times the marginal verifier-row
and head-level cost. Missing timings retain the configured depth. The initial
level always runs; a low-confidence level prevents the next level. Top-k
reduction time is included in proposal timing, with its scratch reserved and
accounted as draft memory. Copied cycles do not train the head cost estimates.

With either flag enabled, private per-session proposals replace shared head
proposals, including in a cohort. External draft models retain their existing
path. These defaults target singleton greedy hybrid MTP; cohort performance
needs separate qualification. Neither flag enables sampled model drafting.

Host token-stream replay can prove that proposal acceptance/rejection preserves
the recorded target bytes. It cannot prove the GPU verifier's arithmetic or a
speedup. Qualify the actual native build and request set, including long-context
requests, before making a performance claim. Report copy acceptance as accepted proposals divided by all
proposed tokens, including the unused tail after the first mismatch.

Idea credit: the copy-first suffix index and the confidence-squared depth cost
rule were studied in TensorFold. The implementation in
`engine/crates/engine/src/hybrid/draft_control.rs` is Vade's own Rust.
