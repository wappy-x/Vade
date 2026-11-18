# Tensor-core contract of the dense shared layers

The dense shared layers move from scalar row math to the tensor cores. Three
weight formats move, one at a time: plain BF16, block FP8 and MXFP8. Each move
changes the token IDs once (a rebase). This page is the contract a move must
keep. The routed experts (NVFP4) made the same move earlier; their contract is
[`tc_contract.rs`](../crates/gpu/src/tc_contract.rs) and it is the pattern here.

Who writes what: the gate seat owns this page's shape, the rules, the gate and
the rebase record. Each format seat owns its own section and edits only that
section, so three branches merge without a conflict.

## The promise (it does not change with a rebase)

A rebase changes which tokens a request gets. It never changes these six rules.
They hold again after EACH rebase, on the new token IDs.

1. The same tokens on every run of the same request.
2. A row alone has the bits of that row in any group, in any slot, beside any
   neighbors, at any group size.
3. A token read by the prompt path has the bits of the same token made by a
   decode step. So a resumed chat equals a cold read, and a second turn equals
   the second turn read cold.
4. Drafted equals plain.
5. One chat alone equals the same chat among several chats.
6. A word function depends on the weight, the input and the column index only.
   Never on the row count, the tile, the thread count, the kernel entry or the
   rank count.

What follows from rule 3 and rule 6: ALL paths of a matrix switch together.
These are the one-row decode step, small row groups (2 to 8 rows), the prompt
tile, the drafting check rows, and every host reference of that matrix. A
format that is switched in the prompt path and not in decode breaks rule 3.

## Rulings

- **A. One word function a format, for first and second matrices.** Four column
  ranges, cut by the column count only. Each range is one ascending chain of
  tensor-core steps from `+0.0`. The join is `fl(fl(p0 + p1) + fl(p2 + p3))`,
  where `fl` is one IEEE round-to-nearest-even FP32 add, never fused, never
  reordered. Then the epilogue. A format with a measured reason for another
  function on FIRST matrices writes it in its section, with the number. A
  SECOND matrix always keeps the four ranges: the pair split across ranks needs
  them.
- **B. The scale.** Where a weight needs a scale (block FP8, MXFP8), the format
  seat chooses where the scale enters the word. The word must be a fixed
  function by rule 6, must have a CPU twin, and must pass the quality check.
  The function is written in this page BEFORE the kernel.
- **C. Twin first.** No wiring into a served path before the device equals the
  CPU twin on the sweep.
- **D. The one-row decode step** may not get slower by more than 1 percent.
- **E. No other weights.** The checkpoint and its pinned revision stay. No
  lower-precision copy of a weight.

## Terms

- **Word:** one output value, from one weight row and one token row.
- **Word function:** the exact arithmetic of a word, bit for bit.
- **Step:** one tensor-core instruction over a fixed number of columns, which
  takes the carried FP32 accumulator and gives the new one.
- **Step zero rules:** a zero product or zero carried accumulator brings no
  exponent to the alignment. A subnormal operand keeps the exponent field's
  `2^-126`; it is not normalized. A result cut to zero is `+0.0`, including a
  negative sum that underflows. The corrected BF16 twin is on `tc-bf-b`,
  commit `24772b376e` (`tc_twin_bf16::step`). The earlier twin on main is not
  a reference for these cases. Evidence: run `tcbf2-tc-3`, 589,824 short-chain
  words; the old model differs on 4,670 words (red note at 21:25:53).
- **Chain:** steps in ascending column order, from `+0.0`.
- **Range:** a run of columns with one chain. A word has four.
- **Epilogue:** what turns the joined accumulator into the stored word (for
  example one BF16 rounding, or a scale and then the rounding).
- **CPU twin:** an independent integer model of the word function, in
  `engine/crates/gpu/src/`. The device must equal it on hard inputs.
- **First and second matrix:** the role rule of
  [`matrix_role.rs`](../crates/tp/src/matrix_role.rs). A first matrix reads the
  model width. A second matrix writes it back (`down_proj`, `o_proj`,
  `out_proj`).

## What every format section holds

A section is complete when it has all six parts. A part that is not done says
`NOT DONE` and what it needs.

1. **Matrices.** Every matrix of the format, with its role (first or second),
   its shape, and how many layers hold it. Checked against the checkpoint's own
   table, not a note.
2. **Word function.** Numbered steps, exact enough to write the twin from:
   - how a weight becomes a tensor-core operand, and why that is exact;
   - how an activation becomes an operand (one rounding, or more);
   - the step: the instruction and its integer model;
   - the four ranges: the column boundaries as a formula of the column count
     only. A boundary never cuts a step. For a scaled format it never cuts a
     scale block or scale group either;
   - the join, in the order of ruling A;
   - where the scale enters (ruling B);
   - the epilogue;
   - what is outside the contract (for example a NaN scale code).
3. **CPU twin.** The file, the function names, and the hard input classes it is
   tested on (subnormals, signed zeros, cancelling sums, wide exponents).
4. **Kernel entries by row count.** A table: entry, row counts it serves, tile
   shape, state. Every entry gives the same word.
5. **Paths wired, with counters.** One line a path: one-row step, row groups of
   2 to 8, prompt tile, drafting check rows, host references and fallbacks.
   Each path has a counter in the served log, so a run proves the path ran on
   the new word. A path with a zero counter in a served run is not proven.
6. **Proofs.** Run tags and numbers, each MEASURED on a device:
   - the device equals the twin (words checked, words that differ);
   - a row alone equals the row in every group size, slot and neighbor set;
   - a wrong-on-purpose control that fails;
   - the one-row step, old against new (ruling D);
   - the bench, old against new, at 1, 2, 4, 8, 256 and 1,024 rows.

## The gate (what lands a format)

After a rebase the old check "the same token IDs as main" is false by design.
The gate replaces it. It runs on the integration branch, on a pair, on the
served model, with the engine's own token ID lists. A format lands only when
every part passes.

No part of the gate compares a new head with the token choices of an older
main (owner's ruling, 2026-11-16 10:11). The six rules are proven on the new
head's OWN token IDs. Old goldens and baselines stay under their dated names.
They are evidence, never a target.

1. **Twice the same (rule 1).** Two fresh servers, the four loads. Every token
   ID list equal between the two runs. The second run never reuses a warm
   server or the prompt cache of the first.
2. **Alone equals group (rules 2 and 5).** The check tool's row sweeps for
   every switched format, and one chat alone against the same chat among 2, 4
   and 8 chats.
3. **Prompt path equals decode path (rule 3).** Two served checks.
   - The continuation: a prompt plus the first k tokens of its answer, read
     cold, must continue with the rest of that answer. In the first request
     decode steps made the rows of those k tokens. In the second request the
     prompt path reads them. This is the check that puts the two paths side by
     side.
   - The kept second turn: turn 2 with the cache against the same turn 2 read
     cold, and the kept turn must report cached tokens. The prompt cache keeps
     the state at the END OF THE PROMPT, not the answer rows (measured on
     main: a kept 8,192-token prompt reports 8,192 cached tokens). So this
     check compares two prompt reads with different group cuts. It proves the
     keep and resume path and rule 2 of the prompt path. It does not prove
     rule 3 by itself.
4. **Drafted equals plain (rule 4).** The same requests with strict drafting at
   depths 1, 3 and 7 and with plain decoding. The drafting path must have run:
   for each depth some request accepts drafted tokens and some request reaches
   a check pass of depth + 1 rows, and every drafted request has a check pass
   of more than one row. (Measured on main: at depth 1 two of three kinds of
   answer propose one token in 256 and accept none, so depth 1 is a thin
   test; depths 2, 3 and 7 propose hundreds.)
5. **Controls.** For each of parts 1 to 4, a wrong-on-purpose run that the gate
   must catch (for example one path left on the old word). A part whose control
   passes is not a proof.

The gate must pass on main before any format lands.

### Receipts: what was proven, and on which head

A receipt belongs to the head it was measured on. It is never relabelled for
a newer head.

- **Main before any rebase (`1508efce96`): PARTIAL.** Main group and drafting
  checks passed in `a-gate-main-a1-1` and `a-gate-main-a2-1`; all six
  continuation/cache checks and three existing row-tool modes passed in
  `a-gate-main-a3-3`. The tile magnitude control was caught in
  `a-gate-control-tile-1`. A fresh one-chat subset repeated exactly in
  `a-gate-main-b1-1` (three lists, 768 tokens, identical build hashes). The
  complete four-load repeat, small control and quality were NOT RUN there.
  See [`tc-gate-status.md`](tc-gate-status.md) and the detailed
  [`table`](tc-gate-table.md).
- **The frozen candidate on its old base (`d433e86721` on main
  `7e8884b5bd`).** The word checks of both switched formats passed on its
  image `8195faaf6f26`. The complete served repeat and the six served rules
  were NOT RUN there: that night's gate stopped at the old quality bar
  (agreement with main's tokens), which the ruling above retired.
- **Rebase train R1, the merged head (branch `tc-rebase-r1`: the frozen
  candidate merged onto main `27276fcc0f`).** The word checks are MEASURED
  again on the merged kernel image `2b06937bc09d` (runs `r1-mx-words-1` and
  `r1-bf-words-1`, in the two format sections). The served parts 1 to 5 on
  the merged head are the gate seat's. They are NOT PROVEN until the gate
  seat writes them into the rebase record.

No format lands on a partial proof, and none lands on a receipt of another
head.

Token equality alone does not prove rule 2 or rule 6 for all words. The format
must also supply device-to-twin sweeps across entries, row counts, slots and
neighbors. A broad magnitude fault is a smoke control only: it does not prove
sensitivity to one path left on a different sum order. A sum-order control
or an old-entry control must fail a word comparison before landing.

## The quality check

The quality check runs beside the gate. A rebase may not make the answers
worse. It has three parts (owner's ruling, 2026-11-16 10:11).

1. **Word accuracy.** On real captured inputs the new word is as close to
   exact arithmetic (the exact sum, rounded once) as today's word.
2. **The outside reference (the maker's own engine).** On the positions where
   the maker's own gap is 1.0 or more, the new head agrees with the maker at
   least as often as main. And the POOLED count of the long-prompt quality
   job is not lower than main's. The split by band is reported, not gated.
3. **The six rules on the new token IDs, with controls** (the gate above).

A drop in part 1 or part 2 stops the landing.

NOT a bar: the share of positions where the new build picks main's token.
MEASURED on five chains over all 2,080 fixed-history positions (2026-11-16):
exact arithmetic itself changes 6 of main's 1,633 clear choices (gap 2.0 or
more), the tensor-core form 3, and every pair of chains is equally far apart
(median rms 0.62 to 0.65 over main's 20 highest logits). So that share
measures rounding noise, not quality. A comparison with an older main may be
kept as labelled evidence. It is never an acceptance target.

MEASURED for rebase train R1 on the frozen candidate `d433e86721` (parts 1
and 2; the merged head is the gate seat's to measure):
- Part 1: SAME ACCURACY. MXFP8, error in BF16 steps, rms over all 3,182,592
  words of three decode steps on the inputs of three chains: scalar 0.2915 /
  0.2901 / 0.2981, tensor-core 0.2992 / 0.2899 / 0.2981. The ideal word is
  0.2887. The difference sits in under 0.1 percent of the words, the ones
  that cancel heavily.
- Part 2: maker gap 1.0 or more: main 172 of 173, candidate 172 of 173. Long
  prompt pool: main 49,699, candidate 49,752 of 67,518.

## The rebase record

One line a rebase, written by the gate seat when a format lands. Each rebase
has its own baseline of answers in a new numbered folder (rebase train R1:
`baseline-tc/02/`). An old baseline or golden is never overwritten: it stays
under its dated name.

| Rebase | Format | Integration head | Baseline folder | Gate | Quality |
|---|---|---|---|---|---|
| 0 | none (main) | `1508efce96fd732edf93704c95c34b813afcebaf` | `baseline/` | PARTIAL; no landing | NOT RUN |
| R1 | MXFP8 layer matrices and the BF16 latent group | the gate seat writes it at landing | `baseline-tc/02/` | NOT RUN on the merged head | parts 1 and 2 MEASURED on the frozen candidate; merged head NOT RUN |
| R3 (landed 2026-11-17 01:17:48 as train 16) | MXFP8 layer matrices on a new word (scales inside the step), for the prompt read | main `1194b2a8e6c56e0c06e3cb8e4febcc8ed5ac79b2` | `baseline-tc/04/` | train 16; the gate seat's record | train 16; the gate seat's record |

## The image target

Since rebase R3 every kernel image is built for `sm_121a`, the GB10's
tensor-core target (`crates/gpu/ptx_stamp.rs`). Until then the production
image had `sm_121`; only the second image and the FP4 variant had `sm_121a`.

- Why: the MXFP8 word of R3 uses the block-scaled tensor-core step inside
  the production image's own entries (the merged decode launches and the
  prompt tile). The driver refuses that step in an `sm_121` image (error
  218).
- Ruling: lead, 2026-11-16 12:25 (`DAY2-RULES.md` of the lab day), four
  conditions; condition 1 ruled met at 12:42.
- Receipt for condition 1, MEASURED bench run `yard-11` (2026-11-16): the
  pinned offline compiler (`ptxas` 13.0.88, the binary `cubin.rs` names by
  hash) compiled the production image of 665 entries twice, for `sm_121`
  and for `sm_121a`, from the same text with the one target line changed.
  The machine code of 664 of 664 compared kernels is byte for byte equal,
  and so are all constant, info, relocation and shared-data sections; two
  note sections differ (the compiler's own record of command and target).
  One entry was not compared in that run: the offline compiler refuses
  `resident_launch_absorb_attention` for `sm_121` (two 32,768-byte block
  arrays in one entry, limit 49,152); ruled separately at 12:42, fixed in
  its own commit on main (branch `tc-absorb-one-array`). For `sm_121a` the
  compiler accepts the entry as it is (runs `r3-code-1` to `r3-code-4`:
  the whole image compiles, no entry taken out).
- The same receipt on the landed R1 text, MEASURED run `r3-code-4`
  (`kernel-bench --mx-tc <old> <new> code`): the R1 image compiled for
  `sm_121a` against the same text compiled for `sm_121`: 657 kernels, 656
  byte for byte equal, 0 differ, 1 without code under `sm_121` (the refused
  entry; its `sm_121a` code equals the R3 image's).
- MEASURED on the build host: the production image text of R3 built for the
  two targets differs in the target line only (0 other lines).
- What it means for proofs: the switch itself changes no kernel code. The
  kernels to prove again on the new image are the ones R3 edits (the MXFP8
  tensor-core bodies and every entry that holds them); the R3 gate on the
  new image is the token proof.
- A fact to know when reading a code list: the Rust compiler numbers
  unrelated kernels differently when any kernel source changes, so a few
  kernels that no commit touched get other machine code from one commit to
  the next, with the same byte count. MEASURED: run `r3-code-1` listed 4
  such kernels, the next head (`r3-code-2`, one file changed) listed 19
  others and none of the 4, the head on the main with R1 (`r3-code-4`)
  lists 1 (`fp4_prefill_tc_down`), and the same change on the main with R2
  (`r3f-code-r2b`) lists 5 others, three of them with another byte count.
  This happens on every commit to the kernel crate; it is not the target
  (the target alone changes 0 kernels in each of these runs). The list of
  the head that lands gets an old-image-against-new-image check of each
  such kernel.
- The precompiled image (`scripts/make-cubin.sh`, `crates/gpu/cubin.rs`):
  its manifest names the target and the compiler arguments. They change
  with this switch in the same rebase (commit `ffd1976017` of the branch),
  and the compiler binary stays the pinned one. The accepted ELF flags stay
  `0x06007902`: MEASURED, the pinned compiler writes them for this image
  under `sm_121a` too.

## Plain BF16 (seat TCBF)

Status: the latent attention group is switched and proven on a device, and
its one-row decode form passes ruling D in the bench (2026-11-16). It is
merged onto main `27276fcc0f` in rebase train R1 (branch `tc-rebase-r1`); the
served gate on the merged head is NOT DONE. The other plain matrices are NOT DONE
(seat TCBF2 has them; they keep today's word until their own rebase).

This section has two groups. A group switches by itself, with all its paths.

- **Group 1, the latent attention (this section).** The four fold operations of
  the absorbed latent attention: query, scores, values, project. Two of them
  use the plain BF16 matrix `kv_b_proj`. The other two are sums of BF16
  activations against the latent cache and use no weight matrix. They are one
  group because they are one chain of five device calls in every path.
- **Group 2, the plain projections.** NOT DONE (seat TCBF2).

### Matrices

| Matrix | Role | Shape a rank | Layers | Group |
|---|---|---|---|---|
| latent `kv_b_proj`, key rows (`W_K`) | first | 32 heads x 512 x 256 | 11 | 1: query |
| latent `kv_b_proj`, value rows (`W_V`) | first | 32 heads x 256 x 512 | 11 | 1: project |
| (no matrix) scores | - | 32 heads x up to 2,051 selected rows x 512 | 11 | 1 |
| (no matrix) values | - | 32 heads x 512 x up to 2,051 selected rows | 11 | 1 |
| indexer `wq_b`, `wk`, `weights_proj`, `index_kpool_compress_gate` | first | - | 11 each | 2 |
| KDA `f_a_proj`, `f_b_proj`, `b_proj`, `g_a_proj`, `g_b_proj` | first | - | 34 each | 2 |
| router `mlp.gate` | first | - | 42 | 2 |

The list of group 2 is the reading of the review seat from the checkpoint's own
table (`quantized_layers` of `hf_quant_config.json`). Seat TCBF did not read
that file. No plain BF16 matrix is a second matrix in this checkpoint. A
latent layer is admitted to the absorbed path by its tensor role, shape and
dtype ([latent absorption](latent-absorption.md)); every latent layer of this
checkpoint is.

### Word function

Each operation is a matrix product `C[m][n] = sum over k of A[m][k] * B[k][n]`
for one outer index. The table is in
[`absorb_tc.rs`](../crates/gpu/src/absorb_tc.rs).

| Operation | Outer | m | n | k (the sum) | A | B |
|---|---|---|---|---|---|---|
| query | head | latent features | tokens | key features | `W_K` | query rows |
| scores | token | selected slots | heads | latent features | selected cache rows | absorbed query |
| values | token | latent features | heads | selected slots | selected cache rows | probabilities |
| project | head | value features | tokens | latent features | `W_V` | mixed latent |

The word of `C[m][n]`:

1. **Weight operand.** The BF16 pattern of the checkpoint, unchanged. Exact:
   the tensor-core instruction takes BF16.
2. **Activation operand.** The high 16 bits of the FP32 word. There is no
   rounding at the read and no low part, in any path. Every producer writes
   BF16 values (the query projection, the layer norm of the cache rows, the
   softmax, the rounds of query and values), so the read is exact for them. A
   word with a low half gives the word of its high half, in every path.
3. **Step.** `mma.sync m16n8k16 bf16` over 16 terms and the carried FP32
   accumulator. Integer model (`tc_twin_bf16::step`): the 16 products and the
   accumulator are lined up on the largest exponent, bits under 25 fraction
   bits are cut toward zero term by term, the sum is exact and is cut toward
   zero to FP32. With an infinity or a NaN among the inputs the step is the
   IEEE sum (`step_any`).
4. **Terms and steps.** Term `k` is `A[m][k] * B[k][n]`, in ascending `k`. The
   terms fill `steps = ceil(terms / 16)` steps. A short last step is filled
   with `+0.0` operands. For values the terms are the token's selected rows in
   the token's selection order, and `terms` is the token's count.
5. **Ranges.** Range `p` is steps `[steps*p/4, steps*(p+1)/4)`. The boundary
   follows the step count only. It never cuts a step.
6. **Chain.** A range is one ascending chain from `+0.0`. An empty range is
   `+0.0`.
7. **Join.** `fl(fl(p0 + p1) + fl(p2 + p3))`, the order of ruling A.
8. **Epilogue (unchanged from before the switch).** Query, values and project:
   one BF16 round that keeps a NaN a NaN (`latent_absorb::round`). Scores: one
   FP32 multiply by `fl(1 / sqrt(key_dim))`; a slot past the token's count is
   `+0.0`. Project in the prompt form: a token with no selected row gives
   `+0.0`.
9. **Outside the contract.** The payload of a NaN (NaN places must agree).

What a word depends on: its `m` row of A, its `n` column of B, and `k`. For
values `terms` is the count of the token itself, the same in the prompt path
and in the decode step. Never the group size, the block, the warp, the staged
chunk or the kernel entry.

No scale: ruling B does not apply.

### CPU twin

[`tc_twin_bf16.rs`](../crates/gpu/src/tc_twin_bf16.rs): `step`, `step_any`,
`fold` (the joined accumulator for any term count), `high` (the operand of an
FP32 word). The word of one operation: `absorb_tc::word`. Both host references
(`latent_absorb::unit`, `latent_absorb_rows::unit`) call it.

Measured step correction: zero products and a carried zero contribute no
alignment exponent. A subnormal operand retains its exponent-field alignment.
A result cut to zero is positive zero. Zero magnitudes are skipped before shifts.
Receipts: `tcbf-takeover-words-04` and `tcbf2-takeover-ring-02`.

Host tests: the twin's own (7), and a host model of the kernel body
(`absorb_tc/tests.rs`) that walks the kernel's blocks, warps, lanes, staged
chunks and range closes with the twin's step and must give the twin word for
every output of the prompt form and of the step form on its bucket grid.

Hard input classes on the device (`absorb-tc-rows`): values near one, a wide
exponent spread, subnormals and zeros, signed zeros, cancelling sums, mantissas
of ones, general FP32 words, rare infinities and NaNs, shares in [0, 1), and
crossed pairs of classes.

### Kernel entries by row count

Two bodies give the one word. They differ only in where a word is computed:
the block, the warp, and the order in which loads are issued. The host model
(`absorb_tc/tests.rs`) walks both and must give the twin word for every output.

- **Prompt form** (`kernels/src/latent_absorption_tile.rs`). A block is one
  outer index, 128 rows of M and 32 columns of N. A warp owns one 16-row M tile
  for the whole sum and runs the four ranges one after another.
- **Step form** (`kernels/src/latent_absorption.rs`), one token. A block has
  256 threads.
  - Query and project: a block is one head and 128 rows of M. A warp owns one
    M tile and runs the four ranges one after another. It issues the weight
    loads of four steps before it uses any of them, and a hint
    (`prefetch.global.L2`) for the weight lines of the steps two batches
    ahead: in a decode step these weights are not in the cache.
  - Scores and values: a block is 2 M tiles and 32 columns of N. Warp `w` owns
    M tile `w / 4` and range `w % 4`, so the four ranges of a sum run side by
    side. The ranges meet in shared memory in the order of the join: range 1
    joins range 0, range 3 joins range 2, then range 2 joins range 0.
  - Why a second form: a warp of this GPU runs in order and waits for each
    load whose value it uses at once. The prompt form stages its B operand
    for 32 steps and issues the A loads one step at a time. At one row that
    cost 36 microseconds a launch for nothing.

| Entry | Serves | Launch |
|---|---|---|
| `latent_absorption_tile` | prompt groups and chat rows that the third form does not take | `latent_absorb_rows::launch` |
| `latent_weights3`, `latent_scores3`, `latent_values3` | prompt groups and chat rows from 17 token rows, the model's shape (the third form, last subsection) | `latent_absorb_rows::third` |
| `latent_absorption` | one decode row: query, project | `causal_ops::absorb_blocks` (`Form::step_blocks`) |
| `latent_absorption_scores` | one decode row: scores | the same, on the grid of the next 256 rows |
| `latent_absorption_values` | one decode row: values | the same |
| `resident_launch_absorb_dense` | the recorded decode step: query, project | the recorded grid |
| `resident_launch_absorb_attention` | the recorded decode step: scores, values | the recorded grid |

The old scalar bodies are deleted. No entry of the image holds the old word.

### Paths wired, with counters

| Path | Entry | Counter in `SERVED PROMPT COUNTS` |
|---|---|---|
| Prompt tile | `latent_absorption_tile`, or the three entries of the third form | `latent_absorption_tile`; and `latent_tile3` for the launches that took the third form |
| Chat rows (5 or more chats in a step) | `latent_absorption_tile` | `latent_absorption_tile` |
| One-row decode step, 2 to 4 chats, drafting check rows | the step entries | `absorb_step` |
| The same under the launch recorder | the two recorded entries | `absorb_step_recorded` |
| Host references | `absorb_tc::word` | host tests |

### Proofs

Kernel proof PASS at `d189770e8c31d2471e9fa45de21a38fc17c18c07`, image
`75162f6703da`, receipt `attn-h3-words`: twin 0/947358, alone/group
0/531467712, direct and recorded step/prompt each 0/21990688; old-word
control differs on 17289/20000 words.

Timing receipt `attn-h3-time` (same head and image; base image
`49a26dae1180`), ms for the four operations of one layer and rank, launch plus
sync, base / tensor cores / tensor cores / base:

| Prompt rows | ms |
|---|---|
| 1 | 0.4188 / 0.2799 / 0.2779 / 0.4201 |
| 2 | 0.4574 / 0.2842 / 0.2821 / 0.4587 |
| 4 | 0.4932 / 0.3040 / 0.3228 / 0.4986 |
| 8 | 0.7411 / 0.3973 / 0.4116 / 0.7527 |
| 256 | 15.4144 / 6.0970 / 6.0783 / 15.2229 |
| 1,024 | 61.3581 / 22.8527 / 22.8358 / 61.4130 |

| One decode row, selected rows | same operands every launch | operands that left the cache |
|---|---|---|
| 16 | 0.0625 / 0.0653 / 0.0651 / 0.0627 | 0.1068 / 0.1281 / 0.1290 / 0.1068 |
| 256 | 0.0797 / 0.0714 / 0.0819 / 0.0812 | 0.1206 / 0.1327 / 0.1328 / 0.1207 |
| 2,051 | 0.1952 / 0.1165 / 0.1161 / 0.1957 | 0.2652 / 0.2420 / 0.2433 / 0.2675 |

Ruling D in the bench: PASS. A served step reads the weights of query and
project from memory, so the right column is the one that counts. The worst
case is 16 selected rows: 22 microseconds more a layer, 0.24 ms for 11 layers,
0.6 percent of a 41.25 ms step (arithmetic from the table). At 256 selected
rows it is 12 microseconds a layer, 0.3 percent. From about 1,000 selected
rows the one-row step is faster than before. The served check of ruling D
belongs to the gate.

What made the old one-row body slow (receipt `tcbf-takeover-time-02`, 0.1968
to 0.2923 ms): it read its operands one load at a time, and a warp waits for
each load it uses at once. The measured dead ends of the step form (a
two-buffer pipeline, touch reads, hints further ahead than two batches) and
the cost of a guard before the pointer in the prompt helpers are in
`reports/ATTN.md` of the night folder.

A wrong-on-purpose build for the gate: branch `tc-bf-control` (control only,
never merged). The one-row query and project join their ranges in the wrong
order there.

MEASURED `r1-bf-words-1` (2026-11-16, rebase train R1, merged head
`70a0019ca1`, image `2b06937bc09d`, tool exit 0, image check passed): twin
0/947358 (6 cases), old-word control differs on 17289/20000 words, 14 group
sizes from 2 to 1,040 tokens 0 differ, alone/group 0/531467712, direct and
recorded step/prompt each 0/21990688. Later commits of the branch change no
kernel file.

The served proof on the merged head is the gate seat's.

### Seat F8TC, 2026-11-16: the third form of the prompt tile (the same word)

Status: WIRED on branch `tc-f8tc-latent`. No new word, no new twin, no new
baseline: the word is the word of this section, and the proof is bits-equal
on the device against the first form. It rides a normal train.

What changed. A prompt group of 17 or more token rows runs its four fold
operations in the kernels of `kernels/src/latent_tile3.rs`: `latent_weights3`
(query and project), `latent_scores3`, `latent_values3`. The host chooses
(`latent_absorb_rows::third`): the model's shape only (32 heads, 512 latent
features, 256 key features, value rows in whole tiles of 16, a selection
table) and operands aligned to 16 bytes. Every other launch keeps `latent_absorption_tile`. The
kernels trap on a launch the host should have declined (another operation
or shape, an operand that is not aligned). One bound is the host's alone,
as it is for the first form: a cache of 2,097,151 rows or more, because the
values body stages a row's byte offset in 32 bits. The decode step (one
row) does not change.

Why the first form was slow. MEASURED (run `f8tc-lat3-4`, 1,024 tokens,
2,051 selected rows a token, a cache of 8,192 rows): a launch is 4.3
(query), 34.4 (scores), 34.4 (values) and 4.3 (project) thousand million
multiply-adds, and the first form ran them at 3,520, 2,570, 4,210 and 3,330
thousand million a second. That is the rate of this box's scalar FP32
kernels (the plain rows tile: 4,646, run `f8tc-plain-ring-1`), not that of
a tensor core. Two causes, in the body and not in the word: every block
staged its B operand again with single-word loads, and a warp asked for
its A operand in the step that used it, so it waited for memory in every
step. The third form runs the same launches at 6,270, 8,000, 8,560 and
5,650 thousand million a second.

What the third form does:

- **Scores.** A block is one token (the first form: 17 blocks a token). The
  token's absorbed query is staged once, with 16-byte loads. A warp walks
  the token's selected rows in tiles of 16. A range of the word is one
  stage of 8 steps; every address in a stage is a register plus a constant;
  the cache rows are read one step ahead.
- **Query and project.** A block is one head and 32 tokens. The tokens'
  rows are staged once. A warp walks the weight rows in tiles of 16 and
  reads the checkpoint's BF16 pairs as they lie.
- **Values.** The two operands of the tensor-core step change sides: the
  heads are on the M side (the probabilities, staged as before) and the
  features of the selected cache rows on the N side. Column `c` of N tile
  `T` of a warp is feature `first + 4 c + T`, so a lane reads its four
  tiles of a cache row with one 16-byte load, and the cache is never
  staged. The terms of a step and their order are those of the word
  function: term `k` is still `A[m][k] * B[k][n]` for slot `k`, in the
  token's selection order, 16 slots a step. Only the side of the
  instruction that holds each factor changes. The step sums the 16
  products and the carried sum; a product does not know which factor
  stood on which side (the twin's `step` takes the two factors of a term
  in either order). The device proof below compares every output word.

Hints for memory (`prefetch`) are NOT in the served entries: with a cache
that fits the GPU's second-level cache they cost time. Two bench entries
keep them (`latent_scores3_p1`, `latent_values3_p4`); they are ahead only
at a cache of 32,768 rows.

Proofs (MEASURED, one box, `kernel-bench --latent3`, run `f8tc-lat3-4`, head
`af5724f928`, image `f428698e1a04`, tool exit 0):

- **Every output word: third form = first form of the same image**, 0 words
  differ. 12 cases: 1 to 1,040 tokens; caches of 100 to 9,001 rows; 17, 40
  and 2,051 slots; counts of zero, of part of a step, of part of a tile and
  full; the served selection and scattered ones; the input classes plain,
  denormals with signed zeros, overflow with NaN and infinity, huge/tiny,
  and add-order sensitive; blocks of 256 and of 384 threads. Query
  85,196,800 words, project 42,598,400, scores 668,011,008, values
  85,196,800 (values: 256 threads only, the body traps on another size).
- **Controls are caught, 6 of 6.** Other operands through the third form
  do not give the first form's words (query: 81,845 of 81,920 words differ;
  scores: 35,200 of 48,000).
- The 16 guard bytes before and after every output do not change.
- The first form equals the CPU twin (the proofs above in this section).
  The third form is compared with it word by word, so on these cases it
  equals the twin.

The same proof on the later heads (MEASURED, one box each, tool exit 0, 0
words differ in 881,003,008, 6 controls caught): run `f8tc-lat3-5`, head
`5f8a3debbc` (main `b5fde524ad` of rebase train R2 merged), image
`fe1626f2ec2e`; run `f8tc-lat3-6`, head `2dbb8b60c0` (the review fix
below), image `f3b588a09199`. The image texts were compared function by
function on the build machine: the merge of R2 changes no latent entry,
and the review fix changes `latent_values3` and its bench form only.

Review fix (seat REVC, 14:29). In the values body the pointer of a
probability slot was formed before the guard of its read. For a slot past
the token's count it could lie past the end of the probability plane.
Nothing was read there, but the language forbids forming it. Head
`2dbb8b60c0` forms the pointer under the guard. Every pointer that is not
used again (the two row pointers after the last range of a row, the word
pointer of a tile past the capacity, an address that only a hint names,
the lane's base in cache row 0) now uses wrapping arithmetic.

Kernel bench, milliseconds a launch, first form then third form (run
`f8tc-lat3-4`; the model's shape, 2,051 selected rows a token):

| Tokens, cache rows | Query | Scores | Values | Project |
|---|---:|---:|---:|---:|
| 1,024, 8,192 | 1.219 / 0.685 | 13.384 / 4.303 | 8.164 / 4.019 | 1.291 / 0.760 |
| 1,024, 2,051 | 1.219 / 0.673 | 8.443 / 3.914 | 5.612 / 3.041 | 1.295 / 0.756 |
| 1,024, 32,768 | 1.211 / 0.676 | 28.978 / 16.428 | 16.534 / 13.261 | 1.279 / 0.763 |
| 256, 8,192 | 0.334 / 0.206 | 3.337 / 1.132 | 2.145 / 1.106 | 0.355 / 0.216 |
| 64, 8,192 | 0.081 / 0.043 | 0.870 / 0.338 | 0.604 / 0.314 | 0.093 / 0.048 |
| 17, 8,192 | 0.046 / 0.021 | 0.263 / 0.130 | 0.205 / 0.105 | 0.039 / 0.023 |
| 8, 8,192 (not taken) | 0.042 / 0.019 | 0.147 / 0.117 | 0.180 / 0.082 | 0.034 / 0.021 |
| 2, 8,192 (not taken) | 0.039 / 0.019 | 0.053 / 0.095 | 0.151 / 0.062 | 0.033 / 0.020 |

A layer's four operations at 1,024 tokens and a cache of 8,192 rows: 24.06
ms to 9.77 ms. The host takes the third form from 17 token rows: below that
the scores body falls behind (2 tokens: 0.053 to 0.095 ms). At a cache of
32,768 rows the gain is smaller (the cache no longer fits the GPU's
second-level cache); the bench forms with hints give scores 13.998 and
values 12.298 ms there.

Served on the main of rebase train R2 (MEASURED, bracket `f8tc-lat3-d`, one
pair, base main `b5fde524ad`, head `2dbb8b60c0`, image `f3b588a09199`,
order base, new, new, base):

- Cold read of 8,192 tokens, first token: 16,484.8, 15,577.0, 15,536.8 and
  16,504.9 ms: 16.495 s to 15.557 s, 0.938 s less. The engine's own
  prefill sums say 0.945 s.
- Token IDs equal the base's: 5 of 5 lists in every arm (992 tokens: three
  chats, the warm chat, the read).
- One-row decode step 41.416 ms to 41.373 ms: not slower.
- Counters, both ranks: `latent_tile3` 528 of `latent_absorption_tile` 528.
  Every prompt tile launch of the group took the third form.

Served before R2 landed (MEASURED, bracket `f8tc-lat3-b`, one pair, base
main `cbfb120bfa`, head `d4b92af43f`, image `f428698e1a04`, the same
order):

- Cold read of 8,192 tokens, first token: 17,662.1, 16,709.9, 16,718.3 and
  17,932.6 ms: 17.797 s to 16.714 s, 1.083 s less. The two base arms are
  0.27 s apart, so the gain is 0.95 s against the first and 1.21 s against
  the last.
- Token IDs equal the base's: 5 of 5 lists in every arm (992 tokens: three
  chats, the warm chat, the read).
- One-row decode step 41.157 ms to 41.084 ms: not slower.
- Counters, both ranks: `latent_tile3` 528 of `latent_absorption_tile` 528.
  Every prompt tile launch of the group took the third form.

Shared bytes of the new entries (MEASURED from the image text): 32,768
(`latent_scores3`, `latent_weights3`) and 34,816 (`latent_values3`); the
offline compiler allows 49,152.

The three unsafe-site rows of the new bodies (`tools/unsafe-allowlist.txt`)
are approved by the maintainer for head `2dbb8b60c0` (ruling 2026-11-16
14:32 and 14:41).

A wrong-on-purpose build for the gate: branch `tc-f8tc-latent-control`
(control only, never merged). With the build word `latent-tile3` the three
bodies give every output word one part in 64 larger, on the joined sum,
before the rounding or the scaling of the word. Without the word its image
equals the image of this branch.

NOT PROVEN, and open:

- The gate on a served pair (rules 1 to 6). The word does not change, so
  the token IDs must equal the base's.
- No host model walks the third bodies (the first form has one in
  `absorb_tc/tests.rs`). The device compares every output word with the
  first form, which is proven equal to the twin.
- If the main image changes its build target (rebase R3), the three
  kernels are proven again on the new image: `kernel-bench --latent3`.

## Block FP8 (seat TCF8, then seat F8ROW)

Status: WIRED on branch `tc-f8tc` (seat F8TC, 2026-11-16), rebase train R2
on top of R1; not landed. Served on a pair: the cold read of 8,192 tokens is
0.85 s faster than R1 and the one-row step is inside ruling D (see "Seat
F8TC" below). The text from "Proofs" to the table is the earlier candidate
of seat TCF8, kept as history: its one-row times are no longer the state.

### Matrices
All block FP8 matrices: shared expert gate, up and down, and latent q_a,
kv_a, q_b and o projections. Storage remains E4M3 with FP32 block scales.
The kernel uses tensor metadata, not the model name.

### Word function
For `n = ceil(cols / block_cols)`, range `p` holds whole scale blocks
`[n*p/4, n*(p+1)/4)`. This is a block-aligned partition. It equals column
quarters when the block count is a multiple of four.

Each weight byte becomes its exact BF16 value. Each input rounds once to
BF16, nearest even. Within each scale block, run ascending 16-column
`mma.sync.m16n8k16` BF16 steps from positive zero. A short step has zero
padding. Multiply the block sum by its FP32 scale, then add to the range
sum from positive zero, in ascending block order. These are separate FP32
multiply and add operations. Join as `fl(fl(R0+R1)+fl(R2+R3))`, then round
to BF16. No fused multiply-add is used in the scale or join.

The step aligns nonzero terms to the largest exponent with 25 fraction
bits, truncates each term toward zero, adds exactly, then truncates to
FP32. Zero products and a zero carried value contribute no exponent.
Subnormal operands retain exponent field -126. A result cut to zero is
positive zero. Nonfinite inputs, NaN weight codes and nonpositive or
nonfinite scales are outside this contract.

A raw quarter job returns one range over all its blocks, without a join
or BF16 round. The caller must preserve whole scale blocks when splitting.
The word does not depend on token count, slot, neighbors or kernel entry.

### CPU twin
[`tc_twin_f8.rs`](../crates/gpu/src/tc_twin_f8.rs) uses an independent
wide-integer step. [`tc_f8.rs`](../crates/gpu/src/tc_f8.rs) is the integer
form shared by the host reference and generic device entries. Both omit
zero terms from exponent alignment and return positive zero on underflow.

### Kernel entries by row count
`fp8_tile` handles one token. `fp8_rows_small`, `fp8_rows_few` and
`fp8_rows_multi` handle token tiles of eight. `fp8_rows_tile` and
`fp8_rows_tile3` handle token tiles of 32 (two bodies, one word; the host
launches the third form where its kernel takes the parts, see the last
subsection). These entries call the bodies of `kernels/src/fp8_tc.rs`.
Generic entries execute the integer word for other shapes.

Decode and small groups (1 to 16 token rows) run `fp8_tc::row_body`: a
block of 4, 8, 16 or 32 warps works on ONE tile of 16 weight rows, each
warp on its own share of the scale blocks; weights are read straight from
global memory with a prefetch one cache line ahead, twice a line. Nothing
of this enters the word. The prompt tile keeps its staged body.
`fp8_tile` is a kernel entry and a recorded launch kind of its own
(`Kind::Fp8Tile`, `resident_launch_fp8_tile`). It is not a branch of
`expert_operation`: inside that large function the same body keeps more
registers a thread and fewer blocks run at once (MEASURED `f8row-var-4`,
shape down: 26.5 us there, 24.4 us as its own entry, main 24.7 us).

### Paths wired, with counters
The candidate replaces the five specialized entries, the generic expert
word and the host expert reference. Launch geometry was changed with the
entries. The host fallback of `tp/src/bounded_decoder.rs` uses
`matrix_role::fp8_host_product`, the contract word (`expert_ops::reference`).

Counters (`prompt_launches.rs`), printed by every rank at the prompt's end
and at the session's end as `SERVED FP8 COUNTS rank= session= at=prompt|end`:

- `fp8_tc_step`: a decode launch of `fp8_tile` (one token row a job).
- `fp8_tc_rows`: a group launch of `fp8_rows_small`, `fp8_rows_few` or
  `fp8_rows_multi`.
- `fp8_tc_tile`: a group launch of `fp8_rows_tile` or `fp8_rows_tile3`.
- `fp8_tc_tile3`: the launches of `fp8_tc_tile` that took `fp8_rows_tile3`
  (the third form of the tile body).
- `fp8_generic_layer`: a block FP8 launch that took a generic body (the
  same word in its integer form). Zero is expected for the model's shapes.

MEASURED on served pairs (both ranks equal): 128 `fp8_tc_step` a decode
step and 170 `fp8_tc_tile` a prompt group (1,360 in a read of 8,192
tokens), run `f8tc-r2-2` on head `f869e0ba86` and run `f8tc-r2-3` on head
`2f6ca41e93`; chats in groups of 2 to 8: `fp8_tc_rows` 87,296; drafting:
`fp8_tc_rows` 68,224, run `f8tc-r2-6-rows` on head `28812a1800`.
`fp8_generic_layer` is 0 in every run.

### Proofs
The bench mode `kernel-bench --fp8-tc BASE CANDIDATE quick` checks raw
FP32 words against the twin, row grouping, output guards, and controls
that change a weight byte or use the wrong range order. Device results
from run `f8-takeover-q5` at `c77c951cda157d6cb066fed0f4a575196cf38faf`
pass 2,031,149 raw-word comparisons, including 12 explicit zero-term words
through six entries. The wrong one-range twin fails on 549/896 down and
640/896 gate/up samples. A changed weight byte changes 8 and 14 words,
respectively, and no other output row. The image hash starts
`783635c1d5e9`; the baseline hash starts `49a26dae1180`.

MEASURED one-row baseline/candidate times in that run are 24.5/45.5 us
for down, 26.0/79.2 us for gate/up and 153.5/312.4 us for o projection.
At 1,024 token rows they are 963.5/643.8, 1054.9/686.7 and
8346.8/5058.7 us. These are kernel bench medians, not served timings.
The one-row regressions exceed ruling D's one-percent limit.
Host validation passes 48 focused tests. Two additional inherited
quarter tests fail: role 3 admission and nonfinite FP8 behavior outside
the stated contract. Served token repeatability, resumed/cold,
drafted/plain, grouped chats, quality and the new baseline are NOT PROVEN.

Final full run: `f8-takeover-full2`, source
`6afc7f2b5e3d9bd96346534fb34c3be739b18b91`, same candidate image as above.
MEASURED 52,319,505 raw-word comparisons pass in 1,737 checks. All six
format shapes pass grouping checks at 23 token counts from 1 through
1,024, including non-tile multiples. Both controls fail on every shape.
This proves the checked kernel word and grouping cases, not served tokens.

MEASURED full-run kernel times below are main/candidate microseconds.
Mac load was 4.71 at start and 4.69 after. Each cell compares alternating
arms on one device; it is not a served speed gate.

| Shape | 1 token | 8 tokens | 1,024 tokens |
|---|---:|---:|---:|
| down 2048x2048 | 24.7/46.3 | 35.8/47.5 | 991.0/674.2 |
| gate/up 1024x4096 | 26.0/81.8 | 43.2/83.9 | 1078.4/706.6 |
| o 2048x16384 | 153.2/318.9 | 203.2/327.5 | 8404.5/5202.5 |
| q_b 8192x1536 | 60.1/87.4 | 84.1/111.4 | 2892.6/2005.5 |
| q_a 768x4096 | 20.9/81.3 | 42.9/83.9 | 725.4/473.9 |
| kv_a 288x4096 | 18.7/80.1 | 41.0/83.9 | 359.1/293.9 |

Ruling D fails for all six shapes. The one-path source check passes.
The unsafe-site check still requires review of `fp8_tc::group` and
`fp8_tc::run`; no allowlist entry was added. The benchmark's separate
unsafe helper was removed before the full run. Full source checks and
whole host suites remain held under the sprint Mac-load order.

### Seat F8ROW, 2026-11-16: the one-row cause and the wired head
Cause of the one-row loss: the first candidate copied each scale block's
weights through shared memory behind two warps a tile, so few memory
requests were in flight and every weight byte was moved twice.

Run `f8row-all-5`, branch `tc-f8row` head `7f9ac4de78` (image hash starts
`8c4059320345`; the next head `1188b0b034` builds the byte-identical image)
against main `8f615ab806` (image hash starts `277fb106b7d8`). MEASURED:
52,319,505 raw-word comparisons pass in 1,737 checks; a row alone equals
the twin and the same row in groups of 23 sizes from 1 to 1,024 on all six
shapes. Kernel bench medians, main/candidate microseconds, each call on
its own cold weights; not a served timing.

| Shape | 1 | 2 | 4 | 8 | 256 | 1,024 tokens |
|---|---:|---:|---:|---:|---:|---:|
| down 2048x2048 | 24.5/24.8 | 30.7/24.7 | 30.6/25.0 | 34.8/25.9 | 267.4/178.8 | 962.0/645.3 |
| gate/up 1024x4096 | 26.1/24.7 | 28.7/25.3 | 39.0/25.2 | 41.9/25.7 | 347.2/229.9 | 1046.8/687.0 |
| o 2048x16384 | 153.3/151.7 | 185.2/152.2 | 169.1/153.6 | 198.4/165.2 | 2219.0/1358.7 | 8394.7/5055.0 |
| q_b 8192x1536 | 60.1/59.7 | 75.9/60.7 | 75.7/61.1 | 82.8/66.6 | 726.2/508.6 | 2915.1/2001.8 |
| q_a 768x4096 | 20.8/20.4 | 24.6/20.6 | 49.0/20.6 | 41.3/20.7 | 193.6/117.5 | 707.7/462.5 |
| kv_a 288x4096 | 18.5/12.3 | 20.4/12.4 | 22.6/12.4 | 40.1/12.4 | 188.6/143.7 | 349.3/284.5 |

Ruling D, run `f8row-var-5` (same images, the other box): the pass of the
170 one-row calls of a step is 5.99 ms on main and 5.84 ms on the
candidate; in the served form (gate and up as one launch of two jobs, 128
launches) 5.69 and 5.57 ms. Five shapes are faster than main in every run.
The shape down is equal to main inside the run noise: candidate over main
0.988 (`f8row-var-4`), 1.010 (`f8row-all-5`) and 1.024 (`f8row-var-5`).
So the step passes ruling D; the shape down alone is not proven inside
1 percent in every run.

Register caps: no entry that holds this body (`fp8_tile`,
`resident_launch_fp8_tile`, `fp8_rows_small`, `fp8_rows_few`,
`fp8_rows_multi`, `fp8_rows_tile` and their recorded entries) is in
`REGISTER_CAPS`, and the built image has no `.maxnreg` on them.

The unsafe-site check passes with eight PROPOSED rows for `fp8_tc.rs` and
`fp8_tc_rows.rs` in `tools/unsafe-allowlist.txt` (they need the review of
the pull request). NOT PROVEN: served token repeatability, resumed against
cold, drafted against plain, grouped chats, quality, and the new baseline.

### Seat F8TC, 2026-11-16: rebase train R2, the served head
Branch `tc-f8tc`, head `28812a18006adf5abcc2b192783d0b9bdd70c471`: R1
(`tc-rebase-r1` `367072d504`) plus the block FP8 trail above. Kernel image
hash starts `0c0099cc0d66`; R1's image `2b06937bc09d`. The word, the twin
and the kernel bodies are those of seat F8ROW, unchanged.

Device proofs on this image, with the served launch geometry of this head
(MEASURED, tool exit 0, image check ok):
- Run `f8tc-fp8-r2-all-1`: 52,319,505 word comparisons in 1,737 checks, the
  twin sweep and a row alone against groups of 1 to 1,024 rows on the six
  shapes; both controls caught. (Run `f8tc-fp8-r1-all-1` gave the same on
  the launch geometry before the shared-memory floor.)
- Run `f8tc-rec-2`, the RECORDED decode entry. The served step does not
  launch `fp8_tile` directly: it replays `resident_launch_fp8_tile` with a
  1,040-byte command. Bench mode `recorded` launches it the same way. Its
  words equal the direct entry in the three roles, the twin on sampled
  rows, and two jobs in one launch equal each matrix alone: 402,766 words
  in 210 checks, 6 controls caught.

Kernel bench, R1's image against this image, microseconds a launch, each
call on its own cold weights (run `f8tc-fp8-r2-all-1`; not a served timing):

| Shape | 1 | 2 | 4 | 8 | 256 | 1,024 | 2,048 token rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| down 2048x2048 | 24.5/24.7 | 30.8/25.0 | 32.2/25.0 | 35.5/25.8 | 274.5/185.0 | 989.3/673.3 | 2036.2/1344.0 |
| gate/up 1024x4096 | 26.4/25.1 | 28.9/25.1 | 40.1/25.3 | 43.4/27.9 | 356.7/237.1 | 1077.5/720.1 | 1991.9/1312.6 |
| o 2048x16384 | 151.7/150.9 | 190.8/151.2 | 170.7/151.8 | 203.9/167.7 | 2249.3/1420.3 | 8410.4/5201.9 | 17452.3/10306.0 |
| q_b 8192x1536 | 59.6/60.7 | 78.3/60.5 | 76.7/60.9 | 83.5/67.7 | 741.2/525.5 | 2891.0/2003.9 | 6017.7/4020.5 |
| q_a 768x4096 | 20.7/20.4 | 24.6/20.6 | 50.9/20.6 | 42.7/20.6 | 198.9/121.6 | 728.1/483.3 | 1576.9/960.3 |
| kv_a 288x4096 | 18.7/12.4 | 20.5/12.5 | 23.6/12.5 | 41.0/12.6 | 194.8/148.5 | 358.3/293.7 | 713.9/596.8 |

At 2 to 8 rows main serves its shared-read entry `fp8_rows_share` for some
cells (down 25.0, 26.7, 33.6; o 154.8, 166.3; q_b 63.4, 65.8, 83.7; q_a
28.1 and 30.8 at 4 and 8): the new entry is level with it at 2 rows and
faster at 4 and 8.

Served, base R1 against this branch, order base, mine, mine, base:

| Bracket, head | Read of 8,192 tokens, first token | One-row step |
|---|---|---|
| `f8tc-r2-2`, `f869e0ba86` | 17.467 to 16.553 s | 41.22 to 43.20 ms: FAIL |
| `f8tc-r2-3`, `2f6ca41e93` | 17.243 to 16.390 s | 41.05 to 41.40 ms: 0.86 percent |
| `f8tc-r2-5`, `28812a1800` | not run (prompt path unchanged) | 41.107 to 41.253 ms: 0.36 percent |

Two causes of the one-row loss, both outside the kernel bodies:
1. The recorder's name table had lost the arm `"fp8_tile"` in a merge. The
   86 one-job launches of a step left the recorded part and ended it. A
   host test now reads the recorder's text
   (`the_recorder_knows_the_fp8_tile_entry_by_name`).
2. The decode launch asked for 3 to 4 KiB of dynamic shared memory, so the
   driver ran it in a 32 KiB configuration. The kernels the step runs
   beside it on a second stream run at 100 KiB (the routed-expert tile, and
   main's block FP8 kernel did too). Traces `f8tc-r2-4-tb` and `-tc`: the
   routed-expert tile started 12 us late 42 times a step. The launch now
   asks for at least `FP8_TILE_SHARED_FLOOR` (10,240) bytes. The body only
   checks that it has enough: no word changes, the image is the same.
   Trace `f8tc-r2-6-tc` of this head: the tile starts on time again (1.9 us
   after the block FP8 launch; 0.7 in the base). It still ends later than
   in the base (its own time 184.0 to 194.8 us beside the block FP8
   launch): that is what is left of the 0.36 percent.
The lesson for the next format: time the RECORDED entry, and look at a
trace of the step for the kernels that run beside the new one.

Served checks on the new token IDs (MEASURED, this head): two runs give the
same IDs (3 lists, 768 tokens); a group of 8 identical chats equals the
chat alone; drafted equals plain at depth 2 and 3 (6 lists, 1,536 tokens).
The full rules 1 to 6 and the quality check are the gate's (seat QUAL).

Word accuracy, the counting rule of 2026-11-16 10:40 and 11:10, on the
1,152 captured matrix inputs (3,778,560 words, three input chains): words
not correctly rounded 732, 762, 760 (today) against 106, 117, 99;
root-mean-square without the 71 words of extreme cancellation 0.998 of
today's on every chain; unfiltered 1.126, 0.970 and 0.618 of today's. The
seat's tool and seat REVC's own reader agree.

NOT PROVEN here: the full gate, the new baseline, the second form of the
prompt tile in a served engine (bench only, 1.2 to 1.65 times faster again
on five shapes). The eight unsafe-site rows stay PROPOSED until the review.

### Seat F8TC, 2026-11-16: the third form of the prompt tile (the same word)

Status: WIRED on branch `tc-f8tc-tile2`. Kernel image `107bba7aa2e9`. No
new word, no new twin, no new baseline: the word is the word of this
section and the proof is bits-equal on the device. It rides a normal train
after R2.

What changed. From 17 token rows the prompt tile is the kernel
`fp8_rows_tile3` (body `fp8_tc::tile3_body`), where the scale blocks are
128 columns wide and a multiple of 16 rows high and the columns are whole
stages of 128 (`fp8_rows::third`; the model's six shapes all are). Every
other launch keeps `fp8_rows_tile`. A block is 8 warps (16 weight rows a
warp) and 4 tiles of 8 token rows a warp: 256 threads, 8,192 bytes of
shared memory. The driver gives the kernel 128 registers a thread.

Why the first form was slow, and what the third does (each MEASURED on one
box, cold weights, microseconds a launch of 2,048 x 2,048 at 1,024 token
rows; the first form takes 660):

| Step | Run | Result |
|---|---|---|
| A stage is one scale block; the tensor core decodes the weights | `f8tc-t3-1` | 439. BF16 patterns read from memory (no decode at all, twice the bytes) take 594: the limit was memory, not the decode. |
| The next stage's weights and activations are prefetched | `f8tc-t3-2` | 324. A pass of the model's 170 calls takes 135.9 ms without the prefetch and 80.7 ms with it. |
| Every address inside a stage is a register plus a constant | `f8tc-t3-4` | 273. The PTX keeps 38 values across the loops, was 76. |

How the tensor core decodes a weight. A lane's 8 weight bytes are the A
operand of two E4M3 steps (`m16n8k32`; the other 16 columns are +0, the
accumulator is +0.0). The B operand holds the value 1.0 (code 0x38) in one
column for each token place and +0 elsewhere. Each output word then has one
term that is not zero: the byte's value times 1.0. So the word is that
value exactly, as FP32, and `cvt.rn.bf16x2.f32` packs two of them into the
BF16 operand, exact again. Against the integer function `tc_f8::e4m3_bf16`:
the same BF16 value for every code of the contract but 0x80, which gives
+0 here and -0 there. A zero operand brings no term to the BF16 step, so
no bit of the word differs (proven on the device, below). The codes 0x7f
and 0xff are outside the contract: NaN here, 480 in the integer function.
The same body with the integer decode is a bench form (`fp8_tc_t3_n4s`):
it gives the same bits and takes 77.09 ms a pass against 67.82.

Proofs (MEASURED, run `f8tc-t3-all-2`, head `d26c46ef76`, one box, exit 0):

- The full standard proof with the host's choice in it: 96,182,947 words
  in 2,089 checks.
- Device = CPU twin on 14 matrix forms (947,423 words); the third form on
  the 8 forms its kernel takes, unrounded and rounded.
- The decode: the zero-term case with signed zero weights at 4 places, and
  all 254 weight codes in all 32 places of a lane pair (16 columns of a
  step, two rows): third form = first form = twin.
- On each of the six model shapes a row alone equals that row in groups of
  1 to 1,024 rows through BOTH tile forms, all 32 bits.
- Controls: a twin with one range fails; one wrong weight byte changes
  words of its own row only, and both tile forms move the same words.
- Variants (run `f8tc-t3-5`): 24 of 24 block sizes and forms equal the twin
  and the small entry, 3,121,462 words each.

Kernel bench (MEASURED, run `f8tc-t3-5`, microseconds a launch, first form
then third form):

| Shape (a rank's part) | 24 rows | 64 rows | 256 rows | 1,024 rows |
|---|---:|---:|---:|---:|
| down 2,048 x 2,048 | 60.0 / 34.7 | 63.0 / 37.4 | 180.9 / 89.1 | 660.0 / 281.6 |
| gate and up 1,024 x 4,096 | 113.8 / 61.8 | 117.3 / 66.8 | 233.4 / 83.0 | 698.6 / 259.9 |
| o 2,048 x 16,384 | 444.7 / 234.2 | 460.0 / 250.6 | 1,382.9 / 622.8 | 5,090.7 / 1,944.5 |
| q_b 8,192 x 1,536 | 97.3 / 66.7 | 146.7 / 93.8 | 519.1 / 273.9 | 2,002.7 / 897.3 |
| q_a 768 x 4,096 | 113.1 / 61.9 | 117.3 / 66.0 | 119.2 / 68.1 | 470.8 / 178.8 |
| kv_a 288 x 4,096 | 141.6 / 59.9 | 145.7 / 64.5 | 145.8 / 66.0 | 291.6 / 85.0 |
| A pass of the model's 170 calls, ms | 20.84 / 11.30 | 22.06 / 12.41 | 51.04 / 22.05 | 172.82 / 67.82 |

Served (MEASURED, bracket `f8tc-t3-b`, one pair, base the R2 head
`a2aff0e01c`, this head `d26c46ef76`, order base, new, new, base):

- The cold read of 8,192 tokens, first token: 16,343.8, 15,568.7,
  15,525.7, 16,321.4 ms. Base 16.333 s, new 15.547 s: 0.785 s less.
- The one-row decode step: 41.193, 41.196, 41.180, 41.290 ms. Base
  41.242, new 41.188: not slower.
- Token IDs: 5 of 5 lists equal the base's in every arm (992 tokens: three
  chats, the warm chat, the read). The same bits, served.
- Counters, both ranks equal: `fp8_tc_tile3` 2,040 of `fp8_tc_tile` 2,040
  on this head (every prompt tile launch of a block FP8 matrix took the
  third form); `fp8_generic_layer` 0.

NOT PROVEN, and open:

- The gate rules on a served pair (rules 1 to 6) for this head. The word
  does not change, so the token IDs must equal the R2 head's.
- The four unsafe-site rows of the new kernel code are PROPOSED
  (`tools/unsafe-allowlist.txt`); the contract review is owed.
- A wrong-on-purpose control for the gate exists on the branch
  `tc-f8tc-tile2-control` (never merges): the build word `f8-tile` makes
  both tile forms wrong.
- For the lead, not taken: the maker's engine rounds the ACTIVATIONS of
  block FP8 to E4M3 (8 bits, one scale a token and 128 columns). The
  yardstick seat measured that form on these shapes (run `yard-5`): a pass
  of the 170 calls would take about 33 ms (ARITHMETIC from its cells:
  42 x 117.5 + 84 x 132.9 + 11 x (949.6 + 438.5 + 71.3 + 50.1) us) against
  67.82 ms here, about 0.28 s a read. It is another word with less
  precision in the arithmetic: a new baseline and the quality check.

## MXFP8 (seat TCMX)

Status of the word below (rebase R3, branch `tc-rebase-r3`, seat TRUTH,
2026-11-16): kernels, twin and host path written; the word is proven on
the device against the twin and meets the word accuracy measure (section
Proofs, runs `r3-proof-5` and `r3-code-4`). R3 was gated and landed as train
16 on 2026-11-17 01:17:48 (main `1194b2a8e6`), for the prompt read.

Status of candidate A: WIRED on branch `tc-onerow` (seat ONEROW, 2026-11-16), merged onto
main `27276fcc0f` in rebase train R1 (branch `tc-rebase-r1`), not landed.
Candidate A, unchanged, passes ruling D in the bench at every decode shape
(runs `onerow-full-1`, `onerow-wr-2`). Every MXFP8 matrix of a layer takes this word in
every served path; the output head keeps the scalar word. The served
proofs on the new token IDs are the gate seat's (numbered baseline).
Candidate B is preserved on `tc-mx-b` with no valid device result.

### Matrices
Checkpoint table inspection by TCMX found MXFP8 q/k/v and o in 34 layers,
dense gate/up/down in three layers, and the output head. The four measured
rank-local shapes are 4096 x 4096, 2048 x 8192, 6144 x 4096 and
2048 x 12288. The all-MX draft layer adds 11 matrices whose shapes and roles
still need a full audit. This is not a complete served admission list.

### Word function
The word of rebase R3 ("scales inside the step"), `cols > 0`,
`cols % 32 == 0`, `G = cols / 32`. It replaces candidate A (the word of
rebase R1, kept below for the record). Reason: speed of the prompt read. In
candidate A two FP32 multiplies and one FP32 add stood between two
tensor-core steps of a range; that ordinary code was the cost of the tile
(MEASURED bench runs `yard-5` to `yard-9`). Owner's rulings of 2026-11-16
(10:11, 10:35, 10:40, 11:10) allow a new word and a new token baseline when
rules 1 to 6 hold after it and the word accuracy measure is met.

1. Use the stored E4M3 weight codes. Weight NaN codes are outside the
   contract. `ws` is the stored UE8M0 scale code of the weight group.
2. For each 32-column activation group, find the largest absolute FP32
   value. Select its scale code `xs` with `activation_scale_code` of the
   CPU twin. Multiply by 2^(127 - xs), then convert each value to E4M3
   with `cvt.rn.satfinite.e4m3x2.f32`. A group with a value that is not
   finite has `xs = 255`; its codes are not part of the contract. (Steps 1
   and 2 are those of candidate A.)
3. One step is the block-scaled instruction
   `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0`
   with the old sum as its accumulator, one scale code a weight row and one
   a token for the step's 32 columns:
   `acc' = step(acc, weight codes, activation codes, ws, xs)`.
   Measured rule (device probes of 2026-11-16, held by the twin): each of
   the 32 products is `significand * significand * 2^(exponent fields + ws
   + xs - 254)`. The products that are not zero, and the old sum when it is
   not zero, are lined up on the largest exponent. Each term keeps 25
   fraction bits under that exponent and is cut toward zero. The terms are
   summed exactly. The sum is cut toward zero to FP32. A result of zero is
   +0.0. A step with no live product leaves the old sum as it is (a zero
   of either sign becomes +0.0).
4. Range p is groups `[G*p/4, G*(p+1)/4)`. A boundary never cuts a scale
   group. A range is cut into SUB-CHAINS of 8 groups (256 columns), counted
   from the range's first group; the last one may be shorter. A sub-chain
   is a chain of steps in ascending group order from +0.0. Nothing stands
   between two steps of a sub-chain: no multiply, no add. The range's part
   is the sum of its sub-chains in ascending order from +0.0, each add one
   IEEE FP32 add (round to nearest), never fused.
   Second-matrix partitioning must retain these global range boundaries
   and sub-chain starts, not recut local column slices.
5. Join `fl(fl(p0+p1)+fl(p2+p3))`, IEEE adds, never fused. Store raw
   FP32. The caller performs its existing BF16 epilogue. Nothing depends
   on row count or slot.

Why sub-chains of 8 (`SUB` in the twin and in the kernels): a step cuts
toward zero, so each step of a chain can lose up to one unit of the sum's
last place, always in the same direction. One chain over a whole range (32
to 96 steps at the model's shapes) FAILED the owner's word accuracy count
(rulings of 10:40 and 11:10). MEASURED on the host, 2026-11-16, on the
3,182,592 real MXFP8 words of three decode steps of three captured chains
(main / tensor-core / exact), words that are not correctly rounded:

| word | main | tc | exact |
|---|---|---|---|
| today's scalar word | 144 | 146 | 159 |
| candidate A (R1) | 155 | 168 | 176 |
| one chain a range | 213 | 226 | 244 |
| sub-chains of 16 | 166 | 193 | 197 |
| sub-chains of 8 (this word) | 139 | 167 | 183 |
| sub-chains of 4 | 151 | 164 | 162 |

The allowed rise over today's count is three times the square root of the
sum of both counts: +50 / +53 / +55 for this word, measured -5 / +21 / +24.
One chain a range was +69 / +80 / +85 against +57 / +58 / +60 allowed. The
rms error without the words of extreme cancellation is 1.00000 times
today's for every form. Tables: the seat's files
`truth/mxr3full-<chain>-s<step>.tsv` (first reading, tool variants) and the
reading with this twin in the proofs below.

NaN and the edges: a scale code 255 (weight or activation), a NaN old sum
or an E4M3 NaN code among the operands of a step makes that step
`0x7fffffff`. An infinite old sum stays. Products that leave FP32 at the
top give an infinity. Served admission must reject words that are not
finite.

What changed against candidate A, in plain terms: the scales used to be
applied to each step's result with two FP32 multiplies (round to nearest),
and the results were added with FP32 adds (round to nearest). Now the
scales are exponents inside the step and the running sum is one more term
of the step's own sum, which cuts toward zero at 25 bits under the largest
term. So a sub-chain has one cut toward zero a step in place of three
roundings to nearest a step, and one FP32 add (round to nearest) every 8
steps in place of one a step.

#### Candidate A (rebase R1), for the record
Steps 1 and 2 as above. Step 3: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
with a +0.0 accumulator, same alignment rule on the 32 products alone.
Step 4: `term = fl(fl(D * 2^(ws-127)) * 2^(xs-127))`, two FP32 multiplies.
Step 5: each range adds its terms in ascending group order from +0.0 with
FP32 adds. Step 6: the same join.

### CPU twin
`crates/gpu/src/tc_twin_mx.rs`: an independent integer model of the step
(`step_model`: products, old sum, two scale codes), a chain of steps
(`chain`), a range as the sum of its sub-chains (`part_encoded`, `SUB`),
the join, and the packed token row (`pack_row`).
`tc_twin_mx_checks.rs` checks decoder and quantizer agreement. Thirteen
host tests pass. A zero product brings no exponent (`ZEROS_COUNT` false).
The step's NaN is the pattern `0x7fffffff` (`STEP_NAN`); the bench's raw
step probe compares it bit for bit, the word gates treat NaN payloads as
equal (the FP32 join can change a payload). Served admission must reject
words that are not finite.

### Kernel entries by row count
One row body for 1 to 8 token rows (`mx_tc::block`): a block of 128 threads
owns 16 weight rows and one token tile of up to 8; one warp per global
range, four range partials joined through shared memory. Each warp asks
for its weight bytes ONE cache line (2 groups, 64 bytes) ahead of its
loads with `prefetch.global.L2`. That is the whole cure of the one-row
slowdown: a warp waits for each load, and 1,024 warps with one load in
flight ran the memory at 112 to 125 GB/s where the scalar kernel's 4,096
warps hold 194 to 221. Nearer is better (2 < 4 < 6 < 8 < 12 < 16 < 32
groups ahead); fewer loads, 16-byte loads, L1 loads and an asynchronous
copy ring gave nothing. A prefetch changes no word.

Entries of the body: `mx_tc_rows` (grid `(ceil(rows/16), ceil(tokens/8))`)
and every entry that takes a descriptor with `mode == MX_TC` (below).
`mx_tc_tile`: 256 threads, grid `(ceil(tokens/64), ceil(rows/128))`;
activation codes and scale codes staged through shared memory; more than 8
token rows; FP32 token rows.
`mx_tc_pack` (256 threads, one thread a group of 32 columns of one token
row) writes PACKED token rows once for a call: a row's E4M3 codes, then one
scale code a group, `cols + cols/32` bytes a row. `mx_tc_tile_packed` is
the same tile on packed rows; `cols` must be a multiple of 1,024. The host
takes the packed rows for every such width (`mx_rows::tc_packed`), so each
tile block reads 33 bytes where it read 128 and does not encode again.
`mx_tc_quantize_pack` (same grid and output as `mx_tc_pack`) reads the RAW
rows and writes the bytes the quantize launch and then `mx_tc_pack` give:
it is the entry the host launches; `mx_tc_pack` stays as the reference of
its byte proof. `mx_tc_tile_packed` takes `round` 0 or 1: with 1 it stores
each word BF16-rounded, the word the round launch gave.
All take `MxProjection`, require FP32 input (the packed tile: packed
rows), `round = 0` (the packed tile: 0 or 1), and 16-byte aligned input and
weight buffers (they trap otherwise). The merged
launches apply their BF16 epilogue to the joined word inside the body.
No entry implements packed BF16 input.

### Paths wired, with counters
The switch is ONE word of the launch record: `MxProjection.mode`
(`gpu/src/next_params.rs`, in the four bytes that were padding; size and
offsets do not move). `MX_SCALAR` = 0 is the old word, `MX_TC` = 1 this
word. A tensor-core descriptor has `ceil(rows/16)` blocks of 128 threads
a token tile of 8 (`mx_tc_blocks`), so it fits the merged grids.

- Kernels: every entry that takes a descriptor reads the mode and runs
  the one row body (`mx_tc::layer_block`): `flash_next_mxfp8_m1`, `_pair`,
  `_bits` (`next_decode::project`), the merged batches
  `flash_resident_projection_batch` and `_bf16`, the row stage
  `flash_resident_projection_bf16_rows` (2 to 4 rows), the row group
  entries `mx_rows_small2/4/8/8w`, and the recorded forms of all of them
  (`resident_launch_mx`, `_mixed`, `_mixed_bf16`, the row stage and row
  group replays; `mx_rows_small::decode`). All four warps of a block take
  the body together (a block owns one descriptor), so its one barrier is
  block-wide. The scalar-only entries `mx_rows_tile` and
  `flash_next_mxfp8_bits_r4` trap on a tensor-core descriptor.
- Device matrix (`gpu/src/backend/mxfp8_projection.rs`):
  `ResidentMxfp8Projection` holds its mode; every descriptor it makes
  carries it (`apply`, both merged batches, `apply_rows`). `apply_rows` of
  a tensor-core matrix: 1 to 8 token rows through the entry name
  `mx_rows_small8` with the tensor-core grid; more through `mx_tc_tile` on
  the decoded FP32 rows in the pieces of `mx_rows::token_limit`. More than
  8 token rows on a multiple of 1,024 columns (`packs_rows`) do not take
  `apply_rows`: `pack_rows` makes the packed rows from the RAW rows in one
  launch (`mx_tc_quantize_pack`; no quantize launch, no FP32 copy of the
  quantized rows), once for the matrices that read the same rows, and
  `apply_packed_rows` runs `mx_tc_tile_packed` with `round = 1` in pieces of
  `mx_rows::TC_PACKED_TOKENS` (1,024) token rows (no round launch). The
  callers are `tp/src/dynamic_projection.rs` (`project_rows`) and the
  shared q, k, v group (`row_stage.rs`, `mx_rows_shared`). A call outside the entries'
  contract is an error, never the scalar word. `mxfp8_quarters` refuses a
  tensor-core matrix (no served caller; a quarter entry must keep the
  global range boundaries of step 5 before it may exist).
- Recorded commands (`gpu/src/resident_launch_encode.rs`): `mx` and
  `mx_rows` copy the mode (host test).
- Which matrices (`tp/src/mxfp8_projection.rs`, `LAYER_WORD`): by tensor
  role, once, at the binding. Every MXFP8 matrix that is not the output
  head takes `MX_TC`; `split` keeps it. The head (`bind_head`, role Head
  in `matrix`) keeps `MX_SCALAR`: its kernels (`gpu/src/backend/
  text_edges.rs`, the head row forms) are not part of this switch.
- Host reference: `Shape::reference_tc` (`gpu/src/mxfp8_projection.rs`) is
  this twin's word with the scalar reference's refusals; the loader's
  `dynamic_reference*` follows the matrix (`Shape::reference_mode`). No
  host path keeps the old sum order for a tensor-core matrix.
- Counters (`gpu/src/prompt_launches.rs`): `mx_tc_step` (a one-row
  descriptor, alone or in a merged batch), `mx_tc_rows_stage` (a merged
  row stage), `mx_tc_rows` (a row group of 1 to 8), `mx_tc_tile` (a prompt
  tile piece, on FP32 rows or on packed rows), and `mx_scalar_layer` (ANY launch of a layer matrix on the
  scalar word: must be 0; the head is not counted). Every rank prints
  `SERVED MX COUNTS rank= session= at=prompt|end` with the five values
  (cumulative for the process) at the prompt's end and the session's end.
  `mx_tc_quantize_pack` (one a packed input: one for the q, k and v of a
  layer, one for every other matrix) is a launch counter beside them; it is
  not one of the five. `mx_tc_pack` stays 0 in a served run.

Main's row paths that came after this wiring (merged in rebase train R1),
and the word each takes:

- The tile choice for 5 to 8 token rows (`small_choice`, `Mx::Tile`) and its
  recorded form `resident_launch_mx_rows_tile`: scalar descriptors only (the
  output head). `apply_rows` of a tensor-core matrix never asks
  `small_choice`. The recorded form runs the scalar tile body, which traps on
  a tensor-core descriptor.
- The two-row shared launch and the merged row stage with plain riders: they
  launch the row entries above, which read the descriptor's mode.

Not switched: the output head; packed BF16 input and rounded GDN
boundaries (`MxLayer::push` refuses them with `MX_TC`); the old native
path `engine/src/flash/next_native.rs` (scalar).

### Proofs
MEASURED `tcmx-tc2-07`, image `bf17a6fbfe83`: original sweep equals twin
on 385686 small-shape words and 7562 sampled model-shape words; alone/group
2049276 words equal; three wrong-on-purpose controls detected. MEASURED `tcmx-takeover-02` after the zero-rule fix: strict raw step
1048576/1048576 equal (no negative zero); G1 385686 small plus
211338 sampled model-shape words equal, including 1024 tokens; G2
2885676 alone/group words equal through 1024-token groups. Wrong-nearest
control differs on 18006/131072; wrong-chain control on 320/1170;
activation and scale controls also caught. Image check passed for commit
`d9396b698b86566c5360af598d76fb40597a4aaf`, image `bf17a6fbfe83`.
NaN payload equality and the entire served admission set are not proven.
The same run confirms cold one-row scalar/candidate 12.74/21.38 ms,
so the corrected candidate still fails ruling D. Mac load 14.74 during
build and 32.22 after the run; no served speed claim.

MEASURED same run, cold 145-call pass at 1/2/4/8 rows, scalar main versus
candidate A in ms: 12.61/12.15/13.34/16.29 versus
22.35/21.52/21.91/21.49. This fails the one-row limit.
Warm 1024-row qkv projection: 4017.7 -> 2772.5 us. This is a kernel
measurement, not a served speed claim. Historical Mac load was not recorded.

`tcmx2-tc2-6` printed the same A image and A timing tool. It ran, but
it does not prove the BF16 expansion candidate's arithmetic or speed.

MEASURED `onerow-full-1` (2026-11-16, head `c04facae1e`, image
`cedcb2a573b3`, base main `57f0869746` image `f067d66fae2e`, tool exit 0,
image check passed): the row body with the one-line prefetch. Raw step
1048576/1048576 equal; G1 385686 small plus 211338 sampled model-shape
words equal; G2 2885676 alone/group words equal through 1024-token
groups; three controls caught. Cold 145-call pass at 1/2/4/8 rows, main
versus this body in ms: 12.61/11.96/13.49/16.30 versus
11.82/11.87/12.00/12.26. One row a shape in us: q k v 82.7 -> 78.8,
o 89.7 -> 81.1, gate/up 119.1 -> 113.8, down 135.1 -> 115.9: ruling D is
met at every decode shape. One shape is slower at 2 rows (o 78.7 -> 81.1
us); the 2-row pass is faster. Kernel measurements, not a served claim.

MEASURED `onerow-wr-2` (2026-11-16, the WIRED head `844f0f0c87`, image
`9e42edd63cac`, same base, tool exit 0, image check passed). The row body
prefetches every group here. Raw step, G1, G2 and the three controls as
above (same counts, 0 differ). WIRED (`kernel-bench --mx-tc ... wired`):
each of the 18 served entries on a tensor-core descriptor equals the twin,
0 words differ: `flash_next_mxfp8_m1`, `flash_resident_projection_batch`,
`flash_resident_projection_bf16` and their recorded forms
`resident_launch_mx`, `_mixed`, `_mixed_bf16` (773 words each, with a
scalar job between two tensor-core jobs: its 302 words equal the base
image); the row stage and its recorded form (6957 words; scalar job 2718
words equal); `mx_rows_small2/4/8/8w` and their four recorded forms (17779
words each); `flash_next_mxfp8_pair` (17006) and `_bits` (19325). Controls:
the scalar word of the same descriptors differs on 344 words; a twin fed a
changed input row differs on 4248 words.
Cold 145-call pass at 1/2/4/8 rows, main versus wired, ms:
12.56/11.79/13.37/16.59 versus 11.67/11.68/11.86/12.19. One row a shape,
us: q k v 82.6 -> 78.0, o 89.2 -> 79.1, gate/up 117.7 -> 113.9, down
132.4 -> 115.4. Slower cells, disclosed: o at 2 rows cold 78.9 -> 79.8 us
(1.1 percent). WARM (one matrix again and again, in cache: not the decode
step's view, where every call reads its own weights) the new body is slower
at 1 and 2 rows: 4096 x 4096 24.7 -> 39.1 and 27.2 -> 39.1 us, 2048 x 8192
24.8 -> 38.9 and 26.4 -> 39.1 us; 6144 x 4096 and 2048 x 12288 at 2 rows
61.5 -> 68.7 and 53.4 -> 59.6 us. The served one-row step is the gate
seat's guard for ruling D. Prompt tile, warm, 1,024 rows: 4133.8 -> 3015.4,
4104.7 -> 2361.7, 7428.5 -> 5751.0, 7556.2 -> 3565.5 us for the four
shapes. Kernel measurements, not a served claim.

MEASURED `r1-mx-words-1` (2026-11-16, rebase train R1, merged head
`70a0019ca1`, image `2b06937bc09d`, base main `27276fcc0f` image
`ead8fed2da5f`, tool exit 0, image check passed). Raw step 1048576/1048576
equal; G1 385686 small plus 211338 sampled model-shape words equal; G2
2885676 alone/group words equal; three controls caught. Each of the 18
served entries on a tensor-core descriptor equals the twin, 0 words differ;
the scalar word of the same descriptors differs on 344 words. Cold 145-call
pass of one row, main versus merged: 11.73 versus 11.30 ms; main's time over
the merged time at 1/2/4/8 rows: 1.127/1.008/1.155/1.349. One 4096 x 4096
matrix at 256 and 1,024 rows: 2.22 and 2.54 times faster. Kernel
measurements, not a served claim. Later commits of the branch change no
kernel file. The served one-row step (ruling D) and the served counters on
the merged head are in the pull request of the branch and in the gate
seat's record.

#### Rebase R3: the word with the scales inside the step

MEASURED `r3-proof-5` (2026-11-16, branch `tc-rebase-r3`, kernel sources of
head `7fd86d2550`, image `54f38e40cca0`, base main `27276fcc0f` image
`ead8fed2da5f`, tool exit 0, image check passed, the bench's verdict
`MXTC PASS`):

- The step rule. Raw step against the twin on 1,048,576 words of 64
  operand classes (8 states of the old sum times 8 code classes, two scale
  codes a case): 0 differ, also under 2^-126. A model with a floor of the
  cut at 2^-149 differs on 2,352 words, at 2^-151 on 404; a model that
  rounds to nearest on 19,382 of 131,072.
- Device = twin: 0 of 385,686 words (700 small fixtures, every word) and 0
  of 211,338 sampled words of the model's shapes, up to 1,024 token rows.
- A row alone = the row in a group: 0 of 2,885,676 words at 16 group sizes
  from 2 to 1,024 and three placements.
- Packed rows: the bytes `mx_tc_pack` writes equal the twin's `pack_row`
  (0 of 3,311,136 bytes); the packed tile equals the tile on FP32 rows (0 of
  16,658,432 words); values that are not finite give the twin's NaN words
  and move no other token (0 of 9,100 words differ).
- The 18 served entries on a tensor-core descriptor equal the twin, 0 words
  differ, and their scalar words equal the base image.
- Controls that must fire, and do: one chain a word 523 of 1,170 words; one
  chain a range 164; sub-chains of 4: 51; sub-chains of 16: 103; a wrong
  activation 120 of 130 on that token and 0 elsewhere; changed weight
  scale codes 9 of 9 on that row and 0 elsewhere; one changed scale byte in
  the packed rows 130 of 130 on that token and 0 elsewhere; the scalar word
  of the same descriptors 353 words.

Word accuracy (the owner's measure of 10:40 and 11:10), MEASURED on the
host with this twin on all 3,182,592 real MXFP8 words of decode steps 1, 5
and 9 of three captured chains (main / tensor-core / exact):

- (a) words not correctly rounded: today's scalar word 144 / 146 / 159,
  this word 139 / 167 / 183. Allowed rise +50 / +53 / +55; measured -5 /
  +21 / +24. PASS on every chain.
- (b) rms without the words of extreme cancellation: 0.288660 / 0.288784 /
  0.288718 against today's 0.288662 / 0.288784 / 0.288719. PASS.
- (c) reported, not gated: unfiltered rms 1.025 / 0.999 / 0.978 times
  today's; worst word 128 / 44 / 64 BF16 steps against 64 / 44 / 128.

Speed in the bench (kernel measurements, not a served claim), same run.
Prompt tile on packed rows, 1,024 token rows in one launch: 550.9 us at
4096 x 4096, 568.0 at 2048 x 8192, 806.0 at 6144 x 4096, 930.8 at
2048 x 12288 (candidate A's tile in the same bench: 1,548 / 1,817 / 2,464 /
3,501). The pack of the rows, once a call: 34 / 200 / 50 / 343 us (run
`r3-proof-4`). One-row step, cold pass of 145 calls: main 11.72 ms, this
word 11.33 (0.967); by shape q k v 0.945, o 1.033, gate/up 0.944, down
1.000. Candidate A in its own landed run had o 1.030 and down 1.015 with
the pass at 0.963: the one-row step did not change with the word.

Machine code (run `r3-code-4`, R1 `367072d504` against this head): 634 of
659 kernels equal, 22 differ, 3 new. 21 of the 22 hold the MXFP8 body and
are the kernels proven above; 1 is compiler drift (`fp4_prefill_tc_down`).
The 3 new: `mx_tc_pack`, `mx_tc_tile_packed`, and
`resident_launch_absorb_attention` (no code under the old target; equal to
the old text compiled for the new one).

On main with the one-array fix merged (head `89e2654cb7`, image
`aad5c6ba00c6`), MEASURED: word proof `r3f-proof-1` `MXTC PASS`, 0 words
differ; the absorb entry inside this image `r3f-absorb-1`, 0 words differ;
strict code list `r3f-code-1`, verdict PASS (the target alone changes 0 of
657 kernels); the drifted kernel `fp4_prefill_tc_down` gives equal output
bytes on the old and the new image (`r3f-drift-1`).

On the main with rebase R2 (`b5fde524ad`, block FP8 on the tensor cores),
MEASURED 2026-11-16, kernel image `f34a01aed6ad` of 673 entries:

- Word proof `r3f-proof-r2a`: `MXTC PASS`, 0 words differ in every gate.
  Prompt tile on packed rows, 1,024 token rows: 544.1 / 561.7 / 800.5 /
  926.7 us (the four shapes in the order above). R2 changes nothing here.
- The absorb entry inside this image, `r3f-absorb-r2a`: 0 words differ.
- Strict code list `r3f-code-r2b` (`kernel-bench --mx-tc <old> <new>
  code-strict`; old = that main with the one-array fix): verdict PASS. The
  pinned compiler; no entry left out of any of the three compiles; code
  sections and names one to one; the target alone changes 0 of 671 kernels.
  The source change: 645 of 673 kernels equal, 26 differ, 2 new
  (`mx_tc_pack`, `mx_tc_tile_packed`).
- Of the 26, 21 hold the MXFP8 body (the word proof covers them). 5 are
  compiler drift, and not the one of the older main: `causal_operation`,
  `expert_operation`, `flash_next_rows_feed`, `resident_launch_causal_chain`
  and `resident_launch_causal_group`. Each gets an old-image-against-new-image
  device check: `--causal-probe <old> <new> bytes`, `chain-same` and `group`,
  `--fp8-tile <old> <new>`, and for the feed kernel
  `--mx-tc <old> <new> feed-pair` (written for this; it also compares with
  the host arithmetic). The runs of the head that is gated are named in that
  head's READY line and in the seat's report.

The strict mode of the code list (`code-strict`) is the receipt of a head
that lands. It fails when the compiler is not the pinned binary, when an
entry was left out of any of the three compiles (old, new, old text for the
new target), when code sections and entry names do not match one to one, or
when a kernel of the old image was not compared by the target-alone check.
The last rule was added after review (2026-11-16 14:04): before it, a kernel
the third compile dropped was not counted. The rules have host tests.

Served, MEASURED bracket `r3-speed-1` (main with R1 / R3 / R3 / main, cold
8K read, one read a load): 17,215.7 / 16,122.0 / 16,159.5 / 17,330.5 ms, so
1.13 s off the read (6.6%); the plain step gains 0.5 to 0.7% in tokens a
second on the three one-row loads. Both ranks show as many `mx_tc_pack`
launches as `mx_tc_tile` launches (145 a pass) and 0 `mx_tc_rows` launches
in the read. The bracket on the main with R2 and the token proof are in the
pull request of the branch and in the gate seat's record.

Control image for the gate: branch `tc-rebase-r3-control` (never merged).
The build word `VADE_GATE_CONTROL` set to `mx3-tile`, `mx3-rows` or
`mx3-one` makes the prompt tile, the row groups or the one-row step wrong on
purpose (the word times 1.015625). Without the word the image is this
head's own image.

### Prompt groups: two launches a product where four stood (2026-11-17)

The same word and the same token IDs as the head above. An MXFP8 product of
a prompt group was four launches a piece of 1,024 rows: quantize the rows,
pack them, the tile, the round. It is two: `mx_tc_quantize_pack` on the raw
rows, and `mx_tc_tile_packed` with `round = 1`.

Why the one launch writes the pack's bytes is in the kernel's comment
(`gpu/kernels/src/mx_tc.rs`) and in the CPU twin
`gpu/src/tc_twin_mx_pack.rs`, whose host test runs the kernel's short way
against quantize-then-pack on 60,000 groups.

MEASURED against the head above (old image `a6236ad9f012`, this image
`e450119b0361`):

- Bytes, run `mxqp-bytes-2`: 235,268,352 bytes of packed rows, 0 differ
  (10 shapes, 8 input kinds). Of 7,129,344 groups: 27,816 not finite (18,531
  of them by a quantized value past the largest float), 818,296 with the
  pack's scale code one below the quantizer's, 576,915 with scale code 0.
  Controls: one changed input word moves 32 of its group's 33 bytes and no
  other, and the first bytes return with the word; the pack alone on the raw
  rows differs in 80,738 of 274,560 bytes.
- The chain to the rounded word, same run: 47,222,784 words, 0 differ; the
  tile with `round = 0` against the old tile: 0 differ; the round moves
  39,673,303 of those words.
- Word proof on this image, run `mxqp-word-1` (`--mx-tc all,tile`): PASS.
- Served bracket `mxqp-speed-1` (old / this / this / old, cold 8K read, first
  token): 12,273.1 / 11,590.8 / 11,572.8 / 12,268.3 ms, so 0.69 s off (5.6%).
  The one-row step does not change. 10 of 10 token ID lists of the two new
  arms equal baseline 04.
- Traced read `mxqp-split-1` against `r3-split-t15-1`, rank 0: the quantize
  launches (616, 0.557 s) and `mx_tc_pack` (1,160, 0.188 s) became
  `mx_tc_quantize_pack` (616, 0.143 s); `bf16_lanes_round` lost 1,160
  launches (0.129 s); `mx_tc_tile_packed` 0.729 s to 0.724 s. The chain went
  from 1.60 s to 0.87 s.

Kernels the compiler moved (strict code list `mxqp-code-1`: 664 equal, 24
differ, 1 new). One is this change (`mx_tc_tile_packed`). 19 hold the same
instructions and constants in both images (text proof `mxqp-text-1`, every
control seen). 4 were compiled another way with one more kernel in the
crate, with no source change, and each passes its old-against-new check:
`resident_launch_causal_group` (`mxqp-group-1`), `fp4_prefill_tc_down`
(`mxqp-down-1`), `mhc_prepare` against the host twin on both images
(`mxqp-mhcbytes-1`, `mxqp-mhcbytesold-1`), and `mhc_boundary_finish`
(`mxqp-finish-1`, `kernel-bench --mhc-sweep`: the old image's finish against
the new image's, 38,356 comparisons of six guarded planes, 6,175,127,344
bytes, 0 differ; 2,508 controls, each a wrong norm weight plane that changes
the output plane alone, and the right plane restores equal bytes). Run
`mxqp-mhctile-1` covers the residual and tile launches only: it times the
finish launch and compares none of its bytes.
