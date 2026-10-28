# Benchmarks

<!-- bench-docs:start -->

### Latest measured main

| Model | Sparks | Path | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) | IDs saved / responses |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.8-flash-next-fp4-local | 2 | tensor split | ON | 53.657 | 99.805 | 128.09 | 78.626 | 163.704 | 228.245 | 2.166 | 66/66 |

Measured main revision: `8b7b866746452c01d60985f4649a588cd080c7bf`.

2 Sparks, tensor split, drafting ON: finished `2026-10-28T15:34:33.951536+00:00`; card SHA256 `dc73321898778eebf0f364a01f05a84ce41c9dcd16e6ec76c6ac587a80ae1eed`.

Rates are card medians in tok/s. C1, C4 and C8 cover all chats together.
Cold 8K is the median time to the first token. IDs saved counts responses
with recorded token IDs. This count does not prove that the IDs match.

Only the newest measured revision in main's first-parent history appears.
Each row uses the last finished, complete, passing, thinking-OFF card for
its Spark count, path and drafting setting. Older revisions do not fill gaps.

From the repository root, refresh both tables with this command.
Replace the quoted path with the folder that holds the cards.

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- bench-docs '<cards-dir>' .
```

The sections below are historical results. Their dates and revision names
apply only to those measurements.

<!-- bench-docs:end -->

This page has every speed we have published for Vade, newest first, and how
to measure your own:

- [2026-10-28 cards after #139 and #140](#current-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks):
  two DGX Sparks on one pair, one drafting-ON card at main `8b7b866` (#140),
  and before it drafting ON and OFF at main `104205a`, with eight chats
  faster after the arena reuse change.
- [2026-10-28 morning cards](#previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks):
  two DGX Sparks at main `bb24e80`, drafting ON on one pair and OFF on the
  other, first-token times for 32K, 64K and 128K prompts, and a second
  question on a long chat with the prefix cache on one Spark.
- [Final 2026-10-27 cards](#previous-2026-10-27-final-main-09e1fc4-home-dgx-sparks):
  two DGX Sparks at main `09e1fc4`, drafting ON on one pair and OFF on the
  other, and first-token times for 32K and 64K prompts.
- [2026-10-27 morning cards](#previous-2026-10-27-morning-main-12142bb-home-dgx-sparks):
  two DGX Sparks on the tensor split as the default, with no settings set, and
  first-token times for 32K and 64K prompts.
- [2026-10-26 cards](#previous-2026-10-26-main-262b0c3-home-dgx-sparks): two
  DGX Sparks on the tensor split with its levels set by hand, next to the
  layer pipeline it replaces.
- [2026-10-25 cards](#previous-2026-10-25-main-80ac2d4-home-dgx-sparks): two
  and four DGX Sparks on main, FP4-Local and FP8, on the layer pipeline.
- [2026-10-24 cards](#previous-2026-10-24-v010-release-5a22651): the v0.1.0
  release on one and two DGX Sparks and on cloud GB10 VMs.
- [Earlier cards from the same day](#earlier-cards-from-the-same-day-main-4681d8b-and-73f7a9d),
  then [how to reproduce a run](#reproduce-a-measurement-in-detail) with
  RigMark or `vade bench`.

Every result names the build and settings we measured, and its timings hold
for that build only. The older FP8 measurements are kept at the end, as
history.

A **receipt** is the saved record of one run: its build, its settings and what
it measured. The receipts behind the 2026-10-24 tables are in
[bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md), as
sanitized copies: private details such as hostnames are replaced, so our
machines appear as home-1 to home-6 and the rented ones as vm-a to vm-d. The
[receipt status](../bench/receipts/README.md) page says what is public. The
2026-10-25 to 2026-10-28 cards are not in the repository yet; each row names
its result folder. The 2026-10-27 and 2026-10-28 folder names end in a pair
suffix that uses internal machine names (and `-final`, `-am`, `-139` or
`-139-c140`), so those
names are given here without the suffix; the time, commit and
drafting setting in each name still identify the card. The first-token,
prefix-cache and repeated cold-prompt timings of 2026-10-28 after #139 come
from separate runs, not from the cards. Each of those tables or sentences
names its run folder, such as `c139-l32`. These folders are not in the
repository yet either.

<a id="current-2026-10-28-after-139-main-104205a-home-dgx-sparks"></a>

## Current (2026-10-28, main 8b7b866 and 104205a, home DGX Sparks)

The newest card is one drafting-ON card at main `8b7b866` (#140), in the
first row below. It is the card in the latest measured table at the top of
this page. No drafting-OFF card ran at `8b7b866`, so that table has no OFF
row. The two cards before it ran at main `104205a`, after the merge of #139,
which lets eight chats reuse their attention arenas. All three ran on one
pair of our own DGX Sparks, home-3 + home-4. The recipe is the same as the
[morning cards](#previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks):
context 32768, output 8192, MTP 3, greedy, cache off, thinking OFF, and no
`VADE_W2_*` name at any server start. Each card passed its **15/15** basic
output checks and ran a prebuilt payload of its commit, installed and
hash-checked on both Sparks. The serving receipts record the tensor split
with the same four levels as before. Each card had its own server start.
The `104205a` cards ran first, ON and then OFF, and the `8b7b866` card ran
right after them.

| Path | Drafting | Commit | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Tensor split | ON | `8b7b866` | home-3 + home-4 | 53.7 | 99.8 | 128.1 | 78.6 | 101.9 | 163.7 | 228.2 | 2.17 | `2026-10-28_1034_qwen3.8-flash-next-nvfp4_local-2spark_8b7b866-drafting-on-thinking-off-split` |
| Tensor split | ON | `104205a` | home-3 + home-4 | 53.8 | 100.2 | 128.7 | 78.9 | 102.3 | 165.3 | 229.6 | 2.19 | `2026-10-28_1009_qwen3.8-flash-next-nvfp4_local-2spark_104205a-drafting-on-thinking-off-split` |
| Tensor split | OFF | `104205a` | home-3 + home-4 | 35.0 | 35.0 | 35.3 | 34.1 | 49.1 | 82.7 | 125.4 | 2.21 | `2026-10-28_1027_qwen3.8-flash-next-nvfp4_local-2spark_104205a-drafting-off-thinking-off-split` |

The next part compares the `104205a` cards with the morning cards. The
`8b7b866` card has its own part after it.

### What changed since the morning cards

**Eight chats with drafting ON are faster.** C8 reached 229.6 tok/s against
217.5 in the morning card (+12.0 tok/s, +5.5 %). The three rounds were
187.1, 229.6 and 230.9 tok/s, against 199.1, 218.9 and 217.5. The first
round is still low. In the server log, the first eight-chat round allocates
arenas for the four chats that the card had not run before, and the later
rounds reuse them. Over each card, 8 admissions allocated 96 arenas, the
other 58 admissions reused 696, and no arena was freed directly. With drafting OFF,
C8 reached 125.4 against 124.3 on the same pair (+1.1 tok/s, +0.9 %).

**The other rates are the same.** With drafting OFF, on the same pair as the
morning OFF card, every other rate is within 0.8 % of it. With drafting ON,
the other rates are 0.3 % to 0.9 % lower than the morning ON card, which ran
on home-1 + home-2. The last earlier ON card on home-3 + home-4 (2026-10-27
morning) had prose at 53.5, so this pair runs a little lower on one chat.
The cold 8K prompt of the OFF card took 2.21 s against 2.17 s on the same
pair in the morning. The ON card took 2.19 s, as in the morning.

**The answers did not change.** The token IDs matched in 66 of 66 responses:
the ON card against the OFF card, and each card against the morning cards.

**Long prompts and the prefix cache did not change.** The 32K, 64K and 128K
cells ran on home-3 + home-4 with the morning recipe, one server start per
prompt. All four runs of a prompt gave the same token IDs as the morning
runs. Every median is within 0.19 s of the morning. The receipts are the run
folders `c139-l32`, `c139-l64` and `c139-l128`, one for each prompt.

| Prompt | Tokens | First token, drafting ON (s) | First token, drafting OFF (s) |
| --- | ---: | ---: | ---: |
| 32K | 32,099 | 10.07, 10.14 | 10.28, 10.07 |
| 64K | 64,217 | 19.95, 20.09 | 19.38, 19.31 |
| 128K | 129,220 | 42.48, 42.56 | 40.34, 40.17 |

On one DGX Spark (home-3) with context 65536 and the prefix cache on, the
second question on the 32,103-token chat reused 32,064 tokens and gave its
first token in 0.63 s. With the cache off it took 16.23 s. Both answers have
the same token IDs as the morning cell. The receipt is the run folder
`c139-warm`.

### The card at main 8b7b866 (#140)

#140 forms the prompt HC down projection from the residual rows. One
drafting-ON card ran at main `8b7b866` on home-3 + home-4, right after the
cards above. Its token IDs matched the `104205a` card and the morning card in
66 of 66 responses. Its rates are 0.3 % to 1.0 % lower than the `104205a`
card from one server start each; #140 does not change decode.

The first-token cells used the same recipe, one server start per prompt. The
receipts are the run folders `c140-l8`, `c140-l32`, `c140-l64` and
`c140-l128`. Each prompt ran four times, and all four runs of a prompt gave
the same token IDs, in 16 of 16 runs. Only the 32K, 64K and 128K prompts
also ran at `104205a`. Their 12 runs match the `104205a` runs. The 8K prompt
did not run at `104205a`. Its token IDs match a separate 8K check at main
`05a1251` on another pair.

| Prompt | Tokens | First token, drafting ON (s) | First token, drafting OFF (s) | `104205a`, drafting ON (s) | `104205a`, drafting OFF (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 8,009 | 2.34, 2.38 | 2.38, 2.25 | not measured | not measured |
| 32K | 32,099 | 10.01, 9.98 | 10.14, 9.92 | 10.07, 10.14 | 10.28, 10.07 |
| 64K | 64,217 | 19.87, 19.84 | 19.15, 19.14 | 19.95, 20.09 | 19.38, 19.31 |
| 128K | 129,220 | 43.54, 42.74 | 39.93, 39.73 | 42.48, 42.56 | 40.34, 40.17 |

The 32K and 64K prompts and the 128K prompt with drafting OFF reached their
first token 0.8 % to 1.5 % sooner. The 128K prompt with drafting ON was
slower on this start; its first run took 43.54 s, 0.8 s to 1.1 s more than
the other 128K runs with drafting ON that day. One start does not show if
this is a change or a slow request. Twenty cold 8K prompts on one server
start gave a median of 2.18 s (receipt: the run folder `c140-c8x20`).

<a id="current-2026-10-28-morning-main-bb24e80-home-dgx-sparks"></a>

## Previous (2026-10-28 morning, main bb24e80, home DGX Sparks)

These are the morning RigMark cards of 2026-10-28, measured on the same two
pairs of our own DGX Sparks as the
[final 2026-10-27 cards](#previous-2026-10-27-final-main-09e1fc4-home-dgx-sparks),
at main `bb24e80` (#136). The recipe is the same: context 32768, output 8192,
MTP 3, greedy, cache off, thinking OFF, and no `VADE_W2_*` name at any server
start. Each card passed its **15/15** basic output checks and ran a prebuilt
payload of its commit, installed and hash-checked on both Sparks. Both serving
receipts record the tensor split with the same four levels as before.
home-1 + home-2 ran drafting ON, and home-3 + home-4 ran drafting OFF, each
on its own server start.

| Path | Drafting | Commit | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Tensor split | ON | `bb24e80` | home-1 + home-2 | 54.3 | 100.7 | 129.3 | 79.4 | 103.0 | 165.8 | 217.5 | 2.19 | `2026-10-28_0634_qwen3.8-flash-next-nvfp4_local-2spark_bb24e80-drafting-on-thinking-off-split` |
| Tensor split | OFF | `bb24e80` | home-3 + home-4 | 35.1 | 35.3 | 35.4 | 33.9 | 49.0 | 82.1 | 124.3 | 2.17 | `2026-10-28_0640_qwen3.8-flash-next-nvfp4_local-2spark_bb24e80-drafting-off-thinking-off-split` |

### What changed since the final 2026-10-27 cards

**The cards are the same.** On the same pairs, every rate is within 0.9 % of
the final 2026-10-27 cards. The largest move is drafting OFF structured
output, 35.4 tok/s against 35.1. Eight chats with drafting ON reached 217.5
against 218.8; the three rounds were 199.1, 218.9 and 217.5 tok/s, and the
first round was low in the final cards too (198.5). The night's engine
changes are for long prompts and the prefix cache; the KV page ledger (#133)
is off by default.

**The answers did not change.** The token IDs matched in 66 of 66 responses:
the ON card against the OFF card, and each card against the final 2026-10-27
cards.

**A 64K prompt reaches its first token about 1 s sooner.** Two Sparks now
read the middle of a prompt of 32K tokens or more in passes of the whole
prompt workspace (#135). These cells ran on home-1 + home-2 with no
`VADE_W2_*` name, one server start per prompt, context 69632 for 32K and 64K
and 131072 for 128K. The cache was off. Each prompt ran four times, twice
with drafting OFF and twice with drafting ON, with a 1024-token cap. The
table gives both runs of each. All four runs of a prompt gave the same token
IDs, and the 32K and 64K IDs match the final 2026-10-27 runs.

| Prompt | Tokens | First token, drafting ON (s) | First token, drafting OFF (s) | Final 2026-10-27, drafting ON (s) |
| --- | ---: | ---: | ---: | ---: |
| 32K | 32,099 | 10.03, 10.09 | 10.24, 10.03 | 10.24, 10.01 |
| 64K | 64,217 | 20.04, 19.99 | 19.33, 19.22 | 21.06, 21.06 |
| 128K | 129,220 | 42.47, 42.53 | 40.16, 39.98 | not measured |

The final 2026-10-27 runs used a different client with the same prompts.

**One Spark answers the next question on a long chat from its cache.** This
cell ran on one DGX Spark (home-1) with context 65536 and the prefix cache on
(`VADE_FLASHNEXT_PREFIX_CACHE=1`), drafting ON. The first question carried a
32,103-token prompt and gave its first token in 16.33 s. The second question
(32,309 tokens) reused 32,064 tokens from the cache and gave its first token
in 0.62 s. The same second question with the cache off took 16.29 s. Both
answers have the same token IDs with and without the cache. Before #136, at
this context the cache had no room to store the first question. Two Sparks
do not use the prefix cache yet.

<a id="current-2026-10-27-final-main-09e1fc4-home-dgx-sparks"></a>

## Previous (2026-10-27 final, main 09e1fc4, home DGX Sparks)

These are the last RigMark cards of 2026-10-27, measured on two pairs of our
own DGX Sparks at main `09e1fc4`. Each card passed its **15/15** basic output
checks and ran a prebuilt payload of its commit, installed and hash-checked on
both Sparks. The recipe is the one of the
[morning cards](#previous-2026-10-27-morning-main-12142bb-home-dgx-sparks):
context 32768, output 8192, MTP 3, greedy, cache off, thinking OFF; prose,
code and structured 5 times with an 8192-token cap, a cold 8K prompt 3 times,
and C1/C2/C4/C8 for 3 rounds with a 256-token cap. No server start set a
`VADE_W2_*` name. Both serving receipts record the tensor split with the same
four levels as the morning cards.

One thing differs from the morning: each pair ran one card. home-1 + home-2
ran drafting ON, and home-3 + home-4 ran drafting OFF, each on its own server
start.

| Path | Drafting | Commit | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Tensor split | ON | `09e1fc4` | home-1 + home-2 | 54.1 | 100.6 | 129.6 | 79.0 | 102.4 | 165.2 | 218.8 | 2.19 | `2026-10-27_2309_qwen3.8-flash-next-nvfp4_local-2spark_09e1fc4-drafting-on-thinking-off-split` |
| Tensor split | OFF | `09e1fc4` | home-3 + home-4 | 35.2 | 35.4 | 35.1 | 33.8 | 49.0 | 82.6 | 123.9 | 2.17 | `2026-10-27_2314_qwen3.8-flash-next-nvfp4_local-2spark_09e1fc4-drafting-off-thinking-off-split` |

### What changed since the morning cards

**A cold 8K prompt reaches its first token sooner.** It took 2.19 s with
drafting ON and 2.17 s with drafting OFF. On the same pairs in the morning it
took 2.80 s and 2.77 s. On two Sparks, each Spark now reads half of a cold
prompt of 2K to 8K tokens (#130).

**Drafting OFF is faster.** On home-3 + home-4, one chat wrote prose at
35.2 tok/s and code at 35.4 tok/s, against 29.7 and 29.6 in the morning.
Eight chats reached 123.9 against 110.8. The plain one-token step now splits
its output head by vocabulary halves across the two Sparks (#129) and runs as
captured CUDA graphs (#131). Eight chats without drafting are still slower
than on the layer pipeline in the 2026-10-26 cards (137.0).

**Drafting ON is the same.** On home-1 + home-2, prose was 54.1 tok/s against
53.9 in the morning, code 100.6 against 100.4 and structured 129.6 against
130.3. Eight chats reached 218.8 against 221.8. The three rounds were 198.5,
220.4 and 218.8 tok/s; in the morning they were 203.8, 221.8 and 223.2. This
is one card, so we do not read it as a loss.

**The answers did not change.** The token IDs matched in 66 of 66 responses:
the ON card against the OFF card, and each card against the morning cards.

**Long prompts.** These ran on a second start of home-1 + home-2 with context
69632, because a 64K prompt does not fit in 32768. Drafting was ON as served,
the cache was off, and each prompt ran twice. The table gives the first run
of each prompt. The second runs were 10.01 s and 21.06 s. The two runs of a
prompt gave the same token IDs, and they match the morning runs.

| Sparks | 32K prompt, 32,099 tokens: first token (s) | 64K prompt, 64,217 tokens: first token (s) |
| --- | ---: | ---: |
| home-1 + home-2, final | 10.24 | 21.06 |
| home-1 + home-2, morning | 10.10 | 21.54 |

### One Spark beside the final pair

The one-Spark cards below ran on `8a349e9ae50b7ad284ccd427e165378c6644490a`
after the final pair cards. That commit changed docs only, so its engine tree
is the same as `09e1fc4`. The one-Spark ON and OFF cards share one server
start. Their context, output, prompts and request counts match the final
pair recipe above. Thinking was OFF and the cache was off.

| Setup | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One DGX Spark, `8a349e9` | ON | 42.243 | 80.571 | 100.151 | 64.229 | 143.496 | 169.937 | 3.798 |
| Two DGX Sparks, final `09e1fc4` | ON | 54.088 | 100.616 | 129.586 | 79.006 | 165.194 | 218.787 | 2.190 |
| One DGX Spark, `8a349e9` | OFF | 27.917 | 27.777 | 28.000 | 27.175 | 85.626 | 136.173 | 3.758 |
| Two DGX Sparks, final `09e1fc4` | OFF | 35.229 | 35.384 | 35.144 | 33.761 | 82.601 | 123.883 | 2.166 |

Rates are card medians in tok/s. C1 through C8 count all chats together.
These are separate cards, not a controlled speed comparison on the same
hardware. The one-Spark source cards are
`2026-10-28_0011_qwen3.8-flash-next-nvfp4_local-1spark_8a349e9-drafting-on-thinking-off-onespark`
and
`2026-10-28_0024_qwen3.8-flash-next-nvfp4_local-1spark_8a349e9-drafting-off-thinking-off-onespark`.
Their card JSON SHA256 values are
`deda98b35323209ee9919712e0f5dec8c33c88e8dbda1fcbfa62a11e517e9981`
and `24e785936f009a02a395717cbf1305c4828ac3eecd9be25749947af4291b7f7e`,
respectively. These cards are not yet in the repository.

**The token IDs match.** The saved comparison found identical IDs in all
66 responses, one-Spark ON against OFF, and each against both final
two-Spark cards. The one-Spark cards also match the one-Spark cards from
2026-10-24. This is a separate result from the speed table and from the
cards' basic output checks. It covers these requests, not every prompt.

### Reproduce this table

Use the build, checkpoint and request recipe named in the source card.
Install it as [INSTALL.md](../INSTALL.md) shows. Use the same private
configuration for every command. Set `VADE_API_URL` to the endpoint that
`vade status` reports. Keep model files, configuration and receipts outside
the source tree.

From the controller, start the pair with the measured limits:

```sh
vade up --sparks 2 --context 32768 --output 8192 --mtp 3
vade status --json > status-two-sparks.json
vade verify -n 3 --prompt short --max-tokens 256 --receipt ids-two-short.json
vade verify -n 3 --prompt 2k --max-tokens 256 --receipt ids-two-2k.json
vade verify -n 3 --prompt 8k --max-tokens 256 --receipt ids-two-8k.json
```

Each `verify` command must report `IDENTICAL` and exit successfully. It checks
the generated token IDs with drafting ON and OFF on the running engine.
Keep its receipt with the timing results. A missing ID or an incomplete
request is not an exactness pass. For independent engines on separate
Sparks, use the [second-engine check](../cli/README.md#compare-with-a-second-engine).
The ranks of a pair are one engine, so they do not count as that comparison.

The published speed table uses a pinned card client and RigMark's prompt
corpus. Its recipe is five natural-stop samples each for prose, code and
structured output, with an 8192-token cap; three cold/replay pairs at 8192
prompt tokens; and three rounds at each of C1, C2, C4 and C8, with a
256-token cap. Its seed is 20260905. The cards name upstream RigMark commit
`52c29c3fc1d741b9aef8895b23372aafa5dc6161`. Chat requests use temperature 0,
top-p 1, top-k 0, repetition penalty 1, presence and frequency penalties 0,
cache off and thinking off. ON omits `draft`; OFF sends `"draft":"plain"`.
Prefill uses token IDs at `/v1/completions`, with `draft` set to `plain`,
`max_tokens` set to 8 and `ignore_eos` true. It uses the same greedy fields
and a fresh cache salt for each cold/replay pair.

The optional external RigMark tool can run the same timed cells. Its prompt
corpus, request counts and timing formulas match the card recipe. Its
prefill requests differ: public RigMark omits top-p, top-k, penalties and
`draft`, which the card client sets explicitly. `--extra-body` affects chat
requests only. It cannot change that prefill body. See the
[pinned request code](https://github.com/alexellis/rigmark/blob/52c29c3fc1d741b9aef8895b23372aafa5dc6161/bench.py).

Use a separate checkout and set `RIGMARK_DIR` to it. Run the client on the
HTTP-serving Spark and set `VADE_API_URL` to its local endpoint. This
matches the client placement of the recorded cards.

```sh
cd "$RIGMARK_DIR"
git checkout --detach 52c29c3fc1d741b9aef8895b23372aafa5dc6161
./rigmark configure
```

Set `metadata.json` to the actual build, checkpoint and hardware. Set
`context_limit` to 32768. The cards use checkpoint revision
`b9ce18a6887e18c089fdea8509a33e63429a3300` and identity
`d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`.
Then run ON and OFF on the same server start. This pinned RigMark version
calls the command `run`.

```sh
./rigmark run --base-url "$VADE_API_URL" --model qwen3.8-flash-next-fp4-local \
  --label vade-two-served --comparison-id 2026-10-11-vade-rigmark-compat-thinking-off \
  --metadata metadata.json --runs 5 --decode-tokens 8192 \
  --prefill-depths 8192 --prefill-runs 3 --concurrency 1,2,4,8 \
  --concurrency-runs 3 --concurrency-tokens 256 --concurrency-workload code \
  --seed 20260905 --output rigmark-two-served.json \
  --extra-body '{"cache":false,"chat_template_kwargs":{"enable_thinking":false},"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0}'
./rigmark run --base-url "$VADE_API_URL" --model qwen3.8-flash-next-fp4-local \
  --label vade-two-plain --comparison-id 2026-10-11-vade-rigmark-compat-thinking-off \
  --metadata metadata.json --runs 5 --decode-tokens 8192 \
  --prefill-depths 8192 --prefill-runs 3 --concurrency 1,2,4,8 \
  --concurrency-runs 3 --concurrency-tokens 256 --concurrency-workload code \
  --seed 20260905 --output rigmark-two-plain.json \
  --extra-body '{"cache":false,"chat_template_kwargs":{"enable_thinking":false},"draft":"plain","top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0}'
```

**Exact card replay is not yet available from this repository.** The pinned
card client and its saved-card ID comparator are private measurement tools.
The public [RigMark instructions](#run-rigmark-against-vade) give the workload
settings. They do not supply that client or its token-ID comparison. A full
replay must save every request and token-ID array, match requests across
ON/OFF and Spark counts, and compare every ID in order. Saving IDs alone is
not a match check.

The Rust CLI can run this separate timing check on the same server:

```sh
vade bench --concurrency 1,2,4,8 --max-tokens 256 --repeats 3 \
  --output timing-two-served.json
vade bench --draft plain --concurrency 1,2,4,8 --max-tokens 256 --repeats 3 \
  --output timing-two-plain.json
vade down
```

These commands use the CLI's bundled prompt and fixed output lengths. They
do not reproduce the card's prose, code or structured cells. For the
one-Spark check, repeat in a separate receipt folder with
`vade up --sparks 1 --context 32768 --output 8192 --mtp 3`. Keep the same
checkpoint and request settings. Never label these CLI timings as a replay
of the table.

<a id="current-2026-10-27-main-12142bb-home-dgx-sparks"></a>

## Previous (2026-10-27 morning, main 12142bb, home DGX Sparks)

Two Sparks now run the **tensor split** by default, with no settings to set
(#121). `vade up --sparks 2` starts it as it is: both Sparks work on every
token, as the [2026-10-26 cards](#previous-2026-10-26-main-262b0c3-home-dgx-sparks)
describe. On those cards the split's four levels were still set by hand.
These cards set none of them.

These are RigMark cards measured on 2026-10-27 on two pairs of our own DGX
Sparks, at main `12142bb`. Each card passed its **15/15** basic output checks
and ran a prebuilt payload of its commit, installed and hash-checked on both
Sparks. The recipe is the one of the 2026-10-26 cards: context 32768, output
8192, MTP 3, greedy, cache off, thinking OFF; prose, code and structured 5
times with an 8192-token cap, a cold 8K prompt 3 times, and C1/C2/C4/C8 for
3 rounds with a 256-token cap. The drafting ON and OFF cards of each pair ran
on one server start. That start set no `VADE_W2_*` name. On both pairs the
serving receipt records the tensor split and the same four levels the
2026-10-26 cards set by hand: the experts split by output columns, the RDMA
write wire, projection level 3 and the output head split by vocabulary
halves.

| Path | Drafting | Commit | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Tensor split | ON | `12142bb` | home-1 + home-2 | 53.9 | 100.4 | 130.3 | 76.9 | 103.2 | 165.7 | 221.8 | 2.80 | `2026-10-27_0725_qwen3.8-flash-next-nvfp4_local-2spark_12142bb-drafting-on-thinking-off-split` |
| Tensor split | OFF | `12142bb` | home-1 + home-2 | 30.1 | 30.0 | 30.1 | 29.2 | 44.1 | 75.9 | 111.6 | 2.78 | `2026-10-27_0736_qwen3.8-flash-next-nvfp4_local-2spark_12142bb-drafting-off-thinking-off-split` |
| Tensor split | ON | `12142bb` | home-3 + home-4 | 53.5 | 99.7 | 129.7 | 75.2 | 103.0 | 164.8 | 221.6 | 2.80 | `2026-10-27_0741_qwen3.8-flash-next-nvfp4_local-2spark_12142bb-drafting-on-thinking-off-split` |
| Tensor split | OFF | `12142bb` | home-3 + home-4 | 29.7 | 29.6 | 29.8 | 28.8 | 43.8 | 75.3 | 110.8 | 2.77 | `2026-10-27_0753_qwen3.8-flash-next-nvfp4_local-2spark_12142bb-drafting-off-thinking-off-split` |

### What changed since the 2026-10-26 cards

**You set nothing.** The 2026-10-26 split cards set four `VADE_W2_*` names by
hand. These cards set none, and the receipts show the same path and levels.
`vade status` shows the path and its levels. The layer pipeline no longer
serves two Sparks.

**The speed is the same as with the levels set by hand.** With drafting ON,
prose was 53.9 and 53.5 tok/s here, against 54.2 on the 2026-10-26 split
card at `262b0c3`; code was 100.4 and 99.7 against 100.9, and eight chats
221.8 and 221.6 against 220.0. With drafting OFF, one chat wrote code at
30.0 and 29.6 tok/s, against 29.8. A cold 8K prompt took 2.77 to 2.80 s,
against 2.78 and 2.81 s on those cards.

**The answers did not change.** The token IDs matched in 66 of 66 responses:
drafting ON against OFF on the same server start, pair against pair, and
against the 2026-10-26 cards.

**Eight chats without drafting are still slower on the split.** They reached
111.6 and 110.8 tok/s here. The layer pipeline reached 137.0 in the
2026-10-26 cards. Drafting is on by default, so `vade up --sparks 2` serves
the ON rows; this gap is still on our list.

**Long prompts no longer wait at the swap, and no longer stall mid-read.**
Two changes since the 2026-10-26 cards change how a long prompt reaches its
first token on two Sparks. The prompt's last rows now run like its earlier
tiles, so the first token no longer waits while the two Sparks swap the
prompt's layer state (#114). That swap now runs before the second token, and
it goes GPU to GPU over the RDMA link instead of through host memory and
TCP. The first Spark also reads the second Spark's replies for each prompt
tile as they arrive, while its own GPU works (#120). Before, it read them
only late,
and on a long prompt from a chat that drafts, both Sparks could wait on each
other for seconds in the middle of the read.

The long prompts below ran on a second start of the same build on each pair,
with context 69632, because a 64K prompt does not fit in 32768. Drafting was
ON as served, the cache was off, and each prompt ran twice. The table gives
the first run of each prompt. The second runs were 10.09 s and 21.55 s on
home-1 + home-2 and 10.23 s and 21.51 s on home-3 + home-4. On both pairs,
the two runs of a prompt gave the same token IDs, and the pairs matched each
other.

| Sparks | 32K prompt, 32,099 tokens: first token (s) | 64K prompt, 64,217 tokens: first token (s) |
| --- | ---: | ---: |
| home-1 + home-2 | 10.10 | 21.54 |
| home-3 + home-4 | 10.24 | 21.60 |

<a id="current-2026-10-26-main-262b0c3-home-dgx-sparks"></a>

## Previous (2026-10-26, main 262b0c3, home DGX Sparks)

`vade up --sparks 2` runs the **tensor split**. Both Sparks work on every
token: each one computes half of every expert and of most projections, and for
one chat also half of the linear-attention heads and of the output head; the
two halves meet over the RDMA link. Before, two Sparks ran the **layer
pipeline**, where each Spark holds half of the layers and one token's work
passes from one GPU to the other. That path is gone: the split is the only
two-Spark path.

These are RigMark cards measured on 2026-10-26 on our own DGX Sparks. Each
card passed its **15/15** basic output checks and ran a prebuilt payload of
its commit, installed and hash-checked on both Sparks. `99e8f62` is the
published main of that afternoon; `262b0c3` also splits the experts of verify
steps of up to 32 rows, the size of an eight-chat step. The recipe is the one of
the [2026-10-25 cards](#previous-2026-10-25-main-80ac2d4-home-dgx-sparks):
context 32768, output 8192, MTP 3, greedy, cache off, thinking OFF; prose,
code and structured 5 times with an 8192-token cap, a cold 8K prompt 3 times,
and C1/C2/C4/C8 for 3 rounds with a 256-token cap. These builds came before
the split became the default, so their split cards set its four levels by hand
(`VADE_W2_TP=1 VADE_W2_TP_WIRE=write VADE_W2_TP_PROJ=3 VADE_W2_TP_HEAD=1`); the
same four are now the only code, and those names are gone.

| Path | Drafting | Commit | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Tensor split | ON | `262b0c3` | home-1 + home-2 | 54.2 | 100.9 | 130.4 | 76.8 | 101.0 | 161.4 | 220.0 | 2.81 | `2026-10-26_2157_qwen3.8-flash-next-nvfp4_local-2spark_262b0c3-drafting-on-thinking-off-split` |
| Tensor split | OFF | `262b0c3` | home-1 + home-2 | 30.0 | 29.8 | 30.1 | 29.0 | 43.8 | 74.6 | 110.6 | 2.78 | `2026-10-26_2209_qwen3.8-flash-next-nvfp4_local-2spark_262b0c3-drafting-off-thinking-off-split` |
| Tensor split | ON | `99e8f62` | home-1 + home-2 | 54.2 | 100.3 | 130.2 | 77.8 | 100.7 | 155.4 | 169.5 | 2.81 | `2026-10-26_1615_qwen3.8-flash-next-nvfp4_local-2spark_99e8f62-drafting-on-thinking-off-split` |
| Tensor split | ON | `99e8f62` | home-3 + home-4 | 54.0 | 100.4 | 131.2 | 75.4 | 100.5 | 149.8 | 176.3 | 2.79 | `2026-10-26_1613_qwen3.8-flash-next-nvfp4_local-2spark_99e8f62-drafting-on-thinking-off-split` |
| Tensor split | OFF | `99e8f62` | home-1 + home-2 | 29.9 | 30.0 | 30.3 | 29.1 | 43.9 | 74.4 | 110.8 | 2.76 | `2026-10-26_1627_qwen3.8-flash-next-nvfp4_local-2spark_99e8f62-drafting-off-thinking-off-split` |
| Tensor split | OFF | `99e8f62` | home-3 + home-4 | 29.8 | 29.6 | 29.9 | 28.9 | 43.5 | 72.4 | 110.4 | 2.76 | `2026-10-26_1625_qwen3.8-flash-next-nvfp4_local-2spark_99e8f62-drafting-off-thinking-off-split` |
| Layer pipeline | ON | `99e8f62` | home-5 + home-6 | 42.2 | 80.3 | 100.8 | 61.9 | 91.6 | 143.2 | 201.7 | 2.78 | `2026-10-26_1611_qwen3.8-flash-next-nvfp4_local-2spark_99e8f62-drafting-on-thinking-off` |
| Layer pipeline | OFF | `99e8f62` | home-5 + home-6 | 28.2 | 28.0 | 28.2 | 27.5 | 47.2 | 85.9 | 137.0 | 2.78 | `2026-10-26_1623_qwen3.8-flash-next-nvfp4_local-2spark_99e8f62-drafting-off-thinking-off` |

### What the split changes

**One chat writes faster.** With drafting ON (the default), the split wrote
prose at 54.2 tok/s against 42.2 on the pipeline, code at 100.9 against 80.3
and structured output at 130.4 against 100.8. With drafting OFF, one chat went
from 28.0 to 29.8 tok/s of code.

**Eight chats with drafting ON are faster too, on this build.** C8 was 220.0
tok/s on the split at `262b0c3`, against 201.7 on the pipeline. On `99e8f62`
the split reached only 169.5 and 176.3: an eight-chat verify step has more
than 16 rows, and that build ran the experts of such a step whole on each
Spark. `262b0c3` splits them up to 32 rows.

**Eight chats without drafting are slower on the split.** With drafting OFF,
C8 was 110.6 tok/s on the split against 137.0 on the pipeline, and C4 74.6
against 85.9. Drafting is on by default, so `vade up --sparks 2` serves the
ON rows; this gap is on our list.

**The cold 8K prompt takes the same time:** 2.81 s on the split, 2.78 s on
the pipeline. Much longer prompts reached their first token later on the
split on these builds, because the first token waited for the two Sparks to
swap the prompt's layer state through the host. Two later changes move that
swap after the first token and stop a stall in the middle of a long read.
The [2026-10-27 cards](#current-2026-10-27-main-12142bb-home-dgx-sparks) run
with both changes and give the first-token times. These cards do not measure
them.

**Every level is the only code.** `vade status` shows the path and its
levels, and the engine's serving receipt records them under
`vade.numerics.path` and `vade.numerics.tensor_split`.

<a id="current-2026-10-25-main-80ac2d4-home-dgx-sparks"></a>

## Previous (2026-10-25, main 80ac2d4, home DGX Sparks)

These are RigMark cards measured on the morning of 2026-10-25 on our own DGX
Sparks, on main `80ac2d4` and on `6cb46bf`, the main commit from earlier that
morning; each row names its commit. Every number is copied from the card in
the result folder named on its row, and that folder's README gives the boxes,
launcher and settings. Each card passed its **15/15** basic output checks.
Unlike the v0.1.0 cards, these did not install from the public repository:
each ran a prebuilt payload of its commit, installed and hash-checked on every
Spark.

There is no one-Spark card in this set. The one-Spark rows for comparison are
in the [previous section](#previous-2026-10-24-v010-release-5a22651), measured
on 2026-10-24 at `5a22651`.

**FP4-Local recipe.** Model `nvidia/Qwen3.8-Flash-Next-FP4-Local`; the lab
launcher with the `vade up --sparks 2` or `--sparks 4` controller defaults
(context 32768, output 8192, MTP 3); the layer pipeline, each Spark holding a
slice of the model's layers; greedy decoding; cache off; thinking OFF. Each
card ran single-stream prose, code and structured writing **5 times** with an
**8192-token cap**, a cold 8K prompt **3 times**, and C1/C2/C4/C8 for **3
rounds** with a **256-token cap**. In the ON cards requests are sent as they
are (drafting is the server default); in the OFF cards every request sends
`"draft":"plain"`. The OFF and ON card of a pair ran on one server start,
except the `80ac2d4` pair, where the OFF card ran on a second start of the
same launcher, payload and flags.

**FP8 recipe.** Checkpoint `Qwen/Qwen3.8-Flash-Next-FP8`; a lab launcher
script, not the public `vade up` path (context 32768, output 16384, MTP 2);
the world-2 ring or the world-4 path; greedy; thinking OFF; cache off for the
decode and concurrency rows. (The cold 8K pair includes a replay that restores
all 8192 tokens; the cold numbers published here restore zero.) Three things
differ from the FP4-Local cards:

- The cold 8K prompt was measured **once**, one cold/replay pair instead of
  three, because FP8 takes longer per cold prompt (6.00 to 7.05 s in these
  cards).
- Drafting on FP8 works only at one chat. In the ON cards, the single-stream,
  prefill and C1 responses were served at draft depth 2, and every C2, C4 and
  C8 response was served plain, so the ON and OFF cards differ only in the
  one-chat columns.
- The `6cb46bf` ON and OFF cards of each size ran on one server start, with
  **62/62** identical token IDs between them, as their READMEs record.

In the tables, prose, code and structured are one chat at a time, in tok/s.
C1, C2, C4 and C8 are 1, 2, 4 and 8 chats at once, in tok/s for the whole
group. Cold 8K prompt is the seconds to the first token after an 8K-token
prompt the server has not seen. Structured output is a predictable JSON
workload, not a measure of general agent performance.

### FP4-Local

| Setup | Drafting | Commit | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two DGX Sparks (home-1 + home-2) | OFF | `80ac2d4` | 27.9 | 27.9 | 28.2 | 27.3 | 46.9 | 80.9 | 136.2 | 2.80 | `2026-10-25_1003_qwen3.8-flash-next-nvfp4_local-2spark_80ac2d4-drafting-off-thinking-off` |
| Two DGX Sparks (home-1 + home-2) | ON | `80ac2d4` | 42.0 | 79.3 | 98.6 | 60.8 | 91.2 | 142.9 | 176.4 | 2.80 | `2026-10-25_0941_qwen3.8-flash-next-nvfp4_local-2spark_80ac2d4-drafting-on-thinking-off` |
| Two DGX Sparks (home-5 + home-6) | OFF | `6cb46bf` | 28.1 | 28.0 | 28.4 | 27.5 | 47.2 | 83.2 | 135.3 | 2.78 | `2026-10-25_0807_qwen3.8-flash-next-nvfp4_local-2spark_6cb46bf-drafting-off-thinking-off` |
| Two DGX Sparks (home-5 + home-6) | ON | `6cb46bf` | 41.6 | 79.2 | 98.7 | 60.7 | 91.3 | 141.5 | 174.7 | 2.78 | `2026-10-25_0813_qwen3.8-flash-next-nvfp4_local-2spark_6cb46bf-drafting-on-thinking-off` |
| Four DGX Sparks (home-1 + home-2 + home-3 + home-4) | OFF | `6cb46bf` | 28.4 | 28.4 | 28.7 | 27.8 | 47.8 | 82.6 | 137.5 | 2.19 | `2026-10-25_0836_qwen3.8-flash-next-nvfp4_local-4spark_6cb46bf-drafting-off-thinking-off` |
| Four DGX Sparks (home-1 + home-2 + home-3 + home-4) | ON | `6cb46bf` | 42.1 | 80.3 | 101.4 | 62.1 | 92.3 | 141.3 | 136.5 | 2.19 | `2026-10-25_0842_qwen3.8-flash-next-nvfp4_local-4spark_6cb46bf-drafting-on-thinking-off` |

### FP8

| Setup | Drafting | Commit | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two DGX Sparks (home-1 + home-2) | OFF | `6cb46bf` | 26.7 | 26.6 | 26.7 | 25.7 | 39.8 | 63.2 | 94.5 | 6.00 | `2026-10-25_0905_qwen3.8-flash-next-fp8_local-2spark_6cb46bf-drafting-off-thinking-off` |
| Two DGX Sparks (home-1 + home-2) | ON | `6cb46bf` | 34.0 | 47.6 | 49.2 | 42.2 | 40.8 | 64.5 | 94.8 | 6.12 | `2026-10-25_0853_qwen3.8-flash-next-fp8_local-2spark_6cb46bf-drafting-on-thinking-off` |
| Two DGX Sparks (home-3 + home-4) | ON | `80ac2d4` | 34.0 | 48.0 | 49.1 | 42.1 | 40.4 | 63.8 | 94.8 | 6.14 | `2026-10-25_0945_qwen3.8-flash-next-fp8_local-2spark_80ac2d4-drafting-on-thinking-off` |
| Four DGX Sparks (home-1 + home-2 + home-3 + home-4) | OFF | `6cb46bf` | 34.6 | 34.4 | 34.5 | 33.3 | 53.7 | 83.6 | 125.2 | 6.88 | `2026-10-25_0814_qwen3.8-flash-next-fp8_local-4spark_6cb46bf-drafting-off-thinking-off` |
| Four DGX Sparks (home-1 + home-2 + home-3 + home-4) | ON | `6cb46bf` | 44.5 | 62.5 | 65.5 | 56.2 | 51.9 | 82.7 | 122.0 | 7.05 | `2026-10-25_0821_qwen3.8-flash-next-fp8_local-4spark_6cb46bf-drafting-on-thinking-off` |

The FP8 drafting-OFF card at `80ac2d4` was not run; its README records a
hard cutoff at 09:45.

### Reading these cards

**Drafting ON is the lever for one chat.** On the `80ac2d4` pair it took code
from 27.9 to 79.3 tok/s, prose from 27.9 to 42.0 and structured from 28.2 to
98.6. On four Sparks it took code from 28.4 to 80.3. On FP8 it took code from
26.6 to 47.6 on two Sparks and from 34.4 to 62.5 on four.

**A second or a fourth Spark does not make one chat faster on this build.**
The FP4-Local world-2 and world-4 path is a layer pipeline: each Spark holds a
slice of the layers, and one token's work passes from GPU to GPU, so at any
moment one GPU is working on a given chat. With drafting ON, one chat wrote
79.3 tok/s of code on two Sparks and 80.3 on four, next to 76.7 on one Spark
in the v0.1.0 card. With drafting OFF it sat at 27.9 on two Sparks and 28.4 on
four, next to 27.8 on one.

**What more Sparks buy is prompt reading and room.** The pipeline splits the
prompt work, so the first token after a cold 8K prompt came in 2.80 s on two
Sparks and 2.19 s on four, against 3.86 s on one Spark in the v0.1.0 card.
The batched rates held as Sparks were added (C4 with drafting ON: 142.9 on
two, 141.3 on four), with more memory behind them for context and parallel
chats.

**Eight chats on four Sparks ran without drafting.** C8 with drafting ON was
176.4 on the `80ac2d4` pair but 136.5 on four Sparks, below that card's own
C4. The receipts show why: on four Sparks all 24 eight-chat responses were
served plain (draft depth 0), while on two Sparks 12 of the 24 were served
with drafting at depth 3. "ON" names the requested default; on this build the
server falls back to plain decoding for eight chats on the four-stage
pipeline. That fallback is the next thing to fix.

**The layer-pipeline change merged on 2026-10-25 is in these cards.** On
two Sparks with drafting ON, code went from 65.1 tok/s on the v0.1.0 pair to
79.3 at `80ac2d4`, prose from 38.3 to 42.0 and structured from 67.7 to 98.6,
while the cold 8K prompt stayed at 2.80 s (2.79 s on the v0.1.0 pair).

**FP8 reads long prompts slowly on this path.** The cold 8K prompt took
6.00 to 7.05 s on FP8 against 2.19 to 2.80 s on FP4-Local, measured once per
FP8 card, and the FP8 cards used output 16384 and MTP 2 where the FP4-Local
cards used output 8192 and MTP 3, so the two tables are not a like-for-like
comparison of the checkpoints.

<a id="current-2026-10-24"></a>
<a id="current-results-2026-10-24"></a>

## Previous (2026-10-24, v0.1.0 release 5a22651)

[RigMark](https://github.com/alexellis/rigmark) is a public benchmark suite,
and a card is the recorded result of one RigMark run. These are the cards for
the v0.1.0 release, commit `5a22651`, installed from the public repository
and measured on the evening of 2026-10-24. Every number in this section is
copied from a card or from the README in the same receipt folder.

**Recipe.** Model `nvidia/Qwen3.8-Flash-Next-FP4-Local`; the public
`vade launch` and `vade up` defaults (context 32768, output 8192); greedy
decoding; cache off; thinking OFF. Each card ran single-stream prose, code and
structured writing **5 times** with an **8192-token cap**, a cold 8K prompt
**3 times**, and C1/C2/C4/C8 for **3 rounds** with a **256-token cap**.
Drafting is Vade's speculative decoding: an extra model head proposes tokens
with multi-token prediction (MTP), and the model checks them. It is ON by
default, so the ON cards send requests as they are; in the OFF cards, every
request sends `"draft":"plain"`.

In the tables, prose, code and structured are one chat at a time. C1, C2, C4
and C8 are 1, 2, 4 and 8 chats at once, and their rate covers the whole group.
TTFT is the time to the first token, measured at the client, and tok/s is
completion tokens per second of wall time. Structured output is a predictable
JSON workload, not a measure of general agent performance.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks-2026-10-24-dark.svg">
  <img src="assets/benchmarks-2026-10-24-light.svg" width="100%" alt="Bar chart comparing one DGX Spark with two, drafting ON (the default). In the 2026-10-24 runs, a second Spark answered a long prompt faster, but one chat did not write faster. Code, one chat: 76.7 tok/s on 1 Spark, 65.1 on 2 Sparks. Prose, one chat: 41.7 and 38.3. Eight chats together: 173.9 and 168.6. First reply after an 8K-token prompt: 3.86 s on 1 Spark, 2.79 s on 2 Sparks. Across all our runs, eight chats gave 130.6 to 173.9 tok/s on six single Sparks and 167.6 to 168.7 on three pairs.">
</picture>

The figure compares one of our Sparks with two, both with drafting ON, the
default. It draws two cards from the home table below: one Spark ON and two
Sparks ON. Drafting OFF and the cloud VMs are in the tables only. To redraw
it, run `cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- benchmarks-chart`; it reads the receipts and
rewrites both SVG files in [docs/assets](assets/README.md).

### Home

These cards ran on our own DGX Sparks (driver 580.178.04). Both one-Spark
cards ran on home-1. The pair, home-1 and home-2, runs the layer pipeline:
each Spark holds a slice of the model's layers and passes its work to the
next.

| Setup | Drafting | Prose tok/s | Code tok/s | Structured tok/s | Cold 8K TTFT (s) | Cold prefill tok/s | C1 | C2 | C4 | C8 | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| One DGX Spark | OFF | 27.8 | 27.8 | 28.1 | 3.82 | 2,145 | 27.1 | 47.1 | 85.2 | 136.1 | [`2026-10-24_1937_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1937_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/) |
| One DGX Spark | ON | 41.7 | 76.7 | 88.7 | 3.86 | 2,124 | 61.4 | 91.7 | 141.5 | 173.9 | [`2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/) |
| Two DGX Sparks | OFF | 27.9 | 27.9 | 28.2 | 2.80 | 2,923 | 27.3 | 47.1 | 84.9 | 135.1 | [`2026-10-24_2012_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_2012_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off/) |
| Two DGX Sparks | ON | 38.3 | 65.1 | 67.7 | 2.79 | 2,931 | 55.7 | 86.8 | 137.8 | 168.6 | [`2026-10-24_2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off/) |

### Cloud

These cards ran on four rented GB10 VMs from one provider, which we call cloud
A (vm-a to vm-d). Each is a container with one GB10, 116 GiB RAM and driver
580.173.02, and they are joined by 200G point-to-point links over TCP. The
one-VM cards ran on vm-a alone. The two-VM cards ran vm-a and vm-b as one
engine, and the four-VM cards ran all four as a chain, each holding a slice of
the model's layers.

| Setup | Drafting | Prose tok/s | Code tok/s | Structured tok/s | Cold 8K TTFT (s) | Cold prefill tok/s | C1 | C2 | C4 | C8 | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| One cloud GB10 VM | OFF | 22.5 | 24.4 | 26.2 | 3.99 | 2,055 | 24.1 | 42.5 | 75.2 | 73.3 | [`2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vma`](../bench/receipts/2026-10-24/2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vma/) |
| One cloud GB10 VM | ON | 31.6 | 61.1 | 77.3 | 3.98 | 2,057 | 49.1 | 75.6 | 112.9 | 97.5 | [`2026-10-24_1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vma`](../bench/receipts/2026-10-24/2026-10-24_1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vma/) |
| Two cloud GB10 VMs | OFF | 22.9 | 24.4 | 25.0 | 2.95 | 2,774 | 23.2 | 41.2 | 73.1 | 112.2 | [`2026-10-24_2027_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-cloud-a`](../bench/receipts/2026-10-24/2026-10-24_2027_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-cloud-a/) |
| Two cloud GB10 VMs | ON | 30.5 | 54.7 | 59.3 | 2.95 | 2,774 | 46.0 | 72.7 | 112.4 | 138.3 | [`2026-10-24_2035_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-cloud-a`](../bench/receipts/2026-10-24/2026-10-24_2035_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-cloud-a/) |
| Four cloud GB10 VMs | OFF | 20.7 | 21.9 | 22.6 | 2.38 | 3,439 | 21.0 | 37.5 | 68.3 | 104.2 | [`2026-10-24_2204_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-off-thinking-off-cloud-a`](../bench/receipts/2026-10-24/2026-10-24_2204_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-off-thinking-off-cloud-a/) |
| Four cloud GB10 VMs | ON | 29.0 | 51.6 | 56.6 | 2.42 | 3,390 | 43.6 | 70.7 | 108.1 | 104.2 | [`2026-10-24_2213_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-on-thinking-off-cloud-a`](../bench/receipts/2026-10-24/2026-10-24_2213_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-on-thinking-off-cloud-a/) |

### What this means

**Drafting is the big win.** On one Spark it takes code from 27.8 to 76.7
tok/s and prose from 27.8 to 41.7, and it makes one chat faster on every
setup we measured.

**In the 2026-10-24 runs, a second Spark read prompts faster but did not write
faster.** The first token after a cold 8K prompt came in 2.80 s on two Sparks
instead of 3.82 s on one. One chat was no faster: with drafting ON, code was
65.1 tok/s on the pair against 76.7 on one Spark, and eight chats together
landed close to one Spark's rate. A change to the layer pipeline, merged on
2026-10-25 after those runs, raised two-Spark code to 79.3 tok/s; see
[reading these cards](#reading-these-cards).

**In the cloud, a second VM serves more chats; four do not.** Eight chats
together went from 73.3 tok/s on one VM to 112.2 on two with drafting OFF, and
from 97.5 to 138.3 with drafting ON. Four VMs gave 104.2 either way. They did
read a cold 8K prompt fastest (2.38 s), but one chat was slower there than on
one VM. A chain is for models too large for a single machine; in these runs it
did not make one chat faster.

These are separate runs on the machines named above, not a controlled
comparison on one machine.

### Known limits

- **One chat barely moves from Spark to Spark.** On six of our Sparks measured
  one at a time, drafting OFF gave 27.6 to 27.8 tok/s on prose and 27.7 to
  27.8 on code, and drafting ON gave 41.3 to 41.7 on prose and 76.1 to 76.7 on
  code.
- **The 8-chat rate varies from run to run.** At eight chats the engine serves
  some chats without drafting; each card's README says how many. On the same
  six Sparks, the drafting-ON 8-chat rate ranged from 130.6 to 173.9 tok/s, and
  one drafting-OFF run (home-6) landed at 86.6, a drop its receipt does not
  explain. The three home pairs agreed closely: 134.3 to 135.2 with drafting
  OFF and 167.6 to 168.7 with drafting ON.
- **The cloud 8-chat rate varies too.** Across the single-VM cards on cloud A
  (six per setting), it ranged from 97.5 to 109.6 tok/s with drafting ON and
  from 69.0 to 74.9 with drafting OFF. In the four-VM drafting-ON run the
  engine served all eight chats without drafting, which is why both four-VM
  rows show 104.2.
- **Cloud VMs are slower on one chat than our Sparks.** Across the single-VM
  cards on cloud A (six per setting), drafting OFF gave 22.1 to 22.9 tok/s on
  prose and 23.2 to 25.7 on code; drafting ON gave 31.1 to 32.0 on prose and
  56.1 to 63.1 on code. A single GB10 container at a second provider (cloud B)
  gave prose 21.9, code 26.4 and eight chats 118.7 with drafting OFF, and 37.2,
  72.2 and 133.5 with drafting ON
  ([OFF](../bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-b/),
  [ON](../bench/receipts/2026-10-24/2026-10-24_1949_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-b/)).
- **Output checks are not exactness.** Each card passed its **15/15** basic
  output checks (the card calls them gates). Those checks do not prove
  identical token IDs on every run, or model quality; `vade verify` and the
  [exactness rule](architecture.md#exactness-and-model-quality) cover that.
  The cards' replay TTFT is an immediate repeat of a prompt, not proof of a
  cache hit.

The [receipt folder](../bench/receipts/2026-10-24/README.md) holds every
v0.1.0 receipt, including the other Sparks and VMs behind these ranges.

### How to reproduce

1. Install v0.1.0 as [INSTALL.md](../INSTALL.md) shows, then start the engine
   with `vade launch` on one Spark or `vade up --sparks 2` from a controller.
   Keep the default context and output.
2. Run `vade bench` with no flags for a quick timing check. It measures the
   server as it is (drafting ON) and writes `./vade-bench.json` with a readable
   `.txt` table next to it; it is not a card. See [Run vade bench](#run-vade-bench).
3. For a card, run RigMark as [Run RigMark against Vade](#run-rigmark-against-vade)
   shows: once with requests as served (drafting ON) and once with
   `"draft":"plain"` in every request (drafting OFF).
4. Compare your card with the row for the same number of machines and the same
   drafting setting.

## Earlier cards from the same day (main 4681d8b and 73f7a9d)

Every run below used the same recipe: the model
`nvidia/Qwen3.8-Flash-Next-FP4-Local`, context **32768**, greedy sampling (the
most likely token every time), cache off, and thinking on or off as each row
says.

The terms the tables use:

- **RigMark card**: RigMark is a public benchmark suite, and a card is the
  recorded result of one run of it: single-stream prose, code and structured
  writing, cold 8K prompt reading, and capped concurrent streams.
- **Drafting**: Vade's speculative decoding. An extra model head proposes
  tokens with multi-token prediction (MTP), and the target model verifies
  them. A card has **drafting OFF** unless its folder name says `drafting-on`;
  OFF requests send `"draft":"plain"`.
- **C1, C2, C4 and C8**: 1, 2, 4 and 8 concurrent streams.
- **TTFT**: time to first token, measured at the client.
- **Tok/s**: completion tokens per second of wall time. A token is a small
  piece of text, often part of a word.
  - For concurrent streams, the rate covers the whole group of streams (the
    cohort).
  - The cards mark their single-stream decode speeds as estimates, and those
    speeds count streamed reasoning too.
  - The separate timing runs divide by the whole request time, so their rates
    are not computed the same way as the cards' decode column.

**Which builds we measured.** The tables for our own Sparks use the
afternoon's **release-build cards**, made from commit `73f7a9d` on the main
branch (written as main `73f7a9d` from here on). The v0.1.0 engine changes one
thing compared with that build: its startup memory check, which the engine
calls admission, now counts free memory differently (main `2c64d0a`). The
v0.1.0 cards are in the
[section above](#previous-2026-10-24-v010-release-5a22651).

The cloud cards and the first two cloud timing runs come from main
`4681d8b`, earlier the same day; the third timing run used build `8c9411f`.

Each row quotes its source folder and links to the sanitized copy in this
repository. Card numbers come from that folder's `*.card.txt`; recipe details
come from the card and its `README.txt`. For folders marked **timing run, not
a card**, the results are in the folder's README, with the timing JSON (one
entry per request or per round) next to it.

### Our DGX Sparks

All six rows here are release-build (`73f7a9d`) receipts. The one-Spark OFF and
ON cards ran on **two different Sparks of ours** with the same build; we did
not alternate modes on one machine, and their clocks differed too. Thinking ON
in these cards means thinking at low reasoning effort, shown as ON/low.

Each one-Spark card ran:

- single-stream cells: **5 runs** with an **8192-token cap**;
- cold prefill (reading the prompt): **3 pairs**;
- C1/C2/C4/C8: **3 rounds** with a **256-token cap**.

Structured output is a predictable JSON workload, not a measure of general
agent performance.

| Hardware | Drafting | Thinking | Prose tok/s | Code tok/s | Structured tok/s | Cold 8K TTFT (s) | Cold prefill tok/s | C1 | C2 | C4 | C8 | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| One DGX Spark | OFF | OFF | 27.7 | 27.7 | 28.0 | 3.87 | 2,118 | 27.1 | 46.9 | 85.0 | 135.9 | [`2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off/) |
| One DGX Spark | ON | OFF | 42.0 | 76.7 | 88.9 | 3.85 | 2,129 | 61.0 | 91.2 | 144.4 | 172.6 | [`2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off/) |
| One DGX Spark | OFF | ON/low | 27.6 | 27.7 | 28.0 | 3.82 | 2,144 | 26.7 | 46.3 | 79.9 | 127.2 | [`2026-10-24_1458_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-on`](../bench/receipts/2026-10-24/2026-10-24_1458_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-on/) |
| One DGX Spark | ON | ON/low | 43.2 | 73.8 | 86.2 | 4.08 | 2,008 | 52.1 | 84.7 | 128.6 | 153.4 | [`2026-10-24_1445_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-on`](../bench/receipts/2026-10-24/2026-10-24_1445_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-on/) |

Our two-Spark pair runs the layer pipeline: each Spark holds a contiguous range
of the model's layers and passes its intermediate state to the next. Its card
files are summaries from a custom runner, with a different workload from
RigMark's: **3** prose samples with a **1024-token cap**, **6** cold prefill
samples, and **8 rounds** for each group of streams with a **256-token cap**,
after a warmup, so its prose rows are not directly comparable with the
one-Spark RigMark workload.

Both rows come from **one server start**, OFF block first. For C16 the runner
offered **16** streams at once; the engine admitted **9** of them as active
streams, the rest were retried, and their wait counts in the group's wall time.
In all **201/201** measured pairs, drafting ON produced the same token IDs as
drafting OFF.

| Hardware | Drafting | Thinking | Prose tok/s | Cold 8K TTFT (s) | C8 tok/s | C16 tok/s | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Two DGX Sparks | OFF | OFF | 27.705 | 2.776 | 133.877 | 135.197 | [`2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off/) |
| Two DGX Sparks | ON | OFF | 38.386 | 2.853 | 162.558 | 147.049 | [`2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off/) |

**Release build versus the morning build.** On one Spark, every single-stream
and C8 cell of the release build is within **1 %** of the morning card with the
same settings. The largest moves were in other cells: C4 with drafting OFF and
thinking ON moved from **81.9** to **79.9 tok/s**, and cold 8K TTFT with
drafting ON and thinking ON moved from **3.81** to **4.08 s**. The morning
and release-build one-Spark cards ran on different Sparks of ours.

On two Sparks, C8 with drafting OFF moved from **134.561** to
**133.877 tok/s**. With drafting ON, **C8 went from 134.421 to
162.558 tok/s** and C16 from **134.241** to **147.049**. The release build
includes the two-Spark drafting fix from main `407b6ae`, described under
[known limits](#known-limits-1).

The morning cards from the previous build, main `4681d8b`, are linked here.
In each one-Spark link the first ON/OFF is drafting and the second is
thinking; the pair links name drafting only. One Spark:
[0656 OFF/OFF](../bench/receipts/2026-10-24/2026-10-24_0656_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-off/), [0650 ON/OFF](../bench/receipts/2026-10-24/2026-10-24_0650_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-off/),
[0710 OFF/ON](../bench/receipts/2026-10-24/2026-10-24_0710_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-on/), [0657 ON/ON](../bench/receipts/2026-10-24/2026-10-24_0657_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-on/); our pair:
[0720 OFF](../bench/receipts/2026-10-24/2026-10-24_0720_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/), [0711 ON](../bench/receipts/2026-10-24/2026-10-24_0711_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-on-thinking-off/).

### Cloud GB10 VMs

These ran on **GB10 virtual machines (VMs) rented from a cloud provider**, with
**116 GB RAM each**; see [renting GB10s in the cloud](#rented-cloud-gb10-vms)
for what to expect when you set one up.

Both cards follow the same single-stream and capped-concurrency recipe as our
one-Spark cards above, with drafting OFF and thinking OFF. The pair ran the
layer pipeline over a 200G QSFP link between the two VMs, using RoCE (RDMA over
Converged Ethernet); the one-VM card ran on one VM alone. The pair's drivers
were **580.178.04** and **580.159.03**; the one-VM card used **580.159.03**.

Both cloud cards set an explicit 100 GiB engine memory budget
(`SEQPAR_MEMORY_BYTES`), because inside the VMs' containers the engine's
derived memory ceiling came out slightly short; the
[cloud section below](#rented-cloud-gb10-vms) explains why.

| Hardware | Drafting | Thinking | Prose tok/s | Code tok/s | Structured tok/s | Cold 8K TTFT (s) | Cold prefill tok/s | C1 | C2 | C4 | C8 | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two cloud GB10 VMs | OFF | OFF | 23.2 | 25.4 | 26.5 | 3.03 | 2,701 | 21.7 | 40.0 | 71.2 | 70.0 | [`2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/) |
| One cloud GB10 VM | OFF | OFF | 23.4 | 24.7 | 26.0 | 3.89 | 2,105 | 24.7 | 42.1 | 42.1 | 43.0 | [`2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off/) |

The next rows are **timing runs, not cards**. Each one ran ON, then OFF, then
ON on a server with drafting enabled. ON requests sent `"draft":"strict"` and
OFF requests sent `"draft":"plain"`. All three had thinking OFF.

- Code medians use **5** OFF and **10** ON samples; prose uses **1** OFF and
  **2** ON samples.
- The third run repeated the two-VM sequence on build `8c9411f`. Its served
  engine is identical to main `407b6ae`, the fix described under known limits
  below.
- In the one-VM run, the MTP-depth field in the responses read zero even for
  ON requests. Its README therefore uses the draft field each request sent to
  tell ON from OFF.

| Hardware | What it is | Thinking | OFF code tok/s | ON code tok/s | OFF prose tok/s | ON prose tok/s | C8 result | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| One cloud GB10 VM | timing run, not a card | OFF | 25.8 | 64.0 | 24.5 | 33.6 | Not measured | [`2026-10-24_1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card`](../bench/receipts/2026-10-24/2026-10-24_1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card/) |
| Two cloud GB10 VMs | timing run, not a card | OFF | 26.2 | 56.5 | 25.2 | 32.0 | First ON round 85.4 tok/s; later ON rounds failed | [`2026-10-24_1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card`](../bench/receipts/2026-10-24/2026-10-24_1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card/) |
| Two cloud GB10 VMs, build `8c9411f` | timing run, not a card | OFF | 26.3 | 57.0 | 25.4 | 31.6 (median of 29.8 and 33.4) | ON rounds 81.3 / 92.0 / 92.2 tok/s, OFF rounds 78.4 / 80.9 / 78.9 tok/s, 0 errors | [`2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card`](../bench/receipts/2026-10-24/2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card/) |

### What this means

In these measurements, turning drafting on makes the biggest difference. On
one of our Sparks, code goes from **27.7** to **76.7 tok/s** and C8 from
**135.9** to **172.6 tok/s**; in the one-VM cloud timing run, code goes from
**25.8** to **64.0 tok/s**.

In the 2026-10-24 OFF cards, a second Spark did not substantially raise
single-stream speed. It did improve prefill and could raise batched
throughput: cloud C8 went from **43.0** to **70.0 tok/s**. On our own pair with
drafting OFF, C8 was effectively the same as on one Spark (**133.877** versus
**135.9 tok/s**); with drafting ON the release-build pair reached
**162.558 tok/s**, below one Spark's **172.6 tok/s**.

The rented VMs are broadly similar to our Sparks on single-stream speed and
prefill, but their batched decode levels off: one VM's C2/C4/C8 rates are
**42.1 / 42.1 / 43.0 tok/s**. The pair reaches **71.2 / 70.0 tok/s** at C4/C8,
with C2 at **40.0 tok/s**.

These are observations across the named runs, not a controlled comparison on
the same machine.

### Known limits

- **Less memory headroom on the 116 GB VMs.** The first two-VM timing README
  (build `4681d8b`) records an engine memory ceiling of **88.4 GiB**. After a
  successful C8 drafting round and then a plain round, the next drafting
  rounds failed: **8/8** requests in each returned `generation_error`. The
  cause was too little allocation headroom once pooled storage had grown.

  Main `407b6ae` fixed this by changing the pipeline's C8 policy and
  reclaiming the workers' decode pools. The 12:59 rerun of the same sequence
  on that engine completed **3/3** drafting rounds after plain rounds with
  **0** errors (**81.3 / 92.0 / 92.2 tok/s**). Pooled storage reached
  **94.6 GB**, under the budget for each rank (a rank is one machine of the
  pair). The 116 GB VMs still have less headroom than our **128 GB** Sparks.
- **Drafting on our pair at C8, before and after the fix.** On the morning
  card build, `4681d8b`, drafting did not help our pair at C8: ON and OFF were
  effectively equal, **134.421** versus **134.561 tok/s**. The fix in main
  `407b6ae` raised it on the same pair from **133.871** to **165.560 tok/s**
  in an alternating before/after comparison (baseline, fix, fix, baseline).
  That comparison is not a card and its own receipt is not published; the
  12:59 README quotes the same rates and rounds the gain to +23.7 %. The
  release-build cards above confirm the change: **162.558** ON versus
  **133.877 tok/s** OFF.
- **Output checks are not exactness.** The one-Spark ON cards from our Sparks
  report **15/15** basic output checks passed (the cards call these checks
  gates), while their READMEs mark the runs `PARTIAL`. Those checks do not
  establish exactness (identical token IDs on every run and every Spark) or
  model quality. Replay TTFT is an immediate repeat, not proof of a cache hit.

## Reproduce a measurement in detail

To compare your Spark with the tables, run RigMark as the next section shows.
`vade bench` gives you a timing check of your own, but it does not produce
cards.

Follow the [FP4-Local serving guide](../cli/docs/fp4-local.md) and keep the
measured context and output limits explicit. `vade launch` (run on the Spark
itself) and `vade up` (run from a controller, the computer that drives your
Sparks) both start at context 32768 and output 8192. `vade launch` lowers a
default context only when the engine refuses it for memory; a controller never
lowers it, so check `vade status --json` before you compare runs.

After `vade launch` or `vade up`, `vade bench` works without flags. Keep the
same `VADE_HOME` / `VADE_CONFIG` and it finds the endpoint, reads the model ID
from `/health`, and measures the server's default drafting policy with a
bundled offline prompt. Receipts go to `./vade-bench.json`, with a text table
next to it.

A zero-flag run uses a different recipe from a card, so its numbers don't line
up with the tables:

- It sends one bundled writing prompt. RigMark uses its own prose, code and
  structured prompts.
- It runs C1, C2, C4, C8 and C16 once each, with no warmup. A card runs C1, C2,
  C4 and C8 for 3 rounds each.
- Like the card's group rounds, it stops each reply at 256 tokens, but it also
  asks the server to keep going past the end-of-answer token, so every reply is
  exactly 256 tokens long.
- It has no single-stream cells with the 8192-token cap. It reads a cold 8K
  prompt only when you add `--prefill-body` and `--checkpoint-hash` (see
  [Run vade bench](#run-vade-bench)).

Both keep greedy decoding, the cache off and thinking off. On the server behind
the one-Spark v0.1.0 cards, a zero-flag `vade bench` with drafting as served
gave C1 45.466 and C8 185.144 tok/s. The drafting-ON card from that same
server shows C1 61.4 and C8 173.9. Both are recorded in the
[ON card's receipt folder](../bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/).

To compare runs with the same saved prompts and plain decoding (drafting off),
use `--draft plain`; it works on a drafting-enabled server without a restart.
Set `VADE_API_URL` to the running server's HTTP base URL and keep the same
prompt file between runs:

```sh
vade bench --url "$VADE_API_URL" --model qwen3.8-flash-next-fp4-local --draft plain \
  --prompts ./prompts.txt --concurrency 1,2,4,8 \
  --max-tokens 256 --repeats 3 --output ./vade-bench.json
```

This times cohorts with the Rust client: groups of concurrent requests that
each write exactly the capped number of tokens. It does not recreate the
RigMark prompt set, and it does not turn a timing run into a card. The cards'
ON recipes leave out the draft field to get the server default; the timing
runs send `"draft":"strict"`, which you ask for with `--draft strict`. Current
bench requests keep thinking off. Whenever you quote a result, keep its
thinking setting, estimator and source folder with it. See
[Run vade bench](#run-vade-bench) for every option.

## Run RigMark against Vade

[RigMark](https://github.com/alexellis/rigmark) is the public benchmark that
produces the cards on this page. Run it on the Spark that serves HTTP (rank 0
of a pair), against the loopback endpoint that `vade launch` or `vade up`
reports (`http://127.0.0.1:18080` by default).

**Turn thinking off in the request.** Vade answers chat requests with thinking
ON unless the request says otherwise. The FP4-Local server has no
server-level switch for that default, and neither `vade launch` nor `vade up`
sets one. RigMark sends no thinking field by default, and each decode sample
has an **8192-token cap shared by reasoning and the answer**. With RigMark's
defaults, most code and prose samples spend the whole cap reasoning: they stop
with `finish_reason` `length`, show no answer, and fail RigMark's output
checks. Set the field yourself with RigMark's `--extra-body`, which merges JSON
into every chat request. The server reads, in this order
([protocol.rs](../engine/crates/serve/src/protocol.rs)):

1. `chat_template_kwargs.enable_thinking`, or a top-level `enable_thinking`;
2. `reasoning_effort`: `none` or `minimal` turns thinking off; `low`,
   `medium`, `high` or `xhigh` turns it on at that effort;
3. the server default, ON.

This is the command we use on one Spark, from a RigMark checkout:

```sh
git clone https://github.com/alexellis/rigmark
cd rigmark
./rigmark configure
./rigmark bench --base-url http://127.0.0.1:18080 --model auto \
  --label vade-fp4local-1spark --comparison-id your-sweep-id \
  --metadata metadata.json --prefill-depths 8192 --concurrency 1,2,4,8 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

- `./rigmark configure` writes `metadata.json`. Set its `context_limit` to the
  context your server reports (`Context` in `vade status`, or
  `caps.context_limit` in `/health`); the cards used **32768**.
- `--prefill-depths 8192`: by default RigMark also measures 32768 and 65536,
  and it refuses any depth where depth + 8 exceeds `context_limit`. The cards
  measure 8K only.
- `--concurrency 1,2,4,8` adds C8 to RigMark's default C1/C2/C4. Every
  2026-10-24 RigMark card on this page ran with it, the two-VM cloud card 1006
  included (C8 **70.0 tok/s**).
- Everything else is a RigMark default: 5 runs each of prose, code and
  structured output with the 8192-token cap, 3 cold/replay prefill pairs, and
  3 rounds of code at each concurrency with a 256-token cap. RigMark fixes
  temperature 0, top-p 1 and seed 20260905.
- Older RigMark checkouts call the `bench` subcommand `run`.

The published cards send a longer body. To compare with a card, send the
identical body, because RigMark only compares receipts with the same request
body:

```json
{"cache": false, "chat_template_kwargs": {"enable_thinking": false}, "draft": "plain",
 "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "top_k": 0}
```

`"draft": "plain"` makes a drafting-OFF card; the drafting-ON cards leave the
field out and get the server default. `cache`, the neutral penalties and
`top_k` spell out the defaults. Thinking-ON cards replace
`"enable_thinking": false` with `"enable_thinking": true, "reasoning_effort": "low"`
inside `chat_template_kwargs`.

**What "DO NOT HEADLINE" means.** RigMark checks each of the 15 single-stream
samples: code and prose need visible answer text and a normal stop; structured
output needs the requested JSON values. If any sample fails, the card header
marks the result `DO NOT HEADLINE`, and you should not quote its rates as a
result. On 2026-10-24 a RigMark run with its default
settings against a two-VM pair scored **6/15**: code 0/5 and prose 1/5 hit the
cap while still reasoning, and structured passed 5/5. With thinking off
through `--extra-body`, the cards above pass **15/15**. These checks, which
RigMark calls gates, confirm that answers completed; they are not an exactness
or quality check.

**What a good card looks like** with thinking off:

- Header: `●  15/15 BASIC OUTPUT GATES PASSED`.
- SINGLE STREAM: PROSE, CODE and STRUCTURED*, each `✓ 5/5`.
- PREFILL: one 8K row with cold tok/s, cold TTFT and replay TTFT.
- CAPPED CONCURRENT GENERATION: `C1 | C2 | C4 | C8`, then
  `C8: 0/24 normal stops; 24/24 with visible output`. Every capped stream stops
  at 256 tokens, so zero normal stops is expected. With thinking on, those 256
  tokens are often all reasoning, and the visible-output count drops toward 0.
- SETTINGS: `SUITE: CUSTOM SETTINGS`, `Changed: --prefill-depths=8192`,
  `Changed: --concurrency=1,2,4,8`, and `REQUEST:` showing your body. Custom
  settings are expected and do not mean "DO NOT HEADLINE".
- Rates should land close to the [matching row](#current-results-2026-10-24)
  for the same Spark count and drafting mode (thinking OFF; for thinking ON see
  the [receipt folder](../bench/receipts/2026-10-24/README.md)).

## Run vade bench

Build the CLI once, then measure a running server with one command. After
`vade launch` or `vade up`, a bare `vade bench` uses the selected endpoint, the
model that `/health` reports and a bundled prompt. The explicit recipe below
picks retained prompts and plain drafting instead:

```sh
cargo build --release --offline --manifest-path cli/Cargo.toml
cli/target/release/vade bench --url "$VADE_API_URL" \
  --model qwen3.8-flash-next-fp4-local --draft plain --prompts /path/to/retained-prompts.txt --concurrency 1,8,16 \
  --max-tokens 256 --repeats 3 --output /path/to/private-receipt.json
cli/target/release/vade bench --self-test
```

The prompt file holds one literal UTF-8 prompt per nonempty line. Keep the same
prose, code and reasoning prompts between comparisons; the client cycles
through them and never generates replacements. Concurrency 1 always uses the
first prompt, so measure one-chat prose and code with separate retained files.
`--bodies-dir` sends saved, unchanging request bodies from the shared-reader
workload instead: 16 chats that share one long context, each asking for a
separate response. That directory holds the shared context, and the complete
requests sit in a `capture/` directory next to it.

For cold reading, add `--prefill-body tools/fixtures/probe-8192.json`,
`--prefill-body-sha256 11869d33b6fe2411447a7c9fd34ef76895ca53017f7505e2d63ac77783df945f`
and `--checkpoint-hash HASH`. It makes three uncached probes, requires exactly
8,192 server-reported prompt tokens and checks the build and checkpoint
identity. Those are three requests within one server start. For a performance
claim, repeat after at least three fresh server starts per build and alternate
A/B.

### What the receipt records

- The hash of each source file, and the hash of the exact compact UTF-8
  request sent to the server.
- The sampling settings every request sends explicitly: temperature 0, top-p
  1, top-k 0, neutral penalties, cache off and thinking off. Drafting follows
  the server default unless you pass `--draft plain` or `--draft strict`.
- The server's sampling echo for every response. A response with no echo is
  marked `sampling_source=request`, so the client never invents the settings
  that took effect.

The JSON receipt keeps the timing client's `schema_version`, `status`, `arms`,
`cells`, `requests`, `prefill_8k`, `receipts`, sampling, identities, memory,
observation and batch-log fields. The `.txt` file next to it gives readable
results and response records. Failed or partial cells keep their records but
publish no rate.

### How rates are computed

Cohort throughput is all completion tokens divided by the time from the first
request sent to the last one completed. `writing_tokens_per_s` uses the older
scoreboard formula, `(completion_tokens - 1) * 1000 / (elapsed_ms - ttft_ms)`:
every final completion token minus one. TTFT here counts the first nonempty
answer, reasoning or tool-call data, so it can come before the first answer
text. The receipt records `first_content_ms` on its own line. It leaves
`answer_content_tokens_per_s` as null, because the server's final usage report
does not say how many tokens came before the first answer text. Do not fill
that gap by subtracting reasoning estimates or by counting streamed (SSE)
chunks.

### Plain HTTP and drafting

`vade bench` speaks plain HTTP, including through an existing local SSH
tunnel. HTTPS is unsupported because the standard library has no TLS
implementation; use an existing local HTTP tunnel or a trusted TLS terminator.
Drafting-enabled servers are supported. Receipts keep each requested draft mode
apart from the server-reported MTP depth and build SHA, and keep response
observations separately. A reported depth of zero does not by itself prove
drafting was off; only receipts where every request asked for plain are
labeled drafting off. Neither the requested policy nor the server's reports
independently prove that drafting took effect.

### Options

Options carried over from the older timing client keep their names;
`--draft plain|strict` is new. The table lists each option, its default and
what it changes in the receipt.

The A/B options alternate a baseline server (A) with a candidate (B). ABBA
runs them A, B, B, A; ABA runs A, B, then A again (A′). Comparing the two A
runs shows how much the baseline drifts on its own.

| Option | Meaning or exception |
| --- | --- |
| `-h`, `--help` | Show timing options without connecting. |
| `--url URL` | Server base URL, with or without `/v1`; defaults to `VADE_API_URL` or the selected installation's endpoint; HTTP only. |
| `--model ID` | Model sent and checked; defaults to `/health` model, falling back to `qwen3.8-flash-next-fp4-local`. |
| `--draft plain\|strict` | Ask every request for plain decoding or strict drafting; by default the server's own policy applies. |
| `--prompts FILE` | One literal prompt per nonempty line; defaults to a bundled offline writing prompt. |
| `--bodies-dir DIR` | Saved shared-reader bodies, with `capture/` next to them; needs all 16 reader requests. |
| `--concurrency LIST` | Distinct positive comma-separated counts; default `1,2,4,8,16`. |
| `--max-tokens N` | Required completion count; default 256. A request that stops early makes its cell ineligible. |
| `--repeats N` | Repeats per cohort; default 1. |
| `--prefill-body FILE` | Three uncached, pinned 8,192-token probes. |
| `--prefill-body-sha256 HASH` | Verify frozen file bytes before requesting. |
| `--checkpoint-hash HASH` | Required checkpoint pin for 8k and alternating comparisons. |
| `--build-hash-field PATH` | Explicit dotted `/health` build field. |
| `--checkpoint-hash-field PATH` | Explicit dotted `/health` checkpoint field. |
| `--ab-urls A B` | ABBA order; averages the two candidate B runs and compares the two baseline A runs for noise. |
| `--aba-urls A B A_PRIME` | ABA order: A, B, then A′. |
| `--launch-commands COMMAND ...` | Two server start commands for ABBA, or three for ABA. The bench runs each one in its own process group and stops that group after its runs. |
| `--rank0-log FILE` | Read only what the server adds to the rank 0 log during each cell: batch row counts and memory byte counters. |
| `--aba-rank0-logs A B A_PRIME` | Separate ABA logs. |
| `--ab-rank0-logs A B` | Separate ABBA logs. |
| `--observe-cmd COMMAND` | Repeat per rank; record start/end GPU CSV and uptime with the same `VADE_*` context variables. |
| `--observe-timeout SECONDS` | Per-rank observer timeout; default 30. |
| `--output FILE` | JSON receipt and sibling text table; defaults to `./vade-bench.json`. An existing file is replaced. |
| `--out FILE` | Alias for `--output FILE`. |
| `--timeout SECONDS` | HTTP I/O and cohort barrier timeout; default 1800. |
| `--startup-timeout SECONDS` | Readiness timeout for servers started by `--launch-commands`; default 600. |
| `--api-key-file FILE` | Optional runtime Bearer credential file; `VADE_API_KEY_FILE` fallback remains. Credentials are excluded from receipts. |
| `--self-test` | Small local mock-stream check; no Spark or model qualification. |

### The older suite and the retired Python client

The existing `vade-bench` crate is folded into the CLI as a library, so there
is one command to run. `vade bench --suite` keeps its earlier decode,
concurrency, cache, priority, long-prompt, follow-up and shared-system
workloads; see [its guide](../engine/tools/bench/README.md) for those options.
That suite uses a different, historical receipt format and workload, so for
the current scoreboard choose the default timing protocol.

The former Python client and its operational wrapper (shim) are retired. Their
original files, and the evidence that the two clients matched on a real
server, are kept outside this repository. A live check on a pair needs at
least three runs of each client on identical inputs. A Mac mock check verifies
parsing and receipt arithmetic, and supports no Spark speed claim.

## Rented cloud GB10 VMs

Leave time for the model download before you benchmark rented VMs. The
verified FP4-Local snapshot is about **102 GB**. On 2026-10-24, rented GB10 VMs
downloaded from Hugging Face at **8–22 MB/s each**: about 80 minutes for one VM
at the top of that range, and over three hours at the bottom. For the cloud
cards we downloaded and verified the snapshot on one VM, copied it to the other
over the 200G link between them, and checked its SHA-256 again there. Without
such a link, copy the verified folder over the provider's network instead and
run the [INSTALL.md step 3](../INSTALL.md#step-3-get-the-model) manifest check
on the copy; the step 3 helper always fetches and verifies the whole snapshot.

In container-based VMs, the host page cache that holds the freshly downloaded
files can be hidden from the container's view of memory. GB10 memory is shared
by the CPU and GPU, so the engine's memory check can then refuse to start
(`Requested size does not fit`, with only about 1 GiB available), even though
the host could reclaim that cache. Main `2c64d0a`, part of v0.1.0, counts that
cache as free (Linux `MemAvailable`); the cloud cards above predate it and set
an explicit budget instead.

## Client time and server time

Client time is the wait seen by the program making the request. Server time is
the work done inside one chosen part of the engine. Always say which boundary
you measured.

| Metric | Definition | What it includes |
| --- | --- | --- |
| Client time to first token (TTFT) | First observable generated stream data minus request-send time | Connection, request transfer, queueing, prompt processing and transfer of the first generated data. The client accepts nonempty answer text, reasoning text or tool-call data; an empty stream header does not count. |
| Total client throughput | Sum of server `usage.completion_tokens` divided by the interval from the first request sent to the last `[DONE]` received | The whole cohort, including reading, queueing and response transfer. Adding individual chat rates does not produce this metric. |
| Older RigMark one-chat writing rate | `(completion_tokens - 1) / (last nonempty output event time - first nonempty output event time)` | Answer text or reasoning counts as output. It excludes the initial wait and the tail from the last output event through `[DONE]`. |
| Legacy one-chat scoreboard writing rate (default `vade bench`, and our earlier post-install check) | `(completion_tokens - 1) * 1000 / (elapsed_ms - ttft_ms)` | TTFT counts the first nonempty answer, reasoning or tool-call output. The denominator ends at stream completion after `[DONE]`, including terminal response-transfer overhead. It excludes the initial wait. |
| Answer-content writing rate | `answer_content_tokens_per_s`, currently null | First answer arrival is recorded as `first_content_ms`, but final completion usage does not identify the token count before that arrival. A content clock alone cannot establish an answer-only rate. |
| Server `prefill_ms` | Engine time spent processing the prompt | It excludes time before the engine admits the request and excludes client transfer. |
| Server `admission_to_first_logits_ms` | First model scores minus engine admission time | It includes work after admission; it can miss a wait before admission. |
| Server decode-step time | Time spent advancing the active decode group | It is a diagnostic, not a client rate. State whether ramp-up, partial groups, drain and host/RDMA waiting are included. |

`vade bench` implements the Rust client definitions; see
[the Rust benchmark library](../engine/tools/bench/README.md). Count output
tokens from final server usage. A streamed event can carry several tokens, so
counting events as tokens is invalid unless that boundary has been checked for
the tested server.

The former Python client is retired from the repository. Its source identity
and the post-install check's writing method were recorded in the retired receipt
`night-2026-10-11.json`, which is no longer published
([receipt status](../bench/receipts/README.md)). That receipt also holds two
lessons about timing:

- The pinned RigMark estimator takes its token count from final usage and its
  time interval from streamed events. The retained run reported one token per
  output event, but buffering can still limit how precisely the client times
  it. Its interval also ends at a different point from the post-install
  check's, which runs through `[DONE]`.
- The earlier RigMark run showed a wait, after a finished answer, before the
  engine admitted (accepted) the next request. A server admission timer could
  therefore look short while the next client waited much longer. Keep both
  clocks in follow-up and agent-turn receipts.

## Request settings and cache state

Send each sampling field explicitly. This example asks for plain drafting;
current `vade bench` leaves out the draft field unless you pass `--draft`:

```json
{
  "temperature": 0,
  "top_p": 1,
  "top_k": 0,
  "repetition_penalty": 1,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "cache": false,
  "draft": "plain",
  "enable_thinking": false,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

Set the output cap explicitly too. Record the seed you sent, or the effective
seed the server returns. From every response, save the effective sampling,
thinking, draft mode, MTP depth and actual draft-use fields. FP4-Local uses
greedy sampling when no sampling field is supplied. If any sampling field is
supplied, the unspecified ones come from the checkpoint's
`generation_config.json`. What a request asked for is not proof of the
settings that took effect. See the
[request parser](../engine/crates/serve/src/protocol.rs).

The older RigMark body, in the retired receipt `night-2026-10-11.json`, sent
temperature, top-p and seed. Its effective echo filled top-k and neutral
penalties from defaults; temperature zero made the run greedy. The receipt
disclosed this rather than presenting it as compliance with the later
all-fields-explicit protocol. That run also used a raised ceiling on streamed
output bytes.

Measure cache-off and cache-hit workloads separately. For cache-off timing,
check that usage and the prefix-reuse header both show zero reused tokens. For
a hit, keep the prime request, the replay request, the restored-token count,
the cache budget and the eviction state. Similar text is not proof of a cache
hit. Cache-accounting failure and answer-byte equality are separate results.
The same retired receipt recorded both for a QSA candidate build. QSA is the
model's indexed attention, which reads only the token blocks that a small
indexer selects.

“Cold” in the reading rows means an uncached prompt with weights already
loaded. It does not include model download, server build or weight loading.
Record whether it was the first request after startup or a later uncached
request. A fresh server start and another probe on the same server are
different kinds of repetition.

## Repetition, noise and ties

To compare a change, use the same Sparks and fresh servers that take turns
between the baseline and the candidate. The current rule asks for at least
three fresh starts per build. Keep the workload, client, context, output length
and sampling fixed. Report medians and full ranges, and name any warmups or
discarded runs. Overlapping ranges are a **tie**. A shorter median alone does
not show a speed gain.

Record both the requested and the actual output lengths. A natural early stop
is useful evidence about that answer, but it cannot count toward a
fixed-output timing cell. Report failures and exclusions alongside the
eligible runs. Do not rebuild a missing client rate from a GPU step timer.

The retired receipt `night-2026-10-11.json` holds two different kinds of
evidence. Its QSA comparison has eligible fresh-server results for the
retained shared-reader workload, but its other prompt cells ended early, so
its full timing matrix is incomplete. Its clean-install comparison measured
against historical results from another pair, with a stated historical noise
rule, and its raw cold-prompt ranges did not overlap.

## Build, hardware and memory identity

A useful receipt records:

- the checkpoint revision and hash scope;
- the tokenizer and chat-template identity;
- the source commit and its dirty state (whether there were uncommitted
  changes);
- the server, PTX (GPU kernel code) and client hashes;
- toolchains and driver;
- rank count and topology;
- context and cache budgets;
- prompt identity, output count, concurrency and arrival pattern.

Record clocks, temperature and competing traffic when you have them. Keep the
exact memory bytes for each rank with each rate. Mark missing fields as
unmeasured, and never fill them in from an assumption.

Profiled runs explain where the time goes. Publish unprofiled timing
separately: profiling overhead and overlapping spans do not make a performance
receipt. Ranks run at the same time, so adding rank times does not give wall
time.

## What these results do not show

- These rows are not a matched speed comparison with another engine.
- Writing speed and capped concurrent generation say nothing about useful
  coding tasks completed, harness response time after tools, or agent
  reliability.
- Several things are checked separately and are not all shown by these
  receipts: native repeatability (the same token IDs on every run), agreement
  with a previous build, model quality, and whether a batch keeps running when
  one of its rows is rejected. The default builds named here still skip that
  last check, which deliberately feeds a bad row into a batch.
- Neither a tested context setting nor a longer configured window proves
  long-memory quality at every length. The October 12 quality evidence in the
  [historical section](#historical-fp8-measurements) is still partial. The
  failed context candidate was a historical guard (retired receipt
  `context-candidate.json`).
- A server with MTP enabled can fall back to plain decoding. In the extra run
  that followed the post-install check, no drafting actually happened, so that run
  supports no claim of a speculative-decoding speedup (retired receipt
  `night-2026-10-11.json`).
- A Mac build check says nothing about GPU arithmetic or speed.

Repeatability requires identical token IDs for the same request and settings
across runs and Sparks, including plain and strict drafting.
[Architecture](architecture.md#exactness-and-model-quality) explains the full
byte-for-byte contract and the quality gates that arithmetic changes must pass.

## Where the receipts are, and older comparisons

Start with the tested setup and timing instructions in
[INSTALL.md](../INSTALL.md). The
[timing tools guide](../tools/README.md#timing-the-installed-server) describes
the client. Measure only on Sparks with no other GPU work running.

The older public FP8 summaries have been removed; the
[historical section](#historical-fp8-measurements) below still cites some of
them as retired receipts. The current tables above quote card folders and
timing READMEs. Their sanitized copies are in
[bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md): cards,
READMEs and small JSON receipts, with hostnames, addresses and private paths
replaced by neutral tokens. Original prompts, replies, logits and the full card
JSON receipts are not included. See the
[receipt status](../bench/receipts/README.md).

[Historical engine comparisons](../engine/docs/benchmarks.md) cover older
models and workloads. Read their conditions before you compare rates. Use the
explicit methods and limits that come with each receipt; a historical result
says nothing about current-build performance.

## Historical FP8 measurements

These results describe older builds and settings, and they are not FP4-Local
performance claims. Everything in this section comes from retired receipts
that are no longer published ([receipt status](../bench/receipts/README.md)):
the October 12 results from `defaults-2026-10-12.json`, and the earlier
measurements and memory figures from `night-2026-10-11.json` and
`exact-path-gate.json`.

### October 12 overview: Flash-Next on two Sparks

With the October 12 product defaults, Vade started answering an uncached 8k
prompt in about **7.8 seconds** and wrote a single answer at about
**23.2–23.4 tokens per second**. The final RigMark run on build `9bcb8170`
passed every line of its report with thinking off and with thinking on/low, on
two DGX Sparks using the official Qwen3.8-Flash-Next FP8 checkpoint. Passing a
report line means the requests finished and passed RigMark's checks; it does
not mean every workload got faster.

### October 12 product defaults

We measured at context **131,072**, output cap **8,192**, MTP **off**, with
the automatic session pool and the retained cache. Each rank (each Spark) held
one share of the official-precision weights, and the two ranks were linked in
an RDMA ring (remote direct memory access, which moves data straight between
the Sparks' memory). At native startup the engine set aside room for one
full-context session. A near-full cached continuation restored
**130,560 tokens**, read 256 new tokens and generated 256 more, filling the
131,072-token window; its answer bytes and token IDs matched an uncached
continuation. We ran these engine checks with a separate measurement launcher,
so they say nothing about starting and stopping the engine with the current
CLI.

#### Final RigMark

All **108 requests**, **184 gate cells** (single pass/fail checks) and
**24 reporting rows** passed. Each thinking mode got one fresh server: five
answers each for prose, code and structured output; three cold/replay pairs at
each of 8k, 32k and 64k; and three cohorts each at one, two and four chats.
Every sample is included. The table shows thinking-off medians and full ranges.
Its memory column gives whole-suite allocation peaks for ranks 0/1, not the
extra memory for that row; each rank held **91,709,826,224 weight bytes**.

| Workload | Client result, median [min–max] | Peak bytes, rank 0 / rank 1 |
| --- | --- | --- |
| Cold 8,192-token prompt | TTFT 7.794 s [7.793–7.802] | 113,946,349,572 / 113,946,349,572 |
| Cold 32,768-token prompt | TTFT 31.878 s [31.843–31.891] | 113,946,349,572 / 113,946,349,572 |
| Cold 65,536-token prompt | TTFT 66.705 s [66.666–66.822] | 113,946,349,572 / 113,946,349,572 |
| Write prose, one chat | 23.305 tok/s [23.273–23.309] | 113,946,349,572 / 113,946,349,572 |
| Write code, one chat | 23.186 tok/s [23.142–23.262] | 113,946,349,572 / 113,946,349,572 |
| Write structured output, one chat | 23.370 tok/s [23.325–23.402] | 113,946,349,572 / 113,946,349,572 |
| Two concurrent chats, 256 output tokens each | 35.097 total tok/s [34.964–35.283] | 113,946,349,572 / 113,946,349,572 |
| Four concurrent chats, 256 output tokens each | 54.227 total tok/s [53.929–55.061] | 113,946,349,572 / 113,946,349,572 |

Writing time leaves out the wait for the first token and ends at the last
nonempty output event; concurrency time includes reading, queueing and
transfer through completion. The October 12 receipt recorded every row,
including warm cache replays and thinking on/low, whose whole-suite peak was
113,978,286,084 bytes on each rank. The automatic pool was 24,264,576,444 bytes
per rank for thinking off and 24,071,770,351 for thinking on/low. Retention was
18,125,450,072 bytes per rank in both launches. Capacity varies with available
memory; 23 minimum-size session slots meant **one** full-context slot.

RigMark stopped at 64k prefill and four concurrent chats. Its 131,072 context
setting is not a 128k prefill timing row or a full-length multi-chat test.
Historical comparisons with the scalar night build and the MTP-2 preview
changed both builds and settings, and did not use fresh alternating A/B starts
(fresh servers taking turns between baseline A and candidate B). The roughly
6% lower plain writing rate versus that night build reflects both arithmetic
changes and default changes; it is separate from the cost isolated below.

RigMark's chat requests sent every greedy sampling field explicitly. Its
pinned upstream **prefill** (prompt-reading) requests sent temperature zero but
left out top-p, top-k, penalties and seed, to keep the upstream method. We
checked every response: effective prefill top-p was 0.95, top-k 20 and
penalties 1/0/0, with per-request seeds. The receipt keeps this exception on
record. New timing requests must set every sampling field explicitly, as
[Request settings and cache state](#request-settings-and-cache-state)
describes.

#### Accepted pool and context cost

A separate comparison used three fresh alternating starts per build, with cache
and thinking off, MTP off, explicit greedy sampling and 256 output tokens per
request. Baseline A was `41fa8cf8`; candidate B was `9bcb8170`. One-chat A used
32,768 context and B used the 131,072 default. The eight- and sixteen-chat
cells used **32,768 on both builds**. Rates below are medians [min–max]; peaks
are the largest lifetime device allocation through that cell on either build,
equal on both ranks. Each rank held 91,709,826,224 weight bytes.

| Workload | A tok/s [range] | B tok/s [range] | Peak bytes, rank 0 / rank 1 | Finding |
| --- | --- | --- | --- | --- |
| One-chat prose writing | 23.837 [23.802–23.838] | 23.513 [23.333–23.721] | 93,218,536,452 / 93,218,536,452 | 1.36% slower |
| One-chat code writing | 23.785 [23.710–23.861] | 23.494 [23.438–23.650] | 93,278,108,676 / 93,278,108,676 | 1.22% slower |
| One-chat reasoning-prompt writing, thinking off | 23.764 [23.605–23.858] | 23.691 [23.445–23.787] | 93,278,108,676 / 93,278,108,676 | Tie |
| Eight-chat total | 90.205 [90.020–90.227] | 90.402 [89.601–90.505] | 94,424,406,720 / 94,424,406,720 | Tie |
| Sixteen-chat total | 117.134 [116.938–117.828] | 117.251 [117.160–117.492] | 95,733,390,512 / 95,733,390,512 | Tie |

All tested answer bytes and token IDs matched across builds and starts. The
prose and code ranges did not overlap, so the original strict not-slower
verdict was **FAIL**. On October 12 the maintainers accepted this roughly
**1.3% one-chat cost** for the change to the 128k pool and default; that
decision does not turn the failed not-slower check into a pass. The
concurrency ranges overlap, so those are ties. These runs used the native
client, so their workloads differ from RigMark and from the older eight-chat
scoreboard.

In a separate check of startup with default settings, the engine negotiated a
24,025,760,392-byte pool and 18,125,450,072 bytes of retention per rank. That
covers an 18,116,712,944-byte full-context reservation plus 8,737,128 bytes of
publication metadata. The near-full continuation peaked at 105,751,095,300
device bytes per rank. Memory readings at each allocation milestone and once a
second are not enough to establish continuous driver-memory pressure.
Override and refusal cases were not part of that check.

Earlier October 12 long-memory checks at `41fa8cf8` passed all four 128k
teacher-document noise/flatness bounds. Eighteen of 24 paired synthetic
retrieval cases passed with no lost baseline successes; **six remain
unverified**. This is partial quality evidence from that arithmetic build, not
a fresh quality suite at `9bcb8170`.

### Earlier measurements

An arm is one side of an A/B comparison: A is the baseline and B the
candidate, and A′ and B′ are further arms of the same two builds.

| Workload and measured build | Result | Evidence and limits | Receipt |
| --- | --- | --- | --- |
| Cold 8,192-token prompt, resident weights, cache off; source engine `7a26f6ae` | 13.481 s client time to first token | One successful unprofiled sanity sample; no median or speed-gain claim. | `night-2026-10-11.json` |
| Same prompt length, cache off; earlier candidate `a727601f` | 13.693 / 13.680 s medians in two fresh arms | Three probes per arm; historical gate and sanity timing, with separate missing checks. | `exact-path-gate.json` |
| Eight distinct short-prompt chats, 256 output tokens each; source QSA candidate `1bc9b667` | 62.709 total client tok/s; range 61.988–62.911 | Three fresh servers; every timed request completed the fixed output count. Strict HTTP cache accounting failed; native output comparisons passed. | `night-2026-10-11.json` |
| One chat, prose; source engine `4c13c018`, RigMark, thinking off | 25.011 decode tok/s; range 24.919–25.110 | Five natural-stop answers passed basic output checks. RigMark prefill was skipped; the suite was incomplete. | `night-2026-10-11.json` |
| One chat, code; same RigMark build and settings | 24.838 decode tok/s; range 24.504–24.865 | Five natural-stop answers passed basic output checks. This measures writing speed, not successful coding tasks. | `night-2026-10-11.json` |
| Clean-install check, monorepo `38c7b47`, retained cold 8k and shared-reader eight-chat requests | 13.814 s TTFT, range 13.745–13.913; 60.506 total tok/s, range 60.084–61.139 | Three fresh launches, three repeats each. Different pair from the older shared-reader/8k reference; the old executable was not rerun alongside it. | `night-2026-10-11.json` |

Commit IDs from the source engine repository are separate from commit IDs in
this repository (the monorepo). A result holds only for the build and workload
named in its receipt; it does not carry over to a later merge or a
documentation change.

The legacy FP8 Flash-Next output default is 65,536 tokens, with context
131,072. The October 12 measurements above used an 8,192-token output cap, so
they do not cover the new output default. The earlier timing and installation
results used 32,768 context tokens. Keep `--context 32768 --output 8192`
explicit when you reproduce them, and report the actual pool and cache bytes;
the October 12 startup, continuation and timing results above are separate
evidence. The historical manual-install check does not cover the current
Rust lifecycle.

### Memory per rank in the FP8 receipts

The earlier headline setup used the official FP8 checkpoint, tensor-parallel
execution (each layer split across both Sparks) over a two-Spark RDMA ring, and
a 32,768-token context. Its reading and shared-reader eight-chat timing used
cache off, thinking off and plain decoding. The older RigMark rows reported no
reused prompt tokens; their cache-isolation limitation was disclosed
separately. The retired receipt `night-2026-10-11.json` summarized selected
response checks and the artifact identities each rank reported.

The retired receipts reported these bytes for each rank:

| Workload | Weight bytes per rank | Current device bytes per rank | Reported allocation peak per rank | Receipt |
| --- | ---: | ---: | ---: | --- |
| Cold-prompt sanity check, unprofiled | 91,709,826,224 | Not retained in this summary | 93,564,704,076 | `night-2026-10-11.json` |
| Earlier candidate, reading and C8 arms | Not retained in this summary | 92,928,634,296 at C8 health checks | 94,427,138,068 in A/B reading+C8 arms; 93,564,704,076 in B′/A′ reading-only arms | `exact-path-gate.json` |
| QSA eight-chat candidate, both ranks | 91,709,826,224 | 92,928,635,704 | 94,373,705,044–94,426,210,720 | `night-2026-10-11.json` |
| Older RigMark prose/code | Unmeasured in this receipt | Unmeasured in this receipt | Unmeasured in this receipt | `night-2026-10-11.json` |
| Clean-install shared-reader/8k, both ranks | 91,709,826,224 | 93,151,759,800 | 94,427,138,068 | `night-2026-10-11.json` |

Peaks cover explicit allocations seen over a server's lifetime and can include
earlier cells. These figures do not show total physical-memory use or
driver/graph allocation size. Missing values stay unknown; we never borrow
memory figures from another build.
