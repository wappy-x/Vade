# Benchmarks

<!-- bench-docs:start -->

### Current main

These rows measured main `dc32b7c` on our DGX Sparks on the morning of 2026-10-31. Each server start ran one card with drafting on and one card with drafting off. A card is the result file of one RigMark run. The one-Spark rows are the mean of two server starts. The other rows are one start each.

[Results and receipts](results.md) links the receipt of every row. It also gives the machines, the server commands and the request settings.

**One chat (tok/s)**

<div align="center">

| Setup and drafting | Prose | Code | Structured |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark · on | 42.530 | 81.138 | 100.808 |
| FP4-Local · 1 Spark · off | 27.911 | 27.761 | 28.012 |
| FP4-Local · 2 Sparks · on | 54.878 | 102.314 | 130.800 |
| FP4-Local · 2 Sparks · off | 36.047 | 35.840 | 36.142 |
| FP4-Local · 4 Sparks · on | 63.036 | 113.025 | 145.238 |
| FP4-Local · 4 Sparks · off | 42.084 | 41.486 | 42.072 |
| FP8 · 2 Sparks · on | 36.624 | 51.293 | 52.763 |
| FP8 · 2 Sparks · off | 27.072 | 26.913 | 27.114 |
| FP8 · 4 Sparks · on | 49.508 | 69.340 | 72.248 |
| FP8 · 4 Sparks · off | 35.902 | 35.641 | 35.955 |

</div>

**Chats at once (total tok/s)**

<div align="center">

| Setup and drafting | C1 | C4 | C8 |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark · on | 65.688 | 144.681 | 178.981 |
| FP4-Local · 1 Spark · off | 27.115 | 85.004 | 138.609 |
| FP4-Local · 2 Sparks · on | 79.632 | 166.796 | 232.198 |
| FP4-Local · 2 Sparks · off | 34.375 | 88.178 | 134.414 |
| FP4-Local · 4 Sparks · on | 86.962 | 177.581 | 245.145 |
| FP4-Local · 4 Sparks · off | 38.080 | 95.113 | 143.015 |
| FP8 · 2 Sparks · on | 45.436 | 66.578 | 98.670 |
| FP8 · 2 Sparks · off | 26.089 | 66.273 | 98.059 |
| FP8 · 4 Sparks · on | 61.435 | 88.017 | 130.999 |
| FP8 · 4 Sparks · off | 34.592 | 87.157 | 131.147 |

</div>

**Path and cold 8K prompt**

<div align="center">

| Setup | Path | Cold 8K (s), drafting-on card | Cold 8K (s), drafting-off card |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | local | 3.769 | 3.778 |
| FP4-Local · 2 Sparks | tensor split | 2.153 | 2.131 |
| FP4-Local · 4 Sparks | tensor split in parts | 1.924 | 1.916 |
| FP8 · 2 Sparks | tensor parallel | 5.971 | 5.952 |
| FP8 · 4 Sparks | tensor parallel | 6.923 | 6.873 |

</div>

Rates are in completion tokens per second. Prose, code and structured output each use one chat. C1, C4 and C8 give the total rates for one, four and eight chats at once. Cold 8K is the time to the first token after an uncached 8192-token prompt. Both cards send it with drafting off. The [2026-10-31 section](#current-2026-10-31-morning-main-dc32b7c-home-dgx-sparks) gives the 32K chat and the token ID checks.

These tables are written by hand, because the one-Spark rows average two server starts. The `bench-docs` tool cannot make these tables. It takes one card per setup, only from the newest main that has cards, and it does not average server starts ([bench_docs.rs](../tools/src/bench_docs.rs)). From the repository root, this command replaces this block and the README block with such a table, from a folder of RigMark card files:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- bench-docs '<cards-dir>' .
```

The full card files are private. Their sanitized extracts are in [bench/receipts/2026-10-31](../bench/receipts/2026-10-31/README.md).

Each section below names its date and main. Its results apply only to that build.

<!-- bench-docs:end -->

## How the current numbers were measured

The recipe of the [current table](#current-main) (main `dc32b7c`) is on [results and receipts](results.md), with a sanitized receipt for each card.

This section gives the recipe behind the [2026-10-30 section](#previous-2026-10-30-morning-main-380d8e5-home-dgx-sparks) and the 2026-10-28 and 2026-10-29 rows. Every number there comes from a card file or from a lab report line. Those records are not in the repository.

A revision such as main `380d8e5` and a pull request such as #142 belong to the private development history, so they are not in this repository. The v0.1.0 commits `5a22651` and `99e8f62` are the exception ([glossary](GLOSSARY.md#ho), "Main `abc1234`").

### Models and builds

These are the two checkpoints. The identity is the SHA256 that the server reports.

- **fp4-local.** The checkpoint is `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`. Its identity is `d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`.
- **fp8.** The checkpoint is `Qwen/Qwen3.8-Flash-Next-FP8` at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`. Its identity is `8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03`.

These are the builds:

<div align="center">

| Main | Date | Server build | Used for |
| :---: | :---: | :---: | :---: |
| `dc32b7c` | 2026-10-31 | flash-serve `55bdc7a8`, PTX `84c09350` | The current table ([receipts](../bench/receipts/2026-10-31/README.md)) |
| `380d8e5` | 2026-10-30 | flash-serve `d56d255c`, PTX `84c09350` | Drafting-on rows, the 32K chat, the context run |
| `51dce02` (engine `380d8e5`) | 2026-10-30 | flash-serve `44e95e44` | Two pairs at once |
| `cbbb11b` | 2026-10-29 | flash-serve `cad21fbe`, PTX `4288b2a8` | Drafting off: fp4-local on one Spark, fp8 on two Sparks |
| `76977df` | 2026-10-28 | flash-serve `4dedff9d`, PTX `4288b2a8` | Drafting off: fp4-local on two and four Sparks, fp8 on four Sparks |

</div>

### Requests

Every request used greedy decoding: temperature 0, top-p 1, top-k 0, repetition penalty 1, and presence and frequency penalties 0. Chat requests also set seed 20260905, thinking off and `cache` false.

Drafting-on requests leave out the `draft` field, so the server uses its default: depth 3 on FP4-Local and depth 2 on FP8. Drafting-off requests send `"draft":"plain"`.

A RigMark card sends 66 requests. It uses the prompt set of RigMark commit `52c29c3fc1d741b9aef8895b23372aafa5dc6161` (prompts version 1.0.0). The prompt and reply lengths below are from the two-Spark FP4-Local card of 2026-10-30.

- **Prose, code, structured.** Each is one chat, with prompts of 117 to 145 tokens. The reply is up to 8192 tokens, to the natural stop (551 to 1,794 tokens). Each runs 5 times. The table gives the median of the 5 writing rates.
- **C1, C2, C4, C8.** These are 1, 2, 4 or 8 chats at once, with code prompts of 140 to 147 tokens. Each chat gets exactly 256 tokens. Each runs 3 rounds. The table gives the median of the 3 round rates.
- **Cold 8K.** The prompt has 8,192 tokens and goes out with drafting off and a new cache salt for each pair. The reply is 8 tokens. The card sends 3 cold and replay pairs. The table gives the median time to the first token of the 3 cold requests.

When a setup has two or three server starts, each cell is the mean of the start medians.

The `cbbb11b` rows used our own card client instead. It sends 64 requests: no C2 round and five cold 8K prompts. In those rows, C4 and C8 are the mean of the three round medians on FP4-Local, and the median of the three rounds on FP8.

### Each table

- **The 2026-10-30 speed table, drafting on.** One Spark ran `vade up --sparks 1 --context 32768 --output 8192`. Two Sparks ran `vade up --sparks 2 --context 32768 --output 8192`. FP4-Local on four Sparks, and FP8 on two and four Sparks, ran the lab launcher with context 32768 and output 8192 (FP4-Local) or 16384 (FP8). There were 3 starts on one and two Sparks and 2 on the others. The record is lab report `own-rigmark-eod-r2`, cards `2026-10-30_0702` to `0804`.
- **Drafting off at `76977df` (2026-10-28).** Two and four Sparks ran the lab launcher with the same limits, in 1 start. The record is the 2026-10-28 cards with drafting off.
- **Drafting off at `cbbb11b` (2026-10-29).** One Spark ran `vade up --sparks 1` with context 32768 and output 8192. FP8 on two Sparks ran the lab launcher with context 32768 and output 16384. There were 2 starts. The records are lab reports `own-card-1s-eod-2-r2` and `own-premerge-fp8b-r2`.
- **32K chat, drafting on.** It ran on the card start. On one Spark, it ran on two other starts at context 65536. There were 2 or 3 starts. The record is `own-rigmark-eod-r2`.
- **Context on one Spark, drafting on.** It ran `vade up --sparks 1 --context 262144 --output 8192`, in 1 start. The record is `own-rigmark-eod-r2`.
- **Two pairs at once, drafting on.** Each pair ran `vade up --sparks 2`, in 2 starts. The record is lab report `own-two-pairs-r1`.

The 32K chat sends a 32,099-token question. Then it sends a second question on the same chat: 32,305 tokens with a 54-token reply. With FP8 on two Sparks, the second question has 32,238 tokens and a 53-token reply. The first reply was 179 tokens long, and 112 with FP8 on two Sparks. The "Second question, cache" column is the first of two cached runs.

The context run sent prompts of 64,217 and 129,220 tokens with replies of 200 and 179 tokens. It also sent a 64,444-token second question with a 56-token reply. Its writing speed is the same decode rate that the cards use.

### What you can run today

- **Start one or two Sparks as in the cards: available.** Run `vade up --sparks 1 --context 32768 --output 8192 --mtp 3` or `vade up --sparks 2 --context 32768 --output 8192 --mtp 3`.
- **Start four Sparks or FP8: available, but the cards used a lab launcher, not these commands.** Run `vade up --sparks 4 --context 32768 --output 8192 --mtp 3` or `vade up --fp8 --sparks 2 --context 32768 --output 16384 --mtp 2` (or `--sparks 4`).
- **Check token IDs, drafting on against off: available.** Run `vade verify -n 3 --prompt 8k --max-tokens 256`. It checks one server, not one card against another.
- **Run the 66-request card: available.** From a RigMark checkout at `52c29c3`, run `vade bench --card --prompts rigmark/prompts.json --receipt bench/receipts/2026-10-31/<file>.json`. It sends the card's requests and prints `ids 66/66 identical` when the token IDs match the receipt. On 2026-10-31 it matched the one-Spark drafting-on and drafting-off receipts. We have not run it against the other receipts ([details](results.md#what-a-stranger-can-and-cannot-re-run)).
- **Time your own server: available.** Run `vade bench --concurrency 1,2,4,8 --max-tokens 256 --repeats 3`. It uses other prompts, so its numbers do not compare with the cards.
- **Run the 32K chat or the context run: not available.** No `vade` command times a second question with and without the cache, and the prompts are not in the repository.

## All results, newest first

This page lists every speed we have published for Vade, newest first. It also shows how to measure your own.

<div align="center">

| Section | What it is for |
| :---: | :---: |
| [2026-10-31 morning cards](#current-2026-10-31-morning-main-dc32b7c-home-dgx-sparks) | One, two and four Sparks at main `dc32b7c`, FP4-Local and FP8, drafting on and off, with public receipts |
| [2026-10-30 morning cards](#previous-2026-10-30-morning-main-380d8e5-home-dgx-sparks) | One, two and four Sparks at main `380d8e5`, drafting on, with second starts |
| [2026-10-29 cards](#current-2026-10-29-main-cbbb11b-home-dgx-sparks) | One and two Sparks at main `cbbb11b`, with 64K and 128K prompts on one Spark |
| [Confirm run on main `285fa8a`](#confirm-run-main-285fa8a) | Two Sparks, the new main against the main before it |
| [2026-10-28 end-of-day cards](#current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks) | One, two and four Sparks at main `76977df`, FP4-Local and FP8 |
| [2026-10-28 cards after #142](#previous-2026-10-28-after-142-main-9d2ec6c-home-dgx-sparks) | Two Sparks at main `9d2ec6c`, with a 32K and a 100K chat |
| [2026-10-28 cards after #139 and #140](#previous-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks) | Two Sparks at main `8b7b866` and `104205a` |
| [2026-10-28 morning cards](#previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks) | Two Sparks at main `bb24e80`, with 32K, 64K and 128K prompts |
| [Final 2026-10-27 cards](#previous-2026-10-27-final-main-09e1fc4-home-dgx-sparks) | Two Sparks at main `09e1fc4`, with 32K and 64K prompts |
| [2026-10-27 morning cards](#previous-2026-10-27-morning-main-12142bb-home-dgx-sparks) | Two Sparks on the tensor split as the default |
| [2026-10-26 cards](#previous-2026-10-26-main-262b0c3-home-dgx-sparks) | Two Sparks on the tensor split, next to the layer pipeline |
| [2026-10-25 cards](#previous-2026-10-25-main-80ac2d4-home-dgx-sparks) | Two and four Sparks on the layer pipeline, FP4-Local and FP8 |
| [2026-10-24 cards](#previous-2026-10-24-v010-release-5a22651) | The v0.1.0 release on one and two Sparks and on cloud GB10 VMs |
| [Earlier cards from the same day](#earlier-cards-from-the-same-day-main-4681d8b-and-73f7a9d) | The cards of 2026-10-24 at main `4681d8b` and `73f7a9d` |
| [How to reproduce a run](#reproduce-a-measurement-in-detail) | RigMark or `vade bench` |

</div>

Every result names the build and settings that we measured, and its timings hold for that build only. The older FP8 measurements are kept at the end, as history.

A **receipt** is the saved record of one run: its build, its settings and what it measured. The receipts behind the 2026-10-24 tables are in [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md), as sanitized copies. Private details such as hostnames are replaced, so our machines appear as home-1 to home-6 and the rented ones as vm-a to vm-d. The [receipt status](../bench/receipts/README.md) page says what is public.

The 2026-10-25 to 2026-10-28 cards are not in the repository yet. Each row names its result folder. The 2026-10-27 and 2026-10-28 folder names end in a pair suffix that uses internal machine names (and `-final`, `-am`, `-139`, `-139-c140`, `-142`, `-eod-1s`, `-eod-2s`, `-eod-4s`, `-eod-fp8-2s` or `-eod-fp8-4s`). So this page gives those names without the suffix. The time, commit and drafting setting in each name still identify the card.

The first-token, prefix-cache and repeated cold-prompt timings of 2026-10-28 after #139, after #142 and at the end of the day come from separate runs, not from the cards. Each of those tables or sentences names its run folder, such as `c139-l32`, `r142-wtT-x1` or `eod-wtT-x1`. These folders are not in the repository yet either.

<a id="current-2026-10-31-morning-main-dc32b7c-home-dgx-sparks"></a>
## Current (2026-10-31 morning, main dc32b7c, our DGX Sparks)

On the morning of 2026-10-31 we measured main `dc32b7c` on one, two and four Sparks, with FP4-Local and with FP8. Every server ran the same `flash-serve` binary (SHA256 `55bdc7a8...`, PTX `84c09350`), with greedy decoding, thinking off and no `VADE_` setting.

Each server start ran one card with drafting on and one with drafting off. The 32K chat ran on the two-Spark and four-Spark starts. On one Spark, the card ran on two starts, and the 32K chat on two more starts at context 65536.

[Results and receipts](results.md) gives the table, the receipt of each card, the server commands and what you can run again. The speeds are in the [current table](#current-main).

**What changed since the 2026-10-30 cards.**

- **FP4-Local on four Sparks.** Four Sparks linked by RDMA now run the tensor split in parts, not the layer chain. Prose went from 43.397 tok/s on the evening of 2026-10-30 (main `7943f57`, layer chain) to 63.036 tok/s.
- **FP8 on four Sparks.** Prose went from 46.759 to 49.508 tok/s after a change that sends the FP8 commands between ranks over RDMA.
- **One and two Sparks.** The one-Spark and two-Spark rows changed by at most 2.2 % in any cell.

The two before values are private cards of 2026-10-30 that this page does not list elsewhere.

**The 32K chat.** These are the times to the first token, in seconds. The one-Spark row is the mean of two starts.

<div align="center">

| Setup | First question | Second question, cache | Second question, no cache |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 15.63 | 0.36 | 15.79 |
| FP4-Local · 2 Sparks | 10.09 | 0.76 | 9.95 |
| FP4-Local · 4 Sparks | 8.98 | 8.72 | 8.74 |
| FP8 · 2 Sparks | 24.12 | 0.94 | 23.91 |
| FP8 · 4 Sparks | 27.81 | 1.06 | 28.47 |

</div>

The cached run reused 32,064 tokens on every setup except four FP4-Local Sparks, which reused 0. Four FP4-Local Sparks keep no prefix cache, so the second question reads the whole prompt again.

**Token IDs.**

- **FP4-Local.** All eight FP4-Local cards (one, two and four Sparks, drafting on and off) gave the same token IDs in 66 of 66 responses.
- **FP8 at one Spark count.** FP8 drafting on and off gave the same token IDs in 66 of 66 responses on two Sparks, and on four.
- **FP8, two against four Sparks.** The cards matched in 7 of 66 ([FP8 on two and four Sparks](#fp8-on-two-and-four-sparks-different-token-ids)). A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order, so four Sparks now give the same token IDs as two ([changelog](../CHANGELOG.md)). These cards predate that change.

**Limits.**

- **Starts.** There was one server start per setup, except on one Spark.
- **Launcher.** The four-Spark and FP8 starts used a private lab launcher.
- **Clients.** The 32K chat client is private. The card client is public: `vade bench --card` ([details](results.md#what-a-stranger-can-and-cannot-re-run)).

<a id="previous-2026-10-30-morning-main-380d8e5-home-dgx-sparks"></a>
## Previous (2026-10-30 morning, main 380d8e5, our DGX Sparks)

On the morning of 2026-10-30 we measured main `380d8e5` (#201) on one, two
and four Sparks, with FP4-Local and with FP8. Every server ran the prebuilt
payload of `380d8e5` (flash-serve `d56d255c`, PTX `84c09350`), with greedy
decoding and thinking OFF. FP4-Local used MTP 3 and output 8192. FP8 used
MTP 2 and output 16384. No `VADE_` environment setting was set. Each card
start had context 32768 and ran one RigMark card with drafting ON (66 responses). On
two and four Sparks, the same start then ran a 32K-token chat. On one
Spark, the chat ran on two more starts at context 65536, and one more start
at context 262144 served long prompts. The Starts column counts the card
starts, and the last row the context start. We did not run drafting OFF
cards.

| Setup | Path | Server | Starts |
| --- | --- | --- | ---: |
| FP4-Local, one Spark | local | `vade install` and `vade up --sparks 1` from a bundle of the payload | 3 (on two Sparks) |
| FP4-Local, two Sparks | tensor split | `vade up --sparks 2` from the same bundle | 3 (two pairs of Sparks) |
| FP4-Local, four Sparks | layer chain | lab launcher, four Sparks in a chain of direct links | 2 |
| FP8, two Sparks | ring | lab launcher | 2 |
| FP8, four Sparks | path (tensor parallel on four Sparks) | lab launcher, the same four Sparks | 2 |
| FP4-Local, one Spark, context 262144 | local | `vade up --sparks 1 --context 262144` from the same bundle | 1 |

### Speed

Rates in tok/s, cold 8K in seconds. Each card gives the median of its runs.
C2, C4 and C8 are the median of three rounds. With two or three starts, a
cell is the mean of the start medians.

| Model | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp4-local | 1 | 42.597 | 81.372 | 100.747 | 65.924 | 91.339 | 144.829 | 179.467 | 3.761 |
| fp4-local | 2 | 54.879 | 102.190 | 129.957 | 79.572 | 103.715 | 166.133 | 231.882 | 2.153 |
| fp4-local | 4 | 43.099 | 82.173 | 102.364 | 66.123 | 92.821 | 145.608 | 228.526 | 2.178 |
| fp8 | 2 | 36.637 | 51.302 | 52.731 | 45.119 | 41.769 | 65.930 | 98.339 | 5.957 |
| fp8 | 4 | 47.321 | 66.450 | 69.085 | 59.242 | 55.449 | 86.447 | 129.360 | 6.854 |

The starts of one setup agreed within 0.7 % in every card cell on one Spark.
The largest gaps were on FP8 with four Sparks, where the second start was
faster in every one-chat cell: prose 46.783 and 47.859 tok/s (2.30 %), code
65.530 and 67.370 (2.81 %). With FP4-Local on two Sparks, structured output
gave 130.833, 128.910 and 130.128 tok/s (1.47 % between the first two
starts). FP8 on two Sparks agreed within 0.4 % except C8, 98.933 and 97.744
tok/s (1.20 %).

### A 32K-token chat and its second question

Each chat sent one 32,099-token question, then a second question on the
same chat (32,238 tokens with FP8 on two Sparks, 32,305 on the others). The
second question ran twice with the cache, then once with `cache` false.
Times are seconds to the first token, the mean of the starts. On one Spark,
the chat ran on two starts of its own at context 65536. The other chats ran
on the card's start.

| Setup | First question | Second question, cache | Tokens reused | Second question, no cache |
| --- | ---: | ---: | ---: | ---: |
| FP4-Local, one Spark | 15.66 | 0.36 | 32,064 | 15.84 |
| FP4-Local, two Sparks | 10.13 | 0.75 | 32,064 | 9.93 |
| FP4-Local, four Sparks | 10.52 | 10.30 | 0 | 10.28 |
| FP8, two Sparks | 24.11 | 0.94 | 32,064 | 23.93 |
| FP8, four Sparks | 27.49 | 1.05 | 32,064 | 27.54 |

The FP4-Local chain on four Sparks reused no tokens. The layer chain runs
with the prefix cache off: its first rank logged `prefix_cache=false` and
`prefix_cache_room_bytes=0`.
On every start, the second question gave the same token IDs with the cache
and without it.

### Context on one Spark

One Spark served `vade up --sparks 1 --context 262144 --output 8192` on one
server start, with the client and prompts of the
[2026-10-29 context run](#context-on-one-spark-main-cbbb11b). Each cold
prompt ran twice with `cache` false. Between the two runs, a 64K-token chat
sent a second question with the cache on.

| Prompt | Prompt tokens | First token (s) | Writing speed (tok/s) |
| --- | ---: | ---: | ---: |
| 64K, cold | 64,217 | 32.86, 33.08 | 55.4, 57.3 |
| 128K, cold | 129,220 | 71.31, 71.74 | 50.9, 51.8 |
| 64K chat, second question, cache (64,192 tokens reused) | 64,444 | 0.48 | 75.6 |
| 64K chat, second question, `cache` false | 64,444 | 33.19 | 76.9 |

Both cold prompts gave the same token IDs as on 2026-10-29 at `cbbb11b`, in
every run. The second question gave the same token IDs with the cache and
without it.

### Token IDs

- FP4-Local: the cards on one, two and four Sparks gave the same token IDs,
  66 of 66 responses, in every start. They match the 2026-10-28 end-of-day
  cards (66 of 66) and the 2026-10-29 runs (60 of 60; the other 6 requests
  have a different form in the 2026-10-29 card client). The 32K chat gave
  the same IDs on one, two and four Sparks.
- FP8: both two-Spark starts match each other and the 2026-10-28
  end-of-day two-Spark card (66 of 66), and both four-Spark starts match the
  2026-10-28 end-of-day four-Spark card (66 of 66). Two and four Sparks still
  give different IDs from each other: 7 of 66 responses are the same, as on
  2026-10-28 (see [FP8 on two and four Sparks](#fp8-on-two-and-four-sparks-different-token-ids)).

### Two pairs at once

Later that morning, at main `51dce02` (engine `380d8e5`), we ran the same
card on two separate FP4-Local pairs, each its own tensor split, alone and
then both at once. Each pair cell is the mean of two starts. The four-Spark
chain column is the first start of its morning card, not the two-start mean
above. Rates in tok/s, cold 8K in seconds.

| Cell | Four-Spark chain | Pair 1 at once | Pair 2 at once | Both pairs, sum |
| --- | ---: | ---: | ---: | ---: |
| Prose | 43.117 | 55.811 | 54.812 | 110.623 |
| Code | 82.140 | 104.906 | 102.487 | 207.394 |
| C4 | 145.751 | 168.238 | 167.404 | 335.643 |
| C8 | 228.156 | 233.339 | 231.824 | 465.163 |
| Cold 8K (s) | 2.171 | 2.148 | 2.140 | |

The two pairs did not slow each other: with eight chats on each, they gave
465.163 tok/s at once and 464.423 tok/s as the sum of their runs alone. Each
pair's token IDs matched the morning pair cards in 66 of 66 responses.

### Limits of the 2026-10-30 runs

- Two or three server starts per card setup. The context row had one start.
- No drafting OFF cards. The drafting OFF rows of the current table are
  older.
- The records of these runs are not in the repository yet. They are the lab
  report `own-rigmark-eod-r2` and twelve card folders dated `2026-10-30`,
  from `0702` to `0804`.

## History

The sections below are older runs, newest first. Each names its date and its
main, and each stays as it was measured. They are not current: for the
current numbers, use the [current table](#current-main).

<a id="current-2026-10-29-main-cbbb11b-home-dgx-sparks"></a>
<a id="previous-2026-10-29-main-cbbb11b-home-dgx-sparks"></a>
## Previous (2026-10-29, main cbbb11b, our DGX Sparks)

On 2026-10-29 we measured main `cbbb11b` (#186) again on one Spark and on
two Sparks. Four Sparks were not measured again. Every start ran the
prebuilt payload of `cbbb11b` (flash-serve `cad21fbe`, PTX `4288b2a8`),
with context 32768, greedy decoding, thinking OFF and the prefix cache on
(the default). FP4-Local used MTP 3 and output 8192. FP8 used MTP 2 and
output 16384. Each server start ran a drafting ON card, a drafting OFF card
and a 32K-token chat.

These runs used our own card client, not RigMark. Each card sends 64
requests: prose, code and structured output, one, four and eight chats at
once, and five cold 8K prompts. It sends the cold 8K prompt with drafting
OFF (`"draft": "plain"`).

| Setup | Server starts | Notes |
| --- | ---: | --- |
| FP4-Local, one Spark | 2 | `vade install` and `vade up --sparks 1` from a bundle of the payload |
| FP4-Local, two Sparks (tensor split) | 1 | Drafting ON card and the 32K chat. The drafting OFF card measured only C8 |
| FP8, two Sparks (ring) | 2 | Drafting ON and OFF cards and the 32K chat on each start |

### Speed

Rates in tok/s, cold 8K in seconds. With two starts, a cell is the mean of
the two start medians. The records behind every table in this section are
named under [limits of these runs](#limits-of-these-runs).

| Model | Sparks | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp4-local | 1 | ON | 41.944 | 80.247 | 99.962 | 64.058 | 145.691 | 176.690 | 3.791 |
| fp4-local | 1 | OFF | 27.891 | 27.707 | 27.915 | 27.079 | 85.042 | 138.608 | 3.764 |
| fp4-local | 2 | ON | 54.299 | 100.418 | 128.086 | 78.009 | 164.517 | 228.004 | 2.145 |
| fp4-local | 2 | OFF | | | | | | 126.398 | |
| fp8 | 2 | ON | 36.549 | 51.362 | 52.610 | 44.995 | 67.593 | 97.303 | 5.961 |
| fp8 | 2 | OFF | 26.887 | 26.727 | 26.914 | 26.050 | 66.682 | 97.378 | 5.966 |

Each concurrency level runs three rounds. In the FP4-Local drafting ON
cards, the first C4 and C8 rounds were the slowest. For example, the first
one-Spark start gave 171.645, 177.842 and 179.926 tok/s in its three C8
rounds. Other cards did not show this. The same start's drafting OFF C8
rounds gave 139.410, 138.126 and 138.706 tok/s, and FP8 start `a2` gave C4
rounds of 67.859, 68.329 and 66.377 tok/s. On the FP4-Local rows, C4 and C8 are the mean of the
three round medians, so each round counts once. On the FP8 rows, C4 and C8
are the median of the three rounds. In every drafting ON C4 and C8 round of
these runs, all chats started together (split rounds 0 in each record).

The two starts on one Spark gave prose 41.976 and 41.911 tok/s (0.15 %
apart), code 80.376 and 80.117 (0.32 %) and C8 176.471 and 176.909 (0.25 %).
The two FP8 starts gave C4 67.327 and 67.859 tok/s (0.79 %) and C8 97.731
and 96.874 (0.88 %).

### A 32K-token chat and its second question

Each start sent one 32,099-token question, then a second question on the
same chat (32,305 tokens with FP4-Local, 32,238 with FP8). The second
question ran three times: twice with the cache, then once with `cache`
false. Times are seconds to the first token, one value per start.

| Setup | First question | Second question, cache | Tokens reused | Second question, no cache |
| --- | ---: | ---: | ---: | ---: |
| FP4-Local, one Spark | 15.95, 15.94 | 0.53, 0.53 | 32,064 | 16.13, 16.17 |
| FP4-Local, two Sparks | 10.09 | 0.75 | 32,064 | 9.91 |
| FP8, two Sparks | 24.08, 24.02 | 0.89, 0.89 | 32,064 | 23.91, 23.89 |

The second cached run took 0.21 s on one Spark, 0.47 s on two Sparks and
0.31 s with FP8, because it also reused the first run's prompt. On every
start, the second question gave the same token IDs with the cache and
without it.

### Token IDs

- FP4-Local: the cards on one Spark with drafting ON and OFF, in both starts,
  and the card on two Sparks with drafting ON gave the same token IDs, 64 of
  64 responses. They also match the cards of the main before. The 32K chat
  gave the same IDs on one Spark and on two Sparks.
- FP8 on two Sparks: drafting ON and OFF gave the same token IDs, 64 of 64
  responses, in both starts.

### Context on one Spark (main cbbb11b)

On 2026-10-29 (record `own-ctx-1spark-2-r1`), one Spark served
`vade up --sparks 1 --context 262144
--output 8192` at main `cbbb11b`, with FP4-Local, drafting ON (depth 3),
greedy decoding and thinking OFF, on one server start. The runs went in this
order: the 64K prompt, the 128K prompt, the 64K prompt again, a 64K-token
chat that sent a second question with the cache on, then the 128K prompt
again. The cold prompts used `cache` false, and both runs of each gave the
same token IDs.

| Prompt | Prompt tokens | First token (s) | Writing speed (tok/s) |
| --- | ---: | ---: | ---: |
| 64K, cold | 64,217 | 33.18, 33.36 | 56.5, 56.7 |
| 128K, cold | 129,220 | 71.85, 71.93 | 50.3, 51.3 |
| 64K chat, second question, cache (64,192 tokens reused) | 64,444 | 0.54 | 74.6 |
| 64K chat, second question, `cache` false | 64,444 | 33.59 | 76.0 |

The second question gave the same token IDs with the cache and without it.

The earlier 64K and 128K first-token times on this page ran on two Sparks,
so they do not compare with these.

### Limits of these runs

- The two-Spark FP4-Local card had one server start, and its drafting OFF
  card measured only C8. The full two-Spark drafting OFF card is still the
  2026-10-28 card at `76977df`.
- Four Sparks were not measured on `cbbb11b`.
- The card client sends 64 requests, and RigMark sends 66 with two chats at
  once as well. So these rows compare only roughly with the RigMark cards.
- The records of these runs are not in the repository yet. They are the
  lab reports `own-card-1s-eod-2-r2` (one Spark, starts `r2s1` and `r2s2`),
  `own-premerge-fp4c-r1` (two Sparks, the CARD entry for start `pc186`),
  `own-premerge-fp8b-r2` (FP8, the final CARD entry, starts `a1` and `a2`)
  and `own-ctx-1spark-2-r1` (the context test).

<a id="confirm-run-main-285fa8a"></a>
## Confirm run (main 285fa8a, after the end-of-day cards)

Main `285fa8a` changed the engine after the end-of-day cards at `76977df`.
With drafting on, two Sparks now keep up to 131,072 prompt rows for the
drafter. Before, they kept 32,768. The
[changelog](../CHANGELOG.md) describes the change. The 2026-10-28 end-of-day cards measure `76977df`.
They do not measure `285fa8a`.

A confirm run on 2026-10-28 compared `285fa8a` with the main before it,
`54392da`. Its record is `2026-10-28_1713_confirm_285fa8a-vs-54392da`. The
record is not in the repository yet. The run used one pair of our DGX
Sparks, two Sparks on the tensor split, with drafting ON. It used our own
64-request card client, not the RigMark recipe of the end-of-day cards: it
has no C2 round and five cold 8K prompts per server start. Each card had its own server start. The four card starts ran in this
order: `a1` (old main), `x1` (new main), `x2` (new main), `a2` (old main).
Then a 129,220-token prompt ran on its own server start for each main:
`t128-x1` (new main), then `t128-a1` (old main).

Card medians per start (rates in tok/s, cold 8K in seconds). The change
compares the mean of the two new-main starts with the mean of the two
old-main starts:

| Cell | `a1` old | `a2` old | `x1` new | `x2` new | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 54.862 | 55.015 | 54.378 | 55.014 | -0.44 % |
| Code | 99.259 | 98.661 | 98.790 | 98.890 | -0.12 % |
| Structured | 124.354 | 124.763 | 124.225 | 124.630 | -0.11 % |
| C1 | 64.794 | 64.461 | 64.475 | 64.549 | -0.18 % |
| C4 | 146.941 | 147.217 | 146.599 | 147.113 | -0.15 % |
| C8, all rounds | 206.735 | 205.135 | 205.594 | 195.717 | -2.56 % |
| Cold 8K prompt, first token (s) | 2.183 | 2.182 | 2.167 | 2.181 | 0.39 % less time |
| Responses with token IDs | 64 | 64 | 64 | 64 | IDs identical in all four starts |

The C8 loss comes from one slow start. In `x2`, the three C8 rounds reached
177.5, 195.7 and 210.7 tok/s. The other starts had a middle round between
205.1 and 206.7 tok/s. We have seen this slow-start mode on main before. The
confirm read also compared C8 on the rounds where all eight chats started
together: the first outputs of the chats came within 0.5 s of each other.
It kept 5 of 6 rounds on each main and gave 206.735 tok/s on the old main and
205.594 on the new main (-0.55 %).

The 129,220-token prompt ran four requests per start, in this order:
drafting OFF, drafting ON, drafting OFF, drafting ON. Times are seconds to
the first token:

| Request | `t128-a1` old | `t128-x1` new | Change |
| --- | ---: | ---: | ---: |
| Drafting ON, first and second request | 42.086, 42.094 | 40.426, 40.305 | -4.10 % (means 42.090 and 40.366) |
| Drafting OFF, first and second request | 39.784, 39.659 | 39.756, 39.700 | +0.02 % (means 39.722 and 39.728) |

All eight long-prompt replies had the same token IDs, on both mains, with
drafting ON and OFF.

Limits of this run:

- One pair, two starts per main for the card, and one start per main for the
  long prompt. This is a check that nothing got worse, not a full card.
- The records of this run are not in the repository yet.

<a id="context-windows-main-285fa8a"></a>
## Context windows (main 285fa8a)

On 2026-10-28, a context test checked that the large context windows start
and answer. It used a build of main `285fa8a` (flash-serve
`320bcc58`, PTX `4288b2a8`) on our DGX Sparks, with FP4-Local, drafting ON
(depth 3), greedy decoding and thinking OFF. Each setup had a fresh server
start and two cold requests (`cache` false), each with a 64-token reply. The
one-Spark setup ran on two Sparks, each as its own server.

| Setup | Context | Prompt tokens | First token (s) | Writing speed (tok/s) | GPU memory peak per process |
| --- | ---: | ---: | ---: | ---: | ---: |
| One Spark (first) | 131,072 | 120,624 | 66.31, 66.47 | 48.81, 49.69 | 95,653 MiB |
| One Spark (second) | 131,072 | 120,624 | 66.23, 66.45 | 48.27, 49.38 | 95,653 MiB |
| Two Sparks, tensor split | 262,144 | 201,654 | 68.02, 68.10 | 42.69, 41.22 | 101,474 and 101,399 MiB |

The engine refused no request and did not lower either context. Each setup
gave the same token IDs in both runs, and the two one-Spark servers gave the
same IDs. The memory peaks are samples of each engine process's GPU memory,
taken about once a second.

Limits of this test:

- The prompts did not fill the whole window. The test did not run several
  long chats at once, drafting OFF or the full exactness gate.
- The one-Spark and two-Spark prompts have different lengths, so their IDs
  do not compare one Spark with two for the same request.
- Main `250fb49` changed how chats reserve memory after this test. The test
  did not run on that main.
- The records of this test are not in the repository yet.

**One Spark at 262,144.** On 2026-10-29, a second context test (run
`c262144`) served `--context 262144` on one Spark at main `250fb49`
(flash-serve `284f0570`, PTX `4288b2a8`), after the change to memory
reservation. It sent the same 201,654-token request as the two-Spark setup
above, twice, on one server start, with the same settings.

| Setup | Context | Prompt tokens | First token (s) | Writing speed (tok/s) | GPU memory peak per process |
| --- | ---: | ---: | ---: | ---: | ---: |
| One Spark | 262,144 | 201,654 | 120.32, 120.78 | 48.39, 50.02 | 100,786 MiB |

The engine refused no request. Both runs gave the same token IDs. The
records of this run are not in the repository yet.

<a id="card-at-main-250fb49"></a>
## Card at main 250fb49 (after the change to memory reservation)

Main `250fb49` changed how a Flash-Next chat reserves memory: a chat now
reserves only the KV pages for the rows it uses, not the pages for the
whole context. The [changelog](../CHANGELOG.md) describes the change.

**Capacity.** A capacity test on 2026-10-28 started the engine with
`--context 131072` on two Sparks and sent 16 short chats (1,247 and 1,248
tokens), one each second; each asks for 1,200 output tokens, so all 16 are
sent before the first one ends. All 16 were active together, against 2 on
main `54392da`. The
median queue wait was 0.16 s for chats 1 to 9 and 1.01 s for chats 10 to
16, against 44.8 s and 128.9 s on `54392da`. The 0.16 s and 1.01 s come
from run `cap-r3-x`, which measured the change of main `250fb49` before it
merged, on a patch commit that is not public. The `54392da` figures come
from run `cap-c2b-a`, on a different pair of Sparks. An earlier run of the
change on that pair, `cap-c2b-x`, gave 1.02 s for chats 10 to 16; the
changelog entry quotes that run. The token IDs did not change.

**Speed.** On 2026-10-28, a card measured main `250fb49` on one pair of
our DGX Sparks (tensor split), in three fresh server starts. It used our
own card client, with context 32768, output 8192, drafting ON (depth 3),
thinking OFF and five cold 8K prompt runs per start. The client sends the
cold 8K prompt with drafting OFF (`"draft": "plain"`). Medians in tok/s:

| Prose | Code | Structured | C1 | C4 | C8 | Cold 8K prompt, first token (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.119 | 100.621 | 129.514 | 78.927 | 165.061 | 224.065 | 2.142 |

Every start gave the same token IDs, and they matched the cards before the
change. C8 is the median of all nine rounds. On the seven rounds where all
eight chats started together, C8 was 230.531.

Limits of this card:

- It uses a different client and a different pair from the end-of-day
  cards, so it does not compare speeds with them.
- The records of this card and of the capacity test are not in the
  repository yet.

<a id="current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks"></a>
<a id="previous-2026-10-28-end-of-day-main-76977df-home-dgx-sparks"></a>
## Previous (2026-10-28 end of day, main 76977df, our DGX Sparks)

These cards ran at main `76977df`. They do not measure later engine changes
on main. Main `76977df` changes only documentation after `9d2ec6c`
(#142), so the engine is the same as in the cards after #142. Every card ran
the prebuilt payload of `76977df` (flash-serve `4dedff9d`, PTX `4288b2a8`),
with context 32768, greedy decoding, thinking OFF and no `VADE_` name at
any server start. Each card passed its **15/15** basic output checks and had its own server start. Drafting OFF sends `draft: plain` on
every request. The one-Spark cards used `vade install` and
`vade up --sparks 1` from a bundle of the same payload. The two-Spark and
four-Spark cards used the lab launcher. FP4-Local used MTP 3 and output 8192.
FP8 used MTP 2 and output 16384, as on 2026-10-25 and 2026-10-27. All cards
used the same pinned RigMark client, with three cold 8K prompt pairs (66
responses).

Card identities (finish time and SHA256 of each card file):

- 1 Spark, local, drafting OFF: finished `2026-10-28T18:43:01.904860+00:00`; card SHA256 `257c5dcf80ed8ad1cdbd266735e3eba7dd9a7ad02e15ae4cf1e55dad3ef3ea54`.
- 1 Spark, local, drafting ON: finished `2026-10-28T18:36:30.110067+00:00`; card SHA256 `e1f30da8e92d727a5f49c9493b0acdb4a4ea6f86d4937444e7ff4741b7a11221`.
- 2 Sparks, ring, drafting OFF: finished `2026-10-28T19:44:41.463006+00:00`; card SHA256 `1ee05800f41f2424e80de00ca1c09dacdc3c899ccbce6cbbc494c2801964f69e`.
- 2 Sparks, ring, drafting ON: finished `2026-10-28T19:40:31.787574+00:00`; card SHA256 `24134290373fe44fb8fbc5be0277f8ceab2499a17347b57441e51a0ecece3997`.
- 2 Sparks, tensor split, drafting OFF: finished `2026-10-28T18:58:50.757964+00:00`; card SHA256 `d9edf41a3cea1b240fb8e330862a8835fc6102b867606786d4fca9e99c58933a`.
- 2 Sparks, tensor split, drafting ON: finished `2026-10-28T18:55:24.379380+00:00`; card SHA256 `9f6adc1a65ce34c2ca8c96b19fd660a2c00608b8e8c7868200656a57c7a333cf`.
- 4 Sparks, path, drafting OFF: finished `2026-10-28T20:03:46.518365+00:00`; card SHA256 `56ccef9d52df49d8faf9fba12843c0e1725b88a56f73b4d22eaa1b36ab76721b`.
- 4 Sparks, path, drafting ON: finished `2026-10-28T19:52:29.112169+00:00`; card SHA256 `0bce50fb52d3b546063248dca6bbadbe1f8f43debb4e522d838ff4a05fca6512`.
- 4 Sparks, pipeline, drafting OFF: finished `2026-10-28T19:30:09.669329+00:00`; card SHA256 `c9c4beed5eb735a624e1e14b340f39585ba15f471abbafd627582759136a3ae5`.
- 4 Sparks, pipeline, drafting ON: finished `2026-10-28T19:17:33.795856+00:00`; card SHA256 `d7c59327f7f1686889a753ccda4ece58d850abf6bc3f482d88a453ef9487ab11`.

### FP4-Local

| Sparks | Path | Drafting | Machines | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | local | ON | home-5 | 42.0 | 80.5 | 99.9 | 64.1 | 90.6 | 143.9 | 177.5 | 3.76 | `2026-10-28_1336_qwen3.8-flash-next-nvfp4_local-1spark_76977df-drafting-on-thinking-off` |
| 1 | local | OFF | home-6 | 27.6 | 27.6 | 27.9 | 27.0 | 46.6 | 83.6 | 130.6 | 3.81 | `2026-10-28_1343_qwen3.8-flash-next-nvfp4_local-1spark_76977df-drafting-off-thinking-off` |
| 2 | tensor split | ON | home-3 + home-4 | 53.8 | 100.0 | 128.4 | 76.9 | 100.7 | 162.5 | 229.5 | 2.19 | `2026-10-28_1355_qwen3.8-flash-next-nvfp4_local-2spark_76977df-drafting-on-thinking-off-split` |
| 2 | tensor split | OFF | home-1 + home-2 | 35.1 | 35.4 | 35.5 | 33.8 | 49.2 | 82.5 | 126.3 | 2.15 | `2026-10-28_1358_qwen3.8-flash-next-nvfp4_local-2spark_76977df-drafting-off-thinking-off-split` |
| 4 | layer chain | ON | home-1 + home-2 + home-3 + home-4 | 42.8 | 81.3 | 101.7 | 65.8 | 92.3 | 145.8 | 192.9 | 2.18 | `2026-10-28_1417_qwen3.8-flash-next-nvfp4_local-4spark_76977df-drafting-on-thinking-off-chain` |
| 4 | layer chain | OFF | home-1 + home-2 + home-3 + home-4 | 28.2 | 28.2 | 28.5 | 27.6 | 47.4 | 85.0 | 139.3 | 2.18 | `2026-10-28_1430_qwen3.8-flash-next-nvfp4_local-4spark_76977df-drafting-off-thinking-off-chain` |

### FP8

| Sparks | Path | Drafting | Machines | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | ring | ON | home-1 + home-2 | 36.6 | 50.9 | 52.7 | 45.1 | 41.9 | 66.5 | 98.3 | 5.97 | `2026-10-28_1440_qwen3.8-flash-next-fp8_local-2spark_76977df-drafting-on-thinking-off` |
| 2 | ring | OFF | home-3 + home-4 | 26.8 | 26.7 | 26.9 | 25.8 | 41.7 | 66.6 | 99.3 | 6.01 | `2026-10-28_1444_qwen3.8-flash-next-fp8_local-2spark_76977df-drafting-off-thinking-off` |
| 4 | path | ON | home-1 + home-2 + home-3 + home-4 | 46.7 | 65.1 | 68.0 | 58.5 | 55.0 | 86.4 | 128.8 | 6.91 | `2026-10-28_1452_qwen3.8-flash-next-fp8_local-4spark_76977df-drafting-on-thinking-off-chain` |
| 4 | path | OFF | home-1 + home-2 + home-3 + home-4 | 35.3 | 34.7 | 35.3 | 33.9 | 55.6 | 88.2 | 128.2 | 6.95 | `2026-10-28_1503_qwen3.8-flash-next-fp8_local-4spark_76977df-drafting-off-thinking-off-chain` |

The cards ran in pairs at the same time on separate machines: the one-Spark
cards together, the two-Spark FP4-Local cards together, and the two-Spark
FP8 cards together. The four-Spark cards ran alone. The result folder names
end in a machine suffix and `-eod-1s`, `-eod-2s`, `-eod-4s`, `-eod-fp8-2s`
or `-eod-fp8-4s`; the table gives the names without the suffix.

**FP4-Local gave the same answers on one, two and four Sparks.** The token
IDs matched in 66 of 66 responses in every compare: drafting ON against OFF,
one Spark against two and four, and the one-Spark and two-Spark cards
against the cards of the same Spark count earlier on 2026-10-28. No earlier
four-Spark FP4-Local card ran on 2026-10-28.

**FP8 gives different answers on two and on four Sparks.** On each Spark
count, the drafting-ON card matched the drafting-OFF card in 66 of 66
responses. The two-Spark cards matched the 2026-10-27 FP8 cards in 66 of 66,
and the four-Spark cards matched the 2026-10-25 four-Spark FP8 cards in 62 of
62 (those cards have one cold 8K pair, not three). But the two-Spark and
four-Spark cards matched in only 7 of 66 responses. The 2026-10-25 FP8 cards
show the same gap: 5 of 62. This is a known limit of FP8 on this main. It is
not a change of this main. See
[the cause](#fp8-on-two-and-four-sparks-different-token-ids) below.

**FP8 on four Sparks is not the layer chain.** FP8 runs tensor parallel on
two Sparks (a ring) and on four Sparks (a path): every Spark runs every layer
and holds a quarter of each one. Only FP4-Local uses the layer chain on four
Sparks. The four-Spark FP8 result folder names end in `-chain`. That suffix
names the four-Spark set of cards, not the FP8 layout.

**Four Sparks run one chat at the speed of one Spark.** The layer chain
passes each token through the four Sparks in turn, so one chat does not get
faster. With drafting ON, eight chats reached 192.9 tok/s on four Sparks,
against 177.5 on one Spark and 229.5 on the two-Spark tensor split. On
2026-10-25 the four-Spark chain served eight chats without drafting, at
136.5 tok/s; now it drafts at every chat count. The cold 8K prompt is fastest
on two and four Sparks (2.18 to 2.19 s, against 3.76 s on one Spark).

**FP8 drafts at one chat only.** With drafting ON, FP8 drafted at depth 2 on
the one-chat cells and on C1. It served C2 to C8 at depth 0, except one C4
stream on two Sparks that drafted (the `draft_used` field of round 1,
stream index 3). So its C4 and
C8 rates with drafting ON (66.5 and 98.3 tok/s) are close to the rates with
drafting OFF (66.6 and 99.3).

**One rate moved between two starts.** On one Spark with drafting OFF, C8
reached 130.6 tok/s, against 137.6 on the same machine at 13:02 the same day
(main `6926aaa`, the same engine). The best of the three rounds was 137.9 in
both cards, so this is the spread between server starts.

### FP8 on two and four Sparks: different token IDs

FP8 is a supported route on two and on four Sparks ([FP8 install](install-fp8.md)). In the cards below, the two Spark counts gave different token IDs for the same request and settings:

<div align="center">

| Cards | Main | Two-Spark against four-Spark responses with identical token IDs |
| :---: | :---: | :---: |
| 2026-10-31 morning | `dc32b7c` | 7 of 66 |
| 2026-10-30 morning | `380d8e5` | 7 of 66 |
| 2026-10-28 end of day | `76977df` | 7 of 66 |
| 2026-10-25 | `6cb46bf` | 5 of 62 |

</div>

On each Spark count alone, FP8 was stable. At `76977df`, drafting on and off matched in 66 of 66 responses on two Sparks and on four. The two-Spark cards matched the 2026-10-27 FP8 cards in 66 of 66. The four-Spark cards matched the 2026-10-25 four-Spark cards in 62 of 62.

The [exactness rule](architecture.md#exactness-and-model-quality) asks for the same token IDs on every supported Spark count. FP4-Local meets it on one, two and four Sparks. In these cards, FP8 did not meet it.

A host test found the cause: the order of additions differed between the two rank counts. On four Sparks, three projections split their inner sum across the ranks: the output projections of attention and of the linear-attention layers, and the down projection of the shared expert. The ranks then added their parts in a different order than on two Sparks, and the routed experts grouped their blocks differently. The results differed in the last bits, and greedy decoding could then pick a different token.

**The fix.** A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order. Four Sparks now give the same token IDs as two Sparks. In the pre-merge check, four Sparks gave the same token IDs as two Sparks in 66 of 66 responses, with drafting on and with drafting off (private record, not in this repository). Six Sparks still give other IDs than two ([changelog](../CHANGELOG.md)).

The cards on this page predate the fix, so compare their FP8 results only at the same Spark count. `vade verify` checks one running engine, so it does not show a difference between Spark counts.

### A long chat and its second question

These runs used one fresh FP4-Local server per run, drafting ON, outside the
cards. Each run asked a first question on a 32K-token chat (32,103 tokens).
Then it asked a second question on the same chat (32,309 tokens) with the
cache allowed, and the same question again with `cache` set to false. Times
are seconds to the first token. The token IDs were the same in every run, on
every Spark count, with the cache on or off.

| Sparks | Context | Prefix cache | First question | Second question | Reused tokens | Second question, cache false | Run folder |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 65536 | off | 16.04 | 16.03 | 0 | 16.08 | `eod-long` on home-5 |
| 1 | 65536 | on | 16.23 | 0.63 | 32,064 | 16.17 | `eod-long` on home-6 |
| 2 | 131072 | off | 10.31 | 9.95 | 0 | 9.97 | `eod-wtT-a1` |
| 2 | 131072 | on | 11.09 | 0.76 | 32,064 | 9.98 | `eod-wtT-x1` |
| 4 | 65536 | off | 10.77 | 10.33 | 0 | 10.90 | `long32k` |

The prefix cache was turned on with `VADE_FLASHNEXT_PREFIX_CACHE=1` at the
server start, as [the prefix cache](../INSTALL.md#the-prefix-cache)
says. These runs did not try the cache on four Sparks, and did not measure
FP8.

<a id="current-2026-10-28-after-142-main-9d2ec6c-home-dgx-sparks"></a>

<a id="previous-2026-10-28-after-142-main-9d2ec6c-home-dgx-sparks"></a>
## Previous (2026-10-28 after #142, main 9d2ec6c, our DGX Sparks)

These cards ran at main `9d2ec6c`, after the merge of #142. With #142, the
prefix cache can restore a prompt on both Sparks of the tensor split. The
cache stays off unless `VADE_FLASHNEXT_PREFIX_CACHE=1` is set, so the cards
measure the default path. All cards ran on home-3 + home-4 with the recipe
of the [cards after #139](#previous-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks):
context 32768, output 8192, MTP 3, greedy, cache off, thinking OFF, and no
`VADE_` name at any server start. Each card passed its **15/15** basic
output checks and ran the prebuilt payload of `9d2ec6c`, installed and
hash-checked on both Sparks. Each card had its own server start. The
drafting-ON card ran first, then the drafting-OFF card, then the prefix-cache
runs below, then a second drafting-OFF card. The table at the top of this
page does not show these cards.

| Path | Drafting | Commit | Sparks | Prose | Code | Structured | C1 | C2 | C4 | C8 | Cold 8K prompt (s) | Result folder |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Tensor split | ON | `9d2ec6c` | home-3 + home-4 | 53.8 | 100.0 | 128.5 | 78.9 | 102.2 | 164.1 | 227.4 | 2.14 | `2026-10-28_1228_qwen3.8-flash-next-nvfp4_local-2spark_9d2ec6c-drafting-on-thinking-off-split` |
| Tensor split | OFF | `9d2ec6c` | home-3 + home-4 | 35.2 | 35.5 | 35.6 | 34.2 | 48.5 | 82.0 | 126.0 | 2.16 | `2026-10-28_1239_qwen3.8-flash-next-nvfp4_local-2spark_9d2ec6c-drafting-off-thinking-off-split` |
| Tensor split | OFF, second start | `9d2ec6c` | home-3 + home-4 | 35.1 | 35.5 | 35.4 | 34.3 | 49.1 | 82.4 | 125.3 | 2.17 | `2026-10-28_1311_qwen3.8-flash-next-nvfp4_local-2spark_9d2ec6c-drafting-off-thinking-off-split` |

**The answers did not change.** The token IDs matched in 66 of 66
responses: the ON card against each OFF card, and every card against the
`8b7b866` card. The first OFF card also matched the `104205a` OFF card.

**The rates did not change.** With drafting ON, every rate is within 0.4 %
of the `8b7b866` card on the same pair. C8 reached 227.4 tok/s against
228.2, and the cold 8K prompt took 2.14 s against 2.17 s. With drafting OFF,
the first start had C2 at 48.5 tok/s and C4 at 82.0. These are 1.2 % and
0.9 % below the `104205a` OFF card (49.1 and 82.7). The second start had
49.1 and 82.4. Over the two starts, every OFF rate is within 0.7 % of the
`104205a` card, except code at 35.5 against 35.0 (1.3 % faster). The cold
8K prompt took 2.16 s and 2.17 s, against 2.21 s.

### The prefix cache on two Sparks

These runs used the test client of #142 on home-3 + home-4 at main
`9d2ec6c`. Each run started a fresh server with drafting ON. Each run asked a first
question on a long chat. Then it asked a second question on the same chat
with the cache allowed, and the same second question again with `cache` set
to false. Three runs had the cache on (`VADE_FLASHNEXT_PREFIX_CACHE=1`), and
three ran the default path, in the order default, on, on, default, default,
on. The driver also sets `VADE_W2_PP_RDMA=1`, which this engine does not
read. The receipts are the run folders `r142-wtT-a1` to `r142-wtT-a3` and
`r142-wtT-x1` to `r142-wtT-x3` for 32K, and `r142-wtV-a1` to `r142-wtV-a3`
and `r142-wtV-x1` to `r142-wtV-x3` for 100K. Each cell gives the three runs
in order. Times are seconds to the first token.

| Chat | Context | Cache | First question | Second question | Reused tokens | Second question, cache false |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 32K (32,103 tokens) | 131072 | off (default) | 10.38, 10.40, 10.34 | 10.01, 9.98, 9.98 | 0 | 10.02, 9.99, 9.99 |
| 32K (32,103 tokens) | 131072 | on | 11.09, 11.07, 11.07 | 0.77, 0.76, 0.75 | 32,064 | 9.99, 9.98, 9.99 |
| 100K (100,364 tokens) | 110592 | off (default) | 31.75, 32.22, 31.81 | 31.84, 31.86, 31.79 | 0 | 31.87, 31.96, 31.79 |
| 100K (100,364 tokens) | 110592 | on | 32.97, 33.14, 33.07 | 1.62, 1.62, 1.60 | 100,352 | 32.15, 31.80, 31.85 |

In every run with the cache on, the second Spark restored the same number of
tokens that the first Spark reused. In both arms, every first question gave
the same token IDs, and so did every second question, in 18 of 18 requests
at each size.

**The cache costs time on the first question.** While it reads the first
question, the engine saves the prompt state on both Sparks. At 32K the
median first question took 11.07 s with the cache on, against 10.38 s
(+0.69 s, +6.7 %). At 100K it took 33.07 s, against 31.81 s (+1.26 s,
+4.0 %). In each pair of neighboring runs, the cost was 0.67 to 0.73 s at
32K and 0.92 to 1.26 s at 100K.

**At 100K, only the second question gains.** To save the second question,
the cache drops the saved first question. Then memory pressure drops the new
copy too, and the server log shows 0 retained bytes. So a third question on
the same chat would read the whole prompt again. These runs did not measure
100K at context 131072.

<a id="current-2026-10-28-after-139-main-104205a-home-dgx-sparks"></a>
<a id="current-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks"></a>

<a id="previous-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks"></a>
## Previous (2026-10-28, main 8b7b866 and 104205a, our DGX Sparks)

The newest card of this set is one drafting-ON card at main `8b7b866`
(#140), in the first row below. No drafting-OFF card ran at `8b7b866`. The
two cards before it ran at main `104205a`, after the merge of #139,
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

<a id="previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks"></a>
## Previous (2026-10-28 morning, main bb24e80, our DGX Sparks)

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

<a id="previous-2026-10-27-final-main-09e1fc4-home-dgx-sparks"></a>
## Previous (2026-10-27 final, main 09e1fc4, our DGX Sparks)

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

**Card replay.** `vade bench --card` sends this recipe and saves every token
ID in card order. With `--receipt` or `--compare` it compares the IDs with a
receipt or with another card
([details](results.md#what-a-stranger-can-and-cannot-re-run)). This table's
cards have no receipt in this repository, and we matched the client only
against the 2026-10-31 one-Spark receipts. Saving IDs alone is not a match
check: compare every ID in order, for the same requests.

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

<a id="previous-2026-10-27-morning-main-12142bb-home-dgx-sparks"></a>
## Previous (2026-10-27 morning, main 12142bb, our DGX Sparks)

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

<a id="previous-2026-10-26-main-262b0c3-home-dgx-sparks"></a>
## Previous (2026-10-26, main 262b0c3, our DGX Sparks)

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

<a id="previous-2026-10-25-main-80ac2d4-home-dgx-sparks"></a>
## Previous (2026-10-25, main 80ac2d4, our DGX Sparks)

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

**Drafting ON makes one chat faster.** On the `80ac2d4` pair it took code
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

**More Sparks read prompts faster and give more memory.** The pipeline splits the
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

**Drafting gives the largest gain for one chat.** On one Spark it takes code from 27.8 to 76.7
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

- **One chat runs at almost the same speed on each Spark.** On six of our Sparks measured
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

## Summaries moved from the README

These dated summaries were in the README until 2026-10-28. They are kept here
as they were. Their dates and revision names apply only to those
measurements.

### One, two and four Sparks, 2026-10-28, end of day

Cards ran at main `76977df` on one, two and four Sparks, with FP4-Local on
each and FP8 on two and four. The engine is the same as in the cards after
#142. The four-Spark cards and the two-Spark FP4-Local drafting OFF card
are still in the table at the top of this page. With drafting
ON, FP4-Local reached these rates (tok/s, from the cards):

| Sparks | Prose | Code | Structured | C8 | Cold 8K prompt (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.0 | 80.5 | 99.9 | 177.5 | 3.76 |
| 2 | 53.8 | 100.0 | 128.4 | 229.5 | 2.19 |
| 4 | 42.8 | 81.3 | 101.7 | 192.9 | 2.18 |

On a 32K-token chat with the prefix cache on, the second question took
0.63 s on one Spark and 0.76 s on two. The full tables are on the
[benchmarks page](#current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks).

### Two Sparks, 2026-10-28, after #142

Two-Spark cards ran at main `9d2ec6c` (#142). With #142, the prefix cache
can restore a prompt on both Sparks of the tensor split. The cache is still
off by default at that main, so these cards measure the default path of
that time. The table at the top of this page does not show these cards. Their token IDs match the `8b7b866` card in 66 of 66
responses. With drafting ON, every rate is within 0.4 % of that card. With
drafting OFF, two server starts average within 0.7 % of the `104205a` OFF
card, and code is 1.3 % faster.

With the cache on, the second question on a 32K-token chat took 0.75 to
0.77 s instead of 9.98 to 10.01 s. On a 100K-token chat at context 110592,
it took 1.60 to 1.62 s instead of 31.79 to 31.86 s. The first question took
0.67 to 1.26 s longer, because the engine saves the prompt state. The full
tables are on the
[benchmarks page](#previous-2026-10-28-after-142-main-9d2ec6c-home-dgx-sparks).

### Two Sparks, 2026-10-28, after #139 and #140

One drafting-ON card ran at main `8b7b866` (#140). No drafting-OFF card ran
at that revision. Its token IDs match the `104205a` card in 66 of 66
responses, and its rates are within 1 % of that card.

The cards before it ran at main `104205a`, after #139. With #139, eight
chats reuse their attention arenas. Eight chats with drafting ON reached
229.6 tok/s, against 217.5 in the morning card (+12.0 tok/s, +5.5 %). With
drafting OFF, eight chats reached 125.4 against 124.3 on the same pair. The
other rates are within 1 % of the morning cards. The long-prompt cells and
the one-Spark prefix-cache cell did not change, and the token IDs match the
morning cards in 66 of 66 responses. The full tables are on the
[benchmarks page](#previous-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks).

### Two Sparks, 2026-10-28

The morning RigMark cards of 2026-10-28 ran at main `bb24e80` on the same two
pairs and with the same recipe as the final 2026-10-27 cards below. Every
rate is within 0.9 % of those cards, and the token IDs match them in 66 of 66
responses. Long prompts read faster: two Sparks now read the middle of a
prompt of 32K tokens or more in wider passes. With drafting ON, a
64,217-token prompt reached its first token in 20.04 s instead of 21.06 s,
and a 129,220-token prompt in 42.47 s. A 32,099-token prompt took 10.03 s,
as before. The full tables are on the
[benchmarks page](#previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks).

### Two Sparks, 2026-10-27

`vade up --sparks 2` runs the tensor split by default. You set nothing. The
final RigMark cards of 2026-10-27 ran on two pairs of our DGX Sparks at main
`09e1fc4`, with no split settings in the environment, context 32768, greedy
decoding and thinking OFF. One pair ran drafting ON (the default). The other
pair ran drafting OFF. Each card had its own server start. The two cards
match each other in 66 of 66 responses, and they match that morning's cards
at `12142bb`.

| Two DGX Sparks | Drafting | Prose tok/s | Code tok/s | Structured tok/s | 1 chat tok/s | 4 chats tok/s | 8 chats tok/s | First token, cold 8192-token prompt (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tensor split (the default), home-1 + home-2 | ON | 54.1 | 100.6 | 129.6 | 79.0 | 165.2 | 218.8 | 2.19 |
| Tensor split (the default), home-3 + home-4 | OFF | 35.2 | 35.4 | 35.1 | 33.8 | 82.6 | 123.9 | 2.17 |

Prose, code and structured are one chat at a time, 5 runs each with an
8192-token cap. The 1-, 4- and 8-chat rates cover all chats together, 3 rounds
with a 256-token cap. Structured output is a predictable JSON workload, not a
measure of general agent performance.

Against that morning's cards, a cold 8K prompt now reaches its first token
in 2.19 s instead of 2.80 s with drafting ON, and in 2.17 s instead of 2.77 s
with drafting OFF: each Spark reads half of a cold prompt of 2K to 8K tokens.
Drafting OFF is faster: on the same pair, one chat wrote code at 35.4 tok/s
instead of 29.6, and eight chats reached 123.9 instead of 110.8. With
drafting ON, prose, code and structured output stay within 1 tok/s of the
morning cards.

Long prompts reach their first token sooner than they did on the 2026-10-26
builds, for two reasons. The first token no longer waits while the two Sparks
swap a long prompt's layer state, and that swap now goes GPU to GPU over the
RDMA link (#114). And the first Spark now reads the second Spark's
prompt-tile replies as they arrive, so a long prompt from a chat that drafts
no longer stalls in the middle of the read (#120). On a second start of the
drafting-ON pair with context 69632, a 32,099-token prompt reached its first
token in 10.24 s, and a 64,217-token prompt took 21.06 s. The morning cards,
the drafting-OFF details and the full tables are on the
[benchmarks page](#current-2026-10-27-final-main-09e1fc4-home-dgx-sparks).

### Two Sparks, 2026-10-26

`vade up --sparks 2` runs the tensor split. These RigMark cards ran on two of
our DGX Sparks on 2026-10-26, drafting ON (the default), context 32768,
greedy decoding and thinking OFF. The split row is main `262b0c3` with the
split's four levels set by hand, the same four that are now the only code. The
pipeline row is the path two Sparks ran before, measured on main `99e8f62`
the same afternoon. That path no longer exists: the tensor split is the only
two-Spark path.

| Two DGX Sparks | Prose tok/s | Code tok/s | Structured tok/s | 1 chat tok/s | 4 chats tok/s | 8 chats tok/s | First token, cold 8192-token prompt (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tensor split (the default) | 54.2 | 100.9 | 130.4 | 76.8 | 161.4 | 220.0 | 2.81 |
| Layer pipeline (before) | 42.2 | 80.3 | 100.8 | 61.9 | 143.2 | 201.7 | 2.78 |

Prose, code and structured are one chat at a time, 5 runs each with an
8192-token cap. The 1-, 4- and 8-chat rates cover all chats together, 3 rounds
with a 256-token cap. Structured output is a predictable JSON workload, not a
measure of general agent performance. The drafting-OFF cards and a second pair
are on the [benchmarks page](#previous-2026-10-26-main-262b0c3-home-dgx-sparks).

### The v0.1.0 release, 2026-10-24

These are the RigMark cards for the v0.1.0 release (commit `5a22651`),
measured on 2026-10-24, when two Sparks ran the layer pipeline.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks-2026-10-24-dark.svg">
  <img src="assets/benchmarks-2026-10-24-light.svg" width="100%" alt="Bar chart comparing one DGX Spark with two, drafting ON (the default). In the 2026-10-24 runs, a second Spark answered a long prompt faster, but one chat did not write faster. Code, one chat: 76.7 tok/s on 1 Spark, 65.1 on 2 Sparks. Prose, one chat: 41.7 and 38.3. Eight chats together: 173.9 and 168.6. First reply after a cold 8192-token prompt: 3.86 s on 1 Spark, 2.79 s on 2 Sparks. Across all our runs, eight chats gave 130.6 to 173.9 tok/s on six single Sparks and 167.6 to 168.7 on three pairs.">
</picture>

The chart compares one of our Sparks with two, drafting ON.
`cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- benchmarks-chart` redraws it from the
[2026-10-24 receipts](../bench/receipts/2026-10-24/README.md). The table adds
drafting OFF and the cloud VMs.

How to read it:

- Every card used context 32768, greedy decoding and thinking OFF. In the OFF
  rows, every request sent `"draft":"plain"`.
- Code and prose are one chat at a time, 5 runs each with an 8192-token cap.
  The 8-chat rate covers all eight chats together, 3 rounds with a 256-token
  cap, so 24 streams in all; the table says how many of them the engine served
  without drafting. The first-token time is after a cold 8192-token prompt, 3
  runs.
- Tok/s is completion tokens per second of wall-clock time.
  [Client time and server time](#client-time-and-server-time)
  explains where that clock can start.
- We have not run another engine on this checkpoint, so this is not a
  head-to-head comparison.

| Setup | Drafting | Code tok/s | Prose tok/s | 8 chats tok/s | 8-chat streams served without drafting | First token, cold 8192-token prompt (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| One DGX Spark | OFF | 27.8 | 27.8 | 136.1 | all (OFF) | 3.82 |
| One DGX Spark | ON | 76.7 | 41.7 | 173.9 | 10/24 | 3.86 |
| Two DGX Sparks | OFF | 27.9 | 27.9 | 135.1 | all (OFF) | 2.80 |
| Two DGX Sparks | ON | 65.1 | 38.3 | 168.6 | 12/24 | 2.79 |
| One cloud GB10 VM | OFF | 24.4 | 22.5 | 73.3 | all (OFF) | 3.99 |
| One cloud GB10 VM | ON | 61.1 | 31.6 | 97.5 | 10/24 | 3.98 |
| Two cloud GB10 VMs | OFF | 24.4 | 22.9 | 112.2 | all (OFF) | 2.95 |
| Two cloud GB10 VMs | ON | 54.7 | 30.5 | 138.3 | 11/24 | 2.95 |
| Four cloud GB10 VMs | OFF | 21.9 | 20.7 | 104.2 | all (OFF) | 2.38 |
| Four cloud GB10 VMs | ON | 51.6 | 29.0 | 104.2 | 24/24 | 2.42 |

The main results:

- **One chat runs at the same speed on every Spark we tried.** On six of our
  Sparks, one at a time, drafting OFF gave 27.6 to 27.8 tok/s on prose and
  27.7 to 27.8 on code; drafting ON gave 41.3 to 41.7 on prose and 76.1 to
  76.7 on code.
- **The 8-chat rate moves more.** Under that load the engine serves some chats
  without drafting, so with drafting ON it ranged from 130.6 to 173.9 tok/s
  across the six Sparks; the one-Spark ON row is the top of that range. One
  drafting-OFF run on one of the six landed at 86.6
  ([receipt](../bench/receipts/2026-10-24/2026-10-24_1941_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-home-6/)),
  a drop we have not reproduced or explained.
- **Four cloud VMs did not make one chat faster.** They run as one engine,
  each VM holding a slice of the layers. A chain is meant for models too big
  for one machine. In its drafting-ON run the engine served all eight chats
  without drafting, so both four-VM rows show the same 8-chat rate.

Each row has its own receipt, named by the time of its run:

- One DGX Spark: [OFF (1937)](../bench/receipts/2026-10-24/2026-10-24_1937_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/), [ON (1943)](../bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/)
- Two DGX Sparks: [OFF (2012)](../bench/receipts/2026-10-24/2026-10-24_2012_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off/), [ON (2019)](../bench/receipts/2026-10-24/2026-10-24_2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off/)
- One cloud GB10 VM: [OFF (1946)](../bench/receipts/2026-10-24/2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vma/), [ON (1954)](../bench/receipts/2026-10-24/2026-10-24_1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vma/)
- Two cloud GB10 VMs: [OFF (2027)](../bench/receipts/2026-10-24/2026-10-24_2027_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-cloud-a/), [ON (2035)](../bench/receipts/2026-10-24/2026-10-24_2035_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-cloud-a/)
- Four cloud GB10 VMs: [OFF (2204)](../bench/receipts/2026-10-24/2026-10-24_2204_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-off-thinking-off-cloud-a/), [ON (2213)](../bench/receipts/2026-10-24/2026-10-24_2213_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-on-thinking-off-cloud-a/)

Raw benchmark output stays private until it has been reviewed for publication;
[the receipts list](../bench/receipts/README.md) says what is public.

## Reproduce a measurement in detail

To compare your Spark with the tables, run RigMark as the next section shows. `vade bench` gives you a timing check of your own, but it does not make cards.

Follow the [FP4-Local serving guide](../cli/docs/fp4-local.md), and keep the measured context and output limits explicit. `vade launch` runs on the Spark itself. `vade up` runs from a controller, the computer that drives your Sparks. Both start at context 32768 and output 8192. `vade launch` lowers a default context only when the engine refuses it for memory. A controller never lowers it. So check `vade status --json` before you compare runs.

After `vade launch` or `vade up`, `vade bench` works without flags. Keep the same `VADE_HOME` and `VADE_CONFIG`. Then `vade bench` finds the endpoint, reads the model ID from `/health`, and measures the server's default drafting policy with a bundled offline prompt. Receipts go to `./vade-bench.json`, with a text table next to it.

A run with no flags uses a different recipe from a card, so its numbers do not match the tables:

- **Prompt.** It sends one bundled writing prompt. RigMark uses its own prose, code and structured prompts.
- **Rounds.** It runs C1, C2, C4, C8 and C16 once each, with no warmup. A card runs C1, C2, C4 and C8 for 3 rounds each.
- **Reply length.** Like the group rounds of a card, it stops each reply at 256 tokens. It also asks the server to continue past the end-of-answer token, so every reply is exactly 256 tokens long.
- **Single-stream cells.** It has no single-stream cells with the 8192-token cap. It reads a cold 8K prompt only when you add `--prefill-body` and `--checkpoint-hash` (see [Run vade bench](#run-vade-bench)).

Both keep greedy decoding, the cache off and thinking off. On the server behind the one-Spark v0.1.0 cards, a `vade bench` run with no flags and with drafting as served gave C1 45.466 and C8 185.144 tok/s. The drafting-on card from that same server shows C1 61.4 and C8 173.9. Both are recorded in the [ON card's receipt folder](../bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/).

To compare runs with the same saved prompts and plain decoding (drafting off), use `--draft plain`. It works on a server with drafting on, without a restart. Set `VADE_API_URL` to the running server's HTTP base URL, and keep the same prompt file between runs:

```sh
vade bench --url "$VADE_API_URL" --model qwen3.8-flash-next-fp4-local --draft plain \
  --prompts ./prompts.txt --concurrency 1,2,4,8 \
  --max-tokens 256 --repeats 3 --output ./vade-bench.json
```

This command times cohorts with the Rust client. A cohort is a group of requests at once, and each request writes exactly the capped number of tokens. The command does not recreate the RigMark prompt set, and it does not turn a timing run into a card.

The drafting-on recipes of the cards leave out the draft field to get the server default. The timing runs send `"draft":"strict"`, which you ask for with `--draft strict`. Current bench requests keep thinking off. When you quote a result, keep its thinking setting, estimator and source folder with it. See [Run vade bench](#run-vade-bench) for every option.

## Run RigMark against Vade

[RigMark](https://github.com/alexellis/rigmark) is the public benchmark that makes the cards on this page. Run it on the Spark that serves HTTP (rank 0 of a pair). Point it at the loopback endpoint that `vade launch` or `vade up` reports (`http://127.0.0.1:18080` by default).

**Turn thinking off in the request.** Vade answers chat requests with thinking on unless the request says otherwise. The FP4-Local server has no server-level switch for that default, and neither `vade launch` nor `vade up` sets one.

RigMark sends no thinking field by default. Each decode sample has an **8192-token cap shared by reasoning and the answer**. With the RigMark defaults, most code and prose samples spend the whole cap on reasoning. They stop with `finish_reason` `length`, show no answer, and fail the RigMark output checks.

Set the field yourself with the RigMark option `--extra-body`, which merges JSON into every chat request. The server reads these settings in this order ([protocol.rs](../engine/crates/serve/src/protocol.rs)):

1. `chat_template_kwargs.enable_thinking`, or a top-level `enable_thinking`.
2. `reasoning_effort`. `none` or `minimal` turns thinking off. `low`, `medium`, `high` or `xhigh` turns it on at that effort.
3. The server default, which is on.

This is the command that we use on one Spark, from a RigMark checkout:

```sh
git clone https://github.com/alexellis/rigmark
cd rigmark
./rigmark configure
./rigmark bench --base-url http://127.0.0.1:18080 --model auto \
  --label vade-fp4local-1spark --comparison-id your-sweep-id \
  --metadata metadata.json --prefill-depths 8192 --concurrency 1,2,4,8 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

- **`./rigmark configure`.** It writes `metadata.json`. Set its `context_limit` to the context that your server reports (`Context` in `vade status`, or `caps.context_limit` in `/health`). The cards used **32768**.
- **`--prefill-depths 8192`.** By default, RigMark also measures 32768 and 65536. It refuses any depth where the depth + 8 is more than `context_limit`. The cards measure 8K only.
- **`--concurrency 1,2,4,8`.** It adds C8 to the RigMark default of C1, C2 and C4. Every 2026-10-24 RigMark card on this page ran with it, including the two-VM cloud card 1006 (C8 **70.0 tok/s**).
- **Everything else.** The rest is a RigMark default: 5 runs each of prose, code and structured output with the 8192-token cap, 3 cold and replay prefill pairs, and 3 rounds of code at each concurrency with a 256-token cap. RigMark fixes temperature 0, top-p 1 and seed 20260905.
- **Older checkouts.** Older RigMark checkouts call the `bench` subcommand `run`.

The published cards send a longer body. To compare with a card, send the identical body, because RigMark compares receipts only when they have the same request body:

```json
{"cache": false, "chat_template_kwargs": {"enable_thinking": false}, "draft": "plain",
 "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "top_k": 0}
```

`"draft": "plain"` makes a drafting-off card. The drafting-on cards leave the field out and get the server default. `cache`, the neutral penalties and `top_k` spell out the defaults. Cards with thinking on replace `"enable_thinking": false` with `"enable_thinking": true, "reasoning_effort": "low"` inside `chat_template_kwargs`.

**What "DO NOT HEADLINE" means.** RigMark checks each of the 15 single-stream samples. Code and prose need visible answer text and a normal stop. Structured output needs the requested JSON values. If any sample fails, the card header marks the result `DO NOT HEADLINE`. Then do not quote its rates as a result.

On 2026-10-24, a RigMark run with its default settings against a two-VM pair scored **6/15**. Code (0/5) and prose (1/5) hit the cap while they were still reasoning, and structured output passed 5/5. With thinking off through `--extra-body`, the cards above pass **15/15**. These checks, which RigMark calls gates, confirm that the answers completed. They are not a check of exactness or quality.

**What a good card looks like** with thinking off:

- **Header.** The header says `●  15/15 BASIC OUTPUT GATES PASSED`.
- **SINGLE STREAM.** PROSE, CODE and STRUCTURED* each show `✓ 5/5`.
- **PREFILL.** One 8K row shows the cold tok/s, the cold TTFT and the replay TTFT.
- **CAPPED CONCURRENT GENERATION.** The section shows `C1 | C2 | C4 | C8`, then `C8: 0/24 normal stops; 24/24 with visible output`. Every capped stream stops at 256 tokens, so zero normal stops is expected. With thinking on, those 256 tokens are often all reasoning, and the visible-output count drops toward 0.
- **SETTINGS.** The section shows `SUITE: CUSTOM SETTINGS`, `Changed: --prefill-depths=8192`, `Changed: --concurrency=1,2,4,8`, and `REQUEST:` with your body. Custom settings are expected and do not mean "DO NOT HEADLINE".
- **Rates.** The rates should be close to the [matching row](#current-results-2026-10-24) for the same Spark count and drafting mode, with thinking off. For thinking on, see the [receipt folder](../bench/receipts/2026-10-24/README.md).

## Run vade bench

Build the CLI once. Then measure a running server with one command. After `vade launch` or `vade up`, a bare `vade bench` uses the selected endpoint, the model that `/health` reports and a bundled prompt. The explicit recipe below picks retained prompts and plain drafting instead:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
cli/target/release/vade bench --url "$VADE_API_URL" \
  --model qwen3.8-flash-next-fp4-local --draft plain --prompts /path/to/retained-prompts.txt --concurrency 1,8,16 \
  --max-tokens 256 --repeats 3 --output /path/to/private-receipt.json
cli/target/release/vade bench --self-test
```

The prompt file holds one literal UTF-8 prompt per nonempty line. Keep the same prose, code and reasoning prompts between comparisons. The client cycles through them and never makes replacements. Concurrency 1 always uses the first prompt, so measure one-chat prose and code with separate retained files.

`--bodies-dir` sends saved, unchanging request bodies from the shared-reader workload instead. That workload is 16 chats that share one long context, and each chat asks for a separate response. The directory holds the shared context. The complete requests are in a `capture/` directory next to it.

For a cold read, add `--prefill-body tools/fixtures/probe-8192.json`, `--prefill-body-sha256 11869d33b6fe2411447a7c9fd34ef76895ca53017f7505e2d63ac77783df945f` and `--checkpoint-hash HASH`. The client then makes three uncached probes. It requires exactly 8,192 prompt tokens as the server reports them, and it checks the build and checkpoint identity. Those are three requests within one server start. For a performance claim, repeat after at least three fresh server starts per build, and alternate A and B.

### What the receipt records

- **Hashes.** The hash of each source file, and the hash of the exact compact UTF-8 request sent to the server.
- **Sampling settings.** The settings that every request sends explicitly: temperature 0, top-p 1, top-k 0, neutral penalties, cache off and thinking off. Drafting follows the server default unless you pass `--draft plain` or `--draft strict`.
- **The server's echo.** The server's sampling echo for every response. A response with no echo is marked `sampling_source=request`, so the client never invents the settings that took effect.

The JSON receipt keeps these fields of the timing client: `schema_version`, `status`, `arms`, `cells`, `requests`, `prefill_8k`, `receipts`, sampling, identities, memory, observation and batch-log fields. The `.txt` file next to it gives readable results and response records. Failed or partial cells keep their records but publish no rate.

### How rates are computed

Cohort throughput is all completion tokens divided by the time from the first request sent to the last one completed.

`writing_tokens_per_s` uses the older scoreboard formula, `(completion_tokens - 1) * 1000 / (elapsed_ms - ttft_ms)`. That is every final completion token minus one. TTFT here counts the first nonempty answer, reasoning or tool-call data, so it can come before the first answer text. The receipt records `first_content_ms` on its own line.

The receipt leaves `answer_content_tokens_per_s` as null. The final usage report of the server does not say how many tokens came before the first answer text. Do not fill that gap by a subtraction of reasoning estimates or by a count of streamed (SSE) chunks.

### Plain HTTP and drafting

`vade bench` speaks plain HTTP, also through an existing local SSH tunnel. HTTPS is not supported, because the standard library has no TLS implementation. Use an existing local HTTP tunnel or a trusted TLS terminator.

Servers with drafting on are supported. Receipts keep each requested draft mode apart from the MTP depth and build SHA that the server reports. They keep response observations separately. A reported depth of zero does not by itself prove that drafting was off. Only receipts where every request asked for plain are labeled drafting off. Neither the requested policy nor the reports of the server prove by themselves that drafting took effect.

### Options

Options from the older timing client keep their names. `--draft plain|strict` is new. The table lists each option, its default and what it changes in the receipt.

The A/B options alternate a baseline server (A) with a candidate (B). ABBA runs them A, B, B, A. ABA runs A, B, then A again (A′). A comparison of the two A runs shows how much the baseline drifts on its own.

<div align="center">

| Option | Meaning or exception |
| :---: | :---: |
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

</div>

### The older suite and the retired Python client

The existing `vade-bench` crate is part of the CLI as a library, so there is one command to run. `vade bench --suite` keeps its earlier workloads: decode, concurrency, cache, priority, long prompts, follow-ups and shared system prompts. See [its guide](../engine/tools/bench/README.md) for those options. That suite uses a different, historical receipt format and workload. For the current scoreboard, choose the default timing protocol.

The former Python client and its operational wrapper (shim) are retired. Their original files, and the evidence that the two clients matched on a real server, are kept outside this repository. A live check on a pair needs at least three runs of each client on identical inputs. A mock check on a Mac verifies the parsing and the receipt arithmetic. It supports no claim about Spark speed.

## Rented cloud GB10 VMs

Leave time for the model download before you benchmark rented VMs. The verified FP4-Local snapshot is about **102 GB**. On 2026-10-24, rented GB10 VMs downloaded from Hugging Face at **8 to 22 MB/s each**. That is about 80 minutes for one VM at the top of that range, and over three hours at the bottom.

For the cloud cards, we downloaded and verified the snapshot on one VM. We copied it to the other VM over the 200G link between them and checked its SHA-256 again there. Without such a link, copy the verified folder over the provider's network instead. Then run the [INSTALL.md step 3](../INSTALL.md#step-3-get-the-model) manifest check on the copy. The step 3 helper always fetches and verifies the whole snapshot.

In container-based VMs, the container's view of memory can hide the host page cache that holds the newly downloaded files. The CPU and GPU of the GB10 share memory. So the memory check of the engine can then refuse to start (`Requested size does not fit`, with only about 1 GiB available), even though the host could reclaim that cache. Main `2c64d0a`, part of v0.1.0, counts that cache as free (Linux `MemAvailable`). The cloud cards above predate it and set an explicit budget instead.

## Client time and server time

Client time is the wait that the program which makes the request sees. Server time is the work done inside one chosen part of the engine. Always say which boundary you measured.

- **Client time to first token (TTFT).** It is the first observable stream data minus the time the request was sent. It includes the connection, the request transfer, the queue, the prompt processing and the transfer of the first generated data. The client accepts nonempty answer text, reasoning text or tool-call data. An empty stream header does not count.
- **Total client throughput.** It is the sum of the server's `usage.completion_tokens` divided by the interval from the first request sent to the last `[DONE]` received. It includes the whole cohort, with reading, queueing and response transfer. A sum of individual chat rates does not give this metric.
- **Older RigMark one-chat writing rate.** It is `(completion_tokens - 1) / (last nonempty output event time - first nonempty output event time)`. Answer text or reasoning counts as output. It excludes the initial wait and the tail from the last output event through `[DONE]`.
- **Legacy one-chat scoreboard writing rate.** The default `vade bench` and our earlier post-install check use it. It is `(completion_tokens - 1) * 1000 / (elapsed_ms - ttft_ms)`. TTFT counts the first nonempty answer, reasoning or tool-call output. The denominator ends at stream completion after `[DONE]`, including the transfer overhead of the end of the response. It excludes the initial wait.
- **Answer-content writing rate.** It is `answer_content_tokens_per_s`, which is null today. The client records the arrival of the first answer as `first_content_ms`. But the final usage report does not give the token count before that arrival. A content clock alone cannot give an answer-only rate.
- **Server `prefill_ms`.** It is the engine time spent on the prompt. It excludes the time before the engine admits the request, and it excludes the client transfer.
- **Server `admission_to_first_logits_ms`.** It is the time of the first model scores minus the engine admission time. It includes the work after admission. It can miss a wait before admission.
- **Server decode-step time.** It is the time spent to advance the active decode group. It is a diagnostic, not a client rate. State whether it includes ramp-up, partial groups, drain and host or RDMA waits.

`vade bench` implements the Rust client definitions. See [the Rust benchmark library](../engine/tools/bench/README.md). Count output tokens from the final usage of the server. A streamed event can carry several tokens. So a count of events as tokens is not valid, unless that boundary has been checked for the tested server.

The former Python client is retired from the repository. Its source identity and the writing method of the post-install check were recorded in the retired receipt `night-2026-10-11.json`, which is no longer published ([receipt status](../bench/receipts/README.md)). That receipt also holds two lessons about timing:

- **Two clocks in RigMark.** The pinned RigMark estimator takes its token count from the final usage and its time interval from streamed events. The retained run reported one token per output event, but buffering can still limit how precisely the client times it. Its interval also ends at a different point from that of the post-install check, which runs through `[DONE]`.
- **A wait before admission.** The earlier RigMark run showed a wait after a finished answer, before the engine admitted (accepted) the next request. So a server admission timer could look short while the next client waited much longer. Keep both clocks in follow-up and agent-turn receipts.

## Request settings and cache state

Send each sampling field explicitly. This example asks for plain drafting. Current `vade bench` leaves out the draft field unless you pass `--draft`:

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

Set the output cap explicitly too. Record the seed that you sent, or the effective seed that the server returns. From every response, save the fields for the effective sampling, thinking, draft mode, MTP depth and actual draft use.

FP4-Local uses greedy sampling when a request supplies no sampling field. If a request supplies any sampling field, the fields it does not give come from the checkpoint's `generation_config.json`. What a request asked for is not proof of the settings that took effect. See the [request parser](../engine/crates/serve/src/protocol.rs).

The older RigMark body, in the retired receipt `night-2026-10-11.json`, sent temperature, top-p and seed. Its effective echo filled top-k and neutral penalties from the defaults. Temperature zero made the run greedy. The receipt disclosed this. It did not present it as compliance with the later protocol, in which every field is explicit. That run also used a raised ceiling on streamed output bytes.

Measure cache-off and cache-hit workloads separately.

- **Cache off.** Check that the usage and the prefix-reuse header both show zero reused tokens.
- **Cache hit.** Keep the prime request, the replay request, the restored-token count, the cache budget and the eviction state. Similar text is not proof of a cache hit.

A failure of cache accounting and the equality of answer bytes are separate results. The same retired receipt recorded both for a QSA candidate build. QSA is the model's indexed attention, which reads only the token blocks that a small indexer selects.

"Cold" in the reading rows means an uncached prompt with the weights already loaded. It does not include the model download, the server build or the weight load. Record whether it was the first request after startup or a later uncached request. A fresh server start and another probe on the same server are different kinds of repetition.

## Repetition, noise and ties

To compare a change, use the same Sparks and fresh servers that take turns between the baseline and the candidate. The current rule asks for at least three fresh starts per build. Keep the workload, client, context, output length and sampling fixed. Report medians and full ranges, and name any warmups or discarded runs. Overlapping ranges are a **tie**. A shorter median alone does not show a speed gain.

Record both the requested and the actual output lengths. A natural early stop is useful evidence about that answer, but it cannot count toward a fixed-output timing cell. Report failures and exclusions with the eligible runs. Do not build a missing client rate again from a GPU step timer.

The retired receipt `night-2026-10-11.json` holds two different kinds of evidence:

- **The QSA comparison.** It has eligible fresh-server results for the retained shared-reader workload. But its other prompt cells ended early, so its full timing matrix is incomplete.
- **The clean-install comparison.** It measured against historical results from another pair, with a stated historical noise rule. Its raw cold-prompt ranges did not overlap.

## Columns for eight Sparks

Eight Sparks form one ring ([Eight Sparks](eight-sparks.md)). A run uses an arc of that ring: neighbors in cable order, rank 0 first. Our version 2 cards add the columns below. We call that card format RigMark v2. It is not a release of the public RigMark project. **No value is measured yet.** Every cell in these columns is still to be measured.

- **8 of 8.** Eight Sparks run a server. The arc in rank order is `eight`: spark-1 to spark-8, rank 0 spark-1. The path in this build is the FP4-Local layer chain of eight. No other Sparks are left.
- **4 of 8.** Four Sparks run a server. The arcs in rank order are `quad-a` (spark-1 to spark-4, rank 0 spark-1), then `quad-b` (spark-5 to spark-8, rank 0 spark-5). The path in this build is the FP4-Local tensor split in parts over RDMA (a layer chain of four over TCP), and FP8 four-way TP on a path. The other Sparks run no server.
- **2 × 4 of 8.** Eight Sparks run two servers, `quad-a` and `quad-b`, at the same time. The path in this build is two FP4-Local four-Spark servers (the split in parts over RDMA, chains over TCP). No other Sparks are left.
- **FP8, 8 of 8.** This column has no server and no arc. The CLI refuses it with `FP8 needs 2, 4, or 6 machines` ([cli/src/config.rs](../cli/src/config.rs)). It is not measured.

**What each column measures.** It measures the same cells as the other columns, on the same build, checkpoint and request bodies:

- **One chat.** It measures prose, code and structured output with one chat.
- **Chats at once.** It measures the C1, C4, C8 and C16 total rates.
- **First token.** It measures the cold 8K and 32K first-token times.
- **Context.** It records the largest context that the server reports.

Each cell gives drafting on and off, each from two or more fresh server starts, with the spread between the starts.

**Rules for these columns.**

- **4 of 8.** It gives one value per arc, `quad-a` and `quad-b`, not a mean. The difference between them is the spread from arc to arc.
- **2 × 4 of 8.** It runs the same cell on both servers at the same time. A multi-chat cell gives the sum of the total rates of the two servers, with the value of each server next to it. A one-chat cell gives the value of each server.
- **Token IDs.** For the same request and settings, each column must give the same token IDs as the one-, two- and four-Spark columns. The card states this from the saved token ID hashes, as `vade verify` records them (`reference_ids_sha256`).
- **Shape.** A 4 of 8 arc and four neighbors of a six-Spark ring are both a path of four over three cables. A difference between those two columns is a machine or cable difference, not a shape difference. That difference is still to be measured.
- **Records.** Each card records the arc as the labels in rank order, with the revision, hardware, checkpoint identity, settings and raw results. It names no machine and no address.

## Build, hardware and memory identity

A useful receipt records these items:

- **Checkpoint.** It records the checkpoint revision and the hash scope.
- **Template.** It records the tokenizer and chat-template identity.
- **Source.** It records the source commit and its dirty state (whether there were uncommitted changes).
- **Binaries.** It records the server, PTX (GPU kernel code) and client hashes.
- **Tools.** It records the toolchains and the driver.
- **Topology.** It records the rank count and the topology.
- **Budgets.** It records the context and cache budgets.
- **Workload.** It records the prompt identity, output count, concurrency and arrival pattern.

Record the clocks, the temperature and any competing traffic when you have them. Keep the exact memory bytes for each rank with each rate. Mark missing fields as unmeasured, and never fill them in from an assumption.

Profiled runs explain where the time goes. Publish unprofiled timing separately, because profiling overhead and overlapping spans do not make a performance receipt. Ranks run at the same time, so a sum of rank times does not give wall time.

## What these results do not show

- **No matched comparison.** These rows are not a matched speed comparison with another engine.
- **No task results.** Writing speed and capped concurrent generation say nothing about useful coding tasks completed, harness response time after tools, or agent reliability.
- **Separate checks.** These receipts do not show all of the separate checks: native repeatability (the same token IDs on every run, with the same settings), agreement with a previous build, model quality, and whether a batch keeps running when one of its rows is rejected. The default builds named here still skip that last check, which deliberately feeds a bad row into a batch.
- **Long-memory quality.** Neither a tested context setting nor a longer configured window proves long-memory quality at every length. The October 12 quality evidence in the [historical section](#historical-fp8-measurements) is still partial. The failed context candidate was a historical guard (retired receipt `context-candidate.json`).
- **Drafting speedup.** A server with MTP on can fall back to plain decoding. In the extra run after the post-install check, no drafting happened. So that run supports no claim of a speedup from speculative decoding (retired receipt `night-2026-10-11.json`).
- **Mac builds.** A Mac build check says nothing about GPU arithmetic or speed.

Repeatability requires identical token IDs for the same request and settings across runs and Sparks, including plain and strict drafting. [Architecture](architecture.md#exactness-and-model-quality) explains the full byte-for-byte contract and the quality gates that arithmetic changes must pass.

## Where the receipts are, and older comparisons

Start with the tested setup and timing instructions in [INSTALL.md](../INSTALL.md). The [timing tools guide](../tools/README.md#timing-the-installed-server) describes the client. Measure only on Sparks with no other GPU work running.

The older public FP8 summaries are removed. The [historical section](#historical-fp8-measurements) below still cites some of them as retired receipts. The tables name their source cards and timing records.

Sanitized receipts for the 2026-10-24 tables are in [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md). Receipts for the 2026-10-25 to 2026-10-30 tables are not published here. The 2026-10-24 receipts are cards, READMEs and small JSON receipts. Neutral tokens replace the hostnames, addresses and private paths. The original prompts, replies, logits and the full card JSON receipts are not included. See the [receipt status](../bench/receipts/README.md).

[Historical engine comparisons](../engine/docs/benchmarks.md) cover older models and workloads. Read their conditions before you compare rates. Use the explicit methods and limits that come with each receipt. A historical result says nothing about the performance of the current build.

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
**23.2 to 23.4 tokens per second**. The final RigMark run on build `9bcb8170`
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

The FP8 Flash-Next output default is 65,536 tokens, with context
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
