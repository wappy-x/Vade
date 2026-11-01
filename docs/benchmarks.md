# Benchmarks

This page gives the current benchmark numbers of Vade, and it shows how to measure your own Spark. FP4-Local and FP8 are the two formats of the Qwen3.8 Flash-Next model that Vade runs.

The [benchmark history](history/benchmarks-2026-10.md) keeps every older run.

<!-- bench-docs:start -->

## Current main

These are the facts behind the tables:

- **FP8 rows.** We measured main `acb49bc` on our DGX Sparks on the afternoon of 2026-11-01.
- **FP4-Local rows.** We measured main `dc32b7c` on the morning of 2026-10-31.
- **Cards.** Each server start ran one card with drafting on and one card with drafting off. A card is the result file of one RigMark run.
- **Drafting.** The model's own prediction head guesses a few tokens ahead, and the full model checks each guess. Drafting on means a fixed draft depth: 3 for FP4-Local and 2 for FP8.
- **Default FP8 depth.** A default FP8 server chooses the depth itself. The [morning cards of 2026-11-01](results.md#current-fp8-main-1a07e3a-2026-11-01-morning) give those speeds.
- **Server starts.** The one-Spark rows are the mean of two server starts. The other rows are one start each.

[Results and receipts](results.md) links the receipt of every row. It also gives the machines, the server commands and the request settings.

**One chat (tok/s)**

<div align="center">

| Setup | Prose | Code | Structured |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark · on | 42.530 | 81.138 | 100.808 |
| FP4-Local · 1 Spark · off | 27.911 | 27.761 | 28.012 |
| FP4-Local · 2 Sparks · on | 54.878 | 102.314 | 130.800 |
| FP4-Local · 2 Sparks · off | 36.047 | 35.840 | 36.142 |
| FP4-Local · 4 Sparks · on | 63.036 | 113.025 | 145.238 |
| FP4-Local · 4 Sparks · off | 42.084 | 41.486 | 42.072 |
| FP8 · 2 Sparks · on | 39.504 | 55.734 | 57.265 |
| FP8 · 2 Sparks · off | 28.374 | 28.205 | 28.454 |
| FP8 · 4 Sparks · on | 52.621 | 74.237 | 75.489 |
| FP8 · 4 Sparks · off | 36.916 | 37.459 | 37.684 |

</div>

**Chats at once (total tok/s)**

<div align="center">

| Setup | C1 | C4 | C8 |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark · on | 65.688 | 144.681 | 178.981 |
| FP4-Local · 1 Spark · off | 27.115 | 85.004 | 138.609 |
| FP4-Local · 2 Sparks · on | 79.632 | 166.796 | 232.198 |
| FP4-Local · 2 Sparks · off | 34.375 | 88.178 | 134.414 |
| FP4-Local · 4 Sparks · on | 86.962 | 177.581 | 245.145 |
| FP4-Local · 4 Sparks · off | 38.080 | 95.113 | 143.015 |
| FP8 · 2 Sparks · on | 48.842 | 72.889 | 106.440 |
| FP8 · 2 Sparks · off | 27.349 | 73.767 | 106.392 |
| FP8 · 4 Sparks · on | 63.668 | 95.638 | 136.174 |
| FP8 · 4 Sparks · off | 36.223 | 91.895 | 136.567 |

</div>

**Path and cold 8K prompt**

<div align="center">

| Setup | Path | Drafting-on card (s) | Drafting-off card (s) |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | local | 3.769 | 3.778 |
| FP4-Local · 2 Sparks | tensor split | 2.153 | 2.131 |
| FP4-Local · 4 Sparks | tensor split in parts | 1.924 | 1.916 |
| FP8 · 2 Sparks | tensor parallel | 5.899 | 5.891 |
| FP8 · 4 Sparks | tensor parallel | 6.866 | 6.801 |

</div>

How to read the tables:

- **Rates.** Rates are in completion tokens per second.
- **Prose, code and structured.** Each uses one chat.
- **C1, C4 and C8.** These give the total rates for one, four and eight chats at once.
- **Cold 8K.** This is the time to the first token after an uncached 8192-token prompt. Both cards send it with drafting off.
- **More numbers.** For FP8, [results and receipts](results.md) gives the 32K chat and the token ID checks. For FP4-Local, the [2026-10-31 section](#current-2026-10-31-morning-main-dc32b7c-home-dgx-sparks) gives them.

We write these tables by hand, because the one-Spark rows average two server starts. The `bench-docs` tool cannot make these tables. It takes one card per setup, only from the newest main that has cards. It does not average server starts ([bench_docs.rs](../tools/src/bench_docs.rs)).

This command writes such a table into this block and into the README block. It reads a folder of RigMark card files. Run it from the repository root:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- bench-docs '<cards-dir>' .
```

The full card files are private. Their sanitized extracts are in [bench/receipts/2026-11-01](../bench/receipts/2026-11-01/README.md) for FP8 and in [bench/receipts/2026-10-31](../bench/receipts/2026-10-31/README.md) for FP4-Local.

Each section below names its date and main. Its results apply only to that build.

<!-- bench-docs:end -->

## How the current numbers were measured

The recipe of the [current table](#current-main) (main `acb49bc` for FP8, main `dc32b7c` for FP4-Local) is on [results and receipts](results.md), with a sanitized receipt for each card.

A revision such as main `acb49bc` belongs to the private development history ([glossary](GLOSSARY.md#ho), "Main `abc1234`").

### Models and builds

These are the two checkpoints. The identity is the SHA256 that the server reports.

- **fp4-local.** The checkpoint is `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`. Its identity is `d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`.
- **fp8.** The checkpoint is `Qwen/Qwen3.8-Flash-Next-FP8` at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`. Its identity is `8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03`.

These are the builds of the current numbers:

- **Main `acb49bc`, 2026-11-01 afternoon.** The server build is flash-serve `3c3661e2` with PTX `f5f78a91`. It made the current FP8 rows ([receipts](../bench/receipts/2026-11-01/README.md)).
- **Main `1a07e3a`, 2026-11-01 morning.** The server build is flash-serve `c5edc3e1` with PTX `f5f78a91`. It made the newest FP8 rows with the depth chosen by the server ([receipts](../bench/receipts/2026-11-01/README.md)).
- **Main `dc32b7c`, 2026-10-31.** The server build is flash-serve `55bdc7a8` with PTX `84c09350`. It made the current FP4-Local rows ([receipts](../bench/receipts/2026-10-31/README.md)).

### Requests

Every request used greedy decoding: temperature 0, top-p 1, top-k 0, repetition penalty 1, and presence and frequency penalties 0. Chat requests also set seed 20260905, thinking off and `cache` false.

Drafting-on requests leave out the `draft` field, so the server drafts at the depth of its start: depth 3 on FP4-Local and depth 2 on FP8. Drafting-off requests send `"draft":"plain"`.

A RigMark card sends 66 requests. It uses the prompt set of RigMark commit `52c29c3fc1d741b9aef8895b23372aafa5dc6161` (prompts version 1.0.0). The prompt and reply lengths below are from the two-Spark FP4-Local card of 2026-10-30.

- **Prose, code, structured.** Each is one chat, with prompts of 117 to 145 tokens. The reply is up to 8192 tokens, to the natural stop (551 to 1,794 tokens). Each runs 5 times. The table gives the median of the 5 writing rates.
- **C1, C2, C4, C8.** These are 1, 2, 4 or 8 chats at once, with code prompts of 140 to 147 tokens. Each chat gets exactly 256 tokens. Each runs 3 rounds. The table gives the median of the 3 round rates.
- **Cold 8K.** The prompt has 8,192 tokens and goes out with drafting off and a new cache salt for each pair. The reply is 8 tokens. The card sends 3 cold and replay pairs. The table gives the median time to the first token of the 3 cold requests.

When a setup has two or three server starts, each cell is the mean of the start medians.

### What you can run today

- **Start one or two Sparks as in the cards: available.** Run `vade up --sparks 1 --context 32768 --output 8192 --mtp 3` or `vade up --sparks 2 --context 32768 --output 8192 --mtp 3`.
- **Start four Sparks or FP8: available, but the cards used a lab launcher, not these commands.** Run `vade up --sparks 4 --context 32768 --output 8192 --mtp 3` or `vade up --fp8 --sparks 2 --context 32768 --output 16384 --mtp 2` (or `--sparks 4`).
- **Check token IDs, drafting on against off: available.** Run `vade verify -n 3 --prompt 8k --max-tokens 256`. It checks one server, not one card against another.
- **Run the 66-request card: available.** From a RigMark checkout at `52c29c3`, run `vade bench --card --prompts rigmark/prompts.json --receipt bench/receipts/2026-10-31/<file>.json`. It sends the card's requests and prints `ids 66/66 identical` when the token IDs match the receipt. On 2026-10-31 it matched the one-Spark drafting-on and drafting-off receipts. We have not run it against the other receipts ([details](results.md#what-a-stranger-can-and-cannot-re-run)).
- **Time your own server: available.** Run `vade bench --concurrency 1,2,4,8 --max-tokens 256 --repeats 3`. It uses other prompts, so its numbers do not compare with the cards.
- **Run the 32K chat or the context run: not available.** No `vade` command times a second question with and without the cache, and the prompts are not in the repository.

## All results, newest first

Every speed that we have published for Vade is on one of these pages or sections.

<div align="center">

| Section | What it is for |
| :---: | :---: |
| [2026-11-01 afternoon FP8 cards](results.md#current-fp8-main-acb49bc-2026-11-01-afternoon) | Two and four Sparks at main `acb49bc`, FP8, drafting on and off, with public receipts |
| [2026-11-01 morning FP8 cards](results.md#current-fp8-main-1a07e3a-2026-11-01-morning) | Two and four Sparks at main `1a07e3a`, FP8, with the depth chosen by the server, with public receipts |
| [2026-10-31 morning cards](#current-2026-10-31-morning-main-dc32b7c-home-dgx-sparks) | One, two and four Sparks at main `dc32b7c`, FP4-Local and FP8, drafting on and off, with public receipts |
| [Benchmark history](history/benchmarks-2026-10.md) | Every older run, from 2026-10-30 back, with the older FP8 measurements |
| [How to reproduce a run](#reproduce-a-measurement-in-detail) | RigMark or `vade bench` |

</div>

Every result names the build and settings that we measured, and its timings hold for that build only.

<a id="a-long-chat-and-its-second-question"></a>
<a id="card-at-main-250fb49-after-the-change-to-memory-reservation"></a>
<a id="confirm-run-main-285fa8a"></a>
<a id="current-2026-10-26-main-262b0c3-home-dgx-sparks"></a>
<a id="current-2026-10-27-final-main-09e1fc4-home-dgx-sparks"></a>
<a id="current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks"></a>
<a id="current-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks"></a>
<a id="fp8-on-two-and-four-sparks-different-token-ids"></a>
<a id="previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks"></a>
<a id="reading-these-cards"></a>
<a id="the-prefix-cache-on-two-sparks"></a>

**Older links.** Some older pages link to a section that moved. Those links land here. The [benchmark history](history/benchmarks-2026-10.md) holds the section, under the same name.

<a id="current-2026-10-31-morning-main-dc32b7c-home-dgx-sparks"></a>
## 2026-10-31 morning (FP4-Local current, FP8 replaced)

On the morning of 2026-10-31 we measured main `dc32b7c` on one, two and four Sparks, with FP4-Local and with FP8. Every server ran the same `flash-serve` binary (SHA256 `55bdc7a8...`, PTX `84c09350`), with greedy decoding, thinking off and no `VADE_` setting.

Each server start ran one card with drafting on and one with drafting off. The 32K chat ran on the two-Spark and four-Spark starts. On one Spark, the card ran on two starts, and the 32K chat on two more starts at context 65536.

[Results and receipts](results.md) gives the table, the receipt of each card, the server commands and what you can run again. The FP4-Local speeds are in the [current table](#current-main). The [2026-11-01 FP8 cards](results.md#current-fp8-main-acb49bc-2026-11-01-afternoon) replace the FP8 speeds of this section.

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
- **FP8, two against four Sparks.** The cards matched in 7 of 66 ([FP8 on two and four Sparks](history/benchmarks-2026-10.md#fp8-on-two-and-four-sparks-different-token-ids)). A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order, so four Sparks now give the same token IDs as two ([changelog](../CHANGELOG.md)). These cards predate that change.

**Limits.**

- **Starts.** There was one server start per setup, except on one Spark.
- **Launcher.** The four-Spark and FP8 starts used a private lab launcher.
- **Clients.** The 32K chat client is private. The card client is public: `vade bench --card` ([details](results.md#what-a-stranger-can-and-cannot-re-run)).

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
- **`--concurrency 1,2,4,8`.** It adds C8 to the RigMark default of C1, C2 and C4. Every 2026-10-24 RigMark card ran with it, including the two-VM cloud card 1006 (C8 **70.0 tok/s**).
- **Everything else.** The rest is a RigMark default: 5 runs each of prose, code and structured output with the 8192-token cap, 3 cold and replay prefill pairs, and 3 rounds of code at each concurrency with a 256-token cap. RigMark fixes temperature 0, top-p 1 and seed 20260905.
- **Older checkouts.** Older RigMark checkouts call the `bench` subcommand `run`.

The published cards send a longer body. To compare with a card, send the identical body, because RigMark compares receipts only when they have the same request body:

```json
{"cache": false, "chat_template_kwargs": {"enable_thinking": false}, "draft": "plain",
 "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "top_k": 0}
```

`"draft": "plain"` makes a drafting-off card. The drafting-on cards leave the field out and get the server default. `cache`, the neutral penalties and `top_k` spell out the defaults. Cards with thinking on replace `"enable_thinking": false` with `"enable_thinking": true, "reasoning_effort": "low"` inside `chat_template_kwargs`.

**What "DO NOT HEADLINE" means.** RigMark checks each of the 15 single-stream samples. Code and prose need visible answer text and a normal stop. Structured output needs the requested JSON values. If any sample fails, the card header marks the result `DO NOT HEADLINE`. Then do not quote its rates as a result.

On 2026-10-24, a RigMark run with its default settings against a two-VM pair scored **6/15**. Code (0/5) and prose (1/5) hit the cap while they were still reasoning, and structured output passed 5/5. With thinking off through `--extra-body`, the cards pass **15/15**. These checks, which RigMark calls gates, confirm that the answers completed. They are not a check of exactness (the same request with the same settings gives the same token IDs) or of quality.

**What a good card looks like** with thinking off:

- **Header.** The header says `●  15/15 BASIC OUTPUT GATES PASSED`.
- **SINGLE STREAM.** PROSE, CODE and STRUCTURED* each show `✓ 5/5`.
- **PREFILL.** One 8K row shows the cold tok/s, the cold TTFT and the replay TTFT.
- **CAPPED CONCURRENT GENERATION.** The section shows `C1 | C2 | C4 | C8`, then `C8: 0/24 normal stops; 24/24 with visible output`. Every capped stream stops at 256 tokens, so zero normal stops is expected. With thinking on, those 256 tokens are often all reasoning, and the visible-output count drops toward 0.
- **SETTINGS.** The section shows `SUITE: CUSTOM SETTINGS`, `Changed: --prefill-depths=8192`, `Changed: --concurrency=1,2,4,8`, and `REQUEST:` with your body. Custom settings are expected and do not mean "DO NOT HEADLINE".
- **Rates.** The rates should be close to the [matching row](history/benchmarks-2026-10.md#current-results-2026-10-24) for the same Spark count and drafting mode, with thinking off. For thinking on, see the [receipt folder](../bench/receipts/2026-10-24/README.md).

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

In container-based VMs, the container's view of memory can hide the host page cache that holds the newly downloaded files. The CPU and GPU of the GB10 share memory. So the memory check of the engine can then refuse to start (`Requested size does not fit`, with only about 1 GiB available), even though the host could reclaim that cache. Main `2c64d0a`, part of v0.1.0, counts that cache as free (Linux `MemAvailable`). The cloud cards predate it and set an explicit budget instead.

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
- **Long-memory quality.** Neither a tested context setting nor a longer configured window proves long-memory quality at every length. The October 12 quality evidence in the [historical section](history/benchmarks-2026-10.md#historical-fp8-measurements) is still partial. The failed context candidate was a historical guard (retired receipt `context-candidate.json`).
- **Drafting speedup.** A server with MTP on can fall back to plain decoding. In the extra run after the post-install check, no drafting happened. So that run supports no claim of a speedup from speculative decoding (retired receipt `night-2026-10-11.json`).
- **Mac builds.** A Mac build check says nothing about GPU arithmetic or speed.

Repeatability requires identical token IDs for the same request and settings across runs and Sparks, including plain and strict drafting. [Architecture](architecture.md#exactness-and-model-quality) explains the full byte-for-byte contract and the quality gates that arithmetic changes must pass.

## Where the receipts are, and older comparisons

Start with the tested setup and timing instructions in [INSTALL.md](../INSTALL.md). The [timing tools guide](../tools/README.md#timing-the-installed-server) describes the client. Measure only on Sparks with no other GPU work running.

The older public FP8 summaries are removed. The [historical section](history/benchmarks-2026-10.md#historical-fp8-measurements) still cites some of them as retired receipts. The tables name their source cards and timing records.

Sanitized receipts for the 2026-10-24 tables are in [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md). Receipts for the 2026-10-25 to 2026-10-30 tables are not published here. The 2026-10-24 receipts are cards, READMEs and small JSON receipts. Neutral tokens replace the hostnames, addresses and private paths. The original prompts, replies, logits and the full card JSON receipts are not included. See the [receipt status](../bench/receipts/README.md).

[Historical engine comparisons](../engine/docs/benchmarks.md) cover older models and workloads. Read their conditions before you compare rates. Use the explicit methods and limits that come with each receipt. A historical result says nothing about the performance of the current build.
