# Benchmarks

The release path is FP4-Local on one DGX Spark. The current results below name
exact measured builds and recipes; later changes do not inherit their timings.
The older FP8 measurements are retained as history. Sanitized copies of the
current source folders are in [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md);
see the [receipt status](../bench/receipts/README.md) for what is available publicly.

## Current (2026-10-24)

Builds: the home tables use the **release-build cards** from the afternoon,
engine `73f7a9d` (release build of main `73f7a9d`). The v0.1.0 engine differs
from it only in how startup admission counts free memory (main `2c64d0a`); no
cards exist for that build. The cloud tables and the timing runs are from main `4681d8b` earlier the same day;
no release-build cloud cards exist yet.

Recipe: `nvidia/Qwen3.8-Flash-Next-FP4-Local`, context **32768**, greedy,
cache off, thinking as stated in each row. A **RigMark card** is the recorded
result of one run of the RigMark benchmark suite: single-stream prose, code and
structured writing, cold 8K prompt reading, and capped concurrent streams. A
card uses **drafting OFF** unless its folder says `drafting-on`; OFF requests
use `"draft":"plain"`. **C1, C2, C4 and C8 mean 1, 2, 4 and 8 concurrent
streams.** TTFT is client time to first token. Tok/s means completion tokens per wall
second; concurrent rates cover the whole cohort. The cards label their
single-stream decode rates as estimates, including streamed reasoning. The
separate timing runs use total request wall time, so their rates are not the
same estimator as the cards' decode column.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks-2026-10-24-dark.svg">
  <img src="assets/benchmarks-2026-10-24-light.svg" width="100%" alt="Grouped bar chart of decode speed in tokens per second. On one home DGX Spark, release build 73f7a9d, drafting ON raises single-stream code from 27.7 to 76.7, prose from 27.7 to 42.0, and the 8-stream aggregate from 135.9 to 172.6. Cloud GB10 cards on main 4681d8b with drafting OFF: one VM code 24.7, prose 23.4, 8 streams 43.0; two VMs code 25.4, prose 23.2, 8 streams 70.0. Cloud drafting ON values are timing runs, not cards: one VM code 64.0 and prose 33.6, two VMs code 56.5 and prose 32.0.">
</picture>

Every label in the figure is the value printed in the receipt named under it:
the home panel draws the release-build cards 1444 (OFF) and 1438 (ON) from the
one-Spark table below; the cloud panels draw the `4681d8b` cards 1046 and 1006
and timing runs 1042 and 1114. The
cloud drafting ON bars are textured because they come from timing runs, not
cards, and use the timing runs' estimator described above. Reproduce the figure: `python3 -I tools/benchmarks-chart.py`, which reads
the receipts and rewrites both SVG files in [docs/assets](assets/README.md).

Each row quotes its source folder and links to its sanitized copy in the
repository. Card numbers come from that folder's `*.card.txt`; recipe details
come from the card and its `README.txt`. The folders labelled **timing run, not
a card** supply README results, with their per-request JSON alongside.

### Home DGX Spark

All six home rows are release-build (`73f7a9d`) receipts. The one-Spark OFF and
ON cards were measured on **different home Sparks** running the same build, not
by alternating modes on the same machine. Their clocks differed too. Thinking
ON means ON/low for these cards. Single-stream cells used **5 runs** with an
**8192-token cap**; cold prefill used **3 pairs**; C1/C2/C4/C8 used **3 rounds**
with a **256-token cap**. Structured output is a predictable JSON workload, not
a measure of general agent performance.

| Sparks | Drafting | Thinking | Prose tok/s | Code tok/s | Structured tok/s | Cold 8K TTFT (s) | Cold prefill tok/s | C1 | C2 | C4 | C8 | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| One | OFF | OFF | 27.7 | 27.7 | 28.0 | 3.87 | 2,118 | 27.1 | 46.9 | 85.0 | 135.9 | [`2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off/) |
| One | ON | OFF | 42.0 | 76.7 | 88.9 | 3.85 | 2,129 | 61.0 | 91.2 | 144.4 | 172.6 | [`2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off/) |
| One | OFF | ON/low | 27.6 | 27.7 | 28.0 | 3.82 | 2,144 | 26.7 | 46.3 | 79.9 | 127.2 | [`2026-10-24_1458_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-on`](../bench/receipts/2026-10-24/2026-10-24_1458_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-on/) |
| One | ON | ON/low | 43.2 | 73.8 | 86.2 | 4.08 | 2,008 | 52.1 | 84.7 | 128.6 | 153.4 | [`2026-10-24_1445_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-on`](../bench/receipts/2026-10-24/2026-10-24_1445_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-on/) |

The home pair uses the layer pipeline: each Spark holds a contiguous range of
the model's layers and passes intermediate state to the next. Its card files are custom runner
summaries: **3** prose samples with a **1024-token cap**, **6** cold prefill
samples, and **8 rounds** per cohort with a **256-token cap**, after a warmup.
These prose rows are not directly comparable with the one-Spark RigMark
workload. Both rows come from **one server start**, OFF block first. C16
offered **16** streams; this start admitted **9** active streams and retried
excess requests, including their wait in the cohort wall time. Token IDs were
identical ON versus OFF in **201/201** measured pairs.

| Sparks | Drafting | Thinking | Prose tok/s | Cold 8K TTFT (s) | C8 tok/s | C16 tok/s | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Two | OFF | OFF | 27.705 | 2.776 | 133.877 | 135.197 | [`2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off/) |
| Two | ON | OFF | 38.386 | 2.853 | 162.558 | 147.049 | [`2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off/) |

**Release build versus the morning build.** The one-Spark numbers are
unchanged between the two builds: every single-stream and C8 cell is within
**1 %** of the morning card for the same configuration. The largest
differences are C4 with drafting OFF and thinking ON (**81.9** to **79.9**) and
cold 8K TTFT with drafting ON and thinking ON (**3.81** to **4.08 s**). The
morning and release-build one-Spark cards ran on different home Sparks. The two-Spark
drafting OFF row is also unchanged (C8 **134.561** to **133.877**), while
two-Spark **C8 with drafting ON went from 134.421 to 162.558 tok/s** (C16
**134.241** to **147.049**); the release build includes the two-Spark
drafting fix from main `407b6ae`.

Previous build (main `4681d8b`, morning): one Spark
[0656 OFF/OFF](../bench/receipts/2026-10-24/2026-10-24_0656_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-off/), [0650 ON/OFF](../bench/receipts/2026-10-24/2026-10-24_0650_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-off/),
[0710 OFF/ON](../bench/receipts/2026-10-24/2026-10-24_0710_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-on/), [0657 ON/ON](../bench/receipts/2026-10-24/2026-10-24_0657_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-on/); home pair
[0720 OFF](../bench/receipts/2026-10-24/2026-10-24_0720_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/), [0711 ON](../bench/receipts/2026-10-24/2026-10-24_0711_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-on-thinking-off/).

### Cloud

These were **rented GB10 VMs at a cloud provider**, with **116 GB RAM each**;
see [Rented cloud GB10 VMs](#rented-cloud-gb10-vms) for setup expectations.
The pair used drivers **580.178.04** and **580.159.03**; the one-VM card used
**580.159.03**. Both cards used the same single-stream and capped-concurrency
recipe as the home one-Spark cards above, with drafting OFF and thinking OFF.
The pair used the layer pipeline over a 200G QSFP RoCE link between the two
VMs (RDMA); the one-VM card used local execution. Both cloud cards set an
explicit 100 GiB engine memory budget (`SEQPAR_MEMORY_BYTES`) because the
engine's derived memory ceiling came out slightly short inside the VMs'
containers; see [Rented cloud GB10 VMs](#rented-cloud-gb10-vms).

| VMs | Drafting | Thinking | Prose tok/s | Code tok/s | Structured tok/s | Cold 8K TTFT (s) | Cold prefill tok/s | C1 | C2 | C4 | C8 | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two | OFF | OFF | 23.2 | 25.4 | 26.5 | 3.03 | 2,701 | 21.7 | 40.0 | 71.2 | 70.0 | [`2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/) |
| One | OFF | OFF | 23.4 | 24.7 | 26.0 | 3.89 | 2,105 | 24.7 | 42.1 | 42.1 | 43.0 | [`2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off`](../bench/receipts/2026-10-24/2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off/) |

The following are **timing runs, not cards**. Each used an ON/OFF/ON sequence
on a drafting-enabled server: `"draft":"strict"` for ON and
`"draft":"plain"` for OFF. The code medians used **5** OFF and **10** ON
samples; prose used **1** OFF and **2** ON samples. All three timing runs used
thinking OFF. The third run repeated the two-VM sequence on build `8c9411f`,
whose served engine is identical to main `407b6ae`, the fix described under
known limits below. The one-VM README says the returned MTP-depth field was zero even
for ON requests, so it identifies modes by the request's draft field.

| VMs | Kind | Thinking | OFF code tok/s | ON code tok/s | OFF prose tok/s | ON prose tok/s | C8 result | Source folder |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| One | timing run, not a card | OFF | 25.8 | 64.0 | 24.5 | 33.6 | Not measured | [`2026-10-24_1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card`](../bench/receipts/2026-10-24/2026-10-24_1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card/) |
| Two | timing run, not a card | OFF | 26.2 | 56.5 | 25.2 | 32.0 | First ON round 85.4 tok/s; later ON rounds failed | [`2026-10-24_1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card`](../bench/receipts/2026-10-24/2026-10-24_1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card/) |
| Two, build `8c9411f` | timing run, not a card | OFF | 26.3 | 57.0 | 25.4 | 31.6 (median of 29.8 and 33.4) | ON rounds 81.3 / 92.0 / 92.2 tok/s, OFF rounds 78.4 / 80.9 / 78.9 tok/s, 0 errors | [`2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card`](../bench/receipts/2026-10-24/2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card/) |

### What this means

Drafting ON is the main lever in these measurements: home one-Spark code goes
from **27.7** to **76.7 tok/s** and C8 from **135.9** to **172.6 tok/s**; the
cloud one-VM timing run goes from **25.8** to **64.0 tok/s** for code. Adding a
second Spark does not substantially raise single-stream speed in the OFF
cards. It improves prefill and can raise batched throughput: cloud C8 goes
from **43.0** to **70.0 tok/s**. Home pair C8 with drafting OFF is effectively
unchanged from one Spark (**133.877** versus **135.9 tok/s**); with drafting ON
the release-build pair reaches **162.558 tok/s**, below one Spark's
**172.6 tok/s**. Rented VMs are broadly similar to home on
single-stream speed and prefill, but their batched decode levels off: the
one-VM C2/C4/C8 rates are **42.1 / 42.1 / 43.0 tok/s**. The pair reaches
**71.2 / 70.0 tok/s** at C4/C8; its C2 rate is **40.0 tok/s**. These are
observations across the named runs, not a controlled same-machine comparison.

### Known limits

- On the **116 GB** VMs, the first two-VM timing README (build `4681d8b`)
  records an engine memory ceiling of **88.4 GiB**. After a successful C8
  drafting round followed by a plain round, the next drafting rounds failed:
  **8/8** requests in each returned `generation_error`, attributed to
  allocation headroom after pooled storage grew. Main `407b6ae` (pipeline C8
  policy and worker decode-pool reclamation) addressed this: the
  12:59 rerun of the same sequence on that engine completed **3/3** drafting
  rounds after plain rounds with **0** errors (**81.3 / 92.0 / 92.2 tok/s**),
  with pooled storage at **94.6 GB** under the rank budget. The 116 GB VMs
  still have less headroom than **128 GB** home Sparks.
- On the morning card build `4681d8b`, home two-Spark C8 drafting ON was
  effectively equal to OFF: **134.421** versus **134.561 tok/s**. The fix in
  main `407b6ae` raised it on the same home pair from **133.871** to
  **165.560 tok/s** in an alternating before/after comparison (baseline, fix,
  fix, baseline). That comparison is not a card and its own receipt is not
  published; the 12:59 README quotes the same rates and rounds the gain to
  +23.7 %. The release-build cards above confirm the change: **162.558** ON
  versus **133.877 tok/s** OFF.
- The home one-Spark ON cards report **15/15** basic output checks passed
  (the cards call them gates), while their READMEs mark the runs `PARTIAL`.
  Those checks do not establish exactness or model quality. Replay TTFT is an immediate
  repeat, not proof of a cache hit.

## Reproduce the current measurements

Use the [FP4-Local serving guide](../cli/docs/fp4-local.md) and keep the
measured context and output limits explicit. `vade launch` on the Spark and
`vade up` from a controller both start at context 32768 and output 8192;
`vade launch` lowers a default context only when the engine refuses it for
memory; a controller never lowers it, so check `vade status --json` before
comparing runs. After `vade launch` or `vade up`, `vade bench` works
without flags: keep the same `VADE_HOME` / `VADE_CONFIG` to
discover the endpoint, read the model ID from `/health`, and measure the
server's default drafting policy with a bundled offline prompt. Receipts go
to `./vade-bench.json` and its sibling text table.

For a comparison using retained prompts and plain decoding, use `--draft plain`; it works on a
drafting-enabled server without a restart. Set `VADE_API_URL` to the running
server's HTTP base URL and retain the same prompt file between runs:

```sh
vade bench --url "$VADE_API_URL" --model qwen3.8-flash-next-fp4-local --draft plain \
  --prompts ./prompts.txt --concurrency 1,2,4,8 \
  --max-tokens 256 --repeats 3 --output ./vade-bench.json
```

This records the Rust client's fixed-output cohort timings; it does not
recreate the RigMark corpus or turn a timing run into a card. The cards' ON
recipes omit the draft field to use the server default; the timing runs use
`"draft":"strict"`. To request that mode with the current client, pass
`--draft strict`. Current bench requests keep thinking off. Keep the thinking
setting, estimator and source folder with any quoted result. See
[Run vade bench](#run-vade-bench) for all options.

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
defaults, most code and prose samples spend the whole cap reasoning, stop with
`finish_reason` `length` and no visible answer, and fail RigMark's output
checks. Set the field yourself with RigMark's `--extra-body`, which merges JSON
into every chat request. The server reads, in this order
([protocol.rs](../engine/crates/serve/src/protocol.rs)):

1. `chat_template_kwargs.enable_thinking`, or a top-level `enable_thinking`;
2. `reasoning_effort`: `none` or `minimal` turns thinking off; `low`,
   `medium`, `high` or `xhigh` turns it on at that effort;
3. the server default, ON.

The invocation we use on one Spark, from a RigMark checkout:

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
- `--prefill-depths 8192`: RigMark's default also measures 32768 and 65536, and
  it refuses any depth where depth + 8 exceeds `context_limit`. The cards
  measure 8K only.
- `--concurrency 1,2,4,8` adds C8 to RigMark's default C1/C2/C4. The two-Spark
  run reported above kept the default C1/C2/C4.
- The rest are RigMark defaults: 5 runs each of prose, code and structured
  output with the 8192-token cap, 3 cold/replay prefill pairs, and 3 rounds of
  code at each concurrency with a 256-token cap. Temperature 0, top-p 1 and seed
  20260905 are fixed by RigMark.
- Older RigMark checkouts call the `bench` subcommand `run`.

The published cards send a longer body. To compare with a card, send the
identical body, because RigMark compares only receipts with the same request
body:

```json
{"cache": false, "chat_template_kwargs": {"enable_thinking": false}, "draft": "plain",
 "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "top_k": 0}
```

`"draft": "plain"` makes a drafting-OFF card; the drafting-ON cards omit the
field and get the server default. `cache` and the neutral penalties and
`top_k` make the defaults explicit. Thinking-ON cards replace
`"enable_thinking": false` with `"enable_thinking": true, "reasoning_effort": "low"`
inside `chat_template_kwargs`.

**What "DO NOT HEADLINE" means.** RigMark checks each of the 15 single-stream
samples: code and prose need visible answer text and a normal stop; structured
output needs the requested JSON values. If any sample fails, the card header
reads `▲ n/15 BASIC OUTPUT GATES PASSED — DO NOT HEADLINE`, and its rates
should not be quoted as a result. On 2026-10-24 a RigMark run with its default
settings against a two-VM pair scored **6/15**: code 0/5 and prose 1/5 hit
the cap while still reasoning, and structured passed 5/5. With thinking off
through `--extra-body`, the cards above pass **15/15**. These checks confirm
that answers completed; they are not an exactness or quality gate.

**Expected card shape** with thinking off:

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
- Rates should be close to the [matching row](#current-2026-10-24)
  for the same Spark count, drafting mode and thinking setting.

## Rented cloud GB10 VMs

Allow time for the model download before benchmarking rented VMs. The verified
FP4-Local snapshot is about **102 GB**. On 2026-10-24, rented GB10 VMs downloaded
from Hugging Face at **8–22 MB/s each**: about 80 minutes for one VM at the top
of that range and over three hours at the bottom. For the cloud cards, the
snapshot was downloaded and verified on one VM, then copied to the other over
the 200G link between the two VMs and its SHA-256 checked again there. Without
such a link, copy the verified folder over the provider network instead and run
the [INSTALL.md step 3](../INSTALL.md#step-3-get-the-model) manifest check on
the copy; the step 3 helper always fetches and verifies the whole snapshot. In container-based VMs, the
host page cache that holds the freshly downloaded files can be hidden from
the container's memory view. GB10 memory is shared by the CPU and GPU, so the
engine's memory check can then refuse to start
(`Requested size does not fit`, with only about 1 GiB available) even though
the host could reclaim that cache. Main `2c64d0a`, part of v0.1.0, counts that
cache as free (Linux `MemAvailable`); the cloud cards above predate it and set
an explicit budget instead.

## Historical FP8 measurements

The following results describe older builds and settings. The private night
receipts have been removed from the public tree; historical receipt names
below identify the retired evidence, not downloadable current receipts.
See the [receipt status](../bench/receipts/README.md). These results are not
FP4-Local performance claims.

### October 12 overview (two-Spark Flash-Next)

At the October 12 product defaults, Vade started answering an uncached 8k
prompt in about **7.8 seconds** and wrote a single answer at about
**23.2–23.4 tokens per second**. A token is a small piece of text, often part
of a word. The final RigMark at `9bcb8170` passed every reporting line with
thinking off and with thinking on/low, on two DGX Sparks using the official
Qwen3.8-Flash-Next FP8 checkpoint. A reporting pass means the requests completed
and passed the suite's checks; it does not mean every workload became faster.
Retired receipt `defaults-2026-10-12.json` ([receipt status](../bench/receipts/README.md))

### October 12 product defaults

The measured settings were context **131,072**, output cap **8,192**, MTP
**off**, automatic session pool and retained cache, one official-precision
weight share per rank, and a two-rank RDMA ring. Native startup funded one
full-context session. A near-full cached continuation restored **130,560
tokens**, read 256 new tokens and generated 256 more, filling the 131,072-token
window; its answer bytes and token IDs matched an uncached continuation.
These engine checks used a separate measurement launcher and do not qualify
the current CLI lifecycle.
Retired receipt `defaults-2026-10-12.json` ([receipt status](../bench/receipts/README.md))

#### Final RigMark

All **108 requests**, **184 gate cells** and **24 reporting rows** passed.
There was one fresh server per thinking mode: five answers each for prose,
code and structured output; three cold/replay pairs at each of 8k, 32k and 64k;
and three cohorts each at one, two and four chats. Every sample is included.
The table shows thinking-off medians and full ranges. Its memory column gives
whole-suite allocation peaks for ranks 0/1, not the extra memory for that row;
each rank held **91,709,826,224 weight bytes**.

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

Writing excludes the first-token wait and ends at the last nonempty output
event; concurrency includes reading, queueing and transfer through completion.
The retired receipt `defaults-2026-10-12.json` ([receipt status](../bench/receipts/README.md)) recorded every row,
including warm cache replays and thinking on/low, whose whole-suite peak was
113,978,286,084 bytes on each rank. The automatic pool was 24,264,576,444 bytes
per rank for thinking off and 24,071,770,351 for thinking on/low. Retention was
18,125,450,072 bytes per rank in both launches. Capacity varies with available
memory; 23 minimum-size session slots meant **one** full-context slot.

RigMark stopped at 64k prefill and four concurrent chats. Its 131,072 context
setting is not a 128k prefill timing row or a full-length multi-chat test.
Historical comparisons with the scalar night build and MTP-2 preview changed
builds and settings and did not use fresh alternating A/B starts. The roughly
6% lower plain writing rate versus that night build includes both arithmetic
and default changes; it is separate from the cost isolated below.

RigMark chat requests sent every greedy sampling field explicitly. The pinned
upstream **prefill** requests sent temperature zero but omitted top-p, top-k,
penalties and seed, to preserve the upstream method. Every response
was checked: effective prefill top-p was 0.95, top-k 20 and penalties 1/0/0,
with per-request seeds. This exception is preserved in the receipt; new timing
requests must follow the explicit-fields rule below.

#### Accepted pool and context cost

A separate comparison used three fresh alternating starts per build, cache
and thinking off, MTP off, explicit greedy sampling and 256 output tokens per
request. Baseline A was `41fa8cf8`; candidate B was `9bcb8170`. One-chat A used
32,768 context and B used the 131,072 default. The eight- and sixteen-chat
cells used **32,768 on both builds**. Rates below are medians [min–max]; peaks
are the maximum lifetime device allocation through that cell on either build,
equal on both ranks. Each rank held 91,709,826,224 weight bytes.

| Workload | A tok/s [range] | B tok/s [range] | Peak bytes, rank 0 / rank 1 | Finding |
| --- | --- | --- | --- | --- |
| One-chat prose writing | 23.837 [23.802–23.838] | 23.513 [23.333–23.721] | 93,218,536,452 / 93,218,536,452 | 1.36% slower |
| One-chat code writing | 23.785 [23.710–23.861] | 23.494 [23.438–23.650] | 93,278,108,676 / 93,278,108,676 | 1.22% slower |
| One-chat reasoning-prompt writing, thinking off | 23.764 [23.605–23.858] | 23.691 [23.445–23.787] | 93,278,108,676 / 93,278,108,676 | Tie |
| Eight-chat total | 90.205 [90.020–90.227] | 90.402 [89.601–90.505] | 94,424,406,720 / 94,424,406,720 | Tie |
| Sixteen-chat total | 117.134 [116.938–117.828] | 117.251 [117.160–117.492] | 95,733,390,512 / 95,733,390,512 | Tie |

All tested answer bytes and token IDs matched across builds and starts. The
prose/code ranges did not overlap, so the original strict not-slower verdict
was **FAIL**. The maintainers accepted this roughly **1.3% one-chat cost** for
the 128k pool/default change on October 12; that decision does not turn the
failed not-slower check into a pass. Concurrency ranges overlap and are ties. These native-client
workloads differ from RigMark and the older eight-chat scoreboard.
Retired receipt `defaults-2026-10-12.json` ([receipt status](../bench/receipts/README.md))

The separate default-startup check negotiated a 24,025,760,392-byte pool and
18,125,450,072-byte retention per rank, covering an 18,116,712,944-byte
full-context reservation plus 8,737,128 bytes of publication metadata. The
near-full continuation peaked at 105,751,095,300 device bytes per rank.
Allocation milestones and one-second memory samples do not establish continuous
driver-memory pressure. Override and refusal cases were not part of that check.

Earlier October 12 long-memory checks at `41fa8cf8` passed all four 128k
teacher-document noise/flatness bounds. Eighteen of 24 paired synthetic
retrieval cases passed with no lost baseline successes; **six remain
unverified**. This is partial quality evidence from that arithmetic build,
not a fresh quality suite at `9bcb8170`.
Retired receipt `defaults-2026-10-12.json` ([receipt status](../bench/receipts/README.md))

### Earlier measurements

| Workload and measured build | Result | Evidence and limits |
| --- | --- | --- |
| Cold 8,192-token prompt, resident weights, cache off; source engine `7a26f6ae` | 13.481 s client time to first token | One successful unprofiled sanity sample; no median or speed-gain claim. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| Same prompt length, cache off; earlier candidate `a727601f` | 13.693 / 13.680 s medians in two fresh arms | Three probes per arm; historical gate and sanity timing, with separate missing checks. Retired receipt `exact-path-gate.json` ([receipt status](../bench/receipts/README.md)) |
| Eight distinct short-prompt chats, 256 output tokens each; source QSA candidate `1bc9b667` | 62.709 total client tok/s; range 61.988–62.911 | Three fresh servers; every timed request completed the fixed output count. Strict HTTP cache accounting failed; native output comparisons passed. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| One chat, prose; source engine `4c13c018`, RigMark, thinking off | 25.011 decode tok/s; range 24.919–25.110 | Five natural-stop answers passed basic output checks. RigMark prefill was skipped; the suite was incomplete. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| One chat, code; same RigMark build and settings | 24.838 decode tok/s; range 24.504–24.865 | Five natural-stop answers passed basic output checks. This measures writing speed, not successful coding tasks. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| Clean-install canary, monorepo `38c7b47`, retained cold 8k and shared-reader eight-chat requests | 13.814 s TTFT, range 13.745–13.913; 60.506 total tok/s, range 60.084–61.139 | Three fresh launches, three repeats each. Different pair from the older shared-reader/8k reference; the old executable was not rerun alongside it. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |

Source engine revisions and monorepo revisions are different identities. A result qualifies the
build and workload named in its receipt. A later merge or documentation change
does not inherit a fresh measurement.

The legacy FP8 Flash-Next output default is 65,536 tokens, with context 131,072;
the October 12 measurements above used an 8,192-token output cap and do not
qualify the new output default. These earlier timing and installation results
used 32,768 context tokens. Keep `--context 32768 --output 8192` explicit when
reproducing them and report actual pool/cache bytes; the October 12 startup,
continuation and timing results above are separate evidence. The historical
manual-install canary does not qualify the current Rust lifecycle.

## Client time and server time

Client time measures the wait seen by the program making the request. Server
time measures work inside a chosen part of the engine. Always name the boundary.

| Metric | Definition | What it includes |
| --- | --- | --- |
| Client time to first token (TTFT) | First observable generated stream data minus request-send time | Connection, request transfer, queueing, prompt processing and transfer of the first generated data. The client accepts nonempty answer text, reasoning text or tool-call data; an empty stream header does not count. |
| Total client throughput | Sum of server `usage.completion_tokens` divided by the interval from the first request sent to the last `[DONE]` received | The whole cohort, including reading, queueing and response transfer. Adding individual chat rates does not produce this metric. |
| Older RigMark one-chat writing rate | `(completion_tokens - 1) / (last nonempty output event time - first nonempty output event time)` | Answer text or reasoning counts as output. It excludes the initial wait and the tail from the last output event through `[DONE]`. |
| Legacy one-chat scoreboard writing rate (canary and default `vade bench`) | `(completion_tokens - 1) * 1000 / (elapsed_ms - ttft_ms)` | TTFT counts the first nonempty answer, reasoning or tool-call output. The denominator ends at stream completion after `[DONE]`, including terminal response-transfer overhead. It excludes the initial wait. |
| Answer-content writing rate | `answer_content_tokens_per_s`, currently null | First answer arrival is recorded as `first_content_ms`, but final completion usage does not identify the token count before that arrival. A content clock alone cannot establish an answer-only rate. |
| Server `prefill_ms` | Engine time spent processing the prompt | It excludes time before the engine admits the request and excludes client transfer. |
| Server `admission_to_first_logits_ms` | First model scores minus engine admission time | It includes work after admission; it can miss a wait before admission. |
| Server decode-step time | Time spent advancing the active decode group | It is a diagnostic, not a client rate. State whether ramp-up, partial groups, drain and host/RDMA waiting are included. |

The Rust client definitions are implemented by `vade bench` in
[the Rust benchmark library](../engine/tools/bench/README.md). The former
Python client is retired from the repository; its historical source identity
and the canary writing method were recorded in the
retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)). Count output tokens from
final server usage. A streamed event can contain several tokens; counting events
as tokens is invalid unless that boundary has been checked for the tested server.

The pinned RigMark estimator uses final usage for its token count and streamed
events for its time interval. The retained run reported one token per output
event; buffering can still limit client timing precision. Its boundary differs
from the canary's interval through `[DONE]`.
Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md))

The earlier RigMark run exposed a wait before engine admission after a finished
answer. A server admission timer could therefore look short while the next
client waited much longer. Keep both clocks in follow-up and agent-turn receipts.
Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md))

## Request settings and cache state

Send each sampling field explicitly. This example requests plain drafting;
current `vade bench` omits the draft field unless `--draft` is supplied:

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

Set the output cap explicitly too. Record a supplied seed or the effective seed
returned by the server. Save the effective sampling, thinking, draft mode, MTP
depth and actual draft-use fields from every response. FP4-Local uses greedy
sampling when no sampling field is supplied. If any sampling field is supplied,
unspecified fields come from the checkpoint's `generation_config.json`.
Request intent alone is not proof of effective settings. See the
[request parser](../engine/crates/serve/src/protocol.rs).

The older RigMark body sent temperature, top-p and seed. Its effective echo
filled top-k and neutral penalties from defaults; temperature zero made the run
greedy. That was disclosed in its receipt rather than presented as compliance
with the later all-fields-explicit protocol. It also used an increased streamed
output byte ceiling. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md))

Measure cache-off and cache-hit workloads separately. For cache-off timing,
verify zero reused tokens in usage and the prefix-reuse header. For a hit, retain
the prime request, replay request, restored-token count, cache budget and eviction
state. Similar text is not proof of a cache hit. Cache-accounting failure and
answer-byte equality are separate results; the retired QSA receipt recorded both.
Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md))

“Cold” in the reading rows means an uncached prompt with weights already loaded.
It excludes model download, server build and weight loading. Record whether it
is the first request after startup or a later uncached request. A fresh server
start and another probe on the same server are different repetitions.

## Repetition, noise and ties

For a change comparison, use the same Sparks and fresh alternating servers
for the baseline and candidate. The current rule requires at least three fresh
starts per build. Keep the workload, client, context, output length and sampling
fixed. Report medians and full ranges, and name any warmups or discarded runs.
Overlapping ranges are a **tie**. A shorter median alone does not establish a
speed gain.

Record both requested and actual output lengths. A natural early stop is useful
evidence for that answer, but it is ineligible for a fixed-output timing cell.
Report failures and exclusions alongside eligible runs. Do not reconstruct a
missing client rate from a GPU step timer.

The QSA comparison has eligible fresh-server evidence for its retained shared-reader
workload; other prompt cells ended early and left its full timing matrix
incomplete. The clean-install comparison used historical results from another
pair and a stated historical noise criterion. Its cold-prompt raw ranges were
separated. Those were different qualifications, recorded in the
retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)).

## Build, hardware and memory identity

A useful receipt records the checkpoint revision and hash scope; tokenizer and
chat-template identity; source commit and dirty state; server, PTX and client
hashes; toolchains and driver; rank count and topology; context and cache budgets;
prompt identity, output count, concurrency and arrival pattern. Record clocks,
temperature and competing traffic when available. Mark missing fields as
unmeasured. Never fill them from an assumption.

The earlier headline setup used the official FP8 checkpoint, tensor parallel
execution over a two-Spark RDMA ring and a 32,768-token context. Reading and shared-reader
eight-chat timing use cache off, thinking off and plain decoding. The older
RigMark rows reported no reused prompt tokens; their cache-isolation limitation
was disclosed separately. Selected response checks and reported rank artifact
identities were summarized in the
retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)).

Keep exact bytes for each rank with each rate. The retired receipts reported:

| Workload | Weight bytes per rank | Current device bytes per rank | Reported allocation peak per rank | Evidence |
| --- | ---: | ---: | ---: | --- |
| Unprofiled reading sanity | 91,709,826,224 | Not retained in this summary | 93,564,704,076 | Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| Earlier candidate reading/C8 arms | Not retained in this summary | 92,928,634,296 at C8 health checks | 94,427,138,068 in A/B reading+C8 arms; 93,564,704,076 in B′/A′ reading-only arms | Retired receipt `exact-path-gate.json` ([receipt status](../bench/receipts/README.md)) |
| QSA eight-chat candidate, both ranks | 91,709,826,224 | 92,928,635,704 | 94,373,705,044–94,426,210,720 | Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| Older RigMark prose/code | Unmeasured in this receipt | Unmeasured in this receipt | Unmeasured in this receipt | Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |
| Clean-install shared-reader/8k, both ranks | 91,709,826,224 | 93,151,759,800 | 94,427,138,068 | Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md)) |

Peaks cover explicit allocations observed over a server's lifetime and can include
earlier cells. These figures do not establish total physical-memory use or
driver/graph allocation size. Missing values stay unknown rather than borrowing
memory figures from another build.

Profiled runs explain where time goes. Publish unprofiled timing separately;
profiling overhead and overlapping spans do not form a performance receipt.
Ranks run concurrently, so adding rank times does not yield wall time.

## Qualification and limits

- These rows do not establish a matched speed comparison with another engine.
- Writing speed and capped concurrent generation do not establish useful coding
  tasks completed, harness response time after tools, or agent reliability.
- Native repeatability, agreement with a previous build, model quality and
  survival of a rejected batch row are separate checks. The headline receipts
  do not establish all of them; poison-row qualification remains skipped in
  the named default builds.
- Neither a tested context setting nor a longer configured window proves
  long-memory quality at every length. The October 12 quality evidence above
  remains partial. The failed context candidate was a historical guard; see
  retired receipt `context-candidate.json` ([receipt status](../bench/receipts/README.md)).
- An MTP-enabled server can fall back to plain decoding. The canary's supplement
  observed no actual drafting, so it supplies no speculative-decode speedup
  claim. Retired receipt `night-2026-10-11.json` ([receipt status](../bench/receipts/README.md))
- A Mac build check supplies no GPU arithmetic or speed qualification.

Repeatability requires identical token IDs for the same request and settings
across runs and Sparks, including plain and strict drafting.
The full byte contract and arithmetic-change quality gates are explained in
[Architecture](architecture.md#exactness-and-model-quality).

## Receipts and reproduction

Start with the tested setup and timing instructions in [INSTALL.md](../INSTALL.md).
The [timing tools guide](../tools/README.md#timing-the-installed-server) describes the client.
Measure only on Sparks with no other GPU workload running.

The older public FP8 summaries have been removed. The current tables above
quote card folders and timing READMEs whose sanitized copies are in
[bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md): cards,
READMEs and small JSON receipts, with hostnames, addresses and private paths
replaced by neutral tokens. Original prompts, replies, logits and the full card
JSON receipts are not included. See the [receipt status](../bench/receipts/README.md).

[Historical engine comparisons](../engine/docs/benchmarks.md) preserve older
models and workloads. Their conditions must be read before comparing rates.
Use the explicit methods and qualification limits with each receipt; a historical
result does not establish current-build performance.

## Run vade bench

Build the CLI once, then use one command to measure a running server.
After `vade launch` or `vade up`, bare `vade bench` uses the selected endpoint,
the health model and a bundled prompt. The explicit recipe below instead
selects retained prompts and plain drafting:

```sh
cargo build --release --offline --manifest-path cli/Cargo.toml
cli/target/release/vade bench --url "$VADE_API_URL" \
  --model qwen3.8-flash-next-fp4-local --draft plain --prompts /path/to/retained-prompts.txt --concurrency 1,8,16 \
  --max-tokens 256 --repeats 3 --output /path/to/private-receipt.json
cli/target/release/vade bench --self-test
```

The prompt file has one literal UTF-8 prompt per nonempty line. Retain prose,
code and reasoning prompts between comparisons; the client cycles through
them without generating replacements. Concurrency 1 always uses the first
prompt, so measure one-chat prose and code with separate retained files.
`--bodies-dir` instead uses frozen shared-reader
request bodies and their sibling `capture/` directory. The client retains
source-file hashes and hashes the actual compact UTF-8 request sent on the
wire. It explicitly sends temperature 0, top-p 1, top-k 0, neutral penalties,
cache off and thinking off. Drafting follows the server default unless
`--draft plain` or `--draft strict` is supplied. Check every response's sampling
echo; a server without an echo is marked `sampling_source=request` rather
than silently inventing effective settings.

For cold reading, add `--prefill-body tools/fixtures/probe-8192.json`,
`--prefill-body-sha256 11869d33b6fe2411447a7c9fd34ef76895ca53017f7505e2d63ac77783df945f`
and `--checkpoint-hash HASH`. It makes three uncached probes, requires exactly
8,192 server-reported prompt tokens and checks build/checkpoint identity.
Those are three requests within one server start. Repeat after at least three
fresh server starts per build and alternate A/B for a performance claim.

The JSON receipt preserves the timing client's `schema_version`, `status`, `arms`,
`cells`, `requests`, `prefill_8k`, `receipts`, sampling, identities, memory,
observation and batch-log fields. The sibling `.txt` file gives readable
results and response records. Failed or partial cells keep their records but
publish no rate. Cohort throughput divides all completion tokens by the first
request's send time through the last request's completion time.
`writing_tokens_per_s` preserves the legacy scoreboard estimator,
`(completion_tokens - 1) * 1000 / (elapsed_ms - ttft_ms)`. It uses all final
completion tokens minus one; TTFT counts nonempty answer, reasoning or tool-call
data and can precede the first answer text. The receipt separately records
`first_content_ms` and publishes `answer_content_tokens_per_s` as null: standard
final usage does not identify tokens before first answer content. Do not subtract
reasoning estimates or count SSE chunks to invent that numerator.

`vade bench` accepts plain HTTP, including an existing local SSH tunnel.
HTTPS is unsupported because the standard library has no TLS implementation;
use an existing local HTTP tunnel or trusted TLS terminator. Drafting-enabled
servers are supported. Receipts separate each requested draft mode from the
server-reported MTP depth and build SHA, and retain response observations
separately. A reported depth of zero alone does not prove drafting was off;
only receipts where every request asked for plain label drafting off. Requested
policy and server reports do not independently prove effective drafting.

The historical timing options are mapped below, with current defaults and
receipt meanings. The current command also accepts `--draft plain|strict`.

| Historical timing option | Rust equivalent | Meaning or exception |
| --- | --- | --- |
| `-h`, `--help` | `vade bench --help` | Show timing options without connecting. |
| `--url URL` | `--url URL` | Server base URL, with or without `/v1`; defaults to `VADE_API_URL` or the selected installation's endpoint; HTTP only. |
| `--model ID` | `--model ID` | Model sent and checked; defaults to `/health` model, falling back to `qwen3.8-flash-next-fp4-local`. |
| `--prompts FILE` | `--prompts FILE` | One literal prompt per nonempty line; defaults to a bundled offline writing prompt. |
| `--bodies-dir DIR` | `--bodies-dir DIR` | Retained bodies with sibling `capture/`; frozen readers 1–16. |
| `--concurrency LIST` | `--concurrency LIST` | Distinct positive comma-separated counts; default `1,2,4,8,16`. |
| `--max-tokens N` | `--max-tokens N` | Required completion count; default 256. Early stops fail eligibility. |
| `--repeats N` | `--repeats N` | Repeats per cohort; default 1. |
| `--prefill-body FILE` | `--prefill-body FILE` | Three uncached, pinned 8,192-token probes. |
| `--prefill-body-sha256 HASH` | `--prefill-body-sha256 HASH` | Verify frozen file bytes before requesting. |
| `--checkpoint-hash HASH` | `--checkpoint-hash HASH` | Required checkpoint pin for 8k and alternating comparisons. |
| `--build-hash-field PATH` | `--build-hash-field PATH` | Explicit dotted `/health` build field. |
| `--checkpoint-hash-field PATH` | `--checkpoint-hash-field PATH` | Explicit dotted `/health` checkpoint field. |
| `--ab-urls A B` | `--ab-urls A B` | A/B/B/A; average candidate B runs and compare baseline A/A′ noise. |
| `--aba-urls A B A_PRIME` | `--aba-urls A B A_PRIME` | A/B/A′ order. |
| `--launch-commands COMMAND ...` | `--launch-commands COMMAND ...` | Two foreground commands for ABBA, or three for ABA; stop owned groups after their matrices. |
| `--rank0-log FILE` | `--rank0-log FILE` | Parse only appended batch rows and bytes during each cell. |
| `--aba-rank0-logs A B A_PRIME` | `--aba-rank0-logs A B A_PRIME` | Separate ABA logs. |
| `--ab-rank0-logs A B` | `--ab-rank0-logs A B` | Separate ABBA logs. |
| `--observe-cmd COMMAND` | `--observe-cmd COMMAND` | Repeat per rank; record start/end GPU CSV and uptime with the same `VADE_*` context variables. |
| `--observe-timeout SECONDS` | `--observe-timeout SECONDS` | Per-rank observer timeout; default 30. |
| `--output FILE` | `--output FILE` | JSON receipt and sibling text table; defaults to `./vade-bench.json`. |
| `--timeout SECONDS` | `--timeout SECONDS` | HTTP I/O and cohort barrier timeout; default 1800. |
| `--startup-timeout SECONDS` | `--startup-timeout SECONDS` | Owned-server readiness timeout; default 600. |
| `--api-key-file FILE` | `--api-key-file FILE` | Optional runtime Bearer credential file; `VADE_API_KEY_FILE` fallback remains. Credentials are excluded from receipts. |
| `--self-test` | `--self-test` | Small local mock-stream check; no Spark or model qualification. |

The existing `vade-bench` crate is folded into the CLI as a library, so there
is one runnable command. `vade bench --suite` keeps its earlier decode,
concurrency, cache, priority, long-prompt, follow-up and shared-system workloads.
See [its guide](../engine/tools/bench/README.md) for those options. That suite
uses a different historical receipt format and workload; choose the default
timing protocol for the current scoreboard. The former Python implementation
and operational shim are retired. Their
original bytes and real-server parity evidence are kept outside this repository;
a live pair check requires at least three runs of each client on identical inputs. A Mac mock check verifies
parsing and receipt arithmetic, and supplies no Spark speed claim.
