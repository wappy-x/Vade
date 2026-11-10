# Benchmarks

This page retains historical benchmark records, including the unrounded
[full data](#full-data), build identities, methods and losses. The measurements
apply to those builds and settings. They do not establish current performance
or current behavior of the compared projects. [Reproduce](#reproduce) describes
the benchmark procedure; retain new receipts when testing another build.

**Jump to:** [Qwen3-8B](#qwen3-8b) · [Qwen3-32B](#qwen3-32b) ·
[Qwen3-30B-A3B](#qwen3-30b-a3b) · [Determinism checks](#determinism-checks) ·
[Reproduce](#reproduce) · [Full data](#full-data)

<details>
<summary><b>Terms on this page</b>: token, tok/s, first token, full answer, drafting, session, tuned, window</summary>

<dl>
<dt>Token</dt>
<dd>The unit a model reads and writes, often part of a word.</dd>
<dt>Throughput (tok/s)</dt>
<dd>Answer tokens per second, counted over the whole test, including prefill
(reading the prompt) and queueing.</dd>
<dt>First token (TTFT)</dt>
<dd>Time to first token: how long the first answer token takes.</dd>
<dt>Full answer</dt>
<dd>Time to the last answer token.</dd>
<dt>Drafting</dt>
<dd>Speculative decoding: a small draft model proposes a few tokens ahead and
the big model checks them in one pass. Vade checks every proposal, so its answer
does not change.</dd>
<dt>Session</dt>
<dd>One fresh server start, with the tests run once.</dd>
<dt>Tuned, memory candidate</dt>
<dd>A rival setting with a smaller memory pool. SGLang's is its reduced pool,
the smallest tried.</dd>
<dt>Window A, B, C</dt>
<dd>The three measurement rounds. Each table names its window and Vade
build.</dd>
</dl>

</details>

## Verified results

**Three wins and three ties, all on Qwen3-8B, with three sessions on each
side.**

tok/s: higher is better. Seconds and ms: lower is better.

Against vLLM 0.31, both drafting with the same Qwen3-0.6B draft model:

| Qwen3-8B | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **23.9** | 22.7 | Vade +5% |
| 16 requests (tok/s) | 306.1 | 315.6 | Tie |

Against NGC vLLM 25.09 at stock settings, with neither side drafting:

| Qwen3-8B | Vade | NGC vLLM 25.09 | Result |
| --- | ---: | ---: | --- |
| Follow-up, first token (ms) | 195 | 194 | Tie |
| Follow-up, full answer (s) | **11.1** | 11.9 | Vade +7% |
| Shared prompt, 16 requests (tok/s) | 153.9 | 156.4 | Tie |
| Startup (s) | **3.6** | 138 | Vade 38× |

- **16 requests is a tie** although vLLM's median is 3% higher: the session
  ranges overlap.
- **Builds:** Vade `03068e9` against vLLM 0.31 (Window C); Vade `79a11cc`
  against NGC vLLM 25.09 (Window B).

## Where each side leads

Use the model tables and their build/window labels below. The full data retains
all measured cells; a historical comparison is not a current product ranking.

## How the comparison works

**Same Spark, same time window, same official BF16 weights and identical
requests; each rival at its best measured setting for each row.**

- **Five tests,** sent byte for byte the same to every engine by one frozen
  client: 1 request; 16 requests at once; a 28,672-token long prompt; a
  follow-up turn on that conversation; 16 requests sharing one 8,192-token
  system prompt. Startup and peak GPU memory as well.
- **Rivals:** SGLang 0.5.21, vLLM 0.31, NGC vLLM 25.09 (NVIDIA's packaged
  vLLM v0.10.1.1) and TensorRT-LLM 1.3.0rc13, each on the same Spark and in
  the same time window as the Vade build it is compared with. Each rival
  shows its best result for the row, from any setting we ran, named under
  each table.
- **Result column:** the winner's lead. +5% means 5% faster, or the other
  engine needed 5% more time or memory; 38× means 38 times. Bold marks the
  better number. Losses and ties stay next to the wins.
- **Ties and verified:** anything within 3% is a tie, and a tie is never a
  win. Verified needs three fresh server starts on each side and session
  ranges that do not overlap. Anything less is preliminary, marked \*.
  Memory rows always count as preliminary.
- **Where Vade drafts:** in the builds these tables measured, only 1 request
  and 16 requests, on Qwen3-8B and Qwen3-32B. Later recorded builds also drafted
  Qwen3-30B-A3B and the follow-up answer (see Later merged changes). A rival at stock settings never drafts, so Vade's default
  against a stock rival is drafting against no drafting, not stock against
  stock. Fair rows pair drafting with drafting, or Vade's `--draft plain`
  with a stock rival.
- **Sampling:** every table here ran with sampling off (temperature 0).
  Use a separate receipt for sampled requests.
- **Repeated text:** the long-prompt and shared-prompt tests repeat one
  paragraph. Vade's drafting can copy it, so a gap in Vade's favour on those
  tests gets no verdict until they use natural text.

<a id="test-names"></a>

<details>
<summary><b>The tests</b>: what each one sends and measures, with its record name</summary>

The record name of each test is in capitals. A *frozen client* is a
benchmark program kept at a fixed version, so every engine gets identical
requests.

- **1 request** (ONE): one request, up to 128 answer tokens. Measures
  answer tokens per second (tok/s).
- **16 requests** (SIXTEEN): 16 different requests sent at once, up to 128
  answer tokens each. Measures total tok/s across all 16.
- **Long prompt** (LONG): one prompt of exactly 28,672 tokens, up to 16
  answer tokens. Measures time to first token, recorded in milliseconds
  (ms).
- **Follow-up** (FOLLOWUP): a 28,672-token first turn, then the whole
  conversation sent again with one new question, up to 128 answer tokens. A
  server can reuse its KV cache (its memory of the conversation so far)
  instead of reading turn one again. Measures the second answer's first
  token, in ms, and its full answer.
- **Shared prompt** (SHARED): 16 requests at once that all start with the
  same 8,192-token system prompt, up to 128 answer tokens each. Measures
  total tok/s.

**Startup** is the time from launching the server until it answers its
health check, in seconds (s). **Peak memory** is the highest GPU memory the
server used, sampled every 100 ms, in GB (1,000,000,000 bytes) or exact
bytes.

</details>

<details>
<summary><b>More about the setup</b>: deterministic modes, memory rows, margins and Vade builds</summary>

- **Deterministic modes.** A rival's deterministic mode has its own table,
  not a column in the main ones.
- **Memory rows** use Vade's default, which also loads the draft model on
  Qwen3-8B and Qwen3-32B. With drafting off, Vade's peak is about 2 GB lower
  ([Full data](#full-data)).
- **Margins** come from the unrounded values in [Full data](#full-data).
- **Which Vade.** Results come from three measurement windows, A, B and C,
  and each table names its window. The rival tables use Vade builds
  `79a11cc` and `03068e9`. No rival has been measured against the code now
  in main. Later Vade changes are measured against earlier Vade in
  [Later merged changes](#later-merged-changes).

</details>

## Qwen3-8B

**Even or ahead on 1 request, 16 requests and the long prompt; far ahead on
startup.** Behind when a rival drafts the follow-up answer, on the shared
prompt against SGLang, and on memory against memory-tuned rival settings.

<details>
<summary><b>Qwen3-8B</b>: Vade against SGLang 0.5.21</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-8B | Vade | SGLang 0.5.21 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | 23.5 | 23.2 | Tie* |
| 16 requests (tok/s) | 302.9 | 296.1 | Tie* |
| Long prompt, first token (s) | 8.3 | 8.3 | Tie* |
| Follow-up, first token (ms) | 196 | 194 | Tie* |
| Follow-up, full answer (s) | 11.3 | **6.3** | SGLang +80%* |
| Shared prompt, 16 requests (tok/s) | 152.0 | **159.6** | SGLang +5%* |
| Startup (s) | **3.5** | 162 | Vade 46×* |
| Peak memory, stock (GB) | **26.5** | 109.7 | Vade 4.1×* |
| Peak memory, rival tuned (GB) | 26.5 | **24.8** | SGLang +7%* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window A.** Vade `79a11cc` at its defaults, three sessions. SGLang
  0.5.21, one session per setting.
- **SGLang's best setting per row.** Drafting with Qwen3-0.6B: 1 request, 16
  requests, follow-up full answer. Stock: startup, long prompt, follow-up
  first token, stock memory. Smallest pool: shared prompt, tuned memory.
- **Startup** is a cold start for both (model files dropped from memory
  first). The tuned memory row is SGLang's smallest pool tried. Vade's own
  minimum settings go lower (21.9 GB) but lose most multi-chat speed.

</details>

<details>
<summary><b>Qwen3-8B</b>: Vade against vLLM 0.31</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-8B | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **23.9** | 22.7 | Vade +5% |
| 16 requests (tok/s) | 306.1 | 315.6 | Tie |
| Long prompt, first token (s) | **8.3** | 8.7 | Vade +5%* |
| Follow-up, first token (ms) | **195** | 203 | Vade +4.0%* |
| Follow-up, full answer (s) | 11.1 | **6.7** | vLLM +65%* |
| Shared prompt, 16 requests (tok/s) | 153.9 | 154.7 | Tie* |
| Startup (s) | **3.6** | 199 | Vade 55×* |
| Peak memory, stock (GB) | **26.5** | 117.7 | Vade 4.4×* |
| Peak memory, rival tuned (GB) | 26.5 | **24.5** | vLLM +8%* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **1 request and 16 requests: Window C, verified.** Vade `03068e9` and
  vLLM, both drafting with Qwen3-0.6B, three sessions each.
- **16 requests is a tie** although vLLM's median is 3% higher: the session
  ranges overlap (Vade 306.1 to 307.4, vLLM 300.7 to 325.7). This
  three-session recheck replaced a single Window B session that reached
  327.2.
- **Other rows: Window B.** Vade `79a11cc`, three sessions; vLLM one session
  per setting. vLLM's best: its tuned setting for startup, long prompt,
  follow-up first token, shared prompt and tuned memory; drafting for the
  follow-up full answer; stock for stock memory.

</details>

<details>
<summary><b>Qwen3-8B</b>: Vade against NGC vLLM 25.09</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-8B | Vade | NGC vLLM 25.09 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | 23.8 | invalid | No valid result |
| 16 requests (tok/s) | 304.9 | invalid | No valid result |
| Long prompt, first token (s) | 8.3 | invalid | No valid result |
| Follow-up, first token (ms) | 195 | 194 | Tie |
| Follow-up, full answer (s) | 11.1 | **3.1** | NGC vLLM 3.6×* |
| Shared prompt, 16 requests (tok/s) | 153.9 | 157.2 | Tie* |
| Startup (s) | **3.6** | 138 | Vade 38× |
| Peak memory, stock (GB) | **26.5** | 116.8 | Vade 4.4×* |
| Peak memory, rival tuned (GB) | 26.5 | **25.0** | NGC vLLM +6%* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window B.** Vade `79a11cc`, three sessions. NGC vLLM 25.09 is NVIDIA's
  packaged vLLM v0.10.1.1. Stock has three sessions; n-gram drafting and the
  tuned setting have one each.
- **Best setting per row.** Stock: startup, follow-up first token, stock
  memory. N-gram drafting (guesses copied from earlier text): follow-up full
  answer. Tuned: shared prompt, tuned memory.
- **Verified, both at stock with no drafting on either side:** Vade finished
  the follow-up full answer 7% sooner (11.1 s against 11.9 s). The shared
  prompt tied (153.9 against 156.4).
- **No valid result** for 1 request, 16 requests and long prompt: this
  version leaves out the token counts the test client checks.

</details>

<details>
<summary><b>Qwen3-8B</b>: Vade against TensorRT-LLM 1.3.0rc13</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-8B | Vade | TensorRT-LLM | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | 23.8 | invalid | No valid result |
| 16 requests (tok/s) | 304.9 | invalid | No valid result |
| Long prompt, first token (s) | 8.3 | invalid | No valid result |
| Follow-up, first token (ms) | **195** | 202 | Vade +3.5%* |
| Follow-up, full answer (s) | **11.1** | 11.9 | Vade +7%* |
| Shared prompt, 16 requests (tok/s) | **153.9** | 125.6 | Vade +22%* |
| Startup (s) | **3.6** | 183 | Vade 50×* |
| Peak memory, stock (GB) | **26.5** | 93.0 | Vade 3.5×* |
| Peak memory, rival tuned (GB) | 26.5 | 26.3 | Tie* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window B.** Vade `79a11cc`, three sessions. TensorRT-LLM 1.3.0rc13, one
  session per setting, input limit raised to 32,768 tokens. No drafting or
  deterministic setting was measured.
- **No valid result** for 1 request, 16 requests and long prompt: this
  version rejected three extra fields in the test's requests.
- **Best setting per row.** Stock, except startup and tuned memory, which
  come from its tuned setting. TensorRT-LLM keeps some weights in host
  memory, so its memory count is unverified.

</details>

<a id="qwen3-8b-deterministic-modes"></a>

<details>
<summary><b>Qwen3-8B</b>: deterministic modes of SGLang and vLLM</summary>

Vade's default gives the same answer every time. These rival modes aim for
the same. Vade still drafts 1 request and 16 requests here; the rivals do
not.

tok/s: higher is better. Seconds and ms: lower is better. Against SGLang 0.5.21's
deterministic mode:

| Qwen3-8B, deterministic | Vade | SGLang 0.5.21 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **23.5** | 13.3 | Vade +76%* |
| 16 requests (tok/s) | **302.9** | 203.4 | Vade +49%* |
| Long prompt, first token (s) | **8.3** | 23.2 | Vade 2.8×* |
| Follow-up, first token (ms) | **196** | 365 | Vade +87%* |
| Follow-up, full answer (s) | **11.3** | 12.2 | Vade +8%* |
| Shared prompt, 16 requests (tok/s) | **152.0** | 122.2 | Vade +24%* |

\* preliminary: one session on at least one side.

- **Window A.** Vade `79a11cc` at its defaults, three sessions. SGLang
  0.5.21 in its deterministic mode with Triton attention, one session, no
  drafting.

Against vLLM 0.31's deterministic mode:

| Qwen3-8B, deterministic | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **23.8** | 14.0 | Vade +70%* |
| 16 requests (tok/s) | **304.9** | 212.4 | Vade +44%* |
| Long prompt, first token (s) | **8.3** | 9.5 | Vade +15%* |
| Follow-up, first token (ms) | **195** | 248 | Vade +27%* |
| Follow-up, full answer (s) | **11.1** | 16.2 | Vade +45%* |
| Shared prompt, 16 requests (tok/s) | **153.9** | 139.8 | Vade +10%* |

\* preliminary: one session on at least one side.

- **Window B.** Vade `79a11cc` at its defaults, three sessions. vLLM 0.31
  with `VLLM_BATCH_INVARIANT=1`, one session, no drafting.

</details>

## Qwen3-32B

**Ahead on 1 request and startup; even or ahead on 16 requests and the long
prompt.** Behind on the shared prompt against SGLang and vLLM, on the
follow-up's full answer against SGLang drafting and TensorRT-LLM, and on
follow-up first token and tuned memory against vLLM. Every Qwen3-32B result
is preliminary.

<details>
<summary><b>Qwen3-32B</b>: Vade against SGLang 0.5.21</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-32B | Vade | SGLang 0.5.21 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **6.9** | 6.3 | Vade +10%* |
| 16 requests (tok/s) | **90.0** | 86.8 | Vade +3.8%* |
| Long prompt, first token (s) | 32.5 | 32.7 | Tie* |
| Follow-up, first token (ms) | **601** | 758 | Vade +26%* |
| Follow-up, full answer (s) | 17.5 | **10.7** | SGLang +63%* |
| Shared prompt, 16 requests (tok/s) | 38.9 | **51.4** | SGLang +32%* |
| Startup (s) | **9.8** | 610 | Vade 62×* |
| Peak memory, stock (GB) | **81.6** | 116.0 | Vade +42%* |
| Peak memory, rival tuned (GB) | 81.6 | 80.1 | Tie* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window A.** Vade `79a11cc`, one session. SGLang 0.5.21, one session per
  setting.
- **SGLang's best setting per row.** Drafting with Qwen3-0.6B: 1 request, 16
  requests, follow-up full answer, shared prompt. Smallest pool: long
  prompt, follow-up first token, tuned memory. Stock: startup, stock memory.
- **Startup** is a cold start for both. Both ended the follow-up at 52
  tokens. Vade's own minimum memory setting goes lower (75.2 GB) but loses
  most multi-chat speed.
- **At stock settings** SGLang was also ahead on the follow-up full answer
  (16.5 s) and the shared prompt (+7%).

</details>

<details>
<summary><b>Qwen3-32B</b>: Vade against vLLM 0.31</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-32B | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **6.9** | 6.3 | Vade +10%* |
| 16 requests (tok/s) | 89.8 | 91.4 | Tie* |
| Long prompt, first token (s) | **32.6** | 34.7 | Vade +6%* |
| Follow-up, first token (ms) | 602 | **576** | vLLM +4.5%* |
| Follow-up, full answer (s) | 17.3 | 14.5 | Not comparable |
| Shared prompt, 16 requests (tok/s) | 39.4 | **43.8** | vLLM +11%* |
| Startup (s) | **10.1** | 424 | Vade 42×* |
| Peak memory, stock (GB) | **81.6** | 119.1 | Vade +46%* |
| Peak memory, rival tuned (GB) | 81.6 | **76.0** | vLLM +7%* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window B.** Vade `79a11cc`, two sessions: one paired with the stock
  runs, one with the drafting and memory runs. The paired session alone
  gives the same results. vLLM, one session per setting.
- **vLLM's best setting per row.** Drafting with Qwen3-0.6B: 1 request, 16
  requests, shared prompt. Tuned: startup, long prompt, follow-up first
  token, tuned memory. Stock: stock memory.
- **Follow-up full answer:** vLLM stopped at 46 tokens and Vade at 52, so
  the times are not like-for-like.

</details>

<details>
<summary><b>Qwen3-32B</b>: Vade against NGC vLLM 25.09</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-32B | Vade | NGC vLLM 25.09 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | 6.9 | invalid | No valid result |
| 16 requests (tok/s) | 89.8 | invalid | No valid result |
| Long prompt, first token (s) | 32.6 | invalid | No valid result |
| Follow-up, first token (ms) | 602 | 609 | Tie* |
| Follow-up, full answer (s) | 17.3 | 14.6 | Not comparable |
| Shared prompt, 16 requests (tok/s) | 39.4 | 38.9 | Tie* |
| Startup (s) | **10.1** | 380 | Vade 38×* |
| Peak memory, stock (GB) | **81.6** | 116.8 | Vade +43%* |
| Peak memory, rival tuned (GB) | 81.6 | not run | Not measured |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window B.** Vade `79a11cc`, two sessions. NGC vLLM stock, one session;
  its n-gram drafting was no better on any valid row. No tuned setting was
  measured.
- **No valid result** for 1 request, 16 requests and long prompt (missing
  token counts). **Follow-up full answer:** NGC stopped at 46 tokens and
  Vade at 52.

</details>

<details>
<summary><b>Qwen3-32B</b>: Vade against TensorRT-LLM 1.3.0rc13</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-32B | Vade | TensorRT-LLM | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **6.9** | 3.6 | Vade +95%* |
| 16 requests (tok/s) | **89.9** | 53.4 | Vade +68%* |
| Long prompt, first token (s) | **32.6** | 36.5 | Vade +12%* |
| Follow-up, first token (ms) | **607** | 651 | Vade +7%* |
| Follow-up, full answer (s) | 17.3 | **16.6** | TensorRT-LLM +4.1%* |
| Shared prompt, 16 requests (tok/s) | **39.4** | 32.2 | Vade +22%* |
| Startup (s) | **18.1** | 471 | Vade 26×* |
| Peak memory, stock (GB) | 81.6 | 82.3 | Tie* |
| Peak memory, rival tuned (GB) | 81.6 | 81.5 | Tie* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window C.** Vade `03068e9`, one session. TensorRT-LLM, one session per
  setting, with a small request adapter that accepts the test's extra
  fields. Kernels, memory pools and scheduling stay stock.
- **Best setting per row.** Stock, except follow-up first token and tuned
  memory. 1 request, 16 requests and long prompt use the test client's own
  token count, because the server left it out.
- **Follow-up full answer comes from Window B** (Vade `79a11cc`), where both
  ended at 52 tokens. In Window C, TensorRT-LLM stopped at 46. Its memory
  count is unverified.

</details>

<details>
<summary><b>Qwen3-32B</b>: deterministic modes of SGLang and vLLM</summary>

tok/s: higher is better. Seconds and ms: lower is better. Against SGLang 0.5.21's
deterministic mode:

| Qwen3-32B, deterministic | Vade | SGLang 0.5.21 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **6.9** | 3.3 | Vade 2.1×* |
| 16 requests (tok/s) | **90.0** | 50.7 | Vade +78%* |
| Long prompt, first token (s) | **32.5** | 81.6 | Vade 2.5×* |
| Follow-up, first token (ms) | **601** | 1,239 | Vade 2.1×* |
| Follow-up, full answer (s) | **17.5** | 18.3 | Vade +4.4%* |
| Shared prompt, 16 requests (tok/s) | **38.9** | 33.9 | Vade +15%* |

\* preliminary: one session on at least one side.

- **Window A.** Vade `79a11cc` at its defaults, one session. SGLang 0.5.21
  in its deterministic mode with Triton attention, one session, no drafting.

Against vLLM 0.31's deterministic mode:

| Qwen3-32B, deterministic | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **6.9** | 3.3 | Vade 2.1×* |
| 16 requests (tok/s) | **89.9** | 51.3 | Vade +75%* |
| Long prompt, first token (s) | **32.6** | 38.8 | Vade +19%* |
| Follow-up, first token (ms) | **607** | 650 | Vade +7%* |
| Follow-up, full answer (s) | 17.3 | 18.7 | Not comparable |
| Shared prompt, 16 requests (tok/s) | 39.4 | 38.8 | Tie* |

\* preliminary: one session on at least one side.

- **Window C.** Vade `03068e9` at its defaults, one session. vLLM 0.31 with
  `VLLM_BATCH_INVARIANT=1`, one session, no drafting. The follow-up ended at
  46 tokens on vLLM and 52 on Vade.

</details>

## Qwen3-30B-A3B

**Ahead of both rivals on 1 request, the long prompt, follow-up first token
and startup.** Behind vLLM on 16 requests, the shared prompt and tuned
memory. Every Qwen3-30B-A3B result is preliminary.

A mixture-of-experts model: each token uses 8 of 128 small expert networks
per layer. The build measured here never drafted it; current main does.
SGLang has not been measured on it.

<details>
<summary><b>Qwen3-30B-A3B</b>: Vade against vLLM 0.31</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-30B-A3B | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **33.5** | 28.4 | Vade +18%* |
| 16 requests (tok/s) | 149.2 | **190.6** | vLLM +28%* |
| Long prompt, first token (s) | **7.8** | 9.7 | Vade +24%* |
| Follow-up, first token (ms) | **305** | 365 | Vade +20%* |
| Follow-up, full answer (s) | 2.6 | 2.3 | Not comparable |
| Shared prompt, 16 requests (tok/s) | 103.6 | **122.7** | vLLM +18%* |
| Startup (s) | **9.8** | 512 | Vade 52×* |
| Peak memory, stock (GB) | **68.8** | 118.4 | Vade +72%* |
| Peak memory, rival tuned (GB) | 68.8 | **66.2** | vLLM +3.9%* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window C.** Vade `03068e9`, median of three sessions. vLLM, one session
  per setting.
- **vLLM's best setting per row.** Stock: 1 request, long prompt, follow-up
  first token, shared prompt, stock memory. Tuned: tuned memory. Drafting
  retry without graphs ("eager"): startup.
- **16 requests** comes from drafting with Qwen3-0.6B. That setting then
  crashed on the long prompt. Its eager retry finished every test and
  reached 168.0 tok/s, still 13% ahead of Vade.
- **Follow-up full answer:** vLLM stopped at 46 tokens and Vade at 50.

</details>

<details>
<summary><b>Qwen3-30B-A3B</b>: Vade against TensorRT-LLM 1.3.0rc13</summary>

tok/s: higher is better. Seconds, ms and GB: lower is better.

| Qwen3-30B-A3B | Vade | TensorRT-LLM | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **33.5** | 29.6 | Vade +13%* |
| 16 requests (tok/s) | **149.2** | 141.1 | Vade +6%* |
| Long prompt, first token (s) | **7.8** | 9.2 | Vade +17%* |
| Follow-up, first token (ms) | **305** | 336 | Vade +10%* |
| Follow-up, full answer (s) | 2.6 | 2.6 | Tie* |
| Shared prompt, 16 requests (tok/s) | 103.6 | 104.8 | Tie* |
| Startup (s) | **9.8** | 431 | Vade 44×* |
| Peak memory, stock (GB) | **68.8** | 72.5 | Vade +5%* |
| Peak memory, rival tuned (GB) | 68.8 | 68.6 | Tie* |

\* preliminary: one session on at least one side. Memory rows always count
as preliminary.

- **Window C.** Vade `03068e9`, median of three sessions. TensorRT-LLM, one
  session per setting, with the request adapter. 1 request, 16 requests and
  long prompt use the client's own token count.
- **Best setting per row.** Stock, except startup and tuned memory (tuned).
  Both ended the follow-up at 50 tokens. TensorRT-LLM's memory count is
  unverified.

</details>

<details>
<summary><b>Qwen3-30B-A3B</b>: deterministic mode of vLLM</summary>

tok/s: higher is better. Seconds and ms: lower is better. Against vLLM 0.31's
deterministic mode:

| Qwen3-30B-A3B, deterministic | Vade | vLLM 0.31 | Result |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | **33.5** | 25.9 | Vade +29%* |
| 16 requests (tok/s) | **149.2** | 112.4 | Vade +33%* |
| Long prompt, first token (s) | **7.8** | 10.7 | Vade +37%* |
| Follow-up, first token (ms) | **305** | 395 | Vade +29%* |
| Follow-up, full answer (s) | 2.6 | 3.8 | Not comparable |
| Shared prompt, 16 requests (tok/s) | 103.6 | **118.9** | vLLM +15%* |

\* preliminary: one session on at least one side.

- **Window C.** Vade `03068e9`, median of three sessions; it does not draft
  this model. vLLM 0.31 with `VLLM_BATCH_INVARIANT=1`, one session, no
  drafting. The follow-up ended at 46 tokens on vLLM and 50 on Vade.

</details>

## Later merged changes

**Since the rival runs, merged changes made the shared prompt 10% to 20%
faster and cut the follow-up's first-token time by 15% to 33%, at the cost
of higher peak memory and a verified 3% loss on Qwen3-30B-A3B's 16
requests. The two newest merges more than recovered that loss (Qwen3-30B-A3B
16 requests 141 to 186 tok/s) and draft the follow-up's answer (Qwen3-8B
11.4 s to 6.3 s).**

- These compare a Vade build with the build it started from, on the same
  Spark, with default launches. They are **not rival results** and are not
  copied into the tables above.
- Qwen3 rows use frozen client `1d80e7c`.
- A *pair* is one fresh session of each build. One pair is a smoke check,
  not a verified result; most rows below rest on one pair.

<details>
<summary><b>Faster mixture-of-experts single request</b> (exp1, merged as <code>03068e9</code>)</summary>

Three alternating pairs. tok/s: higher is better. Seconds, ms and GB: lower
is better.

| Qwen3-30B-A3B | Before `79a11cc` | After `53bc7b9` | Change |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | 30.7 | **32.9** | 7% faster |
| Follow-up, full answer (s) | 2.7 | **2.6** | 3% less time |
| Follow-up, first token (ms) | 294 | 309 | Tie |
| Peak memory (GB) | 68.6 | 68.6 | Tie* |

\* preliminary: memory rows always count as preliminary.

- **Follow-up, first token** is recorded as a tie, although its median was
  5% slower.
- 16 requests (147.8 to 147.7), shared prompt (101.5 to 101.7) and Qwen3-8B
  1 request (23.9 to 23.9) tied.
- Answers did not change: all 13,587 token IDs, texts, finish reasons and
  token counts match the earlier build.
- A second round with five pairs measured 1 request at 30.8 to 32.9 tok/s.

</details>

<details>
<summary><b>Memory admission</b> (admit, merged as <code>b636c42</code>)</summary>

What changed: the server charges each conversation's real BF16 memory,
keeps draft memory only for the draft's window, and lets the prompt cache
borrow idle memory, evicting the least recently used entries under
pressure.

One pair, default launches. Vade does not draft the follow-up. Lower is
better in both tables.

| Follow-up, first token (ms) | Before `03068e9` | After `3911e2b` | Change |
| --- | ---: | ---: | --- |
| Qwen3-8B | 194 | **130** | 33% less time* |
| Qwen3-32B | 628 | **491** | 22% less time* |
| Qwen3-30B-A3B | 311 | **265** | 15% less time* |

The cost, in peak GPU memory:

| Peak memory (GB) | Before `03068e9` | After `3911e2b` | Change |
| --- | ---: | ---: | --- |
| Qwen3-8B | **26.5** | 31.3 | 18% more memory* |
| Qwen3-32B | **81.6** | 89.6 | 10% more memory* |
| Qwen3-30B-A3B | **68.8** | 71.4 | 4% more memory* |

\* preliminary: one pair.

- An earlier three-pair gate, against `0f595e0`, also cut the follow-up's
  first token: 197 to 130 ms, 612 to 486 ms and 306 to 267 ms. It showed the
  same memory rise: 26.5 to 31.3, 81.6 to 89.6 and 68.8 to 71.4 GB.
- Every other test stayed within 2% in this pair. Qwen3-8B's long-prompt
  first token was 1.6% slower (8.14 to 8.27 s); that is not verified.

Capacity on Qwen3-32B:

- **Admission gate.** Two independent 40,000-token prompts ran at the same
  time, with 16-token answers. Four 28,672-token prompts all finished, at
  most three at a time; four at once, seen in an earlier round, was not
  reproduced. The capacity run peaked at 109,851,967,488 sampled bytes.
- **Combined main `430fdbc`.** Two 40,000-token prompts ran at the same time
  and each finished 128 tokens, in 165,765 and 166,091 ms, at 90,469,040,128
  peak bytes. One run.
- **YaRN** (an optional context extension) started at 131,072 tokens.
  Starting at that length does not prove complete answers at that length.

</details>

<details>
<summary><b>Shared-prompt attention</b> (shared9, merged as <code>430fdbc</code>)</summary>

What changed: one fixed, tensor-core decode attention order for single
chats, batches and cached prefixes. It keeps every path identical to the
others. It does not promise identical output to the older attention code.

One pair. Only rows that moved 3% or more. tok/s: higher is better.

| Throughput (tok/s) | Before `03068e9` | After `39aceb2` | Change |
| --- | ---: | ---: | --- |
| Qwen3-8B, shared prompt | 151.8 | **167.2** | 10% faster* |
| Qwen3-32B, shared prompt | 39.1 | **43.5** | 11% faster* |
| Qwen3-30B-A3B, shared prompt | 100.5 | **120.9** | 20% faster* |
| Qwen3-32B, 1 request | 6.9 | **7.3** | 5% faster* |
| Qwen3-32B, 16 requests | 90.1 | **93.9** | 4% faster* |
| Qwen3-30B-A3B, 16 requests | **146.0** | 140.9 | 3% slower |

The follow-up's first token, in ms (lower is better):

| Follow-up, first token (ms) | Before `03068e9` | After `39aceb2` | Change |
| --- | ---: | ---: | --- |
| Qwen3-8B | **198** | 204 | 3% more time* |
| Qwen3-30B-A3B | **311** | 325 | 5% more time* |

\* preliminary: one pair. The Qwen3-30B-A3B 16-request loss is verified by
a three-pair gate.

- **The Qwen3-30B-A3B 16-request loss is real.** An earlier three-pair gate
  measured 145.80 to 141.01 tok/s, at 65,138,589,696 bytes on both builds.
  The new attention order changes which experts some tokens pick, so more
  expert weights are read. The loss was accepted at merge, not fixed.
- **Peak memory did not change** (26.5, 81.6 and 68.8 GB). Every other speed
  and first-token result moved less than 3%.
- **Shorter follow-up totals are not speedups.** Qwen3-32B and Qwen3-30B-A3B
  follow-ups ended naturally with fewer tokens after the change.

</details>

<a id="gemma-4-31b-and-qwen38-27b-vade-only"></a>

<details>
<summary><b>Gemma 4 31B and Qwen3.8-27B</b> (Vade only)</summary>

No rival comparison exists yet for these models.

**Gemma's official assistant draft** (merged as `bb8ef34`). Three runs of
each, client 3 (not frozen client 4). tok/s: higher is better; GB: lower is
better.

| Gemma 4 31B | No draft | Assistant draft | Change |
| --- | ---: | ---: | --- |
| 1 request (tok/s) | 3.7 | **8.3** | 2.2× faster |
| Peak memory (GB) | 62.1 | 63.1 | Tie* |

\* preliminary: memory rows always count as preliminary. Peak memory rose
2%, within the 3% tie rule.

- 16 requests stayed level: 47.7 tok/s both ways. Gemma drafts only when one
  request is active.

**Merge check into main** (`9c3fe9c`, merged as `bb8ef34`), frozen client
4, Vade only:

| Vade only | Gemma 4 31B | Qwen3.8-27B BF16 |
| --- | ---: | ---: |
| 1 request (tok/s) | 8.3 | 7.1 |
| 16 requests (tok/s) | 47.3 | 59.2 |
| Long prompt, first token (s) | 62.1 | Not measured |
| Peak memory (GB) | 77.8 | 58.8 |

- **Sessions.** Gemma: three pairs for 1 request and 16 requests, one for
  the long prompt. Qwen3.8-27B: one pair.
- **Drafting.** Gemma's 1 request used its assistant draft. Both models
  decode 16 requests plainly. The record does not say whether Gemma's long
  prompt or Qwen3.8-27B's 1 request drafted. The server drafts a single chat
  by default when a draft is available; Qwen3.8-27B BF16 falls back to its
  built-in prediction head.

</details>

<details>
<summary><b>Combined build</b> (train5, merged as <code>eb67c8b</code>)</summary>

Three alternating pairs per model against main `603f95b`, default launches.
Higher is better.

| 16 requests (tok/s) | Before `603f95b` | After `8b2183c` | Change |
| --- | ---: | ---: | --- |
| Qwen3-8B | 308.8 | **321.3** | 4% faster |
| Qwen3-32B | 93.8 | **97.6** | 4% faster |
| Qwen3-30B-A3B | 141.4 | **186.5** | 32% faster |

- Qwen3-8B 1 request: 24.0 to 24.4 tok/s. Qwen3.8-27B 1 request: 7.01 to
  7.18 tok/s; long prompt, first token: 25.0 to 24.0 s.
- Everything else tied, including the shared prompt on all three Qwen3
  models (Qwen3-30B-A3B 129.0 to 130.1 tok/s).
- Answers did not change on the Qwen3 models: all paired token IDs, texts,
  finish reasons and token counts match. Qwen3.8-27B now matches its
  official reference on 224 of 224 IDs, up from 196.
- Peak memory, Qwen3-30B-A3B: 71.4 to 74.1 GB, with its official draft model
  loaded.*

\* preliminary: memory rows always count as preliminary.

</details>

<details>
<summary><b>Drafted follow-up answer</b> (draftwin, merged as <code>5806ecc</code>)</summary>

Three alternating pairs per model against main `eb67c8b`. Lower is better.

| Follow-up, full answer (s) | Before `eb67c8b` | After `0aa9808` | Change |
| --- | ---: | ---: | --- |
| Qwen3-8B | 11.4 | **6.3** | 44% less time |
| Qwen3-32B | 14.9 | **10.0** | 33% less time |

- The draft model now keeps the whole conversation in its own cache, so it
  can propose the follow-up's answer. Every proposal is checked, so answers
  did not change: 216 of 216 paired replies match.
- Follow-up first token, 1 request, 16 requests, long prompt and shared
  prompt tied.
- Peak memory: Qwen3-8B 31.3 to 34.7 GB, Qwen3-32B 89.6 to 92.9 GB.*

\* preliminary: memory rows always count as preliminary.

</details>

## Determinism checks

**Vade gave the same reply alone and batched on all three Qwen3 models;
every rival at stock settings differed on at least one model.** The rivals'
deterministic modes matched.

- A rival *spot check* sent one prompt alone and the same prompt inside a
  batch of 16, then compared the replies. It is not full qualification.
- [Validation](validation.md) has Vade's full evidence. This section keeps
  the summary and the rival spot checks.

<details>
<summary><b>Vade's determinism and official-agreement results</b>, in short</summary>

The final shared9 gate checked the determinism promise on all three Qwen3
models:

- Logits (the model's raw score for every possible next token) were
  bit-identical alone and batched, at batch widths (chats run together) of
  1 to 64.
- A sweep over six cached-prefix lengths (the opening part of a prompt that
  is already in the cache) used widths 1, 2, 4, 8, 16, 32 and 64; the full
  64-by-6 combination was not run.
- Each of the 16 requests from the 16-request test, replayed alone on a
  fresh server, matched its batched reply: 16 of 16 per model.
- Cached-versus-cold shared-prompt replies matched: 16 of 16 replies and
  2,048 token IDs per model.

Agreement with the official model is a separate check, and it is not
perfect. [Validation](validation.md#promise-2-agreement-with-the-official-model)
has the counts for each model and build, and every disagreement.

</details>

<details>
<summary><b>Rival spot checks</b>: which engines gave the same reply alone and batched</summary>

| Engine and setting | Qwen3-8B | Qwen3-32B | Qwen3-30B-A3B |
| --- | --- | --- | --- |
| Vade default and plain | Same | Same | Same |
| SGLang stock | Differs | Differs | Not run |
| SGLang drafting | Unverified | Differs | Not run |
| SGLang deterministic | Same IDs | Same IDs | Not run |
| NGC vLLM 25.09 stock | Differs in 2 of 3 | Differs | Not run |
| vLLM 0.31 stock | Differs | Differs | Same |
| vLLM 0.31 deterministic | Same | Same | Same |
| vLLM 0.31 drafting | Text differs | Not run | Text differs |
| vLLM 0.31 memory candidate | Not run | Not run | Text differs |
| TensorRT-LLM | Unverified | Differs | Text differs |

- **Same:** the same token IDs and text. **Same IDs:** the same 128 token
  IDs. **Text differs:** the text differed; token IDs were not available.
- **SGLang stock** differed after a cache flush. **NGC vLLM** on Qwen3-8B
  was the same in 1 of 3 sessions and differed in 2. **vLLM drafting** on
  Qwen3-8B: the text differed in 2 of 3 sessions.
- **vLLM memory candidate** is the expert model's tuned setting.
- **TensorRT-LLM** on Qwen3-8B is unverified because its requests were
  rejected; on Qwen3-32B it differed in native token IDs. The API-adapted
  TensorRT-LLM rows do not establish deterministic inference.

</details>

## Reproduce

**Run the same five tests on your own Spark, by hand or with a coding
agent.** Results count only under the
[rules](#rules-for-a-valid-session-and-a-win) below.

### Do it yourself

Build both revisions and the frozen client, then run three fresh sessions
per build, alternating, while sampling GPU memory.

Run these from the checkout root, in `bash`, on one Spark. The record names
of the five tests are ONE, SIXTEEN, LONG, FOLLOWUP and SHARED
([The tests](#test-names)).

**Build once**

1. Build the server revision you want to measure. To compare two builds,
   keep each revision in its own checkout and build both.

   ```sh
   cargo kernels
   cargo build --release --offline --workspace --exclude spark-rust-kernels
   ```

2. Build the frozen client once, from its own commit, next to the checkout.

   ```sh
   git worktree add ../vade-bench-1d80e7c 1d80e7c
   cargo +1.99.0 build --release \
     --manifest-path ../vade-bench-1d80e7c/tools/bench/Cargo.toml
   ```

3. Set the model and addresses. Change the first two lines for another
   model.

   ```sh
   MODEL=models/Qwen/Qwen3-8B
   ID=qwen3-8b
   URL=http://localhost:8080
   BENCH=../vade-bench-1d80e7c/tools/bench/target/release/vade-bench
   ```

**Run one session**

4. Start one default server and wait until it reports `ready`. The default
   launch listens on the local loopback address, port 8080. Startup time
   runs from launch to that first successful health check.

   ```sh
   ./target/release/spark-serve "$MODEL" & PID=$!
   until curl -sf "$URL/health" | grep -q '"status":"ready"'; do sleep 0.1; done
   ```

5. In a second terminal, sample GPU memory every 100 ms. Sum the server's
   processes in each sample.

   ```sh
   nvidia-smi --query-compute-apps=timestamp,pid,used_gpu_memory \
     --format=csv,noheader,nounits -lms 100 > memory.csv
   ```

6. Run the five tests in this order.

   ```sh
   C=(--model "$ID" --tokenizer "$MODEL" --repeats 1 --timeout-secs 900
      --budget-secs 1200 --server-pid "$PID")
   "$BENCH" "$URL" "${C[@]}" --only concurrency --concurrency 1 \
     --max-tokens 128 --output one.json
   "$BENCH" "$URL" "${C[@]}" --only concurrency --concurrency 16 \
     --max-tokens 128 --output sixteen.json
   "$BENCH" "$URL" "${C[@]}" --only long --long-target 28672 \
     --long-output 16 --output long.json
   "$BENCH" "$URL" "${C[@]}" --only followup --max-tokens 128 \
     --output followup.json
   "$BENCH" "$URL" "${C[@]}" --only shared-system --max-tokens 128 \
     --output shared.json
   ```

7. Stop the server, then stop the sampler with Ctrl-C.

   ```sh
   kill -TERM "$PID"
   ```

**Repeat, then decide**

8. Repeat steps 4 to 7 with a fresh server each time, alternating the two
   builds: A B A B A B. Use default launches only: no slot, budget,
   context, output or draft overrides.

9. Check each session against the
   [session rules](#rules-for-a-valid-session-and-a-win) before using it.

10. Decide the result with the
    [rules for a win](#rules-for-a-valid-session-and-a-win).

[Validation](validation.md) lists the exactness checks a change must also
pass.

### Ask your coding agent

```text
Reproduce a Vade benchmark on this DGX Spark by following the "Reproduce"
section of docs/benchmarks.md. Build both server revisions being compared
on this Spark, and build the frozen client at commit 1d80e7c. Alternate
three fresh default-server sessions per revision (A B A B A B). In each
session run ONE, SIXTEEN, LONG, FOLLOWUP and SHARED in that order while
sampling GPU memory every 100 ms. Keep every request status, input and
output token count, and full time. Report medians, ranges and peak bytes,
and label drafting on or off for every workload. Run as the current user,
never use sudo, and do not commit or push. Do not compare runs from
different Sparks, and do not call a preliminary result a win.
```

### Rules for a valid session and a win

**A session counts only when every request passed; a win over a rival needs
three sessions each, separate ranges and a 3% gap.**

A session is valid when:

- **Requests:** no request failed. Any failed request makes that rate
  invalid.
- **16-request fairness:** the slowest first text is at most twice the
  median.
- **Long prompt input:** exactly 28,672 tokens.
- **Natural end:** an answer that ends early is kept; record the actual
  output tokens and full time.

A result counts as better when:

- **Vade build against Vade build:** ranges separate and medians at least
  1% apart. For overlapping medians 1% to 3% apart, add two pairs and
  require four of five pairs to agree.
- **Vade against a rival:** at least three sessions each, ranges separate,
  medians at least 3% apart.

## Full data

**The precise record behind every table: each setting, exact bytes and
session ranges.** Record tables keep the record's own names for the tests.

<a id="what-was-compared"></a>

<details>
<summary><b>What was compared</b>: engines, versions, settings and request adapters</summary>

- **Vade** `79a11cc` (Windows A and B) and `03068e9` (Window C). Settings:
  default launch; `--draft plain`; minimum-memory settings (Window A). The
  default launch passes only the model and a local `--bind` address.
  Qwen3-8B and Qwen3-32B draft with the official Qwen3-0.6B; Qwen3-30B-A3B
  never drafts.
- **SGLang** 0.5.21, official arm64 image. Settings: stock; drafting with
  the same Qwen3-0.6B; deterministic; reduced pool. Deterministic uses
  Triton attention. The default deterministic attention (FlashInfer) failed
  Qwen3-8B LONG.
- **vLLM** 0.31.0 (`vllm/vllm-openai:latest`). Settings: stock; drafting
  with the same Qwen3-0.6B, three tokens; deterministic
  (`VLLM_BATCH_INVARIANT=1`); memory candidates; eager drafting retry for
  the expert model.
- **NGC vLLM** 25.09 (vLLM v0.10.1.1). Settings: stock; n-gram drafting;
  memory candidate. It has no batch-invariant switch. Its bundled docs warn
  that draft-model drafting is unsupported and document n-gram drafting.
- **TensorRT-LLM** 1.3.0rc13. Settings: stock memory pools; memory
  candidates. Input and sequence limits raised to 32,768 tokens. No
  drafting or deterministic configuration was measured.

On this page, a *tuned* setting or *memory candidate* is a rival setting
with a smaller memory pool, and the *smallest pool* is SGLang's reduced
pool. Rivals ran stock, with the same draft model where supported, and in
their available deterministic modes. Not every engine supported or finished
every setting. Nothing is inferred from a missing row.

**Request adapters.** The frozen client needs streamed token counts and
thinking turned off. Each rival got the smallest adapter that does that.
Kernels, scheduling and memory pools stay stock.

- **SGLang** adds `--stream-response-default-include-usage` and
  `--default-chat-template-kwargs '{"enable_thinking":false}'`. Without
  them, it fails the client's token-count checks.
- **vLLM** adds `--enable-force-include-usage` and the official chat
  template with thinking turned off. Even so, NGC vLLM 25.09 leaves token
  counts out when a request has no `stream_options`, so its ONE, SIXTEEN
  and LONG cells are invalid.
- **TensorRT-LLM** uses the same no-thinking template and
  `--max_num_tokens 32768 --max_seq_len 32768`. In Window B it rejects three
  extra fields in the frozen request body (`enable_thinking`, `priority`,
  `cache`), so its ONE, SIXTEEN and LONG cells are invalid. In Window C a
  small schema adapter accepts and ignores those fields. It does not change
  kernels, sampling, memory pools or the scheduler. Those rows say
  "API-adapted".

**Clients.** Vade, SGLang and vLLM use frozen client `1d80e7c`. Window C's
TensorRT-LLM rows use frozen client 4. The two clients send byte-identical
Qwen3 requests. Client 4 counts the answer tokens itself when a server
leaves the count out; the record tables flag those values (client). A
rejected request is a protocol limit, not an engine speed.

**Rival image identities.** All engines served official BF16 Qwen3 weights.
BF16 is a 16-bit number format.

- **SGLang** `lmsysorg/sglang:v0.5.21-cu130`:
  `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
- **NGC vLLM 25.09** (vLLM v0.10.1.1):
  `sha256:15f380ad9c32f0ac57ac16e4b778c6f733c88b9ffe3a936035d0a59ad17b1aab`
- **vLLM** `vllm/vllm-openai:latest`, installed 0.31.0:
  `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`
- **TensorRT-LLM 1.3.0rc13**:
  `sha256:2f0b65e1e33bbe30b352897bf2a1fb9e46120b869809fa5d2991d66c7b259fbe`

</details>

<a id="read-the-tables"></a>

<details>
<summary><b>How the numbers were measured</b>, and what the record flags mean</summary>

**Speed**

- Rates count every finished answer token over the whole test's time,
  including reading the prompt and waiting in a queue. A test in which any
  request fails produces no rate.
- The long prompt and follow-up also record the full time. That stops a
  server from looking fast by sending the first token early and the rest
  late.
- An answer may end early at its natural end. That is valid. When two
  engines return different token counts, their full times are not
  like-for-like.

**Memory**

- Memory is summed over the server's GPU processes, sampled every 100 ms
  with `nvidia-smi`. Its MiB values (1 MiB = 1,048,576 bytes) are converted
  to bytes.
- Samples are lower bounds. They miss some host and shared mappings.
- CPU memory (RSS) and GPU bytes overlap on the Spark's unified memory.
  Never add them.
- TensorRT-LLM keeps some weights in host or shared memory. How
  `nvidia-smi` counts that is unverified, so total-memory verdicts involving
  TensorRT-LLM are unverified.
- Window A's peak includes startup. Windows B and C show the highest test
  sample.

**Startup**

- Startup is a fresh process launch with existing file and compiler caches.
- Window A also measured controlled cold and warm starts for some rows:
  *cold* means the model files were dropped from the page cache first,
  *warm* means they were already in memory. Compiler caches were kept in
  both.

**Sessions and flags**

- **Sessions** counts fresh server starts. Medians are reported across
  sessions; memory is the maximum.
- Rows named *minimum memory* or *reduced pool* changed the memory capacity
  from the default.

<dl>
<dt>on, off</dt>
<dd>Drafting on or off for that test.</dd>
<dt>(client)</dt>
<dd>Token count from the client's own fallback, because the server left usage
out. It skips control tokens and the end-of-text token, so it can differ from a
server count.</dd>
<dt>(u)</dt>
<dd>Startup cache condition not controlled.</dd>
<dt>(overlap)</dt>
<dd>Startup overlapped another engine's aborted startup; not comparable.</dd>
<dt>invalid</dt>
<dd>The request did not meet the client's checks (missing token counts or a
rejected request body). Not a speed result.</dd>
<dt>failed</dt>
<dd>Queued or rejected by the server. Not a speed result.</dd>
<dt>n/a</dt>
<dd>Not measured.</dd>
</dl>

</details>

<details>
<summary><b>Window A notes</b>: Vade <code>79a11cc</code> against SGLang 0.5.21</summary>

Vade `79a11cc` against SGLang 0.5.21. The follow-up measurements below retain the initial Vade sessions.

**Sessions and scope**

- **Sessions.** Qwen3-8B Vade default and plain keep three initial
  sessions. Every other row has one session, so every SGLang comparison is
  preliminary. A fresh follow-up Vade default check stayed within 3% of the initial sessions.
- **Startup.** Vade default's controlled cold and warm starts come from
  the follow-up session.
- **Qwen3-32B FOLLOWUP** ends naturally at 52 tokens in every setting.
- **Qwen3-30B-A3B** was not finished in this window. No SGLang expert row
  exists in any window.

**Settings**

- **SGLang drafting** (STANDALONE) uses Qwen3-0.6B with
  `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.
  This is SGLang's generic smaller-draft recipe; no model-specific trained
  EAGLE draft was tested.
- **SGLang deterministic** adds
  `--enable-deterministic-inference --attention-backend triton`. With the
  default FlashInfer attention instead, Qwen3-8B LONG crashed allocating a
  16,777,216-byte workspace, and disabling prefill CUDA graphs did not fix
  it.
- **SGLang reduced pool** uses `--max-total-tokens 28956`: the largest
  prompt plus the requested output plus 8. It is the smallest value tried,
  not a proven minimum.
- **Vade minimum memory** uses two slots, so chats queue and follow-up reuse
  is lost. The low bytes cost a lot of throughput. Settings: Qwen3-8B
  default `--context-tokens 28948 --gpu-budget-mib 28503`; Qwen3-8B plain
  `--draft plain --context-tokens 28948 --gpu-budget-mib 26164` (one MiB
  less fails at that context); Qwen3-32B plain
  `--draft plain --context-tokens 28956 --gpu-budget-mib 79479`.
- **SGLang's own best settings.** Drafting gives its best ONE, SIXTEEN and
  FOLLOWUP full time, plus Qwen3-32B SHARED. Stock and reduced pool give its
  best LONG, plus Qwen3-8B SHARED.

</details>

<details>
<summary><b>Qwen3-8B, Window A</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 3 | 3.529 / 3.000 | 26.48 |
| Vade plain (`--draft plain`) | 3 | 2.815 (u) | 24.51 |
| Vade default, minimum memory | 1 | 3.010 (u) | 23.13 |
| Vade plain, minimum memory | 1 | 2.789 (u) | 21.86 |
| Vade default, follow-up check | 1 | 2.937 (u) | 26.48 |
| SGLang stock | 1 | 161.877 / 189.049 | 109.71 |
| SGLang drafting | 1 | 216.929 (u) | 109.95 |
| SGLang deterministic | 1 | 231.414 (u) | 110.66 |
| SGLang reduced pool | 1 | 172.721 (u) | 24.79 |

Startup shows cold / warm where both were measured; (u) means the cache
condition was not controlled. SGLang drafting is STANDALONE with
Qwen3-0.6B; SGLang deterministic uses Triton attention.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 23.468 tok/s | 5,454.3 | 18,320,719,872 |
| SIXTEEN, 2,048 tokens | 302.949 tok/s | 6,760.2 | 19,958,595,584 |
| LONG, 16 tokens | 8,277.500 ms | 9,661.2 | 24,668,798,976 |
| FOLLOWUP, 128 tokens | 195.600 ms | 11,251.8 | 24,836,571,136 |
| SHARED, 2,048 tokens | 151.976 tok/s | 13,475.8 | 26,480,738,304 |

**Vade plain**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 14.782 tok/s | 8,659.0 | 17,058,234,368 |
| SIXTEEN, 2,048 tokens | 221.794 tok/s | 9,233.8 | 17,986,224,128 |
| LONG, 16 tokens | 8,297.500 ms | 9,590.8 | 22,697,476,096 |
| FOLLOWUP, 128 tokens | 198.200 ms | 11,175.9 | 22,864,199,680 |
| SHARED, 2,048 tokens | 152.436 tok/s | 13,435.2 | 24,509,415,424 |

**Vade default, minimum memory**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 23.377 tok/s | 5,475.4 | 17,993,564,160 |
| SIXTEEN, 2,048 tokens | 50.207 tok/s | 40,791.4 | 18,241,028,096 |
| LONG, 16 tokens | 8,298.100 ms | 9,683.9 | 22,977,445,888 |
| FOLLOWUP, 128 tokens | 8,195.200 ms | 19,184.4 | 23,128,440,832 |
| SHARED, 2,048 tokens | 22.241 tok/s | 92,083.4 | 23,128,440,832 |

**Vade plain, minimum memory**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 14.788 tok/s | 8,655.6 | 16,820,207,616 |
| SIXTEEN, 2,048 tokens | 29.386 tok/s | 69,693.1 | 16,968,056,832 |
| LONG, 16 tokens | 8,298.200 ms | 9,594.7 | 21,704,474,624 |
| FOLLOWUP, 128 tokens | 8,211.800 ms | 19,120.3 | 21,855,469,568 |
| SHARED, 2,048 tokens | 22.238 tok/s | 92,094.4 | 21,855,469,568 |

**Vade default, follow-up**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 22.833 tok/s | 5,605.9 | 18,320,719,872 |
| SIXTEEN, 2,048 tokens | 299.897 tok/s | 6,829.0 | 19,958,595,584 |
| LONG, 16 tokens | 8,377.189 ms | 9,786.4 | 24,668,798,976 |
| FOLLOWUP, 128 tokens | 194.382 ms | 11,452.4 | 24,836,571,136 |
| SHARED, 2,048 tokens | 150.148 tok/s | 13,639.8 | 26,480,738,304 |

**SGLang stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 13.749 tok/s | 9,309.6 | 109,191,364,608 |
| SIXTEEN, 2,048 tokens | 223.873 tok/s | 9,148.0 | 109,256,376,320 |
| LONG, 16 tokens | 8,256.217 ms | 9,615.4 | 109,524,811,776 |
| FOLLOWUP, 128 tokens | 193.731 ms | 11,683.4 | 109,524,811,776 |
| SHARED, 2,048 tokens | 157.517 tok/s | 13,001.8 | 109,524,811,776 |

**SGLang drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 23.187 tok/s | 5,520.4 | 109,644,349,440 |
| SIXTEEN, 2,048 tokens | 296.127 tok/s | 6,916.0 | 109,686,292,480 |
| LONG, 16 tokens | 10,404.922 ms | 11,470.1 | 109,954,727,936 |
| FOLLOWUP, 128 tokens | 353.070 ms | 6,252.9 | 109,954,727,936 |
| SHARED, 2,048 tokens | 149.001 tok/s | 13,744.9 | 109,954,727,936 |

**SGLang deterministic**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 13.314 tok/s | 9,614.0 | 109,942,145,024 |
| SIXTEEN, 2,048 tokens | 203.399 tok/s | 10,068.9 | 109,984,088,064 |
| LONG, 16 tokens | 23,157.525 ms | 24,561.0 | 109,984,088,064 |
| FOLLOWUP, 128 tokens | 365.193 ms | 12,185.5 | 109,984,088,064 |
| SHARED, 2,048 tokens | 122.207 tok/s | 16,758.5 | 109,986,185,216 |

**SGLang reduced pool**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 13.718 tok/s | 9,330.6 | 24,265,097,216 |
| SIXTEEN, 2,048 tokens | 223.735 tok/s | 9,153.7 | 24,309,137,408 |
| LONG, 16 tokens | 8,313.348 ms | 9,678.3 | 24,577,572,864 |
| FOLLOWUP, 128 tokens | 198.305 ms | 11,738.1 | 24,577,572,864 |
| SHARED, 2,048 tokens | 159.632 tok/s | 12,829.5 | 24,577,572,864 |

</details>

<details>
<summary><b>Qwen3-32B, Window A</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 1 | 9.816 / 10.406 | 81.61 |
| Vade plain (`--draft plain`) | 1 | 8.811 (u) | 79.78 |
| Vade plain, minimum memory | 1 | 9.186 (u) | 75.18 |
| SGLang stock | 1 | 610.418 / 571.409 | 116.05 |
| SGLang drafting | 1 | 623.994 (u) | 115.74 |
| SGLang deterministic | 1 | 705.320 (u) | 116.50 |
| SGLang reduced pool | 1 | 575.371 (u) | 80.08 |

Startup shows cold / warm where both were measured; (u) means the cache
condition was not controlled. SGLang drafting is STANDALONE with
Qwen3-0.6B; SGLang deterministic uses Triton attention.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 6.911 tok/s | 18,520.3 | 67,940,384,768 |
| SIXTEEN, 2,048 tokens | 90.045 tok/s | 22,744.3 | 70,290,243,584 |
| LONG, 16 tokens | 32,507.030 ms | 37,621.4 | 78,149,320,704 |
| FOLLOWUP, 52 tokens | 600.706 ms | 17,504.4 | 78,551,973,888 |
| SHARED, 2,048 tokens | 38.859 tok/s | 52,703.6 | 81,608,572,928 |

**Vade plain**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.651 tok/s | 35,057.2 | 66,531,098,624 |
| SIXTEEN, 2,048 tokens | 55.349 tok/s | 37,001.4 | 68,156,391,424 |
| LONG, 16 tokens | 32,683.458 ms | 37,500.2 | 76,642,516,992 |
| FOLLOWUP, 52 tokens | 604.379 ms | 17,056.2 | 76,910,952,448 |
| SHARED, 2,048 tokens | 39.143 tok/s | 52,320.9 | 79,784,050,688 |

**Vade plain, minimum memory**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.675 tok/s | 34,825.4 | 66,138,931,200 |
| SIXTEEN, 2,048 tokens | 7.262 tok/s | 282,023.3 | 66,396,880,896 |
| LONG, 16 tokens | 32,953.328 ms | 37,752.8 | 74,908,172,288 |
| FOLLOWUP, 52 tokens | 32,874.328 ms | 49,113.2 | 75,177,656,320 |
| SHARED, 2,048 tokens | 5.764 tok/s | 355,332.9 | 75,096,915,968 |

**SGLang stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.608 tok/s | 35,474.4 | 114,352,455,680 |
| SIXTEEN, 2,048 tokens | 53.676 tok/s | 38,155.1 | 114,415,370,240 |
| LONG, 16 tokens | 33,173.852 ms | 37,791.2 | 114,954,338,304 |
| FOLLOWUP, 52 tokens | 783.128 ms | 16,454.1 | 114,954,338,304 |
| SHARED, 2,048 tokens | 41.669 tok/s | 49,149.5 | 114,956,435,456 |

**SGLang drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 6.302 tok/s | 20,311.2 | 114,964,824,064 |
| SIXTEEN, 2,048 tokens | 86.776 tok/s | 23,600.9 | 115,031,932,928 |
| LONG, 16 tokens | 34,919.287 ms | 37,592.8 | 115,568,803,840 |
| FOLLOWUP, 52 tokens | 869.805 ms | 10,727.6 | 115,568,803,840 |
| SHARED, 2,048 tokens | 51.410 tok/s | 39,836.8 | 115,570,900,992 |

**SGLang deterministic**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.338 tok/s | 38,342.2 | 115,534,200,832 |
| SIXTEEN, 2,048 tokens | 50.725 tok/s | 40,374.7 | 115,555,172,352 |
| LONG, 16 tokens | 81,563.751 ms | 86,586.8 | 115,555,172,352 |
| FOLLOWUP, 52 tokens | 1,238.769 ms | 18,279.9 | 115,555,172,352 |
| SHARED, 2,048 tokens | 33.916 tok/s | 60,384.5 | 115,559,366,656 |

**SGLang reduced pool**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.607 tok/s | 35,490.9 | 78,359,035,904 |
| SIXTEEN, 2,048 tokens | 54.423 tok/s | 37,630.9 | 78,417,756,160 |
| LONG, 16 tokens | 32,673.330 ms | 37,306.5 | 78,956,724,224 |
| FOLLOWUP, 52 tokens | 757.834 ms | 16,486.5 | 78,956,724,224 |
| SHARED, 2,048 tokens | 42.061 tok/s | 48,690.7 | 78,956,724,224 |

</details>

<details>
<summary><b>Window B notes</b>: Vade <code>79a11cc</code> against NGC vLLM, vLLM 0.31.0 and TensorRT-LLM (initial measurements)</summary>

Vade `79a11cc` against NGC vLLM 25.09, vLLM 0.31.0 and TensorRT-LLM
1.3.0rc13. Record: `sbvllmtrt` `SCOREBOARD.md`. Client `1d80e7c` for all
rows.

**Sessions and verdicts**

- **Sessions.** Only Qwen3-8B Vade default and NGC vLLM stock have three
  alternating sessions. Every other row is preliminary.
- **Verified against NGC vLLM stock on Qwen3-8B:** startup and FOLLOWUP
  full time are Vade wins; FOLLOWUP first text and SHARED are ties.
- **Qwen3-32B Vade default** combines two sessions: an early one paired with
  the stock rivals, and a late one paired with the drafting and memory runs.
  The combined row is not a gated comparison. The late session alone: ONE
  6.90 tok/s, SIXTEEN 89.70 tok/s, LONG 32,641.60 ms, FOLLOWUP 604.58 ms,
  SHARED 39.41 tok/s.
- **vLLM drafting's single-session Qwen3-8B SIXTEEN** (327.15 against
  304.85 tok/s) was rechecked in Window C with three sessions each: a tie.

**Settings and limits**

- **Memory candidates** are tested settings, not proven minimums. NGC vLLM
  and vLLM on Qwen3-8B use `--gpu-memory-utilization 0.20`. vLLM on
  Qwen3-32B uses `--max-model-len 29184 --kv-cache-memory-bytes 7654604800`.
  TensorRT-LLM on Qwen3-8B uses `--free_gpu_memory_fraction 0.06`.
- **N-gram drafting** on NGC vLLM is the record's `ngram 5 / lookup 4`
  setting.
- **Invalid cells** for NGC vLLM and TensorRT-LLM are protocol limits, not
  engine losses. The record keeps their raw times.
- **Qwen3-32B FOLLOWUP** ends naturally at 46 tokens on vLLM and NGC vLLM,
  and at 52 on Vade and TensorRT-LLM.
- **Not measured:** Qwen3-32B NGC vLLM memory candidate, vLLM deterministic
  and TensorRT-LLM memory candidate. Qwen3-30B-A3B was not started in this
  window.

</details>

<details>
<summary><b>Qwen3-8B, Window B</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 3 | 3.64 | 26.48 |
| Vade plain (`--draft plain`) | 1 | 2.71 | 24.51 |
| NGC vLLM 25.09 stock | 3 | 137.85 | 116.77 |
| NGC vLLM 25.09 n-gram drafting | 1 | 160.88 | 115.85 |
| NGC vLLM 25.09 memory candidate | 1 | 148.80 | 25.04 |
| vLLM 0.31.0 stock | 1 | 257.96 | 117.68 |
| vLLM 0.31.0 drafting | 1 | 277.99 | 117.42 |
| vLLM 0.31.0 deterministic | 1 | 204.92 | 118.68 |
| vLLM 0.31.0 memory candidate | 1 | 198.53 | 24.53 |
| TensorRT-LLM, raised limits | 1 | 205.22 | 93.02 |
| TensorRT-LLM memory candidate | 1 | 182.75 | 26.28 |

vLLM 0.31.0 drafting uses Qwen3-0.6B.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 23.77 tok/s | 5,385.7 | 18,320,719,872 |
| SIXTEEN, 2,048 tokens | 304.85 tok/s | 6,718.0 | 19,958,595,584 |
| LONG, 16 tokens | 8,267.70 ms | 9,632.4 | 24,668,798,976 |
| FOLLOWUP, 128 tokens | 195.47 ms | 11,109.0 | 24,836,571,136 |
| SHARED, 2,048 tokens | 153.88 tok/s | 13,308.9 | 26,480,738,304 |

**Vade plain**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 15.05 tok/s | 8,505.2 | 17,058,234,368 |
| SIXTEEN, 2,048 tokens | 224.69 tok/s | 9,114.6 | 17,986,224,128 |
| LONG, 16 tokens | 8,218.70 ms | 9,497.0 | 22,697,476,096 |
| FOLLOWUP, 128 tokens | 191.76 ms | 11,024.9 | 22,864,199,680 |
| SHARED, 2,048 tokens | 154.13 tok/s | 13,287.1 | 24,509,415,424 |

**NGC stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 9,561.5 | 116,770,471,936 |
| SIXTEEN, tokens n/a | invalid | 9,260.2 | 116,770,471,936 |
| LONG, tokens n/a | invalid | 12,923.3 | 116,770,471,936 |
| FOLLOWUP, 128 tokens | 193.73 ms | 11,900.0 | 116,770,471,936 |
| SHARED, 2,048 tokens | 156.38 tok/s | 13,096.2 | 116,770,471,936 |

**NGC n-gram drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 10,137.2 | 115,845,627,904 |
| SIXTEEN, tokens n/a | invalid | 9,566.3 | 115,845,627,904 |
| LONG, tokens n/a | invalid | 12,892.3 | 115,845,627,904 |
| FOLLOWUP, 128 tokens | 256.59 ms | 3,053.7 | 115,845,627,904 |
| SHARED, 2,048 tokens | 148.08 tok/s | 13,830.3 | 115,845,627,904 |

**NGC memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 9,529.7 | 25,041,043,456 |
| SIXTEEN, tokens n/a | invalid | 9,176.2 | 25,041,043,456 |
| LONG, tokens n/a | invalid | 12,783.2 | 25,041,043,456 |
| FOLLOWUP, 128 tokens | 194.47 ms | 11,893.8 | 25,041,043,456 |
| SHARED, 2,048 tokens | 157.16 tok/s | 13,031.0 | 25,041,043,456 |

**vLLM stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 13.38 tok/s | 9,569.5 | 117,405,908,992 |
| SIXTEEN, 2,048 tokens | 220.84 tok/s | 9,273.6 | 117,405,908,992 |
| LONG, 16 tokens | 9,025.18 ms | 10,301.2 | 117,682,733,056 |
| FOLLOWUP, 128 tokens | 256.06 ms | 12,187.7 | 117,682,733,056 |
| SHARED, 2,048 tokens | 148.38 tok/s | 13,802.5 | 117,682,733,056 |

**vLLM drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 21.79 tok/s | 5,875.5 | 117,137,473,536 |
| SIXTEEN, 2,048 tokens | 327.15 tok/s | 6,260.1 | 117,162,639,360 |
| LONG, 16 tokens | 11,542.72 ms | 12,784.0 | 117,422,686,208 |
| FOLLOWUP, 128 tokens | 440.86 ms | 6,746.0 | 117,422,686,208 |
| SHARED, 2,048 tokens | 128.30 tok/s | 15,962.1 | 117,422,686,208 |

**vLLM deterministic**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 14.02 tok/s | 9,129.9 | 118,402,056,192 |
| SIXTEEN, 2,048 tokens | 212.42 tok/s | 9,641.3 | 118,402,056,192 |
| LONG, 16 tokens | 9,508.39 ms | 11,220.7 | 118,678,880,256 |
| FOLLOWUP, 128 tokens | 247.97 ms | 16,162.4 | 118,678,880,256 |
| SHARED, 2,048 tokens | 139.82 tok/s | 14,647.5 | 118,678,880,256 |

**vLLM memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 13.65 tok/s | 9,375.0 | 24,250,417,152 |
| SIXTEEN, 2,048 tokens | 228.02 tok/s | 8,981.6 | 24,250,417,152 |
| LONG, 16 tokens | 8,689.23 ms | 10,045.2 | 24,527,241,216 |
| FOLLOWUP, 128 tokens | 203.26 ms | 11,771.4 | 24,527,241,216 |
| SHARED, 2,048 tokens | 154.67 tok/s | 13,241.2 | 24,527,241,216 |

**TensorRT-LLM**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 1.3 | 93,014,982,656 |
| SIXTEEN, tokens n/a | invalid | 14.3 | 93,014,982,656 |
| LONG, tokens n/a | invalid | 1.9 | 93,014,982,656 |
| FOLLOWUP, 128 tokens | 202.24 ms | 11,911.4 | 93,014,982,656 |
| SHARED, 2,048 tokens | 125.62 tok/s | 16,303.7 | 93,017,079,808 |

**TensorRT-LLM memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 1.8 | 26,272,071,680 |
| SIXTEEN, tokens n/a | invalid | 9.9 | 26,272,071,680 |
| LONG, tokens n/a | invalid | 1.6 | 26,272,071,680 |
| FOLLOWUP, 128 tokens | 202.67 ms | 11,923.3 | 26,272,071,680 |
| SHARED, 2,048 tokens | 108.93 tok/s | 18,801.5 | 26,276,265,984 |

</details>

<details>
<summary><b>Qwen3-32B, Window B</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 2 | 10.12 | 81.61 |
| Vade plain (`--draft plain`) | 1 | 8.99 | 79.78 |
| NGC vLLM 25.09 stock | 1 | 380.33 | 116.79 |
| NGC vLLM 25.09 n-gram drafting | 1 | 406.15 | 117.28 |
| vLLM 0.31.0 stock | 1 | 502.31 | 119.10 |
| vLLM 0.31.0 drafting | 1 | 559.65 | 118.44 |
| vLLM 0.31.0 memory candidate | 1 | 424.47 | 75.99 |
| TensorRT-LLM, raised limits | 1 | 480.98 | 82.34 |

vLLM 0.31.0 drafting uses Qwen3-0.6B.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 6.90 tok/s | 18,561.2 | 67,940,384,768 |
| SIXTEEN, 2,048 tokens | 89.76 tok/s | 22,817.2 | 70,290,243,584 |
| LONG, 16 tokens | 32,621.48 ms | 37,740.3 | 78,149,320,704 |
| FOLLOWUP, 52 tokens | 602.03 ms | 17,275.3 | 78,551,973,888 |
| SHARED, 2,048 tokens | 39.39 tok/s | 51,998.2 | 81,608,572,928 |

**Vade plain**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.67 tok/s | 34,836.0 | 66,531,098,624 |
| SIXTEEN, 2,048 tokens | 55.66 tok/s | 36,795.5 | 68,156,391,424 |
| LONG, 16 tokens | 32,539.10 ms | 37,333.5 | 76,642,516,992 |
| FOLLOWUP, 52 tokens | 602.22 ms | 16,940.0 | 76,910,952,448 |
| SHARED, 2,048 tokens | 39.46 tok/s | 51,896.0 | 79,784,050,688 |

**NGC stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 36,042.2 | 116,583,825,408 |
| SIXTEEN, tokens n/a | invalid | 37,519.8 | 116,583,825,408 |
| LONG, tokens n/a | invalid | 53,645.7 | 116,793,540,608 |
| FOLLOWUP, 46 tokens | 608.74 ms | 14,649.0 | 116,793,540,608 |
| SHARED, 2,048 tokens | 38.86 tok/s | 52,703.5 | 116,793,540,608 |

**NGC n-gram drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 36,077.3 | 117,282,177,024 |
| SIXTEEN, tokens n/a | invalid | 41,098.3 | 117,284,274,176 |
| LONG, tokens n/a | invalid | 53,902.1 | 117,284,274,176 |
| FOLLOWUP, 46 tokens | 624.29 ms | 14,650.0 | 117,284,274,176 |
| SHARED, 2,048 tokens | 37.19 tok/s | 55,062.4 | 117,284,274,176 |

**vLLM stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.58 tok/s | 35,705.3 | 118,527,885,312 |
| SIXTEEN, 2,048 tokens | 54.89 tok/s | 37,309.7 | 118,527,885,312 |
| LONG, 16 tokens | 34,721.66 ms | 39,300.5 | 119,098,310,656 |
| FOLLOWUP, 46 tokens | 640.30 ms | 14,786.3 | 119,098,310,656 |
| SHARED, 2,048 tokens | 40.98 tok/s | 49,972.6 | 119,098,310,656 |

**vLLM drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 6.27 tok/s | 20,401.0 | 117,852,602,368 |
| SIXTEEN, 2,048 tokens | 91.37 tok/s | 22,413.8 | 117,892,448,256 |
| LONG, 16 tokens | 37,681.57 ms | 41,327.7 | 118,441,902,080 |
| FOLLOWUP, 46 tokens | 1,097.30 ms | 13,065.8 | 118,441,902,080 |
| SHARED, 2,048 tokens | 43.81 tok/s | 46,745.1 | 118,441,902,080 |

**vLLM memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.60 tok/s | 35,540.4 | 75,989,254,144 |
| SIXTEEN, 2,048 tokens | 55.26 tok/s | 37,064.1 | 75,989,254,144 |
| LONG, 16 tokens | 34,672.95 ms | 39,316.1 | 75,989,254,144 |
| FOLLOWUP, 46 tokens | 575.95 ms | 14,518.9 | 75,989,254,144 |
| SHARED, 2,048 tokens | 41.41 tok/s | 49,456.6 | 75,989,254,144 |

**TensorRT-LLM**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, tokens n/a | invalid | 22.5 | 82,340,478,976 |
| SIXTEEN, tokens n/a | invalid | 34.9 | 82,340,478,976 |
| LONG, tokens n/a | invalid | 13.5 | 82,340,478,976 |
| FOLLOWUP, 52 tokens | 640.93 ms | 16,590.0 | 82,340,478,976 |
| SHARED, 2,048 tokens | 32.18 tok/s | 63,634.1 | 82,340,478,976 |

</details>

<details>
<summary><b>Window C notes</b>: Vade <code>03068e9</code> against vLLM 0.31.0 and TensorRT-LLM (follow-up measurements)</summary>

Vade `03068e9`, rebuilt on the test Spark, against vLLM 0.31.0 and
TensorRT-LLM 1.3.0rc13. Record: `sbvllmtrt` `SCOREBOARD-R2.md`. This round
is kept separate from Window B, not merged into it.

**Sessions and ranges**

- **Clients.** `1d80e7c` for Vade and vLLM; client 4 for the API-adapted
  TensorRT-LLM rows (see [What was compared](#what-was-compared)).
- **Sessions.** Qwen3-32B: one per setting. Qwen3-30B-A3B: Vade default
  three, every rival one, so every expert comparison is preliminary.
  Qwen3-8B: ONE and SIXTEEN only, three alternating pairs.
- **Qwen3-8B ranges.** ONE: Vade 23.88 to 24.03, vLLM drafting 22.59 to
  22.72 tok/s (verified Vade win). SIXTEEN: Vade 306.09 to 307.38 tok/s at
  19,958,595,584 peak bytes; vLLM drafting 300.70 to 325.66 tok/s at
  117,877,768,192 bytes. The ranges overlap, so SIXTEEN is a tie despite
  vLLM's higher median. Startup: Vade 3.49 to 3.79 s, vLLM 228.35 to
  241.63 s.
- **Qwen3-30B-A3B Vade ranges** (three sessions): ONE 33.18 to 33.81 tok/s,
  SIXTEEN 145.10 to 150.61 tok/s, LONG 7,820.46 to 7,845.66 ms, FOLLOWUP
  303.71 to 307.43 ms, SHARED 101.95 to 105.34 tok/s, startup 9.69 to
  10.12 s.
- **Expert Vade pairing.** Vade's three expert sessions each anchor
  different rivals: stock with sessions 1 and 2; drafting, deterministic and
  memory candidate with session 2; eager retry and TensorRT-LLM with session
  3. The combined Vade row is not a gated overall comparison.

**Failures and overlaps**

- **Expert vLLM drafting** passed ONE and SIXTEEN in graph mode, then failed
  LONG: a locked 7 MB expert workspace needed 8 MB, and the later workloads
  failed too. The eager retry (`--enforce-eager`, which runs vLLM without
  its captured GPU graphs) finished all five. Its first ONE request took
  39,864.3 ms (34,942.07 ms to first text) at 120,717,312,000 peak bytes;
  the cause of that delay is not established.
- **Failed candidate.** TensorRT-LLM's Qwen3-32B memory fraction 0.15
  setting shrank its sequence limit to 2,176 tokens, queued SIXTEEN and
  rejected LONG and SHARED. It is not a viable setting.
- **Startup overlap.** The Qwen3-32B TensorRT-LLM memory candidate's 427.77
  s startup overlapped an aborted rival startup for its first 40 seconds.
  Its timed workloads ran after cleanup.

**Settings**

- **Memory candidates.** vLLM on Qwen3-30B-A3B uses
  `--max-model-len 29184 --kv-cache-memory-bytes 2870476800` (98,304 BF16
  KV bytes per token; 29,200 token slots). TensorRT-LLM uses
  `--extra_llm_api_options` with `kv_cache_config.max_tokens=29216`.
  Minimum memory and total physical memory are unverified for every engine.
- **FOLLOWUP lengths.** Qwen3-32B ends naturally at 46 tokens on vLLM and
  TensorRT-LLM and 52 on Vade. Qwen3-30B-A3B ends at 46 on vLLM and 50 on
  Vade and TensorRT-LLM.
- **Startup** keeps existing file and compiler caches. Controlled cold
  startup was not measured in this window.

</details>

<details>
<summary><b>Qwen3-32B, Window C</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 1 | 18.10 | 81.61 |
| vLLM 0.31 deterministic | 1 | 454.26 | 120.57 |
| TensorRT-LLM stock, API-adapted | 1 | 470.73 | 82.28 |
| TensorRT-LLM memory, API-adapted | 1 | 427.77 (overlap) | 81.50 |
| TensorRT-LLM fraction 0.15 | 0 | 287.30 | 67.83 |

TensorRT-LLM memory is the API-adapted memory candidate; (overlap) means
its startup overlapped another engine's aborted startup. TensorRT-LLM
fraction 0.15 failed after ONE, so it has no complete session.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting on for ONE and SIXTEEN:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 6.93 tok/s | 18,468.6 | 67,940,384,768 |
| SIXTEEN, 2,048 tokens | 89.87 tok/s | 22,789.2 | 70,290,243,584 |
| LONG, 16 tokens | 32,573.59 ms | 37,688.8 | 78,149,320,704 |
| FOLLOWUP, 52 tokens | 606.70 ms | 17,308.6 | 78,551,973,888 |
| SHARED, 2,048 tokens | 39.37 tok/s | 52,015.1 | 81,608,572,928 |

**vLLM deterministic**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.30 tok/s | 38,845.6 | 119,997,988,864 |
| SIXTEEN, 2,048 tokens | 51.31 tok/s | 39,915.8 | 119,997,988,864 |
| LONG, 16 tokens | 38,846.62 ms | 44,671.4 | 120,568,414,208 |
| FOLLOWUP, 46 tokens | 650.30 ms | 18,709.6 | 120,568,414,208 |
| SHARED, 2,048 tokens | 38.75 tok/s | 52,848.4 | 120,568,414,208 |

**TensorRT-LLM stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens (client) | 3.55 tok/s | 36,042.5 | 82,273,370,112 |
| SIXTEEN, 2,048 tokens (client) | 53.42 tok/s | 38,339.6 | 82,275,467,264 |
| LONG, 16 tokens (client) | 36,516.00 ms | 41,192.0 | 82,275,467,264 |
| FOLLOWUP, 46 tokens | 655.11 ms | 14,650.0 | 82,275,467,264 |
| SHARED, 2,048 tokens | 32.21 tok/s | 63,579.0 | 82,275,467,264 |

**TensorRT-LLM memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens (client) | 3.50 tok/s | 36,561.6 | 81,499,521,024 |
| SIXTEEN, 2,048 tokens (client) | 53.16 tok/s | 38,522.4 | 81,501,618,176 |
| LONG, 16 tokens (client) | 36,789.38 ms | 41,534.7 | 81,501,618,176 |
| FOLLOWUP, 46 tokens | 650.77 ms | 14,839.0 | 81,501,618,176 |
| SHARED, 2,048 tokens | 32.19 tok/s | 63,631.7 | 81,501,618,176 |

**TensorRT-LLM fraction 0.15**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens (client) | 3.56 tok/s | 35,931.8 | 67,830,284,288 |
| SIXTEEN, tokens n/a | failed | n/a | 67,832,381,440 |
| LONG, tokens n/a | failed | n/a | 67,832,381,440 |
| FOLLOWUP, tokens n/a | failed | n/a | 67,832,381,440 |
| SHARED, tokens n/a | failed | n/a | 67,832,381,440 |

</details>

<details>
<summary><b>Qwen3-30B-A3B, Window C</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 3 | 9.84 | 68.77 |
| Vade plain (`--draft plain`) | 1 | 9.38 | 68.76 |
| vLLM 0.31 stock | 1 | 563.10 | 118.38 |
| vLLM 0.31 drafting | 0 | 595.18 | 119.83 |
| vLLM 0.31 drafting, eager retry | 1 | 511.99 | 120.72 |
| vLLM 0.31 deterministic | 1 | 480.27 | 120.98 |
| vLLM 0.31 memory candidate | 1 | 512.61 | 66.16 |
| TensorRT-LLM stock, API-adapted | 1 | 519.59 | 72.54 |
| TensorRT-LLM memory, API-adapted | 1 | 430.58 | 68.62 |

vLLM 0.31 drafting uses Qwen3-0.6B; it failed after SIXTEEN, so it has no
complete session, and its eager retry finished all five tests.
TensorRT-LLM memory is the API-adapted memory candidate.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 33.48 tok/s | 3,822.7 | 61,975,035,904 |
| SIXTEEN, 2,048 tokens | 149.19 tok/s | 13,727.7 | 62,587,404,288 |
| LONG, 16 tokens | 7,840.90 ms | 8,528.5 | 67,890,053,120 |
| FOLLOWUP, 50 tokens | 305.06 ms | 2,588.5 | 68,091,379,712 |
| SHARED, 2,048 tokens | 103.61 tok/s | 19,766.2 | 68,766,662,656 |

**Vade plain**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 33.21 tok/s | 3,854.8 | 61,975,035,904 |
| SIXTEEN, 2,048 tokens | 148.83 tok/s | 13,760.4 | 62,587,404,288 |
| LONG, 16 tokens | 7,868.53 ms | 8,563.9 | 67,890,053,120 |
| FOLLOWUP, 50 tokens | 308.14 ms | 2,611.2 | 68,091,379,712 |
| SHARED, 2,048 tokens | 102.00 tok/s | 20,077.5 | 68,761,419,776 |

**vLLM stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 28.44 tok/s | 4,500.4 | 118,050,783,232 |
| SIXTEEN, 2,048 tokens | 126.61 tok/s | 16,175.6 | 118,050,783,232 |
| LONG, 16 tokens | 9,725.51 ms | 10,244.5 | 118,380,036,096 |
| FOLLOWUP, 46 tokens | 364.65 ms | 2,321.6 | 118,380,036,096 |
| SHARED, 2,048 tokens | 122.70 tok/s | 16,690.8 | 118,380,036,096 |

**vLLM drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 26.57 tok/s | 4,817.8 | 119,776,739,328 |
| SIXTEEN, 2,048 tokens | 190.56 tok/s | 10,747.0 | 119,797,710,848 |
| LONG, tokens n/a | failed | n/a | 119,797,710,848 |
| FOLLOWUP, tokens n/a | failed | n/a | 119,831,265,280 |
| SHARED, tokens n/a | failed | n/a | 119,810,293,760 |

**vLLM drafting, eager**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 3.21 tok/s | 39,864.3 | 120,717,312,000 |
| SIXTEEN, 2,048 tokens | 168.02 tok/s | 12,189.3 | 120,717,312,000 |
| LONG, 16 tokens | 12,392.16 ms | 13,514.6 | 120,717,312,000 |
| FOLLOWUP, 46 tokens | 585.98 ms | 3,846.5 | 120,239,161,344 |
| SHARED, 2,048 tokens | 96.47 tok/s | 21,230.5 | 120,239,161,344 |

**vLLM deterministic**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 25.93 tok/s | 4,935.5 | 120,885,084,160 |
| SIXTEEN, 2,048 tokens | 112.38 tok/s | 18,223.6 | 120,885,084,160 |
| LONG, 16 tokens | 10,708.19 ms | 11,177.0 | 120,981,553,152 |
| FOLLOWUP, 46 tokens | 394.63 ms | 3,810.9 | 120,981,553,152 |
| SHARED, 2,048 tokens | 118.89 tok/s | 17,226.6 | 120,981,553,152 |

**vLLM memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 28.04 tok/s | 4,565.6 | 65,827,504,128 |
| SIXTEEN, 2,048 tokens | 136.41 tok/s | 15,013.9 | 65,827,504,128 |
| LONG, 16 tokens | 9,820.96 ms | 10,319.4 | 66,160,951,296 |
| FOLLOWUP, 46 tokens | 414.44 ms | 2,378.9 | 66,160,951,296 |
| SHARED, 2,048 tokens | 122.52 tok/s | 16,715.3 | 66,160,951,296 |

**TensorRT-LLM stock**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens (client) | 29.55 tok/s | 4,330.9 | 72,533,147,648 |
| SIXTEEN, 2,048 tokens (client) | 141.07 tok/s | 14,517.5 | 72,535,244,800 |
| LONG, 16 tokens (client) | 9,188.66 ms | 9,867.6 | 72,535,244,800 |
| FOLLOWUP, 50 tokens | 336.49 ms | 2,550.8 | 72,535,244,800 |
| SHARED, 2,048 tokens | 104.79 tok/s | 19,544.2 | 72,535,244,800 |

**TensorRT-LLM memory candidate**, drafting off:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens (client) | 29.30 tok/s | 4,368.8 | 68,612,521,984 |
| SIXTEEN, 2,048 tokens (client) | 135.91 tok/s | 15,068.3 | 68,616,716,288 |
| LONG, 16 tokens (client) | 9,199.68 ms | 9,881.0 | 68,616,716,288 |
| FOLLOWUP, 50 tokens | 336.81 ms | 2,563.7 | 68,616,716,288 |
| SHARED, 2,048 tokens | 103.01 tok/s | 19,881.3 | 68,616,716,288 |

</details>

<details>
<summary><b>Qwen3-8B, Window C</b>: every setting and recorded value</summary>

| Setting | Sessions | Startup (s) | Peak (GB) |
| --- | ---: | ---: | ---: |
| Vade default | 3 | 3.70 | 19.96 |
| vLLM 0.31 drafting | 3 | 239.23 | 117.88 |

Both rows have three sessions, for ONE and SIXTEEN only. vLLM 0.31
drafting uses Qwen3-0.6B.

Each setting below lists every test with its answer tokens, result, full
time and peak bytes.

**Vade default**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 23.89 tok/s | 5,358.3 | 18,320,719,872 |
| SIXTEEN, 2,048 tokens | 306.10 tok/s | 6,690.6 | 19,958,595,584 |

**vLLM drafting**, drafting on:

| Test | Result | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| ONE, 128 tokens | 22.69 tok/s | 5,640.0 | 117,852,602,368 |
| SIXTEEN, 2,048 tokens | 315.59 tok/s | 6,489.4 | 117,877,768,192 |

</details>

<details>
<summary><b>Exact values behind Later merged changes</b></summary>

- **exp1** (`79a11cc` to `53bc7b9`, three pairs): ONE 30.748 to 32.895 tok/s
  at 68,561,141,760 to 68,565,336,064 peak bytes. SIXTEEN 147.774 to 147.729,
  SHARED 101.536 to 101.677 tok/s; FOLLOWUP first text 294.005 to 309.491 ms,
  full time 2,710.156 to 2,619.242 ms; Qwen3-8B ONE 23.870 to 23.864 tok/s.
  Five-pair round: ONE 30.813 to 32.877 tok/s.
- **admit** (`03068e9` to `3911e2b`, one pair): FOLLOWUP first text
  194.485 to 129.653, 628.224 to 490.994 and 310.544 to 265.001 ms; peak bytes
  26,480,738,304 to 31,260,147,712, 81,608,572,928 to 89,610,256,384 and
  68,766,662,656 to 71,383,908,352. Qwen3-8B LONG 8,140.674 to 8,269.870 ms.
  Three-pair gate against `0f595e0`: 197.219 to 130.050, 612.077 to 485.668
  and 306.274 to 266.548 ms.
- **shared9** (`03068e9` to `39aceb2`, one pair): SHARED 151.80 to 167.16,
  39.11 to 43.49 and 100.50 to 120.93 tok/s. Qwen3-32B ONE 6.89 to 7.26 and
  SIXTEEN 90.09 to 93.93 tok/s (drafting on). Qwen3-30B-A3B SIXTEEN
  145.99 to 140.91 tok/s. FOLLOWUP first text 198 to 204 ms (Qwen3-8B) and
  311 to 325 ms (Qwen3-30B-A3B). Peak bytes 26,480,738,304 and 81,608,572,928
  unchanged; 68,761,419,776 to 68,766,662,656 on Qwen3-30B-A3B.
- **Gemma assistant draft:** ONE 3.717 to 8.312 tok/s, peak bytes
  62,128,128,000 to 63,105,400,832. SIXTEEN 47.730 tok/s plain at
  69,042,438,144 bytes and 47.762 tok/s with the assistant loaded at
  70,908,903,424 bytes.
- **Merge check `9c3fe9c`:** Gemma 4 31B ONE 8.281 tok/s (assistant on),
  SIXTEEN 47.253 tok/s (off), LONG first text 62,072 ms, peak 77,808,533,504
  bytes. Qwen3.8-27B BF16 ONE 7.069 tok/s, SIXTEEN 59.154 tok/s (off), peak
  58,809,384,960 bytes.

</details>

### Older measurements

[FP32-cache measurements](history/fp32-cache-measurements.md),
[long-context measurements](history/longctx-performance.md) and
[scale measurements](../crates/tp/MEASUREMENTS.md) keep earlier kernels,
precisions and clients. They do not measure current defaults and do not
belong in these rival tables.

[The benchmark client guide](../tools/bench/README.md) explains its JSON
output, token counting and memory scopes.

**Where next:** [Validation](validation.md) · [Models](models.md) ·
[Documentation index](README.md)
