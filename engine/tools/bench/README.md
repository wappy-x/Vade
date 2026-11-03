# Benchmark library for vade bench

**`vade bench` measures a chat server from the outside, over HTTP, the way an
app sees it.** It times the first token, the answer speed, many concurrent
requests, long prompts and follow-up turns.

The `vade-bench` crate is our own library, written against the Rust standard
library alone. It is linked into the CLI and builds no separate program. By
default, `vade bench` times several capped chats at once, and it can also run
cold (uncached) 8k prompt probes. It keeps the same metrics and JSON receipt
fields as before (a receipt is the saved record of a run). See the
[current command guide and the full table of old and new options](../../../docs/benchmarks.md#run-vade-bench).

The separate `kernel-bench --mhc-bytes X.ptx` command checks logical FP32
[Sinkhorn stream primitives](../../crates/model/MHC.md) on a CUDA host.
It uses synthetic inputs. It does not time a chat server or prove token IDs.

## Default timing benchmark

With `vade launch` or `vade up` already running, use the same `VADE_HOME` and
`VADE_CONFIG` in another terminal:

```sh
vade bench
```

The CLI discovers the selected installation's recorded endpoint, as
`vade verify` does. The benchmark reads the model ID from `/health` and writes
`./vade-bench.json`. If `/health` gives no model ID, it uses
`qwen3.8-flash-next-fp4-local`. Pass `--url URL` for another running server,
`--model ID` to pick the model yourself, or `--output PATH` to keep a separate
receipt.

Drafting is Vade's speculative decoding. By default, requests use whatever
drafting policy the server has. Only requests sent with `--draft plain` can
produce a drafting-off receipt, and neither `--draft plain` nor
`--draft strict` needs a server restart:

```sh
vade bench --draft plain --output vade-bench-plain.json
vade bench --help
```

Every request records `requested_draft_mode` (`server`, `plain` or `strict`);
the receipt header uses `decoding.request_mode` and labels drafting
`server-default`, `off` or `strict-requested`.

`decoding.server_mtp_depth` copies the depth that `/health` reports, word for
word, along with the field it came from. On the FP4-Local path this value
currently reads 0 even when drafting is on, and the receipt label says so.
Once `/health` reports the real depth, the receipt will show it with no change
to the benchmark. The receipt also records `server_build_sha` from `/health`.

`draft_observations` keeps the modes and depths that responses report,
separately; those reports do not prove that drafting was actually used. A
health depth of zero never turns a server-default measurement into a
drafting-off one. The client still checks any draft override you pass and
that the server identity stays the same. Replies with contradictory sampling
settings invalidate the rates.

The default timing requests use a bundled offline writing prompt with greedy
sampling, cache off and thinking off. They ignore the end-of-sequence token
(EOS) so every answer fills the fixed output budget. Completion counts come
from server usage. An early stop or a failed request means the whole group of
concurrent requests (the cohort) gets no rate. `--prompts FILE` replaces the
bundled prompt with one prompt per nonempty UTF-8 line. `--self-test` checks
streams and arithmetic locally.

## Retained development suite

The rest of this page covers the older development suite, which you select
with `vade bench --suite`. It reuses the existing HTTP, report, tokenizer and
memory-probe code, so those workloads stay available through the same command.
Its receipt format and prompt definitions differ from the default timing
protocol, and it does not replace a named frozen scoreboard client. Both paths
count real tokens, never streamed chunks.

<details>
<summary><b>Terms on this page</b>: TTFT, decode rate, aggregate rate, prefill, RSS</summary>

<dl>
<dt>TTFT</dt>
<dd>Time to first token: from opening the connection to the first piece of
answer text.</dd>
<dt>Decode rate</dt>
<dd>Answer tokens per second after the first token.</dd>
<dt>Aggregate rate</dt>
<dd>All answer tokens from a group of chats, divided by the group's total
time.</dd>
<dt>Prefill</dt>
<dd>Reading the prompt, before the first answer token.</dd>
<dt>RSS</dt>
<dd>Resident set size: the host memory a process holds, as Linux reports
it.</dd>
</dl>

</details>

## Before you compare anything

**The recorded scoreboards use a frozen client, not a fresh build of this
folder.** A frozen client is a pinned client build that stays unchanged.

<dl>
<dt>Frozen client <code>1d80e7c</code></dt>
<dd>The Qwen3 scoreboards in
<a href="../../docs/benchmarks.md">Benchmarks</a>.</dd>
<dt>Frozen client 4</dt>
<dd>Version 4 of the frozen client, used for Gemma and Qwen3.8 gate runs (the
pass/fail comparisons a change has to clear) and for TensorRT-LLM rows with
requests adapted to its API. Its Qwen3 request bodies are byte-identical to
<code>1d80e7c</code>.</dd>
<dt>A build of this checkout</dt>
<dd>Development only. It does not replace a named frozen client.</dd>
</dl>

The workloads differ too, so do not mix them in one comparison.
[Benchmarks](../../docs/benchmarks.md) records the versions, adapters, exact
arguments and the repeat and noise rules.

<details>
<summary><b>How the workloads differ</b></summary>

- The frozen client's SIXTEEN test (16 different requests sent at once)
  cycles prose, code and reasoning prompts, each with a reader suffix.
- This checkout's ordinary concurrency case repeats its prose prompt and adds
  a numbered closing line to each copy. Its shared-system case cycles the
  three base prompts (prose, code and reasoning), each with a reader suffix:
  a closing line that asks for a separate response for one numbered reader,
  so every chat's text is different.

</details>

## Run it

**Build the client, point it at a running server, read the table it prints.**

### Do it yourself

1. Build the client.

   ```sh
   # Run from the checkout root.
   cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
   ```

2. Start the server you want to measure in another terminal, for example
   `engine/target/release/spark-serve models/Qwen/Qwen3-8B`.

3. Run the development suite against it.

   ```sh
   cli/target/release/vade bench --suite http://localhost:8080 \
     --tokenizer models/Qwen/Qwen3-8B
   ```

4. Read the table it prints. Full results go to `bench-results.json`, saved
   after every case.

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, build the client with
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml.
spark-serve is already running with Qwen3-8B on loopback port 8080. Run
cli/target/release/vade bench --suite http://localhost:8080 --tokenizer
models/Qwen/Qwen3-8B --only concurrency --concurrency 16 --repeats 1
--max-tokens 128, then report aggregate_tokens_per_sec and any failed
request from bench-results.json. This is a development run, not a frozen
scoreboard. Do not commit or push.
```

[Benchmarks](../../docs/benchmarks.md) lists the exact commands behind the
recorded figures.

<details>
<summary><b>Tips for clean runs</b></summary>

Setting up:

- The URL can be the server's address, its `/v1` path or the full
  chat-completions URL.
- Run the client on the same Spark, to keep network delay out of the
  numbers.
- `--tokenizer` points at the model folder, so prompts are sized exactly.
  Setting `MODEL_DIR` does the same. Without it, long prompts are only roughly
  sized, and the true count comes from the server's usage report.
- For a gate run (a pass/fail comparison a change has to clear), leave the
  server's sizing on automatic and use the named frozen client, not this
  build.

Running:

- Development requests must fit the server's reported context and input and
  output limits.
- Choose longer client timeouts for long prompts.
- An answer can end early at the end-of-text token. Always check the actual
  token count and finish reason in the JSON.

</details>

## What the development suite runs

**Five cases, all run by `vade bench --suite` by default (`--only all`).**

- **Single chats** (`--only decode`). Prose, code-editing and reasoning
  prompts. 32 answer tokens, three cold runs each.
- **Chats at once** (`--only concurrency`). 1, 4, 8 and 16 prose chats, each
  with its own numbered closing line in this checkout. The frozen client
  mixes prose, code and reasoning prompts. This case can ask for more chats
  than the server has active slots; the time they spend queueing is included.
- **Follow-up with a saved session** (`--only cache`). A first turn, three
  follow-ups that reuse it, and the same follow-up sent fresh.
- **Priority** (`--only priority`). A short main chat, alone and then while a
  background chat reads an 8k prompt.
- **Long prompts** (`--only long`). Prompts of about 1,024, 8,192 and 28,672
  tokens, 2 answer tokens.

<details>
<summary><b>More modes</b>: follow-up, shared system prompt, sizes, request files</summary>

The two modes behind the recorded tables that compare Vade with other
engines:

- `--only followup --tokenizer MODEL --max-tokens 128 --repeats 1`. A
  28,672-token first turn, then the full conversation plus a short new
  question. Reports follow-up TTFT without any session.
- `--only shared-system --tokenizer MODEL --max-tokens 128 --repeats 1`. 16
  concurrent requests that share an 8,192-token system message, then differ.

Sizes and request files:

- `--only long --long-target 38912 --long-output 64`. One prompt of the given
  size.
- `--only cache --cache-target 28672 --max-tokens 64 --repeats 1`. The
  `cache` case with a 28k first turn.
- `--only long-output`. One prompt that asks for 2,048 answer tokens.
- `--only request --long-target 28672 --long-output 2 --output request.json`.
  Writes the long-prompt request body to a file without sending it.

The `cache` and `priority` cases use Vade's own request fields. The
`followup` and `shared-system` modes send only standard OpenAI fields.

</details>

## How it measures

**Real tokens, timed from the connection, with failures left empty rather
than estimated.**

- **TTFT** runs from opening the TCP connection to the first non-empty answer
  text.
- **Decode rate** leaves the first token out.
- **Aggregate rate** includes prefill and queueing. If any chat fails or
  times out, it is left empty rather than estimated.

<details>
<summary><b>The exact rules, and how results are reported</b></summary>

TTFT: headers, role-only updates, thinking text and token-ID-only events do
not count.

Decode rate:

```text
(answer tokens - tokens up to the first text) / (finish time - first text time)
```

The final usage count is required. It includes the end-of-text token.

Aggregate rate is every answer token from a group, divided by the group's
total time.

How results are reported:

- The printed table shows the first repeat. With several chats, it shows
  their median.
- Every sample and every failure is kept in the JSON.
- Each reply in the JSON records the arrival time of every token
  (`token_elapsed_ms`), so speed can be checked over parts of a long answer.
- A request that times out is recorded as unfinished, not as a measured TTFT.

</details>

## Settings

**Development-suite defaults are greedy (temperature 0), seed omitted,
thinking off.** Precision and Spark count are labels only; the client cannot
check them.

<details>
<summary><b>All options, with defaults</b></summary>

<dl>
<dt><code>--model ID</code></dt>
<dd>Model name sent and checked in replies. Default: <code>qwen3-8b</code>.</dd>
<dt><code>--precision NAME</code></dt>
<dd>Label recorded in the results. Default: <code>BF16</code>.</dd>
<dt><code>--sparks N</code></dt>
<dd>Spark count recorded in the results. Default: 1.</dd>
<dt><code>--label NAME</code></dt>
<dd>Run label recorded in the results. Default: <code>baseline</code>.</dd>
<dt><code>--output FILE</code></dt>
<dd>Results file. Default: <code>bench-results.json</code>.</dd>
<dt><code>--tokenizer DIR</code></dt>
<dd>Model folder used to size prompts. Default: <code>MODEL_DIR</code> if
set.</dd>
<dt><code>--timeout-secs N</code></dt>
<dd>Time limit per request. Default: 90.</dd>
<dt><code>--budget-secs N</code></dt>
<dd>Time limit for the whole suite. Default: 840.</dd>
<dt><code>--max-tokens N</code></dt>
<dd>Answer tokens per request, 2 to 8,192. Default: 32.</dd>
<dt><code>--repeats N</code></dt>
<dd>Repeats per case, 1 to 10. Default: 3.</dd>
<dt><code>--concurrency N</code></dt>
<dd>Run only this many concurrent requests, 1 to 32. Default: 1, 4, 8 and
16.</dd>
<dt><code>--long-target N</code></dt>
<dd>Long prompt size. Default: 1,024, 8,192, 28,672.</dd>
<dt><code>--long-output N</code></dt>
<dd>Answer length for the long prompts. Default: 2.</dd>
<dt><code>--cache-target N</code></dt>
<dd>First-turn size for the <code>cache</code> case. Default: the built-in prose
prompt.</dd>
<dt><code>--server-pid PID</code></dt>
<dd>Memory sampling, below. Default: off.</dd>
<dt><code>--memory-file FILE</code></dt>
<dd>Memory record, below. Default: off.</dd>
</dl>

- Every result records the model, precision, Spark count and all sampling
  settings.
- A reasoning prompt does not turn thinking on.

</details>

## Memory

**`--server-pid PID` samples the server's resident memory every 20 ms and
reads its lifetime peak from Linux `/proc`.** Missing values are written as
null, never as zero.

<details>
<summary><b>Memory files, and the optional CUDA allocation probe</b></summary>

The client's own memory is reported separately. `--memory-file FILE`
attaches a JSON memory record, such as the probe's output below, to each
result. Resident memory and CUDA allocations measure different things.
Neither alone is the server's full unified-memory use.

The probe counts successful `cuMemAlloc_v2`, `cuMemAllocManaged` and
`cuMemFree_v2` calls in the server process only. It passes every call
through unchanged and changes nothing on the system.

1. Build it on the Spark.

   ```sh
   rustc --edition=2024 --crate-type=cdylib -C opt-level=3 \
     engine/tools/bench/src/memory_probe.rs \
     -o engine/tools/bench/target/release/libbench_memory.so
   ```

2. Start the server with the probe loaded.

   ```sh
   VADE_BENCH_MEMORY_FILE="$PWD/server-memory.json" \
   LD_PRELOAD="$PWD/engine/tools/bench/target/release/libbench_memory.so" \
     engine/target/release/spark-serve models/Qwen/Qwen3-8B
   ```

3. Run the client with `--server-pid` and `--memory-file server-memory.json`.

Limits of the probe:

- It reports peaks over the server's whole life, not per case. For a clean
  peak per case, restart the server before each one.
- **It sees only those three calls.** Once one of Vade's KV caches (the
  attention state stored for a conversation) passes 2 MiB, it grows through
  CUDA virtual memory, which the probe does not see. Its totals leave that
  memory out.
- Writing the memory file adds a little time to each allocation, which can
  show up in TTFT. Steady decoding allocates nothing, so it is not affected.

These optional variables split the total into parts:

- `VADE_BENCH_WEIGHTS_BYTES`: size of the weight allocation, counted as
  weights.
- `VADE_BENCH_KV_ALLOCATION_BYTES`: size of one KV allocation, counted as KV
  cache.
- `VADE_BENCH_KV_ARENA_TOKEN_BYTES` and `VADE_BENCH_KV_MAX_TOKENS`: count
  allocations of a whole number of tokens, up to this many, as KV cache.

</details>

## Other servers

**The client works with any server that streams OpenAI-style chat replies.**
The `followup` and `shared-system` modes omit Vade's cache, session and priority
fields. They still send `top_k` and `repetition_penalty`, which are not standard
OpenAI fields; confirm the server accepts them. Changing only the URL is not a
fair run for the other cases.

<details>
<summary><b>Exactly what is sent, and other notes</b></summary>

- The `followup` and `shared-system` modes send only `model`, `messages`,
  `stream`, `stream_options.include_usage`, `temperature`, `top_p`, `top_k`,
  `repetition_penalty`, `presence_penalty`, `frequency_penalty` and
  `max_tokens`. Sampling is explicit greedy: temperature 0, top_p 1, top_k 0,
  repetition_penalty 1, presence_penalty 0 and frequency_penalty 0.
- They turn thinking off with `/no_think` in the message. For servers that
  need it, set `enable_thinking=false` in the template.
- The other cases add Vade's `cache`, `session` and `priority` fields.
- When the final usage event includes `vade`, the client requires a greedy
  `vade.sampling` echo (top_k 0 or 1, all other values as above). Missing or
  non-greedy sampling fields fail the reply. A reply without `vade` is accepted
  with `"vade": null` in its receipt and a stderr warning that sampling is
  unverified. The JSON report's `missing_sampling_echoes` counts all such
  replies, including background and unfinished requests. The `flash-bench`
  server gate, a separate pass/fail check, still rejects missing echoes.
- When a request times out, the client sends one extra byte. Vade reads that
  as a cancellation.
- It is not a general OpenAI SDK and does not speak TLS.

</details>

## Test it

**The tests use real local sockets. They are not a speed measurement.**

```sh
cargo +1.99.0 test --offline --manifest-path engine/tools/bench/Cargo.toml
```

<details>
<summary><b>What the tests and the client check</b></summary>

The tests check timing rules, final usage, chunked HTTP, split UTF-8
characters and an expired deadline.

The client exits with an error if a cached follow-up differs from the fresh
one, or on any HTTP or protocol error. Timeouts are recorded as unfinished
and do not fail the run by themselves.

</details>

## Paired kernel microbench

You only need this if you are changing GPU kernels; it is not a serving
benchmark.

`kernel-bench` times two sets of production GPU kernels against each other in
one process, for example a baseline (A) and a candidate (X). Each side being
timed is an arm. The kernels come as PTX, NVIDIA's GPU assembly code. This
tool shows where kernel time goes. It does not measure end-to-end serving
speed or check that answers stay exact.

### Build and run

This builds the independent Linux CUDA Driver client:

```sh
cargo build --release --offline --manifest-path engine/tools/bench/Cargo.toml --bin kernel-bench
```

It uses only the Rust standard library, plus our own kernel-parameter (ABI)
and SHA-256 code. Only the module that calls the CUDA driver, and the shared
kernel-parameter code, may use unsafe code; the rest of the crate denies it.

Run `kernel-bench --a A.ptx --x X.ptx` with complete production layer PTX
files. Both modules share one process, stream and fixture. A full run aims for
about five minutes, not counting the first JIT compile. The main options:

- `--list` prints every case with its kernel and shape.
- `--case NAME,...` selects cases.
- `--ms 5000` sets the measured GPU milliseconds per arm and case.
- `--self-test` checks the host-side SHA-256, statistics and shapes.

For GPU identity testing, pass the same PTX to both arms and repeat in three
separate processes.

The `fp8_w2_*` cases cover grouped FP8 decode gate and down projections.
The `fp8_prompt_gate-*` cases cover prompt row boundaries, rank-sliced widths
and larger synthetic routes. Their larger route counts are derived from an
FP4 capture; they are not a measured FP8 prompt distribution. Inputs and
weights vary by row and scale block. Each run checks every owned output and
poisoned padding or guard byte before timing; the reported floor counts
unique bytes and is a bandwidth bound, not a compute-throughput estimate.

For a paired gate/up candidate that halves the prompt gate grid, set
`KBENCH_FP8_PAIRED=x` (or `1`) for X, `a` for A, or `both` for both arms.
Leave it unset for the ordinary grid. Decode grids stay unchanged. Receipts
record both effective grids and the pairing flags so reversed A/X runs can
be reproduced.

### What the cases cover

- The dense attention projections (Q and O) and the GDN input and output
  projections run with M, the number of token rows in one launch, set to 1, 8
  or 16. GDN is GatedDeltaNet, the model's linear-attention layer. All three
  sizes use the low-latency `flash_bf16_mma_latency` kernel, as the product
  does for up to 16 rows.
- FP4 gate and down use ten selected experts at widths 1, 4, 8 and 16.
- Prompt cases use four 2048-row chunks with balanced top-10 routes over all
  512 experts.
- HC up, a hyper-connection kernel, and the full vocabulary head use their
  product layouts and grids. Hyper-connections keep four parallel residual
  streams in the model.

The operands are synthetic finite values at real shapes, not checkpoint
answers or measured model routing distributions. Every case rotates identical
weights through at least 1 GiB in contiguous arenas (above the 64 MiB
minimum), with at least two distinct physical copies so that cache placement
averages out.

### How it checks and times

Before timing, each copy's complete logical output is filled with a poison
pattern, the kernel is launched, and the output is checked to be finite. All
copies and both arms must have the same output SHA-256; the vocabulary head's
hashes also include its per-CTA (per GPU thread block) selection partials.
Any mismatch exits nonzero.

CUDA graphs replay only the selected kernel between two `%globaltimer` stamp
nodes. Each case warms both graphs for at least 250 ms before it collects
samples, so even small graphs get more than a few milliseconds for the clock
to settle. CUDA events bracket each graph replay. The arms alternate, and
their order reverses on successive samples. The small stamp and graph
scheduling cost is included in the timings; host allocation, upload and
hashing are not timed.

For each kernel, the JSONL output reports both timings, min/median/max, the
within-process CV (coefficient of variation: how widely the timings spread),
PTX and source hashes, stored bytes and the grade. Calculate the
process-to-process CV over the three process medians.

### Floor and grades

Floor time is the time to move the compulsory stored weight, input and output
bytes at 230 GB/s. Prompt cases also compute 2MNK / 90 TFLOPS and take the
larger floor. A kernel's percentage is its floor time divided by its measured
median time per launch, times 100, so 100 % would mean it ran exactly at the
floor. The grade follows from that percentage: A+ at 95 % or more, A at 85 %,
B at 65 %, C at 40 %, D at 15 %, and F below 15 %.

### FP8 MoE layer

`kernel-bench --a A.ptx --x X.ptx --fp8-moe CONFIG.json` times the W8A8
grouped expert layer instead of the case list. It reads the hidden size,
expert width, expert count, experts per token and layer count from the
checkpoint's `config.json` (inside `text_config` when present), fills all
experts with synthetic finite E4M3 weights, and runs the token counts in
`KBENCH_FP8MOE_M` (default `1,3,8,c16,c17,c31`; each 1 to 31, and a leading
`c` sends every token to the same ten experts, so 17 or more fill a second
sixteen-row plane) in two layouts: one GPU with every channel, and rank 0 of
two Sparks (local gate widths, including one-block and empty shards, and down
over its half of the outputs). Each arm replays `KBENCH_FP8MOE_RING` cold
route sets (default 16) in a CUDA graph; a stamp after every stage (route,
token quantize, gate and up, SwiGLU and quantize, down) splits the time. A
malformed or unknown setting is an error. Both arms must write the same routed
rows and, for every token and expert slot, the same E4M3 middle codes and
scales: for each route set alone, and again for each route set replayed as its
own graph after the timed replays.

`KBENCH_FP8MOE_X=fused` (or `pipeline`) gives arm X the fused gate mode,
`KBENCH_FP8MOE_XROUTE=small` (or `kernels`) its one-launch route for 2 to 31
tokens, and `KBENCH_FP8MOE_XENTRIES=GATE,DOWN` other gate and down entries with
the same parameters. `KBENCH_FP8MOE_A=fused` does the same for arm A.

For placement diagnosis, `VADE_KBENCH_ARENA_PAD_MIB` reserves 0..64 MiB
before each fixture; default zero. Allocation/upload stays outside timings.

### HC regression kit

[`HC-REGRESSION-CASES.txt`](HC-REGRESSION-CASES.txt) is the case list shipped
with the mHC regression kit. `hc_regression_kit_cases_exist` checks that each
name exists in this benchmark. Before packaging, build the main benchmark
too. Check the shipped list against both `--list` outputs. A case must use
the same kernel, dimensions, and fixture kind in both revisions.

From an extracted prebuilt kit on an authorized, released Spark:

```sh
sha256sum -c SHA256SUMS && ./kernel-bench --a A.ptx --x X.ptx \
  --case "$(cat HC-REGRESSION-CASES.txt)" --ms 100
```

Require identical A and X bytes for every case. Do not use `--allow-diff`,
`--det`, or `--order`. Keep full output and compare result hashes across
fresh processes and Sparks. Timing is incidental to this byte check.

**Where next:** [Benchmarks](../../docs/benchmarks.md) · [Server and API](../../docs/server.md) ·
[spark-serve](../../crates/serve/README.md)
