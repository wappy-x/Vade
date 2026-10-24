# Benchmark library for vade bench

**`vade bench` measures a chat server from the outside, over HTTP, the way an
app sees it.** It times the first token, the answer speed, many concurrent
requests, long prompts and follow-up turns.

The `vade-bench` crate is a first-party, standard-library-only library linked
into the CLI. It builds no separate executable. The default `vade bench`
protocol runs capped concurrent chats, with optional cold 8k probes,
while retaining its metrics and JSON receipt fields. See the
[current command guide and complete migration option table](../../../docs/benchmarks.md#run-vade-bench).

## Default timing benchmark

With `vade launch` or `vade up` already running, use the same `VADE_HOME` and
`VADE_CONFIG` in another terminal:

```sh
vade bench
```

The CLI discovers the selected installation's recorded endpoint, as
`vade verify` does. The benchmark reads the model ID from `/health` and writes
`./vade-bench.json`. If health omits a model ID, it uses
`qwen3.8-flash-next-fp4-local`. Supply `--url URL` for another running server,
`--model ID` to select the model explicitly, or `--output PATH` to retain a
separate receipt.

By default, requests use the server's drafting policy as served. Every request
records `requested_draft_mode` (`server`, `plain` or `strict`); the receipt header
uses `decoding.request_mode` and labels drafting `server-default`, `off` or
`strict-requested`. Only requests explicitly sent with `--draft plain` can
produce a drafting-off receipt. Neither override requires restarting the server.

`decoding.server_mtp_depth` copies the `/health` depth verbatim, with its source
field and the label "server-reported (known to read 0 on the FP4-Local path until
the health fix lands)". The receipt also records `server_build_sha` from `/health`.
The health fix will flow through without a benchmark change. `draft_observations`
retains response-reported modes and depths separately; these reports do not prove
actual draft use. Health depth zero never turns a server-default measurement into
a drafting-off measurement. Explicit request overrides and stable server identity
are still checked; contradictory sampling replies invalidate rates.

```sh
vade bench --draft plain --output vade-bench-plain.json
vade bench --help
```

The default timing requests use a bundled offline writing prompt with greedy
sampling, cache off and thinking off. They ignore EOS to generate the fixed
output budget. Completion counts come from server usage; early stops and
failed requests make the cohort ineligible for a rate. `--prompts FILE`
replaces the bundled prompt with one prompt per nonempty UTF-8 line.
`--self-test` checks streams and arithmetic locally.

## Retained development suite

The rest of this page describes the older development suite, selected with
`vade bench --suite`. Its existing HTTP, report, tokenizer and memory-probe code
is reused, keeping those workloads available through the same command. Its
receipt format and prompt definitions differ from the default timing protocol;
it does not replace a named frozen scoreboard client. Both paths count real
tokens, never streamed chunks.

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
folder.**

<dl>
<dt>Frozen client <code>1d80e7c</code></dt>
<dd>The Qwen3 scoreboards in
<a href="../../docs/benchmarks.md">Benchmarks</a>.</dd>
<dt>Frozen client 4</dt>
<dd>Gemma and Qwen3.8 gates, and API-adapted TensorRT-LLM rows. Its Qwen3
request bodies are byte-identical to <code>1d80e7c</code>.</dd>
<dt>A build of this checkout</dt>
<dd>Development only. It does not replace a named frozen client.</dd>
</dl>

The workloads differ too, so do not mix them in one comparison.
[Benchmarks](../../docs/benchmarks.md) records the versions, adapters, exact
arguments and the repeat and noise rules.

<details>
<summary><b>How the workloads differ</b></summary>

- The frozen SIXTEEN prompt set cycles prose, code and reasoning prompts,
  each with a reader suffix.
- This checkout's ordinary concurrency case repeats its prose prompt with
  reader suffixes. Its shared-system case cycles the three bases.

</details>

## Run it

**Build the client, point it at a running server, read the table it prints.**

### Do it yourself

1. Build the client.

   ```sh
   # Run from the checkout root.
   cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
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
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml.
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
- For default gates, leave server sizing automatic and use the named frozen
  client.

Running:

- Development requests must fit the server's reported context and input and
  output limits.
- Choose longer client timeouts for long prompts.
- An answer can end early at the end-of-text token. Always check the actual
  token count and finish reason in the JSON.

</details>

## What the development suite runs

**Five cases, all run by `vade bench --suite` by default (`--only all`).** Chats at once can exceed
the server's active slots; queueing time is included.

- **Single chats** (`--only decode`). Prose, code-editing and reasoning
  prompts. 32 answer tokens, three cold runs each.
- **Chats at once** (`--only concurrency`). 1, 4, 8 and 16 prose chats with reader
  suffixes in this checkout. The frozen client uses mixed bases.
- **Follow-up with a saved session** (`--only cache`). A first turn, three
  follow-ups that reuse it, and the same follow-up sent fresh.
- **Priority** (`--only priority`). A short main chat, alone and then while a
  background chat reads an 8k prompt.
- **Long prompts** (`--only long`). Prompts of about 1,024, 8,192 and 28,672
  tokens, 2 answer tokens.

<details>
<summary><b>More modes</b>: follow-up, shared system prompt, sizes, request files</summary>

The two modes behind the recorded rival tables:

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

**Development-suite defaults are greedy (temperature 0), seed omitted, thinking off.** Precision and
Spark count are labels only; the client cannot check them.

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
- **It sees only those three calls.** Once a Vade KV cache passes 2 MiB, it
  grows through CUDA virtual memory, which the probe does not see. Its totals
  leave that memory out.
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
  server gate still rejects missing echoes.
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

**Where next:** [Benchmarks](../../docs/benchmarks.md) · [Server and API](../../docs/server.md) ·
[spark-serve](../../crates/serve/README.md)

## Paired kernel microbench

`cargo build --release --offline --manifest-path engine/tools/bench/Cargo.toml
--bin kernel-bench` builds the independent Linux CUDA Driver client. It uses
only the standard library and existing first-party ABI/SHA code. The CUDA FFI
module explicitly permits unsafe code; the rest of this crate denies it.

Run `kernel-bench --a A.ptx --x X.ptx` with complete production layer PTX files.
Both modules share one process, stream and fixture. `--list` shows 26 cases;
`--case NAME,...` selects them; `--ms 5000` sets measured GPU milliseconds per
arm and case. A complete try targets about five minutes, excluding first JIT.
`--self-test` checks host SHA/statistics/shapes. For GPU identity testing, pass
the same PTX to both arms and repeat in three separate processes.

Dense Q/O/GDN-in/GDN-out shapes use M=1/8/16. M16 follows the product's
`flash_bf16_mma` dispatch because the latency entry only supports M<=8.
FP4 gate/down use ten selected experts at widths 1/4/8/16. Prompt cases use
four 2048-row chunks with balanced top-10 routes over all 512 experts. These
are synthetic finite physical operands at real shapes, not checkpoint answers
or measured model routing distributions. HC up and the full vocabulary head
use their product layouts and grids. Every case rotates identical weights
through at least 1 GiB in contiguous arenas (above the 64 MiB minimum),
with at least two distinct physical copies to average cache placement.

Each copy's complete logical output is poisoned, launched and checked finite
before timing. All copies and both arms must have the same output SHA-256;
head hashes include its per-CTA selection partials. Any mismatch exits nonzero.
CUDA graphs replay only the selected kernel between two `%globaltimer` stamp
nodes. CUDA events bracket graph replay; JSONL reports both timings per kernel,
min/median/max, within-process CV, PTX/source hashes, stored bytes and grade.
The small stamp/graph scheduling cost is included; no host allocation, upload
or hashing is timed. Arms alternate, reversing order on successive samples.
Process-to-process CV must be calculated over the three process medians.

Floor time is compulsory stored weight/input/output bytes / 230 GB/s; prompt
cases also use 2MNK / 90 TFLOPS and take the larger floor. Grades are A+ >=95%,
A >=85%, B >=65%, C >=40%, D >=15%, else F. This is attribution tooling, not an
end-to-end serving speed or answer-exactness qualification.

Each case warms both graphs for at least 250 ms before collecting samples,
so small graphs do not receive only a few milliseconds of clock settling.

For placement diagnosis, `VADE_KBENCH_ARENA_PAD_MIB` reserves 0..64 MiB
before each fixture; default zero. Allocation/upload stays outside timings.
