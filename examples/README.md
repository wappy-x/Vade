# Examples

This page lists short scripts that run against a real Vade engine. Each
script comes with the output that it printed.

Start an engine first. Run `vade launch` on the Spark, or `vade up` from a
controller ([INSTALL.md](../INSTALL.md)). Every script then talks to
`http://127.0.0.1:18080`, the default API address. To use another address,
set `VADE_API_URL`. `vade status` prints the address in use.

Nothing here needs a package install. The scripts use `sh`, `curl` and the
repository's Rust tools.

<div align="center">

| Script | What it shows | Runs without a Spark |
| :---: | :---: | :---: |
| [`curl-openai.sh`](curl-openai.sh) | The OpenAI-compatible API with plain curl: models, one answer, one streamed answer | No |
| [`curl-anthropic.sh`](curl-anthropic.sh) | The Anthropic Messages API with plain curl: token count, one answer, one streamed answer | No |
| [`openai_client.sh`](openai_client.sh) | A shell client: streaming, tool calling, and prompt caching across turns | No |
| [`openai_client.rs`](openai_client.rs) | The same client in Rust with only the standard library | No |
| [`mcp_word_count.rs`](mcp_word_count.rs) | A small MCP tool server for a first `vade mcp add` test; see [MCP tools](../docs/mcp.md#a-first-server-with-no-extra-install) | Yes |
| [`claude-code.sh`](claude-code.sh) | Claude Code pointed at Vade's Messages API | No |
| [`codex.sh`](codex.sh) | The Codex CLI pointed at Vade's Responses API, in a private Codex home | No |
| [`bench.sh`](bench.sh) | Reproduce the published one-Spark `vade bench` number | `--self-test` only |
| [`two-sparks.sh`](two-sparks.sh) | One engine across two Sparks, started from a controller, with the settings in [`two-sparks.conf`](two-sparks.conf) | Dry run only (the default) |

</div>

The outputs below come from earlier real runs:

- **One Spark.** curl and the former API client.
- **A second Spark.** The bench example.
- **A Mac.** Claude Code and Codex, through an SSH tunnel to the first Spark.

Every request leaves sampling unset, so the engine decodes greedily. The same
request with the same settings gives the same tokens on every run. Long JSON
lines are shortened with `…` where noted.

## Where these outputs come from

We recorded these outputs on 2026-10-25 on one DGX Spark (GB10, driver 580
series). It ran main `6cb46bf` with the v0.1.0 checkpoint
(`nvidia/Qwen3.8-Flash-Next-FP4-Local`, sha256 `d99ccbed…2716`).

The engine was started with the command line and settings that `vade launch`
uses ([cli/src/launch.rs](../cli/src/launch.rs)):

- **Context.** 32768 tokens.
- **Output.** 8192 tokens.
- **Drafting.** Chosen by the engine: strict, depth 3. The engine guesses up
  to three tokens ahead and keeps only the ones that the model confirms.
- **Prefix cache.** Off, unless a section says otherwise.

Claude Code 2.1.310 and Codex 0.164.0 ran on a Mac through an SSH tunnel to
that Spark. Every response also carries a `vade` object with the sampling
echo, drafting, prefill timing, build and checkpoint identity. Below, it is
shortened to `"vade":{…}`.

## curl against the OpenAI API

`sh examples/curl-openai.sh`

```text
== GET /v1/models
{"object":"list","status":"ready","data":[{"id":"qwen3.8-flash-next-fp4-local","object":"model","owned_by":"vade","context_length":32768,"max_model_len":32768,"max_output_tokens":8192,"input_modalities":["text"]}]}
== POST /v1/chat/completions
{"id":"completion-1","object":"chat.completion","created":1792935961,"model":"qwen3.8-flash-next-fp4-local","choices":[{"index":0,"message":{"role":"assistant","content":"Paris","refusal":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":21,"completion_tokens":2,"total_tokens":23,"prompt_tokens_details":{"cached_tokens":0}},"vade":{…}}
== POST /v1/chat/completions, streamed (server-sent events)
data: {"id":"completion-2","object":"chat.completion.chunk","created":1792935961,"model":"qwen3.8-flash-next-fp4-local","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null,"token_id":16}]}

: prefill

data: {… "delta":{"content":" "},"finish_reason":null,"token_id":220}]}
…  one event per token: "2", " ", "3", " ", "4", " ", "5"
data: {"id":"completion-2","object":"chat.completion.chunk","created":1792935961,"model":"qwen3.8-flash-next-fp4-local","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":10,"total_tokens":34,"prompt_tokens_details":{"cached_tokens":0}},"vade":{…"draft":"strict","mtp_depth":3,"draft_used":true…}}

data: [DONE]
```

Each streamed chunk carries the token ID it decoded (`token_id`), and the
last one carries the usage. `: prefill` is an SSE comment the engine sends
while it reads the prompt.

## curl against the Anthropic API

`sh examples/curl-anthropic.sh`

```text
== POST /v1/messages/count_tokens
{"input_tokens":61}
== POST /v1/messages
{"id":"msg-3","type":"message","role":"assistant","model":"qwen3.8-flash-next-fp4-local","content":[{"type":"text","text":"Paris"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":28,"output_tokens":2,"cache_read_input_tokens":0,"cache_creation_input_tokens":0},"vade":{…}}
== POST /v1/messages, streamed (server-sent events)
event: message_start
data: {"type":"message_start","message":{"id":"msg-4","type":"message","role":"assistant","model":"qwen3.8-flash-next-fp4-local","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":24,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}

event: ping
data: {"type":"ping"}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1"}}
…  one content_block_delta per token: " ", "2", " ", "3", " ", "4", " ", "5"
event: content_block_stop
data: {"type":"content_block_stop","index":0,"vade":{…}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":24,"output_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0},"vade":{…}}

event: message_stop
data: {"type":"message_stop","vade":{…}}
```

## API clients: streaming, tools and caching

Run `sh examples/openai_client.sh`. It builds the Rust JSON helper, which uses
only the standard library. It uses curl for HTTP or HTTPS. The loop prints each
streamed content delta. The Rust version uses `TcpStream` and the same JSON
parser:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml --example openai-client
```

Both accept `stream`, `tools` or `cache` as arguments. Both read
`VADE_API_URL` and `VADE_API_KEY_FILE`. The Rust example supports plain HTTP.
Use the shell example for HTTPS. The saved output below is from the former
client. Host tests check the replacements with mock responses. The
replacements have not been run on a Spark.

```text
[model] qwen3.8-flash-next-fp4-local at http://127.0.0.1:18080
== stream
Tiny silicon heart,
Holds a mind of vast design,
Whispers deep thoughts out.
[usage] prompt 26, completion 21
== tools
[finish_reason] tool_calls
[tool call] get_weather({"city": "Paris"})
[answer] It's currently 18°C in Paris with clear skies. 🌤️
== cache
[turn 1] prompt 2014 tokens, cached 0: 17
[turn 2] prompt 2040 tokens, cached 0: 19
```

The output above was recorded with the prefix cache off, so both counts are
0. The cache is now on by default on one Spark and on two Sparks (the tensor
split). `VADE_FLASHNEXT_PREFIX_CACHE=0` at the engine start, or a saved
`--cache-bytes 0`, turns it off. [The prefix
cache](../INSTALL.md#the-prefix-cache) explains both.

With the cache on, the second turn reads 1984 of its 2040 prompt tokens from
the cache. A second run of the script also finds the first turn's prefix in
the cache. The answers are the same with the cache on or off:

```text
$ sh examples/openai_client.sh cache
[model] qwen3.8-flash-next-fp4-local at http://127.0.0.1:18080
== cache
[turn 1] prompt 2014 tokens, cached 0: 17
[turn 2] prompt 2040 tokens, cached 1984: 19
$ sh examples/openai_client.sh cache
[model] qwen3.8-flash-next-fp4-local at http://127.0.0.1:18080
== cache
[turn 1] prompt 2014 tokens, cached 1984: 17
[turn 2] prompt 2040 tokens, cached 1984: 19
```

The request sends `prompt_cache_key`, which asks the engine to keep the
prompt for the next turn. Usage reports the reuse as
`prompt_tokens_details.cached_tokens` (and as `cache_read_input_tokens` on
the Messages API).

## Claude Code

`sh examples/claude-code.sh`

```text
Vade at http://127.0.0.1:18990 serves qwen3.8-flash-next-fp4-local with a 32768-token context.
A GPU executes thousands of simple threads simultaneously to perform the same operation across vast arrays of data in parallel, whereas a CPU is built to run a few complex, sequential, branch-heavy tasks as fast as possible.
```

The first line is the script's own check, printed to stderr. It shows
`127.0.0.1:18990` instead of the default `127.0.0.1:18080`. The reason is that
the recorded run set `VADE_API_URL` to the Mac's end of the SSH tunnel. The
answer comes from the same settings and prompt, run with `claude --bare -p`.
The script now runs that command.

Without `--bare`, Claude Code 2.1.310 failed on the same engine, also with a
clean `HOME`:
`API Error: 400 Unexpected role "system". Messages roles must be user or assistant`.
The engine accepts only user and assistant roles inside Messages `messages`.
We did not run `--interactive`, which opens `claude --bare`, for this page.
For the Claude Code setup, see the [client setup
guide](../harness/docs/passthrough-matrix.md#claude-code).

## Codex CLI

`sh examples/codex.sh` (Codex's `workdir`, `session id` and
`reasoning summaries` lines left out)

```text
Codex home: /private/var/folders/…/tmp.vjdVif8ex3/codex (config.toml selects provider vade, wire_api responses)
OpenAI Codex v0.164.0
--------
model: qwen3.8-flash-next-fp4-local
provider: vade
approval: never
sandbox: read-only
reasoning effort: none
--------
user
In one sentence: what does a GPU do that a CPU does not? Do not run any commands.
codex
A GPU performs massively parallel computation by executing thousands of simple threads simultaneously on specialized hardware, whereas a CPU is optimized for low-latency, sequential, general-purpose tasks.
tokens used
2,505
```

`vade-harness setup codex` writes a new Codex home. The home sets provider
`vade`, `wire_api = "responses"`, the engine's model and limits, and a
read-only sandbox. Its `launch.sh` starts Codex there in a clean environment.
The [client setup guide](../harness/docs/passthrough-matrix.md) covers
Hermes, OpenCode and Pi the same way.

## Reproduce a benchmark number

`sh examples/bench.sh`

The published one-Spark v0.1.0 receipt records a `vade bench` run with no flags
([README.txt](../bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/README.txt)).
Below is the same recipe on main `6cb46bf`, on another of our Sparks, from
`vade-bench.txt`. It ran as `vade bench --url`, because no installation was
recorded on that Spark:

```text
arm | concurrency | ok | median tok/s | range tok/s | TTFT p50/p95 ms | observed server bytes
single | 1 | ok=1/1 | 45.688 | [45.68828853963755,45.68828853963755] | 184.916/184.916 | device_bytes=93291113144
single | 2 | ok=2/2 | 79.643 | [79.64264100586249,79.64264100586249] | 263.218/263.380 | device_bytes=95730583132
single | 4 | ok=4/4 | 143.574 | [143.57363279033302,143.57363279033302] | 378.312/378.953 | device_bytes=100600368548
single | 8 | ok=8/8 | 176.792 | [176.79185811077582,176.79185811077582] | 807.857/808.961 | device_bytes=107977352636
single | 16 | ok=16/16 | 187.158 | [187.15779756268782,187.15779756268782] | 6210.416/11850.571 | device_bytes=107977352636

Published (v0.1.0): C1 45.466 | C2 79.747 | C4 143.598 | C8 185.144 | C16 189.042 tok/s
```

<div align="center">

| Chats at once | Published (v0.1.0) | This run (main `6cb46bf`) |
| :---: | :---: | :---: |
| 1 | 45.466 | 45.688 |
| 2 | 79.747 | 79.643 |
| 4 | 143.598 | 143.574 |
| 8 | 185.144 | 176.792 |
| 16 | 189.042 | 187.158 |

</div>

One to four chats land within 0.5 % of the receipt. At eight chats, the
engine chooses on each run which chats draft, so that rate moves more. The
benchmarks page reports 130.6 to 173.9 tok/s for the eight-chat card rate
across six Sparks ([known limits](../docs/benchmarks.md#known-limits)).
These are one run each, not a controlled comparison.

Without a Spark, `sh examples/bench.sh --self-test` checks the client
against its built-in mock and ends with:

```text
SELF-TEST PASS: local chunked/fragmented UTF-8 SSE; usage counts; c1/c8/c16; failed-cell null rates; sampling echo/fallback; three pinned 8k probes. Synthetic device_bytes=4096; no GPU.
SELF-TEST arithmetic: 512 tokens / 4.000 s = 128.000 tok/s; TTFT p50=200.000 ms p95=290.000 ms.
```

## Two Sparks

`sh examples/two-sparks.sh` (a dry run, which runs on any Mac or Linux computer)

```text
Dry run. No commands run, files written, or connections opened.
# no build manifest yet; artifact hashes below are placeholders until vade install
Launch plan: qwen3.8-flash-next-fp4-local
  Machines          spark-a, spark-b
  Context           32768 tokens
  Output limit      8192 tokens
  Drafting request  engine default
  Concurrent chats  engine-admitted
  Transport         RDMA
  Endpoint          http://localhost:18080/v1
Start planned for qwen3.8-flash-next-fp4-local on 2 Sparks. Run the same command without --dry-run to start it.
```

To run it for real:

1. Copy [`two-sparks.conf`](two-sparks.conf) and fill in your SSH
   destinations, link addresses and model paths.
2. Run `VADE_CONFIG=/your/copy sh examples/two-sparks.sh --live`.

The live run installs on both Sparks and starts them as one engine. This is
the tensor split: both Sparks work on every layer. The script then checks the
engine with `vade verify` and stops it with `vade down`. Along the way, it
does three more things:

- **Endpoint.** It unsets `VADE_API_URL`, so `vade verify` checks the engine
  at your config's `local_port`. It reads that endpoint from the
  `vade up --sparks 2 --dry-run` plan for its model-list check.
- **Receipt.** It writes the `vade verify` receipt to a new temporary folder
  and prints its path. This keeps the checkout clean for the next
  `vade install`.
- **Stop.** It runs `vade down` from a trap, also when a step fails or you
  press Ctrl-C.

We have not run `--live` for this page. Our published two-Spark results are
on the [benchmarks page](../docs/benchmarks.md).
[Hardware and network setup](../docs/network-setup.md) covers cabling and
addresses. [INSTALL.md](../INSTALL.md#two-and-more-sparks) covers the
controller config, RDMA and TCP.
