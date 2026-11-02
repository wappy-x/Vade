# Serve Qwen3.6-35B on one Spark

**History:** a dated single-Spark route that is not supported in v0.1.0; the supported model is in [INSTALL.md](../../INSTALL.md).

Use the official **Qwen3.6-35B-A3B-FP8** checkpoint on one NVIDIA GB10.
The engine runs on the Spark; the coding harness runs on your Mac against
its HTTP endpoint. This guide uses the current checkout and `spark-serve` directly. The legacy FP8 two-Spark Flash-Next
`vade up` route is a separate deployment.

## Prepare and build

Start with a working GB10 driver, idle GPU and the standard Spark unified
memory. Follow the [one-Spark prerequisites](../quickstart-one-spark.md#0-prerequisites-before-downloads)
and [pinned Rust setup](../quickstart-one-spark.md#1-install-the-pinned-rust-tools).
The small-model disk allowance on that page does not cover this checkpoint;
allow room for the complete official snapshot, build output and download state.
No second Spark, QSFP cable or RDMA configuration is needed.

Clone Vade as in [source setup](../quickstart-one-spark.md#2-get-the-source-and-build-the-engine).
In Bash, from that checkout's root, record the source and build on the Spark:

```sh
set -euo pipefail
git rev-parse HEAD
VADE_ROOT="$PWD"
(cd engine && cargo kernels && cargo +1.99.0 build --release --offline -p spark-serve)
```

Download the pinned official checkpoint with the repository's existing helper:

```sh
MODEL="$VADE_ROOT/models/Qwen/Qwen3.6-35B-A3B-FP8"
sh scripts/fetch-model.sh Qwen/Qwen3.6-35B-A3B-FP8 \
  95a723d08a9490559dae23d0cff1d9466213d989 "$MODEL"
```

The helper requires Rust 1.99.0, curl and network access during setup;
inference uses Rust and the NVIDIA driver. It verifies the official snapshot
and writes a sibling checksum manifest. Continue only after `verified … files`
and exit zero; rerun the same command after an interrupted download.
This guide makes no current blank-OS replay claim.

## Start and send a request

Run in the foreground, keeping this terminal open:

```sh
"$VADE_ROOT/engine/target/release/spark-serve" "$MODEL" \
  --context-tokens 131072 --output-tokens 32768 --max-active 2 \
  --draft plain --prefix-cache on --prefill-chunk 512
```

Embedded PTX is selected automatically. Wait for `ready`, check that the model
is `qwen3.6-35b-a3b-fp8`, and record the admitted context and memory budget.
The default listener is loopback, port 8080. In another Spark Bash terminal,
enter its local HTTP URL, without a trailing slash:

```sh
read -r -p 'Local server URL: ' URL
curl --fail --silent --show-error "$URL/health"
curl --fail --silent --show-error "$URL/v1/models"
curl --fail --silent --show-error "$URL/v1/capabilities"
curl --fail --silent --show-error "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d '{"model":"qwen3.6-35b-a3b-fp8","messages":[{"role":"user","content":"Explain a Rust borrow in one sentence."}],"max_tokens":128,"temperature":0,"top_p":1,"top_k":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":42,"enable_thinking":false,"draft":"plain","cache":true,"kv_checkpoint":true}'
```

Expect a complete answer and usage. Check effective sampling in the response;
omitted settings can inherit sampling from the checkpoint. Wait for health's
checkpoint `hash_state=ready` before collecting timings. Stop the foreground
server with Ctrl-C and wait for it to exit before another launch.

The tiled prompt attention, FP8 GEMM (matrix multiplication), expert-weight
reuse and prompt-copy improvements are automatic for supported geometry.
There is no speedup flag to enable. `--prefill-chunk` controls scheduling;
it does not select different arithmetic. Use the shown setting for this recipe.

## Reuse prompt work across agent turns

Send the full conversation each turn. The server compares the **rendered
token prefix**, including template controls, tools, thinking history and
messages. Matching text alone is insufficient when those fields change.
It restores exact retained KV and recurrent state, then processes the suffix.

There are two ways to find retained state:

- Automatic prefix reuse is on by default (`--prefix-cache on`). It matches
  eligible requests within the cache scope, which defaults to `api-key`.
  Short typed prompts may use pooled execution without automatic lookup;
  explicit checkpoints provide a separate retention path.
  `--prefix-cache off` disables automatic lookup/publication; explicit KV
  checkpoints remain available. `--prefix-cache-scope global` permits shared
  lookup across clients; use `api-key` for separate authenticated clients.
- Explicit prompt retention uses the request field `"kv_checkpoint":true`.
  Read `usage.vade.kv.retention` and `usage.vade.kv.checkpoint`; keep the returned
  `handle` privately. On the next full-conversation request send that value
  as `"kv_handle":"…"`, and request another checkpoint. This works with
  automatic prefix caching off. The handle is a bearer capability, not a
  transcript: the server still verifies the common token prefix.

`"cache":false` opts the request out of automatic reuse and explicit KV
checkpoint retention/reuse. `prompt_cache_key` is only a scheduling hint;
it does not retain state or authorize access. Inspect
`usage.prompt_tokens_details.cached_tokens` and `usage.vade.kv.reused_tokens`
to learn what was actually reused; do not infer a hit from a returned handle.

For typed hybrid prompts, retention keeps the four newest recurrent snapshots
at 2,048-token boundaries plus the final canonical prompt boundary/tail state.
A history rewrite can reuse the newest saved boundary before the change and
replay the suffix. An edit before all retained boundaries may require a cold
read. This is bounded checkpoint reuse, not arbitrary token-by-token rewind.
The internal retention boolean follows cache eligibility or a requested KV
checkpoint; there is no separate snapshot-retention CLI flag.

Retained state shares the session memory pool with live work. Pressure can
evict it; explicit checkpoints also have per-owner entry/byte quotas and a
five-minute expiry. Restarting the process loses the cache. A missing,
expired, malformed or evicted handle falls back to normal prompt processing;
another eligible cache entry can still provide a prefix.
Check `usage.vade.kv.reuse`, `retention` and `checkpoint.expires_at_ms` rather than
assuming a handle guarantees a future hit. Cache state is not saved-session
durability. Native checks must qualify cache retention and MTP rewinds for the
current build.

## Built-in MTP and draft depth

For one chat, stop the plain server and replace its launch with:

```sh
"$VADE_ROOT/engine/target/release/spark-serve" "$MODEL" \
  --context-tokens 131072 --output-tokens 32768 --max-active 1 \
  --mtp --draft strict --draft-depth 2
```

| Control | Meaning for this 35B server |
| --- | --- |
| `--mtp` | Load the checkpoint's built-in multi-token predictor. |
| `--draft-depth 1..4` | Configure proposal depth; every proposal is verified by the target. |
| `--draft-depth 0` | Adaptive depth (the default), not MTP off; the reported ceiling can exceed the depth actually used. |
| `--draft plain` | Decode without drafting; omit `--mtp` too to avoid loading unused draft weights. |
| `--max-active 1` | Required for typed MoE MTP; multiple active chats are refused at startup. |

For the MTP request, omit the earlier JSON `"draft":"plain"` or change it to
`"draft":"strict"`. Check response `draft_used`, `model_draft_used` and
`mtp_depth`: configured depth alone does not show that drafting ran.
External `--draft-model` is unsupported with typed MoE MTP; fast/sampled draft
modes are refused. Seeded sampling is supported with canonical verification.
The separate Flash-Next CLI uses `vade up --mtp 0..3`; that is not the
`spark-serve` syntax. Use the [benchmark method](../benchmarks.md) to compare
proposal depths; deeper proposals are not a promised speed gain.

## Templates and the Mac coding harness

Hybrid chat formatting reads recognized clauses from the checkpoint's
`tokenizer_config.json`, independently of its model name or tensor sizes.
Known official variants cover Qwen3.5 (including the small 0.8B/2B default
thinking-off templates), Qwen3.6 and Qwen3.8/Flash-Next. The larger templates
default thinking on. `--default-thinking on|off` overrides the server default;
request-level thinking/effort controls take precedence. Unknown template
clauses or an altered skeleton fail closed. This is not a general Jinja
interpreter or a promise that every model with a recognized template fits
or has a supported execution path.

On the Mac, forward the Spark's loopback listener using your private SSH
configuration. From a Mac checkout of the same revision, build the harness
and enter the forwarded HTTP URL and your coding workspace:

```sh
cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml
read -r -p 'Forwarded server URL: ' URL
read -r -p 'Coding workspace: ' WORKSPACE
harness/target/release/vade-harness agent --endpoint "$URL" \
  --model qwen3.6-35b-a3b-fp8 --workspace "$WORKSPACE" \
  --session work --approval ask
```

Exit the harness before stopping the server. To resume, repeat the harness
command with `--resume work` in place of `--session work`. Resume loads the
saved journal and frozen rules with compatible model/tool definitions; it
does not rerun historical tools or restore files. Re-read files before acting
on old results. Journals persist locally; server KV reuse remains optional.

Automatic compaction now considers oversized reasoning and refused edit
proposals, including recovery from a context stop after resume/Continue.
It keeps original user instructions, stable system/tools and confirmed
command receipts while recording a lossy summary; raw history stays in the
journal. Safe recovery must still fit a useful reply or it stops. Inspect
`/context`, `/compaction`, or use `/compact` and `/new` when needed.
`--no-auto-compact` disables automatic compaction; `--prewarm-compaction`
opts into a discarded one-token cache warm after compaction. A rewritten
transcript may need to rebuild cache state.

`--reply-tokens N` bounds estimated streaming prose/reasoning. A started tool
call finishes atomically, so this is neither a hard tool-output cap nor a
wall-clock deadline. Malformed edits still require correction and approval;
compaction does not relax the patch grammar. These harness fixes have host
regressions; they do not establish a new real-model coding-task pass rate.

## Limits and troubleshooting

- Native prefill timing excludes model load, HTTP, queueing and tokenization.
  Measure the complete request when evaluating interactive latency.
- Memory depends on context, output room, live sessions and retained state.
  If admission refuses, lower `--context-tokens` or `--max-active` and read
  the reported budget. `--output-tokens` is a cap, not extra context.
  `--gpu-budget-mib` and `--session-budget-mib` set MiB ceilings;
  `--kv-budget-mib` caps KV payload, not a dedicated retention allocation.
  They cannot manufacture free memory. Automatic sizing is the starting point.
- Public typed-hybrid multi-Spark serving, including 122B world 2, remains
  refused. Native tensor-parallel work does not establish HTTP support.
  Flash-Next has a separate serving implementation and deployment guide.
- Nemotron is not supported by this base's model cartridges/templates.
  Template recognition, checkpoint inspection and successful loading alone
  do not qualify a new model.
- Native coding commands require macOS. Linux can serve the engine; it does
  not gain the Mac harness's command execution support by running this guide.
