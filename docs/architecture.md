# How Vade works

Vade is one project: an inference engine and an agent harness built together for
NVIDIA DGX Spark. The engine runs the language model. The harness turns model
replies into a conversation that can read files, propose edits and run approved
commands. The CLI builds, starts and stops the parts you select.

You can use the engine from another compatible client, or use the harness with
another supported HTTP endpoint. They remain separate programs and separate
Cargo workspaces within this repository.

For a first response, follow [INSTALL.md](../INSTALL.md). For measured results and
their limits, read [Benchmarks](benchmarks.md).

## The three parts

| Part | Owns | Does not own |
| --- | --- | --- |
| [Engine](../engine/README.md) | Model loading, tokenization, GPU execution, sampling, request scheduling and model-state caches | Workspace tools or permission to edit your files |
| [Harness](../harness/README.md) | Conversation history, terminal UI, tools, approvals, private session journals and client transport | Model weights, GPU execution or the engine's cached state |
| [CLI](../cli/README.md) | Acquisition/build workflow, configuration, owned processes and startup/shutdown records | Model arithmetic or the agent's tool decisions |

All three use first-party Rust and the standard library. The engine also uses
the CUDA driver and Vade's GPU kernels. They share protocols and development
rules, rather than one combined executable.

## Follow one coding request

```text
you
 │ task and chosen workspace
 ▼
harness ── HTTP request ──► engine controller ──► GPU ranks
   ▲                              │                 │
   └──── streamed answer or tool proposal ◄─────────┘
   │
   ├─ validate proposal, apply approval policy, execute workspace tool
   └─ send the recorded tool result back to the engine for the next reply
```

1. **The harness prepares the conversation.** It includes the task, retained
   messages, tool definitions and fixed request options. A tool result becomes
   part of the next request; the engine does not execute the tool itself.
2. **The client checks what the engine supports.** The optional
   `vade.engine.v1` contract advertises the model, context limits, tools, thinking
   and cache features. Exact admission renders and tokenizes the complete final
   request, then checks whether its prompt plus reserved output fits. This check
   reserves no GPU memory or queue slot; generation can still be refused.
3. **The engine accepts and schedules generation.** It validates limits,
   accounts for available execution memory and looks for a reusable prompt
   prefix. An affinity key helps choose cached state; the key alone does not
   prove that any tokens were reused.
4. **The tokenizer turns the conversation into model input.** The model's chat
   template supplies role and tool framing. A token is a small piece of text
   represented by an integer. The checkpoint, tokenizer, template, tool order
   and thinking settings all affect the input.
5. **The engine reads, then writes.** Reading the input is called *prefill*.
   Producing the reply is called *decode*. Decode updates the chat's model state,
   selects the next token under the effective sampling settings and converts
   tokens back into text. The server streams text, reasoning or tool-call data
   as it becomes available. Completion and usage fields let the client check
   the result; absent telemetry remains unknown.
6. **The harness validates the completed reply before continuing.** Displayed
   deltas are provisional. A complete tool proposal must pass the workspace
   rules and approval policy before execution. Tool outcomes are recorded in a
   private journal before the next model turn. Resume does not blindly execute
   old tools again, and an incomplete tool proposal is not executed.

The [capability, admission and cache contract](../harness/docs/engine-contract.md)
defines negotiation and refusal behavior. The [tool guide](../harness/docs/agent-tools.md)
defines tool arguments, limits and partial-write receipts. The engine's
[HTTP server](../engine/crates/serve/README.md) and
[scheduler](../engine/crates/serve/src/scheduler.rs) implement the serving side.

## What the current Flash-Next path runs

The current release path is `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one Spark
(world 1), served by `flash-serve`. Checkpoint metadata selects the mixed-weight
`NextModel` and its world-one row execution; `--fp4-local` explicitly requires
that checkpoint format. Use `vade launch` on the Spark, or the configured
controller route `vade up --fp4-local`. See [INSTALL.md](../INSTALL.md) and the
dated measurements in [Benchmarks](benchmarks.md).

`next-http` is a compatibility launcher for the separate tensor-map rank path;
it delegates to `flash-serve` with drafting disabled. `next-tp` runs native rank
qualification cases rather than the default HTTP service. Their presence does
not change the one-Spark release default. See the
[serving entry point](../engine/crates/tp/src/bin/flash-serve.rs),
[compatibility launcher](../engine/crates/tp/src/bin/next-http.rs) and
[rank qualification tool](../engine/crates/tp/src/bin/next-tp.rs).

Flash-Next combines attention, linear recurrent layers and routed experts. Full
attention retains keys and values from earlier tokens. The recurrent layers
retain a smaller evolving state. A router chooses expert networks for each token;
it does not read every expert's weights on every decode step.

FP4-Local describes mixed checkpoint storage, not an assertion that all tensors
or operations use four-bit arithmetic. The loader reads each module's NVFP4,
MXFP8 or plain storage from the checkpoint's quantization map and validates its
tensor inventory. See the [mixed-storage loader](../engine/crates/model/src/flash/mixed.rs).
The older two-Spark FP8 route remains a legacy configuration, with its own
[arithmetic record](../engine/docs/flash-fp8-math.md) and
[installation notes](install-legacy-fp8.md).

Multi-token prediction (*MTP*) proposes tokens for the target model to verify.
The default FP4-Local service uses strict drafting with request depth 3. Per-request
`"draft":"strict"` selects drafting ON and `"draft":"plain"` selects OFF;
`"mtp_depth":0` also selects plain decoding when no `draft` mode is named.
For a chat alone, the adaptive chooser can use the resident ceiling: 5 on one
Spark and 3 on the layer pipeline. Prompt lookup proposals are enabled by
default, and world 1 also enables cohort drafting control for concurrent chats.
Actual use depends on the request and scheduling policy; server configuration
alone does not prove that a request drafted. See
[drafting defaults](../engine/crates/engine/src/flash/draft_stack.rs),
[request handling](../engine/crates/serve/src/protocol.rs) and
[cohort policy](../engine/crates/serve/src/mtp_batch.rs).

For FP4-Local, a request that names no sampling fields defaults to greedy
decoding. Naming any sampling field selects the checkpoint's base sampling
settings before the named overrides are applied. Prefix caching is OFF by
default; `VADE_FLASHNEXT_PREFIX_CACHE=1` is supported only at world 1. See the
[sampling parser](../engine/crates/serve/src/protocol.rs) and
[cache switch](../engine/crates/serve/src/prefix_switch.rs).

## Ranks, memory and ownership

A *rank* is one engine process in the group of Sparks serving a model. Rank zero
owns the HTTP listener and coordinates model commands. The other ranks execute
those commands with their local weights and state. Together they compute one
answer; they are not independent model replicas.

At world 2, FP4-Local defaults to the `VADE_W2_PP` layer pipeline: each rank
executes its assigned contiguous layers with world-one arithmetic, and passes
the intermediate state to the next stage. The front owns HTTP and scheduling;
workers execute their layers and relay downstream. Draft verification crosses
the pipeline. `VADE_W2_PP=0` explicitly selects the older tensor-map rank path.
For more than two Sparks, `VADE_W2_PP=1` opts into an N-stage chain; the controller
sets this when `vade up --fp4-local --sparks N` selects a larger group. Supported
counts and memory admission still apply. See
[pipeline selection](../engine/crates/tp/src/flash/settings.rs),
[stage geometry](../engine/crates/engine/src/flash/next_pp.rs) and
[controller selection](../cli/src/app.rs).

The pipeline uses RDMA by default, with a TCP option. RDMA moves data through
registered memory between machines. Startup verifies artifacts and the ranks'
numerical settings before serving; fixed arithmetic and communication order
matter because changing floating-point addition order can change the answer.
See [pipeline startup](../engine/crates/tp/src/flash/next_batched_service.rs) and
[settings agreement](../engine/crates/tp/src/pipeline_settings.rs).

The engine separates these lifetimes:

| Memory | Owner and lifetime |
| --- | --- |
| Resident weights | One copy of the rank's share of model weights, shared by its chats until engine shutdown |
| Active chat state | The session's attention cache, recurrent state and execution buffers |
| Retained prompt state | Engine-owned prefix cache and scoped checkpoints, subject to budgets and eviction |
| Shared scratch | Model/decode workspaces reused in the engine's execution order |
| Host staging and transport | Loading and registered communication buffers, accounted for separately from model state |

Memory admission uses bounds agreed across the ranks so a request can fit on
every rank. Idle cache can be reclaimed to fund active sessions. Closing a chat
does not necessarily evict an independently retained prefix. Cached tokens still
count toward logical context length; reuse saves work rather than increasing the
model's context limit.

Spark CPU and GPU allocations draw from unified memory. Tracked device-allocation
bytes are useful for comparing engine builds, but they do not include every
driver allocation or establish total physical memory consumption. Benchmark
receipts state which memory measure they report.

## Exactness and model quality

Vade's exactness requirement is: **the same setup and request produce identical
token IDs and answer bytes on every Spark, every run**. The setup includes the
build, checkpoint, tokenizer/template, rank layout, arithmetic, thinking and
effective sampling configuration. A particular machine's identity must not
change the answer.

Within that setup, the output must be byte-identical when a request runs alone,
batched with other requests, with its prefix cached or read fresh, and with the
prompt read in different chunks. Under greedy decoding, drafting ON and OFF
must produce the same answer. Request IDs and timestamps in HTTP envelopes are
not the model answer. `vade verify` compares repeated token-ID sequences with
drafting OFF and ON and can include an independent engine with `--machine`;
it fails if an ON run did not actually draft. This is a targeted check, not
proof of every batching, cache or chunking combination. See the
[verification implementation](../cli/src/verify/mod.rs) and
[cohort exactness tests](../engine/crates/serve/tests/cohort_mtp.rs).

Three checks answer different questions:

- **Repeatability:** does this build obey the same-setup contract across its
  execution paths? Run-to-run nondeterminism and reductions that depend on batch
  mates are forbidden.
- **Previous-build preservation:** do the tested outputs match frozen outputs
  from an older Vade build? This is regression evidence for those fixtures.
  The exactness contract permits answers to change across builds or Spark counts
  when the required quality checks pass.
- **Model quality:** does a change preserve task accuracy, instruction following
  and long-context retrieval? Byte equality to an older build does not by itself
  establish agreement with an external reference runtime.

Arithmetic changes need repeatability, layer-difference checks against the
measured numerical noise floor, GSM8K and IFEval comparisons, and long-memory
checks for large changes. [Benchmarks](benchmarks.md) keeps each reported result's qualification separate.
Finite passing fixtures support only their recorded coverage.

## Harness transport and tool boundaries

Native agent mode requires a chosen workspace. Reads, search and exact patches
are bounded by that workspace and tool policy. With the default `--approval ask`,
edits and commands outside the
[read-only allowlist](../harness/docs/agent-tools.md#commands-and-approvals) wait
for approval; allowlisted commands run without asking. A permission granted to
an outside delegated agent is governed by that agent's confirmed policy;
delegation does not make it a native Vade tool.

Native shell execution currently uses the macOS OS sandbox, a cleared
environment, private build storage and no network. Other platforms refuse
native command execution. Existing dependency/build trees can be inaccessible
under this policy, so a successful model response does not establish that an
arbitrary coding project's build will work. The
[harness README](../harness/README.md#native-coding-agent) gives the current
restrictions and verified scope.

The harness can also relay another client's requests through a gateway. It
preserves request and response payloads and does not translate between Chat and
Responses APIs. Multi-upstream routing can retain conversation affinity and use
reported cache telemetry. Compatibility and connection cleanup are separate
from confirmation that remote GPU computation stopped. The harness's HTTP
transport has no native TLS implementation; the configured controller route uses
an [SSH tunnel](../engine/docs/flash-server.md#connect-your-mac). See the
[gateway contract](../harness/docs/engine-contract.md#opaque-gateway) and
[routing guide](../harness/docs/gateway-routing.md).

## CLI boundaries and current limits

The one-Spark launcher `vade launch` discovers an existing checkpoint, builds or
uses local artifacts, starts the resident engine and opens chat. With a selected
controller configuration, `vade install` stages the configured installation and
`vade up` starts its selected ranks and an owned tunnel. Controller `vade up`
selects the FP4-Local release profile by default; `--fp4-local` remains accepted
and `--fp8-legacy` selects the legacy two-Spark FP8 profile. The separate `vade run qwen3-8b` route is legacy. See
[INSTALL.md](../INSTALL.md) and the [launcher implementation](../cli/src/launch.rs).

`vade status` and `vade logs` inspect the selected installation; `vade down`
stops its recorded processes. Process identity and artifact checks determine
which processes the CLI may stop. Configuration, ownership records and model
files remain private. Installation receipts and benchmark receipts qualify
different behavior; see their recorded scope in [INSTALL.md](../INSTALL.md)
and [Benchmarks](benchmarks.md).

Vade is a research alpha. FP4-Local on one Spark is the current release default;
the benchmark page also records home pairs and rented cloud VM measurements.
Additional model parsers, rank-count options and protocol routes in source do
not automatically have the same qualification. Cancellation can wait for an
execution chunk; prompt replay and compaction can add time between turns.
Cache accounting, rejected-row isolation, long-session behavior and complete
native coding workflows need their own receipts. A throughput result alone
does not establish those properties.

Continue with [Benchmarks](benchmarks.md), the
[engine crate map](../engine/crates/README.md), the
[harness docs](../harness/docs/README.md), or the [docs index](README.md).
