# How Vade works

Vade is one project, built for NVIDIA DGX Spark, with an inference engine and
an agent harness at its core. The engine runs the language model. The harness
turns the model's replies into a conversation that can read files, propose
edits and run commands you approve. A third part, the CLI, builds, starts and
stops whichever parts you select.

This page follows one request from the harness to the engine and back. It then
covers what runs on the GPU today, how several Sparks share one model, the
exactness rule, and the limits of the harness and the CLI.

To get a first response, follow [INSTALL.md](../INSTALL.md). For measured
results and their limits, read [Benchmarks](benchmarks.md).

## The three parts

| Part | Responsible for | Not responsible for |
| --- | --- | --- |
| [Engine](../engine/README.md) | Model loading, tokenization, GPU execution, sampling, request scheduling and model-state caches | Workspace tools or permission to edit your files |
| [Harness](../harness/README.md) | Conversation history, terminal UI, tools, approvals, private session journals and client transport | Model weights, GPU execution or the engine's cached state |
| [CLI](../cli/README.md) | Acquiring and building the parts, configuration, the processes it owns, and startup and shutdown records | Model arithmetic or the agent's tool decisions |

The engine and the harness are separate programs, each in its own Cargo
workspace inside this repository. You can use the engine from another
compatible client, or point the harness at another supported HTTP endpoint.

All three parts are first-party Rust, built on the standard library alone.
Beyond that, the engine talks to the CUDA driver and runs Vade's own GPU
kernels. The parts share protocols and development rules, but they are not one
combined executable.

## Follow one coding request

Here is what happens when you give the harness a coding task.

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

A *[rank](GLOSSARY.md#pr)* is one engine process, one per Spark. The engine
controller in the diagram is rank zero: it owns the HTTP listener and
coordinates the model commands that the ranks carry out on their GPUs. On one
Spark there is a single rank, which does both jobs.

1. **The harness prepares the conversation.** The request carries the task,
   the messages the harness has kept, the tool definitions and fixed request
   options. The engine never runs a tool itself. A tool's result becomes part
   of the next request.
2. **The harness asks what the engine supports.** An engine can advertise
   the optional `vade.engine.v1` contract: its model, context limits, tools,
   thinking and cache features. The harness can then ask whether a request
   will fit before sending it (*exact admission*). This is a check, not a
   reservation, so the engine can still refuse the request later.
3. **The engine accepts and schedules generation.** It checks limits, accounts
   for the execution memory available and looks for a prompt prefix it can
   reuse. An *affinity key*, a label that ties a request to earlier cached
   state, steers it toward reusable work. The key is a hint; the usage
   figures show whether reuse actually happened.
4. **The tokenizer turns the conversation into model input.** A *token* is a
   small piece of text, represented by an integer. The model's chat template
   supplies the framing for roles and tools. The checkpoint, tokenizer,
   template, tool order and thinking settings all affect the input.
5. **The engine reads, then writes.** Reading the input is called *prefill*;
   producing the reply is called *decode*. Decode updates the chat's model
   state, picks the next token under the sampling settings in effect, and
   turns tokens back into text. The server streams text, reasoning or
   tool-call data as soon as each piece is ready. At the end, the server
   reports completion and usage figures; those are the only numbers the
   client sees.
6. **The harness checks the finished reply before going on.** Text shown while
   it streams is provisional. A complete tool proposal runs only after it
   passes the workspace rules and the approval policy. An incomplete tool
   proposal never runs. Tool outcomes are written to a private journal before
   the next model turn, and resuming a session does not blindly run old tools
   again.

For the details:

- The [capability, admission and cache contract](../harness/docs/engine-contract.md)
  defines how negotiation and refusals work.
- The [tool guide](../harness/docs/agent-tools.md) defines tool arguments,
  limits and partial-write receipts (the saved record of a write that only
  partly finished).
- On the serving side, the engine's
  [HTTP server](../engine/crates/serve/README.md) and
  [scheduler](../engine/crates/serve/src/scheduler.rs) do the work.

## What the current Flash-Next path runs

The current release runs `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one Spark.
In engine terms that is *world 1*: a group of one rank. The server program is
[`flash-serve`](../engine/crates/tp/src/bin/flash-serve.rs). It reads the
checkpoint's own metadata to pick the code that runs the model, the
mixed-weight `NextModel`. The `--fp4-local` flag goes one step further: with
it, the server refuses to start unless the checkpoint is in that format.

There are two ways to start it:

- `vade launch`, run on the Spark itself.
- `vade up`, run from a
  [controller](../INSTALL.md#route-b-from-a-controller) that drives the Spark
  over SSH. FP4-Local is its default profile.

[INSTALL.md](../INSTALL.md) walks through both, and
[Benchmarks](benchmarks.md) has the dated measurements.

Two other programs sit next to `flash-serve` and cover special cases. Neither
changes the one-Spark default.

| Program | What it does |
| --- | --- |
| [`next-http`](../engine/crates/tp/src/bin/next-http.rs) | A compatibility launcher. It hands off to `flash-serve` with drafting off, on the older [tensor-map rank path](GLOSSARY.md#sz), where every Spark runs every layer but holds only its share of each one. |
| [`next-tp`](../engine/crates/tp/src/bin/next-tp.rs) | The rank qualification tool. It runs the tests that qualify the rank code; it is not the HTTP service. |

### The model and how its weights are stored

Flash-Next combines three kinds of layer:

- **Full attention** keeps the keys and values of earlier tokens.
- **Linear recurrent layers** keep a smaller state that evolves as they go.
- **Routed experts**: a router picks expert networks for each token, so a
  decode step does not read every expert's weights.

The name FP4-Local describes how the checkpoint stores its tensors, which is
a mix of formats. It does not mean that every tensor or operation uses
four-bit arithmetic. The loader reads each module's NVFP4, MXFP8 or plain
storage from the checkpoint's quantization map and checks its tensor
inventory. See the
[mixed-storage loader](../engine/crates/model/src/flash/mixed.rs).

The older two-Spark Flash-Next FP8 route remains as a legacy configuration. It
has its own [arithmetic record](../engine/docs/flash-fp8-math.md) and
[installation notes](install-legacy-fp8.md).

### Default sampling and caching

For FP4-Local, a request that names no sampling fields gets greedy decoding:
the model always takes its single most likely next token. Naming any sampling
field loads the checkpoint's base sampling settings first, then applies the
overrides you named.

Prefix caching saves the work of reading the start of a prompt, so a later
request that begins the same way can skip it. It is OFF by default.
`VADE_FLASHNEXT_PREFIX_CACHE=1` turns it on, and is supported only at world 1.
See the [sampling parser](../engine/crates/serve/src/protocol.rs) and
[cache switch](../engine/crates/serve/src/prefix_switch.rs).

### Drafting

Multi-token prediction (*MTP*) guesses a few tokens ahead, and the target
model then verifies those guesses. We call this drafting. By default, the
FP4-Local service drafts three tokens ahead and keeps only the guesses the
model confirms, so the answer is the same as without drafting. That default is
the `strict` mode. A request can change it:

| In the request | Effect |
| --- | --- |
| `"draft":"strict"` | Drafting ON |
| `"draft":"plain"` | Drafting OFF |
| `"mtp_depth":0` | Plain decoding, when no `draft` mode is named |

On top of that default, the engine has three drafting behaviors of its own.
Each applies on one Spark and on the two-Spark layer pipeline, which is
described under [Two or more Sparks](#two-or-more-sparks).

- **Adaptive depth.** When one chat runs alone, the engine adjusts how far
  ahead it drafts, up to a ceiling: 5 tokens on one Spark and 7 on the layer
  pipeline.
- **Prompt lookup.** The engine copies guesses from matching earlier text in
  the same chat. It is on by default, and runs only for greedy requests that
  draft. `VADE_MTP_LOOKUP=0` turns it off.
- **[Cohort drafting](GLOSSARY.md#ac).** The engine can also draft for
  several chats running at the same time, up to eight.

The older tensor-map rank path never drafts.

The server's settings only say what is allowed. Whether a given request
actually drafted depends on the request and on the scheduler. See
[drafting defaults](../engine/crates/engine/src/flash/draft_stack.rs),
[request handling](../engine/crates/serve/src/protocol.rs),
[prompt lookup](../engine/crates/engine/src/flash/next_lookup.rs) and
[cohort policy](../engine/crates/serve/src/mtp_batch.rs).

## Ranks, memory and ownership

When several Sparks serve one model, each Spark runs one rank, and rank zero
owns the HTTP listener, as in the diagram above. The other ranks carry out its
model commands with their own local weights and state. Together they compute
one answer; they are not independent model [replicas](GLOSSARY.md#pr), each
answering on its own.

### Two or more Sparks

With two Sparks (*world 2*, two ranks), FP4-Local runs as a layer pipeline by
default:

- Each rank runs the block of consecutive layers assigned to it, using
  world-one arithmetic (the same arithmetic as on one Spark). It then passes
  the intermediate state to the next stage.
- The front stage owns HTTP and scheduling. The workers run their layers and
  relay results downstream.
- Draft verification crosses the pipeline, so drafting stays available.

The `VADE_W2_PP` switch picks the layout:

| `VADE_W2_PP` | Two Sparks | More than two Sparks |
| --- | --- | --- |
| unset | Layer pipeline | Older tensor-map rank path |
| `0` | Older tensor-map rank path | Older tensor-map rank path |
| `1` | Layer pipeline | N-stage chain; `vade up --sparks N` sets this for you |

Supported Spark counts and the memory check described in
[What the engine keeps in memory](#what-the-engine-keeps-in-memory) still
apply. See
[pipeline selection](../engine/crates/tp/src/flash/settings.rs),
[stage geometry](../engine/crates/engine/src/flash/next_pp.rs) and
[controller selection](../cli/src/app.rs).

The stages talk over [RDMA](GLOSSARY.md#pr) (remote direct memory access,
a direct memory-to-memory transfer between Sparks) by default; TCP also works.

Before serving, the ranks confirm that they share the same artifacts and
numerical settings. The order of arithmetic and communication is fixed,
because adding floating-point numbers in a different order can change the
answer. See
[pipeline startup](../engine/crates/tp/src/flash/next_batched_service.rs) and
[settings agreement](../engine/crates/tp/src/pipeline_settings.rs).

### What the engine keeps in memory

The engine splits a rank's memory into five kinds and keeps them apart:

| Memory | What it holds | How long it lasts |
| --- | --- | --- |
| Resident weights | One copy of this rank's share of the model, shared by all its chats | Until the engine shuts down |
| Active chat state | One chat's attention cache, recurrent state and execution buffers | Until the chat ends |
| Retained prompt state | The engine's prefix cache and saved checkpoints | Until evicted under its memory budget |
| Shared scratch | Working space that every model step reuses | Reserved at startup, reused step after step |
| Host staging and transport | Buffers for loading and for network transfers, counted apart from model state | Kept for reuse |

*Memory admission* is the check that a request fits in memory. Every rank
agrees on the same limits before serving starts, so a request that rank zero
accepts also fits on the other ranks.

A few rules follow from these lifetimes:

- Idle cache can be reclaimed to make room for active sessions.
- Closing a chat does not necessarily evict a prefix that the engine has
  retained separately.
- Cached tokens still count toward the logical context length. Reuse saves
  work; it does not raise the model's context limit.

On a Spark, CPU and GPU allocations both draw from the same unified memory.
The engine tracks the bytes it allocates on the device, and that figure is
useful for comparing one engine build with another. It leaves out some driver
allocations, though, and it is not the Spark's total physical memory use. Each
benchmark [receipt](GLOSSARY.md#pr), the saved record of a benchmark run,
states which memory measure it reports.

## Exactness and model quality

Vade's exactness rule is: **the same setup and request produce identical token
IDs and answer bytes on every Spark, every run**. The setup includes the build,
checkpoint, tokenizer and template, rank layout, arithmetic, thinking and the
sampling configuration in effect. Which particular machine runs the request
must not change the answer.

Within one setup, the output must be byte-identical in each of these cases:

- whether a request runs alone or batched with other requests;
- whether its prefix is cached or read fresh;
- however the prompt is split into chunks for reading.

Under greedy decoding, drafting ON and OFF must produce the same answer.
Request IDs and timestamps in HTTP envelopes are not part of the model answer.

`vade verify` compares repeated token-ID sequences with drafting OFF and ON.
It can include an independent engine with `--machine`, and it fails if an ON
run did not actually draft. This is a targeted check, not proof of every
batching, cache or chunking combination. See the
[verification implementation](../cli/src/verify/mod.rs) and
[cohort exactness tests](../engine/crates/serve/tests/cohort_mtp.rs).

Three checks answer three different questions:

- **Repeatability:** does this build keep the same-setup rule across all its
  execution paths? Nondeterminism from run to run is forbidden, and so are
  reductions that depend on which other requests share the batch.
- **Previous-build preservation:** do the saved test cases (fixtures) still
  match an older Vade build? A new build or a different Spark count is a
  different setup, so its answers may change, but only if the quality checks
  below pass.
- **Model quality:** does a change keep task accuracy, instruction following
  and long-context retrieval? Matching an older Vade build is a separate
  question from matching an outside reference runtime.

A change to the arithmetic needs all of the following:

- repeatability;
- layer-difference checks against the measured numerical noise floor;
- GSM8K (grade-school math) and IFEval (instruction following) comparisons;
- long-memory checks, for large changes.

A passing set of saved test cases tells you only about the cases it covers.
[Benchmarks](benchmarks.md) keeps each reported result's qualification
separate.

## How the harness runs tools and relays requests

### Native agent mode

Native agent mode needs a workspace you choose. Reads, search and exact
patches stay within that workspace and its tool policy. With the default
`--approval ask`, edits and commands outside the
[read-only allowlist](../harness/docs/agent-tools.md#commands-and-approvals)
wait for your approval. Allowlisted commands run without asking.

You can also hand work to an outside agent. That agent does not become a
native Vade tool: it follows the permission policy you confirmed for it, not
Vade's.

Native shell commands run only on macOS for now, inside its sandbox: a
cleared environment, private build storage and no network. Tools run on the
computer where you start the harness. On Linux, including the Spark itself,
the harness can read and patch files but will not run commands. To let the
agent run commands, start the harness on a Mac and point it at the Spark.

Inside the sandbox, a command may not reach a project's existing dependency
and build trees, so a good model reply does not mean an arbitrary coding
project will build. The
[harness README](../harness/README.md#native-coding-agent) gives the current
restrictions and what has been verified.

### Gateway mode

In gateway mode, the harness relays another client's requests to one or more
upstream servers. It passes request and response payloads through as they
are, and does not translate between the Chat and Responses APIs. When it
routes across several upstreams, it can keep each conversation on the same
upstream and use the cache telemetry the upstreams report.

Two cautions. First, a cleanly closed connection does not mean the remote
GPU has stopped working on that request. Second, the harness speaks plain HTTP
with no built-in TLS, so between machines use the
[SSH tunnel](../engine/docs/flash-server.md#connect-your-mac) that the
controller route sets up. See the
[gateway contract](../harness/docs/engine-contract.md#opaque-gateway) and
[routing guide](../harness/docs/gateway-routing.md).

## What the CLI starts and stops

The one-Spark launcher `vade launch` finds an existing checkpoint, builds or
uses local artifacts, starts the resident engine and opens chat.

On the controller route, a [controller](GLOSSARY.md#ac) (a second computer,
or one of the Sparks acting as its own controller) drives the Sparks over SSH. There,
`vade install` stages the configured installation, and `vade up` starts the
selected ranks plus a tunnel that it owns. By default, `vade up` picks the
FP4-Local release profile. `--fp4-local` is still accepted, and
`--fp8-legacy` selects the legacy two-Spark FP8 profile. The separate
`vade run qwen3-8b` route is also legacy. See
[INSTALL.md](../INSTALL.md) and the [launcher implementation](../cli/src/launch.rs).

`vade status` and `vade logs` inspect the selected installation, and
`vade down` stops the processes the CLI recorded. Before it stops anything,
the CLI checks each process's identity and artifacts, so it stops only
processes it owns. Its configuration, ownership records and model files stay
private.

`vade install` keeps a record of what it staged, with a SHA256 hash for each
build, and later commands check against it. A benchmark receipt records one
measured run. The two prove different things; [INSTALL.md](../INSTALL.md) and
[Benchmarks](benchmarks.md) spell out each.

## Current limits

Vade is a research alpha. The release default is FP4-Local on one Spark. The
benchmark page also has results from cabled pairs of Sparks and from rented
cloud VMs. The other model parsers, Spark counts and protocol routes in the
source are not qualified to the same level.

Known limits:

- Cancelling can wait for the current chunk of GPU work to finish.
- Replaying or compacting a prompt can add time between turns.

Speed numbers don't cover the following; each needs its own test:

- how cache use is counted;
- whether one failed request in a batch leaves the other requests unharmed,
  for example when the model's expert router rejects it;
- long sessions;
- complete native coding workflows.

Continue with [Benchmarks](benchmarks.md), the
[engine crate map](../engine/crates/README.md), the
[harness docs](../harness/docs/README.md), or the [docs index](README.md).
