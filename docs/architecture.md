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

All three parts are first-party Rust and use the standard library. The engine
also uses the CUDA driver and Vade's own GPU kernels. The parts share protocols
and development rules, but they are not one combined executable.

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

A *rank* is one engine process, one per Spark. The engine controller in the
diagram is rank zero: it owns the HTTP listener and coordinates the model
commands that the ranks carry out on their GPUs. On one Spark there is a single
rank, which does both jobs.

1. **The harness prepares the conversation.** The request carries the task,
   the messages the harness has kept, the tool definitions and fixed request
   options. The engine never runs a tool itself. A tool's result becomes part
   of the next request.
2. **The client checks what the engine supports.** The optional
   `vade.engine.v1` contract advertises the model, context limits, tools,
   thinking and cache features. *Exact admission* is the check that a request
   will fit. It renders and tokenizes the complete final request, then checks
   whether its prompt plus the reserved output fits. This check reserves no
   GPU memory or queue slot, so generation can still be refused.
3. **The engine accepts and schedules generation.** It checks limits, accounts
   for the execution memory available and looks for a prompt prefix it can
   reuse. An *affinity key* helps it choose: a label that ties a request to
   earlier cached state. The key alone does not prove that any tokens were
   reused.
4. **The tokenizer turns the conversation into model input.** A *token* is a
   small piece of text, represented by an integer. The model's chat template
   supplies the framing for roles and tools. The checkpoint, tokenizer,
   template, tool order and thinking settings all affect the input.
5. **The engine reads, then writes.** Reading the input is called *prefill*.
   Producing the reply is called *decode*. Decode updates the chat's model
   state, picks the next token under the sampling settings in effect, and
   turns tokens back into text. The server streams text, reasoning or
   tool-call data as soon as it is ready. Completion and usage fields let the
   client check the result. A number the server does not report is one the
   client cannot know.
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
In engine terms that is *world 1*: a group of one rank. The server is
`flash-serve`. The checkpoint's own metadata selects the code that runs it,
the mixed-weight `NextModel`, and the `--fp4-local` flag explicitly requires
that checkpoint format.

There are two ways to start it:

- `vade launch`, run on the Spark itself.
- `vade up --fp4-local`, through the configured
  [controller route](../INSTALL.md#route-b-from-a-controller), where `vade`
  drives the Spark over SSH.

See [INSTALL.md](../INSTALL.md) for both, and the dated measurements in
[Benchmarks](benchmarks.md).

The [serving entry point](../engine/crates/tp/src/bin/flash-serve.rs) is
`flash-serve`. Two other programs cover special cases. Neither changes the
one-Spark default.

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

FP4-Local describes how the checkpoint stores its tensors, in a mix of
formats. It does not claim that every tensor or operation uses four-bit
arithmetic. The loader reads each module's NVFP4, MXFP8 or plain storage from
the checkpoint's quantization map and checks its tensor inventory. See the
[mixed-storage loader](../engine/crates/model/src/flash/mixed.rs).

The older two-Spark Flash-Next FP8 route remains as a legacy configuration. It
has its own [arithmetic record](../engine/docs/flash-fp8-math.md) and
[installation notes](install-legacy-fp8.md).

### Default sampling and caching

For FP4-Local, a request that names no sampling fields gets greedy decoding:
the model always takes its single most likely next token. Naming any sampling
field loads the checkpoint's base sampling settings first, then applies the
overrides you named.

Prefix caching is OFF by default. `VADE_FLASHNEXT_PREFIX_CACHE=1` turns it on,
and is supported only at world 1. See the
[sampling parser](../engine/crates/serve/src/protocol.rs) and
[cache switch](../engine/crates/serve/src/prefix_switch.rs).

### Drafting

Multi-token prediction (*MTP*) proposes tokens for the target model to verify.
We call this drafting. The default FP4-Local service uses strict drafting with
request depth 3. A request can choose:

| In the request | Effect |
| --- | --- |
| `"draft":"strict"` | Drafting ON |
| `"draft":"plain"` | Drafting OFF |
| `"mtp_depth":0` | Plain decoding, when no `draft` mode is named |

Around that default, the engine has these behaviors. The layer pipeline
mentioned here is the two-Spark setup described in the next section.

- **Adaptive depth.** When one chat runs alone, the engine adjusts how far
  ahead it drafts, up to a ceiling: 5 tokens on one Spark and 3 on the layer
  pipeline.
- **Prompt lookup.** This copies guesses from matching earlier text in the
  same chat. It is on by default on both, and runs only for greedy requests
  that draft. `VADE_MTP_LOOKUP=0` turns it off.
- **Cohort drafting.** On one Spark and on the two-Spark pipeline, the engine
  also manages drafting across chats running at the same time.
- **Tensor-map rank path.** It never drafts.

The server settings say only what is allowed. Whether a request actually
drafted depends on the request and the scheduler. See
[drafting defaults](../engine/crates/engine/src/flash/draft_stack.rs),
[request handling](../engine/crates/serve/src/protocol.rs),
[prompt lookup](../engine/crates/engine/src/flash/next_lookup.rs) and
[cohort policy](../engine/crates/serve/src/mtp_batch.rs).

## Ranks, memory and ownership

When several Sparks serve one model, each runs one rank. Rank zero owns the
HTTP listener and coordinates model commands. The other ranks carry out those
commands with their local weights and state. Together they compute one answer;
they are not independent model replicas.

### Two or more Sparks

At world 2 (two ranks), FP4-Local defaults to the `VADE_W2_PP` layer
pipeline:

- Each rank runs the block of consecutive layers assigned to it, using
  world-one arithmetic (the same arithmetic as on one Spark). It then passes
  the intermediate state to the next stage.
- The front stage owns HTTP and scheduling. The workers run their layers and
  relay results downstream.
- Draft verification crosses the pipeline.

`VADE_W2_PP=0` explicitly selects the older tensor-map rank path instead. For
more than two Sparks, `VADE_W2_PP=1` opts into an N-stage chain. The
controller sets this when `vade up --fp4-local --sparks N` selects a larger
group. Supported counts and memory admission still apply. See
[pipeline selection](../engine/crates/tp/src/flash/settings.rs),
[stage geometry](../engine/crates/engine/src/flash/next_pp.rs) and
[controller selection](../cli/src/app.rs).

The pipeline uses RDMA by default, which moves data between machines through
registered memory; TCP is the other option. Before serving, startup checks the
artifacts and the ranks' numerical settings. The order of arithmetic and
communication is fixed, because changing the order of floating-point
additions can change the answer. See
[pipeline startup](../engine/crates/tp/src/flash/next_batched_service.rs) and
[settings agreement](../engine/crates/tp/src/pipeline_settings.rs).

### What the engine keeps in memory

The engine keeps these lifetimes separate:

| Memory | Who owns it, and for how long |
| --- | --- |
| Resident weights | One copy of the rank's share of model weights, shared by its chats until engine shutdown |
| Active chat state | The session's attention cache, recurrent state and execution buffers |
| Retained prompt state | Engine-owned prefix cache and scoped checkpoints, subject to budgets and eviction |
| Shared scratch | Model/decode workspaces reused in the engine's execution order |
| Host staging and transport | Loading and registered communication buffers, accounted for separately from model state |

*Memory admission* is the check that a request fits in memory. Every rank
agrees on the same limits before serving starts, so a request that rank zero
accepts also fits on the other ranks.

A few rules follow from these lifetimes:

- Idle cache can be reclaimed to make room for active sessions.
- Closing a chat does not necessarily evict a prefix that is retained on its
  own.
- Cached tokens still count toward the logical context length. Reuse saves
  work; it does not raise the model's context limit.

On a Spark, CPU and GPU allocations both draw from the same unified memory.
Tracked device-allocation bytes are useful for comparing engine builds. They do
not include every driver allocation, though, and do not show total physical
memory use. Benchmark receipts, the saved record of each benchmark run, state
which memory measure they report.

## Exactness and model quality

Vade's exactness rule is: **the same setup and request produce identical token
IDs and answer bytes on every Spark, every run**. The setup includes the build,
checkpoint, tokenizer and template, rank layout, arithmetic, thinking and the
sampling configuration in effect. Which particular machine runs the request
must not change the answer.

Within one setup, the output must be byte-identical in each of these cases:

- whether a request runs alone or batched with other requests;
- whether its prefix is cached or read fresh;
- whatever chunks the prompt is read in.

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
- **Previous-build preservation:** do the tested outputs match frozen outputs
  from an older Vade build? That tells you about regressions in those saved
  test cases (fixtures). The exactness rule allows answers to change across
  builds or Spark counts when the required quality checks pass.
- **Model quality:** does a change keep task accuracy, instruction following
  and long-context retrieval? Matching an older build byte for byte does not,
  by itself, show agreement with an outside reference runtime.

Arithmetic changes need all of the following:

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

If you hand work to an outside agent, that agent's own confirmed policy
governs any permission you grant it. Handing work to it does not make it a
native Vade tool.

Native shell commands currently run in the macOS sandbox, with a cleared
environment, private build storage and no network. Other platforms refuse
native command execution. Under this policy, existing dependency and build
trees can be out of reach. A successful model response therefore does not mean
an arbitrary coding project's build will work. The
[harness README](../harness/README.md#native-coding-agent) gives the current
restrictions and what has been verified.

### Gateway mode

The harness can also relay another client's requests through a gateway. It
keeps request and response payloads as they are, and does not translate
between the Chat and Responses APIs. When it routes across several upstreams,
it can keep each conversation on the same upstream and use the cache telemetry
the upstreams report.

A compatible relay and a cleanly closed connection do not confirm that the
remote GPU work has stopped. The harness's HTTP transport has no built-in TLS;
the configured controller route uses an
[SSH tunnel](../engine/docs/flash-server.md#connect-your-mac). See the
[gateway contract](../harness/docs/engine-contract.md#opaque-gateway) and
[routing guide](../harness/docs/gateway-routing.md).

## What the CLI starts and stops

The one-Spark launcher `vade launch` finds an existing checkpoint, builds or
uses local artifacts, starts the resident engine and opens chat.

With a controller configuration selected, `vade install` stages the configured
installation, and `vade up` starts its selected ranks and a tunnel it owns.
Controller `vade up` picks the FP4-Local release profile by default.
`--fp4-local` is still accepted, and `--fp8-legacy` selects the legacy
two-Spark FP8 profile. The separate `vade run qwen3-8b` route is legacy. See
[INSTALL.md](../INSTALL.md) and the [launcher implementation](../cli/src/launch.rs).

`vade status` and `vade logs` inspect the selected installation. `vade down`
stops the processes it recorded. Process identity and artifact checks decide
which processes the CLI may stop. Configuration, ownership records and model
files stay private.

Installation receipts and benchmark receipts vouch for different behavior. See
the recorded scope of each in [INSTALL.md](../INSTALL.md) and
[Benchmarks](benchmarks.md).

## Current limits

Vade is a research alpha. The release default is FP4-Local on one Spark. The
benchmark page also has results from cabled pairs and rented cloud VMs. Other
model parsers, Spark counts and protocol routes in the source are not
qualified to the same level.

Known limits:

- Cancelling can wait for the current chunk of GPU work to finish.
- Replaying or compacting a prompt can add time between turns.

A speed result alone does not show any of the following. Each needs its own
evidence:

- how cache use is counted;
- whether a request the expert router rejects fails alone, without disturbing
  the rest of its batch;
- long sessions;
- complete native coding workflows.

Continue with [Benchmarks](benchmarks.md), the
[engine crate map](../engine/crates/README.md), the
[harness docs](../harness/docs/README.md), or the [docs index](README.md).
