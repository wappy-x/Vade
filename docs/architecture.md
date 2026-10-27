# How Vade works

Vade runs a large language model on an NVIDIA DGX Spark and answers chat
requests over HTTP. It is one repository with three programs: an inference
engine that runs the model on the Spark's GPU, an agent harness that turns the
model's replies into a coding conversation, and a CLI that builds, starts and
stops them. All three are first-party Rust built on the standard library
alone, and the engine talks straight to NVIDIA's driver.

The release serves one model, `nvidia/Qwen3.8-Flash-Next-FP4-Local`, on one
Spark by default, and can spread it across a chain of Sparks. Two ideas shape
almost every design choice on this page:

- **Exactness.** The same request on the same setup returns the same token IDs
  every time, whether it ran alone or batched with other chats, whether its
  prompt came from the cache or was read fresh, however the prompt was split
  into chunks, and, for greedy requests, with drafting on or off.
- **The bytes as stored.** Writing an answer is limited by how fast the GPU
  can read weights from memory, so the kernels read each weight in the format
  the checkpoint stores it in, and we judge each kernel against the time it
  would take just to move those bytes.

This page follows one request from the network to the GPU and back, then
explains each part in turn: the model and its number formats, the kernels,
drafting, several Sparks, the prefix cache, memory, exactness, and how we
measure all of it. To install Vade, follow [INSTALL.md](../INSTALL.md). For the
measured results, read [Benchmarks](benchmarks.md).

A few words come up in every section of this page:

- A **token** is a small piece of text, represented by an integer ID. The
  model reads tokens and predicts the next one.
- **Prefill** is reading the prompt, many tokens at a time. **Decode** is
  writing the answer, one token per step.
- **BF16**, **FP8** and **FP4** are 16-, 8- and 4-bit number formats. **FP32**
  is the ordinary 32-bit float, used here for running sums and activations.
- A **kernel** is a small program that runs on the GPU. **PTX** is NVIDIA's
  portable assembly language for kernels.
- A **checkpoint** is the published set of model files: weights,
  configuration and tokenizer.

Everything else is defined where it first appears, and in the
[glossary](GLOSSARY.md).

## The three parts

| Part | Responsible for | Not responsible for |
| --- | --- | --- |
| [Engine](../engine/README.md) | Loading the model, tokenization, GPU execution, sampling, scheduling requests, and model-state caches | Workspace tools, or permission to edit your files |
| [Harness](../harness/README.md) | Conversation history, the terminal UI, tools, approvals, private session journals, and the optional gateway | Model weights, GPU execution, or the engine's cached state |
| [CLI](../cli/README.md) | Acquiring and building the parts, configuration, the processes it owns, startup and shutdown | Model arithmetic, or the agent's tool decisions |

The engine and the harness are separate programs in separate Cargo
workspaces. Any OpenAI-compatible client can use the engine, and the harness
can point at another compatible server. The engine itself is a workspace of
packages; the ones on the release path are:

```text
engine/crates/
  serve      HTTP server, scheduler, sampling, prompt cache   (spark-serve)
  tokenizer  chat templates, text <-> token IDs                (engine-tokenizer)
  model      checkpoint metadata, tensor checks, layer plans   (engine-model)
  engine     the Flash-Next model: layers, sessions, drafting  (spark-engine)
  gpu        CUDA driver interface, memory, launches, graphs   (spark-gpu-probe)
  gpu/kernels  the GPU kernels, Rust compiled to PTX           (spark-rust-kernels)
  tp         flash-serve, the multi-Spark services             (spark-tp)
  fabric     peer discovery and RDMA                           (engine-fabric)
```

The [package map](../engine/crates/README.md) lists all fifteen.

## Follow one request

Here is the path of one chat request through a one-Spark engine.

```text
client (harness, IDE, curl)
   │ HTTP
   ▼
gateway (optional)   loopback relay; may add greedy defaults
   │
   ▼
flash-serve, rank 0
   │
   ├─ engine thread    reads the socket without blocking
   │     │ complete request
   │     ▼
   ├─ prepare thread   parses JSON, renders the chat template, tokenizes
   │     │ token IDs
   │     ▼
   ├─ engine thread    scheduler loop: admit, look up the cache, pick rows
   │     │
   │     ▼
   ├─ NextModel        prompt tiles, decode steps, drafting
   │     │
   │     ▼
   ├─ GPU kernels      recorded CUDA graphs replay each decode step
   │     │ next token
   │     ▼
   └─ engine thread    turns tokens into text, streams it (SSE), ends with usage
         │
         ▼
      client
```

**The gateway, if you use one.** A client can talk to the engine directly, or
through a Vade gateway. The harness's gateway mode listens on `127.0.0.1:8787`
by default ([`args.rs`](../harness/src/cli/args.rs)). The standalone
`vade-gateway` is a loopback-only relay that forwards every path without
translating between APIs ([invisible gateway](../harness/docs/invisible-gateway.md)).
When a request names none of the sampling fields, the harness gateway inserts
greedy defaults; `vade-gateway` does so only when started with
`--greedy-when-unset` ([`invisible.rs`](../harness/src/gateway/invisible.rs)).
If a request names any sampling field, its body passes through byte for byte.
That rule is defined once, in the engine's model crate
([`sampling_policy.rs`](../engine/crates/model/src/sampling_policy.rs)), and the
harness compiles the same source file instead of taking a dependency on the
engine ([`sampling.rs`](../harness/src/gateway/sampling.rs)). With several upstream
engines, the gateway can keep a conversation on the same engine; that
affinity is a routing hint, never a claim that a cache was hit
([routing](../harness/docs/gateway-routing.md)).

**The HTTP layer.** On rank zero, `flash-serve` binds a standard-library
`TcpListener` ([`flash-serve.rs`](../engine/crates/tp/src/bin/flash-serve.rs)).
The server's HTTP code is Vade's own
([`http.rs`](../engine/crates/serve/src/http.rs)). It refuses a non-loopback
bind unless you pass `--allow-remote`, and stops with
`remote bind requires explicit --allow-remote`.

**Preparing the request.** The engine thread reads sockets without blocking
and hands each complete request to a separate thread, which parses the JSON,
applies the chat template and tokenizes the prompt, so that parsing a long
prompt never holds up GPU work
([`scheduler.rs`](../engine/crates/serve/src/scheduler.rs)). The tokenizer is
Vade's own BPE implementation
([tokenizer](../engine/crates/tokenizer/README.md)). Sampling is decided here
too: a request that names no sampling field is decoded greedily, and a
request that names any one of them takes the checkpoint's own settings for
the rest ([`protocol.rs`](../engine/crates/serve/src/protocol.rs)). Note that
`"temperature": 0` alone counts as naming a field, so it brings in the
checkpoint's penalties. For a greedy answer, send no sampling fields, or set
all of them to neutral values (the [README](../README.md) lists them).

**The scheduler.** Everything that touches the GPU runs on one thread. The
scheduler loop accepts prepared requests, checks that each one fits in the
memory pool, looks for a reusable cached prefix, and then decides which rows
of work run in the next model step. Decode steps for all active chats are
batched together, and prompt reading shares the GPU with them in bounded
time slices, so neither starves the other. A request can also carry a
priority that lets foreground work go first
([`scheduler.rs`](../engine/crates/serve/src/scheduler.rs),
[`dispatch.rs`](../engine/crates/serve/src/scheduler/dispatch.rs)). On one
Spark the engine holds up to sixteen resident chats, as many as memory admits
([`next_batched.rs`](../engine/crates/engine/src/flash/next_batched.rs)).

**The model step.** The prompt is read in tiles of thousands of rows at a
time, and the answer is written one step at a time, with every active chat in
the same step. Decode steps replay recorded CUDA graphs instead of launching
each kernel from the host. With drafting on, a step may also check several
guessed tokens per chat (see [Drafting](#drafting)).

**Back to the client.** Each new token is turned back into text and streamed
as soon as it exists, as server-sent events. At the end the server reports
completion and usage figures. Text a client sees while it streams is
provisional for the harness: a tool call runs only once its proposal is
complete and has passed the workspace rules and your approval policy (see
[the harness](#the-harness-and-the-cli)).

## Which program serves what

The repository holds more than one server, from different stages of the
project. Only one is the release path.

| Program | What it is |
| --- | --- |
| [`flash-serve`](../engine/crates/tp/src/bin/flash-serve.rs) | **The release server.** Serves FP4-Local on one Spark, across two Sparks, or as a longer chain. It also serves the legacy Flash-Next FP8 checkpoint. `vade launch` and `vade up` start it. |
| [`spark-serve`](../engine/crates/serve/src/bin/spark-serve.rs) | The original one-Spark server for earlier models (Qwen3, Qwen3.8-27B, Gemma 4). It shares the HTTP and scheduler library with `flash-serve` but is not a v0.1.0 route; see [its README](../engine/crates/serve/README.md). |
| [`next-http`](../engine/crates/tp/src/bin/next-http.rs) | A compatibility launcher that hands off to `flash-serve` on the older tensor-map rank path, with drafting off. |
| [`next-tp`](../engine/crates/tp/src/bin/next-tp.rs) | The rank qualification tool, not an HTTP service. |
| [`tp-serve`](../engine/crates/tp/src/bin/tp-serve.rs) | An older HTTP service over tensor-parallel rank commands. |

`flash-serve` reads the checkpoint's own metadata to choose its model code:
the mixed storage of an FP4-Local checkpoint selects `NextModel`, and the
`--fp4-local` flag only asserts that the checkpoint really is that kind
([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)).

## Standard-library Rust, and why

Vade has **zero third-party dependencies**. Every Rust package uses only
`std` (and `core` on the GPU) plus the first-party crates in this tree. A test
in the CLI counts dependencies in every manifest and lockfile and fails if
the count rises above zero ([`ratchet.rs`](../cli/tests/ratchet.rs),
[baseline](../cli/ratchet-baseline.txt)).

That means Vade carries its own versions of things most projects import: an
HTTP/1.1 server, a bounded JSON parser
([engine-base](../engine/crates/base/README.md)), streaming SHA-256 for
checkpoint verification ([engine-hash](../engine/crates/hash/README.md)), a
BPE tokenizer, a hand-declared interface to the CUDA driver
([`driver.rs`](../engine/crates/gpu/src/driver.rs)), RDMA through
`libibverbs` loaded at run time ([`verbs.rs`](../engine/crates/fabric/src/verbs.rs)),
and every GPU kernel. No CUDA toolkit, cuBLAS, CUTLASS, NCCL or Python runs in
the engine. When two workspaces need the same code, one compiles the other's
source file with a `#[path]` attribute rather than adding a Cargo dependency.

We pay that cost for three reasons:

1. **One arithmetic order everywhere.** Exactness depends on every floating
   point sum being added in the same order on every path. That is only
   possible if we own every kernel that touches the numbers. A library that
   picks a different reduction split for a different batch size would break
   the promise silently. The [kernel crate](../engine/crates/gpu/kernels/README.md)
   states this as its reason to exist.
2. **The whole stack is readable.** Every line between the HTTP socket and the
   GPU instruction is in this repository, so a reviewer can audit it and a
   bug cannot hide in a version bump. No dependency update can change behavior
   between builds, and once the pinned toolchains are installed, the host
   checks run fully offline ([AGENTS.md](../AGENTS.md#local-validation)).
3. **Small attack surface.** No registry or git dependency can be swapped in
   the supply chain, because there are none.

The price is `unsafe` code at four boundaries that the standard library cannot
cover: the kernels themselves, the CUDA driver calls, kernel launches, and RDMA
memory shared with the network card. The [unsafe inventory](../engine/docs/unsafe.md)
counts every site, and the house rule is that no change may add new ones.

## What the current Flash-Next path runs

### The model

Flash-Next is a hybrid. The model code derives its layer list from the
checkpoint's `layer_types`: each `linear_attention` layer becomes a
GatedDeltaNet layer, and each `full_attention` layer becomes an indexed sparse
attention layer ([`flash/mod.rs`](../engine/crates/model/src/flash/mod.rs)).

- **Indexed attention (QSA)** layers are attention layers: for every token
  read so far they store two vectors, a *key* and a *value*, which later
  tokens compare against to decide what to look back at. Those stored vectors
  are the **KV cache**, and it grows with the conversation. Here a small
  indexer scores blocks of earlier tokens, and attention reads only the
  selected blocks plus the newest tokens
  ([`qsa.rs`](../engine/crates/engine/src/flash/qsa.rs),
  [indexer contract](../engine/docs/qsa-indexer.md)).
- **GatedDeltaNet (GDN)** layers are linear recurrent layers: instead of
  storing every past token, each one folds what it has read into a
  fixed-size state that is updated as each token arrives
  ([`gdn.rs`](../engine/crates/engine/src/flash/gdn.rs)).
- **Routed experts.** After attention, a router picks a few expert networks
  for each token, alongside a shared expert, so a decode step reads only the
  selected experts' weights ([`moe.rs`](../engine/crates/engine/src/flash/moe.rs)).
- **Hyper-connections** keep four parallel residual streams instead of one,
  mixed before and combined after each block
  ([`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs)).
- **Per-layer and n-gram embeddings (PLE)** add a large table, looked up by
  a hash of the most recent tokens, that enters at layer 1.
- **An MTP layer** (multi-token prediction) predicts tokens beyond the next
  one. Vade uses it to draft.

One token's trip through the model looks like this
([`next_batched.rs`](../engine/crates/engine/src/flash/next_batched.rs)):

```text
token ID
  │
  ▼
embedding ──► four residual streams
  │
  ▼  (PLE joins at layer 1)
┌─ each layer ───────────────────────────────────────────┐
│  mix streams ─► attention (GDN or QSA) ─► combine      │
│  mix streams ─► router + experts + shared ─► combine   │
└────────────────────────────────────────────────────────┘
  │
  ▼
final mix ─► output head ─► next-token scores ─► sampling
```

Activations stay in FP32 storage throughout
([`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs)). An FP32
reference implementation of the decoder on the CPU, written from the
published PyTorch model, stays in the tree as the oracle that the GPU path is
checked against ([`refcpu.rs`](../engine/crates/model/src/flash/refcpu.rs)).

### FP4-Local and its number formats

"FP4-Local" names how the checkpoint stores its tensors, and it is a mix. The
loader reads each module's storage from the checkpoint's quantization map and
checks the tensor inventory against it
([`mixed.rs`](../engine/crates/model/src/flash/mixed.rs)):

| Storage | Layout | Used for |
| --- | --- | --- |
| NVFP4 (NVIDIA's 4-bit float format) | 4-bit E2M1 values, one E4M3 scale per 16 values, one FP32 multiplier per matrix ([`nvfp4.rs`](../engine/crates/model/src/flash/nvfp4.rs)) | Routed experts, the MTP layer's experts and the PLE table |
| MXFP8 (microscaling 8-bit float) | 8-bit E4M3 values, one power-of-two UE8M0 scale per 32 values ([`mxfp8.rs`](../engine/crates/model/src/flash/mxfp8.rs)) | Attention-side projections (full attention, GDN and the indexer) and the shared experts |
| BF16 and FP32 | Plain storage | The output head, routers, norms and the remaining small tensors |

Three details matter for exactness and speed:

- **Lossless scale compression.** FP4-Local stores the NVFP4 scales in a
  compressed form. The loader expands it back to the exact E4M3 bytes, and
  every byte pattern survives the round trip
  ([`csf_expand`](../engine/crates/model/src/flash/nvfp4.rs)).
- **FP4 weights, BF16 arithmetic.** GB10's native FP4 matrix instruction needs
  FP4 activations too. Vade keeps activations at higher precision instead: the
  expert kernels decode each FP4 weight in registers and multiply on the BF16
  tensor cores, reading every packed weight byte once per tile
  ([`fp4_experts.rs`](../engine/crates/gpu/kernels/src/fp4_experts.rs),
  [`fp4_exact.rs`](../engine/crates/gpu/kernels/src/fp4_exact.rs)). Vade does
  not implement the checkpoint's 4-bit activation quantizer
  ([`fp4_abi.rs`](../engine/crates/gpu/src/fp4_abi.rs)).
- **FP32 activations on BF16 tensor cores.** Projections split each FP32
  activation into a BF16 high part and a BF16 residual part, and run both
  through the tensor cores with FP32 sums. This is the one projection mode;
  there is no lower-precision shortcut
  ([Flash arithmetic](../engine/docs/flash-fp8-math.md),
  [`flash_bf16.rs`](../engine/crates/gpu/kernels/src/flash_bf16.rs)).

The arithmetic behind every returned token, the target model and its
verification of drafts, never re-quantizes the checkpoint: it reads the tensors
in the formats above. The BF16 output head may be re-encoded in place as
LBF12, a lossless packing that decodes to the identical BF16 bits
(`VADE_HEAD_LBF12=0` keeps plain BF16).

The drafter is the exception, and it is on by default whenever drafting is.
At load Vade builds lower-precision copies that only propose tokens: an MXFP8
copy and an NVFP4 copy of the scored output-head rows
(`VADE_FLASHNEXT_DRAFT_MX8=0` and `VADE_FLASHNEXT_DRAFT_FP4=0` turn them off),
and MXFP8 copies of the MTP layer's seven BF16 matrices (`fc_embedding`,
`fc_hidden`, `q_proj`, `k_proj`, `v_proj`, the indexer's `index_qk_proj` and
`o_proj`; `VADE_FLASHNEXT_DRAFT_MX8_LAYER=0` turns them off)
([`next_batched.rs`](../engine/crates/engine/src/flash/next_batched.rs),
[`head.rs`](../engine/crates/engine/src/flash/head.rs),
[`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs)). Verification
keeps the full BF16 head and the checkpoint's weights, so these copies change
which tokens are proposed and how many are accepted, not the token IDs
returned. Any other change that would compute below the checkpoint's precision
needs the full exactness gate, quality evidence against main and a reviewed
card before it can become the default. Keep one production path, as required
by [AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate).

### The legacy FP8 checkpoint

Before FP4-Local, Vade served `Qwen/Qwen3.8-Flash-Next-FP8` across two Sparks.
That checkpoint stores FP8 weights in 128×128 blocks with dynamic activation
scales ([`flash/mod.rs`](../engine/crates/model/src/flash/mod.rs)), and its
experts follow the reference W8A8 arithmetic recorded in
[Flash arithmetic](../engine/docs/flash-fp8-math.md). `flash-serve` still
serves it with tensor parallelism, where both Sparks run every layer and each
holds its share of each one, and `vade up --fp8-legacy` selects it.
It is kept as a legacy configuration with its own
[installation notes](install-legacy-fp8.md); it is not a v0.1.0 route.

## The kernels and the as-stored floor

### Rust on the GPU

The kernels are ordinary `#![no_std]` Rust, compiled by the official Rust
compiler's NVPTX backend to PTX, NVIDIA's portable GPU assembly
([`lib.rs`](../engine/crates/gpu/kernels/src/lib.rs)). A few instructions
have no Rust spelling, such as the tensor-core multiply and asynchronous
copies into shared memory, so the kernels use LLVM intrinsics and short inline
PTX for those. `cargo kernels` builds them with the pinned nightly compiler and
stamps the PTX with a hash of its sources and flags
([build-kernels](../engine/crates/gpu/build-kernels/README.md)). At startup,
`flash-serve` hands the PTX to the driver, which compiles it to GB10 machine
code ([`driver.rs`](../engine/crates/gpu/src/driver.rs)).

### Exact by construction

Exactness is not a test we hope passes; the kernels are built so that a row's
result cannot depend on what else is running:

- **Fixed reduction trees.** A projection splits its inner dimension into four
  ranges whose boundaries depend only on the matrix shape, and merges them as
  `(p0 + p1) + (p2 + p3)`, independent of the number of tokens or the output
  tiling ([`canonical.rs`](../engine/crates/gpu/kernels/src/canonical.rs),
  [`flash_bf16.rs`](../engine/crates/gpu/kernels/src/flash_bf16.rs)).
- **Row kernels.** Decode and draft verification use kernels whose per-row
  sums never depend on how many rows share the launch, so a chat's bytes in a
  batch equal its bytes alone
  ([`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)).
- **Aligned prompt tiles.** Prompt tiles start on absolute 64-token
  boundaries, and the rows inside a tile are exact whatever the tile size, so
  how a prompt is chunked does not change its result
  ([`next_prefix.rs`](../engine/crates/engine/src/flash/next_prefix.rs),
  [`next_rows.rs`](../engine/crates/engine/src/flash/next_rows.rs)).
- **No order-dependent arithmetic.** The house rules forbid floating-point
  atomics, reductions that depend on arrival order, and reduction splits that
  change with batch size, prompt length or occupancy
  ([AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate)).

### The as-stored floor

When the engine writes an answer, each step reads the weights of every layer
and of the selected experts, but does very little arithmetic per byte. Decode
is therefore limited by memory bandwidth, and the fastest a decode kernel can
possibly run is the time to move the bytes it must read. The question that
matters is: which bytes?

Vade's answer is **the bytes as the checkpoint stores them**. A kernel that
first expands MXFP8 weights to BF16 in memory and then reads the BF16 copy
moves twice the bytes it needs. So for decode launches of up to 16 rows, Vade
keeps an MXFP8 "twin" of each expanded projection and reads that instead. The
kernel expands each fragment in registers to exactly the BF16 values the
load-time expansion produced, then runs the same multiply-and-merge order, so
the output bytes are identical to the BF16 path's while it reads half the
weight bytes ([`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs),
[`dense_latency.rs`](../engine/crates/gpu/kernels/src/dense_latency.rs)). The
same idea re-encodes the BF16 output head losslessly at 12 bits per weight
wherever a slot's exponents fit, decoding the identical BF16 bits in the
kernel ([`head.rs`](../engine/crates/engine/src/flash/head.rs)), and the FP4
expert kernels read their packed nibbles directly. Prompt reading, which is
limited by arithmetic rather than bandwidth, keeps the BF16 copy.

That gives every kernel an honest target. The paired microbenchmark
`kernel-bench` computes each case's **compulsory bytes** from the storage
formulas (FP4 at half a byte per value plus its scales, MXFP8 at one byte plus
a scale per 32, BF16 at two), adds the input and output it must touch, and
divides by a fixed bandwidth constant to get a floor time; prompt cases also
take a compute floor and use the larger of the two
([`kernel_floor.rs`](../engine/tools/bench/src/kernel_floor.rs)). A kernel's
score is its floor time divided by its measured time, so 100 % would mean it
moves exactly the stored bytes at full speed, and the tool grades it from A+
to F ([floor and grades](../engine/tools/bench/README.md#floor-and-grades)).
Before it times anything, `kernel-bench` checks that the baseline and the
candidate produce the same output bytes, and by default it exits with an
error on any mismatch ([paired kernel microbench](../engine/tools/bench/README.md#paired-kernel-microbench)).

## Drafting

Drafting is Vade's name for speculative decoding with multi-token prediction
(MTP). Writing one token at a time leaves most of the GPU's arithmetic idle
while it waits on memory. Checking several tokens in one pass reads the
weights once for all of them, so it costs not much more than writing one. If
something cheap can guess the next few tokens, the model can confirm them all
at once.

```text
pending token t0
  │
  ├─ guess: MTP head proposes g1 g2 g3 (prompt lookup may add more)
  │
  ├─ check: the model reads t0 g1 g2 g3 in one pass, as rows
  │         and computes its own choice after each one
  │
  ├─ keep:  the longest run of guesses that match the model's own
  │         choices, plus the model's next token after that run
  │
  └─ roll back: everything after the first mismatch is undone
```

**Where the guesses come from.** The checkpoint's MTP layer proposes a short
chain of tokens; a proposal stops early when the drafter is unsure of its next
guess ([`draft_stack.rs`](../engine/crates/engine/src/flash/draft_stack.rs)).
For greedy requests, prompt lookup also copies guesses from earlier text in
the same chat that matches the last eight tokens
([`next_lookup.rs`](../engine/crates/engine/src/flash/next_lookup.rs)).
Neither source can authorize a token on its own: verification picks every
token that is returned.

**Why the answer does not change.** Under greedy decoding, the kept tokens are
by definition the tokens the model would have chosen one at a time, provided
that checking a guess in a batch of rows computes exactly the same numbers as
decoding it alone. That is what the row kernels guarantee: verification stays
on kernels whose per-row sums do not depend on how many rows share the launch
([`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)). So the
drafting-ON answer is byte-identical to the drafting-OFF answer
([`draft_stack.rs`](../engine/crates/engine/src/flash/draft_stack.rs)).

**Rolling back a hybrid model.** Rolling back an attention layer means
truncating its key/value cache, but a GDN layer's recurrent state has already
absorbed the rejected guesses and cannot be truncated. Vade keeps one snapshot
of the state at the start of the check plus small tapes of each row's
projection inputs, and restores a rejected suffix by replaying only the GDN
recurrence ([`next_mtp.rs`](../engine/crates/engine/src/flash/next_mtp.rs),
[native decode notes](../engine/docs/flashnext-native.md)).

**Turning it ON and OFF.** Drafting is on by default, in the `strict` mode
that keeps only verified tokens. A request can override it
([`protocol.rs`](../engine/crates/serve/src/protocol.rs)):

| In the request | Effect |
| --- | --- |
| nothing | The server default: drafting ON, `strict` |
| `"draft":"strict"` | Drafting ON |
| `"draft":"plain"` | Drafting OFF, one token per step |
| `"mtp_depth":0` | Drafting OFF, when no `draft` mode is named |
| `"mtp_depth":1`..`3` | Drafting ON at that depth |

The default request depth is three
([`next_lookup.rs`](../engine/crates/engine/src/flash/next_lookup.rs)). On
top of that, the engine has three behaviors of its own
([`draft_stack.rs`](../engine/crates/engine/src/flash/draft_stack.rs),
[`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)):

- **Adaptive depth.** For a chat running alone whose request names no
  depth (or names depth 3), the engine picks each cycle's depth to
  maximize the expected committed tokens per millisecond, from that
  request's acceptance so far and the measured costs
  ([`draft.rs`](../engine/crates/engine/src/draft.rs)), up to a ceiling of 7
  tokens, on one Spark and on the layer pipeline.
- **Cohort drafting.** With several chats at once, the release server starts
  in a guarded mode that drafts for groups of up to eight chats, on one Spark
  and on the two-Spark pipeline, keeping each step's verify rows within the
  width of the row kernels. (The library's older automatic mode, which some
  engine notes still describe, stopped at five.)
- **A cost guard.** With many chats at once, the engine compares the
  measured cost per token with and without drafting, and serves chats without
  drafting when that is at least 8 % cheaper per token
  ([`flash-serve.rs`](../engine/crates/tp/src/bin/flash-serve.rs)). On the
  two- and four-Spark layer pipelines by default, and with
  `VADE_MTP_COHORT=wide` on one Spark, five to eight chats instead draft three
  tokens each in one 32-row verify step, and this guard is opt-in;
  `VADE_MTP_COHORT=off` on every pipeline Spark brings the guard back
  ([c8-cohort-experiment.md](../engine/docs/c8-cohort-experiment.md)). In the
  8-chat benchmark runs some chats were served without drafting, and each
  card's README says how many ([Benchmarks](benchmarks.md#known-limits)).

On one Spark, drafting takes code from 27.8 to 76.7 tok/s and prose from 27.8
to 41.7 in the v0.1.0 cards ([what this means](benchmarks.md#what-this-means)).
The older tensor-map rank path never drafts.

<a id="several-sparks-the-layer-pipeline-and-the-chain"></a>
## Several Sparks: the tensor split, layer pipeline and chain

The two-Spark tensor split shares matrix work between both GPUs during a
decode step. It divides eligible expert and projection outputs by tensor
dimensions, gathers the results in a fixed order, and can divide the
linear-attention heads and the vocabulary head too
([`next_tp.rs`](../engine/crates/engine/src/flash/next_tp.rs),
[`next_tp_heads.rs`](../engine/crates/engine/src/flash/next_tp_heads.rs)).
It is the two-Spark default: at startup each rank sets every proven level
that is unset (`VADE_W2_TP=1`, `VADE_W2_TP_WIRE=write`, `VADE_W2_TP_PROJ=3`,
`VADE_W2_TP_HEAD=1`; these names are temporary)
([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)). The split keeps
the layer pipeline's links, recipe and session protocol, so the pipeline
below is also its frame. The engine reference describes the settings and its
limits ([two-Spark tensor split](../engine/docs/next-http.md)).
`VADE_W2_PATH=pipeline`, a debug switch, runs the two-stage layer pipeline
instead until that code is removed.

A Spark has one GPU and one pool of memory. To use three or more, Vade splits
the model by layers, not by matrices. Each Spark runs one
engine process, called a *[rank](GLOSSARY.md#pr)*, and the ranks form a chain
over direct cables:

```text
        HTTP
         │
┌────────▼─────────┐    ┌─────────────────┐    ┌─────────────────┐
│ rank 0 (front)   │───►│ rank 1          │───►│ rank 2 (back)   │
│ scheduler, head, │    │ layers [16, 32) │    │ layers [32, 48) │
│ sampling, drafts │◄───│                 │◄───│                 │
│ layers [0, 16)   │    └─────────────────┘    └─────────────────┘
└──────────────────┘
  final results travel back up the same cables; no ring is closed
```

(The layer counts are for a three-rank chain over the release checkpoint's 48
layers.)

**Who does what.** The front rank keeps the HTTP server, the scheduler, the
embeddings, the output head, sampling and drafting. Each rank runs one
contiguous interval of layers, passes its result to the next, and the final
residuals come back to the front over the same full-duplex cables
([`next_pp.rs`](../engine/crates/engine/src/flash/next_pp.rs),
[`next_service.rs`](../engine/crates/tp/src/flash/next_service.rs)). The
intervals depend only on the layer and rank counts: the layers are divided
evenly, and the front always keeps at least two because layer 1 consumes the
per-layer embeddings. Since each stage needs at least two layers, a chain can
be as long as half the layer count, which is 24 Sparks for this checkpoint.

**The same arithmetic as one Spark.** Every stage runs the one-Spark kernels
in the same layer order and passes FP32 values across each boundary
unchanged, so the pipeline does the same arithmetic as a single Spark
([`next_pp.rs`](../engine/crates/engine/src/flash/next_pp.rs)). Drafting stays
available because verification rows cross the pipeline like any other rows.

**Choosing the layout.** Two Sparks run the tensor split by default. Longer
chains are opt-in with `VADE_W2_PP=1`, which `vade up --sparks N` sets for
you when N is above two
([`settings.rs`](../engine/crates/tp/src/flash/settings.rs),
[`plan.rs`](../cli/src/app/plan.rs)):

| `VADE_W2_PP` | Two Sparks | More than two Sparks |
| --- | --- | --- |
| unset | Tensor split (`VADE_W2_PATH=pipeline`: layer pipeline) | Older tensor-map rank path (four Sparks only) |
| `0` | Older tensor-map rank path | Older tensor-map rank path (four Sparks only) |
| `1` | Layer pipeline | N-stage chain |

**Transport.** Decode steps and short prompt tiles travel on a polled RDMA
(RoCE) wire, while queued prompt tiles stay on TCP; a TCP-only chain is also
supported ([`next_service.rs`](../engine/crates/tp/src/flash/next_service.rs),
[FP4-Local launch](../cli/docs/fp4-local.md)). The pipeline's recipe, which
covers drafting across stages, queued prompt tiles, the RDMA wire, 2048-row
prompt tiles and a bounded output head, is pinned at startup on every rank so
that all of them read one setting
([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)).

**Agreeing before serving.** Before the first request, the ranks verify that
they run the same PTX and checkpoint artifacts, and every worker reserves
memory for the number of chats the front will admit
([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).

**What the measured layer pipeline provides.** In the
2026-10-25 cards the first token after a cold 8K prompt came in 2.80 s on two
Sparks and 2.19 s on four, against 3.86 s on one Spark in the v0.1.0 card.
Only one pipeline GPU works on a given chat at a time. With drafting ON,
code ran at 79.3 tok/s on two Sparks and
80.3 on four, next to 76.7 on one. A chain is mainly for models too large for
a single machine
([reading these cards](benchmarks.md#reading-these-cards)).

## The prefix cache

The prefix cache lets a new request skip reading the start of its prompt when
an earlier request began the same way. It is **off by default**.
`VADE_FLASHNEXT_PREFIX_CACHE=1` turns it on, and then requests that send
`cache:true` or a `prompt_cache_key` keep their prompt for reuse
([`prefix_switch.rs`](../engine/crates/serve/src/prefix_switch.rs)).

A hybrid model makes this harder than usual. In a pure attention model, a
cached prefix is just the first rows of the key/value cache. A GDN layer's
recurrent state, though, summarizes the whole prompt so far, and cannot be cut
back to an earlier position. Vade solves this with checkpoints at aligned
positions. Because prompt tiles always start on absolute 64-token boundaries,
the recurrent state after a prompt's last whole tile is exactly the state a
cold read of any prompt with that prefix would hold at the same position. The
cache saves the GDN states, the PLE history and the draft head's last hidden
row there, copies the attention cache rows below that position, and a later
request resumes reading at the checkpoint
([`next_prefix.rs`](../engine/crates/engine/src/flash/next_prefix.rs)). No
kernel, tile or arithmetic order changes, so a cached answer equals a fresh
one.

Retention is elastic: by default cached prefixes may use up to half the
session pool, and admission reclaims them before it refuses a live
request
([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).
The cache works on one Spark only. The layer pipeline refuses it at startup,
because the back ranks keep no checkpoints of their own layers yet
([`prefix_switch.rs`](../engine/crates/serve/src/prefix_switch.rs)).

## Ranks, memory and ownership

On a Spark, CPU and GPU share one pool of unified memory. At startup the
engine measures what is available, keeps back a safety reserve of a tenth of
it (at least 2 GiB), loads one copy of the weights, and plans a session pool
from what is left; if one full-context chat does not fit, it refuses to start
and says why ([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).
It then keeps five kinds of memory apart:

| Memory | What it holds | How long it lasts |
| --- | --- | --- |
| Resident weights | One copy of this rank's share of the model, shared by all its chats | Until the engine shuts down |
| Active chat state | One chat's attention cache, recurrent state and execution buffers | Until the chat ends |
| Retained prompt state | The prefix cache's checkpoints | Until reclaimed under the pool's budget |
| Shared scratch | Working space and CUDA graphs that every model step reuses | Reserved at startup |
| Host staging and transport | Buffers for loading and for network transfers | Kept for reuse |

*Memory admission* is the check that a request fits. It charges each request
for the tokens it may grow to, and reclaims idle cache first
([`dispatch.rs`](../engine/crates/serve/src/scheduler/dispatch.rs)). Freed chat buffers wait in a
reserve for the next request instead of being returned to the driver, because
allocating device memory is slow enough to leave the GPU idle between
requests ([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).
On several Sparks, every rank funds the same admitted chats, so a request the
front accepts also fits downstream. Cached tokens still count toward the
context length: reuse saves work, but does not raise the context limit.

The engine tracks the device bytes it allocates
([`flash-serve.rs`](../engine/crates/tp/src/bin/flash-serve.rs)). That figure
is good for comparing one build with another, but it leaves out some driver
allocations and is not the Spark's total physical memory use. Each benchmark
[receipt](GLOSSARY.md#pr) states which memory measure it reports.

## Exactness and model quality

Vade's exactness rule is: **the same setup and request produce identical token
IDs on every Spark, every run**. The setup includes the build, checkpoint,
tokenizer and template, rank layout, thinking and the sampling settings in
effect. Within one setup, the answer must be byte-identical whether a request
runs alone or batched, whether its prefix is cached or read fresh, however the
prompt is split into chunks, and, under greedy decoding, with drafting ON or
OFF. Request IDs and timestamps in HTTP envelopes are not part of the answer.
The mechanisms behind this are the ones above: fixed reduction trees, row
kernels, aligned tiles and checkpoints, and verification that picks every
token.

`vade verify` checks it on your own machine. It sends the same greedy
requests repeatedly with drafting OFF and ON, compares every token ID, and
fails if an ON run did not actually draft. `--machine` adds an independent
engine on another machine to the comparison
([`verify/mod.rs`](../cli/src/verify/mod.rs)). It is a targeted check, not
proof of every batching, cache or chunking combination; the
[cohort tests](../engine/crates/serve/tests/cohort_mtp.rs) and the device gate
in [AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate) cover more. Any
change to engine arithmetic, scheduling, caching or transport must match
across at least three runs on at least two Sparks before it merges.

Exactness is a different question from quality. Three checks answer three
questions:

- **Repeatability:** does this build keep the same-setup rule on every
  execution path?
- **Previous-build preservation:** do the saved test cases still match an
  older Vade build? A new build or a different Spark count is a different
  setup, so its answers may change, but only if the quality checks pass.
- **Model quality:** does a change keep task accuracy, instruction following
  and long-context retrieval? Matching an older Vade build is a separate
  question from matching an outside reference runtime.

An intentional change to arithmetic order needs the full exactness gate,
reviewed golden updates and quality evidence against main before it becomes
the default. Record the change and its results in a benchmark card. Keep one
production path; a runtime switch does not replace this evidence. Quality
checks include layer differences against the measured numerical noise floor,
GSM8K and IFEval comparisons, and long-memory checks for large changes
([CLI validation](../cli/docs/test.md), [evals](../cli/evals.md)).

## What is measured, and how

Vade measures at four levels, and keeps their claims apart. Every published
number is tied to a **receipt**: the saved record of one run, with its build, hardware,
checkpoint, settings and raw results.

| Level | Tool | What it shows | What it does not show |
| --- | --- | --- | --- |
| Kernel | `kernel-bench` | Time per kernel against the as-stored floor, baseline and candidate in one process, with identical output bytes required | End-to-end speed, or exactness of the whole model |
| Server | `vade bench` | Time to first token, tokens per second and concurrent throughput, measured from outside over HTTP | Model quality |
| Card | RigMark | The published cards: prose, code, structured output, cold 8K prompts, and 1 to 8 chats at once | Exactness; its output checks only confirm that answers completed |
| Exactness | `vade verify` | Identical token IDs across runs and drafting modes | Speed, or quality |

`vade bench` counts real tokens from the server's usage figures, never
streamed chunks, and if any request in a group stops early or fails, the whole
group gets no rate ([bench library](../engine/tools/bench/README.md),
[run vade bench](benchmarks.md#run-vade-bench)). The published cards come from
RigMark ([run RigMark](benchmarks.md#run-rigmark-against-vade)), and each
links its sanitized receipt folder.

A few v0.1.0 results, all from [Benchmarks](benchmarks.md#previous-2026-10-24-v010-release-5a22651),
on one home Spark with drafting ON unless noted:

| Measure | Result |
| --- | --- |
| One chat, code | 76.7 tok/s (27.8 with drafting OFF) |
| One chat, prose | 41.7 tok/s (27.8 with drafting OFF) |
| Eight chats together | 173.9 tok/s |
| First token after a cold 8K prompt | 3.86 s; 2.79 s on two Sparks |

The benchmark page explains the recipe, the run-to-run ranges, the cloud VMs
and the limits of each figure: see
[client time and server time](benchmarks.md#client-time-and-server-time) and
[what these results do not show](benchmarks.md#what-these-results-do-not-show).

## The harness and the CLI

### Native agent mode

The harness's native agent works inside a workspace you choose. Reads, search
and exact patches stay within it. With the default `--approval ask`, edits and
commands outside the
[read-only allowlist](../harness/docs/agent-tools.md#commands-and-approvals)
wait for your approval. A tool's result is written to a private journal and
sent back to the engine as part of the next request; the engine never runs a
tool itself. An engine can advertise the optional `vade.engine.v1` contract,
which lets the harness ask whether a request will fit before sending it
([engine contract](../harness/docs/engine-contract.md)).

Shell commands run only on macOS for now, inside `sandbox-exec` with a cleared
environment and no network. On Linux, including the Spark itself, the agent
can read and patch files but will not run commands; to let it run commands,
start the harness on a Mac and point it at the Spark
([harness README](../harness/README.md#native-coding-agent)).

### Gateway mode

The gateway relays another client's requests to one or more engines, as
described in [Follow one request](#follow-one-request). It speaks plain HTTP
with no built-in TLS, so between machines use the
[SSH tunnel](../engine/docs/flash-server.md#connect-your-mac) that the
controller route sets up. A cleanly closed connection does not prove the
remote GPU has stopped working on that request
([gateway contract](../harness/docs/engine-contract.md#opaque-gateway)).

### What the CLI starts and stops

`vade launch` runs on the Spark: it finds an existing checkpoint, builds or
uses local artifacts, starts the engine and opens chat. From a separate
[controller](GLOSSARY.md#ac), `vade install` stages a build and `vade up`
starts the selected ranks over SSH, plus a tunnel it owns
([`launch.rs`](../cli/src/launch.rs), [INSTALL.md](../INSTALL.md)).
`vade status` and `vade logs` inspect the installation, and `vade down` stops
only the processes the CLI recorded, after checking each one's identity.

## Current limits

This is Vade 0.1.0, an early release. Its default is FP4-Local on one Spark. The
benchmark page also has results from cabled pairs of Sparks and from rented
cloud VMs; chains longer than two Sparks need their own live exactness
qualification ([FP4-Local launch](../cli/docs/fp4-local.md)). The other model
parsers, servers and protocol routes in the source are not qualified to the
same level.

- The prefix cache is off by default and works on one Spark only.
- `"temperature": 0` alone is not greedy; it brings in the checkpoint's other
  sampling settings.
- A cancel is handled by the scheduler between model steps, so it waits for
  the GPU work already running
  ([`dispatch.rs`](../engine/crates/serve/src/scheduler/dispatch.rs)).
- The agent's shell tool needs macOS.
- Speed numbers do not cover cache accounting, isolation of one failing
  request from the rest of its batch, long sessions, or complete coding
  workflows; each needs its own test.

Continue with [Benchmarks](benchmarks.md), the
[engine package map](../engine/crates/README.md), the
[harness docs](../harness/docs/README.md), or the [docs index](README.md).
