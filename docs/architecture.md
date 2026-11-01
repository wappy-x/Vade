# How Vade works

Vade runs a large language model on an NVIDIA DGX Spark and answers chat requests over HTTP. It is one repository with three programs:

- **The engine.** The inference engine runs the model on the Spark's GPU.
- **The harness.** The agent harness turns the model's replies into a coding conversation.
- **The CLI.** The CLI builds, starts and stops the other two.

All three are first-party Rust built on the standard library alone. The engine talks straight to NVIDIA's driver.

Vade serves `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one Spark by default. On main, two Sparks run it as a tensor split. Four Sparks over RDMA run the tensor split in parts. Four Sparks over TCP and every other count of three or more run it as a chain. Main also serves `Qwen/Qwen3.8-Flash-Next-FP8` on 2, 4 or 6 Sparks.

Two ideas shape almost every design choice on this page:

- **Exactness.** The same request, with the same settings, on the same setup returns the same token IDs every time. This holds when the request runs alone or in a batch with other chats. It holds when the prompt comes from the cache or is read fresh, and however the prompt is split into chunks. For greedy requests, it also holds with drafting on or off.
- **The bytes as stored.** The speed at which Vade writes an answer is limited by how fast the GPU can read weights from memory. So the kernels read each weight in the format that the checkpoint stores it in. We judge each kernel against the time it would take just to move those bytes.

First we follow one request from the network to the GPU and back. Then we explain each part: the model and its number formats, the kernels, drafting, several Sparks, the prefix cache, memory, exactness, and how we measure it. To install Vade, follow [INSTALL.md](../INSTALL.md). For the measured results, read [Benchmarks](benchmarks.md).

These words come up in every section of this page:

- **Prefill and decode.** Prefill reads the prompt, many tokens at a time. Decode writes the answer, one token per step.
- **BF16, FP8 and FP4.** These are 16-, 8- and 4-bit number formats. FP32 is the ordinary 32-bit float. Vade uses FP32 for running sums and activations.
- **Kernel and PTX.** A kernel is a small program that runs on the GPU. PTX is NVIDIA's portable assembly language for kernels.
- **Checkpoint.** A checkpoint is the published set of model files: weights, configuration and tokenizer.

This page defines every other term where it first appears. The [glossary](GLOSSARY.md) defines them too.

## The three parts

- **[Engine](../engine/README.md).** The engine loads the model and does the tokenization, GPU execution, sampling, request scheduling and model-state caches. It is not responsible for the workspace tools or the permission to edit your files.
- **[Harness](../harness/README.md).** The harness is responsible for the conversation history, the terminal UI, the tools, the approvals, the private session journals and the optional gateway. It is not responsible for the model weights, the GPU execution or the engine's cached state.
- **[CLI](../cli/README.md).** The CLI gets and builds the parts. It is responsible for the configuration, the processes that it owns, and startup and shutdown. It is not responsible for the model arithmetic or the agent's tool decisions.

The engine and the harness are separate programs in separate Cargo workspaces. Any OpenAI-compatible client can use the engine, and the harness can point at another compatible server. The engine itself is a workspace of packages. These packages are on the release path:

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

**The gateway, if you use one.** A client can talk to the engine directly, or through a Vade gateway.

- **Harness gateway.** The harness's gateway mode listens on `127.0.0.1:8787` by default ([`args.rs`](../harness/src/cli/args.rs)).
- **Standalone gateway.** `vade-gateway` is a loopback-only relay. It forwards every path and does not translate between APIs ([invisible gateway](../harness/docs/invisible-gateway.md)).

When a request names none of the sampling fields, the harness gateway inserts greedy defaults. `vade-gateway` does so only when you start it with `--greedy-when-unset` ([`invisible.rs`](../harness/src/gateway/invisible.rs)). If a request names any sampling field, its body passes through byte for byte.

That rule is defined once, in the engine's model crate ([`sampling_policy.rs`](../engine/crates/model/src/sampling_policy.rs)). The harness compiles the same source file instead of a dependency on the engine ([`sampling.rs`](../harness/src/gateway/sampling.rs)). With several upstream engines, the gateway can keep a conversation on the same engine. That affinity is a routing hint. It is never a claim that a cache was hit ([routing](../harness/docs/gateway-routing.md)).

**The HTTP layer.** On rank zero, `flash-serve` binds a standard-library `TcpListener` ([`flash-serve.rs`](../engine/crates/tp/src/bin/flash-serve.rs)). The server's HTTP code is Vade's own ([`http.rs`](../engine/crates/serve/src/http.rs)). It refuses a non-loopback bind unless you pass `--allow-remote`. Without that flag, it stops with `remote bind requires explicit --allow-remote`.

**Preparing the request.** The engine thread reads sockets without blocking. It hands each complete request to a separate thread. That thread parses the JSON, applies the chat template and tokenizes the prompt. So the parse of a long prompt never holds up GPU work ([`scheduler.rs`](../engine/crates/serve/src/scheduler.rs)). The tokenizer is Vade's own BPE implementation ([tokenizer](../engine/crates/tokenizer/README.md)).

Sampling is decided here too:

- **FP4-Local.** A request that names no sampling field is decoded greedily. A request that names any one of them takes the checkpoint's own settings for the rest ([`protocol.rs`](../engine/crates/serve/src/protocol.rs)).
- **Greedy steps.** A step is greedy when the temperature is 0 and the penalties are neutral ([`draft.rs`](../engine/crates/engine/src/draft.rs)). FP4-Local's own penalties are neutral, so `"temperature": 0` alone stays greedy.
- **FP8.** A field left out takes the checkpoint's sampled value. So send `"temperature": 0` with the three neutral penalties ([OpenAI API](openai-api.md)).

**The scheduler.** Everything that touches the GPU runs on one thread. The scheduler loop accepts prepared requests and checks that each one fits in the memory pool. It looks for a reusable cached prefix. Then it decides which rows of work run in the next model step.

Decode steps for all active chats run together in one batch. Prompt reading shares the GPU with them in bounded time slices, so neither one starves the other. A request can also carry a priority that lets foreground work go first ([`scheduler.rs`](../engine/crates/serve/src/scheduler.rs), [`dispatch.rs`](../engine/crates/serve/src/scheduler/dispatch.rs)). On one Spark, the engine holds up to sixteen resident chats, as many as memory admits ([`next_batched.rs`](../engine/crates/engine/src/flash/next_batched.rs)).

**The model step.** The engine reads the prompt in tiles of thousands of rows at a time. It writes the answer one step at a time, with every active chat in the same step. Decode steps replay recorded CUDA graphs. They do not launch each kernel from the host. With drafting on, a step can also check several guessed tokens per chat (see [Drafting](#drafting)).

**Back to the client.** The engine turns each new token back into text and streams it at once, as server-sent events. At the end, the server reports completion and usage figures. For the harness, text that a client sees during the stream is provisional. A tool call runs only when its proposal is complete and has passed the workspace rules and your approval policy (see [the harness](#the-harness-and-the-cli)).

## Which program serves what

The repository holds more than one server, from different stages of the project. Only one is the release path.

- **[`flash-serve`](../engine/crates/tp/src/bin/flash-serve.rs): the release server.** It serves FP4-Local on one Spark, across two Sparks, or as a longer chain. It also serves the Flash-Next FP8 checkpoint on 2, 4 or 6 Sparks. `vade launch` and `vade up` start it.
- **[`spark-serve`](../engine/crates/serve/src/bin/spark-serve.rs): the original one-Spark server.** It serves earlier models (Qwen3, Qwen3.8-27B, Gemma 4). It shares the HTTP and scheduler library with `flash-serve`, but it is not a v0.1.0 route. See [its README](../engine/crates/serve/README.md).
- **[`next-http`](../engine/crates/tp/src/bin/next-http.rs): a compatibility launcher.** It hands off to `flash-serve` on the older tensor-map rank path, with drafting off.
- **[`next-tp`](../engine/crates/tp/src/bin/next-tp.rs): the rank qualification tool.** It is not an HTTP service.
- **[`tp-serve`](../engine/crates/tp/src/bin/tp-serve.rs): an older HTTP service.** It serves tensor-parallel rank commands.

`flash-serve` reads the checkpoint's own metadata to choose its model code. The mixed storage of an FP4-Local checkpoint selects `NextModel`. The `--fp4-local` flag only asserts that the checkpoint really is that kind ([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)).

## Standard-library Rust, and why

Vade has **zero third-party dependencies**. Every Rust package uses only `std` (and `core` on the GPU) plus the first-party crates in this tree. A test in the CLI counts the dependencies in every manifest and lockfile. It fails if the count rises above zero ([`ratchet.rs`](../cli/tests/ratchet.rs), [baseline](../cli/ratchet-baseline.txt)).

So Vade carries its own versions of things that most projects import:

- **HTTP.** Vade has its own HTTP/1.1 server.
- **JSON.** Vade has its own bounded JSON parser ([engine-base](../engine/crates/base/README.md)).
- **Hashes.** Vade has its own streaming SHA-256 for checkpoint verification ([engine-hash](../engine/crates/hash/README.md)).
- **Tokenizer.** Vade has its own BPE tokenizer.
- **CUDA driver.** Vade has its own hand-declared interface to the CUDA driver ([`driver.rs`](../engine/crates/gpu/src/driver.rs)).
- **RDMA.** Vade reaches RDMA through `libibverbs`, which it loads at run time ([`verbs.rs`](../engine/crates/fabric/src/verbs.rs)).
- **Kernels.** Vade has its own code for every GPU kernel.

No CUDA toolkit, cuBLAS, CUTLASS, NCCL or Python runs in the engine. When two workspaces need the same code, one compiles the other's source file with a `#[path]` attribute. It does not add a Cargo dependency.

We pay that cost for three reasons:

1. **One arithmetic order everywhere.** Exactness depends on every floating-point sum being added in the same order on every path. That is possible only if we own every kernel that touches the numbers. A library that picks a different reduction split for a different batch size would break the promise silently. The [kernel crate](../engine/crates/gpu/kernels/README.md) states this as its reason to exist.
2. **The whole stack is readable.** Every line between the HTTP socket and the GPU instruction is in this repository. So a reviewer can audit it, and a bug cannot hide in a version bump. No dependency update can change behavior between builds. When the pinned toolchains are installed, the host checks run fully offline ([AGENTS.md](../AGENTS.md#local-validation)).
3. **Small attack surface.** No registry or git dependency can be swapped in the supply chain, because there are none.

The price is `unsafe` code at four boundaries that the standard library cannot cover: the kernels themselves, the CUDA driver calls, the kernel launches, and the RDMA memory shared with the network card. The [unsafe inventory](../engine/docs/unsafe.md) counts every site. The house rule is that no change may add new ones.

## What the current Flash-Next path runs

### The model

Flash-Next is a hybrid. The model code derives its layer list from the checkpoint's `layer_types`. Each `linear_attention` layer becomes a GatedDeltaNet layer. Each `full_attention` layer becomes an indexed sparse attention layer ([`flash/mod.rs`](../engine/crates/model/src/flash/mod.rs)).

- **Indexed attention (QSA).** These layers are attention layers. For every token read so far, they store two vectors, a *key* and a *value*. Later tokens compare against them to decide what to look back at. Those stored vectors are the **KV cache**, and it grows with the conversation. Here a small indexer scores blocks of earlier tokens, and attention reads only the selected blocks plus the newest tokens ([`qsa.rs`](../engine/crates/engine/src/flash/qsa.rs), [indexer contract](../engine/docs/qsa-indexer.md)).
- **GatedDeltaNet (GDN).** These layers are linear recurrent layers. They do not store every past token. Each one folds what it has read into a fixed-size state, which it updates as each token arrives ([`gdn.rs`](../engine/crates/engine/src/flash/gdn.rs)).
- **Routed experts.** After attention, a router picks a few expert networks for each token, next to a shared expert. So a decode step reads only the weights of the selected experts ([`moe.rs`](../engine/crates/engine/src/flash/moe.rs)).
- **Hyper-connections.** These keep four parallel residual streams instead of one. The streams are mixed before each block and combined after it ([`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs)).
- **Per-layer and n-gram embeddings (PLE).** These add a large table that enters at layer 1. A hash of the most recent tokens selects the row.
- **An MTP layer.** The multi-token prediction layer predicts tokens beyond the next one. Vade uses it to draft.

This is the path of one token through the model ([`next_batched.rs`](../engine/crates/engine/src/flash/next_batched.rs)):

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

Activations stay in FP32 storage throughout ([`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs)). The tree also keeps an FP32 reference implementation of the decoder on the CPU, written from the published PyTorch model. It is the oracle that the GPU path is checked against ([`refcpu.rs`](../engine/crates/model/src/flash/refcpu.rs)).

### FP4-Local and its number formats

"FP4-Local" names how the checkpoint stores its tensors, and it is a mix. The loader reads the storage of each module from the checkpoint's quantization map. It checks the tensor inventory against that map ([`mixed.rs`](../engine/crates/model/src/flash/mixed.rs)).

- **NVFP4 (NVIDIA's 4-bit float format).** The layout is 4-bit E2M1 values, one E4M3 scale per 16 values and one FP32 multiplier per matrix ([`nvfp4.rs`](../engine/crates/model/src/flash/nvfp4.rs)). It stores the routed experts, the experts of the MTP layer and the PLE table.
- **MXFP8 (microscaling 8-bit float).** The layout is 8-bit E4M3 values with one power-of-two UE8M0 scale per 32 values ([`mxfp8.rs`](../engine/crates/model/src/flash/mxfp8.rs)). It stores the attention-side projections (full attention, GDN and the indexer) and the shared experts.
- **BF16 and FP32.** These are plain storage. They hold the output head, the routers, the norms and the remaining small tensors.

Three details matter for exactness and speed:

- **Lossless scale compression.** FP4-Local stores the NVFP4 scales in a compressed form. The loader expands them back to the exact E4M3 bytes, and every byte pattern survives the round trip ([`csf_expand`](../engine/crates/model/src/flash/nvfp4.rs)).
- **FP4 weights, BF16 arithmetic.** The native FP4 matrix instruction of the GB10 needs FP4 activations too. Vade keeps activations at higher precision instead. The expert kernels decode each FP4 weight in registers and multiply on the BF16 tensor cores. They read every packed weight byte once per tile ([`fp4_experts.rs`](../engine/crates/gpu/kernels/src/fp4_experts.rs), [`fp4_exact.rs`](../engine/crates/gpu/kernels/src/fp4_exact.rs)). Vade does not implement the checkpoint's 4-bit activation quantizer ([`fp4_abi.rs`](../engine/crates/gpu/src/fp4_abi.rs)).
- **FP32 activations on BF16 tensor cores.** Projections split each FP32 activation into a BF16 high part and a BF16 residual part. Both parts go through the tensor cores with FP32 sums. This is the one projection mode. There is no lower-precision shortcut ([Flash arithmetic](../engine/docs/flash-fp8-math.md), [`flash_bf16.rs`](../engine/crates/gpu/kernels/src/flash_bf16.rs)).

The target model and its verification of drafts make every returned token. That arithmetic never quantizes the checkpoint again. It reads the tensors in the formats above. The BF16 output head can be encoded again in place as LBF12, a lossless packing that decodes to the identical BF16 bits. `VADE_HEAD_LBF12=0` keeps plain BF16 for the head. On FP4-Local, the same setting also leaves out the LBF12 copies of the HC up weights.

On FP8, decode rows also read LBF12 copies, but only on a rank whose weights are resident. A rank that streams its weights gets no copies. A rank streams its weights when drafting is off and its budget is below the weights plus the reserve. A resident FP8 rank gets these copies:

- **HC up weights.** It gets copies of all of its HC up weights. These copies have no room check, and the load plan does not count them.
- **Attention projections.** It gets copies of its packed BF16 attention projections with K 2560 or 3072, including those of the MTP layer: GDN `in_proj_qkv`, `in_proj_z` and `out_proj`, and QSA `q_proj` (query and gate) and `o_proj` ([`attention_matrices`](../engine/crates/engine/src/flash/weights.rs)). Vade makes each of these copies only when it fits above the execution reserve. QSA `k_proj` and `v_proj`, the indexer's `index_qk_proj`, and GDN `in_proj_a` and `in_proj_b` get no copy.

An LBF12 copy takes 392 bytes for each block of 8 rows by 32 columns (512 bytes in BF16), plus flag words (`lbf12_bytes` in [`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs)). Each rank prints the sizes at startup. On two Sparks with MTP depth 2 and context 32768, both ranks printed `flash_session attention_lbf12 matrices=134 bytes=2059973376` and `flash_session hc_up_lbf12 matrices=100 bytes=502272000`.

The drafter is the exception, and it is on by default whenever drafting is on. At load, Vade builds lower-precision copies that only propose tokens:

- **Output head.** Vade builds an MXFP8 copy and an NVFP4 copy of the scored output-head rows. `VADE_FLASHNEXT_DRAFT_MX8=0` and `VADE_FLASHNEXT_DRAFT_FP4=0` turn them off.
- **MTP layer.** Vade builds MXFP8 copies of the seven BF16 matrices of the MTP layer: `fc_embedding`, `fc_hidden`, `q_proj`, `k_proj`, `v_proj`, the indexer's `index_qk_proj` and `o_proj`. `VADE_FLASHNEXT_DRAFT_MX8_LAYER=0` turns them off.

The code is in [`next_batched.rs`](../engine/crates/engine/src/flash/next_batched.rs), [`head.rs`](../engine/crates/engine/src/flash/head.rs) and [`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs). Verification keeps the full BF16 head and the checkpoint's weights. So these copies change which tokens are proposed and how many are accepted. They do not change the token IDs returned.

Any other change that would compute below the checkpoint's precision needs three things before it can become the default: the full exactness gate, quality evidence against main and a reviewed card. Keep one production path, as [AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate) requires.

### The FP8 checkpoint

Vade also serves `Qwen/Qwen3.8-Flash-Next-FP8`, on 2, 4 or 6 Sparks. Our cards cover two and four Sparks.

That checkpoint stores FP8 weights in 128×128 blocks with dynamic activation scales ([`flash/mod.rs`](../engine/crates/model/src/flash/mod.rs)). Its experts follow the reference W8A8 arithmetic recorded in [Flash arithmetic](../engine/docs/flash-fp8-math.md).

`flash-serve` serves it with tensor parallelism: every Spark runs every layer and holds its share of each one. `vade up --fp8` selects it. It is a supported route with its own [installation guide](install-fp8.md).

## The kernels and the as-stored floor

### Rust on the GPU

The kernels are ordinary `#![no_std]` Rust. The NVPTX backend of the official Rust compiler compiles them to PTX, NVIDIA's portable GPU assembly ([`lib.rs`](../engine/crates/gpu/kernels/src/lib.rs)). A few instructions have no Rust spelling, such as the tensor-core multiply and asynchronous copies into shared memory. For those, the kernels use LLVM intrinsics and short inline PTX.

`cargo kernels` builds the kernels with the pinned nightly compiler. It stamps the PTX with a hash of its sources and flags ([build-kernels](../engine/crates/gpu/build-kernels/README.md)). At startup, `flash-serve` hands the PTX to the driver, which compiles it to GB10 machine code ([`driver.rs`](../engine/crates/gpu/src/driver.rs)).

### Exact by construction

The kernels add the terms of each row in a fixed order. So the result of a row does not depend on what else is running:

- **Fixed reduction trees.** A projection splits its inner dimension into four ranges. The boundaries of the ranges depend only on the matrix shape. The projection merges them as `(p0 + p1) + (p2 + p3)`, independent of the number of tokens or the output tiling ([`canonical.rs`](../engine/crates/gpu/kernels/src/canonical.rs), [`flash_bf16.rs`](../engine/crates/gpu/kernels/src/flash_bf16.rs)).
- **Row kernels.** Decode and draft verification use kernels whose per-row sums never depend on how many rows share the launch. So the bytes of a chat in a batch equal its bytes alone ([`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)).
- **Aligned prompt tiles.** Prompt tiles start on absolute 64-token boundaries. The rows inside a tile are exact whatever the tile size. So the way a prompt is chunked does not change its result ([`next_prefix.rs`](../engine/crates/engine/src/flash/next_prefix.rs), [`next_rows.rs`](../engine/crates/engine/src/flash/next_rows.rs)).
- **No order-dependent arithmetic.** The house rules forbid floating-point atomics, reductions that depend on arrival order, and reduction splits that change with batch size, prompt length or occupancy ([AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate)).

### The as-stored floor

When the engine writes an answer, each step reads the weights of every layer and of the selected experts. But it does very little arithmetic per byte. So memory bandwidth limits decode. The fastest time for a decode kernel is the time to move the bytes that it must read. The question that matters is this: which bytes?

Vade's answer is **the bytes as the checkpoint stores them**. A kernel that first expands MXFP8 weights to BF16 in memory and then reads the BF16 copy moves twice the bytes that it needs. So for decode launches of up to 16 rows, Vade keeps an MXFP8 "twin" of each expanded projection and reads the twin instead.

The kernel expands each fragment in registers to exactly the BF16 values that the load-time expansion made. Then it runs the same multiply-and-merge order. So its output bytes are identical to those of the BF16 path, and it reads half the weight bytes ([`blocks.rs`](../engine/crates/engine/src/flash/blocks.rs), [`dense_latency.rs`](../engine/crates/gpu/kernels/src/dense_latency.rs)).

The same idea applies in two more places:

- **Output head.** Vade encodes the BF16 output head again, losslessly, at 12 bits per weight wherever the exponents of a slot fit. The kernel decodes the identical BF16 bits ([`head.rs`](../engine/crates/engine/src/flash/head.rs)).
- **FP4 experts.** The FP4 expert kernels read their packed nibbles directly.

Prompt reading is limited by arithmetic, not bandwidth, so it keeps the BF16 copy.

So every kernel has an honest target. The paired microbenchmark `kernel-bench` computes the **compulsory bytes** of each case from the storage formulas: FP4 at half a byte per value plus its scales, MXFP8 at one byte plus a scale per 32, and BF16 at two. It adds the input and output that the case must touch. Then it divides by a fixed bandwidth constant to get a floor time. Prompt cases also take a compute floor and use the larger of the two ([`kernel_floor.rs`](../engine/tools/bench/src/kernel_floor.rs)).

The score of a kernel is its floor time divided by its measured time. So 100 % would mean that it moves exactly the stored bytes at full speed. The tool grades each kernel from A+ to F ([floor and grades](../engine/tools/bench/README.md#floor-and-grades)). Before it times anything, `kernel-bench` checks that the baseline and the candidate make the same output bytes. By default, it exits with an error on any mismatch ([paired kernel microbench](../engine/tools/bench/README.md#paired-kernel-microbench)).

## Drafting

Drafting is Vade's name for speculative decoding with multi-token prediction (MTP). When the GPU writes one token at a time, most of its arithmetic waits on memory. A check of several tokens in one pass reads the weights once for all of them, so it costs not much more than one token. If something cheap can guess the next few tokens, the model can confirm them all at once.

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

**Where the guesses come from.** The checkpoint's MTP layer proposes a short chain of tokens. A proposal stops early when the drafter is unsure of its next guess ([`draft_stack.rs`](../engine/crates/engine/src/flash/draft_stack.rs)). For greedy requests, prompt lookup also copies guesses from earlier text in the same chat that matches the last eight tokens ([`next_lookup.rs`](../engine/crates/engine/src/flash/next_lookup.rs)). Neither source can authorize a token on its own. Verification picks every token that is returned.

**Why the answer does not change.** Under greedy decoding, the kept tokens are by definition the tokens that the model would choose one at a time. This holds if a check of a guess in a batch of rows computes exactly the same numbers as decode of that guess alone. The row kernels guarantee that. Verification stays on kernels whose per-row sums do not depend on how many rows share the launch ([`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)). So the drafting-on answer is byte-identical to the drafting-off answer ([`draft_stack.rs`](../engine/crates/engine/src/flash/draft_stack.rs)).

**Rolling back a hybrid model.** To roll back an attention layer, Vade truncates its key/value cache. But the recurrent state of a GDN layer has already absorbed the rejected guesses, and Vade cannot truncate it. So Vade keeps one snapshot of the state at the start of the check, plus small tapes of the projection inputs of each row. It restores a rejected suffix by a replay of only the GDN recurrence ([`next_mtp.rs`](../engine/crates/engine/src/flash/next_mtp.rs), [native decode notes](../engine/docs/flashnext-native.md)).

**Turning it on and off.** Drafting is on by default, in the `strict` mode that keeps only verified tokens. A request can override it ([`protocol.rs`](../engine/crates/serve/src/protocol.rs)):

<div align="center">

| In the request | Effect |
| :---: | :---: |
| nothing | The server default: drafting on, `strict` |
| `"draft":"strict"` | Drafting on |
| `"draft":"plain"` | Drafting off, one token per step |
| `"mtp_depth":0` | Drafting off, when no `draft` mode is named |
| `"mtp_depth":1`..`3` | Drafting on at that depth |

</div>

The default request depth is three ([`next_lookup.rs`](../engine/crates/engine/src/flash/next_lookup.rs)). In addition, the engine has three behaviors of its own ([`draft_stack.rs`](../engine/crates/engine/src/flash/draft_stack.rs), [`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)):

- **Adaptive depth.** This applies to a chat that runs alone, when its request names no depth or names depth 3. For that chat, the engine picks the depth of each cycle to get the most expected committed tokens per millisecond. It uses the acceptance of that request so far and the costs ([`draft.rs`](../engine/crates/engine/src/draft.rs)). The ceiling is 7 tokens, on one Spark, on the tensor split and on the layer chain. On FP4-Local, the costs are the verify cost prior and a draft price. Each chat measures that price over its first 16 cycles, rounds it to a power of two and keeps it. So the same request on the same setup drafts the same depths every run.
- **Cohort drafting.** With several chats at once, the release server starts in a guarded mode. It drafts for groups of up to eight chats, on one Spark and on the two-Spark tensor split. It keeps the verify rows of each step within the width of the row kernels. The older automatic mode of the library stopped at five. Some engine notes still describe that mode.
- **A cost guard.** With many chats at once, the engine compares the measured cost per token with and without drafting. It serves chats without drafting when that is at least 8 % cheaper per token ([`flash-serve.rs`](../engine/crates/tp/src/bin/flash-serve.rs)). Some setups work differently. On the two-Spark tensor split and on four Sparks (the split in parts or the layer chain) by default, and with `VADE_MTP_COHORT=wide` on one Spark, five to eight chats draft three tokens each in one 32-row verify step instead. There the guard is opt-in, and `VADE_MTP_COHORT=off` on every pipeline Spark brings it back ([c8-cohort-experiment.md](../engine/docs/c8-cohort-experiment.md)). In the 8-chat benchmark runs, some chats were served without drafting. The README of each card says how many ([Benchmarks](history/benchmarks-2026-10.md#known-limits)).

On one Spark, drafting takes code from 27.8 to 76.7 tok/s and prose from 27.8 to 41.7 in the v0.1.0 cards ([what this means](history/benchmarks-2026-10.md#what-this-means)). The older tensor-map rank path never drafts.

<a id="several-sparks-the-layer-pipeline-and-the-chain"></a>
## Several Sparks: the tensor split, layer pipeline and chain

The two-Spark tensor split shares matrix work between both GPUs during a decode step. It divides eligible expert and projection outputs by tensor dimensions and gathers the results in a fixed order. It can also divide the linear-attention heads and the vocabulary head ([`next_tp.rs`](../engine/crates/engine/src/flash/next_tp.rs), [`next_tp_heads.rs`](../engine/crates/engine/src/flash/next_tp_heads.rs)).

It is the only two-Spark path, and every proven level is the only code. No environment name selects or tunes a level ([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)). The split keeps the links, recipe and session protocol of the layer pipeline, so the pipeline below is also its frame. The engine reference describes the split and its limits ([two-Spark tensor split](../engine/docs/next-http.md)). It needs RDMA.

**Cold prompts on two Sparks.** Two Sparks read a cold prompt in one of three ways:

- **2,048 to 8,192 tokens: the token split** ([`next_tp_seq.rs`](../engine/crates/engine/src/flash/next_tp_seq.rs)). The first Spark reads the first half of the prompt through the first two layers and hands it on. The second Spark reads it through the other 46 layers. At the same time, the first Spark reads the second half through every layer, one layer behind. Before each layer, it takes the attention and linear-attention state of the second Spark, so the math of each token is the same as on one Spark. After the first token, the first Spark sends the rest of the state back, and decode runs on the tensor split.
- **More than 8,192 and at most 20,480 tokens: layer by layer on both Sparks** ([`next_split.rs`](../engine/crates/engine/src/flash/next_split.rs)). The first Spark reads every token through the first two layers. The second Spark reads the first half through the other 46 layers, in tiles of 2,048 tokens, and hands on the state of each layer. The first Spark reads the second half through each layer after it takes that state. So the first token comes sooner. `VADE_LONG_SPLIT=0` turns this path off.
- **Longer prompts: two-stage pipeline tiles.** Prompts on a Spark that cannot hold the extra rows run this way too. A prompt of at least seven workspaces (28,672 rows with the 4,096-row default) takes whole-workspace tiles in the middle. Its first and last tiles keep 2,048 rows ([`tp_switch.rs`](../engine/crates/engine/src/flash/tp_switch.rs)). Other prompts run in tiles of 2,048 tokens.

The token split needs a prompt workspace of 4,096 rows on each Spark instead of 2,048. So two Sparks now hold one chat fewer at a 32,768-token context (8 instead of 9). In exchange, both Sparks work together on a cold prompt of 2,048 to 8,192 tokens.

**Three or more Sparks.** A Spark has one GPU and one pool of memory. To use three or more Sparks, Vade usually splits the model by layers, not by matrices. Four Sparks over RDMA run the tensor split in parts (see **Choosing the layout** below). Each Spark runs one engine process, called a *[rank](GLOSSARY.md#pr)*, and the ranks form a chain over direct cables:

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

The layer counts are for a three-rank chain over the 48 layers of the release checkpoint.

**Who does what.** The front rank keeps the HTTP server, the scheduler, the embeddings, the output head, sampling and drafting. Each rank runs one contiguous interval of layers and passes its result to the next. The final residuals come back to the front over the same full-duplex cables ([`next_pp.rs`](../engine/crates/engine/src/flash/next_pp.rs), [`next_service.rs`](../engine/crates/tp/src/flash/next_service.rs)).

The intervals depend only on the layer and rank counts. The layers are divided evenly. The front always keeps at least two layers, because layer 1 consumes the per-layer embeddings. Each stage needs at least two layers, so a chain can be as long as half the layer count. That is 24 Sparks for this checkpoint.

**The same arithmetic as one Spark.** Every stage runs the one-Spark kernels in the same layer order. It passes FP32 values across each boundary unchanged. So the pipeline does the same arithmetic as a single Spark ([`next_pp.rs`](../engine/crates/engine/src/flash/next_pp.rs)). Drafting stays available, because verification rows cross the pipeline like any other rows.

**Choosing the layout.** The engine chooses the path itself. No switch selects it ([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)).

- **FP4-Local on two Sparks.** It runs the tensor split.
- **FP4-Local on more Sparks over RDMA.** It runs the split in parts when two checks pass. Every split width must cut into whole parts (`TpLayout::for_config` in [`tp_switch.rs`](../engine/crates/engine/src/flash/tp_switch.rs)). The mailbox must have room for that many ranks (`split_accepts` in `settings.rs`). For this checkpoint, that is four Sparks. Eight Sparks pass the width check but not the mailbox check.
- **All other counts.** They run the N-stage chain.
- **`--tensor-map` on four Sparks.** It runs the older rank path.
- **FP8.** FP8 runs tensor parallel: on a ring on two Sparks and on a path on four.

The serving receipt names the path and the reason for it (`vade.numerics.path` and `path_reason`).

**Transport.** Decode steps and short prompt tiles travel on a polled RDMA (RoCE) wire. Queued prompt tiles stay on TCP. A TCP-only chain of three or more Sparks is also supported ([`next_service.rs`](../engine/crates/tp/src/flash/next_service.rs), [FP4-Local launch](../cli/docs/fp4-local.md)).

The recipe of the pipeline covers drafting across stages, queued prompt tiles, the RDMA wire, 2048-row prompt tiles and a bounded output head. It is pinned at startup on every rank, so all of them read one setting ([`settings.rs`](../engine/crates/tp/src/flash/settings.rs)).

**Agreeing before serving.** Before the first request, the ranks verify that they run the same PTX and checkpoint artifacts. Every worker reserves memory for the number of chats that the front will admit ([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).

**What the measured layer pipeline provided.** These numbers are from the 2026-10-25 cards, when two Sparks still ran the layer pipeline. Today two Sparks run the tensor split.

- **First token.** In those cards, the first token after a cold 8K prompt came in 2.80 s on two Sparks and 2.19 s on four. The v0.1.0 card on one Spark gave 3.86 s.
- **One chat.** Only one pipeline GPU works on a given chat at a time. With drafting on, code ran at 79.3 tok/s on two Sparks and 80.3 on four, next to 76.7 on one.

A chain is mainly for models too large for a single machine ([reading these cards](history/benchmarks-2026-10.md#reading-these-cards)).

## The prefix cache

The prefix cache lets a new request skip the start of its prompt when an earlier request began the same way. It is **on by default** on one Spark, on the two-Spark tensor split and on FP8. FP4-Local on three or more Sparks (the layer chain, or the split in parts on four Sparks over RDMA) runs without it.

Requests that send `cache:true` or a `prompt_cache_key` keep their prompt for reuse. `VADE_FLASHNEXT_PREFIX_CACHE=0` turns the cache off ([`prefix_switch.rs`](../engine/crates/serve/src/prefix_switch.rs)).

**The room of the cache.** At startup, the cache takes a room of its own from the session pool. The room is half of the bytes that no full-context chat can use, from 512 MiB to 4 GiB, and never more than those bytes. On the two-Spark tensor split, each rank keeps a copy, and the room is at least one full-context chat (up to 4 GiB).

- **Room.** Admission never uses the room, and host memory pressure does not reclaim it. So a snapshot fits even after a burst of chats fills the spare buffers.
- **Beyond the room.** The cache uses pool bytes that live chats leave free, and it gives them back first.
- **Under host memory pressure.** The cache gives those bytes back 30 seconds after the last live request. Before then, the funded reserves keep what an eviction returns, so an eviction frees no host memory.

A restored prefix holds the same state that a cold prompt computes, so the token IDs do not change ([`prefix_switch.rs`](../engine/crates/serve/src/prefix_switch.rs)).

**Checkpoints in a hybrid model.** A hybrid model makes the cache harder than usual. In a pure attention model, a cached prefix is just the first rows of the key/value cache. But the recurrent state of a GDN layer summarizes the whole prompt so far. Vade cannot cut it back to an earlier position.

Vade solves this with checkpoints at aligned positions. Prompt tiles always start on absolute 64-token boundaries. So the recurrent state after the last whole tile of a prompt is exactly the state that a cold read of any prompt with that prefix would hold at the same position. At that position, the cache saves the GDN states, the PLE history and the last hidden row of the draft head. It copies the attention cache rows below that position. A later request resumes the prompt read at the checkpoint ([`next_prefix.rs`](../engine/crates/engine/src/flash/next_prefix.rs)). No kernel, tile or arithmetic order changes, so a cached answer equals a fresh one.

**Elastic retention.** By default, cached prefixes can use up to half the session pool. Admission reclaims them before it refuses a live request ([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).

**The cache on two Sparks.** The cache works locally and on the two-Spark tensor split.

- **Saved states.** Each split rank saves its own state. A shared prefix ID ties the saved states together.
- **Restore.** The existing ordered control link completes the worker restore before the next prompt rows run.
- **Memory.** The cache uses the larger rank charge and the smaller available memory bound. A snapshot that does not fit that bound is not kept, and the next turn reads its prompt again.

For example, on two Sparks at `--context 131072`, a 100K-token prompt needs a 6.2 GB front snapshot. But the live request leaves 5.3 to 6.0 GB, so its second turn reads the whole prompt again. At `--context 110592`, the same snapshot fits.

The layer chain and the split in parts on four Sparks still refuse the cache at startup ([`prefix_switch.rs`](../engine/crates/serve/src/prefix_switch.rs)).

## Ranks, memory and ownership

On a Spark, the CPU and GPU share one pool of unified memory. At startup, the engine measures what is available. It keeps back a safety reserve of a tenth of it (at least 2 GiB) and loads one copy of the weights. Then it plans a session pool from what is left. If one full-context chat does not fit, the engine refuses to start and says why ([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).

The engine keeps five kinds of memory apart:

- **Resident weights.** Each rank holds one copy of its share of the model, and all its chats share it. The weights stay until the engine shuts down.
- **Active chat state.** Each chat has its own attention cache, recurrent state and execution buffers. They stay until the chat ends.
- **Retained prompt state.** The prefix cache keeps its checkpoints here. They stay until the engine reclaims them under the budget of the pool.
- **Shared scratch.** Every model step reuses the same working space and CUDA graphs. The engine reserves them at startup.
- **Host staging and transport.** The engine keeps buffers for loading and for network transfers. It keeps them for reuse.

*Memory admission* is the check that a request fits. It charges each request for the tokens that it can grow to, and it reclaims idle cache first ([`dispatch.rs`](../engine/crates/serve/src/scheduler/dispatch.rs)).

Freed chat buffers wait in a reserve for the next request. They do not go back to the driver, because an allocation of device memory is slow enough to leave the GPU idle between requests ([`next_batched_service.rs`](../engine/crates/tp/src/flash/next_batched_service.rs)).

On several Sparks, every rank funds the same admitted chats. So a request that the front accepts also fits downstream. Cached tokens still count toward the context length. Reuse saves work, but it does not raise the context limit.

The engine tracks the device bytes that it allocates ([`flash-serve.rs`](../engine/crates/tp/src/bin/flash-serve.rs)). That figure is good for a comparison of one build with another. But it leaves out some driver allocations, and it is not the total physical memory use of the Spark. Each benchmark [receipt](GLOSSARY.md#pr) states which memory measure it reports.

## Exactness and model quality

Vade's exactness rule is this: **the same setup and request produce identical token IDs on every Spark, every run**. The setup includes the build, checkpoint, tokenizer and template, rank layout, thinking and the sampling settings in effect.

Within one setup, the answer must be byte-identical in each of these cases:

- **Alone or batched.** The request runs alone or in a batch with other chats.
- **Cached or fresh.** The prefix comes from the cache or is read fresh.
- **Chunked.** The prompt is split into chunks in any way.
- **Drafting.** Under greedy decoding, drafting is on or off.

Request IDs and timestamps in HTTP envelopes are not part of the answer. The mechanisms behind the rule are the ones above: fixed reduction trees, row kernels, aligned tiles and checkpoints, and verification that picks every token.

`vade verify` checks the rule on your own machine. It sends the same greedy requests several times with drafting off and on, and it compares every token ID. It fails if a drafting-on run did not draft. `--machine` adds an independent engine on another machine to the comparison ([`verify/mod.rs`](../cli/src/verify/mod.rs)). It is a targeted check. It does not prove every mix of batching, cache and chunks. The [cohort tests](../engine/crates/serve/tests/cohort_mtp.rs) and the device gate in [AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate) cover more. Any change to engine arithmetic, scheduling, caching or transport must match across at least three runs on at least two Sparks before it merges.

Exactness is a different question from quality. Three checks answer three questions:

- **Repeatability.** Does this build keep the same-setup rule on every execution path?
- **Previous-build preservation.** Do the saved test cases still match an older Vade build? A new build is a different setup. A different supported Spark count is not: it must give the same token IDs. So its answers may change, but only if the quality checks pass.
- **Model quality.** Does a change keep task accuracy, instruction following and long-context retrieval? A match with an older Vade build is a separate question from a match with an outside reference runtime.

An intentional change to the arithmetic order needs three things before it becomes the default: the full exactness gate, reviewed golden updates, and quality evidence against main. Record the change and its results in a benchmark card. Keep one production path. A runtime switch does not replace this evidence.

Quality checks include layer differences against the measured numerical noise floor, GSM8K and IFEval comparisons, and long-memory checks for large changes ([CLI validation](../cli/docs/test.md), [evals](../cli/evals.md)).

## What is measured, and how

Vade measures at four levels and keeps their claims apart. Every published number is tied to a **receipt**: the saved record of one run, with its build, hardware, checkpoint, settings and raw results.

- **Kernel: `kernel-bench`.** It shows the time per kernel against the as-stored floor, with the baseline and the candidate in one process. It requires identical output bytes. It does not show end-to-end speed or the exactness of the whole model.
- **Server: `vade bench`.** It shows the time to the first token, the tokens per second and the throughput of chats at once, measured from outside over HTTP. It does not show model quality.
- **Card: RigMark.** It gives the published cards: prose, code, structured output, cold 8K prompts, and 1 to 8 chats at once. It does not show exactness. Its output checks only confirm that the answers completed.
- **Exactness: `vade verify`.** It shows identical token IDs across runs and drafting modes. It does not show speed or quality.

`vade bench` counts real tokens from the server's usage figures, never streamed chunks. If any request in a group stops early or fails, the whole group gets no rate ([bench library](../engine/tools/bench/README.md), [run vade bench](benchmarks.md#run-vade-bench)). The published cards come from RigMark ([run RigMark](benchmarks.md#run-rigmark-against-vade)), and each card links its sanitized receipt folder.

These are a few v0.1.0 results, all from [Benchmarks](history/benchmarks-2026-10.md#previous-2026-10-24-v010-release-5a22651). They ran on one of our Sparks with drafting on, unless a line says otherwise.

<div align="center">

| Measure | Result |
| :---: | :---: |
| One chat, code | 76.7 tok/s (27.8 with drafting off) |
| One chat, prose | 41.7 tok/s (27.8 with drafting off) |
| Eight chats together | 173.9 tok/s |
| First token after a cold 8K prompt | 3.86 s; 2.79 s on two Sparks |

</div>

The benchmark page explains the recipe, the ranges from run to run, the cloud VMs and the limits of each figure. See [client time and server time](benchmarks.md#client-time-and-server-time) and [what these results do not show](benchmarks.md#what-these-results-do-not-show).

## The harness and the CLI

### Native agent mode

The harness's native agent works inside a workspace that you choose. Reads, searches and exact patches stay within it. With the default `--approval ask`, edits wait for your approval. So does every command that no allow-list rule matches (see [shell safety](shell-safety.md)). Blocked commands never run.

The harness writes a tool's result to a private journal. It sends the result back to the engine as part of the next request. The engine never runs a tool itself. An engine can advertise the optional `vade.engine.v1` contract. With it, the harness can ask whether a request will fit before it sends the request ([engine contract](../harness/docs/engine-contract.md)).

Shell commands run only on macOS for now, inside `sandbox-exec` with a cleared environment and no network. On Linux, including the Spark itself, the agent can read and patch files but does not run commands. To let it run commands, start the harness on a Mac and point it at the Spark ([harness README](../harness/README.md#native-coding-agent)).

### Gateway mode

The gateway relays the requests of another client to one or more engines, as described in [Follow one request](#follow-one-request). It speaks plain HTTP with no built-in TLS. So between machines, use the [SSH tunnel](../engine/docs/flash-server.md#connect-your-mac) that the controller route sets up. A cleanly closed connection does not prove that the remote GPU has stopped work on that request ([gateway contract](../harness/docs/engine-contract.md#opaque-gateway)).

### What the CLI starts and stops

`vade launch` runs on the Spark. It finds an existing checkpoint, builds or uses local artifacts, starts the engine and opens chat.

From a separate [controller](GLOSSARY.md#ac), `vade install` stages a build. Then `vade up` starts the selected ranks over SSH, plus a tunnel that it owns ([`launch.rs`](../cli/src/launch.rs), [INSTALL.md](../INSTALL.md)).

`vade status` and `vade logs` inspect the installation. `vade down` stops only the processes that the CLI recorded, after it checks the identity of each one.

## Current limits

This page describes main after v0.1.0, the latest release. Its default is FP4-Local on one Spark. The benchmark page also has results from cabled pairs, from a group of four Sparks and from rented cloud VMs.

- **FP4-Local on four Sparks.** FP4-Local gave the same token IDs on four Sparks as on one and two Sparks in 66 of 66 responses. That held on the layer chain in the 2026-10-28 cards at `76977df` and the 2026-10-30 morning cards at `380d8e5`. It held on the split in parts over RDMA in the 2026-10-31 morning cards at `dc32b7c` ([results](results.md#token-ids)).
- **FP8 on two and four Sparks.** FP8 on two and on four Sparks gave the same token IDs in only 7 of 66 responses in those same cards ([benchmarks](history/benchmarks-2026-10.md#fp8-on-two-and-four-sparks-different-token-ids)). A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order, so four Sparks now give the same token IDs as two. The four-Spark cards of 2026-11-01 show it ([FP8 token IDs](results.md#fp8-token-ids)). Six Sparks still give other IDs ([changelog](../CHANGELOG.md)). The cards above predate that change, so compare their FP8 results only at the same Spark count.
- **Long chains.** Chains longer than four Sparks have no live exactness check yet ([FP4-Local launch](../cli/docs/fp4-local.md)).
- **Other code.** The other model parsers, servers and protocol routes in the source are not qualified to the same level.
- **The prefix cache.** The prefix cache is on by default. It works locally, on the two-Spark tensor split and on FP8. FP4-Local on three or more Sparks (the layer chain, or the split in parts on four Sparks over RDMA) runs without it.
- **Greedy requests.** On FP4-Local, a request with no sampling field is greedy. `"temperature": 0` alone is greedy too, because the checkpoint's own penalties are neutral. On FP8, a greedy request needs `"temperature": 0` and the three neutral penalties ([OpenAI API](openai-api.md)).
- **Cancel.** The scheduler handles a cancel between model steps, so the cancel waits for the GPU work that is already running ([`dispatch.rs`](../engine/crates/serve/src/scheduler/dispatch.rs)).
- **The shell tool.** The agent's shell tool needs macOS.
- **What speed numbers do not cover.** They do not cover cache accounting, isolation of one failing request from the rest of its batch, long sessions or complete coding workflows. Each of these needs its own test.

Continue with these pages:

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [Benchmarks](benchmarks.md) | The measured results and how to repeat them |
| [Engine package map](../engine/crates/README.md) | Every engine package |
| [Harness docs](../harness/docs/README.md) | Every harness guide |
| [Docs index](README.md) | Every other page |

</div>
