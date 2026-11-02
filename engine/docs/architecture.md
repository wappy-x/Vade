# Architecture

**Vade is one Rust program: it reads official model files, runs them on the
Spark's GPU and answers chat requests over HTTP.** It runs on top of NVIDIA's
DGX OS and GPU driver.

This page follows one request from the HTTP socket to the GPU and back. It is
for developers and agents changing the code, and for anyone curious;
[Design principles](concepts.md) explains *why* the engine works this way.

The server on this page, `spark-serve`, serves Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B,
Qwen3.8-27B (BF16 and official FP8) and Gemma 4 31B. [Models](models.md) lists
them. The v0.1.0 release model, Qwen3.8-Flash-Next-FP4-Local, runs in a
separate server, `flash-serve`, which `vade launch` and `vade up` start on one
Spark or across several; see [FP4-Local](../../cli/docs/fp4-local.md).

<details>
<summary><b>Terms on this page</b>: token, tokenizer, prefill, decode, KV cache, BF16, kernel, tensor cores, PTX, unified and pinned memory, rank, RDMA</summary>

<dl>
<dt>Token</dt>
<dd>The unit a model reads and writes, often part of a word.</dd>
<dt>Tokenizer</dt>
<dd>Maps text to tokens and back.</dd>
<dt>Prefill</dt>
<dd>Reading the prompt: all its tokens go through the model together.</dd>
<dt>Decode</dt>
<dd>Writing the answer, one token per step.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>BF16, FP8, FP32</dt>
<dd>16-bit, 8-bit and 32-bit number formats.</dd>
<dt>Kernel</dt>
<dd>A small program that runs on the GPU.</dd>
<dt>Tensor cores</dt>
<dd>The GPU's matrix-multiply units.</dd>
<dt>PTX</dt>
<dd>NVIDIA's portable GPU instruction format.</dd>
<dt>Unified memory</dt>
<dd>The Spark's CPU and GPU share one pool of memory.</dd>
<dt>Pinned memory</dt>
<dd>CPU-side memory the GPU or network card can copy from directly.</dd>
<dt>Rank</dt>
<dd>One Spark's process in a multi-Spark group, numbered from 0.</dd>
<dt>RDMA</dt>
<dd>The network card copies memory straight between machines.</dd>
</dl>

- Prefill is heavy on arithmetic. Each decode step reads every weight once,
  so memory bandwidth limits it.
- The KV cache holds the attention keys and values of every token already
  read.
- Makers publish weights in BF16 or FP8; FP32 holds running sums.
- The driver turns PTX into GB10 machine code when the server starts.
- RDMA skips the usual network stack.

Sizes: MiB means 1,048,576 bytes. GB means 1,000,000,000 bytes.

</details>

## The promise

**A prompt gets exactly the same reply, alone or in a batch, with its
beginning from the cache or read fresh.**

Agreement with the official model is a separate, measured claim, and it is
not perfect. [Validation](validation.md#promise-2-agreement-with-the-official-model)
has the counts for every model and build, and every disagreement.

## Design choices

**The server, tokenizer, networking and GPU kernels are all Vade's own Rust,
on top of NVIDIA's driver.**

<dl>
<dt>Rust only</dt>
<dd>Only Rust's <code>std</code>, and <code>core</code> on the GPU; no outside
crates.</dd>
<dt>Own GPU kernels</dt>
<dd>Built by Rust's compiler, plus short inline PTX blocks.</dd>
<dt>No NVIDIA math libraries</dt>
<dd>No cuBLAS, CUTLASS or other NVIDIA math library runs.</dd>
<dt>Driver only</dt>
<dd>Vade calls NVIDIA's driver and the network card's library.</dd>
<dt>Official weights</dt>
<dd>Makers' releases at their precision, never quantized by Vade.</dd>
<dt>One copy of the weights</dt>
<dd>Each Spark holds them once, shared by every chat.</dd>
<dt>One arithmetic order</dt>
<dd>Same order alone or batched, cached or fresh.</dd>
<dt>Exact by default</dt>
<dd>Answer-changing settings need a flag or request to turn on.</dd>
</dl>

- Inline PTX covers the few instructions Rust's compiler cannot produce.
  NVIDIA's documents and CUTLASS are read as reference only.
- No CUDA toolkit, and no Python in the engine.
- In shard-only multi-Spark mode, each number is still added up in the same
  order as on one Spark.

## The path of a request

**A request passes through four parts, and each new token streams back as
soon as it exists.**

```text
client --HTTP--> serve --> tokenizer --> engine <--> gpu (GB10)
   ^                                        |
   +------- text, streamed token by token --+
```

1. **serve** accepts the request, checks its limits, queues it by priority,
   reserves memory and looks for a cached prompt beginning.
2. **tokenizer** applies the model's chat template and turns text into token
   IDs.
3. **engine** reads the prompt (prefill), then writes the answer one token
   per step (decode), batched with every other ready chat.
4. **gpu** owns GPU memory and launches the kernels through NVIDIA's driver.
5. Each new token goes back through the tokenizer to text, and serve streams
   it out straight away.

The weights load once, at start. Every request uses that one copy.

<details>
<summary><b>The parts</b>: which folder does what</summary>

<dl>
<dt><code>crates/serve</code></dt>
<dd>HTTP, the scheduler, sampling, the prompt cache and the
<code>spark-serve</code> program.</dd>
<dt><code>crates/tokenizer</code></dt>
<dd>Chat templates, and text to tokens and back.</dd>
<dt><code>crates/model</code></dt>
<dd>Reads and checks the config, and each weight's name, shape and file
position.</dd>
<dt><code>crates/engine</code></dt>
<dd>Runs the model's layers; owns each chat's KV cache and scratch memory.</dd>
<dt><code>crates/gpu</code></dt>
<dd>Talks to NVIDIA's driver, owns GPU memory and launches kernels.</dd>
<dt><code>crates/gpu/kernels</code></dt>
<dd>Matrix multiplies, attention, expert routing and the rest of each
layer.</dd>
<dt><code>crates/tp</code></dt>
<dd>Splitting a model across Sparks, and the launcher.</dd>
<dt><code>crates/seqpar</code></dt>
<dd>Splitting a long prompt across Sparks, peer admission and sequence transport.</dd>
<dt><a href="../crates/fabric/README.md"><code>crates/fabric</code></a></dt>
<dd>Shared peer discovery, RDMA verbs, link capabilities, exchange API and profiling.</dd>
<dt><code>tools/bench</code></dt>
<dd>The HTTP benchmark client.</dd>
<dt><code>tools/export</code></dt>
<dd>Copies an allowlisted snapshot for the separate shareable repository.</dd>
<dt>Privately retained reference oracle</dt>
<dd>Generates official reference data with Transformers outside this repository.</dd>
</dl>

- `crates/engine` also holds the Flash-Next engine, in `src/flash/`. Its
  server, `flash-serve`, lives in `crates/tp`.
- `tools/export` is described on [Export](export.md).
- The official reference oracle uses Hugging Face Transformers and is kept
  privately outside this repository. [Reference provenance](../tools/truth/README.md)
  describes the saved fixtures; the oracle never runs in the engine.

</details>

## GPU kernels and the driver

**The kernels are ordinary Rust, compiled to PTX and carried inside the
server's own program file.** Vade talks to NVIDIA's driver directly, as an
ordinary user.

<details>
<summary><b>GPU kernels in Rust</b>: inline PTX, tensor cores and one attention order</summary>

- Rust's NVPTX target compiles them to PTX.
- The server carries that PTX inside its own program file, so nothing needs
  to be installed next to it.
- The driver turns the PTX into GB10 machine code when the server starts.

A few instructions have no Rust equivalent: the tensor-core multiply,
asynchronous copies into on-chip shared memory and some number-format
conversions. For those, the kernels contain short blocks of inline PTX.

Tensor cores do the heavy lifting whenever there are many rows of work at
once: a long prompt, or several chats in one step. Prefill and decode read
the same packed copy of the weights.

Every BF16 decode path, alone, batched, checking guesses or reading a shared
cached prefix, runs attention through one function in one fixed order. The
[GPU crate](../crates/gpu/README.md#decode-attention-order) spells that order
out.

</details>

<details>
<summary><b>Talking to the driver directly</b>: <code>libcuda</code> and <code>libibverbs</code></summary>

- **`libcuda`**, NVIDIA's driver library: GPU memory, loading kernels and
  running them.
- **`libibverbs`**, the standard Linux library for RDMA network cards:
  transfers between Sparks. It is opened only when an RDMA link starts, so a
  single Spark never needs it.

Vade runs as an ordinary user. It uses network links that are already set up
and makes no network or system changes. See
[Multiple Sparks](multi-spark.md#cabling).

</details>

## Memory

**Weights, every chat's KV cache, scratch space and the operating system all
come from the Spark's one unified pool.** CPU and GPU memory reports can
count the same bytes, so adding them together overstates use.

<details>
<summary><b>Weights, conversations, budgets and unloading</b></summary>

**Weights**

- Reader threads read the weight files through four 8 MiB pinned buffers.
- Each piece goes straight to its final place in GPU memory. No second copy
  stays on the CPU side.
- On the way in, the weights are packed into the order the kernels read
  fastest. That changes the order of the numbers, not their values.

<dl>
<dt>Qwen3-8B and Qwen3-32B</dt>
<dd>One per checkpoint file; one copy of the weights.</dd>
<dt>Qwen3-30B-A3B</dt>
<dd>One block for the whole model.</dd>
<dt>Qwen3.8-27B, BF16 or FP8</dt>
<dd>One resident checkpoint copy; FP8 stays FP8.</dd>
<dt>Gemma 4 31B</dt>
<dd>One text-weight copy.</dd>
<dt>Qwen3.8-Flash-Next-FP8</dt>
<dd>Runs in the separate Flash server, not in this one.</dd>
</dl>

Gemma 4 31B's optional assistant is a separate official model.

**Conversations**

- Small KV caches use compact allocations.
- Larger ones grow by adding 2 MiB physical memory pages behind a fixed
  address, so the old cache is never copied.
- Chats that share a beginning share those pages. A chat copies a page only
  before it writes to a shared one.

**Budgets.** The server plans its limits from the model and the free memory
at start, then enforces them with its own allocator. It keeps separate
budgets for active chats, saved prompts and shared workspace, and reports
each in bytes on its `ready` line. [Server and API](server.md#how-it-sizes-itself)
has the details.

**Unloading.** Stopping the server, or swapping the model, frees every byte
the model held. Before loading another model, the server checks that its
allocated bytes are back to zero.

</details>

## Precision follows the model

**Weights stay in the format the maker published, BF16 or FP8.**

- **BF16 models** multiply in BF16 on the tensor cores and accumulate in FP32.
- **FP8 models** keep their FP8 weights and scales resident. Flash's routed
  experts use W8A8; covered BF16 projections use its fixed split mode.
- **The KV cache** is BF16 on one Spark.

The per-model table, with bytes, is on [Models](models.md#memory-and-context).

<details>
<summary><b>FP8 scales, Flash tensor-core arithmetic, and opt-in changes</b></summary>

- **FP8 scales.** FP8 checkpoints come with scales: one multiplier per block
  of weights. Vade applies them as each weight is used, so it never holds a
  BF16 copy of an FP8 model.
- **One Flash projection mode.** Covered BF16 projections carry each FP32
  activation as BF16 high and residual parts. Tensor-core products accumulate
  in FP32 with a fixed reduction order for single, batched, cached, chunked
  and greedy-MTP paths. There is no scalar-FP32 alternative. Other models'
  native BF16 paths keep their existing arithmetic.
- **Repeatability scope.** The same machine, build and setup must give the
  same bytes. Flash answers may differ from older builds. Each supported
  Spark count must give the same token IDs (FP8 on six Sparks does not yet);
  new arithmetic gets new goldens and separate quality evidence.
- **Opt-in changes.** Settings that can change the output, such as `fast`
  drafting and YaRN context extension, stay off unless they are turned on.

</details>

## How a change is judged correct

**Two checks: the same answer on every path, and agreement with the official
model.**

1. **Same answer, every path.** Batched output is identical to one-at-a-time
   output, including on long prompts the reference cannot reach. Cached
   and fresh starts match too.
2. **Agreement with the official model.** Greedy tokens and logits (the raw
   score for every possible next token) are compared with Hugging Face
   Transformers at the checkpoint's own precision. For Qwen3-30B-A3B, the
   older fixtures also compare which experts each token picks.

Older FP32 reference files remain for debugging only.
[Validation](validation.md) has the method, the results and the commands.

## Scheduling

**One scheduler owns the GPU.** At each step it batches every chat that is
ready to decode, so the weights are read once for all of them.

<details>
<summary><b>Pauses, priority, expert models, checked guessing and prompt caching</b></summary>

- **Pauses.** A long prefill pauses between chunks or groups of layers, so
  other chats can take a step. A GPU kernel that is already running cannot be
  interrupted.
- **Priority.** Higher-priority requests go first at those pauses. Ordinary
  clients default to priority 0. With an owner key and at least three active
  slots, one slot and memory for a short request are reserved for that owner.
- **Expert models.** Dense models share one weight read across every chat in
  a decode step. On the expert model, a batched step reads each selected
  expert once for all the chats routed to it. Different prompts usually pick
  more experts than the same prompt repeated, so a benchmark that repeats one
  prompt overstates expert-model throughput.
- **Checked guessing.** Speculative decoding runs inside the same step:
  guessed tokens are checked in one pass, and only tokens the model would have
  written itself are kept. [Speculative decoding](speculative-decoding.md) explains it.
- **Prompt caching.** Caching runs before prefill. When a request starts with
  the same tokens as a recent request, or as a chat that is still running,
  Vade reuses that cached beginning and reads only the rest. A request that
  fails discards its cache.

</details>

## Several Sparks on one answer

**Tensor parallelism splits every layer across Sparks, and every rank ends
up with the whole result.** Each rank keeps its share of the weights and the
KV cache for its own attention heads, and the ranks exchange pieces.

[Multiple Sparks](multi-spark.md) has the setup steps and the tested scope.

<details>
<summary><b>The picture, and how it keeps one Spark's result</b></summary>

```text
                     one request, one answer
                               |
                +--------------+--------------+
                |                             |
                v                             v
   +------------------------+    +------------------------+
   | rank 0                 |    | rank 1                 |
   | its share of weights   |    | its share of weights   |
   | and cache              |    | and cache              |
   +------------------------+    +------------------------+
                |                             |
                +<------- RDMA, cable ------->+
                |                             |
                +---- pieces exchanged, so ---+
                  every rank has the whole result
                               |
                               v
                         the next token
```

- **Same result as one Spark.** In every layer the ranks exchange their
  pieces, so every rank ends up with the whole result. Each number is still
  computed in full on one rank, adding its terms in the same order as a
  single Spark. The recorded check: Qwen3-8B on two ranks produced the same
  tokens as one Spark on the retained short, medium and long checks.
- **Transport.** The exchange uses RDMA. On the GB10 the network card cannot
  reach GPU memory, so data passes through pinned memory on the CPU side. A
  cable has two rails (network interfaces), and Vade uses both when both have
  addresses.
- **Trust.** Before work starts, the ranks check that they run the same
  program with the same settings and layout. They prove to each other that
  they belong together with a shared secret.
- **Layout.** The launcher arranges Sparks in a line, or a ring when the last
  one cables back to the first. Odd and even counts both work.

Splitting a long prompt works differently. Each Spark holds the full weights
and reads one slice of the prompt, passing its cache along.

The last Spark ends up with the whole cache. It either writes the answer
itself, or hands the cache out for split decoding (hybrid mode with prompt
splitting switched on). This path still hands the cache off in FP32.

</details>

## Reused data

**Vade includes no third-party code.** A few files are reused as data, with
their licenses kept; [Provenance](provenance.md) gives each file's source,
revision, hash and notice.

<details>
<summary><b>The reused files</b></summary>

- Qwen3-8B's config and weight index, used to test the loader.
- Qwen3's and Gemma 4's official chat templates. Vade's own Rust formatter
  follows them; no template engine runs.
- Unicode character tables, generated by a Rust tool from the official
  Unicode data files.
- Reference prompts, tokens and scores in `fixtures/`, made with the makers'
  official code. See [Validation](validation.md).

</details>

## Not done yet

**Five pieces of the design are not finished on main.**

- **One weight allocation for Qwen3-8B and Qwen3-32B.** They still use one per
  checkpoint file.
- **A BF16 cache everywhere.** Prompt splitting, including hybrid mode with
  it switched on, keeps its cache in FP32.
- **Full official-model agreement.** The CUDA suites, up to 28,672 input
  tokens, record disagreements on all three Qwen3 models; see
  [Validation](validation.md#qwen3-cuda-acceptance-suite).
- **Larger groups.** Each small exchange between Sparks still goes through
  the CPU side, which limits how much more Sparks help.
- **Flash-Next in this server.** Flash-Next models run in their own server,
  `flash-serve`, on one Spark or across several.

**Where next:** [Design principles](concepts.md) ·
[Speculative decoding](speculative-decoding.md) · [Credits](CREDITS.md)
