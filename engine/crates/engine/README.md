# spark-engine

**The engine runs language models on the Spark's GPU. It loads the weights,
keeps each conversation's memory, and turns tokens into next-token scores.**

This page is for people reading or changing the engine. To just run Vade,
start with the [one-Spark quickstart](../../../docs/quickstart-one-spark.md).

<details>
<summary><b>Terms on this page</b>: KV cache, prefill, decode, logits, session, BF16, FP8, FP32, PTX</summary>

<dl>
<dt>KV cache</dt>
<dd>The model's memory of the tokens it has already read, so a new token does
not reread the whole conversation.</dd>
<dt>Prefill</dt>
<dd>Reading the prompt, before the first answer token.</dd>
<dt>Decode</dt>
<dd>Producing the answer, one token per step.</dd>
<dt>Logits</dt>
<dd>The model's raw score for every possible next token.</dd>
<dt>Session</dt>
<dd>One conversation: its position, its KV cache and its working memory.</dd>
<dt>BF16, FP8, FP32</dt>
<dd>16-bit, 8-bit and 32-bit number formats.</dd>
<dt>PTX</dt>
<dd>NVIDIA's portable GPU instruction format, written as text. The driver
compiles it for the GB10 at start.</dd>
</dl>

</details>

## Where it fits

**Models and sessions live here. Model files come in from engine-model, GPU
work goes out through spark-gpu-probe.**

<dl>
<dt><a href="../model/README.md">engine-model</a></dt>
<dd>Reads and checks model files. Finds every weight for the engine.</dd>
<dt><a href="../tokenizer/README.md">engine-tokenizer</a></dt>
<dd>Text to token IDs and back. A dependency: the engine works on token IDs;
text handling happens in the server.</dd>
<dt><a href="../gpu/README.md">spark-gpu-probe</a></dt>
<dd>GPU memory, weight loading, kernels. All GPU work goes through it.</dd>
<dt><a href="../serve/README.md">spark-serve</a></dt>
<dd>The chat server. Built on this crate.</dd>
<dt><a href="../tp/README.md">spark-tp</a>,
<a href="../seqpar/README.md">spark-seqpar</a></dt>
<dd>Work across several Sparks. Built on this crate.</dd>
</dl>

## The promise this code keeps

**Same answer every time, checked against the official model.**

- **Same answer every time** is the contract. The October 14 Qwen3-8B chat
  restart checks passed, but the current batched prefill check has a **known
  issue, under investigation**; see the qualification below.
- **Checked against the official model.** That agreement is measured and is
  not perfect. [Validation](../../docs/validation.md) lists every
  disagreement.

<details>
<summary><b>The measured agreement, in numbers</b></summary>

On the broader Qwen3 suites, on current main `603f95b`, Vade matched the
official reference at 1,523 of 1,536 positions for Qwen3-8B, 1,515 of 1,536
for Qwen3-32B and 1,498 of 1,536 for Qwen3-30B-A3B, in one session per model.

The earlier build `80e616c`, before the shared-prefix change (shared9),
matched 1,522, 1,517 and 1,502. Validation also covers the narrower Qwen3.8
and Gemma checks.

</details>

## Build and run the checks

**Six commands check Qwen3-8B: agreement with the official model, batching,
caching, prefill, timing and clean unloads.** You need an idle Spark with Vade [installed](../../../docs/quickstart-one-spark.md)
and Qwen3-8B downloaded. Each program prints its usage when run without
arguments.

### Do it yourself

1. [Stop `spark-serve`](../../docs/server.md#stopping) so the GPU is free.

2. Build the kernels and the engine programs.

   ```sh
   # Run from the monorepo root, then enter the engine workspace.
   cd engine
   cargo kernels
   cargo +1.99.0 build --release -p spark-engine
   ```

3. Set two shortcuts for the commands below.

   ```sh
   PTX=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
   MODEL=models/Qwen/Qwen3-8B
   ```

4. Run the check you need. Each comment says what the command proves.

   ```sh
   # Agreement with Hugging Face Transformers.
   ./target/release/hf-batch-check "$PTX" "$MODEL" fixtures/qwen3/short.json \
     fixtures/qwen3/medium.json fixtures/qwen3/long.json --bf16-kv

   # Batched verification matches one-at-a-time decode.
   ./target/release/batch-bench "$PTX" "$MODEL" 2048 0 verify

   # Cache fork, truncate and replay give the same logits.
   ./target/release/batch-bench "$PTX" "$MODEL" cache

   # Known issue, under investigation: current batched-last-row check fails.
   # Run separately; stop and report its failure without relaxing tolerances.
   # Sequential advance/step matches step-every-token, bit for bit.
   # Prompt All/Last/None heads preserve final/next logit bits within each phase.
   ./target/release/batch-bench "$PTX" "$MODEL" prefill

   # Timing for prompts up to 2,048 tokens, 32 answer tokens.
   ./target/release/batch-bench "$PTX" "$MODEL" 2048 32 bench

   # Load, generate 8 tokens and unload, twice. Tokens must match and
   # GPU memory must return to zero.
   ./target/release/load-check "$PTX" "$MODEL" 2112 2 8
   ```

**Current qualification:** at `e335236a11d681c473de0dce92dece1562d6efa0`,
the one-GB10 loader, batched verification and cache checks passed, but prefill
failed with `FAIL batch last row: logit 7 is nonfinite or differs`. Follow-up
found finite values 16.25 versus 16.125; the historical
`7a34e19cbae9cf98db781bfa7344b06568e86f37` passed on the same GB10.
This is a **known issue, under investigation**. The suite stopped at prefill;
later load/unload and candidate resource checks were not run. A repeated chat
answer is a narrower check. Keep this failure in your qualification report.
   The `prefill` check compares every logit bit for the four-token fixture and
   its continuations. Prompt and decode have separate fixed arithmetic orders;
   see the [phase-order contract](../gpu/README.md#decode-attention-order) and
   [session methods](#qwen3-phase-order). `verify_batch` preserves sequential
   decode bits. This fixture does not qualify all lengths, caches or serving modes.

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, stop spark-serve if it is running
(systemctl --user stop vade-serve.service, or the --daemon PID file under
runtime/state/vade). Enter the engine workspace. Build the kernels with `cargo kernels`, then
`cargo +1.99.0 build --release -p spark-engine`. Run hf-batch-check,
batch-bench (verify, cache, prefill) and load-check against
models/Qwen/Qwen3-8B exactly as listed in crates/engine/README.md.
Run checks one at a time and stop at the first failure. Prefill has a known
issue under investigation; do not call it passing. Report each command's
final lines. Do not change code, commit or push.
```

<details>
<summary><b>Reading the results</b></summary>

- The bundled Qwen3-8B fixtures were made in FP32. Against them, one
  long-case position differs, where two tokens score almost the same. It
  still counts as a disagreement. The
  [fixtures page](../../fixtures/qwen3/README.md) explains it.
- [Models](../../docs/models.md) explains how to read the Hugging Face
  comparison.
- [Benchmarks](../../docs/benchmarks.md) has dated speed and memory figures.

</details>

## What it runs

**Three Qwen3 models, Qwen3.8-27B and Gemma 4 31B in `spark-serve`, plus
Qwen3.8-Flash-Next-FP8 in the two-Spark `flash-serve` server.**

| Model | Code | Note |
| --- | --- | --- |
| Qwen3-8B, Qwen3-32B | `src/qwen.rs` | Dense Qwen3 |
| Qwen3-30B-A3B | `src/qwen.rs` | Expert Qwen3 |
| Qwen3.8-27B, BF16 and FP8 | `src/hybrid.rs` | Hybrid |
| Gemma 4 31B | `src/gemma4.rs` | Text only |
| Qwen3.8-Flash-Next-FP8 | `src/flash/` | Two-Spark `flash-serve` |

- **Dense Qwen3** runs entirely on the GPU, in `src/qwen.rs` and helpers.
- **Expert Qwen3** takes the same path, with expert routing: each token uses
  8 of 128 experts per layer.
- **Hybrid Qwen3.8-27B**, BF16 and official FP8, mixes ordinary attention
  with recurrent linear attention.
- **Gemma 4 31B** text, and Google's official assistant draft, live in
  `src/gemma4.rs` and `src/gemma4/`.
- **Qwen3.8-Flash-Next-FP8** serves chat requests on two Sparks through
  `flash-serve`, separately from `spark-serve`. Follow the root
  [INSTALL.md](../../../INSTALL.md).

## How it works

**One copy of the weights, BF16 multiplies with FP32 sums, and one fixed
order of arithmetic, so batched decoding matches decoding alone, bit for
bit.**

<details>
<summary><b>The design in five rules</b>: weights, precision, order, KV cache, sessions</summary>

- **One copy of the weights.** Reader threads pack the weights for the
  kernels, at the checkpoint's precision, and copy them straight to their
  final place in GPU memory. No host copy is kept.
- **BF16 multiplies, FP32 sums,** matching the checkpoint's own precision.
- **One order of arithmetic.** Single, batched and verification passes add up
  in the same fixed order, so batch size never changes an answer.
- **KV cache in BF16 by default.** FP32 and a scaled 16-bit integer format
  are opt-in. The cache grows in 2 MiB pages and is never copied to grow.
- **Sessions.** The model owns the weights. Each session owns its position,
  its working memory and a reference to its KV cache.

</details>

<details>
<summary><b>How a model loads</b></summary>

1. Reader threads read the weight files through four 8 MiB pinned buffers.
   Pinned means host memory the GPU can copy from directly.
2. Each thread packs 64-row by 64-column blocks into the layout the kernels
   use. Projections whose width is not divisible by 64 use eight-row blocks.
3. The packed rows go straight to their final place in GPU memory.

No second copy of the weights is kept on the host. Nothing is repacked on the
GPU afterwards.

The kernels come from `cargo kernels`. The build embeds the layer kernels in
each program; `SPARK_LAYER_PTX` points the build at a different layer kernel
file. The check programs take the main kernel file as their first argument.

</details>

<a id="qwen3-phase-order"></a>
<details>
<summary><b>Qwen3 session methods and prefill chunk sizes</b></summary>

Create models and sessions inside `spark_gpu_probe::driver::with_backend`.
"Greedy" means always picking the token with the highest score.

<dl>
<dt><code>step(token)</code></dt>
<dd>Reads one token and returns the logits for the next one.</dd>
<dt><code>step_greedy(token)</code></dt>
<dd>Same, but returns the chosen token. Ties go to the lowest ID.</dd>
<dt><code>advance(token)</code></dt>
<dd>Adds one token to the KV cache without computing logits.</dd>
<dt><code>prefill(tokens)</code></dt>
<dd>Reads a prompt in chunks and returns the last row of logits.</dd>
<dt><code>prefill_advance(tokens)</code></dt>
<dd>Reads a prompt chunk without computing logits. The server uses it for middle
chunks.</dd>
<dt><code>forward_batch(tokens)</code></dt>
<dd>Returns logits for every row, up to 512 rows.</dd>
<dt><code>verify_batch(tokens)</code></dt>
<dd>Checks up to 512 proposed tokens in one pass, keeping every logit bit of
one-at-a-time decode. Prompt APIs use their own fixed order; see the
<a href="../gpu/README.md#decode-attention-order">phase-order contract</a> and
the <a href="#build-and-run-the-checks">head-skipping check</a>.</dd>
<dt><code>fork()</code></dt>
<dd>Starts a new session that shares the existing KV cache.</dd>
<dt><code>checkpoint()</code>, <code>truncate(position)</code></dt>
<dd>Note the current position, then go back to it later.</dd>
<dt><code>reset()</code></dt>
<dd>Empties the session.</dd>
</dl>

Prefill reads a prompt in chunks. Chunk size in tokens, by prompt length in
tokens:

| Model | Under 6,144 | 6,144 or more |
| --- | ---: | ---: |
| Dense Qwen3 | 512 | 4,096 |
| Expert Qwen3 | 1,024 | Up to 16,384 |

Expert Qwen3 splits a long prompt into equal chunks of at most 16,384
tokens.

</details>

<details>
<summary><b>Why batched and single answers match</b></summary>

Single-row decode, shared decode and verification use the same BF16 multiply
order.

1. The 16-column input tiles are split into four fixed ranges. Boundary `i`
   sits at `floor(i * tile_count / 4)`.
2. Each range adds up its products in FP32.
3. The four partial sums merge as `(range0 + range1) + (range2 + range3)`.

This order does not depend on how many chats share the batch.

Expert models check proposed tokens one at a time, which keeps the same
arithmetic as plain decode.

</details>

<details>
<summary><b>Shared and forked caches</b></summary>

- A forked session shares committed cache pages. Before writing, it copies
  only the unfinished tail. Small caches copy their committed prefix.
- Shared decode can read the original prefix with a compact private tail.
  Ordinary session calls restore a contiguous view when needed.
- Truncate and replay also detach shared storage before writing, so sessions
  keep their new tokens separate.

`cache_bytes` counts every allocation a session can reach. Shared memory is
counted in each session, so adding up several sessions overstates the total.

</details>

<details>
<summary><b>KV cache formats and growth</b></summary>

Pick the format before the session's first token. The default follows the
checkpoint's precision. Cache size per token, on Qwen3-8B:

| Format | Bytes per token |
| --- | ---: |
| BF16, the default | 147,456 |
| FP32 | 294,912 |
| Scaled 16-bit integer | 149,760 |

- FP32: `session.configure_kv_bf16(false)`.
- Scaled 16-bit integer: `session.configure_kv_format(KvFormat::ScaledI16)`.
- At 2,048 tokens, BF16 uses 288 MiB and FP32 uses 576 MiB.
- The scaled integer format stores eight FP32 scales for each group of eight
  heads.
- Sequence-parallel prefill requires FP32.

The cache grows in 2 MiB pages through CUDA virtual memory. New pages are
mapped behind the existing ones, so the old cache is never copied. Small
caches use ordinary allocations until they reach one page.

</details>

## Prefill details

**Prefill rounds each multiply input to BF16 once and adds up in FP32, for
every chunk size and cache format.**

<details>
<summary><b>Attention, shared passes and the opt-in fast path</b></summary>

- Attention during prefill uses BF16 inputs and FP32 sums. Heads that share a
  KV head reuse the same loaded data.
- For Qwen3, the server waits 2 ms to collect prompts and cached suffixes of
  up to 256 new tokens. One shared pass holds up to 2,048 rows, or 256 while
  a foreground chat decodes.
- An unfinished chunk keeps its shape until it commits. A short remaining
  tail can then join a shared pass.
- `SPARK_ADAPTIVE_PREFILL=1` turns on a faster attention path for prompts of
  at least 6,144 tokens with an FP32 cache. **It changes the output,** so it
  is off by default.

`SPARK_PROFILE=1` reports a late 16-chat step's GPU phase times, launch and
wait counts, and tracked peak device bytes.

</details>

## Hooks for sequence-parallel prefill

**Sequence-parallel prefill splits one long prompt across several processes
or Sparks, each reading one slice.** The [spark-seqpar](../seqpar/README.md)
crate drives these hooks: `sequence_begin`, `sequence_project`,
`sequence_export_into`, `sequence_import`, `sequence_finish_layer` and
`sequence_finish`.

<details>
<summary><b>The rules for using the hooks</b></summary>

1. The session must be empty, hold the full model and use an FP32 cache.
2. Layers run in order.
3. Before a layer finishes, the caller must supply the KV cache for every
   earlier token in the prompt.
4. `sequence_finish` returns the logits for the last local token and commits
   the cache.
5. Only the process that owns the prompt's final token can go on to decode.

Ordinary `step`, `fork` and `truncate` calls are refused while a sequence is
in progress. `reset` cancels it. The hooks use the same arithmetic and chunk
sizes as ordinary FP32 prefill.

</details>

**Where next:** [spark-gpu-probe](../gpu/README.md) · [spark-serve](../serve/README.md) ·
[Validation](../../docs/validation.md)
