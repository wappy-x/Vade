# DGX Spark platform

**Vade is built for one machine, the NVIDIA DGX Spark, and does not try to
run everywhere.** Its GB10 chip pairs a Blackwell GPU with an Arm CPU on
128 GB of unified memory.

This page covers what Vade uses on the Spark, what it leaves alone, and what
runs on one Spark or several. It is for anyone deciding whether Vade fits
their hardware.

<details>
<summary><b>Terms on this page</b>: GB10, unified memory, <code>sm_121</code>, NVPTX, PTX, <code>core</code>, KV cache, BF16, tensor cores</summary>

<dl>
<dt>GB10</dt>
<dd>The Spark's chip: a GPU and an Arm CPU sharing one memory pool.</dd>
<dt>Unified memory</dt>
<dd>That shared pool, used by weights, conversations and the OS.</dd>
<dt><code>sm_121</code></dt>
<dd>The GB10 GPU's architecture name, which compilers target.</dd>
<dt>NVPTX, PTX</dt>
<dd>Rust's GPU compiler target, and the NVIDIA format it produces.</dd>
<dt><code>core</code></dt>
<dd>Rust's minimal standard library, which can run on a GPU.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>BF16</dt>
<dd>A 16-bit number format.</dd>
<dt>Tensor cores</dt>
<dd>The GPU's matrix-multiply units.</dd>
</dl>

</details>

## What Vade targets

**The GPU runs every kernel, the CPU runs the server, and one memory pool
holds everything.**

<dl>
<dt>GB10 GPU</dt>
<dd>Runs every kernel, built for native <code>sm_121</code>.</dd>
<dt>Arm CPU</dt>
<dd>Runs the server, tokenizer and scheduler.</dd>
<dt>Unified memory</dt>
<dd>One copy of the weights, every chat's KV cache, scratch.</dd>
<dt>Checkpoint storage</dt>
<dd>Read through small pinned buffers into GPU memory.</dd>
<dt>Cables between Sparks</dt>
<dd>Carry exchanges when a model is split across Sparks.</dd>
</dl>

- The Rust NVPTX build rebuilds `core` for the GPU's native `sm_121`.
- Weight files go straight to GPU memory; no second copy stays on the CPU
  side.

## What Vade leaves alone

**Vade changes nothing on the system, and runs as an ordinary user.**

- NVIDIA's DGX OS and GPU driver stay as installed.
- Vade makes no system or network changes.
- It includes no general hardware abstraction layer and no replacement system
  components.

## What runs where

**One Spark runs every supported model; two Sparks run the dense Qwen3 models
split with tensor parallelism.**

- **One Spark: supported.** Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B, Qwen3.8-27B
  (BF16 and official FP8) and Gemma 4 31B.
- **Two Sparks: supported.** Dense Qwen3-8B and Qwen3-32B, split with tensor
  parallelism, for generation and HTTP serving. That includes a hybrid mode
  that splits prefill and then splits decode. Qwen3-8B on two ranks matched
  one Spark on the retained checks. On this build, Qwen3-32B is not
  requalified and speed across ranks is unmeasured;
  [Multiple Sparks](multi-spark.md#what-has-been-tested) records exactly what
  was checked.
- **More than two Sparks: unverified.** The same dense Qwen3 path. The
  launcher accepts it; physical runs remain unverified on this build.

## Defaults on one Spark

**Exact output, one copy of the weights, and memory reserved before a chat
starts.**

- One copy of the weights, shared by every chat. BF16 models keep one BF16
  copy; official FP8 weights stay FP8.
- Tensor-core prefill (prompt reading), and a BF16 KV cache.
- Bounded admission: each chat reserves its memory before it starts.
- Experimental `fast` draft verification stays opt-in.

[Architecture](architecture.md) explains the engine, and
[Install](install.md) sets it up.

## Where the numbers are

**Current comparisons are on Benchmarks; older records are kept in History
and do not measure today's defaults.**

- [Benchmarks](benchmarks.md) holds the current comparisons. Each table names
  its prompt lengths, output lengths, precision and memory scope.
- [Multiple Sparks](multi-spark.md#what-has-been-tested) records what has been
  tested across Sparks.
- [History](history/README.md) keeps older records, such as the
  [FP32-cache measurements](history/fp32-cache-measurements.md) and
  [long-context results](history/longctx-performance.md) from 2 October 2026.
  Their FP32 KV results do not measure the current BF16 default.

**Where next:** [Install](install.md) · [Architecture](architecture.md) ·
[Multiple Sparks](multi-spark.md)
