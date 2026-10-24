# Design principles

**Four ground rules decide how Vade is built, and every other choice follows
from them.** This page gives each rule and the reason for it.

It is for anyone who wants the why: deterministic answers, precision that
follows the checkpoint, memory and batching, and safe speculative decoding.
[Architecture](architecture.md) follows a request through the code.

<details>
<summary><b>Terms on this page</b>: token, weights, BF16, tensor cores, KV cache, prefill, decode, context, dense and expert models</summary>

<dl>
<dt>Token</dt>
<dd>The unit a model reads and writes, often part of a word.</dd>
<dt>Weights</dt>
<dd>The model's learned numbers, published by its maker.</dd>
<dt>BF16, FP8, FP32</dt>
<dd>16-bit, 8-bit and 32-bit number formats.</dd>
<dt>Tensor cores</dt>
<dd>The GPU's matrix-multiply units.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Prefill, decode</dt>
<dd>Reading the prompt; then writing the answer, one token per step.</dd>
<dt>Context</dt>
<dd>The prompt and answer together, counted in tokens.</dd>
<dt>Dense model</dt>
<dd>Every token uses all of the weights, as in Qwen3-8B.</dd>
<dt>Mixture of experts</dt>
<dd>A few of many small experts per token, as in Qwen3-30B-A3B.</dd>
</dl>

The KV cache holds the attention keys and values of every token already
read. The experts in a mixture of experts are small networks.

GB means 1,000,000,000 bytes.

</details>

## Ground rules

**Exact by default, Vade's own kernels, official models only, and the
maker's precision.**

<dl>
<dt>Exact by default</dt>
<dd>Exactly the same answer, alone or batched, cached or fresh.</dd>
<dt>Own kernels</dt>
<dd>Vade's own Rust and PTX; no NVIDIA math library runs.</dd>
<dt>Official models only</dt>
<dd>No community edits or community quantizations.</dd>
<dt>Checkpoint precision</dt>
<dd>Weights stay in the maker's format, never re-quantized.</dd>
</dl>

- Anything that can change an answer is opt-in.
- NVIDIA's documents and CUTLASS are read as reference only.
- Official means the makers' own releases.

## The same answer every time

**Vade adds up each number in one fixed order, so a prompt gets the same
answer however it runs.**

A computer adds numbers in some order. With rounding, a different order can
give a slightly different sum, and that can change which token the model
picks.

An engine that picks a different code path for a bigger batch, or for a
cached prompt, can end up adding in a different order. Vade keeps one order
whether the chat runs alone, in a batch, with guessed tokens being checked,
or from a cached beginning.

- **Checked.** The merged Qwen3 gates matched alone-versus-batched and
  cached-versus-fresh output on their recorded workloads.
- **Written down.** The [GPU crate](../crates/gpu/README.md#decode-attention-order)
  spells out the attention order.
- **Pooled prefill, fixed.** Prompts prefilled together in one pooled pass
  now match the same prompts prefilled one at a time:
  [Validation](validation.md#pooled-prefill-fixed).
- **A separate claim.** Agreement with the official model is measured, and
  it is not perfect; [Validation](validation.md) has the counts and every
  disagreement.

## Precision follows the checkpoint

**Weights stay in the format the maker published, and run at that
precision.**

- **BF16 models.** BF16 multiplies on the tensor cores, with FP32 sums. That
  is what the hardware does natively. Softmax and other sums also run in
  FP32.
- **FP8 models.** The FP8 weights and their scales stay as published. No
  second BF16 weight copy is kept. Flash's routed experts use W8A8; its BF16
  projections follow the rule below.
- **One Flash projection mode.** Covered BF16 projections split each FP32
  activation into a BF16 high part and a BF16 residual, multiply both on the
  tensor cores and accumulate in FP32. This mode is always on. Other models'
  native BF16 paths keep their existing arithmetic.

Repeatability applies to the same machine, build and setup. Flash answers may
differ from older builds; changing arithmetic needs fresh determinism and
quality evidence. Activation/cache quantization and approximate verification
remain explicit choices.

Greedy tokens are judged against the makers' reference at the checkpoint's
own precision; FP32 reference runs are for debugging only.
[Models](models.md#memory-and-context) has the per-model table.

## Context and memory

**Vade sizes itself at startup, from the model's shape and the free unified
memory, and has no fixed slots.**

It picks the context limit (prompt plus answer), how many requests run at
once and the memory budgets. The `ready` line reports the choices in tokens
and bytes; each model's context ceiling is on
[Models](models.md#memory-and-context).

<details>
<summary><b>No fixed slots, one pool, growth without copying</b></summary>

- **No fixed slots.** Each request reserves room for its prompt plus the
  answer length it asks for. A long conversation and many short ones share
  one pool.
- **What a conversation costs.** A full 40,960-token Qwen3-8B conversation
  needs 6.0 GB of BF16 KV cache.
  [Server and API](server.md#how-it-sizes-itself) lists every model.
- **One pool for everything.** Weights, KV cache, scratch, pinned buffers and
  Linux share the Spark's memory. CPU and GPU reports can count the same
  bytes, so adding them overstates use.
- **Growth without copying.** Large caches grow by adding physical pages
  behind a fixed address. Branches share pages and copy one only before
  writing to it. Released pages can be kept briefly for reuse; see
  [Server and API](server.md#how-it-sizes-itself).

</details>

## Batch throughput

**Decode reads every weight once per step, so batching lets several chats
share that one read.**

- **Dense models** share one weight read across every chat in a decode step.
- **The mixture-of-experts model**, Qwen3-30B-A3B, shares the work only
  between chats that pick the same expert. Different prompts usually pick
  more experts than the same prompt repeated, so a benchmark that repeats one
  prompt overstates its throughput.
- **How Vade counts.** All finished answer tokens over the whole group's
  time, including prefill and time in the queue. A group in which some
  requests fail does not produce a rate.

[Benchmarks](benchmarks.md#read-the-tables) has the method.

## Speculative decoding stays exact

**For greedy requests, speculative decoding returns exactly the tokens the
model would have written on its own.**

Speculative decoding, which Vade's flags call drafting, proposes several
tokens cheaply, and the model checks them in one weight pass. Vade keeps the
proposals up to the first one the model disagrees with, and drops the rest.

- **Only when it pays.** That depends on how often the proposals are right
  and how much proposing costs. Vade measures both, and falls back to plain
  decoding when it does not pay.
- **One exception.** The experimental `fast` policy can change output, and
  it is off by default.

[Speculative decoding](speculative-decoding.md) has the rules for each
model, and [Server and API](server.md#speculative-decoding) the controls.

## Where the weights live across Sparks

**More Sparks mean fewer weight bytes to read on each one, but each exchange
between Sparks costs time, which can cancel the gain.**

In tensor parallelism, each Spark holds part of every layer and the ranks
exchange pieces so every rank ends up with the whole result, still added in
one Spark's order.

- **Hybrid mode** keeps a full copy on each Spark and splits prefill instead.
- **Helper Sparks** for long prompts exist, but do not yet work with normal
  `spark-serve` sessions.
- **Odd counts work.**
- **Measure, do not assume.** Speed across Sparks on the current build is
  unmeasured; [Multiple Sparks](multi-spark.md#what-has-been-tested) records
  what has been tested.

**Where next:** [Architecture](architecture.md) · [Multiple Sparks](multi-spark.md) ·
[Documentation index](README.md)
