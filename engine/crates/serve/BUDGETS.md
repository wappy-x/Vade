# Context and serving budgets

**How `spark-serve` decides how much memory each request may use.** This
page is for people reading or changing the server's memory code. It
describes the rules, not a measured speed or memory peak.

The user-facing limits are in [Server and API](../../docs/server.md#how-it-sizes-itself).

<details>
<summary><b>Terms on this page</b>: context, KV cache, admission, slot, prefix cache, MiB, GiB</summary>

<dl>
<dt>Context</dt>
<dd>The prompt plus the answer, in tokens.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Admission</dt>
<dd>The check that a request fits in memory before it starts.</dd>
<dt>Slot</dt>
<dd>One place for an active chat. Slots share one memory pool.</dd>
<dt>Prefix cache</dt>
<dd>Saved KV cache from earlier prompts, reused when a new prompt starts the
same way.</dd>
<dt>MiB, GiB</dt>
<dd>1,048,576 bytes and 1,073,741,824 bytes.</dd>
</dl>

</details>

## In short

**Each request reserves its prompt plus its requested answer before it runs.
If the pool is full, it waits.**

- The server sizes each model from its checkpoint and the memory Linux
  reports as available.
- It leaves 10% of that memory, and at least 1 GiB, for everything else.
- Weights and the measured shared workspace are subtracted first.
- Limits are ceilings. Conversation memory is not grabbed at start-up.

## The flags

**Seven flags, each setting one limit.**

<dl>
<dt><code>--context-tokens N</code></dt>
<dd>Prompt plus answer capacity, within the model's ceiling and the memory
plan.</dd>
<dt><code>--input-tokens N</code></dt>
<dd>Longest prompt. Defaults to the context minus one.</dd>
<dt><code>--output-tokens N</code></dt>
<dd>Longest answer. Omitted or oversized request limits use the smaller of this
cap and the context left after the tokenized prompt.</dd>
<dt><code>--gpu-budget-mib N</code></dt>
<dd>GPU allocation ceiling.</dd>
<dt><code>--kv-budget-mib N</code></dt>
<dd>Model-specific KV allowance. For Qwen3, this internal ledger is sized as if
the KV were FP32, although the served KV is BF16.</dd>
<dt><code>--session-budget-mib N</code></dt>
<dd>Total capacity for active and retained conversations.</dd>
<dt><code>--max-active N</code></dt>
<dd>Active slots, 2 to 32. A slot is not a full-context allocation.</dd>
</dl>

Automatic sizing reads Linux `MemAvailable`, which includes file cache the
system can reclaim.

## What admission does not prove

**Fitting in memory says nothing about answer quality at a new context
length.** YaRN, an explicit Qwen3 setting for longer context, may change
answers and is off by default. [Models](../../docs/models.md) and
[Validation](../../docs/validation.md) record the supported ceilings and
measured reference agreement.

## The detail

**What each request is charged, how active work and the prefix cache share
one pool, and the other model layouts.**

<details>
<summary><b>What a Qwen3 request is charged</b></summary>

The BF16 KV cache a request holds is:

```text
layers × KV heads × head dimension × 2 (keys and values) × tokens × 2 bytes
```

What admission charges for each request:

- Those real BF16 bytes for its prompt plus requested answer.
- Bounded allowances for scratch memory, page rounding and host-side vectors.
- Draft state only for its bounded window. Plain and shared-prefix requests
  need only the target model's state.

What it does not charge:

- A second, FP32-sized copy of the KV cache.
- Shared, read-only physical pages more than once.

</details>

<details>
<summary><b>Active work and the prefix cache share one pool</b></summary>

The active and retained ceilings overlap: the prefix cache can borrow
session capacity that is idle.

1. Before new work starts, the scheduler evicts least recently used cache
   entries until active reservations, plus retained pages not already
   covered by a request, fit the pool.
2. Up to 128 entries can be retained.
3. Unique physical KV pages are counted once. Token and logit vectors are
   charged per entry.

Completed sessions can delay their CUDA teardown while a follow-up arrives.
Their resident bytes stay charged and can be reclaimed under pressure. Cached
snapshots keep no compute scratch.

Recorded once on main `430fdbc`: two independent 40,000-token Qwen3-32B
prompts ran at the same time. Their admission reservation was 28,038,531,328
bytes, and the sampled GPU peak was 90,469,040,128 bytes.

</details>

<details>
<summary><b>Other model layouts</b></summary>

<dl>
<dt>Qwen3.8-27B</dt>
<dd>BF16 KV for its full-attention layers, FP32 recurrent state, and optional
MTP or sibling draft state.</dd>
<dt>Gemma 4 31B</dt>
<dd>Local and global KV caches, plus a separate optional allowance for the
assistant draft.</dd>
</dl>

Each model's admission follows its real cache layout. The Qwen3 per-token
formula must not be applied to a whole hybrid or Gemma model.

</details>

<details>
<summary><b>The owner slot</b></summary>

With at least three slots and an owner key, Qwen3 admission can keep one slot
and memory for a short owner request. Without that key, all slots are
ordinary client capacity.

More slots do not guarantee that every long request runs at once.

</details>

**Where next:** [spark-serve](README.md) ·
[Server and API](../../docs/server.md#how-it-sizes-itself)
