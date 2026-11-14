# Qwen3-30B-A3B reference fixtures

> **Legacy reference data.** These files are older and narrower than the
> broader BF16 CUDA suite in `fixtures/qwen3-30b-a3b-cuda-suite/`.
> [Validation](../../docs/validation.md) records reference versions, every
> strict-agreement failure and the scope of each check.

**This folder is for people checking Vade's expert model against saved
answers from the official model code.** There are no model weights and no real
conversations in these files.

## In short

**Native BF16 references for three made-up chats. The check fails today, on
the long case's routing limit.**

<dl>
<dt>Model</dt>
<dd>Official Qwen3-30B-A3B, Qwen's expert model: each layer holds many small
networks (experts), and each token uses only a few.</dd>
<dt>Made with</dt>
<dd>Hugging Face Transformers on the CPU, using the makers' own plain ("eager")
attention, router and expert code. Vade took no part.</dd>
<dt>Reference precision</dt>
<dd><b>Native BF16</b>, the checkpoint's own precision.</dd>
<dt>Cases</dt>
<dd>Three made-up chats: short, medium and long.</dd>
<dt>Extra data</dt>
<dd>The experts the router (the small network that picks them) chose, and their
weights.</dd>
<dt>Used by</dt>
<dd><code>hf-batch-check</code></dd>
<dt>Latest recorded result</dt>
<dd><b>Fails</b>, on the long case's routing limit. Answer tokens match 50 of
50.</dd>
</dl>

## Run the check

**Stop the server, build the checker, run one command.** You need the
Qwen3-30B-A3B checkpoint, downloaded as in [Models](../../docs/models.md).

### Do it yourself

1. [Stop `spark-serve`](../../docs/server.md#stopping) so the GPU is free.

2. Build the kernels and the checker.

   ```sh
   # Run from the checkout root.
   cargo kernels
   cargo +1.99.0 build --release -p spark-engine --bin hf-batch-check
   ```

3. Run it.

   ```sh
   ./target/release/hf-batch-check \
     target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx \
     models/Qwen/Qwen3-30B-A3B fixtures/qwen3-30b-a3b/short.json \
     fixtures/qwen3-30b-a3b/medium.json fixtures/qwen3-30b-a3b/long.json \
     --bf16-kv
   ```

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, with Qwen3-30B-A3B in
models/Qwen/Qwen3-30B-A3B, stop spark-serve if it is running. Build the
kernels with `cargo kernels`, then build hf-batch-check with
`cargo +1.99.0 build --release -p spark-engine --bin hf-batch-check`. Run it
on the three files in fixtures/qwen3-30b-a3b/ with --bf16-kv, as shown in
fixtures/qwen3-30b-a3b/README.md. Report every "HF fixture=" line, the
routing results and the final line. Do not edit the fixtures or their
tolerances. Do not commit or push.
```

## What to expect

**The check fails today, and the failure is kept visible.** On the latest
recorded run (main `803bd3d`), every answer token matches, but the long case
changes too many expert choices. The fixtures and tolerances stay unchanged.

<details>
<summary><b>The recorded results, build by build</b></summary>

| Result | Value | Build |
| --- | --- | --- |
| Matching answer tokens | **50 of 50** | Main `803bd3d` |
| Before the shared-prefix change | 49 of 50 | Main `03068e9` |
| Long case, changed experts | 25 (23 allowed) | All three |

- **50 of 50** is the latest recorded run, on main `803bd3d`. It was first
  seen on the shared-prefix attention build `39aceb2`, merged into main as
  `430fdbc`.
- **49 of 50** was measured on main `03068e9`, in the same session as
  `39aceb2`.
- **Still failing,** unchanged on all three builds above: the long case's
  routing limit, 25 changed expert choices against 23 allowed.

On current main (`603f95b`) this fixture still gives 50 of 50 IDs and still
fails the routing limit (25 changed expert choices, 23 allowed), one session.

The token that used to differ: medium case, answer index 14. Vade used to
pick token 264, where this CPU fixture has 2441. The official model on CUDA
picks 264 there too. [Validation](../../docs/validation.md#expert-model-cpu-and-cuda-references-disagree)
shows both runs' scores.

A stricter test would hold scores and routing weights to the
`fp32_comparison` numbers themselves, not twice them. Vade does not meet that
on every case. Neither does the official Transformers code on a GPU: against
its own CPU run, its short-case scores differ by 1.44, where that bound is
0.967.

A match covers only these saved prompts and positions. On the broader expert
suite, Vade matched 1,498 of 1,536 positions on current main `603f95b`, in
one session, and failed strict agreement. The earlier build `80e616c`,
before the shared-prefix change (shared9), matched 1,502.

</details>

<details>
<summary><b>What the checker requires, in full</b></summary>

Scores, or logits, are the model's raw score for every possible next token.
It passes only when all three hold:

1. Every answer token matches.
2. Scores stay within twice the `fp32_comparison` difference for that case.
3. At most 3% of the router's expert choices differ.

It prints the routing-weight error but does not fail on it.

</details>

<details>
<summary><b>What is in the folder</b></summary>

<dl>
<dt><code>short.json</code>, <code>medium.json</code>,
<code>long.json</code></dt>
<dd>The messages, the formatted prompt, its token IDs, the reference answer, and
the top 20 scores at each checked position.</dd>
<dt><code>short-0.f32</code>, <code>short-7.f32</code>, and so on</dt>
<dd>The full row of scores at one position. BF16 values stored as little-endian
FP32, with no added precision.</dd>
<dt><code>short-routing.json</code>, and so on</dt>
<dd>The experts the router chose, and their weights, at the saved
positions.</dd>
</dl>

Prompt length and checked positions per case:

| Case | Prompt tokens | Checked positions |
| --- | ---: | ---: |
| Short | 25 | 8 |
| Medium | 281 | 18 |
| Long | 1,981 | 24 |

Each case also has an `fp32_comparison` block. It records how far an earlier
FP32 reference run was from this BF16 run. Those runs also used different
library versions, so the numbers mix precision and version differences. They
are not a measure of run-to-run noise.

An earlier FP32 run is kept separately in
[`qwen3-30b-a3b-fp32`](../qwen3-30b-a3b-fp32/README.md) as a debugging
reference. Settings: thinking off; each answer token is the highest-scoring
one. Library versions are in each JSON file. BF16 and FP32 are 16-bit and
32-bit number formats; FP32 keeps more digits.

</details>

**Where next:** [Validation](../../docs/validation.md) · [Models](../../docs/models.md) ·
[FP32 fixtures](../qwen3-30b-a3b-fp32/README.md)
