# Qwen3-32B reference fixtures

> **Legacy debugging data.** These files are older and narrower than the
> broader BF16 CUDA suites in `fixtures/qwen3-32b-cuda-suite/`.
> [Validation](../../docs/validation.md) records reference versions, every
> strict-agreement failure and the scope of each check.

**This folder holds fixtures: saved expected output that a check compares
Vade's Qwen3-32B against.** It is for people debugging that model. There are
no model weights and no real conversations in these files.

## In short

**The same three made-up chats as Qwen3-8B, made by the official model code
in FP32, for debugging.**

<dl>
<dt>Model</dt>
<dd>Official Qwen3-32B.</dd>
<dt>Made with</dt>
<dd>Hugging Face Transformers on the CPU, loading one layer at a time, with
plain ("eager") attention. Vade took no part.</dd>
<dt>Reference precision</dt>
<dd>FP32, promoted from the BF16 checkpoint. So these files are a debugging aid,
not the pass mark for a BF16 model.</dd>
<dt>Cases</dt>
<dd>The same three made-up chats as <a href="../qwen3/README.md">Qwen3-8B</a>:
short, medium and long.</dd>
<dt>Used by</dt>
<dd><code>hf-batch-check</code></dd>
</dl>

BF16 and FP32 are 16-bit and 32-bit number formats. FP32 keeps more digits;
the model's own weights are BF16.

## Run the check

**Stop the server, build the checker, run one command.** You need the Qwen3-32B
checkpoint, downloaded as in [Models](../../docs/models.md).

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
     models/Qwen/Qwen3-32B fixtures/qwen3-32b/short.json \
     fixtures/qwen3-32b/medium.json fixtures/qwen3-32b/long.json --bf16-kv
   ```

It prints the top-token matches (`top1`), the score errors, and the KV cache
(the model's memory of the conversation) and peak GPU memory.

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, with Qwen3-32B in
models/Qwen/Qwen3-32B, stop spark-serve if it is running. Build the kernels
with `cargo kernels`, then build hf-batch-check with
`cargo +1.99.0 build --release -p spark-engine --bin hf-batch-check`. Run it
on the three files in fixtures/qwen3-32b/ with --bf16-kv, as shown in
fixtures/qwen3-32b/README.md. Report every "HF fixture=" line and the final
line. Do not edit the fixtures or their tolerances. Do not commit or push.
```

## What to expect

**56 of 56 positions match these files. The broader suite still has
disagreements.**

| Comparison | Matching positions |
| --- | ---: |
| These FP32 files | 56 of 56 |
| Native BF16, earlier build | 56 of 56 |

The native BF16 reference was made on the CPU, is kept outside this
repository and was recorded on an earlier build.

A match covers only these saved prompts and positions. On the broader suite,
Vade does not yet match every position; [Validation](../../docs/validation.md)
lists every disagreement.

<details>
<summary><b>The broader suite figures, and what is in the folder</b></summary>

On the broader Qwen3-32B suite, Vade matched 1,515 of 1,536 positions on
current main `603f95b`, in one session, and failed strict agreement. The
earlier build `80e616c`, before the shared-prefix change (shared9), matched
1,517.

<dl>
<dt><code>short.json</code>, <code>medium.json</code>,
<code>long.json</code></dt>
<dd>One chat each: the messages, the formatted prompt, its token IDs, the
reference answer, and the top 20 scores at each checked position.</dd>
<dt><code>short-0.f32</code>, <code>short-7.f32</code>, and so on</dt>
<dd>The full row of scores at one position, as little-endian FP32 numbers.</dd>
</dl>

Scores, or logits, are the model's raw score for every possible next token.
Top-1 is the highest-scoring token at a position. Settings: thinking off;
each answer token is the highest-scoring one. Library versions are in each
JSON file.

| Case | Prompt tokens | Checked positions |
| --- | ---: | ---: |
| Short | 25 | 8 |
| Medium | 281 | 24 |
| Long | 1,981 | 24 |

</details>

**Where next:** [Validation](../../docs/validation.md) · [Models](../../docs/models.md) ·
[Qwen3-8B fixtures](../qwen3/README.md)
