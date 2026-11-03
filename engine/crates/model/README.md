# engine-model

**This crate reads and checks a model's files before any weight reaches the
GPU.** It catches a wrong, missing or cut-off file early, with a clear error,
instead of halfway through a load. It uses only the Rust standard library.

This page is for people reading or changing how Vade reads model folders.
To download a model, see [Models](../../docs/models.md).

[Deferred GLM text metadata](GLM.md) describes config and index inspection.
It does not enable model execution. [Host CSF decoding](CSF.md) describes
the scale-byte codec and its public source.

[Sinkhorn stream blocks](MHC.md) documents the mHC math, host reference,
device primitives, and pending device gates.

<details>
<summary><b>Terms on this page</b>: checkpoint, tensor, shard, safetensors, tied embeddings</summary>

<dl>
<dt>Checkpoint</dt>
<dd>A model folder: <code>config.json</code>,
<code>model.safetensors.index.json</code> and the weight files the index
names.</dd>
<dt>Tensor</dt>
<dd>One named block of weights, such as one layer's attention matrix.</dd>
<dt>Shard</dt>
<dd>One weight file. Large models are split across several.</dd>
<dt>safetensors</dt>
<dd>The weight-file format: a small JSON header, then raw numbers.</dd>
<dt>Tied embeddings</dt>
<dd>The input and output layers share one weight matrix.</dd>
</dl>

</details>

## Where it fits

**Every crate that opens a model folder or a JSON file uses this one.**

<dl>
<dt><a href="../gpu/README.md">spark-gpu-probe</a></dt>
<dd>Loads the weights this crate has located and checked.</dd>
<dt><a href="../engine/README.md">spark-engine</a>,
<a href="../serve/README.md">spark-serve</a></dt>
<dd>Use it to read configs and find every tensor.</dd>
<dt><a href="../tp/README.md">spark-tp</a></dt>
<dd>Uses it to read configs and find every tensor.</dd>
<dt><a href="../tokenizer/README.md">engine-tokenizer</a>,
<a href="../../tools/bench/README.md">vade-bench</a></dt>
<dd>Use its JSON reader.</dd>
</dl>

## Inspect a checkpoint

**One command prints a model folder's shape and sizes without reading any
weights.**

### Do it yourself

1. Build the inspector.

   ```sh
   # Run from the checkout root.
   cargo +1.99.0 build --release -p engine-model --bin inspect-checkpoint
   ```

2. Point it at a model folder.

   ```sh
   ./target/release/inspect-checkpoint models/Qwen/Qwen3-8B
   ```

   It prints `validation=metadata-and-shape-only`, the shard and tensor
   counts, the total tensor bytes and the model's shape. It reads no weights.

3. Optional: look at one weight file. It prints the tensor count and the
   header and payload sizes.

   ```sh
   cargo +1.99.0 build --release -p engine-model --bin inspect-tensor-file
   ./target/release/inspect-tensor-file \
     models/Qwen/Qwen3-8B/model-00001-of-00005.safetensors
   ```

### Ask your coding agent

```text
In the Vade checkout, build inspect-checkpoint with
cargo +1.99.0 build --release -p engine-model --bin inspect-checkpoint,
then run ./target/release/inspect-checkpoint on the model folder I name.
Show the printed summary. Report any error exactly. Do not download,
delete or change any model file.
```

## Test it

**The unit tests need no GPU.** They check the parser against the bundled
Qwen3-8B metadata described below.

```sh
cargo +1.99.0 test -p engine-model --offline
```

## What it accepts

**Official layouts of the supported models, and nothing else.**

Accepted:

- Qwen3, dense (`Qwen3ForCausalLM`) and expert (`Qwen3MoeForCausalLM`).
- Qwen3.8-27B, BF16 and official FP8.
- Gemma 4 31B text and its official assistant.
- Qwen3 SiLU activation and the optional official YaRN setting.
- Separate input and output embeddings.

Rejected with an error:

- Unrecognized architectures or unsupported geometry.
- Non-BF16 Qwen3 weights, and unpublished or re-quantized layouts.
- Multimodal serving and Gemma 4 26B-A4B execution.
- Unsupported Qwen3 bias, scaling or sliding-window settings.
- Tied embeddings as a main model. These load only as a draft model, such as
  Qwen3-0.6B.

## What the checks do not prove

**The checks cover names, shapes, types, index entries and byte counts, not
the weight bytes themselves.** To prove the bytes are the published ones,
compare each file's `sha256sum` with the checksum on the model's official
page.

Model folders must not change while Vade has them open.

<details>
<summary><b>What is in the crate</b></summary>

<dl>
<dt><code>json</code></dt>
<dd>A small JSON parser with fixed size and depth limits.</dd>
<dt><code>safetensors</code></dt>
<dd>Reads weight-file headers and the shard index without reading the
weights.</dd>
<dt><code>qwen3</code></dt>
<dd>Checks the model config and the name, shape and type of every tensor.</dd>
<dt><code>qwen3_5</code></dt>
<dd>Qwen3.8/3.5 text geometry, BF16 and FP8 layout, and recurrent state.</dd>
<dt><code>gemma4</code>, <code>gemma4_assistant</code></dt>
<dd>Gemma 4 31B text weights and Google's official assistant.</dd>
<dt><code>cartridge</code></dt>
<dd>Identifies a model folder for <code>spark-serve --list</code> and for
start-up.</dd>
<dt><code>flash</code></dt>
<dd>Checkpoint layout for the Qwen3.8-Flash-Next research engine.</dd>
<dt><code>placement</code></dt>
<dd>Pure ownership calculations in global model coordinates, independent of links.</dd>
</dl>

</details>

<details>
<summary><b>Size limits that protect against oversized files</b></summary>

| Item | Limit |
| --- | ---: |
| Weight-file header | 16 MiB |
| Shard index | 16 MiB |
| `config.json` | 1 MiB |
| JSON nesting depth | 32 |
| Weight files per checkpoint | 1,024 |

</details>

<details>
<summary><b>The bundled Qwen3-8B metadata</b></summary>

`evidence/` holds the official Qwen3-8B `config.json` and weight index,
unchanged. [Provenance](../../docs/provenance.md) records their source
revision and checksums.

| Property | Value |
| --- | ---: |
| Layers | 36 |
| Hidden width | 4,096 |
| Feed-forward width | 12,288 |
| Query heads, KV heads | 32, 8 |
| Vocabulary | 151,936 |
| BF16 tensors | 399 |
| Total weight bytes | 16,381,470,720 |

The config allows 40,960 positions. That is the model's limit, not a claim
about answer quality at that length.

</details>

**Where next:** [spark-gpu-probe](../gpu/README.md) ·
[Historical memory ledger](MEMORY.md) ·
[Qwen3-8B reference fixtures](../../fixtures/qwen3/README.md)
