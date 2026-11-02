# Models

**Six official models have a single-Spark serving path. Flash-Next FP8 runs
on 2, 4 or 6 linked Sparks through `flash-serve`.**

The single-Spark catalog below covers text models from Qwen3, Qwen3.8 and Gemma 4.
Four more are in development and 29 are planned. Vade retains the official
checkpoint weights at their published precision. For the current
Flash-Next routes, use [INSTALL.md](../../docs/INSTALL.md), the
[Flash server guide](flash-server.md) and its
[dated benchmark evidence](../../docs/benchmarks.md).

For anyone picking a model or setting one up. Speed is on
[Benchmarks](benchmarks.md); correctness is on [Validation](validation.md).

<details>
<summary><b>Terms on this page</b>: checkpoint, BF16, FP8, dense, mixture of experts, hybrid, KV cache, context, draft model</summary>

<dl>
<dt>Checkpoint</dt>
<dd>A model's published weight files, plus its config and tokenizer.</dd>
<dt>BF16, FP8</dt>
<dd>16-bit and 8-bit number formats for weights. FP8 files also carry
<em>scales</em> that restore each value's size.</dd>
<dt>Re-quantize</dt>
<dd>Convert weights to a smaller number format. Vade never does this.</dd>
<dt>Dense model</dt>
<dd>Every token uses all of the weights.</dd>
<dt>Mixture of experts</dt>
<dd>Each token uses a few small <em>expert</em> networks out of many. In a name
like 30B-A3B, 30B is the total size (30 billion parameters) and A3B the part
active for each token (about 3 billion).</dd>
<dt>Hybrid model</dt>
<dd>Ordinary attention layers mixed with recurrent layers that carry a running
summary forward.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Context</dt>
<dd>The most tokens one conversation can hold: prompt and answer together.</dd>
<dt>Draft model</dt>
<dd>A small model that guesses tokens ahead. By default the big model checks
every guess, so answers do not change.</dd>
<dt>Revision pin</dt>
<dd>The exact commit of a model repository, so everyone downloads identical
files.</dd>
<dt>GB, GiB</dt>
<dd>GB is 1,000,000,000 bytes. GiB is 1,073,741,824 bytes, about 1.07 GB.</dd>
</dl>

</details>

## Runs today

**Six models in main, each on one Spark; Qwen3-8B and Qwen3-32B can also be split across several.**

Flash-Next uses `flash-serve`, described [below](#qwen38-flash-next-fp8), rather
than the single-Spark `spark-serve` route in this table. Available source paths
do not imply that each model has the same qualification coverage.

| Model | Maker | Kind | Download (GiB) |
| --- | --- | --- | ---: |
| Qwen3-8B | Alibaba Qwen | Dense | 16 |
| Qwen3-32B | Alibaba Qwen | Dense | 62 |
| Qwen3-30B-A3B | Alibaba Qwen | Mixture of experts | 57 |
| Qwen3.8-27B | Alibaba Qwen | Hybrid | 52 |
| Qwen3.8-27B-FP8 | Alibaba Qwen | Hybrid, FP8 | 29 |
| Gemma 4 31B | Google | Dense | 59 |

- Qwen3.8-27B-FP8 is the official FP8 release. Gemma 4 31B is served as text
  only; images and other inputs are not.
- Splitting across Sparks, and what has been tested that way:
  [Multiple Sparks](multi-spark.md).

<details>
<summary><b>How they differ</b>, and other Qwen3 checkpoints</summary>

- **Dense (Qwen3-8B, Qwen3-32B).** Every token uses all of the weights.
- **Mixture of experts (Qwen3-30B-A3B).** Each layer holds 128 small expert
  networks, and each token uses 8 of them. All weights stay in memory.
  Different prompts running together usually pick different experts, so they
  read more weights per step than one prompt repeated.
- **Hybrid (Qwen3.8-27B).** Ordinary attention layers mixed with recurrent
  linear-attention layers, which carry a running FP32 state.
- **Gemma 4 31B.** A dense model that mixes local attention (a recent window
  of tokens) with global attention (the whole conversation).

The Qwen3 loader also accepts other official Qwen3 BF16 checkpoints with the
same layout. Those are untested, so they are not listed. The Qwen3.8 and
Gemma loaders accept only their validated layouts.

</details>

## In development and planned

### Qwen3.6-35B-A3B-FP8: world 1 candidate

**World-1 HTTP serving is enabled and eager.** The generic hybrid route serves
the official FP8 MoE checkpoint through the typed loader. `spark-serve` sets
the typed path's graph storage budget to 0, so HTTP capture/replay is disabled.
Native typed decode can capture graphs with a funded allowance; see
[Typed decode graphs](typed-decode-graphs.md). The 35B HTTP bytes gate remains
pending; enabling the route does not qualify HTTP correctness or performance.

The candidate uses the common typed loader, fixed per-token K128 W8A8
projections, canonical BF16 projections, token-step GDN and causal attention.
Its qualification API supports text, one rank, eager execution, batch widths
1/2/8/16, cache miss/hit/fork/append, scheduled prompt chunks and built-in
greedy MTP depths 2/3. These are implemented capabilities requiring native
proof. Multimodal inputs, multi-rank serving, sequence-parallel peers and fast
drafting remain refused. Native graph capture/replay needs its own GPU proof;
HTTP remains eager. Qwen3.5-122B-A10B-FP8 awaits the world 2 milestone.

Typed eager decode retains the exact host-written GDN descriptors rather than
downloading their immutable history addresses in every layer. Plain greedy
singleton serving selects on the device; its full logits remain there and are
downloaded only when the host token filter rejects the selected token. Sampled
requests and requests with history penalties keep the full-row sampling path.
The fixed FP8 entry uses the existing M16/N64 tile for at most 16 rows. Both
tiles retain four dependent K32 MMAs, separate FP32 scale multiplications,
ascending K128 additions and the same output rounding. Paired projections
reuse quantization only with unchanged input, row/K extents and scratch;
requests partitioned across multiple 256-row projections requantize each part.

Native capture retains the complete typed layer loop and optional device
argmax. Profiling and the host routing-rejection hook run eager. Token and
descriptor uploads, output/router readbacks and waits stay outside capture.
KV replacement, attention-score bucket growth, row count, output mode,
verification layout and session identity invalidate affected graphs.
No typed graph qualification is claimed here.

Header-derived conservative funding for the official 35B checkpoint, in bytes:

| Owner | Draft off | Draft on |
| --- | ---: | ---: |
| Resident plus startup allowance | 35,777,954,688 | 36,631,694,848 |
| Shared row/draft workspace | 588,208,128 | 2,860,905,088 |
| One full request, context 131,072 | 3,529,749,824 | 5,059,809,088 |
| One full request, context 262,144 | 6,218,298,688 | 8,302,006,080 |

The active/cache/in-flight pool funds retained owners together; active slots
are derived from funded request bytes (automatic cap 16, explicit cap 32).
These bounds are not measured peaks or qualified context/concurrency limits.
HTTP qualification, context and throughput remain unadvertised.

Acceptance requires one physical Spark and pinned checkpoint, source,
binary/PTX and arithmetic identities: full finite logit rows for prompts
285/2048/8192, operator boundaries, mixed batches, all cache/chunk cells,
poison survivors, repeated starts, verifier versus sequential target rows,
MTP 0/2/3 token bytes, operator/dtype/scale proofs and measured memory pressure
with exact admission refusals. Requests must record temperature 0, top_p 1,
top_k 1, repetition penalty 1, presence/frequency penalties 0 and seed 42.
32k/128k retrieval and capacity need separate receipts before advertising
those windows; 262,144 needs its own capacity/exactness run. Earlier cloud
short-prompt and cold long-prompt receipts cover only part of this matrix.

**Four additional models are being built outside main and 29 more are planned.**

<details>
<summary><b>In development</b>: four models built outside main</summary>

Their adapters are built outside main and cannot be installed from it. Their
reference agreement and serving checks are unfinished, and no speed or
correctness result from that work is a claim about main.

| Model | Maker | Download (GiB) | Sparks planned |
| --- | --- | ---: | ---: |
| gpt-oss-20b | OpenAI | 13 | 1 |
| Mistral Small 3.2 24B | Mistral AI | 45 | 1 |
| Phi-4 | Microsoft | 28 | 1 |
| Gemma 4 26B-A4B | Google | 49 | 1 |

Gemma 4 26B-A4B is a mixture of experts.

</details>

<details>
<summary><b>Planned</b>: 29 models, with sizes and planned Spark counts</summary>

Official releases only: the makers' own files, or later NVIDIA's own
conversions as a separate class. No community fine-tunes and no community
quantizations.

- One Spark has 128 GB of memory, shared by the weights, the conversations
  and the system. So models up to about 90 GB are planned on one Spark and
  larger ones on two or more.
- The official files are already downloaded. The Spark counts are a plan,
  not a measured result.
- Sizes are the downloaded model folders, in GiB.

| Model | Maker | Download (GiB) | Sparks planned |
| --- | --- | --- | --- |
| Qwen3-14B | Alibaba Qwen | 28 | 1 |
| Qwen3.6-35B-A3B | Alibaba Qwen | 67; FP8: 35 | 1 |
| Qwen3-Coder-Next (FP8) | Alibaba Qwen | 75 | 1 |
| Qwen3.5-122B-A10B (FP8) | Alibaba Qwen | 119 | 2 |
| Qwen3-235B-A22B | Alibaba Qwen | 438; FP8: 221 | 4 or more |
| Qwen3.8-Flash-Next (BF16) | Alibaba Qwen | 336 | 2 to 4 |
| Gemma 4 12B | Google | 23 | 1 |
| Llama 3.1 8B | Meta | 15 | 1 |
| Llama 3.2 1B | Meta | 3 | 1 |
| Llama 3.2 3B | Meta | 6 | 1 |
| Llama 3.3 70B | Meta | 132 | 2 |
| Llama 4 Scout 17B-16E | Meta | 203 | 2 to 4 |
| gpt-oss-120b (MXFP4) | OpenAI | 61 | 1 |
| Devstral Small 2 24B | Mistral AI | 25 | 1 |
| Mistral Small 4 119B | Mistral AI | 113 | 2 |
| Mistral Medium 3.5 128B | Mistral AI | 125 | 2 |
| Nemotron 3 Nano 4B | NVIDIA | 8 | 1 |
| Nemotron 3 Nano 30B-A3B | NVIDIA | 59 | 1 |
| Nemotron 3.5 Lightning 30B-A3B | NVIDIA | 62 | 1 |
| Nemotron 3 Super 120B-A12B | NVIDIA | 231 | 2 to 4 |
| Kolibri-1 78B-A3.5B | Aleph Alpha | FP8: 74; BF16: 146 | 1 (FP8), 2 (BF16) |
| GLM-4.7-Flash | Z.ai | 59 | 1 |
| GLM-4.5-Air (FP8) | Z.ai | 105 | 2 |
| GLM-5.3-Flash | Z.ai | 306 | 4 |
| Kimi-Linear-48B-A3B | Moonshot AI | 92 | 1 to 2 |
| DeepSeek-V4-Flash | DeepSeek | 156 | 2 |
| DeepSeek-V4.1-Flash | DeepSeek | 476 | 4 to 8 |
| MiniMax-M2.7 | MiniMax | 215 | 2 to 4 |
| Step-3.7-Flash (NVFP4) | StepFun | 121 | 2 |

- A format in brackets is the maker's own release: OpenAI's own MXFP4 for
  gpt-oss-120b, the maker's NVFP4 for Step-3.7-Flash, and the named FP8 or BF16
  format for the others.
- Qwen3.8-Flash-Next-FP8 now has a two-Spark serving path; see
  [its current scope](#qwen38-flash-next-fp8). The unquantized BF16 checkpoint
  is not qualified by the FP8 results.

**Later: NVIDIA's 4-bit and FP8 conversions**

NVIDIA publishes its own NVFP4 (4-bit) and FP8 versions of many models:
Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, Gemma 4, Llama 3.1, 3.3 and 4,
Mistral Medium 3.5, DeepSeek V4 Flash, GLM-5.3-Flash and Nemotron 3 Ultra
550B. They are smaller and faster, but less precise than the maker's release.
Vade will test them later as their own clearly labelled "fast and small"
class, never mixed into the determinism results.

</details>

## Memory and context

**The single-Spark models use 16.4 GB (Qwen3-8B) to 65.5 GB (Qwen3-32B) of
weights. Flash-Next FP8 splits its larger checkpoint across two Sparks.**

One conversation holds up to 40,960 tokens on Qwen3 (131,072 with YaRN, an
optional stretch), 262,144 on Qwen3.8 and 32,768 on Gemma.
Flash-Next's product default is 131,072 context tokens with a 65,536-token
output cap; the checkpoint's longer native ceiling is not the product default.
Explicit output limits may equal context; each reply is capped at context minus
the tokenized prompt length. The one-hour generation deadline still applies.

<details>
<summary><b>Memory and precision</b>: weights, number formats and KV cache for each model</summary>

Weights in memory, in the makers' official formats:

| Model | Weights (GB) | Weight format | KV cache |
| --- | ---: | --- | --- |
| Qwen3-8B | 16.4 | BF16 | BF16 |
| Qwen3-32B | 65.5 | BF16 | BF16 |
| Qwen3-30B-A3B | 61.1 | BF16 | BF16 |
| Qwen3.8-27B | 53.8 | BF16 | BF16, FP32 state |
| Qwen3.8-27B-FP8 | 29.5 | FP8 with scales | BF16, FP32 state |
| Gemma 4 31B | 61.4 | BF16 | BF16 local, global |
| Qwen3.8-Flash-Next-FP8 | 185.5 | FP8 with scales | FP32 KV and recurrent state |

**Arithmetic, model by model:**

- **Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B:** BF16 multiplications, FP32 sums.
  For Qwen3-8B and Qwen3-32B the KV cache is BF16 in `spark-serve` and in
  shard-only multi-Spark mode.
- **Qwen3.8-27B:** native BF16 operation boundaries. BF16 full-attention KV
  cache; FP32 recurrent state.
- **Qwen3.8-27B-FP8:** official FP8 weights and scales, plus BF16 parts.
  Scaled weight conversion rounded to BF16; FP32 accumulation and state.
- **Gemma 4 31B:** 61,394,690,680 bytes of official BF16 text tensors. BF16
  projections with FP32 accumulation.
- **Qwen3.8-Flash-Next-FP8:** official FP8 expert tensors and scales, plus BF16
  parts. Experts use FP8 weights and dynamically quantized FP8 activations,
  with FP32 accumulation and explicit BF16 rounding boundaries. Covered BF16
  projections use mandatory high/residual BF16 tensor-core operands with FP32
  accumulation; activations remain stored in FP32. See the
  [arithmetic contract](flash-fp8-math.md) for the exact scope and reference limits.

**Also:**

- Vade keeps one copy of the weights, shared by every chat, and never
  re-quantizes them. The single-Spark Qwen3.8-27B-FP8 path converts scaled
  weights to BF16 as they are used; no full BF16 copy is kept. Flash-Next's
  default expert multiplication uses FP8 operands as described above.
- Weight sizes leave out the KV cache, scratch memory and CPU-side memory.
  [Server and API](server.md#how-it-sizes-itself) lists KV cache bytes per
  token for the Qwen3 models.
- Any setting that goes below the checkpoint's precision must be chosen
  explicitly.
- The hybrid multi-Spark mode still uses an FP32 KV cache; see
  [Multiple Sparks](multi-spark.md#not-done-yet).

</details>

<details>
<summary><b>Context length</b>: the table, and what YaRN changes</summary>

Context covers the prompt and the answer together.

| Model | Ceiling (tokens) | With YaRN (tokens) |
| --- | ---: | ---: |
| Qwen3-8B | 40,960 | 131,072 |
| Qwen3-32B | 40,960 | 131,072 |
| Qwen3-30B-A3B | 40,960 | 131,072 |
| Qwen3.8-27B, BF16 or FP8 | 262,144 | Not supported |
| Qwen3.8-Flash-Next-FP8 | 262,144 native; 131,072 product default | Not supported |
| Gemma 4 31B | 32,768 | Not supported |

- The Qwen3 and Qwen3.8 ceilings are the models' native limits. Gemma's is
  the server's automatic cap. The single-Spark server uses the ceiling by
  default, picks a shorter limit when memory is tight, and shows actual limits
  on its `ready` line. Flash-Next uses its product default unless an explicit
  context overrides it; automatic memory sizing must fund a full-context
  session or refuse startup.
- YaRN (a way to stretch context, turned on with `--rope-scaling yarn`) is
  optional. It uses factor 4 over 32,768 original positions, and it **can
  change answers**.
- KV cache, and Qwen3.8's recurrent state, grow as a conversation grows.
- A request without an answer limit can use the rest of the context.
  [Server and API](server.md#long-conversations) explains the limits and
  settings.

</details>

## Draft models

**A small draft model guesses tokens ahead, and by default the big model checks every guess, so answers do not change.**

<details>
<summary><b>Which draft each model uses</b>, how it finds it, and the flags</summary>

In the default `auto` policy, and in `strict`, the target model checks every
guess before it is returned, so drafting changes speed, not greedy answers.
The experimental `fast` policy can change output and is off by default.

| Model | Draft | Draft weights |
| --- | --- | --- |
| Qwen3-8B | Qwen3-0.6B | 1.2 GB |
| Qwen3-32B | Qwen3-0.6B | 1.2 GB |
| Qwen3.8-27B | Qwen3.5-0.8B | Not recorded |
| Gemma 4 31B | Official assistant | 939,037,192 bytes |

Qwen3-0.6B and the Gemma 4 31B assistant are BF16. Draft models are never
served on their own.

- **Qwen3-8B and Qwen3-32B** find a matching `Qwen3-0.6B` folder next to
  their own.
- **Qwen3-30B-A3B** decodes plainly, because expert drafting was slower.
- **Qwen3.8-27B**, BF16 or FP8, can use an official `Qwen3.5-0.8B` sibling
  folder. Without one, BF16 uses its built-in multi-token prediction (MTP)
  head by default. The FP8 checkpoint's built-in draft does not start yet;
  use plain decoding or a compatible sibling. Qwen3.8 drafts only for one
  active chat.
- **Gemma 4 31B** finds Google's official assistant in a sibling folder
  named after the target folder plus `-assistant`. Only one active chat uses
  it; two or more decode plainly.

Flags: `--draft-model PATH` picks a draft explicitly, and `--draft plain`
turns drafting off. [Speculative decoding](speculative-decoding.md) explains
the checking, the per-model rules and the limits of the evidence.

</details>

## Download

**Four Qwen3 models at pinned revisions, with the same helper as Install.**

### Do it yourself

1. Define the `fetch_model` download helper in your terminal. Copy it from
   [Install step 4](install.md#4-download-qwen3-8b); it is the same helper.

2. Fetch the models you want. Each line downloads one model into
   `models/Qwen/<model>`.

   ```sh
   fetch_model Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
   fetch_model Qwen/Qwen3-32B 9216db5781bf21249d130ec9da846c4624c16137
   fetch_model Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
   fetch_model Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca
   ```

3. Stop the server if it is running. A running server lowers the free-memory
   figure, so a model that fits can show `fits=no`.
   [Server and API](server.md#stopping) shows how to stop it each way it can run.

4. List the installed models.

   ```sh
   ./target/release/spark-serve --list models/Qwen
   ```

   Each model gets one line: architecture (such as `Qwen3ForCausalLM`),
   precision, file and weight bytes, whether it is supported, whether it fits
   in memory, and how many Sparks it would need. Trimmed, a line looks like
   this:

   ```text
   name="Qwen3-8B" ... precision="BF16" supported=yes fits=yes needs_sparks=1
   ```

### Ask your coding agent

```text
Download Qwen3-32B for Vade on this DGX Spark by following the "Download"
section of docs/models.md. Use the pinned revision from that page,
run everything as the current user, and never use sudo. When it finishes,
stop the Vade server if it is running, run the listing command from that
section and show the result.
```

Replace `Qwen3-32B` with the model you want.

<details>
<summary><b>Pinned revisions</b>, and things to know about downloads</summary>

| Model | Revision | Weight files |
| --- | --- | ---: |
| Qwen3-8B | `b968826` | 5 |
| Qwen3-32B | `9216db5` | 17 |
| Qwen3-30B-A3B | `ad44e77` | 16 |
| Qwen3-0.6B | `c1899de` | 1 |
| Qwen3.8-Flash-Next-FP8 | `236dfdf` | 131 |

Each revision's files on Hugging Face:
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) ·
[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/tree/9216db5781bf21249d130ec9da846c4624c16137) ·
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39) ·
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) ·
[Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13)

- Keep the config, tokenizer and weights from the same revision. When
  changing revisions, start from an empty model folder.
- Completed files are skipped. An interrupted file downloads again.
- Keep the folder names. The server uses the folder's lower-case name as the
  model ID, such as `qwen3-32b`.
- A `Qwen3-0.6B` folder next to Qwen3-8B or Qwen3-32B is picked up as its
  draft model. `--draft-model PATH` picks another; `--draft plain` turns it
  off.
- `fits` and `needs_sparks` are memory estimates only. They do not mean a
  model has been tested across several Sparks.

</details>

## Download Qwen3.8 or Gemma

**Revision pins for these are not recorded here, so pick one on the official page and keep every file from that one revision.**

This section covers the single-Spark Qwen3.8-27B and Gemma paths. Flash-Next
users should follow the [two-Spark installation guide](../../docs/INSTALL.md),
which records its checkpoint revision and weight-verification procedure.

Official pages: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ·
[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) ·
[Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it). Check the
official page and the files used for validation before downloading.

### Do it yourself

1. Pick a revision on the official page and write it down. Do not substitute
   a community quantization, and do not mix files from different revisions.

2. Download `config.json`, the tokenizer files, the weight index and every
   weight file it names into one model folder. Gemma also needs the official
   `chat_template.jinja`. Flash-Next uses the template in `tokenizer_config.json`;
   its optional `chat_template.jinja` must match, ignoring trailing newlines.
   Keep the published FP8 scales and draft tensors.

3. To use a draft, download its official checkpoint into the matching
   sibling folder, or pass `--draft-model PATH`.

4. List and start the model, then read the actual context and memory limits
   on the `ready` line. Set `MODEL` to the folder you downloaded into.

   ```sh
   ./target/release/spark-serve --list models
   MODEL=models/google/gemma-4-31B-it
   ./target/release/spark-serve "$MODEL"
   ```

- The Qwen3 download helper expects Qwen3's vocabulary files; it is not a
  Gemma download recipe.
- Which revision the validated Qwen3.8 and Gemma files came from is
  unverified on this page.

### Ask your coding agent

```text
Download the official Gemma 4 31B checkpoint for Vade on this DGX Spark by
following "Download Qwen3.8 or Gemma" in docs/models.md. Use only the
official repository. Record the revision you download and keep every file
from that one revision. Run as the current user and never use sudo. Then run
spark-serve --list on the models folder and show the result.
```

Replace `Gemma 4 31B` with the model you want.

## Check a download

**You can run the loader's file check alone, without the server.** It checks every tensor's name, shape, type and file position before loading weights onto the GPU.

### Do it yourself

1. Build the inspector, from the checkout root.

   ```sh
   cargo +1.99.0 build --release -p engine-model --bin inspect-checkpoint
   ```

2. Point it at a model folder.

   ```sh
   ./target/release/inspect-checkpoint models/Qwen/Qwen3-8B
   ```

   It reports `validation=metadata-and-shape-only`, the shard and tensor
   counts, `tensor_bytes` and `payload_bytes_read=0`. A missing or cut-off
   file produces an error.

3. This check does not read every weight byte. To verify a download
   completely, compare each file's `sha256sum` with the checksum Hugging Face
   publishes for that file.

### Ask your coding agent

```text
Check the downloaded model in models/Qwen/Qwen3-8B by following "Check a
download" in docs/models.md. Build the inspector, run it, and report its
validation line, shard and tensor counts and any error. Then compare each
file's sha256sum with the checksum Hugging Face publishes for that file.
Do not delete or re-download anything without asking.
```

## How correctness is checked

**Vade's output is compared with the makers' reference code running the official checkpoint; agreement is measured, and it is not perfect.**

- The reference is Hugging Face Transformers, at the checkpoint's own
  precision.
- Compared: the greedy token (the single most likely next token) and the
  logits (the score for every possible next token) at saved positions, each
  started from the reference's own history.
- [Validation](validation.md) owns that record: the broad CUDA suites, the
  older short fixtures, every disagreement, and the commands to run each
  check yourself.
- The loader's own file check is [above](#check-a-download).

## Qwen3.8-Flash-Next-FP8

**Flash-Next FP8 serves HTTP requests across 2, 4 or 6 Sparks through `flash-serve`.**

The FP8 deployment splits the checkpoint across Sparks. Each rank retains
its share of the weights; `spark-serve` and the dense-model `tp-serve` are
different serving paths and cannot load this checkpoint.

Follow the [FP8 install](../../docs/install-fp8.md) for the procedure and the
[Flash server guide](flash-server.md) for clients. Defaults are 131,072 context
tokens, a 65,536-token output cap and drafting depth 2. Explicit output limits may equal
context; each reply is capped at the room left after the tokenized prompt. The
[benchmark record](../../docs/benchmarks.md) separates native engine results
from the still-unqualified Rust install/start/stop lifecycle. Other rank counts
and the BF16 checkpoint need their own qualification.

## Not done yet

**Remaining model limits.**

- Serve the [in-development and planned](#in-development-and-planned)
  models.
- Serve Qwen3-0.6B on its own. It shares its input and output weights, and
  the server does not run such checkpoints on their own.
- Flash-Next results on two Sparks do not establish qualification on larger
  configurations.

**Where next:** [Server and API](server.md) · [Benchmarks](benchmarks.md) ·
[Multiple Sparks](multi-spark.md) · [Credits](CREDITS.md)
