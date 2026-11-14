# Qwen3-30B-A3B FP32 reference fixtures

> **Legacy debugging data.** This is an earlier, independent FP32 reference
> run. The BF16 fixtures in [`qwen3-30b-a3b`](../qwen3-30b-a3b/README.md) and
> the broader BF16 CUDA suite are the current references.
> [Validation](../../docs/validation.md) records reference versions, every
> strict-agreement failure and the scope of each check.

**This folder is for people debugging the expert model against a reference
made without any Vade code.**

<details>
<summary><b>Terms on this page</b>: expert model, BF16, FP32, RSS</summary>

<dl>
<dt>Expert model</dt>
<dd>Each layer holds many small networks, called experts. Each token passes
through only a few of them.</dd>
<dt>BF16, FP32</dt>
<dd>16-bit and 32-bit number formats. FP32 keeps more digits; the model's own
weights are BF16.</dd>
<dt>RSS</dt>
<dd>Resident set size: the memory a process holds, as Linux reports it.</dd>
</dl>

</details>

## In short

**An FP32 reference for three chats, made on the CPU. It does not test Vade.**

<dl>
<dt>Model</dt>
<dd>Official Qwen3-30B-A3B, BF16 checkpoint.</dd>
<dt>Made with</dt>
<dd>Hugging Face Transformers on the CPU. No Vade code and no GPU were
used.</dd>
<dt>Reference precision</dt>
<dd>FP32. The BF16 weights were promoted to FP32 exactly.</dd>
<dt>Cases</dt>
<dd>The same three chats as <code>fixtures/qwen3</code> and
<code>fixtures/qwen3-32b</code>: 25, 281 and 1,981 prompt tokens.</dd>
<dt>What it proves</dt>
<dd>A reference for these prompts only. It does not test Vade, or native BF16
arithmetic.</dd>
</dl>

## Reproduction provenance

The streaming reference oracle is retained privately outside this repository,
with its compatible CPU environment. Regeneration needs access to that private
oracle under an assigned owner task. The recorded procedure used separate
fresh processes for the short, medium and long cases, each with 12 CPU threads,
an empty output directory and all 16 official safetensors shards. No packages
were installed and the committed fixtures were never overwritten. The process
wall time and peak RSS were measured with `/usr/bin/time -v`.

The expected Hugging Face meta-device warning described the empty model
skeleton; each module that ran used loaded CPU weights. Compare regenerated
top-1 IDs with the recorded results below. See
[reference provenance](../../tools/truth/README.md) for the oracle boundary.

<details>
<summary><b>How it was made, and what the streaming loader changed</b></summary>

Words: the **router** is the small network that picks each token's experts,
and how much weight each gets. **Scores** (logits) are the model's raw score
for every possible next token.

Model shape: 48 layers, hidden size 2,048, 32 attention heads, 4 KV heads,
head dimension 128, and 128 experts per layer with 8 picked per token.

- Activations, KV (the model's memory of the conversation), attention and
  expert arithmetic all ran in FP32.
- Attention and experts used the plain ("eager") implementations.
- Generation was greedy, with thinking off.
- The installed `Qwen3MoeForCausalLM`, decoder, router and expert classes did
  all the model arithmetic.

The run streamed weights from disk so a whole FP32 model never had to fit in
memory. Only **where the weights live** changed:

- Attention, norm and router modules loaded on the CPU for their own forward
  calls.
- Expert weights loaded one expert's projections at a time.
- The checkpoint stores gate and up projections separately; they were joined
  in the order Hugging Face expects.
- The embedding and output projection were each streamed separately.
- No complete layer's expert bank, and no whole-model FP32 copy, was ever
  built. KV stayed in RAM.

The Hugging Face expert forward, routing, accumulation order and generation
were not changed.

</details>

<details>
<summary><b>Files, and the routing files</b></summary>

Each case JSON uses the existing fixture format:

- the top-1 token and top-20 scores at each generated position;
- references to the full first and last score rows;
- library versions, generation seconds, process peak RSS and CPU thread
  count.

Each `.f32` file holds 151,936 little-endian FP32 values (607,744 bytes).

Generation was capped at 8 tokens for the short case and 24 for medium and
long, with the model's normal stop at EOS, the end-of-text token.

Each `<case>-routing.json` contains:

<dl>
<dt><code>name</code></dt>
<dd>The case name.</dd>
<dt><code>num_experts</code></dt>
<dd>128</dd>
<dt><code>experts_per_token</code></dt>
<dd>8</dd>
<dt><code>norm_topk_prob</code></dt>
<dd><code>true</code></dd>
<dt><code>positions</code></dt>
<dd>Two objects: the first and last saved full-score rows.</dd>
<dt><code>row</code></dt>
<dd>Zero-based index into the case JSON's <code>rows</code>.</dd>
<dt><code>input_position</code></dt>
<dd>Zero-based position of the input token whose hidden state produced those
scores: <code>prompt_ids.length - 1 + row</code>.</dd>
<dt><code>layers</code></dt>
<dd>One entry per layer, 0 through 47, each with <code>indices</code> and
<code>weights</code>.</dd>
</dl>

About `indices` and `weights`:

- They are two parallel arrays of length 8, in the router's original top-k
  order. Expert indices start at zero.
- Weights are FP32 router softmax probabilities after picking the top 8 and
  renormalizing. They sum to about 1.
- They were captured directly from the Hugging Face router output, not
  recomputed by a separate implementation.
- Compact JSON keeps the full serialized FP32 values; nothing is rounded.

Row 0 captures the routing of the final prompt token, which predicts the
first answer token. Later rows capture the routing of the previous generated
token. The two snapshots cover 96 layer-and-position pairs per case.

</details>

<details>
<summary><b>Recorded results: environment, timings, memory and top-1 tokens</b></summary>

**Environment:** PyTorch `2.14.1+cpu`, Transformers `5.19.0.dev0`, NumPy
`2.5.3`, safetensors `0.8.0`. The installed Transformers includes
`Qwen3MoeForCausalLM`. No packages were installed. Each process used 12 CPU
threads and one interop thread. The checkpoint index reports 61,064,245,248
bytes of BF16 tensors.

Time per case:

| Case | Greedy rows | Generation (s) | Process wall (s) |
| --- | ---: | ---: | ---: |
| short | 8 | 41.624 | 43.93 |
| medium | 18 | 89.682 | 91.95 |
| long | 24 | 150.729 | 153.01 |

Size and memory per case:

| Case | Prompt tokens | Peak RSS (bytes) |
| --- | ---: | ---: |
| short | 25 | 3,140,800,512 |
| medium | 281 | 3,221,651,456 |
| long | 1,981 | 4,138,528,768 |

- Generation seconds cover `model.generate`.
- Process wall time and peak RSS were measured with `/usr/bin/time -v`, each
  case in its own process.
- RSS includes the process's mapped pages, loaded FP32 weights, activations,
  KV and scratch. It is not whole-machine memory or file-cache use.
- The six full score rows take 3,646,464 bytes. The routing files are 20,275,
  20,256 and 20,278 bytes.

**Reference top-1 token IDs, in generation order:**

```text
short:  [785, 6722, 315, 9625, 374, 12095, 13, 151645]
medium: [16, 13, 10783, 95953, 8056, 13, 2303, 17, 13, 1032, 6983, 20638,
         68671, 311, 2441, 25144, 13, 151645]
long:   [28715, 389, 279, 2390, 8388, 11, 279, 2326, 8426, 2268, 8773, 11181,
         1573, 8568, 279, 3942, 6832, 4126, 525, 1447, 44364, 14374, 220, 16]
```

- Short ends with EOS at row 7.
- Medium reaches EOS at row 17, so its last full row is `medium-17.f32`.
  Normal stopping was kept.
- Long stops at the 24-token cap, with its last full row at 23.
- The saved continuations are the model's own output, including any content
  errors. They have not been edited.

**Artifact checks that passed:**

- Identical source prompts and token IDs.
- Finite full scores.
- Full-row top-1 and top-20 agree with the JSON.
- At both saved positions in every case, all 48 layers have eight unique,
  in-range expert indices with positive normalized weights.

These check the files are consistent. They do not test Vade.

</details>

**Where next:** [BF16 fixtures](../qwen3-30b-a3b/README.md) ·
[Validation](../../docs/validation.md)
