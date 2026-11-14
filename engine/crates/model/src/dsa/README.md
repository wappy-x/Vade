# Host DSA indexer reference

[`dsa.rs`](../dsa.rs) supplies a host f32 reference for a sparse attention
indexer with compressed key pools. It does not enable a model or change the
serving path. It has no dependency on the separate host MLA reference.

## Public operation sources

The operation source is the local public Transformers file
`src/transformers/models/glm5_next/modeling_glm5_next.py`.
Its `Glm5NextTextIndexer` class defines projection, compression and selection.
The inspected file has SHA256
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
This hash identifies the inspected bytes. It does not assert a Git revision.
The source copy and fetch receipt are not included in this repository.

That class constructs `LayerNorm` with epsilon `1e-6`.
The public ONNX Runtime file
`onnxruntime/quantization/fusions/fusion_layernorm.py` defines the mean,
centered variance, epsilon, gain and bias graph.
The inspected local file has SHA256
`5347c96082498172da1d61742ec120746b3fd7d8a9b27ff397cd1ba9b0806784`.
No public source file or weight payload is copied into this module.

## Inputs and tensor shapes

`Config::from_json` accepts a text config or a root with `text_config`.
It reads every dimension and the selection budget from data.
It does not read a model name. An explicit false `index_kpool_compress`
is rejected. The source defines compressed pools only.

Let `C` be `hidden_size`, `Q` be `q_lora_rank`, `H` be `index_n_heads`,
`D` be `index_head_dim`, and `P` be `index_kpool`.
The shapes below follow the source class constructor.
All matrix weights use row-major order.

| Weight | Logical shape |
| --- | --- |
| `wq_b` | `[H * D, Q]` |
| `wk` | `[D, C]` |
| `weights_proj` | `[H, C]` |
| `k_norm_weight`, `k_norm_bias` | `[D]` each |
| `index_kpool_compress_gate` | `[D, C]` |
| `index_kpool_compress_ape` | `[P, D]` |

These are logical shapes, not verified checkpoint storage shapes.
The caller supplies decoded f32 weights, hidden rows and normalized Q latents.
The Q latent is the output of the caller's `q_a_layernorm(q_a_proj(x))`.
The source applies no rotary transform in this indexer.
Its stored `qk_rope_head_dim` does not change this operation.

## Fixed host arithmetic

Projection dots visit input dimensions in ascending order.
K norm subtracts the mean and uses the population variance.
It sums centered squares in channel order, divides by `D`, and adds epsilon.
It computes the reciprocal square root. It multiplies each centered value by
that inverse, then by the gain. It adds the bias last.
The reciprocal multiplication is this reference's fixed f32 order.
It is not a claim about Torch kernel rounding.

Pools start at the first valid raw cache slot. Each pool has contiguous slots.
A hole invalidates its pool. The reference does not compact valid tokens.
Each key channel has its own softmax across the pool slots.
Its logit is the projected compression gate plus the slot's APE value.
Softmax subtracts the maximum, computes `exp`, sums, and divides in slot order.
The weighted key sum also follows slot order.

Each head score is `relu(dot(query_head, pool_key) * (1 / sqrt(D)))`.
Each head weight is `weights_proj(x)[head] * (1 / sqrt(H))`.
Multiply the head score by its head weight, then sum in head order.
Head weights can be negative. ReLU precedes that multiplication.
These formulas follow `Glm5NextTextIndexer.forward` in the named source.

Products and sums are separate f32 operations. No reduction depends on chunk
size. The host math library supplies `sqrt` and `exp`.
The reference does not claim equal bits across host targets or on a GPU.

## Selection and cache

A complete pool is a candidate only when its final token is visible.
The pool budget is `index_topk / P`, rounded down as in the source.
Sort by descending score. Ties, including equal signed zeros, use the lower
first token index. This tie rule is the Vade contract; Torch leaves ties open.
Expand each selected pool in slot order. Tail mode appends the source's
current incomplete tail. Invalid queries select no tokens.

`select` returns valid raw indices. It omits the source's invalid padding
entries. Pool score order is retained. Treat the result as a set when forming
the sparse attention mask. With internal padding holes, the source's tail
formula can repeat a selected index. Its `build_attention_mask_from_topk`
method combines such entries into a set. The host list preserves that behavior.

Each `Cache` belongs to one sequence and one layer with fixed weights.
It stores projected keys, gate logits and validity. `run_chunk` appends all
rows before it selects each row at its causal position. An error restores
the prior cache. The caller owns batch separation and cache lifetime.

## Checks and work left

[`tests.rs`](tests.rs) contains hand vectors for projections, biased norm,
gate and APE compression, signed head scores and exact selection.
It compares complete prompt runs with chunked runs across pool boundaries.
It compares selected indices and compressed key bits. It also covers padding,
holes, whole-pool budgets, invalid inputs and cache rollback.
The existing QSA sources and tests are unchanged.

GPU work still needs tensor admission, storage decoding, fixed-order kernels,
pool cache ownership, sparse MLA masks and drafting state rules.
Cross-layer index sharing remains outside this reference.
Device reductions, storage rounding and transcendental math need comparison
with these vectors. The full device exactness gate must pass before serving
integration. Host checks do not establish token IDs or speed.
