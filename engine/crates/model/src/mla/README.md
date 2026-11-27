# Host latent attention reference

[`mla.rs`](../mla.rs) provides a scalar f32 reference for multi-head latent
attention. [`tests.rs`](tests.rs) checks it. This module does not change model
admission, serving, sampling, or drafting.

## Inputs and layout

`Config::from_json` reads dimensions from a text config or its `text_config`
object. It does not read or branch on a model name. It checks positive ranks,
the head split, arithmetic size bounds, and the RMS norm epsilon. The RoPE
width can be zero. The Q path requires a positive low-rank dimension.

`Reference::new` checks the lengths and finite values of borrowed f32 weights.
The caller must decode checkpoint storage into these weights. Each matrix
stores output rows, then input columns. The shape names below are config
fields: `H` is `hidden_size`, `A` is `num_attention_heads`, `Q` is
`q_lora_rank`, `L` is `kv_lora_rank`, `N` is `qk_nope_head_dim`, `R` is
`qk_rope_head_dim`, and `V` is `v_head_dim`.

| Weight | Shape |
| --- | --- |
| `q_a_proj` | `[Q, H]` |
| `q_a_layernorm` | `[Q]` |
| `q_b_proj` | `[A * (N + R), Q]` |
| `kv_a_proj_with_mqa` | `[L + R, H]` |
| `kv_a_layernorm` | `[L]` |
| `kv_b_proj` | `[A * (N + V), L]` |

The Q path applies `q_a_proj`, weighted RMS norm, then `q_b_proj`. Each Q
head has a non-rotary part followed by a rotary part. The KV A projection
produces a compressed latent followed by a shared rotary key. Only the latent
passes through the KV RMS norm. The KV B projection produces a non-rotary key
followed by a value for each head.

`Rope` supplies cosine and sine values for the current token position. It
rotates adjacent pairs in the Q rotary part and in the shared rotary key.
The outputs keep adjacent pair order. A checkpoint adapter must map its
weight layout to this order. The caller owns position and frequency rules.
This module does not implement a checkpoint's RoPE scaling policy.

## Cache and attention

`LatentCache` stores only the normalized latent and the rotated shared key
for each token. It holds one sequence for one layer. Keep the layer weights
fixed while the cache is in use. `push` checks the token before it writes.
`step` projects a token, appends it, and attends to the full prefix including
that token. It restores the prior cache contents if attention returns an
error. Reserved capacity can grow after such an error.

`attend` expands keys and values on read. It uses dense causal attention.
The caller must give it a prefix with no future tokens. It returns the
concatenated value heads before the output projection. Sparse key selection,
attention bias, output projection, quantized cache storage, and cache
truncation are outside this reference.

Each projection and score reduction visits dimensions in ascending order.
RMS norm uses the ascending sum of squares, divided by its rank, then adds
epsilon. It multiplies each value by the reciprocal square root, then by the
norm weight. Scores multiply the Q/K dot product by the reciprocal square
root of the total Q head width. Softmax subtracts the maximum and sums the
exponentials in token order. Each output sums normalized probability times
value in token order. The code uses separate f32 products and sums.

The reference expands KV before each score or value reduction. It does not
absorb the KV B matrix into Q or the output. That change would regroup the
arithmetic and could change f32 bits. Host `sqrt` and `exp` results do not
establish bit equality on a GPU or another host target.

## Proof and public config

The unit tests compare cache outputs with `f32::to_bits`. The independent
test path expands K and V when each token enters its cache. The tests check
each prefix for prompt lengths and decode counts specified in
`compare_caches` in [`tests.rs`](tests.rs). They cover zero and nonzero RoPE
widths. Separate hand vectors check projection, norm, head split, rotation,
and attention. The full cache exists only in test code.

The [pinned public GLM config](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/f5a0baba30120da613e27973ee3535a725d4ece3/config.json)
gives `L=512`, `A=64`, `N=256`, `R=0`, and `V=256`. With f32 storage,
`cache_bytes_per_token` gives `(512 + 0) * 4 = 2,048` latent bytes and
`64 * (256 + 0 + 256) * 4 = 131,072` full KV bytes. These are payload bytes
per token per MLA layer. They exclude allocation overhead, sparse indexer
state, and linear attention state. The public config byte test checks these
values. This calculation is not a measured device allocation.

The [primary GLM implementation](https://github.com/huggingface/transformers/blob/e03c353939b8396e1e29d92e4d11245dca493dd1/src/transformers/models/glm5_next/modeling_glm5_next.py#L994-L1113)
specifies the projection, RMS norm, and latent cache stages. The public
config uses no rotary dimensions in MLA. The nonzero rotary tests exercise
the explicit adjacent-pair contract above. They do not prove a checkpoint
adapter for another rotary layout.

## GPU work left

A GPU path needs a config and tensor plan, storage decoding, deterministic
projections and norms, cache ownership, position handling, KV expansion, and
fixed attention reductions. It also needs the model's sparse indexer and
selection rules before a GLM inference claim. Test cache lifecycle, causal
masks, prefill, decode, chunks, batches, and drafting state. Then run the
device exactness and quality gates. This host reference provides no device
or speed result.
