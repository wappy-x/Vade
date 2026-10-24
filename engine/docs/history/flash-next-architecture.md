# Qwen3.8-Flash-Next-FP8 text architecture

> **Historical specification.** Written on 2 October 2026 with the first
> Flash Next research-engine work (main commit `efee11b`). It specifies the
> official checkpoint's text path as the installed Hugging Face Transformers
> code implemented it then. It is not a status report for Vade.
>
> Flash Next runs only in a separate two-Spark research engine. Neither
> `spark-serve` nor `tp-serve` serves it, and it is distinct from the
> supported Qwen3.8-27B hybrid. Current status is in
> [Models](../models.md#qwen38-flash-next-fp8). The reference data is described
> in the [fixture guide](../../fixtures/qwen3.8-flash-next-fp8/README.md).

**What this page covers**

- The official model's text path, layer by layer, with shapes and the
  reference source for each step.
- How its FP8 weights are stored and converted, and how much memory its
  decode state needs.
- The original CPU reference run for the short test case, under an older
  weight-conversion policy.

## Terms on this page

<dl>
<dt>FP8 (E4M3), BF16, FP32</dt>
<dd>8-bit, 16-bit and 32-bit number formats. E4M3 is the FP8 layout used
here.</dd>
<dt>Dequantization</dt>
<dd>Turning stored FP8 values back into a wider format, using their scales.</dd>
<dt>Residual stream</dt>
<dd>The running vector that each layer adds its result to.</dd>
<dt>Hyper-connection (HC)</dt>
<dd>A learned mix that keeps four residual streams and decides how much each
block adds to each stream.</dd>
<dt>GatedDeltaNet (GDN)</dt>
<dd>A linear-attention layer that keeps a fixed-size recurrent matrix instead of
a growing history.</dd>
<dt>Indexed attention, QSA</dt>
<dd>Softmax attention limited to the token blocks that a small indexer
selects.</dd>
<dt>MoE</dt>
<dd>Mixture of experts: a router picks a few expert networks for each
token.</dd>
<dt>PLE, n-gram embedding</dt>
<dd>An extra embedding of hashed pairs and triples of recent tokens, added
before decoder 1's attention.</dd>
<dt>RoPE</dt>
<dd>Rotary position embedding: rotates query and key coordinates by token
position.</dd>
<dt>RMSNorm</dt>
<dd>Root-mean-square normalization of a vector.</dd>
<dt>GQA</dt>
<dd>Grouped-query attention: several query heads share one key/value head.</dd>
<dt>MTP</dt>
<dd>A multi-token prediction head, for guessing tokens ahead.</dd>
<dt>Oracle</dt>
<dd>The CPU reference run that produces expected outputs.</dd>
<dt><code>B</code>, <code>T</code>, <code>S</code></dt>
<dd>Batch size, current query length, and total cached length.</dd>
</dl>

## The model in one minute

<dl>
<dt>Layers</dt>
<dd>48 decoder layers: 36 GatedDeltaNet layers and 12 indexed-attention layers,
in a repeating pattern of three GDN layers then one indexed layer.</dd>
<dt>Width</dt>
<dd>Hidden width 2,560, carried as four residual streams: 10,240 values per
token.</dd>
<dt>Experts</dt>
<dd>Every layer has a routed MoE with 512 experts, 10 chosen per token, plus one
always-active shared expert.</dd>
<dt>Extra embedding</dt>
<dd>Decoder 1 also adds the n-gram (PLE) embedding.</dd>
<dt>Vocabulary</dt>
<dd>248,320 tokens, with separate input-embedding and output-head weights.</dd>
<dt>Storage</dt>
<dd>Routed experts and the n-gram table are stored as FP8; most other parts are
BF16.</dd>
<dt>MTP</dt>
<dd>The checkpoint contains an MTP head, but the installed reference code does
not run it.</dd>
</dl>

## Source references and scope

This page specifies the official checkpoint's text path as implemented by
the installed Hugging Face Transformers **5.19.0.dev0**, source commit
`9d8c3a24d434d4a958a23562f576126924358b7f` (a clean source checkout).

- The checkpoint's `config.json` says `transformers_version: 5.8.0.dev0`. That
  is its save-time version, not the version that ran this reference.
- The installed `Qwen4ExpTextConfig`, `Qwen4ExpTextDecoderLayer` and
  `Qwen4ExpForCausalLM` classes exist.
- The full 48-layer causal model was built on PyTorch's `meta` device without
  loading weights: 1,164 parameter tensors, 176,943,899,520 elements, all on
  `meta`.

Paths below are relative to the Transformers checkout at that commit. Line
numbers refer to that revision; class and function names are the primary
anchors.

<dl>
<dt>M</dt>
<dd><code>src/transformers/models/qwen4_exp/modeling_qwen4_exp.py</code></dd>
<dt>C</dt>
<dd><code>src/transformers/models/qwen4_exp/configuration_qwen4_exp.py</code></dd>
<dt>F</dt>
<dd><code>src/transformers/integrations/finegrained_fp8.py</code></dd>
<dt>Q</dt>
<dd><code>src/transformers/quantizers/quantizer_finegrained_fp8.py</code></dd>
<dt>K</dt>
<dd><code>src/transformers/cache_utils.py</code></dd>
<dt>W</dt>
<dd><code>src/transformers/conversion_mapping.py</code></dd>
<dt>L</dt>
<dd><code>src/transformers/core_model_loading.py</code></dd>
</dl>

- M is generated from `src/transformers/models/qwen4_exp/modular_qwen4_exp.py`.
- Checkpoint facts below come from its official `config.json`,
  `model.safetensors.index.json`, and complete safetensors headers.
- No vision tower is needed for a text-only call.
- The outer checkpoint architecture is `Qwen4ExpForConditionalGeneration`; its
  text weights use `model.language_model.*`. The text-only
  `Qwen4ExpForCausalLM` implements the same decoder and output projection
  under `model.*`. M: `Qwen4ExpModel`, `Qwen4ExpForConditionalGeneration`,
  `Qwen4ExpForCausalLM`.

## Dimensions and exact layer sequence

| Quantity | Value |
| --- | --- |
| Hidden width `D` | 2560 |
| Residual streams `H` | 4 |
| Residual width `HD` | 10240 |
| Vocabulary `V` | 248320 |
| Decoder layers | 48 |
| RMS epsilon | `1e-6` |
| MLP activation | SiLU |

- All linear projections are bias-free unless stated otherwise.
- The text config's BOS and EOS are both token 248044. This EOS defines PLE
  hash-history boundaries.
- Separately, `generation_config.json` specifies EOS IDs `[248046,248044]`
  and padding ID 248044.
- Embeddings and output weights are untied.

C: `Qwen4ExpTextConfig`; checkpoint `text_config` and `generation_config.json`.

The saved sequence is exactly `(linear_attention, linear_attention,
linear_attention, full_attention)` repeated 12 times. **The installed config
rewrites every `full_attention` entry to `indexed_attention`.** These layers
have a QSA indexer; treating them as unrestricted full attention at long
context is incorrect. C: `Qwen4ExpTextConfig.__post_init__`, lines 166 to 187.

Layers are numbered from zero. The only additional operation is the PLE
embedding, which precedes layer 1's attention block.

| Linear layers | Indexed layer |
| --- | ---: |
| 0, 1, 2 | 3 |
| 4, 5, 6 | 7 |
| 8, 9, 10 | 11 |
| 12, 13, 14 | 15 |
| 16, 17, 18 | 19 |
| 20, 21, 22 | 23 |
| 24, 25, 26 | 27 |
| 28, 29, 30 | 31 |
| 32, 33, 34 | 35 |
| 36, 37, 38 | 39 |
| 40, 41, 42 | 43 |
| 44, 45, 46 | 47 |

- Every layer contains a routed-plus-shared MoE and separate attention and
  MLP hyper-connections.
- There are 36 GatedDeltaNet and 12 indexed-attention blocks.
- `ple_layer_ids=[2]` uses **one-based** numbering, hence decoder index 1.

M: `Qwen4ExpTextDecoderLayer.__init__`, lines 1254 to 1267.

## Residual streams and layer order

Token lookup gives `[B,T,2560]`; the model repeats that vector four times to
`[B,T,10240]`. Each decoder then runs:

1. On decoder 1 only, add PLE's `[B,T,10240]` output to the residual streams.
2. The attention hyper-connection mixes the four streams down to
   `[B,T,2560]` and computes four injection gates. Run GatedDeltaNet or
   indexed attention. Multiply its output by each injection gate and add it
   to the original streams.
3. The MLP hyper-connection repeats the mix-and-inject process around the MoE.

There are no conventional `input_layernorm` / `post_attention_layernorm`
residual blocks. The normalization is inside each learned hyper-connection.
M: `Qwen4ExpTextModel.forward`, lines 1465 to 1489;
`Qwen4ExpTextDecoderLayer.forward`, lines 1270 to 1306.

For residual vector `x` and learned weights, the hyper-connection is:

```text
n = group_rmsnorm(x, group_size=2560)           # four independent groups
a = sigmoid(W_up * silu((W_down * n) / 4))     # 10240 -> 320 -> 10240
mixed = mean_over_4_streams(a * n)             # [B,T,2560]
inject = 2 * sigmoid((W_inject * n) / 4)       # [B,T,4]
updated_stream[h] = x_stream[h] + inject[h] * block(mixed)
```

- `Qwen4ExpTextRMSNorm` computes normalization and multiplication in FP32,
  using `(1 + weight)`, then casts to the input dtype.
- Grouped norms have one learned weight per feature, not a single weight
  shared by all four groups.

M: `Qwen4ExpTextRMSNorm`, lines 148 to 172; `Qwen4ExpTextGatedResidual`, lines
999 to 1027.

## Linear attention: GatedDeltaNet

This is a gated delta-rule recurrent matrix state. It is not softmax attention
and not a scalar-state Mamba recurrence. M: `Qwen4ExpTextGatedDeltaNet`,
lines 461 to 621.

<dl>
<dt><code>in_proj_qkv</code></dt>
<dd>2560 to 10240, which is 2048 Q + 2048 K + 6144 V.</dd>
<dt>Depthwise <code>conv1d.weight</code></dt>
<dd><code>[10240,1,4]</code>, no bias, SiLU after convolution.</dd>
<dt>Q and K before repetition</dt>
<dd><code>[B,T,16,128]</code> each.</dd>
<dt>V and <code>in_proj_z</code> output</dt>
<dd><code>[B,T,48,128]</code> each.</dd>
<dt><code>in_proj_b</code>, <code>in_proj_a</code></dt>
<dd>2560 to 48 each.</dd>
<dt><code>dt_bias</code>, <code>A_log</code></dt>
<dd><code>[48]</code> each.</dd>
<dt>Recurrent state</dt>
<dd><code>[B,48,128,128]</code>, FP32.</dd>
<dt>Output</dt>
<dd>48 × 128 = 6144, to 2560.</dd>
</dl>

Step by step:

- Convolution is causal over the **unconvolved** QKV projection.
- Each Q/K head is repeated three times, giving 48 heads aligned with V.
- Q and K are normalized by `x * rsqrt(sum(x*x)+1e-6)`; then Q alone is
  scaled by `1/sqrt(128)`.
- `beta=sigmoid(in_proj_b(x))`;
  `g=-exp(A_log.float()) * softplus(in_proj_a(x).float()+dt_bias)`.

For each head and token, with K column vector `k`, V vector `v`, and the
normalized, scaled Q vector `q`, the recurrence is:

```text
S_decay = exp(g) * S_previous
delta = beta * (v - transpose(k) * S_decay)
S = S_decay + outer(k, delta)                  # [128,128], key axis first
y = transpose(q) * S                          # [128]
```

Two official code paths:

- The cached single-token path runs `torch_recurrent_gated_delta_rule`.
- Prefill and multi-token calls use `torch_chunk_gated_delta_rule` with chunk
  size 64, zero padding to a chunk boundary, FP32 triangular solves and the
  mathematically equivalent scan.
- Their reduction orders differ, so a reference should keep the official
  dispatch rather than use one path for both.
- The returned recurrent state is FP32 even for BF16 activations.

M: `l2norm`, `torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule`, lines 252 to 457.

After the recurrence:

- Each 128-wide V head gets RMS normalization with a direct learned `weight`
  (not `1+weight`).
- It is then multiplied by `sigmoid(z)`, because this checkpoint sets
  `output_gate_type="sigmoid"`.
- The 6144 features pass through `out_proj`.

M: `Qwen4ExpTextRMSNormGated`, lines 176 to 192;
`Qwen4ExpTextGatedDeltaNet.forward`.

Decode cache per linear layer:

- The last four raw QKV projections, `conv_states[0]: [B,10240,4]`.
- The recurrent matrix, `recurrent_states[0]: [B,48,128,128]`.
- These are fixed-size rolling states, not an append-only history of K and V.
  The current token's projections update the convolution and then the
  recurrent matrix.

M: `causal_conv1d_update`, `causal_conv1d_fn`; K:
`LinearAttentionLayer.update_conv_state`, `update_recurrent_state`, lines
1025 to 1113.

## Indexed attention and QSA

M: `Qwen4ExpTextAttention`, lines 815 to 897; `Qwen4ExpTextQSAIndexer`, lines
669 to 775; `eager_attention_forward`, lines 790 to 812.

Main attention, shapes:

- 24 Q heads, 2 KV heads, head width 256, GQA factor 12.
- `q_proj` maps 2560 to 12288, reshaped to `[B,T,24,512]` and split **within
  each head** into a 256-wide Q and a 256-wide output gate. This is **not** a
  split of the whole flat projection into two contiguous 6144-wide vectors.
- `k_proj` and `v_proj` each map 2560 to 512.

Main attention, computation:

- Q and K get separate 256-wide `1+weight` RMSNorm and RoPE. V is neither
  normalized nor rotated.
- Attention logits are scaled by `1/sqrt(256)=1/16`, masked causally and by
  QSA, and softmaxed in FP32.
- After the weighted V sum, flatten the 24 heads, multiply by the
  per-feature `sigmoid(gate)`, and project 6144 to 2560 through `o_proj`.
- Dropout is zero for this checkpoint.

QSA is a selection mask, not compressed V attention:

1. `index_qk_proj` maps 2560 to 640: four index Q heads of width 128 and one raw
   K head of width 128. RMS-normalize the index Q and apply the same
   64-dimensional partial RoPE used by main attention.
2. List the visible causal token indices for each query. Divide the visible
   sequence into complete groups of four; keep a tail of zero to three tokens.
3. Average each group's **raw** index K vectors in FP32, cast to the raw-key
   dtype, RMS-normalize, and apply RoPE at the group's **first token**
   position.
4. Dot each pooled key with the four index queries in FP32. Apply ReLU to
   each dot, sum the four results and divide by `sqrt(128)`.
5. Select up to `2048/4=512` complete groups with the highest scores, expand
   them to their original token indices, and always include the incomplete
   tail. Intersect this selection with the original causal and padding mask.

What this means:

- The effective maximum is 2048 selected complete-block tokens plus up to
  three tail tokens.
- SHORT (25 prompt tokens, 8 generated) selects every visible token, but the
  official indexer still runs.

What is cached:

- Main attention caches all previous normalized, rotated K and unmodified V,
  each `[B,2,S,256]`.
- The indexer separately caches every **unnormalized, unrotated** raw index K,
  `[B,S,128]`.
- No Q cache is needed, and QSA does not bound KV memory.
- The official code pools raw keys again for each query. A different pooling
  cache would need its own correctness argument.

K: `DynamicIndexedLayer.update_indexer`, lines 331 to 365; M:
`Qwen4ExpTextQSAIndexer.forward` and `Qwen4ExpTextAttention.forward`.

## Rotary positions

- RoPE type is `default`, base (theta) `10000000`, partial factor `0.25`, main
  head width 256. So exactly the first **64** coordinates rotate and 192 pass
  unchanged.
- The 32 inverse frequencies are `theta^(-2i/64)`, `i=0..31`.
- `rotate_half` pairs the first and second halves of these 64 coordinates,
  not adjacent even and odd coordinates.
- QSA rotates the first 64 of each 128-wide index head.
- No RoPE is applied inside GatedDeltaNet.

M: `Qwen4ExpTextRotaryEmbedding`, lines 77 to 145; `rotate_half`,
`apply_rotary_pos_emb`, lines 624 to 666.

The three position axes use `mrope_section=[11,11,10]`:

- H replaces frequency indices `1,4,...,31`, W replaces `2,5,...,29`, and T
  supplies the remaining indices.
- The resulting 32 frequencies are duplicated to width 64.
- Pure text uses the same steadily increasing position on all three axes,
  which makes this ordinary partial RoPE.
- Sin and cos are computed in FP32 and cast to the activation dtype.

Positions:

- The model keeps full three-axis `position_ids: [3,B,S]` in its cache so QSA
  can rotate pooled historical blocks. Text-only code can derive them from
  positions.
- `max_position_embeddings=262144`. This default RoPE has no configured
  dynamic scaling.

M: `Qwen4ExpTextRotaryEmbedding.recomposition_frequencies` and
`Qwen4ExpTextModel.forward`, lines 1423 to 1457.

## N-gram embedding and PLE (decoder 1 only)

M: `Qwen4ExpTextNGramEmbedding`, lines 1076 to 1176; `Qwen4ExpTextPLELayer`,
lines 1179 to 1251.

Hashing:

- Use the current token and up to two previous token IDs.
- Eight heads hash bigrams and eight hash trigrams: 16 heads, each returning
  160 values, concatenated to 2560.
- For n-gram order `n`, the hash is XOR over
  `token[t-j] * layer_multipliers[j]`, `j=0..n-1`. Reduce this integer modulo
  each head's prime vocabulary size, then add the head's cumulative offset.
- Missing history is EOS 248044. A token after EOS starts a new segment;
  earlier context across that EOS is replaced by EOS padding. EOS itself can
  see earlier tokens in its segment. `_shift_right_ignore_eos` defines this
  boundary behaviour exactly.

Stored constants:

- The checkpoint stores `layer_multipliers`, `ngram_heads_vocab_sizes` and
  `ngram_heads_offsets` as I64 buffers. **Load those buffers.**
- Config defaults would generate multipliers from seed 1234 and PLE ordinal 0
  through `_splitmix64` and `_build_layer_multipliers`, not a platform hash.
- Head vocabulary sizes are the first 16 primes at or above 20,000,000:

```text
20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
```

- The stored multipliers are `[23703573157769,20109073645365,8052911324071]`.
- The stored prime sizes match the list above; the offsets are their
  exclusive prefix sums.
- Integer hash arithmetic uses signed I64 multiplication and bitwise XOR,
  followed by a nonnegative remainder with each positive prime.

Table size:

- The primes sum to 320,001,446. Padding to a multiple of 128 gives a logical
  table `[320001536,160]`.
- The table alone holds **51,200,245,760 FP8 bytes**. As FP32 it would take
  **204,800,983,040 bytes**, so a streaming oracle must gather only the
  requested rows.
- Checkpoint `split_ngram_parts=128` means numeric-order row concatenation of
  `ngram_embedding.shard_0.weight` through `shard_127.weight`, each
  `[2500012,160]` = **400,001,920 FP8 bytes**.
- These are tensor pieces, not a one-to-one description of the 131 physical
  safetensors files.

W: `qwen4_exp_text` conversion, lines 1918 to 1928; L: `Concatenate`,
`dot_natural_key`.

PLE computation, from the concatenated lexical embedding `e` and the
four-stream input `x`:

```text
k = group_rmsnorm(key_proj(e))                # 2560 -> 10240 -> [4,2560]
q = group_rmsnorm(x)                          # [4,2560]
v = value_proj(e)                            # 2560 -> 2560, shared across streams
a[h] = dot(k[h], q[h]) / sqrt(2560)
a[h] = sign(a[h]) * sqrt(max(abs(a[h]), 1e-6))
u[h] = sigmoid(a[h]) * v
PLE = flatten(u) + silu(depthwise_conv(group_rmsnorm(flatten(u))))
```

- The convolution has 10240 channels, kernel width 4, dilation 3 and no bias.
  So its left context is nine tokens, and it includes the current input.
- Norms use `1+weight`.
- Padding masks zero both the direct gated values and the normalized
  convolution inputs.
- The decoder adds PLE before its attention hyper-connection.

Decode state:

- Decode keeps the normalized gated inputs in `conv_states[1]: [B,10240,9]`
  and token IDs in `conv_states[2]: [B,2]` (I64). Neither is interchangeable
  with GatedDeltaNet's `conv_states[0]`.
- C: `Qwen4ExpTextConfig.__post_init__` allocates three convolution-state
  slots when PLE is present; only the PLE layer fills slots 1 and 2.

## Routed experts, shared expert and normalization

Routing:

- Every layer has 512 experts and routes each token to 10.
- The input to the router and experts is the MLP hyper-connection's
  2560-wide mixed vector.
- The bias-free router weight is `[512,2560]`. Compute all 512 logits, softmax
  in FP32, take `torch.topk(...,10)`, then divide the selected probabilities
  by their sum.
- `norm_topk_prob` is absent from the saved config but defaults to **true**
  in C.
- There is no sigmoid router, expert-bias correction, group routing, or extra
  routed scaling factor in this installed class.

M: `Qwen4ExpTextTopKRouter`, lines 956 to 974.

Experts:

- Each expert computes `W_down(silu(W_gate*x) * (W_up*x))`, with
  intermediate width 640. Gate and up matrices are `[640,2560]`; down is
  `[2560,640]`.
- The official in-memory layout packs gate/up as `[512,1280,2560]` and down
  as `[512,2560,640]`. Checkpoint storage has separate projection tensors per
  expert.
- The eager implementation goes through active experts in expert-index
  order, multiplies each result by its selected routing probability, and
  accumulates by token using `index_add_`. Keep that order for the FP32 oracle.

M: `Qwen4ExpTextExperts`, lines 917 to 953; W: `qwen2_moe` conversions inherited
by `qwen3_5_moe_text` and `qwen4_exp_text`.

Shared expert:

- A separate, always-active shared expert has the same 2560 to 640 to 2560
  SiLU gated MLP.
- Its output is multiplied by `sigmoid(shared_expert_gate(x))`, where the gate
  is a bias-free 2560-to-1 projection.
- Add this to the routed sum. There is no later MoE normalization.
- The surrounding hyper-connection injects the sum into the four residual
  streams.
- Router auxiliary loss is for training only and does not change inference
  hidden states.

M: `Qwen4ExpTextMLP`, `Qwen4ExpTextSparseMoeBlock`, lines 900 to 997.

## Final output and MTP boundary

- After decoder 47, `hyper_connection_mixer` applies the same normalized
  low-rank mix as above with `use_combine=False`. It returns `[B,T,2560]`
  with no injection gates or residual addition.
- There is **no separate final RMSNorm**.
- The untied, bias-free `lm_head.weight: [248320,2560]` produces logits.
  `logits_to_keep` may restrict positions before this projection.

M: `Qwen4ExpTextModel`, `Qwen4ExpForCausalLM.forward`,
`Qwen4ExpForConditionalGeneration.forward`.

The MTP head:

- The checkpoint declares one MTP layer: `mtp.hybrid=true`,
  `mtp.layer_types=["full_attention"]`, `mtp.num_hidden_layers=1`,
  `mtp.rope_theta=10000000`, `mtp_use_dedicated_embeddings=false`, and
  `mtp_use_hidden_state_from_layer=null`.
- Its index contains 3,101 `mtp.*` tensors, including `fc_embedding`,
  `fc_hidden`, hyper-connection mixers, one indexed attention block and MoE
  expert projections.
- **The installed Qwen4Exp classes do not build or run this MTP head.**
  `Qwen4ExpPreTrainedModel` and the text-only causal class explicitly ignore
  unexpected `^mtp.*` checkpoint keys (M, lines 1318 and 1594). Their ordinary
  forward pass emits main-head logits only.
- The exact MTP forward order and training shift cannot be specified from
  these classes or the tensor names. Implementing MTP needs an authoritative,
  compatible implementation. It is outside this main-head greedy SHORT
  reference.

## FP8 storage and BF16 dequantization with FP32 promotion

The checkpoint config is `quant_method="fp8"`, dynamic activations,
`weight_block_size=[128,128]`, `weight_per_tensor=false`,
`act_per_tensor=false`. This does not mean every weight is FP8.

| Checkpoint on disk | Bytes |
| --- | ---: |
| Tensor payload, 131 files | 185,502,232,570 |
| Including headers | 185,523,317,458 |
| Smallest file | 877,982,872 |
| Largest file | 2,136,177,320 |
| FP8 payload | 174,512,783,360 |
| BF16 payload | 10,989,448,930 |
| I64 payload | 280 |

These are disk sizes of the 131 safetensors files, not a streaming process's
measured RSS.

| Stored decoder-layer payload | Bytes |
| --- | ---: |
| Linear layer without PLE | 2,671,683,008 |
| Indexed-attention layer | 2,658,658,816 |
| Linear layer 1 including PLE | 53,937,608,410 |

Full per-file and per-layer sizes are recorded in
`fixtures/qwen3.8-flash-next-fp8/short.json` under `memory`.

### Routed expert weights

Routed expert projection weights are `F8_E4M3` (PyTorch `float8_e4m3fn`).
Each gate/up projection has a BF16 `weight_scale_inv: [5,20]`; down uses
`[20,5]`. Despite the suffix, these stored values **multiply** the decoded
FP8 values:

```text
W_fp32[row,col] = float32(bfloat16(
    float32(W_fp8[row,col])
    * float32(weight_scale_inv[row//128,col//128])))
```

- F: `Fp8Dequantize._dequantize_one`, lines 1040 to 1090, first promotes weight
  and scale to FP32, derives block sizes from the weight and scale-grid
  shapes, broadcasts scales over blocks, then casts to the requested output
  dtype.
- The required expert path requests BF16 output, rounds the scaled weight to
  BF16, then promotes it to FP32 for multiplication and accumulation with FP32
  activations. This is weight dequantization, not activation quantization.
- The earlier oracle requested FP32 dequantization directly and skipped the
  required BF16 rounding. Its expert outputs cannot certify this path.
- This checkpoint uses BF16 scales, not UE8M0 exponent bytes.

One expert is three FP8 matrices plus their scales.

| Expert weight size | Bytes |
| --- | ---: |
| One expert, as stored | **4,915,800** |
| One expert, expanded to FP32 | **19,660,800** |
| One layer's 512 experts, stored | **2,516,889,600** |

Stream selected experts one at a time instead of expanding the full packed
expert parameter.

### The n-gram table

The n-gram table uses a separate scheme: one global BF16
`ngram_embedding.weight_scale: [1]`, with no 128×128 scale grid.

- F: `FP8Embedding.forward`, lines 344 to 365, gathers FP8 rows, casts them to
  the scale's dtype, then multiplies by the scale.
- For FP32 arithmetic, promote the scale to FP32 before calling this
  operation. It then computes `float32(row) * float32(scale)`. The
  checkpoint's default BF16 compute path rounds the product to BF16.
- `replace_with_fp8_embedding` keeps row-wise dequantization even when
  full-model `dequantize=True` is requested (F, lines 818 to 831).
- Do not expand all 51.2 GB of table storage to FP32, or treat its scalar as
  an inverse.

### Which modules are not FP8

`modules_to_convert=["ple.ple_embedding.ngram_embedding"]` explicitly opts
that embedding into FP8 replacement. It does not limit normal eligible linear
and expert conversion to that embedding. The saved `modules_to_not_convert`
list contains 943 names. Its exact families are:

Model-wide, and in every layer:

- `lm_head`, `model.language_model.embed_tokens`, and the final mixer's
  `input_mix_weight_down` / `input_mix_weight_up`.
- Every layer's attention and MLP hyper-connection `block_inject_weight`,
  `input_mix_weight_down`, `input_mix_weight_up`.
- Every layer's MoE `gate`, `shared_expert.{gate_proj,up_proj,down_proj}`, and
  `shared_expert_gate`.

By layer type:

- Every linear layer's `conv1d`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`,
  `in_proj_z`, `out_proj`.
- Every indexed layer's `self_attn.{q_proj,k_proj,v_proj,o_proj}` and
  `self_attn.indexer.index_qk_proj`.
- Decoder 1's PLE `conv1d`, `key_proj`, `value_proj`.

Vision and MTP:

- Vision's 27 blocks' `attn.{proj,qkv}`, `mlp.{linear_fc1,linear_fc2}`, plus
  `merger.{linear_fc1,linear_fc2}`, `patch_embed.proj`, `pos_embed`.
- MTP's `fc_embedding`, `fc_hidden`, final mixer down/up, and its single
  layer's hyper-connection, router/shared-expert, main attention and
  index-QK projection families listed above.

More notes:

- Norm weights, `A_log`, `dt_bias`, and the integer hash buffers are not
  ordinary eligible linear or expert modules, and are not made FP8 by that
  conversion.
- The token embedding and output matrix are stored BF16, each
  **1,271,398,400 bytes**, or **2,542,796,800 bytes** when expanded to FP32.
- Keep stored dtypes until their tensors are needed.

Q: `FineGrainedFP8HfQuantizer` `_normalize_modules_to_not_convert`,
`_process_model_before_weight_loading`, `update_weight_conversions`; F:
`replace_with_fp8_linear`, `FP8Experts`.

### What the CPU oracle does and does not cover

- It uses official BF16 expert dequantization, followed by FP32 weight
  promotion, activations and accumulation.
- It does not imitate the runtime's dynamically quantized FP8 activation
  matrix multiplies. It should not be described as bit-identical to that
  hardware path.
- Q: `FineGrainedFP8HfQuantizer.validate_environment` selects dequantization
  for a prequantized checkpoint without a GPU or XPU. F: `FP8Linear.forward`,
  `FP8Experts.forward` and `Fp8Dequantize` tell those execution paths apart.

## Decode memory contract (batch size one)

These are tensor payload bytes derived from the shapes above. They exclude
allocator overhead, weights, scratch, masks, logits and the temporary
convolution concatenation. They are not measurements of peak RSS. No cache
quantization is assumed.

Two columns, in bytes: *FP32* is the FP32 reference; *BF16* means BF16
activations with FP32 recurrent state.

| Persistent state | FP32 (bytes) | BF16 (bytes) |
| --- | ---: | ---: |
| Recurrent matrix, one layer | 3,145,728 | 3,145,728 |
| QKV conv history, one layer | 163,840 | 81,920 |
| PLE conv history, whole model | 368,640 | 184,320 |
| PLE two token IDs, whole model | 16 | 16 |
| Fixed total, 36 layers and PLE | 119,513,104 | 116,379,664 |
| One indexed layer, per token | 4,608 | 2,304 |
| 12 indexed layers, per token | 55,296 | 27,648 |
| Position history, per token | 24 | 24 |

- *One layer* means one linear-attention layer; *conv history* is the QKV
  or PLE convolution history.
- *Fixed total* covers all 36 linear layers plus PLE.
- An indexed layer stores K, V and the raw index K for each token.
- The position history is the full three-axis I64 history.

Other sizes:

- A decoder output snapshot has 10240 features per position: **40,960
  bytes** in FP32.
- The final mixed hidden state has 2560 features: **10,240 bytes**.
- A complete FP32 vocabulary logit row is **993,280 bytes**.

A streamed implementation can drop each layer's weights after running that
layer, while keeping its fixed recurrent state or growing attention state
for decoding. Routed expert weights, the n-gram table and the output head
need their own streaming and gather policies. "One layer at a time" alone
does not bound the PLE table allocation.

## Original SHORT reference (previous dequantization policy)

> These measurements describe the original run, which dequantized expert
> weights straight to FP32. The step-4 oracle now rounds dequantized expert
> weights to BF16 before FP32 promotion, and MLP acceptance requires short
> artifacts generated under that policy. These historical numbers are not
> acceptance evidence for the corrected policy.

How it ran:

- The streaming CPU oracle completed the 25-token prompt and eight greedy
  output tokens.
- It used the official decoder classes, eager attention and CPU delta-rule
  fallbacks, FP32 arithmetic, 12 Torch threads, and no dynamic activation
  quantization.
- Greedy argmax overrides the saved generation default `do_sample=true`.

The output:

- Top-1 IDs are `[760,6511,314,9338,369,11751,13,248046]`, decoding to
  `The capital of France is Paris.<|im_end|>`.

| Measurement | Value |
| --- | ---: |
| Generation time (s) | **128.462708** |
| Peak RSS (bytes) | **6,284,836,864** |
| Total process wall time (s) | **131.35** |
| Time to first logits (s) | **61.625** |
| Largest streamed module (bytes) | **2,542,796,800** |
| Linear-attention module (bytes) | **231,834,496** |
| Indexed-attention module (bytes) | **205,786,112** |
| Final cache (bytes) | **121,283,344** |

- Total process wall time was at the same peak RSS.
- The largest streamed module is the largest single FP32 weight module: the
  output head or the token embedding.
- Both attention-module rows are complete modules, in FP32 parameters.
- The final cache size is shape-derived.

What the numbers mean:

- These timings describe this CPU reference run, not engine speed.
- The final cache holds 32 consumed token positions: 25 prompt tokens plus
  seven generated tokens fed back. Its size is
  `119513104 + (55296 + 24) * 32`.
- Peak RSS also includes conversion buffers, mapped pages, allocator overhead
  and scratch. It is not just the sum of these persistent parameter and cache
  counts.

Artifacts:

- `fixtures/qwen3.8-flash-next-fp8/short.json`.
- Full FP32 logit rows `short-0.f32` and `short-7.f32`.
- Hidden-state snapshots `short-layers.json`.
- Router selections and weights `short-routing.json`.

The saved rows are the first and last generated-token decisions, at
zero-based input positions 24 and 31. MTP execution and the runtime's FP8
activation path remain unverified. The SHORT main-head reference is complete.
