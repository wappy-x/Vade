# KDA host reference

[`kda.rs`](../kda.rs) defines a fixed-order f32 host reference for Kimi Delta
Attention. It does not admit a checkpoint, load weights, or change serving.
The caller supplies decoded weights and config data. The current GDN code
and its tests are unchanged.

## Public source

The operation source is the locally saved public Transformers file
`src/transformers/models/glm5_next/modeling_glm5_next.py`.
The saved file has SHA256
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
This hash identifies the inspected bytes. The related
[host operation notes](../host_ops/README.md#local-public-evidence) record the
same hash and the commit cited by the code.
No receipt proves that these bytes match that commit or a release tag.

The source sections define these operations:

| Source section | Host operation |
| --- | --- |
| `Glm5NextTextForgetGate`, lines 339-369 | Low-rank forget projection and channel decay |
| `Glm5NextTextRMSNormGated`, lines 373-395 | RMS norm, norm weight, then sigmoid gate |
| `causal_conv1d_update` and `causal_conv1d_fn`, lines 405-445 | Causal depthwise convolution |
| `l2norm`, lines 448-459 | Q/K normalization by division |
| `recurrent_kimi_delta_attention`, lines 462-513 | Ordered delta-rule recurrence |
| `Glm5NextTextLinearAttention`, lines 621-770 | Projections, gates, and output projection |

The local public [`flash/refcpu.rs`](../flash/refcpu.rs) defines the SiLU
formula used for the activation named by the source and config.

The public source combines the Q/K/V channels for its depthwise convolution.
Each channel has its own weights and history. The host reference stores the
Q, K, and V channel blocks separately. This follows from the independent
depthwise channel operations. The source does not verify the checkpoint's
named split-convolution row mapping. This module does not provide that loader.

## Config and weights

`Config::from_json` accepts a text config or its `text_config` envelope.
The public [checkpoint config](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/f5a0baba30120da613e27973ee3535a725d4ece3/config.json)
supplies nested `linear_attn_config` fields: `num_heads`, `head_dim`,
`short_conv_kernel_size`, and `gate_lower_bound`.
The parent fields are `hidden_size`, `rms_norm_eps`, and `hidden_act`.
The saved config has SHA256
`dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8`.

The parser also accepts the explicit flat fields read by the operation
source: `linear_num_heads`, `linear_head_dim`, `linear_conv_kernel_dim`,
and `linear_lower_bound`. It rejects a mix of both forms. It supplies no
default dimensions. The local sources do not define the framework's config
conversion. Parsing these fields is not a claim about that conversion.

Only the source's SiLU convolution activation is implemented. Other
activation names return an error. A null lower bound selects softplus.
A finite negative lower bound selects bounded sigmoid decay.

Let `H` be the head count, `D` the head width, `C` the hidden width, `T` the
convolution width, and `W = H * D`. These symbols come from `Config`.
All matrices are row-major. All weights are bias-free except `dt_bias`.

| Weight | Shape |
| --- | --- |
| `q_proj`, `k_proj`, `v_proj` | `[W, C]` |
| `q_conv1d`, `k_conv1d`, `v_conv1d` | `[W, T]`, oldest tap first |
| `b_proj` | `[H, C]` |
| `f_a_proj`, `g_a_proj` | `[D, C]` |
| `f_b_proj`, `g_b_proj` | `[W, D]` |
| `dt_bias`, `a_log` | `[W]`, `[H]` |
| `o_norm`, `o_proj` | `[D]`, `[C, W]` |

## Arithmetic contract

All dot products visit the input dimension in ascending order. The host uses
separate f32 products and sums. It does not use fused multiply-add.
The source's L2 norm divides each Q/K value by
`sqrt(sum(x*x) + 1e-6)`. Q then multiplies by `1 / sqrt(D)`.
The epsilon in this formula is an operation constant from `l2norm`.

`beta = sigmoid(b_proj(x))` is scalar per head.
The forget projection is `f_b_proj(f_a_proj(x))`, with no activation between
the projections. Add `dt_bias` per key channel to get `z`.
Let `rate = exp(a_log[head])`. The log decay is
`lower_bound * sigmoid(rate * z)` when the bound is set.
Otherwise, it is `-rate * softplus(z)`. The source's softplus uses `z`
when `z > 20`, and `ln(1 + exp(z))` otherwise.

State has shape `[H, D, D]`, in head, key, value order. For each token and
head, the reference performs these operations in order:

1. Multiply each key row of state by `exp(log_decay[key])`.
2. Compute `memory[value] = sum_key(state[key,value] * k[key])`.
3. Compute `delta[value] = (v[value] - memory[value]) * beta`.
4. Add `k[key] * delta[value]` to each state cell.
5. Compute `core[value] = sum_key(state[key,value] * q[key])`.

The output gate is `g_b_proj(g_a_proj(x))`, with no activation between them.
Apply per-head RMS norm to core, multiply by `o_norm`, then multiply by the
sigmoid output gate. Apply `o_proj` last.

## Chunk and state contract

`step` accepts a token. `scan` accepts a flat token-major chunk. Both call
the same token implementation. An empty chunk leaves all state bits intact.
Neither API regroups arithmetic at a chunk boundary. The public source's
parallel chunk formula regroups sums. This host reference uses its recurrent
equations for both APIs to meet the bitwise chunk contract.

State retains the previous `T - 1` raw projection rows for each convolution.
The remaining state is the recurrent matrix. Clone state to save a prefix.
Use the same layer weights for its lifetime. Input, shape, and detected
arithmetic errors leave the caller's state unchanged.

[`tests.rs`](tests.rs) checks hand vectors and bitwise chunk continuations.
The host math library supplies exp, log, and sqrt. These tests do not prove
cross-target bits, GPU exactness, token IDs, or speed.
