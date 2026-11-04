# Deferred BF16 cast requirements

`LayerPlan::bf16_cast_contract` attaches dtype requirements to the existing
operation graph. It uses operation kinds, edges, and buffer shapes. It does
not dispatch on a model name. It does not change `reference::forward`,
`reference::replay`, Qwen arithmetic, or model admission.

This is a **conditional fallback profile**. Activations and ordinary
parameters must be BF16. Explicit FP32 exceptions still apply. The operation
description has no checked parameter-dtype inventory. Quantized Linear
replacements and cache classes are absent from the saved source. Thus this
profile is not a verified checkpoint dtype claim. Every operation has a
refusal reason. `Contract::require_complete` always returns an error.
Do not use this profile to allocate device state or admit a model.

## Source record

S is the saved `tf_modeling_glm5_next.py` execution file. Its SHA256 is
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
The fetch record dates it to November 1, 2026. It came from the unpinned
[Transformers source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py).
The hash identifies this audit's input. That moving link does not. S line
spans below refer to the saved file. Use its hash to identify a local copy.

P is primitive source revision `1a265298310c61a383fc6dd574abfbae731b0138`.
It is not in this tree's history. The table records sources at P.
This contract reads those references. It does not import kernels,
inherit device results, or qualify the combined image.

The primitive paths below are relative to `engine/crates/` at P.
The last column compares this tree with P. A shared path does not imply equal code.

| Key | Reference source at P | Current tree boundary |
| --- | --- | --- |
| H | `gpu/kernels/src/mhc_math.rs`: prepare, collapse, combine, head; `model/src/mhc.rs` | The math file is equal. The host mHC file differs. |
| K | `gpu/kernels/src/flash_kda.rs`; `gpu/kernels/src/flash_gdn.rs`: convolution, step_value, gate; `model/src/kda.rs` | The KDA kernel file is absent. The GDN kernel and host KDA files differ. |
| I | `gpu/src/pooled_indexer_params.rs`: key_norm, pool_channel, weighted_score; `gpu/src/pooled_indexer.rs` | These files and named functions are absent. |
| A | `gpu/kernels/src/qsa_common.rs`: dot, exponential; `gpu/src/qsa.rs`: indexed_attention; `gpu/src/mla.rs` | The QSA files differ. `indexed_attention` and the MLA file are absent. |
| R | `gpu/kernels/src/corrected_router.rs`; `gpu/kernels/src/ungated_moe.rs` | These files are equal. |
| W | `model/src/flash/fp8_block.rs`: project_block_ref; `model/src/flash/nvfp4.rs`: project_ref | The FP8 block file and `project_block_ref` are absent. The NVFP4 file is equal. |
| B | `gpu/kernels/src/flash_bf16.rs`; `gpu/kernels/src/dense_latency.rs` | These files are equal. |
| N | `gpu/kernels/src/flash_ops.rs`: normalize; `gpu/kernels/src/layer.rs`: sum | These files are equal. |
| U | `model/src/swiglu.rs`: reference_with_exp; `gpu/kernels/src/swiglu.rs` | These files differ. The capped kernel `Config` from P is absent. |

P also supplies `AttentionShape` and `Select::score_offsets` in `qsa_common.rs`.
This tree lacks them. `TextDescription::swiglu` from P's host SwiGLU file is present with the same body; it moved to `model/src/plan/description.rs`.
Later primitive merges must supply or reconcile the missing sources and changed contracts.

## Boundary table

BF means BF16. F means FP32. A BF plane can use widened FP32 host storage.
Widening preserves its bits. It does not permit arbitrary FP32 values.
`Unresolved` means the saved source does not state the internal arithmetic
dtype. It is not permission to choose an order. All fixed orders below
describe P references. They do not prove framework equality.

| Logical boundary | Input; accumulator; output/cache | Required rounding and source | Fixed primitive order; exact gap |
| --- | --- | --- | --- |
| Residual and mHC prepare | BF streams; F; F scratch/coefficients | Widen flattened streams and fn. Coefficients stay F. S 209-215, 296-322. | H sums ascending flattened features. Row softmax precedes column normalization. Later iterations normalize rows then columns. Each sum ascends its axis. Framework reduction trees and Linear backend are absent. |
| mHC collapse | F pre, BF residual; F; BF | F product/sum, then BF collapsed result. S 328. | H visits source streams in ascending order with separate product/add. H lacks the final cast. The candidate `mhc_bf16_collapse` adds it. Framework sum tree is absent. |
| Ordinary RMSNorm | BF input/weight; F; BF | **Cast normalized value before weight multiply.** Round BF weight product. S 76-81. This applies to input, post, query latent, KV latent, and final norm. | N has lane feature partials, then shuffle-down 16, 8, 4, 2, 1. Precision zero uses norm_square_add. Other precision uses separate squares. Neither supplies the pre-weight cast. Hub RMSNorm and framework mean tree are absent. |
| Ordinary projections | BF input/weight; unresolved; BF | Each ordinary Linear returns BF under this profile. Low-rank down/up projections have separate outputs. S 97-103, 630-661, 1187-1192. | B uses tiled MMA, operand planes, and fixed range merges. Packing and tree binding are unresolved. S does not state a matmul tree. |
| Quantized projections and KV up | Requested BF input; unresolved; requested BF output | ModelOpt Linear replacement source is absent. Do not infer the activation quantizer or output cast from a storage recipe. S 1150-1172, 1187-1199. | W's FP8 block and NVFP4 oracles use F FMA in ascending K. FP8 dequantized weight multiplication rounds to F before FMA. B's MXFP8 tile tree is distinct. Backend equivalence is unproved. |
| Router projection | BF input/weight widened to F; F operation dtype; F logits | No BF logits cast. S 157-159. | Projection tree is absent. A BF16 projection target alone cannot implement this FP32 router. |
| QKV convolution/history | BF projections; F fallback conv; BF convolved output; history destination dtype unresolved | Fallback casts input/history to F conv weights. Convolution and SiLU run in F. Return to input dtype. S 419-425, 439-448, 1378. | K visits oldest tap first. Logical target uses precision zero. Its BF16 weight ABI, history packing, and F output need reconciliation. Cache class and optional conv backend are absent. |
| KDA prepare | BF convolved QKV, BF forget/beta projections; F; F prepared | Forget projection widens before decay math. **Sigmoid beta returns BF before recurrence widens it.** S 358-364, 475-486, 733-734. | K sums four feature squares per lane in feature-group order, then shuffle-down 16, 8, 4, 2, 1. Primitive beta omits BF sigmoid rounding. Framework/FLA norm tree is absent. |
| KDA recurrence | F prepared/state; F; BF recurrent output, F next state | Cast output to initial query dtype; retain state in F. S 475-513, 530-612, 763. | K visits causal tokens. Prediction and output folds visit ascending key coordinates with separate product/add. Primitive output has no BF cast. Framework step/chunk algorithms differ. FLA is absent. |
| KDA gated norm | BF recurrence/gate; F; BF | F norm, F weight multiply, F sigmoid gate multiply, then BF output. S 382-393, 766-768. | K uses four feature partials per lane and fixed warp shuffle. Framework mean tree and hub replacement are absent. |
| Latent append/gather | BF latent before cache; no sum; unresolved physical cache, requested BF compact scratch | Preserve BF bits for the requested compact binding. Copy counted rows only. Zero unused rows. S 1190-1199, 1258-1276 plus A's compact binding. | No reduction. Cache backend dtype is unknown. A's F cache is a reference, not checkpoint storage proof. |
| Index key LayerNorm | BF projection/affine weights; unresolved; BF | Return BF LayerNorm output. S 835-836. | I sums ascending features for mean, then centered squares, then F affine output. Imported LayerNorm implementation/tree is absent. I lacks the output cast. |
| Index cache/pool | BF keys/gates before cache; unresolved physical cache; F logits/softmax, unresolved pool sum; requested BF pooled key | F gate/APE add and softmax; nan_to_num; **BF probabilities**, BF probability/key products, BF pool result if cache retains BF. S 838-846, 948-1005. | I visits ascending pool positions for max, denominator, and weighted sum. It omits probability/product casts. Cache class and framework sum tree are absent. |
| Index score | BF query/pool; F; F | Widen Q/K. Head projection returns its weight dtype before widening and head scaling. S 862-867. | I: ascending feature dot, scale, ReLU, signed head weight, ascending head sum. Weight headers and framework matmul trees are absent. |
| Pool selection | F scores; integer; U32 IDs/count | No BF cast. Invalid signed -1 uses U32 max. S 875-912. | I: score order, lower pool ID ties, then ascending raw IDs. Framework topk ties are unspecified. Mask padding needs a session binding. |
| Attention scores | BF Q/K; unresolved; BF | Eager matmul returns BF before scale. Round scale product and mask add. S 1088-1092, 1273-1275. | A dots ascend features with separate product/add. A stays F. Framework matmul tree and eager/SDPA choice are absent. |
| Attention softmax | BF scores widened to F; F; BF probability | F softmax then query-dtype BF cast. Inference dropout is zero. S 1094-1095. | A max/denominator visit counted selected positions in ascending order. A omits BF output cast. Framework and SDPA trees are absent. |
| Attention values | BF probabilities/V; unresolved; BF attended | BF matmul output precedes output projection. S 1096-1099. | A visits selected positions in ascending order with separate product/add. A returns F. Framework and SDPA matmul trees are absent. |
| Corrected routing | F logits/bias; F; U32 IDs/F weights | No BF cast. Bias affects choice only. S 157-182. | R uses ascending group/expert IDs for ties. It adds each group's top pair. Selected uncorrected weights fold in selection order, then divide and scale. Framework unsorted topk/sum order is absent. |
| Dense/shared/expert SwiGLU | BF gate/up; pointwise BF results; BF middle | Clamp before SiLU. Round SiLU, then product. Scalar cap must be representable in BF. S 97-103, 136-141. | U has no reduction. It rounds projections/cap, SiLU, and product. Framework SiLU and quantized projection code are absent. |
| Expert/shared fold | BF down, F weights; BF fold; BF FFN output | F weighted product, BF contribution, BF each add, BF shared add last. S 120-134, 199-206. | R folds ascending expert IDs, independent of selected slot order. Single-token fallback has one contribution per expert. Multi-token index_add and expert replacement are unverified. |
| mHC combine | BF residual/block, F coefficients; unresolved matmul; BF streams | BF post/comb; BF post product; BF residual matmul result; BF final add. S 1336-1338, 1345-1347. | H residual fold ascends source streams, then adds post product. H uses uncast F coefficients/terms. The candidate `mhc_bf16_combine` adds these casts. Framework matmul tree is absent. |

All ordinary parameter dtypes in this table are conditional BF16 requirements.
S 1378 explicitly keeps correction bias, conv weights, dt_bias, and A_log in
FP32. S 159 widens router weights. S 303 widens mHC fn. S 388 widens gated
norm weights. No stored tensor header is proved by those conversions.

The descriptor carries every logical input, output, scratch, and returned
plane shape. It does not change the existing FP32 `Buffer` type. Counts are
metadata from the compact binding. They exclude padding. Compact attention
uses local slot IDs, separate from raw cache positions.
Physical history, latent cache, and index cache descriptors stay `Unresolved`.
The source explicitly fixes only recurrent state to FP32.

## Implemented host operations

`round_bf16` uses Vade's existing nearest-even conversion. It rejects
nonfinite inputs and finite FP32 values that round to BF16 infinity. It
preserves signed zero and BF16 subnormals. This finite-only host contract
does not assert the flush behavior of an unpinned device backend.

`norm_weight` accepts a supplied normalized FP32 value and an exact BF16
weight. It implements only the cast-before-weight boundary. It does not
choose an RMS reduction or reciprocal-root implementation.

`expert_fold` accepts unique expert IDs, FP32 route weights, exact BF16 down
rows, and an exact BF16 shared row. It implements R's single-token expert
order and BF cast points. It does not project or select experts.

`gather_bf16` checks extents, count, causal bounds, ascending unique IDs, and
exact BF source values. It reads only counted rows. It does not read padding
IDs or unused cache rows. It zeroes unused compact rows. It does not append,
truncate, allocate, or own persistent state.

Run `cargo +1.99.0 test --offline --locked -p engine-model --test cast_contract`
from `engine/`. Hand vectors test ties, carry, subnormals, cast-before-weight,
per-add rounding, expert order, route precision, shared addition, padding,
causal bounds, and dtype edges. These are host tests, not device proof.

## State and device handoff

For the [state plan](STATE-PLAN.md), retain FP32 recurrent state. Treat BF history,
latent, and index planes here as requirements for the fallback profile only.
Do not choose physical cache widths until headers/backends are verified.
Keep latent cache separate from expanded K/V scratch. Preserve counted
gather and local slot IDs. Valid-token contexts in `LayerPlan` do not define
padded-batch state updates. Those need separate work.

Device work needs a pinned backend for every refused boundary, checked
headers, projection packing, BF epilogues, fixed reductions, and intermediate
buffer comparisons. Then it needs a complete layer/stack driver, session
lifecycle, tensor split binding, and the full exactness gate.

Outside the layer, S 332-336 defines the head class and final stream mean. S 1494 expands
embedding streams. S 1510 applies mean and final norm. Their reduction and
storage bindings, final head quantization, and the appended draft wrapper
remain unbound. Missing inputs include hub implementations, FLA/conv code,
framework version/build, cache classes, and ModelOpt Linear replacements.
Desired token outputs cannot fill these source gaps.
