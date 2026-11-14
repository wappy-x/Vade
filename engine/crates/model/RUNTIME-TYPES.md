# Deferred runtime weight and activation map

This map separates stored weights, logical scalar values, and runtime operands.
It does not qualify a checkpoint runtime or open service admission.
An index name or a kernel target does not establish a runtime producer.

## Source and status rules

The scalar source is `761962546fcfdd351858c5ccef20317a199f9376`.
This branch retains its [layer reference](M2-REFERENCE.md) and [stack reference](DECODER-REFERENCE.md).
The scalar profile name remains `logical-f32-v1`.
All scalar floating planes hold FP32 values.
Every scalar dot uses separate products and ascending feature adds.
It uses host exponentials and square roots.
It does not insert checkpoint activation casts or activation quantizers.

S means the saved execution source identified in [LAYER-PLAN.md](LAYER-PLAN.md#source-record).
Its SHA256 is `9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
S line ranges below identify source operations, not a pinned framework build.
The ModelOpt Linear replacements, cache classes, hub norms, and optional convolution and FLA backends remain absent.

R means resident source `8719ddb304c1778c154bb7441a758e0121c4cbf9`.
Relevant paths are `engine/crates/tp/RESIDENT-OPERANDS.md` and `engine/crates/tp/src/layer_operands.rs`.
Its cast map is `engine/crates/model/CAST-CONTRACT.md`.
Those resident files are present in this branch.
Their source status does not establish a device result on this branch.

| Status | Meaning |
| --- | --- |
| Implemented scalar | This branch computes the stated logical FP32 values. |
| Implemented candidate | The named source binds a primitive with its own arithmetic contract. Checkpoint equivalence remains open. |
| Declaration | A plan names an operation, dtype requirement, or kernel target. It does not bind the complete producer. |
| Absent | No complete checkpoint producer supplies the required values to this consumer. |

## Stored type to operand producer

The [storage map](src/plan/storage.rs) chooses recipes by module prefix and exclusions.
The [checkpoint reader](examples/layer-reference/checkpoint.rs) reads and decodes the stored components.
Header checks remain distinct from effective runtime load policy.
These rows state the reader's accepted types. They do not assert unseen checkpoint headers.

| Stored type and components | Implemented scalar producer and result | Runtime producer and consumer status |
| --- | --- | --- |
| Plain BF16 | `decode_plain`: exact widening to FP32. Embedding rows use the same widening. | R binds plain BF16 ordinary projections through packed upload and native BF16 projection. It supplies local BF16 rows. R also binds stored-precision norm and mHC parameters. These are candidates. Effective upstream load policy remains absent. |
| Plain FP32 | `decode_plain`: retain FP32 bits. | R preserves stored values for norms and mHC operands. This branch adds a [plain FP32 native candidate](../tp/ORDERED-PROJECTION.md) with the scalar product/add order. It binds retained file rows to the generic backend's loaded matrix. Checkpoint runtime profiles remain typed refusals. Effective runtime loads and framework projection equivalence remain absent. |
| MXFP8: E4M3 weights; U8 UE8M0 scales per input group | `mxfp8::dequantize`: exact FP64 product rounded to FP32. Decoded values feed scalar projection. Group geometry comes from `mxfp8::BLOCK`. | The [stored projection candidate](../tp/MXFP8-PROJECTION.md) binds complete selected rows after an ordinary norm. It preserves the norm dtype and applies a BF16 epilogue after the fixed device tree. This candidate does not qualify the checkpoint Linear replacement in R. Runtime activation quantization and epilogue equivalence remain open. |
| FP8 blocks: E4M3 weights; BF16 or FP32 block scales | Widen scales. Round each E4M3 times scale product to FP32. Scalar projection then uses separate products and adds. Block edges come from the recipe. | `fp8_block_weight_only` is a declared layer target. Source `97eaa35fdd7484cfa4c715c263e32e659d2924f5` supplies BF16 inputs, FP32 scaled weights, and ascending FP32 FMA. That primitive is not integrated here. R has no quantized projection binder. |
| NVFP4: packed U8 E2M1 pairs; direct E4M3 block scales; floating global scalar | `nvfp4::dequantize`: `E2M1 * E4M3 * global`, in that order, to FP32. Low nibbles precede high nibbles. Block geometry comes from the decoder. | `fp4_exact_project` implements a W4A16 primitive with packed BF16 inputs and ascending FP32 FMA. The layer target is a declaration. Checkpoint replacement and BF16 output epilogue remain absent. |
| NVFP4 with CSF32 scale components | `Csf32::expand` produces exact E4M3 scale bytes. The same NVFP4 decoder consumes them. It adds no activation cast. | Host scale expansion can supply the primitive's expanded scales. It does not establish an activation producer. |
| NVFP4 `input_scale`, when declared | Check a positive finite scalar and record its identity. Do not apply it. | A calibration scalar does not define the activation quantizer. `NVFP4` and `W4A16_NVFP4` recipes remain distinct. The former cannot silently inherit the latter's A16 primitive contract. |
| Legacy CSF scale components | The scalar checkpoint reader refuses this encoding. The separate existing legacy decoder is not selected here. | No complete deferred layer producer is bound here. |

The scalar reader accepts BF16 or FP32 global and activation scalars through `decode_plain`.
A stricter runtime header contract must establish their actual stored types.
For a logical matrix `[R,K]`, MXFP8 stores scales `[R,K/32]`.
NVFP4 stores packed weights `[R,K/2]` and expanded block scales `[R,K/16]`.
These format units come from the [MXFP8](src/flash/mxfp8.rs) and [NVFP4](src/flash/nvfp4.rs) decoders.
FP8 block scales use `[ceil(R/br),ceil(K/bc)]`, with optional singleton axes.
The recipe supplies `br` and `bc`. The scalar reader accepts a final clipped block.
CSF32 uses U8 bases/codes, little-endian U32 page bases, and little-endian U16 offsets/exceptions.
The [codec](CSF.md) defines those component byte layouts.
The scalar reader passes compressed payload bytes to that codec. It does not establish every compressed component header dtype.
The NVFP4 decode rounding rule in [M2-REFERENCE.md](M2-REFERENCE.md#error-and-byte-rules) remains unchanged.
It is a weight decode bound. It is not a decoder output bound.

## Resolve recipes for a checkpoint description

The [saved config](tests/fixtures/glm/config.json) declares the following representative families.
This table describes recipes. It does not assert physical header or effective parameter dtypes.
The command below prints the exact consumer, weight name, selected recipe, and gather for a requested layer.
It opens no shard. `OP` names the consumer of each following `WEIGHT` row.

| Declared family | Checkpoint roles in the saved fixture | Producer/consumer chain |
| --- | --- | --- |
| MXFP8 | KDA Q/K/V and output; dense FFN gate/up/down; text head | Stored E4M3/UE8M0 decode to scalar F; projection; later scalar consumer. Runtime activation/epilogue binding remains absent. |
| FP8 block weight-only | MLA query down/up, KV down, output; shared FFN gate/up/down | Scaled E4M3 decode to scalar F; projection; latent norm, attention combine, or shared fold. Runtime quantized producers remain absent. |
| NVFP4 with CSF32 and activation scale | Routed expert gate/up/down | Expanded scale bytes; NVFP4 decode; selected projections; capped middle; ordered fold. Runtime activation quantizer remains absent. |
| Plain | mHC, ordinary norms, KDA low-rank gates/beta and convolution/decay, MLA KV up, indexer, router, embedding | Plain BF16/F32 decode; the role consumer in the next table. `Plain` alone cannot choose between BF16 and FP32. |

Run from `engine/`. Select each desired layer and topology from the description:

```sh
cargo +1.99.0 run --offline --locked -p engine-model --example layer-plan -- \
  crates/model/tests/fixtures/glm/config.json 0 1 1 0
cargo +1.99.0 run --offline --locked -p engine-model --example layer-plan -- \
  crates/model/tests/fixtures/glm/config.json 3 1 1 0
cargo +1.99.0 run --offline --locked -p engine-model --example layer-plan -- \
  crates/model/tests/fixtures/glm/config.json 4 1 1 0
```

These arguments select representative dense KDA, expert MLA, and expert KDA layers from the fixture.
The examples are source inspection. A printed target is still a declaration.

## Weight roles and their consumers

Layer names below omit `model.language_model.layers.L.`.
Braces name separate tensors. Each projection uses its own selected storage recipe.
No ordinary parameter's effective loaded dtype follows from the config's requested dtype alone.

| Checkpoint weight role | Producer and immediate consumer | Runtime value requirement and open boundary |
| --- | --- | --- |
| `hc_{attn,ffn}_fn` | Plain decode; each side's `prepare` coefficient projection. | S 301-303 widens the effective loaded fn to FP32. R preserves stored values. A prior FP32-to-BF16 runtime load would change the operand. That load policy is unqualified. |
| `hc_{attn,ffn}_{base,scale}` | Plain decode; coefficient scale and bias before sigmoid/Sinkhorn. | S 306-315 uses effective parameters with FP32 projection values. R preserves stored values. Parameter load and framework reduction equivalence remain open. |
| `{input_layernorm,post_attention_layernorm}.weight` | Plain decode; `input_norm` and `post_norm`. | S 76-81 casts normalized activations back before parameter multiply. Effective weight dtype controls the product dtype. R implements stored-precision identity load only. |
| `self_attn.{q,k,v}_proj.weight` | Recipe decode; separate KDA `q`, `k`, `v` projections. | Separate BF16 output boundaries are conditional requirements. Quantized replacement producers are absent. |
| `self_attn.{q,k,v}_conv1d.weight` | Plain decode; KDA causal convolution. The reader removes a stored singleton group axis. | S 1378 retains FP32 convolution parameters. S 439-448 widens inputs to that dtype, then returns to input dtype. History layout and backend remain unbound. |
| `self_attn.{f_a_proj,f_b_proj}.weight` | Recipe decode; `forget_down`, then `forget`. | Preserve the separate low-rank output boundary. Widen the final forget value for decay math. Runtime quantized producers remain absent. |
| `self_attn.b_proj.weight` | Recipe decode; `beta_logits`, then sigmoid beta. | Round beta to input dtype before recurrence widens it. The scalar profile omits that rounding. |
| `self_attn.{A_log,dt_bias}` | Plain decode; `kda.prepare` decay. | S 1378 retains FP32. Prepared decay and recurrent state stay FP32. Physical argument layout needs a complete producer binding. |
| `self_attn.{g_a_proj,g_b_proj}.weight` | Recipe decode; `output_gate_down`, then `output_gate`. | Preserve the separate low-rank output boundary. Gated norm widens the final gate. |
| `self_attn.o_norm.weight` | Plain decode; KDA `gated_norm`. | S 382-393 widens effective norm weights to FP32. Normalize, multiply weight, apply sigmoid gate, then cast the result. |
| `self_attn.o_proj.weight` | Recipe decode; `attention`, from gathered KDA or MLA features. | Complete global inputs precede projection. Gather output rows before attention mHC combine. |
| `self_attn.{q_a_proj,q_b_proj}.weight` | Recipe decode; `query_down`, then normalized latent, then `query`. | Query latent norm separates the down/up projections. Quantized runtime outputs remain unbound. |
| `self_attn.q_a_layernorm.weight` | Plain decode; `query_latent`. | Ordinary norm cast-before-weight rule. Effective parameter load remains unqualified. |
| `self_attn.kv_a_proj_with_mqa.weight` | Recipe decode; `kv_down`, then KV latent norm. | Keep projection output distinct from the normalized cache row. |
| `self_attn.kv_a_layernorm.weight` | Plain decode; `kv_latent`, then cache append. | Ordinary norm rule. Physical latent cache dtype remains absent. |
| `self_attn.kv_b_proj.weight` | Recipe decode; `kv_expand` after counted latent gather. | Preserve each head's adjacent K/V rows. Produce counted slot-major K/V planes. No expanded-K/V persistent cache is declared. |
| `self_attn.indexer.wq_b.weight` | Recipe decode; `index_query`, from normalized query latent. | Widen query values for FP32 index scores after the projection output boundary. |
| `self_attn.indexer.wk.weight` | Recipe decode; `index_key_raw`, then key LayerNorm. | LayerNorm must receive the projection's actual output dtype. |
| `self_attn.indexer.k_norm.{weight,bias}` | Plain decode; `index_key` centered affine LayerNorm. | S 835-836 requires the LayerNorm output dtype. Internal framework reduction and affine parameter load remain unqualified. |
| `self_attn.indexer.index_kpool_compress_gate` | Plain decode; `index_gate` matrix projection. It has no `.weight` suffix. | An arbitrary FP32 matrix producer remains absent in R. Preserve this projection boundary before cache/pool. |
| `self_attn.indexer.index_kpool_compress_ape` | Plain decode; `pool` gate/position logits. | Add in FP32 before pool softmax. Physical key/gate cache types remain absent. |
| `self_attn.indexer.weights_proj.weight` | Recipe decode; `index_head_weights`, then `pool_scores`. | S 862-867 casts input to effective weight dtype. Output returns to that dtype before FP32 widening and head scaling. This is not the router projection. |
| `mlp.{gate_proj,up_proj,down_proj}.weight` | Recipe decode; dense gate/up, capped middle, and `ffn_output`. | Preserve projection outputs and capped SwiGLU casts. Gather middle before down; gather down before FFN mHC combine. |
| `mlp.gate.weight` | Plain decode; `router_logits`, then corrected routing. | S 159 widens input and effective weight to FP32. Logits stay FP32. R explicitly refuses this projection. This branch can bind its plain FP32 rows under the logical scalar profile only. Effective runtime load and framework matmul remain unqualified. |
| `mlp.gate.e_score_correction_bias` | Plain decode; corrected route selection only. | S 1378 retains FP32. Use uncorrected sigmoid scores for selected weights. Bias does not enter the weighted expert fold. |
| `mlp.experts.E.{gate_proj,up_proj,down_proj}.weight` | Recipe decode for selected expert IDs only; `expert_gate`, `expert_up`, `expert_down`. | Bind actual quantized producers. Preserve original expert IDs, separate gate/up outputs, and full down rows before the ordered fold. |
| `mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight` | Recipe decode; shared gate/up, capped middle, and `shared_output`. | Gather middle and down. Add shared output once, after routed contributions. There is no shared router gate. |
| `model.language_model.embed_tokens.weight` | BF16 row decode; stack residual stream copies. | Scalar copy is implemented. Effective runtime embedding load and resident text integration remain unqualified. |
| `model.language_model.norm.weight` | Plain decode; stack norm after parameter-free stream mean. | Ordinary norm cast-before-weight rule is absent from the scalar profile. No resident text driver binds it here. |
| `lm_head.weight`, or tied embedding | Recipe decode; full scalar logits. | Runtime head quantization and output type remain absent. Finite scalar argmax selects the lowest ID on ties. |
| Appended draft `{eh_proj,enorm,hnorm,shared_head.norm}` and draft body | Metadata declarations only. The scalar stack visits decoder layers only. | Draft fusion order, head sharing, load/cast policy, and complete runtime producers remain absent. They cannot inherit a decoder profile. |

## Activation and state boundaries

B means the conditional BF16 activation requirement from R's cast contract.
F means FP32. These letters do not assert a qualified checkpoint dtype.
The scalar profile keeps every listed floating result in F, with no B casts.

| Boundary | Required runtime conversion and order | Source and producer limit |
| --- | --- | --- |
| Both mHC prepares | Widen residuals and effective fn to F. Normalize flattened features. Keep coefficients in F. | S 209-215, 296-322. R binds candidate resident mHC. Effective loads and framework trees remain open. |
| Both mHC collapses | F pre times widened streams; sum source streams; cast to B. | S 328. R binds candidate BF16 collapse. Scalar reference omits its cast. |
| Input, post, query-latent, KV-latent, final RMSNorm | Normalize in F. Cast normalized value before parameter multiply. Then apply effective weight dtype. | S 76-81. A stored FP32 weight can produce F under identity load. R's norm preserves this distinction. |
| Every ordinary and low-rank projection output | Preserve a result at every Linear call. Conditional ordinary input/weight/output are B. | S 97-103, 630-661, 1187-1199. R binds plain BF16 ordinary projections only. Quantized accumulator, operand quantizer, and casts remain absent. |
| Router projection and route | Widen input and effective weight to F. Keep logits, sigmoid, correction, group scores, selected weights, normalization, and scaling in F. | S 157-182. Router projection remains absent in R. Backend selection ties and sum order remain unqualified. |
| KDA convolution/history | Widen Q/K/V and history to F convolution parameters. Convolve and apply SiLU in F. Return to B. Copy history through destination dtype. | S 419-425, 439-448, 1378. Physical history dtype and packing remain unbound. |
| KDA prepare | Widen convolved Q/K/V and forget values. Compute F decay. Round sigmoid beta to B, then widen. | S 358-364, 475-486, 733-734. Declared prepare primitive omits beta rounding. |
| KDA step | Retain F recurrent state. Visit causal tokens in order. Cast recurrent output to initial query dtype, conditionally B. | S 475-513, 530-612, 763. Primitive and scalar F outputs do not establish this cast. |
| KDA gated norm | Widen recurrent output, norm weight, and gate. Normalize, multiply weight, multiply sigmoid gate in F. Cast output to B. | S 382-393, 766-768. Framework reduction and hub implementation remain absent. |
| MLA cache append and compact latent gather | Preserve latent bits through actual cache dtype. Gather counted causal rows only. Zero unused compact rows. | S 1190-1199, 1258-1276. Cache class is absent. B compact scratch is a conditional requirement. |
| Index key LayerNorm | Return projection dtype, conditionally B. Do not infer the internal sum tree from this output. | S 835-836. Named primitive's F output needs a cast and backend qualification. |
| Index pool | F gate/APE logits and softmax; `nan_to_num`; B probabilities; B probability/key products; pool result in cache dtype. | S 838-846, 948-1005. Scalar pooling omits B boundaries. Physical cache and framework sum tree remain absent. |
| Index scores and selection | Widen Q/K to F. Head weights cross their own projection dtype before F scaling. Scores stay F. Emit integer raw IDs/count. | S 862-912. Lower-ID ties and ascending raw IDs are Vade requirements. Framework topk ties remain unspecified. |
| MLA attention scores | Eager QK matmul returns B before scale. Round scale product and mask add to B. | S 1088-1092, 1273-1275. Declared FP32 scores omit these casts. Eager/SDPA choice remains absent. |
| MLA attention probabilities | Widen scores to F for softmax. Cast probability to query dtype, conditionally B. | S 1094-1095. Declared FP32 softmax is not a complete producer. |
| MLA attended values | Probability/value matmul returns B before output projection. Counted local slots address compact K/V. | S 1096-1099. Declared FP32 output needs conversion and backend qualification. |
| Dense/shared/expert capped SwiGLU | Preserve B gate/up. Clamp first. Round SiLU to B, then round product to B. | S 97-103, 136-141. Scalar reference omits casts. An uncapped fused epilogue cannot substitute. |
| Expert fold | F route weight times B down row; cast contribution to B; round each expert add to B. Add B shared output once, last. | S 120-134, 199-206. Visit ascending expert IDs, not selected slot order. Multi-token backend order remains unqualified. |
| Both mHC combines | Cast post and matrix coefficients to B. Round post/block product and residual matmul separately. Round final add to B. | S 1336-1338, 1345-1347. R binds candidate residual operations. Actual attention and FFN output producers remain incomplete. |
| Final mean, norm, head | Mean returns activation dtype before norm. Preserve norm's pre-weight cast and actual head replacement output. | S 332-336, 1494, 1510, 2258. Scalar F mean/norm/logits are implemented. Resident text producers remain absent. |

## Ordered gather boundaries

Each rank owns complete tensor units and complete input features for each dot.
A gather copies intervals into global positions. It never adds partial dot products.
The [layer plan](src/plan/layer.rs) declares these boundaries in operation order:

| Producer output | Complete consumer after gather |
| --- | --- |
| KDA `gated` | Attention output projection. Head intervals preserve feature order. |
| MLA `attended` | Attention output projection. Head intervals preserve value feature order. |
| `attention` | Attention mHC combine. Hidden row intervals become replicated. |
| Dense `dense.middle` | Dense down projection, with complete intermediate features. |
| Dense `ffn_output` | FFN mHC combine. |
| Expert `expert_middle` | Selected down projections. Gather the feature axis within each selected slot. |
| Expert `expert_down` | Replicated expert fold. Gather hidden rows within each slot. Retain original expert IDs. |
| Shared `shared.middle` | Shared down projection, with complete intermediate features. |
| Shared `shared_output` | Replicated expert fold. Add this complete row last. |

The expert fold's `ffn_output` is replicated. It has no extra gather.
Low-rank down projections, the router, and the indexer are replicated. They need no rank gather.
KDA convolution/state and MLA attention work retain local head intervals between these boundaries.
The separate MLA `gather` compacts cache rows. It is not a rank collective.
It checks ascending unique causal raw IDs and count. Later attention uses compact local slot IDs.

[PairPlan](../tp/LAYER-TRANSPORT.md) binds declared gathers to ordered host byte exchanges.
Its caller supplies checked physical dtypes. Its completion precedes the next step.
R's projection adapter returns local rows. It does not connect device completion to that transport.
A runtime must retain producer owners through completion and gather before the listed consumer reads.
State publication must follow all compute and transport. Synthetic transport cannot establish those producer lifetimes.

## Typed refusal before scalar execution

[`WeightSource::profile`](src/plan/layer/reference.rs) is a required declaration. It has no default.
[`Profile::require_scalar`](src/plan/layer/profile.rs) accepts only `LogicalF32V1`.
It returns `UnsupportedProfile` with `Boundary::EffectiveParameterLoad` for either unqualified runtime profile.
This is the first unresolved weight boundary. Widening after load cannot recover values already rounded during load.
The full stack also needs an effective embedding load before the first layer.

`Decoder::new` checks the profile before header reads or state allocation.
`append_observed` checks before state copies, payload reads, or observers.
`forward_from` checks before any layer compute or weight read.
The checkpoint reader explicitly declares `LogicalF32V1`. Raw tensor fixtures have the same logical meaning.
No new engine flag, precision default, or numerical path exists.
The other profile variants remain requests that always fail this check.
They cannot authorize parameter conversion, resident allocation, or service execution.

The [host fixtures](tests/decoder_reference.rs) test accepted scalar values and rejected runtime profiles.
They also change a source profile after a successful append and check refusal without reads or state changes.
These fixtures do not qualify runtime casts, a device layer, logits parity, or Spark token IDs.
No useful new device operation follows from this refusal check. This change needs no device kit.
