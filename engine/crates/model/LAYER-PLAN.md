# Deferred decoder layer program

`plan::layer::LayerPlan` binds the operation data in `TextDescription`.
It describes a valid token at a specified context length. The context
includes that token. Repeat the same program in causal token order for a
prompt or a cached continuation. This fixes the recurrence and attention
sum order. It does not choose a different prompt algorithm.

The program is executable on the host through `reference::replay`.
`reference::forward` has a separate schedule. It does not walk the program.
Both consume decoded FP32 weights and small supplied tensors. The tests
compare actual operation events, weight shapes, input shapes, output shapes,
types, settings, and host results. A reordered or missing operation fails.
The comparison does not run a CUDA kernel.

The [causal stack reference](DECODER-REFERENCE.md) connects these layer calls
to the final stream mean, norm, and full-vocabulary head. It retains each
layer's causal state. Its scalar results do not open device admission.

`LayerPlan::validate` checks input, weight, and output counts for each
operation kind. Expert weight counts and the shared input follow the layer
config. Replay rejects invalid counts before it dispatches an operation.

`ConfigPlan::into_model_plan` still refuses this description. There is no
GLM service, tokenizer, weight loader, or device launcher in this change.
Kernel targets name the source revisions that supply the primitives.
Most of those revisions are separate jobs. Their code must be integrated
and qualified before device execution. A named target is not a device pass.

## Print a program

Run from `engine/`. These commands read the committed metadata fixture.
They open no shard and download no data. The fixture record is
[here](tests/fixtures/glm/README.md).

```sh
cargo +1.99.0 run --offline --locked -p engine-model --example layer-plan -- \
  crates/model/tests/fixtures/glm/config.json 0 7 2 0
cargo +1.99.0 run --offline --locked -p engine-model --example layer-plan -- \
  crates/model/tests/fixtures/glm/config.json 3 7 2 1
cargo +1.99.0 test --offline --locked -p engine-model --test layer_binding -- --nocapture
```

The positional arguments are config path, layer index, context length,
part count, and rank index. Each `OP` line gives inputs and a kernel target.
`OUTPUT`, `WEIGHT`, and `INPUT` give global shapes and local intervals.
An absent interval means replication. `GATHER` means copy each interval
into its global position. It does not mean sum rank partials.

The `MODEL_BINDING` and `MODEL_LAYER` test lines are the source for operation
counts. They enumerate the config schedule. The small tests also exercise
each attention and FFN combination. No model dimension selects a model name.

## Operation order

Let `D` be hidden width and `N` be the residual stream count. The input and
output have shape `[N,D]`. The same wrapper applies to either token mixer:

1. Prepare attention mHC coefficients from normalized raw streams.
2. Collapse raw streams with the pre coefficients. Apply the input RMS norm.
3. Run KDA or indexed NoPE latent attention.
4. Combine its output with the raw attention residual. The matrix uses
   `[source,destination]` order.
5. Prepare FFN mHC coefficients from this new residual. Collapse and apply
   the post-attention RMS norm.
6. Run the dense FFN or the routed experts and shared FFN.
7. Combine the FFN output with its raw residual.

KDA projects Q, K, and V separately. It applies causal convolution and SiLU.
It then applies the forget down/up projections and the beta projection.
Prepare normalizes Q and K and computes channel decay and sigmoid beta.
The ordered recurrence updates state before it computes the output.
The output gate uses its down/up projections. Weighted RMS norm and sigmoid
gating precede the output projection. The plan uses precision zero for the
logical FP32 KDA primitives. It does not reuse the scalar-decay GDN scan.

Latent attention projects and normalizes the query latent. It then expands
the query. It projects and normalizes the KV latent and appends it to cache.
The indexer projects its query from the normalized query latent. It projects
and normalizes its key from the block input. Learned gate logits and the
position table pool complete groups. Signed head weights scale ReLU scores.
Selection emits ascending raw IDs and a count. The incomplete tail follows
the configured policy. Key LayerNorm has the operator epsilon in `Settings`.

Gather copies only counted latent rows. KV up projection expands these rows
before attention. It splits each head's adjacent key and value rows into
token-major `[slot,head,feature]` planes. The caller supplies `local_slots`
as U32 values with the selected capacity. Each counted slot must satisfy
`local_slots[j] == j` for `j < selected_count[0]`. Entries after the count
are unused. Scores and values use those local IDs after gather.
They do not reuse raw cache positions against the compact scratch plane.
Softmax and value sums use ascending selected position order. The form is
explicit. No absorbed form or execution switch is present.

The dense FFN projects gate and up, applies capped SwiGLU, and projects down.
The expert FFN projects router logits and selects corrected sigmoid scores.
Correction bias affects selection only. Group scores use the largest pair.
Ties prefer the lower group or expert ID. Selected weights come from the
uncorrected sigmoid scores. Normalization and routed scaling follow config.
Experts use separate gate/up projections, capped SwiGLU, and down projection.
The output fold visits expert IDs in ascending order. Shared output is
added once after that fold. It has no router weight or shared gate.

## Kernel targets and remaining gaps

The revision constants in [layer.rs](src/plan/layer.rs) identify the prior
jobs. The table updates the loader report's operation gaps. A primitive
exists at the stated revision. Complete device binding remains open.

| Operation | Kernel target | Remaining device work |
| --- | --- | --- |
| FP8 block weight-only projection | `fp8_block_weight_only`, job own-glm-fp8-proj | Check actual E4M3 weight headers and floating scale headers. Widen scales losslessly. Bind BF16 inputs and original block offsets. |
| MXFP8 projection | `flash_next_mxfp8_m1`, main | Bind stored E4M3 and UE8M0 bytes directly. Validate scale ranges and operand boundaries. |
| BF16 projection | `flash_bf16_input`, `flash_bf16_pack`, `flash_bf16_mma`, main | Check stored shapes. Preserve full input K and fixed projection arithmetic. Bind operand packing. |
| Expert NVFP4 projection | `fp4_exact_project`, main | Expand CSF32 on the host. Bind expanded scales and global scales. Pack A16 operands. Use separate gate/up outputs before the capped epilogue. |
| KDA convolution and history | `flash_gdn_conv`, `flash_gdn_history`, job own-glm-kda | Pack separate conv matrices and map logical history to the channel-major device history. Use the supported tap count. |
| KDA prepare and ordered recurrence | `flash_kda_prepare`, `flash_kda_step`, job own-glm-kda | Map logical prepared rows to separate ABI planes. Check the kernel head dimension. Qualify decode and repeated-token prompt execution. |
| KDA gated output norm | `flash_gdn_gate`, job own-glm-kda | Bind precision zero, configured epsilon, widened norm weights, and the supported head dimension. |
| MLA cache insertion, gather, and KV split | `flash_hybrid_copy`, main and job own-glm-mla | Build bounded copy descriptors. Bind token-major K/V strides and counted local slot IDs. Cache keeps only the latent. |
| MLA scores, softmax, and values | `qsa_fp32_attention_scores`, `qsa_fp32_softmax`, `qsa_fp32_values`, job own-glm-mla-fix2 | Integrate the dimension ABI. Resolve the outstanding device illegal-address failure in that job's report. Verify output bounds and BF16 boundaries. |
| Index key LayerNorm | `pooled_key_norm`, job own-glm-indexer | Bind centered affine norm with bias and the operator epsilon. Check stored scale and bias shapes. |
| Learned pooling and scoring | `pooled_keys`, `pooled_scores`, job own-glm-indexer | Bind gate/position tensors. Retain complete pools and an incomplete tail in a device session. Qualify probability casts and signed weights. |
| Pool selection | `qsa_fp32_select`, job own-glm-indexer | Integrate its generalized selection ABI with MLA's separate attention ABI. Preserve counts, padding, causality, and tie order. |
| Corrected sigmoid routing | `corrected_sigmoid_route`, job own-glm-routing | Bind FP32 logits, bias, group data, and sticky status. Qualify selection and normalization against the checkpoint contract. |
| Ungated expert/shared fold | `ungated_moe_finish`, job own-glm-routing | Bind complete down rows and shared output. Qualify the kernel's BF16 product/add boundaries. |
| mHC prepare, collapse, and combine | `mhc_prepare`, `mhc_collapse`, `mhc_combine`, job own-glm-mhc | Widen coefficient weights and bind normalized scratch. Qualify residual and coefficient casts. |
| Capped SwiGLU | `layer_swiglu`, job own-glm-capped-swiglu | Pass the config bound to the standalone epilogue. Keep existing fused uncapped kernels out of this program. Qualify BF16 boundaries. |
| Ordinary input/post RMS norm | `flash_norm`, main | Use direct weights, configured epsilon, and no weight offset. Qualify casts and reduction order. |
| Tensor split gathers | Existing host transport, with `flash_hybrid_copy` for local copies | Integrate global-order copy gathers and their stream lifetimes. There is no partial-sum all-reduce in this description. |
| Embedding stream expansion, outside this layer | `flash_embedding_expand`, main | The existing entry has a fixed stream count. Other stream counts need a kernel. Bind the fixture count after header checks. |
| Final mean, norm, and head, outside this layer | `mhc_head`, `flash_norm`, `flash_next_mxfp8_m1` | Bind the parameter-free mean, final norm, and stored head. No final text-head driver is supplied here. |
| Appended prediction layer, outside this layer | No complete kernel program | Fusion order, norm boundaries, head sharing, cache rollback, and a verified draft wrapper remain missing. |

`LayerPlan::missing` lists every step for which the operation geometry has
no target. This includes unsupported convolution taps, KDA head dimensions,
projection layouts, mixed expert layouts, and finish geometry. Small host
tensors can run even when their geometry has no CUDA target. The saved
fixture's `MODEL_BINDING` output reports its target coverage. Coverage is
not device admission or exactness proof. New kernel arithmetic is not added
in this change.

The logical FP32 reference uses host exponentials and square roots. Its dot
products use ascending feature order. The device targets have their own
fixed reduction and FMA contracts. The reference does not claim device bit
equality. It also does not reproduce checkpoint BF16 casts, quantization,
or framework reduction trees. These remain device and quality gates.

[BF16 boundary requirements](CAST-CONTRACT.md) give a separate conditional
dtype profile for this graph. They record explicit source casts and refused
backend gaps. They do not change this FP32 reference or admit device use.

## Buffers and ownership

Each program lists external inputs, produced buffers, step scratch, and
returned buffers. Shapes are global logical extents. `elements` checks byte
arithmetic before allocation. `lifetimes` maps external inputs and step
outputs to `(birth, last_use)`. It does not include step scratch. External
inputs have birth `None`. A step output has birth `Some(step_index)`.
Last use is inclusive. For each returned buffer, `steps.len()` is the
last-use sentinel. Returned state and output remain live through the call.
A session retains returned state for the next token.

KDA retains `[head,key,value]` recurrent state and separate convolution
histories. The host history shape is `[QKV,past-tap,channel]`. The physical
convolution ABI uses a different layout and a reserved history slot.
The prepared host record is `[head,Q/K/V/decay/beta]`. The physical KDA ABI
uses separate head-major planes. These layouts require bounded copy or pack
descriptors. They are not interchangeable pointer casts.

Latent attention retains normalized `[context,latent]` rows. The host
indexer retains raw normalized keys and gate logits. Its pool plane is
`[complete-pool,index-feature]`. The device session can compact completed
pools, but that memory optimization has no device binding here. The selected
IDs have reserved capacity for the token budget and incomplete tail. Counts
exclude padding. Expanded K/V scratch scales with selected capacity, not
full context. The final output has the input's stream shape.

Weights carry logical matrix shapes, storage recipes, and tensor splits.
The [header admission checks](HEADER-ADMISSION.md) validate physical storage,
including convolution singleton axes and stored mHC scalar ranks.
Logical shapes alone do not prove those properties. Device MMA input planes, layout padding,
status buffers, and launch descriptors still need an allocation plan.

The [checked host state plan](STATE-PLAN.md) now supplies local byte offsets,
capacity checks, and separate state owners. It uses these logical lifetimes.
Its remaining-gap list names the device ABI storage that it does not cover.

## Tensor parallelism

The same global operation sequence applies to each topology. `Topology`
computes equal contiguous intervals of complete tensor units. It rejects
an indivisible unit. The examples above show the requested pair. They are
metadata checks on the host, not a pair run.

| Tensor or operation | Ownership |
| --- | --- |
| Residuals, mHC, ordinary norms, router | Replicated |
| KDA Q/K/V, conv weights/history, prepare, recurrence, output gate | Head intervals. Gate-down projections remain replicated. The output norm weight is shared. |
| MLA query/KV down projections and latent cache | Replicated |
| MLA query up, KV up, score/softmax/value work | Head intervals. Each KV-up weight interval keeps a head's adjacent K and V rows together. |
| Learned indexer and selected raw IDs | Replicated |
| Dense/shared/expert gate and up | Intermediate output-row intervals |
| Capped SwiGLU | Same intermediate intervals |
| Attention and FFN down/output projections | Hidden output-row intervals, with complete global input K |

Gather the complete gated KDA output or attended MLA output before attention
output projection. Gather each FFN middle before down projection. Gather
attention output, expert down rows, and shared output before replicated
residual or expert folds. Each dot visits its full global input dimension
in the same order. No rank produces a partial dot for another rank to add.
Retain original block origins when slicing quantized scale grids.

This is the proposed tensor split for this deferred program. It does not
change the current serving splitter. It has no measured memory or speed
result. Exactness across device counts still needs the full device gate.

## Source record

The metadata fixture record names its checkpoint revision. Operation order
comes from the saved `tf_modeling_glm5_next.py` used by the listed jobs.
Its saved SHA256 is
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
The source fetch used an unpinned Transformers branch. The saved decoder
forward defines the wrapper order. The saved KDA, attention, indexer, router,
MLP, and hyper-connection definitions supply their component math. No
reference framework or Python program was run for this change.

MODEL-GATE OPEN means the logical plan and host contract can be reviewed and
used to prepare primitive qualification. The complete GLM request gate in
the prior MLA and indexer reports still requires model admission. This
program does not make that full device gate runnable by itself.
