# engine/crates/model/src/plan

**Rust modules for `plan`, part of the `engine-model` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`adapter.rs`](adapter.rs) | Config and tensor metadata for schemas with missing execution operations. |
| [`config.rs`](config.rs) | Config adapters for checkpoint-derived geometry and supported semantics. |
| [`description.rs`](description.rs) | Host descriptions for operations that have no execution binding. |
| [`dsa.rs`](dsa.rs) | Logical tensor, compressed pool and per-query selection metadata. |
| [`error.rs`](error.rs) | Typed failures while reading, validating and reconciling a model plan. |
| [`glm.rs`](glm.rs) | Translate the GLM text schema into operation data for host inspection. |
| [`glm_index.rs`](glm_index.rs) | Bind index names to tensor roles and scale components. |
| [`inkling.rs`](inkling.rs) | Translate the looped text schema into operation data. This is host metadata only. |
| [`inkling_index.rs`](inkling_index.rs) | Give each stored tensor name of the looped text family a role, a type and a shape. |
| [`layer.rs`](layer.rs) | Logical decoder order, kernel targets, buffers, and tensor splits. |
| [`loop_load.rs`](loop_load.rs) | The load plan of the looped text family: what each device tensor is made from. Host metadata only. |
| [`loop_state.rs`](loop_state.rs) | State of one chat of the looped text family, for each layer visit, and the byte ledger of one device. |
| [`loop_widen.rs`](loop_widen.rs) | Turn stored 4-bit and 8-bit expert matrices into BF16 at load. Host arithmetic only. |
| [`loop_packed.rs`](loop_packed.rs) | The contract and the host twin of an expert product that reads the stored 4-bit and 8-bit codes. Host arithmetic only. |
| [`decoder_reference.rs`](decoder_reference.rs) | Causal scalar stack, text logits, saved state, and failure rollback. See the [reference contract](../../DECODER-REFERENCE.md). |
| [`layer/`](layer/README.md) | Independent scalar forward and program replay. |
| [`ffn_tests.rs`](ffn_tests.rs) | Deferred FFN tests for the shared host schedule builder. |
| [`mod.rs`](mod.rs) | A checkpoint-derived, typed model plan, independent of loading and serving. |
| [`profile.rs`](profile.rs) | Count logical compulsory reads from model geometry. |
| [`runtime.rs`](runtime.rs) | Hybrid runtime geometry and tensor roles, including draft layers. |
| [`storage.rs`](storage.rs) | Read mixed storage recipes without device admission. |
| [`storage_groups.rs`](storage_groups.rs) | Storage groups of a quantization sidecar that lists tensor names. Host metadata only. |
| [`admission.rs`](admission.rs) | Header inventory checks and device upload byte views for deferred text. |
| [`state.rs`](state.rs) | Checked host state byte layouts, capacity, and saved boundaries. |
| [`tensors.rs`](tensors.rs) | Expected tensor names, storage, roles and header reconciliation. |
| [`tensor_split.rs`](tensor_split.rs) | Checked tensor intervals and original projection scale origins. See the [transport contract](../../TENSOR-SPLIT.md). |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `plan`.

## Deferred config plans

The public `plan::ConfigPlan` re-export names
`description::ConfigPlan`, which returns `TextDescription`. It validates
operation data and maps index names. The separate `adapter::ConfigPlan`
below returns `DeferredPlan` for partial header checks. These APIs have
different validation scopes. See [the API table](../../GLM.md).

`adapter::ConfigPlan::from_config` reads supported and deferred schemas. A data table
accepts `glm5_next` and `glm5_next_text`. A deferred plan has no executable
`ModelShape`. `into_model_plan` and `ModelPlan::from_config` return the named
operation gaps before loading weights.

`DeferredPlan::layer_types` accepts `linear_attention` and
`deepseek_sparse_attention` only. An unknown attention label fails admission.

`DeferredPlan::ffn_layers` records `Dense` or `Experts` at each absolute layer
index. Decoder entries come first. Appended draft entries follow them.
`mlp_layer_types`, when present, must have one `dense` or `sparse` label per
decoder layer. A wrong length or unknown label fails.

Without a list, `first_k_dense_replace` sets the dense prefix. Its default is
zero. Later layers use experts when the config declares a positive
`n_routed_experts`, `num_experts` or `num_local_experts` count. Present count
fields must agree. The host router uses the same reader. An absent or null
count uses the list to establish whether experts exist. Without either source, all layers are dense.

The shared [`FfnSchedule`](../host_ops/schedule.rs) builder makes this plan.
An explicit list takes priority over the prefix for decoder layers. The list
does not include draft layers. Draft entries use the prefix rule at their absolute
indices, starting at `num_hidden_layers`. The draft tensor mapping remains
open. The execution schedule gap remains open too. Only deferred metadata
admission calls this planner. The Qwen config path does not call it.

`adapter::ConfigPlan::with_headers` checks the known shared expert mappings. It keeps
the stored name for recipe and file lookup. It also records the canonical
`shared_expert` name and the existing shared expert role. It does not add a
shared gate. Unmapped tensors remain in `unmapped_tensors`. This is a partial
header check, not a complete checkpoint validation.

The supported plan path reads the singular `mlp.shared_expert` names only.
The served code uses those names. A checkpoint with the plural
`mlp.shared_experts` names fails header reconciliation with a named error.
Only the deferred adapter maps the plural names.

Mixed metadata accepts `exclude_modules` without `ignore` and an absent or
null `kv_cache_quant_algo`. MXFP8 reads `group_size` `32` or
`weight_block_size` `[1,32]`. NVFP4 and W4A16_NVFP4 require `group_size` `16`
and refuse any `weight_block_size`. FP8 block storage reads `weight_scale` and
keeps its declared layout. The public reference and synthetic layouts are in
the [fixture notes](../../tests/fixtures/glm-plan/README.md).

An unknown scale codec remains a gap. It cannot fall back to plain weights.
An explicit exclusion or a more specific supported recipe still takes
priority. The new FP8 storage also needs an execution binding. Existing
executable schemas reject that storage until a binding is available.

The appended draft count is metadata. The local public source does not define
the full `eh_proj`, `enorm`, `hnorm` and `shared_head.norm` semantics. Those
tensors stay unmapped. No `mtp.*` alias is inferred from their names.

## Looped text plans

`description::ConfigPlan` has a third arm, `Loop(LoopTextDescription)`, for
`inkling2_loop` and `inkling2_loop_text`. [`inkling.rs`](inkling.rs) reads the
field names of the released config only. It refuses an unknown field and an
unknown value, and it names the field. The description gives the attention
data of each stored layer, the block mix, the loop range, the router, the
soft caps and the visit list. The visit list is the run order for one token.
A stored layer in the loop range occurs one time for each pass.

`LoopTextDescription` is a sibling of `TextDescription`. The text service and
`Cartridge::resident_text` do not receive it. `into_model_plan` always
refuses it. A parsed config is not support.

[`storage_groups.rs`](storage_groups.rs) reads `hf_quant_config.json`. This
sidecar lists tensor names in groups, and each group names one stored format.
`StorageGroups::storage_map` cross-checks the groups with `config.json` and
returns a `StorageMap` whose keys are full tensor names. A scale tensor is
not listed. `GroupKind::companions` gives its name, type and shape from the
stored shape of the listed tensor. `ScaleEncoding::AmaxDenominator` in
[`storage.rs`](storage.rs) marks 4-bit storage with one `amax` for each
matrix of a stack and one denominator. Header admission in
[`admission.rs`](admission.rs) refuses that encoding.

[`loop_plan.rs`](../../tests/loop_plan.rs) holds the host tests. The
[fixture notes](../../tests/fixtures/loop-plan/README.md) describe its data. Four layout facts
are named constants, and no run has checked them yet: `FIXED_TAP_ORDER`,
`TOKEN_TAP_ORDER` and `TOKEN_TAP_INDEX` in `inkling.rs`, and `NVFP4_NIBBLES`
and `GATE_UP_ROWS` in `storage_groups.rs`.

[`inkling_index.rs`](inkling_index.rs) gives each stored tensor name of this schema a role, a type and a
shape. `IndexInput::from_description` takes the sizes from `LoopTextDescription` and the stored form of each
listed name from `StorageGroups`. `IndexInput::bind` checks the header of the main file and, when it is
present, the header of the draft file `mtp.safetensors`. A name with no role, a missing name, a wrong type
and a wrong shape are refusals, so a bound index has no name without a role. The draft names are bound as
"not run". `model.visual.*` names are bound as not used. The index works out the scale tensors of an expert
stack from the config, and `GroupKind::companions` works them out from the stored shape. A test holds the
two equal.

A bound index is metadata. `BoundIndex::require_execution` and `BoundIndex::require_image_input` always
refuse. `plan::unserved_reason` gives the text that the launcher, `vade models` and
`Cartridge::resident_text` print for this schema: what was read and what is not built. It returns `None`
for every other schema. [`inkling_index.rs`](../../tests/inkling_index.rs) holds the host tests, and the
[fixture notes](../../tests/fixtures/inkling-index/README.md) describe their data.

[`loop_load.rs`](loop_load.rs) makes the load plan from a bound index and the headers of the weight files.
A device tensor has one of three forms. `Stored` keeps the bytes of one stored tensor. `LeadingRows` keeps
the first rows of the output head, because the head stores padded rows that no token uses.
`WidenedExperts` is one stack of expert matrices as BF16. A stack is made from its stored codes, its block
scales and, for 4-bit storage, the factor of each expert. A matrix that holds gate rows and up rows is
un-mixed: all gate rows, then all up rows. The image tower is not loaded. `expert_tensor` is the one place
that decides the device form of the experts.

[`loop_widen.rs`](loop_widen.rs) holds the arithmetic of the widen step. A 4-bit weight is the code value
times the block scale times the factor of the expert, rounded to BF16 one time. The factor is `amax`
divided by the denominator, and both are rounded to BF16 before the division, as the reference loader
does. An 8-bit weight is the stored value times a power of two, which is exact in BF16. This is the form
that the reference code runs. It is not the stored precision.

[`loop_packed.rs`](loop_packed.rs) holds the contract of an expert product that reads the stored codes and
their block scales, and its host twin. The product gives the words of the BF16 row product on the widened
matrix, so a change to this form moves no token ID. No GPU kernel reads this form yet.

[`loop_state.rs`](loop_state.rs) plans the state of one chat for each layer visit. A window visit keeps a
ring of `sliding_window_size` key rows and value rows. A visit that sees every earlier token keeps one row
for each token of the context, in whole launch buckets. Each visit keeps four convolution rings.
`LoopLedger` sums the device bytes of the weights and of one chat. `context_limit` gives the largest
context that fits in a byte budget, and `LoopLedger::admit` refuses a context that does not fit, with the
fields of the memory refusal of the text service. [`loop_load.rs`](../../tests/loop_load.rs) holds the host
tests of the three files. No service loads this family yet: these plans admit no execution.

## Text edges

`text_edges.rs` binds embedding, final norm, head and scale headers.
It retains stored dtypes and does not open executable model admission.

**How it fits:** see the [`engine-model` README](../../README.md) for what the crate does and how it connects to the rest.

The [appended MTP source gap](mtp-source-gap.md) records the missing GLM
definitions and the existing Qwen draft contracts.

## DSA plan data

`dsa::DsaPlan` describes a compressed-pool sparse attention indexer.
It reads explicit config fields. It does not read a model name or supply an
executable `ModelPlan`. It does not change Qwen plan code. See
[Qwen plan changes](#qwen-plan-changes) for this stack.

The source is `Glm5NextTextIndexer` in the public Transformers file
`src/transformers/models/glm5_next/modeling_glm5_next.py`.
The inspected local copy has SHA256
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
This identifies the source bytes, not a Git revision.
No public source file is copied here.

Let `C` be `hidden_size`, `Q` be `q_lora_rank`, `H` be `index_n_heads`,
`D` be `index_head_dim`, `P` be `index_kpool`, and `K` be `index_topk`.
The source constructor defines these logical shapes:

| Module suffix | Role | Shape |
| --- | --- | --- |
| `wq_b.weight` | Query projection | `[H * D, Q]` |
| `wk.weight` | Key projection | `[D, C]` |
| `weights_proj.weight` | Head weights | `[H, C]` |
| `k_norm.weight` | Key norm gain | `[D]` |
| `k_norm.bias` | Key norm bias | `[D]` |
| `index_kpool_compress_ape` | Pool position logits | `[P, D]` |
| `index_kpool_compress_gate` | Pool gate projection | `[D, C]` |

The caller supplies the module prefix to `with_headers`.
Binding checks names, shapes, dtypes and byte spans in that subtree.
It preserves stored names and offsets. It accepts unpacked BF16, F16 and F32
headers. These are supported logical storage types, not a claim about a
published checkpoint. Packed weights and scale companions fail closed.
Quantization recipes are not interpreted by this operation plan.

The source packs a cache row as `[key D, gate D, valid 1]`.
`layout` gives packed rows and upper bounds for compressed key, raw-index and
validity tensors. Their axes are `[batch, slot, channel]`,
`[batch, pool, channel]`, `[batch, pool, pool slot]`, and `[batch, pool]`.
The source removes pool slots that are invalid in every batch row.
The plan cannot determine that final pool count from config alone.
It does not select cache dtype or allocate a persistent compressed cache.

Pools begin at the first valid raw slot. A hole invalidates the complete pool.
Compression uses a softmax over pool slots for each channel. Its logits are
the gate projection plus APE. The source casts probabilities to key dtype
before the weighted sum. Query scores use the normalized MLA Q residual.
The source defines no indexer rotary transform.

The source score is the sum over heads of
`weights_proj(x)[head] / sqrt(H) * relu(dot(query_head, pool_key) / sqrt(D))`.
`query` returns eligible complete pools and tail indices before this score.
It does not rank candidates. A pool is eligible only when its last raw slot
is visible. The pool budget is `floor(K / P)`.
The plan stores selection geometry. Score arithmetic, key norm epsilon and
dtype casts remain outside the plan data.
The output width is `K + P - 1` with the tail, or `K` without it.
The source returns int32 indices with `-1` padding. These rules come from
`forward`, not its conflicting output-shape docstring.
The operation plan retains floor division for the synthetic reference cases.
It does not claim full config-schema validation.

The source tail starts at `first_valid + visible_valid_count - tail_count`,
where `tail_count = visible_valid_count % P`.
This can repeat a selected token when the mask has internal holes.
The source combines indices into a set for the sparse attention mask.
The source does not define tie order. `TieOrder::Unspecified` records that gap.
The host reference uses lower raw indices for ties as a separate Vade contract.

## Host proof and open work

[`dsa_plan.rs`](../../tests/dsa_plan.rs) reads a synthetic safetensors header.
Its reader rejects payload reads. The test binds weights by role and runs
the non-square reference shape through the plan. It imports the existing
[`dsa` host reference](../dsa.rs). Its arithmetic is unchanged.
It uses decoded f32 values.
It does not prove Torch dtype rounding or device behavior.

This operation plan stays separate from the deferred config metadata above.
It can be attached to that metadata in a later change.
The [header admission checks](../../HEADER-ADMISSION.md) now validate storage
dtypes, quantization companions, and codec metadata against shard headers.
They do not qualify device projection packing. Layer ownership, cross-layer
selection sharing, MTP sharing, sparse MLA integration and cache lifetime
remain outside this operation plan. The local source does not define
`index_share_for_mtp_iteration` or `indexer_rope_interleave` operations.
No such operation is inferred. An explicit false `index_kpool_compress`
is rejected because the source only defines compressed pools.

This change does not alter model execution, scheduling, transport or drafting.
Host tests do not establish token exactness or speed. Device integration still
needs its own source-backed plan and the device gate.

## Qwen plan changes

This stack changes these parts of the Qwen plan code:

- The supported mixed config path calls `Mixed::parse` in
  [`mixed.rs`](../flash/mixed.rs). It now accepts an absent
  `kv_cache_quant_algo`, `exclude_modules` without `ignore`, and MXFP8 with
  only `weight_block_size` `[1,32]`. It now refuses NVFP4 and W4A16_NVFP4
  recipes that contain `weight_block_size`. Previously, it ignored that field.
- Header reconciliation in [`tensors.rs`](tensors.rs) refuses plural
  `mlp.shared_experts` names with a named error. Singular names give the same plan.
- The mixed memory classes in [`mixed.rs`](../flash/mixed.rs) check block FP8
  storage first. The supported config path refuses block FP8 storage, so
  supported plans keep the same classes.

The Qwen plan byte tests and their baseline files stay unchanged.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
