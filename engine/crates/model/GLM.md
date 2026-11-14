# Deferred text metadata

The crate has separate description and partial-header APIs:

| API | Deferred result | Purpose |
| --- | --- | --- |
| `plan::ConfigPlan` (`plan::description::ConfigPlan`) | `TextDescription` | Validate operation fields and map index names. |
| `plan::adapter::ConfigPlan` | `DeferredPlan` | Check partial shared-expert headers and list execution gaps. |

Both APIs read `glm5_next` and `glm5_next_text`. They are distinct types and
validate different fields. The description ignores the root vision config.
The adapter records a vision gap when that config is present. It does not
validate the vision fields. Neither deferred result can enable execution.

Main runs the checkpoint through a separate path: the diagnostic text service
in `engine/crates/tp`. It starts only with `--diagnostic-unqualified-text`, on
exactly two Sparks, with a loopback HTTP bind. It is not qualified. See
[the text service](../tp/RESIDENT-TEXT-SERVICE.md) and
[the module map](../../docs/module-map.md#follow-one-glm-request). Normal
startup keeps the refusal below.

`TextDescription::require_resident_service` keeps service admission closed.
Its success type is `Infallible`. No metadata flag can supply that value.
The description's `into_model_plan`, cartridge preflight, and executable-kind
detection use this refusal. Cartridge inspection keeps the same refusal reason.
Preflight rejects the description before it reads a weight index or tokenizer.
The supported cartridge paths keep their existing config and weight checks.

The refusal names the absent complete resident decoder and service Factory.
Individual primitives and complete headers do not supply complete producers.
Layer, text, causal state, and ordered transport bindings remain open.
Source precision, casts, quantizers, and reductions still need qualification.
Tokenizer, chat template, stop set, appended prediction semantics, and token-ID
proof remain open. This boundary adds no request policy or execution kind.
The [host refusal tests](tests/service_admission.rs) cover this distinction.

`ModelPlan::from_config` now calls `plan::adapter::ConfigPlan::from_config`
and then `into_model_plan`. GLM returns `MissingOps` at that boundary.
Qwen continues to use `plan::config` through the supported adapter branch.

The description carries operation data. It has KDA dimensions, latent
attention dimensions, pooled-indexer settings, Sinkhorn settings, routing
settings, an activation bound, and the complete EOS list. Layer order comes
from the config lists. The explicit MLP list has priority over the source
default `first_k_dense_replace`. The redundant KDA and full-attention lists
must agree with the attention list.

The [SwiGLU reference](SWIGLU.md) defines the bound and dtype boundaries.
It supplies activation data for a future binding. Normal GLM execution stays
deferred; only the diagnostic text service runs the checkpoint.

`TextDescription::map_index` reads index JSON only. It maps names to
`TensorRole` and storage components. It also supplies the existing `Role`
for common placements and expert projections. New attention and residual
roles use `Part::Whole` as a conservative metadata label. They must not enter
the current splitter as executable shapes. Vision uses `Place::Unused`.
Unknown text names, out-of-range indices, and wrong scale components fail.
The method does not check inventory completeness, dtypes, or shapes.

`StorageMap` reads the config or sidecar recipe. It keeps MXFP8, FP8 block
weights, direct NVFP4 scales, legacy CSF, and CSF32 distinct. It applies
exclusions before the longest module-prefix match. Vision recipes are
ignored. The shared mixed Flash parser now accepts `exclude_modules`, an
absent or null `kv_cache_quant_algo`, MXFP8 `weight_block_size`, and
`FP8_PB_WO` metadata. These changes also apply to Qwen config parsing.
Existing supported plans refuse `FP8_PB_WO` with `MissingOps` before device
handoff. See [the mixed parser tests](src/plan/mixed_tests.rs) and
[the Qwen plan changes](src/plan/README.md#qwen-plan-changes).

[`plan::admission`](HEADER-ADMISSION.md) checks actual shard headers against
the required operation inventory. It checks physical shapes, dtypes, scale
components, scalar ranks, and byte extents. It supplies host byte views for
a later device upload. It keeps execution deferred.

[`csf::Csf32`](CSF.md) expands the new scale layout on the host. It uses the
existing NVFP4 arithmetic after byte expansion. It is separate from the
legacy CSF byte layout. The comparison test in
[`tests/csf32.rs`](tests/csf32.rs) checks both layouts against the same scale
bytes and FP32 results.

The config declares appended prediction layers. The index reader labels
their stored tensors as draft data. It does not define fusion order, head
sharing, or a draft forward pass. Both `ConfigPlan::into_model_plan` methods
refuse deferred metadata before loading or execution. The description API
returns `Unsupported`; the adapter returns `MissingOps` with named gaps.

The DSA, KDA, MLA and mHC config readers use `shapes::CONFIG_BYTES` as their
byte limit. The mHC host
reference and description use `shapes::MAX_SINKHORN_ITERS` as the iteration
ceiling. This is a host resource limit, not a checkpoint default or a change
to the configured iteration count. See [the mHC tests](src/mhc/tests.rs).

Run the metadata inspector from `engine/`:

```sh
cargo +1.99.0 run --offline --locked -p engine-model --example glm-metadata -- \
  /path/to/config.json /path/to/model.safetensors.index.json \
  /path/to/hf_quant_config.json
```

The inspector never opens a shard. Its printed byte count comes from the
index. It is not a measurement of resident memory. Kernel and tokenizer
qualification and device exactness remain open.

The [logical layer program](LAYER-PLAN.md) binds the operation descriptions
to revision-qualified kernel targets. It has buffer shapes, tensor splits,
an independent FP32 host forward, and an updated operation gap table.
It preserves deferred device admission.

The public metadata source and its saved revision are in the
[fixture record](tests/fixtures/glm/README.md). The operation source is the
saved Transformers `modeling_glm5_next.py` file. Its fetch used an unpinned
branch. This description does not claim a pinned operation implementation.

The [routing math table](ROUTING.md) defines the corrected sigmoid kind,
fixed tie order, host references, and device payload. The description checks
group capacity and the FP32 routing scale. Device execution remains deferred.
