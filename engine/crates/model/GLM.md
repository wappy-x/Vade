# Deferred text metadata

`plan::ConfigPlan::from_config` reads the GLM text schema beside the existing
Qwen adapter. It returns `Deferred(TextDescription)` for `glm5_next` and
`glm5_next_text`. The root vision config is ignored. Qwen still uses
`ModelPlan::from_config` without a change to that parser.

The description carries operation data. It has KDA dimensions, latent
attention dimensions, pooled-indexer settings, Sinkhorn settings, routing
settings, an activation bound, and the complete EOS list. Layer order comes
from the config lists. The explicit MLP list has priority over the source
default `first_k_dense_replace`. The redundant KDA and full-attention lists
must agree with the attention list.

The [SwiGLU reference](SWIGLU.md) defines the bound and dtype boundaries.
It supplies activation data for a future binding. GLM execution stays deferred.

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
ignored. The current Qwen storage parser and device handoff are unchanged.

[`csf::Csf32`](CSF.md) expands the new scale layout on the host. It uses the
existing NVFP4 arithmetic after byte expansion. It is separate from the
legacy CSF byte layout. The comparison test in
[`tests/csf32.rs`](tests/csf32.rs) checks both layouts against the same scale
bytes and FP32 results.

The config declares appended prediction layers. The index reader labels
their stored tensors as draft data. It does not define fusion order, head
sharing, or a draft forward pass. `ConfigPlan::into_model_plan` refuses the
deferred description before loading or execution.

Run the metadata inspector from `engine/`:

```sh
cargo +1.99.0 run --offline --locked -p engine-model --example glm-metadata -- \
  /path/to/config.json /path/to/model.safetensors.index.json \
  /path/to/hf_quant_config.json
```

The inspector never opens a shard. Its printed byte count comes from the
index. It is not a measurement of resident memory. Kernel work, real header
checks, tokenizer checks, and device exactness remain open.

The public metadata source and its saved revision are in the
[fixture record](tests/fixtures/glm/README.md). The operation source is the
saved Transformers `modeling_glm5_next.py` file. Its fetch used an unpinned
branch. This description does not claim a pinned operation implementation.

The [routing math table](ROUTING.md) defines the corrected sigmoid kind,
fixed tie order, host references, and device payload. The description checks
group capacity and the FP32 routing scale. Device execution remains deferred.
