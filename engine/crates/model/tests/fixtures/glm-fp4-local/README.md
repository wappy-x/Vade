# GLM FP4-Local metadata fixture

This fixture records a loader comparison. It does not add model support.
It contains no weights or tensor payload. The inspect path reads the header
as metadata. It does not open a model shard.

Run this command from `engine/`:

```sh
cargo +1.99.0 run --offline --locked -p engine-model --bin inspect-checkpoint -- --layout-fixture crates/model/tests/fixtures/glm-fp4-local
```

## Evidence and limits

The public source is `nvidia/GLM-5.3-Flash-FP4-Local` at revision
`f5a0baba30120da613e27973ee3535a725d4ece3`.

- [Config](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/f5a0baba30120da613e27973ee3535a725d4ece3/config.json)
- [Index](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/f5a0baba30120da613e27973ee3535a725d4ece3/model.safetensors.index.json)
- [Model card](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)

The saved config has SHA256
`dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8`.
The saved index has SHA256
`1981941b2fabb9271ae3a3cb76be6ea41927f730b6633471d3c147106703aeea`.
These hashes identify the public JSON sources, not this reduced fixture.

`config.json` is a hand-written subset of the public fields. It retains the
public values and schema gaps. It keeps `exclude_modules`, which the mixed
parser accepts without `ignore`. The parser also accepts an absent or null
`kv_cache_quant_algo`. The fixture omits both `ignore` and `kv_cache_quant_algo`.
It keeps the declared `weight_block_size` for the selected decoder MXFP8
modules. It does not replace that field with `group_size`.

All tensor names in the fixture index occur in the public index. The fixture
changes each shard name to `fixture.safetensors`. That file does not exist.
The fixture index size describes the virtual payload from the fixture header.
The public index contains names and shard paths. It has no shapes or dtypes.

Every dtype, shape, and offset in `fixture.safetensors.header.json` is test
data. No public shard header was read. The test shapes do not match the full
config dimensions. The CSF component sizes do not define a valid codec.
No CSF payload is provided or decoded. The actual component dtypes, ranks,
lengths, and byte order remain unknown.

The header uses the loader's raw NVFP4 and MXFP8 shape formulas for the
selected direct-scale examples. A focused test checks those synthetic
formulas. The inspect output still states
`logical-scale-pairing=not-validated`. It does not validate a complete model
plan or claim that the public checkpoint uses those exact stored shapes.

`audit.json` contains a curated comparison. The operation list comes from
the public config and tensor names. It is not automatic operation detection.
The list names missing operations and missing schema or tensor mappings
separately. Existing dense FFN and MTP primitives do not establish support
for this checkpoint's layer schedule or MTP schema.

## Storage map

For these formulas, `R` is the output dimension. `K` is the unpacked input
dimension. The formulas below come from the current
[mixed inventory](../../../src/flash/mixed.rs) and
[NVFP4 reference](../../../src/flash/nvfp4.rs).

| Current loader role | Name suffix | Dtype and shape |
| --- | --- | --- |
| Packed NVFP4 values | `weight` | U8 `[R,K/2]` |
| Direct NVFP4 block scales | `weight_scale` | E4M3 `[R,K/16]` |
| NVFP4 global multiplier | `weight_scale_2` | F32 scalar |
| Activation calibration | `input_scale` | F32 scalar; not applied to W4A16 weights |
| Current CSF fixed part | `weight_scale.nvfp4_csf_fixed` | U8 `[R/16,16*(1+(K/16)/2)]` |
| Current CSF exceptions | `weight_scale.nvfp4_csf_exceptions` | U32 vector |
| MXFP8 values | `weight` | E4M3 `[R,K]` |
| MXFP8 block scales | `weight_scale` | U8 UE8M0 bytes `[R,K/32]` |

The mixed inventory requires NVFP4 `K` to be divisible by `16`.
Its MXFP8 `K` must be divisible by `32`. The current CSF decoder requires
`R` to be divisible by `16` and `K` to be divisible by `32`.

The current NVFP4 reference computes `E2M1 * E4M3 * global`.
Its existing CSF expansion preserves E4M3 bytes. That codec is not evidence
for the GLM `csf_32x2_v1` encoding.

The public config declares `csf_32x2_v1` for decoder routed experts.
The public index stores `csf32_bases`, `csf32_codes`, `csf32_page_bases`,
`csf32_offsets`, and `csf32_exceptions` under `weight_scale`.
The mixed Flash loader does not read these names or this encoding.
The separate [GLM description](../../../GLM.md) maps CSF32 component names.
The host [CSF32 decoder](../../../CSF.md) expands the encoded scales.
These host operations do not add a mixed Flash execution binding.

The public config declares direct `W4A16_NVFP4` scales for MTP experts.
Their raw scale role has a loader counterpart. The MTP plan and names remain
unsupported. The public config declares MXFP8 with `weight_block_size`
`[1,32]` for the head. The mixed parser accepts this form as `Storage::Mxfp8`.
Recipe acceptance does not validate the public tensor layout.

The public shared-expert recipe is `FP8_PB_WO` with `weight_block_size`
`[128,128]`. The mixed parser accepts this recipe as `Storage::Fp8Block`.
The supported plan then returns `MissingOps` because the execution binding
is absent. The deferred adapter maps the plural `mlp.shared_experts` names.
This metadata mapping does not add execution support. Its public scale
name is `weight_scale`. The current standard FP8 inventory instead expects
`weight_scale_inv` with BF16 block scales. The public stored scale dtype
needs header evidence. The fixture's F32 scale is only test data.

## Work left

Obtain a public header manifest or approved header samples for every storage
class. Record the checkpoint revision with them. Verify packed dimensions,
scale dtypes, scalar ranks, CSF component lengths, and scale placement.
Obtain the `csf_32x2_v1` format specification and a small codec example.
Then test the conversion against decoded scale bytes.

The public config gives hidden size `4096` and routed FFN width `2048`.
These values imply logical expert gate/up shapes `[2048,4096]` and down
shape `[4096,2048]`. This inference does not establish stored shapes.
The current fixture tests inspection and gap reporting only.
