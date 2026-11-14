# Public GLM metadata fixture

`config.json` is the saved public config for
`nvidia/GLM-5.3-Flash-FP4-Local`. The fetch record names repository revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. It records a fetch on
2026-11-01. No weight payload is included.

Source:
[config.json](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/resolve/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/config.json).
The checkpoint card names the NVIDIA Open Model Agreement.

`index-patterns.txt` is a name summary of the saved public index. `N` marks
a layer index. `E` marks an expert index. Each row starts with the number of
matching names. The last field is a shard count from the source summary.
This file contains no tensor dtypes or shapes.

Source:
[model.safetensors.index.json](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/resolve/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/model.safetensors.index.json).

Fixture SHA256 values from the job's `shasum -a 256` output:

| File | SHA256 |
| --- | --- |
| config.json | dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8 |
| index-patterns.txt | 95b096c7ddf033295002ee0d8471e930d5cc73459fa79ff788827785b3b06bca |
| public-header-samples.json | 5df5ee3bfdf191ebc838c2fd962d8d0ef4a65b437252846558e1fd86450c99b6 |


Tests check schema translation and name binding. These fixtures cannot prove
stored layout, model output, or device exactness.

`public-header-samples.json` contains metadata copied from public safetensors
headers at the same repository revision. Bounded HTTP range reads fetched
only the length prefixes and header JSON. No weight payload was fetched.
The source files are the shard names in the public index linked above.
The samples cover a linear dense layer, a latent expert layer, the stored
appended layer, and the model embedding, norm, and head.

Dtypes, shapes, and data offsets are unchanged from those headers. Each
sample's offsets refer to its original shard. They do not describe a single
combined file. The admission unit test checks the complete component groups
in this sample. It does not read scale payloads or prove device behavior.

`bounded-headers.json` is a synthetic complete text inventory for
[`header_admission.rs`](../../header_admission.rs). Its dimensions are set by
that test's description. Its byte ranges are contiguous. The test supplies
synthetic payload bytes. It covers plain, MXFP8, FP8 block weight-only,
direct NVFP4, and CSF32 NVFP4 storage. It is not a saved model payload.
