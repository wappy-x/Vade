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

Tests check schema translation and name binding. These fixtures cannot prove
stored layout, model output, or device exactness.
