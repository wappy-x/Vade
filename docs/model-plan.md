# Model plan

`ModelPlan` turns a checkpoint's metadata into a typed description of the
model: its layers, dimensions, tensor roles and storage formats. It reads
`config.json`, the safetensors index and each shard's header. It never reads
the weights themselves, and it allocates no GPU memory.

The code lives in [engine-model](../engine/crates/model/src/plan/mod.rs). The
typed [hybrid loader](../engine/crates/engine/src/hybrid/forward.rs) calls
`inspect_parts`, so it loads from the same index and shard headers the plan
was built from.

## How to call it

| Call | What it does |
| --- | --- |
| `ModelPlan::from_config` | Parses config bytes. The tensor list stays incomplete until headers are added |
| `ModelPlan::with_headers` | Checks tensor names, shapes, dtypes and roles against the plan |
| `ModelPlan::inspect` | Reads the config, index and shard headers from a checkpoint folder |
| `ModelPlan::inspect_parts` | Does the same, and also returns the index and shard headers it read |

The [config adapters](../engine/crates/model/src/plan/config.rs) define which
model families are supported, and reject settings they don't understand.
[Tensor reconciliation](../engine/crates/model/src/plan/tensors.rs) checks the
stored tensors, including tied embeddings and mixed-format metadata. A plan can
describe BF16, block-scaled FP8 and mixed-format storage. That doesn't mean
every serving path can run every format.

## What a valid plan proves

A valid plan shows that the metadata and shapes agree. It doesn't show that
the weight files match their published hashes, that the tokenizer and chat
template are supported, that the model fits in memory, or that the GPU
arithmetic is qualified. Installation, runtime binding and device validation
check those.

[Runtime profiles](../engine/crates/model/src/plan/profile.rs) and
[projection records](../engine/crates/model/src/plan/runtime.rs) connect the
checked metadata to what execution needs. The [splitter](splitter.md) uses the
checked shape to plan which Spark owns what.

## Run its tests

From `engine/`:

```sh
cargo +1.99.0 test --locked --offline -p engine-model --lib plan::tests -- --nocapture
```

Two tests need a real checkpoint and are skipped by default. To run them, set
`VADE_FLASH_CONFIG` to a checkpoint's `config.json` and `VADE_FLASH_CHECKPOINT`
to its folder, then add `--ignored` after the `--`. The config test expects a
block-FP8 Flash config. A host pass without them says nothing about a
checkpoint you downloaded.

See [adding a model](adding-a-model.md) for the other checks a new model needs.
