# Model plan

`ModelPlan` reads bounded checkpoint metadata and returns a typed description
of layers, dimensions, tensor roles and storage. It reads safetensors headers,
not weight payloads, and does not allocate GPU memory.

The implementation is in [engine-model](../engine/crates/model/src/plan/mod.rs).
The typed [hybrid loader](../engine/crates/engine/src/hybrid/forward.rs) uses
`inspect_parts`, preserving the exact inspected index and shard metadata for
loading. This code is part of the current tree.

## How to call it

| Call | Input and result |
| --- | --- |
| `ModelPlan::from_config` | Parses config bytes; tensor inventory still awaits headers |
| `ModelPlan::with_headers` | Reconciles names, shapes, dtypes and roles against the plan |
| `ModelPlan::inspect` | Reads bounded config, index and shard headers from a checkpoint directory |
| `ModelPlan::inspect_parts` | Also returns the exact inspected index and shard metadata |

[Config adapters](../engine/crates/model/src/plan/config.rs) define supported
families and reject unsupported semantics. [Tensor reconciliation](../engine/crates/model/src/plan/tensors.rs)
checks stored inventory, including tied embeddings and mixed-format metadata.
The plan describes supported BF16, block-scaled FP8 and mixed-format storage;
this does not imply every format is executable on every serving path.

## Limits

A valid plan establishes metadata and shape consistency. It does not establish
that weight bytes match their official hashes, that the tokenizer/template is
supported, that memory admission succeeds, or that device arithmetic is qualified.
Those checks belong to installation, runtime binding and native validation.

[Runtime profiles](../engine/crates/model/src/plan/profile.rs) and
[projection records](../engine/crates/model/src/plan/runtime.rs) connect checked
metadata to the execution requirements. The [splitter](splitter.md) plans
ownership and budgets from the reconciled shape.

## Run its tests

From `engine/`:

```sh
cargo test --offline -p engine-model --lib plan::tests -- --nocapture
```

Checkpoint-dependent tests use `VADE_FLASH_CONFIG` and `VADE_FLASH_CHECKPOINT`.
Read the test output for skipped native inventory checks; a host pass without
those inputs does not validate a downloaded checkpoint.

See [adding a model](adding-a-model.md) for the remaining integration checks.
