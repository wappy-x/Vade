# Adding a checkpoint

Read [AGENTS.md](../AGENTS.md) first.

Vade supports a new checkpoint only after these parts are implemented and qualified: its metadata, tensor storage, execution, tokenizer, memory admission and serving behavior. A parsed config does not prove that the checkpoint can run. The release is text only, so a vision tower is out of scope.

Model-generic code derives its behavior from three sources: the checkpoint's config, its safetensors metadata and the checked template semantics. An architecture adapter may recognize the declared `model_type` and `architectures`. These rules apply:

- **No name dispatch.** Do not dispatch on a repository name, a folder name or a marketing label.
- **No model-specific tuning.** Do not add model-specific tuning, copied dimensions, per-checkpoint kernel constants or performance exceptions.
- **Clear refusals.** Reject anything unsupported with a clear error.

## 1. Find the entry path and the file list

The CLI catalog, [cli/src/catalog.rs](../cli/src/catalog.rs), pins a revision and the SHA-256 of every file for the legacy Qwen3-8B BF16 download. It is not a general model registry. [cli/src/cli.rs](../cli/src/cli.rs) restricts the command name: it accepts only `vade run qwen3-8b`. The download reads the catalog in [cli/src/run.rs](../cli/src/run.rs). So another catalog entry also needs changes to the command selection and the download. A change to the file list alone does not add engine support.

The Flash-Next launcher does not download. It finds a checkpoint that is already on disk. [cli/src/launch/model.rs](../cli/src/launch/model.rs) does three checks:

1. It checks the config, index and shard files.
2. It runs the engine's config parser and the quantization-sidecar check.
3. It checks the generation and tokenizer metadata.

From a controller, the checkpoint's location and identity go in the deployment config. See [the FP4-Local example](../cli/examples/fp4-local.conf). Record the publisher's revision, the license and the file hashes you trust. Keep weights and credentials outside the source tree, and never commit them.

[`vade pull`](pull-models.md) downloads a checkpoint from the Hub and runs the same launcher checks on it. A pass there is not qualification. See section 5.

## 2. Add only the metadata support that is missing

[engine/crates/model/src/plan/mod.rs](../engine/crates/model/src/plan/mod.rs) reads `config.json`, `model.safetensors.index.json` and the shard headers, with size limits. Then it checks the tensor list without loading any weights. The typed plan supports a fixed set of architecture adapters. It does not support any Transformers config.

For a family or storage format that the typed plan does not know yet, start with these files:

- **[plan/config.rs](../engine/crates/model/src/plan/config.rs).** This file holds the config dispatch and geometry.
- **[plan/tensors.rs](../engine/crates/model/src/plan/tensors.rs).** This file holds the names, shapes, dtypes and tied heads.
- **[plan/runtime.rs](../engine/crates/model/src/plan/runtime.rs).** This file holds the projection limits that execution needs.

The older ordinary-attention path has its own adapters in [cartridge.rs](../engine/crates/model/src/cartridge.rs) and [qwen3.rs](../engine/crates/model/src/qwen3.rs). Not every family goes through `ModelPlan`.

The `glm5_next` text family does not go through `ModelPlan` either. [plan/description.rs](../engine/crates/model/src/plan/description.rs) sends its config to [plan/glm.rs](../engine/crates/model/src/plan/glm.rs), which returns a `TextDescription`. [plan/glm_index.rs](../engine/crates/model/src/plan/glm_index.rs) gives each tensor name a role.

## 3. Bind execution and account for memory

Use [SplitRequest](../engine/crates/model/src/split/mod.rs) and [SplitPlan::new](../engine/crates/model/src/split/mod.rs) for the number of Sparks, the tensor ownership, the workspace reserves, the cache and state, and the drafting budgets. Update [plan/profile.rs](../engine/crates/model/src/plan/profile.rs) when a new operation changes the profile accounting.

Main has two engines for current work, and they do not share execution code. Flash-Next is in [engine/crates/engine/src/flash/](../engine/crates/engine/src/flash/README.md). The `glm5_next` text engine is in [engine/crates/tp/src/bounded_decoder.rs](../engine/crates/tp/src/bounded_decoder.rs) and the files around it. The [module map](../engine/docs/module-map.md) follows one request through each engine. A checkpoint with a different architecture needs its own execution code of this kind. Read both request paths before you start.

On the typed hybrid path, execution starts at [engine/crates/engine/src/hybrid/forward.rs](../engine/crates/engine/src/hybrid/forward.rs).

For Flash-Next, [engine/crates/tp/src/flash/settings.rs](../engine/crates/tp/src/flash/settings.rs) selects `NextModel` when the checkpoint has mixed storage. The `--fp4-local` flag refuses a checkpoint without mixed storage. Extend the loader and the serving checks together. A new catalog label must never let through a shape, format or topology that is otherwise refused.

## 4. Check text handling and serving

Read [engine/crates/tokenizer/src/chat.rs](../engine/crates/tokenizer/src/chat.rs) and the [template check](../engine/crates/tokenizer/src/chat.rs). The template itself sets what the template supports, and Vade refuses an unknown template. Add a template or adapter only with its license. Also add tests of prompts and token IDs that cover tools, thinking controls and stopping.

The GLM chat renderer and its output parser are in the serve package: [tokenizer/glm.rs](../engine/crates/serve/src/tokenizer/glm.rs) and [chat_output/glm.rs](../engine/crates/serve/src/chat_output/glm.rs).

Check the generation defaults and the request handling in [engine/crates/serve/src/protocol.rs](../engine/crates/serve/src/protocol.rs).

- **FP4-Local.** The FP4-Local route (`next_model`) decodes greedily when a request names no sampling field ([flash-serve.rs](../engine/crates/tp/src/bin/flash-serve.rs)).
- **FP8.** The retained FP8 route ([cli/src/launch/model.rs](../cli/src/launch/model.rs)) sets `next_model` to false ([settings.rs](../engine/crates/tp/src/flash/settings.rs)). It uses the checkpoint's own sampling defaults instead.

On either route, one sampling field in a request changes how the checkpoint defaults apply.

## 5. Qualify it before you call it supported

From `engine/`, build the kernels first, as [AGENTS.md](../AGENTS.md) describes. Then run these host tests, and then the full local validation block in AGENTS.md:

```sh
cargo +1.99.0 test --offline -p engine-model --lib plan::tests
cargo +1.99.0 test --offline -p engine-model --lib split::tests
cargo +1.99.0 test --offline -p engine-tokenizer --lib
```

Host tests cover metadata, shape rules, layouts, budgets and refusals. Checkpoint and device tests need a Spark and the weights. So a pass on a Mac does not qualify inference.

Show identical token IDs across runs and Sparks with the same setup. Show them alone, batched, cached and chunked, with drafting on and off. Compare quality against the publisher's checkpoint as a separate step from repeatability inside Vade. Qualify every topology and precision that you advertise. A result on one path does not carry over to another.

See [validation](../engine/docs/validation.md), [the limits of the CLI gate](../cli/docs/test.md) and [how we benchmark](benchmarks.md). Record the source and build identities, the checkpoint hashes, the complete request settings and the raw receipts. The older `flash-bench` determinism recipe is not evidence that a new FP4-Local checkpoint has passed its gate.
