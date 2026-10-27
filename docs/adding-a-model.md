# Adding a checkpoint

Read [AGENTS.md](../AGENTS.md) first. Vade supports a new checkpoint only after
its metadata, tensor storage, execution, tokenizer, memory admission and
serving behavior are implemented and qualified. Parsing its config doesn't
prove it can run. The release is text-only, so a vision tower is out of scope.

Model-generic code derives its behavior from the checkpoint's config, its
safetensors metadata and the checked template semantics. An architecture
adapter may recognize the declared `model_type` and `architectures`. Don't
dispatch on a repository name, a folder name or a marketing label. Don't add
model-specific tuning, copied dimensions, per-checkpoint kernel constants or
performance exceptions. Reject anything unsupported with a clear error.

## 1. Find the entry path and the file list

The CLI catalog, [cli/src/catalog.rs](../cli/src/catalog.rs), pins a
revision and the SHA-256 of every file for the legacy Qwen3-8B BF16 download.
It is not a general model registry. The command name is restricted in
[cli/src/cli.rs](../cli/src/cli.rs) (only `vade run qwen3-8b` is accepted), and the download reads the
catalog in [cli/src/run.rs](../cli/src/run.rs). Adding another catalog
entry means updating command selection and download as well. Changing the
file list alone doesn't add engine support.

The Flash-Next launcher doesn't download; it finds a checkpoint already on
disk. It checks the config, index and shard files in
[cli/src/launch/model.rs](../cli/src/launch/model.rs). It runs the
engine's config parser and quantization-sidecar check at
[cli/src/launch/model.rs](../cli/src/launch/model.rs), and checks the
generation and tokenizer metadata at
[cli/src/launch/model.rs](../cli/src/launch/model.rs). From a
controller, the checkpoint's location and identity go in the deployment config;
see [the FP4-Local example](../cli/examples/fp4-local.conf). Record the
publisher's revision, the license and file hashes you trust. Keep weights and
credentials outside the source tree, and never commit them.

## 2. Add only the metadata support that is missing

[engine/crates/model/src/plan/mod.rs](../engine/crates/model/src/plan/mod.rs)
reads `config.json`, `model.safetensors.index.json` and the shard headers,
with size limits. It then checks the tensor list without loading any weights.
The typed plan supports a fixed set of architecture adapters, not any
Transformers config.

For a family or storage format the typed plan doesn't know yet, start with
[plan/config.rs](../engine/crates/model/src/plan/config.rs) for config
dispatch and geometry,
[plan/tensors.rs](../engine/crates/model/src/plan/tensors.rs) for
names, shapes, dtypes and tied heads, and
[plan/runtime.rs](../engine/crates/model/src/plan/runtime.rs) for the
projection limits execution needs. The older ordinary-attention path has its
own adapters in [cartridge.rs](../engine/crates/model/src/cartridge.rs)
and [qwen3.rs](../engine/crates/model/src/qwen3.rs). Not every family
goes through `ModelPlan`.

## 3. Bind execution and account for memory

Use [SplitRequest](../engine/crates/model/src/split/mod.rs) and
[SplitPlan::new](../engine/crates/model/src/split/mod.rs) for the number of
Sparks, tensor ownership, workspace reserves, cache and state, and drafting
budgets. Update [plan/profile.rs](../engine/crates/model/src/plan/profile.rs)
when a new operation changes the profile accounting.

On the typed hybrid path, execution starts at
[engine/crates/engine/src/hybrid/forward.rs](../engine/crates/engine/src/hybrid/forward.rs).
For Flash-Next, [engine/crates/tp/src/flash/settings.rs](../engine/crates/tp/src/flash/settings.rs)
selects `NextModel` when the checkpoint has mixed storage, and the
`--fp4-local` flag refuses a checkpoint without it. Extend the loader and the
serving checks together. A new catalog label must never let through a shape,
format or topology that is otherwise refused.

## 4. Check text handling and serving

Read [engine/crates/tokenizer/src/chat.rs](../engine/crates/tokenizer/src/chat.rs)
and the [template check](../engine/crates/tokenizer/src/chat.rs).
What a template supports comes from the template itself, and an unknown
template is refused. Add a template or adapter only with its license and with
tests of prompts and token IDs, covering tools, thinking controls and
stopping.

Check generation defaults and request handling in
[engine/crates/serve/src/protocol.rs](../engine/crates/serve/src/protocol.rs).
The FP4-Local route (`next_model`) decodes greedily when a request names no
sampling field ([flash-serve.rs](../engine/crates/tp/src/bin/flash-serve.rs)).
The retained FP8 route ([cli/src/launch/model.rs](../cli/src/launch/model.rs))
sets `next_model` to false ([settings.rs](../engine/crates/tp/src/flash/settings.rs))
and uses the checkpoint's own sampling defaults instead. On either route,
naming even one sampling field changes how the checkpoint defaults apply.

## 5. Qualify it before you call it supported

From `engine/`, build the kernels first as [AGENTS.md](../AGENTS.md) describes.
Then run these host tests, followed by the full local validation block there:

```sh
cargo +1.99.0 test --offline -p engine-model --lib plan::tests
cargo +1.99.0 test --offline -p engine-model --lib split::tests
cargo +1.99.0 test --offline -p engine-tokenizer --lib
```

Host tests cover metadata, shape rules, layouts, budgets and refusals.
Checkpoint and device tests need a Spark and the weights, so a pass on a Mac
doesn't qualify inference. Show identical token IDs across runs and Sparks
with the same setup: alone, batched, cached and chunked, with drafting ON and
OFF. Compare quality against the publisher's checkpoint as a separate step
from repeatability inside Vade. Qualify every topology and precision you
advertise; a result on one path doesn't carry over to another.

See [validation](../engine/docs/validation.md),
[the limits of the CLI gate](../cli/docs/test.md) and
[how we benchmark](benchmarks.md). Record the source and build identities,
checkpoint hashes, complete request settings and raw receipts. The older
`flash-bench` determinism recipe is not evidence that a new FP4-Local
checkpoint has passed its gate.
