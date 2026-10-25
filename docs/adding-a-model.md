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

The source links below were checked at commit
`daf087d7adcd426c707a9073bfa99cd1ff4f297b`.

## 1. Find the entry path and the file list

The CLI catalog, [cli/src/catalog.rs:5](../cli/src/catalog.rs#L5), pins a
revision and the SHA-256 of every file for the legacy Qwen3-8B BF16 download.
It is not a general model registry. The command name is restricted in
[cli/src/cli.rs:161](../cli/src/cli.rs#L161) (only `vade run qwen3-8b` is accepted), and the download reads the
catalog in [cli/src/run.rs:7](../cli/src/run.rs#L7). Adding another catalog
entry means updating command selection and download as well. Changing the
file list alone doesn't add engine support.

The Flash-Next launcher doesn't download; it finds a checkpoint already on
disk. It checks the config, index and shard files in
[cli/src/launch/model.rs:251](../cli/src/launch/model.rs#L251). It runs the
engine's config parser and quantization-sidecar check at
[cli/src/launch/model.rs:346](../cli/src/launch/model.rs#L346), and checks the
generation and tokenizer metadata at
[cli/src/launch/model.rs:379](../cli/src/launch/model.rs#L379). From a
controller, the checkpoint's location and identity go in the deployment config;
see [the FP4-Local example](../cli/examples/fp4-local.conf). Record the
publisher's revision, the license and file hashes you trust. Keep weights and
credentials outside the source tree, and never commit them.

## 2. Add only the metadata support that is missing

[engine/crates/model/src/plan/mod.rs:57](../engine/crates/model/src/plan/mod.rs#L57)
reads `config.json`, `model.safetensors.index.json` and the shard headers,
with size limits. It then checks the tensor list without loading any weights.
The typed plan supports a fixed set of architecture adapters, not any
Transformers config.

For a family or storage format the typed plan doesn't know yet, start with
[plan/config.rs:115](../engine/crates/model/src/plan/config.rs#L115) for config
dispatch and geometry,
[plan/tensors.rs:187](../engine/crates/model/src/plan/tensors.rs#L187) for
names, shapes, dtypes and tied heads, and
[plan/runtime.rs:33](../engine/crates/model/src/plan/runtime.rs#L33) for the
projection limits execution needs. The older ordinary-attention path has its
own adapters in [cartridge.rs:31](../engine/crates/model/src/cartridge.rs#L31)
and [qwen3.rs:42](../engine/crates/model/src/qwen3.rs#L42). Not every family
goes through `ModelPlan`.

## 3. Bind execution and account for memory

Use [SplitRequest](../engine/crates/model/src/split/mod.rs#L12) and
[SplitPlan::new](../engine/crates/model/src/split/mod.rs#L71) for the number of
Sparks, tensor ownership, workspace reserves, cache and state, and drafting
budgets. Update [plan/profile.rs:8](../engine/crates/model/src/plan/profile.rs#L8)
when a new operation changes the profile accounting.

On the typed hybrid path, execution starts at
[engine/crates/engine/src/hybrid/forward.rs:15](../engine/crates/engine/src/hybrid/forward.rs#L15).
For Flash-Next, [engine/crates/tp/src/flash/settings.rs:124](../engine/crates/tp/src/flash/settings.rs#L124)
selects `NextModel` when the checkpoint has mixed storage, and the
`--fp4-local` flag refuses a checkpoint without it. Extend the loader and the
serving checks together. A new catalog label must never let through a shape,
format or topology that is otherwise refused.

## 4. Check text handling and serving

Read [engine/crates/tokenizer/src/chat.rs:9](../engine/crates/tokenizer/src/chat.rs#L9)
and the [template check at line 175](../engine/crates/tokenizer/src/chat.rs#L175).
What a template supports comes from the template itself, and an unknown
template is refused. Add a template or adapter only with its license and with
tests of prompts and token IDs, covering tools, thinking controls and
stopping.

Check generation defaults and request handling in
[engine/crates/serve/src/protocol.rs:190](../engine/crates/serve/src/protocol.rs#L190).
The FP4-Local route (`next_model`) decodes greedily when a request names no
sampling field ([flash-serve.rs:261](../engine/crates/tp/src/bin/flash-serve.rs#L261)).
The retained FP8 route ([cli/src/launch/model.rs:340](../cli/src/launch/model.rs#L340))
sets `next_model` to false ([settings.rs:134](../engine/crates/tp/src/flash/settings.rs#L134))
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
