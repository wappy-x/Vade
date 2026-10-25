# Adding a checkpoint

Read [AGENTS.md](../AGENTS.md) first. A new checkpoint is supported only after
its metadata, tensor storage, execution, tokenizer, memory admission and serving
behavior have been implemented and qualified. Parsing its config is not proof
that it can run. The release scope is text-only; the vision tower is out of scope.

Model-generic means deriving behavior from checkpoint config, safetensors
metadata and validated template semantics. Architecture adapters may recognize
declared `model_type` and `architectures`; do not dispatch on a repository name,
directory name or marketing label. No model-specific tuning, copied dimensions,
per-checkpoint kernel constants or performance exceptions. Reject unsupported
semantics explicitly.

The source pointers below were verified on main at
`ed6898b4c285250aff83be002056b7cd0fa22071`.

## 1. Identify the entry path and checkpoint inventory

The CLI catalog is [cli/src/catalog.rs:5](../cli/src/catalog.rs#L5): a pinned
revision and SHA-256 inventory for the legacy official Qwen3-8B BF16 download.
It is not a general model registry. Its command name is restricted in
[cli/src/cli.rs:158](../cli/src/cli.rs#L158), and acquisition consumes that
inventory in [cli/src/run.rs:6](../cli/src/run.rs#L6). Adding another downloadable
catalog entry would also require updating command selection and acquisition;
changing the inventory alone does not add engine support.

The Flash-Next launcher discovers an existing local checkpoint instead. It
checks config, index and shard paths in
[cli/src/launch/model.rs:235](../cli/src/launch/model.rs#L235), uses the engine's
config parser and quantization-sidecar detector at
[cli/src/launch/model.rs:332](../cli/src/launch/model.rs#L332), and checks
generation/tokenizer metadata at
[cli/src/launch/model.rs:365](../cli/src/launch/model.rs#L365). For the controller
route, the checkpoint location and identity belong in the deployment config;
see [the FP4-Local example](../cli/examples/fp4-local.conf). Record the official
revision, licence and trusted file hashes. Keep weights and credentials outside
the source tree; never commit them.

## 2. Implement only missing metadata semantics

[engine/crates/model/src/plan/mod.rs:57](../engine/crates/model/src/plan/mod.rs#L57)
reads bounded `config.json`, `model.safetensors.index.json` and shard headers,
then reconciles the tensor inventory without loading payloads. The typed plan
supports explicit architecture adapters, not arbitrary Transformers configs.

For a missing typed-plan family or storage format, inspect
[plan/config.rs:115](../engine/crates/model/src/plan/config.rs#L115) for config
dispatch and geometry, [plan/tensors.rs:187](../engine/crates/model/src/plan/tensors.rs#L187)
for names, shapes, dtypes and tied-head reconciliation, and
[plan/runtime.rs:33](../engine/crates/model/src/plan/runtime.rs#L33) for executable
projection constraints. The existing ordinary-attention path has separate
adapters in [cartridge.rs:32](../engine/crates/model/src/cartridge.rs#L32) and
[qwen3.rs:42](../engine/crates/model/src/qwen3.rs#L42); do not assume every family
passes through `ModelPlan`.

## 3. Bind execution and account for memory

Use [SplitRequest:12](../engine/crates/model/src/split/mod.rs#L12) and
[SplitPlan::new:71](../engine/crates/model/src/split/mod.rs#L71) for topology,
tensor ownership, workspace reserves, cache/state and drafting budgets.
Update [plan/profile.rs:8](../engine/crates/model/src/plan/profile.rs#L8) when a
new operation changes the profile accounting.

For the typed hybrid path, execution starts at
[engine/crates/engine/src/hybrid/forward.rs:15](../engine/crates/engine/src/hybrid/forward.rs#L15).
For Flash-Next, [engine/crates/tp/src/flash/settings.rs:120](../engine/crates/tp/src/flash/settings.rs#L120)
selects `NextModel` from mixed storage; the FP4-Local flag asserts that contract.
Extend the applicable loader and serving capability checks together. A new
catalog label must never bypass a rejected shape, format or topology.

## 4. Verify text and serving behavior

Inspect [engine/crates/tokenizer/src/chat.rs:9](../engine/crates/tokenizer/src/chat.rs#L9)
and [template validation at line 175](../engine/crates/tokenizer/src/chat.rs#L175).
Supported template semantics come from template data; unknown templates fail
closed. Add a template/adapter only with its licence and prompt/token-ID tests,
including tools, thinking controls and stopping behavior. Check generation
defaults and request handling in
[engine/crates/serve/src/protocol.rs:189](../engine/crates/serve/src/protocol.rs#L189):
The mixed FP4-Local / `next_model` route uses greedy decoding when sampling
fields are absent ([flash-serve.rs:261](../engine/crates/tp/src/bin/flash-serve.rs#L261));
the retained FP8 route ([cli/src/launch/model.rs:340](../cli/src/launch/model.rs#L340))
selects `next_model = false` ([settings.rs:131](../engine/crates/tp/src/flash/settings.rs#L131))
and applies checkpoint generation sampling defaults instead. Naming a sampling
field changes how checkpoint defaults are applied.

## 5. Qualify before advertising support

From `engine/`, build kernels first as described in [AGENTS.md](../AGENTS.md),
then run these host tests and the full local-validation block there:

```sh
cargo +1.99.0 test --offline -p engine-model --lib plan::tests
cargo +1.99.0 test --offline -p engine-model --lib split::tests
cargo +1.99.0 test --offline -p engine-tokenizer --lib
```

Host tests cover metadata, shape rules, layouts, budgets and rejection paths.
Checkpoint/device tests require a Spark and separately supplied weights; a Mac
host pass does not qualify inference. Establish identical token IDs across
runs and Sparks with the same setup, including alone/batched/cached/chunked
requests and drafting ON/OFF. Compare numerical quality against the official
checkpoint separately from repeatability within Vade. Qualify each advertised
topology and precision rather than generalizing a result from another path.

See [validation](../engine/docs/validation.md),
[CLI gate limitations](../cli/docs/test.md) and
[benchmark methodology](benchmarks.md). Record source/artifact identities,
checkpoint hashes, complete request settings and raw receipts. The existing
Flash `flash-bench` determinism recipe is not evidence that a new FP4-Local
checkpoint has passed its gate.
