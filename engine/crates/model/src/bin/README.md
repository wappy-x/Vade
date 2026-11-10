# engine/crates/model/src/bin

**Programs built from the `engine-model` crate, one per file.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`inspect-checkpoint.rs`](inspect-checkpoint.rs) | Check a checkpoint, or print a header fixture and its written loader gap map. |
| [`inspect-tensor-file.rs`](inspect-tensor-file.rs) | The `inspect-tensor-file` program. |

**Why a folder:** Cargo builds every file in a `bin/` folder as its own program.

## Inspect a layout fixture

From `engine/`, run:

```sh
cargo +1.99.0 run --offline --locked -p engine-model --bin inspect-checkpoint -- \
  --layout-fixture crates/model/tests/fixtures/glm-fp4-local
```

This mode reads `config.json`, `model.safetensors.index.json`, `audit.json`,
and one `<shard>.header.json` file for each shard in the index. It does not
open a weight file. The normal safetensors reader checks the header shapes,
byte offsets, and dtypes against a declared file length. No payload exists.
The inspector also checks that the index and audit cover every header tensor.

The audit is data. Its `source` and `scope` fields state the evidence limits.
Its `operations` rows contain `name`, `evidence`, and `gap` text. Its `tensors`
object uses each exact tensor name as a key. Each tensor row contains `role`,
`loader`, and `gap` text. These rows are a written review, not detected engine
capabilities. The inspector quotes them in a fixed order.

The output includes the current model-plan and mixed-storage parser results.
It also checks each storage recipe in isolation. That check supplies the
outer fields for a mixed ModelOpt config, including a null
`kv_cache_quant_algo`. That field is optional in the parser. It does not change the
fixture config or create a load plan. Recipe acceptance does not qualify
the tensor layout or the model for execution.

The [GLM fixture](../../tests/fixtures/glm-fp4-local/README.md) uses public names
and a small synthetic header. Its shapes are not observed model shapes.
The fixture is not a checkpoint. A successful command means that the fixture
is valid. It does not mean that the model can run.

**How it fits:** see the [`engine-model` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
