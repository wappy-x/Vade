# Appended MTP source gap

The reviewed GLM source does not define the appended MTP layer. Keep its
tensor names unmapped until a public source defines the operations. A tensor
name or matching shape does not prove that an existing draft role is correct.

## Source evidence

The reviewed file is `tf_modeling_glm5_next.py`. Its SHA256 is
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
Line references below refer to that file.

- Line 1379 ignores `layers\.45\.` and `layers\.\d+\.shared_head\.`
  during loading. This is an ignore rule, not a mapping rule.
- Lines 1435 to 1437 construct only `config.num_hidden_layers` decoder layers.
- Lines 1497 to 1510 run those layers and the target output normalization.
- A search for `eh_proj`, `enorm`, `hnorm`, `shared_head`, `nextn`, `mtp`,
  `MTP`, and `predict` finds only the ignore rule at line 1379.

| Requested name or area | Evidence still required |
| --- | --- |
| `eh_proj` | Definition, input order, input width, and output width. |
| `enorm` | Definition, normalized input, width, and norm convention. |
| `hnorm` | Definition, hidden-state input, width, and norm convention. |
| `shared_head.norm` | Definition, output position, width, and sharing rules. |
| Appended layer body | Constructor, layer schedule, stored names, and relation to the decoder body. |

The decoder class does not establish the appended layer body. It selects
attention and FFN types from config at lines 1282 to 1291. It defines
`attn_hc` and `ffn_hc` at lines 1297 to 1298. Its attention class defines
low-rank projections at lines 1127 to 1141. These definitions do not prove
that an appended layer uses the current direct Q/K/V draft plan.

## Existing Qwen contracts

The [tensor plan](tensors.rs) gives body tensors
`Place::Draft { layer: Some(id) }`. It gives wrapper tensors
`Place::Draft { layer: None }` with `Part::Whole`.

Let `H` be the configured hidden width. Let `W` be the residual stream width.
The tensor plan derives both values from config.

| Layout | Wrapper tensors and logical shapes |
| --- | --- |
| Hybrid | `mtp.fc.weight` has shape `[H, 2H]`. `mtp.norm.weight`, `mtp.pre_fc_norm_embedding.weight`, and `mtp.pre_fc_norm_hidden.weight` have shape `[H]`. |
| Flash | `mtp.fc_embedding.weight` and `mtp.fc_hidden.weight` have shape `[H, H]`. The embedding norm has shape `[H]`. The hidden norm has shape `[W]`. The mixer also has hyper-connection tensors. |

The [hybrid draft implementation](../../../engine/src/hybrid_mtp.rs) places
normalized embeddings before normalized hidden states in the concat input.
The [typed loader contract](../qwen3_5_plan.rs) requires BF16 concat storage
and a non-indexed attention draft block.
The [Flash draft implementation](../../../engine/src/flash/session/mtp.rs)
applies separate projections to the embedding and residual streams.
It then adds their outputs.

The GLM source does not establish either fusion rule. Do not choose a layout
from the model name, or replace the appended layer prefix with `mtp.layers`
without a source definition. Keep stored names for file and storage-recipe
lookup when a verified mapping becomes available.

Synthetic header checks can preserve the existing Qwen contracts and reject
unverified names. They cannot prove GLM MTP semantics, checkpoint storage, or
device exactness.
