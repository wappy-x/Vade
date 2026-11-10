# Host operation references

[`router.rs`](router.rs) defines fixed corrected sigmoid selection and reuses
the existing softmax reference. [`moe.rs`](moe.rs) defines the source-order
expert fold and ungated shared addition. The [math table](../../ROUTING.md)
records the source lines, dtype boundaries, and device gates.

These operations do not admit a deferred model for execution. See the
[metadata description](../../GLM.md).

## Feed-forward config references

`host_ops::Router` is the config reference in `ffn_router.rs` below.
It keeps the config defaults and optional-bias API from the reference series.
`host_ops::router::Router` is the separate description-bound reference above.
It requires explicit corrected-routing selectors when it reads config data.
The references keep their own input and tie-order contracts. Neither is a
device execution path.

Select a reference by its input and rounding contract before device wiring.
The following references are not interchangeable oracles.

| Public reference | Contract |
| --- | --- |
| `host_ops::Router` | Config defaults and optional correction bias. Softmax normalization follows its config. |
| `host_ops::router::Router` | Typed routing description or explicit corrected selectors. The softmax constructor reuses the existing normalized reference. |
| `host_ops::Swiglu` | Logical f32 vectors, host SiLU exponential, and no BF16 rounding boundaries. |
| `swiglu::SwiGlu` | Scalar epilogue with explicit FP32 or BF16 boundaries. The default exponential is `flash::nvfp4::exp_ref`. |

Use the description-bound router to check typed routing data. Use
`swiglu::SwiGlu` to check a declared epilogue precision. The config vector
references supply host formula checks only. The deferred layer scalar forward
has its own logical FP32 formulas. These checks do not establish device bits.

These references read operation fields from a text config or its `text_config`
object. They do not read `model_type`. They do not admit a model, load weights,
or change a device path. The current Qwen router and its tests stay unchanged.

| File | Input and result |
| --- | --- |
| [ffn_router.rs](ffn_router.rs) | Projected logits and a correction bias produce expert IDs and weights. |
| [swiglu.rs](swiglu.rs) | Projected gate and up vectors produce the capped activation. |
| [schedule.rs](schedule.rs) | Config fields produce layer kinds by absolute layer index. |

## Router order

Corrected routing uses `scoring_func: "sigmoid"` and `topk_method: "noaux_tc"`.
The router also supports `softmax` with `greedy`. Missing selectors select
that pair. Expert and selection counts are required. Group counts default to
one. Normalization defaults to true. Routed scaling defaults to one.
The public config source maps `num_local_experts` to `n_routed_experts`.
The router and the schedule share one reader for those names and `num_experts`.
A null count is absent. Present counts must agree.

The operation follows `Glm5NextTextTopkRouter` in the modeling source below:

1. Compute the sigmoid score for each expert.
2. Add `e_score_correction_bias` to form selection scores only.
3. Divide experts into contiguous groups. Add the largest corrected score
   in each group to the next largest corrected score.
4. Select groups by descending group score. Exclude experts in other groups.
5. Select experts by descending corrected score across the kept groups.
6. Gather the original sigmoid scores for weights.
7. If requested, sum those weights and add `1e-20`. Divide each weight by
   that sum. Apply `routed_scaling_factor` last.

All arithmetic uses f32. Equal group scores prefer the lower group ID.
Equal expert scores prefer the lower expert ID. Scores use `f32::total_cmp`,
which orders positive zero before negative zero in descending order.
Returned experts follow descending corrected score, then lower ID. The weight
sum follows this order from left to right. The group sum adds the largest
score to the next largest score. These orders do not depend on a batch.

The source uses unsorted top-k and does not define tie order. The order above
is this host reference's contract. It does not claim device library bit parity.
The saved public config labels the formula with `sigmoid` and `noaux_tc`.
The modeling source defines that formula without a method switch. Other
selector pairs remain unsupported except for the softmax pair above.
Positive finite scales are a supported-input restriction of this reference.
The public source does not enforce that restriction.

## Cap and layer schedule

SwiGLU accepts an absent `hidden_act` or the value `silu`.
An absent or null `swiglu_limit` keeps the uncapped operation.
With a cap, use the following order.
Clamp gate above at the limit. Clamp up between the negative and positive
limit. Apply SiLU to the clamped gate, then multiply by the clamped up value.
The gate has no lower bound. Each element uses f32 in ascending index order.
The existing `refcpu::silu` helper computes `gate * (1 / (1 + exp(-gate)))`.
The multiply by up follows that result. This does not claim PyTorch bit parity.
The host checks require a positive finite cap and finite inputs.
The multiply can overflow f32. The operation keeps that result.

The schedule requires `num_hidden_layers`. An explicit `mlp_layer_types`
list must have that length. `dense` selects a dense block. `sparse` selects experts.
This is a layer-kind reference. It does not validate projection geometry.
`FfnSchedule::from_config` gives decoder layers only. The deferred plan
adapter uses the crate-internal `from_object`. It appends draft layers after
the decoder layers at absolute indices.

Without a list, `first_k_dense_replace` selects a dense prefix. Later layers
use experts when the config has a positive expert count. An absent, null, or
zero expert count produces dense layers. This existing fallback is separate
from the explicit-list rule in the GLM source.

The GLM modeling source reads only
the explicit list. The configuration source supplies a fixed model default
when the list is absent. It does not derive the list from this field. The
modular source removes the field with `AttributeError()`. This reference does
not copy that model default. An explicit list takes priority over the prefix.
The prefix must still be a valid count within the decoder layer count.

## Local public evidence

The source files below are public source copies. They are not build inputs.
The code cites Transformers commit `f90e93c4f83974f634860e246cdfd2c68666d605`.
The hashes identify the local source text used here. No receipt proves that
this text matches that commit.

| File | SHA256 |
| --- | --- |
| `tf_modeling_glm5_next.py` | `9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87` |
| `tf_configuration_glm5_next.py` | `5dbd107e664aac78b5b33eb33e238dba62d8482135aa77799a56aedbca1ce959` |
| `tf_modular_glm5_next.py` | `f52ebd8d8db9a6606706b0b0144df7bb208373d8b8094c9f3e9d2d6d8e7884b2` |
| `config.json` | `dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8` |

Evidence locations in those files:

- Modeling lines 143-182 define the router. The normalization epsilon is at
  line 179. The group score uses the largest pair at lines 162-166.
- Modeling lines 85-104 and 136-141 define the cap in dense and expert blocks.
- Modeling lines 1290-1292 select layer kinds from `mlp_layer_types`.
- Configuration line 98 defines the expert-count alias. Lines 33-34 define
  the list values. Lines 159-162 define the fixed model default list.
- Modular line 174 removes `first_k_dense_replace`.
- Saved public config lines 3229 and 3233 label the router operation.

## Proof and remaining device work

The unit tests use small configs and hand vectors. They check correction-only
selection, group sums, masked negative scores, ties, normalization, scaling,
the epsilon, clamp order, and explicit layer kinds. They also check malformed
config data, bad vector lengths, and non-finite inputs.

The device path still needs config and tensor-plan bindings for the router
projection and correction bias. It needs corrected group selection and weight
generation with an agreed fixed order. It needs the cap in dense, shared, and
routed expert projections. It needs per-layer allocation and dispatch from
the explicit schedule. The existing generic prefix fallback does not prove
GLM source behavior. Other model-admission gaps stay open.

Host tests do not establish GPU exactness or speed. Device integration needs
the full authorized exactness gate. That gate must preserve current Qwen token
IDs and compare the affected run modes and supported topologies.

**Back:** [Model source](../README.md)
