# Corrected sigmoid routing

The saved config selects `RouterKind::CorrectedSigmoid` from `scoring_func`
and `topk_method`. A model name does not select the operation. The typed
description and `host_ops::router::Router` use the same kind. The description
remains deferred. This change does not admit a GLM request for execution.

The operation source is Transformers `modeling_glm5_next.py`, SHA256
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
It was saved from an unpinned main branch. The line citations below refer to
that saved file. They do not claim a pinned upstream implementation.
The public source URL is
<https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py>.

The checkpoint values come from the saved
[config fixture](tests/fixtures/glm/config.json), `text_config`.
Its [source record](tests/fixtures/glm/README.md) identifies the checkpoint.
The config SHA256 is
`dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8`.
The fixture sets `n_routed_experts=288`, `num_experts_per_tok=8`,
`n_group=1`, `topk_group=1`, `norm_topk_prob=true`,
`routed_scaling_factor=2.5`, `n_shared_experts=1`,
`moe_router_dtype=float32`, and `dtype=bfloat16`.
These are fixture values, not kernel constants.

| Stage | Formula and order | Saved modeling source |
| --- | --- | --- |
| Projection | Cast hidden rows and router weights to FP32. Compute the linear projection without bias. | Lines 147-159 |
| Score | `s[e] = 1 / (1 + exp(-logit[e]))`, in FP32. | Line 160 |
| Correction | `c[e] = s[e] + correction_bias[e]`. Keep `s` unchanged. | Lines 155, 161 |
| Group score | Split consecutive expert IDs into equal groups. Add each group's largest corrected score to its next largest corrected score. This step also applies when all groups are kept. | Lines 162-166 |
| Group selection | Keep `topk_group` groups by group score. Mask every excluded expert with negative infinity. A zero mask is wrong when kept scores are negative. | Lines 167-175 |
| Expert selection | Select `num_experts_per_tok` experts globally within the kept groups. There is no per-group quota. | Line 176 |
| Tie and output order | The source uses unsorted top-k. It does not specify ties or slot order. Vade chooses descending corrected score, then lower ID. Group ties also choose lower ID. Signed zeros tie. | Lines 164, 167, 176; the fixed order is a Vade contract |
| Weight gather | Gather original `s[e]`, without correction bias. | Line 177 |
| Normalize | If configured, add the selected scores left to right in the fixed router-slot order. Add `1e-20` after the sum. Divide each selected score by that denominator. | Lines 178-180; the fixed reduction order is a Vade contract |
| Scale | Multiply each weight by `routed_scaling_factor` after optional division. Keep weights in FP32. | Line 181 |
| Routed FFN | Cap gate above at `swiglu_limit`. Cap up at both signed bounds. Apply SiLU, multiply by up, then apply down. | Lines 131-141 |
| Expert addition | Visit hit experts in ascending ID. Multiply the completed down row by its FP32 weight. Cast each contribution to the activation dtype. Add it to the activation-typed running result. The BF16 reference rounds each contribution and each addition. | Lines 122-134 |
| Shared addition | Compute the shared MLP from the original input. Add it once after the routed sum. Apply no shared gate and no route weight. Shared width is the configured shared count times expert width. | Lines 195-205; shared MLP lines 85-103 |

`Router::route` reuses the corrected sigmoid host reference from the prior
router work. Its hand vectors cover correction-only selection, group top-pair
scores, negative masks, ties, normalization, epsilon, scaling, and invalid
inputs. `Router::softmax` calls the existing Qwen FP32 host reference.
It adds no softmax arithmetic. The tests compare that path's weight bits.

`host_ops::moe::ungated_shared` accepts completed down rows in router-slot
order. It sorts slots by expert ID, keeps route weights in FP32, and applies
the supplied activation rounding at the source boundaries. Pass identity
for FP32 or round-to-nearest-even for BF16. It is a join reference. It does
not compute expert projections. Input validation defines the supported
reference domain; it is not a claim about upstream error handling.

The device payload has these entries:

- `corrected_sigmoid_route` consumes projected FP32 logits and correction
  bias. It writes u32 IDs, FP32 weights, and sticky row status. Independent
  lanes compute scores. A fixed serial selection and weight fold follow the
  block barrier. The exponential reuses `flash_moe::exp`.
- `ungated_moe_finish` consumes completed, rank-reduced down rows in
  router-slot order and a separate shared row. It sorts slots by expert ID.
  It applies the source BF16 boundaries and adds shared last. It reads no
  shared-gate tensor. Status rejects a row before any ID or value read.

The descriptors in `gpu/src/router_abi.rs` check dimensions, byte arithmetic,
alignment, and disjoint spans. Owners must also check allocation sizes,
contexts, and lifetimes. These entries are a device payload. Production
GLM execution is still deferred. Only device gate tests call the entries.

The current Qwen softmax router, fixed finish, shared scheduling, transport,
and expert kernels are unchanged. Its finish rounds route weights to BF16
and uses an FP32 expert fold. Its shared addition also has a sigmoid gate.
Those operations cannot implement the source table above.

NVFP4 decoding and projection primitives remain reusable where their dtype,
shape, and reduction order agree. `fp4_exact.rs` supplies scalar ordered
projection arithmetic. `fp4_experts.rs` supplies BF16 tensor-core expert
arithmetic for its supported layout. Neither supplies the configured
activation cap. Their current grouped paths also have separate layout
constraints. This payload does not claim full GLM expert support or replace
those kernels. Header binding, capped activation, projection order, grouped
layout, and complete model execution still need their own gates.

The device byte tests are in `flash::router_tests`. They cover host hand
vectors, repeated rows, row partitions, ties, rejected rows, sticky status,
and poisoned output tails. The finish tests cover expert order, BF16
boundaries, FP32 weights, and ungated shared-last addition. They are ignored
on a host without an authorized CUDA device. A host test does not prove
device bytes, token IDs, or speed. The device exponential also does not
claim bit parity with a particular PyTorch or CUDA library.

Run the byte gate on an authorized CUDA host from `engine/`, after building
kernels:

```sh
cargo +1.99.0 test --offline --locked -p spark-engine --lib flash::router_tests:: -- --ignored --nocapture --test-threads=1
```

Before production integration, repeat the device gate with captured graph
replay. Then run the repository token-ID gate across supported Spark counts,
drafting modes, batches, cache states, and prompt chunks. This host job
makes no device pass or performance claim.
