# Checkpoint layer reference

The `layer-reference` example saves a logical FP32 reference from a local checkpoint.
It does not run CUDA or admit a model for service.
Its schedule comes from [`reference.rs`](src/plan/layer/reference.rs).
That schedule is separate from the layer plan builder.
The [runtime type map](RUNTIME-TYPES.md) separates scalar values from unqualified runtime producers.
Every weight source declares its profile. Scalar entry points reject an unsupported profile before execution.

Run from `engine/`:

```sh
cargo +1.99.0 run --release --offline --locked -p engine-model --example layer-reference -- \
  generate MODEL NEW_OUT 9703,1879
cargo +1.99.0 run --release --offline --locked -p engine-model --example layer-reference -- \
  verify NEW_OUT
cargo +1.99.0 run --release --offline --locked -p engine-model --example layer-reference -- \
  compare EXPECTED ACTUAL
```

The example needs existing local checkpoint files. It does not fetch weights.
Use the prebuilt executable on a Spark. Do not build it there.

## Inputs and outputs

The ID list is a caller argument. No prompt selects engine behavior.
The tool accepts 1 through 16 IDs. It requires a KDA first decoder.
The argument checks in `layer-reference.rs` define these tool limits.
An error can leave an unsealed output folder. Keep it and choose a new path for a retry.
The example selects the first decoder and the first KDA decoder with experts.
The checkpoint config selects their dimensions and storage recipes.
Each layer receives the stored embedding row, widened from BF16 to FP32.
Each residual stream receives the same row.
Each layer starts with zero history and recurrent state.
That layer carries its state across the supplied IDs.
The layers are isolated tests. The expert layer does not receive earlier decoder outputs.

The tool decodes plain BF16 and FP32, MXFP8, block FP8, and NVFP4 with direct or CSF32 scales.
It reads selected expert matrices only. It checks headers for every logical expert weight.
It does not quantize activations or insert checkpoint BF16 activation casts.
Stored activation scale scalars have identity receipts. They do not change this FP32 profile.
The [layer plan](LAYER-PLAN.md) states the scalar reduction and operation order.

Each safetensors file holds one tensor named `value`.
Its payload holds contiguous row-major, little-endian values.
`TENSORS.tsv` gives its name, dtype, shape, payload byte count, and file SHA256.
The comparator checks each file header against that manifest before it compares payload bytes.
U32 planes retain exact integer IDs. F32 planes retain signed zero.
The files include inputs, intermediate outputs, layer outputs, history, and recurrent state.
`WEIGHTS.tsv` identifies each consumed payload range and its SHA256.
Offsets are relative to each safetensors data section.
Embedding receipts identify only the selected rows.
`PROFILE.txt` identifies config bytes, index bytes, IDs, and profile rules.
`SHA256SUMS` covers every saved file except itself.
These receipts identify consumed bytes. They do not verify every checkpoint shard against a public digest.

## Error and byte rules

The `logical-f32-v1` comparison has maximum absolute error 0 and maximum relative error 0.
`POLICY.txt` records those limits. The comparator also requires identical tensor bytes.
It rejects nonfinite floats, changed policies, missing files, and changed weight or input identity.
It prints the observed maximum absolute error before it refuses changed finite values.
Signed zero and integer IDs must also match.

This is a strict rule for the stated scalar profile.
Its exponentials use the host math library through `f32::exp`.
Identical bytes are expected for the same binary and host math environment.
The Mac oracle does not qualify cross-OS math parity with a Spark.
Keep such differences as diagnostic results. Do not widen the limit to hide them.
It is not a qualified error band for an unspecified CUDA backend.
Repeated processes on each rank must have identical bytes.
Ranks must compare tensors in global row order after the normal layer gathers.
The repeat count comes from the [exactness gate](../../../AGENTS.md#exactness-law-and-merge-gate).

NVFP4 weight decode computes `E2M1 * E4M3 * global`.
The first product is exact in FP32. The stored FP32 global multiplier causes one rounding step.
`FP4-BANDS.tsv` measures its absolute, relative, and ULP error against the exact FP64 product.
The source is the [NVFP4 decoder](src/flash/nvfp4.rs).
This rounding bound is half one FP32 ULP for finite results under nearest-even rounding.
It is a weight decode bound. It is not a decoder output bound.

Original quantization loss occurs before these stored bytes exist.
An FP4 code does not identify the original weight within its quantizer interval.
The lowest interval also has no useful uniform relative error bound near zero.
Norms, gates, recurrence, and expert selection can change the resulting decoder error.
A percentage for stored FP4 weights therefore cannot set a decoder output limit.
The original prequantized weights and a complete backend contract are absent from this profile.

## M2 device gate

A scalar reference pass does not prove a device layer pass.
The device owner must bind the full decoder, checkpoint casts, physical state, and ordered rank gathers.
The owner must state whether the device implements this exact scalar profile or another source-backed profile.
Any nonzero numerical band needs evidence for that specified profile.
Do not increase this comparator's limit to accept an unexplained mismatch.
Keep the separate repeat and cross-rank byte rule exact.

The prebuilt scalar tool can check checkpoint decode and host arithmetic on each target.
Such a run does not exercise the GPU, tensor split, or transport.
No `M2 PASS` follows from that run.
