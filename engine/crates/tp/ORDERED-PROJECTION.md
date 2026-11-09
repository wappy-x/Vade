# Plain FP32 projection operand

This part supplies an isolated native operand producer for `logical-f32-v1`.
It does not qualify checkpoint runtime loads or admit a complete layer.
The [runtime map](../model/RUNTIME-TYPES.md) retains those open cells.

## Written operation contract

The [scalar layer reference](../model/src/plan/layer/reference.rs) defines `project` and `dot`.
Each output row starts with positive zero.
Each feature produces a separate FP32 product.
The row adds that product in ascending feature order.
The operation inserts no BF16 cast, activation quantizer, bias, or epilogue.
The [layer plan](../model/LAYER-PLAN.md#buffers-and-ownership) retains complete input features on each rank.
The [reference profile](../model/M2-REFERENCE.md#error-and-byte-rules) requires exact bytes for this logical operation.

The G6 source at `8719ddb304c1778c154bb7441a758e0121c4cbf9` refuses plain FP32 projections.
Its `RESIDENT-OPERANDS.md` binds ordinary BF16 projections only.
This part fills the plain FP32 primitive boundary under the existing scalar contract.
It does not choose the checkpoint runtime's effective parameter dtype or matrix reduction.

## Retained file and native binding

[`PlainF32Projection`](src/ordered_projection.rs) accepts an ordinary projection from a valid logical layer plan.
The physical header must match the logical matrix and declare plain FP32 storage.
The binding borrows the validated `TensorFile` descriptor.
It computes the declared output-row interval and retains every input feature.
The existing arena loader copies those original bytes without conversion.
The existing typed `Matrix` view retains the allocation after the local arena handle drops.

`Profile::ConditionalBf16` and `Profile::CheckpointRuntime` return typed profile errors before plan or header access.
Quantized storage, BF16 weights, unsupported operations, and mismatched headers also return typed errors.
Effective parameter load policy remains the first runtime gap.
Do not bypass that refusal by relabeling runtime values as logical reference values.

[`Backend::project_f32_ordered`](../gpu/src/backend/ordered_projection.rs) accepts the loaded matrix and owned device vectors.
It checks context, storage, shape, complete row layout, exact byte extents, alignment, address overflow, and output overlap.
It refuses segmented vectors because virtual addresses alone do not prove disjoint physical pages.
The generic `matvec_device` dispatch uses this same FP32 entry for plain FP32 matrices.
Its existing BF16 branches retain their arithmetic.
The synchronous wrapper completes submitted work before its borrowed operands can leave the call.
It also waits after a failed launch attempt.

The [native entry](../gpu/src/ordered_projection_kernel.rs) assigns each output row to one thread.
Explicit `mul.rn.f32` and `add.rn.f32` instructions prevent product fusion and reduction reassociation.
The entry preserves FP32 subnormals and adds no flush instruction.
Its sole unsafe block has a local safety contract.
The block still needs source review and the repository's unsafe allowlist ruling.

The primitive does not certify finite output values.
Callers must check arithmetic results before they use the scalar reference's finite tensor API.
Local projection rows require the layer's declared gather before any replicated consumer.
This part binds no transport completion, session state, or resident stack loop.

## Small native proof

The [probe](src/bin/ordered-projection-check.rs) uses synthetic retained checkpoint files.
The build host supplies the expected bytes.
The fixture checks cancellation, separate products, subnormal values, short rows, output tails, and replicated low-rank rows.
Device runs load the same file bytes through the production arena loader.
Each run saves actual bytes before comparison and writes a completion marker only after success.
Repeat comparison checks markers, topology, row metadata, complete extents, and bytes.
Split comparison requires each rank once, in order, and compares complete input-row projections against the full run.

Run the host oracle from `engine/`:

```sh
cargo +1.99.0 run --offline --locked -p spark-tp --bin ordered-projection-check -- host NEW_EXPECTED
```

Use only the shipped executable on a device:

```sh
./ordered-projection-check device EXPECTED NEW_OUT PARTS RANK
./ordered-projection-check compare RUN_A RUN_B
./ordered-projection-check compare-split FULL_RUN RANK_RUN...
```

Host tests cover shape, dtype, bounds, format refusal, retained descriptors, scalar projection bytes, and incomplete proof files.
Their synthetic rank comparisons provide no device result.
The native kit can establish only this isolated projection's byte behavior.
Real checkpoint M2 intermediates, complete layer M2, full logits M3, and token exactness remain open.
