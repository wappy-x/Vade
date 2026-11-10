# Stored MXFP8 projection candidate

The binder follows a declared producer in the deferred layer program.
It reads the declared storage recipe and the admitted component headers.
It preserves complete input features and the topology's selected complete output rows.
It reads only those weight rows and their matching scale rows through retained file descriptors.
The checkpoint files must stay immutable.

The [binder](src/mxfp8_projection.rs) accepts E4M3 weights and U8 UE8M0 scales.
The [format decoder](../model/src/flash/mxfp8.rs) defines the scale group size.
The binder refuses ordinary weights and other quantized recipes.
It also refuses router outputs, unresolved input dtypes, and incomplete feature rows.
Norm output precision follows the stored parameter dtype.
Other input precision follows the conditional layer boundary contract.
This prepares operands for query, key, value, low-rank, and output projections.
It does not construct missing attention producers or complete their gathers.

## Dynamic activation contract

The [dynamic adapter](src/dynamic_projection.rs) rounds source activations to BF16.
It then quantizes complete feature groups to E4M3FN with unsigned E8M0 scales.
The [activation reference](../gpu/src/dynamic_mxfp8.rs) defines scale selection and nearest-even rounding.
Its group size and finite range come from the numeric format.
They do not depend on a model name, prompt, batch size, or topology.
Scale selection uses the smallest finite power of two that fits the group's maximum magnitude.
Zero and subnormal groups use the minimum scale when required.

The bounded adapter stages activations through the host quantizer.
It copies the exact decoded quantized values back to the owning device stream.
The existing projection kernel consumes these FP32 lanes and the original stored weights.
It keeps its fixed reduction tree and BF16 epilogue.
The adapter adds no unsafe site or alternate reduction path.
This implementation makes no speed claim.

Run `checkpoint-prefix-check attention-projections suite MODEL INPUT_IDS NEW_OUT PARTS`.
The probe selects the contiguous norm-fed MXFP8 projections before the next attention operation.
It saves the prefix, packed activations, scales, decoded activations, accumulators, and output rows.
It compares device bytes with the CPU reference and fresh device runs.
It joins selected output rows against the unsplit CPU reference.
Use `attention-projections compare-suite LEFT RIGHT` for the cross-Spark comparison.
The probe writes the next operation to `NEXT_CELL`.
It does not execute that operation or generate output token IDs.

The [attention binder test](tests/attention_projections.rs) covers both mixer programs.
It checks complete matrix rows after norm, low-rank, gated-norm, and value producers.
These synthetic tests do not establish a complete layer or a device pass.

## Stored-identity diagnostic contract

The activation type follows the norm's stored parameter type.
A BF16 parameter produces exact BF16 values in FP32 lanes.
An FP32 parameter produces FP32 values, including its residual bits.
The [device owner](../gpu/src/backend/mxfp8_projection.rs) checks the completed norm output variant before launch.
It does not narrow FP32 input or insert an activation quantizer.

The existing `flash_next_mxfp8_m1` kernel consumes the original weight and scale bytes.
Its fixed reduction uses ascending feature tiles and fused FP32 multiply-add operations.
The final lane reduction follows the kernel's fixed down-shuffle tree.
Each token uses that same kernel and complete feature row.
The launch dimensions do not select a different reduction order.

The accumulator stays FP32 for inspection.
The existing BF16 boundary kernel rounds the accumulator once with round-to-nearest-even.
The result is a local output row interval with typed BF16 lanes.
The declared gather must complete before a later operation uses a replicated output.

The [host reference](../gpu/src/mxfp8_projection.rs) checks these candidate contracts.
It rejects nonfinite operands, incomplete features, incomplete scale groups, and BF16 output overflow.
The tests cover FP32 input residuals, BF16 ties, scale boundaries, row selection, and the fixed reduction tree.

## Checkpoint proof remains open

The stored-identity command remains a diagnostic with no activation quantizer.
It cannot stand for the dynamic activation contract.
The dynamic adapter defines the native activation boundary for measurement.
It does not prove parity with a separate runtime backend.
Effective parameter loads also remain unqualified.
The scalar runtime profiles and service admission keep their existing refusals.
Complete layers and generated token IDs remain unproved.

## Byte comparison probe

Use `checkpoint-prefix-check projection suite MODEL INPUT_IDS NEW_OUT PARTS`.
The [suite source](src/bin/checkpoint-prefix-check/projection.rs) bounds the input ID count and defines the repeat count.
It tests every topology rank in fresh device processes on each authorized Spark.
It saves the complete stored prefix, projection accumulator, and BF16 output bytes.
Each device run must match its host reference and the prior device run.
The selected output rows must join to the unsplit host reference.
This file comparison does not implement transport or a device gather.

Use `checkpoint-prefix-check projection compare-suite LEFT RIGHT` to compare the complete Spark result trees.
The comparator checks source and operand identities, geometry, input IDs, scope, manifests, and every plane.
The suite does not start a service or generate tokens.
Build the probe on the host and ship the executable with its checksum.
