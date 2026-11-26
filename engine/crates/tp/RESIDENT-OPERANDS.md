# Resident layer operands

The [operand binder](src/layer_operands.rs) adds candidate device operands to the
[deferred layer boundary](LAYER-DRIVER.md).
It does not implement a checkpoint Factory or a complete decoder.
The full layer still refuses execution.

## Norm and residual owners

`prepare_stored_norm_checkpoint` reads the retained checkpoint descriptors.
It accepts an ordinary RMSNorm operation with a plain parameter vector.
It widens stored BF16 values without loss.
It preserves stored FP32 values.
It rejects nonfinite values before device allocation.
It does not select the effective upstream loaded parameter dtype.
The stored-precision identity load remains a candidate policy.
The [cast contract](../model/CAST-CONTRACT.md) retains the source gaps.

`ResidentNorm` reuses the existing model RMS and pointwise kernels.
The reduction follows the fixed feature tree in those kernels.
The normalized value rounds to BF16 before the parameter multiply.
A BF16 parameter gives a BF16 product.
An FP32 parameter keeps its FP32 product.
This tree does not claim equality with the scalar layer reference or an upstream framework.

`prepare_residual_block_checkpoint` binds adjacent mHC prepare, collapse, and norm operations.
The same method serves the attention side and the feed-forward side.
It checks the operation data flow and hidden width.
The resident pass retains the residual and coefficients until the block producer finishes.
`finish` consumes that pass and a typed BF16 block output.
No host activation read occurs between these resident operations.
Each submitted operation completes before its temporary owners can drop.

## Plain projections

`prepare_bf16_projection_checkpoint` binds an ordinary BF16-output projection.
The parameter must have plain BF16 storage.
The binding retains complete input features and the plan's output-row interval.
The native-layout upload permutes weight bytes without conversion.
It uses the unblocked row fragments that the native projection reads.
It pads the final row tile with zeros.
The existing native BF16 projection supplies the device arithmetic.
A final typed copy preserves the public BF16 owner invariant.
This copy needs an extra allocation and launch. No speed claim follows.

The adapter returns local output rows.
The layer's declared gather must finish before a replicated consumer uses them.
The adapter does not bind transport or publish session state.
It refuses router FP32 projections, stored FP32 matrices, and quantized matrices.
Those operations need separate checked bindings and source evidence.

## Prebuilt operand probe

The [resident-layer-check binary](src/bin/resident-layer-check.rs) uses a small
synthetic checkpoint with retained file descriptors.
The [fixture](src/bin/resident-layer-check/fixture.rs) supplies each dimension and parameter.
The probe binds KDA and latent-attention layer descriptions with dense and routed feed-forward descriptions.
It executes their residual prefixes and mHC completion with a supplied synthetic block row.
That row does not come from an attention or expert implementation.
Sparse projection fixtures copy a selected input feature.
They permit an exact projection check without an assumed arbitrary-weight reduction tree.
Separate norm cases check odd widths, multiple reduction chunks, and parameter precision.

Run the host oracle on the build host:

```sh
cargo run --offline --locked -p spark-tp --bin resident-layer-check -- host NEW_EXPECTED
```

Run only the prebuilt executable on a device:

```sh
./resident-layer-check device EXPECTED NEW_OUT RANK PARTS
./resident-layer-check compare RUN_A RUN_B
./resident-layer-check compare-split FULL_RUN RANK_RUN...
```

The device probe compares every saved output against the host oracle bytes.
It saves actual bytes before comparison and marks only a complete successful run.
It refuses a reused output directory.
The comparator checks complete markers, saved topology, extents, and bytes.
It compares local projections when the topology is equal.
The split comparator joins all rank intervals in global order.
It rejects missing or repeated ranks.
Host comparator tests use explicit synthetic markers. They do not execute a GPU.

This probe does not run KDA, MLA, routing, experts, or a full layer loop.
It does not use the real checkpoint or produce logits or token IDs.
It cannot establish M2, M3, device performance, or full decoder readiness.
