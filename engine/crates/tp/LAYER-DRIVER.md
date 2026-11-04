# Deferred layer boundary

[`layer_driver`](src/layer_driver.rs) checks the boundary between a logical
layer and device arguments. It does not execute or admit a model.
The boundary accepts both mixer forms and dense or routed feed-forward blocks.
Its device implementation covers the mHC operations only.
The saved config test retains the first decoder's dimensions and storage recipes.
Its generated headers are fixtures. They are not real checkpoint evidence.

`LayerBoundary::new` uses the existing contracts:

- [`LayerPlan`](../model/LAYER-PLAN.md) supplies the operation order and tensor splits.
- [`HeaderAdmission`](../model/HEADER-ADMISSION.md) supplies physical weight dtypes, axes, and components.
- [`Contract`](../model/CAST-CONTRACT.md) supplies conditional cast requirements and source gaps.
- [`StatePlan`](../model/STATE-PLAN.md) supplies logical lifetimes and state ownership.
- [`Ownership` and `GatherLayout`](../model/TENSOR-SPLIT.md) supply complete tensor intervals and byte copies.

The boundary checks the state plan against the complete description.
Equal state byte counts do not establish a matching operation program.
It keeps every weight's physical component view and logical ownership interval.
Projection views retain the complete input dimension and original scale coordinates.
The boundary does not convert the host state plan's element widths into physical cache widths.

The argument implementation covers `mhc_prepare` through the existing
[`Prepare` ABI](../gpu/src/mhc_params.rs). `prepare_mhc` reads admitted components
through the caller's reader. It widens stored BF16 parameters to FP32 exactly.
Stored FP32 parameters keep their bits. Both use the primitive's row-major layout.
The result records the stored dtypes. It changes no quantized projection or reduction.
`prepare_mhc_checkpoint` checks the bound headers and calls the existing retained-file reader.
It does not reopen paths or add another loader.
Matching headers do not prove checkpoint payload identity.

`MhcOperands::arguments` checks every address and scalar in the shared ABI.
It rejects wrong extents, non-FP32 planes, fragment packing, unaligned addresses,
overflow, overlap, and conflicting rank or context metadata.
These checks establish an argument layout only.
Caller-supplied addresses do not prove device allocation ownership or uploaded contents.
On Linux, `MhcOperands::upload` creates a resident mHC primitive through the
GPU backend. Its parameter vectors retain the widened checkpoint bits.
It does not accept caller-supplied device addresses.
Other operations return an error that names the missing checked ABI binding.
An entry listed in a logical target is not an implemented driver binding.

Gather descriptors follow operation order and each operation's declared output order.
They use the conditional cast contract's element width.
The host adapter calls the existing byte exchange and copies rank packets into global coordinates.
It does not add partial sums or cast values.
The caller must not treat these host copies as device transport completion.

`require_executable` returns the existing incomplete-source error.
No capability flag, model name, or environment setting bypasses it.
The active serving path does not call this module.

## Open boundaries

The [cast contract](../model/CAST-CONTRACT.md) keeps framework reduction trees,
quantized Linear implementations, and physical cache dtypes unresolved.
The FP32 mHC primitive does not prove equivalence to those source backends.
The ordinary norm also lacks the source's cast before weight multiplication.

The [resident session owners](SESSION-STATE.md) supply device state banks and
ordered state transitions. A future device driver uses their `Session` owner.
The model crate's `SessionState` remains the host reference owner for tests.
Physical producers and transport completion still need driver bindings.

The [resident mHC adapter](../gpu/src/backend/mhc.rs) owns real vectors.
It checks the backend context, complete extents, alignment, and overlap.
Every launch completes before it returns, including a failed launch attempt.
The residual input stays owned with its coefficients through the block.
`Bf16Vector` records BF16 values in FP32 lanes. Each element uses four bytes.
It does not represent packed BF16 storage or choose a cache dtype.
Host uploads must be finite and exact BF16. A resident cast uses RN-even.
Device results can be nonfinite. The type does not certify arithmetic quality.

The candidate collapse rounds the completed FP32 stream sum to BF16.
The candidate combine rounds post and comb coefficients before use.
It rounds the residual sum and block product separately, then rounds their sum.
Both reductions use ascending source-stream order.
These explicit casts follow the saved source sites in the cast contract.
Framework reduction equality and full-model qualification remain open.
The existing FP32 primitive entries and existing model execution stay unchanged.

The driver still must bind KDA history packing and every remaining primitive argument.
It must retain allocations through transport completion.
It must preserve the existing append, rollback, save, and restore contracts.
The current host state tests do not prove device lifetimes or rollback.

A complete layer byte kit needs a qualified real-weight reference and physical bindings.
This boundary provides neither a new oracle nor a tolerance rule.
It cannot support such a kit while `require_executable` refuses the layer.

## Host checks

[`layer_driver` tests](tests/layer_driver.rs) cover header-backed operand conversion,
complete argument binding, malformed physical layouts, unsupported operations,
state identity and lifetime checks, and rank-ordered byte gathers.
They also preserve saved state when admission fails.
No test reads real weights or claims device bytes, token IDs, quality, or speed.

The [`mhc-resident` probe](../gpu/examples/mhc-resident.rs) writes host expected
buffers or checks resident device buffers against those saved bytes.
It saves actual buffers before comparison and rejects a reused output directory.
It also compares reversed single-row execution with batched execution.
Its fixtures check primitive arithmetic and the explicit BF16 boundaries.
They do not supply a checkpoint layer oracle or a complete GLM gate.
