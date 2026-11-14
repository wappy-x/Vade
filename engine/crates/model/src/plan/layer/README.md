# Logical layer reference

[`reference.rs`](reference.rs) runs the independent scalar forward and
replays the declared program. Both use supplied decoded FP32 tensors.
They record computed shapes for comparison with the program.
[`profile.rs`](profile.rs) rejects unqualified runtime operands before scalar execution.
The [runtime type map](../../../RUNTIME-TYPES.md) records the first open load boundary and later producers.

`WeightSource` and `forward_from` load decoded weights as each operation needs them.
Expert operations load only selected expert matrices. Their trace keeps all logical weight shapes.
The [checkpoint tool](../../../M2-REFERENCE.md) saves tensors and byte receipts for this scalar profile.

The [layer plan](../../../LAYER-PLAN.md) gives the kernel targets, buffer
layouts, tensor split, source record, and remaining device gaps.
