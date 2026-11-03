# Logical layer reference

[`reference.rs`](reference.rs) runs the independent scalar forward and
replays the declared program. Both use supplied decoded FP32 tensors.
They record computed shapes for comparison with the program.

The [layer plan](../../../LAYER-PLAN.md) gives the kernel targets, buffer
layouts, tensor split, source record, and remaining device gaps.
