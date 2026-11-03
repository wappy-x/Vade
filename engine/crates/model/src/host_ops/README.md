# Host operation references

[`router.rs`](router.rs) defines fixed corrected sigmoid selection and reuses
the existing softmax reference. [`moe.rs`](moe.rs) defines the source-order
expert fold and ungated shared addition. The [math table](../../ROUTING.md)
records the source lines, dtype boundaries, and device gates.

These operations do not admit a deferred model for execution. See the
[metadata description](../../GLM.md).
