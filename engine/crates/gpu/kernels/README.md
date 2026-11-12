# crates/gpu/kernels

**`spark-rust-kernels`: the engine's GPU kernels, written in Rust.** They compile to PTX, NVIDIA's portable GPU instruction format.

**What is here:** matrix multiplies, attention, expert routing, FP8 and FP4 math, and the other steps of each model layer. Each file starts with a one-line summary; the list is in [`src/`](src/README.md).

**Why:** writing every kernel in Rust removes the need for the CUDA toolkit and keeps one order of operations on every path, which is what makes answers repeatable.

**How it fits:** built by [`build-kernels`](../build-kernels/README.md); loaded and launched by the [GPU crate](../README.md). It is left out of a plain Mac workspace check because it targets the GPU.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
