# crates/gpu/build-kernels

**`spark-build-kernels`, the tool behind `cargo kernels`.** It builds the GPU kernels with the pinned nightly Rust compiler, then copies and stamps the PTX files.

**Why:** the kernels need a nightly compiler and the `nvptx64-nvidia-cuda` target, while the server builds on stable Rust. One command keeps both steps in order.

**How it fits:** the server embeds the PTX this tool produces. See the [GPU crate README](../README.md) and [Install, Build](../../../docs/install.md#3-build).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
