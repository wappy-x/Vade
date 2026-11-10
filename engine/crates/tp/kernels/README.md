# crates/tp/kernels

**`spark-tp-kernels`: an optional GPU kernel for split models.** It keeps the same BF16 multiply order as the engine's own kernels.

**Why:** when a model is split across Sparks, each share must still add numbers in the same order, so answers match the one-Spark result.

**How it fits:** built and loaded as described in the [tensor-parallel crate README](../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
