# kernel-bench

**Measures:** the speed and output bytes of single kernels on one Spark, without loading the decoder. It selects matrix shapes from a checkpoint header and compares a baseline and a candidate kernel image.

**Source:** the runner is [`kernel-bench.rs`](../../../engine/crates/tp/src/bin/kernel-bench.rs) (feature `kernel-bench`). The claim wrapper is [`kernel-bench-one.sh`](../../../engine/scripts/kernel-bench-one.sh). The guide is [kernel-bench.md](../../../engine/docs/kernel-bench.md). These files are already in Git, so this folder points to them.

**Build,** from `engine/`: `cargo +1.99.0 kernels`, then `cargo +1.99.0 build --release --locked -p spark-tp --features kernel-bench --bin kernel-bench`.

**Run:** `kernel-bench catalog CHECKPOINT` lists the shapes. On one empty Spark: `bash engine/scripts/kernel-bench-one.sh RUNNER CHECKPOINT NEW_OUT COMMAND_FILE`. The wrapper writes the claim and release lines in `~/gpu.lock`.

**Prints:** with no arguments, the runner prints `KERNEL BENCH ERROR usage: ...` (exit 1) and the wrapper prints its usage (exit 2). A real run prints `CLAIM`, writes the results to NEW_OUT, then prints `RELEASE`.
