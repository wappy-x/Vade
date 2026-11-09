# step-direct

**Measures:** whether one direct Step launch writes the same recurrent state and output bytes as the earlier StepColumns plus StepLayout pair, and the time of each. It covers dimension 128 with head counts 1, 2, 3, 17, 63 and 64, ten value patterns each, three steps per pattern.

**Source:** [`kernel_stepdirect.rs`](../../../engine/tools/bench/src/kernel_stepdirect.rs), run through the `--causal-probe` mode of the bench workspace's `kernel-bench`. The guide is [step-direct-screen.md](../../../engine/docs/step-direct-screen.md). These files are already in Git, so this folder points to them.

**Build,** after `cargo kernels`: `cargo +1.99.0 build --release --locked --manifest-path engine/tools/bench/Cargo.toml --bin kernel-bench`.

**Run,** on one empty Spark: `kernel-bench --causal-probe base.cubin candidate.cubin step-direct`. Then run the wrong-image control, `kernel-bench --causal-probe base.cubin base.cubin step-direct`. The control must fail.

**Prints:** `CAUSALP baseline_sha256=... candidate_sha256=...`, 60 `STEPDIRECT BYTES PASS` lines, a summary of 540 comparisons, two `STEPDIRECT TIME` lines, then `STEPDIRECT PASS`. A missing image file stops the run before the GPU opens.
