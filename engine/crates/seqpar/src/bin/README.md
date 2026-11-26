# engine/crates/seqpar/src/bin

**Programs built from the `spark-seqpar` crate, one per file.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`seqpar-generate.rs`](seqpar-generate.rs) | Generate from token IDs using sequence-parallel prefill, then one-rank decode. |
| [`seqpar-launch.rs`](seqpar-launch.rs) | Run finite prefill jobs on loopback processes or discovered Spark neighbours. |
| [`seqpar-worker.rs`](seqpar-worker.rs) | Keep one model warm for idle-peer sequence prefill jobs. |

**Why a folder:** Cargo builds every file in a `bin/` folder as its own program.

**How it fits:** see the [`spark-seqpar` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
