# engine/crates/seqpar/src

**The Rust source of the `spark-seqpar` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`control.rs`](control.rs) | Challenge-response admission for control and data peers. |
| [`coordinator.rs`](coordinator.rs) | Warm peer admission and prefill for a server retaining the final causal rank. |
| [`discovery`](../../fabric/src/discovery.rs) | Fabric peer discovery, re-exported under the original name. |
| [`lib.rs`](lib.rs) | Sequence-parallel prefill: replicated model, contiguous positions, ring K/V exchange. |
| [`partition.rs`](partition.rs) | Contiguous token partitions; later tokens can be assigned smaller slices. |
| [`prefill.rs`](prefill.rs) | Layer-stepped prefill with replicated weights and ring K/V exchange. |
| [`rdma.rs`](rdma.rs) | RC SEND/RECV for host K/V buffers, adapted from Vade scale transport. |
| [`transport.rs`](transport.rs) | FP32 all-gather and fixed-rank-order reduction over ring neighbours. |
| [`verbs`](../../fabric/src/verbs.rs) | Fabric rdma-core ABI and lazy loading, re-exported on Linux. |
| [`bin/`](bin/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`spark-seqpar` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
