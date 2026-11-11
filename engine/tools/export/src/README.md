# engine/tools/export/src

**The Rust source of the `vade-export` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`main.rs`](main.rs) | Exact-file sanitized snapshots; never copy implementation Git history. |
| [`scan.rs`](scan.rs) | Reject private literals before any export writes. |
| [`sha256.rs`](sha256.rs) | SHA-256 for export manifests and offline data provenance. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** see the [`vade-export` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
