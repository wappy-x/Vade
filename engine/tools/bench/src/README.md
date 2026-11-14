# engine/tools/bench/src

**The Rust source of the `vade-bench` library, linked into `vade bench`.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`first_tokens.rs`](first_tokens.rs) | Prepares chat IDs and checks greedy repeat receipts with the shared client. |
| [`first_tokens/`](first_tokens/) | Local mock checks for the chat proof client. |
| [`first_tokens_main.rs`](first_tokens_main.rs) | Prebuilt entry point for the chat proof client. |
| [`http.rs`](http.rs) | No summary comment yet; read the file. |
| [`lib.rs`](lib.rs) | Dispatches the timing protocol or retained development suite. |
| [`main.rs`](main.rs) | The earlier Rust development workloads, imported as the `suite` module. |
| [`timed/`](timed/) | Python-compatible request, stream, receipt and observer implementation. |
| [`memory_probe.rs`](memory_probe.rs) | Optional Linux CUDA allocation probe, compiled separately with rustc. |
| [`report.rs`](report.rs) | No summary comment yet; read the file. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

**How it fits:** the CLI exposes one `vade bench` command. Its default preserves
the timing protocol; `--suite` selects the older development cases. See the
[library README](../README.md) for the distinction and retained workloads.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
