# Measurement and evaluation commands

Use the first-party Rust commands in `vade`; all use the standard library.
Runtime inputs, generated prompts, server captures, dumps and credentials stay
in private runtime storage. Endpoints and paths are explicit command arguments.

| Command or reference | Purpose |
| --- | --- |
| `vade bench` | Measure client timing, concurrency and cold 8k reading with explicit sampling and private receipts. |
| `vade test longctx` | Freeze 32k/128k retrieval and scoring suites, capture explicitly supplied servers, compare aligned logits and an amplification floor. |
| `vade test gdn-state` | Compare complete GDN state dumps: strict byte identity, FP32 controls or the recorded diagnostic floor. |
| `vade test unix-flags-abi` | Typecheck audited Unix file flag constants for the host or a supported target. |
| [Long-context procedure](longctx-eval.md) | Historical pinned-arm specification and limitations; current policy governs acceptance. |

Build the CLI with `cargo build --offline --manifest-path cli/Cargo.toml`.
Run from the monorepo root:

```sh
cli/target/debug/vade bench --help
cli/target/debug/vade test longctx --help
cli/target/debug/vade test gdn-state --help
cli/target/debug/vade test unix-flags-abi --help
cli/target/debug/vade bench --self-test
```

Use explicit `--output` for timing receipts. Without it, `VADE_ORCH_ROOT`
selects its private results directory; the ignored local runtime is the fallback.
The [timing command guide](benchmarks.md#run-vade-bench) and
[CLI evaluation guide](../cli/evals.md) describe options and receipts.
Start the product engine with [`vade up`](../cli/README.md#multi-spark-source-setup).

The earlier Python references and their host tests are retired from the product
repository. Their originals and parity evidence remain private. Host replay
and mock-stream checks establish tool behavior. Inference correctness and
throughput require owner-scheduled Spark qualification with frozen inputs.
