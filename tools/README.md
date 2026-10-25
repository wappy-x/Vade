# Measurement inputs

Product startup and shutdown use Rust [`vade up` and `vade down`](../cli/README.md#multi-spark-source-setup).
Current development happens directly in the monorepo; historical import tools
and migration reference implementations remain in the private workshop.

## Timing the installed server

`vade bench` sends explicit greedy sampling, measures client wall-clock time,
and records server token counts and memory. Follow
[INSTALL.md](../INSTALL.md#6-run-the-timing-benchmark) for real requests and
[the option table](../docs/benchmarks.md#run-vade-bench) for timing options.

```sh
cargo build --release --offline --manifest-path cli/Cargo.toml
cli/target/release/vade bench --self-test
cli/target/release/vade bench --help
```

The self-test uses a local mock server and does not validate GPU inference.
The implementation is [the benchmark library](../engine/tools/bench/README.md),
linked into the CLI. Its development workloads use `vade bench --suite`.
Supply endpoints, frozen prompts, checkpoint pins and rank observation commands
explicitly. `--output` chooses a private receipt path; `VADE_ORCH_ROOT` can select
a private results root. Raw records remain local; reviewed small summaries
belong in [bench/receipts](../bench/receipts/README.md).

The frozen synthetic [8,192-token probe](fixtures/probe-8192.json) is retained
from the pre-install baseline. Its SHA256 is
`11869d33b6fe2411447a7c9fd34ef76895ca53017f7505e2d63ac77783df945f`;
the client also requires exactly 8,192 server-reported prompt tokens.

Use [`vade test longctx` and `vade test gdn-state`](../cli/evals.md) for
long-context and GDN comparisons, and the [Rust gate commands](../cli/docs/test.md)
for build and determinism checks.

## Documentation links

Run `python3 tools/check-doc-links.py` from the checkout to check local
Markdown/text-document links, anchors and embedded asset paths. CI runs the
same check. It reads tracked and unignored new documents and makes no network
requests. External URLs are outside its scope.

## Benchmark figure

Run `python3 -I tools/benchmarks-chart.py` from the checkout to redraw the
[results figure](../docs/benchmarks.md#current-2026-10-24), which compares one
DGX Spark with two, drafting ON. It uses only the Python standard library,
reads the v0.1.0 drafting-ON home cards and their READMEs in
[bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md), and writes
the light and dark SVG files in [docs/assets](../docs/assets/README.md). It
stops if a value is missing, a label would collide or run past the margin, a
receipt does not name the expected build, or any of those cards (six single
Sparks, three pairs) no longer says what the headline says.
