# Measurement inputs

This folder holds the benchmark probe, the repository checks and the script
that draws the benchmark figure. To start and stop the engine itself, use
`vade launch` on a Spark ([cli/README.md](../cli/README.md#start-the-engine-on-your-spark))
or `vade up` and `vade down` from a controller
([controller commands](../cli/README.md#controller-commands)).

## Timing the installed server

`vade bench` sends explicit greedy sampling, measures client wall-clock time,
and records server token counts and memory. Follow
[INSTALL.md](../INSTALL.md#run-the-benchmark) for real requests and
[the option table](../docs/benchmarks.md#run-vade-bench) for timing options.

```sh
cargo build --release --offline --manifest-path cli/Cargo.toml
cli/target/release/vade bench --self-test
cli/target/release/vade bench --help
```

The self-test uses a local mock server and does not check GPU inference.
The implementation is [the benchmark library](../engine/tools/bench/README.md),
linked into the CLI. Its development workloads use `vade bench --suite`; for
those, pass the endpoints, frozen prompts, checkpoint pins and rank observation
commands yourself. `--output` chooses a private receipt path, and
`VADE_ORCH_ROOT` can select a private results root. Raw records stay on your
machine; only reviewed small summaries belong in
[bench/receipts](../bench/receipts/README.md).

The frozen synthetic [8,192-token probe](fixtures/probe-8192.json) is kept
from the pre-install baseline. Its SHA256 is
`11869d33b6fe2411447a7c9fd34ef76895ca53017f7505e2d63ac77783df945f`,
and the client also requires the server to report exactly 8,192 prompt tokens.

Use [`vade test longctx` and `vade test gdn-state`](../cli/evals.md) for
long-context and GDN-state comparisons, and the [Rust gate commands](../cli/docs/test.md)
for build and determinism checks.

## Documentation links

Run `python3 tools/check-doc-links.py` from the checkout to check local
Markdown and text-document links, anchors and embedded asset paths. CI runs the
same check. It reads tracked documents and new ones that are not ignored, and
makes no network requests, so external URLs are not checked.

## Bundle producer checks

Run `python3 tools/test-bundle-producer.py` for both bundle test suites. They
exercise [`scripts/make-bundle.sh`](../scripts/make-bundle.sh) without Rust builds or a
GPU. The tests run the real shell producer with
[simulated build tools](fixtures/bundle-producer-tool.py), then use
[`check-bundle.py`](check-bundle.py) to verify the archive and manifest. They
cover publication races, failure cleanup and path sanitization.
The release workflow also runs the checker against real built artifacts.

Run `python3 tools/test-bundle-checker.py` for the checker's archive layout,
manifest and corruption cases alone. It uses in-memory tar fixtures and
simulates decompression, so no external build or compression tools are needed.

## Example script checks

Run `python3 tools/test-two-sparks.py` to exercise
[`examples/two-sparks.sh`](../examples/two-sparks.sh) with `vade` and `curl`
replaced by stubs, from a checkout path and temporary folder holding `&`, `|`,
`\` and spaces. It checks that the dry-run config copy keeps those paths
literally, that a failed preview keeps its exit status, and that the live
steps run in order and call `vade down` after a failure once startup has
begun. CI runs the same check. It needs no Spark, SSH or network, and does not
show that a real two-Spark start works.

## CI without a GPU

[`ci-no-gpu.py`](ci-no-gpu.py) is for CI only. On a Linux host without CUDA,
after the real PTX is built, it generates a stand-in CUDA library in
`engine/target/ci-no-gpu` whose every device call aborts, so the engine links
and any device test that runs by accident fails instead of passing. Device
tests must carry a libtest ignore reason. The CI workflow
(`.github/workflows/ci.yml`) adds that folder to `LIBRARY_PATH` and
`LD_LIBRARY_PATH`. Never ship this library.

## Benchmark figure

Run `python3 -I tools/benchmarks-chart.py` from the checkout to redraw the
[results figure](../docs/benchmarks.md#previous-2026-10-24-v010-release-5a22651),
which compares one DGX Spark with two, drafting ON. It uses only the Python
standard library, reads the v0.1.0 drafting-ON home cards and their READMEs in
[bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md), and writes
the light and dark SVG files in [docs/assets](../docs/assets/README.md). It
stops if a value is missing, a label would collide or run past the margin, a
receipt does not name the expected build, or any of those cards (six single
Sparks, three pairs) no longer says what the headline says.
