# Vade engine

The engine loads supported open-weight models on NVIDIA DGX Spark and serves
streaming requests. Its host code and GPU kernels are written in Rust. GPU
kernels compile to PTX; the host uses the CUDA driver API without requiring a
CUDA toolkit or NVIDIA math libraries.

## Quick start

Start with the repository [INSTALL.md](../INSTALL.md). It covers the pinned
compilers, verified checkpoint files and the Flash-Next FP4-Local launcher.
For a single Ubuntu GB10, use the [Qwen3-8B quickstart](../docs/quickstart-one-spark.md)
or [Qwen3.6-35B-A3B-FP8 guide](../docs/serve-35b-one-spark.md).

For direct launches, see the [Flash-Next FP4-Local server](docs/next-http.md).
The [legacy FP8 profile](../docs/install-legacy-fp8.md) uses separate checkpoint,
topology and drafting settings.

## How it compares

[Benchmark history](docs/benchmarks.md) preserves measured builds, workload
settings and qualification limits. [Repository benchmarks](../docs/benchmarks.md)
link the Flash-Next summaries to [committed receipts](../bench/receipts/README.md).
These are historical results, not current-build performance promises.

## Determinism

The engine uses fixed reduction orders to make execution reproducible across
the supported paths. Tests compare batching, caching, drafting and chunking
where those combinations are implemented. Some optional arithmetic modes
have different numerical behavior; consult the relevant model and kernel
reference before changing switches.

Repeatability within Vade and agreement with an upstream implementation are
separate checks. Results apply to the exact model, build, settings and request
set recorded in [validation](docs/validation.md). Portable host oracles check
layouts and operation order; native GPU tests are needed to establish device
results and performance.

## Models

Supported architectures and checkpoint requirements are described in
[Models](docs/models.md). The [model registry](crates/model/README.md) describes
how metadata binds to execution. Download the complete pinned checkpoint and
verify its supporting files as well as its weights.

## Multiple Sparks

The engine contains tensor-parallel and pipeline execution paths. Their
supported models, world sizes and transports differ. Use
[Multiple Sparks](docs/multi-spark.md), [Flash-Next FP8](docs/flash-server.md)
or [native FP4-Local](docs/next-http.md) for the intended deployment.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md) and [design principles](docs/concepts.md)
- [Server and API](docs/server.md)
- [Speculative decoding](docs/speculative-decoding.md)
- [Build and host checks](../.github/workflows/ci.yml)

## Credits and license

The source is licensed under [Apache-2.0](LICENSE). Each model keeps its own
license. [Credits](docs/CREDITS.md), [implementation attribution](CREDITS-NOTES.md)
and [NOTICE](NOTICE) record the sources used.
