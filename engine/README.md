# Vade engine

The engine loads supported open-weight models on NVIDIA DGX Spark and serves
streaming requests. Its host code and GPU kernels are written in Rust. The
kernels compile to [PTX](../docs/GLOSSARY.md#pr), NVIDIA's portable GPU code
format, and the host talks to the GPU through the
[CUDA driver API](../docs/GLOSSARY.md#ac), the layer every NVIDIA system
already has. You don't need a CUDA toolkit or NVIDIA's math libraries.

## Quick start

Start with the repository [INSTALL.md](../docs/INSTALL.md). It covers the pinned
compilers, the model download and its checks, and the launcher for the default
model, `nvidia/Qwen3.8-Flash-Next-FP4-Local` (the
[FP4-Local](../docs/GLOSSARY.md#dg) profile).

Two older one-Spark guides serve other models: the
[Qwen3-8B route](../docs/quickstart-one-spark.md#qwen3-8b-bf16-ubuntu-to-a-first-answer)
and, as history, the
[Qwen3.6-35B-A3B-FP8 guide](../docs/history/serve-35b-one-spark.md).

To start the FP4-Local server by hand instead of through `vade`, see
[next-http](docs/next-http.md). The [FP8 install](../docs/install-fp8.md)
gives the FP8 checkpoint, topology and drafting settings.

## How it compares

[Benchmark history](docs/benchmarks.md) keeps each measured build with its
workload settings and the limits of what it proves. The
[repository benchmarks](../docs/benchmarks.md) link the Flash-Next summaries to
the [committed receipts](../bench/receipts/README.md). These are past results
for those builds, not a promise about the build you run.

## Determinism

The engine adds numbers up in a fixed order, so the same request gives the same
result on the supported paths. Tests check that batching, caching, drafting and
chunking leave results unchanged wherever those combinations are implemented.
Some optional arithmetic modes behave differently; read the reference for that
model and kernel before you change a switch.

Two questions are checked separately: does Vade repeat itself, and does it
agree with an upstream implementation? Each result holds only for the model,
build, settings and requests recorded in [validation](docs/validation.md).
Tests that run on an ordinary computer check memory layouts and the order of
operations. Only tests on the GPU itself can establish device results and
speed.

## Models

[Models](docs/models.md) lists the supported architectures and what each
checkpoint must contain. The [model registry](crates/model/README.md) explains
how a model's metadata is tied to the code that runs it. Download the whole
pinned [checkpoint](../docs/GLOSSARY.md#ac) and check every file, not only the
weights: [Check the files](../docs/INSTALL.md#check-the-files) gives the command.

## Multiple Sparks

The engine has two ways to split a model across Sparks:

- the [layer pipeline](../docs/GLOSSARY.md#ho), which FP4-Local runs as a
  chain on three or more Sparks when the tensor split does not apply: each
  Spark runs its own block of layers and passes the result on;
- [tensor parallelism](../docs/GLOSSARY.md#sz), which splits each layer's
  tensors so several GPUs work on a token together. FP4-Local on two Sparks,
  and on four over RDMA, runs the tensor split, and FP8 runs tensor parallel on two and four
  Sparks.

Which models, Spark counts and network transports each path supports differs.
Pick the guide for your setup: [Multiple Sparks](docs/multi-spark.md),
[Flash-Next FP8](docs/flash-server.md) or [native FP4-Local](docs/next-http.md).

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md) and [design principles](docs/concepts.md)
- [Server and API](docs/server.md)
- [Speculative decoding](docs/speculative-decoding.md) (drafting)
- [CI workflow](../.github/workflows/ci.yml): the build and host checks

## Credits and license

The source is licensed under [Apache-2.0](LICENSE). Each model keeps its own
license. [Credits](../docs/CREDITS.md) names everyone Vade builds on, on one page.
[NOTICE](NOTICE) records the files kept from others.
