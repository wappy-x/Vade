# Documentation

**Every Vade page, one line each, grouped by what you want to do.** New
here? Start with the [README](../README.md). Each page says at the top what
it covers and who it is for, and ends with where to go next.

## Get started

**Set up and run Vade on one or more DGX Sparks.**

- [Installation](../../INSTALL.md) and [Flash server](next-http.md).
  Current FP4-Local launcher, Mac tunnel, harness build and client setup.
  [Legacy FP8 installation](../../docs/install-legacy-fp8.md) remains separate.
- [One-Spark quickstart](../../docs/quickstart-one-spark.md). Current Ubuntu GB10
  route: prerequisites, engine workspace paths, complete official download and
  checksum commands, first chat and answer comparison. Use this
  route in place of the older [install page](install.md).
- [One-Spark server and API](server.md). Run `spark-serve`, connect apps through the
  OpenAI-compatible API, every flag, every limit.
- [Models](models.md). Which models run, context limits, precision and the
  roadmap. For executable Qwen3-8B full-file checksum verification, use
  [quickstart step 3](../../docs/quickstart-one-spark.md#3-download-and-verify-the-official-small-model).
- [Run, update, remove](upkeep.md). Keep Vade running as a service, update
  it, remove it.
- [Multiple Sparks](multi-spark.md). Serve Qwen3-8B or Qwen3-32B across
  cabled Sparks with tensor parallelism.

## Performance

**What was measured, against which engines, and how to repeat it.**

- [Benchmarks](benchmarks.md). Vade against SGLang, vLLM and TensorRT-LLM on
  the same Spark: every table, setting and byte, and how to reproduce them.
- [Validation](validation.md). The determinism checks and the measured
  agreement with the official model, with every disagreement.

## How it works

**The engine from the inside.**

- [Architecture](architecture.md). The path of a request, the crates, the
  GPU kernels, memory, scheduling and multi-Spark mechanics.
- [Design principles](concepts.md). The ground rules and the reasons behind
  them: determinism, precision, memory, batching, speculative decoding.
- [Speculative decoding](speculative-decoding.md). How Vade proposes tokens
  ahead without changing answers, per model, and how to measure it.
- [DGX Spark platform](platform.md). What Vade targets on the Spark, what it
  leaves alone, what runs on one Spark or several.

## Project

**Records that make the project checkable.**

- [Credits](CREDITS.md). Every person, team, paper and project Vade learned
  from, with where to find them and what Vade used.
- [Provenance](provenance.md). Reused data files with hashes and notices;
  which code is Vade's own.
- [Link checks](link-checks.md). Whether every outside link and profile could
  be reached, and how it was checked.
- [History](history/README.md). Older records and research pages that do
  not describe current main.

For maintainers and writers:

- [Export](export.md). How the allowlisted shareable copy is made, and what
  it checks.

## Changing the code

**Each crate and tool has its own README for people reading or changing it.**

Serving on one Spark:

- `crates/serve`: [spark-serve](../crates/serve/README.md), the HTTP server,
  scheduler, prompt cache and memory admission; plus
  [budgets](../crates/serve/BUDGETS.md).
- `crates/engine`: [spark-engine](../crates/engine/README.md), which runs the
  model's layers and owns each request's KV cache.
- `crates/gpu`: [spark-gpu-probe](../crates/gpu/README.md), the GPU layer:
  driver calls, memory, kernels, the decode attention order.
- `crates/model`: [engine-model](../crates/model/README.md), which reads and
  checks checkpoints; plus the historical
  [memory ledger](../crates/model/MEMORY.md).
- `crates/tokenizer`: [engine-tokenizer](../crates/tokenizer/README.md), text
  to tokens and back, and chat templates.

Across Sparks:

- `crates/tp`: [spark-tp](../crates/tp/README.md), tensor parallelism across
  Sparks; plus the historical [measurements](../crates/tp/MEASUREMENTS.md)
  and the [design note](../crates/tp/MULTI-SPARK.md).
- `crates/seqpar`: [spark-seqpar](../crates/seqpar/README.md), splitting a
  long prompt across Sparks, and the RDMA transport.

Tools and reference data:

- `tools/bench`: [vade-bench](../tools/bench/README.md), the HTTP benchmark
  client.
- `fixtures/`: saved reference outputs the checks compare against, for
  [Qwen3-8B](../fixtures/qwen3/README.md),
  [Qwen3-32B](../fixtures/qwen3-32b/README.md),
  [Qwen3-30B-A3B](../fixtures/qwen3-30b-a3b/README.md),
  [Qwen3-30B-A3B FP32](../fixtures/qwen3-30b-a3b-fp32/README.md) and
  [Flash Next FP8](../fixtures/qwen3.8-flash-next-fp8/README.md).

**Where next:** [README](../README.md) · [One-Spark quickstart](../../docs/quickstart-one-spark.md) ·
[Benchmarks](benchmarks.md)
