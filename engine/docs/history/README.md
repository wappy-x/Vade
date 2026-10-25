# History

**Nothing here describes current main.** These are older records and research
pages, kept so a past number or decision can be traced, for anyone checking
how a result came about.

- **[Flash Next architecture](flash-next-architecture.md).** The official
  Qwen3.8-Flash-Next-FP8 text path, layer by layer, as the reference code had
  it. Main `efee11b`, 2 October 2026. Current information:
  [Models](../models.md#qwen38-flash-next-fp8).
- **[FP32-cache measurements](fp32-cache-measurements.md).** Qwen3-8B speed, drafting
  and memory, with an FP32 KV cache (the model's conversation memory) by
  default. Main `1d52612`, 2 October 2026. Current information:
  [Benchmarks](../benchmarks.md), [Validation](../validation.md).
- **[Long-context results](longctx-performance.md).** Qwen3-8B prompt reading
  and decode speed at 8,192 to 38,912 prompt tokens, with an FP32 KV cache.
  Main `67127c8`, 2 October 2026. Current information:
  [Benchmarks](../benchmarks.md), [Validation](../validation.md).
- **[Qwen3-8B integration record](integration-2026-10-02.md).** The gates,
  memory and speed of one past merge of two branches, with an FP32 KV cache.
  2 October 2026. Current information: [Benchmarks](../benchmarks.md),
  [Validation](../validation.md).

**Both measurement pages ran with an FP32 KV cache by default.** BF16 was
only an option then. Current main keeps it in BF16, a 16-bit format, so their
FP32 numbers do not measure today's defaults.

**Where next:** [Benchmarks](../benchmarks.md) · [Validation](../validation.md) ·
[Documentation index](../README.md)
