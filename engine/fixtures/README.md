# fixtures

**Saved expected output that the engine is checked against.** A fixture is a known-good answer kept on file.

| Folders | What they hold |
| --- | --- |
| `*-cuda-suite/` | 24 test prompts per model, from 32 to 28,672 tokens, with the official model's top next-token choices. The main agreement check. |
| `qwen3/`, `qwen3-32b/`, `qwen3-30b-a3b/`, `qwen3-30b-a3b-fp32/` | Older, narrower score rows (`.f32`) kept for debugging. Each has a README. |
| `qwen3.8-flash-next-fp8/` | Reference rows for the Flash-Next research engine. |

**Why:** answers are compared with the official weights running in Hugging Face Transformers, so a wrong result is caught, not guessed at.

**How it fits:** the checks that read these files, and every result, are in [Validation](../docs/validation.md). The [reference generators](../tools/truth/README.md) are retained privately outside this repository.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
