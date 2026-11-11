# qwen3-30b-a3b-cuda-suite

**24 test prompts for Qwen3-30B-A3B, with the official model's top-20 next-token choices.**

| Files | What they are |
| --- | --- |
| [`manifest.json`](manifest.json) | The checkpoint, its pinned revision, and the hash of every model file used. |
| `case-NN-<kind>-<length>.json` | One prompt each: prose, code, math, chat, multi-turn or reasoning, 32 to 28,672 tokens. |

**Why:** a broad, fixed set of prompts makes agreement with the official model measurable, at short and long lengths.

**How it fits:** made by the [privately retained reference oracle](../../tools/truth/README.md) with official BF16 weights on CUDA; read by the agreement check in [Validation](../../docs/validation.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
