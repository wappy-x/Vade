# Credits

**Vade's code is its own, written from scratch in Rust, and many of its ideas came from other people's work.** This page introduces the engine and harness credit lists, and summarizes the harness notes.

No source code was copied, translated or adapted from any project credited here. A credit means Vade learned from that work, not that their code runs in Vade.

<br>

## In 30 seconds

| Part | Credit list | What it covers |
| --- | --- | --- |
| **Vade engine** | [Engine credits](engine/docs/CREDITS.md) | Model makers, reference code, papers, the DGX Spark community, NVIDIA, compilers and standards |
| **Vade harness** | [Harness notes](harness/CREDITS-NOTES.md), summarized [below](#harness) | The coding-agent clients and terminal designs the harness studied |
| **Third-party files** | [NOTICE](NOTICE) and [engine/NOTICE](engine/NOTICE) | The few files Vade ships from others, with their licenses |

<br>

---

## Engine

**The full list lives in [engine/docs/CREDITS.md](engine/docs/CREDITS.md).** Each entry says what Vade took and links the people, papers and projects.

- [Model makers](engine/docs/CREDITS.md#model-makers): the teams whose official models define what the engine must compute
- [Reference code](engine/docs/CREDITS.md#reference-code): the makers' official code that its answers are checked against
- [Attention and determinism](engine/docs/CREDITS.md#attention-and-determinism) · [Serving many requests](engine/docs/CREDITS.md#serving-many-requests) · [Speculative decoding](engine/docs/CREDITS.md#speculative-decoding) · [Model building blocks](engine/docs/CREDITS.md#model-building-blocks)
- [DGX Spark community](engine/docs/CREDITS.md#dgx-spark-community) · [NVIDIA and transport](engine/docs/CREDITS.md#nvidia-and-transport) · [Compiler and standards](engine/docs/CREDITS.md#compiler-and-standards)
- [Studied, not used](engine/docs/CREDITS.md#studied-not-used) · [Reused data](engine/docs/CREDITS.md#reused-data)

<br>

---

## Harness

**What the harness learned from, and from whom.** The harness's own notes are in [harness/CREDITS-NOTES.md](harness/CREDITS-NOTES.md).

**Client setup.** For Codex, Hermes, OpenCode and Pi, client setup follows each client's own pinned command-line help and configuration docs. Hermes's context exception and file-write limits come from its pinned initialization and file-safety modules. The templates, the model catalog prompt and the instruction to call a tool right away instead of promising to are Vade's own Rust output; no client code or prompt was copied. Source links and pins are listed in [the passthrough matrix](harness/docs/passthrough-matrix.md#sources-and-credit).

**Terminal presentation.** Native terminal presentation follows the clarity goals we set after comparing Claude Code and Codex CLI: a quiet answer footer, a highlighted command picker, and readable Markdown. The renderer, cell layout and PTY regression tests are original std-only Rust; no competitor source, assets or dependencies were used.

**Design references.** Hermes Agent, the Pi coding agent and Codex CLI informed the harness's interface choices, as conceptual references, not dependencies: see [Design references](harness/README.md#design-references) in the harness README.

The harness design review also studied stable prefixes, continuation identity,
session trees, cancellation and isolated child contexts in these pinned sources:

[Codex](https://github.com/openai/codex/blob/2fdf047c9631c9ed01a31b62efb7891718a931a8/README.md), [Pi](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/README.md), [Hermes](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/README.md), [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/blob/5badb15009ae1756c3afe0ae0cef1faafc290ccc/README.md), [OpenCode](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/README.md), [Aider](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/README.md), [Goose](https://github.com/aaif-goose/goose/blob/9560429ff982bbfecec5094963e5f34696b63a1c/README.md), [Claude Code](https://code.claude.com/docs/en/how-claude-code-works).
These were source/documentation studies, not interoperability measurements.

<br>

---

## Corrections

**To report a missing or wrong credit, open an issue in the repository's issue tracker.** Name the idea, its original source and the part of Vade that uses it. An earlier original source is always welcome.

<br>

**Back:** [README](README.md) · **Next:** [Engine credits](engine/docs/CREDITS.md) →
