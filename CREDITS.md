# Credits

This page names the work that Vade builds on.

Vade's code is written from scratch in Rust, and much of its design is our own work. Other parts use ideas from published papers and from other projects. The page lists the original work first. Then it links the engine and harness credit lists and summarizes the harness notes.

No source code was copied from any project credited here. A credit names an idea or a reference behavior and the part of Vade that uses it. It does not mean that their code runs in Vade.

One file follows a credited reference closely. The FP32 CPU reference [refcpu.rs](engine/crates/model/src/flash/refcpu.rs) computes the formulas of the Transformers file `modeling_qwen4_exp.py` (revision `b0cdc61`) in the same order. Its comments cite the lines of that file.

<br>

## Original work

**These parts were designed and built in this project.** Each item names where to check it in the tree. Where an item uses a published idea, the credit lists name that idea and its source.

- **The exactness law and its gates.** The same request with the same setup must give identical token IDs on every run. This holds alone or batched, cached or fresh, and with drafting on or off. The same request must also give the same IDs across supported Spark counts. The rule and the merge gate are in [AGENTS.md](AGENTS.md#exactness-law-and-merge-gate) and [architecture](docs/architecture.md#exactness-and-model-quality). Batch-invariant arithmetic is a published idea, credited under [Attention and determinism](engine/docs/CREDITS.md#attention-and-determinism). The rule does not hold everywhere yet. On FP8, six Sparks give other token IDs than two Sparks ([changelog](CHANGELOG.md), [Exactness](README.md#exactness)).
- **Standard-library Rust, GPU kernels included.** The engine, the CLI, the harness and the repository tools use no third-party crates. The GPU kernels are Rust compiled for NVPTX, in [engine/crates/gpu/kernels](engine/crates/gpu/kernels) ([What it is](README.md#what-it-is)).
- **The tensor split and the RDMA fabric.** The split runs one model across two or four Sparks. The Sparks send data to each other over RDMA ([engine/crates/tp](engine/crates/tp/README.md), [verbs.rs](engine/crates/fabric/src/verbs.rs)). Tensor parallelism follows the Megatron-LM paper. The transport ideas that we studied are credited under [DGX Spark community](engine/docs/CREDITS.md#dgx-spark-community) and [NVIDIA and transport](engine/docs/CREDITS.md#nvidia-and-transport). The rank code and the transport are our own Rust.
- **The Flash-Next MTP drafting path.** The checkpoint's own multi-token prediction (MTP) head proposes tokens. A batched check keeps only the tokens that the model would have chosen ([next_mtp.rs](engine/crates/engine/src/flash/next_mtp.rs)). Under greedy decoding, drafting on and off give identical token IDs ([Drafting](docs/architecture.md#drafting)). Draft-and-verify and MTP are published ideas, credited under [Speculative decoding](engine/docs/CREDITS.md#speculative-decoding).
- **The agent harness and the operator CLI.** `vade chat` opens a native coding agent ([harness](harness/README.md)). The `vade` CLI installs, starts, checks and stops the engine ([cli](cli/README.md), [What you get](README.md#what-you-get)). On Linux, the Spark included, the agent runs no shell commands. Its shell tool needs the macOS sandbox ([What works today](README.md#what-works-today-and-what-doesnt-yet)).
- **Cards, gates and receipts.** `vade verify` sends the same requests with drafting on and off. Then it checks that the token IDs match ([Verify exactness](cli/README.md#verify-exactness)). `vade bench` measures the running engine and writes a receipt. A change to engine behavior must pass a device gate on real Sparks before it merges. Its card joins the exactness and measurement results ([Exactness and evidence](CONTRIBUTING.md#exactness-and-evidence), [The measurement card](CONTRIBUTING.md#the-measurement-card)). A gated change that was rebased onto a moved main passes a screen on the Sparks first.

<br>

## In 30 seconds

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [Engine credits](engine/docs/CREDITS.md) | Model makers, reference code, papers, the DGX Spark community, NVIDIA, compilers and standards |
| [Harness notes](harness/CREDITS-NOTES.md) | The coding-agent clients and terminal designs that the harness studied |
| [NOTICE](NOTICE) | The few files that Vade ships from others, with their licenses |
| [engine/NOTICE](engine/NOTICE) | The same list of third-party files, kept with the engine |

</div>

The [Harness](#harness) section below summarizes the harness notes.

<br>

---

## Engine

**The full list is in [engine/docs/CREDITS.md](engine/docs/CREDITS.md).** Each entry names the idea or behavior that Vade uses. It links the people, papers and projects.

<div align="center">

| Section | What it covers |
| :---: | :---: |
| [Model makers](engine/docs/CREDITS.md#model-makers) | The teams whose official models define what the engine must compute |
| [Reference code](engine/docs/CREDITS.md#reference-code) | The makers' official code that the engine's answers are checked against |
| [Attention and determinism](engine/docs/CREDITS.md#attention-and-determinism) | The papers behind the attention algorithms and the fixed order of additions |
| [Serving many requests](engine/docs/CREDITS.md#serving-many-requests) | Batching, memory pages, prefix reuse, scheduling and the rival engines |
| [Speculative decoding](engine/docs/CREDITS.md#speculative-decoding) | The draft-and-check idea that Vade calls drafting |
| [Model building blocks](engine/docs/CREDITS.md#model-building-blocks) | The parts that the official models are built from |
| [DGX Spark community](engine/docs/CREDITS.md#dgx-spark-community) | People and projects that published work on the same hardware |
| [NVIDIA and transport](engine/docs/CREDITS.md#nvidia-and-transport) | NVIDIA documents and libraries, and the RDMA and collective-communication work |
| [Compiler and standards](engine/docs/CREDITS.md#compiler-and-standards) | The build tools and the file and wire formats |
| [Studied, not used](engine/docs/CREDITS.md#studied-not-used) | Work that the project examined and chose not to adopt |
| [Reused data](engine/docs/CREDITS.md#reused-data) | The few files kept unchanged, as data, with their licenses |

</div>

<br>

---

## Harness

**This section names the ideas that the harness studied, and their sources.** The harness's own notes are in [harness/CREDITS-NOTES.md](harness/CREDITS-NOTES.md).

**Client setup.** For Codex, Hermes, OpenCode and Pi, client setup follows the pinned command-line help and configuration docs of each client. The Hermes context exception and file-write limits come from its pinned initialization and file-safety modules. The templates, the model catalog prompt and the instruction to call a tool right away instead of promising to call it are Vade's own Rust output. No client code or prompt was copied. The source links and pins are in [the client setup guide](harness/docs/passthrough-matrix.md#sources-and-credit).

**Terminal presentation.** We compared Claude Code and Codex CLI. Then we set three clarity goals for the native terminal: a quiet answer footer, a highlighted command picker and readable Markdown. The renderer, the cell layout and the PTY regression tests are original Rust that uses only the standard library. No competitor source, assets or dependencies were used.

**Design references.** Hermes Agent, the Pi coding agent and Codex CLI informed the interface choices of the harness. They are references for ideas, not dependencies. See [Design references](harness/README.md#design-references) in the harness README.

**Design review.** The harness design review also studied stable prefixes, continuation identity, session trees, cancellation and isolated child contexts. It used these pinned sources:

- [Codex](https://github.com/openai/codex/blob/2fdf047c9631c9ed01a31b62efb7891718a931a8/README.md)
- [Pi](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/README.md)
- [Hermes](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/README.md)
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/blob/5badb15009ae1756c3afe0ae0cef1faafc290ccc/README.md)
- [OpenCode](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/README.md)
- [Aider](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/README.md)
- [Goose](https://github.com/aaif-goose/goose/blob/9560429ff982bbfecec5094963e5f34696b63a1c/README.md)
- [Claude Code](https://code.claude.com/docs/en/how-claude-code-works)

These were studies of source code and documentation. They were not interoperability measurements.

<br>

---

## Corrections

**To report a missing or wrong credit, open an issue in the repository's issue tracker.** Name the idea, its original source and the part of Vade that uses it. We also want to know about an earlier original source.

<br>

- **Back:** [README](README.md)
- **Next:** [Engine credits](engine/docs/CREDITS.md)
