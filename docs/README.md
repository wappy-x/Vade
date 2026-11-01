# Documentation

These are Vade's main pages, grouped by what you want to do. [Why Vade](why-vade.md) says what Vade is, why it exists and what "exact" means.

- **One Spark.** Run `curl -fsSL https://vade.lol/install.sh | sh`, then `vade pull` and `vade up`. [INSTALL.md](../INSTALL.md) has the full steps, the source build and every setting. You need no cable between Sparks and no network setup.
- **Two or more Sparks.** Do the INSTALL steps 1 to 3 on every Spark. Cable the Sparks as [network setup](network-setup.md) shows. Then drive them from a controller, as [two and more Sparks](../INSTALL.md#two-and-more-sparks) shows.
- **The FP8 checkpoint.** FP8 needs 2, 4 or 6 Sparks. Follow the [FP8 install](install-fp8.md).

## Install and operate

### On one Spark

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [Installation](../INSTALL.md) | The full install, settings and recovery |
| [Agent quickstart](agents/quickstart.md) | Setup steps written for coding agents |
| [First chat on one Spark](../harness/docs/first-run.md) | What `vade launch` does and the settings it saves |
| [Download models](pull-models.md) | `vade pull` and `vade models` |
| [Shell safety](shell-safety.md) | How the agent asks before it runs a command |
| [CLI reference](../cli/README.md) | Every `vade` command |
| [Saved chat sessions](sessions.md) | List, resume, export and delete your `vade chat` sessions |
| [Reading files](reading-files.md) | How `vade read` and the agent turn documents into Markdown |
| [MCP tools in chat](mcp.md) | How to add the MCP servers you already use to `vade chat` |
| [Examples](../examples/README.md) | Short scripts against a running engine |
| [OpenAI-compatible API](openai-api.md) | Endpoints and request fields |
| [Sampling](sampling.md) | The greedy default, seeds and what is exact |

</div>

### From a controller, or with more Sparks

A controller is a computer that drives one or more Sparks over SSH with the `vade` CLI and a config file. It is usually a second Linux or macOS computer. A Spark can also be its own controller. The [glossary](GLOSSARY.md) explains the other terms.

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [Working from a controller](../INSTALL.md#route-b-from-a-controller) | The controller config, `vade install` and `vade up` |
| [One-Spark quickstart](quickstart-one-spark.md) | FP4-Local on one Spark from a controller |
| [Network setup](network-setup.md) | Parts, cables, addresses and link checks |
| [Spark setup](spark-setup.md) | The OS and network preparation for each linked Spark |
| [Eight Sparks](eight-sparks.md) | Eight Sparks cabled as one ring |
| [FP4-Local on Vade's own server](../engine/docs/next-http.md) | The engine server started by hand |

</div>

The one-Spark quickstart puts `vade-gateway` in front of the engine. The gateway is a relay between your client and the engine. An older second part of that page builds Qwen3-8B directly on the Spark.

### FP8 on 2, 4 or 6 Sparks

These pages cover the FP8 checkpoint, `Qwen/Qwen3.8-Flash-Next-FP8`. It is a supported route on 2, 4 or 6 linked Sparks.

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [FP8 install](install-fp8.md) | FP8 on two and four Sparks |
| [Fresh install](fresh-install.md) | A longer checklist for FP8 on two Sparks |
| [Four Sparks](four-sparks.md) | How to recable two FP8 pairs into one group of four |

</div>

## Understand and extend

These pages explain how Vade works and how to change it. They are for contributors.

- **Before you change code.** Read the [agent rules](../AGENTS.md), the [contribution process](../CONTRIBUTING.md) and the [design decisions](decisions/README.md). [Local CI](ci.md) gives the CI commands to run on your own computer.
- **How the parts fit together.** Read the [architecture](architecture.md) and the [glossary](GLOSSARY.md). The [engine documentation](../engine/docs/README.md) and the [harness documentation](../harness/README.md) go deeper.
- **Long chats.** [Context and compaction](context-and-compaction.md) explains how a chat continues past the context window.
- **Adding a model.** Start with [adding a model](adding-a-model.md). The [model plan](model-plan.md) reads the checkpoint's layout without loading weights. The [splitter](splitter.md) plans how the model's tensors are divided.
- **Engine notes.** [Typed hybrid state receipts](typed-state-qualification.md) explains how tests compare a chat's saved state byte for byte. Two notes cover the GPU kernels that read prompts: [prompt weight reuse](prefill-weight-reuse.md) and [Flash-Next prompt projections](flashnext-prefill-mma.md).
- **Install design.** [Install and flows](design/install-and-flows.md) gives the steps from install to a running engine, as a person or an agent sees them.

## Evidence and attribution

These pages give what we measured, what backs it up, and who we credit.

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [Results](results.md) | Every current benchmark cell, with its build, settings and receipt |
| [Draft notes for the next release](release-notes/0.2.0.md) | Notes for v0.2.0, which is not released yet |
| [Benchmarks](benchmarks.md) | How we benchmark, the current results and older results |
| [Agent task suite](harness-suite.md) | 22 tasks for the coding agent and its grades |
| [Receipts](../bench/receipts/README.md) | Which run records (receipts) are public |
| [Engine validation](../engine/docs/validation.md) | How the engine is checked, on any computer and on a Spark |
| [History](history/README.md) | Older evidence we kept |
| [Security](../SECURITY.md) | How to report a vulnerability privately |
| [Credits](../CREDITS.md) | The work Vade builds on |
| [License](../LICENSE) | The Apache-2.0 license |
| [Notice](../NOTICE) | The licenses of the bundled files |
| [Documentation assets](assets/README.md) | The README banner and the benchmark figure |

</div>
