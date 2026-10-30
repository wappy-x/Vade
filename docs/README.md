# Documentation

Vade's main pages, grouped by what you want to do.

- **One Spark:** start with [INSTALL.md](../INSTALL.md). It takes you from
  prerequisites to `vade launch` and a first chat, and needs no cable between
  Sparks and no network setup.
- **Two or more Sparks:** do INSTALL's steps 1 to 3 on every Spark, cable them
  with [network setup](network-setup.md), then drive them from a controller as
  INSTALL's [two and more Sparks](../INSTALL.md#two-and-more-sparks) shows.
- **The FP8 checkpoint:** it needs 2, 4 or 6 Sparks. Follow the
  [FP8 install](install-fp8.md).

## Install and operate

### On one Spark

| Guide | What it covers |
| --- | --- |
| [Installation](../INSTALL.md) | What you need first, downloading the model at its pinned revision, and launching it on the Spark. It also covers the optional controller and linked-Spark routes |
| [First chat on one Spark](../harness/docs/first-run.md) | What `vade launch` does: release bundle or source build, how it finds the model you downloaded, the settings it saves, and how to start and stop later |
| [Download models](pull-models.md) | `vade pull`: download a checkpoint from Hugging Face with resume and file checks. `vade models`: list local models and what can serve them |
| [Shell safety](shell-safety.md) | Agent commands ask by default; what the prompt shows, exact and `<path>` allow-list rules, blocked commands, `--yes`, runs without a terminal and the experimental checker |
| [CLI reference](../cli/README.md) | The `vade` commands: install, start, chat, check exactness, benchmark, diagnose, recover and stop |
| [Saved chat sessions](sessions.md) | List, resume, export and delete your `vade chat` sessions |
| [Reading files](reading-files.md) | How `vade read` and the agent turn documents (text, HTML, CSV, RTF, data, code) into Markdown with a header line |
| [MCP tools in chat](mcp.md) | Add the MCP servers you already use to `vade chat`: `vade mcp add`, `vade mcp test`, `vade tools`, the config files, approvals and limits |
| [Examples](../examples/README.md) | Short examples you can run: curl against the OpenAI and Anthropic APIs, a Rust client, Claude Code, Codex, `vade bench` and a two-Spark launch |
| [OpenAI-compatible API](openai-api.md) | Base URLs, a curl and a Python example, the chat fields Vade accepts, streaming, tool calls, errors, API keys and the known gaps. It also has notes for Continue and Open WebUI |
| [Sampling](sampling.md) | Greedy as the default, the sampling fields and their order, seeds and how to repeat a sampled reply, where sampling runs on two or more Sparks, drafting, and what is exact |

### From a controller, or with more Sparks

A controller is a computer that drives one or more Sparks over SSH with the
`vade` CLI and a config file. It is usually a second Linux or macOS computer,
but a Spark can be its own controller. The [glossary](GLOSSARY.md) explains
the other terms.

| Guide | What it covers |
| --- | --- |
| [Working from a controller](../INSTALL.md#route-b-from-a-controller) | Writing the controller's config file, then running `vade install` and `vade up` from the controller |
| [One-Spark quickstart](quickstart-one-spark.md) | From a controller: start FP4-Local on one Spark, put the gateway (`vade-gateway`, a relay between your client and the engine) in front of it, and send a first chat. An older second part builds Qwen3-8B directly on the Spark |
| [Network setup](network-setup.md) | Linking Sparks: what to buy, cabling, addresses and link checks |
| [Spark setup](spark-setup.md) | The OS and network preparation for each linked Spark, and what its checks can and cannot confirm |
| [FP4-Local on Vade's own server](../engine/docs/next-http.md) | For advanced use: starting the Flash-Next server by hand, without `vade up`, and the ways to arrange your Sparks. It is an engine reference, not a first guide |

### FP8 on 2, 4 or 6 Sparks

These pages cover the FP8 checkpoint, `Qwen/Qwen3.8-Flash-Next-FP8`. It is a
supported route on 2, 4 or 6 linked Sparks.

| Guide | What it covers |
| --- | --- |
| [FP8 install](install-fp8.md) | The FP8 route: build, weights, config, `vade up --fp8` on two and four Sparks, and the measured speeds |
| [Fresh install](fresh-install.md) | A longer step-by-step checklist and CLI configuration for FP8 on two Sparks |
| [Four Sparks](four-sparks.md) | Recabling two FP8 pairs into one group of four |

## Understand and extend

How Vade works, and how to change it. These pages are for contributors.

- **Before you change code:** the [agent rules](../AGENTS.md), the
  [contribution process](../CONTRIBUTING.md) and the
  [design decisions](decisions/README.md)
- **How the parts fit together:** [architecture](architecture.md),
  [glossary](GLOSSARY.md), [engine documentation](../engine/docs/README.md)
  and [harness documentation](../harness/README.md)
- **Long chats:** [context and compaction](context-and-compaction.md) (how a
  chat continues past the context window)
- **Adding a model:** [adding a model](adding-a-model.md), the
  [model plan](model-plan.md) (reads the checkpoint's layout without loading
  weights) and the [splitter](splitter.md) (plans how the model's tensors are
  divided)
- **Engine notes:** [typed hybrid state receipts](typed-state-qualification.md)
  (how tests compare a chat's saved state byte for byte), plus two notes on the
  GPU kernels that read prompts:
  [prompt weight reuse](prefill-weight-reuse.md) and
  [Flash-Next prompt projections](flashnext-prefill-mma.md)

## Evidence and attribution

What we measured, what backs it up, and who we credit.

- [What is new in v0.2.0, in plain words](release-notes/0.2.0.md)
- [How we benchmark, the current results and older results](benchmarks.md)
- [Which run records (receipts) are public](../bench/receipts/README.md)
- [Engine validation](../engine/docs/validation.md)
- [Older evidence we kept](history/README.md)
- [Security](../SECURITY.md), [credits](../CREDITS.md), [license](../LICENSE), [notice](../NOTICE)
- [Documentation assets](assets/README.md)
