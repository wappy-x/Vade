# Documentation

Vade's main pages, grouped by what you want to do.

If you have one DGX Spark, the first table is all you need. Start with
[INSTALL.md](../INSTALL.md), the single install guide for
**nvidia/Qwen3.8-Flash-Next-FP4-Local**: you download the model, run
`vade launch` on the Spark, and chat. A one-Spark launch needs no cable between
Sparks and no network setup. Driving the Spark from a second computer, or
linking more Sparks, comes later and has its own pages below.

## Install and operate

### On one Spark

| Guide | What it covers |
| --- | --- |
| [Installation](../INSTALL.md) | What you need first, downloading the model at its pinned version, and launching it on the Spark. It also covers the optional controller and linked-Spark routes |
| [First chat on one Spark](../harness/docs/first-run.md) | What `vade launch` does: how it finds the model you downloaded, the settings it saves, and how to start and stop later |
| [CLI reference](../cli/README.md) | The `vade` commands: install, start, diagnose, recover and stop |

### From a controller, or with more Sparks

A controller is a computer, usually a second Linux or macOS computer, that
drives your Sparks over SSH with the `vade` CLI. The [glossary](GLOSSARY.md)
explains the other terms on these pages.

| Guide | What it covers |
| --- | --- |
| [Working from a controller](../INSTALL.md#route-b-from-a-controller) | Running `vade up` from the controller, which reads its settings from the config file that `VADE_CONFIG` points to |
| [One-Spark quickstart](quickstart-one-spark.md) | From a controller: start FP4-Local on one Spark, put the gateway (the harness mode that relays requests from other coding tools) in front of it, and send a first chat. A second, older part builds a different model, Qwen3-8B, directly on the Spark |
| [Network setup](network-setup.md) | Linking Sparks: what to buy, cabling, addresses and link checks |
| [Spark setup](spark-setup.md) | The OS and network preparation for each linked Spark, and what its checks can and cannot confirm |
| [FP4-Local on Vade's own server](../engine/docs/next-http.md) | Starting the FP4-Local checkpoint on Vade's Flash-Next server by hand, and the ways to arrange your Sparks |

### Legacy models

These pages cover the older FP8 checkpoint. A new install does not need them.

| Guide | What it covers |
| --- | --- |
| [Legacy FP8 installation](install-legacy-fp8.md) | The older FP8 install, which has its own checkpoint and launcher profile |
| [Fresh install](fresh-install.md) | A step-by-step checklist and CLI configuration for the FP8 install on two Sparks |
| [Four Sparks](four-sparks.md) | Turning two FP8 pairs into one group of four |

## Understand and extend

How Vade works, and how to change it. These pages are for contributors.

- **Before you change code:** the [agent rules](../AGENTS.md), the
  [contribution process](../CONTRIBUTING.md) and the
  [design decisions](decisions/README.md)
- **How the parts fit together:** [architecture](architecture.md),
  [glossary](GLOSSARY.md), [engine documentation](../engine/docs/README.md)
  and [harness documentation](../harness/README.md)
- **Adding a model:** [adding a model](adding-a-model.md), the
  [model plan](model-plan.md) (reads the checkpoint's layout without loading
  weights) and the [splitter](splitter.md) (plans how the model's tensors are
  divided)
- **Engine notes:** [typed-state qualification](typed-state-qualification.md)
  (how the engine accounts for each chat's saved state), plus two notes on the
  GPU kernels that process prompts:
  [prompt weight reuse](prefill-weight-reuse.md) and
  [Flash-Next prompt projections](flashnext-prefill-mma.md)

## Evidence and attribution

What we measured, what backs it up, and who we credit.

- [How we benchmark, the current results and older results](benchmarks.md)
- [Which run records (receipts) are public](../bench/receipts/README.md)
- [Engine validation](../engine/docs/validation.md)
- [Older evidence we kept](history/README.md)
- [Security](../SECURITY.md), [credits](../CREDITS.md), [licence](../LICENSE), [notice](../NOTICE)
- [Documentation assets](assets/README.md)
