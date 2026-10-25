# Documentation

Start with [INSTALL.md](../INSTALL.md), the single install guide for
**nvidia/Qwen3.8-Flash-Next-FP4-Local**. You begin on one Spark with
`vade launch`, and later you can add a controller (a separate computer that
drives your Sparks) or more Sparks. Legacy models have their own guides, listed
at the end of the table below.

## Install and operate

| Guide | What it covers |
| --- | --- |
| [Installation](../INSTALL.md) | What you need first, downloading the model at its pinned version, launching on the Spark, setting up a controller, and linking more Sparks |
| [First chat on one Spark](../harness/docs/first-run.md) | `vade launch`, how it finds the model, saved settings, and starting and stopping on the Spark |
| [Working from a controller](../INSTALL.md#route-b-from-a-controller) | Running `vade up` from another computer, set up with `VADE_CONFIG` |
| [One-Spark quickstart](quickstart-one-spark.md) | Running FP4-Local from a controller through the gateway, plus a separate older setup for Qwen3-8B on Ubuntu |
| [FP4-Local on Vade's own server](../engine/docs/next-http.md) | Running the FP4-Local checkpoint on Vade's Flash-Next server, and the ways to arrange your Sparks |
| [Network setup](network-setup.md) | Hardware, addresses and link checks |
| [Spark setup](spark-setup.md) | Preparing the system, and what the setup checks can and cannot confirm |
| [CLI reference](../cli/README.md) | Install, start, diagnose, recover and stop |
| [Legacy FP8 installation](install-legacy-fp8.md) | The older FP8 install, which has its own checkpoint and launcher profile |
| [Fresh install](fresh-install.md) | A checklist and CLI configuration for the legacy FP8 install |
| [Four Sparks](four-sparks.md) | Legacy: the Flash-Next FP8 setup for a larger group |

## Understand and extend

How Vade works, and how to change it. These pages are for contributors.

- [Agent rules](../AGENTS.md), [contribution process](../CONTRIBUTING.md)
- [Architecture](architecture.md), [glossary](GLOSSARY.md), [decisions](decisions/README.md)
- [Engine documentation](../engine/docs/README.md), [harness documentation](../harness/README.md)
- [Model plan](model-plan.md) (reads the checkpoint's layout without loading weights),
  [splitter](splitter.md) (plans how the model's tensors are divided),
  [adding a model](adding-a-model.md)
- [Typed-state qualification](typed-state-qualification.md) (how the engine accounts for each chat's saved state)
- [Prompt weight reuse](prefill-weight-reuse.md) and
  [Flash-Next prompt projections](flashnext-prefill-mma.md) (two notes on GPU kernels that process prompts)

## Evidence and attribution

What we measured, what backs it up, and who we credit.

- [How we benchmark, the current results and older results](benchmarks.md)
- [Which run records (receipts) are public](../bench/receipts/README.md)
- [Engine validation](../engine/docs/validation.md)
- [Older evidence we kept](history/README.md)
- [Security](../SECURITY.md), [credits](../CREDITS.md), [licence](../LICENSE), [notice](../NOTICE)
- [Documentation assets](assets/README.md)
