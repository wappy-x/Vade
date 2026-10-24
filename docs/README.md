# Documentation

[INSTALL.md](../INSTALL.md) is the single installation entry point for
**nvidia/Qwen3.8-Flash-Next-FP4-Local**: start directly on one Spark with
`vade launch`, then consider a controller or more Sparks. Legacy models have
separate guides.

## Install and operate

| Guide | Scope |
| --- | --- |
| [Installation](../INSTALL.md) | Prerequisites, pinned FP4-Local download, local launch, controller setup and multiple Sparks |
| [First chat on one Spark](../harness/docs/first-run.md) | `vade launch`, model discovery, saved settings and local lifecycle |
| [Controller route](../INSTALL.md#route-b-from-a-controller) | `VADE_CONFIG` and `vade up` from another computer |
| [Legacy FP8 installation](install-legacy-fp8.md) | Separate checkpoint and launcher profile |
| [Fresh install](fresh-install.md) | Legacy FP8 installation checklist and CLI configuration |
| [One-Spark quickstart](quickstart-one-spark.md) | FP4-Local from a controller through the gateway; separate legacy Qwen3-8B Ubuntu route |
| [Native FP4-Local](../engine/docs/next-http.md) | Native Flash-Next server and topology options |
| [Network setup](network-setup.md) | Hardware, addressing and link checks |
| [Spark setup](spark-setup.md) | System preparation and its verification limits |
| [Four Sparks](four-sparks.md) | Legacy: Flash-Next FP8 configuration for a larger group |
| [CLI reference](../cli/README.md) | Install, start, diagnose, recover and stop |

## Understand and extend

- [Agent rules](../AGENTS.md), [contribution process](../CONTRIBUTING.md)
- [Architecture](architecture.md), [glossary](GLOSSARY.md), [decisions](decisions/README.md)
- [Engine documentation](../engine/docs/README.md), [harness documentation](../harness/README.md)
- [Model plan](model-plan.md), [splitter](splitter.md), [adding a model](adding-a-model.md)
- [Typed-state qualification](typed-state-qualification.md)
- [Prompt weight reuse](prefill-weight-reuse.md), [Flash-Next prompt projections](flashnext-prefill-mma.md)

## Evidence and attribution

- [Benchmark methods, current cards and historical results](benchmarks.md)
- [Measurement receipt availability](../bench/receipts/README.md)
- [Engine validation](../engine/docs/validation.md)
- [Retained historical evidence](history/README.md)
- [Security](../SECURITY.md), [credits](../CREDITS.md), [license](../LICENSE), [notice](../NOTICE)
- [Documentation assets](assets/README.md)
