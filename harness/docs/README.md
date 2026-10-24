# Harness docs

**Every harness guide, one line each.** New here? Start with the [harness README](../README.md).

| Guide | What it covers |
| --- | --- |
| [First run](first-run.md) | First chat on one Spark from one terminal: launch with existing model files, saved defaults and repeat runs. |
| [Agent tools](agent-tools.md) | The native agent's tools and session contract: read, list, exact edits, approved commands. |
| [Context and projects](context-and-projects.md) | Project rules files, context limits and how long sessions are kept in bounds. |
| [Child agents](child-agents.md) | Bounded helper agents a native session can start, list and stop. |
| [Engine contract](engine-contract.md) | Capabilities, admission and cache reporting agreed with the engine. |
| [Engine checkpoints](engine-checkpoints.md) | Optional saved prompt states on the engine, to skip re-reading. |
| [Bridge contract](bridge-contract.md) | Handing a turn to an installed Codex or Hermes agent. |
| [Gateway routing](gateway-routing.md) | Cache-aware routing across several engines. |
| [Other clients](passthrough-matrix.md) | Using Codex, Hermes, OpenCode or Pi through the gateway. |
| [Setup and doctor](setup-doctor.md) | Writing client configurations, and checking an endpoint. |
| [Workload](workload.md) | Measuring many concurrent agent sessions. |
| [Performance](performance.md) | Local mock measurements and the prompt-prefix contract. |

Raw measurement output: [performance-samples.txt](performance-samples.txt), [performance-contract-samples.txt](performance-contract-samples.txt).

Research scope and attribution: [research/](research/README.md).

**How it fits:** the harness is a client of the [engine](../../engine/README.md); see [Architecture](../../docs/architecture.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
