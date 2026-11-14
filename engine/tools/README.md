# tools

**Programs that support the engine but are not part of it.**

| Item | What it does |
| --- | --- |
| [`bench/`](bench/README.md) | `vade-bench`: measures a chat server from the outside, over HTTP, the way an app sees it. |
| [`export/`](export/README.md) | `vade-export`: copies an allowlisted, checked snapshot of the engine. |
| [Official reference provenance](truth/README.md) | Saved-fixture methods; the Python oracle is retained privately outside this repository. |
| [`oracles/`](oracles/README.md) | Host proof and one-Spark A/X gate for MoE prefill rasterization. |
| [`draft-prompts.rs`](draft-prompts.rs) | Writes a fixed set of tokenized test prompts (code, quotes, JSON, prose) for drafting checks. |

**How it fits:** none of these run inside the server. They measure it, check it, or package it.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
