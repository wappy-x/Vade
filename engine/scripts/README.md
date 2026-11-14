# scripts

**Source export allowlist and Rust gate references.**

| Item | What it does |
| --- | --- |
| [`export-allowlist.txt`](export-allowlist.txt) | The only files the exporter may copy into a shareable snapshot. |
| [`gates/`](gates/README.md) | Guide to Rust checks that must pass before a change is merged. |

**How it fits:** the exporter is [`tools/export`](../tools/export/README.md); the gates are explained in [Validation](../docs/validation.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
