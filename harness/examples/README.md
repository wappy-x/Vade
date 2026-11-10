# harness/examples

**Small runnable programs that show one harness feature each, offline.**

| Example | What it shows |
| --- | --- |
| [`child_agents_mock.rs`](child_agents_mock.rs) | A parent agent and its child agents, with no files, network or model. |
| [`perf_mock.rs`](perf_mock.rs) | A local benchmark against a synthetic upstream, using the real CLI output. |
| [`prompt_memory.rs`](prompt_memory.rs) | How much memory a native session uses, with scripted tools and model. |

Run one with `cargo run --offline --example <name>`.

**How it fits:** the measurements they produce are described in [Performance](../docs/performance.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
