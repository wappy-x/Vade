# harness/src/mcp

**Rust modules for `mcp`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`cli.rs`](cli.rs) | `vade mcp list\|add\|remove\|test\|trust` and `vade tools`. |
| [`client.rs`](client.rs) | One MCP server process: JSON-RPC 2.0, one message per line on stdin and stdout. |
| [`config.rs`](config.rs) | MCP server lists: `<project>/.vade/mcp.json` and the user's `mcp.json` in the Vade home. |
| [`tests.rs`](tests.rs) | Tests for the MCP config, naming and result rules. |

The user guide is [MCP tools in Vade chat](../../../docs/mcp.md).

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `mcp`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
