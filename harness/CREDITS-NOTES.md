# Sources studied

For Claude Code, Codex, Hermes, OpenCode and Pi, client setup follows each
client's own pinned command-line help and configuration docs. Hermes's context
exception and file-write limits come from its pinned initialization and
file-safety modules. The templates, the model catalog prompt and the instruction
to call a tool right away instead of promising to are Vade's own Rust output; no
client code or prompt was copied. Source links and pins are listed in [the
passthrough matrix](docs/passthrough-matrix.md#sources-and-credit).

Native terminal presentation follows the clarity goals we set after comparing
Claude Code and Codex CLI: a quiet answer footer, a highlighted command picker,
and readable Markdown. The renderer, cell layout and PTY regression tests are
original std-only Rust; no competitor source, assets or dependencies were used.

Stable-prefix compaction builds on the harness's own session journal and the
points where it can safely cut history. Prefix continuity and causal suffix
rebuilding follow the ideas already credited to Codex, Pi and Hermes in [context
controls](docs/context-and-projects.md) and [Credits](../docs/CREDITS.md#harness).
The prefix builder, admission coordination and opt-in discarded warm are
original std-only Rust. Tokenization and cache identity remain with the separate
Engine; no tokenizer implementation, competitor code or cryptography was
imported.
