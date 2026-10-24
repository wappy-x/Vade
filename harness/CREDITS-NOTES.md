# Sources studied

Dedicated client setup uses the pinned Codex, Hermes, OpenCode and Pi command-line
help and local configuration documentation to determine their configuration and
instruction contracts. The Hermes context exception and file-write boundary are
documented in its pinned initialization and file-safety modules. The templates,
model catalog prompt and immediate-tool instruction are original Rust output;
no client implementation or prompt was copied. Source links and pins are listed
in [the passthrough matrix](docs/passthrough-matrix.md#sources-and-credit).

Native terminal presentation follows the clarity goals in the operator's
Claude Code / Codex CLI comparison: a quiet answer footer, a highlighted command
picker, and readable Markdown. The renderer, cell layout and PTY regressions are
original std-only Rust; no competitor source, assets or dependencies were used.

Stable-prefix compaction builds on the journal and legal-cut implementation
at `35784b9`. Prefix continuity and causal suffix rebuilding follow the ideas
already credited to Codex, Pi and Hermes in [context controls](docs/context-and-projects.md)
and [Credits](../CREDITS.md#harness). The prefix
builder, admission coordination and opt-in discarded warm are original std-only
Rust. Tokenization and cache identity remain with the separate Engine; no
tokenizer implementation, competitor code or cryptography was imported.
