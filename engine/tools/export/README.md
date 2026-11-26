# tools/export

**`vade-export`, the engine's exporter.** It copies only allowlisted files into a separate folder, after checking each one.

**What it checks:** unsafe paths, keys and tokens, credential lines, home-folder paths, network and hardware addresses, private names from a denylist kept outside the tree, and binary files.

**What it never does:** commit, push or publish.

**Scope:** the bundled allowlist is historical and incomplete for current main. Use the complete repository for current builds. Refresh the snapshot input and allowlist before using this utility; it fails closed on missing or rejected files.

**How it fits:** the full guide is [Export](../../docs/export.md); the list of files is [`scripts/export-allowlist.txt`](../../scripts/export-allowlist.txt).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
