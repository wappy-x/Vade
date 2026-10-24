# Benchmark receipts

Sanitized FP4-Local receipts for the 2026-10-24 benchmark tables are in
[2026-10-24](2026-10-24/README.md). The stale FP8 summaries have been removed:
`night-2026-10-11.json`, `defaults-2026-10-12.json`, `install-canary.json`,
`exact-path-abba.json`, `exact-path-gate.json`, `determinism.json` and
`context-candidate.json`.

The [current benchmark tables](../../docs/benchmarks.md#current-2026-10-24)
transcribe the named RigMark cards and distinguish the separate timing runs.
Each table row links to the sanitized copy of its source folder: the card,
README and small JSON receipts. The full card JSON receipts, raw prompts,
replies and logs are not distributed in this repository. No inference was
repeated to prepare these docs, and no replacement receipt was fabricated
from a card summary.

Keep future receipts small and reviewed. Exclude hostnames, usernames, machine
addresses, private filesystem paths, credentials and raw conversations. Include
the measured build, checkpoint identity, request settings, drafting mode,
completion-token counts, wall-clock measurements and limitations. Store raw
[benchmark output](../../ops/scripts/tools/README.md) privately until it has been
reviewed for publication.
