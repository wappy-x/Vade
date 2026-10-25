# Benchmark receipts

A receipt is the saved record of one benchmark run: its build, its settings and
what it measured. The sanitized FP4-Local receipts behind the 2026-10-24
benchmark tables are in [2026-10-24](2026-10-24/README.md).

The [current benchmark tables](../../docs/benchmarks.md#current-results-2026-10-24)
copy their numbers from three kinds of source: RigMark cards (the result files
that the RigMark benchmark writes), the custom runner summaries for our
two-Spark pair, and separate timing runs, which the tables keep apart. Each
table row links to the sanitized copy of its source folder: the card or
summary, the README and small JSON receipts. The full card JSON receipts, raw
prompts, replies and logs are not in this repository. Every number in those
tables comes from these runs. Nothing here was measured separately for the
docs, and no receipt here was rebuilt from a card summary.

The older FP8 summaries have been removed from this folder:
`night-2026-10-11.json`, `defaults-2026-10-12.json`, `install-canary.json`,
`exact-path-abba.json`, `exact-path-gate.json`, `determinism.json` and
`context-candidate.json`. The
[historical section](../../docs/benchmarks.md#historical-fp8-measurements) of
the benchmarks page still names some of them as retired receipts.

If you add a receipt to this folder, keep it small and have it reviewed.

- **Include** the measured build, checkpoint identity, request settings,
  drafting mode, completion-token counts, wall-clock measurements and
  limitations.
- **Leave out** hostnames, usernames, machine addresses, private filesystem
  paths, credentials and raw conversations.

Keep raw [benchmark output](../../ops/scripts/tools/README.md) private until
it has been reviewed for publication.
