# Benchmark receipts

A receipt is the saved record of one benchmark run: its build, its settings and what it measured. Each folder here holds sanitized extracts of the run records for one date. [Results](../../docs/results.md) is the current page, and it links each folder.

These are the folders, newest first:

- **[2026-11-01](2026-11-01/README.md).** Current. FP8 on two and four Sparks, at main `acb49bc` (afternoon). The morning cards at main `1a07e3a` are history, but its chooser cards are still the newest.
- **[2026-10-31](2026-10-31/README.md).** Current for FP4-Local on one, two and four Sparks, at main `dc32b7c`. Its FP8 receipts are history: the 2026-11-01 receipts replace them.
- **[2026-10-24](2026-10-24/README.md).** History. FP4-Local receipts behind the v0.1.0 tables, in the [2026-10-24 section](../../docs/benchmarks.md#current-results-2026-10-24) of the benchmark history.

The 2026-10-24 tables copy their numbers from three kinds of source: RigMark cards (the result files that the RigMark benchmark writes), the custom runner summaries for our two-Spark pair, and separate timing runs, which the tables keep apart. Each table row links to the sanitized copy of its source folder: the card or summary, the README and small JSON receipts.

In every folder, the full card files, raw prompts, replies and logs are not in this repository. Every number in the tables comes from these runs. Nothing here was measured separately for the docs, and no receipt here was rebuilt from a card summary.

FP8 receipts from before 2026-10-31 are retired and not in this folder. The
[historical section](../../docs/benchmarks.md#historical-fp8-measurements) of
the benchmarks page still quotes some of them by name.

If you add a receipt to this folder, keep it small and have it reviewed.

- **Include** the measured build, checkpoint identity, request settings,
  drafting mode, completion-token counts, wall-clock measurements and
  limitations.
- **Leave out** hostnames, usernames, machine addresses, private filesystem
  paths, credentials and raw conversations.

Keep raw [benchmark output](../../ops/scripts/tools/README.md) private until
it has been reviewed for publication.
