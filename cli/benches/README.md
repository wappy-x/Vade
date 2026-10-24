# Owner attach benchmark

On an idle Mac, run `python3 cli/benches/owner_attach.py` from the checkout.
This optional benchmark discards one warmup, runs 20 sequential samples, and
compares the 95th percentile with a one-second target. Use `--samples` or
`--target-seconds` to state a different measurement contract. Keep other builds
and tests stopped while measuring, and retain the host/load details with results.

The fixture reuses saved state and runs three real subprocesses as worker,
health, and chat doubles. It does not measure a GPU engine or inference.
Correctness tests assert unchanged records and exact calls independently of
performance. A parent process enforces a 30-second deadline for the attach
fixture and kills/reaps its process group on timeout, including interactive
chat. A deliberately blocked chat regression checks descendant cleanup. The
listener and health doubles have their own five-second command deadlines.
