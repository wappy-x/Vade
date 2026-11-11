# flywheel

**Measures:** several runtime settings in one pair of processes that load the checkpoint once. Head and candidate settings alternate (H, A, H, B, H, ...). Each setting gets an untimed warmup and three timed repeats per input length. The audit reports exactness against the first head, the speed of each cell against its adjacent heads with a noise floor, declared counters and live bytes.

**Parts:** `flywheel4.rs` prepares, claims, launches, collects and releases. `flywheel4_receipts.rs` audits bytes and builds the paired speed table. `flywheel4_counters.rs` freezes the counter declarations. `flywheel-retain.rs` prunes and verifies receipts. `run-flywheel4.sh` cleans up on exit. The native side is `resident-decode-check --jobs=FILE` (one load, many jobs; on main). The per-job `--flywheel=` settings exist only on the screen's own native branch.

**Build,** in this folder: `rustc +1.99.0 --edition=2024 -D warnings -O flywheel4.rs -o flywheel4`, and the same for `flywheel-retain.rs`. For the tests, add `--test` and run the output (9 driver tests, 4 retain tests). Set `VADE_KIT_HOSTS` to the private copy of [hosts.example](../hosts.example).

**Run:** `./flywheel4 prepare REV TAG A,B short`, then `bash run-flywheel4.sh <receipt>`, then `./flywheel-retain verify <receipt>`. This copy is the run-4 driver (base `a578ac0b4`, defer-gather and conv control). For a new setting, copy it to new versioned files.

**Prints:** with no arguments, `FLYWHEEL ERROR usage: ...` and exit 1. A run prints the rows of `TABLE.tsv` and appends one log line per event. The only changes from the reviewed copy move paths, box pairs and the build box to the hosts file ([SOURCES.txt](../SOURCES.txt)).
