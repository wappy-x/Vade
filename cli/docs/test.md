# Build and exactness checks with vade test

`vade test build` checks that the engine builds, and that a source snapshot
builds too. The snapshot is a copy of the files named in the export
allowlist. The export stops if any of those files contains a name from your
private denylist, an untracked file of host aliases, usernames and other local
names. `vade test determinism` checks that a fixed setup returns the same
answer bytes whether a request runs alone, batched, cached, chunked or with
multi-token prediction (MTP, also called drafting).

Both commands are part of the CLI, written in Rust on the standard library.
They drive two existing Rust tools: the exporter `vade-export`, which writes
the snapshot, and the `flash-bench` benchmark, which runs and compares the
GPU cases.

This determinism runner uses the older `flash-bench` model path and fixtures.
Its one-Spark option does not prove that it covers FP4-Local. Do not use an
FP8 fixture pass as the FP4-Local device gate. `vade verify` checks token IDs
on a running server, but it does not cover the full batch, cache and chunk
matrix either.

Two words come up often on this page. A *gate* is a set of checks a change
must pass before it can merge. A *receipt* is the saved record of a run and
what it checked.

The CLI's own tests use stand-in programs to check how these commands start
and supervise their steps. The GPU cases, and receipts from two Sparks, need a
run on real Sparks. The full exactness gate in
[CONTRIBUTING.md](../../CONTRIBUTING.md#exactness-and-evidence) decides which
checks qualify a change. These two commands replace older shell scripts;
[the appendix](#appendix-the-retired-shell-gates) maps each old step to the
new command.

Build `vade` from the root of the monorepo with the pinned host compiler:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
cli/target/release/vade test build --help
cli/target/release/vade test determinism --help
```

## What you provide

The commands run before any lifecycle configuration or authentication
handling, so they need no `VADE_CONFIG`, `VADE_HOME` or API key file. They
don't read shell configuration or guess which machine, model, endpoint or
server to use. You pass every path, binary and fixture explicitly, export the
runtime environment, start each rank's process and start the server when a
test needs one. Use `vade up` to start the server that Vade manages, and
`vade down` to stop it. Keep deployment plans and raw results outside Git.

`vade test` prints its messages in the C/POSIX locale and runs every child
command with `LC_ALL=C`, whatever your locale is. Timeout receipts use the
same English text and ASCII quotes as GNU timeout in that locale.

## Build gate

From the monorepo root or `engine/`, run:

```sh
vade test build --dry-run
vade test build --denylist "$PRIVATE_DENYLIST"
```

`VADE_ROOT` picks the checkout when you run from somewhere else. If you set it
explicitly to an empty or invalid root, the command fails before Cargo runs;
if you leave it unset, the checkout is found from the current directory. Help
and dry-run need no valid checkout. The selected engine directory (the root
itself, or its `engine/` child) must have its own `Cargo.toml` and
`tools/export/Cargo.toml`. An exporter marker on its own does not qualify,
even inside a valid checkout.

The denylist must be an external, readable, nonempty private file, and its
contents are never printed. Follow [source export](../../engine/docs/export.md)
and use a clean, committed checkout. Every Cargo command runs offline. The gate
builds in the existing engine target directory, then builds an exported
snapshot in its own temporary target directory. It does not install a
toolchain, run inference or publish the snapshot.

On a Mac, the first step checks the host kernel builder without generating
NVPTX. On Linux aarch64, it generates kernels with the separately pinned
nightly, and then generates kernels inside the snapshot too. A Mac PASS is a
host build result, not a GPU result. For a native qualification (qualifying a
build on the Sparks themselves), you still need to compare the server binary
and PTX hashes across every selected rank.

Options and environment:

| Option or input | What it does |
| --- | --- |
| `--help`, `-h` | Print help, exit 0, perform no steps. |
| `--denylist FILE` | Select the external private denylist; overrides `PRIVATE_DENYLIST`. |
| `PRIVATE_DENYLIST` | Default denylist path when `--denylist` is absent. |
| `--features FEATURES` | Forward each occurrence unchanged to the source release build, snapshot release build and final `cargo check`. |
| `--no-default-features` | Forward unchanged to those same three commands. It disables the gate's per-row capability, not engine/server per-row rejection. |
| `--dry-run` | Print the command preview without running Cargo, creating temporary files or printing PASS lines. |
| Unknown option or missing option value | Print input failure and exit 2. |
| Working directory | Resolve `VADE_ROOT` or the current checkout root/engine directory, then run Cargo from `engine/`. |
| `CARGO_NET_OFFLINE=true` | Set for gate child commands in addition to their existing `--offline` arguments. |
| `TMPDIR` | Select temporary storage using the host's existing temporary-directory setting. |
| Pinned host and kernel toolchains, optional `KERNEL_BIN` | Use the pinned toolchains as they are; install nothing. |

## Determinism gate

A rank is one Spark's process. Run the native form (the GPU run on the Sparks
themselves) on both ranks at the same time. Only `RANK` and the new rank-local
dump directory change between them.

The test needs three fixtures (saved test inputs): prompts of 8192, 2048 and
285 tokens. Each needs at least 64 frozen continuation labels, the saved next
tokens the test feeds in. Pass the fixtures, and any retained HTTP bodies, in
the order **8192,2048,285**. The HTTP phase finishes each prompt before it
sends the mixed cohort (a group of different requests sent together), so this
order makes the 285-token prompt the most recently cached one when the cohort
starts.

Set the same fresh `TP_RUN_ID` on both ranks, and use the launcher environment
for these Sparks: a private `SEQPAR_CONTROL_SECRET`, `SEQPAR_RDMA=1`,
`TP_TOPOLOGY=ring`, `SPARK_PROFILE=0` and the memory budget. Then run:

```sh
vade test determinism --require-poison \
  --bench "$BENCH" --ptx "$PTX" --dump-dir "$DUMP" --groups 256,512 \
  -- "$MODEL" "$FIXTURES" "$ADDR0" "$ADDR1" "$RANK" --world 2
```

| Variable | What it is |
| --- | --- |
| `$BENCH` | The `flash-bench` executable |
| `$PTX` | The PTX file |
| `$DUMP` | A new rank-local directory |
| `$MODEL`, `$FIXTURES` | The model and the fixture list |
| `$ADDR0`, `$ADDR1` | The shared pair endpoints |
| `$RANK` | 0 or 1 |

The option table further down describes each flag.

On one Spark, pass `--world 1` and rank 0. You still have to give both
address arguments. At startup each rank prints a *seal line*,
`registrations sealed before weights rank=R world=W rdma=...`, which records
the rank and world size it started as. A one-Spark run expects that line to
end in `rdma=false`, and with `--url` it takes a single `--server-logs` entry.

The gate has two phases. The native phase runs the logit checks on the GPU.
The optional HTTP phase then sends requests from rank 0 to a running server.
Before rank 0 starts the HTTP phase, the native runs free the GPU memory their
model used. You start the usual server yourself and keep both startup logs;
the gate never starts or stops that server. To run both phases in one go, add
`--url`, `--server-logs`, optional `--bodies` and `--wait-server` before `--`.
Or run the separate server form after the native run completes:

```sh
vade test determinism --server-only --bench "$BENCH" --url "$URL" \
  --server-logs "$LOG0" "$LOG1" --fixtures "$FIXTURES" --bodies "$BODIES"
```

For four ranks, follow [the world-four procedure below](#world-four-goldens).

The poison case rejects one row (one request slot in the batch) on purpose
and checks that the other rows are unchanged. `--require-poison` needs a
completed native poison PASS in the same invocation. A server-only run has no
native poison result, so it fails with that option. For a merge receipt that
needs both, use the combined native and HTTP form.

Without `--bodies`, the server receives each fixture's exact prompt token IDs
through `/v1/completions`. With bodies, it receives the retained chat requests
through `/v1/chat/completions`. For TLS, open a local tunnel and pass its root
HTTP URL; the gate adds no endpoint of its own. Sampling and accounting checks
stay in the existing `flash-bench` HTTP client.

Options and environment:

| Option or input | What it does |
| --- | --- |
| `--help`, `-h` | Print help, exit 0 and perform no work. |
| `--bench BIN` | Path to the `flash-bench` executable (required). The gate uses its capability check, native runs, server client and byte comparator. |
| `--ptx PTX` | Require an existing PTX file for native mode. |
| `--dump-dir DIR` | Require a new rank-local directory; retain dumps, MTP logs and reference manifest there. |
| `--groups N,N` | Two prefill group sizes, default `256,512`; the native gate requires distinct multiples of 64 up to 512. |
| `--require-poison` | Turn poison SKIP into FAIL and exit 1, including in server-only mode. |
| `--url URL` | Rank zero checks `/health`, then runs the existing HTTP gate. Require every rank's server log. |
| `--server-logs LOG0 LOG1 [LOG2 LOG3]` | One readable startup log per selected rank, in rank order, checked after `/health` succeeds. Each must contain that rank's own seal line with the right rank, world and RDMA setting. |
| `--bodies REQUEST.json,...` | Forward matching retained chat bodies in fixture order; otherwise use fixture prompt token IDs. |
| `--wait-server SECONDS` | Health wait in whole seconds, default 0; each health attempt times out after two seconds, with one second between attempts. |
| `--server-only` | Skip native work; require `--fixtures`, `--url`, no native positionals and every server log. World defaults to 2; pass `--world 1` or `--world 4` for other sizes. |
| `--fixtures FIXTURE.json,...` | Select fixtures for server-only mode. Native mode uses the positional fixture list. |
| `--selftest` | No GPU: corrupt one byte of a full vocabulary dump and require the existing comparator's expected FAIL and detail. Incompatible with `--require-poison`. |
| `-- MODEL FIXTURES ADDR0 ADDR1 RANK [--world 1\|2\|4]` | World defaults to 2: shared pair endpoints and rank 0/1. World 1: rank 0 on one Spark. World 4: this rank's LISTEN/NEXT endpoints and rank 0..3. A leading `--world` is also accepted; conflicting leading/trailing sizes fail. |
| `TP_RUN_ID` | Require a fresh value, the same on every rank, for native work; append `-mtp-0`, `-mtp-3`, `-mtp-15` for the three runs. A missing or empty value exits 1. |
| `TMPDIR` | Select temporary storage for the selftest and captured helper output using the host's existing setting. |
| Private transport/model runtime environment | Inherit unchanged; you choose the hardware and memory limits. For world 4, `TP_TOPOLOGY=path` also adds native `--path`; `ring` or unset uses the native ring. Other topology values fail. |
| `VADE_TEST_ROUTING_REJECT` | Set by the gate, not by you: the compiled poison hook for MTP 0, and `0` for MTP 3 and 15. The gate detects whether the build expects the older `1:3` form or the width-aware `1:3:16` form. |
| Missing option value, unknown option or invalid checked input | Print input failure and exit 2; native environment/setup failures retain exit 1. |

Your `flash-bench` build defines every native and server case. The CLI only
starts and supervises the checks; it doesn't change their arithmetic or the
byte comparator. Full vocabulary rows are little-endian FP32, and every byte
is compared, signed zero included.

In the table below, a *lane* is one request slot in a batched run.
*Teacher-forced* means the test feeds the saved labels in as the next inputs,
instead of the model's own choices; the frozen continuation labels pick those
positions. In the results, REF means the rows were captured as the reference
and nothing was compared yet.

| Case | What the Rust code does, and the evidence |
| --- | --- |
| Cold batch one, MTP 0 | Captures 64 positions per fixture as the reference. 192 rows per rank, shown as REF with zero bytes compared. |
| Homogeneous widths 2/8/16 | Prefills one row, restores its snapshot into 2, 8 or 16 independent sessions, captures only the lead row's prompt end, and compares every row's 63 actual decode outputs. A copied prompt-end value never counts as a PASS comparison. |
| Groups 256/512 | Compares both prefill group sizes with the same one-row reference. Attention keeps its 64-token tile. |
| Cold/cache replay | Compares a replay from a saved and restored snapshot with the cold reference. |
| Follow-up | After 63 decode inputs, canonicalizes the session and restores it exactly into a fresh one, then compares one more input with a cold prefill of the prompt plus all 64 labels. |
| MTP 3 cold and replay | Uses real proposals, commits the accepted prefix and rolls back the rest. Reports proposals and acceptances, and fails when none is accepted. |
| MTP 15 rows 4/8/16 | Restores each prompt into fresh sessions and compares all 63 teacher-forced positions, from verification launches at depths 3, 7 and 15, with the one-row reference. |
| Heterogeneous native widths 3..16 | Cycles through batch widths 3 to 16 and compares each lane with its position run alone. 577 rows, 573,122,560 bytes per rank. |
| Shifted native mixed16 | Cycles the prompt lengths 8192, 2048 and 285, moves lane `r` ahead by `r` labels, then makes 48 full-width passes. 768 rows, 762,839,040 bytes per rank. |
| Mixed16 poison | Rejects row 3 on purpose during the first 16-row pass, keeps that slot as padding, and checks that the other rows match a mixed16 run without the hook. 720 rows, 715,161,600 bytes per rank: the first call compares 15 rows (14,899,200 bytes), the remaining 47 compare 705 rows (700,262,400 bytes). |
| Mixed16 bounds | Runs under the same shared 120-second work cap, checked between calls, with one copy of the weights and the previous peak-memory bound. The cap on the whole invocation also catches hangs. |
| HTTP cold/prime/hit | For each fixture, takes a cold reference with the cache off, then a prime and a hit with the cache on. The cold run must reuse zero tokens, and the hit must reuse tokens. |
| HTTP homogeneous cohorts 2/8/16 | Sends the same cached request 2, 8 or 16 times at once, greedy (`temperature=0`, `seed=0`), with plain drafting and a 64-token limit. Every reply must match the cold reference. A cache miss is logged, not failed, because concurrent requests can evict idle prefixes. |
| HTTP mixed cohort | Sends the three prompt lengths together, in fixture order, and compares them with their cold responses. A cache miss is logged, not failed. |
| HTTP comparisons | Compares token IDs, finish reason, and prompt, completion and total usage. Missing replies, IDs or usage fail. A natural end of sequence (EOS) counts in usage but has no streamed ID. |

Native dump names stay
`DIR/CASE/prompt-NNNNN/session-NNNNN/position-NNNNN-rank-R.f32`.
The extra mixed16 captures need 762,839,040 bytes per rank, or 1,478,000,640
with poison enabled. Keep raw captures on the Sparks or on an external drive,
and copy only small logs, hashes and receipts to your own private records.
Don't commit raw results or machine values.

## World-four goldens

World four means four Sparks running one job together. The four-Spark
support on this page prepares for native qualification on the Sparks; it is
not that qualification. A host PASS does not qualify its transport, memory
use, arithmetic or performance. Read the
[four-Spark cabling guide](../../docs/four-sparks.md) first. Leave the
two-Spark goldens (the saved reference answers for two Sparks) intact; they
are not byte references for four Sparks.

1. Pick the four Sparks and write their ranks, 0 to 3 in cable order, in a
   private settings file. Record the source SHA, compiler, official
   checkpoint revision, hash and precision, logical topology, network rail
   mapping, complete runtime settings and effective sampling. Build every rank
   and require matching engine, bench and PTX hashes. `vade test build` still
   runs on each machine independently; it does not show that the ranks match
   each other.
2. Pick a new private folder for these goldens. Name it by world (4),
   logical topology, the ordered assignment of ranks to Sparks, the build and
   the settings. Use separate folders for `path` and `ring`, even when the
   same four cables form a physical ring.

   Note: `vade up --sparks 4` always launches a logical path, even on a
   physical ring, and leaves the closing cable unused.

   A change to the world, rank order, topology, math, checkpoint or any
   setting needs a new baseline and a new quality qualification. Never
   overwrite an older baseline.
3. Freeze the fixtures in the order 8192,2048,285 with at least 64
   continuation labels. You may reuse existing frozen prompt and continuation
   **input IDs**: the native matrix teacher-forces these inputs and captures
   fresh world-four full-row references. Old `full_logits` files and two-Spark
   output bytes are not world-four goldens. Keep the fixture hashes and the
   matching HTTP bodies. Every request must explicitly set temperature 0,
   top_p 1, top_k 0, repetition penalty 1, and presence and frequency
   penalties 0. Keep the response's echo of the effective sampling.
4. Start the native gate on ranks 0,1,2,3 at the same time, with one fresh
   shared `TP_RUN_ID`, the private control secret, RDMA enabled, profiling
   off, identical fixtures and the same memory budget. Each rank uses its
   own new dump directory and its own LISTEN/NEXT endpoints. The cabling guide
   lists a path's endpoints. Native `flash-bench` uses them directly, without
   adding the server's command/data port offset. For the default logical path:

   ```sh
   TP_TOPOLOGY=path vade test determinism --require-poison \
     --bench "$BENCH" --ptx "$PTX" --dump-dir "$DUMP" --groups 256,512 \
     -- "$MODEL" "$FIXTURES" "$LISTEN" "$NEXT" "$RANK" --world 4
   ```

   For a native ring, set `TP_TOPOLOGY=ring` and use each rank's closing-ring
   neighbor endpoints. All four native runs must exit 0 with poison PASS. Each MTP 0/3/15 log must
   seal its own rank at world 4. The full gate's poison check uses the
   sixteen-row case; the separate asymmetric eight-slot routing-injection
   matrix is still a world-two diagnostic.
5. Keep the new `world-4-rank-R.conf` sidecars and `reference-rank-R.sha256`
   manifests. Each sidecar records world, logical topology, rank, local
   endpoints, run ID, groups, selected runtime variables, bench and PTX hashes
   and the ordered fixture hashes. Join all four sidecars to the private
   settings file from step 1; a single rank can't describe the complete
   physical layout. Each rank must have 192 full-vocabulary reference
   rows. REF means captured, not qualified.
6. Repeat from fresh native processes, with a new shared run ID and new dump
   directories, keeping the same world, topology, rank order and settings.
   Require another complete PASS on all four ranks, and identical per-rank
   reference manifests between the two runs. For any row that differs, use the
   existing `flash-bench --gate-compare` entry point to report the exact byte
   location. Don't compare one rank's files with another rank's files, and
   don't normalize differences away.
7. Start the matching four-rank server with `vade up --sparks 4` (add
   `--fp8` if the native run used the FP8 checkpoint). Keep all four
   startup logs and run the HTTP phase from rank zero:

   ```sh
   TP_TOPOLOGY=path vade test determinism --server-only --world 4 --bench "$BENCH" --url "$URL" \
     --server-logs "$LOG0" "$LOG1" "$LOG2" "$LOG3" \
     --fixtures "$FIXTURES" --bodies "$BODIES"
   ```

   World-four HTTP checks need `/health` to report ready,
   `vade.numerics.world_size=4` and the selected `vade.numerics.topology`.
   Four stale log files can't turn a world-two endpoint into a world-four
   PASS.

   Because `vade up --sparks 4` always launches a logical path (step 2), set
   `TP_TOPOLOGY=path` for the HTTP phase, even when the cables form a ring.
   Native gates can use `ring`. An HTTP run on a ring is not qualified yet,
   because `vade up` has no way to start a four-Spark server on a ring.

   A server-only run reports poison as SKIP by design, so keep the earlier four
   native poison PASS receipts. For one combined receipt, give the native
   command the URL, all four logs, the bodies and a bounded server wait, before
   `--`, and start the server yourself once every rank has released its
   native model. `vade test` never starts a server. Exercise both server MTP
   settings, 0 and 3, keeping each startup's rank logs and comparing answers
   between runs.
8. Before you promote these captures to goldens, finish the quality checks:
   previous-build noise and floor against a baseline with the **same
   world-four ranks, topology and settings**, plus GSM8K, IFEval and
   32k/128k retrieval against the current `main` branch. Two- and four-Spark
   answers may differ. A ULP (unit in the last place) is the gap between two
   neighboring floating-point values. The ULP floor comparison only applies
   within one setup, so the logit difference between two and four Sparks is
   not such a comparison. Keep PASS, FAIL and SKIP results separate, then stop
   everything you started on the Sparks. Goldens stay private. Promotion is a
   recorded decision based on evidence, not a rename that turns an incomplete
   run into a PASS.

### Freezing a new world-four continuation

The MTP cases still need real accepted proposals. If the existing frozen input
labels produce zero accepted proposals on world four, the gate fails; don't
weaken that condition. To freeze a new world-four continuation instead:

- Copy the seed fixtures to new private files. Keep each `prompt_ids` array,
  and keep 64 rows without `full_logits`.
- Run the existing native `flash-bench` on all four ranks with MTP 0, one
  session, matching LISTEN/NEXT, `--world 4`, and `--path` for a path. Leave
  `FLASH_TEACHER_FORCING` and `FLASH_CACHE_REPLAY` unset.
- For each fixture, take exactly the 64 ordered IDs from its
  `flash_bench session=... fixture=... tokens=[...]` line and check that all
  four ranks print the same array. Then replace that fixture's rows with
  `{"token": ID}` objects in exactly that order. The run ends with a mismatch
  against the old labels. That is an expected bootstrap failure, never a
  qualifying PASS.
- Run the newly frozen fixtures again and require exit 0 and
  `tokens_match=yes` on every rank before you use them in steps 4–8. Stop on
  missing rows, rank arrays that diverge, transport errors or any mismatch
  during this verification run.

## Appendix: the retired shell gates

`vade test build` and `vade test determinism` replace two older shell scripts,
which are no longer in the repository. The tables below map each old step to
what the Rust command does now, for anyone comparing new receipts with old
ones. The old scripts printed messages in the caller's locale; the Rust
commands always use the C/POSIX locale, so compare old receipts with that in
mind.

### Build gate

| Old shell-gate step | What the Rust command does, and its receipt |
| --- | --- |
| `uname` platform selection | Accept Darwin or Linux aarch64/arm64; unsupported platform exits 2. |
| Require Cargo and a readable nonempty denylist | Validate before running the four build steps; input failure exits 2. |
| `mktemp` snapshot, step log and isolated export target; traps | Create private temporary storage and remove only that owned storage on completion. Source target output remains. |
| Mac: `cargo check --offline -p spark-build-kernels` | First step; print `PASS kernels-host (Mac; NVPTX unbuilt)` on success. |
| Spark: `cargo kernels` | First step; print `PASS kernels` on success. |
| `cargo build --release --workspace --exclude spark-rust-kernels --offline` plus feature arguments | Second step; print `PASS workspace` on success. |
| `cargo run -p vade-export --offline -- --source . --destination ... --denylist ...` | Run the same existing Rust exporter from the engine directory. |
| Snapshot: `cargo kernels` on Spark, then the same release workspace build with its isolated `CARGO_TARGET_DIR` | Complete the third step; print `PASS export` only when export and all snapshot builds succeed. |
| `cargo check --offline` plus feature arguments | Fourth step; print `PASS check` on success. |
| Capture each step's combined stdout/stderr | Successful build logs stay quiet; failure prints `FAIL <step> (exit <status>)` and its captured log. |
| Stop at the first failed child | Return that child's exit status; successful gate exits 0. |
| Interrupt/termination traps | Signal the gate's owned child group and clean its temporary storage; interrupt and termination retain exit 130/143. |

### Determinism gate

| Old shell-gate step | What the Rust command does, and its receipt |
| --- | --- |
| Executable and mode/input validation | Retain the original option and positional checks; neither configuration nor a host list supplies omitted inputs. |
| `--selftest`: `dd`, `cp`, one-byte overwrite | Standard-library file operations create 993,280-byte reference/copy files and set byte 47 in the copy to 1. |
| `--gate-compare ... 285 7 63` | Same native comparator; require exit 1, `logits-dump-compare 1 993280 1 FAIL`, and `prompt=285 row=7 position=63 vocabulary=11 byte=3`. Print the comparator output and selftest PASS; unexpected outcome exits 1. |
| `SECONDS`, `remaining`, GNU `timeout --foreground --kill-after=5s` | Rust elapsed-time and supervision retain the normal invocation's 900-second hang cap and five-second termination grace for the directly owned child, matching foreground timeout behavior. Selftest follows its original separate path. External GNU timeout is no longer needed. |
| Print `case rows bytes_compared bytes_different result` | Same five-column table header, case labels, counts, REF/PASS/FAIL/SKIP values and stderr diagnostics. |
| `--gate-capabilities` | Read the same exact compiled capability line; missing/unrecognized output fails. Report detection on stderr. |
| Select compiled poison capability | Per-row off: expect `0 0 0 SKIP`; per-row on: expect `720 715161600 0 PASS` and select the detected hook width. |
| Loop MTP depths `0 3 15` | Invoke the same benchmark with PTX, unchanged native positionals, `--mtp`, `--gate`, `--prefill-group` and `--gate-group`; capture combined output to `mtp-DEPTH.log`. After cap expiry, retain each remaining log with GNU timeout's C-locale empty-interval diagnostics, run its seal/incomplete checks and finish the native phase without launching more benchmarks. |
| Selected receipt/diagnostic lines | Forward every selected rank's `logits-rank-R-*` table rows; forward first differences, gate completion/MTP/mixed16/skip details and FAIL diagnostics to stderr. |
| Require `registrations sealed before weights rank=R world=W rdma=true` | Check the exact rank and world in each native MTP log and each supplied server log; a wrong-rank, wrong-world or missing seal fails even when bytes match. |
| Require `flash_gate complete` and child success | Incomplete native runs emit the same incomplete FAIL row and last 12 log lines; child failure fails the gate. |
| Check MTP-0 mixed and poison receipt rows | Require exactly one each with the exact expected labels, row counts, compared bytes and outcomes. |
| `find reference ...`, byte-sorted paths, `sha256sum` | Standard-library traversal preserves GNU `find -P` behavior without following directory symlinks; SHA-256 writes the same sorted `reference-rank-R.sha256` contents and requires 192 full rows. External `sha256sum` is no longer needed. |
| End native phase | Print native elapsed receipt and `logits_finished` handoff diagnostic after model release. |
| Rank-zero health polling | Keep `curl --noproxy '*' --fail --silent --max-time 2` on the same root `/health`; default no retry, otherwise one-second retries within the invocation cap. Curl remains required for `--url`; actual HTTP comparisons use the Rust benchmark client. |
| `--gate-server URL FIXTURES [--bodies ...]` | Same existing Rust HTTP gate, under the remaining invocation time; its output is forwarded. |
| EXIT summary | Keep elapsed/480-second target/900-second hang-cap fields, poison SKIP/PASS/FAIL and `result=<status>`. The 480-second target is reported, not enforced as failure. |
| Exit result | Success 0; checked option/input failure 2; native environment/setup failure, mismatch, incomplete work, timeout or required-poison failure 1. |
