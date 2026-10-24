# Build and exactness checks with vade test

`vade test build` checks that the engine and a source snapshot without private
data build. `vade test determinism` checks that a fixed setup returns the
same answer bytes alone, batched, cached, chunked and with multi-token
prediction (MTP). Both commands are first-party Rust in the CLI, using the
standard library. They reuse the existing Rust exporter and `flash-bench`.

The former engine shell gates and operational wrappers have been retired in
favor of these commands. Host checks and fake executables verify orchestration;
GPU cases and pair receipts require a run on real Sparks. The full exactness
gate in [CONTRIBUTING.md](../../CONTRIBUTING.md#exactness-and-evidence) determines which checks qualify a change.

Build `vade` from the monorepo root with the pinned host compiler:

```sh
cargo build --release --offline --manifest-path cli/Cargo.toml
cli/target/release/vade test build --help
cli/target/release/vade test determinism --help
```

The commands run before lifecycle configuration and authentication handling;
they need no `VADE_CONFIG`, `VADE_HOME` or API key file. No machine, model,
endpoint or server lifecycle is inferred. Paths, private runtime environment,
all rank processes and server startup remain the caller's responsibility.

`vade test` uses C/POSIX diagnostic text and runs all its child commands with
`LC_ALL=C`, regardless of the caller's locale. Timeout receipts use the same
English text and ASCII quotes as GNU timeout in that locale. The former shell
gates inherited the caller's locale; historical receipt comparisons use C/POSIX
diagnostics.

## Build gate

From the monorepo root or `engine/`, run:

```sh
vade test build --dry-run
vade test build --denylist "$PRIVATE_DENYLIST"
```

`VADE_ROOT` can select the checkout when running elsewhere. An explicitly set
empty or invalid root fails before Cargo runs; an unset root keeps checkout
discovery from the current directory. Help and dry-run need no valid checkout.
The selected engine directory (the root itself or its `engine/` child) must
have its own `Cargo.toml` and `tools/export/Cargo.toml`; an exporter marker alone
does not qualify, even beneath a valid checkout.
The denylist must
be an external, readable, nonempty private file; its contents are never
printed. Follow [source export](../../engine/docs/export.md) and use a clean,
committed checkout. All Cargo commands run offline. The gate builds in the
existing engine target directory, then builds an exported snapshot with its
own temporary target directory. It does not install a toolchain, run inference
or publish that snapshot.

On a Mac, the first step checks the host kernel builder without generating
NVPTX. On Linux aarch64, it generates kernels with the separately pinned
nightly, then also generates kernels inside the snapshot. A Mac PASS is a
host build result, not a GPU result. For a native qualification, the caller
still compares the server binary and PTX hashes across every selected rank.

| Command option or input | Rust equivalent and preserved behavior |
| --- | --- |
| Build entry point | `vade test build` |
| `--help`, `-h` | Print help, exit 0, perform no steps. |
| `--denylist FILE` | Select the same external private denylist; overrides `PRIVATE_DENYLIST`. |
| `PRIVATE_DENYLIST` | Default denylist path when `--denylist` is absent. |
| `--features FEATURES` | Forward each occurrence unchanged to the source release build, snapshot release build and final `cargo check`. |
| `--no-default-features` | Forward unchanged to those same three commands. It disables the gate's per-row capability, not engine/server per-row rejection. |
| `--dry-run` | Print the same command preview without Cargo execution, temporary files or PASS lines. |
| Unknown option or missing option value | Print input failure and exit 2. |
| Script-relative engine working directory | Resolve `VADE_ROOT` or the current checkout root/engine directory, then run Cargo from `engine/`. |
| `CARGO_NET_OFFLINE=true` | Set for gate child commands in addition to their existing `--offline` arguments. |
| `TMPDIR` | Select temporary storage using the host's existing temporary-directory setting. |
| Pinned host and kernel toolchains, optional `KERNEL_BIN` | Keep the existing Cargo/toolchain behavior; install nothing. |

| Established step | Rust equivalent and receipt |
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

## Determinism gate

Invoke the native form concurrently on both ranks, changing only `RANK` and
the new rank-local dump directory. Keep fixtures and retained HTTP bodies in
matching order **8192,2048,285**. Each fixture needs at least 64 frozen
continuation labels. The newest short prompt then remains available for the
mixed HTTP cohort. Set the same fresh `TP_RUN_ID` on both ranks, and use the
assigned launcher environment: private `SEQPAR_CONTROL_SECRET`,
`SEQPAR_RDMA=1`, `TP_TOPOLOGY=ring`, `SPARK_PROFILE=0` and memory budget.

```sh
vade test determinism --require-poison \
  --bench "$BENCH" --ptx "$PTX" --dump-dir "$DUMP" --groups 256,512 \
  -- "$MODEL" "$FIXTURES" "$ADDR0" "$ADDR1" "$RANK" --world 2
```

The logit runs release their GPU model before rank zero reaches an optional
HTTP phase. The caller starts the usual server and retains both startup
logs; the gate never starts or stops that server. To combine the phases, add
`--url`, `--server-logs`, optional `--bodies` and `--wait-server` before `--`.
Alternatively run the retained separate server form after native completion:

```sh
vade test determinism --server-only --bench "$BENCH" --url "$URL" \
  --server-logs "$LOG0" "$LOG1" --fixtures "$FIXTURES" --bodies "$BODIES"
```

For four ranks, use [the world-four procedure below](#world-four-goldens).
The default and original two-rank native argument order remain unchanged.

`--require-poison` requires a completed native poison PASS in this invocation;
a server-only run has no native poison result and therefore fails with that
option. Use the combined native/HTTP form for a merge receipt that needs both.
Without `--bodies`, the server receives the fixture's exact prompt token IDs
through `/v1/completions`. With bodies, it receives the retained chat requests
through `/v1/chat/completions`. Use a root HTTP URL through a local tunnel for
TLS; the gate adds no endpoint. Sampling and accounting checks remain in the
existing `flash-bench` HTTP client.

| Command option or input | Rust equivalent and preserved behavior |
| --- | --- |
| Exactness entry point | `vade test determinism` |
| `--help`, `-h` | Print help, exit 0 and perform no work. |
| `--bench BIN` | Require the explicitly supplied executable; reuse its capabilities, native, server and comparator entry points. |
| `--ptx PTX` | Require an existing PTX file for native mode. |
| `--dump-dir DIR` | Require a new rank-local directory; retain dumps, MTP logs and reference manifest there. |
| `--groups N,N` | Same two prefill groups, default `256,512`; the native gate requires distinct multiples of 64 up to 512. |
| `--require-poison` | Turn poison SKIP into FAIL and exit 1, including in server-only mode. |
| `--url URL` | Rank zero checks `/health`, then runs the existing HTTP gate. Require every rank's server log. |
| `--server-logs LOG0 LOG1 [LOG2 LOG3]` | Require one locally readable startup log per selected rank, in rank order, after health succeeds; each must have its own rank/world RDMA seal. |
| `--bodies REQUEST.json,...` | Forward matching retained chat bodies in fixture order; otherwise use fixture prompt token IDs. |
| `--wait-server SECONDS` | Same nonnegative integer health wait, default 0; two-second health attempts and one-second retry interval. |
| `--server-only` | Skip native work; require `--fixtures`, `--url`, no native positionals and every server log. World defaults to 2; select `--world 4` explicitly for four Sparks. |
| `--fixtures FIXTURE.json,...` | Select fixtures for server-only mode. Native mode uses the positional fixture list. |
| `--selftest` | No GPU: corrupt one byte of a full vocabulary dump and require the existing comparator's expected FAIL and detail. Incompatible with `--require-poison`. |
| `-- MODEL FIXTURES ADDR0 ADDR1 RANK [--world 2\|4]` | World defaults to 2: shared pair endpoints and rank 0/1. World 4: this rank's LISTEN/NEXT endpoints and rank 0..3. A leading `--world` is also accepted; conflicting leading/trailing sizes fail. |
| `TP_RUN_ID` | Require a fresh common value for native work; append `-mtp-0`, `-mtp-3`, `-mtp-15` for the three runs. Rust rejects a missing/empty value with exit 1. The Mac's Bash 3.2 fatal-expansion/EXIT-trap path can incorrectly exit 0; compare against the Bash fault result on a Spark. |
| `TMPDIR` | Select temporary storage for the selftest and captured helper output using the host's existing setting. |
| Private transport/model runtime environment | Inherit unchanged; caller chooses assigned hardware and memory limits. For world 4, `TP_TOPOLOGY=path` also adds native `--path`; `ring` or unset uses the native ring. Other topology values fail. |
| `VADE_TEST_ROUTING_REJECT` | Gate overrides it: compiled poison hook for MTP 0 and `0` for MTP 3/15. Detection chooses legacy `1:3` or width-aware `1:3:16`. |
| Missing option value, unknown option or invalid checked input | Print input failure and exit 2; native environment/setup failures retain exit 1. |

| Established step | Rust equivalent and receipt |
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

The caller's `flash-bench` still defines every native and server case. The
CLI changes how the checks are invoked and supervised, not their arithmetic
or byte comparator. Full vocabulary rows use little-endian FP32 and compare
every byte, including signed zero. Frozen continuation labels determine the
same teacher-forced positions as before.

| Preserved case | Existing Rust operation and evidence |
| --- | --- |
| Cold batch one, MTP 0 | Capture 64 positions per fixture as `reference`: 192 rows per rank, REF with zero compared bytes. |
| Homogeneous widths 2/8/16 | Restore independent sessions; capture the lead's prompt end and compare every row's 63 actual decode outputs. Copied prompt-end values are never PASS comparisons. |
| Groups 256/512 | Compare both prefill group choices against the same one-row reference; attention retains its 64-token tile. |
| Cold/cache replay | Compare snapshot/restore replay with the cold reference. |
| Follow-up | After 63 decode inputs, canonicalize and exactly restore; compare one suffix input with cold prefill of prompt plus all 64 labels. |
| MTP 3 cold and replay | Use real proposals, accepted-prefix commit and rollback; report proposals/acceptance and fail when none is accepted. |
| MTP 15 rows 4/8/16 | Restore fresh sessions and compare all 63 teacher-forced positions from verification launches at depths 3/7/15 with the one-row reference. |
| Heterogeneous native widths 3..16 | Compare each lane with its solo position: 577 rows, 573,122,560 bytes per rank. |
| Shifted native mixed16 | Cycle prompt lengths 8192/2048/285; advance lane `r` by `r` labels, then make 48 full-width passes: 768 rows, 762,839,040 bytes per rank. |
| Mixed16 poison | Reject row 3 inside the first 16-row pass, retain that slot as padding and compare survivors with no-hook mixed16: 720 rows, 715,161,600 bytes per rank. First call compares 15 rows/14,899,200 bytes; remaining 47 compare 705 rows/700,262,400 bytes. |
| Mixed16 bounds | Same shared 120-second work cap checked between calls, one weight copy and previous peak-memory bound; the invocation cap also bounds hangs. |
| HTTP cold/prime/hit | Per fixture: cache off cold reference; cache on prime and hit comparisons. Cold must reuse zero tokens, hit must reuse tokens. |
| HTTP homogeneous cohorts 2/8/16 | Same cached request bodies concurrently, `temperature=0`, `seed=0`, plain drafting and 64-token limit; each reply must report reuse. |
| HTTP mixed cohort | Three prompt lengths together, in fixture order, compared with their cold responses; each must report cache reuse. |
| HTTP comparisons | Compare token IDs, finish reason and prompt/completion/total usage; missing replies, IDs or usage fail. Natural EOS contributes usage without a streamed ID. |

Native dump names remain
`DIR/CASE/prompt-NNNNN/session-NNNNN/position-NNNNN-rank-R.f32`.
The extra mixed16 captures require 762,839,040 bytes per rank, or
1,478,000,640 with poison enabled. Keep raw captures on assigned hardware or
the assigned external evidence volume, and copy only small logs, hashes and
receipts to the private workshop. Do not commit raw results or machine values.

## World-four goldens

Four-Spark support here is preparation for native qualification. A host PASS
does not qualify its transport, memory usage, arithmetic or performance.
Use the [four-Spark cabling guide](../../docs/four-sparks.md) first. Keep the
two-Spark goldens intact; they are not byte references for four Sparks.

1. Reserve the four assigned machines and record their ordered ranks 0–3 in a
   private settings manifest. Record the source SHA, compiler, official
   checkpoint revision/hash/precision, logical topology, network rail mapping,
   complete runtime settings and effective sampling. Build every rank and
   require matching engine, bench and PTX hashes. `vade test build` still runs
   independently on each machine; it does not establish cross-rank identity.
2. Choose a new private golden root keyed by **world 4, logical topology,
   ordered physical rank assignment, build and settings**. Use distinct paths
   for `path` and `ring`, even on the same four-cable physical ring. The default
   `vade up --sparks 4` selects a logical path and leaves the closing cable unused.
   A changed world, rank order, topology, math, checkpoint or setting requires
   a new baseline and quality qualification; never overwrite an older baseline.
3. Freeze fixtures in order 8192,2048,285 with at least 64 continuation labels.
   Existing frozen prompt and continuation **input IDs** may be reused: the
   native matrix teacher-forces these inputs and captures fresh world-four
   full-row references. Old `full_logits` files and two-Spark output bytes are
   not world-four goldens. Retain fixture hashes and the matching HTTP bodies.
   Every request must explicitly use temperature 0, top_p 1, top_k 0,
   repetition penalty 1, and presence/frequency penalties 0. Retain the
   response's effective sampling echo.
4. Launch the native gate concurrently on ranks 0,1,2,3 with one fresh shared
   `TP_RUN_ID`, private control secret, RDMA enabled, profiling off, identical
   fixtures and the assigned memory budget. Each rank uses its own new dump
   directory and LISTEN/NEXT endpoints. A path's endpoints are listed in the
   cabling guide; native `flash-bench` consumes these directly, without adding
   the server's command/data port offset. For the default logical path:

   ```sh
   TP_TOPOLOGY=path vade test determinism --require-poison \
     --bench "$BENCH" --ptx "$PTX" --dump-dir "$DUMP" --groups 256,512 \
     -- "$MODEL" "$FIXTURES" "$LISTEN" "$NEXT" "$RANK" --world 4
   ```

   For an explicitly assigned native ring, set `TP_TOPOLOGY=ring` and use each
   rank's closing-ring neighbor endpoints. All four native invocations must
   exit 0 with poison PASS. Each MTP 0/3/15 log must seal its own rank at world 4.
   Full-gate poison uses the sixteen-row case; the separate asymmetric
   eight-slot routing-injection matrix remains a world-two diagnostic.
5. Preserve the new `world-4-rank-R.conf` sidecars and
   `reference-rank-R.sha256` manifests. Each sidecar records world, logical
   topology, rank, local endpoints, run ID, groups, selected runtime variables,
   bench/PTX hashes and ordered fixture hashes. Join all four sidecars to the
   private settings manifest from step 1; a single rank cannot describe the
   complete physical assignment. Each rank must have 192 full-vocabulary
   reference rows. REF means captured, not qualified.
6. Repeat from fresh native processes with a new shared run ID and new dump
   directories, preserving the same world/topology/rank order/settings. Require
   another complete PASS on all four ranks and identical per-rank reference
   manifests between the two runs. Use the existing `flash-bench --gate-compare`
   entry point for any differing row to report the exact byte location. Do not
   compare one rank's files with another rank's files or normalize differences.
7. Start the matching four-rank server through the owned lifecycle. Retain all
   four startup logs and run the HTTP phase from rank zero:

   ```sh
   TP_TOPOLOGY=path vade test determinism --server-only --world 4 --bench "$BENCH" --url "$URL" \
     --server-logs "$LOG0" "$LOG1" "$LOG2" "$LOG3" \
     --fixtures "$FIXTURES" --bodies "$BODIES"
   ```

   World-four HTTP checks require `/health` to report ready,
   `vade.numerics.world_size=4` and the selected `vade.numerics.topology`. Supply
   `TP_TOPOLOGY=path` for the default product launcher; native rings use `ring`.
   Four stale log files cannot turn a world-two endpoint into a world-four PASS.
   `vade up --sparks 4` always launches a logical path. Logical-ring native
   gates are supported; ring HTTP remains separate and unqualified until the
   owner assigns an owned Rust lifecycle route. A physical ring running the
   product's logical path uses the command above.
   Server-only poison is SKIP by design; retain the prior four native poison
   PASS receipts. For one combined receipt, supply URL, all four logs, bodies
   and a bounded server wait to the native command, before `--`, and arrange
   the caller-owned handoff after every rank releases its native model. The
   wrapper never starts a server. Exercise both server MTP settings 0 and 3,
   keeping each startup's rank logs and comparing answers between runs.
8. Before promoting these captures to goldens, complete the owner-required
   quality checks: previous-build noise/floor against a baseline with the
   **same world-four ranks, topology and settings**, plus GSM8K, IFEval and
   32k/128k retrieval against current main. Two- and four-Spark answers may
   differ; their logit difference is not a same-setup ULP floor comparison.
   Retain PASS/FAIL/SKIP separately;
   return the hardware empty and release its reservations. Goldens remain
   private. Promotion is a recorded evidence decision, not a rename that
   turns an incomplete run into PASS.

The MTP cases still require real accepted proposals. If existing frozen input
labels yield zero accepted proposals on world four, the gate fails; do not
weaken that condition. To freeze a new world-four continuation, copy the seed
fixtures to new private files, preserve each `prompt_ids` array, and keep 64
rows without `full_logits`. Run the existing native `flash-bench` on all four
ranks with MTP 0, one session, matching LISTEN/NEXT, `--world 4` and `--path`
for a path. Leave `FLASH_TEACHER_FORCING` and `FLASH_CACHE_REPLAY` unset. For
each fixture, take exactly the 64 ordered IDs from its
`flash_bench session=... fixture=... tokens=[...]` line; verify all four ranks
emit the same array, then replace that fixture's rows with
`{"token": ID}` objects in that exact order. Its terminal old-label mismatch
is an expected bootstrap failure, never a qualifying PASS. Run those newly
frozen fixtures again and require exit 0 and `tokens_match=yes` on every rank
before using them in steps 4–8. Stop on missing rows, divergent rank arrays,
transport errors or any mismatch during this verification run.

## Runtime and qualification ownership

`vade test build` takes the selected checkout and explicit build arguments.
`vade test determinism` takes explicit fixtures, binaries, runtime environment
and optional HTTP inputs. The caller exports any required private values;
these commands do not read shell configuration or infer machine access.
Use `vade up` for the owned product server lifecycle and `vade down` for cleanup.
Keep deployment-specific plans and raw results outside Git.
