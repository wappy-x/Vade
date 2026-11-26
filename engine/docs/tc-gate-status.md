# TCGATE final report

Updated 2026-11-15 22:28. Branch `tc-switch`, draft PR 256:
https://github.com/wappy-x/vade-next/pull/256
No format source has merged. No old baseline or golden changed; no new
baseline was written. This is not READY TRAIN or a complete landing gate.

The lead at 21:46:27 limited claims to measured evidence and deferred the
pair-model driver. At 22:04:23 the lead allowed one 12-minute turn after
SHORT: tile control first, a fresh repeat only if time remained, then stop.
The turn ran 22:17:13 to 22:27:01. Both claims released, both folders removed,
and the waiting line was left. No further gate work is running or queued.

## Measured table
Main: `1508efce96fd732edf93704c95c34b813afcebaf`.
The generated detailed table is `tc-gate-table.md` beside this report.

| Part | What it proves | Run tag | Result |
|---|---|---|---|
| Full fresh repeat | All four loads on two fresh servers | second full set not run | NOT PROVEN |
| Fresh subset | Three one-chat lists, 768 tokens; distinct starts, same hashes, cache off | a-gate-main-a1-1 / a-gate-main-b1-1 | PASS 3/3 |
| Groups | 34 alone/group lists, 4,352 tokens; groups 2/4/8 formed | a-gate-main-a1-1 | PASS |
| Drafting | 15 strict/plain lists, 3,840 tokens; depths 1/2/3/7 | a-gate-main-a2-1 | PASS |
| Continuation and cache | Six checks; kept turn reused 8,192 of 8,759 prompt tokens | a-gate-main-a3-3 | PASS 6/6 |
| Row tools | prompt-rows, nv-tc-groups, dense-tc-rows | a-gate-main-a3-3 | PASS 3/3 modes |
| Tile smoke control | Changed plain answers and short continuations | a-gate-control-tile-1 | CAUGHT |
| Small smoke control | Alone/group and drafted/plain fault detection | not run in assigned turn | NOT PROVEN |
| Quality | Same-history main and maker comparison | not run | NOT PROVEN |

Main's old-baseline verdicts: A1 56/56 lists SAME AS MAIN, A2 25/25,
B1 4/4 (832 tokens, includes warm-up). Gate repeat excludes warm-up and
compares three lists, 768 tokens. A3 has no old-baseline keys because it
uses new continuation requests; its six internal checks pass.

Dense row tools checked 266,240 + 15,600 twin words with zero differences;
alone/group batch counts were 3,446,016 and 220,545,024 words with zero
differences. These are existing main kernels, not future format entries.
At strict depth 1, structured/code each propose one token and accept none;
prose accepts eight. Coverage is checked by depth and by multirow passes.
Cache/cold alone is not decode/prompt proof. Continuations read answer rows
cold to compare them with rows made by decode.

## Tile control and partial repeat diagnostics
Control commit `771797e921d8585e5d8a121b26ea50690e5dbaef`, CONTROL=tile,
kernel image `2ddd67bae7d4`. All requests completed. All 12 common lists
(1,254 tokens) differ from main. Fixed plain requests first differ at code
token0 (7984/154842), prose token1 (3968/264), structured token1 (4718/220).
Short continuations differ at k1 token1 (15504/975), k17 token23 (320/1959),
k40 token11 (3772/4977). Long continuation, cache/cold and short-base repeat
still match: those checks alone miss this fault. The control checker
`--expect-fail=1,3 --only=1,3` exits 0 with both parts CAUGHT.
See `tc-gate-tile-control.txt` and `tc-gate-repeat-subset.txt`.
Earlier exit-6 queue waits stopped before any claim or build. They are not
failed device runs or caught controls. The first real tile run succeeded.

## Reusable command
From the night directory, after reading the latest pair order:

```sh
TCGATE_SEAT=TCBF bash kit/TCGATE-check.sh FULL_COMMIT two NEW_TAG
```

Use TCF8 or TCMX for that seat. It runs two fresh copies of the served cells
and row tools, then quality. Each part uses queue.sh and releases its claim
and waiting turn. Output: `runs/NEW_TAG-table.md`. Optional environment
values TCGATE_TILE_CONTROL and TCGATE_SMALL_CONTROL name smoke-control run
folders. Missing evidence prevents a complete pass. Allow about an hour
plus queue time (ESTIMATE). All three format inboxes received usage.
The runner is not authorized to exceed tonight's seat-specific turn limit.
Source copies are versioned at `tools/tc-gate/` on tc-switch. Seventeen
checker tests pass, including a stub-only seven-call runner test on Bash 3;
seven Python clients/checkers and the shell wrappers pass syntax checks.
No complete end-to-end format gate or quality run is claimed.

## Contract and integration
The common step now says a zero term brings no exponent, subnormal operands
retain the exponent field, and a result cut to zero is positive zero.
It names corrected BF16 twin commit 24772b376e on tc-bf-b. The earlier main
twin is not a reference for these cases. BF16's own section remains NOT DONE.
MXFP8 and Block FP8 sections were imported as documentation only. MX records
its corrected device words and failed one-row limit. FP8 records full kernel
word/group receipts and failed one-row limit on all six checked shapes.
Both retain missing served paths and references. No format is READY or landed.

## WHAT IS LEFT
- Complete the fresh four-load repeat and run the small smoke control. The
  assigned turn covered only tile plus the one-chat repeat. The main gate
  is therefore partially proven, not completely proven.
- The hostile reader's sum-order objection stays OPEN. Output scaling is not
  a subtle sum-order fault. A format needs a caught word-level old-order
  control plus device/twin sweeps across entries, row sizes, slots and
  neighbors. Full rules 2 and 6 are not claimed.
- The real-model pair driver for --decode-rows, --batch-steps and --verify-rows
  is not implemented; the lead explicitly deferred it tonight.
- BF16 must bring corrected twin/device words, direct/recorded entry checks,
  full path coverage and one-row timing within ruling D.
- Block FP8 must bring all-path wiring including the old host fallback,
  recorded-path coverage and one-row timing within ruling D.
- MXFP8 must first meet ruling D, then complete shape/scale admission and
  switch every served entry and reference together.
- Every format then needs served repeat, groups, drafting, prompt/decode,
  resume/cold, path counters and independent quality on its wired head.
- Quality clients and saved inputs exist, but no same-day main or candidate
  quality measurement ran. The saved maker score 198/208 is from the earlier
  comparison, not today's engine. The lead deferred work after this table.
- TCGATE-baseline.sh uses ids.py collect in a new numbered folder and refuses
  an existing folder. No rebase occurred, so it was not used on a new format.
- Source checks held by Mac load under the lead's draft-only exception.
  No speed claim is made by this gate seat.
