> Historical SGLang decision, not a Vade qualification. Retained with corrected provenance at `be85a6a`. Raw evidence and upstream implementations remain private and local. Reproducibility and quality equivalence are separate findings; read the [historical corrections](README.md) before comparing speed numbers.

# exact-top-k candidate promotion decision (1 Oct 2026)

Candidate: the previous production build plus lookup memory splitting, scatter gating,
DSA indexer query sharding and exact deterministic top-k (`SPARK_DSA_TOPK_EXACT=1`).
Production at decision time: `previous production build`. Rollback after promotion: `previous production build`, then `older fallback build`.
The recorded promotion standard: no promotion on "baseline equally bad"; every INCONCLUSIVE
resolved or justified; "no regression vs baseline" and "correctness demonstrated" are separate findings.
Labels: [M] measured (file / ledger row), [C] computed.

## 1. Gates

| # | Gate | Evidence | Result | No regression vs production | Correctness demonstrated |
|---|---|---|---|---|---|
| 1 | Top-k selection exact at 128k-512k | census gate on exact-top-k candidate: 107 stock-overflow rows, wrong 0, 162/162 dumps canonical; on the stock page path 25 rows wrong 0 (row 75) [M] | PASS | yes (stock: 24-25 wrong rows per census) | **yes**: selection equals exact top-k on every recorded and synthetic row; GPU check on 173 dumps + synthetic worst cases (row 75) |
| 2 | Run-to-run reproducibility (long context) | document A/A bit-identical at 4k / 32k / 128k on exact-top-k candidate (fid-x6) and previous production build-exact (fid-x5wtf), both arms recomputed (row 75) [M] | PASS | yes (stock diverged from position 2,049, mean 0.23-0.50 nats) | **yes** (bit-identical) |
| 3 | Decode reproducibility | decode A/A on exact-top-k candidate: 2 lookup-off replies of 6,144 tokens identical; lookup-on 2/2 identical [M] | PASS | yes | **yes** (bit-identical) |
| 4 | Decode vs prefill agreement | arbiter on exact-top-k candidate byte-identical to production at short context; same rate as bare stock SGLang without speculation (row 77) [M] | PASS (no regression) | yes | not demonstrable: no independent reference implementation fits; inherited stock numerics at near-bistable copy sites (row 77) |
| 5 | Park / resume state | park_state_check: parked 2,100 tokens mean swing 0.066, > 1 nat 12.9/1k vs solo 1,710 tokens 0.070, 12.3/1k; 3 parks in log; outputs identical across 3 reps [M] | PASS | yes | **yes** for "no state corruption" (parked = solo floor) |
| 6 | Mixed-width speculative state | run 1: 0 defects; pick coverage 2.85 < 3.0 judged drafts per request. Rerun with --n-pick 96 (mixed-width repeat): windows / cancel / mixed / switch: 0 defects; **pick**: still 2.85 per request (a per-request RATE fixed by the fixture, not raisable by more samples) with 274 judged drafts (L8 154, L16 120), pick distribution on vs off TV 0.036 (chi2 p 0.64) = off vs off2 0.038 (p 0.41) → justified as sufficient evidence of no lookup bias; **park/long FAIL**: parked A diverges from solo at token 102 (gap 3.46); A's own decode~TF: 7 tokens > 1 nat in 576 (12.2/1k, max 21.4: 'grows with the line grows with', 'three weeks of excessive signal voltage') ALL while decoding in a batch of 2 with B; solo refs on the same prompt 2 in 558 (3.6/1k, max 1.5). park_state_check parked = solo floor (12.9 vs 12.3/1k). Open question → batch-of-2 copy arbiter on exact-top-k candidate vs stock MTP vs bare stock [M] | no lookup-specific or batch-specific defect found (row 79): bare stock no-spec shows the same intrusions (batch-2 max 23.3 nats); park state = solo floor; quality equivalence not demonstrated (gate 10) | yes (same class as stock) | not demonstrable (inherited base-stack numerics) |
| 7 | Correctness suites | p10 suite: initial correctness check CLIENT CONNECT TIMEOUT (Mac → pair-b-rank0 TCP connect, Errno 60, server served 161 requests in that window with no error; ping 0% loss after) — resolved as a client-side failure by correctness repeat PASS and the normal-priority suite's initial correctness check PASS; long-context check PASS (longctx, 256k, 512k, longqa, longcancel, nearlimit); normal suite initial correctness check PASS (GSM8K / MMLU / vision / marker / loop) [M] | PASS (one resolved client failure) | yes | functional gates pass |
| 8 | Speed / responsiveness / agents | speed check PASS (single, prefill, cacheprefill, conc4, scale, interfere, interfere_prio, priority, saturated, saturated_p20); client check PASS (tools, compaction, mixed, mixed_prio, cancel, recovery); sched_edge PASS; resume_variance PASS [M] | PASS | yes | n/a |
| 9 | Shared-prefix cache | G1: branch 16k 24/24, concurrent 24/24, agent2 8k 16/16, deltas ≤ 0.0045 [M] | PASS | yes | **yes** (treefix row 66) |
| 10 | Copy quality, lookup on vs off (G0) | 24-prompt panel: rule PASS (first-error median 1/1, p 0.73); secondary edits 1,902 vs 961 (p 0.011 on this panel; pooled with 6-prompt pair panel p 0.06); no accepted lookup draft wrong; stock and bare stock show the same intrusions (row 79). No lookup-specific defect found; quality equivalence not demonstrated (the W16 excess is not significant, but its interval still permits a material increase) | PASS (rule) with a disclosed secondary signal; lookup stays on | yes vs stock class | not demonstrable; quality equivalence not demonstrated |
| 11 | Soak 50 min | agents + long-prefill exit 0; max temp 80 / 83 / 81 / 81 °C, throttle none, SM clock min 2,392 MHz; min MemAvailable pair-b-rank0 13.1 GB (earlier candidate 30 Sep 12.0, previous production build 6.7); swap-out 0; effective link errors 0 [M] | PASS | yes (memory headroom better than previous production build) | n/a |
| 12 | Logs | TRIPWIRE / guard / Traceback / illegal memory on the validation load: clean [M] | PASS (note: the 3 tracebacks on the recorder load are stock `_freeze_gc_after_server_warmup` 10 s warm-up read timeouts at startup, server ready the same second — benign) | | |

## 2. Speed vs production and vs earlier candidate [M]

| Metric | previous production build (row 58) | earlier candidate | exact-top-k candidate |
|---|---|---|---|
| greedy decode tok/s (writing+code, bs 1) | — | 42.25 (C 67.02 ms) | 42.27 (C 66.99 ms) |
| cold TTFT 8k / 32k | — | 3.30 / 10.93 s | 3.37 / 11.04 s |
| cold TTFT 128k / 256k / 510k | 50.4 / 103.1 / 210.8 s | 42.73 / 86.25 / 168.32 s | 42.84 / 86.54 / 168.29 s |
| sampled think / answer tok/s (T 1.0 max effort, 12 requests) | — | 39.31 / 56.93 (30 Sep historical validation) | 39.82 / 60.31 (texts differ: content noise) |

The exact-top-k cost is within single-sample noise: decode equal, prefill +0.1-2% at 8k-32k and ≤ 0.3% at 128k+.

## 3. Decision

**PROMOTED 1 Oct 15:10** (DECISIONS.md). Post-switch gate PASS. Measured regression: main-chat TTFT during agent prefill 1.22-1.23 s → 1.77-1.78 s (retained private request records). Rollback: previous production build.
