<cloud> rented pair, 2 x GB10 VM (116 GB RAM each), driver 580.159.03. TIMING RUN, NOT A CARD. 2026-10-24 17:44-17:59Z.
Purpose: rerun the exact sequence that failed in the 1114 run (drafting ON, single-stream timing, then 8-stream drafting rounds) on the
fixed engine. Engine build 8c9411f86a2694ee9ea4fceceda1a8dabc1b3dfa on both VMs (sha256 of flash-serve 735751f9...; its served
engine is identical to main 407b6ae ("agree pipeline C8 policy and reclaim worker decode pools"). RUN_ID <cloud>-1792863842.
Model nvidia/Qwen3.8-Flash-Next-FP4-Local (checkpoint d99ccbed...), context 32768, greedy, thinking OFF, world 2 layer pipeline, RDMA on.
Scripts: timing-on.py (single stream, on/off/on) and timing-c8.py (8 threads x 256 tokens, on/off x 3), same as the 1114 run.

Single stream (tok/s = completion tokens per wall second):
  ON  code  57.0 median of n=10 (51.3 54.5 56.6 56.9 57.2 57.5 56.5 57.5 57.1 57.5; first run is the warm-up)   ON prose 29.8, 33.4
  OFF code  26.3 median of n=5  (26.0 26.3 26.4 26.3 26.3)                                                        OFF prose 25.4
  -> drafting ON is 2.2x on code, as on the 1114 run (56.5 vs 26.3 then).

8 streams x 256 tokens, aggregate tok/s (median per-stream tok/s in parentheses), errors:
  round 1: ON 81.3 (14.6) 0 errors   OFF 78.4 (19.2) 0 errors
  round 2: ON 92.0 (16.1) 0 errors   OFF 80.9 (20.2) 0 errors
  round 3: ON 92.2 (16.2) 0 errors   OFF 78.9 (18.8) 0 errors
  1114 run on the pre-fix build 4681d8b for comparison: round 1 ON 85.4 / OFF 70.4, rounds 2 and 3 ON FAILED 8/8 (generation_error,
  pooled storage grew past the 88.42 GiB ceiling of a 116 GB VM).

Result: the 8-stream drafting failure on 116 GB VMs is FIXED on the released engine: 3/3 ON rounds after plain rounds, 0 errors.
/health at the end reported device_bytes 94580681588 (pooled storage 94.6 GB, under the 95.4 GB rank budget). The drafting gain at
8 streams on this pair is modest (+4 % to +17 % over OFF); at home (128 GB Sparks, spark-a+spark-b) the same fix gave +23.7 % (133.871 -> 165.560).
Known reporting bug visible in these receipts: response metadata and /health say mtp_depth 0 while the rank log shows draft strict
depth 3 for the same requests (only the reporting is wrong; decode is unaffected).

Files: pair-fix-run.log (driver log), pair-timing-fix.json, pair-c8-fix.json, pair-fix-health.json, rank0-fix.log, rank1-fix.log.
Afterwards: servers stopped 17:59Z, control secrets wiped on both VMs, verified 18:01Z (0 flash-serve processes, no secret files).
