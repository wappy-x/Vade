Complete RigMark 1.3.0 card — v0.1.0 RELEASE TEST (informative, no gate), 1 x GB10, DRAFTING OFF, thinking OFF; Vade 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e.
Box: home-4 (DGX Spark) (1 x NVIDIA DGX Spark GB10, driver 580.178.04); NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716; engine topology tag 'local'.
Server: public v0.1.0 `vade launch` defaults, HTTP 127.0.0.1:18080; context 32768 / output 8192; drafting OFF (every request draft plain against the default drafting-ON server); thinking OFF.
Test-start: launch ready 19:24:35 CT; card started 19:29:17 CT
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Compare (descriptive only): 1-Spark 2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off (engine 73f7a9d, same recipe).
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK.
Install: clone-and-build route (rustup toolchains already present, cargo +1.99.0 kernels, cargo build cli, vade launch); fresh VADE_HOME; wall 249.3 s from the first paste line to "Launch ready", model checksum excluded; model checksum (sha256sum -c committed manifest, 33 files OK) 86.1 s.
vade verify: IDENTICAL: 20/20 runs, 1 machine, 128 tokens.
Public vade bench (zero flags, drafting server-default): C1 45.205 tok/s TTFT p50/p95 185.405/185.405 ms ; C2 79.257 tok/s TTFT p50/p95 281.321/281.322 ms ; C4 141.925 tok/s TTFT p50/p95 452.447/452.874 ms ; C8 126.067 tok/s TTFT p50/p95 579.834/7162.412 ms ; C16 141.107 tok/s TTFT p50/p95 10769.805/21690.906 ms; receipt vade-bench.json in this folder.
Server = public v0.1.0 vade launch defaults: output limit 8192 (min(context, 8192)); the 73f7a9d reference cards ran output 16384. Decode cap 8192 unchanged.
Measurement client: our full-card-api.py + CARD_DRAFT switch (plain = draft plain per request, served = draft omitted) with the rigmark copy of the 2026-10-24 home-1 ON card (world 1 local, accepts mtp_depth 0/2/3); finalizer = finalize-cloud-a4.py adapted for world 1 and the drafting variant.
started 19:29, ended 19:41 (CDT)
Token identity (compare-exactness.py, token_ids): 66/66 identical vs 2026-10-24_1444 1-Spark 73f7a9d OFF card, 66/66 vs 2026-10-22_1237 1-Spark, 66/66 vs 2026-10-22_1608 2-Spark, 66/66 vs the home-3 v0.1.0 OFF card (2026-10-24_1938); exact = YES.
