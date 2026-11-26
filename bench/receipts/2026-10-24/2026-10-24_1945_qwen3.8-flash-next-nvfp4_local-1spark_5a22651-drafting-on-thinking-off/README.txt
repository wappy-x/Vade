Complete RigMark 1.3.0 card — v0.1.0 RELEASE TEST (informative, no gate), 1 x GB10, DRAFTING ON, thinking OFF; Vade 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e.
Box: home-3 (DGX Spark) (1 x NVIDIA DGX Spark GB10, driver 580.178.04); NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716; engine topology tag 'local'.
Server: public v0.1.0 `vade launch` defaults, HTTP 127.0.0.1:18080; context 32768 / output 8192; drafting ON as served (requests omit draft; server default drafting); thinking OFF.
Test-start: launch ready 19:21:55 CT; card started 19:38:48 CT
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Compare (descriptive only): 1-Spark 2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off (engine 73f7a9d, same recipe).
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK.
Install: release-bundle route (git clone main + v0.1.0 SHA256SUMS + bundle, sha256sum -c, tar --zstd -xf, ./vade launch); fresh VADE_HOME; wall 102.5 s from the first paste line to "Launch ready", model checksum excluded; model checksum (sha256sum -c committed manifest, 33 files OK) 73.1 s.
vade verify: IDENTICAL: 20/20 runs, 1 machine, 128 tokens.
Public vade bench (zero flags, drafting server-default): C1 45.277 tok/s TTFT p50/p95 184.319/184.319 ms ; C2 78.940 tok/s TTFT p50/p95 281.913/282.146 ms ; C4 141.687 tok/s TTFT p50/p95 406.321/406.847 ms ; C8 125.311 tok/s TTFT p50/p95 578.245/7212.052 ms ; C16 140.180 tok/s TTFT p50/p95 10825.040/21841.371 ms; receipt vade-bench.json in this folder.
Server = public v0.1.0 vade launch defaults: output limit 8192 (min(context, 8192)); the 73f7a9d reference cards ran output 16384. Decode cap 8192 unchanged.
Measurement client: our full-card-api.py + CARD_DRAFT switch (plain = draft plain per request, served = draft omitted) with the rigmark copy of the 2026-10-24 home-1 ON card (world 1 local, accepts mtp_depth 0/2/3); finalizer = finalize-cloud-a4.py adapted for world 1 and the drafting variant.
Engine-selected drafting per response: decode 15 strict mtp3; C1 3 strict mtp3; C2 6 strict mtp3; C4 12 strict mtp3; C8 11 plain mtp0, 13 strict mtp3; prefill rows carry the pinned draft plain.
started 19:38, ended 19:45 (CDT)
Token identity: 66/66 identical token_ids vs 2026-10-24_1438 1-Spark 73f7a9d drafting-ON card (compare-exactness.py); 66/66 identical vs the home-3 v0.1.0 drafting-OFF card 2026-10-24_1938 (same requests minus draft); exact = YES.
