Complete RigMark 1.3.0 card, 1 Spark, DRAFTING ON AS SERVED, THINKING OFF; Vade v0.1.0 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e from public GitHub.
Box: home-1, 1 x NVIDIA DGX Spark GB10, driver 580.178.04. Install route: INSTALL.md release-bundle route (clone, v0.1.0 SHA256SUMS + bundle, sha256sum -c, tar --zstd -xf, ./vade launch). Checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716 (HF revision b9ce18a6887e18c089fdea8509a33e63429a3300); engine topology tag 'local'.
Server: public `vade launch` defaults (context 32768 / output 8192); HTTP 127.0.0.1:18080; drafting ON as served (requests omit draft; engine default); thinking OFF. One server process for this card.
Test-start: vade launch 'Launch ready', chat hello answered, /exit; vade status ready; then vade verify, vade bench, OFF card, ON card on the same server
Pinned RigMark 52c29c3 corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs; C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK.
Install: 107.2 s wall from the first paste line (git clone) to 'Launch ready' (clone + bundle download + sha256sum -c + extract 5.3 s; ./vade launch 101.9 s = local bundle check 0.5 s + model load 100.6 s). Model manifest check timed separately: 80.2 s, 33/33 OK. No documented command needed an edit.
vade verify: IDENTICAL: 20/20 runs, 1 machine, 128 tokens (draft modes plain/strict; reference ids sha256 1231c2da8c7193dbb56029a729f3ec78674d8cc273c3ee2a6462d79c9d22c84a).
Public `vade bench` (zero flags, drafting as served, receipt ./vade-bench.json sha256 0582e9bbc00a4fcf838c16e66212cb200a7e1be3ebf9101dc8914c5f579ed3bd): C1 45.466 tok/s (1/1, TTFT p50 191.272 ms) | C2 79.747 tok/s (2/2, TTFT p50 271.200 ms) | C4 143.598 tok/s (4/4, TTFT p50 396.424 ms) | C8 185.144 tok/s (8/8, TTFT p50 705.843 ms) | C16 189.042 tok/s (16/16, TTFT p50 6025.812 ms).
Token-id identity: drafting ON 66/66 responses identical (token_ids) to this server's drafting-OFF card (home-1, 19:37) and to home-2's build-route OFF card; OFF matched the 10-22 1-Spark/2-Spark refs 66/66; exact = YES. Served drafting: decode 15/15 strict depth 3; C1/C2/C4 strict 3; C8 14/24 strict 3 + 10/24 plain (engine-selected).
Measurement client: pinned RigMark 52c29c3 one-Spark full-card API copied to the box (stdlib only), CARD_DRAFT=served: requests omit draft, drafting as the server selects (health: draft strict, mtp_depth 3).
Code: git clone --branch main https://github.com/wappy-x/vade.git at 5a22651 (v0.1.0) and, bundle route, the v0.1.0 release assets; model = pinned HF revision already on the box, manifest-checked before use.
started 19:37, ended 19:43 (CDT)
