Complete RigMark 1.3.0 card — 1 x GB10, rented GB10 VM, DRAFTING OFF, thinking OFF; Vade 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e (public v0.1.0 release).
Box: rented GB10 VM (1 NVIDIA GB10 VM (container), 116 GiB RAM, driver 580.173.02 reported by vade launch); NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ manifest d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716; engine topology tag 'local'.
Server: vade launch defaults from the release bundle; HTTP 127.0.0.1:18080; context 32768 / output 8192; drafting OFF (every request draft plain on the default drafting-ON server); thinking OFF.
Test-start: not recorded
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK.
started 20:17, ended 20:32 (CDT)
Job: v0.1.0 release test E2-J2. The planned 2-Spark controller pair (Route B) could not run: the controller key is not accepted by the second VM and Vade never edits SSH settings, so this VM ran as a single (second fresh install of the day on this VM).
Install (public v0.1.0, INSTALL.md release-bundle route pasted as written, fresh box without rustup): clone 4.4 s + bundle download/check/extract 0.8 s + ./vade launch to 'Launch ready' 69.5 s (CLI: Launch ready in 69.404 s) = 74.7 s; model sha256 -c 125.5 s (33/33 OK) measured separately
vade verify: IDENTICAL: 20/20 runs, 1 machine, 128 tokens
Public vade bench (no flags, drafting as served; receipt vade-bench.json in this folder, status READY): 1 ok=1/1 33.184 ; 2 ok=2/2 64.413 ; 4 ok=4/4 111.828 ; 8 ok=8/8 99.346 ; 16 ok=16/16 121.830 (C, ok, median tok/s).
Token-id identity: 66/66 identical token ids vs the 1-Spark card 2026-10-22_1237 (6032c30), the 2-Spark card 2026-10-22_1608 (bc85082), this VM's earlier drafting-OFF card today (first fresh install) and the other rented VM's drafting-OFF card of this job (compare-exactness.py); exact = YES.
Measurement client: pinned RigMark 52c29c3 card-tools copy run on the box against the public server; finalized on the Mac.
