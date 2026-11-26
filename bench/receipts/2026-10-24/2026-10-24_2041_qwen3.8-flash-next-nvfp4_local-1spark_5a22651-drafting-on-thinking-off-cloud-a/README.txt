Complete RigMark 1.3.0 card — 1 x GB10, rented GB10 VM, DRAFTING ON, thinking OFF; Vade 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e (public v0.1.0 release).
Box: rented GB10 VM (1 NVIDIA GB10 VM (container), 116 GiB RAM, driver 580.178.04 reported by vade launch); NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ manifest d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716; engine topology tag 'local'.
Server: vade launch defaults from the release bundle; HTTP 127.0.0.1:18080; context 32768 / output 8192; drafting ON (requests omit draft = server default, engine-selected MTP); thinking OFF.
Test-start: not recorded
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK.
started 20:32, ended 20:41 (CDT)
Job: v0.1.0 release test E2-J2. The planned 2-Spark controller pair (Route B) could not run: the controller key is not accepted by the second VM and Vade never edits SSH settings, so this VM ran as a single (second fresh install of the day on this VM).
Install (public v0.1.0, INSTALL.md release-bundle route pasted as written, fresh box without rustup): clone 3.2 s + bundle download/check/extract 2.1 s + ./vade launch to 'Launch ready' 67.1 s (CLI: Launch ready in 66.629 s) = 72.4 s; model sha256 -c 125.2 s (33/33 OK) measured separately
vade verify: IDENTICAL: 20/20 runs, 1 machine, 128 tokens
Public vade bench (no flags, drafting as served; receipt vade-bench.json in this folder, status READY): 1 ok=1/1 30.435 ; 2 ok=2/2 58.672 ; 4 ok=4/4 108.956 ; 8 ok=8/8 98.887 ; 16 ok=16/16 125.657 (C, ok, median tok/s).
Token-id identity: 66/66 identical token ids vs this VM's drafting-OFF card (draft plain) of the same server start (request bodies equal except the draft field and per-pair prefill cache_salt), 66/66 vs this VM's earlier drafting-ON card today (first fresh install) and 66/66 vs the other rented VM's drafting-ON card of this job (compare-exactness.py); exact = YES.
Measurement client: pinned RigMark 52c29c3 card-tools copy run on the box against the public server; finalized on the Mac.
