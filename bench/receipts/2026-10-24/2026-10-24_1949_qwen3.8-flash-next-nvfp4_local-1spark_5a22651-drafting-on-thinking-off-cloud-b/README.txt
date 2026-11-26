Complete RigMark 1.3.0 card — v0.1.0 release test, rented GB10 VM, informative, no gate; DRAFTING ON; Vade 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e: world 1, public v0.1.0 release bundle installed by following INSTALL.md only NEW FOLDER.
1 x GB10: rented GB10 VM (1 VM, one NVIDIA GB10, 121.7 GiB RAM, driver 595.99.02, Ubuntu 24.04.4 aarch64 Docker container); no fabric (single machine); NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716 served with --fp4-local; engine topology tag 'local'.
Server: flash-serve world 1 started by ./vade launch (bundle defaults); HTTP 127.0.0.1:18080; context 32768; drafting ON as served (requests omit draft; server default resident MTP + lookup); thinking OFF. Command: ./vade launch from the public v0.1.0 release bundle vade-launch-bundle-v1-5a22651-aarch64.tar.zst (flash-serve 0 1 ... 32768 127.0.0.1:18080 0 --fp4-local; drafting default ON, mtp engine_default = depth 3)
Test-start: vade status ready, vade verify IDENTICAL: 20/20 runs, 1 machine, 128 tokens; then public vade bench; then this card
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK; rank logs 0/1 present.
rented GB10 VM (v0.1.0 release test 2026-10-24 evening CT); code only from public GitHub wappy-x/vade main + v0.1.0 release assets; model pinned b9ce18a6887e18c089fdea8509a33e63429a3300, manifest check 33 files OK.
Install timing (bundle route, INSTALL.md blocks verbatim as root): prereq apt 4.7 s, clone 4.4 s, bundle download+sha256+extract 1.1 s, model checksum 120.0 s, ./vade launch to Launch ready 119.1 s (load 117.7 s); first paste line to Launch ready 249.2 s wall (129.3 s excluding the model checksum).
Public vade bench receipt (./vade-bench.json, zero flags, drafting as served, status READY, build 5a22651, 256 tokens x1 repeat): C1 34.348 tok/s TTFT p50 424.4 ms; C2 72.732 tok/s TTFT p50 285.4 ms; C4 77.125 tok/s TTFT p50 1665.7 ms; C8 17.738 tok/s TTFT p50 74638.5 ms; C16 134.497 tok/s TTFT p50 9584.6 ms.
Drafting ON card: requests omit the draft field = drafting as served (server default).
Server output cap is the vade launch default 8192 (health caps 32768/8192); the card decode cap 8192 fits it. Thinking OFF.
Driver 595.99.02 is newer than the tested 580 series.
Exactness: token ids 66/66 identical vs the 2026-10-24 1438 1-Spark drafting-ON card (73f7a9d, compare-exactness.py) and 66/66 identical vs this box's drafting-OFF card (same requests apart from the draft field).
Prefill rows use the pinned RigMark prefill body (draft plain), as on the 1438 card; 9 of 24 C8 streams were decoded plain by the server's own choice (1438 card: 12 of 24); all decode rows drafted (strict, MTP depth 3).
started 19:43, ended 19:49 (CDT)
