Complete RigMark 1.3.0 card — v0.1.0 RELEASE TEST, 1 x rented GB10 VM, drafting OFF (every request draft plain, mtp_depth 0), thinking OFF; Vade 5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e (public tag v0.1.0).
Box: rented GB10 VM (1 rented VM (container), one NVIDIA GB10, 116 GiB RAM, driver 580.173.02); NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716 (vade launch, --fp4-local); engine topology tag 'local'.
Server: vade launch defaults (fresh VADE_HOME); HTTP 127.0.0.1:18080; context 32768 / output 8192; drafting OFF (every request draft plain, mtp_depth 0); thinking OFF. Command: vade launch (INSTALL.md Route A, fresh VADE_HOME, defaults)
Test-start: not recorded in env.json
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
Compare (descriptive only, home Sparks, engine 73f7a9d): drafting OFF 2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off; drafting ON 2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK; rank logs 0/1 present.
started 19:31, ended 19:46 (CDT)
Box: vm-a = rented GB10 VM (container; engine admission uses MemAvailable 115.8 GiB). Code: public github.com/wappy-x/vade main = v0.1.0 5a22651 (git clone --branch main), INSTALL.md clone-and-build route (blank box: no rustup before).
Install timing (wall, INSTALL.md blocks pasted in order; model wait excluded): pre-launch blocks 123.710 s (rustup download+init, toolchains 1.99.0 + nightly-2026-10-01, clone, cargo +1.99.0 kernels, cargo build cli) + vade launch 'Launch ready in 160.746 s' (Install 85.539 s, Up 74.371 s) = 284.456 s. First paste line to Launch ready incl. model-copy wait + check = 488.1 s.
Model check: sha256sum -c cli/examples/qwen3.8-flash-next-fp4-local.sha256 = 33/33 OK in 63.172 s (checkpoint copied to this VM over the 200G link from the box that downloaded revision b9ce18a6887e18c089fdea8509a33e63429a3300).
vade verify: IDENTICAL: 20/20 runs, 1 machine, 128 tokens (168.9 s).
Public vade bench (no flags, drafting as served, receipt vade-bench.json sha256 9acf3a56765ada62f9016c6d120ec561df84077f4c7535bf8cb842d6e48e0ea4): median tok/s C1 31.021; C2 60.788; C4 101.537; C8 94.422; C16 123.136.
Drafting OFF card = every request "draft":"plain" against the default drafting-ON vade launch server; drafting ON card = requests omit draft (as served). Same server process for both cards (OFF first).
Card tools: pinned RigMark 52c29c3 card-tools copy (topology patch) + lane patch FLASHNEXT_DRAFT=plain|served (served: omit draft, accept mtp_depth 0..8); finalizer = finalize-cloud-a4.py world-1/drafting-aware lane derivative.
Exactness: token ids of all 66 card responses identical between the two rented VMs (OFF cards) and versus home 1-Spark card 2026-10-24_1444 (73f7a9d) plus the 2026-10-22 1-Spark and 2-Spark references: 66/66 each, exact = YES (compare-exactness.py).
