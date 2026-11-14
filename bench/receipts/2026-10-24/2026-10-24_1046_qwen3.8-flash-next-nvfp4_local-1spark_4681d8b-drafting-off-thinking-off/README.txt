Complete RigMark 1.3.0 card — FIRST 1-SPARK RUN (informative, no gate), MTP OFF, DRAFTING OFF; Vade 4681d8b2c83f3df09926046dd6bd8b7fd124ce69: world 1 layer pipeline (--mtp 0, greedy).
1 x GB10: rented <cloud> 2-node GB10 cluster (<cloud-cluster>, entry <cloud-id>); this card used ONE node (w1) alone (1 VM, one NVIDIA GB10, 116 GB RAM, 18 vCPU, driver 580.159.03); none used (world 1, loopback control port on the VM); ranks: <cloud-vm>-w1 rank0 (HTTP 127.0.0.1:8000) 127.0.0.1; NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716 served with --fp4-local; engine topology tag 'local'.
Server: flash-serve world 1 native rank path (--tensor-map, --mtp 0); rank0 binds HTTP 127.0.0.1:8000; context 32768 / output 16384; --mtp 0 and requests draft plain = drafting OFF; thinking OFF. Command: flash-serve 0 1 127.0.0.1:29970 127.0.0.1:29970 spark_layer_kernels.ptx <model> 32768 127.0.0.1:8000 0 --fp4-local --mtp 0 (SEQPAR_MEMORY_BYTES=107374182400 explicit budget for the VM, FLASH_SERVE_OUTPUT_TOKENS=16384)
Test-start: health ready on w1 at 15:30Z, card started 15:30:21Z
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
First 1-Spark run on this engine: informative, no gate; no 1-Spark baseline to compare against.
Compare (different Spark counts, descriptive only): 2-Spark 2026-10-22_1608_qwen3.8-flash-next-nvfp4_local-2spark_bc85082-drafting-off-thinking-off; 1-Spark main 2026-10-22_1237_qwen3.8-flash-next-nvfp4_local-1spark_6032c30-thinking-off.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK; rank logs 1/1 present.
Same checkpoint files as the 2-Spark cloud card 2026-10-24_1006 (copied from the entry node over the 200G link, sha256 verified).
Container memory: an explicit SEQPAR_MEMORY_BYTES=100 GiB budget plus a 100 GiB squeeze right before the start (the engine's derived ceiling came out 0.07-0.20 GiB short because the host caches the weight read).
Driver 580.159.03 on this VM (as provided by <cloud>).
Same build (4681d8b) and recipe as the home 1-Spark cards; drafting OFF (--mtp 0), thinking per the card's own switch.
started 15:30Z, ended 15:46Z
