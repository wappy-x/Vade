Complete RigMark 1.3.0 card — FIRST 2-SPARK RUN (informative, no gate), MTP OFF, DRAFTING OFF; Vade 4681d8b2c83f3df09926046dd6bd8b7fd124ce69: world 2 layer pipeline (--mtp 0, greedy).
2 x GB10: rented <cloud> 2-node GB10 cluster (<cloud-cluster>, entry <cloud-id>) (2 VMs, one NVIDIA GB10 each, 116 GB RAM per VM, driver 580.178.04 (rank 0) / 580.159.03 (rank 1)); 200G QSFP point-to-point ring (two links between the two VMs), RoCE, SEQPAR_RDMA=1; ranks: <cloud-vm> rank0 (HTTP 127.0.0.1:8000) <fabric>; <cloud-vm>-w1 rank1 <fabric>; NVFP4 FP4-Local checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local @ d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716 served with --fp4-local; engine topology tag 'pipeline'.
Server: flash-serve world 2 native rank path (--tensor-map, --mtp 0); rank0 binds HTTP 127.0.0.1:8000; context 32768 / output 16384; --mtp 0 and requests draft plain = drafting OFF; thinking OFF. Command: flash-serve <rank> 2 <listen> <next> spark_layer_kernels.ptx <model> 32768 127.0.0.1:8000 0 --fp4-local --mtp 0 (layer pipeline, SEQPAR_MEMORY_BYTES=107374182400 explicit budget for the container, SEQPAR_RDMA=1 TP_TOPOLOGY=ring FLASH_SERVE_OUTPUT_TOKENS=16384)
Test-start: health ready + one chat request OK at 14:51Z
Pinned RigMark corpus and request functions; decode 5 x 8192 cap; prefill 3 cold/replay 8192 pairs (standard); C1/C2/C4/C8 x3 x256; seed 20260905.
Completed.
First 2-Spark run on this engine: informative, no gate; no 2-Spark baseline to compare against.
Compare (different Spark counts, descriptive only): 2-Spark 2026-10-22_1608_qwen3.8-flash-next-nvfp4_local-2spark_bc85082-drafting-off-thinking-off; 1-Spark main 2026-10-22_1237_qwen3.8-flash-next-nvfp4_local-1spark_6032c30-thinking-off.
Status PASS; gates 15/15; audit: 66 responses, 0 with provenance errors; health OK; rank logs 2/2 present.
Checkpoint fetched from Hugging Face on rank 0 (sha256 of every file checked against the LFS oid) and copied to rank 1 over the 200G link.
Container memory: the engine's derived ceiling (CUDA free + reclaimable cache) came out 0.07-0.09 GiB short on both ranks because the host caches the weight read; an explicit SEQPAR_MEMORY_BYTES=100 GiB budget was set, as the engine's own advice for containers says.
Rank 1 ships driver 580.159.03, rank 0 580.178.04 (as provided by <cloud>).
Same build (4681d8b) and recipe as the home 2-Spark card 2026-10-24_0720 (spark-a+spark-b).
started 14:51Z, ended 15:06Z
