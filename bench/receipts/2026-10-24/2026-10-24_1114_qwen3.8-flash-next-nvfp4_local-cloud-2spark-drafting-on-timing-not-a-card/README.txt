NOT A RIGMARK CARD. Timing run only, drafting ON vs OFF on the 2-Spark cloud pair, 2026-10-24 15:59-16:14Z.
Box: <cloud> <cloud-cluster> cloud pair (two GB10 VMs, 116 GB RAM each, 200G RoCE ring), world 2 = layer pipeline, HTTP on the entry VM.
Build: Vade main 4681d8b2c83f3df09926046dd6bd8b7fd124ce69 (same build as every 10-24 card). Model: nvidia/Qwen3.8-Flash-Next-FP4-Local, context 32768, output cap 16384, greedy, thinking off in the timing bodies.
Server started with drafting ENABLED (default mtp depth 3; the OFF cards used --mtp 0). ON rows = draft "strict" (server default); OFF rows = draft "plain" (per-request switch, same server).
Scripts: timing-on.py (single stream, 6 distinct prose/code prompts, on/off/on) and timing-c8.py (8 threads, the card's concurrency/8 bodies, on/off x3). tok/s = completion tokens / wall seconds.
Single stream (medians):
  OFF code 26.2 (n=5)   OFF prose 25.2 (n=1)
  ON  code 56.5 (n=10, min 49.7 first run, max 58.1)   ON prose 32.0 (n=2: 30.3, 33.7)
C8 aggregate tok/s, 8 streams x 256 tokens:
  round 1: ON 85.4 (median stream 18.6)   OFF 70.4
  round 2: ON FAILED 8/8 (generation_error)   OFF 78.6
  round 3: ON FAILED 8/8 (generation_error)   OFF 78.7
Failure cause (rank 0 log, <path> on the entry VM, lines 676-760): "Requested size does not fit. Asked: allocation 218898432 bytes; live 94764697368; Available 88.42 GiB (94941617357) from the configured allocation ceiling; Gap 0.06 GiB". The ceiling on these VMs is 88.4 GiB (MemAvailable x 0.9 cap; SEQPAR_MEMORY_BYTES=100 GiB was set but the lower cap wins; rank 1 reports budget_bytes=95411181773). After the first C8 drafting round and one C8 plain round the pooled storage had grown from 85.2 GB to 94.8 GB live, so the next 8 drafting sessions could not get their 209 MB and every request in the round failed instead of queueing. Single-stream on/off/on transitions never failed. The same transition has headroom on a 128 GB home Spark, which is why the home C8 ON/OFF cards show no errors.
Comparison (OFF cards, same build): cloud pair card 10:06 C8 70.0 / code 25.4 / prose 23.2; cloud 1-VM card 10:46 C8 43.0; home 1-Spark cards 06:5x C8 OFF 136.8 / ON 171.7, prose ON 42.0 / OFF 27.8.
A second engine check on the same pair, 16:19-16:38Z: with the capacity set to 4 active sessions per rank and a common 108 GB memory budget, C8 aggregate medians were main (A) drafting ON 91.2 / OFF 73.7 tok/s and candidate 8c9411f (X) ON 100.0 / OFF 78.0; 64 exact SSE responses identical across A/X and ON/OFF; zero errors in its ON/OFF/ON/OFF sequences (the allocation failure above did not reproduce at that capacity setting). SM clocks ~2470 MHz on both ranks.

18:0xZ update: the failing sequence was rerun on the fixed engine (main 407b6ae) and passed 3/3 ON rounds with 0 errors; see 2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card/README.txt.
