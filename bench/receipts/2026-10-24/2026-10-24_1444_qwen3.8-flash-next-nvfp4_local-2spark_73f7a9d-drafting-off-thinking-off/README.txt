Engine 73f7a9dc157fe4e0d0217aacc4fe57a16812bb9b (release build, main 73f7a9d); exact two-Spark layer pipeline on spark-a (rank 0, front, HTTP) + spark-b (rank 1, back).
Drafting OFF, thinking OFF. Status PASS.
Build: prebuilt payload pre-73f7a9d (build-provenance sha 73f7a9d), installed on both Sparks by scp + tar -xf + sha256sum -c ARTIFACTS.sha256 (41/41 OK each).
Served sha256 (both Sparks): flash-serve 7d52cc1baf26ac4fcab1937ceaf2f70d5010eca0ef253b8db1bdf7e327cd4513; spark_layer_kernels.ptx 75858a8845518b3abe6ae1025288351c4d7f1a493fe0f5da33fac47edb7250d2.
Server (one start for both cards; rank supervisor script (not published), flock <path>):
  spark-a: <path> 0 2 <fabric-a>:18750 <fabric-b>:18750 <path> <path> 32768 127.0.0.1:18761 0 --fp4-local
  spark-b: <path> 1 2 <fabric-b>:18750 127.0.0.1:0 <path> <path> 32768 127.0.0.1:18761 0 --fp4-local
  env (both ranks, LC_ALL=C, inherited VADE_/FLASH_/SEQPAR_/TP_/MTP scrubbed first): FLASH_SERVE_OUTPUT_TOKENS=16384 LC_ALL=C MTP=3 SEQPAR_CONTROL_SECRET=<ephemeral session value> TP_RUN_ID=<run id> VADE_PROMPT_TILE=2048 VADE_W2_PP=1 VADE_W2_PP_ASYNC=1 VADE_W2_PP_MTP=1 VADE_W2_PP_RDMA=1
  fabric: QSFP point-to-point; addresses written as <fabric-a>/<fabric-b>; data transport uses LISTEN/NEXT ports + 1.
Client: the two-Spark benchmark client (not published) on spark-a against 127.0.0.1:18761 (every request sends draft=plain on the same server); --cold-runs 6 --prose-runs 3 --c8-runs 8 --c16-runs 8.
Order: smoke OFF, smoke ON, then the OFF block (cold, prose, C8, C16), then the ON block. Folder time = end of this mode's block (local clock).
Cells (this card): cold 8K actual-mode TTFT median n=6/6, one-chat prose median n=3/3 (1024 cap), C8 median 8 rounds, C16 median 8 rounds (256 cap, +1 warmup each).
C16: 16 offered chats; pipeline admits active=9 this start, excess HTTP 503 retried with identical sealed body; tok/s over first-attempt to last-end wall. Attempts receipted per stream.
Token IDs ON vs OFF (measured pairs): 201/201 identical; smoke_equal=True. Groups: {"c16": {"identical": 128, "total": 128}, "c16_warmup": {"identical": 16, "total": 16}, "c8": {"identical": 64, "total": 64}, "c8_warmup": {"identical": 8, "total": 8}, "cold": {"identical": 6, "total": 6}, "prose": {"identical": 3, "total": 3}, "smoke": {"identical": 1, "total": 1}}.
Median SM clock (union of measured windows): {"spark-a": 2450.0, "spark-b": 2457.0} MHz (spark-a/spark-b).
Every greedy request uses seed 20260905 (concurrency adds stream index) and cache=false; cold prompts are exact 8192 tokens with unique nonces. Greedy IDs are seed independent.
Recipe: the 2026-10-24 morning two-Spark cards (4681d8b, 0711/0720) reproduced with the same scripts, env, ports, prompts, sealed bodies and counts; differences: build 73f7a9d, OFF block first.
Files: thinking-off.card.txt/.json (card), receipts/ (raw per-request bodies, responses with token IDs, manifest, health, plan, control and measure logs, per-Spark command/env/ready/process/hash-check/clocks/telemetry), logs/ (server logs of both ranks). The same single-start receipts are in both the OFF and ON folders.
Published copies are sanitized: Spark names spark-a/spark-b, home paths ~, user <user>, fabric addresses <fabric-a>/<fabric-b>, run id <run id>.
