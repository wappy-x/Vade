NOT A RIGMARK CARD. Timing run only (drafting ON vs OFF, single stream), 2026-10-24 15:1x-15:42Z.
Box: <cloud> <cloud-cluster> cloud pair, ENTRY VM (<cloud-vm>, one GB10 VM, 116 GB RAM, driver 580.178.04), world 1 (one Spark, no link).
Build: Vade main 4681d8b2c83f3df09926046dd6bd8b7fd124ce69 (same build as the 10:06 2-Spark cloud card and the home cards).
Model: nvidia/Qwen3.8-Flash-Next-FP4-Local (same checkpoint sha as the cards). Context 32768, thinking OFF (the request bodies came from the thinking-off card; the server log shows thinking:false on every timing request), greedy, output cap 16384. [Corrected 16:48Z: an earlier version of this file said thinking on.]
Server started with drafting ENABLED (default mtp depth 3). Rows with draft="strict" = drafting ON; rows with draft="plain" = drafting OFF (per-request switch, same server).
Script: <path> timing-on.py (6 distinct prose/code decode prompts, modes on/off/on). tok_s_wall = completion_tokens / wall seconds, single stream.
Summary (medians of tok_s_wall):
  OFF code  25.8 tok/s (n=5)     ON code  64.0 tok/s (n=10, min 56.4, max 66.2)   -> 2.5x
  OFF prose 24.5 tok/s (n=1)     ON prose 33.6 tok/s (n=2,  31.7 / 35.6)          -> 1.4x
Caveat: the mtp_depth field in the rows reads 0 for every row (the response does not report the server depth); ON/OFF is the draft field.
Reference (10-23/10-24 cards, one Spark, same build family): C8 ON 171.7 vs OFF 136.8 tok/s aggregate.
