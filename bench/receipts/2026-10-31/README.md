# Receipts, 2026-10-31 morning, main dc32b7c

These receipts back the [current results](../../../docs/results.md). Each
JSON file is a sanitized extract of one card file or one 32K chat file. The
full card files are private: each one is about 2 MB and holds the prompts,
the replies and the run timestamps.

## What was run

| Item | Value |
| --- | --- |
| Date | 2026-10-31, from 11:26 to about 12:25 UTC (the file mtimes of the last 32K chats) |
| Build | main `dc32b7c1366c56c29fc1f3fee9e45fc85b6e8afa`, clean tree, from a source archive. In the public repository, commit `db6b7fcd1f0ed97d6d21084eab1d206cc0d5ef3c` has the same tree (`aea2c506bbbc3c6b811d265aa7c2998d5916aa02`). |
| Server binary | `flash-serve` SHA256 `55bdc7a8a5dbc1d3c5b61a4065da0921a28bf1f5643a1463ed3b428aabb43b4d`, the same file on every Spark |
| GPU code | PTX SHA256 `84c093501cfd74f3d7d9a9748e814a179d3586a41529766e8c794984aa94006c` |
| FP4-Local checkpoint | `nvidia/Qwen3.8-Flash-Next-FP4-Local` revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, identity `d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716` |
| FP8 checkpoint | `Qwen/Qwen3.8-Flash-Next-FP8` revision `236dfdf285828023ca3bcd3f37366c58a3469b13`, identity `8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03` |
| Machines | DGX Spark (NVIDIA GB10), Linux 7.0.0-1019-nvidia aarch64, driver 580.178.04. GPU clocks were not locked; `nvidia-smi` read an SM clock of 2405 to 2411 MHz at each server start. |
| Links | Two Sparks: one direct cable, RDMA. Four Sparks: four machines cabled in a line (rank 0 to 3), RDMA. |
| Environment | No `VADE_` setting on any Spark |

| Setup | Server start | Context | Output limit | Drafting depth | Path that `/health` reports | Starts |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| FP4-Local, one Spark | `vade up --sparks 1 --context 32768 --output 8192` | 32768 | 8192 | 3 | none (one Spark) | 2 |
| FP4-Local, two Sparks | `vade up --sparks 2 --context 32768 --output 8192` | 32768 | 8192 | 3 | `tensor_split` | 1 |
| FP4-Local, four Sparks | private launcher, see below | 32768 | 8192 | 3 | `tensor_split` | 1 |
| FP8, two Sparks | private launcher, see below | 32768 | 16384 | 2 | `ring` | 1 |
| FP8, four Sparks | private launcher, see below | 32768 | 16384 | 2 | `ring` | 1 |

The private launcher starts the same `flash-serve` binary with the same context,
output limit and drafting depth, over RDMA, with no `VADE_` setting. It is a
private script. The public commands for the same setups are
`vade up --sparks 4 --context 32768 --output 8192` and
`vade up --fp8 --sparks 2 --context 32768 --output 16384` (or `--sparks 4`).
We did not card those public commands on this date.

Each server start ran one card with drafting ON (its 60 chat requests leave
out `draft`, so the server drafts at its default depth) and one card with
drafting OFF (its 60 chat requests send `"draft": "plain"`). In both cards,
the six 8K prompt requests send `"draft": "plain"` and no cache opt-in. On two and four Sparks, the
same start also ran the 32K chat: after both cards on two FP4-Local Sparks,
between the two cards on the four-Spark and FP8 starts. On one Spark, the
32K chat ran on two separate starts with context 65536.

## The files

| File | What it holds |
| --- | --- |
| `<setup>-on.json`, `<setup>-off.json` | One card: build, checkpoint identity, server path, limits, client revision and prompt hash, request settings, the card's own median for each cell, the output checks, and a SHA256 of all 66 token ID arrays |
| `<setup>-chat32k.json` | The 32K chat: prompt and reply lengths, time to the first token, cached tokens and a SHA256 of the token IDs for each turn |
| `fresh-install-2026-10-30.txt` | The fresh clone and `sh scripts/install.sh` timing on one Spark |

`token_ids_sha256` is the SHA256 of the compact JSON array of the 66 token ID
arrays, in this order: the prose, code and structured runs; the cold and then
the warm 8K runs; the C1, C2, C4 and C8 rounds and their streams. Two cards
with the same value gave the same token IDs in every response. All eight
FP4-Local cards (one, two and four Sparks, drafting ON and OFF) have
`805f06d0...`. The FP8 two-Spark cards have `2af8329f...` and the FP8
four-Spark cards `b078ba58...`.

The extract was made with `jq` from each card file. Nothing was measured
again for these files, and no value was rounded.

## What you cannot re-run from this repository

- **The card client is private.** It sends the RigMark prompt set of
  revision `52c29c3fc1d741b9aef8895b23372aafa5dc6161` (prompts
  `0c3ac401...`) with the request settings in each file, and it records the
  token IDs. Public [RigMark](https://github.com/alexellis/rigmark) at that
  revision runs the same cells, but its cold 8K request differs
  ([details](../../../docs/benchmarks.md#run-rigmark-against-vade)). We have
  not checked whether a public RigMark card records the same token ID arrays.
- **The 32K chat client and its prompt are private.** `vade chat` prints the
  cached tokens and the time to the first token for each turn, but it sends
  other prompts.
- **The four-Spark and FP8 starts used a private launcher** (see above).
