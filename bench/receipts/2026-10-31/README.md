# Receipts, 2026-10-31 morning, main dc32b7c

These receipts back the [current FP4-Local results](../../../docs/results.md#current-fp4-local-main-dc32b7c-2026-10-31-morning). The [2026-11-01 receipts](../2026-11-01/README.md) replace the FP8 numbers here. A card is the result file of one RigMark run. Each JSON file here is a sanitized extract of one card file or of one 32K chat file.

The full card files are private. Each one is about 2 MB and holds the prompts, the replies and the run timestamps.

## What was run

- **Date.** The runs were on 2026-10-31, from 11:26 to about 12:25 UTC. The end time is the file mtime of the last 32K chats.
- **Build.** The build is main `dc32b7c1366c56c29fc1f3fee9e45fc85b6e8afa`, clean tree, from a source archive. In the public repository, commit `db6b7fcd1f0ed97d6d21084eab1d206cc0d5ef3c` has the same tree (`aea2c506bbbc3c6b811d265aa7c2998d5916aa02`).
- **Server binary.** The `flash-serve` SHA256 is `55bdc7a8a5dbc1d3c5b61a4065da0921a28bf1f5643a1463ed3b428aabb43b4d`. Every Spark ran the same file.
- **GPU code.** The PTX SHA256 is `84c093501cfd74f3d7d9a9748e814a179d3586a41529766e8c794984aa94006c`.
- **FP4-Local checkpoint.** The checkpoint is `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`. Its identity is `d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`.
- **FP8 checkpoint.** The checkpoint is `Qwen/Qwen3.8-Flash-Next-FP8` at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`. Its identity is `8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03`.
- **Machines.** The machines are DGX Spark (NVIDIA GB10), Linux 7.0.0-1019-nvidia aarch64, driver 580.178.04. GPU clocks were not locked. `nvidia-smi` read an SM clock of 2405 to 2411 MHz at each server start.
- **Links.** Two Sparks use one direct cable with RDMA. Four Sparks are four machines cabled in a line (rank 0 to 3), with RDMA.
- **Environment.** No Spark had a `VADE_` setting.

### Server starts

<div align="center">

| Setup | Server start | Path that `/health` reports | Starts |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | `vade up --sparks 1 --context 32768 --output 8192` | none (one Spark) | 2 |
| FP4-Local · 2 Sparks | `vade up --sparks 2 --context 32768 --output 8192` | `tensor_split` | 1 |
| FP4-Local · 4 Sparks | private launcher, see below | `tensor_split` | 1 |
| FP8 · 2 Sparks | private launcher, see below | `ring` | 1 |
| FP8 · 4 Sparks | private launcher, see below | `ring` | 1 |

</div>

### Limits

<div align="center">

| Setup | Context | Output limit | Drafting depth |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 32768 | 8192 | 3 |
| FP4-Local · 2 Sparks | 32768 | 8192 | 3 |
| FP4-Local · 4 Sparks | 32768 | 8192 | 3 |
| FP8 · 2 Sparks | 32768 | 16384 | 2 |
| FP8 · 4 Sparks | 32768 | 16384 | 2 |

</div>

The private launcher starts the same `flash-serve` binary with the same context, output limit and drafting depth, over RDMA, with no `VADE_` setting. It is a private script. These are the public commands for the same setups:

```sh
vade up --sparks 4 --context 32768 --output 8192
vade up --fp8 --sparks 2 --context 32768 --output 16384
vade up --fp8 --sparks 4 --context 32768 --output 16384
```

We did not run a card with those public commands on this date.

### What each start ran

- **Drafting on.** Each server start ran one card with drafting on. Its 60 chat requests leave out `draft`, so the server drafts at its default depth.
- **Drafting off.** Each server start ran one card with drafting off. Its 60 chat requests send `"draft": "plain"`.
- **The 8K prompts.** In both cards, the six 8K prompt requests send `"draft": "plain"` and no cache opt-in.
- **The 32K chat.** On two and four Sparks, the same start also ran the 32K chat. On two FP4-Local Sparks, it ran after both cards. On the four-Spark and FP8 starts, it ran between the two cards. On one Spark, the 32K chat ran on two separate starts with context 65536.

## The files

- **`<setup>-on.json` and `<setup>-off.json`.** Each file holds one card. It gives the build, the checkpoint identity, the server path, the limits, the client revision and prompt hash, and the request settings. It also gives the card's own median for each cell, the output checks, and a SHA256 of all 66 token ID arrays.
- **`<setup>-chat32k.json`.** Each file holds the 32K chat. It gives the prompt and reply lengths, the time to the first token, the cached tokens and a SHA256 of the token IDs for each turn.
- **`fresh-install-2026-10-30.txt`.** This file holds the timing of the fresh clone and `sh scripts/install.sh` on one Spark.

`token_ids_sha256` is the SHA256 of the compact JSON array of the 66 token ID arrays. The arrays are in this order:

1. The prose, code and structured runs.
2. The cold 8K runs, then the warm 8K runs.
3. The C1, C2, C4 and C8 rounds and their streams.

Two cards with the same value gave the same token IDs in every response. All eight FP4-Local cards (one, two and four Sparks, drafting on and off) have `805f06d0...`. The FP8 two-Spark cards have `2af8329f...`, and the FP8 four-Spark cards have `b078ba58...`.

We made the extract with `jq` from each card file. Nothing was measured again for these files, and no value was rounded.

## What you can and cannot re-run from this repository

- **The card client is now public.** `vade bench --card` sends the RigMark prompt set of revision `52c29c3fc1d741b9aef8895b23372aafa5dc6161` (prompts `0c3ac401...`) with the request settings in each file. It records the token IDs and compares them with a file here ([details](../../../docs/results.md#what-a-stranger-can-and-cannot-re-run)). On 2026-10-31 it matched the one-Spark drafting-on and drafting-off receipts. We have not run it against the two-Spark, four-Spark or FP8 receipts. Plain public [RigMark](https://github.com/alexellis/rigmark) at that revision sends a different cold 8K request ([details](../../../docs/benchmarks.md#run-rigmark-against-vade)).
- **The 32K chat client and its prompt are private.** `vade chat` prints the cached tokens and the time to the first token for each turn, but it sends other prompts.
- **The four-Spark and FP8 starts used a private launcher.** See [Server starts](#server-starts).
