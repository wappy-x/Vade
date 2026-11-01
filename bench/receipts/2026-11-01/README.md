# Receipts, 2026-11-01 morning, main 1a07e3a

These receipts back the [current results](../../../docs/results.md). A card is the result file of one RigMark run. Each JSON file here is a sanitized extract of one card file or of one 32K chat file.

The full card files are private. Each one is 2.0 to 2.7 MB and holds the prompts, the replies and the run timestamps.

These receipts cover FP8 only. The FP4-Local receipts are still the [2026-10-31 receipts](../2026-10-31/README.md).

## What was run

- **Date.** The runs were on 2026-11-01, from 12:25 to 13:39 UTC. The times come from the start and stop logs of the server starts.
- **Build.** The build is main `1a07e3ac48c5c943df21d2c842e6a7b97c03276a`, clean tree, from a source archive. Main `d913e84886cdb6f5bada6329d7a6e92cbffa72ce` has the same engine. Between the two, only documentation, help text and one example config changed.
- **Server binary.** The `flash-serve` SHA256 is `c5edc3e140c0e04800cac441e98b05f0221dae760b570bf631798a06cac6c741`. Every Spark ran the same file. The start logs check it on each rank.
- **GPU code.** The PTX SHA256 is `f5f78a9169779304635fc5ef0c2796efd248d1cc1ae45f6410d979017aed017e`.
- **FP8 checkpoint.** The checkpoint identity is `8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03`. This is the same checkpoint as on 2026-10-31.
- **Machines.** The machines are DGX Spark (NVIDIA GB10), platform `Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39`. GPU clocks were not locked. These start logs do not record the driver version or the clocks. Each server started with every GPU at 46 C or lower.
- **Links.** Two Sparks use one direct cable with RDMA. Four Sparks are four machines cabled in a line (rank 0 to 3), with RDMA. These are the same machines as the 2026-10-31 FP8 cards.
- **Environment.** On four Sparks, no Spark had a `VADE_` setting. The two-Spark start logs do not record this count.

### Server starts

<div align="center">

| Setup | Drafting depth | Path that `/health` reports | Starts |
| :---: | :---: | :---: | :---: |
| FP8 · 2 Sparks | 2, fixed | `ring` | 1 |
| FP8 · 2 Sparks | chosen by the server | `ring` | 1 |
| FP8 · 4 Sparks | 2, fixed | `ring` | 1 |
| FP8 · 4 Sparks | chosen by the server | `ring` | 1 |

</div>

There were four server starts. Every start used a private launcher with context 32768 and output limit 16384, over RDMA. The fixed-depth starts pass drafting depth 2. The chooser starts pass no drafting depth, so the server chooses it. On both chooser starts, `/health` reported depth 7.

One more two-Spark chooser start, before the counted one, gave no card. The card client then accepted only depths 0, 2 and 3, and it stopped at the first chat. The chooser cards used a copy of the client that also accepts depths 4 to 7. Nothing else in the client changed.

These are the public commands for the same setups:

```sh
vade up --fp8 --sparks 2 --context 32768 --output 16384
vade up --fp8 --sparks 4 --context 32768 --output 16384
```

We did not run a card with those public commands on this date.

### What each start ran

- **Drafting on.** Each fixed-depth start ran one card with drafting on. Its 60 chat requests leave out `draft`, so the server drafts at its depth.
- **Drafting off.** Each fixed-depth start ran one card with drafting off. Its 60 chat requests send `"draft": "plain"`.
- **Chooser.** Each chooser start ran one card with drafting on, with the same requests as the drafting-on card.
- **The 8K prompts.** In every card, the 8K prompt requests send `"draft": "plain"` and no cache opt-in.
- **The 32K chat.** On two Sparks, the fixed-depth start ran the 32K chat after both cards. On four Sparks, it ran between the two cards. The two-Spark chooser start also ran the 32K chat. That file is not here.
- **Two-Spark card shape.** The two-Spark cards send five cold and five warm 8K prompts, not three, and they leave out the C2 round. So they hold 64 responses, not 66. The four-Spark cards hold the usual 66.

## The files

- **`fp8-2spark-on.json` and `fp8-2spark-off.json`.** Two Sparks, drafting depth 2, drafting on and off.
- **`fp8-2spark-on-chooser.json`.** Two Sparks, depth chosen by the server, drafting on.
- **`fp8-2spark-chat32k.json`.** Two Sparks, the 32K chat of the depth 2 start.
- **`fp8-4spark-on.json` and `fp8-4spark-off.json`.** Four Sparks, drafting depth 2, drafting on and off.
- **`fp8-4spark-on-chooser.json`.** Four Sparks, depth chosen by the server, drafting on.
- **`fp8-4spark-chat32k.json`.** Four Sparks, the 32K chat of the depth 2 start.

Each card file gives the build, the checkpoint identity, the server path, the limits, the client revision and prompt hash, and the request settings. It also gives the card's own median for each cell, the output checks, and a SHA256 of all token ID arrays. Each chat file gives the prompt and reply lengths, the time to the first token, the cached tokens and a SHA256 of the token IDs for each turn.

`token_ids_sha256` is the SHA256 of the compact JSON array of the card's token ID arrays. The arrays are in this order:

1. The prose, code and structured runs.
2. The cold 8K runs, then the warm 8K runs.
3. The C1, C2, C4 and C8 rounds and their streams. The two-Spark cards have no C2.

Two cards with the same value gave the same token IDs in every response. These are the values:

- **Four Sparks.** The drafting-on, drafting-off and chooser cards all have `2af8329ff3867b00165411156482d906c05cf6416bbabe2f3fe6bbe524e124a2`. This is also the value of the 2026-10-31 FP8 two-Spark cards.
- **Two Sparks.** The drafting-on, drafting-off and chooser cards all have `f1c1a5ff1b333158bbc9f9e93bd24bc3f980966852f9050607493710c750bada`. This value covers 64 responses, so it cannot equal a 66-response value.
- **Two against four Sparks.** We compared the full card files. The 15 single chats, the first three cold and warm 8K runs and the C1, C4 and C8 streams gave the same token IDs on two and four Sparks.
- **The 32K chat.** Both chats have the same turn digests, `28a6dca8...` and `b01b5a6b...`. They are the same as the 2026-10-31 FP8 two-Spark chat.

We made the extract with `jq` from each card file. Nothing was measured again for these files, and no value was rounded.

## What you can and cannot re-run from this repository

- **The card client is public.** `vade bench --card` sends the 66 requests of the RigMark prompt set of revision `52c29c3fc1d741b9aef8895b23372aafa5dc6161` (prompts `0c3ac401...`). With `--receipt`, it compares its token IDs with a 66-response file here ([details](../../../docs/results.md#what-a-stranger-can-and-cannot-re-run)). It cannot check the 64-response two-Spark files. We have not run it against these receipts.
- **The 32K chat client and its prompt are private.** `vade chat` prints the cached tokens and the time to the first token for each turn, but it sends other prompts.
- **The starts used a private launcher.** See [Server starts](#server-starts).
