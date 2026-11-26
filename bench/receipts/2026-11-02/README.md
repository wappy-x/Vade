# Receipts, 2026-11-02 morning, main 8976c4f

These receipts back the [current FP4-Local results](../../../docs/results.md#current-fp4-local-main-8976c4f-2026-11-02-morning). A card is the result file of one RigMark run. Each JSON file here is a sanitized extract of one card file or of one 32K chat file.

The full card files are private. Each one is 2.0 to 2.7 MB and holds the prompts, the replies and the run timestamps.

These receipts cover FP4-Local only. The FP8 receipts are still the [2026-11-01 receipts](../2026-11-01/README.md).

## What was run

- **Date.** The runs were on 2026-11-02, from 12:44 to about 13:22 UTC. The start time comes from the first card file. The end time is the file mtime of the last 32K chat.
- **Build.** The build is main `8976c4f21259e579531003341ab3a5222709490f`, clean tree, from a source archive. In the public repository, commit `42dce142dda4f62bd19c214b4baa46ba751aff6a` has the same tree (`da775b701c5156e0fe31ef6270402f291737b66f`). Main `407aa46eca91cb7fc64e7d699761f99b40d544cb` has the same engine. Between the two, only the command line tool, its tests and the changelog changed.
- **Server binary.** The `flash-serve` SHA256 is `0f446fe3771c3c9abc30ccbda501391944a5fddc24c5abe30506ca82bba525c8`. Every Spark ran the same file.
- **GPU code.** The PTX SHA256 is `f5f78a9169779304635fc5ef0c2796efd248d1cc1ae45f6410d979017aed017e`.
- **FP4-Local checkpoint.** The checkpoint is `nvidia/Qwen3.8-Flash-Next-FP4-Local`. Its identity is `d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`. This is the same checkpoint as on 2026-10-31.
- **Machines.** The machines are DGX Spark (NVIDIA GB10), platform `Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39`, driver 580.178.04. GPU clocks were not locked. `nvidia-smi` read an SM clock of 2405 to 2411 MHz and a GPU temperature of 42 to 51 C at each server start.
- **Links.** Two Sparks use one direct cable with RDMA. Four Sparks are four machines cabled in a line (rank 0 to 3), with RDMA.
- **Environment.** No Spark had a `VADE_` setting.

### Server starts

<div align="center">

| Setup | Server start | Path that `/health` reports | Starts |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | `vade up --sparks 1 --context 32768 --output 8192` | none (one Spark) | 1 |
| FP4-Local · 2 Sparks | `vade up --sparks 2 --context 32768 --output 8192` | `tensor_split` | 1 |
| FP4-Local · 4 Sparks | `vade up --sparks 4 --context 32768 --output 8192` | `tensor_split` | 1 |

</div>

Every start used context 32768, output limit 8192 and the default drafting depth. `/health` reported depth 3 on each start.

### What each start ran

- **Drafting on.** Each server start ran one card with drafting on. Its 60 chat requests leave out `draft`, so the server drafts at its default depth.
- **Drafting off.** Each server start ran one card with drafting off. Its 60 chat requests send `"draft": "plain"`.
- **The 8K prompts.** In every card, the 8K prompt requests send `"draft": "plain"` and no cache opt-in. No cold request and no replay restored a cached token.
- **The 32K chat.** Each start ran the 32K chat three times. On one Spark, the chats ran between the two cards. On two and four Sparks, they ran after both cards. The chat files here hold the first of the three chats.
- **One-Spark card shape.** The one-Spark cards send five cold and five warm 8K prompts, not three, and they leave out the C2 round. So they hold 64 responses, not 66. The two-Spark and four-Spark cards hold the usual 66.

## The files

- **`fp4-local-1spark-on.json` and `fp4-local-1spark-off.json`.** One Spark, drafting on and off.
- **`fp4-local-1spark-chat32k.json`.** One Spark, the 32K chat.
- **`fp4-local-2spark-on.json` and `fp4-local-2spark-off.json`.** Two Sparks, drafting on and off.
- **`fp4-local-2spark-chat32k.json`.** Two Sparks, the 32K chat.
- **`fp4-local-4spark-on.json` and `fp4-local-4spark-off.json`.** Four Sparks, drafting on and off.
- **`fp4-local-4spark-chat32k.json`.** Four Sparks, the 32K chat.

Each card file gives the build, the checkpoint identity, the server path, the limits, the client revision and prompt hash, and the request settings. It also gives the card's own median for each cell, the output checks, and a SHA256 of all token ID arrays. Each chat file gives the prompt and reply lengths, the time to the first token, the cached tokens and a SHA256 of the token IDs for each turn.

`token_ids_sha256` is the SHA256 of the compact JSON array of the card's token ID arrays. The arrays are in this order:

1. The prose, code and structured runs.
2. The cold 8K runs, then the warm 8K runs.
3. The C1, C2, C4 and C8 rounds and their streams. The one-Spark cards have no C2.

Two cards with the same value gave the same token IDs in every response. These are the values:

- **Two and four Sparks.** The drafting-on and drafting-off cards all have `805f06d0a7e80c33d3c311b65178aadc29de800a5750602e3dbdc83c60fd006a`. This is also the value of all eight FP4-Local cards of 2026-10-31.
- **One Spark.** The drafting-on and drafting-off cards both have `c58a4a3096dc9edd737abad0e9d124b7d454ae91ea9fc0e847f566b441d0aed0`. This value covers 64 responses, so it cannot equal a 66-response value.
- **One against two and four Sparks.** We compared the full card files with the lab's reference cards. The one-Spark cards matched the one-Spark reference cards in 64 of 64 responses. The four-Spark cards matched the two-Spark card and the one-Spark reference card in 66 of 66.
- **The 32K chat.** The chat files of one, two and four Sparks have the same turn digests, `8f7032ee...` and `82fe908a...`. They are the same as the 2026-10-31 FP4-Local chats. In each file, the second question gave the same token IDs with the cache and without it (`turn2_ids_equal`).

We made the extract with `jq` from each card file. Nothing was measured again for these files, and no value was rounded.

## What you can and cannot re-run from this repository

- **The card client is public.** `vade bench --card` sends the 66 requests of the RigMark prompt set of revision `52c29c3fc1d741b9aef8895b23372aafa5dc6161` (prompts `0c3ac401...`). With `--receipt`, it compares its token IDs with a 66-response file here ([details](../../../docs/results.md#what-a-stranger-can-and-cannot-re-run)). It cannot check the 64-response one-Spark files. We have not run it against these receipts.
- **The 32K chat client and its prompt are private.** `vade chat` prints the cached tokens and the time to the first token for each turn, but it sends other prompts.
