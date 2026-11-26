# Results and receipts

This page gives the current benchmark numbers of Vade. Qwen3.8 Flash-Next comes first (FP8, then FP4-Local), followed by [GLM 5.3 Flash](#glm-53-flash-two-sparks-rc24) and [Inkling 2 Mini](#inkling-2-mini-one-spark-rc26). Verified measurements from unmerged private development branches are kept in their own section at the end ([unmerged private-branch receipts](#unmerged-private-branch-receipts-not-on-main)).

The [README](../README.md) quotes some of these numbers. Each number has a record. A record in this repository is linked. When a record is not in this repository, the page says so.

These words appear on the whole page:

- **Speeds.** All speeds are completion tokens per second (tok/s).
- **Times.** All times are in seconds.
- **Card.** A card is the result file of one RigMark run.
- **Drafting.** The model's own prediction head guesses a few tokens ahead, and the full model checks each guess.
- **Main.** A short revision such as `dc32b7c` names a commit of the private development history. The public repository holds squashed copies of those commits ([glossary](GLOSSARY.md#ho), "Main `abc1234`"). Each receipt gives the full 40-character revision.

## Current FP8: main acb49bc, 2026-11-01 afternoon

These are the FP8 numbers of the afternoon of 2026-11-01. Each list is one setup.

- **Prose, code and structured** are one chat each.
- **1, 2, 4 and 8 chats** give the total rate of that many chats at once.
- **8K first token** is the time to the first token after a cold 8192-token prompt. Cold means uncached.

**Two Sparks, drafting on, depth 2**

- **Prose:** 39.504 tok/s
- **Code:** 55.734 tok/s
- **Structured:** 57.265 tok/s
- **1 chat:** 48.842 tok/s
- **4 chats:** 72.889 tok/s
- **8 chats:** 106.440 tok/s
- **8K first token:** 5.899 s
- **Receipt:** [fp8-2spark-acb49bc-on.json](../bench/receipts/2026-11-01/fp8-2spark-acb49bc-on.json)

**Two Sparks, drafting off**

- **Prose:** 28.374 tok/s
- **Code:** 28.205 tok/s
- **Structured:** 28.454 tok/s
- **1 chat:** 27.349 tok/s
- **4 chats:** 73.767 tok/s
- **8 chats:** 106.392 tok/s
- **8K first token:** 5.891 s
- **Receipt:** [fp8-2spark-acb49bc-off.json](../bench/receipts/2026-11-01/fp8-2spark-acb49bc-off.json)

**Four Sparks, drafting on, depth 2**

- **Prose:** 52.621 tok/s
- **Code:** 74.237 tok/s
- **Structured:** 75.489 tok/s
- **1 chat:** 63.668 tok/s
- **2 chats:** 59.880 tok/s
- **4 chats:** 95.638 tok/s
- **8 chats:** 136.174 tok/s
- **8K first token:** 6.866 s
- **Receipt:** [fp8-4spark-acb49bc-on.json](../bench/receipts/2026-11-01/fp8-4spark-acb49bc-on.json)

**Four Sparks, drafting off**

- **Prose:** 36.916 tok/s
- **Code:** 37.459 tok/s
- **Structured:** 37.684 tok/s
- **1 chat:** 36.223 tok/s
- **2 chats:** 57.625 tok/s
- **4 chats:** 91.895 tok/s
- **8 chats:** 136.567 tok/s
- **8K first token:** 6.801 s
- **Receipt:** [fp8-4spark-acb49bc-off.json](../bench/receipts/2026-11-01/fp8-4spark-acb49bc-off.json)

**How we ran these setups**

The receipts page is [bench/receipts/2026-11-01](../bench/receipts/2026-11-01/README.md). The files of this section have `acb49bc` in their names. The page gives the build and binary hashes, the checkpoint identity, the server starts and the request settings.

- **Machines.** The machines are DGX Spark, NVIDIA GB10, with the clocks not locked.
- **Build.** Public commit `952791b` has the tree of main `acb49bc`.
- **Server starts.** Each setup had one server start, from a private launcher with context 32768, output limit 16384 and drafting depth 2.
- **What each start ran.** Each start ran the drafting-on card, the drafting-off card and the 32K chat.
- **Depth chosen by the server.** These starts ran no card with the depth chosen by the server. The newest chooser cards are the [morning cards](#current-fp8-main-1a07e3a-2026-11-01-morning).

These are the public commands for the same setups. We did not run a card with them on this date.

```sh
vade up --fp8 --sparks 2 --context 32768 --output 16384
vade up --fp8 --sparks 4 --context 32768 --output 16384
```

How to read the lists:

- **Medians.** Each number is the card's own median. Every list is one server start.
- **Requests.** The request settings are the same as in the [morning cards](#current-fp8-main-1a07e3a-2026-11-01-morning).
- **Two-Spark cards.** They hold 64 responses, and they have no 2 chats rate.
- **Cold prompts.** No cold request restored a cached token. Every replay restored all 8,192 tokens.

**The 32K-token chat**

The chat client and the questions are the same as in the morning. On two Sparks the chat ran after both cards. On four Sparks it ran between them.

<div align="center">

| Setup | First question | Second question, cache | Second question, no cache |
| :---: | :---: | :---: | :---: |
| FP8 · 2 Sparks | 23.840 | 0.816 | 23.732 |
| FP8 · 4 Sparks | 27.337 | 0.755 | 27.279 |

</div>

The cached run reused 32,064 tokens on both setups. The second question gave the same token IDs with the cache and without it (`turn2_ids_equal`). The receipts are [two Sparks](../bench/receipts/2026-11-01/fp8-2spark-acb49bc-chat32k.json) and [four Sparks](../bench/receipts/2026-11-01/fp8-4spark-acb49bc-chat32k.json).

### FP8 token IDs

- **FP8, four Sparks.** The drafting-on and drafting-off cards have the same `token_ids_sha256`, `2af8329ff3867b00165411156482d906c05cf6416bbabe2f3fe6bbe524e124a2`. This is also the value of the morning four-Spark cards and of the 2026-10-31 FP8 two-Spark cards. So these four-Spark cards gave the same token IDs as those two-Spark cards in all 66 responses.
- **FP8, two Sparks.** The drafting-on and drafting-off cards have the same value, `f1c1a5ff1b333158bbc9f9e93bd24bc3f980966852f9050607493710c750bada`. This is also the value of the morning two-Spark cards. It covers 64 responses.
- **FP8, two against four Sparks.** We compared the full card files. Every response that both cards sent gave the same token IDs, with drafting on and off: the 15 single chats, the first three cold and warm 8K runs, and the C1, C4 and C8 streams.
- **The 32K chat.** Both setups gave the turn digests `28a6dca8...` and `b01b5a6b...`, the same as in the morning.
- **Re-run.** `vade bench --card --receipt` can check the four-Spark files. It cannot check the 64-response two-Spark files. We have not run it against these receipts.

<a id="current-fp8-main-1a07e3a-2026-11-01-morning"></a>

## Earlier FP8: main 1a07e3a, 2026-11-01 morning

The [afternoon cards](#current-fp8-main-acb49bc-2026-11-01-afternoon) replace the depth 2 and drafting-off rows of this section. Its chooser rows are still the newest chooser cards.

**Two Sparks, drafting on, depth 2**

- **Prose:** 37.845 tok/s
- **Code:** 52.981 tok/s
- **Structured:** 54.460 tok/s
- **1 chat:** 46.438 tok/s
- **4 chats:** 67.836 tok/s
- **8 chats:** 98.257 tok/s
- **8K first token:** 5.960 s
- **Receipt:** [fp8-2spark-on.json](../bench/receipts/2026-11-01/fp8-2spark-on.json)

**Two Sparks, drafting on, depth chosen by the server**

- **Prose:** 37.562 tok/s
- **Code:** 66.130 tok/s
- **Structured:** 80.845 tok/s
- **1 chat:** 49.928 tok/s
- **4 chats:** 68.266 tok/s
- **8 chats:** 98.457 tok/s
- **8K first token:** 5.959 s
- **Receipt:** [fp8-2spark-on-chooser.json](../bench/receipts/2026-11-01/fp8-2spark-on-chooser.json)

**Two Sparks, drafting off**

- **Prose:** 27.650 tok/s
- **Code:** 27.534 tok/s
- **Structured:** 27.721 tok/s
- **1 chat:** 26.374 tok/s
- **4 chats:** 67.944 tok/s
- **8 chats:** 98.645 tok/s
- **8K first token:** 5.953 s
- **Receipt:** [fp8-2spark-off.json](../bench/receipts/2026-11-01/fp8-2spark-off.json)

**Four Sparks, drafting on, depth 2**

- **Prose:** 51.362 tok/s
- **Code:** 72.301 tok/s
- **Structured:** 74.507 tok/s
- **1 chat:** 62.353 tok/s
- **2 chats:** 58.652 tok/s
- **4 chats:** 92.300 tok/s
- **8 chats:** 131.399 tok/s
- **8K first token:** 6.821 s
- **Receipt:** [fp8-4spark-on.json](../bench/receipts/2026-11-01/fp8-4spark-on.json)

**Four Sparks, drafting on, depth chosen by the server**

- **Prose:** 46.858 tok/s
- **Code:** 90.067 tok/s
- **Structured:** 110.847 tok/s
- **1 chat:** 66.792 tok/s
- **2 chats:** 58.401 tok/s
- **4 chats:** 91.582 tok/s
- **8 chats:** 132.127 tok/s
- **8K first token:** 6.813 s
- **Receipt:** [fp8-4spark-on-chooser.json](../bench/receipts/2026-11-01/fp8-4spark-on-chooser.json)

**Four Sparks, drafting off**

- **Prose:** 37.172 tok/s
- **Code:** 36.951 tok/s
- **Structured:** 36.667 tok/s
- **1 chat:** 35.833 tok/s
- **2 chats:** 56.584 tok/s
- **4 chats:** 89.246 tok/s
- **8 chats:** 131.591 tok/s
- **8K first token:** 6.821 s
- **Receipt:** [fp8-4spark-off.json](../bench/receipts/2026-11-01/fp8-4spark-off.json)

**How we ran these setups**

The receipts page is [bench/receipts/2026-11-01](../bench/receipts/2026-11-01/README.md). It gives the build and binary hashes, the checkpoint identity, the server starts and the request settings.

- **Machines.** The machines are DGX Spark, NVIDIA GB10, with the clocks not locked.
- **Builds.** Main `d913e84` has the same engine as main `1a07e3a`. Public commit `3f02077` has the tree of main `1a07e3a`.
- **Launcher.** Each setup had two server starts. Both used a private launcher with context 32768 and output limit 16384.
- **First start.** It passed drafting depth 2. It ran the drafting-on card, the drafting-off card and the 32K chat.
- **Second start.** It passed no drafting depth, so the server chose the depth. It ran one drafting-on card. `/health` reported depth 7 on both of these starts.

These are the public commands for the same setups. We did not run a card with them on this date.

```sh
vade up --fp8 --sparks 2 --context 32768 --output 16384
vade up --fp8 --sparks 4 --context 32768 --output 16384
```

How to read the lists:

- **Medians.** Each number is the card's own median. Prose, code and structured use 5 runs. The chat rates are C1, C2, C4 and C8 in the card, and they use 3 rounds. The 8K first token uses 5 cold requests on two Sparks and 3 on four Sparks. Every list is one server start.
- **Two-Spark cards.** The two-Spark cards leave out the C2 round and send five cold and five warm 8K prompts. So they hold 64 responses, not 66.
- **Requests.** The chat and 8K request settings are the same as in the [2026-10-31 cards](#current-fp4-local-main-dc32b7c-2026-10-31-morning). The chats are greedy, with seed 20260905, thinking off and `cache` false. The 8K token prompts send `"draft": "plain"`, a new cache salt for each cold request and a replay with the same salt. No cold request restored a cached token. Every replay restored all 8,192 tokens. The drafting-off card sends `"draft": "plain"` in its 60 chats. The drafting-on cards leave `draft` out.
- **Noise.** Starts of the same setup have differed by up to 2.81 % in one cell ([history](history/benchmarks-2026-10.md#speed)). So treat a smaller gap between two lists as noise.

**The 32K-token chat**

The chat client and the first question are the same as on 2026-10-31. The second question holds the first answer, so its length follows the token IDs. On both setups it is 32,238 tokens. The chat ran on the depth 2 start.

<div align="center">

| Setup | First question | Second question, cache | Second question, no cache |
| :---: | :---: | :---: | :---: |
| FP8 · 2 Sparks | 24.080 | 0.890 | 23.909 |
| FP8 · 4 Sparks | 27.376 | 0.781 | 27.357 |

</div>

The cached run reused 32,064 tokens on both setups. The second question gave the same token IDs with the cache and without it (`turn2_ids_equal`). The receipts are [two Sparks](../bench/receipts/2026-11-01/fp8-2spark-chat32k.json) and [four Sparks](../bench/receipts/2026-11-01/fp8-4spark-chat32k.json).

### FP8 token IDs, morning cards

- **FP8, four Sparks.** The drafting-on, drafting-off and chooser cards have the same `token_ids_sha256`, `2af8329ff3867b00165411156482d906c05cf6416bbabe2f3fe6bbe524e124a2`. This is also the value of the 2026-10-31 FP8 two-Spark cards. So these four-Spark cards gave the same token IDs as those two-Spark cards in all 66 responses.
- **FP8, two Sparks.** The drafting-on, drafting-off and chooser cards have the same value, `f1c1a5ff1b333158bbc9f9e93bd24bc3f980966852f9050607493710c750bada`. It covers 64 responses.
- **FP8, two against four Sparks on 2026-11-01.** We compared the full card files. Every response that both cards sent gave the same token IDs: the 15 single chats, the first three cold and warm 8K runs, and the C1, C4 and C8 streams.
- **The 32K chat.** Both setups gave the turn digests `28a6dca8...` and `b01b5a6b...`, the same as the 2026-10-31 FP8 two-Spark chat.
- **Re-run.** `vade bench --card --receipt` can check the four-Spark files. It cannot check the 64-response two-Spark files. We have not run it against these receipts.

## Current FP4-Local: main 8976c4f, 2026-11-02 morning

These FP4-Local rates and chat times on one, two and four Sparks are current. The [2026-10-31 cards](#current-fp4-local-main-dc32b7c-2026-10-31-morning) are history.

The receipts page is [bench/receipts/2026-11-02](../bench/receipts/2026-11-02/README.md). It gives the build and binary hashes, the checkpoint identity, the server start command of each setup and the request settings. The machines are DGX Spark, NVIDIA GB10, driver 580.178.04, with the clocks not locked. Main `8976c4f` has the engine of main `407aa46`. Between the two, only the command line tool changed.

**One chat, drafting on (tok/s)**

<div align="center">

| Setup | Prose | Code | Structured |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 42.123 | 81.263 | 101.055 |
| FP4-Local · 2 Sparks | 54.530 | 101.713 | 130.258 |
| FP4-Local · 4 Sparks | 61.881 | 112.551 | 144.282 |

</div>

**Chats at once, drafting on (total tok/s)**

<div align="center">

| Setup | C1 | C4 | C8 |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 65.965 | 145.338 | 179.802 |
| FP4-Local · 2 Sparks | 79.461 | 167.075 | 230.931 |
| FP4-Local · 4 Sparks | 87.485 | 176.042 | 243.521 |

</div>

**Path, prose with drafting off, and the cold 8K prompt**

<div align="center">

| Setup | Path | Prose, drafting off (tok/s) | Cold 8K (s) |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | local | 27.940 | 3.773 |
| FP4-Local · 2 Sparks | tensor split | 36.061 | 2.119 |
| FP4-Local · 4 Sparks | tensor split in parts | 41.845 | 1.933 |

</div>

The drafting-off cards come from the same server starts as the drafting-on cards. Each drafting-off receipt also holds every other cell of its card. These are the receipts:

- **FP4-Local, one Spark:** [drafting on](../bench/receipts/2026-11-02/fp4-local-1spark-on.json) and [drafting off](../bench/receipts/2026-11-02/fp4-local-1spark-off.json).
- **FP4-Local, two Sparks:** [drafting on](../bench/receipts/2026-11-02/fp4-local-2spark-on.json) and [drafting off](../bench/receipts/2026-11-02/fp4-local-2spark-off.json).
- **FP4-Local, four Sparks:** [drafting on](../bench/receipts/2026-11-02/fp4-local-4spark-on.json) and [drafting off](../bench/receipts/2026-11-02/fp4-local-4spark-off.json).

How to read the tables:

- **Medians.** Each cell is the card's own median. Prose, code and structured use 5 runs. C1, C4 and C8 use 3 rounds. Cold 8K uses 5 cold requests on one Spark and 3 on two and four Sparks. Every row is one server start.
- **Requests.** Prose, code and structured are one chat each, up to 8192 tokens, to the natural stop. C1, C4 and C8 are the total rate of 1, 4 or 8 chats at once, 256 tokens each. Cold 8K is the time to the first token of an uncached 8192-token prompt. Both cards send it with drafting off.
- **Chat settings.** The 60 chat requests of each card are greedy: temperature 0, top-p 1, top-k 0, neutral penalties, seed 20260905, thinking off and `cache` false.
- **8K settings.** The 8K requests send token prompts with temperature 0, top-p 1, top-k 0, neutral penalties and `"draft": "plain"`. They send no seed and no `cache` field. Each cold request has a new cache salt and a replay with the same salt. No cold request and no replay restored a cached token. The Cold 8K column uses the cold requests only.
- **Card shape.** The one-Spark cards send five cold and five warm 8K prompts, not three, and they leave out the C2 round. So they hold 64 responses, not 66.
- **Noise.** Each row has one server start. Starts of the same setup have differed by up to 2.81 % in one cell: code on FP8 on four Sparks gave 65.530 and 67.370 tok/s on 2026-10-30. So treat a smaller gap between two rows as noise ([history](history/benchmarks-2026-10.md#speed)).

### FP4-Local 32K-token chat

The chat sends one 32,099-token question. Then it sends a second question on the same chat (32,305 tokens). The second question ran twice with the cache and once with `cache` false. The table gives the time to the first token of the first question, of the first cached run and of the uncached run.

<div align="center">

| Setup | First question | Second question, cache | Second question, no cache |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 15.758 | 0.366 | 15.937 |
| FP4-Local · 2 Sparks | 10.139 | 0.767 | 9.894 |
| FP4-Local · 4 Sparks | 9.696 | 1.058 | 9.061 |

</div>

The cached run reused 32,064 tokens on every setup, four Sparks included. On every start, the second question gave the same token IDs with the cache and without it (`turn2_ids_equal` in each receipt). Each start ran the chat three times. The receipts hold the first chat.

These are the receipts:

- **FP4-Local, one Spark:** [chat](../bench/receipts/2026-11-02/fp4-local-1spark-chat32k.json).
- **FP4-Local, two Sparks:** [chat](../bench/receipts/2026-11-02/fp4-local-2spark-chat32k.json).
- **FP4-Local, four Sparks:** [chat](../bench/receipts/2026-11-02/fp4-local-4spark-chat32k.json).

### FP4-Local token IDs

- **One Spark, drafting on and off.** Both cards have the same `token_ids_sha256`. So every one of the 64 responses had the same token IDs. In a private lab comparison, both cards also matched the lab's one-Spark reference cards in 64 of 64 responses.
- **Two and four Sparks, drafting on and off.** All four cards have the same `token_ids_sha256`, `805f06d0...`. So every one of the 66 responses had the same token IDs. This is also the value of all eight FP4-Local cards of 2026-10-31.
- **One against four Sparks.** In a private lab comparison, the four-Spark cards matched the lab's one-Spark reference card in 66 of 66 responses. So four Sparks give the same token IDs as two and as one.

<a id="current-fp4-local-main-dc32b7c-2026-10-31-morning"></a>

## Earlier FP4-Local: main dc32b7c, 2026-10-31 morning

The [2026-11-02 FP4-Local cards](#current-fp4-local-main-8976c4f-2026-11-02-morning) replace these numbers. The FP8 cards of this date are [earlier cards](#earlier-fp8-cards-of-2026-10-31), and the [2026-11-01 FP8 cards](#current-fp8-main-acb49bc-2026-11-01-afternoon) replace them.

The receipts page is [bench/receipts/2026-10-31](../bench/receipts/2026-10-31/README.md). It gives the build and binary hashes, the checkpoint identities, the server start command of each setup and the request settings. The machines are DGX Spark, NVIDIA GB10, driver 580.178.04, with the clocks not locked.

**One chat, drafting on (tok/s)**

<div align="center">

| Setup | Prose | Code | Structured |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 42.530 | 81.138 | 100.808 |
| FP4-Local · 2 Sparks | 54.878 | 102.314 | 130.800 |
| FP4-Local · 4 Sparks | 63.036 | 113.025 | 145.238 |

</div>

**Chats at once, drafting on (total tok/s)**

<div align="center">

| Setup | C1 | C4 | C8 |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 65.688 | 144.681 | 178.981 |
| FP4-Local · 2 Sparks | 79.632 | 166.796 | 232.198 |
| FP4-Local · 4 Sparks | 86.962 | 177.581 | 245.145 |

</div>

**Path, prose with drafting off, and the cold 8K prompt**

<div align="center">

| Setup | Path | Prose, drafting off (tok/s) | Cold 8K (s) |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | local | 27.911 | 3.769 |
| FP4-Local · 2 Sparks | tensor split | 36.047 | 2.153 |
| FP4-Local · 4 Sparks | tensor split in parts | 42.084 | 1.924 |

</div>

The drafting-off cards come from the same server starts as the drafting-on cards. Each drafting-off receipt also holds every other cell of its card. These are the receipts:

- **FP4-Local, one Spark, drafting on:** [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-on.json) and [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-on.json).
- **FP4-Local, one Spark, drafting off:** [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-off.json) and [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-off.json). The prose value 27.911 is the mean of the two cards.
- **FP4-Local, two Sparks:** [drafting on](../bench/receipts/2026-10-31/fp4-local-2spark-on.json) and [drafting off](../bench/receipts/2026-10-31/fp4-local-2spark-off.json).
- **FP4-Local, four Sparks:** [drafting on](../bench/receipts/2026-10-31/fp4-local-4spark-on.json) and [drafting off](../bench/receipts/2026-10-31/fp4-local-4spark-off.json).

How to read the tables:

- **Medians.** Each cell is the card's own median. Prose, code and structured use 5 runs. C1, C4 and C8 use 3 rounds. Cold 8K uses 3 cold requests. The one-Spark row is the mean of the medians of its two starts. Every other row is one server start.
- **Requests.** Prose, code and structured are one chat each, up to 8192 tokens, to the natural stop. C1, C4 and C8 are the total rate of 1, 4 or 8 chats at once, 256 tokens each. Cold 8K is the time to the first token of an uncached 8192-token prompt. Both cards send it with drafting off.
- **Chat settings.** The 60 chat requests of each card are greedy: temperature 0, top-p 1, top-k 0, neutral penalties, seed 20260905, thinking off and `cache` false.
- **8K settings.** The six 8K requests send token prompts with temperature 0, top-p 1, top-k 0, neutral penalties and `"draft": "plain"`. They send no seed and no `cache` field. Three are cold requests, each with a new cache salt. Each cold request has a replay with the same salt. No cold request restored a cached token. The replays restored all 8,192 tokens on FP8 and none on FP4-Local, which keeps a prompt only on request. The Cold 8K column uses the cold requests only.
- **Noise.** Most rows have one server start. Starts of the same setup have differed by up to 2.81 % in one cell: code on FP8 on four Sparks gave 65.530 and 67.370 tok/s on 2026-10-30. So treat a smaller gap between two rows as noise ([history](history/benchmarks-2026-10.md#speed)).

### The 32K-token chat

The chat sends one 32,099-token question. Then it sends a second question on the same chat (32,305 tokens; 32,238 with FP8 on two Sparks). The second question ran twice with the cache and once with `cache` false. The table gives the time to the first token of the first question, of the first cached run and of the uncached run.

<div align="center">

| Setup | First question | Second question, cache | Second question, no cache |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark (mean of 2 starts) | 15.628 | 0.357 | 15.791 |
| FP4-Local · 2 Sparks | 10.092 | 0.758 | 9.947 |
| FP4-Local · 4 Sparks | 8.984 | 8.725 | 8.745 |

</div>

The cached run reused 32,064 tokens on every setup except four FP4-Local Sparks, which reused 0. Those runs predate the prefix cache on the split in parts, so their second question read the whole prompt again. On every start of 2026-10-31, FP8 included, the second question gave the same token IDs with the cache and without it (`turn2_ids_equal` in each receipt).

These are the receipts:

- **FP4-Local, one Spark:** [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-chat32k.json) and [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-chat32k.json).
- **FP4-Local, two Sparks:** [chat](../bench/receipts/2026-10-31/fp4-local-2spark-chat32k.json).
- **FP4-Local, four Sparks:** [chat](../bench/receipts/2026-10-31/fp4-local-4spark-chat32k.json).

### Earlier FP8 cards of 2026-10-31

The [2026-11-01 FP8 cards](#current-fp8-main-acb49bc-2026-11-01-afternoon) replace these numbers. They stay here because the [token ID checks](#token-ids) below use them. The path of both setups is tensor parallel. The 32K chat lines give the seconds to the first token.

**FP8 on two Sparks**

- **Prose:** 36.624 tok/s
- **Code:** 51.293 tok/s
- **Structured:** 52.763 tok/s
- **Prose, drafting off:** 27.072 tok/s
- **1 chat:** 45.436 tok/s
- **4 chats:** 66.578 tok/s
- **8 chats:** 98.670 tok/s
- **8K first token:** 5.971 s
- **32K chat, first question:** 24.116 s
- **32K chat, second question, cache:** 0.938 s
- **32K chat, second question, no cache:** 23.908 s
- **Receipts:** [drafting on](../bench/receipts/2026-10-31/fp8-2spark-on.json), [drafting off](../bench/receipts/2026-10-31/fp8-2spark-off.json) and [chat](../bench/receipts/2026-10-31/fp8-2spark-chat32k.json)

**FP8 on four Sparks**

- **Prose:** 49.508 tok/s
- **Code:** 69.340 tok/s
- **Structured:** 72.248 tok/s
- **Prose, drafting off:** 35.902 tok/s
- **1 chat:** 61.435 tok/s
- **4 chats:** 88.017 tok/s
- **8 chats:** 130.999 tok/s
- **8K first token:** 6.923 s
- **32K chat, first question:** 27.807 s
- **32K chat, second question, cache:** 1.059 s
- **32K chat, second question, no cache:** 28.475 s
- **Receipts:** [drafting on](../bench/receipts/2026-10-31/fp8-4spark-on.json), [drafting off](../bench/receipts/2026-10-31/fp8-4spark-off.json) and [chat](../bench/receipts/2026-10-31/fp8-4spark-chat32k.json)

### Token IDs

- **FP4-Local, one, two and four Sparks, drafting on and off.** All eight cards have the same `token_ids_sha256`. So every one of the 66 responses had the same token IDs.
- **FP8, two Sparks.** The drafting-on and drafting-off cards have the same value.
- **FP8, four Sparks.** The drafting-on and drafting-off cards have the same value.
- **FP8, two against four Sparks.** The values differ. Compared response by response, 7 of 66 are the same and 59 differ. This count is from a private lab run on 2026-10-31 (not in this repository). It broke the rule that supported Spark counts give the same token IDs. A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order, so four Sparks now give the same token IDs as two ([changelog](../CHANGELOG.md)). These cards predate that change. The [2026-11-01 FP8 cards](#fp8-token-ids) were made after it.

### What a stranger can and cannot re-run

- **Build the measured tree and start one or two Sparks as in the cards: yes.** Check out public commit `db6b7fc` (`db6b7fcd1f0ed97d6d21084eab1d206cc0d5ef3c`). Its tree `aea2c506` is the tree of main `dc32b7c`. Then use the commands in the receipts page.
- **Start four Sparks or FP8 as in the cards: not with the same launcher.** The cards used a private launcher. `vade up --sparks 4` and `vade up --fp8` start the same path, but we have not run a card on them on this date.
- **Send the 66 card requests: yes.** Clone [RigMark](https://github.com/alexellis/rigmark) and check out revision `52c29c3`. Then run `vade bench --card --prompts rigmark/prompts.json`. It sends the card's 66 requests in the card's order with the card's request bodies, and it saves every token ID in `vade-card.json`. Add `--draft plain` for a drafting-off card. Plain RigMark sends a different cold 8K request ([details](benchmarks.md#run-rigmark-against-vade)).
- **Compare token IDs with these receipts: yes.** Add `--receipt bench/receipts/2026-10-31/<file>.json`. The client prints `ids 66/66 identical` when its `token_ids_sha256` equals the receipt's. On 2026-10-31 it printed that line against `fp4-local-1spark-start1-on.json` (drafting on) and `fp4-local-1spark-start1-off.json` (`--draft plain`). That run used one rented GB10 machine that served main `acf5264` with the settings of `vade up --sparks 1 --context 32768 --output 8192`. We have not run it against the two-Spark, four-Spark or FP8 receipts.
- **Run the 32K chat: no.** The client and its prompt are private.
- **Check drafting on against off on your own server: yes.** Run `vade verify`.
- **Time your own server: yes.** Run `vade bench`. It uses other prompts, so its numbers do not compare with the cards.

## Install time

The receipt is [fresh-install-2026-10-30.txt](../bench/receipts/2026-10-31/fresh-install-2026-10-30.txt).

The run was on 2026-10-30, on one DGX Spark with no Rust and no Vade files and with the model already on disk. The steps used a clone of the public repository at `82ec816`, which has the tree of main `0213a82`.

1. The clone took 0.69 s from a local bundle.
2. `sh scripts/install.sh` took 176.8 s.
3. The first `vade up` took 67.4 s.

## GLM 5.3 Flash, two Sparks, rc.24

GLM is a diagnostic text service, not qualified. These numbers come from the
[rc.24 release](https://github.com/wappy-x/Vade/releases/tag/v0.2.0-rc.24).
One pair of Sparks ran four starts in the order rc.23, rc.24, rc.24, rc.23.
Each cell is the mean of two starts. The cards are private lab runs (not in
this repository).

| Load | rc.23 (tok/s) | rc.24 (tok/s) |
| --- | ---: | ---: |
| One chat, drafting on, prose | 27.15 | 29.22 |
| One chat, drafting on, code | 33.12 | 35.82 |
| One chat, drafting on, structured | 43.59 | 46.23 |
| Default request (one chat) | 36.89 | 39.29 |
| Two plain chats (total) | 33.81 | 34.88 |
| Four plain chats (total) | 49.94 | 51.50 |
| Eight plain chats (total) | 64.34 | 65.55 |

- **One chat, drafting off.** Prose 24.246 / 24.258 / 24.128 / 24.254 tok/s,
  code 24.297 / 24.251 / 24.225 / 24.253, structured 24.278 / 24.141 /
  24.191 / 24.229 (order rc.23, rc.24, rc.24, rc.23).
- **8K first token.** rc.23: 11.767 s and 10.787 s. rc.24: 10.815 s and
  10.728 s.
- **Short-prompt first token.** 497.93 ms on rc.23, 499.05 ms on rc.24 (mean).
- **Token IDs.** All 69 compared answer lists and 16,800 token IDs matched.

<a id="inkling-2-mini-one-spark-rc26"></a>

## Inkling 2 Mini, one Spark, live main and rc.26

Inkling 2 Mini is a text-only preview, not qualified. All Inkling runs below are private lab runs (not committed under `bench/receipts/`); the live `main` rows are also published at `https://vade.lol/results/`.

### Released candidates (rc.25 and rc.26)

Source: the [rc.26 release](https://github.com/wappy-x/Vade/releases/tag/v0.2.0-rc.26). Two runs each on one Spark.

<div align="center">

| Measure | rc.25 engine | rc.26 engine |
| :--- | ---: | ---: |
| First token after a cold 8,192-token prompt | 96.7 s | 50.6 s |

</div>

### Live on main (measured build 315ae0521a, 2026-11-26)

Current `main` carries the GPU kernel code of the measured build `315ae0521a` byte for byte. It adds four bit-exact GPU kernel changes atop `b8d0625e1f`: (1) branch-free Vade-BF11-BP escape decode in the `plain_tc_rows` panel paths (three stream-word loads and byte permutes per tile, no loop and no branch; `69fcc1270b`, `ec466a0317`; `plain_tc_rows` stays at 96/100 regs); (2) the fast kernels fetch their launch command through L1 and prefetch the next four commands into L2 (`fast_mix_blend`, `fast_norm`, `fast_row_dots`, `fast_router_pick`, `fast_capped_gate`, `moe_group_route`; `f82ea9f9b1`, `517a80f454`; `fast_mix_blend` 64/68 regs); (3) 9- to 32-input draft panels on small matrices split into 8-input warps, and draft attention prefetches K and V into L1 ahead of the key walk (`8cf0008be5`, `d1ad83f3f6`, `7652b750f6`; `inkling_draft_panel` 76/76 regs, `inkling_draft_lanes` 62/64 regs); and (4) the `prefill_flash_window` relative-bias loop reads four terms per `ld.shared.v4` load (`e2ca7abfce`, `621753439f`; 168/168 regs, 101,184 B SMEM). Ten Inkling PTX functions changed; the other 35, the tensor-core PTX and all 4 shared Qwen/GLM PTX files are unchanged. Beneath them, `b8d0625e1f` added three bit-exact host-side changes atop `00a6754ce9`: Mode 0 dense `O_proj` at `M=9..16` uses the 4-warp cooperative-W tile (`a1fc00a452`, `2412d7d7e8`; one PTX function, `plain_tc_rows`, reads the new predicate; register table unchanged), the resident launch recorder updates a same-kernel CUDA graph in place when the retained graph set is full (`e623b76ba6`, `a233cc6349`, `714494b64d`), and `DEVICE_ROWS` is back to 16 so 9- to 16-row decode steps keep the row form of the expert products (`b8d0625e1f`). Compared with `rc.26`, live `main` cuts resident weights on GPU from 67.85 GB to **20.69 GB** (`20,693,571,840` bytes: 18.93 GB of packed weights and model parameters from commits `a8f8f10d68` to `488adccbc6`, plus 1.77 GB (`1,765,693,696` bytes, 4,855,424 tiles, 0 outliers, `-139.8 MB` vs `c975fa3203`) of lossless Vade-BF11-BP dense panels for tensor-core products and 16-row K/V verification), raises the configured context cap to **1,048,576** tokens (`b8ea3cc310`, `53cc85d989`), keeps the prompt cache on by default with DMA cached-state restore, enables explicit `vade up --mtp 0..=7` up to the checkpoint's declared draft depth (`b26499d4cc`, default `--mtp 0`), and stacks five bit-exact upgrades atop `3e8c3a8cc2`: `C2..C32` batched multi-chat drafting (`DRAFT_CHAT_LIMIT=32`, `VERIFY_ROW_CAP=256`), unwrapped multi-chat KV verify write, zero-copy `ShortConvMulti` rings, fused multi-chat prefill KV/conv, `FAST_STREAM_OPS` `SoftcapArgmax` (`STREAM_OP_SOFTCAP_ARGMAX=7`), and eviction-free cohort policy (`e8672e62f0`, `a5bbab6367`); two-stage bit-exact `PREFILL_FLASH_WINDOW` (`548e3be650`, 35th Inkling kernel, `167/168` regs, `101,184 B` SMEM, `1 blk/SM`) and fused `BM=8` 2-blk/SM `PREFILL_GATE_FUSE` (`8e49ce6081`, 36th Inkling kernel, `128/128` regs, `47,424 B` SMEM, `2 blk/SM`) with lazy message metadata (`85fd72bf57`); Vade-BF11-BP dense panels extended to `M=7..8` verify (`2425759a23`, `73a3cd7737`), L2-streaming Vade-BF11-BP `plain_tc_rows` (`96/100` regs, `0` spills, `0 B` extra VRAM), and serpentine `plain_tc_big` (`122/124` regs, `a10ca15c40`, `4025f1bf93`, `c0f27dfcfe`); and direct `KvFinishDirect` (`NORM_MODE_KV_DIRECT=3`), `ShortConvDirect` (`NORM_MODE_SHORT_CONV_DIRECT=4`), and `flash_window_split` high-row guards (`37254e373c`, `03d95c7c1e`, `00a6754ce9`). All earlier Wave 2E, 2D, 2C, 2B, 2A and Wave 1 levers remain stacked beneath them (register-pipelined MXFP8 + zero-pad XOR NVFP4 MoE + 576t `MIX_ONE` and `GateBlendNorm` + 36x64 `RING_CONV_ONE` + 4-edge MoE/Mix PDL `3e8c3a8cc2`; M=2..=16 single-pass strided MoE verify `plain_packed_rows` Mode A + `P_DEPTH=12` wide BF16 panel ring & `FEW_GATE_BLEND` `a82cfaf49a`; 4-warp cooperative Vade-BF11-BP panels `2e6e7097ed`; dynamic-SMEM `plain_tc_rows` `29e72bb7f9`; `prefill_packed_tc` `3200285397`; multi-chat batched drafting `6d174bcf61`; packed-BF16 prefill pipeline `3224b43b53`; wide 2-tile `d=8..15` drafting `585a352603`; deferred stream-op payload loads `c975fa3203`; wide-decode zero-sync grouped packed TC MoE `105f298913`; 32-row small-pass / `MIX_ONE` / cohort attention `ecbc8a2b45`; single-launch LM head `258b4ef158`; and linear cohort admission + bounded draft KV pool `813689a0f1`).

One Spark ran direct `flash-serve --diagnostic-unqualified-text --mtp 7` (drafted) and `--mtp 0` (plain) with greedy decoding and reasoning `effort 0` from a clean archive of `315ae0521a` (`dirty=false`), producing two fresh full RigMark cards (`--runs 5 --decode-tokens 8192 --prefill-depths 8192 --concurrency 8`), published at `vade.lol/results` as `ink-20261126-0432-315ae0521-main-auto` (receipt digests `c1a97bd12877b637` and `cb07ce685c0effc3`) and `ink-20261126-0441-315ae0521-main-plain` (`b8a1460ee64ad670` and `b12ab530d29767a4`). The `C2`, `C4`, `C16`, `C32` and cold 32K rows below come from matched runs on a clean archive of `621753439f` (`dirty=false`), the tested parent of `315ae0521a` with the same `engine/` tree (`2db7901d8d`), on the same Spark as the cards and as `00a6754ce9`. Through `vade up`, Inkling defaults to plain decode (`--mtp 0`) with prompt caching on and accepts explicit `--mtp 0..=7`.

<div align="center">

| Load | Live `main` | Measured build |
| :--- | ---: | :---: |
| One chat, adaptive drafting (`--mtp 7`), prose | 51.0 tok/s (50.97; range 48.69–51.59) | `315ae0521a` |
| One chat, adaptive drafting (`--mtp 7`), code | 91.9 tok/s (91.88; range 86.48–95.21) | `315ae0521a` |
| One chat, adaptive drafting (`--mtp 7`), structured | 142.4 tok/s (142.42; range 135.01–145.67) | `315ae0521a` |
| One chat, plain decode, prose | 43.5 tok/s (43.50; range 43.47–43.64) | `315ae0521a` |
| One chat, plain decode, code | 42.7 tok/s (42.66; range 42.56–42.73) | `315ae0521a` |
| One chat, plain decode, structured | 43.8 tok/s (43.78; range 43.75–43.78) | `315ae0521a` |
| Two chats at once (`C2`, 256-tok effort-0 code, plain / auto) | 66.1 / 99.8 tok/s (peak 73.4 / 123.5; `1.51x`) | `621753439f` |
| Four chats at once (`C4`, 256-tok effort-0 code, plain / auto) | 109.8 / 136.9 tok/s (peak 117.9 / 142.0; `1.25x`) | `621753439f` |
| Eight chats at once, drafting requested (`C8`, total) | 197.7 tok/s (197.70; peak 222.51; per-stream 29.29; TTFT 0.552 s) | `315ae0521a` |
| Eight plain chats at once (`C8`, total) | 185.2 tok/s (185.22; peak 187.52; per-stream 24.18; TTFT 0.509 s) | `315ae0521a` |
| Sixteen chats at once (`C16`, 256-tok effort-0 code, plain / auto) | 264.5 / 318.4 tok/s (peak 274.5 / 319.8; per-stream 17.45 / 22.08; `1.20x`) | `621753439f` |
| Thirty-two chats at once (`C32`, 256-tok effort-0 code, plain / auto) | 337.0 / 415.1 tok/s (peak 341.2 / 421.9; per-stream 11.50 / 15.90; `1.23x`) | `621753439f` |
| First token, cold 8,192-token prompt (drafted / plain card) | 1.94 s (1.935 s; 4,234.3 tok/s) / 1.92 s (1.920 s; 4,266.4 tok/s) | `315ae0521a` |
| First token, cached 8,192-token replay (drafted / plain card) | 0.007 s / 0.007 s (8,192 of 8,192 tokens restored) | `315ae0521a` |
| First token, cold 32,768-token prompt (`ctx 65536`, 3 runs) | 8.31 s (8.308 s; 3,944.2 tok/s; peak 8.292 s / 3,951.5 tok/s; warm replay 0.016 s) | `621753439f` |

</div>

- **Checks.** On `315ae0521a`, all 15 RigMark single-chat output checks passed on both cards (prose 5/5, code 5/5, structured 5/5). On `621753439f` (same `engine/` tree), the device checks `tc-rows-check` (35,300,944 words), `few-check`, `staged-check` and `loop-pack-check --sweep-only` passed, the GPU crate unit tests (674) and the resident eviction test passed, and 8 of 8 `compute-sanitizer` runs (`memcheck` + `synccheck` on `few-check`, `staged-check`, `tc-rows-check`, `loop-pack-check --sweep-only`) reported 0 errors. All 36 Inkling plus 6 tensor-core kernels built with 0 register spills; only three register counts moved, each inside its cap with the same blocks per SM (`inkling_draft_panel` 74 to 76/76, `prefill_flash_window` 167 to 168/168, `fast_mix_blend` 56 to 64/68). Other key kernels: `prefill_gate_fuse` 128/128 regs / 47,424 B smem / 2 blk/SM; `plain_tc_rows` 96/100 regs / 0 B static dynamic SMEM / 2 blk/SM; `plain_tc_big` 122/124 regs / 49,152 B smem / 2 blk/SM; `fast_norm` 111/120 regs / 16,672 B smem / 4 blk/SM; `plain_packed_picked` 130/132 regs / 18,496 B smem / 3 blk/SM; `fast_stream_ops` 78/78 regs / 512 B smem / 3 blk/SM; `fast_router_pick` 50/51 regs / 1,152 B smem / 5 blk/SM; `fast_row_dots` 78/80 regs / 6 blk/SM; `moe_group_route` 40/44 regs / 77,824 B smem / 1 blk/SM; `inkling_draft_lanes` 62/64 regs / 64 B smem / 8 blk/SM; `inkling_draft_panel` 76/76 regs / 0 B smem / 6 blk/SM. All 4 shared Qwen/GLM PTX outputs remained byte-identical to `9dcaae4f29`.
- **Memory.** `315ae0521a` reported 20.69 GB of resident weights (`20,693,571,840` bytes, including `1,765,693,696` bytes of lossless Vade-BF11-BP dense panels across `4,855,424` tiles with 0 outliers, the same as `00a6754ce9` and `b8d0625e1f`) and admitted 37 concurrent chats at a 32,768-token context with `--mtp 7` on the drafted-card start and the matched-run start (38 on the served-check start of `621753439f`), and 44 chats plain. The `--mtp 7` count follows the free memory of the Spark at start.
- **Token IDs.** `main` (`315ae0521a`) stays bit-exact with the owner-approved self-exact Inkling baseline `e38ae8201a`, with `00a6754ce9` and with `b8d0625e1f`. On `315ae0521a`, cross-card plain vs drafted replies (and both cards against `b8d0625e1f`, `00a6754ce9` and `e38ae8201a`) matched identically on 15/15 decode streams (36,452 completion tokens, 0 mismatches), 6/6 prefill streams and 24/24 C8 streams (6,192 tokens, 0 mismatches). On `621753439f` (same `engine/` tree), served multi-width exactness matched `e38ae8201a`, `00a6754ce9` and `b8d0625e1f` on 160/160 streams (`exact32`, 5,120 token IDs, solo median 44.131 tok/s against 43.633 tok/s on `00a6754ce9` on the same Spark) plus 31/31 C1, C2, C4, C8 and C16 drafted cohorts (`draft-widths`, 992 token IDs); matched effort-0 code concurrency (`C2`–`C32`, 3 rounds each) matched plain-vs-auto on 186/186 streams (47,616 tokens) and matched `00a6754ce9` on 186/186 streams in each mode; and 6/6 cold/warm 8K and 32K prefill streams matched `00a6754ce9` and `78bb835132` (`c0d4ef05cf`).
- **Earlier mains.** Build `b8d0625e1f` (2026-11-26, `O_proj` 4-warp tile at 9–16 rows, in-place CUDA graph updates, `DEVICE_ROWS = 16`) measured 49.42 / 88.01 / 136.09 tok/s adaptive, 41.86 / 41.40 / 42.43 tok/s plain, 199.90 tok/s drafted C8 (181.14 tok/s plain, TTFT 0.554 s / 0.509 s), 1.985 s / 1.905 s cold 8K, and 0.007 s warm 8K replay (`ink-20261126-0339-b8d0625e1-main-auto`, `ink-20261126-0349-b8d0625e1-main-plain`, on a different Spark than `315ae0521a` and `00a6754ce9`). Build `00a6754ce9` (2026-11-26, `C2..C32` multi-chat drafting, fused prefill flash/gate, Vade-BF11-BP `M=7..8` and `plain_tc_rows`, direct KV/ShortConv decode) measured 50.44 / 87.72 / 129.49 tok/s adaptive, 42.90 / 42.04 / 43.15 tok/s plain, 200.32 tok/s drafted C8 (183.38 tok/s plain, TTFT 0.548 s / 0.505 s), 65.0 / 92.9 tok/s `C2`, 108.8 / 126.6 tok/s `C4`, 228.0 / 303.0 tok/s `C16`, 330.8 / 411.5 tok/s `C32`, 1.928 s / 1.923 s cold 8K, 0.007 s warm 8K replay, and 8.310 s cold 32K (`ink-20261126-0034-00a6754ce-main-auto`, `ink-20261126-0043-00a6754ce-main-plain`, on the same Spark as `315ae0521a`). Build `3e8c3a8cc2` (2026-11-25, register-pipelined MXFP8 + zero-pad XOR NVFP4 MoE + 576t `MIX_ONE` & `GateBlendNorm` + 36x64 `RING_CONV_ONE` + 4-edge MoE/Mix PDL) measured 50.02 / 87.14 / 133.21 tok/s adaptive, 42.79 / 41.94 / 43.03 tok/s plain, 172.90 tok/s drafted C8 (161.00 tok/s plain, TTFT 0.935 s / 0.879 s), 81.09 tok/s drafted `C2`, 122.02 tok/s drafted `C4`, 223.6 / 225.5 tok/s `C16`, 304.3 / 303.2 tok/s `C32`, 1.980 s / 1.980 s cold 8K, 0.007 s warm 8K replay, and 8.936 s cold 32K (`ink-20261125-2316-3e8c3a8cc-main-auto`, `ink-20261125-2325-3e8c3a8cc-main-plain`). Build `a82cfaf49a` (2026-11-25, M=2..=16 strided MoE verify + `P_DEPTH=12` wide BF16 panel ring) measured 49.43 / 87.20 / 132.47 tok/s adaptive, 41.46 / 40.66 / 41.70 tok/s plain, 169.79 tok/s drafted C8 (159.82 tok/s plain), 82.26 tok/s drafted `C2`, 126.66 tok/s drafted `C4`, 223.1 / 223.2 tok/s `C16`, 305.8 / 304.2 tok/s `C32`, 2.010 s / 1.994 s cold 8K, 0.007 s warm 8K replay, and 8.797 s cold 32K (`ink-20261125-2125-a82cfaf49-main-auto`, `ink-20261125-2135-a82cfaf49-main-plain`). Build `78bb835132` (2026-11-25, lossless Vade-BF11-BP panels, dynamic-SMEM dense TC, BF16 TC prefill MoE) measured 49.22 / 86.99 / 130.54 tok/s adaptive, 41.08 / 40.39 / 41.39 tok/s plain, 164.20 tok/s drafted C8 (160.27 tok/s plain), 217.5 / 217.6 tok/s C16 and 321.1 / 318.7 tok/s C32 (plain / auto), 1.972 s / 1.980 s cold 8K, 0.007 s warm 8K replay, and 8.680 s cold 32K (`ink-20261125-2107-78bb83513-main-auto`, `ink-20261125-2051-78bb83513-main-plain`). Build `6d174bcf61` (2026-11-25, multi-chat batched drafting) measured 47.11 / 83.36 / 127.63 tok/s adaptive, 37.34 / 36.70 / 37.52 tok/s plain, 160.73 tok/s drafted C8 (161.42 tok/s plain), 79.37 tok/s drafted `C2` (`1.39x` vs 57.26 plain) and 113.87 tok/s drafted `C4` (`1.19x` vs 95.33 plain), 2.026 s / 2.043 s cold 8K, 0.007 s warm 8K replay, and 8.914 s cold 32K (`ink-20261125-2030-6d174bcf6-main-auto`, `ink-20261125-2039-6d174bcf6-main-plain`). Build `c975fa3203` (2026-11-25) measured 47.05 / 82.57 / 128.99 tok/s adaptive, 37.31 / 36.71 / 37.48 tok/s plain, 161.09 tok/s plain C8 (160.32 tok/s drafted card), 218.3 / 217.7 tok/s C16, 279.0 / 279.3 tok/s C32, 2.040 s / 2.116 s cold 8K, 0.007 s warm 8K replay, and 8.959 s cold 32K (`ink-20261125-1953-c975fa320-main-auto`, `ink-20261125-1935-c975fa320-main-plain`). Build `f54f31841d` (2026-11-25) measured 47.61 / 84.17 / 118.26 tok/s adaptive, 37.79 / 37.11 / 37.97 tok/s plain, 162.34 tok/s plain C8 (161.08 tok/s drafted card), 218.5 / 218.6 tok/s C16, 274.5 / 271.9 tok/s C32, 2.623 s / 2.628 s cold 8K, 0.008 s / 0.007 s warm 8K replay, and 11.646 s cold 32K (`ink-20261125-1841-f54f31841-main-auto`, `ink-20261125-1824-f54f31841-main-plain`, 36 chats at 32K `--mtp 7`). Build `aed09f3c2d` (2026-11-25) measured 47.54 / 84.26 / 118.59 tok/s adaptive, 37.64 / 37.01 / 37.90 tok/s plain, 158.79 tok/s plain C8 (150.57 tok/s drafted card), 2.642 s / 2.637 s cold 8K, and 0.007 s warm 8K replay (`ink-20261125-1742-aed09f3c2-main-auto`, `ink-20261125-1724-aed09f3c2-main-plain`, 25 chats at 32K `--mtp 7`). Build `e38ae8201a` (2026-11-25, owner-approved split-KV tensor-core self-exact baseline) measured 46.71 / 79.79 / 110.91 tok/s adaptive, 36.95 / 36.34 / 37.10 tok/s plain, 157.51 tok/s plain C8, 2.639 s / 2.647 s cold 8K, and 0.023 s warm 8K replay (`ink-20261125-1328-e38ae8201-main-auto`, `ink-20261125-1310-e38ae8201-main-plain`). Build `b17bbd7ab1` (2026-11-25) measured 40.70 / 65.46 / 99.74 tok/s with adaptive drafting (matching earlier tensor-core reference `9dcaae4f29` across 35,154 drafted token IDs); parent build `58de812ac6` measured 34.2 / 33.1 / 34.8 tok/s plain, 140.4 tok/s C8, 2.74 s cold 8K, and 12.19 s cold 32K. Earlier scalar build `9704d39cb9` (2026-11-24) measured 26.5 / 36.1 / 51.4 tok/s drafted, 72.3 tok/s C8, and 4.52 s cold 8K (`vade.lol/results` row `ink-20261124-1434-9704d39cb-effort-0-adaptive-mtp`).

## Unmerged private-branch receipts (not on main)

The rows below come from verified private development branches. **None of these revisions is on `main`.** `main` (`315ae0521a`) has part of the changes of `ca1c5ccf95` and `35a7f358ae` in another form, and its own measured numbers are in the section above. Do not expect the numbers below from `main` or from `vade.lol/install.sh` today. All runs used Inkling 2 Mini on one DGX Spark with direct `flash-serve --diagnostic-unqualified-text --mtp 7`, greedy decoding and reasoning `effort 0`.

<div align="center">

| Private revision | Date | Measured highlights | Exactness and status |
| :--- | :---: | :--- | :--- |
| `ca1c5ccf95` | 2026-11-25 | Adaptive prose 36.0, code 59.6, structured 90.6 tok/s; cold 8K 4.20 s (1,949.5 tok/s read). The draft model runs its projections as NVFP4 in memory. Published as `vade.lol/results` short row `ink-20261125-0635-ca1c5ccf9-nvfp4-draft-effort-0` (digests `e68232ad527cf656`, `77bd8696848346d5`). | 15/15 output checks; all 104,340 compared token IDs matched `9dcaae4f29` (card, width-matrix, fixed-depth, cold and eight-chat streams); this revision is not on `main`. |
| `35a7f358ae` | 2026-11-25 | Adaptive prose 31.76, code 50.90, structured 81.25 tok/s; plain prose 31.62, code 30.92, structured 31.96 tok/s; C8 132.02 tok/s (private decode-stack qualification log, one Spark). | 15/15 output checks; 35,154 decode, 48 prefill, 6,144 concurrency and 8,896 width-matrix token IDs matched `9dcaae4f29`; 0 memcheck and 0 synccheck errors; this revision is not on `main`. |
| `490d479e7b` | 2026-11-25 | Matched 3-start pairs: cold 8K median 2.85 s (2.847 s vs base 2.982 s; 2.862 s vs `9dcaae4f29` 4.131 s); cold 32K median 12.15 s (12.146 s vs `9dcaae4f29` 17.189 s) (private prefill qualification log, one Spark). | 140/140 ID matrix, 36/36 solo (9,216 tokens) and 30/30 cold/replay token IDs matched `9dcaae4f29`; 0 cached tokens on every cold run; 0 memcheck and 0 synccheck errors; this revision is not on `main`. |

</div>

<a id="other-numbers-the-readme-quotes"></a>

## Earlier numbers on other pages

These numbers come from earlier runs. Their records are private and not in this repository, except where a line links a record.

- **One Spark at `--context 262144`.** A 64K prompt gave its first token in 32.97 s, and a 128K prompt in 71.53 s (means of two runs). The run was on 2026-10-30 at main `380d8e5`. The record is a private lab run on 2026-10-30 (not in this repository).
- **Two Sparks at `--context 262144`.** A 201,654-token prompt gave its first token in 68.02 s and in 68.10 s. The run was on 2026-10-28 at main `285fa8a`. The record is a private lab run on 2026-10-28 (not in this repository).
- **Two FP4-Local pairs at once.** Eight chats on each pair gave 465.2 tok/s in total (means of two starts). The run was on 2026-10-30 at main `51dce02` (engine `380d8e5`). The record is a private lab run on 2026-10-30 (not in this repository). The [two pairs at once](history/benchmarks-2026-10.md#two-pairs-at-once) section has the table.
- **Bundle steps to `Launch ready`.** The steps took 101.262 s. The run was on 2026-10-28 with the v0.1.0 bundle (`5a22651`). The record is `2026-10-28_1305_bundle-install_5a22651`.
- **v0.1.0 on one Spark, drafting on.** Code ran at 76.7 tok/s and C8 at 173.9 tok/s. The run was on 2026-10-24 at `5a22651`. The record is in this repository: [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md).
- **Rented cloud GB10 VMs.** They downloaded the checkpoint at 8 to 22 MB/s on 2026-10-24 ([rented cloud GB10 VMs](benchmarks.md#rented-cloud-gb10-vms)).

The [benchmark history](history/benchmarks-2026-10.md) keeps every earlier card, newest first.
