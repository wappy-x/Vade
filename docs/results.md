# Results and receipts

This page gives the current benchmark numbers of Vade. The FP8 numbers come first, and the FP4-Local numbers follow. FP4-Local and FP8 are the two formats of the Qwen3.8 Flash-Next model that Vade runs.

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

## Current FP4-Local: main dc32b7c, 2026-10-31 morning

These FP4-Local numbers are current. The FP8 cards of this date are [earlier cards](#earlier-fp8-cards-of-2026-10-31), and the [2026-11-01 FP8 cards](#current-fp8-main-acb49bc-2026-11-01-afternoon) replace them.

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

The cached run reused 32,064 tokens on every setup except four FP4-Local Sparks, which reused 0. Four FP4-Local Sparks keep no prefix cache, so their second question reads the whole prompt again. On every start of 2026-10-31, FP8 included, the second question gave the same token IDs with the cache and without it (`turn2_ids_equal` in each receipt).

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
