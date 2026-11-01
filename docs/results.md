# Results and receipts

This page gives the numbers that the [README](../README.md) quotes. For each number, it gives the record that the number comes from. A record in this repository is linked. A record that is not in this repository is named, and the page says so.

All speeds are completion tokens per second (tok/s). Times are in seconds. A card is the result file of one RigMark run.

A short revision such as `dc32b7c` names a commit of the private development history. The public repository holds squashed copies of those commits ([glossary](GLOSSARY.md#ho), "Main `abc1234`"). Each receipt gives the full 40-character revision.

## Current main: dc32b7c, 2026-10-31 morning

The receipts are in [bench/receipts/2026-10-31](../bench/receipts/2026-10-31/README.md).

That page gives the build and binary hashes and the checkpoint identities. It names the machines: DGX Spark, NVIDIA GB10, driver 580.178.04, with the clocks not locked. It also gives the server start command of each setup and the request settings.

**One chat, drafting on (tok/s)**

<div align="center">

| Setup | Prose | Code | Structured |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 42.530 | 81.138 | 100.808 |
| FP4-Local · 2 Sparks | 54.878 | 102.314 | 130.800 |
| FP4-Local · 4 Sparks | 63.036 | 113.025 | 145.238 |
| FP8 · 2 Sparks | 36.624 | 51.293 | 52.763 |
| FP8 · 4 Sparks | 49.508 | 69.340 | 72.248 |

</div>

**Chats at once, drafting on (total tok/s)**

<div align="center">

| Setup | C1 | C4 | C8 |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 65.688 | 144.681 | 178.981 |
| FP4-Local · 2 Sparks | 79.632 | 166.796 | 232.198 |
| FP4-Local · 4 Sparks | 86.962 | 177.581 | 245.145 |
| FP8 · 2 Sparks | 45.436 | 66.578 | 98.670 |
| FP8 · 4 Sparks | 61.435 | 88.017 | 130.999 |

</div>

**Path, prose with drafting off, and the cold 8K prompt**

<div align="center">

| Setup | Path | Prose, drafting off (tok/s) | Cold 8K (s) |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | local | 27.911 | 3.769 |
| FP4-Local · 2 Sparks | tensor split | 36.047 | 2.153 |
| FP4-Local · 4 Sparks | tensor split in parts | 42.084 | 1.924 |
| FP8 · 2 Sparks | tensor parallel | 27.072 | 5.971 |
| FP8 · 4 Sparks | tensor parallel | 35.902 | 6.923 |

</div>

The drafting-off cards come from the same server starts as the drafting-on cards. Each drafting-off receipt also holds every other cell of its card. These are the receipts:

- **FP4-Local, one Spark, drafting on:** [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-on.json) and [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-on.json).
- **FP4-Local, one Spark, drafting off:** [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-off.json) and [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-off.json). The prose value 27.911 is the mean of the two cards.
- **FP4-Local, two Sparks:** [drafting on](../bench/receipts/2026-10-31/fp4-local-2spark-on.json) and [drafting off](../bench/receipts/2026-10-31/fp4-local-2spark-off.json).
- **FP4-Local, four Sparks:** [drafting on](../bench/receipts/2026-10-31/fp4-local-4spark-on.json) and [drafting off](../bench/receipts/2026-10-31/fp4-local-4spark-off.json).
- **FP8, two Sparks:** [drafting on](../bench/receipts/2026-10-31/fp8-2spark-on.json) and [drafting off](../bench/receipts/2026-10-31/fp8-2spark-off.json).
- **FP8, four Sparks:** [drafting on](../bench/receipts/2026-10-31/fp8-4spark-on.json) and [drafting off](../bench/receipts/2026-10-31/fp8-4spark-off.json).

How to read the tables:

- **Medians.** Each cell is the card's own median. Prose, code and structured use 5 runs. C1, C4 and C8 use 3 rounds. Cold 8K uses 3 cold requests. The one-Spark row is the mean of the medians of its two starts. Every other row is one server start.
- **Requests.** Prose, code and structured are one chat each, up to 8192 tokens, to the natural stop. C1, C4 and C8 are the total rate of 1, 4 or 8 chats at once, 256 tokens each. Cold 8K is the time to the first token of an uncached 8192-token prompt. Both cards send it with drafting off.
- **Chat settings.** The 60 chat requests of each card are greedy: temperature 0, top-p 1, top-k 0, neutral penalties, seed 20260905, thinking off and `cache` false.
- **8K settings.** The six 8K requests send token prompts with temperature 0, top-p 1, top-k 0, neutral penalties and `"draft": "plain"`. They send no seed and no `cache` field. Three are cold requests, each with a new cache salt. Each cold request has a replay with the same salt. No cold request restored a cached token. The replays restored all 8,192 tokens on FP8 and none on FP4-Local, which keeps a prompt only on request. The Cold 8K column uses the cold requests only.
- **Noise.** Most rows have one server start. Starts of the same setup have differed by up to 2.81 % in one cell: code on FP8 on four Sparks gave 65.530 and 67.370 tok/s on 2026-10-30. So treat a smaller gap between two rows as noise ([history](benchmarks.md#speed)).

### The 32K-token chat

The chat sends one 32,099-token question. Then it sends a second question on the same chat (32,305 tokens; 32,238 with FP8 on two Sparks). The second question ran twice with the cache and once with `cache` false. The table gives the time to the first token of the first question, of the first cached run and of the uncached run.

<div align="center">

| Setup | First question | Second question, cache | Second question, no cache |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark (mean of 2 starts) | 15.628 | 0.357 | 15.791 |
| FP4-Local · 2 Sparks | 10.092 | 0.758 | 9.947 |
| FP4-Local · 4 Sparks | 8.984 | 8.725 | 8.745 |
| FP8 · 2 Sparks | 24.116 | 0.938 | 23.908 |
| FP8 · 4 Sparks | 27.807 | 1.059 | 28.475 |

</div>

The cached run reused 32,064 tokens on every setup except four FP4-Local Sparks, which reused 0. Four FP4-Local Sparks keep no prefix cache, so their second question reads the whole prompt again. On every start, the second question gave the same token IDs with the cache and without it (`turn2_ids_equal` in each receipt).

These are the receipts:

- **FP4-Local, one Spark:** [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-chat32k.json) and [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-chat32k.json).
- **FP4-Local, two Sparks:** [chat](../bench/receipts/2026-10-31/fp4-local-2spark-chat32k.json).
- **FP4-Local, four Sparks:** [chat](../bench/receipts/2026-10-31/fp4-local-4spark-chat32k.json).
- **FP8, two Sparks:** [chat](../bench/receipts/2026-10-31/fp8-2spark-chat32k.json).
- **FP8, four Sparks:** [chat](../bench/receipts/2026-10-31/fp8-4spark-chat32k.json).

### Token IDs

- **FP4-Local, one, two and four Sparks, drafting on and off.** All eight cards have the same `token_ids_sha256`. So every one of the 66 responses had the same token IDs.
- **FP8, two Sparks.** The drafting-on and drafting-off cards have the same value.
- **FP8, four Sparks.** The drafting-on and drafting-off cards have the same value.
- **FP8, two against four Sparks.** The values differ. Compared response by response, 7 of 66 are the same and 59 differ. This count is from the private card comparison (private report `own-rigmark-morning-r1`), not from a file here. It broke the rule that supported Spark counts give the same token IDs. A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order, so four Sparks now give the same token IDs as two ([changelog](../CHANGELOG.md)). These cards predate that change.

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

## Other numbers the README quotes

These numbers come from earlier runs. Their records are private and not in this repository, except where a line links a record.

- **One Spark at `--context 262144`.** A 64K prompt gave its first token in 32.97 s, and a 128K prompt in 71.53 s (means of two runs). The run was on 2026-10-30 at main `380d8e5`. The record is the context run in private report `own-rigmark-eod-r2`.
- **Two Sparks at `--context 262144`.** A 201,654-token prompt gave its first token in 68.02 s and in 68.10 s. The run was on 2026-10-28 at main `285fa8a`. The record is private report `own-ctx-r1`.
- **Two FP4-Local pairs at once.** Eight chats on each pair gave 465.2 tok/s in total (means of two starts). The run was on 2026-10-30 at main `51dce02` (engine `380d8e5`). The record is private report `own-two-pairs-r1` ([two pairs at once](benchmarks.md#two-pairs-at-once)).
- **Bundle steps to `Launch ready`.** The steps took 101.262 s. The run was on 2026-10-28 with the v0.1.0 bundle (`5a22651`). The record is `2026-10-28_1305_bundle-install_5a22651`.
- **v0.1.0 on one Spark, drafting on.** Code ran at 76.7 tok/s and C8 at 173.9 tok/s. The run was on 2026-10-24 at `5a22651`. The record is in this repository: [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md).
- **Rented cloud GB10 VMs.** They downloaded the checkpoint at 8 to 22 MB/s on 2026-10-24 ([rented cloud GB10 VMs](benchmarks.md#rented-cloud-gb10-vms)).

The [benchmark history](benchmarks.md) keeps every earlier card, newest first.
