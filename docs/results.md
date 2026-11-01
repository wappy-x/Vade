# Results and receipts

This page gives the numbers that the [README](../README.md) quotes, and for
each one the record it comes from. A record in this repository is linked. A
record that is not in this repository is named, and the page says so.

All speeds are completion tokens per second (tok/s). Times are seconds. A
short revision such as `dc32b7c` names a commit of the private development
history; the public repository holds squashed copies of it
([glossary](GLOSSARY.md#ho), "Main `abc1234`"). Each receipt gives the full
40-character revision.

## Current main: dc32b7c, 2026-10-31 morning

**Receipts:** [bench/receipts/2026-10-31](../bench/receipts/2026-10-31/README.md).
That page gives the build and binary hashes, the checkpoint identities, the
machines (DGX Spark, NVIDIA GB10, driver 580.178.04, clocks not locked), the
server start command of each setup and the request settings.

| Model | Sparks | Path | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) | Receipts |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp4-local | 1 | local | ON | 42.530 | 81.138 | 100.808 | 65.688 | 144.681 | 178.981 | 3.769 | [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-on.json), [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-on.json) |
| fp4-local | 2 | tensor split | ON | 54.878 | 102.314 | 130.800 | 79.632 | 166.796 | 232.198 | 2.153 | [card](../bench/receipts/2026-10-31/fp4-local-2spark-on.json) |
| fp4-local | 4 | tensor split in parts | ON | 63.036 | 113.025 | 145.238 | 86.962 | 177.581 | 245.145 | 1.924 | [card](../bench/receipts/2026-10-31/fp4-local-4spark-on.json) |
| fp8 | 2 | tensor parallel | ON | 36.624 | 51.293 | 52.763 | 45.436 | 66.578 | 98.670 | 5.971 | [card](../bench/receipts/2026-10-31/fp8-2spark-on.json) |
| fp8 | 4 | tensor parallel | ON | 49.508 | 69.340 | 72.248 | 61.435 | 88.017 | 130.999 | 6.923 | [card](../bench/receipts/2026-10-31/fp8-4spark-on.json) |

Drafting OFF, from the same server starts (prose, in tok/s): FP4-Local 27.911
on one Spark (mean of the
[start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-off.json) and
[start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-off.json)
cards), [36.047](../bench/receipts/2026-10-31/fp4-local-2spark-off.json) on
two and [42.084](../bench/receipts/2026-10-31/fp4-local-4spark-off.json) on
four; FP8 [27.072](../bench/receipts/2026-10-31/fp8-2spark-off.json) on two
and [35.902](../bench/receipts/2026-10-31/fp8-4spark-off.json) on four. Each
OFF file has every other cell too.

How to read the table:

- Each cell is the card's own median: 5 runs for prose, code and
  structured; 3 rounds for C1, C4 and C8; 3 cold requests for Cold 8K. The
  one-Spark row is the mean of the medians of its two starts. Every other
  row is one server start.
- Prose, code and structured are one chat each, up to 8192 tokens, to the
  natural stop. C1, C4 and C8 are the total rate of 1, 4 or 8 chats at once,
  256 tokens each. Cold 8K is the time to the first token of an uncached
  8192-token prompt, sent with drafting OFF in both cards.
- The 60 chat requests of each card are greedy: temperature 0, top-p 1,
  top-k 0, neutral penalties, seed 20260905, thinking OFF and `cache` false.
  The six 8K requests send token prompts with temperature 0, top-p 1, top-k
  0, neutral penalties and `"draft": "plain"`, with no seed and no `cache`
  field: three cold requests, each with a new cache salt, and a replay of
  each with the same salt. No cold request restored a cached token. The
  replays restored all 8,192 tokens on FP8 and none on FP4-Local, which
  keeps a prompt only on request. The Cold 8K column uses the cold
  requests only.
- Most rows have one server start. Starts of the same setup have differed
  by up to 2.81 % in one cell (code, FP8 on four Sparks, 65.530 and 67.370
  tok/s on 2026-10-30), so treat a smaller gap between two rows as noise
  ([history](benchmarks.md#speed)).

### The 32K-token chat

One 32,099-token question, then a second question on the same chat (32,305
tokens; 32,238 with FP8 on two Sparks). The second question ran twice with
the cache and once with `cache` false. The table gives the first token time
of the first question, of the first cached run and of the uncached run.

| Setup | First question | Second question, cache | Tokens reused | Second question, no cache | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| FP4-Local, one Spark (mean of 2 starts) | 15.628 | 0.357 | 32,064 | 15.791 | [start 1](../bench/receipts/2026-10-31/fp4-local-1spark-start1-chat32k.json), [start 2](../bench/receipts/2026-10-31/fp4-local-1spark-start2-chat32k.json) |
| FP4-Local, two Sparks | 10.092 | 0.758 | 32,064 | 9.947 | [chat](../bench/receipts/2026-10-31/fp4-local-2spark-chat32k.json) |
| FP4-Local, four Sparks | 8.984 | 8.725 | 0 | 8.745 | [chat](../bench/receipts/2026-10-31/fp4-local-4spark-chat32k.json) |
| FP8, two Sparks | 24.116 | 0.938 | 32,064 | 23.908 | [chat](../bench/receipts/2026-10-31/fp8-2spark-chat32k.json) |
| FP8, four Sparks | 27.807 | 1.059 | 32,064 | 28.475 | [chat](../bench/receipts/2026-10-31/fp8-4spark-chat32k.json) |

Four FP4-Local Sparks keep no prefix cache, so their second question reads
the whole prompt again. On every start, the second question gave the same
token IDs with the cache and without it (`turn2_ids_equal` in each receipt).

### Token IDs

- **FP4-Local, one, two and four Sparks, drafting ON and OFF:** all eight cards
  have the same `token_ids_sha256`, so every one of the 66 responses had the
  same token IDs.
- **FP8, two Sparks:** ON and OFF have the same value. **FP8, four Sparks:**
  ON and OFF have the same value.
- **FP8, two against four Sparks:** the values differ. Compared response by
  response, 7 of 66 are the same and 59 differ. This count is from the
  private card comparison (private report `own-rigmark-morning-r1`), not from a
  file here. It breaks the rule that supported Spark counts give the same
  token IDs; compare FP8 results only at the same Spark count.

### What a stranger can and cannot re-run

| Step | Can you run it? |
| --- | --- |
| Build the measured tree and start one or two Sparks as in the cards | Yes. Check out public commit `db6b7fc` (`db6b7fcd1f0ed97d6d21084eab1d206cc0d5ef3c`), whose tree `aea2c506` is the tree of main `dc32b7c`, then use the commands in the receipts page |
| Start four Sparks or FP8 as in the cards | The cards used a private launcher. `vade up --sparks 4` and `vade up --fp8` start the same path, but we have not carded them on this date |
| Send the 66 card requests | Yes. Clone [RigMark](https://github.com/alexellis/rigmark) and check out revision `52c29c3`, then run `vade bench --card --prompts rigmark/prompts.json`. It sends the card's 66 requests in the card's order with the card's request bodies, and saves every token ID in `vade-card.json`. Add `--draft plain` for a drafting-OFF card. Plain RigMark sends a different cold 8K request ([details](benchmarks.md#run-rigmark-against-vade)) |
| Compare token IDs with these receipts | Yes. Add `--receipt bench/receipts/2026-10-31/<file>.json`. The client prints `ids 66/66 identical` when its `token_ids_sha256` equals the receipt's. On 2026-10-31 it printed that line against `fp4-local-1spark-start1-on.json` (drafting ON) and `fp4-local-1spark-start1-off.json` (`--draft plain`), on one rented GB10 machine that served main `acf5264` with the settings of `vade up --sparks 1 --context 32768 --output 8192`. We have not run it against the two-Spark, four-Spark or FP8 receipts |
| Run the 32K chat | No. The client and its prompt are private |
| Check drafting ON against OFF on your own server | Yes: `vade verify` |
| Time your own server | Yes: `vade bench`. It uses other prompts, so its numbers do not compare with the cards |

## Install time

**Receipt:** [fresh-install-2026-10-30.txt](../bench/receipts/2026-10-31/fresh-install-2026-10-30.txt).

On 2026-10-30, on one DGX Spark with no Rust and no Vade files and the model
already on disk, a clone of the public repository at `82ec816` (the tree of
main `0213a82`) took 0.69 s from a local bundle, `sh scripts/install.sh` took
176.8 s and the first `vade up` took 67.4 s.

## Other numbers the README quotes

These come from earlier runs. Their records are private, not in
this repository.

| Claim | Date and build | Record (private) |
| --- | --- | --- |
| One Spark at `--context 262144`: a 64K prompt gave its first token in 32.97 s and a 128K prompt in 71.53 s (means of two runs) | 2026-10-30, main `380d8e5` | Private report `own-rigmark-eod-r2`, context run |
| Two Sparks at `--context 262144`: a 201,654-token prompt gave its first token in 68.02 and 68.10 s | 2026-10-28, main `285fa8a` | Private report `own-ctx-r1` |
| Two FP4-Local pairs at once, eight chats on each pair: 465.2 tok/s in total (means of two starts) | 2026-10-30, main `51dce02` (engine `380d8e5`) | Private report `own-two-pairs-r1` ([two pairs at once](benchmarks.md#two-pairs-at-once)) |
| Bundle steps to `Launch ready`: 101.262 s | 2026-10-28, v0.1.0 bundle (`5a22651`) | Record `2026-10-28_1305_bundle-install_5a22651` |
| v0.1.0 on one Spark, drafting ON: code 76.7 tok/s, C8 173.9 tok/s | 2026-10-24, `5a22651` | In this repository: [bench/receipts/2026-10-24](../bench/receipts/2026-10-24/README.md) |
| Rented cloud GB10 VMs downloaded the checkpoint at 8 to 22 MB/s | 2026-10-24 | [Rented cloud GB10 VMs](benchmarks.md#rented-cloud-gb10-vms) |

The [benchmark history](benchmarks.md) keeps every earlier card, newest first.
