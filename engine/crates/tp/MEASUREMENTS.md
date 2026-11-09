# Historical scale measurements

> **Historical record.** These numbers come from commit `076ac7c`, before the
> tensor-parallel code was adapted to the current engine. **No Spark
> measurements were made for the current integration.** They are kept as
> earlier evidence only. To measure today's build, use the commands in the
> [crate README](README.md) on two, three or four Sparks.

This page is for anyone asking how the earlier tensor-parallel build scaled
from one Spark to three.

<details>
<summary><b>Terms on this page</b>: rank, TP2, TP3, TTFT64, TTFT1024, Verify16, app bytes, KV</summary>

<dl>
<dt>Rank</dt>
<dd>One Spark's process in the group.</dd>
<dt>TP2, TP3</dt>
<dd>Tensor parallelism across two or three Sparks.</dd>
<dt>TTFT64, TTFT1024</dt>
<dd>Time to first token for a 64-token or 1,024-token prompt.</dd>
<dt>Verify16</dt>
<dd>Checking 16 proposed tokens in one pass.</dd>
<dt>App bytes</dt>
<dd>Device, pinned and tracked heap peaks added together, excluding the driver
and code.</dd>
<dt>KV</dt>
<dd>The model's memory of the conversation so far.</dd>
</dl>

</details>

## Speed and memory per rank

**On that earlier build, two Sparks decoded at 28.254 tok/s and three at
37.981, against 15.476 on one.** Official Qwen3-8B, BF16 weights. Each value
is the slower rank's median over three measured repeats after warm-up.

tok/s: higher is better. ms: lower is better.

| Speed | 1 Spark | 2 Sparks | 3 Sparks |
| --- | ---: | ---: | ---: |
| Decode (tok/s) | 15.476 | 28.254 | 37.981 |
| TTFT64 (ms) | 104.387 | 95.753 | 100.053 |
| TTFT1024 (ms) | 812.261 | 641.292 | 708.412 |
| Verify16 (ms) | 111.482 | 69.862 | 67.179 |

App bytes per rank during each test, lower is better:

| App bytes per rank | 1 Spark | 2 Sparks | 3 Sparks |
| --- | ---: | ---: | ---: |
| Decode | 15,198,000,000 | 7,618,000,000 | 5,233,000,000 |
| TTFT64 | 15,197,000,000 | 7,608,000,000 | 5,206,000,000 |
| TTFT1024 | 15,729,000,000 | 7,926,000,000 | 5,495,000,000 |
| Verify16 | 15,242,000,000 | 7,662,000,000 | 5,275,000,000 |

- The two-Spark decode result, 28.254 tok/s, **missed its 29 tok/s target.**
- There was no four-Spark measurement for that revision.
- The byte values in the app-bytes table were reported rounded to decimal GB,
  so they are approximate.

## Correctness at the time

**TP2 and TP3 matched the Hugging Face reference on 56 of 56 positions, on the
old engine snapshot.**

- TP2 and TP3 passed the core checks and matched the Hugging Face reference
  on 56 of 56 teacher-forced positions.
- These results used the old engine snapshot and must be repeated after the
  merge. GPU overlap and physical four-rank runs remain unverified.

<details>
<summary><b>Setup details</b></summary>

- FP32 KV cache, split by heads. One resident weight share per rank.
- Prompts read in 512-token chunks. Prompt IDs were 1 to T.
- Decode generated 32 tokens after a 64-token prefix. Strict verification
  used 16 rows after a 64-token prefix.
- Profiling was off.
- Earlier designs and rejected tuning paths are left out. None of them
  establishes performance for the merged engine hooks.

</details>

<details>
<summary><b>Memory of the largest rank during the 1,024-token prompt, exact bytes</b></summary>

| Memory part | 1 Spark | 2 Sparks | 3 Sparks |
| --- | ---: | ---: | ---: |
| Weights and auxiliary | 15,137,984,512 | 7,569,887,232 | 5,173,039,104 |
| KV capacity | 301,989,888 | 150,994,944 | 113,246,208 |
| Device scratch | 270,716,300 | 178,136,856 | 165,452,356 |
| Pinned | 8,388,608 | 16,777,216 | 33,554,432 |
| Tracked heap | 9,904,570 | 10,429,618 | 9,905,458 |

With three ranks, the KV heads were split 3, 3 and 2.

</details>

<details>
<summary><b>The correctness checks in full</b></summary>

- The core checks were the top token after 64- and 1,024-token prompts, 32
  decode IDs, strict 16-row verification bits, and truncate-then-reappend.
- Against the Hugging Face reference, the largest full-score differences
  were 0.00045204 (TP2) and 0.00047493 (TP3).

</details>

**Where next:** [spark-tp](README.md) · [Multiple Sparks](../../docs/multi-spark.md)
