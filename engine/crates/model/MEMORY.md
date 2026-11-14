# GB10 memory ledger

> **Historical record, 2026-10-02.** It shows where memory went for one
> Qwen3-8B session on an earlier build. Current main also serves resident
> official FP8 hybrid weights and Gemma text models, and keeps its KV cache in
> BF16 by default. The FP32 KV figures and the 173 GB capacity estimate below
> are not current allocations or measured model loads.
> [Models](../../docs/models.md) records today's supported precision.

This page is for anyone checking how memory was accounted for on that build.

<details>
<summary><b>Terms on this page</b>: KV cache, arena, RSS, pinned staging, RoPE table, BF16, FP8, FP32, GB</summary>

<dl>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Arena</dt>
<dd>One large GPU allocation that holds many weight matrices.</dd>
<dt>RSS</dt>
<dd>Resident set size: the host memory a process holds, as Linux reports
it.</dd>
<dt>Pinned staging</dt>
<dd>Host memory the GPU can copy from directly, used only while loading.</dd>
<dt>RoPE table</dt>
<dd>Precomputed rotation values that tell the model each token's position.</dd>
<dt>BF16, FP8, FP32</dt>
<dd>16-bit, 8-bit and 32-bit number formats.</dd>
<dt>GB</dt>
<dd>1,000,000,000 bytes. GiB is 1,073,741,824 bytes. MiB is 1,048,576
bytes.</dd>
</dl>

</details>

## One Qwen3-8B session

**The weights lived only on the GPU, and every measured unload returned
tracked GPU bytes to zero.**

Setup: CUDA Driver API 13.0, one session, 512-token capacity, a four-token
prompt. Matrices sit in packed GPU arenas, including the embedding. No copy
of the weights stays on the host. Norms stay in FP32 to match the kernels of
that build.

| Allocation | Bytes |
| --- | ---: |
| BF16 matrices (GPU) | 16,380,854,272 |
| FP32 normalization weights (GPU) | 1,232,896 |
| RoPE table, 512 positions (GPU) | 262,144 |
| FP32 keys and values (GPU) | 150,994,944 |
| Session activations and scratch | 1,011,080 |
| Pinned read staging | 33,554,432 |
| Host RSS at first token, lowest | 105,009,152 |
| Host RSS at first token, highest | 105,140,224 |
| Peak host RSS | 137,777,152 |

- The BF16 matrices include the embedding and output head. The keys and
  values cover 36 layers and 512 positions.
- Pinned read staging exists only during loading.
- Host RSS at first token includes the driver and runtime; lowest and
  highest are over the final five swaps. Peak host RSS includes temporary staging.

What the numbers show:

- Every measured unload of the real engine returned tracked GPU bytes to zero.
  Available system memory and host RSS stayed flat across model swaps.
- GPU bytes are not part of process RSS on this path.

<details>
<summary><b>More notes on these numbers</b></summary>

- The checkpoint has 16,381,470,720 bytes of BF16 data. Its 616,448 bytes of
  normalization weights expand only into the GPU FP32 norms above.
- Arena memory is freed with the last matrix that uses it.
- Reader threads finish before staging is freed.
- Header and norm reads go through the normal file cache and can leave small
  cached pages. Matrix data uses direct reads (`O_DIRECT`), falling back to
  ordinary reads when the file system refuses them (`EINVAL`).

</details>

## Historical 173 GB FP8 capacity estimate

**A calculation, not a measurement: how a 173 GB FP8 model would split across
Sparks.** It reserved 12 GB per Spark for the OS, runtime and headroom. GB is
1,000,000,000 bytes. Figures are per rank: one Spark each.

| Sparks | Weights (GB) | Left (GB) |
| ---: | ---: | ---: |
| 1 | 173.000 | does not fit |
| 2 | 86.500 | 31.423 |
| 3 | 57.667 | 60.256 |
| 4 | 43.250 | 74.673 |
| 5 | 34.600 | 83.323 |
| 6 | 28.833 | 89.090 |
| 7 | 24.714 | 93.208 |
| 8 | 21.625 | 96.298 |

"Left" is what remains on each rank after the 12 GB reserve. The weight figures
are ideal shares.

<details>
<summary><b>Starting figures and limits of the estimate</b></summary>

Starting figures:

- Plan with about 121 GiB (129.923 GB) of unified memory usable by CUDA, not
  the machine's 130.661 GB physical-memory figure.
- As an illustration, reserve 12 GB for the OS, runtime and headroom.
- What remains on each rank must hold the KV cache and other state,
  activations, communication buffers, copies of the embedding and output
  head, scales, and any unevenness between ranks.

Limits:

- These are ideal capacity figures, not measured FP8 loads, and not a promise
  that a model would be admitted.
- FP8 weights and their scales stay at checkpoint precision. The
  tensor-parallel adapter this estimate described accepted BF16.
- Row selections load directly into compact per-rank arenas. Column
  partitions need packed per-rank files.
- The estimate called for re-sharding the 173 GB model offline onto each
  rank's own NVMe drive, with 32 MiB of temporary staging per loading rank.

</details>

**Where next:** [engine-model](README.md) · [Models](../../docs/models.md) ·
[Server and API](../../docs/server.md#how-it-sizes-itself)
