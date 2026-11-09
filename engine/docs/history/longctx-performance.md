# Long-context results

> **Historical record.** These measurements were taken on 2 October 2026, on
> the build added to main in commit `67127c8` (long-context prefill attention
> and shared KV prefixes for forked conversations). They do not describe
> current main.
>
> That build differs from current main in ways that change the numbers:
>
> - Its main runs stored the KV cache in FP32. BF16 KV was opt-in then; it is
>   now the serving default.
> - Its HTTP runs used a benchmark client that was not frozen, unlike the
>   current scoreboard clients.
>
> Current figures and the same-session comparison method are in
> [Benchmarks](../benchmarks.md). Current checks are in
> [Validation](../validation.md).

**What this page covers**

- Prompt-reading time and decode speed for Qwen3-8B at 8,192, 28,672 and
  38,912 prompt tokens, before and after the long-context attention work.
- Where the prefill time and memory went, and how forked and cached
  conversations improved.
- The checks that passed, the targets still open, and the commands recorded.

## Terms on this page

<dl>
<dt>TTFT</dt>
<dd>Time to first token: how long reading the prompt takes before the answer
starts.</dd>
<dt>Prefill, decode</dt>
<dd>Reading the prompt; then writing the answer, one token per step.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>KV packing</dt>
<dd>Rearranging the KV cache into the layout attention reads.</dd>
<dt>Fork</dt>
<dd>A new conversation branch that shares an existing conversation's cache.</dd>
<dt>HF</dt>
<dd>The official Hugging Face Transformers reference.</dd>
<dt>Device peak</dt>
<dd>The most GPU memory allocated at once.</dd>
<dt>Host HWM, RSS</dt>
<dd>CPU-side memory: its high-water mark, and the amount in use.</dd>
<dt>B, GB</dt>
<dd>Bytes; decimal gigabytes (1,000,000,000 bytes).</dd>
</dl>

## Setup

- One DGX Spark, Qwen3-8B with BF16 weights.
- **Baseline** is the checkout the long-context round started from, which
  already had query-tiled attention and growable KV memory.
- **Engine runs** use prompt token IDs 1 to N, followed by 32 greedy steps.
  Engine profiling adds CUDA timing events.
- **HTTP runs** use the benchmark's library prose, 64 output tokens, plain
  decoding and 512-token prefill chunks.
- Times exclude model loading. Each table row is one run.

## Results in short

- **Faster prompt reading.** Engine TTFT at 38,912 tokens fell from 57.804 s
  to 35.439 s, and at 28,672 tokens from 33.870 s to 21.832 s.
- **Where it came from.** Attention at 38,912 tokens fell from 39.398 s to
  20.779 s, and KV packing from 3.732 s to 0.047 s.
- **Same output.** Prompt-end logits are bit-identical to the saved baseline
  at all three lengths. All 192 HTTP token IDs and the output text match the
  baseline.
- **Faster cached follow-up.** A cached 28k follow-up fell from 807.684 ms to
  400.008 ms, and its device peak from 42,272,774,308 B to 25,618,054,180 B.
- **Still open.** The revised 20 s and 30 s TTFT targets, the FP32 decode
  targets of 12 and 11 tokens/s, and the 100 ms cached-follow-up target.

## Engine results

Baseline to final, at each prompt length in tokens. Seconds and GB: lower is
better. tok/s: higher is better.

| Engine | 8,192 | 28,672 | 38,912 |
| --- | ---: | ---: | ---: |
| TTFT (s) | 4.973 to 3.955 | 33.870 to 21.832 | 57.804 to 35.439 |
| Decode (tok/s) | 12.263 to 12.597 | 8.318 to 9.381 | 7.236 to 8.339 |
| Peak device (GB) | 18.970 to 18.955 | 25.196 to 25.006 | 28.309 to 28.032 |
| Host HWM, final (bytes) | 145,014,784 | 145,002,496 | 165,888,000 |

- Peaks include prefill and decode, and exclude the separate fork
  experiment.
- The host HWM row is the final build's native host high-water mark.

## HTTP results

Baseline to final, at each prompt length in tokens.

| HTTP | 8,192 | 28,672 | 38,912 |
| --- | ---: | ---: | ---: |
| TTFT (s) | 5.112 to 4.103 | 34.595 to 22.467 | 58.900 to 36.281 |
| Decode (tok/s) | 12.338 to 12.213 | 8.174 to 8.933 | 6.928 to 7.853 |
| Peak device (GB) | 18.999 to 18.971 | 25.207 to 25.012 | 28.311 to 28.033 |

- All 192 HTTP token IDs and the output text match the baseline.
- Server lifetime RSS high-water was 271,724,544 B before and 214,052,864 B
  after.
- CUDA allocation accounting and RSS overlap on unified memory. Do not add
  them together as separate physical use.

## Where the prefill time went

CUDA event profiles, in seconds, baseline to final, at each prompt length
in tokens:

| Prefill part | 8,192 | 28,672 | 38,912 |
| --- | ---: | ---: | ---: |
| Projections, with final head | 2.565 to 2.564 | 8.985 to 9.031 | 12.402 to 12.402 |
| Input packing | 0.184 to 0.184 | 0.646 to 0.635 | 0.893 to 0.886 |
| Attention | 1.757 to 0.866 | 21.183 to 11.127 | 39.398 to 20.779 |
| KV packing | 0.179 to 0.047 | 2.067 to 0.048 | 3.732 to 0.047 |
| QK, norm, RoPE, KV writes | 0.042 to 0.043 | 0.154 to 0.155 | 0.208 to 0.206 |
| Allocation, KV growth wall | 0.083 to 0.086 | 0.268 to 0.283 | 0.396 to 0.352 |

- Copies outside packing take under 0.008 ms per prefill.
- Final RMS and residual kernels cost 0.141, 0.496 and 0.690 s at the three
  lengths.
- Event totals, allocation wall time, stream wait and total wall time have
  different scopes. Do not add overlapping timers.
- Large-context attention includes its on-chip conversion work.

How the attention time came down:

1. **Shared-memory swizzling** alone cut 28k and 38.9k attention to 9.385 s
   and 17.558 s, with 2.073 s and 3.824 s still spent packing KV.
2. **On-chip conversion** of long prefixes removed that separate packing pass,
   while keeping scratch memory bounded.
3. **Tried and removed:** wider key tiles and separate correction
   accumulators did not improve wall time.
4. **Kept:** hardware BF16 pair conversion and register probability fragments.
   They leave the retained FP32 values and the final prompt logits unchanged.

## Memory at 38,912 tokens

The prefill peak is exactly 28,011,116,064 B:

| Component | Bytes |
|---|---:|
| BF16 weights, one copy | 16,380,854,272 |
| Model norms and RoPE | 21,188,608 |
| FP32 KV | 11,475,615,744 |
| Activations, projection packing | 96,536,596 |
| Bounded KV split views | 33,554,432 |
| Decode workspace | 3,366,412 |

- Decode's next physical KV page raises the run peak to 28,032,036,876 B.
- KV and workspace sizes follow the live capacity.
- The fused grouped decode and verification path holds no full score matrix.
- Shared physical allocations are counted once, even when several snapshots
  map them.

## BF16 KV (opt-in then, default now)

With BF16 KV, at each prompt length in tokens:

| BF16 KV | 8,192 | 28,672 | 38,912 |
| --- | ---: | ---: | ---: |
| TTFT (s) | 3.552 | 16.725 | 25.720 |
| Decode (tok/s) | 12.922 | 10.775 | 9.928 |
| Peak device (GB) | 17.747 | 20.778 | 22.294 |
| Host HWM (bytes) | 144,990,208 | 145,010,688 | 166,260,736 |

## Correctness checks

- **HF, FP32 KV.** 56 of 56, with the maximum saved-logit error unchanged at
  0.00964928.
- **HF, BF16 KV.** 56 of 56, with a maximum error of 0.22369194. That does not
  meet the FP32 tolerance of 0.01.
- **Exact verification.** Passed widths 1 to 16, six prefixes and four
  repetitions, in both KV formats.
- **KV isolation.** A temporary probe passed append, truncate, nested fork,
  parent drop and zero allocation balance. It was then removed from the tree.
- **Load check.** The final load check passes two cycles with 512 exact
  token and score matches, and zero tracked GPU bytes after each unload. Host
  HWM was 147,910,656 B. Thread and file-descriptor counts stayed at 3 and 33.

## Forked and cached conversations

- **28k engine fork.** The first step fell from 460.841 ms and a 33.634 GB
  peak to 160.908 ms and a 25.160 GB peak. The latter includes 51.917 ms of
  allocation and mapping, 0.888 ms of copying, and the forward pass.
- **A GB10 limit.** The GB10 accepts aliases of whole allocations but returns
  CUDA error 801 for partial aliases. An incomplete physical allocation is
  therefore copied. That copy can still be large if the original cache was
  reserved in one piece.
- **HTTP cached follow-up.** Fell from 807.684 ms to 400.008 ms, reusing
  28,668 tokens of a 28,752-token request. Device peak fell from
  42,272,774,308 B to 25,618,054,180 B. Final server host HWM was
  230,088,704 B. Cached and fresh token IDs and text match.
- **Coalescing prefix access calls.** Measured 400.008 ms against 402.614 ms
  before. That small difference is within run-to-run variation.

The 100 ms cached-follow-up target remains open. A paged KV pointer table, or
append ownership, could avoid remapping every prefix allocation for a new
writable branch.

## Targets

| Target | Result |
| --- | --- |
| Original TTFT, 5, 25 and 40 s | Pass |
| Memory: weights, KV and 1 GB | Pass |
| Revised TTFT, 20 s and 30 s | Open |
| FP32 decode, 12 and 11 tok/s | Open |
| Cached follow-up in 100 ms | Open |

The FP32 decode targets exceed the measured combined weight and KV read
budget: the final 38.9k profile is 67.814 ms of projections plus 50.888 ms
of attention.

The next step named then: optimize the 11.127 s and 20.779 s attention cost
at 28k and 38.9k, and evaluate projection optimizations together.

## Reproduce

These commands reproduce that measurement using `batch-bench` in place of
the retired `longctx-bench` alias. They measure current defaults; label any
new run with the commit it measured.

### Ask your coding agent

```text
Re-run the long-context measurement recorded in docs/history/longctx-performance.md
on this DGX Spark, using the "Reproduce" commands on that page. Stop any
running Vade server first so the GPU is idle. Run as the current user and
never use sudo. Report the commit you measured, every TTFT, decode rate and
peak-memory figure, and whether the token IDs matched. Label the results as a
new measurement of that commit, not as the historical build's numbers. Do not
commit or push.
```

### Do it yourself

1. Set the paths and build. `BIND` is a loopback socket you choose, and `URL`
   is its HTTP origin.

   ```sh
   # Run from the checkout root.
   P=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
   M=models/Qwen/Qwen3-8B
   cargo kernels
   cargo build --release --offline -p spark-engine -p spark-serve
   ```

2. Run the engine profile and checks. The logit file is written when absent
   and compared when present, so keep a baseline file before changing
   kernels.

   ```sh
   target/release/batch-bench "$P" "$M" 38912 32 profile baseline-38912.f32
   target/release/batch-bench "$P" "$M" 38912 32 profile-bf16
   target/release/hf-batch-check "$P" "$M" fixtures/qwen3/{short,medium,long}.json
   target/release/batch-bench "$P" "$M" 2048 32 verify
   target/release/load-check "$P" "$M" 1024 2 512
   ```

3. In a separate terminal, start the server with the recorded settings.

   ```sh
   target/release/spark-serve "$P" "$M" --draft plain --bind "$BIND" \
     --context-tokens 40960 --input-tokens 38912 --output-tokens 2048 \
     --kv-budget-mib 11520 --session-budget-mib 73728 --gpu-budget-mib 102400 \
     --max-active 2 --prefill-chunk 512 --generation-timeout-secs 900
   ```

4. Build the benchmark client, then run the long-prompt case.

   ```sh
   cargo build --release --manifest-path tools/bench/Cargo.toml
   tools/bench/target/release/vade-bench "$URL" --tokenizer "$M" \
     --only long --long-target 38912 --long-output 64 --max-tokens 64 \
     --repeats 1 --timeout-secs 600 --budget-secs 900 --output longctx-http.json
   ```

5. For the cached follow-up, replace
   `--only long --long-target 38912 --long-output 64` with
   `--only cache --cache-target 28672`.

Notes on the tools:

- Use `batch-bench` for both long-context profiles and shorter verification runs.
- Profile array indices 0 to 9 keep their operation order; 10 is input
  packing, 11 is KV packing and 12 is copies.
- Allocation timing includes mapping and scratch allocations.
