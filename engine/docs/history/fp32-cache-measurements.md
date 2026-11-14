# FP32-cache integration measurements

> **Historical record.** These measurements were taken on 2 October 2026, on
> the build added to main in commit `1d52612` (draft-ahead decoding,
> query-tiled prefill attention and growable KV cache). They do not describe
> current main.
>
> That build differs from current main in ways that change the numbers:
>
> - It used FP32 activations and an FP32 KV cache by default. BF16 KV was an
>   explicit option. Current serving uses a BF16 KV cache, and official
>   resident FP8 where applicable.
> - STRICT drafting was the single-Spark server default, and FAST was an
>   explicit option. The current default is `auto`.
> - Its prefill precision policy used FP16 for some projections and two BF16
>   parts for the others. Current Qwen3 paths use native BF16 operands;
>   Flash's covered BF16 projections use one fixed high/residual tensor-core
>   mode, with separate validation and goldens.
>
> Current figures are in [Benchmarks](../benchmarks.md). Current checks are in
> [Validation](../validation.md).

**What this page covers**

- Prompt-reading time and decode speed for Qwen3-8B on that build, with FP32
  and BF16 KV caches.
- HTTP results for plain, STRICT and FAST drafting on four kinds of prompt.
- The correctness checks, verification timing and memory ledger recorded then.

## Terms on this page

<dl>
<dt>TTFT</dt>
<dd>Time to first token: from sending the request to receiving the first answer
token.</dd>
<dt>Prefill, decode</dt>
<dd>Reading the prompt; then writing the answer, one token per step.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>BF16, FP16, FP32</dt>
<dd>16-bit and 32-bit number formats.</dd>
<dt>Plain, STRICT, FAST</dt>
<dd>No drafting; exact checked drafting; experimental guarded drafting.</dd>
<dt>HF</dt>
<dd>The official Hugging Face Transformers reference.</dd>
<dt>Teacher forcing</dt>
<dd>Feeding the reference's own tokens at each position, so one early difference
cannot spoil the rest.</dd>
<dt>Top-1</dt>
<dd>The highest-scoring next token.</dd>
<dt>Verification width</dt>
<dd>How many proposed rows are checked in one pass.</dd>
<dt>Device peak</dt>
<dd>The most GPU memory allocated at once.</dd>
<dt>Host RSS, high-water</dt>
<dd>CPU-side memory in use, and its maximum.</dd>
<dt>B</dt>
<dd>Bytes</dd>
</dl>

## Setup

- One DGX Spark, Qwen3-8B, one packed BF16 copy of the weights.
- FP32 activations; FP32 KV cache unless the table says BF16.
- These are measured workloads, not limits for arbitrary prompts.
- Benchmark mode measures speed. Correctness comes from the separate checks
  below.

## Results in short

- **TTFT target missed.** The 2,048-token TTFT target of 800 ms was **not
  met**: the retained 512-token-chunk implementation measured 907.780 ms.
- **Decode.** FP32 decode from 2,048 to 2,112 positions measured 14.260
  tokens/s.
- **Drafting.** Greedy STRICT reached 103.807 tokens/s on the code-edit prompt
  and 124.275 on the JSON edit, against 14.897 and 14.892 plain. Prose
  gained nothing: 14.899 STRICT against 14.956 plain.
- **Same output.** All 24 HTTP responses matched plain decoding's token IDs,
  streamed text and usage.
- **Reference.** HF teacher forcing matched 56 of 56 top-1 positions with
  FP32 KV. BF16 KV also matched 56 of 56, but with larger errors and only 11
  of 32 matching long-continuation tokens, so BF16 KV stayed opt-in then.

## Prefill and decode

Native `batch-bench ... 2048 64 bench`. Prompt IDs are 1 through N, and model
loading is excluded.

- Every row shares a model allocation of 16,383,184,896 B and a process host
  peak RSS of 152,338,432 B.
- State and scratch include the persistent decode workspace.

TTFT in ms, lower is better:

| Prompt tokens | FP32 KV | BF16 KV |
| ---: | ---: | ---: |
| 64 | 92.584 | 90.480 |
| 512 | 212.769 | 201.876 |
| 1024 | 425.696 | 408.374 |
| 2048 | 907.780 | 834.689 |

Memory with FP32 KV, in bytes:

| Prompt tokens | Physical KV | State and scratch | Device peak |
| ---: | ---: | ---: | ---: |
| 64 | 18,874,368 | 14,220,960 | 16,416,280,224 |
| 512 | 150,994,944 | 102,360,480 | 16,636,540,320 |
| 1024 | 301,989,888 | 106,554,784 | 16,791,729,568 |
| 2048 | 603,979,776 | 114,943,392 | 17,102,108,064 |

Memory with BF16 KV, in bytes:

| Prompt tokens | Physical KV | State and scratch | Device peak |
| ---: | ---: | ---: | ---: |
| 64 | 9,437,184 | 13,696,680 | 16,406,318,760 |
| 512 | 75,497,472 | 98,166,184 | 16,556,848,552 |
| 1024 | 150,994,944 | 98,166,184 | 16,632,346,024 |
| 2048 | 301,989,888 | 98,166,184 | 16,783,340,968 |

Decode from 2,048 through 2,112 positions, including real KV growth across
the 2,048 boundary:

| KV | ms per token | Tokens/s | Device peak (B) |
| --- | ---: | ---: | ---: |
| FP32 | 70.126 | 14.260 | 17,139,789,196 |
| BF16 | 69.943 | 14.297 | 16,837,799,308 |

For FP32, the device figure is both the retained and the peak value.

## What changed inside prefill

<dl>
<dt>Before</dt>
<dd>The prior validated build measured 101, 315, 681 and 1,539 ms at 64, 512,
1,024 and 2,048 tokens.</dd>
<dt>Profile</dt>
<dd>The final event profile measured 908.895 ms wall time at 2,048 tokens:
699.424 ms in projection and packing kernels, and 131.994 ms in attention and
FP32 KV packing. Its device peak was 17,102,050,720 B and host peak 144,764,928
B. Projections remained the largest opportunity. Profiling adds event
overhead.</dd>
<dt>Larger chunks, not kept</dt>
<dd>A 1,024-token-chunk experiment passed HF with a long-fixture TTFT of 878.205
ms and a peak of 17,190,219,168 B, against 914.342 ms and 17,102,071,200 B with
512-token chunks. It was not kept because of the extra 88,147,968 B and the
longer scheduling step.</dd>
<dt>FP16 down projections, not kept</dt>
<dd>Their maximum HF error exceeded 0.01.</dd>
<dt>Precision policy kept then</dt>
<dd>FP16 only for the fused gate/up projection in zero-based layers 18 to 35;
other projections used two BF16 components.</dd>
<dt>Scratch reuse</dt>
<dd>Reusing scratch across chunks within one <code>prefill()</code> call cut the
native 2,048 timing from 922.799 to 907.780 ms without raising its peak
allocation. The long HF fixture fell to 895.113 ms with identical saved-logit
errors. Scratch is released after the final row and before a scheduler
yield.</dd>
</dl>

## HTTP drafting results

How these were measured:

- Sampling used seed 42, temperature 0.7, top-k 20 and top-p 0.8. Greedy used
  temperature zero.
- Each request set `cache:false`, and the server was fresh before these runs.
- Rates are completion tokens divided by the time from the final prefill
  heartbeat to response completion. EOS counts in usage.
- TTFT is from the socket request to the first token ID.

Tokens per second, higher is better:

| Prompt, sampling | Plain | STRICT | FAST |
| --- | ---: | ---: | ---: |
| Code edit, greedy | 14.897 | 103.807 | 94.058 |
| Code edit, sampled | 14.677 | 94.077 | 84.034 |
| JSON edit, greedy | 14.892 | 124.275 | 110.167 |
| JSON edit, sampled | 14.679 | 109.487 | 96.743 |
| Quoted summary, greedy | 15.054 | 25.670 | 25.308 |
| Quoted summary, sampled | 14.848 | 26.971 | 26.516 |
| Original prose, greedy | 14.956 | 14.899 | 14.910 |
| Original prose, sampled | 14.748 | 14.640 | 14.622 |

TTFT in ms, lower is better:

| Prompt, sampling | Plain | STRICT | FAST |
| --- | ---: | ---: | ---: |
| Code edit, greedy | 122.487 | 191.794 | 192.553 |
| Code edit, sampled | 123.938 | 193.093 | 196.141 |
| JSON edit, greedy | 122.360 | 190.816 | 191.764 |
| JSON edit, sampled | 122.126 | 192.408 | 193.907 |
| Quoted summary, greedy | 99.852 | 170.838 | 169.404 |
| Quoted summary, sampled | 100.397 | 169.863 | 171.657 |
| Original prose, greedy | 95.319 | 162.464 | 164.017 |
| Original prose, sampled | 94.411 | 164.958 | 164.501 |

Device peak in bytes. Quoted summary and original prose had the same peak in
all three modes:

| Prompt, sampling | Plain | STRICT | FAST |
| --- | ---: | ---: | ---: |
| Code edit, greedy | 16,611,212,684 | 16,619,264,064 | 16,626,312,664 |
| Code edit, sampled | 16,611,212,684 | 16,619,264,064 | 16,626,312,664 |
| JSON edit, greedy | 16,611,212,684 | 16,625,526,236 | 16,626,312,664 |
| JSON edit, sampled | 16,611,212,684 | 16,625,526,236 | 16,626,312,664 |
| Quoted summary, greedy | 16,497,966,476 | 16,497,966,476 | 16,497,966,476 |
| Quoted summary, sampled | 16,497,966,476 | 16,497,966,476 | 16,497,966,476 |
| Original prose, greedy | 16,497,966,476 | 16,497,966,476 | 16,497,966,476 |
| Original prose, sampled | 16,497,966,476 | 16,497,966,476 | 16,497,966,476 |

Tokens in and out:

| Prompt, sampling | Input tokens | Output tokens |
| --- | ---: | ---: |
| Code edit, greedy | 162 | 125 |
| Code edit, sampled | 162 | 125 |
| JSON edit, greedy | 158 | 118 |
| JSON edit, sampled | 158 | 118 |
| Quoted summary, greedy | 126 | 60 |
| Quoted summary, sampled | 126 | 56 |
| Original prose, greedy | 51 | 128 |
| Original prose, sampled | 51 | 128 |

What the tables show:

<dl>
<dt>Same output</dt>
<dd>All 24 responses match plain decoding's token IDs, streamed text and
completion usage.</dd>
<dt>Memory</dt>
<dd>The HTTP model allocation is 16,383,135,744 B. Code and JSON finish with
150,994,944 B of KV; quotes and prose with 75,497,472 B. Device peaks include
brief overlap while a small cache grows, plus verification scratch.</dd>
<dt>Host memory</dt>
<dd>The process host RSS high-water for the whole HTTP run, including the later
cache checks, was 192,393,216 B. This is one run-level peak, not a per-row
figure.</dd>
<dt>FAST guard</dt>
<dd>FAST guards fired zero times on these prompts. That does not prove FAST
equivalent in general, or bound drift in its approximate KV.</dd>
<dt>Acceptance</dt>
<dd>STRICT code committed 110 drafted tokens in 15 cycles, JSON 106 in 12, and
the quoted summary 27 in 33 greedy cycles and 29 sampled cycles. Prose chose
serial width: 128 cycles, no proposed rows.</dd>
<dt>First token</dt>
<dd>STRICT holds its first token until its KV commits, which adds about one
decode step to the time to first token.</dd>
</dl>

## HTTP regression check

281-token prompt:

| Measurement | Plain | STRICT |
| --- | ---: | ---: |
| Cold exact replay, TTFT (ms) | 149.724 | 237.632 |
| Cached replay, TTFT (ms) | 1.100 | 163.467 |
| Follow-up, TTFT (ms) | 78.969 | 156.700 |
| Decode, 64 tokens (tok/s) | 14.906 | 15.895 |
| Foreground TTFT (ms) | 166.002 | 237.823 |

- Cold exact replay is one request; foreground TTFT was measured during a
  background prefill; decode produced 64 output tokens.
- These checks deliberately keep several branches. The combined regression
  run's device peak was 18,377,739,852 B, including cached sessions. The
  common HTTP host peak above also includes these checks.
- Both modes pass fresh-versus-cached output comparison, divergent branch
  isolation, foreground priority and disconnect cleanup.
- **Why STRICT's cached TTFT is slower.** STRICT completion snapshots clear
  stale logits, so exact-prefix reuse replays the final committed row.
  Together with commit-before-stream, that explains the cached TTFT
  regression.

## Correctness checks

<dl>
<dt>Build and tests</dt>
<dd>119 workspace tests pass. The native PTX and release workspace builds
succeed.</dd>
<dt>Verification</dt>
<dd>Every verification width from 1 to 16 matches serial decoding bit for bit,
over four repeats, at prefixes 0, 32, 63, 64, 2000 and 2040, with FP32 and BF16
KV. This includes crossings of the 64- and 2,048-position boundaries. STRICT
uses measured exact costs, not FAST kernels.</dd>
<dt>HF teacher forcing, FP32 KV</dt>
<dd>56 of 56 top-1, maximum saved-logit error 0.00964928. Short, medium and long
full-row maxima are 0.00040054, 0.00627232 and 0.00964928. The native FP32 check
matches every prompt top-1 decision and 32 of 32 continuation tokens. Appended
chunk errors are at most 0.00551987. The approximate-path threshold uses the HF
gate's 0.01; exact verification still compares bits.</dd>
<dt>HF teacher forcing, BF16 KV</dt>
<dd>Also 56 of 56, but the maximum saved-logit error is 0.22078133, and the
native long continuation matches only 11 of 32 FP32 tokens. BF16 KV stayed
opt-in on that build.</dd>
<dt>Load check</dt>
<dd>Two cycles, identical eight token IDs and scores, and zero tracked GPU bytes
after each unload. The first cycle's load took 2,333.131 ms; process start to
first token took 2,670.590 ms. Host peak 147,689,472 B; pinned load-staging peak
38,666,240 B. No host copy of the weights and no device conversion copy were
kept.</dd>
<dt>Cache and prefill checks</dt>
<dd>The cache check passes 10 exact full-logit comparisons. The prefill check
passes five comparisons and skips three prompt heads.</dd>
<dt>GPU API checks</dt>
<dd>They pass, including uneven-K exact verification, attention shards and zero
BF16 KV bit mismatches.</dd>
</dl>

## Exact verification timing

Final exact verification medians from `batch-bench ... 2048 0 verify`. Device
peaks include prompt preparation. Process host peak was 152,154,112 B.

| FP32 prefix | Rows | Verify (ms) | Device peak (B) |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 67.392 | 16,385,052,044 |
| 0 | 4 | 67.044 | 16,389,195,692 |
| 0 | 8 | 69.244 | 16,393,634,236 |
| 0 | 16 | 79.471 | 16,402,511,324 |
| 2000 | 1 | 70.210 | 17,102,050,720 |
| 2000 | 4 | 72.140 | 17,102,050,720 |
| 2000 | 8 | 78.186 | 17,102,050,720 |
| 2000 | 16 | 96.897 | 17,102,050,720 |

## Memory ledger

Native run with room for 2,144 positions:

| Item | Bytes |
| --- | ---: |
| BF16 weights, one copy | 16,380,854,272 |
| Model norms and RoPE tables | 2,330,624 |
| FP32 KV at 2,048 | 603,979,776 |
| Persistent decode state | 1,629,580 |
| Extra prefill scratch at 2,048 | 113,313,812 |
| Prefill device peak | 17,102,108,064 |
| FP32 physical KV at 2,112 | 754,974,720 |
| Decode device peak at 2,112 | 17,139,789,196 |
| Host RSS high-water, benchmark | 152,338,432 |

- **Page size.** The CUDA allocation granularity is 2,097,152 B per KV vector.
- **Growth.** From 2,048 to 2,112 positions, 72 vectors each mapped one more
  page, keeping their virtual addresses and old physical pages. There is no
  replacement arena.
- **Slack.** The final pages include 132,120,576 B beyond the 2,112-position
  logical FP32 payload.
- **Small contexts** use compact ordinary arenas. Snapshot forks copy the
  committed prefix when they change it.
- **Do not add scopes.** Host RSS and driver allocations overlap on unified
  memory. Do not add them up as one measured physical peak.

## What was left open

- Lower prefill projection time enough to pass 800 ms, while keeping the HF
  gate.
- Keep prompt-boundary snapshots and logits alongside committed completions,
  so STRICT cache hits avoid replaying the final row.
- Multi-Spark draft integration was not changed in this round. Back ends
  without a draft target kept the existing sequential path.

## Reproduce

These are the commands recorded for that build. On current main the same
programs exist, but they measure current defaults; label any new run with the
commit it measured.

### Ask your coding agent

```text
Re-run the measurements recorded in docs/history/fp32-cache-measurements.md on this DGX
Spark, using the "Reproduce" commands on that page. Stop any running Vade
server first so the GPU is idle. Run as the current user and never use sudo.
Report the commit you measured and every TTFT, decode, verification and
memory figure, plus whether every check passed. Label the results as a new
measurement of that commit, not as the historical build's numbers. Do not
commit or push.
```

### Do it yourself

1. From the Spark checkout, with Rust on `PATH`, set the paths and build. No
   outside crates or new reference-generation dependencies are needed.

   ```sh
   P=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
   M="models/Qwen/Qwen3-8B"
   cargo kernels
   cargo build --release --offline --workspace --exclude spark-rust-kernels
   cargo test --offline --workspace --exclude spark-rust-kernels
   ```

2. Run the native speed, correctness and memory checks.

   ```sh
   target/release/batch-bench "$P" "$M" 2048 64 bench
   target/release/batch-bench "$P" "$M" 2048 32 check
   target/release/batch-bench "$P" "$M" 2048 0 verify
   target/release/hf-batch-check "$P" "$M" fixtures/qwen3/{short,medium,long}.json
   target/release/hf-batch-check "$P" "$M" \
     fixtures/qwen3/{short,medium,long}.json --bf16-kv
   target/release/load-check "$P" "$M" 2112 2 8
   target/release/batch-bench "$P" "$M" cache
   target/release/batch-bench "$P" "$M" prefill
   target/release/batch-bench "$P" "$M" 2048 0 profile
   ```

3. Build the prompt tool and write the HTTP prompts.

   ```sh
   rustc --edition=2024 -C panic=abort tools/draft-prompts.rs \
     -L dependency=target/release/deps \
     --extern engine_tokenizer=target/release/libengine_tokenizer.rlib \
     -o target/release/draft-prompts
   target/release/draft-prompts "$M" /tmp/vade-fp32-prompts
   ```

4. In a separate terminal, start the server. Set `ADDRESS` to a loopback
   socket. Stop the server after the checks.

   ```sh
   target/release/spark-serve "$P" "$M" --bind "$ADDRESS" --output-tokens 256
   ```

5. On that fresh server, before creating any saved conversation snapshots,
   run the HTTP checks.

   ```sh
   target/release/serve-draft-check "$ADDRESS" /tmp/vade-fp32-prompts 128
   target/release/serve-http-check "$ADDRESS" 64 fixtures/qwen3/medium.json
   VADE_CHECK_DRAFT=strict target/release/serve-http-check "$ADDRESS" 64 \
     fixtures/qwen3/medium.json
   ```
