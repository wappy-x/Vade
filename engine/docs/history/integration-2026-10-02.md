# Historical integration record

> **Historical record, 2026-10-02.** This page records one past integration.
> Its launch settings, FP32 KV cache figures and older benchmark client are not
> today's serving defaults, and the page is not a scoreboard against other
> engines. For current results, see [Benchmarks](../benchmarks.md) and
> [Validation](../validation.md).

## In short

| Item | Result |
| --- | --- |
| Outcome | **Passed**, on every requested gate |
| What was merged | The must-fix branch (`mustfix`, `4c3b09a`) and the draft branch (`ldraft`, `41075a2` / `3664201`), onto `22510ca` |
| Model | Official Qwen3-8B, BF16 weights |
| KV cache | FP32 unless a row names another format |
| Weights in memory | One arena of **16,380,854,272 bytes** per process |
| Build | Release kernels and an offline, locked workspace build passed |
| Test code added | No new test files. The existing batch benchmark gained two lines, under 10% of the product lines added |
| Clean-up | The frozen run folders were not touched, and the GPU processes used for measuring were stopped |

Terms used on this page:

| Term | Meaning |
| --- | --- |
| Gate | A pass-or-fail check the candidate had to pass |
| KV cache | The model's memory of the conversation so far |
| TTFT | Time to first token: from opening the connection to the first answer text |
| STRICT | Draft decoding where the full model checks every guess, so output matches plain decoding |
| Peak device bytes | The largest total of requested GPU allocations. A ledger figure, not physical memory in use |
| Host HWM | Host high-water mark: the most CPU-side memory the process held |
| seqpar | Sequence-parallel prefill: one prompt split across processes |
| Peer | A helper process that reads part of a long prompt |
| Rank | One process in a split run. Local ranks share one GPU and take turns on it |

## What changed

- These stayed as they were: sizes taken from the checkpoint, the default
  GPU budgets, how peers are scheduled, recovery after a model swap, and exact
  arithmetic.
- From `mustfix` we kept two things: its measured starting estimate of
  drafting cost (the prior) and its shared cost scale. They replaced
  `ldraft`'s approach, which explored per-width costs only at the end points.
- Background verification now uses 32 rows.
- Shared workspace is now subtracted **before** the budget is split into
  active and retained pools. Subtracting it afterwards had stopped valid
  long-context servers from starting.

## Gate results

| Check | Result | Peak device bytes |
|---|---|---:|
| Hugging Face reference, FP32 and scaled 16-bit KV | 56/56 each; largest absolute logit error 0.00210190 and 0.00607681 | 17,076,569,488 / 16,774,317,456 |
| Exactness, FP32 / BF16 / scaled 16-bit KV | 96 cases each, four repetitions; zero bit mismatches | 17,161,095,628 / 16,859,105,740 / 16,925,837,200 |
| HTTP checks, plain and STRICT | Six cases each pass, including on the final revision of the memory-reservation code | 18,960,277,208 |
| 16 chats, 128 output tokens each | **197.557682 tok/s**, 10,366.592590 ms; all 29 replies at widths 1, 4, 8 and 16 match the token IDs from isolated runs | **17,947,975,104** |
| 28,672-token profile | **15,728.673 ms**, top token 198 | **25,257,528,928** |
| seqpar, two local ranks | Same eight token IDs as ordinary decode | Rank 0: 22,491,758,176; rank 1: 25,634,999,904 |
| Live model self-swap | Unload to 0 bytes; ready before and after at **16,689,469,696 bytes**; same 32 token IDs; 2,437.636 ms total | 17,947,975,104 over the process lifetime |
| STRICT code prompt, run 1 | **119.439 tok/s**, 125 tokens generated, matches plain | **16,623,580,620** |
| STRICT code prompt, run 2 | **119.147 tok/s**, 125 tokens generated, matches plain | **16,623,580,620** |
| STRICT code prompt, run 3 | **119.647 tok/s**, 125 tokens generated, matches plain | **16,623,580,620** |
| One 28,672-token HTTP request alone | **15,831.600 ms TTFT**, eight output tokens | **25,411,968,672** |
| A main chat sent 500.268 ms into that long request | **233.392 ms TTFT**, matches isolated token IDs | **25,422,449,900**, shared |

## Notes on the results

**STRICT stability.** In this three-run sample, the earlier STRICT
instability is fixed. All three runs beat the earlier 117.8 tok/s reference,
with a 0.500 tok/s spread and identical output. We have not checked stability
over longer runs.

**Long HTTP margin.** The lone long request passed with only **68.400 ms**
to spare against its 15.9 s limit.

- The first merged code missed, at 15,983.921 ms with a 25,411,968,672-byte
  peak.
- Skipping redundant full-reservation scans between prompt chunks produced
  the passing run.
- When a main chat interrupts it, the long request's TTFT is 16,211.901 ms
  at a 25,422,449,900-byte shared peak. The 15.9 s limit applies only to the
  lone request.

**Memory detail.**

| Run | Measured |
| --- | --- |
| 28,672-token profile | Model and RoPE table **16,396,787,712 B**; KV **8,455,716,864 B**; scratch **405,024,352 B**; host HWM **149,803,008 B** |
| 16-chat run | Host HWM **225,087,488 B**; pinned **4,194,304 B** |
| STRICT code runs | Host HWM at most **150,167,552 B**; retained device **16,541,712,844 B**; KV **150,994,944 B** |
| Long HTTP request | KV **8,606,711,808 B**; workspace **152,363,072 B**; pinned **16,777,216 B**; host HWM not sampled |

**Token IDs.**

- Profile and local-rank runs: `[198,198,26238,871,8563,8563,8563,8563]`.
- Long HTTP request, alone and interrupted, matching the prior candidate:
  `[785,6290,6733,374,88677,29388,264,14796]`.

Local ranks take turns on one GPU. These runs do not measure physical
multi-Spark performance.

## Cache doubling: reproduced and fixed

**The problem.** With caching on and no peers, the unmodified code reserved
memory for the full prompt.

| Step | Device bytes before the fix |
| --- | ---: |
| 8,300-token prompt, two plain output tokens (one decode step after prefill) | 21,657,299,596 |
| After the 8,318-token follow-up | 26,791,127,692 |

Both figures were gigabytes above the prior candidate build.

**The fix.** `DeviceState::reserve_prompt_cache` now reserves separate
**512-row physical blocks**, and it stops at the real end of the prompt.

- It skips capacity already reserved.
- Batch preparation detaches shared tails before writing.
- Complete blocks stay shareable at an unaligned end point.

**The result.**

| Current device bytes | Prior candidate | Final, no peer | Final, one peer | Final minus prior |
|---|---:|---:|---:|---:|
| After the first decode step | 19,118,080,076 | **19,241,380,492** | **19,241,380,492** | **123,300,416** |
| After the follow-up | 19,420,069,964 | **19,543,370,380** | **19,543,370,380** | **123,300,416** |

- Both differences are below **150,000,000 bytes**. They include
  **122,776,640 bytes** of newly admitted workspace.
- With and without a peer, the difference is **0 bytes**.
- Both runs reuse 8,296 tokens. Cold and follow-up token IDs: `[785,6290]`.
- All cached outputs match with and without a peer. Cached follow-ups match
  fresh output.

| Run | Cold / follow-up TTFT | Peak bytes, cold / follow-up |
| --- | ---: | ---: |
| Final, no peer | **3,672.071 / 175.863 ms** | 19,493,579,424 / 19,543,370,380 |
| Final, one peer | **4,571.468 / 169.908 ms** | Server 19,292,566,176 / 19,543,370,380; worker 18,309,643,872 |

An extra fresh follow-up comparison peaked at **22,513,478,304 bytes**
locally and **22,312,465,056 bytes** with a peer.

The authenticated `seqpar_prefill` message confirmed that the peer ran its
part. Over loopback, the transport fell back to TCP on its own, and the
server never fell back to reading the prompt by itself. Each process held one
copy of the weights. No distributed speed-up is claimed.

## Reproduce

This is a historical record, not a current procedure. To check today's
build, follow [Validation](../validation.md) and
[Benchmarks](../benchmarks.md), which have numbered steps and a block to
give your coding agent.

These are the commands used at the time. Set `P` to the built kernel file,
`M` to the official checkpoint, and `BIND`, `URL` and `PID` to a local server
you own. Raw logs stayed in temporary files on the measuring Spark; they are
not in the repository.

```sh
cargo kernels
cargo build --release --offline --locked --workspace --exclude spark-rust-kernels
target/release/hf-batch-check "$P" "$M" fixtures/qwen3/{short,medium,long}.json
# Repeat with --i16-kv.
for F in fp32 bf16 i16; do target/release/batch-bench "$P" "$M" 2048 0 verify "$F"; done
target/release/batch-bench "$P" "$M" 28672 8 profile fp32
SEQPAR_MODEL_DIR="$M" SEQPAR_SERIAL_COMPUTE=1 target/release/seqpar-launch --local 2 -- /tmp/seqmain-tokens-28672.txt 8
for I in 1 2 3; do target/release/draft-generate "$P" "$M" /tmp/integ-prose-prompts/code.ids 512 greedy 42 16 adaptive lookup strict bench; done
target/release/serve-http-check "$BIND" 64 fixtures/qwen3/medium.json
VADE_CHECK_DRAFT=strict target/release/serve-http-check "$BIND" 64 fixtures/qwen3/medium.json
tools/bench/target/release/vade-bench "$URL" --tokenizer "$M" --only concurrency --max-tokens 128 --repeats 1 --timeout-secs 180 --budget-secs 240 --server-pid "$PID" --output /tmp/concurrency.json
tools/bench/target/release/vade-bench "$URL" --tokenizer "$M" --only cache --cache-target 8300 --max-tokens 2 --repeats 1 --timeout-secs 180 --budget-secs 240 --server-pid "$PID" --output /tmp/cache.json
```

Server settings per run, as (context, input, KV MiB, session MiB, max
active, draft):

| Run | Settings |
| --- | --- |
| HTTP checks | (9216, 8192, 3072, 81920, 4, plain) |
| 16 chats and swap | (2048, 1536, 1024, 8192, 32, plain) |
| Cache | (32768, 30000, 12288, 49152, 2, plain) |
| Long request | (32768, 30000, 12288, 65536, 4, strict) |

- GPU budget was 102,400 MiB, except the 16-chat and swap runs, which used
  the checkpoint-derived default.
- Output limit was 512 tokens for the checks and the 16-chat run, 256
  otherwise. Timeout 180 s. Configured prefill chunk 512.
- Peer cache run: one authenticated worker, context 32,768, worker budget
  51,539,607,552 bytes.

How the special runs were driven:

- **Long request body:** `vade-bench --only request --long-target 28672 --long-output 8`.
- **Long request client:** a temporary Rust client reusing
  `tools/bench/src/http.rs` sent that body. A second request sent 500 ms
  later measured the interruption. TTFT counted to the first non-empty
  content, not to the headers.
- **Clean-up:** the temporary client and driver scripts were removed after
  measuring. Request bodies and results were kept on the Spark.
- **Swap:** a raw completion of `[1,42,1024,17]` with 32 plain output tokens,
  then `/v1/admin/swap` to `Qwen3-8B`, then the same request again, comparing
  tokens and health.

## Files changed

| Crate | Files |
| --- | --- |
| engine | `device.rs`, `qwen.rs` (shareable reservation, spans, sliced chunks); `multi.rs` (bounded workspace/C2); `draft.rs` (cost scale); `bin/batch-bench.rs` (boundary coverage) |
| gpu | `backend.rs`, `batch.rs`, `multi.rs` (spans and boundaries) |
| serve | `budget.rs`, `cache.rs`, `lib.rs`, `qwen.rs`, `scheduler.rs`, `bin/spark-serve.rs` (admission, which decides which requests may start; the cache; and prompt chunks) |
| docs | Existing READMEs, draft docs and this record |

## Not verified in this record

- Physical multi-Spark runs and RDMA.
- Longer stability runs.
- The imported 4,096- and 16,384-token boundary cases. The unchanged 2,048
  verify gate did not exercise them.
