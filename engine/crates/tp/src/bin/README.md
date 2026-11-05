# engine/crates/tp/src/bin

**Programs built from the `spark-tp` crate, one per file.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`flash-serve.rs`](flash-serve.rs) | Flash-Next rank service: authenticated HTTP on rank zero, command workers elsewhere. |
| [`session-state-bytes.rs`](session-state-bytes.rs) | Device allocation and opaque state-copy gate. No checkpoint or model arithmetic. |
| [`session-plan-check.rs`](session-plan-check.rs) | Conditional physical allocation and continuation gate at deferred plan dimensions. |
| [`tp-bench.rs`](tp-bench.rs) | TP throughput, strict verification and rollback using the real model. |
| [`tp-generate.rs`](tp-generate.rs) | Greedy generation with the same device engine used by batched TP. |
| [`tp-hf-check.rs`](tp-hf-check.rs) | TP prefill/decode qualification against saved independent HF fixtures. |
| [`tp-launch.rs`](tp-launch.rs) | Discover direct QSFP neighbours and supervise a distributed launch over SSH. |
| [`tp-ptx.rs`](tp-ptx.rs) | Compose an optional TP entry point with the untouched shared PTX module. |
| [`tp-serve.rs`](tp-serve.rs) | Existing HTTP service backed by rank-zero TP session commands. |
| [`tp-launch/`](tp-launch/README.md) | Folder: see its README. |

**Why a folder:** Cargo builds every file in a `bin/` folder as its own program.

`flash-serve` also serves the ModelOpt mixed FP4-Local checkpoint on one GPU.
Use the existing positional form with `RANK=0`, `WORLD=1`, a production PTX
file and the checkpoint directory. `--fp4-local` asserts the checkpoint type;
the mixed storage metadata also selects the Next backend automatically.
World 1 keeps the rows/MTP service. World 2 serves the same engine as the
two-rank tensor split, the only two-Spark path: it runs on the layer pipeline's
links and protocol (rank 1 starts at layer 2) and splits each routed expert by
columns, the projections and the linear-attention heads across both ranks, with
world 1's MTP default, queued prompt tiles, the RDMA wire, 2048-row prompt tiles
and the bounded head (`VADE_PROMPT_TILE`, `VADE_HEAD_BOUNDS`; `0` turns one off).
At world 2, flash-serve refuses `SEQPAR_RDMA=0` (the split needs RDMA) and
`--tensor-map`. See [Flash-Next NVFP4 serving](../../../../docs/next-http.md).
World 4 selects the pinned native rank backend with `--tensor-map PATH`; each
rank uses the same invocation and only rank zero binds HTTP. Other worlds
run the chain below.

On the FP8 rank path one chat chooses its MTP depth each cycle, up to resident
depth 7, from its measured verify times and draft acceptance. `VADE_FLASH_MTP_DEPTH`
(1..7) sets that ceiling; `VADE_MTP_ADAPT=0` or an explicit `--mtp` keeps the fixed
serving depth. Verification is strict, so the depth changes only speed, never tokens.
Its drafts score an MXFP8 copy of the rank's head rows (`VADE_FP8_DRAFT_MX8`, unset or
`1` = on, `0` = the BF16 head); verification always uses the BF16 head.
On a path of four or more FP8 ranks whose data ring runs RDMA, small rank
commands go through RDMA mailboxes, rank to rank, instead of the TCP chain. The
command bytes do not change, so tokens do not change. Two or three ranks and
TCP transport keep their usual command path.

`WORLD=3..24` without `--tensor-map` runs the layer pipeline as a chain, with
equal contiguous layer ranges by default. It needs a cable chain, not a closing
ring: `LISTEN` is the worker's local upstream interface and `NEXT` is its
downstream neighbor. Rank zero ignores `LISTEN`; the last rank ignores `NEXT`.
Middle ranks relay final residuals back on the same full-duplex cables. Do not
pass `--tensor-map` for this mode; `--mtp 0` selects plain decoding. Chains
require their own GPU exactness qualification; host protocol tests alone do not qualify them.

For FP4-Local the context defaults to 32768 and output is capped at
min(16384, context). FP8 keeps its existing defaults. The checkpoint position
ceiling and shared prompt-plus-output admission are enforced.
`SEQPAR_MEMORY_BYTES` optionally caps total device allocations. Startup reserves
safety headroom, weights and shared scratch, then admits up to eight independent
sessions within the remaining memory pool. It refuses startup if one fully funded
session cannot fit. Private admission covers target KV/index, GDN/PLE state,
prompt scratch, the full-context draft KV/index and all retained MTP prefixes.
Scratch growth releases its old allocation before replacement; a mixed-capacity
cohort charges its workspace to its largest-context session.

The world-one rows service batches independent decode and verifier rows. HTTP prompt slices
are capped and funded at 512; native sessions admit chunks up to 2048. MTP defaults
to strict depth 2, with resident depth 3. `--mtp 0..3` sets the default, and the
Next-only `mtp_depth:0..3` field selects each request's depth on Completions, Chat,
Responses and Messages. Zero selects plain decoding. Invalid or nonresident
depths are rejected before admission. Sampled requests use plain decoding.
Verifier output is projected in slices of at most 16 rows, preserving each row's
arithmetic through the maximum native 16-chat/depth-3 cohort (64 rows).

The rank backend (world 4) defaults to plain decoding and refuses startup `--mtp 1..3` or
`MTP=1..3`. All four HTTP APIs reject `mtp_depth > 0` before admission; zero and
an absent depth select plain. The shared Next-only depth validation leaves FP8
request policy unchanged. Context is fixed at 32768 and output is capped at
16384. Each rank must fund all eight full-context native session reservations.
The native backend preserves its scalar C1 and fixed C2..C8 rank cohorts, ring
collectives and per-world arithmetic. Requests require penalty-free greedy
sampling. See [rank serving](../../../../docs/next-http.md) for the launch form.

Completed/error world-one sessions reset QSA, GDN and PLE/hash state before reuse. Prefix
caching is disabled. Non-final prompt tokens skip the output head/argmax. Sparse
indexing matches the checkpoint beyond 2048 tokens; `--indexer dense` is refused
by this service. HTTP, SSE, templates/thinking, EOS/stops, sampling, usage and
context validation use the shared server. Nonfinite output fails its request;
sticky device faults follow the shared fatal-error handling. GPU byte and timing
qualification remain a separate gate. See [rows admission](../../../../docs/flash-next-rows.md).

**How it fits:** see the [`spark-tp` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../../docs/README.md)
# Plain FP32 probe

[`ordered-projection-check.rs`](ordered-projection-check.rs) checks isolated logical FP32 projection bytes through retained checkpoint files.
The [contract](../../ORDERED-PROJECTION.md) states its proof limits.
`checkpoint-prefix-check.rs` joins real stored embedding rows to attention mHC and input norm.
See [the prefix cell](../../CHECKPOINT-PREFIX.md) for its primitive scope and remaining decoder cells.
