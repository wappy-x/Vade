# Flash-Next NVFP4 serving

Start at [INSTALL.md](../../INSTALL.md) for system and compiler prerequisites.
This page describes the native **FP4-Local** checkpoint route. The FP8
`vade up` deployment has different defaults.

## One Spark and the two-Spark pipeline

`flash-serve --fp4-local` selects the native model. With one rank, a loopback
launch from a built checkout has this shape; set `PTX` and `MODEL` to the
verified artifact and checkpoint paths:

```sh
flash-serve 0 1 127.0.0.1:0 127.0.0.1:0 "$PTX" "$MODEL" 127.0.0.1:8080 0 --fp4-local
```

For two ranks, `VADE_W2_PP` defaults to enabled and selects a layer pipeline.
Launch each rank with its own control address and its peer's address, the same
fresh `TP_RUN_ID`, and the same `SEQPAR_CONTROL_SECRET`. Non-loopback control
requires that secret. The pipeline uses the world-one execution path across
layer partitions; do not supply `--tensor-map` for this route.
`VADE_W2_PP=0` selects the separate rank backend described below.

Current defaults are defined by the
[launcher](../crates/tp/src/bin/flash-serve.rs),
[drafting settings](../crates/engine/src/flash/draft_stack.rs),
[cohort policy](../crates/serve/src/mtp_batch.rs) and
[prompt scheduling](../crates/engine/src/flash/next_rows.rs):

- Context defaults to 32768 and output is capped at 16384, subject to admission.
- Startup/request MTP depth defaults to 3. The resident draft ceiling is 5 on
  one Spark and 7 on the pipeline. The adaptive chooser is enabled by default;
  request depth and resident ceiling are distinct. `--mtp 0` selects plain decoding.
- Automatic cohort drafting covers up to five active chats; larger cohorts use
  pooled plain steps. `VADE_MTP_BATCH` and `VADE_MTP_C8` are explicit overrides.
- Local prompt tiles are automatic: shorter prompts use 4096 rows, while
  prompts ending at or beyond 8192 tokens use 8192 rows. The pipeline defaults
  to 2048-row tiles. `VADE_PROMPT_TILE` can select a fixed supported tile size.
- Prompt lookup is controlled separately by `VADE_MTP_LOOKUP`; startup with
  MTP disabled also disables lookup.

These are source-defined defaults, not performance measurements. Verify the
loaded checkpoint, effective settings and health identity before timing.
The [checkpoint and kernel reference](fp4-kernels.md) describes storage.

## Native rank backend

The following world-two/four backend is separate from the default two-rank
pipeline. Its retained qualification is incomplete; do not generalize the
pipeline's behavior to it. Worlds expose the same lowercase checkpoint-folder
model ID, with world and topology as separate metadata. `next-http` selects
this backend when `VADE_W2_PP` is unset.

`flash-serve` owns startup, authentication, checkpoint selection, request parsing,
templates, SSE and prompt/output admission for every world. Mixed FP4-Local
metadata selects world-one rows/MTP or the world-2/4 native rank backend.
The rank-local arena, session schedule, scalar C1 and C2..C8 cohorts are retained
from the pinned world path. `next-tp` remains the native capture command.

Every rank needs the same fresh `TP_RUN_ID`, authenticated TP control secret and
production binary/PTX. Use the existing RDMA transport and `TP_TOPOLOGY=ring`
for the W2 or W4 ring. Set topology explicitly; W4 otherwise inherits the existing
path default. Invocation:

```text
flash-serve RANK WORLD LISTEN NEXT PTX MODEL 32768 BIND REQUESTS --tensor-map TENSOR_MAP --mtp 0 [--allow-remote] [--api-key-file PATH]
```

Only rank zero binds HTTP. `REQUESTS=0` serves continuously. World-2/4 context is
fixed at 32768; `FLASH_SERVE_OUTPUT_TOKENS` defaults to 16384 and accepts
1..=16384. Remote HTTP requires `--allow-remote` and an API key. Checkpoint,
build, PTX, world and topology metadata use the same loaders as world one.
Wait for checkpoint hashing to become ready before timing.

The native rank backend defaults to plain decoding and refuses positive
startup MTP depth. Its request `mtp_depth` must be zero. The one-Spark and
pipeline routes accept request depths 0..3, bounded by the resident model;
the adaptive policy described above may propose more tokens for eligible
requests. Invalid field types and unsupported depths fail before admission.
Other engines ignore this Next-only field in the translated API paths.

`next-http` is a compatibility launcher that execs the sibling `flash-serve`;
it has no request parser or serve loop. Its original positional form and
`NEXT_TP_HTTP=1` requirement remain; `OUTPUT` maps to
`FLASH_SERVE_OUTPUT_TOKENS`. It supplies context 32768, `--tensor-map` and
`--mtp 0`. Install both binaries together. Worlds 2/4 retain the pinned native
greedy defaults even when checkpoint metadata requests sampling. World one and
FP8 retain their checkpoint generation defaults. Qualification requests send
all greedy fields explicitly.

Worlds 2/4 refuse `--verify-full` and `FLASH_VERIFY_FULL=1` before rank startup:
the native shard loader does not implement full payload verification. Use world
one when full verification is required. Normal metadata/header verification and
rank artifact agreement remain enabled.

The shared OpenAI-compatible HTTP/JSON/SSE scheduler admits eight full-context
sessions, queues in FIFO order, and rotates one causal token per turn. Prompt
work also yields after one token. Cancellation and disconnect destruction send
CLOSE to every worker before a slot can be reused; new IDs never reuse state.
The adapter validates a whole cohort before any mutation.

Only penalty-free temperature-zero generation with plain drafting is accepted;
unsupported sampling/drafting fails before admission or a rank command. The
prompt adapter supplies finite selection rows from native greedy IDs. Cache
retention and MTP are disabled. A failed rank command poisons the controller.

Each rank must fund eight reservations: config-derived full-context KV and
recurrent bytes plus 512 MiB per session for M1 scratch and alignment. Startup
uses the lowest available rank budget, then caps device allocations to this
aggregate pool plus loaded weights. This conservative sizing is host-only;
actual GPU allocation and serving qualification remain outstanding.

## Replicated dense compute with world-two/four expert split

Set `VADE_FLASHNEXT_FP4_EP2=1` on **every** rank of `flash-serve` or `next-tp`.
The legacy switch name also enables W4 EP; no new arithmetic switch is needed.
Unset or `0` preserves the original layout and packet bytes. Other values and
worlds are rejected. Serving accepts only unset or `NEXT_TP_SPLIT=ep`; the
unqualified `ep2tp2` split is rejected before tokenizer, listener or GPU setup.
The world-one `flash-serve` Next adapters do not implement this layout.

For 512 routed experts, rank zero owns IDs 0..255 and rank one owns 256..511;
odd counts use contiguous balanced ranges, giving the extra expert to rank zero.
At W4 the ranges are 0..127, 128..255, 256..383 and 384..511. Remainders
go to the earliest ranks. Each expert's containers stay intact on one rank.
All packed values, scales and scalars follow their expert. Attention, GDN,
shared experts, router, HC and PLE projections replicate. Embedding/PLE disk
rows and the vocabulary head stay sharded. Startup funds the replicated caches.

Every rank sums its selected routed experts in ascending global ID, starting
from FP32 zero. One rank-major gather of H FP32 values implements the all-reduce:
copy rank zero, add ranks 1 through WORLD-1 in ascending rank order, then add
the replicated shared expert once. Every rank executes exactly that FP32 fold;
transport arrival order never determines arithmetic. This is one reduction per
layer at W2 and W4, including status and route agreement in the same gather.
Only then run HC/residual combination; no collective/normalization fusion.
The packet also carries one status float and the sorted top-k IDs/weights.
Local attention errors propagate through that same gather so peers fail together.
At H=2560/top-10, each contributed packet is 10,324 bytes/one collective,
versus 30,808 bytes/two collectives for the original native W2 adapter. These
figures exclude link framing and the per-token front/head/control packets.
W4 gathers three peer packets per rank (30,972 received payload bytes/layer);
ring forwarding and the two direct neighbor rails use the existing transport.
Startup discovery only reads interfaces and addresses; it never configures them.

Record weights, binary/PTX, topology and settings with every golden set.
Sum order can differ across these backends; same-setup repeatability alone does
not establish cross-topology equality. Compare the required topologies explicitly. `next-tp` case files
remain `NAME PROMPT_CSV EXPECTED_CSV`; expected IDs normally come from that
setup's frozen set. For a **comparison** with the one-Spark CPU reference, set
`NEXT_TP_CPU_REFERENCE=1` on every rank: receipts report agreement counts and
retain generated IDs, full trace and routes without gating on CPU token equality.
This comparison does not qualify determinism. Repeat each mode three times and
compare same-rank full `.f32`/`.routes` files and generated IDs byte for byte.
Rank-local vocabulary rows differ; compare each rank with its own prior capture.

For timing, start fresh paired `flash-serve` processes in alternating mode order
0,1,0,1,0,1 (same binaries/PTX, fresh `TP_RUN_ID`, ring, MTP off). After each
start run the existing client through the rank-zero loopback tunnel:

```sh
vade bench --url "$URL" --model flash-next-nvfp4 --prompts "$PROMPTS" \
  --concurrency 1,8 --max-tokens 256 --repeats 3 --output "$RECEIPT"
```

The client sends all greedy parameters explicitly. Retain effective sampling,
total/per-chat tok/s, bytes per rank, medians/ranges and identical-answer checks
from the private receipts. Overlapping timing ranges are a tie. GPU qualification
and the separate full-row determinism gate remain required before deployment.

## Integrated rank rows

W2/W4 use rank rows for cohorts of 2..=8 and the existing scalar path for one chat. `VADE_FLASHNEXT_WORLD_ROWS` is retired; it has no effect. The HTTP scheduler
admits eight foreground/background rows. Native capture applies the same rule
with `NEXT_TP_COHORT=1|8`. Each row retains ascending expert and rank folds.
Native GPU validation is required for each supported world and for comparisons
across worlds; host coverage alone is insufficient.

QSA/GDN layer objects and their plans stay with each chat, including across reset. Reset clears recurrence and cache cursors while retaining allocation
owners. Disk embedding/PLE descriptors and PLE global scale are cached once;
bounded reads use the same local ownership and CPU dequantization as before.
One expert cohort workspace is funded in each rank arena; per-chat KV, recurrence and conservative 512 MiB scratch reservations remain separate.
Full native captures use buffered writes, flushed with errors at every logits
boundary, to avoid one filesystem write per float.

This rank backend uses native M16 expert tensor cores. Multi-token prompt tiles, batched frontend uploads, device
frontend and per-width graphs are not ported: the native adapter's sharded
frontend and host rank collectives require a rank-aware runner before the
world-one pool can be reused. The M32 prefill TC kernel needs those prompt
chunks; no performance or GPU equivalence claim is made for this integration.
