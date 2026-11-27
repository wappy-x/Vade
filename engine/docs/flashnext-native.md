# Flash-Next native MXFP8 decode

`VADE_FLASHNEXT_FAST_PROJ=1` selects native E4M3 weights with one UE8M0 scale per
32 K elements for all 384 mixed-checkpoint projections: GDN, full attention
Q/K/V/O, indexer QK, and shared experts. `0` (the default) retains the original
expanded BF16 path. GPU determinism, route-aligned CPU accuracy and timing
must qualify before its default can change. The text-only filter excludes vision and MTP. There are no new dependencies or activation/KV formats.

The projection schedule ports the selected first-party MXFP8 prototype. Lane l accumulates coordinates 256*t+8*l+j in ascending t/j using
FP32 FMA. The warp reduces with shfl-down distances 16,8,4,2,1. Output rows have
exclusive writers. M1 uses packed pair conversion and row+8 prefetch. General
M2..8 uses the port-mxfp8-gemm retained fixed eight-row reuse and row+8 prefetch;
larger general batches use the bit decoder. Rounded GDN uses
the pair decoder at every M, as in `gdn-layer-fused`. Ordinary projections
preserve FP32 inputs: rounding them to BF16 discards the residual retained by
the expanded path's hi/lo operands. Only the explicit rounded GDN contract
(`round=1`) rounds FP32 inputs and outputs; packed BF16 inputs read their bytes
directly. Scale application still precedes FP32 FMA, with the same K order.

The fused single-row GDN experiment was removed after its norm audit failed.
The reference and row adapters retain their original recurrent schedules.

Rounded GDN and already-BF16 projection inputs retain the selected prototype's
byte targets. General FP32 projection inputs intentionally retain more precision
than that prototype; they require the route-aligned CPU gate against PLAIN.
The SIMT reduction differs from the expanded tensor-core reduction order. No product speedup or full-model
quality result is implied by a Mac build.

GPU tests in `spark-gpu-probe --test next_decode` require original-prototype
fixtures through `VADE_NEXT_FIXTURES`; the ignored tests fail if invoked
without them. They compare all projection output bytes for M1..16,
both projection input representations, descriptor phases and repeat runs. A fixture adapter and frozen original sources accompany
the builder report outside the product tree. No foreign CUDA source is linked
into the product; all new entries are Rust NVPTX.

The existing `spark-engine --test flashnext_gpu` gate retains CPU-reference
tolerance checks, adds full-vocabulary repeat/chunk byte comparisons, and can
write full logits to `VADE_FLASHNEXT_LOGITS` for comparisons across fresh
processes. Timing prints `ms_per_token` after prefill, without debug recording.
Use a full-model GPU box for both arms, running bytes before timings. Keep raw
logits on that box. The loader reports native and expanded storage separately.

The retained port uses row-major weight/scale bytes. CPU accuracy uses normwise
BF16 tolerance; near-zero componentwise relative error is diagnostic, while
prototype identity compares bytes.

Packed expert decode uses the GPU-proven inflight variant 3: two register batches,
L2 lookahead and two warps per CTA. The ascending K16 recurrence is unchanged.
Variants 1/2, the experiment selector and the old compact entries are absent.
Every compact task uses the same variant-3 geometry; wider eligible prompt work
retains its distinct M32/N128/K64 tile. Credit: the supplied llama.cpp GB10 study
for the prefetch idea; implementation is first-party.

MTP verification scans every chat's causal rows in one GDN launch per layer.
The verifier retains the 128 recurrent cells per value column in registers and
uses the same ascending `next_math::column` arithmetic as plain decoding. The
entry matrix is written to one reusable snapshot during its initial load.
A partial accept rebuilds the accepted convolution history from the tape in
one launch over all GDN layers. In most forms it runs no GDN scan again.
Small QKV/a/b projection tapes still restore the forms that keep the replay:
a window over 8 rows that crosses a chunk end with an early accept, and the
chunk-off form.
PLE history and draft hidden prefixes remain separately retained. No dense,
attention, expert or head projection is repeated during rollback.

For depth 2, retained state/tapes/prefixes total 124,838,400 bytes per chat,
versus 358,662,144 bytes for three complete prefixes. The commit table adds 96,768 bytes at that depth.
Recurrent matrix traffic falls from six matrix transfers to three: entry load,
entry snapshot store and final live-state store. A window inside one chunk
keeps the live state and rebuilds its history. A short window over a chunk
end whose accept stops before the chunk end swaps entry/live owners and
copies the accepted tail slots back. An accept through the chunk end keeps
the live state. Only the
forms above replay GDN rows. Accepting the complete verifier runs no commit work. Admission funds the single snapshot, maximum-depth
tapes, the copy table and every small prefix, with padding.

`VADE_DECODE_PROFILE=gpu` enables the existing event collector around target
verification, including preparation and argmax readback. `next_rows_profile`
reports dense, GDN, attention, routed experts, head and other GPU nanoseconds,
plus host launch/wait API time, enqueue/gap envelopes and total wall time. Host envelopes overlap GPU
work and include profiling overhead; they must not be added to GPU times. Dense
includes shared-expert projections. Timings for speed decisions use profiling
OFF; stage diagnosis uses profiling ON separately. The public backend wrapper
also allows a measurement caller to attribute plain row steps with the same
classes. No arithmetic or serving default changes with profiling.
