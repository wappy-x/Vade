# Exact prompt recurrence

The typed hybrid GDN dispatcher combines consecutive live prompt rows only
when every descriptor has identical state/history addresses and no output
snapshot. `flash_gdn_scan_exact` consumes their token-major prepared inputs
in order. Independent sessions, snapshots and single-token spans retain
`flash_gdn_step_batch`.

One 128-thread block owns a head. Each thread loads its 128 FP32 state cells
once, retains them across the token loop, and stores them at tile end.
Every token retains the original decay multiply, ascending 128-term
prediction sum, delta, state updates and ascending 128-term output sum.
There is no reduction reassociation or approximate chunk transform.
FP32 state storage between tiles preserves every bit, so tile boundaries
need no fixed alignment. Preparation, history publication and gating keep
their existing arithmetic and layouts.

At 30 GDN layers and 256-token prompt tiles, 8,192 tokens require 960
recurrence launches rather than 245,760. This is a dispatch count, not a
measured speedup. Qualification compares every recurrence output and final
state bit at lengths 1/2/63/64/65/255/256 and arbitrary splits, and checks
compiled PTX for FMA contraction and ordered scalar dependencies. Native
qualification must also check physical register allocation, full model
bytes and synchronized timing on the target GPU.

Idea credit: FLA fused recurrent token loops and FlashInfer's SM12x fused
GDN step (PR 4708). The implementation uses our existing step arithmetic.

Flash-Next slot replay uses two warps per CTA, each owning 32 adjacent value
features. The 96 CTAs cover all 48 heads without changing any state address.
Constant-index loads and stores keep each feature's 128 cells in registers;
the shared ascending recurrence and MTP `output_state` destination stay intact.
A scalar FP32 instruction across a warp transfers 128 contiguous state bytes.
Per-thread vector loads would cross feature ownership in this fixed layout and
require extra shuffles; this path keeps the coalesced warp transfers instead.
Physical registers, achieved occupancy and speed require native qualification.
