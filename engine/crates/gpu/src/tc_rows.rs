//! Tensor-core BF16 multi-row product (`PLAIN_TC_ROWS`, kind 103) and device-side MoE route grouping
//! (`MOE_GROUP_ROUTE`, kind 104) for Inkling 2 Mini.
//!
//! Pure layout, admission, and host checks shared between host and `kernels-inkling` (no dependency on
//! `kernels/`, keeping Qwen and GLM byte-for-byte untouched).
//!
//! Determinism contract:
//! - Every output element `(m, n)` is accumulated along `k = 0, 16, 32, ..., cols - 16` in strictly
//!   ascending K order using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` from `+0.0`.
//! - Each FP32 activation is rounded once to BF16 (round-to-nearest-even via `cvt.rn.bf16x2.f32`).
//! - Zero FP atomics, zero split-K, and padded tile lanes never affect valid rows or columns.
//! - `MOE_GROUP_ROUTE` partitions `0..rows*picks` into 8 fixed slices and scatters in strictly
//!   ascending pick order per expert with zero atomics.

/// Threads per block of `PLAIN_TC_ROWS` (8 warps: 2 M-warps x 4 N-warps).
pub const TC_THREADS: usize = 256;
/// Token rows (M), weight rows (N), and K columns per CTA tile.
pub const TC_BM: usize = 64;
pub const TC_BN: usize = 128;
/// N columns per CTA for the BF16 product. The packed-weight product uses TC_PACKED_BN.
pub const TC_PLAIN_BN: usize = 64;
/// Stored rows (N) per CTA of the packed MoE product (`PLAIN_PACKED_TC_ROWS`); each block takes two route tiles
/// (`2 * TC_BM` input rows) of one expert.
pub const TC_PACKED_BN: usize = 128;
pub const TC_BK: usize = 32;
/// Input rows (M) and weight rows (N) per CTA of the big-tile dense product (`PLAIN_TC_BIG`, prefill only).
pub const BIG_BM: usize = 128;
pub const BIG_BN: usize = 128;
/// The fewest input rows and weight rows of a dense product that takes `PLAIN_TC_BIG` (smaller products, every decode
/// product among them, stay on `PLAIN_TC_ROWS`). Lowered to 257 so Cold 8K tail chunks (`M = 388`) also run `PLAIN_TC_BIG`.
pub const TC_BIG_INPUTS_MIN: usize = 257;
pub const TC_BIG_ROWS_MIN: usize = 512;
/// Mode word of a `PLAIN_TC_BIG` command for a narrow product (at most `NARROW_BN` weight rows): the input is the FP32
/// rows (one input, or the two halves of a pair), rounded with `cvt.rn.bf16x2.f32` as they are read (the rounding of
/// `TC_MODE_PACK`), so no pack launch runs. `NARROW_BM` input rows a block.
pub const TC_MODE_NARROW: u64 = 3;
pub const NARROW_BM: usize = 16;
pub const NARROW_BN: usize = 128;
/// Mode word of a `PLAIN_TC_BIG` command for fused prefill `GateOuter (GateForget+GateGain) + GateBlend + RoundedNorm`
/// across 1..=3 gates (`grid = (count, rows.div_ceil(GATE_OUTER_BM), 1)`). Eliminates `c.round(&inner)`, `TC_MODE_PACK`,
/// the `[rows, 2 * dim]` intermediate gate buffer, and the separate `STREAM_OP_GATE_BLEND_NORM` launch.
pub const TC_MODE_GATE_BLEND_NORM: u64 = 4;
/// Mode word of a `PLAIN_TC_BIG` command for narrow BF16-input products (`rows <= NARROW_BN`, single or dual half-width
/// packed BF16 inputs): `NARROW_HALF_BM = 64` rows/CTA, 4 `cp.async` stages (`49,152 B` exact), `ldmatrix_x4` on A & B.
pub const TC_MODE_NARROW_HALF: u64 = 5;
pub const NARROW_HALF_BM: usize = 64;
pub const GATE_OUTER_BM: usize = 8;
pub const GATE_OUTER_K_MAX: usize = 64;
/// Maximum hidden width supported by `TC_MODE_GATE_BLEND_NORM` (`11` SMEM chunks + `8` register chunks = `19 * 128`).
pub const GATE_OUTER_DIM_MAX: usize = 2432;
pub const GATE_OUTER_GATES_MAX: usize = 3;
/// Payload word offset of the `TC_MODE_GATE_BLEND_NORM` gate table (6 words/gate: weights, inner, kept, msg, norm_w, out).
pub const GATE_OUTER_TABLE_AT: usize = 16;
/// Bit 0 of payload word 8 in `TC_MODE_GATE_BLEND_NORM`: write packed BF16 outputs (`rows * dim / 2` u32 words).
pub const GATE_OUTER_HALF_OUT: u64 = 1;
const _: () = assert!(8 * (GATE_OUTER_TABLE_AT + 6 * GATE_OUTER_GATES_MAX) <= 1024);
/// Columns must be multiples of 64 (whole stages) up to 8,192.
pub const TC_STAGE: usize = 64;
pub const TC_COLS_MAX: usize = 8192;
/// Maximum weight rows per expert matrix in MoE / packed-unit forms.
pub const TC_ROWS_MAX: usize = 16_384;
/// Maximum weight rows for dense Mode 0 (`PLAIN_TC_ROWS` / `PLAIN_TC_BIG`), covering full LM heads in one launch.
pub const TC_DENSE_ROWS_MAX: usize = 262_144;
/// Small-batch dense tile for `PLAIN_TC_ROWS` Mode 0 (`inputs <= TC_SMALL_BM`): wave-matched 1/2/4-warp tiles.
pub const TC_SMALL_BM: usize = 32;
pub const TC_SMALL_BN: usize = 128;
/// Single-warp small-N dense tile constants (`PLAIN_TC_ROWS`).
pub const TC_TINY_THREADS: usize = 32;
pub const TC_TINY_BN: usize = 8;
pub const TC_TINY_ROWS_MAX: usize = 2560;
pub const TC_INPUTS_MAX: usize = 4096;
/// Maximum split output buffers in one fused dense projection (e.g. Q+Rel+Sink+Gate+K+KTap+V+VTap).
pub const TC_SPLIT_MAX: usize = 8;
/// Maximum grouped narrow dual-input matrices in one launch (`GateDown` Q/K/V).
pub const TC_GROUP_MAX: usize = 3;
/// First payload word of the extra grouped matrix triples `(weights, input1, input2)` for groups 1..3.
pub const TC_GROUP_WORD: usize = 16;
/// Mode word of `PLAIN_TC_ROWS`: dense (packed BF16), routed MoE, FP32-to-BF16 pack, or direct-FP32 dense.
pub const TC_MODE_DENSE: u64 = 0;
pub const TC_MODE_MOE: u64 = 1;
pub const TC_MODE_PACK: u64 = 2;
pub const TC_MODE_DENSE_FP32: u64 = 4;
/// Payload word 13 flags for `TC_MODE_DENSE_FP32`.
pub const TC_FLAG_ROUND_OUT: u64 = 1;
pub const TC_FLAG_GROUP_PAIR: u64 = 2;
/// Bit 2 of payload word 13 in `TC_MODE_DENSE_FP32`: narrow 1-warp path reads packed BF16 input(s) directly.
pub const TC_FLAG_HALF_IN: u64 = 4;
/// Bit 3 of payload word 13 of a dense `PLAIN_TC_BIG` command with one output: the half form of the output. Word 2
/// holds `inputs * rows` BF16 values, 16 bits each (`rows` is even): `fixed_math::rn` of each product word. It is for
/// a reader that rounds each word it reads (the convolution after attention, `fast_ops::SHORT_CONV_HALF_IN_WORD`):
/// that reader makes the same words from half the bytes. The launch also writes the last `taps` input rows, as they
/// are, to the ring after the step (`TC_BIG_RING_WORD`, `[taps][rows]` FP32): input row `m` to slot
/// `(first + m) mod taps`. Bits 8..16 of word 13 are the slot of input row `inputs - taps`, the first row that goes
/// to the ring (so the launch has no division), and bits 16..24 are `taps` (`tc_big_half_out_word`).
pub const TC_BIG_HALF_OUT: u64 = 8;
pub const TC_BIG_RING_WORD: usize = 14;
/// Bit 63 of the meta word of a split output of a dense `PLAIN_TC_BIG` command (`start | len << 32`): the half form
/// of that output. It holds `inputs * len` BF16 values, 16 bits each: the pair rounding (`cvt.rn.bf16x2.f32`) of each
/// product word, for a reader that makes this rounding of each word it reads (`flash_window::FW_QG_HALF`).
pub const TC_SPLIT_HALF_OUT: u64 = 1 << 63;
#[inline(always)]
pub const fn tc_big_half_out_word(first: usize, taps: usize, inputs: usize) -> u64 {
    TC_BIG_HALF_OUT | ((((first + inputs - taps) % taps) as u64) << 8) | ((taps as u64) << 16)
}
/// Whether a dense product with a packed BF16 input takes `PLAIN_TC_BIG` and can write the half form of its output.
#[inline(always)]
pub const fn tc_big_half_out_admitted(rows: usize, cols: usize, inputs: usize, first: usize, taps: usize) -> bool {
    tc_shape_admitted(rows, cols, inputs) && tc_big_wanted(rows, inputs) && !tc_narrow_wanted(rows, inputs) && !tc_narrow_1w_wanted(rows, inputs)
        && rows % 2 == 0 && taps >= 1 && taps <= 255 && taps <= inputs && first < taps
}
/// Payload word offset of optional lossless Vade-BF11-BP `bf16_panels` descriptor `(panels_base, first_band | (slices << 32))`
/// for dense `PLAIN_TC_ROWS` Mode 0 commands (past the 8-segment split table at words 16..32; zero = raw BF16).
pub const TC_PANEL_WORD: usize = 32;
const _: () = assert!(16 + 2 * TC_SPLIT_MAX <= TC_PANEL_WORD && 8 * (TC_PANEL_WORD + 2) <= 1024);

/// Narrow 1-warp deep-pipelined direct-FP32 tile (`BM=8, BN=4, BK=64`: 8 stages of `4 x 144 B` padded W = 4,608 B).
pub const TC_N1_BM: usize = 8;
pub const TC_N1_BN: usize = 4;
pub const TC_N1_BK: usize = 64;
pub const TC_N1_INPUTS_MAX: usize = 256;
pub const TC_N1_SMEM: usize = 8 * 4 * (TC_N1_BK * 2 + 16); // 4,608 bytes
/// Narrow 4-warp (`rows <= 64`, `BM=32, BN=64`) and 8-warp (`64 < rows <= 128`, `BM=32, BN=128`) prefill direct-FP32 tiles.
pub const TC_N4_BM: usize = 32;
pub const TC_N4_BN: usize = 64;
pub const TC_N4_THREADS: usize = 128;
pub const TC_N4_SMEM: usize = 4 * (TC_N4_BM * TC_BK * 4 + TC_N4_BN * TC_BK * 2); // 32,768 bytes
pub const TC_N8_BN: usize = 128;
pub const TC_N8_SMEM: usize = 3 * (TC_N4_BM * TC_BK * 4 + TC_N8_BN * TC_BK * 2); // 36,864 bytes
/// Short-K (`cols == 64`) direct-FP32 single-shot tile (`32x32` 64t = 8,192 B; `64x64` 256t = 17,408 B w/ v4 epilogue).
pub const TC_SHORT_SMALL_SMEM: usize = 2 * (32 * TC_BK * 2 + 32 * TC_BK * 2); // 8,192 bytes
pub const TC_SHORT_BIG_SMEM: usize = 64 * 68 * 4; // 17,408 bytes (>= 16,384 B for A+B, bank-conflict-free 64x64 FP32 epilogue)
/// Small-M wide `BK=64` Weights-as-A tiles (`evict_first` / L2-cached register-prefetched W -> 1-stage private W SMEM + 2-stage X SMEM):
/// - 1-warp `16x16x64`: 2KB W + 4KB X = 6,144 B (14 blk/SM)
/// - 2-warp `16x32x64` (`inputs <= 16`): 4KB W + 4KB X = 8,192 B (10 blk/SM)
/// - 2-warp `32x32x64`: 4KB W + 8KB X = 12,288 B (7 blk/SM -> all 205/269 QKV blocks in 1 wave!)
/// - 4-warp `16x64x64` (`inputs <= 16`): 8KB W + 4KB X = 12,288 B (7 blk/SM)
/// - 4-warp `32x64x64`: 8KB W + 8KB X = 16,384 B (5 blk/SM = 20 warps/SM on LM Head!)
pub const TC_W1_M16_SMEM: usize = 16 * 64 * 2 + 2 * (16 * 64 * 2); // 6,144 bytes
pub const TC_W1_M32_SMEM: usize = 16 * 64 * 2 + 2 * (32 * 64 * 2); // 10,240 bytes
pub const TC_W2_M16_SMEM: usize = 32 * 64 * 2 + 2 * (16 * 64 * 2); // 8,192 bytes
pub const TC_W2_SMEM: usize = 32 * 64 * 2 + 2 * (32 * 64 * 2); // 12,288 bytes
pub const TC_W4_M16_SMEM: usize = 64 * 64 * 2 + 2 * (16 * 64 * 2); // 12,288 bytes
pub const TC_W4_SMEM: usize = 64 * 64 * 2 + 2 * (32 * 64 * 2); // 16,384 bytes
/// 4-warp cooperative-W `BN=16, BK=64` tiles (`m_tiles=1`, 2-stage W = 4,096 B, 2-chunk-ahead W in regs):
/// - `17..=32` (`O_proj`, `K/V`): 4KB W + 2x4KB X = 12,288 B (7 blk/SM; 12 warps/SM on O_proj)
/// - `33..=128` (`M=125` prefill on `QKV`, `O_proj`, `K/V`): 4KB W + 16KB X = 20,480 B (4 blk/SM = 16 warps/SM)
pub const TC_COOP_M32_SMEM: usize = 2 * (16 * 64 * 2) + 2 * (32 * 64 * 2); // 12,288 bytes
pub const TC_COOP_M128_SMEM: usize = 2 * (16 * 64 * 2) + 128 * 64 * 2; // 20,480 bytes
/// Large prefill direct-FP32 `64x128x32` x 3 stages dynamic SMEM (`36,864 B`, 2 blk/SM, overlapped A loads, zero pack).
pub const TC_BIG_FP32_SMEM: usize = 3 * (TC_BM * TC_BK * 2 + BIG_BN * TC_BK * 2); // 36,864 bytes
/// Medium-M (`129..=256`) packed `32x32x32` x 4 stages dynamic SMEM.
pub const TC_MED_W2_SMEM: usize = 4 * (32 * TC_BK * 2 + 32 * TC_BK * 2); // 16,384 bytes
/// General `64x64x32` 4-stage dynamic SMEM (medium dense & MoE).
pub const TC_GEN_STAGES: usize = 4;
pub const TC_GEN_SMEM: usize = TC_GEN_STAGES * (TC_BM * TC_BK * 2 + TC_PLAIN_BN * TC_BK * 2); // 32,768 bytes

/// Threads per block and limits of `MOE_GROUP_ROUTE`.
pub const MOE_ROUTE_THREADS: usize = 256;
pub const MOE_ROWS_MAX: usize = 4096;
pub const MOE_PICKS_MAX: usize = 32;
pub const MOE_ALL_MAX: usize = 32_768;
pub const MOE_EXPERTS_MAX: usize = 512;
/// Phase selectors (payload word 10) for single-block (`rows <= 256`) and deterministic 3-phase multi-block
/// `MOE_GROUP_ROUTE` (`rows >= MOE_ROUTE_MULTI_ROWS_MIN`).
pub const MOE_ROUTE_PHASE_SINGLE: u64 = 0;
pub const MOE_ROUTE_PHASE_COUNT: u64 = 1;
pub const MOE_ROUTE_PHASE_SCAN: u64 = 2;
pub const MOE_ROUTE_PHASE_SCATTER: u64 = 3;
pub const MOE_ROUTE_MULTI_ROWS_MIN: usize = 257;
pub const MOE_ROUTE_BLOCKS_MAX: usize = 32;

/// Number of CTAs for `MOE_GROUP_ROUTE`: 1 block for small batches (`rows <= 256`), up to 32 blocks for prefill.
#[inline(always)]
pub const fn moe_route_blocks(rows: usize) -> usize {
    if rows < MOE_ROUTE_MULTI_ROWS_MIN {
        1
    } else {
        let b = rows / 64;
        if b < 2 { 2 } else if b > MOE_ROUTE_BLOCKS_MAX { MOE_ROUTE_BLOCKS_MAX } else { b }
    }
}

/// True when `PLAIN_TC_ROWS` Mode 0 uses a small-batch tile (`inputs <= TC_SMALL_BM`).
#[inline(always)]
pub const fn tc_small_wanted(inputs: usize) -> bool {
    inputs >= 1 && inputs <= TC_SMALL_BM
}

/// True when a dense shape uses the 1-warp deep-pipelined direct-FP32 narrow path (`rows <= 128, inputs <= 256`).
#[inline(always)]
pub const fn tc_narrow_1w_wanted(rows: usize, inputs: usize) -> bool {
    rows >= 1 && rows <= NARROW_BN && inputs >= 1 && inputs <= TC_N1_INPUTS_MAX
}

/// True when a dense shape uses the 4-warp prefill direct-FP32 narrow path (`rows <= 128, inputs > 256`).
#[inline(always)]
pub const fn tc_narrow_4w_wanted(rows: usize, inputs: usize) -> bool {
    rows >= 1 && rows <= NARROW_BN && inputs > TC_N1_INPUTS_MAX && inputs <= TC_INPUTS_MAX
}

/// True when a dense shape uses the single-shot direct-FP32 `cols == 64` path (`GateOuter`).
#[inline(always)]
pub const fn tc_short_k_wanted(rows: usize, cols: usize) -> bool {
    cols == TC_STAGE && rows > NARROW_BN
}

/// True when a large prefill dense product takes the overlapped direct-FP32 `64x128x32` path in `PLAIN_TC_ROWS`
/// (`K/V`, `K+KTap`: `512 <= rows <= 1280, inputs >= 512, cols > 64`, eliminating `TC_MODE_PACK`).
#[inline(always)]
pub const fn tc_big_fp32_wanted(rows: usize, cols: usize, inputs: usize) -> bool {
    inputs >= TC_BIG_INPUTS_MIN && rows >= TC_BIG_ROWS_MIN && rows <= 1280 && cols > TC_STAGE
}

/// True when a dense product uses `TC_MODE_DENSE_FP32` directly (zero `TC_MODE_PACK`):
/// - All narrow (`rows <= 128`, non-split)
/// - All short-K (`cols == 64`, non-pair, `GateOuter`; split allowed for `inputs <= 32`)
/// - Narrow-band large prefill (`inputs >= 512, 512 <= rows <= 1280, cols > 64`, non-pair, single or split)
#[inline(always)]
pub const fn tc_fp32_direct_wanted(rows: usize, cols: usize, inputs: usize, split: bool, pair: bool) -> bool {
    if !pair && tc_big_fp32_wanted(rows, cols, inputs) {
        return true;
    }
    if !pair && tc_short_k_wanted(rows, cols) && (!split || tc_small_wanted(inputs)) {
        return true;
    }
    if split { return false; }
    rows <= NARROW_BN && inputs >= 1 && inputs <= TC_INPUTS_MAX
}

/// True when `PLAIN_TC_ROWS` uses the 1-warp narrow/medium tile (`inputs <= 32 && rows <= 2560`).
#[inline(always)]
pub const fn tc_tiny_wanted(rows: usize, inputs: usize) -> bool {
    tc_small_wanted(inputs) && rows <= TC_TINY_ROWS_MAX
}

/// True when Mode 0 dense uses the 4-warp cooperative-W `BN=16, BK=64` `evict_first` path (`m_tiles=1`):
/// - `17..=32` with `rows <= 2560` (`O_proj`, `K/V`, `K+KTap`: 4x warps/SM, 2-stage X = 12,288 B)
/// - `9..=16` with `1280 < rows <= 2560` (`O_proj`, the panel shapes): the 1-warp tile ran one warp a 16-row band
///   (144 warps on 48 SMs, 134 us at C16 decode, 73 us here); the four warps of a band split its weight chunks and
///   each output keeps its one ascending-k MMA chain (same words). `K/V` and `K+KTap` (`rows <= 1280`, raw weights)
///   stay on the 1-warp tile there (29 us against 31 us at C16 decode).
/// - `33..=128` with `rows <= 16_384 && cols > 64` (C32 `M=125` prefill on `QKV`, `O_proj`, `K/V`: 1-stage X = 20,480 B)
#[inline(always)]
pub const fn tc_coop_w_wanted(rows: usize, cols: usize, inputs: usize) -> bool {
    (inputs > 16 && inputs <= TC_SMALL_BM && rows <= TC_TINY_ROWS_MAX)
        || (inputs > TC_COOP_SMALL_MIN && inputs <= 16 && rows > TC_PANEL_ROWS_MIN && rows <= TC_TINY_ROWS_MAX)
        || (inputs > TC_SMALL_BM && inputs <= 128 && rows <= 16_384 && cols > TC_STAGE)
}
/// Inputs at or below this keep the 1-warp tile (`tc_tiny_wanted`) for every `rows <= 2560`.
pub const TC_COOP_SMALL_MIN: usize = 8;
/// Dense matrices of at most this many rows (`K/V` 1024, `K+KTap` 1056) read raw weights, not `bf16_panels`.
pub const TC_PANEL_ROWS_MIN: usize = 1280;

/// True when a dense product with registered `bf16_panels` takes `PLAIN_TC_PANEL64` (kind 124): 33..=128 input rows
/// (C16/C32 verify) of a matrix wider than `TC_TINY_ROWS_MAX` rows (LM head, `QKV`). One warp decodes a 16-row band
/// in registers and multiplies up to 64 inputs; smaller matrices (`O_proj` 2304) keep the 4-warp cooperative-W tile.
#[inline(always)]
pub const fn tc_panel64_wanted(rows: usize, cols: usize, inputs: usize) -> bool {
    inputs > TC_SMALL_BM && inputs <= 128 && rows > TC_TINY_ROWS_MAX && cols > TC_STAGE
}
/// Warps a block of `PLAIN_TC_PANEL64`: 8 for the LM head (more bands share each staged `X` slab), else 4 (a fuller
/// last wave on 48 multiprocessors for `QKV`).
#[inline(always)]
pub const fn tc_panel64_warps(rows: usize) -> usize {
    if rows > 16_384 { 8 } else { 4 }
}
/// Rows of one staged `X` slab of `PLAIN_TC_PANEL64`: the inputs rounded up to a whole `n8` tile.
#[inline(always)]
pub const fn tc_panel64_x_rows(inputs: usize) -> usize {
    inputs.div_ceil(8) * 8
}
/// Dynamic shared bytes of `PLAIN_TC_PANEL64`: two 64-column `X` slabs plus each warp's escape cursors.
#[inline(always)]
pub const fn tc_panel64_smem(inputs: usize, warps: usize, slices: usize) -> usize {
    2 * tc_panel64_x_rows(inputs) * 128 + warps * slices * 128
}
/// Launch geometry `(threads, blocks_x, dynamic_smem_bytes)` of `PLAIN_TC_PANEL64`: one warp a (16-row band,
/// 64-input half).
#[inline(always)]
pub const fn tc_panel64_spec(rows: usize, inputs: usize, slices: usize) -> (usize, u32, usize) {
    let warps = tc_panel64_warps(rows);
    (warps * 32, (rows.div_ceil(16) * inputs.div_ceil(64)).div_ceil(warps) as u32, tc_panel64_smem(inputs, warps, slices))
}
/// Device-side admission guard for `PLAIN_TC_PANEL64`: an admitted dense shape, the host's thread count, and enough
/// dynamic shared memory for the slabs and cursors.
#[inline(always)]
pub const fn tc_panel64_admitted(rows: usize, cols: usize, inputs: usize, threads: usize, shared: usize, slices: usize) -> bool {
    let warps = tc_panel64_warps(rows);
    tc_shape_admitted(rows, cols, inputs) && cols > TC_STAGE && threads == warps * 32
        && shared >= tc_panel64_smem(inputs, warps, slices)
}

/// True when medium-M (`129..=256`, `128 < rows <= 2560`, `cols > 64`) uses the 2-warp `32x32` tiled path to saturate 48 SMs.
#[inline(always)]
pub const fn tc_med_w2_wanted(rows: usize, cols: usize, inputs: usize) -> bool {
    inputs > 128 && inputs <= 256 && rows > NARROW_BN && rows <= TC_TINY_ROWS_MAX && cols > TC_STAGE
}

/// Launch geometry `(threads, blocks_x, dynamic_smem_bytes)` for `PLAIN_TC_ROWS` dense Mode 0 or Mode 4.
/// Both host (`backend/tc_rows.rs`) and device (`plain_tc_rows.rs`) call this single source of truth.
#[inline(always)]
pub const fn tc_dense_spec(rows: usize, cols: usize, inputs: usize, fp32_direct: bool) -> (usize, u32, usize) {
    if fp32_direct {
        if tc_big_fp32_wanted(rows, cols, inputs) {
            return (TC_THREADS, (rows.div_ceil(BIG_BN) * inputs.div_ceil(TC_BM)) as u32, TC_BIG_FP32_SMEM);
        }
        if tc_narrow_1w_wanted(rows, inputs) {
            return (TC_TINY_THREADS, (rows.div_ceil(TC_N1_BN) * inputs.div_ceil(TC_N1_BM)) as u32, TC_N1_SMEM);
        }
        if tc_narrow_4w_wanted(rows, inputs) {
            return if rows <= TC_N4_BN {
                (TC_N4_THREADS, inputs.div_ceil(TC_N4_BM) as u32, TC_N4_SMEM)
            } else {
                (TC_THREADS, inputs.div_ceil(TC_N4_BM) as u32, TC_N8_SMEM)
            };
        }
        if tc_short_k_wanted(rows, cols) {
            return if inputs <= TC_SMALL_BM {
                (64, rows.div_ceil(32) as u32, TC_SHORT_SMALL_SMEM)
            } else {
                (TC_THREADS, (inputs.div_ceil(TC_BM) * rows.div_ceil(TC_PLAIN_BN)) as u32, TC_SHORT_BIG_SMEM)
            };
        }
    }
    if tc_coop_w_wanted(rows, cols, inputs) {
        let smem = if inputs <= TC_SMALL_BM { TC_COOP_M32_SMEM } else { TC_COOP_M128_SMEM };
        (128, rows.div_ceil(16) as u32, smem)
    } else if tc_small_wanted(inputs) {
        if rows <= TC_TINY_ROWS_MAX {
            (TC_TINY_THREADS, rows.div_ceil(16) as u32, TC_W1_M16_SMEM)
        } else if rows <= 16_384 {
            let smem = if inputs <= 16 { TC_W2_M16_SMEM } else { TC_W2_SMEM };
            (64, rows.div_ceil(32) as u32, smem)
        } else {
            let smem = if inputs <= 16 { TC_W4_M16_SMEM } else { TC_W4_SMEM };
            (128, rows.div_ceil(64) as u32, smem)
        }
    } else if tc_med_w2_wanted(rows, cols, inputs) {
        (64, (inputs.div_ceil(32) * rows.div_ceil(32)) as u32, TC_MED_W2_SMEM)
    } else {
        (TC_THREADS, (rows.div_ceil(TC_PLAIN_BN) * inputs.div_ceil(TC_BM)) as u32, TC_GEN_SMEM)
    }
}

/// Dynamic shared bytes for `PLAIN_TC_ROWS` Mode 0 when streaming lossless Vade-BF11-BP `bf16_panels`:
/// - Path 4 (`tc_small_wanted` without `tc_coop_w_wanted`): zero W SMEM (panels decode directly into registers);
///   stages `part_slabs` 64-col slabs of BF16 `X` (`<= 12,288 B`, 4 slabs for `M <= 16`, 3 slabs for `M <= 32`)
///   followed by per-warp escape cursor tables (`warps * slices * 128 B`).
/// - Path 4b (`tc_coop_w_wanted`): 4-chunk decoded W (`8,192 B`) + 4-chunk X (`16,384 B`) + escape cursor table
///   (`slices * 128 B`) for `inputs <= TC_SMALL_BM`, else raw `TC_COOP_M128_SMEM`.
#[inline(always)]
pub const fn tc_dense_panel_smem(rows: usize, cols: usize, inputs: usize, slices: usize) -> usize {
    if tc_coop_w_wanted(rows, cols, inputs) {
        if inputs <= TC_SMALL_BM {
            8_192 + 16_384 + slices * 128
        } else {
            TC_COOP_M128_SMEM
        }
    } else if tc_small_wanted(inputs) {
        let bm = if inputs <= 16 { 16 } else { 32 };
        let warps = if rows <= TC_TINY_ROWS_MAX { 1 } else if rows <= 16_384 { 2 } else { 4 };
        let max_slabs = if bm == 16 { 4 } else { 3 };
        let chunks = cols / 64;
        let part_slabs = if chunks < max_slabs { chunks } else { max_slabs };
        part_slabs * bm * 128 + warps * slices * 128
    } else {
        tc_dense_spec(rows, cols, inputs, false).2
    }
}

/// Threads per block for `PLAIN_TC_ROWS` Mode 0 (packed BF16 dense).
#[inline(always)]
pub const fn tc_dense_threads(rows: usize, inputs: usize) -> usize {
    tc_dense_spec(rows, TC_STAGE * 2, inputs, false).0
}

/// Shape admission for `PLAIN_TC_ROWS` dense mode (and per-expert matrix shape in MoE mode).
#[inline(always)]
pub const fn tc_shape_admitted(rows: usize, cols: usize, inputs: usize) -> bool {
    rows >= 1 && rows <= TC_DENSE_ROWS_MAX
        && cols >= TC_STAGE && cols <= TC_COLS_MAX && cols % TC_STAGE == 0
        && rows.saturating_mul(cols / 2) <= u32::MAX as usize
        && inputs >= 1 && inputs <= TC_INPUTS_MAX
}

/// Device-side admission guard for `PLAIN_TC_ROWS` Mode 0 / Mode 4 (dense).
#[inline(always)]
pub const fn tc_dense_admitted(rows: usize, cols: usize, inputs: usize, weights: usize, input: usize, threads: usize) -> bool {
    tc_shape_admitted(rows, cols, inputs) && threads == tc_dense_spec(rows, cols, inputs, false).0 && (weights | input) & 15 == 0
}

/// Device-side admission guard for `PLAIN_TC_ROWS` Mode 4 (direct-FP32 dense).
#[inline(always)]
pub const fn tc_dense_fp32_admitted(rows: usize, cols: usize, inputs: usize, weights: usize, input: usize, threads: usize, shared_bytes: usize) -> bool {
    let (want_t, _, want_smem) = tc_dense_spec(rows, cols, inputs, true);
    tc_shape_admitted(rows, cols, inputs) && threads == want_t && shared_bytes >= want_smem && (weights | input) & 15 == 0
}

/// Grid blocks for `PLAIN_TC_ROWS` Mode 0 (packed BF16 dense).
#[inline(always)]
pub const fn tc_dense_blocks(rows: usize, inputs: usize) -> u32 {
    tc_dense_spec(rows, TC_STAGE * 2, inputs, false).1
}

/// Device-side admission guard for `PLAIN_TC_BIG`: an admitted dense shape and launch. With `TC_INPUTS_MAX`,
/// `TC_DENSE_ROWS_MAX` and `TC_COLS_MAX`, every word offset into the weights and every value offset into an FP32 input
/// (so every word offset into a packed input) is below 2^32.
#[inline(always)]
pub const fn tc_big_admitted(rows: usize, cols: usize, inputs: usize, weights: usize, input: usize, threads: usize) -> bool {
    tc_shape_admitted(rows, cols, inputs) && threads == TC_THREADS && (weights | input) & 15 == 0
}
const _: () = assert!(TC_DENSE_ROWS_MAX * (TC_COLS_MAX / 2) <= u32::MAX as usize);
const _: () = assert!(TC_INPUTS_MAX * TC_COLS_MAX <= u32::MAX as usize);

/// True when a dense product of this shape takes `PLAIN_TC_BIG`.
#[inline(always)]
pub const fn tc_big_wanted(rows: usize, inputs: usize) -> bool {
    inputs >= TC_BIG_INPUTS_MIN && rows >= TC_BIG_ROWS_MIN
}

/// True when a dense product of this shape takes the narrow form of `PLAIN_TC_BIG` (`TC_MODE_NARROW`).
#[inline(always)]
pub const fn tc_narrow_wanted(rows: usize, inputs: usize) -> bool {
    inputs >= TC_BIG_INPUTS_MIN && rows <= NARROW_BN
}

/// Grid blocks for the narrow form of `PLAIN_TC_BIG`: one block a `NARROW_BM` input rows.
#[inline(always)]
pub const fn tc_narrow_blocks(inputs: usize) -> u32 {
    inputs.div_ceil(NARROW_BM) as u32
}

/// Grid blocks for `TC_MODE_NARROW_HALF`: one block per `NARROW_HALF_BM` (64) packed-BF16 input rows.
#[inline(always)]
pub const fn tc_narrow_half_blocks(inputs: usize) -> u32 {
    inputs.div_ceil(NARROW_HALF_BM) as u32
}

/// Device-side admission guard for `TC_MODE_GATE_BLEND_NORM` inside `PLAIN_TC_BIG`.
#[inline(always)]
pub const fn gate_outer_admitted(rows: usize, dim: usize, k: usize, count: usize, threads: usize) -> bool {
    threads == TC_THREADS
        && rows >= 1 && rows <= TC_INPUTS_MAX
        && dim >= 128 && dim <= GATE_OUTER_DIM_MAX && dim % 128 == 0
        && k >= 16 && k <= GATE_OUTER_K_MAX && k % 16 == 0
        && count >= 1 && count <= GATE_OUTER_GATES_MAX
}

/// Grid blocks along Y for `TC_MODE_GATE_BLEND_NORM` (`grid = (count, gate_outer_row_blocks(rows), 1)`).
#[inline(always)]
pub const fn gate_outer_row_blocks(rows: usize) -> u32 {
    rows.div_ceil(GATE_OUTER_BM) as u32
}

/// Grid blocks for `PLAIN_TC_BIG`: groups of 8 row tiles, column tiles within a group.
#[inline(always)]
pub const fn tc_big_blocks(rows: usize, inputs: usize) -> u32 {
    (rows.div_ceil(BIG_BN) * inputs.div_ceil(BIG_BM)) as u32
}

/// Upper bound on M-tiles across `experts` experts for `all = rows * picks` picks (`ceil(c_e/BM) <= floor(c_e/BM) + 1`).
#[inline(always)]
pub const fn moe_max_tiles(all: usize, experts: usize) -> usize {
    all / TC_BM + experts
}

/// Words (`u32`) of `tile_meta_out`: 1 header word (`total_m_tiles`) + 3 words per M-tile (`expert, first_pos, m_count`).
#[inline(always)]
pub const fn moe_tile_meta_words(all: usize, experts: usize) -> usize {
    1 + 3 * moe_max_tiles(all, experts)
}

/// Grid blocks for `PLAIN_TC_ROWS` Mode 1 (device-routed MoE): N-tiles fastest so inactive tail M-tiles exit cleanly.
#[inline(always)]
pub const fn tc_moe_blocks(rows: usize, max_m_tiles: usize) -> u32 {
    (rows.div_ceil(TC_BN) * max_m_tiles) as u32
}

/// Grid blocks for `PLAIN_PACKED_TC_ROWS`: N-tiles of `TC_PACKED_BN` fastest, one block row a route tile (the block
/// of an odd route tile of its expert returns at once).
#[inline(always)]
pub const fn tc_packed_moe_blocks(rows: usize, max_m_tiles: usize) -> u32 {
    (rows.div_ceil(TC_PACKED_BN) * max_m_tiles) as u32
}

/// Device-side admission guard for `PLAIN_TC_ROWS` Mode 1 (device-routed MoE).
#[inline(always)]
pub const fn tc_moe_admitted(rows: usize, cols: usize, all: usize, experts: usize, max_m_tiles: usize, stack: usize, input: usize, threads: usize) -> bool {
    rows >= 1 && rows <= TC_ROWS_MAX
        && cols >= TC_STAGE && cols <= TC_COLS_MAX && cols % TC_STAGE == 0
        && all >= 1 && all <= MOE_ALL_MAX
        && experts >= 1 && experts <= MOE_EXPERTS_MAX
        && max_m_tiles == moe_max_tiles(all, experts)
        && threads == TC_THREADS
        && (stack | input) & 15 == 0
}

/// Shape admission for `PLAIN_PACKED_TC_ROWS` (kind 106, device-routed packed NVFP4/MXFP8 MoE).
#[inline(always)]
pub const fn tc_packed_moe_shape_admitted(rows: usize, cols: usize, wide: bool, pairs: bool) -> bool {
    let w = crate::packed_product::whole(wide);
    rows >= 1 && rows <= TC_ROWS_MAX
        && cols >= w && cols <= TC_COLS_MAX && cols % w == 0
        && (!pairs || rows % 2 == 0)
}

/// Device-side admission guard for `PLAIN_PACKED_TC_ROWS` (kind 106).
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "shape, packed form, MoE counts, unit size, addresses and threads")]
pub const fn tc_packed_moe_admitted(
    rows: usize, cols: usize, wide: bool, pairs: bool, all: usize, experts: usize,
    max_m_tiles: usize, unit_bytes: usize, stack: usize, input: usize, threads: usize,
) -> bool {
    tc_packed_moe_shape_admitted(rows, cols, wide, pairs)
        && all >= 1 && all <= MOE_ALL_MAX
        && experts >= 1 && experts <= MOE_EXPERTS_MAX
        && max_m_tiles == moe_max_tiles(all, experts)
        && unit_bytes == crate::packed_product::layout(rows, cols, wide).bytes
        && threads == TC_THREADS
        && (stack | input) & 15 == 0
}

/// Device-side admission guard for `MOE_GROUP_ROUTE`.
#[inline(always)]
pub const fn moe_route_admitted(rows: usize, each: usize, picks: usize, experts: usize, max_m_tiles: usize, threads: usize) -> bool {
    threads == MOE_ROUTE_THREADS
        && rows >= 1 && rows <= MOE_ROWS_MAX
        && picks >= 1 && picks <= MOE_PICKS_MAX
        && rows * picks <= MOE_ALL_MAX
        && each >= 2 * picks
        && experts >= 1 && experts <= MOE_EXPERTS_MAX
        && max_m_tiles == moe_max_tiles(rows * picks, experts)
}

/// Minimum input rows for `PLAIN_TC_ROWS` to replace per-row `plain_staged` on dense projections.
#[cfg(not(target_arch = "nvptx64"))]
pub const TC_ROWS_INPUTS_MIN: usize = 1;

/// Matrix rows per part when a dense matrix exceeds `TC_DENSE_ROWS_MAX` (or its `u32` BF16 word limit).
#[cfg(not(target_arch = "nvptx64"))]
pub const fn tc_tile_part(rows: usize, cols: usize, inputs: usize) -> Option<usize> {
    if inputs < TC_ROWS_INPUTS_MIN || rows == 0 || !tc_shape_admitted(1, cols, inputs) { return None; }
    let half_cols = cols / 2;
    let max_by_u32 = u32::MAX as usize / half_cols;
    let max_rows = if max_by_u32 < TC_DENSE_ROWS_MAX { max_by_u32 } else { TC_DENSE_ROWS_MAX };
    if rows <= max_rows { return Some(rows); }
    let parts = rows.div_ceil(max_rows);
    let part = rows.div_ceil(parts).div_ceil(TC_BN) * TC_BN;
    let last = rows - (rows.div_ceil(part) - 1) * part;
    if part <= max_rows && last >= 1 { Some(part) } else { None }
}

/// Byte span of one device operand.
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span { pub address: u64, pub bytes: u64 }
#[cfg(not(target_arch = "nvptx64"))]
impl Span {
    fn ends(self) -> bool { self.address.checked_add(self.bytes).is_some() }
    fn meets(self, other: Span) -> bool {
        self.address < other.address.saturating_add(other.bytes) && other.address < self.address.saturating_add(self.bytes)
    }
}

/// One split output target for fused dense projections (`PLAIN_TC_ROWS` Mode 0, `split_count > 0`).
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitOut { pub span: Span, pub cols: usize }

/// Host check for `PLAIN_TC_ROWS` Mode 0 with a single output (`[inputs, stride]` FP32).
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_tc_dense(rows: usize, cols: usize, inputs: usize, stride: usize, weights: Span, input: Span, output: Span) -> Result<(), &'static str> {
    check_tc_dense_form(rows, cols, inputs, stride, false, weights, input, output)
}

/// Host check for `PLAIN_TC_ROWS` / `PLAIN_TC_BIG` dense product with FP32 (`half_in == false`) or packed BF16 (`half_in == true`) input.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, input form and three device spans")]
pub fn check_tc_dense_form(rows: usize, cols: usize, inputs: usize, stride: usize, half_in: bool, weights: Span, input: Span, output: Span) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 || inputs == 0 { return Err("tc rows: no rows, no columns or no inputs"); }
    if !tc_shape_admitted(rows, cols, inputs) { return Err("tc rows: shape outside tensor-core tile envelope"); }
    if stride < rows || stride > u32::MAX as usize { return Err("tc rows: stride below rows or above ABI"); }
    if weights.bytes != (rows as u64).checked_mul(cols as u64).and_then(|n| n.checked_mul(2)).unwrap_or(u64::MAX) {
        return Err("tc rows: weights differ from rows x cols BF16");
    }
    let in_elem = if half_in { 2u64 } else { 4u64 };
    if input.bytes != (inputs as u64).checked_mul(cols as u64).and_then(|n| n.checked_mul(in_elem)).unwrap_or(u64::MAX) {
        return Err("tc rows: input differs from inputs x cols");
    }
    let out_bytes = (inputs as u64 - 1).checked_mul(stride as u64).and_then(|w| w.checked_add(rows as u64)).and_then(|w| w.checked_mul(4));
    if out_bytes != Some(output.bytes) { return Err("tc rows: output differs from (inputs - 1) x stride + rows FP32"); }
    if weights.address == 0 || input.address == 0 || output.address == 0 { return Err("tc rows: null operand"); }
    if !weights.ends() || !input.ends() || !output.ends() { return Err("tc rows: operand ends above address space"); }
    if weights.address % 16 != 0 { return Err("tc rows: weights not 16-byte aligned"); }
    if input.address % 16 != 0 { return Err("tc rows: input not 16-byte aligned"); }
    if output.address % 4 != 0 { return Err("tc rows: output not word aligned"); }
    if output.meets(weights) || output.meets(input) { return Err("tc rows: output overlaps an operand"); }
    Ok(())
}

/// Host check for the half form of a dense `PLAIN_TC_BIG` product (`TC_BIG_HALF_OUT`): a packed BF16 input, an output
/// of `inputs * rows` BF16 values and the ring of `taps * rows` FP32 words, apart from each other and the operands.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, ring slot and four device spans")]
pub fn check_tc_big_half_out(rows: usize, cols: usize, inputs: usize, first: usize, taps: usize, weights: Span, input: Span, output: Span, ring: Span) -> Result<(), &'static str> {
    if !tc_big_half_out_admitted(rows, cols, inputs, first, taps) { return Err("tc big half: shape outside the big tile envelope"); }
    let bytes = |a: usize, b: usize, each: u64| (a as u64).checked_mul(b as u64).and_then(|n| n.checked_mul(each));
    if Some(weights.bytes) != bytes(rows, cols, 2) { return Err("tc big half: weights differ from rows x cols BF16"); }
    if Some(input.bytes) != bytes(inputs, cols, 2) { return Err("tc big half: input differs from inputs x cols BF16"); }
    if Some(output.bytes) != bytes(inputs, rows, 2) { return Err("tc big half: output differs from inputs x rows BF16"); }
    if Some(ring.bytes) != bytes(taps, rows, 4) { return Err("tc big half: ring differs from taps x rows FP32"); }
    if weights.address == 0 || input.address == 0 || output.address == 0 || ring.address == 0 { return Err("tc big half: null operand"); }
    if !weights.ends() || !input.ends() || !output.ends() || !ring.ends() { return Err("tc big half: operand ends above address space"); }
    if weights.address % 16 != 0 || input.address % 16 != 0 { return Err("tc big half: operand not 16-byte aligned"); }
    if output.address % 4 != 0 || ring.address % 4 != 0 { return Err("tc big half: output not word aligned"); }
    if output.meets(weights) || output.meets(input) || output.meets(ring) || ring.meets(weights) || ring.meets(input) {
        return Err("tc big half: output overlaps an operand");
    }
    Ok(())
}

/// Host check for `PLAIN_TC_ROWS` Mode 0 with two concatenated half-width inputs (`[inputs, cols/2]` each).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape and four device spans")]
pub fn check_tc_pair(rows: usize, cols: usize, inputs: usize, weights: Span, input1: Span, input2: Span, output: Span) -> Result<(), &'static str> {
    check_tc_pair_form(rows, cols, inputs, false, weights, input1, input2, output)
}

/// Host check for dual-input tensor-core product with FP32 (`half_in == false`) or packed BF16 (`half_in == true`) inputs.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, input form and four device spans")]
pub fn check_tc_pair_form(rows: usize, cols: usize, inputs: usize, half_in: bool, weights: Span, input1: Span, input2: Span, output: Span) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 || inputs == 0 { return Err("tc rows pair: no rows, no columns or no inputs"); }
    if !tc_shape_admitted(rows, cols, inputs) || !(cols / 2).is_multiple_of(TC_BK) {
        return Err("tc rows pair: shape outside tensor-core dual-input envelope");
    }
    let half = cols / 2;
    if weights.bytes != (rows as u64).checked_mul(cols as u64).and_then(|n| n.checked_mul(2)).unwrap_or(u64::MAX) {
        return Err("tc rows pair: weights differ from rows x cols BF16");
    }
    let in_elem = if half_in { 2u64 } else { 4u64 };
    let half_bytes = (inputs as u64).checked_mul(half as u64).and_then(|n| n.checked_mul(in_elem));
    if Some(input1.bytes) != half_bytes || Some(input2.bytes) != half_bytes {
        return Err("tc rows pair: input differs from inputs x (cols/2)");
    }
    if Some(output.bytes) != (inputs as u64).checked_mul(rows as u64).and_then(|n| n.checked_mul(4)) {
        return Err("tc rows pair: output differs from inputs x rows FP32");
    }
    if weights.address == 0 || input1.address == 0 || input2.address == 0 || output.address == 0 { return Err("tc rows pair: null operand"); }
    if !weights.ends() || !input1.ends() || !input2.ends() || !output.ends() { return Err("tc rows pair: operand ends above address space"); }
    if weights.address % 16 != 0 || input1.address % 16 != 0 || input2.address % 16 != 0 { return Err("tc rows pair: weights or inputs not 16-byte aligned"); }
    if output.address % 4 != 0 { return Err("tc rows pair: output not word aligned"); }
    if output.meets(weights) || output.meets(input1) || output.meets(input2) { return Err("tc rows pair: output overlaps an operand"); }
    Ok(())
}

/// Host check for grouped dual-input narrow `PLAIN_TC_ROWS` Mode 4 (`1..=3` matrices of one shape).
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_tc_group_pair(rows: usize, cols: usize, inputs: usize, items: &[(Span, Span, Span)], output: Span) -> Result<(), &'static str> {
    if items.is_empty() || items.len() > TC_GROUP_MAX { return Err("tc rows group pair: group count must be 1..=3"); }
    if !tc_narrow_1w_wanted(rows, inputs) { return Err("tc rows group pair: shape outside narrow 1-warp envelope"); }
    let each = (inputs as u64).checked_mul(rows as u64).and_then(|n| n.checked_mul(4)).ok_or("tc rows group pair: output bytes overflow")?;
    if Some(output.bytes) != each.checked_mul(items.len() as u64) {
        return Err("tc rows group pair: output differs from groups x inputs x rows FP32");
    }
    if output.address == 0 || !output.ends() || output.address % 4 != 0 { return Err("tc rows group pair: bad output address or alignment"); }
    for (g, &(w, x1, x2)) in items.iter().enumerate() {
        let part = Span { address: output.address + g as u64 * each, bytes: each };
        check_tc_pair(rows, cols, inputs, w, x1, x2, part)?;
        if output.meets(w) || output.meets(x1) || output.meets(x2) {
            return Err("tc rows group pair: output overlaps an operand");
        }
    }
    Ok(())
}

/// Host check for `PLAIN_TC_ROWS` Mode 0 with 1..=8 split outputs (`[inputs, seg.cols]` FP32 each).
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_tc_split(rows: usize, cols: usize, inputs: usize, weights: Span, input: Span, outputs: &[SplitOut]) -> Result<(), &'static str> {
    check_tc_split_form(rows, cols, inputs, false, weights, input, outputs)
}

/// Host check for split-output tensor-core product with FP32 (`half_in == false`) or packed BF16 (`half_in == true`) input.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_tc_split_form(rows: usize, cols: usize, inputs: usize, half_in: bool, weights: Span, input: Span, outputs: &[SplitOut]) -> Result<(), &'static str> {
    check_tc_split_halves(rows, cols, inputs, half_in, 0, weights, input, outputs)
}

/// `check_tc_split_form` with the outputs of `halves` (bit `i`: output `i`) in the half form
/// (`TC_SPLIT_HALF_OUT`): such an output holds `inputs * seg.cols` BF16 values, 16 bits each.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, forms and three device spans")]
pub fn check_tc_split_halves(rows: usize, cols: usize, inputs: usize, half_in: bool, halves: usize, weights: Span, input: Span, outputs: &[SplitOut]) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 || inputs == 0 { return Err("tc rows split: no rows, no columns or no inputs"); }
    if halves >> outputs.len().min(usize::BITS as usize - 1) != 0 { return Err("tc rows split: a half form names no output"); }
    if !tc_shape_admitted(rows, cols, inputs) { return Err("tc rows split: shape outside tensor-core tile envelope"); }
    if outputs.is_empty() || outputs.len() > TC_SPLIT_MAX { return Err("tc rows split: split output count must be 1..=8"); }
    if weights.bytes != (rows as u64).checked_mul(cols as u64).and_then(|n| n.checked_mul(2)).unwrap_or(u64::MAX) {
        return Err("tc rows split: weights differ from rows x cols BF16");
    }
    let in_elem = if half_in { 2u64 } else { 4u64 };
    if input.bytes != (inputs as u64).checked_mul(cols as u64).and_then(|n| n.checked_mul(in_elem)).unwrap_or(u64::MAX) {
        return Err("tc rows split: input differs from inputs x cols");
    }
    if weights.address == 0 || input.address == 0 { return Err("tc rows split: null operand"); }
    if !weights.ends() || !input.ends() { return Err("tc rows split: operand ends above address space"); }
    if weights.address % 16 != 0 { return Err("tc rows split: weights not 16-byte aligned"); }
    if input.address % 16 != 0 { return Err("tc rows split: input not 16-byte aligned"); }
    let mut sum_cols = 0usize;
    for (i, seg) in outputs.iter().enumerate() {
        if seg.cols == 0 || !seg.cols.is_multiple_of(2) { return Err("tc rows split: each split width must be positive and even"); }
        sum_cols = sum_cols.checked_add(seg.cols).ok_or("tc rows split: split widths overflow")?;
        let expect = (inputs as u64).checked_mul(seg.cols as u64).and_then(|w| w.checked_mul(if (halves >> i) & 1 != 0 { 2 } else { 4 }));
        if expect != Some(seg.span.bytes) { return Err("tc rows split: split output size differs from inputs x seg.cols values"); }
        if seg.span.address == 0 || !seg.span.ends() { return Err("tc rows split: null or overflowing split output"); }
        if seg.span.address % 4 != 0 { return Err("tc rows split: split output not word aligned"); }
        if seg.span.meets(weights) || seg.span.meets(input) { return Err("tc rows split: split output overlaps an operand"); }
        for other in &outputs[..i] {
            if seg.span.meets(other.span) { return Err("tc rows split: two split outputs overlap"); }
        }
    }
    if sum_cols != rows { return Err("tc rows split: split widths do not sum to matrix rows"); }
    Ok(())
}

/// Host check for `MOE_GROUP_ROUTE`.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape and five device buffers")]
pub fn check_moe_group_route(rows: usize, each: usize, picks: usize, experts: usize, route: Span, weights_out: Span, in_rows_out: Span, out_slots_out: Span, tile_meta_out: Span) -> Result<(), &'static str> {
    if rows == 0 || picks == 0 || experts == 0 { return Err("moe route: zero rows, picks or experts"); }
    if rows > MOE_ROWS_MAX || picks > MOE_PICKS_MAX || experts > MOE_EXPERTS_MAX { return Err("moe route: dimensions exceed envelope"); }
    let all = rows.checked_mul(picks).filter(|&a| a <= MOE_ALL_MAX).ok_or("moe route: rows x picks exceeds envelope")?;
    if each < 2 * picks { return Err("moe route: route stride below 2 x picks"); }
    if route.bytes != (rows as u64).checked_mul(each as u64).and_then(|w| w.checked_mul(4)).unwrap_or(u64::MAX) {
        return Err("moe route: route buffer size differs from rows x each");
    }
    let all_bytes = all as u64 * 4;
    if weights_out.bytes != all_bytes || in_rows_out.bytes != all_bytes || out_slots_out.bytes != all_bytes {
        return Err("moe route: output table size differs from rows x picks");
    }
    let meta_bytes = moe_tile_meta_words(all, experts) as u64 * 4;
    if tile_meta_out.bytes != meta_bytes { return Err("moe route: tile_meta size differs from expected words"); }
    let outs = [weights_out, in_rows_out, out_slots_out, tile_meta_out];
    if route.address == 0 || !route.ends() || route.address % 4 != 0 { return Err("moe route: bad route address or alignment"); }
    for (i, &out) in outs.iter().enumerate() {
        if out.address == 0 || !out.ends() || out.address % 4 != 0 { return Err("moe route: bad output address or alignment"); }
        if out.meets(route) { return Err("moe route: output overlaps route input"); }
        for &prev in &outs[..i] {
            if out.meets(prev) { return Err("moe route: two outputs overlap"); }
        }
    }
    Ok(())
}

/// Host check for `PLAIN_TC_ROWS` Mode 1 (device-routed MoE stack).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, expert count, stack, metadata and operands")]
pub fn check_tc_moe(rows: usize, cols: usize, all: usize, experts: usize, stack: Span, tile_meta: Span, input: Span, in_count: usize, in_table: Option<Span>, output: Span, out_count: usize, out_table: Option<Span>) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 || all == 0 || experts == 0 { return Err("tc moe: zero extent"); }
    if rows > TC_ROWS_MAX || !tc_shape_admitted(rows, cols, 1) || all > MOE_ALL_MAX || experts > MOE_EXPERTS_MAX {
        return Err("tc moe: shape or counts outside envelope");
    }
    let expert_bytes = (rows as u64).checked_mul(cols as u64).and_then(|n| n.checked_mul(2)).ok_or("tc moe: matrix bytes overflow")?;
    if expert_bytes % 16 != 0 { return Err("tc moe: expert matrix stride not 16-byte aligned"); }
    if (experts as u64).checked_mul(expert_bytes) != Some(stack.bytes) { return Err("tc moe: stack size differs from experts x rows x cols BF16"); }
    if tile_meta.bytes != moe_tile_meta_words(all, experts) as u64 * 4 { return Err("tc moe: tile_meta size differs from expected words"); }
    if in_count == 0 || (in_table.is_none() && in_count != all) { return Err("tc moe: bad input row count"); }
    if out_count == 0 || (out_table.is_none() && out_count != all) { return Err("tc moe: bad output row count"); }
    if (in_count as u64).checked_mul(cols as u64).and_then(|w| w.checked_mul(4)) != Some(input.bytes) {
        return Err("tc moe: input size differs from in_count x cols FP32");
    }
    if (out_count as u64).checked_mul(rows as u64).and_then(|w| w.checked_mul(4)) != Some(output.bytes) {
        return Err("tc moe: output size differs from out_count x rows FP32");
    }
    if let Some(t) = in_table { if t.bytes != all as u64 * 4 || t.address == 0 || !t.ends() || t.address % 4 != 0 { return Err("tc moe: bad in_table"); } }
    if let Some(t) = out_table { if t.bytes != all as u64 * 4 || t.address == 0 || !t.ends() || t.address % 4 != 0 { return Err("tc moe: bad out_table"); } }
    if stack.address == 0 || tile_meta.address == 0 || input.address == 0 || output.address == 0 { return Err("tc moe: null operand"); }
    if !stack.ends() || !tile_meta.ends() || !input.ends() || !output.ends() { return Err("tc moe: operand ends above address space"); }
    if stack.address % 16 != 0 || input.address % 16 != 0 { return Err("tc moe: stack or input not 16-byte aligned"); }
    if tile_meta.address % 4 != 0 || output.address % 4 != 0 { return Err("tc moe: tile_meta or output not word aligned"); }
    if output.meets(stack) || output.meets(tile_meta) || output.meets(input) || in_table.is_some_and(|t| output.meets(t)) || out_table.is_some_and(|t| output.meets(t)) {
        return Err("tc moe: output overlaps an operand");
    }
    Ok(())
}

/// Host check for `PLAIN_PACKED_TC_ROWS` (kind 106, device-routed packed NVFP4/MXFP8 MoE stack).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, packed form, expert count, stack, metadata and operands")]
pub fn check_tc_packed_moe(
    rows: usize, cols: usize, wide: bool, pairs: bool, all: usize, experts: usize,
    stack: Span, tile_meta: Span, input: Span, in_count: usize, in_table: Option<Span>,
    output: Span, out_count: usize, out_table: Option<Span>, gate: Option<Span>,
) -> Result<(), &'static str> {
    check_tc_packed_moe_form(rows, cols, wide, pairs, all, experts, stack, tile_meta, input, in_count, in_table, output, out_count, out_table, gate, false)
}

/// `check_tc_packed_moe` with the output form: `half` (`packed_product::HALF_OUT`) on an ungated launch needs no
/// `PAIRS` and even `rows`, and `output` holds `out_count * rows` BF16 values (`out_count * rows / 2` words); on a
/// gated launch `output` holds `out_count * rows / 2` BF16 values. The input is FP32 words.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, packed form, expert count, stack, metadata, operands and form")]
pub fn check_tc_packed_moe_form(
    rows: usize, cols: usize, wide: bool, pairs: bool, all: usize, experts: usize,
    stack: Span, tile_meta: Span, input: Span, in_count: usize, in_table: Option<Span>,
    output: Span, out_count: usize, out_table: Option<Span>, gate: Option<Span>, half: bool,
) -> Result<(), &'static str> {
    check_tc_packed_moe_input(rows, cols, wide, pairs, all, experts, stack, tile_meta, input, in_count, in_table, output, out_count, out_table, gate, half, false)
}

/// `check_tc_packed_moe_form` with the input form: `half_in` (`packed_product::HALF_IN`) means `input` holds
/// `in_count * cols` BF16 values, two a word, else FP32 words.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, packed form, expert count, stack, metadata, operands and forms")]
pub fn check_tc_packed_moe_input(
    rows: usize, cols: usize, wide: bool, pairs: bool, all: usize, experts: usize,
    stack: Span, tile_meta: Span, input: Span, in_count: usize, in_table: Option<Span>,
    output: Span, out_count: usize, out_table: Option<Span>, gate: Option<Span>, half: bool, half_in: bool,
) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 || all == 0 || experts == 0 { return Err("tc packed moe: zero extent"); }
    if half && gate.is_none() && (pairs || rows % 2 != 0) { return Err("tc packed moe: an ungated half output needs no pairs and even rows"); }
    // Gated `GateUp`: mixed gate and up rows, `rows / 2` output words a row, two coefficient tables of
    // `CAPPED_GATE_TABLE_WORDS` FP32 words each.
    let width = if gate.is_some() { rows / 2 } else { rows };
    if let Some(t) = gate {
        if !pairs { return Err("tc packed moe: a gated product needs mixed gate and up rows"); }
        if t.bytes != 2 * crate::fast_ops::CAPPED_GATE_TABLE_WORDS as u64 * 4 || t.address == 0 || !t.ends() || t.address % 4 != 0 {
            return Err("tc packed moe: bad gate coefficient tables");
        }
        if output.meets(t) { return Err("tc packed moe: output overlaps the gate coefficient tables"); }
    }
    if !tc_packed_moe_shape_admitted(rows, cols, wide, pairs) || all > MOE_ALL_MAX || experts > MOE_EXPERTS_MAX {
        return Err("tc packed moe: shape or counts outside envelope");
    }
    let unit_bytes = crate::packed_product::layout(rows, cols, wide).bytes as u64;
    if unit_bytes % 16 != 0 { return Err("tc packed moe: packed unit not 16-byte aligned"); }
    if (experts as u64).checked_mul(unit_bytes) != Some(stack.bytes) {
        return Err("tc packed moe: stack size differs from experts x packed unit bytes");
    }
    if tile_meta.bytes != moe_tile_meta_words(all, experts) as u64 * 4 { return Err("tc packed moe: tile_meta size differs from expected words"); }
    if in_count == 0 || (in_table.is_none() && in_count != all) { return Err("tc packed moe: bad input row count"); }
    if out_count == 0 || (out_table.is_none() && out_count != all) { return Err("tc packed moe: bad output row count"); }
    let input_bytes = if half_in { 2 } else { 4 };
    if (in_count as u64).checked_mul(cols as u64).and_then(|w| w.checked_mul(input_bytes)) != Some(input.bytes) {
        return Err("tc packed moe: input size differs from in_count x cols (FP32, or BF16 when half_in)");
    }
    let value_bytes = if half { 2 } else { 4 };
    if (out_count as u64).checked_mul(width as u64).and_then(|w| w.checked_mul(value_bytes)) != Some(output.bytes) {
        return Err("tc packed moe: output size differs from out_count x rows FP32 (rows / 2 when gated, BF16 when half)");
    }
    if let Some(t) = in_table { if t.bytes != all as u64 * 4 || t.address == 0 || !t.ends() || t.address % 4 != 0 { return Err("tc packed moe: bad in_table"); } }
    if let Some(t) = out_table { if t.bytes != all as u64 * 4 || t.address == 0 || !t.ends() || t.address % 4 != 0 { return Err("tc packed moe: bad out_table"); } }
    if stack.address == 0 || tile_meta.address == 0 || input.address == 0 || output.address == 0 { return Err("tc packed moe: null operand"); }
    if !stack.ends() || !tile_meta.ends() || !input.ends() || !output.ends() { return Err("tc packed moe: operand ends above address space"); }
    if stack.address % 16 != 0 || input.address % 16 != 0 { return Err("tc packed moe: stack or input not 16-byte aligned"); }
    if tile_meta.address % 4 != 0 || output.address % 4 != 0 { return Err("tc packed moe: tile_meta or output not word aligned"); }
    if output.meets(stack) || output.meets(tile_meta) || output.meets(input) || in_table.is_some_and(|t| output.meets(t)) || out_table.is_some_and(|t| output.meets(t)) {
        return Err("tc packed moe: output overlaps an operand");
    }
    Ok(())
}

/// Host check for `TC_MODE_GATE_BLEND_NORM` (1..=3 gates: `weights`, `inner`, `kept`, `msg`, optional `norm_w`, `output`).
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_tc_gate_blend_norm(
    rows: usize, dim: usize, k: usize, eps: f32, half_out: bool,
    entries: &[(Span, Span, Span, Span, Option<Span>, Span)],
) -> Result<(), &'static str> {
    if !gate_outer_admitted(rows, dim, k, entries.len(), TC_THREADS) || !(eps > 0.0 && eps.is_finite()) {
        return Err("tc gate blend norm: shape or epsilon outside envelope");
    }
    let w_bytes = (2 * dim as u64).checked_mul(k as u64).and_then(|n| n.checked_mul(2)).ok_or("tc gate blend norm: weight overflow")?;
    let in_bytes = (rows as u64).checked_mul(k as u64).and_then(|n| n.checked_mul(4)).ok_or("tc gate blend norm: inner overflow")?;
    let v_bytes = (rows as u64).checked_mul(dim as u64).and_then(|n| n.checked_mul(4)).ok_or("tc gate blend norm: row overflow")?;
    let out_bytes = if half_out { v_bytes / 2 } else { v_bytes };
    let nw_bytes = (dim as u64) * 4;
    for &(weights, inner, kept, msg, norm_w, output) in entries {
        if weights.bytes != w_bytes || inner.bytes != in_bytes || kept.bytes != v_bytes
            || msg.bytes != v_bytes || output.bytes != out_bytes
        {
            return Err("tc gate blend norm: operand size mismatch");
        }
        if weights.address == 0 || inner.address == 0 || kept.address == 0 || msg.address == 0 || output.address == 0
            || !weights.ends() || !inner.ends() || !kept.ends() || !msg.ends() || !output.ends()
        {
            return Err("tc gate blend norm: null or overflowing operand");
        }
        if weights.address % 16 != 0 || inner.address % 16 != 0
            || kept.address % 4 != 0 || msg.address % 4 != 0 || output.address % 4 != 0
        {
            return Err("tc gate blend norm: bad operand alignment");
        }
        if let Some(nw) = norm_w {
            if nw.bytes != nw_bytes || nw.address == 0 || !nw.ends() || nw.address % 4 != 0 {
                return Err("tc gate blend norm: bad norm weight span");
            }
        }
    }
    for (i, &(_, _, _, _, _, out)) in entries.iter().enumerate() {
        for &(weights, inner, kept, msg, norm_w, _) in entries {
            if out.meets(weights) || out.meets(inner) || out.meets(kept) || out.meets(msg)
                || norm_w.is_some_and(|nw| out.meets(nw))
            {
                return Err("tc gate blend norm: output overlaps input");
            }
        }
        for &(_, _, _, _, _, other_out) in entries.iter().skip(i + 1) {
            if out.meets(other_out) { return Err("tc gate blend norm: outputs overlap"); }
        }
    }
    Ok(())
}

/// Exact CPU integer twin of one output word of `PLAIN_TC_ROWS`:
/// rounds each FP32 activation once to BF16 (nearest-even), then runs `tc_twin_bf16::step_any`
/// over 16-column steps in strictly ascending K order from `+0.0`.
#[cfg(not(target_arch = "nvptx64"))]
pub fn tc_word(weight_bf16: &[u16], input_f32: &[f32]) -> f32 {
    assert!(weight_bf16.len() == input_f32.len() && weight_bf16.len().is_multiple_of(16));
    let mut acc = 0u32;
    let mut x_bf16 = [0u16; 16];
    for step in 0..weight_bf16.len() / 16 {
        let base = step * 16;
        for j in 0..16 { x_bf16[j] = crate::tc_twin_bf16::bf16(input_f32[base + j]); }
        acc = crate::tc_twin_bf16::step_any(&weight_bf16[base..base + 16], &x_bf16, acc);
    }
    f32::from_bits(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn span(address: u64, bytes: u64) -> Span { Span { address, bytes } }

    #[test]
    fn tc_dense_admission_and_refusals() {
        let (rows, cols, inputs) = (2304, 2304, 64);
        let w = span(1 << 20, (rows * cols * 2) as u64);
        let x = span(1 << 30, (inputs * cols * 4) as u64);
        let y = span(1 << 31, (inputs * rows * 4) as u64);
        assert_eq!(check_tc_dense(rows, cols, inputs, rows, w, x, y), Ok(()));
        assert!(tc_dense_admitted(rows, cols, inputs, w.address as usize, x.address as usize, 128));
        assert!(tc_dense_admitted(rows, cols, 512, w.address as usize, x.address as usize, TC_THREADS));
        assert_eq!(tc_dense_blocks(2304, 1024), 36 * 16);
        assert_eq!(tc_dense_spec(2304, 3072, 32, false), (128, 144, TC_COOP_M32_SMEM));
        assert_eq!(tc_dense_spec(1024, 2304, 32, false), (128, 64, TC_COOP_M32_SMEM));
        assert_eq!(tc_dense_spec(2304, 3072, 16, false), (128, 144, TC_COOP_M32_SMEM));
        assert_eq!(tc_dense_spec(1024, 2304, 9, false), (TC_TINY_THREADS, 64, TC_W1_M16_SMEM));
        assert_eq!(tc_dense_spec(1056, 2304, 16, false), (TC_TINY_THREADS, 66, TC_W1_M16_SMEM));
        assert_eq!(tc_dense_spec(2304, 3072, 9, false), (128, 144, TC_COOP_M32_SMEM));
        assert_eq!(tc_dense_spec(2304, 3072, 8, false), (TC_TINY_THREADS, 144, TC_W1_M16_SMEM));
        assert_eq!(tc_dense_spec(1024, 2304, 8, false), (TC_TINY_THREADS, 64, TC_W1_M16_SMEM));
        assert_eq!(tc_dense_spec(6656, 2304, 32, false), (64, 208, TC_W2_SMEM));
        assert_eq!(tc_dense_spec(6656, 2304, 16, false), (64, 208, TC_W2_M16_SMEM));
        assert_eq!(tc_dense_spec(200_058, 2304, 32, false), (128, 3126, TC_W4_SMEM));
        assert_eq!(tc_dense_spec(200_058, 2304, 16, false), (128, 3126, TC_W4_M16_SMEM));
        assert_eq!(tc_dense_spec(1056, 2304, 4000, true), (TC_THREADS, 9 * 63, TC_BIG_FP32_SMEM));
        assert!(!tc_big_fp32_wanted(2304, 3072, 4000));
        assert!(!tc_big_fp32_wanted(8600, 2304, 4000));
        assert_eq!(tc_dense_spec(64, 4608, 32, true), (32, 64, TC_N1_SMEM));
        assert_eq!(tc_dense_spec(128, 2304, 32, true), (32, 128, TC_N1_SMEM));
        assert_eq!(tc_dense_spec(64, 4608, 4000, true), (128, 125, TC_N4_SMEM));
        assert_eq!(tc_dense_spec(128, 2304, 4000, true), (256, 125, TC_N8_SMEM));
        assert_eq!(tc_dense_spec(4608, 64, 32, true), (64, 144, TC_SHORT_SMALL_SMEM));
        assert_eq!(tc_dense_spec(4608, 64, 4000, true), (256, 63 * 72, TC_SHORT_BIG_SMEM));
        assert_eq!(tc_dense_spec(1024, 2304, 125, false), (128, 64, TC_COOP_M128_SMEM));
        assert_eq!(tc_dense_spec(2304, 3072, 125, false), (128, 144, TC_COOP_M128_SMEM));
        assert_eq!(tc_dense_spec(8600, 2304, 125, false), (128, 538, TC_COOP_M128_SMEM));
        assert_eq!(tc_dense_spec(1024, 2304, 180, false), (64, 6 * 32, TC_MED_W2_SMEM));
        assert_eq!(tc_dense_panel_smem(2304, 3072, 8, 8), 4 * 16 * 128 + 8 * 128);
        assert_eq!(tc_dense_panel_smem(2304, 3072, 16, 8), 8_192 + 16_384 + 8 * 128);
        assert_eq!(tc_dense_panel_smem(6656, 2304, 16, 4), 4 * 16 * 128 + 2 * 4 * 128);
        assert_eq!(tc_dense_panel_smem(6656, 2304, 32, 4), 3 * 32 * 128 + 2 * 4 * 128);
        assert_eq!(tc_dense_panel_smem(200_058, 2304, 32, 1), 3 * 32 * 128 + 4 * 128);
        assert_eq!(tc_dense_panel_smem(2304, 3072, 32, 8), 8_192 + 16_384 + 8 * 128);
        assert_eq!(tc_dense_panel_smem(2304, 3072, 125, 8), TC_COOP_M128_SMEM);
        assert_eq!(tc_tile_part(200_058, 2304, 1024), Some(200_058));
        assert_eq!(tc_tile_part(200_058, 2304, 32), Some(200_058));
        assert_eq!(tc_tile_part(TC_DENSE_ROWS_MAX + 128, 2304, 32), Some(131_200));
        assert_eq!(tc_tile_part(2304, 2304, 7), Some(2304));
        assert_eq!(tc_tile_part(2304, 2304, 0), None);
        assert!(check_tc_dense(0, cols, inputs, rows, w, x, y).is_err());
        assert!(check_tc_dense(rows, 96, inputs, rows, span(w.address, (rows * 96 * 2) as u64), span(x.address, (inputs * 96 * 4) as u64), y).is_err());
        assert!(check_tc_dense(rows, cols, inputs, rows - 1, w, x, y).is_err());
        assert!(check_tc_dense(rows, cols, inputs, rows, span(w.address + 8, w.bytes), x, y).is_err());
        assert!(check_tc_dense(rows, cols, inputs, rows, w, span(x.address + 4, x.bytes), y).is_err());
        assert!(check_tc_dense(rows, cols, inputs, rows, w, x, span(y.address + 2, y.bytes)).is_err());
        assert!(check_tc_dense(rows, cols, inputs, rows, w, x, span(x.address, y.bytes)).is_err());
    }

    #[test]
    fn tc_split_admission_and_refusals() {
        let (cols, inputs) = (2304, 64);
        let segs = [3072usize, 384, 24, 3072, 1024, 1024];
        let rows: usize = segs.iter().sum();
        assert_eq!(rows, 8600);
        let w = span(1 << 20, (rows * cols * 2) as u64);
        let x = span(1 << 30, (inputs * cols * 4) as u64);
        let mut outs = Vec::new();
        let mut addr = 1u64 << 32;
        for &c in &segs {
            let bytes = (inputs * c * 4) as u64;
            outs.push(SplitOut { span: span(addr, bytes), cols: c });
            addr += bytes + 64;
        }
        assert_eq!(check_tc_split(rows, cols, inputs, w, x, &outs), Ok(()));
        // Odd split width refused.
        let mut bad = outs.clone();
        bad[2].cols = 23;
        assert!(check_tc_split(rows - 1, cols, inputs, span(w.address, ((rows - 1) * cols * 2) as u64), x, &bad).is_err());
        // Overlapping split outputs refused.
        let mut bad = outs.clone();
        bad[1].span.address = bad[0].span.address;
        assert!(check_tc_split(rows, cols, inputs, w, x, &bad).is_err());
        // Sum mismatch refused.
        assert!(check_tc_split(rows, cols, inputs, w, x, &outs[..5]).is_err());
    }

    #[test]
    fn moe_route_and_tc_moe_checks() {
        let (rows, picks, experts) = (4096, 8, 128);
        let all = rows * picks;
        assert_eq!(moe_max_tiles(all, experts), 640);
        let route = span(1 << 20, (rows * 2 * picks * 4) as u64);
        let w_out = span(1 << 24, (all * 4) as u64);
        let in_r = span(1 << 25, (all * 4) as u64);
        let out_s = span(1 << 26, (all * 4) as u64);
        let meta = span(1 << 27, (moe_tile_meta_words(all, experts) * 4) as u64);
        assert_eq!(check_moe_group_route(rows, 2 * picks, picks, experts, route, w_out, in_r, out_s, meta), Ok(()));
        assert!(check_moe_group_route(rows, 2 * picks - 1, picks, experts, route, w_out, in_r, out_s, meta).is_err());
        assert!(check_moe_group_route(rows, 2 * picks, picks, experts, route, w_out, w_out, out_s, meta).is_err());

        let (n, k) = (2304, 2304);
        let stack = span(1 << 32, (experts * n * k * 2) as u64);
        let x = span(1 << 30, (rows * k * 4) as u64);
        let y = span(1 << 31, (all * n * 4) as u64);
        assert_eq!(check_tc_moe(n, k, all, experts, stack, meta, x, rows, Some(in_r), y, all, None), Ok(()));
        assert!(check_tc_moe(n, k, all, experts, stack, meta, x, rows, None, y, all, None).is_err());
        assert!(check_tc_moe(n, k, all, experts, stack, meta, x, rows, Some(in_r), span(stack.address, y.bytes), all, None).is_err());

        // Packed MoE (NVFP4 GateUp with PAIRS + MXFP8 Down).
        for &(wide, pairs, pn, pk) in &[(false, true, 2304usize, 2304usize), (true, false, 2304, 1152)] {
            let ubytes = crate::packed_product::layout(pn, pk, wide).bytes as u64;
            let pstack = span(1 << 32, experts as u64 * ubytes);
            let px = span(1 << 30, (rows * pk * 4) as u64);
            let py = span(1 << 31, (all * pn * 4) as u64);
            assert_eq!(check_tc_packed_moe(pn, pk, wide, pairs, all, experts, pstack, meta, px, rows, Some(in_r), py, all, None, None), Ok(()));
            assert!(tc_packed_moe_admitted(pn, pk, wide, pairs, all, experts, moe_max_tiles(all, experts), ubytes as usize, pstack.address as usize, px.address as usize, TC_THREADS));
            assert!(check_tc_packed_moe(pn, pk, wide, pairs, all, experts, span(pstack.address, pstack.bytes - 16), meta, px, rows, Some(in_r), py, all, None, None).is_err());
            assert!(check_tc_packed_moe(pn, pk, wide, pairs, all, experts, pstack, meta, px, rows, Some(in_r), span(pstack.address, py.bytes), all, None, None).is_err());
        }
        // Gated GateUp: half-width output, mixed rows only, exact table size, no overlap with the output.
        {
            let (pn, pk) = (2304usize, 2304usize);
            let ubytes = crate::packed_product::layout(pn, pk, false).bytes as u64;
            let pstack = span(1 << 32, experts as u64 * ubytes);
            let px = span(1 << 30, (rows * pk * 4) as u64);
            let half = span(1 << 31, (all * pn / 2 * 4) as u64);
            let table = span(1 << 29, (2 * crate::fast_ops::CAPPED_GATE_TABLE_WORDS * 4) as u64);
            let check = |pairs: bool, out: Span, t: Span| check_tc_packed_moe(pn, pk, false, pairs, all, experts, pstack, meta, px, rows, Some(in_r), out, all, None, Some(t));
            assert_eq!(check(true, half, table), Ok(()));
            assert!(check(false, half, table).is_err());
            assert!(check(true, span(half.address, half.bytes * 2), table).is_err());
            assert!(check(true, half, span(table.address, table.bytes - 4)).is_err());
            assert!(check(true, half, span(table.address + 2, table.bytes)).is_err());
            assert!(check(true, half, span(half.address, table.bytes)).is_err());
        }
        // Half Down (`HALF_OUT`): `all * rows` BF16 values, ungated with no pairs only; the FP32 size is refused.
        {
            let (pn, pk) = (2304usize, 1152usize);
            let ubytes = crate::packed_product::layout(pn, pk, true).bytes as u64;
            let pstack = span(1 << 32, experts as u64 * ubytes);
            let px = span(1 << 30, (rows * pk * 4) as u64);
            let (half, full) = (span(1 << 31, (all * pn * 2) as u64), span(1 << 31, (all * pn * 4) as u64));
            let check = |pairs: bool, out: Span, gate: Option<Span>| check_tc_packed_moe_form(pn, pk, true, pairs, all, experts, pstack, meta, px, rows, Some(in_r), out, all, None, gate, true);
            assert_eq!(check(false, half, None), Ok(()));
            assert!(check(false, full, None).is_err());
            assert!(check(true, half, None).is_err());
            assert!(check(false, half, Some(span(1 << 29, (2 * crate::fast_ops::CAPPED_GATE_TABLE_WORDS * 4) as u64))).is_err());
        }
        // Odd rows refused with PAIRS; 64 cols refused with MXFP8 (requires multiple of 128).
        let u64 = crate::packed_product::layout(64, 64, true).bytes as u64;
        assert!(check_tc_packed_moe(64, 64, true, false, all, experts, span(1 << 32, experts as u64 * u64), meta, span(1 << 30, (rows * 64 * 4) as u64), rows, Some(in_r), span(1 << 31, (all * 64 * 4) as u64), all, None, None).is_err());
        let u63 = crate::packed_product::layout(63, 64, false).bytes as u64;
        assert!(check_tc_packed_moe(63, 64, false, true, all, experts, span(1 << 32, experts as u64 * u63), meta, span(1 << 30, (rows * 64 * 4) as u64), rows, Some(in_r), span(1 << 31, (all * 63 * 4) as u64), all, None, None).is_err());
    }
}
