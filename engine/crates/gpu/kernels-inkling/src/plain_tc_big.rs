//! Tensor-core BF16 big-tile dense product for prefill (`resident_launch_plain_tc_big`, kind `PLAIN_TC_BIG = 115`).
//!
//! Computes the same words as `PLAIN_TC_ROWS` Mode 0 (`TC_MODE_DENSE`): `Y = X * W^T` for the packed BF16 input
//! `X: [M, K]` (the `TC_MODE_PACK` launch rounds the FP32 input once with `cvt.rn.bf16x2.f32` before this launch) and
//! `W: [N, K]` (BF16, two a word, row-major), into one `[M, stride]` FP32 output or up to 8 split outputs.
//!
//! Each output word is the chain of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` over `k = 0, 16, ...,
//! K - 16` in strictly ascending order from `+0.0`: the chain of `PLAIN_TC_ROWS`. A word depends only on its own
//! input row and weight row, so the larger tile does not change any word.
//!
//! Tile: `BIG_BM = 128` input rows x `BIG_BN = 128` weight rows x `TC_BK = 32` columns a step, 256 threads = 8 warps
//! (2 row warps x 4 column warps), each warp `64 x 32` (4 x 4 MMA tiles, 64 FP32 accumulators a thread). The block
//! reads each A and B element once a step for 128 output columns or rows (64 in `PLAIN_TC_ROWS`), so the L2 reads of
//! a product halve, and a warp runs 16 MMAs for 6 `ldmatrix` (4 for 3 in `PLAIN_TC_ROWS`).
//!
//! Shared memory (`49,152` bytes static, 2 blocks/SM on SM121a): three `cp.async` stages of A (`128 x 32` BF16 =
//! 8,192 B) and B (8,192 B), XOR-swizzled for conflict-free `ldmatrix`. Two steps are in flight while one runs.
//!
//! Launch: grid = `tc_big_blocks(rows, inputs)`, 256 threads, no dynamic shared memory. The host uses this form for
//! large prefill products only (`tc_rows::tc_big_wanted`); decode products stay on `PLAIN_TC_ROWS`.
//!
//! Narrow form (`TC_MODE_NARROW`, at most `NARROW_BN = 128` weight rows, `tc_rows::tc_narrow_wanted`): the input is
//! the FP32 rows themselves (one input, or the two halves of a `GateDown` pair), copied with `cp.async` four steps
//! deep and rounded with `cvt.rn.bf16x2.f32` as each A fragment is made, the rounding `TC_MODE_PACK` does, so no pack
//! launch runs. A block takes `NARROW_BM = 16` input rows and all 128 (zero-padded) weight rows; warp `w` takes
//! columns `16 w .. 16 w + 16` (2 MMA tiles). The words are those of `PLAIN_TC_ROWS` (the same BF16 values, the same
//! chain). Grid = `tc_narrow_blocks(inputs)`.

use crate::block_ops::{NORM_CHUNK, root};
use crate::fixed_math::{rn, tanh};
use crate::tc_ptx::{
    cp_async_wait_two, cvt_bf16x2, l2_evict_first, ld_global_nc_u32, ld_global_nc_v4, ld_global_nc_v4_u32,
    ldmatrix_x4, load_u32, mma_m16n8k16_bf16, shfl_xor_f32, shfl_xor_u32, st_shared_zero16, store_u32,
};
use crate::tc_rows::{
    BIG_BM, BIG_BN, GATE_OUTER_BM, GATE_OUTER_HALF_OUT, GATE_OUTER_TABLE_AT, NARROW_BM, NARROW_BN,
    NARROW_HALF_BM, TC_BK, TC_MODE_DENSE, TC_MODE_GATE_BLEND_NORM, TC_MODE_NARROW,
    TC_MODE_NARROW_HALF, TC_SPLIT_MAX, TC_THREADS, gate_outer_admitted, gate_outer_row_blocks,
    tc_big_admitted,
};

const A_BYTES: usize = BIG_BM * TC_BK * 2; // 8,192 bytes
const B_BYTES: usize = BIG_BN * TC_BK * 2; // 8,192 bytes
const STAGE_BYTES: usize = A_BYTES + B_BYTES; // 16,384 bytes
const STAGES: usize = 3;
const BLOCK_BYTES: usize = STAGES * STAGE_BYTES; // 49,152 bytes
/// Bytes between two 16-row MMA tiles of a stage (16 rows of 64 bytes).
const TILE16: u32 = 16 * 64;
/// Fused `TC_MODE_GATE_BLEND_NORM` shared layout inside `BLOCK_BYTES`:
/// `11` chunks of `8` rows x `136` floats (`+8` pad eliminates bank conflicts across `(g, t)`) = `47,872 B`,
/// followed by `8` rows x `33` u32 chunk-sum/scale slots = `1,056 B` (`48,928 B <= 49,152 B`).
const GO_SMEM_CHUNKS: usize = 11;
const GO_ROW_STRIDE: usize = NORM_CHUNK + 8; // 136 floats (544 B)
const GO_CHUNK_WORDS: usize = GATE_OUTER_BM * GO_ROW_STRIDE; // 1,088 floats (4,352 B)
const GO_SUMS_AT: u32 = (GO_SMEM_CHUNKS * GO_CHUNK_WORDS * 4) as u32; // 47,872 B
const _: () = assert!(BIG_BM == 128 && BIG_BN == 128 && TC_BK == 32 && TC_THREADS == 256);
const _: () = assert!(BLOCK_BYTES == 49_152);
const _: () = assert!(GO_SUMS_AT as usize + GATE_OUTER_BM * 33 * 4 <= BLOCK_BYTES);
/// Narrow form: four stages of 16 FP32 A rows (32 values and 8 pad words a row, so the 64-bit fragment loads of a
/// half-warp meet 32 distinct banks) and 128 BF16 B rows.
const N_STAGES: usize = 4;
const NA_STRIDE: usize = 160;
const NA_BYTES: usize = NARROW_BM * NA_STRIDE; // 2,560 bytes
const NB_BYTES: usize = NARROW_BN * TC_BK * 2; // 8,192 bytes
const N_STAGE: usize = NA_BYTES + NB_BYTES; // 10,752 bytes
const NH_A_BYTES: usize = NARROW_HALF_BM * TC_BK * 2; // 4,096 bytes
const NH_STAGE: usize = NH_A_BYTES + NB_BYTES; // 12,288 bytes
const _: () = assert!(NARROW_BM == 16 && NARROW_HALF_BM == 64 && NARROW_BN == 128 && N_STAGE % 16 == 0 && N_STAGES * N_STAGE <= BLOCK_BYTES);
const _: () = assert!(NH_STAGE % 16 == 0 && N_STAGES * NH_STAGE == BLOCK_BYTES);

/// Zero-bank-conflict XOR-swizzled byte offset within a 64-byte-stride BF16 tile (`row`, 16-byte `chunk` in 0..4).
#[inline(always)]
const fn swiz(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Adaptive serpentine L2 tile swizzle for `TC_MODE_DENSE`:
/// - Wide weight matrices (`n_tiles > m_tiles`, e.g. `QKV`, `Up/Gate`): width-8 N-strips traverse M serpentine so
///   the smaller `W` strip stays L2-resident across all `m_tiles` instead of reloading `W` every M-strip.
/// - Medium/narrow matrices (`n_tiles <= m_tiles`): height-16 M-strips for `n_tiles > 12` (`O_proj`), height-8 for
///   `n_tiles <= 12` (`K/V`) so active `X + W` stays inside the 24 MB L2.
#[inline(always)]
const fn big_tile_coords(block: usize, rows: usize, inputs: usize) -> (usize, usize) {
    let m_tiles = inputs.div_ceil(BIG_BM);
    let n_tiles = rows.div_ceil(BIG_BN);
    if n_tiles > m_tiles {
        let strip_w = 8;
        let group_size = strip_w * m_tiles;
        let group = block / group_size;
        let first_n = group * strip_w;
        let width = if n_tiles - first_n < strip_w { n_tiles - first_n } else { strip_w };
        let within = block - group * group_size;
        let m_raw = within / width;
        let tile_m = if (group & 1) == 0 { m_raw } else { m_tiles - 1 - m_raw };
        (tile_m, first_n + within % width)
    } else {
        let strip_h = if n_tiles > 12 { 16 } else { 8 };
        let group_size = strip_h * n_tiles;
        let group = block / group_size;
        let first_m = group * strip_h;
        let height = if m_tiles - first_m < strip_h { m_tiles - first_m } else { strip_h };
        let within = block - group * group_size;
        let n_raw = within / height;
        let tile_n = if (group & 1) == 0 { n_raw } else { n_tiles - 1 - n_raw };
        (first_m + within % height, tile_n)
    }
}

// The dense form of the command at payload `$w`, shared memory at `$s`. `$h` is true in the copy for the launches with a
// BF16 output (`replay`); the other copy has no code for them. `$S` is the dollar sign of the macros inside.
// SAFETY of a use: the dense-form points of the `replay` contract.
macro_rules! dense_body { ($S:tt, $h:expr, $w:ident, $s:ident) => {{
        let weights = *$w as *const u32;
        let input = *$w.add(1) as *const u32;
        let rows = *$w.add(3) as usize;
        let k = *$w.add(4) as usize;
        let inputs = *$w.add(5) as usize;
        let stride = *$w.add(6) as usize;
        let mode = *$w.add(7);
        let in_table = *$w.add(8) as *const u32;
        let out_table = *$w.add(10) as *const u32;
        let input2 = *$w.add(12) as *const u32;
        let split_count = *$w.add(15) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        if mode != TC_MODE_DENSE || !in_table.is_null() || !out_table.is_null() || !input2.is_null()
            || split_count > TC_SPLIT_MAX || (split_count == 0 && stride < rows)
            || !tc_big_admitted(rows, k, inputs, weights as usize, input as usize, threads)
        {
            core::arch::nvptx::trap();
        }

        let m_tiles = inputs.div_ceil(BIG_BM);
        let n_tiles = rows.div_ceil(BIG_BN);
        if block >= m_tiles * n_tiles { return; }
        let (tile_m, tile_n) = big_tile_coords(block, rows, inputs);

        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (warp_m, warp_n) = (warp / 4, warp % 4);

        // Word offsets (below 2^32, `tc_big_admitted`).
        let kw = (k / 2) as u32;
        let row = tid / 4;
        let chunk = tid % 4;
        let a_r = tile_m * BIG_BM + row;
        let b_r = tile_n * BIG_BN + row;
        let (a_ok0, a_ok1) = (a_r < inputs, a_r + 64 < inputs);
        let (b_ok0, b_ok1) = (b_r < rows, b_r + 64 < rows);
        let a_src = a_r as u32 * kw + chunk as u32 * 4;
        let b_src = b_r as u32 * kw + chunk as u32 * 4;
        let step64 = 64 * kw;
        let (dst0, dst1) = (swiz(row, chunk), swiz(row + 64, chunk));
        let mut stage = 0usize;
        while stage < STAGES {
            let s = $s + (stage * STAGE_BYTES) as u32;
            if !a_ok0 { st_shared_zero16(s + dst0); }
            if !a_ok1 { st_shared_zero16(s + dst1); }
            if !b_ok0 { st_shared_zero16(s + A_BYTES as u32 + dst0); }
            if !b_ok1 { st_shared_zero16(s + A_BYTES as u32 + dst1); }
            stage += 1;
        }
        let issue = |s: u32, c_idx: u32| {
            let at = c_idx * 16;
            if a_ok0 { crate::ptx::copy16(s + dst0, input.add((a_src + at) as usize)); }
            if a_ok1 { crate::ptx::copy16(s + dst1, input.add((a_src + step64 + at) as usize)); }
            let b = s + A_BYTES as u32;
            if b_ok0 { crate::ptx::copy16(b + dst0, weights.add((b_src + at) as usize)); }
            if b_ok1 { crate::ptx::copy16(b + dst1, weights.add((b_src + step64 + at) as usize)); }
        };

        let a_off = swiz(warp_m * 64 + (lane & 15), lane >> 4);
        let b_off = A_BYTES as u32 + swiz(warp_n * 32 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
        let (mut c00, mut c01, mut c02, mut c03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c10, mut c11, mut c12, mut c13) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c20, mut c21, mut c22, mut c23) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c30, mut c31, mut c32, mut c33) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let chunks = (k / TC_BK) as u32;

        issue($s, 0);
        crate::ptx::commit();
        if chunks > 1 { issue($s + STAGE_BYTES as u32, 1); }
        crate::ptx::commit();

        macro_rules! mma4 {
            ($S a:expr, $S bp0:ident, $S bp1:ident, $S d0:ident, $S d1:ident, $S d2:ident, $S d3:ident) => {{
                $S d0 = mma_m16n8k16_bf16($S a, [$S bp0[0], $S bp0[1]], $S d0);
                $S d1 = mma_m16n8k16_bf16($S a, [$S bp0[2], $S bp0[3]], $S d1);
                $S d2 = mma_m16n8k16_bf16($S a, [$S bp1[0], $S bp1[1]], $S d2);
                $S d3 = mma_m16n8k16_bf16($S a, [$S bp1[2], $S bp1[3]], $S d3);
            }};
        }

        let last = $s + 2 * STAGE_BYTES as u32;
        let (mut cur, mut ahead) = ($s, last);
        let mut c_idx = 0u32;
        while c_idx < chunks {
            crate::ptx::wait_one();
            core::arch::nvptx::_syncthreads();
            // Double-buffered `ldmatrix` across all 8 A-tiles and 2 K-substeps; start tensor core pipes BEFORE
            // issuing the next `cp.async` stage so shared-to-MMA latency overlaps global prefetch address math.
            let (a_lo, b_lo) = (cur + a_off, cur + b_off);
            let (a_hi, b_hi) = (cur + (a_off ^ 32), cur + (b_off ^ 32));
            let bp0 = ldmatrix_x4(b_lo);
            let bp1 = ldmatrix_x4(b_lo + TILE16);
            let a0 = ldmatrix_x4(a_lo);
            let a1 = ldmatrix_x4(a_lo + TILE16);
            mma4!(a0, bp0, bp1, c00, c01, c02, c03);
            let a2 = ldmatrix_x4(a_lo + 2 * TILE16);
            mma4!(a1, bp0, bp1, c10, c11, c12, c13);
            let a3 = ldmatrix_x4(a_lo + 3 * TILE16);
            mma4!(a2, bp0, bp1, c20, c21, c22, c23);
            if c_idx + 2 < chunks { issue(ahead, c_idx + 2); }
            crate::ptx::commit();
            let bp2 = ldmatrix_x4(b_hi);
            let bp3 = ldmatrix_x4(b_hi + TILE16);
            let b0 = ldmatrix_x4(a_hi);
            mma4!(a3, bp0, bp1, c30, c31, c32, c33);
            let b1 = ldmatrix_x4(a_hi + TILE16);
            mma4!(b0, bp2, bp3, c00, c01, c02, c03);
            let b2 = ldmatrix_x4(a_hi + 2 * TILE16);
            mma4!(b1, bp2, bp3, c10, c11, c12, c13);
            let b3 = ldmatrix_x4(a_hi + 3 * TILE16);
            mma4!(b2, bp2, bp3, c20, c21, c22, c23);
            mma4!(b3, bp2, bp3, c30, c31, c32, c33);
            ahead = cur;
            cur = if cur == last { $s } else { cur + STAGE_BYTES as u32 };
            c_idx += 1;
        }
        crate::ptx::wait();

        // Epilogue: store 4x4 MMA output tiles (64x32 per warp) with `evict_first` v2 stores.
        // Recompute coords from special registers + volatile payload reads so they do not hold live regs across MMA.
        let output = $w.add(2).read_volatile() as *mut f32;
        let rows = $w.add(3).read_volatile() as usize;
        let inputs = $w.add(5).read_volatile() as usize;
        let stride = $w.add(6).read_volatile() as usize;
        let split_count = $w.add(15).read_volatile() as usize;
        let blk_e = core::arch::nvptx::_block_idx_x() as usize;
        let tid_e = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp_e, lane_e) = (tid_e >> 5, tid_e & 31);
        let (warp_m_e, warp_n_e) = (warp_e >> 2, warp_e & 3);
        let (tile_m_e, tile_n_e) = big_tile_coords(blk_e, rows, inputs);
        if split_count > TC_SPLIT_MAX || (split_count == 0 && stride < rows) || tile_n_e * BIG_BN >= rows {
            core::arch::nvptx::trap();
        }
        // The half form (`tc_rows::TC_BIG_HALF_OUT` in word 13, one output, an even `rows`): word 2 holds
        // `inputs * rows` BF16 values, 16 bits each: `rn` of each product word, the rounding that the reader of this
        // output applies to each word it reads. The last `taps` input rows also go as they are to the ring after the
        // step (word 14, `[taps][rows]` FP32): input row `inputs - taps + j` to slot `(slot + j) mod taps`.
        // SAFETY: the host (`tc_rows::check_tc_big_half_out`) checked that the output holds `inputs * rows / 2` words
        // and the ring `taps * rows` words, word-aligned and apart from each other and from the operands. Each pair
        // of columns has one thread a row, so each output word and each ring word has one writer.
        let half_e = if $h { $w.add(13).read_volatile() } else { 0 };
        let ring_e = if $h { $w.add(14).read_volatile() as *mut f32 } else { core::ptr::null_mut() };
        let half_out = $h && (half_e & crate::tc_rows::TC_BIG_HALF_OUT) != 0;
        let (slot_e, taps_e) = (((half_e >> 8) & 0xff) as usize, ((half_e >> 16) & 0xff) as usize);
        if half_out && (split_count != 0 || stride != rows || !rows.is_multiple_of(2) || ring_e.is_null() || taps_e == 0
            || taps_e > inputs || slot_e >= taps_e || ((output as usize) | (ring_e as usize)) & 3 != 0)
        {
            core::arch::nvptx::trap();
        }
        let evict = l2_evict_first();
        let (g, t) = (lane_e >> 2, lane_e & 3);
        let m_base = tile_m_e * BIG_BM + warp_m_e * 64 + g;
        // A split output in the half form (`tc_rows::TC_SPLIT_HALF_OUT` in its meta word): it holds `inputs * seg_len`
        // BF16 values, 16 bits each, the pair rounding of its reader (`cvt_bf16x2`, the attention prologue). The
        // target of such a pair is its 16-bit slot of input row 0; the host checked the size of that output.
        let target = |ni: usize| -> (*mut f32, usize, bool, bool) {
            let col = tile_n_e * BIG_BN + warp_n_e * 32 + ni * 8 + t * 2;
            if col >= rows { return (core::ptr::null_mut(), 0, false, false); }
            if split_count == 0 { return (output.add(col), stride, col + 1 < rows, false); }
            let mut seg = 0usize;
            while seg < split_count {
                let meta = *$w.add(16 + 2 * seg + 1);
                let seg_start = (meta & 0xffff_ffff) as usize;
                let seg_len = ((if $h { meta & !crate::tc_rows::TC_SPLIT_HALF_OUT } else { meta }) >> 32) as usize;
                if col >= seg_start && col + 1 < seg_start + seg_len {
                    let seg_ptr = *$w.add(16 + 2 * seg) as *mut f32;
                    if $h && (meta & crate::tc_rows::TC_SPLIT_HALF_OUT) != 0 {
                        return (seg_ptr.cast::<u16>().add(col - seg_start).cast::<f32>(), seg_len, true, true);
                    }
                    return (seg_ptr.add(col - seg_start), seg_len, true, false);
                }
                seg += 1;
            }
            core::arch::nvptx::trap();
        };
        macro_rules! store_col {
            ($S ni:expr, $S d0:ident, $S d1:ident, $S d2:ident, $S d3:ident) => {{
                let (at, step, pair, seg_half) = target($S ni);
                if !at.is_null() {
                    let put = |m: usize, v0: f32, v1: f32| {
                        if m < inputs && half_out {
                            let col = tile_n_e * BIG_BN + warp_n_e * 32 + $S ni * 8 + t * 2;
                            crate::tc_st_u32_hint!(output.cast::<u32>().add((m * rows + col) >> 1), (rn(v0).to_bits() >> 16) | (rn(v1).to_bits() & 0xffff_0000), evict);
                            if m + taps_e >= inputs {
                                let j = slot_e + (m + taps_e - inputs); // below `2 * taps`: no division in the launch
                                tc_st_pair_hint!(ring_e.add((if j >= taps_e { j - taps_e } else { j }) * rows + col), true, v0, v1, evict);
                            }
                        } else if m < inputs && seg_half {
                            crate::tc_st_u32_hint!(at.cast::<u16>().add(m * step).cast::<u32>(), cvt_bf16x2(v0, v1), evict);
                        } else if m < inputs {
                            tc_st_pair_hint!(at.add(m * step), pair, v0, v1, evict);
                        }
                    };
                    put(m_base, $S d0[0], $S d0[1]);
                    put(m_base + 8, $S d0[2], $S d0[3]);
                    put(m_base + 16, $S d1[0], $S d1[1]);
                    put(m_base + 24, $S d1[2], $S d1[3]);
                    put(m_base + 32, $S d2[0], $S d2[1]);
                    put(m_base + 40, $S d2[2], $S d2[3]);
                    put(m_base + 48, $S d3[0], $S d3[1]);
                    put(m_base + 56, $S d3[2], $S d3[3]);
                }
            }};
        }
        store_col!(0, c00, c10, c20, c30);
        store_col!(1, c01, c11, c21, c31);
        store_col!(2, c02, c12, c22, c32);
        store_col!(3, c03, c13, c23, c33);
        }}; }
/// Replay one recorded command of kind `PLAIN_TC_BIG`.
/// SAFETY (caller contract, `resident_launch_plain_tc_big`): `table` and `index` point to an initialized 16-byte
/// aligned `Command` of kind `PLAIN_TC_BIG` recorded by `Backend` after `tc_rows::check_tc_dense` or `check_tc_split`
/// verified every extent, alignment and non-overlap, with the payload of a `PLAIN_TC_ROWS` Mode 0 command and its
/// `TC_MODE_PACK` launch recorded before it.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind traps before reading payload words.
    // - The host checks the weights and output spans and records the pack before the product. `weights` holds
    //   `rows * cols` BF16 values, the packed input holds `inputs * cols` BF16 values, `cols` is a multiple of 64,
    //   and every output span is word-aligned, non-overlapping, and sized to hold all written elements
    //   (`(inputs - 1) * stride + rows` for single, `inputs * seg_len` with even `seg_len` summing to `rows` for
    //   split). Device re-checks `tc_big_admitted`, the mode, null row tables and second input, `stride >= rows`,
    //   and `split_count`.
    // - Every A copy reads row `r < inputs` at words `r * cols / 2 + step * 16 + chunk * 4` (`step < cols / 32`,
    //   `chunk < 4`): inside the packed input, 16-byte aligned. Every B copy reads weight row `n < rows` the same
    //   way: inside `weights`, 16-byte aligned. Rows at or above `inputs` / `rows` are never read (zeroed once).
    //   `tc_big_admitted` bounds both word offsets below 2^32.
    // - Narrow form (`TC_MODE_NARROW`): the host checks the FP32 input spans as for `PLAIN_TC_ROWS` (one input of
    //   `inputs * cols` values, or two halves of `inputs * cols / 2` values with `(cols / 2) % 32 == 0`) and records no
    //   pack. Every A copy reads row `r < inputs` at value `r * width + step_value * 32 + part * 4` of its input
    //   (`width` = `cols`, or `cols / 2` in the half that holds the step): inside it and 16-byte aligned (`cols` is a
    //   multiple of 64). Device re-checks the mode, `rows <= NARROW_BN`, null row tables, `split_count == 0`,
    //   `stride >= rows` and the alignment of the second input.
    // - Shared memory stays strictly inside the block's static `BLOCK_BYTES` (49,152 bytes; the narrow form uses
    //   43,008 of them). One `shared` declaration serves both forms.
    // - All 256 threads participate in every `bar.sync 0` (a block returns whole before its first barrier) and
    //   every warp-synchronous `ldmatrix` / `mma.sync` (each condition around them is uniform over a warp).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_TC_BIG { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        // A dense launch with a BF16 output (`tc_rows::TC_BIG_HALF_OUT` in word 13, or `tc_rows::TC_SPLIT_HALF_OUT` in
        // the meta word of a split output) runs its own copy of the dense form. The selector `sel` is not a form
        // value for such a launch, so its copy is the last branch and `ptxas` puts its code after all other forms.
        // The gate form and the dense form of all other launches then have no code of the BF16 outputs before them
        // (measured: such code before the gate form adds 4 us to each of its decode launches; after it, none).
        let mut halves = false;
        if *words.add(7) == TC_MODE_DENSE {
            halves = (*words.add(13) & crate::tc_rows::TC_BIG_HALF_OUT) != 0;
            let mut seg = 0usize;
            while seg < TC_SPLIT_MAX {
                halves |= (seg as u64) < *words.add(15) && (*words.add(16 + 2 * seg + 1) & crate::tc_rows::TC_SPLIT_HALF_OUT) != 0;
                seg += 1;
            }
        }
        let sel = if halves { u64::MAX } else { *words.add(7) };
        if sel == TC_MODE_NARROW {
            narrow(words, smem);
            return;
        }
        if sel == TC_MODE_NARROW_HALF {
            // Narrow BF16-input form (`BM = 64`, `BN <= 128`, 4 `cp.async` stages = 49,152 B == `BLOCK_BYTES`,
            // XOR-swizzled `ldmatrix_x4` on both A and B, all 256 threads loading A and all 8 warps active on
            // `GateDown` (`rows <= 64`) and `Router` (`65..=128`)). Supports single or dual half-width BF16 inputs.
            // SAFETY: host checked via `check_tc_dense_form(..., true)` or `check_tc_pair_form(..., true)` that
            // weights, input(s) (16B aligned) and output (4B aligned, disjoint) match extents. Shared memory stays
            // within `4 * NH_STAGE = 49,152 == BLOCK_BYTES`. All 256 threads participate in every `_syncthreads`.
            let weights = *words as *const u32;
            let input = *words.add(1) as *const u32;
            let rows = *words.add(3) as usize;
            let k = *words.add(4) as usize;
            let inputs = *words.add(5) as usize;
            let stride = *words.add(6) as usize;
            let in_table = *words.add(8) as *const u32;
            let out_table = *words.add(10) as *const u32;
            let input2 = *words.add(12) as *const u32;
            let split_count = *words.add(15) as usize;
            let threads = core::arch::nvptx::_block_dim_x() as usize;
            let block = core::arch::nvptx::_block_idx_x() as usize;
            if rows > NARROW_BN || !in_table.is_null() || !out_table.is_null() || split_count != 0 || stride < rows
                || (!input2.is_null() && ((input2 as usize & 15) != 0 || (k / 2) % TC_BK != 0))
                || !tc_big_admitted(rows, k, inputs, weights as usize, input as usize, threads)
            {
                core::arch::nvptx::trap();
            }
            if block * NARROW_HALF_BM >= inputs { return; }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let (warp, lane) = (tid >> 5, tid & 31);
            let (warp_m, warp_n) = (warp >> 2, warp & 3);
            let (row, chunk) = (tid >> 2, tid & 3);
            let a_r = block * NARROW_HALF_BM + row;
            let a_ok = a_r < inputs;
            let half_cols = if input2.is_null() { k } else { k / 2 };
            let half_w = half_cols / 2;
            let a_src = (a_r * half_w + chunk * 4) as u32;
            let kw = (k / 2) as u32;
            let (b_ok0, b_ok1) = (row < rows, row + 64 < rows);
            let b_src = row as u32 * kw + chunk as u32 * 4;
            let (dst0, dst1) = (swiz(row, chunk), swiz(row + 64, chunk));
            let mut stage = 0usize;
            while stage < N_STAGES {
                let s = smem + (stage * NH_STAGE) as u32;
                if !a_ok { st_shared_zero16(s + dst0); }
                if !b_ok0 { st_shared_zero16(s + NH_A_BYTES as u32 + dst0); }
                if !b_ok1 { st_shared_zero16(s + NH_A_BYTES as u32 + dst1); }
                stage += 1;
            }
            let half_steps = (half_cols / TC_BK) as u32;
            let issue = |s: u32, c_idx: u32| {
                if a_ok {
                    let (from, step) = if c_idx < half_steps { (input, c_idx) } else { (input2, c_idx - half_steps) };
                    crate::ptx::copy16(s + dst0, from.add((a_src + step * 16) as usize));
                }
                let b = s + NH_A_BYTES as u32;
                let at = c_idx * 16;
                if b_ok0 { crate::ptx::copy16(b + dst0, weights.add((b_src + at) as usize)); }
                if b_ok1 { crate::ptx::copy16(b + dst1, weights.add((b_src + 64 * kw + at) as usize)); }
            };
            let wide_n = rows > 64;
            let n_step = if wide_n { 32usize } else { 16usize };
            let a_off = swiz(warp_m * 32 + (lane & 15), lane >> 4);
            let b_off = NH_A_BYTES as u32 + swiz(warp_n * n_step + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
            let busy = warp_n * n_step < rows;
            let has_m1 = block * NARROW_HALF_BM + warp_m * 32 + 16 < inputs;
            let (mut c00, mut c01, mut c02, mut c03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
            let (mut c10, mut c11, mut c12, mut c13) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
            let chunks = (k / TC_BK) as u32;
            let mut c_idx = 0u32;
            while c_idx < (N_STAGES - 1) as u32 {
                if c_idx < chunks { issue(smem + c_idx * NH_STAGE as u32, c_idx); }
                crate::ptx::commit();
                c_idx += 1;
            }
            let last = smem + ((N_STAGES - 1) * NH_STAGE) as u32;
            let (mut cur, mut ahead) = (smem, last);
            c_idx = 0;
            while c_idx < chunks {
                cp_async_wait_two();
                core::arch::nvptx::_syncthreads();
                if c_idx + 3 < chunks { issue(ahead, c_idx + 3); }
                crate::ptx::commit();
                if busy {
                    macro_rules! nh_ksub {
                        ($xm:literal) => {{
                            let a0 = ldmatrix_x4(cur + (a_off ^ $xm));
                            let b0 = ldmatrix_x4(cur + (b_off ^ $xm));
                            if has_m1 {
                                let a1 = ldmatrix_x4(cur + ((a_off + TILE16) ^ $xm));
                                if wide_n {
                                    let b1 = ldmatrix_x4(cur + (b_off ^ $xm) + TILE16);
                                    c00 = mma_m16n8k16_bf16(a0, [b0[0], b0[1]], c00);
                                    c01 = mma_m16n8k16_bf16(a0, [b0[2], b0[3]], c01);
                                    c10 = mma_m16n8k16_bf16(a1, [b0[0], b0[1]], c10);
                                    c11 = mma_m16n8k16_bf16(a1, [b0[2], b0[3]], c11);
                                    c02 = mma_m16n8k16_bf16(a0, [b1[0], b1[1]], c02);
                                    c03 = mma_m16n8k16_bf16(a0, [b1[2], b1[3]], c03);
                                    c12 = mma_m16n8k16_bf16(a1, [b1[0], b1[1]], c12);
                                    c13 = mma_m16n8k16_bf16(a1, [b1[2], b1[3]], c13);
                                } else {
                                    c00 = mma_m16n8k16_bf16(a0, [b0[0], b0[1]], c00);
                                    c01 = mma_m16n8k16_bf16(a0, [b0[2], b0[3]], c01);
                                    c10 = mma_m16n8k16_bf16(a1, [b0[0], b0[1]], c10);
                                    c11 = mma_m16n8k16_bf16(a1, [b0[2], b0[3]], c11);
                                }
                            } else if wide_n {
                                let b1 = ldmatrix_x4(cur + (b_off ^ $xm) + TILE16);
                                c00 = mma_m16n8k16_bf16(a0, [b0[0], b0[1]], c00);
                                c01 = mma_m16n8k16_bf16(a0, [b0[2], b0[3]], c01);
                                c02 = mma_m16n8k16_bf16(a0, [b1[0], b1[1]], c02);
                                c03 = mma_m16n8k16_bf16(a0, [b1[2], b1[3]], c03);
                            } else {
                                c00 = mma_m16n8k16_bf16(a0, [b0[0], b0[1]], c00);
                                c01 = mma_m16n8k16_bf16(a0, [b0[2], b0[3]], c01);
                            }
                        }};
                    }
                    nh_ksub!(0u32);
                    nh_ksub!(32u32);
                }
                ahead = cur;
                cur = if cur == last { smem } else { cur + NH_STAGE as u32 };
                c_idx += 1;
            }
            crate::ptx::wait();
            let output = words.add(2).read_volatile() as *mut f32;
            let rows = words.add(3).read_volatile() as usize;
            let inputs = words.add(5).read_volatile() as usize;
            let stride = words.add(6).read_volatile() as usize;
            let evict = l2_evict_first();
            let (g, t) = (lane >> 2, lane & 3);
            let m0 = block * NARROW_HALF_BM + warp_m * 32 + g;
            let put = |m: usize, col: usize, v0: f32, v1: f32| {
                if m < inputs && col < rows {
                    tc_st_pair_hint!(output.add(m * stride + col), col + 1 < rows, v0, v1, evict);
                }
            };
            let col = warp_n * n_step + t * 2;
            put(m0, col, c00[0], c00[1]);
            put(m0 + 8, col, c00[2], c00[3]);
            put(m0, col + 8, c01[0], c01[1]);
            put(m0 + 8, col + 8, c01[2], c01[3]);
            if wide_n {
                put(m0, col + 16, c02[0], c02[1]);
                put(m0 + 8, col + 16, c02[2], c02[3]);
                put(m0, col + 24, c03[0], c03[1]);
                put(m0 + 8, col + 24, c03[2], c03[3]);
            }
            if has_m1 {
                put(m0 + 16, col, c10[0], c10[1]);
                put(m0 + 24, col, c10[2], c10[3]);
                put(m0 + 16, col + 8, c11[0], c11[1]);
                put(m0 + 24, col + 8, c11[2], c11[3]);
                if wide_n {
                    put(m0 + 16, col + 16, c12[0], c12[1]);
                    put(m0 + 24, col + 16, c12[2], c12[3]);
                    put(m0 + 16, col + 24, c13[0], c13[1]);
                    put(m0 + 24, col + 24, c13[2], c13[3]);
                }
            }
            return;
        }
        if sel == TC_MODE_GATE_BLEND_NORM {
            // Fused prefill `GateOuter (GateForget+GateGain) + GateBlend + RoundedNorm` (`BM = 8` rows/CTA,
            // 1..=3 gates on `grid.x` so 3 gates sharing `kept` hit L2 simultaneously while all weight matrices
            // stay resident in L2). Keeps `<= 11` blended chunks in bank-conflict-free SMEM (`GO_ROW_STRIDE = 136`)
            // and at most `1` chunk/warp in registers (`r0..r31`, 0 stack, 0 spill, bit-exact with `plain_tc_big` +
            // `GateBlend` + `fast_norm`).
            // SAFETY: host checked via `tc_rows::check_tc_gate_blend_norm` that all `count` gate tuples have valid
            // extents, alignments (weights/inner 16B, kept/msg/norm_w/output 4B), and mutually disjoint outputs.
            // Shared memory uses `GO_SUMS_AT + 8 * 33 * 4 = 48,928 <= BLOCK_BYTES` (49,152 B). Every warp reaches
            // every `mma.sync` and `shfl.sync` uniformly, and all 256 threads reach every `_syncthreads`.
            let dim = *words.add(3) as usize;
            let k = *words.add(4) as usize;
            let rows = *words.add(5) as usize;
            let count = *words.add(6) as usize;
            let threads = core::arch::nvptx::_block_dim_x() as usize;
            let gate_i = core::arch::nvptx::_block_idx_x() as usize;
            let row_block = core::arch::nvptx::_block_idx_y() as usize;
            if !gate_outer_admitted(rows, dim, k, count, threads)
                || core::arch::nvptx::_grid_dim_x() as usize != count
                || core::arch::nvptx::_grid_dim_y() as usize != gate_outer_row_blocks(rows) as usize
                || gate_i >= count
            {
                core::arch::nvptx::trap();
            }
            let r0 = row_block * GATE_OUTER_BM;
            if r0 >= rows { return; }
            let base = GATE_OUTER_TABLE_AT + gate_i * 6;
            let weights = *words.add(base) as *const u32;
            let inner = *words.add(base + 1) as *const f32;
            let kept = *words.add(base + 2) as *const f32;
            let msg = *words.add(base + 3) as *const f32;
            if weights.is_null() || inner.is_null() || kept.is_null() || msg.is_null()
                || ((weights as usize) | (inner as usize)) & 15 != 0
                || ((kept as usize) | (msg as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let (warp, lane) = (tid >> 5, tid & 31);
            let (g, t) = (lane >> 2, lane & 3);
            let r = r0 + g;
            let r_ok = r < rows;
            let k_steps = k / 16;
            let kw = k / 2;
            let chunks = dim / NORM_CHUNK;
            macro_rules! ldf {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
            // Load the entire `K <= 64` BF16 `A` row fragment for row `r` into 8 scalar registers ONCE (0 SMEM!).
            let load_a = |ks: usize| -> (u32, u32) {
                if r_ok && ks < k_steps {
                    let p = inner.add(r * k + ks * 16 + t * 2);
                    (cvt_bf16x2(ldf!(p), ldf!(p.add(1))), cvt_bf16x2(ldf!(p.add(8)), ldf!(p.add(9))))
                } else {
                    (0u32, 0u32)
                }
            };
            let (a00, a20) = load_a(0);
            let (a01, a21) = load_a(1);
            let (a02, a22) = load_a(2);
            let (a03, a23) = load_a(3);
            let k_row = if r_ok { kept.add(r * dim) } else { core::ptr::null() };
            let m_row = if r_ok { msg.add(r * dim) } else { core::ptr::null() };
            let evict = l2_evict_first();
            let k_v2 = ((kept as usize) & 7) == 0;
            let m_v2 = ((msg as usize) & 7) == 0;
            macro_rules! transpose_b_pair {
                ($v:expr) => {{
                    let v = $v;
                    let (mut r0, mut r1, mut r2, mut r3) = (v[0], v[1], v[2], v[3]);
                    let p0 = shfl_xor_u32(if (t & 1) == 0 { r1 } else { r0 }, 1);
                    let p1 = shfl_xor_u32(if (t & 1) == 0 { r3 } else { r2 }, 1);
                    if (t & 1) == 0 { r1 = p0; r3 = p1; } else { r0 = p0; r2 = p1; }
                    let q0 = shfl_xor_u32(if (t & 2) == 0 { r2 } else { r0 }, 2);
                    let q1 = shfl_xor_u32(if (t & 2) == 0 { r3 } else { r1 }, 2);
                    if (t & 2) == 0 { r2 = q0; r3 = q1; } else { r0 = q0; r1 = q1; }
                    ([r0, r1], [r2, r3])
                }};
            }
            macro_rules! blend_ni {
                ($nf:expr) => {{
                    let nf = $nf;
                    let wf_row = weights.add((nf + g) * kw);
                    let wg_row = weights.add((dim + nf + g) * kw);
                    // Coalesced 16-byte `W_outer` loads (128 B cache line in 2 bursts per 4 lanes) + 4x4 butterfly
                    // transpose; compute `df` then issue `kept`/`msg` before `dg` so regs stay <= 124 (0 spill).
                    let (df0, df1) = if k_steps >= 4 {
                        let vf0 = ld_global_nc_v4_u32(wf_row.add(t * 4));
                        let vf1 = ld_global_nc_v4_u32(wf_row.add(16 + t * 4));
                        let (bf0, bf1) = transpose_b_pair!(vf0);
                        let mut df = mma_m16n8k16_bf16([a00, 0, a20, 0], bf0, [0.0f32; 4]);
                        df = mma_m16n8k16_bf16([a01, 0, a21, 0], bf1, df);
                        let (bf2, bf3) = transpose_b_pair!(vf1);
                        df = mma_m16n8k16_bf16([a02, 0, a22, 0], bf2, df);
                        df = mma_m16n8k16_bf16([a03, 0, a23, 0], bf3, df);
                        (df[0], df[1])
                    } else if k_steps >= 2 {
                        let (bf0, bf1) = transpose_b_pair!(ld_global_nc_v4_u32(wf_row.add(t * 4)));
                        let mut df = mma_m16n8k16_bf16([a00, 0, a20, 0], bf0, [0.0f32; 4]);
                        df = mma_m16n8k16_bf16([a01, 0, a21, 0], bf1, df);
                        if k_steps == 3 {
                            let wf_p = wf_row.add(t);
                            df = mma_m16n8k16_bf16([a02, 0, a22, 0], [ld_global_nc_u32(wf_p.add(16)), ld_global_nc_u32(wf_p.add(20))], df);
                        }
                        (df[0], df[1])
                    } else {
                        let wf_p = wf_row.add(t);
                        let df = mma_m16n8k16_bf16([a00, 0, a20, 0], [ld_global_nc_u32(wf_p), ld_global_nc_u32(wf_p.add(4))], [0.0f32; 4]);
                        (df[0], df[1])
                    };
                    // Issue `kept` and `msg` loads with `evict_first` BEFORE `dg` MMAs + tanh so DRAM latency hides.
                    let (k0, k1, m0, m1) = if r_ok {
                        let col = nf + t * 2;
                        let kp = k_row.add(col);
                        let mp = m_row.add(col);
                        let (k0, k1) = if k_v2 { tc_ld_nc_v2_f32_hint!(kp, evict) } else { (ldf!(kp), ldf!(kp.add(1))) };
                        let (m0, m1) = if m_v2 { tc_ld_nc_v2_f32_hint!(mp, evict) } else { (ldf!(mp), ldf!(mp.add(1))) };
                        (k0, k1, m0, m1)
                    } else {
                        (0.0f32, 0.0f32, 0.0f32, 0.0f32)
                    };
                    let (dg0, dg1) = if k_steps >= 4 {
                        let vg0 = ld_global_nc_v4_u32(wg_row.add(t * 4));
                        let vg1 = ld_global_nc_v4_u32(wg_row.add(16 + t * 4));
                        let (bg0, bg1) = transpose_b_pair!(vg0);
                        let mut dg = mma_m16n8k16_bf16([a00, 0, a20, 0], bg0, [0.0f32; 4]);
                        dg = mma_m16n8k16_bf16([a01, 0, a21, 0], bg1, dg);
                        let (bg2, bg3) = transpose_b_pair!(vg1);
                        dg = mma_m16n8k16_bf16([a02, 0, a22, 0], bg2, dg);
                        dg = mma_m16n8k16_bf16([a03, 0, a23, 0], bg3, dg);
                        (dg[0], dg[1])
                    } else if k_steps >= 2 {
                        let (bg0, bg1) = transpose_b_pair!(ld_global_nc_v4_u32(wg_row.add(t * 4)));
                        let mut dg = mma_m16n8k16_bf16([a00, 0, a20, 0], bg0, [0.0f32; 4]);
                        dg = mma_m16n8k16_bf16([a01, 0, a21, 0], bg1, dg);
                        if k_steps == 3 {
                            let wg_p = wg_row.add(t);
                            dg = mma_m16n8k16_bf16([a02, 0, a22, 0], [ld_global_nc_u32(wg_p.add(16)), ld_global_nc_u32(wg_p.add(20))], dg);
                        }
                        (dg[0], dg[1])
                    } else {
                        let wg_p = wg_row.add(t);
                        let dg = mma_m16n8k16_bf16([a00, 0, a20, 0], [ld_global_nc_u32(wg_p), ld_global_nc_u32(wg_p.add(4))], [0.0f32; 4]);
                        (dg[0], dg[1])
                    };
                    if r_ok {
                        ((1.0f32 - tanh(df0)) * k0 + tanh(dg0) * m0, (1.0f32 - tanh(df1)) * k1 + tanh(dg1) * m1)
                    } else {
                        (0.0f32, 0.0f32)
                    }
                }};
            }
            // Phase 1: SMEM-backed chunks (`ch < smem_end`, 0 `rv` regs live).
            // When `smem_end <= 10` and `smem_end % 8 == 2` (e.g. `dim = 2304` -> 10 SMEM chunks), split the 2
            // leftover chunks across all 8 warps (4 warps/chunk, 4 `ni` each) using unused SMEM slot 10 for partial
            // sums so every warp runs identical work (36 `blend_ni!` vs 48 on warps 0..1 -> zero barrier tail).
            const GO_SCRATCH_AT: u32 = (10 * GO_CHUNK_WORDS * 4) as u32;
            let smem_end = chunks.saturating_sub(8);
            let bal2 = smem_end >= 2 && smem_end <= 10 && (smem_end & 7) == 2;
            let full_end = if bal2 { smem_end - 2 } else { smem_end };
            let mut ch = warp;
            while ch < full_end {
                let col_base = ch * NORM_CHUNK;
                let smem_chunk = smem + (ch * GO_CHUNK_WORDS * 4 + g * GO_ROW_STRIDE * 4 + t * 8) as u32;
                let (mut se0, mut se1, mut se2, mut se3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let (mut so0, mut so1, mut so2, mut so3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let mut ni = 0usize;
                while ni < 16 {
                    let (v0, v1) = blend_ni!(col_base + ni * 8);
                    match ni & 3 {
                        0 => { se0 += v0 * v0; so0 += v1 * v1; }
                        1 => { se1 += v0 * v0; so1 += v1 * v1; }
                        2 => { se2 += v0 * v0; so2 += v1 * v1; }
                        _ => { se3 += v0 * v0; so3 += v1 * v1; }
                    }
                    let dst = smem_chunk + (ni * 32) as u32;
                    core::arch::asm!("st.shared.v2.f32 [{p}], {{{a}, {b}}};", p = in(reg32) dst, a = in(reg32) v0, b = in(reg32) v1);
                    ni += 1;
                }
                let e8 = (se0 + se2) + (se1 + se3);
                let o8 = (so0 + so2) + (so1 + so3);
                let e4 = e8 + shfl_xor_f32(e8, 2);
                let o4 = o8 + shfl_xor_f32(o8, 2);
                let e2 = e4 + shfl_xor_f32(e4, 1);
                let o2 = o4 + shfl_xor_f32(o4, 1);
                if t == 0 {
                    store_u32(smem + GO_SUMS_AT + ((g * 33 + ch) * 4) as u32, (e2 + o2).to_bits());
                }
                ch += 8;
            }
            if bal2 {
                let ci = warp >> 2;
                let sw = warp & 3;
                let ch_c = full_end + ci;
                let col_base = ch_c * NORM_CHUNK;
                let smem_chunk = smem + (ch_c * GO_CHUNK_WORDS * 4 + g * GO_ROW_STRIDE * 4 + t * 8) as u32;
                let (mut se, mut so) = (0.0f32, 0.0f32);
                let mut k4 = 0usize;
                while k4 < 4 {
                    let ni = sw + k4 * 4;
                    let (v0, v1) = blend_ni!(col_base + ni * 8);
                    se += v0 * v0;
                    so += v1 * v1;
                    let dst = smem_chunk + (ni * 32) as u32;
                    core::arch::asm!("st.shared.v2.f32 [{p}], {{{a}, {b}}};", p = in(reg32) dst, a = in(reg32) v0, b = in(reg32) v1);
                    k4 += 1;
                }
                let sc_dst = smem + GO_SCRATCH_AT + ((((ci * 8 + g) * 4 + sw) * 4 + t) * 8) as u32;
                store_u32(sc_dst, se.to_bits());
                store_u32(sc_dst + 4, so.to_bits());
            }
            // Phase 2: at most ONE overflow chunk per warp (`reg_ch = smem_end + warp`), kept in scalar registers
            // (`rv0..rv31`, single loop body with `match ni` so `tanh` is emitted once -> fits L1 I-cache, 0 spill).
            let reg_ch = smem_end + warp;
            let (mut rv0, mut rv1, mut rv2, mut rv3, mut rv4, mut rv5, mut rv6, mut rv7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut rv8, mut rv9, mut rv10, mut rv11, mut rv12, mut rv13, mut rv14, mut rv15) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut rv16, mut rv17, mut rv18, mut rv19, mut rv20, mut rv21, mut rv22, mut rv23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut rv24, mut rv25, mut rv26, mut rv27, mut rv28, mut rv29, mut rv30, mut rv31) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            if reg_ch < chunks {
                let col_base = reg_ch * NORM_CHUNK;
                let (mut se0, mut se1, mut se2, mut se3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let (mut so0, mut so1, mut so2, mut so3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let mut ni = 0u32;
                while ni < 16 {
                    let mut ni_v = ni;
                    core::arch::asm!("/* {0} */", inout(reg32) ni_v, options(nomem, nostack));
                    let (v0, v1) = blend_ni!(col_base + (ni_v as usize) * 8);
                    match ni_v {
                        0 => { rv0 = v0; rv1 = v1; se0 += v0 * v0; so0 += v1 * v1; }
                        1 => { rv2 = v0; rv3 = v1; se1 += v0 * v0; so1 += v1 * v1; }
                        2 => { rv4 = v0; rv5 = v1; se2 += v0 * v0; so2 += v1 * v1; }
                        3 => { rv6 = v0; rv7 = v1; se3 += v0 * v0; so3 += v1 * v1; }
                        4 => { rv8 = v0; rv9 = v1; se0 += v0 * v0; so0 += v1 * v1; }
                        5 => { rv10 = v0; rv11 = v1; se1 += v0 * v0; so1 += v1 * v1; }
                        6 => { rv12 = v0; rv13 = v1; se2 += v0 * v0; so2 += v1 * v1; }
                        7 => { rv14 = v0; rv15 = v1; se3 += v0 * v0; so3 += v1 * v1; }
                        8 => { rv16 = v0; rv17 = v1; se0 += v0 * v0; so0 += v1 * v1; }
                        9 => { rv18 = v0; rv19 = v1; se1 += v0 * v0; so1 += v1 * v1; }
                        10 => { rv20 = v0; rv21 = v1; se2 += v0 * v0; so2 += v1 * v1; }
                        11 => { rv22 = v0; rv23 = v1; se3 += v0 * v0; so3 += v1 * v1; }
                        12 => { rv24 = v0; rv25 = v1; se0 += v0 * v0; so0 += v1 * v1; }
                        13 => { rv26 = v0; rv27 = v1; se1 += v0 * v0; so1 += v1 * v1; }
                        14 => { rv28 = v0; rv29 = v1; se2 += v0 * v0; so2 += v1 * v1; }
                        _ => { rv30 = v0; rv31 = v1; se3 += v0 * v0; so3 += v1 * v1; }
                    }
                    ni += 1;
                }
                let e8 = (se0 + se2) + (se1 + se3);
                let o8 = (so0 + so2) + (so1 + so3);
                let e4 = e8 + shfl_xor_f32(e8, 2);
                let o4 = o8 + shfl_xor_f32(o8, 2);
                let e2 = e4 + shfl_xor_f32(e4, 1);
                let o2 = o4 + shfl_xor_f32(o4, 1);
                if t == 0 {
                    store_u32(smem + GO_SUMS_AT + ((g * 33 + reg_ch) * 4) as u32, (e2 + o2).to_bits());
                }
            }
            core::arch::nvptx::_syncthreads();
            // Reload epilogue-only payload words after the barrier so they do not occupy registers across MMA/tanh.
            let eps = f32::from_bits(words.add(9).read_volatile() as u32);
            let mut total = if bal2 {
                let (e8, o8) = if lane < 8 {
                    let ci = lane >> 2;
                    let tl = lane & 3;
                    let sb = smem + GO_SCRATCH_AT + ((((ci * 8 + warp) * 4) * 4 + tl) * 8) as u32;
                    let se0 = f32::from_bits(load_u32(sb));
                    let so0 = f32::from_bits(load_u32(sb + 4));
                    let se1 = f32::from_bits(load_u32(sb + 32));
                    let so1 = f32::from_bits(load_u32(sb + 36));
                    let se2 = f32::from_bits(load_u32(sb + 64));
                    let so2 = f32::from_bits(load_u32(sb + 68));
                    let se3 = f32::from_bits(load_u32(sb + 96));
                    let so3 = f32::from_bits(load_u32(sb + 100));
                    ((se0 + se2) + (se1 + se3), (so0 + so2) + (so1 + so3))
                } else {
                    (0.0f32, 0.0f32)
                };
                let e4 = e8 + shfl_xor_f32(e8, 2);
                let o4 = o8 + shfl_xor_f32(o8, 2);
                let e2 = e4 + shfl_xor_f32(e4, 1);
                let o2 = o4 + shfl_xor_f32(o4, 1);
                let csum = (e2 + o2).to_bits();
                let (c0_bits, c1_bits): (u32, u32);
                core::arch::asm!(
                    "shfl.sync.idx.b32 {d0}, {v}, 0, 31, 0xffffffff; shfl.sync.idx.b32 {d1}, {v}, 4, 31, 0xffffffff;",
                    d0 = out(reg32) c0_bits, d1 = out(reg32) c1_bits, v = in(reg32) csum,
                    options(nomem, nostack),
                );
                if lane == full_end {
                    f32::from_bits(c0_bits)
                } else if lane == full_end + 1 {
                    f32::from_bits(c1_bits)
                } else if lane < chunks {
                    f32::from_bits(load_u32(smem + GO_SUMS_AT + ((warp * 33 + lane) * 4) as u32))
                } else {
                    0.0f32
                }
            } else if lane < chunks {
                f32::from_bits(load_u32(smem + GO_SUMS_AT + ((warp * 33 + lane) * 4) as u32))
            } else {
                0.0f32
            };
            total += shfl_xor_f32(total, 16);
            total += shfl_xor_f32(total, 8);
            total += shfl_xor_f32(total, 4);
            total += shfl_xor_f32(total, 2);
            total += shfl_xor_f32(total, 1);
            if lane == 0 {
                store_u32(smem + GO_SUMS_AT + ((warp * 33 + 32) * 4) as u32, (1.0f32 / root(total / dim as f32 + eps)).to_bits());
            }
            core::arch::nvptx::_syncthreads();
            let norm_w = words.add(base + 4).read_volatile() as *const f32;
            let output = words.add(base + 5).read_volatile() as *mut f32;
            if output.is_null() || ((norm_w as usize) | (output as usize)) & 3 != 0 {
                core::arch::nvptx::trap();
            }
            // Coalesced 32-lanes-per-row epilogue: warp `w` owns row `r0 + w` so every global store is a full
            // 256 B (`v2.u32` half_out) or 512 B (`v4.f32`) transaction with zero partial-sector write amplification.
            let flags = words.add(8).read_volatile();
            let half_out = (flags & GATE_OUTER_HALF_OUT) != 0;
            let nw_v4 = ((norm_w as usize) & 15) == 0;
            let r_w = r0 + warp;
            let rw_ok = r_w < rows;
            let scale_w = f32::from_bits(load_u32(smem + GO_SUMS_AT + ((warp * 33 + 32) * 4) as u32));
            let o_f32 = if rw_ok { output.add(r_w * dim) } else { core::ptr::null_mut() };
            let o_u32 = if rw_ok { output.cast::<u32>().add(r_w * (dim / 2)) } else { core::ptr::null_mut() };
            macro_rules! emit_smem_chunk {
                ($slot:expr, $ch:expr) => {{
                    if rw_ok {
                        let col = ($ch) * NORM_CHUNK + lane * 4;
                        let sp = smem + (($slot) * GO_CHUNK_WORDS * 4 + warp * GO_ROW_STRIDE * 4 + lane * 16) as u32;
                        let raw = crate::ptx::load4(sp);
                        let (v0, v1, v2, v3) = (
                            f32::from_bits(raw[0]) * scale_w, f32::from_bits(raw[1]) * scale_w,
                            f32::from_bits(raw[2]) * scale_w, f32::from_bits(raw[3]) * scale_w,
                        );
                        let (y0, y1, y2, y3) = if norm_w.is_null() {
                            (rn(v0), rn(v1), rn(v2), rn(v3))
                        } else {
                            let nw = if nw_v4 {
                                ld_global_nc_v4(norm_w.add(col))
                            } else {
                                let p = norm_w.add(col);
                                [ldf!(p), ldf!(p.add(1)), ldf!(p.add(2)), ldf!(p.add(3))]
                            };
                            (rn(v0 * nw[0]), rn(v1 * nw[1]), rn(v2 * nw[2]), rn(v3 * nw[3]))
                        };
                        if half_out {
                            tc_st_v2_u32_hint!(o_u32.add(col / 2), cvt_bf16x2(y0, y1), cvt_bf16x2(y2, y3), evict);
                        } else {
                            tc_st_v4_f32_hint!(o_f32.add(col), y0, y1, y2, y3, evict);
                        }
                    }
                }};
            }
            let mut ch = 0usize;
            while ch < smem_end {
                emit_smem_chunk!(ch, ch);
                ch += 1;
            }
            if chunks > smem_end {
                core::arch::nvptx::_syncthreads();
                if reg_ch < chunks {
                    let slot = reg_ch - smem_end;
                    let smem_chunk = smem + (slot * GO_CHUNK_WORDS * 4 + g * GO_ROW_STRIDE * 4 + t * 8) as u32;
                    macro_rules! st_reg {
                        ($ni:literal, $a:expr, $b:expr) => {{
                            let dst = smem_chunk + ($ni * 32) as u32;
                            core::arch::asm!("st.shared.v2.f32 [{p}], {{{a}, {b}}};", p = in(reg32) dst, a = in(reg32) $a, b = in(reg32) $b);
                        }};
                    }
                    st_reg!(0, rv0, rv1); st_reg!(1, rv2, rv3); st_reg!(2, rv4, rv5); st_reg!(3, rv6, rv7);
                    st_reg!(4, rv8, rv9); st_reg!(5, rv10, rv11); st_reg!(6, rv12, rv13); st_reg!(7, rv14, rv15);
                    st_reg!(8, rv16, rv17); st_reg!(9, rv18, rv19); st_reg!(10, rv20, rv21); st_reg!(11, rv22, rv23);
                    st_reg!(12, rv24, rv25); st_reg!(13, rv26, rv27); st_reg!(14, rv28, rv29); st_reg!(15, rv30, rv31);
                }
                core::arch::nvptx::_syncthreads();
                let mut ch = smem_end;
                while ch < chunks {
                    emit_smem_chunk!(ch - smem_end, ch);
                    ch += 1;
                }
            }
            return;
        }
        if sel == TC_MODE_DENSE { dense_body!($, false, words, smem); return; }
        dense_body!($, true, words, smem);
    }
}

/// The narrow form (`TC_MODE_NARROW`) of one `PLAIN_TC_BIG` command at payload `words`, shared memory at `smem`.
/// SAFETY: the contract of `replay` (its narrow-form points); `smem` is the block's `BLOCK_BYTES` shared buffer.
#[inline(always)]
unsafe fn narrow(words: *const u64, smem: u32) {
    // SAFETY: the narrow-form points of the `replay` contract: the command words are the host's checked narrow command,
    // every global read is a checked FP32 input row or weight row below its extent, every store a checked output word,
    // and shared memory stays inside the block's `BLOCK_BYTES` buffer at `smem`.
    unsafe {
        let weights = *words as *const u32;
        let input = *words.add(1) as *const f32;
        let rows = *words.add(3) as usize;
        let k = *words.add(4) as usize;
        let inputs = *words.add(5) as usize;
        let stride = *words.add(6) as usize;
        let in_table = *words.add(8) as *const u32;
        let out_table = *words.add(10) as *const u32;
        let input2 = *words.add(12) as *const f32;
        let split_count = *words.add(15) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        if rows > NARROW_BN || !in_table.is_null() || !out_table.is_null() || split_count != 0 || stride < rows
            || (!input2.is_null() && ((input2 as usize & 15) != 0 || (k / 2) % TC_BK != 0))
            || !tc_big_admitted(rows, k, inputs, weights as usize, input as usize, threads)
        {
            core::arch::nvptx::trap();
        }
        if block * NARROW_BM >= inputs { return; }

        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        // A copies: threads 0..128, row `tid / 8`, 16-byte part `tid % 8` of the 128-byte FP32 step row. Value offsets
        // in the input that holds the step (below 2^32, `tc_big_admitted`); a pair's halves are `cols / 2` wide.
        let a_mine = tid < 128;
        let (a_row, a_part) = (tid / 8 % NARROW_BM, tid % 8);
        let a_r = block * NARROW_BM + a_row;
        let a_ok = a_mine && a_r < inputs;
        let half = if input2.is_null() { k } else { k / 2 };
        let a_src = (a_r * half + a_part * 4) as u32;
        let a_dst = (a_row * NA_STRIDE + a_part * 16) as u32;
        // B copies: rows `row` and `row + 64` of the 128 (zero-padded) weight rows, 16-byte chunk `chunk`.
        let kw = (k / 2) as u32;
        let (row, chunk) = (tid / 4, tid % 4);
        let (b_ok0, b_ok1) = (row < rows, row + 64 < rows);
        let b_src = row as u32 * kw + chunk as u32 * 4;
        let (dst0, dst1) = (swiz(row, chunk), swiz(row + 64, chunk));
        let mut stage = 0usize;
        while stage < N_STAGES {
            let s = smem + (stage * N_STAGE) as u32;
            if a_mine && !a_ok { st_shared_zero16(s + a_dst); }
            if !b_ok0 { st_shared_zero16(s + NA_BYTES as u32 + dst0); }
            if !b_ok1 { st_shared_zero16(s + NA_BYTES as u32 + dst1); }
            stage += 1;
        }
        let half_steps = (half / TC_BK) as u32;
        let issue = |s: u32, c_idx: u32| {
            if a_ok {
                let (from, step) = if c_idx < half_steps { (input, c_idx) } else { (input2, c_idx - half_steps) };
                crate::ptx::copy16(s + a_dst, from.add((a_src + step * 32) as usize).cast());
            }
            let b = s + NA_BYTES as u32;
            let at = c_idx * 16;
            if b_ok0 { crate::ptx::copy16(b + dst0, weights.add((b_src + at) as usize)); }
            if b_ok1 { crate::ptx::copy16(b + dst1, weights.add((b_src + 64 * kw + at) as usize)); }
        };

        // Fragment offsets: A row `g` (and `g + 8`) at its 64-bit pair `t`; B as in the big form, warp columns 16 w.
        let (g, t) = (lane / 4, lane % 4);
        let a_off = (g * NA_STRIDE + t * 8) as u32;
        let b_off = NA_BYTES as u32 + swiz(warp * 16 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
        let busy = warp * 16 < rows;
        let pair = |at: u32| -> u32 {
            let v = crate::ptx::load2(at);
            cvt_bf16x2(f32::from_bits(v[0]), f32::from_bits(v[1]))
        };
        let (mut c0, mut c1) = ([0.0f32; 4], [0.0f32; 4]);
        let chunks = (k / TC_BK) as u32;

        // Prologue: steps 0, 1 and 2 in flight (one group each, empty past the last step).
        let mut c_idx = 0u32;
        while c_idx < (N_STAGES - 1) as u32 {
            if c_idx < chunks { issue(smem + c_idx * N_STAGE as u32, c_idx); }
            crate::ptx::commit();
            c_idx += 1;
        }
        // Step `c` on stage `c % 4`: the wait leaves steps `c + 1` and `c + 2` in flight; the barrier makes step `c`
        // visible and ends every read of stage `(c + 3) % 4` (step `c - 1`) before its next copy. Each accumulator
        // takes its 16-column substeps in strictly ascending K order.
        let last = smem + ((N_STAGES - 1) * N_STAGE) as u32;
        let (mut cur, mut ahead) = (smem, last);
        c_idx = 0;
        while c_idx < chunks {
            cp_async_wait_two();
            core::arch::nvptx::_syncthreads();
            if c_idx + 3 < chunks { issue(ahead, c_idx + 3); }
            crate::ptx::commit();
            if busy {
                let mut ks = 0u32;
                while ks < 2 {
                    let a_at = cur + a_off + ks * 64;
                    let a = [pair(a_at), pair(a_at + 8 * NA_STRIDE as u32), pair(a_at + 32), pair(a_at + 8 * NA_STRIDE as u32 + 32)];
                    let b = ldmatrix_x4(cur + (b_off ^ (ks * 32)));
                    c0 = mma_m16n8k16_bf16(a, [b[0], b[1]], c0);
                    c1 = mma_m16n8k16_bf16(a, [b[2], b[3]], c1);
                    ks += 1;
                }
            }
            ahead = cur;
            cur = if cur == last { smem } else { cur + N_STAGE as u32 };
            c_idx += 1;
        }

        // Epilogue: rows `g` and `g + 8` of the block, columns `16 w + 8 ni + 2 t` and the next.
        let output = words.add(2).read_volatile() as *mut f32;
        let m0 = block * NARROW_BM + g;
        let put = |m: usize, col: usize, v0: f32, v1: f32| {
            if m < inputs && col < rows {
                *output.add(m * stride + col) = v0;
                if col + 1 < rows { *output.add(m * stride + col + 1) = v1; }
            }
        };
        let col = warp * 16 + t * 2;
        put(m0, col, c0[0], c0[1]);
        put(m0 + 8, col, c0[2], c0[3]);
        put(m0, col + 8, c1[0], c1[1]);
        put(m0 + 8, col + 8, c1[2], c1[3]);
    }
}
