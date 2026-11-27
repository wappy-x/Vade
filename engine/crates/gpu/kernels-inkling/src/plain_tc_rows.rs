//! Tensor-core BF16 multi-row product (`resident_launch_plain_tc_rows`, kind `PLAIN_TC_ROWS = 103`).
//!
//! Computes `Y = X * W^T` for `X: [M, K]` (FP32, rounded once to BF16 via `cvt.rn.bf16x2.f32`) and
//! `W: [N, K]` (packed BF16, 2 values per u32, row-major), outputting `Y: [M, N]` FP32.
//!
//! Zero static shared memory: every path validates `%dynamic_smem_size` before any shared access so
//! 1-, 2-, 4-, and 8-warp tiles reach full SM occupancy on SM121a.
//!
//! Fixed deterministic reduction order: `k = 0, 16, 32, ..., K - 16` in strictly ascending order
//! from `+0.0`, zero FP atomics, zero split-K.

use core::arch::asm;
use crate::fixed_math::rn;
use crate::tc_ptx::{
    cp_async_wait_one, cp_async_wait_two, cp_async_wait_zero, cvt_bf16x2, l2_evict_first, ld_global_nc_u32,
    ld_global_nc_v2_u32, ld_global_nc_v4, ld_global_nc_v4_u32_hint, ldmatrix_x4, load_u32,
    mma_m16n8k16_bf16, st_global_f32_hint, st_shared_v4, st_shared_zero16,
};
use crate::tc_rows::{
    BIG_BN, TC_BIG_FP32_SMEM, TC_BK, TC_BM, TC_FLAG_GROUP_PAIR, TC_FLAG_HALF_IN, TC_FLAG_ROUND_OUT, TC_GEN_SMEM,
    TC_GEN_STAGES, TC_GROUP_MAX, TC_GROUP_WORD, TC_MODE_DENSE, TC_MODE_DENSE_FP32, TC_MODE_MOE, TC_MODE_PACK,
    TC_N1_BK, TC_N1_BM, TC_N1_BN, TC_N1_SMEM, TC_N4_BM, TC_N4_BN, TC_N4_SMEM, TC_N8_BN, TC_N8_SMEM,
    TC_PANEL_WORD, TC_PLAIN_BN as TC_BN, TC_SMALL_BM, TC_SPLIT_MAX, TC_THREADS, tc_big_fp32_wanted, tc_coop_w_wanted,
    tc_dense_admitted, tc_dense_fp32_admitted, tc_dense_panel_smem, tc_dense_spec, tc_med_w2_wanted, tc_moe_admitted,
    tc_narrow_1w_wanted, tc_narrow_4w_wanted, tc_short_k_wanted, tc_small_wanted,
};

const GEN_A_BYTES: usize = TC_BM * TC_BK * 2; // 4,096 bytes
const GEN_B_BYTES: usize = TC_BN * TC_BK * 2; // 4,096 bytes
const GEN_STAGE_BYTES: usize = GEN_A_BYTES + GEN_B_BYTES; // 8,192 bytes
const BIG_A_BYTES: usize = TC_BM * TC_BK * 2; // 4,096 bytes
const BIG_STAGE_BYTES: usize = BIG_A_BYTES + BIG_BN * TC_BK * 2; // 12,288 bytes
const BIG_TILE16: u32 = 16 * 64;
const N1_ROW_BYTES: usize = TC_N1_BK * 2 + 16; // 144 bytes (bank-conflict-free 4x64 BF16)
const N1_STAGE_BYTES: usize = TC_N1_BN * N1_ROW_BYTES; // 576 bytes
const N1_STAGES: usize = 8;
const N4_A_BYTES: usize = TC_N4_BM * TC_BK * 4; // 4,096 bytes
const N4_B_BYTES: usize = TC_N4_BN * TC_BK * 2; // 4,096 bytes
const N4_STAGE_BYTES: usize = N4_A_BYTES + N4_B_BYTES; // 8,192 bytes
const N8_B_BYTES: usize = TC_N8_BN * TC_BK * 2; // 8,192 bytes
const N8_STAGE_BYTES: usize = N4_A_BYTES + N8_B_BYTES; // 12,288 bytes
const _: () = assert!(TC_BM == 64 && TC_BN == 64 && TC_BK == 32 && TC_THREADS == 256 && TC_GEN_STAGES == 4);
const _: () = assert!(TC_GEN_STAGES * GEN_STAGE_BYTES == TC_GEN_SMEM && 4 * N4_STAGE_BYTES == TC_N4_SMEM && 3 * N8_STAGE_BYTES == TC_N8_SMEM);
const _: () = assert!(N1_STAGES * N1_STAGE_BYTES == TC_N1_SMEM && TC_N1_BM == 8 && TC_N1_BN == 4 && TC_N1_BK == 64);
const _: () = assert!(BIG_BN == 128 && 3 * BIG_STAGE_BYTES == TC_BIG_FP32_SMEM);

/// Zero-bank-conflict XOR-swizzled byte offset within a 64-byte-stride BF16 tile (`row`, 16-byte `chunk` in 0..4).
#[inline(always)]
const fn swiz(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Zero-bank-conflict XOR-swizzled byte offset within a 128-byte-stride BF16 tile (`row`, 16-byte `chunk` in 0..8).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 128 + ((chunk ^ (row & 7)) * 16)) as u32
}

/// Zero-bank-conflict XOR-swizzled byte offset within a 128-byte-stride FP32 tile (`row`, 16-byte `chunk` in 0..8)
/// for 64-bit `ld.shared.v2` fragment reads across a half-warp with zero padding bytes.
#[inline(always)]
const fn swiz_f32(row: usize, chunk: usize) -> u32 {
    (row * 128 + ((chunk ^ ((row & 3) * 2)) * 16)) as u32
}

/// Replay one recorded command of kind `PLAIN_TC_ROWS`.
/// SAFETY (caller contract, `resident_launch_plain_tc_rows`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `PLAIN_TC_ROWS` recorded by `Backend` after `tc_rows::check_tc_dense`,
/// `check_tc_pair`, `check_tc_group_pair`, `check_tc_split`, or `check_tc_moe` verified every extent,
/// alignment, and non-overlap.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind traps before reading payload words.
    // - Mode 2 (pack): checked FP32 source of `rows` (multiple of 8) floats, private disjoint output of `rows / 2`
    //   packed BF16 words. Uses 0 shared memory.
    // - Mode 4 (direct-FP32 dense): host checked weights, FP32 input(s), optional group triples (1..=3), and
    //   single or 1..=8 disjoint even-width split output span(s). Device re-checks `tc_dense_fp32_admitted`,
    //   `%dynamic_smem_size`, alignment, mode flags, split count, and group index before any memory access.
    // - Mode 0 (packed BF16 dense / split): host checked weights, packed input(s), and single or 1..=8 disjoint
    //   even-width split outputs. Device re-checks `tc_dense_admitted`, `%dynamic_smem_size`, `stride >= rows`,
    //   `split_count <= TC_SPLIT_MAX`, and dual-input constraints before any memory access.
    // - Mode 1 (device-routed MoE): host verified via `check_tc_moe`. Device re-checks `tc_moe_admitted`,
    //   `%dynamic_smem_size >= TC_GEN_SMEM`, `total_m_tiles <= max_m_tiles`, `expert < experts`, `m <= TC_BM`,
    //   `first_pos + m <= all`, and every table index `< in_count` / `< out_count` before dereference.
    // - Dynamic shared memory stays strictly inside `%dynamic_smem_size` checked at entry.
    // - All active threads of a CTA participate in every `bar.sync 0` and every warp-synchronous collective.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_TC_ROWS { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let weights_base = *words as *const u32;
        let input = *words.add(1) as *const f32;
        let rows = *words.add(3) as usize;
        let k = *words.add(4) as usize;
        let inputs = *words.add(5) as usize;
        let stride = *words.add(6) as usize;
        let mode = *words.add(7);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;

        if mode == TC_MODE_PACK {
            let output = *words.add(2) as *mut u32;
            let chunk = block * TC_THREADS + core::arch::nvptx::_thread_idx_x() as usize;
            if threads != TC_THREADS || rows % 8 != 0 { core::arch::nvptx::trap(); }
            if chunk < rows / 8 {
                let src = input.add(chunk * 8);
                let lo = ld_global_nc_v4(src);
                let hi = ld_global_nc_v4(src.add(4));
                output.add(chunk * 4).cast::<[u32; 4]>().write([
                    cvt_bf16x2(lo[0], lo[1]), cvt_bf16x2(lo[2], lo[3]),
                    cvt_bf16x2(hi[0], hi[1]), cvt_bf16x2(hi[2], hi[3]),
                ]);
            }
            return;
        }

        if ((*words.add(8) | *words.add(10)) as usize) & 3 != 0 { core::arch::nvptx::trap(); }
        let smem = 0u32;
        let shared_bytes: u32;
        asm!("mov.u32 {bytes}, %dynamic_smem_size;", bytes = out(reg32) shared_bytes);

        macro_rules! store_pair {
            ($dst:expr, $pair_ok:expr, $v0:expr, $v1:expr) => {{
                let dst: *mut f32 = $dst;
                let (v0, v1): (f32, f32) = ($v0, $v1);
                if $pair_ok && ((dst as usize) & 7) == 0 {
                    asm!("st.global.v2.f32 [{p}], {{{a}, {b}}};",
                        p = in(reg64) dst, a = in(reg32) v0, b = in(reg32) v1, options(nostack));
                } else {
                    *dst = v0;
                    if $pair_ok { *dst.add(1) = v1; }
                }
            }};
        }
        macro_rules! store_split_pair {
            ($output:expr, $stride:expr, $split_count:expr, $m_lim:expr, $m0:expr, $m1:expr, $or0:expr, $or1:expr, $col:expr, $v:expr) => {{
                let col: usize = $col;
                if col < rows {
                    let v: [f32; 4] = $v;
                    if $split_count == 0 {
                        let pair_ok = col + 1 < rows;
                        if $m0 < $m_lim { store_pair!($output.add($or0 * $stride + col), pair_ok, v[0], v[1]); }
                        if $m1 < $m_lim { store_pair!($output.add($or1 * $stride + col), pair_ok, v[2], v[3]); }
                    } else {
                        let mut seg = 0usize;
                        let mut found = false;
                        while seg < $split_count {
                            let meta = *words.add(16 + 2 * seg + 1);
                            let seg_start = (meta & 0xffff_ffff) as usize;
                            let seg_len = (meta >> 32) as usize;
                            if col >= seg_start && col + 1 < seg_start + seg_len {
                                let seg_ptr = *words.add(16 + 2 * seg) as *mut f32;
                                let lc = col - seg_start;
                                if $m0 < $m_lim { store_pair!(seg_ptr.add($or0 * seg_len + lc), true, v[0], v[1]); }
                                if $m1 < $m_lim { store_pair!(seg_ptr.add($or1 * seg_len + lc), true, v[2], v[3]); }
                                found = true;
                                break;
                            }
                            seg += 1;
                        }
                        if !found { core::arch::nvptx::trap(); }
                    }
                }
            }};
        }

        // ============================================================================================================
        // MODE 4: Direct-FP32 Dense Paths (Narrow Decode/Prefill + Short-K GateOuter, zero TC_MODE_PACK)
        // ============================================================================================================
        if mode == TC_MODE_DENSE_FP32 {
            let input2_0 = *words.add(12) as *const f32;
            let flags = *words.add(13);
            let group_count = *words.add(14) as usize;
            let split_count = *words.add(15) as usize;
            let round_out = (flags & TC_FLAG_ROUND_OUT) != 0;
            let grouped = (flags & TC_FLAG_GROUP_PAIR) != 0;
            let half_in = (flags & TC_FLAG_HALF_IN) != 0;
            if (*words.add(8) | *words.add(10)) != 0 || split_count > TC_SPLIT_MAX
                || (split_count == 0 && stride < rows)
                || (flags & !(TC_FLAG_ROUND_OUT | TC_FLAG_GROUP_PAIR | TC_FLAG_HALF_IN)) != 0
                || (grouped && (split_count != 0 || half_in))
                || (half_in && (split_count != 0 || !tc_narrow_1w_wanted(rows, inputs)))
                || (!input2_0.is_null() && (split_count != 0 || (input2_0 as usize & 15) != 0 || (k / 2) % TC_BK != 0))
                || !tc_dense_fp32_admitted(rows, k, inputs, weights_base as usize, input as usize, threads, shared_bytes as usize)
            {
                core::arch::nvptx::trap();
            }

            // Path 6: Overlapped Direct-FP32 `64x128x32` x 3 stages (`36,864 B` dynamic SMEM, 2 blk/SM, 32 accums, zero pack!).
            // Covers `K/V`, `K+KTap` (`512 <= rows <= 1280, inputs >= 512, cols > 64`). Unconditional clamped `LDG.128`
            // on `A(c_idx+2)` before all 16 MMAs prevents conditional-refill compiler stalls; tile coords recomputed at epilogue.
            if tc_big_fp32_wanted(rows, k, inputs) {
                if grouped || round_out || !input2_0.is_null() { core::arch::nvptx::trap(); }
                let m_tiles = inputs.div_ceil(TC_BM);
                let n_tiles = rows.div_ceil(BIG_BN);
                if block >= m_tiles * n_tiles { return; }
                let group_size = 8 * n_tiles;
                let first_m = (block / group_size) * 8;
                let height = (m_tiles - first_m).min(8);
                let within = block % group_size;
                let tile_m = first_m + within % height;
                let tile_n = within / height;

                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid / 32, tid % 32);
                let (warp_m, warp_n) = (warp / 4, warp % 4);
                let (row, chunk) = (tid / 4, tid % 4);
                let kw = k / 2;
                let a_r = tile_m * TC_BM + row;
                let b_r = tile_n * BIG_BN + row;
                let a_ok = a_r < inputs;
                let (b_ok0, b_ok1) = (b_r < rows, b_r + 64 < rows);
                let a_ptr = input.add(a_r.min(inputs - 1) * k + chunk * 8);
                let b_ptr0 = weights_base.add(b_r.min(rows - 1) * kw + chunk * 4);
                let b_ptr1 = weights_base.add((b_r + 64).min(rows - 1) * kw + chunk * 4);
                let (dst0, dst1) = (swiz(row, chunk), swiz(row + 64, chunk));

                let mut stage = 0usize;
                while stage < 3 {
                    let s = smem + (stage * BIG_STAGE_BYTES) as u32;
                    if !a_ok { st_shared_zero16(s + dst0); }
                    if !b_ok0 { st_shared_zero16(s + BIG_A_BYTES as u32 + dst0); }
                    if !b_ok1 { st_shared_zero16(s + BIG_A_BYTES as u32 + dst1); }
                    stage += 1;
                }
                let issue_b = |s: u32, c_idx: u32| {
                    let b = s + BIG_A_BYTES as u32;
                    let at = (c_idx * 16) as usize;
                    if b_ok0 { crate::ptx::copy16(b + dst0, b_ptr0.add(at)); }
                    if b_ok1 { crate::ptx::copy16(b + dst1, b_ptr1.add(at)); }
                };
                let fetch_a = |c_idx: u32| -> ([f32; 4], [f32; 4]) {
                    let p = a_ptr.add((c_idx * 32) as usize);
                    (ld_global_nc_v4(p), ld_global_nc_v4(p.add(4)))
                };
                let store_a = |s: u32, lo: [f32; 4], hi: [f32; 4]| {
                    if a_ok {
                        st_shared_v4(s + dst0, [
                            cvt_bf16x2(lo[0], lo[1]), cvt_bf16x2(lo[2], lo[3]),
                            cvt_bf16x2(hi[0], hi[1]), cvt_bf16x2(hi[2], hi[3]),
                        ]);
                    }
                };

                let a_off = swiz(warp_m * 32 + (lane & 15), lane >> 4);
                let b_off = BIG_A_BYTES as u32 + swiz(warp_n * 32 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
                let (mut c00, mut c01, mut c02, mut c03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let (mut c10, mut c11, mut c12, mut c13) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let chunks = (k / TC_BK) as u32;
                let last_c = chunks - 1;

                issue_b(smem, 0);
                crate::ptx::commit();
                let (lo0, hi0) = fetch_a(0);
                if chunks > 1 { issue_b(smem + BIG_STAGE_BYTES as u32, 1); }
                crate::ptx::commit();
                let (lo1, hi1) = fetch_a(1u32.min(last_c));
                store_a(smem, lo0, hi0);
                if chunks > 1 { store_a(smem + BIG_STAGE_BYTES as u32, lo1, hi1); }

                macro_rules! big_k_sub {
                    ($a_at:expr, $b_at:expr) => {{
                        let (a_at, b_at) = ($a_at, $b_at);
                        let bp0 = ldmatrix_x4(b_at);
                        let bp1 = ldmatrix_x4(b_at + BIG_TILE16);
                        let a0 = ldmatrix_x4(a_at);
                        c00 = mma_m16n8k16_bf16(a0, [bp0[0], bp0[1]], c00);
                        c01 = mma_m16n8k16_bf16(a0, [bp0[2], bp0[3]], c01);
                        c02 = mma_m16n8k16_bf16(a0, [bp1[0], bp1[1]], c02);
                        c03 = mma_m16n8k16_bf16(a0, [bp1[2], bp1[3]], c03);
                        let a1 = ldmatrix_x4(a_at + BIG_TILE16);
                        c10 = mma_m16n8k16_bf16(a1, [bp0[0], bp0[1]], c10);
                        c11 = mma_m16n8k16_bf16(a1, [bp0[2], bp0[3]], c11);
                        c12 = mma_m16n8k16_bf16(a1, [bp1[0], bp1[1]], c12);
                        c13 = mma_m16n8k16_bf16(a1, [bp1[2], bp1[3]], c13);
                    }};
                }

                let last = smem + 2 * BIG_STAGE_BYTES as u32;
                let (mut cur, mut ahead) = (smem, last);
                let mut c_idx = 0u32;
                while c_idx < chunks {
                    cp_async_wait_one();
                    core::arch::nvptx::_syncthreads();
                    let has_ahead = c_idx + 2 < chunks;
                    if has_ahead { issue_b(ahead, c_idx + 2); }
                    crate::ptx::commit();
                    let (a_lo, a_hi) = fetch_a((c_idx + 2).min(last_c));
                    big_k_sub!(cur + a_off, cur + b_off);
                    big_k_sub!(cur + (a_off ^ 32), cur + (b_off ^ 32));
                    if has_ahead { store_a(ahead, a_lo, a_hi); }
                    ahead = cur;
                    cur = if cur == last { smem } else { cur + BIG_STAGE_BYTES as u32 };
                    c_idx += 1;
                }
                crate::ptx::wait();

                let tid_e = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp_e, lane_e) = (tid_e >> 5, tid_e & 31);
                let (warp_m, warp_n) = (warp_e >> 2, warp_e & 3);
                let block_e = core::arch::nvptx::_block_idx_x() as usize;
                let output = words.add(2).read_volatile() as *mut f32;
                let rows = words.add(3).read_volatile() as usize;
                let inputs = words.add(5).read_volatile() as usize;
                let stride = words.add(6).read_volatile() as usize;
                let split_count = words.add(15).read_volatile() as usize;
                let m_tiles = inputs.div_ceil(TC_BM);
                let n_tiles = rows.div_ceil(BIG_BN);
                let group_size = 8 * n_tiles;
                let first_m = (block_e / group_size) * 8;
                let height = (m_tiles - first_m).min(8);
                let within = block_e % group_size;
                let tile_m = first_m + within % height;
                let tile_n = within / height;
                if split_count > TC_SPLIT_MAX || (split_count == 0 && stride < rows) || tile_n * BIG_BN >= rows {
                    core::arch::nvptx::trap();
                }
                let evict = l2_evict_first();
                let (g, t) = (lane_e / 4, lane_e % 4);
                let m_base = tile_m * TC_BM + warp_m * 32 + g;
                let target = |ni: usize| -> (*mut f32, usize, bool) {
                    let col = tile_n * BIG_BN + warp_n * 32 + ni * 8 + t * 2;
                    if col >= rows { return (core::ptr::null_mut(), 0, false); }
                    if split_count == 0 { return (output.add(col), stride, col + 1 < rows); }
                    let mut seg = 0usize;
                    while seg < split_count {
                        let meta = *words.add(16 + 2 * seg + 1);
                        let seg_start = (meta & 0xffff_ffff) as usize;
                        let seg_len = (meta >> 32) as usize;
                        if col >= seg_start && col + 1 < seg_start + seg_len {
                            let seg_ptr = *words.add(16 + 2 * seg) as *mut f32;
                            return (seg_ptr.add(col - seg_start), seg_len, true);
                        }
                        seg += 1;
                    }
                    core::arch::nvptx::trap();
                };
                macro_rules! big_store_col {
                    ($ni:expr, $d0:ident, $d1:ident) => {{
                        let (at, step, pair) = target($ni);
                        if !at.is_null() {
                            let put = |m: usize, v0: f32, v1: f32| {
                                if m < inputs {
                                    let dst = at.add(m * step);
                                    if pair && ((dst as usize) & 7) == 0 {
                                        asm!("st.global.L2::cache_hint.v2.f32 [{p}], {{{a}, {b}}}, {q};",
                                            p = in(reg64) dst, a = in(reg32) v0, b = in(reg32) v1, q = in(reg64) evict, options(nostack));
                                    } else {
                                        st_global_f32_hint(dst, v0, evict);
                                        if pair { st_global_f32_hint(dst.add(1), v1, evict); }
                                    }
                                }
                            };
                            put(m_base, $d0[0], $d0[1]);
                            put(m_base + 8, $d0[2], $d0[3]);
                            put(m_base + 16, $d1[0], $d1[1]);
                            put(m_base + 24, $d1[2], $d1[3]);
                        }
                    }};
                }
                big_store_col!(0, c00, c10);
                big_store_col!(1, c01, c11);
                big_store_col!(2, c02, c12);
                big_store_col!(3, c03, c13);
                return;
            }

            // Path 1: Narrow 1-warp deep-pipelined direct-FP32 (`rows <= 128, inputs <= 256`: BM=8, BN=4, BK=64).
            // 8 stages of 4x144B padded W = 4,608 B, bar.warp.sync only, L1 X prefetch + L2 W prefetch, 1..=3 groups.
            if tc_narrow_1w_wanted(rows, inputs) {
                if split_count != 0 { core::arch::nvptx::trap(); }
                let output0 = *words.add(2) as *mut f32;
                let group = core::arch::nvptx::_block_idx_y() as usize;
                let (w_ptr, x1_ptr, x2_ptr, out_ptr) = match group {
                    0 => {
                        if grouped && (group_count == 0 || group_count > TC_GROUP_MAX) { core::arch::nvptx::trap(); }
                        (weights_base, input, input2_0, output0)
                    }
                    1 | 2 if grouped && group < group_count && group_count <= TC_GROUP_MAX => {
                        let base = TC_GROUP_WORD + (group - 1) * 3;
                        let gw = *words.add(base) as *const u32;
                        let gx1 = *words.add(base + 1) as *const f32;
                        let gx2 = *words.add(base + 2) as *const f32;
                        if (gw as usize | gx1 as usize | gx2 as usize) & 15 != 0 || gx2.is_null() {
                            core::arch::nvptx::trap();
                        }
                        (gw, gx1, gx2, output0.add(group * inputs * stride))
                    }
                    _ => core::arch::nvptx::trap(),
                };
                let m_tiles = inputs.div_ceil(TC_N1_BM);
                let n_tiles = rows.div_ceil(TC_N1_BN);
                if block >= m_tiles * n_tiles { return; }
                let m_base = (block % m_tiles) * TC_N1_BM;
                let n_base = (block / m_tiles) * TC_N1_BN;

                let lane = core::arch::nvptx::_thread_idx_x() as usize;
                let (g, s) = (lane >> 2, lane & 3);
                let (r, w_chunk) = (lane >> 3, lane & 7);
                let gr = n_base + r;
                let m_row = m_base + g;
                let in_valid = m_row < inputs;
                let half = if x2_ptr.is_null() { k } else { k / 2 };
                let kw = k / 2;
                let w_dst_off = (r * N1_ROW_BYTES + w_chunk * 16) as u32;
                let b_row_off = (g * N1_ROW_BYTES + 4 * s) as u32;
                let b_valid = g < TC_N1_BN;
                let chunks = k / TC_N1_BK;

                macro_rules! issue_w_n1 {
                    ($ch:expr, $stg:expr) => {{
                        let dst = $stg + w_dst_off;
                        if gr < rows {
                            crate::ptx::copy16_cached(dst, w_ptr.add(gr * kw + $ch * 32 + w_chunk * 4));
                        } else {
                            st_shared_zero16(dst);
                        }
                    }};
                }
                let half_w = half / 2;
                macro_rules! prefetch_x_n1 {
                    ($ch:expr) => {{
                        let ch: usize = $ch;
                        if half_in {
                            if lane < 8 && ch < chunks {
                                let pr = m_base + lane;
                                if pr < inputs {
                                    let kb = ch * TC_N1_BK;
                                    let (src, off) = if kb < half { (x1_ptr.cast::<u32>(), kb / 2) } else { (x2_ptr.cast::<u32>(), (kb - half) / 2) };
                                    asm!("prefetch.global.L1 [{a}];", a = in(reg64) src.add(pr * half_w + off) as u64);
                                }
                            }
                        } else if lane < 16 && ch < chunks {
                            let pr = m_base + (lane >> 1);
                            if pr < inputs {
                                let kb = ch * TC_N1_BK + (lane & 1) * 32;
                                let (src, off) = if kb < half { (x1_ptr, kb) } else { (x2_ptr, kb - half) };
                                asm!("prefetch.global.L1 [{a}];", a = in(reg64) src.add(pr * half + off) as u64);
                            }
                        }
                    }};
                }

                let mut st = 0usize;
                while st + 1 < N1_STAGES {
                    if st < chunks { issue_w_n1!(st, smem + (st * N1_STAGE_BYTES) as u32); }
                    crate::ptx::commit();
                    st += 1;
                }
                prefetch_x_n1!(0);
                prefetch_x_n1!(1);

                let mut acc = [0.0f32; 4];
                let m_safe = m_row.min(inputs - 1);
                let mut c_idx = 0usize;
                if half_in {
                    let x1_row_h = x1_ptr.cast::<u32>().add(m_safe * half_w + s);
                    let x2_row_h = if x2_ptr.is_null() { x1_row_h } else { x2_ptr.cast::<u32>().add(m_safe * half_w + s) };
                    while c_idx < chunks {
                        let ahead = c_idx + N1_STAGES - 1;
                        if ahead < chunks { issue_w_n1!(ahead, smem + ((ahead % N1_STAGES) * N1_STAGE_BYTES) as u32); }
                        crate::ptx::commit();
                        let pc = c_idx + N1_STAGES + 2;
                        if lane < 4 && pc < chunks && n_base + lane < rows {
                            asm!("prefetch.global.L2 [{a}];", a = in(reg64) w_ptr.add((n_base + lane) * kw + pc * 32) as u64);
                        }
                        prefetch_x_n1!(c_idx + 2);
                        let kb = c_idx * TC_N1_BK;
                        let stg = smem + ((c_idx % N1_STAGES) * N1_STAGE_BYTES) as u32 + b_row_off;
                        let h_off = kb / 2;
                        let src_lo = if kb < half { x1_row_h.add(h_off) } else { x2_row_h.add(h_off - half_w) };
                        let src_hi = if kb + 32 < half { x1_row_h.add(h_off + 16) } else { x2_row_h.add(h_off + 16 - half_w) };
                        let (a00, a02) = (ld_global_nc_u32(src_lo), ld_global_nc_u32(src_lo.add(4)));
                        let (a10, a12) = (ld_global_nc_u32(src_lo.add(8)), ld_global_nc_u32(src_lo.add(12)));
                        let (a20, a22) = (ld_global_nc_u32(src_hi), ld_global_nc_u32(src_hi.add(4)));
                        let (a30, a32) = (ld_global_nc_u32(src_hi.add(8)), ld_global_nc_u32(src_hi.add(12)));
                        asm!("cp.async.wait_group 6;");
                        crate::ptx::warp_sync();
                        macro_rules! kstep_n1h {
                            ($ks:literal, $a0:expr, $a2:expr) => {{
                                let (b0, b1) = if b_valid { (load_u32(stg + ($ks * 32) as u32), load_u32(stg + ($ks * 32 + 16) as u32)) } else { (0u32, 0u32) };
                                acc = mma_m16n8k16_bf16([$a0, 0u32, $a2, 0u32], [b0, b1], acc);
                            }};
                        }
                        kstep_n1h!(0, a00, a02); kstep_n1h!(1, a10, a12); kstep_n1h!(2, a20, a22); kstep_n1h!(3, a30, a32);
                        crate::ptx::warp_sync();
                        c_idx += 1;
                    }
                } else {
                    let x1_row = x1_ptr.add(m_safe * half + 2 * s).cast::<u32>();
                    let x2_row = if x2_ptr.is_null() { x1_row } else { x2_ptr.add(m_safe * half + 2 * s).cast::<u32>() };
                    while c_idx < chunks {
                        let ahead = c_idx + N1_STAGES - 1;
                        if ahead < chunks { issue_w_n1!(ahead, smem + ((ahead % N1_STAGES) * N1_STAGE_BYTES) as u32); }
                        crate::ptx::commit();
                        let pc = c_idx + N1_STAGES + 2;
                        if lane < 4 && pc < chunks && n_base + lane < rows {
                            asm!("prefetch.global.L2 [{a}];", a = in(reg64) w_ptr.add((n_base + lane) * kw + pc * 32) as u64);
                        }
                        prefetch_x_n1!(c_idx + 2);
                        let kb = c_idx * TC_N1_BK;
                        let stg = smem + ((c_idx % N1_STAGES) * N1_STAGE_BYTES) as u32 + b_row_off;
                        let src_lo = if kb < half { x1_row.add(kb) } else { x2_row.add(kb - half) };
                        let src_hi = if kb + 32 < half { x1_row.add(kb + 32) } else { x2_row.add(kb + 32 - half) };
                        let p0 = ld_global_nc_v2_u32(src_lo); let p1 = ld_global_nc_v2_u32(src_lo.add(8));
                        let p2 = ld_global_nc_v2_u32(src_lo.add(16)); let p3 = ld_global_nc_v2_u32(src_lo.add(24));
                        let p4 = ld_global_nc_v2_u32(src_hi); let p5 = ld_global_nc_v2_u32(src_hi.add(8));
                        let p6 = ld_global_nc_v2_u32(src_hi.add(16)); let p7 = ld_global_nc_v2_u32(src_hi.add(24));
                        asm!("cp.async.wait_group 6;");
                        crate::ptx::warp_sync();
                        macro_rules! kstep_n1 {
                            ($ks:literal, $pa:ident, $pb:ident) => {{
                                let (b0, b1) = if b_valid { (load_u32(stg + ($ks * 32) as u32), load_u32(stg + ($ks * 32 + 16) as u32)) } else { (0u32, 0u32) };
                                let a0 = cvt_bf16x2(f32::from_bits($pa[0]), f32::from_bits($pa[1]));
                                let a2 = cvt_bf16x2(f32::from_bits($pb[0]), f32::from_bits($pb[1]));
                                acc = mma_m16n8k16_bf16([a0, 0u32, a2, 0u32], [b0, b1], acc);
                            }};
                        }
                        kstep_n1!(0, p0, p1); kstep_n1!(1, p2, p3); kstep_n1!(2, p4, p5); kstep_n1!(3, p6, p7);
                        crate::ptx::warp_sync();
                        c_idx += 1;
                    }
                }
                crate::ptx::wait();

                if s < 2 && in_valid {
                    let col = n_base + 2 * s;
                    if col < rows {
                        let (a, b) = if round_out { (rn(acc[0]), rn(acc[1])) } else { (acc[0], acc[1]) };
                        store_pair!(out_ptr.add(m_row * stride + col), col + 1 < rows, a, b);
                    }
                }
                return;
            }

            // Path 2: Prefill Narrow direct-FP32 (`rows <= 128, inputs > 256`):
            // - `rows <= 64` (GateDown): 4 warps, BM=32, BN=64, 4 stages (32,768 B, 3 blk/SM), n_tiles=1.
            // - `64 < rows <= 128` (Router): 8 warps, BM=32, BN=128, 3 stages (36,864 B, 2 blk/SM), n_tiles=1.
            if tc_narrow_4w_wanted(rows, inputs) {
                if grouped || split_count != 0 { core::arch::nvptx::trap(); }
                let m_tiles = inputs.div_ceil(TC_N4_BM);
                if block >= m_tiles { return; }
                let tile_m = block;
                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid / 32, tid % 32);
                let (g, t) = (lane / 4, lane % 4);
                let half = if input2_0.is_null() { k } else { k / 2 };
                let half_steps = (half / TC_BK) as u32;
                let kw = (k / 2) as u32;
                let chunks = (k / TC_BK) as u32;
                let pair_s = |at: u32| -> u32 {
                    let v = crate::ptx::load2(at);
                    cvt_bf16x2(f32::from_bits(v[0]), f32::from_bits(v[1]))
                };
                macro_rules! prefill_narrow {
                    ($wn_cnt:literal, $stages:literal, $stg_bytes:expr, $b_step:literal, $a_two:literal, $wait_fn:ident) => {{
                        let (warp_m, warp_n) = (warp / $wn_cnt, warp % $wn_cnt);
                        let (a_row0, a_chunk) = (tid / 8, tid % 8);
                        let a_r0 = tile_m * TC_N4_BM + a_row0;
                        let a_ok0 = a_r0 < inputs;
                        let a_dst0 = swiz_f32(a_row0, a_chunk);
                        let a_src0 = (a_r0 * half + a_chunk * 4) as u32;
                        let a_r1 = a_r0 + 16;
                        let a_ok1 = $a_two && a_r1 < inputs;
                        let a_dst1 = swiz_f32(a_row0 + 16, a_chunk);
                        let a_src1 = (a_r1 * half + a_chunk * 4) as u32;
                        let (b_row0, b_chunk) = (tid / 4, tid % 4);
                        let (b_ok0, b_ok1) = (b_row0 < rows, b_row0 + $b_step < rows);
                        let (b_dst0, b_dst1) = (swiz(b_row0, b_chunk), swiz(b_row0 + $b_step, b_chunk));
                        let (b_src0, b_src1) = (
                            b_row0 as u32 * kw + b_chunk as u32 * 4,
                            (b_row0 + $b_step) as u32 * kw + b_chunk as u32 * 4,
                        );
                        let mut st = 0usize;
                        while st < $stages {
                            let s = smem + (st * $stg_bytes) as u32;
                            if !a_ok0 { st_shared_zero16(s + a_dst0); }
                            if $a_two && !a_ok1 { st_shared_zero16(s + a_dst1); }
                            let sb = s + N4_A_BYTES as u32;
                            if !b_ok0 { st_shared_zero16(sb + b_dst0); }
                            if !b_ok1 { st_shared_zero16(sb + b_dst1); }
                            st += 1;
                        }
                        let issue = |s: u32, c_idx: u32| {
                            let (src, step) = if c_idx < half_steps { (input, c_idx) } else { (input2_0, c_idx - half_steps) };
                            let a_off = step * 32;
                            if a_ok0 { crate::ptx::copy16_stream(s + a_dst0, src.add((a_src0 + a_off) as usize).cast::<u32>()); }
                            if a_ok1 { crate::ptx::copy16_stream(s + a_dst1, src.add((a_src1 + a_off) as usize).cast::<u32>()); }
                            let sb = s + N4_A_BYTES as u32;
                            let b_off = c_idx * 16;
                            if b_ok0 { crate::ptx::copy16_cached(sb + b_dst0, weights_base.add((b_src0 + b_off) as usize)); }
                            if b_ok1 { crate::ptx::copy16_cached(sb + b_dst1, weights_base.add((b_src1 + b_off) as usize)); }
                        };
                        let r_top = warp_m * 16 + g;
                        let a_off_top = swiz_f32(r_top, t >> 1) + ((t & 1) * 8) as u32;
                        let a_off_bot = swiz_f32(r_top + 8, t >> 1) + ((t & 1) * 8) as u32;
                        let b_off0 = N4_A_BYTES as u32 + swiz(warp_n * 32 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
                        let b_off1 = b_off0 + 16 * 64;
                        let (mut c0, mut c1, mut c2, mut c3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                        let ahead_dist = ($stages - 1) as u32;
                        let mut c_idx = 0u32;
                        while c_idx < ahead_dist {
                            if c_idx < chunks { issue(smem + c_idx * $stg_bytes as u32, c_idx); }
                            crate::ptx::commit();
                            c_idx += 1;
                        }
                        let last = smem + ahead_dist * $stg_bytes as u32;
                        let (mut cur, mut ahead) = (smem, last);
                        c_idx = 0;
                        while c_idx < chunks {
                            $wait_fn();
                            core::arch::nvptx::_syncthreads();
                            if c_idx + ahead_dist < chunks { issue(ahead, c_idx + ahead_dist); }
                            crate::ptx::commit();
                            let a0 = [pair_s(cur + a_off_top), pair_s(cur + a_off_bot), pair_s(cur + (a_off_top ^ 32)), pair_s(cur + (a_off_bot ^ 32))];
                            let bp0 = ldmatrix_x4(cur + b_off0);
                            let bp1 = ldmatrix_x4(cur + b_off1);
                            c0 = mma_m16n8k16_bf16(a0, [bp0[0], bp0[1]], c0);
                            c1 = mma_m16n8k16_bf16(a0, [bp0[2], bp0[3]], c1);
                            c2 = mma_m16n8k16_bf16(a0, [bp1[0], bp1[1]], c2);
                            c3 = mma_m16n8k16_bf16(a0, [bp1[2], bp1[3]], c3);
                            let a1 = [pair_s(cur + (a_off_top ^ 64)), pair_s(cur + (a_off_bot ^ 64)), pair_s(cur + (a_off_top ^ 96)), pair_s(cur + (a_off_bot ^ 96))];
                            let bp0 = ldmatrix_x4(cur + (b_off0 ^ 32));
                            let bp1 = ldmatrix_x4(cur + (b_off1 ^ 32));
                            c0 = mma_m16n8k16_bf16(a1, [bp0[0], bp0[1]], c0);
                            c1 = mma_m16n8k16_bf16(a1, [bp0[2], bp0[3]], c1);
                            c2 = mma_m16n8k16_bf16(a1, [bp1[0], bp1[1]], c2);
                            c3 = mma_m16n8k16_bf16(a1, [bp1[2], bp1[3]], c3);
                            ahead = cur;
                            cur = if cur == last { smem } else { cur + $stg_bytes as u32 };
                            c_idx += 1;
                        }
                        crate::ptx::wait();
                        let output = words.add(2).read_volatile() as *mut f32;
                        let stride = words.add(6).read_volatile() as usize;
                        let m0 = tile_m * TC_N4_BM + warp_m * 16 + g;
                        let m1 = m0 + 8;
                        let store_n = |ni: usize, v: [f32; 4]| {
                            let col = warp_n * 32 + ni * 8 + t * 2;
                            if col >= rows { return; }
                            let pair_ok = col + 1 < rows;
                            if m0 < inputs {
                                let (a, b) = if round_out { (rn(v[0]), rn(v[1])) } else { (v[0], v[1]) };
                                store_pair!(output.add(m0 * stride + col), pair_ok, a, b);
                            }
                            if m1 < inputs {
                                let (a, b) = if round_out { (rn(v[2]), rn(v[3])) } else { (v[2], v[3]) };
                                store_pair!(output.add(m1 * stride + col), pair_ok, a, b);
                            }
                        };
                        store_n(0, c0); store_n(1, c1); store_n(2, c2); store_n(3, c3);
                    }};
                }
                if rows <= TC_N4_BN {
                    prefill_narrow!(2, 4, N4_STAGE_BYTES, 32, true, cp_async_wait_two);
                } else {
                    prefill_narrow!(4, 3, N8_STAGE_BYTES, 64, false, cp_async_wait_one);
                }
                return;
            }

            // Path 3: Short-K (`cols == 64, rows > 128`, `GateOuter`): single-shot direct-FP32 (zero pack, 1 sync).
            // For `inputs > 32`, stages 64x64 FP32 output through 17,408 B SMEM (64x68 padded) for 256B-coalesced v4 writes.
            if tc_short_k_wanted(rows, k) {
                if grouped || round_out || !input2_0.is_null() { core::arch::nvptx::trap(); }
                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid / 32, tid % 32);
                let (g, t) = (lane / 4, lane % 4);
                let kw = (k / 2) as u32;
                if inputs <= TC_SMALL_BM {
                    let n_tiles = rows.div_ceil(32);
                    if block >= n_tiles { return; }
                    let tile_n = block;
                    let (r0, chunk) = (tid / 4, tid % 4);
                    let r1 = r0 + 16;
                    let (sw0, sw1) = (swiz(r0, chunk), swiz(r1, chunk));
                    let (bn0, bn1) = (tile_n * 32 + r0, tile_n * 32 + r1);
                    let mut st = 0usize;
                    while st < 2 {
                        let s = smem + (st * 4096) as u32;
                        let sb = s + 2048;
                        let k_off = st * 32;
                        if bn0 < rows { crate::ptx::copy16(sb + sw0, weights_base.add(bn0 * kw as usize + k_off / 2 + chunk * 4)); }
                        else { st_shared_zero16(sb + sw0); }
                        if bn1 < rows { crate::ptx::copy16(sb + sw1, weights_base.add(bn1 * kw as usize + k_off / 2 + chunk * 4)); }
                        else { st_shared_zero16(sb + sw1); }
                        macro_rules! load_a_f32 {
                            ($r:expr, $sw:expr) => {{
                                if $r < inputs {
                                    let p = input.add($r * 64 + k_off + chunk * 8);
                                    let lo = ld_global_nc_v4(p);
                                    let hi = ld_global_nc_v4(p.add(4));
                                    st_shared_v4(s + $sw, [
                                        cvt_bf16x2(lo[0], lo[1]), cvt_bf16x2(lo[2], lo[3]),
                                        cvt_bf16x2(hi[0], hi[1]), cvt_bf16x2(hi[2], hi[3]),
                                    ]);
                                } else {
                                    st_shared_zero16(s + $sw);
                                }
                            }};
                        }
                        load_a_f32!(r0, sw0);
                        load_a_f32!(r1, sw1);
                        st += 1;
                    }
                    crate::ptx::commit();
                    cp_async_wait_zero();
                    core::arch::nvptx::_syncthreads();

                    let a_ld_c0 = lane >> 4;
                    let (a_off0, a_off1) = (swiz(lane & 15, a_ld_c0), swiz(16 + (lane & 15), a_ld_c0));
                    let b_off0 = 2048 + swiz(warp * 16 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
                    let has_m1 = inputs > 16;
                    let (mut c00, mut c01, mut c10, mut c11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                    let mut st = 0usize;
                    while st < 2 {
                        let s = smem + (st * 4096) as u32;
                        let mut ks = 0u32;
                        while ks < 2 {
                            let xm = ks * 32;
                            let a0 = ldmatrix_x4(s + (a_off0 ^ xm));
                            let bp = ldmatrix_x4(s + (b_off0 ^ xm));
                            c00 = mma_m16n8k16_bf16(a0, [bp[0], bp[1]], c00);
                            c01 = mma_m16n8k16_bf16(a0, [bp[2], bp[3]], c01);
                            if has_m1 {
                                let a1 = ldmatrix_x4(s + (a_off1 ^ xm));
                                c10 = mma_m16n8k16_bf16(a1, [bp[0], bp[1]], c10);
                                c11 = mma_m16n8k16_bf16(a1, [bp[2], bp[3]], c11);
                            }
                            ks += 1;
                        }
                        st += 1;
                    }
                    let output = words.add(2).read_volatile() as *mut f32;
                    let stride = words.add(6).read_volatile() as usize;
                    let split_count = words.add(15).read_volatile() as usize;
                    let col0 = tile_n * 32 + warp * 16 + t * 2;
                    store_split_pair!(output, stride, split_count, inputs, g, g + 8, g, g + 8, col0, c00);
                    store_split_pair!(output, stride, split_count, inputs, g, g + 8, g, g + 8, col0 + 8, c01);
                    if has_m1 {
                        store_split_pair!(output, stride, split_count, inputs, 16 + g, 24 + g, 16 + g, 24 + g, col0, c10);
                        store_split_pair!(output, stride, split_count, inputs, 16 + g, 24 + g, 16 + g, 24 + g, col0 + 8, c11);
                    }
                } else {
                    if split_count != 0 { core::arch::nvptx::trap(); }
                    let m_tiles = inputs.div_ceil(TC_BM);
                    let n_tiles = rows.div_ceil(TC_BN);
                    if block >= m_tiles * n_tiles { return; }
                    let group_size = 8 * n_tiles;
                    let first_m = (block / group_size) * 8;
                    let height = (m_tiles - first_m).min(8);
                    let within = block % group_size;
                    let tile_m = first_m + within % height;
                    let tile_n = within / height;

                    let (row, chunk) = (tid / 4, tid % 4);
                    let sw = swiz(row, chunk);
                    let a_r = tile_m * TC_BM + row;
                    let b_r = tile_n * TC_BN + row;
                    let mut st = 0usize;
                    while st < 2 {
                        let s = smem + (st * GEN_STAGE_BYTES) as u32;
                        let sb = s + GEN_A_BYTES as u32;
                        let k_off = st * 32;
                        if b_r < rows { crate::ptx::copy16_cached(sb + sw, weights_base.add(b_r * kw as usize + k_off / 2 + chunk * 4)); }
                        else { st_shared_zero16(sb + sw); }
                        if a_r < inputs {
                            let p = input.add(a_r * 64 + k_off + chunk * 8);
                            let lo = ld_global_nc_v4(p);
                            let hi = ld_global_nc_v4(p.add(4));
                            st_shared_v4(s + sw, [
                                cvt_bf16x2(lo[0], lo[1]), cvt_bf16x2(lo[2], lo[3]),
                                cvt_bf16x2(hi[0], hi[1]), cvt_bf16x2(hi[2], hi[3]),
                            ]);
                        } else {
                            st_shared_zero16(s + sw);
                        }
                        st += 1;
                    }
                    crate::ptx::commit();
                    cp_async_wait_zero();
                    core::arch::nvptx::_syncthreads();

                    let (warp_m, warp_n) = (warp / 4, warp % 4);
                    let a_ld_c0 = lane >> 4;
                    let (a_off0, a_off1) = (swiz(warp_m * 32 + (lane & 15), a_ld_c0), swiz(warp_m * 32 + 16 + (lane & 15), a_ld_c0));
                    let b_off0 = GEN_A_BYTES as u32 + swiz(warp_n * 16 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
                    let (mut c00, mut c01, mut c10, mut c11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                    let mut st = 0usize;
                    while st < 2 {
                        let s = smem + (st * GEN_STAGE_BYTES) as u32;
                        let mut ks = 0u32;
                        while ks < 2 {
                            let xm = ks * 32;
                            let a0 = ldmatrix_x4(s + (a_off0 ^ xm));
                            let a1 = ldmatrix_x4(s + (a_off1 ^ xm));
                            let bp = ldmatrix_x4(s + (b_off0 ^ xm));
                            c00 = mma_m16n8k16_bf16(a0, [bp[0], bp[1]], c00);
                            c01 = mma_m16n8k16_bf16(a0, [bp[2], bp[3]], c01);
                            c10 = mma_m16n8k16_bf16(a1, [bp[0], bp[1]], c10);
                            c11 = mma_m16n8k16_bf16(a1, [bp[2], bp[3]], c11);
                            ks += 1;
                        }
                        st += 1;
                    }
                    core::arch::nvptx::_syncthreads();
                    let lm0 = warp_m * 32 + g;
                    let ln0 = warp_n * 16 + t * 2;
                    macro_rules! st_s2 {
                        ($mr:expr, $nc:expr, $a:expr, $b:expr) => {{
                            let p = smem + (($mr * 68 + $nc) * 4) as u32;
                            asm!("st.shared.v2.f32 [{p}], {{{a}, {b}}};", p = in(reg32) p, a = in(reg32) $a, b = in(reg32) $b);
                        }};
                    }
                    st_s2!(lm0, ln0, c00[0], c00[1]);
                    st_s2!(lm0 + 8, ln0, c00[2], c00[3]);
                    st_s2!(lm0, ln0 + 8, c01[0], c01[1]);
                    st_s2!(lm0 + 8, ln0 + 8, c01[2], c01[3]);
                    st_s2!(lm0 + 16, ln0, c10[0], c10[1]);
                    st_s2!(lm0 + 24, ln0, c10[2], c10[3]);
                    st_s2!(lm0 + 16, ln0 + 8, c11[0], c11[1]);
                    st_s2!(lm0 + 24, ln0 + 8, c11[2], c11[3]);
                    core::arch::nvptx::_syncthreads();

                    let output = words.add(2).read_volatile() as *mut f32;
                    let stride = words.add(6).read_volatile() as usize;
                    let evict = l2_evict_first();
                    let out_r0 = tid / 16;
                    let out_c = (tid % 16) * 4;
                    let gn = tile_n * TC_BN + out_c;
                    if gn < rows {
                        let full4 = gn + 3 < rows;
                        let mut step = 0usize;
                        while step < 4 {
                            let rl = out_r0 + step * 16;
                            let gm = tile_m * TC_BM + rl;
                            if gm < inputs {
                                let sp = smem + ((rl * 68 + out_c) * 4) as u32;
                                let v = crate::ptx::load4(sp);
                                let dst = output.add(gm * stride + gn);
                                if full4 && ((dst as usize) & 15) == 0 {
                                    asm!("st.global.L2::cache_hint.v4.u32 [{p}], {{{a},{b},{c},{d}}}, {q};",
                                        p = in(reg64) dst, a = in(reg32) v[0], b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3],
                                        q = in(reg64) evict, options(nostack));
                                } else {
                                    *dst = f32::from_bits(v[0]);
                                    if gn + 1 < rows { *dst.add(1) = f32::from_bits(v[1]); }
                                    if gn + 2 < rows { *dst.add(2) = f32::from_bits(v[2]); }
                                    if gn + 3 < rows { *dst.add(3) = f32::from_bits(v[3]); }
                                }
                            }
                            step += 1;
                        }
                    }
                }
                return;
            }
            core::arch::nvptx::trap();
        }

        // ============================================================================================================
        // MODE 0 (packed BF16 dense / split, optional lossless bf16_panels) & MODE 1 (device-routed MoE)
        // ============================================================================================================
        let (weights, base, m, tile_m_off, tile_n, input2) = if mode == TC_MODE_DENSE {
            let split_count = *words.add(15) as usize;
            let input2 = *words.add(12) as *const f32;
            let (use_panels, panel_slices) = {
                let pb = *words.add(TC_PANEL_WORD);
                let pm = *words.add(TC_PANEL_WORD + 1);
                if pb == 0 {
                    if pm != 0 { core::arch::nvptx::trap(); }
                    (false, 0usize)
                } else {
                    let sl = (pm >> 32) as usize;
                    if (pb & 15) != 0 || !matches!(sl, 1 | 4 | 8) { core::arch::nvptx::trap(); }
                    (true, sl)
                }
            };
            let want_smem = if use_panels { tc_dense_panel_smem(rows, k, inputs, panel_slices) } else { tc_dense_spec(rows, k, inputs, false).2 };
            if split_count > TC_SPLIT_MAX || (split_count == 0 && stride < rows) || (shared_bytes as usize) < want_smem {
                core::arch::nvptx::trap();
            }
            if !input2.is_null() && (split_count != 0 || (input2 as usize & 15) != 0 || (k / 2) % TC_BK != 0) {
                core::arch::nvptx::trap();
            }
            if !tc_dense_admitted(rows, k, inputs, weights_base as usize, input as usize, threads) { core::arch::nvptx::trap(); }

            macro_rules! mma_x4 {
                ($a:expr, $xb:expr, $xo:expr, $xm:literal, $c0:ident, $h1:expr, $c1:ident, $h2:expr, $c2:ident, $h3:expr, $c3:ident) => {{
                    let a: [u32; 4] = $a;
                    let x0: u32 = ($xb + $xo) ^ $xm;
                    let x1: u32 = x0 ^ 16;
                    $c0 = mma_m16n8k16_bf16(a, [load_u32(x0), load_u32(x1)], $c0);
                    if $h1 { $c1 = mma_m16n8k16_bf16(a, [load_u32(x0 + 1024), load_u32(x1 + 1024)], $c1); }
                    if $h2 { $c2 = mma_m16n8k16_bf16(a, [load_u32(x0 + 2048), load_u32(x1 + 2048)], $c2); }
                    if $h3 { $c3 = mma_m16n8k16_bf16(a, [load_u32(x0 + 3072), load_u32(x1 + 3072)], $c3); }
                }};
            }
            macro_rules! emit_w_as_a {
                ($coop:literal, $c0:expr, $h1:expr, $c1:expr, $h2:expr, $c2:expr, $h3:expr, $c3:expr) => {{
                    let output = words.add(2).read_volatile() as *mut f32;
                    let rows = words.add(3).read_volatile() as usize;
                    let inputs = words.add(5).read_volatile() as usize;
                    let stride = words.add(6).read_volatile() as usize;
                    let split_count = words.add(15).read_volatile() as usize;
                    let tid_e = core::arch::nvptx::_thread_idx_x() as usize;
                    let (warp_e, lane_e) = (tid_e >> 5, tid_e & 31);
                    let (g_e, s_e) = (lane_e >> 2, lane_e & 3);
                    let blk_e = core::arch::nvptx::_block_idx_x() as usize;
                    let first_row_e = if $coop {
                        blk_e * 16
                    } else {
                        (blk_e * (core::arch::nvptx::_block_dim_x() as usize >> 5) + warp_e) * 16
                    };
                    let m_w_e = if $coop { warp_e * if inputs <= TC_SMALL_BM { 8 } else { 32 } } else { 0 };
                    let evict = l2_evict_first();
                    let resolve = |r: usize| -> (*mut f32, usize) {
                        if r >= rows { return (core::ptr::null_mut(), 0); }
                        if split_count == 0 { return (output.add(r), stride); }
                        let mut seg = 0usize;
                        while seg < split_count {
                            let meta = *words.add(16 + 2 * seg + 1);
                            let seg_start = (meta & 0xffff_ffff) as usize;
                            let seg_len = (meta >> 32) as usize;
                            if r >= seg_start && r < seg_start + seg_len {
                                let seg_ptr = *words.add(16 + 2 * seg) as *mut f32;
                                return (seg_ptr.add(r - seg_start), seg_len);
                            }
                            seg += 1;
                        }
                        core::arch::nvptx::trap();
                    };
                    let (p0, s0) = resolve(first_row_e + g_e);
                    let (p1, s1) = resolve(first_row_e + g_e + 8);
                    let t0 = m_w_e + 2 * s_e;
                    macro_rules! emit_tile {
                        ($t_add:literal, $c:expr) => {{
                            let ta = t0 + $t_add;
                            let tb = ta + 1;
                            if !p0.is_null() {
                                if ta < inputs { st_global_f32_hint(p0.add(ta * s0), $c[0], evict); }
                                if tb < inputs { st_global_f32_hint(p0.add(tb * s0), $c[1], evict); }
                            }
                            if !p1.is_null() {
                                if ta < inputs { st_global_f32_hint(p1.add(ta * s1), $c[2], evict); }
                                if tb < inputs { st_global_f32_hint(p1.add(tb * s1), $c[3], evict); }
                            }
                        }};
                    }
                    emit_tile!(0, $c0);
                    if $h1 { emit_tile!(8, $c1); }
                    if $h2 { emit_tile!(16, $c2); }
                    if $h3 { emit_tile!(24, $c3); }
                }};
            }

            // Path 4b: 4-warp Cooperative-W `BN=16, BK=64` `evict_first` (`m_tiles=1`, W read ONCE, 2-chunk-ahead W!).
            // - With `bf16_panels`: warp `w` loads tile `c*4+w` (-25% DRAM), decodes once into linear lane-order SMEM;
            //   all 4 warps read `ptx::load4` (0 bank conflicts, 0 `ldmatrix`).
            if tc_coop_w_wanted(rows, k, inputs) {
                if (*words.add(8) | *words.add(10)) != 0 { core::arch::nvptx::trap(); }
                let n_tiles = rows.div_ceil(16);
                if block >= n_tiles { return; }
                let first_row = block * 16;
                let x_2stage = inputs <= TC_SMALL_BM;
                let bm = if x_2stage { 32usize } else { 128usize };
                let bm_w = if x_2stage { 8usize } else { 32usize };

                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid / 32, tid % 32);
                let (g, s) = (lane >> 2, lane & 3);
                let (w_r, w_c) = (tid >> 3, tid & 7);
                let m_w = warp * bm_w;
                let kw = k / 2;
                let chunks = k / 64;
                let last_c = chunks - 1;

                let x_base = if use_panels && x_2stage { smem + 8192 } else { smem + 4096 };
                let x_stride = (bm * 128) as u32;
                let x_off0 = swiz128(g, 0) + (s * 4) as u32;
                let evict = l2_evict_first();

                {
                    let z_slabs = if use_panels && x_2stage { 4u32 } else if x_2stage { 2u32 } else { 1u32 };
                    let mut rl = w_r;
                    while rl < bm {
                        if rl >= inputs {
                            let dst = swiz128(rl, w_c);
                            let mut zi = 0u32;
                            while zi < z_slabs {
                                st_shared_zero16(x_base + zi * x_stride + dst);
                                zi += 1;
                            }
                        }
                        rl += 16;
                    }
                }
                let issue_x = |x_stg: u32, ch: usize| {
                    let in2 = words.add(12).read_volatile() as *const u32;
                    let half = if in2.is_null() { k } else { k >> 1 };
                    let k_col = ch * 64 + w_c * 8;
                    let (src_x, x_col) = if k_col < half { (input.cast::<u32>(), k_col) } else { (in2, k_col - half) };
                    let x_word_off = x_col >> 1;
                    let mut rl = w_r;
                    while rl < inputs {
                        crate::ptx::copy16_cached(x_stg + swiz128(rl, w_c), src_x.add(rl * (half >> 1) + x_word_off));
                        rl += 16;
                    }
                };

                let (mut c0, mut c1, mut c2, mut c3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let has_m1 = !x_2stage && m_w + 8 < inputs;
                let has_m2 = !x_2stage && m_w + 16 < inputs;
                let has_m3 = !x_2stage && m_w + 24 < inputs;

                if use_panels && x_2stage {
                    tc_coop_w_panels!(
                        words, k, block, warp, lane,
                        m_w, smem, x_base, x_stride, x_off0, evict,
                        issue_x, c0
                    );
                } else {
                    let w_dst = swiz128(w_r, w_c);
                    let off0 = swiz128(lane & 15, lane >> 4);
                    let w_row_ptr = weights_base.add((first_row + w_r).min(rows - 1) * kw + w_c * 4);
                    let load_w = |ch: usize| -> [u32; 4] { ld_global_nc_v4_u32_hint(w_row_ptr.add(ch * 32), evict) };
                    let prefetch_w = |ch: usize| {
                        if tid < 16 && ch < chunks && first_row + tid < rows {
                            asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights_base.add((first_row + tid) * kw + ch * 32) as u64);
                        }
                    };
                    issue_x(x_base, 0);
                    crate::ptx::commit();
                    let w0 = load_w(0);
                    let mut w_nxt = load_w(1usize.min(last_c));
                    prefetch_w(2); prefetch_w(3); prefetch_w(4);
                    st_shared_v4(smem + w_dst, w0);

                    let mut c_idx = 0usize;
                    while c_idx < chunks {
                        cp_async_wait_zero();
                        core::arch::nvptx::_syncthreads();
                        let cur_w = smem + ((c_idx & 1) as u32) * 2048;
                        let nxt_w = smem + (((c_idx + 1) & 1) as u32) * 2048;
                        let cur_x = if x_2stage { x_base + ((c_idx & 1) as u32) * x_stride } else { x_base };
                        if x_2stage && c_idx + 1 < chunks { issue_x(x_base + (((c_idx + 1) & 1) as u32) * x_stride, c_idx + 1); crate::ptx::commit(); }
                        let w_after = load_w((c_idx + 2).min(last_c));
                        prefetch_w(c_idx + 5);
                        let x_w = cur_x + (m_w * 128) as u32;
                        mma_x4!(ldmatrix_x4(cur_w + off0), x_w, x_off0, 0u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        mma_x4!(ldmatrix_x4(cur_w + (off0 ^ 32)), x_w, x_off0, 32u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        mma_x4!(ldmatrix_x4(cur_w + (off0 ^ 64)), x_w, x_off0, 64u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        mma_x4!(ldmatrix_x4(cur_w + (off0 ^ 96)), x_w, x_off0, 96u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        st_shared_v4(nxt_w + w_dst, w_nxt);
                        w_nxt = w_after;
                        if !x_2stage && c_idx + 1 < chunks { core::arch::nvptx::_syncthreads(); issue_x(x_base, c_idx + 1); crate::ptx::commit(); }
                        c_idx += 1;
                    }
                }
                emit_w_as_a!(true, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                return;
            }

            // Path 4: Small-M Wide `BK=64` Weights-as-A (1w/2w/4w, m_tiles=1):
            // - With `bf16_panels`: 0 B W SMEM, 4-tile sliding register ring decoded directly into MMA A fragments!
            // - Without panels: unconditional clamped `LDG.128.EF` -> 1-stage private W SMEM + `ldmatrix_x4`.
            if tc_small_wanted(inputs) {
                if (*words.add(8) | *words.add(10)) != 0 { core::arch::nvptx::trap(); }
                let warps = threads / 32;
                let bm = if inputs <= 16 { 16usize } else { 32usize };
                let bn = warps * 16;
                let n_tiles = rows.div_ceil(bn);
                if block >= n_tiles { return; }
                let tile_n = block;

                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid / 32, tid % 32);
                let (g, s) = (lane >> 2, lane & 3);
                let kw = k / 2;
                let chunks = (k / 64) as u32;

                let x_base = if use_panels { smem } else { smem + (warps * 2048) as u32 };
                let x_stride = (bm * 128) as u32;
                let x_off0 = swiz128(g, 0) + (s * 4) as u32;
                let (x_row0, x_chunk) = (tid >> 3, tid & 7);
                let x_row_step = warps * 4;
                let evict = l2_evict_first();

                let init_slabs = if use_panels { chunks.min(if bm == 16 { 4 } else { 3 }) } else { 2 };
                let mut rl = x_row0;
                while rl < bm {
                    if rl >= inputs {
                        let dst = swiz128(rl, x_chunk);
                        let mut si = 0u32;
                        while si < init_slabs {
                            st_shared_zero16(x_base + si * x_stride + dst);
                            si += 1;
                        }
                    }
                    rl += x_row_step;
                }
                let issue_x = |x_stg: u32, ch: u32| {
                    let in2 = words.add(12).read_volatile() as *const u32;
                    let half = if in2.is_null() { k } else { k >> 1 };
                    let tid_x = core::arch::nvptx::_thread_idx_x() as usize;
                    let (xr0, xc) = (tid_x >> 3, tid_x & 7);
                    let x_step = (core::arch::nvptx::_block_dim_x() as usize) >> 3;
                    let k_col = (ch as usize) * 64 + xc * 8;
                    let (src_x, x_col) = if k_col < half { (input.cast::<u32>(), k_col) } else { (in2, k_col - half) };
                    let mut rl = xr0;
                    while rl < inputs {
                        crate::ptx::copy16_cached(x_stg + swiz128(rl, xc), src_x.add(rl * (half >> 1) + (x_col >> 1)));
                        rl += x_step;
                    }
                };

                let (mut c0, mut c1, mut c2, mut c3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let (has_m1, has_m2, has_m3) = (inputs > 8, bm > 16 && inputs > 16, bm > 16 && inputs > 24);

                if use_panels {
                    tc_path4_panels!(
                        words, rows, k,
                        warps, tile_n, warp, lane,
                        init_slabs, x_stride, x_off0, evict,
                        c0, has_m1, c1, has_m2, c2, has_m3, c3
                    );
                } else {
                    let first_row = tile_n * bn + warp * 16;
                    let (r_base, w_chunk) = (lane >> 3, lane & 7);
                    let w_warp = smem + (warp * 2048) as u32;
                    let w_dst0 = swiz128(r_base, w_chunk);
                    let off0 = swiz128(lane & 15, lane >> 4);
                    let wp0 = weights_base.add((first_row + r_base).min(rows - 1) * kw + w_chunk * 4);
                    let wp1 = weights_base.add((first_row + r_base + 4).min(rows - 1) * kw + w_chunk * 4);
                    let wp2 = weights_base.add((first_row + r_base + 8).min(rows - 1) * kw + w_chunk * 4);
                    let wp3 = weights_base.add((first_row + r_base + 12).min(rows - 1) * kw + w_chunk * 4);
                    let last_c = chunks - 1;
                    let load_w = |ch: u32| -> ([u32; 4], [u32; 4], [u32; 4], [u32; 4]) {
                        let off = (ch as usize) * 32;
                        (ld_global_nc_v4_u32_hint(wp0.add(off), evict), ld_global_nc_v4_u32_hint(wp1.add(off), evict),
                         ld_global_nc_v4_u32_hint(wp2.add(off), evict), ld_global_nc_v4_u32_hint(wp3.add(off), evict))
                    };
                    let prefetch_w = |ch: u32| {
                        if lane < 16 && ch < chunks && first_row + lane < rows {
                            asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights_base.add((first_row + lane) * kw + (ch as usize) * 32) as u64);
                        }
                    };
                    issue_x(x_base, 0);
                    crate::ptx::commit();
                    let (mut w0, mut w1, mut w2, mut w3) = load_w(0);
                    prefetch_w(1); prefetch_w(2); prefetch_w(3); prefetch_w(4);

                    let mut c_idx = 0u32;
                    while c_idx < chunks {
                        st_shared_v4(w_warp + w_dst0, w0);
                        st_shared_v4(w_warp + ((w_dst0 + 512) ^ 64), w1);
                        st_shared_v4(w_warp + w_dst0 + 1024, w2);
                        st_shared_v4(w_warp + ((w_dst0 + 1536) ^ 64), w3);
                        cp_async_wait_zero();
                        if warps == 1 { crate::ptx::warp_sync(); } else { core::arch::nvptx::_syncthreads(); }
                        let cur_x = x_base + (c_idx & 1) * x_stride;
                        if c_idx + 1 < chunks { issue_x(x_base + ((c_idx + 1) & 1) * x_stride, c_idx + 1); crate::ptx::commit(); }
                        let nw = load_w((c_idx + 1).min(last_c));
                        w0 = nw.0; w1 = nw.1; w2 = nw.2; w3 = nw.3;
                        prefetch_w(c_idx + 5);
                        mma_x4!(ldmatrix_x4(w_warp + off0), cur_x, x_off0, 0u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        mma_x4!(ldmatrix_x4(w_warp + (off0 ^ 32)), cur_x, x_off0, 32u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        mma_x4!(ldmatrix_x4(w_warp + (off0 ^ 64)), cur_x, x_off0, 64u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        mma_x4!(ldmatrix_x4(w_warp + (off0 ^ 96)), cur_x, x_off0, 96u32, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                        crate::ptx::warp_sync();
                        c_idx += 1;
                    }
                }
                emit_w_as_a!(false, c0, has_m1, c1, has_m2, c2, has_m3, c3);
                return;
            }

            // Path 5a: Medium-M 2-warp `32x32x32` x 4 stages (`129..=256, 128 < rows <= 2560, cols > 64`).
            if tc_med_w2_wanted(rows, k, inputs) {
                if (*words.add(8) | *words.add(10)) != 0 { core::arch::nvptx::trap(); }
                let m_tiles = inputs.div_ceil(32);
                let n_tiles = rows.div_ceil(32);
                if block >= m_tiles * n_tiles { return; }
                let group_size = 8 * n_tiles;
                let first_m = (block / group_size) * 8;
                let height = (m_tiles - first_m).min(8);
                let within = block % group_size;
                let (tile_m, tile_n) = (first_m + within % height, within / height);
                let m_off = tile_m * 32;
                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid / 32, tid % 32);
                let (r0, chunk) = (tid / 4, tid % 4);
                let r1 = r0 + 16;
                let (sw0, sw1) = (swiz(r0, chunk), swiz(r1, chunk));
                let (a_r0, a_r1) = (m_off + r0, m_off + r1);
                let (a_ok0, a_ok1) = (a_r0 < inputs, a_r1 < inputs);
                let has_m1 = m_off + 16 < inputs;
                let (b_r0, b_r1) = (tile_n * 32 + r0, tile_n * 32 + r1);
                let (b_ok0, b_ok1) = (b_r0 < rows, b_r1 < rows);
                const S_BYTES: usize = 4096;
                let mut st = 0usize;
                while st < 4 {
                    let s = smem + (st * S_BYTES) as u32;
                    if !a_ok0 { st_shared_zero16(s + sw0); }
                    if has_m1 && !a_ok1 { st_shared_zero16(s + sw1); }
                    if !b_ok0 { st_shared_zero16(s + 2048 + sw0); }
                    if !b_ok1 { st_shared_zero16(s + 2048 + sw1); }
                    st += 1;
                }
                let kw = (k / 2) as u32;
                let (b_src0, b_src1) = (b_r0 as u32 * kw + chunk as u32 * 4, b_r1 as u32 * kw + chunk as u32 * 4);
                let half = if input2.is_null() { k } else { k / 2 };
                let (half_w, half_steps) = ((half / 2) as u32, (half / TC_BK) as u32);
                let (a_src0, a_src1) = (a_r0 as u32 * half_w + chunk as u32 * 4, a_r1 as u32 * half_w + chunk as u32 * 4);
                let issue = |s: u32, c_idx: u32| {
                    let (src_a, step_a) = if c_idx < half_steps { (input.cast::<u32>(), c_idx) } else { (input2.cast::<u32>(), c_idx - half_steps) };
                    let at_a = step_a * 16;
                    if a_ok0 { crate::ptx::copy16_cached(s + sw0, src_a.add((a_src0 + at_a) as usize)); }
                    if a_ok1 { crate::ptx::copy16_cached(s + sw1, src_a.add((a_src1 + at_a) as usize)); }
                    let at_b = c_idx * 16;
                    if b_ok0 { crate::ptx::copy16(s + 2048 + sw0, weights_base.add((b_src0 + at_b) as usize)); }
                    if b_ok1 { crate::ptx::copy16(s + 2048 + sw1, weights_base.add((b_src1 + at_b) as usize)); }
                };
                let a_ld_c0 = lane >> 4;
                let (a_off0, a_off1) = (swiz(lane & 15, a_ld_c0), swiz(16 + (lane & 15), a_ld_c0));
                let b_off = 2048 + swiz(warp * 16 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
                let (mut c00, mut c01, mut c10, mut c11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let chunks = (k / TC_BK) as u32;
                let mut c_idx = 0u32;
                while c_idx < 3 {
                    if c_idx < chunks { issue(smem + c_idx * S_BYTES as u32, c_idx); }
                    crate::ptx::commit();
                    c_idx += 1;
                }
                let last = smem + 3 * S_BYTES as u32;
                let (mut cur, mut ahead) = (smem, last);
                c_idx = 0;
                while c_idx < chunks {
                    cp_async_wait_two();
                    core::arch::nvptx::_syncthreads();
                    if c_idx + 3 < chunks { issue(ahead, c_idx + 3); }
                    crate::ptx::commit();
                    let mut ks = 0u32;
                    while ks < 2 {
                        let xm = ks * 32;
                        let a0 = ldmatrix_x4(cur + (a_off0 ^ xm));
                        let bp = ldmatrix_x4(cur + (b_off ^ xm));
                        c00 = mma_m16n8k16_bf16(a0, [bp[0], bp[1]], c00);
                        c01 = mma_m16n8k16_bf16(a0, [bp[2], bp[3]], c01);
                        if has_m1 {
                            let a1 = ldmatrix_x4(cur + (a_off1 ^ xm));
                            c10 = mma_m16n8k16_bf16(a1, [bp[0], bp[1]], c10);
                            c11 = mma_m16n8k16_bf16(a1, [bp[2], bp[3]], c11);
                        }
                        ks += 1;
                    }
                    ahead = cur;
                    cur = if cur == last { smem } else { cur + S_BYTES as u32 };
                    c_idx += 1;
                }
                crate::ptx::wait();

                let output = words.add(2).read_volatile() as *mut f32;
                let stride = words.add(6).read_volatile() as usize;
                let split_count = words.add(15).read_volatile() as usize;
                let (g, t) = (lane / 4, lane % 4);
                let col0 = tile_n * 32 + warp * 16 + t * 2;
                store_split_pair!(output, stride, split_count, inputs, m_off + g, m_off + g + 8, m_off + g, m_off + g + 8, col0, c00);
                store_split_pair!(output, stride, split_count, inputs, m_off + g, m_off + g + 8, m_off + g, m_off + g + 8, col0 + 8, c01);
                if has_m1 {
                    store_split_pair!(output, stride, split_count, inputs, m_off + 16 + g, m_off + 24 + g, m_off + 16 + g, m_off + 24 + g, col0, c10);
                    store_split_pair!(output, stride, split_count, inputs, m_off + 16 + g, m_off + 24 + g, m_off + 16 + g, m_off + 24 + g, col0 + 8, c11);
                }
                return;
            }

            let m_tiles = inputs.div_ceil(TC_BM);
            let n_tiles = rows.div_ceil(TC_BN);
            if block >= m_tiles * n_tiles { return; }
            let group_size = 8 * n_tiles;
            let first_m = (block / group_size) * 8;
            let height = (m_tiles - first_m).min(8);
            let within = block % group_size;
            let tm = first_m + within % height;
            let tn = within / height;
            (weights_base, 0usize, inputs, tm * TC_BM, tn, input2)
        } else if mode == TC_MODE_MOE {
            let tile_meta = *words.add(12) as *const u32;
            let max_m_tiles = *words.add(13) as usize;
            let experts = *words.add(14) as usize;
            if (tile_meta as usize & 3) != 0 || stride < rows || (shared_bytes as usize) < TC_GEN_SMEM { core::arch::nvptx::trap(); }
            if !tc_moe_admitted(rows, k, inputs, experts, max_m_tiles, weights_base as usize, input as usize, threads) {
                core::arch::nvptx::trap();
            }
            let n_tiles = rows.div_ceil(TC_BN);
            let tn = block % n_tiles;
            let tm = block / n_tiles;
            let total_m_tiles = tile_meta.read() as usize;
            if total_m_tiles > max_m_tiles { core::arch::nvptx::trap(); }
            if tm >= total_m_tiles || tn * TC_BN >= rows { return; }
            let entry = tile_meta.add(1 + 3 * tm);
            let expert = entry.read() as usize;
            let first_pos = entry.add(1).read() as usize;
            let m_cnt = entry.add(2).read() as usize;
            if expert >= experts || m_cnt == 0 || m_cnt > TC_BM || first_pos + m_cnt > inputs { core::arch::nvptx::trap(); }
            let expert_words = rows * (k / 2);
            (weights_base.add(expert * expert_words), first_pos, m_cnt, 0usize, tn, core::ptr::null())
        } else {
            core::arch::nvptx::trap();
        };

        // Path 5b: General 8-warp 64x64x32 x 4 stages (medium dense & MoE).
        let in_table = *words.add(8) as *const u32;
        let in_count = *words.add(9) as usize;
        let place = |tbl: *const u32, count: usize, t: usize| -> usize {
            let row = if tbl.is_null() { t } else { tbl.add(t).read() as usize };
            if row >= count { core::arch::nvptx::trap(); }
            row
        };
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (warp_m, warp_n) = (warp / 4, warp % 4);
        let (row, chunk) = (tid / 4, tid % 4);
        let a_valid = tile_m_off + row < m;
        let a_src_row = if a_valid { place(in_table, in_count, base + tile_m_off + row) } else { 0 };
        let b_r0 = tile_n * TC_BN + row;
        let b_valid = b_r0 < rows;
        let sw = swiz(row, chunk);

        let mut st = 0usize;
        while st < TC_GEN_STAGES {
            let s = smem + (st * GEN_STAGE_BYTES) as u32;
            if !a_valid { st_shared_zero16(s + sw); }
            if !b_valid { st_shared_zero16(s + GEN_A_BYTES as u32 + sw); }
            st += 1;
        }
        let kw = (k / 2) as u32;
        let b_start = b_r0 as u32 * kw + chunk as u32 * 4;
        let half = if input2.is_null() { k } else { k / 2 };
        let (half_w, half_steps) = ((half / 2) as u32, (half / TC_BK) as u32);
        let a_start = a_src_row as u32 * half_w + chunk as u32 * 4;
        let issue_gen = |s: u32, c_idx: u32| {
            if b_valid { crate::ptx::copy16(s + GEN_A_BYTES as u32 + sw, weights.add((b_start + c_idx * 16) as usize)); }
            if a_valid {
                let (src_a, step_a) = if c_idx < half_steps { (input.cast::<u32>(), c_idx) } else { (input2.cast::<u32>(), c_idx - half_steps) };
                crate::ptx::copy16_cached(s + sw, src_a.add((a_start + step_a * 16) as usize));
            }
        };

        let a_ld_c0 = lane >> 4;
        let (a_off_00, a_off_01) = (swiz(warp_m * 32 + (lane & 15), a_ld_c0), swiz(warp_m * 32 + 16 + (lane & 15), a_ld_c0));
        let b_off_00 = GEN_A_BYTES as u32 + swiz(warp_n * 16 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
        let (mut c00, mut c01, mut c10, mut c11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let chunks = (k / TC_BK) as u32;
        let mut c_idx = 0u32;
        while c_idx < 3 {
            if c_idx < chunks { issue_gen(smem + c_idx * GEN_STAGE_BYTES as u32, c_idx); }
            crate::ptx::commit();
            c_idx += 1;
        }
        let last = smem + 3 * GEN_STAGE_BYTES as u32;
        let (mut cur, mut ahead) = (smem, last);
        c_idx = 0;
        while c_idx < chunks {
            cp_async_wait_two();
            core::arch::nvptx::_syncthreads();
            if c_idx + 3 < chunks { issue_gen(ahead, c_idx + 3); }
            crate::ptx::commit();
            let mut ks = 0u32;
            while ks < 2 {
                let xm = ks * 32;
                let a0 = ldmatrix_x4(cur + (a_off_00 ^ xm));
                let a1 = ldmatrix_x4(cur + (a_off_01 ^ xm));
                let bp0 = ldmatrix_x4(cur + (b_off_00 ^ xm));
                c00 = mma_m16n8k16_bf16(a0, [bp0[0], bp0[1]], c00);
                c01 = mma_m16n8k16_bf16(a0, [bp0[2], bp0[3]], c01);
                c10 = mma_m16n8k16_bf16(a1, [bp0[0], bp0[1]], c10);
                c11 = mma_m16n8k16_bf16(a1, [bp0[2], bp0[3]], c11);
                ks += 1;
            }
            ahead = cur;
            cur = if cur == last { smem } else { cur + GEN_STAGE_BYTES as u32 };
            c_idx += 1;
        }
        crate::ptx::wait();

        let output = words.add(2).read_volatile() as *mut f32;
        let stride = words.add(6).read_volatile() as usize;
        let out_table = words.add(10).read_volatile() as *const u32;
        let out_count = words.add(11).read_volatile() as usize;
        let split_count = words.add(15).read_volatile() as usize;
        let (g, t) = (lane / 4, lane % 4);
        let col0 = tile_n * TC_BN + warp_n * 16 + t * 2;
        let m0 = tile_m_off + warp_m * 32 + g;
        let m1 = m0 + 8;
        let out_r0 = if m0 < m { place(out_table, out_count, base + m0) } else { 0 };
        let out_r1 = if m1 < m { place(out_table, out_count, base + m1) } else { 0 };
        store_split_pair!(output, stride, split_count, m, m0, m1, out_r0, out_r1, col0, c00);
        store_split_pair!(output, stride, split_count, m, m0, m1, out_r0, out_r1, col0 + 8, c01);

        let m2 = tile_m_off + warp_m * 32 + 16 + g;
        let m3 = m2 + 8;
        let out_r2 = if m2 < m { place(out_table, out_count, base + m2) } else { 0 };
        let out_r3 = if m3 < m { place(out_table, out_count, base + m3) } else { 0 };
        store_split_pair!(output, stride, split_count, m, m2, m3, out_r2, out_r3, col0, c10);
        store_split_pair!(output, stride, split_count, m, m2, m3, out_r2, out_r3, col0 + 8, c11);
    }
}
