//! Bit-exact fused prefill message-meta and message-gate kernels (`resident_launch_prefill_gate_fuse`,
//! kind `PREFILL_GATE_FUSE = 123`).
//!
//! Eliminates `MIX_LIST_NORM_HALF`'s 2.72 GB/pass global materialization of FP32 `messages` and BF16 `msg_normed_half`,
//! eliminates all per-chunk `GateDown` launches (`TC_MODE_NARROW_HALF`), and runs at **2 blocks/SM** (`47,424 B` SMEM,
//! `<= 128` regs, `0` stack, `0` spill) with interleaved `(r_tile, gate)` CTAs sharing L2 on `S_0..S_{kept-1}`,
//! 4-stage loop-free `cp.async` `GateDown`, straight-line `BM = 8` `GateOuter + GateBlend + RoundedNorm`, and
//! 100% coalesced `st.global.v4.b32` BF16 stores.
//!
//! Bit-exactness invariants (Rule 5):
//! - `PG_OP_MSG_META` computes identical 32-lane warp-butterfly dots, `fixed_math::exp` softmax weights, and 128-element
//!   chunk-tree `RoundedNorm` scales to `MIX_LIST_NORM_HALF`.
//! - `PG_OP_FULL` Phase 1 accumulates `GateDown` columns `0..dim` (`kept_normed_half`) FIRST in strictly ascending
//!   `k = 0, 16, ..., dim - 16` order, then columns `dim..2*dim` (`msg_normed_half` synthesized on-the-fly from lazy
//!   sources `S_0..S_{kept-1}`) into the SAME FP32 accumulator in strictly ascending `k = dim, dim + 16, ..., 2*dim - 16`
//!   order — identical to `TC_MODE_NARROW_HALF`.
//! - `PG_OP_FULL` Phase 2 computes `GateOuter` in strictly ascending `k = 0, 16, ..., gate_rank - 16` order, exact
//!   `GateBlend` `(1 - tanh(f)) * kept + tanh(g) * msg`, and exact 128-chunk `RoundedNorm` tree — identical to
//!   `TC_MODE_GATE_BLEND_NORM`.

use crate::block_ops::{NORM_CHUNK, root};
use crate::fixed_math::{exp, rn, tanh};
use crate::prefill_gate::{
    PG_BM, PG_DIM_MAX, PG_FLAG_HALF_OUT, PG_GATE_AT, PG_GATE_STRIDE, PG_K_MAX,
    PG_KEPT_MAX, PG_META_THREADS, PG_OP_FULL, PG_OP_MSG_META, PG_OP_WORD, full_admitted,
    full_blocks, msg_meta_admitted, msg_meta_batch_rows, msg_meta_blocks,
};
use crate::tc_ptx::{
    cp_async_wait_two, cp_async_wait_zero, cvt_bf16x2, ld_global_nc_u32, ld_global_nc_v2_u32,
    ld_global_nc_v4, ldmatrix_x2, ldmatrix_x4, load_u32, mma_m16n8k16_bf16, shfl_xor_f32,
    st_shared_zero16, store_u32,
};

/// Unified `47,424 B` static SMEM (`2 * 47,424 = 94,848 B <= 101,376 B` -> **2 blocks/SM** on SM121a):
/// - `0 .. 45,056`: main workspace (`11` chunks of `8 x 128` FP32)
///   - `PG_OP_MSG_META`: `r_batch * kept * dim` FP32 (`<= 38,912 B`)
///   - `PG_OP_FULL` Phase 1 Half 1: 4 stages of `512 + 4,096 = 4,608 B` (`18,432 B`)
///   - `PG_OP_FULL` Phase 1 Half 2: 4 stages of `4 * 1,280 + 4,096 = 9,216 B` (`36,864 B`)
///   - `PG_OP_FULL` Phase 2: `11` XOR-swizzled `8 x 128` FP32 chunks (`45,056 B`), then reclaimed for
///     coalesced `half_out` register-chunk staging (`8 * 8 * 272 = 17,408 B`).
/// - `45,056 .. 46,208`: `INNER_SMEM` (`8` rows x `36` u32 = `1,152 B`, bank-free padded stride `144 B`).
/// - `46,208 .. 47,264`: `SUMS` (`8` rows x `33` u32 = `1,056 B`).
/// - `47,264 .. 47,392`: `MSG_W` (`8` rows x `4` kept FP32 = `128 B`).
/// - `47,392 .. 47,424`: `MSG_NS` (`8` rows FP32 = `32 B`).
const WORK_BYTES: usize = 11 * PG_BM * NORM_CHUNK * 4; // 45,056 B
const INNER_AT: u32 = WORK_BYTES as u32; // 45,056
const INNER_ROW_BYTES: usize = 36 * 4; // 144 B (32 u32 packed BF16 + 4 u32 pad -> bank-free)
const INNER_BYTES: usize = PG_BM * INNER_ROW_BYTES; // 1,152 B
const SUMS_AT: u32 = INNER_AT + INNER_BYTES as u32; // 46,208
const SUMS_BYTES: usize = PG_BM * 33 * 4; // 1,056 B
const MSG_W_AT: u32 = SUMS_AT + SUMS_BYTES as u32; // 47,264
const MSG_NS_AT: u32 = MSG_W_AT + (PG_BM * PG_KEPT_MAX * 4) as u32; // 47,392
const BLOCK_BYTES: usize = (MSG_NS_AT as usize) + PG_BM * 4; // 47,424 B

const H1_A_BYTES: usize = PG_BM * 32 * 2; // 512 B (`8 x 32` BF16)
const H1_B_BYTES: usize = PG_K_MAX * 32 * 2; // 4,096 B (`64 x 32` BF16)
const H1_STAGE: usize = H1_A_BYTES + H1_B_BYTES; // 4,608 B
const H1_STAGES: usize = 4;

const H2_S_ROW: usize = 160; // 32 floats + 8 pad floats (16B aligned, bank-free for load2)
const H2_S_E: usize = PG_BM * H2_S_ROW; // 1,280 B
const H2_B_OFF: u32 = (PG_KEPT_MAX * H2_S_E) as u32; // 5,120 B
const H2_STAGE: usize = PG_KEPT_MAX * H2_S_E + H1_B_BYTES; // 9,216 B
const H2_STAGES: usize = 4;

const EPI_ROW_BYTES: usize = 68 * 4; // 272 B (64 u32 packed BF16 + 4 u32 pad, bank-free)
const EPI_WARP_BYTES: usize = PG_BM * EPI_ROW_BYTES; // 2,176 B

const _: () = assert!(WORK_BYTES == 45_056 && BLOCK_BYTES == 47_424 && 2 * BLOCK_BYTES <= 101_376);
const _: () = assert!(H1_STAGES * H1_STAGE <= WORK_BYTES && H2_STAGES * H2_STAGE <= WORK_BYTES);
const _: () = assert!(2 * 2 * PG_DIM_MAX * 4 <= WORK_BYTES && 4 * PG_DIM_MAX * 4 <= WORK_BYTES);
const _: () = assert!(8 * EPI_WARP_BYTES <= WORK_BYTES);
const _: () = assert!(INNER_AT % 16 == 0 && SUMS_AT % 16 == 0 && MSG_W_AT % 16 == 0 && MSG_NS_AT % 16 == 0);

/// Zero-bank-conflict XOR-swizzled offset within a 64-byte-stride BF16 tile (`row`, 16B `chunk` in 0..4).
#[inline(always)]
const fn swiz64(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Bank-free XOR-swizzled offset within one `8 x 128` FP32 Phase-2 chunk (`row` in 0..8, `ni` in 0..16).
/// Each 8-float (32-byte) block stays contiguous for coalesced `load4` + `st.global.v4.b32` reads.
#[inline(always)]
const fn swiz_chunk_ni(row: usize, ni: usize) -> u32 {
    (row * 512 + ((ni ^ (row & 3)) * 32)) as u32
}

/// Replay one recorded command of kind `PREFILL_GATE_FUSE` (kind 123).
/// SAFETY (caller contract, `resident_launch_prefill_gate_fuse`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `PREFILL_GATE_FUSE` recorded by `Backend` in `backend/prefill_gate.rs`
/// after `prefill_gate::check_msg_meta` or `prefill_gate::check_full` verified all extents, alignments, and non-overlap.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind or sub-op traps before reading operands.
    // - `PG_OP_MSG_META`: host checked via `check_msg_meta` that `queries` (`asked * dim`), each `sources[e]`
    //   (`rows * dim`), `weights_out` (`asked * rows * kept`), and `nscales_out` (`asked * rows`) have valid
    //   extents, alignments, and disjoint outputs. All 256 threads reach every `_syncthreads` and full 32-lane
    //   warps reach every `shfl_xor`.
    // - `PG_OP_FULL`: host checked via `check_full` that every source, `w_down`, `w_outer`, `kept_val`,
    //   `kept_normed_half`, `msg_weights`, `msg_nscales`, optional `norm_w`, and output have valid extents,
    //   alignments, and pairwise disjoint outputs. All 256 threads reach every `_syncthreads` and `cp.async.wait_group`;
    //   every warp reaches `ldmatrix`, `mma`, and `shfl` uniformly. All SMEM accesses lie within `BLOCK_BYTES` (47,424 B).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PREFILL_GATE_FUSE {
            core::arch::nvptx::trap();
        }
        let w = c.payload.as_ptr().cast::<u64>();
        let subop = *w.add(PG_OP_WORD);
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        if subop == PG_OP_MSG_META {
            replay_msg_meta(w, smem);
        } else if subop == PG_OP_FULL {
            replay_full(w, smem);
        } else {
            core::arch::nvptx::trap();
        }
    }
}

/// Sub-op 0 (`PG_OP_MSG_META`): lazy message softmax weights and exact `RoundedNorm` scales for `Token::enter`.
/// SAFETY: caller contract of `replay` for `PG_OP_MSG_META`; `smem` is `BLOCK_BYTES` shared memory.
#[inline(always)]
unsafe fn replay_msg_meta(w: *const u64, smem: u32) {
    // SAFETY: validated by `check_msg_meta` on host and re-checked below; all 256 threads sync once after `cp.async`.
    unsafe {
        let queries = *w.add(1) as *const f32;
        let weights_out = *w.add(2) as *mut f32;
        let nscales_out = *w.add(3) as *mut f32;
        let rows = *w.add(4) as usize;
        let asked = *w.add(5) as usize;
        let kept = *w.add(6) as usize;
        let dim = *w.add(7) as usize;
        let se = *w.add(8);
        let scale = f32::from_bits(se as u32);
        let eps = f32::from_bits((se >> 32) as u32);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        if !msg_meta_admitted(rows, asked, kept, dim, threads)
            || core::arch::nvptx::_grid_dim_x() as usize != msg_meta_blocks(rows, kept) as usize
            || ((queries as usize) & 15) != 0
            || (((weights_out as usize) | (nscales_out as usize)) & 3) != 0
        {
            core::arch::nvptx::trap();
        }
        let s0 = *w.add(9) as *const f32;
        let s1 = if kept >= 2 { *w.add(10) as *const f32 } else { s0 };
        let s2 = if kept >= 3 { *w.add(11) as *const f32 } else { s0 };
        let s3 = if kept >= 4 { *w.add(12) as *const f32 } else { s0 };
        if ((s0 as usize) | (s1 as usize) | (s2 as usize) | (s3 as usize)) & 15 != 0 {
            core::arch::nvptx::trap();
        }
        let batch = msg_meta_batch_rows(kept); // 2 for kept <= 2, 1 for kept in 3..=4
        let r0 = block * batch;
        if r0 >= rows {
            return;
        }
        let r_cnt = (rows - r0).min(batch);
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let dim4 = dim >> 2;
        let total_chunks = r_cnt * kept * dim4;
        let mut idx = tid;
        while idx < total_chunks {
            let slot = idx / dim4;
            let c4 = idx % dim4;
            let rb = slot / kept;
            let e = slot % kept;
            let src_base = match e { 0 => s0, 1 => s1, 2 => s2, _ => s3 };
            crate::ptx::copy16(
                smem + (idx * 16) as u32,
                src_base.add((r0 + rb) * dim + c4 * 4).cast::<u32>(),
            );
            idx += PG_META_THREADS;
        }
        crate::ptx::commit();
        cp_async_wait_zero();
        core::arch::nvptx::_syncthreads();

        let (warp, lane) = (tid >> 5, tid & 31);
        let warps = PG_META_THREADS >> 5;
        let chunks = dim >> 7;
        let row_stride_b = (kept * dim * 4) as u32;
        let src_stride_b = (dim * 4) as u32;
        macro_rules! ldf {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
        macro_rules! bfly {($v:ident)=>{{
            $v += shfl_xor_f32($v, 16);
            $v += shfl_xor_f32($v, 8);
            $v += shfl_xor_f32($v, 4);
            $v += shfl_xor_f32($v, 2);
            $v += shfl_xor_f32($v, 1);
        }}}

        let mut q = warp;
        while q < asked {
            let q_ptr = queries.add(q * dim);
            // Accumulate dots for all `r_cnt * kept <= 4` pairs in ONE pass over `Q[q]`!
            let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            if kept <= 2 {
                let b0 = smem;
                let b1 = smem + row_stride_b;
                let mut j = lane;
                while j < dim {
                    let qv = ldf!(q_ptr.add(j));
                    let jo = (j * 4) as u32;
                    d0 += f32::from_bits(load_u32(b0 + jo)) * qv;
                    if kept == 2 { d1 += f32::from_bits(load_u32(b0 + src_stride_b + jo)) * qv; }
                    if r_cnt >= 2 {
                        d2 += f32::from_bits(load_u32(b1 + jo)) * qv;
                        if kept == 2 { d3 += f32::from_bits(load_u32(b1 + src_stride_b + jo)) * qv; }
                    }
                    j += 32;
                }
            } else {
                // kept in 3..=4, r_cnt == 1
                let b0 = smem;
                let mut j = lane;
                while j < dim {
                    let qv = ldf!(q_ptr.add(j));
                    let jo = (j * 4) as u32;
                    d0 += f32::from_bits(load_u32(b0 + jo)) * qv;
                    d1 += f32::from_bits(load_u32(b0 + src_stride_b + jo)) * qv;
                    d2 += f32::from_bits(load_u32(b0 + 2 * src_stride_b + jo)) * qv;
                    if kept == 4 { d3 += f32::from_bits(load_u32(b0 + 3 * src_stride_b + jo)) * qv; }
                    j += 32;
                }
            }
            bfly!(d0); bfly!(d1); bfly!(d2); bfly!(d3);

            let mut rb = 0usize;
            while rb < r_cnt {
                let (dot0, dot1, dot2, dot3) = if kept <= 2 {
                    if rb == 0 { (d0, d1, 0.0f32, 0.0f32) } else { (d2, d3, 0.0f32, 0.0f32) }
                } else {
                    (d0, d1, d2, d3)
                };
                let mut largest = f32::NEG_INFINITY;
                largest = largest.max(dot0 * scale);
                if kept >= 2 { largest = largest.max(dot1 * scale); }
                if kept >= 3 { largest = largest.max(dot2 * scale); }
                if kept >= 4 { largest = largest.max(dot3 * scale); }
                let e0 = exp(dot0 * scale - largest);
                let e1 = if kept >= 2 { exp(dot1 * scale - largest) } else { 0.0f32 };
                let e2 = if kept >= 3 { exp(dot2 * scale - largest) } else { 0.0f32 };
                let e3 = if kept >= 4 { exp(dot3 * scale - largest) } else { 0.0f32 };
                let mut total_e = 0.0f32;
                total_e += e0;
                if kept >= 2 { total_e += e1; }
                if kept >= 3 { total_e += e2; }
                if kept >= 4 { total_e += e3; }
                let w0 = e0 / total_e;
                let w1 = if kept >= 2 { e1 / total_e } else { 0.0f32 };
                let w2 = if kept >= 3 { e2 / total_e } else { 0.0f32 };
                let w3 = if kept >= 4 { e3 / total_e } else { 0.0f32 };
                let row_g = r0 + rb;
                if lane == 0 {
                    let wp = weights_out.add((q * rows + row_g) * kept);
                    *wp = w0;
                    if kept >= 2 { *wp.add(1) = w1; }
                    if kept >= 3 { *wp.add(2) = w2; }
                    if kept >= 4 { *wp.add(3) = w3; }
                }
                let sb0 = smem + (rb as u32) * row_stride_b;
                let sb1 = sb0 + src_stride_b;
                let sb2 = sb1 + src_stride_b;
                let sb3 = sb2 + src_stride_b;
                let mut my_ch_sum = 0.0f32;
                let mut ch = 0usize;
                while ch < chunks {
                    let mut sq = 0.0f32;
                    let mut g4 = 0usize;
                    while g4 < 4 {
                        let jo = ((ch * NORM_CHUNK + lane + g4 * 32) * 4) as u32;
                        let mut m = 0.0f32;
                        m += w0 * f32::from_bits(load_u32(sb0 + jo));
                        if kept >= 2 { m += w1 * f32::from_bits(load_u32(sb1 + jo)); }
                        if kept >= 3 { m += w2 * f32::from_bits(load_u32(sb2 + jo)); }
                        if kept >= 4 { m += w3 * f32::from_bits(load_u32(sb3 + jo)); }
                        sq += m * m;
                        g4 += 1;
                    }
                    bfly!(sq);
                    if lane == ch { my_ch_sum = sq; }
                    ch += 1;
                }
                let mut total_sq = my_ch_sum;
                bfly!(total_sq);
                if lane == 0 {
                    *nscales_out.add(q * rows + row_g) = 1.0f32 / root(total_sq / dim as f32 + eps);
                }
                rb += 1;
            }
            q += warps;
        }
    }
}

/// Sub-op 1 (`PG_OP_FULL`): fused `GateDown + GateOuter + GateBlend + RoundedNorm` (`BM = 8`, `1` gate/CTA,
/// `2` blocks/SM, `<= 128` regs, `0` stack/spill, interleaved `(r_tile, gate)` for L2 source reuse).
/// SAFETY: caller contract of `replay` for `PG_OP_FULL`; `smem` is `BLOCK_BYTES` shared memory.
#[inline(always)]
unsafe fn replay_full(w: *const u64, smem: u32) {
    // SAFETY: validated by `check_full` on host and re-checked below; all 256 threads participate in every
    // `_syncthreads` and `cp.async.wait_group`; every warp participates uniformly in `ldmatrix`, `mma`, and `shfl`.
    unsafe {
        let rows = *w.add(1) as usize;
        let dim = *w.add(2) as usize;
        let k = *w.add(3) as usize;
        let count = *w.add(4) as usize;
        let kept = *w.add(5) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        if !full_admitted(rows, dim, k, count, kept, threads)
            || core::arch::nvptx::_grid_dim_x() as usize != full_blocks(rows, count) as usize
        {
            core::arch::nvptx::trap();
        }
        // Interleave `(r_tile, gi)` so sibling gates `0..count` of the same 8 rows share L2 on `S_0..S_{kept-1}`.
        let r_tile = block / count;
        let gi = block % count;
        let r0 = r_tile * PG_BM;
        if r0 >= rows {
            return;
        }
        let g_base = PG_GATE_AT + gi * PG_GATE_STRIDE;
        let s0 = *w.add(7) as *const f32;
        let s1 = if kept >= 2 { *w.add(8) as *const f32 } else { s0 };
        let s2 = if kept >= 3 { *w.add(9) as *const f32 } else { s0 };
        let s3 = if kept >= 4 { *w.add(10) as *const f32 } else { s0 };
        let wd = *w.add(g_base) as *const u32;
        let kh = *w.add(g_base + 3) as *const u32;
        if ((s0 as usize) | (s1 as usize) | (s2 as usize) | (s3 as usize) | (wd as usize) | (kh as usize)) & 15 != 0 {
            core::arch::nvptx::trap();
        }

        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid >> 5, tid & 31);
        let (g_l, t) = (lane >> 2, lane & 3);

        // Preload `msg_weights[8][4]` and `msg_nscales[8]` for this gate into SMEM once.
        if tid < PG_BM * PG_KEPT_MAX {
            let rl = tid >> 2;
            let e = tid & 3;
            let val = if r0 + rl < rows && e < kept {
                let mw_p = *w.add(g_base + 4) as *const f32;
                *mw_p.add((r0 + rl) * kept + e)
            } else {
                0.0f32
            };
            store_u32(smem + MSG_W_AT + (tid * 4) as u32, val.to_bits());
        } else if tid < PG_BM * PG_KEPT_MAX + PG_BM {
            let rl = tid - PG_BM * PG_KEPT_MAX;
            let val = if r0 + rl < rows {
                let mn_p = *w.add(g_base + 5) as *const f32;
                *mn_p.add(r0 + rl)
            } else {
                0.0f32
            };
            store_u32(smem + MSG_NS_AT + (rl * 4) as u32, val.to_bits());
        }

        // =========================================================================
        // PHASE 1 HALF 1: `GateDown` cols `0..dim` (`kept_normed_half * W_down[:, 0..dim]^T`)
        // Loop-free 4-stage `cp.async`: each thread issues 1 B copy (+ 1 A copy for tid < 32).
        // =========================================================================
        let steps = (dim >> 5) as u32;
        let half_w = dim >> 1;
        let a_r = tid >> 2;
        let a_ch = tid & 3;
        let a_dst = swiz64(a_r, a_ch);
        let b_n = tid >> 2;
        let b_ch = tid & 3;
        let b_dst = H1_A_BYTES as u32 + swiz64(b_n, b_ch);
        let a_ok = tid < 32 && r0 + a_r < rows;
        let b_ok = b_n < k;

        let mut st = 0usize;
        while st < H1_STAGES {
            let sb = smem + (st * H1_STAGE) as u32;
            if tid < 32 && !a_ok { st_shared_zero16(sb + a_dst); }
            if !b_ok { st_shared_zero16(sb + b_dst); }
            st += 1;
        }
        let issue_h1 = |sb: u32, step: u32| {
            let step_u = (step as usize) * 16;
            if a_ok {
                crate::ptx::copy16(sb + a_dst, kh.add((r0 + a_r) * half_w + step_u + a_ch * 4));
            }
            if b_ok {
                crate::ptx::copy16(sb + b_dst, wd.add(b_n * dim + step_u + b_ch * 4));
            }
        };

        let mut c_idx = 0u32;
        while c_idx < (H1_STAGES - 1) as u32 {
            if c_idx < steps { issue_h1(smem + c_idx * H1_STAGE as u32, c_idx); }
            crate::ptx::commit();
            c_idx += 1;
        }

        let p1_busy = warp * 8 < k;
        let a_off = swiz64(lane & 7, lane >> 4);
        let b_off_h1 = H1_A_BYTES as u32 + swiz64(warp * 8 + (lane & 7), (lane >> 3) & 1);
        let mut c0 = [0.0f32; 4];

        let last_h1 = smem + ((H1_STAGES - 1) * H1_STAGE) as u32;
        let (mut cur, mut ahead) = (smem, last_h1);
        c_idx = 0;
        while c_idx < steps {
            cp_async_wait_two();
            core::arch::nvptx::_syncthreads();
            if c_idx + 3 < steps { issue_h1(ahead, c_idx + 3); }
            crate::ptx::commit();
            if p1_busy {
                let a0 = ldmatrix_x4(cur + a_off);
                let b0 = ldmatrix_x2(cur + b_off_h1);
                c0 = mma_m16n8k16_bf16(a0, b0, c0);
                let a1 = ldmatrix_x4(cur + (a_off ^ 32));
                let b1 = ldmatrix_x2(cur + (b_off_h1 ^ 32));
                c0 = mma_m16n8k16_bf16(a1, b1, c0);
            }
            ahead = cur;
            cur = if cur == last_h1 { smem } else { cur + H1_STAGE as u32 };
            c_idx += 1;
        }
        cp_async_wait_zero();
        core::arch::nvptx::_syncthreads();

        // =========================================================================
        // PHASE 1 HALF 2: `GateDown` cols `dim..2*dim` (`msg_normed_half * W_down[:, dim..2*dim]^T`)
        // Loop-free 4-stage `cp.async`: each thread issues <= 1 S copy + 1 B copy.
        // =========================================================================
        let s_e = tid >> 6;
        let s_r = (tid >> 3) & 7;
        let s_p4 = tid & 7;
        let s_in_e = tid < kept * 64;
        let s_ok = s_in_e && r0 + s_r < rows;
        let s_dst = (s_e * H2_S_E + s_r * H2_S_ROW + s_p4 * 16) as u32;
        let b2_dst = H2_B_OFF + swiz64(b_n, b_ch);
        let s_src_base = match s_e { 0 => s0, 1 => s1, 2 => s2, _ => s3 };

        st = 0;
        while st < H2_STAGES {
            let sb = smem + (st * H2_STAGE) as u32;
            if s_in_e && !s_ok { st_shared_zero16(sb + s_dst); }
            if !b_ok { st_shared_zero16(sb + b2_dst); }
            st += 1;
        }
        let issue_h2 = |sb: u32, step: u32| {
            let step_f = (step as usize) * 32;
            let step_u = (step as usize) * 16;
            if s_ok {
                crate::ptx::copy16(sb + s_dst, s_src_base.add((r0 + s_r) * dim + step_f + s_p4 * 4).cast::<u32>());
            }
            if b_ok {
                crate::ptx::copy16(sb + b2_dst, wd.add(b_n * dim + half_w + step_u + b_ch * 4));
            }
        };

        c_idx = 0;
        while c_idx < (H2_STAGES - 1) as u32 {
            if c_idx < steps { issue_h2(smem + c_idx * H2_STAGE as u32, c_idx); }
            crate::ptx::commit();
            c_idx += 1;
        }

        // Load this thread's row `g_l` softmax weights & nscale once into registers for Half 2.
        let mw_row = smem + MSG_W_AT + (g_l * PG_KEPT_MAX * 4) as u32;
        let mw0 = f32::from_bits(load_u32(mw_row));
        let mw1 = if kept >= 2 { f32::from_bits(load_u32(mw_row + 4)) } else { 0.0f32 };
        let ns0 = f32::from_bits(load_u32(smem + MSG_NS_AT + (g_l * 4) as u32));
        let s_off = (g_l * H2_S_ROW + t * 8) as u32;
        let b_off_h2 = H2_B_OFF + swiz64(warp * 8 + (lane & 7), (lane >> 3) & 1);
        let last_h2 = smem + ((H2_STAGES - 1) * H2_STAGE) as u32;
        let (mut cur2, mut ahead2) = (smem, last_h2);
        c_idx = 0;
        while c_idx < steps {
            cp_async_wait_two();
            core::arch::nvptx::_syncthreads();
            if c_idx + 3 < steps { issue_h2(ahead2, c_idx + 3); }
            crate::ptx::commit();
            if p1_busy {
                let mut ks = 0u32;
                while ks < 2 {
                    let so = cur2 + s_off + ks * 64;
                    let bo = cur2 + (b_off_h2 ^ (ks * 32));
                    let v00 = crate::ptx::load2(so);
                    let v02 = crate::ptx::load2(so + 32);
                    let mut m00 = 0.0f32; let mut m01 = 0.0f32;
                    let mut m20 = 0.0f32; let mut m21 = 0.0f32;
                    m00 += mw0 * f32::from_bits(v00[0]);
                    m01 += mw0 * f32::from_bits(v00[1]);
                    m20 += mw0 * f32::from_bits(v02[0]);
                    m21 += mw0 * f32::from_bits(v02[1]);
                    if kept >= 2 {
                        let v10 = crate::ptx::load2(so + H2_S_E as u32);
                        let v12 = crate::ptx::load2(so + H2_S_E as u32 + 32);
                        m00 += mw1 * f32::from_bits(v10[0]);
                        m01 += mw1 * f32::from_bits(v10[1]);
                        m20 += mw1 * f32::from_bits(v12[0]);
                        m21 += mw1 * f32::from_bits(v12[1]);
                    }
                    if kept >= 3 {
                        let mw2 = f32::from_bits(load_u32(mw_row + 8));
                        let v20 = crate::ptx::load2(so + (2 * H2_S_E) as u32);
                        let v22 = crate::ptx::load2(so + (2 * H2_S_E) as u32 + 32);
                        m00 += mw2 * f32::from_bits(v20[0]);
                        m01 += mw2 * f32::from_bits(v20[1]);
                        m20 += mw2 * f32::from_bits(v22[0]);
                        m21 += mw2 * f32::from_bits(v22[1]);
                    }
                    if kept >= 4 {
                        let mw3 = f32::from_bits(load_u32(mw_row + 12));
                        let v30 = crate::ptx::load2(so + (3 * H2_S_E) as u32);
                        let v32 = crate::ptx::load2(so + (3 * H2_S_E) as u32 + 32);
                        m00 += mw3 * f32::from_bits(v30[0]);
                        m01 += mw3 * f32::from_bits(v30[1]);
                        m20 += mw3 * f32::from_bits(v32[0]);
                        m21 += mw3 * f32::from_bits(v32[1]);
                    }
                    let a = [cvt_bf16x2(m00 * ns0, m01 * ns0), 0u32, cvt_bf16x2(m20 * ns0, m21 * ns0), 0u32];
                    let b = ldmatrix_x2(bo);
                    c0 = mma_m16n8k16_bf16(a, b, c0);
                    ks += 1;
                }
            }
            ahead2 = cur2;
            cur2 = if cur2 == last_h2 { smem } else { cur2 + H2_STAGE as u32 };
            c_idx += 1;
        }
        cp_async_wait_zero();

        // Store rounded BF16 `inner[8][64]` into bank-free `INNER_SMEM` (zero global round-trip!).
        let inner_v = if p1_busy { cvt_bf16x2(c0[0], c0[1]) } else { 0u32 };
        store_u32(smem + INNER_AT + (g_l * INNER_ROW_BYTES + (warp * 4 + t) * 4) as u32, inner_v);
        core::arch::nvptx::_syncthreads();

        // =========================================================================
        // PHASE 2: `GateOuter + GateBlend + RoundedNorm` (`BM = 8`, 8 warps, 2 blk/SM!)
        // Identical 8-reg `A` + 32-reg `rv0..rv31` structure as `plain_tc_big` (`<= 128` regs!).
        // =========================================================================
        let w_outer = *w.add(g_base + 1) as *const u32;
        let kept_val = *w.add(g_base + 2) as *const f32;
        if ((w_outer as usize) | (kept_val as usize)) & 15 != 0 {
            core::arch::nvptx::trap();
        }
        let k_steps = k >> 4;
        let kw = k >> 1;
        let chunks = dim >> 7;
        let smem_end = chunks.saturating_sub(8); // 10 for dim=2304, <= 11 for dim<=2432
        let r = r0 + g_l;
        let r_ok = r < rows;

        // Load full `K <= 64` BF16 `inner[g_l]` row fragment into 8 registers ONCE (zero bank conflicts!).
        let in_r = smem + INNER_AT + (g_l * INNER_ROW_BYTES + t * 4) as u32;
        let (a00, a20) = if r_ok { (load_u32(in_r), load_u32(in_r + 16)) } else { (0, 0) };
        let (a01, a21) = if r_ok && k_steps > 1 { (load_u32(in_r + 32), load_u32(in_r + 48)) } else { (0, 0) };
        let (a02, a22) = if r_ok && k_steps > 2 { (load_u32(in_r + 64), load_u32(in_r + 80)) } else { (0, 0) };
        let (a03, a23) = if r_ok && k_steps > 3 { (load_u32(in_r + 96), load_u32(in_r + 112)) } else { (0, 0) };

        let r_off = r * dim;
        let k_row = if r_ok { kept_val.add(r_off).cast::<u32>() } else { core::ptr::null() };
        let s0_row = if r_ok { s0.add(r_off).cast::<u32>() } else { core::ptr::null() };
        let s1_row = if r_ok && kept >= 2 { s1.add(r_off).cast::<u32>() } else { core::ptr::null() };

        macro_rules! blend_ni {
            ($nf:expr) => {{
                let nf = $nf;
                let wf_p = w_outer.add((nf + g_l) * kw + t);
                let wg_p = w_outer.add((dim + nf + g_l) * kw + t);
                let mut df = mma_m16n8k16_bf16([a00, 0, a20, 0], [ld_global_nc_u32(wf_p), ld_global_nc_u32(wf_p.add(4))], [0.0f32; 4]);
                let mut dg = mma_m16n8k16_bf16([a00, 0, a20, 0], [ld_global_nc_u32(wg_p), ld_global_nc_u32(wg_p.add(4))], [0.0f32; 4]);
                if k_steps > 1 {
                    df = mma_m16n8k16_bf16([a01, 0, a21, 0], [ld_global_nc_u32(wf_p.add(8)), ld_global_nc_u32(wf_p.add(12))], df);
                    dg = mma_m16n8k16_bf16([a01, 0, a21, 0], [ld_global_nc_u32(wg_p.add(8)), ld_global_nc_u32(wg_p.add(12))], dg);
                }
                if k_steps > 2 {
                    df = mma_m16n8k16_bf16([a02, 0, a22, 0], [ld_global_nc_u32(wf_p.add(16)), ld_global_nc_u32(wf_p.add(20))], df);
                    dg = mma_m16n8k16_bf16([a02, 0, a22, 0], [ld_global_nc_u32(wg_p.add(16)), ld_global_nc_u32(wg_p.add(20))], dg);
                }
                if k_steps > 3 {
                    df = mma_m16n8k16_bf16([a03, 0, a23, 0], [ld_global_nc_u32(wf_p.add(24)), ld_global_nc_u32(wf_p.add(28))], df);
                    dg = mma_m16n8k16_bf16([a03, 0, a23, 0], [ld_global_nc_u32(wg_p.add(24)), ld_global_nc_u32(wg_p.add(28))], dg);
                }
                if r_ok {
                    let col = nf + t * 2;
                    let kv = ld_global_nc_v2_u32(k_row.add(col));
                    let sv0 = ld_global_nc_v2_u32(s0_row.add(col));
                    let mut m0 = 0.0f32; let mut m1 = 0.0f32;
                    m0 += mw0 * f32::from_bits(sv0[0]);
                    m1 += mw0 * f32::from_bits(sv0[1]);
                    if kept >= 2 {
                        let sv1 = ld_global_nc_v2_u32(s1_row.add(col));
                        m0 += mw1 * f32::from_bits(sv1[0]);
                        m1 += mw1 * f32::from_bits(sv1[1]);
                    }
                    if kept >= 3 {
                        let mw2 = f32::from_bits(load_u32(mw_row + 8));
                        let sv2 = ld_global_nc_v2_u32((*w.add(9) as *const f32).add(r_off + col).cast::<u32>());
                        m0 += mw2 * f32::from_bits(sv2[0]);
                        m1 += mw2 * f32::from_bits(sv2[1]);
                    }
                    if kept >= 4 {
                        let mw3 = f32::from_bits(load_u32(mw_row + 12));
                        let sv3 = ld_global_nc_v2_u32((*w.add(10) as *const f32).add(r_off + col).cast::<u32>());
                        m0 += mw3 * f32::from_bits(sv3[0]);
                        m1 += mw3 * f32::from_bits(sv3[1]);
                    }
                    (
                        (1.0f32 - tanh(df[0])) * f32::from_bits(kv[0]) + tanh(dg[0]) * m0,
                        (1.0f32 - tanh(df[1])) * f32::from_bits(kv[1]) + tanh(dg[1]) * m1,
                    )
                } else {
                    (0.0f32, 0.0f32)
                }
            }};
        }

        // Phase 2a: SMEM-backed chunks (`ch < smem_end`, 0 `rv` regs live).
        let mut ch = warp;
        while ch < smem_end {
            let col_base = ch * NORM_CHUNK;
            let chunk_base = smem + (ch * PG_BM * NORM_CHUNK * 4) as u32;
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
                let dst = chunk_base + swiz_chunk_ni(g_l, ni) + (t * 8) as u32;
                store_u32(dst, v0.to_bits());
                store_u32(dst + 4, v1.to_bits());
                ni += 1;
            }
            let e8 = (se0 + se2) + (se1 + se3);
            let o8 = (so0 + so2) + (so1 + so3);
            let e4 = e8 + shfl_xor_f32(e8, 2);
            let o4 = o8 + shfl_xor_f32(o8, 2);
            let e2 = e4 + shfl_xor_f32(e4, 1);
            let o2 = o4 + shfl_xor_f32(o4, 1);
            if t == 0 {
                store_u32(smem + SUMS_AT + ((g_l * 33 + ch) * 4) as u32, (e2 + o2).to_bits());
            }
            ch += 8;
        }

        // Phase 2b: Register-resident chunk (`reg_ch = smem_end + warp`), 32 scalar regs `rv0..rv31` straight-line.
        let reg_ch = smem_end + warp;
        let (mut rv0, mut rv1, mut rv2, mut rv3, mut rv4, mut rv5, mut rv6, mut rv7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut rv8, mut rv9, mut rv10, mut rv11, mut rv12, mut rv13, mut rv14, mut rv15) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut rv16, mut rv17, mut rv18, mut rv19, mut rv20, mut rv21, mut rv22, mut rv23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut rv24, mut rv25, mut rv26, mut rv27, mut rv28, mut rv29, mut rv30, mut rv31) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        if reg_ch < chunks {
            let col_base = reg_ch * NORM_CHUNK;
            let (mut se0, mut se1, mut se2, mut se3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut so0, mut so1, mut so2, mut so3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            macro_rules! step_reg {($ni:expr, $ve:ident, $vo:ident, $se:ident, $so:ident) => {{
                let (v0, v1) = blend_ni!(col_base + $ni * 8);
                $ve = v0;
                $vo = v1;
                $se += v0 * v0;
                $so += v1 * v1;
            }}}
            step_reg!(0,  rv0,  rv1,  se0, so0);
            step_reg!(1,  rv2,  rv3,  se1, so1);
            step_reg!(2,  rv4,  rv5,  se2, so2);
            step_reg!(3,  rv6,  rv7,  se3, so3);
            step_reg!(4,  rv8,  rv9,  se0, so0);
            step_reg!(5,  rv10, rv11, se1, so1);
            step_reg!(6,  rv12, rv13, se2, so2);
            step_reg!(7,  rv14, rv15, se3, so3);
            step_reg!(8,  rv16, rv17, se0, so0);
            step_reg!(9,  rv18, rv19, se1, so1);
            step_reg!(10, rv20, rv21, se2, so2);
            step_reg!(11, rv22, rv23, se3, so3);
            step_reg!(12, rv24, rv25, se0, so0);
            step_reg!(13, rv26, rv27, se1, so1);
            step_reg!(14, rv28, rv29, se2, so2);
            step_reg!(15, rv30, rv31, se3, so3);
            let e8 = (se0 + se2) + (se1 + se3);
            let o8 = (so0 + so2) + (so1 + so3);
            let e4 = e8 + shfl_xor_f32(e8, 2);
            let o4 = o8 + shfl_xor_f32(o8, 2);
            let e2 = e4 + shfl_xor_f32(e4, 1);
            let o2 = o4 + shfl_xor_f32(o4, 1);
            if t == 0 {
                store_u32(smem + SUMS_AT + ((g_l * 33 + reg_ch) * 4) as u32, (e2 + o2).to_bits());
            }
        }
        core::arch::nvptx::_syncthreads();

        // Reload epilogue-only pointers after barrier so they occupy 0 regs across MMA/tanh.
        let ef = w.add(6).read_volatile();
        let eps = f32::from_bits(ef as u32);
        let half_out = ((ef >> 32) & PG_FLAG_HALF_OUT) != 0;
        let norm_w = w.add(g_base + 6).read_volatile() as *const f32;
        let out_p = w.add(g_base + 7).read_volatile() as *mut f32;
        if (!norm_w.is_null() && ((norm_w as usize) & 15) != 0)
            || ((out_p as usize) & if half_out { 15 } else { 3 }) != 0
        {
            core::arch::nvptx::trap();
        }

        // Row-wise `RoundedNorm` scale reduction: warp `w` reduces row `w` (`0..8`) in parallel!
        let mut tot = if lane < chunks {
            f32::from_bits(load_u32(smem + SUMS_AT + ((warp * 33 + lane) * 4) as u32))
        } else { 0.0f32 };
        tot += shfl_xor_f32(tot, 16);
        tot += shfl_xor_f32(tot, 8);
        tot += shfl_xor_f32(tot, 4);
        tot += shfl_xor_f32(tot, 2);
        tot += shfl_xor_f32(tot, 1);
        if lane == 0 {
            store_u32(smem + SUMS_AT + ((warp * 33 + 32) * 4) as u32, (1.0f32 / root(tot / dim as f32 + eps)).to_bits());
        }
        core::arch::nvptx::_syncthreads();

        // Epilogue Part 1: drain SMEM chunks with 100% coalesced `st.global.v4.b32`!
        let has_nw = !norm_w.is_null();
        let mut ch = warp;
        while ch < smem_end {
            let col_base = ch * NORM_CHUNK;
            let chunk_base = smem + (ch * PG_BM * NORM_CHUNK * 4) as u32;
            if half_out {
                let out_u32 = out_p.cast::<u32>();
                let ni = lane & 15;
                let rl_sub = lane >> 4;
                let mut pr = 0u32;
                while pr < 4 {
                    let mut pr_v = pr;
                    core::arch::asm!("/* {0} */", inout(reg32) pr_v, options(nomem, nostack));
                    let rl = (pr_v as usize) * 2 + rl_sub;
                    if r0 + rl < rows {
                        let sc = f32::from_bits(load_u32(smem + SUMS_AT + ((rl * 33 + 32) * 4) as u32));
                        let src_s = chunk_base + swiz_chunk_ni(rl, ni);
                        let u0 = crate::ptx::load4(src_s);
                        let (p0, p1) = if has_nw {
                            let nw0 = ld_global_nc_v4(norm_w.add(col_base + ni * 8));
                            (
                                cvt_bf16x2((f32::from_bits(u0[0]) * sc) * nw0[0], (f32::from_bits(u0[1]) * sc) * nw0[1]),
                                cvt_bf16x2((f32::from_bits(u0[2]) * sc) * nw0[2], (f32::from_bits(u0[3]) * sc) * nw0[3]),
                            )
                        } else {
                            (
                                cvt_bf16x2(f32::from_bits(u0[0]) * sc, f32::from_bits(u0[1]) * sc),
                                cvt_bf16x2(f32::from_bits(u0[2]) * sc, f32::from_bits(u0[3]) * sc),
                            )
                        };
                        let u1 = crate::ptx::load4(src_s + 16);
                        let (p2, p3) = if has_nw {
                            let nw1 = ld_global_nc_v4(norm_w.add(col_base + ni * 8 + 4));
                            (
                                cvt_bf16x2((f32::from_bits(u1[0]) * sc) * nw1[0], (f32::from_bits(u1[1]) * sc) * nw1[1]),
                                cvt_bf16x2((f32::from_bits(u1[2]) * sc) * nw1[2], (f32::from_bits(u1[3]) * sc) * nw1[3]),
                            )
                        } else {
                            (
                                cvt_bf16x2(f32::from_bits(u1[0]) * sc, f32::from_bits(u1[1]) * sc),
                                cvt_bf16x2(f32::from_bits(u1[2]) * sc, f32::from_bits(u1[3]) * sc),
                            )
                        };
                        let dst = out_u32.add((r0 + rl) * half_w + (col_base >> 1) + ni * 4);
                        core::arch::asm!(
                            "st.global.v4.b32 [{dst}], {{{a0}, {a1}, {a2}, {a3}}};",
                            dst = in(reg64) dst,
                            a0 = in(reg32) p0, a1 = in(reg32) p1, a2 = in(reg32) p2, a3 = in(reg32) p3,
                            options(nostack),
                        );
                    }
                    pr += 1;
                }
            } else {
                let ni = lane >> 1;
                let half_b = lane & 1;
                let nw = if has_nw { ld_global_nc_v4(norm_w.add(col_base + ni * 8 + half_b * 4)) } else { [1.0; 4] };
                let mut rl = 0u32;
                while rl < PG_BM as u32 {
                    let mut rl_v = rl;
                    core::arch::asm!("/* {0} */", inout(reg32) rl_v, options(nomem, nostack));
                    let rlu = rl_v as usize;
                    if r0 + rlu < rows {
                        let sc = f32::from_bits(load_u32(smem + SUMS_AT + ((rlu * 33 + 32) * 4) as u32));
                        let u = crate::ptx::load4(chunk_base + swiz_chunk_ni(rlu, ni) + (half_b * 16) as u32);
                        let (y0, y1, y2, y3) = if has_nw {
                            (
                                rn((f32::from_bits(u[0]) * sc) * nw[0]),
                                rn((f32::from_bits(u[1]) * sc) * nw[1]),
                                rn((f32::from_bits(u[2]) * sc) * nw[2]),
                                rn((f32::from_bits(u[3]) * sc) * nw[3]),
                            )
                        } else {
                            (
                                rn(f32::from_bits(u[0]) * sc),
                                rn(f32::from_bits(u[1]) * sc),
                                rn(f32::from_bits(u[2]) * sc),
                                rn(f32::from_bits(u[3]) * sc),
                            )
                        };
                        let dst = out_p.add((r0 + rlu) * dim + col_base + ni * 8 + half_b * 4);
                        *dst = y0; *dst.add(1) = y1; *dst.add(2) = y2; *dst.add(3) = y3;
                    }
                    rl += 1;
                }
            }
            ch += 8;
        }

        // Epilogue Part 2: drain register-resident chunk `reg_ch` (reusing dead SMEM for coalesced `st.global.v4.b32`!).
        if half_out {
            core::arch::nvptx::_syncthreads();
            if reg_ch < chunks {
                let col_base = reg_ch * NORM_CHUNK;
                let sc0 = f32::from_bits(load_u32(smem + SUMS_AT + ((g_l * 33 + 32) * 4) as u32));
                let w_smem = smem + (warp * EPI_WARP_BYTES) as u32;
                let r_smem = w_smem + (g_l * EPI_ROW_BYTES) as u32;
                macro_rules! stage_reg_half {($ni:expr, $ve:ident, $vo:ident) => {{
                    if r_ok {
                        let col = col_base + $ni * 8 + t * 2;
                        let cu = ($ni * 4 + t) as u32;
                        let packed = if has_nw {
                            let nw = ld_global_nc_v2_u32(norm_w.add(col).cast());
                            cvt_bf16x2(($ve * sc0) * f32::from_bits(nw[0]), ($vo * sc0) * f32::from_bits(nw[1]))
                        } else {
                            cvt_bf16x2($ve * sc0, $vo * sc0)
                        };
                        store_u32(r_smem + cu * 4, packed);
                    }
                }}}
                stage_reg_half!(0,  rv0,  rv1);
                stage_reg_half!(1,  rv2,  rv3);
                stage_reg_half!(2,  rv4,  rv5);
                stage_reg_half!(3,  rv6,  rv7);
                stage_reg_half!(4,  rv8,  rv9);
                stage_reg_half!(5,  rv10, rv11);
                stage_reg_half!(6,  rv12, rv13);
                stage_reg_half!(7,  rv14, rv15);
                stage_reg_half!(8,  rv16, rv17);
                stage_reg_half!(9,  rv18, rv19);
                stage_reg_half!(10, rv20, rv21);
                stage_reg_half!(11, rv22, rv23);
                stage_reg_half!(12, rv24, rv25);
                stage_reg_half!(13, rv26, rv27);
                stage_reg_half!(14, rv28, rv29);
                stage_reg_half!(15, rv30, rv31);
                crate::ptx::warp_sync();
                let out_u32 = out_p.cast::<u32>();
                let r_valid = rows.saturating_sub(r0).min(PG_BM);
                let mut p = 0u32;
                while p < 4 {
                    let mut p_v = p;
                    core::arch::asm!("/* {0} */", inout(reg32) p_v, options(nomem, nostack));
                    let cidx = (p_v as usize) * 32 + lane;
                    let rl = cidx >> 4;
                    let ch16 = cidx & 15;
                    if rl < r_valid {
                        let vals = crate::ptx::load4(w_smem + (rl * EPI_ROW_BYTES + ch16 * 16) as u32);
                        let dst = out_u32.add((r0 + rl) * half_w + (col_base >> 1) + ch16 * 4);
                        core::arch::asm!(
                            "st.global.v4.b32 [{dst}], {{{a0}, {a1}, {a2}, {a3}}};",
                            dst = in(reg64) dst,
                            a0 = in(reg32) vals[0], a1 = in(reg32) vals[1], a2 = in(reg32) vals[2], a3 = in(reg32) vals[3],
                            options(nostack),
                        );
                    }
                    p += 1;
                }
            }
        } else if reg_ch < chunks && r_ok {
            let col_base = reg_ch * NORM_CHUNK;
            let sc0 = f32::from_bits(load_u32(smem + SUMS_AT + ((g_l * 33 + 32) * 4) as u32));
            macro_rules! store_reg_f32 {($ni:expr, $ve:ident, $vo:ident) => {{
                let col = col_base + $ni * 8 + t * 2;
                let dst = out_p.add(r_off + col);
                if has_nw {
                    let nw = ld_global_nc_v2_u32(norm_w.add(col).cast());
                    *dst = rn(($ve * sc0) * f32::from_bits(nw[0]));
                    *dst.add(1) = rn(($vo * sc0) * f32::from_bits(nw[1]));
                } else {
                    *dst = rn($ve * sc0);
                    *dst.add(1) = rn($vo * sc0);
                }
            }}}
            store_reg_f32!(0,  rv0,  rv1);
            store_reg_f32!(1,  rv2,  rv3);
            store_reg_f32!(2,  rv4,  rv5);
            store_reg_f32!(3,  rv6,  rv7);
            store_reg_f32!(4,  rv8,  rv9);
            store_reg_f32!(5,  rv10, rv11);
            store_reg_f32!(6,  rv12, rv13);
            store_reg_f32!(7,  rv14, rv15);
            store_reg_f32!(8,  rv16, rv17);
            store_reg_f32!(9,  rv18, rv19);
            store_reg_f32!(10, rv20, rv21);
            store_reg_f32!(11, rv22, rv23);
            store_reg_f32!(12, rv24, rv25);
            store_reg_f32!(13, rv26, rv27);
            store_reg_f32!(14, rv28, rv29);
            store_reg_f32!(15, rv30, rv31);
        }
    }
}
