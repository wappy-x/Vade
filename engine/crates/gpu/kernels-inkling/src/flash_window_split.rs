//! GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention (`FLASH_WINDOW_SPLIT16 = 118`,
//! `FLASH_WINDOW_SPLIT32 = 119`) and deterministic Split-KV reduction (`FLASH_WINDOW_REDUCE = 120`).
//!
//! Packs `rows_per_cta * (heads / kv_heads)` queries sharing one KV head into `16` or `32` MMA rows so each KV
//! token is read from DRAM once across all GQA query heads and verify rows, while 4 warps split `BKV = 32` keys
//! in `Q * K^T` and split `HEAD_DIM = 128` (32 cols/warp) in `P * V`. Zero-copy provisional sliding-window deltas
//! read committed keys from `keys/values` and provisional rows from `delta_keys/delta_values` with zero ring copy.

use crate::fixed_math::{exp, rn, sigmoid};
use crate::flash_window_abi::{
    FW_HEAD_DIM, FW_REL_DIM, FW_SPLIT_BKV, FW_SP_FLAGS_WORD, FW_SP_LIVE_AT, FW_SP_PLANE_COUNT_WORD,
    FW_SP_PLANES_AT, FW_SP_POS_AT, FW_SP_ROWS_PER_CTA_WORD, FW_SP_SPLIT_START_WORD, FW_THREADS,
    flash_window_split_admitted,
};
use crate::tc_ptx::{
    cp_async_wait_one, cp_async_wait_zero, cvt_bf16x2, ldmatrix_x2, ldmatrix_x4, ldmatrix_x4_trans,
    load_u32, mma_m16n8k16_bf16, shfl_xor_f32, st_shared_zero16, store_u16, store_u32,
};

const M_PITCH: usize = FW_REL_DIM + 4;

/// Zero-bank-conflict XOR-swizzled byte offset within a 256-byte-stride BF16 tile (`row`, 16B `chunk` in 0..16).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 256 + ((chunk ^ (row & 7)) * 16)) as u32
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Zero-bank-conflict XOR-swizzled byte offset within a 64-byte-stride BF16 tile (`row`, 16B `chunk` in 0..4).
#[inline(always)]
const fn swiz64(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention (`FLASH_WINDOW_SPLIT16` when `Q_GROUPS == 1`,
/// `FLASH_WINDOW_SPLIT32` when `Q_GROUPS == 2`).
///
/// Packs `rows_per_cta * (heads / kv_heads)` queries sharing one KV head into `16 * Q_GROUPS` MMA rows so every KV
/// token is loaded from DRAM once across all GQA query heads and verify rows, while 4 warps split `BKV = 32` keys in
/// `Q * K^T` and split `HEAD_DIM = 128` (32 cols/warp) in `P * V`. Zero-copy provisional sliding-window deltas read
/// committed keys from `keys/values` and provisional rows from `delta_keys/delta_values` without copying the KV ring.
///
/// SAFETY (caller contract, `resident_launch_flash_window_split16` / `resident_launch_flash_window_split32`):
/// `table` and `index` point to an initialized 16-byte aligned `Command` of matching kind recorded by
/// `Backend::flash_window_split_into` after verifying all operand bounds, alignments, non-overlap, and grid geometry.
#[inline(always)]
pub(crate) unsafe fn replay_split<const Q_GROUPS: usize, const SP_SMEM_BYTES: usize>(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized read-only Command; wrong kind traps before reading payload words.
    // - Device re-checks `flash_window_split_admitted`, grid dimensions, optional delta alignment, and per-row
    //   `live <= view` and `live <= pos + 1`, trapping on any malformed command or step.
    // - Shared memory stays strictly inside the static `SP_SMEM_BYTES` buffer (`26,496` B for `Q_GROUPS=1`,
    //   `33,536` B for `Q_GROUPS=2`). Every shared sub-region (`Q`, `K`, `V`, `P`, `M`, `T`, `RED_M`, `RED_L`,
    //   `KLO`, `KHI`) is disjoint; `RED_M` and `RED_L` are separate so cross-warp max and sum reductions never race.
    // - Global KV reads are guarded by `col >= k_min && col <= k_max`: committed ring rows index `col % view < view`,
    //   straight planes index `col < view`, and provisional delta rows index `col - first_pos < rows`.
    // - Partial output writes index `((row * heads + head) * num_splits + s_local)` within host-checked `partials`.
    // - All 128 threads participate in every `_syncthreads()`, `ldmatrix`, `mma`, and `shfl_xor`.
    unsafe {
        let c = &*table.add(index as usize);
        let expect_kind = if Q_GROUPS == 1 {
            crate::inkling_launch::FLASH_WINDOW_SPLIT16
        } else {
            crate::inkling_launch::FLASH_WINDOW_SPLIT32
        };
        if c.kind != expect_kind { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let query = *w as *const f32;
        let mut keys = *w.add(1) as *const u32;
        let mut values = *w.add(2) as *const u32;
        let moved = *w.add(3) as *const f32;
        let rel_table = *w.add(4) as *const f32;
        let delta_keys = *w.add(5) as *const u32;
        let delta_values = *w.add(6) as *const u32;
        let steps = *w.add(7) as *const f32;
        let partials = *w.add(8) as *mut f32;
        let rows = *w.add(9) as usize;
        let heads = *w.add(10) as usize;
        let kv_heads = *w.add(11) as usize;
        let view = *w.add(12) as usize;
        let extent = *w.add(13) as usize;
        let scale = f32::from_bits(*w.add(14) as u32);
        let first_pos = *w.add(15) as usize;
        let flags = *w.add(FW_SP_FLAGS_WORD);
        let split_kv = (flags & 0xffff_ffff) as usize;
        let num_splits = ((flags >> 32) & 0x7fff_ffff) as usize;
        let is_ring = (flags >> 63) != 0;
        let query_scale = *w.add(17) as *const f32;
        let moved_scale = *w.add(18) as *const f32;
        let position = *w.add(19) as *const f32;
        let query_bound = f32::from_bits(*w.add(20) as u32);
        let moved_bound = f32::from_bits(*w.add(21) as u32);
        let split_start_global = *w.add(FW_SP_SPLIT_START_WORD) as usize;
        let plane_count = *w.add(FW_SP_PLANE_COUNT_WORD) as usize;
        let rows_per_cta = *w.add(FW_SP_ROWS_PER_CTA_WORD) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let bx = core::arch::nvptx::_block_idx_x() as usize;
        let s_local = core::arch::nvptx::_block_idx_y() as usize;

        if !flash_window_split_admitted(
            rows, heads, kv_heads, view, extent, split_kv, num_splits, Q_GROUPS, rows_per_cta, plane_count,
            query as usize, keys as usize, values as usize, moved as usize, partials as usize, threads,
        ) {
            core::arch::nvptx::trap();
        }
        let has_delta = !delta_keys.is_null();
        if has_delta != !delta_values.is_null() || (has_delta && (((delta_keys as usize) | (delta_values as usize)) & 15 != 0 || plane_count != 0)) {
            core::arch::nvptx::trap();
        }
        let group = heads / kv_heads;
        let q_ctas = rows.div_ceil(rows_per_cta);
        if core::arch::nvptx::_grid_dim_x() as usize != q_ctas * kv_heads || (core::arch::nvptx::_grid_dim_y() as usize) < num_splits {
            core::arch::nvptx::trap();
        }
        if s_local >= num_splits {
            return;
        }
        let q_cta = bx / kv_heads;
        let kv_head = bx % kv_heads;
        let r_base = q_cta * rows_per_cta;
        let m_cta = if rows - r_base > rows_per_cta { rows_per_cta } else { rows - r_base };
        let q_cnt = m_cta * group;

        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);

        let o_total_words = rows * heads * num_splits * FW_HEAD_DIM;
        let ml_base = partials.add(o_total_words);
        let write_empty = || {
            core::arch::asm!("griddepcontrol.wait;", options(nostack));
            if tid < q_cnt {
                let row = r_base + tid / group;
                let head = kv_head * group + tid % group;
                let slot_idx = (row * heads + head) * num_splits + s_local;
                *ml_base.add(slot_idx * 2) = f32::NEG_INFINITY;
                *ml_base.add(slot_idx * 2 + 1) = 0.0f32;
            }
        };

        // Shared memory layout (all 16-byte aligned, strictly disjoint).
        let qt: usize = 16 * Q_GROUPS;
        const SP_T_PITCH: usize = 48;
        let sp_q_bytes: usize = qt * FW_HEAD_DIM * 2;
        const SP_K_BYTES: usize = FW_SPLIT_BKV * FW_HEAD_DIM * 2;
        const SP_V_BYTES: usize = FW_SPLIT_BKV * FW_HEAD_DIM * 2;
        let sp_p_bytes: usize = qt * FW_SPLIT_BKV * 2;
        let sp_m_bytes: usize = qt * M_PITCH * 4;
        const SP_T_BYTES: usize = FW_REL_DIM * SP_T_PITCH * 4;
        let sp_red_bytes: usize = 4 * qt * 4;
        let sp_meta_bytes: usize = 2 * qt * 4;
        if sp_q_bytes + SP_K_BYTES + SP_V_BYTES + sp_p_bytes + sp_m_bytes + SP_T_BYTES + 2 * sp_red_bytes + sp_meta_bytes != SP_SMEM_BYTES {
            core::arch::nvptx::trap();
        }
        let smem = crate::ptx::shared::<SP_SMEM_BYTES>();
        let q_base = smem;
        let k_base = q_base + sp_q_bytes as u32;
        let v_base = k_base + SP_K_BYTES as u32;
        let p_base = v_base + SP_V_BYTES as u32;
        let m_base = p_base + sp_p_bytes as u32;
        let t_smem = m_base + sp_m_bytes as u32;
        let red_m = t_smem + SP_T_BYTES as u32;
        let red_l = red_m + sp_red_bytes as u32;
        let klo_smem = red_l + sp_red_bytes as u32;
        let khi_smem = klo_smem + (qt * 4) as u32;

        // Resolve per-CTA KV planes, key interval [k_min, k_max], and split key bounds [first_t, last_t).
        let (pos_cta0, k_min, k_max, has_any_live, s_abs) = if plane_count != 0 {
            let chat = r_base;
            keys = *w.add(FW_SP_PLANES_AT + 2 * chat) as *const u32;
            values = *w.add(FW_SP_PLANES_AT + 2 * chat + 1) as *const u32;
            if ((keys as usize) | (values as usize)) & 15 != 0 { core::arch::nvptx::trap(); }
            let pos = *w.add(FW_SP_POS_AT + chat) as usize;
            let live = *w.add(FW_SP_LIVE_AT + chat) as usize;
            if live > view || live > pos + 1 || (!is_ring && live > 0 && pos >= view) { core::arch::nvptx::trap(); }
            if live == 0 {
                write_empty();
                return;
            }
            let klo = pos + 1 - live;
            let s_start = klo / split_kv;
            if tid < qt {
                if tid < q_cnt {
                    store_u32(klo_smem + (tid * 4) as u32, klo as u32);
                    store_u32(khi_smem + (tid * 4) as u32, pos as u32);
                } else {
                    store_u32(klo_smem + (tid * 4) as u32, 1);
                    store_u32(khi_smem + (tid * 4) as u32, 0);
                }
            }
            (pos, klo, pos, true, s_start + s_local)
        } else {
            let mut kmin = usize::MAX;
            let mut kmax = 0usize;
            let mut any = false;
            let mut r_loc = 0usize;
            while r_loc < m_cta {
                let row = r_base + r_loc;
                let lv_f = *steps.add(2 * row);
                let live = lv_f as usize;
                let pos = first_pos + row;
                if live as f32 != lv_f || live > view || live > pos + 1 || (!is_ring && !has_delta && live > 0 && pos >= view) {
                    core::arch::nvptx::trap();
                }
                if live > 0 {
                    let klo = pos + 1 - live;
                    if klo < kmin { kmin = klo; }
                    if pos > kmax { kmax = pos; }
                    any = true;
                }
                r_loc += 1;
            }
            if !any {
                write_empty();
                return;
            }
            if tid < qt {
                if tid < q_cnt {
                    let r_loc = tid / group;
                    let row = r_base + r_loc;
                    let live = (*steps.add(2 * row)) as usize;
                    let pos = first_pos + row;
                    if live > 0 {
                        store_u32(klo_smem + (tid * 4) as u32, (pos + 1 - live) as u32);
                        store_u32(khi_smem + (tid * 4) as u32, pos as u32);
                    } else {
                        store_u32(klo_smem + (tid * 4) as u32, 1);
                        store_u32(khi_smem + (tid * 4) as u32, 0);
                    }
                } else {
                    store_u32(klo_smem + (tid * 4) as u32, 1);
                    store_u32(khi_smem + (tid * 4) as u32, 0);
                }
            }
            (first_pos + r_base, kmin, kmax, true, split_start_global + s_local)
        };

        let split_k0 = s_abs * split_kv;
        let split_k1 = split_k0 + split_kv;
        if !has_any_live || split_k0 > k_max || split_k1 <= k_min {
            write_empty();
            return;
        }
        let first_t = if k_min > split_k0 {
            split_k0 + ((k_min - split_k0) / FW_SPLIT_BKV) * FW_SPLIT_BKV
        } else {
            split_k0
        };
        let end_align = ((k_max / FW_SPLIT_BKV) + 1) * FW_SPLIT_BKV;
        let last_t = if end_align < split_k1 { end_align } else { split_k1 };
        if first_t >= last_t {
            write_empty();
            return;
        }

        // Fused HeadClip of Q[0..q_cnt] and M[0..q_cnt] into Q_smem and M_smem (8 lanes/query, 16 queries/pass,
        // bit-exact with `fast_head_clip_into` and `flash_window` FW_CLIP_IN).
        {
            let (grp8, l8) = (lane / 8, lane % 8);
            macro_rules! group_zero {
                ($sum:expr) => {{
                    let mut sum: f32 = $sum;
                    sum += shfl_xor_f32(sum, 4);
                    sum += shfl_xor_f32(sum, 2);
                    sum += shfl_xor_f32(sum, 1);
                    let bits: u32;
                    core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, {src}, 31, 0xffffffff;", got = out(reg32) bits, v = in(reg32) sum.to_bits(), src = in(reg32) (lane & 24) as u32);
                    f32::from_bits(bits)
                }};
            }
            let mut pass = 0usize;
            while pass < Q_GROUPS {
                let qi = pass * 16 + grp8 * 4 + warp;
                let live = qi < q_cnt;
                let (row, head) = if live { (r_base + qi / group, kv_head * group + qi % group) } else { (0, 0) };
                let factor = if position.is_null() || !live { 0.0f32 } else { *position.add(row) };
                let finish = |v: f32, sc: f32| {
                    let clipped = rn(v * sc);
                    if position.is_null() { clipped } else { rn(clipped * factor) }
                };
                let x = query.add((row * heads + head) * FW_HEAD_DIM);
                let value = |j: usize| {
                    let v = rn(*x.add(j));
                    if query_scale.is_null() { v } else { v * *query_scale.add(head * FW_HEAD_DIM + j) }
                };
                let v: [f32; 16] = if live {
                    [
                        value(l8), value(l8 + 8), value(l8 + 16), value(l8 + 24),
                        value(l8 + 32), value(l8 + 40), value(l8 + 48), value(l8 + 56),
                        value(l8 + 64), value(l8 + 72), value(l8 + 80), value(l8 + 88),
                        value(l8 + 96), value(l8 + 104), value(l8 + 112), value(l8 + 120),
                    ]
                } else { [0.0f32; 16] };
                macro_rules! chain {
                    ($k:literal) => {{
                        let mut sum = 0.0f32;
                        sum += v[$k] * v[$k];
                        sum += v[$k + 4] * v[$k + 4];
                        sum += v[$k + 8] * v[$k + 8];
                        sum += v[$k + 12] * v[$k + 12];
                        sum
                    }};
                }
                let (low, high) = (chain!(0) + chain!(2), chain!(1) + chain!(3));
                let qsc = crate::block_attention::clip_factor(crate::block_ops::root(group_zero!(low + high)), query_bound);
                macro_rules! put_q {
                    ($($m:literal)*) => {$(
                        let word = if live { finish(v[$m], qsc).to_bits() } else { 0 };
                        store_u16(q_base + swiz128(qi, $m) + (2 * l8) as u32, (word >> 16) as u16);
                    )*};
                }
                put_q!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
                let mrow = moved.add((row * heads + head) * FW_REL_DIM);
                let mvalue = |j: usize| {
                    let v = rn(*mrow.add(j));
                    if moved_scale.is_null() { v } else { v * *moved_scale.add(head * FW_REL_DIM + j) }
                };
                let (mv0, mv1) = if live { (mvalue(l8), mvalue(l8 + 8)) } else { (0.0f32, 0.0f32) };
                let (mut s0, mut s1) = (0.0f32, 0.0f32);
                s0 += mv0 * mv0;
                s1 += mv1 * mv1;
                let msc = crate::block_attention::clip_factor(crate::block_ops::root(group_zero!((s0 + 0.0f32) + (s1 + 0.0f32))), moved_bound);
                let (w0, w1) = if live { (rn(finish(mv0, msc)), rn(finish(mv1, msc))) } else { (0.0, 0.0) };
                store_u32(m_base + ((qi * M_PITCH + l8) * 4) as u32, w0.to_bits());
                store_u32(m_base + ((qi * M_PITCH + l8 + 8) * 4) as u32, w1.to_bits());
                pass += 1;
            }
        }
        // Wait for any PDL-overlapping KV finish (`FastRowDots` `KV_FINISH_TAG`) before reading `keys`/`values`.
        core::arch::asm!("griddepcontrol.wait;", options(nostack));

        let view_pow2 = (view & (view - 1)) == 0;
        let view_u32 = view as u32;
        let view_mask = view_u32 - 1;
        let k_min_u32 = k_min as u32;
        let k_max_u32 = k_max as u32;
        let first_pos_u32 = first_pos as u32;
        let pos_cta0_u32 = pos_cta0 as u32;
        let extent_u32 = extent as u32;
        let m_cta_u32 = m_cta as u32;
        let kv_heads_u32 = kv_heads as u32;
        let kv_head_u32 = kv_head as u32;
        let tid_u32 = tid as u32;
        macro_rules! issue_kv {
            ($dst_base:expr, $plane:expr, $delta:expr, $tb:expr) => {{
                let tb_u: u32 = $tb;
                let mut ci = 0u32;
                while ci < 4 {
                    let idx = ci * (FW_THREADS as u32) + tid_u32;
                    let r = idx >> 4;
                    let ch = idx & 15;
                    let dst = $dst_base + ((r << 8) + ((ch ^ (r & 7)) << 4));
                    let col = tb_u + r;
                    if col >= k_min_u32 && col <= k_max_u32 {
                        if has_delta && col >= first_pos_u32 {
                            let row_off = ((col - first_pos_u32) * kv_heads_u32 + kv_head_u32) as usize * 64 + (ch * 4) as usize;
                            crate::ptx::copy16(dst, $delta.add(row_off));
                        } else {
                            let phys = if is_ring {
                                if view_pow2 { col & view_mask } else { col % view_u32 }
                            } else {
                                col
                            };
                            let row_off = (phys * kv_heads_u32 + kv_head_u32) as usize * 64 + (ch * 4) as usize;
                            crate::ptx::copy16(dst, $plane.add(row_off));
                        }
                    } else {
                        st_shared_zero16(dst);
                    }
                    ci += 1;
                }
            }};
        }

        macro_rules! issue_rel {
            ($tb:expr) => {{
                let tb_u: u32 = $tb;
                if tb_u + (FW_SPLIT_BKV as u32) + extent_u32 > pos_cta0_u32 && tb_u < pos_cta0_u32 + m_cta_u32 {
                    let d_first = pos_cta0_u32 as i32 - (tb_u + (FW_SPLIT_BKV as u32) - 1) as i32;
                    let i_lim = m_cta_u32 + (FW_SPLIT_BKV as u32) - 1;
                    let mut k = tid_u32;
                    while k < (FW_REL_DIM * SP_T_PITCH) as u32 {
                        let (j, i) = (k / (SP_T_PITCH as u32), k % (SP_T_PITCH as u32));
                        let d = d_first + i as i32;
                        let dst = t_smem + (k << 2);
                        if d >= 0 && (d as u32) < extent_u32 && i < i_lim {
                            crate::ptx::copy4(dst, rel_table.add((j * extent_u32 + d as u32) as usize).cast::<u32>());
                        } else {
                            store_u32(dst, 0);
                        }
                        k += FW_THREADS as u32;
                    }
                }
            }};
        }

        // Prologue: issue K[first_t]+T[first_t] (group 0) AND V[first_t] (group 1) together, wait for group 0 only
        // so V[first_t] stays in flight overlapped with Q*K^T + bias + softmax!
        let first_t_u32 = first_t as u32;
        let last_t_u32 = last_t as u32;
        issue_kv!(k_base, keys, delta_keys, first_t_u32);
        issue_rel!(first_t_u32);
        crate::ptx::commit();
        issue_kv!(v_base, values, delta_values, first_t_u32);
        crate::ptx::commit();
        cp_async_wait_one();
        core::arch::nvptx::_syncthreads();

        // Per-thread query rows in 32-bit registers to keep register pressure low (3 blocks/SM).
        let r0 = g as u32;
        let r1 = r0 + 8;
        let r2 = 16 + r0;
        let r3 = 24 + r0;
        let klo0 = load_u32(klo_smem + (r0 << 2));
        let khi0 = load_u32(khi_smem + (r0 << 2));
        let klo1 = load_u32(klo_smem + (r1 << 2));
        let khi1 = load_u32(khi_smem + (r1 << 2));
        let (klo2, khi2, klo3, khi3) = if Q_GROUPS == 2 {
            (
                load_u32(klo_smem + (r2 << 2)),
                load_u32(khi_smem + (r2 << 2)),
                load_u32(klo_smem + (r3 << 2)),
                load_u32(khi_smem + (r3 << 2)),
            )
        } else { (1u32, 0u32, 1u32, 0u32) };
        let q_cnt_u32 = q_cnt as u32;
        let group_u32 = group as u32;
        let warp_u32 = warp as u32;
        let t_u32 = t as u32;
        let qt_u32 = qt as u32;
        let rloc0 = if r0 < q_cnt_u32 { r0 / group_u32 } else { 0 };
        let rloc1 = if r1 < q_cnt_u32 { r1 / group_u32 } else { 0 };
        let rloc2 = if Q_GROUPS == 2 && r2 < q_cnt_u32 { r2 / group_u32 } else { 0 };
        let rloc3 = if Q_GROUPS == 2 && r3 < q_cnt_u32 { r3 / group_u32 } else { 0 };
        let col_off = 8 * warp_u32 + 2 * t_u32;
        let rel_i0 = rloc0 + (FW_SPLIT_BKV as u32 - 1) - col_off;
        let rel_i1 = rloc1 + (FW_SPLIT_BKV as u32 - 1) - col_off;
        let rel_i2 = rloc2 + (FW_SPLIT_BKV as u32 - 1) - col_off;
        let rel_i3 = rloc3 + (FW_SPLIT_BKV as u32 - 1) - col_off;

        // Online softmax state for this split alone (sink is folded once in FLASH_WINDOW_REDUCE).
        let (mut m_r0, mut m_r1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (mut l_r0, mut l_r1) = (0.0f32, 0.0f32);
        let (mut m_r2, mut m_r3) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (mut l_r2, mut l_r3) = (0.0f32, 0.0f32);

        // Output accumulators for this warp's 32 columns (`d = 32*warp .. 32*warp + 32`).
        let (mut o00, mut o01, mut o02, mut o03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o10, mut o11, mut o12, mut o13) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        let q_ld_r = lane & 15;
        let q_ld_c = lane >> 4;
        let k_ld_r = 8 * warp + (lane & 7);
        let k_ld_c = (lane >> 3) & 1;
        let v_ld_r0 = lane & 15;
        let v_ld_r1 = 16 + v_ld_r0;
        let v_ld_sub = lane >> 4;

        let mut t_base = first_t_u32;
        while t_base < last_t_u32 {
            let next_t = t_base + FW_SPLIT_BKV as u32;

            // 1. Q * K^T: warp w computes its 8 keys (`t_base + 8*w .. +8`) over ascending d = 0..128.
            let mut s0 = [0.0f32; 4];
            let mut s1 = [0.0f32; 4];
            let mut ks = 0usize;
            while ks < 8 {
                let kb = ldmatrix_x2(k_base + swiz128(k_ld_r, ks * 2 + k_ld_c));
                let qa0 = ldmatrix_x4(q_base + swiz128(q_ld_r, ks * 2 + q_ld_c));
                s0 = mma_m16n8k16_bf16(qa0, kb, s0);
                if Q_GROUPS == 2 {
                    let qa1 = ldmatrix_x4(q_base + swiz128(16 + q_ld_r, ks * 2 + q_ld_c));
                    s1 = mma_m16n8k16_bf16(qa1, kb, s1);
                }
                ks += 1;
            }

            // 2. Relative bias + score scale + causal/window mask.
            let need_rel = t_base + (FW_SPLIT_BKV as u32) + extent_u32 > pos_cta0_u32 && t_base < pos_cta0_u32 + m_cta_u32;
            let mut b0 = [0.0f32; 4];
            let mut b1 = [0.0f32; 4];
            if need_rel {
                let mrow0 = m_base + r0 * (M_PITCH as u32 * 4);
                let mrow1 = m_base + r1 * (M_PITCH as u32 * 4);
                let mrow2 = if Q_GROUPS == 2 { m_base + r2 * (M_PITCH as u32 * 4) } else { 0 };
                let mrow3 = if Q_GROUPS == 2 { m_base + r3 * (M_PITCH as u32 * 4) } else { 0 };
                let mut j = 0u32;
                while j < FW_REL_DIM as u32 {
                    let trow = t_smem + j * (SP_T_PITCH as u32 * 4);
                    let a0 = f32::from_bits(load_u32(mrow0 + (j << 2)));
                    let a1 = f32::from_bits(load_u32(mrow1 + (j << 2)));
                    let t00 = f32::from_bits(load_u32(trow + (rel_i0 << 2)));
                    let t01 = f32::from_bits(load_u32(trow + ((rel_i0 - 1) << 2)));
                    let t10 = f32::from_bits(load_u32(trow + (rel_i1 << 2)));
                    let t11 = f32::from_bits(load_u32(trow + ((rel_i1 - 1) << 2)));
                    b0[0] += a0 * t00; b0[1] += a0 * t01; b0[2] += a1 * t10; b0[3] += a1 * t11;
                    if Q_GROUPS == 2 {
                        let a2 = f32::from_bits(load_u32(mrow2 + (j << 2)));
                        let a3 = f32::from_bits(load_u32(mrow3 + (j << 2)));
                        let t20 = f32::from_bits(load_u32(trow + (rel_i2 << 2)));
                        let t21 = f32::from_bits(load_u32(trow + ((rel_i2 - 1) << 2)));
                        let t30 = f32::from_bits(load_u32(trow + (rel_i3 << 2)));
                        let t31 = f32::from_bits(load_u32(trow + ((rel_i3 - 1) << 2)));
                        b1[0] += a2 * t20; b1[1] += a2 * t21; b1[2] += a3 * t30; b1[3] += a3 * t31;
                    }
                    j += 1;
                }
            }
            let c0 = t_base + col_off;
            let c1 = c0 + 1;
            macro_rules! eval_sc {
                ($val:expr, $klo:expr, $khi:expr, $col:expr, $bsum:expr) => {{
                    let col: u32 = $col;
                    if col < $klo || col > $khi {
                        f32::NEG_INFINITY
                    } else {
                        let dist = $khi - col;
                        let scaled = rn(rn($val) * scale);
                        if need_rel && dist < extent_u32 { rn(scaled + rn($bsum)) } else { scaled }
                    }
                }};
            }
            s0[0] = eval_sc!(s0[0], klo0, khi0, c0, b0[0]);
            s0[1] = eval_sc!(s0[1], klo0, khi0, c1, b0[1]);
            s0[2] = eval_sc!(s0[2], klo1, khi1, c0, b0[2]);
            s0[3] = eval_sc!(s0[3], klo1, khi1, c1, b0[3]);
            if Q_GROUPS == 2 {
                s1[0] = eval_sc!(s1[0], klo2, khi2, c0, b1[0]);
                s1[1] = eval_sc!(s1[1], klo2, khi2, c1, b1[1]);
                s1[2] = eval_sc!(s1[2], klo3, khi3, c0, b1[2]);
                s1[3] = eval_sc!(s1[3], klo3, khi3, c1, b1[3]);
            }

            // 3. Cross-warp deterministic max reduction across all 32 keys.
            macro_rules! warp_max {
                ($a:expr, $b:expr) => {{
                    let mut m = fmax($a, $b);
                    m = fmax(m, shfl_xor_f32(m, 1));
                    m = fmax(m, shfl_xor_f32(m, 2));
                    m
                }};
            }
            let mw0 = warp_max!(s0[0], s0[1]);
            let mw1 = warp_max!(s0[2], s0[3]);
            if t_u32 == 0 {
                store_u32(red_m + ((warp_u32 * qt_u32 + r0) << 2), mw0.to_bits());
                store_u32(red_m + ((warp_u32 * qt_u32 + r1) << 2), mw1.to_bits());
            }
            if Q_GROUPS == 2 {
                let mw2 = warp_max!(s1[0], s1[1]);
                let mw3 = warp_max!(s1[2], s1[3]);
                if t_u32 == 0 {
                    store_u32(red_m + ((warp_u32 * qt_u32 + r2) << 2), mw2.to_bits());
                    store_u32(red_m + ((warp_u32 * qt_u32 + r3) << 2), mw3.to_bits());
                }
            }
            // Barrier #1: all warps finished reading K[t_base] & T[t_base] and writing red_m.
            core::arch::nvptx::_syncthreads();

            // Immediately issue K[next_t] + T[next_t] BEFORE softmax exp + P*V (double-overlapped with V[t_base]!).
            if next_t < last_t_u32 {
                issue_kv!(k_base, keys, delta_keys, next_t);
                issue_rel!(next_t);
                crate::ptx::commit();
            }

            macro_rules! tile_max {
                ($r:expr) => {{
                    let a = f32::from_bits(load_u32(red_m + ($r << 2)));
                    let b = f32::from_bits(load_u32(red_m + ((qt_u32 + $r) << 2)));
                    let c = f32::from_bits(load_u32(red_m + ((2 * qt_u32 + $r) << 2)));
                    let d = f32::from_bits(load_u32(red_m + ((3 * qt_u32 + $r) << 2)));
                    fmax(fmax(a, b), fmax(c, d))
                }};
            }
            macro_rules! to_p {
                ($v:expr, $mn:expr) => {{
                    let v: f32 = $v;
                    if v == f32::NEG_INFINITY { 0.0f32 } else { exp(v - $mn) }
                }};
            }
            let mt0 = tile_max!(r0);
            let mt1 = tile_max!(r1);
            let mn0 = if mt0 == f32::NEG_INFINITY { m_r0 } else { fmax(m_r0, mt0) };
            let mn1 = if mt1 == f32::NEG_INFINITY { m_r1 } else { fmax(m_r1, mt1) };
            let p00 = to_p!(s0[0], mn0); let p01 = to_p!(s0[1], mn0);
            let p02 = to_p!(s0[2], mn1); let p03 = to_p!(s0[3], mn1);
            store_u32(p_base + swiz64(r0 as usize, warp) + (t_u32 << 2), cvt_bf16x2(p00, p01));
            store_u32(p_base + swiz64(r1 as usize, warp) + (t_u32 << 2), cvt_bf16x2(p02, p03));
            let mut lw0 = p00 + p01; lw0 += shfl_xor_f32(lw0, 1); lw0 += shfl_xor_f32(lw0, 2);
            let mut lw1 = p02 + p03; lw1 += shfl_xor_f32(lw1, 1); lw1 += shfl_xor_f32(lw1, 2);
            if t_u32 == 0 {
                store_u32(red_l + ((warp_u32 * qt_u32 + r0) << 2), lw0.to_bits());
                store_u32(red_l + ((warp_u32 * qt_u32 + r1) << 2), lw1.to_bits());
            }
            let (mut mt2, mut mt3, mut mn2, mut mn3) = (f32::NEG_INFINITY, f32::NEG_INFINITY, m_r2, m_r3);
            if Q_GROUPS == 2 {
                mt2 = tile_max!(r2);
                mt3 = tile_max!(r3);
                mn2 = if mt2 == f32::NEG_INFINITY { m_r2 } else { fmax(m_r2, mt2) };
                mn3 = if mt3 == f32::NEG_INFINITY { m_r3 } else { fmax(m_r3, mt3) };
                let p10 = to_p!(s1[0], mn2); let p11 = to_p!(s1[1], mn2);
                let p12 = to_p!(s1[2], mn3); let p13 = to_p!(s1[3], mn3);
                store_u32(p_base + swiz64(r2 as usize, warp) + (t_u32 << 2), cvt_bf16x2(p10, p11));
                store_u32(p_base + swiz64(r3 as usize, warp) + (t_u32 << 2), cvt_bf16x2(p12, p13));
                let mut lw2 = p10 + p11; lw2 += shfl_xor_f32(lw2, 1); lw2 += shfl_xor_f32(lw2, 2);
                let mut lw3 = p12 + p13; lw3 += shfl_xor_f32(lw3, 1); lw3 += shfl_xor_f32(lw3, 2);
                if t_u32 == 0 {
                    store_u32(red_l + ((warp_u32 * qt_u32 + r2) << 2), lw2.to_bits());
                    store_u32(red_l + ((warp_u32 * qt_u32 + r3) << 2), lw3.to_bits());
                }
            }

            // Wait for V[t_base] only (keep K[next_t]+T[next_t] in flight!) and sync P_smem + red_l.
            if next_t < last_t_u32 { cp_async_wait_one(); } else { cp_async_wait_zero(); }
            core::arch::nvptx::_syncthreads();

            macro_rules! tile_sum {
                ($r:expr) => {{
                    let a = f32::from_bits(load_u32(red_l + ($r << 2)));
                    let b = f32::from_bits(load_u32(red_l + ((qt_u32 + $r) << 2)));
                    let c = f32::from_bits(load_u32(red_l + ((2 * qt_u32 + $r) << 2)));
                    let d = f32::from_bits(load_u32(red_l + ((3 * qt_u32 + $r) << 2)));
                    (a + b) + (c + d)
                }};
            }
            macro_rules! update_row_lo {
                ($mt:expr, $mn:expr, $mr:ident, $lr:ident, $r:expr, $oa:ident, $ob:ident, $oc:ident, $od:ident) => {
                    if $mt != f32::NEG_INFINITY {
                        let lt = tile_sum!($r);
                        if $mr == f32::NEG_INFINITY {
                            $lr = lt;
                            $oa[0] = 0.0; $oa[1] = 0.0; $ob[0] = 0.0; $ob[1] = 0.0;
                            $oc[0] = 0.0; $oc[1] = 0.0; $od[0] = 0.0; $od[1] = 0.0;
                        } else if $mn > $mr {
                            let al = exp($mr - $mn);
                            $lr = $lr * al + lt;
                            $oa[0] *= al; $oa[1] *= al; $ob[0] *= al; $ob[1] *= al;
                            $oc[0] *= al; $oc[1] *= al; $od[0] *= al; $od[1] *= al;
                        } else {
                            $lr += lt;
                        }
                        $mr = $mn;
                    }
                };
            }
            macro_rules! update_row_hi {
                ($mt:expr, $mn:expr, $mr:ident, $lr:ident, $r:expr, $oa:ident, $ob:ident, $oc:ident, $od:ident) => {
                    if $mt != f32::NEG_INFINITY {
                        let lt = tile_sum!($r);
                        if $mr == f32::NEG_INFINITY {
                            $lr = lt;
                            $oa[2] = 0.0; $oa[3] = 0.0; $ob[2] = 0.0; $ob[3] = 0.0;
                            $oc[2] = 0.0; $oc[3] = 0.0; $od[2] = 0.0; $od[3] = 0.0;
                        } else if $mn > $mr {
                            let al = exp($mr - $mn);
                            $lr = $lr * al + lt;
                            $oa[2] *= al; $oa[3] *= al; $ob[2] *= al; $ob[3] *= al;
                            $oc[2] *= al; $oc[3] *= al; $od[2] *= al; $od[3] *= al;
                        } else {
                            $lr += lt;
                        }
                        $mr = $mn;
                    }
                };
            }
            update_row_lo!(mt0, mn0, m_r0, l_r0, r0, o00, o01, o02, o03);
            update_row_hi!(mt1, mn1, m_r1, l_r1, r1, o00, o01, o02, o03);
            if Q_GROUPS == 2 {
                update_row_lo!(mt2, mn2, m_r2, l_r2, r2, o10, o11, o12, o13);
                update_row_hi!(mt3, mn3, m_r3, l_r3, r3, o10, o11, o12, o13);
            }

            // 4. O += P * V for warp's 32 columns (`d = 32*warp .. +32`), kt = 0 (keys 0..16) then kt = 1 (keys 16..32).
            let vb0_a = ldmatrix_x4_trans(v_base + swiz128(v_ld_r0, 4 * warp + v_ld_sub));
            let vb0_b = ldmatrix_x4_trans(v_base + swiz128(v_ld_r0, 4 * warp + 2 + v_ld_sub));
            let pa0_k0 = ldmatrix_x4(p_base + swiz64(q_ld_r, q_ld_c));
            o00 = mma_m16n8k16_bf16(pa0_k0, [vb0_a[0], vb0_a[1]], o00);
            o01 = mma_m16n8k16_bf16(pa0_k0, [vb0_a[2], vb0_a[3]], o01);
            o02 = mma_m16n8k16_bf16(pa0_k0, [vb0_b[0], vb0_b[1]], o02);
            o03 = mma_m16n8k16_bf16(pa0_k0, [vb0_b[2], vb0_b[3]], o03);
            if Q_GROUPS == 2 {
                let pa1_k0 = ldmatrix_x4(p_base + swiz64(16 + q_ld_r, q_ld_c));
                o10 = mma_m16n8k16_bf16(pa1_k0, [vb0_a[0], vb0_a[1]], o10);
                o11 = mma_m16n8k16_bf16(pa1_k0, [vb0_a[2], vb0_a[3]], o11);
                o12 = mma_m16n8k16_bf16(pa1_k0, [vb0_b[0], vb0_b[1]], o12);
                o13 = mma_m16n8k16_bf16(pa1_k0, [vb0_b[2], vb0_b[3]], o13);
            }
            let vb1_a = ldmatrix_x4_trans(v_base + swiz128(v_ld_r1, 4 * warp + v_ld_sub));
            let vb1_b = ldmatrix_x4_trans(v_base + swiz128(v_ld_r1, 4 * warp + 2 + v_ld_sub));
            let pa0_k1 = ldmatrix_x4(p_base + swiz64(q_ld_r, 2 + q_ld_c));
            o00 = mma_m16n8k16_bf16(pa0_k1, [vb1_a[0], vb1_a[1]], o00);
            o01 = mma_m16n8k16_bf16(pa0_k1, [vb1_a[2], vb1_a[3]], o01);
            o02 = mma_m16n8k16_bf16(pa0_k1, [vb1_b[0], vb1_b[1]], o02);
            o03 = mma_m16n8k16_bf16(pa0_k1, [vb1_b[2], vb1_b[3]], o03);
            if Q_GROUPS == 2 {
                let pa1_k1 = ldmatrix_x4(p_base + swiz64(16 + q_ld_r, 2 + q_ld_c));
                o10 = mma_m16n8k16_bf16(pa1_k1, [vb1_a[0], vb1_a[1]], o10);
                o11 = mma_m16n8k16_bf16(pa1_k1, [vb1_a[2], vb1_a[3]], o11);
                o12 = mma_m16n8k16_bf16(pa1_k1, [vb1_b[0], vb1_b[1]], o12);
                o13 = mma_m16n8k16_bf16(pa1_k1, [vb1_b[2], vb1_b[3]], o13);
            }

            if next_t < last_t_u32 {
                // All warps finished reading V[t_base]; issue V[next_t] and wait for K[next_t]+T[next_t]!
                core::arch::nvptx::_syncthreads();
                issue_kv!(v_base, values, delta_values, next_t);
                crate::ptx::commit();
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
            }
            t_base = next_t;
        }

        // Epilogue: store unnormalized split partials O_s[d] and (m_s, l_s).
        macro_rules! store_group {
            ($ra:expr, $rb:expr, $ma:expr, $la:expr, $mb:expr, $lb:expr, $oa:ident, $ob:ident, $oc:ident, $od:ident) => {{
                let ra = $ra as usize;
                let rb = $rb as usize;
                if ra < q_cnt {
                    let row = r_base + ra / group;
                    let head = kv_head * group + ra % group;
                    let slot_idx = (row * heads + head) * num_splits + s_local;
                    let dst = partials.add(slot_idx * FW_HEAD_DIM + 32 * warp + 2 * t);
                    *dst = $oa[0]; *dst.add(1) = $oa[1];
                    *dst.add(8) = $ob[0]; *dst.add(9) = $ob[1];
                    *dst.add(16) = $oc[0]; *dst.add(17) = $oc[1];
                    *dst.add(24) = $od[0]; *dst.add(25) = $od[1];
                    if warp == 0 && t == 0 {
                        *ml_base.add(slot_idx * 2) = $ma;
                        *ml_base.add(slot_idx * 2 + 1) = $la;
                    }
                }
                if rb < q_cnt {
                    let row = r_base + rb / group;
                    let head = kv_head * group + rb % group;
                    let slot_idx = (row * heads + head) * num_splits + s_local;
                    let dst = partials.add(slot_idx * FW_HEAD_DIM + 32 * warp + 2 * t);
                    *dst = $oa[2]; *dst.add(1) = $oa[3];
                    *dst.add(8) = $ob[2]; *dst.add(9) = $ob[3];
                    *dst.add(16) = $oc[2]; *dst.add(17) = $oc[3];
                    *dst.add(24) = $od[2]; *dst.add(25) = $od[3];
                    if warp == 0 && t == 0 {
                        *ml_base.add(slot_idx * 2) = $mb;
                        *ml_base.add(slot_idx * 2 + 1) = $lb;
                    }
                }
            }};
        }
        store_group!(r0, r1, m_r0, l_r0, m_r1, l_r1, o00, o01, o02, o03);
        if Q_GROUPS == 2 {
            store_group!(r2, r3, m_r2, l_r2, m_r3, l_r3, o10, o11, o12, o13);
        }
    }
}

/// Deterministic Split-KV reduction (`FLASH_WINDOW_REDUCE`): folds `0..num_splits` in ascending order onto the
/// exact attention sink (`m = rn(sink), l = 1.0, O = 0.0`), applies `sigmoid(rn(gate))`, and writes BF16-rounded output.
///
/// SAFETY (caller contract, `resident_launch_flash_window_reduce`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `FLASH_WINDOW_REDUCE` recorded by `Backend::flash_window_split_into`.
#[inline(always)]
pub(crate) unsafe fn replay_reduce(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized read-only Command; wrong kind traps before reading payload words.
    // - Device re-checks `rows`, `heads`, `num_splits`, `threads == 128`, and `grid == rows * heads`.
    // - Every thread `d = tid < 128` reads only `partials[(block * num_splits + s) * 128 + d]`,
    //   `ml[(block * num_splits + s) * 2 .. +2]`, `sink[block]`, `gate[block * 128 + d]`, and writes
    //   `output[block * 128 + d]` within host-checked disjoint extents.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FLASH_WINDOW_REDUCE { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let partials = *w as *const f32;
        let sink = *w.add(1) as *const f32;
        let gate = *w.add(2) as *const f32;
        let output = *w.add(3) as *mut f32;
        let rows = *w.add(4) as usize;
        let heads = *w.add(5) as usize;
        let num_splits = *w.add(6) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let grid = core::arch::nvptx::_grid_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        if rows == 0 || heads == 0 || num_splits == 0 || threads != FW_THREADS || grid != rows * heads {
            core::arch::nvptx::trap();
        }
        let ml_base = partials.add(rows * heads * num_splits * FW_HEAD_DIM);
        let base_slot = block * num_splits;
        let mut m = rn(*sink.add(block));
        let mut l = 1.0f32;
        let mut acc = 0.0f32;
        let mut s = 0usize;
        while s < num_splits {
            let slot = base_slot + s;
            let m_s = *ml_base.add(slot * 2);
            let l_s = *ml_base.add(slot * 2 + 1);
            if l_s != 0.0f32 {
                let m_new = fmax(m, m_s);
                let alpha = exp(m - m_new);
                let beta = exp(m_s - m_new);
                let o_s = *partials.add(slot * FW_HEAD_DIM + tid);
                acc = acc * alpha + o_s * beta;
                l = l * alpha + l_s * beta;
                m = m_new;
            }
            s += 1;
        }
        let inv = 1.0f32 / l;
        let off = block * FW_HEAD_DIM + tid;
        *output.add(off) = rn(rn(acc * inv) * sigmoid(rn(*gate.add(off))));
    }
}
