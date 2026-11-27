//! GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention (`FLASH_WINDOW_SPLIT16 = 118`,
//! `FLASH_WINDOW_SPLIT32 = 119`) and L2-banded deterministic Split-KV reduction (`FLASH_WINDOW_REDUCE = 120`).
//!
//! - `FLASH_WINDOW_SPLIT16` (`Q_GROUPS == 1`, `25,248 B` SMEM, 4 blocks/SM): packs `<= 16` queries (`<= 9` rows) per
//!   KV head; keeps row bounds and scalars in SMEM `c_base` (0 global `steps` reads in loop), uses vectorized
//!   `ld_global_nc_v4` HeadClip, 4-term coalesced `T` staging (`ld.shared.v4`) + `fast_rel` interior ring path, and
//!   normal L2-cached `cp.async.cg.L2::256B` for ring KV (`is_ring`) with `L2::evict_first` gated to non-ring KV.
//! - `FLASH_WINDOW_SPLIT32` (`Q_GROUPS == 2`, `33,440 B` SMEM, 3 blocks/SM): warp-per-16-Q-rows packs up to `48`
//!   queries (`<= 16` verify rows at 3:1 GQA) into one single CTA with zero `P`/`red_m`/`red_l` SMEM, 2 barriers/32 keys,
//!   single-`kb` ping-pong + BF16x2-packed `ps0..ps3`/`pf0..pf1` around `RelBias` (`170` regs), SMEM-cached `live[0..16]`,
//!   and bit-exact 4-group softmax sum matching `FLASH_WINDOW_SPLIT16` bit-for-bit.
//! - Both entries stream non-ring global KV (`!is_ring`) with `cp.async.cg.shared.global.L2::cache_hint.L2::256B`
//!   (`evict_first`) while keeping ring KV (`is_ring`) and `<= 16 MiB` wave-aligned partials hot in GB10's 24 MiB L2.

use crate::fixed_math::{rn, sigmoid};
use crate::flash_window_abi::{
    FW_HEAD_DIM, FW_REDUCE_FLAGS_WORD, FW_REDUCE_LOAD_STATE, FW_REDUCE_STATE_WORD,
    FW_REDUCE_STORE_STATE, FW_REL_DIM, FW_SPLIT_BKV, FW_SP_FLAGS_WORD, FW_SP_PLANE_COUNT_WORD,
    FW_SP_PLANES_AT, FW_SP_ROWS_PER_CTA_WORD, FW_SP_SPLIT_START_WORD, FW_THREADS,
    flash_window_split_admitted,
};
use crate::tc_ptx::{
    cp_async_wait_one, cp_async_wait_zero, cvt_bf16x2, l2_evict_first, ld_global_nc_v4,
    ldmatrix_x2, ldmatrix_x4, ldmatrix_x4_trans, load_u32, mma_m16n8k16_bf16, shfl_xor_f32,
    st_shared_zero16, store_u32,
};

const SP16_T_SLOTS: usize = 40;
const SP16_T_PLANE_BYTES: usize = SP16_T_SLOTS * 16;
const SP48_T_SLOTS: usize = 48;
const SP48_T_PLANE_BYTES: usize = SP48_T_SLOTS * 16;
// 96 B CTA-uniform scalars & pointers + 64 B `live[0..16]` (16B-aligned, zero global `steps` reads in loop).
const C_BYTES: usize = 160;
const SP16_SMEM_BYTES: usize = 16 * FW_HEAD_DIM * 2
    + FW_SPLIT_BKV * FW_HEAD_DIM * 2
    + FW_SPLIT_BKV * FW_HEAD_DIM * 2
    + 16 * FW_SPLIT_BKV * 2
    + 16 * FW_REL_DIM * 2
    + 4 * SP16_T_PLANE_BYTES
    + 2 * (4 * 16 * 4)
    + C_BYTES;
const SP48_SMEM_BYTES: usize = 48 * FW_HEAD_DIM * 2
    + FW_SPLIT_BKV * FW_HEAD_DIM * 2
    + FW_SPLIT_BKV * FW_HEAD_DIM * 2
    + 48 * FW_REL_DIM * 2
    + 4 * SP48_T_PLANE_BYTES
    + C_BYTES;
const _: () = assert!(SP16_SMEM_BYTES == 25_248 && SP16_SMEM_BYTES * 4 <= 101_376);
const _: () = assert!(SP48_SMEM_BYTES == 33_440 && SP48_SMEM_BYTES * 3 <= 101_376);
const _: () = assert!(core::f32::consts::LOG2_E.to_bits() == 0x3fb8_aa3b && 12_582_912.0f32.to_bits() == 0x4b40_0000);

/// Zero-bank-conflict XOR-swizzled byte offset within a 256-byte-stride BF16 tile (`row`, 16B `chunk` in 0..16).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 256 + ((chunk ^ (row & 7)) * 16)) as u32
}

/// Zero-bank-conflict XOR-swizzled byte offset within a 64-byte-stride BF16 tile (`row`, 16B `chunk` in 0..4).
#[inline(always)]
const fn swiz64(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Zero-bank-conflict 8-byte XOR-swizzled byte offset within packed BF16 `M` (`row`, 4-BF16 quad `jq` in 0..4).
#[inline(always)]
const fn swiz_m8(row: usize, jq: usize) -> u32 {
    ((row * 4 + (jq ^ ((row >> 2) & 1))) * 8) as u32
}

// Pure register PTX macros expanded inside the existing `unsafe` blocks of `replay_split` / `replay_reduce`.
macro_rules! fmax_hw { ($a:expr, $b:expr) => {{ let d: f32; core::arch::asm!("max.f32 {d}, {a}, {b};", d = out(reg32) d, a = in(reg32) ($a as f32), b = in(reg32) ($b as f32), options(pure, nomem, nostack)); d }}; }
macro_rules! unpack_bf16x2 { ($p:expr) => {{ let p: u32 = $p; let lo: u32; core::arch::asm!("prmt.b32 {lo}, {p}, 0, 0x1044;", lo = out(reg32) lo, p = in(reg32) p, options(pure, nomem, nostack)); (f32::from_bits(lo), f32::from_bits(p & 0xffff_0000)) }}; }
macro_rules! rn2 { ($a:expr, $b:expr) => {{ unpack_bf16x2!(cvt_bf16x2($a, $b)) }}; }
macro_rules! pack_hi16 { ($a:expr, $b:expr) => {{ let d: u32; core::arch::asm!("prmt.b32 {d}, {a}, {b}, 0x7632;", d = out(reg32) d, a = in(reg32) ($a as f32).to_bits(), b = in(reg32) ($b as f32).to_bits(), options(pure, nomem, nostack)); d }}; }
macro_rules! st_v2_f32 { ($ptr:expr, $a:expr, $b:expr) => {{ core::arch::asm!("st.global.v2.f32 [{p}], {{{a}, {b}}};", p = in(reg64) ($ptr as *mut f32), a = in(reg32) ($a as f32), b = in(reg32) ($b as f32), options(nostack)); }}; }
macro_rules! copy16_kv_evict { ($dst:expr, $src:expr, $pol:expr) => {{ core::arch::asm!("cp.async.cg.shared.global.L2::cache_hint.L2::256B [{d}], [{s}], 16, {p};", d = in(reg32) ($dst as u32), s = in(reg64) ($src as *const u32), p = in(reg64) ($pol as u64), options(nostack)); }}; }
macro_rules! ld_smem_u64 { ($addr:expr) => {{ let v: u64; core::arch::asm!("ld.shared.u64 {v}, [{p}];", v = out(reg64) v, p = in(reg32) ($addr as u32), options(nostack)); v }}; }
// Read-only L1-cached command-word reload (non-pure asm prevents hoisting across loops without L1-bypassing `ld.volatile`).
macro_rules! ld_cmd_u64 { ($w:expr, $idx:expr) => {{ let v: u64; core::arch::asm!("ld.global.nc.u64 {v}, [{p}];", v = out(reg64) v, p = in(reg64) (($w as *const u64).add($idx)), options(nostack)); v }}; }
macro_rules! exp_nonpos {
    ($t_in:expr) => {{
        let t_raw: f32 = $t_in;
        let t = if t_raw < -87.0f32 { -87.0f32 } else { t_raw };
        let (k, n): (f32, f32);
        core::arch::asm!("{{ .reg .f32 p; mul.rn.f32 p, {t}, 0f3fb8aa3b; add.rn.f32 {k}, p, 0f4b400000; add.rn.f32 {n}, {k}, 0fcb400000; }}", k = out(reg32) k, n = out(reg32) n, t = in(reg32) t, options(pure, nomem, nostack));
        let r = core::f32::math::mul_add(-n, 0.0000014286068f32, core::f32::math::mul_add(-n, 0.693145751953125f32, t));
        let p = core::f32::math::mul_add(1.0 / 5040.0f32, r, 1.0 / 720.0);
        let p = core::f32::math::mul_add(core::f32::math::mul_add(core::f32::math::mul_add(p, r, 1.0 / 120.0), r, 1.0 / 24.0), r, 1.0 / 6.0);
        let p = core::f32::math::mul_add(core::f32::math::mul_add(core::f32::math::mul_add(p, r, 0.5), r, 1.0), r, 1.0);
        p * f32::from_bits(k.to_bits().wrapping_add(127) << 23)
    }};
}

/// GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention (`FLASH_WINDOW_SPLIT16` when `Q_GROUPS == 1`,
/// `FLASH_WINDOW_SPLIT32` when `Q_GROUPS == 2`).
///
/// SAFETY (caller contract, `resident_launch_flash_window_split16` / `resident_launch_flash_window_split32`):
/// `table` and `index` point to an initialized 16-byte aligned `Command` of matching kind recorded by
/// `Backend::flash_window_split_into` after verifying all operand bounds, alignments, non-overlap, and grid geometry.
#[inline(always)]
pub(crate) unsafe fn replay_split<const Q_GROUPS: usize, const SP_SMEM_BYTES: usize>(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized read-only Command; wrong kind traps before reading payload words.
    // - Device re-checks `flash_window_split_admitted`, grid dimensions, optional delta alignment, multi-chat group
    //   bounds (`m_cta <= rows_per_cta`, `r_base + m_cta <= rows`), and per-row `live <= view` and `live <= pos + 1`,
    //   trapping on any malformed command or step.
    // - Static shared memory is strictly sized per monomorphized entry (`25,248` B for `Q_GROUPS=1`, `33,440` B for
    //   `Q_GROUPS=2`), with every sub-region 16-byte aligned and disjoint. Thread 0 writes `c_base` (`160 B`, including
    //   `live[0..m_cta]`) before the prologue `_syncthreads()`, after which `c_base` is read-only.
    // - Global KV reads (`cp.async.cg...L2::256B` for ring/delta, `L2::evict_first` gated to `!is_ring` global KV) and
    //   `prefetch.global.L2` hints are strictly guarded by `col >= k_min && col <= k_max`: ring planes index
    //   `col % view < view`, straight planes index `col < view`, delta rows index `col - first_pos < rows`.
    // - Partial writes use 8-byte aligned `st.global.v2.f32` at `((row * heads + head) * num_splits + s_local) * 128 + even`.
    // - All 128 threads participate in every `_syncthreads()` and `cp.async.wait_group`; every warp participates
    //   uniformly in every `ldmatrix`, `mma`, `shfl_xor`, and `ballot`.
    unsafe {
        let _ = SP_SMEM_BYTES;
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
        let raw_rpc = *w.add(FW_SP_ROWS_PER_CTA_WORD);
        let rows_per_cta = (raw_rpc & 0xffff_ffff) as usize;
        let spc_raw = (raw_rpc >> 32) as usize;
        let splits_per_cta = if spc_raw == 0 { 1 } else { spc_raw };
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let bx = core::arch::nvptx::_block_idx_x() as usize;
        let by = core::arch::nvptx::_block_idx_y() as usize;

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
        let q_ctas = if plane_count != 0 { plane_count } else { rows.div_ceil(rows_per_cta) };
        let gy = core::arch::nvptx::_grid_dim_y() as usize;
        if core::arch::nvptx::_grid_dim_x() as usize != q_ctas * kv_heads || gy.saturating_mul(splits_per_cta) < num_splits {
            core::arch::nvptx::trap();
        }
        let s_begin = by * splits_per_cta;
        if s_begin >= num_splits {
            return;
        }
        let s_end = if s_begin + splits_per_cta < num_splits { s_begin + splits_per_cta } else { num_splits };

        let q_cta = bx / kv_heads;
        let kv_head = bx % kv_heads;
        let (r_base, m_cta, first_pos_cta) = if plane_count != 0 {
            keys = *w.add(FW_SP_PLANES_AT + 3 * q_cta) as *const u32;
            values = *w.add(FW_SP_PLANES_AT + 3 * q_cta + 1) as *const u32;
            let meta = *w.add(FW_SP_PLANES_AT + 3 * q_cta + 2);
            let rb = (meta & 0xffff) as usize;
            let mc = ((meta >> 16) & 0xff) as usize;
            let fp = (meta >> 24) as usize;
            if ((keys as usize) | (values as usize)) & 15 != 0 || mc == 0 || mc > rows_per_cta || rb + mc > rows {
                core::arch::nvptx::trap();
            }
            (rb, mc, fp)
        } else {
            let rb = q_cta * rows_per_cta;
            let mc = if rows - rb > rows_per_cta { rows_per_cta } else { rows - rb };
            (rb, mc, first_pos + rb)
        };
        let q_cnt = m_cta * group;

        // Validate every row step in this CTA and compute key bounds [k_min, k_max] and interior bounds [max_klo, min_khi].
        let mut k_min = usize::MAX;
        let mut k_max = 0usize;
        let mut max_klo = 0usize;
        let mut min_khi = usize::MAX;
        let mut all_live = true;
        let mut any = false;
        let mut r_loc = 0usize;
        while r_loc < m_cta {
            let row = r_base + r_loc;
            let lv_f = *steps.add(2 * row);
            let live = lv_f as usize;
            let pos = first_pos_cta + r_loc;
            if live as f32 != lv_f || live > view || live > pos + 1 || (!is_ring && !has_delta && live > 0 && pos >= view) {
                core::arch::nvptx::trap();
            }
            if live > 0 {
                let klo = pos + 1 - live;
                if klo < k_min { k_min = klo; }
                if pos > k_max { k_max = pos; }
                if klo > max_klo { max_klo = klo; }
                if pos < min_khi { min_khi = pos; }
                any = true;
            } else {
                all_live = false;
            }
            r_loc += 1;
        }
        let s_start_cta = if any && plane_count != 0 { k_min / split_kv + split_start_global } else { split_start_global };
        let range_k0 = (s_start_cta + s_begin) * split_kv;
        let range_k1 = (s_start_cta + s_end) * split_kv;
        if !any || range_k0 > k_max || range_k1 <= k_min {
            core::arch::asm!("griddepcontrol.wait;", options(nostack));
            let tid0 = core::arch::nvptx::_thread_idx_x() as usize;
            if tid0 < q_cnt {
                let ml_b = partials.add(rows * heads * num_splits * FW_HEAD_DIM);
                let row = r_base + tid0 / group;
                let head = kv_head * group + tid0 % group;
                let base_slot = (row * heads + head) * num_splits;
                let mut sl = s_begin;
                while sl < s_end {
                    st_v2_f32!(ml_b.add((base_slot + sl) * 2), f32::NEG_INFINITY, 0.0f32);
                    sl += 1;
                }
            }
            return;
        }

        // Vectorized bit-exact HeadClip prologue (`ld_global_nc_v4` + 4x4 transpose + `st.shared.v2.b32`):
        // - Q: 1 warp per query row (4 rows/pass), 1 coalesced 16B load/lane (16x fewer global loads than scalar).
        // - M: 4 lanes per query row (32 rows/pass across 4 warps), 1 coalesced 16B load/lane, warp-uniform shuffles.
        macro_rules! head_clip_vec {
            ($q_base:expr, $m_base:expr, $max_q_rows:expr) => {{
                let tid_h = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp_h, lane_h) = (tid_h >> 5, tid_h & 31);
                let (ql, _) = (lane_h >> 3, lane_h & 7);
                let has_qs = !query_scale.is_null();
                let has_ms = !moved_scale.is_null();
                let has_pos = !position.is_null();
                let q_st_off = swiz128(0, lane_h >> 1) + ((lane_h & 1) * 8) as u32;
                let q_lim = if $max_q_rows == 16 { 16usize } else { ((q_cnt + 15) & !15).min($max_q_rows) };
                let mut q_base_i = 0usize;
                while q_base_i < q_lim {
                    let qi = q_base_i + warp_h;
                    let dst = $q_base + (qi * 256) as u32 + (q_st_off ^ (((qi & 7) * 16) as u32));
                    if qi < q_cnt {
                        let row = r_base + qi / group;
                        let head = kv_head * group + qi % group;
                        let x4 = ld_global_nc_v4(query.add((row * heads + head) * FW_HEAD_DIM + lane_h * 4));
                        let (mut v0, mut v1) = rn2!(x4[0], x4[1]);
                        let (mut v2, mut v3) = rn2!(x4[2], x4[3]);
                        if has_qs {
                            let qs_p = query_scale.add(head * FW_HEAD_DIM + lane_h * 4);
                            let qs4 = if (query_scale as usize) & 15 == 0 {
                                ld_global_nc_v4(qs_p)
                            } else {
                                [*qs_p, *qs_p.add(1), *qs_p.add(2), *qs_p.add(3)]
                            };
                            v0 *= qs4[0]; v1 *= qs4[1]; v2 *= qs4[2]; v3 *= qs4[3];
                        }
                        let (a0, a1, a2, a3) = (v0 * v0, v1 * v1, v2 * v2, v3 * v3);
                        let p0 = shfl_xor_f32(if (ql & 1) == 0 { a1 } else { a0 }, 8);
                        let (e0, e1) = if (ql & 1) == 0 { (a0, p0) } else { (p0, a1) };
                        let p1 = shfl_xor_f32(if (ql & 1) == 0 { a3 } else { a2 }, 8);
                        let (e2, e3) = if (ql & 1) == 0 { (a2, p1) } else { (p1, a3) };
                        let s0 = shfl_xor_f32(if (ql & 2) == 0 { e2 } else { e0 }, 16);
                        let s1 = shfl_xor_f32(if (ql & 2) == 0 { e3 } else { e1 }, 16);
                        let (t0, t1, t2, t3) = if (ql & 2) == 0 { (e0, e1, s0, s1) } else { (s0, s1, e2, e3) };
                        let mut sum = 0.0f32;
                        sum += t0; sum += t1; sum += t2; sum += t3;
                        sum += shfl_xor_f32(sum, 4);
                        sum += shfl_xor_f32(sum, 2);
                        sum += shfl_xor_f32(sum, 1);
                        sum += shfl_xor_f32(sum, 16);
                        sum += shfl_xor_f32(sum, 8);
                        let qsc = crate::block_attention::clip_factor(crate::block_ops::root(sum), query_bound);
                        let (pw0, pw1) = if !has_pos {
                            (cvt_bf16x2(v0 * qsc, v1 * qsc), cvt_bf16x2(v2 * qsc, v3 * qsc))
                        } else {
                            let fac = *position.add(row);
                            let (c0, c1) = rn2!(v0 * qsc, v1 * qsc);
                            let (c2, c3) = rn2!(v2 * qsc, v3 * qsc);
                            (cvt_bf16x2(c0 * fac, c1 * fac), cvt_bf16x2(c2 * fac, c3 * fac))
                        };
                        core::arch::asm!(
                            "st.shared.v2.b32 [{dst}], {{{a}, {b}}};",
                            dst = in(reg32) dst, a = in(reg32) pw0, b = in(reg32) pw1, options(nostack)
                        );
                    } else {
                        core::arch::asm!(
                            "st.shared.v2.b32 [{dst}], {{0, 0}};",
                            dst = in(reg32) dst, options(nostack)
                        );
                    }
                    q_base_i += 4;
                }
                let (m_grp, m_c) = (lane_h >> 2, lane_h & 3);
                let mut m_pass = 0usize;
                while m_pass * 32 < q_lim {
                    let qi = m_pass * 32 + warp_h * 8 + m_grp;
                    let active = qi < q_cnt;
                    let (row, head) = if active { (r_base + qi / group, kv_head * group + qi % group) } else { (0, 0) };
                    let x4 = if active {
                        ld_global_nc_v4(moved.add((row * heads + head) * FW_REL_DIM + m_c * 4))
                    } else {
                        [0.0f32; 4]
                    };
                    let (mut v0, mut v1) = rn2!(x4[0], x4[1]);
                    let (mut v2, mut v3) = rn2!(x4[2], x4[3]);
                    if active && has_ms {
                        let ms_p = moved_scale.add(head * FW_REL_DIM + m_c * 4);
                        let ms4 = if (moved_scale as usize) & 15 == 0 {
                            ld_global_nc_v4(ms_p)
                        } else {
                            [*ms_p, *ms_p.add(1), *ms_p.add(2), *ms_p.add(3)]
                        };
                        v0 *= ms4[0]; v1 *= ms4[1]; v2 *= ms4[2]; v3 *= ms4[3];
                    }
                    let mut s0 = v0 * v0; let mut s1 = v1 * v1; let mut s2 = v2 * v2; let mut s3 = v3 * v3;
                    s0 += shfl_xor_f32(s0, 2); s1 += shfl_xor_f32(s1, 2);
                    s2 += shfl_xor_f32(s2, 2); s3 += shfl_xor_f32(s3, 2);
                    s0 += shfl_xor_f32(s0, 1); s1 += shfl_xor_f32(s1, 1);
                    s2 += shfl_xor_f32(s2, 1); s3 += shfl_xor_f32(s3, 1);
                    if qi < q_lim {
                        let (pw0, pw1) = if active {
                            let msc = crate::block_attention::clip_factor(crate::block_ops::root((s0 + s2) + (s1 + s3)), moved_bound);
                            if !has_pos {
                                (cvt_bf16x2(v0 * msc, v1 * msc), cvt_bf16x2(v2 * msc, v3 * msc))
                            } else {
                                let fac = *position.add(row);
                                let (c0, c1) = rn2!(v0 * msc, v1 * msc);
                                let (c2, c3) = rn2!(v2 * msc, v3 * msc);
                                (cvt_bf16x2(c0 * fac, c1 * fac), cvt_bf16x2(c2 * fac, c3 * fac))
                            }
                        } else {
                            (0u32, 0u32)
                        };
                        let m_dst = $m_base + swiz_m8(qi, m_c);
                        core::arch::asm!(
                            "st.shared.v2.b32 [{dst}], {{{a}, {b}}};",
                            dst = in(reg32) m_dst, a = in(reg32) pw0, b = in(reg32) pw1, options(nostack)
                        );
                    }
                    m_pass += 1;
                }
            }};
        }

        // Coalesced asynchronous 4-term `T` staging (`cp.async.ca` 4B, 128B-coalesced per warp into 4x(slots x 16B)).
        // Invoked before `crate::ptx::commit()` of the paired `K` tile so `T` overlaps with compute with zero load stalls.
        macro_rules! stage_rel_t {
            ($t_smem:expr, $t_slots:expr, $t_plane_bytes:expr, $rel_p:expr, $tb:expr, $pos0:expr, $mc:expr, $ext_u:expr) => {{
                let tb_u: u32 = $tb;
                let pos0_u: u32 = $pos0;
                let mc_u: u32 = $mc;
                let ext_u: u32 = $ext_u;
                if tb_u + (FW_SPLIT_BKV as u32) + ext_u > pos0_u && tb_u < pos0_u + mc_u {
                    let rel_p: *const u32 = $rel_p;
                    let d_first = pos0_u as i32 - (tb_u + (FW_SPLIT_BKV as u32) - 1) as i32;
                    let lim = (mc_u + (FW_SPLIT_BKV as u32) - 1).min($t_slots as u32);
                    let mut tid_r = core::arch::nvptx::_thread_idx_x() as u32;
                    core::arch::asm!("/* {0} */", inout(reg32) tid_r, options(nomem, nostack));
                    let sub = tid_r >> 5;
                    let mut slot = tid_r & 31;
                    while slot < lim {
                        let d = d_first + slot as i32;
                        let in_b = d >= 0 && (d as u32) < ext_u;
                        let mut q = 0u32;
                        while q < 4 {
                            let dst = $t_smem + q * ($t_plane_bytes as u32) + (slot << 4) + (sub << 2);
                            if in_b {
                                crate::ptx::copy4(dst, rel_p.add(((q * 4 + sub) * ext_u + d as u32) as usize));
                            } else {
                                store_u32(dst, 0);
                            }
                            q += 1;
                        }
                        slot += 32;
                    }
                }
            }};
        }

        // Streaming KV `cp.async.cg.L2::256B`: `L2::evict_first` gated strictly to `!is_ring` global KV so huge global
        // scans never evict L2 partials, while small L2-resident ring windows (`is_ring`) and deltas keep normal L2 caching.
        macro_rules! issue_kv_core {
            ($dst_base:expr, $plane:expr, $delta_w:expr, $tb:expr, $kv_stride:expr, $kv_head_u:expr, $fb:expr, $kmin:expr, $kmax:expr, $view_u:expr, $fp0:expr) => {{
                let tb_u: u32 = $tb;
                let mut tc = core::arch::nvptx::_thread_idx_x() as u32;
                core::arch::asm!("/* {0} */", inout(reg32) tc, options(nomem, nostack));
                let r_copy0 = tc >> 4;
                let ch_copy = tc & 15;
                let dst0 = $dst_base + (r_copy0 << 8) + ((ch_copy ^ (r_copy0 & 7)) << 4);
                let plane: *const u32 = $plane;
                let kv_stride: usize = $kv_stride;
                let kv_head_col = (($kv_head_u) * 64 + ch_copy * 4) as usize;
                let fb: u32 = $fb;
                let kmin: u32 = $kmin;
                let kmax: u32 = $kmax;
                let view_u: u32 = $view_u;
                if (fb & 2) == 0 && tb_u >= kmin && tb_u + 31 <= kmax {
                    if (fb & 1) == 0 {
                        let pol = l2_evict_first();
                        let p0 = plane.add((tb_u + r_copy0) as usize * kv_stride + kv_head_col);
                        copy16_kv_evict!(dst0, p0, pol);
                        copy16_kv_evict!(dst0 + 2048, p0.add(8 * kv_stride), pol);
                        copy16_kv_evict!(dst0 + 4096, p0.add(16 * kv_stride), pol);
                        copy16_kv_evict!(dst0 + 6144, p0.add(24 * kv_stride), pol);
                    } else if (fb & 4) != 0 {
                        let vm = view_u - 1;
                        let c0 = tb_u + r_copy0;
                        crate::ptx::copy16_stream(dst0, plane.add(((c0 & vm) as usize) * kv_stride + kv_head_col));
                        crate::ptx::copy16_stream(dst0 + 2048, plane.add((((c0 + 8) & vm) as usize) * kv_stride + kv_head_col));
                        crate::ptx::copy16_stream(dst0 + 4096, plane.add((((c0 + 16) & vm) as usize) * kv_stride + kv_head_col));
                        crate::ptx::copy16_stream(dst0 + 6144, plane.add((((c0 + 24) & vm) as usize) * kv_stride + kv_head_col));
                    } else {
                        let c0 = tb_u + r_copy0;
                        crate::ptx::copy16_stream(dst0, plane.add(((c0 % view_u) as usize) * kv_stride + kv_head_col));
                        crate::ptx::copy16_stream(dst0 + 2048, plane.add((((c0 + 8) % view_u) as usize) * kv_stride + kv_head_col));
                        crate::ptx::copy16_stream(dst0 + 4096, plane.add((((c0 + 16) % view_u) as usize) * kv_stride + kv_head_col));
                        crate::ptx::copy16_stream(dst0 + 6144, plane.add((((c0 + 24) % view_u) as usize) * kv_stride + kv_head_col));
                    }
                } else {
                    let delta = ld_cmd_u64!(w, $delta_w) as *const u32;
                    let fp0: u32 = $fp0;
                    let mut ci = 0u32;
                    while ci < 4 {
                        let dst = dst0 + ci * 2048;
                        let col = tb_u + r_copy0 + ci * 8;
                        if col >= kmin && col <= kmax {
                            if (fb & 2) != 0 && col >= fp0 {
                                let row_off = ((col - fp0) as usize) * kv_stride + kv_head_col;
                                crate::ptx::copy16_stream(dst, delta.add(row_off));
                            } else {
                                let phys = if (fb & 1) != 0 {
                                    if (fb & 4) != 0 { col & (view_u - 1) } else { col % view_u }
                                } else {
                                    col
                                };
                                crate::ptx::copy16_stream(dst, plane.add((phys as usize) * kv_stride + kv_head_col));
                            }
                        } else {
                            st_shared_zero16(dst);
                        }
                        ci += 1;
                    }
                }
            }};
        }

        macro_rules! prefetch_kv_core {
            ($k_plane:expr, $v_plane:expr, $pf_tb:expr, $kv_stride:expr, $kv_head_u:expr, $fb:expr, $kmin:expr, $kmax:expr) => {{
                let tb_p: u32 = $pf_tb;
                let fb: u32 = $fb;
                // Only prefetch streaming non-ring global KV (`!is_ring && !has_delta`); small rings stay L2-resident.
                if (fb & 3) == 0 && tb_p >= ($kmin) && tb_p + 31 <= ($kmax) {
                    let mut tc = core::arch::nvptx::_thread_idx_x() as u32;
                    core::arch::asm!("/* {0} */", inout(reg32) tc, options(nomem, nostack));
                    let col = tb_p + ((tc & 63) >> 1);
                    let base = if tc >= 64 { $v_plane } else { $k_plane };
                    let pf_col = (($kv_head_u) * 64 + (tc & 1) * 32) as usize;
                    let addr = base.add((col as usize) * ($kv_stride) + pf_col) as u64;
                    core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) addr, options(nostack));
                }
            }};
        }

        let view_pow2 = (view & (view - 1)) == 0;
        let fbits = (is_ring as u32) | ((has_delta as u32) << 1) | ((view_pow2 as u32) << 2) | ((all_live as u32) << 3);

        // Shared `c_base` initializer + SMEM-backed KV/prefetch/empty helpers (keeps both entries at max SM occupancy).
        macro_rules! init_c_smem {
            ($c_base:expr) => {{
                if core::arch::nvptx::_thread_idx_x() == 0 {
                    let cb: u32 = $c_base;
                    store_u32(cb, view as u32); store_u32(cb + 4, extent as u32);
                    store_u32(cb + 8, k_min as u32); store_u32(cb + 12, k_max as u32);
                    store_u32(cb + 16, max_klo as u32); store_u32(cb + 20, min_khi as u32);
                    store_u32(cb + 24, first_pos as u32); store_u32(cb + 28, first_pos_cta as u32);
                    store_u32(cb + 32, m_cta as u32); store_u32(cb + 36, (kv_heads * 64) as u32);
                    store_u32(cb + 40, kv_head as u32); store_u32(cb + 44, fbits);
                    store_u32(cb + 48, s_start_cta as u32); store_u32(cb + 52, split_kv as u32);
                    store_u32(cb + 56, s_end as u32); store_u32(cb + 60, q_cnt as u32);
                    store_u32(cb + 64, r_base as u32); store_u32(cb + 68, group as u32);
                    let (kp, vp, rp) = (keys as u64, values as u64, rel_table as u64);
                    store_u32(cb + 72, kp as u32); store_u32(cb + 76, (kp >> 32) as u32);
                    store_u32(cb + 80, vp as u32); store_u32(cb + 84, (vp >> 32) as u32);
                    store_u32(cb + 88, rp as u32); store_u32(cb + 92, (rp >> 32) as u32);
                    let mut rl = 0usize;
                    while rl < m_cta { store_u32(cb + 96 + (rl * 4) as u32, (*steps.add(2 * (r_base + rl))) as u32); rl += 1; }
                }
            }};
        }
        macro_rules! issue_kv_c {
            ($cb:expr, $dst_base:expr, $is_v:literal, $tb:expr) => {{
                let cb: u32 = $cb;
                issue_kv_core!($dst_base, ld_smem_u64!(cb + if $is_v { 80 } else { 72 }) as *const u32, if $is_v { 6 } else { 5 }, $tb,
                    load_u32(cb + 36) as usize, load_u32(cb + 40), load_u32(cb + 44), load_u32(cb + 8), load_u32(cb + 12), load_u32(cb), load_u32(cb + 24));
            }};
        }
        macro_rules! prefetch_kv_c {
            ($cb:expr, $pf_tb:expr) => {{
                let cb: u32 = $cb;
                prefetch_kv_core!(ld_smem_u64!(cb + 72) as *const u32, ld_smem_u64!(cb + 80) as *const u32, $pf_tb,
                    load_u32(cb + 36) as usize, load_u32(cb + 40), load_u32(cb + 44), load_u32(cb + 8), load_u32(cb + 12));
            }};
        }
        macro_rules! write_empty_c {
            ($cb:expr, $s_loc:expr) => {{
                let cb: u32 = $cb;
                let tid_e = core::arch::nvptx::_thread_idx_x() as usize;
                let qc = load_u32(cb + 60) as usize;
                if tid_e < qc {
                    let grp = load_u32(cb + 68) as usize;
                    let rb = load_u32(cb + 64) as usize;
                    let kvh = load_u32(cb + 40) as usize;
                    let part_e = ld_cmd_u64!(w, 8) as *mut f32;
                    let rows_e = ld_cmd_u64!(w, 9) as usize;
                    let heads_e = ld_cmd_u64!(w, 10) as usize;
                    let ns_e = ((ld_cmd_u64!(w, FW_SP_FLAGS_WORD) >> 32) & 0x7fff_ffff) as usize;
                    let ml_e = part_e.add(rows_e * heads_e * ns_e * FW_HEAD_DIM);
                    let row = rb + tid_e / grp;
                    let head = kvh * grp + tid_e % grp;
                    st_v2_f32!(ml_e.add(((row * heads_e + head) * ns_e + $s_loc) * 2), f32::NEG_INFINITY, 0.0f32);
                }
            }};
        }

        if Q_GROUPS == 1 {
            // === FLASH_WINDOW_SPLIT16: 25,248 B SMEM (4 blocks/SM), <= 128 regs, 0 global `steps` reads in loop ===
            let smem = crate::ptx::shared::<SP16_SMEM_BYTES>();
            let q_base = smem;
            let k_base = q_base + (16 * FW_HEAD_DIM * 2) as u32;
            let v_base = k_base + (FW_SPLIT_BKV * FW_HEAD_DIM * 2) as u32;
            let p_base = v_base + (FW_SPLIT_BKV * FW_HEAD_DIM * 2) as u32;
            let m_base = p_base + (16 * FW_SPLIT_BKV * 2) as u32;
            let t_smem = m_base + (16 * FW_REL_DIM * 2) as u32;
            let red_m = t_smem + (4 * SP16_T_PLANE_BYTES) as u32;
            let red_l = red_m + (4 * 16 * 4) as u32;
            let c_base = red_l + (4 * 16 * 4) as u32;

            init_c_smem!(c_base);
            head_clip_vec!(q_base, m_base, 16);
            core::arch::asm!("griddepcontrol.wait;", options(nostack));
            if q_cnt <= 8 {
                let tid_z = core::arch::nvptx::_thread_idx_x() as usize;
                store_u32(p_base + swiz64((tid_z & 31) / 4 + 8, tid_z >> 5) + ((tid_z & 3) << 2) as u32, 0);
            }
            core::arch::nvptx::_syncthreads();
            // Precompute loop-invariant `rloc0 | (rloc1 << 4)` once (eliminates 2x `div.u32` per tile inside the inner loop).
            let rloc_pack = {
                let r0_i = ((core::arch::nvptx::_thread_idx_x() as u32) & 31) >> 2;
                let qc_i = load_u32(c_base + 60);
                let grp_i = load_u32(c_base + 68);
                let l0 = if r0_i < qc_i { r0_i / grp_i } else { 0 };
                let l1 = if r0_i + 8 < qc_i { (r0_i + 8) / grp_i } else { 0 };
                l0 | (l1 << 4)
            };

            let mut s_local = s_begin as u32;
            while s_local < load_u32(c_base + 56) {
                let split_k0 = (load_u32(c_base + 48) + s_local) * load_u32(c_base + 52);
                let split_k1 = split_k0 + load_u32(c_base + 52);
                let kmin = load_u32(c_base + 8);
                let kmax = load_u32(c_base + 12);
                if split_k0 > kmax || split_k1 <= kmin {
                    write_empty_c!(c_base, s_local as usize);
                    s_local += 1;
                    continue;
                }
                let first_t = if kmin > split_k0 { split_k0 + ((kmin - split_k0) & !31) } else { split_k0 };
                let end_align = (kmax | 31) + 1;
                let last_t = if end_align < split_k1 { end_align } else { split_k1 };
                if first_t >= last_t {
                    write_empty_c!(c_base, s_local as usize);
                    s_local += 1;
                    continue;
                }
                issue_kv_c!(c_base, k_base, false, first_t);
                stage_rel_t!(
                    t_smem, SP16_T_SLOTS, SP16_T_PLANE_BYTES, ld_smem_u64!(c_base + 88) as *const u32,
                    first_t, load_u32(c_base + 28), load_u32(c_base + 32), load_u32(c_base + 4)
                );
                crate::ptx::commit();
                issue_kv_c!(c_base, v_base, true, first_t);
                crate::ptx::commit();
                if first_t + 32 < last_t {
                    prefetch_kv_c!(c_base, first_t + 32);
                }
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();

                let (mut m_r0, mut m_r1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                let (mut l_r0, mut l_r1) = (0.0f32, 0.0f32);
                let (mut o00, mut o01, mut o02, mut o03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let has_hi = load_u32(c_base + 60) > 8;

                let mut t_base = first_t;
                while t_base < last_t {
                    let next_t = t_base + FW_SPLIT_BKV as u32;

                    // 1. Q * K^T behind opaque tid barrier (0 hoisted lane regs -> <= 128 regs = 4 blocks/SM).
                    let mut tid_k = core::arch::nvptx::_thread_idx_x() as u32;
                    core::arch::asm!("/* {0} */", inout(reg32) tid_k, options(nomem, nostack));
                    let warp_k = tid_k >> 5;
                    let lane_k = tid_k & 31;
                    let q_ld_r = lane_k & 15;
                    let q_cur = q_base + q_ld_r * 256;
                    let qs = ((q_ld_r & 7) ^ (lane_k >> 4)) * 16;
                    let k_ld_r = warp_k * 8 + (lane_k & 7);
                    let k_cur = k_base + k_ld_r * 256;
                    let ks_m = ((k_ld_r & 7) ^ ((lane_k >> 3) & 1)) * 16;
                    let mut s0 = [0.0f32; 4];
                    let mut ks = 0u32;
                    while ks < 8 {
                        let kb = ldmatrix_x2(k_cur + ((ks * 32) ^ ks_m));
                        let qa0 = ldmatrix_x4(q_cur + ((ks * 32) ^ qs));
                        s0 = mma_m16n8k16_bf16(qa0, kb, s0);
                        ks += 1;
                    }

                    // 2. Fast interior (no bias / fast_rel ring) vs boundary path (live[] read from SMEM `c_base + 96`).
                    let ext_u = load_u32(c_base + 4);
                    let pos0 = load_u32(c_base + 28);
                    let mc = load_u32(c_base + 32);
                    let fb = load_u32(c_base + 44);
                    let need_rel = t_base + (FW_SPLIT_BKV as u32) + ext_u > pos0 && t_base < pos0 + mc;
                    let interior_geom = (fb & 8) != 0 && t_base >= load_u32(c_base + 16) && t_base + 31 <= load_u32(c_base + 20);
                    let cta_interior = interior_geom && !need_rel;
                    let fast_rel = interior_geom && need_rel && (pos0 + mc - 1 - t_base) < ext_u;
                    let no_mask = cta_interior || fast_rel;
                    let r0 = lane_k >> 2;
                    let r1 = r0 + 8;
                    let t_u = lane_k & 3;
                    if cta_interior {
                        let (a0, a1) = rn2!(s0[0], s0[1]);
                        let (b0, b1) = rn2!(a0 * scale, a1 * scale);
                        s0[0] = b0; s0[1] = b1;
                        if has_hi {
                            let (a2, a3) = rn2!(s0[2], s0[3]);
                            let (b2, b3) = rn2!(a2 * scale, a3 * scale);
                            s0[2] = b2; s0[3] = b3;
                        }
                    } else {
                        let rloc0 = rloc_pack & 15;
                        let rloc1 = rloc_pack >> 4;
                        let col_off = warp_k * 8 + 2 * t_u;
                        let (mut b00, mut b01, mut b02, mut b03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        if need_rel {
                            let mut at0 = t_smem + ((rloc0 + (FW_SPLIT_BKV as u32 - 1) - col_off) << 4);
                            let mut at1 = t_smem + ((rloc1 + (FW_SPLIT_BKV as u32 - 1) - col_off) << 4);
                            let mut jq = 0usize;
                            while jq < 4 {
                                let [w0, w1] = crate::ptx::load2(m_base + swiz_m8(r0 as usize, jq));
                                let (a0, a1) = unpack_bf16x2!(w0);
                                let (a2, a3) = unpack_bf16x2!(w1);
                                let ta = crate::ptx::load4(at0);
                                b00 += a0 * f32::from_bits(ta[0]); b00 += a1 * f32::from_bits(ta[1]);
                                b00 += a2 * f32::from_bits(ta[2]); b00 += a3 * f32::from_bits(ta[3]);
                                let tb = crate::ptx::load4(at0 - 16);
                                b01 += a0 * f32::from_bits(tb[0]); b01 += a1 * f32::from_bits(tb[1]);
                                b01 += a2 * f32::from_bits(tb[2]); b01 += a3 * f32::from_bits(tb[3]);
                                if has_hi {
                                    let [u0, u1] = crate::ptx::load2(m_base + swiz_m8(r1 as usize, jq));
                                    let (m0, m1) = unpack_bf16x2!(u0);
                                    let (m2, m3) = unpack_bf16x2!(u1);
                                    let tc = crate::ptx::load4(at1);
                                    b02 += m0 * f32::from_bits(tc[0]); b02 += m1 * f32::from_bits(tc[1]);
                                    b02 += m2 * f32::from_bits(tc[2]); b02 += m3 * f32::from_bits(tc[3]);
                                    let td = crate::ptx::load4(at1 - 16);
                                    b03 += m0 * f32::from_bits(td[0]); b03 += m1 * f32::from_bits(td[1]);
                                    b03 += m2 * f32::from_bits(td[2]); b03 += m3 * f32::from_bits(td[3]);
                                }
                                at0 += SP16_T_PLANE_BYTES as u32;
                                at1 += SP16_T_PLANE_BYTES as u32;
                                jq += 1;
                            }
                        }
                        if fast_rel {
                            let (a0, a1) = rn2!(s0[0], s0[1]);
                            let (sc0, sc1) = rn2!(a0 * scale, a1 * scale);
                            let (rb0, rb1) = rn2!(b00, b01);
                            let (d0, d1) = rn2!(sc0 + rb0, sc1 + rb1);
                            s0[0] = d0; s0[1] = d1;
                            if has_hi {
                                let (a2, a3) = rn2!(s0[2], s0[3]);
                                let (sc2, sc3) = rn2!(a2 * scale, a3 * scale);
                                let (rb2, rb3) = rn2!(b02, b03);
                                let (d2, d3) = rn2!(sc2 + rb2, sc3 + rb3);
                                s0[2] = d2; s0[3] = d3;
                            }
                        } else {
                            let qc = load_u32(c_base + 60);
                            let lv0 = if r0 < qc { load_u32(c_base + 96 + (rloc0 << 2)) } else { 0 };
                            let lv1 = if r1 < qc { load_u32(c_base + 96 + (rloc1 << 2)) } else { 0 };
                            let khi0 = pos0 + rloc0;
                            let khi1 = pos0 + rloc1;
                            let c0 = t_base + col_off;
                            let c1 = c0 + 1;
                            macro_rules! eval_sc {
                                ($val:expr, $lv:expr, $khi:expr, $col:expr, $bsum:expr) => {{
                                    let col: u32 = $col;
                                    if $lv == 0 || col > $khi || ($khi - col) >= $lv {
                                        f32::NEG_INFINITY
                                    } else {
                                        let scaled = rn(rn($val) * scale);
                                        if need_rel && ($khi - col) < ext_u { rn(scaled + rn($bsum)) } else { scaled }
                                    }
                                }};
                            }
                            s0[0] = eval_sc!(s0[0], lv0, khi0, c0, b00);
                            s0[1] = eval_sc!(s0[1], lv0, khi0, c1, b01);
                            if has_hi {
                                s0[2] = eval_sc!(s0[2], lv1, khi1, c0, b02);
                                s0[3] = eval_sc!(s0[3], lv1, khi1, c1, b03);
                            }
                        }
                    }

                    // 3. Cross-warp deterministic max reduction across all 32 keys.
                    macro_rules! warp_max {
                        ($a:expr, $b:expr) => {{
                            let mut m = fmax_hw!($a, $b);
                            m = fmax_hw!(m, shfl_xor_f32(m, 1));
                            m = fmax_hw!(m, shfl_xor_f32(m, 2));
                            m
                        }};
                    }
                    let mw0 = warp_max!(s0[0], s0[1]);
                    if t_u == 0 { store_u32(red_m + ((warp_k * 16 + r0) << 2), mw0.to_bits()); }
                    if has_hi {
                        let mw1 = warp_max!(s0[2], s0[3]);
                        if t_u == 0 { store_u32(red_m + ((warp_k * 16 + r1) << 2), mw1.to_bits()); }
                    }
                    core::arch::nvptx::_syncthreads();

                    if next_t < last_t {
                        issue_kv_c!(c_base, k_base, false, next_t);
                        stage_rel_t!(
                            t_smem, SP16_T_SLOTS, SP16_T_PLANE_BYTES, ld_smem_u64!(c_base + 88) as *const u32,
                            next_t, load_u32(c_base + 28), load_u32(c_base + 32), load_u32(c_base + 4)
                        );
                        crate::ptx::commit();
                        if next_t + 32 < last_t {
                            prefetch_kv_c!(c_base, next_t + 32);
                        }
                    }

                    macro_rules! tile_max {
                        ($r:expr) => {{
                            let a = f32::from_bits(load_u32(red_m + ($r << 2)));
                            let b = f32::from_bits(load_u32(red_m + ((16 + $r) << 2)));
                            let c = f32::from_bits(load_u32(red_m + ((32 + $r) << 2)));
                            let d = f32::from_bits(load_u32(red_m + ((48 + $r) << 2)));
                            fmax_hw!(fmax_hw!(a, b), fmax_hw!(c, d))
                        }};
                    }
                    macro_rules! to_p {
                        ($v:expr, $mn:expr) => {{
                            let v: f32 = $v;
                            if !no_mask && v == f32::NEG_INFINITY { 0.0f32 } else { exp_nonpos!(v - $mn) }
                        }};
                    }
                    let mt0 = tile_max!(r0);
                    let mn0 = if !no_mask && mt0 == f32::NEG_INFINITY { m_r0 } else { fmax_hw!(m_r0, mt0) };
                    let p00 = to_p!(s0[0], mn0); let p01 = to_p!(s0[1], mn0);
                    store_u32(p_base + swiz64(r0 as usize, warp_k as usize) + (t_u << 2), cvt_bf16x2(p00, p01));
                    let mut lw0 = p00 + p01;
                    lw0 += shfl_xor_f32(lw0, 1); lw0 += shfl_xor_f32(lw0, 2);
                    if t_u == 0 { store_u32(red_l + ((warp_k * 16 + r0) << 2), lw0.to_bits()); }

                    let mut mt1 = f32::NEG_INFINITY;
                    let mut mn1 = m_r1;
                    if has_hi {
                        mt1 = tile_max!(r1);
                        mn1 = if !no_mask && mt1 == f32::NEG_INFINITY { m_r1 } else { fmax_hw!(m_r1, mt1) };
                        let p02 = to_p!(s0[2], mn1); let p03 = to_p!(s0[3], mn1);
                        store_u32(p_base + swiz64(r1 as usize, warp_k as usize) + (t_u << 2), cvt_bf16x2(p02, p03));
                        let mut lw1 = p02 + p03;
                        lw1 += shfl_xor_f32(lw1, 1); lw1 += shfl_xor_f32(lw1, 2);
                        if t_u == 0 { store_u32(red_l + ((warp_k * 16 + r1) << 2), lw1.to_bits()); }
                    }
                    if next_t < last_t { cp_async_wait_one(); } else { cp_async_wait_zero(); }
                    core::arch::nvptx::_syncthreads();

                    macro_rules! tile_sum {
                        ($r:expr) => {{
                            let a = f32::from_bits(load_u32(red_l + ($r << 2)));
                            let b = f32::from_bits(load_u32(red_l + ((16 + $r) << 2)));
                            let c = f32::from_bits(load_u32(red_l + ((32 + $r) << 2)));
                            let d = f32::from_bits(load_u32(red_l + ((48 + $r) << 2)));
                            (a + b) + (c + d)
                        }};
                    }
                    if no_mask || mt0 != f32::NEG_INFINITY {
                        let lt = tile_sum!(r0);
                        if m_r0 == f32::NEG_INFINITY { l_r0 = lt; o00[0] = 0.0; o00[1] = 0.0; o01[0] = 0.0; o01[1] = 0.0; o02[0] = 0.0; o02[1] = 0.0; o03[0] = 0.0; o03[1] = 0.0; }
                        else if mn0 > m_r0 { let al = exp_nonpos!(m_r0 - mn0); l_r0 = l_r0 * al + lt; o00[0] *= al; o00[1] *= al; o01[0] *= al; o01[1] *= al; o02[0] *= al; o02[1] *= al; o03[0] *= al; o03[1] *= al; }
                        else { l_r0 += lt; }
                        m_r0 = mn0;
                    }
                    if has_hi && (no_mask || mt1 != f32::NEG_INFINITY) {
                        let lt = tile_sum!(r1);
                        if m_r1 == f32::NEG_INFINITY { l_r1 = lt; o00[2] = 0.0; o00[3] = 0.0; o01[2] = 0.0; o01[3] = 0.0; o02[2] = 0.0; o02[3] = 0.0; o03[2] = 0.0; o03[3] = 0.0; }
                        else if mn1 > m_r1 { let al = exp_nonpos!(m_r1 - mn1); l_r1 = l_r1 * al + lt; o00[2] *= al; o00[3] *= al; o01[2] *= al; o01[3] *= al; o02[2] *= al; o02[3] *= al; o03[2] *= al; o03[3] *= al; }
                        else { l_r1 += lt; }
                        m_r1 = mn1;
                    }

                    // 4. O += P * V for warp's 32 columns.
                    let v_ld_r0 = (lane_k & 15) as usize;
                    let v_ld_r1 = 16 + v_ld_r0;
                    let v_ld_sub = (lane_k >> 4) as usize;
                    let w_u = warp_k as usize;
                    let vb0_a = ldmatrix_x4_trans(v_base + swiz128(v_ld_r0, 4 * w_u + v_ld_sub));
                    let vb0_b = ldmatrix_x4_trans(v_base + swiz128(v_ld_r0, 4 * w_u + 2 + v_ld_sub));
                    let pa0_k0 = ldmatrix_x4(p_base + swiz64(v_ld_r0, v_ld_sub));
                    o00 = mma_m16n8k16_bf16(pa0_k0, [vb0_a[0], vb0_a[1]], o00);
                    o01 = mma_m16n8k16_bf16(pa0_k0, [vb0_a[2], vb0_a[3]], o01);
                    o02 = mma_m16n8k16_bf16(pa0_k0, [vb0_b[0], vb0_b[1]], o02);
                    o03 = mma_m16n8k16_bf16(pa0_k0, [vb0_b[2], vb0_b[3]], o03);
                    let vb1_a = ldmatrix_x4_trans(v_base + swiz128(v_ld_r1, 4 * w_u + v_ld_sub));
                    let vb1_b = ldmatrix_x4_trans(v_base + swiz128(v_ld_r1, 4 * w_u + 2 + v_ld_sub));
                    let pa0_k1 = ldmatrix_x4(p_base + swiz64(v_ld_r0, 2 + v_ld_sub));
                    o00 = mma_m16n8k16_bf16(pa0_k1, [vb1_a[0], vb1_a[1]], o00);
                    o01 = mma_m16n8k16_bf16(pa0_k1, [vb1_a[2], vb1_a[3]], o01);
                    o02 = mma_m16n8k16_bf16(pa0_k1, [vb1_b[0], vb1_b[1]], o02);
                    o03 = mma_m16n8k16_bf16(pa0_k1, [vb1_b[2], vb1_b[3]], o03);

                    if next_t < last_t {
                        core::arch::nvptx::_syncthreads();
                        issue_kv_c!(c_base, v_base, true, next_t);
                        crate::ptx::commit();
                        cp_async_wait_one();
                        core::arch::nvptx::_syncthreads();
                    }
                    t_base = next_t;
                }

                // Prefetch the first tile of the next split while storing this split's L2-resident partials.
                if s_local + 1 < load_u32(c_base + 56) {
                    let next_k0 = (load_u32(c_base + 48) + s_local + 1) * load_u32(c_base + 52);
                    if next_k0 <= load_u32(c_base + 12) {
                        prefetch_kv_c!(c_base, next_k0);
                    }
                }

                let part_e = ld_cmd_u64!(w, 8) as *mut f32;
                let heads_e = ld_cmd_u64!(w, 10) as usize;
                let ns_e = ((ld_cmd_u64!(w, FW_SP_FLAGS_WORD) >> 32) & 0x7fff_ffff) as usize;
                let ml_e = part_e.add((ld_cmd_u64!(w, 9) as usize) * heads_e * ns_e * FW_HEAD_DIM);
                let tid_e = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp_e, lane_e) = (tid_e >> 5, tid_e & 31);
                let (g_e, t_e) = (lane_e >> 2, lane_e & 3);
                let qc_e = load_u32(c_base + 60) as usize;
                let rb_e = load_u32(c_base + 64) as usize;
                let grp_e = load_u32(c_base + 68) as usize;
                let kvh_e = load_u32(c_base + 40) as usize;
                if g_e < qc_e {
                    let row = rb_e + g_e / grp_e;
                    let head = kvh_e * grp_e + g_e % grp_e;
                    let slot_idx = (row * heads_e + head) * ns_e + s_local as usize;
                    let dst = part_e.add(slot_idx * FW_HEAD_DIM + 32 * warp_e + 2 * t_e);
                    st_v2_f32!(dst, o00[0], o00[1]); st_v2_f32!(dst.add(8), o01[0], o01[1]);
                    st_v2_f32!(dst.add(16), o02[0], o02[1]); st_v2_f32!(dst.add(24), o03[0], o03[1]);
                    if warp_e == 0 && t_e == 0 { st_v2_f32!(ml_e.add(slot_idx * 2), m_r0, l_r0); }
                }
                let rb_i = g_e + 8;
                if rb_i < qc_e {
                    let row = rb_e + rb_i / grp_e;
                    let head = kvh_e * grp_e + rb_i % grp_e;
                    let slot_idx = (row * heads_e + head) * ns_e + s_local as usize;
                    let dst = part_e.add(slot_idx * FW_HEAD_DIM + 32 * warp_e + 2 * t_e);
                    st_v2_f32!(dst, o00[2], o00[3]); st_v2_f32!(dst.add(8), o01[2], o01[3]);
                    st_v2_f32!(dst.add(16), o02[2], o02[3]); st_v2_f32!(dst.add(24), o03[2], o03[3]);
                    if warp_e == 0 && t_e == 0 { st_v2_f32!(ml_e.add(slot_idx * 2), m_r1, l_r1); }
                }
                if s_local + 1 < load_u32(c_base + 56) {
                    core::arch::nvptx::_syncthreads();
                }
                s_local += 1;
            }
        } else {
            // === FLASH_WINDOW_SPLIT32 (48-row warp-per-16-Q-rows): 33,440 B SMEM (3 blocks/SM), 170 regs ===
            let smem = crate::ptx::shared::<SP48_SMEM_BYTES>();
            let q_base = smem;
            let k_base = q_base + (48 * FW_HEAD_DIM * 2) as u32;
            let v_base = k_base + (FW_SPLIT_BKV * FW_HEAD_DIM * 2) as u32;
            let m_base = v_base + (FW_SPLIT_BKV * FW_HEAD_DIM * 2) as u32;
            let t_smem = m_base + (48 * FW_REL_DIM * 2) as u32;
            let c_base = t_smem + (4 * SP48_T_PLANE_BYTES) as u32;

            init_c_smem!(c_base);
            head_clip_vec!(q_base, m_base, 48);
            core::arch::asm!("griddepcontrol.wait;", options(nostack));
            core::arch::nvptx::_syncthreads();
            macro_rules! ldm_and_mma2 {
                ($ptr:expr, $qa:expr, $kb:expr, $sa:ident, $sb:ident) => {{
                    let (n0, n1, n2, n3): (u32, u32, u32, u32);
                    core::arch::asm!(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{n0}, {n1}, {n2}, {n3}}}, [{p}];\n\
                         mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{a0},{a1},{a2},{a3}}}, {{{q0},{q1},{q2},{q3}}}, {{{b0},{b1}}}, {{{a0},{a1},{a2},{a3}}};\n\
                         mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{c0},{c1},{c2},{c3}}}, {{{q0},{q1},{q2},{q3}}}, {{{b2},{b3}}}, {{{c0},{c1},{c2},{c3}}};",
                        n0 = out(reg32) n0, n1 = out(reg32) n1, n2 = out(reg32) n2, n3 = out(reg32) n3,
                        p = in(reg32) ($ptr),
                        q0 = in(reg32) $qa[0], q1 = in(reg32) $qa[1], q2 = in(reg32) $qa[2], q3 = in(reg32) $qa[3],
                        b0 = in(reg32) $kb[0], b1 = in(reg32) $kb[1], b2 = in(reg32) $kb[2], b3 = in(reg32) $kb[3],
                        a0 = inout(reg32) $sa[0], a1 = inout(reg32) $sa[1], a2 = inout(reg32) $sa[2], a3 = inout(reg32) $sa[3],
                        c0 = inout(reg32) $sb[0], c1 = inout(reg32) $sb[1], c2 = inout(reg32) $sb[2], c3 = inout(reg32) $sb[3],
                        options(nostack),
                    );
                    [n0, n1, n2, n3]
                }};
            }

            let mut s_local = s_begin as u32;
            while s_local < load_u32(c_base + 56) {
                let split_k0 = (load_u32(c_base + 48) + s_local) * load_u32(c_base + 52);
                let split_k1 = split_k0 + load_u32(c_base + 52);
                let kmin = load_u32(c_base + 8);
                let kmax = load_u32(c_base + 12);
                if split_k0 > kmax || split_k1 <= kmin {
                    write_empty_c!(c_base, s_local as usize);
                    s_local += 1;
                    continue;
                }
                let first_t = if kmin > split_k0 { split_k0 + ((kmin - split_k0) & !31) } else { split_k0 };
                let end_align = (kmax | 31) + 1;
                let last_t = if end_align < split_k1 { end_align } else { split_k1 };
                if first_t >= last_t {
                    write_empty_c!(c_base, s_local as usize);
                    s_local += 1;
                    continue;
                }

                // Prologue: load K[first_t] + T[first_t], sync once.
                issue_kv_c!(c_base, k_base, false, first_t);
                stage_rel_t!(
                    t_smem, SP48_T_SLOTS, SP48_T_PLANE_BYTES, ld_smem_u64!(c_base + 88) as *const u32,
                    first_t, load_u32(c_base + 28), load_u32(c_base + 32), load_u32(c_base + 4)
                );
                crate::ptx::commit();
                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();

                let (mut m_r0, mut m_r1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                let (mut l_r0, mut l_r1) = (0.0f32, 0.0f32);
                let (mut o0, mut o1, mut o2, mut o3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let (mut o4, mut o5, mut o6, mut o7) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let (mut o8, mut o9, mut o10, mut o11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let (mut o12, mut o13, mut o14, mut o15) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let w_id = (core::arch::nvptx::_thread_idx_x() as u32) >> 5;
                let warp_active = w_id < 3 && (w_id * 16) < load_u32(c_base + 60);

                let mut t_base = first_t;
                while t_base < last_t {
                    let next_t = t_base + FW_SPLIT_BKV as u32;
                    issue_kv_c!(c_base, v_base, true, t_base);
                    crate::ptx::commit();
                    if next_t < last_t {
                        prefetch_kv_c!(c_base, next_t);
                    }

                    let mut pa0 = [0u32; 4];
                    let mut pa1 = [0u32; 4];
                    if warp_active {
                        // 1. S = Q * K^T (16 x 32 per warp, ascending d = 0..128) with single-`kb` fused LDSM+HMMA.
                        let mut tid_k = core::arch::nvptx::_thread_idx_x() as u32;
                        core::arch::asm!("/* {0} */", inout(reg32) tid_k, options(nomem, nostack));
                        let warp_k = tid_k >> 5;
                        let lane_k = tid_k & 31;
                        let q_ld_row = warp_k * 16 + (lane_k & 15);
                        let mut q_cur = q_base + q_ld_row * 256;
                        let mut qs = ((q_ld_row & 7) ^ (lane_k >> 4)) * 16;
                        let k_ld_row0 = ((lane_k >> 4) * 8) + (lane_k & 7);
                        let mut k_cur = k_base + k_ld_row0 * 256;
                        let mut ks_m = ((k_ld_row0 & 7) ^ ((lane_k >> 3) & 1)) * 16;
                        core::arch::asm!("/* {0} {1} {2} {3} */", inout(reg32) q_cur, inout(reg32) k_cur, inout(reg32) qs, inout(reg32) ks_m, options(nomem, nostack));
                        let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                        let mut kb = ldmatrix_x4(k_cur + ks_m);
                        let mut ks = 0u32;
                        while ks < 7 {
                            let qa = ldmatrix_x4(q_cur + ((ks * 32) ^ qs));
                            let cur_o = (ks * 32) ^ ks_m;
                            let nxt_o = ((ks + 1) * 32) ^ ks_m;
                            kb = ldm_and_mma2!(k_cur + 4096 + cur_o, qa, kb, s0, s1);
                            kb = ldm_and_mma2!(k_cur + nxt_o, qa, kb, s2, s3);
                            ks += 1;
                        }
                        let qa7 = ldmatrix_x4(q_cur + ((7 * 32) ^ qs));
                        kb = ldm_and_mma2!(k_cur + 4096 + ((7 * 32) ^ ks_m), qa7, kb, s0, s1);
                        s2 = mma_m16n8k16_bf16(qa7, [kb[0], kb[1]], s2);
                        s3 = mma_m16n8k16_bf16(qa7, [kb[2], kb[3]], s3);

                        // 2. Fast interior vs register-compressed RelBias / boundary (`ps0..ps3` + `pf0..pf1` -> <= 168 regs).
                        let ext_u = load_u32(c_base + 4);
                        let pos0 = load_u32(c_base + 28);
                        let mc = load_u32(c_base + 32);
                        let fb = load_u32(c_base + 44);
                        let need_rel = t_base + (FW_SPLIT_BKV as u32) + ext_u > pos0 && t_base < pos0 + mc;
                        let interior_geom = (fb & 8) != 0 && t_base >= load_u32(c_base + 16) && t_base + 31 <= load_u32(c_base + 20);
                        let cta_interior = interior_geom && !need_rel;
                        let fast_rel = interior_geom && need_rel && (pos0 + mc - 1 - t_base) < ext_u;
                        let no_mask = cta_interior || fast_rel;
                        let sc_u = f32::from_bits(ld_cmd_u64!(w, 14) as u32);
                        if cta_interior {
                            macro_rules! sc_fast {
                                ($s:ident) => {{
                                    let (a0, a1) = rn2!($s[0], $s[1]); let (a2, a3) = rn2!($s[2], $s[3]);
                                    let (b0, b1) = rn2!(a0 * sc_u, a1 * sc_u); let (b2, b3) = rn2!(a2 * sc_u, a3 * sc_u);
                                    $s = [b0, b1, b2, b3];
                                }};
                            }
                            sc_fast!(s0); sc_fast!(s1); sc_fast!(s2); sc_fast!(s3);
                        } else {
                            macro_rules! pack_sc {
                                ($s:ident) => {{
                                    let (a0, a1) = rn2!($s[0], $s[1]);
                                    let (a2, a3) = rn2!($s[2], $s[3]);
                                    [cvt_bf16x2(a0 * sc_u, a1 * sc_u), cvt_bf16x2(a2 * sc_u, a3 * sc_u)]
                                }};
                            }
                            let ps0 = pack_sc!(s0);
                            let ps1 = pack_sc!(s1);
                            let ps2 = pack_sc!(s2);
                            let ps3 = pack_sc!(s3);
                            let r0 = warp_k * 16 + (lane_k >> 2);
                            let r1 = r0 + 8;
                            let t_u = lane_k & 3;
                            let qc = load_u32(c_base + 60);
                            let grp = load_u32(c_base + 68);
                            let rloc0 = if r0 < qc { r0 / grp } else { 0 };
                            let rloc1 = if r1 < qc { r1 / grp } else { 0 };
                            macro_rules! fin_ps {
                                ($ps:expr, $off:expr, $b:expr) => {{
                                    let [p01, p23]: [u32; 2] = $ps;
                                    let (c0, c1) = unpack_bf16x2!(p01);
                                    let (c2, c3) = unpack_bf16x2!(p23);
                                    let [b00, b01, b10, b11]: [f32; 4] = $b;
                                    if fast_rel {
                                        let (rb0, rb1) = rn2!(b00, b01);
                                        let (rb2, rb3) = rn2!(b10, b11);
                                        let (d0, d1) = rn2!(c0 + rb0, c1 + rb1);
                                        let (d2, d3) = rn2!(c2 + rb2, c3 + rb3);
                                        [d0, d1, d2, d3]
                                    } else {
                                        let lv0 = if r0 < qc { load_u32(c_base + 96 + (rloc0 << 2)) } else { 0 };
                                        let lv1 = if r1 < qc { load_u32(c_base + 96 + (rloc1 << 2)) } else { 0 };
                                        let khi0 = pos0 + rloc0;
                                        let khi1 = pos0 + rloc1;
                                        let col0 = t_base + $off + 2 * t_u;
                                        let col1 = col0 + 1;
                                        macro_rules! eval_sc {
                                            ($sc:expr, $lv:expr, $khi:expr, $col:expr, $bsum:expr) => {{
                                                let col: u32 = $col;
                                                if $lv == 0 || col > $khi || ($khi - col) >= $lv {
                                                    f32::NEG_INFINITY
                                                } else if need_rel && ($khi - col) < ext_u {
                                                    rn($sc + rn($bsum))
                                                } else {
                                                    $sc
                                                }
                                            }};
                                        }
                                        [
                                            eval_sc!(c0, lv0, khi0, col0, b00),
                                            eval_sc!(c1, lv0, khi0, col1, b01),
                                            eval_sc!(c2, lv1, khi1, col0, b10),
                                            eval_sc!(c3, lv1, khi1, col1, b11),
                                        ]
                                    }
                                }};
                            }
                            if need_rel {
                                let at0_start = t_smem + ((rloc0 + (FW_SPLIT_BKV as u32 - 1) - 2 * t_u) << 4);
                                let at1_start = t_smem + ((rloc1 + (FW_SPLIT_BKV as u32 - 1) - 2 * t_u) << 4);
                                macro_rules! m_quad {
                                    ($row:expr, $jq:expr) => {{
                                        let [w0, w1] = crate::ptx::load2(m_base + swiz_m8($row as usize, $jq));
                                        let (m0, m1) = unpack_bf16x2!(w0);
                                        let (m2, m3) = unpack_bf16x2!(w1);
                                        [m0, m1, m2, m3]
                                    }};
                                }
                                macro_rules! t_acc {
                                    ($b:ident, $m:ident, $at:expr) => {{
                                        let t = crate::ptx::load4($at);
                                        $b += $m[0] * f32::from_bits(t[0]); $b += $m[1] * f32::from_bits(t[1]);
                                        $b += $m[2] * f32::from_bits(t[2]); $b += $m[3] * f32::from_bits(t[3]);
                                    }};
                                }
                                let (mut b00, mut b01, mut b02, mut b03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                                let (mut b10, mut b11, mut b12, mut b13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                                let (mut jq, mut at0, mut at1) = (0usize, at0_start, at1_start);
                                while jq < 4 {
                                    let a = m_quad!(r0, jq);
                                    let m = m_quad!(r1, jq);
                                    t_acc!(b00, a, at0);       t_acc!(b01, a, at0 - 16);
                                    t_acc!(b02, m, at1);       t_acc!(b03, m, at1 - 16);
                                    t_acc!(b10, a, at0 - 128); t_acc!(b11, a, at0 - 144);
                                    t_acc!(b12, m, at1 - 128); t_acc!(b13, m, at1 - 144);
                                    at0 += SP48_T_PLANE_BYTES as u32;
                                    at1 += SP48_T_PLANE_BYTES as u32;
                                    jq += 1;
                                }
                                let f0 = fin_ps!(ps0, 0, [b00, b01, b02, b03]);
                                let f1 = fin_ps!(ps1, 8, [b10, b11, b12, b13]);
                                let pf0 = [pack_hi16!(f0[0], f0[1]), pack_hi16!(f0[2], f0[3])];
                                let pf1 = [pack_hi16!(f1[0], f1[1]), pack_hi16!(f1[2], f1[3])];
                                let (mut b20, mut b21, mut b22, mut b23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                                let (mut b30, mut b31, mut b32, mut b33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                                jq = 0; at0 = at0_start; at1 = at1_start;
                                while jq < 4 {
                                    let a = m_quad!(r0, jq);
                                    let m = m_quad!(r1, jq);
                                    t_acc!(b20, a, at0 - 256); t_acc!(b21, a, at0 - 272);
                                    t_acc!(b22, m, at1 - 256); t_acc!(b23, m, at1 - 272);
                                    t_acc!(b30, a, at0 - 384); t_acc!(b31, a, at0 - 400);
                                    t_acc!(b32, m, at1 - 384); t_acc!(b33, m, at1 - 400);
                                    at0 += SP48_T_PLANE_BYTES as u32;
                                    at1 += SP48_T_PLANE_BYTES as u32;
                                    jq += 1;
                                }
                                s2 = fin_ps!(ps2, 16, [b20, b21, b22, b23]);
                                s3 = fin_ps!(ps3, 24, [b30, b31, b32, b33]);
                                let (u00, u01) = unpack_bf16x2!(pf0[0]); let (u02, u03) = unpack_bf16x2!(pf0[1]);
                                let (u10, u11) = unpack_bf16x2!(pf1[0]); let (u12, u13) = unpack_bf16x2!(pf1[1]);
                                s0 = [u00, u01, u02, u03];
                                s1 = [u10, u11, u12, u13];
                            } else {
                                s0 = fin_ps!(ps0, 0, [0.0; 4]);
                                s1 = fin_ps!(ps1, 8, [0.0; 4]);
                                s2 = fin_ps!(ps2, 16, [0.0; 4]);
                                s3 = fin_ps!(ps3, 24, [0.0; 4]);
                            }
                        }

                        // 3. Warp-local online softmax: bit-exact 4-group sum order, retiring p00..p13 before p20..p33.
                        let mut mt0 = fmax_hw!(fmax_hw!(fmax_hw!(s0[0], s0[1]), fmax_hw!(s1[0], s1[1])), fmax_hw!(fmax_hw!(s2[0], s2[1]), fmax_hw!(s3[0], s3[1])));
                        mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 1));
                        mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 2));
                        let mut mt1 = fmax_hw!(fmax_hw!(fmax_hw!(s0[2], s0[3]), fmax_hw!(s1[2], s1[3])), fmax_hw!(fmax_hw!(s2[2], s2[3]), fmax_hw!(s3[2], s3[3])));
                        mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 1));
                        mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 2));
                        let mn0 = if !no_mask && mt0 == f32::NEG_INFINITY { m_r0 } else { fmax_hw!(m_r0, mt0) };
                        let mn1 = if !no_mask && mt1 == f32::NEG_INFINITY { m_r1 } else { fmax_hw!(m_r1, mt1) };
                        macro_rules! to_p {
                            ($v:expr, $mn:expr) => {{
                                let v: f32 = $v;
                                if !no_mask && v == f32::NEG_INFINITY { 0.0f32 } else { exp_nonpos!(v - $mn) }
                            }};
                        }
                        macro_rules! grp8_sum {
                            ($a:expr, $b:expr) => {{
                                let mut s = $a + $b;
                                s += shfl_xor_f32(s, 1);
                                s += shfl_xor_f32(s, 2);
                                s
                            }};
                        }
                        let (lt0_lo, lt1_lo) = {
                            let p00 = to_p!(s0[0], mn0); let p01 = to_p!(s0[1], mn0);
                            let p02 = to_p!(s0[2], mn1); let p03 = to_p!(s0[3], mn1);
                            let p10 = to_p!(s1[0], mn0); let p11 = to_p!(s1[1], mn0);
                            let p12 = to_p!(s1[2], mn1); let p13 = to_p!(s1[3], mn1);
                            pa0 = [cvt_bf16x2(p00, p01), cvt_bf16x2(p02, p03), cvt_bf16x2(p10, p11), cvt_bf16x2(p12, p13)];
                            (grp8_sum!(p00, p01) + grp8_sum!(p10, p11), grp8_sum!(p02, p03) + grp8_sum!(p12, p13))
                        };
                        let (lt0, lt1) = {
                            let p20 = to_p!(s2[0], mn0); let p21 = to_p!(s2[1], mn0);
                            let p22 = to_p!(s2[2], mn1); let p23 = to_p!(s2[3], mn1);
                            let p30 = to_p!(s3[0], mn0); let p31 = to_p!(s3[1], mn0);
                            let p32 = to_p!(s3[2], mn1); let p33 = to_p!(s3[3], mn1);
                            pa1 = [cvt_bf16x2(p20, p21), cvt_bf16x2(p22, p23), cvt_bf16x2(p30, p31), cvt_bf16x2(p32, p33)];
                            (lt0_lo + (grp8_sum!(p20, p21) + grp8_sum!(p30, p31)), lt1_lo + (grp8_sum!(p22, p23) + grp8_sum!(p32, p33)))
                        };

                        let chg = crate::ptx::ballot(
                            ((no_mask || mt0 != f32::NEG_INFINITY) && (m_r0 == f32::NEG_INFINITY || mn0 > m_r0))
                            || ((no_mask || mt1 != f32::NEG_INFINITY) && (m_r1 == f32::NEG_INFINITY || mn1 > m_r1)),
                        ) != 0;
                        if !chg {
                            if no_mask || mt0 != f32::NEG_INFINITY { l_r0 += lt0; }
                            if no_mask || mt1 != f32::NEG_INFINITY { l_r1 += lt1; }
                        } else {
                            macro_rules! z_acc { ($a:literal, $b:literal, $($o:ident)*) => { $($o[$a] = 0.0; $o[$b] = 0.0;)* }; }
                            macro_rules! sc_acc { ($al:expr, $a:literal, $b:literal, $($o:ident)*) => { $($o[$a] *= $al; $o[$b] *= $al;)* }; }
                            if no_mask || mt0 != f32::NEG_INFINITY {
                                if m_r0 == f32::NEG_INFINITY { l_r0 = lt0; z_acc!(0, 1, o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                                else if mn0 > m_r0 { let al = exp_nonpos!(m_r0 - mn0); l_r0 = l_r0 * al + lt0; sc_acc!(al, 0, 1, o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                                else { l_r0 += lt0; }
                                m_r0 = mn0;
                            }
                            if no_mask || mt1 != f32::NEG_INFINITY {
                                if m_r1 == f32::NEG_INFINITY { l_r1 = lt1; z_acc!(2, 3, o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                                else if mn1 > m_r1 { let al = exp_nonpos!(m_r1 - mn1); l_r1 = l_r1 * al + lt1; sc_acc!(al, 2, 3, o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                                else { l_r1 += lt1; }
                                m_r1 = mn1;
                            }
                        }
                    }

                    // Barrier #1: V[t_base] ready and all warps finished reading K[t_base] & T[t_base].
                    cp_async_wait_zero();
                    core::arch::nvptx::_syncthreads();
                    if next_t < last_t {
                        issue_kv_c!(c_base, k_base, false, next_t);
                        stage_rel_t!(
                            t_smem, SP48_T_SLOTS, SP48_T_PLANE_BYTES, ld_smem_u64!(c_base + 88) as *const u32,
                            next_t, load_u32(c_base + 28), load_u32(c_base + 32), load_u32(c_base + 4)
                        );
                        crate::ptx::commit();
                    }

                    // 4. O += P * V[t_base] (16 x 128 x 32 per active warp, overlapped with K[next_t] load).
                    if warp_active {
                        let mut tid_v = core::arch::nvptx::_thread_idx_x() as u32;
                        core::arch::asm!("/* {0} */", inout(reg32) tid_v, options(nomem, nostack));
                        let lv = tid_v & 31;
                        let mut v_cur = v_base + (lv & 15) * 256;
                        let mut vs = ((lv & 7) ^ (lv >> 4)) * 16;
                        core::arch::asm!("/* {0} {1} */", inout(reg32) v_cur, inout(reg32) vs, options(nomem, nostack));
                        macro_rules! pv_pair {
                            ($pair:expr, $oa:ident, $ob:ident) => {
                                let v_off = ($pair * 32) ^ vs;
                                let vb0 = ldmatrix_x4_trans(v_cur + v_off);
                                $oa = mma_m16n8k16_bf16(pa0, [vb0[0], vb0[1]], $oa);
                                $ob = mma_m16n8k16_bf16(pa0, [vb0[2], vb0[3]], $ob);
                                let vb1 = ldmatrix_x4_trans(v_cur + 4096 + v_off);
                                $oa = mma_m16n8k16_bf16(pa1, [vb1[0], vb1[1]], $oa);
                                $ob = mma_m16n8k16_bf16(pa1, [vb1[2], vb1[3]], $ob);
                            };
                        }
                        pv_pair!(0, o0, o1); pv_pair!(1, o2, o3);
                        pv_pair!(2, o4, o5); pv_pair!(3, o6, o7);
                        pv_pair!(4, o8, o9); pv_pair!(5, o10, o11);
                        pv_pair!(6, o12, o13); pv_pair!(7, o14, o15);
                    }

                    // Barrier #2: K[next_t]+T[next_t] ready and all warps finished reading V[t_base].
                    if next_t < last_t {
                        cp_async_wait_zero();
                        core::arch::nvptx::_syncthreads();
                    }
                    t_base = next_t;
                }

                // Prefetch the first tile of the next split while storing this split's L2-resident partials.
                if s_local + 1 < load_u32(c_base + 56) {
                    let next_k0 = (load_u32(c_base + 48) + s_local + 1) * load_u32(c_base + 52);
                    if next_k0 <= load_u32(c_base + 12) {
                        prefetch_kv_c!(c_base, next_k0);
                    }
                }

                if warp_active {
                    let part_e = ld_cmd_u64!(w, 8) as *mut f32;
                    let heads_e = ld_cmd_u64!(w, 10) as usize;
                    let ns_e = ((ld_cmd_u64!(w, FW_SP_FLAGS_WORD) >> 32) & 0x7fff_ffff) as usize;
                    let ml_e = part_e.add((ld_cmd_u64!(w, 9) as usize) * heads_e * ns_e * FW_HEAD_DIM);
                    let tid_e = core::arch::nvptx::_thread_idx_x() as usize;
                    let (warp_e, lane_e) = (tid_e >> 5, tid_e & 31);
                    let (g_e, t_e) = (lane_e >> 2, lane_e & 31 & 3);
                    let qc_e = load_u32(c_base + 60) as usize;
                    let rb_e = load_u32(c_base + 64) as usize;
                    let grp_e = load_u32(c_base + 68) as usize;
                    let kvh_e = load_u32(c_base + 40) as usize;
                    let ra = warp_e * 16 + g_e;
                    if ra < qc_e {
                        let row = rb_e + ra / grp_e;
                        let head = kvh_e * grp_e + ra % grp_e;
                        let slot_idx = (row * heads_e + head) * ns_e + s_local as usize;
                        let dst = part_e.add(slot_idx * FW_HEAD_DIM + 2 * t_e);
                        macro_rules! st0 {
                            ($($ni:literal $o:ident),*) => {$( st_v2_f32!(dst.add($ni * 8), $o[0], $o[1]); )*};
                        }
                        st0!(0 o0, 1 o1, 2 o2, 3 o3, 4 o4, 5 o5, 6 o6, 7 o7, 8 o8, 9 o9, 10 o10, 11 o11, 12 o12, 13 o13, 14 o14, 15 o15);
                        if t_e == 0 { st_v2_f32!(ml_e.add(slot_idx * 2), m_r0, l_r0); }
                    }
                    let rb_i = ra + 8;
                    if rb_i < qc_e {
                        let row = rb_e + rb_i / grp_e;
                        let head = kvh_e * grp_e + rb_i % grp_e;
                        let slot_idx = (row * heads_e + head) * ns_e + s_local as usize;
                        let dst = part_e.add(slot_idx * FW_HEAD_DIM + 2 * t_e);
                        macro_rules! st1 {
                            ($($ni:literal $o:ident),*) => {$( st_v2_f32!(dst.add($ni * 8), $o[2], $o[3]); )*};
                        }
                        st1!(0 o0, 1 o1, 2 o2, 3 o3, 4 o4, 5 o5, 6 o6, 7 o7, 8 o8, 9 o9, 10 o10, 11 o11, 12 o12, 13 o13, 14 o14, 15 o15);
                        if t_e == 0 { st_v2_f32!(ml_e.add(slot_idx * 2), m_r1, l_r1); }
                    }
                }
                if s_local + 1 < load_u32(c_base + 56) {
                    core::arch::nvptx::_syncthreads();
                }
                s_local += 1;
            }
        }
    }
}

/// Deterministic Split-KV reduction (`FLASH_WINDOW_REDUCE`): folds `0..num_splits` in ascending order onto the
/// exact attention sink (`m = rn(sink), l = 1.0, O = 0.0`) or prior L2-band `state`, applies `sigmoid(rn(gate))` on
/// the final band, and writes BF16-rounded output (0 bit diffs vs single-band reduction).
///
/// SAFETY (caller contract, `resident_launch_flash_window_reduce`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `FLASH_WINDOW_REDUCE` recorded by `Backend::flash_window_split_into`.
#[inline(always)]
pub(crate) unsafe fn replay_reduce(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized read-only Command; wrong kind traps before reading payload words.
    // - Device re-checks `rows`, `heads`, `num_splits`, `threads == 128`, `grid == rows * heads`, and band state flags.
    // - Every thread `d = tid < 128` reads only `partials[(block * num_splits + s) * 128 + d]`,
    //   `ml[(block * num_splits + s) * 2 .. +2]`, `sink[block]`, `gate[block * 128 + d]`, optional `state`, and writes
    //   either `state` (intermediate bands) or `output[block * 128 + d]` (final band) within host-checked disjoint extents.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FLASH_WINDOW_REDUCE { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let partials = *w as *const f32;
        let sink = *w.add(1) as *const f32;
        let rows = *w.add(4) as usize;
        let heads = *w.add(5) as usize;
        let num_splits = *w.add(6) as usize;
        let r_flags = *w.add(FW_REDUCE_FLAGS_WORD);
        let state_init = *w.add(FW_REDUCE_STATE_WORD) as *const f32;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let grid = core::arch::nvptx::_grid_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        if rows == 0 || heads == 0 || num_splits == 0 || threads != FW_THREADS || grid != rows * heads
            || (r_flags & !(FW_REDUCE_LOAD_STATE | FW_REDUCE_STORE_STATE)) != 0
            || (r_flags != 0 && state_init.is_null())
        {
            core::arch::nvptx::trap();
        }
        let ml_base = partials.add(rows * heads * num_splits * FW_HEAD_DIM);
        let base_slot = block * num_splits;
        let (mut m, mut l, mut acc) = if (r_flags & FW_REDUCE_LOAD_STATE) != 0 {
            let sml = state_init.add(rows * heads * FW_HEAD_DIM);
            (*sml.add(block * 2), *sml.add(block * 2 + 1), *state_init.add(block * FW_HEAD_DIM + tid))
        } else {
            (rn(*sink.add(block)), 1.0f32, 0.0f32)
        };
        macro_rules! fold {
            ($m_s:expr, $l_s:expr, $o_s:expr) => {{
                let (m_s, l_s): (f32, f32) = ($m_s, $l_s);
                if l_s != 0.0f32 {
                    let o_s: f32 = $o_s;
                    if m_s > m {
                        let alpha = exp_nonpos!(m - m_s);
                        acc = acc * alpha + o_s;
                        l = l * alpha + l_s;
                        m = m_s;
                    } else {
                        let beta = exp_nonpos!(m_s - m);
                        acc += o_s * beta;
                        l += l_s * beta;
                    }
                }
            }};
        }
        const REDUCE_AHEAD: usize = 8;
        const REDUCE_HINT_TURNS: usize = 4;
        macro_rules! hint_turn {
            ($first:expr) => {{
                let first: usize = $first;
                if first + REDUCE_AHEAD <= num_splits {
                    let hint_slot = base_slot + first;
                    core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) ml_base.add(hint_slot * 2) as u64, options(nostack));
                    let hint_part = partials.add(hint_slot * FW_HEAD_DIM + tid);
                    let mut k = 0usize;
                    while k < REDUCE_AHEAD {
                        core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) hint_part.add(k * FW_HEAD_DIM) as u64, options(nostack));
                        k += 1;
                    }
                }
            }};
        }
        let mut turn = 1usize;
        while turn < REDUCE_HINT_TURNS {
            hint_turn!(turn * REDUCE_AHEAD);
            turn += 1;
        }
        let mut s = 0usize;
        while s + REDUCE_AHEAD <= num_splits {
            hint_turn!(s + REDUCE_HINT_TURNS * REDUCE_AHEAD);
            let slot = base_slot + s;
            let ml = ml_base.add(slot * 2);
            let part = partials.add(slot * FW_HEAD_DIM + tid);
            let (m0, l0, o0) = (*ml, *ml.add(1), *part);
            let (m1, l1, o1) = (*ml.add(2), *ml.add(3), *part.add(FW_HEAD_DIM));
            let (m2, l2, o2) = (*ml.add(4), *ml.add(5), *part.add(2 * FW_HEAD_DIM));
            let (m3, l3, o3) = (*ml.add(6), *ml.add(7), *part.add(3 * FW_HEAD_DIM));
            let (m4, l4, o4) = (*ml.add(8), *ml.add(9), *part.add(4 * FW_HEAD_DIM));
            let (m5, l5, o5) = (*ml.add(10), *ml.add(11), *part.add(5 * FW_HEAD_DIM));
            let (m6, l6, o6) = (*ml.add(12), *ml.add(13), *part.add(6 * FW_HEAD_DIM));
            let (m7, l7, o7) = (*ml.add(14), *ml.add(15), *part.add(7 * FW_HEAD_DIM));
            fold!(m0, l0, o0); fold!(m1, l1, o1); fold!(m2, l2, o2); fold!(m3, l3, o3);
            fold!(m4, l4, o4); fold!(m5, l5, o5); fold!(m6, l6, o6); fold!(m7, l7, o7);
            s += REDUCE_AHEAD;
        }
        while s < num_splits {
            let slot = base_slot + s;
            fold!(*ml_base.add(slot * 2), *ml_base.add(slot * 2 + 1), *partials.add(slot * FW_HEAD_DIM + tid));
            s += 1;
        }
        let off = block * FW_HEAD_DIM + tid;
        if (ld_cmd_u64!(w, FW_REDUCE_FLAGS_WORD) & FW_REDUCE_STORE_STATE) != 0 {
            let st_e = ld_cmd_u64!(w, FW_REDUCE_STATE_WORD) as *mut f32;
            let sml_e = st_e.add(grid * FW_HEAD_DIM);
            *st_e.add(off) = acc;
            if tid == 0 { st_v2_f32!(sml_e.add(block * 2), m, l); }
        } else {
            let gate_e = ld_cmd_u64!(w, 2) as *const f32;
            let out_e = ld_cmd_u64!(w, 3) as *mut f32;
            let inv = 1.0f32 / l;
            *out_e.add(off) = rn(rn(acc * inv) * sigmoid(rn(*gate_e.add(off))));
        }
    }
}
