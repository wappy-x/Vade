//! Lossless `bf16_panels` (`Vade-BF11-BP`, 11.15 bpw) PTX fetch, bit-parallel decode, and streaming-store macros
//! shared by `plain_tc_rows` and `plain_tc_big`.
//!
//! Pure macros expanding inside the caller's existing checked `unsafe` block (0 additional `unsafe` sites).
//! Reuses the exact Vade-BF11-BP encoding registered by `register_bf16_panels` (`0 B` extra VRAM):
//! each 64-column chunk (4 tiles = 32 weights/lane) stores `CHUNK_WORDS = 352` words (`1,408 B` = 11 cache lines,
//! `-30%` vs `2,048 B` raw BF16) with per-band top-7 LUTs and rare coalesced escape pool (plus `is_raw_exp` fallback).

#[macro_export]
macro_rules! tc_prmt_b32 {
    ($lo:expr, $hi:expr, $mask:literal) => {{
        let w: u32;
        core::arch::asm!(
            "prmt.b32 {w}, {lo}, {hi}, {m};",
            w = out(reg32) w, lo = in(reg32) $lo, hi = in(reg32) $hi, m = const $mask,
            options(pure, nomem, nostack)
        );
        w
    }};
}

#[macro_export]
macro_rules! tc_lop3_b32 {
    ($a:expr, $b:expr, $c:expr, $imm:literal) => {{
        let d: u32;
        core::arch::asm!(
            "lop3.b32 {d}, {a}, {b}, {c}, {imm};",
            d = out(reg32) d, a = in(reg32) $a, b = in(reg32) $b, c = in(reg32) $c,
            imm = const $imm, options(pure, nomem, nostack)
        );
        d
    }};
}

#[macro_export]
macro_rules! tc_unpack_bf16_bp {
    ($sm4:expr, $er4:expr) => {{
        let (sm4, er4): (u32, u32) = ($sm4, $er4);
        let lo4 = tc_lop3_b32!(er4, sm4, 0x8080_8080u32, 0xE4);
        let hi4 = tc_lop3_b32!(sm4, er4, 0x8080_8080u32, 0xE4);
        (tc_prmt_b32!(lo4, hi4, 0x5140), tc_prmt_b32!(lo4, hi4, 0x7362))
    }};
}

#[macro_export]
macro_rules! tc_fetch_bp_sm0 {
    ($chunk_band:expr, $lane:expr, $ch:expr, $evict:expr) => {{
        let p = ($chunk_band).add(($ch as usize) * crate::bf16_panels::CHUNK_WORDS + ($lane) * 4);
        let (a0, a1, a2, a3): (u32, u32, u32, u32);
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a},{b},{c},{d}}}, [{p}], {h};",
            a = out(reg32) a0, b = out(reg32) a1, c = out(reg32) a2, d = out(reg32) a3,
            p = in(reg64) p, h = in(reg64) $evict, options(readonly, nostack)
        );
        [a0, a1, a2, a3]
    }};
}

#[macro_export]
macro_rules! tc_fetch_bp_rest {
    ($chunk_band:expr, $lane:expr, $ch:expr, $evict:expr) => {{
        let base = ($chunk_band).add(($ch as usize) * crate::bf16_panels::CHUNK_WORDS);
        let l: usize = $lane;
        let (b0, b1, b2, b3): (u32, u32, u32, u32);
        let (e0, e1, eh): (u32, u32, u32);
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a},{b},{c},{d}}}, [{p}], {h};",
            a = out(reg32) b0, b = out(reg32) b1, c = out(reg32) b2, d = out(reg32) b3,
            p = in(reg64) base.add(crate::bf16_panels::CHUNK_SM1_OFF + l * 4), h = in(reg64) $evict, options(readonly, nostack)
        );
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.L2::128B.v2.u32 {{{a},{b}}}, [{p}], {h};",
            a = out(reg32) e0, b = out(reg32) e1,
            p = in(reg64) base.add(crate::bf16_panels::CHUNK_E3_LO_OFF + l * 2), h = in(reg64) $evict, options(readonly, nostack)
        );
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.L2::128B.u32 {a}, [{p}], {h};",
            a = out(reg32) eh,
            p = in(reg64) base.add(crate::bf16_panels::CHUNK_E3_HI_OFF + l), h = in(reg64) $evict, options(readonly, nostack)
        );
        ([b0, b1, b2, b3], e0, e1, eh)
    }};
}

#[macro_export]
macro_rules! tc_decode_bp_tile {
    ($sm_lo:expr, $sm_hi:expr, $lo0:expr, $hi0:expr, $lo1:expr, $hi1:expr,
     $lut0:expr, $lut1:expr, $is_raw:expr, $words:expr, $esc_off:expr, $eq0:expr, $esc_slot:expr, $ch:expr, $h0:literal, $lane:expr) => {{
        let (mut er0, mut er1): (u32, u32);
        if $is_raw {
            let esc_base = ($words.add(crate::tc_rows::TC_PANEL_WORD).read_volatile() as *const u32).add($esc_off as usize);
            let rbase = ($eq0 as usize) * crate::bf16_panels::ESC_STEP_WORDS
                + ($ch as usize) * crate::bf16_panels::RAW_EXP_CHUNK_WORDS
                + if $h0 < 4 { ($lane) * 4 + $h0 } else { 128 + ($lane) * 4 + ($h0 - 4) };
            er0 = crate::tc_ptx::ld_global_nc_u32(esc_base.add(rbase));
            er1 = crate::tc_ptx::ld_global_nc_u32(esc_base.add(rbase + 1));
        } else {
            let sel0 = tc_lop3_b32!(($lo0) & 0x3333u32, $hi0, 0x4444u32, 0xF8);
            let sel1 = tc_lop3_b32!(($lo1) & 0x3333u32, $hi1, 0x4444u32, 0xF8);
            core::arch::asm!("prmt.b32 {d}, {a}, {b}, {s};",
                d = out(reg32) er0, a = in(reg32) $lut0, b = in(reg32) $lut1, s = in(reg32) sel0,
                options(pure, nomem, nostack));
            core::arch::asm!("prmt.b32 {d}, {a}, {b}, {s};",
                d = out(reg32) er1, a = in(reg32) $lut0, b = in(reg32) $lut1, s = in(reg32) sel1,
                options(pure, nomem, nostack));
            let esc0 = sel0.wrapping_add(0x1111u32) & 0x8888u32;
            let esc1 = sel1.wrapping_add(0x1111u32) & 0x8888u32;
            if (esc0 | esc1) != 0 {
                let esc_base = ($words.add(crate::tc_rows::TC_PANEL_WORD).read_volatile() as *const u32).add($esc_off as usize);
                let slot: u32 = $esc_slot;
                let mut pos = crate::tc_ptx::load_u32(slot);
                let mut m0 = esc0;
                while m0 != 0 {
                    let shift = (m0.trailing_zeros() - 3) << 1;
                    let word = crate::tc_ptx::ld_global_nc_u32(
                        esc_base.add(((pos >> 2) as usize) * crate::bf16_panels::ESC_STEP_WORDS + ($lane))
                    );
                    let b = (word >> ((pos & 3) << 3)) & 0xFF;
                    er0 = (er0 & !(0xFFu32 << shift)) | (b << shift);
                    pos += 1;
                    m0 &= m0 - 1;
                }
                let mut m1 = esc1;
                while m1 != 0 {
                    let shift = (m1.trailing_zeros() - 3) << 1;
                    let word = crate::tc_ptx::ld_global_nc_u32(
                        esc_base.add(((pos >> 2) as usize) * crate::bf16_panels::ESC_STEP_WORDS + ($lane))
                    );
                    let b = (word >> ((pos & 3) << 3)) & 0xFF;
                    er1 = (er1 & !(0xFFu32 << shift)) | (b << shift);
                    pos += 1;
                    m1 &= m1 - 1;
                }
                crate::tc_ptx::store_u32(slot, pos);
            }
        }
        let (a0, a1) = tc_unpack_bf16_bp!($sm_lo, er0);
        let (a2, a3) = tc_unpack_bf16_bp!($sm_hi, er1);
        [a0, a1, a2, a3]
    }};
}

/// 8-byte global read-only load with L2 cache hint (`ld.global.nc.L2::cache_hint.v2.f32`).
#[macro_export]
macro_rules! tc_ld_nc_v2_f32_hint {
    ($p:expr, $q:expr) => {{
        let (a, b): (f32, f32);
        core::arch::asm!("ld.global.nc.L2::cache_hint.v2.f32 {{{a}, {b}}}, [{p}], {q};",
            a = out(reg32) a, b = out(reg32) b, p = in(reg64) $p, q = in(reg64) $q, options(readonly, nostack));
        (a, b)
    }};
}

/// 8-byte global store with L2 eviction hint (`st.global.L2::cache_hint.v2.f32`), falling back to 4-byte hint stores.
#[macro_export]
macro_rules! tc_st_pair_hint {
    ($dst:expr, $pair_ok:expr, $v0:expr, $v1:expr, $q:expr) => {{
        let dst: *mut f32 = $dst;
        let (v0, v1, q): (f32, f32, u64) = ($v0, $v1, $q);
        if $pair_ok && ((dst as usize) & 7) == 0 {
            core::arch::asm!("st.global.L2::cache_hint.v2.f32 [{p}], {{{a}, {b}}}, {q};",
                p = in(reg64) dst, a = in(reg32) v0, b = in(reg32) v1, q = in(reg64) q, options(nostack));
        } else {
            crate::tc_ptx::st_global_f32_hint(dst, v0, q);
            if $pair_ok { crate::tc_ptx::st_global_f32_hint(dst.add(1), v1, q); }
        }
    }};
}

/// 4-byte global u32 store with L2 eviction hint (`st.global.L2::cache_hint.u32`).
#[macro_export]
macro_rules! tc_st_u32_hint {
    ($dst:expr, $v:expr, $q:expr) => {{
        core::arch::asm!("st.global.L2::cache_hint.u32 [{p}], {v}, {q};",
            p = in(reg64) $dst, v = in(reg32) $v, q = in(reg64) $q, options(nostack));
    }};
}

/// 8-byte global u32x2 store with L2 eviction hint (`st.global.L2::cache_hint.v2.u32`), falling back to 4-byte stores.
#[macro_export]
macro_rules! tc_st_v2_u32_hint {
    ($dst:expr, $v0:expr, $v1:expr, $q:expr) => {{
        let dst: *mut u32 = $dst;
        let (v0, v1, q): (u32, u32, u64) = ($v0, $v1, $q);
        if ((dst as usize) & 7) == 0 {
            core::arch::asm!("st.global.L2::cache_hint.v2.u32 [{p}], {{{a}, {b}}}, {q};",
                p = in(reg64) dst, a = in(reg32) v0, b = in(reg32) v1, q = in(reg64) q, options(nostack));
        } else {
            tc_st_u32_hint!(dst, v0, q);
            tc_st_u32_hint!(dst.add(1), v1, q);
        }
    }};
}

/// 16-byte global f32x4 store with L2 eviction hint (`st.global.L2::cache_hint.v4.f32`), falling back to v2/scalar stores.
#[macro_export]
macro_rules! tc_st_v4_f32_hint {
    ($dst:expr, $v0:expr, $v1:expr, $v2:expr, $v3:expr, $q:expr) => {{
        let dst: *mut f32 = $dst;
        let (v0, v1, v2, v3, q): (f32, f32, f32, f32, u64) = ($v0, $v1, $v2, $v3, $q);
        if ((dst as usize) & 15) == 0 {
            core::arch::asm!("st.global.L2::cache_hint.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}}, {q};",
                p = in(reg64) dst, a = in(reg32) v0, b = in(reg32) v1, c = in(reg32) v2, d = in(reg32) v3,
                q = in(reg64) q, options(nostack));
        } else {
            tc_st_pair_hint!(dst, true, v0, v1, q);
            tc_st_pair_hint!(dst.add(2), true, v2, v3, q);
        }
    }};
}

/// Lossless Vade-BF11-BP inner K-loop for `PLAIN_TC_ROWS` Path 4 (`M <= 32`):
/// - Reuses registered Vade-BF11-BP panels (`0 B` extra VRAM, `-30%` DRAM vs raw BF16).
/// - Per-warp escape cursors sit in tiny SMEM (`slices * 128 B/warp`), keeping register pressure flat (`<= 96` regs).
/// - Multi-slab pre-staged X (`4` slabs = `256` cols for `M <= 16`, `3` slabs = `192` cols for `M = 17..=32`) +
///   split `sm0`/`rest` software pipeline + 3-chunk-ahead L2 prefetch hide DRAM latency with zero W SMEM.
#[macro_export]
macro_rules! tc_path4_panels {
    (
        $words:expr, $rows:expr, $k:expr,
        $warps:expr, $tile_n:expr, $warp:expr, $lane:expr,
        $part_slabs:expr, $x_stride:expr, $x_off0:expr, $evict:expr,
        $c0:ident, $has_m1:expr, $c1:ident, $has_m2:expr, $c2:ident, $has_m3:expr, $c3:ident
    ) => {{
        let panels_base = $words.add(crate::tc_rows::TC_PANEL_WORD).read_volatile() as *const u32;
        let meta_word = $words.add(crate::tc_rows::TC_PANEL_WORD + 1).read_volatile();
        let first_band = (meta_word & 0xffff_ffff) as usize;
        let slices_want = (meta_word >> 32) as usize;
        let chunks = $k / crate::bf16_panels::CHUNK_COLS;
        let chunks_u = chunks as u32;
        let hdr = crate::tc_ptx::ld_global_nc_v4_u32(panels_base);
        let (meta_off, chunks_off, esc_off, hdr3) = (hdr[0] as usize, hdr[1] as usize, hdr[2], hdr[3]);
        if ((hdr3 & 0xFFFF) as usize) != chunks || ((hdr3 >> 16) as usize) != slices_want {
            core::arch::nvptx::trap();
        }
        let bands = $rows.div_ceil(16);
        let own = ($tile_n * $warps + $warp).min(bands - 1);
        let abs_band = first_band + own;
        let meta_ptr = panels_base.add(meta_off + abs_band * crate::bf16_panels::BAND_META_WORDS);
        let meta0 = crate::tc_ptx::ld_global_nc_v4_u32(meta_ptr);
        let (lut0, lut1, esc_meta) = (meta0[0], meta0[1], meta0[2]);
        let is_raw_exp = (esc_meta & 1) != 0;
        let eq0 = esc_meta >> 1;
        let slices_mask = (slices_want - 1) as u32;
        let esc_smem = $part_slabs * $x_stride + ($warp * slices_want * 128) as u32;
        if !is_raw_exp {
            let mut q = 0usize;
            while q < slices_want {
                let rel = if q == 0 {
                    0u32
                } else if q == 1 {
                    meta0[3] & 0xFFFF
                } else if q == 2 {
                    meta0[3] >> 16
                } else {
                    let w = crate::tc_ptx::ld_global_nc_u32(meta_ptr.add(4 + ((q - 3) >> 1)));
                    (w >> (((q - 3) & 1) * 16)) & 0xFFFF
                };
                crate::tc_ptx::store_u32(esc_smem + ((q * 32 + $lane) * 4) as u32, (eq0 + rel) << 2);
                q += 1;
            }
        }
        let chunk_band = panels_base.add(chunks_off + abs_band * chunks * crate::bf16_panels::CHUNK_WORDS);
        macro_rules! sync_p4 { () => { if core::arch::nvptx::_block_dim_x() == 32 { crate::ptx::warp_sync(); } else { core::arch::nvptx::_syncthreads(); } }; }
        macro_rules! prefetch_p4 {
            ($ch:expr) => {{
                let ch_p: u32 = $ch;
                if $lane < 11 && ch_p < chunks_u {
                    core::arch::asm!(
                        "prefetch.global.L2 [{a}];",
                        a = in(reg64) chunk_band.add((ch_p as usize) * crate::bf16_panels::CHUNK_WORDS + $lane * 32) as u64,
                        options(readonly, nostack)
                    );
                }
            }};
        }
        macro_rules! stage_x_p4 {
            ($ch_start:expr) => {{
                let cs_x: u32 = $ch_start;
                let in1 = $words.add(1).read_volatile() as *const u32;
                let k_v = $words.add(4).read_volatile() as usize;
                let m_v = $words.add(5).read_volatile() as usize;
                let in2 = $words.add(12).read_volatile() as *const u32;
                let half = if in2.is_null() { k_v } else { k_v >> 1 };
                let tid_x = core::arch::nvptx::_thread_idx_x() as usize;
                let (xr0, xc) = (tid_x >> 3, tid_x & 7);
                let x_step = (core::arch::nvptx::_block_dim_x() as usize) >> 3;
                let x_str = if m_v <= 16 { 2048u32 } else { 4096u32 };
                let max_s = if m_v <= 16 { 4u32 } else { 3u32 };
                let slabs_now = (chunks_u - cs_x).min(max_s);
                let mut si = 0u32;
                while si < slabs_now {
                    let k_col = ((cs_x + si) as usize) * 64 + xc * 8;
                    let (src_x, x_col) = if k_col < half { (in1, k_col) } else { (in2, k_col - half) };
                    let dst_base = si * x_str;
                    let mut rl = xr0;
                    while rl < m_v {
                        let dst = dst_base + ((rl * 128 + ((xc ^ (rl & 7)) * 16)) as u32);
                        crate::ptx::copy16_cached(dst, src_x.add(rl * (half >> 1) + (x_col >> 1)));
                        rl += x_step;
                    }
                    si += 1;
                }
                crate::ptx::commit();
            }};
        }
        let mut sm0 = tc_fetch_bp_sm0!(chunk_band, $lane, 0u32, $evict);
        let (mut sm1, mut w0, mut w1, mut w2) = tc_fetch_bp_rest!(chunk_band, $lane, 0u32, $evict);
        prefetch_p4!(1u32); prefetch_p4!(2u32); prefetch_p4!(3u32);
        stage_x_p4!(0u32);
        crate::tc_ptx::cp_async_wait_zero();
        sync_p4!();

        let (mut ch, mut part_end, mut x_slab) = (0u32, $part_slabs, $x_off0);
        while ch < chunks_u {
            if ch == part_end {
                sync_p4!();
                stage_x_p4!(ch);
                crate::tc_ptx::cp_async_wait_zero();
                sync_p4!();
                part_end = ch + if $has_m2 { 3u32 } else { 4u32 };
                x_slab = $x_off0;
            }
            prefetch_p4!(ch + 4);
            let q_slot = esc_smem + (((ch & slices_mask) * 32 + ($lane as u32)) << 2);
            macro_rules! mma_p4 {
                ($a:expr, $xm:literal) => {{
                    let a: [u32; 4] = $a;
                    let xb: u32 = x_slab;
                    $c0 = crate::tc_ptx::mma_m16n8k16_bf16(a, [crate::tc_ptx::load_u32(xb ^ $xm), crate::tc_ptx::load_u32(xb ^ ($xm ^ 16))], $c0);
                    if $has_m1 { $c1 = crate::tc_ptx::mma_m16n8k16_bf16(a, [crate::tc_ptx::load_u32((xb ^ $xm) + 1024), crate::tc_ptx::load_u32((xb ^ ($xm ^ 16)) + 1024)], $c1); }
                    if $has_m2 { $c2 = crate::tc_ptx::mma_m16n8k16_bf16(a, [crate::tc_ptx::load_u32((xb ^ $xm) + 2048), crate::tc_ptx::load_u32((xb ^ ($xm ^ 16)) + 2048)], $c2); }
                    if $has_m3 { $c3 = crate::tc_ptx::mma_m16n8k16_bf16(a, [crate::tc_ptx::load_u32((xb ^ $xm) + 3072), crate::tc_ptx::load_u32((xb ^ ($xm ^ 16)) + 3072)], $c3); }
                }};
            }
            let nxt_c = if ch + 1 < chunks_u { ch + 1 } else { ch };
            let a0 = tc_decode_bp_tile!(sm0[0], sm0[1], w0, w2, w0 >> 2, w2 >> 1, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, ch, 0, $lane);
            mma_p4!(a0, 0u32);
            let a1 = tc_decode_bp_tile!(sm0[2], sm0[3], w0 >> 16, w2 << 2, w0 >> 18, w2 << 1, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, ch, 2, $lane);
            mma_p4!(a1, 32u32);
            sm0 = tc_fetch_bp_sm0!(chunk_band, $lane, nxt_c, $evict);
            let a2 = tc_decode_bp_tile!(sm1[0], sm1[1], w1, w2 >> 16, w1 >> 2, w2 >> 17, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, ch, 4, $lane);
            mma_p4!(a2, 64u32);
            let a3 = tc_decode_bp_tile!(sm1[2], sm1[3], w1 >> 16, w2 >> 14, w1 >> 18, w2 >> 15, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, ch, 6, $lane);
            mma_p4!(a3, 96u32);
            let nr = tc_fetch_bp_rest!(chunk_band, $lane, nxt_c, $evict);
            sm1 = nr.0; w0 = nr.1; w1 = nr.2; w2 = nr.3;
            ch += 1;
            x_slab += if $has_m2 { 4096u32 } else { 2048u32 };
        }
        let _ = (sm0, sm1, w0, w1, w2);
    }};
}

/// Lossless Vade-BF11-BP inner K-loop for `PLAIN_TC_ROWS` Path 4b (`tc_coop_w_wanted`, `inputs <= 32`, `slices in {4, 8}`):
/// 4 warps share 1 band (`BN = 16`), each decoding its own disjoint chunk `base_ch + warp` (`4` chunks = `256` cols/step)
/// into `8 KB` W SMEM while staging 4 slabs of `X` (`16 KB`), then all 4 warps execute 16 MMAs in strictly ascending K order.
#[macro_export]
macro_rules! tc_coop_w_panels {
    (
        $words:expr, $k:expr, $block:expr, $warp:expr, $lane:expr,
        $m_w:expr, $smem:expr, $x_base:expr, $x_stride:expr, $x_off0:expr, $evict:expr,
        $issue_x:ident, $c0:ident
    ) => {{
        let panels_base = $words.add(crate::tc_rows::TC_PANEL_WORD).read_volatile() as *const u32;
        let meta_word = $words.add(crate::tc_rows::TC_PANEL_WORD + 1).read_volatile();
        let first_band = (meta_word & 0xffff_ffff) as usize;
        let slices_want = (meta_word >> 32) as usize;
        let chunks = $k / crate::bf16_panels::CHUNK_COLS;
        let hdr = crate::tc_ptx::ld_global_nc_v4_u32(panels_base);
        let (meta_off, chunks_off, esc_off, hdr3) = (hdr[0] as usize, hdr[1] as usize, hdr[2], hdr[3]);
        if ((hdr3 & 0xFFFF) as usize) != chunks || ((hdr3 >> 16) as usize) != slices_want
            || slices_want < 4 || !chunks.is_multiple_of(4)
        {
            core::arch::nvptx::trap();
        }
        let abs_band = first_band + $block;
        let meta_ptr = panels_base.add(meta_off + abs_band * crate::bf16_panels::BAND_META_WORDS);
        let meta0 = crate::tc_ptx::ld_global_nc_v4_u32(meta_ptr);
        let (lut0, lut1, esc_meta) = (meta0[0], meta0[1], meta0[2]);
        let is_raw_exp = (esc_meta & 1) != 0;
        let eq0 = esc_meta >> 1;
        let slices_mask = (slices_want - 1) as u32;
        let esc_smem = $smem + 24_576u32;
        if !is_raw_exp {
            let mut q = $warp;
            while q < slices_want {
                let rel = if q == 0 {
                    0u32
                } else if q == 1 {
                    meta0[3] & 0xFFFF
                } else if q == 2 {
                    meta0[3] >> 16
                } else {
                    let w = crate::tc_ptx::ld_global_nc_u32(meta_ptr.add(4 + ((q - 3) >> 1)));
                    (w >> (((q - 3) & 1) * 16)) & 0xFFFF
                };
                crate::tc_ptx::store_u32(esc_smem + ((q * 32 + $lane) * 4) as u32, (eq0 + rel) << 2);
                q += 4;
            }
        }
        let chunk_band = panels_base.add(chunks_off + abs_band * chunks * crate::bf16_panels::CHUNK_WORDS);
        let lane16 = ($lane * 16) as u32;
        let w_dst = $smem + ($warp * 2048) as u32 + lane16;
        let mut base_ch = 0usize;
        while base_ch < chunks {
            $issue_x($x_base, base_ch);
            $issue_x($x_base + $x_stride, base_ch + 1);
            $issue_x($x_base + 2 * $x_stride, base_ch + 2);
            $issue_x($x_base + 3 * $x_stride, base_ch + 3);
            crate::ptx::commit();
            let my_ch = base_ch + $warp;
            if $lane < 11 && my_ch + 4 < chunks {
                core::arch::asm!(
                    "prefetch.global.L2 [{a}];",
                    a = in(reg64) chunk_band.add((my_ch + 4) * crate::bf16_panels::CHUNK_WORDS + $lane * 32) as u64,
                    options(readonly, nostack)
                );
            }
            let q_slot = esc_smem + ((((my_ch as u32) & slices_mask) * 32 + ($lane as u32)) << 2);
            let sm0 = tc_fetch_bp_sm0!(chunk_band, $lane, my_ch, $evict);
            let (sm1, w0, w1, w2) = tc_fetch_bp_rest!(chunk_band, $lane, my_ch, $evict);
            let a0 = tc_decode_bp_tile!(sm0[0], sm0[1], w0, w2, w0 >> 2, w2 >> 1, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, my_ch, 0, $lane);
            crate::tc_ptx::st_shared_v4(w_dst, a0);
            let a1 = tc_decode_bp_tile!(sm0[2], sm0[3], w0 >> 16, w2 << 2, w0 >> 18, w2 << 1, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, my_ch, 2, $lane);
            crate::tc_ptx::st_shared_v4(w_dst + 512, a1);
            let a2 = tc_decode_bp_tile!(sm1[0], sm1[1], w1, w2 >> 16, w1 >> 2, w2 >> 17, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, my_ch, 4, $lane);
            crate::tc_ptx::st_shared_v4(w_dst + 1024, a2);
            let a3 = tc_decode_bp_tile!(sm1[2], sm1[3], w1 >> 16, w2 >> 14, w1 >> 18, w2 >> 15, lut0, lut1, is_raw_exp, $words, esc_off, eq0, q_slot, my_ch, 6, $lane);
            crate::tc_ptx::st_shared_v4(w_dst + 1536, a3);
            crate::tc_ptx::cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            let mut sub = 0u32;
            while sub < 4 {
                let cur_w = $smem + sub * 2048;
                let x_w = $x_base + sub * $x_stride + ($m_w * 128) as u32;
                $c0 = crate::tc_ptx::mma_m16n8k16_bf16(crate::ptx::load4(cur_w + lane16), [crate::tc_ptx::load_u32(x_w + $x_off0), crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 16))], $c0);
                $c0 = crate::tc_ptx::mma_m16n8k16_bf16(crate::ptx::load4(cur_w + 512 + lane16), [crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 32)), crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 48))], $c0);
                $c0 = crate::tc_ptx::mma_m16n8k16_bf16(crate::ptx::load4(cur_w + 1024 + lane16), [crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 64)), crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 80))], $c0);
                $c0 = crate::tc_ptx::mma_m16n8k16_bf16(crate::ptx::load4(cur_w + 1536 + lane16), [crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 96)), crate::tc_ptx::load_u32(x_w + ($x_off0 ^ 112))], $c0);
                sub += 1;
            }
            if base_ch + 4 < chunks {
                core::arch::nvptx::_syncthreads();
            }
            base_ch += 4;
        }
    }};
}
