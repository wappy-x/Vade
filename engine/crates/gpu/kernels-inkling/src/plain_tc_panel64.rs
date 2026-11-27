//! Tensor-core BF16 dense product over lossless Vade-BF11-BP `bf16_panels` for verify-size batches
//! (`resident_launch_plain_tc_panel64`, kind `PLAIN_TC_PANEL64 = 124`).
//!
//! Computes `Y = X * W^T` for packed BF16 `X: [M, K]` and a registered `bf16_panels` matrix `W: [N, K]`, writing
//! FP32 `Y: [M, N]` (strided, or 1..=8 split outputs). The host routes 33..=128 input rows of matrices with more than
//! `TC_TINY_ROWS_MAX` rows here (`tc_rows::tc_panel64_wanted`): the LM head, `QKV`, and the wide dense matrices.
//!
//! - One warp owns one 16-row band and one 64-input half (`M <= 64`: one half; `65..=128`: two warps a band).
//!   It decodes the band chunk by chunk in registers with the same Vade-BF11-BP decode as `PLAIN_TC_ROWS` Path 4
//!   (`tc_decode_bp_tile!`), so every slice count (1, 4, 8) works, and multiplies each decoded `16 x 16` tile with
//!   up to 8 input tiles (`m16n8k16`, 32 FP32 accumulators a lane).
//! - The block's warps share a 2-slot shared-memory ring of 64-column `X` slabs (`cp.async`, one chunk ahead), one
//!   barrier a chunk. Every warp of the block reaches the same barriers; warps past the last band decode the last
//!   band and store nothing.
//! - Fixed deterministic reduction order: each output is one MMA chain over `k = 0, 16, ..., K - 16` in strictly
//!   ascending order from `+0.0`, the order of every `PLAIN_TC_ROWS` path, so the words equal theirs bit for bit.
//!   Zero FP atomics, zero split-K.

use crate::tc_ptx::{cp_async_wait_zero, l2_evict_first, ld_global_nc_u32, ld_global_nc_v4_u32, load_u32, mma_m16n8k16_bf16, st_global_f32_hint, st_shared_zero16, store_u32};
use crate::tc_rows::{TC_MODE_DENSE, TC_PANEL_WORD, TC_SPLIT_MAX, tc_panel64_admitted, tc_panel64_x_rows};

/// Replay one recorded command of kind `PLAIN_TC_PANEL64`.
/// SAFETY (caller contract, `resident_launch_plain_tc_panel64`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `PLAIN_TC_PANEL64` recorded by `Backend` after `tc_rows::check_tc_dense_form`,
/// `check_tc_pair_form`, or `check_tc_split_form` verified every extent, alignment, and non-overlap, with a
/// registered `bf16_panels` descriptor for the weight rows (`Backend::tc_panels_for`).
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind traps before reading payload words.
    // - Host checked weights, packed BF16 input(s), and single or 1..=8 disjoint even-width split outputs. The device
    //   re-checks mode, panel descriptor and header, `tc_panel64_admitted` (shape, threads, `%dynamic_smem_size`),
    //   alignment, `stride >= rows`, `split_count <= TC_SPLIT_MAX`, and dual-input constraints before any access.
    // - Panel reads stay inside the registered descriptor: bands are clamped to `rows.div_ceil(16) - 1` and the escape
    //   stream reads follow the bounds proof of `tc_decode_bp_tile!`.
    // - Input reads cover rows `< inputs` and columns `< K` of the checked input span(s); output writes cover rows
    //   `< rows` and inputs `< inputs` only, through the checked stride or split table.
    // - Shared memory: `2 * x_rows * 128` bytes of `X` slabs plus `warps * slices * 128` bytes of escape cursors, all
    //   inside the `%dynamic_smem_size` checked at entry.
    // - All threads of a CTA reach every `bar.sync 0` (no thread returns before the last barrier).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_TC_PANEL64 { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let weights_base = *words as *const u32;
        let input = *words.add(1) as *const u32;
        let rows = *words.add(3) as usize;
        let k = *words.add(4) as usize;
        let inputs = *words.add(5) as usize;
        let stride = *words.add(6) as usize;
        let mode = *words.add(7);
        let input2 = *words.add(12) as *const u32;
        let split_count = *words.add(15) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let shared_bytes: u32;
        core::arch::asm!("mov.u32 {bytes}, %dynamic_smem_size;", bytes = out(reg32) shared_bytes);

        let panels_base = *words.add(TC_PANEL_WORD) as *const u32;
        let meta_word = *words.add(TC_PANEL_WORD + 1);
        let slices_want = (meta_word >> 32) as usize;
        if mode != TC_MODE_DENSE || (*words.add(8) | *words.add(10)) != 0 || panels_base.is_null()
            || (panels_base as usize & 15) != 0 || !matches!(slices_want, 1 | 4 | 8)
        {
            core::arch::nvptx::trap();
        }
        if !tc_panel64_admitted(rows, k, inputs, threads, shared_bytes as usize, slices_want)
            || ((weights_base as usize) | (input as usize)) & 15 != 0
            || split_count > TC_SPLIT_MAX || (split_count == 0 && stride < rows)
        {
            core::arch::nvptx::trap();
        }
        if !input2.is_null() && (split_count != 0 || (input2 as usize & 15) != 0 || (k / 2) % 32 != 0) {
            core::arch::nvptx::trap();
        }

        let warps = threads / 32;
        let halves = inputs.div_ceil(64);
        let bands = rows.div_ceil(16);
        if block * warps >= bands * halves { return; }
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, s) = (lane >> 2, lane & 3);
        let vw = block * warps + warp;
        let (band_raw, half) = (vw / halves, vw % halves);
        let live = band_raw < bands;
        let own = if live { band_raw } else { bands - 1 };
        let m0 = half * 64;
        let has1 = m0 + 8 < inputs;
        let has2 = m0 + 16 < inputs;
        let has3 = m0 + 24 < inputs;
        let has4 = m0 + 32 < inputs;
        let has5 = m0 + 40 < inputs;
        let has6 = m0 + 48 < inputs;
        let has7 = m0 + 56 < inputs;

        // Shared layout: slot `q` of `X` at `q * x_slot` (`x_rows` rows of 128 bytes, 16-byte chunks XOR-swizzled by
        // `row & 7`), then each warp's escape cursors (`slices * 32` words).
        let x_rows = tc_panel64_x_rows(inputs);
        let x_slot = (x_rows * 128) as u32;
        let esc_smem = 2 * x_slot + (warp * slices_want * 128) as u32;
        let x_off0 = (g * 128 + ((g & 7) * 16) + s * 4) as u32 + (m0 * 128) as u32;
        let evict = l2_evict_first();

        // Zero the pad rows `inputs..x_rows` of both slots once (their products land only in unstored columns).
        {
            let (xr0, xc) = (tid >> 3, tid & 7);
            let x_step = threads >> 3;
            let mut rl = inputs + xr0;
            while rl < x_rows {
                let dst = (rl * 128 + ((xc ^ (rl & 7)) * 16)) as u32;
                st_shared_zero16(dst);
                st_shared_zero16(x_slot + dst);
                rl += x_step;
            }
        }

        let chunks = k / crate::bf16_panels::CHUNK_COLS;
        let chunks_u = chunks as u32;
        let hdr = ld_global_nc_v4_u32(panels_base);
        let (meta_off, chunks_off, esc_off, hdr3) = (hdr[0] as usize, hdr[1] as usize, hdr[2], hdr[3]);
        if ((hdr3 & 0xFFFF) as usize) != chunks || ((hdr3 >> 16) as usize) != slices_want {
            core::arch::nvptx::trap();
        }
        let first_band = (meta_word & 0xffff_ffff) as usize;
        let abs_band = first_band + own;
        let meta_ptr = panels_base.add(meta_off + abs_band * crate::bf16_panels::BAND_META_WORDS);
        let meta0 = ld_global_nc_v4_u32(meta_ptr);
        let (lut0, lut1, esc_meta) = (meta0[0], meta0[1], meta0[2]);
        let is_raw_exp = (esc_meta & 1) != 0;
        let eq0 = esc_meta >> 1;
        let slices_mask = (slices_want - 1) as u32;
        let esc_lane = panels_base.add(esc_off as usize + lane);
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
                    let w = ld_global_nc_u32(meta_ptr.add(4 + ((q - 3) >> 1)));
                    (w >> (((q - 3) & 1) * 16)) & 0xFFFF
                };
                store_u32(esc_smem + ((q * 32 + lane) * 4) as u32, (eq0 + rel) << 2);
                core::arch::asm!(
                    "prefetch.global.L1 [{a}];",
                    a = in(reg64) esc_lane.add(((eq0 + rel) as usize) * crate::bf16_panels::ESC_STEP_WORDS) as u64,
                    options(readonly, nostack)
                );
                q += 1;
            }
        }
        let chunk_band = panels_base.add(chunks_off + abs_band * chunks * crate::bf16_panels::CHUNK_WORDS);

        macro_rules! prefetch_w {
            ($ch:expr) => {{
                let ch_p: u32 = $ch;
                if lane < 11 && ch_p < chunks_u {
                    core::arch::asm!(
                        "prefetch.global.L2 [{a}];",
                        a = in(reg64) chunk_band.add((ch_p as usize) * crate::bf16_panels::CHUNK_WORDS + lane * 32) as u64,
                        options(readonly, nostack)
                    );
                }
            }};
        }
        // Stage the 64 columns of chunk `$ch` of every input row into slot `$slot` (`cp.async`, one commit group).
        macro_rules! stage_x {
            ($slot:expr, $ch:expr) => {{
                let ch_x: u32 = $ch;
                let dst_base: u32 = $slot;
                let in1 = words.add(1).read_volatile() as *const u32;
                let in2 = words.add(12).read_volatile() as *const u32;
                let k_v = words.add(4).read_volatile() as usize;
                let m_v = words.add(5).read_volatile() as usize;
                let half_k = if in2.is_null() { k_v } else { k_v >> 1 };
                let tid_x = core::arch::nvptx::_thread_idx_x() as usize;
                let (xr0, xc) = (tid_x >> 3, tid_x & 7);
                let x_step = (core::arch::nvptx::_block_dim_x() as usize) >> 3;
                let k_col = (ch_x as usize) * 64 + xc * 8;
                let (src_x, x_col) = if k_col < half_k { (in1, k_col) } else { (in2, k_col - half_k) };
                let mut rl = xr0;
                while rl < m_v {
                    let dst = dst_base + ((rl * 128 + ((xc ^ (rl & 7)) * 16)) as u32);
                    crate::ptx::copy16_cached(dst, src_x.add(rl * (half_k >> 1) + (x_col >> 1)));
                    rl += x_step;
                }
                crate::ptx::commit();
            }};
        }

        let (mut c0, mut c1, mut c2, mut c3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c4, mut c5, mut c6, mut c7) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        stage_x!(0u32, 0u32);
        let mut sm0 = tc_fetch_bp_sm0!(chunk_band, lane, 0u32, evict);
        let (mut sm1, mut w0, mut w1, mut w2) = tc_fetch_bp_rest!(chunk_band, lane, 0u32, evict);
        prefetch_w!(1u32); prefetch_w!(2u32); prefetch_w!(3u32);

        let mut ch = 0u32;
        while ch < chunks_u {
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            if ch + 1 < chunks_u { stage_x!(((ch + 1) & 1) * x_slot, ch + 1); }
            prefetch_w!(ch + 4);
            let q_slot = esc_smem + (((ch & slices_mask) * 32 + (lane as u32)) << 2);
            let mut pos = if is_raw_exp { 0u32 } else { load_u32(q_slot) };
            if !is_raw_exp {
                // Look-ahead: the next chunk's slice will read steps `q` and `q + 1` of its stream (both in bounds).
                let nq = load_u32(esc_smem + ((((ch + 1) & slices_mask) * 32 + (lane as u32)) << 2));
                core::arch::asm!(
                    "prefetch.global.L1 [{a}];",
                    a = in(reg64) esc_lane.add((((nq >> 2) + 1) as usize) * crate::bf16_panels::ESC_STEP_WORDS) as u64,
                    options(readonly, nostack)
                );
            }
            let xb = (ch & 1) * x_slot;
            macro_rules! mma_tile {
                ($a:expr, $xm:literal) => {{
                    let a: [u32; 4] = $a;
                    let p0: u32 = xb + (x_off0 ^ $xm);
                    let p1: u32 = xb + (x_off0 ^ ($xm ^ 16));
                    c0 = mma_m16n8k16_bf16(a, [load_u32(p0), load_u32(p1)], c0);
                    if has1 { c1 = mma_m16n8k16_bf16(a, [load_u32(p0 + 1024), load_u32(p1 + 1024)], c1); }
                    if has2 { c2 = mma_m16n8k16_bf16(a, [load_u32(p0 + 2048), load_u32(p1 + 2048)], c2); }
                    if has3 { c3 = mma_m16n8k16_bf16(a, [load_u32(p0 + 3072), load_u32(p1 + 3072)], c3); }
                    if has4 { c4 = mma_m16n8k16_bf16(a, [load_u32(p0 + 4096), load_u32(p1 + 4096)], c4); }
                    if has5 { c5 = mma_m16n8k16_bf16(a, [load_u32(p0 + 5120), load_u32(p1 + 5120)], c5); }
                    if has6 { c6 = mma_m16n8k16_bf16(a, [load_u32(p0 + 6144), load_u32(p1 + 6144)], c6); }
                    if has7 { c7 = mma_m16n8k16_bf16(a, [load_u32(p0 + 7168), load_u32(p1 + 7168)], c7); }
                }};
            }
            let nxt_c = if ch + 1 < chunks_u { ch + 1 } else { ch };
            let a0 = tc_decode_bp_tile!(sm0[0], sm0[1], w0, w2, w0 >> 2, w2 >> 1, lut0, lut1, is_raw_exp, words, esc_off, eq0, esc_lane, pos, ch, 0, lane);
            mma_tile!(a0, 0u32);
            let a1 = tc_decode_bp_tile!(sm0[2], sm0[3], w0 >> 16, w2 << 2, w0 >> 18, w2 << 1, lut0, lut1, is_raw_exp, words, esc_off, eq0, esc_lane, pos, ch, 2, lane);
            mma_tile!(a1, 32u32);
            sm0 = tc_fetch_bp_sm0!(chunk_band, lane, nxt_c, evict);
            let a2 = tc_decode_bp_tile!(sm1[0], sm1[1], w1, w2 >> 16, w1 >> 2, w2 >> 17, lut0, lut1, is_raw_exp, words, esc_off, eq0, esc_lane, pos, ch, 4, lane);
            mma_tile!(a2, 64u32);
            let a3 = tc_decode_bp_tile!(sm1[2], sm1[3], w1 >> 16, w2 >> 14, w1 >> 18, w2 >> 15, lut0, lut1, is_raw_exp, words, esc_off, eq0, esc_lane, pos, ch, 6, lane);
            mma_tile!(a3, 96u32);
            if !is_raw_exp { store_u32(q_slot, pos); }
            let nr = tc_fetch_bp_rest!(chunk_band, lane, nxt_c, evict);
            sm1 = nr.0; w0 = nr.1; w1 = nr.2; w2 = nr.3;
            ch += 1;
        }
        let _ = (sm0, sm1, w0, w1, w2);

        if !live { return; }
        // Lane `(g, s)` holds rows `first_row + g` (`[0]`, `[1]`) and `first_row + g + 8` (`[2]`, `[3]`) of inputs
        // `m0 + 8 j + 2 s` and `+ 1` for tile `j`.
        let output = words.add(2).read_volatile() as *mut f32;
        let first_row = own * 16;
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
        let (p0, s0) = resolve(first_row + g);
        let (p1, s1) = resolve(first_row + g + 8);
        let t0 = m0 + 2 * s;
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
        emit_tile!(0, c0);
        if has1 { emit_tile!(8, c1); }
        if has2 { emit_tile!(16, c2); }
        if has3 { emit_tile!(24, c3); }
        if has4 { emit_tile!(32, c4); }
        if has5 { emit_tile!(40, c5); }
        if has6 { emit_tile!(48, c6); }
        if has7 { emit_tile!(56, c7); }
    }
}
