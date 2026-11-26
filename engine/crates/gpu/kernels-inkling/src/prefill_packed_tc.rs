//! Bit-exact W4A16/W8A16 BF16 tensor-core prefill MoE kernel (`resident_launch_prefill_packed_tc`,
//! kind `PREFILL_PACKED_TC = 121`).
//!
//! Three sub-operations selected by payload word `PREFILL_OP_WORD` (18):
//! 1. `PREFILL_OP_PACK`: vectorized 8-float/lane FP32 -> packed BF16 activation packer (`cvt.rn.bf16x2.f32`,
//!    `L2::evict_first` on one-use FP32 reads so BF16 outputs stay hot in L2; bit-exact with `TC_MODE_PACK`).
//! 2. `PREFILL_OP_FOLD`: vectorized BF16 `RouteFold` (`ld.global.nc.L2::cache_hint.v4.u32` `evict_first`,
//!    8 BF16s/lane, `st.global.v4.b32`; bit-exact with `fast_route_fold_half_into`).
//! 3. `PREFILL_OP_GEMM`: `PAIR_M = 128` (two paired route tiles of one expert) x `BN = 128` grouped MoE GEMM
//!    using exact SM121a BF16 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` in strictly ascending
//!    `k = 0, 16, ..., K - 16` order from `+0.0`, producing bit-for-bit identical results to `PLAIN_PACKED_TC_ROWS`:
//!    - `half_m` split (`m <= 64`): distributes odd tail tiles across BOTH `warp_m = 0` (`0..31`) and `warp_m = 1`
//!      (`32..63`), eliminating idle M-warps on unpaired 64-row tiles and small batches.
//!    - `mode == 2`: in-place `pack_expert_tagged` MMA-lane NVFP4 panels. `42,016 B` SMEM (`2 blk/SM`):
//!      `3` A stages (`24,576 B`, `wait_group 1`) + `2` unpacked B stages (`16,384 B`) + `1 KB` `(V1*2^112, V3_half*2^112)`
//!      LUT + `32 B` `META`. Cooperative 1x unpack via zero-IMAD 8-inst `shl+prmt 0xbb99` LUT unpack interleaved
//!      inside `k_sub_0` HMMAs, `.L2::256B` codes + `.L2::128B` scales preserving `PREFILL_M_GROUP = 8` L2 reuse.
//!    - `mode == 1`: shipped tiled MXFP8. `3`-stage `cp.async.cg` (`38,400 B <= 40,960 B`, `wait_group 1`),
//!      zero SMEM LUT, 5-inst exact `cvt.rn.f16x2.e4m3x2` register unpack.
//!    - `mode == 0`: shipped tiled NVFP4. `3`-stage `cp.async.cg` + `1 KB` `shl+prmt` LUT exact unpack.
//!    - Exact `FAST_CAPPED_GATE` LUT gating (`coefficients`), and bank-free dead-SMEM coalesced `st.global.v4.b32`
//!      BF16 epilogues (`128 B/row` for `GateUp`, `256 B/row` `evict_first` for `Down`).

use crate::fast_ops::CAPPED_GATE_TABLE_WORDS;
use crate::fixed_math::rn;
use crate::packed_product::{
    COLS, NVFP4_PANEL_MAGIC_HI, NVFP4_PANEL_MAGIC_LO, PAIRS, WIDE, code_copy, layout, output_row, scale_copy,
};
use crate::prefill_moe::{
    PREFILL_BN, PREFILL_HALF_IN, PREFILL_M_GROUP, PREFILL_OP_FOLD, PREFILL_OP_GEMM, PREFILL_OP_PACK,
    PREFILL_OP_WORD, PREFILL_OUT_BF16, PREFILL_OUT_BF16_GATED, PREFILL_PAIR_M, PREFILL_THREADS,
    fold_admitted, gemm_admitted, pack_admitted,
};
use crate::tc_ptx::{
    cp_async_wait_one, cp_async_wait_zero, cvt_bf16x2, l2_evict_first, ld_global_nc_u32,
    ld_global_nc_v4_u32_hint, ldmatrix_x4, load_u32, mma_m16n8k16_bf16, st_shared_v4, st_shared_zero16,
    store_u16, store_u32,
};
use crate::tc_rows::TC_BM;

/// Bytes between two 16-row MMA tiles of an XOR-swizzled `128 x 32` BF16 stage (`16 * 64`).
const TILE16: u32 = 16 * 64;
/// Unified `42,016 B` static SMEM layout (`2 blk/SM` on SM121a `101,376 B` SMEM):
/// - `mode == 2`: `3` A stages (`24,576 B`) + `2` unpacked B stages (`16,384 B`) + `1 KB` LUT + `32 B` META.
/// - `mode == 0/1`: `3` stages of `12,800 B` (`8,192 B` A + `4,096 B` packed B + `512 B` scales) = `38,400 B <= 40,960 B`.
const A_BYTES: usize = PREFILL_PAIR_M * 32 * 2; // 8,192 B (`128 x 32` BF16)
const A_STAGES: usize = 3;
const B_AT: usize = A_STAGES * A_BYTES; // 24,576 B
const B_BYTES: usize = PREFILL_BN * 32 * 2; // 8,192 B (`128 x 32` BF16 unpacked)
const B_STAGES: usize = 2;
const STAGE32: usize = A_BYTES + PREFILL_BN * (32 + 4); // 12,800 B (mode 0/1)
const LUT_AT: usize = B_AT + B_STAGES * B_BYTES; // 40,960 B (16B aligned)
const LUT_BYTES: usize = 256 * 4; // 1,024 B (`(V1*2^112, V3_half*2^112)` per scale)
const META_AT: usize = LUT_AT + LUT_BYTES; // 41,984 B (16B aligned)
const META_BYTES: usize = 32; // `base, m, tile_n, words_lo, words_hi, half_m`
const BLOCK_BYTES: usize = META_AT + META_BYTES; // 42,016 B

const _: () = assert!(PREFILL_PAIR_M == 128 && PREFILL_BN == 128 && PREFILL_THREADS == 256);
const _: () = assert!(B_AT == 24_576 && LUT_AT == 40_960 && META_AT == 41_984 && BLOCK_BYTES == 42_016);
const _: () = assert!(3 * STAGE32 <= LUT_AT);
/// Dead SMEM `0..40,960` reused by BF16 GateUp (`128 x 72 u16 = 18,432 B`) and Down (`128 x 68 u32 = 34,816 B`).
const _: () = assert!(PREFILL_PAIR_M * 72 * 2 <= LUT_AT && PREFILL_PAIR_M * 68 * 4 <= LUT_AT);

/// Zero-bank-conflict XOR-swizzled byte offset within a 64-byte-stride BF16 tile (`row`, 16B `chunk` in 0..4).
#[inline(always)]
const fn swiz(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Bit-exact 8-instruction zero-IMAD SM121a NVFP4 byte -> BF16x2 unpack (`2` nibbles with scale pair `v1_2`, `v3h_2`
/// pre-scaled by `2^112`). Uses `shl + prmt 0xbb99` on the ALU pipe instead of FMA-pipe `mul.lo.u32`.
macro_rules! unpack_nv_byte {
    ($byte_u32:expr, $v1_2:expr, $v3h_2:expr) => {{
        let out: u32;
        core::arch::asm!(
            "{{ .reg .b8 rb; .reg .b32 h, m3, base, t1, p; \
             cvt.u8.u32 rb, {b}; \
             cvt.rn.f16x2.e2m1x2 h, rb; \
             shl.b32 m3, h, 6; \
             prmt.b32 m3, m3, m3, 0xbb99; \
             lop3.b32 base, {v1}, {v3}, m3, 0xd8; \
             shr.u32 t1, h, 3; \
             and.b32 t1, t1, 0x0f800f80; \
             lop3.b32 p, t1, h, 0x80008000, 0xf8; \
             mul.rn.bf16x2 {out}, base, p; }}",
            out = out(reg32) out,
            b = in(reg32) $byte_u32,
            v1 = in(reg32) $v1_2,
            v3 = in(reg32) $v3h_2,
            options(pure, nomem, nostack),
        );
        out
    }};
}

/// Bit-exact 5-instruction SM121a MXFP8 2-byte -> BF16x2 unpack (for `sc <= 142`, exact two-step for `sc > 142`).
macro_rules! unpack_mx_pair {
    ($pair_u16:expr, $sc_p1:expr, $sc_p2:expr, $two_step:expr) => {{
        let mut out: u32;
        core::arch::asm!(
            "{{ .reg .b32 h, t, b; \
             cvt.rn.f16x2.e4m3x2 h, {p}; \
             shr.u32 t, h, 3; \
             and.b32 t, t, 0x0fff0fff; \
             lop3.b32 b, t, h, 0x80008000, 0xf8; \
             mul.rn.bf16x2 {out}, b, {s1}; }}",
            out = out(reg32) out,
            p = in(reg16) $pair_u16,
            s1 = in(reg32) $sc_p1,
            options(pure, nomem, nostack),
        );
        if $two_step {
            core::arch::asm!(
                "mul.rn.bf16x2 {out}, {out}, {s2};",
                out = inout(reg32) out,
                s2 = in(reg32) $sc_p2,
                options(pure, nomem, nostack),
            );
        }
        out
    }};
}

/// Replay one recorded command of kind `PREFILL_PACKED_TC` (kind 121).
/// SAFETY (caller contract, `resident_launch_prefill_packed_tc`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `PREFILL_PACKED_TC` recorded by `Backend` in `backend/prefill_moe.rs`
/// after `prefill_moe::check_pack`, `prefill_moe::check_fold`, or `prefill_moe::check_gemm` verified all
/// extents, alignments, and disjoint outputs.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind or unknown sub-op traps before any operand dereference.
    // - `PREFILL_OP_PACK`: host checked `check_pack` (16B aligned `input` of `elements` FP32 words and disjoint
    //   16B aligned `output` of `elements / 2` u32 words, `elements % 8 == 0`).
    // - `PREFILL_OP_FOLD`: host checked `check_fold` (`weights[rows * picks]`, 16B aligned BF16
    //   `downs[rows * picks * dim / 2]`, disjoint 16B aligned FP32 `output[rows * dim]`, `dim % 8 == 0`).
    // - `PREFILL_OP_GEMM`: host checked `check_gemm` (contiguous 16B aligned `stack` of `experts * unit_bytes`,
    //   16B aligned packed BF16 `input`, disjoint `output` matching `format`, aligned `tile_meta` and index tables,
    //   and optional `2 * CAPPED_GATE_TABLE_WORDS` FP32 `coefficients` tables built before this launch). Device
    //   re-checks `gemm_admitted`, `half_in`, `total_m_tiles <= max_m_tiles`, paired tile adjacency, `base + m <= all`,
    //   and every table index before use.
    // - All 256 threads of every non-exiting CTA participate in every `_syncthreads()` and `cp.async.wait_group`,
    //   and all 32 lanes of every warp participate in every `ldmatrix` and `mma.sync` (tile guards are warp-uniform).
    // - All shared memory accesses stay strictly inside the CTA's static `BLOCK_BYTES` (`42,016` B) allocation.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PREFILL_PACKED_TC {
            core::arch::nvptx::trap();
        }
        let words = c.payload.as_ptr().cast::<u64>();
        let subop = *words.add(PREFILL_OP_WORD);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;

        // ---------------------------------------------------------------------
        // Sub-op 1: Vectorized FP32 -> packed BF16 activation packer (bit-exact with TC_MODE_PACK).
        // ---------------------------------------------------------------------
        if subop == PREFILL_OP_PACK {
            let input = *words as *const u32;
            let output = *words.add(1) as *mut u32;
            let elements = *words.add(2) as usize;
            if !pack_admitted(elements, input as usize, output as usize, threads) {
                core::arch::nvptx::trap();
            }
            let chunk = block * PREFILL_THREADS + tid;
            if chunk < elements / 8 {
                let evict = l2_evict_first();
                let src = input.add(chunk * 8);
                let lo = ld_global_nc_v4_u32_hint(src, evict);
                let hi = ld_global_nc_v4_u32_hint(src.add(4), evict);
                let p0 = cvt_bf16x2(f32::from_bits(lo[0]), f32::from_bits(lo[1]));
                let p1 = cvt_bf16x2(f32::from_bits(lo[2]), f32::from_bits(lo[3]));
                let p2 = cvt_bf16x2(f32::from_bits(hi[0]), f32::from_bits(hi[1]));
                let p3 = cvt_bf16x2(f32::from_bits(hi[2]), f32::from_bits(hi[3]));
                core::arch::asm!(
                    "st.global.v4.b32 [{p}], {{{a0}, {a1}, {a2}, {a3}}};",
                    p = in(reg64) output.add(chunk * 4),
                    a0 = in(reg32) p0, a1 = in(reg32) p1, a2 = in(reg32) p2, a3 = in(reg32) p3,
                    options(nostack),
                );
            }
            return;
        }

        // ---------------------------------------------------------------------
        // Sub-op 2: Vectorized BF16 RouteFold (bit-exact with fast_route_fold_half_into).
        // ---------------------------------------------------------------------
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        if subop == PREFILL_OP_FOLD {
            let w_ptr = *words as *const f32;
            let downs = *words.add(1) as *const u32;
            let output = *words.add(2) as *mut f32;
            let rows = *words.add(3) as usize;
            let picks = *words.add(4) as usize;
            let dim = *words.add(5) as usize;
            if !fold_admitted(rows, picks, dim, w_ptr as usize, downs as usize, output as usize, threads)
                || block >= rows
            {
                core::arch::nvptx::trap();
            }
            if tid < picks {
                store_u32(smem + (tid * 4) as u32, (*w_ptr.add(block * picks + tid)).to_bits());
            }
            core::arch::nvptx::_syncthreads();
            let evict = l2_evict_first();
            let dim_u32 = dim / 2;
            let d_base = downs.add(block * picks * dim_u32);
            let o_base = output.add(block * dim);
            let mut ch = tid * 8;
            while ch < dim {
                let [mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7] = [0.0f32; 8];
                let mut p = 0usize;
                while p < picks {
                    let wt = f32::from_bits(load_u32(smem + (p * 4) as u32));
                    let [u0, u1, u2, u3] = ld_global_nc_v4_u32_hint(d_base.add(p * dim_u32 + ch / 2), evict);
                    s0 += f32::from_bits(u0 << 16) * wt;
                    s1 += f32::from_bits(u0 & 0xffff_0000) * wt;
                    s2 += f32::from_bits(u1 << 16) * wt;
                    s3 += f32::from_bits(u1 & 0xffff_0000) * wt;
                    s4 += f32::from_bits(u2 << 16) * wt;
                    s5 += f32::from_bits(u2 & 0xffff_0000) * wt;
                    s6 += f32::from_bits(u3 << 16) * wt;
                    s7 += f32::from_bits(u3 & 0xffff_0000) * wt;
                    p += 1;
                }
                core::arch::asm!(
                    "st.global.v4.b32 [{p0}], {{{a0}, {a1}, {a2}, {a3}}}; \
                     st.global.v4.b32 [{p1}], {{{b0}, {b1}, {b2}, {b3}}};",
                    p0 = in(reg64) o_base.add(ch),
                    p1 = in(reg64) o_base.add(ch + 4),
                    a0 = in(reg32) rn(s0), a1 = in(reg32) rn(s1), a2 = in(reg32) rn(s2), a3 = in(reg32) rn(s3),
                    b0 = in(reg32) rn(s4), b1 = in(reg32) rn(s5), b2 = in(reg32) rn(s6), b3 = in(reg32) rn(s7),
                    options(nostack),
                );
                ch += threads * 8;
            }
            return;
        }

        // ---------------------------------------------------------------------
        // Sub-op 0: Bit-Exact BF16 Tensor-Core W4A16/W8A16 Grouped Prefill MoE GEMM (`128 M x 128 N`).
        // ---------------------------------------------------------------------
        if subop != PREFILL_OP_GEMM {
            core::arch::nvptx::trap();
        }
        let stack = *words as *const u8;
        let input = *words.add(1) as *const u32;
        let output = *words.add(2) as *mut f32;
        let rows = *words.add(3) as usize;
        let col_meta = *words.add(4);
        let col_word = col_meta as u32;
        let format = col_meta >> 32;
        let k = (col_word & COLS) as usize;
        let wide = col_word & WIDE != 0;
        let pairs = col_word & PAIRS != 0;
        let all = *words.add(5) as usize;
        let stride = *words.add(6) as usize;
        let unit_bytes = *words.add(7) as usize;
        let in_table = *words.add(8) as *const u32;
        let in_count = *words.add(9) as usize;
        let out_table = *words.add(10) as *const u32;
        let tile_meta = *words.add(12) as *const u32;
        let max_m_tiles = *words.add(13) as usize;
        let experts = *words.add(14) as usize;
        let coefficients = *words.add(15) as *const f32;
        let gated = !coefficients.is_null();
        let half_in = *words.add(17) == PREFILL_HALF_IN;

        if ((in_table as usize | out_table as usize | tile_meta as usize) & 3) != 0
            || tile_meta.is_null()
            || !half_in
            || !gemm_admitted(
                rows, k, wide, pairs, format, gated, all, stride, experts, max_m_tiles, unit_bytes,
                stack as usize, input as usize, output as usize, coefficients as usize, threads,
            )
        {
            core::arch::nvptx::trap();
        }

        // 2D L2 CTA swizzling: `PREFILL_M_GROUP = 8` neighboring route tiles traverse each N-tile together.
        let n_tiles = rows.div_ceil(PREFILL_BN);
        if block >= n_tiles * max_m_tiles {
            return;
        }
        let group = block / (PREFILL_M_GROUP * n_tiles);
        let group_len = (max_m_tiles - PREFILL_M_GROUP * group).min(PREFILL_M_GROUP);
        let within = block % (PREFILL_M_GROUP * n_tiles);
        let tile_n = within / group_len;
        let tm = group * PREFILL_M_GROUP + within % group_len;
        let total_m_tiles = tile_meta.read() as usize;
        if total_m_tiles > max_m_tiles {
            core::arch::nvptx::trap();
        }
        if tm >= total_m_tiles || tile_n * PREFILL_BN >= rows {
            return;
        }

        // Pair route tiles `2r` and `2r + 1` of the same expert into one `PAIR_M = 128` block.
        let meta = |t: usize| -> (usize, usize, usize) {
            let entry = tile_meta.add(1 + 3 * t);
            (entry.read() as usize, entry.add(1).read() as usize, entry.add(2).read() as usize)
        };
        let (expert, base, m_first) = meta(tm);
        if expert >= experts || m_first == 0 || m_first > TC_BM {
            core::arch::nvptx::trap();
        }
        let (mut lo, mut hi) = (0usize, tm);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if meta(mid).0 < expert { lo = mid + 1; } else { hi = mid; }
        }
        let nth = tm - lo;
        if meta(lo).0 != expert || meta(lo).1 + nth * TC_BM != base {
            core::arch::nvptx::trap();
        }
        if nth % 2 == 1 {
            return;
        }
        let m_second = if tm + 1 < total_m_tiles {
            let (next_expert, next_base, next_m) = meta(tm + 1);
            if next_expert == expert {
                if m_first != TC_BM || next_base != base + TC_BM || next_m == 0 || next_m > TC_BM {
                    core::arch::nvptx::trap();
                }
                next_m
            } else {
                0
            }
        } else {
            0
        };
        let m = m_first + m_second;
        if base + m > all {
            core::arch::nvptx::trap();
        }

        let unit = stack.add(expert * unit_bytes);
        let parts = layout(rows, k, wide);
        if in_count * (k / 2) >= u32::MAX as usize || unit_bytes >= u32::MAX as usize {
            core::arch::nvptx::trap();
        }
        let mode: u32 = if !wide && rows % 64 == 0 && k % 64 == 0 && {
            let t32 = unit.add(parts.table).cast::<u32>();
            t32.add(0x7f * 8).read() == NVFP4_PANEL_MAGIC_LO
                && t32.add(0x7f * 8 + 1).read() == NVFP4_PANEL_MAGIC_HI
        } {
            2
        } else if wide {
            1
        } else {
            0
        };

        let place = |tbl: *const u32, count: usize, t: usize| -> usize {
            let row = if tbl.is_null() { t } else { tbl.add(t).read() as usize };
            if row >= count { core::arch::nvptx::trap(); }
            row
        };

        // 2 M-warps x 4 N-warps, with `half_m` split (`m <= 64` -> `warp_m * 32`, both M-warps active!).
        let (warp, lane) = (tid >> 5, tid & 31);
        let (warp_m, warp_n) = (warp >> 2, warp & 3);
        let half_m = m <= TC_BM;
        let m_row_base = if half_m { warp_m * 32 } else { warp_m * 64 };
        let m_row_cap = if half_m { 32 } else { 64 };
        let subs = m.saturating_sub(m_row_base).min(m_row_cap).div_ceil(16) as u32;

        // Save 24 B of tile metadata in `META_AT` (0 live regs in K-loop).
        if tid == 0 {
            let meta_p = smem + META_AT as u32;
            let w_addr = words as usize as u64;
            store_u32(meta_p, base as u32);
            store_u32(meta_p + 4, m as u32);
            store_u32(meta_p + 8, tile_n as u32);
            store_u32(meta_p + 12, w_addr as u32);
            store_u32(meta_p + 16, (w_addr >> 32) as u32);
            store_u32(meta_p + 20, u32::from(half_m));
        }

        // For NVFP4 (`mode == 2` or `mode == 0`), build the 1 KB `(V1 * 2^112, V3_half * 2^112)` LUT once per CTA.
        if !wide {
            let v_up: u32 = if tid == 0x7f || tid == 0xff {
                0
            } else {
                let raw = ld_global_nc_u32(unit.add(parts.table + tid * 32 + 4).cast());
                let scaled: u32;
                core::arch::asm!(
                    "mul.rn.bf16x2 {d}, {a}, {b};",
                    d = out(reg32) scaled, a = in(reg32) raw, b = in(reg32) 0x7780_7780u32,
                    options(pure, nomem, nostack),
                );
                scaled
            };
            store_u32(smem + LUT_AT as u32 + (tid * 4) as u32, v_up);
        }

        // BF16 activation A indexing (`a_row` and `a_row + 64`, 16B `a_chunk` in 0..4).
        let row_words = k / 2;
        let a_row = tid >> 2;
        let a_chunk = tid & 3;
        let a_src0 = if a_row < m { (place(in_table, in_count, base + a_row) * row_words + a_chunk * 4) as u32 } else { u32::MAX };
        let a_src1 = if a_row + 64 < m { (place(in_table, in_count, base + a_row + 64) * row_words + a_chunk * 4) as u32 } else { u32::MAX };
        let a_dst0 = swiz(a_row, a_chunk);
        let a_off = swiz(m_row_base + (lane & 15), lane >> 4);

        // 16 MMA accumulators (`4` M-tiles x `4` N-tiles = `64 x 32` per warp), strictly in registers.
        let (mut c00, mut c01, mut c02, mut c03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c10, mut c11, mut c12, mut c13) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c20, mut c21, mut c22, mut c23) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c30, mut c31, mut c32, mut c33) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        macro_rules! k_sub_mma {
            ($a_at:expr, $bp0:ident, $bp1:ident) => {{
                let a_at = $a_at;
                if subs == 4 {
                    let a0 = ldmatrix_x4(a_at);
                    let a1 = ldmatrix_x4(a_at + TILE16);
                    c00 = mma_m16n8k16_bf16(a0, [$bp0[0], $bp0[1]], c00);
                    c01 = mma_m16n8k16_bf16(a0, [$bp0[2], $bp0[3]], c01);
                    c02 = mma_m16n8k16_bf16(a0, [$bp1[0], $bp1[1]], c02);
                    c03 = mma_m16n8k16_bf16(a0, [$bp1[2], $bp1[3]], c03);
                    let a2 = ldmatrix_x4(a_at + 2 * TILE16);
                    c10 = mma_m16n8k16_bf16(a1, [$bp0[0], $bp0[1]], c10);
                    c11 = mma_m16n8k16_bf16(a1, [$bp0[2], $bp0[3]], c11);
                    c12 = mma_m16n8k16_bf16(a1, [$bp1[0], $bp1[1]], c12);
                    c13 = mma_m16n8k16_bf16(a1, [$bp1[2], $bp1[3]], c13);
                    let a3 = ldmatrix_x4(a_at + 3 * TILE16);
                    c20 = mma_m16n8k16_bf16(a2, [$bp0[0], $bp0[1]], c20);
                    c21 = mma_m16n8k16_bf16(a2, [$bp0[2], $bp0[3]], c21);
                    c22 = mma_m16n8k16_bf16(a2, [$bp1[0], $bp1[1]], c22);
                    c23 = mma_m16n8k16_bf16(a2, [$bp1[2], $bp1[3]], c23);
                    c30 = mma_m16n8k16_bf16(a3, [$bp0[0], $bp0[1]], c30);
                    c31 = mma_m16n8k16_bf16(a3, [$bp0[2], $bp0[3]], c31);
                    c32 = mma_m16n8k16_bf16(a3, [$bp1[0], $bp1[1]], c32);
                    c33 = mma_m16n8k16_bf16(a3, [$bp1[2], $bp1[3]], c33);
                } else if subs > 0 {
                    let a0 = ldmatrix_x4(a_at);
                    c00 = mma_m16n8k16_bf16(a0, [$bp0[0], $bp0[1]], c00);
                    c01 = mma_m16n8k16_bf16(a0, [$bp0[2], $bp0[3]], c01);
                    c02 = mma_m16n8k16_bf16(a0, [$bp1[0], $bp1[1]], c02);
                    c03 = mma_m16n8k16_bf16(a0, [$bp1[2], $bp1[3]], c03);
                    if subs > 1 {
                        let a1 = ldmatrix_x4(a_at + TILE16);
                        c10 = mma_m16n8k16_bf16(a1, [$bp0[0], $bp0[1]], c10);
                        c11 = mma_m16n8k16_bf16(a1, [$bp0[2], $bp0[3]], c11);
                        c12 = mma_m16n8k16_bf16(a1, [$bp1[0], $bp1[1]], c12);
                        c13 = mma_m16n8k16_bf16(a1, [$bp1[2], $bp1[3]], c13);
                        if subs > 2 {
                            let a2 = ldmatrix_x4(a_at + 2 * TILE16);
                            c20 = mma_m16n8k16_bf16(a2, [$bp0[0], $bp0[1]], c20);
                            c21 = mma_m16n8k16_bf16(a2, [$bp0[2], $bp0[3]], c21);
                            c22 = mma_m16n8k16_bf16(a2, [$bp1[0], $bp1[1]], c22);
                            c23 = mma_m16n8k16_bf16(a2, [$bp1[2], $bp1[3]], c23);
                        }
                    }
                }
            }};
        }

        if mode == 2 {
            // In-place MMA-lane NVFP4 panel path (`pack_expert_tagged`):
            // - 3-stage `cp.async` A (`24,576 B`, `wait_group 1`) + 2-stage unpacked B (`16,384 B`) + 1 KB `shl+prmt` LUT!
            // - `.L2::256B` codes + `.L2::128B` scales WITHOUT `evict_first` so `PREFILL_M_GROUP = 8` shares weights in L2!
            // - Interleaved unpack inside `k_sub_0` HMMAs (fills SM121a HMMA shadow slots + launches `fetch_b64` early!).
            for st in 0..A_STAGES as u32 {
                let s = smem + st * A_BYTES as u32;
                if a_src0 == u32::MAX { st_shared_zero16(s + a_dst0); }
                if a_src1 == u32::MAX { st_shared_zero16(s + a_dst0 + 4096); }
            }
            macro_rules! issue_a {
                ($st:expr, $c_idx:expr) => {{
                    let s = smem + ($st as u32) * A_BYTES as u32;
                    let at = ($c_idx as u32) * 16;
                    if a_src0 != u32::MAX { crate::ptx::copy16(s + a_dst0, input.add((a_src0 + at) as usize)); }
                    if a_src1 != u32::MAX { crate::ptx::copy16(s + a_dst0 + 4096, input.add((a_src1 + at) as usize)); }
                }};
            }

            let stages = (k / 64) as u32;
            let (b_code0, b_scale0) = if tile_n * PREFILL_BN + warp * 16 < rows {
                let chunk0 = ((tile_n * 2 + (warp >> 2)) * (k >> 6)) * 4 + (warp & 3);
                ((chunk0 * 512 + lane * 16) as u32, (parts.scales + chunk0 * 64 + (lane >> 2) * 8) as u32)
            } else {
                (u32::MAX, 0)
            };
            let b_wr0 = smem + B_AT as u32 + (warp * 1024 + lane * 16) as u32;
            let s0_b = smem + B_AT as u32 + (warp_n * 2 * 1024 + lane * 16) as u32;
            if b_code0 == u32::MAX {
                for b_st in 0..B_STAGES as u32 {
                    let b_base = b_wr0 + b_st * B_BYTES as u32;
                    st_shared_zero16(b_base);
                    st_shared_zero16(b_base + 512);
                }
            }

            macro_rules! fetch_b64 {
                ($s:expr) => {{
                    let st_idx: u32 = $s;
                    if b_code0 != u32::MAX {
                        let (q0, q1, q2, q3): (u32, u32, u32, u32);
                        let (sc0, sc1): (u32, u32);
                        core::arch::asm!(
                            "ld.global.nc.L2::256B.v4.u32 {{{q0}, {q1}, {q2}, {q3}}}, [{pc}]; \
                             ld.global.nc.L2::128B.v2.u32 {{{s0}, {s1}}}, [{ps}];",
                            q0 = out(reg32) q0, q1 = out(reg32) q1, q2 = out(reg32) q2, q3 = out(reg32) q3,
                            s0 = out(reg32) sc0, s1 = out(reg32) sc1,
                            pc = in(reg64) unit.add((b_code0 + st_idx * 2048) as usize),
                            ps = in(reg64) unit.add((b_scale0 + st_idx * 256) as usize),
                            options(readonly, nostack),
                        );
                        ([q0, q1, q2, q3], [sc0, sc1])
                    } else {
                        ([0u32; 4], [0u32; 2])
                    }
                }};
            }
            macro_rules! unpack_panel_k16 {
                ($w:expr, $s_pair:expr) => {{
                    let w: u32 = $w;
                    let s_pair: u32 = $s_pair;
                    let st_raw = load_u32(smem + LUT_AT as u32 + (s_pair & 255) * 4);
                    let (vt1, vt3): (u32, u32);
                    core::arch::asm!(
                        "prmt.b32 {vt1}, {st}, {st}, 0x1010; \
                         prmt.b32 {vt3}, {st}, {st}, 0x3232;",
                        vt1 = out(reg32) vt1, vt3 = out(reg32) vt3,
                        st = in(reg32) st_raw,
                        options(pure, nomem, nostack),
                    );
                    let b0 = unpack_nv_byte!(w, vt1, vt3);
                    let b1 = unpack_nv_byte!(w >> 16, vt1, vt3);
                    let sb_raw = load_u32(smem + LUT_AT as u32 + ((s_pair >> 8) & 255) * 4);
                    let (vb1, vb3): (u32, u32);
                    core::arch::asm!(
                        "prmt.b32 {vb1}, {sb}, {sb}, 0x1010; \
                         prmt.b32 {vb3}, {sb}, {sb}, 0x3232;",
                        vb1 = out(reg32) vb1, vb3 = out(reg32) vb3,
                        sb = in(reg32) sb_raw,
                        options(pure, nomem, nostack),
                    );
                    let b2 = unpack_nv_byte!(w >> 8, vb1, vb3);
                    let b3 = unpack_nv_byte!(w >> 24, vb1, vb3);
                    [b0, b1, b2, b3]
                }};
            }
            macro_rules! put_panel_half {
                ($stage:expr, $sub_off:expr, $w:expr, $s_pair:expr) => {{
                    if b_code0 != u32::MAX {
                        st_shared_v4(
                            b_wr0 + ($stage as u32) * B_BYTES as u32 + ($sub_off as u32),
                            unpack_panel_k16!($w, $s_pair),
                        );
                    }
                }};
            }
            macro_rules! k_sub_panel {
                ($a_at:expr, $b_at:expr) => {{
                    let (a_at, b_at) = ($a_at, $b_at);
                    let bp0 = crate::ptx::load4(b_at);
                    let bp1 = crate::ptx::load4(b_at + 1024);
                    k_sub_mma!(a_at, bp0, bp1);
                }};
            }

            // Prologue: A steps 0 and 1 in flight, fetch B stage 0, barrier for LUT, unpack B0.
            issue_a!(0, 0);
            crate::ptx::commit();
            issue_a!(1, 1);
            crate::ptx::commit();
            let (mut b_q, mut b_sc) = fetch_b64!(0);
            core::arch::nvptx::_syncthreads();
            put_panel_half!(0, 0, b_q[0], b_sc[0]);
            put_panel_half!(0, 512, b_q[1], b_sc[0] >> 16);

            let mut a_cur = 0u32;
            let mut s = 0u32;
            while s < stages {
                let more = s + 1 < stages;
                // Half 0 (`k = 0..32` of `s`): wait for A[a_cur] + B0 visible!
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
                let a_wr = if a_cur == 0 { 2 } else { a_cur - 1 };
                if more { issue_a!(a_wr, 2 * s + 2); }
                crate::ptx::commit();
                let a_stage = smem + a_cur * A_BYTES as u32;
                // Unpack B1 (`k = 32..64`) so `b_q/b_sc` are dead during `k_sub_0`, then early `fetch_b64!(s + 1)`!
                put_panel_half!(1, 0, b_q[2], b_sc[1]);
                put_panel_half!(1, 512, b_q[3], b_sc[1] >> 16);
                k_sub_panel!(a_stage + a_off, s0_b);
                if more {
                    let nb = fetch_b64!(s + 1);
                    b_q = nb.0;
                    b_sc = nb.1;
                }
                k_sub_panel!(a_stage + (a_off ^ 32), s0_b + 512);
                a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };

                // Half 1 (`k = 32..64` of `s`): wait for A[a_cur] + B1 visible!
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
                let a_wr = if a_cur == 0 { 2 } else { a_cur - 1 };
                if more { issue_a!(a_wr, 2 * s + 3); }
                crate::ptx::commit();
                let a_stage = smem + a_cur * A_BYTES as u32;
                k_sub_panel!(a_stage + a_off, s0_b + B_BYTES as u32);
                k_sub_panel!(a_stage + (a_off ^ 32), s0_b + (B_BYTES + 512) as u32);
                if more {
                    put_panel_half!(0, 0, b_q[0], b_sc[0]);
                    put_panel_half!(0, 512, b_q[1], b_sc[0] >> 16);
                }
                a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };
                s += 1;
            }
        } else {
            // Shipped tiled layout (`mode == 1` MXFP8, `mode == 0` NVFP4):
            // 3 stages of `K = 32` (`12,800 B` each -> `38,400 B <= 40,960 B`, `wait_group 1`, 1 barrier per 32 cols),
            // `cp.async.cg` on codes (L1-bypass hardware async!), `half_m` split, pipelined `LDSM`, 5-inst MXFP8 / 8-inst NVFP4 unpack.
            let steps = (k / 32) as u32;
            let (g, t) = (lane >> 2, lane & 3);
            let b_row = tid >> 1;
            let b_half = tid & 1;
            let global_b_row = tile_n * PREFILL_BN + b_row;
            let b_valid = global_b_row < rows;
            let (b_code_off, b_scale_off): (u32, u32) = if b_valid {
                let code_off = code_copy(global_b_row, 0, 0, k, wide);
                let scale_off = parts.scales + scale_copy(global_b_row, 0, k, wide);
                if wide {
                    ((code_off + b_half * 16) as u32, if b_half == 0 { scale_off as u32 } else { u32::MAX })
                } else {
                    (
                        if b_half == 0 { code_off as u32 } else { u32::MAX },
                        if b_half == 0 { scale_off as u32 } else { u32::MAX },
                    )
                }
            } else {
                (u32::MAX, u32::MAX)
            };
            let b_dst_off = if wide {
                (A_BYTES + b_row * 32 + (b_half ^ ((b_row >> 2) & 1)) * 16) as u32
            } else {
                (A_BYTES + b_row * 16) as u32
            };
            let b_sc_dst_off = (A_BYTES + PREFILL_BN * 32 + b_row * 4) as u32;

            for st in 0..3u32 {
                let st_base = smem + st * STAGE32 as u32;
                if a_src0 == u32::MAX { st_shared_zero16(st_base + a_dst0); }
                if a_src1 == u32::MAX { st_shared_zero16(st_base + a_dst0 + 4096); }
                if !b_valid {
                    if wide || b_half == 0 { st_shared_zero16(st_base + b_dst_off); }
                    if b_half == 0 { store_u32(st_base + b_sc_dst_off, 0); }
                }
            }

            macro_rules! issue32 {
                ($st:expr, $step:expr) => {{
                    let st_base = smem + ($st as u32) * STAGE32 as u32;
                    let step_u: u32 = $step;
                    let a_w = step_u * 16;
                    if a_src0 != u32::MAX { crate::ptx::copy16(st_base + a_dst0, input.add((a_src0 + a_w) as usize)); }
                    if a_src1 != u32::MAX { crate::ptx::copy16(st_base + a_dst0 + 4096, input.add((a_src1 + a_w) as usize)); }
                    if b_code_off != u32::MAX {
                        let b_co = if wide {
                            (step_u >> 1) * 2048 + (step_u & 1) * 32
                        } else {
                            (step_u >> 1) * 1024 + (step_u & 1) * 16
                        };
                        core::arch::asm!(
                            "cp.async.cg.shared.global [{d}], [{p}], 16;",
                            d = in(reg32) (st_base + b_dst_off),
                            p = in(reg64) unit.add((b_code_off + b_co) as usize),
                            options(nostack),
                        );
                    }
                    if b_scale_off != u32::MAX {
                        let b_so = if wide { (step_u >> 2) * 128 } else { (step_u >> 1) * 128 };
                        core::arch::asm!(
                            "cp.async.ca.shared.global [{d}], [{p}], 4;",
                            d = in(reg32) (st_base + b_sc_dst_off),
                            p = in(reg64) unit.add((b_scale_off + b_so) as usize),
                            options(nostack),
                        );
                    }
                    crate::ptx::commit();
                }};
            }

            issue32!(0, 0);
            if steps > 1 { issue32!(1, 1); }

            let b_r0 = warp_n * 32 + g;
            let mut step = 0u32;
            let mut cur = 0u32;
            while step < steps {
                if step + 1 < steps { cp_async_wait_one(); } else { cp_async_wait_zero(); }
                core::arch::nvptx::_syncthreads();
                let nxt2 = if cur == 0 { 2 } else { cur - 1 };
                if step + 2 < steps { issue32!(nxt2, step + 2); }
                if subs > 0 {
                    let st_base = smem + cur * STAGE32 as u32;
                    let sc_base = st_base + (A_BYTES + PREFILL_BN * 32) as u32;
                    if wide {
                        let sc_shift = (step & 3) * 8;
                        macro_rules! load_mx_row {
                            ($row:expr, $k_sub:expr) => {{
                                let row: usize = $row;
                                let sc = (load_u32(sc_base + (row * 4) as u32) >> sc_shift) & 255;
                                let (p1, p2, two) = if sc <= 142 {
                                    (((sc + 112) << 7) * 0x0001_0001, 0, false)
                                } else {
                                    (0x7780_7780, (sc << 7) * 0x0001_0001, true)
                                };
                                let row_off = st_base + (A_BYTES + row * 32 + ($k_sub ^ ((row >> 2) & 1)) * 16) as u32;
                                let c0 = crate::ptx::load_u16(row_off + (t * 2) as u32) as u16;
                                let c1 = crate::ptx::load_u16(row_off + (8 + t * 2) as u32) as u16;
                                [unpack_mx_pair!(c0, p1, p2, two), unpack_mx_pair!(c1, p1, p2, two)]
                            }};
                        }
                        for k_sub in 0..2usize {
                            let a_at = st_base + if k_sub == 0 { a_off } else { a_off ^ 32 };
                            let r0 = load_mx_row!(b_r0, k_sub);
                            let r1 = load_mx_row!(b_r0 + 8, k_sub);
                            let r2 = load_mx_row!(b_r0 + 16, k_sub);
                            let r3 = load_mx_row!(b_r0 + 24, k_sub);
                            let bp0 = [r0[0], r0[1], r1[0], r1[1]];
                            let bp1 = [r2[0], r2[1], r3[0], r3[1]];
                            k_sub_mma!(a_at, bp0, bp1);
                        }
                    } else {
                        let half_shift = (step & 1) * 16;
                        macro_rules! load_nv_row {
                            ($row:expr, $k_sub:expr) => {{
                                let row: usize = $row;
                                let sc = (load_u32(sc_base + (row * 4) as u32) >> (half_shift + ($k_sub * 8) as u32)) & 255;
                                let s_raw = load_u32(smem + LUT_AT as u32 + sc * 4);
                                let (v1, v3): (u32, u32);
                                core::arch::asm!(
                                    "prmt.b32 {v1}, {s}, {s}, 0x1010; prmt.b32 {v3}, {s}, {s}, 0x3232;",
                                    v1 = out(reg32) v1, v3 = out(reg32) v3, s = in(reg32) s_raw,
                                    options(pure, nomem, nostack),
                                );
                                let row_off = st_base + (A_BYTES + row * 16 + $k_sub * 8) as u32;
                                let w_lo = load_u32(row_off) >> (t * 8);
                                let w_hi = load_u32(row_off + 4) >> (t * 8);
                                [unpack_nv_byte!(w_lo, v1, v3), unpack_nv_byte!(w_hi, v1, v3)]
                            }};
                        }
                        for k_sub in 0..2usize {
                            let a_at = st_base + if k_sub == 0 { a_off } else { a_off ^ 32 };
                            let r0 = load_nv_row!(b_r0, k_sub);
                            let r1 = load_nv_row!(b_r0 + 8, k_sub);
                            let r2 = load_nv_row!(b_r0 + 16, k_sub);
                            let r3 = load_nv_row!(b_r0 + 24, k_sub);
                            let bp0 = [r0[0], r0[1], r1[0], r1[1]];
                            let bp1 = [r2[0], r2[1], r3[0], r3[1]];
                            k_sub_mma!(a_at, bp0, bp1);
                        }
                    }
                }
                cur = if cur == 2 { 0 } else { cur + 1 };
                step += 1;
            }
        }

        // ---------------------------------------------------------------------
        // Epilogue: reload cold command words & metadata from `META_AT` (0 regs held in K-loop).
        // ---------------------------------------------------------------------
        let meta_p = smem + META_AT as u32;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid >> 5, tid & 31);
        let (warp_m, warp_n) = (warp >> 2, warp & 3);
        let (g, t) = (lane >> 2, lane & 3);
        let base = load_u32(meta_p) as usize;
        let m = load_u32(meta_p + 4) as usize;
        let tile_n = load_u32(meta_p + 8) as usize;
        let w_lo = load_u32(meta_p + 12) as u64;
        let w_hi = load_u32(meta_p + 16) as u64;
        let half_m = load_u32(meta_p + 20) != 0;
        let m_row_base = if half_m { warp_m * 32 } else { warp_m * 64 };
        let m_row_cap = if half_m { 32 } else { 64 };
        let subs = m.saturating_sub(m_row_base).min(m_row_cap).div_ceil(16);
        let words = ((w_lo | (w_hi << 32)) as usize) as *const u64;
        let output = words.add(2).read_volatile() as *mut f32;
        let rows = words.add(3).read_volatile() as usize;
        let col_meta = words.add(4).read_volatile();
        let pairs = (col_meta as u32) & PAIRS != 0;
        let format = col_meta >> 32;
        let stride = words.add(6).read_volatile() as usize;
        let out_table = words.add(10).read_volatile() as *const u32;
        let out_count = words.add(11).read_volatile() as usize;
        let coefficients = words.add(15).read_volatile() as *const f32;
        let gated = !coefficients.is_null();

        let place_out = |row: usize| -> usize {
            if row >= m { return u32::MAX as usize; }
            let idx = base + row;
            let r = if out_table.is_null() { idx } else { ld_global_nc_u32(out_table.add(idx)) as usize };
            if r >= out_count { core::arch::nvptx::trap(); }
            r
        };

        // Exact `FAST_CAPPED_GATE` LUT lookup (bit-for-bit identical to `plain_packed_tc_rows.rs`).
        macro_rules! capped {
            ($gate:expr, $up:expr) => {{
                let gi = (rn($gate).to_bits() >> 16) as usize;
                let ui = (rn($up).to_bits() >> 16) as usize;
                let gc = f32::from_bits(ld_global_nc_u32(coefficients.add(gi).cast()));
                let uc = f32::from_bits(ld_global_nc_u32(coefficients.add(CAPPED_GATE_TABLE_WORDS + ui).cast()));
                rn(gc * uc)
            }};
        }

        // Format 1: Exact gated `GateUp` -> coalesced 128-byte BF16 output via dead SMEM `0..18,432` (`128 x 72` u16).
        if format == PREFILL_OUT_BF16_GATED {
            core::arch::nvptx::_syncthreads();
            let col_w = warp_n * 16;
            macro_rules! stage_gate_one {
                ($r0:expr, $r1:expr, $ni:expr, $a:expr) => {{
                    let a = $a;
                    let c_off = col_w + $ni * 4 + t;
                    let h0 = (capped!(a[0], a[1]).to_bits() >> 16) as u16;
                    let h1 = (capped!(a[2], a[3]).to_bits() >> 16) as u16;
                    store_u16(smem + (($r0 * 72 + c_off) * 2) as u32, h0);
                    store_u16(smem + (($r1 * 72 + c_off) * 2) as u32, h1);
                }};
            }
            macro_rules! stage_gate_row {
                ($mi:expr, $d0:ident, $d1:ident, $d2:ident, $d3:ident) => {
                    if subs > $mi {
                        let r0 = m_row_base + $mi * 16 + g;
                        let r1 = r0 + 8;
                        stage_gate_one!(r0, r1, 0, $d0);
                        stage_gate_one!(r0, r1, 1, $d1);
                        stage_gate_one!(r0, r1, 2, $d2);
                        stage_gate_one!(r0, r1, 3, $d3);
                    }
                };
            }
            stage_gate_row!(0, c00, c01, c02, c03);
            stage_gate_row!(1, c10, c11, c12, c13);
            stage_gate_row!(2, c20, c21, c22, c23);
            stage_gate_row!(3, c30, c31, c32, c33);
            core::arch::nvptx::_syncthreads();

            let width = rows / 2;
            let lane8 = tid & 7;
            let row_base = tid >> 3;
            let col_start = tile_n * (PREFILL_BN / 2) + lane8 * 8;
            let out_u32 = output.cast::<u32>();
            let aligned_v4 = (output as usize & 15) == 0 && (stride & 7) == 0 && col_start + 8 <= width;
            let mut it = 0usize;
            while it < 4 {
                let row = it * 32 + row_base;
                if row < m && col_start < width {
                    let out_r = place_out(row);
                    let v = crate::ptx::load4(smem + (row * 144 + lane8 * 16) as u32);
                    let dst = out_u32.add((out_r * stride + col_start) / 2);
                    if aligned_v4 {
                        core::arch::asm!(
                            "st.global.v4.b32 [{p}], {{{a0}, {a1}, {a2}, {a3}}};",
                            p = in(reg64) dst,
                            a0 = in(reg32) v[0], a1 = in(reg32) v[1], a2 = in(reg32) v[2], a3 = in(reg32) v[3],
                            options(nostack),
                        );
                    } else {
                        let rem = (width - col_start).min(8) / 2;
                        if rem > 0 { *dst = v[0]; }
                        if rem > 1 { *dst.add(1) = v[1]; }
                        if rem > 2 { *dst.add(2) = v[2]; }
                        if rem > 3 { *dst.add(3) = v[3]; }
                    }
                }
                it += 1;
            }
            return;
        }

        // Format 2: Coalesced 256-byte (`16` lanes x `st.global.L2::cache_hint.v4.b32` `evict_first`) BF16 Down epilogue
        // via dead SMEM `0..34,816` (`128 x 68` u32).
        if format == PREFILL_OUT_BF16 {
            core::arch::nvptx::_syncthreads();
            let col_u32 = warp_n * 16;
            macro_rules! stage_bf16_one {
                ($r0:expr, $r1:expr, $ni:expr, $a:expr) => {{
                    let a = $a;
                    let c_off = col_u32 + $ni * 4 + t;
                    let p0 = (rn(a[0]).to_bits() >> 16) | (rn(a[1]).to_bits() & 0xffff_0000);
                    let p1 = (rn(a[2]).to_bits() >> 16) | (rn(a[3]).to_bits() & 0xffff_0000);
                    store_u32(smem + (($r0 * 68 + c_off) * 4) as u32, p0);
                    store_u32(smem + (($r1 * 68 + c_off) * 4) as u32, p1);
                }};
            }
            macro_rules! stage_bf16_row {
                ($mi:expr, $d0:ident, $d1:ident, $d2:ident, $d3:ident) => {
                    if subs > $mi {
                        let r0 = m_row_base + $mi * 16 + g;
                        let r1 = r0 + 8;
                        stage_bf16_one!(r0, r1, 0, $d0);
                        stage_bf16_one!(r0, r1, 1, $d1);
                        stage_bf16_one!(r0, r1, 2, $d2);
                        stage_bf16_one!(r0, r1, 3, $d3);
                    }
                };
            }
            stage_bf16_row!(0, c00, c01, c02, c03);
            stage_bf16_row!(1, c10, c11, c12, c13);
            stage_bf16_row!(2, c20, c21, c22, c23);
            stage_bf16_row!(3, c30, c31, c32, c33);
            core::arch::nvptx::_syncthreads();

            let evict = l2_evict_first();
            let lane16 = tid & 15;
            let row_base = tid >> 4;
            let col_start = tile_n * PREFILL_BN + lane16 * 8;
            let out_u32 = output.cast::<u32>();
            let aligned_v4 = (output as usize & 15) == 0 && (stride & 7) == 0 && col_start + 8 <= rows;
            let mut it = 0usize;
            while it < 8 {
                let row = it * 16 + row_base;
                if row < m && col_start < rows {
                    let out_r = place_out(row);
                    let v = crate::ptx::load4(smem + ((row * 68 + lane16 * 4) * 4) as u32);
                    let dst = out_u32.add((out_r * stride + col_start) / 2);
                    if aligned_v4 {
                        core::arch::asm!(
                            "st.global.L2::cache_hint.v4.b32 [{p}], {{{a0}, {a1}, {a2}, {a3}}}, {q};",
                            p = in(reg64) dst,
                            a0 = in(reg32) v[0], a1 = in(reg32) v[1], a2 = in(reg32) v[2], a3 = in(reg32) v[3],
                            q = in(reg64) evict,
                            options(nostack),
                        );
                    } else {
                        let rem = (rows - col_start).min(8) / 2;
                        if rem > 0 { *dst = v[0]; }
                        if rem > 1 { *dst.add(1) = v[1]; }
                        if rem > 2 { *dst.add(2) = v[2]; }
                        if rem > 3 { *dst.add(3) = v[3]; }
                    }
                }
                it += 1;
            }
            return;
        }

        // Format 0: FP32 output (ungated or exact LUT-gated).
        macro_rules! store_tile {
            ($out_r0:expr, $out_r1:expr, $ni:expr, $v:expr) => {{
                let (out_r0, out_r1, v) = ($out_r0, $out_r1, $v);
                let col = tile_n * PREFILL_BN + warp_n * 32 + $ni * 8 + t * 2;
                if col < rows {
                    if gated {
                        if out_r0 != u32::MAX as usize { *output.add(out_r0 * stride + col / 2) = capped!(v[0], v[1]); }
                        if out_r1 != u32::MAX as usize { *output.add(out_r1 * stride + col / 2) = capped!(v[2], v[3]); }
                    } else {
                        let c0 = output_row(col, rows, pairs);
                        let has_c1 = col + 1 < rows;
                        let c1 = if has_c1 { output_row(col + 1, rows, pairs) } else { 0 };
                        if out_r0 != u32::MAX as usize {
                            *output.add(out_r0 * stride + c0) = v[0];
                            if has_c1 { *output.add(out_r0 * stride + c1) = v[1]; }
                        }
                        if out_r1 != u32::MAX as usize {
                            *output.add(out_r1 * stride + c0) = v[2];
                            if has_c1 { *output.add(out_r1 * stride + c1) = v[3]; }
                        }
                    }
                }
            }};
        }
        macro_rules! store_rows {
            ($mi:expr, $d0:ident, $d1:ident, $d2:ident, $d3:ident) => {
                if subs > $mi {
                    let m0 = m_row_base + $mi * 16 + g;
                    let r0 = place_out(m0);
                    let r1 = place_out(m0 + 8);
                    store_tile!(r0, r1, 0, $d0);
                    store_tile!(r0, r1, 1, $d1);
                    store_tile!(r0, r1, 2, $d2);
                    store_tile!(r0, r1, 3, $d3);
                }
            };
        }
        store_rows!(0, c00, c01, c02, c03);
        store_rows!(1, c10, c11, c12, c13);
        store_rows!(2, c20, c21, c22, c23);
        store_rows!(3, c30, c31, c32, c33);
    }
}
