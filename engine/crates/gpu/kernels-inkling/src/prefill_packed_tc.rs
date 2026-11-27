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
//!    - Balanced 16-row `m_step = m.div_ceil(32).max(1) * 16` split: distributes rows evenly across `warp_m = 0`
//!      and `warp_m = 1` at 16-row MMA granularity, eliminating idle M-warps and barrier skew on unpaired/tail tiles.
//!    - `mode == 2`: in-place `pack_expert_tagged` MMA-lane NVFP4 panels. `42,016 B` SMEM (`2 blk/SM`):
//!      `3` streamed A stages (`24,576 B`, `copy16_stream`, `wait_group 1`) + `2` unpacked B stages (`16,384 B`) +
//!      `1 KB` `(V1*2^112, V3_half*2^112)` LUT + `32 B` `META`. Raw B words of stage `s + 2` are fetched into
//!      registers at the end of half 0 (3 halves before unpack), and each warp cooperatively unpacks 16x16 weights
//!      via zero-IMAD 8-inst `shl+prmt 0xbb99` interleaved inside `slab_full!` HMMAs. `.L2::256B` codes +
//!      `.L2::128B` scales with serpentine `tile_n` swizzle preserve `PREFILL_M_GROUP = 8` L2 weight reuse.
//!    - `mode == 1`: shipped tiled MXFP8, coalesced. The A ring of `mode == 2` + `2` raw B stages of `64` columns
//!      (`128 x 64 B`: the 4 row blocks of 32 rows copied whole, so a warp's `cp.async.cg` reads 512 contiguous
//!      bytes, 4 full lines, in place of 16 half lines of the 32-column form) + the 4-byte scale copies in the
//!      LUT bytes (`2 x 512 B`). 5-inst exact `cvt.rn.f16x2.e4m3x2` register unpack, zero SMEM LUT.
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
    ld_global_nc_v4_u32_hint, ldmatrix_x4, load_u32, mma_m16n8k16_bf16, shfl_xor_u32,
    st_shared_v4, st_shared_zero16, store_u32,
};
use crate::tc_rows::TC_BM;

/// Bytes between two 16-row MMA tiles of an XOR-swizzled `128 x 32` BF16 stage (`16 * 64`).
const TILE16: u32 = 16 * 64;
/// Unified `42,016 B` static SMEM layout (`2 blk/SM` on SM121a `101,376 B` SMEM):
/// - `mode == 2`: `3` A stages (`24,576 B`) + `2` unpacked B stages (`16,384 B`) + `1 KB` LUT + `32 B` META.
/// - `mode == 1`: `3` A stages (`24,576 B`) + `2` raw MXFP8 B stages (`128 x 64 B`) + `2 x 512 B` scale copies + `32 B` META.
/// - `mode == 0`: `3` stages of `12,800 B` (`8,192 B` A + `4,096 B` packed B + `512 B` scales) = `38,400 B <= 40,960 B`.
const A_BYTES: usize = PREFILL_PAIR_M * 32 * 2; // 8,192 B (`128 x 32` BF16)
const A_STAGES: usize = 3;
const B_AT: usize = A_STAGES * A_BYTES; // 24,576 B
const B_BYTES: usize = PREFILL_BN * 32 * 2; // 8,192 B (`128 x 32` BF16 unpacked)
const B_STAGES: usize = 2;
const STAGE32: usize = A_BYTES + PREFILL_BN * (32 + 4); // 12,800 B (mode 0)
/// Bytes of one MXFP8 scale copy stage (`mode == 1`): 4 bytes (two 64-column stages) of each of the 128 rows.
const MX_SCALE_BYTES: usize = PREFILL_BN * 4;
const LUT_AT: usize = B_AT + B_STAGES * B_BYTES; // 40,960 B (16B aligned)
const LUT_BYTES: usize = 256 * 4; // 1,024 B (`(V1*2^112, V3_half*2^112)` per scale)
const META_AT: usize = LUT_AT + LUT_BYTES; // 41,984 B (16B aligned)
const META_BYTES: usize = 32; // `base, m, tile_n, words_lo, words_hi, m_step`
const BLOCK_BYTES: usize = META_AT + META_BYTES; // 42,016 B

const _: () = assert!(PREFILL_PAIR_M == 128 && PREFILL_BN == 128 && PREFILL_THREADS == 256);
const _: () = assert!(B_AT == 24_576 && LUT_AT == 40_960 && META_AT == 41_984 && BLOCK_BYTES == 42_016);
const _: () = assert!(3 * STAGE32 <= LUT_AT && 2 * MX_SCALE_BYTES == LUT_BYTES && B_BYTES == PREFILL_BN * 64);
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

        // 2D L2 CTA swizzling: `PREFILL_M_GROUP = 8` neighboring route tiles traverse each N-tile together
        // (serpentine `tile_n` order keeps trailing N-tiles hot in L2 across adjacent groups of hot experts).
        let n_tiles = rows.div_ceil(PREFILL_BN);
        if block >= n_tiles * max_m_tiles {
            return;
        }
        let group = block / (PREFILL_M_GROUP * n_tiles);
        let group_len = (max_m_tiles - PREFILL_M_GROUP * group).min(PREFILL_M_GROUP);
        let within = block % (PREFILL_M_GROUP * n_tiles);
        let n_raw = within / group_len;
        let tile_n = if (group & 1) == 0 { n_raw } else { n_tiles - 1 - n_raw };
        let tm = group * PREFILL_M_GROUP + within % group_len;
        let total_m_tiles = ld_global_nc_u32(tile_meta) as usize;
        if total_m_tiles > max_m_tiles {
            core::arch::nvptx::trap();
        }
        if tm >= total_m_tiles || tile_n * PREFILL_BN >= rows {
            return;
        }

        // Pair route tiles `2r` and `2r + 1` of the same expert into one `PAIR_M = 128` block.
        let (warp, lane) = (tid >> 5, tid & 31);
        let meta = |t: usize| -> (usize, usize, usize) {
            let entry = tile_meta.add(1 + 3 * t);
            (
                ld_global_nc_u32(entry) as usize,
                ld_global_nc_u32(entry.add(1)) as usize,
                ld_global_nc_u32(entry.add(2)) as usize,
            )
        };
        let (expert, base, m_first) = meta(tm);
        if expert >= experts || m_first == 0 || m_first > TC_BM {
            core::arch::nvptx::trap();
        }
        // 32-ary warp-parallel search (`<= 2` serial global loads for up to 1,024 tiles vs 10 serial loads).
        let (mut lo, mut hi) = (0usize, tm);
        while lo < hi {
            let step = (hi - lo).div_ceil(32);
            let probe = lo + lane * step;
            let ge = probe >= hi || ld_global_nc_u32(tile_meta.add(1 + 3 * probe)) as usize >= expert;
            let first = crate::ptx::ballot(ge).trailing_zeros() as usize;
            if first == 0 {
                hi = lo;
            } else {
                hi = (lo + first * step).min(hi);
                lo += (first - 1) * step + 1;
            }
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
            ld_global_nc_u32(t32.add(0x7f * 8)) == NVFP4_PANEL_MAGIC_LO
                && ld_global_nc_u32(t32.add(0x7f * 8 + 1)) == NVFP4_PANEL_MAGIC_HI
        } {
            2
        } else if wide {
            1
        } else {
            0
        };

        let place = |tbl: *const u32, count: usize, t: usize| -> usize {
            let row = if tbl.is_null() { t } else { ld_global_nc_u32(tbl.add(t)) as usize };
            if row >= count { core::arch::nvptx::trap(); }
            row
        };

        // 2 M-warps x 4 N-warps balanced at 16-row MMA granularity:
        // `m_step` in `{16, 32, 48, 64}` keeps both M-warps within 1 M-tile (cuts critical-path MMAs on `17..=32`
        // and `65..=96` and removes barrier skew across M-warps).
        let (warp_m, warp_n) = (warp >> 2, warp & 3);
        let m_step = m.div_ceil(32).max(1) * 16;
        let m_row_base = warp_m * m_step;
        let subs = m.saturating_sub(m_row_base).min(m_step).div_ceil(16) as u32;

        // Save 24 B of tile metadata in `META_AT` (0 live regs in K-loop).
        if tid == 0 {
            let meta_p = smem + META_AT as u32;
            let w_addr = words as usize as u64;
            store_u32(meta_p, base as u32);
            store_u32(meta_p + 4, m as u32);
            store_u32(meta_p + 8, tile_n as u32);
            store_u32(meta_p + 12, w_addr as u32);
            store_u32(meta_p + 16, (w_addr >> 32) as u32);
            store_u32(meta_p + 20, m_step as u32);
        }

        // Issue NVFP4 LUT global read BEFORE `in_table` reads so both global latencies overlap in the prologue.
        let lut_raw: u32 = if !wide && tid != 0x7f && tid != 0xff {
            ld_global_nc_u32(unit.add(parts.table + tid * 32 + 4).cast())
        } else {
            0
        };

        // BF16 activation A indexing (`a_row` and `a_row + 64`, 16B `a_chunk` in 0..4).
        let row_words = k / 2;
        let a_row = tid >> 2;
        let a_chunk = tid & 3;
        let a_src0 = if a_row < m { (place(in_table, in_count, base + a_row) * row_words + a_chunk * 4) as u32 } else { u32::MAX };
        let a_src1 = if a_row + 64 < m { (place(in_table, in_count, base + a_row + 64) * row_words + a_chunk * 4) as u32 } else { u32::MAX };
        let a_dst0 = swiz(a_row, a_chunk);
        let a_off = swiz(m_row_base + (lane & 15), lane >> 4);

        // For NVFP4 (`mode == 2` or `mode == 0`), finish the 1 KB `(V1 * 2^112, V3_half * 2^112)` LUT in SMEM.
        if !wide {
            let v_up: u32 = if tid == 0x7f || tid == 0xff {
                0
            } else {
                let scaled: u32;
                core::arch::asm!(
                    "mul.rn.bf16x2 {d}, {a}, {b};",
                    d = out(reg32) scaled, a = in(reg32) lut_raw, b = in(reg32) 0x7780_7780u32,
                    options(pure, nomem, nostack),
                );
                scaled
            };
            store_u32(smem + LUT_AT as u32 + (tid * 4) as u32, v_up);
        }

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

        // The A ring of `mode == 2` and `mode == 1`: 3 stages of 32 columns (`24,576 B`, `wait_group 1`).
        if mode != 0 {
            for st in 0..A_STAGES as u32 {
                let s = smem + st * A_BYTES as u32;
                if a_src0 == u32::MAX { st_shared_zero16(s + a_dst0); }
                if a_src1 == u32::MAX { st_shared_zero16(s + a_dst0 + 4096); }
            }
        }
        macro_rules! issue_a {
            ($st:expr, $c_idx:expr) => {{
                let s = smem + ($st as u32) * A_BYTES as u32;
                let at = ($c_idx as u32) * 16;
                if a_src0 != u32::MAX { crate::ptx::copy16(s + a_dst0, input.add((a_src0 + at) as usize)); }
                if a_src1 != u32::MAX { crate::ptx::copy16(s + a_dst0 + 4096, input.add((a_src1 + at) as usize)); }
            }};
        }

        if mode == 1 {
            // Coalesced MXFP8: B stage `s` (64 columns) holds the 4 row blocks of the tile copied whole (`row * 64 B`,
            // the four 16-byte parts of a row XOR-swizzled by `(row >> 1) & 3`, bank-free for the 8 rows a read). This
            // thread copies parts `tid` and `tid + 256` (512 contiguous bytes a warp); warp 0 copies the 4-byte scale
            // copies of the stage pair (`s / 2`, 128 contiguous bytes a row block). Groups: `{A(2s+2), B(s+1)}` at
            // half 0 of stage `s`, `{A(2s+3)}` at half 1, so `wait_group 1` before half `h` of stage `s` has
            // `A(2s+h)` and `B(s)` landed. B buffer `(s+1) & 1` was last read in stage `s-1`, before the barrier.
            let stages = (k / 64) as u32;
            let (g, t) = (lane >> 2, lane & 3);
            let tile_row = tile_n * PREFILL_BN;
            // Part `tid` (row block `tid >> 7`, rows 0..63 of the tile) and part `tid + 256` (row block + 2, 64 rows
            // and `k * 64` code bytes further, the same swizzle 4,096 B further in the stage).
            let b_src0 = (code_copy(tile_row + (tid >> 7) * 32, 0, 0, k, true) + (tid & 127) * 16) as u32;
            let b_dst0 = ((tid >> 2) * 64 + (((tid & 3) ^ ((tid >> 3) & 3)) * 16)) as u32;
            let b_ok0 = tile_row + (tid >> 7) * 32 < rows;
            let b_ok1 = tile_row + (tid >> 7) * 32 + 64 < rows;
            let sc_src = if tid < 32 && tile_row + (tid >> 3) * 32 < rows {
                (parts.scales + scale_copy(tile_row + (tid >> 3) * 32, 0, k, true) + (tid & 7) * 16) as u32
            } else {
                u32::MAX
            };
            for bs in 0..2u32 {
                let bb = smem + B_AT as u32 + bs * B_BYTES as u32;
                if !b_ok0 { st_shared_zero16(bb + b_dst0); }
                if !b_ok1 { st_shared_zero16(bb + b_dst0 + 4096); }
                if tid < 32 && sc_src == u32::MAX {
                    st_shared_zero16(smem + (LUT_AT + tid * 16) as u32 + bs * MX_SCALE_BYTES as u32);
                }
            }
            macro_rules! issue_b {
                ($s:expr) => {{
                    let s_idx: u32 = $s;
                    let bb = smem + B_AT as u32 + (s_idx & 1) * B_BYTES as u32 + b_dst0;
                    let src = unit.add((b_src0 + s_idx * 2048) as usize);
                    if b_ok0 { crate::ptx::copy16(bb, src.cast()); }
                    if b_ok1 { crate::ptx::copy16(bb + 4096, src.add(k * 64).cast()); }
                    if sc_src != u32::MAX {
                        crate::ptx::copy16(
                            smem + (LUT_AT + tid * 16) as u32 + (s_idx & 1) * MX_SCALE_BYTES as u32,
                            unit.add((sc_src + (s_idx >> 1) * 128) as usize).cast(),
                        );
                    }
                }};
            }

            issue_a!(0, 0);
            issue_b!(0);
            crate::ptx::commit();
            issue_a!(1, 1);
            crate::ptx::commit();
            let b_r0 = warp_n * 32 + g;
            let mut a_cur = 0u32;
            let mut s = 0u32;
            macro_rules! mx_half {
                ($h:expr) => {{
                    let h: u32 = $h;
                    cp_async_wait_one();
                    core::arch::nvptx::_syncthreads();
                    let a_wr = if a_cur == 0 { 2 } else { a_cur - 1 };
                    if s + 1 < stages {
                        issue_a!(a_wr, 2 * s + 2 + h);
                        if h == 0 { issue_b!(s + 1); }
                    }
                    crate::ptx::commit();
                    if subs > 0 {
                        let a_stage = smem + a_cur * A_BYTES as u32;
                        let st_b = smem + B_AT as u32 + (s & 1) * B_BYTES as u32;
                        let sc_b = smem + LUT_AT as u32 + (s & 1) * MX_SCALE_BYTES as u32;
                        let sc_shift = ((s & 1) * 2 + h) * 8;
                        macro_rules! load_mx_row {
                            ($row:expr, $k_sub:expr) => {{
                                let row: usize = $row;
                                let sc = (load_u32(sc_b + (row * 4) as u32) >> sc_shift) & 255;
                                let (p1, p2, two) = if sc <= 142 {
                                    (((sc + 112) << 7) * 0x0001_0001, 0, false)
                                } else {
                                    (0x7780_7780, (sc << 7) * 0x0001_0001, true)
                                };
                                let part = (h as usize * 2 + $k_sub) ^ ((row >> 1) & 3);
                                let row_off = st_b + (row * 64 + part * 16) as u32;
                                let c0 = crate::ptx::load_u16(row_off + (t * 2) as u32) as u16;
                                let c1 = crate::ptx::load_u16(row_off + (8 + t * 2) as u32) as u16;
                                [unpack_mx_pair!(c0, p1, p2, two), unpack_mx_pair!(c1, p1, p2, two)]
                            }};
                        }
                        for k_sub in 0..2usize {
                            let a_at = a_stage + if k_sub == 0 { a_off } else { a_off ^ 32 };
                            let r0 = load_mx_row!(b_r0, k_sub);
                            let r1 = load_mx_row!(b_r0 + 8, k_sub);
                            let r2 = load_mx_row!(b_r0 + 16, k_sub);
                            let r3 = load_mx_row!(b_r0 + 24, k_sub);
                            let bp0 = [r0[0], r0[1], r1[0], r1[1]];
                            let bp1 = [r2[0], r2[1], r3[0], r3[1]];
                            k_sub_mma!(a_at, bp0, bp1);
                        }
                    }
                    a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };
                }};
            }
            while s < stages {
                mx_half!(0);
                mx_half!(1);
                s += 1;
            }
        } else if mode == 2 {
            // In-place MMA-lane NVFP4 panel path (`pack_expert_tagged`):
            // - 3-stage `cp.async.cg.L2::256B` streamed A (`24,576 B`, `wait_group 1`) + 2-stage unpacked B (`16,384 B`) + 1 KB LUT.
            // - `.L2::256B` codes + `.L2::128B` scales WITHOUT `evict_first` so `PREFILL_M_GROUP = 8` shares weights in L2.
            // - Raw words of stage `s + 2` are fetched into registers at the end of half 0 of stage `s` (three halves before unpack).
            // - A warp unpacks one code word (`16 x 16` weights) to the other B stage inside each k16 slab of 16 MMAs.
            let stages = (k / 64) as u32;
            let b_valid = tile_n * PREFILL_BN + warp * 16 < rows;
            let (b_code0, b_scale0) = if b_valid {
                let chunk0 = ((tile_n * 2 + (warp >> 2)) * (k >> 6)) * 4 + (warp & 3);
                ((chunk0 * 512 + lane * 16) as u32, (parts.scales + chunk0 * 64 + (lane >> 2) * 8) as u32)
            } else {
                (0, 0)
            };
            let b_wr0 = smem + B_AT as u32 + (warp * 1024 + lane * 16) as u32;
            let s0_b = smem + B_AT as u32 + (warp_n * 2 * 1024 + lane * 16) as u32;

            macro_rules! issue_a_stream {
                ($st:expr, $c_idx:expr) => {{
                    let s = smem + ($st as u32) * A_BYTES as u32;
                    let at = ($c_idx as u32) * 16;
                    if a_src0 != u32::MAX { crate::ptx::copy16_stream(s + a_dst0, input.add((a_src0 + at) as usize)); }
                    if a_src1 != u32::MAX { crate::ptx::copy16_stream(s + a_dst0 + 4096, input.add((a_src1 + at) as usize)); }
                }};
            }

            // The 16 code bytes and 8 scale bytes of this lane for stage `$s`. A panel past `rows` or a stage past
            // `stages` gives codes 0 with scale byte `0x7f`, whose LUT word is 0: the unpacked weights are `+0.0`.
            macro_rules! fetch_b64 {
                ($s:expr) => {{
                    let st_idx: u32 = $s;
                    if b_valid && st_idx < stages {
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
                        ([0u32; 4], [0x7f7f_7f7fu32; 2])
                    }
                }};
            }
            // Expand a raw LUT word `(V1*2^112, V3_half*2^112)` into the two BF16x2 operands of `unpack_nv_byte!`.
            macro_rules! expand_scale {
                ($st_raw:expr) => {{
                    let st_raw: u32 = $st_raw;
                    let (v1, v3): (u32, u32);
                    core::arch::asm!(
                        "prmt.b32 {v1}, {st}, {st}, 0x1010; \
                         prmt.b32 {v3}, {st}, {st}, 0x3232;",
                        v1 = out(reg32) v1, v3 = out(reg32) v3,
                        st = in(reg32) st_raw,
                        options(pure, nomem, nostack),
                    );
                    [v1, v3]
                }};
            }
            // The LUT word of scale byte `$sc & 255` as the two operands of `unpack_nv_byte!`.
            macro_rules! scale_x2 {
                ($sc:expr) => {
                    expand_scale!(load_u32(smem + LUT_AT as u32 + (($sc) & 255) * 4))
                };
            }
            macro_rules! mma_row {
                ($a:ident, $bp0:ident, $bp1:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {{
                    $c0 = mma_m16n8k16_bf16($a, [$bp0[0], $bp0[1]], $c0);
                    $c1 = mma_m16n8k16_bf16($a, [$bp0[2], $bp0[3]], $c1);
                    $c2 = mma_m16n8k16_bf16($a, [$bp1[0], $bp1[1]], $c2);
                    $c3 = mma_m16n8k16_bf16($a, [$bp1[2], $bp1[3]], $c3);
                }};
            }
            macro_rules! slab_full {
                ($a_at:expr, $b_at:expr, $w:expr, $sc:expr, $dst:expr) => {{
                    let (a_at, b_at, w, dst, sc): (u32, u32, u32, u32, u32) = ($a_at, $b_at, $w, $dst, $sc);
                    let bp0 = crate::ptx::load4(b_at);
                    let bp1 = crate::ptx::load4(b_at + 1024);
                    let a0 = ldmatrix_x4(a_at);
                    let a1 = ldmatrix_x4(a_at + TILE16);
                    let st_raw = load_u32(smem + LUT_AT as u32 + (sc & 255) * 4);
                    let sb_raw = load_u32(smem + LUT_AT as u32 + ((sc >> 8) & 255) * 4);
                    mma_row!(a0, bp0, bp1, c00, c01, c02, c03);
                    let pt = expand_scale!(st_raw);
                    let o0 = unpack_nv_byte!(w, pt[0], pt[1]);
                    let a2 = ldmatrix_x4(a_at + 2 * TILE16);
                    mma_row!(a1, bp0, bp1, c10, c11, c12, c13);
                    let o1 = unpack_nv_byte!(w >> 16, pt[0], pt[1]);
                    let a3 = ldmatrix_x4(a_at + 3 * TILE16);
                    let pb = expand_scale!(sb_raw);
                    mma_row!(a2, bp0, bp1, c20, c21, c22, c23);
                    let o2 = unpack_nv_byte!(w >> 8, pb[0], pb[1]);
                    mma_row!(a3, bp0, bp1, c30, c31, c32, c33);
                    let o3 = unpack_nv_byte!(w >> 24, pb[0], pb[1]);
                    st_shared_v4(dst, [o0, o1, o2, o3]);
                }};
            }
            macro_rules! slab_part {
                ($a_at:expr, $b_at:expr, $w:expr, $sc:expr, $dst:expr) => {{
                    let (a_at, b_at, w, dst, sc): (u32, u32, u32, u32, u32) = ($a_at, $b_at, $w, $dst, $sc);
                    let st_raw = load_u32(smem + LUT_AT as u32 + (sc & 255) * 4);
                    let sb_raw = load_u32(smem + LUT_AT as u32 + ((sc >> 8) & 255) * 4);
                    if subs > 0 {
                        let bp0 = crate::ptx::load4(b_at);
                        let bp1 = crate::ptx::load4(b_at + 1024);
                        let a0 = ldmatrix_x4(a_at);
                        mma_row!(a0, bp0, bp1, c00, c01, c02, c03);
                        if subs > 1 {
                            let a1 = ldmatrix_x4(a_at + TILE16);
                            mma_row!(a1, bp0, bp1, c10, c11, c12, c13);
                            if subs > 2 {
                                let a2 = ldmatrix_x4(a_at + 2 * TILE16);
                                mma_row!(a2, bp0, bp1, c20, c21, c22, c23);
                            }
                        }
                    }
                    let pt = expand_scale!(st_raw);
                    let o0 = unpack_nv_byte!(w, pt[0], pt[1]);
                    let o1 = unpack_nv_byte!(w >> 16, pt[0], pt[1]);
                    let pb = expand_scale!(sb_raw);
                    let o2 = unpack_nv_byte!(w >> 8, pb[0], pb[1]);
                    let o3 = unpack_nv_byte!(w >> 24, pb[0], pb[1]);
                    st_shared_v4(dst, [o0, o1, o2, o3]);
                }};
            }
            macro_rules! half_put {
                ($a_stage:expr, $b_at:expr, $w0:expr, $sc0:expr, $w1:expr, $sc1:expr, $dst:expr) => {{
                    let (a_stage, b_at, dst): (u32, u32, u32) = ($a_stage, $b_at, $dst);
                    if subs == 4 {
                        slab_full!(a_stage + a_off, b_at, $w0, $sc0, dst);
                        slab_full!(a_stage + (a_off ^ 32), b_at + 512, $w1, $sc1, dst + 512);
                    } else {
                        slab_part!(a_stage + a_off, b_at, $w0, $sc0, dst);
                        slab_part!(a_stage + (a_off ^ 32), b_at + 512, $w1, $sc1, dst + 512);
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

            // Prologue: fetch raw B stages 0 and 1 FIRST alongside A steps 0 and 1, barrier for LUT, then unpack B0.
            let (c0_q, c0_sc) = fetch_b64!(0);
            let (mut n_q, mut n_sc) = fetch_b64!(1);
            issue_a_stream!(0, 0);
            crate::ptx::commit();
            issue_a_stream!(1, 1);
            crate::ptx::commit();
            core::arch::nvptx::_syncthreads();
            {
                let (t0, b0) = (scale_x2!(c0_sc[0]), scale_x2!(c0_sc[0] >> 8));
                st_shared_v4(b_wr0, [
                    unpack_nv_byte!(c0_q[0], t0[0], t0[1]), unpack_nv_byte!(c0_q[0] >> 16, t0[0], t0[1]),
                    unpack_nv_byte!(c0_q[0] >> 8, b0[0], b0[1]), unpack_nv_byte!(c0_q[0] >> 24, b0[0], b0[1]),
                ]);
                let (t1, b1) = (scale_x2!(c0_sc[0] >> 16), scale_x2!(c0_sc[0] >> 24));
                st_shared_v4(b_wr0 + 512, [
                    unpack_nv_byte!(c0_q[1], t1[0], t1[1]), unpack_nv_byte!(c0_q[1] >> 16, t1[0], t1[1]),
                    unpack_nv_byte!(c0_q[1] >> 8, b1[0], b1[1]), unpack_nv_byte!(c0_q[1] >> 24, b1[0], b1[1]),
                ]);
            }
            let (mut c_q2, mut c_q3, mut c_sc1) = (c0_q[2], c0_q[3], c0_sc[1]);

            let b1_rd = s0_b + B_BYTES as u32;
            let b1_wr = b_wr0 + B_BYTES as u32;
            let mut a_cur = 0u32;
            let mut s = 0u32;
            while s < stages {
                let more = s + 1 < stages;
                // Half 0 (`k = 0..32` of `s`): A[a_cur] and B stage 0 are visible; unpack `c_q2/c_q3/c_sc1` -> B1,
                // then immediately fetch `s + 2` (since `c_q2/c_q3/c_sc1` are dead, max 12 live raw regs!).
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
                let a_wr = if a_cur == 0 { 2 } else { a_cur - 1 };
                if more { issue_a_stream!(a_wr, 2 * s + 2); }
                crate::ptx::commit();
                let a_stage = smem + a_cur * A_BYTES as u32;
                half_put!(a_stage, s0_b, c_q2, c_sc1, c_q3, c_sc1 >> 16, b1_wr);
                let (f_q, f_sc) = if more { fetch_b64!(s + 2) } else { ([0u32; 4], [0x7f7f_7f7fu32; 2]) };
                a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };

                // Half 1 (`k = 32..64` of `s`): A[a_cur] and B stage 1 are visible; unpack `n_q[0..2]` -> B0 of `s + 1`.
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
                let a_wr = if a_cur == 0 { 2 } else { a_cur - 1 };
                if more { issue_a_stream!(a_wr, 2 * s + 3); }
                crate::ptx::commit();
                let a_stage = smem + a_cur * A_BYTES as u32;
                if more {
                    half_put!(a_stage, b1_rd, n_q[0], n_sc[0], n_q[1], n_sc[0] >> 16, b_wr0);
                    c_q2 = n_q[2];
                    c_q3 = n_q[3];
                    c_sc1 = n_sc[1];
                    n_q = f_q;
                    n_sc = f_sc;
                } else {
                    k_sub_panel!(a_stage + a_off, b1_rd);
                    k_sub_panel!(a_stage + (a_off ^ 32), b1_rd + 512);
                }
                a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };
                s += 1;
            }
        } else {
            // Shipped tiled NVFP4 layout (`mode == 0`):
            // 3 stages of `K = 32` (`12,800 B` each -> `38,400 B <= 40,960 B`, `wait_group 1`, 1 barrier per 32 cols),
            // `cp.async.cg` on codes (L1-bypass hardware async!), `half_m` split, pipelined `LDSM`, 8-inst NVFP4 unpack.
            let steps = (k / 32) as u32;
            let (g, t) = (lane >> 2, lane & 3);
            let b_row = tid >> 1;
            let b_half = tid & 1;
            let global_b_row = tile_n * PREFILL_BN + b_row;
            let b_valid = global_b_row < rows;
            let (b_code_off, b_scale_off): (u32, u32) = if b_valid && b_half == 0 {
                (code_copy(global_b_row, 0, 0, k, false) as u32, (parts.scales + scale_copy(global_b_row, 0, k, false)) as u32)
            } else {
                (u32::MAX, u32::MAX)
            };
            let b_dst_off = (A_BYTES + b_row * 16) as u32;
            let b_sc_dst_off = (A_BYTES + PREFILL_BN * 32 + b_row * 4) as u32;

            for st in 0..3u32 {
                let st_base = smem + st * STAGE32 as u32;
                if a_src0 == u32::MAX { st_shared_zero16(st_base + a_dst0); }
                if a_src1 == u32::MAX { st_shared_zero16(st_base + a_dst0 + 4096); }
                if !b_valid && b_half == 0 {
                    st_shared_zero16(st_base + b_dst_off);
                    store_u32(st_base + b_sc_dst_off, 0);
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
                        let b_co = (step_u >> 1) * 1024 + (step_u & 1) * 16;
                        core::arch::asm!(
                            "cp.async.cg.shared.global [{d}], [{p}], 16;",
                            d = in(reg32) (st_base + b_dst_off),
                            p = in(reg64) unit.add((b_code_off + b_co) as usize),
                            options(nostack),
                        );
                    }
                    if b_scale_off != u32::MAX {
                        let b_so = (step_u >> 1) * 128;
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
        let m_step = load_u32(meta_p + 20) as usize;
        let m_row_base = warp_m * m_step;
        let subs = m.saturating_sub(m_row_base).min(m_step).div_ceil(16);
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
        // Pair adjacent BF16 columns via `shfl_xor_u32(..., 1)` -> bank-free 32-bit `store_u32` (0 conflicts, 1/2 stores)
        // and start row 0's `coefficients` LUT reads before the barrier to hide global latency.
        if format == PREFILL_OUT_BF16_GATED {
            let col_w = warp_n * 16;
            macro_rules! pack_gate_one {
                ($a:expr) => {{
                    let a = $a;
                    let h0 = capped!(a[0], a[1]).to_bits() >> 16;
                    let h1 = capped!(a[2], a[3]).to_bits() >> 16;
                    let p0 = h0 | (shfl_xor_u32(h0, 1) << 16);
                    let p1 = h1 | (shfl_xor_u32(h1, 1) << 16);
                    [p0, p1]
                }};
            }
            let r0_p = if subs > 0 {
                [pack_gate_one!(c00), pack_gate_one!(c01), pack_gate_one!(c02), pack_gate_one!(c03)]
            } else {
                [[0u32; 2]; 4]
            };
            core::arch::nvptx::_syncthreads();
            macro_rules! store_gate_pair {
                ($r0:expr, $r1:expr, $ni:expr, $p:expr) => {{
                    let p = $p;
                    if (t & 1) == 0 {
                        let c_off = col_w + $ni * 4 + t;
                        store_u32(smem + (($r0 * 72 + c_off) * 2) as u32, p[0]);
                        store_u32(smem + (($r1 * 72 + c_off) * 2) as u32, p[1]);
                    }
                }};
            }
            if subs > 0 {
                let r0 = m_row_base + g;
                let r1 = r0 + 8;
                store_gate_pair!(r0, r1, 0, r0_p[0]);
                store_gate_pair!(r0, r1, 1, r0_p[1]);
                store_gate_pair!(r0, r1, 2, r0_p[2]);
                store_gate_pair!(r0, r1, 3, r0_p[3]);
            }
            macro_rules! stage_gate_row {
                ($mi:expr, $d0:ident, $d1:ident, $d2:ident, $d3:ident) => {
                    if subs > $mi {
                        let r0 = m_row_base + $mi * 16 + g;
                        let r1 = r0 + 8;
                        let p0 = pack_gate_one!($d0);
                        let p1 = pack_gate_one!($d1);
                        let p2 = pack_gate_one!($d2);
                        let p3 = pack_gate_one!($d3);
                        store_gate_pair!(r0, r1, 0, p0);
                        store_gate_pair!(r0, r1, 1, p1);
                        store_gate_pair!(r0, r1, 2, p2);
                        store_gate_pair!(r0, r1, 3, p3);
                    }
                };
            }
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
