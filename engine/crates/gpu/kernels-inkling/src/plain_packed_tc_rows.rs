//! Tensor-core BF16 multi-row packed MoE expert product (`resident_launch_plain_packed_tc_rows`,
//! kind `PLAIN_PACKED_TC_ROWS = 106`).
//!
//! Computes `Y = X * W_packed^T` for device-routed MoE experts stored in their shipped tiled packed
//! NVFP4 (`wide = false`, 4-bit codes + 1 scale/16 cols + 8 KB BF16 table) or MXFP8 (`wide = true`,
//! 8-bit codes + 1 power-of-two scale/32 cols + 1 KB FP32 table) units (`packed_product::layout`),
//! reading `tile_meta` produced by `MOE_GROUP_ROUTE` with ZERO widened BF16 duplicate and ZERO CPU sync.
//!
//! Unpacks each `128 x 32` weight tile on the fly into XOR-swizzled BF16 shared memory and executes
//! the exact SM121a `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` reduction in strictly
//! ascending `k = 0, 16, 32, ..., K - 16` order from `+0.0`, producing bit-for-bit identical results
//! to `PLAIN_TC_ROWS` Mode 1 on the widened BF16 expert stack. A word depends only on its own input row and
//! weight row, so the tile shape below does not change any word.
//!
//! Tile: 128 input rows (the two route tiles `2r` and `2r + 1` of one expert) x 128 stored rows
//! (`TC_PACKED_BN`) x 32 columns a step, 256 threads = 8 warps (2 row warps x 4 column warps), each warp `64 x 32`
//! (4 x 4 MMA tiles, 32 MMAs a step between two barriers). A warp skips the 16-row MMA tiles above the rows of its
//! block. The block unpacks each weight once for up to 128 input rows and reads each input value once for 128 stored
//! rows.
//!
//! Input: BF16 rows (`packed_product::HALF_IN` in payload word 17), two a word, copied with `cp.async` one step
//! ahead. The host rounds an FP32 input with `cvt.rn.bf16x2.f32` first (`TC_MODE_PACK`): the same rounding the MMA
//! input had before.
//!
//! Shared memory (`40,960` bytes static, 2 blocks/SM on SM121a):
//! - Two A stages (`128 x 32` BF16 = 8,192 B each) = `16,384` B.
//! - Two B stages (`128 x 32` BF16 = 8,192 B each) = `16,384` B.
//! - Expert weight lookup table (`ROWS_TABLE_BYTES = 8,192` B), copied once per CTA in prologue.
//!
//! Launch: grid = `tc_packed_moe_blocks(rows, max_m_tiles)`, 256 threads, no dynamic shared memory. The block of an odd
//! route tile of its expert returns at once (its even partner takes its rows).
//!
//! Gated form (prefill `GateUp`, payload word 15 non-null): with `PAIRS`, the lane that holds stored rows `2i`
//! (gate `i`) and `2i + 1` (up `i`) of one input row writes `rn(gate_table[rn(gate)] * up_table[rn(up)])` to
//! word `i` of a `rows / 2` wide output row, from the two exact coefficient tables of `FAST_CAPPED_GATE`
//! (mode 2). These are the words of the separate `FAST_CAPPED_GATE` launch on the ungated output, with no
//! `[all, rows]` FP32 buffer written or read. With word 16 = `HALF_OUT` the gated value is stored as the BF16 value
//! that `cvt.rn.bf16x2.f32` makes of it (one 16-bit value a gated column): the input of the `Down` product as it
//! rounds it.
//!
//! Half form (prefill `Down`, payload word 16 = `HALF_OUT`): an ungated launch with no `PAIRS` and even `rows`
//! stores each output as `rn(output)` in BF16, two a word (the even column in the low half), and `stride` counts
//! BF16 values. `RouteFold` with `FOLD_HALF` reads them; it rounds each down word with `rn` in either form.

use crate::fast_ops::CAPPED_GATE_TABLE_WORDS;
use crate::fixed_math::rn;
use crate::packed_product::{COLS, HALF_IN, HALF_OUT, PAIRS, ROWS_TABLE_BYTES, WIDE, code_copy, layout, output_row, power, scale_copy, table_copies};
use crate::tc_ptx::{
    cp_async_wait_one, cp_async_wait_zero, cvt_bf16x2, l2_evict_first, ld_global_nc_u8, ld_global_nc_u32,
    ld_global_nc_v2_u32, ld_global_nc_v4_u32, ld_global_nc_v4_u32_hint, ldmatrix_x4, mma_m16n8k16_bf16,
    shfl_xor_u32, st_shared_v4, st_shared_zero16,
};
use crate::tc_rows::{TC_BK, TC_BM, TC_PACKED_BN, TC_THREADS, tc_packed_moe_admitted};

/// Input rows of a block: two route tiles of one expert.
const PAIR_M: usize = 2 * TC_BM;
const A_BYTES: usize = PAIR_M * TC_BK * 2; // 8,192 bytes
const B_BYTES: usize = TC_PACKED_BN * TC_BK * 2; // 8,192 bytes
const A_STAGES: usize = 2;
const B_STAGES: usize = 2;
const B_AT: usize = A_STAGES * A_BYTES; // 16,384 bytes
const TABLE_AT: usize = B_AT + B_STAGES * B_BYTES; // 32,768 bytes (16B aligned)
const BLOCK_BYTES: usize = TABLE_AT + ROWS_TABLE_BYTES; // 40,960 bytes
/// Register-NVFP4 fast path (`mode == 3`) drops the 8 KB SMEM table and uses 3 A stages (24 KB) + 2 B stages (16 KB).
const REG_A_STAGES: usize = 3;
const REG_B_AT: usize = REG_A_STAGES * A_BYTES; // 24,576 bytes
/// Bytes between two 16-row MMA tiles of a stage (16 rows of 64 bytes).
const TILE16: u32 = 16 * 64;
const _: () = assert!(TC_BM == 64 && TC_PACKED_BN == 128 && TC_BK == 32 && TC_THREADS == 256 && PAIR_M == 128);
const _: () = assert!(B_AT == 16_384 && TABLE_AT == 32_768 && BLOCK_BYTES == 40_960 && TABLE_AT % 16 == 0);
const _: () = assert!(REG_B_AT + B_STAGES * B_BYTES == BLOCK_BYTES);

/// Zero-bank-conflict XOR-swizzled byte offset within a 64-byte-stride BF16 tile (`row`, 16B `chunk` in 0..4).
#[inline(always)]
const fn swiz(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Replay one recorded command of kind `PLAIN_PACKED_TC_ROWS`.
/// SAFETY (caller contract, `resident_launch_plain_packed_tc_rows`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `PLAIN_PACKED_TC_ROWS` recorded by `Backend::project_tc_packed_moe_into`,
/// `Backend::project_tc_packed_gate_into` or their half forms after `tc_rows::check_tc_packed_moe_form` verified the
/// mode-specific extents, alignments, and non-overlap.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind traps before reading payload words.
    // - Host verified via `tc_rows::check_tc_packed_moe_form` that `stack` holds `experts` contiguous 16B-aligned
    //   packed units of `unit_bytes = packed_product::layout(rows, cols, wide).bytes`, `tile_meta` holds
    //   `1 + 3 * max_m_tiles` words written by `MOE_GROUP_ROUTE`, `input` holds `in_count * cols` BF16 values (two a
    //   word, 16B aligned; word 17 = `HALF_IN`), and any non-null `in_table` / `out_table` holds `all` u32 words.
    //   Ungated output holds `out_count * rows` FP32 words with payload stride `rows` (BF16 values with `HALF_OUT`);
    //   gated output holds `out_count * (rows / 2)` FP32 words (BF16 values with `HALF_OUT`) with payload stride
    //   `rows / 2`. In every form it is disjoint from every input and table.
    // - Device re-checks `tc_packed_moe_admitted`, `stride >= width` (`rows` ungated, `rows / 2` gated), alignment
    //   of all tables, word 17, `total_m_tiles <= max_m_tiles`, `expert < experts`, each route tile's `m_cnt in
    //   1..=TC_BM`, that a paired tile follows its partner (`first_pos + TC_BM`, partner full), `first_pos + m <=
    //   all`, and every table index `< in_count` / `< out_count` before dereference.
    // - `cols` is a multiple of 64 (`tc_packed_moe_shape_admitted`), so a BF16 input row is `cols / 2` words, a whole
    //   number of 16-byte copies, and every A copy of step `c_idx` (`row * cols / 2 + c_idx * 16 + chunk * 4` words,
    //   `row < in_count`, `c_idx < cols / 32`, `chunk < 4`) lies inside `input` and is 16-byte aligned.
    // - For every valid row `b_r < rows` and `c_idx < cols / 32`, `code_copy(b_r, stage64, 0, k, wide)` +
    //   `lane_col / 2` (NVFP4, 4B aligned, 4 bytes) or + `lane_col` (MXFP8, 8B aligned, 8 bytes) lies strictly
    //   inside the unit's code section, and `scale_copy` + sub-index lies strictly inside its scale section;
    //   table copies stay strictly inside `table_bytes(wide)` <= `ROWS_TABLE_BYTES`.
    // - Output stores use `output_row(col, rows, pairs)`, a bijection on `0..rows` (host-tested), so every
    //   `(out_r, out_c)` has a single writer and lies inside `output`.
    // - Gated (word 15 non-null): host verified that `pairs` holds, `output` holds `out_count * rows / 2` FP32
    //   words (BF16 values with `HALF_OUT`), and word 15 points to `2 * CAPPED_GATE_TABLE_WORDS` FP32 words (word
    //   aligned, disjoint from `output`) whose build launch was recorded before this one. Device re-checks `pairs`,
    //   alignment and `stride >= rows / 2`. A store goes to `col / 2 < rows / 2` of its row: one writer per value
    //   (the lane with even `col`). Table indices are the upper 16 bits of an FP32 word, so `< CAPPED_GATE_TABLE_WORDS`.
    // - Half (word 16 = `HALF_OUT`, ungated): host verified that the launch has no `PAIRS`, `rows` is even, and
    //   `output` holds `out_count * rows / 2` words. Device re-checks those. `col` is even and `col + 1 < rows`,
    //   so the word `(out_r * stride + col) / 2` (stride = rows, even) holds columns `col` and `col + 1` of its
    //   row: one writer per word, inside `output`.
    // - Shared memory stays strictly inside the block's static `BLOCK_BYTES` (40,960 bytes).
    // - All 256 threads participate in every `bar.sync 0` (a block returns whole before its first barrier) and
    //   every warp-synchronous `ldmatrix` / `mma.sync` (each tile condition is uniform over a warp).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_PACKED_TC_ROWS { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let stack = *words as *const u8;
        let input = *words.add(1) as *const u32;
        let _output = *words.add(2) as *mut f32;
        let rows = *words.add(3) as usize;
        let col_word = *words.add(4) as u32;
        let k = (col_word & COLS) as usize;
        let wide = col_word & WIDE != 0;
        let pairs = col_word & PAIRS != 0;
        let all = *words.add(5) as usize;
        let stride = *words.add(6) as usize;
        let unit_bytes = *words.add(7) as usize;
        let in_table = *words.add(8) as *const u32;
        let in_count = *words.add(9) as usize;
        let out_table = *words.add(10) as *const u32;
        let _out_count = *words.add(11) as usize;
        let tile_meta = *words.add(12) as *const u32;
        let max_m_tiles = *words.add(13) as usize;
        let experts = *words.add(14) as usize;
        // Word 15: null, or the two exact `CappedGate` coefficient tables (gated `GateUp`, see the module notes).
        let coefficients = *words.add(15) as *const f32;
        let gated = !coefficients.is_null();
        // Word 16: `HALF_OUT` stores BF16 outputs (see the module notes), else 0.
        let half = *words.add(16) == HALF_OUT;
        // Word 17: `HALF_IN`, the input is BF16 rows. No other input form is read.
        let half_in = *words.add(17) == HALF_IN;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;

        let width = if gated { rows / 2 } else { rows };
        if ((in_table as usize | out_table as usize | tile_meta as usize | coefficients as usize) & 3) != 0
            || tile_meta.is_null() || stride < width || (gated && !pairs) || !half_in
            || (half && !gated && (pairs || rows % 2 != 0 || stride % 2 != 0))
        {
            core::arch::nvptx::trap();
        }
        if !tc_packed_moe_admitted(rows, k, wide, pairs, all, experts, max_m_tiles, unit_bytes, stack as usize, input as usize, threads) {
            core::arch::nvptx::trap();
        }

        let n_tiles = rows.div_ceil(TC_PACKED_BN);
        let tile_n = block % n_tiles;
        let tm = block / n_tiles;
        let total_m_tiles = tile_meta.read() as usize;
        if total_m_tiles > max_m_tiles { core::arch::nvptx::trap(); }
        if tm >= total_m_tiles || tile_n * TC_PACKED_BN >= rows { return; }

        // Route tile `t`: (expert, first_pos, m_count). The route writes the tiles of each expert one after the other,
        // in rising expert order, each full but the last.
        let meta = |t: usize| -> (usize, usize, usize) {
            let entry = tile_meta.add(1 + 3 * t);
            (entry.read() as usize, entry.add(1).read() as usize, entry.add(2).read() as usize)
        };
        let (expert, base, m_first) = meta(tm);
        if expert >= experts || m_first == 0 || m_first > TC_BM { core::arch::nvptx::trap(); }
        // Pairs of route tiles: the first route tile of this expert (binary search over the rising expert numbers), then
        // the place `nth` of tile `tm` among the tiles of its expert. Places 2r and 2r + 1 make one block of 128 rows;
        // the block of an odd place returns (no barrier has run).
        let (mut lo, mut hi) = (0usize, tm);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if meta(mid).0 < expert { lo = mid + 1; } else { hi = mid; }
        }
        let nth = tm - lo;
        if meta(lo).0 != expert || meta(lo).1 + nth * TC_BM != base { core::arch::nvptx::trap(); }
        if nth % 2 == 1 { return; }
        let m_second = if tm + 1 < total_m_tiles {
            let (next_expert, next_base, next_m) = meta(tm + 1);
            if next_expert == expert {
                if m_first != TC_BM || next_base != base + TC_BM || next_m == 0 || next_m > TC_BM { core::arch::nvptx::trap(); }
                next_m
            } else {
                0
            }
        } else {
            0
        };
        let m = m_first + m_second;
        if base + m > all { core::arch::nvptx::trap(); }

        let unit = stack.add(expert * unit_bytes);
        let parts = layout(rows, k, wide);
        // The index math below is 32-bit, reserving `u32::MAX` as the invalid-row sentinel.
        if in_count * (k / 2) >= u32::MAX as usize || unit_bytes >= u32::MAX as usize { core::arch::nvptx::trap(); }
        let (mode, factor): (u32, f32) = if !wide && rows % 64 == 0 && k % 64 == 0 && {
            let t32 = unit.add(parts.table).cast::<u32>();
            t32.add(0x7f * 8).read() == crate::packed_product::NVFP4_PANEL_MAGIC_LO
                && t32.add(0x7f * 8 + 1).read() == crate::packed_product::NVFP4_PANEL_MAGIC_HI
        } {
            let t32 = unit.add(parts.table).cast::<u32>();
            if t32.add(0xff * 8).read() == crate::packed_product::NVFP4_FACTOR_MAGIC_LO
                && t32.add(0xff * 8 + 1).read() == crate::packed_product::NVFP4_FACTOR_MAGIC_HI
            {
                (3, f32::from_bits(t32.add(0xff * 8 + 2).read()))
            } else {
                (2, 0.0f32)
            }
        } else if wide {
            (1, 0.0f32)
        } else {
            (0, 0.0f32)
        };

        let place = |tbl: *const u32, count: usize, t: usize| -> usize {
            let row = if tbl.is_null() { t } else { tbl.add(t).read() as usize };
            if row >= count { core::arch::nvptx::trap(); }
            row
        };

        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (warp_m, warp_n) = (warp / 4, warp % 4);
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        // 16-row MMA tiles of this warp that hold rows of the block (0 to 4); the rest are skipped.
        let subs = m.saturating_sub(warp_m * 64).min(64).div_ceil(16);

        // Copy expert's weight lookup table into shared memory once per CTA for LUT modes (`mode < 3`).
        // `mode == 3` (NVFP4) decodes directly in SM121a registers (`cvt.rn.f16x2.{e4m3x2,e2m1x2}`).
        if mode < 3 {
            let mut t_copy = tid;
            let t_copies = table_copies(wide);
            while t_copy < t_copies {
                crate::ptx::copy16(smem + TABLE_AT as u32 + (t_copy * 16) as u32, unit.add(parts.table + t_copy * 16).cast());
                t_copy += TC_THREADS;
            }
        }

        // A copies of this thread: rows `a_row` and `a_row + 64`, 16-byte chunk `a_chunk` of the 64-byte step row.
        // Word offsets into `input` (`u32::MAX` marks rows >= m). `swiz(r + 64, c) == swiz(r, c) + 4096`.
        let row_words = k / 2;
        let a_row = tid / 4;
        let a_chunk = tid % 4;
        let a_src0 = if a_row < m { (place(in_table, in_count, base + a_row) * row_words + a_chunk * 4) as u32 } else { u32::MAX };
        let a_src1 = if a_row + 64 < m { (place(in_table, in_count, base + a_row + 64) * row_words + a_chunk * 4) as u32 } else { u32::MAX };
        let a_dst0 = swiz(a_row, a_chunk);
        // Rows above `m` are never copied: zero them once in every stage (3 A stages in `mode >= 3`, else 2).
        let a_stages = if mode >= 3 { REG_A_STAGES } else { A_STAGES };
        let mut stage = 0usize;
        while stage < a_stages {
            let s = smem + (stage * A_BYTES) as u32;
            if a_src0 == u32::MAX { st_shared_zero16(s + a_dst0); }
            if a_src1 == u32::MAX { st_shared_zero16(s + a_dst0 + 4096); }
            stage += 1;
        }
        // The A copies of step `c_idx` into stage `stage`.
        let issue_a = |stage: u32, c_idx: u32| {
            let s = smem + stage * A_BYTES as u32;
            if a_src0 != u32::MAX { crate::ptx::copy16(s + a_dst0, input.add((a_src0 + c_idx * 16) as usize)); }
            if a_src1 != u32::MAX { crate::ptx::copy16(s + a_dst0 + 4096, input.add((a_src1 + c_idx * 16) as usize)); }
        };

        // ldmatrix per-lane swizzled A offset for k_sub = 0; k_sub = 1 is `^ 32` (chunks 2, 3), next 16-row tile `+ TILE16`.
        let a_off = swiz(warp_m * 64 + (lane & 15), lane >> 4);

        // 16 MMA accumulators (4 M-tiles x 4 N-tiles) kept strictly in registers (zero stack frame).
        let (mut c00, mut c01, mut c02, mut c03) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c10, mut c11, mut c12, mut c13) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c20, mut c21, mut c22, mut c23) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c30, mut c31, mut c32, mut c33) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        // 64-column stages (`cols` is a multiple of 64, `tc_packed_moe_shape_admitted`); two 32-column steps each.
        let stages = (k / 64) as u32;

        macro_rules! tile {
            ($a_at:expr, $bp0:ident, $bp1:ident, $i:expr, $d0:ident, $d1:ident, $d2:ident, $d3:ident) => {
                if subs > $i {
                    let a = ldmatrix_x4($a_at + $i * TILE16);
                    $d0 = mma_m16n8k16_bf16(a, [$bp0[0], $bp0[1]], $d0);
                    $d1 = mma_m16n8k16_bf16(a, [$bp0[2], $bp0[3]], $d1);
                    $d2 = mma_m16n8k16_bf16(a, [$bp1[0], $bp1[1]], $d2);
                    $d3 = mma_m16n8k16_bf16(a, [$bp1[2], $bp1[3]], $d3);
                }
            };
        }

        if mode == 3 {
            // Zero-LUT SM121a register NVFP4 panel path (`stage64`):
            // - Uses native `cvt.rn.f16x2.{e4m3x2,e2m1x2}` + exact FP16 multiply (`<= 6` sig bits in `[2^-10, 2688]`)
            //   + FP32 `* factor` + `cvt.rn.bf16x2.f32`: bit-for-bit identical to `loop_widen::nvfp4_weight`,
            //   eliminating the 8 KB table copy and all `LDS.U16` LUT bank conflicts.
            // - Repurposes the freed 8 KB SMEM into a 3-stage `A` `cp.async` ring (`wait_one`, 2 A steps in flight).
            // - Marks streaming expert weight loads `L2::evict_first` so hot `A` input activations stay in L2.
            let (b_code0, b_scale0) = if tile_n * TC_PACKED_BN + warp * 16 < rows {
                let chunk0 = ((tile_n * 2 + (warp >> 2)) * (k >> 6)) * 4 + (warp & 3);
                ((chunk0 * 512 + lane * 16) as u32, (parts.scales + chunk0 * 64 + (lane >> 2) * 8) as u32)
            } else {
                (u32::MAX, 0)
            };
            let b_store_off = (warp * 1024 + lane * 16) as u32;
            let b_read_off = (warp_n * 2 * 1024 + lane * 16) as u32;
            if b_code0 == u32::MAX {
                let mut b_st = 0usize;
                while b_st < B_STAGES {
                    let b_base = smem + (REG_B_AT + b_st * B_BYTES) as u32 + b_store_off;
                    st_shared_zero16(b_base);
                    st_shared_zero16(b_base + 512);
                    b_st += 1;
                }
            }
            let evict = l2_evict_first();
            let fetch_b64 = |s: u32| -> ([u32; 4], [u32; 2]) {
                if b_code0 != u32::MAX {
                    (
                        ld_global_nc_v4_u32_hint(unit.add((b_code0 + s * 2048) as usize).cast(), evict),
                        ld_global_nc_v2_u32(unit.add((b_scale0 + s * 256) as usize).cast()),
                    )
                } else {
                    ([0u32; 4], [0u32; 2])
                }
            };
            let put_panel_b = |stage: u32, w0: u32, w1: u32, sc32: u32| {
                if b_code0 == u32::MAX { return; }
                let b_base = smem + REG_B_AT as u32 + stage * B_BYTES as u32 + b_store_off;
                let unpack16_reg = |w: u32, s2: u32| -> [u32; 4] {
                    let (r0, r1, r2, r3): (u32, u32, u32, u32);
                    core::arch::asm!(
                        "{{",
                        ".reg .b16 %hs0, %hs1, %ht, %hb, %h0, %h1;",
                        ".reg .b8  %b0, %b1, %b2, %b3;",
                        ".reg .b32 %sc, %st, %sb, %p0, %p1, %p2, %p3;",
                        ".reg .f32 %f0, %f1;",
                        "mov.b32 {{%hs0, %hs1}}, {s2};",
                        "cvt.rn.f16x2.e4m3x2 %sc, %hs0;",
                        "mov.b32 {{%ht, %hb}}, %sc;",
                        "mov.b32 %st, {{%ht, %ht}};",
                        "mov.b32 %sb, {{%hb, %hb}};",
                        "mov.b32 {{%b0, %b1, %b2, %b3}}, {w};",
                        "cvt.rn.f16x2.e2m1x2 %p0, %b0;",
                        "cvt.rn.f16x2.e2m1x2 %p1, %b1;",
                        "cvt.rn.f16x2.e2m1x2 %p2, %b2;",
                        "cvt.rn.f16x2.e2m1x2 %p3, %b3;",
                        "mul.rn.f16x2 %p0, %p0, %st;",
                        "mul.rn.f16x2 %p1, %p1, %sb;",
                        "mul.rn.f16x2 %p2, %p2, %st;",
                        "mul.rn.f16x2 %p3, %p3, %sb;",
                        "mov.b32 {{%h0, %h1}}, %p0;",
                        "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                        "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                        "cvt.rn.bf16x2.f32 {r0}, %f1, %f0;",
                        "mov.b32 {{%h0, %h1}}, %p2;",
                        "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                        "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                        "cvt.rn.bf16x2.f32 {r1}, %f1, %f0;",
                        "mov.b32 {{%h0, %h1}}, %p1;",
                        "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                        "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                        "cvt.rn.bf16x2.f32 {r2}, %f1, %f0;",
                        "mov.b32 {{%h0, %h1}}, %p3;",
                        "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                        "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                        "cvt.rn.bf16x2.f32 {r3}, %f1, %f0;",
                        "}}",
                        w = in(reg32) w,
                        s2 = in(reg32) s2,
                        fac = in(reg32) factor,
                        r0 = out(reg32) r0,
                        r1 = out(reg32) r1,
                        r2 = out(reg32) r2,
                        r3 = out(reg32) r3,
                        options(pure, nomem, nostack),
                    );
                    [r0, r1, r2, r3]
                };
                st_shared_v4(b_base, unpack16_reg(w0, sc32));
                st_shared_v4(b_base + 512, unpack16_reg(w1, sc32 >> 16));
            };
            macro_rules! k_sub_reg {
                ($a_at:expr, $b_at:expr) => {{
                    let (a_at, b_at) = ($a_at, $b_at);
                    let bp0 = crate::ptx::load4(b_at);
                    let bp1 = crate::ptx::load4(b_at + 1024);
                    tile!(a_at, bp0, bp1, 0, c00, c01, c02, c03);
                    tile!(a_at, bp0, bp1, 1, c10, c11, c12, c13);
                    tile!(a_at, bp0, bp1, 2, c20, c21, c22, c23);
                    tile!(a_at, bp0, bp1, 3, c30, c31, c32, c33);
                }};
            }

            // Prologue: A steps 0 and 1 in flight (2 commit groups), B step 0 decoded into stage 0 with zero wait.
            issue_a(0, 0);
            crate::ptx::commit();
            issue_a(1, 1);
            crate::ptx::commit();
            let (mut b_q, mut b_sc) = fetch_b64(0);
            put_panel_b(0, b_q[0], b_q[1], b_sc[0]);

            let s0_b = smem + REG_B_AT as u32 + b_read_off;
            let s1_b = smem + (REG_B_AT + B_BYTES) as u32 + b_read_off;
            let mut a_cur = 0u32;
            let mut a_wr = 2u32;
            let mut s = 0u32;
            while s < stages {
                let more = s + 1 < stages;
                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
                if more { issue_a(a_wr, 2 * s + 2); }
                crate::ptx::commit();
                let a_stage = smem + a_cur * A_BYTES as u32;
                if subs > 0 { k_sub_reg!(a_stage + a_off, s0_b); }
                put_panel_b(1, b_q[2], b_q[3], b_sc[1]);
                if more {
                    let nb = fetch_b64(s + 1);
                    b_q = nb.0;
                    b_sc = nb.1;
                }
                if subs > 0 { k_sub_reg!(a_stage + (a_off ^ 32), s0_b + 512); }
                a_wr = a_cur;
                a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };

                cp_async_wait_one();
                core::arch::nvptx::_syncthreads();
                if more { issue_a(a_wr, 2 * s + 3); }
                crate::ptx::commit();
                let a_stage = smem + a_cur * A_BYTES as u32;
                if subs > 0 { k_sub_reg!(a_stage + a_off, s1_b); }
                if more { put_panel_b(0, b_q[0], b_q[1], b_sc[0]); }
                if subs > 0 { k_sub_reg!(a_stage + (a_off ^ 32), s1_b + 512); }
                a_wr = a_cur;
                a_cur = if a_cur == 2 { 0 } else { a_cur + 1 };
                s += 1;
            }
        } else if mode == 2 {
            // In-place MMA-lane NVFP4 panel fast path (`stage64`):
            // - One 100%-coalesced 16B weight load + 8B scale load per 64 columns (`fetch_b64`).
            // - Unpacks directly into `[warp][k16][lane]` MMA B-operand register layout (`2x st_shared_v4`,
            //   zero swizzle, zero bank conflicts) and reads back via `ptx::load4` (`LDS.128`).
            let (b_code0, b_scale0) = if tile_n * TC_PACKED_BN + warp * 16 < rows {
                let chunk0 = ((tile_n * 2 + (warp >> 2)) * (k >> 6)) * 4 + (warp & 3);
                ((chunk0 * 512 + lane * 16) as u32, (parts.scales + chunk0 * 64 + (lane >> 2) * 8) as u32)
            } else {
                (u32::MAX, 0)
            };
            let b_store_off = (warp * 1024 + lane * 16) as u32;
            let b_read_off = (warp_n * 2 * 1024 + lane * 16) as u32;
            if b_code0 == u32::MAX {
                let mut b_st = 0usize;
                while b_st < B_STAGES {
                    let b_base = smem + (B_AT + b_st * B_BYTES) as u32 + b_store_off;
                    st_shared_zero16(b_base);
                    st_shared_zero16(b_base + 512);
                    b_st += 1;
                }
            }
            let fetch_b64 = |s: u32| -> ([u32; 4], [u32; 2]) {
                if b_code0 != u32::MAX {
                    (
                        ld_global_nc_v4_u32(unit.add((b_code0 + s * 2048) as usize).cast()),
                        ld_global_nc_v2_u32(unit.add((b_scale0 + s * 256) as usize).cast()),
                    )
                } else {
                    ([0u32; 4], [0u32; 2])
                }
            };
            let put_panel_b = |stage: u32, w0: u32, w1: u32, sc32: u32| {
                if b_code0 == u32::MAX { return; }
                let b_base = smem + B_AT as u32 + stage * B_BYTES as u32 + b_store_off;
                let table_at = smem + TABLE_AT as u32;
                let unpack16 = |w: u32, s2: u32| -> [u32; 4] {
                    let top = table_at + (s2 & 255) * 32;
                    let bot = table_at + ((s2 >> 8) & 255) * 32;
                    let u0 = crate::ptx::load_u16(top + (w & 15) * 2);
                    let u1 = crate::ptx::load_u16(top + ((w >> 4) & 15) * 2);
                    let u2 = crate::ptx::load_u16(bot + ((w >> 8) & 15) * 2);
                    let u3 = crate::ptx::load_u16(bot + ((w >> 12) & 15) * 2);
                    let u4 = crate::ptx::load_u16(top + ((w >> 16) & 15) * 2);
                    let u5 = crate::ptx::load_u16(top + ((w >> 20) & 15) * 2);
                    let u6 = crate::ptx::load_u16(bot + ((w >> 24) & 15) * 2);
                    let u7 = crate::ptx::load_u16(bot + ((w >> 28) & 15) * 2);
                    [u0 | (u1 << 16), u4 | (u5 << 16), u2 | (u3 << 16), u6 | (u7 << 16)]
                };
                st_shared_v4(b_base, unpack16(w0, sc32));
                st_shared_v4(b_base + 512, unpack16(w1, sc32 >> 16));
            };
            macro_rules! k_sub_panel {
                ($a_at:expr, $b_at:expr) => {{
                    let (a_at, b_at) = ($a_at, $b_at);
                    let bp0 = crate::ptx::load4(b_at);
                    let bp1 = crate::ptx::load4(b_at + 1024);
                    tile!(a_at, bp0, bp1, 0, c00, c01, c02, c03);
                    tile!(a_at, bp0, bp1, 1, c10, c11, c12, c13);
                    tile!(a_at, bp0, bp1, 2, c20, c21, c22, c23);
                    tile!(a_at, bp0, bp1, 3, c30, c31, c32, c33);
                }};
            }

            // Prologue: table and A step 0 landed, B step 0 stored; B step 1 already in `b_q[2..4]/b_sc[1]`.
            issue_a(0, 0);
            crate::ptx::commit();
            let (mut b_q, mut b_sc) = fetch_b64(0);
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            put_panel_b(0, b_q[0], b_q[1], b_sc[0]);

            let s0_b = smem + B_AT as u32 + b_read_off;
            let s1_b = smem + (B_AT + B_BYTES) as u32 + b_read_off;
            let mut s = 0u32;
            while s < stages {
                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
                issue_a(1, 2 * s + 1);
                crate::ptx::commit();
                if subs > 0 { k_sub_panel!(smem + a_off, s0_b); }
                put_panel_b(1, b_q[2], b_q[3], b_sc[1]);
                let more = s + 1 < stages;
                if more {
                    let nb = fetch_b64(s + 1);
                    b_q = nb.0;
                    b_sc = nb.1;
                }
                if subs > 0 { k_sub_panel!(smem + (a_off ^ 32), s0_b + 512); }

                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
                if more { issue_a(0, 2 * s + 2); }
                crate::ptx::commit();
                if subs > 0 { k_sub_panel!(smem + A_BYTES as u32 + a_off, s1_b); }
                if more { put_panel_b(0, b_q[0], b_q[1], b_sc[0]); }
                if subs > 0 { k_sub_panel!(smem + A_BYTES as u32 + (a_off ^ 32), s1_b + 512); }
                s += 1;
            }
        } else {
            // Shipped layout (`mode < 2`): `b_row` and `b_row + 64`, `swiz(b_row + 64, q) == b_swiz0 + 4096`.
            let b_row = tid >> 2;
            let b_quarter = (tid & 3) as u32;
            let b_place = |r: usize| -> (u32, u32) {
                if r >= rows { return (u32::MAX, 0); }
                let code = code_copy(r, 0, 0, k, wide) as u32;
                let scale = (parts.scales + scale_copy(r, 0, k, wide)) as u32;
                if wide { (code + b_quarter * 8, scale) } else { (code + b_quarter * 4, scale + (b_quarter >> 1)) }
            };
            let (b_code0, b_scale0) = b_place(tile_n * TC_PACKED_BN + b_row);
            let (b_code1, b_scale1) = b_place(tile_n * TC_PACKED_BN + b_row + 64);
            let b_swiz0 = swiz(b_row, b_quarter as usize);
            let mut b_st = 0usize;
            while b_st < B_STAGES {
                let b_base = smem + (B_AT + b_st * B_BYTES) as u32;
                if b_code0 == u32::MAX { st_shared_zero16(b_base + b_swiz0); }
                if b_code1 == u32::MAX { st_shared_zero16(b_base + b_swiz0 + 4096); }
                b_st += 1;
            }
            let b_off = swiz(warp_n * 32 + ((lane >> 4) * 8) + (lane & 7), (lane >> 3) & 1);
            let fetch_row = |code: u32, scale: u32, s: u32, h: u32| -> ([u32; 2], u32) {
                let (mut q, mut sc) = ([0u32; 2], 0u32);
                if code != u32::MAX {
                    if mode == 1 {
                        q = ld_global_nc_v2_u32(unit.add((code + s * 2048 + h * 32) as usize).cast());
                        sc = ld_global_nc_u8(unit.add((scale + (s >> 1) * 128 + (s & 1) * 2 + h) as usize));
                    } else {
                        q = [ld_global_nc_u32(unit.add((code + s * 1024 + h * 16) as usize).cast()), 0];
                        sc = ld_global_nc_u8(unit.add((scale + s * 128 + h * 2) as usize));
                    }
                }
                (q, sc)
            };
            let fetch = |s: u32, h: u32| -> (([u32; 2], u32), ([u32; 2], u32)) {
                (fetch_row(b_code0, b_scale0, s, h), fetch_row(b_code1, b_scale1, s, h))
            };
            let put_row = |at: u32, code: u32, (q, sc): ([u32; 2], u32)| {
                if code == u32::MAX { return; }
                let table_at = smem + TABLE_AT as u32;
                if mode == 1 {
                    let p = power(sc);
                    let unpack4 = |w: u32| -> (u32, u32) {
                        let v0 = crate::ptx::load(table_at + (w & 255) * 4);
                        let v1 = crate::ptx::load(table_at + ((w >> 8) & 255) * 4);
                        let v2 = crate::ptx::load(table_at + ((w >> 16) & 255) * 4);
                        let v3 = crate::ptx::load(table_at + (w >> 24) * 4);
                        (cvt_bf16x2(v0 * p, v1 * p), cvt_bf16x2(v2 * p, v3 * p))
                    };
                    let (p0, p1) = unpack4(q[0]);
                    let (p2, p3) = unpack4(q[1]);
                    st_shared_v4(at, [p0, p1, p2, p3]);
                } else {
                    let line = table_at + sc * 32;
                    let w = q[0];
                    let u0 = crate::ptx::load_u16(line + (w & 15) * 2);
                    let u1 = crate::ptx::load_u16(line + ((w >> 4) & 15) * 2);
                    let u2 = crate::ptx::load_u16(line + ((w >> 8) & 15) * 2);
                    let u3 = crate::ptx::load_u16(line + ((w >> 12) & 15) * 2);
                    let u4 = crate::ptx::load_u16(line + ((w >> 16) & 15) * 2);
                    let u5 = crate::ptx::load_u16(line + ((w >> 20) & 15) * 2);
                    let u6 = crate::ptx::load_u16(line + ((w >> 24) & 15) * 2);
                    let u7 = crate::ptx::load_u16(line + ((w >> 28) & 15) * 2);
                    st_shared_v4(at, [u0 | (u1 << 16), u2 | (u3 << 16), u4 | (u5 << 16), u6 | (u7 << 16)]);
                }
            };
            let put = |stage: u32, (r0, r1): (([u32; 2], u32), ([u32; 2], u32))| {
                let b_base = smem + B_AT as u32 + stage * B_BYTES as u32;
                put_row(b_base + b_swiz0, b_code0, r0);
                put_row(b_base + b_swiz0 + 4096, b_code1, r1);
            };
            macro_rules! k_sub {
                ($a_at:expr, $b_at:expr) => {{
                    let (a_at, b_at) = ($a_at, $b_at);
                    let bp0 = ldmatrix_x4(b_at);
                    let bp1 = ldmatrix_x4(b_at + TILE16);
                    tile!(a_at, bp0, bp1, 0, c00, c01, c02, c03);
                    tile!(a_at, bp0, bp1, 1, c10, c11, c12, c13);
                    tile!(a_at, bp0, bp1, 2, c20, c21, c22, c23);
                    tile!(a_at, bp0, bp1, 3, c30, c31, c32, c33);
                }};
            }

            // Prologue: the table and A step 0 landed, B step 0 stored, the packed words of step 1 in registers.
            issue_a(0, 0);
            crate::ptx::commit();
            let first = fetch(0, 0);
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            put(0, first);
            let mut ahead = fetch(0, 1);

            let mut s = 0u32;
            while s < stages {
                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
                issue_a(1, 2 * s + 1);
                crate::ptx::commit();
                if subs > 0 { k_sub!(smem + a_off, smem + B_AT as u32 + b_off); }
                put(1, ahead);
                let more = s + 1 < stages;
                if more { ahead = fetch(s + 1, 0); }
                if subs > 0 { k_sub!(smem + (a_off ^ 32), smem + B_AT as u32 + (b_off ^ 32)); }

                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
                if more { issue_a(0, 2 * s + 2); }
                crate::ptx::commit();
                if subs > 0 { k_sub!(smem + A_BYTES as u32 + a_off, smem + (B_AT + B_BYTES) as u32 + b_off); }
                if more {
                    put(0, ahead);
                    ahead = fetch(s + 1, 1);
                }
                if subs > 0 { k_sub!(smem + A_BYTES as u32 + (a_off ^ 32), smem + (B_AT + B_BYTES) as u32 + (b_off ^ 32)); }
                s += 1;
            }
        }

        // Epilogue: store 4x4 MMA output tiles (64x32 per warp), mapping `PAIRS` (gate rows first, then up rows).
        // The words read only here are read again now (the same command words), so they hold no register in the loop.
        let output = words.add(2).read_volatile() as *mut f32;
        let rows = words.add(3).read_volatile() as usize;
        let pairs = (words.add(4).read_volatile() as u32) & PAIRS != 0;
        let stride = words.add(6).read_volatile() as usize;
        let out_table = words.add(10).read_volatile() as *const u32;
        let out_count = words.add(11).read_volatile() as usize;
        let coefficients = words.add(15).read_volatile() as *const f32;
        let half = words.add(16).read_volatile() == HALF_OUT;
        let gated = !coefficients.is_null();
        let width = if gated { rows / 2 } else { rows };
        if ((out_table as usize | coefficients as usize) & 3) != 0 || stride < width || (gated && !pairs) || tile_n * TC_PACKED_BN >= rows
            || (half && !gated && (pairs || rows % 2 != 0 || stride % 2 != 0))
        {
            core::arch::nvptx::trap();
        }
        let (g, t) = (lane / 4, lane % 4);
        // Gated: `rn(gate_table[rn(gate)] * up_table[rn(up)])`, the words of `FAST_CAPPED_GATE` with its tables.
        let capped = |gate: f32, up: f32| -> f32 {
            let gi = (rn(gate).to_bits() >> 16) as usize;
            let ui = (rn(up).to_bits() >> 16) as usize;
            rn(*coefficients.add(gi) * *coefficients.add(CAPPED_GATE_TABLE_WORDS + ui))
        };
        let store_tile = |m0: usize, m1: usize, out_r0: usize, out_r1: usize, ni: usize, v: [f32; 4]| {
            let col = tile_n * TC_PACKED_BN + warp_n * 32 + ni * 8 + t * 2;
            if gated && half && stride % 2 == 0 && rows % 4 == 0 {
                // Coalesce adjacent gated BF16 outputs across even/odd lane `t` into single 32-bit global stores.
                // Every lane of the warp participates uniformly in `shfl_xor_u32` before bounds gating.
                let valid = col < rows;
                let h0 = if valid && m0 < m { cvt_bf16x2(capped(v[0], v[1]), 0.0) & 0xffff } else { 0 };
                let h1 = if valid && m1 < m { cvt_bf16x2(capped(v[2], v[3]), 0.0) & 0xffff } else { 0 };
                let p0 = h0 | (shfl_xor_u32(h0, 1) << 16);
                let p1 = h1 | (shfl_xor_u32(h1, 1) << 16);
                if valid && (t & 1) == 0 {
                    let words = output.cast::<u32>();
                    if m0 < m { *words.add((out_r0 * stride + col / 2) / 2) = p0; }
                    if m1 < m { *words.add((out_r1 * stride + col / 2) / 2) = p1; }
                }
                return;
            }
            if col >= rows { return; }
            if gated {
                // `col` is even and `rows` is even: stored rows `col`, `col + 1` are gate and up of output `col / 2`.
                if half {
                    // The BF16 value of the gated word, as the `Down` product rounds its input (`cvt.rn.bf16x2.f32`).
                    let values = output.cast::<u16>();
                    if m0 < m { *values.add(out_r0 * stride + col / 2) = (cvt_bf16x2(capped(v[0], v[1]), 0.0) & 0xffff) as u16; }
                    if m1 < m { *values.add(out_r1 * stride + col / 2) = (cvt_bf16x2(capped(v[2], v[3]), 0.0) & 0xffff) as u16; }
                    return;
                }
                if m0 < m { *output.add(out_r0 * stride + col / 2) = capped(v[0], v[1]); }
                if m1 < m { *output.add(out_r1 * stride + col / 2) = capped(v[2], v[3]); }
                return;
            }
            if half {
                // `col` is even and `rows` is even, so `col + 1 < rows`: one word holds both BF16 values.
                let pack = |a: f32, b: f32| (rn(a).to_bits() >> 16) | (rn(b).to_bits() & 0xffff_0000);
                let words = output.cast::<u32>();
                if m0 < m { *words.add((out_r0 * stride + col) / 2) = pack(v[0], v[1]); }
                if m1 < m { *words.add((out_r1 * stride + col) / 2) = pack(v[2], v[3]); }
                return;
            }
            let c0 = output_row(col, rows, pairs);
            let has_c1 = col + 1 < rows;
            let c1 = if has_c1 { output_row(col + 1, rows, pairs) } else { 0 };
            if m0 < m {
                *output.add(out_r0 * stride + c0) = v[0];
                if has_c1 { *output.add(out_r0 * stride + c1) = v[1]; }
            }
            if m1 < m {
                *output.add(out_r1 * stride + c0) = v[2];
                if has_c1 { *output.add(out_r1 * stride + c1) = v[3]; }
            }
        };

        macro_rules! store_rows {
            ($i:expr, $d0:ident, $d1:ident, $d2:ident, $d3:ident) => {
                if subs > $i {
                    let m0 = warp_m * 64 + $i * 16 + g;
                    let m1 = m0 + 8;
                    let out_r0 = if m0 < m { place(out_table, out_count, base + m0) } else { 0 };
                    let out_r1 = if m1 < m { place(out_table, out_count, base + m1) } else { 0 };
                    store_tile(m0, m1, out_r0, out_r1, 0, $d0);
                    store_tile(m0, m1, out_r0, out_r1, 1, $d1);
                    store_tile(m0, m1, out_r0, out_r1, 2, $d2);
                    store_tile(m0, m1, out_r0, out_r1, 3, $d3);
                }
            };
        }
        store_rows!(0, c00, c01, c02, c03);
        store_rows!(1, c10, c11, c12, c13);
        store_rows!(2, c20, c21, c22, c23);
        store_rows!(3, c30, c31, c32, c33);
    }
}
