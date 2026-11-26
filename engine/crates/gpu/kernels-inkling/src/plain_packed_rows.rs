//! The packed rows tile (`packed_product`, THE PACKED ROWS TILE): the entry `resident_launch_plain_packed_rows` of the
//! Inkling image (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_PACKED_ROWS`.
//!
//! One launch gives the products of the expert runs of a visit: run `r` is ONE matrix in its stored packed form (its
//! unit: codes, block scales and weight table, `packed_product::layout`) with its launch rows. It is the grouped form
//! of the rows tile of the staged BF16 product (`plain_staged_rows.rs`) with one change: a thread does not read two
//! words of a widened BF16 matrix, it widens four weights of the unit to the BF16 value that the widened form holds
//! there (`plain_packed.rs`: a table read in the 4-bit form; the stored value times its power of two, rounded to BF16,
//! in the 8-bit form). The chains are the chains of the rows tile, in the same order with the same operations, so every
//! output word is the word of the rows tile on the widened matrix: one ascending FP32 chain over the columns, from 0.0.
//! With `PAIRS` stored row `s` is output row `packed_product::output_row(s)` (gate rows first).
//!
//! So the device needs no widened copy of an expert stack: the packed stack that the one-row products read
//! (`plain_packed.rs`) is the only form of the experts.
//!
//! The host checks every extent and every alignment before it records a launch (`packed_product::check_rows_grouped`).
//!
//! A block takes 64 weight rows of one run and 64 of its launch rows; each of its 256 threads owns 4 weight rows x 4
//! launch rows (16 independent chains). Columns pass through shared memory in chunks of 32, two stages deep, as in
//! the BF16 rows tile. The weight table of the run's unit is copied to shared memory once, before the first stage.
//!
//! Launch: grid = the blocks of the block table, 256 threads, no dynamic shared memory.
use crate::packed_product::{COLS, PAIRS, ROWS_TABLE_BYTES, WIDE, layout, output_row, power, round_bf16, rows_admitted, table_copies};
use core::arch::nvptx;

/// Weight rows and launch rows of a block.
const BR: usize = crate::staged_product::ROWS_TILE;
const BT: usize = crate::staged_product::ROWS_TILE;
/// Columns of a chunk, and shared words of one row of a chunk (the columns and a pad of 4 words).
const KC: usize = 32;
const SK: usize = 36;
/// Shared words of one stage, the shared byte of the weight table, and the shared bytes of a block.
const STAGE: usize = (BR + BT) * SK;
const TABLE_AT: usize = 2 * STAGE * 4;
const BLOCK_BYTES: usize = TABLE_AT + ROWS_TABLE_BYTES;
const _: () = assert!(crate::staged_product::ROWS_THREADS == BR / 4 * (BT / 4) && crate::staged_product::STAGE % KC == 0);
const _: () = assert!(TABLE_AT == 36_864 && BLOCK_BYTES == 45_056 && TABLE_AT % 16 == 0);
// A thread copies the launch rows `tid / 8` and `tid / 8 + 32` of its tile (the items after the weight items).
const _: () = assert!(BR * 8 % crate::staged_product::ROWS_THREADS == 0 && BT * 8 == 2 * crate::staged_product::ROWS_THREADS);

/// The replay of one recorded command of this kind: `table[index]` holds fifteen words, those of the grouped form of
/// the BF16 rows tile (`plain_staged_rows.rs`) with word 0 unused and word 4 the column word
/// (`packed_product::cols_word`: the columns, `WIDE`, `PAIRS`): the input (input count x cols FP32 values), the output
/// ((output count - 1) x stride + rows words), rows, the column word, the launch rows and the stride, the input table
/// (null, or launch rows u32 row numbers) and the input count, the output table and the output count, the run table
/// (16 bytes a run: the unit address, its first launch row, its count) and the runs, the block table (8 bytes a block:
/// its run, its launch tile | weight tile << 16) and the blocks.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_packed_rows`): `table` and `index` are the
/// arguments the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::project_packed_rows_grouped_into, after
    // packed_product::check_rows_grouped passed for its words: every run's unit is exactly layout(rows, cols, wide)
    // bytes, 16-byte aligned, and meets no output; the input is exactly input count x cols words and 16-byte aligned;
    // the output is exactly (output count - 1) x stride + rows words that meet no operand, with stride at least rows;
    // cols is whole stages of the form (64 columns, 128 in the 8-bit form) up to 4,608; with PAIRS rows is even; first +
    // count of every run is at most the launch rows; each count is 1 to 1,024. All operands live through completion.
    // The four tables are host words that the host checked (each row table exactly launch-rows words, every index below
    // its count, the output rows distinct; the run and block tables built by the host from the checked runs), staged
    // with this command in the recorder's own stage area (record_resident_tabled) and written by nothing else before
    // this replay. Here the block is compared with the blocks and the run with the runs before either table is read
    // at it, the alignment of each table is checked, first + count is compared with the launch rows, every index word
    // is compared with its count before any use, and `rows_admitted` (the shape, the form, the block size, the
    // alignment of the unit and the input) is asked before the first read; any failure traps.
    // For each valid row, code_copy locates its aligned 64-column tile. A four-column group
    // reads two or four bytes wholly inside it. scale_copy locates four consecutive scale bytes;
    // the offset is 0..4 in either form. rows_admitted requires complete stages for both reads.
    // The table part is copied in 16-byte copies below table_bytes(wide), from its
    // 16-byte aligned start inside the unit. Shared reads of the table: a 2-byte pattern at scale x 32 + code x 2 (both
    // below 256 and 16) or a word at code x 4 (code below 256), all inside its NARROW_TABLE bytes. Input reads: one
    // 16-byte copy (4 words) of an input row below the input count, from a column below cols that is a multiple of 4.
    // Each (launch row, weight row) word is stored at output row x stride + output_row(weight row), only for a launch
    // row below the run's count and a weight row below rows; output_row is a bijection of 0..rows, so with stride at
    // least rows each word has exactly one writer and the largest offset is (output count - 1) x stride + rows - 1.
    // Every shared read and write stays in the block's own static array of BLOCK_BYTES. All 256 threads reach every
    // barrier; the loop bounds depend only on the shape.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_PACKED_ROWS { nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (input, output) = (*words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, word, launch_rows) = (*words.add(3) as usize, *words.add(4) as u32, *words.add(5) as usize);
        let (k, wide, pairs) = ((word & COLS) as usize, word & WIDE != 0, word & PAIRS != 0);
        let stride = *words.add(6) as usize;
        if stride < rows { nvptx::trap(); }
        let (in_table, in_count) = (*words.add(7) as *const u32, *words.add(8) as usize);
        let (out_table, out_count) = (*words.add(9) as *const u32, *words.add(10) as usize);
        if (in_table as usize | out_table as usize) & 3 != 0 { nvptx::trap(); }
        // The operand row of launch row `t`: the table word, or `t` without a table; never at or above `count`.
        let place = |table: *const u32, count: usize, t: usize| -> usize {
            let row = if table.is_null() { t } else { table.add(t).read() as usize };
            if row >= count { nvptx::trap(); }
            row
        };
        let (runs_table, runs) = (*words.add(11) as *const u32, *words.add(12) as usize);
        let (blocks_table, blocks) = (*words.add(13) as *const u32, *words.add(14) as usize);
        let block = nvptx::_block_idx_x() as usize;
        if runs_table.is_null() || (runs_table as usize & 15) != 0 || (blocks_table as usize & 7) != 0 || block >= blocks { nvptx::trap(); }
        let entry = blocks_table.add(2 * block);
        let (run, tiles) = (entry.read() as usize, entry.add(1).read() as usize);
        if run >= runs { nvptx::trap(); }
        // 16 bytes a run: the unit address (two words, low first), its first launch row, its count.
        let at = runs_table.add(4 * run);
        let unit = (at.read() as usize | (at.add(1).read() as usize) << 32) as *const u8;
        let (base, m) = (at.add(2).read() as usize, at.add(3).read() as usize);
        if base + m > launch_rows { nvptx::trap(); }
        let (tile_t, tile_r) = (tiles & 0xffff, tiles >> 16);
        let threads = nvptx::_block_dim_x() as usize;
        if !rows_admitted(rows, k, wide, pairs, m, unit as usize, input as usize, threads) { nvptx::trap(); }
        if tile_r * BR >= rows { return; }
        let tid = nvptx::_thread_idx_x() as usize;
        let (ty, tx) = (tid / 16, tid % 16);
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        let parts = layout(rows, k, wide);
        let (codes, scales) = (unit, unit.add(parts.scales));
        let t32 = unit.add(parts.table).cast::<u32>();
        let panel = !wide && rows % 64 == 0 && k % 64 == 0
            && t32.add(0x7f * 8).read() == crate::packed_product::NVFP4_PANEL_MAGIC_LO
            && t32.add(0x7f * 8 + 1).read() == crate::packed_product::NVFP4_PANEL_MAGIC_HI;
        let reg_nvfp4 = !wide
            && t32.add(0xff * 8).read() == crate::packed_product::NVFP4_FACTOR_MAGIC_LO
            && t32.add(0xff * 8 + 1).read() == crate::packed_product::NVFP4_FACTOR_MAGIC_HI;
        let factor = if reg_nvfp4 { f32::from_bits(t32.add(0xff * 8 + 2).read()) } else { 0.0f32 };
        // Copy weight table into shared memory only when not using bit-exact SM121a register NVFP4 decode.
        if !reg_nvfp4 {
            let mut t = tid;
            while t < table_copies(wide) { crate::ptx::copy16(smem + (TABLE_AT + t * 16) as u32, unit.add(parts.table + t * 16).cast()); t += 256; }
            crate::ptx::commit();
            crate::ptx::wait();
            core::arch::nvptx::_syncthreads();
        }
        let table_at = smem + TABLE_AT as u32;
        let mode: u32 = if panel { 2 } else if wide { 1 } else { 0 };
        let first_row = tile_t * BT + tid / 8;
        let in_low = if first_row < m { input.add(place(in_table, in_count, base + first_row) * k) } else { input };
        let in_high = if first_row + 32 < m { input.add(place(in_table, in_count, base + first_row + 32) * k) } else { input };
        let chunks = (k / KC) as u32;
        let rows_u32 = rows as u32;
        let k_u32 = k as u32;
        let m_u32 = m as u32;
        let tile_r_base = (tile_r * BR) as u32;
        let tile_t_base = (tile_t * BT) as u32;
        let tid_u32 = tid as u32;
        // Copy chunk `c` of this block's 64 weight rows (widened from the unit) and 64 input rows into stage `c & 1`.
        let issue = |c: u32| {
            let stage = smem + (c & 1) * (STAGE * 4) as u32;
            let first = c * KC as u32;
            let mut item = tid_u32;
            while item < ((BR + BT) * 8) as u32 {
                let which = item >> 9;
                let row = (item >> 3) & 63;
                let quad = item & 7;
                let column = first + (quad << 2);
                let dst = stage + ((which * BR as u32 + row) * (SK * 4) as u32 + (quad << 4));
                if which == 0 {
                    let r = tile_r_base + row;
                    if r < rows_u32 {
                        let mut e = 0u32;
                        if mode == 1 {
                            let (code, scale) = crate::packed_product::rows_quad(r as usize, column as usize, k_u32 as usize, true);
                            let four = codes.add(code).cast::<u32>().read();
                            let power = power(*scales.add(scale) as u32);
                            while e < 4 {
                                let value = crate::ptx::load(table_at + ((four >> (8 * e)) & 255) * 4);
                                // One rounded product (no add follows it, so it cannot fuse), then the BF16
                                // rounding: the weight of `plain_packed.rs` (`wide`).
                                crate::ptx::store(dst + 4 * e, round_bf16(value * power));
                                e += 1;
                            }
                        } else {
                            let (four, sc) = if mode == 2 {
                                let ch = (((r >> 6) * (k_u32 >> 6) + (column >> 6)) << 2) | ((r >> 4) & 3);
                                let within = (column >> 4) & 3;
                                let pair = r & 7;
                                let bot = (r >> 3) & 1;
                                let base = (ch << 9) + (((pair << 2) | ((column >> 1) & 3)) << 4) + (within << 2) + (((column >> 3) & 1) << 1) + bot;
                                let b0 = *codes.add(base as usize) as u32;
                                let b1 = *codes.add(base as usize + 16) as u32;
                                let s = *scales.add(((ch << 6) + (pair << 3) + (within << 1) + bot) as usize) as u32;
                                (b0 | (b1 << 8), s)
                            } else {
                                // The low half of a byte is the even column.
                                let (code, scale) = crate::packed_product::rows_quad(r as usize, column as usize, k_u32 as usize, false);
                                (codes.add(code).cast::<u16>().read() as u32, *scales.add(scale) as u32)
                            };
                            if reg_nvfp4 {
                                let (w0, w1): (u32, u32);
                                core::arch::asm!(
                                    "{{",
                                    ".reg .b16 %hs, %ht, %hb, %h0, %h1;",
                                    ".reg .b8  %b0, %b1, %b2, %b3;",
                                    ".reg .b32 %sc, %st, %p0, %p1;",
                                    ".reg .f32 %f0, %f1;",
                                    "mov.b32 {{%b0, %b1, %b2, %b3}}, {four};",
                                    "cvt.u16.u32 %hs, {sc};",
                                    "cvt.rn.f16x2.e4m3x2 %sc, %hs;",
                                    "mov.b32 {{%ht, %hb}}, %sc;",
                                    "mov.b32 %st, {{%ht, %ht}};",
                                    "cvt.rn.f16x2.e2m1x2 %p0, %b0;",
                                    "cvt.rn.f16x2.e2m1x2 %p1, %b1;",
                                    "mul.rn.f16x2 %p0, %p0, %st;",
                                    "mul.rn.f16x2 %p1, %p1, %st;",
                                    "mov.b32 {{%h0, %h1}}, %p0;",
                                    "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                    "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                    "cvt.rn.bf16x2.f32 {w0}, %f1, %f0;",
                                    "mov.b32 {{%h0, %h1}}, %p1;",
                                    "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                    "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                    "cvt.rn.bf16x2.f32 {w1}, %f1, %f0;",
                                    "}}",
                                    four = in(reg32) four,
                                    sc = in(reg32) sc,
                                    fac = in(reg32) factor,
                                    w0 = out(reg32) w0,
                                    w1 = out(reg32) w1,
                                    options(pure, nomem, nostack)
                                );
                                crate::ptx::store(dst, f32::from_bits(w0 << 16));
                                crate::ptx::store(dst + 4, f32::from_bits(w0 & 0xffff_0000));
                                crate::ptx::store(dst + 8, f32::from_bits(w1 << 16));
                                crate::ptx::store(dst + 12, f32::from_bits(w1 & 0xffff_0000));
                            } else {
                                let line = table_at + sc * 32;
                                while e < 4 {
                                    crate::ptx::store(dst + 4 * e, f32::from_bits(crate::ptx::load_u16(line + ((four >> (4 * e)) & 15) * 2) << 16));
                                    e += 1;
                                }
                            }
                        }
                    }
                } else {
                    let t = tile_t_base + row;
                    if t < m_u32 {
                        let src = if item < (BR * 8 + 256) as u32 { in_low } else { in_high };
                        crate::ptx::copy16(dst, src.add(column as usize).cast());
                    }
                }
                item += 256;
            }
        };
        let mut acc = [[0.0f32; 4]; 4];
        issue(0);
        crate::ptx::commit();
        let mut c = 0u32;
        while c < chunks {
            if c + 1 < chunks { issue(c + 1); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
            core::arch::nvptx::_syncthreads();
            let stage = smem + (c & 1) * (STAGE * 4) as u32;
            let mut q = 0u32;
            while q < (KC / 4) as u32 {
                let mut w = [[0u32; 4]; 4];
                let mut x = [[0u32; 4]; 4];
                for i in 0..4u32 { w[i as usize] = crate::ptx::load4(stage + (((ty as u32 * 4 + i) * SK as u32 + q * 4) * 4)); }
                for j in 0..4u32 { x[j as usize] = crate::ptx::load4(stage + (((BR as u32 + tx as u32 + 16 * j) * SK as u32 + q * 4) * 4)); }
                // The arithmetic of the BF16 rows tile, term for term (plain_staged_rows.rs).
                for e in 0..4 {
                    for i in 0..4 { for j in 0..4 { acc[i][j] += f32::from_bits(x[j][e]) * f32::from_bits(w[i][e]); } }
                }
                q += 1;
            }
            core::arch::nvptx::_syncthreads();
            c += 1;
        }
        let mut target = [0usize; 4];
        for (j, slot) in target.iter_mut().enumerate() {
            let token = tile_t * BT + tx + 16 * j;
            if token < m { *slot = place(out_table, out_count, base + token); }
        }
        for i in 0..4 {
            let row = tile_r * BR + ty * 4 + i;
            if row < rows {
                let at = output_row(row, rows, pairs);
                for j in 0..4 {
                    let token = tile_t * BT + tx + 16 * j;
                    if token < m { *output.add(target[j] * stride + at) = acc[i][j]; }
                }
            }
        }
    }
}
