//! The packed row product (`packed_product`): the kernel body.
//!
//! A direct product reads one matrix unit in its stored packed form and gives the words of `project_bf16_ordered` on
//! the matrix widened to BF16. Each output is one chain over the columns in rising order from +0.0, each product
//! rounded by its own `mul.rn.f32` before its `add.rn.f32`, no fused multiply-add. The weight of a place is the BF16
//! value the widened form holds there: a table read in the 4-bit form; the stored value times its power of two,
//! rounded to BF16, in the 8-bit form.
//!
//! The body is the function `tile`, with three helpers (`issue`, `narrow`, `wide`). Each of the four is an `unsafe fn`
//! with its SAFETY note: four unsafe sites, in this file, in the visible form (Ruling R6h; the macro form of Ruling
//! R6f is gone). `run_packed` of this image (`lib.rs`) calls `tile` with the five words of a recorded launch. The host
//! checks every extent and every alignment before it records a launch (`packed_product::check`); the note of `tile`
//! says which check covers which access.
//!
//! A routed tile uses one MMA warp per 16 stored rows and shares a tile of up to eight input rows.
//! Each lane unpacks its BF16 weight pairs from staged codes and scales. Every input slot takes K in the
//! same order, from zero, even when a different number of picks share the expert.
//!
//! For a direct product, a block of `packed_product::THREADS` threads runs row blocks of 32 stored rows: block `bx` of a grid `gx` wide takes
//! row blocks `bx`, `bx + gx`, ... in turn, as one stream of stages, and copies the weight table once
//! (`packed_product::spread` sets `gx`). Warp 0 runs the 32 chains: a lane adds the
//! rounded products of its row from shared memory, in column order. The four other warps are makers: they copy the
//! codes, the scale bytes and the inputs of a stage of 64 columns into shared memory (`cp.async`, `NS` stages deep),
//! and make the rounded products of the stage one stage before the chain reads them. Two product buffers take turns,
//! and one barrier a stage keeps a buffer from being written while a lane still reads it.
use crate::packed_product::{BLOCK_ROWS, COLS, NARROW_TABLE, PAIRS, PICKED_THREADS, STAGE, THREADS, WIDE, admitted, code_copy, code_stage, input_copy, layout, output_row, power, round_bf16, scale_copy, stages, table_copies};
use crate::ptx::{commit, copy4, copy16, load, load2, load4, load_u8, load_u16};
use core::arch::nvptx::_syncthreads;
use core::arch::{asm, nvptx};
use crate::stream_pairs::gate_tanh;
use crate::tc_ptx::{cvt_bf16x2, ld_global_nc_v4, load_u32, mma_m16n8k16_bf16, store_u32};

const WARP_ROWS: usize = 16;
const PICKED_WARPS: usize = PICKED_THREADS / 32;
const PICKED_ROWS: usize = PICKED_WARPS * WARP_ROWS;
const fn picked_scale_at(wide: bool) -> usize { PICKED_ROWS * code_stage(wide) }
const fn picked_input_at(wide: bool) -> usize { picked_scale_at(wide) + PICKED_ROWS * 4 }
const fn picked_stage_bytes(wide: bool) -> usize { picked_input_at(wide) + 8 * CH * 2 }
const fn picked_table_at(wide: bool) -> usize { 2 * picked_stage_bytes(wide) }
const PICKED_NARROW_BYTES: usize = picked_table_at(false) + NARROW_TABLE;
const PICKED_WIDE_BYTES: usize = picked_table_at(true) + crate::packed_product::WIDE_TABLE;
const PANEL_X_BYTES: usize = 10_240;
const PANEL_META_AT: u32 = PANEL_X_BYTES as u32;
const PANEL_TABLE_AT: usize = PANEL_X_BYTES + 64;
const PANEL_BLOCK_BYTES: usize = PANEL_TABLE_AT + NARROW_TABLE;
const PANEL_NOLUT_X_BYTES: usize = PANEL_BLOCK_BYTES - 64;
const PANEL_NOLUT_META_AT: u32 = PANEL_NOLUT_X_BYTES as u32;
pub(crate) const PICKED_BLOCK_BYTES: usize = if PANEL_BLOCK_BYTES > PICKED_NARROW_BYTES && PANEL_BLOCK_BYTES > PICKED_WIDE_BYTES {
    PANEL_BLOCK_BYTES
} else if PICKED_NARROW_BYTES > PICKED_WIDE_BYTES {
    PICKED_NARROW_BYTES
} else {
    PICKED_WIDE_BYTES
};
const _: () = assert!(PICKED_BLOCK_BYTES == 18_496 && PANEL_NOLUT_X_BYTES == 18_432 && PICKED_BLOCK_BYTES % 16 == 0 && crate::packed_product::PANEL_INPUT_COLS * 2 <= picked_table_at(false));

#[inline(always)]
fn picked_input_offset(row: u32, column: u32) -> u32 {
    (row * CH as u32 + (column ^ (row * 8))) * 2
}

// Split rows that would otherwise share one bank. The XOR keeps each
// 16-byte copy intact and uses the same address for the operand load.
#[inline(always)]
fn picked_code_offset(row: u32, column: u32, width: u32) -> u32 {
    let base = row * width;
    base + (column ^ ((base / 128 * 16) & (width - 16)))
}

/// Columns of a stage, and columns of the unit of work of one maker thread.
pub(crate) const CH: usize = STAGE;
pub(crate) const UNIT: usize = 16;
/// Threads that make products.
pub(crate) const MAKERS: usize = THREADS - 32;
/// Stages of copies in flight.
pub(crate) const NS: usize = 4;
/// Stages a row prefetch runs ahead of the stage in work.
pub(crate) const AHEAD: usize = 8;
/// Shared bytes of the products of one row in a stage, with a 16-byte pad so that eight neighbouring rows read
/// 16 bytes each from distinct banks.
pub(crate) const P_ROW: usize = CH * 4 + 16;
pub(crate) const P_BYTES: usize = BLOCK_ROWS * P_ROW;
/// A stage holds one copy of all 32 code rows, their scale words and the input.
const fn code_pitch(wide: bool) -> usize { if wide { CH + 16 } else { CH / 2 + 16 } }
const fn scale_at(wide: bool) -> usize { BLOCK_ROWS * code_pitch(wide) }
const fn x_at(wide: bool) -> usize { scale_at(wide) + BLOCK_ROWS * 4 }
const fn stage_bytes(wide: bool) -> usize { x_at(wide) + CH * 4 }
const STAGES_AT: usize = 2 * P_BYTES;
const fn table_at(wide: bool) -> usize { STAGES_AT + NS * stage_bytes(wide) }
pub(crate) const BLOCK_BYTES: usize = table_at(false) + NARROW_TABLE;
const _: () = assert!(BLOCK_ROWS == 32 && MAKERS == 128 && CH == 64 &&
    table_at(true) + crate::packed_product::WIDE_TABLE <= BLOCK_BYTES && BLOCK_BYTES % 16 == 0);

/// The rounded product of column `e` of the four columns in `word`: a 4-bit code picks its BF16 weight in the table
/// line of its scale byte. `x` holds the bits of the input word.
#[inline(always)]
// SAFETY: `line` is the shared address of one line of the weight table of this block (32 bytes inside
// `table_at(wide)..BLOCK_BYTES`, copied there before the first product). The read is 2 bytes at an offset under 32.
unsafe fn narrow(line: u32, word: u32, e: u32, x: u32) -> f32 {
    let weight = load_u16(line + ((word >> (4 * e)) & 15) * 2) << 16;
    let product: f32;
    asm!("mul.rn.f32 {p}, {a}, {b};", p = out(reg32) product, a = in(reg32) f32::from_bits(weight), b = in(reg32) f32::from_bits(x), options(nomem, nostack));
    product
}

/// The rounded product of column `e` of the four columns in `word`: an 8-bit value times its power of two, rounded
/// to BF16, times the input word with the bits `x`.
#[inline(always)]
// SAFETY: `values` is the shared address of the weight table of this block (256 words inside
// `table_at(wide)..BLOCK_BYTES`, copied there before the first product). The read is one word at an index under 256.
unsafe fn wide(values: u32, power: f32, word: u32, e: u32, x: u32) -> f32 {
    let value = load(values + ((word >> (8 * e)) & 255) * 4);
    let (scaled, product): (f32, f32);
    asm!("mul.rn.f32 {p}, {a}, {b};", p = out(reg32) scaled, a = in(reg32) value, b = in(reg32) power, options(nomem, nostack));
    asm!("mul.rn.f32 {p}, {a}, {b};", p = out(reg32) product, a = in(reg32) round_bf16(scaled), b = in(reg32) f32::from_bits(x), options(nomem, nostack));
    product
}

/// Pick indices in original route order, up to eight in one word. The ordinal
/// distinguishes the blocks of each participating pick in their joint grid.
#[derive(Clone, Copy)]
pub(crate) struct SharedPicks {pub(crate) indices:u64,pub(crate) count:usize,pub(crate) stride:usize,pub(crate) repeat:usize,pub(crate) ordinal:usize,pub(crate) one_row:bool,pub(crate) panels:*const u8}
impl SharedPicks {
    #[inline(always)]
    fn index(self,slot:usize)->usize {((self.indices>>(8*slot))&255) as usize}
}
#[derive(Clone, Copy)]
struct Stored { codes: *const u8, scales: *const u8, input: *const f32, rows: usize, cols: usize, wide: bool, panel: bool }

/// The copies of stage `s` of the row block from row `first` into slot `slot` (under `NS`) of shared memory: the
/// codes and the scale bytes of its 32 rows, and the 64 inputs. Maker `j` takes every `MAKERS`-th copy. A row at or
/// above `rows` is not copied.
#[inline(always)]
// SAFETY: the caller is `tile`, after its words passed `admitted`, with `s` under the stages of the matrix
// (`stages(cols)`) and `slot` under `NS`. Every source offset comes from `packed_product::{code_copy, scale_copy, input_copy}`, and the
// host test `every_copy_of_the_body_is_inside_its_part_and_the_last_one_ends_at_its_end` holds them: a code copy
// reads 16 bytes that end at or before `tiled_rows(rows) * code_stride` of `codes`; a scale copy reads 4 bytes that
// end at or before `tiled_rows(rows) * scale_stride` of `scales` (both parts are stored in tiles of 32 rows; in the
// 8-bit form a row has an even count of stages, so the pair of stages is one copy). When `m.panel` is true, `!m.wide`
// and `rows % 64 == 0`, so every panel chunk `(group64 * stages + s) * 4 + p` lies inside the unit's code and scale parts.
// An input copy ends within
// `cols * 4` bytes of the selected input row; checked pick indices select the row, and zero stride selects the base row.
// The host checks the full input extent. Every source and target is on the
// boundary of its copy size. Targets: slot `slot` of the block's own shared memory, inside `STAGES_AT..table_at(m.wide)`.
unsafe fn issue(slot: usize, s: usize, smem: u32, j: usize, first: usize, m: Stored) {
    let slot = smem + (STAGES_AT + slot * stage_bytes(m.wide)) as u32;
    if m.panel {
        if j < 32 {
            let r = first + j;
            if r < m.rows {
                let panel_idx = r >> 4;
                let chunk = ((panel_idx / 4 * stages(m.cols) + s) * 4) + (panel_idx & 3);
                let pair = r & 7;
                let bot = (r >> 3) & 1;
                let mut sc_word = 0u32;
                let sc_base = chunk * 64 + pair * 8 + bot;
                let c_base = chunk * 512 + pair * 64 + bot;
                let dst_row = slot + (j * code_pitch(false)) as u32;
                let mut within = 0usize;
                while within < 4 {
                    sc_word |= (*m.scales.add(sc_base + within * 2) as u32) << (8 * within);
                    let w_off = c_base + within * 4;
                    let lo = (*m.codes.add(w_off) as u32)
                        | ((*m.codes.add(w_off + 16) as u32) << 8)
                        | ((*m.codes.add(w_off + 32) as u32) << 16)
                        | ((*m.codes.add(w_off + 48) as u32) << 24);
                    let hi = (*m.codes.add(w_off + 2) as u32)
                        | ((*m.codes.add(w_off + 18) as u32) << 8)
                        | ((*m.codes.add(w_off + 34) as u32) << 16)
                        | ((*m.codes.add(w_off + 50) as u32) << 24);
                    store_u32(dst_row + (within * 8) as u32, lo);
                    store_u32(dst_row + (within * 8 + 4) as u32, hi);
                    within += 1;
                }
                store_u32(slot + (scale_at(false) + j * 4) as u32, sc_word);
            }
        } else if j < 32 + CH / 4 {
            let i = j - 32;
            copy16(slot + (x_at(false) + i * 16) as u32, m.input.add(input_copy(s, i)).cast());
        }
        return;
    }
    let parts = code_stage(m.wide) / 16;
    let mut t = j;
    while t < 32 * parts + 32 + CH/4 {
        if t < 32 * parts {
            let (row, part) = (t & 31, t >> 5);
            if first + row < m.rows { copy16(slot + (row * code_pitch(m.wide) + part * 16) as u32, m.codes.add(code_copy(first + row, s, part, m.cols, m.wide)).cast()); }
        } else if t < 32 * parts + 32 {
            let row = t - 32 * parts;
            if first + row < m.rows { copy4(slot + (scale_at(m.wide) + row * 4) as u32, m.scales.add(scale_copy(first + row, s, m.cols, m.wide)).cast()); }
        } else {
            let i = t - 32 * parts - 32;
            copy16(slot+(x_at(m.wide)+i*16) as u32,m.input.add(input_copy(s,i)).cast());
        }
        t += MAKERS;
    }
}

/// The body of one thread. `weights` is the unit of the matrix, `word` the column word
/// (`packed_product::cols_word`); `output` gets `rows` words, gate rows first.
#[inline(always)]
// SAFETY: the caller passes the five words of a launch that the host check `packed_product::check` passed, in a
// launch of `packed_product::THREADS` threads a block and any grid width (`packed_product::spread`). That check gives: `weights` is the start of
// `layout(rows, cols, wide).bytes` readable bytes on a 16-byte boundary; `input` is `cols` readable words on a
// 16-byte boundary; `output` is `rows` writable words on a word boundary that overlap neither operand.
// Global reads: the copies of the weight table (16 bytes each, under `table_bytes(wide)` from the table part of the
// unit) and the copies of `issue` (see its note); a prefetch names a place inside the unit and reads nothing.
// Global write: one word at `output_row(row, rows, pairs)`, which is under `rows`, by the one lane that owns `row`,
// and only for a row under `rows`. A block runs the row blocks `bx + k * gx` under `rows.div_ceil(BLOCK_ROWS)`;
// blocks of one launch have distinct `bx < gx`, so each row block, and each row, belongs to exactly one block.
// For a routed tile, the input and output pointers span the full checked launch. The group's
// indices select distinct checked input and output slots. The block ordinal and warp
// give each warp in the PICKED_THREADS-thread block disjoint 16-row tiles. The
// two block stages and the shared weight table fit PICKED_BLOCK_BYTES. The code
// and scale copies address the same layout as code_copy and scale_copy. Each copy
// takes one aligned interval from a live row and stage. A fragment past the last
// weight row or input slot contributes zero and makes no load or store.
// Every thread waits for its copies and reaches the block barrier before reading
// the next stage. The same barrier ensures the previous stage is no longer live.
// A non-null caps pointer names four initialized u64 words from the checked command.
// There are an even number of stored rows. Both lanes of each gate pair execute the
// same shuffle and the even row alone writes the compact result.
// Without a group, the caller supplies its aligned shared::<BLOCK_BYTES>() buffer.
// Two product buffers take turns, and each stage barrier protects the active buffer.
// Words that do not pass `admitted` (shape, alignment, thread count) trap before any access.
pub(crate) unsafe fn tile<const FORM:u8>(smem:u32,weights: *const u8, input: *const f32, output: *mut f32, rows: u32, word: u32,caps:*const u64,group:Option<SharedPicks>) {
    let wide_form=match FORM {0=>false,1=>true,_=>word&WIDE!=0};
    let pairs=word&PAIRS!=0;
    let (rows, cols) = (rows as usize, (word & COLS) as usize);
    let threads = nvptx::_block_dim_x() as usize;
    // Words of a shape, an alignment or a block size that the tile does not take trap here, before any access. The
    // lengths of the operands and their overlap are the host check's alone (`packed_product::check`).
    let expected_threads = if group.is_some() { PICKED_THREADS } else { THREADS };
    if threads != expected_threads || !admitted(rows, cols, wide_form, pairs, weights as usize, input as usize, THREADS) { nvptx::trap(); }
    let tid = nvptx::_thread_idx_x() as usize;
    let lane = tid & 31;
    // Bytes of one stored row of codes and of scales, and code bytes of one row in a stage.
    let parts = layout(rows, cols, wide_form);
    let codes = weights;
    let scales = weights.add(parts.scales);
    let table = weights.add(parts.table);
    if let Some(group) = group {
        // Packed offsets fit u32: the admitted matrix has at most ELEMENTS_MAX
        // values, and each input row in the launch has at most COLS_MAX words.
        let (rows, cols, tid, lane) = (rows as u32, cols as u32, tid as u32, lane as u32);
        let (count, stride, repeat, ordinal) = (group.count as u32, group.stride as u32, group.repeat as u32, group.ordinal as u32);
        let pick = |slot: u32| group.index(slot as usize) as u32;
        let block = nvptx::_block_idx_x() as u32 * count + ordinal;
        let step = nvptx::_grid_dim_x() as u32 * count * PICKED_ROWS as u32;
        if block * PICKED_ROWS as u32 >= rows { return; }
        let panel_ptr = if !wide_form && rows % 64 == 0 {
            if !group.panels.is_null() {
                group.panels
            } else {
                let t32 = table.cast::<u32>();
                if *t32.add(0x7f * 8) == crate::packed_product::NVFP4_PANEL_MAGIC_LO
                    && *t32.add(0x7f * 8 + 1) == crate::packed_product::NVFP4_PANEL_MAGIC_HI
                {
                    weights
                } else {
                    core::ptr::null()
                }
            }
        } else {
            core::ptr::null()
        };
        if !panel_ptr.is_null() {
            let t32 = table.cast::<u32>();
            let reg_nvfp4 = *t32.add(0xff * 8) == crate::packed_product::NVFP4_FACTOR_MAGIC_LO
                && *t32.add(0xff * 8 + 1) == crate::packed_product::NVFP4_FACTOR_MAGIC_HI;
            let factor_bits = if reg_nvfp4 { *t32.add(0xff * 8 + 2) } else { 0u32 };
            let meta_at = if reg_nvfp4 { PANEL_NOLUT_META_AT } else { PANEL_META_AT };
            let x_budget = if reg_nvfp4 { PANEL_NOLUT_X_BYTES as u32 } else { PANEL_X_BYTES as u32 };
            let table_addr = smem + PANEL_TABLE_AT as u32;
            if !reg_nvfp4 {
                for i in (tid..table_copies(false) as u32).step_by(PICKED_THREADS) {
                    copy16(table_addr + i * 16, table.add((i * 16) as usize).cast());
                }
            }
            if tid == 0 {
                let put64 = |off: u32, v: u64| {
                    store_u32(smem + meta_at + off, v as u32);
                    store_u32(smem + meta_at + off + 4, (v >> 32) as u32);
                };
                put64(0, output as u64);
                put64(8, caps as u64);
                put64(16, group.indices);
                put64(24, input as u64);
                store_u32(smem + meta_at + 32, stride);
                store_u32(smem + meta_at + 36, repeat);
                store_u32(smem + meta_at + 40, step);
                store_u32(smem + meta_at + 44, factor_bits);
            }
            let chunks = cols / 64;
            let max_slab = ((x_budget / count) - 32) / 128;
            let one_shot_x = chunks <= max_slab;
            let slab_chunks: u32 = if one_shot_x { chunks } else { max_slab };
            let row_stride: u32 = slab_chunks * 128 + 32;
            macro_rules! load_meta64 {
                ($off:expr) => {{
                    (load_u32(smem + meta_at + $off) as u64)
                        | ((load_u32(smem + meta_at + $off + 4) as u64) << 32)
                }};
            }
            if one_shot_x {
                let pairs_in_slab = chunks * 16;
                let mut g_idx = 0u32;
                while g_idx < count {
                    let xb = (pick(g_idx) / repeat) * stride;
                    let g_smem = smem + g_idx * row_stride;
                    let mut pair = tid;
                    while pair < pairs_in_slab {
                        let c_i = pair >> 4;
                        let ks = pair & 15;
                        let xp = input.add((xb + c_i * 64 + (ks >> 2) * 16 + (ks & 3) * 2) as usize);
                        let lo = crate::tc_ptx::ld_global_nc_v2_u32(xp.cast());
                        let hi = crate::tc_ptx::ld_global_nc_v2_u32(xp.add(8).cast());
                        let dst = g_smem + c_i * 128 + ks * 8;
                        store_u32(dst, cvt_bf16x2(f32::from_bits(lo[0]), f32::from_bits(lo[1])));
                        store_u32(dst + 4, cvt_bf16x2(f32::from_bits(hi[0]), f32::from_bits(hi[1])));
                        pair += PICKED_THREADS as u32;
                    }
                    g_idx += 1;
                }
            }
            if !reg_nvfp4 {
                commit();
                crate::ptx::wait();
            }
            _syncthreads();

            let (g, s) = (lane >> 2, lane & 3);
            let warp = tid / 32;
            let g_slot = if g < count { g } else { 0 };
            let x_base = smem + g_slot * row_stride + s * 8;
            let codes_at = panel_ptr.cast::<u32>();
            let scales_at = panel_ptr.add(parts.scales).cast::<u32>();
            let l2_evict: u64;
            asm!("createpolicy.fractional.L2::evict_first.b64 {p}, 1.0;", p = out(reg64) l2_evict, options(pure, nomem, nostack));
            let mut first = block * PICKED_ROWS as u32;
            while first < rows {
                let stripes = first / 64 * chunks;
                macro_rules! fetch_panel {
                    ($chunk:expr) => {{
                        let stripe = (stripes + $chunk) * 4 + warp;
                        let c_addr = codes_at.add((stripe * 128 + lane * 4) as usize);
                        let s_addr = scales_at.add((stripe * 16 + g * 2) as usize);
                        let (a, b, c, d, s0, s1): (u32, u32, u32, u32, u32, u32);
                        asm!("ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a},{b},{c},{d}}},[{addr}],{h};",
                             a=out(reg32) a, b=out(reg32) b, c=out(reg32) c, d=out(reg32) d,
                             addr=in(reg64) c_addr, h=in(reg64) l2_evict, options(readonly, nostack));
                        asm!("ld.global.nc.L2::cache_hint.L2::128B.v2.u32 {{{s0},{s1}}},[{addr}],{h};",
                             s0=out(reg32) s0, s1=out(reg32) s1,
                             addr=in(reg64) s_addr, h=in(reg64) l2_evict, options(readonly, nostack));
                        ([a, b, c, d], [s0, s1])
                    }};
                }
                let mut acc = [0.0f32; 4];
                if reg_nvfp4 {
                    // Bit-exact SM121a register NVFP4 decode (`cvt.rn.f16x2.{e4m3x2,e2m1x2}` + exact FP16 mul +
                    // FP32 `* factor` + `cvt.rn.bf16x2.f32`), zero 8 KB SMEM LUT, DEPTH=6 chunks in flight.
                    let factor = f32::from_bits(load_u32(smem + meta_at + 44));
                    const DEPTH_REG: u32 = 6;
                    macro_rules! step_reg {
                        ($x_addr:expr, $packed:expr, $scales:expr) => {{
                            let packed: u32 = $packed;
                            let scale_bits: u32 = $scales;
                            let (r0, r1, r2, r3): (u32, u32, u32, u32);
                            asm!(
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
                                "mov.b32 {{%h0, %h1}}, %p1;",
                                "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                "cvt.rn.bf16x2.f32 {r1}, %f1, %f0;",
                                "mov.b32 {{%h0, %h1}}, %p2;",
                                "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                "cvt.rn.bf16x2.f32 {r2}, %f1, %f0;",
                                "mov.b32 {{%h0, %h1}}, %p3;",
                                "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                "cvt.rn.bf16x2.f32 {r3}, %f1, %f0;",
                                "}}",
                                w = in(reg32) packed,
                                s2 = in(reg32) scale_bits,
                                fac = in(reg32) factor,
                                r0 = out(reg32) r0,
                                r1 = out(reg32) r1,
                                r2 = out(reg32) r2,
                                r3 = out(reg32) r3,
                                options(pure, nomem, nostack)
                            );
                            let values = load2($x_addr);
                            acc = mma_m16n8k16_bf16([r0, r1, r2, r3], values, acc);
                        }};
                    }
                    macro_rules! consume_reg {
                        ($slot:expr, $x:expr) => {{
                            let (code_words, scale_words): ([u32; 4], [u32; 2]) = $slot;
                            let x: u32 = $x;
                            step_reg!(x, code_words[0], scale_words[0]);
                            step_reg!(x + 32, code_words[1], scale_words[0] >> 16);
                            step_reg!(x + 64, code_words[2], scale_words[1]);
                            step_reg!(x + 96, code_words[3], scale_words[1] >> 16);
                        }};
                    }
                    if one_shot_x {
                        let last_c = chunks - 1;
                        let mut ring = [([0u32; 4], [0u32; 2]); DEPTH_REG as usize];
                        macro_rules! prime6 {
                            ($($j:literal)*) => {$( ring[$j as usize] = fetch_panel!(if $j < last_c { $j } else { last_c }); )*};
                        }
                        prime6!(0 1 2 3 4 5);
                        let mut ci = 0u32;
                        while ci + DEPTH_REG <= chunks {
                            let xb = x_base + ci * 128;
                            macro_rules! stage6 {
                                ($($j:literal)*) => {$(
                                    consume_reg!(ring[$j as usize], xb + $j * 128);
                                    let nxt = ci + $j + DEPTH_REG;
                                    ring[$j as usize] = fetch_panel!(if nxt < last_c { nxt } else { last_c });
                                )*};
                            }
                            stage6!(0 1 2 3 4 5);
                            ci += DEPTH_REG;
                        }
                        let xb = x_base + ci * 128;
                        macro_rules! tail6 {
                            ($($j:literal)*) => {$( if ci + $j < chunks { consume_reg!(ring[$j as usize], xb + $j * 128); } )*};
                        }
                        tail6!(0 1 2 3 4);
                    } else {
                        let mut slab_start = 0u32;
                        while slab_start < chunks {
                            let slab_len = (chunks - slab_start).min(slab_chunks);
                            _syncthreads();
                            let in_ptr = load_meta64!(24) as *const f32;
                            let in_stride = load_u32(smem + meta_at + 32);
                            let in_repeat = load_u32(smem + meta_at + 36);
                            let idx_word = load_meta64!(16);
                            let pairs_in_slab = slab_len * 16;
                            let mut g_idx = 0u32;
                            while g_idx < count {
                                let pk = ((idx_word >> (8 * g_idx)) & 255) as u32;
                                let xb = (pk / in_repeat) * in_stride + slab_start * 64;
                                let g_smem = smem + g_idx * row_stride;
                                let mut pair = tid;
                                while pair < pairs_in_slab {
                                    let c_i = pair >> 4;
                                    let ks = pair & 15;
                                    let xp = in_ptr.add((xb + c_i * 64 + (ks >> 2) * 16 + (ks & 3) * 2) as usize);
                                    let lo = crate::tc_ptx::ld_global_nc_v2_u32(xp.cast());
                                    let hi = crate::tc_ptx::ld_global_nc_v2_u32(xp.add(8).cast());
                                    let dst = g_smem + c_i * 128 + ks * 8;
                                    store_u32(dst, cvt_bf16x2(f32::from_bits(lo[0]), f32::from_bits(lo[1])));
                                    store_u32(dst + 4, cvt_bf16x2(f32::from_bits(hi[0]), f32::from_bits(hi[1])));
                                    pair += PICKED_THREADS as u32;
                                }
                                g_idx += 1;
                            }
                            _syncthreads();
                            let slab_last = slab_start + slab_len - 1;
                            let mut ring = [([0u32; 4], [0u32; 2]); DEPTH_REG as usize];
                            macro_rules! prime6s {
                                ($($j:literal)*) => {$(
                                    let p = slab_start + $j;
                                    ring[$j as usize] = fetch_panel!(if p < slab_last { p } else { slab_last });
                                )*};
                            }
                            prime6s!(0 1 2 3 4 5);
                            let mut ci = 0u32;
                            while ci + DEPTH_REG <= slab_len {
                                let xb = x_base + ci * 128;
                                macro_rules! stage6s {
                                    ($($j:literal)*) => {$(
                                        consume_reg!(ring[$j as usize], xb + $j * 128);
                                        let nxt = slab_start + ci + $j + DEPTH_REG;
                                        ring[$j as usize] = fetch_panel!(if nxt < slab_last { nxt } else { slab_last });
                                    )*};
                                }
                                stage6s!(0 1 2 3 4 5);
                                ci += DEPTH_REG;
                            }
                            let xb = x_base + ci * 128;
                            macro_rules! tail6s {
                                ($($j:literal)*) => {$( if ci + $j < slab_len { consume_reg!(ring[$j as usize], xb + $j * 128); } )*};
                            }
                            tail6s!(0 1 2 3 4);
                            slab_start += slab_len;
                        }
                    }
                } else {
                    const DEPTH: u32 = 4;
                    macro_rules! step_smem {
                        ($x_addr:expr, $packed:expr, $scales:expr) => {{
                            let packed = $packed;
                            let scale_bits = $scales;
                            let top = table_addr + 32 * (scale_bits & 255);
                            let bottom = table_addr + 32 * ((scale_bits >> 8) & 255);
                            let decode = |line: u32, byte: u32| load_u16(line + 2 * (byte & 15)) | (load_u16(line + 2 * ((byte >> 4) & 15)) << 16);
                            let weights = [decode(top, packed), decode(bottom, packed >> 8), decode(top, packed >> 16), decode(bottom, packed >> 24)];
                            let values = load2($x_addr);
                            acc = mma_m16n8k16_bf16(weights, values, acc);
                        }};
                    }
                    macro_rules! consume_smem {
                        ($slot:expr, $x:expr) => {{
                            let (code_words, scale_words): ([u32; 4], [u32; 2]) = $slot;
                            let x: u32 = $x;
                            step_smem!(x, code_words[0], scale_words[0]);
                            step_smem!(x + 32, code_words[1], scale_words[0] >> 16);
                            step_smem!(x + 64, code_words[2], scale_words[1]);
                            step_smem!(x + 96, code_words[3], scale_words[1] >> 16);
                        }};
                    }
                    if one_shot_x {
                        let last_c = chunks - 1;
                        let mut ring = [([0u32; 4], [0u32; 2]); DEPTH as usize];
                        macro_rules! prime4 {
                            ($($j:literal)*) => {$( ring[$j as usize] = fetch_panel!(if $j < last_c { $j } else { last_c }); )*};
                        }
                        prime4!(0 1 2 3);
                        let mut ci = 0u32;
                        while ci + DEPTH <= chunks {
                            let xb = x_base + ci * 128;
                            macro_rules! stage4 {
                                ($($j:literal)*) => {$(
                                    consume_smem!(ring[$j as usize], xb + $j * 128);
                                    let nxt = ci + $j + DEPTH;
                                    ring[$j as usize] = fetch_panel!(if nxt < last_c { nxt } else { last_c });
                                )*};
                            }
                            stage4!(0 1 2 3);
                            ci += DEPTH;
                        }
                        let xb = x_base + ci * 128;
                        macro_rules! tail4 {
                            ($($j:literal)*) => {$( if ci + $j < chunks { consume_smem!(ring[$j as usize], xb + $j * 128); } )*};
                        }
                        tail4!(0 1 2);
                    } else {
                        let mut slab_start = 0u32;
                        while slab_start < chunks {
                            let slab_len = (chunks - slab_start).min(slab_chunks);
                            _syncthreads();
                            let in_ptr = load_meta64!(24) as *const f32;
                            let in_stride = load_u32(smem + meta_at + 32);
                            let in_repeat = load_u32(smem + meta_at + 36);
                            let idx_word = load_meta64!(16);
                            let pairs_in_slab = slab_len * 16;
                            let mut g_idx = 0u32;
                            while g_idx < count {
                                let pk = ((idx_word >> (8 * g_idx)) & 255) as u32;
                                let xb = (pk / in_repeat) * in_stride + slab_start * 64;
                                let g_smem = smem + g_idx * row_stride;
                                let mut pair = tid;
                                while pair < pairs_in_slab {
                                    let c_i = pair >> 4;
                                    let ks = pair & 15;
                                    let xp = in_ptr.add((xb + c_i * 64 + (ks >> 2) * 16 + (ks & 3) * 2) as usize);
                                    let lo = crate::tc_ptx::ld_global_nc_v2_u32(xp.cast());
                                    let hi = crate::tc_ptx::ld_global_nc_v2_u32(xp.add(8).cast());
                                    let dst = g_smem + c_i * 128 + ks * 8;
                                    store_u32(dst, cvt_bf16x2(f32::from_bits(lo[0]), f32::from_bits(lo[1])));
                                    store_u32(dst + 4, cvt_bf16x2(f32::from_bits(hi[0]), f32::from_bits(hi[1])));
                                    pair += PICKED_THREADS as u32;
                                }
                                g_idx += 1;
                            }
                            _syncthreads();
                            let slab_last = slab_start + slab_len - 1;
                            let mut ring = [([0u32; 4], [0u32; 2]); DEPTH as usize];
                            macro_rules! prime4s {
                                ($($j:literal)*) => {$(
                                    let p = slab_start + $j;
                                    ring[$j as usize] = fetch_panel!(if p < slab_last { p } else { slab_last });
                                )*};
                            }
                            prime4s!(0 1 2 3);
                            let mut ci = 0u32;
                            while ci + DEPTH <= slab_len {
                                let xb = x_base + ci * 128;
                                macro_rules! stage4s {
                                    ($($j:literal)*) => {$(
                                        consume_smem!(ring[$j as usize], xb + $j * 128);
                                        let nxt = slab_start + ci + $j + DEPTH;
                                        ring[$j as usize] = fetch_panel!(if nxt < slab_last { nxt } else { slab_last });
                                    )*};
                                }
                                stage4s!(0 1 2 3);
                                ci += DEPTH;
                            }
                            let xb = x_base + ci * 128;
                            macro_rules! tail4s {
                                ($($j:literal)*) => {$( if ci + $j < slab_len { consume_smem!(ring[$j as usize], xb + $j * 128); } )*};
                            }
                            tail4s!(0 1 2);
                            slab_start += slab_len;
                        }
                    }
                }
                let out_ptr = load_meta64!(0) as *mut f32;
                let caps_ptr = load_meta64!(8) as *const u64;
                let idx_word = load_meta64!(16);
                let top = first + warp * 16 + g;
                let mut col_i = 0u32;
                while col_i < 2 {
                    let slot = 2 * s + col_i;
                    if slot < count {
                        let pk = ((idx_word >> (8 * slot)) & 255) as u32;
                        let (v0, v1) = if col_i == 0 { (acc[0], acc[2]) } else { (acc[1], acc[3]) };
                        if caps_ptr.is_null() {
                            let row_offset = |row: u32| if pairs { row / 2 + (row & 1) * (rows / 2) } else { row };
                            let out = out_ptr.add((pk * rows) as usize);
                            *out.add(row_offset(top) as usize) = v0;
                            *out.add(row_offset(top + 8) as usize) = v1;
                        } else {
                            use crate::fixed_math::{rn, sigmoid};
                            let half = (g & 1) as usize;
                            let cap = f32::from_bits(*caps_ptr.add(half) as u32);
                            let inv = f32::from_bits(*caps_ptr.add(2 + half) as u32);
                            let columns = (count + 1 - (slot & 1)) / 2;
                            let mask = 0x1111_1111u32 * ((1u32 << columns) - 1);
                            let out = out_ptr.add((pk * (rows / 2)) as usize);
                            let write = |row: u32, value: f32| {
                                let value = rn(value);
                                let capped = gate_tanh(value * inv) * cap;
                                let other: u32;
                                asm!("shfl.sync.bfly.b32 {out},{v},4,31,{mask};",
                                     out=out(reg32) other, v=in(reg32) capped.to_bits(), mask=in(reg32) mask);
                                if half == 0 {
                                    *out.add((row / 2) as usize) = rn((capped * sigmoid(value)) * f32::from_bits(other));
                                }
                            };
                            write(top, v0);
                            write(top + 8, v1);
                        }
                    }
                    col_i += 1;
                }
                first += load_u32(smem + meta_at + 40);
            }
            return;
        }
        let table_addr = smem + picked_table_at(wide_form) as u32;
        for i in (tid..table_copies(wide_form) as u32).step_by(PICKED_THREADS) {
            copy16(table_addr + i * 16, table.add((i * 16) as usize).cast());
        }
        // A supplied table can define other values. Compare all 256 words
        // before replacing any table reads with the native E4M3 conversion.
        let native = if wide_form {
            let pair = (tid | ((128 + tid) << 8)) as u16;
            let (lo, hi): (f32, f32);
            asm!("{{ .reg .b32 h; .reg .b16 a,b; cvt.rn.f16x2.e4m3x2 h,{pair}; mov.b32 {{a,b}},h; cvt.f32.f16 {lo},a; cvt.f32.f16 {hi},b; }}",
                 pair=in(reg16) pair, lo=out(reg32) lo, hi=out(reg32) hi, options(pure,nomem,nostack));
            let table = table.cast::<u32>();
            let ok = *table.add(tid as usize) == (if tid == 127 { 0 } else { lo.to_bits() })
                && *table.add(tid as usize + 128) == (if tid == 127 { 0 } else { hi.to_bits() });
            let result: u32;
            asm!("{{ .reg .pred p,q; setp.ne.u32 p,{ok},0; bar.red.and.pred q,0,p; selp.u32 {result},1,0,q; }}",
                 ok=in(reg32) u32::from(ok), result=out(reg32) result);
            result != 0
        } else { false };
        let code_bytes = code_stage(wide_form) as u32;
        let scale_at = picked_scale_at(wide_form) as u32;
        let input_at = picked_input_at(wide_form) as u32;
        let stage_bytes = picked_stage_bytes(wide_form) as u32;
        let chunks = cols / CH as u32;
        let (g, s) = (lane >> 2, lane & 3);
        let (x_row, x_col) = (tid / 16, (tid % 16) * 4);
        let x_live = x_row < count;
        let x_base = if x_live { (pick(x_row) / repeat) * stride + x_col } else { 0 };
        let warp_row = tid / 32 * WARP_ROWS as u32;
        let (copy_row, copy_part) = (tid * 16 / code_bytes, tid * 16 % code_bytes);

        let mut first = block * PICKED_ROWS as u32;
        while first < rows {
            let stage = |chunk: u32| smem + (chunk & 1) * stage_bytes;
            let (r0, r1) = (warp_row + g, warp_row + g + 8);
            let live0 = first + r0 < rows;
            let live1 = first + r1 < rows;
            let code_base = first * chunks * code_bytes;
            let scale_stages = if wide_form { chunks / 2 } else { chunks };
            let scale_base = first * scale_stages * 4;
            macro_rules! issue_picked {
                ($chunk:expr) => {{
                    let chunk = $chunk;
                    let dst = stage(chunk);
                    let group_rows = BLOCK_ROWS as u32;
                    let row_start = ((copy_row / group_rows * chunks + chunk) * group_rows + copy_row % group_rows) * code_bytes;
                    let code_off = code_base + row_start + copy_part;
                    if first + copy_row < rows {
                        copy16(dst + picked_code_offset(copy_row, copy_part, code_bytes), codes.add(code_off as usize).cast());
                    }
                    if wide_form && first + copy_row + group_rows < rows {
                        copy16(dst + picked_code_offset(copy_row + group_rows, copy_part, code_bytes),
                               codes.add((code_off + group_rows * chunks * code_bytes) as usize).cast());
                    }
                    if tid < PICKED_ROWS as u32 && first + tid < rows {
                        let scale_chunk = if wide_form { chunk / 2 } else { chunk };
                        let scale_off = scale_base + ((tid / group_rows * scale_stages + scale_chunk) * group_rows + tid % group_rows) * 4;
                        copy4(dst + scale_at + tid * 4, scales.add(scale_off as usize).cast());
                    }
                    if x_live {
                        let values = ld_global_nc_v4(input.add((x_base + chunk * CH as u32) as usize));
                        let p = dst + input_at + picked_input_offset(x_row, x_col);
                        store_u32(p, cvt_bf16x2(values[0], values[1]));
                        store_u32(p + 4, cvt_bf16x2(values[2], values[3]));
                    }
                }};
            }
            let mut acc = [0.0f32; 4];
            for ahead in 0..=chunks {
                if ahead < chunks { issue_picked!(ahead); }
                commit();
                if ahead != 0 {
                    let chunk = ahead - 1;
                    let cur = stage(chunk);
                    // 4-bit form: the scale word and the two 16-byte halves of each row of this stage, read once a stage.
                    // Byte 8k+s of a row is byte s of word 2k, byte 8k+s+4 is byte s of word 2k+1 (the same bytes as
                    // one load a byte).
                    let narrow_row = |row: u32, live: bool| if wide_form || !live { (0u32, [0u32; 4], [0u32; 4]) } else {
                        (load_u32(cur + scale_at + row * 4), load4(cur + picked_code_offset(row, 0, code_bytes)), load4(cur + picked_code_offset(row, 16, code_bytes)))
                    };
                    let (n0, n1) = (narrow_row(r0, live0), narrow_row(r1, live1));
                    for k in 0..4u32 {
                        macro_rules! weight_pairs {
                            ($row:expr, $narrow:expr) => {{
                                let column = 16 * k + 2 * s;
                                let scale_row = cur + scale_at + $row * 4;
                                if wide_form {
                                    let scale = load_u8(scale_row + (chunk & 1) * 2 + k / 2);
                                    let multiplier = power(scale);
                                    let code = cur + picked_code_offset($row, column, code_bytes);
                                    let unpack = |bits: u32| {
                                        if native && (1..143).contains(&scale) {
                                            // The exact half values become BF16 at 2^-112.
                                            // Move that power into the packed scale. Preserve
                                            // the table's zero at either reserved NaN code.
                                            let clean = (if bits & 127 == 127 { 0 } else { bits & 255 })
                                                | (if bits & 0x7f00 == 0x7f00 { 0 } else { bits & 0xff00 });
                                            let h: u32;
                                            asm!("cvt.rn.f16x2.e4m3x2 {h},{pair};", h=out(reg32) h, pair=in(reg16) clean as u16, options(pure,nomem,nostack));
                                            let values = (h & 0x80008000) | ((h >> 3) & 0x0fff0fff);
                                            let shifted = ((scale + 112) << 7) * 0x10001;
                                            let result: u32;
                                            asm!("mul.rn.bf16x2 {result},{values},{shifted};", result=out(reg32) result,
                                                 values=in(reg32) values, shifted=in(reg32) shifted, options(pure,nomem,nostack));
                                            result
                                        } else {
                                            cvt_bf16x2(load(table_addr + 4 * (bits & 255)) * multiplier,
                                                       load(table_addr + 4 * (bits >> 8)) * multiplier)
                                        }
                                    };
                                    (unpack(load_u16(code)), unpack(load_u16(code + 8)))
                                } else {
                                    let (scale_word, low, high) = $narrow;
                                    let line = table_addr + 32 * ((scale_word >> (8 * k)) & 255);
                                    let (even, odd) = match k { 0 => (low[0], low[1]), 1 => (low[2], low[3]), 2 => (high[0], high[1]), _ => (high[2], high[3]) };
                                    let byte = |w: u32| (w >> (8 * s)) & 255;
                                    let unpack = |bits: u32| load_u16(line + 2 * (bits & 15)) | (load_u16(line + 2 * (bits >> 4)) << 16);
                                    (unpack(byte(even)), unpack(byte(odd)))
                                }
                            }};
                        }
                        let (a0, a2) = if live0 { weight_pairs!(r0, n0) } else { (0, 0) };
                        let (a1, a3) = if live1 { weight_pairs!(r1, n1) } else { (0, 0) };
                        let b = if g < count {
                            let x = cur + input_at;
                            [load_u32(x + picked_input_offset(g, k * 16 + 2 * s)),
                             load_u32(x + picked_input_offset(g, k * 16 + 2 * s + 8))]
                        } else { [0, 0] };
                        acc = mma_m16n8k16_bf16([a0, a1, a2, a3], b, acc);
                    }
                }
                crate::ptx::wait();
                _syncthreads();
            }
            let (t0, t1) = (2 * s, 2 * s + 1);
            let (r0, r1) = (first + r0, first + r1);
            if caps.is_null() {
                let row_offset = |row| if pairs { row / 2 + (row & 1) * (rows / 2) } else { row };
                macro_rules! write_column {
                    ($slot:expr, $v0:expr, $v1:expr) => {{
                        if $slot < count {
                            let out = output.add((pick($slot) * rows) as usize);
                            if live0 { *out.add(row_offset(r0) as usize) = $v0; }
                            if live1 { *out.add(row_offset(r1) as usize) = $v1; }
                        }
                    }};
                }
                write_column!(t0, acc[0], acc[2]);
                write_column!(t1, acc[1], acc[3]);
            } else {
                use crate::fixed_math::{rn, sigmoid};
                let half = (g & 1) as usize;
                let cap = f32::from_bits(*caps.add(half) as u32);
                let inv = f32::from_bits(*caps.add(2 + half) as u32);
                // Each mask repeats the live token columns over all row groups.
                // Lane bit 2 selects the adjacent gate row of the same token.
                macro_rules! write_gate {
                    ($slot:expr, $v0:expr, $v1:expr) => {{
                        if $slot < count {
                            let columns = (count + 1 - ($slot & 1)) / 2;
                            let mask = 0x1111_1111u32 * ((1u32 << columns) - 1);
                            let out = output.add((pick($slot) * (rows / 2)) as usize);
                            let store_gate = |row: u32, value: f32| {
                                let value = rn(value);
                                let capped = gate_tanh(value * inv) * cap;
                                let other: u32;
                                asm!("shfl.sync.bfly.b32 {out}, {v}, 4, 31, {mask};",
                                     out=out(reg32) other, v=in(reg32) capped.to_bits(), mask=in(reg32) mask);
                                if half == 0 && row < rows {
                                    *out.add((row / 2) as usize) = rn((capped * sigmoid(value)) * f32::from_bits(other));
                                }
                            };
                            store_gate(r0, $v0);
                            store_gate(r1, $v1);
                        }
                    }};
                }
                write_gate!(t0, acc[0], acc[2]);
                write_gate!(t1, acc[1], acc[3]);
            }
            first += step;
        }
        return;
    }
    // Warp 0 runs the direct chains. The other threads make products.
    let maker = tid >= 32;
    let j = tid.wrapping_sub(32);
    let active = nvptx::_grid_dim_x() as usize;
    let tile_index = nvptx::_block_idx_x() as usize;
    let direct_panel = !wide_form && rows % 64 == 0 && {
        let t32 = table.cast::<u32>();
        *t32.add(0x7f * 8) == crate::packed_product::NVFP4_PANEL_MAGIC_LO
            && *t32.add(0x7f * 8 + 1) == crate::packed_product::NVFP4_PANEL_MAGIC_HI
    };
    let m = Stored { codes, scales, input, rows, cols, wide: wide_form, panel: direct_panel };
    let chunks = stages(cols);
    // The row blocks of this block: `bx`, `bx + gx`, ... below `rows`, one after the other. Their stages run as ONE
    // stream, so the copies of the next row block start while the last stages of this one run, and the weight table
    // is copied once a block. A place in the stream is (stage, first row of its row block); `next` steps it without a
    // division. A place whose first row is at or above `rows` is past the end of the stream.
    let step = active * BLOCK_ROWS;
    let start = (0usize, tile_index * BLOCK_ROWS);
    let next = |(c, first): (usize, usize)| if c + 1 == chunks { (0, first + step) } else { (c + 1, first) };

    // The place of the next copies (`ahead`), of the next prefetch (`fetch`), of the products (`made`), of the chain.
    let mut ahead = start;
    let mut fetch = start;
    let mut n = 0;
    while n < AHEAD { fetch = next(fetch); n += 1; }
    if maker {
        // The weight table of the unit, into shared memory.
        let mut t = j;
        while t < table_copies(wide_form) { copy16(smem + (table_at(wide_form) + t * 16) as u32, table.add(t * 16).cast()); t += MAKERS; }
        let mut g = 0;
        while g + 1 < NS {
            if ahead.1 < rows { issue(g, ahead.0, smem, j, ahead.1, m); }
            ahead = next(ahead);
            commit();
            g += 1;
        }
        asm!("cp.async.wait_group {n};", n = const (NS - 2));
    }
    _syncthreads();
    let (mut made, mut chain) = (start, start);
    let mut sum = 0.0f32;
    let mut g = 0usize;
    loop {
        let (c, first) = made;
        let live = first < rows;
        if maker {
            if ahead.1 < rows { issue((g + NS - 1) % NS, ahead.0, smem, j, ahead.1, m); }
            ahead = next(ahead);
            commit();
            if j < BLOCK_ROWS && fetch.1 + j < rows {
                asm!("prefetch.global.L2 [{a}];", a = in(reg64) codes.add(code_copy(fetch.1 + j, fetch.0, 0, cols, wide_form)) as u64);
                asm!("prefetch.global.L2 [{a}];", a = in(reg64) scales.add(scale_copy(fetch.1 + j, fetch.0, cols, wide_form)) as u64);
            }
            fetch = next(fetch);
            if live {
                // The rounded products of stage `c` of the row block from row `first`: a unit is 16 columns of one row.
                let slot = smem + (STAGES_AT + (g % NS) * stage_bytes(wide_form)) as u32;
                let buffer = smem + ((g & 1) * P_BYTES) as u32;
                let mut u = j;
                while u < 32 * (CH / UNIT) {
                    let (row,k)=(u & 31,u >> 5);
                    if first + row < rows {
                        let out = buffer + (row * P_ROW + k * UNIT * 4) as u32;
                        let x_at = x_at(wide_form);
                        let xs = slot + (x_at + k * UNIT * 4) as u32;
                        if wide_form {
                            let scale = load_u8(slot + (scale_at(wide_form) + row * 4 + (c & 1) * 2 + k / 2) as u32);
                            let power = power(scale);
                            let v = load4(slot + (row * code_pitch(m.wide) + k * 16) as u32);
                            let values = smem + table_at(wide_form) as u32;
                            let mut q = 0;
                            while q < 4 {
                                let x = load4(xs + (q * 16) as u32);
                                let (p0, p1, p2, p3) = (wide(values, power, v[q], 0, x[0]), wide(values, power, v[q], 1, x[1]), wide(values, power, v[q], 2, x[2]), wide(values, power, v[q], 3, x[3]));
                                asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) out + (q * 16) as u32, a = in(reg32) p0, b = in(reg32) p1, c = in(reg32) p2, d = in(reg32) p3);
                                q += 1;
                            }
                        } else {
                            let scale = load_u8(slot + (scale_at(wide_form) + row * 4 + k) as u32);
                            let line = smem + table_at(wide_form) as u32 + scale * 32;
                            let pair = load2(slot + (row * code_pitch(m.wide) + k * 8) as u32);
                            let mut q = 0;
                            while q < 4 {
                                let x = load4(xs + (q * 16) as u32);
                                // The low half of a byte is the even column.
                                let four = pair[q / 2] >> (16 * (q & 1));
                                let (p0, p1, p2, p3) = (narrow(line, four, 0, x[0]), narrow(line, four, 1, x[1]), narrow(line, four, 2, x[2]), narrow(line, four, 3, x[3]));
                                asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) out + (q * 16) as u32, a = in(reg32) p0, b = in(reg32) p1, c = in(reg32) p2, d = in(reg32) p3);
                                q += 1;
                            }
                        }
                    }
                    u += MAKERS;
                }
            }
            asm!("cp.async.wait_group {n};", n = const (NS - 2));
        }
        if tid < 32 && g >= 1 {
            // The chain of this lane's row over the products of the previous place; at the last stage of a row block
            // the lane stores its row and starts the next row block's chain from +0.0.
            let (pc, pfirst) = chain;
            let matrix_row=pfirst+lane;
            if matrix_row < rows {
                let base = smem + (((g - 1) & 1) * P_BYTES + lane * P_ROW) as u32;
                let mut q = 0;
                while q < CH / 4 {
                    let w = load4(base + (q * 16) as u32);
                    let mut e = 0;
                    while e < 4 {
                        asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) f32::from_bits(w[e]), options(nomem, nostack));
                        e += 1;
                    }
                    q += 1;
                }
                if pc + 1 == chunks {
                    *output.add(output_row(matrix_row,rows,pairs))=sum;
                }
            }
            if pc + 1 == chunks { sum = 0.0; }
            chain = next(chain);
        }
        _syncthreads();
        // The place past the end: its iteration ran only the chain of the last place.
        if !live { break; }
        made = next(made);
        g += 1;
    }
}
