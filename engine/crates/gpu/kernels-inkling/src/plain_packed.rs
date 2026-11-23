//! The packed row product (`packed_product`): the kernel body.
//!
//! One launch reads the unit of ONE matrix in its stored packed form and gives the words of `project_bf16_ordered` on
//! the matrix widened to BF16: each output is one chain over the columns in rising order from +0.0, each product
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
//! A block of `packed_product::THREADS` threads owns 32 stored rows. Warp 0 runs the 32 chains: a lane adds the
//! rounded products of its row from shared memory, in column order. The four other warps are makers: they copy the
//! codes, the scale bytes and the inputs of a stage of 64 columns into shared memory (`cp.async`, `NS` stages deep),
//! and make the rounded products of the stage one stage before the chain reads them. Two product buffers take turns,
//! and one barrier a stage keeps a buffer from being written while a lane still reads it.
use crate::packed_product::{BLOCK_ROWS, COLS, NARROW_TABLE, PAIRS, STAGE, THREADS, WIDE, admitted, code_copy, code_stage, code_stride, input_copy, layout, output_row, power, round_bf16, scale_copy, scale_stride, stages, table_copies};
use crate::ptx::{commit, copy4, copy16, load, load2, load4, load_u8, load_u16, shared, sync};
use core::arch::{asm, nvptx};

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
/// Shared bytes of the codes of one row in a stage (64 in the 8-bit form), with a pad. A multiple of 16: a 16-byte
/// copy needs an aligned target.
pub(crate) const CODE_ROW: usize = CH + 16;
pub(crate) const SCALE_AT: usize = BLOCK_ROWS * CODE_ROW;
pub(crate) const X_AT: usize = SCALE_AT + BLOCK_ROWS * 4;
pub(crate) const STAGE_BYTES: usize = X_AT + CH * 4;
pub(crate) const STAGES_AT: usize = 2 * P_BYTES;
pub(crate) const TABLE_AT: usize = STAGES_AT + NS * STAGE_BYTES;
pub(crate) const BLOCK_BYTES: usize = TABLE_AT + NARROW_TABLE;
const _: () = assert!(BLOCK_ROWS == 32 && MAKERS == BLOCK_ROWS * (CH / UNIT) && CODE_ROW % 16 == 0 && P_ROW % 16 == 0 && STAGE_BYTES % 16 == 0);
/// No more shared memory than the BF16 tile of the general image (`plain_bf16.rs`) takes.
const _: () = assert!(BLOCK_BYTES <= 38_912);

/// The rounded product of column `e` of the four columns in `word`: a 4-bit code picks its BF16 weight in the table
/// line of its scale byte. `x` holds the bits of the input word.
#[inline(always)]
// SAFETY: `line` is the shared address of one line of the weight table of this block (32 bytes inside
// `TABLE_AT..BLOCK_BYTES`, copied there before the first product). The read is 2 bytes at an offset under 32.
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
// `TABLE_AT..BLOCK_BYTES`, copied there before the first product). The read is one word at an index under 256.
unsafe fn wide(values: u32, power: f32, word: u32, e: u32, x: u32) -> f32 {
    let value = load(values + ((word >> (8 * e)) & 255) * 4);
    let (scaled, product): (f32, f32);
    asm!("mul.rn.f32 {p}, {a}, {b};", p = out(reg32) scaled, a = in(reg32) value, b = in(reg32) power, options(nomem, nostack));
    asm!("mul.rn.f32 {p}, {a}, {b};", p = out(reg32) product, a = in(reg32) round_bf16(scaled), b = in(reg32) f32::from_bits(x), options(nomem, nostack));
    product
}

/// The stored rows of one launch, as the copies of a stage read them.
#[derive(Clone, Copy)]
struct Stored { codes: *const u8, scales: *const u8, input: *const f32, rows: usize, cols: usize, wide: bool }

/// The copies of stage `s` into its slot of shared memory: the codes and the scale bytes of the 32 rows of the
/// block from row `first`, and the 64 inputs. Maker `j` takes every `MAKERS`-th copy. A row at or above `rows` is
/// not copied.
#[inline(always)]
// SAFETY: the caller is `tile`, after its words passed `admitted`, with `s` under the stages of the matrix
// (`stages(cols)`). Every source offset comes from `packed_product::{code_copy, scale_copy, input_copy}`, and the
// host test `every_copy_of_the_body_is_inside_its_part_and_the_last_one_ends_at_its_end` holds them: a code copy
// reads 16 bytes that end at or before `rows * code_stride` of `codes`; a scale copy reads 4 bytes that end at or
// before `rows * scale_stride` of `scales` (in the 8-bit form a row has an even count of stages, so the pair of
// stages is inside the row); an input copy reads 16 bytes that end at or before `cols * 4` bytes of `input`. All
// three lie in the unit or in the input that the host check measured, and every source and target is on the
// boundary of its copy size. Targets: slot `s % NS` of the block's own shared memory, inside `STAGES_AT..TABLE_AT`.
unsafe fn issue(s: usize, smem: u32, j: usize, first: usize, m: Stored) {
    let slot = smem + (STAGES_AT + (s % NS) * STAGE_BYTES) as u32;
    let parts = code_stage(m.wide) / 16;
    let mut t = j;
    while t < 32 * parts + 32 + CH / 4 {
        if t < 32 * parts {
            let (row, part) = (t & 31, t >> 5);
            if first + row < m.rows { copy16(slot + (row * CODE_ROW + part * 16) as u32, m.codes.add(code_copy(first + row, s, part, m.cols, m.wide)).cast()); }
        } else if t < 32 * parts + 32 {
            let row = t - 32 * parts;
            if first + row < m.rows { copy4(slot + (SCALE_AT + row * 4) as u32, m.scales.add(scale_copy(first + row, s, m.cols, m.wide)).cast()); }
        } else {
            let i = t - 32 * parts - 32;
            copy16(slot + (X_AT + i * 16) as u32, m.input.add(input_copy(s, i)).cast());
        }
        t += MAKERS;
    }
}

/// The body of one thread. `weights` is the unit of the matrix, `word` the column word
/// (`packed_product::cols_word`); `output` gets `rows` words, gate rows first.
#[inline(always)]
// SAFETY: the caller passes the five words of a launch that the host check `packed_product::check` passed, in a
// launch of `packed_product::geometry(rows)`. That check gives: `weights` is the start of
// `layout(rows, cols, wide).bytes` readable bytes on a 16-byte boundary; `input` is `cols` readable words on a
// 16-byte boundary; `output` is `rows` writable words on a word boundary that overlap neither operand.
// Global reads: the copies of the weight table (16 bytes each, under `table_bytes(wide)` from the table part of the
// unit) and the copies of `issue` (see its note); a prefetch names a place inside the unit and reads nothing.
// Global write: one word at `output_row(row, rows, pairs)`, which is under `rows`, by the one lane that owns `row`,
// and only for a row under `rows`. Shared memory: every address is inside the `BLOCK_BYTES` of this block; two
// product buffers take turns and the barrier of each stage keeps a buffer from being written while a lane reads it.
// Words that do not pass `admitted` (shape, alignment, thread count) trap before any access.
pub(crate) unsafe fn tile(weights: *const u8, input: *const f32, output: *mut f32, rows: u32, word: u32) {
    let (wide_form, pairs) = (word & WIDE != 0, word & PAIRS != 0);
    let (rows, cols) = (rows as usize, (word & COLS) as usize);
    let threads = nvptx::_block_dim_x() as usize;
    // Words of a shape, an alignment or a block size that the tile does not take trap here, before any access. The
    // lengths of the operands and their overlap are the host check's alone (`packed_product::check`).
    if !admitted(rows, cols, wide_form, pairs, weights as usize, input as usize, threads) { nvptx::trap(); }
    let tid = nvptx::_thread_idx_x() as usize;
    let first = nvptx::_block_idx_x() as usize * BLOCK_ROWS;
    let smem = shared::<BLOCK_BYTES>();
    // Warp 0 runs the chains. The other threads make products; `j` is a maker's number among them.
    let maker = tid >= 32;
    let j = tid.wrapping_sub(32);
    let lane = tid & 31;
    // Bytes of one stored row of codes and of scales, and code bytes of one row in a stage.
    let (code_stride, scale_stride, code_stage) = (code_stride(cols, wide_form), scale_stride(cols, wide_form), code_stage(wide_form));
    let parts = layout(rows, cols, wide_form);
    let codes = weights;
    let scales = weights.add(parts.scales);
    let table = weights.add(parts.table);
    let m = Stored { codes, scales, input, rows, cols, wide: wide_form };
    let chunks = stages(cols);

    if maker {
        // The weight table of the unit, into shared memory.
        let mut t = j;
        while t < table_copies(wide_form) { copy16(smem + (TABLE_AT + t * 16) as u32, table.add(t * 16).cast()); t += MAKERS; }
        let mut s = 0;
        while s + 1 < NS {
            if s < chunks { issue(s, smem, j, first, m); }
            commit();
            s += 1;
        }
        asm!("cp.async.wait_group {n};", n = const (NS - 2));
    }
    sync();
    let live = tid < 32 && first + lane < rows;
    let mut sum = 0.0f32;
    let mut c = 0;
    while c <= chunks {
        if maker {
            if c + NS - 1 < chunks { issue(c + NS - 1, smem, j, first, m); }
            commit();
            if j < 32 && c + AHEAD < chunks && first + j < rows {
                asm!("prefetch.global.L2 [{a}];", a = in(reg64) codes.add((first + j) * code_stride + (c + AHEAD) * code_stage) as u64);
                asm!("prefetch.global.L2 [{a}];", a = in(reg64) scales.add((first + j) * scale_stride + (c + AHEAD) * (code_stage / if wide_form { 32 } else { 8 })) as u64);
            }
            if c < chunks {
                // The rounded products of stage `c`: a unit is 16 columns of one row.
                let slot = smem + (STAGES_AT + (c % NS) * STAGE_BYTES) as u32;
                let buffer = smem + ((c & 1) * P_BYTES) as u32;
                let mut u = j;
                while u < 32 * (CH / UNIT) {
                    let (row, k) = (u & 31, u >> 5);
                    if first + row < rows {
                        let out = buffer + (row * P_ROW + k * UNIT * 4) as u32;
                        let xs = slot + (X_AT + k * UNIT * 4) as u32;
                        if wide_form {
                            let scale = load_u8(slot + (SCALE_AT + row * 4 + (c & 1) * 2 + k / 2) as u32);
                            let power = power(scale);
                            let v = load4(slot + (row * CODE_ROW + k * 16) as u32);
                            let values = smem + TABLE_AT as u32;
                            let mut q = 0;
                            while q < 4 {
                                let x = load4(xs + (q * 16) as u32);
                                let (p0, p1, p2, p3) = (wide(values, power, v[q], 0, x[0]), wide(values, power, v[q], 1, x[1]), wide(values, power, v[q], 2, x[2]), wide(values, power, v[q], 3, x[3]));
                                asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) out + (q * 16) as u32, a = in(reg32) p0, b = in(reg32) p1, c = in(reg32) p2, d = in(reg32) p3);
                                q += 1;
                            }
                        } else {
                            let scale = load_u8(slot + (SCALE_AT + row * 4 + k) as u32);
                            let line = smem + TABLE_AT as u32 + scale * 32;
                            let pair = load2(slot + (row * CODE_ROW + k * 8) as u32);
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
        if live && c >= 1 {
            // The chain of this lane's row over the products of stage `c - 1`.
            let base = smem + (((c - 1) & 1) * P_BYTES + lane * P_ROW) as u32;
            let mut q = 0;
            while q < CH / 4 {
                let p = load4(base + (q * 16) as u32);
                let mut e = 0;
                while e < 4 {
                    asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) f32::from_bits(p[e]), options(nomem, nostack));
                    e += 1;
                }
                q += 1;
            }
        }
        sync();
        c += 1;
    }
    if live { *output.add(output_row(first + lane, rows, pairs)) = sum; }
}
