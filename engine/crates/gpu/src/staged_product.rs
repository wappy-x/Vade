//! The staged BF16 row product: one launch gives the product of ONE matrix of BF16 values (two a word, row-major)
//! with one input row of FP32 values. Each output word is one chain over the columns in rising order from +0.0, each
//! product rounded by its own multiply before its add, no fused multiply-add: the words of `project_bf16_ordered`.
//!
//! The product is an entry of its own (`resident_launch::Kind::PlainStaged`, Rulings R6f and R6g): its kernel body
//! (`kernels/src/plain_staged.rs`) is in no entry that another caller launches. This file holds the launch geometry,
//! the admission and the host check of a launch. Pure functions, shared by the kernel body and the host, so one
//! predicate decides both. The kernel crate includes this file by path. It calls no function of another file: a call
//! from here must not change how another entry is built.
//!
//! A matrix takes this product when its columns are whole stages of 64 and it is inside the envelope below. A
//! matrix above the envelope runs in parts of whole blocks (`part_rows`), one launch a part, into one output: a row
//! is one chain, so the parts give the words of one launch.

/// Columns of one stage of the warp tile.
pub const STAGE: usize = 64;
/// The envelope: at most this many columns and this many weights in one launch (the envelope the device sweeps of
/// the warp tile cover, `plain_admission::in_envelope`; a host test compares the two).
pub const COLS_MAX: usize = 8192;
pub const ELEMENTS_MAX: usize = 1 << 25;
/// Threads of a block: one warp alone for a matrix of few rows, four warps above (`plain_admission::bf16_geometry`;
/// a host test compares the two).
pub const NARROW_THREADS: usize = 32;
pub const TILE_THREADS: usize = 128;
pub const NARROW_MAX_ROWS: usize = 512;

/// A shape the tile takes: whole stages inside the envelope.
#[inline(always)]
pub const fn shape_admitted(rows: usize, cols: usize) -> bool {
    cols >= STAGE && cols <= COLS_MAX && cols % STAGE == 0 && rows >= 1 && rows <= ELEMENTS_MAX / cols
}
/// Launch geometry: (blocks, threads of a block). A warp owns 32 rows.
#[inline(always)]
pub const fn geometry(rows: usize) -> (u32, u32) {
    if rows <= NARROW_MAX_ROWS { (rows.div_ceil(32) as u32, NARROW_THREADS as u32) } else { (rows.div_ceil(128) as u32, TILE_THREADS as u32) }
}
/// The rule of the kernel body: an admitted shape in its own geometry, with 16-byte aligned weights and input. The
/// body asks it before its first read. The host check below holds for every launch that is recorded, so it is true
/// in every launch.
#[inline(always)]
pub const fn admitted(rows: usize, cols: usize, weights: usize, input: usize, threads: usize) -> bool {
    shape_admitted(rows, cols) && threads == geometry(rows).1 as usize && (weights | input) & 15 == 0
}
/// Rows of one launch for a matrix of whole stages that is above the envelope: the most whole four-warp blocks that
/// stay inside it. A caller runs such a matrix in parts of this many rows. `rows` for a matrix that needs no parts,
/// and for a shape that the tile does not take.
#[cfg(not(target_arch = "nvptx64"))]
pub const fn part_rows(rows: usize, cols: usize) -> usize {
    if cols < STAGE || cols > COLS_MAX || cols % STAGE != 0 || rows <= ELEMENTS_MAX / cols { return rows; }
    let most = ELEMENTS_MAX / cols / TILE_THREADS * TILE_THREADS;
    if most == 0 { rows } else { most }
}

/// The bytes of one operand of a launch on the device.
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span { pub address: u64, pub bytes: u64 }
#[cfg(not(target_arch = "nvptx64"))]
impl Span {
    fn ends(self) -> bool { self.address.checked_add(self.bytes).is_some() }
    fn meets(self, other: Span) -> bool { self.address < other.address.saturating_add(other.bytes) && other.address < self.address.saturating_add(self.bytes) }
}
/// The host check of one launch. It runs before the launch is recorded and it names what it refuses.
/// It makes every read and every write of the kernel body stay inside its operand:
/// - the weights hold exactly `rows x cols` BF16 values and the columns are whole stages, so every 16-byte weight copy
///   (8 values of one row) and every prefetch address is inside the weights;
/// - the input holds exactly `cols` words, so every 16-byte input copy (4 words below `cols`) is inside it;
/// - the output holds exactly `rows` words and meets no operand, so the one store of a row lands inside it;
/// - the weights and the input are 16-byte aligned, so every 16-byte copy has an aligned source.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check(rows: usize, cols: usize, weights: Span, input: Span, output: Span) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 { return Err("staged product: the matrix has no rows or no columns"); }
    if cols > COLS_MAX || cols % STAGE != 0 { return Err("staged product: the columns are not whole stages of 64 up to 8192"); }
    if rows > ELEMENTS_MAX / cols { return Err("staged product: the matrix is above the envelope of the tile"); }
    if weights.bytes != rows as u64 * cols as u64 * 2 { return Err("staged product: the weights differ from rows x columns of BF16 values"); }
    if weights.address == 0 || input.address == 0 || output.address == 0 { return Err("staged product: an operand is null"); }
    if input.bytes != cols as u64 * 4 { return Err("staged product: the input differs from the columns of the matrix"); }
    if output.bytes != rows as u64 * 4 { return Err("staged product: the output differs from the rows of the matrix"); }
    if !weights.ends() || !input.ends() || !output.ends() { return Err("staged product: an operand ends above the address space"); }
    if weights.address % 16 != 0 { return Err("staged product: the weights are not 16-byte aligned"); }
    if input.address % 16 != 0 { return Err("staged product: the input is not 16-byte aligned"); }
    if output.address % 4 != 0 { return Err("staged product: the output is not word aligned"); }
    if output.meets(weights) || output.meets(input) { return Err("staged product: the output overlaps an operand"); }
    Ok(())
}

/// THE ROWS TILE (`resident_launch::Kind::PlainStagedRows`, kernel body `kernels/src/plain_staged_rows.rs`): one launch
/// gives the product of one BF16 matrix with `inputs` input rows, output `[inputs][rows]`. Each word is the word of
/// the row product on its input row: one chain over the columns in rising order from +0.0, each product rounded
/// before its add. A block of `ROWS_THREADS` threads owns 64 weight rows and 64 input rows; a thread owns 4 x 4
/// chains. An entry of its own, as the row product.
pub const ROWS_THREADS: usize = 256;
/// Weight rows and input rows of one block.
pub const ROWS_TILE: usize = 64;
/// The envelope of the rows tile (the shapes the device check runs): weight rows above the narrow matrices, whole
/// stages of columns, at most this many input rows.
pub const ROWS_MATRIX_ROWS_MIN: usize = 193;
pub const ROWS_MATRIX_ROWS_MAX: usize = 16384;
pub const ROWS_COLS_MAX: usize = 4096;
pub const ROWS_INPUTS_MAX: usize = 1024;
/// A shape the rows tile takes.
#[inline(always)]
pub const fn rows_shape_admitted(rows: usize, cols: usize, inputs: usize) -> bool {
    rows >= ROWS_MATRIX_ROWS_MIN && rows <= ROWS_MATRIX_ROWS_MAX && cols >= STAGE && cols <= ROWS_COLS_MAX && cols % STAGE == 0 && inputs >= 1 && inputs <= ROWS_INPUTS_MAX
}
/// Blocks of the launch: the tiles of input rows fastest, so neighbouring blocks share weight lines.
#[inline(always)]
pub const fn rows_blocks(rows: usize, inputs: usize) -> u32 { (rows.div_ceil(ROWS_TILE) * inputs.div_ceil(ROWS_TILE)) as u32 }
/// The rule of the kernel body of the rows tile: an admitted shape in blocks of `ROWS_THREADS` threads, with 16-byte
/// aligned weights and input. The body asks it before its first read; the host check holds for every recorded launch.
#[inline(always)]
pub const fn rows_admitted(rows: usize, cols: usize, inputs: usize, weights: usize, input: usize, threads: usize) -> bool {
    rows_shape_admitted(rows, cols, inputs) && threads == ROWS_THREADS && (weights | input) & 15 == 0
}
/// The host check of one launch of the rows tile. It runs before the launch is recorded and it names what it refuses.
/// `stride` is the words from one input row to the next in the output (`rows` for an output of its own; the rows of
/// the whole matrix when this launch is one part of a matrix and writes its columns in place).
/// It makes every read and every write of the kernel body stay inside its operand:
/// - the weights hold exactly `rows x cols` BF16 values and the columns are whole stages, so the two words (4 values
///   of one row, from a column that is a multiple of 4) that a thread reads are inside the weights;
/// - the input holds exactly `inputs x cols` words, so every 16-byte input copy (4 words of one input row) is inside it;
/// - the output holds exactly `(inputs - 1) x stride + rows` words, `stride` is at least `rows`, and the output meets
///   no operand, so the one store of an (input row, weight row) pair, at `input row x stride + weight row`, lands
///   inside it;
/// - the weights and the input are 16-byte aligned, so every word read and every 16-byte copy has an aligned source.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_rows(rows: usize, cols: usize, inputs: usize, stride: usize, weights: Span, input: Span, output: Span) -> Result<(), &'static str> {
    check_rows_at(rows, cols, inputs, stride, weights, input, Rows { count: inputs, table: None }, output, Rows { count: inputs, table: None })
}
/// The rows of an operand of an indexed launch of the rows tile: the rows the operand holds, and the host words of
/// the table of row numbers of the launch, or None for the rows in order. The device copy of a table is never the
/// caller's: the recorder stages these words with the command (`Backend::project_staged_rows_at_into`), so the replay
/// reads exactly the words this check saw.
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug)]
pub struct Rows<'t> { pub count: usize, pub table: Option<&'t [u32]> }
/// `check_rows` for a launch whose launch row `t` reads input row `inputs_at` table word `t` and writes output row
/// `outputs_at` table word `t` (the expert runs of a step). Beyond `check_rows` it holds:
/// - the input holds exactly `inputs_at.count x cols` words and the output `(outputs_at.count - 1) x stride + rows`
///   words; without a table the count is the launch rows;
/// - each table holds exactly `inputs` words, and every word is below its count (the body checks each index word
///   again and traps);
/// - the output rows of the launch are distinct, so every output word has one writer;
/// - every extent is computed with checked arithmetic: a sum or a product above 64 bits is a refusal.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "the shape, the stride, the three operands and the two tables")]
pub fn check_rows_at(rows: usize, cols: usize, inputs: usize, stride: usize, weights: Span, input: Span, inputs_at: Rows<'_>, output: Span, outputs_at: Rows<'_>) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 || inputs == 0 { return Err("staged rows product: no rows, no columns or no input rows"); }
    if cols > ROWS_COLS_MAX || cols % STAGE != 0 { return Err("staged rows product: the columns are not whole stages of 64 up to 4096"); }
    if !(ROWS_MATRIX_ROWS_MIN..=ROWS_MATRIX_ROWS_MAX).contains(&rows) { return Err("staged rows product: the rows of the matrix are outside the envelope of the tile"); }
    if inputs > ROWS_INPUTS_MAX { return Err("staged rows product: the input rows are above the envelope of the tile"); }
    if stride < rows || stride > u32::MAX as usize { return Err("staged rows product: the output stride is below the rows of the matrix or above the ABI"); }
    if weights.bytes != rows as u64 * cols as u64 * 2 { return Err("staged rows product: the weights differ from rows x columns of BF16 values"); }
    for at in [inputs_at, outputs_at] {
        if at.count == 0 || at.count > u32::MAX as usize || (at.table.is_none() && at.count != inputs) { return Err("staged rows product: the rows of an operand differ from the launch rows without a table, or are 0 or above the ABI"); }
    }
    let input_bytes = (inputs_at.count as u64).checked_mul(cols as u64 * 4);
    if input_bytes != Some(input.bytes) { return Err("staged rows product: the input differs from the input rows x the columns of the matrix"); }
    let output_bytes = (outputs_at.count as u64 - 1).checked_mul(stride as u64).and_then(|words| words.checked_add(rows as u64)).and_then(|words| words.checked_mul(4));
    if output_bytes != Some(output.bytes) { return Err("staged rows product: the output differs from the input rows x the stride"); }
    if weights.address == 0 || input.address == 0 || output.address == 0 { return Err("staged rows product: an operand is null"); }
    if !weights.ends() || !input.ends() || !output.ends() { return Err("staged rows product: an operand ends above the address space"); }
    if weights.address % 16 != 0 { return Err("staged rows product: the weights are not 16-byte aligned"); }
    if input.address % 16 != 0 { return Err("staged rows product: the input is not 16-byte aligned"); }
    if output.address % 4 != 0 { return Err("staged rows product: the output is not word aligned"); }
    if output.meets(weights) || output.meets(input) { return Err("staged rows product: the output overlaps an operand"); }
    for at in [inputs_at, outputs_at] {
        let Some(words) = at.table else { continue };
        if words.len() != inputs { return Err("staged rows product: a row table differs from the launch rows"); }
        if words.iter().any(|&row| row as usize >= at.count) { return Err("staged rows product: a row table names a row outside its operand"); }
    }
    if let Some(words) = outputs_at.table {
        let mut sorted = words.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) { return Err("staged rows product: the output table names a row twice"); }
    }
    Ok(())
}
/// Fewest input rows for which one launch of the rows tile beats one launch of the row product for each input row.
/// MEASURED on one Spark with this tile body (`loop-load-check --product-rows`): the tile takes 80 to 200
/// microseconds for 1 to 16 input rows of a matrix of 2.6 to 7 million weights (it meets the row product at 5 to 7
/// input rows) and 1.5 to 3.5 microseconds a row from 256.
#[cfg(not(target_arch = "nvptx64"))]
pub const ROWS_TILE_INPUTS_MIN: usize = 8;
/// True when the rows tile is the form for `inputs` input rows of this shape: cost only, the words are the same.
#[cfg(not(target_arch = "nvptx64"))]
pub fn rows_tile_ahead(rows: usize, cols: usize, inputs: usize) -> bool { rows_tile_part(rows, cols, inputs).is_some() }
/// Matrix rows of one launch of the rows tile when it is the form, else None. A matrix above `ROWS_MATRIX_ROWS_MAX`
/// runs in parts of whole 64-row tiles, as even as they can be, each inside the envelope; each part writes its
/// columns of the output in place (the stride of `check_rows`). A row is one chain, so the parts give the words of
/// one launch.
#[cfg(not(target_arch = "nvptx64"))]
pub fn rows_tile_part(rows: usize, cols: usize, inputs: usize) -> Option<usize> {
    if inputs < ROWS_TILE_INPUTS_MIN || rows < ROWS_MATRIX_ROWS_MIN || !rows_shape_admitted(ROWS_MATRIX_ROWS_MIN, cols, inputs) { return None; }
    if rows <= ROWS_MATRIX_ROWS_MAX { return Some(rows); }
    let parts = rows.div_ceil(ROWS_MATRIX_ROWS_MAX);
    let part = rows.div_ceil(parts).div_ceil(ROWS_TILE) * ROWS_TILE;
    let last = rows - (rows.div_ceil(part) - 1) * part;
    (part <= ROWS_MATRIX_ROWS_MAX && last >= ROWS_MATRIX_ROWS_MIN).then_some(part)
}

/// Most input rows of one launch of the general rows kernel (the prompt group cap, `plain_tile::MAX_TOKENS`).
#[cfg(not(target_arch = "nvptx64"))]
pub const GENERAL_INPUTS_MAX: usize = 1024;
/// Fewest input rows for which one launch of the general rows kernel beats one launch of the row product for each input row.
#[cfg(not(target_arch = "nvptx64"))]
pub const ROWS_INPUTS_MIN: usize = 16;
/// Most weights of a matrix inside the envelope of the warp tile for which that holds.
#[cfg(not(target_arch = "nvptx64"))]
pub const ROWS_WEIGHTS_MAX: usize = 1 << 21;
/// Fewest input rows for which one launch of the general rows kernel beats the parts of a matrix above the envelope.
#[cfg(not(target_arch = "nvptx64"))]
pub const ROWS_OVER_INPUTS_MIN: usize = 4;
/// The form of a BF16 matrix product for `inputs` input rows, with the kernels that exist: true = one launch of the
/// general rows kernel for all input rows (`prompt_ops::Op::ProjectRowsBf16`), false = one launch of the BF16 row
/// product for each input row. Both forms give the same words; this rule is cost only. MEASURED on one Spark
/// (`loop-load-check --product-rows`): the row product takes 15 to 30 microseconds for a matrix of 2.6 to 7 million
/// weights; the general rows kernel does about 1.05e11 multiply-adds a second after about 180 microseconds. A matrix
/// above the envelope of the warp tile reads its weights one time for all rows in the general rows kernel: it is
/// ahead from 4 input rows.
#[cfg(not(target_arch = "nvptx64"))]
pub fn rows_one_launch(rows: usize, cols: usize, inputs: usize) -> bool {
    if !(2..=GENERAL_INPUTS_MAX).contains(&inputs) || cols % 2 != 0 || rows == 0 || cols == 0 { return false; }
    if part_rows(rows, cols) < rows { return inputs >= ROWS_OVER_INPUTS_MIN; }
    inputs >= ROWS_INPUTS_MIN && rows <= ROWS_WEIGHTS_MAX / cols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plain_admission as plain;

    #[test]
    fn the_form_for_many_input_rows_follows_the_measured_cost() {
        assert_eq!((GENERAL_INPUTS_MAX, ROWS_INPUTS_MAX), (crate::plain_tile::MAX_TOKENS, crate::plain_tile::MAX_TOKENS));
        // Large matrices inside the envelope: one launch of the row product for each input row.
        for (rows, cols) in [(2304usize, 2304usize), (2304, 1152), (3072, 2304), (1024, 2304), (2304, 3072)] {
            for inputs in [1usize, 2, 16, 256, 1024, 1025] { assert!(!rows_one_launch(rows, cols, inputs), "{rows}x{cols} {inputs}"); }
        }
        // Small matrices: the general rows kernel from 16 input rows.
        for (rows, cols) in [(24usize, 2304usize), (128, 2304), (64, 4608), (2304, 64), (384, 2304)] {
            assert!(!rows_one_launch(rows, cols, 15) && rows_one_launch(rows, cols, 16) && rows_one_launch(rows, cols, 1024) && !rows_one_launch(rows, cols, 1025), "{rows}x{cols}");
        }
        // A matrix above the envelope: the general rows kernel from 4 input rows.
        assert!(!rows_one_launch(200_058, 2304, 3) && rows_one_launch(200_058, 2304, 4) && rows_one_launch(200_058, 2304, 1024));
        assert!(!rows_one_launch(0, 2304, 64) && !rows_one_launch(2304, 0, 64) && !rows_one_launch(24, 2305, 64) && !rows_one_launch(24, 2304, 1) && !rows_one_launch(24, 2304, 0));
    }

    #[test]
    fn the_envelope_and_the_geometry_are_the_ones_of_the_warp_tile() {
        assert_eq!((COLS_MAX, ELEMENTS_MAX, NARROW_THREADS, TILE_THREADS, NARROW_MAX_ROWS), (plain::COLS_MAX, plain::ELEMENTS_MAX, plain::NARROW_THREADS, plain::TILE_THREADS, plain::NARROW_MAX_ROWS));
        for rows in (0..=70_000usize).chain([200_058, 524_288, 1 << 25]) { assert_eq!(geometry(rows), plain::bf16_geometry(rows), "{rows}"); }
        for cols in (0..=8_400usize).step_by(4) {
            for rows in [0usize, 1, 31, 32, 512, 513, 2304, 4096, 4097, 14_563, 14_564, 200_058, 524_288, 524_289] {
                assert_eq!(shape_admitted(rows, cols), cols % STAGE == 0 && plain::in_envelope(rows, cols), "{rows}x{cols}");
            }
        }
    }
    #[test]
    fn the_kernel_rule_asks_the_shape_the_geometry_and_the_alignment() {
        for (rows, cols) in [(2304usize, 2304usize), (2304, 1152), (3072, 2304), (1024, 2304), (384, 2304), (24, 2304), (2304, 3072), (64, 4608), (2304, 64), (128, 2304), (14_464, 2304), (1, 64), (4096, 8192)] {
            let own = geometry(rows).1 as usize;
            for threads in [0usize, 1, 32, 64, 96, 128, 160, 256] { for weights in [256usize, 8, 264] { for input in [512usize, 4] {
                assert_eq!(admitted(rows, cols, weights, input, threads), threads == own && (weights | input) & 15 == 0, "{rows}x{cols} threads {threads}");
            } } }
        }
        for (rows, cols) in [(2304usize, 2296usize), (2304, 2312), (64, 96), (64, 32), (200_058, 2304), (14_564, 2304), (4, 8256), (0, 2304), (64, 0)] {
            assert!(!shape_admitted(rows, cols), "{rows}x{cols}");
            for threads in [32usize, 128] { assert!(!admitted(rows, cols, 256, 512, threads), "{rows}x{cols}"); }
        }
    }
    #[test]
    fn a_matrix_above_the_envelope_runs_in_parts_of_whole_blocks() {
        assert_eq!(part_rows(200_058, 2304), 14_464);
        assert!(shape_admitted(14_464, 2304) && shape_admitted(200_058 % 14_464, 2304) && geometry(14_464).1 as usize == TILE_THREADS);
        for (rows, cols) in [(2304usize, 2304usize), (14_563, 2304), (200_058, 2296), (200_058, 16_384), (5, 8192), (200_058, 32)] { assert_eq!(part_rows(rows, cols), rows, "{rows}x{cols}"); }
        assert_eq!(part_rows(1 << 20, 8192), 4096);
    }
    fn span(address: u64, bytes: u64) -> Span { Span { address, bytes } }
    /// The operands of a good launch of a 2304 x 2304 matrix.
    fn good() -> (usize, usize, Span, Span, Span) { (2304, 2304, span(1 << 20, 2304 * 2304 * 2), span(1 << 30, 2304 * 4), span(1 << 31, 2304 * 4)) }
    #[test]
    fn the_host_check_takes_a_good_launch() {
        let (rows, cols, weights, input, output) = good();
        assert_eq!(check(rows, cols, weights, input, output), Ok(()));
        assert_eq!(check(1, 64, span(16, 128), span(4096, 256), span(8192 + 4, 4)), Ok(()));
        assert_eq!(check(4096, 8192, span(1 << 40, 4096 * 8192 * 2), span(16, 8192 * 4), span(1 << 20, 4096 * 4)), Ok(()));
    }
    #[test]
    fn the_host_check_names_each_refusal() {
        let (rows, cols, weights, input, output) = good();
        let refused = |result: Result<(), &'static str>, text: &str| assert!(result.is_err_and(|error| error.contains(text)), "{result:?} lacks {text}");
        // The shape.
        refused(check(0, cols, weights, input, output), "no rows or no columns");
        refused(check(rows, 0, weights, input, output), "no rows or no columns");
        refused(check(rows, 2296, span(1 << 20, 2304 * 2296 * 2), span(1 << 30, 2296 * 4), output), "not whole stages");
        refused(check(4, 8256, span(1 << 20, 4 * 8256 * 2), span(1 << 30, 8256 * 4), span(1 << 31, 16)), "not whole stages");
        refused(check(14_564, cols, span(1 << 20, 14_564 * 2304 * 2), input, span(1 << 31, 14_564 * 4)), "above the envelope");
        // Each extent, one value short and one value long.
        for bytes in [weights.bytes - 2, weights.bytes + 2, 0] { refused(check(rows, cols, span(weights.address, bytes), input, output), "the weights differ"); }
        for bytes in [input.bytes - 4, input.bytes + 4, 0] { refused(check(rows, cols, weights, span(input.address, bytes), output), "the input differs"); }
        for bytes in [output.bytes - 4, output.bytes + 4, 0] { refused(check(rows, cols, weights, input, span(output.address, bytes)), "the output differs"); }
        // The end of each operand.
        refused(check(rows, cols, span(u64::MAX - 15, weights.bytes), input, output), "above the address space");
        refused(check(rows, cols, weights, span(u64::MAX - 15, input.bytes), output), "above the address space");
        refused(check(rows, cols, weights, input, span(u64::MAX - 3, output.bytes)), "above the address space");
        // Each alignment.
        for step in [1u64, 2, 4, 8] { refused(check(rows, cols, span(weights.address + step, weights.bytes), input, output), "weights are not 16-byte aligned"); }
        for step in [1u64, 4, 8] { refused(check(rows, cols, weights, span(input.address + step, input.bytes), output), "input is not 16-byte aligned"); }
        for step in [1u64, 2, 3] { refused(check(rows, cols, weights, input, span(output.address + step, output.bytes)), "output is not word aligned"); }
        // An output that meets an operand: its first word, its last word, and the whole operand inside the output.
        refused(check(rows, cols, weights, input, span(weights.address, output.bytes)), "overlaps an operand");
        refused(check(rows, cols, weights, input, span(weights.address + weights.bytes - 4, output.bytes)), "overlaps an operand");
        refused(check(rows, cols, weights, input, span(input.address, output.bytes)), "overlaps an operand");
        refused(check(rows, cols, weights, span(output.address + 16, input.bytes), output), "overlaps an operand");
        // An output that ends where an operand starts is no overlap.
        assert_eq!(check(rows, cols, weights, input, span(weights.address - output.bytes, output.bytes)), Ok(()));
        assert_eq!(check(rows, cols, weights, input, span(input.address + input.bytes, output.bytes)), Ok(()));
    }
    #[test]
    fn the_rows_tile_takes_large_matrices_of_whole_stages() {
        assert_eq!((ROWS_THREADS as u32, ROWS_TILE, ROWS_MATRIX_ROWS_MAX), (crate::plain_tile::THREADS, 64, crate::plain_tile::MAX_ROWS));
        assert_eq!(ROWS_MATRIX_ROWS_MIN, crate::plain_tile::NARROW_MAX_ROWS + 1);
        for (rows, cols) in [(2304usize, 2304usize), (2304, 1152), (3072, 2304), (1024, 2304), (2304, 3072), (384, 2304), (2304, 64), (193, 64), (16384, 4096)] {
            for inputs in [1usize, 4, 16, 64, 256, 1024] {
                assert!(rows_shape_admitted(rows, cols, inputs), "{rows}x{cols} {inputs}");
                assert_eq!(Some(rows_blocks(rows, inputs)), crate::plain_tile::blocks(rows, inputs), "{rows}x{cols} {inputs}");
                for threads in [0usize, 32, 128, 256, 512] { for weights in [256usize, 8, 264] { for input in [512usize, 4] {
                    assert_eq!(rows_admitted(rows, cols, inputs, weights, input, threads), threads == ROWS_THREADS && (weights | input) & 15 == 0, "{rows}x{cols} {inputs} threads {threads}");
                } } }
                assert_eq!(rows_tile_ahead(rows, cols, inputs), inputs >= 8, "{rows}x{cols} {inputs}");
            }
            assert!(!rows_shape_admitted(rows, cols, 0) && !rows_shape_admitted(rows, cols, 1025), "{rows}x{cols}");
        }
        // Not on the tile: the rows of a narrow matrix, columns that are no whole stages, columns past the tile, too many rows.
        for (rows, cols) in [(192usize, 2304usize), (24, 2304), (64, 4608), (2304, 2296), (2304, 32), (2304, 4160), (2304, 8192), (16385, 2304), (200_058, 2304), (0, 2304), (2304, 0)] {
            for inputs in [1usize, 16, 1024] {
                assert!(!rows_shape_admitted(rows, cols, inputs) && !rows_admitted(rows, cols, inputs, 256, 512, ROWS_THREADS), "{rows}x{cols} {inputs}");
                // Only a matrix above the rows of the tile takes it, in parts.
                assert_eq!(rows_tile_ahead(rows, cols, inputs), rows > ROWS_MATRIX_ROWS_MAX && cols == 2304 && inputs >= 8, "{rows}x{cols} {inputs}");
            }
        }
    }
    #[test]
    fn a_matrix_above_the_rows_of_the_tile_runs_in_even_parts_of_whole_tiles() {
        assert_eq!(rows_tile_part(2304, 2304, 64), Some(2304));
        assert_eq!(rows_tile_part(16_384, 2304, 8), Some(16_384));
        assert_eq!(rows_tile_part(200_058, 2304, 1024), Some(15_424));
        for (rows, cols) in [(200_058usize, 2304usize), (16_385, 64), (32_768, 4096), (100_000, 1152), (16_384 * 7 + 193, 128)] {
            for inputs in [8usize, 64, 1024] {
                let part = rows_tile_part(rows, cols, inputs).expect("parts");
                assert!(part % ROWS_TILE == 0 && part <= ROWS_MATRIX_ROWS_MAX && rows_shape_admitted(part, cols, inputs), "{rows}x{cols}");
                let last = rows - (rows.div_ceil(part) - 1) * part;
                assert!(rows_shape_admitted(last, cols, inputs), "{rows}x{cols}: last part {last}");
            }
            assert_eq!(rows_tile_part(rows, cols, 7), None);
        }
        for (rows, cols, inputs) in [(192usize, 2304usize, 64usize), (200_058, 4160, 64), (200_058, 2296, 64), (2304, 2304, 1025)] { assert_eq!(rows_tile_part(rows, cols, inputs), None, "{rows}x{cols} {inputs}"); }
    }
    /// The operands of a good launch of the rows tile: a 2304 x 2304 matrix and 64 input rows.
    fn good_rows() -> (usize, usize, usize, Span, Span, Span) { (2304, 2304, 64, span(1 << 20, 2304 * 2304 * 2), span(1 << 30, 64 * 2304 * 4), span(1 << 31, 64 * 2304 * 4)) }
    #[test]
    fn the_host_check_of_the_rows_tile_takes_a_good_launch() {
        let (rows, cols, inputs, weights, input, output) = good_rows();
        assert_eq!(check_rows(rows, cols, inputs, rows, weights, input, output), Ok(()));
        assert_eq!(check_rows(193, 64, 1, 193, span(16, 193 * 64 * 2), span(1 << 20, 256), span((1 << 21) + 4, 193 * 4)), Ok(()));
        assert_eq!(check_rows(16384, 4096, 1024, 16384, span(1 << 40, 16384 * 4096 * 2), span(1 << 30, 1024 * 4096 * 4), span(1 << 20, 1024 * 16384 * 4)), Ok(()));
    }
    #[test]
    fn the_host_check_of_the_rows_tile_names_each_refusal() {
        let (rows, cols, inputs, weights, input, output) = good_rows();
        let refused = |result: Result<(), &'static str>, text: &str| assert!(result.is_err_and(|error| error.contains(text)), "{result:?} lacks {text}");
        // The shape.
        refused(check_rows(0, cols, inputs, 0, weights, input, output), "no rows, no columns or no input rows");
        refused(check_rows(rows, 0, inputs, rows, weights, input, output), "no rows, no columns or no input rows");
        refused(check_rows(rows, cols, 0, rows, weights, input, output), "no rows, no columns or no input rows");
        refused(check_rows(rows, 2296, inputs, rows, span(1 << 20, 2304 * 2296 * 2), span(1 << 30, 64 * 2296 * 4), output), "not whole stages");
        refused(check_rows(rows, 4160, inputs, rows, span(1 << 20, 2304 * 4160 * 2), span(1 << 30, 64 * 4160 * 4), output), "not whole stages");
        refused(check_rows(192, cols, inputs, 192, span(1 << 20, 192 * 2304 * 2), input, span(1 << 31, 64 * 192 * 4)), "outside the envelope");
        refused(check_rows(16385, cols, inputs, 16385, span(1 << 20, 16385 * 2304 * 2), input, span(1 << 31, 64 * 16385 * 4)), "outside the envelope");
        refused(check_rows(rows, cols, 1025, rows, weights, span(1 << 30, 1025 * 2304 * 4), span(1 << 31, 1025 * 2304 * 4)), "input rows are above the envelope");
        // Each extent, one value short and one value long.
        for bytes in [weights.bytes - 2, weights.bytes + 2, 0] { refused(check_rows(rows, cols, inputs, rows, span(weights.address, bytes), input, output), "the weights differ"); }
        for bytes in [input.bytes - 4, input.bytes + 4, cols as u64 * 4, 0] { refused(check_rows(rows, cols, inputs, rows, weights, span(input.address, bytes), output), "the input differs"); }
        for bytes in [output.bytes - 4, output.bytes + 4, rows as u64 * 4, 0] { refused(check_rows(rows, cols, inputs, rows, weights, input, span(output.address, bytes)), "the output differs"); }
        // The stride: below the rows, and a part of a larger output with its own extent.
        refused(check_rows(rows, cols, inputs, rows - 1, weights, input, output), "stride");
        assert_eq!(check_rows(rows, cols, inputs, 3 * rows, weights, input, span(output.address, ((inputs as u64 - 1) * 3 * rows as u64 + rows as u64) * 4)), Ok(()));
        refused(check_rows(rows, cols, inputs, 3 * rows, weights, input, output), "the output differs");
        // The end of each operand.
        refused(check_rows(rows, cols, inputs, rows, span(u64::MAX - 15, weights.bytes), input, output), "above the address space");
        refused(check_rows(rows, cols, inputs, rows, weights, span(u64::MAX - 15, input.bytes), output), "above the address space");
        refused(check_rows(rows, cols, inputs, rows, weights, input, span(u64::MAX - 3, output.bytes)), "above the address space");
        // Each alignment.
        for step in [1u64, 2, 4, 8] { refused(check_rows(rows, cols, inputs, rows, span(weights.address + step, weights.bytes), input, output), "weights are not 16-byte aligned"); }
        for step in [1u64, 4, 8] { refused(check_rows(rows, cols, inputs, rows, weights, span(input.address + step, input.bytes), output), "input is not 16-byte aligned"); }
        for step in [1u64, 2, 3] { refused(check_rows(rows, cols, inputs, rows, weights, input, span(output.address + step, output.bytes)), "output is not word aligned"); }
        // An output that meets an operand.
        refused(check_rows(rows, cols, inputs, rows, weights, input, span(weights.address, output.bytes)), "overlaps an operand");
        refused(check_rows(rows, cols, inputs, rows, weights, input, span(weights.address + weights.bytes - 4, output.bytes)), "overlaps an operand");
        refused(check_rows(rows, cols, inputs, rows, weights, input, span(input.address + input.bytes - 4, output.bytes)), "overlaps an operand");
        refused(check_rows(rows, cols, inputs, rows, weights, span(output.address + 16, input.bytes), output), "overlaps an operand");
        // An output that ends where an operand starts is no overlap.
        assert_eq!(check_rows(rows, cols, inputs, rows, weights, input, span(input.address + input.bytes, output.bytes)), Ok(()));
    }
    /// A good indexed launch: 16 launch rows of a 64-row input, in reverse, to rows 3, 7, 11 .. of a 64-row output.
    fn indexed() -> (Vec<u32>, Vec<u32>) { ((0..16u32).map(|t| 63 - 2 * t).collect(), (0..16u32).map(|t| 4 * t + 3).collect()) }
    #[test]
    fn the_host_check_of_index_rows_takes_a_good_launch_and_names_each_refusal() {
        let (_, _, _, _, _, output) = good_rows();
        let (ins, outs) = indexed();
        fn at(count: usize, words: &[u32]) -> Rows<'_> { Rows { count, table: Some(words) } }
        fn check(inputs_at: Rows<'_>, outputs_at: Rows<'_>, output: Span) -> Result<(), &'static str> {
            let (rows, cols, _, weights, input, _) = good_rows();
            check_rows_at(rows, cols, 16, rows, weights, input, inputs_at, output, outputs_at)
        }
        let refused = |result: Result<(), &'static str>, text: &str| assert!(result.is_err_and(|error| error.contains(text)), "{result:?} lacks {text}");
        assert_eq!(check(at(64, &ins), at(64, &outs), output), Ok(()));
        assert_eq!(check(at(64, &ins), Rows { count: 16, table: None }, span(output.address, 16 * 2304 * 4)), Ok(()));
        refused(check(Rows { count: 16, table: None }, at(64, &outs), output), "the input differs");
        // Without a table the count is the launch rows; a count of 0 or above the ABI.
        refused(check(Rows { count: 64, table: None }, at(64, &outs), output), "differ from the launch rows without a table");
        refused(check(at(0, &ins), at(64, &outs), output), "differ from the launch rows without a table");
        refused(check(at(1 << 33, &ins), at(64, &outs), output), "differ from the launch rows without a table");
        // The extents follow the counts.
        refused(check(at(63, &[0; 16]), at(64, &outs), output), "the input differs");
        refused(check(at(64, &ins), at(65, &outs), output), "the output differs");
        // A table of the wrong length.
        refused(check(at(64, &ins[..15]), at(64, &outs), output), "a row table differs");
        refused(check(at(64, &ins), at(64, &[outs.as_slice(), &[1]].concat()), output), "a row table differs");
        // An index at the count, of each table.
        let mut bad = ins.clone();
        bad[7] = 64;
        refused(check(at(64, &bad), at(64, &outs), output), "names a row outside its operand");
        let mut bad = outs.clone();
        bad[15] = 64;
        refused(check(at(64, &ins), at(64, &bad), output), "names a row outside its operand");
        // An output row named twice (two writers of one word); an input row named twice is good (two launch rows
        // may read one row).
        let mut twice = outs.clone();
        twice[9] = twice[2];
        refused(check(at(64, &ins), at(64, &twice), output), "names a row twice");
        assert_eq!(check(at(64, &[5; 16]), at(64, &outs), output), Ok(()));
    }
    /// Review REVPROD finding 2: extents that pass 64 bits are refused, never wrapped; and the other bounds.
    #[test]
    fn the_host_check_of_the_rows_tile_refuses_extents_above_64_bits_null_operands_and_a_wide_stride() {
        let (rows, cols, _, weights, input, output) = good_rows();
        let refused = |result: Result<(), &'static str>, text: &str| assert!(result.is_err_and(|error| error.contains(text)), "{result:?} lacks {text}");
        let one = span(1 << 30, 64 * 4);
        // 2^31 x 2^31 + 256 words: above 2^62 words, so above 2^64 bytes; it must not wrap to a small extent.
        let wide = (1usize << 31) + 1;
        for bytes in [1024u64, 0, u64::MAX] {
            refused(check_rows_at(256, 64, 1, 1 << 31, span(1 << 20, 256 * 64 * 2), one, Rows { count: 1, table: None }, span(1 << 40, bytes), Rows { count: wide, table: Some(&[1]) }), "the output differs");
        }
        refused(check_rows(rows, cols, 64, u32::MAX as usize + 1, weights, input, output), "stride");
        for (w, x, o) in [(0u64, input.address, output.address), (weights.address, 0, output.address), (weights.address, input.address, 0)] {
            refused(check_rows(rows, cols, 64, rows, span(w, weights.bytes), span(x, input.bytes), span(o, output.bytes)), "an operand is null");
        }
        let (rows, cols, weights, input, output) = good();
        for (w, x, o) in [(0u64, input.address, output.address), (weights.address, 0, output.address), (weights.address, input.address, 0)] {
            refused(check(rows, cols, span(w, weights.bytes), span(x, input.bytes), span(o, output.bytes)), "an operand is null");
        }
    }
}
