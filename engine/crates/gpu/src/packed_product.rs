//! The packed row product: one launch reads ONE matrix in its stored packed form (4-bit or 8-bit codes and their block
//! scales) and gives the words of `project_bf16_ordered` on the same matrix widened to BF16: each output is one chain
//! over the columns in rising order from +0.0, each product rounded before its add, the sum not rounded.
//!
//! This file is the layout of the unit, the launch geometry, the admission and the host check of a launch. Pure
//! functions, shared by the kernel body (`kernels-inkling/src/plain_packed.rs`) and the host, so one predicate decides both.
//!
//! The unit of one matrix, in bytes: the codes, then the block scales, then the weight table. Each part starts on a
//! multiple of 16 bytes and the unit ends on one. The codes and the scales are stored in tiles (`tiled`): for each
//! block of 32 rows, the bytes of one stage (64 columns) of its 32 rows, then the next stage, so that a block reads
//! each stage as one run. The rows are rounded up to a whole block and the columns to `whole(wide)`; the added bytes
//! are zero.
//! - 4-bit form: codes `rows x cols / 2` (two a byte; the low half of a byte is the even column), one scale byte for
//!   16 columns of a row, and a table `[scale byte][code]` of 256 x 16 BF16 patterns. The host makes the table with
//!   the widen arithmetic of the BF16 form, so a weight is one table read.
//! - 8-bit form (`WIDE`): codes `rows x cols` (one value byte a weight), one scale byte for 32 columns of a row (a
//!   power of two, `2 ^ (scale - 127)`), and a table of the 256 FP32 values of a value byte. A weight is its value
//!   times its power, rounded to BF16 (`round_bf16`).
//!
//! Row order: with `PAIRS` stored row `2i` is output row `i` and stored row `2i + 1` is output row `rows / 2 + i`
//! (gate rows first, then up rows). Without it stored row `i` is output row `i`.

/// In the column word of a launch: 8-bit codes. Else 4-bit codes.
pub const WIDE: u32 = 1 << 30;
/// In the column word of a launch: gate rows and up rows are mixed in storage (see the row order above).
pub const PAIRS: u32 = 1 << 29;
/// The columns in the column word of a launch.
pub const COLS: u32 = 0xffff;
/// In payload word 16 of a routed packed launch (`PLAIN_PACKED_TC_ROWS`): store each output as `rn(output)` in BF16,
/// two a word (the even column in the low half), not as FP32. Only an ungated launch with no `PAIRS` and even rows.
/// `RouteFold` with `FOLD_HALF` reads these words: each down word is `rn(down)` there too, so the fold is the same.
pub const HALF_OUT: u64 = 1;
/// In payload word 17 of a routed packed launch (`PLAIN_PACKED_TC_ROWS`): the input rows are BF16 values, two a word
/// (the even column in the low half). The launch reads no other input form: the host rounds an FP32 input with
/// `cvt.rn.bf16x2.f32` first, the rounding the MMA input had in the launch before.
pub const HALF_IN: u64 = 1;
/// Stored rows of one block: the lanes of its chain warp.
pub const BLOCK_ROWS: usize = 32;
/// Threads of a block: one warp runs the 32 chains, four warps make the rounded products one stage ahead.
pub const THREADS: usize = 160;
/// Routed product threads. Each warp owns 16 stored rows.
pub const PICKED_THREADS: usize = 128;
/// Columns of one stage.
pub const STAGE: usize = 64;
/// The envelope of the tile: at most this many columns and this many weights. The device check ran the two matrix
/// shapes of the checkpoint it was built for and a sweep of small shapes; a larger matrix is refused.
pub const COLS_MAX: usize = 8192;
pub const ELEMENTS_MAX: usize = 1 << 25;
/// Bytes of the weight table of a unit: 256 x 16 BF16 patterns, or 256 FP32 values.
pub const NARROW_TABLE: usize = 256 * 16 * 2;
pub const WIDE_TABLE: usize = 256 * 4;
/// Two qNaN-tagged words at the unused E4M3 NaN scale row (`0x7f`) of a 4-bit table
/// marking a unit whose codes and scales are stored in MMA-lane panel order (`decode_panels::pack_expert`).
pub const NVFP4_PANEL_MAGIC_LO: u32 = 0x7fc0_4e50;
pub const NVFP4_PANEL_MAGIC_HI: u32 = 0x7fc0_414e;
/// Two qNaN-tagged words + expert `factor` FP32 bits at the unused E4M3 `-NaN` scale row (`0xff`) of a 4-bit table,
/// enabling bit-exact register-only SM121a NVFP4 decode (`cvt.rn.f16x2.{e4m3x2,e2m1x2}`) with zero 8 KB SMEM LUT.
pub const NVFP4_FACTOR_MAGIC_LO: u32 = 0x7fc0_4641;
pub const NVFP4_FACTOR_MAGIC_HI: u32 = 0x7fc0_4354;

/// The one-row input must fit below the packed tile's shared lookup table.
pub const PANEL_INPUT_COLS: usize = 3328;
/// Payload words of a picked command: word 17 marks a single decoded token (`one_row`), word 18 is zero or the one-row
/// expert panel copy of the stack (`decode_panels::pack_expert`). Words 14 to 16 hold the repeat and the strided route.
pub const PICKED_ONE_ROW_WORD: usize = 17;
pub const fn pad16(bytes: usize) -> usize { (bytes + 15) & !15 }
/// Code bytes of one stored row.
pub const fn code_stride(cols: usize, wide: bool) -> usize { if wide { cols } else { cols / 2 } }
/// Scale bytes of one stored row.
pub const fn scale_stride(cols: usize, wide: bool) -> usize { if wide { cols / 32 } else { cols / 16 } }
pub const fn table_bytes(wide: bool) -> usize { if wide { WIDE_TABLE } else { NARROW_TABLE } }
/// Columns that the column count must be a multiple of. The 8-bit form copies the scale bytes of two stages at once.
pub const fn whole(wide: bool) -> usize { if wide { 2 * STAGE } else { STAGE } }

/// Stages of a matrix.
pub const fn stages(cols: usize) -> usize { cols / STAGE }
/// Code bytes of one stored row in a stage.
pub const fn code_stage(wide: bool) -> usize { if wide { STAGE } else { STAGE / 2 } }

// The copies of the kernel body, each as the offset of its first byte (or word) from the start of its part. The body
// takes every source address from these functions, and the host test `every_copy_of_the_body_...` holds each copy
// inside its part for a sweep of shapes. So a change of the layout cannot move a read without a failing test.
/// The 16-byte code copy `part` (below `code_stage / 16`) of stage `stage` of stored row `row`: its first byte in the
/// codes.
#[inline(always)]
pub const fn code_copy(row: usize, stage: usize, part: usize, cols: usize, wide: bool) -> usize { tiled(row, stage, code_stage(wide), stages(cols)) + part * 16 }
/// The 4-byte scale copy of stage `stage` of stored row `row`: its first byte in the scales. The 8-bit form has two
/// scale bytes a stage, and the copy takes the four bytes of an even stage and the odd stage after it.
#[inline(always)]
pub const fn scale_copy(row: usize, stage: usize, cols: usize, wide: bool) -> usize { if wide { tiled(row, stage / 2, 4, stages(cols) / 2) } else { tiled(row, stage, 4, stages(cols)) } }
/// The 16-byte input copy `part` (below `STAGE / 4`) of stage `stage`: its first word in the input.
#[inline(always)]
pub const fn input_copy(stage: usize, part: usize) -> usize { stage * STAGE + part * 4 }
/// The 16-byte copies of the weight table; copy `t` starts at byte `t * 16` of the table.
pub const fn table_copies(wide: bool) -> usize { table_bytes(wide) / 16 }

/// The place of piece `piece` (of `width` bytes) of stored row `row` in a part that is stored in tiles: the codes
/// and the scales of a unit keep, for each block of 32 rows, the pieces of its rows one stage (or one scale copy)
/// after the other, so the copies of one stage of a block are one run of `32 * width` bytes. A row has `per_row`
/// pieces. The same order as `loop_packed_unit::UnitShape` on the host.
#[inline(always)]
pub const fn tiled(row: usize, piece: usize, width: usize, per_row: usize) -> usize { ((row / BLOCK_ROWS) * per_row + piece) * BLOCK_ROWS * width + (row % BLOCK_ROWS) * width }
/// Stored rows of a part in tiles: the rows rounded up to a whole block of 32. The rows above `rows` are zero.
pub const fn tiled_rows(rows: usize) -> usize { rows.div_ceil(BLOCK_ROWS) * BLOCK_ROWS }
/// Stored columns of a part in tiles: the columns rounded up to `whole(wide)`. Equal to `cols` for an admitted shape;
/// the layout of another shape keeps the same bytes as the host (`loop_packed_unit`), for the load plan.
pub const fn tiled_cols(cols: usize, wide: bool) -> usize { cols.div_ceil(whole(wide)) * whole(wide) }
/// Byte offsets inside a unit. The codes start at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout { pub scales: usize, pub table: usize, pub bytes: usize }
pub const fn layout(rows: usize, cols: usize, wide: bool) -> Layout {
    let scales = pad16(tiled_rows(rows) * code_stride(tiled_cols(cols, wide), wide));
    let table = scales + pad16(tiled_rows(rows) * scale_stride(tiled_cols(cols, wide), wide));
    Layout { scales, table, bytes: table + pad16(table_bytes(wide)) }
}
/// A shape the tile takes: whole stages inside the envelope, and an even row count for mixed rows.
pub const fn shape_admitted(rows: usize, cols: usize, wide: bool, pairs: bool) -> bool {
    cols >= whole(wide) && cols <= COLS_MAX && cols % whole(wide) == 0 && rows >= 1 && rows <= ELEMENTS_MAX / cols && (!pairs || rows % 2 == 0)
}
/// What the kernel body asks before its first read: an admitted shape, the block size of the tile, and 16-byte
/// aligned operands. The host check below asks the same and more, so a recorded launch always passes here.
pub const fn admitted(rows: usize, cols: usize, wide: bool, pairs: bool, unit: usize, input: usize, threads: usize) -> bool {
    threads == THREADS && shape_admitted(rows, cols, wide, pairs) && (unit | input) & 15 == 0
}
/// The most matrices of one grouped launch (`resident_launch_plain_packed_group`): one a block row of the grid. The
/// command holds their unit addresses after its six words.
pub const GROUP_MAX: usize = 32;
/// A picked command holds a pointer to the ID vector. Its payload size is independent of the pick count.
pub const PICKS_MAX: usize = 128;
/// The words of a grouped command before the unit addresses: input, output, rows, column word, matrices, input
/// words a matrix (0: one input for all), then two words unused, so the addresses start at word 8.
pub const GROUP_UNITS_AT: usize = 8;
/// Launch geometry (blocks, threads) for a matrix of `rows` stored rows.
pub const fn geometry(rows: usize) -> (u32, u32) { (rows.div_ceil(BLOCK_ROWS) as u32, THREADS as u32) }
/// Blocks of a launch that run at once on one multiprocessor: two of `BLOCK_BYTES` shared bytes each.
pub const BLOCKS_A_MULTIPROCESSOR: usize = 2;
/// The width of the grid of a launch of `picks` matrices of `rows` stored rows on a device of `multiprocessors`:
/// the fewest blocks a matrix that still fill every place of one wave, at least one and at most one a row block.
/// Each block then runs its row blocks in turn and copies the weight table once (`plain_packed::tile`).
pub const fn spread(rows: usize, picks: usize, multiprocessors: usize) -> u32 {
    let row_blocks = rows.div_ceil(BLOCK_ROWS);
    let places = multiprocessors * BLOCKS_A_MULTIPROCESSOR;
    let wanted = if picks == 0 { row_blocks } else { places.div_ceil(picks) };
    (if wanted < 1 { 1 } else if wanted > row_blocks { row_blocks } else { wanted }) as u32
}
/// The column word of a launch.
pub const fn cols_word(cols: usize, wide: bool, pairs: bool) -> u32 { cols as u32 | if wide { WIDE } else { 0 } | if pairs { PAIRS } else { 0 } }
/// The output row of a stored row.
pub const fn output_row(stored: usize, rows: usize, pairs: bool) -> usize { if pairs { stored / 2 + (stored & 1) * (rows / 2) } else { stored } }

/// Round a finite FP32 value to BF16, nearest even, and give it as FP32 (the low half word is zero). It is the
/// rounding of the widen step of the BF16 form. The host refuses a weight that is not finite.
#[inline(always)]
pub fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}
/// `2 ^ (scale - 127)` for a scale byte of the 8-bit form: the byte is the exponent field. Byte 0 is below the
/// normal range. Byte 255 is not finite; the host refuses a matrix that uses it.
#[inline(always)]
pub fn power(scale: u32) -> f32 { f32::from_bits(if scale == 0 { 1 << 22 } else { scale << 23 }) }

// THE PACKED ROWS TILE (`resident_launch::Kind::PlainPackedRows`, kernel body `kernels-inkling/src/plain_packed_rows.rs`):
// the rows tile of the staged BF16 product (`staged_product`, THE ROWS TILE) in its grouped form, with each run's
// matrix read from its packed unit. A block widens the weights of its stage from the codes, the scale bytes and the
// weight table of the unit to the BF16 value of the widened form (as `plain_packed.rs` does), then runs the chains of
// the BF16 rows tile on them. So each word is the word of the rows tile on the widened matrix, gate rows first.
/// Shared bytes of the weight table of a packed rows block: the larger table (the 4-bit form).
pub const ROWS_TABLE_BYTES: usize = NARROW_TABLE;
/// Offsets within the code and scale sections for a group of four consecutive weights.
#[inline(always)]
pub const fn rows_quad(row: usize, column: usize, cols: usize, wide: bool) -> (usize, usize) {
    let stage = column / STAGE;
    let lane = column % STAGE;
    let code = code_copy(row, stage, 0, cols, wide);
    let scale = scale_copy(row, stage, cols, wide);
    if wide { (code + lane, scale + column / 32 % 4) }
    else { (code + lane / 2, scale + lane / 16) }
}

/// The rule of the kernel body of the packed rows tile: the rule of the BF16 rows tile (`staged_product::rows_admitted`,
/// with the unit in place of the weights), whole stages of the packed form, and an even row count for mixed rows.
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "the shape, the form, the launch rows, the two addresses and the block size")]
pub const fn rows_admitted(rows: usize, cols: usize, wide: bool, pairs: bool, inputs: usize, unit: usize, input: usize, threads: usize) -> bool {
    crate::staged_product::rows_admitted(rows, cols, inputs, unit, input, threads) && cols % whole(wide) == 0 && (!pairs || rows % 2 == 0)
}

/// One operand of a launch: its device address and its bytes.
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span { pub address: u64, pub bytes: u64 }
#[cfg(not(target_arch = "nvptx64"))]
impl Span {
    fn end(self) -> Option<u64> { self.address.checked_add(self.bytes) }
    fn meets(self, other: Span) -> bool { self.address < other.address.saturating_add(other.bytes) && other.address < self.address.saturating_add(self.bytes) }
}
/// The host check of one launch. It runs before the launch is recorded and it names what it refuses.
/// It makes every read and every write of the kernel body stay inside its operand:
/// - the unit holds exactly the codes, the scales and the table of `rows x cols` weights in this form, so every code,
///   scale and table read is inside the unit;
/// - the input holds exactly `cols` words and the columns are whole stages, so every input read is inside it;
/// - the output holds exactly `rows` words and meets no operand, so the one store a stored row lands inside it;
/// - the unit and the input are 16-byte aligned, so every 16-byte copy has an aligned source.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check(rows: usize, cols: usize, wide: bool, pairs: bool, unit: Span, input: Span, output: Span) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 { return Err("packed product: the matrix has no rows or no columns"); }
    if cols > COLS_MAX || cols % whole(wide) != 0 { return Err("packed product: the columns are not whole stages (64 columns, 128 with 8-bit codes) up to 8192"); }
    if rows > ELEMENTS_MAX / cols { return Err("packed product: the matrix is above the envelope of the tile"); }
    if pairs && rows % 2 != 0 { return Err("packed product: mixed gate rows and up rows need an even row count"); }
    if unit.bytes != layout(rows, cols, wide).bytes as u64 { return Err("packed product: the unit differs from the bytes of the codes, scales and table of this shape"); }
    if input.bytes != cols as u64 * 4 { return Err("packed product: the input differs from the columns of the matrix"); }
    if output.bytes != rows as u64 * 4 { return Err("packed product: the output differs from the rows of the matrix"); }
    if unit.end().is_none() || input.end().is_none() || output.end().is_none() { return Err("packed product: an operand ends above the address space"); }
    // No device vector has address 0; a null operand is refused (review REVPACK finding 3).
    if unit.address == 0 || input.address == 0 || output.address == 0 { return Err("packed product: an operand has address 0"); }
    if unit.address % 16 != 0 { return Err("packed product: the unit is not 16-byte aligned"); }
    if input.address % 16 != 0 { return Err("packed product: the input is not 16-byte aligned"); }
    if output.address % 4 != 0 { return Err("packed product: the output is not word aligned"); }
    if output.meets(unit) || output.meets(input) { return Err("packed product: the output overlaps an operand"); }
    Ok(())
}

/// The host check of one grouped launch: `units.len()` matrices of one shape, each its own unit; one input for all
/// (`shared`) or one input a matrix, back to back; the outputs back to back, `rows` words a matrix. Each matrix
/// passes `check` with its own unit, its own input and its own output; then the whole output meets no unit and not the
/// whole input. Block row `e` of the launch reads only unit `e`, input `e` and writes only output `e`.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_group(rows: usize, cols: usize, wide: bool, pairs: bool, units: &[Span], input: Span, shared: bool, output: Span) -> Result<(), &'static str> {
    if units.is_empty() || units.len() > GROUP_MAX { return Err("packed group: the matrices of a group are 1 to 32"); }
    check_units(rows,cols,wide,pairs,units,input,shared,output)
}
#[cfg(not(target_arch = "nvptx64"))]
fn check_units(rows:usize,cols:usize,wide:bool,pairs:bool,units:&[Span],input:Span,shared:bool,output:Span)->Result<(),&'static str> {
    let picks = units.len() as u64;
    // Checked products: a shape far above the envelope is refused here, not by an overflow (review REVPACK finding 2).
    let bytes = |count: usize, times: u64| (count as u64).checked_mul(4).and_then(|b| b.checked_mul(times));
    let (Some(input_bytes), Some(output_bytes)) = (bytes(cols, 1), bytes(rows, 1)) else { return Err("packed group: the shape is above the envelope of the tile"); };
    let (Some(all_inputs), Some(all_outputs)) = (bytes(cols, if shared { 1 } else { picks }), bytes(rows, picks)) else { return Err("packed group: the shape is above the envelope of the tile"); };
    if input.bytes != all_inputs { return Err("packed group: the input differs from the columns of the matrices"); }
    if output.bytes != all_outputs { return Err("packed group: the output differs from the rows of the matrices"); }
    if input.end().is_none() || output.end().is_none() { return Err("packed group: an operand ends above the address space"); }
    for (e, &unit) in units.iter().enumerate() {
        let e = e as u64;
        let one = Span { address: if shared { input.address } else { input.address + e * input_bytes }, bytes: input_bytes };
        check(rows, cols, wide, pairs, unit, one, Span { address: output.address + e * output_bytes, bytes: output_bytes })?;
        if output.meets(unit) { return Err("packed group: the output overlaps an operand"); }
    }
    if output.meets(input) { return Err("packed group: the output overlaps an operand"); }
    Ok(())
}

/// The host check of one grouped launch of the packed rows tile: run `r` is the packed unit of one matrix
/// (`runs[r].0`, exactly `layout(rows, cols, wide).bytes`, 16-byte aligned) and a count of launch rows (`runs[r].1`).
/// Beside the form checks here (whole stages of the form, an even row count for mixed rows) it is the grouped check of
/// the BF16 rows tile (`staged_product::check_rows_grouped_of`) with the unit as the matrix: so the input, the output,
/// the two tables and every run hold what the body reads and writes. Inside a unit the body reads a code word, a
/// scale byte and the 16-byte copies of the table only at places of a row below `rows` and a column below `cols`, all
/// inside the parts that `layout` gives. It gives the launch rows.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "the shape, the form, the stride, the runs, the operands and the two tables")]
pub fn check_rows_grouped(rows: usize, cols: usize, wide: bool, pairs: bool, stride: usize, runs: &[(Span, usize)], input: Span, inputs_at: crate::staged_product::Rows<'_>, output: Span, outputs_at: crate::staged_product::Rows<'_>) -> Result<usize, &'static str> {
    use crate::staged_product as staged;
    if cols == 0 || cols % whole(wide) != 0 { return Err("packed rows product: the columns are not whole stages (64 columns, 128 with 8-bit codes)"); }
    if pairs && rows % 2 != 0 { return Err("packed rows product: mixed gate rows and up rows need an even row count"); }
    // The bounds first, so the unit bytes below are computed only for a shape inside the envelope.
    if rows == 0 || rows > staged::ROWS_MATRIX_ROWS_MAX || cols > staged::ROWS_COLS_MAX { return Err("packed rows product: the matrix is outside the envelope of the tile"); }
    let unit = layout(rows, cols, wide).bytes as u64;
    let span = |s: Span| staged::Span { address: s.address, bytes: s.bytes };
    let runs = runs.iter().map(|&(s, count)| (span(s), count)).collect::<Vec<_>>();
    staged::check_rows_grouped_of(rows, cols, unit, stride, &runs, span(input), inputs_at, span(output), outputs_at)
}

/// The most experts of one stack that a launch with expert numbers from the device may name: every number below it is
/// exact in FP32, so the device test `id < experts as f32` and the whole-number test are exact.
pub const PICKED_EXPERTS_MAX: usize = 1 << 16;
/// The host check of one launch whose matrices the route picked on the device: `stack` holds exactly `experts` units
/// of one shape back to back, `ids` holds exactly `picks` FP32 words (the expert numbers; the device traps on a word
/// that is not a whole number below `experts`), input and output as `check_group`. Every unit of the stack passes the
/// single check with its input and output, because every unit has the bytes and the alignment of the first one; the
/// whole output meets no byte of the stack, of `ids` or of the input.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "the shape, the stack, the numbers and the operands")]
pub fn check_picked(rows: usize, cols: usize, wide: bool, pairs: bool, stack: Span, experts: usize, ids: Span, input: Span, shared: bool, output: Span) -> Result<(), &'static str> {
    if ids.bytes == 0 || ids.bytes % 4 != 0 || ids.bytes / 4 > PICKS_MAX as u64 { return Err("packed picks: the picks of a launch are 1 to 128"); }
    let picks = (ids.bytes / 4) as usize;
    check_picked_strided(rows, cols, wide, pairs, stack, experts, ids, picks, picks, picks, input, if shared { 0 } else { 1 }, output)
}

/// Host check for strided/repeated picked packed products: `picks` total expert picks organized as
/// `picks / picks_per_row` token rows in `ids` with row stride `id_stride >= picks_per_row` words.
/// `repeat == 0` shares one input row across all picks; `repeat >= 1` maps pick `j` to input row `j / repeat`.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, strided route layout and operands")]
pub fn check_picked_strided(
    rows: usize, cols: usize, wide: bool, pairs: bool, stack: Span, experts: usize,
    ids: Span, picks: usize, picks_per_row: usize, id_stride: usize,
    input: Span, repeat: usize, output: Span,
) -> Result<(), &'static str> {
    if experts == 0 || experts > PICKED_EXPERTS_MAX { return Err("packed picks: the experts of a stack are 1 to 65536"); }
    if picks == 0 || picks > PICKS_MAX { return Err("packed picks: the picks of a launch are 1 to 128"); }
    if picks_per_row == 0 || !picks.is_multiple_of(picks_per_row) || id_stride < picks_per_row {
        return Err("packed picks: bad strided route layout");
    }
    if repeat > 0 && !picks.is_multiple_of(repeat) { return Err("packed picks: bad input repeat count"); }
    let token_rows = picks / picks_per_row;
    let id_words = (token_rows as u64).checked_mul(id_stride as u64).and_then(|w| w.checked_mul(4));
    if Some(ids.bytes) != id_words { return Err("packed picks: route buffer size differs from strided layout"); }
    if ids.end().is_none() || stack.end().is_none() { return Err("packed picks: an operand ends above the address space"); }
    if ids.address == 0 || ids.address % 4 != 0 { return Err("packed picks: the expert numbers are not word aligned"); }
    let unit = layout(rows.min(ELEMENTS_MAX), cols.min(COLS_MAX), wide).bytes as u64;
    if unit % 16 != 0 { return Err("packed picks: a unit is not a whole number of 16-byte copies"); }
    let input_rows = picks.checked_div(repeat).map_or(1u64, |rows| rows as u64);
    let bytes = |count: usize, times: u64| (count as u64).checked_mul(4).and_then(|b| b.checked_mul(times));
    let (Some(input_bytes), Some(output_bytes)) = (bytes(cols, 1), bytes(rows, 1)) else { return Err("packed group: the shape is above the envelope of the tile"); };
    let (Some(all_inputs), Some(all_outputs)) = (bytes(cols, input_rows), bytes(rows, picks as u64)) else { return Err("packed group: the shape is above the envelope of the tile"); };
    if input.bytes != all_inputs { return Err("packed group: the input differs from the columns of the matrices"); }
    if output.bytes != all_outputs { return Err("packed group: the output differs from the rows of the matrices"); }
    if input.end().is_none() || output.end().is_none() { return Err("packed group: an operand ends above the address space"); }
    for e in 0..picks as u64 {
        let in_row = if repeat == 0 { 0 } else { e / repeat as u64 };
        let one = Span { address: input.address + in_row * input_bytes, bytes: input_bytes };
        check(rows, cols, wide, pairs, Span { address: stack.address, bytes: unit }, one, Span { address: output.address + e * output_bytes, bytes: output_bytes })?;
    }
    if output.meets(input) { return Err("packed group: the output overlaps an operand"); }
    if (experts as u64).checked_mul(unit) != Some(stack.bytes) { return Err("packed picks: the stack differs from its experts' units"); }
    if output.meets(stack) || output.meets(ids) { return Err("packed picks: the output overlaps an operand"); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const AT: u64 = 0x2_0000_0000;
    /// Every refusal of the group check, and the groups it takes.
    #[test]
    fn the_group_check_refuses_what_a_launch_cannot_take_and_takes_a_good_group() {
        let (rows, cols) = (64usize, 128usize);
        let one = layout(rows, cols, false).bytes as u64;
        let units = |n: usize| (0..n as u64).map(|e| Span { address: AT + e * 0x10_0000, bytes: one }).collect::<Vec<_>>();
        let input = |n: u64| Span { address: AT + (1 << 30), bytes: cols as u64 * 4 * n };
        let output = |n: u64| Span { address: AT + (2 << 30), bytes: rows as u64 * 4 * n };
        assert_eq!(check_group(rows, cols, false, false, &units(8), input(1), true, output(8)), Ok(()));
        assert_eq!(check_group(rows, cols, false, true, &units(8), input(8), false, output(8)), Ok(()));
        assert_eq!(check_group(rows, cols, false, false, &units(GROUP_MAX), input(1), true, output(GROUP_MAX as u64)), Ok(()));
        let refused = |r: Result<(), &'static str>, text: &str| assert_eq!(r, Err(text));
        refused(check_group(rows, cols, false, false, &[], input(1), true, output(0)), "packed group: the matrices of a group are 1 to 32");
        refused(check_group(rows, cols, false, false, &units(GROUP_MAX + 1), input(1), true, output(33)), "packed group: the matrices of a group are 1 to 32");
        refused(check_group(rows, cols, false, false, &units(8), input(8), true, output(8)), "packed group: the input differs from the columns of the matrices");
        refused(check_group(rows, cols, false, false, &units(8), input(1), false, output(8)), "packed group: the input differs from the columns of the matrices");
        refused(check_group(rows, cols, false, false, &units(8), input(1), true, output(7)), "packed group: the output differs from the rows of the matrices");
        refused(check_group(rows, cols, false, false, &units(8), input(1), true, Span { address: u64::MAX - 8, bytes: rows as u64 * 32 }), "packed group: an operand ends above the address space");
        // A unit of another shape, a unit not aligned: the single check names it.
        let mut bad = units(8); bad[5].bytes -= 16;
        refused(check_group(rows, cols, false, false, &bad, input(1), true, output(8)), "packed product: the unit differs from the bytes of the codes, scales and table of this shape");
        let mut bad = units(8); bad[7].address += 4;
        refused(check_group(rows, cols, false, false, &bad, input(1), true, output(8)), "packed product: the unit is not 16-byte aligned");
        // The whole output over another matrix's unit (not its own), and over the input of another matrix.
        let mut bad = units(8); bad[6].address = output(8).address + 16 - one;
        refused(check_group(rows, cols, false, false, &bad, input(1), true, output(8)), "packed group: the output overlaps an operand");
        let over = Span { address: input(8).address + cols as u64 * 4 * 8 - rows as u64 * 4, bytes: rows as u64 * 4 * 8 };
        refused(check_group(rows, cols, false, false, &units(8), input(8), false, over), "packed group: the output overlaps an operand");
        refused(check_group(rows, cols, false, true, &units(8), input(1), true, Span { address: AT + 1, bytes: rows as u64 * 32 }), "packed product: the output is not word aligned");
    }
    /// Every refusal of the check of a grouped launch of the packed rows tile, and the launches it takes: the form
    /// checks here, then the unit bytes and every check of the BF16 rows tile with the unit as the matrix.
    #[test]
    fn the_packed_rows_check_refuses_what_a_launch_cannot_take_and_takes_good_runs() {
        use crate::staged_product::Rows;
        let (rows, cols) = (128usize, 256usize);
        for wide in [false, true] {
            let unit = layout(rows, cols, wide).bytes as u64;
            let runs = |counts: &[usize]| counts.iter().enumerate().map(|(e, &n)| (Span { address: AT + e as u64 * 0x10_0000, bytes: unit }, n)).collect::<Vec<_>>();
            let input = |n: u64| Span { address: AT + (1 << 30), bytes: cols as u64 * 4 * n };
            let output = |n: u64| Span { address: AT + (2 << 30), bytes: rows as u64 * 4 * n };
            let at = |n: usize| Rows { count: n, table: None };
            assert_eq!(check_rows_grouped(rows, cols, wide, true, rows, &runs(&[3, 64, 1]), input(68), at(68), output(68), at(68)), Ok(68));
            let ids = (0..68u32).rev().collect::<Vec<_>>();
            assert_eq!(check_rows_grouped(rows, cols, wide, false, rows, &runs(&[67, 1]), input(5), Rows { count: 5, table: Some(&vec![4; 68]) }, output(68), Rows { count: 68, table: Some(&ids) }), Ok(68));
            let refused = |r: Result<usize, &'static str>, text: &str| assert_eq!(r, Err(text));
            refused(check_rows_grouped(rows, 192, wide, false, rows, &runs(&[1]), Span { bytes: 192 * 4, ..input(1) }, at(1), output(1), at(1)),
                if wide { "packed rows product: the columns are not whole stages (64 columns, 128 with 8-bit codes)" } else { "staged rows product: the weights differ from the bytes of one matrix of rows x columns" });
            refused(check_rows_grouped(rows - 1, cols, wide, true, rows, &runs(&[1]), input(1), at(1), Span { bytes: (rows as u64 - 1) * 4, ..output(1) }, at(1)), "packed rows product: mixed gate rows and up rows need an even row count");
            refused(check_rows_grouped(rows, 8192, wide, false, rows, &runs(&[1]), input(1), at(1), output(1), at(1)), "packed rows product: the matrix is outside the envelope of the tile");
            let mut bad = runs(&[2, 2]); bad[1].0.bytes -= 16;
            refused(check_rows_grouped(rows, cols, wide, false, rows, &bad, input(4), at(4), output(4), at(4)), "staged rows product: the weights differ from the bytes of one matrix of rows x columns");
            let mut bad = runs(&[2, 2]); bad[0].0.address += 4;
            refused(check_rows_grouped(rows, cols, wide, false, rows, &bad, input(4), at(4), output(4), at(4)), "staged rows product: the weights are not 16-byte aligned");
            let mut bad = runs(&[2, 2]); bad[1].0.address = output(4).address;
            refused(check_rows_grouped(rows, cols, wide, false, rows, &bad, input(4), at(4), output(4), at(4)), "staged rows product: the output overlaps an operand");
            refused(check_rows_grouped(rows, cols, wide, false, rows, &runs(&[1025]), input(1025), at(1025), output(1025), at(1025)), "staged rows product: a run has no launch row or more than the tile takes");
            refused(check_rows_grouped(rows, cols, wide, false, rows, &runs(&[2]), input(2), at(2), output(2), Rows { count: 2, table: Some(&[1, 1]) }), "staged rows product: the output table names a row twice");
        }
    }
    /// The grid width: one wave of places on the device, at most one block a row block, at least one.
    #[test]
    fn the_spread_fills_one_wave_and_never_passes_the_row_blocks() {
        // 2304 rows: 72 row blocks. 48 multiprocessors: 96 places.
        assert_eq!(spread(2304, 8, 48), 12);
        assert_eq!(spread(2304, 1, 48), 72);
        assert_eq!(spread(2304, 32, 48), 3);
        assert_eq!(spread(64, 1, 48), 2);
        assert_eq!(spread(33, 3, 48), 2);
        assert_eq!(spread(2304, 8, 0), 1);
        assert_eq!(spread(2304, 0, 48), 72);
        for rows in [1usize, 31, 32, 33, 2304, 1 << 20] { for picks in 1..=GROUP_MAX { for sms in [1usize, 20, 48, 160] {
            let w = spread(rows, picks, sms) as usize;
            assert!(w >= 1 && w <= rows.div_ceil(BLOCK_ROWS));
        } } }
    }
    /// Every refusal of the check of a launch with expert numbers from the device, and the launches it takes.
    #[test]
    fn the_picked_check_refuses_what_a_launch_cannot_take_and_takes_a_good_stack() {
        let (rows, cols, experts) = (64usize, 128usize, 40usize);
        let unit = layout(rows, cols, false).bytes as u64;
        let stack = Span { address: AT, bytes: unit * experts as u64 };
        let ids = |n: u64| Span { address: AT + (3 << 30), bytes: 4 * n };
        let input = |n: u64| Span { address: AT + (1 << 30), bytes: cols as u64 * 4 * n };
        let output = |n: u64| Span { address: AT + (2 << 30), bytes: rows as u64 * 4 * n };
        assert_eq!(check_picked(rows, cols, false, false, stack, experts, ids(8), input(1), true, output(8)), Ok(()));
        assert_eq!(check_picked(rows, cols, false, true, stack, experts, ids(8), input(8), false, output(8)), Ok(()));
        assert_eq!(check_picked(rows, cols, false, false, stack, experts, ids(128), input(128), false, output(128)), Ok(()));
        assert_eq!(check_picked(rows, cols, false, false, Span { address: AT, bytes: unit }, 1, ids(1), input(1), true, output(1)), Ok(()));
        let refused = |r: Result<(), &'static str>, text: &str| assert_eq!(r, Err(text));
        refused(check_picked(rows, cols, false, false, Span { address: AT, bytes: 0 }, 0, ids(8), input(1), true, output(8)), "packed picks: the experts of a stack are 1 to 65536");
        refused(check_picked(rows, cols, false, false, stack, PICKED_EXPERTS_MAX + 1, ids(8), input(1), true, output(8)), "packed picks: the experts of a stack are 1 to 65536");
        refused(check_picked(rows, cols, false, false, stack, experts, ids(0), input(1), true, output(0)), "packed picks: the picks of a launch are 1 to 128");
        refused(check_picked(rows, cols, false, false, stack, experts, ids(129), input(1), true, output(129)), "packed picks: the picks of a launch are 1 to 128");
        refused(check_picked(rows, cols, false, false, stack, experts, Span { address: AT + (3 << 30), bytes: 30 }, input(1), true, output(8)), "packed picks: the picks of a launch are 1 to 128");
        refused(check_picked(rows, cols, false, false, stack, experts, Span { address: u64::MAX - 8, bytes: 32 }, input(1), true, output(8)), "packed picks: an operand ends above the address space");
        refused(check_picked(rows, cols, false, false, stack, experts, Span { address: 0, bytes: 32 }, input(1), true, output(8)), "packed picks: the expert numbers are not word aligned");
        refused(check_picked(rows, cols, false, false, stack, experts, Span { address: AT + (3 << 30) + 2, bytes: 32 }, input(1), true, output(8)), "packed picks: the expert numbers are not word aligned");
        // The stack: one unit short, one unit long, not aligned, at address 0.
        refused(check_picked(rows, cols, false, false, Span { address: AT, bytes: unit * (experts as u64 - 1) }, experts, ids(8), input(1), true, output(8)), "packed picks: the stack differs from its experts' units");
        refused(check_picked(rows, cols, false, false, Span { address: AT, bytes: unit * (experts as u64 + 1) }, experts, ids(8), input(1), true, output(8)), "packed picks: the stack differs from its experts' units");
        refused(check_picked(rows, cols, false, false, Span { address: AT + 4, bytes: stack.bytes }, experts, ids(8), input(1), true, output(8)), "packed product: the unit is not 16-byte aligned");
        refused(check_picked(rows, cols, false, false, Span { address: 0, bytes: stack.bytes }, experts, ids(8), input(1), true, output(8)), "packed product: an operand has address 0");
        // The input and the output as the group check, and a shape that overflows.
        refused(check_picked(rows, cols, false, false, stack, experts, ids(8), input(8), true, output(8)), "packed group: the input differs from the columns of the matrices");
        refused(check_picked(rows, cols, false, false, stack, experts, ids(8), input(1), true, output(7)), "packed group: the output differs from the rows of the matrices");
        refused(check_picked(1usize << 62, cols, false, false, stack, experts, ids(8), input(1), true, output(8)), "packed group: the shape is above the envelope of the tile");
        // The output over the last unit of the stack (no single unit check sees it), and over the expert numbers.
        let over_stack = Span { address: AT + unit * (experts as u64 - 1), bytes: rows as u64 * 4 * 8 };
        refused(check_picked(rows, cols, false, false, stack, experts, ids(8), input(1), true, over_stack), "packed picks: the output overlaps an operand");
        let over_ids = Span { address: ids(8).address - 16, bytes: rows as u64 * 4 * 8 };
        refused(check_picked(rows, cols, false, false, stack, experts, ids(8), input(1), true, over_ids), "packed picks: the output overlaps an operand");
        // Review REVIDS: a stack that wraps the address space, columns that overflow, an output over the input.
        refused(check_picked(rows, cols, false, false, Span { address: u64::MAX - 15, bytes: stack.bytes }, experts, ids(8), input(1), true, output(8)), "packed picks: an operand ends above the address space");
        refused(check_picked(rows, 1usize << 62, false, false, stack, experts, ids(8), input(1), true, output(8)), "packed group: the shape is above the envelope of the tile");
        refused(check_picked(rows, usize::MAX, false, false, stack, experts, ids(8), input(8), false, output(8)), "packed group: the shape is above the envelope of the tile");
        let over_input = Span { address: input(8).address + cols as u64 * 4 * 8 - 16, bytes: rows as u64 * 4 * 8 };
        refused(check_picked(rows, cols, false, false, stack, experts, ids(8), input(8), false, over_input), "packed group: the output overlaps an operand");
    }
    /// Review REVPACK findings 2, 3 and 4: a shape far above the envelope is refused by the group check (no overflow
    /// panic); an operand that ends above the address space and an operand at address 0 are refused by both checks.
    #[test]
    fn the_checks_refuse_an_overflowing_shape_an_operand_past_the_address_space_and_address_zero() {
        let (rows, cols) = (64usize, 128usize);
        let unit = Span { address: AT, bytes: layout(rows, cols, false).bytes as u64 };
        let input = Span { address: AT + (1 << 30), bytes: cols as u64 * 4 };
        let output = Span { address: AT + (2 << 30), bytes: rows as u64 * 4 };
        assert_eq!(check(rows, cols, false, false, unit, input, output), Ok(()));
        let refused = |r: Result<(), &'static str>, text: &str| assert_eq!(r, Err(text));
        // Finding 2: rows or columns whose bytes overflow u64, one matrix and a full group.
        for (r, c) in [(1usize << 62, cols), (rows, 1usize << 62), (usize::MAX, cols), (rows, usize::MAX)] {
            refused(check_group(r, c, false, false, &[unit], input, true, output), "packed group: the shape is above the envelope of the tile");
            refused(check_group(r, c, false, false, &[unit; GROUP_MAX], input, false, output), "packed group: the shape is above the envelope of the tile");
        }
        // Bytes that fit u64 but whose product with the picks does not.
        refused(check_group(rows, 1usize << 59, false, false, &[unit; 16], input, false, output), "packed group: the shape is above the envelope of the tile");
        // Finding 4: the input or the output ends above the address space.
        let past = |span: Span| Span { address: u64::MAX - 8, bytes: span.bytes };
        refused(check(rows, cols, false, false, unit, past(input), output), "packed product: an operand ends above the address space");
        refused(check(rows, cols, false, false, unit, input, past(output)), "packed product: an operand ends above the address space");
        refused(check_group(rows, cols, false, false, &[unit], past(input), true, output), "packed group: an operand ends above the address space");
        // Finding 3: an operand at address 0, in the single check and through the group check.
        let zero = |span: Span| Span { address: 0, bytes: span.bytes };
        for (u, i, o) in [(zero(unit), input, output), (unit, zero(input), output), (unit, input, zero(output))] {
            refused(check(rows, cols, false, false, u, i, o), "packed product: an operand has address 0");
            refused(check_group(rows, cols, false, false, &[u], i, true, o), "packed product: an operand has address 0");
        }
    }
    /// The copies of the body (`code_copy`, `scale_copy`, `input_copy`, `table_copies`) against the layout: every copy
    /// is inside its part, starts on the boundary of its size, and the copies of a part end exactly at its end.
    #[test]
    fn every_copy_of_the_body_is_inside_its_part_and_the_last_one_ends_at_its_end() {
        let mut shapes = 0;
        for wide in [false, true] {
            for cols in (whole(wide)..=1152).step_by(whole(wide)).chain([2304, COLS_MAX]) {
                for rows in [1usize, 2, 31, 32, 33, 64, 70, 160] {
                    assert!(shape_admitted(rows, cols, wide, false), "{rows} x {cols} wide {wide}");
                    let unit = layout(rows, cols, wide);
                    // The three parts of the unit: the codes from 0, the scales, the table. Each part holds whole
                    // blocks of 32 rows; the rows above `rows` are zero and never copied.
                    let (code_end, scale_end) = (tiled_rows(rows) * code_stride(cols, wide), tiled_rows(rows) * scale_stride(cols, wide));
                    assert!(code_end <= unit.scales && unit.scales + scale_end <= unit.table && unit.table + table_copies(wide) * 16 == unit.bytes);
                    assert_eq!((unit.scales % 16, unit.table % 16, unit.bytes % 16), (0, 0, 0));
                    let (mut code_last, mut scale_last, mut input_last) = (0, 0, 0);
                    let (mut code_seen, mut scale_seen) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
                    for row in 0..tiled_rows(rows) {
                        for stage in 0..stages(cols) {
                            for part in 0..code_stage(wide) / 16 {
                                let at = code_copy(row, stage, part, cols, wide);
                                assert!(at % 16 == 0 && at + 16 <= code_end && code_seen.insert(at), "codes {rows} x {cols} wide {wide}: row {row} stage {stage} part {part}");
                                code_last = code_last.max(at + 16);
                            }
                            let at = scale_copy(row, stage, cols, wide);
                            // The 8-bit form copies the scales of an even stage and the odd stage after it at once.
                            assert!(at % 4 == 0 && at + 4 <= scale_end && (scale_seen.insert(at) || (wide && stage % 2 == 1)), "scales {rows} x {cols} wide {wide}: row {row} stage {stage}");
                            scale_last = scale_last.max(at + 4);
                        }
                    }
                    for stage in 0..stages(cols) {
                        for part in 0..STAGE / 4 {
                            let at = input_copy(stage, part);
                            assert!(at + 4 <= cols, "input {cols}: stage {stage} part {part}");
                            input_last = input_last.max(at + 4);
                        }
                    }
                    assert_eq!((code_last, scale_last, input_last), (code_end, scale_end, cols), "{rows} x {cols} wide {wide}");
                    // Distinct copies that fill each part: no stored byte is left out and none is read twice.
                    assert_eq!((code_seen.len() * 16, scale_seen.len() * 4), (code_end, scale_end), "{rows} x {cols} wide {wide}");
                    // The copies of one row are its whole stride, so no stored byte of a row is left out.
                    assert_eq!((stages(cols) * code_stage(wide), stages(cols) * if wide { 2 } else { 4 }), (code_stride(cols, wide), scale_stride(cols, wide)));
                    shapes += 1;
                }
            }
        }
        assert!(shapes >= 2 * 8 * 11, "{shapes}");
    }
    fn spans(rows: usize, cols: usize, wide: bool) -> (Span, Span, Span) {
        let unit = Span { address: AT, bytes: layout(rows, cols, wide).bytes as u64 };
        let input = Span { address: AT + (1 << 30), bytes: cols as u64 * 4 };
        (unit, input, Span { address: AT + (2 << 30), bytes: rows as u64 * 4 })
    }
    #[test]
    fn the_layout_pads_each_part_and_the_real_shapes_are_admitted() {
        // 2,304 x 2,304 of 4-bit codes: 2,654,208 code bytes, 331,776 scale bytes, the table of 8,192.
        assert_eq!(layout(2304, 2304, false), Layout { scales: 2_654_208, table: 2_985_984, bytes: 2_994_176 });
        assert_eq!(layout(2304, 1152, true), Layout { scales: 2_654_208, table: 2_737_152, bytes: 2_738_176 });
        // A part holds whole blocks of 32 rows.
        assert_eq!(layout(3, 64, false), Layout { scales: 1024, table: 1152, bytes: 1152 + 8192 });
        assert_eq!(layout(1, 128, true), Layout { scales: 4096, table: 4224, bytes: 4224 + 1024 });
        assert_eq!((layout(33, 64, false), layout(3, 48, false), layout(1, 32, true)), (layout(64, 64, false), layout(32, 64, false), layout(1, 128, true)));
        // The tiles: stage 1 of row 0 comes after stage 0 of all 32 rows of the block; block 1 after all of block 0.
        assert_eq!((code_copy(0, 0, 0, 128, false), code_copy(1, 0, 0, 128, false), code_copy(0, 1, 0, 128, false), code_copy(32, 0, 0, 128, false)), (0, 32, 1024, 2048));
        assert_eq!((scale_copy(1, 0, 256, true), scale_copy(1, 1, 256, true), scale_copy(0, 2, 256, true)), (4, 4, 128));
        for (rows, cols) in [(2304, 2304), (2304, 1152)] {
            for wide in [false, true] {
                let (unit, input, output) = spans(rows, cols, wide);
                for pairs in [false, true] {
                    assert_eq!(check(rows, cols, wide, pairs, unit, input, output), Ok(()));
                    assert!(admitted(rows, cols, wide, pairs, unit.address as usize, input.address as usize, THREADS));
                }
            }
        }
        assert_eq!(geometry(2304), (72, 160));
        assert_eq!(geometry(33), (2, 160));
        assert_eq!(cols_word(2304, true, true), 2304 | WIDE | PAIRS);
        assert_eq!((cols_word(1152, false, false), cols_word(1152, false, false) & COLS), (1152, 1152));
        assert_eq!((output_row(0, 8, true), output_row(1, 8, true), output_row(6, 8, true), output_row(7, 8, true)), (0, 4, 3, 7));
        assert_eq!(output_row(5, 8, false), 5);
    }
    #[test]
    fn every_refusal_of_the_host_check_has_its_text() {
        let (rows, cols) = (64, 256);
        let (unit, input, output) = spans(rows, cols, false);
        let refused = |rows, cols, wide, pairs, unit, input, output, text: &str| {
            let error = check(rows, cols, wide, pairs, unit, input, output).unwrap_err();
            assert!(error.contains(text), "{error}");
        };
        assert_eq!(check(rows, cols, false, true, unit, input, output), Ok(()));
        refused(0, cols, false, false, unit, input, output, "no rows or no columns");
        refused(rows, 0, false, false, unit, input, output, "no rows or no columns");
        // Columns that are not whole stages, in each form, and columns above the envelope.
        refused(rows, 96, false, false, unit, input, output, "not whole stages");
        refused(rows, 192, true, false, unit, input, output, "not whole stages");
        refused(rows, 8192 + 64, false, false, unit, input, output, "not whole stages");
        refused((1 << 25) / cols + 1, cols, false, false, unit, input, output, "above the envelope");
        let odd = spans(63, cols, false);
        refused(63, cols, false, true, odd.0, odd.1, odd.2, "even row count");
        // A unit of another shape or of the other form, one byte more, one byte less.
        refused(rows, cols, true, false, unit, input, output, "the unit differs");
        refused(rows, cols, false, false, Span { bytes: unit.bytes + 16, ..unit }, input, output, "the unit differs");
        refused(rows, cols, false, false, Span { bytes: unit.bytes - 16, ..unit }, input, output, "the unit differs");
        refused(rows, cols, false, false, unit, Span { bytes: input.bytes - 4, ..input }, output, "the input differs");
        refused(rows, cols, false, false, unit, Span { bytes: input.bytes + 4, ..input }, output, "the input differs");
        refused(rows, cols, false, false, unit, input, Span { bytes: output.bytes - 4, ..output }, "the output differs");
        refused(rows, cols, false, false, unit, input, Span { bytes: output.bytes + 4, ..output }, "the output differs");
        refused(rows, cols, false, false, Span { address: u64::MAX - 15, ..unit }, input, output, "above the address space");
        refused(rows, cols, false, false, Span { address: AT + 8, ..unit }, input, output, "the unit is not 16-byte aligned");
        refused(rows, cols, false, false, unit, Span { address: input.address + 4, ..input }, output, "the input is not 16-byte aligned");
        refused(rows, cols, false, false, unit, input, Span { address: output.address + 2, ..output }, "the output is not word aligned");
        // An output inside the unit, at its last word, and one that ends inside the input.
        refused(rows, cols, false, false, unit, input, Span { address: unit.address + unit.bytes - 4, ..output }, "overlaps an operand");
        refused(rows, cols, false, false, unit, input, Span { address: input.address - output.bytes + 4, ..output }, "overlaps an operand");
        // An output that ends where an operand starts is beside it, not in it.
        assert_eq!(check(rows, cols, false, false, unit, input, Span { address: input.address - output.bytes, ..output }), Ok(()));
        // The kernel's own question: another block size, a shape outside, an operand that is not aligned.
        assert!(admitted(rows, cols, false, false, AT as usize, AT as usize, THREADS));
        assert!(!admitted(rows, cols, false, false, AT as usize, AT as usize, 128));
        assert!(!admitted(rows, 96, false, false, AT as usize, AT as usize, THREADS));
        assert!(!admitted(rows, 192, true, false, AT as usize, AT as usize, THREADS));
        assert!(!admitted(rows, cols, false, false, AT as usize + 8, AT as usize, THREADS));
        assert!(!admitted(rows, cols, false, false, AT as usize, AT as usize + 4, THREADS));
        assert!(!admitted(63, cols, false, true, AT as usize, AT as usize, THREADS));
    }
    #[test]
    fn the_round_and_the_power_are_the_arithmetic_of_the_widen_step() {
        // A value that BF16 holds stays; a value half way goes to the even pattern; one above half way goes up.
        assert_eq!(round_bf16(1.5).to_bits(), 1.5f32.to_bits());
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8000)).to_bits(), 0x3f80_0000);
        assert_eq!(round_bf16(f32::from_bits(0x3f81_8000)).to_bits(), 0x3f82_0000);
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8001)).to_bits(), 0x3f81_0000);
        assert_eq!(round_bf16(-0.0).to_bits(), 0x8000_0000);
        assert_eq!((power(127), power(128), power(126), power(1)), (1.0, 2.0, 0.5, f32::MIN_POSITIVE));
        assert_eq!(power(0), f32::MIN_POSITIVE / 2.0);
        assert!(power(255).is_infinite());
    }
}
