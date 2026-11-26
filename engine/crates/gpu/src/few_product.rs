//! The small-batch BF16 product: one launch gives the product of ONE matrix of BF16 values (two a word, row-major)
//! with a few input rows of FP32 values (`FEW_MIN` to `FEW_MAX`), output `[inputs][rows]`. It is the form of the
//! decode step of several chats: each chat brings one input row, and the matrix is read once for all of them.
//!
//! Each output word accumulates BF16 MMA groups in rising K order from +0.0.
//! An input row uses the same chain for every supported batch size.
//!
//! An entry of its own (`resident_launch::Kind::PlainStagedFew`), with its own body; the row product, its rows tile
//! and their files do not change. This file holds the launch geometry, the admission and the host check of a launch:
//! pure functions, shared by the kernel body and the host, so one predicate decides both. The kernel crate includes
//! this file by path. It calls only the shape rule of `staged_product`, which is pure too.
use crate::staged_product::{ELEMENTS_MAX, STAGE};
#[cfg(not(target_arch = "nvptx64"))]
use crate::staged_product::Span;

/// Most columns of a launch. The body streams 64-column stages and sizes no array by the columns, so it is not bound
/// to the 8,192 columns of the row product; this bound admits the draft model's down projection (9,216 columns).
/// The envelope (rows x cols at most `ELEMENTS_MAX`) still holds.
pub const FEW_COLS_MAX: usize = 16_384;
/// Fewest and most input rows of one launch.
pub const FEW_MIN: usize = 1;
pub const FEW_MAX: usize = 16;
/// Threads of a narrow block (one warp on 4 weight rows) and a wide block (4 warps on 64 weight rows).
pub const FEW_NARROW_THREADS: usize = 32;
pub const FEW_THREADS: usize = 128;
/// Pipeline stages of a wide block: 3 stages x (8,192 B weights + 2,048 B shared BF16 inputs) = 30,720 bytes (3 blocks/SM = 12 warps/SM).
pub const FEW_STAGES: usize = 3;
/// Shared bytes of one weight row in a wide stage (64 BF16 = 128 bytes), of the weights of a stage (8,192 bytes),
/// of one full stage (10,240 bytes), and of a block (30,720 bytes).
pub const FEW_ROW_BYTES: usize = STAGE * 2;
pub const FEW_INPUT_AT: usize = 64 * FEW_ROW_BYTES;
pub const FEW_STAGE_BYTES: usize = FEW_INPUT_AT + FEW_MAX * STAGE * 2;
pub const FEW_BLOCK_BYTES: usize = FEW_STAGES * FEW_STAGE_BYTES;
const _: () = assert!(FEW_ROW_BYTES.is_multiple_of(16) && FEW_INPUT_AT == 8_192 && FEW_STAGE_BYTES == 10_240 && FEW_BLOCK_BYTES == FEW_STAGES * 10_240);

/// Command flags for `PLAIN_STAGED_FEW` (payload word 8).
pub const FEW_ROUND_OUT: u64 = 1;
pub const FEW_SPLIT_OUT: u64 = 2;
/// Maximum output buffers in a split `PLAIN_STAGED_FEW` launch.
pub const FEW_SPLIT_MAX: usize = 8;
/// The attached gate blend of a few-row forget-and-gain product (2026-11-25, W-BOSS-SC-DENSE-VERIFY): the recorder
/// joins the product (`2 dim` rows of 64 columns, 2 to 16 input rows, no other flag) with the GateBlend of its rows
/// that reads it (`Backend::attach_gate_blend_rows`), as it joins the one-row pair. The kept rows, the message rows
/// and the blended output (`inputs x dim` words each) in words `FEW_BLEND_WORD`, `+ 1` and `+ 2` (the split words,
/// unused without the split flag); the launch then has `dim / 32` blocks (as many as the product's 64-row blocks).
pub const FEW_GATE_BLEND: u64 = 4;
pub const FEW_BLEND_WORD: usize = 10;
/// Grouped dual-input narrow launch (up to 3 matrices of one shape across `block_idx_y = 0..3`):
/// group 0 uses words `0, 1, 7`; groups 1 and 2 use `(weight, input1, input2)` at `FEW_GROUP_WORD + 3*(g-1)`.
pub const FEW_GROUP_PAIR: u64 = 8;
pub const FEW_GROUP_WORD: usize = 14;
/// Direct register-ring BF16 panel path for `9..=16` input rows (`plain_staged_few.rs`): stages input rows in
/// dynamic shared memory in parts of whole `FEW_PANEL_RING_DEPTH` turns and decodes each 16x16 BF16 panel tile
/// once across both 8-row MMA slabs (`c0` and `c1`). Payload word 7 holds `part_tiles`; words `FEW_PANEL_WORD`
/// and `+ 2` hold the panel data and metadata addresses. May combine with `FEW_SPLIT_OUT`.
pub const FEW_PANEL_RING: u64 = 16;
/// Tiles in flight per warp for `FEW_PANEL_RING` (keeps `resident_launch_plain_staged_few` under 144 regs, 0 spill).
pub const FEW_PANEL_RING_DEPTH: usize = 8;
/// The shape the attached gate blend takes: `rows = 2 dim` weight rows of 64 columns, `dim` a whole number of 32-row
/// blocks, 2 to 16 input rows (1 or 2 MMA N-tiles; one row is the one-row entry's attached form).
pub const fn few_gate_blend_admitted(rows: usize, cols: usize, inputs: usize) -> bool {
    let dim = rows / 2;
    cols == STAGE && rows == 2 * dim && dim >= 32 && dim.is_multiple_of(32) && inputs >= 2 && inputs <= FEW_MAX
        && few_shape_admitted(rows, cols, inputs)
}
const _: () = assert!(FEW_GATE_BLEND & (FEW_ROUND_OUT | FEW_SPLIT_OUT | FEW_GROUP_PAIR | FEW_PANEL_RING) == 0
    && FEW_GROUP_PAIR & (FEW_ROUND_OUT | FEW_SPLIT_OUT | FEW_PANEL_RING) == 0
    && FEW_PANEL_RING & (FEW_ROUND_OUT | FEW_SPLIT_OUT) == 0
    && FEW_BLEND_WORD + 2 < crate::bf16_panels::FEW_PANEL_WORD
    && FEW_GROUP_WORD + 6 <= crate::bf16_panels::FEW_PANEL_WORD);

/// The one-row entry (`resident_launch::Kind::PlainStagedOne`, `kernels-inkling/src/plain_staged_one.rs`): its wide
/// form double-buffers (2 stages, 20,480 shared bytes), and it takes only dynamic shared memory, so a panel launch
/// asks for `3 x cols` bytes (the BF16 input and the metadata of a 64-row group).
pub const ONE_STAGES: usize = 2;
pub const ONE_BLOCK_BYTES: usize = ONE_STAGES * FEW_STAGE_BYTES;
/// Payload words of a panel launch of the one-row entry: the panel base address and starting 16-row band.
pub const ONE_PANEL_WORD: usize = 10;
const _: () = assert!(ONE_BLOCK_BYTES == 20_480 && ONE_PANEL_WORD > 9);
/// Envelope for one-row/small-batch Vade-BF11-BP panel launches (allows 1-launch vocabulary head up to 1B weights).
pub const ONE_PANEL_ELEMENTS_MAX: usize = 1 << 30;
/// The panel form takes a matrix of at least 256 columns whose 1-row BF16 input fits its shared bytes.
pub const fn one_panel_admitted(rows: usize, cols: usize) -> bool {
    rows > 0 && cols >= 256 && cols <= FEW_COLS_MAX && cols.is_multiple_of(STAGE)
        && cols * 2 <= ONE_BLOCK_BYTES && rows <= ONE_PANEL_ELEMENTS_MAX / cols
}
/// The panel form of the one-row entry also takes 2 to `ONE_PANEL_INPUTS` input rows (2026-11-25, W-BOSS-SC-DENSE-VERIFY):
/// one `m16n8k16` column an input row, so a tile is one MMA for any of them, and the weights are read once with the
/// same ring of loads as one row. A word of input row `t` is the chain of that row alone: an MMA column does not meet
/// another column, so the words do not depend on the count of input rows (as in the few-row entry).
pub const ONE_PANEL_INPUTS: usize = 8;
/// Payload words of the panel form past `ONE_PANEL_WORD + 1`: the flags (0 or `FEW_SPLIT_OUT`) and the
/// count of split outputs in the next word; the 64-column chunks of input staged per part (`ONE_PART_WORD`); split
/// output `i` (address, then width) at `ONE_SPLIT_WORD + 2 i`; fused panel pair (`b_y, b_r, b_x, b_panels, b_first_band,
/// bands_a, total_blocks`) at `ONE_PAIR_WORD..+7`.
pub const ONE_FLAGS_WORD: usize = 8;
pub const ONE_PART_WORD: usize = 12;
pub const ONE_SPLIT_WORD: usize = 16;
pub const ONE_PAIR_WORD: usize = 32;
/// Third fused independent panel item (`c_y, c_r, c_x, c_panels, c_first_band, bands_ab`) for 3-way `Q+K+V` fusion.
pub const ONE_TRIPLE_WORD: usize = 40;
/// Tiles of one turn of the legacy panel load ring (kept for host unit tests).
pub const ONE_RING: usize = 12;
/// Shared bytes a multiprocessor gives its blocks, and the bytes each block also reserves (GB10, SM121a).
pub const SM_SHARED_BYTES: usize = 102_400;
pub const BLOCK_RESERVED_BYTES: usize = 1_024;
/// Most dynamic shared bytes of a panel launch: a larger size needs the opt-in attribute of the function, which a
/// retained graph could meet at a smaller setting.
pub const ONE_SHARED_MAX: usize = 49_152;
const _: () = assert!(ONE_FLAGS_WORD + 1 < ONE_PANEL_WORD && ONE_PART_WORD == ONE_PANEL_WORD + 2 && ONE_SPLIT_WORD > ONE_PART_WORD
    && ONE_PAIR_WORD >= ONE_SPLIT_WORD + 2 * FEW_SPLIT_MAX && ONE_TRIPLE_WORD >= ONE_PAIR_WORD + 7 && ONE_TRIPLE_WORD + 6 <= 128 && ONE_PANEL_INPUTS == 8);
/// Shared bytes of one staged BF16 input row for `part_chunks` 64-column chunks (`128 B/chunk`).
/// Multi-row launches (`inputs > 1`) add 64 bytes so adjacent `g` rows map to opposite 64-byte bank halves
/// (zero bank conflicts on `LDS.128`). Single-row launches (`inputs == 1`) have zero padding (`cols * 2` B).
pub const fn one_panel_row_bytes_chunks(inputs: usize, part_chunks: usize) -> usize {
    part_chunks * 128 + if inputs > 1 { 64 } else { 0 }
}
/// Double-buffered SMEM handoff bytes for Warps `1..4` -> Warp `0` in 4-warp cooperative ascending-K `Vade-BF11-BP`:
/// `2 slots * 3 producer warps * 4 tiles * 32 lanes * 16 B = 12,288 B`.
pub const COOP_W_BYTES: usize = 12_288;
/// Shared bytes of a Vade-BF11-BP panel launch with `slices` streams (`1` = independent warps, `4` = 4-warp
/// cooperative ascending-K with `COOP_W_BYTES` double-buffered weight handoff). Avoids `ONE_BLOCK_BYTES` (20,480)
/// so PDL `dependent_launch` stays enabled on `cols = 4096, inputs = 1`.
pub const fn one_panel_shared_slices(inputs: usize, part_chunks: usize, slices: usize) -> usize {
    let x_bytes = inputs * one_panel_row_bytes_chunks(inputs, part_chunks);
    if slices == 4 {
        let total = x_bytes + COOP_W_BYTES;
        if total == ONE_BLOCK_BYTES { total + 128 } else { total }
    } else {
        x_bytes
    }
}
/// Largest `part_chunks` dividing `chunks` (`cols / 64`, and a multiple of 4 when `slices == 4`) whose shared bytes
/// fit `budget`.
pub const fn one_panel_part_chunks(inputs: usize, chunks: usize, budget: usize, slices: usize) -> Option<usize> {
    if inputs == 0 || inputs > ONE_PANEL_INPUTS || chunks == 0 || !matches!(slices, 1 | 4) {
        return None;
    }
    if slices == 4 && !chunks.is_multiple_of(4) {
        return None;
    }
    if one_panel_shared_slices(inputs, chunks, slices) <= budget {
        return Some(chunks);
    }
    let step = if slices == 4 { 4 } else { 1 };
    let mut part = (chunks / 2) / step * step;
    while part >= step {
        if chunks.is_multiple_of(part) && one_panel_shared_slices(inputs, part, slices) <= budget {
            return Some(part);
        }
        part -= step;
    }
    None
}
/// Shared bytes of one staged input row of a legacy panel launch (kept for tests).
pub const fn one_panel_row_bytes(inputs: usize, part: usize) -> usize { part * 32 + if inputs > 1 { 32 } else { 0 } }
/// Shared bytes of a legacy panel launch with `warps` warps per block (`1`, `2`, or `4`).
pub const fn one_panel_shared_warps(inputs: usize, cols: usize, part: usize, warps: usize) -> usize {
    inputs * one_panel_row_bytes(inputs, part) + warps * (cols / 4)
}
/// Shared bytes of a legacy 4-warp panel launch.
pub const fn one_panel_shared(inputs: usize, cols: usize, part: usize) -> usize { one_panel_shared_warps(inputs, cols, part, 4) }
/// Legacy staged-part helper (kept for tests).
pub const fn one_panel_part_warps(inputs: usize, cols: usize, budget: usize, warps: usize) -> Option<usize> {
    let tiles = cols / 16;
    if inputs == 0 || inputs > ONE_PANEL_INPUTS || tiles == 0 || (warps != 1 && warps != 2 && warps != 4) { return None; }
    if one_panel_shared_warps(inputs, cols, tiles, warps) <= budget { return Some(tiles); }
    let mut part = tiles / ONE_RING * ONE_RING;
    while part >= ONE_RING {
        if tiles.is_multiple_of(part) && one_panel_shared_warps(inputs, cols, part, warps) <= budget { return Some(part); }
        part -= ONE_RING;
    }
    None
}
/// Legacy 4-warp staged-part helper.
pub const fn one_panel_part(inputs: usize, cols: usize, budget: usize) -> Option<usize> {
    one_panel_part_warps(inputs, cols, budget, 4)
}
/// The shared budget of a panel launch of `blocks` blocks on `sms` multiprocessors, capped by `max_blocks_per_sm`:
/// keeps full register occupancy (6 blk/SM at 128t, 4 at 192t, 3 at 256t) without exceeding `ONE_SHARED_MAX`.
pub const fn one_panel_budget_capped(blocks: usize, sms: usize, max_blocks_per_sm: usize) -> usize {
    let raw_each = if sms == 0 { 1 } else { blocks.div_ceil(sms) };
    let cap = if max_blocks_per_sm == 0 { 1 } else { max_blocks_per_sm };
    let each = if raw_each == 0 { 1 } else if raw_each < cap { raw_each } else { cap };
    let fit = SM_SHARED_BYTES / each;
    let fit = fit.saturating_sub(BLOCK_RESERVED_BYTES);
    if fit < ONE_SHARED_MAX { fit } else { ONE_SHARED_MAX }
}
/// The shared budget of a panel launch of `blocks` blocks on `sms` multiprocessors.
pub const fn one_panel_budget(blocks: usize, sms: usize) -> usize {
    let each = if sms == 0 { 1 } else { blocks.div_ceil(sms) };
    let fit = SM_SHARED_BYTES / if each == 0 { 1 } else { each };
    let fit = fit.saturating_sub(BLOCK_RESERVED_BYTES);
    if fit < ONE_SHARED_MAX { fit } else { ONE_SHARED_MAX }
}
/// The rule of the panel form for `inputs` rows: 1 to `ONE_PANEL_INPUTS` rows, a matrix of at least 256 columns.
pub const fn one_panel_rows_admitted(rows: usize, cols: usize, inputs: usize) -> bool {
    rows > 0 && cols >= 256 && cols <= FEW_COLS_MAX && cols.is_multiple_of(STAGE)
        && inputs >= 1 && inputs <= ONE_PANEL_INPUTS && rows <= ONE_PANEL_ELEMENTS_MAX / cols
}
/// Admission for `FEW_PANEL_RING`: 9 to 16 input rows, a matrix of at least 256 columns.
pub const fn few_panel_ring_admitted(rows: usize, cols: usize, inputs: usize) -> bool {
    rows > 0 && cols >= 256 && inputs > ONE_PANEL_INPUTS && inputs <= FEW_MAX && few_shape_admitted(rows, cols, inputs)
}
/// Column tiles of input staged per part for `FEW_PANEL_RING` (`9..=16` rows, `warps` warps per block).
pub const fn few_panel_ring_part_warps(inputs: usize, cols: usize, budget: usize, warps: usize) -> Option<usize> {
    let tiles = cols / 16;
    if inputs <= ONE_PANEL_INPUTS || inputs > FEW_MAX || tiles == 0 || (warps != 1 && warps != 2 && warps != 4) { return None; }
    if one_panel_shared_warps(inputs, cols, tiles, warps) <= budget { return Some(tiles); }
    let mut part = tiles / FEW_PANEL_RING_DEPTH * FEW_PANEL_RING_DEPTH;
    while part >= FEW_PANEL_RING_DEPTH {
        if tiles.is_multiple_of(part) && one_panel_shared_warps(inputs, cols, part, warps) <= budget { return Some(part); }
        part -= FEW_PANEL_RING_DEPTH;
    }
    None
}

/// A shape the launch takes: whole stages of columns inside the envelope of the row product, and `FEW_MIN` to
/// `FEW_MAX` input rows.
#[inline(always)]
pub const fn few_shape_admitted(rows: usize, cols: usize, inputs: usize) -> bool {
    cols >= STAGE && cols <= FEW_COLS_MAX && cols.is_multiple_of(STAGE) && rows >= 1 && rows <= ELEMENTS_MAX / cols && inputs >= FEW_MIN && inputs <= FEW_MAX
}
/// Launch geometry: four-row blocks for a narrow matrix, or 64-row blocks for a wide matrix.
#[inline(always)]
pub const fn few_geometry(rows: usize, cols: usize, stride: usize) -> (u32, u32) {
    if crate::staged_narrow::shape_admitted(stride, cols) { (crate::staged_narrow::blocks(rows), FEW_NARROW_THREADS as u32) } else { (rows.div_ceil(64) as u32, FEW_THREADS as u32) }
}
/// The rule of the kernel body: an admitted shape in its own geometry, with 16-byte aligned weights and input. The
/// body asks it before its first read. The host check below holds for every launch that is recorded.
#[inline(always)]
pub const fn few_admitted(rows: usize, cols: usize, inputs: usize, stride: usize, weights: usize, input: usize, threads: usize) -> bool {
    few_shape_admitted(rows, cols, inputs) && stride >= rows && threads == few_geometry(rows, cols, stride).1 as usize
        && (weights | input) & 15 == 0
}
/// Rows of one launch for a matrix of whole stages that is above the envelope: the most whole warps that stay inside
/// it. A caller runs such a matrix in parts of this many rows, each writing its columns of the output in place
/// (`stride`). `rows` for a matrix that needs no parts, and for a shape that the launch does not take.
#[cfg(not(target_arch = "nvptx64"))]
pub const fn few_part_rows(rows: usize, cols: usize) -> usize {
    if cols < STAGE || cols > FEW_COLS_MAX || !cols.is_multiple_of(STAGE) || rows <= ELEMENTS_MAX / cols { return rows; }
    let most = ELEMENTS_MAX / cols / 32 * 32;
    if most == 0 { rows } else { most }
}
/// The host check of one launch. It runs before the launch is recorded and it names what it refuses. `stride` is the
/// words from one input row to the next in the output (`rows` for an output of its own; the rows of the whole matrix
/// when this launch is one part of a matrix). It makes every read and every write of the kernel body stay inside
/// its operand:
/// - the weights hold exactly `rows x cols` BF16 values and the columns are whole stages, so every 16-byte weight
///   copy (8 values of one row) is inside the weights;
/// - the input holds exactly `inputs x cols` words, so every 16-byte input copy (4 words of one input row, below
///   `cols`) is inside it;
/// - the output holds exactly `(inputs - 1) x stride + rows` words, `stride` is at least `rows`, and the output meets no
///   operand, so the one store of an (input row, matrix row) pair, at `input row x stride + matrix row`, lands inside;
/// - the weights and the input are 16-byte aligned, so every 16-byte copy has an aligned source;
/// - every extent is computed with checked arithmetic.
#[cfg(not(target_arch = "nvptx64"))]
pub fn few_check(rows: usize, cols: usize, inputs: usize, stride: usize, weights: Span, input: Span, output: Span) -> Result<(), &'static str> {
    few_check_limit(rows, cols, inputs, stride, weights, input, output, ELEMENTS_MAX)
}
/// Host check of a Vade-BF11-BP panel launch (permits up to `ONE_PANEL_ELEMENTS_MAX` weights in one launch).
#[cfg(not(target_arch = "nvptx64"))]
pub fn one_panel_check(rows: usize, cols: usize, inputs: usize, stride: usize, weights: Span, input: Span, output: Span) -> Result<(), &'static str> {
    few_check_limit(rows, cols, inputs, stride, weights, input, output, ONE_PANEL_ELEMENTS_MAX)
}
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, stride, operands, and element envelope")]
fn few_check_limit(rows: usize, cols: usize, inputs: usize, stride: usize, weights: Span, input: Span, output: Span, elements_max: usize) -> Result<(), &'static str> {
    if rows == 0 || cols == 0 { return Err("few-row product: the matrix has no rows or no columns"); }
    if !(FEW_MIN..=FEW_MAX).contains(&inputs) { return Err("few-row product: the input rows are outside 1 to 16"); }
    if cols > FEW_COLS_MAX || !cols.is_multiple_of(STAGE) { return Err("few-row product: the columns are not whole stages of 64 up to 16384"); }
    if rows > elements_max / cols { return Err("few-row product: the matrix is above the envelope"); }
    if stride < rows || stride > u32::MAX as usize { return Err("few-row product: the output stride is below the rows of the matrix or above the ABI"); }
    if weights.bytes != rows as u64 * cols as u64 * 2 { return Err("few-row product: the weights differ from rows x columns of BF16 values"); }
    if input.bytes != inputs as u64 * cols as u64 * 4 { return Err("few-row product: the input differs from the input rows x the columns of the matrix"); }
    let output_bytes = (inputs as u64 - 1).checked_mul(stride as u64).and_then(|words| words.checked_add(rows as u64)).and_then(|words| words.checked_mul(4));
    if output_bytes != Some(output.bytes) { return Err("few-row product: the output differs from the input rows x the stride"); }
    if weights.address == 0 || input.address == 0 || output.address == 0 { return Err("few-row product: an operand is null"); }
    let ends = |span: Span| span.address.checked_add(span.bytes).is_some();
    if !ends(weights) || !ends(input) || !ends(output) { return Err("few-row product: an operand ends above the address space"); }
    if !weights.address.is_multiple_of(16) { return Err("few-row product: the weights are not 16-byte aligned"); }
    if !input.address.is_multiple_of(16) { return Err("few-row product: the input is not 16-byte aligned"); }
    if !output.address.is_multiple_of(4) { return Err("few-row product: the output is not word aligned"); }
    let meets = |a: Span, b: Span| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
    if meets(output, weights) || meets(output, input) { return Err("few-row product: the output overlaps an operand"); }
    Ok(())
}

/// Host check for dual-input `PLAIN_STAGED_FEW` over `[input1[t], input2[t]]` (`inputs x cols/2` each).
#[cfg(not(target_arch = "nvptx64"))]
pub fn few_check_pair(rows: usize, cols: usize, inputs: usize, weights: Span, input1: Span, input2: Span, output: Span) -> Result<(), &'static str> {
    if !cols.is_multiple_of(2 * STAGE) { return Err("few-row pair: each half must be whole stages of 64"); }
    if !few_shape_admitted(rows, cols, inputs) { return Err("few-row pair: shape outside envelope"); }
    let half = cols / 2;
    let half_bytes = (inputs as u64).checked_mul(half as u64).and_then(|w| w.checked_mul(4));
    let out_bytes = (inputs as u64).checked_mul(rows as u64).and_then(|w| w.checked_mul(4));
    if weights.bytes != rows as u64 * cols as u64 * 2 || Some(input1.bytes) != half_bytes || Some(input2.bytes) != half_bytes || Some(output.bytes) != out_bytes {
        return Err("few-row pair: operand extent mismatch");
    }
    let ends = |span: Span| span.address != 0 && span.address.checked_add(span.bytes).is_some();
    if !ends(weights) || !ends(input1) || !ends(input2) || !ends(output) { return Err("few-row pair: null or overflowing operand"); }
    if (weights.address | input1.address | input2.address) & 15 != 0 || output.address & 3 != 0 { return Err("few-row pair: operand alignment mismatch"); }
    let meets = |a: Span, b: Span| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
    if meets(output, weights) || meets(output, input1) || meets(output, input2) { return Err("few-row pair: output overlaps input"); }
    Ok(())
}

/// Host check for grouped dual-input narrow `PLAIN_STAGED_FEW` (1 to 3 narrow matrices of one shape in one launch).
#[cfg(not(target_arch = "nvptx64"))]
pub fn few_check_group_pair(rows: usize, cols: usize, inputs: usize, items: &[(Span, Span, Span)], output: Span) -> Result<(), &'static str> {
    if items.is_empty() || items.len() > 3 { return Err("few-row group pair: groups must be 1 to 3"); }
    if !crate::staged_narrow::shape_admitted(rows, cols) { return Err("few-row group pair: shape is not narrow"); }
    let each_bytes = (inputs as u64).checked_mul(rows as u64).and_then(|w| w.checked_mul(4));
    let total_bytes = each_bytes.and_then(|b| b.checked_mul(items.len() as u64));
    if total_bytes != Some(output.bytes) || output.address.checked_add(output.bytes).is_none() { return Err("few-row group pair: output extent mismatch"); }
    let each = each_bytes.unwrap();
    let meets = |a: Span, b: Span| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
    for (g, &(w, a, b)) in items.iter().enumerate() {
        let out_g = Span { address: output.address + g as u64 * each, bytes: each };
        few_check_pair(rows, cols, inputs, w, a, b, out_g)?;
        if meets(output, w) || meets(output, a) || meets(output, b) { return Err("few-row group pair: output overlaps operand"); }
    }
    Ok(())
}

/// Host check for split-output `PLAIN_STAGED_FEW` writing directly into `1..=FEW_SPLIT_MAX` contiguous outputs.
#[cfg(not(target_arch = "nvptx64"))]
pub fn few_check_split(rows: usize, cols: usize, inputs: usize, weights: Span, input: Span, splits: &[(Span, usize)]) -> Result<(), &'static str> {
    if !few_shape_admitted(rows, cols, inputs) { return Err("few-row split: shape outside envelope"); }
    if splits.is_empty() || splits.len() > FEW_SPLIT_MAX { return Err("few-row split: bad split count"); }
    let sum: usize = splits.iter().try_fold(0usize, |acc, &(_, w)| if w > 0 { acc.checked_add(w) } else { None }).ok_or("few-row split: empty or overflowing split width")?;
    if sum != rows { return Err("few-row split: split widths differ from matrix rows"); }
    let in_bytes = (inputs as u64).checked_mul(cols as u64).and_then(|w| w.checked_mul(4));
    if weights.bytes != rows as u64 * cols as u64 * 2 || Some(input.bytes) != in_bytes { return Err("few-row split: input or weight extent mismatch"); }
    let ends = |span: Span| span.address != 0 && span.address.checked_add(span.bytes).is_some();
    if !ends(weights) || !ends(input) || (weights.address | input.address) & 15 != 0 { return Err("few-row split: bad weight or input alignment"); }
    let meets = |a: Span, b: Span| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
    for (i, &(sp, w)) in splits.iter().enumerate() {
        let want = (inputs as u64).checked_mul(w as u64).and_then(|n| n.checked_mul(4));
        if Some(sp.bytes) != want || !ends(sp) || sp.address & 3 != 0 || meets(sp, weights) || meets(sp, input) {
            return Err("few-row split: bad output span or overlap");
        }
        for &(other, _) in &splits[i + 1..] {
            if meets(sp, other) { return Err("few-row split: outputs overlap each other"); }
        }
    }
    Ok(())
}
/// The words of one launch on the host, for the device checks: `weights` are the BF16 patterns (`rows x cols`,
/// row-major), `input` is `[inputs][cols]`; the result is `[inputs][rows]`. Each word is one chain over the columns in
/// rising order from +0.0, each product rounded before its add (Rust's f32 `*` and `+` round to nearest even, as
/// `mul.rn.f32` and `add.rn.f32`).
#[cfg(not(target_arch = "nvptx64"))]
pub fn few_reference(rows: usize, cols: usize, inputs: usize, weights: &[u16], input: &[f32]) -> Vec<f32> {
    assert!(weights.len() == rows * cols && input.len() == inputs * cols, "few-row reference: the operands differ from the shape");
    let mut out = Vec::with_capacity(inputs * rows);
    for t in 0..inputs {
        let x = &input[t * cols..(t + 1) * cols];
        for r in 0..rows {
            let w = &weights[r * cols..(r + 1) * cols];
            let mut sum = 0.0f32;
            for c in 0..cols { let product = f32::from_bits(u32::from(w[c]) << 16) * x[c]; sum += product; }
            out.push(sum);
        }
    }
    out
}

#[cfg(all(test, not(target_arch = "nvptx64")))]
mod tests {
    use super::*;
    fn span(address: u64, words: u64) -> Span { Span { address, bytes: words * 4 } }
    #[test]
    fn the_shape_the_geometry_and_the_parts() {
        assert!(few_shape_admitted(2048, 2304, 8) && few_shape_admitted(1, 64, 1) && few_shape_admitted(1, 64, 16));
        assert!(!few_shape_admitted(2048, 2304, 0) && !few_shape_admitted(2048, 2304, 17) && !few_shape_admitted(2048, 2300, 8));
        assert!(!few_shape_admitted(ELEMENTS_MAX / 2304 + 1, 2304, 8) && !few_shape_admitted(8, FEW_COLS_MAX + 64, 8));
        // The draft model's down projection: 2,304 rows of 9,216 columns, the 8 rows of a draft block.
        assert!(few_shape_admitted(2304, 9216, 8) && few_shape_admitted(8, FEW_COLS_MAX, 8) && few_part_rows(2304, 9216) == 2304);
        assert_eq!(few_geometry(1, 256, 1), (1, 32));
        assert_eq!(few_geometry(256, 256, 256), (64, 32));
        assert_eq!(few_geometry(257, 256, 257), (65, 32));
        assert_eq!(few_geometry(257, 192, 257), (5, 128));
        assert_eq!(few_geometry(257, 256, 4096), (5, 128));
        assert_eq!(few_geometry(4096, 256, 4096), (64, 128));
        assert_eq!(few_geometry(4097, 256, 4097), (65, 128));
        assert!(few_admitted(4096, 2304, 8, 4096, 0x1000, 0x2000, 128));
        assert!(!few_admitted(4096, 2304, 8, 4096, 0x1008, 0x2000, 128));
        assert!(!few_admitted(4096, 2304, 8, 4096, 0x1000, 0x2000, 32));
        assert!(few_admitted(257, 256, 8, 257, 0x1000, 0x2000, 32));
        assert!(!few_admitted(257, 256, 8, 257, 0x1000, 0x2000, 128));
        // A matrix above the envelope runs in parts of whole warps inside it; the head of 200,058 rows of 2,304 columns.
        let part = few_part_rows(200_058, 2304);
        assert!(part.is_multiple_of(32) && part * 2304 <= ELEMENTS_MAX && (part + 32) * 2304 > ELEMENTS_MAX && few_shape_admitted(part, 2304, 8));
        assert_eq!(few_part_rows(4096, 2304), 4096);
    }
    #[test]
    fn the_panel_form_stages_whole_ring_turns_that_fit() {
        // The decode shapes on 48 multiprocessors: a head part (228 blocks) keeps five blocks a multiprocessor, the
        // attention stack (103 blocks) three, the output projection (36 blocks) one.
        let budget = |blocks| one_panel_budget(blocks, 48);
        assert_eq!((budget(228), budget(103), budget(36), budget(48)), (19_456, 33_109, ONE_SHARED_MAX, ONE_SHARED_MAX));
        // One row: the whole input, as the one-row form always staged it (2 x cols bytes, then the cols metadata bytes).
        assert_eq!(one_panel_part(1, 2304, budget(228)), Some(144));
        assert_eq!(one_panel_shared(1, 2304, 144), 2304 * 3);
        assert_eq!(one_panel_part(3, 2304, budget(228)), Some(144));
        // More than one row: each row padded by one tile (32 bytes).
        assert_eq!((one_panel_row_bytes(1, 48), one_panel_row_bytes(2, 48)), (1_536, 1_568));
        assert_eq!(one_panel_shared(8, 2304, 48), 8 * 1_568 + 2304);
        // Eight rows of a head part: three parts of four ring turns; of the attention stack: two parts of six.
        assert_eq!(one_panel_part(8, 2304, budget(228)), Some(48));
        assert_eq!(one_panel_part(8, 2304, budget(103)), Some(72));
        assert_eq!(one_panel_part(4, 2304, budget(228)), Some(72));
        // The output projection: one part up to five rows, then two halves (no part may pass ONE_SHARED_MAX).
        assert_eq!(one_panel_part(5, 3072, budget(36)), Some(192));
        assert_eq!(one_panel_part(8, 3072, budget(36)), Some(96));
        for (inputs, cols, blocks) in [(8, 2304, 228), (8, 3072, 36), (6, 2304, 135), (2, 256, 300)] {
            let part = one_panel_part(inputs, cols, budget(blocks)).unwrap();
            assert!((cols / 16).is_multiple_of(part) && (part == cols / 16 || part.is_multiple_of(ONE_RING)));
            assert!(one_panel_shared(inputs, cols, part) <= budget(blocks));
        }
        // Nothing fits, or the rows are outside the form: no part.
        assert_eq!(one_panel_part(8, 2304, 2304 + 8 * 11 * 32), None);
        assert_eq!((one_panel_part(0, 2304, ONE_SHARED_MAX), one_panel_part(9, 2304, ONE_SHARED_MAX)), (None, None));
        assert!(one_panel_rows_admitted(2304, 3072, 8) && !one_panel_rows_admitted(2304, 3072, 9) && !one_panel_rows_admitted(2304, 192, 2));
    }
    #[test]
    fn the_check_names_each_refusal() {
        let (rows, cols, inputs) = (256usize, 128usize, 4usize);
        let weights = span(0x10_0000, (rows * cols / 2) as u64);
        let input = span(0x20_0000, (inputs * cols) as u64);
        let output = span(0x30_0000, (inputs * rows) as u64);
        assert_eq!(few_check(rows, cols, inputs, rows, weights, input, output), Ok(()));
        // A part of a larger matrix: the output holds (inputs - 1) x stride + rows words.
        assert_eq!(few_check(rows, cols, inputs, 1000, weights, input, span(0x30_0000, (3 * 1000 + rows) as u64)), Ok(()));
        let refused = |result: Result<(), &'static str>, word: &str| assert!(result.is_err_and(|why| why.contains(word)), "{word}");
        refused(few_check(rows, cols, 0, rows, weights, span(0x20_0000, 0), span(0x30_0000, 0)), "1 to 16");
        refused(few_check(rows, cols, 17, rows, weights, input, output), "1 to 16");
        refused(few_check(rows, 100, inputs, rows, weights, input, output), "whole stages");
        refused(few_check(rows, cols, inputs, rows - 1, weights, input, output), "stride");
        refused(few_check(rows, cols, inputs, rows, span(0x10_0000, 1), input, output), "weights differ");
        refused(few_check(rows, cols, inputs, rows, weights, span(0x20_0000, 1), output), "input differs");
        refused(few_check(rows, cols, inputs, rows, weights, input, span(0x30_0000, 1)), "output differs");
        refused(few_check(rows, cols, inputs, rows, span(0, (rows * cols / 2) as u64), input, output), "null");
        refused(few_check(rows, cols, inputs, rows, span(0x10_0008, (rows * cols / 2) as u64), input, output), "weights are not");
        refused(few_check(rows, cols, inputs, rows, weights, span(0x20_0008, (inputs * cols) as u64), output), "input is not");
        refused(few_check(rows, cols, inputs, rows, weights, input, span(0x30_0002, (inputs * rows) as u64)), "word aligned");
        refused(few_check(rows, cols, inputs, rows, weights, input, span(0x20_0000, (inputs * rows) as u64)), "overlaps");
        refused(few_check(rows, cols, inputs, rows, weights, input, span(u64::MAX - 8, (inputs * rows) as u64)), "address space");
    }
    #[test]
    fn the_reference_is_the_row_product_of_each_input_row() {
        let (rows, cols, inputs) = (5usize, 128usize, 3usize);
        let weights = (0..rows * cols).map(|k| (((k * 37 + 11) % 251) as u16) << 6 | 0x3c00).collect::<Vec<_>>();
        let input = (0..inputs * cols).map(|k| ((k * 13 + 5) % 29) as f32 * 0.03125 - 0.4).collect::<Vec<_>>();
        let all = few_reference(rows, cols, inputs, &weights, &input);
        for t in 0..inputs {
            let one = few_reference(rows, cols, 1, &weights, &input[t * cols..(t + 1) * cols]);
            assert_eq!(all[t * rows..(t + 1) * rows].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), one.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
    }
}
