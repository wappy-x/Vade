//! The narrow staged BF16 row product: one launch gives the product of ONE matrix of few rows (BF16 values, two a
//! word, row-major) with one input row of FP32 values. Each output word is one chain over the columns in rising order
//! from +0.0 using BF16 MMA over successive groups of 16 columns: the same K order as the prefill product.
//!
//! The product is an entry of its own in the Inkling image (`resident_launch::Kind::PlainStagedNarrow`, kernel body
//! `kernels-inkling/src/plain_staged_narrow.rs`, Ruling R6h). A matrix of few rows leaves most multiprocessors idle in
//! the row product (one warp owns 32 rows there): GateDown [64 x 4608] runs as 2 blocks, Sink [24 x 2304] as 1. Here a
//! one-warp block owns `ROWS_A_BLOCK` rows and all 32 lanes cooperate on the MMA. So the
//! same matrix runs as 16 blocks (GateDown) or 6 (Sink), each with its own bytes in flight; the time of a block is
//! close to the chain of one row.
//!
//! This file holds the geometry, the admission and the host check of a launch: pure functions shared by the kernel
//! body and the host, so one predicate decides both. The kernel crate includes this file by path. On the device it
//! calls no function of another file; on the host its check adds the narrow shape to `staged_product::check`.

/// Four weight rows per block (`m16n8k16` with `A = X`, `B = W`).
pub const ROWS_A_BLOCK: usize = 4;
/// Threads of a block: one warp with an ascending-K chain.
pub const THREADS: usize = 32;
/// Columns of one stage, and stages in flight.
pub const STAGE_COLS: usize = 256;
pub const STAGES: usize = 4;
/// The envelope: at most this many rows (above it the row product fills the device) and this many columns.
pub const ROWS_MAX: usize = 512;
pub const COLS_MAX: usize = 8192;
/// A shape the narrow product takes: 1 to 512 rows, whole stages of 256 columns up to 8192.
#[inline(always)]
pub const fn shape_admitted(rows: usize, cols: usize) -> bool {
    rows >= 1 && rows <= ROWS_MAX && cols >= STAGE_COLS && cols <= COLS_MAX && cols % STAGE_COLS == 0
}
/// Blocks of a launch: one block a group of `ROWS_A_BLOCK` rows.
#[inline(always)]
pub const fn blocks(rows: usize) -> u32 { rows.div_ceil(ROWS_A_BLOCK) as u32 }
/// THE GROUPED FORM (seat W-DECODE-SGLANG-3): one launch gives the products of up to `GROUPS_MAX` matrices of the
/// same shape, each with its own input row, into one output `[groups][rows]`. Each word is the word of the product of
/// its own matrix alone (the same chain), so a group changes no word; it saves the launches, and the chains of the
/// matrices run at once (a chain-bound product: GateDown of the three attention gates of a loop visit).
/// The command: word 5 is the count of matrices; matrix `g >= 1` has its weights in word `4 + 2 g` and its input in
/// word `5 + 2 g` (matrix 0 in words 0 and 1, the output in word 2, as the one-matrix form).
pub const GROUPS_MAX: usize = 4;
/// The command word of the weights (`input` false) or of the input (`input` true) of matrix `group`.
#[inline(always)]
pub const fn group_word(group: usize, input: bool) -> usize { if group == 0 { input as usize } else { 4 + 2 * group + input as usize } }
/// Blocks of a launch of `groups` matrices: the blocks of one matrix for each, matrix after matrix.
#[inline(always)]
pub const fn group_blocks(rows: usize, groups: usize) -> u32 { blocks(rows) * groups as u32 }
/// A count of matrices the grouped form takes.
#[inline(always)]
pub const fn groups_admitted(groups: usize) -> bool { groups >= 1 && groups <= GROUPS_MAX }
/// Payload word holding optional flags for `PLAIN_STAGED_NARROW` (word 12, after 4 groups in words 0..12).
pub const NARROW_FLAGS_WORD: usize = 12;
/// Fuse `rn()` BF16 rounding into the output store of `PLAIN_STAGED_NARROW`.
pub const NARROW_ROUND_OUT: u64 = 1;
/// Dual-input `[input1, input2]` (`cols / 2` FP32 words each; requires `cols % (2 * STAGE_COLS) == 0`).
/// `input2` for group `g` sits at `NARROW_INPUT2_WORD + g`.
pub const NARROW_DUAL_INPUT: u64 = 2;
pub const NARROW_INPUT2_WORD: usize = 13;
const _: () = assert!(group_word(GROUPS_MAX - 1, true) < NARROW_FLAGS_WORD && (NARROW_INPUT2_WORD + GROUPS_MAX) * 8 <= 1024);
/// The rule of the kernel body: an admitted shape in one-warp blocks, with 16-byte aligned weights and input. The
/// body asks it before its first read. The host check below holds for every recorded launch.
#[inline(always)]
pub const fn admitted(rows: usize, cols: usize, weights: usize, input: usize, threads: usize) -> bool {
    shape_admitted(rows, cols) && threads == THREADS && (weights | input) & 15 == 0
}
/// Host check for grouped dual-input narrow products (`[input1, input2]`, each `cols / 2` words).
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_group_pair(rows: usize, cols: usize, items: &[(crate::staged_product::Span, crate::staged_product::Span, crate::staged_product::Span)], output: crate::staged_product::Span) -> Result<(), &'static str> {
    use crate::staged_product::Span;
    if !groups_admitted(items.len()) { return Err("staged narrow pair: a group holds 1 to 4 matrices"); }
    if !shape_admitted(rows, cols) || !cols.is_multiple_of(2 * STAGE_COLS) {
        return Err("staged narrow pair: shape outside dual-input envelope");
    }
    let part = rows as u64 * 4;
    let w_bytes = rows as u64 * cols as u64 * 2;
    let half_bytes = (cols / 2) as u64 * 4;
    if output.bytes != part * items.len() as u64 || output.address == 0 || output.address & 3 != 0 || output.address.checked_add(output.bytes).is_none() {
        return Err("staged narrow pair: bad output span");
    }
    let meets = |a: Span, b: Span| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
    for &(w, x1, x2) in items {
        let valid = |s: Span, want: u64| s.bytes == want && s.address != 0 && s.address & 15 == 0 && s.address.checked_add(s.bytes).is_some();
        if !valid(w, w_bytes) || !valid(x1, half_bytes) || !valid(x2, half_bytes) {
            return Err("staged narrow pair: operand extent or alignment mismatch");
        }
        if meets(output, w) || meets(output, x1) || meets(output, x2) {
            return Err("staged narrow pair: output overlaps an operand");
        }
    }
    Ok(())
}
/// The host check of one launch: the narrow shape, then every check of the row product (`staged_product::check`):
/// the weights hold exactly `rows x cols` BF16 values, the input exactly `cols` words, the output exactly `rows`
/// words that meet no operand, nothing null or past the address space, the weights and the input 16-byte aligned.
/// So every 16-byte copy of the body (8 values of a row below `rows`, or 4 input words, from a column below `cols`
/// that is a multiple of 8 or 4) is inside its operand, and each store of a row below `rows` is inside the output.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check(rows: usize, cols: usize, weights: crate::staged_product::Span, input: crate::staged_product::Span, output: crate::staged_product::Span) -> Result<(), &'static str> {
    if !shape_admitted(rows, cols) { return Err("staged narrow product: the shape is outside the narrow envelope (1 to 512 rows, whole stages of 256 columns up to 8192)"); }
    crate::staged_product::check(rows, cols, weights, input, output)
}
/// The host check of one launch of the grouped form: 1 to `GROUPS_MAX` matrices, as many inputs, an output of exactly
/// `groups x rows` words. Matrix `g` with its input and its part of the output (`rows` words from word `g x rows`)
/// passes `check`, and the whole output meets no matrix and no input of any group. So every read of block `b` (matrix
/// `b / blocks(rows)`) is inside that matrix's operands, and its stores land in that matrix's part of the output.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_group(rows: usize, cols: usize, weights: &[crate::staged_product::Span], inputs: &[crate::staged_product::Span], output: crate::staged_product::Span) -> Result<(), &'static str> {
    use crate::staged_product::Span;
    if !groups_admitted(weights.len()) || inputs.len() != weights.len() { return Err("staged narrow product: a group holds 1 to 4 matrices, each with one input"); }
    if !shape_admitted(rows, cols) { return Err("staged narrow product: the shape is outside the narrow envelope (1 to 512 rows, whole stages of 256 columns up to 8192)"); }
    let part = rows as u64 * 4;
    if output.bytes != part * weights.len() as u64 { return Err("staged narrow product: the output differs from the rows of the matrices of the group"); }
    let meets = |a: Span, b: Span| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
    for (g, (&w, &x)) in weights.iter().zip(inputs).enumerate() {
        let out = Span { address: output.address.checked_add(part * g as u64).ok_or("staged narrow product: the output ends above the address space")?, bytes: part };
        crate::staged_product::check(rows, cols, w, x, out)?;
        if meets(output, w) || meets(output, x) { return Err("staged narrow product: the output overlaps an operand"); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staged_product::Span;
    fn span(address: u64, bytes: u64) -> Span { Span { address, bytes } }
    #[test]
    fn the_narrow_product_takes_the_few_row_matrices_of_the_model() {
        for (rows, cols) in [(64usize, 4608usize), (128, 2304), (24, 2304), (384, 2304), (1, 256), (512, 8192)] {
            assert!(shape_admitted(rows, cols), "{rows}x{cols}");
            assert_eq!(blocks(rows) as usize, rows.div_ceil(ROWS_A_BLOCK), "{rows}x{cols}");
            for threads in [0usize, 32, 64, 128] { for weights in [256usize, 8] { for input in [512usize, 4] {
                assert_eq!(admitted(rows, cols, weights, input, threads), threads == THREADS && (weights | input) & 15 == 0, "{rows}x{cols} {threads}");
            } } }
        }
        // Not narrow: more rows, columns that are no whole stages of 256 (the gate pairs, 64 columns), too many columns.
        for (rows, cols) in [(513usize, 2304usize), (1024, 2304), (2304, 64), (64, 4672), (64, 8448), (0, 2304), (64, 0)] {
            assert!(!shape_admitted(rows, cols) && !admitted(rows, cols, 256, 512, THREADS), "{rows}x{cols}");
        }
    }
    #[test]
    fn the_host_check_takes_a_good_launch_and_names_each_refusal() {
        let (rows, cols) = (64usize, 4608usize);
        let (weights, input, output) = (span(1 << 20, (rows * cols * 2) as u64), span(1 << 30, cols as u64 * 4), span(1 << 31, rows as u64 * 4));
        assert_eq!(check(rows, cols, weights, input, output), Ok(()));
        let refused = |result: Result<(), &'static str>, text: &str| assert!(result.is_err_and(|error| error.contains(text)), "{result:?} lacks {text}");
        refused(check(513, cols, span(1 << 20, 513 * cols as u64 * 2), input, span(1 << 31, 513 * 4)), "outside the narrow envelope");
        refused(check(rows, 4672, span(1 << 20, rows as u64 * 4672 * 2), span(1 << 30, 4672 * 4), output), "outside the narrow envelope");
        refused(check(rows, cols, span(1 << 20, (rows * cols * 2 - 2) as u64), input, output), "the weights differ");
        refused(check(rows, cols, weights, span(1 << 30, cols as u64 * 4 - 4), output), "the input differs");
        refused(check(rows, cols, weights, input, span(1 << 31, rows as u64 * 4 + 4)), "the output differs");
        refused(check(rows, cols, span(0, weights.bytes), input, output), "an operand is null");
        refused(check(rows, cols, span(u64::MAX - 15, weights.bytes), input, output), "ends above the address space");
        refused(check(rows, cols, span((1 << 20) + 8, weights.bytes), input, output), "not 16-byte aligned");
        refused(check(rows, cols, weights, span((1 << 30) + 4, input.bytes), output), "not 16-byte aligned");
        refused(check(rows, cols, weights, input, span((1 << 20) + 64, output.bytes)), "overlaps an operand");
        // Review REVNARROW: the shape refusals through check, the output alignment, a null input or output.
        refused(check(0, cols, span(1 << 20, 0), input, span(1 << 31, 0)), "outside the narrow envelope");
        refused(check(rows, 0, span(1 << 20, 0), span(1 << 30, 0), output), "outside the narrow envelope");
        refused(check(rows, 8448, span(1 << 20, rows as u64 * 8448 * 2), span(1 << 30, 8448 * 4), output), "outside the narrow envelope");
        refused(check(rows, cols, weights, input, span((1 << 31) + 2, output.bytes)), "not word aligned");
        refused(check(rows, cols, weights, span(0, input.bytes), output), "an operand is null");
        refused(check(rows, cols, weights, input, span(0, output.bytes)), "an operand is null");
    }
    #[test]
    fn the_grouped_form_takes_up_to_four_matrices_and_names_each_refusal() {
        let (rows, cols) = (64usize, 4608usize);
        let (bytes, x) = ((rows * cols * 2) as u64, cols as u64 * 4);
        let w = |g: u64| span((1 << 20) + g * (1 << 22), bytes);
        let i = |g: u64| span((1 << 30) + g * (1 << 16), x);
        let out = |groups: u64| span(1 << 31, rows as u64 * 4 * groups);
        for groups in 1..=GROUPS_MAX as u64 {
            let (ws, xs) = ((0..groups).map(w).collect::<Vec<_>>(), (0..groups).map(i).collect::<Vec<_>>());
            assert_eq!(check_group(rows, cols, &ws, &xs, out(groups)), Ok(()), "{groups}");
            assert_eq!(group_blocks(rows, groups as usize), 16 * groups as u32);
        }
        // The command words: matrix 0 in words 0 and 1, matrix g in words 4 + 2g and 5 + 2g (after the count, word 5).
        assert_eq!([group_word(0, false), group_word(0, true), group_word(1, false), group_word(1, true), group_word(3, true)], [0, 1, 6, 7, 11]);
        assert!(group_word(GROUPS_MAX - 1, true) * 8 + 8 <= 1024 && !groups_admitted(0) && !groups_admitted(5));
        let refused = |result: Result<(), &'static str>, text: &str| assert!(result.is_err_and(|error| error.contains(text)), "{result:?} lacks {text}");
        let (ws, xs) = ((0..3).map(w).collect::<Vec<_>>(), (0..3).map(i).collect::<Vec<_>>());
        refused(check_group(rows, cols, &[], &[], out(0)), "1 to 4 matrices");
        refused(check_group(rows, cols, &[w(0); 5], &[i(0); 5], out(5)), "1 to 4 matrices");
        refused(check_group(rows, cols, &ws, &xs[..2], out(3)), "1 to 4 matrices");
        refused(check_group(rows, cols, &ws, &xs, out(2)), "the output differs");
        refused(check_group(513, cols, &ws, &xs, span(1 << 31, 513 * 4 * 3)), "outside the narrow envelope");
        refused(check_group(rows, cols, &[w(0), span(w(1).address, bytes - 2), w(2)], &xs, out(3)), "the weights differ");
        refused(check_group(rows, cols, &ws, &[i(0), i(1), span(i(2).address + 4, x)], out(3)), "not 16-byte aligned");
        // The output of a later matrix must not meet an operand of an earlier one, and the reverse.
        refused(check_group(rows, cols, &ws, &[i(0), span(out(3).address + rows as u64 * 8, x), i(2)], out(3)), "overlaps an operand");
        refused(check_group(rows, cols, &[span(out(3).address + 256, bytes), w(1), w(2)], &xs, out(3)), "overlaps an operand");
    }
}
