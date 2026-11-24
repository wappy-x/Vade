//! The narrow staged BF16 row product: one launch gives the product of ONE matrix of few rows (BF16 values, two a
//! word, row-major) with one input row of FP32 values. Each output word is one chain over the columns in rising order
//! from +0.0, each product rounded by its own multiply before its add, no fused multiply-add: the words of the staged
//! row product (`staged_product`, kind 90) and of `project_bf16_ordered`.
//!
//! The product is an entry of its own in the Inkling image (`resident_launch::Kind::PlainStagedNarrow`, kernel body
//! `kernels-inkling/src/plain_staged_narrow.rs`, Ruling R6h). A matrix of few rows leaves most multiprocessors idle in
//! the row product (one warp owns 32 rows there): GateDown [64 x 4608] runs as 2 blocks, Sink [24 x 2304] as 1. Here a
//! one-warp block owns `ROWS_A_BLOCK` rows: all 32 lanes copy the stages, lane `r` runs the chain of row `r`. So the
//! same matrix runs as 16 blocks (GateDown) or 6 (Sink), each with its own bytes in flight; the time of a block is
//! close to the chain of one row.
//!
//! This file holds the geometry, the admission and the host check of a launch: pure functions shared by the kernel
//! body and the host, so one predicate decides both. The kernel crate includes this file by path. On the device it
//! calls no function of another file; on the host its check adds the narrow shape to `staged_product::check`.

/// Rows of one block: the lanes that run a chain.
pub const ROWS_A_BLOCK: usize = 4;
/// Threads of a block: one warp.
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
/// The rule of the kernel body: an admitted shape in one-warp blocks, with 16-byte aligned weights and input. The
/// body asks it before its first read. The host check below holds for every recorded launch.
#[inline(always)]
pub const fn admitted(rows: usize, cols: usize, weights: usize, input: usize, threads: usize) -> bool {
    shape_admitted(rows, cols) && threads == THREADS && (weights | input) & 15 == 0
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staged_product::Span;
    fn span(address: u64, bytes: u64) -> Span { Span { address, bytes } }
    #[test]
    fn the_narrow_product_takes_the_few_row_matrices_of_the_model() {
        for (rows, cols) in [(64usize, 4608usize), (128, 2304), (24, 2304), (384, 2304), (1, 256), (512, 8192)] {
            assert!(shape_admitted(rows, cols), "{rows}x{cols}");
            assert_eq!(blocks(rows) as usize, rows.div_ceil(4), "{rows}x{cols}");
            for threads in [0usize, 32, 64, 128] { for weights in [256usize, 8] { for input in [512usize, 4] {
                assert_eq!(admitted(rows, cols, weights, input, threads), threads == 32 && (weights | input) & 15 == 0, "{rows}x{cols} {threads}");
            } } }
        }
        // Not narrow: more rows, columns that are no whole stages of 256 (the gate pairs, 64 columns), too many columns.
        for (rows, cols) in [(513usize, 2304usize), (1024, 2304), (2304, 64), (64, 4672), (64, 8448), (0, 2304), (64, 0)] {
            assert!(!shape_admitted(rows, cols) && !admitted(rows, cols, 256, 512, 32), "{rows}x{cols}");
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
}
