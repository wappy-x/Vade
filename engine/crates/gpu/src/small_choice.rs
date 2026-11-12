//! Which small-count kernel runs a prompt row operation or a group launch, as pure functions of the shape, so the ORDER of the choice is tested on the
//! host (`prompt_operation` and `expert_group` call these first and fall back to lane K's and lane R's tiles, then to the old kernels, when they return None).
//! Each kernel's own predicate is in its module (`plain_small`, `rows_small`, `nv_small`, `fp8_small`); the order here is what decides,
//! for a shape that two kernels admit, which one runs.
use crate::prompt_ops::{Op, Shape};
use crate::{fp8_rows, fp8_small, mx_rows, mx_small, nv_rows, nv_small, plain_small, rows_small};

/// Entry name, blocks, threads of a launch.
pub type Launch = (&'static str, u32, u32);

/// The small-count kernel for a prompt row operation, if one is admitted. Tried before the tiles of lane K and lane R.
/// Order: plain rows (`plain_rows_few` for at most 4 tokens and 512 rows, else `plain_rows_small`), then StepRows (their operations differ, so at most one admits a shape). Wrapped Prepare, GatedNorm and Route are NOT here: lane P's
/// `prompt_prepare`, `prompt_gated_norm` and `prompt_route` (prompt_admission::fast_kernel) are chosen for them.
pub fn prompt(shape: &Shape, input: u64, weight: u64) -> Option<Launch> {
    let (rows, tokens, cols) = (shape.rows as usize, shape.tokens as usize, shape.cols as usize);
    if shape.op == Op::ProjectRows && plain_small::few_admitted(rows, tokens, cols, input, weight) {
        return Some(("plain_rows_few", plain_small::few_blocks(rows)?, plain_small::threads(tokens)));
    }
    if shape.op == Op::ProjectRows && plain_small::admitted(rows, tokens, cols, input, weight) {
        return Some(("plain_rows_small", plain_small::blocks(rows)?, plain_small::threads(tokens)));
    }
    if rows_small::admitted(shape) { let (blocks, threads) = rows_small::geometry(shape); return Some(("prompt_rows_small", blocks, threads)); }
    None
}

/// The kernel for an MXFP8 projection of `tokens` rows (`ResidentMxfp8Projection::apply_rows`, FP32 input, round 0): addresses are the device addresses of the
/// weights, the input and the output.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Mx {
    /// `mx_rows_small2`, `mx_rows_small4`, `mx_rows_small8` or `mx_rows_small8w` (entry name, tokens a block): 2 to 8 tokens of the listed shapes.
    Small(&'static str, usize),
    /// `mx_rows_tile`, in pieces of at most `mx_rows::MAX_TOKENS` tokens.
    Tile,
    /// The old M kernels (1 token, and every call neither of the others admits).
    Old,
}
/// Order: the small entries FIRST (their predicate holds its own alignment checks and the sweep lists: 2 to 8 tokens),
/// then the tile for 2 or more tokens (12288 columns since `mx_rows::DENSE_COLS`), then the old kernels. One token always keeps the old M1 kernel.
pub fn mx(rows: usize, cols: usize, tokens: usize, weights: u64, input: u64, output: u64) -> Mx {
    if mx_small::admitted(rows, cols, tokens, weights, input, output) { if let Some((entry, per, _)) = mx_small::entry(tokens, cols) { return Mx::Small(entry, per); } }
    if tokens != 1 && mx_rows::admitted(rows, cols, tokens.min(mx_rows::token_limit(rows,cols)), 0, 0, weights, input, output) { return Mx::Tile; }
    Mx::Old
}

/// What the group launch is made of, as `expert_group` knows it from its table.
pub struct Group<'a> {
    pub nv: &'a [nv_rows::Part],
    pub fp8: &'a [fp8_rows::Part],
    /// Every part is NvRows with 16-column blocks.
    pub all_nv: bool,
    /// Every part is Fp8Rows.
    pub all_fp8: bool,
    /// Output words a unit (the kernels need 1).
    pub words: u32,
    /// The parts' first input rows are cumulative (`nv_small::contiguous`).
    pub cumulative: bool,
    /// Two or more parts, all on ONE first input row (matrices that read the same rows).
    pub shared_rows: bool,
    pub input: u64,
}

/// The small-count kernel for a group launch: entry, blocks, threads, dynamic shared bytes. Tried before lane K's `nv_rows_tile` and `fp8_rows_tile`.
/// Order: NVFP4 (`nv_rows_small`); FP8 with every part at most 2 tokens, or one part of 3 or 4 tokens over at most 768 rows (`fp8_rows_small`); FP8 with a part of 3 or more tokens (`fp8_rows_multi`).
/// ONE part of 3 or 4 tokens over more than 768 rows takes `fp8_rows_few` where `fp8_small::few_wins` (up to 4 MB of weights), else `fp8_rows_multi`.
/// All need one output word a unit and a cumulative table (the tables their sweeps run).
/// SHARED ROWS: FP8 parts of one shape that all read the same rows (`fp8_small::shared_admitted`) take ONE `fp8_rows_multi` launch; no other kernel here takes a table that is not cumulative.
pub fn group(g: &Group) -> Option<(&'static str, u32, u32, u32)> {
    if g.words != 1 { return None; }
    if !g.cumulative {
        if g.shared_rows && g.all_fp8 && fp8_small::shared_admitted(g.fp8, g.input) { return Some(("fp8_rows_multi", fp8_small::multi_blocks(g.fp8)?, fp8_small::MULTI_THREADS, 0)); }
        return None;
    }
    if g.all_nv && nv_small::admitted(g.nv, g.input) { return Some(("nv_rows_small", nv_small::blocks(g.nv)?, nv_small::THREADS, nv_small::SHARED)); }
    if g.all_fp8 {
        let longest = g.fp8.iter().map(|p| p.tokens).max().unwrap_or(0);
        // One part of 3 or 4 tokens over at most FP8_SMALL_RULE_ROWS rows (the model's q_a 768 x 4096 and kv_a 256 x 4096): fp8_rows_small is faster than fp8_rows_multi there
        // (29 against 51 and 23 against 49 us at 3 tokens; one Spark 07:35, kernel-bench --fp8-small time), because multi has one block for 32 rows of the part.
        let few_rows = g.fp8.len() == 1 && (3..=4).contains(&longest) && g.fp8[0].rows <= fp8_small::SMALL_RULE_ROWS;
        if (longest <= 2 || few_rows) && fp8_small::admitted(g.fp8, g.input) { return Some(("fp8_rows_small", fp8_small::blocks(g.fp8)?, fp8_small::THREADS, fp8_small::SHARED)); }
        if (3..=4).contains(&longest) && fp8_small::few_wins(g.fp8[0].rows, g.fp8[0].cols, longest) && fp8_small::few_admitted(g.fp8, g.input) { return Some(("fp8_rows_few", fp8_small::few_blocks(g.fp8)?, fp8_small::FEW_THREADS, 0)); }
        if longest >= 3 && fp8_small::multi_admitted(g.fp8, g.input) { return Some(("fp8_rows_multi", fp8_small::multi_blocks(g.fp8)?, fp8_small::MULTI_THREADS, 0)); }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    fn project(rows: u32, tokens: usize, cols: u32) -> Shape { let mut s = Shape::new(Op::ProjectRows, tokens).unwrap(); s.rows = rows; s.cols = cols; s }
    fn wrapped(op: crate::causal_ops::Op, tokens: usize) -> Shape { Shape::wrapped(crate::causal_ops::Shape { op, heads: 64, dim: 128, taps: 4, rows: 288, cols: 8, epsilon: 1e-5, bound: -5.0 }, tokens).unwrap() }
    fn step(tokens: usize) -> Shape { let mut s = Shape::new(Op::StepRows, tokens).unwrap(); s.heads = 64; s.dim = 128; s }
    #[test]
    fn a_prompt_operation_takes_its_small_kernel_before_any_tile() {
        // A plain shape both `plain_small` and lane K's tile admit goes to plain_small, for every token count of the small range, and to nobody above it.
        for tokens in 1..=16 {
            let want = if tokens <= 4 { ("plain_rows_few", 8, 128) } else { ("plain_rows_small", 2, 128) };
            assert_eq!(prompt(&project(64, tokens, 4096), 256, 512), Some(want), "tokens {tokens}"); assert!(crate::plain_tile::admitted(64, tokens, 4096, 256, 512));
        }
        // The edge of the few rule: 512 rows and 4 tokens are few; 1000 rows or 5 tokens keep plain_rows_small.
        assert_eq!(prompt(&project(512, 4, 4096), 256, 512), Some(("plain_rows_few", 64, 128)));
        assert_eq!(prompt(&project(1000, 4, 4096), 256, 512), Some(("plain_rows_small", 32, 128)));
        assert_eq!(prompt(&project(8192, 3, 128), 256, 512), Some(("plain_rows_small", 256, 128)));
        assert_eq!(prompt(&project(512, 5, 4096), 256, 512), Some(("plain_rows_small", 16, 128)));
        assert_eq!(prompt(&project(64, 17, 4096), 256, 512), None);
        assert_eq!(prompt(&project(64, 4, 4096), 8, 512), None);
        assert_eq!(prompt(&project(130, 4, 4096), 256, 512), None, "130 rows is not in the list");
        for tokens in 1..=16 {
            for op in [crate::causal_ops::Op::Route, crate::causal_ops::Op::Prepare, crate::causal_ops::Op::GatedNorm] { assert_eq!(prompt(&wrapped(op, tokens), 0, 0), None, "lane P's kernels own the wrapped operations"); }
            assert_eq!(prompt(&step(tokens), 0, 0), Some(("prompt_rows_small", 256, 32)), "StepRows takes ours before lane R's step_rows_block");
        }
        assert_eq!(prompt(&step(17), 0, 0), None);
        assert_eq!(prompt(&step(0), 0, 0), None);
        assert_eq!(prompt(&Shape::new(Op::Transpose, 4).unwrap(), 0, 0), None);
    }
    fn nv(rows: usize, cols: usize, tokens: &[usize]) -> Vec<nv_rows::Part> { tokens.iter().map(|&t| nv_rows::Part { rows, cols, tokens: t, weight: 256, scales: 512 }).collect() }
    fn fp8(rows: usize, cols: usize, tokens: &[usize]) -> Vec<fp8_rows::Part> { tokens.iter().map(|&t| fp8_rows::Part { rows, cols, block_rows: 128, block_cols: 128, tokens: t, weight: 256, scales: 512 }).collect() }
    fn pick(nv: &[nv_rows::Part], fp8: &[fp8_rows::Part], all_nv: bool, all_fp8: bool, words: u32, cumulative: bool) -> Option<&'static str> {
        group(&Group { nv, fp8, all_nv, all_fp8, words, cumulative, shared_rows: false, input: 256 }).map(|c| c.0)
    }
    #[test]
    fn a_group_launch_takes_its_small_kernel_before_the_tiles_and_only_on_the_tables_it_was_swept_on() {
        // NVFP4: small parts go to nv_rows_small even where lane K's tile would also admit them (dense enough), never when a part is over 16 tokens.
        let dense = nv(1024, 4096, &[2; 64]);
        assert!(nv_rows::admitted(&dense, 256, 8) && pick(&dense, &[], true, false, 1, true) == Some("nv_rows_small"));
        let hot = { let mut v = nv(1024, 4096, &[1; 30]); v.extend(nv(1024, 4096, &[16, 11, 5])); v };
        assert_eq!(pick(&hot, &[], true, false, 1, true), Some("nv_rows_small"));
        assert_eq!(pick(&nv(1024, 4096, &[3; 40]), &[], true, false, 1, true), None, "mean over 2: lane K's tile");
        assert_eq!(pick(&nv(1024, 4096, &[17]), &[], true, false, 1, true), None);
        // FP8: parts of 1 to 2 tokens take fp8_rows_small, 3 to 16 fp8_rows_multi, over 16 neither (lane K's tile).
        for tokens in 1..=16usize {
            let want = if tokens <= 2 { "fp8_rows_small" } else if tokens <= 4 { "fp8_rows_few" } else { "fp8_rows_multi" };
            assert_eq!(pick(&[], &fp8(1024, 4096, &[tokens]), false, true, 1, true), Some(want), "tokens {tokens}");
            assert!(fp8_rows::admitted(&fp8(1024, 4096, &[tokens]), 256), "lane K's tile admits it too: the order decides");
        }
        assert_eq!(pick(&[], &fp8(1024, 4096, &[17]), false, true, 1, true), None);
        assert_eq!(pick(&[], &fp8(1024, 4096, &[1, 2, 2, 1]), false, true, 1, true), Some("fp8_rows_small"));
        assert_eq!(pick(&[], &fp8(1024, 4096, &[1, 2, 3, 1]), false, true, 1, true), Some("fp8_rows_multi"), "one part of 3 tokens among several decides multi: few is for ONE part");
        // 3 or 4 tokens over more than 768 rows: fp8_rows_few up to 4 MB of weights, else multi; 5 tokens and more: multi.
        for (rows, cols, tokens, want) in [(1024, 4096, 3, "fp8_rows_few"), (2048, 2048, 3, "fp8_rows_few"), (2048, 16384, 3, "fp8_rows_multi"), (8192, 1536, 3, "fp8_rows_multi"),
            (1024, 4096, 4, "fp8_rows_few"), (2048, 2048, 4, "fp8_rows_few"), (2048, 16384, 4, "fp8_rows_multi"), (8192, 1536, 4, "fp8_rows_multi"), (1024, 4096, 5, "fp8_rows_multi"), (768, 4096, 4, "fp8_rows_small")] {
            assert_eq!(pick(&[], &fp8(rows, cols, &[tokens]), false, true, 1, true), Some(want), "{rows}x{cols} {tokens} tokens");
        }
        assert_eq!(pick(&[], &fp8(1024, 4096, &[4, 1, 3, 2]), false, true, 1, true), Some("fp8_rows_multi"), "several parts keep multi: few is admitted for one part only");
        // One part of 3 or 4 tokens over at most 768 rows: the small kernel (q_a, kv_a); 5 tokens or several parts of 768 rows keep multi (1024 rows: few, below).
        for (rows, tokens, want) in [(768, 3, "fp8_rows_small"), (768, 4, "fp8_rows_small"), (288, 3, "fp8_rows_small"), (288, 4, "fp8_rows_small"), (768, 5, "fp8_rows_multi"), (1024, 3, "fp8_rows_few"), (1024, 4, "fp8_rows_few"),
            (256, 1, "fp8_rows_small"), (256, 2, "fp8_rows_small"), (256, 3, "fp8_rows_small"), (256, 4, "fp8_rows_small"), (256, 5, "fp8_rows_multi"), (256, 8, "fp8_rows_multi"), (256, 16, "fp8_rows_multi")] {
            assert_eq!(pick(&[], &fp8(rows, 4096, &[tokens]), false, true, 1, true), Some(want), "{rows} rows {tokens} tokens");
        }
        assert_eq!(pick(&[], &[fp8(768, 4096, &[3])[0], fp8(768, 4096, &[3])[0]], false, true, 1, true), Some("fp8_rows_multi"), "two parts keep multi");
        // The shapes of the FP8 list on launches with more than 12 parts: only the small ones (item 1 of the review).
        assert_eq!(pick(&[], &fp8(288, 4096, &[1; 100]), false, true, 1, true), Some("fp8_rows_small"));
        assert_eq!(pick(&[], &fp8(2048, 4096, &[1; 12]), false, true, 1, true), Some("fp8_rows_small"));
        assert_eq!(pick(&[], &fp8(2048, 4096, &[1; 13]), false, true, 1, true), None);
        assert_eq!(pick(&[], &fp8(2048, 4096, &[3; 13]), false, true, 1, true), None);
        // SHARED ROWS: 2 to 4 FP8 parts of one shape, block shape and count on the SAME rows (not cumulative) take ONE fp8_rows_multi launch with a block
        // for each 32 rows of each part, for every count of 1 to 16; any other table that is not cumulative keeps the old kernels.
        let shared = |fp8: &[fp8_rows::Part], all_fp8: bool, words: u32, shared_rows: bool| group(&Group { nv: &[], fp8, all_nv: false, all_fp8, words, cumulative: false, shared_rows, input: 256 });
        for tokens in 1..=16usize { for parts in 2..=4usize {
            assert_eq!(shared(&fp8(1024, 4096, &vec![tokens; parts]), true, 1, true), Some(("fp8_rows_multi", (32 * parts) as u32, 128, 0)), "{parts} parts of {tokens} tokens");
        } }
        assert_eq!(shared(&fp8(1024, 4096, &[4, 4]), true, 1, false), None, "the caller did not say the rows are the same");
        assert_eq!(shared(&fp8(1024, 4096, &[4; 5]), true, 1, true), None, "5 parts");
        assert_eq!(shared(&fp8(1024, 4096, &[4, 3]), true, 1, true), None, "counts differ");
        assert_eq!(shared(&fp8(1024, 4096, &[17, 17]), true, 1, true), None, "17 tokens");
        assert_eq!(shared(&fp8(1024, 4096, &[4, 4]), false, 1, true), None, "not all FP8");
        assert_eq!(shared(&fp8(1024, 4096, &[4, 4]), true, 2, true), None, "two output words a unit");
        assert_eq!(shared(&[fp8(1024, 4096, &[4])[0], fp8(2048, 4096, &[4])[0]], true, 1, true), None, "shapes differ");
        // Tables the sweeps did not run: not cumulative, or more than one output word a unit: the old kernels, for every one of the three.
        for tokens in [1usize, 2, 3, 16] { assert_eq!(pick(&[], &fp8(1024, 4096, &[tokens]), false, true, 1, false), None, "fp8 not cumulative, {tokens}"); }
        assert_eq!(pick(&dense, &[], true, false, 1, false), None);
        assert_eq!(pick(&dense, &[], true, false, 2, true), None);
        assert_eq!(pick(&[], &fp8(1024, 4096, &[3]), false, true, 2, true), None);
        // A launch that is neither all NVFP4 nor all FP8 (a quantize part among them) keeps the old path.
        assert_eq!(pick(&dense, &fp8(1024, 4096, &[3]), false, false, 1, true), None);
    }
    #[test]
    fn an_mxfp8_projection_takes_the_small_entry_first_then_the_tile_then_the_old_kernels() {
        let w = 256; let i = 512; let o = 4;
        // 1 token old, 2 small2, 3 and 4 small4, 5 to 8 small8 (small8w past 4,096 columns), 9 and more tile.
        for (cols, rows) in [(8192, 2048), (4096, 4096), (4096, 6144)] {
            let eight = if cols > 4096 { "mx_rows_small8w" } else { "mx_rows_small8" };
            let got: Vec<Mx> = [1, 2, 3, 4, 5, 8, 9, 16, 17].iter().map(|&t| mx(rows, cols, t, w, i, o)).collect();
            assert_eq!(got, [Mx::Old, Mx::Small("mx_rows_small2", 2), Mx::Small("mx_rows_small4", 4), Mx::Small("mx_rows_small4", 4), Mx::Small(eight, 8), Mx::Small(eight, 8), Mx::Tile, Mx::Tile, Mx::Tile], "{rows}x{cols}");
        }
        // 12,288 columns (the model's down projection, `mx_rows::DENSE_COLS`): 2 to 8 tokens reach the small entries, 9 or more take the tile
        // (they ran the old 16-token kernels until the device sweep covered this column count). Its neighbours keep the old kernels.
        let got: Vec<Mx> = [1, 2, 3, 4, 5, 8, 9, 16, 17, 256, 1024].iter().map(|&t| mx(2048, 12288, t, w, i, o)).collect();
        assert_eq!(got, [Mx::Old, Mx::Small("mx_rows_small2", 2), Mx::Small("mx_rows_small4", 4), Mx::Small("mx_rows_small4", 4), Mx::Small("mx_rows_small8w", 8), Mx::Small("mx_rows_small8w", 8), Mx::Tile, Mx::Tile, Mx::Tile, Mx::Tile, Mx::Tile]);
        for cols in [12256, 12320, 10240, 16384] { assert_eq!(mx(2048, cols, 16, w, i, o), Mx::Old, "{cols}"); }
        // Every swept (rows, cols) takes a small entry at 2 to 8 tokens: no listed shape is dead behind the tile's limits.
        for &rows in crate::small_sweep::MX_ROWS { for &cols in crate::small_sweep::MX_COLS {
            for t in 2..=8 { assert_eq!(matches!(mx(rows, cols, t, w, i, o), Mx::Small(..)), rows * cols <= crate::small_sweep::MX_MAX_BYTES, "{rows}x{cols} tokens {t}"); }
        } }
        // A shape off the lists keeps the tile (rows up to 8,192; columns up to 8,192, and 12,288) or the old kernels.
        assert_eq!(mx(4000, 4096, 3, w, i, o), Mx::Tile);
        assert_eq!(mx(4000, 12288, 3, w, i, o), Mx::Tile);
        assert_eq!(mx(4000, 12320, 3, w, i, o), Mx::Old);
        assert_eq!(mx(9000, 4096, 3, w, i, o), Mx::Old);
        // An unaligned operand: weights and input need 16 bytes, the output 4.
        assert_eq!(mx(4096, 4096, 3, 8, i, o), Mx::Old);
        assert_eq!(mx(4096, 4096, 3, w, 8, o), Mx::Old);
        assert_eq!(mx(4096, 4096, 3, w, i, 2), Mx::Old);
        assert_eq!(mx(4096, 4096, 0, w, i, o), Mx::Old);
    }
}
