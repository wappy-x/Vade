//! Kernel kit: the CPU twin (reference) of the QUARTER WORD of a second matrix in block FP8 and in MXFP8.
//! (NVFP4: `tc_twin::chain_quarters`. Plain BF16: `tc_twin_bf16::parts`.)
//!
//! A second matrix (shared expert down, latent o, KDA out_proj, dense down) is split by INPUT COLUMNS in the pair
//! split. The word of one output row and one token row, for `cols` input columns:
//! - Quarter q = columns [q * cols/4, (q+1) * cols/4). A quarter holds whole storage steps: whole scale blocks
//!   (block FP8) or whole 256-column tiles (MXFP8).
//! - `acc[q]` = the format's own chain of today, run on quarter q alone as if it were the whole matrix: anchored at
//!   the quarter's first column, from +0.0, and with NO round at its end (raw FP32 bits).
//!   Block FP8 (`expert_ops`, `Op::Fp8Serial`): one ascending chain, `sum = fl(sum + fl(fl(e4m3(w) * scale) * bf16(x)))`.
//!   MXFP8 (`text_reference::project_row`): 32 lanes; lane L takes columns `tile*256 + L*8 .. +8` of every tile of
//!   the quarter in ascending order, `sum = fma(fl(e4m3(w) * 2^(s-127)), x, sum)`; then the fixed down-shuffle tree.
//! - `joined = fl(fl(acc[0] + acc[1]) + fl(acc[2] + acc[3]))`; `word = bf16(joined)`. Neither format has an epilogue
//!   scale (the block scales are inside the products), so the shared expert's `acc` enters the expert fold raw: it is
//!   the `shared` operand of `tc_twin_fold::quarter_fold`.
//!
//! The owner split is `tc_twin_fold`'s: an owner sends `partial(acc, owner, parts)`, the REDUCE joins the partials in
//! owner order. The order depends on the column index only, so 1, 2 and 4 owners give the same bits.
//! A NaN: the CPU and the device may hold different NaN bits in a raw accumulator; the finished word is the canonical
//! NaN on both (`bf16_word`). Std only. No unsafe code.

use crate::tc_twin::Rng;
use crate::tc_twin_fold::{bf16_word, join};

/// The columns of quarter `q`. None unless a quarter is a whole number of `step` columns.
pub fn quarter(cols: usize, q: usize, step: usize) -> Option<std::ops::Range<usize>> {
    if q >= 4 || cols == 0 || step == 0 || !cols.is_multiple_of(4 * step) { return None; }
    Some(q * (cols / 4)..(q + 1) * (cols / 4))
}

/// E4M3 as the kernels decode a stored byte: -0.0 is kept; the codes 0x7f and 0xff are NaN (a loader refuses them).
pub fn e4m3(byte: u8) -> f32 {
    let (e, f) = (u32::from(byte >> 3 & 15), u32::from(byte & 7));
    let v = if e == 0 { f as f32 / 512.0 } else if byte & 127 == 127 { f32::NAN } else { f32::from_bits((e + 120) << 23 | f << 20) };
    if byte & 128 == 0 { v } else { -v }
}

/// The MXFP8 scale of 32 columns: 2^(byte - 127). Byte 0 stands for 2^-127.
pub fn mx_scale(byte: u8) -> f32 { f32::from_bits(if byte == 0 { 0x0040_0000 } else { u32::from(byte) << 23 }) }

/// Block FP8, one quarter of one row: the raw accumulator. `codes` and `x` are the quarter's columns; `scales` holds
/// one FP32 scale a block of `block_cols` columns of the quarter.
pub fn fp8_chain(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32]) -> u32 {
    assert!(codes.len() == x.len() && block_cols > 0 && scales.len() == codes.len().div_ceil(block_cols), "block FP8 quarter extent");
    let mut sum = 0.0f32;
    for (column, (&code, &value)) in codes.iter().zip(x).enumerate() {
        let weight = e4m3(code) * scales[column / block_cols];
        let term = weight * f32::from_bits(bf16_word(value));
        sum += term;
    }
    sum.to_bits()
}

/// MXFP8, one quarter of one row: the raw accumulator. `codes` and `x` are the quarter's columns (whole 256-column
/// tiles); `scales` holds one byte a 32 columns. The input stays FP32, as in the decode kernel.
pub fn mx_chain(codes: &[u8], scales: &[u8], x: &[f32]) -> u32 {
    assert!(codes.len() == x.len() && codes.len().is_multiple_of(256) && scales.len() == codes.len() / 32, "MXFP8 quarter extent");
    let mut lanes = [0.0f32; 32];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        for tile in 0..codes.len() / 256 {
            for column in tile * 256 + lane * 8..tile * 256 + lane * 8 + 8 {
                let weight = e4m3(codes[column]) * mx_scale(scales[column / 32]);
                *sum = weight.mul_add(x[column], *sum);
            }
        }
    }
    for distance in [16, 8, 4, 2, 1] {
        let old = lanes;
        for lane in 0..32 - distance { lanes[lane] += old[lane + distance]; }
    }
    lanes[0].to_bits()
}

/// The four raw quarter accumulators of one block-FP8 row. `scales`: one a block of the whole row.
pub fn fp8_quarters(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32]) -> Option<[u32; 4]> {
    let mut acc = [0u32; 4];
    for (q, slot) in acc.iter_mut().enumerate() {
        let r = quarter(codes.len(), q, block_cols)?;
        *slot = fp8_chain(&codes[r.clone()], scales.get(r.start / block_cols..r.end / block_cols)?, block_cols, x.get(r)?);
    }
    Some(acc)
}

/// The four raw quarter accumulators of one MXFP8 row. `scales`: one byte a 32 columns of the whole row.
pub fn mx_quarters(codes: &[u8], scales: &[u8], x: &[f32]) -> Option<[u32; 4]> {
    let mut acc = [0u32; 4];
    for (q, slot) in acc.iter_mut().enumerate() {
        let r = quarter(codes.len(), q, 256)?;
        *slot = mx_chain(&codes[r.clone()], scales.get(r.start / 32..r.end / 32)?, x.get(r)?);
    }
    Some(acc)
}

/// The finished word of a block-FP8 or MXFP8 second matrix with no fold: (joined accumulator, word).
pub fn second_word(acc: [u32; 4]) -> (u32, u32) {
    let joined = join(&acc);
    (joined, bf16_word(f32::from_bits(joined)))
}

// ------------------------------------------------------------------ hard inputs
/// One weight row in both formats (the same codes, each format's own scales) and one token row.
pub struct QuarterRow { pub class: &'static str, pub codes: Vec<u8>, pub fp8_scales: Vec<f32>, pub mx_scales: Vec<u8>, pub x: Vec<f32> }
/// The hard classes: ordinary values; any finite FP32 input; signed zero inputs; subnormal inputs; one NaN input; one
/// infinite input; two products that are opposite infinities; quarters 1 and 3 that are the negatives of 0 and 2.
pub const QUARTER_CLASSES: usize = 8;
pub fn hard_rows(n: usize, cols: usize, block_cols: usize, seed: u64) -> Vec<QuarterRow> {
    assert!(quarter(cols, 0, 256).is_some() && quarter(cols, 0, block_cols).is_some(), "a quarter is whole tiles and whole blocks");
    let mut r = Rng(seed);
    let (width, fp8_blocks, mx_blocks) = (cols / 4, cols / 4 / block_cols, cols / 4 / 32);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Finite codes only; about one code in 16 is a signed zero.
        let mut codes: Vec<u8> = (0..cols).map(|_| { let c = r.next_u64() as u8; if c & 127 == 127 { c & 0x80 | 0x38 } else if r.below(16) == 0 { c & 0x80 } else { c } }).collect();
        let mut fp8_scales: Vec<f32> = (0..cols / block_cols).map(|_| f32::from_bits(ordinary(&mut r, -6) & 0x7fff_ffff)).collect();
        let mut mx_scales: Vec<u8> = (0..cols / 32).map(|_| (119 + r.below(9)) as u8).collect();
        let mut x: Vec<f32> = (0..cols).map(|_| f32::from_bits(ordinary(&mut r, 0))).collect();
        let at = r.below(cols as u64) as usize;
        let class = match i % QUARTER_CLASSES {
            0 => "ordinary values",
            1 => { for v in &mut x { let b = r.next_u64() as u32; *v = f32::from_bits(if b & 0x7f80_0000 == 0x7f80_0000 { b & 0xbfff_ffff } else { b }); } "any finite FP32 input" }
            2 => { for v in &mut x { *v = f32::from_bits((r.below(2) as u32) << 31); } "signed zero inputs" }
            3 => { for v in &mut x { *v = f32::from_bits(r.next_u64() as u32 & 0x807f_ffff); } "subnormal inputs" }
            4 => { x[at] = f32::from_bits(0x7f80_0000 | ((r.below(2) as u32) << 31) | (1 + r.below(0x7f_ffff) as u32)); "one NaN input" }
            5 => { x[at] = f32::from_bits(0x7f80_0000 | ((r.below(2) as u32) << 31)); codes[at] = 0x38 | (r.below(2) as u8) << 7; "one infinite input" }
            6 => { let other = (at + 1 + r.below(cols as u64 - 1) as usize) % cols;
                x[at] = f32::INFINITY; x[other] = f32::NEG_INFINITY; codes[at] = 0x38; codes[other] = 0x38; "opposite infinite products" }
            _ => { for q in [1usize, 3] {
                    for j in 0..width { codes[q * width + j] = codes[(q - 1) * width + j] ^ 0x80; x[q * width + j] = x[(q - 1) * width + j]; }
                    for b in 0..fp8_blocks { fp8_scales[q * fp8_blocks + b] = fp8_scales[(q - 1) * fp8_blocks + b]; }
                    for b in 0..mx_blocks { mx_scales[q * mx_blocks + b] = mx_scales[(q - 1) * mx_blocks + b]; }
                }
                "quarters that cancel in the join" }
        };
        out.push(QuarterRow { class, codes, fp8_scales, mx_scales, x });
    }
    out
}
/// A finite FP32 value of either sign with its exponent within 8 of `exponent`.
fn ordinary(r: &mut Rng, exponent: i32) -> u32 {
    let e = (127 + exponent + r.below(17) as i32 - 8) as u32;
    ((r.below(2) as u32) << 31) | (e << 23) | (r.next_u64() as u32 & 0x7f_ffff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_ops::{self, Arguments, Op, Shape, FP8_QUARTER};
    use crate::tc_twin_fold::{partial, reduce, reduce_word};

    fn is_nan(b: u32) -> bool { b & 0x7fff_ffff > 0x7f80_0000 }
    fn same(a: u32, b: u32) -> bool { a == b || (is_nan(a) && is_nan(b)) }
    /// (columns, block columns): a quarter of one MXFP8 tile and two FP8 blocks, and one of two tiles and four blocks.
    const SHAPES: [(usize, usize); 2] = [(1024, 128), (2048, 128)];
    fn fp8_shape(cols: usize, block_cols: usize, role: u32) -> Shape {
        Shape { op: Op::Fp8Serial, rows: 1, cols: cols as u32, block_rows: 1, block_cols: block_cols as u32, tokens: 1, global: 1.0, activation: 1.0, bound: 0.0, role }
    }

    /// Block FP8: the twin's chain on a quarter is the host reference of the decode kernel on that quarter as its own
    /// matrix: raw with the role FP8_QUARTER, and today's rounded word without it.
    #[test]
    fn the_fp8_quarter_is_todays_chain_on_the_quarter() {
        let mut compared = 0usize;
        for (cols, block_cols) in SHAPES {
            for row in hard_rows(QUARTER_CLASSES * 40, cols, block_cols, 0xf8 + cols as u64) {
                let acc = fp8_quarters(&row.codes, &row.fp8_scales, block_cols, &row.x).unwrap();
                for (q, &raw) in acc.iter().enumerate() {
                    let r = quarter(cols, q, block_cols).unwrap();
                    let scales: Vec<u8> = row.fp8_scales[r.start / block_cols..r.end / block_cols].iter().flat_map(|s| s.to_le_bytes()).collect();
                    let (x, codes) = (&row.x[r.clone()], &row.codes[r]);
                    // The reference refuses nonfinite inputs and outputs; those rows keep their class laws below.
                    let Ok(got) = expert_ops::reference(fp8_shape(cols / 4, block_cols, FP8_QUARTER), [x, &[], &[], &[]], codes, &scales) else { continue };
                    assert_eq!(got[0].to_bits(), raw, "{} quarter {q}", row.class);
                    let rounded = expert_ops::reference(fp8_shape(cols / 4, block_cols, 0), [x, &[], &[], &[]], codes, &scales).unwrap();
                    assert_eq!(rounded[0].to_bits(), bf16_word(f32::from_bits(raw)), "{} quarter {q}", row.class);
                    compared += 1;
                }
            }
        }
        assert!(compared > 2 * 40 * 4 * 3, "{compared}");
    }

    /// The raw quarter role exists on the block-FP8 decode operation only, for whole scale blocks, and only the tile
    /// kernel may run it (the one-warp body rounds every word).
    #[test]
    fn the_raw_quarter_role_is_a_tile_kernel_form_of_the_fp8_decode_operation() {
        assert!(fp8_shape(512, 128, FP8_QUARTER).lengths().is_some());
        assert!(fp8_shape(500, 128, FP8_QUARTER).lengths().is_none(), "a quarter is whole scale blocks");
        assert!(fp8_shape(512, 128, FP8_QUARTER + 1).lengths().is_none());
        for op in [Op::Fp8Project, Op::Fp8Rows, Op::NvProject, Op::PlainSerial, Op::SwiGlu, Op::Quantize] {
            assert!(Shape { op, bound: 10.0, block_cols: 16, ..fp8_shape(512, 128, FP8_QUARTER) }.lengths().is_none(), "{op:?}");
        }
        let job = |cols: usize, role: u32, weight: u64| Arguments { input: [0x1000, 4, 4, 4], weight, scales: 0x3000, output: 0x4000, shape: fp8_shape(cols, 128, role) };
        let mut four = [job(512, FP8_QUARTER, 0x2000); 4];
        for job in &mut four { job.shape.rows = 4096; }
        assert_eq!(expert_ops::launch_geometry(&four, 77, 4).unwrap(), (4 * 256, expert_ops::FP8_TILE_THREADS, expert_ops::FP8_TILE_SHARED));
        // Not a tile launch (an unaligned owner; more than 8 jobs): today's jobs fall back to the one-warp body, a raw quarter is refused.
        assert_eq!(expert_ops::launch_geometry(&[job(512, 0, 0x2008)], 77, 1).unwrap(), (77, 32, 0));
        assert!(expert_ops::launch_geometry(&[job(512, FP8_QUARTER, 0x2008)], 77, 1).is_err());
        assert!(expert_ops::launch_geometry(&[job(512, FP8_QUARTER, 0x2000); 9], 77, 9).is_err());
        assert!(expert_ops::batch_shapes_agree(four.iter().map(|job| job.shape)));
        assert!(!expert_ops::batch_shapes_agree([four[0].shape, Shape { role: 0, ..four[0].shape }].into_iter()), "raw and rounded jobs do not share a launch");
    }

    /// MXFP8: the twin's lanes on a quarter are today's host reference on that quarter as its own matrix. The decode
    /// kernel stores this raw FP32 sum already (its BF16 round is a launch of its own), so an MXFP8 quarter needs no
    /// new kernel form: one projection a quarter.
    #[test]
    fn the_mx_quarter_is_todays_lanes_on_the_quarter() {
        let mut compared = 0usize;
        for (cols, block_cols) in SHAPES {
            for row in hard_rows(QUARTER_CLASSES * 40, cols, block_cols, 0x38 + cols as u64) {
                let acc = mx_quarters(&row.codes, &row.mx_scales, &row.x).unwrap();
                for (q, &raw) in acc.iter().enumerate() {
                    let r = quarter(cols, q, 256).unwrap();
                    let Ok(want) = crate::text_reference::project_row(&row.codes[r.clone()], &row.mx_scales[r.start / 32..r.end / 32], &row.x[r]) else { continue };
                    assert_eq!(want.to_bits(), raw, "{} quarter {q}", row.class);
                    compared += 1;
                }
            }
        }
        assert!(compared > 2 * 40 * 4 * 3, "{compared}");
    }

    /// Every hard class keeps its law in both formats, and 1, 2 and 4 owners give the same bits.
    #[test]
    fn hard_inputs_keep_their_laws_and_owners_agree() {
        let mut seen = std::collections::BTreeSet::new();
        for (cols, block_cols) in SHAPES {
            for (i, row) in hard_rows(QUARTER_CLASSES * 60, cols, block_cols, 0xd2 + cols as u64).iter().enumerate() {
                seen.insert(row.class);
                let forms = [("fp8", fp8_quarters(&row.codes, &row.fp8_scales, block_cols, &row.x).unwrap()), ("mx", mx_quarters(&row.codes, &row.mx_scales, &row.x).unwrap())];
                for (format, acc) in forms {
                    let (joined, word) = second_word(acc);
                    let at = format!("row {i} ({}), {format}, {cols} columns", row.class);
                    for parts in [1usize, 2, 4] {
                        let sent: Vec<u32> = (0..parts).map(|owner| partial(acc, owner, parts)).collect();
                        assert!(same(reduce(&sent), joined), "{at}, {parts} owners");
                    }
                    // The pair's exchange: the REDUCE of the two partial sums with the one BF16 round and no scale.
                    assert_eq!(reduce_word(partial(acc, 0, 2), partial(acc, 1, 2), None, true), word, "{at}");
                    match row.class {
                        "signed zero inputs" => assert_eq!((acc, word), ([0; 4], 0), "{at}: only zero products give +0.0"),
                        "one NaN input" | "opposite infinite products" => assert_eq!(word, 0x7fff_0000, "{at}"),
                        "one infinite input" => assert_eq!(word & 0x7fff_ffff, 0x7f80_0000, "{at}"),
                        "quarters that cancel in the join" => assert_eq!((joined, word), (0, 0), "{at}"),
                        "subnormal inputs" => assert!(word & 0x7f80_0000 != 0x7f80_0000, "{at}"),
                        _ => {}
                    }
                }
            }
        }
        assert_eq!(seen.len(), QUARTER_CLASSES);
    }

    /// The quarter word is a NEW function: it is not today's one chain (or one set of lanes) over the whole row.
    #[test]
    fn the_quarter_word_is_not_todays_whole_row_word() {
        let (cols, block_cols) = (2048, 128);
        let (mut fp8, mut mx, mut cancel, mut rows) = (0usize, 0usize, 0usize, 0usize);
        for row in hard_rows(QUARTER_CLASSES * 100, cols, block_cols, 0x9e) {
            if !matches!(row.class, "ordinary values" | "quarters that cancel in the join") { continue; }
            let whole = (fp8_chain(&row.codes, &row.fp8_scales, block_cols, &row.x), mx_chain(&row.codes, &row.mx_scales, &row.x));
            let joined = (second_word(fp8_quarters(&row.codes, &row.fp8_scales, block_cols, &row.x).unwrap()).0, second_word(mx_quarters(&row.codes, &row.mx_scales, &row.x).unwrap()).0);
            rows += 1; fp8 += (whole.0 != joined.0) as usize; mx += (whole.1 != joined.1) as usize;
            if row.class == "quarters that cancel in the join" { cancel += (whole.0 != 0 && whole.1 != 0) as usize; }
        }
        assert!(rows == 200 && fp8 > 60 && mx > 60 && cancel > 50, "{fp8} {mx} {cancel} of {rows}");
    }

    /// Plain BF16: the four range accumulators of `tc_twin_bf16` are the quarters of a column split.
    #[test]
    fn the_bf16_parts_join_to_the_canonical_word_on_every_owner_count() {
        let mut r = Rng(0xbf16);
        for cols in [64usize, 256, 1024] {
            for _ in 0..50 {
                let weight: Vec<u16> = (0..cols).map(|_| (ordinary(&mut r, -3) >> 16) as u16).collect();
                let input: Vec<f32> = (0..cols).map(|_| f32::from_bits(ordinary(&mut r, 0))).collect();
                let parts = crate::tc_twin_bf16::parts(&weight, &input);
                let canonical = crate::tc_twin_bf16::canonical(&weight, &input);
                for owners in [1usize, 2, 4] {
                    let sent: Vec<u32> = (0..owners).map(|owner| partial(parts, owner, owners)).collect();
                    assert_eq!(reduce(&sent), canonical, "{cols} columns, {owners} owners");
                }
                assert_eq!(second_word(parts).1, crate::tc_twin_bf16::word(canonical));
            }
        }
    }
}
