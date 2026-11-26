//! Kernel kit, dense family: the CPU twin (independent reference) of the block FP8 word on tensor cores.
//! The contract is `docs/tc-dense-contract.md`, section Block FP8; the form that kernels and the host reference run is
//! `tc_f8`. This file shares no code with `tc_f8`: the step is its own wide-integer function (`step`), a weight is
//! decoded as a float and its BF16 pattern is read from the float, and the scale and the join are plain IEEE FP32
//! operations.
//!
//! The word of one weight row and one token row:
//! 1. A weight byte is its E4M3 value, held exactly as a BF16 value.
//! 2. An activation is rounded once to BF16, round to nearest even.
//! 3. A scale block of `block_cols` columns: `S_b` = one ascending chain of 16-column steps from +0.0.
//! 4. `T_b = fl(S_b * scale_b)`.
//! 5. Four ranges of the `n` scale blocks, range `p` = blocks `[n*p/4, n*(p+1)/4)`: `R_p = fl(R_p + T_b)`, ascending,
//!    from +0.0.
//! 6. `acc = fl(fl(R_0 + R_1) + fl(R_2 + R_3))`.
//! 7. `word = bf16(acc)`.
//!
//! A raw job (one column quarter of a second matrix in the pair split) is `range` over its own columns: steps 3 to 5
//! with one range and no round. Std only. No unsafe code.
use crate::tc_twin_bf16 as bf;

/// E4M3 as a float: -0.0 is kept; the codes 0x7f and 0xff are NaN (outside the contract; a loader refuses them).
pub fn e4m3(byte: u8) -> f32 {
    let (e, f) = (u32::from(byte >> 3 & 15), u32::from(byte & 7));
    let v = if e == 0 { f as f32 / 512.0 } else if byte & 127 == 127 { f32::NAN } else { f32::from_bits((e + 120) << 23 | f << 20) };
    if byte & 128 == 0 { v } else { -v }
}
/// The BF16 operand of a weight byte: the high half of its float. Exact: the low half is zero for every E4M3 value.
pub fn operand(byte: u8) -> u16 {
    let bits = e4m3(byte).to_bits();
    assert!(bits & 0xffff == 0 || byte & 127 == 127, "an E4M3 value is a BF16 value");
    (bits >> 16) as u16
}
/// One 16-column step for one output word: the GB10 instruction `mma.sync m16n8k16 bf16`. The terms are the 16
/// products and the carried FP32 accumulator. The terms that are not zero are lined up on the largest exponent among
/// them; bits under 25 fraction bits are cut toward zero term by term; the sum is exact; it is cut toward zero to
/// FP32. Device facts (measured 2026-11-15 on short chains of tiny, subnormal and zero operands): a term that is zero
/// (a product with a zero operand, a zero carried value) brings NO exponent to the alignment; a result that is cut to
/// zero is +0.0. An infinite or NaN accumulator is returned unchanged (outside the contract).
pub fn step(weight: &[u16; 16], input: &[u16; 16], acc: u32) -> u32 {
    if acc & 0x7f80_0000 == 0x7f80_0000 { return acc; }
    // A term: negative, magnitude, the exponent of the magnitude's unit bit, the exponent the term is lined up by.
    let mut terms: Vec<(bool, u128, i32, i32)> = Vec::with_capacity(17);
    let field = |v: u16| -> (u128, i32) { let (e, f) = (i32::from(v >> 7 & 255), u128::from(v & 127)); if e == 0 { (f, -126) } else { (f + 128, e - 127) } };
    for (&w, &x) in weight.iter().zip(input) {
        let ((sw, ew), (sx, ex)) = (field(w), field(x));
        if sw * sx != 0 { terms.push(((w ^ x) & 0x8000 != 0, sw * sx, ew + ex - 14, ew + ex)); }
    }
    let (e, f) = ((acc >> 23 & 255) as i32, u128::from(acc & 0x7f_ffff));
    if e != 0 { terms.push((acc >> 31 != 0, f + (1 << 23), e - 150, e - 127)); } else if f != 0 { terms.push((acc >> 31 != 0, f, -149, -126)); }
    let Some(top) = terms.iter().map(|term| term.3).max() else { return 0; };
    let unit = top - 25;
    let mut sum = 0i128;
    for (negative, magnitude, exponent, _) in terms {
        let shift = exponent - unit;
        let cut = if shift >= 0 { magnitude << shift } else if shift <= -127 { 0 } else { magnitude >> -shift };
        sum += if negative { -(cut as i128) } else { cut as i128 };
    }
    let magnitude = sum.unsigned_abs();
    if magnitude == 0 { return 0; }
    let high = 127 - magnitude.leading_zeros() as i32 + unit;
    let sign = u32::from(sum < 0) << 31;
    if high > 127 { return sign | 0x7f80_0000; }
    let shift = (if high < -126 { -149 } else { high - 23 }) - unit;
    let kept = (if shift <= 0 { magnitude << -shift } else if shift >= 128 { 0 } else { magnitude >> shift }) as u32;
    if kept == 0 { return 0; }
    if high < -126 { sign | kept } else { sign | ((high + 127) as u32) << 23 | (kept & 0x7f_ffff) }
}
/// Step 3: the sum of one scale block (FP32 bits). `codes` and `x` are the block's columns. A last step of fewer than
/// 16 columns is filled with +0 operands.
pub fn block(codes: &[u8], x: &[f32]) -> u32 {
    assert_eq!(codes.len(), x.len(), "block FP8 block extent");
    let mut acc = 0u32;
    for (c, v) in codes.chunks(16).zip(x.chunks(16)) {
        let (mut weight, mut input) = ([0u16; 16], [0u16; 16]);
        for (slot, &code) in weight.iter_mut().zip(c) { *slot = operand(code); }
        for (slot, &value) in input.iter_mut().zip(v) { *slot = bf::bf16(value); }
        acc = step(&weight, &input, acc);
    }
    acc
}
/// Steps 3 to 5 for a run of whole scale blocks: the range accumulator (FP32 bits). `codes` and `x` are the run's
/// columns; `scales` holds one FP32 scale a block of the run. This is also the raw word of a quarter job.
pub fn range(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32]) -> u32 {
    assert!(codes.len() == x.len() && block_cols > 0 && scales.len() == codes.len().div_ceil(block_cols), "block FP8 range extent");
    let mut sum = 0.0f32;
    for ((c, v), &scale) in codes.chunks(block_cols).zip(x.chunks(block_cols)).zip(scales) {
        let scaled = f32::from_bits(block(c, v)) * scale;
        sum += scaled;
    }
    sum.to_bits()
}
/// The scale blocks of range `part` of `blocks` scale blocks.
pub fn blocks_of(blocks: usize, part: usize) -> (usize, usize) { (blocks * part / 4, blocks * (part + 1) / 4) }
/// The four range accumulators of one word, before the join. `scales`: one a block of the whole row.
pub fn parts(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32]) -> [u32; 4] {
    assert!(codes.len() == x.len() && block_cols > 0 && scales.len() == codes.len().div_ceil(block_cols), "block FP8 row extent");
    let mut parts = [0u32; 4];
    for (part, slot) in parts.iter_mut().enumerate() {
        let (first, end) = blocks_of(scales.len(), part);
        let (start, stop) = (first * block_cols, (end * block_cols).min(codes.len()));
        *slot = if first == end { 0 } else { range(&codes[start..stop], &scales[first..end], block_cols, &x[start..stop]) };
    }
    parts
}
/// Step 6: the joined FP32 accumulator of one word (bits).
pub fn accumulator(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32]) -> u32 {
    let p = parts(codes, scales, block_cols, x).map(f32::from_bits);
    ((p[0] + p[1]) + (p[2] + p[3])).to_bits()
}
/// Step 7: the output word, the accumulator rounded to BF16, held in FP32 (bits).
pub fn word(acc: u32) -> u32 { u32::from(bf::bf16(f32::from_bits(acc))) << 16 }

// ------------------------------------------------------------------ hard inputs
fn mixed(index: usize, salt: usize) -> u32 {
    let mut value = (index as u32).wrapping_add((salt as u32 + 1).wrapping_mul(0x9e37_79b9));
    value = (value ^ (value >> 16)).wrapping_mul(0x85eb_ca6b);
    value = (value ^ (value >> 13)).wrapping_mul(0xc2b2_ae35);
    value ^ (value >> 16)
}
/// Activation classes of the hard inputs.
pub const CLASSES: usize = 7;
/// One activation. Classes: 0 values near one; 1 a wide exponent spread; 2 subnormals and zeros among small values;
/// 3 signed zeros among ordinary values; 4 plus and minus one in turn (sums that cancel); 5 mantissas of ones;
/// 6 a general FP32 value (it has bits under BF16, so the one rounding of step 2 is seen).
pub fn activation(index: usize, salt: usize, class: usize) -> f32 {
    let h = mixed(index, salt);
    let (sign, frac) = (((h >> 16) & 0x8000) as u16, (h & 127) as u16);
    let pattern = match class % CLASSES {
        0 => sign | ((124 + (h >> 8) % 6) as u16) << 7 | frac,
        1 => sign | ((97 + (h >> 8) % 60) as u16) << 7 | frac,
        2 => sign | match h & 3 { 0 => 0, 1 => frac, _ => ((1 + (h >> 8) % 6) as u16) << 7 | frac },
        3 => if h & 7 == 0 { sign } else { sign | ((120 + (h >> 8) % 12) as u16) << 7 | frac },
        4 => if index.is_multiple_of(2) { 0x3f80 } else { 0xbf80 },
        5 => sign | ((122 + (h >> 8) % 8) as u16) << 7 | 0x7f,
        _ => return f32::from_bits((h & 0x807f_ffff) | ((120 + (h >> 23) % 12) << 23)),
    };
    f32::from_bits(u32::from(pattern) << 16)
}
/// Weight classes of the hard inputs.
pub const WEIGHT_CLASSES: usize = 4;
/// One weight byte, never a NaN code. Classes: 0 any code; 1 every code in turn (row and column walk all 254 values);
/// 2 many zeros and values under 2^-6 (the exponent field 0); 3 the largest magnitudes with both signs.
pub fn weight(index: usize, salt: usize, class: usize) -> u8 {
    let h = mixed(index, salt + 77);
    let code = match class % WEIGHT_CLASSES {
        0 => h as u8,
        1 => (index + salt) as u8,
        2 => if h & 3 == 0 { h as u8 } else { (h >> 8) as u8 & 0x87 },
        _ => (h as u8 & 0x80) | 0x70 | (h >> 8) as u8 & 7,
    };
    if code & 127 == 127 { code ^ 1 } else { code }
}
/// One block scale: positive and finite, over 24 binades, with a full mantissa.
pub fn scale(index: usize, salt: usize) -> f32 {
    let h = mixed(index, salt + 991);
    (1.0 + (h >> 8) as f32 / 16_777_216.0) * 2f32.powi((h % 24) as i32 - 14)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tc_f8;
    #[test]
    fn zero_terms_have_no_alignment_exponent() {
        let (mut w, mut x) = ([0u16; 16], [0u16; 16]);
        w[1] = operand(0x3f); x[1] = bf::bf16(1.9921875 * 2f32.powi(-120));
        let want = (1.875 * f32::from_bits(u32::from(x[1]) << 16)).to_bits();
        for zero in [0u16, 0x8000] {
            w[0] = zero; x[0] = bf::bf16(2f32.powi(22));
            for acc in [0, 0x8000_0000] {
                assert_eq!(step(&w, &x, acc), want);
                assert_eq!(tc_f8::bf16_step(&w, &x, acc), want);
            }
        }
    }
    #[test]
    fn negative_underflow_and_all_zero_steps_are_positive_zero() {
        let (mut w, mut x) = ([0u16; 16], [0u16; 16]);
        for acc in [0, 0x8000_0000] {
            assert_eq!(step(&w, &x, acc), 0);
            assert_eq!(tc_f8::bf16_step(&w, &x, acc), 0);
        }
        w[0] = 0x8080; x[0] = 0x3380;
        assert_eq!(step(&w, &x, 0), 0);
        assert_eq!(tc_f8::bf16_step(&w, &x, 0), 0);
    }
    /// Today's word on main before the rebase (`fp8_row`): one FP32 chain over all columns.
    fn old_word(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32]) -> u32 {
        let mut sum = 0.0f32;
        for (column, (&code, &value)) in codes.iter().zip(x).enumerate() {
            sum += (e4m3(code) * scales[column / block_cols]) * f32::from_bits(u32::from(bf::bf16(value)) << 16);
        }
        word(sum.to_bits())
    }
    fn shared(codes: &[u8], scales: &[f32], block_cols: usize, x: &[f32], raw: bool) -> u32 {
        tc_f8::accumulator(codes.len(), block_cols, raw, &|c| codes[c], &|c| x[c], &|b| scales[b]).to_bits()
    }
    #[test]
    fn every_e4m3_value_is_its_bf16_operand_in_all_three_forms() {
        for code in 0..=255u8 {
            if code & 127 == 127 { continue; }
            let want = operand(code);
            assert_eq!(f32::from_bits(u32::from(want) << 16).to_bits(), e4m3(code).to_bits(), "code {code:#x}");
            assert_eq!(tc_f8::e4m3_bf16(code), want, "code {code:#x}");
        }
        for pair in 0..=0xffffu32 {
            let (a, b) = (pair as u8, (pair >> 8) as u8);
            assert_eq!(tc_f8::e4m3x2_bf16x2(pair), u32::from(tc_f8::e4m3_bf16(a)) | u32::from(tc_f8::e4m3_bf16(b)) << 16, "pair {pair:#x}");
            // The kernels' slow-lane test sees every byte with the exponent field 0.
            for word in [pair, pair << 16, pair << 8 | 0x0800_0008, pair | 0x0808_0000] {
                assert_eq!(tc_f8::any_small(word), word.to_le_bytes().iter().any(|byte| byte & 0x78 == 0), "word {word:#x}");
            }
        }
    }
    #[test]
    fn the_four_byte_form_is_the_operand_of_each_byte() {
        // Register 0 holds bytes 0 and 2, register 1 holds bytes 1 and 3 (the row kernel's pairing).
        let want = |word: u32| -> [u32; 2] {
            let b = word.to_le_bytes();
            [u32::from(tc_f8::e4m3_bf16(b[0])) | u32::from(tc_f8::e4m3_bf16(b[2])) << 16, u32::from(tc_f8::e4m3_bf16(b[1])) | u32::from(tc_f8::e4m3_bf16(b[3])) << 16]
        };
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state as u32 };
        let mut checked = 0u64;
        // Every pair of bytes in every pair of places, beside two other bytes of every kind.
        for pair in 0..=0xffffu32 {
            let (a, b) = (pair & 0xff, pair >> 8);
            let other = next();
            for (i, j) in [(0u32, 1u32), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                let mask = !(0xffu32 << (8 * i) | 0xffu32 << (8 * j));
                for rest in [other & mask, 0, 0x8080_8080 & mask, 0x0101_0101 & mask, 0x7e7e_7e7e & mask] {
                    let word = rest | a << (8 * i) | b << (8 * j);
                    assert_eq!(tc_f8::e4m3x4_bf16x4(word), want(word), "word {word:#x}");
                    assert_eq!(tc_f8::e4m3x4_slow(word), want(word), "slow lane, word {word:#x}");
                    if !tc_f8::any_small(word) { assert_eq!(tc_f8::e4m3x4_fast(word), want(word), "common lane, word {word:#x}"); }
                    checked += 1;
                }
            }
        }
        for _ in 0..4_000_000 { let word = next(); assert_eq!(tc_f8::e4m3x4_bf16x4(word), want(word), "word {word:#x}"); checked += 1; }
        assert_eq!(checked, 65536 * 30 + 4_000_000);
    }
    #[test]
    fn small_integers_give_the_exact_dot_product() {
        // Products and partial sums are small integers and the scales are powers of two: nothing is cut in any order.
        let cols = 512;
        let codes: Vec<u8> = (0..cols).map(|i| [0x38u8, 0x40, 0x44, 0xb8, 0xc0, 0x00, 0x80, 0x48][i % 8]).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i * 5 + 1) % 29) as f32 - 14.0).collect();
        let scales: Vec<f32> = (0..cols / 128).map(|b| [1.0f32, 2.0, 0.5, 4.0][b % 4]).collect();
        let want: f32 = codes.iter().zip(&x).enumerate().map(|(c, (&w, &v))| e4m3(w) * v * scales[c / 128]).sum();
        assert_eq!(f32::from_bits(accumulator(&codes, &scales, 128, &x)), want);
        assert_eq!(shared(&codes, &scales, 128, &x, false), want.to_bits());
    }
    #[test]
    fn the_shared_step_is_the_twin_step_on_hard_operands() {
        let mut checked = 0;
        for seed in 0..40_000usize {
            let (wc, xc) = (seed % WEIGHT_CLASSES, seed / WEIGHT_CLASSES % CLASSES);
            let (mut weight, mut input) = ([0u16; 16], [0u16; 16]);
            for k in 0..16 {
                // Weights of the step: E4M3 operands, and on some seeds any BF16 pattern (the step itself is format-blind).
                weight[k] = if seed % 5 == 4 { bf::bf16(activation(k, seed + 9, seed / 7)) } else { operand(super::weight(seed * 16 + k, seed, wc)) };
                input[k] = bf::bf16(activation(seed * 16 + k, seed + 3, xc));
            }
            // The carried accumulator: zero, a subnormal, a value near the products, far above them, far below them, negative zero.
            for acc in [0u32, 0x8000_0000, mixed(seed, 1) & 0x007f_ffff, activation(seed, 5, 0).to_bits(), activation(seed, 6, 1).to_bits(),
                (activation(seed, 7, 6) * 1.0e6).to_bits(), (activation(seed, 8, 6) * 1.0e-9).to_bits(), f32::INFINITY.to_bits()] {
                assert_eq!(tc_f8::bf16_step(&weight, &input, acc), step(&weight, &input, acc), "seed {seed} acc {acc:#x}");
                checked += 1;
            }
        }
        assert_eq!(checked, 320_000);
    }
    #[test]
    fn the_shared_word_is_the_twin_word_on_every_class_and_shape() {
        let mut checked = 0;
        // (cols, block_cols): the model's shapes cut down, short blocks, a block that is not whole steps, one block,
        // fewer blocks than ranges, a column count that is not whole steps.
        for (shape, &(cols, bc)) in [(512usize, 128usize), (2048, 128), (1536, 128), (256, 16), (96, 32), (200, 100), (128, 128), (48, 16), (33, 16), (80, 48), (7, 3)].iter().enumerate() {
            for seed in 0..(CLASSES * WEIGHT_CLASSES * 3) {
                let codes: Vec<u8> = (0..cols).map(|c| weight(c + seed * 4099, shape, seed % WEIGHT_CLASSES)).collect();
                let x: Vec<f32> = (0..cols).map(|c| activation(c, shape * 131 + seed, seed / WEIGHT_CLASSES)).collect();
                let scales: Vec<f32> = (0..cols.div_ceil(bc)).map(|b| scale(b, shape * 17 + seed)).collect();
                let twin = accumulator(&codes, &scales, bc, &x);
                assert_eq!(shared(&codes, &scales, bc, &x, false), twin, "cols {cols} block {bc} seed {seed}");
                assert_eq!(shared(&codes, &scales, bc, &x, true), range(&codes, &scales, bc, &x), "raw: cols {cols} block {bc} seed {seed}");
                checked += 1;
            }
        }
        assert_eq!(checked, 11 * CLASSES * WEIGHT_CLASSES * 3);
    }
    #[test]
    fn a_range_holds_whole_scale_blocks_and_the_four_cover_every_block_once() {
        for blocks in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 32, 128] {
            let mut next = 0;
            for part in 0..4 { let (first, end) = blocks_of(blocks, part); assert_eq!(first, next); next = end; assert_eq!(tc_f8::range(blocks, part), (first, end)); }
            assert_eq!(next, blocks);
        }
    }
    #[test]
    fn a_raw_quarter_job_is_the_range_of_the_whole_row() {
        // The model's column counts: 12, 16, 32 and 128 scale blocks. Each quarter alone gives the range of the whole row.
        for (cols, bc) in [(1536usize, 128usize), (2048, 128), (4096, 128), (16384, 128)] {
            let codes: Vec<u8> = (0..cols).map(|c| weight(c, cols, 0)).collect();
            let x: Vec<f32> = (0..cols).map(|c| activation(c, cols, 6)).collect();
            let scales: Vec<f32> = (0..cols / bc).map(|b| scale(b, cols)).collect();
            let whole = parts(&codes, &scales, bc, &x);
            for (q, &part) in whole.iter().enumerate() {
                let (columns, blocks) = (q * cols / 4..(q + 1) * cols / 4, q * scales.len() / 4..(q + 1) * scales.len() / 4);
                assert_eq!(range(&codes[columns.clone()], &scales[blocks.clone()], bc, &x[columns.clone()]), part, "cols {cols} quarter {q}");
                assert_eq!(shared(&codes[columns.clone()], &scales[blocks], bc, &x[columns], true), part);
            }
        }
    }
    /// The last BF16 round hides most order differences (about one word in a thousand moves). So the test first counts
    /// the accumulators that differ, and then SEARCHES seeded rows for finished words that tell the new word from the
    /// old chain, and the four ranges from one range, and checks that it found them.
    #[test]
    fn the_word_is_told_apart_from_the_old_chain_and_from_one_range() {
        let (cols, bc) = (512usize, 128usize);
        let row = |seed: usize| -> (Vec<u8>, Vec<f32>, Vec<f32>) {
            ((0..cols).map(|c| weight(c + seed * 2053, 3, 0)).collect(), (0..cols / bc).map(|b| scale(b, seed)).collect(), (0..cols).map(|c| activation(c, seed, 6)).collect())
        };
        let old_acc = |codes: &[u8], scales: &[f32], x: &[f32]| -> u32 {
            let mut sum = 0.0f32;
            for (column, (&code, &value)) in codes.iter().zip(x).enumerate() { sum += (e4m3(code) * scales[column / bc]) * f32::from_bits(u32::from(bf::bf16(value)) << 16); }
            sum.to_bits()
        };
        let (mut acc_age, mut acc_ranges) = (0, 0);
        for seed in 0..400usize {
            let (codes, scales, x) = row(seed);
            let new = accumulator(&codes, &scales, bc, &x);
            if new != old_acc(&codes, &scales, &x) { acc_age += 1; }
            if new != range(&codes, &scales, bc, &x) { acc_ranges += 1; }
        }
        assert!(acc_age >= 200 && acc_ranges >= 40, "accumulators that differ of 400: old chain {acc_age}, one range {acc_ranges}");
        // Finished words. Measured on this search: 38 of 200,000 rows tell the old chain from the new word after the
        // BF16 round, and 1 of 200,000 tells one range from four. So a check that sees only finished words cannot see
        // a wrong join order: the device checks compare the unrounded accumulator (`expert_ops::FP8_UNROUNDED`).
        let (mut by_age, mut same) = (0, 0);
        for seed in 0..200_000usize {
            let (codes, scales, x) = row(seed);
            if word(accumulator(&codes, &scales, bc, &x)) != old_word(&codes, &scales, bc, &x) { by_age += 1; } else { same += 1; }
            if by_age >= 3 && same >= 3 { break; }
        }
        assert!(by_age >= 3 && same >= 3, "finished words: old chain {by_age}, equal {same}");
    }
    #[test]
    fn the_activation_is_rounded_once_and_the_scale_enters_once_a_block() {
        let (cols, bc) = (256usize, 128usize);
        let codes: Vec<u8> = (0..cols).map(|c| weight(c, 5, 0)).collect();
        let general: Vec<f32> = (0..cols).map(|c| activation(c, 9, 6)).collect();
        let rounded: Vec<f32> = general.iter().map(|&v| f32::from_bits(u32::from(bf::bf16(v)) << 16)).collect();
        let scales = [scale(0, 1), scale(1, 1)];
        assert_ne!(general, rounded);
        assert_eq!(accumulator(&codes, &scales, bc, &general), accumulator(&codes, &scales, bc, &rounded));
        // Doubling a block scale doubles that block's term exactly: the scale is outside the block's chain.
        let p = parts(&codes, &scales, bc, &general);
        let q = parts(&codes, &[scales[0] * 2.0, scales[1]], bc, &general);
        // Two blocks: ranges 0 and 2 are empty, range 1 is block 0, range 3 is block 1.
        assert_eq!((p[0], p[2]), (0, 0));
        assert_eq!(f32::from_bits(q[1]), 2.0 * f32::from_bits(p[1]));
        assert_eq!(q[3], p[3]);
    }
}
