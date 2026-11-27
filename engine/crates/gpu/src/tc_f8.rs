//! Contract of the block FP8 word on tensor cores (`docs/tc-dense-contract.md`, section Block FP8), as integer code.
//! Host and device share this file.
//!
//! The word of one weight row and one token row, for a matrix of `cols` columns whose scale blocks are `block_cols`
//! columns wide:
//! 1. A weight byte is an E4M3 value. Every E4M3 value is a BF16 value, so the operand is exact (`e4m3_bf16`).
//! 2. An activation is rounded once to BF16, round to nearest even.
//! 3. A scale block is the columns `[b * block_cols, min((b + 1) * block_cols, cols))`. Its sum `S_b` is one ascending
//!    chain of 16-column steps from +0.0, anchored at the block's first column. A step is the GB10 instruction
//!    `mma.sync m16n8k16 bf16` (`bf16_step`: the 16 products and the carried FP32 accumulator are lined up on the largest
//!    exponent of a term that is not zero, bits under 25 fraction bits are cut toward zero term by term, the sum is
//!    exact, and it is cut toward zero to FP32; a zero result is +0.0). A step of fewer than 16 columns (the end of a
//!    block) is filled with +0 operands: a zero term adds nothing and moves nothing.
//! 4. The scale enters once a block, on the block's sum: `T_b = fl(S_b * scale_b)`. One IEEE FP32 multiply.
//! 5. The `n = ceil(cols / block_cols)` scale blocks are cut into four ranges by the block count only: range `p` =
//!    blocks `[n*p/4, n*(p+1)/4)`. So a range boundary never cuts a scale block or a step.
//!    `R_p` = one ascending chain from +0.0: `R_p = fl(R_p + T_b)`. One IEEE FP32 add a block, never fused.
//! 6. `acc = fl(fl(R_0 + R_1) + fl(R_2 + R_3))`.
//! 7. `word = bf16(acc)`.
//!
//! A RAW job (the pair split's one column quarter of a second matrix, `expert_ops::FP8_QUARTER`) is ONE range over all
//! the job's blocks, and it stops after step 5: no join and no BF16 round. When `n` is a multiple of 4 the raw word of
//! quarter `p` is `R_p` of the whole matrix, so 1, 2 and 4 owners give the bits of step 6.
//!
//! Nothing here depends on the row count, the tile, the thread count, the kernel entry or the rank count.
//! Outside the contract: the weight codes 0x7f and 0xff (NaN in E4M3; a loader refuses them), a scale that is not a
//! positive finite number, an activation that is not finite.
//!
//! The tensor-core kernels (`kernels/src/fp8_tc.rs`) run the instruction. The generic device bodies and the host
//! reference (`expert_ops::unit`) run this file. `tc_twin_f8` is the independent reference both are tested against.
//! No unsafe code.

/// Columns of one tensor-core step.
pub const STEP: usize = 16;
/// The BF16 pattern of the value of an E4M3 byte. Exact for every code but 0x7f and 0xff (outside the contract: they
/// give the pattern the arithmetic below gives, 480 with the sign).
#[inline(always)]
pub const fn e4m3_bf16(code: u8) -> u16 {
    let sign = ((code as u16) & 0x80) << 8;
    let (e, f) = (((code >> 3) & 15) as u16, (code & 7) as u16);
    // A normal code is (1 + f/8) * 2^(e - 7): FP32 exponent field e + 120, fraction f in the top three bits.
    if e != 0 { return sign | ((e + 120) << 7) | (f << 4); }
    // Exponent field 0: f * 2^-9, a normal BF16 number (or zero).
    sign | small(f as u32) as u16
}
/// `f * 2^-9` as a BF16 pattern, f = 0..8. Two packed words (16 bits a value, f ascending from bit 0), no memory table.
#[inline(always)]
const fn small(f: u32) -> u32 {
    let packed: u64 = if f & 4 == 0 { 0x3bc0_3b80_3b00_0000 } else { 0x3c60_3c40_3c20_3c00 };
    ((packed >> ((f & 3) * 16)) & 0xffff) as u32
}
/// Two E4M3 bytes (the low 16 bits of `pair`, first byte low) to two BF16 patterns (first in the low half): the
/// register form of the kernels. The same values as `e4m3_bf16` on each byte.
#[inline(always)]
pub fn e4m3x2_bf16x2(pair: u32) -> u32 {
    let lanes = (pair & 0xff) | ((pair & 0xff00) << 8);
    let sign = (lanes & 0x0080_0080) << 8;
    let magnitude = lanes & 0x007f_007f;
    let normal = (magnitude << 4) + 0x3c00_3c00;
    let exponent = magnitude & 0x0078_0078;
    if exponent & 0xffff != 0 && exponent >> 16 != 0 { return normal | sign; }
    let fixed = |lane: u32| -> u32 {
        let (m, n) = ((magnitude >> lane) & 0x7f, (normal >> lane) & 0xffff);
        if m & 0x78 != 0 { n } else { small(m) }
    };
    (fixed(0) | fixed(16) << 16) | sign
}
/// True when one of the four E4M3 bytes of `word` has the exponent field 0 (a zero or a value under 2^-6): the
/// kernels' test for the slow lane of `e4m3x2_bf16x2`.
#[inline(always)]
pub const fn any_small(word: u32) -> bool {
    let e = word & 0x7878_7878;
    (e.wrapping_sub(0x0101_0101) & !e & 0x8080_8080) != 0
}
/// Four E4M3 bytes of one weight row (`word`, the first column in the low byte) to four BF16 patterns in two
/// registers: `[0]` holds the patterns of bytes 0 (low half) and 2, `[1]` those of bytes 1 (low half) and 3. The same
/// values as `e4m3_bf16` on each byte. The row kernel (`fp8_tc::row_body`) feeds the two registers to the two halves of
/// the instruction's k index and pairs the activation columns the same way, so every product meets its own column.
/// The common lane (no byte with exponent field 0) is six register operations a pair: a byte in a 16-bit lane,
/// shifted by 4, is `s e e e e f f f 0 0 0 0`; the sign moves from bit 11 to bit 15 (`+ 15 * bit 11`) and the
/// exponent bias 120 is added (0x3c00). No carry leaves a lane: the largest lane is 0xc3f0.
#[inline(always)]
pub fn e4m3x4_bf16x4(word: u32) -> [u32; 2] { if any_small(word) { e4m3x4_slow(word) } else { e4m3x4_fast(word) } }
/// The common lane of `e4m3x4_bf16x4`: right when no byte of `word` has the exponent field 0 (`!any_small(word)`).
#[inline(always)]
pub const fn e4m3x4_fast(word: u32) -> [u32; 2] {
    let (even, odd) = ((word << 4) & 0x0ff0_0ff0, (word >> 4) & 0x0ff0_0ff0);
    [even.wrapping_add((even & 0x0800_0800).wrapping_mul(15)).wrapping_add(0x3c00_3c00),
        odd.wrapping_add((odd & 0x0800_0800).wrapping_mul(15)).wrapping_add(0x3c00_3c00)]
}
/// The slow lane of `e4m3x4_bf16x4`: right for every word (the two-byte form on bytes 0 and 2, and on bytes 1 and 3).
#[inline(always)]
pub fn e4m3x4_slow(word: u32) -> [u32; 2] {
    [e4m3x2_bf16x2((word & 0xff) | ((word >> 8) & 0xff00)), e4m3x2_bf16x2(((word >> 8) & 0xff) | ((word >> 16) & 0xff00))]
}
/// Significand and the exponent of its unit bit, of one BF16 pattern.
#[inline(always)]
const fn split(v: u16) -> (u32, i32) {
    let (e, f) = (((v >> 7) & 255) as i32, (v & 127) as u32);
    if e == 0 { (f, -126) } else { (128 + f, e - 127) }
}
/// One 16-column step for one output word: 16 BF16 weights, 16 BF16 activations and the carried FP32 accumulator
/// (bits) give the new accumulator (bits). The integer function of `mma.sync m16n8k16 bf16` for one word.
/// Device facts (measured 2026-11-15 on short chains of tiny, subnormal and zero operands): a term that is zero (a
/// product with a zero operand, a zero carried value) brings NO exponent to the alignment, and a result that is cut
/// to zero is +0.0.
#[inline(always)]
pub fn bf16_step(weight: &[u16; STEP], input: &[u16; STEP], acc: u32) -> u32 {
    if acc & 0x7f80_0000 == 0x7f80_0000 { return acc; }
    const NONE: i32 = i32::MIN;
    let (ae, af) = (((acc >> 23) & 255) as i32, acc & 0x7f_ffff);
    let (am, alow, atop) = if ae == 0 { (af, -149, -126) } else { (af | 0x80_0000, ae - 150, ae - 127) };
    let mut magnitude = [0u32; STEP];
    let mut low = [0i32; STEP];
    let mut top = if am != 0 { atop } else { NONE };
    let mut k = 0;
    while k < STEP {
        let ((ws, we), (xs, xe)) = (split(weight[k]), split(input[k]));
        magnitude[k] = ws * xs;
        low[k] = we + xe - 14;
        if magnitude[k] != 0 && we + xe > top { top = we + xe; }
        k += 1;
    }
    if top == NONE { return 0; }
    let unit = top - 25;
    // A term that is not zero starts at most 11 bits over the unit; a zero term is placed nowhere.
    let place = |m: u32, at: i32| -> i64 {
        let shift = at - unit;
        (if m == 0 || shift <= -64 { 0 } else if shift >= 0 { u64::from(m) << shift } else { u64::from(m) >> (-shift) }) as i64
    };
    let mut sum = place(am, alow);
    if acc >> 31 != 0 { sum = -sum; }
    k = 0;
    while k < STEP {
        let value = place(magnitude[k], low[k]);
        if (weight[k] ^ input[k]) & 0x8000 != 0 { sum -= value; } else { sum += value; }
        k += 1;
    }
    let m = sum.unsigned_abs();
    if m == 0 { return 0; }
    let sign = u32::from(sum < 0) << 31;
    let high = 63 - m.leading_zeros() as i32 + unit;
    if high > 127 { return sign | 0x7f80_0000; }
    let kept_from = if high < -126 { -149 } else { high - 23 };
    let shift = kept_from - unit;
    let kept = if shift <= 0 { m << (-shift) } else if shift >= 64 { 0 } else { m >> shift };
    if kept == 0 { return 0; }
    if high < -126 { sign | kept as u32 } else { sign | (((high + 127) as u32) << 23) | (kept as u32 & 0x7f_ffff) }
}
/// Step 2: an FP32 value rounded once to BF16, round to nearest even (the bits `cvt.rn.bf16x2.f32` gives a finite value).
#[inline(always)]
pub const fn bf16(x: f32) -> u16 { let b = x.to_bits(); (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16 }
/// Scale blocks of range `part` of `blocks` scale blocks.
#[inline(always)]
pub const fn range(blocks: usize, part: usize) -> (usize, usize) { (blocks * part / 4, blocks * (part + 1) / 4) }
/// Step 3: the sum of the scale block of columns `[start, end)`. `code(c)` is the weight byte and `x(c)` the
/// activation of column `c`.
#[inline(always)]
pub fn block_sum(start: usize, end: usize, code: &impl Fn(usize) -> u8, x: &impl Fn(usize) -> f32) -> u32 {
    let (mut acc, mut column) = (0u32, start);
    while column < end {
        let (mut weight, mut input) = ([0u16; STEP], [0u16; STEP]);
        let mut k = 0;
        while k < STEP && column + k < end {
            weight[k] = e4m3_bf16(code(column + k));
            input[k] = bf16(x(column + k));
            k += 1;
        }
        acc = bf16_step(&weight, &input, acc);
        column += STEP;
    }
    acc
}
/// Steps 3 to 5 for the scale blocks `[first, end)`: the range accumulator. `scale(b)` is the scale of block `b`.
#[inline(always)]
pub fn range_sum(cols: usize, block_cols: usize, first: usize, end: usize, code: &impl Fn(usize) -> u8, x: &impl Fn(usize) -> f32, scale: &impl Fn(usize) -> f32) -> f32 {
    let (mut sum, mut block) = (0.0f32, first);
    while block < end {
        let start = block * block_cols;
        let stop = if start + block_cols < cols { start + block_cols } else { cols };
        let scaled = f32::from_bits(block_sum(start, stop, code, x)) * scale(block);
        sum += scaled;
        block += 1;
    }
    sum
}
/// Steps 1 to 6: the joined FP32 accumulator of one word. `raw`: ONE range over all blocks and no join (a raw job).
#[inline(always)]
pub fn accumulator(cols: usize, block_cols: usize, raw: bool, code: &impl Fn(usize) -> u8, x: &impl Fn(usize) -> f32, scale: &impl Fn(usize) -> f32) -> f32 {
    let blocks = cols.div_ceil(block_cols);
    if raw { return range_sum(cols, block_cols, 0, blocks, code, x, scale); }
    let part = |p: usize| { let (first, end) = range(blocks, p); range_sum(cols, block_cols, first, end, code, x, scale) };
    let (low, high) = (part(0) + part(1), part(2) + part(3));
    low + high
}
