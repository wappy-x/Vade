//! Portable GPU instruction-order reference. No GPU or payload I/O.
//! MMA tile rounding is a widened-dot hypothesis pending native calibration;
//! instruction boundaries are explicit, but hardware bit parity is unqualified.
use std::ops::Range;

pub const BLOCK: usize = 128;
pub type Result<T> = std::result::Result<T, &'static str>;

/// E4M3FN: signed zero, subnormals, finite exponent 15, and two signed NaNs.
pub fn decode_e4m3(code: u8) -> f32 {
    let sign = u32::from(code & 128) << 24;
    let magnitude = code & 127;
    let exponent = magnitude >> 3;
    let fraction = magnitude & 7;
    let bits = if magnitude == 127 { 0x7fc0_0000 }
        else if exponent == 0 { (f32::from(fraction) / 512.0).to_bits() }
        else { ((u32::from(exponent) + 120) << 23) | (u32::from(fraction) << 20) };
    f32::from_bits(sign | bits)
}

/// Nearest even, finite saturation; NaNs retain their sign and become code 127.
pub fn encode_e4m3(value: f32) -> u8 {
    let sign = ((value.to_bits() >> 24) & 128) as u8;
    let magnitude = value.abs();
    if magnitude.is_nan() { return sign | 127; }
    if magnitude >= 448.0 { return sign | 126; }
    let (mut low, mut high) = (0u8, 126u8);
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if decode_e4m3(middle) <= magnitude { low = middle; } else { high = middle; }
    }
    let midpoint = (decode_e4m3(low) + decode_e4m3(high)) * 0.5;
    sign | if magnitude > midpoint || (magnitude == midpoint && low & 1 != 0) { high } else { low }
}

pub fn widen_bf16(bits: u16) -> f32 { f32::from_bits(u32::from(bits) << 16) }

/// BF16 nearest-even, including subnormals/overflow; preserve and quiet NaNs.
pub fn encode_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 { return (bits >> 16) as u16 | 0x0040; }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn round_bf16(value: f32) -> f32 { widen_bf16(encode_bf16(value)) }

/// Row-major codes and FP32 scales can be cloned/replayed without requantizing.
#[derive(Clone, Debug)]
pub struct Quantized {
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
}

/// Each row/K128 group owns its amax. Round BF16 input explicitly before calling.
/// Stored scale = amax/448, even zero; only the divisor is floored at 1e-12.
pub fn quantize(input: &[f32], cols: usize) -> Result<Quantized> {
    if cols == 0 || !cols.is_multiple_of(BLOCK) || !input.len().is_multiple_of(cols) {
        return Err("activation shape must contain complete K128 rows");
    }
    if input.iter().any(|v| !v.is_finite()) { return Err("non-finite activation"); }
    let mut codes = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len() / BLOCK);
    for group in input.chunks_exact(BLOCK) {
        let amax = group.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        let scale = amax / 448.0;
        scales.push(scale);
        codes.extend(group.iter().map(|value| encode_e4m3(*value / scale.max(1e-12))));
    }
    Ok(Quantized { rows: input.len() / cols, cols, codes, scales })
}

/// Weights [N,K], BF16 multipliers [N/128,K/128], output [M,N] in FP32.
/// `blocks` retains ORIGINAL K coordinates for a logical rank's partial sum.
/// fp8.rs:364-395: four dependent K32 MMAs, then dot*(sa*sw), ascending K128.
/// Empty ranges yield zero partials. Rank folding and BF16 boundaries are separate.
pub fn block_gemm(input: &Quantized, weights: &[u8], inverse_scales: &[u16],
    outputs: usize, blocks: Range<usize>) -> Result<Vec<f32>> {
    block_gemm_with(input, weights, inverse_scales, outputs, blocks, |a, b| {
        let mut acc = 0.0;
        for (a, b) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
            acc = mma_round(acc, a.iter().zip(b).map(|(&x, &w)|
                (decode_e4m3(x), decode_e4m3(w))));
        }
        acc
    })
}

/// Historical scalar FP32 dot product, retained to compare recorded results and errors.
pub fn block_gemm_scalar(input: &Quantized, weights: &[u8], inverse_scales: &[u16],
    outputs: usize, blocks: Range<usize>) -> Result<Vec<f32>> {
    block_gemm_with(input, weights, inverse_scales, outputs, blocks, |a, b|
        a.iter().zip(b).fold(0.0, |sum, (&x, &w)|
            sum + decode_e4m3(x) * decode_e4m3(w)))
}

// One FP32 accumulator boundary per MMA instruction. Widened products/sum do
// not specify the GPU's undocumented internal reduction or guard-bit behavior.
fn mma_round(acc: f32, products: impl Iterator<Item = (f32, f32)>) -> f32 {
    products.fold(f64::from(acc), |sum, (a, b)| sum + f64::from(a) * f64::from(b)) as f32
}

fn block_gemm_with(input: &Quantized, weights: &[u8], inverse_scales: &[u16],
    outputs: usize, blocks: Range<usize>, dot: impl Fn(&[u8], &[u8]) -> f32) -> Result<Vec<f32>> {
    let cols = input.cols;
    if cols == 0 || !cols.is_multiple_of(BLOCK) || !outputs.is_multiple_of(BLOCK) {
        return Err("matrix dimensions must be K128/N128 aligned");
    }
    let groups = cols / BLOCK;
    if input.rows.checked_mul(cols) != Some(input.codes.len())
        || input.rows.checked_mul(groups) != Some(input.scales.len())
        || outputs.checked_mul(cols) != Some(weights.len())
        || (outputs / BLOCK).checked_mul(groups) != Some(inverse_scales.len())
        || blocks.start > blocks.end || blocks.end > groups {
        return Err("matrix extent or original block range mismatch");
    }
    if input.codes.iter().chain(weights).any(|v| v & 127 == 127)
        || input.scales.iter().any(|v| !v.is_finite() || *v < 0.0)
        || inverse_scales.iter().any(|v| { let s = widen_bf16(*v); !s.is_finite() || s < 0.0 }) {
        return Err("non-finite code or invalid multiplier");
    }
    let count = input.rows.checked_mul(outputs).ok_or("output extent overflow")?;
    let mut result = vec![0.0; count];
    for row in 0..input.rows {
        for out in 0..outputs {
            let mut sum = 0.0f32;
            for block in blocks.clone() {
                let col = block * BLOCK;
                let dot = dot(&input.codes[row * cols + col..row * cols + col + BLOCK],
                    &weights[out * cols + col..out * cols + col + BLOCK]);
                // fp8.rs:379-389: widen BF16, multiply scales, multiply dot,
                // then add to FP32. Generated PTX has separate mul/add, no FMA.
                let scale = input.scales[row * groups + block]
                    * widen_bf16(inverse_scales[out / BLOCK * groups + block]);
                sum += dot * scale;
            }
            if !sum.is_finite() { return Err("non-finite projection result"); }
            result[row * outputs + out] = sum;
        }
    }
    Ok(result)
}

/// canonical.rs:23-27; ranges count K16 tiles, including uneven quarters.
pub fn canonical_k16_ranges(cols: usize) -> Result<[Range<usize>; 4]> {
    if cols == 0 || !cols.is_multiple_of(16) { return Err("BF16 K must be K16 aligned"); }
    let tiles = cols / 16;
    Ok(std::array::from_fn(|part| tiles * part / 4..tiles * (part + 1) / 4))
}

/// Row-major BF16 weights [N,K], FP32 inputs [M,K], FP32 partials [M,N].
/// gemm.rs:304-312 splits hi=RN_BF16(x), lo=RN_BF16(x-hi).
/// flash_bf16.rs:178,229-280 chains K16, hi+lo per range; :407-411 merges
/// (P0+P1)+(P2+P3). Round the global output separately with round_bf16.
/// This shares the provisional MMA primitive above; native parity is pending.
pub fn bf16_gemm(input: &[f32], weights: &[u16], cols: usize, outputs: usize) -> Result<Vec<f32>> {
    let ranges = canonical_k16_ranges(cols)?;
    if outputs == 0 || !input.len().is_multiple_of(cols)
        || outputs.checked_mul(cols) != Some(weights.len()) {
        return Err("BF16 matrix extent mismatch");
    }
    if input.iter().any(|v| !v.is_finite()) || weights.iter().any(|&v| !widen_bf16(v).is_finite()) {
        return Err("non-finite BF16 operand");
    }
    let split: Vec<_> = input.iter().map(|&x| {
        let hi = round_bf16(x); (hi, round_bf16(x - hi))
    }).collect();
    let rows = input.len() / cols;
    let mut result = vec![0.0; rows.checked_mul(outputs).ok_or("output extent overflow")?];
    for row in 0..rows {
        for out in 0..outputs {
            let mut partials = [0.0f32; 4];
            for (part, tiles) in ranges.iter().enumerate() {
                let (mut hi, mut lo) = (0.0, 0.0);
                for tile in tiles.clone() {
                    let operands = || (tile * 16..tile * 16 + 16).map(|c|
                        (split[row * cols + c], widen_bf16(weights[out * cols + c])));
                    hi = mma_round(hi, operands().map(|((h, _), w)| (h, w)));
                    lo = mma_round(lo, operands().map(|((_, l), w)| (l, w)));
                }
                partials[part] = hi + lo;
            }
            let value = (partials[0] + partials[1]) + (partials[2] + partials[3]);
            if !value.is_finite() { return Err("non-finite projection result"); }
            result[row * outputs + out] = value;
        }
    }
    Ok(result)
}

/// Start at rank zero and fold ascending ranks; no rounding before the fold.
pub fn fold_ranks(partials: &[&[f32]]) -> Result<Vec<f32>> {
    let first = partials.first().ok_or("missing rank zero")?;
    if partials.iter().any(|p| p.len() != first.len() || p.iter().any(|v| !v.is_finite())) {
        return Err("rank width mismatch or non-finite partial");
    }
    let mut sum = first.to_vec();
    for partial in &partials[1..] {
        for (value, addend) in sum.iter_mut().zip(*partial) { *value += *addend; }
    }
    if sum.iter().any(|v| !v.is_finite()) { return Err("non-finite rank sum"); }
    Ok(sum)
}

pub struct Expert<'a> {
    pub id: u32,
    pub weight: f32,
    pub ranks: &'a [&'a [f32]],
}

/// Ascending unique expert IDs, shared last. Rank fold -> BF16 down -> BF16
/// weight/product -> FP32 expert sum -> BF16 routed sum -> FP32 shared addition.
/// `shared_weight` is the already-evaluated sigmoid gate; residual is external.
/// Rejected rows are errors, never partially published outputs.
pub fn ordered_finish(experts: &[Expert<'_>], shared: &[&[f32]], shared_weight: f32) -> Result<Vec<f32>> {
    if !shared_weight.is_finite() || !(0.0..=1.0).contains(&shared_weight) {
        return Err("invalid shared gate");
    }
    let shared = fold_ranks(shared)?;
    let mut order: Vec<_> = experts.iter().collect();
    order.sort_by_key(|expert| expert.id);
    if order.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err("duplicate expert ID");
    }
    let mut sum = vec![0.0f32; shared.len()];
    for expert in order {
        if !expert.weight.is_finite() || !(0.0..=1.0).contains(&expert.weight) {
            return Err("invalid routing weight");
        }
        let down = fold_ranks(expert.ranks)?;
        if down.len() != sum.len() { return Err("expert width mismatch"); }
        for (value, partial) in sum.iter_mut().zip(down) {
            *value += round_bf16(round_bf16(partial) * round_bf16(expert.weight));
        }
    }
    for (value, partial) in sum.iter_mut().zip(shared) {
        *value = round_bf16(*value) + partial * shared_weight;
        if !value.is_finite() { return Err("non-finite expert result"); }
    }
    Ok(sum)
}
