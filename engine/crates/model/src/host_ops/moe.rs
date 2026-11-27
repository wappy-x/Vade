//! Source-order expert fold and ungated shared addition on one token.
use crate::{Result, invalid};

/// `partials` contains one down-projection row per router slot.
/// Fold experts in ascending ID, then add shared once, without a gate.
/// `round` supplies the source activation boundary: identity for FP32,
/// round-to-nearest-even BF16 for the saved checkpoint. Route weights stay FP32.
/// The caller supplies completed down projections. This is not a projection oracle.
pub fn ungated_shared(ids: &[usize], weights: &[f32], partials: &[f32], shared: &[f32],
    round: impl Fn(f32) -> f32) -> Result<Vec<f32>> {
    let width = shared.len();
    if ids.is_empty() || weights.len() != ids.len() || width == 0
        || ids.len().checked_mul(width) != Some(partials.len())
        || weights.iter().chain(partials).chain(shared).any(|v| !v.is_finite()) {
        return Err(invalid("invalid expert or shared rows"));
    }
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&slot| ids[slot]);
    if order.windows(2).any(|s| ids[s[0]] == ids[s[1]]) { return Err(invalid("expert IDs must be unique")); }
    let mut output = vec![0.0f32; width];
    for slot in order {
        for col in 0..width {
            let product = round(partials[slot * width + col]) * weights[slot];
            let contribution = round(product);
            output[col] = round(output[col] + contribution);
        }
    }
    for col in 0..width { output[col] = round(output[col] + round(shared[col])); }
    if output.iter().any(|v| !v.is_finite()) { return Err(invalid("expert sum exceeds the activation range")); }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(v: f32) -> f32 { let b = v.to_bits(); f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000) }
    #[test]
    fn expert_id_order_and_shared_last_are_fixed() {
        let run = |ids: &[usize], rows: &[f32]| ungated_shared(ids,&[1.0;3],rows,&[1.0],|v| v).unwrap();
        // Ascending ID: (2^24 + 1) - 2^24 = 0 in FP32. Shared adds 1 last.
        assert_eq!(run(&[2,0,1],&[-16777216.0,16777216.0,1.0]),[1.0]);
        assert_eq!(run(&[0,1,2],&[16777216.0,1.0,-16777216.0]),[1.0]);
    }
    #[test]
    fn shared_is_unweighted_and_not_sigmoid_gated() {
        assert_eq!(ungated_shared(&[7,2],&[0.25,0.5],&[8.0,4.0],&[3.0],|v| v).unwrap(),[7.0]);
    }
    #[test]
    fn bf16_rounds_each_expert_add_and_preserves_fp32_weights() {
        assert_eq!(ungated_shared(&[0,1,2],&[1.0;3],&[256.0,1.0,-256.0],&[0.0],bf16).unwrap(),[0.0]);
        assert_eq!(ungated_shared(&[0],&[1.003],&[256.0],&[-256.0],bf16).unwrap(),[0.0]);
        // Rounding the route weight before multiplication changes this product.
        let weight = 1.003f32;
        let down = 1.5f32;
        let actual = ungated_shared(&[0],&[weight],&[down],&[0.0],bf16).unwrap();
        assert_eq!(actual,[bf16(down * weight)]);
        assert_ne!(actual,[bf16(down * bf16(weight))]);
    }
    #[test]
    fn rejects_invalid_rows_and_overflow() {
        for (ids,weights,rows,shared) in [(vec![],vec![],vec![],vec![0.0]),
            (vec![1,1],vec![1.0;2],vec![1.0;2],vec![0.0]),
            (vec![1],vec![],vec![1.0],vec![0.0]),(vec![1],vec![1.0],vec![],vec![0.0]),
            (vec![1],vec![f32::NAN],vec![1.0],vec![0.0]),
            (vec![1],vec![2.0],vec![f32::MAX],vec![0.0]),
            (vec![1],vec![1.0],vec![0.0],vec![f32::INFINITY])] {
            assert!(ungated_shared(&ids,&weights,&rows,&shared,|v| v).is_err());
        }
    }
}
