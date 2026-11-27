//! Software round-to-nearest-even conversion shared by host and GPU code.

/// Match cvt.rn.bf16.f32's canonical quiet NaN observed by tensor-probe.
/// Preserve the old bit trick for every non-NaN, including overflow to infinity.
#[inline(always)]
pub fn rounded(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = if bits & 0x7fff_ffff > 0x7f80_0000 { 0x7fff_0000 }
        else { bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000 };
    f32::from_bits(rounded)
}

#[cfg(test)]
#[test]
fn nan_and_infinity() {
    for bits in [0x7f80_0001, 0x7fff_ffff, 0xff80_0001, 0xffff_ffff] { assert_eq!(rounded(f32::from_bits(bits)).to_bits(), 0x7fff_0000); }
    for bits in [0, 0x8000_0000, 0x7f80_0000, 0xff80_0000] { assert_eq!(rounded(f32::from_bits(bits)).to_bits(), bits); }
}
