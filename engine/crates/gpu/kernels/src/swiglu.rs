//! Element arithmetic for the standalone SwiGLU epilogue. Zero disables the bound.
#[inline(always)]
fn round(x: f32) -> f32 {
    let bits = x.to_bits();
    if bits & 0x7f800000 == 0x7f800000 { return x; }
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

/// Finite projection inputs and a validated bound. The exponential is supplied
/// by the caller. An uncapped launch retains its existing exponential.
#[inline(always)]
pub fn apply(gate: f32, up: f32, bf16: bool, limit: f32, exp: impl Fn(f32) -> f32) -> f32 {
    let (mut g, mut u) = if bf16 { (round(gate), round(up)) } else { (gate, up) };
    if limit > 0.0 {
        let limit = if bf16 { round(limit) } else { limit };
        if g > limit { g = limit; }
        if u > limit { u = limit; }
        if u < -limit { u = -limit; }
    }
    if bf16 { round(round(g / (1.0 + exp(-g))) * u) }
    else { g / (1.0 + exp(-g)) * u }
}
