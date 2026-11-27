//! Integer hash and row arithmetic shared by the host oracle tests and PTX.
pub fn rows(token: u32, hist: &mut [i64; 2], p: &[i64; 35]) -> [u64; 16] {
    let prev = if hist[0] == 248044 { [248044; 2] } else { *hist };
    let bigram = (token as i64).wrapping_mul(p[0]) ^ prev[0].wrapping_mul(p[1]);
    let trigram = bigram ^ prev[1].wrapping_mul(p[2]);
    let mut out = [0; 16];
    for h in 0..16 { out[h] = ((if h < 8 { bigram } else { trigram }).rem_euclid(p[3+h]) + p[19+h]) as u64; }
    *hist = [token as i64, prev[0]]; out
}
// The reference rounds only the global multiply; keep the row in FP32.
pub fn dequant(packed: u8, high: bool, scale: u8, global: f32) -> f32 {
    let n = if high { packed >> 4 } else { packed & 15 };
    let v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(n & 7) as usize];
    let e = (scale >> 3) & 15;
    let s = if e == 0 { (scale & 7) as f32 / 512.0 }
        else { f32::from_bits(((e as u32 + 120) << 23) | (((scale & 7) as u32) << 20)) };
    (if n & 8 == 0 { v } else { -v }) * s * global
}
