//! QSA FP32 contract, shared verbatim by our CPU oracle and CUDA kernels.
pub const DIM: usize = 128;
pub const HEADS: usize = 4;
pub const RATIO: usize = 4;
pub const BUDGET: usize = 2048;
pub const WIDTH: usize = BUDGET + RATIO - 1;
pub const LIMIT: usize = 1 << 20;
pub const CHUNK: usize = 1024;
pub type Rope = [f32; 64]; // cos[32], sin[32], supplied by the architecture layer.

/// Projected Q/K [rows,640], query RoPE [rows,64], completed-group RoPE
/// [floor((past+rows)/4)-floor(past/4),64], shared norm weights [128].
/// Ring [4,128] holds raw keys at absolute position % 4. Cache [capacity/4,128].
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Prepare {
    pub qk: u64, pub query_rope: u64, pub group_rope: u64,
    pub qw: u64, pub kw: u64, pub ring: u64, pub cache: u64, pub query: u64,
    pub rows: u32, pub past: u32, pub epsilon: f32,
}
/// Prepared queries [rows,4,128]; scores [rows,stride], stride >= last/4.
/// Selected [rows,2051] is ascending, u32::MAX padded; counts [rows] is u32.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Select {
    pub query: u64, pub cache: u64, pub scores: u64, pub selected: u64, pub counts: u64,
    pub rows: u32, pub past: u32, pub stride: u32,
}
/// Q/output [rows,24,256], K/V [capacity,2,256], weights [rows,24,2051].
/// Main Q/K are already normalized/rotated. Output precedes gate/projection.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Attention {
    pub query: u64, pub keys: u64, pub values: u64, pub selected: u64,
    pub counts: u64, pub weights: u64, pub output: u64, pub rows: u32,
}

#[inline(always)]
pub fn root(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    { let y; unsafe { core::arch::asm!("sqrt.rn.f32 {y}, {x};", y=out(reg32)y, x=in(reg32)x); } y }
    #[cfg(not(target_arch = "nvptx64"))]
    { x.sqrt() }
}
/// Nonpositive exponential: range reduction and FP32 Taylor polynomial.
/// Fixed operations avoid CPU libm vs GPU approximate-instruction differences.
#[inline(always)]
pub fn exponential(x: f32) -> f32 {
    if x < -104.0 { return 0.0; }
    let n = (x * core::f32::consts::LOG2_E) as i32;
    let r = (x - n as f32 * 0.69314575) - n as f32 * 1.4286068e-6;
    let mut p = 1.0 / 479001600.0;
    for c in [1.0/39916800.0, 1.0/3628800.0, 1.0/362880.0, 1.0/40320.0,
        1.0/5040.0, 1.0/720.0, 1.0/120.0, 1.0/24.0, 1.0/6.0, 0.5, 1.0, 1.0] { p = p * r + c; }
    // Split the exponent to preserve subnormals without a CPU/GPU FTZ mode.
    let m = if n < -126 { -126 } else { n };
    (p * f32::from_bits(((n - m + 127) as u32) << 23)) * f32::from_bits(((m + 127) as u32) << 23)
}
#[inline(always)]
pub fn normalize_rotate(mut x: [f32; DIM], w: &[f32; DIM], rope: &Rope, eps: f32) -> [f32; DIM] {
    let mut sum = 0.0;
    for v in x { sum += v * v; }
    let inv = 1.0 / root(sum / DIM as f32 + eps);
    for i in 0..DIM { x[i] = (x[i] * inv) * (1.0 + w[i]); }
    for i in 0..32 {
        let (a, b) = (x[i], x[i + 32]);
        x[i] = a * rope[i] - b * rope[i + 32];
        x[i + 32] = b * rope[i] + a * rope[i + 32];
    }
    x
}
#[inline(always)]
pub fn dot(q: &[f32], k: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..q.len() { s += q[i] * k[i]; }
    s
}
#[inline(always)]
pub fn score(q: &[f32], k: &[f32]) -> f32 {
    let mut sum = 0.0;
    for h in 0..HEADS { sum += dot(&q[h * DIM..(h + 1) * DIM], k).max(0.0); }
    sum / root(DIM as f32)
}
