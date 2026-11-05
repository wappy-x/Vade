//! Fixed FP32 mHC operations. No reduction depends on row count or occupancy.
use super::qsa_common::{dot, exponential, root};
use crate::mhc_params::Shape;

fn sigmoid(x: f32) -> f32 {
    // Reuse the QSA nonpositive exponential, including its fixed polynomial.
    let e = exponential(-x.abs());
    if x >= 0.0 { 1.0 / (1.0 + e) } else { e / (1.0 + e) }
}
/// One token. The caller supplies exact extents and finite intermediates.
pub fn prepare(s: Shape, x: &[f32], weights: &[f32], base: &[f32], scale: &[f32],
    normalized: &mut [f32], coefficients: &mut [f32]) {
    normalize(s,x,normalized);
    for i in 0..s.coefficients() { coefficients[i]=project(s,i,weights,base,scale,normalized); }
    activate(s,coefficients);
}
/// Keep the original ascending feature sum for each token's RMS boundary.
pub fn normalize(s:Shape,x:&[f32],normalized:&mut[f32]) {
    let wide=s.wide();
    let mut squares = 0.0f32;
    // Load independent products together. Keep every addition in the original
    // ascending order, including the tail. This does not split the reduction.
    let mut blocks=x.chunks_exact(8);
    for b in blocks.by_ref() {
        let a=b[0]*b[0];let c=b[1]*b[1];let d=b[2]*b[2];let e=b[3]*b[3];
        let f=b[4]*b[4];let g=b[5]*b[5];let h=b[6]*b[6];let i=b[7]*b[7];
        squares+=a;squares+=c;squares+=d;squares+=e;squares+=f;squares+=g;squares+=h;squares+=i;
    }
    for &v in blocks.remainder() {squares+=v*v;}
    let inv = 1.0 / root(squares / wide as f32 + s.rms_eps);
    for i in 0..wide { normalized[i] = x[i] * inv; }
}
/// Coefficient rows are independent. Each dot keeps its full ascending fold.
pub fn project(s:Shape,i:usize,weights:&[f32],base:&[f32],scale:&[f32],normalized:&[f32])->f32 {
    let n=s.streams as usize;let wide=s.wide();
    let group=if i<n {0} else if i<2*n {1} else {2};
    dot(&weights[i*wide..(i+1)*wide],normalized)*scale[group]+base[i]
}
/// Run only after all independent coefficient rows have completed.
pub fn activate(s:Shape,coefficients:&mut[f32]) {
    let n=s.streams as usize;
    for i in 0..n {
        coefficients[i] = sigmoid(coefficients[i]) + s.eps;
        coefficients[n + i] = 2.0 * sigmoid(coefficients[n + i]);
    }
    let comb = &mut coefficients[2 * n..];
    for row in comb.chunks_exact_mut(n) {
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() { max = max.max(v); }
        let mut sum = 0.0f32;
        for v in row.iter_mut() { *v = exponential(*v - max); sum += *v; }
        for v in row { *v = *v / sum + s.eps; }
    }
    columns(comb, n, s.eps);
    for _ in 1..s.iterations {
        for row in comb.chunks_exact_mut(n) {
            let mut sum = 0.0f32;
            for &v in row.iter() { sum += v; }
            let divisor = sum + s.eps;
            for v in row { *v /= divisor; }
        }
        columns(comb, n, s.eps);
    }
}
fn columns(comb: &mut [f32], n: usize, eps: f32) {
    for dst in 0..n {
        let mut sum = 0.0f32;
        for src in 0..n { sum += comb[src * n + dst]; }
        let divisor = sum + eps;
        for src in 0..n { comb[src * n + dst] /= divisor; }
    }
}
pub fn collapse(x: &[f32], pre: &[f32], n: usize, h: usize, d: usize) -> f32 {
    let mut value = 0.0f32;
    for src in 0..n { value += pre[src] * x[src * h + d]; }
    value
}
pub fn combine(x: &[f32], coefficients: &[f32], block: f32, n: usize, h: usize, dst: usize, d: usize) -> f32 {
    let mut value = 0.0f32;
    for src in 0..n { value += coefficients[2 * n + src * n + dst] * x[src * h + d]; }
    coefficients[n + dst] * block + value
}
pub fn head(x: &[f32], n: usize, h: usize, d: usize) -> f32 {
    let mut value = 0.0f32;
    for src in 0..n { value += x[src * h + d]; }
    value / n as f32
}
