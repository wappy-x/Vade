//! Host fixtures for the mHC byte kit. Source math has an independent oracle.
#![forbid(unsafe_code)]
use crate::{Result, mhc_math as math, mhc_params::Shape};
use engine_model::mhc::{Config, Reference, Weights};
pub fn bytes(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
pub fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 13 + i / 7 + salt) % 31) as f32 / 16.0 - 0.9375).collect()
}
pub fn shapes() -> Vec<Shape> {
    let saved = Config::from_json(include_bytes!("../../../crates/model/tests/fixtures/glm/config.json")).unwrap();
    [(1, 3, 1, 1), (2, 2, 1, 3), (3, 7, 2, 5), (5, 129, 3, 17),
        (saved.hc_mult as u32, saved.hidden_size as u32, saved.hc_sinkhorn_iters as u32, 3)]
        .into_iter().map(|(streams, hidden, iterations, rows)| Shape { rows, streams, hidden, iterations,
            rms_eps: saved.rms_norm_eps, eps: saved.hc_eps }).collect()
}
pub struct Fixture {
    pub shape: Shape, pub input: Vec<f32>, pub weights: Vec<f32>, pub base: Vec<f32>,
    pub scale: Vec<f32>, pub block: Vec<f32>,
}
#[derive(Debug, PartialEq)]
pub struct Output {
    pub normalized: Vec<f32>, pub coefficients: Vec<f32>, pub collapsed: Vec<f32>,
    pub combined: Vec<f32>, pub head: Vec<f32>,
}
impl Output {
    pub fn buffers(&self) -> [&[f32]; 5] {
        [&self.normalized, &self.coefficients, &self.collapsed, &self.combined, &self.head]
    }
}
impl Fixture {
    pub fn new(shape: Shape) -> Self {
        shape.validate().unwrap(); let t = shape.rows as usize; let p = shape.coefficients();
        Self { shape, input: data(t * shape.wide(), 3),
            weights: data(p * shape.wide(), 11).into_iter().map(|v| v / 64.0).collect(),
            base: data(p, 7), scale: vec![0.25, 0.5, 0.125], block: data(t * shape.hidden as usize, 23) }
    }
    pub fn reference(&self) -> Output {
        let s = self.shape; let n = s.streams as usize; let h = s.hidden as usize;
        let (w, p, t) = (s.wide(), s.coefficients(), s.rows as usize);
        let mut out = Output { normalized: vec![0.0; t * w], coefficients: vec![0.0; t * p],
            collapsed: vec![0.0; t * h], combined: vec![0.0; t * w], head: vec![0.0; t * h] };
        for r in 0..t {
            let x = &self.input[r * w..(r + 1) * w];
            let c = &mut out.coefficients[r * p..(r + 1) * p];
            math::prepare(s, x, &self.weights, &self.base, &self.scale, &mut out.normalized[r * w..(r + 1) * w], c);
            for d in 0..h {
                out.collapsed[r * h + d] = math::collapse(x, c, n, h, d);
                for dst in 0..n { out.combined[r * w + dst * h + d] = math::combine(x, c, self.block[r * h + d], n, h, dst, d); }
            }
            let y = &out.combined[r * w..(r + 1) * w];
            for d in 0..h { out.head[r * h + d] = math::head(y, n, h, d); }
        }
        out
    }
    /// This independent numerical check does not establish GPU byte equality.
    pub fn source_check(&self, out: &Output) -> Result<f32> {
        let s = self.shape; let h = s.hidden as usize; let n = s.streams as usize;
        let c = Config { hidden_size: h, hc_mult: n, hc_sinkhorn_iters: s.iterations as usize,
            rms_norm_eps: s.rms_eps, hc_eps: s.eps };
        let reference = Reference::new(c, Weights { projection: &self.weights, base: &self.base, scale: &self.scale }).map_err(|e| e.to_string())?;
        let mut peak = 0.0f32;
        let mut near = |a: &[f32], b: &[f32]| -> Result<()> {
            if a.len() != b.len() { return Err("mHC source oracle length mismatch".into()); }
            for (&a, &b) in a.iter().zip(b) {
                let error = (a - b).abs(); peak = peak.max(error);
                if !a.is_finite() || !b.is_finite() || error > 3e-6 * b.abs().max(1.0) {
                    return Err(format!("mHC source oracle: actual={a} expected={b} error={error}"));
                }
            }
            Ok(())
        };
        for r in 0..s.rows as usize {
            let x = &self.input[r * s.wide()..(r + 1) * s.wide()];
            let mixed = reference.mix(x).map_err(|e| e.to_string())?;
            let coeff = [&mixed.pre[..], &mixed.post[..], &mixed.comb[..]].concat();
            let y = reference.combine(x, &mixed, &self.block[r * h..(r + 1) * h]).map_err(|e| e.to_string())?;
            let head = c.mean_head(&y).map_err(|e| e.to_string())?;
            near(&out.coefficients[r * s.coefficients()..(r + 1) * s.coefficients()], &coeff)?;
            near(&out.collapsed[r * h..(r + 1) * h], &mixed.collapsed)?;
            near(&out.combined[r * s.wide()..(r + 1) * s.wide()], &y)?;
            near(&out.head[r * h..(r + 1) * h], &head)?;
        }
        Ok(peak)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_source_math_covers_all_device_fixtures() {
        for shape in shapes() {
            let fixture = Fixture::new(shape); let out = fixture.reference();
            let peak = fixture.source_check(&out).unwrap();
            println!("mHC host source N={} D={} rows={} iterations={} max_abs={peak} tolerance=0.000003*max(1,abs(expected))",
                shape.streams, shape.hidden, shape.rows, shape.iterations);
        }
    }
    #[test]
    fn head_and_residual_orientation_have_independent_hand_answers() {
        let x = [1.0, 2.0, 3.0, 4.0]; let c = [0.75, 0.25, 0.5, 1.5, 0.5, 0.25, 0.125, 0.75];
        assert_eq!(math::collapse(&x, &c, 2, 2, 0), 1.5);
        assert_eq!(math::combine(&x, &c, 4.0, 2, 2, 0, 0), 2.875);
        assert_eq!(math::combine(&x, &c, 4.0, 2, 2, 1, 0), 8.5);
        assert_eq!(math::head(&[16777216.0, 1.0, -16777216.0], 3, 1, 0).to_bits(), 0.0f32.to_bits());
    }
    #[test]
    fn abi_sizes_and_invalid_shapes() {
        use crate::mhc_params::*; use std::mem::size_of;
        assert_eq!([size_of::<Shape>(), size_of::<Prepare>(), size_of::<Collapse>(), size_of::<Combine>(), size_of::<Head>()], [24, 80, 40, 48, 32]);
        let good = shapes()[0];
        for shape in [Shape { rows: 0, ..good }, Shape { streams: 0, ..good }, Shape { hidden: 0, ..good },
            Shape { iterations: 0, ..good }, Shape { iterations: 1_048_577, ..good },
            Shape { streams: u32::MAX, ..good }, Shape { rows: u32::MAX, ..good },
            Shape { eps: f32::INFINITY, ..good }, Shape { eps: 0.0, ..good },
            Shape { rms_eps: f32::NAN, ..good }, Shape { rms_eps: -1.0, ..good }] { assert!(shape.validate().is_err()); }
    }
}
