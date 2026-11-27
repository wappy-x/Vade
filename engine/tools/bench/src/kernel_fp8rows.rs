//! Same-bytes sweep and timing for `fp8_rows_tile` against the production `expert_group` Fp8Rows path, in
//! one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
fn e2m1(n: u8) -> f32 {
    let magnitude = u32::from(n & 7);
    let bits = if magnitude < 2 { magnitude * 0x3f00_0000 } else { ((magnitude / 2 + 126) << 23) | ((magnitude & 1) << 22) };
    f32::from_bits(bits | (u32::from(n & 8) << 28))
}
fn e4m3(b: u8) -> f32 {
    let (e, f) = ((b >> 3) & 15, b & 7);
    let v = if e == 0 { f as f32 / 512.0 } else if b & 127 == 127 { f32::NAN } else { f32::from_bits(((u32::from(e) + 120) << 23) | (u32::from(f) << 20)) };
    if b & 128 == 0 { v } else { -v }
}
fn lut() -> Vec<f32> { let mut v = vec![]; for b in 0..=255u8 { v.push(e2m1(b & 15)); v.push(e2m1(b >> 4)); } for b in 0..=255u8 { v.push(e4m3(b)); } v }
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn bf16(x: f32) -> f32 { let b = x.to_bits(); if x.is_nan() { return x; } f32::from_bits(((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16) }
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
/// Finite BF16-rounded activations. 0 plain, 1 wide range, 2 denormal and +-0, 3 large.
fn activations(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| { let (u, s) = (r.unit(), r.next()); bf16(match kind {
        0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        _ => u.signum() * (1.0 + u.abs()) * 2f32.powi(40) }) }).collect()
}
#[derive(Clone, Copy)]
struct Geometry { rows: usize, cols: usize, br: usize, bc: usize, tokens: usize }
fn entry(g: Geometry, weight: u64, scales: u64) -> Vec<u8> {
    let mut v = vec![];
    for x in [7u32, g.rows as u32, g.cols as u32, g.br as u32, g.bc as u32, g.tokens as u32] { v.extend(x.to_le_bytes()); }
    for x in [1.0f32, 1.0, 0.0] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]);
    v.extend(weight.to_le_bytes()); v.extend(scales.to_le_bytes());
    let scale_words = g.rows.div_ceil(g.br) * g.cols.div_ceil(g.bc);
    for x in [0u32, 0, (g.tokens * g.cols) as u32, (g.rows * g.cols) as u32, (scale_words * 4) as u32, 0] { v.extend(x.to_le_bytes()); }
    v
}
fn args(table: u64, input: u64, output: u64, lutp: u64, units: usize, stride: usize) -> Vec<u8> {
    let mut v = vec![]; for x in [table, input, output, lutp] { v.extend(x.to_le_bytes()); }
    for x in [1u32, units as u32, 1, stride as u32] { v.extend(x.to_le_bytes()); } v
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Fixture { geo: Geometry, table: u64, input: u64, lutp: u64, units: usize }
/// mode 1: code (row + column) % 256 with NaN codes remapped, scale from a fixed table.
fn build(g: &mut Gpu, geo: Geometry, mode: u32, kind: u32, seed: u64) -> Result<Fixture> {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..geo.rows * geo.cols).map(|i| { let code = if mode == 1 { ((i / geo.cols + i % geo.cols) % 256) as u8 } else { (r.next() >> 8) as u8 }; if code & 127 == 127 { code ^ 1 } else { code } }).collect();
    let scales: Vec<f32> = (0..geo.rows.div_ceil(geo.br) * geo.cols.div_ceil(geo.bc)).map(|_| { let x = r.next(); let e = (x % 24) as i32 - 14; (1.0 + ((x >> 8) % 1000) as f32 / 1000.0) * 2f32.powi(e) }).collect();
    let wb = Buf::new(g, &w)?; let sb = Buf::new(g, &le(&scales))?; let xb = Buf::new(g, &le(&activations(kind, geo.tokens * geo.cols, seed + 5)))?;
    let table = Buf::new(g, &entry(geo, wb.address, sb.address))?; let l = Buf::new(g, &le(&lut()))?;
    Ok(Fixture { geo, table: table.address, input: xb.address, lutp: l.address, units: geo.rows * geo.tokens })
}
fn launch_reference(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.units, fx.geo.cols);
    g.launch(f, [fx.units.div_ceil(32) as u32, 1, 1], 32, 0, &mut [a.as_mut_ptr().cast()])
}
fn launch_tile(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.units, fx.geo.cols);
    g.launch(f, [fx.geo.tokens.div_ceil(64) as u32, fx.geo.rows.div_ceil(64) as u32, 1], 256, 0, &mut [a.as_mut_ptr().cast()])
}
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("FP8ROWS baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "expert_group")?; let tile = g.function(mb, "fp8_rows_tile")?;
    let started = Instant::now(); let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
    let check = |g: &mut Gpu, geo: Geometry, mode: u32, kind: u32, seed: u64| -> Result<()> {
        let fx = build(g, geo, mode, kind, seed)?;
        let out_a = Buf::new(g, &vec![0xff; fx.units * 4])?; let out_b = Buf::new(g, &vec![0xff; fx.units * 4])?;
        out_a.poison(g)?; out_b.poison(g)?;
        launch_reference(g, reference, &fx, out_a.address)?; launch_tile(g, tile, &fx, out_b.address)?; g.sync()?;
        let (x, y) = (out_a.read(g)?, out_b.read(g)?);
        if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("FP8ROWS BYTE FAIL rows={} cols={} br={} bc={} tokens={} mode={mode} act_kind={kind} first_word_and_byte={at:?}", geo.rows, geo.cols, geo.br, geo.bc, geo.tokens)); }
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("FP8ROWS guard changed".into()); }
        comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (fx.units * 4) as u64); g.clear()
    };
    // Every non-NaN weight code, in every row, with varied scales.
    for (kind, tokens, br, bc) in [(0u32, 3usize, 128usize, 128usize), (1, 17, 16, 16), (2, 9, 32, 64), (3, 65, 1, 16), (0, 64, 64, 256)] { check(&mut g, Geometry { rows: 512, cols: 256, br, bc, tokens }, 1, kind, 11 + kind as u64)?; }
    println!("FP8ROWS all codes ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    let mut seed = 500u64;
    for &(br, bc) in &[(128usize, 128usize), (16, 16), (32, 64), (1, 32), (64, 256)] {
        for &rows in &[1usize, 2, 31, 63, 64, 65, 129, 200, 1000] {
            for &cols in &[16usize, 32, 48, 64, 80, 128, 192, 256, 512, 1024, 4096] {
                for &tokens in &[1usize, 2, 7, 16, 17, 63, 64, 65, 100, 130, 300] {
                    if rows * cols * tokens > 60_000_000 || rows * tokens > 80_000 { continue; }
                    seed += 1; check(&mut g, Geometry { rows, cols, br, bc, tokens }, 0, (seed % 4) as u32, seed)?;
                }
            }
        }
    }
    println!("FP8ROWS sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    for (rows, cols, tokens, kind) in [(2048usize, 2048usize, 256usize, 0u32), (1024, 4096, 256, 1), (2048, 16384, 130, 2), (2048, 2048, 17, 3)] { check(&mut g, Geometry { rows, cols, br: 128, bc: 128, tokens }, 0, kind, 900 + rows as u64)?; }
    println!("FP8ROWS BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
    // Timing at the model's shapes, one group of 256 tokens.
    for (rows, cols) in [(2048usize, 2048usize), (1024, 4096), (2048, 16384)] {
        let fx = build(&mut g, Geometry { rows, cols, br: 128, bc: 128, tokens: 256 }, 0, 0, 31)?;
        let out = Buf::new(&mut g, &vec![0u8; fx.units * 4])?;
        let (mut ta, mut tb) = (vec![], vec![]);
        for repeat in 0..5 {
            for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                if arm == 0 { launch_reference(&g, reference, &fx, out.address)?; } else { launch_tile(&g, tile, &fx, out.address)?; } g.sync()?;
                let n = 3; let t0 = Instant::now();
                for _ in 0..n { if arm == 0 { launch_reference(&g, reference, &fx, out.address)?; } else { launch_tile(&g, tile, &fx, out.address)?; } }
                g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let (ra, rb) = (median(ta), median(tb)); let macs = (rows * cols * 256) as f64;
        println!("FP8ROWS TIME rows={rows} cols={cols} tokens=256 reference_us={ra:.0} tile_us={rb:.0} ratio={:.2} reference_us_per_token_row={:.2} tile_us_per_token_row={:.2} reference_GMACs={:.0} tile_GMACs={:.0}", ra / rb, ra / 256.0, rb / 256.0, macs / ra / 1e3, macs / rb / 1e3);
        g.clear()?;
    }
    println!("FP8ROWS PASS");
    Ok(())
}
