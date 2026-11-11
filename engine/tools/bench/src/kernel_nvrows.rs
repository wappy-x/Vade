//! Same-bytes sweep and timing for `nv_rows_tile` against the production `expert_group` NvRows path, in
//! one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
#[path = "kernel_nvwide.rs"] mod nvwide;
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
/// Mirror of gpu/src/expert_ops.rs decode_lut: 256 (low, high) pairs then 256 E4M3 values.
fn lut() -> Vec<f32> { let mut v = vec![]; for b in 0..=255u8 { v.push(e2m1(b & 15)); v.push(e2m1(b >> 4)); } for b in 0..=255u8 { v.push(e4m3(b)); } v }
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
/// Activation rows: per 16-column block, 16 values then the block scale.
fn activations(kind: u32, blocks: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1); let mut v = vec![];
    for i in 0..blocks * 17 {
        let (u, s) = (r.unit(), r.next());
        v.push(if i % 17 == 16 { match kind { 5 => e4m3(0x30 + (s % 24) as u8), 2 => if s % 3 == 0 { 0.0 } else { 0.015625 * (1.0 + u.abs()) }, 3 => 1024.0 * (1.0 + u.abs()), _ => 0.25 * (1.0 + u.abs()) } } else { match kind {
            5 => e2m1((s % 16) as u8),
            0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
            _ => if i % 5003 == 5 { f32::NAN } else if i % 5003 == 21 { f32::INFINITY } else { u * 8.0 } } });
    }
    v
}
struct Part { m: usize, weight: u64, scales: u64 }
fn entry(rows: usize, cols: usize, p: &Part, first: usize, first_row: usize, params: (f32, f32)) -> Vec<u8> {
    let mut v = vec![];
    for x in [6u32, rows as u32, cols as u32, 1, 16, p.m as u32] { v.extend(x.to_le_bytes()); }
    for x in [params.0, params.1, 0.0f32] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]);
    v.extend(p.weight.to_le_bytes()); v.extend(p.scales.to_le_bytes());
    for x in [first as u32, first_row as u32, (p.m * (cols / 16 * 17)) as u32, (rows * cols / 2) as u32, (rows * cols / 16) as u32, 0] { v.extend(x.to_le_bytes()); }
    v
}
fn args(table: u64, input: u64, output: u64, lut: u64, count: usize, units: usize, stride: usize) -> Vec<u8> {
    let mut v = vec![]; for x in [table, input, output, lut] { v.extend(x.to_le_bytes()); }
    for x in [count as u32, units as u32, 1, stride as u32] { v.extend(x.to_le_bytes()); } v
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Fixture { rows: usize, cols: usize, parts: Vec<Part>, table: u64, input: u64, lutp: u64, units: usize, params: (f32, f32) }
/// Build parts with the given token counts. `mode` 1: row r has weight byte r%256 and scale byte r/256 everywhere.
fn build(g: &mut Gpu, rows: usize, cols: usize, counts: &[usize], mode: u32, kind: u32, params: (f32, f32), seed: u64, shared_weights: Option<&[(u64, u64)]>) -> Result<Fixture> {
    let mut r = Rng(seed | 1); let mut parts = vec![]; let mut input = vec![];
    let l = Buf::new(g, &le(&lut()))?;
    for (e, &m) in counts.iter().enumerate() {
        let (w, s) = if let Some(sw) = shared_weights { sw[e % sw.len()] } else {
            let w: Vec<u8> = (0..rows * cols / 2).map(|i| if mode == 1 { ((i / (cols / 2)) & 255) as u8 } else { r.next() as u8 }).collect();
            let s: Vec<u8> = (0..rows * cols / 16).map(|i| if mode == 1 { ((i / (cols / 16)) >> 8 & 255) as u8 } else { let x = r.next(); if x.is_multiple_of(16) { (x >> 8) as u8 } else { 0x20 + ((x >> 8) % 56) as u8 } }).collect();
            (Buf::new(g, &w)?.address, Buf::new(g, &s)?.address) };
        parts.push(Part { m, weight: w, scales: s });
        for t in 0..m { input.extend(activations(kind, cols / 16, seed + 977 * e as u64 + t as u64)); }
    }
    let (mut table, mut first, mut first_row) = (vec![], 0usize, 0usize);
    for p in &parts { table.extend(entry(rows, cols, p, first, first_row, params)); first += rows * p.m; first_row += p.m; }
    Ok(Fixture { rows, cols, table: Buf::new(g, &table)?.address, input: Buf::new(g, &le(&input))?.address, lutp: l.address, units: first, params, parts })
}
fn launch_reference(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.parts.len(), fx.units, fx.cols / 16 * 17);
    g.launch(f, [fx.units.div_ceil(32) as u32, 1, 1], 32, 0, &mut [a.as_mut_ptr().cast()])
}
fn launch_tile(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.parts.len(), fx.units, fx.cols / 16 * 17);
    let max_m = fx.parts.iter().map(|p| p.m).max().unwrap_or(1);
    // nv_rows_tile geometry since lane nv-wide: 256 rows and 128 threads a block. NV_ROWS_TILE_OLD=1 launches an image
    // built before it (128 rows, 64 threads).
    let (rows, threads) = if std::env::var_os("NV_ROWS_TILE_OLD").is_some() { (128, 64) } else { (256, 128) };
    g.launch(f, [max_m.div_ceil(8) as u32, fx.rows.div_ceil(rows) as u32, fx.parts.len() as u32], threads, 0, &mut [a.as_mut_ptr().cast()])
}
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    if std::env::var_os("NV_WIDE").is_some() { return nvwide::run(base_path, candidate_path); }
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("NVROWS baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "expert_group")?; let tile = g.function(mb, "nv_rows_tile")?;
    let started = Instant::now(); let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
    let _ = ptr::<u8>;
    let check = |g: &mut Gpu, fx: &Fixture, label: &str| -> Result<()> {
        let out_a = Buf::new(g, &vec![0xff; fx.units * 4])?; let out_b = Buf::new(g, &vec![0xff; fx.units * 4])?;
        out_a.poison(g)?; out_b.poison(g)?;
        launch_reference(g, reference, fx, out_a.address)?; launch_tile(g, tile, fx, out_b.address)?; g.sync()?;
        let (x, y) = (out_a.read(g)?, out_b.read(g)?);
        if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("NVROWS BYTE FAIL {label} rows={} cols={} parts={:?} params={:?} first_word_and_byte={at:?}", fx.rows, fx.cols, fx.parts.iter().map(|p| p.m).collect::<Vec<_>>(), fx.params)); }
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("NVROWS guard changed".into()); }
        comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (fx.units * 4) as u64); Ok(())
    };
    // All 256 weight bytes x all 256 scale bytes.
    for (kind, params, counts) in [(0u32, (0.75f32, 0.125f32), vec![3usize]), (1, (1.0, 1.0), vec![17]), (2, (0.0009765625, 0.5), vec![5, 16]), (3, (3.0, 0.0078125), vec![1]), (4, (0.75, 0.125), vec![2, 33])] {
        let fx = build(&mut g, 65536, 128, &counts, 1, kind, params, 11 + kind as u64, None)?; check(&mut g, &fx, "all-pairs")?; g.clear()?;
    }
    println!("NVROWS all 256x256 code pairs ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Family sweep.
    let counts_list: Vec<Vec<usize>> = vec![vec![1], vec![7], vec![16], vec![17], vec![31, 32, 33], vec![1, 2, 3, 50], vec![100], vec![64, 1, 16], vec![15, 15, 15, 15, 15, 15, 15, 15], vec![9, 4, 22, 1, 35, 16, 8], vec![1, 1, 1, 2, 3, 7, 5, 8, 9, 4, 6, 1]];
    let mut seed = 500u64;
    for &rows in &[1usize, 2, 31, 32, 33, 127, 128, 129, 200, 256, 257, 1024] {
        for &cols in &[128usize, 256, 384, 512, 1024, 2048, 4096] {
            for counts in &counts_list {
                if rows * cols * counts.len() > 24_000_000 || rows * cols > 4_000_000 && counts.len() > 3 { continue; }
                seed += 1; let fx = build(&mut g, rows, cols, counts, 0, (seed % 5) as u32, if seed.is_multiple_of(2) { (0.75, 0.125) } else { (1.0 / 3.0, 7.0) }, seed, None)?; check(&mut g, &fx, "sweep")?; g.clear()?;
            }
        }
    }
    println!("NVROWS sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Real layer shapes: 64 uneven expert parts, gate/up and down.
    let mut r = Rng(4242);
    let counts: Vec<usize> = (0..64).map(|_| 1 + (r.next() % 40) as usize).collect();
    for (rows, cols, kind) in [(1024usize, 4096usize, 0u32), (2048, 2048, 0), (1024, 4096, 2), (2048, 2048, 4), (1024, 4096, 1)] {
        let fx = build(&mut g, rows, cols, &counts, 0, kind, (0.75, 0.125), 900 + rows as u64 + kind as u64, None)?; check(&mut g, &fx, "real")?; g.clear()?;
    }
    println!("NVROWS BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
    // Timing: one layer's launch, 288 expert parts with uneven token counts, mean 7 (about 2,000 token-expert rows).
    let mut r = Rng(99);
    for (rows, cols, label) in [(1024usize, 4096usize, "gate_up"), (2048, 2048, "down")] {
        let counts: Vec<usize> = (0..288).map(|_| 1 + (r.next() % 13) as usize).collect(); let tokens: usize = counts.iter().sum();
        let mut sw = vec![]; let mut r2 = Rng(7);
        for _ in 0..288 {
            let w: Vec<u8> = (0..rows * cols / 2).map(|_| r2.next() as u8).collect(); let s: Vec<u8> = (0..rows * cols / 16).map(|_| 0x30 + (r2.next() % 20) as u8).collect();
            sw.push((Buf::new(&mut g, &w)?.address, Buf::new(&mut g, &s)?.address));
        }
        let fx = build(&mut g, rows, cols, &counts, 0, 0, (0.75, 0.125), 31, Some(&sw))?;
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
        let (ra, rb) = (median(ta), median(tb)); let macs = (rows * cols * tokens) as f64;
        println!("NVROWS TIME {label} rows={rows} cols={cols} parts=288 token_rows={tokens} reference_us={ra:.0} tile_us={rb:.0} ratio={:.2} reference_us_per_token_row={:.2} tile_us_per_token_row={:.2} reference_GMACs={:.0} tile_GMACs={:.0}", ra / rb, ra / tokens as f64, rb / tokens as f64, macs / ra / 1e3, macs / rb / 1e3);
        g.clear()?;
    }
    println!("NVROWS PASS");
    Ok(())
}
