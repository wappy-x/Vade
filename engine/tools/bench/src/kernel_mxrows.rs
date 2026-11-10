//! Same-bytes sweep and timing for `mx_rows_tile` against the production MXFP8 prompt-row path
//! (`apply_rows`: groups of 16 tokens, kernel chosen by token count), in one process on one device.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
#[allow(dead_code, clippy::duplicate_mod)] #[path = "../../../crates/gpu/src/swept.rs"] mod swept;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// Decoded activation values. 0 plain, 1 wide range, 2 denormal and +-0, 3 large, 4 sparse NaN and inf.
fn activations(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| { let (u, s) = (r.unit(), r.next()); match kind {
        0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
        _ => if i % 7919 == 5 { f32::NAN } else if i % 7919 == 21 { f32::INFINITY } else { u * 8.0 } } }).collect()
}
#[derive(Clone, Copy)]
struct Proj { weights: u64, scales: u64, input: u64, output: u64, rows: u32, cols: u32, tokens: u32 }
fn proj_bytes(p: &Proj) -> Vec<u8> {
    let mut v = vec![]; for x in [p.weights, p.scales, p.input, p.output] { v.extend(x.to_le_bytes()); }
    for x in [p.rows, p.cols, p.tokens, 0, 0] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]); v
}
/// Mirror of MxProjection::tile and kernel (round 0).
fn tile(p: &Proj) -> u32 {
    if p.tokens > 1 && p.tokens <= 8 { 8 } else if p.tokens == 1 || p.rows < 128 { 1 } else if p.rows <= 640 || p.cols >= 4096 { if p.tokens <= 2 { 2 } else { 4 } } else { p.tokens }
}
fn kernel(p: &Proj) -> &'static str { if p.tokens == 1 { "flash_next_mxfp8_m1" } else if p.tokens <= 8 { "flash_next_mxfp8_pair" } else { "flash_next_mxfp8_bits" } }
/// The production loop: groups of 16 tokens, one launch each.
fn launch_reference(g: &Gpu, fns: &[(&str, H)], p: &Proj) -> Result<()> {
    let mut token = 0u32;
    while token < p.tokens {
        let count = (p.tokens - token).min(16);
        let q = Proj { input: p.input + (token as u64) * (p.cols as u64) * 4, output: p.output + (token as u64) * (p.rows as u64) * 4, tokens: count, ..*p };
        let blocks = q.rows.div_ceil(4) * q.tokens.div_ceil(tile(&q));
        let mut layer = proj_bytes(&q); layer.resize(9 * 56, 0);
        let mut ends = [0u32; 10]; ends[1] = blocks; for e in ends { layer.extend(e.to_le_bytes()); } layer.extend(1u32.to_le_bytes()); layer.extend([0u8; 4]);
        let f = fns.iter().find(|f| f.0 == kernel(&q)).ok_or("kernel")?.1;
        g.launch(f, [blocks, 1, 1], 128, 0, &mut [layer.as_mut_ptr().cast()])?;
        token += count;
    }
    Ok(())
}
const MX_WARPS: usize = 12;
fn launch_tile(g: &Gpu, f: H, p: &Proj) -> Result<()> {
    let mut a = proj_bytes(p);
    g.launch(f, [p.rows.div_ceil(MX_WARPS as u32 * 4), p.tokens.div_ceil(16), 1], (MX_WARPS * 32) as u32, 0, &mut [a.as_mut_ptr().cast()])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Fixture { p: Proj }
/// mode 1: row r holds code (r + column) % 256 and scale byte r / 256.
fn build(g: &mut Gpu, rows: usize, cols: usize, tokens: usize, mode: u32, kind: u32, seed: u64, out_words: usize) -> Result<(Fixture, Buf, Buf)> {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..rows * cols).map(|i| if mode == 1 { ((i / cols + i % cols) % 256) as u8 } else { let x = r.next(); let code = (x >> 8) as u8; if code & 127 == 127 && !x.is_multiple_of(4096) { code & 0xf0 | 5 } else { code } }).collect();
    let s: Vec<u8> = (0..rows * cols / 32).map(|i| if mode == 1 { ((i / (cols / 32)) / 256) as u8 } else { let x = r.next(); if x.is_multiple_of(32) { (x >> 8) as u8 } else { 110 + ((x >> 8) % 36) as u8 } }).collect();
    let wb = Buf::new(g, &w)?; let sb = Buf::new(g, &s)?; let xb = Buf::new(g, &le(&activations(kind, tokens * cols, seed + 5)))?;
    let oa = Buf::new(g, &vec![0xff; out_words * 4])?; let ob = Buf::new(g, &vec![0xff; out_words * 4])?;
    let p = Proj { weights: wb.address, scales: sb.address, input: xb.address, output: oa.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
    Ok((Fixture { p }, oa, ob))
}
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXROWS baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let names = ["flash_next_mxfp8_m1", "flash_next_mxfp8_pair", "flash_next_mxfp8_bits"];
    let fns: Vec<(&str, H)> = names.iter().map(|n| g.function(ma, n).map(|h| (*n, h))).collect::<Result<_>>()?;
    let tile_fn = g.function(mb, "mx_rows_tile")?;
    let started = Instant::now(); let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
    let check = |g: &mut Gpu, rows: usize, cols: usize, tokens: usize, mode: u32, kind: u32, seed: u64| -> Result<()> {
        let (fx, oa, ob) = build(g, rows, cols, tokens, mode, kind, seed, rows * tokens)?;
        oa.poison(g)?; ob.poison(g)?;
        launch_reference(g, &fns, &fx.p)?; launch_tile(g, tile_fn, &Proj { output: ob.address, ..fx.p })?; g.sync()?;
        let (x, y) = (oa.read(g)?, ob.read(g)?);
        if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXROWS BYTE FAIL rows={rows} cols={cols} tokens={tokens} mode={mode} act_kind={kind} first_word_and_byte={at:?}")); }
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXROWS guard changed".into()); }
        comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * tokens * 4) as u64); g.clear()
    };
    // All 256 weight codes x all 256 scale bytes (cols 256: every code appears in every row).
    for (kind, tokens) in [(0u32, 3usize), (1, 17), (2, 9), (3, 16), (4, 33)] { check(&mut g, 65536, 256, tokens, 1, kind, 11 + kind as u64)?; }
    println!("MXROWS all 256x256 code pairs ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    let mut seed = 500u64;
    for &rows in &[1usize, 3, 4, 5, 15, 16, 17, 31, 33, 47, 48, 49, 64, 96, 100, 129, 640, 641, 1030] {
        for &cols in &[32usize, 64, 96, 224, 256, 288, 512, 800, 1024, 2048, 4096] {
            for &tokens in &[1usize, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
                if rows * cols > 3_000_000 && tokens > 17 || rows * cols * tokens > 40_000_000 { continue; }
                seed += 1; check(&mut g, rows, cols, tokens, 0, (seed % 5) as u32, seed)?;
            }
        }
    }
    println!("MXROWS sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Grok's finding (2026-11-06): every admitted value of each axis (swept.rs), two-axis grids at a small third, the tile edges, the eight corners.
    for &rows in &swept::mx_rows_axis() { seed += 1; check(&mut g, rows, 32, 17, 0, (seed % 5) as u32, seed)?; }
    for &cols in &swept::mx_cols_axis() { seed += 1; check(&mut g, 129, cols, 17, 0, (seed % 5) as u32, seed)?; seed += 1; check(&mut g, 1, cols, 1, 0, (seed % 5) as u32, seed)?; }
    for &tokens in &swept::mx_tokens_axis() { seed += 1; check(&mut g, 49, 256, tokens, 0, (seed % 5) as u32, seed)?; seed += 1; check(&mut g, 8192, 32, tokens, 0, (seed % 5) as u32, seed)?; }
    println!("MXROWS axes ok (rows 1..=8192, cols 32..=8192, tokens 1..=256, every value) comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Every 48-row block edge and one either side (rows), against the token tile edges, at two column counts.
    let mut edges: Vec<usize> = vec![]; for k in 1..=(8192 / 48 + 1) { for d in [47usize, 48, 49] { let r = (k - 1) * 48 + d; if (1..=8192).contains(&r) { edges.push(r); } } } edges.push(8192); edges.sort(); edges.dedup();
    for &rows in &edges { for &tokens in &[1usize, 16, 17, 255, 256] { for &cols in &[32usize, 288] { seed += 1; check(&mut g, rows, cols, tokens, 0, (seed % 5) as u32, seed)?; } } }
    for (rows, cols, tokens) in swept::mx_corners() { for kind in [0u32, 4] { seed += 1; check(&mut g, rows, cols, tokens, 0, kind, seed)?; } }
    println!("MXROWS edges and corners ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    for (rows, cols, tokens, kind) in [(4096usize, 4096usize, 256usize, 0u32), (2048, 8192, 256, 0), (6144, 4096, 64, 1), (4096, 4096, 17, 2), (4096, 4096, 256, 4), (2048, 2048, 1, 0),
        // Grok's review of READ-SPLIT: the admitted edges (rows 8192, columns 32 to 8192, one token) that no sweep reached.
        (8192, 256, 1, 0), (8192, 32, 256, 0), (8192, 8192, 16, 1), (8192, 256, 256, 2), (7001, 288, 17, 3), (49, 8192, 256, 0)] { check(&mut g, rows, cols, tokens, 0, kind, 900 + rows as u64 + kind as u64)?; }
    println!("MXROWS BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
    // Timing at the model's shapes, one group of 256 tokens (16 production launches against one).
    for (rows, cols, tokens) in [(4096usize, 4096usize, 256usize), (2048, 8192, 256), (6144, 4096, 256), (4096, 4096, 16)] {
        let (fx, oa, _ob) = build(&mut g, rows, cols, tokens, 0, 0, 31, rows * tokens)?; let _ = &oa;
        let (mut ta, mut tb) = (vec![], vec![]);
        for repeat in 0..7 {
            for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                if arm == 0 { launch_reference(&g, &fns, &fx.p)?; } else { launch_tile(&g, tile_fn, &fx.p)?; } g.sync()?;
                let n = 5; let t0 = Instant::now();
                for _ in 0..n { if arm == 0 { launch_reference(&g, &fns, &fx.p)?; } else { launch_tile(&g, tile_fn, &fx.p)?; } }
                g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let (ra, rb) = (median(ta), median(tb)); let macs = (rows * cols * tokens) as f64;
        println!("MXROWS TIME rows={rows} cols={cols} tokens={tokens} reference_us={ra:.0} tile_us={rb:.0} ratio={:.2} reference_us_per_token_row={:.2} tile_us_per_token_row={:.2} reference_GMACs={:.0} tile_GMACs={:.0}", ra / rb, ra / tokens as f64, rb / tokens as f64, macs / ra / 1e3, macs / rb / 1e3);
        g.clear()?;
    }
    println!("MXROWS PASS");
    Ok(())
}
