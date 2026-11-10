//! Same-bytes sweep and timing for `mx_rows_small2/4` (2 to 4 tokens) against the production MXFP8 prompt-row path
//! (`apply_rows`: groups of 16 tokens, kernel chosen by token count) and `mx_rows_tile`, in one process on one device.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, sha256};
use std::time::Instant;
use super::envelope as e;
const GUARD: [u8; 16] = [0x5a; 16];
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
fn launch_tile(g: &Gpu, f: H, p: &Proj) -> Result<()> {
    let mut a = proj_bytes(p);
    g.launch(f, [p.rows.div_ceil(48), p.tokens.div_ceil(16), 1], 384, 0, &mut [a.as_mut_ptr().cast()])
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
/// Shared weights and scales for every token count and activation kind of one shape (random codes, plain scale bytes).
fn weights_of(g: &mut Gpu, rows: usize, cols: usize, seed: u64) -> Result<(u64, u64)> {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..rows * cols).map(|_| { let x = r.next(); let code = (x >> 8) as u8; if code & 127 == 127 && !x.is_multiple_of(4096) { code & 0xf0 | 5 } else { code } }).collect();
    let sc: Vec<u8> = (0..rows * cols / 32).map(|_| { let x = r.next(); if x.is_multiple_of(32) { (x >> 8) as u8 } else { 110 + ((x >> 8) % 36) as u8 } }).collect();
    Ok((Buf::new(g, &w)?.address, Buf::new(g, &sc)?.address))
}
fn build_shared(g: &mut Gpu, rows: usize, cols: usize, tokens: usize, kind: u32, seed: u64, wa: (u64, u64)) -> Result<(Fixture, Buf, Buf)> {
    let xb = Buf::new(g, &le(&activations(kind, tokens * cols, seed + 5)))?;
    let oa = Buf::new(g, &vec![0xff; rows * tokens * 4])?; let ob = Buf::new(g, &vec![0xff; rows * tokens * 4])?;
    let p = Proj { weights: wa.0, scales: wa.1, input: xb.address, output: oa.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
    Ok((Fixture { p }, oa, ob))
}
/// `mx_rows_small*`: entry by token count, grid (ceil(rows/16), ceil(tokens/T)).
fn small_entry(tokens: u32) -> (&'static str, u32) { match tokens { 2 => ("mx_rows_small2", 2), _ => ("mx_rows_small4", 4) } }
fn launch_small(g: &Gpu, fs: &[(&str, H)], p: &Proj) -> Result<()> {
    let (name, per) = small_entry(p.tokens); let f = fs.iter().find(|f| f.0 == name).ok_or("small entry")?.1;
    let mut a = proj_bytes(p);
    g.launch(f, [p.rows.div_ceil(16), p.tokens.div_ceil(per), 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
pub(crate) fn run(base_path: &str, candidate_path: &str, time: bool) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXSMALL baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let names = ["flash_next_mxfp8_m1", "flash_next_mxfp8_pair", "flash_next_mxfp8_bits"];
    let fns: Vec<(&str, H)> = names.iter().map(|n| g.function(ma, n).map(|h| (*n, h))).collect::<Result<_>>()?;
    let tile_fn = g.function(ma, "mx_rows_tile")?;
    let smalls: Vec<(&str, H)> = ["mx_rows_small2", "mx_rows_small4"].iter().map(|n| g.function(mb, n).map(|h| (*n, h))).collect::<Result<_>>()?;
    let started = Instant::now();
    if !time {
        let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
        let check = |g: &mut Gpu, rows: usize, cols: usize, tokens: usize, mode: u32, kind: u32, seed: u64| -> Result<()> {
            let (fx, oa, ob) = build(g, rows, cols, tokens, mode, kind, seed, rows * tokens)?;
            oa.poison(g)?; ob.poison(g)?;
            launch_reference(g, &fns, &fx.p)?; launch_small(g, &smalls, &Proj { output: ob.address, ..fx.p })?; g.sync()?;
            let (x, y) = (oa.read(g)?, ob.read(g)?);
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXSMALL BYTE FAIL rows={rows} cols={cols} tokens={tokens} mode={mode} act_kind={kind} first_word_and_byte={at:?}")); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXSMALL guard changed".into()); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * tokens * 4) as u64); g.clear()
        };
        // All 256 weight codes x all 256 scale bytes (cols 256: every code appears in every row).
        for (kind, tokens) in [(0u32, 2usize), (1, 3), (2, 4), (3, 2), (4, 4)] { check(&mut g, 65536, 256, tokens, 1, kind, 11 + kind as u64)?; }
        println!("MXSMALL all 256x256 code pairs ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Every listed (rows, cols) of `small_sweep` (weights at most MX_MAX_BYTES) at every token count 2 to 4 and five activation kinds.
        let mut seed = 500u64;
        for &rows in e::MX_ROWS { for &cols in e::MX_COLS {
            if rows * cols > e::MX_MAX_BYTES { continue; }
            let wa = weights_of(&mut g, rows, cols, 7 + rows as u64 * 131 + cols as u64)?;
            for tokens in e::MX_MIN_TOKENS..=e::MX_MAX_TOKENS { for kind in 0..5u32 {
                seed += 1;
                let (fx, oa, ob) = build_shared(&mut g, rows, cols, tokens, kind, seed, wa)?;
                oa.poison(&g)?; ob.poison(&g)?;
                launch_reference(&g, &fns, &fx.p)?; launch_small(&g, &smalls, &Proj { output: ob.address, ..fx.p })?; g.sync()?;
                let (x, y) = (oa.read(&g)?, ob.read(&g)?);
                if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXSMALL BYTE FAIL rows={rows} cols={cols} tokens={tokens} act_kind={kind} first_word_and_byte={at:?}")); }
                if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXSMALL guard changed".into()); }
                comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * tokens * 4) as u64);
            } }
            g.clear()?;
        } }
        println!("MXSMALL sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        for (rows, cols) in [(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] { for tokens in 2..=4usize { for kind in 0..5u32 { check(&mut g, rows, cols, tokens, 0, kind, 900 + rows as u64 + kind as u64 + tokens as u64)?; } } }
        println!("MXSMALL BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    // Rank-local MXFP8 shapes of the model and their calls in a pass: q, k, v 34 layers each (4096 x 4096), o 34 (2048 x 8192), dense gate and up 3 each (6144 x 4096), down 3 (2048 x 12288).
    let shapes: [(usize, usize, f64, &str); 4] = [(4096, 4096, 102.0, "qkv"), (2048, 8192, 34.0, "o"), (6144, 4096, 6.0, "gate_up"), (2048, 12288, 3.0, "down")];
    println!("MXSMALL TIME shape tokens | production_us mx_rows_tile_us mx_rows_small_us | GB/s | floor_us_at_223GBs");
    let mut totals: Vec<(usize, f64, f64, f64, f64)> = vec![];
    for &(rows, cols, calls, label) in &shapes {
        let bytes = (rows * cols) as f64 * (1.0 + 1.0 / 32.0);
        for &tokens in &[2usize, 3, 4] {
            let k = ((150e6 / bytes).ceil() as usize).clamp(2, 32);
            let mut fxs = vec![]; for i in 0..k { fxs.push(build(&mut g, rows, cols, tokens, 0, 0, 31 + i as u64, rows * tokens)?.0); }
            let mut t = [vec![], vec![], vec![]];
            for repeat in 0..5 { for arm in (0..3).map(|a| (a + repeat) % 3) {
                let run = |g: &Gpu, i: usize| -> Result<()> { match arm { 0 => launch_reference(g, &fns, &fxs[i % k].p), 1 => launch_tile(g, tile_fn, &fxs[i % k].p), _ => launch_small(g, &smalls, &fxs[i % k].p) } };
                for i in 0..k { run(&g, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..k { run(&g, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / k as f64);
            } }
            let [ra, rb, rc] = t.map(median);
            println!("MXSMALL TIME {label} rows={rows} cols={cols} tokens={tokens} production_us={ra:.0} mx_rows_tile_us={rb:.0} mx_rows_small_us={rc:.0} GBs={:.0}/{:.0}/{:.0} floor_us_at_223GBs={:.0}", bytes / ra / 1e3, bytes / rb / 1e3, bytes / rc / 1e3, bytes / 223e3);
            totals.push((tokens, ra * calls / 1e3, rb * calls / 1e3, rc * calls / 1e3, bytes / 223e3 * calls / 1e3));
            g.clear()?;
        }
    }
    for &tokens in &[2usize, 3, 4] {
        let (mut ra, mut rb, mut rc, mut f) = (0.0, 0.0, 0.0, 0.0);
        for t in totals.iter().filter(|t| t.0 == tokens) { ra += t.1; rb += t.2; rc += t.3; f += t.4; }
        println!("MXSMALL PASS tokens={tokens} ms_a_pass: production={ra:.1} mx_rows_tile={rb:.1} mx_rows_small={rc:.1} floor={f:.1}");
    }
    println!("MXSMALL TIME DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
