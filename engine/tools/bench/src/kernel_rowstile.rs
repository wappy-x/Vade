//! Same-bytes sweep and timing for the tiled plain token rows (`plain_rows_tile`) against lane R's
//! `prompt_rows` ProjectRows fast path, in one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
#[allow(dead_code)] #[path = "../../../crates/gpu/src/swept.rs"] mod swept;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op ProjectRows is 1, inner Project is 5.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// 0 plain, 1 wide range, 2 denormals and signed zeros, 3 large finite, 4 overflow with NaN and inf, 5 huge/tiny alternation.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.next();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(60),
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i % 4099 == 7 { f32::NAN } else if i % 4099 == 11 { f32::INFINITY } else { v } }
            _ => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
        }
    }).collect()
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// 16-byte guards on both sides; the data address stays 16-byte aligned (allocations are 256-byte aligned).
struct Buffer { base: u64, address: u64, bytes: usize }
impl Buffer {
    fn new(g: &mut Gpu, values: &[f32]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), &le(values), GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: values.len() * 4 })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
fn args(input: &Buffer, weight: &Buffer, output: &Buffer, rows: usize, tokens: usize, cols: usize) -> Arguments {
    let inner = Inner { op: 5, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    Arguments { input: [input.address, 0, 0, 0, 0], lengths: [(tokens * cols) as u64, 0, 0, 0, 0], weight: weight.address, weights: 0, output: output.address,
        units: (rows * tokens) as u64, words: 1, shape: PShape { op: 1, tokens: tokens as u32, heads: 0, dim: 0, taps: 0, rows: rows as u32, cols: cols as u32, first: 0, slots: 0, inner } }
}
fn launch_reference(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }
fn launch_tile(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> {
    let blocks = (a.shape.tokens as usize).div_ceil(64) * (a.shape.rows as usize).div_ceil(64);
    g.launch(f, [blocks as u32, 1, 1], 256, 0, &mut [ptr(a)])
}
fn launch_narrow(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> {
    let blocks = (a.shape.tokens as usize).div_ceil(32) * (a.shape.rows as usize).div_ceil(16);
    g.launch(f, [blocks as u32, 1, 1], 128, 0, &mut [ptr(a)])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
pub(crate) fn run(base_path: &str, candidate_path: &str, narrow_only: bool) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("ROWSTILE baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "prompt_rows")?; let tile = g.function(mb, "plain_rows_tile")?;
    if narrow_only { let narrow = g.function(mb, "plain_rows_narrow")?; narrow_screen(&mut g, reference, tile, narrow)?; println!("ROWSTILE PASS (narrow only)"); return Ok(()); }
    let started = Instant::now(); let (mut comparisons, mut bytes) = (0u64, 0u64);
    let cols_list = [4usize, 8, 12, 28, 32, 36, 60, 64, 68, 100, 128, 132, 252, 256, 508, 512, 516, 1024, 2052, 4096];
    let rows_list = [1usize, 2, 3, 16, 17, 63, 64, 65, 129, 256, 1000];
    let tokens_list = [1usize, 2, 3, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300];
    let (rows_max, tokens_max) = (1000usize, 300usize);
    for &cols in &cols_list {
        let kinds: &[(u32, u32)] = if cols <= 516 { &[(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3), (0, 4)] } else { &[(5, 5), (2, 2), (4, 0)] };
        for &(ik, wk) in kinds {
            let weight = Buffer::new(&mut g, &pattern(wk, rows_max * cols, 100 + cols as u64 + wk as u64))?;
            let input = Buffer::new(&mut g, &pattern(ik, tokens_max * cols, 200 + cols as u64 * 3 + ik as u64))?;
            let out_a = Buffer::new(&mut g, &vec![f32::NAN; rows_max * tokens_max])?; let out_b = Buffer::new(&mut g, &vec![f32::NAN; rows_max * tokens_max])?;
            for &rows in &rows_list { for &tokens in &tokens_list {
                out_a.poison(&g)?; out_b.poison(&g)?;
                let (mut xa, mut xb) = (args(&input, &weight, &out_a, rows, tokens, cols), args(&input, &weight, &out_b, rows, tokens, cols));
                launch_reference(&g, reference, &mut xa)?; launch_tile(&g, tile, &mut xb)?; g.sync()?;
                let (x, y) = (out_a.read(&g)?, out_b.read(&g)?);
                if x != y {
                    let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i));
                    return Err(format!("ROWSTILE BYTE FAIL rows={rows} tokens={tokens} cols={cols} input_pattern={ik} weight_pattern={wk} first_word_and_byte={at:?}"));
                }
                if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSTILE guard changed".into()); }
                comparisons += 1; bytes += (rows * tokens * 4) as u64;
            } }
        }
        println!("ROWSTILE cols={cols} ok comparisons_so_far={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        g.clear()?;
    }
    // Random shapes: tile edges in every direction. Fixed seed, so the list is reproducible.
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for (ik, wk) in [(0u32, 0u32), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3)] {
        let (rmax, tmax, cmax) = (2100usize, 640usize, 1024usize);
        let weight = Buffer::new(&mut g, &pattern(wk, rmax * cmax, 900 + wk as u64))?; let input = Buffer::new(&mut g, &pattern(ik, tmax * cmax, 901 + ik as u64))?;
        let out_a = Buffer::new(&mut g, &vec![f32::NAN; rmax * tmax])?; let out_b = Buffer::new(&mut g, &vec![f32::NAN; rmax * tmax])?;
        for _ in 0..400 {
            let (rows, tokens, cols) = (1 + (rng.next() % rmax as u64) as usize, 1 + (rng.next() % tmax as u64) as usize, 4 * (1 + (rng.next() % (cmax as u64 / 4)) as usize));
            out_a.poison(&g)?; out_b.poison(&g)?;
            let (mut xa, mut xb) = (args(&input, &weight, &out_a, rows, tokens, cols), args(&input, &weight, &out_b, rows, tokens, cols));
            launch_reference(&g, reference, &mut xa)?; launch_tile(&g, tile, &mut xb)?; g.sync()?;
            let (x, y) = (out_a.read(&g)?, out_b.read(&g)?);
            if x != y { return Err(format!("ROWSTILE BYTE FAIL random rows={rows} tokens={tokens} cols={cols} patterns={ik},{wk}")); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSTILE guard changed".into()); }
            comparisons += 1; bytes += (rows * tokens * 4) as u64;
        }
        println!("ROWSTILE random patterns={ik},{wk} 400 shapes ok comparisons_so_far={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        g.clear()?;
    }
    // Real shapes: 16384 rows x 512 columns, several token counts, both pattern families.
    for (tokens, rows, cols, ik, wk) in [(256usize, 16384usize, 512usize, 0u32, 0u32), (256, 16384, 512, 5, 5), (256, 16384, 512, 4, 4), (64, 16384, 512, 5, 5), (1, 16384, 512, 5, 5),
        (130, 8192, 512, 2, 2), (300, 16384, 576, 5, 5), (256, 2048, 4096, 5, 5), (257, 1024, 2052, 1, 1), (512, 16384, 512, 5, 5), (1024, 16384, 512, 5, 5), (1024, 2048, 4096, 5, 5), (300, 16384, 4096, 4, 4), (1024, 1000, 36, 2, 2)] {
        let weight = Buffer::new(&mut g, &pattern(wk, rows * cols, 7 + cols as u64))?; let input = Buffer::new(&mut g, &pattern(ik, tokens * cols, 11))?;
        let out_a = Buffer::new(&mut g, &vec![f32::NAN; rows * tokens])?; let out_b = Buffer::new(&mut g, &vec![f32::NAN; rows * tokens])?;
        out_a.poison(&g)?; out_b.poison(&g)?;
        let (mut xa, mut xb) = (args(&input, &weight, &out_a, rows, tokens, cols), args(&input, &weight, &out_b, rows, tokens, cols));
        launch_reference(&g, reference, &mut xa)?; launch_tile(&g, tile, &mut xb)?; g.sync()?;
        let (x, y) = (out_a.read(&g)?, out_b.read(&g)?);
        if x != y { return Err(format!("ROWSTILE BYTE FAIL real shape rows={rows} tokens={tokens} cols={cols} patterns={ik},{wk}")); }
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSTILE guard changed".into()); }
        comparisons += 1; bytes += (rows * tokens * 4) as u64;
        println!("ROWSTILE real rows={rows} tokens={tokens} cols={cols} patterns={ik},{wk} equal");
        g.clear()?;
    }
    println!("ROWSTILE BYTES PASS comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    // Timing at the real weight shape. Two weight copies alternate so each call streams weights from memory.
    let (rows, cols) = (16384usize, 512usize);
    for tokens in [256usize, 128, 64, 32, 16, 8, 1, 512] {
        let w0 = Buffer::new(&mut g, &pattern(0, rows * cols, 1))?; let w1 = Buffer::new(&mut g, &pattern(0, rows * cols, 2))?;
        let input = Buffer::new(&mut g, &pattern(0, tokens * cols, 3))?; let out = Buffer::new(&mut g, &vec![0.0; rows * tokens])?;
        let (mut ta, mut tb) = (vec![], vec![]);
        for repeat in 0..7 {
            for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                let weights = [&w0, &w1];
                let mut warm = args(&input, weights[0], &out, rows, tokens, cols);
                if arm == 0 { launch_reference(&g, reference, &mut warm)?; } else { launch_tile(&g, tile, &mut warm)?; } g.sync()?;
                let n = 20; let t0 = Instant::now();
                for i in 0..n {
                    let mut x = args(&input, weights[i % 2], &out, rows, tokens, cols);
                    if arm == 0 { launch_reference(&g, reference, &mut x)?; } else { launch_tile(&g, tile, &mut x)?; }
                }
                g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let (ra, rb) = (median(ta), median(tb)); let macs = (rows * cols * tokens) as f64;
        println!("ROWSTILE TIME rows={rows} cols={cols} tokens={tokens} reference_us_per_call={ra:.1} tile_us_per_call={rb:.1} ratio={:.2} reference_us_per_token_row={:.2} tile_us_per_token_row={:.2} reference_GMACs={:.0} tile_GMACs={:.0}",
            ra / rb, ra / tokens as f64, rb / tokens as f64, macs / ra / 1e3, macs / rb / 1e3);
        g.clear()?;
    }
    let narrow = g.function(mb, "plain_rows_narrow")?; narrow_screen(&mut g, reference, tile, narrow)?;
    println!("ROWSTILE PASS");
    Ok(())
}

/// `plain_rows_narrow` (16 rows x 32 tokens, 128 threads) against `prompt_rows`, and against the 64 x 64 tile for time.
/// The sweep walks the swept lists of `plain_tile::narrow_admitted` (rows 1 to 192, tokens 1 to 512, columns 4 to 4096).
fn narrow_screen(g: &mut Gpu, reference: H, tile: H, narrow: H) -> Result<()> {
    let started = Instant::now(); let (mut comparisons, mut bytes) = (0u64, 0u64);
    let cols_list = [4usize, 8, 12, 28, 32, 36, 60, 64, 68, 100, 128, 132, 252, 256, 508, 512, 516, 1024, 2052, 4096];
    let rows_list = [1usize, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 191, 192];
    let tokens_list = [1usize, 2, 3, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300, 511, 512];
    for &cols in &cols_list {
        let kinds: &[(u32, u32)] = if cols <= 516 { &[(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3), (0, 4)] } else { &[(5, 5), (2, 2), (4, 0)] };
        for &(ik, wk) in kinds {
            let weight = Buffer::new(g, &pattern(wk, 192 * cols, 300 + cols as u64 + wk as u64))?;
            let input = Buffer::new(g, &pattern(ik, 512 * cols, 400 + cols as u64 * 3 + ik as u64))?;
            let out_a = Buffer::new(g, &vec![f32::NAN; 192 * 512])?; let out_b = Buffer::new(g, &vec![f32::NAN; 192 * 512])?;
            for &rows in &rows_list { for &tokens in &tokens_list {
                out_a.poison(g)?; out_b.poison(g)?;
                let (mut xa, mut xb) = (args(&input, &weight, &out_a, rows, tokens, cols), args(&input, &weight, &out_b, rows, tokens, cols));
                launch_reference(g, reference, &mut xa)?; launch_narrow(g, narrow, &mut xb)?; g.sync()?;
                let (x, y) = (out_a.read(g)?, out_b.read(g)?);
                if x != y { return Err(format!("ROWSTILE NARROW BYTE FAIL rows={rows} tokens={tokens} cols={cols} patterns={ik},{wk}")); }
                if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSTILE NARROW guard changed".into()); }
                comparisons += 1; bytes += (rows * tokens * 4) as u64;
            } }
        }
        g.clear()?;
    }
    let mut rng = Rng(0x51ed270b0f1a3c49);
    for (ik, wk) in [(0u32, 0u32), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3)] {
        let (rmax, tmax, cmax) = (192usize, 512usize, 1024usize);
        let weight = Buffer::new(g, &pattern(wk, rmax * cmax, 950 + wk as u64))?; let input = Buffer::new(g, &pattern(ik, tmax * cmax, 951 + ik as u64))?;
        let out_a = Buffer::new(g, &vec![f32::NAN; rmax * tmax])?; let out_b = Buffer::new(g, &vec![f32::NAN; rmax * tmax])?;
        for _ in 0..300 {
            let (rows, tokens, cols) = (1 + (rng.next() % rmax as u64) as usize, 1 + (rng.next() % tmax as u64) as usize, 4 * (1 + (rng.next() % (cmax as u64 / 4)) as usize));
            out_a.poison(g)?; out_b.poison(g)?;
            let (mut xa, mut xb) = (args(&input, &weight, &out_a, rows, tokens, cols), args(&input, &weight, &out_b, rows, tokens, cols));
            launch_reference(g, reference, &mut xa)?; launch_narrow(g, narrow, &mut xb)?; g.sync()?;
            let (x, y) = (out_a.read(g)?, out_b.read(g)?);
            if x != y { return Err(format!("ROWSTILE NARROW BYTE FAIL random rows={rows} tokens={tokens} cols={cols} patterns={ik},{wk}")); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSTILE NARROW guard changed".into()); }
            comparisons += 1; bytes += (rows * tokens * 4) as u64;
        }
        g.clear()?;
    }
    // Astra's finding (2026-11-06): every admitted value of each axis (swept.rs), grids of two axes at a small third, the eight corners.
    {
        let mut check = |g: &mut Gpu, weight: &Buffer, input: &Buffer, out_a: &Buffer, out_b: &Buffer, rows: usize, tokens: usize, cols: usize, tag: u32| -> Result<()> {
            out_a.poison(g)?; out_b.poison(g)?;
            let (mut xa, mut xb) = (args(input, weight, out_a, rows, tokens, cols), args(input, weight, out_b, rows, tokens, cols));
            launch_reference(g, reference, &mut xa)?; launch_narrow(g, narrow, &mut xb)?; g.sync()?;
            let (x, y) = (out_a.read(g)?, out_b.read(g)?);
            if x != y { return Err(format!("ROWSTILE NARROW BYTE FAIL axis rows={rows} tokens={tokens} cols={cols} tag={tag}")); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSTILE NARROW guard changed".into()); }
            comparisons += 1; bytes += (rows * tokens * 4) as u64; Ok(())
        };
        // rows x tokens, every pair, at columns 4 and 36 (one 16-byte quad; one full chunk plus a quad).
        for (cols, wk, ik) in [(4usize, 5u32, 5u32), (36, 2, 0)] {
            let weight = Buffer::new(g, &pattern(wk, 192 * cols, 600 + cols as u64))?; let input = Buffer::new(g, &pattern(ik, 512 * cols, 601 + cols as u64))?;
            let out_a = Buffer::new(g, &vec![f32::NAN; 192 * 512])?; let out_b = Buffer::new(g, &vec![f32::NAN; 192 * 512])?;
            for &rows in &swept::narrow_rows_axis() { for &tokens in &swept::narrow_tokens_axis() { check(g, &weight, &input, &out_a, &out_b, rows, tokens, cols, 0)?; } }
            g.clear()?;
        }
        // every column count, at eight shapes (corners of rows x tokens and tile edges).
        let cols_max = swept::NARROW_COLS_MAX;
        let weight = Buffer::new(g, &pattern(0, 192 * cols_max, 700))?; let input = Buffer::new(g, &pattern(0, 512 * cols_max, 701))?;
        let weight5 = Buffer::new(g, &pattern(5, 192 * cols_max, 702))?; let input5 = Buffer::new(g, &pattern(5, 512 * cols_max, 703))?;
        let out_a = Buffer::new(g, &vec![f32::NAN; 192 * 512])?; let out_b = Buffer::new(g, &vec![f32::NAN; 192 * 512])?;
        for &(rows, tokens) in &[(1usize, 1usize), (17, 33), (64, 256), (192, 512), (192, 1), (1, 512), (65, 257), (129, 300)] {
            for &cols in &swept::narrow_cols_axis() { let (w, i) = if cols % 8 == 0 { (&weight, &input) } else { (&weight5, &input5) }; check(g, w, i, &out_a, &out_b, rows, tokens, cols, 1)?; }
        }
        for (rows, cols, tokens) in swept::narrow_corners() { check(g, &weight, &input, &out_a, &out_b, rows, tokens, cols, 2)?; check(g, &weight5, &input5, &out_a, &out_b, rows, tokens, cols, 3)?; }
        g.clear()?;
    }
    println!("ROWSTILE NARROW BYTES PASS comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    for (rows, tokens, cols) in [(64usize, 256usize, 4096usize), (64, 256, 2048), (64, 256, 1024), (64, 256, 512), (192, 256, 4096), (128, 256, 4096), (192, 256, 2048), (64, 128, 4096), (64, 512, 4096), (16, 256, 4096)] {
        let weight = Buffer::new(g, &pattern(0, rows * cols, 1))?; let input = Buffer::new(g, &pattern(0, tokens * cols, 3))?; let out = Buffer::new(g, &vec![0.0; rows * tokens])?;
        let (mut ta, mut tb) = (vec![], vec![]);
        for repeat in 0..7 {
            for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                let mut warm = args(&input, &weight, &out, rows, tokens, cols);
                if arm == 0 { launch_tile(g, tile, &mut warm)?; } else { launch_narrow(g, narrow, &mut warm)?; } g.sync()?;
                let n = 50; let t0 = Instant::now();
                for _ in 0..n { let mut x = args(&input, &weight, &out, rows, tokens, cols); if arm == 0 { launch_tile(g, tile, &mut x)?; } else { launch_narrow(g, narrow, &mut x)?; } }
                g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let (ra, rb) = (median(ta), median(tb)); let macs = (rows * cols * tokens) as f64;
        println!("ROWSTILE NARROW TIME rows={rows} cols={cols} tokens={tokens} tile64_us={ra:.1} narrow_us={rb:.1} ratio={:.2} tile64_GMACs={:.0} narrow_GMACs={:.0}", ra / rb, macs / ra / 1e3, macs / rb / 1e3);
        g.clear()?;
    }
    Ok(())
}
