//! Lane P device screen for the prompt-group (reading) rows: same-bytes sweep and timing of the candidate image against the baseline
//! image, in one process on one device. The baseline arm is the generic `prompt_rows` kernel on a zero-filled output (what the host
//! did); the candidate arm is the new kernel on a poisoned output (the host skips the fill). Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
#[allow(dead_code, clippy::duplicate_mod)] #[path = "../../../crates/gpu/src/swept.rs"] mod swept;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
}
/// Random 32-bit patterns: every NaN payload, denormals, infinities and signed zeros occur; moves must keep all bits.
fn bits(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| match r.next() % 16 { 0 => f32::from_bits(0x7fc0_0000 | (r.next() as u32 & 0x3fffff)), 1 => -0.0, 2 => f32::INFINITY, 3 => f32::from_bits(r.next() as u32 & 0x7fffff), _ => f32::from_bits((r.next() >> 16) as u32 ^ i as u32) }).collect()
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
struct Buffer { base: u64, address: u64, bytes: usize }
impl Buffer {
    fn new(g: &mut Gpu, values: &[f32]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), &le(values), GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: values.len() * 4 })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
    fn zero(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0u8; self.bytes]) }
}
/// Mirror of gpu/src/causal_ops.rs `Shape` (inner) and gpu/src/prompt_ops.rs `Shape` / `Arguments` (repr(C)). Op numbers follow the enum order.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct PArgs { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
const PERMUTE: u32 = 13; const PLACE: u32 = 14; const PLACE_TILE: u32 = 16; // ScoresMap (15) was added after ExpertPlace
fn empty() -> Inner { Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 } }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn args(shape: PShape, input: [u64; 5], lengths: [usize; 5], out: u64, units: usize, words: usize) -> PArgs {
    PArgs { input, lengths: lengths.map(|n| n as u64), weight: 4, weights: 0, output: out, units: units as u64, words: words as u64, shape }
}
fn base_launch(g: &Gpu, f: H, a: &mut PArgs) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }

/// A permute case: `rows` output rows of `cols` words from `source` rows through a table.
struct Permute { rows: usize, cols: usize, source: usize }
struct Place { tokens: usize, k: usize, experts: usize, d: usize, parts: usize }
/// The expert grouping of rows.rs: rows `token*k+slot` with distinct experts per token, grouped by ascending expert id.
/// Returns (places `[rows,3]`, rows).
fn grouping(c: &Place, seed: u64) -> (Vec<f32>, usize) {
    let mut r = Rng(seed | 1); let rows = c.tokens * c.k;
    let mut groups: Vec<Vec<usize>> = vec![vec![]; c.experts];
    for t in 0..c.tokens { let mut chosen: Vec<usize> = vec![]; while chosen.len() < c.k { let e = (r.next() % c.experts as u64) as usize; if !chosen.contains(&e) { chosen.push(e); } }
        for (slot, &e) in chosen.iter().enumerate() { groups[e].push(t * c.k + slot); } }
    let mut places = vec![0.0f32; 3 * rows]; let mut at = 0usize;
    for g in &groups { for (position, &row) in g.iter().enumerate() { places[3 * row] = at as f32; places[3 * row + 1] = g.len() as f32; places[3 * row + 2] = position as f32; } at += g.len(); }
    (places, rows)
}
/// Mirror of prompt_ops::place_table for a valid grouping.
fn place_table(places: &[f32]) -> Vec<f32> {
    let rows = places.len() / 3; let mut inverse = vec![0.0f32; rows];
    for row in 0..rows { inverse[(places[3 * row] + places[3 * row + 2]) as usize] = row as f32; }
    [places.to_vec(), inverse].concat()
}
fn permute_case(g: &mut Gpu, base: H, cand: H, c: &Permute, seed: u64, grids: &[u32]) -> Result<u64> {
    let mut r = Rng(seed | 1);
    let table: Vec<f32> = (0..c.rows).map(|_| (r.next() % c.source as u64) as f32).collect();
    let source = Buffer::new(g, &bits(c.source * c.cols, seed))?; let table_b = Buffer::new(g, &table)?;
    let words = c.rows * c.cols;
    let shape = PShape { op: PERMUTE, tokens: 1, heads: 0, dim: 0, taps: 0, rows: c.rows as u32, cols: c.cols as u32, first: c.source as u32, slots: 0, inner: empty() };
    let a = args(shape, [source.address, table_b.address, 4, 4, 4], [c.source * c.cols, c.rows, 0, 0, 0], 0, words, 1);
    let out_a = Buffer::new(g, &vec![f32::NAN; words])?; out_a.zero(g)?;
    let mut x = PArgs { output: out_a.address, ..a }; base_launch(g, base, &mut x)?; g.sync()?; let want = out_a.read(g)?;
    let mut n = 0;
    for &grid in grids {
        let out_b = Buffer::new(g, &vec![f32::NAN; words])?; out_b.poison(g)?;
        let mut y = PArgs { output: out_b.address, ..a };
        g.launch(cand, [grid.min(c.rows as u32).max(1), 1, 1], 256, 0, &mut [ptr(&mut y)])?; g.sync()?;
        let got = out_b.read(g)?;
        if got != want { return Err(format!("READP PERMUTE BYTE FAIL rows={} cols={} source={} grid={grid} first_diff_word={:?}", c.rows, c.cols, c.source, want.iter().zip(&got).position(|(p, q)| p != q).map(|i| i.saturating_sub(16) / 4))); }
        if got[..16] != GUARD || got[got.len() - 16..] != GUARD { return Err(format!("READP PERMUTE guard changed rows={} cols={}", c.rows, c.cols)); }
        n += 1;
    }
    Ok(n)
}
fn place_case(g: &mut Gpu, base: H, cand: H, c: &Place, seed: u64, grids: &[(u32, u32)]) -> Result<u64> {
    let (places, rows) = grouping(c, seed); let cols = c.d * c.parts; let slots = c.d * rows;
    let exchange = Buffer::new(g, &bits(slots * c.parts, seed ^ 0x77))?;
    let old_table = Buffer::new(g, &places)?; let tile_table = Buffer::new(g, &place_table(&places))?;
    let words = rows * cols;
    let shape = |op| PShape { op, tokens: 1, heads: 0, dim: c.d as u32, taps: c.parts as u32, rows: rows as u32, cols: cols as u32, first: 0, slots: slots as u32, inner: empty() };
    let out_a = Buffer::new(g, &vec![f32::NAN; words])?; out_a.zero(g)?;
    let mut x = args(shape(PLACE), [exchange.address, old_table.address, 4, 4, 4], [slots * c.parts, 3 * rows, 0, 0, 0], out_a.address, words, 1);
    base_launch(g, base, &mut x)?; g.sync()?; let want = out_a.read(g)?;
    let mut n = 0;
    for &(gx, gy) in grids {
        let out_b = Buffer::new(g, &vec![f32::NAN; words])?; out_b.poison(g)?;
        let mut y = args(shape(PLACE_TILE), [exchange.address, tile_table.address, 4, 4, 4], [slots * c.parts, 4 * rows, 0, 0, 0], out_b.address, words, 1);
        let (fx, fy) = (rows.div_ceil(32) as u32, (cols / 32) as u32);
        g.launch(cand, [gx.min(fx).max(1), gy.min(fy).max(1), 1], 256, 0, &mut [ptr(&mut y)])?; g.sync()?;
        let got = out_b.read(g)?;
        if got != want { return Err(format!("READP PLACE BYTE FAIL tokens={} k={} experts={} d={} parts={} grid={gx}x{gy} first_diff_word={:?}", c.tokens, c.k, c.experts, c.d, c.parts, want.iter().zip(&got).position(|(p, q)| p != q).map(|i| i.saturating_sub(16) / 4))); }
        if got[..16] != GUARD || got[got.len() - 16..] != GUARD { return Err(format!("READP PLACE guard changed tokens={} d={}", c.tokens, c.d)); }
        n += 1;
    }
    Ok(n)
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("READP baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let base = g.function(ma, "prompt_rows")?;
    let started = Instant::now();
    if mode == "move-bytes" {
        let cand_permute = g.function(mb, "prompt_permute")?; let cand_place = g.function(mb, "prompt_place")?;
        let (mut cases, mut total) = (0u64, 0u64);
        for &cols in &[4usize, 8, 12, 16, 32, 36, 1020, 1024, 2048, 4096, 4100, 8192, 16384] {
            for &rows in &[1usize, 2, 3, 255, 256, 257, 2048] {
                for &source in &[1usize, 7, 256, 300] {
                    if rows * cols > 9_000_000 || source * cols > 9_000_000 || (rows > 300 && cols > 4100 && source > 7) { continue; }
                    if rows == 2048 && ![1024usize, 4096, 4100].contains(&cols) { continue; }
                    cases += 1; total += permute_case(&mut g, base, cand_permute, &Permute { rows, cols, source }, cases * 17, &[1, 3, 64, 1 << 20])?; g.clear()?;
                }
            }
            println!("READP permute cols={cols} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        }
        for &(tokens, k, experts, d, parts) in &[(1usize, 1usize, 1usize, 32usize, 1usize), (1, 1, 5, 32, 2), (1, 8, 8, 32, 2), (3, 2, 4, 32, 1), (3, 2, 4, 64, 2), (5, 3, 288, 96, 1), (7, 8, 288, 64, 2), (31, 8, 288, 32, 1),
            (32, 8, 288, 1024, 2), (33, 8, 288, 1024, 2), (64, 8, 16, 2048, 2), (100, 4, 9, 512, 4), (256, 8, 288, 1024, 2), (256, 8, 288, 2048, 2), (256, 8, 20, 1024, 2), (256, 8, 8, 2048, 2), (128, 8, 288, 2048, 1),
            (1024, 8, 288, 64, 2), (300, 8, 288, 1024, 2), (8192, 8, 288, 32, 1), (64, 1, 288, 4096, 1), (17, 6, 7, 160, 3), (256, 8, 288, 128, 16)] {
            if experts < k { continue; }
            cases += 1; total += place_case(&mut g, base, cand_place, &Place { tokens, k, experts, d, parts }, cases * 31, &[(1u32 << 20, 1u32 << 20), (1, 1), (2, 3), (1 << 20, 1)])?; g.clear()?;
            println!("READP place tokens={tokens} k={k} experts={experts} d={d} parts={parts} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        }
        println!("READP MOVE BYTES PASS cases={cases} comparisons={total} digest={} elapsed_s={:.1}", sha256::digest(format!("{cases}{total}").as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode == "move-time" {
        let cand_permute = g.function(mb, "prompt_permute")?; let cand_place = g.function(mb, "prompt_place")?;
        // Real group: 256 tokens x 8 experts of 288, hidden 4096 on two ranks.
        { let c = Permute { rows: 2048, cols: 4096, source: 256 };
          let table: Vec<f32> = (0..c.rows).map(|i| (i / 8) as f32).collect();
          let source = Buffer::new(&mut g, &bits(c.source * c.cols, 1))?; let tb = Buffer::new(&mut g, &table)?; let words = c.rows * c.cols;
          let out = Buffer::new(&mut g, &vec![0.0; words])?;
          let shape = PShape { op: PERMUTE, tokens: 1, heads: 0, dim: 0, taps: 0, rows: c.rows as u32, cols: c.cols as u32, first: c.source as u32, slots: 0, inner: empty() };
          let a = args(shape, [source.address, tb.address, 4, 4, 4], [c.source * c.cols, c.rows, 0, 0, 0], out.address, words, 1);
          pair(&g, "permute 2048 rows x 4096 (256 source rows)", &mut |g, arm| { let mut x = a; if arm == 0 { base_launch(g, base, &mut x) } else { g.launch(cand_permute, [c.rows as u32, 1, 1], 256, 0, &mut [ptr(&mut x)]) } })?; }
        for &(label, d, parts) in &[("place gate 2048 rows x 2048 (d 1024 x 2)", 1024usize, 2usize), ("place down 2048 rows x 4096 (d 2048 x 2)", 2048, 2)] {
            let c = Place { tokens: 256, k: 8, experts: 288, d, parts };
            let (places, rows) = grouping(&c, 9); let cols = d * parts; let slots = d * rows; let words = rows * cols;
            let exchange = Buffer::new(&mut g, &bits(slots * parts, 3))?; let old_t = Buffer::new(&mut g, &places)?; let new_t = Buffer::new(&mut g, &place_table(&places))?;
            let out = Buffer::new(&mut g, &vec![0.0; words])?;
            let shape = |op| PShape { op, tokens: 1, heads: 0, dim: d as u32, taps: parts as u32, rows: rows as u32, cols: cols as u32, first: 0, slots: slots as u32, inner: empty() };
            let a = args(shape(PLACE), [exchange.address, old_t.address, 4, 4, 4], [slots * parts, 3 * rows, 0, 0, 0], out.address, words, 1);
            let b2 = args(shape(PLACE_TILE), [exchange.address, new_t.address, 4, 4, 4], [slots * parts, 4 * rows, 0, 0, 0], out.address, words, 1);
            pair(&g, label, &mut |g, arm| { if arm == 0 { let mut x = a; base_launch(g, base, &mut x) } else { let mut y = b2; g.launch(cand_place, [rows.div_ceil(32) as u32, (cols / 32) as u32, 1], 256, 0, &mut [ptr(&mut y)]) } })?;
        }
        return Ok(());
    }
    if let Some(which) = mode.strip_prefix("list-") { MODULE_A.with(|c| c.set(ma)); return list_bytes(&mut g, base, mb, which, started); }
    if let Some(which) = mode.strip_prefix("fast-bytes-") { return fast_bytes(&mut g, base, mb, which, started); }
    if mode == "mhc-bytes" || mode == "mhc-time" {
        let project = g.function(ma, "mhc_boundary_project")?;
        return if mode == "mhc-bytes" { mhc_bytes(&mut g, project, mb, started) } else { mhc_time(&mut g, project, mb) };
    }
    if mode == "fast-time" { return fast_time(&mut g, base, mb); }
    Err(format!("READP unknown mode {mode}"))
}

// ---- wrapped causal operations and the convolution rows ----
const WRAPPED: u32 = 0; const CONV_ROWS: u32 = 4;
const PREPARE: u32 = 1; const GATED_NORM: u32 = 3; const ROUTE: u32 = 14; const EXPERT_FOLD: u32 = 15;
/// 0 plain, 1 wide range, 2 denormals and signed zeros, 3 large finite, 4 overflow with NaN and inf, 5 huge/tiny alternation, 6 add-order sensitive, 7 all ones, 8 cancellation.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = ((r.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0; let s = r.next();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(60),
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i % 4099 == 7 { f32::NAN } else if i % 4099 == 11 { f32::INFINITY } else if i % 4099 == 13 { f32::NEG_INFINITY } else { v } }
            5 => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
            6 => match i % 4 { 0 => 16777216.0, 1 => 1.0 + (s % 3) as f32, 2 => -16777216.0, _ => 1.0 },
            7 => 1.0,
            8 => if i % 2 == 0 { 1.0 + u * 1e-6 } else { -1.0 + u * 1e-6 },
            // Few levels: corrected expert scores tie.
            9 => (s % 5) as f32 - 2.0,
            // Expert logits that overflow the sigmoid both ways.
            10 => if s.is_multiple_of(3) { 200.0 } else if s % 3 == 1 { -200.0 } else { u },
            _ => 0.0,
        }
    }).collect()
}
/// Expert ids of a token: 0 distinct random, 1 duplicates (ids 0..3), 2 all equal, 3 descending, 4 ascending, 5 a NaN planted, 6 negatives and a large id, 7 gaps (multiples of 37).
fn ids(kind: u32, rows: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    let mut chosen: Vec<f32> = vec![];
    for i in 0..rows {
        let v = match kind {
            0 => { loop { let e = (r.next() % 288) as f32; if !chosen.contains(&e) { break e; } } }
            1 => (r.next() % 4) as f32, 2 => 7.0, 3 => (rows - i) as f32 * 3.0, 4 => i as f32 * 3.0,
            5 => if i == rows / 2 { f32::NAN } else { (r.next() % 288) as f32 },
            6 => match i % 3 { 0 => -3.0, 1 => 1e9, _ => (r.next() % 50) as f32 },
            _ => (r.next() % 8) as f32 * 37.0,
        };
        chosen.push(v);
    }
    chosen
}
struct Fx { inputs: Vec<Buffer>, weight: Buffer, shape: PShape, lengths: [usize; 5], units: usize, words: usize }
/// (input lengths, weight length, units, words) of the prompt rows, mirroring `Shape::lengths`.
fn wlengths(s: PShape) -> ([usize; 5], usize, usize, usize) {
    let (m, h, d, t, i) = (s.tokens as usize, s.heads as usize, s.dim as usize, s.taps as usize, s.inner);
    if s.op == CONV_ROWS { let w = h * d; return ([m * w, m * w, m * w, 3 * (t - 1) * w, 0], 3 * w * t, 3 * w * m, 1); }
    let (ih, id, ir, ik) = (i.heads as usize, i.dim as usize, i.rows as usize, i.cols as usize); let w = ih * id;
    match i.op {
        PREPARE => ([m * 3 * w, m * w, m * ih, 0, 0], ih + w, m * ih, 4 * id + 1),
        GATED_NORM => ([m * w, m * w, 0, 0, 0], id, m * ih, id),
        ROUTE => ([m * ir, 0, 0, 0, 0], ir, m, 2 * ik + 2 * ir + 2 * ih),
        _ => ([m * ir, m * ir, m * ir * ik, if i.taps != 0 { m * ik } else { 0 }, 0], 0, m * ik, 1),
    }
}
fn wfixture(g: &mut Gpu, s: PShape, ik: u32, wk: u32, idk: u32, seed: u64) -> Result<Fx> {
    let (lens, weight, units, words) = wlengths(s);
    let mut inputs = vec![];
    for (n, &len) in lens.iter().enumerate() {
        let v = if s.op == WRAPPED && s.inner.op == EXPERT_FOLD && n == 0 { (0..s.tokens as usize).flat_map(|t| ids(idk, s.inner.rows as usize, seed * 3 + t as u64)).collect() }
            else if s.op == WRAPPED && s.inner.op == ROUTE { pattern(if ik < 11 { ik } else { 9 }, len.max(1), seed * 7 + n as u64) }
            else { pattern((ik + n as u32) % 9, len.max(1), seed * 7 + n as u64) };
        inputs.push(Buffer::new(g, &v)?);
    }
    let wv = if s.op == WRAPPED && s.inner.op == ROUTE { pattern(if wk == 9 { 9 } else { wk % 9 }, weight.max(1), seed * 13 + 5) } else { pattern(wk, weight.max(1), seed * 13 + 5) };
    Ok(Fx { inputs, weight: Buffer::new(g, &wv)?, shape: s, lengths: lens, units, words })
}
fn wargs(f: &Fx, out: &Buffer) -> PArgs {
    let mut input = [4u64; 5]; for (i, b) in f.inputs.iter().enumerate() { if f.lengths[i] > 0 { input[i] = b.address; } }
    PArgs { input, lengths: f.lengths.map(|n| n as u64), weight: f.weight.address, weights: wlengths(f.shape).1 as u64, output: out.address, units: f.units as u64, words: f.words as u64, shape: f.shape }
}
/// Mirror of prompt_admission::fast_blocks.
fn fast_blocks(s: PShape) -> u32 {
    let (m, i) = (s.tokens as usize, s.inner);
    let b = if s.op == WRAPPED { match i.op { PREPARE | GATED_NORM => (m * i.heads as usize).div_ceil(4), ROUTE => m.div_ceil(4), _ => (m * i.cols as usize).div_ceil(128) } }
        else { (3 * s.heads as usize * s.dim as usize * m.div_ceil(32)).div_ceil(128) };
    b.max(1) as u32
}
fn fast_name(s: PShape) -> &'static str {
    if s.op == CONV_ROWS { return "prompt_conv"; }
    match s.inner.op { PREPARE => "prompt_prepare", GATED_NORM => "prompt_gated_norm", ROUTE => "prompt_route", _ => "prompt_fold" }
}
fn wshape(m: usize, inner: Inner) -> PShape { PShape { op: WRAPPED, tokens: m as u32, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, first: 0, slots: 0, inner } }
/// One same-bytes comparison: baseline generic kernel from a zero fill, candidate entry from poison, at several grids.
fn fast_compare(g: &mut Gpu, base: H, cand: H, fx: &Fx, grids: &[u32], label: &str) -> Result<u64> {
    let words = fx.units * fx.words;
    let out_a = Buffer::new(g, &vec![f32::NAN; words])?; out_a.zero(g)?;
    let mut x = wargs(fx, &out_a); base_launch(g, base, &mut x)?; g.sync()?; let want = out_a.read(g)?;
    let mut n = 0;
    for &grid in grids {
        let out_b = Buffer::new(g, &vec![f32::NAN; words])?; out_b.poison(g)?;
        let mut y = wargs(fx, &out_b);
        g.launch(cand, [grid.max(1), 1, 1], 128, 0, &mut [ptr(&mut y)])?; g.sync()?;
        let got = out_b.read(g)?;
        if got != want { return Err(format!("READP FAST BYTE FAIL {label} grid={grid} first_diff_word={:?}", want.iter().zip(&got).position(|(p, q)| p != q).map(|i| i.saturating_sub(16) / 4))); }
        if got[..16] != GUARD || got[got.len() - 16..] != GUARD { return Err(format!("READP FAST guard changed {label}")); }
        n += 1;
    }
    Ok(n)
}
fn fast_bytes(g: &mut Gpu, base: H, mb: H, which: &str, started: Instant) -> Result<()> {
    let (mut cases, mut total) = (0u64, 0u64);
    let ms = [1usize, 2, 3, 5, 64, 255, 256, 257];
    let kinds: [(u32, u32); 8] = [(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (6, 8), (8, 3), (3, 3)];
    if which == "prepare" || which == "gated" {
        let (op, name) = if which == "prepare" { (PREPARE, "prompt_prepare") } else { (GATED_NORM, "prompt_gated_norm") };
        let cand = g.function(mb, name)?;
        for &m in &ms { for &h in &[1usize, 2, 3, 31, 32, 33, 64, 65, 128] { for &d in &[1usize, 2, 31, 32, 33, 64, 100, 127, 128, 129, 255, 1024] {
            if m * h * d > 1_200_000 || (m > 5 && m < 255 && ![64usize].contains(&m)) { continue; }
            if m > 5 && !(h == 64 || h == 3 || h == 128) { continue; }
            let inner = Inner { op, heads: h as u32, dim: d as u32, taps: 0, rows: 0, cols: 0, epsilon: if op == GATED_NORM { 1e-6 } else { 0.0 }, bound: if op == PREPARE { -5.0 } else { 0.0 } };
            for (ki, &(ik, wk)) in kinds.iter().enumerate() {
                if m * h * d > 300_000 && ki > 2 { continue; }
                let fx = wfixture(g, wshape(m, inner), ik, wk, 0, 100 + cases)?;
                cases += 1; total += fast_compare(g, base, cand, &fx, &[fast_blocks(fx.shape), 1, 3, 1 << 20], &format!("{which} m={m} h={h} d={d} kinds={ik},{wk}"))?; g.clear()?;
            }
        } }
        println!("READP {which} m={m} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
    }
    if which == "route" {
        let cand = g.function(mb, "prompt_route")?;
        for &m in &[1usize, 2, 3, 4, 5, 255, 256, 257] { for &r in &[2usize, 3, 8, 31, 32, 33, 63, 64, 65, 100, 287, 288, 289, 511, 512] { for &k in &[1usize, 2, 7, 8, 9, 31, 32, 33, 63, 64] {
            if k > r || (m > 5 && !(r == 288 || r == 33 || r == 512 || r == 2)) || (m > 5 && ![1usize, 8, 33, 64].contains(&k)) { continue; }
            for &taps in &[0u32, 1] { for &(ik, wk) in &[(0u32, 0u32), (9, 9), (9, 2), (10, 0), (4, 0), (2, 2), (1, 1), (7, 7), (6, 8), (10, 9)] {
                if m > 5 && ![(0u32, 0u32), (9, 9), (4, 0), (10, 9)].contains(&(ik, wk)) { continue; }
                let inner = Inner { op: ROUTE, heads: 1, dim: 1, taps, rows: r as u32, cols: k as u32, epsilon: 0.0, bound: 2.5 };
                let fx = wfixture(g, wshape(m, inner), ik, wk, 0, 200 + cases)?;
                cases += 1; total += fast_compare(g, base, cand, &fx, &[fast_blocks(fx.shape), 1, 2, 1 << 20], &format!("route m={m} r={r} k={k} taps={taps} kinds={ik},{wk}"))?; g.clear()?;
            } }
        } }
        println!("READP route m={m} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
    }
    if which == "fold" {
        let cand = g.function(mb, "prompt_fold")?;
        for &m in &[1usize, 2, 3, 255, 256, 257] { for r in 1..=8usize { for &k in &[1usize, 2, 31, 32, 33, 127, 128, 129, 4096] { for &taps in &[0u32, 1] {
            if m > 3 && (!(k == 4096 || k == 33) || ![1usize, 8].contains(&r)) { continue; }
            for idk in 0..8u32 { for &(ik, wk) in &[(0u32, 0u32), (5, 5), (4, 0), (6, 8), (2, 2)] {
                if m > 3 && (idk > 1 || ik > 0 || (k > 1000 && taps == 0)) { continue; }
                let inner = Inner { op: EXPERT_FOLD, heads: 0, dim: 0, taps, rows: r as u32, cols: k as u32, epsilon: 0.0, bound: 0.0 };
                let fx = wfixture(g, wshape(m, inner), ik, wk, idk, 300 + cases)?;
                cases += 1; total += fast_compare(g, base, cand, &fx, &[fast_blocks(fx.shape), 1, 2, 1 << 20], &format!("fold m={m} r={r} k={k} taps={taps} ids={idk} kinds={ik},{wk}"))?; g.clear()?;
            } }
        } } }
        println!("READP fold m={m} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
    }
    if which == "conv" {
        let cand = g.function(mb, "prompt_conv")?;
        for &m in &[1usize, 2, 3, 31, 32, 33, 64, 255, 256, 257] { for t in 1..=8usize { for &(h, d) in &[(1usize, 1usize), (1, 2), (3, 11), (1, 32), (1, 33), (2, 64), (8, 128), (64, 128)] {
            if m * h * d * 3 > 2_000_000 || (m > 33 && (h * d != 8192 && h * d != 16 * 33)) { continue; }
            if m > 3 && ![1usize, 4, 8].contains(&t) { continue; }
            let w = h * d; let _ = w;
            for &(ik, wk) in &[(0u32, 0u32), (5, 5), (4, 0), (2, 2), (6, 8), (1, 1)] {
                if h * d * m > 100_000 && (ik, wk) != (0, 0) && (ik, wk) != (4, 0) { continue; }
                let s = PShape { op: CONV_ROWS, tokens: m as u32, heads: h as u32, dim: d as u32, taps: t as u32, rows: 0, cols: 0, first: 0, slots: 0, inner: empty() };
                let fx = wfixture(g, s, ik, wk, 0, 400 + cases)?;
                cases += 1; total += fast_compare(g, base, cand, &fx, &[fast_blocks(fx.shape), 1, 3, 1 << 20], &format!("conv m={m} taps={t} h={h} d={d} kinds={ik},{wk}"))?; g.clear()?;
            }
        } }
        println!("READP conv m={m} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
    }
    println!("READP FAST BYTES PASS {which} cases={cases} comparisons={total} digest={} elapsed_s={:.1}", sha256::digest(format!("{which}{cases}{total}").as_bytes()), started.elapsed().as_secs_f64());
    Ok(())
}
fn fast_time(g: &mut Gpu, base: H, mb: H) -> Result<()> {
    let m = 256usize;
    let cases: Vec<(&str, PShape)> = vec![
        ("prepare 64x128 x256 tokens", wshape(m, Inner { op: PREPARE, heads: 64, dim: 128, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: -5.0 })),
        ("gated_norm 64x128 x256", wshape(m, Inner { op: GATED_NORM, heads: 64, dim: 128, taps: 0, rows: 0, cols: 0, epsilon: 1e-6, bound: 0.0 })),
        ("route 288 experts top 8 x256", wshape(m, Inner { op: ROUTE, heads: 1, dim: 1, taps: 1, rows: 288, cols: 8, epsilon: 0.0, bound: 2.5 })),
        ("expert_fold 8 of 4096 x256", wshape(m, Inner { op: EXPERT_FOLD, heads: 0, dim: 0, taps: 1, rows: 8, cols: 4096, epsilon: 0.0, bound: 0.0 })),
        ("conv rows 64x128 taps 4 x256", PShape { op: CONV_ROWS, tokens: m as u32, heads: 64, dim: 128, taps: 4, rows: 0, cols: 0, first: 0, slots: 0, inner: empty() }),
    ];
    for (label, s) in cases {
        let fx = wfixture(g, s, 0, 0, 0, 77)?; let words = fx.units * fx.words;
        let out = Buffer::new(g, &vec![0.0; words])?; let a = wargs(&fx, &out); let cand = g.function(mb, fast_name(s))?; let blocks = fast_blocks(s);
        pair(g, label, &mut |g, arm| { let mut x = a; if arm == 0 { base_launch(g, base, &mut x) } else { g.launch(cand, [blocks, 1, 1], 128, 0, &mut [ptr(&mut x)]) } })?;
        g.clear()?;
    }
    Ok(())
}

// ---- the token-tiled mHC project ----
/// Mirror of gpu/src/mhc_params.rs `Shape` and `Boundary` (repr(C)).
#[repr(C)] #[derive(Clone, Copy)]
struct MShape { rows: u32, streams: u32, hidden: u32, iterations: u32, rms_eps: f32, eps: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Boundary { input: u64, previous_input: u64, previous_coefficients: u64, residual: u64, weights: u64, base: u64, scale: u64, mixing_normalized: u64, coefficients: u64, collapsed: u64,
    norm_weight: u64, norm_partials: u64, output: u64, shape: MShape, norm_epsilon: f32, norm_bf16: u32,
    prefetch: u64, prefetch_bytes: u64, prefetch_policy: u32, prefetch_reserved: u32 }
struct MFx { input: Buffer, prev_in: Option<Buffer>, prev_co: Option<Buffer>, weights: Buffer, base: Buffer, scale: Buffer, shape: MShape }
fn mfixture(g: &mut Gpu, rows: usize, streams: usize, hidden: usize, previous: bool, ik: u32, wk: u32, seed: u64) -> Result<MFx> {
    let (wide, count) = (streams * hidden, streams * (streams + 2));
    let input = Buffer::new(g, &pattern(ik, rows * if previous { hidden } else { wide }, seed * 7 + 1))?;
    let (prev_in, prev_co) = if previous { (Some(Buffer::new(g, &pattern((ik + 1) % 9, rows * wide, seed * 7 + 2))?), Some(Buffer::new(g, &pattern(if ik == 4 { 0 } else { (ik + 2) % 9 }, rows * count, seed * 7 + 3))?)) } else { (None, None) };
    let weights = Buffer::new(g, &pattern(wk, count * wide, seed * 13 + 5))?;
    let base = Buffer::new(g, &pattern(wk, count, seed * 13 + 6))?; let scale = Buffer::new(g, &pattern(wk, 3, seed * 13 + 7))?;
    Ok(MFx { input, prev_in, prev_co, weights, base, scale, shape: MShape { rows: rows as u32, streams: streams as u32, hidden: hidden as u32, iterations: 20, rms_eps: 1e-6, eps: 1e-6 } })
}
fn mboundary(f: &MFx, residual: &Buffer, mixing: &Buffer, coefficients: &Buffer) -> Boundary {
    Boundary { input: f.input.address, previous_input: f.prev_in.as_ref().map_or(0, |b| b.address), previous_coefficients: f.prev_co.as_ref().map_or(0, |b| b.address), residual: residual.address,
        weights: f.weights.address, base: f.base.address, scale: f.scale.address, mixing_normalized: mixing.address, coefficients: coefficients.address, collapsed: 4, norm_weight: 4, norm_partials: 4, output: 4,
        shape: f.shape, norm_epsilon: 1e-6, norm_bf16: 0, prefetch: 0, prefetch_bytes: 0, prefetch_policy: 0, prefetch_reserved: 0 }
}
fn mhc_threads(s: MShape) -> u32 { (s.streams * (s.streams + 2)).min(32) * 32 }
fn mhc_compare(g: &mut Gpu, base: H, resid: H, cand: H, f: &MFx, grids: &[u32], label: &str) -> Result<u64> {
    let s = f.shape; let (rows, wide, count) = (s.rows as usize, (s.streams * s.hidden) as usize, (s.streams * (s.streams + 2)) as usize);
    let mk = |g: &mut Gpu, n: usize, zero: bool| -> Result<Buffer> { let b = Buffer::new(g, &vec![f32::NAN; n])?; if zero { b.zero(g)?; } else { b.poison(g)?; } Ok(b) };
    let (ra, ma, ca) = (mk(g, rows * wide, true)?, mk(g, rows * wide, true)?, mk(g, rows * count, true)?);
    let mut x = mboundary(f, &ra, &ma, &ca);
    g.launch(base, [s.rows, 1, 1], mhc_threads(s), 0, &mut [ptr(&mut x)])?; g.sync()?;
    let want = [ra.read(g)?, ma.read(g)?, ca.read(g)?]; let mut n = 0;
    for &blocks in grids {
        let (rb, mb2, cb) = (mk(g, rows * wide, false)?, mk(g, rows * wide, false)?, mk(g, rows * count, false)?);
        let mut y = mboundary(f, &rb, &mb2, &cb);
        g.launch(resid, [(rows * wide).div_ceil(256) as u32, 1, 1], 256, 0, &mut [ptr(&mut y)])?;
        g.launch(cand, [blocks, 1, 1], 256, 0, &mut [ptr(&mut y)])?; g.sync()?;
        for (name, w, got) in [("residual", &want[0], rb.read(g)?), ("normalized", &want[1], mb2.read(g)?), ("coefficients", &want[2], cb.read(g)?)] {
            if *w != got { return Err(format!("READP MHC BYTE FAIL {label} plane={name} blocks={blocks} first_diff_word={:?}", w.iter().zip(&got).position(|(p, q)| p != q).map(|i| i.saturating_sub(16) / 4))); }
            if got[..16] != GUARD || got[got.len() - 16..] != GUARD { return Err(format!("READP MHC guard changed {label} plane={name}")); }
        }
        n += 1;
    }
    Ok(n)
}
fn mhc_bytes(g: &mut Gpu, base: H, mb: H, started: Instant) -> Result<()> {
    let cand = g.function(mb, "mhc_project_tile")?; let resid = g.function(mb, "mhc_residual_rows")?; let (mut cases, mut total) = (0u64, 0u64);
    let kinds: [(u32, u32); 6] = [(0, 0), (5, 5), (2, 2), (4, 0), (6, 8), (1, 1)];
    for &rows in &[9usize, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300] { for &streams in &[1usize, 2, 3, 4] { for &hidden in &[1usize, 2, 3, 4, 5, 7, 8, 31, 32, 33, 111, 112, 113, 127, 128, 129, 224, 225, 1000, 4096] {
        if (streams * hidden) % 4 != 0 { continue; }
        if rows * streams * hidden > 4_300_000 || (rows > 33 && !(hidden == 4096 || hidden == 113 || hidden == 224) ) || (rows > 33 && streams != 4 && streams != 1) { continue; }
        if rows > 17 && rows < 255 && ![64usize, 32, 31, 127, 128, 129].contains(&rows) { continue; }
        for previous in [false, true] { for (ki, &(ik, wk)) in kinds.iter().enumerate() {
            if (rows * streams * hidden > 800_000 && ki > 1) || (rows > 33 && ki > 3) { continue; }
            let f = mfixture(g, rows, streams, hidden, previous, ik, wk, 500 + cases)?;
            cases += 1; total += mhc_compare(g, base, resid, cand, &f, &[rows.div_ceil(8) as u32, rows.div_ceil(8) as u32 + 5, 1 << 20], &format!("mhc rows={rows} n={streams} h={hidden} previous={previous} kinds={ik},{wk}"))?; g.clear()?;
        } }
    } }
    println!("READP mhc rows={rows} ok cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
    println!("READP MHC BYTES PASS cases={cases} comparisons={total} digest={} elapsed_s={:.1}", sha256::digest(format!("mhc{cases}{total}").as_bytes()), started.elapsed().as_secs_f64());
    Ok(())
}
fn mhc_time(g: &mut Gpu, base: H, mb: H) -> Result<()> {
    let cand = g.function(mb, "mhc_project_tile")?; let resid = g.function(mb, "mhc_residual_rows")?;
    for &(rows, previous) in &[(256usize, true), (256, false), (64, true), (16, true)] {
        let f = mfixture(g, rows, 4, 4096, previous, 0, 0, 91)?; let s = f.shape; let (wide, count) = ((s.streams * s.hidden) as usize, (s.streams * (s.streams + 2)) as usize);
        let (r, m, c) = (Buffer::new(g, &vec![0.0; rows * wide])?, Buffer::new(g, &vec![0.0; rows * wide])?, Buffer::new(g, &vec![0.0; rows * count])?);
        let b = mboundary(&f, &r, &m, &c); let blocks = rows.div_ceil(8) as u32;
        pair(g, &format!("mhc project rows={rows} previous={previous}"), &mut |g, arm| { let mut x = b; if arm == 0 { g.launch(base, [s.rows, 1, 1], mhc_threads(s), 0, &mut [ptr(&mut x)]) } else { g.launch(resid, [(rows * wide).div_ceil(256) as u32, 1, 1], 256, 0, &mut [ptr(&mut x)])?; g.launch(cand, [blocks, 1, 1], 256, 0, &mut [ptr(&mut x)]) } })?;
        g.clear()?;
    }
    Ok(())
}

// The sweeps of the standing rule: they iterate `swept.rs` and nothing else, so the host test that compares the predicates of
// prompt_admission with those lists is a test of what these modes run. Every token count 1..=256 of the model's shape.
thread_local! { static MODULE_A: std::cell::Cell<H> = const { std::cell::Cell::new(std::ptr::null_mut()) }; }
fn g_module_a(_g: &Gpu) -> H { MODULE_A.with(|c| c.get()) }
fn list_bytes(g: &mut Gpu, base: H, mb: H, which: &str, started: Instant) -> Result<()> {
    let (mut cases, mut total) = (0u64, 0u64);
    let tick = |cases: u64, total: u64, label: &str| { if cases.is_multiple_of(64) { println!("READP list {label} cases={cases} comparisons={total} elapsed_s={:.1}", started.elapsed().as_secs_f64()); } };
    match which {
        "move" => {
            let (cp, cl) = (g.function(mb, "prompt_permute")?, g.function(mb, "prompt_place")?);
            for (rows, cols, source) in swept::permute_cases() {
                cases += 1; total += permute_case(g, base, cp, &Permute { rows, cols, source }, 9000 + cases * 17, &[rows as u32, 3])?; g.clear()?; tick(cases, total, "permute");
            }
            for (tokens, d, parts) in swept::place_cases() {
                cases += 1; total += place_case(g, base, cl, &Place { tokens, k: swept::ROUTED, experts: 288, d, parts }, 9000 + cases * 31, &[(1u32 << 20, 1u32 << 20), (2, 3)])?; g.clear()?; tick(cases, total, "place");
            }
        }
        "prepare" => {
            let (cp, cg) = (g.function(mb, "prompt_prepare")?, g.function(mb, "prompt_gated_norm")?);
            for (m, h, d, is_prepare) in swept::prepare_cases() {
                let inner = Inner { op: if is_prepare { PREPARE } else { GATED_NORM }, heads: h as u32, dim: d as u32, taps: 0, rows: 0, cols: 0, epsilon: if is_prepare { 0.0 } else { 1e-6 }, bound: if is_prepare { -5.0 } else { 0.0 } };
                for &(ik, wk) in &[(0u32, 0u32), (5, 5)] {
                    let fx = wfixture(g, wshape(m, inner), ik, wk, 0, 700 + cases)?;
                    cases += 1; total += fast_compare(g, base, if is_prepare { cp } else { cg }, &fx, &[fast_blocks(fx.shape), 3], &format!("list prepare={is_prepare} m={m} kinds={ik},{wk}"))?; g.clear()?; tick(cases, total, "prepare");
                }
            }
        }
        "route-fold" => {
            let (cr, cf) = (g.function(mb, "prompt_route")?, g.function(mb, "prompt_fold")?);
            for (m, r, k, t) in swept::route_cases() {
                let inner = Inner { op: ROUTE, heads: 1, dim: 1, taps: t as u32, rows: r as u32, cols: k as u32, epsilon: 0.0, bound: 2.5 };
                for &(ik, wk) in &[(0u32, 0u32), (9, 9), (10, 9)] {
                    let fx = wfixture(g, wshape(m, inner), ik, wk, 0, 800 + cases)?;
                    cases += 1; total += fast_compare(g, base, cr, &fx, &[fast_blocks(fx.shape), 2], &format!("list route m={m} kinds={ik},{wk}"))?; g.clear()?; tick(cases, total, "route");
                }
            }
            for (m, r, k, t) in swept::fold_cases() {
                let inner = Inner { op: EXPERT_FOLD, heads: 0, dim: 0, taps: t as u32, rows: r as u32, cols: k as u32, epsilon: 0.0, bound: 0.0 };
                for &(idk, ik, wk) in &[(0u32, 0u32, 0u32), (1, 5, 5), (0, 4, 0)] {
                    let fx = wfixture(g, wshape(m, inner), ik, wk, idk, 900 + cases)?;
                    cases += 1; total += fast_compare(g, base, cf, &fx, &[fast_blocks(fx.shape), 2], &format!("list fold m={m} ids={idk} kinds={ik},{wk}"))?; g.clear()?; tick(cases, total, "fold");
                }
            }
        }
        "conv" => {
            let cc = g.function(mb, "prompt_conv")?;
            for (m, h, d, t) in swept::conv_cases() {
                for &(ik, wk) in &[(0u32, 0u32), (5, 5), (4, 0)] {
                    let s = PShape { op: CONV_ROWS, tokens: m as u32, heads: h as u32, dim: d as u32, taps: t as u32, rows: 0, cols: 0, first: 0, slots: 0, inner: empty() };
                    let fx = wfixture(g, s, ik, wk, 0, 1000 + cases)?;
                    cases += 1; total += fast_compare(g, base, cc, &fx, &[fast_blocks(fx.shape), 3], &format!("list conv m={m} kinds={ik},{wk}"))?; g.clear()?; tick(cases, total, "conv");
                }
            }
        }
        "mhc" => {
            let (cand, resid, project) = (g.function(mb, "mhc_project_tile")?, g.function(mb, "mhc_residual_rows")?, g.function(g_module_a(g), "mhc_boundary_project")?);
            for (rows, streams, hidden) in swept::mhc_cases() {
                for previous in [false, true] { for &(ik, wk) in &[(0u32, 0u32), (5, 5)] {
                    let f = mfixture(g, rows, streams, hidden, previous, ik, wk, 1100 + cases)?;
                    cases += 1; total += mhc_compare(g, project, resid, cand, &f, &[rows.div_ceil(8) as u32, rows.div_ceil(8) as u32 + 5], &format!("list mhc rows={rows} previous={previous} kinds={ik},{wk}"))?; g.clear()?; tick(cases, total, "mhc");
                } }
            }
        }
        _ => return Err(format!("READP unknown list mode {which}")),
    }
    println!("READP LIST BYTES PASS {which} cases={cases} comparisons={total} digest={} elapsed_s={:.1}", sha256::digest(format!("{which}{cases}{total}").as_bytes()), started.elapsed().as_secs_f64());
    Ok(())
}
/// Alternating timing of the two arms (0 baseline, 1 candidate): median of nine batches of eight launches.
fn pair(g: &Gpu, label: &str, run: &mut dyn FnMut(&Gpu, usize) -> Result<()>) -> Result<()> {
    let mut t = [vec![], vec![]];
    for repeat in 0..9 { for k in 0..2 { let arm = (k + repeat) % 2; run(g, arm)?; g.sync()?; let n = 8usize; let t0 = Instant::now(); for _ in 0..n { run(g, arm)?; } g.sync()?; t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64); } }
    let (tb, tc) = (median(t[0].clone()), median(t[1].clone()));
    println!("READP TIME {label} base_us={tb:.1} cand_us={tc:.1} speedup={:.2}", tb / tc); Ok(())
}
