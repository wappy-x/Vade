//! Same-bytes sweep and timing for the staged recurrent update (`step_rows_tile`, `Op::StepRows`) against
//! `step_rows_block` in the BASELINE image (what main launches for a prompt group) and against the generic unit of
//! `prompt_rows`. One process, one device. Run: kernel-bench --step-tile <baseline image> <candidate image> [bytes|time|all].
//! The admitted set is the lists of `crates/gpu/src/step_tile.rs`, included here by path: every listed head size and
//! head count runs, with the token count at every listed edge. Both outputs start poisoned with different bytes, so a
//! word either kernel leaves unwritten fails the compare. Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
#[allow(dead_code)] #[path = "../../../crates/gpu/src/step_tile.rs"] mod envelope;
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op StepRows is 7.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
const STEP_ROWS: u32 = 7;
struct Rng(u64);
impl Rng {
    fn step(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.step() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// 0 plain (small values keep the recurrence in range), 1 wide range, 2 denormals and signed zeros, 3 large finite
/// (sums overflow to inf), 4 huge/tiny alternation, 5 sparse NaN and infinities.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.step();
        match kind {
            0 => u * 0.0625,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(70),
            4 => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
            _ => match i % 4099 { 7 => f32::NAN, 11 => f32::INFINITY, 13 => f32::NEG_INFINITY, _ => u * 0.5 },
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
    fn fill(&self, g: &Gpu, byte: u8) -> Result<()> { g.upload_at(self.address, &vec![byte; self.bytes]) }
}
#[derive(Clone, Copy)]
struct Case { tokens: usize, heads: usize, dim: usize }
impl Case {
    /// Input rows (keys, update, target, a spare vector and beta a head and token), the state, the decay.
    fn lengths(&self) -> [usize; 3] { [self.tokens * self.heads * (4 * self.dim + 1), self.heads * self.dim * self.dim, self.tokens * self.heads * self.dim] }
    /// Output words: the state, then every token's output.
    fn words(&self) -> usize { self.heads * self.dim * (self.dim + self.tokens) }
}
fn args(c: &Case, x: &[Buffer; 3], out: &Buffer) -> Arguments {
    let inner = Inner { op: 5, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    let l = c.lengths();
    Arguments { input: [x[0].address, x[1].address, x[2].address, 0, 0], lengths: [l[0] as u64, l[1] as u64, l[2] as u64, 0, 0], weight: 4, weights: 0, output: out.address,
        units: (c.heads * c.dim) as u64, words: (c.dim + c.tokens) as u64,
        shape: PShape { op: STEP_ROWS, tokens: c.tokens as u32, heads: c.heads as u32, dim: c.dim as u32, taps: 0, rows: 0, cols: 0, first: 0, slots: 0, inner } }
}
/// The generic launch: one thread a column, 128 threads a block.
fn launch_rows(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }
/// `step_rows_block` and `step_rows_tile` share their grid: a block for each head and 32 values.
fn launch_block(g: &Gpu, f: H, c: &Case, a: &mut Arguments) -> Result<()> {
    let blocks = envelope::blocks(c.heads, c.dim).ok_or("step grid overflow")?;
    g.launch(f, [blocks, 1, 1], envelope::THREADS, 0, &mut [ptr(a)])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn inputs(g: &mut Gpu, c: &Case, kind: u32, seed: u64) -> Result<[Buffer; 3]> {
    let l = c.lengths();
    // The decay stays plain in the finite kinds (a real decay is in 0..1); the hard kinds reach it too.
    Ok([Buffer::new(g, &pattern(kind, l[0], seed))?, Buffer::new(g, &pattern(kind, l[1], seed + 3))?, Buffer::new(g, &pattern(if kind == 0 { 0 } else { kind }, l[2], seed + 7))?])
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (ia, ib) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("STEPTILE baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&ia), sha256::digest(&ib));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&ia)?, g.module(&ib)?);
    let (rows_fn, block_fn, tile_fn) = (g.function(ma, "prompt_rows")?, g.function(ma, "step_rows_block")?, g.function(mb, "step_rows_tile")?);
    let started = Instant::now(); let (comparisons, words) = (std::cell::Cell::new(0usize), std::cell::Cell::new(0usize));
    // `generic`: also compare with the generic unit of `prompt_rows` (it needs a zeroed output; it is slow on large shapes).
    let check = |g: &mut Gpu, c: Case, kind: u32, seed: u64, generic: bool| -> Result<()> {
        if !envelope::admitted(c.tokens, c.heads, c.dim) { return Err(format!("case outside the admitted set: tokens={} heads={} dim={}", c.tokens, c.heads, c.dim)); }
        let x = inputs(g, &c, kind, seed)?;
        let (a, b) = (Buffer::new(g, &vec![0.0; c.words()])?, Buffer::new(g, &vec![0.0; c.words()])?);
        a.fill(g, 0xff)?; b.fill(g, 0xee)?;
        let (mut aa, mut ab) = (args(&c, &x, &a), args(&c, &x, &b));
        launch_block(g, block_fn, &c, &mut aa)?; launch_block(g, tile_fn, &c, &mut ab)?; g.sync()?;
        let (p, q) = (a.read(g)?, b.read(g)?);
        if p[..16] != GUARD || p[p.len() - 16..] != GUARD || q[..16] != GUARD || q[q.len() - 16..] != GUARD { return Err("STEPTILE output guard changed".into()); }
        if p != q {
            let at = p.chunks_exact(4).zip(q.chunks_exact(4)).position(|(p, q)| p != q).unwrap_or(0).saturating_sub(4);
            return Err(format!("STEPTILE BYTE FAIL against step_rows_block tokens={} heads={} dim={} kind={kind} word={at} state_words={}", c.tokens, c.heads, c.dim, c.heads * c.dim * c.dim));
        }
        if generic {
            let r = Buffer::new(g, &vec![0.0; c.words()])?; let mut ar = args(&c, &x, &r);
            launch_rows(g, rows_fn, &mut ar)?; g.sync()?;
            if r.read(g)? != q { return Err(format!("STEPTILE BYTE FAIL against prompt_rows tokens={} heads={} dim={} kind={kind}", c.tokens, c.heads, c.dim)); }
        }
        comparisons.set(comparisons.get() + 1); words.set(words.get() + c.words());
        g.clear()
    };
    if mode == "bytes" || mode == "all" {
        let mut seed = 300u64;
        // Every listed head size and head count, at stage edges of the token count, every data kind; the generic unit too.
        for &dim in envelope::DIMS { for &tokens in &[17usize, 18, 19, 20, 21, 33, 65] { for kind in 0..6u32 { seed += 1; check(&mut g, Case { tokens, heads: 3, dim }, kind, seed, true)?; } } }
        for &heads in envelope::HEADS { for kind in 0..6u32 { seed += 1; check(&mut g, Case { tokens: 67, heads, dim: 128 }, kind, seed, heads <= 8)?; } }
        println!("STEPTILE lists ok (head sizes, head counts) comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Every listed token edge at a small word and at the model's head size.
        for &tokens in envelope::TOKEN_EDGES { seed += 1; check(&mut g, Case { tokens, heads: 1, dim: 128 }, (seed % 6) as u32, seed, true)?; seed += 1; check(&mut g, Case { tokens, heads: 2, dim: 128 }, ((seed + 3) % 6) as u32, seed, tokens <= 129)?; }
        println!("STEPTILE edges ok (tokens) comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // The model's word (local heads of 128) at a full prompt group, at tail groups and at the floor.
        for &(tokens, heads) in &[(1024usize, 32usize), (1024, 64), (1024, 33), (512, 32), (256, 32), (62, 32), (17, 32), (127, 64), (1023, 32)] {
            for kind in [0u32, 2, 5] { seed += 1; check(&mut g, Case { tokens, heads, dim: 128 }, kind, seed, false)?; }
        }
        println!("STEPTILE BYTES PASS comparisons={} words={} elapsed_s={:.1}", comparisons.get(), words.get(), started.elapsed().as_secs_f64());
    }
    if mode == "time" || mode == "all" {
        // The model's word at a full group and at smaller groups. Several input copies rotate, so a launch streams its
        // rows from memory as a layer's fresh rows do in a read.
        for &(tokens, heads) in &[(1024usize, 32usize), (1024, 64), (512, 32), (256, 32), (128, 32), (64, 32), (32, 32), (17, 32), (17, 64)] {
            let c = Case { tokens, heads, dim: 128 };
            let copies = ((512usize << 20) / (c.lengths()[0] * 4 + c.lengths()[2] * 4)).clamp(2, 8);
            let xs = (0..copies).map(|i| inputs(&mut g, &c, 0, 31 + i as u64 * 13)).collect::<Result<Vec<_>>>()?;
            let (a, b) = (Buffer::new(&mut g, &vec![0.0; c.words()])?, Buffer::new(&mut g, &vec![0.0; c.words()])?);
            let (mut ta, mut tb) = (vec![], vec![]);
            for repeat in 0..5 {
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let n = copies; let t0 = Instant::now();
                    for i in 0..n { let (mut aa, mut ab) = (args(&c, &xs[i], &a), args(&c, &xs[i], &b));
                        if arm == 0 { launch_block(&g, block_fn, &c, &mut aa)?; } else { launch_block(&g, tile_fn, &c, &mut ab)?; } }
                    g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let spread = |v: &[f64]| { let (lo, hi) = v.iter().fold((f64::MAX, 0f64), |(l, h), x| (l.min(*x), h.max(*x))); hi - lo };
            let (ra, rb) = (median(ta.clone()), median(tb.clone()));
            println!("STEPTILE TIME tokens={tokens} heads={heads} dim=128 block_us={ra:.0} tile_us={rb:.0} ratio={:.2} block_us_a_token={:.3} tile_us_a_token={:.3} block_spread_us={:.0} tile_spread_us={:.0} input_copies={copies}",
                ra / rb, ra / tokens as f64, rb / tokens as f64, spread(&ta), spread(&tb));
            g.clear()?;
        }
    }
    println!("STEPTILE DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
