//! Same-bytes sweep and timing for the tiled pool scores (`pool_scores_tile`, `Op::PoolScoresRows`) against the
//! generic unit of `prompt_rows` in the BASELINE image (what main launches for this operation). One process, one device.
//! Run: kernel-bench --pool-scores <baseline image> <candidate image> [bytes|time|all]. Safe code only.
//! The admitted set is the lists of `crates/gpu/src/pool_scores_tile.rs`, included here by path: every listed dim,
//! head count and pool size runs, with pool and token counts at every listed edge and the causal context at both
//! sides of the pool boundaries, so the predicate cannot admit an unswept value. Both outputs start poisoned.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
#[allow(dead_code)] #[path = "../../../crates/gpu/src/pool_scores_tile.rs"] mod envelope;
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op PoolScoresRows is 8.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
const POOL_SCORES_ROWS: u32 = 8;
struct Rng(u64);
impl Rng {
    fn step(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.step() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// 0 plain, 1 wide range, 2 denormals and signed zeros, 3 large finite (sums overflow to inf), 4 huge/tiny alternation,
/// 5 sparse NaN and infinities.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.step();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(70),
            4 => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
            _ => match i % 4099 { 7 => f32::NAN, 11 => f32::INFINITY, 13 => f32::NEG_INFINITY, _ => u * 8.0 },
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
struct Case { rows: usize, tokens: usize, heads: usize, dim: usize, taps: usize, first: usize }
fn args(c: &Case, q: &Buffer, k: &Buffer, w: &Buffer, out: &Buffer) -> Arguments {
    let inner = Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    let (m, h, d, r) = (c.tokens, c.heads, c.dim, c.rows);
    Arguments { input: [q.address, k.address, w.address, 4, 4], lengths: [(m * h * d) as u64, (r * d) as u64, (m * h) as u64, 0, 0],
        weight: 4, weights: 0, output: out.address, units: (m * r) as u64, words: 1,
        shape: PShape { op: POOL_SCORES_ROWS, tokens: m as u32, heads: h as u32, dim: d as u32, taps: c.taps as u32, rows: r as u32, cols: 0, first: c.first as u32, slots: 0, inner } }
}
/// The generic launch: one thread a word, 128 threads a block.
fn launch_rows(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }
fn launch_tile(g: &Gpu, f: H, c: &Case, a: &mut Arguments) -> Result<()> {
    let blocks = envelope::blocks(c.rows, c.tokens).ok_or("pool scores grid overflow")?;
    g.launch(f, [blocks, 1, 1], envelope::THREADS, 0, &mut [ptr(a)])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Fixture { q: Buffer, k: Buffer, w: Buffer, a: Buffer, b: Buffer }
fn fixture(g: &mut Gpu, c: &Case, kind: u32, seed: u64) -> Result<Fixture> {
    let (m, h, d, r) = (c.tokens, c.heads, c.dim, c.rows);
    // Head weights of both signs in every kind, so the clamp at zero and negative terms both occur.
    Ok(Fixture { q: Buffer::new(g, &pattern(kind, m * h * d, seed))?, k: Buffer::new(g, &pattern(kind, r * d, seed + 3))?, w: Buffer::new(g, &pattern(if kind == 5 { 5 } else { 0 }, m * h, seed + 7))?,
        a: Buffer::new(g, &vec![0.0; m * r])?, b: Buffer::new(g, &vec![0.0; m * r])? })
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (ia, ib) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("POOLSCORES baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&ia), sha256::digest(&ib));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&ia)?, g.module(&ib)?);
    let (rows_fn, tile_fn) = (g.function(ma, "prompt_rows")?, g.function(mb, "pool_scores_tile")?);
    let started = Instant::now(); let (comparisons, words, zero_words) = (std::cell::Cell::new(0usize), std::cell::Cell::new(0usize), std::cell::Cell::new(0usize));
    let check = |g: &mut Gpu, c: Case, kind: u32, seed: u64| -> Result<()> {
        if !envelope::admitted(c.rows, c.tokens, c.heads, c.dim, c.taps, 0, 0) { return Err(format!("case outside the admitted set: {} {} {} {} {}", c.rows, c.tokens, c.heads, c.dim, c.taps)); }
        let fx = fixture(g, &c, kind, seed)?;
        fx.a.fill(g, 0xff)?; fx.b.fill(g, 0xee)?;
        let (mut aa, mut ab) = (args(&c, &fx.q, &fx.k, &fx.w, &fx.a), args(&c, &fx.q, &fx.k, &fx.w, &fx.b));
        launch_rows(g, rows_fn, &mut aa)?; launch_tile(g, tile_fn, &c, &mut ab)?; g.sync()?;
        let (x, y) = (fx.a.read(g)?, fx.b.read(g)?);
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD || y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err("POOLSCORES output guard changed".into()); }
        if x != y {
            let at = x.chunks_exact(4).zip(y.chunks_exact(4)).position(|(p, q)| p != q).unwrap_or(0).saturating_sub(4);
            return Err(format!("POOLSCORES BYTE FAIL pools={} tokens={} heads={} dim={} taps={} first={} kind={kind} word={} token={} pool={}", c.rows, c.tokens, c.heads, c.dim, c.taps, c.first, at, at / c.rows, at % c.rows));
        }
        comparisons.set(comparisons.get() + 1); words.set(words.get() + c.rows * c.tokens); zero_words.set(zero_words.get() + x[16..x.len() - 16].chunks_exact(4).filter(|w| w == &[0, 0, 0, 0]).count());
        g.clear()
    };
    if mode == "bytes" || mode == "all" {
        let mut seed = 100u64;
        // Every listed dim, head count and pool size, at tile edges of pools and tokens, every activation kind.
        for &dim in envelope::DIMS { for kind in 0..6u32 { seed += 1; check(&mut g, Case { rows: 130, tokens: 65, heads: 2, dim, taps: 4, first: 400 + dim }, kind, seed)?; } }
        // The token floor (17) and the counts just above it, at the model's word.
        for tokens in [17usize, 18, 31, 32, 33] { for kind in [0u32, 2, 5] { seed += 1; check(&mut g, Case { rows: 600, tokens, heads: 32, dim: 128, taps: 4, first: 2300 }, kind, seed)?; } }
        // The work floor (8,192 pool x token words) from both shapes that meet it exactly.
        for (rows, tokens) in [(128usize, 64usize), (482, 17), (8, 1024), (129, 64)] { for kind in [0u32, 2, 5] { seed += 1; check(&mut g, Case { rows, tokens, heads: 32, dim: 128, taps: 4, first: rows * 4 - tokens / 2 }, kind, seed)?; } }
        for &heads in envelope::HEADS { for kind in 0..6u32 { seed += 1; check(&mut g, Case { rows: 130, tokens: 67, heads, dim: 12, taps: 4, first: 500 }, kind, seed)?; } }
        // The causal context: no pool complete (all zero), the first pool completing inside the group, a pool boundary at
        // each side of a token tile edge, every pool complete, and more context than pools.
        for &taps in envelope::TAPS { for first in [0usize, 1, taps.saturating_sub(1), taps, taps + 1, 63 * taps, 64 * taps, 65 * taps, 129 * taps + 1, 1000 * taps] {
            seed += 1; check(&mut g, Case { rows: 130, tokens: 130, heads: 3, dim: 8, taps, first }, (seed % 6) as u32, seed)?;
        } }
        println!("POOLSCORES lists ok (dims, heads, pool sizes x contexts) comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Pool and token counts: every listed edge against every listed edge, at a small word (2 heads of 8), with the context
        // set so that the last token completes about three quarters of the pools.
        for &rows in envelope::ROW_EDGES { for &tokens in envelope::TOKEN_EDGES {
            if rows * tokens > 4_500_000 && !(rows == envelope::MAX_ROWS && tokens == envelope::MAX_TOKENS) && !(tokens == envelope::MAX_TOKENS || rows == envelope::MAX_ROWS) { continue; }
            // Under the work floor the engine keeps the row kernel: not a launch of the tile.
            if !envelope::admitted(rows, tokens, 2, 8, 4, 0, 0) { continue; }
            seed += 1; let first = (rows * 4 * 3 / 4).saturating_sub(tokens / 2);
            check(&mut g, Case { rows, tokens, heads: 2, dim: 8, taps: 4, first }, (seed % 6) as u32, seed)?;
        } }
        println!("POOLSCORES edges ok (pools x tokens) comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // The model's word (32 heads of 128, pools of 4) at the groups of an 8,192-token read and at odd sizes.
        // Then the groups of a 32,768-token read (pools 4,096, 6,144 and 8,192).
        for (rows, tokens, first) in [(256usize, 1024usize, 1usize), (512, 1024, 1025), (1024, 1024, 3073), (2048, 1024, 7169), (2048, 64, 8129), (520, 17, 2065), (1039, 62, 4097), (300, 255, 950), (2051, 129, 8076),
            (4096, 1024, 15361), (6144, 1024, 23553), (8192, 1024, 31745), (8192, 40, 32729)] {
            for kind in [0u32, 2, 5] { seed += 1; check(&mut g, Case { rows, tokens, heads: 32, dim: 128, taps: 4, first }, kind, seed)?; }
        }
        println!("POOLSCORES BYTES PASS comparisons={} words={} zero_words={} elapsed_s={:.1}", comparisons.get(), words.get(), zero_words.get(), started.elapsed().as_secs_f64());
    }
    if mode == "time" || mode == "all" {
        // The model's word at the first, a middle and the last group of an 8,192-token read, a 256-token group, and a tail group.
        // Then the groups of a 32,768-token read: pools 4,096 (context 16,384), 6,144 (24,576) and 8,192 (32,768).
        for (rows, tokens, first) in [(256usize, 1024usize, 1usize), (1024, 1024, 3073), (2048, 1024, 7169), (1040, 256, 3905), (520, 17, 2065), (1039, 62, 4097), (128, 64, 449),
            (4096, 1024, 15361), (6144, 1024, 23553), (8192, 1024, 31745)] {
            let c = Case { rows, tokens, heads: 32, dim: 128, taps: 4, first };
            let fx = fixture(&mut g, &c, 0, 31)?;
            let (mut aa, mut ab) = (args(&c, &fx.q, &fx.k, &fx.w, &fx.a), args(&c, &fx.q, &fx.k, &fx.w, &fx.b));
            let (mut ta, mut tb) = (vec![], vec![]);
            for repeat in 0..5 {
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    if arm == 0 { launch_rows(&g, rows_fn, &mut aa)?; } else { launch_tile(&g, tile_fn, &c, &mut ab)?; } g.sync()?;
                    let n = 3; let t0 = Instant::now();
                    for _ in 0..n { if arm == 0 { launch_rows(&g, rows_fn, &mut aa)?; } else { launch_tile(&g, tile_fn, &c, &mut ab)?; } }
                    g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let spread = |v: &[f64]| { let (lo, hi) = v.iter().fold((f64::MAX, 0f64), |(l, h), x| (l.min(*x), h.max(*x))); hi - lo };
            let (ra, rb) = (median(ta.clone()), median(tb.clone()));
            // Live products: each token scores (first + token) / 4 pools, capped at the pools of the launch.
            let live: usize = (0..tokens).map(|t| ((first + t) / 4).min(rows)).sum::<usize>() * 32 * 128;
            println!("POOLSCORES TIME pools={rows} tokens={tokens} first={first} rows_us={ra:.0} tile_us={rb:.0} ratio={:.2} rows_GMACs={:.0} tile_GMACs={:.0} rows_spread_us={:.0} tile_spread_us={:.0}",
                ra / rb, live as f64 / ra / 1e3, live as f64 / rb / 1e3, spread(&ta), spread(&tb));
            g.clear()?;
        }
    }
    println!("POOLSCORES DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
