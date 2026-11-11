//! Same-bytes sweep for the tiled attention scores (`scores_rows_tile`) against the two `prompt_rows` paths that
//! compute the same words: the per-unit ScoresMap path (what a shape the tile does not admit runs) and the per-word
//! ScoresRows line path. One process, one device, one image. Safe code only.
//! The admitted set is the lists of `crates/gpu/src/scores_tile.rs`, included here by path: every listed dim, taps and
//! head count runs, with row and token counts at every listed edge, so the predicate cannot admit an unswept value.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
#[allow(dead_code)] #[path = "../../../crates/gpu/src/scores_tile.rs"] mod envelope;
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op ScoresRows is 10, ScoresMap is 15.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
const SCORES_ROWS: u32 = 10;
const SCORES_MAP: u32 = 15;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// 0 plain, 1 wide range, 2 denormals and signed zeros, 3 large finite (sums overflow to inf), 4 huge/tiny alternation.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.next();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(70),
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
    fn fill(&self, g: &Gpu, byte: u8) -> Result<()> { g.upload_at(self.address, &vec![byte; self.bytes]) }
}
/// Selected rows of each token (`[tokens, slots+1]`, the count last) and the map `[rows, tokens]` from pair to slot.
/// Every token's count differs (some select nothing); rows 0 and rows-1 are always among the candidates, so the
/// first and last row tiles are hit; the rest are spread by a token-dependent stride.
fn selection(rows: usize, tokens: usize, slots: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let mut selected = vec![0.0f32; tokens * (slots + 1)]; let mut map = vec![-1.0f32; rows * tokens]; let mut pairs = 0;
    for token in 0..tokens {
        let want = if token % 11 == 5 { 0 } else { 1 + (token * 7) % slots };
        let mut ids = vec![0, rows - 1]; ids.extend((0..want).map(|j| (token * 29 + j * 13 + j * j) % rows)); ids.sort_unstable(); ids.dedup(); ids.truncate(want);
        for (slot, &id) in ids.iter().enumerate() { selected[token * (slots + 1) + slot] = id as f32; map[id * tokens + token] = slot as f32; }
        selected[token * (slots + 1) + slots] = ids.len() as f32; pairs += ids.len();
    }
    (selected, map, pairs)
}
struct Case { rows: usize, tokens: usize, heads: usize, dim: usize, taps: usize, slots: usize }
fn args(c: &Case, op: u32, q: &Buffer, k: &Buffer, sel: &Buffer, map: &Buffer, out: &Buffer) -> Arguments {
    let inner = Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    let (m, h, d, t, r, s) = (c.tokens, c.heads, c.dim, c.taps, c.rows, c.slots);
    let units = m * h;
    // ScoresMap on prompt_rows: one thread a unit. ScoresRows lines: one thread a (unit, slot) word.
    let threads = if op == SCORES_ROWS { units * s } else { units };
    Arguments { input: [q.address, k.address, sel.address, if op == SCORES_MAP { map.address } else { 4 }, 4],
        lengths: [(m * h * d) as u64, (r * h * (d + t)) as u64, (m * (s + 1)) as u64, if op == SCORES_MAP { (r * m) as u64 } else { 0 }, 0],
        weight: 4, weights: 0, output: out.address, units: threads as u64, words: s as u64,
        shape: PShape { op, tokens: m as u32, heads: h as u32, dim: d as u32, taps: t as u32, rows: r as u32, cols: s as u32, first: 0, slots: s as u32, inner } }
}
fn launch_rows(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }
fn launch_tile(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> {
    let s = a.shape; let blocks = envelope::blocks(s.rows as usize, s.tokens as usize, s.heads as usize).ok_or("scores tile grid overflow")?;
    g.launch(f, [blocks, 1, 1], envelope::THREADS, 0, &mut [ptr(a)])
}
/// One shape, three kernels, byte-equal outputs. The per-unit ScoresMap path writes every word (zeros past the count),
/// so its output is poisoned first; the ScoresRows line path and the tile write only selected words, so theirs are
/// zeroed first, as `prompt_operation` does on the stream. `q` and `k` may be larger than the shape: only the shape is read.
fn compare(g: &mut Gpu, fns: (H, H), c: &Case, q: &Buffer, k: &Buffer) -> Result<u64> {
    let (m, h, d, t, r, s) = (c.tokens, c.heads, c.dim, c.taps, c.rows, c.slots);
    if !envelope::admitted(r, m, h, d, t, q.address, k.address) { return Err(format!("SCORESTILE case outside the admitted set rows={r} tokens={m} heads={h} dim={d} taps={t}")); }
    if q.bytes < m * h * d * 4 || k.bytes < r * h * (d + t) * 4 { return Err("SCORESTILE fixture smaller than the shape".into()); }
    let (selected, map, _) = selection(r, m, s);
    let sel = Buffer::new(g, &selected)?; let mapb = Buffer::new(g, &map)?;
    let outs = [Buffer::new(g, &vec![0.0; m * h * s])?, Buffer::new(g, &vec![0.0; m * h * s])?, Buffer::new(g, &vec![0.0; m * h * s])?];
    outs[0].fill(g, 0xff)?;
    let mut a = [args(c, SCORES_MAP, q, k, &sel, &mapb, &outs[0]), args(c, SCORES_ROWS, q, k, &sel, &mapb, &outs[1]), args(c, SCORES_MAP, q, k, &sel, &mapb, &outs[2])];
    launch_rows(g, fns.0, &mut a[0])?; launch_rows(g, fns.0, &mut a[1])?; launch_tile(g, fns.1, &mut a[2])?; g.sync()?;
    let (x, y, z) = (outs[0].read(g)?, outs[1].read(g)?, outs[2].read(g)?);
    for (name, other) in [("rows-lines", &y), ("tile", &z)] {
        if &x != other {
            let at = x.iter().zip(other.iter()).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i));
            return Err(format!("SCORESTILE BYTE FAIL {name} rows={r} tokens={m} heads={h} dim={d} taps={t} slots={s} first_word_and_byte={at:?}"));
        }
        if other[..16] != GUARD || other[other.len() - 16..] != GUARD { return Err(format!("SCORESTILE guard changed {name} rows={r} tokens={m}")); }
    }
    if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("SCORESTILE guard changed reference".into()); }
    Ok((m * h * s * 4) as u64)
}
/// Queries and keys for shapes up to these sizes, one pattern family each.
fn fixture(g: &mut Gpu, tokens: usize, rows: usize, heads: usize, dim: usize, taps: usize, kinds: (u32, u32), seed: u64) -> Result<(Buffer, Buffer)> {
    Ok((Buffer::new(g, &pattern(kinds.0, tokens * heads * dim, seed))?, Buffer::new(g, &pattern(kinds.1, rows * heads * (dim + taps), seed * 3 + 1))?))
}
pub(crate) fn run(image_path: &str) -> Result<()> {
    let image = std::fs::read(image_path).map_err(|e| e.to_string())?;
    println!("SCORESTILE image_sha256={} dims={:?} taps={:?} heads={:?} row_edges={:?} token_edges={:?}", sha256::digest(&image), envelope::DIMS, envelope::TAPS, envelope::HEADS, envelope::ROW_EDGES, envelope::TOKEN_EDGES);
    let mut g = Gpu::new()?; let module = g.module(&image)?;
    let fns = (g.function(module, "prompt_rows")?, g.function(module, "scores_rows_tile")?);
    let started = Instant::now(); let (mut comparisons, mut bytes) = (0u64, 0u64);
    let kinds = [(0u32, 0u32), (1, 1), (2, 2), (4, 4), (3, 3), (0, 4)];
    // 1. Every listed dim, at every row edge up to 4,097 and every token edge up to 257, heads 3, taps cycling the list;
    //    the pattern family changes with the dim.
    for (i, &dim) in envelope::DIMS.iter().enumerate() {
        let taps = envelope::TAPS[i % envelope::TAPS.len()];
        let (q, k) = fixture(&mut g, 257, 4097, 3, dim, taps, kinds[i % kinds.len()], dim as u64)?;
        for &rows in envelope::ROW_EDGES.iter().filter(|&&r| r <= 4097) { for &tokens in envelope::TOKEN_EDGES.iter().filter(|&&t| t <= 257) {
            let c = Case { rows, tokens, heads: 3, dim, taps, slots: 37.min(rows + 2) };
            bytes += compare(&mut g, fns, &c, &q, &k)?; comparisons += 1;
        } }
        g.clear()?;
        println!("SCORESTILE dim={dim} taps={taps} patterns={:?} edges ok comparisons_so_far={comparisons} elapsed_s={:.1}", kinds[i % kinds.len()], started.elapsed().as_secs_f64());
    }
    // 2. Every listed taps and head count, at tile-edge shapes, with every pattern family.
    for (i, &taps) in envelope::TAPS.iter().enumerate() { for (j, &heads) in envelope::HEADS.iter().enumerate() {
        let (q, k) = fixture(&mut g, 65, 129, heads, 256, taps, kinds[(i + j) % kinds.len()], (taps * 17 + heads) as u64)?;
        for &(rows, tokens, dim) in &[(65usize, 65usize, 36usize), (129, 63, 256), (1, 1, 4), (128, 64, 252)] {
            let c = Case { rows, tokens, heads, dim, taps, slots: 33.min(rows + 2) };
            bytes += compare(&mut g, fns, &c, &q, &k)?; comparisons += 1;
        }
        g.clear()?;
    } }
    println!("SCORESTILE taps x heads ok comparisons_so_far={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    // 3. The interval ends: rows 16,383 and 16,384, tokens 1,023 and 1,024, with the widest and the model's dims
    //    (Grok's named shape rows=16384 tokens=1024 dim=4096 taps=4 among them), and the model's shape at 1 and 2 ranks.
    for (rows, tokens, heads, dim, taps, slots, kind) in [(16384usize, 1024usize, 1usize, 4096usize, 4usize, 64usize, (0u32, 0u32)), (16383, 1023, 1, 4096, 4, 64, (4, 4)),
        (16384, 1, 1, 4096, 512, 64, (2, 2)), (1, 1024, 1, 4096, 4, 3, (1, 1)), (16384, 1024, 1, 4, 4, 64, (3, 3)), (16384, 1024, 32, 256, 256, 64, (0, 0)),
        (16383, 1024, 64, 256, 256, 32, (4, 4)), (4097, 1023, 32, 256, 256, 2051, (0, 4)), (16384, 129, 128, 4, 4, 16, (2, 2))] {
        let (q, k) = fixture(&mut g, tokens, rows, heads, dim, taps, kind, (rows + tokens + dim) as u64)?;
        let c = Case { rows, tokens, heads, dim, taps, slots };
        bytes += compare(&mut g, fns, &c, &q, &k)?; comparisons += 1;
        g.clear()?;
        println!("SCORESTILE end rows={rows} tokens={tokens} heads={heads} dim={dim} taps={taps} slots={slots} patterns={kind:?} equal elapsed_s={:.1}", started.elapsed().as_secs_f64());
    }
    println!("SCORESTILE BYTES PASS comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    Ok(())
}
