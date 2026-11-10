//! Lane P device screen for `transpose_smem` (32 x 32 shared-memory tile) against the generic `prompt_rows` Transpose unit and the older
//! `transpose_tiles` (4 x 4 tiles): a byte-for-byte comparison of EVERY admitted (rows, cols) pair (`swept::transpose_pairs()`, with and without the
//! finite check), each in buffers of exactly its size with a guard before and after (so an overrun by one word shows), then a self-test that the guard
//! check catches a deliberate overrun, then timing at the real exchange shapes, in one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
#[allow(dead_code, clippy::duplicate_mod)] #[path = "../../../crates/gpu/src/swept.rs"] mod swept;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op Transpose is 2.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
fn le(values: &[u32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// Words are copied, so the pattern is any bits: a counter hash, all finite when `finite` (the exchange check traps on a non-finite word with `first`).
fn pattern(n: usize, seed: u64, finite: bool) -> Vec<u32> {
    let mut x = seed | 1;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let w = (x >> 16) as u32; if finite { w & 0xbfff_ffff | 0x3000_0000 & !0x4000_0000 } else { w } }).collect()
}
struct Buffer { base: u64, address: u64, bytes: usize }
impl Buffer {
    fn new(g: &mut Gpu, words: &[u32]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), &le(words), GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: words.len() * 4 })
    }
    /// Data plus both guards.
    fn read_all(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    /// Both 16-byte guards, the one before the data and the one after it.
    fn guards_ok(&self, g: &Gpu) -> Result<bool> { let all = self.read_all(g)?; Ok(all[..16] == GUARD && all[all.len() - 16..] == GUARD) }
}
fn args(input: &Buffer, output: &Buffer, rows: usize, cols: usize, first: u32) -> Arguments {
    let inner = Inner { op: 5, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    Arguments { input: [input.address, 0, 0, 0, 0], lengths: [(rows * cols) as u64, 0, 0, 0, 0], weight: 4, weights: 0, output: output.address,
        units: (rows * cols) as u64, words: 1, shape: PShape { op: 2, tokens: 1, heads: 0, dim: 0, taps: 0, rows: rows as u32, cols: cols as u32, first, slots: 0, inner } }
}
fn launch_reference(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }
fn launch_tiles(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { let mut b = *a; b.units = a.units / 16; g.launch(f, [(b.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut b)]) }
fn launch_smem(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [a.shape.cols.div_ceil(32), a.shape.rows.div_ceil(32), 1], 256, 0, &mut [ptr(a)]) }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
/// One shape: input, reference output and candidate output in buffers of exactly their size (poisoned outputs, guards on both sides of all three).
/// Returns the launch of the candidate's output bytes for the caller's own checks. `candidate` picks the kernel under test.
fn compare(g: &mut Gpu, input_all: &[u32], rows: usize, cols: usize, first: u32, reference: H, candidate: &dyn Fn(&Gpu, &mut Arguments) -> Result<()>) -> Result<()> {
    let words = rows * cols;
    let input = Buffer::new(g, &input_all[..words])?;
    let out_a = Buffer::new(g, &vec![0xffff_ffff; words])?; let out_b = Buffer::new(g, &vec![0xffff_ffff; words])?;
    let (mut xa, mut xb) = (args(&input, &out_a, rows, cols, first), args(&input, &out_b, rows, cols, first));
    launch_reference(g, reference, &mut xa)?; candidate(g, &mut xb)?; g.sync()?;
    let (x, y) = (out_a.read_all(g)?, out_b.read_all(g)?);
    if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q); return Err(format!("TRANSP BYTE FAIL rows={rows} cols={cols} first={first} first_diff_byte={at:?}")); }
    if !out_a.guards_ok(g)? || !out_b.guards_ok(g)? || !input.guards_ok(g)? { return Err(format!("TRANSP guard changed rows={rows} cols={cols} first={first}")); }
    g.clear()
}
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("TRANSP baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "prompt_rows")?; let tiles = g.function(ma, "transpose_tiles")?; let smem = g.function(mb, "transpose_smem")?;
    let started = Instant::now();
    let pairs = swept::transpose_pairs();
    let largest = pairs.iter().map(|p| p.0 * p.1).max().unwrap_or(0);
    let input_all = pattern(largest, 0x9e37_79b9, true);
    let candidate_smem = |g: &Gpu, a: &mut Arguments| launch_smem(g, smem, a);
    // 0. The guard check must catch an overrun, or the whole sweep proves nothing: give the kernel an output 32 bytes too small and require the trailing
    // guard (16 bytes) to show it. A check that stays green here fails the run.
    {
        let (rows, cols) = (64usize, 64usize); let words = rows * cols;
        let input = Buffer::new(&mut g, &input_all[..words])?;
        let short = Buffer::new(&mut g, &vec![0xffff_ffffu32; words - 8])?;
        let mut x = args(&input, &short, rows, cols, 0);
        launch_smem(&g, smem, &mut x)?; g.sync()?;
        if short.guards_ok(&g)? { return Err("TRANSP SELF-TEST FAIL: a 32-byte overrun left both guards intact".into()); }
        g.clear()?;
        println!("TRANSP self-test ok: a deliberate 32-byte overrun is caught by the trailing guard");
    }
    // 1. Every admitted pair, with and without the finite check, byte for byte against the generic kernel.
    let (mut comparisons, mut bytes) = (0u64, 0u64);
    for &(rows, cols) in &pairs {
        for first in [0u32, 1] {
            compare(&mut g, &input_all, rows, cols, first, reference, &candidate_smem)?;
            comparisons += 1; bytes += (rows * cols * 4) as u64;
        }
    }
    println!("TRANSP every admitted pair ok pairs={} comparisons={comparisons} elapsed_s={:.1}", pairs.len(), started.elapsed().as_secs_f64());
    // 2. The older 4 x 4 tile kernel is the same bits as the generic one on the exchange shapes (so the smem kernel replaces equal output).
    for (rows, cols) in [(32768usize, 256usize), (16384, 256), (4096, 36 / 4 * 4), (1536, 68)] {
        compare(&mut g, &input_all, rows, cols, 0, reference, &|g: &Gpu, a: &mut Arguments| launch_tiles(g, tiles, a))?;
        comparisons += 1; bytes += (rows * cols * 4) as u64;
    }
    println!("TRANSP BYTES PASS pairs={} comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", pairs.len(), sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    // 3. Timing at the exchange shapes of the model.
    let input = Buffer::new(&mut g, &input_all)?; let out = Buffer::new(&mut g, &vec![0; largest])?;
    for (rows, cols) in [(32768usize, 256usize), (16384, 256), (8192, 256), (4096, 256), (1024, 256), (32768, 128)] {
        let (mut t0, mut t1, mut t2) = (vec![], vec![], vec![]);
        for repeat in 0..7 {
            for arm in if repeat % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                let run = |g: &Gpu| -> Result<()> { let mut x = args(&input, &out, rows, cols, 1); match arm { 0 => launch_reference(g, reference, &mut x), 1 => launch_tiles(g, tiles, &mut x), _ => launch_smem(g, smem, &mut x) } };
                run(&g)?; g.sync()?;
                let n = 20; let t = Instant::now(); for _ in 0..n { run(&g)?; } g.sync()?;
                let us = t.elapsed().as_secs_f64() * 1e6 / n as f64; match arm { 0 => t0.push(us), 1 => t1.push(us), _ => t2.push(us) }
            }
        }
        let (a, t, s) = (median(t0), median(t1), median(t2)); let mb = (rows * cols * 8) as f64 / 1e6;
        println!("TRANSP TIME rows={rows} cols={cols} generic_us={a:.1} tiles4x4_us={t:.1} smem_us={s:.1} smem_GBs={:.0} speedup_vs_tiles={:.2}", mb / s * 1e3, t / s);
    }
    println!("TRANSP PASS");
    Ok(())
}
