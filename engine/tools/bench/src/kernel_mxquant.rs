//! MX-QUANT device screen (exact change): the quantize pass before every MXFP8 prompt projection
//! (`Op::Mxfp8Quantize`, k = token rows * cols words in one launch; `dynamic_projection::project_rows`).
//! The baseline image runs it on the generic entry (`causal_operation`: one thread a word; each word reads its whole
//! 32-word group and searches the scale again), because `fast_admitted` listed column counts up to 16,384 only. The
//! candidate image admits token rows to the warp version (`causal_fast`: one warp a group, one maximum, the unchanged
//! `mxfp8_quantize::finish`).
//! Run: kernel-bench --mx-quant <baseline image> <candidate image> [bytes|time|time-warp|all].
//! Bytes: every word of the candidate's warp launch = the baseline's generic launch, from poisoned outputs, with guards:
//! both zero policies, planes 0 to 2, five input kinds (2 = FP32 subnormals and signed zeros, 4 = NaN and infinities),
//! at both ends of the token-row family (20,480 and 12,582,912 words), at 1,024 rows of each width and at odd row counts.
//! Time: 7 alternating repeats of 5 launches, median, at the read's shapes. Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn step(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.step() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn words(&self, g: &Gpu) -> Result<Vec<u32>> {
        let x = g.read(self.base, self.bytes + 32)?;
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXQUANT output guard changed".into()); }
        Ok(x[16..x.len() - 16].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    fn poison(&self, g: &Gpu, byte: u8) -> Result<()> { g.upload_at(self.address, &vec![byte; self.bytes]) }
}
/// Input kinds: 0 plain, 1 wide range, 2 FP32 subnormals and signed zeros, 3 large, 4 wide range with NaN, infinities and
/// the largest finite value (a group that holds one gives NaN in all 32 lanes on both entries).
fn input(kind: u32, n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| { let (u, s) = (r.unit(), r.step()); match kind {
        0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
        _ => match i % 4099 { 7 => f32::NAN, 11 => f32::INFINITY, 13 => f32::NEG_INFINITY, 17 => f32::from_bits(0x7f7f_ffff), _ => u * f32::from_bits(((s % 60 + 97) as u32) << 23) } } })
        .flat_map(|v| v.to_le_bytes()).collect()
}
/// `causal_ops::Arguments` (80 bytes): four inputs, weight, output, then the shape (op, heads, dim, taps, rows = plane,
/// cols = words, epsilon, bound). An absent owner is the engine's placeholder 4.
fn arguments(op: u32, plane: u32, k: usize, source: u64, output: u64) -> Vec<u8> {
    let mut v = vec![]; for x in [source, 4, 4, 4, 4, output] { v.extend(x.to_le_bytes()); }
    for x in [op, 0, 0, 0, plane, k as u32] { v.extend(x.to_le_bytes()); } v.extend([0u8; 8]); v
}
/// `warp` false: the generic geometry (`causal_ops::geometry` default, one thread a word). true: one warp a group, four a block.
fn launch(g: &Gpu, f: H, warp: bool, op: u32, plane: u32, k: usize, source: u64, output: u64) -> Result<()> {
    let mut a = arguments(op, plane, k, source, output);
    let blocks = if warp { (k / 32).div_ceil(4) } else { k.div_ceil(128) } as u32;
    g.launch(f, [blocks, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
const QUANTIZE: u32 = 18; const QUANTIZE_POSITIVE_ZERO: u32 = 27;
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXQUANT baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let (generic, warp) = (g.function(ma, "causal_operation")?, g.function(mb, "causal_fast")?);
    // The baseline's own warp entry: what main launches for single rows and (since pull request 56) for token rows.
    let base_warp = g.function(ma, "causal_fast")?;
    let started = Instant::now(); let mut fails = 0usize;
    if mode == "bytes" || mode == "all" {
        // The drafting and many-chat shapes (5 to 8 rows of 4,096, 3 to 8 of 8,192, 2 to 8 of 12,288), then the prompt groups.
        for (cols, tokens) in [(4096usize, 5usize), (4096, 6), (4096, 7), (4096, 8), (8192, 3), (8192, 4), (8192, 5), (8192, 6), (8192, 7), (8192, 8), (12288, 2), (12288, 3), (12288, 4), (12288, 5), (12288, 6), (12288, 7), (12288, 8),
            (4096, 257), (4096, 1024), (8192, 1023), (8192, 1024), (12288, 511), (12288, 1024)] {
            let k = cols * tokens; let (mut words, mut differ, mut launches) = (0usize, 0usize, 0usize);
            for kind in 0..5u32 {
                let source = Buf::new(&mut g, &input(kind, k, 77 + u64::from(kind) + k as u64))?;
                let (x, y) = (Buf::new(&mut g, &vec![0xff; k * 4])?, Buf::new(&mut g, &vec![0xee; k * 4])?);
                for op in [QUANTIZE, QUANTIZE_POSITIVE_ZERO] { for plane in 0..3u32 {
                    // Every policy and plane on the plain kind and at 1,024 rows; the prompt path's own launch (plane 0) on every kind.
                    if !(kind == 0 || tokens == 1024 || plane == 0) { continue; }
                    x.poison(&g, 0xff)?; y.poison(&g, 0xee)?;
                    launch(&g, generic, false, op, plane, k, source.address, x.address)?; launch(&g, warp, true, op, plane, k, source.address, y.address)?; g.sync()?;
                    let (wx, wy) = (x.words(&g)?, y.words(&g)?);
                    words += wx.len(); launches += 1; differ += wx.iter().zip(&wy).filter(|(p, q)| p != q).count();
                } }
                g.clear()?;
            }
            println!("MXQUANT BYTES cols={cols} tokens={tokens} words={k} launches_a_side={launches} compared_words={words} differing={differ}");
            if differ != 0 { fails += 1; }
        }
        // Warp entry against warp entry (a change of `mxfp8_quantize::finish` itself): the single-row counts of the old list
        // and token rows, every policy, plane and input kind. An image that holds no token-row admission is skipped by its trap.
        for (cols, tokens) in [(32usize, 1usize), (64, 1), (160, 1), (1024, 1), (4096, 1), (8192, 1), (12288, 1), (4096, 4), (4096, 5), (8192, 3), (4096, 257), (4096, 1024), (8192, 1024), (12288, 1024)] {
            let k = cols * tokens; let (mut words, mut differ, mut launches) = (0usize, 0usize, 0usize);
            for kind in 0..5u32 {
                let source = Buf::new(&mut g, &input(kind, k, 177 + u64::from(kind) + k as u64))?;
                let (x, y) = (Buf::new(&mut g, &vec![0xff; k * 4])?, Buf::new(&mut g, &vec![0xee; k * 4])?);
                for op in [QUANTIZE, QUANTIZE_POSITIVE_ZERO] { for plane in 0..3u32 {
                    x.poison(&g, 0xff)?; y.poison(&g, 0xee)?;
                    launch(&g, base_warp, true, op, plane, k, source.address, x.address)?; launch(&g, warp, true, op, plane, k, source.address, y.address)?; g.sync()?;
                    let (wx, wy) = (x.words(&g)?, y.words(&g)?);
                    words += wx.len(); launches += 1; differ += wx.iter().zip(&wy).filter(|(p, q)| p != q).count();
                } }
                g.clear()?;
            }
            println!("MXQUANT WARP BYTES cols={cols} tokens={tokens} words={k} launches_a_side={launches} compared_words={words} differing={differ}");
            if differ != 0 { fails += 1; }
        }
        println!("MXQUANT BYTES {} fails={fails} elapsed_s={:.1}", if fails == 0 { "PASS" } else { "FAIL" }, started.elapsed().as_secs_f64());
    }
    if mode == "time" || mode == "time-warp" || mode == "all" {
        for (cols, tokens) in [(4096usize, 1024usize), (8192, 1024), (12288, 1024), (4096, 256), (4096, 16), (4096, 1), (8192, 1)] {
            let k = cols * tokens;
            let source = Buf::new(&mut g, &input(0, k, 5))?; let out = Buf::new(&mut g, &vec![0xff; k * 4])?;
            let (mut ta, mut tb) = (vec![], vec![]);
            for repeat in 0..7 {
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    // Mode time-warp: the baseline arm is the baseline's own warp entry (a change of `finish` itself).
                    let go = |g: &Gpu| if arm == 0 { launch(g, if mode == "time-warp" { base_warp } else { generic }, mode == "time-warp", QUANTIZE, 0, k, source.address, out.address) } else { launch(g, warp, true, QUANTIZE, 0, k, source.address, out.address) };
                    go(&g)?; g.sync()?;
                    let n = 5; let t0 = Instant::now();
                    for _ in 0..n { go(&g)?; }
                    g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let spread = |v: &[f64]| { let (lo, hi) = v.iter().fold((f64::MAX, 0f64), |(l, h), x| (l.min(*x), h.max(*x))); hi - lo };
            let (ra, rb) = (median(ta.clone()), median(tb.clone()));
            println!("MXQUANT TIME cols={cols} tokens={tokens} generic_us={ra:.1} warp_us={rb:.1} ratio={:.2} generic_spread_us={:.1} warp_spread_us={:.1}", ra / rb, spread(&ta), spread(&tb));
            g.clear()?;
        }
    }
    println!("MXQUANT DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    if fails != 0 { return Err(format!("MXQUANT bytes failed: {fails}")); }
    Ok(())
}
