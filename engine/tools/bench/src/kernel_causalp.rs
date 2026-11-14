//! Lane P device screen for the single-token causal operations (`causal_operation`, `resident_launch_causal`): same-bytes
//! sweep and timing of the candidate image against the baseline image, in one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
#[allow(dead_code, clippy::duplicate_mod)] #[path = "../../../crates/gpu/src/swept.rs"] mod swept;
const POOL_CHANNELS: u32 = 25;
#[path = "kernel_stepdirect.rs"] mod stepdirect;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// 0 plain, 1 wide range, 2 denormals and signed zeros, 3 large finite, 4 overflow with NaN and inf, 5 huge/tiny alternation,
/// 6 add-order sensitive (2^24 steps, cancelling pairs), 7 all ones, 8 mixed sign cancellation.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.next();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(60),
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i % 4099 == 7 { f32::NAN } else if i % 4099 == 11 { f32::INFINITY } else if i % 4099 == 13 { f32::NEG_INFINITY } else { v } }
            5 => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
            6 => match i % 4 { 0 => 16777216.0, 1 => 1.0 + (s % 3) as f32, 2 => -16777216.0, _ => 1.0 },
            7 => 1.0,
            9 => f32::from_bits([0x7fc1_2345,0xffc2_3456,0x7f81_2345,0x7f80_0000,0xff80_0000,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_0000][i%10]),
            _ => if i % 2 == 0 { 1.0 + u * 1e-6 } else { -1.0 + u * 1e-6 },
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

/// Mirror of gpu/src/causal_ops.rs `Arguments` and `Shape` (repr(C), 80 bytes). Op numbers follow the enum order.
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, output: u64, shape: Shape }
const PREPARE: u32 = 1; const GATED_NORM: u32 = 3; const POOL_SCORES: u32 = 8; const STEP_COLUMNS: u32 = 20; const STEP_LAYOUT: u32 = 21;
const STEP_PLANAR: u32 = 34; const CONV_PARTS: u32 = 35; const CONV: u32 = 0; const MXFP8: u32 = 18; const MXFP8_POSITIVE_ZERO: u32 = 27; const STEP: u32 = 2; const GATHER: u32 = 10; const SWIGLU: u32 = 4; const CONV_CHANNELS: u32 = 22; const CONV_LAYOUT: u32 = 23; const ROUTE: u32 = 14; const EXPERT_FOLD: u32 = 15; const KEY_NORM: u32 = 6; const SOFTMAX: u32 = 12; const UNPACK_KV: u32 = 19;
/// (input lengths, weight length, units, words) as `Shape::lengths` gives them for these operations.
fn lengths(s: Shape) -> ([usize; 4], usize, usize, usize) {
    let (h, d, r) = (s.heads as usize, s.dim as usize, s.rows as usize); let w = h * d;
    match s.op {
        PREPARE => ([3 * w, w, h, 0], h + w, h, 4 * d + 1),
        GATED_NORM => ([w, w, 0, 0], d, h, d),
        POOL_SCORES => ([h * d, r * d, h, 0], 0, r.max(1), 1),
        STEP_COLUMNS => ([h * (4 * d + 1), w * d, 0, 0], 0, w, d + 1),
        STEP => ([h * (4 * d + 1), w * d, 0, 0], 0, h, d * (d + 1)),
        STEP_LAYOUT => ([w * (d + 1), 0, 0, 0], 0, w * (d + 1), 1),
        SOFTMAX => ([h * r, 1, 0, 0], 0, h, r),
        SWIGLU => ([r, r, 0, 0], 0, r, 1),
        MXFP8 | MXFP8_POSITIVE_ZERO => ([s.cols as usize, 0, 0, 0], 0, s.cols as usize, 1),
        CONV => ([3 * w, 3 * (s.taps as usize - 1) * w, 0, 0], 3 * w * s.taps as usize, 3, w * s.taps as usize),
        CONV_CHANNELS => ([3 * w, 3 * (s.taps as usize - 1) * w, 0, 0], 3 * w * s.taps as usize, 3 * w, s.taps as usize),
        CONV_LAYOUT => ([3 * w * s.taps as usize, 0, 0, 0], 0, 3 * w * s.taps as usize, 1),
        ROUTE => ([r, 0, 0, 0], r, 1, 2 * s.cols as usize + 2 * r + 2 * h),
        EXPERT_FOLD => ([r, r, r * s.cols as usize, if s.taps != 0 { s.cols as usize } else { 0 }], 0, s.cols as usize, 1),
        KEY_NORM => ([d, 0, 0, 0], 2 * d, 1, d),
        POOL_CHANNELS => ([(r - 1) * d, (r - 1) * d, d, d], s.taps as usize * d, d, 2 * r + r / s.taps as usize),
        GATHER => ([r * d, s.cols as usize, 1, 0], 0, s.cols as usize, d),
        UNPACK_KV => ([s.taps as usize * h * (d + s.cols as usize), 0, 0, 0], 0, r * h * (d + s.cols as usize), 1),
        _ => unreachable!(),
    }
}
/// Mirror of gpu/src/causal_ops.rs `fast_admitted` and `geometry` (tested there). Every admitted head and dim value is a swept one.
fn admitted(s: Shape) -> bool {
    let (h, d, r, k) = (s.heads as usize, s.dim as usize, s.rows as usize, s.cols as usize);
    match s.op {
        PREPARE | GATED_NORM => matches!(h, 1 | 2 | 3 | 4 | 5 | 31 | 32 | 33 | 63 | 64 | 65 | 100 | 127 | 128) && matches!(d, 1 | 2 | 5 | 31 | 32 | 33 | 64 | 100 | 127 | 128 | 129 | 255 | 1023 | 1024) && h * d <= 140000 && (h < 100 || matches!(d, 1 | 128 | 1024)),
        STEP_COLUMNS | STEP_LAYOUT => d == 128 && matches!(h, 1 | 2 | 3 | 17 | 63 | 64),
        CONV => matches!(h, 1 | 2 | 3 | 8 | 17 | 63 | 64 | 65) && matches!(d, 1 | 3 | 32 | 127 | 128 | 129) && matches!(s.taps, 2..=5),
        MXFP8 | MXFP8_POSITIVE_ZERO => r <= 2 && (matches!(k, 32 | 64 | 96 | 128 | 160 | 1024 | 2048 | 4096 | 8192 | 12288 | 16384) || k % 4096 == 0 && k > 16384 && k <= 1024 * 12288),
        KEY_NORM => matches!(d, 1 | 2 | 5 | 31 | 32 | 33 | 64 | 100 | 127 | 128 | 129 | 255 | 1023 | 1024),
        POOL_SCORES => matches!(d, 1 | 2 | 31 | 32 | 33 | 127 | 128 | 255 | 256) && matches!(h, 1 | 2 | 3 | 31 | 32 | 33 | 63 | 64 | 65 | 100 | 128) && r <= 4096,
        SOFTMAX => matches!(h, 1 | 2 | 3 | 32 | 33 | 64 | 65 | 100 | 128) && (1..=4096).contains(&r),
        GATHER => matches!(d, 1 | 3 | 32 | 33 | 512 | 1024) && (1..=16384).contains(&r) && (1..=4096).contains(&k),
        POOL_CHANNELS => swept::POOL_CHANNEL_DIMS.contains(&d) && swept::POOL_CHANNEL_TAPS.contains(&(s.taps as usize)) && (1..=swept::POOL_CHANNEL_ROWS_MAX).contains(&r),
        _ => false,
    }
}
fn fast_geometry(s: Shape, units: usize) -> (u32, u32) {
    if !admitted(s) { return (units.div_ceil(128) as u32, 128); }
    match s.op { PREPARE | GATED_NORM | POOL_CHANNELS => (units.div_ceil(4) as u32, 128), POOL_SCORES | GATHER => (units.div_ceil(4).next_multiple_of(64) as u32, 128), STEP_COLUMNS | SOFTMAX => (s.heads, 128), CONV => ((3 * s.heads as usize * s.dim as usize).div_ceil(128) as u32, 128), KEY_NORM => (1, 128), MXFP8 | MXFP8_POSITIVE_ZERO => ((s.cols as usize / 32).div_ceil(4) as u32, 128), _ => (s.heads * 17, 128) }
}
fn launch_causal(g: &Gpu, f: H, a: &mut Arguments, geometry: (u32, u32)) -> Result<()> { g.launch(f, [geometry.0, 1, 1], geometry.1, 0, &mut [ptr(a)]) }
fn launch_replay(g: &Gpu, f: H, table: u64, geometry: (u32, u32)) -> Result<()> { let (mut t, mut i) = (table, 0u64); g.launch(f, [geometry.0, 1, 1], geometry.1, 0, &mut [ptr(&mut t), ptr(&mut i)]) }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
/// Table entry for `resident_launch_causal` (kind 1): 1040 bytes, payload = the 80-byte Arguments.
fn command(a: &Arguments, kind: u32) -> Vec<u8> {
    let mut c = vec![0u8; 1040]; c[..4].copy_from_slice(&kind.to_le_bytes());
    let mut p = vec![]; for w in a.input { p.extend(w.to_le_bytes()); } p.extend(a.weight.to_le_bytes()); p.extend(a.output.to_le_bytes());
    let s = a.shape; for w in [s.op, s.heads, s.dim, s.taps, s.rows, s.cols] { p.extend(w.to_le_bytes()); } p.extend(s.epsilon.to_le_bytes()); p.extend(s.bound.to_le_bytes());
    assert_eq!(p.len(), 80); c[16..96].copy_from_slice(&p); c
}
struct Fixture { inputs: Vec<Buffer>, weight: Buffer, shape: Shape, units: usize, words: usize }
fn fixture(g: &mut Gpu, s: Shape, ik: u32, wk: u32, seed: u64) -> Result<Fixture> { fixture_count(g, s, ik, wk, seed, 0) }
/// `count`: the live count word of Softmax / Gather; Gather ids are drawn from the rows.
fn fixture_count(g: &mut Gpu, s: Shape, ik: u32, wk: u32, seed: u64, count: usize) -> Result<Fixture> {
    let (lens, weight, units, words) = lengths(s);
    let mut inputs = vec![];
    for (i, n) in lens.iter().enumerate() { inputs.push(Buffer::new(g, &pattern(if i == 0 { ik } else { (ik + i as u32) % 9 }, (*n).max(1), seed * 7 + i as u64))?); }
    if s.op == SOFTMAX { inputs[1] = Buffer::new(g, &[count as f32])?; }
    if s.op == GATHER {
        let mut rng = Rng(seed | 1); let ids: Vec<f32> = (0..lens[1].max(1)).map(|_| (rng.next() % s.rows.max(1) as u64) as f32).collect();
        inputs[1] = Buffer::new(g, &ids)?; inputs[2] = Buffer::new(g, &[count as f32])?;
    }
    let weight = Buffer::new(g, &pattern(wk, weight.max(1), seed * 13 + 5))?;
    Ok(Fixture { inputs, weight, shape: s, units, words })
}
fn arguments(f: &Fixture, out: &Buffer) -> Arguments {
    let mut input = [4u64; 4]; for (i, b) in f.inputs.iter().enumerate() { input[i] = b.address; }
    Arguments { input, weight: f.weight.address, output: out.address, shape: f.shape }
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("CAUSALP baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let (base_direct, cand_direct, cand_fast_direct) = (g.function(ma, "causal_operation")?, g.function(mb, "causal_operation")?, g.function(mb, "causal_fast")?);
    let (base_replay, cand_replay, cand_fast_replay) = (g.function(ma, "resident_launch_causal")?, g.function(mb, "resident_launch_causal")?, g.function(mb, "resident_launch_causal_fast")?);
    let started = Instant::now();
    if mode == "step-direct" {return stepdirect::run(&mut g,ma,mb);}
    if mode == "select-bytes" || mode == "select-time" {
        let (base_sel, cand_sel) = (g.function(ma, "prompt_rows")?, g.function(mb, "prompt_select_rows")?);
        select_screen(&mut g, base_sel, cand_sel, mode == "select-time")?;
        println!("CAUSALP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64()); return Ok(());
    }
    if mode == "route" {
        route_screen(&mut g, base_replay, cand_replay, cand_direct, started)?;
        println!("CAUSALP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64()); return Ok(());
    }
    if mode == "planar" {
        let base_fast_replay = g.function(ma, "resident_launch_causal_fast")?;
        planar_screen(&mut g, base_direct, cand_fast_direct, cand_fast_replay, base_replay, base_fast_replay, started)?;
        println!("CAUSALP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64()); return Ok(());
    }
    if mode == "fold-bytes" || mode == "fold-time" {
        fold_screen(&mut g, ma, mb, mode == "fold-time", started)?;
        println!("CAUSALP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64()); return Ok(());
    }
    let shape = |op: u32, heads: usize, dim: usize, rows: usize| Shape { op, heads: heads as u32, dim: dim as u32, taps: 0, rows: rows as u32, cols: 0, epsilon: if op == GATED_NORM { 1e-6 } else { 0.0 }, bound: if op == PREPARE { -5.0 } else { 0.0 } };
    if mode == "bytes" || mode == "bytes-b" || mode == "bytes-c" || mode == "bytes-d" || mode == "bytes-e" || mode == "bytes-f" || mode == "bytes-k" {
        let mut specs: Vec<(Shape, usize)> = vec![];
        let hs = [1usize, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129]; let ds = [1usize, 2, 5, 31, 32, 33, 64, 100, 127, 128, 129, 255, 1023, 1024];
        for op in [PREPARE, GATED_NORM] { for &h in &hs { for &d in &ds { if h * d <= 140000 && (h < 100 || [1usize, 128, 1024].contains(&d)) { specs.push((shape(op, h, d, 0), 0)); } } } }
        for &h in &[1usize, 2, 3, 31, 32, 33, 63, 64, 65, 100, 128, 129] { for &d in &[1usize, 2, 31, 32, 33, 127, 128, 255, 256, 257] { for &r in &[0usize, 1, 2, 3, 4, 5, 127, 128, 129, 1040, 4095, 4096, 4097] { specs.push((shape(POOL_SCORES, h, d, r), 0)); } } }
        for op in [STEP_COLUMNS, STEP_LAYOUT] { for &h in &[1usize, 2, 3, 17, 63, 64, 65] { specs.push((shape(op, h, 128, 0), 0)); } for &d in &[1usize, 2, 31, 64, 127, 129] { specs.push((shape(op, 3, d, 0), 0)); } }
        // The review's examples outside the envelope (they keep the original path).
        specs.push((shape(POOL_SCORES, 1, 4096, 1), 0)); specs.push((shape(SOFTMAX, 1, 128, 65536), 65536)); { let mut g = shape(GATHER, 0, 4096, 1); g.cols = 1; specs.push((g, 1)); }
        for &r in &[1usize, 7, 128, 129, 4096] { let mut s = shape(SWIGLU, 0, 0, r); s.bound = 2.0; specs.push((s, 0)); }
        // KeyNorm: every swept dim, and values just outside the list (generic path); epsilon variants.
        for &d in &[1usize, 2, 3, 4, 5, 31, 32, 33, 34, 64, 100, 127, 128, 129, 130, 255, 256, 1023, 1024, 1025] { for &eps in &[1e-6f32, 1e-3, 0.5] { let mut s = shape(KEY_NORM, 0, d, 0); s.epsilon = eps; specs.push((s, 0)); } }
        // MXFP8 quantize, both policies, planes 0 to 2: the admitted column counts, and multiples of 32 just outside the list (generic path).
        // 32,768 is 8 token rows of 4,096 (admitted since the token-row family; its large counts are swept by --mx-quant).
        for op in [MXFP8, MXFP8_POSITIVE_ZERO] { for &k in &[32usize, 64, 96, 128, 160, 192, 224, 1024, 2048, 2080, 4096, 4128, 8160, 8192, 8224, 12288, 16384, 16416, 20000, 32768] { for plane in 0..=2u32 { let mut s = shape(op, 0, 0, plane as usize); s.cols = k as u32; specs.push((s, 0)); } } }
        // Conv: the admitted heads, dims and taps in every combination (small enough to run all patterns), and neighbours that keep the generic launch.
        for &h in &[1usize, 2, 3, 4, 8, 17, 63, 64, 65, 66] { for &d in &[1usize, 3, 32, 100, 127, 128, 129] { for &t in &[1u32, 2, 3, 4, 5, 6] { if h * d <= 66 * 129 { let mut s = shape(CONV, h, d, 0); s.taps = t; specs.push((s, 0)); } } } }
        for &(h, d, t) in &[(1usize, 1usize, 1u32), (2, 3, 4), (8, 128, 4), (64, 128, 4)] { for op in [CONV_CHANNELS, CONV_LAYOUT] { let mut s = shape(op, h, d, 0); s.taps = t; specs.push((s, 0)); } }
        for &r in &[1usize, 2, 8] { for &k in &[1usize, 5, 4096] { for &taps in &[0u32, 1] { let mut s = shape(EXPERT_FOLD, 0, 0, r); s.cols = k as u32; s.taps = taps; specs.push((s, 0)); } } }
        for &(h, d, k, t) in &[(2usize, 3usize, 5u32, 1usize), (2, 3, 5, 3), (32, 512, 512, 64)] { let mut s = shape(UNPACK_KV, h, d, t); s.taps = t as u32; s.cols = k; specs.push((s, 0)); }
        // Epsilon and bound variants.
        for &(eps, bound) in &[(1e-6f32, -5.0f32), (1e-3, -0.001), (0.5, -100.0)] { for &(h, d) in &[(64usize, 128usize), (3, 33)] { let mut s = shape(GATED_NORM, h, d, 0); s.epsilon = eps; specs.push((s, 0)); let mut p = shape(PREPARE, h, d, 0); p.bound = bound; specs.push((p, 0)); } }
        for &h in &[1usize, 2, 3, 32, 33, 64, 65, 100, 128, 129] { for &r in &[1usize, 2, 31, 32, 33, 127, 128, 129, 1000, 2051, 4000, 4096, 4097] {
            for &c in &[0usize, 1, 2, r / 2, r.saturating_sub(1), r] { if c <= r { specs.push((shape(SOFTMAX, h, 128, r), c)); } } } }
        for &r in &[1usize, 5, 128, 2051, 5000, 16384, 16385] { for &d in &[1usize, 3, 32, 33, 512, 1024, 1025] { for &k in &[1usize, 2, 127, 128, 129, 2051, 4096, 4097] {
            if r * d > 1_000_000 && ![1usize, 2051, 4097].contains(&k) { continue; }
            let counts: &[usize] = if r * d > 1_000_000 { &[k / 2, k] } else { &[0usize, 1, k / 2, k, k + 5] };
            for &c in counts { let mut s = shape(GATHER, 0, d, r); s.cols = k as u32; specs.push((s, c)); } } } }
        // PoolChannels (READ-SPLIT): the swept lists of swept.rs and nothing else; heads 0 and 64 (not read), values just outside every list keep the original entry.
        for &d in &swept::POOL_CHANNEL_DIMS { for &t in &swept::POOL_CHANNEL_TAPS { for &r in &swept::pool_channel_rows(t) {
            if r * d > 700_000 && ![1usize, 4096, 16384].contains(&r) { continue; }
            specs.push((Shape { op: POOL_CHANNELS, heads: if r % 2 == 0 { 0 } else { 64 }, dim: d as u32, taps: t as u32, rows: r as u32, cols: 0, epsilon: 0.0, bound: 0.0 }, 0)); } } }
        for &(d, t, r) in &[(128usize, 4usize, 16385usize), (130, 4, 100), (128, 6, 100), (127, 4, 4097), (4096, 4, 3), (128, 130, 400), (128, 9, 77)] {
            specs.push((Shape { op: POOL_CHANNELS, heads: 0, dim: d as u32, taps: t as u32, rows: r as u32, cols: 0, epsilon: 0.0, bound: 0.0 }, 0)); }
        let (mut comparisons, mut bytes) = (0u64, 0u64);
        let kinds: [(u32, u32); 8] = [(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (6, 8), (8, 3), (3, 3)];
        let which = match mode { "bytes-b" => 1, "bytes-c" => 2, "bytes-d" => 3, "bytes-e" => 4, "bytes-f" => 5, "bytes-k" => 6, _ => 0 };
        for (n, (s, count)) in specs.iter().enumerate() {
            if which != match s.op { CONV | KEY_NORM | MXFP8 | MXFP8_POSITIVE_ZERO => 6, POOL_CHANNELS => if s.taps < 100 { 4 } else { 5 }, SOFTMAX => 1, GATHER => if s.rows as usize * s.dim as usize > 1_000_000 { 3 } else { 2 }, _ => 0 } { continue; }
            let big = s.heads as usize * s.dim as usize > 20000 && s.op != POOL_SCORES && s.op != SOFTMAX && s.op != GATHER || (s.op == STEP_LAYOUT || s.op == STEP_COLUMNS) && s.heads > 17;
            for (ki, &(ik, wk)) in kinds.iter().enumerate() {
                if big && ki > 3 { continue; }
                if s.op == POOL_CHANNELS && s.rows as usize * s.dim as usize > 100_000 && ki > 1 { continue; }
                if s.op == GATHER && s.rows as usize * s.dim as usize > 1_000_000 && ki > 0 { continue; }
                if s.op == GATHER && s.cols as usize * s.dim as usize > 400_000 && ki > 1 { continue; }
                if s.heads as usize * s.dim as usize > 30000 && s.op != POOL_SCORES && ki > 1 { continue; }
                let fx = fixture_count(&mut g, *s, ik, wk, 100 + n as u64, *count)?;
                let units = fx.units; let words = fx.words;
                let mut outs = vec![];
                // Arms: baseline direct; candidate direct with its geometry; candidate direct with the default geometry;
                // baseline replay; candidate replay (command table), both with the candidate geometry rules.
                for arm in 0..5 {
                    // The baseline arms start from the zero fill the host used to make; the candidate arms from poison (the host now skips the fill).
                    let out = Buffer::new(&mut g, &vec![f32::NAN; units * words])?; if arm == 0 || arm == 3 { g.upload_at(out.address, &vec![0u8; units * words * 4])?; } else { out.poison(&g)?; }
                    let mut args = arguments(&fx, &out);
                    let default = (units.div_ceil(128) as u32, 128); let fast = fast_geometry(*s, units);
                    match arm {
                        0 => launch_causal(&g, base_direct, &mut args, default)?,
                        1 => launch_causal(&g, if admitted(*s) { cand_fast_direct } else { cand_direct }, &mut args, fast)?,
                        2 => launch_causal(&g, if admitted(*s) { cand_fast_direct } else { cand_direct }, &mut args, default)?,
                        3 => { let t = g.upload(&command(&args, 1))?; launch_replay(&g, base_replay, t, default)?; }
                        _ => { let t = g.upload(&command(&args, if admitted(*s) { 19 } else { 1 }))?; launch_replay(&g, if admitted(*s) { cand_fast_replay } else { cand_replay }, t, fast)?; }
                    }
                    g.sync()?; outs.push(out.read(&g)?);
                }
                // Conv: what main launches is the split chain, ConvChannels then ConvLayout, both from the baseline image. Every arm must match its bytes too.
                let chain = if s.op == CONV {
                    let w3 = 3 * s.heads as usize * s.dim as usize; let taps = s.taps as usize;
                    let channels = Buffer::new(&mut g, &vec![f32::NAN; w3 * taps])?; channels.poison(&g)?;
                    let layout = Buffer::new(&mut g, &vec![f32::NAN; w3 * taps])?; layout.poison(&g)?;
                    let mut first = arguments(&fx, &channels); first.shape.op = CONV_CHANNELS;
                    launch_causal(&g, base_direct, &mut first, (w3.div_ceil(128) as u32, 128))?;
                    let mut second = Arguments { input: [channels.address, 4, 4, 4], weight: 4, output: layout.address, shape: Shape { op: CONV_LAYOUT, ..*s } };
                    launch_causal(&g, base_direct, &mut second, ((w3 * taps).div_ceil(128) as u32, 128))?;
                    g.sync()?; Some(layout.read(&g)?)
                } else { None };
                if let Some(c) = &chain {
                    for (arm, y) in outs.iter().enumerate() { if c != y { return Err(format!("CAUSALP CONV CHAIN BYTE FAIL arm={arm} heads={} dim={} taps={} patterns={ik},{wk} first_diff_byte={:?}", s.heads, s.dim, s.taps, c.iter().zip(y).position(|(p, q)| p != q))); } }
                }
                for (arm, y) in outs.iter().enumerate().skip(1) {
                    let reference = if arm == 4 { &outs[3] } else { &outs[0] };
                    if reference != y { return Err(format!("CAUSALP BYTE FAIL arm={arm} op={} heads={} dim={} rows={} eps={} bound={} patterns={ik},{wk} first_diff_byte={:?}", s.op, s.heads, s.dim, s.rows, s.epsilon, s.bound, reference.iter().zip(y).position(|(p, q)| p != q))); }
                    if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("CAUSALP guard changed arm={arm} op={} heads={} dim={}", s.op, s.heads, s.dim)); }
                    comparisons += 1; bytes += (units * words * 4) as u64;
                }
                g.clear()?;
            }
            if n % 50 == 0 { println!("CAUSALP spec {n}/{} ok comparisons={comparisons} elapsed_s={:.1}", specs.len(), started.elapsed().as_secs_f64()); }
        }
        println!("CAUSALP BYTES PASS specs={} comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", specs.len(), sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    }
    if mode == "time" {
        let cases: Vec<(&str, Shape, usize)> = vec![("step_columns 64x128", shape(STEP_COLUMNS, 64, 128, 0), 0), ("step_layout 64x128", shape(STEP_LAYOUT, 64, 128, 0), 0), ("prepare 64x128", shape(PREPARE, 64, 128, 0), 0),
            ("gated_norm 64x128", shape(GATED_NORM, 64, 128, 0), 0), ("pool_scores 64x128 r=16", shape(POOL_SCORES, 64, 128, 16), 0), ("pool_scores 64x128 r=260", shape(POOL_SCORES, 64, 128, 260), 0), ("pool_scores 64x128 r=1040", shape(POOL_SCORES, 64, 128, 1040), 0),
            ("prepare 32x128", shape(PREPARE, 32, 128, 0), 0), ("step_columns 32x128", shape(STEP_COLUMNS, 32, 128, 0), 0),
            ("route 288 experts top 8", { let mut s = shape(ROUTE, 1, 1, 288); s.cols = 8; s.taps = 1; s.bound = 2.5; s }, 0),
            ("expert_fold 8 of 4096", { let mut s = shape(EXPERT_FOLD, 0, 0, 8); s.cols = 4096; s.taps = 1; s }, 0),
            ("mxfp8_quantize k=8192 plane 0 (KDA output, 34 a token)", { let mut s = shape(MXFP8, 0, 0, 0); s.cols = 8192; s }, 0),
            ("mxfp8_quantize k=4096 plane 0 (KDA input, 34 a token)", { let mut s = shape(MXFP8, 0, 0, 0); s.cols = 4096; s }, 0),
            ("mxfp8_quantize_positive_zero k=8192 plane 0", { let mut s = shape(MXFP8_POSITIVE_ZERO, 0, 0, 0); s.cols = 8192; s }, 0),
            ("mxfp8_quantize k=8192 plane 1", { let mut s = shape(MXFP8, 0, 0, 1); s.cols = 8192; s }, 0),
            ("key_norm 128", { let mut s = shape(KEY_NORM, 0, 128, 0); s.epsilon = 1e-6; s }, 0),
            ("key_norm 1024", { let mut s = shape(KEY_NORM, 0, 1024, 0); s.epsilon = 1e-6; s }, 0),
            ("pool_channels 128 ch, 4 taps, 8192 rows", { let mut s = shape(POOL_CHANNELS, 0, 128, 8192); s.taps = 4; s }, 0),
            ("pool_channels 128 ch, 4 taps, 2048 rows", { let mut s = shape(POOL_CHANNELS, 0, 128, 2048); s.taps = 4; s }, 0),
            ("pool_channels 128 ch, 128 taps, 8192 rows", { let mut s = shape(POOL_CHANNELS, 0, 128, 8192); s.taps = 128; s }, 0),
            ("softmax 32 heads x 2051 slots", shape(SOFTMAX, 32, 128, 2051), 2051),
            ("gather 2051 of 4158 rows x 512", { let mut s = shape(GATHER, 0, 512, 4158); s.cols = 2051; s }, 2051),
            ("unpack_kv 2051 slots x 32768 words (not mine; occupancy check)", { let mut s = shape(UNPACK_KV, 32, 512, 2051); s.taps = 2051; s.cols = 512; s }, 0)
        ];
        for (label, s, count) in cases {
            let big = s.op == STEP_COLUMNS || s.op == STEP_LAYOUT; let copies = if s.op == UNPACK_KV { 1 } else if big { 24 } else { 6 };
            let mut sets = vec![];
            for c in 0..copies { let fx = fixture_count(&mut g, s, 0, 0, 900 + c as u64, count)?; let out = Buffer::new(&mut g, &vec![0.0; fx.units * fx.words])?; sets.push((arguments(&fx, &out), fx, out)); }
            let units = sets[0].1.units; let default = (units.div_ceil(128) as u32, 128); let fast = fast_geometry(s, units);
            let tables: Vec<u64> = sets.iter().map(|(a, _, _)| g.upload(&command(a, 1)).unwrap()).collect();
            let fast_tables: Vec<u64> = sets.iter().map(|(a, _, _)| g.upload(&command(a, if admitted(s) { 19 } else { 1 })).unwrap()).collect();
            let mut times = [vec![], vec![]];
            for repeat in 0..9 { for k in 0..2 {
                let arm = (k + repeat) % 2;
                let (f, geo, tb) = if arm == 0 { (base_replay, default, &tables) } else { (if admitted(s) { cand_fast_replay } else { cand_replay }, fast, &fast_tables) };
                launch_replay(&g, f, tb[0], geo)?; g.sync()?;
                let n = 24usize; let t0 = Instant::now();
                for i in 0..n { launch_replay(&g, f, tb[(repeat * 5 + i) % copies], geo)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            } }
            let (tb, tc) = (median(times[0].clone()), median(times[1].clone()));
            println!("CAUSALP TIME {label} base_us={tb:.1} cand_us={tc:.1} speedup={:.2} geometry_base={default:?} geometry_cand={fast:?}", tb / tc);
            g.clear()?;
        }
    }
    if mode == "time" {
        // Conv at the real shape (64 heads x 128, 4 taps): main's two launches (ConvChannels then ConvLayout) against the one direct launch, replay entries, same inputs.
        let s = Shape { taps: 4, ..shape(CONV, 64, 128, 0) };
        let w3 = 3 * 64 * 128usize; let copies = 6usize;
        let mut sets = vec![];
        for c in 0..copies {
            let fx = fixture(&mut g, s, 0, 0, 700 + c as u64)?;
            let direct = Buffer::new(&mut g, &vec![0.0; fx.units * fx.words])?; let channels = Buffer::new(&mut g, &vec![0.0; w3 * 4])?; let layout = Buffer::new(&mut g, &vec![0.0; w3 * 4])?;
            let mut first = arguments(&fx, &channels); first.shape.op = CONV_CHANNELS;
            let second = Arguments { input: [channels.address, 4, 4, 4], weight: 4, output: layout.address, shape: Shape { op: CONV_LAYOUT, ..s } };
            let fast = arguments(&fx, &direct);
            sets.push((g.upload(&command(&first, 1))?, g.upload(&command(&second, 1))?, g.upload(&command(&fast, 19))?, fx, direct, channels, layout));
        }
        let (g1, g2, gf) = ((w3.div_ceil(128) as u32, 128u32), ((w3 * 4).div_ceil(128) as u32, 128u32), fast_geometry(s, 3));
        let mut times = [vec![], vec![]];
        for repeat in 0..9 { for k in 0..2 {
            let arm = (k + repeat) % 2;
            launch_replay(&g, if arm == 0 { base_replay } else { cand_fast_replay }, if arm == 0 { sets[0].0 } else { sets[0].2 }, if arm == 0 { g1 } else { gf })?; g.sync()?;
            let n = 24usize; let t0 = Instant::now();
            for i in 0..n { let set = &sets[(repeat * 5 + i) % copies];
                if arm == 0 { launch_replay(&g, base_replay, set.0, g1)?; launch_replay(&g, base_replay, set.1, g2)?; } else { launch_replay(&g, cand_fast_replay, set.2, gf)?; } }
            g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
        } }
        let (tb, tc) = (median(times[0].clone()), median(times[1].clone()));
        println!("CAUSALP TIME conv 64x128 4 taps, main two launches against one direct launch: base_us={tb:.1} cand_us={tc:.1} speedup={:.2} geometry_cand={gf:?}", tb / tc);
        g.clear()?;
    }
    let _ = (base_direct, STEP);
    println!("CAUSALP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}

/// `causal_ops::FOLD_ROUNDS`: the `dim` mark of the ExpertFold that rounds its row words as it reads them (the round launch
/// inside). Every other `dim` is the fold alone, on the generic entry.
const FOLD_ROUNDS: u32 = u32::MAX;
/// Mirror of gpu/src/mhc_boundary.rs `LaneCast` (repr(C)): the argument of `bf16_lanes_round` and the payload of a round command.
#[repr(C)] #[derive(Clone, Copy)]
struct LaneCast { input: u64, output: u64, words: u32 }
/// Mirror of `fast_admitted` for ExpertFold: the rounding fold only (`dim` FOLD_ROUNDS), 1 to 8 picks, 1 to 8,192 words, with or without the shared row.
fn fold_admitted(s: Shape) -> bool { s.op == EXPERT_FOLD && s.dim == FOLD_ROUNDS && (1..=8).contains(&s.rows) && (1..=8192).contains(&s.cols) && s.taps <= 1 }
/// Row words that are hard for a BF16 round: every tie at the midpoint (low half 0x8000) with an even and an odd high half, the
/// words next to the midpoint, NaN payloads, infinities, signed zeros, subnormals and the largest finite values.
fn fold_hard(n: usize, seed: u64) -> Vec<f32> {
    let fixed = [0x7fc1_2345u32, 0xffc2_3456, 0x7f81_2345, 0x7fff_ffff, 0xffff_ffff, 0x7f80_0000, 0xff80_0000, 0x8000_0000, 0, 1, 0x8000_0001, 0x0080_0000, 0x007f_ffff, 0x7f7f_ffff, 0xff7f_ffff, 0x7f7f_8000, 0x7f7e_8000];
    let mut r = Rng(seed | 1);
    (0..n).map(|i| { let s = r.next();
        f32::from_bits(match i % 4 { 0 => fixed[(s % fixed.len() as u64) as usize], 1 => ((s >> 16) as u32 & 0xffff_0000) | 0x8000, 2 => ((s >> 16) as u32 & 0xffff_0000) | if s & 1 == 0 { 0x7fff } else { 0x8001 }, _ => (s >> 20) as u32 })
    }).collect()
}
/// Pick ids of one case: 0 ascending, 1 and 7 distinct in a mixed order (as a route gives them), 2 descending, 3 with repeats,
/// 4 any floats, 5 with a NaN and an infinity, 6 signed zeros and negative ids.
fn fold_ids(kind: u32, picks: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    match kind {
        0 => (0..picks).map(|i| i as f32).collect(),
        2 => (0..picks).map(|i| (picks - i) as f32 * 7.0).collect(),
        3 => (0..picks).map(|i| (i / 2) as f32).collect(),
        4 => pattern(0, picks, seed),
        5 => (0..picks).map(|i| match i { 0 => f32::INFINITY, 1 => f32::NAN, 2 => f32::NEG_INFINITY, _ => i as f32 }).collect(),
        6 => (0..picks).map(|i| match i % 4 { 0 => 0.0, 1 => -0.0, 2 => -(i as f32), _ => i as f32 }).collect(),
        _ => { let mut ids: Vec<f32> = vec![]; while ids.len() < picks { let id = (r.next() % 288) as f32; if !ids.contains(&id) { ids.push(id); } } ids }
    }
}
/// EXPERT GLUE (fold-bytes, fold-time): the fold of the routed experts, `ExpertFold`, alone and with the round of its rows
/// (`dim` FOLD_ROUNDS: the round launch followed by the fold). Baseline, all from the baseline image: the generic
/// `causal_operation` fold; for the rounding fold the separate round launch (`bf16_lanes_round`) over the row words, then
/// that fold on the rounded words. Candidate, from
/// the candidate image: the entry the engine launches for the op (`causal_fast` when admitted, else its generic entry),
/// direct with its grid, with one block and with spare blocks (the kernel walks its words, so any grid is correct), and as a
/// replay command. Every candidate output must hold the baseline's bytes. Outputs start poisoned: the fold writes every word.
fn fold_screen(g: &mut Gpu, ma: H, mb: H, time: bool, started: Instant) -> Result<()> {
    let (base_fold, base_round, base_replay, base_round_replay) = (g.function(ma, "causal_operation")?, g.function(ma, "bf16_lanes_round")?, g.function(ma, "resident_launch_causal")?, g.function(ma, "resident_launch_round")?);
    let (cand_generic, cand_fast, cand_replay, cand_fast_replay) = (g.function(mb, "causal_operation")?, g.function(mb, "causal_fast")?, g.function(mb, "resident_launch_causal")?, g.function(mb, "resident_launch_causal_fast")?);
    let shape = |rounds: u32, picks: usize, words: usize, shared: u32| Shape { op: EXPERT_FOLD, heads: 0, dim: rounds, taps: shared, rows: picks as u32, cols: words as u32, epsilon: 0.0, bound: 0.0 };
    // One case: its four inputs on the device, and for the rounding fold the rounded rows the baseline's round launch stored.
    struct Case { ids: Buffer, weights: Buffer, rows: Buffer, shared: Buffer, rounded: Option<Buffer> }
    let case = |g: &mut Gpu, s: Shape, kind: u32, seed: u64| -> Result<Case> {
        let (picks, words) = (s.rows as usize, s.cols as usize);
        let rows = match kind { 1 | 5 => fold_hard(picks * words, seed + 2), 0 => pattern(0, picks * words, seed + 2), 2 => pattern(2, picks * words, seed + 2), 3 => pattern(4, picks * words, seed + 2),
            4 => pattern(3, picks * words, seed + 2), 6 => pattern(6, picks * words, seed + 2), _ => pattern(1, picks * words, seed + 2) };
        let weights = pattern([0u32, 0, 1, 0, 3, 4, 6, 8][kind as usize % 8], picks, seed + 1);
        let c = Case { ids: Buffer::new(g, &fold_ids(kind, picks, seed))?, weights: Buffer::new(g, &weights)?, rows: Buffer::new(g, &rows)?,
            shared: Buffer::new(g, &pattern([0u32, 5, 2, 1, 4, 3, 8, 0][kind as usize % 8], words, seed + 3))?, rounded: None };
        if s.dim == 0 { return Ok(c); }
        let rounded = Buffer::new(g, &vec![f32::NAN; picks * words])?; rounded.poison(g)?;
        let mut cast = LaneCast { input: c.rows.address, output: rounded.address, words: (picks * words) as u32 };
        g.launch(base_round, [(picks * words).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut cast)])?; g.sync()?;
        Ok(Case { rounded: Some(rounded), ..c })
    };
    // Baseline arguments: the fold alone, on the rounded rows when the candidate rounds. Candidate arguments: its form, on the rows.
    let base_args = |c: &Case, s: Shape, out: &Buffer| Arguments { input: [c.ids.address, c.weights.address, c.rounded.as_ref().map_or(c.rows.address, |r| r.address), if s.taps != 0 { c.shared.address } else { 4 }],
        weight: 4, output: out.address, shape: Shape { dim: 0, ..s } };
    let cand_args = |c: &Case, s: Shape, out: &Buffer| Arguments { input: [c.ids.address, c.weights.address, c.rows.address, if s.taps != 0 { c.shared.address } else { 4 }], weight: 4, output: out.address, shape: s };
    if !time {
        let (mut specs, mut comparisons, mut bytes, mut fast) = (0u64, 0u64, 0u64, 0u64);
        for rounds in [0, FOLD_ROUNDS] { for picks in 1..=9usize { for &words in &[1usize, 2, 5, 127, 128, 129, 1024, 2048, 4096, 8191, 8192, 8193] { for shared in 0..=1u32 {
            let s = shape(rounds, picks, words, shared); let admitted = fold_admitted(s);
            // The rounding fold has its device version only: outside the admitted shapes the backend refuses it, so nothing runs.
            if rounds != 0 && !admitted { continue; }
            specs += 1;
            for kind in 0..8u32 {
                if words > 4096 && kind > 3 { continue; }
                let c = case(g, s, kind, 4000 + specs * 31 + kind as u64)?;
                let default = (words.div_ceil(128) as u32, 128u32);
                let mut outs = vec![];
                // Arm 0: baseline direct (zero fill first, as the host of the baseline made it). Arm 1: baseline replay command.
                // Arms 2 to 4: candidate direct with its grid, one block, spare blocks. Arm 5: candidate replay command.
                for arm in 0..6 {
                    if !admitted && (arm == 3 || arm == 4) { continue; }
                    let out = Buffer::new(g, &vec![f32::NAN; words])?; if arm < 2 { g.upload_at(out.address, &vec![0u8; words * 4])?; } else { out.poison(g)?; }
                    match arm {
                        0 => launch_causal(g, base_fold, &mut base_args(&c, s, &out), default)?,
                        1 => { let t = g.upload(&command(&base_args(&c, s, &out), 1))?; launch_replay(g, base_replay, t, default)?; }
                        2 => launch_causal(g, if admitted { cand_fast } else { cand_generic }, &mut cand_args(&c, s, &out), default)?,
                        3 => launch_causal(g, cand_fast, &mut cand_args(&c, s, &out), (1, 128))?,
                        4 => launch_causal(g, cand_fast, &mut cand_args(&c, s, &out), (default.0 + 3, 128))?,
                        _ => { let t = g.upload(&command(&cand_args(&c, s, &out), if admitted { 19 } else { 1 }))?; launch_replay(g, if admitted { cand_fast_replay } else { cand_replay }, t, default)?; }
                    }
                    g.sync()?; outs.push((arm, out.read(g)?));
                }
                let reference = outs[0].1.clone();
                for (arm, y) in outs.iter().skip(1) {
                    if &reference != y { return Err(format!("CAUSALP FOLD BYTE FAIL arm={arm} rounds={rounds} picks={picks} words={words} shared={shared} kind={kind} first_diff_byte={:?}", reference.iter().zip(y).position(|(p, q)| p != q).map(|at| at.saturating_sub(16)))); }
                    if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("CAUSALP FOLD guard changed arm={arm} rounds={rounds} picks={picks} words={words}")); }
                    comparisons += 1; bytes += (words * 4) as u64; if *arm >= 2 && admitted { fast += 1; }
                }
                // The inputs must be as they were: the candidate reads the rows and writes only its output.
                for b in [&c.ids, &c.weights, &c.rows, &c.shared] { let now = b.read(g)?; if now[..16] != GUARD || now[now.len() - 16..] != GUARD { return Err(format!("CAUSALP FOLD input guard changed rounds={rounds} picks={picks} words={words}")); } }
                g.clear()?;
            }
        } } } }
        println!("CAUSALP FOLD BYTES PASS specs={specs} comparisons={comparisons} fast_comparisons={fast} output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    // Time: device time of a captured graph of 64 launches (as the engine replays its recorded pieces), between two device
    // stamps, 31 replays an arm, arms interleaved, six fixtures in rotation. us a launch (or a pair of launches) = graph time / 64.
    let tm = g.module(super::TIMER.as_bytes())?; let timer = g.function(tm, "kb_stamp")?;
    const RUN: usize = 64;
    for (label, rounds, picks, words) in [
        ("round of 16384 words, then the fold of 8 picks, 2048 words and the shared row (main's two launches) against one launch of the rounding fold", FOLD_ROUNDS, 8usize, 2048usize),
        ("round of 32768 words, then the fold of 8 picks, 4096 words and the shared row (two launches) against one launch of the rounding fold", FOLD_ROUNDS, 8, 4096)] {
        let s = shape(rounds, picks, words, 1); let default = (words.div_ceil(128) as u32, 128u32); let round_grid = ((picks * words).div_ceil(128) as u32, 128u32);
        // Six fixtures in rotation. The baseline's round writes its own buffer inside the graph, and its fold reads that buffer.
        let mut sets = vec![];
        for copy in 0..6u64 {
            let c = case(g, s, 7, 9000 + copy)?;
            let (base_out, cand_out, round_out) = (Buffer::new(g, &vec![0.0; words])?, Buffer::new(g, &vec![0.0; words])?, Buffer::new(g, &vec![0.0; picks * words])?);
            let mut base = base_args(&c, s, &base_out); if rounds != 0 { base.input[2] = round_out.address; }
            // A round command (kind 5): its payload is the LaneCast.
            let mut round = vec![0u8; 1040]; round[..4].copy_from_slice(&5u32.to_le_bytes());
            round[16..24].copy_from_slice(&c.rows.address.to_le_bytes()); round[24..32].copy_from_slice(&round_out.address.to_le_bytes()); round[32..36].copy_from_slice(&((picks * words) as u32).to_le_bytes());
            let tables = (g.upload(&round)?, g.upload(&command(&base, 1))?, g.upload(&command(&cand_args(&c, s, &cand_out), 19))?);
            sets.push((tables, c, base_out, cand_out, round_out));
        }
        let times = g.upload(&[0u8; 16])?;
        {
            let gs: &Gpu = g;
            let capture = |arm: usize| super::capture_graph(gs, || {
                let mut at = times; gs.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut at)])?;
                for i in 0..RUN { let t = sets[i % sets.len()].0;
                    if arm == 0 { if rounds != 0 { launch_replay(gs, base_round_replay, t.0, round_grid)?; } launch_replay(gs, base_replay, t.1, default)?; } else { launch_replay(gs, cand_fast_replay, t.2, default)?; } }
                at = times + 8; gs.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut at)])
            });
            let (base_graph, cand_graph) = (capture(0)?, capture(1)?);
            let graphs = [&base_graph, &cand_graph];
            for graph in graphs { super::sample(graph, times)?; }
            // The two arms wrote the same bytes in every fixture.
            for (_, _, base_out, cand_out, _) in &sets { if base_out.read(gs)? != cand_out.read(gs)? { return Err(format!("CAUSALP FOLD TIME outputs differ: {label}")); } }
            let mut us = [vec![], vec![]];
            for repeat in 0..31 { for k in 0..2 { let arm = (k + repeat) % 2; us[arm].push(super::sample(graphs[arm], times)?.1 / RUN as f64); } }
            let (base, cand) = (median(us[0].clone()), median(us[1].clone()));
            let least = |v: &Vec<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("CAUSALP FOLD TIME {label}: base_us={base:.3} cand_us={cand:.3} saved_us={:.3} base_min_us={:.3} cand_min_us={:.3} launches_a_graph={RUN} replays=31 grid={default:?}", base - cand, least(&us[0]), least(&us[1]));
        }
        g.clear()?;
    }
    Ok(())
}

/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); SelectRows is op 9, the inner causal shape unused.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct PArgs { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
struct SelCase { m: usize, r: usize, d: usize, budget: usize, tail: u32, first: usize, slots: usize }
/// 0 random, 1 many ties (16 levels), 2 all equal, 3 ascending, 4 descending, 5 signed zeros and ties, 6 infinities and ties, 7 NaN planted every 977, 8 all NaN, 9 NaN every 31 (many warps) and at index 1.
fn select_scores(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| match kind {
        0 => r.unit() * 8.0,
        1 => (r.next() % 16) as f32 - 8.0,
        2 => 1.5,
        3 => i as f32 * 0.25,
        4 => -(i as f32) * 0.25,
        5 => match r.next() % 4 { 0 => 0.0, 1 => -0.0, 2 => 1.0, _ => -1.0 },
        6 => match r.next() % 5 { 0 => f32::INFINITY, 1 => f32::NEG_INFINITY, 2 => 0.0, _ => (r.next() % 3) as f32 },
        7 => if i % 977 == 5 { f32::NAN } else { r.unit() },
        8 => f32::NAN,
        _ => if i % 31 == 0 || i == 1 { f32::NAN } else { r.unit() },
    }).collect()
}
fn select_launch(g: &Gpu, f: H, c: &SelCase, scores: u64, out: u64, fast: bool) -> Result<()> {
    let inner = Inner { op: 5, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    let shape = PShape { op: 9, tokens: c.m as u32, heads: 0, dim: c.d as u32, taps: c.budget as u32, rows: c.r as u32, cols: c.tail, first: c.first as u32, slots: c.slots as u32, inner };
    let mut a = PArgs { input: [scores, 4, 4, 4, 4], lengths: [(c.m * c.r) as u64, 0, 0, 0, 0], weight: 4, weights: 0, output: out, units: c.m as u64, words: (c.slots + 1) as u64, shape };
    if fast { g.launch(f, [c.m as u32, 1, 1], 1024, 0, &mut [ptr(&mut a)]) } else { g.launch(f, [c.m.div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut a)]) }
}
fn select_screen(g: &mut Gpu, base: H, cand: H, time: bool) -> Result<()> {
    let started = Instant::now();
    if !time {
        let (mut comparisons, mut bytes) = (0u64, 0u64);
        let mut cases = vec![];
        for &d in &[1usize, 2, 4, 8] { for &pools in &[1usize, 2, 3, 512] {
            for &r in &[1usize, 2, 3, 100, 511, 512, 513, 514, 1039, 2048, 8192, 20000, 32768, 32769] {
                if pools * d > 4096 { continue; }
                if r >= 20000 && !(d == 4 && (pools == 1 || pools == 512)) { continue; }
                for &tail in &[0u32, 1] {
                    for &m in &[1usize, 3, 64, 256, 257] {
                        if m > 3 && (![100usize, 1039, 2048].contains(&r) || pools == 2 || pools == 3) { continue; }
                        if r >= 20000 && m > 3 { continue; }
                        let budget = pools * d; let slots = budget + if tail != 0 { d - 1 } else { 0 };
                        // Last token sees the full r groups plus some tail; earlier tokens see fewer.
                        let first = (r * d + (d - 1)).saturating_sub(m - 1);
                        if first + m - 1 > r * d + d - 1 { continue; }
                        cases.push(SelCase { m, r, d, budget, tail, first, slots });
                        if r > 4 && (r * d).saturating_sub(2) + m - 1 < (r + 7) * d + d { cases.push(SelCase { m, r: r + 7, d, budget, tail, first: (r * d).saturating_sub(2), slots: slots + 5 }); }
                    }
                }
            }
        } }
        for (n, c) in cases.iter().enumerate() {
            for kind in 0..10u32 {
                if c.m > 3 && ![0u32, 1, 5, 8, 9].contains(&kind) { continue; }
                if c.r > 3000 && kind > 4 && n % 3 != 0 { continue; }
                let mut values = vec![];
                for t in 0..c.m { values.extend(select_scores(kind, c.r, 1000 + n as u64 * 31 + t as u64)); }
                let scores = Buffer::new(g, &values)?;
                let words = c.m * (c.slots + 1);
                let out_a = Buffer::new(g, &vec![f32::NAN; words])?; out_a.poison(g)?;
                let out_b = Buffer::new(g, &vec![f32::NAN; words])?; out_b.poison(g)?;
                select_launch(g, base, c, scores.address, out_a.address, false)?; select_launch(g, cand, c, scores.address, out_b.address, true)?; g.sync()?;
                let (x, y) = (out_a.read(g)?, out_b.read(g)?);
                if x != y { return Err(format!("CAUSALP SELECT BYTE FAIL m={} r={} d={} budget={} tail={} first={} slots={} kind={kind} first_diff_word={:?}", c.m, c.r, c.d, c.budget, c.tail, c.first, c.slots, x.iter().zip(&y).position(|(p, q)| p != q).map(|b| b / 4))); }
                if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err("CAUSALP SELECT guard changed".into()); }
                comparisons += 1; bytes += (words * 4) as u64;
                g.clear()?;
            }
            if n % 100 == 0 { println!("CAUSALP SELECT case {n}/{} ok comparisons={comparisons} elapsed_s={:.1}", cases.len(), started.elapsed().as_secs_f64()); }
        }
        println!("CAUSALP SELECT BYTES PASS cases={} comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", cases.len(), sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    } else {
        for &(label, r, d) in &[("context 4158", 1039usize, 4usize), ("context 8192", 2048, 4), ("context 32768", 8192, 4), ("context 1024 (all kept)", 256, 4)] {
            let c = SelCase { m: 1, r, d, budget: 2048, tail: 1, first: r * d + 3, slots: 2051 };
            let scores = Buffer::new(g, &select_scores(0, r, 5))?; let out = Buffer::new(g, &vec![0.0; c.slots + 1])?;
            let mut times = [vec![], vec![]];
            for repeat in 0..9 { for k in 0..2 {
                let arm = (k + repeat) % 2; let f = if arm == 0 { base } else { cand };
                select_launch(g, f, &c, scores.address, out.address, arm == 1)?; g.sync()?;
                let n = 16usize; let t0 = Instant::now();
                for _ in 0..n { select_launch(g, f, &c, scores.address, out.address, arm == 1)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            } }
            let (tb, tc) = (median(times[0].clone()), median(times[1].clone()));
            println!("CAUSALP SELECT TIME {label} groups={r} base_us={tb:.1} cand_us={tc:.1} speedup={:.2}", tb / tc);
            g.clear()?;
        }
    }
    Ok(())
}

/// STEP-PLANAR and ConvParts against main's real chains. Baseline (baseline image, generic launches): StepColumns then StepLayout, or ConvChannels then
/// ConvLayout on the packed q|k|v; the host re-places their per-head or per-plane words into the two runs the decoder's joins make. Candidate: the one
/// fast launch, from a poisoned output, with its own geometry, with the default one-block geometry, and through the replay command table.
/// The route pick (`Op::Route`, one group): the candidate image against the baseline image, every output byte, at the
/// served geometry (one block of 512 threads) and at one block of 128 threads (several experts a thread), then the time
/// of the served shape. Patterns hold ties (all equal, pairs equal), NaN, infinities, zeros of both signs and subnormals.
fn route_screen(g: &mut Gpu, base_replay: H, cand_replay: H, cand_direct: H, started: Instant) -> Result<()> {
    let route = |r: usize, k: usize, taps: u32| Shape { op: ROUTE, heads: 1, dim: 1, taps, rows: r as u32, cols: k as u32, epsilon: 0.0, bound: 2.5 };
    let kinds: [(u32, u32); 14] = [(0, 0), (7, 7), (9, 9), (7, 0), (0, 7), (2, 2), (4, 0), (5, 5), (1, 1), (6, 8), (8, 3), (3, 3), (9, 0), (2, 7)];
    let (mut comparisons, mut bytes, mut shapes) = (0u64, 0u64, 0u64);
    for &r in &[1usize, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 96, 128, 255, 256, 257, 288, 289, 320, 400, 480, 507, 511] { for &k in &[1usize, 2, 7, 8, 9, 16, 31, 32, 33, 64] { for taps in [0u32, 1] {
        if k > r { continue; }
        let s = route(r, k, taps); shapes += 1;
        for (ki, &(ik, wk)) in kinds.iter().enumerate() {
            let fx = fixture_count(g, s, ik, wk, 4000 + (r * 131 + k * 7 + ki) as u64, 0)?;
            let mut outs = vec![];
            // Arms: baseline replay 1x512; candidate replay 1x512; candidate direct 1x512; candidate replay 1x128; baseline replay 1x128.
            for arm in 0..5 {
                let out = Buffer::new(g, &vec![f32::NAN; fx.units * fx.words])?; out.poison(g)?;
                let mut args = arguments(&fx, &out);
                match arm {
                    0 => { let t = g.upload(&command(&args, 1))?; launch_replay(g, base_replay, t, (1, 512))?; }
                    1 => { let t = g.upload(&command(&args, 1))?; launch_replay(g, cand_replay, t, (1, 512))?; }
                    2 => launch_causal(g, cand_direct, &mut args, (1, 512))?,
                    3 => { let t = g.upload(&command(&args, 1))?; launch_replay(g, cand_replay, t, (1, 128))?; }
                    _ => { let t = g.upload(&command(&args, 1))?; launch_replay(g, base_replay, t, (1, 128))?; }
                }
                g.sync()?; outs.push(out.read(g)?);
            }
            for (arm, y) in outs.iter().enumerate().skip(1) {
                if &outs[0] != y { return Err(format!("CAUSALP ROUTE BYTE FAIL arm={arm} rows={r} picks={k} taps={taps} patterns={ik},{wk} first_diff_byte={:?}", outs[0].iter().zip(y).position(|(p, q)| p != q))); }
                if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("CAUSALP ROUTE guard changed arm={arm} rows={r} picks={k}")); }
                comparisons += 1; bytes += (fx.units * fx.words * 4) as u64;
            }
            g.clear()?;
        }
    } } }
    println!("CAUSALP ROUTE BYTES PASS shapes={shapes} patterns={} comparisons={comparisons} output_bytes={bytes} elapsed_s={:.1}", kinds.len(), started.elapsed().as_secs_f64());
    // Time at the served shape: 288 experts, 8 picks, one block of 512 threads, replay entries, 6 rotating sets, median of 9 x 24 calls.
    let s = route(288, 8, 1); let copies = 6usize; let mut sets = vec![];
    for c in 0..copies { let fx = fixture_count(g, s, 0, 0, 900 + c as u64, 0)?; let out = Buffer::new(g, &vec![0.0; fx.units * fx.words])?; let a = arguments(&fx, &out); sets.push((g.upload(&command(&a, 1))?, fx, out)); }
    let mut times = [vec![], vec![]];
    for repeat in 0..9 { for k in 0..2 {
        let arm = (k + repeat) % 2; let f = if arm == 0 { base_replay } else { cand_replay };
        launch_replay(g, f, sets[0].0, (1, 512))?; g.sync()?;
        let n = 24usize; let t0 = Instant::now();
        for i in 0..n { launch_replay(g, f, sets[(repeat * 5 + i) % copies].0, (1, 512))?; }
        g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
    } }
    println!("CAUSALP ROUTE TIME 288 experts top 8, one block of 512 threads, 24 launches in a row: baseline image {:.2} us a launch, candidate image {:.2} us", median(times[0].clone()), median(times[1].clone()));
    g.clear()?;
    Ok(())
}
fn planar_screen(g: &mut Gpu, base: H, cand: H, cand_replay: H, _base_replay: H, base_fast_replay: H, started: Instant) -> Result<()> {
    let mut specs: Vec<Shape> = vec![];
    for &h in &[1u32, 2, 3, 17, 32, 63, 64] { specs.push(Shape { op: STEP_PLANAR, heads: h, dim: 128, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 }); }
    for &h in &[1u32, 2, 3, 8, 17, 63, 64, 65] { for &d in &[1u32, 3, 32, 127, 128, 129] { for &t in &[2u32, 3, 4, 5] { specs.push(Shape { op: CONV_PARTS, heads: h, dim: d, taps: t, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 }); } } }
    specs.push(Shape {op:CONV_PARTS,heads:32,dim:128,taps:4,rows:0,cols:0,epsilon:0.0,bound:0.0});
    let kinds: [(u32, u32); 8] = [(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (6, 8), (8, 3), (3, 3)];
    let (mut comparisons, mut bytes) = (0u64, 0u64);
    for (n, s) in specs.iter().enumerate() {
        let (h, d, t) = (s.heads as usize, s.dim as usize, s.taps as usize); let w = h * d;
        let (lens, wlen, _, total) = if s.op == STEP_PLANAR { ([h * (4 * d + 1), h * d * d, 0, 0], 0, 1, h * (d * d + d)) } else { ([w, w, w, 3 * (t - 1) * w], 3 * w * t, 1, 3 * w * t) };
        for (ki, &(ik, wk)) in kinds.iter().enumerate() {
            if h * d > 20000 && ki > 3 { continue; }
            let seed = 300 + n as u64;
            let host: Vec<Vec<f32>> = lens.iter().enumerate().map(|(i, &c)| pattern(if i == 0 { ik } else { (ik + i as u32) % 9 }, c.max(1), seed * 7 + i as u64)).collect();
            let weight_host = pattern(wk, wlen.max(1), seed * 13 + 5);
            let inputs: Vec<Buffer> = host.iter().map(|v| Buffer::new(g, v)).collect::<Result<_>>()?;
            let weight = Buffer::new(g, &weight_host)?;
            // Baseline chain, re-placed.
            let expected: Vec<u8> = if s.op == STEP_PLANAR {
                let cols = Buffer::new(g, &vec![f32::NAN; w * (d + 1)])?; cols.poison(g)?;
                let lay = Buffer::new(g, &vec![f32::NAN; w * (d + 1)])?; lay.poison(g)?;
                let mut a = Arguments { input: [inputs[0].address, inputs[1].address, 4, 4], weight: 4, output: cols.address, shape: Shape { op: STEP_COLUMNS, ..*s } };
                launch_causal(g, base, &mut a, (w.div_ceil(128) as u32, 128))?;
                let mut b = Arguments { input: [cols.address, 4, 4, 4], weight: 4, output: lay.address, shape: Shape { op: STEP_LAYOUT, ..*s } };
                launch_causal(g, base, &mut b, ((w * (d + 1)).div_ceil(128) as u32, 128))?; g.sync()?;
                let r = lay.read(g)?; let words: Vec<[u8; 4]> = r[16..16 + w * (d + 1) * 4].as_chunks::<4>().0.to_vec();
                let mut out = vec![[0u8; 4]; total];
                for head in 0..h { for i in 0..d * d { out[head * d * d + i] = words[head * (d * d + d) + i]; } for i in 0..d { out[h * d * d + head * d + i] = words[head * (d * d + d) + d * d + i]; } }
                out.concat()
            } else {
                let packed = Buffer::new(g, &[host[0].clone(), host[1].clone(), host[2].clone()].concat())?;
                let ch = Buffer::new(g, &vec![f32::NAN; 3 * w * t])?; ch.poison(g)?;
                let lay = Buffer::new(g, &vec![f32::NAN; 3 * w * t])?; lay.poison(g)?;
                let mut a = Arguments { input: [packed.address, inputs[3].address, 4, 4], weight: weight.address, output: ch.address, shape: Shape { op: CONV_CHANNELS, ..*s } };
                launch_causal(g, base, &mut a, ((3 * w).div_ceil(128) as u32, 128))?;
                let mut b = Arguments { input: [ch.address, 4, 4, 4], weight: 4, output: lay.address, shape: Shape { op: CONV_LAYOUT, ..*s } };
                launch_causal(g, base, &mut b, ((3 * w * t).div_ceil(128) as u32, 128))?; g.sync()?;
                let r = lay.read(g)?; let words: Vec<[u8; 4]> = r[16..16 + 3 * w * t * 4].as_chunks::<4>().0.to_vec();
                let history = (t - 1) * w; let mut out = vec![[0u8; 4]; total];
                for plane in 0..3 { for i in 0..w { out[plane * w + i] = words[plane * w * t + i]; } for i in 0..history { out[3 * w + plane * history + i] = words[plane * w * t + w + i]; } }
                out.concat()
            };
            // Candidate arms.
            let fast = if s.op == STEP_PLANAR { (s.heads, 128u32) } else { ((3 * w).div_ceil(128) as u32, 128u32) };
            for arm in 0..3 {
                let out = Buffer::new(g, &vec![f32::NAN; total])?; out.poison(g)?;
                let mut input = [4u64; 4]; for (i, b) in inputs.iter().enumerate() { input[i] = b.address; }
                let mut args = Arguments { input, weight: weight.address, output: out.address, shape: *s };
                match arm {
                    0 => launch_causal(g, cand, &mut args, fast)?,
                    1 => launch_causal(g, cand, &mut args, (1, 128))?,
                    _ => { let table = g.upload(&command(&args, 19))?; launch_replay(g, cand_replay, table, fast)?; }
                }
                g.sync()?; let r = out.read(g)?;
                if r[16..16 + total * 4] != expected[..] { let at = r[16..16 + total * 4].iter().zip(&expected).position(|(p, q)| p != q); return Err(format!("CAUSALP PLANAR BYTE FAIL op={} heads={h} dim={d} taps={t} arm={arm} patterns={ik},{wk} first_diff_byte={at:?}", s.op)); }
                if r[..16] != GUARD || r[r.len() - 16..] != GUARD { return Err(format!("CAUSALP PLANAR guard changed op={} heads={h} dim={d} taps={t} arm={arm}", s.op)); }
                comparisons += 1; bytes += (total * 4) as u64;
            }
            g.clear()?;
        }
        if n % 20 == 0 { println!("CAUSALP PLANAR spec {n}/{} ok comparisons={comparisons} elapsed_s={:.1}", specs.len(), started.elapsed().as_secs_f64()); }
    }
    // Timing at the real shapes, replay entries, 6 rotating sets, median of 9 x 24 calls: main's two launches, lane D's direct Step, the planar launch.
    for heads in [64u32, 32] {
        let (h, d) = (heads as usize, 128usize); let w = h * d; let copies = 6usize;
        let step = Shape { op: STEP_PLANAR, heads, dim: 128, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
        let mut sets = vec![];
        for c in 0..copies {
            let p = Buffer::new(g, &pattern(0, h * (4 * d + 1), 900 + c as u64))?; let st = Buffer::new(g, &pattern(0, h * d * d, 950 + c as u64))?;
            let cols = Buffer::new(g, &vec![0.0; w * (d + 1)])?; let lay = Buffer::new(g, &vec![0.0; w * (d + 1)])?; let direct = Buffer::new(g, &vec![0.0; w * (d + 1)])?; let planar = Buffer::new(g, &vec![0.0; w * (d + 1)])?;
            let a = Arguments { input: [p.address, st.address, 4, 4], weight: 4, output: cols.address, shape: Shape { op: STEP_COLUMNS, ..step } };
            let b = Arguments { input: [cols.address, 4, 4, 4], weight: 4, output: lay.address, shape: Shape { op: STEP_LAYOUT, ..step } };
            let dd = Arguments { input: [p.address, st.address, 4, 4], weight: 4, output: direct.address, shape: Shape { op: STEP, ..step } };
            let pl = Arguments { input: [p.address, st.address, 4, 4], weight: 4, output: planar.address, shape: step };
            sets.push((g.upload(&command(&a, 1))?, g.upload(&command(&b, 1))?, g.upload(&command(&dd, 19))?, g.upload(&command(&pl, 19))?, p, st, cols, lay, direct, planar));
        }
        let mut times = [vec![], vec![], vec![], vec![]];
        for repeat in 0..9 { for k in 0..4 {
            let arm = (k + repeat) % 4;
            // The device admits the old layouts (arms 0 and 1) at 64 heads, not at 32: there only the planar step runs.
            if heads != 64 && arm < 2 { continue; }
            let n = 24usize;
            let run = |i: usize| -> Result<()> { let set = &sets[(repeat * 5 + i) % copies];
                match arm { 0 => { launch_replay(g, _base_replay, set.0, (w.div_ceil(128) as u32, 128))?; launch_replay(g, _base_replay, set.1, ((w * (d + 1)).div_ceil(128) as u32, 128))?; }
                    1 => launch_replay(g, cand_replay, set.2, (heads, 128))?, 2 => launch_replay(g, cand_replay, set.3, (heads, 128))?,
                    // The planar step of the baseline image, same commands and sets: the old body against the new one.
                    _ => launch_replay(g, base_fast_replay, set.3, (heads, 128))? } Ok(()) };
            run(0)?; g.sync()?; let t0 = Instant::now(); for i in 0..n { run(i)?; } g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
        } }
        if heads == 64 { println!("CAUSALP PLANAR TIME step {heads}x128, rotating 6 state sets: main two launches {:.1} us, direct Step (lane D) {:.1} us, StepPlanar {:.1} us, baseline image StepPlanar {:.1} us", median(times[0].clone()), median(times[1].clone()), median(times[2].clone()), median(times[3].clone())); }
        else { println!("CAUSALP PLANAR TIME step {heads}x128, rotating 6 state sets: StepPlanar {:.1} us, baseline image StepPlanar {:.1} us", median(times[2].clone()), median(times[3].clone())); }
        g.clear()?;
    }
    println!("CAUSALP PLANAR BYTES PASS specs={} comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", specs.len(), sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    Ok(())
}
