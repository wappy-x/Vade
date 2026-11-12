//! Same-bytes screen and timing for `mhc_boundary_project`: baseline PTX against
//! candidate PTX in one process on one device. Safe code only; the existing
//! driver wrapper holds the GPU boundary.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use crate::mhc_params::{Boundary, Shape};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// Value patterns. 0 plain, 1 wide dynamic range, 2 denormals and signed zeros,
/// 3 large finite, 4 overflowing squares with a NaN and an infinity,
/// 5 huge/tiny alternation that exposes any change of add order.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.next();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 },
                _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(60),
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i == 7 { f32::NAN } else if i == 11 { f32::INFINITY } else { v } }
            _ => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
        }
    }).collect()
}
struct Buffer { base: u64, address: u64, bytes: usize }
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
impl Buffer {
    fn new(g: &mut Gpu, values: &[f32]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), &le(values), GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: values.len() * 4 })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
struct Fixture { shape: Shape, after: bool, input: Vec<f32>, previous: Vec<f32>, previous_coefficients: Vec<f32>,
    weights: Vec<f32>, base: Vec<f32>, scale: Vec<f32> }
impl Fixture {
    fn new(shape: Shape, after: bool, input_kind: u32, weight_kind: u32, seed: u64) -> Self {
        let (t, w, c, h) = (shape.rows as usize, shape.wide(), shape.coefficients(), shape.hidden as usize);
        Self { shape, after, input: pattern(input_kind, t * if after { h } else { w }, seed), previous: pattern(input_kind, t * w, seed + 1),
            previous_coefficients: pattern(0, t * c, seed + 2),
            weights: pattern(weight_kind, c * w, seed + 3).into_iter().map(|v| if weight_kind == 0 { v / 64.0 } else { v }).collect(),
            base: pattern(0, c, seed + 4), scale: vec![0.25, 0.5, 0.125] }
    }
}
struct Cell { input: Buffer, previous: Buffer, previous_coefficients: Buffer, weights: Buffer, base: Buffer, scale: Buffer,
    residual: Buffer, mixing: Buffer, coefficients: Buffer, other: Buffer, norm: Buffer }
impl Cell {
    fn new(g: &mut Gpu, f: &Fixture) -> Result<Self> {
        let (t, w, c, h) = (f.shape.rows as usize, f.shape.wide(), f.shape.coefficients(), f.shape.hidden as usize);
        let blank = |n| vec![f32::NAN; n];
        Ok(Self { input: Buffer::new(g, &f.input)?, previous: Buffer::new(g, &f.previous)?, previous_coefficients: Buffer::new(g, &f.previous_coefficients)?,
            weights: Buffer::new(g, &f.weights)?, base: Buffer::new(g, &f.base)?, scale: Buffer::new(g, &f.scale)?,
            residual: Buffer::new(g, &blank(t * w))?, mixing: Buffer::new(g, &blank(t * w))?, coefficients: Buffer::new(g, &blank(t * c))?,
            other: Buffer::new(g, &blank(t * h * 2 + t * h.div_ceil(128)))?, norm: Buffer::new(g, &vec![1.0; h])? })
    }
    fn args(&self, f: &Fixture, weights: u64) -> Boundary {
        let h = f.shape.hidden as usize;
        Boundary { input: self.input.address, previous_input: if f.after { self.previous.address } else { 0 },
            previous_coefficients: if f.after { self.previous_coefficients.address } else { 0 }, residual: self.residual.address,
            weights, base: self.base.address, scale: self.scale.address, mixing_normalized: self.mixing.address,
            coefficients: self.coefficients.address, collapsed: self.other.address, norm_weight: self.norm.address,
            norm_partials: self.other.address + (f.shape.rows as usize * h * 8) as u64, output: self.other.address + (f.shape.rows as usize * h * 4) as u64,
            shape: f.shape, norm_epsilon: 1e-5, norm_bf16: 0, prefetch: 0, prefetch_bytes: 0, prefetch_policy: 0, prefetch_reserved: 0 }
    }
    fn planes(&self, g: &Gpu) -> Result<[Vec<u8>; 3]> { Ok([self.residual.read(g)?, self.mixing.read(g)?, self.coefficients.read(g)?]) }
    fn poison(&self, g: &Gpu) -> Result<()> { self.residual.poison(g)?; self.mixing.poison(g)?; self.coefficients.poison(g) }
}
fn shape(rows: u32, streams: u32, hidden: u32) -> Shape { Shape { rows, streams, hidden, iterations: 20, rms_eps: 1e-5, eps: 1e-6 } }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
/// The candidate is the two-launch form: rms (grid rows), then dots (grid rows x coefficients).
fn candidate(g: &Gpu, f: [H; 2], rows: u32, count: u32, threads: (u32, u32), args: &mut Boundary) -> Result<()> {
    g.launch(f[0], [rows, 1, 1], threads.0, 0, &mut [ptr(args)])?;
    g.launch(f[1], [rows * count, 1, 1], threads.1, 0, &mut [ptr(args)])
}
fn part(g: &Gpu, f: [H; 2], rows: u32, count: u32, threads: (u32, u32), which: usize, args: &mut Boundary) -> Result<()> {
    if which == 0 { g.launch(f[0], [rows, 1, 1], threads.0, 0, &mut [ptr(args)]) } else { g.launch(f[1], [rows * count, 1, 1], threads.1, 0, &mut [ptr(args)]) }
}
pub(crate) fn run(base_path: &str, candidate_path: &str, _extra: &[String]) -> Result<()> {
    let oracle=_extra.iter().any(|arg|arg=="--host-oracle");
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MHCPROJ baseline_sha256={} candidate_sha256={} host_strip_fold_oracle={oracle}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let base: H = g.function(ma, "mhc_boundary_project")?;
    let pair: [H; 2] = [g.function(mb, "mhc_boundary_rms")?, g.function(mb, "mhc_boundary_dots")?];
    let mut checks = 0; let mut all = String::new();
    // Real GLM shape: 4 streams x 4096. Small, odd, tail, two-group and four-group shapes follow.
    let shapes = [shape(1, 4, 4096), shape(3, 4, 4096), shape(1, 1, 3), shape(2, 2, 1), shape(3, 3, 7), shape(2, 5, 129),
        shape(1, 4, 4099), shape(2, 6, 517), shape(1, 8, 100), shape(1, 4, 4), shape(2, 4, 1024), shape(1, 4, 2049)];
    // The after-site fast path admits hidden 1024, 2048, 3072 and 4096 with four streams: rows 1 to 8 at each width (the
    // full pattern set at rows 1 and 3 of the two widths that had no device proof before MHC-RMS-b).
    let mut shapes = shapes.to_vec();
    for rows in 1..=8u32 { for hidden in [1024u32, 2048, 3072, 4096] { let s = shape(rows, 4, hidden); if !shapes.iter().any(|t| (t.rows, t.streams, t.hidden) == (s.rows, s.streams, s.hidden)) { shapes.push(s); } } }
    let full = |si: usize, s: &Shape| si < 2 || (s.streams == 4 && s.hidden.is_multiple_of(1024) && s.hidden >= 2048 && (s.rows == 1 || s.rows == 3));
    for (si, s) in shapes.iter().enumerate() {
        for after in [false, true] {
            for (ik, wk) in [(0, 0), (1, 1), (2, 0), (0, 2), (2, 2), (3, 0), (3, 3), (4, 0), (5, 5), (5, 1), (1, 5)] {
                if !full(si, s) && (ik, wk) != (0, 0) && (ik, wk) != (5, 5) && (ik, wk) != (2, 2) { continue; }
                let f = Fixture::new(*s, after, ik, wk, 1000 + si as u64 * 97 + ik as u64 * 7 + wk as u64);
                let (ca, cb) = (Cell::new(&mut g, &f)?, Cell::new(&mut g, &f)?);
                let count = s.coefficients() as u32;
                ca.poison(&g)?; let mut xa = ca.args(&f, ca.weights.address);
                g.launch(base, [s.rows, 1, 1], s.prepare_threads(), 0, &mut [ptr(&mut xa)])?; g.sync()?;
                let mut pa = ca.planes(&g)?;
                if oracle {let expected=super::super::order_gate::boundary_oracle(&g,&xa,false)?;
                    for (index,(plane,values)) in pa.iter_mut().zip(expected).enumerate() {let want=super::super::order_gate::guarded_oracle(&values,plane.len());
                        if *plane!=want {let at=plane.iter().zip(&want).position(|(got,want)|got!=want).unwrap();let word=at/4*4;
                            return Err(format!("MHCPROJ HOST_ORACLE FAIL boundary shape={}x{}x{} after={after} patterns={ik},{wk} plane={} first_byte={at} got_bits={:08x} oracle_bits={:08x}",s.streams,s.hidden,s.rows,["residual","normalized","coefficients"][index],u32::from_le_bytes(plane[word..word+4].try_into().unwrap()),u32::from_le_bytes(want[word..word+4].try_into().unwrap())));}*plane=want;}}
                for threads in [(256, 128), (96, 96), (512, 512), (96, 32 * 5), (128, 256)] {
                    cb.poison(&g)?; let mut xb = cb.args(&f, cb.weights.address);
                    candidate(&g, pair, s.rows, count, threads, &mut xb)?; g.sync()?;
                    let pb = cb.planes(&g)?;
                    for (name, (x, y)) in ["residual", "mixing_normalized", "coefficients"].iter().zip(pa.iter().zip(&pb)) {
                        if x != y {
                            let at = x.iter().zip(y).position(|(p, q)| p != q);
                            return Err(format!("MHCPROJ BYTE FAIL shape={}x{}x{} after={after} input_pattern={ik} weight_pattern={wk} threads={threads:?} plane={name} first_byte={at:?}",
                                s.streams, s.hidden, s.rows));
                        }
                    }
                    for plane in &pb { if plane[..16] != GUARD || plane[plane.len() - 16..] != GUARD { return Err("MHCPROJ guard changed".into()); } }
                    all.push_str(&sha256::digest(&[pb[0].as_slice(), &pb[1], &pb[2]].concat())); checks += 1;
                }
            }
        }
        println!("MHCPROJ BYTES PASS shape={}x{} rows={} checks_so_far={checks}", s.streams, s.hidden, s.rows);
    }
    println!("MHCPROJ BYTES TOTAL checks={checks} planes_per_check=3 digest={}", sha256::digest(all.as_bytes()));
    // Timing. K weight copies rotate so each call reads cold weights, as 90 distinct sites do in a decode token.
    for (rows, copies, threads) in [(1u32, 48usize, (256u32, 256u32)), (1, 1, (256, 256)), (1, 48, (256, 512)), (1, 48, (512, 256)), (1, 48, (512, 512)), (1, 48, (128, 256)), (1, 48, (256, 128)), (2, 48, (256, 256)), (4, 24, (256, 256)), (8, 12, (256, 256)), (16, 8, (256, 256)), (32, 4, (256, 256)), (64, 2, (256, 256)), (128, 2, (256, 256))] {
        let f = Fixture::new(shape(rows, 4, 4096), false, 0, 0, 77); let count = f.shape.coefficients() as u32;
        let cell = Cell::new(&mut g, &f)?; let mut weights = vec![cell.weights.address];
        for _ in 1..copies { weights.push(Buffer::new(&mut g, &f.weights)?.address); }
        let launches = 480usize; let (mut ta, mut tb) = (vec![], vec![]);
        for repeat in 0..7 {
            for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                let mut warm = cell.args(&f, weights[0]);
                if arm == 0 { g.launch(base, [rows, 1, 1], f.shape.prepare_threads(), 0, &mut [ptr(&mut warm)])?; } else { candidate(&g, pair, rows, count, threads, &mut warm)?; }
                g.sync()?;
                let started = Instant::now();
                for i in 0..launches {
                    let mut x = cell.args(&f, weights[i % copies]);
                    if arm == 0 { g.launch(base, [rows, 1, 1], f.shape.prepare_threads(), 0, &mut [ptr(&mut x)])?; } else { candidate(&g, pair, rows, count, threads, &mut x)?; }
                }
                g.sync()?;
                (if arm == 0 { &mut ta } else { &mut tb }).push(started.elapsed().as_secs_f64() * 1e6 / launches as f64);
            }
        }
        let (ma, mb) = (median(ta), median(tb));
        println!("MHCPROJ TIME rows={rows} weight_copies={copies} threads={threads:?} baseline_us_per_call={ma:.2} candidate_us_per_call={mb:.2} ratio={:.2} samples=7x{launches}", ma / mb);
    }
    // Split at the real decode shape: each new launch alone, cold weights, first site and after site.
    for after in [false, true] { for which in 0..2 {
        let f = Fixture::new(shape(1, 4, 4096), after, 0, 0, 77); let count = f.shape.coefficients() as u32; let cell = Cell::new(&mut g, &f)?;
        let mut t = vec![]; let mut x = cell.args(&f, cell.weights.address); part(&g, pair, 1, count, (256, 256), which, &mut x)?; g.sync()?;
        for _ in 0..7 {
            let started = Instant::now();
            for _ in 0..480 { let mut x = cell.args(&f, cell.weights.address); part(&g, pair, 1, count, (256, 256), which, &mut x)?; }
            g.sync()?; t.push(started.elapsed().as_secs_f64() * 1e6 / 480.0);
        }
        println!("MHCPROJ SPLIT {} alone after_site={after} rows=1 threads=256 us_per_call={:.2}", ["rms", "dots"][which], median(t));
    } }
    // The old and the new rms and dots launches alternated in this process (the previous image's against this image's).
    {
        let old_pair: [H; 2] = [g.function(ma, "mhc_boundary_rms")?, g.function(ma, "mhc_boundary_dots")?];
        let mut sum = [0.0f64; 2];
        for after in [false, true] { for which in 0..2 {
            let f = Fixture::new(shape(1, 4, 4096), after, 0, 0, 77); let count = f.shape.coefficients() as u32; let cell = Cell::new(&mut g, &f)?;
            let mut t = [vec![], vec![]];
            for repeat in 0..9 { for k in 0..2 {
                let arm = if repeat % 2 == 0 { k } else { 1 - k }; let pairs = if arm == 0 { old_pair } else { pair };
                let mut x = cell.args(&f, cell.weights.address); part(&g, pairs, 1, count, (256, 256), which, &mut x)?; g.sync()?;
                let started = Instant::now();
                for _ in 0..480 { let mut x = cell.args(&f, cell.weights.address); part(&g, pairs, 1, count, (256, 256), which, &mut x)?; }
                g.sync()?; t[arm].push(started.elapsed().as_secs_f64() * 1e6 / 480.0);
            } }
            let (o, n) = (median(t[0].clone()), median(t[1].clone()));
            println!("MHCPROJ ALT {} after_site={after} old_us={o:.2} new_us={n:.2} saved_us={:.2}", ["rms", "dots"][which], o - n);
            if after { sum[which] = o - n; }
        } }
        println!("MHCPROJ ALT per_token after_site rms_saved_us={:.2} dots_saved_us={:.2} total_ms_at_90_calls={:.3}", sum[0], sum[1], (sum[0] + sum[1]) * 90.0 / 1000.0);
    }
    // The old one-block project, first site and after site.
    for after in [false, true] {
        let f = Fixture::new(shape(1, 4, 4096), after, 0, 0, 77); let cell = Cell::new(&mut g, &f)?;
        let mut t = vec![]; let mut x = cell.args(&f, cell.weights.address); g.launch(base, [1, 1, 1], f.shape.prepare_threads(), 0, &mut [ptr(&mut x)])?; g.sync()?;
        for _ in 0..7 {
            let started = Instant::now();
            for _ in 0..480 { let mut x = cell.args(&f, cell.weights.address); g.launch(base, [1, 1, 1], f.shape.prepare_threads(), 0, &mut [ptr(&mut x)])?; }
            g.sync()?; t.push(started.elapsed().as_secs_f64() * 1e6 / 480.0);
        }
        println!("MHCPROJ OLD project alone after_site={after} rows=1 us_per_call={:.2}", median(t));
    }
    println!("MHCPROJ PASS");
    Ok(())
}
/// Planes both arms write. Every one is compared byte for byte, guards included.
struct Planes { residual: Buffer, mixing: Buffer, coefficients: Buffer, collapsed: Buffer, partials: Buffer, output: Buffer }
impl Planes {
    fn new(g: &mut Gpu, rows: usize, s: Shape) -> Result<Self> {
        let (w, c, h) = (s.wide(), s.coefficients(), s.hidden as usize); let blank = |n: usize| vec![f32::NAN; n];
        Ok(Self { residual: Buffer::new(g, &blank(rows * w))?, mixing: Buffer::new(g, &blank(rows * w))?, coefficients: Buffer::new(g, &blank(rows * c))?,
            collapsed: Buffer::new(g, &blank(rows * h))?, partials: Buffer::new(g, &blank(rows * h.div_ceil(128)))?, output: Buffer::new(g, &blank(rows * h))? })
    }
    fn all(&self) -> [&Buffer; 6] { [&self.residual, &self.mixing, &self.coefficients, &self.collapsed, &self.partials, &self.output] }
}
/// Sweep of the admitted family: the K2 project (`mhc_boundary_project`) against the split pair, each followed by the
/// unchanged finish launch, over rows 1..8, streams 1..32, many hidden widths, first and after sites, both norm
/// precisions, and value patterns that expose any change of add order. All six written planes are compared.
pub(crate) fn sweep(base_path: &str, candidate_path: &str, oracle:bool) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MHCSWEEP baseline_sha256={} candidate_sha256={} host_strip_fold_oracle={oracle}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let (project, finish) = (g.function(ma, "mhc_boundary_project")?, g.function(ma, "mhc_boundary_finish")?);
    let (rms, dots, finish_b) = (g.function(mb, "mhc_boundary_rms")?, g.function(mb, "mhc_boundary_dots")?, g.function(mb, "mhc_boundary_finish")?);
    let started = Instant::now(); let (mut comparisons, mut bytes, mut digest) = (0u64, 0u64, String::new());
    // The swept cases are mhc_params::split_cases, the list a host test compares with the predicate; the weight kinds follow the stream count.
    let mut cases: Vec<(u32, u32, Vec<usize>, Vec<(u32, u32)>)> = vec![];
    for (n, h, rows) in crate::mhc_params::split_cases() {
        let weights: Vec<(u32, u32)> = if h >= 4095 { vec![(5, 5)] } else if n <= 8 { vec![(0, 0), (5, 5), (2, 2), (1, 1)] } else { vec![(5, 5), (2, 2)] };
        cases.push((n, h, rows.into_iter().map(|r| r as usize).collect(), weights));
    }
    // Ring reuse far past the model's width: many chunks at few streams.
    cases.push((4, 65536, vec![1, 2], vec![(5, 5)])); cases.push((1, 1_048_576, vec![1], vec![(5, 5)]));
    let mut last_n = 0;
    for (n, h, rows_list, weight_kinds) in cases {
        let top = *rows_list.iter().max().unwrap();
        let s = Shape { rows: top as u32, streams: n, hidden: h, iterations: 20, rms_eps: 1e-5, eps: 1e-6 }; s.validate()?;
        let (w, c, hh) = (s.wide(), s.coefficients(), h as usize);
        let norm = Buffer::new(&mut g, &pattern(0, hh, 5).iter().map(|v| v.abs() + 0.5).collect::<Vec<_>>())?;
        let (arm_a, arm_b) = (Planes::new(&mut g, top, s)?, Planes::new(&mut g, top, s)?);
        for (ik, wk) in weight_kinds {
            let weights = Buffer::new(&mut g, &pattern(wk, c * w, 3000 + n as u64 * 131 + h as u64 + wk as u64).into_iter().map(|v| if wk == 0 { v / 64.0 } else { v }).collect::<Vec<_>>())?;
            let base = Buffer::new(&mut g, &pattern(0, c, 17 + n as u64))?; let scale = Buffer::new(&mut g, &[0.25, 0.5, 0.125])?;
            for after in [false, true] {
                let seed = 9000 + n as u64 * 977 + h as u64 * 31 + ik as u64;
                let input = Buffer::new(&mut g, &pattern(ik, top * if after { hh } else { w }, seed))?;
                let previous = Buffer::new(&mut g, &pattern(ik, top * w, seed + 1))?; let previous_c = Buffer::new(&mut g, &pattern(0, top * c, seed + 2))?;
                let mut oracle_cache=None;
                for &rows in &rows_list {
                    for bf16 in [0u32, 1] {
                        let shape = Shape { rows: rows as u32, ..s };
                        let args = |p: &Planes| Boundary { input: input.address, previous_input: if after { previous.address } else { 0 },
                            previous_coefficients: if after { previous_c.address } else { 0 }, residual: p.residual.address, weights: weights.address,
                            base: base.address, scale: scale.address, mixing_normalized: p.mixing.address, coefficients: p.coefficients.address,
                            collapsed: p.collapsed.address, norm_weight: norm.address, norm_partials: p.partials.address, output: p.output.address,
                            shape, norm_epsilon: 1e-5, norm_bf16: bf16, prefetch: 0, prefetch_bytes: 0, prefetch_policy: 0, prefetch_reserved: 0 };
                        for p in arm_a.all().into_iter().chain(arm_b.all()) { p.poison(&g)?; }
                        let (mut xa, mut xb) = (args(&arm_a), args(&arm_b));
                        g.launch(project, [rows as u32, 1, 1], shape.prepare_threads(), 0, &mut [ptr(&mut xa)])?;
                        g.launch(finish, [rows as u32, 1, 1], shape.prepare_threads(), 0, &mut [ptr(&mut xa)])?;
                        g.launch(rms, [rows as u32, 1, 1], 256, 0, &mut [ptr(&mut xb)])?;
                        g.launch(dots, [rows as u32 * c as u32, 1, 1], 256, 0, &mut [ptr(&mut xb)])?;
                        g.launch(finish_b, [rows as u32, 1, 1], shape.prepare_threads(), 0, &mut [ptr(&mut xb)])?;
                        g.sync()?;
                        if oracle && oracle_cache.is_none() {let mut full=args(&arm_a);full.shape.rows=top as u32;full.norm_bf16=0;
                            oracle_cache=Some(super::super::order_gate::boundary_oracle(&g,&full,true)?);}
                        for (plane,(name, (pa, pb))) in ["residual", "mixing_normalized", "coefficients", "collapsed", "norm_partials", "output"].iter().zip(arm_a.all().into_iter().zip(arm_b.all())).enumerate() {
                            let (mut x, y) = (pa.read(&g)?, pb.read(&g)?);
                            if let Some(expected)=&oracle_cache {let values=&expected[plane][..rows*(expected[plane].len()/top)];
                                let rounded=if plane==5 && bf16!=0 {Some(values.iter().copied().map(super::super::order_gate::rounded).collect::<Vec<_>>())} else {None};
                                let want=super::super::order_gate::guarded_oracle(rounded.as_deref().unwrap_or(values),x.len());
                                if x!=want {return Err(format!("MHCSWEEP HOST_ORACLE FAIL boundary streams={n} hidden={h} rows={rows} after={after} bf16={bf16} patterns={ik},{wk} plane={name} first_byte={:?}",x.iter().zip(&want).position(|(got,want)|got!=want)));}x=want;}
                            if x != y {
                                let at = x.iter().zip(&y).position(|(p, q)| p != q);
                                return Err(format!("MHCSWEEP BYTE FAIL streams={n} hidden={h} rows={rows} after={after} norm_bf16={bf16} input_pattern={ik} weight_pattern={wk} plane={name} first_byte={at:?}"));
                            }
                            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err(format!("MHCSWEEP guard changed streams={n} hidden={h} rows={rows} plane={name}")); }
                            bytes += x.len() as u64;
                        }
                        comparisons += 1;
                        if comparisons % 997 == 0 { digest = sha256::digest(format!("{digest}{comparisons}{n}{h}{rows}").as_bytes()); }
                    }
                }
            }
        }
        if n != last_n { last_n = n; println!("MHCSWEEP streams={n} ok comparisons_so_far={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
        g.clear()?;
    }
    println!("MHCSWEEP PASS comparisons={comparisons} planes_per_comparison=6 compared_bytes={bytes} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
