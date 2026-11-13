//! Lane P device screen for the plain FP32 row kernel (`project_f32_ordered`): same-bytes sweep and timing of
//! candidate entries against the baseline image's `project_f32_ordered`, in one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
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
/// Mirror of gpu/src/plain_admission.rs `plain_geometry` (tested there): fold blocks (160 threads, 4 rows) for up to 512 rows of at
/// least 1,024 columns, one-warp blocks for other matrices of up to 512 rows, four-warp blocks above.
fn plain_geo(rows: usize, cols: usize) -> (u32, u32) { if rows <= 512 && cols >= 1024 { (rows.div_ceil(4) as u32, 160) } else { head_geo(rows) } }
/// The geometry the head (294a64cef) launches: the baseline arm of every timing uses it, so a time compares the two production paths.
fn head_geo(rows: usize) -> (u32, u32) { if rows <= 512 { (rows.div_ceil(32) as u32, 32) } else { (rows.div_ceil(128) as u32, 128) } }
fn launch_g(g: &Gpu, f: H, weights: u64, input: u64, output: u64, rows: usize, cols: usize, geo: (u32, u32)) -> Result<()> {
    let (mut w, mut x, mut y, mut r, mut c) = (weights, input, output, rows as u32, cols as u32);
    g.launch(f, [geo.0, 1, 1], geo.1, 0, &mut [ptr(&mut w), ptr(&mut x), ptr(&mut y), ptr(&mut r), ptr(&mut c)])
}
fn launch(g: &Gpu, f: H, weights: u64, input: u64, output: u64, rows: usize, cols: usize) -> Result<()> { launch_g(g, f, weights, input, output, rows, cols, (rows.div_ceil(128) as u32, 128)) }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Check<'a> { g: Gpu, reference: H, candidates: &'a [(String, H)], comparisons: u64, bytes: u64 }
impl Check<'_> {
    /// One shape and pattern pair, against every candidate; `skew` bytes shift weights and input off 16-byte alignment.
    fn case(&mut self, rows: usize, cols: usize, wk: u32, ik: u32, seed: u64, skew: usize) -> Result<()> {
        let wv = pattern(wk, rows * cols + 4, seed); let xv = pattern(ik, cols + 4, seed ^ 0x55);
        let weight = Buffer::new(&mut self.g, &wv)?; let input = Buffer::new(&mut self.g, &xv)?;
        let (wa, xa) = (weight.address + skew as u64, input.address + skew as u64);
        let out_a = Buffer::new(&mut self.g, &vec![f32::NAN; rows])?;
        // The reference starts from the zero fill the host used to make; candidates from poison (the host now skips the fill).
        self.g.upload_at(out_a.address, &vec![0u8; rows * 4])?; launch(&self.g, self.reference, wa, xa, out_a.address, rows, cols)?; self.g.sync()?;
        let x = out_a.read(&self.g)?;
        for (name, f) in self.candidates {
            let out_b = Buffer::new(&mut self.g, &vec![f32::NAN; rows])?;
            out_b.poison(&self.g)?; launch_g(&self.g, *f, wa, xa, out_b.address, rows, cols, plain_geo(rows, cols))?; self.g.sync()?;
            let y = out_b.read(&self.g)?;
            // The default 128-thread geometry too: every schedule is correct under any block size.
            let out_c = Buffer::new(&mut self.g, &vec![f32::NAN; rows])?;
            out_c.poison(&self.g)?; launch(&self.g, *f, wa, xa, out_c.address, rows, cols)?; self.g.sync()?;
            if out_c.read(&self.g)? != x { return Err(format!("PLAINP BYTE FAIL (default geometry) {name} rows={rows} cols={cols} weight_pattern={wk} input_pattern={ik} skew={skew}")); }
            if x != y {
                let at = x.iter().zip(&y).position(|(p, q)| p != q);
                return Err(format!("PLAINP BYTE FAIL {name} rows={rows} cols={cols} weight_pattern={wk} input_pattern={ik} skew={skew} first_diff_byte={at:?}"));
            }
            if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("PLAINP guard changed {name} rows={rows} cols={cols}")); }
            self.comparisons += 1; self.bytes += (rows * 4) as u64;
        }
        Ok(())
    }
}
pub(crate) fn run(base_path: &str, candidate_path: &str, names: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("PLAINP baseline_sha256={} candidate_sha256={} candidates={names} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "project_f32_ordered")?;
    let mut candidates = vec![];
    for name in names.split(',').filter(|n| !n.is_empty()) { candidates.push((name.to_string(), g.function(mb, name)?)); }
    let mut check = Check { g, reference, candidates: &candidates, comparisons: 0, bytes: 0 };
    let started = Instant::now();
    if mode == "interleave" { return rowsp1::interleave::run(&mut check.g, (ma, mb)); }
    if mode == "replay" || mode == "all" {
        replay(&mut check.g, (ma, mb))?;
        if mode == "replay" { println!("PLAINP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64()); return Ok(()); }
        check.g.clear()?;
    }
    if mode == "mixed" || mode == "all" {
        let base_batch = check.g.function(ma, "flash_resident_projection_batch")?;
        let mut batch_cands = vec![];
        for name in names.split(',').filter(|n| !n.is_empty()) {
            let entry = if name == "project_f32_ordered" { "flash_resident_projection_batch" } else { name };
            batch_cands.push((entry.to_string(), check.g.function(mb, entry)?));
        }
        mixed(&mut check.g, base_batch, &batch_cands)?;
        if mode == "mixed" { println!("PLAINP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64()); return Ok(()); }
        check.g.clear()?;
    }
    if mode == "bytes" || mode == "both" || mode == "all" {
        // Every column count 1..=600 (all tails of every batch size, aligned and not), few rows, all patterns that matter.
        for cols in 1..=600usize {
            let kinds: &[(u32, u32)] = &[(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3), (6, 7), (0, 4), (8, 8)];
            for &(wk, ik) in kinds { check.case(3, cols, wk, ik, 1000 + cols as u64, 0)?; }
        }
        println!("PLAINP cols 1..=600 ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64());
        // Row counts around every block edge, at the real column counts and their neighbours.
        let rows_list = [1usize, 2, 3, 5, 17, 31, 32, 33, 63, 65, 95, 97, 127, 128, 129, 511, 512, 513, 143, 144, 145, 255, 256, 257, 384, 1000, 2048];
        let cols_list = [4usize, 12, 28, 32, 36, 40, 124, 128, 132, 252, 256, 260, 508, 512, 516, 1020, 1024, 1028, 2044, 2048, 2052, 4092, 4096, 4100, 8192];
        for &cols in &cols_list {
            for &(wk, ik) in &[(0u32, 0u32), (5, 5), (2, 2), (4, 0), (6, 7), (1, 1)] {
                for &rows in &rows_list { check.case(rows, cols, wk, ik, 5000 + cols as u64, 0)?; }
            }
        }
        println!("PLAINP rows x cols ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64());
        // Misaligned weight and input addresses take the scalar path in both arms.
        for &skew in &[4usize, 8, 12] {
            for &cols in &[4usize, 8, 36, 128, 516, 4096] { for &(wk, ik) in &[(0u32, 0u32), (2, 2), (4, 0)] { check.case(37, cols, wk, ik, 9000 + cols as u64, skew)?; } }
        }
        println!("PLAINP misaligned ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64());
        // Random shapes, fixed seed, columns up to 9000 with every residue.
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for round in 0..6000u64 {
            let rows = 1 + (rng.next() % 700) as usize; let cols = 1 + (rng.next() % 9000) as usize;
            let (wk, ik) = [(0u32, 0u32), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3), (6, 7), (8, 8)][(rng.next() % 8) as usize];
            check.case(rows, cols, wk, ik, 20000 + round, 0)?;
        }
        println!("PLAINP random ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64());
        // Real shapes of the decode token.
        for &(rows, cols, wk, ik) in &[(144usize, 4096usize, 0u32, 0u32), (144, 4096, 5, 5), (144, 4096, 4, 4), (144, 4096, 2, 2), (16384, 512, 0, 0), (16384, 512, 5, 5), (16384, 512, 4, 0),
            (384, 4096, 5, 5), (2048, 4096, 5, 5), (4096, 512, 5, 5), (4096, 4096, 0, 0), (65536, 128, 6, 7)] {
            check.case(rows, cols, wk, ik, 777 + cols as u64, 0)?;
        }
        println!("PLAINP real shapes ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64());
        println!("PLAINP BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", check.comparisons, check.bytes, sha256::digest(format!("{}{}", check.comparisons, check.bytes).as_bytes()), started.elapsed().as_secs_f64());
        check.g.clear()?;
    }
    if mode == "cols" {
        // Astra's train 16 review: every column count the deep schedules admit (multiples of four from 32 to COLS_MAX), not only
        // 1..=600 and a list. Few rows at three row counts that cross the warp tile (3, 33 and 130 rows), patterns that matter.
        for cols in (32..=8192usize).step_by(4) {
            let wide = cols > 600;
            for &(rows, wk, ik) in &[(3usize, 0u32, 0u32), (33, 5, 5), (130, 4, 0), (3, 2, 2), (33, 6, 7)] {
                if wide && cols > 4096 && rows == 130 { continue; }
                check.case(rows, cols, wk, ik, 40000 + cols as u64, 0)?;
            }
            if cols % 256 == 0 { println!("PLAINP cols {cols} ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64()); }
        }
        println!("PLAINP COLS PASS comparisons={} output_bytes={} elapsed_s={:.1}", check.comparisons, check.bytes, started.elapsed().as_secs_f64());
    }
    if mode == "corners" {
        // The envelope of plain_admission (32..=8192 columns, rows * columns <= 2^25) at its corners, and just outside it (original loop).
        let shapes: &[(usize, usize)] = &[(1, 8192), (2, 8192), (3, 8192), (33, 8192), (4096, 8192), (4097, 8192), (1, 8196), (1, 16384), (2, 16384), (3, 32768), (65536, 512), (65537, 512),
            (16384, 2048), (16385, 2048), (1 << 20, 32), ((1 << 20) + 1, 32), (100000, 128), (200000, 256), (2048, 16384), (4096, 4096), (8192, 4096), (1, 32), (1, 36), (2, 33 * 4)];
        for &(rows, cols) in shapes {
            let kinds: &[(u32, u32)] = if rows * cols > 1 << 24 { &[(0, 0), (5, 5)] } else { &[(0, 0), (5, 5), (2, 2), (4, 0), (6, 7)] };
            for &(wk, ik) in kinds { check.case(rows, cols, wk, ik, 31000 + (rows + cols) as u64, 0)?; check.g.clear()?; }
            println!("PLAINP corner rows={rows} cols={cols} ok comparisons={} elapsed_s={:.1}", check.comparisons, started.elapsed().as_secs_f64());
        }
        println!("PLAINP CORNERS PASS comparisons={} output_bytes={} elapsed_s={:.1}", check.comparisons, check.bytes, started.elapsed().as_secs_f64());
    }
    if mode == "bf16replay" {
        // The recorded-replay entries the model runs: resident_launch_plain (kind 3, FP32 weights, the head's geometry) against resident_launch_plain_bf16
        // (kind 26, the packed BF16 words, bf16_geometry = head_geo), for every listed shape, from poisoned outputs.
        let (fa, fb) = (check.g.function(ma, "resident_launch_plain")?, check.g.function(mb, "resident_launch_plain_bf16")?);
        for &(rows, cols) in &BF16_STEP_SHAPES {
            for &(wk, ik) in &[(0u32, 0u32), (2, 2), (4, 0), (5, 5), (6, 7)] {
                let wv = pattern(wk, rows * cols + 8, 4242 + (rows + cols) as u64); let xv = pattern(ik, cols + 4, 99 + cols as u64);
                let (patterns, widened) = bf16_patterns(&wv);
                let wide = Buffer::new(&mut check.g, &widened)?; let input = Buffer::new(&mut check.g, &xv)?; let narrow = upload16(&mut check.g, &patterns)?;
                let mut outs = vec![];
                for (f, weights, kind, geo) in [(fa, wide.address, 3u32, (rows.div_ceil(128) as u32, 128u32)), (fb, narrow, 26u32, head_geo(rows))] {
                    let out = Buffer::new(&mut check.g, &vec![f32::NAN; rows])?; out.poison(&check.g)?;
                    let mut payload = vec![];
                    for w in [weights, input.address, out.address, rows as u64, cols as u64] { payload.extend(w.to_le_bytes()); }
                    let table = check.g.upload(&command(kind, &payload))?;
                    launch_replay_t(&check.g, f, table, geo)?; check.g.sync()?;
                    outs.push(out.read(&check.g)?);
                }
                if outs[0] != outs[1] { return Err(format!("PLAINBF REPLAY BYTE FAIL rows={rows} cols={cols} patterns={wk},{ik} first_diff_byte={:?}", outs[0].iter().zip(&outs[1]).position(|(p, q)| p != q))); }
                if outs[1][..16] != GUARD || outs[1][outs[1].len() - 16..] != GUARD { return Err(format!("PLAINBF REPLAY guard rows={rows} cols={cols}")); }
                check.comparisons += 1; check.bytes += (rows * 4) as u64;
            }
            check.g.clear()?;
            println!("PLAINBF REPLAY shape {rows}x{cols} ok comparisons={}", check.comparisons);
        }
        println!("PLAINBF REPLAY PASS comparisons={} output_bytes={} elapsed_s={:.1}", check.comparisons, check.bytes, started.elapsed().as_secs_f64());
        println!("PLAINP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode == "bf16" || mode == "bf16time" || mode == "bf16all" {
        for (name, f) in candidates.iter() { bf16(&mut check, name, *f, mode, &started)?; }
        println!("PLAINP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode == "time" || mode == "both" || mode == "all" {
        // Weights rotate through enough copies (at least 384 MiB) that each call streams them from memory, as in the model.
        let shapes: &[(usize, usize)] = &[(144, 4096), (128, 4096), (384, 4096), (16384, 512), (2048, 512), (2048, 4096), (4096, 512), (4096, 128), (1024, 4096)];
        for &(rows, cols) in shapes {
            let values = pattern(0, rows * cols, 1); let bytes = le(&values);
            let copies = ((384usize << 20) / bytes.len()).clamp(2, 400);
            let mut w = vec![]; for _ in 0..copies { w.push(check.g.upload(&bytes)?); }
            let input = Buffer::new(&mut check.g, &pattern(0, cols, 3))?; let out = Buffer::new(&mut check.g, &vec![0.0; rows])?;
            let mut arms: Vec<(String, H)> = vec![("project_f32_ordered(base)".to_string(), reference)];
            for (n, f) in &candidates { arms.push((n.clone(), *f)); }
            let mut times: Vec<Vec<f64>> = vec![vec![]; arms.len()];
            for repeat in 0..9 {
                for k in 0..arms.len() {
                    let arm = (k + repeat) % arms.len();
                    let geo = if arm == 0 { head_geo(rows) } else { plain_geo(rows, cols) };
                    launch_g(&check.g, arms[arm].1, w[0], input.address, out.address, rows, cols, geo)?; check.g.sync()?;
                    let n = 24usize.min(copies); let t0 = Instant::now();
                    for i in 0..n { launch_g(&check.g, arms[arm].1, w[(repeat * 7 + i) % copies], input.address, out.address, rows, cols, geo)?; }
                    check.g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let floor = (rows * cols * 4) as f64 / 216e3;
            let base = median(times[0].clone());
            for (k, (name, _)) in arms.iter().enumerate() {
                let m = median(times[k].clone());
                println!("PLAINP TIME rows={rows} cols={cols} weight_mib={:.2} floor_us={floor:.1} arm={name} us_per_call={m:.1} speedup_vs_base={:.2}", bytes.len() as f64 / 1048576.0, base / m);
            }
            check.g.clear()?;
        }
    }
    println!("PLAINP DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}

/// Mirror of gpu/src/next_params.rs (repr(C)): MxProjection 56 bytes, MxLayer 552, ResidentProjectionBatch 560.
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MxProjection { weights: u64, scales: u64, input: u64, output: u64, rows: u32, cols: u32, tokens: u32, input_bf16: u32, round: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MxLayer { projections: [MxProjection; 9], ends: [u32; 10], count: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct Batch { layer: MxLayer, plain: u32, round: u32 }
#[derive(Clone, Copy)]
struct Job { plain: bool, rows: usize, cols: usize }
struct Built { batch: Batch, outputs: Vec<(u64, usize)> }
/// Quantized-weight bytes: random E4M3 codes without NaN; scale bytes in a safe exponent range.
fn mx_bytes(n: usize, seed: u64, scales: bool) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| { let v = r.next(); if scales { 118 + (v % 14) as u8 } else { let b = (v >> 8) as u8; if b & 0x7f == 0x7f { 0x38 } else { b } } }).collect()
}
/// One fixture of weights and inputs for `jobs`; each arm gets its own output region (16-byte guards between jobs).
fn build(g: &mut Gpu, jobs: &[Job], round: bool, seed: u64, kinds: (u32, u32)) -> Result<(Vec<(u64, u64, u64, usize)>, Batch)> {
    let mut batch = Batch::default();
    let mut parts = vec![];
    for (i, j) in jobs.iter().enumerate() {
        let s = seed * 31 + i as u64;
        let input = g.upload(&le(&pattern(kinds.1, j.cols, s ^ 5)))?;
        let (weights, scales) = if j.plain { (g.upload(&le(&pattern(kinds.0, j.rows * j.cols, s)))?, 0) } else { (g.upload(&mx_bytes(j.rows * j.cols, s, false))?, g.upload(&mx_bytes(j.rows * j.cols / 32, s ^ 9, true))?) };
        parts.push((weights, scales, input, j.rows));
        let blocks = if j.plain { j.rows.div_ceil(128) } else { j.rows.div_ceil(4) } as u32;
        batch.layer.projections[i] = MxProjection { weights, scales, input, output: 0, rows: j.rows as u32, cols: j.cols as u32, tokens: 1, input_bf16: 0, round: 0 };
        batch.layer.ends[i + 1] = batch.layer.ends[i] + blocks;
        if j.plain { batch.plain |= 1 << i; }
    }
    batch.layer.count = jobs.len() as u32;
    batch.round = round as u32;
    Ok((parts, batch))
}
fn arm_outputs(g: &mut Gpu, jobs: &[Job], batch: &Batch) -> Result<Built> {
    let mut b = *batch; let mut outputs = vec![];
    for (i, j) in jobs.iter().enumerate() {
        let buf = Buffer::new(g, &vec![f32::NAN; j.rows])?; buf.poison(g)?;
        b.layer.projections[i].output = buf.address; outputs.push((buf.base, j.rows * 4 + 32));
    }
    Ok(Built { batch: b, outputs })
}
fn launch_batch(g: &Gpu, f: H, batch: &mut Batch) -> Result<()> { let blocks = batch.layer.ends[batch.layer.count as usize]; g.launch(f, [blocks, 1, 1], 128, 0, &mut [ptr(batch)]) }
fn mixed(g: &mut Gpu, base: H, cands: &[(String, H)]) -> Result<()> {
    let started = Instant::now();
    let (mut comparisons, mut bytes) = (0u64, 0u64);
    let (p, m) = (true, false);
    let job = |plain: bool, rows: usize, cols: usize| Job { plain, rows, cols };
    // Orders and shapes. Plain columns here are multiples of 32 like every real batch (MxLayer::push demands it).
    let mut cases: Vec<Vec<Job>> = vec![
        vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(p, 384, 4096)],
        vec![job(p, 384, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096)],
        vec![job(m, 4096, 4096), job(p, 384, 4096), job(m, 4096, 4096), job(m, 4096, 4096)],
        vec![job(p, 128, 4096), job(p, 128, 4096), job(p, 128, 4096)],
        vec![job(p, 1, 32)], vec![job(m, 1, 32)], vec![job(p, 129, 64), job(m, 5, 32), job(p, 255, 128)],
        vec![job(p, 1, 8192)], vec![job(p, 1, 16384)], vec![job(p, 3, 32768), job(m, 5, 64)], vec![job(p, 2, 8192), job(m, 4, 8192), job(p, 33, 8224)], vec![job(p, 4096, 8192), job(p, 4097, 8192)],
        vec![job(p, 1, 32), job(p, 31, 32), job(p, 129, 32), job(p, 65536, 32)],
        vec![job(m, 17, 96), job(p, 300, 32), job(m, 33, 4096), job(p, 1, 4096), job(p, 128, 512), job(m, 1024, 512), job(p, 256, 256), job(m, 3, 64), job(p, 2, 8192)],
    ];
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..160 {
        let n = 1 + (rng.next() % 9) as usize;
        cases.push((0..n).map(|_| job(rng.next().is_multiple_of(2), 1 + (rng.next() % 600) as usize, 32 * (1 + (rng.next() % 130) as usize))).collect());
    }
    for (index, jobs) in cases.iter().enumerate() {
        for round in [false, true] {
            for &kinds in &[(0u32, 0u32), (5, 5), (2, 2), (4, 0), (6, 7)] {
                if index >= 8 && (kinds != (0, 0) && kinds != (5, 5) && kinds != (2, 2)) { continue; }
                let seed = 100 + index as u64;
                let (_, fixture) = build(g, jobs, round, seed, kinds)?;
                let mut a = arm_outputs(g, jobs, &fixture)?;
                launch_batch(g, base, &mut a.batch)?; g.sync()?;
                for (name, cand) in cands {
                    let mut b = arm_outputs(g, jobs, &fixture)?;
                    launch_batch(g, *cand, &mut b.batch)?; g.sync()?;
                    for ((pa, n), (pb, _)) in a.outputs.iter().zip(&b.outputs) {
                        let (x, y) = (g.read(*pa, *n)?, g.read(*pb, *n)?);
                        if x != y { return Err(format!("PLAINP MIXED BYTE FAIL {name} case={index} jobs={} round={round} patterns={kinds:?} first_diff_byte={:?}", jobs.len(), x.iter().zip(&y).position(|(p, q)| p != q))); }
                        if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("PLAINP MIXED guard changed {name} case={index}")); }
                        comparisons += 1; bytes += (*n - 32) as u64;
                    }
                }
            }
        }
        if index % 40 == 0 { println!("PLAINP MIXED case {index} ok comparisons={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
        g.clear()?;
    }
    println!("PLAINP MIXED BYTES PASS comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    // Timing: the real KDA-like batch, weights rotated through 8 copies so every call streams from memory.
    for (label, jobs) in [("mx3+plain384(plain last)", vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(p, 384, 4096)]),
        ("plain384+mx3(plain first)", vec![job(p, 384, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096)]),
        ("mx3 only", vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096)]),
        ("plain 3 x 128 rows", vec![job(p, 128, 4096), job(p, 128, 4096), job(p, 128, 4096)]),
        ("plain 384 rows", vec![job(p, 384, 4096)])] {
        let copies = 8usize;
        let mut set = vec![];
        for c in 0..copies { let (_, fx) = build(g, &jobs, true, 7 + c as u64, (0, 0))?; set.push(arm_outputs(g, &jobs, &fx)?); }
        let mut arms: Vec<(String, H)> = vec![("base".to_string(), base)];
        for (n, f) in cands { arms.push((n.clone(), *f)); }
        let mut times: Vec<Vec<f64>> = vec![vec![]; arms.len()];
        for repeat in 0..9 {
            for k in 0..arms.len() {
                let arm = (k + repeat) % arms.len(); let f = arms[arm].1;
                launch_batch(g, f, &mut set[0].batch)?; g.sync()?;
                let n = 16usize; let t0 = Instant::now();
                for i in 0..n { launch_batch(g, f, &mut set[(repeat + i) % copies].batch)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let tb = median(times[0].clone());
        for (k, (name, _)) in arms.iter().enumerate() {
            let tc = median(times[k].clone());
            println!("PLAINP MIXED TIME batch=\"{label}\" blocks={} arm={name} us_per_call={tc:.1} speedup_vs_base={:.2}", set[0].batch.layer.ends[set[0].batch.layer.count as usize], tb / tc);
        }
        g.clear()?;
    }
    Ok(())
}

/// Little-endian bytes of a `Command` table entry (1040 bytes: kind, 12 reserved, 1024 payload).
fn command(kind: u32, payload: &[u8]) -> Vec<u8> {
    let mut c = vec![0u8; 1040];
    c[..4].copy_from_slice(&kind.to_le_bytes());
    c[16..16 + payload.len()].copy_from_slice(payload);
    c
}
fn batch_bytes(b: &Batch) -> Vec<u8> {
    let mut v = vec![];
    for p in &b.layer.projections {
        for w in [p.weights, p.scales, p.input, p.output] { v.extend(w.to_le_bytes()); }
        for w in [p.rows, p.cols, p.tokens, p.input_bf16, p.round] { v.extend(w.to_le_bytes()); }
        v.extend([0u8; 4]);
    }
    for e in b.layer.ends { v.extend(e.to_le_bytes()); }
    v.extend(b.layer.count.to_le_bytes());
    v.extend([0u8; 4]);
    v.extend(b.plain.to_le_bytes());
    v.extend(b.round.to_le_bytes());
    assert_eq!(v.len(), 560);
    v
}
fn launch_replay(g: &Gpu, f: H, table: u64, blocks: u32) -> Result<()> { launch_replay_t(g, f, table, (blocks, 128)) }
fn launch_replay_t(g: &Gpu, f: H, table: u64, geo: (u32, u32)) -> Result<()> {
    let (mut t, mut index) = (table, 0u64);
    g.launch(f, [geo.0, 1, 1], geo.1, 0, &mut [ptr(&mut t), ptr(&mut index)])
}
/// The recorded-replay entries the model runs (`resident_launch_plain`, `resident_launch_mixed`): baseline against candidate.
fn replay(g: &mut Gpu, (ma, mb): (H, H)) -> Result<()> {
    let started = Instant::now();
    let (plain_a, plain_b) = (g.function(ma, "resident_launch_plain")?, g.function(mb, "resident_launch_plain")?);
    let (mixed_a, mixed_b) = (g.function(ma, "resident_launch_mixed")?, g.function(mb, "resident_launch_mixed")?);
    let (mut comparisons, mut bytes) = (0u64, 0u64);
    let rows_list = [1usize, 2, 31, 127, 128, 129, 143, 144, 145, 256, 257, 384, 1000, 2048];
    let cols_list = [4usize, 8, 28, 32, 36, 60, 64, 68, 128, 252, 256, 260, 512, 516, 1024, 2052, 4096, 4100];
    for &cols in &cols_list {
        for &(wk, ik) in &[(0u32, 0u32), (5, 5), (2, 2), (4, 0), (6, 7), (1, 1), (3, 3), (8, 8)] {
            let weight = Buffer::new(g, &pattern(wk, 2048 * cols + 4, 300 + cols as u64))?; let input = Buffer::new(g, &pattern(ik, cols + 4, 301 + cols as u64))?;
            for &rows in &rows_list {
                let mut outs = vec![];
                for (fi, f) in [plain_a, plain_b].into_iter().enumerate() {
                    let out = Buffer::new(g, &vec![f32::NAN; rows])?; out.poison(g)?;
                    let mut payload = vec![];
                    for w in [weight.address, input.address, out.address, rows as u64, cols as u64] { payload.extend(w.to_le_bytes()); }
                    let table = g.upload(&command(3, &payload))?;
                    launch_replay_t(g, f, table, if fi == 0 { (rows.div_ceil(128) as u32, 128) } else { plain_geo(rows, cols) })?; g.sync()?;
                    outs.push(out.read(g)?);
                }
                if outs[0] != outs[1] { return Err(format!("PLAINP REPLAY PLAIN BYTE FAIL rows={rows} cols={cols} patterns={wk},{ik} first_diff_byte={:?}", outs[0].iter().zip(&outs[1]).position(|(p, q)| p != q))); }
                if outs[1][..16] != GUARD || outs[1][outs[1].len() - 16..] != GUARD { return Err(format!("PLAINP REPLAY PLAIN guard rows={rows} cols={cols}")); }
                comparisons += 1; bytes += (rows * 4) as u64;
            }
            g.clear()?;
        }
    }
    println!("PLAINP REPLAY PLAIN PASS comparisons={comparisons} output_bytes={bytes} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    let (p, m) = (true, false);
    let job = |plain: bool, rows: usize, cols: usize| Job { plain, rows, cols };
    let mut cases: Vec<Vec<Job>> = vec![
        vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(p, 384, 4096)],
        vec![job(p, 384, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096)],
        vec![job(p, 128, 4096), job(p, 128, 4096), job(p, 128, 4096)], vec![job(p, 1, 32)], vec![job(m, 1, 32)],
        vec![job(m, 17, 96), job(p, 300, 32), job(m, 33, 4096), job(p, 1, 4096), job(p, 128, 512), job(m, 1024, 512), job(p, 256, 256), job(m, 3, 64), job(p, 2, 8192)],
    ];
    let mut rng = Rng(0x0bad_cafe_1234_5678);
    for _ in 0..60 { let n = 1 + (rng.next() % 9) as usize; cases.push((0..n).map(|_| job(rng.next().is_multiple_of(2), 1 + (rng.next() % 600) as usize, 32 * (1 + (rng.next() % 130) as usize))).collect()); }
    let mut mixed_comparisons = 0u64;
    for (index, jobs) in cases.iter().enumerate() {
        for round in [false, true] {
            for &kinds in &[(0u32, 0u32), (5, 5), (2, 2)] {
                let (_, fixture) = build(g, jobs, round, 500 + index as u64, kinds)?;
                let mut outs = vec![];
                for f in [mixed_a, mixed_b] {
                    let arm = arm_outputs(g, jobs, &fixture)?;
                    let table = g.upload(&command(17, &batch_bytes(&arm.batch)))?;
                    launch_replay(g, f, table, arm.batch.layer.ends[arm.batch.layer.count as usize])?; g.sync()?;
                    let mut planes = vec![]; for (pa, n) in &arm.outputs { planes.push(g.read(*pa, *n)?); }
                    outs.push(planes);
                }
                for (x, y) in outs[0].iter().zip(&outs[1]) {
                    if x != y { return Err(format!("PLAINP REPLAY MIXED BYTE FAIL case={index} round={round} patterns={kinds:?}")); }
                    if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("PLAINP REPLAY MIXED guard case={index}")); }
                    mixed_comparisons += 1; bytes += (x.len() - 32) as u64;
                }
            }
        }
        g.clear()?;
    }
    println!("PLAINP REPLAY MIXED PASS comparisons={mixed_comparisons} total_output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{comparisons}{mixed_comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
    // Timing through the replay entries at the real shapes (weights rotated).
    for (label, jobs, plain_kind) in [("router 144x4096", vec![job(p, 144, 4096)], true), ("kv 16384x512", vec![job(p, 16384, 512)], true), ("plain 4096x128", vec![job(p, 4096, 128)], true),
        ("mx3+plain384", vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(p, 384, 4096)], false), ("plain 3x128", vec![job(p, 128, 4096), job(p, 128, 4096), job(p, 128, 4096)], false)] {
        let copies = if jobs.iter().map(|j| j.rows * j.cols).sum::<usize>() > 8_000_000 { 8 } else { 160 };
        let mut tables = vec![]; let mut grids: Vec<(u32, u32)> = vec![]; let mut cgrids: Vec<(u32, u32)> = vec![];
        for c in 0..copies {
            let (_, fx) = build(g, &jobs, true, 9 + c as u64, (0, 0))?; let arm = arm_outputs(g, &jobs, &fx)?;
            if plain_kind {
                let pr = arm.batch.layer.projections[0];
                let mut payload = vec![]; for w in [pr.weights, pr.input, pr.output, pr.rows as u64, pr.cols as u64] { payload.extend(w.to_le_bytes()); }
                tables.push(g.upload(&command(3, &payload))?); grids.push(head_geo(pr.rows as usize)); cgrids.push(plain_geo(pr.rows as usize, pr.cols as usize));
            } else { tables.push(g.upload(&command(17, &batch_bytes(&arm.batch)))?); let blocks = arm.batch.layer.ends[arm.batch.layer.count as usize]; grids.push((blocks, 128)); cgrids.push((blocks, 128)); }
        }
        let arms = if plain_kind { [plain_a, plain_b] } else { [mixed_a, mixed_b] };
        let mut times = [vec![], vec![]];
        for repeat in 0..9 { for k in 0..2 {
            let arm = (k + repeat) % 2;
            let geo = |i: usize| if arm == 1 { cgrids[i] } else { grids[i] };
            launch_replay_t(g, arms[arm], tables[0], geo(0))?; g.sync()?;
            let n = 16usize.min(copies); let t0 = Instant::now();
            for i in 0..n { launch_replay_t(g, arms[arm], tables[(repeat * 5 + i) % copies], geo((repeat * 5 + i) % copies))?; }
            g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
        } }
        let (tb, tc) = (median(times[0].clone()), median(times[1].clone()));
        println!("PLAINP REPLAY TIME {label} base_geo={:?} cand_geo={:?} base_us={tb:.1} cand_us={tc:.1} speedup={:.2}", grids[0], cgrids[0], tb / tc);
        g.clear()?;
    }
    Ok(())
}

/// BF16 weights (project_bf16_ordered): the FP32 copy is the loader's widening of the BF16 patterns (`f32::from_bits(u32::from(u16) << 16)`),
/// the reference is the head's `project_f32_ordered` on that copy, and the candidate reads the patterns. Outputs start poisoned (0xff) with 16-byte
/// guards; the candidate runs in its one-warp geometry, the four-warp geometry and a 64-thread geometry (not admitted: the scalar chain of the same entry).
/// plain_admission::BF16_SHAPES, as one literal (the bench cannot link the gpu crate; a host test there compares the two).
const BF16_STEP_SHAPES: [(usize, usize); 10] = [(16, 4096), (32, 4096), (64, 4096), (128, 4096), (4096, 128), (8192, 128), (2048, 1536), (4096, 1536), (16384, 512), (32768, 512)];
fn bf16_patterns(values: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let patterns: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let widened = patterns.iter().map(|p| f32::from_bits(u32::from(*p) << 16)).collect();
    (patterns, widened)
}
fn upload16(g: &mut Gpu, patterns: &[u16]) -> Result<u64> {
    let bytes: Vec<u8> = patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(g.upload(&[GUARD.as_slice(), &bytes, GUARD.as_slice()].concat())? + 16)
}
fn bf16_case(c: &mut Check, name: &str, f: H, rows: usize, cols: usize, wk: u32, ik: u32, seed: u64, skew: usize) -> Result<()> {
    let wv = pattern(wk, rows * cols + 8, seed); let xv = pattern(ik, cols + 4, seed ^ 0x55);
    let (patterns, widened) = bf16_patterns(&wv);
    let wide = Buffer::new(&mut c.g, &widened)?; let input = Buffer::new(&mut c.g, &xv)?;
    let narrow = upload16(&mut c.g, &patterns)?;
    // A skew of `skew` bytes moves the FP32 weights by skew / 4 elements, so the BF16 weights move by the same elements = skew / 2 bytes.
    let (wa, na, xa) = (wide.address + skew as u64, narrow + (skew / 2) as u64, input.address + skew as u64);
    let out_a = Buffer::new(&mut c.g, &vec![f32::NAN; rows])?;
    out_a.poison(&c.g)?; launch_g(&c.g, c.reference, wa, xa, out_a.address, rows, cols, head_geo(rows))?; c.g.sync()?;
    let x = out_a.read(&c.g)?;
    for (label, geo) in [("warp32", (rows.div_ceil(32) as u32, 32u32)), ("tile128", (rows.div_ceil(128) as u32, 128u32)), ("scalar64", (rows.div_ceil(64) as u32, 64u32))] {
        let out_b = Buffer::new(&mut c.g, &vec![f32::NAN; rows])?;
        out_b.poison(&c.g)?; launch_g(&c.g, f, na, xa, out_b.address, rows, cols, geo)?; c.g.sync()?;
        let y = out_b.read(&c.g)?;
        if x != y {
            let at = x.iter().zip(&y).position(|(p, q)| p != q);
            return Err(format!("PLAINBF BYTE FAIL {name} geometry={label} rows={rows} cols={cols} weight_pattern={wk} input_pattern={ik} skew={skew} first_diff_byte={at:?}"));
        }
        if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("PLAINBF guard changed {name} rows={rows} cols={cols}")); }
        c.comparisons += 1; c.bytes += (rows * 4) as u64;
    }
    Ok(())
}
fn bf16(c: &mut Check, name: &str, f: H, mode: &str, started: &Instant) -> Result<()> {
    if mode != "bf16time" {
        // Every column count the kernel admits (multiples of 8 from 32 to 8192) and the ones just outside it, at row counts that cross the warp edge.
        for cols in (8..=8200usize).step_by(8) {
            for &(rows, wk, ik) in &[(1usize, 0u32, 0u32), (33, 5, 5), (144, 2, 2), (512, 6, 7), (513, 0, 0)] {
                bf16_case(c, name, f, rows, cols, wk, ik, 60000 + cols as u64, 0)?;
            }
            if cols % 1024 == 0 { println!("PLAINBF cols {cols} ok comparisons={} elapsed_s={:.1}", c.comparisons, started.elapsed().as_secs_f64()); c.g.clear()?; }
        }
        c.g.clear()?;
        // Every row count 1..=600 (the warp edges, the 512 limit and past it) at the narrowest, a mid and the real width.
        for rows in 1..=600usize {
            for &cols in &[32usize, 40, 4096] { bf16_case(c, name, f, rows, cols, 0, 0, 70000 + (rows * 10 + cols) as u64, 0)?; }
            if rows % 100 == 0 { c.g.clear()?; }
        }
        c.g.clear()?;
        println!("PLAINBF rows 1..=600 ok comparisons={} elapsed_s={:.1}", c.comparisons, started.elapsed().as_secs_f64());
        // Columns that are not a multiple of 8 take the scalar chain; misaligned addresses too.
        for cols in 1..=300usize { bf16_case(c, name, f, 3, cols, 5, 5, 80000 + cols as u64, 0)?; }
        println!("PLAINBF scalar cols 1..=300 ok comparisons={} elapsed_s={:.1}", c.comparisons, started.elapsed().as_secs_f64());
        // Skews of 4, 8 and 12 bytes (the FP32 reference loads 4-byte words, so 2 would fault in the reference itself).
        for &skew in &[4usize, 8, 12] { for &cols in &[8usize, 32, 40, 128, 4096] { bf16_case(c, name, f, 37, cols, 0, 0, 90000 + cols as u64, skew)?; } }
        c.g.clear()?;
        println!("PLAINBF scalar and misaligned ok comparisons={} elapsed_s={:.1}", c.comparisons, started.elapsed().as_secs_f64());
        // THE ADMITTED LIST (plain_admission::BF16_SHAPES, compared by a host test): every shape, all twelve value pairs, then its neighbours (one row more or fewer,
        // eight columns more or fewer), which are outside the list and take the scalar chain. Each case runs in three geometries from poisoned outputs.
        for &(rows, cols) in &BF16_STEP_SHAPES {
            let pairs: &[(u32, u32)] = if rows * cols > 4_000_000 { &[(0, 0), (2, 2), (5, 5), (6, 7), (4, 0)] } else { &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 0), (5, 5), (6, 7), (7, 7), (8, 8), (0, 4), (2, 0)] };
            for &(wk, ik) in pairs { bf16_case(c, name, f, rows, cols, wk, ik, 555 + (rows + cols) as u64 + wk as u64, 0)?; c.g.clear()?; }
            for (r, k) in [(rows - 1, cols), (rows + 1, cols), (rows, cols - 8), (rows, cols + 8)] { bf16_case(c, name, f, r, k, 0, 0, 666 + (r + k) as u64, 0)?; c.g.clear()?; }
            println!("PLAINBF listed shape {rows}x{cols} ok comparisons={} elapsed_s={:.1}", c.comparisons, started.elapsed().as_secs_f64());
        }
        // The shapes the earlier envelope admitted and the review named, now outside the list.
        for &(rows, cols) in &[(512usize, 8192usize), (513, 8192), (4096, 8192), (65536, 512), (1_048_576, 32)] { bf16_case(c, name, f, rows, cols, 0, 0, 777 + (rows + cols) as u64, 0)?; c.g.clear()?; }
        let mut rng = Rng(0x243f6a8885a308d3);
        for round in 0..1200u64 {
            let rows = 1 + (rng.next() % 700) as usize; let cols = if rng.next().is_multiple_of(4) { 1 + (rng.next() % 9000) as usize } else { 8 * (1 + (rng.next() % 1030) as usize) };
            let (wk, ik) = [(0u32, 0u32), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3), (6, 7), (8, 8)][(rng.next() % 8) as usize];
            bf16_case(c, name, f, rows, cols, wk, ik, 100000 + round, 0)?;
            if round % 100 == 99 { c.g.clear()?; }
        }
        c.g.clear()?;
        println!("PLAINBF BYTES PASS comparisons={} output_bytes={} elapsed_s={:.1}", c.comparisons, c.bytes, started.elapsed().as_secs_f64());
    }
    if mode != "bf16" {
        // Weights rotate through at least 384 MiB of copies per arm so every call streams from memory; median of 9 alternating samples.
        for &(rows, cols) in &[(144usize, 4096usize), (128, 4096), (64, 4096), (32, 4096), (288, 4096), (16384, 512), (2048, 1536), (4096, 128), (4096, 1536), (8192, 128)] {
            let (patterns, widened) = bf16_patterns(&pattern(0, rows * cols, 1));
            let wide_bytes = le(&widened); let narrow_bytes: Vec<u8> = patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
            let wide_copies = ((384usize << 20) / wide_bytes.len()).clamp(2, 600); let narrow_copies = ((384usize << 20) / narrow_bytes.len()).clamp(2, 600);
            let mut wide = vec![]; for _ in 0..wide_copies { wide.push(c.g.upload(&wide_bytes)?); }
            let mut narrow = vec![]; for _ in 0..narrow_copies { narrow.push(c.g.upload(&narrow_bytes)?); }
            let input = Buffer::new(&mut c.g, &pattern(0, cols, 3))?; let out = Buffer::new(&mut c.g, &vec![0.0; rows])?;
            let mut times = [vec![], vec![]];
            for repeat in 0..9 {
                for k in 0..2 {
                    let arm = (k + repeat) % 2;
                    let (func, geo, w, copies) = if arm == 0 { (c.reference, head_geo(rows), &wide, wide_copies) } else { (f, head_geo(rows), &narrow, narrow_copies) };
                    launch_g(&c.g, func, w[0], input.address, out.address, rows, cols, geo)?; c.g.sync()?;
                    let n = 24usize.min(copies); let t0 = Instant::now();
                    for i in 0..n { launch_g(&c.g, func, w[(repeat * 7 + i) % copies], input.address, out.address, rows, cols, geo)?; }
                    c.g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let (a, b) = (median(times[0].clone()), median(times[1].clone()));
            println!("PLAINBF TIME rows={rows} cols={cols} fp32_us={a:.1} bf16_us={b:.1} ratio={:.2} fp32_floor_us={:.1} bf16_floor_us={:.1}", a / b, (rows * cols * 4) as f64 / 223e3, (rows * cols * 2) as f64 / 223e3);
            c.g.clear()?;
        }
    }
    Ok(())
}

#[path = "kernel_rowsp1.rs"] pub(crate) mod rowsp1;
