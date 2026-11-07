//! Same-bytes sweep and timing for `nv_rows_stream` against the production `expert_operation` NvProject
//! batch (lane S's packed kernel), in one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/expert_ops.rs (repr(C)): Shape 36 bytes; Arguments 96; LaunchArguments 112. NvProject is 1.
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, rows: u32, cols: u32, block_rows: u32, block_cols: u32, tokens: u32, global: f32, activation: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, scales: u64, output: u64, shape: Shape }
#[repr(C)] #[derive(Clone, Copy)]
struct Launch { single: Arguments, batch: u64, count: u32 }
fn bytes_of(a: &Arguments) -> Vec<u8> {
    let mut v = vec![];
    for x in a.input.iter().chain([&a.weight, &a.scales, &a.output]) { v.extend(x.to_le_bytes()); }
    let s = a.shape; v.extend(s.op.to_le_bytes()); for x in [s.rows, s.cols, s.block_rows, s.block_cols, s.tokens] { v.extend(x.to_le_bytes()); }
    for x in [s.global, s.activation, s.bound] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]); v
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// Activation rows: per 16-column block, 16 values then the block scale. kind 0 plain, 1 wide range, 2 denormal and +-0,
/// 3 large, 4 includes NaN and inf.
fn activations(kind: u32, cols: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1); let mut v = vec![];
    for i in 0..cols / 16 * 17 {
        let (u, s) = (r.unit(), r.next());
        let x = if i % 17 == 16 { match kind { 2 => if s % 3 == 0 { 0.0 } else { 0.015625 * (1.0 + u.abs()) }, 3 => 1024.0 * (1.0 + u.abs()), _ => 0.25 * (1.0 + u.abs()) } } else { match kind {
            0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
            _ => if i == 5 { f32::NAN } else if i == 21 { f32::INFINITY } else { u * 8.0 } } };
        v.extend(x.to_le_bytes());
    }
    v
}
struct Job { weight: u64, scales: u64, input: u64, output: u64 }
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Run<'a> { reference: H, stream: H, g: &'a mut Gpu }
/// Launch one batch. `table` holds the job descriptors; returns nothing; the caller syncs.
fn launch(g: &Gpu, f: H, v: Geo, shape: Shape, jobs: &[Job], table: u64, out: u64) -> Result<()> {
    let args = Arguments { input: [jobs[0].input, 4, 4, 4], weight: jobs[0].weight, scales: jobs[0].scales, output: out, shape };
    let mut l = Launch { single: args, batch: table, count: jobs.len() as u32 };
    g.launch(f, [shape.rows.div_ceil(v.rows) * jobs.len() as u32, 1, 1], v.threads, v.shared, &mut [ptr(&mut l)])
}
/// Rows per block, threads per block, dynamic shared bytes, and the column multiple the body needs.
#[derive(Clone, Copy)] struct Geo { rows: u32, threads: u32, shared: u32, cols: usize }
const REFERENCE: Geo = Geo { rows: 16, threads: 32, shared: 0, cols: 1 };
const fn tile(r: u32, qn: u32, d: u32) -> Geo { Geo { rows: r, threads: 32 + r * qn, shared: d * (r * qn * 8 + r * (qn.div_ceil(8) * 8) + qn * 80) + 2 * r * (qn * 16 + 4) * 4, cols: if qn * 16 > 128 { qn as usize * 16 } else { 128 } } }
fn table_for(g: &mut Gpu, shape: Shape, jobs: &[Job], out: &Buf) -> Result<u64> {
    let mut t = vec![];
    for (i, j) in jobs.iter().enumerate() { t.extend(bytes_of(&Arguments { input: [j.input, 4, 4, 4], weight: j.weight, scales: j.scales, output: out.address + (i * shape.rows as usize * 4) as u64, shape })); }
    g.upload(&t)
}
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("NVBATCH baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "expert_operation")?;
    let list: [(&str, Geo); 2] = [("nv_rows_stream", Geo { rows: 64, threads: 64, shared: 0, cols: 512 }), ("nv_tile", tile(16, 8, 4))];
    let variants: Vec<(&str, Geo, H)> = list.into_iter().map(|(n, v)| g.function(mb, n).map(|h| (n, v, h))).collect::<Result<Vec<_>>>()?;
        let started = Instant::now(); let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
    // One comparison: build jobs from (weight source, scale source, activation kind), run both, compare outputs.
    let compare = |g: &mut Gpu, rows: usize, cols: usize, count: usize, wmode: u32, kind: u32, params: (f32, f32), seed: u64| -> Result<()> {
        let mut r = Rng(seed | 1); let shape = Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 1, block_cols: 16, tokens: 1, global: params.0, activation: params.1, bound: 0.0 };
        let mut jobs = vec![];
        for j in 0..count {
            let w: Vec<u8> = (0..rows * cols / 2).map(|i| match wmode { 1 => ((i / (cols / 2)) & 255) as u8, _ => r.next() as u8 }).collect();
            let s: Vec<u8> = (0..rows * cols / 16).map(|i| match wmode { 1 => ((i / (cols / 16)) >> 8 & 255) as u8, _ => { let x = r.next(); if x % 16 == 0 { (x >> 8) as u8 } else { 0x20 + ((x >> 8) % 56) as u8 } } }).collect();
            jobs.push(Job { weight: Buf::new(g, &w)?.address, scales: Buf::new(g, &s)?.address, input: Buf::new(g, &activations(kind, cols, seed + 31 * j as u64))?.address, output: 0 });
        }
        let out_a = Buf::new(g, &vec![0xff; rows * count * 4])?; let ta = table_for(g, shape, &jobs, &out_a)?;
        out_a.poison(g)?; launch(g, reference, REFERENCE, shape, &jobs, ta, out_a.address)?; g.sync()?; let x = out_a.read(g)?;
        for &(name, v, f) in variants.iter() {
            if cols % v.cols != 0 { continue; }
            let out_b = Buf::new(g, &vec![0xff; rows * count * 4])?; let tb = table_for(g, shape, &jobs, &out_b)?;
            out_b.poison(g)?; launch(g, f, v, shape, &jobs, tb, out_b.address)?; g.sync()?; let y = out_b.read(g)?;
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("NVBATCH BYTE FAIL variant={name} rows={rows} cols={cols} count={count} wmode={wmode} act_kind={kind} params={params:?} first_word_and_byte={at:?} reference_word={:?} candidate_word={:?} differing_words={}", at.map(|a| x[a.1..a.1 + 4].to_vec()), at.map(|a| y[a.1..a.1 + 4].to_vec()), x.chunks(4).zip(y.chunks(4)).filter(|(p, q)| p != q).count())); }
            if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("NVBATCH guard changed variant={name}")); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * count * 4) as u64);
        }
        g.clear()?; Ok(())
    };
    // All 256 weight bytes x all 256 scale bytes: row r has weight byte r%256 and scale byte r/256 everywhere.
    for (kind, params) in [(0u32, (0.75f32, 0.125f32)), (1, (1.0, 1.0)), (2, (0.0009765625, 0.5)), (3, (3.0, 0.0078125)), (4, (0.75, 0.125))] {
        compare(&mut g, 65536, 256, 1, 1, kind, params, 11 + kind as u64)?;
    }
    compare(&mut g, 65536, 512, 1, 1, 0, (0.75, 0.125), 21)?;
    println!("NVBATCH all 256x256 code pairs ok (6 runs of 65536 rows) comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Real decode shapes with random content.
    for (rows, cols, count, kind) in [(1024usize, 4096usize, 16usize, 0u32), (2048, 2048, 8, 0), (1024, 4096, 16, 1), (2048, 2048, 8, 2), (1024, 4096, 16, 4), (2048, 2048, 8, 3)] {
        compare(&mut g, rows, cols, count, 0, kind, (0.75, 0.125), 100 + rows as u64 + kind as u64)?;
    }
    println!("NVBATCH real shapes ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Partial row tiles: the last block of a job has lanes past the last row. Rows 1, 63 and 65 leave 63, 1 and 63 inactive lanes.
    for rows in [1usize, 63, 65] {
        for (cols, count) in [(512usize, 1usize), (512, 3), (4096, 1), (4096, 16), (8192, 2)] {
            for kind in 0..5u32 { compare(&mut g, rows, cols, count, 0, kind, (0.75, 0.125), 9000 + (rows * 7 + cols + count) as u64 + kind as u64)?; }
        }
    }
    println!("NVBATCH partial tiles rows 1, 63, 65 ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Family sweep.
    let mut seed = 500u64;
    for &rows in &[1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 255, 1000, 1024, 2048] {
        for &cols in &[256usize, 512, 768, 1024, 2048, 4096, 8192] {
            for &count in &[1usize, 2, 3, 8, 16] {
                if rows * cols * count > 40_000_000 { continue; }
                seed += 1; compare(&mut g, rows, cols, count, 0, (seed % 5) as u32, if seed % 2 == 0 { (0.75, 0.125) } else { (1.0 / 3.0, 7.0) }, seed)?;
            }
        }
    }
    println!("NVBATCH BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
    // Activation quantization: the old launch (one thread per 16-value block, 32-thread blocks) against one lane per element
    // (128-thread blocks of eight blocks), both the same `expert_operation` entry of the same image family. Output words must be equal.
    let quant_new = g.function(mb, "expert_operation")?;
    let quant_input = |kind: u32, cols: usize, seed: u64| -> Vec<u8> {
        let mut r = Rng(seed | 1); let mut v: Vec<f32> = vec![];
        let ties = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0];
        while v.len() < cols {
            match kind {
                // Plain and wide-range values.
                0 => for _ in 0..16 { v.push(r.unit() * 4.0); },
                1 => { let e = f32::from_bits(((r.next() % 60 + 97) as u32) << 23); for _ in 0..16 { v.push(r.unit() * e); } }
                // Exact ties: block maximum 6 * 2^j so the scale is exactly 2^j, other values on the E2M1 grid, midpoints and BF16 neighbours.
                2 | 3 => { let j = (r.next() % 30) as i32 - 15; let s = 2f32.powi(j); v.push(6.0 * s * if r.next() & 1 == 0 { 1.0 } else { -1.0 });
                    for _ in 1..16 { let t = ties[(r.next() % ties.len() as u64) as usize] * s; let t = if kind == 3 { match r.next() % 3 { 0 => f32::from_bits(t.to_bits().wrapping_add(1 << 16)), 1 => f32::from_bits(t.to_bits().wrapping_sub(1 << 16)), _ => t } } else { t };
                        v.push(if r.next() & 1 == 0 { t } else { -t }); } }
                // Zeros, negative zeros, denormals, tiny.
                4 => for _ in 0..16 { let x = r.next(); v.push(match x % 6 { 0 => 0.0, 1 => -0.0, 2 => f32::from_bits((x >> 8) as u32 % 8388607 + 1), 3 => -f32::from_bits((x >> 8) as u32 % 8388607 + 1), 4 => r.unit() * 1e-30, _ => r.unit() * 1e-3 }); },
                // Large values, BF16 overflow edges and the infinite and NaN edges (the old kernel is the oracle for the bytes).
                _ => for i in 0..16 { let x = r.next(); v.push(match (x % 9, i) { (0, _) => f32::MAX, (1, _) => -f32::MAX, (2, 3) => f32::INFINITY, (3, 5) => f32::NAN, (4, _) => 1e38 * r.unit(), _ => r.unit() * 1e20 }); },
            }
        }
        v.truncate(cols); v.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    let quant_run = |g: &mut Gpu, f: H, threads: u32, cols: usize, activation: f32, input: &Buf, out: &Buf| -> Result<()> {
        let shape = Shape { op: 0, rows: 1, cols: cols as u32, block_rows: 0, block_cols: 0, tokens: 1, global: 1.0, activation, bound: 0.0 };
        let mut l = Launch { single: Arguments { input: [input.address, 4, 4, 4], weight: 4, scales: 4, output: out.address, shape }, batch: 0, count: 1 };
        let units = (cols / 16) as u32; let per = threads / if threads == 32 { 1 } else { 16 };
        g.launch(f, [units.div_ceil(per), 1, 1], threads, 0, &mut [ptr(&mut l)])
    };
    let mut quant_checks = 0u64;
    let widths: Vec<usize> = crate::quantize_sweep::single_sweep().into_iter().map(|c| c as usize).collect();
    for &cols in &widths {
        for kind in 0..6u32 {
            for (n, activation) in [1.0f32, 0.125, 7.0, 1.0 / 3.0].into_iter().enumerate() {
                if kind > 1 && n > 1 { continue; }
                let input = Buf::new(&mut g, &quant_input(kind, cols, 4000 + cols as u64 * 13 + kind as u64 * 5 + n as u64))?;
                let (oa, ob) = (Buf::new(&mut g, &vec![0xff; cols / 16 * 17 * 4])?, Buf::new(&mut g, &vec![0xff; cols / 16 * 17 * 4])?);
                quant_run(&mut g, reference, 32, cols, activation, &input, &oa)?; quant_run(&mut g, quant_new, 128, cols, activation, &input, &ob)?; g.sync()?;
                let (x, y) = (oa.read(&g)?, ob.read(&g)?);
                if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("NVQUANT BYTE FAIL cols={cols} kind={kind} activation={activation} first_word_and_byte={at:?} reference_word={:?} candidate_word={:?}", at.map(|a| x[a.1..a.1 + 4].to_vec()), at.map(|a| y[a.1..a.1 + 4].to_vec()))); }
                if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("NVQUANT guard changed cols={cols}")); }
                quant_checks += 1; g.clear()?;
            }
        }
    }
    // Batches: several equal-width quantizations in one launch (one job table, one output vector, a range a job).
    let mut batch_checks = 0u64;
    for (cols, count) in crate::quantize_sweep::batch_sweep().into_iter().map(|(c, j)| (c as usize, j)) {
        {
            for kind in 0..6u32 {
                let shape = |a: f32| Shape { op: 0, rows: 1, cols: cols as u32, block_rows: 0, block_cols: 0, tokens: 1, global: 1.0, activation: a, bound: 0.0 };
                let acts = [1.0f32, 0.125, 7.0, 1.0 / 3.0];
                let (units, each) = (cols / 16, cols / 16 * 17);
                let mut inputs = vec![]; let mut singles = vec![];
                for j in 0..count {
                    let input = Buf::new(&mut g, &quant_input(kind, cols, 7000 + cols as u64 * 3 + j as u64 * 17 + kind as u64))?;
                    let out = Buf::new(&mut g, &vec![0xff; each * 4])?;
                    quant_run(&mut g, reference, 32, cols, acts[j % 4], &input, &out)?; singles.push(out); inputs.push(input);
                }
                let all = Buf::new(&mut g, &vec![0xff; each * count * 4])?;
                let mut table = vec![];
                for j in 0..count { table.extend(bytes_of(&Arguments { input: [inputs[j].address, 4, 4, 4], weight: 4, scales: 4, output: all.address + (j * each * 4) as u64, shape: shape(acts[j % 4]) })); }
                let t = g.upload(&table)?;
                let mut l = Launch { single: Arguments { input: [inputs[0].address, 4, 4, 4], weight: 4, scales: 4, output: all.address, shape: shape(acts[0]) }, batch: t, count: count as u32 };
                g.launch(quant_new, [(units as u32).div_ceil(8) * count as u32, 1, 1], 128, 0, &mut [ptr(&mut l)])?; g.sync()?;
                let y = all.read(&g)?;
                for j in 0..count {
                    let x = singles[j].read(&g)?;
                    if x[16..x.len() - 16] != y[16 + j * each * 4..16 + (j + 1) * each * 4] { return Err(format!("NVQUANT BATCH BYTE FAIL cols={cols} count={count} kind={kind} job={j}")); }
                }
                if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("NVQUANT batch guard changed cols={cols} count={count}")); }
                batch_checks += 1; g.clear()?;
            }
        }
    }
    println!("NVQUANT BATCH PASS checks={batch_checks}");
    println!("NVQUANT BYTES PASS checks={quant_checks} widths={} kinds=6 elapsed_s={:.1}", widths.len(), started.elapsed().as_secs_f64());
    for cols in [4096usize, 2048] {
        let input = Buf::new(&mut g, &quant_input(0, cols, 9))?; let out = Buf::new(&mut g, &vec![0u8; cols / 16 * 17 * 4])?;
        let mut times = [vec![], vec![]];
        for repeat in 0..7 { for k in 0..2 { let arm = if repeat % 2 == 0 { k } else { 1 - k }; let (f, t) = if arm == 0 { (reference, 32) } else { (quant_new, 128) };
            quant_run(&mut g, f, t, cols, 0.125, &input, &out)?; g.sync()?; let n = 200; let t0 = Instant::now();
            for _ in 0..n { quant_run(&mut g, f, t, cols, 0.125, &input, &out)?; } g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64); } }
        let (a, b) = (median(times[0].clone()), median(times[1].clone()));
        println!("NVQUANT TIME cols={cols} old_us={a:.2} lanes_us={b:.2} ratio={:.2}", a / b);
    }
    // Timing at the real shapes: eight rotating weight sets so each call reads weights from memory.
    for (rows, cols, count, label) in [(1024usize, 4096usize, 16usize, "gate_up"), (2048, 2048, 8, "down")] {
        let shape = Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 1, block_cols: 16, tokens: 1, global: 0.75, activation: 0.125, bound: 0.0 };
        let mut r = Rng(77); let mut sets = vec![];
        for _ in 0..8 {
            let mut jobs = vec![];
            for j in 0..count {
                let w: Vec<u8> = (0..rows * cols / 2).map(|_| r.next() as u8).collect(); let s: Vec<u8> = (0..rows * cols / 16).map(|_| 0x30 + (r.next() % 20) as u8).collect();
                jobs.push(Job { weight: Buf::new(&mut g, &w)?.address, scales: Buf::new(&mut g, &s)?.address, input: Buf::new(&mut g, &activations(0, cols, 5 + j as u64))?.address, output: 0 });
            }
            let out = Buf::new(&mut g, &vec![0u8; rows * count * 4])?; let table = table_for(&mut g, shape, &jobs, &out)?; sets.push((jobs, out, table));
        }
        let arms: Vec<(String, Geo, H)> = std::iter::once(("reference".to_string(), REFERENCE, reference)).chain(variants.iter().filter(|v| cols % v.1.cols == 0).map(|v| (v.0.to_string(), v.1, v.2))).collect();
        let mut times = vec![vec![]; arms.len()];
        for repeat in 0..7 {
            for k in 0..arms.len() {
                let arm = if repeat % 2 == 0 { k } else { arms.len() - 1 - k };
                let (jobs, out, table) = &sets[0]; launch(&g, arms[arm].2, arms[arm].1, shape, jobs, *table, out.address)?; g.sync()?;
                let n = 32; let t0 = Instant::now();
                for i in 0..n { let (jobs, out, table) = &sets[i % 8]; launch(&g, arms[arm].2, arms[arm].1, shape, jobs, *table, out.address)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let mb_read = (rows * cols / 2 + rows * cols / 16) as f64 * count as f64 / 1e6; let base_us = median(times[0].clone());
        for (k, a) in arms.iter().enumerate() { let us = median(times[k].clone());
            println!("NVBATCH TIME {label} rows={rows} cols={cols} jobs={count} arm={} us={us:.1} ratio_to_reference={:.2} GBs={:.0}", a.0, base_us / us, mb_read * 1e3 / us); }
        g.clear()?;
    }
    // How many distinct weight sets the timing rotates through (1 = warm L2 and TLB, 24 = cold): the production tile only.
    {
        let (rows, cols, count) = (1024usize, 4096usize, 16usize);
        let shape = Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 1, block_cols: 16, tokens: 1, global: 0.75, activation: 0.125, bound: 0.0 };
        let tile_fn = variants.iter().find(|v| v.0 == "nv_tile").ok_or("nv_tile")?; let (tile_geo, tile_h) = (tile_fn.1, tile_fn.2);
        let mut r = Rng(91); let mut sets = vec![];
        for _ in 0..24 {
            let mut jobs = vec![];
            for j in 0..count {
                let w: Vec<u8> = (0..rows * cols / 2).map(|_| r.next() as u8).collect(); let s: Vec<u8> = (0..rows * cols / 16).map(|_| 0x30 + (r.next() % 20) as u8).collect();
                jobs.push(Job { weight: Buf::new(&mut g, &w)?.address, scales: Buf::new(&mut g, &s)?.address, input: Buf::new(&mut g, &activations(0, cols, 5 + j as u64))?.address, output: 0 });
            }
            let out = Buf::new(&mut g, &vec![0u8; rows * count * 4])?; let table = table_for(&mut g, shape, &jobs, &out)?; sets.push((jobs, out, table));
        }
        for nsets in [1usize, 2, 8, 24] {
            let mut times = vec![];
            for _ in 0..7 {
                let n = 48; let (jobs, out, table) = &sets[0]; launch(&g, tile_h, tile_geo, shape, jobs, *table, out.address)?; g.sync()?;
                let t0 = Instant::now();
                for i in 0..n { let (jobs, out, table) = &sets[i % nsets]; launch(&g, tile_h, tile_geo, shape, jobs, *table, out.address)?; }
                g.sync()?; times.push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            println!("NVBATCH WARMTH gate_up weight_sets={nsets} nv_tile_us={:.1}", median(times));
        }
        g.clear()?;
    }
    println!("NVBATCH PASS");
    Ok(())
}
