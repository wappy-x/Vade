//! Small-count NVFP4 expert rows (a draft verify pass; the host sends 1-row decode calls elsewhere): timing and same-bytes sweep of the new
//! kernel against the production `expert_group` NvRows path and lane K`s `nv_rows_tile`. Safe code only. Fixture code copied from kernel_nvrows.rs.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, sha256};
use std::time::Instant;
use super::envelope as e;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
fn e2m1(n: u8) -> f32 {
    let magnitude = u32::from(n & 7);
    let bits = if magnitude < 2 { magnitude * 0x3f00_0000 } else { ((magnitude / 2 + 126) << 23) | ((magnitude & 1) << 22) };
    f32::from_bits(bits | (u32::from(n & 8) << 28))
}
fn e4m3(b: u8) -> f32 {
    let (e, f) = ((b >> 3) & 15, b & 7);
    let v = if e == 0 { f as f32 / 512.0 } else if b & 127 == 127 { f32::NAN } else { f32::from_bits(((u32::from(e) + 120) << 23) | (u32::from(f) << 20)) };
    if b & 128 == 0 { v } else { -v }
}
/// Mirror of gpu/src/expert_ops.rs decode_lut: 256 (low, high) pairs then 256 E4M3 values.
fn lut() -> Vec<f32> { let mut v = vec![]; for b in 0..=255u8 { v.push(e2m1(b & 15)); v.push(e2m1(b >> 4)); } for b in 0..=255u8 { v.push(e4m3(b)); } v }
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
/// Activation rows: per 16-column block, 16 values then the block scale.
fn activations(kind: u32, blocks: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1); let mut v = vec![];
    for i in 0..blocks * 17 {
        let (u, s) = (r.unit(), r.next());
        v.push(if i % 17 == 16 { match kind { 2 => if s % 3 == 0 { 0.0 } else { 0.015625 * (1.0 + u.abs()) }, 3 => 1024.0 * (1.0 + u.abs()), _ => 0.25 * (1.0 + u.abs()) } } else { match kind {
            0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
            _ => if i % 5003 == 5 { f32::NAN } else if i % 5003 == 21 { f32::INFINITY } else { u * 8.0 } } });
    }
    v
}
struct Part { m: usize, weight: u64, scales: u64 }
fn entry(rows: usize, cols: usize, p: &Part, first: usize, first_row: usize, params: (f32, f32)) -> Vec<u8> {
    let mut v = vec![];
    for x in [6u32, rows as u32, cols as u32, 1, 16, p.m as u32] { v.extend(x.to_le_bytes()); }
    for x in [params.0, params.1, 0.0f32] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]);
    v.extend(p.weight.to_le_bytes()); v.extend(p.scales.to_le_bytes());
    for x in [first as u32, first_row as u32, (p.m * (cols / 16 * 17)) as u32, (rows * cols / 2) as u32, (rows * cols / 16) as u32, 0] { v.extend(x.to_le_bytes()); }
    v
}
fn args(table: u64, input: u64, output: u64, lut: u64, count: usize, units: usize, stride: usize) -> Vec<u8> {
    let mut v = vec![]; for x in [table, input, output, lut] { v.extend(x.to_le_bytes()); }
    for x in [count as u32, units as u32, 1, stride as u32] { v.extend(x.to_le_bytes()); } v
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Fixture { rows: usize, cols: usize, parts: Vec<Part>, table: u64, input: u64, lutp: u64, units: usize, params: (f32, f32) }
/// Build parts with the given token counts. `mode` 1: row r has weight byte r%256 and scale byte r/256 everywhere.
fn build(g: &mut Gpu, rows: usize, cols: usize, counts: &[usize], mode: u32, kind: u32, params: (f32, f32), seed: u64, shared_weights: Option<&[(u64, u64)]>) -> Result<Fixture> {
    let mut r = Rng(seed | 1); let mut parts = vec![]; let _stride = cols / 16 * 17; let mut input = vec![];
    let l = Buf::new(g, &le(&lut()))?;
    for (e, &m) in counts.iter().enumerate() {
        let (w, s) = if let Some(sw) = shared_weights { sw[e % sw.len()] } else {
            let w: Vec<u8> = (0..rows * cols / 2).map(|i| if mode == 1 { ((i / (cols / 2)) & 255) as u8 } else { r.next() as u8 }).collect();
            let s: Vec<u8> = (0..rows * cols / 16).map(|i| if mode == 1 { ((i / (cols / 16)) >> 8 & 255) as u8 } else { let x = r.next(); if x % 16 == 0 { (x >> 8) as u8 } else { 0x20 + ((x >> 8) % 56) as u8 } }).collect();
            (Buf::new(g, &w)?.address, Buf::new(g, &s)?.address) };
        parts.push(Part { m, weight: w, scales: s });
        for t in 0..m { input.extend(activations(kind, cols / 16, seed + 977 * e as u64 + t as u64)); }
    }
    let (mut table, mut first, mut first_row) = (vec![], 0usize, 0usize);
    for p in &parts { table.extend(entry(rows, cols, p, first, first_row, params)); first += rows * p.m; first_row += p.m; }
    Ok(Fixture { rows, cols, table: Buf::new(g, &table)?.address, input: Buf::new(g, &le(&input))?.address, lutp: l.address, units: first, params, parts })
}
fn launch_reference(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.parts.len(), fx.units, fx.cols / 16 * 17);
    g.launch(f, [fx.units.div_ceil(32) as u32, 1, 1], 32, 0, &mut [a.as_mut_ptr().cast()])
}
fn launch_tile(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.parts.len(), fx.units, fx.cols / 16 * 17);
    let max_m = fx.parts.iter().map(|p| p.m).max().unwrap_or(1);
    g.launch(f, [max_m.div_ceil(8) as u32, fx.rows.div_ceil(128) as u32, fx.parts.len() as u32], 64, 0, &mut [a.as_mut_ptr().cast()])
}
/// `nv_rows_small`: ceil(rows/16) blocks for each job (one token of one part), 160 threads, dynamic shared 24,064 bytes.
const SMALL_SHARED: u32 = (4 * (16 * 64 + 16 * 8 + 8 * 80) + 2 * 16 * 132 * 4) as u32;
fn launch_small(g: &Gpu, f: H, fx: &Fixture, out: u64) -> Result<()> {
    let mut a = args(fx.table, fx.input, out, fx.lutp, fx.parts.len(), fx.units, fx.cols / 16 * 17);
    let jobs: usize = fx.parts.iter().map(|p| p.m).sum();
    g.launch(f, [(fx.rows.div_ceil(16) * jobs) as u32, 1, 1], 160, SMALL_SHARED, &mut [a.as_mut_ptr().cast()])
}
/// Pass shapes: (label, token counts of each distinct expert part). A pass of R rows has about 8R expert slots; slots that
/// pick the same expert share a part, so a part usually holds one token.
fn pass_parts(rows: usize, seed: u64) -> Vec<usize> {
    let mut r = Rng(seed | 1); let mut counts = vec![0usize; 288];
    for _ in 0..rows * 8 { counts[(r.next() % 288) as usize] += 1; }
    counts.into_iter().filter(|&c| c > 0).collect()
}
pub(crate) fn run(base_path: &str, candidate_path: &str, time: bool) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("NVSMALL baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "expert_group")?; let tile = g.function(ma, "nv_rows_tile")?; let small = g.function(mb, "nv_rows_small")?;
    let started = Instant::now();
    if !time {
        let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
        let check = |g: &mut Gpu, fx: &Fixture, label: &str| -> Result<()> {
            let out_a = Buf::new(g, &vec![0xff; fx.units * 4])?; let out_b = Buf::new(g, &vec![0xff; fx.units * 4])?;
            out_a.poison(g)?; out_b.poison(g)?;
            launch_reference(g, reference, fx, out_a.address)?; launch_small(g, small, fx, out_b.address)?; g.sync()?;
            let (x, y) = (out_a.read(g)?, out_b.read(g)?);
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("NVSMALL BYTE FAIL {label} rows={} cols={} parts={:?} params={:?} first_word_and_byte={at:?}", fx.rows, fx.cols, fx.parts.iter().map(|p| p.m).collect::<Vec<_>>(), fx.params)); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("NVSMALL guard changed".into()); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (fx.units * 4) as u64); Ok(())
        };
        // All 256 weight bytes x all 256 scale bytes, parts of 1 to 4 tokens (and 8, the kernel's own limit).
        for (kind, params, counts) in [(0u32, (0.75f32, 0.125f32), vec![3usize]), (1, (1.0, 1.0), vec![1]), (2, (0.0009765625, 0.5), vec![2, 4]), (3, (3.0, 0.0078125), vec![1, 16]), (4, (0.75, 0.125), vec![2, 1, 12])] {
            let fx = build(&mut g, 65536, 128, &counts, 1, kind, params, 11 + kind as u64, None)?; check(&mut g, &fx, "all-pairs")?; g.clear()?;
        }
        println!("NVSMALL all 256x256 code pairs ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Every listed (rows, cols) of `small_sweep` against count families that satisfy the host predicate (at most 128 parts, 1 to 16 tokens a part,
        // mean at most 2) and seeded random lists of the same class. A pool of six weight sets is shared by the parts of a launch (parts still have their own table entries).
        let ladder: Vec<usize> = (1..=16).chain(std::iter::repeat(1).take(112)).collect();
        let families: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![1, 1], vec![2, 2], vec![1; 30], vec![1; 100], vec![1; 128], vec![2; 64], ladder,
            { let mut v = vec![16usize]; v.extend([1usize; 19]); v }, { let mut v = vec![15usize, 9, 4]; v.extend([1usize; 40]); v }, vec![1, 2, 1, 1, 3, 1, 1, 4, 1, 2, 1, 1]];
        let mut rng = Rng(0x5eed_5eed_1234_9876); let mut seed = 500u64;
        for &rows in e::NV_ROWS { for &cols in e::NV_COLS {
            let mut r2 = Rng(7 + rows as u64 * 131 + cols as u64); let mut sw = vec![];
            for _ in 0..6 {
                let w: Vec<u8> = (0..rows * cols / 2).map(|_| r2.next() as u8).collect(); let sc: Vec<u8> = (0..rows * cols / 16).map(|_| { let x = r2.next(); if x % 16 == 0 { (x >> 8) as u8 } else { 0x20 + ((x >> 8) % 56) as u8 } }).collect();
                sw.push((Buf::new(&mut g, &w)?.address, Buf::new(&mut g, &sc)?.address));
            }
            let mut lists = families.clone();
            for _ in 0..2 {
                loop {
                    let parts = 1 + (rng.next() % 128) as usize;
                    let l: Vec<usize> = (0..parts).map(|_| if rng.next() % 5 < 4 { 1 } else { 2 + (rng.next() % 15) as usize }).collect();
                    if l.iter().sum::<usize>() <= 2 * parts { lists.push(l); break; }
                }
            }
            for counts in &lists {
                seed += 1; let fx = build(&mut g, rows, cols, counts, 0, (seed % 5) as u32, if seed % 2 == 0 { (0.75, 0.125) } else { (1.0 / 3.0, 7.0) }, seed, Some(&sw))?; check(&mut g, &fx, "sweep")?;
            }
            g.clear()?;
        } }
        println!("NVSMALL sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Real layer shapes: the parts of a 1, 2, 4, 8 and 16 row pass (counts from the pass model), both matrices, several activation kinds.
        for &pass_rows in &[1usize, 2, 4, 8, 16] {
            for (rows, cols) in [(1024usize, 4096usize), (2048, 2048)] {
                for kind in [0u32, 1, 2, 4] {
                    let counts: Vec<usize> = pass_parts(pass_rows, 900 + pass_rows as u64).into_iter().take(128).collect();
                    let fx = build(&mut g, rows, cols, &counts, 0, kind, (0.75, 0.125), 900 + rows as u64 + kind as u64 + pass_rows as u64, None)?; check(&mut g, &fx, "real")?; g.clear()?;
                }
            }
        }
        println!("NVSMALL BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    println!("NVSMALL TIME case matrix parts token_rows | expert_group_us nv_rows_tile_us nv_rows_small_us | GB/s of each | floor_us_at_223GBs");
    let mut totals = vec![];
    for (rows_of_matrix, cols, label) in [(1024usize, 4096usize, "gate_up"), (2048, 2048, "down")] {
        let mut r2 = Rng(7); let mut sw = vec![];
        for _ in 0..128 {
            let w: Vec<u8> = (0..rows_of_matrix * cols / 2).map(|_| r2.next() as u8).collect(); let s: Vec<u8> = (0..rows_of_matrix * cols / 16).map(|_| 0x30 + (r2.next() % 20) as u8).collect();
            sw.push((Buf::new(&mut g, &w)?.address, Buf::new(&mut g, &s)?.address));
        }
        // Cases: a pass of R rows (about 8R slots over 288 experts), then dense small parts. Each case rotates through enough
        // different expert sets that the weights of one launch are never in L2 from the launch before (about 120 MB of distinct weights).
        let mut cases: Vec<(String, Vec<usize>)> = [1usize, 2, 4, 8, 16].iter().map(|&r| (format!("{r}"), pass_parts(r, 100 + r as u64).into_iter().take(128).collect())).collect();
        if label == "gate_up" { for (name, c) in [("d4x32", vec![4usize; 32]), ("d3x40", vec![3; 40]), ("d2x64", vec![2; 64])] { cases.push((name.to_string(), c)); } }
        else { for (name, c) in [("d4x32", vec![4usize; 32]), ("d3x40", vec![3; 40]), ("d2x64", vec![2; 64])] { cases.push((name.to_string(), c)); } }
        for (name, counts) in cases {
            let tokens: usize = counts.iter().sum();
            let per_launch = (counts.len() * (rows_of_matrix * cols / 2 + rows_of_matrix * cols / 16)) as f64;
            let k = ((120e6 / per_launch).ceil() as usize).clamp(1, 16);
            let mut fxs = vec![];
            for i in 0..k { let rot: Vec<(u64, u64)> = (0..128).map(|j| sw[(j + i * counts.len()) % 128]).collect(); fxs.push(build(&mut g, rows_of_matrix, cols, &counts, 0, 0, (0.75, 0.125), 31, Some(&rot))?); }
            let out = Buf::new(&mut g, &vec![0u8; fxs[0].units * 4])?;
            let mut t = [vec![], vec![], vec![]];
            for repeat in 0..5 { for arm in (0..3).map(|a| (a + repeat) % 3) {
                let run = |g: &Gpu, i: usize| -> Result<()> { match arm { 0 => launch_reference(g, reference, &fxs[i % k], out.address), 1 => launch_tile(g, tile, &fxs[i % k], out.address), _ => launch_small(g, small, &fxs[i % k], out.address) } };
                for i in 0..k { run(&g, i)?; } g.sync()?;
                let n = k.max(5); let t0 = Instant::now();
                for i in 0..n { run(&g, i)?; }
                g.sync()?; t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            } }
            let [ra, rb, rc] = t.map(median);
            println!("NVSMALL TIME case={name} {label} rows={rows_of_matrix} cols={cols} parts={} token_rows={tokens} rotations={k} expert_group_us={ra:.0} nv_rows_tile_us={rb:.0} nv_rows_small_us={rc:.0} GBs={:.0}/{:.0}/{:.0} floor_us_at_223GBs={:.0}", counts.len(), per_launch / ra / 1e3, per_launch / rb / 1e3, per_launch / rc / 1e3, per_launch / 223e3);
            if let Ok(rows) = name.parse::<usize>() { totals.push((rows, ra, rb, rc, per_launch)); }
        }
        g.clear()?;
    }
    for &pass_rows in &[1usize, 2, 4, 8, 16] {
        let (mut ra, mut rb, mut rc, mut floor) = (0.0, 0.0, 0.0, 0.0);
        for t in totals.iter().filter(|t| t.0 == pass_rows) { ra += t.1; rb += t.2; rc += t.3; floor += t.4 / 223e3; }
        // Three matrices a layer (gate, up and down; gate and up are the same shape as the first row of the pair): scale the pair by 1.5.
        println!("NVSMALL PASS pass_rows={pass_rows} ms_42_layers(3 matrices): expert_group={:.1} nv_rows_tile={:.1} nv_rows_small={:.1} floor={:.1}", ra * 63.0 / 1e3, rb * 63.0 / 1e3, rc * 63.0 / 1e3, floor * 63.0 / 1e3);
    }
    println!("NVSMALL TIME DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
