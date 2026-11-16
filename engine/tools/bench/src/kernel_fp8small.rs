//! Same-bytes sweep and timing for `fp8_rows_tile` against the production `expert_group` Fp8Rows path, in
//! one process on one device. Safe code only.
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
fn lut() -> Vec<f32> { let mut v = vec![]; for b in 0..=255u8 { v.push(e2m1(b & 15)); v.push(e2m1(b >> 4)); } for b in 0..=255u8 { v.push(e4m3(b)); } v }
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn bf16(x: f32) -> f32 { let b = x.to_bits(); if x.is_nan() { return x; } f32::from_bits(((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16) }
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
/// Finite BF16-rounded activations. 0 plain, 1 wide range, 2 denormal and +-0, 3 large.
fn activations(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).enumerate().map(|(i, _)| { let (u, s) = (r.unit(), r.next()); if kind == 4 { return if i % 97 == 5 { f32::from_bits(0x7fc0_1234) } else if i % 89 == 7 { f32::INFINITY } else if i % 83 == 3 { f32::NEG_INFINITY } else { u * 8.0 }; } bf16(match kind {
        0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        _ => u.signum() * (1.0 + u.abs()) * 2f32.powi(40) }) }).collect()
}
#[derive(Clone, Copy)]
struct Geometry { rows: usize, cols: usize, br: usize, bc: usize, tokens: usize }
fn entry(g: Geometry, weight: u64, scales: u64) -> Vec<u8> {
    let mut v = vec![];
    for x in [7u32, g.rows as u32, g.cols as u32, g.br as u32, g.bc as u32, g.tokens as u32] { v.extend(x.to_le_bytes()); }
    for x in [1.0f32, 1.0, 0.0] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]);
    v.extend(weight.to_le_bytes()); v.extend(scales.to_le_bytes());
    let scale_words = g.rows.div_ceil(g.br) * g.cols.div_ceil(g.bc);
    for x in [0u32, 0, (g.tokens * g.cols) as u32, (g.rows * g.cols) as u32, (scale_words * 4) as u32, 0] { v.extend(x.to_le_bytes()); }
    v
}
/// A block scale: normal values, with a zero, a denormal, a huge value and an infinity at fixed places (the kernels must give the reference's bytes for them too).
fn scale_of(k: usize, normal: f32, x: u64) -> f32 {
    match k % 13 { 4 => 0.0, 7 => f32::from_bits((x >> 20) as u32 & 0x7f_ffff), 9 => 3.0e38, 11 => f32::INFINITY, _ => normal }
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
/// A fixture of several parts (same rows and cols, own weights and token counts), the group table of one launch.
struct Multi { rows: usize, cols: usize, counts: Vec<usize>, table: u64, input: u64, lutp: u64, units: usize }
fn build_parts(g: &mut Gpu, rows: usize, cols: usize, br: usize, bc: usize, counts: &[usize], mode: u32, kind: u32, seed: u64, pool: Option<&[(u64, u64)]>) -> Result<Multi> {
    build_table(g, rows, cols, br, bc, counts, mode, kind, seed, pool, false)
}
/// `same_rows`: every part reads the SAME token rows (first row 0 in every entry; the counts are equal), the table the row stage sends for the
/// block-FP8 matrices of one input. Otherwise the first rows are cumulative (a part owns its own rows).
#[allow(clippy::too_many_arguments)]
fn build_table(g: &mut Gpu, rows: usize, cols: usize, br: usize, bc: usize, counts: &[usize], mode: u32, kind: u32, seed: u64, pool: Option<&[(u64, u64)]>, same_rows: bool) -> Result<Multi> {
    let mut r = Rng(seed | 1); let (mut table, mut input) = (vec![], vec![]); let (mut first, mut first_row) = (0usize, 0usize);
    for (e, &m) in counts.iter().enumerate() {
        let (waddr, saddr) = if let Some(p) = pool { p[e % p.len()] } else {
            let w: Vec<u8> = (0..rows * cols).map(|i| if mode == 1 { ((i / cols + i % cols + e) % 256) as u8 } else { (r.next() >> 8) as u8 }).collect();
            let scales: Vec<f32> = (0..rows.div_ceil(br) * cols.div_ceil(bc)).enumerate().map(|(k, _)| { let x = r.next(); let ex = (x % 24) as i32 - 14; scale_of(k, (1.0 + ((x >> 8) % 1000) as f32 / 1000.0) * 2f32.powi(ex), x) }).collect();
            (Buf::new(g, &w)?.address, Buf::new(g, &le(&scales))?.address) };
        let geo = Geometry { rows, cols, br, bc, tokens: m };
        let mut en = entry(geo, waddr, saddr);
        // first and first_row are the u32 words at offsets 56 and 60 of the entry
        en[56..60].copy_from_slice(&(first as u32).to_le_bytes()); en[60..64].copy_from_slice(&(first_row as u32).to_le_bytes());
        table.extend(en);
        if !same_rows || e == 0 { for t in 0..m { input.extend(activations(kind, cols, seed + 977 * e as u64 + t as u64)); } }
        first += rows * m; if !same_rows { first_row += m; }
    }
    let xb = Buf::new(g, &le(&input))?; let tb = Buf::new(g, &table)?; let l = Buf::new(g, &le(&lut()))?;
    Ok(Multi { rows, cols, counts: counts.to_vec(), table: tb.address, input: xb.address, lutp: l.address, units: first })
}
fn margs(fx: &Multi, out: u64) -> Vec<u8> {
    let mut v = vec![]; for x in [fx.table, fx.input, out, fx.lutp] { v.extend(x.to_le_bytes()); }
    for x in [fx.counts.len() as u32, fx.units as u32, 1, fx.cols as u32] { v.extend(x.to_le_bytes()); } v
}
fn mlaunch_reference(g: &Gpu, f: H, fx: &Multi, out: u64) -> Result<()> { let mut a = margs(fx, out); g.launch(f, [fx.units.div_ceil(32) as u32, 1, 1], 32, 0, &mut [a.as_mut_ptr().cast()]) }
fn mlaunch_tile(g: &Gpu, f: H, fx: &Multi, out: u64) -> Result<()> {
    let mut a = margs(fx, out); let most = fx.counts.iter().copied().max().unwrap_or(1);
    g.launch(f, [most.div_ceil(64) as u32, fx.rows.div_ceil(64) as u32, fx.counts.len() as u32], 256, 0, &mut [a.as_mut_ptr().cast()])
}
const SMALL_SHARED: u32 = (4 * 16 * 128 + 4 * 128 * 4 + 4 * 128 * 4 + 3 * 128 * 4 + 2 * 16 * 132 * 4) as u32;
fn mlaunch_small(g: &Gpu, f: H, fx: &Multi, out: u64) -> Result<()> {
    let mut a = margs(fx, out); let jobs: usize = fx.counts.iter().sum();
    g.launch(f, [(fx.rows.div_ceil(16) * jobs) as u32, 1, 1], 160, SMALL_SHARED, &mut [a.as_mut_ptr().cast()])
}
/// `fp8_rows_multi`: 128 threads, `ceil(rows/32)` blocks for each part, static shared memory.
/// `fp8_rows_few`: 128 threads, `ceil(rows/16)` blocks for each part, static shared memory.
fn mlaunch_few(g: &Gpu, f: H, fx: &Multi, out: u64) -> Result<()> {
    let mut a = margs(fx, out);
    g.launch(f, [(fx.rows.div_ceil(16) * fx.counts.len()) as u32, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
fn mlaunch_multi(g: &Gpu, f: H, fx: &Multi, out: u64) -> Result<()> {
    let mut a = margs(fx, out);
    g.launch(f, [(fx.rows.div_ceil(32) * fx.counts.len()) as u32, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
static SHARE_PAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// `fp8_rows_share`: `32 * ceil(m / 2) + 128` threads, `ceil(rows/16)` blocks for each part, static shared memory. Every part has the same count `m`.
fn mlaunch_share(g: &Gpu, f: H, fx: &Multi, out: u64) -> Result<()> {
    let mut a = margs(fx, out); let m = fx.counts[0];
    // Dynamic shared bytes: 3 raw weight stages, 3 scale stages, two product buffers, 3 activation stages (2 from 5 tokens) of 2 * ceil(m / 2) rows
    // of 132 words. `SHARE_PAD`: more bytes than the kernel needs (mode `sharepad`: how many blocks an SM holds decides the launch time).
    let shared = 3 * 16 * 128 + 3 * 128 * 4 + 2 * 16 * 132 * 4 + (if m > 4 { 2 } else { 3 }) * 2 * m.div_ceil(2) * 132 * 4 + SHARE_PAD.load(std::sync::atomic::Ordering::Relaxed);
    g.launch(f, [(fx.rows.div_ceil(16) * fx.counts.len()) as u32, 1, 1], (32 * m.div_ceil(2) + 128) as u32, shared as u32, &mut [a.as_mut_ptr().cast()])
}
/// One launch table: a part for each weight set of `pool`, `m` tokens each, on the token rows `x` given by the caller.
/// `same_rows`: every part reads rows 0..m; otherwise part `e` reads rows `e*m..e*m+m`.
#[allow(clippy::too_many_arguments)]
fn table_of(g: &mut Gpu, rows: usize, cols: usize, br: usize, bc: usize, pool: &[(u64, u64)], m: usize, same_rows: bool, x: &[Vec<f32>]) -> Result<Multi> {
    let mut table = vec![]; let (mut first, mut first_row) = (0usize, 0usize);
    for &(w, s) in pool {
        let mut en = entry(Geometry { rows, cols, br, bc, tokens: m }, w, s);
        en[56..60].copy_from_slice(&(first as u32).to_le_bytes()); en[60..64].copy_from_slice(&(first_row as u32).to_le_bytes()); table.extend(en);
        first += rows * m; if !same_rows { first_row += m; }
    }
    let flat: Vec<f32> = x.iter().flatten().copied().collect();
    let xb = Buf::new(g, &le(&flat))?; let tb = Buf::new(g, &table)?; let l = Buf::new(g, &le(&lut()))?;
    Ok(Multi { rows, cols, counts: vec![m; pool.len()], table: tb.address, input: xb.address, lutp: l.address, units: first })
}
/// The six rank-local block-FP8 matrices of the model: label, rows, cols, matrices of one launch (gate and up read the same rows), launches a decode step.
const SHARE_SHAPES: [(&str, usize, usize, usize, f64); 6] = [("gate_up", 1024, 4096, 2, 42.0), ("down", 2048, 2048, 1, 42.0), ("o_proj", 2048, 16384, 1, 11.0),
    ("q_b", 8192, 1536, 1, 11.0), ("q_a", 768, 4096, 1, 11.0), ("kv_a", 256, 4096, 1, 11.0)];
/// `share`: the byte proof of `fp8_rows_share` (2 to 8 tokens a part, every part the same count).
/// 1. Every launch against `expert_group` on the same table, from poisoned outputs between guards: every weight code, every listed shape and block
///    shape of `small_sweep`, the model's shapes with five activation classes; one part, 2 to 4 parts on the same rows, parts on their own rows.
/// 2. A row alone (the one-row tile `fp8_rows_small`, one token) equals that row in the group, for every slot; the same rows in reverse order give
///    each row the bytes it had in its other slot; a matrix alone equals the matrix beside another in one launch.
/// 3. Wrong on purpose: a row alone compared with ANOTHER slot must differ; the reversed launch compared slot by slot must differ; a changed row must change.
/// 4. A changed neighbor leaves every other row's bytes as they were.
/// One recorded launch as a decode step makes it: a resident launch wrapper (`resident_launch_<entry>`) reads its arguments from a command
/// (kind word, 12 reserved bytes, the argument struct at offset 16; 1,040 bytes) and runs the entry's body.
pub(super) fn launch_wrapped(g: &mut Gpu, wrapper: H, kind: u32, arguments: &[u8], grid: [u32; 3], threads: u32, shared: u32) -> Result<()> {
    let table = command_of(g, kind, arguments)?; launch_command(g, wrapper, table, grid, threads, shared)
}
/// The uploaded command of a recorded launch (made before a graph capture) and its launch.
pub(super) fn command_of(g: &mut Gpu, kind: u32, arguments: &[u8]) -> Result<u64> {
    let mut command = vec![0u8; 1040]; command[..4].copy_from_slice(&kind.to_le_bytes()); command[16..16 + arguments.len()].copy_from_slice(arguments);
    g.upload(&command)
}
pub(super) fn launch_command(g: &Gpu, wrapper: H, table: u64, grid: [u32; 3], threads: u32, shared: u32) -> Result<()> {
    let (mut table, mut index) = (table, 0u64);
    g.launch(wrapper, grid, threads, shared, &mut [super::super::ptr(&mut table), super::super::ptr(&mut index)])
}
/// The recorded wrapper of `fp8_rows_share` (Kind 67) against the direct entry on the six served tables at 2 to 8 tokens: same bytes, and the
/// launch itself is accepted (block size, registers, shared memory).
fn share_wrapper(g: &mut Gpu, share: H, wrapper: H) -> Result<()> {
    let mut comparisons = 0u64;
    for &(label, rows, cols, parts, _) in &SHARE_SHAPES { for m in 2..=8usize {
        let fx = build_table(g, rows, cols, 128, 128, &vec![m; parts], 0, 0, 7100 + m as u64 * 3 + rows as u64, None, parts > 1)?;
        let (oa, ob) = (Buf::new(g, &vec![0u8; fx.units * 4])?, Buf::new(g, &vec![0u8; fx.units * 4])?);
        oa.poison(g)?; ob.poison(g)?;
        mlaunch_share(g, share, &fx, oa.address)?;
        let shared = 3 * 16 * 128 + 3 * 128 * 4 + 2 * 16 * 132 * 4 + (if m > 4 { 2 } else { 3 }) * 2 * m.div_ceil(2) * 132 * 4;
        launch_wrapped(g, wrapper, 67, &margs(&fx, ob.address), [(rows.div_ceil(16) * parts) as u32, 1, 1], (32 * m.div_ceil(2) + 128) as u32, shared as u32).map_err(|e| format!("FP8SHARE WRAPPER LAUNCH FAIL {label} tokens={m}: {e}"))?;
        g.sync().map_err(|e| format!("FP8SHARE WRAPPER RUN FAIL {label} tokens={m}: {e}"))?;
        if oa.read(g)? != ob.read(g)? { return Err(format!("FP8SHARE WRAPPER BYTE FAIL {label} tokens={m}")); }
        comparisons += 1; g.clear()?;
    } }
    println!("FP8SHARE WRAPPER PASS resident_launch_fp8_rows_share equals fp8_rows_share comparisons={comparisons}");
    Ok(())
}
fn share_bytes(g: &mut Gpu, reference: H, small: H, share: H) -> Result<()> {
    use std::cell::Cell;
    let started = Instant::now();
    let (comparisons, compared, alone, controls, neighbors) = (Cell::new(0u64), Cell::new(0u64), Cell::new(0u64), Cell::new(0u64), Cell::new(0u64));
    let run = |g: &mut Gpu, f: H, fx: &Multi, which: u32| -> Result<Vec<u8>> {
        let out = Buf::new(g, &vec![0xff; fx.units * 4])?; out.poison(g)?;
        match which { 0 => mlaunch_reference(g, f, fx, out.address)?, 1 => mlaunch_small(g, f, fx, out.address)?, _ => mlaunch_share(g, f, fx, out.address)? }
        g.sync()?;
        let bytes = out.read(g)?;
        if bytes[..16] != GUARD || bytes[bytes.len() - 16..] != GUARD { return Err("FP8SHARE guard changed".into()); }
        Ok(bytes[16..bytes.len() - 16].to_vec())
    };
    let check = |g: &mut Gpu, fx: &Multi, label: &str| -> Result<Vec<u8>> {
        let (x, y) = (run(g, reference, fx, 0)?, run(g, share, fx, 2)?);
        if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i / 4, i)); return Err(format!("FP8SHARE BYTE FAIL {label} rows={} cols={} parts={:?} first_word_and_byte={at:?}", fx.rows, fx.cols, fx.counts)); }
        comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (fx.units * 4) as u64); Ok(y)
    };
    // The tables of one shape: one part, parts on the same rows, parts on their own rows.
    let lists = |m: usize, wide: bool| -> Vec<(usize, bool)> { let mut v = vec![(1usize, false), (2, true)]; if wide { v.extend([(3, true), (4, true), (2, false), (3, false)]); } let _ = m; v };
    // Every weight code (mode 1 gives each row a shifted code ramp), several scale blocks.
    for &(rows, cols, br, bc) in &[(256usize, 512usize, 16usize, 32usize), (300, 272, 1, 16), (257, 128, 128, 128)] {
        for kind in 0..5u32 { for m in 2..=8usize { for (parts, same) in lists(m, true) {
            let fx = build_table(g, rows, cols, br, bc, &vec![m; parts], 1, kind, 11 + kind as u64 + 7 * m as u64, None, same)?; check(g, &fx, "all-codes")?;
        } } g.clear()?; }
    }
    println!("FP8SHARE all-codes ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // Every listed (rows, cols) and block shape of `small_sweep`, 2 to 8 tokens. Four weight and scale sets are shared by the parts of a launch.
    let mut seed = 500u64;
    for &rows in e::FP8_ROWS { for &cols in e::FP8_COLS {
        if rows * cols > e::FP8_MAX_BYTES { continue; }
        let mut r2 = Rng(3 + rows as u64 * 17 + cols as u64);
        let mut wset = vec![];
        for _ in 0..4 { let w: Vec<u8> = (0..rows * cols).map(|_| (r2.next() >> 8) as u8).collect(); wset.push(Buf::new(g, &w)?.address); }
        for &(br, bc) in e::FP8_BLOCKS {
            let mut pool = vec![];
            for &w in &wset {
                let scales: Vec<f32> = (0..rows.div_ceil(br) * cols.div_ceil(bc)).enumerate().map(|(k, _)| { let x = r2.next(); let ex = (x % 24) as i32 - 14; scale_of(k, (1.0 + ((x >> 8) % 1000) as f32 / 1000.0) * 2f32.powi(ex), x) }).collect();
                pool.push((w, Buf::new(g, &le(&scales))?.address));
            }
            for m in 2..=8usize { for (parts, same) in lists(m, m == 2 || m == 5 || m == 8) {
                if !e::fp8_parts_swept(rows, cols, parts) { continue; }
                seed += 1; let fx = build_table(g, rows, cols, br, bc, &vec![m; parts], 0, (seed % 5) as u32, seed, Some(&pool), same)?; check(g, &fx, "sweep")?;
            } }
        }
        g.clear()?;
    } }
    println!("FP8SHARE sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    for &(label, rows, cols, _, _) in &SHARE_SHAPES {
        // Two weight sets with plain scales (no zero, infinity or denormal), so a control that must differ cannot meet two equal NaN words.
        let mut r2 = Rng(77 + rows as u64 * 13 + cols as u64); let mut pool = vec![];
        for _ in 0..2 {
            let w: Vec<u8> = (0..rows * cols).map(|_| { let b = (r2.next() >> 8) as u8; if b & 127 == 127 { b ^ 1 } else { b } }).collect();
            let sc: Vec<f32> = (0..rows.div_ceil(128) * cols.div_ceil(128)).map(|_| 0.001 * (1.0 + r2.unit().abs())).collect();
            pool.push((Buf::new(g, &w)?.address, Buf::new(g, &le(&sc))?.address));
        }
        for m in 2..=8usize {
            // The model's tables against the reference, five activation classes.
            for kind in 0..5u32 { for (parts, same) in lists(m, false) {
                let fx = build_table(g, rows, cols, 128, 128, &vec![m; parts], 0, kind, 900 + rows as u64 + m as u64 + kind as u64, None, same)?; check(g, &fx, "real")?;
            } }
            // `activations` seeds its generator with `seed | 1`: rows need seeds two apart.
            let x: Vec<Vec<f32>> = (0..m).map(|t| activations(0, cols, 4000 + 64 * m as u64 + 2 * t as u64)).collect();
            let word = |bytes: &[u8], at: usize| -> [u8; 4] { [bytes[at * 4], bytes[at * 4 + 1], bytes[at * 4 + 2], bytes[at * 4 + 3]] };
            let group = table_of(g, rows, cols, 128, 128, &pool[..1], m, false, &x)?; let y = check(g, &group, "group")?;
            // A row alone, through the one-row tile, against its slot and (wrong on purpose) against the next slot.
            for t in 0..m {
                let one = table_of(g, rows, cols, 128, 128, &pool[..1], 1, false, &x[t..t + 1])?; let z = run(g, small, &one, 1)?;
                if (0..rows).any(|r| word(&z, r) != word(&y, r * m + t)) { return Err(format!("FP8SHARE ALONE FAIL {label} tokens={m} slot={t}")); }
                alone.set(alone.get() + 1);
                if (0..rows).all(|r| word(&z, r) == word(&y, r * m + (t + 1) % m)) { return Err(format!("FP8SHARE CONTROL did not fail: {label} tokens={m} a row alone equals another slot")); }
                controls.set(controls.get() + 1);
            }
            // The same rows in reverse order: a row keeps its bytes in its other slot; slot by slot the launch must differ.
            let reversed: Vec<Vec<f32>> = x.iter().rev().cloned().collect();
            let back = table_of(g, rows, cols, 128, 128, &pool[..1], m, false, &reversed)?; let yr = check(g, &back, "reversed")?;
            if (0..rows).any(|r| (0..m).any(|t| word(&yr, r * m + t) != word(&y, r * m + m - 1 - t))) { return Err(format!("FP8SHARE SLOT FAIL {label} tokens={m}: a row changed with its slot")); }
            alone.set(alone.get() + 1);
            if yr == y { return Err(format!("FP8SHARE CONTROL did not fail: {label} tokens={m} reversed rows gave the same bytes")); }
            controls.set(controls.get() + 1);
            // One changed row: its own words change, every other row keeps its bytes.
            let j = m / 2; let mut changed = x.clone(); changed[j] = activations(0, cols, 9000 + 2 * m as u64);
            let other = table_of(g, rows, cols, 128, 128, &pool[..1], m, false, &changed)?; let y2 = check(g, &other, "neighbor")?;
            for t in 0..m {
                let same = (0..rows).all(|r| word(&y2, r * m + t) == word(&y, r * m + t));
                if t == j { if same { return Err(format!("FP8SHARE CONTROL did not fail: {label} tokens={m} a changed row kept its bytes")); } controls.set(controls.get() + 1); }
                else { if !same { return Err(format!("FP8SHARE NEIGHBOR FAIL {label} tokens={m} slot={t}")); } neighbors.set(neighbors.get() + 1); }
            }
            // A matrix alone equals the matrix beside another one on the same rows in ONE launch.
            let both = table_of(g, rows, cols, 128, 128, &pool, m, true, &x)?; let y3 = check(g, &both, "two-parts")?;
            if y3[..rows * m * 4] != y[..] { return Err(format!("FP8SHARE PART FAIL {label} tokens={m}: a matrix changed beside another")); }
            alone.set(alone.get() + 1);
            if y3[rows * m * 4..] == y[..] { return Err(format!("FP8SHARE CONTROL did not fail: {label} tokens={m} the second matrix gave the first one's bytes")); }
            controls.set(controls.get() + 1);
        }
        g.clear()?;
    }
    println!("FP8SHARE BYTES PASS comparisons={} output_bytes={} row_alone_and_slot_checks={} controls_failed_as_they_must={} neighbor_checks={} elapsed_s={:.1}",
        comparisons.get(), compared.get(), alone.get(), controls.get(), neighbors.get(), started.elapsed().as_secs_f64());
    Ok(())
}
/// `sharetime`: us a launch by device graph stamps, weights rotated over 256 MB: the one-row tile (one token), the form the host picks today for
/// `m` tokens (`small_choice::group`), and `fp8_rows_share`. The step lines are ESTIMATES: launch time times launches a decode step.
fn share_time(g: &mut Gpu, small: H, multi_fn: H, few_fn: H, share: H, wrappers: [H; 4], quick: bool) -> Result<()> {
    use super::super::{capture_graph, sample, ptr, TIMER};
    let tm = g.module(TIMER.as_bytes())?; let timer = g.function(tm, "kb_stamp")?;
    let mut step = [[0f64; 4]; 9];
    for &(label, rows, cols, parts, calls) in SHARE_SHAPES.iter().take(if quick { 2 } else { SHARE_SHAPES.len() }) {
        let copies = (256usize << 20).div_ceil(rows * cols * parts).clamp(2, 96);
        let mut pools = vec![];
        for c in 0..copies {
            let mut rng = Rng(701 + c as u64); let mut pool = vec![];
            for _ in 0..parts {
                let w: Vec<u8> = (0..rows * cols).map(|_| { let b = (rng.next() >> 8) as u8; if b & 127 == 127 { b ^ 1 } else { b } }).collect();
                let sc: Vec<f32> = (0..rows.div_ceil(128) * cols.div_ceil(128)).map(|_| 0.001 * (1.0 + rng.unit().abs())).collect();
                pool.push((g.upload(&w)?, g.upload(&le(&sc))?));
            }
            pools.push(pool);
        }
        for m in 2..=8usize {
            if quick && !matches!(m, 4 | 6 | 8) { continue; }
            // The host's choice for this table (mirror of `small_choice::group`): 0 small, 1 multi, 2 few.
            let two_wins = matches!((rows, cols), (1024, 4096) | (2048, 2048) | (2048, 16384) | (8192, 1536));
            let current = if parts > 1 || m >= 5 { 1 } else if m == 2 { if two_wins { 2 } else { 0 } } else if rows <= 768 { 0 } else if rows * cols <= 4_194_304 { 2 } else { 1 };
            let mut sets = vec![];
            for (c, pool) in pools.iter().enumerate() {
                let many = build_table(g, rows, cols, 128, 128, &vec![m; parts], 0, 0, 601 + c as u64, Some(pool), parts > 1)?;
                let one = build_table(g, rows, cols, 128, 128, &vec![1; parts], 0, 0, 601 + c as u64, Some(pool), false)?;
                let out = Buf::new(g, &vec![0u8; many.units * 4])?;
                // The same two launches as a decode step records them: the wrapper kernel reads its arguments from a command.
                let (now_kind, share_command) = ([47u32, 48, 49][current], command_of(g, 67, &margs(&many, out.address))?);
                let now_command = command_of(g, now_kind, &margs(&many, out.address))?;
                sets.push((many, one, out, now_command, share_command));
            }
            let jobs = m * parts; let blocks16 = (rows.div_ceil(16) * parts) as u32;
            let now_dims = [((rows.div_ceil(16) * jobs) as u32, 160u32, SMALL_SHARED), ((rows.div_ceil(32) * parts) as u32, 128, 0), (blocks16, 128, 0)][current];
            let share_shared = (3 * 16 * 128 + 3 * 128 * 4 + 2 * 16 * 132 * 4 + (if m > 4 { 2 } else { 3 }) * 2 * m.div_ceil(2) * 132 * 4) as u32;
            let times = g.upload(&[0; 16])?; let mut samples = vec![Vec::new(); 5];
            {
                let g = &*g; let mut graphs = Vec::new();
                for a in 0..5 { graphs.push(capture_graph(g, || {
                    let mut stamp = times; g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut stamp)])?;
                    for (many, one, out, now_command, share_command) in &sets { match (a, current) {
                        (0, _) => mlaunch_small(g, small, one, out.address)?,
                        (1, 0) => mlaunch_small(g, small, many, out.address)?, (1, 1) => mlaunch_multi(g, multi_fn, many, out.address)?, (1, _) => mlaunch_few(g, few_fn, many, out.address)?,
                        (2, _) => mlaunch_share(g, share, many, out.address)?,
                        (3, _) => launch_command(g, wrappers[current], *now_command, [now_dims.0, 1, 1], now_dims.1, now_dims.2)?,
                        _ => launch_command(g, wrappers[3], *share_command, [blocks16, 1, 1], (32 * m.div_ceil(2) + 128) as u32, share_shared)?,
                    } }
                    stamp = times + 8; g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut stamp)])?; Ok(())
                })?); }
                for _ in 0..3 { for graph in &graphs { sample(graph, times)?; } }
                for repeat in 0..11 { for k in 0..5 { let a = (k + repeat) % 5; samples[a].push(sample(&graphs[a], times)?.1 / copies as f64); } }
            }
            // `rule`: the host rule of `fp8_small::share_wins` (the tables and counts where the share entry was faster).
            let rule = matches!((rows, cols, m), (1024, 4096, 2..=6) | (2048, 2048, 2..=8) | (2048, 16384, 2..=5) | (8192, 1536, 2..=8) | (768, 4096, 3..=8) | (256, 4096, 5..=8));
            let [one_us, now_us, share_us, now_wrapped_us, share_wrapped_us] = [median(samples[0].clone()), median(samples[1].clone()), median(samples[2].clone()), median(samples[3].clone()), median(samples[4].clone())];
            for (a, us) in [one_us, now_us, share_us, if rule { share_us } else { now_us }].into_iter().enumerate() { step[m][a] += us * calls / 1e3; }
            println!("FP8SHARE TIME pad={} family={label} rows={rows} cols={cols} parts={parts} tokens={m} one_row_us={one_us:.3} current={} current_us={now_us:.3} share_us={share_us:.3} current_wrapped_us={now_wrapped_us:.3} share_wrapped_us={share_wrapped_us:.3} share_over_one_row={:.3} share_over_current={:.3} host_rule={} copies={copies} clock=graph_stamps",
                SHARE_PAD.load(std::sync::atomic::Ordering::Relaxed), ["fp8_rows_small", "fp8_rows_multi", "fp8_rows_few"][current], share_us / one_us, share_us / now_us, if rule { "share" } else { "current" });
        }
        g.clear()?;
    }
    for (m, s) in step.iter().enumerate().skip(2) {
        if quick { continue; }
        println!("FP8SHARE STEP tokens={m} ESTIMATE ms_a_step (launch us x launches a step): one_row={:.3} current={:.3} share_everywhere={:.3} host_rule={:.3} saved_by_the_rule={:.3}", s[0], s[1], s[2], s[3], s[1] - s[3]);
    }
    Ok(())
}
/// `pair`: one part against two parts of the same shape in ONE launch (two block-FP8 matrices that read the same rows), us a launch, 1 to 8 tokens.
/// Arms: small, multi, few with one part; small, multi, few with two parts. `fp8_rows_few` takes 1 to 4 tokens a part.
/// The two-part multi arm runs the row stage's table (both parts on the same rows); small and few find a part by its input row, so their two-part
/// arms keep a cumulative table (the same work).
fn pair_time(g: &mut Gpu, small: H, multi_fn: H, few_fn: H) -> Result<()> {
    let started = Instant::now();
    println!("FP8SMALL PAIR shape tokens | one part: small_us multi_us few_us | two parts in one launch: small_us multi_us few_us");
    for &(rows, cols, label) in &[(1024usize, 4096usize, "gate_up"), (2048, 2048, "down"), (768, 4096, "q_a"), (256, 4096, "kv_a")] {
        for tokens in 1..=8usize {
            let k = 12;
            let (mut one, mut two, mut same) = (vec![], vec![], vec![]);
            for i in 0..k {
                one.push(build_parts(g, rows, cols, 128, 128, &[tokens], 0, 0, 31 + i as u64, None)?); two.push(build_parts(g, rows, cols, 128, 128, &[tokens, tokens], 0, 0, 131 + i as u64, None)?);
                same.push(build_table(g, rows, cols, 128, 128, &[tokens, tokens], 0, 0, 231 + i as u64, None, true)?);
            }
            let out = Buf::new(g, &vec![0u8; two[0].units * 4])?;
            let mut t = [vec![], vec![], vec![], vec![], vec![], vec![]];
            for repeat in 0..5 { for arm in (0..6).map(|a| (a + repeat) % 6) {
                if arm % 3 == 2 && tokens > 4 { continue; }
                let run = |g: &Gpu, i: usize| -> Result<()> { let fx = if arm < 3 { &one[i % k] } else if arm == 4 { &same[i % k] } else { &two[i % k] }; match arm % 3 { 0 => mlaunch_small(g, small, fx, out.address), 1 => mlaunch_multi(g, multi_fn, fx, out.address), _ => mlaunch_few(g, few_fn, fx, out.address) } };
                for i in 0..k { run(g, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..k { run(g, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / k as f64);
            } }
            let us = |v: &Vec<f64>| if v.is_empty() { "-".to_string() } else { format!("{:.0}", median(v.clone())) };
            println!("FP8SMALL PAIR {label} rows={rows} cols={cols} tokens={tokens} one: small_us={} multi_us={} few_us={} | two: small_us={} multi_us={} few_us={}", us(&t[0]), us(&t[1]), us(&t[2]), us(&t[3]), us(&t[4]), us(&t[5]));
            g.clear()?;
        }
    }
    println!("FP8SMALL PAIR DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    // `shared`: the byte sweep of `fp8_rows_multi` on tables whose parts all read the same rows (`fp8_small::shared_admitted`).
    let shared = mode == "shared";
    let time = mode == "time"; let multi = mode == "multi" || shared; let few = mode == "few";
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("FP8SMALL baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "expert_group")?; let tile = g.function(ma, "fp8_rows_tile")?; let small = g.function(mb, "fp8_rows_small")?; let multi_fn = g.function(mb, "fp8_rows_multi")?; let few_fn = g.function(mb, "fp8_rows_few")?;
    if mode == "pair" { return pair_time(&mut g, small, multi_fn, few_fn); }
    if mode == "share" { let (share, wrapper) = (g.function(mb, "fp8_rows_share")?, g.function(mb, "resident_launch_fp8_rows_share")?); share_bytes(&mut g, reference, small, share)?; return share_wrapper(&mut g, share, wrapper); }
    let wrappers = |g: &mut Gpu| -> Result<[H; 4]> { Ok([g.function(mb, "resident_launch_fp8_rows_small")?, g.function(mb, "resident_launch_fp8_rows_multi")?, g.function(mb, "resident_launch_fp8_rows_few")?, g.function(mb, "resident_launch_fp8_rows_share")?]) };
    if mode == "sharetime" { let share = g.function(mb, "fp8_rows_share")?; let w = wrappers(&mut g)?; return share_time(&mut g, small, multi_fn, few_fn, share, w, false); }
    // `sharepad`: gate and up and down at 4, 6 and 8 tokens with more dynamic shared memory than the kernel needs: where the launch time jumps is
    // where an SM holds one block less.
    if mode == "sharepad" {
        let share = g.function(mb, "fp8_rows_share")?;
        let w = wrappers(&mut g)?;
        for pad in [0usize, 1056, 2112, 3168, 4224, 6336, 10560] { SHARE_PAD.store(pad, std::sync::atomic::Ordering::Relaxed); share_time(&mut g, small, multi_fn, few_fn, share, w, true)?; }
        return Ok(());
    }
    let started = Instant::now();
    if !time {
        let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
        let check = |g: &mut Gpu, fx: &Multi, label: &str| -> Result<()> {
            let out_a = Buf::new(g, &vec![0xff; fx.units * 4])?; let out_b = Buf::new(g, &vec![0xff; fx.units * 4])?;
            out_a.poison(g)?; out_b.poison(g)?;
            mlaunch_reference(g, reference, fx, out_a.address)?; if multi { mlaunch_multi(g, multi_fn, fx, out_b.address)?; } else if few { mlaunch_few(g, few_fn, fx, out_b.address)?; } else { mlaunch_small(g, small, fx, out_b.address)?; } g.sync()?;
            let (x, y) = (out_a.read(g)?, out_b.read(g)?);
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("FP8SMALL BYTE FAIL {label} rows={} cols={} parts={:?} first_word_and_byte={at:?}", fx.rows, fx.cols, fx.counts)); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("FP8SMALL guard changed".into()); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (fx.units * 4) as u64); Ok(())
        };
        // Every weight code (mode 1 gives each row a shifted code ramp), several scale blocks.
        for &(rows, cols, br, bc) in &[(256usize, 512usize, 16usize, 32usize), (300, 272, 1, 16), (257, 128, 128, 128)] {
            for kind in 0..5u32 { let fx = build_table(&mut g, rows, cols, br, bc, if shared { &[3, 3] } else { &[3, 1] }, 1, kind, 11 + kind as u64, None, shared)?; check(&mut g, &fx, "all-codes")?; g.clear()?; }
        }
        println!("FP8SMALL all-codes ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Every listed (rows, cols) and block shape of `small_sweep` against count families that satisfy the host predicate (at most 128 parts, 1 to 8
        // tokens a part) and seeded random lists of the same class. Four weight and scale sets are shared by the parts of a launch (own table entries).
        let top = if multi { 16usize } else if few { 4 } else { 8 };
        let ladder: Vec<usize> = (1..=top).chain(std::iter::repeat_n(1, 8)).collect();
        let mut families: Vec<Vec<usize>> = (1..=top).map(|t| vec![t]).collect();
        families.extend([vec![1, 1, 1, 1], vec![2, 3, top, 1], vec![top; 3], vec![1; 12], ladder]);
        let mut rng = Rng(0x0f58_0f58_9999_1111); let mut seed = 500u64;
        for &rows in e::FP8_ROWS { for &cols in e::FP8_COLS {
            if rows * cols > e::FP8_MAX_BYTES { continue; }
            let mut r2 = Rng(3 + rows as u64 * 17 + cols as u64);
            let mut wset = vec![];
            for _ in 0..4 { let w: Vec<u8> = (0..rows * cols).map(|_| (r2.next() >> 8) as u8).collect(); wset.push(Buf::new(&mut g, &w)?.address); }
            for &(br, bc) in e::FP8_BLOCKS {
                let mut pool = vec![];
                for (i, &w) in wset.iter().enumerate() {
                    let scales: Vec<f32> = (0..rows.div_ceil(br) * cols.div_ceil(bc)).enumerate().map(|(k, _)| { let x = r2.next(); let ex = (x % 24) as i32 - 14; scale_of(k, (1.0 + ((x >> 8) % 1000) as f32 / 1000.0) * 2f32.powi(ex), x) }).collect();
                    let _ = i; pool.push((w, Buf::new(&mut g, &le(&scales))?.address));
                }
                let mut lists = families.clone();
                { let parts = 1 + (rng.next() % 128) as usize; let l: Vec<usize> = (0..parts).map(|_| if rng.next().is_multiple_of(3) { 1 + (rng.next() % top as u64) as usize } else { 1 }).collect(); lists.push(l); }
                // SHARED ROWS: exactly the admitted tables: 2 to 4 parts (each its own weight and scale set of the pool) of 1 to 16 tokens on the same rows.
                if shared { lists = (2..=4usize).flat_map(|parts| (1..=top).map(move |tokens| vec![tokens; parts])).collect(); }
                for counts in &lists {
                    // Large shapes: launches of at most FP8_BIG_MAX_PARTS parts (small_sweep.rs), the weights dominate the time; the predicates admit exactly that.
                    if !e::fp8_parts_swept(rows, cols, counts.len()) { continue; }
                    seed += 1; let fx = build_table(&mut g, rows, cols, br, bc, counts, 0, (seed % 5) as u32, seed, Some(&pool), shared)?; check(&mut g, &fx, "sweep")?;
                }
            }
            g.clear()?;
        } }
        println!("FP8SMALL sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        for &(rows, cols) in &[(1024usize, 4096usize), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096)] {
            for &tokens in &[1usize, 2, 3, 4, 5, 6, 7, 8, 16] { if few && tokens > 4 { continue; } for kind in [0u32, 1, 2, 3, 4] {
                let counts = if shared { vec![tokens; 2] } else { vec![tokens] };
                let fx = build_table(&mut g, rows, cols, 128, 128, &counts, 0, kind, 900 + rows as u64 + tokens as u64 + kind as u64, None, shared)?; check(&mut g, &fx, "real")?; g.clear()?;
            } }
        }
        println!("FP8SMALL BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    println!("FP8SMALL TIME shape tokens | expert_group_us fp8_rows_tile_us fp8_rows_small_us | GB/s | floor_us_at_223GBs");
    // Real rank-local FP8 shapes of the model and their calls in one forward: gate_up 84, down 42, o_proj, q_b, q_a, kv_a 11 each (170 calls a token); kv_a is 256 x 4096.
    let shapes: [(usize, usize, f64, &str); 6] = [(1024, 4096, 84.0, "gate_up"), (2048, 2048, 42.0, "down"), (2048, 16384, 11.0, "o_proj"), (8192, 1536, 11.0, "q_b"), (768, 4096, 11.0, "q_a"), (256, 4096, 11.0, "kv_a")];
    let mut totals: Vec<(usize, f64, f64, f64, f64, f64, f64, f64)> = vec![];
    for &(rows, cols, calls, label) in &shapes {
        for &tokens in &[1usize, 2, 3, 4, 8, 16] {
            let bytes = (rows * cols) as f64;
            let k = ((150e6 / bytes).ceil() as usize).clamp(2, 48);
            let mut fxs = vec![]; for i in 0..k { fxs.push(build_parts(&mut g, rows, cols, 128, 128, &[tokens], 0, 0, 31 + i as u64, None)?); }
            let out = Buf::new(&mut g, &vec![0u8; fxs[0].units * 4])?;
            let mut t = [vec![], vec![], vec![], vec![], vec![]];
            for repeat in 0..5 { for arm in (0..5).map(|a| (a + repeat) % 5) {
                let run = |g: &Gpu, i: usize| -> Result<()> { match arm { 0 => mlaunch_reference(g, reference, &fxs[i % k], out.address), 1 => mlaunch_tile(g, tile, &fxs[i % k], out.address), 2 => mlaunch_small(g, small, &fxs[i % k], out.address), 4 => { if tokens <= 4 { mlaunch_few(g, few_fn, &fxs[i % k], out.address) } else { mlaunch_multi(g, multi_fn, &fxs[i % k], out.address) } }, _ => mlaunch_multi(g, multi_fn, &fxs[i % k], out.address) } };
                for i in 0..k { run(&g, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..k { run(&g, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / k as f64);
            } }
            let [ra, rb, rc, rd, re] = t.map(median);
            let few_us = if tokens <= 4 { format!("{re:.0}") } else { "-".to_string() };
            println!("FP8SMALL TIME {label} rows={rows} cols={cols} tokens={tokens} expert_group_us={ra:.0} fp8_rows_tile_us={rb:.0} fp8_rows_small_us={rc:.0} fp8_rows_multi_us={rd:.0} fp8_rows_few_us={few_us} GBs={:.0}/{:.0}/{:.0}/{:.0} floor_us_at_223GBs={:.0}", bytes / ra / 1e3, bytes / rb / 1e3, bytes / rc / 1e3, bytes / rd / 1e3, bytes / 223e3);
            // The production choice (mirror of small_choice::group for ONE part; the host test of that function pins the rule): small up to 2 tokens, or 3 and 4 tokens over at most 768 rows;
            // few over more than 768 rows at 3 or 4 tokens up to 4,194,304 weight bytes; multi otherwise.
            let chosen = if tokens <= 2 || ((3..=4).contains(&tokens) && rows <= 768) { rc } else if (3..=4).contains(&tokens) && rows > 768 && rows * cols <= 4_194_304 { re } else { rd };
            // The rule before FP8-ORDER-b and FP8-FEW: small up to 2 tokens, multi from 3.
            let before = if tokens <= 2 { rc } else { rd };
            totals.push((tokens, ra * calls / 1e3, rb * calls / 1e3, rc * calls / 1e3, rd * calls / 1e3, bytes / 223e3 * calls / 1e3, chosen * calls / 1e3, before * calls / 1e3));
            g.clear()?;
        }
    }
    for &tokens in &[1usize, 2, 3, 4, 8, 16] {
        let (mut ra, mut rb, mut rc, mut rd, mut f, mut ch, mut bf) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for t in totals.iter().filter(|t| t.0 == tokens) { ra += t.1; rb += t.2; rc += t.3; rd += t.4; f += t.5; ch += t.6; bf += t.7; }
        println!("FP8SMALL PASS tokens={tokens} ms_a_pass: expert_group={ra:.1} fp8_rows_tile={rb:.1} fp8_rows_small={rc:.1} fp8_rows_multi={rd:.1} floor={f:.1} dispatch_before={bf:.1} dispatch_now={ch:.1}");
    }
    println!("FP8SMALL TIME DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}

#[path = "kernel_shared_fp8.rs"] pub(in super::super) mod shared_forms;
