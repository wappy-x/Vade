//! Same-bytes sweep and timing of the table-form decode attention kernels (`attention_table`: ScoresTable and
//! ValuesTable, keys and values through two 16-byte-entry pointer tables) against the in-place kernels ScoresIndirect
//! and ValuesIndirect (one 8-byte pointer per slot, packed rows) of the baseline image, in one process on one device.
//! Safe code only. The baseline image holds lane D's kernels; the candidate image adds the table bodies.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/causal_ops.rs (repr(C)): Shape 32 bytes, Arguments 80. Op numbers are read from the image's
/// enum order: ScoresIndirect 30, ValuesIndirect 31, ScoresTable 32, ValuesTable 33 (the enum's last four entries).
const SCORES_INDIRECT: u32 = 30;
const VALUES_INDIRECT: u32 = 31;
const SCORES_TABLE: u32 = 32;
const VALUES_TABLE: u32 = 33;
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, output: u64, shape: Shape }
fn bytes_of(a: &Arguments) -> Vec<u8> {
    let mut v = vec![];
    for x in a.input.iter().chain([&a.weight, &a.output]) { v.extend(x.to_le_bytes()); }
    let s = a.shape; for x in [s.op, s.heads, s.dim, s.taps, s.rows, s.cols] { v.extend(x.to_le_bytes()); }
    for x in [s.epsilon, s.bound] { v.extend(x.to_le_bytes()); } v
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// kind 0 plain, 1 wide range, 2 denormals and +-0, 3 large, 4 NaN and inf sprinkled, 5 add-order sensitive,
/// 6 extremes, 7 probabilities-like (positive, sums near one).
fn value(kind: u32, r: &mut Rng) -> f32 {
    let (u, s) = (r.unit(), r.next());
    match kind {
        0 => u,
        1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
        4 => match s % 31 { 0 => f32::NAN, 1 => f32::INFINITY, 2 => f32::NEG_INFINITY, 3 => -f32::NAN, _ => u * 4.0 },
        5 => (if s & 1 == 0 { 1.0 } else { -1.0 }) * 2f32.powi((s >> 8) as i32 % 31),
        6 => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN_POSITIVE, -0.0, 0.0, 3.0e38, 1.0e-45][(s % 9) as usize],
        _ => u.abs() / 64.0,
    }
}
fn floats(kind: u32, n: usize, seed: u64) -> Vec<u8> { let mut r = Rng(seed | 1); (0..n).flat_map(|_| value(kind, &mut r).to_le_bytes()).collect() }
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> { let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() }) }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xa5; self.bytes]) }
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn shape(op: u32, heads: usize, dim: usize, cols: usize, rows: usize, live: usize) -> Shape { Shape { op, heads: heads as u32, dim: dim as u32, taps: live as u32, rows: rows as u32, cols: cols as u32, epsilon: 0.0, bound: 0.0 } }
fn launch(g: &Gpu, f: H, grid: usize, threads: u32, shape: Shape, input0: u64, table: u64, out: u64) -> Result<()> {
    let mut a = Arguments { input: [input0, table, 4, 4], weight: 4, output: out, shape };
    g.launch(f, [grid as u32, 1, 1], threads, 0, &mut [ptr(&mut a)])
}
/// The recorded path: resident_launch_causal replays a 1040-byte command (kind 1, payload = Arguments).
fn recorded(g: &mut Gpu, f: H, grid: usize, threads: u32, shape: Shape, input0: u64, table: u64, out: u64) -> Result<()> {
    let a = Arguments { input: [input0, table, 4, 4], weight: 4, output: out, shape };
    let mut command = vec![0u8; 1040]; command[..4].copy_from_slice(&1u32.to_le_bytes());
    let payload = bytes_of(&a); command[16..16 + payload.len()].copy_from_slice(&payload);
    let mut address = g.upload(&command)?; let mut index = 0u64;
    g.launch(f, [grid as u32, 1, 1], threads, 0, &mut [ptr(&mut address), ptr(&mut index)])
}
fn entry(address: u64, stride: usize) -> Vec<u8> { let mut v = address.to_le_bytes().to_vec(); v.extend((stride as u32).to_le_bytes()); v.extend(0u32.to_le_bytes()); v }
pub(crate) fn run(base_path: &str, candidate_path: &str, time_only: bool) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("ATTNTABLE baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "causal_operation")?; let candidate = g.function(mb, "causal_operation")?; let resident = g.function(mb, "resident_launch_causal")?;
    // Arms: the production dispatch in the candidate image. (name, function, features a lane for values)
    let scores_arms: Vec<(&str, H)> = vec![("prod", candidate)];
    let values_arms: Vec<(&str, H, usize)> = vec![("prod", candidate, 4)];
    let started = Instant::now(); let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
    const KD: usize = 256; const VD: usize = 256;
    // One case. layout 0: every slot a packed row; 1: every slot two separate blocks; 2: mixed by slot parity of a seeded draw.
    let compare = |g: &mut Gpu, heads: usize, rows: usize, live: usize, kinds: [u32; 3], layout: u32, seed: u64| -> Result<()> {
        let row_words = heads * (KD + VD);
        let packed = floats(kinds[0], live * row_words, seed); // keys and values share the packed rows: kind of keys decides
        let mut packed = packed; { let mut r = Rng(seed ^ 0x77); for w in 0..live * heads { for f in KD..KD + VD { let at = (w * (KD + VD) + f) * 4; packed[at..at + 4].copy_from_slice(&value(kinds[1], &mut r).to_le_bytes()); } } }
        let query = floats(kinds[2], heads * KD, seed + 1); let probs = floats(7, heads * rows, seed + 2);
        let arena = Buf::new(g, &packed)?;
        // Separate key and value blocks (head-major, head stride kd / vd), same numbers as the packed rows.
        let (mut key_blocks, mut value_blocks) = (vec![], vec![]);
        for slot in 0..live { for head in 0..heads {
            let at = (slot * heads + head) * (KD + VD) * 4;
            key_blocks.extend_from_slice(&packed[at..at + KD * 4]); value_blocks.extend_from_slice(&packed[at + KD * 4..at + (KD + VD) * 4]); } }
        let (keys, values) = (Buf::new(g, &key_blocks)?, Buf::new(g, &value_blocks)?);
        let mut r = Rng(seed ^ 0x1234);
        let (mut key_table, mut value_table, mut old_table) = (vec![], vec![], vec![]);
        for slot in 0..live {
            let row = arena.address + (slot * row_words * 4) as u64;
            old_table.extend(row.to_le_bytes());
            let blocks = match layout { 0 => false, 1 => true, _ => r.next() & 1 == 0 };
            if blocks { key_table.extend(entry(keys.address + (slot * heads * KD * 4) as u64, KD)); value_table.extend(entry(values.address + (slot * heads * VD * 4) as u64, VD)); }
            else { key_table.extend(entry(row, KD + VD)); value_table.extend(entry(row + (KD * 4) as u64, KD + VD)); }
        }
        let (old, kt, vt) = (g.upload(&old_table)?, g.upload(&key_table)?, g.upload(&value_table)?);
        let (q, p) = (Buf::new(g, &query)?, Buf::new(g, &probs)?);
        let outs = |g: &mut Gpu| -> Result<(Buf, Buf)> { Ok((Buf::new(g, &vec![0xa5; heads * rows * 4])?, Buf::new(g, &vec![0xa5; heads * VD * 4])?)) };
        let (s_base, v_base) = outs(g)?;
        launch(g, reference, (heads * rows).div_ceil(128), 128, shape(SCORES_INDIRECT, heads, KD, VD, rows, live), q.address, old, s_base.address)?;
        launch(g, reference, (heads * VD).div_ceil(128), 128, shape(VALUES_INDIRECT, heads, VD, KD, rows, live), p.address, old, v_base.address)?;
        g.sync()?; let (xs, xv) = (s_base.read(g)?, v_base.read(g)?);
        let (gs, rows_tiles) = (heads * rows.div_ceil(128), rows.div_ceil(128));
        let _ = rows_tiles;
        // Scores arms, direct launch.
        for &(name, f) in scores_arms.iter() {
            let s_new = Buf::new(g, &vec![0xa5; heads * rows * 4])?; s_new.poison(g)?;
            launch(g, f, gs, 128, shape(SCORES_TABLE, heads, KD, VD, rows, live), q.address, kt, s_new.address)?; g.sync()?; let ys = s_new.read(g)?;
            if xs != ys { return Err(format!("ATTNTABLE BYTE FAIL scores arm={name} heads={heads} rows={rows} live={live} kinds={kinds:?} layout={layout} first_byte={:?}", xs.iter().zip(&ys).position(|(p, q)| p != q))); }
            if ys[..16] != GUARD || ys[ys.len() - 16..] != GUARD { return Err("ATTNTABLE guard changed (scores)".to_string()); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (heads * rows * 4) as u64);
        }
        for &(name, f, per) in values_arms.iter() {
            let v_new = Buf::new(g, &vec![0xa5; heads * VD * 4])?; v_new.poison(g)?;
            launch(g, f, heads * (VD / (32 * per)), 32, shape(VALUES_TABLE, heads, VD, KD, rows, live), p.address, vt, v_new.address)?; g.sync()?; let yv = v_new.read(g)?;
            if xv != yv { return Err(format!("ATTNTABLE BYTE FAIL values arm={name} heads={heads} rows={rows} live={live} kinds={kinds:?} layout={layout} first_byte={:?}", xv.iter().zip(&yv).position(|(p, q)| p != q))); }
            if yv[..16] != GUARD || yv[yv.len() - 16..] != GUARD { return Err("ATTNTABLE guard changed (values)".to_string()); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (heads * VD * 4) as u64);
        }
        // The recorded path (production arms only): resident_launch_causal.
        {
            let (s_new, v_new) = outs(g)?; s_new.poison(g)?; v_new.poison(g)?;
            recorded(g, resident, gs, 128, shape(SCORES_TABLE, heads, KD, VD, rows, live), q.address, kt, s_new.address)?;
            recorded(g, resident, heads * (VD / (32 * 4)), 32, shape(VALUES_TABLE, heads, VD, KD, rows, live), p.address, vt, v_new.address)?;
            g.sync()?; let (ys, yv) = (s_new.read(g)?, v_new.read(g)?);
            if xs != ys || xv != yv { return Err(format!("ATTNTABLE BYTE FAIL recorded heads={heads} rows={rows} live={live} kinds={kinds:?} layout={layout}")); }
            comparisons.set(comparisons.get() + 2); compared.set(compared.get() + ((heads * rows + heads * VD) * 4) as u64);
        }
        g.clear()?; Ok(())
    };
    let mut seed = 11u64;
    if !time_only {
        // 1. The real shape (64 heads, 2,051 slots) and its neighbours, every value pattern of keys, values and queries, three layouts.
        for kind in 0..8u32 { for layout in 0..3u32 { seed += 1; compare(&mut g, 64, 2051, 2051, [kind % 7, (kind + 1) % 7, (kind + 3) % 7], layout, seed)?; } }
        println!("ATTNTABLE real shape ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // 2. Every live count 1..=70 and edges, padded rows (live < rows), 1 to 8 heads.
        for live in (1..=70usize).chain([127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 1151, 1152, 1153, 2047, 2048, 2049, 2050, 2051]) {
            for &(heads, rows) in [(1usize, live), (3, live + 5), (8, live.max(1)), (2, live.div_ceil(128) * 128)].iter() {
                seed += 1; compare(&mut g, heads, rows, live, [(seed % 7) as u32, (seed / 3 % 7) as u32, (seed / 5 % 7) as u32], (seed % 3) as u32, seed)?;
            }
        }
        println!("ATTNTABLE live counts ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // 3. Larger: 128 heads, 4,096 rows, and a random spread.
        compare(&mut g, 128, 1100, 1100, [0, 0, 0], 2, 900)?;
        compare(&mut g, 16, 4096, 4096, [5, 5, 5], 1, 901)?;
        let mut r = Rng(0x9e3779b97f4a7c15);
        for _ in 0..80 { seed += 1; let heads = 1 + (r.next() % 70) as usize; let live = 1 + (r.next() % 2100) as usize; let rows = live + (r.next() % 40) as usize;
            if heads * live > 40_000 { continue; } compare(&mut g, heads, rows, live, [(r.next() % 7) as u32, (r.next() % 7) as u32, (r.next() % 7) as u32], (r.next() % 3) as u32, seed)?; }
        println!("ATTNTABLE BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
    } else { println!("ATTNTABLE time only: no byte sweep in this run"); }
    // Timing at the model's shape: 64 heads, 2,051 slots (all live), packed rows read in place. Three arenas rotate so the rows come from memory.
    for &(heads, rows) in [(64usize, 2051usize), (64, 1152)].iter() {
        let row_words = heads * (KD + VD);
        let mut sets = vec![];
        for s in 0..3u64 {
            let arena = Buf::new(&mut g, &floats(0, rows * row_words, 5 + s))?;
            let mut old_table = vec![]; let (mut key_table, mut value_table) = (vec![], vec![]);
            for slot in 0..rows { let row = arena.address + (slot * row_words * 4) as u64; old_table.extend(row.to_le_bytes()); key_table.extend(entry(row, KD + VD)); value_table.extend(entry(row + (KD * 4) as u64, KD + VD)); }
            let (old, kt, vt) = (g.upload(&old_table)?, g.upload(&key_table)?, g.upload(&value_table)?);
            let (q, p) = (Buf::new(&mut g, &floats(0, heads * KD, 9))?, Buf::new(&mut g, &floats(7, heads * rows, 10))?);
            let (so, vo) = (Buf::new(&mut g, &vec![0u8; heads * rows * 4])?, Buf::new(&mut g, &vec![0u8; heads * VD * 4])?);
            sets.push((arena, old, kt, vt, q, p, so, vo));
        }
        // Arms: 0 = old scores, 1 = old values, then each scores arm, then each values arm.
        let narms = 2 + scores_arms.len() + values_arms.len();
        let mut times = vec![vec![]; narms];
        for repeat in 0..7 {
            for k in 0..narms {
                let arm = if repeat % 2 == 0 { k } else { narms - 1 - k };
                let run_one = |g: &Gpu, i: usize| -> Result<()> { let (_, old, kt, vt, q, p, so, vo) = &sets[i % 3];
                    match arm {
                        0 => launch(g, reference, (heads * rows).div_ceil(128), 128, shape(SCORES_INDIRECT, heads, KD, VD, rows, rows), q.address, *old, so.address),
                        1 => launch(g, reference, (heads * VD).div_ceil(128), 128, shape(VALUES_INDIRECT, heads, VD, KD, rows, rows), p.address, *old, vo.address),
                        n if n < 2 + scores_arms.len() => launch(g, scores_arms[n - 2].1, heads * rows.div_ceil(128), 128, shape(SCORES_TABLE, heads, KD, VD, rows, rows), q.address, *kt, so.address),
                        n => { let (_, f, per) = values_arms[n - 2 - scores_arms.len()]; launch(g, f, heads * (VD / (32 * per)), 32, shape(VALUES_TABLE, heads, VD, KD, rows, rows), p.address, *vt, vo.address) }
                    } };
                run_one(&g, 0)?; g.sync()?; let n = 12; let t0 = Instant::now();
                for i in 0..n { run_one(&g, i)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let t = |i: usize| median(times[i].clone());
        println!("ATTNTABLE TIME heads={heads} rows={rows} old scores {:.1} us, old values {:.1} us", t(0), t(1));
        for (n, a) in scores_arms.iter().enumerate() { println!("ATTNTABLE TIME heads={heads} rows={rows} scores arm={} us={:.1}", a.0, t(2 + n)); }
        for (n, a) in values_arms.iter().enumerate() { println!("ATTNTABLE TIME heads={heads} rows={rows} values arm={} us={:.1}", a.0, t(2 + scores_arms.len() + n)); }
        let best_s = (0..scores_arms.len()).map(|n| t(2 + n)).fold(f64::MAX, f64::min); let best_v = (0..values_arms.len()).map(|n| t(2 + scores_arms.len() + n)).fold(f64::MAX, f64::min);
        println!("ATTNTABLE TOKEN heads={heads} rows={rows} 11 layers: old {:.2} ms, production {:.2} ms, best arms {:.2} ms (scores + values kernel time per call times 11)", (t(0) + t(1)) * 11.0 / 1000.0, (t(2) + t(2 + scores_arms.len())) * 11.0 / 1000.0, (best_s + best_v) * 11.0 / 1000.0);
        g.clear()?;
    }
    println!("ATTNTABLE PASS");
    Ok(())
}
