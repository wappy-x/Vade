//! Same-bytes sweep and timing of the block-FP8 decode tile kernel (`fp8_tile`) against the production
//! `expert_operation` one-thread-per-row body (`Op::Fp8Serial`) of glm-next, in one process on one device.
//! Safe code only. The baseline image is the glm-next cubin; the candidate image is the tile build.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/expert_ops.rs (repr(C)): Shape 36 bytes; Arguments 96; LaunchArguments 112. Fp8Serial is 8.
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
/// Production tile geometry (gpu/src/expert_ops.rs FP8_TILE_*; a host test there pins the same numbers).
const PROD: (u32, u32, u32) = (16, 8, 4);
const MAX_ROWS: usize = 16384;
const MAX_COLS: usize = 32768;
/// The shape-only predicate, mirrored from gpu/src/expert_ops.rs `fp8_tile_admits`.
const MAX_JOBS: usize = 8;
fn admits(rows: usize, cols: usize, br: usize, bc: usize, count: usize) -> bool {
    (1..=MAX_JOBS).contains(&count) && (1..=MAX_ROWS).contains(&rows) && (16..=MAX_COLS).contains(&cols) && cols.is_multiple_of(16) && (1..=1024).contains(&br) && bc.is_power_of_two() && (16..=4096).contains(&bc)
}
fn shared_bytes(r: u32, qn: u32, d: u32) -> u32 { let (ck, items) = (qn * 16, r * qn); d * r * ck + d * items * 4 + d * ck * 4 + 3 * ck * 4 + 2 * r * (ck + 4) * 4 }
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// Inputs. kind 0 plain, 1 wide range, 2 denormals and +-0, 3 large, 4 NaN and inf sprinkled, 5 add-order sensitive
/// (random signs, magnitudes 2^0..2^30), 6 BF16 rounding ties and neighbours, 7 all NaN/inf/huge extremes.
fn inputs(kind: u32, cols: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1); let mut v = Vec::with_capacity(cols * 4);
    for _ in 0..cols {
        let (u, s) = (r.unit(), r.next());
        let x = match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
            4 => match s % 29 { 0 => f32::NAN, 1 => f32::INFINITY, 2 => f32::NEG_INFINITY, 3 => -f32::NAN, _ => u * 8.0 },
            5 => (if s & 1 == 0 { 1.0 } else { -1.0 }) * 2f32.powi((s >> 8) as i32 % 31),
            6 => f32::from_bits((((s >> 16) as u32 & 0xffff) << 16) | [0x8000u32, 0x7fff, 0x8001, 0xffff, 0, 1, 0x4000, 0xc000][(s % 8) as usize]),
            _ => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN_POSITIVE, -0.0, 0.0, 3.0e38, 1.0e-45][(s % 9) as usize],
        };
        v.extend(x.to_le_bytes());
    }
    v
}
/// Scales. kind 0 normal positive, 1 includes 0, -0, denormal, huge, inf, NaN, 2 all exactly 1.0, 3 negative and wide.
fn scales(kind: u32, count: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1); let mut v = Vec::with_capacity(count * 4);
    for _ in 0..count {
        let (u, s) = (r.unit(), r.next());
        let x = match kind {
            0 => 0.001953125 * (1.0 + u.abs()),
            1 => match s % 11 { 0 => 0.0, 1 => -0.0, 2 => f32::from_bits((s >> 8) as u32 % 8388607 + 1), 3 => 2f32.powi(100), 4 => f32::INFINITY, 5 => f32::NAN, 6 => -f32::INFINITY, _ => 0.01 * (1.0 + u.abs()) },
            2 => 1.0,
            _ => u * f32::from_bits(((s % 40 + 100) as u32) << 23),
        };
        v.extend(x.to_le_bytes());
    }
    v
}
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> { let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() }) }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xa5; self.bytes]) }
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Job { weight: u64, scales: u64, input: u64 }
/// One launch. `tile`: Some((R, QN, D)) for the tile geometry, None for the baseline (32 threads, 32 rows a block).
fn launch(g: &Gpu, f: H, tile: Option<(u32, u32, u32)>, shape: Shape, jobs: &[Job], table: u64, out: u64, single: bool) -> Result<()> {
    let args = Arguments { input: [jobs[0].input, 4, 4, 4], weight: jobs[0].weight, scales: jobs[0].scales, output: out, shape };
    let mut l = Launch { single: args, batch: if single { 0 } else { table }, count: jobs.len() as u32 };
    let (grid, threads, shared) = match tile {
        Some((r, qn, d)) => (shape.rows.div_ceil(r) * jobs.len() as u32, 32 + r * qn, shared_bytes(r, qn, d)),
        None => (shape.rows.div_ceil(32) * jobs.len() as u32, 32, 0),
    };
    g.launch(f, [grid, 1, 1], threads, shared, &mut [ptr(&mut l)])
}
/// The recorded path: the production graph replays `resident_launch_expert` with a 1040-byte command (kind 2, 16-byte header,
/// the LaunchArguments bytes in the payload) and the same grid, block and dynamic shared bytes as the direct launch.
fn launch_recorded(g: &mut Gpu, f: H, tile: (u32, u32, u32), shape: Shape, jobs: &[Job], table: u64, out: u64, single: bool) -> Result<()> {
    let args = Arguments { input: [jobs[0].input, 4, 4, 4], weight: jobs[0].weight, scales: jobs[0].scales, output: out, shape };
    let mut command = vec![0u8; 1040]; command[..4].copy_from_slice(&2u32.to_le_bytes());
    let mut payload = bytes_of(&args); payload.extend((if single { 0u64 } else { table }).to_le_bytes()); payload.extend((jobs.len() as u32).to_le_bytes());
    command[16..16 + payload.len()].copy_from_slice(&payload);
    let mut address = g.upload(&command)?; let mut index = 0u64;
    let (r, qn, d) = tile;
    g.launch(f, [shape.rows.div_ceil(r) * jobs.len() as u32, 1, 1], 32 + r * qn, shared_bytes(r, qn, d), &mut [ptr(&mut address), ptr(&mut index)])
}
fn table_for(g: &mut Gpu, shape: Shape, jobs: &[Job], out: &Buf) -> Result<u64> {
    let mut t = vec![];
    for (i, j) in jobs.iter().enumerate() { t.extend(bytes_of(&Arguments { input: [j.input, 4, 4, 4], weight: j.weight, scales: j.scales, output: out.address + (i * shape.rows as usize * 4) as u64, shape })); }
    g.upload(&t)
}
/// Cold weights, production recorded launch, and device stamps inside the graph.
/// Counts match the one-row plan. Gate/up is one two-job launch; kv_a has 256 rows.
fn time_recorded(g: &mut Gpu, arms: [H; 2]) -> Result<()> {
    use super::super::{capture_graph, sample, TIMER};
    let timer_module=g.module(TIMER.as_bytes())?;let timer=g.function(timer_module,"kb_stamp")?;
    let mut totals=[0.0;2];
    for (label,rows,cols,count,calls) in [("gate_up",1024usize,4096usize,2usize,42usize),("down",2048,2048,1,42),
        ("o_proj",2048,16384,1,11),("q_b",8192,1536,1,11),("q_a",768,4096,1,11),("kv_a",256,4096,1,11)] {
        let shape=Shape {op:8,rows:rows as u32,cols:cols as u32,block_rows:128,block_cols:128,tokens:1,global:1.0,activation:1.0,bound:0.0};
        let nscales=rows.div_ceil(128)*cols.div_ceil(128);let bytes=count*(rows*cols+nscales*4);
        let copies=(256usize<<20).div_ceil(bytes);let mut commands=vec![];let mut rng=Rng(77);
        for _ in 0..copies {
            // The engine's weight and input views start at allocation boundaries.
            // A 16-byte guard offset changes cache-line traffic in this timing.
            let input=g.upload(&inputs(0,cols,5))?;let mut jobs=vec![];
            for _ in 0..count {
                let weights:Vec<u8>=(0..rows*cols).map(|_|rng.next() as u8 & 0x77).collect();
                jobs.push(Job {weight:g.upload(&weights)?,scales:g.upload(&scales(0,nscales,5))?,input});
            }
            let out=Buf::new(g,&vec![0;rows*count*4])?;let table=table_for(g,shape,&jobs,&out)?;
            let args=Arguments {input:[input,4,4,4],weight:jobs[0].weight,scales:jobs[0].scales,output:out.address,shape};
            let mut payload=bytes_of(&args);payload.extend(table.to_le_bytes());payload.extend((count as u32).to_le_bytes());
            let mut command=vec![0;1040];command[..4].copy_from_slice(&2u32.to_le_bytes());command[16..16+payload.len()].copy_from_slice(&payload);
            commands.push(g.upload(&command)?);
        }
        let stamps=g.upload(&[0;16])?;let mut samples=[vec![],vec![]];
        {
            let capture=|arm:usize|capture_graph(g,|| {
                let mut stamp=stamps;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
                for _ in 0..4 {for &command in &commands {
                    let mut command=command;let mut index=0u64;
                    g.launch(arms[arm],[rows.div_ceil(16) as u32*count as u32,1,1],160,shared_bytes(16,8,4),&mut [ptr(&mut command),ptr(&mut index)])?;
                }}
                stamp=stamps+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;Ok(())
            });
            let graphs=[capture(0)?,capture(1)?];
            for _ in 0..3 {for graph in &graphs {sample(graph,stamps)?;}}
            for repeat in 0..15 {for k in 0..2 {let arm=(repeat+k)%2;let (_,inside)=sample(&graphs[arm],stamps)?;samples[arm].push(inside/(4*copies) as f64);}}
        }
        let medians=[median(samples[0].clone()),median(samples[1].clone())];
        for arm in 0..2 {totals[arm]+=medians[arm]*calls as f64/1000.0;}
        println!("FP8TILE COLD clock=graph_stamps shape={label} rows={rows} cols={cols} count={count} calls={calls} copies={copies} bytes={bytes} main_us={:.3} candidate_us={:.3} main_GBs={:.3} candidate_GBs={:.3} main_samples={:?} candidate_samples={:?}",medians[0],medians[1],bytes as f64/medians[0]/1000.0,bytes as f64/medians[1]/1000.0,samples[0],samples[1]);
        g.clear()?;
    }
    println!("FP8TILE COLD PASS weighted_main_ms={:.6} weighted_candidate_ms={:.6}",totals[0],totals[1]);Ok(())
}
pub(crate) fn run(base_path: &str, candidate_path: &str, time_only: bool) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("FP8TILE baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "expert_operation")?;
    let resident = g.function(mb, "resident_launch_expert")?;
    if time_only {let before=g.function(ma,"resident_launch_expert")?;return time_recorded(&mut g,[before,resident]);}
    // Arm: the production entry in the candidate image at the production geometry.
    let arms: Vec<(&str, (u32, u32, u32), H, bool)> = vec![("prod", PROD, g.function(mb, "expert_operation")?, true)];
    let started = Instant::now(); let (comparisons, compared) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
    let compare = |g: &mut Gpu, rows: usize, cols: usize, br: usize, bc: usize, count: usize, wmode: u32, xkind: u32, skind: u32, single: bool, seed: u64, only_prod: bool| -> Result<()> {
        let outside = !admits(rows, cols, br, bc, count) && only_prod;
        if !outside && !admits(rows, cols, br, bc, count) { return Err(format!("FP8TILE test shape is outside the predicate rows={rows} cols={cols} br={br} bc={bc} count={count}")); }
        let mut r = Rng(seed | 1);
        let shape = Shape { op: 8, rows: rows as u32, cols: cols as u32, block_rows: br as u32, block_cols: bc as u32, tokens: 1, global: 1.0, activation: 1.0, bound: 0.0 };
        let nscales = rows.div_ceil(br) * cols.div_ceil(bc);
        let mut jobs = vec![];
        for j in 0..count {
            let w: Vec<u8> = (0..rows * cols).map(|i| match wmode { 1 => ((i / cols + j) & 255) as u8, 2 => (0x30 + (r.next() % 24) as u8) | if r.next() & 1 == 0 { 0 } else { 0x80 }, 3 => [0x7f, 0xff, 0x00, 0x80, 0x01, 0x81, 0x38, 0xb8][(r.next() % 8) as usize], _ => r.next() as u8 }).collect();
            jobs.push(Job { weight: Buf::new(g, &w)?.address, scales: Buf::new(g, &scales(skind, nscales, seed + 7 * j as u64))?.address, input: Buf::new(g, &inputs(xkind, cols, seed + 31 * j as u64))?.address });
        }
        let out_a = Buf::new(g, &vec![0xa5; rows * count * 4])?; let ta = table_for(g, shape, &jobs, &out_a)?;
        let single = single && count == 1;
        out_a.poison(g)?; launch(g, reference, None, shape, &jobs, ta, out_a.address, single)?; g.sync()?; let x = out_a.read(g)?;
        // Outside the predicate the candidate image must still run the old launch: its expert_operation at 32 threads.
        let old_arm = [("cand_old", PROD, arms[0].2, true)];
        for &(name, geometry, f, exact) in if outside { old_arm.iter() } else { arms.iter() } {
            if !exact || (only_prod && !outside && name != "prod") { continue; }
            let out_b = Buf::new(g, &vec![0xa5; rows * count * 4])?; let tb = table_for(g, shape, &jobs, &out_b)?;
            out_b.poison(g)?; launch(g, f, if outside { None } else { Some(geometry) }, shape, &jobs, tb, out_b.address, single)?; g.sync()?; let y = out_b.read(g)?;
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("FP8TILE BYTE FAIL arm={name} rows={rows} cols={cols} br={br} bc={bc} count={count} wmode={wmode} xkind={xkind} skind={skind} single={single} first_word_and_byte={at:?}")); }
            if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("FP8TILE guard changed arm={name}")); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * count * 4) as u64);
        }
        if !outside {
            // The same job through the recorded path (resident_launch_expert, as the model graph replays it).
            let out_c = Buf::new(g, &vec![0xa5; rows * count * 4])?; let tc = table_for(g, shape, &jobs, &out_c)?;
            out_c.poison(g)?; launch_recorded(g, resident, PROD, shape, &jobs, tc, out_c.address, single)?; g.sync()?; let y = out_c.read(g)?;
            if x != y { return Err(format!("FP8TILE BYTE FAIL recorded path rows={rows} cols={cols} br={br} bc={bc} count={count} wmode={wmode} xkind={xkind} skind={skind} single={single}")); }
            if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err("FP8TILE guard changed recorded path".to_string()); }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * count * 4) as u64);
        }
        g.clear()?; Ok(())
    };
    let sweep = !time_only;
    // 1. Real decode shapes of the model (rank-local), every value pattern.
    let real: [(usize, usize, usize, usize, &str); 6] = [(1024, 4096, 128, 128, "gate_up"), (2048, 2048, 128, 128, "down"), (2048, 16384, 128, 128, "o_proj"), (8192, 1536, 128, 128, "q_b"), (768, 4096, 128, 128, "q_a"), (288, 4096, 128, 128, "kv_a")];
    let mut seed = 100u64;
    if sweep {
    for &(rows, cols, br, bc, _) in real.iter() {
        for xkind in 0..8u32 { for skind in 0..4u32 { seed += 1; compare(&mut g, rows, cols, br, bc, 1, (seed % 4) as u32, xkind, skind, seed.is_multiple_of(2), seed, false)?; } }
    }
    println!("FP8TILE real shapes ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // 2. All 256 weight codes x every value pattern: row r holds code (r + job) % 256 everywhere.
    for xkind in 0..8u32 { for skind in 0..4u32 { seed += 1; compare(&mut g, 512, 256, 1, 16, 1, 1, xkind, skind, true, seed, false)?; compare(&mut g, 300, 144, 128, 128, 2, 1, xkind, skind, false, seed, false)?; } }
    println!("FP8TILE all codes ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // 3. Ragged rows and columns: every row count 1..=70 and edges, many column counts, scale block shapes.
    let row_list: Vec<usize> = (1..=70).chain([95, 96, 97, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2049]).collect();
    let col_list = [16usize, 32, 48, 64, 80, 96, 112, 128, 144, 160, 240, 256, 272, 384, 400, 496, 512, 528, 640, 1024, 1040, 1152, 2048, 2064, 4096, 4112, 8192];
    for &rows in row_list.iter() {
        for (ci, &cols) in col_list.iter().enumerate() {
            if rows * cols > 3_000_000 { continue; }
            seed += 1;
            let (br, bc) = [(1usize, 16usize), (2, 32), (3, 64), (16, 128), (128, 128), (7, 256), (128, 512), (1, 4096)][(seed as usize + ci) % 8];
            let count = [1usize, 1, 2, 3, 5][(seed % 5) as usize];
            compare(&mut g, rows, cols, br, bc, count, (seed % 4) as u32, (seed % 8) as u32, (seed % 4) as u32, seed.is_multiple_of(3), seed, false)?;
        }
    }
    // 3a. Every column count that is a multiple of 16 up to 1040 (all partial last chunks), edge row counts, four scale-block shapes.
    for cols in (16..=1040usize).step_by(16) {
        for &rows in [1usize, 16, 17, 33].iter() {
            for &(br, bc) in [(1usize, 16usize), (128, 128), (3, 64), (16, 512)].iter() {
                seed += 1; compare(&mut g, rows, cols, br, bc, 1, (seed % 4) as u32, (seed % 8) as u32, (seed % 4) as u32, seed.is_multiple_of(2), seed, true)?;
            }
        }
    }
    // 3a2. Every row count 1..=200 at three widths (all ragged row tails of the 16-row blocks).
    for rows in 1..=200usize {
        for &cols in [16usize, 144, 4096].iter() { seed += 1; compare(&mut g, rows, cols, 1 + (seed % 128) as usize, 16 << (seed % 4), 1 + (seed % 2) as usize, (seed % 4) as u32, (seed % 8) as u32, (seed % 4) as u32, seed.is_multiple_of(3), seed, true)?; }
    }
    // 3a3. Batches of 2 to 8 jobs (job-major grid, job table).
    for count in 2..=8usize {
        for &rows in [17usize, 1024].iter() { for &cols in [256usize, 4112].iter() { seed += 1; compare(&mut g, rows, cols, 128, 128, count, (seed % 4) as u32, (seed % 8) as u32, (seed % 4) as u32, false, seed, true)?; } }
    }
    // 3a4. Every batch size 1..=8 (the admitted job counts) over edge rows and three widths, all value patterns in turn.
    for count in 1..=MAX_JOBS {
        for &rows in [1usize, 16, 17, 129, 1024].iter() {
            for &cols in [16usize, 144, 4096].iter() {
                seed += 1; compare(&mut g, rows, cols, 1 + (seed % 128) as usize, 16 << (seed % 4), count, (seed % 4) as u32, (seed % 8) as u32, (seed % 4) as u32, false, seed, true)?;
            }
        }
    }
    // 3a5. More jobs than the predicate admits: shapes that would otherwise take the tile must run the old launch in the new image.
    for &count in [9usize, 10, 12, 16].iter() {
        for &(rows, cols) in [(17usize, 256usize), (1024, 4096), (1, 16)].iter() {
            seed += 1; compare(&mut g, rows, cols, 128, 128, count, (seed % 4) as u32, (seed % 8) as u32, (seed % 4) as u32, false, seed, true)?;
        }
    }
    // 3b. Shapes outside the predicate run the old launch in the candidate image: same bytes as the baseline image.
    for &(rows, cols, br, bc) in [(1usize, 17usize, 1usize, 16usize), (33, 4100, 128, 128), (64, 100, 3, 16), (40, 8, 1, 8), (40, 4096, 128, 100), (40, 4096, 128, 8192), (17, 4096, 1025, 128), (16385, 16, 128, 16), (3, 32800, 128, 128), (50, 4097, 7, 64), (5, 48, 1, 24)].iter() {
        for xkind in [0u32, 4, 5] { seed += 1; compare(&mut g, rows, cols, br, bc, 1 + (seed % 3) as usize, (seed % 4) as u32, xkind, (seed % 4) as u32, seed.is_multiple_of(2), seed, true)?; }
    }
    println!("FP8TILE ragged ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
    // 4. Seeded random shapes over the whole admitted envelope, production arm only.
    let mut r = Rng(0x9e3779b97f4a7c15);
    for _ in 0..1500 {
        seed += 1;
        let rows = 1 + (r.next() % 2200) as usize; let cols = 16 * (1 + (r.next() % if r.next().is_multiple_of(8) { 2048 } else { 300 }) as usize);
        let bc = 16usize << (r.next() % 9); let br = 1 + (r.next() % [1u64, 4, 33, 128, 1024][(r.next() % 5) as usize]) as usize;
        let count = 1 + (r.next() % 6) as usize;
        if rows * cols * count > 40_000_000 { continue; }
        compare(&mut g, rows, cols, br, bc.min(4096), count, (r.next() % 4) as u32, (r.next() % 8) as u32, (r.next() % 4) as u32, r.next().is_multiple_of(2), seed, true)?;
    }
    // 5. Edge of the envelope: largest rows, widest columns.
    compare(&mut g, MAX_ROWS, 256, 128, 128, 1, 0, 0, 0, true, 9001, false)?;
    compare(&mut g, 64, MAX_COLS, 1, 4096, 1, 0, 5, 0, true, 9002, false)?;
    compare(&mut g, 33, MAX_COLS, 128, 16, 2, 0, 1, 1, false, 9003, false)?;
    } else { println!("FP8TILE time only: no byte sweep in this run"); }
    if sweep { println!("FP8TILE BYTES PASS comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64()); }
    // Timing at the real shapes: weights rotate through enough sets (at least 256 MB) that each call reads from memory.
    let mut totals = vec![0.0f64; arms.len() + 1];
    let calls = [84.0f64, 42.0, 11.0, 11.0, 11.0, 11.0];
    for (si, &(rows, cols, br, bc, label)) in real.iter().enumerate() {
        let shape = Shape { op: 8, rows: rows as u32, cols: cols as u32, block_rows: br as u32, block_cols: bc as u32, tokens: 1, global: 1.0, activation: 1.0, bound: 0.0 };
        let mut rng = Rng(77); let nsets = (256usize << 20).div_ceil(rows * cols).clamp(2, 64);
        let (mut sets, nscales) = (vec![], rows.div_ceil(br) * cols.div_ceil(bc));
        for _ in 0..nsets {
            let w: Vec<u8> = (0..rows * cols).map(|_| rng.next() as u8 & 0x77).collect();
            let jobs = vec![Job { weight: Buf::new(&mut g, &w)?.address, scales: Buf::new(&mut g, &scales(0, nscales, 5))?.address, input: Buf::new(&mut g, &inputs(0, cols, 5))?.address }];
            let out = Buf::new(&mut g, &vec![0u8; rows * 4])?; let table = table_for(&mut g, shape, &jobs, &out)?; sets.push((jobs, out, table));
        }
        let mut list: Vec<(String, Option<(u32, u32, u32)>, H)> = vec![("baseline".to_string(), None, reference)];
        for &(name, geometry, f, _) in arms.iter() { list.push((name.to_string(), Some(geometry), f)); }
        let mut times = vec![vec![]; list.len()];
        for repeat in 0..9 {
            for k in 0..list.len() {
                let arm = if repeat % 2 == 0 { k } else { list.len() - 1 - k };
                let (jobs, out, table) = &sets[0]; launch(&g, list[arm].2, list[arm].1, shape, jobs, *table, out.address, true)?; g.sync()?;
                let n = nsets.min(48); let t0 = Instant::now();
                for i in 0..n { let (jobs, out, table) = &sets[i % nsets]; launch(&g, list[arm].2, list[arm].1, shape, jobs, *table, out.address, true)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
        let mb = (rows * cols + nscales * 4) as f64 / 1e6; let base_us = median(times[0].clone());
        for (k, a) in list.iter().enumerate() { let us = median(times[k].clone()); totals[k] += us * calls[si] / 1000.0;
            println!("FP8TILE TIME {label} rows={rows} cols={cols} calls_per_token={} arm={} us={us:.1} ratio_to_baseline={:.2} GBs={:.0}", calls[si], a.0, base_us / us, mb * 1e3 / us); }
        g.clear()?;
    }
    for (k, name) in std::iter::once("baseline").chain(arms.iter().map(|a| a.0)).enumerate() { println!("FP8TILE TOKEN arm={name} ms_per_token={:.2} (sum of the six shapes times their calls per token; kernel time per call, back to back)", totals[k]); }
    println!("FP8TILE PASS");
    Ok(())
}
