//! Lane ZERO-SKIP device gate: the column-tile NVFP4 kernel `nv_cols` against the row-major `nv_tile`, one process, one
//! device. Same logical weights in both layouts. Byte check of `nv_cols` AND of the scalar body on column tiles against the
//! reference `expert_operation` on row-major bytes, then timing at the decode shapes with a set share of exactly-zero
//! activations. Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
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
/// Activation rows: per 16-column block, 16 values then the block scale. A `share` of the values is +0.0 or -0.0.
fn activations(cols: usize, share: f64, seed: u64) -> (Vec<u8>, usize) {
    let mut r = Rng(seed | 1); let mut v = vec![]; let mut zeros = 0;
    for i in 0..cols / 16 * 17 {
        let (u, s) = (r.unit(), r.next());
        let x = if i % 17 == 16 { 0.25 * (1.0 + u.abs()) } else if ((s >> 11) as f64 / (1u64 << 53) as f64) < share { zeros += 1; if s & 1 == 0 { 0.0 } else { -0.0 } } else if u == 0.0 { 0.5 } else { u * 4.0 };
        v.extend(x.to_le_bytes());
    }
    (v, zeros)
}
/// Row-major codes (rows x cols/2 bytes, low nibble first) to the column-tile layout of nv_cols.
fn column_codes(w: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows / 128 * cols * 64];
    for row in 0..rows { for j in 0..cols {
        let nibble = (w[row * (cols / 2) + j / 2] >> (4 * (j & 1))) & 15;
        out[((row / 128) * cols + j) * 64 + (row % 128) / 2] |= nibble << (4 * (row & 1));
    } }
    out
}
/// Row-major scales (rows x cols/16 bytes) to block-major tiles (128 bytes a block a tile).
fn column_scales(s: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks = cols / 16; let mut out = vec![0u8; rows / 128 * blocks * 128];
    for row in 0..rows { for b in 0..blocks { out[((row / 128) * blocks + b) * 128 + row % 128] = s[row * blocks + b]; } }
    out
}
struct Job { weight: u64, scales: u64, column_weight: u64, column_scales: u64 }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
#[derive(Clone, Copy)] struct Geo { rows: u32, threads: u32, shared: u32, columns: bool }
const REFERENCE: Geo = Geo { rows: 16, threads: 32, shared: 0, columns: false };
const fn tile(r: u32, qn: u32, d: u32) -> Geo { Geo { rows: r, threads: 32 + r * qn, shared: d * (r * qn * 8 + r * (qn.div_ceil(8) * 8) + qn * 80) + 2 * r * (qn * 16 + 4) * 4, columns: false } }
/// Mirror of expert_ops::nv_cols_shared and NV_COLS_THREADS; a column-tile job carries block_rows 128 (NV_TILE_ROWS).
fn columns(cols: usize) -> Geo { Geo { rows: 128, threads: 128, shared: ((cols / 16 * 17 * 4).div_ceil(16) * 16 + 3 * 4 * (1024 + 128)) as u32, columns: true } }
/// The scalar body of `expert_operation` on column tiles (16 rows a warp, as for the row-major packed shapes).
const SCALAR_COLUMNS: Geo = Geo { rows: 16, threads: 32, shared: 0, columns: true };
fn layout(shape: Shape, columns: bool) -> Shape { if columns { Shape { block_rows: 128, ..shape } } else { shape } }
fn table_for(g: &mut Gpu, shape: Shape, jobs: &[Job], input: &[u64], out: u64, columns: bool) -> Result<u64> {
    let mut t = vec![];
    for (i, j) in jobs.iter().enumerate() {
        let (weight, scales) = if columns { (j.column_weight, j.column_scales) } else { (j.weight, j.scales) };
        t.extend(bytes_of(&Arguments { input: [input[i], 4, 4, 4], weight, scales, output: out + (i * shape.rows as usize * 4) as u64, shape: layout(shape, columns) }));
    }
    g.upload(&t)
}
fn launch(g: &Gpu, f: H, v: Geo, shape: Shape, jobs: &[Job], input: u64, table: u64, out: u64) -> Result<()> {
    let (weight, scales) = if v.columns { (jobs[0].column_weight, jobs[0].column_scales) } else { (jobs[0].weight, jobs[0].scales) };
    let args = Arguments { input: [input, 4, 4, 4], weight, scales, output: out, shape: layout(shape, v.columns) };
    let mut l = Launch { single: args, batch: table, count: jobs.len() as u32 };
    g.launch(f, [shape.rows.div_ceil(v.rows) * jobs.len() as u32, 1, 1], v.threads, v.shared, &mut [ptr(&mut l)])
}
pub(crate) fn run(image: &str) -> Result<()> {
    let ptx = std::fs::read(image).map_err(|e| e.to_string())?;
    println!("NVCOLS image={image} sha256={}", sha256::digest(&ptx));
    let mut g = Gpu::new()?; let module = g.module(&ptx)?;
    let reference = g.function(module, "expert_operation")?; let tile_fn = g.function(module, "nv_tile")?; let cols_fn = g.function(module, "nv_cols")?;
    let shares = [0.0f64, 0.0685, 0.2298, 0.5];
    for (rows, cols, count, label) in [(2048usize, 2048usize, 8usize, "down"), (1024, 4096, 16, "gate_up")] {
        let shape = Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 1, block_cols: 16, tokens: 1, global: 0.75, activation: 0.125, bound: 0.0 };
        let mut r = Rng(77 + rows as u64); let mut sets = vec![];
        // Eight rotating weight sets so each call reads its weights from memory (the nv-batch method).
        for _ in 0..8 {
            let mut jobs = vec![];
            for _ in 0..count {
                let w: Vec<u8> = (0..rows * cols / 2).map(|_| r.next() as u8).collect(); let s: Vec<u8> = (0..rows * cols / 16).map(|_| 0x30 + (r.next() % 20) as u8).collect();
                jobs.push(Job { weight: g.upload(&w)?, scales: g.upload(&s)?, column_weight: g.upload(&column_codes(&w, rows, cols))?, column_scales: g.upload(&column_scales(&s, rows, cols))? });
            }
            sets.push(jobs);
        }
        let out_bytes = rows * count * 4;
        let outs = [g.upload(&vec![0xffu8; out_bytes])?, g.upload(&vec![0xffu8; out_bytes])?, g.upload(&vec![0xffu8; out_bytes])?, g.upload(&vec![0xffu8; out_bytes])?];
        for (n, &share) in shares.iter().enumerate() {
            let mut inputs = vec![]; let mut zero_values = 0;
            for j in 0..count { let (a, z) = activations(cols, share, 500 + 31 * j as u64 + 1000 * n as u64); zero_values += z; inputs.push(g.upload(&a)?); }
            let zero_share = zero_values as f64 / (cols * count) as f64;
            // Tables: one a set a layout a destination.
            let mut tables = vec![];
            for jobs in &sets { tables.push([table_for(&mut g, shape, jobs, &inputs, outs[0], false)?, table_for(&mut g, shape, jobs, &inputs, outs[1], false)?, table_for(&mut g, shape, jobs, &inputs, outs[2], true)?, table_for(&mut g, shape, jobs, &inputs, outs[3], true)?]); }
            // Bytes: every set, reference against the production tile and the column prototype.
            let mut compared = 0usize;
            for (jobs, table) in sets.iter().zip(&tables) {
                for out in outs { g.upload_at(out, &vec![0xffu8; out_bytes])?; }
                launch(&g, reference, REFERENCE, shape, jobs, inputs[0], table[0], outs[0])?;
                launch(&g, tile_fn, tile(16, 8, 4), shape, jobs, inputs[0], table[1], outs[1])?;
                launch(&g, cols_fn, columns(cols), shape, jobs, inputs[0], table[2], outs[2])?;
                // The scalar body on column tiles zero-fills first, as the product path does for it.
                g.upload_at(outs[3], &vec![0u8; out_bytes])?; launch(&g, reference, SCALAR_COLUMNS, shape, jobs, inputs[0], table[3], outs[3])?; g.sync()?;
                let (x, y, z) = (g.read(outs[0], out_bytes)?, g.read(outs[1], out_bytes)?, g.read(outs[2], out_bytes)?);
                if x != g.read(outs[3], out_bytes)? { return Err(format!("NVCOLS BYTE FAIL scalar body on column tiles {label} share={share}")); }
                if x != y { return Err(format!("NVCOLS BYTE FAIL nv_tile {label} share={share}")); }
                if x != z { let at = x.iter().zip(&z).position(|(p, q)| p != q).map(|i| i / 4); let differing = x.chunks_exact(4).zip(z.chunks_exact(4)).filter(|(p, q)| p != q).count();
                    return Err(format!("NVCOLS BYTE FAIL nv_cols {label} share={share} first_word={at:?} differing_words={differing} of {}", out_bytes / 4)); }
                compared += out_bytes;
            }
            println!("NVCOLS BYTES PASS {label} rows={rows} cols={cols} jobs={count} zero_share={zero_share:.4} compared_bytes={compared} sets=8 arms=nv_tile,nv_cols,scalar_on_column_tiles");
            // Timing: alternate the arms; each timing is 32 calls that rotate the eight sets.
            let arms: [(&str, Geo, H, usize); 2] = [("nv_tile", tile(16, 8, 4), tile_fn, 1), ("nv_cols", columns(cols), cols_fn, 2)];
            let mut times = [vec![], vec![]];
            for repeat in 0..9 { for k in 0..2 {
                let arm = if repeat % 2 == 0 { k } else { 1 - k }; let (_, geo, f, slot) = arms[arm];
                launch(&g, f, geo, shape, &sets[0], inputs[0], tables[0][slot], outs[slot])?; g.sync()?;
                let calls = 32; let t0 = Instant::now();
                for i in 0..calls { launch(&g, f, geo, shape, &sets[i % 8], inputs[0], tables[i % 8][slot], outs[slot])?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
            } }
            let matrix_mb = (rows * cols / 2 + rows * cols / 16) as f64 * count as f64 / 1e6;
            let read_mb = ((rows * cols / 2) as f64 * (1.0 - zero_share) + (rows * cols / 16) as f64) * count as f64 / 1e6;
            let (tile_us, cols_us) = (median(times[0].clone()), median(times[1].clone()));
            println!("NVCOLS TIME {label} zero_share={zero_share:.4} nv_tile_us={tile_us:.1} nv_tile_GBs={:.0} nv_cols_us={cols_us:.1} nv_cols_matrix_GBs={:.0} nv_cols_read_GBs={:.0} cols_over_tile={:.3} bus_floor_us_at_235={:.1} raw_tile={:?} raw_cols={:?}",
                matrix_mb * 1e3 / tile_us, matrix_mb * 1e3 / cols_us, read_mb * 1e3 / cols_us, cols_us / tile_us, read_mb * 1e3 / 235.0,
                times[0].iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>(), times[1].iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>());
        }
        g.clear()?;
    }
    println!("NVCOLS DONE");
    Ok(())
}
