//! Device proof and timing of the block FP8 word on tensor cores (`gpu/src/tc_f8.rs`, kernels `fp8_tc`), in one
//! process on one device. The CPU twin is `gpu/src/tc_twin_f8.rs`, included by path: the file the host tests run.
//! The proofs compare the UNROUNDED joined accumulator (role 3), all 32 bits. The final BF16 round hides most
//! differences (the host test `the_word_is_told_apart_from_the_old_chain_and_from_one_range` counts them).
//! Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use crate::tc_twin_f8 as twin;
use std::time::Instant;

const GUARD: [u8; 16] = [0x5a; 16];
/// Mirrors of gpu/src/expert_ops.rs: the roles of a block FP8 shape, and the two operations.
const ROUNDED: u32 = 0;
const QUARTER: u32 = 2;
const UNROUNDED: u32 = 3;
const FP8_ROWS: u32 = 7;
const FP8_SERIAL: u32 = 8;
/// Mirror of gpu/src/expert_ops.rs `fp8_tc_shared`.
const fn shared(warps: u32, tokens: u32, depth: u32) -> u32 { depth * (warps * 16 * 128 + tokens * 128 * 4) }
/// The geometry of the image before this change (main 1508efce): `fp8_rows_small` and the decode tile.
const OLD_SMALL_SHARED: u32 = 4 * 16 * 128 + 4 * 128 * 4 + 4 * 128 * 4 + 3 * 128 * 4 + 2 * 16 * 132 * 4;
const OLD_DECODE_SHARED: u32 = OLD_SMALL_SHARED;
/// Mirror of gpu/src/expert_ops.rs `FP8_TILE_SHARED_FLOOR`.
const DECODE_SHARED_FLOOR: u32 = 10 * 1024;

/// Mirror of gpu/src/expert_ops.rs (repr(C)): Shape 40 bytes, Arguments 96, LaunchArguments 112.
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, rows: u32, cols: u32, block_rows: u32, block_cols: u32, tokens: u32, global: f32, activation: f32, bound: f32, role: u32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, scales: u64, output: u64, shape: Shape }
#[repr(C)] #[derive(Clone, Copy)]
struct Launch { single: Arguments, batch: u64, count: u32 }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
/// The decode table the old entries read (the new ones do not read it).
fn lut(g: &mut Gpu) -> Result<u64> {
    let e2m1 = |n: u8| { let m = u32::from(n & 7); let bits = if m < 2 { m * 0x3f00_0000 } else { ((m / 2 + 126) << 23) | ((m & 1) << 22) }; f32::from_bits(bits | (u32::from(n & 8) << 28)) };
    let mut v = vec![];
    for b in 0..=255u8 { v.push(e2m1(b & 15)); v.push(e2m1(b >> 4)); }
    for b in 0..=255u8 { v.push(twin::e4m3(b)); }
    g.upload(&le(&v))
}
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> { let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() }) }
    fn output(g: &mut Gpu, words: usize) -> Result<Self> { Self::new(g, &vec![0xff; words * 4]) }
    /// The words between the guards. An error when a guard byte changed.
    fn words(&self, g: &Gpu) -> Result<Vec<u32>> {
        let v = g.read(self.base, self.bytes + 32)?;
        if v[..16] != GUARD || v[v.len() - 16..] != GUARD { return Err("FP8TC FAIL: a guard byte beside an output changed".into()); }
        Ok(v[16..v.len() - 16].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
/// One block FP8 matrix on the device, and its bytes on the host for the twin.
struct Matrix { rows: usize, cols: usize, br: usize, bc: usize, codes: Vec<u8>, scales: Vec<f32>, weight: u64, scale: u64 }
impl Matrix {
    fn new(g: &mut Gpu, rows: usize, cols: usize, br: usize, bc: usize, class: usize, salt: usize) -> Result<Self> {
        let codes: Vec<u8> = (0..rows * cols).map(|i| twin::weight(i, salt, class)).collect();
        let scales: Vec<f32> = (0..rows.div_ceil(br) * cols.div_ceil(bc)).map(|i| twin::scale(i, salt)).collect();
        let (w, s) = (Buf::new(g, &codes)?, Buf::new(g, &le(&scales))?);
        Ok(Self { rows, cols, br, bc, codes, scales, weight: w.address, scale: s.address })
    }
    fn row(&self, row: usize) -> (&[u8], &[f32]) { let n = self.cols.div_ceil(self.bc); (&self.codes[row * self.cols..][..self.cols], &self.scales[(row / self.br) * n..][..n]) }
    /// The twin's unrounded accumulator of (row, token row).
    fn twin(&self, row: usize, x: &[f32]) -> u32 { let (codes, scales) = self.row(row); twin::accumulator(codes, scales, self.bc, x) }
    fn shape(&self, op: u32, tokens: usize, role: u32) -> Shape {
        Shape { op, rows: self.rows as u32, cols: self.cols as u32, block_rows: self.br as u32, block_cols: self.bc as u32, tokens: tokens as u32, global: 1.0, activation: 1.0, bound: 0.0, role }
    }
    /// One `GroupEntry` (80 bytes): this matrix on `tokens` token rows from input row 0.
    fn entry(&self, tokens: usize, role: u32) -> Vec<u8> {
        let mut v = vec![];
        for x in [FP8_ROWS, self.rows as u32, self.cols as u32, self.br as u32, self.bc as u32, tokens as u32] { v.extend(x.to_le_bytes()); }
        for x in [1.0f32, 1.0, 0.0] { v.extend(x.to_le_bytes()); }
        v.extend(role.to_le_bytes()); v.extend(self.weight.to_le_bytes()); v.extend(self.scale.to_le_bytes());
        for x in [0u32, 0, (tokens * self.cols) as u32, (self.rows * self.cols) as u32, (self.scales.len() * 4) as u32, 0] { v.extend(x.to_le_bytes()); }
        v
    }
}
/// The twin's words of `rows` x `xs`, `[row index][token]`, on up to 16 host threads.
fn twin_words(m: &Matrix, rows: &[usize], xs: &[Vec<f32>]) -> Vec<Vec<u32>> {
    let mut out = vec![vec![0u32; xs.len()]; rows.len()];
    let per = rows.len().div_ceil(16).max(1);
    std::thread::scope(|s| {
        for (some, results) in rows.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(move || { for (&row, result) in some.iter().zip(results) { for (x, word) in xs.iter().zip(result.iter_mut()) { *word = m.twin(row, x); } } });
        }
    });
    out
}
/// `count` token rows of the hard classes of the twin (class = row % 7).
fn pool(cols: usize, count: usize, salt: usize) -> Vec<Vec<f32>> {
    (0..count).map(|p| (0..cols).map(|c| twin::activation(p * cols + c, salt, p % twin::CLASSES)).collect()).collect()
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind { Decode, Generic, Small, Few, Multi, Tile, Tile3, OldDecode, OldSmall, OldFew, OldMulti, OldTile }
struct Entries { operation: H, decode: H, generic: H, small: H, few: H, multi: H, tile: H, tile3: Option<H>, share: Option<H> }
/// Mirrors of gpu/src/fp8_rows.rs: the third form of the prompt tile is launched from this many token rows, where the
/// scale blocks are 128 columns and a multiple of 16 rows and the columns are whole stages of 128.
const THIRD_MIN_TOKENS: usize = 17;
fn third_takes(m: &Matrix) -> bool { m.bc == 128 && m.cols % 128 == 0 && m.br % 16 == 0 }
impl Entries {
    fn load(g: &Gpu, module: H) -> Result<Self> {
        // The decode tile: the entry of its own (`fp8_tile`, the served kind Fp8Tile) where the image has it; main runs
        // its decode tile inside `expert_operation`.
        let operation = g.function(module, "expert_operation")?;
        Ok(Self { operation, decode: g.function(module, "fp8_tile").unwrap_or(operation), generic: g.function(module, "expert_group")?, small: g.function(module, "fp8_rows_small")?,
            few: g.function(module, "fp8_rows_few")?, multi: g.function(module, "fp8_rows_multi")?, tile: g.function(module, "fp8_rows_tile")?, tile3: g.function(module, "fp8_rows_tile3").ok(),
            share: g.function(module, "fp8_rows_share").ok() })
    }
    fn of(&self, kind: Kind) -> H {
        match kind { Kind::Decode => self.decode, Kind::OldDecode => self.operation, Kind::Generic => self.generic, Kind::Small | Kind::OldSmall => self.small, Kind::Few | Kind::OldFew => self.few,
            Kind::Multi | Kind::OldMulti => self.multi, Kind::Tile | Kind::OldTile => self.tile, Kind::Tile3 => self.tile3.expect("the image has no fp8_rows_tile3") }
    }
}
/// Mirrors of gpu/src/expert_ops.rs: `fp8_row_shared` and `fp8_row_warps` (FP8_ROW_WARPS = 8, limit 48 KiB).
fn row_shared(warps: u32, blocks: u32, tokens: u32, raw: bool) -> u32 {
    let range_shift = if raw { 0 } else { 2 };
    let per = warps >> range_shift;
    if per <= 1 { return 2048; }
    let most = (((blocks + (1 << range_shift) - 1) >> range_shift) + per - 1) >> per.trailing_zeros();
    2048 + (1 << range_shift) * (per - 1) * most * 8 * tokens.div_ceil(2) * 16
}
fn row_warps(blocks: u32, tokens: u32, raw: bool) -> u32 {
    let mut warps = 8;
    while warps > 4 && row_shared(warps, blocks, tokens, raw) > 48 * 1024 { warps /= 2; }
    warps
}
/// Grid, threads and dynamic shared bytes of a launch: the mirrors of gpu/src/expert_ops.rs (`fp8_tile_block`),
/// gpu/src/fp8_small.rs and gpu/src/fp8_rows.rs for the new image, and of main 1508efce for the old one.
fn geometry(kind: Kind, m: &Matrix, tokens: usize, role: u32) -> ([u32; 3], u32, u32) {
    let (r, t, blocks) = (m.rows as u32, tokens as u32, (m.cols as u32).div_ceil(m.bc as u32));
    match kind {
        // The served decode launch asks for at least FP8_TILE_SHARED_FLOOR bytes (gpu/src/expert_ops.rs): the kernel then
        // runs in the shared-memory configuration of the kernels the step runs beside it.
        Kind::Decode => { let warps = row_warps(blocks, 1, role == QUARTER); ([r.div_ceil(16), 1, 1], 32 * warps, row_shared(warps, blocks, 1, role == QUARTER).max(DECODE_SHARED_FLOOR)) }
        Kind::Generic => ([(r * t).div_ceil(32), 1, 1], 32, 0),
        Kind::Small | Kind::Few | Kind::Multi => { let warps = row_warps(blocks, t.min(8), false); ([t.div_ceil(8) * r.div_ceil(16), 1, 1], 32 * warps, row_shared(warps, blocks, t.min(8), false)) }
        Kind::Tile => ([r.div_ceil(128), t.div_ceil(32), 1], 256, shared(8, t.min(32), 3)),
        // gpu/src/fp8_rows.rs `third`: 8 warps of 16 rows and 4 tiles of 8 token rows a block, the operands of one stage.
        Kind::Tile3 => ([r.div_ceil(128), t.div_ceil(32), 1], 256, 8192),
        Kind::OldDecode => ([r.div_ceil(16), 1, 1], 160, OLD_DECODE_SHARED),
        Kind::OldSmall => ([r.div_ceil(16) * t, 1, 1], 160, OLD_SMALL_SHARED),
        Kind::OldFew => ([r.div_ceil(16), 1, 1], 128, 0),
        Kind::OldMulti => ([r.div_ceil(32), 1, 1], 128, 0),
        Kind::OldTile => ([t.div_ceil(64), r.div_ceil(64), 1], 256, 0),
    }
}
/// One token row through the decode entry (`expert_operation`, Fp8Serial).
fn decode(g: &Gpu, e: &Entries, kind: Kind, m: &Matrix, input: u64, out: u64, role: u32) -> Result<()> {
    let mut l = Launch { single: Arguments { input: [input, 4, 4, 4], weight: m.weight, scales: m.scale, output: out, shape: m.shape(FP8_SERIAL, 1, role) }, batch: 0, count: 1 };
    let (grid, threads, bytes) = geometry(kind, m, 1, role);
    g.launch(e.of(kind), grid, threads, bytes, &mut [ptr(&mut l)])
}
/// The bytes of one launch record of a batch (`Arguments`: seven 64-bit words, then the ten 32-bit words of the shape).
fn job_bytes(m: &Matrix, input: u64, out: u64, role: u32) -> Vec<u8> {
    let s = m.shape(FP8_SERIAL, 1, role);
    let mut a = vec![];
    for x in [input, 4, 4, 4, m.weight, m.scale, out] { a.extend(x.to_le_bytes()); }
    for x in [s.op, s.rows, s.cols, s.block_rows, s.block_cols, s.tokens] { a.extend(x.to_le_bytes()); }
    for x in [s.global, s.activation, s.bound] { a.extend(x.to_le_bytes()); }
    a.extend(s.role.to_le_bytes());
    a
}
/// One token row through TWO matrices in ONE launch of a decode entry (the served form of an expert's gate and up
/// matrices: a batch of two jobs, `batch` on the device holds their two launch records). The grid is the two jobs'
/// row tiles; threads and shared bytes are those of one job.
fn decode_pair(g: &Gpu, entry: H, grid: [u32; 3], threads: u32, bytes: u32, m: &Matrix, input: u64, out: u64, batch: u64) -> Result<()> {
    let mut l = Launch { single: Arguments { input: [input, 4, 4, 4], weight: m.weight, scales: m.scale, output: out, shape: m.shape(FP8_SERIAL, 1, ROUNDED) }, batch, count: 2 };
    g.launch(entry, [2 * grid[0], grid[1], grid[2]], threads, bytes, &mut [ptr(&mut l)])
}
/// `tokens` token rows through a row-group entry. `table` holds `m.entry(tokens, role)`.
fn group(g: &Gpu, e: &Entries, kind: Kind, m: &Matrix, table: u64, input: u64, out: u64, lutp: u64, tokens: usize) -> Result<()> {
    let mut a = vec![];
    for x in [table, input, out, lutp] { a.extend(x.to_le_bytes()); }
    for x in [1u32, (m.rows * tokens) as u32, 1, m.cols as u32] { a.extend(x.to_le_bytes()); }
    let (grid, threads, bytes) = geometry(kind, m, tokens, ROUNDED);
    g.launch(e.of(kind), grid, threads, bytes, &mut [a.as_mut_ptr().cast()])
}
/// Where main launches its shared-read entry `fp8_rows_share` for ONE part (`fp8_small::share_wins` of main 27276fcc0f):
/// one read and one decode of a weight for 2 to 8 token rows, the scalar word.
fn share_wins(rows: usize, cols: usize, tokens: usize) -> bool {
    matches!((rows, cols, tokens), (2048, 2048, 2..=8) | (2048, 16384, 2..=5) | (8192, 1536, 2..=8) | (768, 4096, 3..=8) | (256, 4096, 5..=8))
}
/// `tokens` token rows (2 to 8) of one part through `fp8_rows_share` of an image: the grid, threads and dynamic shared
/// bytes of `fp8_small::share_blocks`, `share_threads` and `share_shared`.
fn share(g: &Gpu, entry: H, m: &Matrix, table: u64, input: u64, out: u64, lutp: u64, tokens: usize) -> Result<()> {
    let mut a = vec![];
    for x in [table, input, out, lutp] { a.extend(x.to_le_bytes()); }
    for x in [1u32, (m.rows * tokens) as u32, 1, m.cols as u32] { a.extend(x.to_le_bytes()); }
    let bytes = 3 * 16 * 128 + 3 * 128 * 4 + 2 * 16 * 132 * 4 + (if tokens > 4 { 2 } else { 3 }) * 2 * tokens.div_ceil(2) * 132 * 4;
    g.launch(entry, [(m.rows as u32).div_ceil(16), 1, 1], (32 * tokens.div_ceil(2) + 128) as u32, bytes as u32, &mut [a.as_mut_ptr().cast()])
}
/// The device words of one launch, `[row * tokens + token]`. The output is poisoned before the launch.
fn words(g: &mut Gpu, e: &Entries, kind: Kind, m: &Matrix, input: u64, lutp: u64, tokens: usize, role: u32) -> Result<Vec<u32>> {
    let out = Buf::output(g, m.rows * tokens)?;
    if matches!(kind, Kind::Decode | Kind::OldDecode) { decode(g, e, kind, m, input, out.address, role)?; }
    else { let table = g.upload(&m.entry(tokens, role))?; group(g, e, kind, m, table, input, out.address, lutp, tokens)?; }
    g.sync()?; out.words(g)
}
/// Every word of `got` equals `want(i)`, or an error that names the first word that differs.
fn same(what: &str, got: &[u32], n: usize, want: impl Fn(usize) -> u32) -> Result<()> {
    if got.len() != n { return Err(format!("FP8TC FAIL {what}: {} words, expected {n}", got.len())); }
    let bad: Vec<usize> = (0..n).filter(|&i| got[i] != want(i)).collect();
    match bad.first() { Some(&i) => Err(format!("FP8TC FAIL {what}: {} of {n} words differ; first at word {i}: device {:08x}, expected {:08x}", bad.len(), got[i], want(i))), None => Ok(()) }
}
/// The token rows of a group of `size`: slot s holds pool row `(s * step + size) % pool`, so a pool row meets other
/// slots, other neighbours and other group sizes in every group.
fn slots(size: usize, pool: usize) -> Vec<usize> { let step = [1usize, 3, 5, 9, 11, 13][size % 6]; (0..size).map(|s| (s * step + size) % pool).collect() }
fn gather(xs: &[Vec<f32>], order: &[usize]) -> Vec<u8> { order.iter().flat_map(|&p| le(&xs[p])).collect() }

struct Tally { checks: u64, words: u64 }
/// Part 1: the device equals the twin, every row and every token, on small matrices of many forms, through every
/// entry: the decode entry (one row), the three small entries, the prompt tile and the generic one-warp body.
fn twin_sweep(g: &mut Gpu, e: &Entries, tally: &mut Tally) -> Result<()> {
    let started = Instant::now();
    // The measured zero-term counterexample: a large input at a zero weight must not
    // discard low bits of the only nonzero product. Compare raw FP32, before BF16.
    let mut m = Matrix::new(g, 1, 16, 1, 16, 0, 1)?;
    m.codes.fill(0); m.codes[1] = 0x3f; m.scales.fill(1.0);
    g.upload_at(m.scale, &le(&m.scales))?;
    let lutp = lut(g)?;
    for zero in [0u8, 0x80] {
        m.codes[0] = zero; g.upload_at(m.weight, &m.codes)?;
        let mut x = vec![0.0f32; 16]; x[0] = 2f32.powi(22); x[1] = 1.9921875 * 2f32.powi(-120);
        let want = (1.875 * x[1]).to_bits();
        if m.twin(0, &x) != want { return Err("FP8TC zero-term analytical reference failed".into()); }
        let input = Buf::new(g, &le(&x))?;
        for kind in [Kind::Decode, Kind::Small, Kind::Few, Kind::Multi, Kind::Tile, Kind::Generic] {
            let got = words(g, e, kind, &m, input.address, lutp, 1, UNROUNDED)?;
            same(&format!("zero-term regression {kind:?} zero={zero:02x}"), &got, 1, |_| want)?;
            tally.checks += 1; tally.words += 1;
        }
    }
    println!("FP8TC ZERO TERM PASS 12 raw words, six entries, signed zero weights");
    g.clear()?;
    // The third tile form has the tensor core decode its weights (`fp8_tc::tile3_body`). The same counterexample on a
    // form it takes (one scale block of 128 columns), signed zero weights: a zero weight must bring no term.
    let mut m = Matrix::new(g, 1, 128, 16, 128, 0, 1)?;
    m.codes.fill(0); m.codes[1] = 0x3f; m.scales.fill(1.0);
    g.upload_at(m.scale, &le(&m.scales))?;
    let lutp = lut(g)?;
    for zero in [0u8, 0x80] {
        for at in [0usize, 2, 17, 127] {
            m.codes.fill(0); m.codes[1] = 0x3f; m.codes[at] = zero; g.upload_at(m.weight, &m.codes)?;
            let mut x = vec![0.0f32; 128]; x[at] = 2f32.powi(22); x[1] = 1.9921875 * 2f32.powi(-120);
            let want = (1.875 * x[1]).to_bits();
            if m.twin(0, &x) != want { return Err("FP8TC zero-term analytical reference failed (third form)".into()); }
            let input = Buf::new(g, &le(&x))?;
            for kind in [Kind::Tile, Kind::Tile3] {
                let got = words(g, e, kind, &m, input.address, lutp, 1, UNROUNDED)?;
                same(&format!("zero-term regression {kind:?} zero={zero:02x} at={at}"), &got, 1, |_| want)?;
                tally.checks += 1; tally.words += 1;
            }
        }
    }
    g.clear()?;
    // Every weight code of the contract (all but 0x7f and 0xff) in every place of a lane: 64 rows of 256 columns walk
    // the 254 codes, so each code meets each of the 16 columns of a step in both rows of a lane pair. The third form
    // (the tensor core's decode) and the first form (the integer decode) against the twin, all 32 bits.
    let valid: Vec<u8> = (0..=255u8).filter(|&c| c != 0x7f && c != 0xff).collect();
    let mut m = Matrix::new(g, 64, 256, 16, 128, 0, 2)?;
    // For one column of a step (col % 16) the index r * 16 + col / 16 takes 512 values in a row: every code twice. The
    // second 32 rows are the first 32 moved by 8 rows: the other row of the lane pair.
    for row in 0..64 { for col in 0..256 {
        let r = if row < 32 { row } else { (row - 32 + 8) % 32 };
        m.codes[row * 256 + col] = valid[(r * 16 + col / 16 + 7 * (col % 16)) % valid.len()];
    } }
    g.upload_at(m.weight, &m.codes)?;
    let mut seen = [[false; 32]; 256];
    for row in 0..64 { for col in 0..256 { seen[m.codes[row * 256 + col] as usize][col % 16 + 16 * usize::from(row % 16 >= 8)] = true; } }
    let places = valid.iter().map(|&c| seen[c as usize].iter().filter(|&&s| s).count()).min().unwrap_or(0);
    if places < 32 { return Err(format!("FP8TC all-codes fixture: a code meets only {places} of the 32 places of a lane pair (16 columns of a step, two rows)")); }
    let lutp = lut(g)?;
    let xs = pool(256, 40, 9);
    let all: Vec<usize> = (0..64).collect();
    let want = twin_words(&m, &all, &xs);
    for size in [1usize, 7, 33] {
        let order = slots(size, xs.len());
        let input = Buf::new(g, &gather(&xs, &order))?;
        for kind in [Kind::Tile, Kind::Tile3] {
            let got = words(g, e, kind, &m, input.address, lutp, size, UNROUNDED)?;
            same(&format!("every weight code, {kind:?} entry, tokens={size}"), &got, 64 * size, |i| want[i / size][order[i % size]])?;
            tally.checks += 1; tally.words += (64 * size) as u64;
        }
    }
    println!("FP8TC THIRD FORM DECODE PASS: zero terms (16 raw words, signed zero weights at 4 places) and all 254 weight codes in all 16 columns of a step and both rows of a lane pair, third form = first form = twin");
    g.clear()?;
    let forms: [(usize, usize, usize, usize); 14] = [(1, 16, 1, 16), (5, 48, 3, 16), (16, 128, 16, 128), (17, 144, 128, 128), (33, 256, 32, 64), (47, 384, 128, 128), (40, 1536, 128, 128),
        (31, 2048, 128, 128), (20, 4096, 128, 128), (9, 4096, 1, 4096), (64, 2048, 16, 16), (130, 640, 128, 128), (200, 2048, 128, 128), (3, 16384, 128, 128)];
    for (salt, &(rows, cols, br, bc)) in forms.iter().enumerate() {
        let m = Matrix::new(g, rows, cols, br, bc, salt % twin::WEIGHT_CLASSES, salt + 1)?;
        let lutp = lut(g)?;
        let xs = pool(cols, 40, salt + 1);
        let all: Vec<usize> = (0..rows).collect();
        let want = twin_words(&m, &all, &xs);
        let label = format!("rows={rows} cols={cols} block={br}x{bc}");
        // One row alone, every activation class: the unrounded word, the rounded word, the raw word of a quarter job.
        for p in 0..twin::CLASSES {
            let x = Buf::new(g, &le(&xs[p]))?;
            let got = words(g, e, Kind::Decode, &m, x.address, lutp, 1, UNROUNDED)?;
            same(&format!("decode entry, unrounded, {label} class={p}"), &got, rows, |r| want[r][p])?;
            let got = words(g, e, Kind::Decode, &m, x.address, lutp, 1, ROUNDED)?;
            same(&format!("decode entry, rounded, {label} class={p}"), &got, rows, |r| twin::word(want[r][p]))?;
            if cols % bc == 0 {
                let got = words(g, e, Kind::Decode, &m, x.address, lutp, 1, QUARTER)?;
                same(&format!("decode entry, raw quarter, {label} class={p}"), &got, rows, |r| { let (codes, scales) = m.row(r); twin::range(codes, scales, bc, &xs[p]) })?;
                tally.checks += 1; tally.words += rows as u64;
            }
            tally.checks += 2; tally.words += 2 * rows as u64;
        }
        for size in [1usize, 2, 3, 7, 8, 9, 16, 17, 31, 32, 33, 40] {
            let order = slots(size, xs.len());
            let input = Buf::new(g, &gather(&xs, &order))?;
            let mut kinds = vec![Kind::Small, Kind::Few, Kind::Multi, Kind::Tile, Kind::Generic];
            if third_takes(&m) { kinds.push(Kind::Tile3); }
            for kind in kinds {
                let got = words(g, e, kind, &m, input.address, lutp, size, UNROUNDED)?;
                same(&format!("{kind:?} entry, unrounded, {label} tokens={size}"), &got, rows * size, |i| want[i / size][order[i % size]])?;
                tally.checks += 1; tally.words += (rows * size) as u64;
            }
            for kind in [Kind::Tile, Kind::Tile3] {
                if kind == Kind::Tile3 && !third_takes(&m) { continue; }
                let got = words(g, e, kind, &m, input.address, lutp, size, ROUNDED)?;
                same(&format!("{kind:?} entry, rounded, {label} tokens={size}"), &got, rows * size, |i| twin::word(want[i / size][order[i % size]]))?;
                tally.checks += 1; tally.words += (rows * size) as u64;
            }
        }
        g.clear()?;
    }
    println!("FP8TC TWIN ok: device = CPU twin on {} matrix forms, 7 activation classes, 4 weight classes; entries decode, small, few, multi, tile, generic, and the third tile form on the 8 forms it takes; checks={} words={} elapsed_s={:.1}",
        forms.len(), tally.checks, tally.words, started.elapsed().as_secs_f64());
    Ok(())
}
/// The model's block FP8 shapes (a rank's part of each matrix). `quick`: the first two.
const SHAPES: [(&str, usize, usize); 6] = [("down", 2048, 2048), ("gate_up", 1024, 4096), ("o_proj", 2048, 16384), ("q_b", 8192, 1536), ("q_a", 768, 4096), ("kv_a", 288, 4096)];
/// Part 2, on the model's shapes: (a) the device equals the twin on sampled rows of a row alone; (b) a row alone
/// equals the row in every group size, slot and neighbour set, through every entry, all rows, all 32 bits;
/// (c) two wrong-on-purpose controls fail.
fn groups(g: &mut Gpu, e: &Entries, old: &Entries, tally: &mut Tally, quick: bool) -> Result<()> {
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        if quick && index >= 2 { break; }
        let started = Instant::now();
        let m = Matrix::new(g, rows, cols, 128, 128, 3, 100 + index)?;
        let lutp = lut(g)?;
        let xs = pool(cols, 28, 100 + index);
        // A row alone: the decode entry, one launch a pool row.
        let mut alone = vec![];
        for x in &xs { let xb = Buf::new(g, &le(x))?; alone.push(words(g, e, Kind::Decode, &m, xb.address, lutp, 1, UNROUNDED)?); }
        // (a) The twin on sampled rows.
        let mut sample: Vec<usize> = [0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, rows / 2, rows - 33, rows - 17, rows - 16, rows - 2, rows - 1].into_iter().filter(|&r| r < rows).collect();
        let mut rng = Rng(77 + index as u64);
        for _ in 0..13 { sample.push((rng.next() % rows as u64) as usize); }
        sample.sort_unstable(); sample.dedup();
        let want = twin_words(&m, &sample, &xs);
        let (mut differ, mut one_range) = (0usize, 0usize);
        for (i, &row) in sample.iter().enumerate() { for p in 0..xs.len() {
            if alone[p][row] != want[i][p] { differ += 1; }
            let (codes, scales) = m.row(row);
            if alone[p][row] != twin::range(codes, scales, 128, &xs[p]) { one_range += 1; }
        } }
        let sampled = sample.len() * xs.len();
        if differ != 0 { return Err(format!("FP8TC FAIL {name} rows={rows} cols={cols}: a row alone differs from the twin in {differ} of {sampled} sampled words")); }
        tally.checks += 1; tally.words += sampled as u64;
        // Control 1: a wrong twin (ONE range over the row instead of four joined) must NOT pass the same check.
        if one_range == 0 { return Err(format!("FP8TC FAIL {name}: the control (a twin with one range) passed the check: the check is blind")); }
        // (b) Every group size against the row alone.
        let sizes: &[usize] = if quick { &[2, 8, 9, 33, 256] } else { &[1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1000, 1024] };
        let mut compared = 0u64;
        for &size in sizes {
            let order = slots(size, xs.len());
            let input = Buf::new(g, &gather(&xs, &order))?;
            let mut kinds = vec![Kind::Tile, Kind::Tile3];
            if size <= 16 { kinds.extend([Kind::Small, Kind::Few, Kind::Multi]); }
            if rows * cols * size <= 600_000_000 { kinds.push(Kind::Generic); }
            for kind in kinds {
                let got = words(g, e, kind, &m, input.address, lutp, size, UNROUNDED)?;
                same(&format!("{kind:?} entry against the row alone, {name} rows={rows} cols={cols} tokens={size}"), &got, rows * size, |i| alone[order[i % size]][i / size])?;
                tally.checks += 1; compared += (rows * size) as u64;
            }
        }
        tally.words += compared;
        // Control 2: one wrong weight byte must fail the check in its own row and in no other row.
        let (size, bad_row, bad_col) = (17usize, rows / 3, cols / 2 + 5);
        let order = slots(size, xs.len());
        let input = Buf::new(g, &gather(&xs, &order))?;
        let at = bad_row * cols + bad_col;
        let wrong = if m.codes[at] == 0x41 { 0x42u8 } else { 0x41 };
        g.upload_at(m.weight + at as u64, &[wrong])?;
        let got = words(g, e, Kind::Tile, &m, input.address, lutp, size, UNROUNDED)?;
        let (mut own, mut other) = (0usize, 0usize);
        for (i, &word) in got.iter().enumerate() { if word != alone[order[i % size]][i / size] { if i / size == bad_row { own += 1; } else { other += 1; } } }
        // The third tile form reads the same wrong byte: the same words must move, and no other.
        let third = words(g, e, Kind::Tile3, &m, input.address, lutp, size, UNROUNDED)?;
        if third != got { return Err(format!("FP8TC FAIL {name}: with one wrong weight byte the third tile form and the first give different words")); }
        g.upload_at(m.weight + at as u64, &[m.codes[at]])?;
        for kind in [Kind::Tile, Kind::Tile3] {
            let got = words(g, e, kind, &m, input.address, lutp, size, UNROUNDED)?;
            same(&format!("{kind:?} entry after the control byte is put back, {name}"), &got, rows * size, |i| alone[order[i % size]][i / size])?;
        }
        if own == 0 || other != 0 { return Err(format!("FP8TC FAIL {name}: the control (one wrong weight byte in row {bad_row}) changed {own} words of its row and {other} words of other rows")); }
        // For the record: how many FINISHED words differ from the word of the old image (the rebase).
        let old_words = words(g, old, Kind::OldTile, &m, input.address, lutp, size, ROUNDED)?;
        let new_words = words(g, e, Kind::Tile, &m, input.address, lutp, size, ROUNDED)?;
        let moved = old_words.iter().zip(&new_words).filter(|(a, b)| a != b).count();
        println!("FP8TC GROUP ok {name} rows={rows} cols={cols}: row alone = twin on {sampled} sampled words; row alone = row in groups {sizes:?} (tile and third tile form; small, few, multi up to 16; generic where small), {compared} words, all 32 bits; CONTROL one-range twin FAILS in {one_range} of {sampled} words; CONTROL wrong byte FAILS in {own} words of its row, 0 of other rows; finished words that differ from the old image: {moved} of {} elapsed_s={:.1}",
            rows * size, started.elapsed().as_secs_f64());
        g.clear()?;
    }
    Ok(())
}
/// Weight bytes like a real matrix: a bell curve whose 3.5 sigma is the largest E4M3 value (448), cut to E4M3.
fn real_codes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| {
        let v = (r.unit() + r.unit() + r.unit() + r.unit()) * 110.85;
        let (a, sign) = (v.abs().min(448.0), if v < 0.0 { 0x80u8 } else { 0 });
        if a < 0.015625 { return sign | (a * 512.0) as u8; }
        let bits = a.to_bits();
        sign | ((((bits >> 23) & 255) as u8 - 120) << 3) | ((bits >> 20) & 7) as u8
    }).collect()
}
/// The entry main 1508efce launches for one part of `tokens` token rows (`small_choice::group` there), and the entry
/// this branch launches (the same choice; the three small entries are one body now).
fn chosen(rows: usize, cols: usize, tokens: usize) -> (Kind, Kind) {
    match tokens {
        1 => (Kind::OldDecode, Kind::Decode),
        2 => (Kind::OldSmall, Kind::Small),
        3..=4 if rows <= 768 => (Kind::OldSmall, Kind::Small),
        3..=4 if rows * cols <= 4_194_304 => (Kind::OldFew, Kind::Few),
        3..=16 => (Kind::OldMulti, Kind::Multi),
        _ if tokens >= THIRD_MIN_TOKENS => (Kind::OldTile, Kind::Tile3),
        _ => (Kind::OldTile, Kind::Tile),
    }
}
/// Old image against new image, microseconds a launch, at 1, 2, 4, 8, 256, 1,024 and 2,048 token rows of each shape
/// (the served read launches at most 1,024 rows a layer run; 2,048 is one launch of twice that). `old_share_us`: main's
/// shared-read entry (the scalar word) where main launches it at 2 to 8 rows.
/// The weights rotate over copies (256 MB a shape) so a launch reads cold weights, as a decode step does.
fn time(g: &mut Gpu, e: &Entries, old: &Entries, quick: bool) -> Result<()> {
    println!("FP8TC TIME shape rows cols tokens | old entry, us | new entry, us | old/new | GMAC/s old, new");
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        if quick && index >= 3 { break; }
        let codes = real_codes(rows * cols, 900 + index as u64);
        let copies = ((256usize << 20) / (rows * cols)).clamp(2, 64);
        let mut r = Rng(31 + index as u64);
        let scales: Vec<f32> = (0..rows.div_ceil(128) * cols.div_ceil(128)).map(|_| 0.001953125 * (1.5 + 0.5 * r.unit())).collect();
        let scale = g.upload(&le(&scales))?;
        let lutp = lut(g)?;
        let mut ms = vec![];
        for copy in 0..copies {
            let mut bytes = codes.clone(); bytes.rotate_left(copy * 4099 % codes.len());
            ms.push(Matrix { rows, cols, br: 128, bc: 128, codes: vec![], scales: scales.clone(), weight: g.upload(&bytes)?, scale });
        }
        let mut line = vec![];
        for tokens in [1usize, 2, 4, 8, 256, 1024, 2048] {
            let x: Vec<f32> = (0..tokens * cols).map(|_| r.unit() * 4.0).collect();
            let input = g.upload(&le(&x))?;
            let out = g.upload(&vec![0u8; rows * tokens * 4])?;
            let (old_kind, new_kind) = chosen(rows, cols, tokens);
            let mut tables = vec![];
            for m in &ms { tables.push(g.upload(&m.entry(tokens, ROUNDED))?); }
            let run = |g: &Gpu, arm: usize, i: usize| -> Result<()> {
                let (image, kind, m) = (if arm == 0 { old } else { e }, if arm == 0 { old_kind } else { new_kind }, &ms[i % copies]);
                if tokens == 1 { decode(g, image, kind, m, input, out, ROUNDED) } else { group(g, image, kind, m, tables[i % copies], input, out, lutp, tokens) }
            };
            let n = if tokens >= 256 { 4 } else { copies };
            // Arm 2: main's shared-read entry, where main launches it for this shape and row count (the old image).
            let shared_entry = if share_wins(rows, cols, tokens) { old.share } else { None };
            let mut t = [vec![], vec![], vec![]];
            for repeat in 0..7 { for arm in if repeat % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                if arm == 2 {
                    let Some(entry) = shared_entry else { continue; };
                    for i in 0..n { share(g, entry, &ms[i % copies], tables[i % copies], input, out, lutp, tokens)?; } g.sync()?;
                    let t0 = Instant::now(); for i in 0..n { share(g, entry, &ms[i % copies], tables[i % copies], input, out, lutp, tokens)?; } g.sync()?;
                    t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                    continue;
                }
                for i in 0..n { run(g, arm, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..n { run(g, arm, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            } }
            let (a, b) = (median(t[0].clone()), median(t[1].clone()));
            let macs = (rows * cols * tokens) as f64;
            let shared_text = if t[2].is_empty() { String::new() } else { let c = median(t[2].clone()); format!(" old_share_us={c:.1} share_over_new={:.2}", c / b) };
            println!("FP8TC TIME {name} rows={rows} cols={cols} tokens={tokens} old={old_kind:?} old_us={a:.1} new={new_kind:?} new_us={b:.1} old_over_new={:.2} old_GMACs={:.0} new_GMACs={:.0}{shared_text}", a / b, macs / a / 1e3, macs / b / 1e3);
            line.push((tokens, a, b));
        }
        let one = line[0];
        println!("FP8TC ROWS {name}: one row old {:.1} us, new {:.1} us (new/old {:.3}); new 2, 4, 8 rows over new one row: {:.2} {:.2} {:.2}; old 8 rows over old one row: {:.2}",
            one.1, one.2, one.2 / one.1, line[1].2 / one.2, line[2].2 / one.2, line[3].2 / one.2, line[3].1 / one.1);
        g.clear()?;
    }
    Ok(())
}
/// The bench entries of the row body (`kernels/src/fp8_tc_rows.rs`): (row-group entry, decode entry, prefetch stride
/// in 16-column steps). The prefetch distance is one line (4 steps) in all of them.
const ROW_ENTRIES: [(&str, &str, u32); 3] = [("fp8_tc_rows_s4", "fp8_tc_decode_s4", 4), ("fp8_tc_rows_s2", "fp8_tc_decode_s2", 2), ("fp8_tc_rows_s1", "fp8_tc_decode_s1", 1)];
/// The warp counts of a row-body block that the variant pass times (32 warps a block are refused by the device:
/// the body keeps more than 64 registers a thread, run `f8row-var-2`).
const ROW_WARPS: [u32; 3] = [4, 8, 16];
/// The model's calls of each shape of `SHAPES` in one forward of one token row (170 calls).
const CALLS: [f64; 6] = [42.0, 84.0, 11.0, 11.0, 11.0, 11.0];
/// One variant of the row body: a bench entry (a prefetch stride) at a warp count. `decode`: the decode entry of
/// its own (one token row a launch, the launch record of `expert_operation`), else the row-group entry.
#[derive(Clone, Copy)]
struct Variant { entry: H, decode: bool, ps: u32, warps: u32, tiles: u32, tile: u32, token_first: bool, third: Option<Third> }
/// The third form of the prompt tile (`fp8_tc::tile3_body`): `park` where the waiting sums are kept (0 registers and
/// the output slot, 1 `p` shared memory and the output slot, 2 `q` shared memory); `dec` 0 the tensor core decodes
/// the weights, 1 `s` the integer function.
#[derive(Clone, Copy)]
struct Third { park: u32, dec: u32 }
impl Variant {
    /// `w<warps>s<stride>` a row-group entry, `d<warps>s<stride>` a decode entry, `d<warps a tile>t<tiles a block>`
    /// a decode entry with several 16-row tiles a block, `t<warps>n<tiles of 8 tokens a warp>` the second form of the
    /// prompt tile (`x`: the token tile is the fastest grid index).
    fn name(&self) -> String {
        if let Some(third) = self.third { return format!("u{}n{}{}{}", self.warps, self.tile, ["", "p", "q"][third.park as usize], ["", "s"][third.dec as usize]); }
        if self.tile > 0 { format!("t{}n{}{}", self.warps, self.tile, if self.token_first { "x" } else { "" }) }
        else if self.tiles > 1 { format!("d{}t{}", self.warps, self.tiles) }
        else { format!("{}{}s{}", if self.decode { "d" } else { "w" }, self.warps, self.ps) }
    }
    /// Shared bytes of a block: `tiles` times the bytes of one tile; the operand stage of a prompt tile.
    fn shared(&self, m: &Matrix, tokens: usize) -> u32 {
        if let Some(third) = self.third { return 2048 * self.tile + self.warps * 32 * self.tile * [0, 16, 32][third.park as usize]; }
        if self.tile > 0 { return 2048 * self.tile; }
        self.tiles * row_shared(self.warps, (m.cols as u32).div_ceil(m.bc as u32), (tokens as u32).min(8), false)
    }
    /// False when the variant has no launch of this token count: a decode entry takes one token row, and a launch
    /// has 48 KiB of dynamic shared memory without a function attribute.
    fn fits(&self, m: &Matrix, tokens: usize) -> bool {
        // The third form takes scale blocks of 128 columns and a multiple of 16 rows, and whole stages (it traps otherwise).
        // (The launch raises the function's dynamic shared limit over 48 KiB.)
        (!self.decode || tokens == 1) && self.shared(m, tokens) <= if self.third.is_some() { 110 * 1024 } else { 48 * 1024 } && (self.third.is_none() || (m.bc == 128 && m.cols % 128 == 0 && m.br % 16 == 0))
    }
}
/// `tokens` token rows through a variant: the row-group entry on the grid (row tiles, token tiles) with a table that
/// holds `m.entry(tokens, role)`, or the decode entry (one token row, `role` in the launch record).
fn rows_launch(g: &Gpu, v: Variant, m: &Matrix, table: u64, input: u64, out: u64, lutp: u64, tokens: usize, role: u32) -> Result<()> {
    if v.decode {
        let mut l = Launch { single: Arguments { input: [input, 4, 4, 4], weight: m.weight, scales: m.scale, output: out, shape: m.shape(FP8_SERIAL, 1, role) }, batch: 0, count: 1 };
        return g.launch(v.entry, [(m.rows as u32).div_ceil(16 * v.tiles), 1, 1], 32 * v.warps * v.tiles, v.shared(m, 1), &mut [ptr(&mut l)]);
    }
    let mut a = vec![];
    for x in [table, input, out, lutp] { a.extend(x.to_le_bytes()); }
    for x in [1u32, (m.rows * tokens) as u32, 1, m.cols as u32] { a.extend(x.to_le_bytes()); }
    if v.tile > 0 {
        let (row_groups, token_tiles) = ((m.rows as u32).div_ceil(16 * v.warps), (tokens as u32).div_ceil(8 * v.tile));
        let grid = if v.token_first { [token_tiles, row_groups, 1] } else { [row_groups, token_tiles, 1] };
        return g.launch(v.entry, grid, 32 * v.warps, v.shared(m, tokens), &mut [a.as_mut_ptr().cast()]);
    }
    g.launch(v.entry, [(m.rows as u32).div_ceil(16), (tokens as u32).div_ceil(8), 1], 32 * v.warps, v.shared(m, tokens), &mut [a.as_mut_ptr().cast()])
}
fn rows_words(g: &mut Gpu, v: Variant, m: &Matrix, input: u64, lutp: u64, tokens: usize, role: u32) -> Result<Vec<u32>> {
    let out = Buf::output(g, m.rows * tokens)?;
    let table = g.upload(&m.entry(tokens, role))?;
    rows_launch(g, v, m, table, input, out.address, lutp, tokens, role)?;
    g.sync()?; out.words(g)
}
/// Every variant gives the contract word: (a) the CPU twin, every row and token, on the small forms of the twin sweep,
/// unrounded and rounded; (b) the word of the candidate's small entry (itself equal to the twin in the full sweep) on
/// every row of the model's shapes; (c) a wrong weight byte changes words of its own row and of no other row.
/// Returns which variants passed, and how many failed for another reason than a block the device refuses. A variant
/// that fails (a wrong word, or a launch the device refuses) is named and is not timed.
fn variants_same(g: &mut Gpu, e: &Entries, vars: &[Variant]) -> Result<(Vec<bool>, usize)> {
    let started = Instant::now();
    let mut ok = vec![true; vars.len()];
    let mut checked = vec![0u64; vars.len()];
    // Driver error 701: the device refuses the block (too many threads for the entry's registers). Such a variant is
    // left out and named; any other failure is a FAIL.
    let wrong = std::cell::Cell::new(0usize);
    let fail = |ok: &mut Vec<bool>, i: usize, v: &Variant, message: String| {
        let refused = message.ends_with("CUDA Driver error 701");
        if ok[i] { println!("FP8TC VAR {} {}: {message}", if refused { "REFUSED" } else { "FAIL" }, v.name()); if !refused { wrong.set(wrong.get() + 1); } }
        ok[i] = false;
    };
    // Token rows a launch: the prompt tile's variants also take more rows than one tile of a warp and of a block.
    let tiled = vars.iter().any(|v| v.tile > 0);
    let small_sizes: Vec<usize> = if tiled { vec![1, 2, 3, 5, 8, 9, 16, 17, 31, 33, 40, 48, 49, 64, 65, 70] } else { vec![1, 2, 3, 4, 5, 7, 8, 9, 16, 17] };
    let model_sizes: Vec<usize> = if tiled { vec![1, 5, 16, 64, 100] } else { vec![1, 2, 4, 5, 8, 16] };
    let forms: [(usize, usize, usize, usize); 14] = [(1, 16, 1, 16), (5, 48, 3, 16), (16, 128, 16, 128), (17, 144, 128, 128), (33, 256, 32, 64), (47, 384, 128, 128), (40, 1536, 128, 128),
        (31, 2048, 128, 128), (20, 4096, 128, 128), (9, 4096, 1, 4096), (64, 2048, 16, 16), (130, 640, 128, 128), (200, 2048, 128, 128), (3, 16384, 128, 128)];
    for (salt, &(rows, cols, br, bc)) in forms.iter().enumerate() {
        let m = Matrix::new(g, rows, cols, br, bc, salt % twin::WEIGHT_CLASSES, salt + 1)?;
        let lutp = lut(g)?;
        let xs = pool(cols, 40, salt + 1);
        let all: Vec<usize> = (0..rows).collect();
        let want = twin_words(&m, &all, &xs);
        for &size in &small_sizes {
            let order = slots(size, xs.len());
            let input = Buf::new(g, &gather(&xs, &order))?;
            for (i, v) in vars.iter().enumerate() {
                if !ok[i] || !v.fits(&m, size) { continue; }
                let label = format!("rows={rows} cols={cols} block={br}x{bc} tokens={size}");
                let got = match rows_words(g, *v, &m, input.address, lutp, size, UNROUNDED) { Ok(got) => got, Err(message) => { fail(&mut ok, i, v, format!("the launch failed ({} threads a block): {message}", 32 * v.warps * v.tiles)); continue; } };
                if let Err(message) = same(&format!("twin, unrounded, {label}"), &got, rows * size, |w| want[w / size][order[w % size]]) { fail(&mut ok, i, v, message); continue; }
                let got = match rows_words(g, *v, &m, input.address, lutp, size, ROUNDED) { Ok(got) => got, Err(message) => { fail(&mut ok, i, v, format!("the launch failed ({} threads a block): {message}", 32 * v.warps * v.tiles)); continue; } };
                if let Err(message) = same(&format!("twin, rounded, {label}"), &got, rows * size, |w| twin::word(want[w / size][order[w % size]])) { fail(&mut ok, i, v, message); continue; }
                checked[i] += 2 * (rows * size) as u64;
            }
        }
        g.clear()?;
    }
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        let m = Matrix::new(g, rows, cols, 128, 128, 3, 100 + index)?;
        let lutp = lut(g)?;
        let xs = pool(cols, 28, 100 + index);
        for &size in &model_sizes {
            // One token row: pool row 0, values near one (every product reaches the word, so the control below sees
            // its wrong byte; run `f8row-var-3` lost its control on a row of a wide exponent spread).
            let order = if size == 1 { vec![0] } else { slots(size, xs.len()) };
            let input = Buf::new(g, &gather(&xs, &order))?;
            let reference = words(g, e, Kind::Small, &m, input.address, lutp, size, UNROUNDED)?;
            for (i, v) in vars.iter().enumerate() {
                if !ok[i] || !v.fits(&m, size) { continue; }
                let got = match rows_words(g, *v, &m, input.address, lutp, size, UNROUNDED) { Ok(got) => got, Err(message) => { fail(&mut ok, i, v, format!("the launch failed ({} threads a block): {message}", 32 * v.warps * v.tiles)); continue; } };
                if let Err(message) = same(&format!("candidate small entry, {name} rows={rows} cols={cols} tokens={size}"), &got, rows * size, |w| reference[w]) { fail(&mut ok, i, v, message); continue; }
                checked[i] += (rows * size) as u64;
            }
            // The control: one wrong weight byte must change words of its own row and of no other row (one token row for
            // the decode entries, five for the row-group entries).
            if size == 5 || size == 1 {
                let (bad_row, bad_col) = (rows / 3, cols / 2 + 5);
                let at = bad_row * cols + bad_col;
                let wrong = if m.codes[at] == 0x41 { 0x42u8 } else { 0x41 };
                g.upload_at(m.weight + at as u64, &[wrong])?;
                for (i, v) in vars.iter().enumerate() {
                    if !ok[i] || !v.fits(&m, size) { continue; }
                    let got = match rows_words(g, *v, &m, input.address, lutp, size, UNROUNDED) { Ok(got) => got, Err(message) => { fail(&mut ok, i, v, format!("the launch failed ({} threads a block): {message}", 32 * v.warps * v.tiles)); continue; } };
                    let (mut own, mut other) = (0usize, 0usize);
                    for (w, &word) in got.iter().enumerate() { if word != reference[w] { if w / size == bad_row { own += 1; } else { other += 1; } } }
                    if own == 0 || other != 0 { fail(&mut ok, i, v, format!("the control (one wrong weight byte in row {bad_row} of {name}) changed {own} words of its row and {other} words of other rows")); }
                }
                g.upload_at(m.weight + at as u64, &[m.codes[at]])?;
            }
        }
        g.clear()?;
    }
    for (i, v) in vars.iter().enumerate() {
        if ok[i] { println!("FP8TC VAR SAME {}: ok, {} words = the twin (14 small forms, unrounded and rounded, token rows {small_sizes:?}) and = the candidate's small entry (6 model shapes, token rows {model_sizes:?}); CONTROL wrong byte caught on every shape", v.name(), checked[i]); }
    }
    println!("FP8TC VAR SAME done: {} of {} variants pass, {} wrong, elapsed_s={:.1}", ok.iter().filter(|&&o| o).count(), vars.len(), wrong.get(), started.elapsed().as_secs_f64());
    Ok((ok, wrong.get()))
}
/// The cold pass of every variant beside main and the candidate: microseconds a launch at 1, 2, 4 and 8 token rows of
/// each model shape, every launch on its own copy of the weights (256 MB of copies a shape), as a decode step reads.
fn variants_cold(g: &mut Gpu, e: &Entries, old: &Entries, vars: &[Variant], ok: &[bool]) -> Result<()> {
    let counts = [1usize, 2, 4, 8];
    let arms = 2 + vars.len();
    // [token count][arm]: the weighted pass of 170 calls, ms; and the worst ratio to main of a one-row shape.
    let mut pass = vec![vec![0.0f64; arms]; counts.len()];
    let mut worst = vec![(0.0f64, ""); arms];
    // The served pass of one token row: 128 launches (42 of gate and up as a batch of two, 42 down, 11 of each other shape).
    let mut served = vec![0.0f64; arms];
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        let codes = real_codes(rows * cols, 900 + index as u64);
        let copies = ((256usize << 20) / (rows * cols)).clamp(2, 64);
        let mut r = Rng(31 + index as u64);
        let scales: Vec<f32> = (0..rows.div_ceil(128) * cols.div_ceil(128)).map(|_| 0.001953125 * (1.5 + 0.5 * r.unit())).collect();
        let scale = g.upload(&le(&scales))?;
        let lutp = lut(g)?;
        let mut ms = vec![];
        for copy in 0..copies {
            let mut bytes = codes.clone(); bytes.rotate_left(copy * 4099 % codes.len());
            ms.push(Matrix { rows, cols, br: 128, bc: 128, codes: vec![], scales: scales.clone(), weight: g.upload(&bytes)?, scale });
        }
        for (ci, &tokens) in counts.iter().enumerate() {
            let x: Vec<f32> = (0..tokens * cols).map(|_| r.unit() * 4.0).collect();
            let input = g.upload(&le(&x))?;
            let out = g.upload(&vec![0u8; rows * tokens * 4])?;
            let (old_kind, new_kind) = chosen(rows, cols, tokens);
            let mut tables = vec![];
            for m in &ms { tables.push(g.upload(&m.entry(tokens, ROUNDED))?); }
            let live: Vec<bool> = (0..arms).map(|arm| arm < 2 || (ok[arm - 2] && vars[arm - 2].fits(&ms[0], tokens))).collect();
            let run = |g: &Gpu, arm: usize, i: usize| -> Result<()> {
                let m = &ms[i % copies];
                if arm >= 2 { return rows_launch(g, vars[arm - 2], m, tables[i % copies], input, out, lutp, tokens, ROUNDED); }
                let (image, kind) = if arm == 0 { (old, old_kind) } else { (e, new_kind) };
                if tokens == 1 { decode(g, image, kind, m, input, out, ROUNDED) } else { group(g, image, kind, m, tables[i % copies], input, out, lutp, tokens) }
            };
            let mut t = vec![vec![]; arms];
            for repeat in 0..5 { for step in 0..arms {
                let arm = (step + repeat * 7) % arms;
                if !live[arm] { continue; }
                for i in 0..copies { run(g, arm, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..copies { run(g, arm, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / copies as f64);
            } }
            let us: Vec<f64> = t.into_iter().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
            let mut line = format!("FP8TC VAR TIME {name} rows={rows} cols={cols} tokens={tokens} main={:.1} cand={:.1} |", us[0], us[1]);
            for (i, v) in vars.iter().enumerate() { line += &format!(" {}={:.1}", v.name(), us[2 + i]); }
            println!("{line}");
            for arm in 0..arms {
                pass[ci][arm] += us[arm] * CALLS[index] / 1e3;
                if tokens == 1 && us[arm] / us[0] > worst[arm].0 { worst[arm] = (us[arm] / us[0], name); }
                if tokens == 1 && name != "gate_up" { served[arm] += us[arm] * CALLS[index] / 1e3; }
            }
            // WARM: every launch on the SAME copy of the weights (they stay in the cache). The served step pre-reads up
            // to 6 MiB of a single block FP8 matrix of 2 MiB or more on a side branch (`bus_shadow::more`), so `down`
            // and `q_a` run on warm bytes there, and the first 6 MiB of `o_proj` and `q_b`.
            if tokens == 1 {
                let mut t = vec![vec![]; arms];
                for repeat in 0..5 { for step in 0..arms {
                    let arm = (step + repeat * 7) % arms;
                    if !live[arm] { continue; }
                    for _ in 0..8 { run(g, arm, 0)?; } g.sync()?;
                    let t0 = Instant::now(); for _ in 0..32 { run(g, arm, 0)?; } g.sync()?;
                    t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / 32.0);
                } }
                let warm: Vec<f64> = t.into_iter().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
                let mut line = format!("FP8TC VAR WARM {name} rows={rows} cols={cols} tokens=1 main={:.1} cand={:.1} |", warm[0], warm[1]);
                for (i, v) in vars.iter().enumerate() { line += &format!(" {}={:.1}", v.name(), warm[2 + i]); }
                println!("{line}");
            }
            // The served form of gate and up: one launch of two jobs (two matrices, one token row, two outputs).
            if tokens == 1 && name == "gate_up" {
                let out2 = g.upload(&vec![0u8; rows * 4])?;
                // Launch i reads copies 2i and 2i + 1: both matrices cold, as in a decode step (run `f8row-var-3` read
                // the second matrix warm: it was the next launch's first).
                let pairs = copies / 2;
                let mut batches = vec![];
                for i in 0..pairs {
                    let mut record = job_bytes(&ms[2 * i], input, out, ROUNDED);
                    record.extend(job_bytes(&ms[2 * i + 1], input, out2, ROUNDED));
                    batches.push(g.upload(&record)?);
                }
                let live: Vec<bool> = (0..arms).map(|arm| arm < 2 || (ok[arm - 2] && vars[arm - 2].decode && vars[arm - 2].fits(&ms[0], 1))).collect();
                let run = |g: &Gpu, arm: usize, i: usize| -> Result<()> {
                    let m = &ms[2 * i];
                    if arm >= 2 { let v = vars[arm - 2]; return decode_pair(g, v.entry, [(rows as u32).div_ceil(16 * v.tiles), 1, 1], 32 * v.warps * v.tiles, v.shared(m, 1), m, input, out, batches[i]); }
                    let (image, kind) = if arm == 0 { (old, old_kind) } else { (e, new_kind) };
                    let (grid, threads, bytes) = geometry(kind, m, 1, ROUNDED);
                    decode_pair(g, image.of(kind), grid, threads, bytes, m, input, out, batches[i])
                };
                let mut t = vec![vec![]; arms];
                for repeat in 0..5 { for step in 0..arms {
                    let arm = (step + repeat * 7) % arms;
                    if !live[arm] { continue; }
                    for i in 0..pairs { run(g, arm, i)?; } g.sync()?;
                    let t0 = Instant::now(); for i in 0..pairs { run(g, arm, i)?; } g.sync()?;
                    t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / pairs as f64);
                } }
                let pair: Vec<f64> = t.into_iter().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
                let mut line = format!("FP8TC VAR PAIR gate_up x2 in one launch rows={rows} cols={cols} tokens=1 main={:.1} cand={:.1} |", pair[0], pair[1]);
                for (i, v) in vars.iter().enumerate() { if v.decode { line += &format!(" {}={:.1}", v.name(), pair[2 + i]); } }
                println!("{line}");
                for arm in 0..arms { served[arm] += pair[arm] * 42.0 / 1e3; }
            }
        }
        g.clear()?;
    }
    for (ci, &tokens) in counts.iter().enumerate() {
        let mut line = format!("FP8TC VAR PASS tokens={tokens} ms_a_pass_of_170_calls main={:.2} cand={:.2} |", pass[ci][0], pass[ci][1]);
        for (i, v) in vars.iter().enumerate() { line += &format!(" {}={:.2}", v.name(), pass[ci][2 + i]); }
        println!("{line}");
    }
    let mut line = format!("FP8TC VAR SERVED tokens=1 ms_a_pass_of_128_launches (gate and up as one launch of two) main={:.2} cand={:.2} |", served[0], served[1]);
    for (i, v) in vars.iter().enumerate() { if v.decode { line += &format!(" {}={:.2}", v.name(), served[2 + i]); } }
    println!("{line}");
    let mut line = format!("FP8TC VAR WORST one-row time over main, of the six shapes: cand={:.3} ({}) |", worst[1].0, worst[1].1);
    for (i, v) in vars.iter().enumerate() { line += &format!(" {}={:.3} ({})", v.name(), worst[2 + i].0, worst[2 + i].1); }
    println!("{line}");
    Ok(())
}
/// The prompt tile's variants beside main's tile and the candidate's tile: microseconds a launch at 64, 256 and
/// 1,024 token rows of each model shape, the weights rotating over copies as in `time`.
fn tiles_cold(g: &mut Gpu, e: &Entries, old: &Entries, vars: &[Variant], ok: &[bool]) -> Result<()> {
    let counts = [24usize, 40, 64, 256, 1024];
    let arms = 2 + vars.len();
    let mut pass = vec![vec![0.0f64; arms]; counts.len()];
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        let codes = real_codes(rows * cols, 700 + index as u64);
        let copies = ((256usize << 20) / (rows * cols)).clamp(2, 8);
        let mut r = Rng(57 + index as u64);
        let scales: Vec<f32> = (0..rows.div_ceil(128) * cols.div_ceil(128)).map(|_| 0.001953125 * (1.5 + 0.5 * r.unit())).collect();
        let scale = g.upload(&le(&scales))?;
        let lutp = lut(g)?;
        let mut ms = vec![];
        for copy in 0..copies {
            let mut bytes = codes.clone(); bytes.rotate_left(copy * 4099 % codes.len());
            ms.push(Matrix { rows, cols, br: 128, bc: 128, codes: vec![], scales: scales.clone(), weight: g.upload(&bytes)?, scale });
        }
        for (ci, &tokens) in counts.iter().enumerate() {
            let x: Vec<f32> = (0..tokens * cols).map(|_| r.unit() * 4.0).collect();
            let input = g.upload(&le(&x))?;
            let out = g.upload(&vec![0u8; rows * tokens * 4])?;
            // Arm 1 is the FIRST form of the tile (`fp8_rows_tile`) at every size, whatever the host launches there.
            let (old_kind, new_kind) = (chosen(rows, cols, tokens).0, Kind::Tile);
            let mut tables = vec![];
            for m in &ms { tables.push(g.upload(&m.entry(tokens, ROUNDED))?); }
            let run = |g: &Gpu, arm: usize, i: usize| -> Result<()> {
                let m = &ms[i % copies];
                if arm >= 2 { return rows_launch(g, vars[arm - 2], m, tables[i % copies], input, out, lutp, tokens, ROUNDED); }
                let (image, kind) = if arm == 0 { (old, old_kind) } else { (e, new_kind) };
                group(g, image, kind, m, tables[i % copies], input, out, lutp, tokens)
            };
            let mut t = vec![vec![]; arms];
            for repeat in 0..3 { for step in 0..arms {
                let arm = (step + repeat * 5) % arms;
                if arm >= 2 && (!ok[arm - 2] || !vars[arm - 2].fits(&ms[0], tokens)) { continue; }
                for i in 0..copies { run(g, arm, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..copies { run(g, arm, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / copies as f64);
            } }
            let us: Vec<f64> = t.into_iter().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
            let tmac = |us: f64| (rows * cols * tokens) as f64 / us / 1e6;
            let mut line = format!("FP8TC TILE TIME {name} rows={rows} cols={cols} tokens={tokens} main={:.1} first={:.1} (TMAC/s {:.2} {:.2}) |", us[0], us[1], tmac(us[0]), tmac(us[1]));
            for (i, v) in vars.iter().enumerate() { line += &format!(" {}={:.1} ({:.2})", v.name(), us[2 + i], tmac(us[2 + i])); }
            println!("{line}");
            for arm in 0..arms { pass[ci][arm] += us[arm] * CALLS[index] / 1e3; }
        }
        g.clear()?;
    }
    for (ci, &tokens) in counts.iter().enumerate() {
        let mut line = format!("FP8TC TILE PASS tokens={tokens} ms_a_pass_of_170_calls main={:.2} first={:.2} |", pass[ci][0], pass[ci][1]);
        for (i, v) in vars.iter().enumerate() { line += &format!(" {}={:.2}", v.name(), pass[ci][2 + i]); }
        println!("{line}");
    }
    Ok(())
}
/// `mode`: `proof` (the twin sweep, the groups and the controls), `time`, `quick` (the twin sweep, two shapes of
/// groups with fewer sizes, three shapes of timing), or empty for the proof and then the timing.
/// Launch kinds of the recorded step (`gpu/src/resident_launch.rs`): main's `Expert` and the block FP8 decode tile.
const KIND_EXPERT: u32 = 2;
const KIND_FP8_TILE: u32 = 72;
/// One recorded command (1,040 bytes): the kind, 12 reserved bytes, then the payload from byte 16. Here the payload is
/// the launch record `LaunchArguments` (112 bytes): one job's 96 bytes, the device address of a batch table, the count.
fn command(kind: u32, m: &Matrix, input: u64, out: u64, role: u32, batch: u64, count: u32) -> Vec<u8> {
    let mut c = vec![0u8; 1040];
    c[..4].copy_from_slice(&kind.to_le_bytes());
    let mut payload = job_bytes(m, input, out, role);
    payload.extend(batch.to_le_bytes()); payload.extend(count.to_le_bytes());
    c[16..16 + payload.len()].copy_from_slice(&payload);
    c
}
/// The recorded launch: a `resident_launch_*` entry reads command 0 of `table`, with the grid of the direct launch.
fn replay(g: &Gpu, entry: H, grid: [u32; 3], threads: u32, bytes: u32, table: u64) -> Result<()> {
    let (mut address, mut index) = (table, 0u64);
    g.launch(entry, grid, threads, bytes, &mut [ptr(&mut address), ptr(&mut index)])
}
/// The decode step's RECORDED entry of block FP8 (`resident_launch_fp8_tile`, kind 72): what the served step replays.
/// Proof on the six model shapes: its words equal the direct entry's (`fp8_tile`) as one job in the three roles and as
/// a batch of two jobs in one launch (the served gate and up), and the twin's on sampled rows (unrounded); a command
/// with another token row is the control; main's recorded entry equals main's direct entry (the harness check).
/// Time, one token row: main's recorded entry (`resident_launch_expert`) against the new one, each beside its direct
/// entry, cold (each launch on its own copy of the weights) and warm (the same matrix again), and the same for the
/// two-job launch of gate and up. The last lines sum a step's 128 launches (42 pairs, 42 down, 11 of each other).
fn recorded(g: &mut Gpu, ma: H, mb: H, e: &Entries, old: &Entries) -> Result<()> {
    let (old_replay, new_replay) = (g.function(ma, "resident_launch_expert")?, g.function(mb, "resident_launch_fp8_tile")?);
    let (mut checks, mut words_same, mut controls) = (0u64, 0u64, 0u64);
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        let (m, m2) = (Matrix::new(g, rows, cols, 128, 128, 3, 300 + index)?, Matrix::new(g, rows, cols, 128, 128, 5, 340 + index)?);
        let lutp = lut(g)?;
        let xs = pool(cols, 7, 300 + index);
        let mut inputs = vec![];
        for x in &xs { inputs.push(Buf::new(g, &le(x))?); }
        let sample: Vec<usize> = [0usize, 1, 15, 16, 17, 127, 128, rows / 2, rows - 17, rows - 16, rows - 1].into_iter().filter(|&r| r < rows).collect();
        let want = twin_words(&m, &sample, &xs);
        for role in [ROUNDED, UNROUNDED, QUARTER] {
            let (grid, threads, bytes) = geometry(Kind::Decode, &m, 1, role);
            for (p, xb) in inputs.iter().enumerate() {
                let direct = words(g, e, Kind::Decode, &m, xb.address, lutp, 1, role)?;
                let out = Buf::output(g, rows)?;
                let table = g.upload(&command(KIND_FP8_TILE, &m, xb.address, out.address, role, 0, 1))?;
                replay(g, new_replay, grid, threads, bytes, table)?; g.sync()?;
                let got = out.words(g)?;
                same(&format!("recorded {name} role={role} token row {p}: the recorded entry against the direct entry"), &got, rows, |i| direct[i])?;
                checks += 1; words_same += rows as u64;
                if role == UNROUNDED {
                    for (i, &row) in sample.iter().enumerate() {
                        if got[row] != want[i][p] { return Err(format!("FP8TC FAIL recorded {name}: row {row} token row {p} differs from the twin")); }
                    }
                    checks += 1; words_same += sample.len() as u64;
                }
                // The control: the command of the next token row gives other words.
                if role == UNROUNDED && p == 0 {
                    let other = Buf::output(g, rows)?;
                    let table = g.upload(&command(KIND_FP8_TILE, &m, inputs[1].address, other.address, role, 0, 1))?;
                    replay(g, new_replay, grid, threads, bytes, table)?; g.sync()?;
                    let differ = other.words(g)?.iter().zip(&direct).filter(|(a, b)| a != b).count();
                    if differ == 0 { return Err(format!("FP8TC FAIL recorded {name}: the control (another token row) gives the same words")); }
                    controls += 1;
                }
            }
        }
        // Two jobs in one recorded launch (the served gate and up): each job's words equal that matrix alone.
        let (grid, threads, bytes) = geometry(Kind::Decode, &m, 1, ROUNDED);
        for (p, xb) in inputs.iter().enumerate().take(3) {
            let (a, b) = (words(g, e, Kind::Decode, &m, xb.address, lutp, 1, ROUNDED)?, words(g, e, Kind::Decode, &m2, xb.address, lutp, 1, ROUNDED)?);
            let (out_a, out_b) = (Buf::output(g, rows)?, Buf::output(g, rows)?);
            let mut record = job_bytes(&m, xb.address, out_a.address, ROUNDED);
            record.extend(job_bytes(&m2, xb.address, out_b.address, ROUNDED));
            let batch = g.upload(&record)?;
            let table = g.upload(&command(KIND_FP8_TILE, &m, xb.address, out_a.address, ROUNDED, batch, 2))?;
            replay(g, new_replay, [2 * grid[0], grid[1], grid[2]], threads, bytes, table)?; g.sync()?;
            let (got_a, got_b) = (out_a.words(g)?, out_b.words(g)?);
            same(&format!("recorded pair {name} token row {p}, first job"), &got_a, rows, |i| a[i])?;
            same(&format!("recorded pair {name} token row {p}, second job"), &got_b, rows, |i| b[i])?;
            checks += 2; words_same += 2 * rows as u64;
        }
        // The harness on main's image: its recorded entry gives its direct entry's words.
        let (grid, threads, bytes) = geometry(Kind::OldDecode, &m, 1, ROUNDED);
        let direct = words(g, old, Kind::OldDecode, &m, inputs[0].address, lutp, 1, ROUNDED)?;
        let out = Buf::output(g, rows)?;
        let table = g.upload(&command(KIND_EXPERT, &m, inputs[0].address, out.address, ROUNDED, 0, 1))?;
        replay(g, old_replay, grid, threads, bytes, table)?; g.sync()?;
        same(&format!("recorded {name}: main's recorded entry against main's direct entry"), &out.words(g)?, rows, |i| direct[i])?;
        checks += 1; words_same += rows as u64;
        println!("FP8TC RECORDED SAME {name} rows={rows} cols={cols}: recorded = direct (3 roles x {} token rows), = twin on {} rows, pair = alone, main recorded = main direct", xs.len(), sample.len());
        g.clear()?;
    }
    println!("FP8TC RECORDED PROOF PASS checks={checks} words={words_same} controls_caught={controls}");
    // Time. Arms: 0 main direct, 1 main recorded, 2 new direct, 3 new recorded.
    println!("FP8TC RECORDED TIME shape | us a launch: main direct, main recorded, new direct, new recorded | new recorded over main recorded");
    let mut step = [[0f64; 4]; 2];
    for (index, &(name, rows, cols)) in SHAPES.iter().enumerate() {
        let codes = real_codes(rows * cols, 900 + index as u64);
        let copies = ((256usize << 20) / (rows * cols)).clamp(2, 64);
        let mut r = Rng(31 + index as u64);
        let scales: Vec<f32> = (0..rows.div_ceil(128) * cols.div_ceil(128)).map(|_| 0.001953125 * (1.5 + 0.5 * r.unit())).collect();
        let scale = g.upload(&le(&scales))?;
        let mut ms = vec![];
        for copy in 0..copies {
            let mut bytes = codes.clone(); bytes.rotate_left(copy * 4099 % codes.len());
            ms.push(Matrix { rows, cols, br: 128, bc: 128, codes: vec![], scales: scales.clone(), weight: g.upload(&bytes)?, scale });
        }
        let x: Vec<f32> = (0..cols).map(|_| r.unit() * 4.0).collect();
        let input = g.upload(&le(&x))?;
        let (out, out2) = (g.upload(&vec![0u8; rows * 4])?, g.upload(&vec![0u8; rows * 4])?);
        let (old_geo, new_geo) = (geometry(Kind::OldDecode, &ms[0], 1, ROUNDED), geometry(Kind::Decode, &ms[0], 1, ROUNDED));
        for jobs in [1usize, 2] {
            if jobs == 2 && name != "gate_up" { continue; }
            // Launch i reads copy i (one job) or copies 2i and 2i + 1 (two jobs): all cold.
            let launches = copies / jobs;
            let (mut batches, mut old_tables, mut new_tables) = (vec![], vec![], vec![]);
            for i in 0..launches {
                let batch = if jobs == 1 { 0 } else {
                    let mut record = job_bytes(&ms[2 * i], input, out, ROUNDED);
                    record.extend(job_bytes(&ms[2 * i + 1], input, out2, ROUNDED));
                    g.upload(&record)?
                };
                batches.push(batch);
                old_tables.push(g.upload(&command(KIND_EXPERT, &ms[jobs * i], input, out, ROUNDED, batch, jobs as u32))?);
                new_tables.push(g.upload(&command(KIND_FP8_TILE, &ms[jobs * i], input, out, ROUNDED, batch, jobs as u32))?);
            }
            let run = |g: &Gpu, arm: usize, i: usize| -> Result<()> {
                let (geo, m) = (if arm < 2 { old_geo } else { new_geo }, &ms[jobs * i]);
                let grid = [jobs as u32 * geo.0[0], geo.0[1], geo.0[2]];
                match arm {
                    0 | 2 => {
                        let mut l = Launch { single: Arguments { input: [input, 4, 4, 4], weight: m.weight, scales: m.scale, output: out, shape: m.shape(FP8_SERIAL, 1, ROUNDED) }, batch: batches[i], count: jobs as u32 };
                        g.launch(if arm == 0 { old.operation } else { e.decode }, grid, geo.1, geo.2, &mut [ptr(&mut l)])
                    }
                    1 => replay(g, old_replay, grid, geo.1, geo.2, old_tables[i]),
                    _ => replay(g, new_replay, grid, geo.1, geo.2, new_tables[i]),
                }
            };
            for (w, warm) in [false, true].into_iter().enumerate() {
                let mut t = [vec![], vec![], vec![], vec![]];
                for repeat in 0..9 { for step in 0..4 {
                    let arm = (step + repeat * 3) % 4;
                    let n = if warm { 32 } else { launches };
                    for i in 0..n { run(g, arm, if warm { 0 } else { i })?; } g.sync()?;
                    let t0 = Instant::now(); for i in 0..n { run(g, arm, if warm { 0 } else { i })?; } g.sync()?;
                    t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                } }
                let us: Vec<f64> = t.into_iter().map(median).collect();
                println!("FP8TC RECORDED TIME {name} rows={rows} cols={cols} jobs={jobs} {} main_direct={:.1} main_recorded={:.1} new_direct={:.1} new_recorded={:.1} new_rec_over_main_rec={:.3} new_rec_over_new_direct={:.3}",
                    if warm { "warm" } else { "cold" }, us[0], us[1], us[2], us[3], us[3] / us[1], us[3] / us[2]);
                // A step's launches of this shape: 42 gate and up pairs, 42 down, 11 of each other shape.
                let calls = match (name, jobs) { ("gate_up", 2) => 42.0, ("gate_up", _) => 0.0, ("down", _) => 42.0, _ => 11.0 };
                for arm in 0..4 { step[w][arm] += us[arm] * calls / 1e3; }
            }
        }
        g.clear()?;
    }
    for (w, label) in ["cold", "warm"].into_iter().enumerate() {
        println!("FP8TC RECORDED STEP {label} ms_a_step_of_128_launches main_direct={:.3} main_recorded={:.3} new_direct={:.3} new_recorded={:.3} new_recorded_minus_main_recorded={:+.3}",
            step[w][0], step[w][1], step[w][2], step[w][3], step[w][3] - step[w][1]);
    }
    Ok(())
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("FP8TC baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?;
    let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let (old, new) = (Entries::load(&g, ma)?, Entries::load(&g, mb)?);
    let started = Instant::now();
    let quick = mode == "quick";
    if mode == "var" {
        let mut vars = vec![];
        for warps in ROW_WARPS { for (rows_entry, decode_entry, ps) in ROW_ENTRIES {
            vars.push(Variant { entry: g.function(mb, rows_entry)?, decode: false, ps, warps, tiles: 1, tile: 0, token_first: false, third: None });
            vars.push(Variant { entry: g.function(mb, decode_entry)?, decode: true, ps, warps, tiles: 1, tile: 0, token_first: false, third: None });
        } }
        // Several 16-row tiles a block (fewer, larger blocks): 4 warps a tile at 2 and 4 tiles, 8 warps a tile at 2.
        for (name, warps, tiles) in [("fp8_tc_decode_t2", 4, 2), ("fp8_tc_decode_t4", 4, 4), ("fp8_tc_decode_t2", 8, 2)] {
            vars.push(Variant { entry: g.function(mb, name)?, decode: true, ps: 4, warps, tiles, tile: 0, token_first: false, third: None });
        }
        let (ok, wrong) = variants_same(&mut g, &new, &vars)?;
        variants_cold(&mut g, &new, &old, &vars, &ok)?;
        if wrong != 0 { return Err("FP8TC FAIL: a variant of the row body does not give the contract word".into()); }
        println!("FP8TC PASS mode={mode} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode == "recorded" {
        recorded(&mut g, ma, mb, &new, &old)?;
        println!("FP8TC PASS mode={mode} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode == "tile3" {
        // The third form of the prompt tile beside the best second forms: the served entry and the bench forms (4, 6
        // and 8 tiles of 8 token rows a warp, the waiting sums in registers or shared memory, the integer decode).
        let mut vars = vec![];
        // (entry, tiles of 8 token rows a warp, where the waiting sums are kept, the weight decode)
        let forms = [("fp8_rows_tile3", 4, 0, 0), ("fp8_tc_t3_n4p", 4, 1, 0), ("fp8_tc_t3_n4q", 4, 2, 0), ("fp8_tc_t3_n4s", 4, 0, 1), ("fp8_tc_t3_n6", 6, 0, 0), ("fp8_tc_t3_n6p", 6, 1, 0), ("fp8_tc_t3_n6q", 6, 2, 0),
            ("fp8_tc_t3_n8p", 8, 1, 0), ("fp8_tc_t3_n8q", 8, 2, 0)];
        // Each form at 8 and 12 warps a block, the forms of 4 tiles at 16 too. The driver's register count decides the
        // most threads a block (MEASURED run `f8tc-t3-4`, cuFuncGetAttribute: 128 registers a thread for the served
        // form, 104 to 136 for the others, 512 or 384 threads a block); a block it refuses is left out below.
        for (name, tile, park, dec) in forms {
            let entry = g.function(mb, name)?;
            for warps in if tile == 4 { vec![8, 12, 16] } else { vec![8, 12] } { vars.push(Variant { entry, decode: false, ps: 4, warps, tiles: 1, tile, token_first: false, third: Some(Third { park, dec }) }); }
        }
        for (name, tile, warps) in [("fp8_tc_tile_n4", 4, 8), ("fp8_tc_tile_n8", 8, 16)] {
            vars.push(Variant { entry: g.function(mb, name)?, decode: false, ps: 4, warps, tiles: 1, tile, token_first: false, third: None });
        }
        let (ok, wrong) = variants_same(&mut g, &new, &vars)?;
        tiles_cold(&mut g, &new, &old, &vars, &ok)?;
        if wrong != 0 { return Err("FP8TC FAIL: a variant of the prompt tile does not give the contract word".into()); }
        println!("FP8TC PASS mode={mode} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode == "tile" {
        // The second form of the prompt tile: 4, 6 and 8 tiles of 8 token rows a warp, blocks of 8 and 4 warps, and
        // the token tile as the fastest grid index.
        let mut vars = vec![];
        for (name, tile, token_first, warps) in [("fp8_tc_tile_n8", 8, false, 8), ("fp8_tc_tile_n8", 8, false, 4), ("fp8_tc_tile_n8t", 8, true, 8), ("fp8_tc_tile_n6", 6, false, 8),
            ("fp8_tc_tile_n4", 4, false, 8), ("fp8_tc_tile_n4", 4, false, 4), ("fp8_tc_tile_n8", 8, false, 16)] {
            vars.push(Variant { entry: g.function(mb, name)?, decode: false, ps: 4, warps, tiles: 1, tile, token_first, third: None });
        }
        let (ok, wrong) = variants_same(&mut g, &new, &vars)?;
        tiles_cold(&mut g, &new, &old, &vars, &ok)?;
        if wrong != 0 { return Err("FP8TC FAIL: a variant of the prompt tile does not give the contract word".into()); }
        println!("FP8TC PASS mode={mode} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if mode != "time" {
        let mut tally = Tally { checks: 0, words: 0 };
        twin_sweep(&mut g, &new, &mut tally)?;
        groups(&mut g, &new, &old, &mut tally, quick)?;
        println!("FP8TC PROOF PASS checks={} words={} digest={} elapsed_s={:.1}", tally.checks, tally.words, sha256::digest(format!("{}{}", tally.checks, tally.words).as_bytes()), started.elapsed().as_secs_f64());
    }
    if mode != "proof" { time(&mut g, &new, &old, quick)?; }
    println!("FP8TC PASS mode={mode} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}

/// Production-shape tile3 replay, one exact unit operand at every live coordinate.
pub(in crate::cuda) fn diag_replay(g:&mut Gpu,module:H)->Result<()> {
 let f=g.function(module,"fp8_rows_tile3")?;
 for (n,k,calls) in [(1024usize,4096usize,672usize),(2048,2048,336),(2048,16384,88),(8192,1536,88),(768,4096,88),(256,4096,88)] {
  let m=1024usize;let w=g.upload(&vec![0x38;n*k])?;let sc=g.words(&vec![1f32.to_bits();n.div_ceil(128)*k.div_ceil(128)])?;
  let x=g.words(&vec![1f32.to_bits();m*k])?;let out=g.upload(&vec![0xa5;m*n*4])?;
  let matrix=Matrix{rows:n,cols:k,br:128,bc:128,codes:vec![],scales:vec![],weight:w,scale:sc};
  let table=g.upload(&matrix.entry(m,ROUNDED))?;
  let launch=|g:&Gpu|{let mut a=vec![];for v in [table,x,out,0]{a.extend(v.to_le_bytes());}for v in [1u32,(m*n) as u32,1,k as u32]{a.extend(v.to_le_bytes());}g.launch(f,[n.div_ceil(128) as u32,m.div_ceil(32) as u32,1],256,8192,&mut [a.as_mut_ptr().cast()])};
  super::super::diag_limits::replay(g,"fp8_rows_tile3",&format!("M={m} N={n} K={k} trace_calls={calls}"),out,m*n,Some((k as f32).to_bits()),&launch)?;g.clear()?;
 }Ok(())
}
