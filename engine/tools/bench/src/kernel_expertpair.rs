//! EXPERT-PAIR: the generic expert entry `expert_operation` of two images on the same launches, byte for byte.
//! Run: kernel-bench --mx-tc <old ptx> <new ptx> expert-pair
//!
//! Use: the device check of this kernel when the code list names it DIFFER though its source did not change (the
//! compiler gives a kernel other machine code when another kernel of the image changes).
//!
//! Every operation of the entry (`gpu/src/expert_ops.rs`, `Op`): Quantize (the one-unit body and the lanes body),
//! QuantizeRows, NvProject (first and second role; one job and a table of three), NvRows, Fp8Project (one job and a
//! table), Fp8Rows, Fp8Serial (rounded, unrounded, raw quarter; one job and a table), PlainSerial (one job and a
//! table), SwiGlu and Fold. The inputs follow the rules of the host reference (`expert_ops::reference`): finite values,
//! NVFP4 activations as the quantizer of the OLD image wrote them, NVFP4 scale codes under 127, FP8 weight codes that
//! are not NaN, positive finite FP8 block scales.
//! The check knows no word function. It proves old image = new image on these launches, not that either is right.
//! In every case: the output must not be the filler (a launch that wrote nothing fails), and changed input values on
//! the new image must change the output (a comparison that cannot fail fails).
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, sha256};

/// Mirrors of `gpu/src/expert_ops.rs` (repr(C)): Shape 40 bytes, Arguments 96, LaunchArguments 112.
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, rows: u32, cols: u32, block_rows: u32, block_cols: u32, tokens: u32, global: f32, activation: f32, bound: f32, role: u32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, scales: u64, output: u64, shape: Shape }
#[repr(C)] #[derive(Clone, Copy)]
struct LaunchArguments { single: Arguments, batch: u64, count: u32 }
const _: () = assert!(size_of::<Shape>() == 40 && size_of::<Arguments>() == 96 && size_of::<LaunchArguments>() == 112);
/// `expert_ops::Op`, in the enum's order.
const QUANTIZE: u32 = 0;
const NV_PROJECT: u32 = 1;
const FP8_PROJECT: u32 = 2;
const SWIGLU: u32 = 3;
const FOLD: u32 = 4;
const QUANTIZE_ROWS: u32 = 5;
const NV_ROWS: u32 = 6;
const FP8_ROWS: u32 = 7;
const FP8_SERIAL: u32 = 8;
const PLAIN_SERIAL: u32 = 9;
/// `expert_ops::QUANTIZE_LANES_THREADS`: a Quantize launch on blocks of this size takes the lanes body.
const LANES_THREADS: u32 = 128;
/// An absent operand: an aligned address that is not null (`backend/expert_ops.rs`, `expert_arguments`).
const ABSENT: u64 = 4;

impl Shape {
    fn new(op: u32, rows: usize, cols: usize) -> Self { Shape { op, rows: rows as u32, cols: cols as u32, block_rows: 0, block_cols: 0, tokens: 1, global: 1.0, activation: 1.0, bound: 0.0, role: 0 } }
    /// `expert_ops::Shape::lengths` for the shapes of this check: FP32 words of the four inputs, weight bytes, scale
    /// bytes, output units, words a unit.
    fn lengths(&self) -> ([usize; 4], usize, usize, usize, usize) {
        let (r, k, m) = (self.rows as usize, self.cols as usize, self.tokens as usize);
        let blocks = || r.div_ceil(self.block_rows as usize) * k.div_ceil(self.block_cols as usize) * 4;
        match self.op {
            QUANTIZE => ([k, 0, 0, 0], 0, 0, k / 16, 17),
            NV_PROJECT => ([k / 16 * 17, 0, 0, 0], r * k / 2, r * k / 16, r, 1),
            FP8_PROJECT | FP8_SERIAL => ([k, 0, 0, 0], r * k, blocks(), r, 1),
            PLAIN_SERIAL => ([k, 0, 0, 0], r * k * 4, 0, r, 1),
            SWIGLU => ([r, r, 0, 0], 0, 0, r, 1),
            FOLD => ([r * k, k, k, r], 0, 0, r, 1),
            QUANTIZE_ROWS => ([m * k, 0, 0, 0], 0, 0, m * k / 16, 17),
            NV_ROWS => ([m * (k / 16) * 17, 0, 0, 0], r * k / 2, r * k / 16, r * m, 1),
            FP8_ROWS => ([m * k, 0, 0, 0], r * k, blocks(), r * m, 1),
            _ => unreachable!("no such expert operation in this check"),
        }
    }
    /// Grid and block size the entry is launched with: the serial body a warp of 32 rows, an NVFP4 or block FP8 table
    /// eight rows a warp (`projection_batch_index`; this check uses no packed NVFP4 table), every other body one unit a
    /// thread; an NVFP4 job alone eight rows a warp (more blocks than a packed shape needs: its spare warps own nothing).
    fn geometry(&self, count: usize, lanes: bool) -> ([u32; 3], u32) {
        let (_, _, _, units, _) = self.lengths();
        let blocks = if lanes { units.div_ceil(LANES_THREADS as usize / 16) } else {
            match self.op { FP8_SERIAL | PLAIN_SERIAL => (self.rows as usize).div_ceil(32) * count, NV_PROJECT => (self.rows as usize).div_ceil(8) * count, FP8_PROJECT if count > 1 => (self.rows as usize).div_ceil(8) * count, _ => units.div_ceil(32) }
        };
        ([blocks as u32, 1, 1], if lanes { LANES_THREADS } else { 32 })
    }
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: u64) -> u64 { (self.next() >> 11) % n }
    /// A finite value: zeros, small values and values up to `span` in magnitude.
    fn value(&mut self, span: f32) -> f32 {
        let unit = (self.below(1 << 20) as f32 / (1 << 19) as f32) - 1.0;
        match self.below(16) { 0 => 0.0, 1 => -0.0, 2 => unit * 1e-6, 3 => unit * 1e-3, _ => unit * span }
    }
    fn values(&mut self, n: usize, span: f32) -> Vec<f32> { (0..n).map(|_| self.value(span)).collect() }
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn words(bytes: &[u8]) -> Vec<f32> { bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() }
fn descriptor(a: &Arguments) -> Vec<u8> {
    let mut v = Vec::with_capacity(96);
    for p in a.input.into_iter().chain([a.weight, a.scales, a.output]) { v.extend(p.to_le_bytes()); }
    let s = a.shape;
    for p in [s.op, s.rows, s.cols, s.block_rows, s.block_cols, s.tokens] { v.extend(p.to_le_bytes()); }
    for p in [s.global, s.activation, s.bound] { v.extend(p.to_le_bytes()); }
    v.extend(s.role.to_le_bytes());
    v
}
/// One job's operands on the host: up to four FP32 inputs, weight bytes, scale bytes.
#[derive(Clone)]
struct Job { shape: Shape, input: [Vec<f32>; 4], weight: Vec<u8>, scales: Vec<u8> }
const FILLER: u8 = 0xa5;
/// Launches `jobs` (one job: no table; more: a table, as the host's projection batch) on `entry`; the output bytes.
fn launch(g: &mut Gpu, entry: H, jobs: &[Job], lanes: bool) -> Result<Vec<u8>> {
    let shape = jobs[0].shape;
    let (_, _, _, units, unit_words) = shape.lengths();
    let each = units * unit_words * 4;
    let output = g.upload(&vec![FILLER; each * jobs.len()])?;
    let mut descriptors = vec![];
    for (index, job) in jobs.iter().enumerate() {
        let mut input = [ABSENT; 4];
        for (slot, values) in job.input.iter().enumerate() { if !values.is_empty() { input[slot] = g.upload(&le(values))?; } }
        let weight = if job.weight.is_empty() { ABSENT } else { g.upload(&job.weight)? };
        let scales = if job.scales.is_empty() { ABSENT } else { g.upload(&job.scales)? };
        descriptors.push(Arguments { input, weight, scales, output: output + (index * each) as u64, shape: job.shape });
    }
    let batch = if jobs.len() == 1 { 0 } else { g.upload(&descriptors.iter().flat_map(descriptor).collect::<Vec<_>>())? };
    let mut arguments = LaunchArguments { single: descriptors[0], batch, count: jobs.len() as u32 };
    let (grid, threads) = shape.geometry(jobs.len(), lanes);
    g.launch(entry, grid, threads, 0, &mut [(&mut arguments as *mut LaunchArguments).cast()])?;
    g.sync()?;
    g.read(output, each * jobs.len())
}
/// The job's operands by the rules of the host reference. `quantized`: an NVFP4 activation the old image's quantizer wrote.
fn job(r: &mut Rng, shape: Shape, quantized: Option<&[f32]>) -> Job {
    let (lengths, weight_bytes, scale_bytes, _, _) = shape.lengths();
    let mut input: [Vec<f32>; 4] = Default::default();
    for (slot, n) in lengths.into_iter().enumerate() { input[slot] = r.values(n, 8.0); }
    if let Some(q) = quantized { assert_eq!(q.len(), lengths[0]); input[0] = q.to_vec(); }
    // Fold: the second input holds the slot ids, each once (the kernel takes the slots in ascending id order).
    if shape.op == FOLD { let k = shape.cols as usize; let mut ids: Vec<f32> = (0..k).map(|i| i as f32).collect(); for i in (1..k).rev() { ids.swap(i, r.below(i as u64 + 1) as usize); } input[1] = ids; }
    let weight: Vec<u8> = match shape.op {
        // E4M3 codes, never a NaN code (0x7f, 0xff).
        FP8_PROJECT | FP8_SERIAL | FP8_ROWS => (0..weight_bytes).map(|_| { let c = r.next() as u8; if c & 127 == 127 { c ^ 1 } else { c } }).collect(),
        PLAIN_SERIAL => le(&r.values(weight_bytes / 4, 2.0)),
        _ => (0..weight_bytes).map(|_| r.next() as u8).collect(),
    };
    let scales: Vec<u8> = match shape.op {
        // Positive finite FP32 block scales from 2^-6 to 2^2.
        FP8_PROJECT | FP8_SERIAL | FP8_ROWS => le(&(0..scale_bytes / 4).map(|_| f32::from_bits(((121 + r.below(8) as u32) << 23) | (r.next() as u32 & 0x7f_ffff))).collect::<Vec<_>>()),
        // UE4M3 codes under 127.
        _ => (0..scale_bytes).map(|_| r.below(127) as u8).collect(),
    };
    Job { shape, input, weight, scales }
}
/// The same job with other values in the first sixteen words of its first input (still valid for the operation).
fn changed(job: &Job) -> Job {
    let mut other = job.clone();
    let grid = matches!(job.shape.op, NV_PROJECT | NV_ROWS);
    // An NVFP4 activation stays on its grid: 1.0 and 2.0 are E2M1 values. The seventeenth word (the block scale) stays.
    for (i, v) in other.input[0].iter_mut().take(16).enumerate() { *v = if grid { if *v == 1.0 { 2.0 } else { 1.0 } } else { *v + 1.0 + i as f32 }; }
    other
}

pub(crate) fn run(old_path: &str, new_path: &str) -> Result<()> {
    let (a, b) = (std::fs::read(old_path).map_err(|e| e.to_string())?, std::fs::read(new_path).map_err(|e| e.to_string())?);
    println!("EXPERT PAIR old_sha256={} new_sha256={} kernel=expert_operation", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?;
    let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let entries = [g.function(ma, "expert_operation")?, g.function(mb, "expert_operation")?];
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    let (mut pass, mut cases, mut bytes) = (true, 0usize, 0usize);
    // One case: the jobs on the old and on the new image, then the changed jobs on the new image.
    let mut case = |g: &mut Gpu, name: &str, jobs: &[Job], lanes: bool| -> Result<Vec<u8>> {
        let old = launch(g, entries[0], jobs, lanes)?;
        let new = launch(g, entries[1], jobs, lanes)?;
        let other: Vec<Job> = jobs.iter().map(changed).collect();
        let control = launch(g, entries[1], &other, lanes)?;
        // Written: no output word is still the filler word (every unit of every job is owned by a thread).
        let (written, equal, fires) = (!new.chunks_exact(4).any(|w| w == [FILLER; 4]), old == new, control != new);
        let differ = old.chunks_exact(4).zip(new.chunks_exact(4)).filter(|(x, y)| x != y).count();
        let ok = written && equal && fires;
        let s = jobs[0].shape;
        println!("EXPERT PAIR {} {name} op={} rows={} cols={} tokens={} blocks={}x{} role={} jobs={} lanes={lanes} words={} words_old_ne_new={differ} output_written={written} control_changes_output={fires}", if ok { "ok" } else { "FAIL" }, s.op, s.rows, s.cols, s.tokens, s.block_rows, s.block_cols, s.role, jobs.len(), new.len() / 4);
        pass &= ok; cases += 1; bytes += new.len();
        g.clear()?;
        Ok(old)
    };
    let with = |shape: Shape, f: &dyn Fn(&mut Shape)| { let mut s = shape; f(&mut s); s };
    // The quantizers. Their outputs (from the old image) are the NVFP4 activations of the cases below.
    for (cols, activation) in [(512usize, 0.75f32), (256, 1.0), (4096, 0.3125)] {
        let s = with(Shape::new(QUANTIZE, 1, cols), &|s| s.activation = activation);
        let j = job(&mut r, s, None);
        case(&mut g, "quantize", std::slice::from_ref(&j), false)?;
        if cols <= 1024 { case(&mut g, "quantize-lanes", std::slice::from_ref(&j), true)?; }
    }
    let quantize = |g: &mut Gpu, r: &mut Rng, case: &mut dyn FnMut(&mut Gpu, &str, &[Job], bool) -> Result<Vec<u8>>, cols: usize, tokens: usize| -> Result<Vec<f32>> {
        let s = with(Shape::new(if tokens == 1 { QUANTIZE } else { QUANTIZE_ROWS }, 1, cols), &|s| { s.tokens = tokens as u32; s.activation = 0.5; });
        Ok(words(&case(g, if tokens == 1 { "quantize-for-nv" } else { "quantize-rows" }, &[job(r, s, None)], false)?))
    };
    // NVFP4 projections: a first matrix, a second matrix (the quarter chain; a packed shape), a table of three.
    let q512 = quantize(&mut g, &mut r, &mut case, 512, 1)?;
    let q256 = quantize(&mut g, &mut r, &mut case, 256, 1)?;
    let first = with(Shape::new(NV_PROJECT, 300, 512), &|s| { s.global = 0.75; s.activation = 0.5; });
    let j = job(&mut r, first, Some(&q512));
    case(&mut g, "nv-project-first", std::slice::from_ref(&j), false)?;
    let second = with(Shape::new(NV_PROJECT, 64, 256), &|s| { s.global = 1.25; s.activation = 0.5; s.role = 1; });
    case(&mut g, "nv-project-second", &[job(&mut r, second, Some(&q256))], false)?;
    let table: Vec<Job> = (0..3).map(|_| job(&mut r, first, Some(&q512))).collect();
    case(&mut g, "nv-project-table", &table, false)?;
    // NVFP4 rows: five token rows, both roles.
    let q5 = quantize(&mut g, &mut r, &mut case, 256, 5)?;
    for role in [0u32, 1] {
        let s = with(Shape::new(NV_ROWS, 40, 256), &|s| { s.tokens = 5; s.global = 0.75; s.activation = 0.5; s.role = role; });
        case(&mut g, "nv-rows", &[job(&mut r, s, Some(&q5))], false)?;
    }
    // Block FP8: the generic projection (rounded, unrounded; a table), the rows, the serial body (rounded, unrounded,
    // raw quarter; a table).
    let fp8 = |op: u32, rows: usize, cols: usize, role: u32, tokens: u32| with(Shape::new(op, rows, cols), &|s| { s.block_rows = 128; s.block_cols = 128; s.role = role; s.tokens = tokens; s.global = 0.5; s.activation = 2.0; });
    for role in [0u32, 3] { case(&mut g, "fp8-project", &[job(&mut r, fp8(FP8_PROJECT, 200, 512, role, 1), None)], false)?; }
    let table: Vec<Job> = (0..3).map(|_| job(&mut r, fp8(FP8_PROJECT, 200, 512, 0, 1), None)).collect();
    case(&mut g, "fp8-project-table", &table, false)?;
    for role in [0u32, 3] { case(&mut g, "fp8-rows", &[job(&mut r, fp8(FP8_ROWS, 40, 512, role, 5), None)], false)?; }
    for role in [0u32, 3, 2] { case(&mut g, "fp8-serial", &[job(&mut r, fp8(FP8_SERIAL, 300, 4096, role, 1), None)], false)?; }
    let table: Vec<Job> = (0..3).map(|_| job(&mut r, fp8(FP8_SERIAL, 300, 1024, 0, 1), None)).collect();
    case(&mut g, "fp8-serial-table", &table, false)?;
    // Plain FP32 rows, SwiGLU, the fold.
    case(&mut g, "plain-serial", &[job(&mut r, Shape::new(PLAIN_SERIAL, 100, 512), None)], false)?;
    let table: Vec<Job> = (0..3).map(|_| job(&mut r, Shape::new(PLAIN_SERIAL, 100, 512), None)).collect();
    case(&mut g, "plain-serial-table", &table, false)?;
    case(&mut g, "swiglu", &[job(&mut r, with(Shape::new(SWIGLU, 1000, 1), &|s| s.bound = 30.0), None)], false)?;
    case(&mut g, "fold", &[job(&mut r, Shape::new(FOLD, 300, 8), None)], false)?;
    println!("EXPERT PAIR {} cases={cases} output_bytes={bytes} (old image = new image in every case; every launch wrote its output; every control changed it)", if pass { "PASS" } else { "FAIL" });
    if pass { Ok(()) } else { Err("EXPERT PAIR FAIL".into()) }
}
