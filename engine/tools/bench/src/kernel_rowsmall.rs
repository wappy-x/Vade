//! Same-bytes sweep and timing for the small-count plain token rows (`plain_rows_small`, 1 to 16 tokens) against lane R's
//! `prompt_rows` ProjectRows fast path and lane K's `plain_rows_tile`, in one process on one device. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
use super::envelope as e;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op ProjectRows is 1, inner Project is 5.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// 0 plain, 1 wide range, 2 denormals and signed zeros, 3 large finite, 4 overflow with NaN and inf, 5 huge/tiny alternation.
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.next();
        match kind {
            0 => u * 4.0,
            1 => u * f32::from_bits(((s % 60 + 97) as u32) << 23),
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(60),
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i % 4099 == 7 { f32::NAN } else if i % 4099 == 11 { f32::INFINITY } else { v } }
            _ => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
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
    fn zero(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0; self.bytes]) }
}
fn args(input: &Buffer, weight: &Buffer, output: &Buffer, rows: usize, tokens: usize, cols: usize) -> Arguments {
    let inner = Inner { op: 5, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    Arguments { input: [input.address, 0, 0, 0, 0], lengths: [(tokens * cols) as u64, 0, 0, 0, 0], weight: weight.address, weights: 0, output: output.address,
        units: (rows * tokens) as u64, words: 1, shape: PShape { op: 1, tokens: tokens as u32, heads: 0, dim: 0, taps: 0, rows: rows as u32, cols: cols as u32, first: 0, slots: 0, inner } }
}
fn launch_reference(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { g.launch(f, [(a.units as usize).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(a)]) }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn launch_tile(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> {
    let blocks = (a.shape.tokens as usize).div_ceil(64) * (a.shape.rows as usize).div_ceil(64);
    g.launch(f, [blocks as u32, 1, 1], 256, 0, &mut [ptr(a)])
}
/// Mirror of gpu/src/plain_small.rs: 32 rows a block, one warp for each 4 tokens.
const MAX_TOKENS: usize = 16;
fn launch_small(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { launch_rows(g, f, a, 32) }
/// Mirror of gpu/src/plain_small.rs `few_blocks`: `plain_rows_few` takes 8 rows a block (and 1 to 4 tokens, at most 512 rows).
const FEW_ROWS_PER_BLOCK: usize = 8;
fn launch_few(g: &Gpu, f: H, a: &mut Arguments) -> Result<()> { launch_rows(g, f, a, FEW_ROWS_PER_BLOCK) }
fn launch_rows(g: &Gpu, f: H, a: &mut Arguments, rows_per_block: usize) -> Result<()> {
    let rows = a.shape.rows as usize;
    g.launch(f, [rows.div_ceil(rows_per_block) as u32, 1, 1], 128, 0, &mut [ptr(a)])
}
/// Compare the reference and the small kernel on one shape. Returns after poisoning both outputs first.
fn compare(g: &Gpu, reference: H, small: H, input: &Buffer, weight: &Buffer, out_a: &Buffer, out_b: &Buffer, rows: usize, tokens: usize, cols: usize, rows_per_block: usize, label: &str) -> Result<u64> {
    out_a.poison(g)?; out_b.poison(g)?;
    let (mut xa, mut xb) = (args(input, weight, out_a, rows, tokens, cols), args(input, weight, out_b, rows, tokens, cols));
    launch_reference(g, reference, &mut xa)?; launch_rows(g, small, &mut xb, rows_per_block)?; g.sync()?;
    let (x, y) = (out_a.read(g)?, out_b.read(g)?);
    if x != y {
        let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i));
        return Err(format!("ROWSMALL BYTE FAIL {label} rows={rows} tokens={tokens} cols={cols} first_word_and_byte={at:?}"));
    }
    if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err(format!("ROWSMALL guard changed {label} rows={rows} tokens={tokens} cols={cols}")); }
    Ok((rows * tokens * 4) as u64)
}

/// KDA rows: StepRows (outer op 7, dim 128). Reference = head's `prompt_rows`.
#[derive(Clone, Copy)] enum Kda { Step }
type KdaSetup = ([usize; 3], usize, usize, usize, PShape, u32, u32);
fn kda_setup(kind: Kda, tokens: usize, h: usize, d: usize, _eps: f32) -> KdaSetup {
    let width = h * d;
    match kind {
        Kda::Step => ([tokens * h * (4 * d + 1), width * d, tokens * width], 0, d + tokens, width,
            PShape { op: 7, tokens: tokens as u32, heads: h as u32, dim: d as u32, taps: 0, rows: 0, cols: 0, first: 0, slots: 0, inner: Inner { op: 5, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 } },
            (h * 4) as u32, 32),
    }
}


fn kda_time(g: &mut Gpu, reference: H, cand: H, block_kernel: H, kind: Kda, tokens: usize, h: usize, d: usize) -> Result<(f64, f64, f64)> {
    let (lengths, weights, words, units, shape, grid, block) = kda_setup(kind, tokens, h, d, 1e-5);
    let inputs: Vec<Buffer> = lengths.iter().enumerate().map(|(i, &n)| Buffer::new(g, &pattern(0, n.max(1), 5 + i as u64))).collect::<Result<_>>()?;
    let weight = Buffer::new(g, &pattern(0, weights.max(1), 6))?; let out = Buffer::new(g, &vec![0.0; units * words])?;
    let build = || Arguments { input: [inputs[0].address, inputs[1].address, inputs[2].address, 0, 0], lengths: [lengths[0] as u64, lengths[1] as u64, lengths[2] as u64, 0, 0],
        weight: weight.address, weights: weights as u64, output: out.address, units: units as u64, words: words as u64, shape };
    // Arms: 0 prompt_rows, 1 the new kernel, 2 lane R's step_rows_block (StepRows only; the same time as arm 0 is reported otherwise).
    let arms = 3;
    let mut t = [vec![], vec![], vec![]];
    for repeat in 0..5 { for arm in (0..arms).map(|a| (a + repeat) % arms) {
        let n = 20; let t0 = Instant::now();
        for _ in 0..n { let mut x = build();
            match arm { 0 => g.launch(reference, [units.div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut x)])?, 1 => g.launch(cand, [grid, 1, 1], block, 0, &mut [ptr(&mut x)])?,
                _ => g.launch(block_kernel, [(h * d / 32) as u32, 1, 1], 32, 0, &mut [ptr(&mut x)])? } }
        g.sync()?; t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
    } }
    let [ra, rb, rc] = t.map(|v| if v.is_empty() { 0.0 } else { median(v) }); Ok((ra, rb, rc))
}
fn kda_case(g: &mut Gpu, reference: H, cand: H, kind: Kda, tokens: usize, h: usize, d: usize, eps: f32, data: u32, seed: u64) -> Result<u64> {
    let (lengths, weights, words, units, shape, grid, block) = kda_setup(kind, tokens, h, d, eps);
    // Plain values keep the arithmetic in range; the hard patterns (NaN, inf, denormals, overflow) are the other data kinds.
    let inputs: Vec<Buffer> = lengths.iter().enumerate().map(|(i, &n)| Buffer::new(g, &pattern(if data == 0 { 0 } else { data }, n.max(1), seed + i as u64 * 7919))).collect::<Result<_>>()?;
    let weight = Buffer::new(g, &pattern(data, weights.max(1), seed + 31))?;
    let total = units * words + if matches!(kind, Kda::Step) { 0 } else { 0 };
    let out_a = Buffer::new(g, &vec![f32::NAN; total])?; let out_b = Buffer::new(g, &vec![f32::NAN; total])?;
    out_a.zero(g)?; out_b.zero(g)?;
    let build = |out: &Buffer| Arguments { input: [inputs[0].address, inputs[1].address, inputs[2].address, 0, 0], lengths: [lengths[0] as u64, lengths[1] as u64, lengths[2] as u64, 0, 0],
        weight: weight.address, weights: weights as u64, output: out.address, units: units as u64, words: words as u64, shape };
    let (mut xa, mut xb) = (build(&out_a), build(&out_b));
    g.launch(reference, [units.div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut xa)])?;
    g.launch(cand, [grid, 1, 1], block, 0, &mut [ptr(&mut xb)])?; g.sync()?;
    let (x, y) = (out_a.read(g)?, out_b.read(g)?);
    if x != y {
        let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i));
        return Err(format!("ROWSMALL KDA BYTE FAIL {} tokens={tokens} heads={h} dim={d} data={data} first_word_and_byte={at:?}", match kind { Kda::Step => "step_rows" }));
    }
    if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("ROWSMALL KDA guard changed".into()); }
    Ok((total * 4) as u64)
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let time = mode == "time";
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("ROWSMALL baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let reference = g.function(ma, "prompt_rows")?; let tile = g.function(ma, "plain_rows_tile")?; let small = g.function(mb, "plain_rows_small")?; let few = g.function(mb, "plain_rows_few")?;
    let started = Instant::now(); let (mut comparisons, mut bytes) = (0u64, 0u64);
    if !time {
        // Every admitted shape: the product of the lists in `small_sweep` (the host predicates admit exactly that product), every token count.
        // Modes: "plain", "kda" (one run each, under two minutes); "" runs both.
        if mode == "" || mode == "plain" {
            for &cols in e::PLAIN_COLS {
                let kinds: &[(u32, u32)] = if cols <= 516 { &[(0, 0), (5, 5), (2, 2), (1, 1), (4, 0), (3, 3), (0, 4)] } else { &[(5, 5), (2, 2), (4, 0)] };
                let rows_max = *e::PLAIN_ROWS.last().unwrap();
                for &(ik, wk) in kinds {
                    let weight = Buffer::new(&mut g, &pattern(wk, rows_max * cols, 100 + cols as u64 + wk as u64))?;
                    let input = Buffer::new(&mut g, &pattern(ik, MAX_TOKENS * cols, 200 + cols as u64 * 3 + ik as u64))?;
                    let out_a = Buffer::new(&mut g, &vec![f32::NAN; rows_max * MAX_TOKENS])?; let out_b = Buffer::new(&mut g, &vec![f32::NAN; rows_max * MAX_TOKENS])?;
                    for &rows in e::PLAIN_ROWS { for tokens in 1..=MAX_TOKENS {
                        bytes += compare(&g, reference, small, &input, &weight, &out_a, &out_b, rows, tokens, cols, 32, &format!("patterns={ik},{wk}"))?; comparisons += 1;
                        // The shapes `plain_rows_few` takes (plain_small::few_admitted): at most 4 tokens and 512 rows of the lists.
                        if rows <= e::PLAIN_FEW_MAX_ROWS && tokens <= e::PLAIN_FEW_MAX_TOKENS {
                            bytes += compare(&g, reference, few, &input, &weight, &out_a, &out_b, rows, tokens, cols, FEW_ROWS_PER_BLOCK, &format!("few patterns={ik},{wk}"))?; comparisons += 1;
                        }
                    } }
                }
                println!("ROWSMALL plain cols={cols} ok comparisons_so_far={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64());
                g.clear()?;
            }
        }
        if mode == "" || mode == "kda" {
            // StepRows (dim 128) against `prompt_rows`: every listed head count, every token count, six data patterns.
            let kda = g.function(mb, "prompt_rows_small")?;
            for &h in e::STEP_HEADS { for tokens in 1..=MAX_TOKENS { for data in 0..6u32 {
                bytes += kda_case(&mut g, reference, kda, Kda::Step, tokens, h, e::STEP_DIM, 0.0, data, 555 + tokens as u64 * 3 + data as u64)?; comparisons += 1;
            } g.clear()?; } }
        }
        println!("ROWSMALL BYTES PASS mode={mode} comparisons={comparisons} output_bytes={bytes} digest={} elapsed_s={:.1}", sha256::digest(format!("{mode}{comparisons}{bytes}").as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    // Timing. Enough weight copies rotate that a call streams its weights from memory, as a layer's matrix does in a pass.
    println!("ROWSMALL TIME kda: kind tokens | reference_us small_us step_rows_block_us (lane R)");
    let kda = g.function(mb, "prompt_rows_small")?; let block_kernel = g.function(ma, "step_rows_block")?;
    for &tokens in &[1usize, 2, 4, 8, 16] {
        let (ra, rb, rc) = kda_time(&mut g, reference, kda, block_kernel, Kda::Step, tokens, 64, 128)?; g.clear()?;
        println!("ROWSMALL TIME kda step_rows h=64 d=128 tokens={tokens} reference_us={ra:.1} small_us={rb:.1} step_rows_block_us={rc:.1} ratio={:.2}", ra / rb);
    }
    println!("ROWSMALL TIME columns: rows cols tokens | reference_us tile_us small_us few_us (1 to 4 tokens, at most 512 rows) | small_over_tile");
    for &cols in &[512usize, 1024, 2048, 4096] {
        for &rows in &[8usize, 32, 64, 128, 192, 256, 512, 1024, 4096, 16384] {
            let copies = ((96usize << 20) / (rows * cols * 4)).clamp(2, 64);
            let weights = (0..copies).map(|c| Buffer::new(&mut g, &pattern(0, rows * cols, 1 + c as u64))).collect::<Result<Vec<_>>>()?;
            for &tokens in &[1usize, 2, 3, 4, 5, 8, 12, 16] {
                let input = Buffer::new(&mut g, &pattern(0, tokens * cols, 3))?; let out = Buffer::new(&mut g, &vec![0.0; rows * tokens])?;
                let mut t = [vec![], vec![], vec![], vec![]];
                for repeat in 0..5 {
                    let order = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 2, 0], [0, 2, 3, 1], [2, 0, 1, 3]][repeat];
                    for arm in order {
                        if arm == 3 && (tokens > e::PLAIN_FEW_MAX_TOKENS || rows > e::PLAIN_FEW_MAX_ROWS) { continue; }
                        let n = copies.min(24); let t0 = Instant::now();
                        for i in 0..n {
                            let mut x = args(&input, &weights[i % copies], &out, rows, tokens, cols);
                            match arm { 0 => launch_reference(&g, reference, &mut x)?, 1 => launch_tile(&g, tile, &mut x)?, 2 => launch_small(&g, small, &mut x)?, _ => launch_few(&g, few, &mut x)? }
                        }
                        g.sync()?; t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                    }
                }
                let [r, ti, s, f] = t.map(|v| if v.is_empty() { f64::NAN } else { median(v) });
                println!("ROWSMALL TIME rows={rows} cols={cols} tokens={tokens} reference_us={r:.1} tile_us={ti:.1} small_us={s:.1} few_us={f:.1} small_over_tile={:.2}", s / ti);
            }
            g.clear()?;
        }
    }
    println!("ROWSMALL TIME DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
