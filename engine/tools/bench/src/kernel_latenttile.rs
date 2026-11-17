//! Grouped latent absorption tile (`latent_absorption_tile`): the entry of one image against the entry of another.
//! Every output byte of the four fold operations (query, scores, values, project), with a wrong-input control and an
//! order control (the same products in another strip order) that must both be seen; then timing at the launch shapes
//! of an 8,192-token read. One process, one device. Safe code only. It does not prove token IDs.
//! Run: kernel-bench --latent-tile <base image> <candidate image> [bytes|time|all|probe=<entry>,<entry>...]
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)). The op numbers follow the enum order there:
/// AbsorbQuery 21, AbsorbScores 22, AbsorbValues 24, AbsorbProject 25. The kernel traps on any other op.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
const OPS: [(u32, &str); 4] = [(21, "query"), (22, "scores"), (24, "values"), (25, "project")];
/// The tile of gpu/src/latent_absorb_rows.rs: a block owns 32 A rows x 64 B rows of one outer index, 256 threads.
const TILE_A: usize = 32; const TILE_B: usize = 64; const THREADS: u32 = 256;
struct Rng(u64);
impl Rng {
    fn step(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.step() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// The activation classes of kernel_plainp.rs: 0 plain, 2 denormals and signed zeros, 4 overflow with NaN and inf,
/// 5 huge/tiny alternation, 6 add-order sensitive (2^24 steps, cancelling pairs). 9: probabilities in [0, 1).
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.step();
        match kind {
            0 => u * 4.0,
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i % 4099 == 7 { f32::NAN } else if i % 4099 == 11 { f32::INFINITY } else if i % 4099 == 13 { f32::NEG_INFINITY } else { v } }
            5 => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
            6 => match i % 4 { 0 => 16777216.0, 1 => 1.0 + (s % 3) as f32, 2 => -16777216.0, _ => 1.0 },
            _ => (u + 1.0) * 0.5,
        }
    }).collect()
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// Two BF16 patterns a word, as the loader packs the key and value weights.
fn bf16_bits(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect() }
/// 16-byte guards on both sides; the data address stays 16-byte aligned (allocations are 256-byte aligned).
#[derive(Clone, Copy)]
struct Buffer { base: u64, address: u64, bytes: usize }
impl Buffer {
    fn raw(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn new(g: &mut Gpu, values: &[f32]) -> Result<Self> { Self::raw(g, &le(values)) }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn fill(&self, g: &Gpu, byte: u8) -> Result<()> { g.upload_at(self.address, &vec![byte; self.bytes]) }
}
/// One launch shape. `rows` is the cache length after the group, `slots` the selection capacity.
#[derive(Clone, Copy)]
struct Case { tokens: usize, heads: usize, dim: usize, taps: usize, cols: usize, rows: usize, slots: usize }
impl Case {
    fn first(&self) -> usize { self.slots.min(self.rows) }
    fn model(tokens: usize, rows: usize) -> Self { Self { tokens, heads: 32, dim: 256, taps: 256, cols: 512, rows, slots: 2051 } }
    /// (outer count, A rows, B rows) and the output words of one operation, as latent_absorb_rows::tile and lengths.
    fn tile(&self, op: u32) -> (usize, usize, usize, usize) {
        let (m, h, v, k, c) = (self.tokens, self.heads, self.taps, self.cols, self.first());
        match op { 21 => (h, m, k, m * h * k), 25 => (h, m, v, m * h * v), 22 => (m, h, c, m * h * c), _ => (m, h, k, m * h * k) }
    }
    fn blocks(&self, op: u32) -> Result<u32> {
        let (outer, a, b, _) = self.tile(op);
        u32::try_from(outer * a.div_ceil(TILE_A) * b.div_ceil(TILE_B)).map_err(|_| "latent tile grid overflow".to_string())
    }
}
/// Selected rows `[tokens, slots+1]`, the count last. `served`: token t sees context rows-tokens+t+1 and keeps
/// 4 x min(context/4, 512) pool rows plus the tail rows, as the bounded decoder does (pools spread over the context,
/// in selection order, not sorted). Otherwise: counts of every residue, some zero, IDs spread over the cache.
fn selection(c: &Case, served: bool, seed: u64) -> Vec<f32> {
    let (m, r, n, cap) = (c.tokens, c.rows, c.slots, c.first()); let mut out = vec![0.0f32; m * (n + 1)];
    for t in 0..m {
        let row = &mut out[t * (n + 1)..(t + 1) * (n + 1)]; let mut count = 0;
        if served {
            let context = r - m + t + 1; let pools = context / 4; let kept = pools.min(512);
            let mut stride = (pools * 5 / 8) | 1; while gcd(stride, pools.max(1)) != 1 { stride += 2; }
            for j in 0..kept { let pool = if pools <= 512 { j } else { (j * stride + t * 7) % pools }; for e in 0..4 { if count < cap { row[count] = (pool * 4 + e) as f32; count += 1; } } }
            for e in 0..context % 4 { if count < cap { row[count] = (pools * 4 + e) as f32; count += 1; } }
        } else {
            let want = if t % 11 == 5 { 0 } else { 1 + (t * 37 + seed as usize * 13) % cap };
            let mut stride = (r * 3 / 7) | 1; while gcd(stride, r) != 1 { stride += 2; }
            for j in 0..want.min(r) { row[count] = ((j * stride + t * 29 + seed as usize) % r) as f32; count += 1; }
        }
        row[n] = count as f32;
    }
    out
}
fn gcd(a: usize, b: usize) -> usize { if b == 0 { a } else { gcd(b, a % b) } }
/// The operands of the four operations of one shape. Each operation has its own synthetic operands.
struct Fixture { query: Buffer, absorbed: Buffer, probability: Buffer, mixed: Buffer, latent: Buffer, selected: Buffer, weight: Buffer }
fn fixture(g: &mut Gpu, c: &Case, kind: u32, served: bool, seed: u64) -> Result<Fixture> {
    let (m, h, d, v, k, r) = (c.tokens, c.heads, c.dim, c.taps, c.cols, c.rows);
    Ok(Fixture { query: Buffer::new(g, &pattern(kind, m * h * d, seed))?, absorbed: Buffer::new(g, &pattern(kind, m * h * k, seed + 1))?,
        probability: Buffer::new(g, &pattern(if kind == 0 { 9 } else { kind }, m * h * c.first(), seed + 2))?, mixed: Buffer::new(g, &pattern(kind, m * h * k, seed + 3))?,
        latent: Buffer::new(g, &pattern(kind, r * k, seed + 4))?, selected: Buffer::new(g, &selection(c, served, seed))?,
        weight: Buffer::raw(g, &bf16_bits(&pattern(kind, h * (d + v) * k, seed + 5)))? })
}
fn arguments(c: &Case, op: u32, fx: &Fixture, output: u64) -> Arguments {
    let inner = Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    let (m, h, d, v, k, r, n) = (c.tokens, c.heads, c.dim, c.taps, c.cols, c.rows, c.slots);
    let (selected, weights) = ((m * (n + 1)) as u64, (h * (d + v) * k / 2) as u64);
    let (input, lengths, weight) = match op {
        21 => ([fx.query.address, 4, 4, 4, 4], [(m * h * d) as u64, 0, 0, 0, 0], fx.weight.address),
        22 => ([fx.absorbed.address, fx.latent.address, fx.selected.address, 4, 4], [(m * h * k) as u64, (r * k) as u64, selected, 0, 0], 4),
        24 => ([fx.probability.address, fx.latent.address, fx.selected.address, 4, 4], [(m * h * c.first()) as u64, (r * k) as u64, selected, 0, 0], 4),
        _ => ([fx.mixed.address, fx.selected.address, 4, 4, 4], [(m * h * k) as u64, selected, 0, 0, 0], fx.weight.address),
    };
    Arguments { input, lengths, weight, weights: if weight == 4 { 0 } else { weights }, output, units: c.tile(op).3 as u64, words: 1,
        shape: PShape { op, tokens: m as u32, heads: h as u32, dim: d as u32, taps: v as u32, rows: r as u32, cols: k as u32, first: c.first() as u32, slots: n as u32, inner } }
}
fn launch(g: &Gpu, f: H, c: &Case, a: &mut Arguments) -> Result<()> { g.launch(f, [c.blocks(a.shape.op)?, 1, 1], THREADS, 0, &mut [ptr(a)]) }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn guarded(x: &[u8]) -> bool { x[..16] == GUARD && x[x.len() - 16..] == GUARD }
/// Both entries on the same operands; both outputs poisoned first with different bytes, so an unwritten word differs.
/// Returns the output bytes compared.
fn compare(g: &mut Gpu, fns: (H, H), c: &Case, fx: &Fixture, label: &str) -> Result<u64> {
    let mut bytes = 0;
    for (op, name) in OPS {
        let words = c.tile(op).3; let (a, b) = (Buffer::new(g, &vec![0.0; words])?, Buffer::new(g, &vec![0.0; words])?);
        a.fill(g, 0xff)?; b.fill(g, 0xee)?;
        launch(g, fns.0, c, &mut arguments(c, op, fx, a.address))?; launch(g, fns.1, c, &mut arguments(c, op, fx, b.address))?; g.sync()?;
        let (x, y) = (a.read(g)?, b.read(g)?);
        if !guarded(&x) || !guarded(&y) { return Err(format!("LATENTTILE output guard changed op={name} {label}")); }
        if x != y {
            let word = x.chunks_exact(4).zip(y.chunks_exact(4)).position(|(p, q)| p != q).unwrap_or(0).saturating_sub(4);
            return Err(format!("LATENTTILE BYTE FAIL op={name} {label} first_word={word} of {words}"));
        }
        bytes += (words * 4) as u64;
    }
    Ok(bytes)
}
/// Scores of the candidate entry on changed operands against its own scores: the number of output words that differ.
fn scores_differ(g: &mut Gpu, f: H, c: &Case, fx: &Fixture, absorbed: &[f32], latent: &[f32]) -> Result<usize> {
    let words = c.tile(22).3; let (a, b) = (Buffer::new(g, &vec![0.0; words])?, Buffer::new(g, &vec![0.0; words])?);
    let changed = Fixture { query: Buffer::new(g, &[0.0])?, absorbed: Buffer::new(g, absorbed)?, probability: Buffer::new(g, &[0.0])?, mixed: Buffer::new(g, &[0.0])?,
        latent: Buffer::new(g, latent)?, selected: fx.selected, weight: Buffer::new(g, &[0.0])? };
    launch(g, f, c, &mut arguments(c, 22, fx, a.address))?; launch(g, f, c, &mut arguments(c, 22, &changed, b.address))?; g.sync()?;
    let (x, y) = (a.read(g)?, b.read(g)?);
    Ok(x.chunks_exact(4).zip(y.chunks_exact(4)).filter(|(p, q)| p != q).count())
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (ia, ib) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("LATENTTILE baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&ia), sha256::digest(&ib));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&ia)?, g.module(&ib)?);
    let fns = (g.function(ma, "latent_absorption_tile")?, g.function(mb, "latent_absorption_tile")?);
    let started = Instant::now();
    if mode == "bytes" || mode == "all" {
        // Controls first, on the candidate entry alone, at the model's shape with 64 tokens.
        // 1. Wrong input: one absorbed-query word of the last token and head changes. Its scores must change.
        // 2. Order: columns 3 and 11 trade places in both operands, so every product is kept and two terms move
        //    between strips 0 and 1. A kernel that keeps SUM-ORDER v1 must give other bits for some outputs.
        let c = Case::model(64, 2112); let fx = fixture(&mut g, &c, 0, true, 7)?;
        let (m, h, k, r) = (c.tokens, c.heads, c.cols, c.rows);
        let (absorbed, latent) = (pattern(0, m * h * k, 8), pattern(0, r * k, 11));
        let mut wrong = absorbed.clone(); let at = m * h * k - k + 5; wrong[at] = f32::from_bits(wrong[at].to_bits() ^ 0x0040_0000);
        let seen = scores_differ(&mut g, fns.1, &c, &fx, &wrong, &latent)?;
        if seen == 0 { return Err("LATENTTILE NEGATIVE FAIL: a changed input word changed no output".into()); }
        println!("LATENTTILE NEGATIVE PASS wrong-input detected different_words={seen}");
        let swap = |values: &[f32]| { let mut v = values.to_vec(); for row in v.chunks_exact_mut(k) { row.swap(3, 11); } v };
        let moved = scores_differ(&mut g, fns.1, &c, &fx, &swap(&absorbed), &swap(&latent))?;
        if moved == 0 { return Err("LATENTTILE ORDER CONTROL FAIL: another strip order gave the same bytes".into()); }
        println!("LATENTTILE ORDER CONTROL PASS same products in another strip order: different_words={moved} of {}", c.tile(22).3);
        g.clear()?;
        let (mut comparisons, mut bytes) = (0u64, 0u64); let mut seed = 100u64;
        // The model's shape (32 local heads, key and value 256, latent 512, 2,051 slots) at 16, 64, 256, 512, 1,024 and 2,048
        // tokens: the first group of a prompt (cache = the group), a group after 2,048 rows, and the last group of
        // an 8,192-token read; every activation class; the served selection and the every-count selection.
        for kind in [0u32, 2, 4, 5, 6] {
            for tokens in [16usize, 64, 256, 512, 1024, 2048] {
                let mut shapes = vec![(tokens, true), (2048 + tokens, true), (tokens + 3, false)];
                if tokens >= 1024 { shapes.push((8192, true)); }
                for (rows, served) in shapes {
                    seed += 1; let c = Case::model(tokens, rows); let fx = fixture(&mut g, &c, kind, served, seed)?;
                    bytes += compare(&mut g, fns, &c, &fx, &format!("tokens={tokens} rows={rows} kind={kind} served={served}"))?; comparisons += 4; g.clear()?;
                }
            }
            println!("LATENTTILE kind={kind} model shape ok comparisons_so_far={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        }
        // Edges outside the model: partial tiles and quads, odd dims, one token, one head, counts of every residue.
        for (tokens, heads, dim, taps, cols, rows, slots) in [(37usize, 3usize, 20usize, 36usize, 70usize, 50usize, 45usize), (1, 1, 4, 4, 2, 1, 1), (65, 33, 257, 255, 514, 300, 131),
            (33, 2, 8, 12, 34, 700, 300), (5, 64, 256, 256, 512, 9, 2051), (130, 1, 1, 1, 6, 131, 7)] {
            for kind in [0u32, 2, 4, 5, 6] {
                seed += 1; let c = Case { tokens, heads, dim, taps, cols, rows, slots }; let fx = fixture(&mut g, &c, kind, false, seed)?;
                bytes += compare(&mut g, fns, &c, &fx, &format!("edge tokens={tokens} heads={heads} dim={dim} taps={taps} cols={cols} rows={rows} slots={slots} kind={kind}"))?; comparisons += 4; g.clear()?;
            }
        }
        println!("LATENTTILE BYTES PASS comparisons={comparisons} output_bytes={bytes} tokens=16,64,256,512,1024,2048 elapsed_s={:.1}", started.elapsed().as_secs_f64());
    }
    // Timing: an 8,192-token read with eight 1,024-row groups or four 2,048-row leased groups, served selection.
    let probes: Vec<(String, H)> = match mode.strip_prefix("probe=") {
        Some(names) => names.split(',').map(|n| Ok((n.to_string(), g.function(mb, n)?))).collect::<Result<_>>()?,
        None => vec![],
    };
    if mode == "time" || mode == "all" || !probes.is_empty() {
        let group_sizes=if probes.is_empty() {vec![1024usize,2048]} else {vec![1024]};
        for tokens in group_sizes {
        let mut saved = [0.0f64; 4]; let mut base_total = [0.0f64; 4];
        let groups: Vec<usize> = if probes.is_empty() { (1..=8192/tokens).map(|i| i * tokens).collect() } else { vec![1024, 4096, 8192] };
        for rows in groups {
            let c = Case::model(tokens, rows); let fx = fixture(&mut g, &c, 0, true, 31)?;
            for (i, (op, name)) in OPS.iter().enumerate() {
                let out = Buffer::new(&mut g, &vec![0.0; c.tile(*op).3])?; let mut a = arguments(&c, *op, &fx, out.address);
                let mut arms: Vec<(String, H)> = vec![("base".into(), fns.0), ("candidate".into(), fns.1)]; arms.extend(probes.iter().cloned());
                let mut times = vec![vec![]; arms.len()];
                for f in arms.iter().map(|x| x.1) { launch(&g, f, &c, &mut a)?; g.sync()?; }
                for repeat in 0..7 {
                    for k in 0..arms.len() {
                        let arm = (repeat + k) % arms.len(); g.sync()?; let t = Instant::now();
                        launch(&g, arms[arm].1, &c, &mut a)?; g.sync()?; times[arm].push(t.elapsed().as_secs_f64() * 1e6);
                    }
                }
                // A probe entry is also compared with the base entry on this shape: SAME or DIFFERENT, never an error
                // (a timing probe may be wrong on purpose).
                let mut verdicts = String::new();
                if !probes.is_empty() {
                    let mut reference = vec![];
                    for (k, arm) in arms.iter().enumerate() {
                        if k == 1 { continue; }
                        out.fill(&g, 0xee)?; launch(&g, arm.1, &c, &mut a)?; g.sync()?; let bytes = out.read(&g)?;
                        if k == 0 { reference = bytes; } else { verdicts += &format!(" {}_bytes={}", arm.0, if bytes == reference { "SAME" } else { "DIFFERENT" }); }
                    }
                }
                let medians: Vec<f64> = times.into_iter().map(median).collect();
                saved[i] += medians[0] - medians[1]; base_total[i] += medians[0];
                let extra: String = arms.iter().zip(&medians).skip(2).map(|(a, m)| format!(" {}_us={m:.1}", a.0)).collect();
                println!("LATENTTILE TIME op={name} tokens={tokens} rows={rows} first={} blocks={} base_us={:.1} candidate_us={:.1} ratio={:.4}{extra}{verdicts}", c.first(), c.blocks(*op)?, medians[0], medians[1], medians[0] / medians[1]);
            }
            g.clear()?;
        }
        if probes.is_empty() {
            // One absorbed layer runs each operation once a group; the model has 11 absorbed layers.
            for (i, (_, name)) in OPS.iter().enumerate() { println!("LATENTTILE FORECAST op={name} tokens={tokens} base_ms_a_read={:.1} saved_ms_a_read={:.1}", base_total[i] * 11.0 / 1e3, saved[i] * 11.0 / 1e3); }
            println!("LATENTTILE FORECAST 8192-token read, tokens={tokens}, 11 layers, one rank: base_ms={:.1} saved_ms={:.1} (bench medians, not a served run)", base_total.iter().sum::<f64>() * 11.0 / 1e3, saved.iter().sum::<f64>() * 11.0 / 1e3);
        }
        }
    }
    println!("LATENTTILE PASS elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
