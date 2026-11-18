//! ROUTE ROWS device gate: `device_route_rows` (the route build for `tokens` rows of one step, one launch, pair
//! order) against `device_route_build` of the same image run once a token. Both arms then project with `nv_cols`.
//! Byte check of every output word and of every job descriptor's matrix words, then timing at the model's shapes.
//! One process, one device. Run: kernel-bench --route-rows <image> [bytes|time|all|build=<base image>|router=<base image>|sum|refuse-<kind>]. Safe code only.
//! `sum` (RESIDUAL SUM, RB4 2026-11-18): the device form of the drafting layer's BF16 residual add against the host form (`residual_sum`).
//! `build=<base image>` (ROUTE BOOKKEEPING, DRAFTR 2026-11-17): the bytes above, then `device_route_rows` of this image
//! against that of the base image, alone: every descriptor, quantizer and quantized word equal, then both timed.
//! `router=<base image>` (RB4 2026-11-17): `prompt_route` (the rows router) of this image against that of the base image,
//! every output word equal, then both timed (`router_ab`); then this image's ROUTE PLANES form against the base image's rows
//! (`router_planes`); then the one-row router against `prompt_route` at one token (`ranked_ab`).
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
/// Mirrors of gpu/src/expert_ops.rs and gpu/src/device_route.rs (repr(C)): Shape 40 bytes, Arguments 96,
/// LaunchArguments 112, Build 80, BuildRows 88. Quantize is op 0, NvProject op 1; role 1 is a second matrix.
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, rows: u32, cols: u32, block_rows: u32, block_cols: u32, tokens: u32, global: f32, activation: f32, bound: f32, role: u32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, scales: u64, output: u64, shape: Shape }
#[repr(C)] #[derive(Clone, Copy)]
struct Launch { single: Arguments, batch: u64, count: u32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Build { table: u64, ids: u64, input: u64, quantized: u64, quantizers: u64, projects: u64, output: u64, experts: u32, picks: u32, phases: u32, down: u32, shared: u32, raw: u32 }
#[repr(C)] #[derive(Clone, Copy)]
struct BuildRows { build: Build, tokens: u32, stride: u32 }
const PICKS: usize = 8;
const THREADS: u32 = 128;
const DESCRIPTOR: usize = 96;
fn bytes_of(a: &Arguments) -> Vec<u8> {
    let mut v = vec![];
    for x in a.input.iter().chain([&a.weight, &a.scales, &a.output]) { v.extend(x.to_le_bytes()); }
    let s = a.shape; for x in [s.op, s.rows, s.cols, s.block_rows, s.block_cols, s.tokens] { v.extend(x.to_le_bytes()); }
    for x in [s.global, s.activation, s.bound] { v.extend(x.to_le_bytes()); } v.extend(s.role.to_le_bytes()); v
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() >> 11) as usize % n }
}
/// Mirror of expert_ops::nv_cols_shared.
fn cols_shared(cols: usize) -> u32 { ((cols / 16 * 17 * 4).div_ceil(16) * 16 + 3 * 4 * (1024 + 128)) as u32 }
/// Input rows: random values with exact zeros of both signs, denormals and fixed edge words, as the route check uses.
fn input_rows(words: usize, r: &mut Rng) -> Vec<u8> {
    let edges = [0u32, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x80800000, 0x3f808000, 0x3f818000, 0xbf808000, 0x3f800000, 0xbf800000, 0x3e800000];
    let mut v = Vec::with_capacity(words * 4);
    for _ in 0..words { let k = r.below(16); let x = if k < 3 { edges[r.below(edges.len())] } else { (r.unit() * 4.0).to_bits() }; v.extend(x.to_le_bytes()); }
    v
}
/// One row of distinct expert IDs.
fn distinct(experts: usize, r: &mut Rng) -> Vec<usize> {
    let mut row: Vec<usize> = vec![];
    while row.len() < PICKS { let id = r.below(experts); if !row.contains(&id) { row.push(id); } }
    row
}
/// ID rows of a step. kind 0: every row its own picks (repeats across rows by chance). 1: every row the same picks
/// (chats in lockstep). 2: neighbours share half their picks. 3: the edges of the table, reversed in odd rows.
fn id_rows(kind: usize, tokens: usize, experts: usize, r: &mut Rng) -> Vec<Vec<usize>> {
    let first = distinct(experts, r);
    (0..tokens).map(|t| match kind {
        0 => distinct(experts, r),
        1 => first.clone(),
        2 => { let mut row = first[..PICKS / 2].to_vec(); while row.len() < PICKS { let id = r.below(experts); if !row.contains(&id) { row.push(id); } } if t % 2 == 1 { row.reverse(); } row }
        _ => { let mut row: Vec<usize> = (0..PICKS / 2).chain(experts - PICKS / 2..experts).collect(); if t % 2 == 1 { row.reverse(); } row }
    }).collect()
}
/// The ID plane: `stride` words a row, the IDs first, the rest poison (NaN) as a route row's other words are not IDs.
fn id_plane(rows: &[Vec<usize>], stride: usize) -> Vec<u8> {
    let mut v = vec![];
    for (t, row) in rows.iter().enumerate() {
        for (slot, &id) in row.iter().enumerate() { let x = if id == 0 && (t + slot) % 2 == 0 { -0.0f32 } else { id as f32 }; v.extend(x.to_le_bytes()); }
        for _ in PICKS..stride { v.extend(f32::NAN.to_le_bytes()); }
    }
    v
}
struct Table { address: u64, first: Arguments, experts: usize, phases: usize, rows: usize, cols: usize, down: bool, shared: bool }
/// A route table of `experts` column-tile matrices a phase, with its owners.
fn table(g: &mut Gpu, rows: usize, cols: usize, experts: usize, phases: usize, down: bool, calibrated: bool, r: &mut Rng) -> Result<Table> {
    let mut bytes = vec![]; let mut first = None;
    for phase in 0..phases { for expert in 0..experts {
        let codes: Vec<u8> = (0..rows * cols / 2).map(|_| r.next() as u8).collect();
        let scales: Vec<u8> = (0..rows * cols / 16).map(|i| if i % 97 == 0 { [0x00u8, 0x80, 0x38, 0x30, 0x28, 0x40, 0x01, 0x81][r.below(8)] } else { 0x30 + r.below(20) as u8 }).collect();
        let shape = Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 128, block_cols: 16, tokens: 1, global: if expert % 2 == 0 { 0.5 } else { 0.75 },
            activation: if calibrated { 0.125 * (1 + (expert + phase) % 7) as f32 } else { 0.25 }, bound: 0.0, role: u32::from(down) };
        let entry = Arguments { input: [4; 4], weight: g.upload(&codes)?, scales: g.upload(&scales)?, output: 4, shape };
        first.get_or_insert(entry); bytes.extend(bytes_of(&entry));
    } }
    Ok(Table { address: g.upload(&bytes)?, first: first.ok_or("empty route table")?, experts, phases, rows, cols, down, shared: !down && !calibrated })
}
/// Scratch and output of one arm for up to `tokens` rows.
struct Arm { quantized: u64, quantizers: u64, projects: u64, output: u64, out_bytes: usize, jobs: usize }
fn arm(g: &mut Gpu, t: &Table, tokens: usize) -> Result<Arm> {
    let jobs = tokens * PICKS * t.phases;
    Ok(Arm { quantized: g.alloc(jobs * (t.cols / 16 * 17) * 4)?, quantizers: g.alloc(jobs * DESCRIPTOR)?, projects: g.alloc(jobs * DESCRIPTOR)?, output: g.alloc(jobs * t.rows * 4)?, out_bytes: jobs * t.rows * 4, jobs })
}
fn build_of(t: &Table, a: &Arm, ids: u64, input: u64) -> Build {
    Build { table: t.address, ids, input, quantized: a.quantized, quantizers: a.quantizers, projects: a.projects, output: a.output,
        experts: t.experts as u32, picks: PICKS as u32, phases: t.phases as u32, down: u32::from(t.down), shared: u32::from(t.shared), raw: 0 }
}
fn project(g: &Gpu, f: H, t: &Table, projects: u64, jobs: usize) -> Result<()> {
    let mut l = Launch { single: t.first, batch: projects, count: jobs as u32 };
    g.launch(f, [(t.rows / 128 * jobs) as u32, 1, 1], THREADS, cols_shared(t.cols), &mut [ptr(&mut l)])
}
/// The route build alone (the descriptors and quantized rows of `rows_arm`).
fn build_rows(g: &Gpu, build_f: H, t: &Table, a: &Arm, ids: u64, stride: usize, input: u64, tokens: usize) -> Result<()> {
    let jobs = tokens * PICKS * t.phases; let quantized = if t.shared { tokens } else { jobs };
    let mut b = BuildRows { build: build_of(t, a, ids, input), tokens: tokens as u32, stride: stride as u32 };
    g.launch(build_f, [(jobs.div_ceil(THREADS as usize) + t.cols.div_ceil(THREADS as usize) * quantized) as u32, 1, 1], THREADS, 0, &mut [ptr(&mut b)])
}
/// The new arm: one build for all rows, one projection of all jobs.
fn rows_arm(g: &Gpu, build_f: H, cols_f: H, t: &Table, a: &Arm, ids: u64, stride: usize, input: u64, tokens: usize) -> Result<()> {
    let jobs = tokens * PICKS * t.phases; let quantized = if t.shared { tokens } else { jobs };
    let head = jobs.div_ceil(THREADS as usize);
    let mut b = BuildRows { build: build_of(t, a, ids, input), tokens: tokens as u32, stride: stride as u32 };
    g.launch(build_f, [(head + t.cols.div_ceil(THREADS as usize) * quantized) as u32, 1, 1], THREADS, 0, &mut [ptr(&mut b)])?;
    project(g, cols_f, t, a.projects, jobs)
}
/// The old arm: the one-token build and its projection, once a token, each token with its own scratch.
fn token_arm(g: &Gpu, build_f: H, cols_f: H, t: &Table, a: &[Arm], ids: u64, stride: usize, input: u64, tokens: usize) -> Result<()> {
    let jobs = PICKS * t.phases; let row = if t.down { PICKS * t.cols } else { t.cols };
    for token in 0..tokens {
        let mut b = build_of(t, &a[token], ids + (token * stride * 4) as u64, input + (token * row * 4) as u64);
        g.launch(build_f, [(1 + t.cols.div_ceil(THREADS as usize) * if t.shared { 1 } else { jobs }) as u32, 1, 1], THREADS, 0, &mut [ptr(&mut b)])?;
        project(g, cols_f, t, a[token].projects, jobs)?;
    }
    Ok(())
}
/// The residual sum's inputs: 0 BF16-valued pairs (the served residual stream), 1 full f32 mantissas, 2 sums on an exact
/// BF16 tie (both parities), 3 denormals and zeros of both signs, 4 large magnitudes whose BF16 round may overflow, 5 near
/// cancellation.
fn sum_pair(kind: usize, i: usize, r: &mut Rng) -> (f32, f32) {
    let bf = |v: f32| f32::from_bits(v.to_bits() & 0xffff0000);
    match kind {
        0 => (bf(r.unit() * 8.0), bf(r.unit() * 8.0)),
        1 => (r.unit() * 100.0, r.unit()),
        2 => (f32::from_bits((0x3f80_0000 + ((r.next() as u32 & 0x7f) << 16)) | 0x8000 | if i % 2 == 1 { 0x10000 } else { 0 }), 0.0),
        3 => { let e = [0u32, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x80800000, 0x00008000, 0x00018000];
            (f32::from_bits(e[i % e.len()]), f32::from_bits(e[(i / e.len() + r.below(e.len())) % e.len()])) }
        4 => (f32::from_bits(0x7f7f0000 | (r.next() as u32 & 0xffff)) * if i % 3 == 0 { -1.0 } else { 1.0 }, f32::from_bits(0x7e000000 | (r.next() as u32 & 0x7fffff))),
        _ => { let a = r.unit() * 1000.0; (a, -a + r.unit() * 1e-3) }
    }
}
/// RESIDUAL SUM (RB4 2026-11-18): `layer_residual` then `logits_bf16` in place, as `Device::bf16_sum` launches them, against
/// the host form `round_bf16(a + b)` a word (engine_model's cast, the host form's own function). Every word the host form
/// accepts must be equal; a word it refuses (a non-finite sum, or a BF16 round that is not finite) must stay non-finite.
/// Then `logits_argmax` and its finish on the sum, as `Device::bf16_sum` queues them: the valid word must be 1.0 exactly
/// when the host form refuses no word. Kinds 6 to 8 are kind 0 with one NaN or infinite operand: first, last, any word.
fn residual_sum(g: &mut Gpu, module: H) -> Result<()> {
    let add_f = g.function(module, "layer_residual")?; let round_f = g.function(module, "logits_bf16")?;
    let (argmax_f, finish_f) = (g.function(module, "logits_argmax")?, g.function(module, "logits_argmax_finish")?);
    let (mut cases, mut compared, mut refused, mut flagged) = (0usize, 0usize, 0usize, 0usize);
    for &words in &[1usize, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 65536] { for kind in 0..9 {
        let mut r = Rng(0x2545f4914f6cdd1d ^ (words as u64) << 8 ^ kind as u64);
        let (mut x, y): (Vec<f32>, Vec<f32>) = (0..words).map(|i| sum_pair(if kind < 6 { kind } else { 0 }, i, &mut r)).unzip();
        if kind >= 6 { let at = match kind { 6 => 0, 7 => words - 1, _ => r.below(words) }; x[at] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][(words + kind) % 3]; }
        let bytes = |v: &[f32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        let (state, residual) = (g.upload(&bytes(&x))?, g.upload(&bytes(&y))?);
        let (mut s, mut a, mut w) = (state, residual, words as u32); let grid = [(words as u32).div_ceil(THREADS), 1, 1];
        g.launch(add_f, grid, THREADS, 0, &mut [ptr(&mut s), ptr(&mut a), ptr(&mut w)])?;
        g.launch(round_f, grid, THREADS, 0, &mut [ptr(&mut s), ptr(&mut w)])?; g.sync()?;
        let got = g.read(state, words * 4)?; let before = refused;
        for (i, word) in got.chunks_exact(4).enumerate() {
            let d = f32::from_le_bytes(word.try_into().map_err(|_| "residual sum word")?);
            match engine_model::plan::layer::cast::round_bf16(x[i] + y[i]) {
                Ok(h) if h.to_bits() == d.to_bits() => compared += 4,
                Ok(h) => return Err(format!("RESIDUAL SUM BYTE FAIL words={words} kind={kind} word={i} x={:08x} y={:08x} host={:08x} device={:08x}", x[i].to_bits(), y[i].to_bits(), h.to_bits(), d.to_bits())),
                Err(_) if !d.is_finite() => refused += 1,
                Err(_) => return Err(format!("RESIDUAL SUM REFUSAL FAIL words={words} kind={kind} word={i}: the host form refuses, the device wrote finite {:08x}", d.to_bits())),
            }
        }
        let count = words.div_ceil(1024); let (partials, result) = (g.upload(&vec![0u8; count * 12])?, g.upload(&[0u8; 12])?);
        let (mut l, mut p, mut o, mut c) = (state, partials, result, count as u32);
        g.launch(argmax_f, [(count as u32 * 32).div_ceil(THREADS), 1, 1], THREADS, 0, &mut [ptr(&mut l), ptr(&mut p), ptr(&mut w)])?;
        g.launch(finish_f, [1, 1, 1], THREADS, 0, &mut [ptr(&mut p), ptr(&mut o), ptr(&mut c)])?; g.sync()?;
        let valid = f32::from_le_bytes(g.read(result, 12)?[8..12].try_into().map_err(|_| "residual sum flag")?);
        if (valid == 1.0) != (refused == before) { return Err(format!("RESIDUAL SUM FLAG FAIL words={words} kind={kind}: valid {valid}, host refused {} words", refused - before)); }
        flagged += usize::from(refused > before); cases += 1;
    } }
    println!("RESIDUAL SUM BYTES COMPLETE cases={cases} compared_bytes={compared} differing_bytes=0 host_refused_words={refused} (each non-finite on the device)");
    println!("RESIDUAL SUM FLAG COMPLETE cases={cases} flagged={flagged} clean={} (valid word 0.0 exactly where the host form refuses a word)", cases - flagged);
    Ok(())
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
pub(crate) fn run(image: &str, mode: &str) -> Result<()> {
    let ptx = std::fs::read(image).map_err(|e| e.to_string())?;
    println!("ROUTEROWS image={image} sha256={} mode={mode}", sha256::digest(&ptx));
    let mut g = Gpu::new()?; let module = g.module(&ptx)?;
    if mode == "sum" { residual_sum(&mut g, module)?; println!("ROUTEROWS DONE"); return Ok(()); }
    let rows_f = g.function(module, "device_route_rows")?; let token_f = g.function(module, "device_route_build")?; let cols_f = g.function(module, "nv_cols")?;
    if let Some(kind) = mode.strip_prefix("refuse-") { return refuse(&mut g, rows_f, cols_f, kind); }
    if mode == "bytes" || mode == "all" { bytes(&mut g, rows_f, token_f, cols_f)?; }
    if mode == "time" || mode == "all" { time(&mut g, rows_f, token_f, cols_f)?; }
    if let Some(base) = mode.strip_prefix("router=") {
        let base_ptx = std::fs::read(base).map_err(|e| e.to_string())?;
        println!("ROUTEROWS ROUTER base={base} sha256={}", sha256::digest(&base_ptx));
        let base_module = g.module(&base_ptx)?;
        let (new_f, base_f) = (g.function(module, "prompt_route")?, g.function(base_module, "prompt_route")?);
        router_ab(&mut g, new_f, base_f)?;
        router_planes(&mut g, new_f, base_f)?;
        let ranked_f = g.function(base_module, "causal_operation")?;
        ranked_ab(&mut g, new_f, ranked_f)?;
        println!("ROUTEROWS DONE");
        return Ok(());
    }
    if let Some(base) = mode.strip_prefix("build=") {
        bytes(&mut g, rows_f, token_f, cols_f)?;
        let base_ptx = std::fs::read(base).map_err(|e| e.to_string())?;
        println!("ROUTEROWS BUILD base={base} sha256={}", sha256::digest(&base_ptx));
        let base_module = g.module(&base_ptx)?; let base_f = g.function(base_module, "device_route_rows")?;
        build_ab(&mut g, rows_f, base_f)?;
    }
    println!("ROUTEROWS DONE");
    Ok(())
}
fn bytes(g: &mut Gpu, rows_f: H, token_f: H, cols_f: H) -> Result<()> {
    let mut total = 0usize; let mut cases = 0usize; let mut descriptors = 0usize;
    // The model's shapes (gate and up 1024 x 4096 a rank, down 2048 x 2048) and the edges of the storage rule.
    for &(rows, cols) in &[(128usize, 512usize), (128, 1024), (256, 2048), (384, 4096), (1024, 4096), (2048, 2048)] {
        for &(down, phases) in &[(false, 1usize), (false, 2), (true, 1)] { for calibrated in [false, true] {
            // Quarter chains of a second matrix need whole quarters of 64-column rounds.
            if down && (cols / 64) % 4 != 0 { continue; }
            let experts = if rows * cols > 1 << 20 { 16 } else { 40 };
            let mut r = Rng(0x9e3779b97f4a7c15 ^ ((rows * 131 + cols) as u64) << 8 ^ (phases as u64) << 4 ^ u64::from(down) << 1 ^ u64::from(calibrated));
            let t = table(g, rows, cols, experts, phases, down, calibrated, &mut r)?;
            let most = 16usize;
            let new = arm(g, &t, most)?; let old = (0..most).map(|_| arm(g, &t, 1)).collect::<Result<Vec<_>>>()?;
            let mut compared = 0usize;
            for &tokens in &[1usize, 2, 3, 4, 5, 8, 16] { for kind in 0..4 { for &stride in &[PICKS, 594] {
                let picks = id_rows(kind, tokens, experts, &mut r);
                let ids = g.upload(&id_plane(&picks, stride))?;
                let input = g.upload(&input_rows(tokens * if down { PICKS } else { 1 } * cols, &mut r))?;
                // Both outputs and both descriptor arrays poisoned with different bytes: an unwritten word cannot compare equal.
                g.upload_at(new.output, &vec![0xffu8; new.out_bytes])?; g.upload_at(new.projects, &vec![0xffu8; new.jobs * DESCRIPTOR])?;
                for a in &old[..tokens] { g.upload_at(a.output, &vec![0xeeu8; a.out_bytes])?; g.upload_at(a.projects, &vec![0xeeu8; a.jobs * DESCRIPTOR])?; }
                rows_arm(g, rows_f, cols_f, &t, &new, ids, stride, input, tokens)?;
                token_arm(g, token_f, cols_f, &t, &old, ids, stride, input, tokens)?; g.sync()?;
                let jobs = tokens * PICKS * phases;
                let got = g.read(new.output, jobs * rows * 4)?; let got_d = g.read(new.projects, jobs * DESCRIPTOR)?;
                // EXPERT ORDER: the device lists a phase's descriptors by (expert, pair). A job's descriptor stands at its pair's
                // rank (written out here, apart from the engine's rule); its input and output rows keep their pair order.
                let flat = picks.iter().flatten().copied().collect::<Vec<_>>();
                let rank = |pair: usize| (0..flat.len()).filter(|&other| flat[other] < flat[pair] || flat[other] == flat[pair] && other < pair).count();
                for token in 0..tokens {
                    let want = g.read(old[token].output, old[token].out_bytes)?; let want_d = g.read(old[token].projects, old[token].jobs * DESCRIPTOR)?;
                    for phase in 0..phases { for slot in 0..PICKS {
                        let (job, one) = ((phase * tokens + token) * PICKS + slot, phase * PICKS + slot);
                        let (x, y) = (&got[job * rows * 4..(job + 1) * rows * 4], &want[one * rows * 4..(one + 1) * rows * 4]);
                        if x != y { let word = x.chunks_exact(4).zip(y.chunks_exact(4)).position(|(p, q)| p != q); let differing = x.chunks_exact(4).zip(y.chunks_exact(4)).filter(|(p, q)| p != q).count();
                            return Err(format!("ROUTEROWS BYTE FAIL rows={rows} cols={cols} phases={phases} down={down} calibrated={calibrated} tokens={tokens} ids={kind} stride={stride} token={token} phase={phase} slot={slot} first_word={word:?} differing_words={differing} of {rows}")); }
                        // The matrix a job names: weight, scales and the ten shape words (the role is the last).
                        let listed = phase * tokens * PICKS + rank(token * PICKS + slot);
                        let (p, q) = (&got_d[listed * DESCRIPTOR..(listed + 1) * DESCRIPTOR], &want_d[one * DESCRIPTOR..(one + 1) * DESCRIPTOR]);
                        if p[32..48] != q[32..48] || p[56..] != q[56..] { return Err(format!("ROUTEROWS DESCRIPTOR FAIL rows={rows} cols={cols} phases={phases} down={down} calibrated={calibrated} tokens={tokens} ids={kind} stride={stride} token={token} phase={phase} slot={slot}")); }
                        // Its output row is its own: job-major in the launch's output.
                        let output = u64::from_le_bytes(p[48..56].try_into().map_err(|_| "descriptor bytes")?);
                        if output != new.output + (job * rows * 4) as u64 { return Err(format!("ROUTEROWS OUTPUT PLACE FAIL tokens={tokens} token={token} phase={phase} slot={slot}")); }
                        compared += rows * 4; descriptors += 1;
                    } }
                }
                cases += 1;
            } } }
            println!("ROUTEROWS BYTES PASS rows={rows} cols={cols} phases={phases} down={down} calibrated={calibrated} shared={} experts={experts} tokens=1,2,3,4,5,8,16 id_kinds=4 strides=8,594 compared_bytes={compared}", t.shared);
            total += compared; g.clear()?;
        } }
    }
    println!("ROUTEROWS BYTES COMPLETE cases={cases} descriptors={descriptors} compared_bytes={total} differing_bytes=0");
    Ok(())
}
fn time(g: &mut Gpu, rows_f: H, token_f: H, cols_f: H) -> Result<()> {
    // The model's routed launches of one layer: gate and up in one launch (1024 x 4096 a rank, 2 phases) and down
    // (2048 x 2048, second matrices). 64 experts a phase, eight rotating pick sets, so a call reads its weights from memory.
    for &(rows, cols, phases, down, label) in &[(1024usize, 4096usize, 2usize, false, "gate_up"), (2048, 2048, 1, true, "down")] {
        let mut r = Rng(0x5151 + rows as u64); let experts = 64;
        let t = table(g, rows, cols, experts, phases, down, false, &mut r)?;
        for &tokens in &[1usize, 2, 4, 8] {
            let new = arm(g, &t, tokens)?; let old = (0..tokens).map(|_| arm(g, &t, 1)).collect::<Result<Vec<_>>>()?;
            let mut sets = vec![];
            for _ in 0..8 { let picks = id_rows(0, tokens, experts, &mut r); sets.push((g.upload(&id_plane(&picks, 594))?, g.upload(&input_rows(tokens * if down { PICKS } else { 1 } * cols, &mut r))?)); }
            let mut times = [vec![], vec![]];
            for repeat in 0..9 { for k in 0..2 {
                let which = if repeat % 2 == 0 { k } else { 1 - k }; let calls = 16;
                let call = |g: &Gpu, set: &(u64, u64)| if which == 0 { rows_arm(g, rows_f, cols_f, &t, &new, set.0, 594, set.1, tokens) } else { token_arm(g, token_f, cols_f, &t, &old, set.0, 594, set.1, tokens) };
                call(g, &sets[0])?; g.sync()?;
                let t0 = Instant::now(); for i in 0..calls { call(g, &sets[i % 8])?; } g.sync()?;
                times[which].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
            } }
            let (a, b) = (median(times[0].clone()), median(times[1].clone()));
            println!("ROUTEROWS TIME {label} rows={rows} cols={cols} phases={phases} tokens={tokens} jobs={} one_launch_us={a:.1} once_a_token_us={b:.1} ratio={:.3} us_a_job={:.2} raw_one={:?} raw_token={:?}",
                tokens * PICKS * phases, a / b, a / (tokens * PICKS * phases) as f64,
                times[0].iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>(), times[1].iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>());
        }
        g.clear()?;
    }
    Ok(())
}
/// ROUTE BOOKKEEPING: the route build of this image (`new_f`) against that of a base image (`base_f`), alone. Bytes: one
/// scratch, poisoned with the same bytes before each build; every descriptor, quantizer and quantized word equal. The
/// model's two routed shapes (calibrated or not) and two storage edges, 1 to 16 rows, 4 ID kinds, 2 strides. Time: the
/// two builds interleaved, 32 calls, median of 9, the model's shapes, 1 to 16 rows, ID kind 0.
fn build_ab(g: &mut Gpu, new_f: H, base_f: H) -> Result<()> {
    let (mut cases, mut compared) = (0usize, 0usize);
    for &(rows, cols, phases, down, calibrated) in &[(1024usize, 4096usize, 2usize, false, false), (1024, 4096, 2, false, true), (2048, 2048, 1, true, false), (2048, 2048, 1, true, true), (128, 512, 1, false, false), (256, 2048, 1, true, true)] {
        let experts = 64; let mut r = Rng(0xb00c ^ ((rows * 7 + cols) as u64) << 6 ^ u64::from(down) << 3 ^ u64::from(calibrated) << 5);
        let t = table(g, rows, cols, experts, phases, down, calibrated, &mut r)?;
        let a = arm(g, &t, 16)?; let words = cols / 16 * 17;
        for &tokens in &[1usize, 2, 3, 4, 5, 8, 16] { for kind in 0..4 { for &stride in &[PICKS, 594] {
            let picks = id_rows(kind, tokens, experts, &mut r);
            let ids = g.upload(&id_plane(&picks, stride))?;
            let input = g.upload(&input_rows(tokens * if down { PICKS } else { 1 } * cols, &mut r))?;
            let jobs = tokens * PICKS * phases; let quantized = if t.shared { tokens } else { jobs };
            let mut got = vec![];
            for f in [base_f, new_f] {
                g.upload_at(a.projects, &vec![0xeeu8; a.jobs * DESCRIPTOR])?; g.upload_at(a.quantizers, &vec![0xeeu8; a.jobs * DESCRIPTOR])?;
                g.upload_at(a.quantized, &vec![0xeeu8; a.jobs * words * 4])?;
                build_rows(g, f, &t, &a, ids, stride, input, tokens)?; g.sync()?;
                got.push([g.read(a.projects, jobs * DESCRIPTOR)?, g.read(a.quantizers, quantized * DESCRIPTOR)?, g.read(a.quantized, quantized * words * 4)?]);
            }
            for (part, name) in ["descriptors", "quantizers", "quantized rows"].iter().enumerate() {
                if got[0][part] != got[1][part] { return Err(format!("ROUTEROWS BUILD BYTE FAIL {name} rows={rows} cols={cols} phases={phases} down={down} calibrated={calibrated} tokens={tokens} ids={kind} stride={stride}")); }
                compared += got[0][part].len();
            }
            cases += 1;
        } } }
        println!("ROUTEROWS BUILD BYTES PASS rows={rows} cols={cols} phases={phases} down={down} calibrated={calibrated} shared={} tokens=1,2,3,4,5,8,16 id_kinds=4 strides=8,594", t.shared);
        g.clear()?;
    }
    println!("ROUTEROWS BUILD BYTES COMPLETE cases={cases} compared_bytes={compared} differing_bytes=0");
    for &(rows, cols, phases, down, label) in &[(1024usize, 4096usize, 2usize, false, "gate_up"), (2048, 2048, 1, true, "down")] {
        let mut r = Rng(0x5153 + rows as u64); let experts = 64;
        let t = table(g, rows, cols, experts, phases, down, false, &mut r)?;
        for &tokens in &[1usize, 2, 3, 4, 8, 16] {
            let a = arm(g, &t, tokens)?; let mut sets = vec![];
            for _ in 0..8 { let picks = id_rows(0, tokens, experts, &mut r); sets.push((g.upload(&id_plane(&picks, 594))?, g.upload(&input_rows(tokens * if down { PICKS } else { 1 } * cols, &mut r))?)); }
            let mut times = [vec![], vec![]];
            for repeat in 0..9 { for k in 0..2 {
                let which = if repeat % 2 == 0 { k } else { 1 - k }; let f = if which == 0 { base_f } else { new_f }; let calls = 32;
                build_rows(g, f, &t, &a, sets[0].0, 594, sets[0].1, tokens)?; g.sync()?;
                let t0 = Instant::now(); for i in 0..calls { build_rows(g, f, &t, &a, sets[i % 8].0, 594, sets[i % 8].1, tokens)?; } g.sync()?;
                times[which].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
            } }
            let (x, y) = (median(times[0].clone()), median(times[1].clone()));
            println!("ROUTEROWS BUILD AB {label} tokens={tokens} pairs={} quantized_rows={} base_us={x:.2} new_us={y:.2} ratio={:.3} saved_us={:.2} raw_base={:?} raw_new={:?}",
                tokens * PICKS, if t.shared { tokens } else { tokens * PICKS * phases }, y / x, x - y,
                times[0].iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>(), times[1].iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>());
        }
        g.clear()?;
    }
    Ok(())
}
/// Mirrors of gpu/src/prompt_ops.rs `Shape` and `Arguments` and gpu/src/causal_ops.rs `Shape` (repr(C), 192 bytes). Op Wrapped
/// is 0, causal Route 14. The router of a rows pass: `prompt_route`, one warp a token, the one shape `route_admitted` takes.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct RowShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct RowArguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: RowShape }
/// The model's router (prompt_admission::route_admitted): 288 experts, 8 picks, one group, normalized.
const EXPERTS: usize = 288;
/// Output words a token: k ids, k weights, then 2 * rows + 2 words the kernel clears.
const ROUTE_WORDS: usize = 2 * PICKS + 2 * EXPERTS + 2;
/// Logits of `tokens` rows. 0 random; 1 a few levels (ties of the score); 2 saturated, with the clamp edges of exp;
/// 3 NaN and both infinities among random values; 4 all equal; 5 signed zeros and denormals.
fn logits(kind: usize, tokens: usize, r: &mut Rng) -> Vec<f32> {
    (0..tokens * EXPERTS).map(|i| match kind {
        0 => r.unit() * 4.0,
        1 => (r.below(5) as f32 - 2.0) * 0.5,
        2 => [87.0f32, -87.0, 88.0, -88.0, 1e30, -1e30, 200.0, -200.0, 0.0][r.below(9)] * if r.below(4) == 0 { 1.0 } else { r.unit() },
        3 => match i % 37 { 3 => f32::NAN, 11 => f32::INFINITY, 19 => f32::NEG_INFINITY, _ => r.unit() * 8.0 },
        4 => 0.25,
        _ => match r.below(4) { 0 => 0.0, 1 => -0.0, 2 => f32::from_bits(1 + r.below(0x7fffff) as u32), _ => -f32::from_bits(1 + r.below(0x7fffff) as u32) },
    }).collect()
}
/// Biases. 0 small random; 1 zeros; 2 three levels (ties of the choice); 3 some minus infinity; 4 some NaN.
fn biases(kind: usize, r: &mut Rng) -> Vec<f32> {
    (0..EXPERTS).map(|i| match kind {
        0 => r.unit() * 0.1,
        1 => 0.0,
        2 => r.below(3) as f32 * 0.25,
        3 => if i % 5 == 0 { f32::NEG_INFINITY } else { r.unit() * 0.1 },
        _ => if i % 41 == 7 { f32::NAN } else { r.unit() * 0.1 },
    }).collect()
}
fn floats(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn route_args(logits: u64, bias: u64, output: u64, tokens: usize, bound: f32) -> RowArguments {
    let inner = Inner { op: 14, heads: 1, dim: 1, taps: 1, rows: EXPERTS as u32, cols: PICKS as u32, epsilon: 0.0, bound };
    RowArguments { input: [logits, 0, 0, 0, 0], lengths: [(tokens * EXPERTS) as u64, 0, 0, 0, 0], weight: bias, weights: 0, output, units: tokens as u64, words: ROUTE_WORDS as u64,
        shape: RowShape { op: 0, tokens: tokens as u32, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, first: 0, slots: 0, inner } }
}
/// prompt_admission::fast_blocks for a Route: a block of 128 threads, four tokens a block.
fn route(g: &Gpu, f: H, a: &mut RowArguments) -> Result<()> { g.launch(f, [(a.shape.tokens as usize).div_ceil(4) as u32, 1, 1], THREADS, 0, &mut [ptr(a)]) }
/// THE ROWS ROUTER A/B (RB4 2026-11-17; built for ROUTER LOADS FIRST, closed in rb4-k2): `prompt_route` of this image (`new_f`)
/// against that of a base image (`base_f`).
/// Bytes: every output word of every token, the two outputs poisoned with different bytes and 64-byte guards on both
/// sides that must stay; 6 logit kinds x 5 bias kinds x 10 row counts x 2 bounds. Time: interleaved, 64 calls, median of 9.
fn router_ab(g: &mut Gpu, new_f: H, base_f: H) -> Result<()> {
    let (mut cases, mut compared) = (0usize, 0usize); let mut r = Rng(0x40e7_e5);
    let most = 64usize; let bytes = most * ROUTE_WORDS * 4;
    let outs = [g.alloc(bytes + 128)?, g.alloc(bytes + 128)?];
    for kind in 0..6 { for bias_kind in 0..5 { for &tokens in &[1usize, 2, 3, 4, 5, 7, 8, 9, 16, 33] { for &bound in &[2.5f32, 1.0] {
        let x = g.upload(&floats(&logits(kind, tokens, &mut r)))?; let w = g.upload(&floats(&biases(bias_kind, &mut r)))?;
        let n = tokens * ROUTE_WORDS * 4; let mut got = vec![];
        for (arm, f) in [base_f, new_f].into_iter().enumerate() {
            let mut fill = vec![0x5au8; 64]; fill.extend(vec![if arm == 0 { 0xeeu8 } else { 0xffu8 }; n]); fill.extend([0x5au8; 64]);
            g.upload_at(outs[arm], &fill)?;
            let mut a = route_args(x, w, outs[arm] + 64, tokens, bound); route(g, f, &mut a)?; g.sync()?;
            let read = g.read(outs[arm], n + 128)?;
            if read[..64] != [0x5au8; 64] || read[64 + n..] != [0x5au8; 64] { return Err(format!("ROUTEROWS ROUTER GUARD FAIL arm={arm} logits={kind} biases={bias_kind} tokens={tokens} bound={bound}")); }
            got.push(read[64..64 + n].to_vec());
        }
        if got[0] != got[1] {
            let word = got[0].chunks_exact(4).zip(got[1].chunks_exact(4)).position(|(p, q)| p != q);
            return Err(format!("ROUTEROWS ROUTER BYTE FAIL logits={kind} biases={bias_kind} tokens={tokens} bound={bound} first_word={word:?}"));
        }
        compared += n; cases += 1;
    } } } }
    println!("ROUTEROWS ROUTER BYTES COMPLETE experts={EXPERTS} picks={PICKS} cases={cases} compared_bytes={compared} differing_bytes=0");
    for &tokens in &[1usize, 2, 4, 8, 16] {
        let mut sets = vec![];
        for _ in 0..8 { sets.push((g.upload(&floats(&logits(0, tokens, &mut r)))?, g.upload(&floats(&biases(0, &mut r)))?)); }
        let mut times = [vec![], vec![]];
        for repeat in 0..9 { for k in 0..2 {
            let which = if repeat % 2 == 0 { k } else { 1 - k }; let f = if which == 0 { base_f } else { new_f }; let calls = 64;
            let mut a = route_args(sets[0].0, sets[0].1, outs[which] + 64, tokens, 2.5); route(g, f, &mut a)?; g.sync()?;
            let t0 = Instant::now();
            for i in 0..calls { let mut a = route_args(sets[i % 8].0, sets[i % 8].1, outs[which] + 64, tokens, 2.5); route(g, f, &mut a)?; }
            g.sync()?; times[which].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
        } }
        let (x, y) = (median(times[0].clone()), median(times[1].clone()));
        println!("ROUTEROWS ROUTER AB tokens={tokens} base_us={x:.2} new_us={y:.2} ratio={:.3} saved_us={:.2} raw_base={:?} raw_new={:?}", y / x, x - y,
            times[0].iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>(), times[1].iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>());
    }
    // LATENCY: one block of 4 warps routes 256 tokens, 64 a warp one after another (the kernel strides its tokens by the
    // grid), so a launch lasts far longer than the launch rate that floors the calls above. Bytes of that launch first.
    // us a token of one warp = launch time / 64 (8 launches back to back, then one wait).
    let long = 256usize; let lb = long * ROUTE_WORDS * 4; let louts = [g.alloc(lb)?, g.alloc(lb)?];
    let (lx, lw) = (g.upload(&floats(&logits(0, long, &mut r)))?, g.upload(&floats(&biases(0, &mut r)))?);
    let one = |g: &Gpu, f: H, out: u64| -> Result<()> { let mut a = route_args(lx, lw, out, long, 2.5); g.launch(f, [1, 1, 1], THREADS, 0, &mut [ptr(&mut a)]) };
    let mut got = vec![];
    for (arm, f) in [base_f, new_f].into_iter().enumerate() {
        g.upload_at(louts[arm], &vec![if arm == 0 { 0xeeu8 } else { 0xffu8 }; lb])?; one(g, f, louts[arm])?; g.sync()?; got.push(g.read(louts[arm], lb)?);
    }
    if got[0] != got[1] { return Err("ROUTEROWS ROUTER LATENCY BYTE FAIL tokens=256 grid=1".into()); }
    let mut times = [vec![], vec![]];
    for repeat in 0..9 { for k in 0..2 {
        let which = if repeat % 2 == 0 { k } else { 1 - k }; let f = if which == 0 { base_f } else { new_f };
        one(g, f, louts[which])?; g.sync()?;
        let t0 = Instant::now(); for _ in 0..8 { one(g, f, louts[which])?; } g.sync()?;
        times[which].push(t0.elapsed().as_secs_f64() * 1e6 / (8.0 * 64.0));
    } }
    let (x, y) = (median(times[0].clone()), median(times[1].clone()));
    println!("ROUTEROWS ROUTER LATENCY bytes_equal=yes tokens_a_warp=64 base_us_a_token={x:.3} new_us_a_token={y:.3} ratio={:.3} saved_us_a_token={:.3} raw_base={:?} raw_new={:?}", y / x, x - y,
        times[0].iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>(), times[1].iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    g.clear()?;
    Ok(())
}
/// ROUTE PLANES (RB4 2026-11-18, P8): `prompt_route` of this image with the outer mark `first = 1` (prompt_ops::ROUTE_PLANES)
/// against the plain rows of the base image: the k IDs of every token at token*k, the k weights at (tokens+token)*k, every
/// other word zero, bit for bit; poisoned outputs, 64-byte guards on both sides; the 600 cases of `router_ab`, plus 64, 128
/// and 256 tokens (GROUP_MAX) for every logit kind, two bias kinds and one bound (636). Then the marked and plain launches
/// timed, interleaved, at 2, 4, 8 and 16 tokens.
fn router_planes(g: &mut Gpu, new_f: H, base_f: H) -> Result<()> {
    let (mut cases, mut compared) = (0usize, 0usize); let mut r = Rng(0x9a_7e5);
    let most = 256usize; let bytes = most * ROUTE_WORDS * 4;
    let outs = [g.alloc(bytes + 128)?, g.alloc(bytes + 128)?];
    for kind in 0..6 { for bias_kind in 0..5 { for &tokens in &[1usize, 2, 3, 4, 5, 7, 8, 9, 16, 33, 64, 128, 256] { for &bound in &[2.5f32, 1.0] {
        if tokens > 33 && (bias_kind > 1 || bound != 2.5) { continue; }
        let x = g.upload(&floats(&logits(kind, tokens, &mut r)))?; let w = g.upload(&floats(&biases(bias_kind, &mut r)))?;
        let n = tokens * ROUTE_WORDS * 4; let mut got = vec![];
        for (arm, f) in [base_f, new_f].into_iter().enumerate() {
            let mut fill = vec![0x5au8; 64]; fill.extend(vec![if arm == 0 { 0xeeu8 } else { 0xffu8 }; n]); fill.extend([0x5au8; 64]);
            g.upload_at(outs[arm], &fill)?;
            let mut a = route_args(x, w, outs[arm] + 64, tokens, bound); a.shape.first = arm as u32; route(g, f, &mut a)?; g.sync()?;
            let read = g.read(outs[arm], n + 128)?;
            if read[..64] != [0x5au8; 64] || read[64 + n..] != [0x5au8; 64] { return Err(format!("ROUTEROWS PLANES GUARD FAIL arm={arm} logits={kind} biases={bias_kind} tokens={tokens} bound={bound}")); }
            got.push(read[64..64 + n].to_vec());
        }
        let plain: Vec<[u8; 4]> = got[0].chunks_exact(4).map(|c| c.try_into().unwrap()).collect();
        let mut want = vec![[0u8; 4]; tokens * ROUTE_WORDS];
        for t in 0..tokens { for slot in 0..PICKS { want[t * PICKS + slot] = plain[t * ROUTE_WORDS + slot]; want[(tokens + t) * PICKS + slot] = plain[t * ROUTE_WORDS + PICKS + slot]; } }
        let want: Vec<u8> = want.concat();
        if want != got[1] {
            let word = want.chunks_exact(4).zip(got[1].chunks_exact(4)).position(|(p, q)| p != q);
            return Err(format!("ROUTEROWS PLANES BYTE FAIL logits={kind} biases={bias_kind} tokens={tokens} bound={bound} first_word={word:?}"));
        }
        compared += n; cases += 1;
    } } } }
    println!("ROUTEROWS PLANES BYTES COMPLETE experts={EXPERTS} picks={PICKS} cases={cases} compared_bytes={compared} differing_bytes=0 (marked new image against the plain rows of the base image, rearranged)");
    for &tokens in &[2usize, 4, 8, 16] {
        let (x, w) = (g.upload(&floats(&logits(0, tokens, &mut r)))?, g.upload(&floats(&biases(0, &mut r)))?);
        let mut times = [vec![], vec![]];
        for repeat in 0..9 { for k in 0..2 {
            let which = if repeat % 2 == 0 { k } else { 1 - k }; let calls = 64;
            let mut a = route_args(x, w, outs[which] + 64, tokens, 2.5); a.shape.first = which as u32; route(g, new_f, &mut a)?; g.sync()?;
            let t0 = Instant::now(); for _ in 0..calls { route(g, new_f, &mut a)?; } g.sync()?; times[which].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
        } }
        let (p, q) = (median(times[0].clone()), median(times[1].clone()));
        println!("ROUTEROWS PLANES TIME tokens={tokens} plain_us={p:.2} planes_us={q:.2} (one launch each; the planes save the joins' launch, not router time)");
    }
    g.clear()?;
    Ok(())
}
/// Mirror of gpu/src/causal_ops.rs `Arguments` (repr(C), 80 bytes): the one-token causal operation.
#[repr(C)] #[derive(Clone, Copy)]
struct CausalArguments { input: [u64; 4], weight: u64, output: u64, shape: Inner }
/// THE ONE-ROW ROUTER (RB4 2026-11-17, a question for the lead, no engine change): the plain step routes a token with
/// `causal_operation` (Route, the ranked form: one block of latent_ops::ROUTE_RANKED_THREADS = 512 threads, the host zeroes
/// its output first). Does `prompt_route` of this image give the same words for one token? Every word of the 60 input
/// kinds of `router_ab` (all 6 logit kinds, 5 bias kinds, 2 bounds), then both timed a call (64 calls, median of 9).
fn ranked_ab(g: &mut Gpu, route_f: H, ranked_f: H) -> Result<()> {
    let mut r = Rng(0x4a4e_4b); let n = ROUTE_WORDS * 4; let outs = [g.alloc(n)?, g.alloc(n)?]; let mut cases = 0usize;
    for kind in 0..6 { for bias_kind in 0..5 { for &bound in &[2.5f32, 1.0] {
        let x = g.upload(&floats(&logits(kind, 1, &mut r)))?; let w = g.upload(&floats(&biases(bias_kind, &mut r)))?;
        g.upload_at(outs[0], &vec![0u8; n])?; g.upload_at(outs[1], &vec![0xffu8; n])?;
        let mut c = CausalArguments { input: [x, 4, 4, 4], weight: w, output: outs[0], shape: Inner { op: 14, heads: 1, dim: 1, taps: 1, rows: EXPERTS as u32, cols: PICKS as u32, epsilon: 0.0, bound } };
        g.launch(ranked_f, [1, 1, 1], 512, 0, &mut [ptr(&mut c)])?;
        let mut a = route_args(x, w, outs[1], 1, bound); route(g, route_f, &mut a)?; g.sync()?;
        let (p, q) = (g.read(outs[0], n)?, g.read(outs[1], n)?);
        if p != q { let word = p.chunks_exact(4).zip(q.chunks_exact(4)).position(|(u, v)| u != v);
            println!("ROUTEROWS RANKED DIFFER logits={kind} biases={bias_kind} bound={bound} first_word={word:?}"); return Ok(()); }
        cases += 1;
    } } }
    println!("ROUTEROWS RANKED BYTES EQUAL cases={cases} words_a_case={ROUTE_WORDS}");
    let x = g.upload(&floats(&logits(0, 1, &mut r)))?; let w = g.upload(&floats(&biases(0, &mut r)))?;
    let mut times = [vec![], vec![]];
    for repeat in 0..9 { for k in 0..2 {
        let which = if repeat % 2 == 0 { k } else { 1 - k }; let calls = 64;
        let call = |g: &Gpu| -> Result<()> { if which == 0 { let mut c = CausalArguments { input: [x, 4, 4, 4], weight: w, output: outs[0], shape: Inner { op: 14, heads: 1, dim: 1, taps: 1, rows: EXPERTS as u32, cols: PICKS as u32, epsilon: 0.0, bound: 2.5 } };
            g.launch(ranked_f, [1, 1, 1], 512, 0, &mut [ptr(&mut c)]) } else { let mut a = route_args(x, w, outs[1], 1, 2.5); route(g, route_f, &mut a) } };
        call(g)?; g.sync()?; let t0 = Instant::now(); for _ in 0..calls { call(g)?; } g.sync()?;
        times[which].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
    } }
    let (a, b) = (median(times[0].clone()), median(times[1].clone()));
    println!("ROUTEROWS RANKED TIME ranked_us={a:.2} prompt_route_us={b:.2} (the latter floors at the launch rate) raw_ranked={:?} raw_route={:?}",
        times[0].iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>(), times[1].iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>());
    Ok(())
}
/// One bad ID row in a launch of 4 rows must stop the launch: the device traps, and the wait reports it. One kind a
/// process (a trap ends the context).
fn refuse(g: &mut Gpu, rows_f: H, cols_f: H, kind: &str) -> Result<()> {
    let mut r = Rng(0x7777); let (rows, cols, experts, tokens) = (128usize, 512usize, 16usize, 4usize);
    let t = table(g, rows, cols, experts, 1, false, true, &mut r)?; let a = arm(g, &t, tokens)?;
    let picks = id_rows(0, tokens, experts, &mut r); let mut plane = id_plane(&picks, PICKS);
    // The bad word sits in row 2, slot 5 (not the first row, not the first slot).
    let bad: f32 = match kind { "repeat" => picks[2][1] as f32, "nan" => f32::NAN, "infinity" => f32::INFINITY, "negative" => -1.0, "fraction" => 1.5, "outside" => experts as f32,
        _ => return Err("refuse kinds: repeat nan infinity negative fraction outside".into()) };
    let at = (2 * PICKS + 5) * 4; plane[at..at + 4].copy_from_slice(&bad.to_le_bytes());
    let ids = g.upload(&plane)?; let input = g.upload(&input_rows(tokens * cols, &mut r))?;
    let launched = rows_arm(g, rows_f, cols_f, &t, &a, ids, PICKS, input, tokens).and_then(|_| g.sync());
    match launched { Err(error) => { println!("ROUTEROWS REFUSED kind={kind}: {error}"); Ok(()) } Ok(()) => Err(format!("ROUTEROWS INVALID ROW WAS NOT REFUSED kind={kind}")) }
}
