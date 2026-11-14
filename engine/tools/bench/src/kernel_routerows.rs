//! ROUTE ROWS device gate: `device_route_rows` (the route build for `tokens` rows of one step, one launch, pair
//! order) against `device_route_build` of the same image run once a token. Both arms then project with `nv_cols`.
//! Byte check of every output word and of every job descriptor's matrix words, then timing at the model's shapes.
//! One process, one device. Run: kernel-bench --route-rows <image> [bytes|time|all|refuse-<kind>]. Safe code only.
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
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
pub(crate) fn run(image: &str, mode: &str) -> Result<()> {
    let ptx = std::fs::read(image).map_err(|e| e.to_string())?;
    println!("ROUTEROWS image={image} sha256={} mode={mode}", sha256::digest(&ptx));
    let mut g = Gpu::new()?; let module = g.module(&ptx)?;
    let rows_f = g.function(module, "device_route_rows")?; let token_f = g.function(module, "device_route_build")?; let cols_f = g.function(module, "nv_cols")?;
    if let Some(kind) = mode.strip_prefix("refuse-") { return refuse(&mut g, rows_f, cols_f, kind); }
    if mode == "bytes" || mode == "all" { bytes(&mut g, rows_f, token_f, cols_f)?; }
    if mode == "time" || mode == "all" { time(&mut g, rows_f, token_f, cols_f)?; }
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
