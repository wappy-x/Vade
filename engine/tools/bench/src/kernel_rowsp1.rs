//! ROW STAGE (from lane multi-row, premise P1): the N-row entry `flash_resident_projection_bf16_rows` (plain rows spread
//! over threads) against the one-row entry `flash_resident_projection_bf16` run once a row. Every output byte, then
//! timing on the real KDA batch and on its MXFP8 and plain parts alone, with weights that stream from memory.
//! Run: kernel-bench --rows-p1 <base ptx> <candidate ptx> [bytes|time|both]. Safe code only.
use super::*;
#[path = "kernel_mxinterleave.rs"] pub(crate) mod interleave;

fn bf16_bits(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect() }
struct P1Fixture { batch: Batch, inputs: Vec<u64> }
/// Which launch an arm makes: the one-row entry on source row `0`, or an N-row entry (`spread` = plain rows over threads).
#[derive(Clone, Copy, PartialEq)]
enum Shape { One(usize), Rows { spread: bool } }
/// Weights and N-row inputs of `jobs`; plain weights are BF16 patterns (the entry's contract). No outputs yet.
fn p1_build(g: &mut Gpu, jobs: &[Job], tokens: usize, round: bool, seed: u64, kinds: (u32, u32)) -> Result<P1Fixture> {
    let mut batch = Batch::default(); let mut inputs = vec![];
    for (i, j) in jobs.iter().enumerate() {
        let s = seed * 31 + i as u64;
        let input = g.upload(&le(&pattern(kinds.1, j.cols * tokens, s ^ 5)))?;
        let (weights, scales) = if j.plain { (g.upload(&bf16_bits(&pattern(kinds.0, j.rows * j.cols, s)))?, 0) }
            else { (g.upload(&mx_bytes(j.rows * j.cols, s, false))?, g.upload(&mx_bytes(j.rows * j.cols / 32, s ^ 9, true))?) };
        batch.layer.projections[i] = MxProjection { weights, scales, input, output: 0, rows: j.rows as u32, cols: j.cols as u32, tokens: 1, input_bf16: 0, round: 0 };
        if j.plain { batch.plain |= 1 << i; }
        inputs.push(input);
    }
    batch.layer.count = jobs.len() as u32; batch.round = round as u32;
    Ok(P1Fixture { batch, inputs })
}
/// The batch of one launch shape with its own poisoned outputs and its own grid.
fn p1_arm(g: &mut Gpu, jobs: &[Job], fx: &P1Fixture, shape: Shape, tokens: usize) -> Result<Built> {
    let mut b = fx.batch; let mut outputs = vec![];
    for (i, j) in jobs.iter().enumerate() {
        let words = if matches!(shape, Shape::One(_)) { j.rows } else { j.rows * tokens };
        let buf = Buffer::new(g, &vec![f32::NAN; words])?; buf.poison(g)?;
        let p = &mut b.layer.projections[i];
        p.output = buf.address;
        let mut blocks = if j.plain { j.rows.div_ceil(128) } else { j.rows.div_ceil(4) } as u32;
        match shape {
            Shape::One(t) => p.input = fx.inputs[i] + (t * j.cols * 4) as u64,
            Shape::Rows { spread } => { p.tokens = tokens as u32; if spread && j.plain { blocks *= tokens as u32; } }
        }
        b.layer.ends[i + 1] = b.layer.ends[i] + blocks;
        outputs.push((buf.base, words * 4 + 32));
    }
    Ok(Built { batch: b, outputs })
}
pub(crate) fn rows_p1(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("ROWSP1 baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut gpu = Gpu::new()?; let (ma, mb) = (gpu.module(&a)?, gpu.module(&b)?);
    let g = &mut gpu;
    let (one_base, one_cand) = (g.function(ma, "flash_resident_projection_bf16")?, g.function(mb, "flash_resident_projection_bf16")?);
    let spread = g.function(mb, "flash_resident_projection_bf16_rows")?;
    let started = Instant::now();
    let (p, m) = (true, false);
    let job = |plain: bool, rows: usize, cols: usize| Job { plain, rows, cols };
    let real = vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(p, 384, 4096)];
    if mode != "time" {
        let mut cases: Vec<Vec<Job>> = vec![
            real.clone(),
            vec![job(p, 384, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096)],
            vec![job(m, 4096, 4096), job(p, 384, 4096), job(m, 1024, 4096), job(m, 1024, 4096)],
            vec![job(p, 128, 4096), job(p, 128, 4096), job(p, 128, 4096)],
            vec![job(p, 1, 32)], vec![job(m, 1, 32)], vec![job(p, 129, 64), job(m, 5, 32), job(p, 255, 128)],
            vec![job(p, 1, 8192)], vec![job(p, 3, 16384), job(m, 5, 64)], vec![job(p, 2, 8192), job(m, 4, 8192), job(p, 33, 8224)],
            vec![job(m, 17, 96), job(p, 300, 32), job(m, 33, 4096), job(p, 1, 4096), job(p, 128, 512), job(m, 1024, 512),
                job(p, 256, 256), job(m, 3, 64), job(p, 2, 8192)],
            // ROW STAGE: rank-local shapes a verify pass sends through the entry on the real model. MXFP8 at 8192 and 12288
            // columns (o_proj, dense down); q, k, v beside the BF16 plain shapes of plain_admission::BF16_SHAPES.
            vec![job(m, 2048, 8192)], vec![job(m, 2048, 12288)],
            vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096), job(p, 64, 4096), job(p, 32, 4096), job(p, 64, 4096)],
            vec![job(p, 4096, 128), job(p, 4096, 128)],
            vec![job(p, 2048, 1536), job(p, 64, 4096), job(p, 16, 4096), job(p, 128, 4096)],
        ];
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..120 {
            let n = 1 + (rng.next() % 9) as usize;
            cases.push((0..n).map(|_| job(rng.next().is_multiple_of(2), 1 + (rng.next() % 600) as usize, 32 * (1 + (rng.next() % 130) as usize))).collect());
        }
        let (mut comparisons, mut bytes, mut one_row_checks) = (0u64, 0u64, 0u64);
        for (index, jobs) in cases.iter().enumerate() {
            for tokens in [2usize, 3, 4] {
                for round in [false, true] {
                    for &kinds in &[(0u32, 0u32), (5, 5), (2, 2), (4, 0), (6, 7)] {
                        if index >= 4 && (kinds != (0, 0) && kinds != (5, 5) && kinds != (2, 2)) { continue; }
                        let fx = p1_build(g, jobs, tokens, round, 300 + index as u64, kinds)?;
                        let mut cands = vec![];
                        for (name, f, wide) in [("rows", spread, true)] {
                            let mut cand = p1_arm(g, jobs, &fx, Shape::Rows { spread: wide }, tokens)?;
                            launch_batch(g, f, &mut cand.batch)?; cands.push((name, cand));
                        }
                        g.sync()?;
                        for token in 0..tokens {
                            let mut reference = p1_arm(g, jobs, &fx, Shape::One(token), tokens)?;
                            launch_batch(g, one_base, &mut reference.batch)?; g.sync()?;
                            if token == 0 && kinds == (0, 0) {
                                // The candidate image's own one-row entry: the same bytes as the baseline image's.
                                let mut twin = p1_arm(g, jobs, &fx, Shape::One(token), tokens)?;
                                launch_batch(g, one_cand, &mut twin.batch)?; g.sync()?;
                                for ((pa, n), (pb, _)) in reference.outputs.iter().zip(&twin.outputs) {
                                    if g.read(*pa, *n)? != g.read(*pb, *n)? { return Err(format!("ROWSP1 ONE-ROW ENTRY CHANGED case={index} tokens={tokens} round={round}")); }
                                    one_row_checks += 1;
                                }
                            }
                            for (i, j) in jobs.iter().enumerate() {
                                let x = g.read(reference.outputs[i].0, reference.outputs[i].1)?;
                                if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err(format!("ROWSP1 guard changed (one row) case={index} job={i}")); }
                                let want = &x[16..16 + j.rows * 4];
                                for (name, cand) in &cands {
                                    let y = g.read(cand.outputs[i].0, cand.outputs[i].1)?;
                                    if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("ROWSP1 guard changed {name} case={index} job={i}")); }
                                    let got = &y[16 + token * j.rows * 4..16 + (token + 1) * j.rows * 4];
                                    if want != got {
                                        let at = want.iter().zip(got).position(|(p, q)| p != q).unwrap() / 4;
                                        let word = |v: &[u8]| u32::from_le_bytes(v[at * 4..at * 4 + 4].try_into().unwrap());
                                        return Err(format!("ROWSP1 BYTE FAIL {name} case={index} job={i} plain={} rows={} cols={}\
                                             tokens={tokens} token={token} round={round} patterns={kinds:?} row={at} one_row={:08x} n_row={:08x}",
                                            j.plain, j.rows, j.cols, word(want), word(got)));
                                    }
                                    comparisons += 1; bytes += (j.rows * 4) as u64;
                                }
                            }
                        }
                    }
                }
            }
            if index % 20 == 0 { println!("ROWSP1 case {index} ok comparisons={comparisons} elapsed_s={:.1}", started.elapsed().as_secs_f64()); }
            g.clear()?;
        }
        println!("ROWSP1 BYTES PASS entries=rows cases={} comparisons={comparisons} output_bytes={bytes} one_row_entry_checks={one_row_checks} elapsed_s={:.1}",
            cases.len(), started.elapsed().as_secs_f64());
    }
    if mode != "bytes" {
        // Timing: the real KDA batch (q, k, v at 4096 x 4096 MXFP8 + 384 plain rows) and its two parts alone; 8 weight copies
        // so every call streams from memory. Arm kinds: 0 one-row entry of the baseline image, 1 of the candidate image,
        // 2 rows (plain rows spread), 4 a loop of one-row launches.
        let (copies, calls, repeats) = (8usize, 16usize, 11usize);
        let mut arms: Vec<(String, usize, u32)> = vec![("one_row_base".into(), 1, 0), ("one_row_candidate_image".into(), 1, 1)];
        for (name, kind) in [("rows", 2u32), ("loop", 4)] { for n in [2usize, 3, 4] { arms.push((format!("{name}{n}"), n, kind)); } }
        for (label, jobs) in [("real: mx3 4096x4096 + plain 384x4096", real.clone()),
            ("mx only: mx3 4096x4096", vec![job(m, 4096, 4096), job(m, 4096, 4096), job(m, 4096, 4096)]),
            ("plain only: 384x4096", vec![job(p, 384, 4096)])] {
            let mut fixtures = vec![];
            for c in 0..copies { fixtures.push(p1_build(g, &jobs, 4, true, 7 + c as u64, (0, 0))?); }
            let mut built: Vec<Vec<Vec<Built>>> = vec![];
            for (_, tokens, kind) in &arms {
                let mut per_copy = vec![];
                for fx in &fixtures {
                    per_copy.push(match kind {
                        2 => vec![p1_arm(g, &jobs, fx, Shape::Rows { spread: true }, *tokens)?],
                        _ => { let mut v = vec![]; for t in 0..*tokens { v.push(p1_arm(g, &jobs, fx, Shape::One(t), *tokens)?); } v }
                    });
                }
                built.push(per_copy);
            }
            let mut times: Vec<Vec<f64>> = vec![vec![]; arms.len()];
            for repeat in 0..repeats {
                for k in 0..arms.len() {
                    let arm = (k + repeat) % arms.len();
                    let f = match arms[arm].2 { 0 => one_base, 2 => spread, _ => one_cand };
                    for b in built[arm][0].iter_mut() { launch_batch(g, f, &mut b.batch)?; }
                    g.sync()?;
                    let t0 = Instant::now();
                    for i in 0..calls { for b in built[arm][(repeat + i) % copies].iter_mut() { launch_batch(g, f, &mut b.batch)?; } }
                    g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
                }
            }
            let weight_bytes: usize = jobs.iter().map(|j| if j.plain { j.rows * j.cols * 2 } else { j.rows * j.cols + j.rows * j.cols / 32 }).sum();
            let one = median(times[0].clone());
            println!("ROWSP1 TIME batch=\"{label}\" weight_mib={:.2} floor_us_at_230={:.1} copies={copies} calls={calls} repeats={repeats}",
                weight_bytes as f64 / 1048576.0, weight_bytes as f64 / 230e9 * 1e6);
            let mut ratio = [[0.0f64; 5]; 5];
            for (k, (name, tokens, kind)) in arms.iter().enumerate() {
                let t = median(times[k].clone());
                let (lo, hi) = (times[k].iter().cloned().fold(f64::MAX, f64::min), times[k].iter().cloned().fold(0.0, f64::max));
                ratio[*kind as usize][*tokens] = t / one;
                println!("ROWSP1 TIME batch=\"{label}\" arm={name} rows={tokens} us_per_step={t:.1} min={lo:.1} max={hi:.1} against_one_row={:.3}", t / one);
            }
            if label.starts_with("real") {
                for (name, kind) in [("rows", 2usize)] {
                    let r = ratio[kind];
                    let verdict = if r[2] >= 1.5 { "KILL (2 rows at or above 1.5x)" } else if r[2] <= 1.15 && r[4] <= 1.35 { "PASS" } else { "BETWEEN THE LINES" };
                    println!("ROWSP1 LINES entry={name} rows2={:.3} (pass <= 1.15, kill >= 1.5) rows3={:.3} rows4={:.3} (pass <= 1.35) verdict={verdict}", r[2], r[3], r[4]);
                }
            }
            g.clear()?;
        }
    }
    println!("ROWSP1 DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
