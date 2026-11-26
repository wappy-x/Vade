//! The ROWS FORM of the latent absorption prompt tile (seat RB3, piece P7): every output byte of `latent_rows_dense`
//! (query, project) and `latent_rows_attention` (scores, values) against `latent_absorption_tile` of the SAME image, for
//! 1 to 8 tokens, then the time of both at 1, 2, 4 and 8 tokens. One process, one device. Safe code only. It does not
//! prove token IDs.
//! Run: kernel-bench --latent-rows <image> [all|proof|time]
//! A second form, "cap128", loads the same image text with `.maxnreg 128` on the two rows entries (what the stamp's
//! register cap would write), so the cap's effect is measured on the same device in the same process.
use super::{Buffer, Case, GUARD, Gpu, Result, THREADS, arguments, fixture, median, ptr, sha256};
use std::time::Instant;
type H = super::super::H;
/// (op number, name, rows-form entry). The kernels trap on any other op.
const OPERATIONS: [(u32, &str, &str); 4] = [(21, "query", "latent_rows_dense"), (22, "scores", "latent_rows_attention"), (24, "values", "latent_rows_attention"), (25, "project", "latent_rows_dense")];
/// The tile's grid (`Form::blocks`): a block for each outer index, 128 rows of M and 32 columns of N.
fn tile_grid(c: &Case, op: u32) -> u32 { let (outer, n, m, _) = c.tile(op); (outer * m.div_ceil(128) * n.div_ceil(32)) as u32 }
/// The rows form's grid (`latent_absorb_rows::rows_form`): the tile's for query and project, `Form::step_blocks` (32 rows
/// of M a block) for scores and values.
fn rows_grid(c: &Case, op: u32) -> u32 { let (outer, n, m, _) = c.tile(op); if matches!(op, 22 | 24) { (outer * m.div_ceil(32) * n.div_ceil(32)) as u32 } else { tile_grid(c, op) } }
fn filled(g: &mut Gpu, words: usize, value: f32) -> Result<Buffer> { Buffer::new(g, &vec![value; words]) }
fn guards(bytes: &[u8]) -> Result<()> {
    if bytes[..16] != GUARD || bytes[bytes.len() - 16..] != GUARD { return Err("output guard changed".into()); }
    Ok(())
}
fn differ(a: &[u8], b: &[u8]) -> u64 { a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).filter(|(x, y)| x != y).count() as u64 }
fn run_one(g: &mut Gpu, f: H, grid: u32, c: &Case, op: u32, fx: &super::Fixture, fill: f32) -> Result<Vec<u8>> {
    let out = filled(g, c.tile(op).3, fill)?; let mut a = arguments(c, op, fx, out.address);
    g.launch(f, [grid, 1, 1], THREADS, 0, &mut [ptr(&mut a)])?; g.sync()?;
    let got = out.read(g)?; guards(&got)?; Ok(got)
}
/// Offline compiler resources of the entries that serve a check pass (registers, shared bytes, spills), before timing.
fn resources(image: &str) -> Result<()> {
    let out = std::env::temp_dir().join(format!("latent-rows-resource-{}.cubin", std::process::id()));
    let result = std::process::Command::new("/usr/local/cuda/bin/ptxas")
        .args(["-arch=sm_121a", "-O3", "-v", image, "-o"]).arg(&out).output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&out);
    let log = String::from_utf8_lossy(&result.stderr);
    // Evidence only: a failed offline compile is reported and the proof and timing still run.
    if !result.status.success() { println!("LATENTROWS ASSEMBLY SKIPPED: {}", log.lines().next().unwrap_or("")); return Ok(()); }
    let mut keep = false;
    for line in log.lines() {
        if line.contains("Compiling entry function") { keep = ["'latent_rows_", "'latent_absorption_tile'", "'latent_absorption'", "'latent_absorption_scores'"].iter().any(|n| line.contains(n)); }
        if keep { println!("LATENTROWS ASSEMBLY {line}"); }
    }
    Ok(())
}
/// Every output byte: rows form = tile, served-like and scattered selections (a zero count at token 5 of a scattered
/// one), every input class of the fixture, 1 to 8 tokens, short and long caches, values of one to four staged chunks.
/// Outputs start poisoned with different fills, so an unwritten word differs.
fn proof(g: &mut Gpu, tile: H, form: &str, rows: &[(&str, H)]) -> Result<()> {
    let (mut total, mut wrong, mut caught) = (0u64, 0u64, 0u64);
    // (tokens, cache rows, input class, served selection, seed)
    let mut cases: Vec<(usize, usize, u32, bool, u64)> = Vec::new();
    for tokens in 1..=8 { cases.push((tokens, 300, 0, true, 31 + tokens as u64)); cases.push((tokens, 8192, 0, true, 41 + tokens as u64)); cases.push((tokens, 2051, 0, false, 5 + tokens as u64)); }
    for (kind, seed) in [(2u32, 7u64), (4, 9), (5, 11), (6, 13)] { cases.push((8, 2051, kind, false, seed)); cases.push((3, 600, kind, true, seed + 1)); cases.push((6, 130, kind, false, seed + 2)); }
    for (index, &(tokens, rows_len, kind, served, seed)) in cases.iter().enumerate() {
        let c = Case::model(tokens, rows_len); let fx = fixture(g, &c, kind, served, seed)?;
        for &(op, name, entry) in &OPERATIONS {
            let f = rows.iter().find(|r| r.0 == entry).ok_or("missing rows entry")?.1;
            let words = c.tile(op).3;
            let want = run_one(g, tile, tile_grid(&c, op), &c, op, &fx, 1.5)?;
            let got = run_one(g, f, rows_grid(&c, op), &c, op, &fx, 2.5)?;
            let d = differ(&want, &got); total += words as u64; wrong += d;
            println!("LATENTROWS PROOF form={form} {name} tokens={tokens} rows={rows_len} selected={} class={kind} served={served} words={words} differ={d}", c.first());
            // Control: other operands. The rows form on them must NOT give the tile's words on the first operands.
            if index == 1 {
                let other = fixture(g, &c, kind, served, seed + 1000)?;
                let again = run_one(g, f, rows_grid(&c, op), &c, op, &other, 2.5)?;
                let d = differ(&want, &again);
                if d == 0 { return Err(format!("LATENTROWS CONTROL FAIL {name}: other operands gave the same words")); }
                caught += 1; println!("LATENTROWS CONTROL {name} other operands: {d} of {words} words differ, caught");
            }
        }
        g.clear()?;
    }
    if wrong != 0 { return Err(format!("LATENTROWS PROOF FAIL: {wrong} of {total} words differ")); }
    println!("LATENTROWS PROOF PASS form={form} words={total} differ=0 controls_caught={caught} cases={}", cases.len());
    Ok(())
}
/// Launch and sync, microseconds: tile and rows form, alternating order, 9 rounds; the median a launch. One latent layer
/// runs the four operations once a pass; the model has 11 latent layers a rank.
fn time(g: &mut Gpu, tile: H, forms: &[(&str, [(&str, H); 2])]) -> Result<()> {
    for rows_len in [256usize, 384, 1024, 4096] {
        for tokens in [1usize, 2, 4, 8] {
            let c = Case::model(tokens, rows_len); let fx = fixture(g, &c, 0, true, 31)?;
            // Arm 0 is the tile; arm 1 + i is form i. Every arm runs in every round, the order rotating.
            let mut layer = vec![0.0f64; forms.len() + 1]; let mut cells = Vec::new();
            for &(op, name, entry) in &OPERATIONS {
                let out = filled(g, c.tile(op).3, 0.0)?; let a = arguments(&c, op, &fx, out.address);
                let mut arms = vec![(tile, tile_grid(&c, op))];
                for (_, rows) in forms { arms.push((rows.iter().find(|r| r.0 == entry).ok_or("missing rows entry")?.1, rows_grid(&c, op))); }
                for &(h, grid) in &arms { let mut x = a; g.launch(h, [grid, 1, 1], THREADS, 0, &mut [ptr(&mut x)])?; g.sync()?; }
                let mut times = vec![Vec::new(); arms.len()];
                for round in 0..9 {
                    for k in 0..arms.len() {
                        let arm = (round + k) % arms.len(); let mut x = a;
                        g.sync()?; let start = Instant::now(); g.launch(arms[arm].0, [arms[arm].1, 1, 1], THREADS, 0, &mut [ptr(&mut x)])?; g.sync()?;
                        times[arm].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
                guards(&out.read(g)?)?;
                let medians: Vec<f64> = times.into_iter().map(median).collect();
                for (i, m) in medians.iter().enumerate() { layer[i] += m; }
                let rest: String = forms.iter().zip(&medians[1..]).map(|(f, m)| format!(" {name}_{}_us={m:.1}", f.0)).collect();
                cells.push(format!("{name}_tile_us={:.1}{rest}", medians[0]));
            }
            let saved: String = forms.iter().zip(&layer[1..]).map(|(f, l)| format!(" layer_{}_us={l:.1} saved_{}_ms_a_pass_11_layers={:.3}", f.0, f.0, (layer[0] - l) * 11.0 / 1e3)).collect();
            println!("LATENTROWS TIME tokens={tokens} rows={rows_len} selected={} {} layer_tile_us={:.1}{saved}", c.first(), cells.join(" "), layer[0]);
            g.clear()?;
        }
    }
    Ok(())
}
pub(crate) fn run(image: &str, mode: &str) -> Result<()> {
    let bytes = std::fs::read(image).map_err(|e| e.to_string())?;
    println!("LATENTROWS image={} mode={mode}", sha256::digest(&bytes));
    if !matches!(mode, "all" | "proof" | "time") { return Err(format!("--latent-rows: the mode is all, proof or time, not {mode}")); }
    if !bytes.starts_with(b"\x7fELF") { resources(image)?; }
    let mut g = Gpu::new()?; let m = g.module(&bytes)?;
    let tile = g.function(m, "latent_absorption_tile")?;
    let rows = [("latent_rows_dense", g.function(m, "latent_rows_dense")?), ("latent_rows_attention", g.function(m, "latent_rows_attention")?)];
    let mut forms = vec![("rows", rows)];
    // The same text with `.maxnreg 128` written after each rows entry's parameter list, as the stamp writes a cap.
    if !bytes.starts_with(b"\x7fELF") {
        let mut text = String::from_utf8(bytes.clone()).map_err(|_| "image is not text")?;
        for name in ["latent_rows_dense", "latent_rows_attention"] {
            let entry = text.find(&format!(".entry {name}(")).ok_or("rows entry missing")?;
            let body = entry + text[entry..].find("\n{").ok_or("rows entry has no body")?;
            if !text[entry..body].contains(".maxnreg") { text.insert_str(body + 1, ".maxnreg 128\n"); }
        }
        let mc = g.module(text.as_bytes())?;
        forms.push(("cap128", [("latent_rows_dense", g.function(mc, "latent_rows_dense")?), ("latent_rows_attention", g.function(mc, "latent_rows_attention")?)]));
    }
    let started = Instant::now();
    if mode != "time" { for (form, rows) in &forms { proof(&mut g, tile, form, rows)?; } }
    if mode != "proof" { time(&mut g, tile, &forms)?; }
    println!("LATENTROWS DONE mode={mode} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
