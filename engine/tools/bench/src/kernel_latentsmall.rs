//! The small form of query and project in the latent absorption prompt tile (seat CHATS4): every output byte of
//! `latent_absorption_tile` in the NEW image against the same entry in the OLD image (the base commit), for 1 to 17
//! tokens and all four operations, then the time of both. One process, one device. Safe code only. It does not prove
//! token IDs. The new body runs for query and project at 1 to 8 tokens; every other launch keeps the tile's own body,
//! so its words must be equal too (a control on the unchanged path).
//! Run: kernel-bench --latent-small <old image> <new image> [all|proof|time]
use super::{Buffer, Case, GUARD, Gpu, Result, THREADS, arguments, fixture, median, ptr, sha256};
use std::time::Instant;
type H = super::super::H;
/// (op number, name). The kernel traps on any other op.
const OPERATIONS: [(u32, &str); 4] = [(21, "query"), (25, "project"), (22, "scores"), (24, "values")];
/// The grid of the tile: a block for each outer index, 128 rows of M and 32 columns of N.
fn grid(c: &Case, op: u32) -> u32 { let (outer, n, m, _) = c.tile(op); (outer * m.div_ceil(128) * n.div_ceil(32)) as u32 }
fn filled(g: &mut Gpu, words: usize, value: f32) -> Result<Buffer> { Buffer::new(g, &vec![value; words]) }
/// The 4-byte words of `a` that differ from `b`.
fn differ(a: &[u8], b: &[u8]) -> u64 { a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).filter(|(x, y)| x != y).count() as u64 }
fn guards(bytes: &[u8]) -> Result<()> {
    if bytes[..16] != GUARD || bytes[bytes.len() - 16..] != GUARD { return Err("output guard changed".into()); }
    Ok(())
}
fn launch(g: &mut Gpu, f: H, c: &Case, op: u32, fx: &super::Fixture, words: usize, fill: f32) -> Result<Vec<u8>> {
    let out = filled(g, words, fill)?; let mut a = arguments(c, op, fx, out.address);
    g.launch(f, [grid(c, op), 1, 1], THREADS, 0, &mut [ptr(&mut a)])?; g.sync()?;
    let got = out.read(g)?; guards(&got)?; Ok(got)
}
/// Every output byte: new = old, served-like and scattered selections (zero counts at token 5 of a scattered one), every
/// input class of the fixture, 1 to 17 tokens, cache lengths short and long. Outputs start poisoned (different fills).
fn proof(g: &mut Gpu, old: H, new: H) -> Result<()> {
    let (mut total, mut wrong, mut caught) = (0u64, 0u64, 0u64);
    // (tokens, cache rows, input class, served selection, seed)
    let mut cases: Vec<(usize, usize, u32, bool, u64)> = Vec::new();
    for tokens in 1..=8 { cases.push((tokens, 8192, 0, true, 31 + tokens as u64)); cases.push((tokens, 300, 0, false, 5 + tokens as u64)); }
    for (kind, seed) in [(2u32, 7u64), (4, 9), (5, 11), (6, 13)] { cases.push((8, 2051, kind, false, seed)); cases.push((3, 5000, kind, true, seed + 1)); }
    for tokens in [9usize, 16, 17] { cases.push((tokens, 8192, 0, true, 41)); }
    for (index, &(tokens, rows, kind, served, seed)) in cases.iter().enumerate() {
        let c = Case::model(tokens, rows); let fx = fixture(g, &c, kind, served, seed)?;
        for &(op, name) in &OPERATIONS {
            let words = c.tile(op).3;
            let want = launch(g, old, &c, op, &fx, words, 1.5)?;
            let got = launch(g, new, &c, op, &fx, words, 2.5)?;
            let d = differ(&want, &got);
            total += words as u64; wrong += d;
            println!("LATENT SMALL PROOF {name} tokens={tokens} rows={rows} class={kind} served={served} words={words} differ={d}");
            // Control on the first cases and on a token count of the new body: other operands must not give the old words.
            // Class 6 takes a few values only (2^24, 1, 2, 3, -2^24), so other seeds can give the same words: no control there.
            if (index < 2 || (tokens == 8 && op == 25)) && kind != 6 {
                let other = fixture(g, &c, kind, served, seed + 1000)?;
                let moved = launch(g, new, &c, op, &other, words, 2.5)?;
                let d = differ(&want, &moved);
                if d == 0 { return Err(format!("LATENT SMALL CONTROL FAIL {name} tokens={tokens}: other operands gave the same words")); }
                caught += 1;
                println!("LATENT SMALL CONTROL {name} tokens={tokens} other operands: {d} of {words} words differ, caught");
            }
        }
        g.clear()?;
    }
    if wrong != 0 { return Err(format!("LATENT SMALL PROOF FAIL: {wrong} of {total} words differ")); }
    println!("LATENT SMALL PROOF PASS words={total} differ=0 controls_caught={caught}");
    Ok(())
}
/// Launch and sync, milliseconds: old and new, forward and back, 9 rounds, the median.
fn time(g: &mut Gpu, old: H, new: H) -> Result<()> {
    for tokens in [1usize, 2, 3, 4, 5, 8, 9, 16] {
        let c = Case::model(tokens, 8192); let fx = fixture(g, &c, 0, true, 31)?;
        for &(op, name) in &OPERATIONS[..2] {
            let out = filled(g, c.tile(op).3, 0.0)?; let mut a = arguments(&c, op, &fx, out.address);
            let arms = [("old", old), ("new", new)];
            for arm in &arms { g.launch(arm.1, [grid(&c, op), 1, 1], THREADS, 0, &mut [ptr(&mut a)])?; g.sync()?; }
            let mut times = [Vec::new(), Vec::new()];
            for _ in 0..9 {
                for i in [0usize, 1, 1, 0] {
                    g.sync()?; let start = Instant::now(); g.launch(arms[i].1, [grid(&c, op), 1, 1], THREADS, 0, &mut [ptr(&mut a)])?; g.sync()?;
                    times[i].push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            guards(&out.read(g)?)?;
            let [o, n] = times;
            println!("LATENT SMALL TIME {name} tokens={tokens} ms: old={:.4} new={:.4}", median(o), median(n));
        }
        g.clear()?;
    }
    Ok(())
}
pub(crate) fn run(old_image: &str, new_image: &str, mode: &str) -> Result<()> {
    let old_bytes = std::fs::read(old_image).map_err(|e| e.to_string())?; let new_bytes = std::fs::read(new_image).map_err(|e| e.to_string())?;
    println!("LATENT SMALL old={} new={} mode={mode}", sha256::digest(&old_bytes), sha256::digest(&new_bytes));
    if !matches!(mode, "all" | "proof" | "time") { return Err(format!("--latent-small: the mode is all, proof or time, not {mode}")); }
    let mut g = Gpu::new()?; let mo = g.module(&old_bytes)?; let mn = g.module(&new_bytes)?;
    let old = g.function(mo, "latent_absorption_tile")?; let new = g.function(mn, "latent_absorption_tile")?;
    if mode != "time" { proof(&mut g, old, new)?; }
    if mode != "proof" { time(&mut g, old, new)?; }
    println!("LATENT SMALL DONE mode={mode}");
    Ok(())
}
