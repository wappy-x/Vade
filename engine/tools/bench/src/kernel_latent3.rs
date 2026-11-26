//! The third form of the latent absorption prompt tile (seat F8TC): every output byte against the first form
//! (`latent_absorption_tile`) of the SAME image, then the time of both. One process, one device. Safe code only. It
//! does not prove token IDs.
//! Run: kernel-bench --latent3 <image> [all|proof|time] [all|query|scores|values|project] [heads]
use super::{Buffer, Case, GUARD, Gpu, Result, THREADS, arguments, fixture, median, ptr, sha256};
use std::time::Instant;
type H = super::super::H;
/// One operation: its number, the bench forms of its third body (name, entry), the blocks of the third form for a
/// token count, and the block sizes tried in the proof and in the timing.
struct Operation { op: u32, name: &'static str, forms: &'static [(&'static str, &'static str)], blocks: fn(usize) -> u32, proof: &'static [u32], time: &'static [u32] }
/// Query and project: a block is one head and 32 tokens.
fn head_blocks(tokens: usize) -> u32 { (32 * tokens.div_ceil(32)) as u32 }
const WEIGHTS: &[(&str, &str)] = &[("third", "latent_weights3"), ("panel", "latent_weights3_panel"), ("tile64", "latent_weights3_tile64")];
const OPERATIONS: [Operation; 4] = [
    Operation { op: 21, name: "query", forms: WEIGHTS, blocks: head_blocks, proof: &[256, 384], time: &[256, 384] },
    Operation { op: 25, name: "project", forms: WEIGHTS, blocks: head_blocks, proof: &[256, 384], time: &[256, 384] },
    // Scores: the served form (no hints) and the bench form with hints for the next stage. A block is one token.
    Operation { op: 22, name: "scores", forms: &[("third", "latent_scores3"), ("packed", "latent_scores3_packed"), ("quarters", "latent_scores3_quarters")], blocks: |tokens| tokens as u32, proof: &[256, 384], time: &[256, 384] },
    // Values: the served form (no hints) and the bench form with hints 4 steps ahead. A block is one token and 128 features.
    Operation { op: 24, name: "values", forms: &[("third", "latent_values3"), ("packed-single", "latent_values3_packed_single"), ("wide", "latent_values3_wide"), ("full", "latent_values3_full"), ("full-park", "latent_values3_full_park")], blocks: |tokens| 4 * tokens as u32, proof: &[256], time: &[256] },
];
fn form_blocks(o: &Operation, c: &Case, name: &str) -> u32 {
    let tokens = c.tokens;
    if c.heads == 16 {
        return match o.op {21|25 => (c.heads * tokens.div_ceil(32)) as u32,24 if !name.starts_with("full") => 2*tokens as u32,_ => tokens as u32};
    }
    if name.starts_with("tile64") { (c.heads * tokens.div_ceil(64)) as u32 } else if name.starts_with("full") { tokens as u32 } else if name.starts_with("wide") { 2 * tokens as u32 } else { (o.blocks)(tokens) }
}
fn forms_for(o: &Operation, heads: usize) -> &[(&str,&str)] {
    if heads != 16 { return o.forms; }
    match o.op {
        21|25 => &[("third","latent_weights3_h16"),("panel","latent_weights3_panel_h16")],
        22 => &[("third","latent_scores3_h16"),("packed","latent_scores3_packed_h16")],
        _ => &[("third","latent_values3_h16"),("packed","latent_values3_packed_h16"),("full-park","latent_values3_full_park_h16")],
    }
}
fn form_shared(o: &Operation, name: &str) -> u32 { if name.starts_with("tile64") { if o.op == 21 { 32768 } else { 65536 } } else { 0 } }
/// The grid of the first form: a block for each outer index, 128 rows of M and 32 columns of N.
fn old_grid(c: &Case, op: u32) -> u32 { let (outer, n, m, _) = c.tile(op); (outer * m.div_ceil(128) * n.div_ceil(32)) as u32 }
/// A driver error that only refuses the launch (too many threads for the registers of the kernel).
fn refused(e: &str) -> bool { e.contains("701") }
fn filled(g: &mut Gpu, words: usize, value: f32) -> Result<Buffer> { Buffer::new(g, &vec![value; words]) }
fn guards(bytes: &[u8]) -> Result<()> {
    if bytes[..16] != GUARD || bytes[bytes.len() - 16..] != GUARD { return Err("output guard changed".into()); }
    Ok(())
}
/// A bench-only high-half cache mirror. Packing is outside the kernel timing; a served pipeline must measure its
/// preparation too. Every ordinary and wrong-input case gets its own mirror from that case's original operand.
fn packed_cache(g: &mut Gpu, fx: &super::Fixture, op: u32) -> Result<Option<Buffer>> {
    if !matches!(op, 22 | 24) { return Ok(None); }
    let raw = fx.latent.read(g)?; guards(&raw)?;
    let bytes: Vec<u8> = raw[16..raw.len()-16].chunks_exact(4).flat_map(|v| [v[2], v[3]]).collect();
    Ok(Some(Buffer::raw(g, &bytes)?))
}
fn with_cache(mut a: super::Arguments, name: &str, packed: Option<Buffer>) -> Result<super::Arguments> {
    if name.starts_with("packed") || name.starts_with("wide") || name.starts_with("full") || name == "park" || name == "quarters" {
        let b = packed.ok_or("packed form has no packed cache")?;
        a.input[1] = b.address; a.lengths[1] = (b.bytes / 4) as u64;
        a.shape.op = match a.shape.op {22=>30,24=>31,_=>return Err("packed cache requires scores or values".into())};
    }
    Ok(a)
}
/// Cost: one offline compiler pass, before GPU timing. Gain: register, shared and spill evidence for the next cut.
/// These are offline compiler resources, not driver occupancy or measured runtime stalls.
fn resources(image: &str, ptx: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(ptx).map_err(|_| "The latent tile PTX is not UTF-8")?;
    let target = text.lines().find_map(|line| {
        line.trim_start().strip_prefix(".target ")
            .and_then(|value| value.split([' ',',','\t']).find(|token| !token.is_empty()))
    }).ok_or("The latent tile PTX has no target")?;
    let out = std::env::temp_dir().join(format!("latent-resource-{}.cubin", std::process::id()));
    let result = std::process::Command::new("/usr/local/cuda/bin/ptxas")
        .arg(format!("-arch={target}")).args(["-O3", "-v", image, "-o"]).arg(&out).output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&out);
    let log = String::from_utf8_lossy(&result.stderr);
    if !result.status.success() { return Err(format!("latent offline compile failed: {log}")); }
    let mut keep = false;
    for line in log.lines() {
        if line.contains("Compiling entry function") { keep = line.contains("'latent_"); }
        if keep { println!("LATENT3 ASSEMBLY {line}"); }
    }
    Ok(())
}
/// Every output byte: third form = first form, on served-like and on scattered selections, counts of every kind
/// (zero, partial steps and tiles, full), small tables, and the hard input classes.
fn proof(g: &mut Gpu, old: H, o: &Operation, forms: &[(&str, H)], heads: usize) -> Result<u64> {
    // (tokens, cache rows, slots, input class, served selection, seed)
    let cases: [(usize, usize, usize, u32, bool, u64); 12] = [(1, 8192, 2051, 0, true, 31), (5, 300, 2051, 0, false, 5), (37, 2051, 2051, 2, false, 7),
        (64, 5000, 2051, 4, false, 9), (64, 8192, 2051, 5, true, 11), (40, 8192, 2051, 6, false, 13), (33, 100, 40, 0, false, 3), (19, 4096, 17, 2, false, 23),
        (256, 8192, 2051, 0, true, 31), (17, 8192, 2051, 0, true, 41), (1024, 8192, 2051, 0, true, 43), (1040, 9001, 2051, 0, false, 47)];
    let (mut total, mut wrong, mut caught) = (0u64, 0u64, 0u64);
    let group_cases = [(256,8192,2051,0,true,49),(512,8192,2051,0,true,51),(768,8192,2051,0,true,53),
        (1024,1024,2051,0,true,55),(1024,4096,2051,0,true,57),(256,8192,2051,6,false,59)];
    let extra = if heads == 16 { &group_cases[..] } else { &[] };
    for (index, (tokens, rows, slots, kind, served, seed)) in cases.into_iter().chain(extra.iter().copied()).enumerate() {
        let c = Case { heads, slots, ..Case::model(tokens, rows) }; let fx = fixture(g, &c, kind, served, seed)?; let words = c.tile(o.op).3;
        let packed = packed_cache(g, &fx, o.op)?;
        let first = filled(g, words, 1.5)?; let mut a = arguments(&c, o.op, &fx, first.address);
        g.launch(old, [old_grid(&c, o.op), 1, 1], THREADS, 0, &mut [ptr(&mut a)])?; g.sync()?;
        let want = first.read(g)?; guards(&want)?;
        for (name, f) in forms {
            for threads in o.proof {
                if *name == "quarters" && *threads != 256 { continue; }
                let third = filled(g, words, 2.5)?; let mut a = with_cache(arguments(&c, o.op, &fx, third.address), name, packed)?;
                match g.launch(*f, [form_blocks(o, &c, name), 1, 1], *threads, form_shared(o, name), &mut [ptr(&mut a)]) {
                    Err(e) if refused(&e) => { println!("LATENT3 PROOF {} form={name} threads={threads} tokens={tokens} REFUSED by the driver", o.name); continue; }
                    other => other?,
                }
                g.sync()?;
                let got = third.read(g)?; guards(&got)?;
                let differ = want.chunks_exact(4).zip(got.chunks_exact(4)).filter(|(x, y)| x != y).count() as u64;
                total += words as u64; wrong += differ;
                println!("LATENT3 PROOF {} form={name} threads={threads} tokens={tokens} rows={rows} slots={slots} class={kind} served={served} words={words} differ={differ}", o.name);
            }
        }
        // Control: other operands. The third form on them must NOT give the words of the first form on the first operands.
        if index == 1 {
            let other = fixture(g, &c, kind, served, seed + 1000)?;
            let other_packed = packed_cache(g, &other, o.op)?;
            for (name, f) in forms {
                let third = filled(g, words, 2.5)?; let mut a = with_cache(arguments(&c, o.op, &other, third.address), name, other_packed)?;
                g.launch(*f, [form_blocks(o, &c, name), 1, 1], o.proof[0], form_shared(o, name), &mut [ptr(&mut a)])?; g.sync()?;
                let got = third.read(g)?; guards(&got)?;
                let differ = want.chunks_exact(4).zip(got.chunks_exact(4)).filter(|(x, y)| x != y).count() as u64;
                if differ == 0 { return Err(format!("LATENT3 CONTROL FAIL {} form={name}: other operands gave the same words", o.name)); }
                caught += 1;
                println!("LATENT3 CONTROL {} form={name} other operands: {differ} of {words} words differ, caught", o.name);
            }
        }
        g.clear()?;
    }
    if wrong != 0 { return Err(format!("LATENT3 PROOF FAIL {}: {wrong} of {total} words differ", o.name)); }
    println!("LATENT3 PROOF PASS {} words={total} differ=0 controls_caught={caught}", o.name);
    Ok(total)
}
/// Launch and sync, milliseconds: the first form, then every form and block size, forward and back, 7 rounds.
fn time(g: &mut Gpu, old: H, pack: Option<H>, o: &Operation, forms: &[(&str, H)], heads: usize) -> Result<()> {
    let cases = [(1024,8192),(1024,2051),(1024,32768),(256,8192),(64,8192),(17,8192),(8,8192),(2,8192)];
    let groups = [(1024,1024),(1024,2048),(1024,3072),(1024,4096),(1024,5120),(1024,6144),(1024,7168),
        (1024,8192),(256,8192),(512,8192),(768,8192),(17,8192),(64,8192)];
    let shapes = if heads == 16 {&groups[..]} else {&cases[..]};
    for &(tokens, rows) in shapes {
        let c = Case {heads,..Case::model(tokens, rows)}; let fx = fixture(g, &c, 0, true, 31)?;
        let packed = packed_cache(g, &fx, o.op)?;
        let out = filled(g, c.tile(o.op).3, 0.0)?; let a = arguments(&c, o.op, &fx, out.address);
        let mut arms = vec![("first".to_string(), old, [old_grid(&c, o.op), 1, 1], THREADS, a)];
        for (name, f) in forms { for threads in o.time {
            if *name == "quarters" && *threads != 256 { continue; }
            arms.push((format!("{name}@{threads}"), *f, [form_blocks(o, &c, name), 1, 1], *threads, with_cache(a, name, packed)?));
        } }
        let mut live = Vec::new();
        for mut arm in arms {
            match g.launch(arm.1, arm.2, arm.3, form_shared(o, &arm.0), &mut [ptr(&mut arm.4)]) {
                Err(e) if refused(&e) => { println!("LATENT3 TIME {} {} REFUSED by the driver", o.name, arm.0); continue; }
                other => other?,
            }
            g.sync()?; live.push(arm);
        }
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); live.len()];
        for _ in 0..7 {
            let order: Vec<usize> = (0..live.len()).chain((0..live.len()).rev()).collect();
            for i in order {
                let mut a = live[i].4;
                g.sync()?; let start = Instant::now(); g.launch(live[i].1, live[i].2, live[i].3, form_shared(o, &live[i].0), &mut [ptr(&mut a)])?; g.sync()?;
                times[i].push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        guards(&out.read(g)?)?;
        let cells: Vec<String> = live.iter().zip(times).map(|(arm, t)| format!("{}={:.4}", arm.0, median(t))).collect();
        println!("LATENT3 TIME {} tokens={tokens} rows={rows} selected={} ms: {}", o.name, c.first(), cells.join(" "));
        if let Some(pack) = pack && o.op == 22 {
            time_pack(g, pack, &c, &fx, packed.ok_or("missing packed cache")?)?;
        }
        g.clear()?;
    }
    Ok(())
}

/// The shared mirror is one lossless pack of the BF16 cache before scores and values.
fn time_pack(g: &mut Gpu, pack: H, c: &Case, fx: &super::Fixture, expected: Buffer) -> Result<()> {
    let want = expected.read(g)?;
    guards(&want)?;
    let words = c.rows * c.cols;
    let bf16 = want[16..want.len() - 16].chunks_exact(2).flat_map(|x| [0, 0, x[0], x[1]]).collect::<Vec<_>>();
    let source = Buffer::raw(g, &bf16)?;
    let destination = filled(g, words / 2, 1.5)?;
    let mut args = arguments(c, 22, fx, destination.address);
    args.shape.op = 28; // WirePackBf16.
    args.shape.tokens = 1;
    args.shape.cols = words as u32;
    args.input = [source.address, 4, 4, 4, 4];
    args.lengths = [words as u64, 0, 0, 0, 0];
    args.units = (words / 2) as u64;
    let grid = [args.units.div_ceil(128) as u32, 1, 1];
    g.launch(pack, grid, 128, 0, &mut [ptr(&mut args)])?;
    g.sync()?;
    let got = destination.read(g)?;
    guards(&got)?;
    if got != want { return Err("the cache pack differs from its canonical BF16 words".into()); }
    let mut samples = Vec::new();
    for _ in 0..7 {
        g.sync()?;
        let start = Instant::now();
        g.launch(pack, grid, 128, 0, &mut [ptr(&mut args)])?;
        g.sync()?;
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    println!("LATENT3 PACK tokens={} rows={} words={} ms={:.4}", c.tokens, c.rows, words / 2, median(samples));
    Ok(())
}
pub(crate) fn run(image: &str, mode: &str, only: &str, heads: usize) -> Result<()> {
    let bytes = std::fs::read(image).map_err(|e| e.to_string())?;
    println!("LATENT3 image={} mode={mode} operations={only} heads={heads}", sha256::digest(&bytes));
    if ![16,32].contains(&heads) {return Err("--latent3 expects 16 or 32 heads".into());}
    if !matches!(mode, "all" | "proof" | "time") { return Err(format!("--latent3: the mode is all, proof or time, not {mode}")); }
    if !matches!(only, "all" | "query" | "scores" | "values" | "project") { return Err(format!("--latent3: the operations are all, query, scores, values or project, not {only}")); }
    if !bytes.starts_with(b"\x7fELF") { resources(image,&bytes)?; }
    let mut g = Gpu::new()?; let m = g.module(&bytes)?;
    let old = g.function(m, "latent_absorption_tile")?;
    let pack = (heads == 16 && mode != "proof" && matches!(only, "all" | "scores"))
        .then(|| g.function(m, "prompt_wire_bf16")).transpose()?;
    for o in &OPERATIONS {
        if only != "all" && only != o.name { continue; }
        let mut forms = Vec::new();
        for (name, entry) in forms_for(o,heads) { forms.push((*name, g.function(m, entry)?)); }
        if mode != "time" { proof(&mut g, old, o, &forms,heads)?; }
        if mode != "proof" { time(&mut g, old, pack, o, &forms,heads)?; }
    }
    println!("LATENT3 DONE mode={mode} operations={only}");
    Ok(())
}
