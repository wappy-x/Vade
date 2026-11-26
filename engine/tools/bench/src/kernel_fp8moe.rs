//! FP8 W8A8 grouped MoE layer (`--fp8-moe CONFIG`): the production decode pipeline
//! (route, token quantize, gate+up, SwiGLU quantize, down) per stage and per layer, arm A
//! against arm X, at checkpoint-config shapes on synthetic finite E4M3 experts.
use super::{Gpu, H, TIMER, fp8, mix, ptr};
use super::super::{Options, Result, median, sha256};
use std::time::Instant;

/// Routed-expert shapes read from config.json (first occurrence of each key).
#[derive(Clone, Copy)]
struct Shape { hidden: usize, intermediate: usize, experts: usize, top: usize, layers: usize }
fn config_number(text: &str, key: &str) -> Result<usize> {
    let at = text.find(&format!("\"{key}\"")).ok_or(format!("config has no {key}"))?;
    let rest = text[at + key.len() + 2..].trim_start().strip_prefix(':').ok_or(format!("config {key}: no value"))?.trim_start();
    rest[..rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len())].parse().map_err(|_| format!("config {key}: not an integer"))
}
fn shape(path: &str) -> Result<Shape> {
    let whole = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let text = whole.find("\"text_config\"").map_or(&whole[..], |at| &whole[at..]);
    let s = Shape { hidden: config_number(text, "hidden_size")?, intermediate: config_number(text, "moe_intermediate_size")?,
        experts: config_number(text, "num_experts")?, top: config_number(text, "num_experts_per_tok")?,
        layers: config_number(text, "num_hidden_layers")? };
    if !s.hidden.is_multiple_of(128) || !s.intermediate.is_multiple_of(128) || s.top == 0 || s.top > s.experts {
        return Err("config shapes are not K128 blocks".into());
    }
    // The production grouped W8A8 entries fix their grids to these dimensions (fp8.rs
    // fp8_grouped_ffn); a checkpoint with others has no production pipeline to time.
    if (s.hidden, s.intermediate, s.experts, s.top) != (2560, 640, 512, 10) {
        return Err("production grouped FP8 kernels cover hidden 2560, intermediate 640, 512 experts, top 10 only".into());
    }
    Ok(s)
}

/// Device state shared by both arms: weights, R route sets, one scratch.
struct Layer {
    tokens: usize, ring: Vec<u64>, error: u64, desc: u64, desc_rank: u64, input: u64, output: u64, act: [u64; 6],
    route: fp8::GroupedRouteParams, task_capacity: usize, distinct: Vec<usize>,
    /// Zeroed fused-gate arrival counters; a complete exchanged middle for rank-0 down.
    counters: u64, full: [u64; 2], rows: usize, concentrated: bool,
    /// Expert of every (token, slot) pair, per route set.
    pair_experts: Vec<Vec<usize>>,
}
/// Rank-0 local gate width of expert `x`: the two-rank cut (3 - x%2 blocks), except that
/// every eighth expert has a one-block shard and every sixteenth an empty one, as
/// narrower worlds give (partial groups, and experts with no gate CTA at all).
fn rank_width(x: usize, intermediate: usize) -> usize {
    (if x % 16 == 7 { 0 } else if x % 8 == 3 { 128 } else { (3 - x % 2) * 128 }).min(intermediate)
}
fn e4m3_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    for (i, chunk) in out.chunks_mut(8).enumerate() {
        let r = mix(seed ^ i as u64);
        for (j, b) in chunk.iter_mut().enumerate() {
            let v = (r >> (j * 8)) as u8;
            // Exponent 4..11 (|x| 2^-3..2^4), random sign and mantissa; never NaN.
            *b = (v & 0x80) | ((4 + (v >> 3) % 8) << 3) | (v & 7);
        }
    }
    out
}
fn scale_bytes(seed: u64, n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| (0x3c00u16 + (mix(seed ^ i as u64) % 64) as u16).to_le_bytes()).collect()
}
fn layer(g: &mut Gpu, s: Shape, tokens: usize, ring: usize, concentrated: bool) -> Result<Layer> {
    let (e, h, i) = (s.experts, s.hidden, s.intermediate);
    let matrix = i * h;
    let mut weights = [0u64; 3];
    let mut scales = [0u64; 3];
    for (m, (w, sc)) in weights.iter_mut().zip(scales.iter_mut()).enumerate() {
        *w = g.alloc(e * matrix)?;
        // 64 MiB host slabs keep upload memory bounded; every expert's bytes differ.
        let slab = (64 << 20) / matrix * matrix;
        for off in (0..e * matrix).step_by(slab) {
            let len = slab.min(e * matrix - off);
            g.upload_at(*w + off as u64, &e4m3_bytes(((0x5eed_0000 + m as u64) << 40) ^ off as u64, len))?;
        }
        *sc = g.upload(&scale_bytes(0x5ca1_e000 + m as u64, e * matrix / 16384))?;
    }
    let (ks, ns) = (h / 128, i / 128);
    let mut table = Vec::with_capacity(e * 7);
    let mut rank = Vec::with_capacity(e * 7);
    for x in 0..e {
        let at = |m: usize| weights[m] + (x * matrix) as u64;
        let sat = |m: usize| scales[m] + (x * ns * ks * 2) as u64;
        table.extend([at(0), sat(0), at(1), sat(1), at(2), sat(2), i as u64]);
        // Rank 0 of the two-rank split: gate/up own the first (3 - e%2) K128 blocks
        // (fp8_middle_unpack's cut); down owns output rows 0..hidden/2 over full K.
        rank.extend([at(0), sat(0), at(1), sat(1), at(2), sat(2), rank_width(x, i) as u64]);
    }
    let words = |v: &[u64]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let desc = g.upload(&words(&table))?;
    let desc_rank = g.upload(&words(&rank))?;
    let pairs = tokens * s.top;
    let mut sets = Vec::new();
    let mut distinct = Vec::new();
    let mut pair_experts = Vec::new();
    for r in 0..ring {
        let mut host = Vec::with_capacity(pairs * 3);
        let mut seen = vec![false; e];
        for t in 0..tokens {
            let mut chosen: Vec<usize> = Vec::new();
            let mut z = 0u64;
            while chosen.len() < s.top {
                // Concentrated: every token routes to the same ten experts (rows fill planes).
                let x = (mix(((r * 1000 + if concentrated { 0 } else { t }) as u64) << 20 ^ z) % e as u64) as usize;
                z += 1;
                if !chosen.contains(&x) { chosen.push(x); }
            }
            for (slot, &x) in chosen.iter().enumerate() {
                seen[x] = true;
                host.extend([t as f32, x as f32, 0.05 + slot as f32 * 0.01]);
            }
        }
        distinct.push(seen.iter().filter(|&&v| v).count());
        pair_experts.push(host.as_chunks::<3>().0.iter().map(|t| t[1] as usize).collect());
        sets.push(g.upload(&host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?);
    }
    let task_capacity = pairs.div_ceil(128) + pairs.min(e);
    let rows = pairs + 15 * pairs.min(e);
    // Scratch word layout of Backend::prepare_grouped (fp8.rs), routing status first.
    let scratch_words = tokens + e + e + 1 + 2 * pairs + tokens + 1 + 2 * task_capacity + 2 * rows;
    let base = g.alloc(scratch_words * 4)?;
    g.upload_at(base, &vec![0; scratch_words * 4])?;
    let counts = base + (tokens * 4) as u64;
    let starts = counts + (e * 4) as u64;
    let pair_to_row = starts + ((e + 1) * 4) as u64;
    let token_pairs = pair_to_row + (pairs * 4) as u64;
    let token_count = token_pairs + (pairs * 4) as u64;
    let tasks = token_count + (tokens * 4) as u64;
    let sorted_row = tasks + 4 + (task_capacity * 8) as u64;
    let output_row = sorted_row + (rows * 4) as u64;
    // grouped_layout(rows, tokens, true): FP32 gate|up, E4M3 middle, its scales, token codes, token scales.
    let sizes = [0, rows * i * 8, rows * i, rows * i / 32, tokens * h, tokens * h / 32];
    let mut act = [0u64; 6];
    let mut end = 0usize;
    for (a, size) in act.iter_mut().zip(sizes) { let start = end.div_ceil(256) * 256; *a = start as u64; end = start + size; }
    let arena = g.alloc(end)?;
    for a in &mut act { *a += arena; }
    let input = g.upload(&(0..tokens * h).flat_map(|j| (((mix(0x1a9 ^ j as u64) % 4001) as f32 - 2000.0) / 1000.0).to_le_bytes()).collect::<Vec<_>>())?;
    let output = g.alloc(tokens * (s.top + 1) * h * 4)?;
    let counter_words = fp8::fused_gate_counters(task_capacity, i);
    let counters = g.alloc(counter_words * 4)?;
    g.upload_at(counters, &vec![0; counter_words * 4])?;
    // Every row equal: the multi-token route's row order inside an expert is not stable
    // (fp8_grouped_count), and production's exchange follows pair_to_row, not row order.
    let (code_row, scale_row) = (e4m3_bytes(0xf011, i), (0..i / 128).map(|j| 0.004 + (mix(0xf5 ^ j as u64) % 64) as f32 * 1e-4).collect::<Vec<f32>>());
    let full = [g.upload(&code_row.repeat(rows))?, g.upload(&scale_row.repeat(rows).iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?];
    let route = fp8::GroupedRouteParams { error: base, pairs: sets[0], counts, starts, pair_to_row, token_pairs, token_count, tasks,
        sorted_row, output_row, experts: e as u32, tokens: tokens as u32, pair_count: pairs as u32, row_capacity: rows as u32 };
    Ok(Layer { tokens, ring: sets, error: base, desc, desc_rank, input, output, act, route, task_capacity, distinct, counters, full, rows, concentrated, pair_experts })
}

/// One stage launch: function, grid, block, and its parameter block.
enum Param { Route(fp8::GroupedRouteParams), Quantize(fp8::QuantizeParams), Grouped(fp8::GroupedParams) }
struct Stage { name: &'static str, f: H, grid: [u32; 3], threads: u32, p: Param }
fn launch(g: &Gpu, st: &mut Stage) -> Result<()> {
    let arg = match &mut st.p { Param::Route(p) => ptr(p), Param::Quantize(p) => ptr(p), Param::Grouped(p) => ptr(p) };
    g.launch(st.f, st.grid, st.threads, 0, &mut [arg])
}

/// Production decode order of Backend::fp8_grouped_ffn (W8A8): world 1 (`rank` false), or
/// rank 0 of two (`rank` true: local gate widths, then down over the owned output half
/// of a complete exchanged middle; transport kernels excluded).
fn pipeline(g: &Gpu, m: H, l: &Layer, s: Shape, set: usize, rank: bool, fused: bool, entries: [&str; 2], small: bool) -> Result<Vec<Stage>> {
    let f = |n: &str| g.function(m, n);
    let tokens = l.tokens;
    let route = fp8::GroupedRouteParams { pairs: l.ring[set], ..l.route };
    let mut out = Vec::new();
    if tokens == 1 {
        out.push(Stage { name: "route", f: f("fp8_grouped_single")?, grid: [1, 1, 1], threads: 256, p: Param::Route(route) });
    } else if small && tokens < 32 {
        out.push(Stage { name: "route", f: f("fp8_grouped_single")?, grid: [1, 1, 1], threads: fp8::SMALL_ROUTE_THREADS, p: Param::Route(route) });
    } else {
        let pairs = tokens * s.top;
        out.push(Stage { name: "route", f: f("fp8_grouped_clear")?, grid: [s.experts.max(tokens * 32).div_ceil(256) as u32, 1, 1], threads: 256, p: Param::Route(route) });
        out.push(Stage { name: "route", f: f("fp8_grouped_count")?, grid: [pairs.div_ceil(256) as u32, 1, 1], threads: 256, p: Param::Route(route) });
        out.push(Stage { name: "route", f: f("fp8_grouped_prefix")?, grid: [1, 1, 1], threads: 32, p: Param::Route(route) });
        out.push(Stage { name: "route", f: f("fp8_grouped_order")?, grid: [tokens.div_ceil(128) as u32, 1, 1], threads: 128, p: Param::Route(route) });
    }
    let [_, middle, middle_q, middle_scales, input_q, input_scales] = l.act;
    out.push(Stage { name: "quantize", f: f("fp8_quantize_128")?, grid: [(tokens * s.hidden / 128).div_ceil(8) as u32, 1, 1], threads: 256,
        p: Param::Quantize(fp8::QuantizeParams { input: l.input, output: input_q, scales: input_scales, rows: tokens as u32,
            cols: s.hidden as u32, round_input: 1, packed_output: 0 }) });
    let planes = fp8::grouped_planes(tokens as u32, false);
    let tasks = l.task_capacity as u32;
    let p = fp8::GroupedParams { route, weights: if rank { l.desc_rank } else { l.desc }, input: input_q, input_scales,
        output: middle, cols: s.hidden as u32, activation_quantization: 1, output_shard: 0 };
    if fused {
        out.push(Stage { name: "gate_swiglu", f: f("fp8_grouped_gate_accurate")?, grid: [(s.intermediate / fp8::FUSED_GATE_CHANNELS) as u32 * planes, tasks, 1],
            threads: 128, p: Param::Grouped(p.fused_gate(middle_q, middle_scales, l.counters)) });
    } else {
        let gate_tiles = (2 * s.intermediate / 64) as u32;
        out.push(Stage { name: "gate_up", f: f(entries[0])?, grid: [gate_tiles * planes, tasks, 1], threads: 128, p: Param::Grouped(p) });
        let groups = (128 * s.intermediate / 128).div_ceil(8) as u32;
        out.push(Stage { name: "swiglu_q", f: f("fp8_grouped_gather")?, grid: [groups, tasks, 1], threads: 256,
            p: Param::Grouped(fp8::GroupedParams { input: middle, input_scales: middle_scales, output: middle_q, ..p }) });
    }
    let down = fp8::GroupedParams { input: middle_q, input_scales: middle_scales, output: l.output, ..p };
    if rank {
        // finish_fp8_middle: full-K middle (the world-1 layout), owned half of the outputs.
        out.push(Stage { name: "down", f: f(entries[1])?, grid: [(s.hidden / 2 / 64) as u32 * planes, tasks, 1], threads: 128,
            p: Param::Grouped(fp8::GroupedParams { weights: l.desc, output_shard: 1, input: l.full[0], input_scales: l.full[1], ..down }) });
    } else {
        out.push(Stage { name: "down", f: f(entries[1])?, grid: [(s.hidden / 64) as u32 * planes, tasks, 1], threads: 128, p: Param::Grouped(down) });
    }
    Ok(out)
}

/// Captured graph of `ring` layers between stamps at `times` and `times + 8`; `stamps`
/// adds a %globaltimer stamp after every named stage.
fn capture<'a>(g: &'a Gpu, layers: &mut [Vec<Stage>], timer: H, times: u64, stamps: Option<u64>) -> Result<super::Graph<'a>> {
    super::capture_graph(g, || {
        let mut at = times;
        g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut at)])?;
        let mut slot = 0u64;
        let mut stamp = |g: &Gpu| -> Result<()> {
            if let Some(base) = stamps { let mut at = base + slot * 8; slot += 1; g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut at)])?; }
            Ok(())
        };
        stamp(g)?;
        for stages in layers.iter_mut() {
            // One stamp closes each named stage (the four multi-token route kernels are one).
            for k in 0..stages.len() {
                launch(g, &mut stages[k])?;
                if k + 1 == stages.len() || stages[k + 1].name != stages[k].name { stamp(g)?; }
            }
        }
        at = times + 8;
        g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut at)])
    })
}

/// No token's routing status may be set.
fn status(g: &Gpu, l: &Layer) -> Result<()> {
    if g.read(l.error, l.tokens * 4)?.iter().any(|&b| b != 0) { return Err("routing status set".into()); }
    Ok(())
}

/// Routed rows of every token (ascending expert slots); the shared slot is not written.
fn routed_hash(g: &Gpu, l: &Layer, s: Shape, rank: bool, set: usize) -> Result<(String, String, Vec<u8>, Vec<u8>)> {
    let all = g.read(l.output, l.tokens * (s.top + 1) * s.hidden * 4)?;
    let row = s.hidden * 4;
    let owned = if rank { row / 2 } else { row };
    let mut bytes = Vec::new();
    for r in 0..l.tokens * (s.top + 1) { if r % (s.top + 1) < s.top { bytes.extend_from_slice(&all[r * row..r * row + owned]); } }
    if !bytes.as_chunks::<4>().0.iter().all(|&b| f32::from_le_bytes(b).is_finite()) {
        return Err("routed output nonfinite or unwritten".into());
    }
    // Middle codes and scales of every (token, slot) pair, located as fp8_middle_pack does.
    let codes = g.read(l.act[2], l.rows * s.intermediate)?;
    let scales = g.read(l.act[3], l.rows * s.intermediate / 32)?;
    let words = |at: u64, n: usize| g.read(at, n * 4).map(|b| b.as_chunks::<4>().0.iter().map(|&w| u32::from_le_bytes(w) as usize).collect::<Vec<_>>());
    let (to_row, starts) = (words(l.route.pair_to_row, l.tokens * s.top)?, words(l.route.starts, s.experts)?);
    let mut middle = Vec::new();
    for (pair, &e) in l.pair_experts[set].iter().enumerate() {
        let width = if rank { rank_width(e, s.intermediate) } else { s.intermediate };
        let (start, row) = (starts[e], to_row[pair] - starts[e]);
        let at = start * s.intermediate + row * width;
        middle.extend_from_slice(&codes[at..at + width]);
        let at = (start * (s.intermediate / 128) + row * (width / 128)) * 4;
        middle.extend_from_slice(&scales[at..at + width / 128 * 4]);
    }
    Ok((sha256::digest(&bytes), sha256::digest(&middle), bytes, middle))
}

pub(in super::super) fn run(o: &Options, config: &str) -> Result<()> {
    let s = shape(config)?;
    let mut g = Gpu::new()?;
    let modules = [g.module(&std::fs::read(&o.a).map_err(|e| format!("{}: {e}", o.a))?)?, g.module(&std::fs::read(&o.x).map_err(|e| format!("{}: {e}", o.x))?)?];
    let tm = g.module(TIMER.as_bytes())?;
    let timer = g.function(tm, "kb_stamp")?;
    // Each mode setting is unset or one of its named values; anything else is an error.
    let choice = |name: &str, on: &str, off: &str| -> Result<bool> {
        match std::env::var(name) {
            Err(_) => Ok(false),
            Ok(v) if v == on => Ok(true),
            Ok(v) if v == off => Ok(false),
            Ok(v) => Err(format!("{name}={v:?}: use {on} or {off}")),
        }
    };
    let fused = choice("KBENCH_FP8MOE_X", "fused", "pipeline")?;
    // KBENCH_FP8MOE_A=fused: both arms fused (X self-determinism; pass the X PTX as --a too).
    let fused_a = choice("KBENCH_FP8MOE_A", "fused", "pipeline")?;
    // KBENCH_FP8MOE_XROUTE=small: X arm routes 2..31 tokens with fp8_grouped_single's one-CTA branch.
    let small = choice("KBENCH_FP8MOE_XROUTE", "small", "kernels")?;
    // KBENCH_FP8MOE_XENTRIES=GATE,DOWN: X arm's grouped gate and down entries (same ABI and grid).
    let x_entries = std::env::var("KBENCH_FP8MOE_XENTRIES").unwrap_or_default();
    let x_entries: Vec<&str> = if x_entries.is_empty() { Vec::new() } else { x_entries.split(',').collect() };
    if !x_entries.is_empty() && (x_entries.len() != 2 || x_entries.iter().any(|v| v.is_empty())) {
        return Err(format!("KBENCH_FP8MOE_XENTRIES={x_entries:?}: need exactly GATE,DOWN"));
    }
    let base_entries = ["fp8_grouped_gate_accurate", "fp8_grouped_down"];
    let entries = [base_entries, if x_entries.len() == 2 { [x_entries[0], x_entries[1]] } else { base_entries }];
    let ring = match std::env::var("KBENCH_FP8MOE_RING") {
        Err(_) => 16usize,
        Ok(v) => v.parse().ok().filter(|&n| (1..=256).contains(&n)).ok_or(format!("KBENCH_FP8MOE_RING={v:?}: need 1..=256 route sets"))?,
    };
    // Token counts 1..=31 (the decode/verify path); a leading c routes every token to the
    // same ten experts, so 17+ tokens occupy a second sixteen-row plane.
    let cells: Vec<(usize, bool)> = std::env::var("KBENCH_FP8MOE_M").unwrap_or("1,3,8,c16,c17,c31".into()).split(',').map(|v| {
        let (c, n) = v.strip_prefix('c').map_or((false, v), |n| (true, n));
        n.parse().ok().filter(|&n| (1..=31).contains(&n)).map(|n| (n, c)).ok_or(format!("KBENCH_FP8MOE_M item {v:?}: need 1..=31, optionally prefixed c"))
    }).collect::<Result<_>>()?;
    println!("fp8moe config hidden={} intermediate={} experts={} top={} layers={} ring={ring} ms/arm={}", s.hidden, s.intermediate, s.experts, s.top, s.layers, o.ms);
    for rank in [false, true] {
        for &(tokens, concentrated) in &cells {
            let started = Instant::now();
            let l = layer(&mut g, s, tokens, ring, concentrated)?;
            let stamps = g.alloc(ring * 16 * 8 + 64)?;
            let times = g.alloc(16)?;
            let mut hashes = Vec::new();
            let mut table: [Vec<(&str, f64)>; 2] = [Vec::new(), Vec::new()];
            let mut whole = [0f64; 2];
            for arm in 0..2 {
                let mut layers = (0..ring).map(|r| pipeline(&g, modules[arm], &l, s, r, rank, if arm == 1 { fused } else { fused_a }, entries[arm], arm == 1 && small)).collect::<Result<Vec<_>>>()?;
                // Every route set alone on poisoned outputs; the counters carry over from set to set.
                let mut sets = Vec::with_capacity(ring);
                for (r, stages) in layers.iter_mut().enumerate() {
                    g.upload_at(l.output, &vec![0xff; l.tokens * (s.top + 1) * s.hidden * 4])?;
                    g.upload_at(l.act[2], &vec![0xff; l.rows * s.intermediate])?;
                    g.upload_at(l.act[3], &vec![0xff; l.rows * s.intermediate / 32])?;
                    for st in stages.iter_mut() { launch(&g, st)?; }
                    g.sync()?;
                    status(&g, &l)?;
                    sets.push(routed_hash(&g, &l, s, rank, r)?);
                }
                let plain = capture(&g, &mut layers, timer, times, None)?;
                let timed = capture(&g, &mut layers, timer, times, Some(stamps))?;
                let mut names: Vec<&str> = layers[0].iter().map(|st| st.name).collect();
                names.dedup();
                let per = names.len();
                let (mut samples, mut stage_samples) = (Vec::new(), vec![Vec::new(); per]);
                let deadline = Instant::now();
                while deadline.elapsed().as_millis() < o.ms as u128 || samples.len() < 5 {
                    samples.push(super::sample(&plain, times)?.1 / ring as f64);
                    super::sample(&timed, times)?;
                    let raw = g.read(stamps, (ring * per + 1) * 8)?;
                    let t: Vec<u64> = raw.as_chunks::<8>().0.iter().map(|&b| u64::from_le_bytes(b)).collect();
                    for (j, acc) in stage_samples.iter_mut().enumerate() {
                        acc.push((0..ring).map(|r| (t[r * per + j + 1] - t[r * per + j]) as f64 / 1000.0).sum::<f64>() / ring as f64);
                    }
                }
                whole[arm] = median(&samples);
                // After the timed replays (counters reused throughout), each route set once
                // more as its own graph on poisoned outputs, against its checked pass.
                for (r, stages) in layers.iter_mut().enumerate() {
                    g.upload_at(l.output, &vec![0xff; l.tokens * (s.top + 1) * s.hidden * 4])?;
                    g.upload_at(l.act[2], &vec![0xff; l.rows * s.intermediate])?;
                    g.upload_at(l.act[3], &vec![0xff; l.rows * s.intermediate / 32])?;
                    let one = capture(&g, std::slice::from_mut(stages), timer, times, None)?;
                    super::sample(&one, times)?;
                    status(&g, &l)?;
                    let replayed = routed_hash(&g, &l, s, rank, r)?;
                    if replayed.0 != sets[r].0 || replayed.1 != sets[r].1 { return Err(format!("arm {arm} route set {r}: outputs after graph replay differ from the checked pass")); }
                }
                hashes.push(sets);
                for (j, acc) in stage_samples.iter().enumerate() { table[arm].push((names[j], median(acc))); }
            }
            let mean_distinct = l.distinct.iter().sum::<usize>() as f64 / ring as f64;
            let width_share = if rank { 0.5 } else { 1.0 };
            let floor_bytes = mean_distinct * 3.0 * (s.hidden * s.intermediate) as f64 * width_share;
            let floor_us = floor_bytes / 230e3;
            let tag = format!("{}-{}m{tokens}", if rank { "w2r0" } else { "w1" }, if l.concentrated { "c" } else { "" });
            let same = |k: usize| (0..ring).all(|r| if k == 0 { hashes[0][r].0 == hashes[1][r].0 } else { hashes[0][r].1 == hashes[1][r].1 });
            let (bytes_equal, middle_equal) = (same(0), same(1));
            let last = [&hashes[0][ring - 1], &hashes[1][ring - 1]];
            println!("fp8moe {tag} distinct_experts={mean_distinct:.1} weight_floor_us={floor_us:.1} (230 GB/s) x={} bytes_equal={bytes_equal} middle_equal={middle_equal} (each of {ring} route sets eager, and as its own graph after the timed replays) A={} X={} setup_s={:.1}",
                if fused { "fused" } else { "pipeline" }, &last[0].0[..16], &last[1].0[..16], started.elapsed().as_secs_f64());
            for j in 0..table[0].len().max(table[1].len()) {
                let cell = |arm: usize| table[arm].get(j).map_or(format!("{:<11} {:>8}", "-", ""), |(n, t)| format!("{n:<11} {t:>8.2}"));
                println!("fp8moe {tag} stage A {} us | X {} us (stamped)", cell(0), cell(1));
            }
            println!("fp8moe {tag} layer A {:.2} us X {:.2} us | per token x{} layers: A {:.3} ms X {:.3} ms | floor {:.0}% A {:.0}% X",
                whole[0], whole[1], s.layers, whole[0] * s.layers as f64 / 1000.0, whole[1] * s.layers as f64 / 1000.0,
                floor_us / whole[0] * 100.0, floor_us / whole[1] * 100.0);
            for (what, a, x, unit) in [("routed", &last[0].2, &last[1].2, if rank { s.hidden * 2 } else { s.hidden * 4 }), ("middle", &last[0].3, &last[1].3, s.intermediate + 20)] {
                let diff: Vec<usize> = (0..a.len()).filter(|&j| a[j] != x[j]).collect();
                if !diff.is_empty() {
                    println!("fp8moe {tag} DIFF {what}: {} bytes differ; first {:?}", diff.len(),
                        diff.iter().take(8).map(|&j| (j / unit, j % unit, a[j], x[j])).collect::<Vec<_>>());
                }
            }
            if !(bytes_equal && middle_equal) && !o.allow_diff { return Err(format!("{tag}: routed output bytes differ between arms")); }
            g.clear()?;
        }
    }
    Ok(())
}
