//! fp4_bw routed decode fixtures for benchmark kinds 20/21.
use super::*;
// Captured C8@8K route: (expert, rows); 60 distinct experts, 80 pairs.
const C8_ROUTE: [(u32, u32); 60] = [
    (7, 1), (13, 1), (32, 1), (42, 2), (45, 1), (48, 2), (54, 1), (56, 1), (59, 1), (60, 1),
    (62, 1), (70, 1), (78, 1), (82, 1), (85, 1), (106, 1), (121, 2), (147, 2), (148, 1), (156, 2),
    (168, 2), (198, 2), (206, 1), (207, 1), (222, 1), (235, 1), (240, 1), (252, 4), (257, 2), (258, 1),
    (286, 1), (303, 1), (309, 5), (313, 1), (323, 1), (325, 3), (336, 1), (349, 1), (382, 1), (383, 1),
    (388, 1), (412, 1), (416, 1), (418, 1), (419, 1), (430, 1), (436, 1), (441, 1), (453, 1), (458, 2),
    (462, 1), (463, 1), (464, 1), (465, 2), (469, 1), (488, 2), (491, 1), (496, 1), (506, 1), (511, 1),
];
// C1: one token's ten distinct experts, spread over the same layer's ids.
const C1_ROUTE: [(u32, u32); 10] =
    [(13, 1), (56, 1), (106, 1), (168, 1), (252, 1), (309, 1), (349, 1), (418, 1), (462, 1), (506, 1)];
/// Ten distinct experts per token for c.m tokens (C1 route, or C8's deal; C5 = its
/// first five tokens): (expert per pair, the source route).
pub(super) fn dealt_pairs(c: &Case) -> Result<(Vec<u32>, &'static [(u32, u32)])> {
    let (tokens, pairs) = (c.m, c.m * 10);
    // C2..C7 = the first tokens of the C8 deal (C5: 39 distinct experts, 50 pairs).
    let (route, dealt): (&'static [(u32, u32)], usize) = match tokens {
        1 => (&C1_ROUTE, 1),
        2..=8 => (&C8_ROUTE, 8),
        _ => return Err(format!("{}: routed cases support 1 to 8 tokens", c.name)),
    };
    if route.iter().map(|r| r.1 as usize).sum::<usize>() != dealt * 10 {
        return Err("route does not cover every pair".into());
    }
    // Deal experts (most rows first) round-robin over tokens: ten distinct per token.
    let mut expanded = route.to_vec();
    expanded.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut token_experts = vec![Vec::new(); dealt];
    let mut i = 0;
    for &(e, count) in &expanded {
        for _ in 0..count {
            token_experts[i % dealt].push(e);
            i += 1;
        }
    }
    token_experts.truncate(tokens);
    let mut pair_expert = vec![0u32; pairs];
    for (t, list) in token_experts.iter().enumerate() {
        if list.len() != 10 || (1..10).any(|j| list[..j].contains(&list[j])) {
            return Err("route deal is not ten distinct experts per token".into());
        }
        for (slot, &e) in list.iter().enumerate() {
            pair_expert[t * 10 + slot] = e;
        }
    }
    Ok((pair_expert, route))
}
/// Production decode launch of fp4_bw_gate (kind 20) / fp4_bw_down (kind 21) for
/// c.m chats: compact route tables as compact.rs builds them, task capacity and
/// grid as next_execution.rs launches them, varied finite weights/scales/inputs
/// so a changed per-output sum order changes output bytes.
pub(super) fn routed_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    if c.kind == 22 {
        return layer_fixture(g, c);
    }
    if c.kind == 23 || c.kind == 27 {
        return route_fixture(g, c);
    }
    if c.kind == 29 {
        return route_silu_fixture(g, c);
    }
    if c.kind == 28 {
        // One token's [1, 512] F32 logits (moe_route's generator) ride in rows.weight2.
        let mut v = route_silu_fixture(g, &Case { m: 1, ..c.clone() })?;
        let logits: Vec<u8> = (0..512).flat_map(|i| (((mix(0x5eed_0000_0000 ^ i as u64) >> 11) as f64 / (1u64 << 53) as f64 * 12.0 - 6.0) as f32).to_le_bytes()).collect();
        v.rows.weight2 = g.upload(&logits)?;
        v.stored += 512 * 4;
        return Ok(v);
    }
    if c.kind == 24 {
        return moe_route_fixture(g, c);
    }
    if c.kind == 25 {
        return finish_fixture(g, c);
    }
    if c.kind == 26 {
        return route_fixture(g, &Case { kind: 23, m: 1, name: String::new(), ..c.clone() });
    }
    let down = c.kind == 21;
    let (tokens, n, k) = (c.m, c.n, c.k);
    let pairs = tokens * 10;
    let (pair_expert, route) = dealt_pairs(c)?;
    let mut counts = vec![0u32; 512];
    for &e in &pair_expert {
        counts[e as usize] += 1;
    }
    let mut starts = vec![0u32; 513];
    let mut tasks = vec![0u32];
    let mut sorted = Vec::with_capacity(pairs);
    let mut pair_to_row = vec![0u32; pairs];
    for e in 0..512usize {
        starts[e] = sorted.len() as u32;
        for first in (0..counts[e]).step_by(128) {
            tasks.extend([e as u32, first]);
            tasks[0] += 1;
        }
        for pair in 0..pairs {
            if pair_expert[pair] as usize != e {
                continue;
            }
            sorted.push(pair as u32);
            let t = pair / 10;
            let rank = (0..10).filter(|&j| pair_expert[t * 10 + j] < e as u32).count();
            pair_to_row[pair] = (t * 11 + rank) as u32;
        }
    }
    starts[512] = sorted.len() as u32;
    let capacity = pairs.div_ceil(128) + pairs.min(512);
    tasks.resize(1 + 2 * capacity, 0);
    let mats = if down { 1 } else { 2 };
    let active: Vec<u32> = route.iter().map(|r| r.0).filter(|&e| counts[e as usize] > 0).collect();
    let matrix_bytes = super::super::floor::nvfp4(n, k);
    // KBENCH_MISALIGN = bytes (multiple of 16) added to every matrix base: production's
    // weight arena aligns NVFP4 tensors to 16 bytes only (engine flash/weights.rs).
    let mis = std::env::var("KBENCH_MISALIGN").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) / 16 * 16 % 256;
    let slot = (matrix_bytes + mis).div_ceil(256) * 256;
    let per_copy = active.len() * mats * slot;
    // KBENCH_ROTATE_MIB widens the span (TLB-cold probe of the 48-layer expert set).
    let copies = rotate_bytes().div_ceil(per_copy).max(2);
    // One host image per copy; data depends on (expert, matrix, offset) only.
    let mut image = vec![0u8; per_copy];
    for (a, &e) in active.iter().enumerate() {
        for m in 0..mats {
            let which = if down { 2 } else { m as u64 };
            let base = (a * mats + m) * slot + mis;
            let seed = (u64::from(e) << 8 | which) << 40;
            for (j, chunk) in image[base..base + n * k / 2].chunks_mut(8).enumerate() {
                chunk.copy_from_slice(&mix(seed ^ j as u64).to_le_bytes()[..chunk.len()]);
            }
            for (j, v) in image[base + n * k / 2..base + n * k / 2 + n * k / 16].iter_mut().enumerate() {
                let r = mix(seed ^ 0x8000_0000 ^ j as u64);
                *v = (((5 + r % 4) << 3) | (r >> 8) % 8) as u8;
            }
            let global = (1.0 + (e % 7) as f32 / 8.0) * if which == 1 { 0.0625 } else { 0.125 };
            image[base + n * k / 2 + n * k / 16..base + matrix_bytes].copy_from_slice(&global.to_le_bytes());
        }
    }
    let slab = g.alloc(copies * per_copy)?;
    eprintln!("weights case={} slab={} experts={} copies={}", c.name, slab, active.len(), copies);
    let mut ring = Vec::new();
    for copy in 0..copies {
        let base = slab + (copy * per_copy) as u64;
        unsafe {
            check((g.d.upload)(base, image.as_ptr().cast(), per_copy))?;
        }
        let mut descriptors = vec![0u8; 512 * 80];
        for (a, &e) in active.iter().enumerate() {
            let matrix = |m: usize| {
                let weights = base + ((a * mats + m) * slot + mis) as u64;
                let scales = weights + (n * k / 2) as u64;
                fp4::Matrix { weights, scales, global_scale: scales + (n * k / 16) as u64 }
            };
            let ex = fp4::Expert {
                gate: if down { fp4::Matrix::default() } else { matrix(0) },
                up: if down { fp4::Matrix::default() } else { matrix(1) },
                down: if down { matrix(0) } else { fp4::Matrix::default() },
                width: 640,
                reserved: 0,
            };
            let mut d = Vec::new();
            for v in [
                ex.gate.weights,
                ex.gate.scales,
                ex.gate.global_scale,
                ex.up.weights,
                ex.up.scales,
                ex.up.global_scale,
                ex.down.weights,
                ex.down.scales,
                ex.down.global_scale,
            ] {
                d.extend(v.to_le_bytes());
            }
            d.extend(ex.width.to_le_bytes());
            d.extend(0u32.to_le_bytes());
            descriptors[e as usize * 80..e as usize * 80 + 80].copy_from_slice(&d);
        }
        ring.push(g.upload(&descriptors)?);
    }
    // Gate reads F32 token rows; down reads the packed BF16 compact middle.
    let input_bytes = if down { pairs * k * 2 } else { tokens * k * 4 };
    let mut host = vec![0u8; input_bytes];
    for (j, chunk) in host.chunks_mut(if down { 2 } else { 4 }).enumerate() {
        let r = mix(0x1234_5678_0000_0000 ^ j as u64);
        if down {
            let bits = ((r & 1) << 15 | (123 + (r >> 1) % 6) << 7 | (r >> 8) & 0x7f) as u16;
            chunk.copy_from_slice(&bits.to_le_bytes());
        } else {
            let bits = ((r & 1) << 31 | (125 + (r >> 1) % 4) << 23 | (r >> 8) & 0x7f_ffff) as u32;
            chunk.copy_from_slice(&bits.to_le_bytes());
        }
    }
    let input = g.upload(&host)?;
    let out_bytes = if down { tokens * 11 * 2560 * 4 } else { pairs * 640 * 2 };
    let out = g.alloc(out_bytes)?;
    let written = if down {
        (0..tokens).flat_map(|t| (0..10).map(move |r| ((t * 11 + r) * 2560 * 4, 1280 * 4))).collect()
    } else {
        Vec::new()
    };
    let error = g.words(&vec![0u32; tokens])?;
    let route = fp8::GroupedRouteParams {
        error,
        pairs: 0,
        counts: g.words(&counts)?,
        starts: g.words(&starts)?,
        pair_to_row: g.words(&pair_to_row)?,
        token_pairs: 0,
        token_count: 0,
        tasks: g.words(&tasks)?,
        sorted_row: g.words(&sorted)?,
        output_row: 0,
        experts: 512,
        tokens: tokens as u32,
        pair_count: pairs as u32,
        row_capacity: pairs as u32,
    };
    let rows = params::Bf16Rows {
        weight: 0,
        output: out,
        rows: n as u32,
        stride: n as u32,
        weight2: 0,
        output2: 0,
        rows2: 0,
        stride2: 0,
        input,
        cols: k as u32,
        tokens: tokens as u32,
        input_bf16: 1,
        hc_silu: 0,
        input_plane_words: 0,
    };
    let unique = active.len() * mats * matrix_bytes;
    let out_live = if down { pairs * 1280 * 4 } else { out_bytes };
    Ok(Fixture {
        rows,
        group: Some(fp8::GroupedParams {
            route,
            weights: ring[0],
            input,
            input_scales: 0,
            output: out,
            cols: 2560,
            activation_quantization: 0,
            output_shard: 0,
        }),
        ring,
        out,
        out_bytes,
        aux: 0,
        aux_bytes: 0,
        // planes = count.div_ceil(16).min(8), as next_execution.rs decode.
        grid: [if down { 160 } else { 40 }, capacity as u32, tokens.div_ceil(16).min(8) as u32],
        threads: 64,
        shared: 0,
        stored: unique + input_bytes + out_live,
        flops: 2 * pairs * n * k * mats,
        chunks: 1,
        down_slots: 0,
        raw: 0,
        pack: null_mut(), norms: [null_mut(); 2],
        weight_bytes: unique,
        written,
        qsa: None,
        gather: None,
        ring_x: vec![],
        scales_x: vec![],
    })
}

/// fp4_bw layer (kind 22): gate (kind 20) then down (kind 21) per ring copy, as
/// next_execution.rs launches them; down reads the gate's packed BF16 output.
/// Gate state rides in the down fixture: ring_x = gate descriptor tables, raw =
/// gate F32 input, aux = gate output (= down input).
fn layer_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let gate = routed_fixture(g, &Case { kind: 20, n: 640, k: 2560, ..c.clone() })?;
    let mut v = routed_fixture(g, &Case { kind: 21, n: 2560, k: 640, ..c.clone() })?;
    let copies = gate.ring.len().min(v.ring.len());
    v.ring.truncate(copies);
    v.ring_x = gate.ring[..copies].to_vec();
    // Production descriptors carry all three matrices: give each gate table copy the
    // down Matrix (bytes 48..72 of the 80-byte Expert) of the same copy.
    for (&gt, &dt) in v.ring_x.iter().zip(&v.ring) {
        let (mut gd, dd) = (g.read(gt, 512 * 80)?, g.read(dt, 512 * 80)?);
        for e in 0..512 {
            gd[e * 80 + 48..e * 80 + 72].copy_from_slice(&dd[e * 80 + 48..e * 80 + 72]);
        }
        g.upload_at(gt, &gd)?;
    }
    v.raw = gate.group.as_ref().ok_or("gate fixture")?.input;
    v.aux = gate.out;
    v.aux_bytes = gate.out_bytes;
    v.stored += gate.stored;
    v.flops += gate.flops;
    v.weight_bytes += gate.weight_bytes;
    Ok(v)
}
/// KBENCH_PDL=1: the X arm launches down with programmatic stream serialization.
pub(super) fn launch_layer(g: &Gpu, down: H, gate: H, v: &mut Fixture, i: usize, arm: usize) -> Result<()> {
    let p = v.group.as_mut().ok_or("layer fixture")?;
    let mut gp = fp8::GroupedParams { weights: v.ring_x[i], input: v.raw, output: v.aux, ..*p };
    g.launch(gate, [40, v.grid[1], v.grid[2]], 64, 0, &mut [ptr(&mut gp)])?;
    p.weights = v.ring[i];
    p.input = v.aux;
    let mut grid = v.grid;
    if arm == 1 {
        if let Some(x) = std::env::var("KBENCH_XGRIDX").ok().and_then(|x| x.parse().ok()) { grid[0] = x; }
    }
    if arm == 1 && std::env::var("KBENCH_PDL").is_ok_and(|x| x == "1") {
        g.launch_pdl(down, grid, v.threads, 0, &mut [ptr(p)])
    } else {
        g.launch(down, grid, v.threads, 0, &mut [ptr(p)])
    }
}
/// Decode route tables (kind 23): flash_next_compact_route on the dealt route's
/// (token, expert, weight) triples, one CTA as next_execution.rs launches it below
/// 64 tokens; the hashed output is the whole bounds buffer (counts, starts,
/// pair_to_row, sorted_row, token_pairs, tasks). "-bad" repeats an expert in token
/// 2 so the failed-token path runs too.
fn route_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (tokens, pairs) = (c.m, c.m * 10);
    let mut pair_expert = if c.name.starts_with("proute-") { prefill_experts(c) } else { dealt_pairs(c)?.0 };
    if c.name.ends_with("-bad") {
        pair_expert[25] = pair_expert[23];
    }
    let mut host = Vec::with_capacity(pairs * 3);
    for (pair, &e) in pair_expert.iter().enumerate() {
        host.extend([(pair / 10) as f32, e as f32, 0.05 + (pair % 10) as f32 * 0.01]);
    }
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let task_words = 2 * (pairs.div_ceil(128) + pairs.min(512));
    let words = 512 + 513 + 3 * pairs + 1 + task_words;
    // route_ids (kind 27): the pairs table follows the tables in `out` (written by A's
    // flash_moe_assignments or by X's fused entry, poisoned before each launch, hashed).
    let ids_route = c.kind == 27;
    let out = g.alloc((words + if ids_route { 3 * pairs } else { 0 }) * 4)?;
    let input = if ids_route { out + (words * 4) as u64 } else { g.upload(&bytes)? };
    let (ids, weights) = if ids_route {
        let w: Vec<u8> = host.chunks(3).flat_map(|t| t[2].to_le_bytes()).collect();
        (g.words(&pair_expert.iter().map(|&e| e as u32).collect::<Vec<_>>())?, g.upload(&w)?)
    } else { (0, 0) };
    let base = out;
    let route = fp8::GroupedRouteParams {
        error: g.words(&vec![0u32; tokens])?,
        pairs: input,
        counts: base,
        starts: base + 512 * 4,
        pair_to_row: base + ((512 + 513) * 4) as u64,
        sorted_row: base + ((512 + 513 + pairs) * 4) as u64,
        token_pairs: base + ((512 + 513 + 2 * pairs) * 4) as u64,
        tasks: base + ((512 + 513 + 3 * pairs) * 4) as u64,
        token_count: 0,
        output_row: 0,
        experts: 512,
        tokens: tokens as u32,
        pair_count: pairs as u32,
        row_capacity: pairs as u32,
    };
    // route_ids: rows.weight / rows.output carry the ids and weights the assignments read.
    let rows = params::Bf16Rows { weight: ids, output: weights, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0,
        input: if c.name.starts_with("proute-") && std::env::var("KBENCH_ROUTE_PREFILL_SEGMENTS").is_ok_and(|v|v=="1") {g.fill((512*pairs.div_ceil(10240)+pairs)*4,255)?}else{0}, cols: 0, tokens: tokens as u32, input_bf16: u32::from(c.name.starts_with("proute-")), hc_silu: 0, input_plane_words: 0 };
    Ok(Fixture {
        rows,
        group: Some(fp8::GroupedParams { route, weights: 0, input: 0, input_scales: 0, output: 0, cols: 2560, activation_quantization: 0, output_shard: 0 }),
        // 256 launches per sample: the route is L2-resident in production as here.
        ring: vec![0; if c.name.starts_with("proute-") { 2 } else { 256 }],
        out,
        out_bytes: (words + if ids_route { 3 * pairs } else { 0 }) * 4,
        // Arm X's own error flags: each arm must flag the bad token itself.
        aux: g.words(&vec![0u32; tokens])?,
        aux_bytes: 0,
        grid: [1, 1, 1],
        threads: 256,
        shared: 0,
        stored: bytes.len() + words * 4,
        flops: 0,
        chunks: 1,
        down_slots: 0,
        raw: 0,
        pack: null_mut(),
        norms: [null_mut(); 2],
        weight_bytes: 0,
        written: Vec::new(),
        qsa: None,
        gather: None,
        ring_x: vec![],
        scales_x: vec![],
    })
}
/// Decode route + shared-expert silu (kind 29): the kind-27 route of c.m tokens plus
/// the shared expert's [c.m][640] F32 gate (varied, both signs; pristine copy in raw, restored
/// into `out` after the route tables and pairs by kb_copy before each launch, both arms) and
/// up (rows.input). A = flash_moe_silu (grid (n / 128) x 128 as launched) + the A PTX's
/// flash_next_compact_route_ids; X = flash_next_compact_route_silu (1 + n / 256 CTAs).
/// Hash: route tables + pairs + silu'd gate.
fn route_silu_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let n = c.m * 640;
    let f32s = |seed: u64| -> Vec<u8> {
        (0..n).flat_map(|i| (((mix(seed ^ i as u64) >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32).to_le_bytes()).collect()
    };
    let mut v = route_fixture(g, &Case { kind: 27, ..c.clone() })?;
    let gate_at = v.out_bytes;
    // route_pack_mask: the shared down's dense hi/lo planes (one 16-row tile row of
    // [m][640]: 2 x 5120 words) follow the gate at a 256-byte boundary, hashed with `out`.
    let planes_at = if c.name.contains("pack") { (gate_at + n * 4).next_multiple_of(256) } else { 0 };
    let out_bytes = if planes_at != 0 { planes_at + 2 * 5120 * 4 } else { gate_at + n * 4 };
    let out = g.alloc(out_bytes)?;
    // Re-point the route tables and pairs into the larger `out` (same offsets).
    let old = v.out;
    let group = v.group.as_mut().ok_or("route fixture")?;
    for a in [&mut group.route.counts, &mut group.route.starts, &mut group.route.pair_to_row, &mut group.route.sorted_row,
        &mut group.route.token_pairs, &mut group.route.tasks, &mut group.route.pairs] {
        *a = *a - old + out;
    }
    v.out = out;
    v.out_bytes = out_bytes;
    v.rows.stride2 = planes_at as u32;
    v.raw = g.upload(&f32s(0x9a7e_0000_0000))?;
    v.rows.input = g.upload(&f32s(0x0b00_0000_0000))?;
    v.rows.rows = gate_at as u32;
    // The plain C1 case uses an empty mask and hashes only the latency pack's live lanes.
    v.rows.hc_silu = u32::from(c.name.ends_with("-plain"));
    if v.rows.hc_silu != 0 {
        let lanes = params::flash_latency_lanes(c.m);
        v.written.push((0, planes_at));
        for plane in 0..2 { for kt in 0..640 / 16 {
            v.written.push((planes_at + (plane * 5120 + kt * 128) * 4, lanes * 16));
        } }
    }
    let m = g.module(super::prompt::copy_ptx().as_bytes()).map_err(|e| format!("copy PTX: {e}"))?;
    v.norms = [null_mut(), g.function(m, "kb_copy")?];
    v.stored += 3 * n * 4;
    if c.name.contains("mask") {
        // rows.output2 = [pristine ids | pristine weights | pooled slot id per row] (0 = no
        // live slot: the mask drops the row's ten pairs); kb_copy restores ids and weights
        // before each launch (both arms), so X masks fresh pairs as A does.
        // ids / weights move to 16-byte padded buffers (kb_copy's words).
        let (pairs, live) = (c.m * 10, if c.name.ends_with("c5") { 5 } else { c.m });
        let q = pairs.div_ceil(4) * 4;
        let mut words = Vec::with_capacity(2 * q + c.m);
        for src in [v.rows.weight, v.rows.output] {
            let b = g.read(src, pairs * 4)?;
            words.extend(b.chunks(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])));
            words.resize(words.len() + q - pairs, 0);
        }
        (v.rows.weight, v.rows.output) = (g.words(&words[..q])?, g.words(&words[q..])?);
        words.extend((0..c.m).map(|r| if r < live { r as u32 + 1 } else { 0 }));
        v.rows.output2 = g.words(&words)?;
        v.stored += 4 * pairs * 4 + c.m * 4;
    }
    Ok(v)
}
/// next_math::SlotStep's kernel ABI (flash_next_route_mask and the _mask route entries).
#[repr(C)]
struct SlotStep { slots: u64, table: u64, layer: u32, rows: u32 }
pub(super) fn launch_route(g: &Gpu, f: H, merge: H, v: &mut Fixture, arm: usize) -> Result<()> {
    let mut route = v.group.as_ref().ok_or("route fixture")?.route;
    if arm == 1 {
        route.error = v.aux;
    }
    if v.rows.input_bf16 == 1 {
        for mut phase in 0u32..4 {
            let grid = match phase { 0 => route.tokens.div_ceil(256), 2 => 1, _ => route.experts };
            let threads=256;
            if (arm==1 || std::env::var("KBENCH_ROUTE_BASE_SEGMENTS").is_ok_and(|v|v=="1")) && std::env::var("KBENCH_ROUTE_PREFILL_SEGMENTS").is_ok_and(|v|v=="1") {
                let mut scratch=v.rows.input;
                g.launch(f,[grid,1,1],256,0,&mut [ptr(&mut route),ptr(&mut phase),ptr(&mut scratch)])?;
            } else {
                g.launch(f, [grid, 1, 1], threads, 0, &mut [ptr(&mut route), ptr(&mut phase)])?;
            }
        }
        return Ok(());
    }
    if !v.norms[1].is_null() {
        // route_silu: restore the gate, then A = main's silu + compact_route_ids, X = the fused entry.
        let n = route.tokens * 640;
        let (mut gate, mut src, mut words) = (v.out + v.rows.rows as u64, v.raw, n / 4);
        g.launch(v.norms[1], [words.div_ceil(256), 1, 1], 256, 0, &mut [ptr(&mut gate), ptr(&mut src), ptr(&mut words)])?;
        let (mut ids, mut weights, mut up, mut count) = (v.rows.weight, v.rows.output, v.rows.input, n);
        if v.rows.output2 != 0 {
            // route_silu_mask / route_full_silu_mask: A = [flash_moe_route +]
            // flash_next_route_mask + flash_next_compact_route_silu, X = the _mask entry.
            let (pairs, q) = (route.tokens * 10, (route.tokens * 10).div_ceil(4) as u64 * 16);
            let slots = v.rows.output2 + 2 * q;
            let plain = v.rows.hc_silu != 0;
            if !plain { for (mut dst, mut src) in [(ids, v.rows.output2), (weights, v.rows.output2 + q)] {
                let mut n = (q / 16) as u32;
                g.launch(v.norms[1], [n.div_ceil(256), 1, 1], 256, 0, &mut [ptr(&mut dst), ptr(&mut src), ptr(&mut n)])?;
            } }
            let mut step = if plain { SlotStep { slots: 0, table: 0, layer: 0, rows: 0 } }
                else { SlotStep { slots, table: slots, layer: 0, rows: route.tokens } };
            let mut r = params::FlashMoeRoute { logits: v.rows.weight2, ids, weights, tokens: 1, norm_topk: 1 };
            // route_pack_mask: X runs 20 silu_pack CTAs. A runs c.kernel, then flash_bf16_input.
            // Plain C1 packs the four consumed lanes. Masked cases pack all 32 lanes.
            let (pack, mut planes) = (v.rows.stride2 != 0, v.out + v.rows.stride2 as u64);
            let grid = [1 + if pack && arm == 1 { 20 } else { n.div_ceil(256) }, 1, 1];
            if arm == 1 || pack {
                let mut args = if v.rows.weight2 != 0 {
                    let mut args = vec![ptr(&mut r), ptr(&mut route), ptr(&mut gate), ptr(&mut up), ptr(&mut count)];
                    if !plain || arm == 1 { args.push(ptr(&mut step)); }
                    args
                } else {
                    vec![ptr(&mut route), ptr(&mut ids), ptr(&mut weights), ptr(&mut gate), ptr(&mut up), ptr(&mut count), ptr(&mut step)]
                };
                if arm == 1 && pack {
                    args.push(ptr(&mut planes));
                }
                g.launch(f, grid, v.threads, 0, &mut args)?;
                if arm == 1 {
                    return Ok(());
                }
                let lanes = if plain { params::flash_latency_lanes(route.tokens as usize) as u32 } else { 32 };
                let mut p = params::FlashBf16Input { input: gate, output: planes, tokens: route.tokens, cols: 640, hc_silu: 0, lanes };
                return g.launch(v.pack, [(5120 / 128 * lanes).div_ceil(128), 1, 1], 128, 0, &mut [ptr(&mut p)]);
            }
            if v.rows.weight2 != 0 {
                g.launch(merge, [1, 1, 1], 128, 0, &mut [ptr(&mut r)])?;
            }
            g.launch(v.norms[0], [pairs.div_ceil(128), 1, 1], 128, 0, &mut [ptr(&mut ids), ptr(&mut weights), ptr(&mut step)])?;
            return g.launch(f, grid, v.threads, 0,
                &mut [ptr(&mut route), ptr(&mut ids), ptr(&mut weights), ptr(&mut gate), ptr(&mut up), ptr(&mut count)]);
        }
        if v.rows.weight2 != 0 {
            // route_full_silu (kind 28): A = flash_moe_route + the shared-expert
            // flash_next_compact_route_silu (the A PTX's C1 path), X = flash_next_route_compact1_silu.
            let mut r = params::FlashMoeRoute { logits: v.rows.weight2, ids, weights, tokens: 1, norm_topk: 1 };
            let grid = [1 + n.div_ceil(256), 1, 1];
            if arm == 0 {
                g.launch(merge, [1, 1, 1], 128, 0, &mut [ptr(&mut r)])?;
                return g.launch(f, grid, v.threads, 0,
                    &mut [ptr(&mut route), ptr(&mut ids), ptr(&mut weights), ptr(&mut gate), ptr(&mut up), ptr(&mut count)]);
            }
            return g.launch(f, grid, v.threads, 0, &mut [ptr(&mut r), ptr(&mut route), ptr(&mut gate), ptr(&mut up), ptr(&mut count)]);
        }
        if v.rows.stride2 != 0 {
            // route_pack: A = flash_next_compact_route_silu (c.kernel, the route-mask entry's verify /
            // draft rows) + main's flash_bf16_input at lanes 32; X = flash_next_compact_route_pack.
            let mut planes = v.out + v.rows.stride2 as u64;
            let mut args = vec![ptr(&mut route), ptr(&mut ids), ptr(&mut weights), ptr(&mut gate), ptr(&mut up), ptr(&mut count)];
            if arm == 1 {
                args.push(ptr(&mut planes));
            }
            g.launch(f, [1 + if arm == 1 { 20 } else { n.div_ceil(256) }, 1, 1], v.threads, 0, &mut args)?;
            if arm == 1 {
                return Ok(());
            }
            let mut p = params::FlashBf16Input { input: gate, output: planes, tokens: route.tokens, cols: 640, hc_silu: 0, lanes: 32 };
            return g.launch(v.pack, [(5120 / 128 * 32u32).div_ceil(128), 1, 1], 128, 0, &mut [ptr(&mut p)]);
        }
        if arm == 0 {
            g.launch(merge, [n.div_ceil(128), 1, 1], 128, 0, &mut [ptr(&mut gate), ptr(&mut up), ptr(&mut count)])?;
            return g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut route), ptr(&mut ids), ptr(&mut weights)]);
        }
        return g.launch(f, [1 + n.div_ceil(256), 1, 1], v.threads, 0,
            &mut [ptr(&mut route), ptr(&mut ids), ptr(&mut weights), ptr(&mut gate), ptr(&mut up), ptr(&mut count)]);
    }
    if v.rows.weight != 0 {
        // route_ids: A = main's flash_moe_assignments (block 128 as launched) then the
        // route; X = flash_next_compact_route_ids, which writes the pairs itself.
        let (mut ids, mut weights, mut pairs, mut tokens) = (v.rows.weight, v.rows.output, route.pairs, route.tokens);
        if arm == 0 {
            g.launch(merge, [route.pair_count.div_ceil(128), 1, 1], 128, 0, &mut [ptr(&mut ids), ptr(&mut weights), ptr(&mut pairs), ptr(&mut tokens)])?;
            return g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut route)]);
        }
        return g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut route), ptr(&mut ids), ptr(&mut weights)]);
    }
    g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut route)])
}
/// Router top-10 (kind 24): flash_moe_route on [tokens, 512] F32 logits (varied,
/// tie-free), one CTA per token; output = ids then weights. KBENCH_ROUTE_THREADS
/// sets X's block size (A launches 128 threads as next_execution.rs does).
fn moe_route_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    if c.name.contains("silu") {
        return moe_silu_fixture(g, c);
    }
    let tokens = c.m;
    let mut host = Vec::with_capacity(tokens * 512 * 4);
    for i in 0..tokens * 512 {
        let r = mix(0x5eed_0000_0000 ^ i as u64);
        let v = ((r >> 11) as f64 / (1u64 << 53) as f64 * 12.0 - 6.0) as f32;
        host.extend(v.to_le_bytes());
    }
    let logits = g.upload(&host)?;
    let out = g.alloc(tokens * 10 * 8)?;
    let mut v = route_fixture(g, &Case { kind: 23, m: tokens, name: String::new(), ..c.clone() })?;
    v.rows.weight = logits;
    v.out = out;
    v.out_bytes = tokens * 10 * 8;
    v.grid = [tokens as u32, 1, 1];
    v.threads = 128;
    v.stored = tokens * 512 * 4 + tokens * 80;
    Ok(v)
}
/// Prompt shared-expert silu (kind 24 "silu" names): [c.m][640] F32 gate (varied,
/// both signs, every 97th value beyond +-87 so exp_split's fallback runs; pristine copy in raw,
/// restored into `out` by kb_copy before each launch, both arms) and up (rows.input); grid
/// (n / 128) x 128 as gdn::launch. A = flash_moe_silu, X = flash_moe_silu_fast. Hash: the gate.
fn moe_silu_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let n = c.m * 640;
    let f32s = |seed: u64, wide: bool| -> Vec<u8> {
        (0..n).flat_map(|i| { let u = (mix(seed ^ i as u64) >> 11) as f64 / (1u64 << 53) as f64;
            ((if wide && i % 97 == 0 { u * 240.0 - 120.0 } else { u * 16.0 - 8.0 }) as f32).to_le_bytes() }).collect()
    };
    let mut v = moe_route_fixture(g, &Case { name: String::new(), m: 1, ..c.clone() })?;
    v.raw = g.upload(&f32s(0x9a7e_0000_0000, true))?;
    v.rows.input = g.upload(&f32s(0x0b00_0000_0000, false))?;
    v.out = g.alloc(n * 4)?;
    v.out_bytes = n * 4;
    let m = g.module(super::prompt::copy_ptx().as_bytes()).map_err(|e| format!("copy PTX: {e}"))?;
    v.norms = [null_mut(), g.function(m, "kb_copy")?];
    v.grid = [n.div_ceil(128) as u32, 1, 1];
    v.threads = 128;
    v.stored = 3 * n * 4;
    Ok(v)
}
pub(super) fn launch_moe_route(g: &Gpu, f: H, v: &mut Fixture, arm: usize) -> Result<()> {
    if !v.norms[1].is_null() {
        let (mut gate, mut src, mut words, mut up) = (v.out, v.raw, (v.out_bytes / 16) as u32, v.rows.input);
        g.launch(v.norms[1], [words.div_ceil(256), 1, 1], 256, 0, &mut [ptr(&mut gate), ptr(&mut src), ptr(&mut words)])?;
        let mut n = (v.out_bytes / 4) as u32;
        return g.launch(f, v.grid, 128, 0, &mut [ptr(&mut gate), ptr(&mut up), ptr(&mut n)]);
    }
    let tokens = v.grid[0] as usize;
    let mut p = params::FlashMoeRoute { logits: v.rows.weight, ids: v.out, weights: v.out + (tokens * 40) as u64, tokens: tokens as u32, norm_topk: 1 };
    let threads = if arm == 1 { std::env::var("KBENCH_ROUTE_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(128) } else { 128 };
    g.launch(f, v.grid, threads, 0, &mut [ptr(&mut p)])
}
/// Expert combine (kind 25): flash_next_k16_finish over [tokens][11][2560] partial
/// slots (ten routed BF16 rows in FP32-stride slots, then the FP32 shared row), the
/// dealt route's ids and varied weights, grid (10, tokens) x 256 as launched; output
/// = the BF16-rounded FP32 rows.
fn finish_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let tokens = c.m;
    let (pair_expert, _) = dealt_pairs(c)?;
    let weights: Vec<f32> = (0..tokens * 10).map(|i| 0.02 + (mix(0xf1_0000 ^ i as u64) % 1000) as f32 * 1e-4).collect();
    let mut partials = vec![0u8; tokens * 11 * 2560 * 4];
    for (j, w) in partials.chunks_mut(4).enumerate() {
        let r = mix(0xf2_0000_0000 ^ j as u64);
        let (slot, col) = (j / 2560 % 11, j % 2560);
        let v = if slot < 10 {
            // Routed slot: BF16 values in the first 2560 u16 of the FP32-stride slot.
            let lo = ((r & 1) << 15 | (120 + (r >> 1) % 8) << 7 | (r >> 8) & 0x7f) as u32;
            let hi = (((r >> 16) & 1) << 15 | (120 + (r >> 17) % 8) << 7 | (r >> 24) & 0x7f) as u32;
            if col < 1280 { lo | hi << 16 } else { 0 }
        } else {
            (((r & 1) << 31) | ((120 + (r >> 1) % 8) << 23) | ((r >> 8) & 0x7f_ffff)) as u32
        };
        w.copy_from_slice(&v.to_le_bytes());
    }
    let gates: Vec<f32> = (0..tokens).map(|t| -1.0 + t as f32 * 0.25).collect();
    let as_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut v = route_fixture(g, &Case { kind: 23, name: String::new(), ..c.clone() })?;
    let out = g.alloc(tokens * 2560 * 4)?;
    let group = v.group.as_mut().ok_or("finish fixture")?;
    // weights rides in rows.input, partials in raw, shared gates in aux; ids in group.weights.
    group.weights = g.words(&pair_expert)?;
    v.rows.input = g.upload(&as_bytes(&weights))?;
    v.raw = g.upload(&partials)?;
    v.aux = g.upload(&as_bytes(&gates))?;
    v.out = out;
    v.out_bytes = tokens * 2560 * 4;
    v.grid = [10, tokens as u32, 1];
    v.stored = partials.len() + tokens * 2560 * 4;
    Ok(v)
}
pub(super) fn launch_finish(g: &Gpu, f: H, v: &mut Fixture) -> Result<()> {
    let group = v.group.as_ref().ok_or("finish fixture")?;
    let mut p = fp8::MoeFinishParams { error: group.route.error, ids: group.weights, weights: v.rows.input, partials: v.raw,
        shared_gate: v.aux, output: v.out, activation_quantization: 1, column_start: 0, columns: 2560 };
    g.launch(f, v.grid, 256, 0, &mut [ptr(&mut p)])
}
/// KBENCH_ICFLUSH=1: before every routed / layer launch, run kb_flush, ~12k distinct
/// straight-line integer instructions (no loops, no barriers, no loads) on 96 CTAs,
/// so each SM starts the timed kernel with cold instruction caches as in production (~35 other kernels per layer). Case icflush times the flush
/// alone: penalty = (kernel + flush) - kernel - flush.
static FLUSH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FLUSH_BUF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn flush_ptx() -> String {
    // Four interleaved integer mad chains seeded from %tid (nothing for ptxas to fold),
    // 12,000 straight-line instructions (~190 KB of SASS), a few us of issue time.
    let mut s = String::from(".version 8.8\n.target sm_121\n.address_size 64\n.visible .entry kb_flush(.param .u64 buf) {\n.reg .b64 p;\n.reg .b32 a0, a1, a2, a3;\n.reg .pred q;\nld.param.u64 p, [buf];\nmov.u32 a0, %tid.x;\nadd.u32 a1, a0, 1;\nadd.u32 a2, a0, 2;\nadd.u32 a3, a0, 3;\n");
    for i in 0..12000u32 {
        let j = i % 4;
        s += &format!("mad.lo.u32 a{j}, a{j}, {}, a{};\n", 2 * (i.wrapping_mul(2_654_435_761) >> 8) + 1, (j + 1) % 4);
    }
    s += "xor.b32 a0, a0, a1;\nxor.b32 a2, a2, a3;\nxor.b32 a0, a0, a2;\nsetp.eq.u32 q, a0, 2654435769;\n@q st.global.u32 [p], a0;\nret;\n}\n";
    s
}
pub(super) fn flush_init(g: &mut Gpu, needed: bool) -> Result<()> {
    if !needed {
        return Ok(());
    }
    let m = g.module(flush_ptx().as_bytes()).map_err(|e| format!("flush PTX: {e}"))?;
    let f = g.function(m, "kb_flush")?;
    FLUSH.store(f as usize, std::sync::atomic::Ordering::Relaxed);
    // Outside g.allocs: Gpu::clear frees those between cases.
    let mut buf = 0u64;
    unsafe { check((g.d.alloc)(&mut buf, 16384))? };
    FLUSH_BUF.store(buf, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub(super) fn flush_on() -> bool {
    std::env::var("KBENCH_ICFLUSH").is_ok_and(|v| v == "1")
}
pub(super) fn icflush(g: &Gpu) -> Result<()> {
    let f = FLUSH.load(std::sync::atomic::Ordering::Relaxed) as H;
    if f.is_null() {
        return Err("flush kernel not loaded".into());
    }
    let mut buf = FLUSH_BUF.load(std::sync::atomic::Ordering::Relaxed);
    g.launch(f, [96, 1, 1], 128, 0, &mut [ptr(&mut buf)])
}

/// Prompt route fixture with ten distinct experts per token; hot keeps one union.
fn prefill_experts(c: &Case) -> Vec<u32> {
    (0..c.m * 10).map(|pair| {
        let token = if c.name.ends_with("-hot") { 0 } else { pair / 10 };
        ((token * 17 + (pair % 10) * 53) % 512) as u32
    }).collect()
}
/// Independent CPU stable sort and prefix oracle for every route output word,
/// including unwritten capacity poison, plus per-arm sticky error status.
pub(super) fn prefill_hash(g: &Gpu, c: &Case, v: &Fixture, arm: usize) -> Result<String> {
    let route = v.group.as_ref().ok_or("prompt route fixture")?.route;
    let pairs = c.m * 10;
    let mut ids = prefill_experts(c);
    let bad = c.name.ends_with("-bad");
    if bad { ids[25] = ids[23]; }
    let live = |pair: usize| !(bad && pair / 10 == 2);
    let mut want = vec![u32::MAX; v.out_bytes / 4];
    let (pair_at, sorted_at, token_at, task_at) = (1025, 1025 + pairs, 1025 + 2 * pairs, 1025 + 3 * pairs);
    let (mut start, mut tasks) = (0usize, 0usize);
    for expert in 0..512 {
        let rows: Vec<usize> = (0..pairs).filter(|&pair| live(pair) && ids[pair] == expert as u32).collect();
        want[expert] = rows.len() as u32;
        want[512 + expert] = start as u32;
        for (i, &pair) in rows.iter().enumerate() { want[sorted_at + start + i] = pair as u32; }
        for first in (0..rows.len()).step_by(128) {
            want[task_at + 1 + tasks * 2] = expert as u32;
            want[task_at + 2 + tasks * 2] = first as u32;
            tasks += 1;
        }
        start += rows.len();
    }
    want[1024] = start as u32;
    want[task_at] = tasks as u32;
    for pair in 0..pairs {
        if !live(pair) { continue; }
        let token = pair / 10;
        let rank = ids[token * 10..(token + 1) * 10].iter().filter(|&&e| e < ids[pair]).count();
        want[pair_at + pair] = (token * 11 + rank) as u32;
        want[token_at + token * 10 + rank] = pair as u32;
    }
    let expected: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut actual = g.read(v.out, v.out_bytes)?;
    if actual != expected {
        let at = actual.chunks_exact(4).zip(expected.chunks_exact(4)).position(|(a,x)| a != x).unwrap();
        return Err(format!("{} arm{arm} CPU route oracle mismatch word{at}", c.name));
    }
    let errors = g.read(if arm == 0 { route.error } else { v.aux }, c.m * 4)?;
    for (token, word) in errors.chunks_exact(4).enumerate() {
        if (u32::from_le_bytes(word.try_into().unwrap()) != 0) != (bad && token == 2) {
            return Err(format!("{} arm{arm} sticky status mismatch token{token}",c.name));
        }
    }
    actual.extend(errors);
    Ok(sha256::digest(&actual))
}
