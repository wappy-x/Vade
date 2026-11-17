//! MX-QUANT device screen (exact change): the quantize pass before every MXFP8 prompt projection
//! (`Op::Mxfp8Quantize`, k = token rows * cols words in one launch; `dynamic_projection::project_rows`).
//! The baseline image runs it on the generic entry (`causal_operation`: one thread a word; each word reads its whole
//! 32-word group and searches the scale again), because `fast_admitted` listed column counts up to 16,384 only. The
//! candidate image admits token rows to the warp version (`causal_fast`: one warp a group, one maximum, the unchanged
//! `mxfp8_quantize::finish`).
//! Run: kernel-bench --mx-quant <baseline image> <candidate image> [bytes|time|time-warp|all|pack|pack-time].
//! Bytes: every word of the candidate's warp launch = the baseline's generic launch, from poisoned outputs, with guards:
//! both zero policies, planes 0 to 2, five input kinds (2 = FP32 subnormals and signed zeros, 4 = NaN and infinities),
//! at both ends of the token-row family (20,480 and 12,582,912 words), at 1,024 rows of each width and at odd row counts.
//! Time: 7 alternating repeats of 5 launches, median, at the read's shapes. Safe code only.
//! Mode pack (the baseline image is the main's, the candidate holds `mx_tc_quantize_pack` and the round inside
//! `mx_tc_tile_packed`): every byte `mx_tc_quantize_pack` writes = the baseline's quantize launch then its `mx_tc_pack`,
//! on eight input kinds (5 to 7 are built for the edges: the largest value of a group rounds down onto 224, the top of the
//! float range, the lowest scale codes), with the count of groups on each edge and controls that must differ; then the
//! whole chain to the rounded output word: baseline quantize, pack, tile, round launch = candidate quantize-and-pack, tile
//! with round = 1. Mode pack-time: the two launches against the one, and the tile with its round launch against the tile
//! that rounds as it stores.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn step(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.step() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn words(&self, g: &Gpu) -> Result<Vec<u32>> {
        let x = g.read(self.base, self.bytes + 32)?;
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXQUANT output guard changed".into()); }
        Ok(x[16..x.len() - 16].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    fn poison(&self, g: &Gpu, byte: u8) -> Result<()> { g.upload_at(self.address, &vec![byte; self.bytes]) }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> {
        let x = g.read(self.base, self.bytes + 32)?;
        if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXQUANT output guard changed".into()); }
        Ok(x[16..x.len() - 16].to_vec())
    }
}
/// Input kinds 5 to 7, BF16 patterns built a group at a time around the quantizer's edges. One lane holds the group's
/// largest value: exponent field 9 to 250 (kind 5), 250 to 254 (kind 6: the last scale codes; a quantized value can pass
/// the largest float) or 0 to 12 (kind 7: scale codes 0 to 4 and FP32 subnormals). Its seven fraction bits are 97 to 104
/// in a quarter of the groups: the largest value then sits in (224, 232] at its scale and rounds down onto 224, the top
/// of the scale code below. The other lanes sit up to 15 binades below it, a sixteenth are zeros, half are negative, and
/// one lane in eight keeps low bits (the BF16 round moves it, at times past the largest value or to an infinity).
fn edge(kind: u32, n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1); let mut v = Vec::with_capacity(n * 4);
    for _ in 0..n / 32 {
        let s = r.step();
        let field = match kind { 5 => 9 + (s % 242) as u32, 6 => 250 + (s % 5) as u32, _ => (s % 13) as u32 };
        let top = match (s >> 8) % 4 { 0 => 97 + ((s >> 16) % 8) as u32, 1 => 96, 2 => 105 + ((s >> 16) % 23) as u32, _ => ((s >> 16) % 128) as u32 };
        let place = (s >> 32) as usize % 32;
        for lane in 0..32 {
            let t = r.step();
            let bits = if lane == place { field << 23 | top << 16 } else if (t >> 20) % 16 == 0 { 0 } else {
                let drop = (t % 16) as u32; let fraction = ((t >> 8) % 128) as u32;
                field.saturating_sub(drop) << 23 | (if drop == 0 { fraction.min(top) } else { fraction }) << 16 };
            let low = if (t >> 44) % 8 == 0 { (t >> 48) as u32 & 0xffff } else { 0 };
            v.extend((bits | low | if (t >> 40) & 1 == 1 { 0x8000_0000 } else { 0 }).to_le_bytes());
        }
    }
    v
}
fn rows_of(kind: u32, n: usize, seed: u64) -> Vec<u8> { if kind < 5 { input(kind, n, seed) } else { edge(kind, n, seed) } }
/// `MxProjection` (56 bytes): weights, scales, input, output, rows, cols, tokens, input_bf16 = 0, round, mode = 0.
fn projection(weights: u64, scales: u64, input: u64, output: u64, shape: (usize, usize, usize), round: u32) -> Vec<u8> {
    let mut v = vec![]; for x in [weights, scales, input, output] { v.extend(x.to_le_bytes()); }
    for x in [shape.0 as u32, shape.1 as u32, shape.2 as u32, 0, round, 0] { v.extend(x.to_le_bytes()); } v
}
/// Bytes of one packed token row: the codes, then one scale code a group of 32 columns.
fn line(cols: usize) -> usize { cols + cols / 32 }
/// `mx_tc_pack` or `mx_tc_quantize_pack`: `tokens` rows of `cols` FP32 words at `input` to the packed rows at `plane`.
fn launch_pack(g: &Gpu, f: H, input: u64, plane: u64, cols: usize, tokens: usize) -> Result<()> {
    let mut a = projection(0, 0, input, plane, (1, cols, tokens), 0);
    g.launch(f, [(tokens * (cols / 32)).div_ceil(256) as u32, 1, 1], 256, 0, &mut [a.as_mut_ptr().cast()])
}
/// `mx_tc_tile_packed` on the packed rows at `plane`; `round` = 1 stores BF16-rounded words.
fn launch_tile(g: &Gpu, f: H, weights: (u64, u64), plane: u64, output: u64, shape: (usize, usize, usize), round: u32) -> Result<()> {
    let mut a = projection(weights.0, weights.1, plane, output, shape, round);
    g.launch(f, [shape.2.div_ceil(64) as u32, shape.0.div_ceil(128) as u32, 1], 256, 0, &mut [a.as_mut_ptr().cast()])
}
/// `bf16_lanes_round` (`LaneCast`: input, output, words).
fn launch_round(g: &Gpu, f: H, input: u64, output: u64, words: usize) -> Result<()> {
    let mut a = vec![]; a.extend(input.to_le_bytes()); a.extend(output.to_le_bytes()); a.extend((words as u32).to_le_bytes()); a.extend([0u8; 4]);
    g.launch(f, [words.div_ceil(128) as u32, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
/// Weight rows for the chain: E4M3 codes of both signs without the NaN code, and scale codes 120 to 131.
fn weights(g: &mut Gpu, rows: usize, cols: usize, seed: u64) -> Result<(Buf, Buf)> {
    let mut r = Rng(seed | 1);
    let codes: Vec<u8> = (0..rows * cols).map(|_| { let s = r.step(); (s % 127) as u8 | ((s >> 32) as u8 & 0x80) }).collect();
    let scales: Vec<u8> = (0..rows * cols / 32).map(|_| 120 + (r.step() % 12) as u8).collect();
    Ok((Buf::new(g, &codes)?, Buf::new(g, &scales)?))
}
/// Input kinds: 0 plain, 1 wide range, 2 FP32 subnormals and signed zeros, 3 large, 4 wide range with NaN, infinities and
/// the largest finite value (a group that holds one gives NaN in all 32 lanes on both entries).
fn input(kind: u32, n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| { let (u, s) = (r.unit(), r.step()); match kind {
        0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
        _ => match i % 4099 { 7 => f32::NAN, 11 => f32::INFINITY, 13 => f32::NEG_INFINITY, 17 => f32::from_bits(0x7f7f_ffff), _ => u * f32::from_bits(((s % 60 + 97) as u32) << 23) } } })
        .flat_map(|v| v.to_le_bytes()).collect()
}
/// `causal_ops::Arguments` (80 bytes): four inputs, weight, output, then the shape (op, heads, dim, taps, rows = plane,
/// cols = words, epsilon, bound). An absent owner is the engine's placeholder 4.
fn arguments(op: u32, plane: u32, k: usize, source: u64, output: u64) -> Vec<u8> {
    let mut v = vec![]; for x in [source, 4, 4, 4, 4, output] { v.extend(x.to_le_bytes()); }
    for x in [op, 0, 0, 0, plane, k as u32] { v.extend(x.to_le_bytes()); } v.extend([0u8; 8]); v
}
/// `warp` false: the generic geometry (`causal_ops::geometry` default, one thread a word). true: one warp a group, four a block.
fn launch(g: &Gpu, f: H, warp: bool, op: u32, plane: u32, k: usize, source: u64, output: u64) -> Result<()> {
    let mut a = arguments(op, plane, k, source, output);
    let blocks = if warp { (k / 32).div_ceil(4) } else { k.div_ceil(128) } as u32;
    g.launch(f, [blocks, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
const QUANTIZE: u32 = 18; const QUANTIZE_POSITIVE_ZERO: u32 = 27;
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXQUANT baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let (generic, warp) = (g.function(ma, "causal_operation")?, g.function(mb, "causal_fast")?);
    // The baseline's own warp entry: what main launches for single rows and (since pull request 56) for token rows.
    let base_warp = g.function(ma, "causal_fast")?;
    let started = Instant::now(); let mut fails = 0usize;
    if mode == "bytes" || mode == "all" {
        // The drafting and many-chat shapes (5 to 8 rows of 4,096, 3 to 8 of 8,192, 2 to 8 of 12,288), then the prompt groups.
        for (cols, tokens) in [(4096usize, 5usize), (4096, 6), (4096, 7), (4096, 8), (8192, 3), (8192, 4), (8192, 5), (8192, 6), (8192, 7), (8192, 8), (12288, 2), (12288, 3), (12288, 4), (12288, 5), (12288, 6), (12288, 7), (12288, 8),
            (4096, 257), (4096, 1024), (8192, 1023), (8192, 1024), (12288, 511), (12288, 1024)] {
            let k = cols * tokens; let (mut words, mut differ, mut launches) = (0usize, 0usize, 0usize);
            for kind in 0..5u32 {
                let source = Buf::new(&mut g, &input(kind, k, 77 + u64::from(kind) + k as u64))?;
                let (x, y) = (Buf::new(&mut g, &vec![0xff; k * 4])?, Buf::new(&mut g, &vec![0xee; k * 4])?);
                for op in [QUANTIZE, QUANTIZE_POSITIVE_ZERO] { for plane in 0..3u32 {
                    // Every policy and plane on the plain kind and at 1,024 rows; the prompt path's own launch (plane 0) on every kind.
                    if !(kind == 0 || tokens == 1024 || plane == 0) { continue; }
                    x.poison(&g, 0xff)?; y.poison(&g, 0xee)?;
                    launch(&g, generic, false, op, plane, k, source.address, x.address)?; launch(&g, warp, true, op, plane, k, source.address, y.address)?; g.sync()?;
                    let (wx, wy) = (x.words(&g)?, y.words(&g)?);
                    words += wx.len(); launches += 1; differ += wx.iter().zip(&wy).filter(|(p, q)| p != q).count();
                } }
                g.clear()?;
            }
            println!("MXQUANT BYTES cols={cols} tokens={tokens} words={k} launches_a_side={launches} compared_words={words} differing={differ}");
            if differ != 0 { fails += 1; }
        }
        // Warp entry against warp entry (a change of `mxfp8_quantize::finish` itself): the single-row counts of the old list
        // and token rows, every policy, plane and input kind. An image that holds no token-row admission is skipped by its trap.
        for (cols, tokens) in [(32usize, 1usize), (64, 1), (160, 1), (1024, 1), (4096, 1), (8192, 1), (12288, 1), (4096, 4), (4096, 5), (8192, 3), (4096, 257), (4096, 1024), (8192, 1024), (12288, 1024)] {
            let k = cols * tokens; let (mut words, mut differ, mut launches) = (0usize, 0usize, 0usize);
            for kind in 0..5u32 {
                let source = Buf::new(&mut g, &input(kind, k, 177 + u64::from(kind) + k as u64))?;
                let (x, y) = (Buf::new(&mut g, &vec![0xff; k * 4])?, Buf::new(&mut g, &vec![0xee; k * 4])?);
                for op in [QUANTIZE, QUANTIZE_POSITIVE_ZERO] { for plane in 0..3u32 {
                    x.poison(&g, 0xff)?; y.poison(&g, 0xee)?;
                    launch(&g, base_warp, true, op, plane, k, source.address, x.address)?; launch(&g, warp, true, op, plane, k, source.address, y.address)?; g.sync()?;
                    let (wx, wy) = (x.words(&g)?, y.words(&g)?);
                    words += wx.len(); launches += 1; differ += wx.iter().zip(&wy).filter(|(p, q)| p != q).count();
                } }
                g.clear()?;
            }
            println!("MXQUANT WARP BYTES cols={cols} tokens={tokens} words={k} launches_a_side={launches} compared_words={words} differing={differ}");
            if differ != 0 { fails += 1; }
        }
        println!("MXQUANT BYTES {} fails={fails} elapsed_s={:.1}", if fails == 0 { "PASS" } else { "FAIL" }, started.elapsed().as_secs_f64());
    }
    if mode == "time" || mode == "time-warp" || mode == "all" {
        for (cols, tokens) in [(4096usize, 1024usize), (8192, 1024), (12288, 1024), (4096, 256), (4096, 16), (4096, 1), (8192, 1)] {
            let k = cols * tokens;
            let source = Buf::new(&mut g, &input(0, k, 5))?; let out = Buf::new(&mut g, &vec![0xff; k * 4])?;
            let (mut ta, mut tb) = (vec![], vec![]);
            for repeat in 0..7 {
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    // Mode time-warp: the baseline arm is the baseline's own warp entry (a change of `finish` itself).
                    let go = |g: &Gpu| if arm == 0 { launch(g, if mode == "time-warp" { base_warp } else { generic }, mode == "time-warp", QUANTIZE, 0, k, source.address, out.address) } else { launch(g, warp, true, QUANTIZE, 0, k, source.address, out.address) };
                    go(&g)?; g.sync()?;
                    let n = 5; let t0 = Instant::now();
                    for _ in 0..n { go(&g)?; }
                    g.sync()?; (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let spread = |v: &[f64]| { let (lo, hi) = v.iter().fold((f64::MAX, 0f64), |(l, h), x| (l.min(*x), h.max(*x))); hi - lo };
            let (ra, rb) = (median(ta.clone()), median(tb.clone()));
            println!("MXQUANT TIME cols={cols} tokens={tokens} generic_us={ra:.1} warp_us={rb:.1} ratio={:.2} generic_spread_us={:.1} warp_spread_us={:.1}", ra / rb, spread(&ta), spread(&tb));
            g.clear()?;
        }
    }
    if mode == "pack" || mode == "pack-time" {
        // Baseline: the quantize launch, `mx_tc_pack`, the tile on packed rows, the round launch. Candidate: one launch
        // for the packed rows, and the tile that rounds as it stores.
        let (old_pack, old_tile, old_round) = (g.function(ma, "mx_tc_pack")?, g.function(ma, "mx_tc_tile_packed")?, g.function(ma, "bf16_lanes_round")?);
        let (new_pack, new_tile) = (g.function(mb, "mx_tc_quantize_pack")?, g.function(mb, "mx_tc_tile_packed")?);
        if mode == "pack" {
            // (a) every byte of the packed rows, all eight kinds.
            let (mut total, mut total_differ) = (0usize, 0usize);
            let mut edges = [0usize; 5];
            // Word counts the baseline's warp quantize entry admits (`causal_ops::fast_admitted`): one row of 1,024 columns, and
            // token rows of the model's three widths (a multiple of 4,096 words past 16,384).
            for (cols, tokens) in [(1024usize, 1usize), (4096, 9), (4096, 65), (4096, 1024), (8192, 9), (8192, 300), (8192, 1024), (12288, 9), (12288, 33), (12288, 1024)] {
                let k = cols * tokens; let (mut bytes, mut differ) = (0usize, 0usize);
                for kind in 0..8u32 {
                    let raw = rows_of(kind, k, 900 + u64::from(kind) + k as u64);
                    let source = Buf::new(&mut g, &raw)?;
                    let (decoded, codes) = (Buf::new(&mut g, &vec![0xff; k * 4])?, Buf::new(&mut g, &vec![0xff; k * 4])?);
                    let (x, y) = (Buf::new(&mut g, &vec![0xee; tokens * line(cols)])?, Buf::new(&mut g, &vec![0xdd; tokens * line(cols)])?);
                    launch(&g, base_warp, true, QUANTIZE, 0, k, source.address, decoded.address)?;
                    launch(&g, base_warp, true, QUANTIZE, 2, k, source.address, codes.address)?;
                    launch_pack(&g, old_pack, decoded.address, x.address, cols, tokens)?;
                    launch_pack(&g, new_pack, source.address, y.address, cols, tokens)?;
                    g.sync()?;
                    let (bx, by, quantizer) = (x.read(&g)?, y.read(&g)?, codes.words(&g)?);
                    bytes += bx.len();
                    let n = bx.iter().zip(&by).filter(|(p, q)| p != q).count();
                    if n > 0 && total_differ + differ == 0 {
                        let i = (0..bx.len()).find(|&i| bx[i] != by[i]).unwrap(); let (token, at) = (i / line(cols), i % line(cols));
                        let group = if at < cols { at / 32 } else { at - cols };
                        let words: Vec<String> = raw[(token * cols + group * 32) * 4..][..128].chunks_exact(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
                        println!("MXQUANT PACK FIRST DIFFER cols={cols} tokens={tokens} kind={kind} token={token} byte={at} group={group} two_launches={:02x} one_launch={:02x} scale_bytes={:02x}/{:02x} input={}", bx[i], by[i], bx[token * line(cols) + cols + group], by[token * line(cols) + cols + group], words.join(","));
                    }
                    differ += n;
                    // The edges this input reached, from the baseline's own bytes: [0] groups, [1] not finite, [2] of those with
                    // finite sources (a quantized value passed the largest float), [3] the scale code one below the quantizer's
                    // (the largest value sits on 224), [4] the quantizer's scale code 0.
                    for token in 0..tokens { for group in 0..cols / 32 {
                        let (packed, own) = (bx[token * line(cols) + cols + group], f32::from_bits(quantizer[token * cols + group * 32]));
                        edges[0] += 1;
                        if packed == 255 { edges[1] += 1; if own.is_finite() { edges[2] += 1; } }
                        else if f32::from(packed) + 1.0 == own { edges[3] += 1; }
                        if own == 0.0 { edges[4] += 1; }
                    } }
                    g.clear()?;
                }
                println!("MXQUANT PACK BYTES cols={cols} tokens={tokens} kinds=8 compared_bytes={bytes} differing={differ}");
                total += bytes; total_differ += differ;
            }
            let reached = edges[1..].iter().all(|n| *n > 0);
            println!("MXQUANT PACK EDGES groups={} not_finite={} of_those_quantized_past_the_largest_float={} scale_code_one_below_the_quantizers={} quantizer_scale_code_0={}", edges[0], edges[1], edges[2], edges[3], edges[4]);
            println!("MXQUANT PACK BYTES {} one launch = quantize then pack: compared_bytes={total} differing={total_differ} every_edge_reached={reached}", if total_differ == 0 && reached { "PASS" } else { "FAIL" });
            if total_differ != 0 || !reached { fails += 1; }
            // (b) controls. One input word changed: only its own item (32 code bytes and one scale byte) may move, and it must.
            {
                let (cols, tokens) = (4096usize, 65usize); let k = cols * tokens;
                let mut raw = rows_of(0, k, 4242);
                let source = Buf::new(&mut g, &raw)?;
                let (x, y) = (Buf::new(&mut g, &vec![0xee; tokens * line(cols)])?, Buf::new(&mut g, &vec![0xdd; tokens * line(cols)])?);
                launch_pack(&g, new_pack, source.address, x.address, cols, tokens)?; g.sync()?;
                let (token, column) = (7usize, 1000usize);
                let first: [u8; 4] = raw[(token * cols + column) * 4..][..4].try_into().unwrap();
                raw[(token * cols + column) * 4..][..4].copy_from_slice(&1.0e6f32.to_le_bytes());
                g.upload_at(source.address, &raw)?;
                launch_pack(&g, new_pack, source.address, y.address, cols, tokens)?; g.sync()?;
                let (bx, by) = (x.read(&g)?, y.read(&g)?);
                let own = |i: usize| i / line(cols) == token && { let at = i % line(cols); if at < cols { at / 32 == column / 32 } else { at - cols == column / 32 } };
                let (hit, stray) = ((0..bx.len()).filter(|&i| own(i) && bx[i] != by[i]).count(), (0..bx.len()).filter(|&i| !own(i) && bx[i] != by[i]).count());
                // The word put back: the first bytes again, from a poisoned output.
                raw[(token * cols + column) * 4..][..4].copy_from_slice(&first);
                g.upload_at(source.address, &raw)?; y.poison(&g, 0xdd)?;
                launch_pack(&g, new_pack, source.address, y.address, cols, tokens)?; g.sync()?;
                let back = y.read(&g)?.iter().zip(&bx).filter(|(p, q)| p != q).count();
                let ok = hit > 0 && stray == 0 && back == 0;
                println!("MXQUANT PACK CONTROL {} one input word changed: bytes_of_its_group_that_differ={hit} of 33 other_bytes={stray}; the word put back: bytes_that_differ_from_the_first_run={back}", if ok { "PASS" } else { "FAIL" });
                if !ok { fails += 1; }
                g.clear()?;
                // The pack without the quantize pass (the baseline's `mx_tc_pack` on the raw rows) is NOT these bytes.
                let raw = rows_of(5, k, 4343);
                let source = Buf::new(&mut g, &raw)?;
                let (x, y) = (Buf::new(&mut g, &vec![0xee; tokens * line(cols)])?, Buf::new(&mut g, &vec![0xdd; tokens * line(cols)])?);
                launch_pack(&g, old_pack, source.address, x.address, cols, tokens)?;
                launch_pack(&g, new_pack, source.address, y.address, cols, tokens)?; g.sync()?;
                let (bx, by) = (x.read(&g)?, y.read(&g)?);
                let n = bx.iter().zip(&by).filter(|(p, q)| p != q).count();
                println!("MXQUANT PACK CONTROL {} the pack alone on the raw rows (no quantize pass) differs: bytes={n} of {}", if n > 0 { "PASS" } else { "FAIL" }, bx.len());
                if n == 0 { fails += 1; }
                g.clear()?;
            }
            // (c) the chain to the rounded output word, device against device, on the model's shapes.
            let (mut words, mut differ, mut raw_differ, mut moved_by_round) = (0usize, 0usize, 0usize, 0usize);
            for (rows, cols) in [(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] { for tokens in [9usize, 65, 1024] { for kind in [0u32, 1, 5] {
                let k = cols * tokens; let shape = (rows, cols, tokens);
                let (w, s) = weights(&mut g, rows, cols, 7000 + (rows + cols + tokens) as u64)?;
                let source = Buf::new(&mut g, &rows_of(kind, k, 7100 + u64::from(kind) + k as u64))?;
                let decoded = Buf::new(&mut g, &vec![0xff; k * 4])?;
                let (px, py) = (Buf::new(&mut g, &vec![0xee; tokens * line(cols)])?, Buf::new(&mut g, &vec![0xdd; tokens * line(cols)])?);
                let (sum, old, new, plain) = (Buf::new(&mut g, &vec![0xff; rows * tokens * 4])?, Buf::new(&mut g, &vec![0xee; rows * tokens * 4])?, Buf::new(&mut g, &vec![0xdd; rows * tokens * 4])?, Buf::new(&mut g, &vec![0xcc; rows * tokens * 4])?);
                launch(&g, base_warp, true, QUANTIZE, 0, k, source.address, decoded.address)?;
                launch_pack(&g, old_pack, decoded.address, px.address, cols, tokens)?;
                launch_tile(&g, old_tile, (w.address, s.address), px.address, sum.address, shape, 0)?;
                launch_round(&g, old_round, sum.address, old.address, rows * tokens)?;
                launch_pack(&g, new_pack, source.address, py.address, cols, tokens)?;
                launch_tile(&g, new_tile, (w.address, s.address), py.address, new.address, shape, 1)?;
                launch_tile(&g, new_tile, (w.address, s.address), py.address, plain.address, shape, 0)?;
                g.sync()?;
                let (sum, old, new, plain) = (sum.words(&g)?, old.words(&g)?, new.words(&g)?, plain.words(&g)?);
                words += old.len();
                let n = (0..old.len()).filter(|&i| old[i] != new[i]).count();
                if n > 0 && differ == 0 { let i = (0..old.len()).find(|&i| old[i] != new[i]).unwrap(); println!("MXQUANT PACK CHAIN FIRST DIFFER rows={rows} cols={cols} tokens={tokens} kind={kind} token={} row={} four_launches={:08x} two_launches={:08x}", i / rows, i % rows, old[i], new[i]); }
                differ += n;
                raw_differ += (0..sum.len()).filter(|&i| sum[i] != plain[i]).count();
                moved_by_round += (0..sum.len()).filter(|&i| sum[i] != old[i]).count();
                g.clear()?;
            } } }
            let ok = differ == 0 && raw_differ == 0 && moved_by_round > 0;
            println!("MXQUANT PACK CHAIN {} quantize, pack, tile, round launch = quantize-and-pack, tile with round 1: words={words} differing={differ}; round 0 on the candidate against the baseline's tile: differing={raw_differ}; control, words the round moves (round 0 is not the chain's word): {moved_by_round}", if ok { "PASS" } else { "FAIL" });
            if !ok { fails += 1; }
            println!("MXQUANT PACK {} fails={fails} elapsed_s={:.1}", if fails == 0 { "PASS" } else { "FAIL" }, started.elapsed().as_secs_f64());
        } else {
            let spread = |v: &[f64]| { let (lo, hi) = v.iter().fold((f64::MAX, 0f64), |(l, h), x| (l.min(*x), h.max(*x))); hi - lo };
            // The two launches (quantize, pack) against the one, at the read's widths.
            for (cols, tokens) in [(4096usize, 1024usize), (8192, 1024), (12288, 1024), (4096, 256), (4096, 64)] {
                let k = cols * tokens;
                let source = Buf::new(&mut g, &input(0, k, 5))?; let decoded = Buf::new(&mut g, &vec![0xff; k * 4])?;
                let plane = Buf::new(&mut g, &vec![0xee; tokens * line(cols)])?;
                let (mut ta, mut tb, mut tq) = (vec![], vec![], vec![]);
                for repeat in 0..7 {
                    for arm in if repeat % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                        let go = |g: &Gpu| -> Result<()> { match arm {
                            0 => { launch(g, base_warp, true, QUANTIZE, 0, k, source.address, decoded.address)?; launch_pack(g, old_pack, decoded.address, plane.address, cols, tokens) }
                            1 => launch_pack(g, new_pack, source.address, plane.address, cols, tokens),
                            _ => launch(g, base_warp, true, QUANTIZE, 0, k, source.address, decoded.address) } };
                        go(&g)?; g.sync()?;
                        let n = 5; let t0 = Instant::now();
                        for _ in 0..n { go(&g)?; }
                        g.sync()?; (match arm { 0 => &mut ta, 1 => &mut tb, _ => &mut tq }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                    }
                }
                let (ra, rb, rq) = (median(ta.clone()), median(tb.clone()), median(tq.clone()));
                println!("MXQUANT PACK TIME cols={cols} tokens={tokens} quantize_then_pack_us={ra:.1} (quantize alone {rq:.1}) one_launch_us={rb:.1} saved_us={:.1} ratio={:.2} spread_us={:.1}/{:.1}", ra - rb, ra / rb, spread(&ta), spread(&tb));
                g.clear()?;
            }
            // The tile and its round launch against the tile that rounds as it stores.
            for (rows, cols) in [(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] {
                let tokens = 1024usize; let k = cols * tokens; let shape = (rows, cols, tokens);
                let (w, s) = weights(&mut g, rows, cols, 9000 + (rows + cols) as u64)?;
                let source = Buf::new(&mut g, &input(0, k, 6))?;
                let plane = Buf::new(&mut g, &vec![0xee; tokens * line(cols)])?;
                let (sum, out) = (Buf::new(&mut g, &vec![0xff; rows * tokens * 4])?, Buf::new(&mut g, &vec![0xee; rows * tokens * 4])?);
                launch_pack(&g, new_pack, source.address, plane.address, cols, tokens)?; g.sync()?;
                let (mut ta, mut tb, mut tt) = (vec![], vec![], vec![]);
                for repeat in 0..7 {
                    for arm in if repeat % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                        let go = |g: &Gpu| -> Result<()> { match arm {
                            0 => { launch_tile(g, old_tile, (w.address, s.address), plane.address, sum.address, shape, 0)?; launch_round(g, old_round, sum.address, out.address, rows * tokens) }
                            1 => launch_tile(g, new_tile, (w.address, s.address), plane.address, out.address, shape, 1),
                            _ => launch_tile(g, old_tile, (w.address, s.address), plane.address, sum.address, shape, 0) } };
                        go(&g)?; g.sync()?;
                        let n = 5; let t0 = Instant::now();
                        for _ in 0..n { go(&g)?; }
                        g.sync()?; (match arm { 0 => &mut ta, 1 => &mut tb, _ => &mut tt }).push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                    }
                }
                let (ra, rb, rt) = (median(ta.clone()), median(tb.clone()), median(tt.clone()));
                println!("MXQUANT PACK TIME rows={rows} cols={cols} tokens={tokens} tile_then_round_us={ra:.1} (tile alone {rt:.1}) tile_with_round_us={rb:.1} saved_us={:.1} ratio={:.2} spread_us={:.1}/{:.1}", ra - rb, ra / rb, spread(&ta), spread(&tb));
                g.clear()?;
            }
        }
    }
    println!("MXQUANT DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    if fails != 0 { return Err(format!("MXQUANT bytes failed: {fails}")); }
    Ok(())
}
