//! MX-TC: the MXFP8 rows on the tensor cores (`mx_tc_rows`, `mx_tc_tile`; `engine/docs/tc-dense-contract.md`, section
//! MXFP8) against their CPU twin (`gpu/src/tc_twin_mx.rs`) and against today's MXFP8 kernels of the base image.
//! Run: kernel-bench --mx-tc <base ptx> <candidate ptx> [probe|gates|wired|step|stage|time|tile|var|all], or a list of
//! modes with commas (probe,gates,tile). The mode `code` alone loads nothing: it compiles both images with the box's
//! offline compiler and lists the kernels whose machine code differs (`kernel_codepair.rs`).
//! The modes `feed-pair` and `expert-pair` alone are the device checks of two kernels the code list can name as drifted
//! (`kernel_feedpair.rs`, `kernel_expertpair.rs`).
//! tile:  the prompt tile alone: token rows a launch from 64 to 1,024, and 1,024 rows in pieces (the host's piece size).
//! wired: the SERVED entries of the candidate image (the one-row layer entry, the merged batches, the row stage, the row
//!        group entries) on descriptors with the tensor-core mode: every word = the twin; a scalar job in the same launch
//!        keeps the base image's words; two controls.
//! stage: the row stage (q, k, v in one launch) at 2 to 8 source rows against the one-row launch, recorded, cold weights.
//! var:   the bench entries of the row body (`mx_tc_rows_*`) against `mx_tc_rows`, every word, then the cold pass of all.
//!
//! probe: one raw step a warp (`mx_tc_probe`: the block-scaled step with an old sum) on eight classes of codes times
//!        eight classes of scale codes and old sums, against a list of integer models (kept fraction bits 18 to 40, zero
//!        products counted or not, the last cut toward zero or to nearest, three floors of the cut): the model the twin
//!        holds must have 0 words that differ; results under 2^-126 are counted apart and must be 0 too. The
//!        activation conversion (`mx_tc_probe_cvt`) against the twin's code rule.
//! gates: G1 every word of the device = the twin on small shapes (rows and tokens off the tile sizes, 32 to 4,096
//!        columns, six weight classes, seven activation classes, far scale codes), sampled words on the model's shapes.
//!        G2 a token row alone = the same row in groups of 2 to 67 rows at three placements (other slot, other
//!        neighbors), every word, device against device. G3 wrong-on-purpose controls that must be caught. G4 the
//!        packed rows: every byte `mx_tc_pack` writes = the twin's packed row; `mx_tc_tile_packed` = `mx_tc_tile`, every
//!        word; one control. A call of more than 8 token rows on a multiple of 1,024 columns takes the packed path in
//!        every gate, as the engine does.
//! time:  old (base image: the one-token kernel, the small entries, the prompt tile) against new. COLD: a pass of the
//!        model's 145 calls of one rank at 1, 2, 4 and 8 token rows, every call on its own weights (the decode step's
//!        view). TIME: one matrix again and again at 1, 2, 4, 8, 256 and 1,024 token rows (the prompt group's view).
//!        Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, sha256};
use std::time::Instant;
#[path = "../../../crates/gpu/src/tc_twin_mx.rs"]
#[allow(dead_code)]
mod twin;
use twin::Rng;

const GUARD: [u8; 16] = [0x5a; 16];
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn poisoned(g: &mut Gpu, words: usize) -> Result<Self> { Self::new(g, &vec![0xff; words * 4]) }
    /// The words, after a check that the guards on both sides are whole.
    fn words(&self, g: &Gpu) -> Result<Vec<u32>> { Ok(self.read(g)?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()) }
    /// The bytes, after the same check.
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> {
        let all = g.read(self.base, self.bytes + 32)?;
        if all[..16] != GUARD || all[all.len() - 16..] != GUARD { return Err("MXTC guard changed: a kernel wrote outside its output".into()); }
        Ok(all[16..all.len() - 16].to_vec())
    }
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
#[derive(Clone, Copy)]
struct Proj { weights: u64, scales: u64, input: u64, output: u64, rows: u32, cols: u32, tokens: u32 }
/// `MxProjection`: four addresses, rows, cols, tokens, input_bf16 = 0, round = 0, padded to 56 bytes.
fn proj_bytes(p: &Proj) -> Vec<u8> {
    let mut v = vec![];
    for x in [p.weights, p.scales, p.input, p.output] { v.extend(x.to_le_bytes()); }
    for x in [p.rows, p.cols, p.tokens, 0, 0] { v.extend(x.to_le_bytes()); }
    v.extend([0u8; 4]);
    v
}
/// `packed`: (`mx_tc_pack`, `mx_tc_tile_packed`).
struct New { rows: H, tile: Option<H>, packed: Option<(H, H)> }
/// Bytes of one packed token row: the codes, then one scale code a group of 32 columns.
fn line(cols: usize) -> usize { cols + cols / 32 }
/// A call the engine packs: more than 8 token rows on a multiple of 1,024 columns.
fn packs(new: &New, p: &Proj) -> bool { new.packed.is_some() && p.tokens > 8 && p.cols.is_multiple_of(1024) }
/// The packed rows of `p`'s token rows, written to `plane`.
fn launch_pack(g: &Gpu, pack: H, p: &Proj, plane: u64) -> Result<()> {
    let mut a = proj_bytes(&Proj { output: plane, ..*p });
    g.launch(pack, [(p.tokens * (p.cols / 32)).div_ceil(256), 1, 1], 256, 0, &mut [a.as_mut_ptr().cast()])
}
/// The tile on the packed rows at `plane`.
fn launch_packed_tile(g: &Gpu, tile: H, p: &Proj, plane: u64) -> Result<()> {
    let mut a = proj_bytes(&Proj { input: plane, ..*p });
    g.launch(tile, [p.tokens.div_ceil(64), p.rows.div_ceil(128), 1], 256, 0, &mut [a.as_mut_ptr().cast()])
}
/// The new entry a call of `tokens` rows takes: from 9 rows the tile (when the image has it), on packed rows when the
/// call is one the engine packs and the caller gives a `plane` (not 0) of `tokens * line(cols)` bytes; else the row entry.
fn launch_new(g: &Gpu, new: &New, p: &Proj, plane: u64) -> Result<()> {
    if let (true, Some((pack, tile))) = (plane != 0 && packs(new, p), new.packed) {
        launch_pack(g, pack, p, plane)?;
        return launch_packed_tile(g, tile, p, plane);
    }
    let mut a = proj_bytes(p);
    match new.tile {
        Some(tile) if p.tokens > 8 => g.launch(tile, [p.tokens.div_ceil(64), p.rows.div_ceil(128), 1], 256, 0, &mut [a.as_mut_ptr().cast()]),
        _ => g.launch(new.rows, [p.rows.div_ceil(16), p.tokens.div_ceil(8), 1], 128, 0, &mut [a.as_mut_ptr().cast()]),
    }
}
/// A plane for the call `p`: its address, or 0 when the call is not packed.
fn plane_for(g: &mut Gpu, new: &New, p: &Proj) -> Result<u64> {
    if !packs(new, p) { return Ok(0); }
    Ok(Buf::new(g, &vec![0xee; p.tokens as usize * line(p.cols as usize)])?.address)
}
fn same(device: u32, twin: u32) -> bool { device == twin || f32::from_bits(device).is_nan() && f32::from_bits(twin).is_nan() }

// ------------------------------------------------------------------ fixtures
struct Fixture { rows: usize, cols: usize, tokens: usize, w: Vec<u8>, s: Vec<u8>, x: Vec<f32> }
/// Weight row `r` has class `r % 6`; token `t` has activation class `t % 7`, and every fifth group of a token row the
/// class `(t + group) % 7`. Weight scale codes are 125 to 128 (every group counts in the word); with `far`, 112 to 139 and
/// one group in eight any code 0 to 254.
fn fixture(rows: usize, cols: usize, tokens: usize, seed: u64, far: bool) -> Fixture {
    let mut r = Rng(seed | 1);
    let groups = cols / 32;
    let w = (0..rows * cols).map(|i| twin::weight_code(&mut r, i, i / cols)).collect();
    let s = (0..rows * groups).map(|_| if !far { 125 + r.below(4) as u8 } else if r.below(8) == 0 { r.below(255) as u8 } else { 112 + r.below(28) as u8 }).collect();
    let mut x = Vec::with_capacity(tokens * cols);
    for t in 0..tokens { for group in 0..groups { x.extend(twin::activation_group(&mut r, group * 32, if group % 5 == 4 { t + group } else { t })); } }
    Fixture { rows, cols, tokens, w, s, x }
}
struct Device { w: Buf, s: Buf, x: Buf }
fn upload(g: &mut Gpu, f: &Fixture) -> Result<Device> { Ok(Device { w: Buf::new(g, &f.w)?, s: Buf::new(g, &f.s)?, x: Buf::new(g, &le(&f.x))? }) }
/// The device words `[tokens][rows]` of the whole fixture in one call.
fn device_words(g: &mut Gpu, new: &New, f: &Fixture, d: &Device) -> Result<Vec<u32>> {
    let out = Buf::poisoned(g, f.rows * f.tokens)?;
    let p = Proj { weights: d.w.address, scales: d.s.address, input: d.x.address, output: out.address, rows: f.rows as u32, cols: f.cols as u32, tokens: f.tokens as u32 };
    let plane = plane_for(g, new, &p)?;
    launch_new(g, new, &p, plane)?;
    g.sync()?;
    out.words(g)
}
fn twin_word(f: &Fixture, encoded: &[(u8, [u8; 32])], row: usize) -> u32 {
    let groups = f.cols / 32;
    twin::word_encoded(&f.w[row * f.cols..(row + 1) * f.cols], &f.s[row * groups..(row + 1) * groups], encoded)
}
/// Device against twin. `sample` = None: every word; Some(n): n words a token row, spread over the rows.
/// Returns (words checked, words that differ, the first that differs as (token, row, device, twin)).
fn against_twin(f: &Fixture, device: &[u32], sample: Option<usize>) -> (usize, usize, Option<(usize, usize, u32, u32)>) {
    let (mut checked, mut differ, mut first) = (0, 0, None);
    for t in 0..f.tokens {
        let encoded = twin::encode_row(&f.x[t * f.cols..(t + 1) * f.cols]);
        let step = sample.map_or(1, |n| (f.rows / n).max(1));
        for row in (0..f.rows).step_by(step) {
            // A sampled walk also visits the last row, where a tile ends.
            for row in if sample.is_some() && row + step >= f.rows { vec![row, f.rows - 1] } else { vec![row] } {
                let (have, want) = (device[t * f.rows + row], twin_word(f, &encoded, row));
                checked += 1;
                if !same(have, want) { differ += 1; first.get_or_insert((t, row, have, want)); }
            }
        }
    }
    (checked, differ, first)
}

// ------------------------------------------------------------------ probe
/// One code of probe class `class % 8`.
fn probe_code(r: &mut Rng, class: usize, weight: bool, k: usize) -> u8 {
    let h = r.next_u64();
    let sign = ((h >> 40) as u8) & 0x80;
    match class % 8 {
        0 => sign | (h % 127) as u8,
        1 => sign | (0x30 + (h % 24) as u8),
        2 => if h % 100 < 85 { sign } else if h & 0x100 == 0 { sign | (0x70 + (h >> 12) as u8 % 15) } else { sign | (1 + (h >> 12) as u8 % 15) },
        3 => sign | ((h >> 8) as u8 % 16),
        4 => if weight { sign | (h % 127) as u8 } else { sign },
        5 => if k % 2 == 0 { 0x38 } else if weight { 0xb8 } else { 0x38 },
        6 => sign | ((1 + (h >> 8) as u8 % 14) << 3) | 7,
        _ => if k == 5 { sign | 0x7e } else { sign | (1 + (h >> 8) as u8 % 23) },
    }
}
/// The 24 scale codes (16 weight rows, then 8 tokens) and the 128 old sums of probe case `c`: class `(c / 8) % 8`.
/// 0 scale codes 127 and old sums +0.0 (the plain step); 1 codes near 127, old sums +0.0; 2 old sums of the products'
/// size; 3 old sums far above the products; 4 far below; 5 any code and any finite old sum; 6 old sums that are multiples
/// of 1/64 (cancellation); 7 by halves: the special values (NaN, infinities, -0.0, subnormals, scale code 255), and
/// results near and under 2^-126 (scale codes 55 to 70).
fn probe_state(r: &mut Rng, c: usize) -> ([u8; 24], Vec<u32>) {
    let float = |r: &mut Rng, low: u32, high: u32| { let h = r.next_u64(); (h as u32 & 0x807f_ffff) | ((low + (h >> 40) as u32 % (high - low + 1)) << 23) };
    let class = (c / 8) % 8;
    let (mut codes, mut old) = ([127u8; 24], vec![0u32; 128]);
    match class {
        0 => {}
        1 => for code in codes.iter_mut() { *code = 120 + r.below(15) as u8; },
        2 | 3 | 4 | 6 => {
            for code in codes.iter_mut() { *code = 124 + r.below(7) as u8; }
            for word in old.iter_mut() { *word = match class { 2 => float(r, 117, 137), 3 => float(r, 147, 167), 4 => float(r, 87, 117), _ => ((r.below(131072) as f32 - 65536.0) / 64.0).to_bits() }; }
        }
        5 => {
            for code in codes.iter_mut() { *code = r.below(255) as u8; }
            for word in old.iter_mut() { *word = float(r, 0, 254); }
        }
        _ => if (c / 64).is_multiple_of(2) {
            for code in codes.iter_mut() { *code = if r.below(6) == 0 { 255 } else { 100 + r.below(51) as u8 }; }
            for word in old.iter_mut() { *word = match r.below(10) { 0 => 0x7fc0_0000, 1 => 0x7f80_0001, 2 => 0xffff_ffff, 3 => 0x7f80_0000, 4 => 0xff80_0000, 5 => 0x8000_0000, 6 => float(r, 0, 0), 7 => 0, _ => float(r, 100, 150) }; }
        } else {
            for code in codes.iter_mut() { *code = 55 + r.below(16) as u8; }
            for word in old.iter_mut() { *word = match r.below(4) { 0 => 0, 1 => float(r, 0, 0), 2 => float(r, 1, 30), _ => 0x8000_0000 }; }
        },
    }
    (codes, old)
}
fn probe(g: &mut Gpu, module: H) -> Result<bool> {
    let f = g.function(module, "mx_tc_probe")?;
    let cases = 8192usize;
    let mut r = Rng(0x51ed_2026);
    let (mut w, mut x, mut s, mut old) = (vec![0u8; cases * 512], vec![0u8; cases * 256], vec![0u8; cases * 32], Vec::with_capacity(cases * 128));
    for c in 0..cases {
        for k in 0..512 { w[c * 512 + k] = probe_code(&mut r, c, true, k % 32); }
        for k in 0..256 { x[c * 256 + k] = probe_code(&mut r, c, false, k % 32); }
        let (codes, sums) = probe_state(&mut r, c);
        s[c * 32..c * 32 + 24].copy_from_slice(&codes);
        old.extend(sums);
    }
    let (wb, xb, sb) = (Buf::new(g, &w)?, Buf::new(g, &x)?, Buf::new(g, &s)?);
    // The kernel reads the old sums from the output words and writes the new sums there.
    let out = Buf::new(g, &old.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let mut a = proj_bytes(&Proj { weights: wb.address, scales: sb.address, input: xb.address, output: out.address, rows: cases as u32, cols: 32, tokens: 8 });
    g.launch(f, [cases as u32, 1, 1], 32, 0, &mut [a.as_mut_ptr().cast()])?;
    g.sync()?;
    let device = out.words(g)?;
    let word = |c: usize, row: usize, token: usize| device[c * 128 + row * 8 + token];
    let model = |c: usize, row: usize, token: usize, m: (i32, bool, bool, i32)|
        twin::step_model(old[c * 128 + row * 8 + token], &w[c * 512 + row * 32..c * 512 + row * 32 + 32], &x[c * 256 + token * 32..c * 256 + token * 32 + 32], s[c * 32 + row], s[c * 32 + 16 + token], m);
    // Sign of a zero result: counted apart, because a +0.0 chain never shows it (fl(+0 + -0) = +0).
    let negative_zeros = device.iter().filter(|&&v| v == 0x8000_0000).count();
    // Coverage of the edges: results under 2^-126 that are not zero, NaN results, infinite results.
    let (tiny, nans, infinite) = (device.iter().filter(|&&v| v & 0x7f80_0000 == 0 && v & 0x007f_ffff != 0).count(), device.iter().filter(|&&v| v & 0x7fff_ffff > 0x7f80_0000).count(), device.iter().filter(|&&v| v & 0x7fff_ffff == 0x7f80_0000).count());
    println!("MXTC PROBE cases={cases} words={} device_negative_zero_words={negative_zeros} device_words_under_2^-126_not_zero={tiny} device_nan_words={nans} device_infinite_words={infinite}", device.len());
    // Every model on the first 1,024 cases (all 64 pairs of classes, both halves of the last one).
    let own = (twin::FRACTION, twin::ZEROS_COUNT, false, twin::LOW);
    let mut best = usize::MAX;
    let mut table = vec![];
    for low in [i32::MIN, -149, -151] { for nearest in [false, true] { for zeros in [true, false] { for fraction in 18..=40 {
        let m = (fraction, zeros, nearest, low);
        let mut differ = 0usize;
        for c in 0..1024 { for row in 0..16 { for token in 0..8 { if word(c, row, token) != model(c, row, token, m) { differ += 1; } } } }
        best = best.min(differ);
        table.push((m, differ));
    } } } }
    let floor = |low: i32| if low == i32::MIN { "none".to_string() } else { format!("2^{low}") };
    for (m, differ) in &table {
        if *differ == best || *m == own || (m.0, m.1, m.2) == (own.0, own.1, own.2) { println!("MXTC PROBE MODEL fraction={} zeros_count={} nearest={} cut_floor={} differ={differ} of {}", m.0, m.1, m.2, floor(m.3), 1024 * 128); }
    }
    // The twin's model on every case. A word under 2^-126 on both sides is counted apart (once the contract's open
    // edge; the twin's floor of the cut closes it, so that count must be 0 too).
    let small = |v: u32| v & 0x7f80_0000 == 0;
    let (mut differ, mut edge, mut shown, mut edge_shown) = (0usize, 0usize, 0, 0);
    let mut by_class = [0usize; 64];
    for c in 0..cases { for row in 0..16 { for token in 0..8 {
        let (have, want) = (word(c, row, token), model(c, row, token, own));
        if have == want { continue; }
        let open = small(have) && small(want);
        if open { edge += 1; } else { differ += 1; by_class[c % 64] += 1; }
        if (open && edge_shown < 4) || (!open && shown < 8) {
            if open { edge_shown += 1; } else { shown += 1; }
            println!("MXTC PROBE {} case={c} codes_class={} state_class={} row={row} token={token} device={have:08x} twin={want:08x} old={:08x} ws={} xs={} weight={:02x?} input={:02x?}", if open { "EDGE" } else { "DIFFER" }, c % 8, (c / 8) % 8,
                old[c * 128 + row * 8 + token], s[c * 32 + row], s[c * 32 + 16 + token], &w[c * 512 + row * 32..c * 512 + row * 32 + 32], &x[c * 256 + token * 32..c * 256 + token * 32 + 32]);
        }
    } } }
    if differ > 0 { println!("MXTC PROBE DIFFER by class pair (codes + 8 * state): {:?}", by_class.iter().enumerate().filter(|(_, n)| **n > 0).collect::<Vec<_>>()); }
    println!("MXTC PROBE STEP {} twin_model=fraction:{},zeros_count:{},cut_toward_zero,cut_floor:{} differ={differ} open_edge_under_2^-126={edge} of {}", if differ == 0 && edge == 0 { "PASS" } else { "FAIL" }, twin::FRACTION, twin::ZEROS_COUNT, floor(twin::LOW), cases * 128);
    // The open edge: the two floors of the cut on every case (0 words = that floor closes the edge).
    for low in [-149, -151] {
        let mut n = 0usize;
        for c in 0..cases { for row in 0..16 { for token in 0..8 { if word(c, row, token) != model(c, row, token, (own.0, own.1, own.2, low)) { n += 1; } } } }
        println!("MXTC PROBE EDGE MODEL cut_floor={} differ={n} of {} (the twin's own: {})", floor(low), cases * 128, differ + edge);
    }
    let wrong_nearest = table.iter().find(|(m, _)| *m == (own.0, own.1, true, own.3)).unwrap().1;
    println!("MXTC PROBE CONTROL {} wrong nearest rounding differ={wrong_nearest} of 131072", if wrong_nearest > 0 { "PASS" } else { "FAIL" });
    // The edge under 2^-126 is closed (the twin's floor of the cut): a word that differs there fails like any other.
    let step_ok = differ == 0 && edge == 0 && negative_zeros == 0 && wrong_nearest > 0;

    // The activation conversion: every E4M3 value (what a quantized group holds after its scale), then general values.
    let f = g.function(module, "mx_tc_probe_cvt")?;
    let mut exact: Vec<f32> = (0..=255u8).filter(|c| c & 0x7f != 0x7f).map(twin::e4m3_value).collect();
    exact.push(0.0);
    let mut general: Vec<f32> = vec![0.0, -0.0, f32::from_bits(1), -f32::from_bits(1), 1e30, -1e30, 448.0, 449.0, 464.0, 480.0, 1.0 / 1024.0, 1.5 / 1024.0];
    for c in 0..126u8 { let middle = ((f64::from(twin::e4m3_value(c)) + f64::from(twin::e4m3_value(c + 1))) / 2.0) as f32; general.extend([middle, -middle, f32::from_bits(middle.to_bits() + 1), f32::from_bits(middle.to_bits() - 1)]); }
    while general.len() < 1 << 18 { let h = r.next_u64(); general.push(f32::from_bits((h as u32 & 0x807f_ffff) | ((100 + (h >> 40) as u32 % 44) << 23))); }
    let mut ok = [true; 2];
    for (index, (name, values)) in [("quantized", &exact), ("general", &general)].into_iter().enumerate() {
        let pairs = values.len() / 2;
        let (xb, out) = (Buf::new(g, &le(&values[..pairs * 2]))?, Buf::poisoned(g, pairs)?);
        let mut a = proj_bytes(&Proj { weights: 0, scales: 0, input: xb.address, output: out.address, rows: pairs as u32, cols: 32, tokens: 1 });
        g.launch(f, [pairs.div_ceil(128) as u32, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])?;
        g.sync()?;
        let device = out.words(g)?;
        let (mut differ, mut shown) = (0usize, 0);
        for i in 0..pairs { for half in 0..2 {
            let (value, have) = (values[2 * i + half], (device[i] >> (8 * half)) as u8);
            let want = twin::e4m3_code(value);
            if have != want { differ += 1; if shown < 4 { shown += 1; println!("MXTC PROBE CVT DIFFER set={name} value={value:e} bits={:08x} device={have:02x} twin={want:02x}", value.to_bits()); } }
        } }
        println!("MXTC PROBE CVT {} set={name} values={} differ={differ}", if differ == 0 { "PASS" } else { "FAIL" }, pairs * 2);
        ok[index] = differ == 0;
    }
    g.clear()?;
    // The general set is not part of the served contract (the quantizer never writes such values); it is reported.
    Ok(step_ok && ok[0])
}

// ------------------------------------------------------------------ gates
fn gates(g: &mut Gpu, new: &New) -> Result<bool> {
    let started = Instant::now();
    let mut pass = true;
    // G1: every word = twin on small shapes.
    let (mut checked, mut differ, mut fixtures) = (0usize, 0usize, 0usize);
    let mut seed = 100u64;
    for &rows in &[1usize, 15, 16, 17, 63, 64, 65, 130] { for &cols in &[32usize, 64, 96, 128, 160, 256, 1024, 4096] { for &tokens in &[1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 65] {
        if rows * cols * tokens > 12_000_000 { continue; }
        seed += 1;
        let f = fixture(rows, cols, tokens, seed, seed % 3 == 0);
        let d = upload(g, &f)?;
        let device = device_words(g, new, &f, &d)?;
        let (c, n, first) = against_twin(&f, &device, None);
        checked += c; differ += n; fixtures += 1;
        if let Some((t, row, have, want)) = first { if differ == n { println!("MXTC G1 FIRST DIFFER rows={rows} cols={cols} tokens={tokens} token={t} row={row} device={have:08x} twin={want:08x}"); } }
        g.clear()?;
    } } }
    println!("MXTC G1 {} device=twin small shapes fixtures={fixtures} words={checked} differ={differ} elapsed_s={:.1}", if differ == 0 { "PASS" } else { "FAIL" }, started.elapsed().as_secs_f64());
    pass &= differ == 0;
    // G1 on the model's shapes (a rank's rows): sampled words, the last row included.
    let (mut checked, mut differ) = (0usize, 0usize);
    for &(rows, cols) in &[(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] { for &tokens in &[1usize, 2, 4, 8, 23, 1024] {
        seed += 1;
        let f = fixture(rows, cols, tokens, seed, false);
        let d = upload(g, &f)?;
        let device = device_words(g, new, &f, &d)?;
        let (c, n, first) = against_twin(&f, &device, Some(48));
        checked += c; differ += n;
        if let Some((t, row, have, want)) = first { println!("MXTC G1 MODEL SHAPE DIFFER rows={rows} cols={cols} tokens={tokens} token={t} row={row} device={have:08x} twin={want:08x}"); }
        g.clear()?;
    } }
    println!("MXTC G1 {} device=twin model shapes sampled words={checked} differ={differ} elapsed_s={:.1}", if differ == 0 { "PASS" } else { "FAIL" }, started.elapsed().as_secs_f64());
    pass &= differ == 0;

    // G2: a token row alone = the row in every group, device against device.
    let (mut compared, mut differ) = (0usize, 0usize);
    for &(rows, cols, tokens) in &[(130usize, 1024usize, 67usize), (2048, 8192, 19), (2048, 12288, 12), (17, 128, 1025)] {
        let f = fixture(rows, cols, tokens, 7000 + rows as u64, false);
        let d = upload(g, &f)?;
        // Alone: one token a launch.
        let alone = Buf::poisoned(g, rows * tokens)?;
        for t in 0..tokens {
            launch_new(g, new, &Proj { weights: d.w.address, scales: d.s.address, input: d.x.address + (t * cols * 4) as u64, output: alone.address + (t * rows * 4) as u64, rows: rows as u32, cols: cols as u32, tokens: 1 }, 0)?;
        }
        // The packed rows of a group: each launch has its own lines of this plane.
        let plane = Buf::new(g, &vec![0xee; tokens * line(cols)])?;
        g.sync()?;
        let alone = alone.words(g)?;
        // Three placements: in order, reversed, rotated by 5. Each is its own input buffer, cut into groups of `size`.
        for placement in 0..3usize {
            let order: Vec<usize> = (0..tokens).map(|i| match placement { 0 => i, 1 => tokens - 1 - i, _ => (i + 5) % tokens }).collect();
            let moved: Vec<f32> = order.iter().flat_map(|&t| f.x[t * cols..(t + 1) * cols].iter().copied()).collect();
            let xb = Buf::new(g, &le(&moved))?;
            for &size in &[2usize, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64, 67, 256, 1024] {
                if size > tokens { continue; }
                let out = Buf::poisoned(g, rows * tokens)?;
                let mut at = 0;
                while at < tokens {
                    let count = size.min(tokens - at);
                    launch_new(g, new, &Proj { weights: d.w.address, scales: d.s.address, input: xb.address + (at * cols * 4) as u64, output: out.address + (at * rows * 4) as u64, rows: rows as u32, cols: cols as u32, tokens: count as u32 }, plane.address + (at * line(cols)) as u64)?;
                    at += count;
                }
                g.sync()?;
                let group = out.words(g)?;
                for (slot, &t) in order.iter().enumerate() { for row in 0..rows {
                    compared += 1;
                    if !same(group[slot * rows + row], alone[t * rows + row]) {
                        if differ == 0 { println!("MXTC G2 FIRST DIFFER rows={rows} cols={cols} placement={placement} group_size={size} token={t} slot={slot} row={row} group={:08x} alone={:08x}", group[slot * rows + row], alone[t * rows + row]); }
                        differ += 1;
                    }
                } }
            }
        }
        g.clear()?;
    }
    println!("MXTC G2 {} alone=group words={compared} differ={differ} group_sizes=2,3,4,5,7,8,9,15,16,17,31,32,64,67,256,1024 placements=3 elapsed_s={:.1}", if differ == 0 { "PASS" } else { "FAIL" }, started.elapsed().as_secs_f64());
    pass &= differ == 0;

    // G3: controls. Each must be caught.
    let mut f = fixture(130, 1024, 9, 9001, false);
    let d = upload(g, &f)?;
    let device = device_words(g, new, &f, &d)?;
    let (_, clean, _) = against_twin(&f, &device, None);
    // (a) one activation value of token 4 doubled on the twin's side only: the device words of token 4 must differ, the other tokens must not.
    f.x[4 * 1024 + 40] = f.x[4 * 1024 + 41] * 2.0 + f.x[4 * 1024 + 40];
    let (mut hit, mut stray) = (0usize, 0usize);
    for t in 0..9 { let encoded = twin::encode_row(&f.x[t * 1024..(t + 1) * 1024]); for row in 0..130 {
        if !same(device[t * 130 + row], twin_word(&f, &encoded, row)) { if t == 4 { hit += 1; } else { stray += 1; } }
    } }
    let control_a = clean == 0 && hit > 0 && stray == 0;
    println!("MXTC G3 CONTROL {} wrong activation on the twin: clean_differ={clean} token_4_words_that_differ={hit} of 130 other_tokens={stray}", if control_a { "PASS" } else { "FAIL" });
    // (b) the join as one chain over all groups in place of four ranges: must differ somewhere.
    let f = fixture(130, 1024, 9, 9001, false);
    let mut wrong = 0usize;
    for t in 0..9 { let encoded = twin::encode_row(&f.x[t * 1024..(t + 1) * 1024]); for row in 0..130 {
        let chain = twin::chain(&f.w[row * 1024..(row + 1) * 1024], &f.s[row * 32..(row + 1) * 32], &encoded, 0, 32);
        if !same(device[t * 130 + row], chain) { wrong += 1; }
    } }
    println!("MXTC G3 CONTROL {} one chain in place of four ranges: words_that_differ={wrong} of 1170", if wrong > 0 { "PASS" } else { "FAIL" });
    // (b2) a range cut another way than the contract's sub-chains of 8 groups: one chain a range, sub-chains of 4, of 16.
    // Each must differ from the device somewhere (at 4,096 columns a range is 32 groups: four sub-chains).
    let control_b2 = {
        let f = fixture(130, 4096, 9, 9002, false);
        let d = upload(g, &f)?;
        let device = device_words(g, new, &f, &d)?;
        let (_, clean, _) = against_twin(&f, &device, None);
        let subs = [32usize, 4, 16];
        let mut wrong = [0usize; 3];
        for t in 0..9 { let encoded = twin::encode_row(&f.x[t * 4096..(t + 1) * 4096]); for row in 0..130 {
            let (w, sc) = (&f.w[row * 4096..(row + 1) * 4096], &f.s[row * 128..(row + 1) * 128]);
            for (slot, &sub) in subs.iter().enumerate() {
                let part: [f32; 4] = std::array::from_fn(|part| { let (start, end) = twin::range(128, part); f32::from_bits(twin::part_encoded(w, sc, &encoded, start, end, sub)) });
                if !same(device[t * 130 + row], ((part[0] + part[1]) + (part[2] + part[3])).to_bits()) { wrong[slot] += 1; }
            }
        } }
        let ok = clean == 0 && wrong.iter().all(|&n| n > 0);
        println!("MXTC G3 CONTROL {} a range cut another way than sub-chains of {}: device=twin differ={clean}; words_that_differ of 1170: one_chain_a_range={} sub_chains_of_4={} sub_chains_of_16={}", if ok { "PASS" } else { "FAIL" }, twin::SUB, wrong[0], wrong[1], wrong[2]);
        ok
    };
    // (c) the weight scale codes of row 77 changed on the device's side only (every group, by one): row 77 must change for
    // every token, no other row may.
    let mut scales = f.s.clone();
    for group in 0..32 { scales[77 * 32 + group] ^= 1; }
    g.upload_at(d.s.address, &scales)?;
    let moved = device_words(g, new, &f, &d)?;
    let (rows_hit, others) = ((0..9).filter(|t| moved[t * 130 + 77] != device[t * 130 + 77]).count(), (0..9 * 130).filter(|i| i % 130 != 77 && moved[*i] != device[*i]).count());
    let control_c = rows_hit == 9 && others == 0;
    println!("MXTC G3 CONTROL {} the weight scale codes of one row changed on the device: row_77_tokens_that_differ={rows_hit} of 9 other_rows={others}", if control_c { "PASS" } else { "FAIL" });
    g.clear()?;
    pass &= control_a && wrong > 0 && control_b2 && control_c;

    // G4: the packed rows.
    let Some((pack, packed_tile)) = new.packed else { println!("MXTC G4 SKIPPED: the image has no mx_tc_pack and mx_tc_tile_packed"); return Ok(pass); };
    // (a) every byte of the plane = the twin's packed row. A group with a value that is not finite: scale byte 255, codes not compared.
    let (mut bytes, mut differ, mut not_finite, mut planted) = (0usize, 0usize, 0usize, 0usize);
    for &(cols, tokens) in &[(1024usize, 1usize), (1024, 9), (1024, 70), (4096, 65), (8192, 300), (12288, 33)] {
        let mut f = fixture(1, cols, tokens, 4000 + (cols + tokens) as u64, tokens % 2 == 0);
        // Three groups with a value that is not finite: a NaN, an infinity in the last column, a negative infinity.
        if tokens > 5 { f.x[3 * cols + 40] = f32::NAN; f.x[5 * cols + cols - 1] = f32::INFINITY; f.x[5 * cols + 70] = f32::NEG_INFINITY; planted += 3; }
        let xb = Buf::new(g, &le(&f.x))?;
        let plane = Buf::new(g, &vec![0xee; tokens * line(cols)])?;
        launch_pack(g, pack, &Proj { weights: 0, scales: 0, input: xb.address, output: 0, rows: 1, cols: cols as u32, tokens: tokens as u32 }, plane.address)?;
        g.sync()?;
        let device = plane.read(g)?;
        for t in 0..tokens {
            let (have, want) = (&device[t * line(cols)..(t + 1) * line(cols)], twin::pack_row(&f.x[t * cols..(t + 1) * cols]));
            for group in 0..cols / 32 {
                bytes += 1;
                if have[cols + group] != want[cols + group] { differ += 1; }
                if want[cols + group] == twin::NOT_FINITE { not_finite += 1; continue; }
                bytes += 32;
                differ += (0..32).filter(|k| have[group * 32 + k] != want[group * 32 + k]).count();
            }
        }
        g.clear()?;
    }
    println!("MXTC G4 {} packed rows=twin bytes={bytes} differ={differ} groups_not_finite={not_finite} planted={planted}", if differ == 0 && not_finite == planted { "PASS" } else { "FAIL" });
    pass &= differ == 0 && not_finite == planted;
    // (b) the tile on packed rows = the tile on FP32 rows, every word, device against device, on the model's shapes.
    let (mut words, mut differ) = (0usize, 0usize);
    if let Some(tile) = new.tile {
        for &(rows, cols) in &[(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] { for &tokens in &[9usize, 64, 65, 1024] {
            let f = fixture(rows, cols, tokens, 4100 + (rows + tokens) as u64, false);
            let d = upload(g, &f)?;
            let (plain, packed, plane) = (Buf::poisoned(g, rows * tokens)?, Buf::poisoned(g, rows * tokens)?, Buf::new(g, &vec![0xee; tokens * line(cols)])?);
            let p = Proj { weights: d.w.address, scales: d.s.address, input: d.x.address, output: plain.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
            let mut a = proj_bytes(&p);
            g.launch(tile, [p.tokens.div_ceil(64), p.rows.div_ceil(128), 1], 256, 0, &mut [a.as_mut_ptr().cast()])?;
            launch_pack(g, pack, &p, plane.address)?;
            launch_packed_tile(g, packed_tile, &Proj { output: packed.address, ..p }, plane.address)?;
            g.sync()?;
            let (plain, packed) = (plain.words(g)?, packed.words(g)?);
            words += plain.len();
            let n = (0..plain.len()).filter(|&i| plain[i] != packed[i]).count();
            if n > 0 && differ == 0 { let i = (0..plain.len()).find(|&i| plain[i] != packed[i]).unwrap(); println!("MXTC G4 FIRST DIFFER rows={rows} cols={cols} tokens={tokens} token={} row={} fp32_rows={:08x} packed_rows={:08x}", i / rows, i % rows, plain[i], packed[i]); }
            differ += n;
            g.clear()?;
        } }
        println!("MXTC G4 {} tile on packed rows=tile on FP32 rows words={words} differ={differ}", if differ == 0 { "PASS" } else { "FAIL" });
        pass &= differ == 0;

    }
    // (d) values that are not finite, through the packed path: every word = the twin (NaN payloads count as equal), the
    // words of a token with such a group are NaN, and every other token's word is what it is without the planted values
    // (the fixture's own large values give some NaN words: an infinity of each sign in two ranges).
    {
        let mut f = fixture(130, 4096, 70, 4300, false);
        let clean = { let d = upload(g, &f)?; device_words(g, new, &f, &d)? };
        for (token, column) in [(3usize, 40usize), (9, 4095), (64, 2048), (69, 0)] { f.x[token * 4096 + column] = if token % 2 == 1 { f32::NAN } else { f32::NEG_INFINITY }; }
        let d = upload(g, &f)?;
        let device = device_words(g, new, &f, &d)?;
        let (checked, differ, first) = against_twin(&f, &device, None);
        let planted = [3usize, 9, 64, 69];
        let (nan_planted, moved_other) = (planted.iter().map(|&t| (0..130).filter(|row| f32::from_bits(device[t * 130 + row]).is_nan()).count()).sum::<usize>(), (0..70 * 130).filter(|i| !planted.contains(&(i / 130)) && device[*i] != clean[*i]).count());
        if let Some((t, row, have, want)) = first { println!("MXTC G4 NOT FINITE FIRST DIFFER token={t} row={row} device={have:08x} twin={want:08x}"); }
        let ok = differ == 0 && nan_planted == 4 * 130 && moved_other == 0;
        println!("MXTC G4 {} values that are not finite through the packed rows: words={checked} differ_from_twin={differ} nan_words_of_the_4_planted_tokens={nan_planted} of 520 words_of_other_tokens_that_moved={moved_other}", if ok { "PASS" } else { "FAIL" });
        pass &= ok;
        g.clear()?;
    }
    // (c) control: one scale byte of token 0 changed in the plane: words of token 0 must change, no other token's may.
    // (Token 0 has activation class 0: every group counts in the word. Token 2 has tiny scales: a dead control.)
    let f = fixture(130, 1024, 9, 9001, false);
    let d = upload(g, &f)?;
    let (out, plane) = (Buf::poisoned(g, 130 * 9)?, Buf::new(g, &vec![0xee; 9 * line(1024)])?);
    let p = Proj { weights: d.w.address, scales: d.s.address, input: d.x.address, output: out.address, rows: 130, cols: 1024, tokens: 9 };
    launch_pack(g, pack, &p, plane.address)?;
    launch_packed_tile(g, packed_tile, &p, plane.address)?;
    g.sync()?;
    let clean = out.words(g)?;
    let mut bytes = plane.read(g)?;
    bytes[1024 + 5] ^= 3;
    g.upload_at(plane.address, &bytes)?;
    launch_packed_tile(g, packed_tile, &p, plane.address)?;
    g.sync()?;
    let moved = out.words(g)?;
    let (hit, stray) = ((0..130).filter(|&row| moved[row] != clean[row]).count(), (130..9 * 130).filter(|&i| moved[i] != clean[i]).count());
    println!("MXTC G4 CONTROL {} one scale byte of token 0 changed in the packed rows: token_0_words_that_differ={hit} of 130 other_tokens={stray}", if hit > 0 && stray == 0 { "PASS" } else { "FAIL" });
    g.clear()?;
    pass &= hit > 0 && stray == 0;
    Ok(pass)
}

// ------------------------------------------------------------------ time
/// Today's kernels of the base image, as the engine picks them by token count (`small_choice::mx`).
struct Old { m1: H, small2: H, small4: H, small8: H, small8w: H, tile: H }
fn launch_old(g: &Gpu, old: &Old, p: &Proj) -> Result<()> {
    let mut a = proj_bytes(p);
    match p.tokens {
        1 => {
            // `MxLayer`: nine projections, ten ends, the count.
            let blocks = p.rows.div_ceil(4);
            let mut layer = proj_bytes(p);
            layer.resize(9 * 56, 0);
            let mut ends = [0u32; 10];
            ends[1] = blocks;
            for e in ends { layer.extend(e.to_le_bytes()); }
            layer.extend(1u32.to_le_bytes());
            layer.extend([0u8; 4]);
            g.launch(old.m1, [blocks, 1, 1], 128, 0, &mut [layer.as_mut_ptr().cast()])
        }
        2 => g.launch(old.small2, [p.rows.div_ceil(8), 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()]),
        3 | 4 => g.launch(old.small4, [p.rows.div_ceil(16), 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()]),
        5..=8 => if p.cols > 4096 { g.launch(old.small8w, [p.rows.div_ceil(16), 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()]) } else { g.launch(old.small8, [p.rows.div_ceil(8), 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()]) },
        _ => {
            // The prompt tile: whole to 1,024 tokens through 16 MiB of weight codes, else pieces of 256 tokens (`mx_rows::token_limit`).
            let limit = if p.rows as usize * p.cols as usize <= 16 << 20 { 1024 } else { 256 };
            let mut first = 0;
            while first < p.tokens {
                let q = Proj { tokens: (p.tokens - first).min(limit), input: p.input + u64::from(first) * u64::from(p.cols) * 4, output: p.output + u64::from(first) * u64::from(p.rows) * 4, ..*p };
                let mut a = proj_bytes(&q);
                g.launch(old.tile, [q.rows.div_ceil(48), q.tokens.div_ceil(16), 1], 384, 0, &mut [a.as_mut_ptr().cast()])?;
                first += limit;
            }
            Ok(())
        }
    }
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn time(g: &mut Gpu, old: &Old, new: &New) -> Result<()> {
    for &(rows, cols) in &[(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] {
        for &tokens in &[1usize, 2, 4, 8, 256, 1024] {
            let f = fixture(rows, cols, tokens, 31, false);
            let d = upload(g, &f)?;
            let out = Buf::poisoned(g, rows * tokens)?;
            let p = Proj { weights: d.w.address, scales: d.s.address, input: d.x.address, output: out.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
            // A packed call: `new` is the tile on rows that are packed already (the engine packs a shared input once for
            // all its matrices), `pack` the one pass that packs them.
            let plane = plane_for(g, new, &p)?;
            if let (true, Some((pack, _))) = (plane != 0, new.packed) { launch_pack(g, pack, &p, plane)?; }
            let arm_launch = |g: &Gpu, arm: usize| -> Result<()> {
                match (arm, new.packed) {
                    (0, _) => launch_old(g, old, &p),
                    (1, Some((_, tile))) if plane != 0 => launch_packed_tile(g, tile, &p, plane),
                    (1, _) => launch_new(g, new, &p, 0),
                    (_, Some((pack, _))) if plane != 0 => launch_pack(g, pack, &p, plane),
                    _ => Ok(()),
                }
            };
            let mut times = [vec![], vec![], vec![]];
            let n = if tokens <= 8 { 20 } else { 3 };
            for repeat in 0..7 {
                for arm in if repeat % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                    if arm == 2 && plane == 0 { continue; }
                    arm_launch(g, arm)?;
                    g.sync()?;
                    let t0 = Instant::now();
                    for _ in 0..n { arm_launch(g, arm)?; }
                    g.sync()?;
                    times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
                }
            }
            let (ra, rb) = (median(times[0].clone()), median(times[1].clone()));
            let pack_us = if plane != 0 { format!("{:.1}", median(times[2].clone())) } else { "none".to_string() };
            let macs = (rows * cols * tokens) as f64;
            println!("MXTC TIME rows={rows} cols={cols} tokens={tokens} old_us={ra:.1} new_us={rb:.1} pack_us={pack_us} old_over_new={:.3} old_GMACs={:.0} new_GMACs={:.0} weight_MiB={:.1}", ra / rb, macs / ra / 1e3, macs / rb / 1e3, (rows * cols) as f64 / 1048576.0);
            g.clear()?;
        }
    }
    Ok(())
}

/// TILE: the prompt tile alone on one matrix, again and again (warm): the token rows of ONE launch from 64 to 1,024, and
/// 1,024 token rows cut into launches of 64 to 512 (the host's piece size, `mx_rows::token_limit`). Same words at every
/// size (G2). us for the stated rows and GMAC/s.
fn tile_sweep(g: &mut Gpu, new: &New) -> Result<()> {
    let Some(tile) = new.tile else { return Err("MXTC TILE: the image has no mx_tc_tile".into()) };
    for form in 0..2 { for &(rows, cols) in &[(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] {
        // Form 0: the tile on FP32 rows. Form 1: the tile on packed rows (packed once, before the clock).
        let (label, tile) = match (form, new.packed) { (0, _) => ("fp32_rows", tile), (_, Some((_, packed))) => ("packed_rows", packed), _ => continue };
        let f = fixture(rows, cols, 1024, 33, false);
        let d = upload(g, &f)?;
        let out = Buf::poisoned(g, rows * 1024)?;
        let (mut input, mut stride) = (d.x.address, cols * 4);
        if let (1, Some((pack, _))) = (form, new.packed) {
            let plane = Buf::new(g, &vec![0xee; 1024 * line(cols)])?;
            launch_pack(g, pack, &Proj { weights: 0, scales: 0, input: d.x.address, output: 0, rows: rows as u32, cols: cols as u32, tokens: 1024 }, plane.address)?;
            g.sync()?;
            (input, stride) = (plane.address, line(cols));
        }
        let launch = |g: &Gpu, first: usize, count: usize| -> Result<()> {
            let mut a = proj_bytes(&Proj { weights: d.w.address, scales: d.s.address, input: input + (first * stride) as u64, output: out.address + (first * rows * 4) as u64, rows: rows as u32, cols: cols as u32, tokens: count as u32 });
            g.launch(tile, [(count as u32).div_ceil(64), (rows as u32).div_ceil(128), 1], 256, 0, &mut [a.as_mut_ptr().cast()])
        };
        // (token rows a launch, launches): whole launches first, then 1,024 rows in pieces.
        let cases: [(usize, usize); 11] = [(64, 1), (128, 1), (256, 1), (384, 1), (512, 1), (768, 1), (1024, 1), (64, 16), (128, 8), (256, 4), (512, 2)];
        let mut times = vec![vec![]; cases.len()];
        for repeat in 0..7 {
            for index in 0..cases.len() {
                let at = if repeat % 2 == 0 { index } else { cases.len() - 1 - index };
                let (count, pieces) = cases[at];
                let run = |g: &Gpu| -> Result<()> { for piece in 0..pieces { launch(g, piece * count, count)?; } Ok(()) };
                run(g)?;
                g.sync()?;
                let t0 = Instant::now();
                for _ in 0..3 { run(g)?; }
                g.sync()?;
                times[at].push(t0.elapsed().as_secs_f64() * 1e6 / 3.0);
            }
        }
        for (index, &(count, pieces)) in cases.iter().enumerate() {
            let us = median(times[index].clone());
            let macs = (rows * cols * count * pieces) as f64;
            println!("MXTC TILE input={label} rows={rows} cols={cols} tokens_a_launch={count} launches={pieces} token_rows={} us={us:.1} GMACs={:.0}", count * pieces, macs / us / 1e3);
        }
        g.clear()?;
    } }
    Ok(())
}

/// The decode step's view: a pass of the model's 145 MXFP8 calls of one rank (q, k, v 34 layers each, o 34, dense gate
/// and up 3 each, down 3), every call on its own weights, so the weights come from memory and not from the cache. Buffers
/// start at their allocation, as the engine's do. us a launch, GB/s of weight bytes, and ms a pass.
fn cold(g: &mut Gpu, old: &Old, new: &New) -> Result<()> {
    let shapes: [(usize, usize, f64, &str); 4] = [(4096, 4096, 102.0, "qkv"), (2048, 8192, 34.0, "o"), (6144, 4096, 6.0, "gate_up"), (2048, 12288, 3.0, "down")];
    let mut totals = vec![];
    for &(rows, cols, calls, label) in &shapes {
        let bytes = (rows * cols) as f64 * (1.0 + 1.0 / 32.0);
        let k = ((150e6 / bytes).ceil() as usize).clamp(2, 32);
        for &tokens in &[1usize, 2, 4, 8] {
            let mut ps = vec![];
            for i in 0..k {
                let f = fixture(rows, cols, tokens, 500 + i as u64, false);
                let out = g.alloc(rows * tokens * 4)?;
                ps.push(Proj { weights: g.upload(&f.w)?, scales: g.upload(&f.s)?, input: g.upload(&le(&f.x))?, output: out, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 });
            }
            let per = 4 * k;
            let (mut ta, mut tb) = (vec![], vec![]);
            for repeat in 0..7 {
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    for p in &ps { if arm == 0 { launch_old(g, old, p)?; } else { launch_new(g, new, p, 0)?; } }
                    g.sync()?;
                    let t0 = Instant::now();
                    for i in 0..per { if arm == 0 { launch_old(g, old, &ps[i % k])?; } else { launch_new(g, new, &ps[i % k], 0)?; } }
                    g.sync()?;
                    (if arm == 0 { &mut ta } else { &mut tb }).push(t0.elapsed().as_secs_f64() * 1e6 / per as f64);
                }
            }
            let (ra, rb) = (median(ta), median(tb));
            println!("MXTC COLD {label} rows={rows} cols={cols} tokens={tokens} copies={k} old_us={ra:.1} ({:.0} GB/s) new_us={rb:.1} ({:.0} GB/s) old_over_new={:.3} floor_us_at_235GBs={:.0}", bytes / ra / 1e3, bytes / rb / 1e3, ra / rb, bytes / 235e3);
            totals.push((tokens, ra * calls / 1e3, rb * calls / 1e3));
            g.clear()?;
        }
    }
    for &tokens in &[1usize, 2, 4, 8] {
        let (a, b) = totals.iter().filter(|t| t.0 == tokens).fold((0.0, 0.0), |s, t| (s.0 + t.1, s.1 + t.2));
        println!("MXTC COLD PASS tokens={tokens} calls=145 old_ms={a:.2} new_ms={b:.2} old_over_new={:.3}", a / b);
    }
    Ok(())
}


// ------------------------------------------------------------------ variants of the row entry
/// The row entries of the candidate image, the contract's `mx_tc_rows` first: the same grid, the same output.
const VARIANTS: [&str; 7] = ["mx_tc_rows", "mx_tc_rows_b0", "mx_tc_rows_b1s1", "mx_tc_rows_b2s1", "mx_tc_rows_b3s2", "mx_tc_rows_b4", "mx_tc_rows_b4s4"];
fn launch_rows(g: &Gpu, entry: H, p: &Proj) -> Result<()> {
    let mut a = proj_bytes(p);
    g.launch(entry, [p.rows.div_ceil(16), p.tokens.div_ceil(8), 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
/// Every variant against `mx_tc_rows`, every word, on shapes off and on the tile sizes, and a control: a variant fed
/// another input row must differ.
fn variants_same(g: &mut Gpu, vars: &[(&'static str, H)]) -> Result<bool> {
    let mut pass = true;
    let shapes: [(usize, usize, usize, bool); 12] = [(130, 4096, 1, false), (130, 4096, 3, true), (67, 8192, 8, false), (40, 12288, 1, true), (33, 96, 5, false), (17, 32, 1, false),
        (50, 64, 2, true), (16, 160, 7, false), (35, 128, 1, true), (48, 512, 4, false), (4096, 4096, 1, false), (2048, 12288, 2, false)];
    let (mut words, mut differ) = (vec![0usize; vars.len()], vec![0usize; vars.len()]);
    let mut control = 0usize;
    for (index, &(rows, cols, tokens, far)) in shapes.iter().enumerate() {
        let f = fixture(rows, cols, tokens, 900 + index as u64, far);
        let d = upload(g, &f)?;
        let mut first: Vec<u32> = vec![];
        for (v, &(_, entry)) in vars.iter().enumerate() {
            let out = Buf::poisoned(g, rows * tokens)?;
            let p = Proj { weights: d.w.address, scales: d.s.address, input: d.x.address, output: out.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
            launch_rows(g, entry, &p)?;
            g.sync()?;
            let have = out.words(g)?;
            if v == 0 { first = have; continue; }
            words[v] += have.len();
            differ[v] += have.iter().zip(&first).filter(|(a, b)| !same(**a, **b)).count();
            if index == 0 {
                // The control: the same variant on a changed input must not give the same words.
                let mut x = f.x.clone();
                for c in 0..cols { x[c] = -x[c] * 1.5; }
                g.upload_at(d.x.address, &le(&x))?;
                launch_rows(g, entry, &p)?;
                g.sync()?;
                control += out.words(g)?.iter().zip(&first).filter(|(a, b)| !same(**a, **b)).count();
                g.upload_at(d.x.address, &le(&f.x))?;
            }
        }
        g.clear()?;
    }
    for (v, &(name, _)) in vars.iter().enumerate().skip(1) {
        let ok = differ[v] == 0 && words[v] > 0;
        println!("MXTC VARIANT {} {name} = mx_tc_rows on {} words of 12 shapes, words_that_differ={}", if ok { "PASS" } else { "FAIL" }, words[v], differ[v]);
        pass &= ok;
    }
    println!("MXTC VARIANT CONTROL {} a changed input row moves {control} words", if control > 0 || vars.len() < 2 { "PASS" } else { "FAIL" });
    Ok(pass && (control > 0 || vars.len() < 2))
}
/// The decode step's view (as `cold`), every variant in one pass over the same weights: us a launch and ms a pass of the
/// 145 calls. Arm 0 is today's kernel of the base image.
fn variants_cold(g: &mut Gpu, old: &Old, vars: &[(&'static str, H)]) -> Result<()> {
    let shapes: [(usize, usize, f64, &str); 4] = [(4096, 4096, 102.0, "qkv"), (2048, 8192, 34.0, "o"), (6144, 4096, 6.0, "gate_up"), (2048, 12288, 3.0, "down")];
    let arms = vars.len() + 1;
    let mut totals = vec![[0.0f64; 4]; arms];
    for &(rows, cols, calls, label) in &shapes {
        let bytes = (rows * cols) as f64 * (1.0 + 1.0 / 32.0);
        let k = ((150e6 / bytes).ceil() as usize).clamp(2, 32);
        for (ti, &tokens) in [1usize, 2, 4, 8].iter().enumerate() {
            let mut ps = vec![];
            for i in 0..k {
                let f = fixture(rows, cols, tokens, 500 + i as u64, false);
                let out = g.alloc(rows * tokens * 4)?;
                ps.push(Proj { weights: g.upload(&f.w)?, scales: g.upload(&f.s)?, input: g.upload(&le(&f.x))?, output: out, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 });
            }
            let per = 4 * k;
            let mut times = vec![vec![]; arms];
            for repeat in 0..7 {
                for step in 0..arms {
                    let arm = (step + repeat) % arms;
                    let go = |g: &Gpu, p: &Proj| if arm == 0 { launch_old(g, old, p) } else { launch_rows(g, vars[arm - 1].1, p) };
                    for p in &ps { go(g, p)?; }
                    g.sync()?;
                    let t0 = Instant::now();
                    for i in 0..per { go(g, &ps[i % k])?; }
                    g.sync()?;
                    times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / per as f64);
                }
            }
            let medians: Vec<f64> = times.into_iter().map(median).collect();
            let mut line = format!("MXTC VCOLD {label} rows={rows} cols={cols} tokens={tokens} floor_us_at_235GBs={:.0} old_us={:.1}", bytes / 235e3, medians[0]);
            for (v, &(name, _)) in vars.iter().enumerate() { line += &format!(" {name}={:.1}", medians[v + 1]); }
            println!("{line}");
            for arm in 0..arms { totals[arm][ti] += medians[arm] * calls / 1e3; }
            g.clear()?;
        }
    }
    for (ti, tokens) in [1usize, 2, 4, 8].iter().enumerate() {
        let mut line = format!("MXTC VCOLD PASS tokens={tokens} calls=145 old_ms={:.2}", totals[0][ti]);
        for (v, &(name, _)) in vars.iter().enumerate() { line += &format!(" {name}={:.2}", totals[v + 1][ti]); }
        println!("{line}");
    }
    Ok(())
}

// ------------------------------------------------------------------ wired: the served entries on a tensor-core descriptor
const MX_TC: u32 = 1;
/// `MxProjection` with its word function (`mode`, the four bytes after `round`).
fn proj_mode(p: &Proj, mode: u32) -> Vec<u8> {
    let mut v = proj_bytes(p);
    v[52..56].copy_from_slice(&mode.to_le_bytes());
    v
}
/// `MxLayer` (552 bytes: nine projections, ten ends, the count) of the jobs (descriptor, mode, blocks); with `batch` a
/// `ResidentProjectionBatch` (560 bytes: the layer, no plain job, round 0). Returns the bytes and the grid.
fn layer_bytes(jobs: &[(Proj, u32, u32)], batch: bool) -> (Vec<u8>, u32) {
    let mut v = vec![];
    for (p, mode, _) in jobs { v.extend(proj_mode(p, *mode)); }
    v.resize(9 * 56, 0);
    let mut ends = [0u32; 10];
    for (i, job) in jobs.iter().enumerate() { ends[i + 1] = ends[i] + job.2; }
    for e in ends { v.extend(e.to_le_bytes()); }
    v.extend((jobs.len() as u32).to_le_bytes());
    v.extend([0u8; 4]);
    if batch { v.extend([0u8; 8]); }
    (v, ends[jobs.len()])
}
/// One launch of `bytes`: direct (the by-value argument of the entry) or, with a kind, as the payload of command 0 of a
/// recorded table through the replay entry of that kind (`resident_launch_*`; 16 bytes of header, 1,024 of payload).
fn launch_bytes(g: &mut Gpu, h: H, replay: Option<u32>, bytes: &[u8], grid: u32) -> Result<()> {
    match replay {
        None => {
            let mut a = bytes.to_vec();
            g.launch(h, [grid, 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])
        }
        Some(kind) => {
            let mut command = vec![0u8; 1040];
            command[..4].copy_from_slice(&kind.to_le_bytes());
            command[16..16 + bytes.len()].copy_from_slice(bytes);
            let (mut table, mut index) = (g.upload(&command)?, 0u64);
            g.launch(h, [grid, 1, 1], 128, 0, &mut [(&mut table as *mut u64).cast(), (&mut index as *mut u64).cast()])
        }
    }
}
/// The BF16 epilogue of the merged launches (RN-even to the upper 16 bits; a non-finite word stays).
fn rb(bits: u32) -> u32 { if bits & 0x7f80_0000 == 0x7f80_0000 { bits } else { bits.wrapping_add(0x7fff + (bits >> 16 & 1)) & 0xffff_0000 } }
/// Every word of the device against the twin, raw or after the BF16 epilogue: (words, words that differ).
fn against_twin_rounded(f: &Fixture, x: &[f32], device: &[u32], rounded: bool) -> (usize, usize) {
    let mut differ = 0;
    for t in 0..f.tokens {
        let encoded = twin::encode_row(&x[t * f.cols..(t + 1) * f.cols]);
        for row in 0..f.rows {
            let want = twin_word(f, &encoded, row);
            if !same(device[t * f.rows + row], if rounded { rb(want) } else { want }) { differ += 1; }
        }
    }
    (f.rows * f.tokens, differ)
}
/// WIRED: every entry of the candidate image that takes an MXFP8 descriptor, given a descriptor with `mode` = tensor-core,
/// must store the twin's word (raw, or BF16-rounded in the merged launches), for every row and token row, also with a
/// scalar job between two tensor-core jobs of one launch. The scalar job must keep the words the BASE image gives it
/// (at 5 to 8 rows of the row stage, where the base image's stage entry traps: the words the base image's one-row entry
/// gives each source row alone).
/// Each served entry is launched directly AND as a recorded command through its replay entry (the served decode path).
/// Controls: the two word functions differ on these fixtures (so the mode is what decides), and a twin fed a changed
/// input row is caught.
fn wired(g: &mut Gpu, ma: H, mb: H) -> Result<bool> {
    // (name, entry, the kind of its recorded command when the entry is a replay entry)
    type Entry = (&'static str, H, Option<u32>);
    let entry = |g: &Gpu, name: &'static str, kind: Option<u32>| -> Result<Entry> { Ok((name, g.function(mb, name)?, kind)) };
    let (m1, pair, bits) = (entry(g, "flash_next_mxfp8_m1", None)?, entry(g, "flash_next_mxfp8_pair", None)?, entry(g, "flash_next_mxfp8_bits", None)?);
    let (batch, batch16, stage) = (entry(g, "flash_resident_projection_batch", None)?, entry(g, "flash_resident_projection_bf16", None)?, entry(g, "flash_resident_projection_bf16_rows", None)?);
    // The recorded forms (`resident_launch::Kind`): Mx 4, Mixed 17, MixedBf16 34, MxRows2/4/8/8w 42 to 45, MixedRows 46.
    let (m1_replay, batch_replay, batch16_replay, stage_replay) = (entry(g, "resident_launch_mx", Some(4))?, entry(g, "resident_launch_mixed", Some(17))?, entry(g, "resident_launch_mixed_bf16", Some(34))?,
        entry(g, "resident_launch_flash_resident_projection_bf16_rows", Some(46))?);
    let small = [entry(g, "mx_rows_small2", None)?, entry(g, "mx_rows_small4", None)?, entry(g, "mx_rows_small8", None)?, entry(g, "mx_rows_small8w", None)?,
        entry(g, "resident_launch_mx_rows_small2", Some(42))?, entry(g, "resident_launch_mx_rows_small4", Some(43))?, entry(g, "resident_launch_mx_rows_small8", Some(44))?, entry(g, "resident_launch_mx_rows_small8w", Some(45))?];
    let (old_m1, old_batch, old_batch16, old_stage) = (g.function(ma, "flash_next_mxfp8_m1")?, g.function(ma, "flash_resident_projection_batch")?, g.function(ma, "flash_resident_projection_bf16")?, g.function(ma, "flash_resident_projection_bf16_rows")?);
    // (entry, words, words that differ from the twin, scalar words, scalar words that differ from the base image)
    let mut table: Vec<(&'static str, usize, usize, usize, usize)> = vec![];
    let mut add = |name: &'static str, tc: (usize, usize), scalar: (usize, usize)| match table.iter_mut().find(|row| row.0 == name) {
        Some(row) => { row.1 += tc.0; row.2 += tc.1; row.3 += scalar.0; row.4 += scalar.1; }
        None => table.push((name, tc.0, tc.1, scalar.0, scalar.1)),
    };
    let (mut modes_differ, mut control) = (0usize, 0usize);
    // A and C take the tensor-core word, B (between them in a launch) the scalar word. Rows off and on 16 and 4.
    let sets: [[(usize, usize); 3]; 4] = [[(130, 4096), (67, 512), (33, 96)], [(17, 32), (5, 64), (16, 160)], [(512, 8192), (100, 4096), (1, 32)], [(49, 12288), (130, 4096), (15, 128)]];
    for (si, set) in sets.iter().enumerate() {
        for tokens in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 16] {
            let f: Vec<Fixture> = set.iter().enumerate().map(|(i, &(rows, cols))| fixture(rows, cols, tokens, 7000 + (si * 100 + tokens * 3 + i) as u64, i == 2)).collect();
            let d: Vec<Device> = f.iter().map(|f| upload(g, f)).collect::<Result<_>>()?;
            let out: Vec<Buf> = f.iter().map(|f| Buf::poisoned(g, f.rows * f.tokens)).collect::<Result<_>>()?;
            let proj = |i: usize, output: u64| Proj { weights: d[i].w.address, scales: d[i].s.address, input: d[i].x.address, output, rows: f[i].rows as u32, cols: f[i].cols as u32, tokens: tokens as u32 };
            let tc_blocks = |i: usize| (f[i].rows as u32).div_ceil(16) * (tokens as u32).div_ceil(8);
            let quads = |i: usize| (f[i].rows as u32).div_ceil(4);
            let poison = |g: &Gpu| -> Result<()> { for (i, o) in out.iter().enumerate() { g.upload_at(o.address, &vec![0xff; f[i].rows * tokens * 4])?; } Ok(()) };
            // A launch of a layer or a batch: A (tensor-core), then B (scalar) when `mixed`, then C (tensor-core).
            let mut merged = |g: &mut Gpu, (name, h, replay): Entry, old: Option<H>, is_batch: bool, rounded: bool| -> Result<()> {
                poison(g)?;
                let mut jobs = vec![(proj(0, out[0].address), MX_TC, tc_blocks(0))];
                if old.is_some() { jobs.push((proj(1, out[1].address), 0, quads(1))); }
                jobs.push((proj(2, out[2].address), MX_TC, tc_blocks(2)));
                let (bytes, grid) = layer_bytes(&jobs, is_batch);
                launch_bytes(g, h, replay, &bytes, grid)?;
                g.sync()?;
                let (a, c) = (out[0].words(g)?, out[2].words(g)?);
                let (ra, rc) = (against_twin_rounded(&f[0], &f[0].x, &a, rounded), against_twin_rounded(&f[2], &f[2].x, &c, rounded));
                let mut scalar = (0, 0);
                if let Some(old) = old {
                    let have = out[1].words(g)?;
                    let want = if tokens > 4 {
                        // `old` is the base image's ONE-ROW entry: a source row a launch, each into its own poisoned words.
                        let mut want = vec![];
                        for t in 0..tokens {
                            let alone = Buf::poisoned(g, f[1].rows)?;
                            let one = Proj { input: d[1].x.address + (t * f[1].cols * 4) as u64, tokens: 1, ..proj(1, alone.address) };
                            let (mut bytes, grid) = layer_bytes(&[(one, 0, quads(1))], is_batch);
                            g.launch(old, [grid, 1, 1], 128, 0, &mut [bytes.as_mut_ptr().cast()])?;
                            g.sync()?;
                            want.extend(alone.words(g)?);
                        }
                        if want.contains(&u32::MAX) { return Err(format!("MXTC WIRED the one-row reference left a word unwritten: entry={name} tokens={tokens}")); }
                        want
                    } else {
                        let alone = Buf::poisoned(g, f[1].rows * tokens)?;
                        let (mut bytes, grid) = layer_bytes(&[(proj(1, alone.address), 0, quads(1))], is_batch);
                        g.launch(old, [grid, 1, 1], 128, 0, &mut [bytes.as_mut_ptr().cast()])?;
                        g.sync()?;
                        alone.words(g)?
                    };
                    scalar = (have.len(), have.iter().zip(&want).filter(|(x, y)| !same(**x, **y)).count());
                }
                add(name, (ra.0 + rc.0, ra.1 + rc.1), scalar);
                Ok(())
            };
            match tokens {
                1 => {
                    for e in [m1, m1_replay] { merged(g, e, Some(old_m1), false, false)?; }
                    for e in [batch, batch_replay] { merged(g, e, Some(old_batch), true, true)?; }
                    for e in [batch16, batch16_replay] { merged(g, e, Some(old_batch16), true, true)?; }
                }
                2..=4 => { for e in [stage, stage_replay] { merged(g, e, Some(old_stage), true, true)?; } merged(g, pair, None, false, false)?; }
                // STAGE EIGHT: the row stage at 5 to 8 rows, also with the scalar job between the two tensor-core jobs. The base
                // image's stage entry traps above 4 rows, so the scalar job's reference is the base image's one-row entry
                // (`flash_resident_projection_bf16`), a source row a launch; `--rows-p1` compares the scalar word alone the same way.
                5..=8 => { for e in [stage, stage_replay] { merged(g, e, Some(old_batch16), true, true)?; } merged(g, pair, None, false, false)?; }
                _ => merged(g, bits, None, false, false)?,
            }
            if tokens <= 8 {
                for &(name, h, replay) in &small {
                    for i in [0usize, 2] {
                        poison(g)?;
                        let p = proj(i, out[i].address);
                        launch_bytes(g, h, replay, &proj_mode(&p, MX_TC), p.rows.div_ceil(16))?;
                        g.sync()?;
                        let have = out[i].words(g)?;
                        add(name, against_twin_rounded(&f[i], &f[i].x, &have, false), (0, 0));
                        if i == 0 && name == "mx_rows_small8" {
                            // Control 1: the scalar word of the same descriptor (the same entry, mode 0, its own grid) differs somewhere.
                            if tokens == 8 && f[0].cols == 4096 {
                                let alone = Buf::poisoned(g, f[0].rows * tokens)?;
                                let mut a = proj_bytes(&proj(0, alone.address));
                                g.launch(h, [p.rows.div_ceil(8), 1, 1], 128, 0, &mut [a.as_mut_ptr().cast()])?;
                                g.sync()?;
                                modes_differ += alone.words(g)?.iter().zip(&have).filter(|(x, y)| !same(**x, **y)).count();
                            }
                            // Control 2: the twin of a changed first input row must not match the device.
                            let mut x = f[0].x.clone();
                            for c in 0..f[0].cols { x[c] = -x[c] * 1.5 + 0.25; }
                            control += against_twin_rounded(&f[0], &x, &have, false).1;
                        }
                    }
                }
            }
            g.clear()?;
        }
    }
    let mut pass = true;
    for &(name, words, differ, scalar, scalar_differ) in &table {
        let ok = words > 0 && differ == 0 && scalar_differ == 0;
        println!("MXTC WIRED {} entry={name} tensor_core_words={words} differ_from_twin={differ} scalar_words={scalar} differ_from_base_image={scalar_differ}", if ok { "PASS" } else { "FAIL" });
        pass &= ok;
    }
    println!("MXTC WIRED CONTROL {} the scalar word of the same descriptors differs on {modes_differ} words (the mode decides the word)", if modes_differ > 0 { "PASS" } else { "FAIL" });
    println!("MXTC WIRED CONTROL {} a twin fed a changed input row differs on {control} words", if control > 0 { "PASS" } else { "FAIL" });
    Ok(pass && modes_differ > 0 && control > 0 && table.len() == 18)
}

/// STEP: the one-row decode step in the forms the engine SERVES (traced runs `attn-main-trace-01` and `onerow-tr-1`): a
/// rank's 74 MXFP8 launches of a step are 37 merged batches (`resident_launch_mixed_bf16`: q k v, and gate with up) and
/// 37 one-job launches (`resident_launch_mx`: o, and down), recorded and replayed. Base image with scalar descriptors
/// against the candidate with tensor-core descriptors and the candidate with scalar descriptors (the unchanged word).
/// Two views: COLD (every launch on its own weights) and PREFIX (the first 6 MiB of the launch's first matrix were read
/// just before, as the engine's side prefetch leaves them; that read is timed alone and taken off). us a launch and ms
/// for the 145 calls of a step. The direct entries of `time` do not show a cost that lives in a recorded entry (its
/// register cap, `ptx_stamp::REGISTER_CAPS`): this mode does.
fn step(g: &mut Gpu, ma: H, mb: H) -> Result<()> {
    let entries = [("resident_launch_mixed_bf16", 34u32, true), ("resident_launch_mx", 4u32, false)];
    let mut handles = vec![];
    for (name, _, _) in entries { handles.push((g.function(ma, name)?, g.function(mb, name)?)); }
    // (label, entry, jobs of a launch, rows, cols, launches a step)
    let launches: [(&str, usize, usize, usize, usize, f64); 4] = [("qkv3", 0, 3, 4096, 4096, 34.0), ("o", 1, 1, 2048, 8192, 34.0), ("gate_up2", 0, 2, 6144, 4096, 3.0), ("down", 1, 1, 2048, 12288, 3.0)];
    let table = |g: &mut Gpu, list: &[(Proj, u32, u32)], kind: u32, batch: bool| -> Result<(u64, u32)> {
        let (payload, grid) = layer_bytes(list, batch);
        let mut command = vec![0u8; 1040];
        command[..4].copy_from_slice(&kind.to_le_bytes());
        command[16..16 + payload.len()].copy_from_slice(&payload);
        Ok((g.upload(&command)?, grid))
    };
    let launch = |g: &Gpu, h: H, t: (u64, u32)| -> Result<()> {
        let (mut table, mut index) = (t.0, 0u64);
        g.launch(h, [t.1, 1, 1], 128, 0, &mut [(&mut table as *mut u64).cast(), (&mut index as *mut u64).cast()])
    };
    let warm = handles[1].0;
    let mut totals = [[0.0f64; 3]; 2];
    for &(label, entry, jobs, rows, cols, count) in &launches {
        let (kind, batch) = (entries[entry].1, entries[entry].2);
        let bytes = (jobs * rows * cols) as f64 * (1.0 + 1.0 / 32.0);
        let k = ((200e6 / bytes).ceil() as usize).clamp(2, 16);
        let prefix_rows = ((6usize << 20) / cols).min(rows) as u32;
        // tables[word][copy]: word 0 scalar descriptors, 1 tensor-core; warmers[copy]: the scalar prefix rows of the first job
        let (mut tables, mut warmers) = ([vec![], vec![]], vec![]);
        for copy in 0..k {
            let mut descriptors = vec![];
            for job in 0..jobs {
                let f = fixture(rows, cols, 1, 600 + (copy * 8 + job) as u64, false);
                descriptors.push(Proj { weights: g.upload(&f.w)?, scales: g.upload(&f.s)?, input: g.upload(&le(&f.x))?, output: g.alloc(rows * 4)?, rows: rows as u32, cols: cols as u32, tokens: 1 });
            }
            for word in 0..2 {
                let list: Vec<(Proj, u32, u32)> = descriptors.iter().map(|p| (*p, if word == 1 { MX_TC } else { 0 }, p.rows.div_ceil(if word == 1 { 16 } else { 4 }))).collect();
                tables[word].push(table(g, &list, kind, batch)?);
            }
            let w = Proj { output: g.alloc(rows * 4)?, rows: prefix_rows, ..descriptors[0] };
            warmers.push(table(g, &[(w, 0, prefix_rows.div_ceil(4))], 4, false)?);
        }
        // arms: base image scalar, candidate tensor-core, candidate scalar
        let arms: [(H, usize); 3] = [(handles[entry].0, 0), (handles[entry].1, 1), (handles[entry].1, 0)];
        let per = 4 * k;
        let mut times = vec![[vec![], vec![]]; 3];
        for repeat in 0..9 {
            for index in 0..3 {
                let arm = if repeat % 2 == 0 { index } else { 2 - index };
                let (h, word) = arms[arm];
                for copy in 0..k { launch(g, h, tables[word][copy])?; }
                g.sync()?;
                let t0 = Instant::now();
                for i in 0..per { launch(g, h, tables[word][i % k])?; }
                g.sync()?;
                times[arm][0].push(t0.elapsed().as_secs_f64() * 1e6 / per as f64);
                for copy in 0..k { launch(g, warm, warmers[copy])?; launch(g, h, tables[word][copy])?; }
                g.sync()?;
                let t0 = Instant::now();
                for i in 0..per { launch(g, warm, warmers[i % k])?; launch(g, h, tables[word][i % k])?; }
                g.sync()?;
                let both = t0.elapsed().as_secs_f64() * 1e6 / per as f64;
                let t0 = Instant::now();
                for i in 0..per { launch(g, warm, warmers[i % k])?; }
                g.sync()?;
                times[arm][1].push(both - t0.elapsed().as_secs_f64() * 1e6 / per as f64);
            }
        }
        for (view, name) in ["COLD", "PREFIX"].iter().enumerate() {
            let us: Vec<f64> = times.iter().map(|t| median(t[view].clone())).collect();
            println!("MXTC STEP {name} {label} entry={} jobs={jobs} rows={rows} cols={cols} copies={k} base_scalar_us={:.1} cand_tc_us={:.1} cand_scalar_us={:.1} tc_over_base={:.3}", entries[entry].0, us[0], us[1], us[2], us[1] / us[0]);
            for arm in 0..3 { totals[view][arm] += us[arm] * count / 1e3; }
        }
        g.clear()?;
    }
    for (view, name) in ["COLD", "PREFIX"].iter().enumerate() {
        let t = totals[view];
        println!("MXTC STEP PASS {name} launches=74 calls=145 (34 qkv3, 34 o, 3 gate_up2, 3 down) base_scalar_ms={:.2} cand_tc_ms={:.2} cand_scalar_ms={:.2} tc_over_base={:.3}", t[0], t[1], t[2], t[1] / t[0]);
    }
    Ok(())
}

/// STAGE: the merged row stage on the tensor-core word in the form the engine serves (a recorded command through its
/// replay entry). q, k and v (three 4096 x 4096 matrices, `MX_TC`) as ONE launch: the one-row launch
/// (`resident_launch_mixed_bf16`) against the row stage launch (`resident_launch_flash_resident_projection_bf16_rows`)
/// at 2, 3, 4, 5 and 8 source rows, and, for information, what the engine does from 5 rows without the stage: three
/// single calls of `resident_launch_mx_rows_small8`. 8 weight copies, so every call streams from memory (the method of
/// `--rows-p1`'s "mx only" arm, which times the scalar word). us a step (a step = the launches of q, k and v).
/// The line of STAGE EIGHT, written before the run: the row stage at 8 rows at or under 1.25 times the one-row launch.
/// The answer is the line's: false (a HIT) fails the run, so a missed line never ends in the run's `MXTC PASS`.
fn stage(g: &mut Gpu, mb: H) -> Result<bool> {
    let (one, rows, single) = (g.function(mb, "resident_launch_mixed_bf16")?, g.function(mb, "resident_launch_flash_resident_projection_bf16_rows")?, g.function(mb, "resident_launch_mx_rows_small8")?);
    let (n, k, copies, calls, repeats) = (4096usize, 4096usize, 8usize, 16usize, 11usize);
    let table = |g: &mut Gpu, kind: u32, payload: &[u8]| -> Result<u64> {
        let mut command = vec![0u8; 1040];
        command[..4].copy_from_slice(&kind.to_le_bytes());
        command[16..16 + payload.len()].copy_from_slice(payload);
        g.upload(&command)
    };
    // (name, source rows, form): form 0 the one-row launch, 1 the row stage, 2 three single calls
    let mut arms: Vec<(String, usize, u8)> = vec![("one_row".into(), 1, 0)];
    for t in [2usize, 3, 4, 5, 8] { arms.push((format!("stage{t}"), t, 1)); }
    for t in [5usize, 8] { arms.push((format!("singles{t}"), t, 2)); }
    // built[arm][copy]: the launches of a step (entry, table, grid)
    let mut built: Vec<Vec<Vec<(H, u64, u32)>>> = vec![vec![]; arms.len()];
    let blocks = (n as u32).div_ceil(16);
    for copy in 0..copies {
        let mut device = vec![];
        for job in 0..3 {
            let f = fixture(n, k, 8, 900 + (copy * 4 + job) as u64, false);
            device.push((g.upload(&f.w)?, g.upload(&f.s)?, g.upload(&le(&f.x))?));
        }
        for (arm, (_, tokens, form)) in arms.iter().enumerate() {
            let mut list = vec![];
            for &(weights, scales, input) in &device {
                list.push((Proj { weights, scales, input, output: g.alloc(n * tokens * 4)?, rows: n as u32, cols: k as u32, tokens: *tokens as u32 }, MX_TC, blocks));
            }
            let launches = if *form == 2 {
                let mut v = vec![];
                for (p, _, _) in &list { v.push((single, table(g, 44, &proj_mode(p, MX_TC))?, blocks)); }
                v
            } else {
                let (payload, grid) = layer_bytes(&list, true);
                vec![(if *form == 0 { one } else { rows }, table(g, if *form == 0 { 34 } else { 46 }, &payload)?, grid)]
            };
            built[arm].push(launches);
        }
    }
    let launch = |g: &Gpu, (h, table, grid): (H, u64, u32)| -> Result<()> {
        let (mut table, mut index) = (table, 0u64);
        g.launch(h, [grid, 1, 1], 128, 0, &mut [(&mut table as *mut u64).cast(), (&mut index as *mut u64).cast()])
    };
    let mut times: Vec<Vec<f64>> = vec![vec![]; arms.len()];
    for repeat in 0..repeats {
        for i in 0..arms.len() {
            let arm = (i + repeat) % arms.len();
            for &l in &built[arm][0] { launch(g, l)?; }
            g.sync()?;
            let t0 = Instant::now();
            for call in 0..calls { for &l in &built[arm][(repeat + call) % copies] { launch(g, l)?; } }
            g.sync()?;
            times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
        }
    }
    let weight_bytes = 3 * (n * k + n * k / 32);
    println!("MXTC STAGE batch=\"q k v: mx3 4096x4096, tensor-core word, recorded\" weight_mib={:.2} floor_us_at_230={:.1} copies={copies} calls={calls} repeats={repeats}",
        weight_bytes as f64 / 1048576.0, weight_bytes as f64 / 230e9 * 1e6);
    let one_us = median(times[0].clone());
    let mut eight = 0.0;
    for (arm, (name, tokens, _)) in arms.iter().enumerate() {
        let t = median(times[arm].clone());
        let (lo, hi) = (times[arm].iter().cloned().fold(f64::MAX, f64::min), times[arm].iter().cloned().fold(0.0, f64::max));
        if name == "stage8" { eight = t / one_us; }
        println!("MXTC STAGE arm={name} rows={tokens} us_per_step={t:.1} min={lo:.1} max={hi:.1} against_one_row={:.3}", t / one_us);
    }
    let met = eight <= 1.25;
    println!("MXTC STAGE LINE {} rows8={eight:.3} (the line: at or under 1.250 times the one-row launch)", if met { "PASS" } else { "HIT" });
    g.clear()?;
    Ok(met)
}

pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    if mode == "code" || mode == "code-strict" { return super::codepair::run(base_path, candidate_path, mode == "code-strict"); }
    if mode == "feed-pair" { return super::feedpair::run(base_path, candidate_path); }
    if mode == "expert-pair" { return super::expertpair::run(base_path, candidate_path); }
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXTC baseline_sha256={} candidate_sha256={} mode={mode}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?;
    let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let new = New { rows: g.function(mb, "mx_tc_rows")?, tile: g.function(mb, "mx_tc_tile").ok(), packed: g.function(mb, "mx_tc_pack").ok().zip(g.function(mb, "mx_tc_tile_packed").ok()) };
    println!("MXTC entries rows=mx_tc_rows tile={} packed={}", if new.tile.is_some() { "mx_tc_tile" } else { "absent (every call takes mx_tc_rows)" }, if new.packed.is_some() { "mx_tc_pack, mx_tc_tile_packed" } else { "absent" });
    let mut pass = true;
    let modes: Vec<&str> = mode.split(',').collect();
    if let Some(unknown) = modes.iter().find(|m| !["probe", "gates", "wired", "step", "stage", "time", "tile", "var", "all"].contains(m)) { return Err(format!("MXTC: no mode {unknown}").into()); }
    // `all` leaves out `tile`, `var` and `stage`.
    let on = |name: &str| modes.contains(&name) || modes.contains(&"all") && !matches!(name, "tile" | "var" | "stage");
    if on("probe") { pass &= probe(&mut g, mb)?; }
    if on("gates") { pass &= gates(&mut g, &new)?; }
    if on("wired") { pass &= wired(&mut g, ma, mb)?; }
    if on("step") { step(&mut g, ma, mb)?; }
    if on("stage") { pass &= stage(&mut g, mb)?; }
    // The bench entries `mx_tc_rows_*` existed for the prefetch study (runs `onerow-v1-1`, `onerow-v2b-1`, `onerow-full-1`);
    // an image without them has nothing to compare, so `all` leaves this mode out.
    if on("var") {
        let vars: Vec<(&'static str, H)> = VARIANTS.iter().filter_map(|name| g.function(mb, name).ok().map(|entry| (*name, entry))).collect();
        println!("MXTC variants: {}", vars.iter().map(|v| v.0).collect::<Vec<_>>().join(" "));
        pass &= variants_same(&mut g, &vars)?;
        let old = Old { m1: g.function(ma, "flash_next_mxfp8_m1")?, small2: g.function(ma, "mx_rows_small2")?, small4: g.function(ma, "mx_rows_small4")?, small8: g.function(ma, "mx_rows_small8")?, small8w: g.function(ma, "mx_rows_small8w")?, tile: g.function(ma, "mx_rows_tile")? };
        variants_cold(&mut g, &old, &vars)?;
    }
    if on("tile") { tile_sweep(&mut g, &new)?; }
    if on("time") {
        let old = Old { m1: g.function(ma, "flash_next_mxfp8_m1")?, small2: g.function(ma, "mx_rows_small2")?, small4: g.function(ma, "mx_rows_small4")?, small8: g.function(ma, "mx_rows_small8")?, small8w: g.function(ma, "mx_rows_small8w")?, tile: g.function(ma, "mx_rows_tile")? };
        cold(&mut g, &old, &new)?;
        time(&mut g, &old, &new)?;
    }
    if !pass { return Err("MXTC FAIL".into()); }
    println!("MXTC PASS");
    Ok(())
}

/// Prompt-shape replay. Descriptors are printed; every output word is checked.
pub(in crate::cuda) fn diag_replay(g:&mut Gpu,module:H)->Result<()> {
 let f=g.function(module,"mx_tc_tile")?;
 for (n,k,m,calls) in [(4096usize,4096usize,1024usize,816usize),(2048,8192,1024,272),(6144,4096,256,192),(2048,12288,256,96)] {
  let w=g.upload(&vec![0x38;n*k])?;let scales=g.upload(&vec![127;n*k/32])?;
  let x=g.words(&vec![1f32.to_bits();m*k])?;let out=g.upload(&vec![0xa5;m*n*4])?;
  let p=Proj{weights:w,scales,input:x,output:out,rows:n as u32,cols:k as u32,tokens:m as u32};
  let launch=|g:&Gpu|{let mut a=proj_bytes(&p);g.launch(f,[m.div_ceil(64) as u32,n.div_ceil(128) as u32,1],256,0,&mut [a.as_mut_ptr().cast()])};
  super::diag_limits::replay(g,"mx_tc_tile",&format!("M={m} N={n} K={k} trace_calls={calls}"),out,m*n,Some((k as f32).to_bits()),&launch)?;g.clear()?;
 }Ok(())
}
