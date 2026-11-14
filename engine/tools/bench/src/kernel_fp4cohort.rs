//! FP4-COHORT premise (lane fp4-cohort, bench only). One packed sm_121a NVFP4 kernel (`fp4_cohort`, FP4 PTX) for a
//! prompt cohort and for the one-token decode step, against its CPU twin and against the decode kernel `nv_cols`
//! (production PTX). Run: kernel-bench --fp4-cohort <production ptx> <fp4 ptx> [gates|time|all].
//!
//! CPU twin of contract (b): per output word, acc = +0; per 64-column step in ascending K, acc = M(acc), where M is the
//! lane multi-row integer model of mma.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X (exact 16-column block sums,
//! exact UE4M3 scale products, terms and the old accumulator aligned on the largest exponent, bits under 35 fraction
//! bits cut toward zero term by term, exact sum, cut toward zero to FP32); word = bf16(fl(acc * fl(global * activation))).
//! One model choice was open after lane multi-row (does a zero block sum's scale exponent count in the alignment?);
//! the calibration fixture below decides it from the device and the run names it.
//!
//! Gates: G0 host self-tests; G1 every word = twin (raw accumulators and final words) on the edge fixtures, all words;
//! sampled words on real shapes; column tiles = row-major; zero-skip = no skip. G2 every word of a token identical at
//! cohort sizes 1, 8, 256, 1,024, 4,096 and 8,192 and at a reversed token order. G3 REPORT words that differ from the
//! rc.8 chain. Lines: L1 TFLOPS at 8,192-row cohorts (line 60), L2 one-token form against nv_cols (line: not more than
//! 1 percent slower). Safe code only.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
/// A kernel form: its function and its tokens a block (grid x = ceil(max tokens / that)).
type F = (H, usize);
/// The word function of this run: 1 = one chain (first matrices), 4 = quarter chains (second matrices; mode suffix ":q").
static CHAINS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
fn chains() -> u32 { CHAINS.load(std::sync::atomic::Ordering::Relaxed) }
/// Shapes the word function admits: whole 64-column steps in every chain.
fn admits(cols: usize) -> bool { cols % (64 * chains() as usize) == 0 }

// ------------------------------------------------------------------ the CPU twin (kit library, shared with the host tests)
#[path = "../../../crates/gpu/src/tc_twin.rs"]
#[allow(dead_code)]
mod tc_twin;
use tc_twin::{DEVICE_MODEL, HALF, Rng, Twin, bf16_round, half, step, ue4m3_f32};

// ------------------------------------------------------------------ fixtures
#[derive(Clone, Copy)]
enum Kind { Random, Small, Sparse, Cancel, Spread, Real }
fn scale_code(r: &mut Rng, k: Kind) -> u8 {
    match k {
        Kind::Small => [0u8, 1, 2, 3, 5, 7, 8, 9, 15][r.below(9) as usize],
        Kind::Spread => if r.below(2) == 0 { 1 + r.below(8) as u8 } else { 0x70 + r.below(15) as u8 },
        Kind::Real => 0x30 + r.below(20) as u8,
        _ => r.below(0x7f) as u8, // every code but NaN (0x7f)
    }
}
fn code(r: &mut Rng, k: Kind, k_index: usize) -> u8 {
    match k {
        Kind::Sparse => if r.below(10) < 8 { if r.below(2) == 0 { 0 } else { 8 } } else { r.below(16) as u8 },
        Kind::Cancel => [2u8, 10][k_index % 2], // +1, -1, +1, -1: every block sum of equal weights is zero
        _ => r.below(16) as u8,
    }
}
struct HostPart { w: Vec<u8>, s: Vec<u8>, tokens: Vec<usize>, global: f32, activation: f32 }
/// Weights as one code a byte (row-major); the activation pool as one code a byte, token-major; scales one byte a block.
struct Fixture { rows: usize, cols: usize, parts: Vec<HostPart>, codes: Vec<u8>, scales: Vec<u8> }
fn fixture(rows: usize, cols: usize, counts: &[usize], kind: Kind, seed: u64) -> Fixture {
    let mut r = Rng(seed);
    let pool: usize = counts.iter().sum();
    let mut codes = Vec::with_capacity(pool * cols);
    for _ in 0..pool { for k in 0..cols { codes.push(code(&mut r, kind, k)); } }
    if let Kind::Cancel = kind { for c in codes.iter_mut().step_by(7) { *c = r.below(16) as u8; } }
    let scales = (0..pool * cols / 16).map(|_| scale_code(&mut r, kind)).collect();
    let mut parts = vec![];
    let mut first = 0;
    for &n in counts {
        let w = (0..rows * cols).map(|_| match kind { Kind::Cancel => 2, _ => code(&mut r, if let Kind::Sparse = kind { Kind::Random } else { kind }, 0) }).collect();
        let s = (0..rows * cols / 16).map(|_| scale_code(&mut r, kind)).collect();
        let global = f32::from_bits(0x3c00_0000 + (r.next_u64() as u32 & 0x07ff_ffff)); // generic, about 2^-7 .. 2^-6
        let activation = f32::from_bits(0x3e00_0000 + (r.next_u64() as u32 & 0x00ff_ffff));
        parts.push(HostPart { w, s, tokens: (first..first + n).collect(), global, activation });
        first += n;
    }
    Fixture { rows, cols, parts, codes, scales }
}
fn pack(codes: &[u8]) -> Vec<u8> { codes.chunks_exact(2).map(|p| p[0] | p[1] << 4).collect() }
/// Row-major codes to column tiles (the layout of nv_cols).
fn column_codes(w: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows / 128 * cols * 64];
    for row in 0..rows { for j in 0..cols { out[((row / 128) * cols + j) * 64 + (row % 128) / 2] |= w[row * cols + j] << (4 * (row & 1)); } }
    out
}
fn column_scales(s: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks = cols / 16; let mut out = vec![0u8; rows / 128 * blocks * 128];
    for row in 0..rows { for b in 0..blocks { out[((row / 128) * blocks + b) * 128 + row % 128] = s[row * blocks + b]; } }
    out
}
/// The twin's raw accumulator and final word for (part, row, local token).
fn twin(t: Twin, fx: &Fixture, part: usize, row: usize, local: usize) -> (u32, u32) {
    let p = &fx.parts[part];
    let token = p.tokens[local];
    let (w, a) = (&p.w[row * fx.cols..][..fx.cols], &fx.codes[token * fx.cols..][..fx.cols]);
    let (sw, sa) = (&p.s[row * fx.cols / 16..][..fx.cols / 16], &fx.scales[token * fx.cols / 16..][..fx.cols / 16]);
    // One chain a word, or four chains (one a quarter of the columns, each from +0) joined fl(fl(a0+a1)+fl(a2+a3)).
    let n = chains() as usize;
    let quarter = fx.cols / n;
    let mut parts = [0u32; 4];
    for (q, part) in parts.iter_mut().enumerate().take(n) {
        for k in 0..quarter / 64 { let c = q * quarter + k * 64; *part = step(t, &w[c..][..64], &a[c..][..64], &sw[c / 16..][..4], &sa[c / 16..][..4], *part); }
    }
    let f = f32::from_bits;
    let acc = if n == 1 { parts[0] } else { ((f(parts[0]) + f(parts[1])) + (f(parts[2]) + f(parts[3]))).to_bits() };
    (acc, bf16_round(f32::from_bits(acc) * (p.global * p.activation)).to_bits())
}
/// The rc.8 chain for the same values: one ascending FP32 sum of ((value * activation) * fold), fold =
/// (e4m3(c) * block activation scale) * (global * activation), no FMA, then the BF16 round.
fn rc8(fx: &Fixture, part: usize, row: usize, local: usize) -> u32 {
    let p = &fx.parts[part];
    let token = p.tokens[local];
    let g = p.global * p.activation;
    let mut sum = 0f32;
    for b in 0..fx.cols / 16 {
        let fold = (ue4m3_f32(p.s[row * fx.cols / 16 + b]) * ue4m3_f32(fx.scales[token * fx.cols / 16 + b])) * g;
        for e in 0..16 {
            let k = b * 16 + e;
            let (v, x) = (half(p.w[row * fx.cols + k]) as f32 * 0.5, half(fx.codes[token * fx.cols + k]) as f32 * 0.5);
            sum += (v * x) * fold;
        }
    }
    bf16_round(sum).to_bits()
}

// ------------------------------------------------------------------ device
fn part_bytes(weight: u64, scales: u64, output: u64, first: u32, tokens: u32, global: f32, activation: f32) -> Vec<u8> {
    let mut v = vec![];
    for x in [weight, scales, output] { v.extend(x.to_le_bytes()); }
    for x in [first, tokens] { v.extend(x.to_le_bytes()); }
    for x in [global, activation] { v.extend(x.to_le_bytes()); }
    v
}
#[repr(C)] #[derive(Clone, Copy)]
struct Args { table: u64, codes: u64, scales: u64, count: u32, rows: u32, cols: u32, layout: u32, skip: u32, raw: u32, chains: u32 }
struct Device { args: Args, max_tokens: usize, outputs: Vec<(u64, usize)> }
/// Uploads a fixture: weights in `layout` (0 row-major, 1 column tiles), the gathered activations, one output a part.
fn upload(g: &mut Gpu, fx: &Fixture, layout: u32, shared: Option<&[(u64, u64)]>) -> Result<Device> {
    let (mut codes, mut scales, mut table, mut outputs) = (vec![], vec![], vec![], vec![]);
    let mut first = 0u32;
    for (i, p) in fx.parts.iter().enumerate() {
        for &t in &p.tokens { codes.extend(pack(&fx.codes[t * fx.cols..][..fx.cols])); scales.extend(&fx.scales[t * fx.cols / 16..][..fx.cols / 16]); }
        let (w, s) = match shared { Some(v) => v[i % v.len()], None => if layout == 1 {
            (g.upload(&column_codes(&p.w, fx.rows, fx.cols))?, g.upload(&column_scales(&p.s, fx.rows, fx.cols))?)
        } else { (g.upload(&pack(&p.w))?, g.upload(&p.s)?) } };
        let words = fx.rows * p.tokens.len();
        let out = g.upload(&vec![0xffu8; words.max(1) * 4])?;
        outputs.push((out, words));
        table.extend(part_bytes(w, s, out, first, p.tokens.len() as u32, p.global, p.activation));
        first += p.tokens.len() as u32;
    }
    codes.extend([0u8; 64]); scales.extend([0u8; 64]);
    let args = Args { table: g.upload(&table)?, codes: g.upload(&codes)?, scales: g.upload(&scales)?, count: fx.parts.len() as u32, rows: fx.rows as u32, cols: fx.cols as u32, layout, skip: 0, raw: 0, chains: chains() };
    Ok(Device { args, max_tokens: fx.parts.iter().map(|p| p.tokens.len()).max().unwrap_or(0), outputs })
}
fn launch(g: &Gpu, f: F, d: &Device, skip: u32, raw: u32) -> Result<()> {
    let mut a = Args { skip, raw, ..d.args };
    g.launch(f.0, [d.max_tokens.div_ceil(f.1).max(1) as u32, d.args.rows / 128, d.args.count], 256, 0, &mut [ptr(&mut a)])
}
fn words(g: &Gpu, d: &Device) -> Result<Vec<Vec<u32>>> {
    d.outputs.iter().map(|&(p, n)| Ok(g.read(p, n * 4)?.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect())).collect()
}

// ------------------------------------------------------------------ gates
fn transpose_selftest() -> Result<()> {
    let swap = |a: &mut u32, b: &mut u32, s: u32, m: u32| { let t = ((*a >> s) ^ *b) & m; *b ^= t; *a ^= t << s; };
    let mut r = Rng(3);
    for _ in 0..1000 {
        let x: [u32; 8] = std::array::from_fn(|_| r.next_u64() as u32);
        let [mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7] = x;
        swap(&mut x0, &mut x4, 16, 0xffff); swap(&mut x1, &mut x5, 16, 0xffff); swap(&mut x2, &mut x6, 16, 0xffff); swap(&mut x3, &mut x7, 16, 0xffff);
        swap(&mut x0, &mut x2, 8, 0x00ff_00ff); swap(&mut x1, &mut x3, 8, 0x00ff_00ff); swap(&mut x4, &mut x6, 8, 0x00ff_00ff); swap(&mut x5, &mut x7, 8, 0x00ff_00ff);
        swap(&mut x0, &mut x1, 4, 0x0f0f_0f0f); swap(&mut x2, &mut x3, 4, 0x0f0f_0f0f); swap(&mut x4, &mut x5, 4, 0x0f0f_0f0f); swap(&mut x6, &mut x7, 4, 0x0f0f_0f0f);
        let y = [x0, x1, x2, x3, x4, x5, x6, x7];
        for i in 0..8 { for j in 0..8 { if (y[i] >> (4 * j)) & 15 != (x[j] >> (4 * i)) & 15 { return Err("FP4COHORT G0 FAIL nibble transpose".into()); } } }
    }
    // The twin on exact values: small codes with unit scales sum exactly.
    let (w, a) = ([2u8; 64], [2u8; 64]);
    if f32::from_bits(step(DEVICE_MODEL, &w, &a, &[0x38; 4], &[0x38; 4], 0)) != 64.0 { return Err("FP4COHORT G0 FAIL twin exact sum".into()); }
    Ok(())
}
/// Compares the device with the twin: every word when `every`, else rows 0, 1 and the last of the first and last part.
fn compare(g: &mut Gpu, f: F, fx: &Fixture, d: &Device, t: Twin, every: bool, label: &str) -> Result<(usize, usize)> {
    let mut got = vec![];
    for raw in [1u32, 0] {
        for o in &d.outputs { g.upload_at(o.0, &vec![0xffu8; o.1.max(1) * 4])?; }
        launch(g, f, d, 0, raw)?; g.sync()?;
        got.push(words(g, d)?);
    }
    let mut checked = 0;
    for (pi, p) in fx.parts.iter().enumerate() {
        if !every && pi != 0 && pi + 1 != fx.parts.len() { continue; }
        let rows: Vec<usize> = if every { (0..fx.rows).collect() } else { vec![0, 1, fx.rows - 1] };
        for &row in &rows { for local in 0..p.tokens.len() {
            let (acc, word) = twin(t, fx, pi, row, local);
            for (raw, want) in [(1, acc), (0, word)] {
                let have = got[1 - raw][pi][row * p.tokens.len() + local];
                if have != want { return Err(format!("FP4COHORT G1 FAIL {label} raw={raw} part={pi} row={row} token={local} device={have:08x} twin={want:08x}")); }
            }
            checked += 1;
        } }
    }
    Ok((checked, 0))
}
/// Two forms of the contract on the same device fixture: every word equal.
fn forms_equal(g: &mut Gpu, a: F, b: F, d: &Device, skip: u32, label: &str) -> Result<()> {
    let mut got = vec![];
    for f in [a, b] {
        for o in &d.outputs { g.upload_at(o.0, &vec![0xffu8; o.1.max(1) * 4])?; }
        launch(g, f, d, skip, 0)?; g.sync()?;
        got.push(words(g, d)?);
    }
    if got[0] != got[1] { return Err(format!("FP4COHORT FAIL {label}: words differ from the gated form")); }
    Ok(())
}
fn same(g: &mut Gpu, f: F, a: &Device, b: &Device, skip: (u32, u32), label: &str) -> Result<usize> {
    for d in [a, b] { for o in &d.outputs { g.upload_at(o.0, &vec![0xffu8; o.1.max(1) * 4])?; } }
    launch(g, f, a, skip.0, 0)?; launch(g, f, b, skip.1, 0)?; g.sync()?;
    let (x, y) = (words(g, a)?, words(g, b)?);
    let n: usize = x.iter().map(|v| v.len()).sum();
    if x != y { return Err(format!("FP4COHORT FAIL {label}: outputs differ")); }
    Ok(n)
}
/// Calibration: zero block sums whose scale product is the largest of the step, beside odd small terms. The two
/// alignment choices give different raw accumulators here; exactly one may match the device on every word.
/// Fixture A (`cancel` false): the big block has all-zero weight codes (every product zero). Fixture B (`cancel` true):
/// the big block's weights are +1, -1, ... against equal activations (sum zero, products nonzero). Three models:
/// zero blocks always count (fits B, not A), never count (fits A, not B), count when a product is nonzero (fits both).
fn calibration_fixture(cancel: bool) -> Fixture {
    let (rows, cols) = (128usize, 256usize);
    let mut fx = fixture(rows, cols, &[64], Kind::Random, 0xca1);
    let mut r = Rng(0xca2);
    // Step k: block k % 4 has zero weight codes and the largest scales (0x7e x 0x7e: exponent 16); the other blocks have
    // exponent -12 (weight 0x08..0x0f, activation subnormal 0x01..0x07). A gap of 28 cuts one bit of the small terms and
    // most of the old accumulator when the zero block counts in the alignment, nothing when it does not.
    for row in 0..rows { for k in 0..cols / 64 {
        let zero = k % 4;
        for e in 0..16 { fx.parts[0].w[row * cols + k * 64 + zero * 16 + e] = if cancel { [2, 10][e % 2] } else { 0 }; }
        for q in 0..4 { fx.parts[0].s[row * cols / 16 + k * 4 + q] = if q == zero { 0x7e } else { 0x08 + r.below(8) as u8 }; }
    } }
    for (i, s) in fx.scales.iter_mut().enumerate() { *s = if (i % (cols / 16)) % 4 == (i % (cols / 16)) / 4 % 4 { 0x7e } else { 1 + r.below(7) as u8 }; }
    if cancel { for t in 0..64 { for k in 0..cols / 64 { for e in 0..16 { fx.codes[t * cols + k * 64 + (k % 4) * 16 + e] = 2; } } } }
    fx
}
fn calibrate(g: &mut Gpu, f: F) -> Result<Twin> {
    let rows = 128usize;
    let models = [Twin { zero_terms_align: true, products: false }, Twin { zero_terms_align: false, products: false }, DEVICE_MODEL];
    let mut bad = [0usize; 3];
    for cancel in [false, true] {
        let fx = calibration_fixture(cancel);
        let d = upload(g, &fx, 1, None)?;
        launch(g, f, &d, 0, 1)?; g.sync()?;
        let got = words(g, &d)?;
        for (i, &t) in models.iter().enumerate() {
            let mut n = 0;
            for row in 0..rows { for local in 0..64 { if twin(t, &fx, 0, row, local).0 != got[0][row * 64 + local] { n += 1; } } }
            println!("FP4COHORT CALIBRATION fixture={} {t:?} raw_words={} differ={n}", if cancel { "B-cancelled-sums" } else { "A-zero-codes" }, rows * 64);
            bad[i] += n;
        }
        g.clear()?;
    }
    let fits: Vec<Twin> = models.iter().zip(bad).filter(|(_, b)| *b == 0).map(|(t, _)| *t).collect();
    match fits.as_slice() { [t] => Ok(*t), [] => Err("FP4COHORT CALIBRATION FAIL: no model fits".into()), _ => Err("FP4COHORT CALIBRATION FAIL: the fixture does not separate the models".into()) }
}

fn gates(g: &mut Gpu, f: F, others: &[(&str, F)]) -> Result<Twin> {
    transpose_selftest()?;
    println!("FP4COHORT G0 PASS nibble transpose 1000 x 64, twin exact sum");
    let failed = std::cell::Cell::new(0usize);
    // A failed calibration fails the run: the gates below still run on DEVICE_MODEL, for diagnosis only.
    let (t, calibration) = crate::check_gate::or_count_failed(calibrate(g, f), DEVICE_MODEL, &failed);
    if let Some(e) = calibration { println!("{e} -> gates continue with DEVICE_MODEL for diagnosis; the run fails"); }
    println!("FP4COHORT TWIN {t:?}");
    let note = |r: Result<()>| { if let Err(e) = r { println!("{e}"); failed.set(failed.get() + 1); } };
    let started = Instant::now();
    let (mut words_all, mut fixtures) = (0usize, 0usize);
    // G1 edge fixtures, every word, both layouts, raw and final.
    let edge: Vec<(usize, usize, Vec<usize>, Kind)> = vec![
        (128, 4096, vec![256], Kind::Random), (128, 1024, vec![200, 1, 129], Kind::Small), (256, 512, vec![33, 64, 7], Kind::Sparse),
        (128, 2048, vec![130], Kind::Cancel), (128, 1024, vec![96], Kind::Spread), (128, 64, vec![1, 2, 8, 9, 31], Kind::Random),
        (384, 128, vec![127, 128], Kind::Random), (128, 8192, vec![17], Kind::Random),
    ];
    for (i, (rows, cols, counts, kind)) in edge.iter().enumerate() {
        if !admits(*cols) { println!("FP4COHORT G1 edge{i} cols={cols} not admitted by chains={} (skipped)", chains()); continue; }
        let fx = fixture(*rows, *cols, counts, *kind, 100 + i as u64);
        let (d1, d0) = (upload(g, &fx, 1, None)?, upload(g, &fx, 0, None)?);
        note(compare(g, f, &fx, &d1, t, true, &format!("edge{i} tiles")).map(|(n, _)| { words_all += n; }));
        note(compare(g, f, &fx, &d0, t, true, &format!("edge{i} rows")).map(|_| ()));
        fixtures += 1;
        note(same(g, f, &d1, &d0, (0, 0), &format!("edge{i} column tiles vs row-major")).map(|_| ()));
        for &(name, o) in others { for d in [&d1, &d0] { note(forms_equal(g, f, o, d, 0, &format!("edge{i} form {name}"))); } }
        g.clear()?;
    }
    println!("FP4COHORT G1 EDGE {} fixtures={fixtures} words_x_layouts_x(raw,final)={} elapsed_s={:.1}", if failed.get() == 0 { "PASS" } else { "FAIL (see lines above)" }, words_all * 2, started.elapsed().as_secs_f64());
    // Zero-skip (one live token a block) = no skip, with many zero activations.
    let fx = fixture(256, 2048, &[1, 1, 1, 1, 1, 1, 1, 1], Kind::Sparse, 77);
    let (a, b) = (upload(g, &fx, 1, None)?, upload(g, &fx, 1, None)?);
    let (mut n, before) = (0, failed.get());
    note(same(g, f, &a, &b, (1, 0), "zero-skip vs no skip").map(|x| { n = x; }));
    note(compare(g, f, &fx, &a, t, true, "zero-skip fixture").map(|_| ()));
    for &(name, o) in others { note(forms_equal(g, f, o, &a, 1, &format!("zero-skip form {name}"))); }
    println!("FP4COHORT G1 SKIP {} words={n}", if failed.get() == before { "PASS" } else { "FAIL (see lines above)" });
    g.clear()?;
    // G1 real shapes (sampled words) + G3 report against the rc.8 chain.
    for (rows, cols, label) in [(1024usize, 4096usize, "gate_up"), (2048, 2048, "down")] {
        let counts: Vec<usize> = route(1024, 0.5, 9).into_iter().take(16).collect();
        let fx = fixture(rows, cols, &counts, Kind::Real, 300 + rows as u64);
        let d = upload(g, &fx, 1, None)?;
        let (mut n, before) = (0, failed.get());
        note(compare(g, f, &fx, &d, t, false, &format!("real {label}")).map(|(x, _)| { n = x; }));
        // G3: rc.8 chain on the same sampled words.
        let (mut differ, mut total) = (0usize, 0usize);
        for pi in [0, fx.parts.len() - 1] { for row in [0, 1, rows - 1] { for local in 0..fx.parts[pi].tokens.len() {
            total += 1; if twin(t, &fx, pi, row, local).1 != rc8(&fx, pi, row, local) { differ += 1; }
        } } }
        println!("FP4COHORT G1 REAL {} {label} sampled_words={n} | G3 REPORT words differing from the rc.8 chain: {differ} of {total}", if failed.get() == before { "PASS" } else { "FAIL (see lines above)" });
        g.clear()?;
    }
    // G2: one token pool, cohorts 1 .. 8,192; each token's words must not move with the cohort or its place.
    let mut all = vec![("gated", f)]; all.extend_from_slice(others);
    note(cohort(g, &all, t));
    if failed.get() > 0 { return Err(format!("FP4COHORT GATES FAIL: {} checks failed (lines above)", failed.get())); }
    if !others.is_empty() { println!("FP4COHORT G1 FORMS PASS every edge and zero-skip word of {:?} equals the gated form (so the twin)", others.iter().map(|o| o.0).collect::<Vec<_>>()); }
    Ok(t)
}
/// Top-8 routing of `group` tokens over 288 experts (skew 0 = uniform), as kernel_nvwide: tokens per expert.
fn route(group: usize, skew: f64, seed: u64) -> Vec<usize> {
    let mut random = Rng(seed); let mut counts = vec![0usize; 288];
    let weights: Vec<f64> = (0..288).map(|i| ((i + 1) as f64).powf(-skew)).collect(); let total: f64 = weights.iter().sum();
    for _ in 0..group {
        let mut chosen: Vec<usize> = vec![];
        while chosen.len() < 8 {
            let mut u = (random.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total; let mut pick = 287;
            for (i, w) in weights.iter().enumerate() { if u < *w { pick = i; break; } u -= w; }
            if !chosen.contains(&pick) { chosen.push(pick); counts[pick] += 1; }
        }
    }
    counts.retain(|&c| c > 0); counts
}
fn cohort(g: &mut Gpu, forms: &[(&str, F)], _t: Twin) -> Result<()> {
    let (rows, cols, experts) = (1024usize, 4096usize, 16usize);
    let pool = 8192usize;
    let base = fixture(rows, cols, &vec![0; experts], Kind::Real, 400);
    let mut r = Rng(401);
    let codes: Vec<u8> = (0..pool * cols).map(|_| r.below(16) as u8).collect();
    let scales: Vec<u8> = (0..pool * cols / 16).map(|_| 0x30 + r.below(20) as u8).collect();
    // Routing: token t goes to experts (t % 16) and ((t * 7 + 3) % 16) when they differ.
    let picks: Vec<Vec<usize>> = (0..pool).map(|t| { let (x, y) = (t % experts, (t * 7 + 3) % experts); if x == y { vec![x] } else { vec![x, y] } }).collect();
    let shared: Vec<(u64, u64)> = base.parts.iter().map(|p| Ok((g.upload(&column_codes(&p.w, rows, cols))?, g.upload(&column_scales(&p.s, rows, cols))?))).collect::<Result<_>>()?;
    let probes = [0usize, 1, 7];
    let mut seen: Vec<Option<Vec<u32>>> = vec![None; probes.len() * experts];
    for (n, reverse) in [(1usize, false), (8, false), (256, false), (1024, false), (4096, false), (8192, false), (8192, true)] {
        let order: Vec<usize> = if reverse { (0..n).rev().collect() } else { (0..n).collect() };
        let mut parts: Vec<HostPart> = base.parts.iter().map(|p| HostPart { w: vec![], s: vec![], tokens: vec![], global: p.global, activation: p.activation }).collect();
        for &tok in &order { for &e in &picks[tok] { parts[e].tokens.push(tok); } }
        let live: Vec<usize> = (0..experts).filter(|&e| !parts[e].tokens.is_empty()).collect();
        let fx = Fixture { rows, cols, parts: live.iter().map(|&e| HostPart { w: vec![], s: vec![], tokens: parts[e].tokens.clone(), global: parts[e].global, activation: parts[e].activation }).collect(), codes: codes.clone(), scales: scales.clone() };
        let sh: Vec<(u64, u64)> = live.iter().map(|&e| shared[e]).collect();
        let d = upload(g, &fx, 1, Some(&sh))?;
        for &(name, f) in forms {
        for o in &d.outputs { g.upload_at(o.0, &vec![0xffu8; o.1.max(1) * 4])?; }
        launch(g, f, &d, 0, 0)?; g.sync()?;
        let got = words(g, &d)?;
        for (pi, &e) in live.iter().enumerate() {
            let m = fx.parts[pi].tokens.len();
            for (qi, &probe) in probes.iter().enumerate() {
                let Some(local) = fx.parts[pi].tokens.iter().position(|&t| t == probe) else { continue };
                let column: Vec<u32> = (0..rows).map(|row| got[pi][row * m + local]).collect();
                let slot = &mut seen[qi * experts + e];
                match slot { None => *slot = Some(column), Some(prev) => if *prev != column {
                    return Err(format!("FP4COHORT G2 FAIL token {probe} expert {e} moved at cohort {n} reverse={reverse} form {name}"));
                } }
            }
        }
        }
        println!("FP4COHORT G2 cohort={n} reverse={reverse} parts={} PASS so far", live.len());
    }
    println!("FP4COHORT G2 PASS probes {probes:?} x their experts identical at cohorts 1, 8, 256, 1024, 4096, 8192 and reversed, in every form {:?}", forms.iter().map(|o| o.0).collect::<Vec<_>>());
    g.clear()?;
    Ok(())
}


// ------------------------------------------------------------------ diagnosis (mode diag): never stops at a mismatch
fn uniform(rows: usize, cols: usize, tokens: usize, w: u8, sw: u8, a: u8, sa: u8) -> Fixture {
    let part = HostPart { w: vec![w; rows * cols], s: vec![sw; rows * cols / 16], tokens: (0..tokens).collect(), global: 1.0, activation: 1.0 };
    Fixture { rows, cols, parts: vec![part], codes: vec![a; tokens * cols], scales: vec![sa; tokens * cols / 16] }
}
fn diag_one(g: &mut Gpu, f: F, fx: &Fixture, layout: u32, label: &str) -> Result<Vec<u32>> {
    let d = upload(g, fx, layout, None)?;
    for o in &d.outputs { g.upload_at(o.0, &vec![0xffu8; o.1.max(1) * 4])?; }
    launch(g, f, &d, 0, 1)?; g.sync()?;
    let dev = words(g, &d)?.remove(0);
    let m = fx.parts[0].tokens.len();
    let want: Vec<u32> = (0..fx.rows * m).map(|i| twin(DEVICE_MODEL, fx, 0, i / m, i % m).0).collect();
    let bad: Vec<usize> = (0..dev.len()).filter(|&i| dev[i] != want[i]).collect();
    let unwritten = dev.iter().filter(|&&w| w == 0xffff_ffff).count();
    let mut rel = 0f64;
    for &i in &bad { let (a, b) = (f32::from_bits(dev[i]) as f64, f32::from_bits(want[i]) as f64); if b != 0.0 && a.is_finite() { rel = rel.max(((a - b) / b).abs()); } }
    let mut line = format!("FP4DIAG {label} layout={layout} words={} differ={} unwritten={unwritten} max_rel={rel:.3e}", dev.len(), bad.len());
    for &i in bad.iter().take(3) {
        let at: Vec<(usize, usize)> = (0..want.len()).filter(|&j| want[j] == dev[i]).take(4).map(|j| (j / m, j % m)).collect();
        line += &format!(" | row={} tok={} dev={:08x}={} twin={:08x}={} twin_value_at={at:?}", i / m, i % m, dev[i], f32::from_bits(dev[i]), want[i], f32::from_bits(want[i]));
    }
    println!("{line}");
    g.clear()?;
    Ok(dev)
}
/// Raw device words to a small file in the working directory (for fitting the twin on the Mac).
fn dump(name: &str, words: &[u32]) {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match std::fs::write(name, &bytes) { Ok(()) => println!("FP4DIAG wrote {name} ({} words)", words.len()), Err(e) => println!("FP4DIAG could not write {name}: {e}") }
}
fn diag(g: &mut Gpu, forms: &[(&str, F)]) -> Result<()> {
    for &(name, f) in forms { for layout in [1u32, 0] {
        let run = |g: &mut Gpu, fx: Fixture, what: &str| -> Result<()> { diag_one(g, f, &fx, layout, &format!("{name} {what}")).map(|_| ()) };
        run(g, uniform(128, 64, 8, 2, 0x38, 2, 0x38), "ones cols64 (want 64.0)")?;
        run(g, uniform(128, 256, 8, 2, 0x38, 2, 0x38), "ones cols256 (want 256.0)")?;
        run(g, uniform(128, 64, 8, 0, 0x38, 2, 0x38), "zero weights (want 0)")?;
        // One live 16-column block q0 (weight code 2 there, 0 elsewhere); weight scale of block q = 2^q.
        for q0 in 0..4 {
            let mut fx = uniform(128, 64, 8, 0, 0x38, 2, 0x38);
            for row in 0..128 { for e in 0..16 { fx.parts[0].w[row * 64 + q0 * 16 + e] = 2; } for q in 0..4 { fx.parts[0].s[row * 4 + q] = 0x38 + 8 * q as u8; } }
            run(g, fx, &format!("block{q0} (want {})", 16 << q0))?;
        }
        // Row scale probe: weight scales of row r = code 0x30 + r % 16; token scale probe: activation scales of token n = 0x30 + n.
        let mut fx = uniform(128, 64, 8, 2, 0x38, 2, 0x38);
        for row in 0..128 { for q in 0..4 { fx.parts[0].s[row * 4 + q] = 0x30 + (row % 16) as u8; } }
        run(g, fx, "row scales")?;
        let mut fx = uniform(128, 64, 8, 2, 0x38, 2, 0x38);
        for n in 0..8 { for q in 0..4 { fx.scales[n * 4 + q] = 0x30 + n as u8; } }
        run(g, fx, "token scales")?;
        // Row code probe: row r has code (r % 7) + 1 in every column; token code probe: token n code n % 7 + 1.
        let mut fx = uniform(128, 64, 8, 2, 0x38, 2, 0x38);
        for row in 0..128 { for k in 0..64 { fx.parts[0].w[row * 64 + k] = (row % 7) as u8 + 1; } }
        run(g, fx, "row codes")?;
        let mut fx = uniform(128, 64, 8, 2, 0x38, 2, 0x38);
        for n in 0..8 { for k in 0..64 { fx.codes[n * 64 + k] = (n % 7) as u8 + 1; } }
        run(g, fx, "token codes")?;
        // Column probe: column k of row r is nonzero only at k = r % 64 (code 2); every token code 2 except token n,
        // whose column (n * 9) % 64 is code 4: row r, token n = 1 + (k == (n * 9) % 64).
        let mut fx = uniform(128, 64, 8, 0, 0x38, 2, 0x38);
        for row in 0..128 { fx.parts[0].w[row * 64 + row % 64] = 2; }
        for n in 0..8 { fx.codes[n * 64 + (n * 9) % 64] = 4; }
        run(g, fx, "column probe")?;
        let mut fx = fixture(128, 64, &[8], Kind::Random, 11); for x in fx.parts[0].s.iter_mut().chain(fx.scales.iter_mut()) { *x = 0x30 + *x % 20; }
        run(g, fx, "random cols64 real scales")?;
        let mut fx = fixture(128, 256, &[40], Kind::Random, 12); for x in fx.parts[0].s.iter_mut().chain(fx.scales.iter_mut()) { *x = 0x30 + *x % 20; }
        run(g, fx, "random cols256 real scales")?;
        let dev = diag_one(g, f, &fixture(128, 256, &[40], Kind::Random, 13), layout, &format!("{name} random cols256 every scale"))?;
        dump(&format!("diag-random13-{name}-layout{layout}.bin"), &dev);
        let dev = diag_one(g, f, &calibration_fixture(false), layout, &format!("{name} calibration fixture"))?;
        dump(&format!("diag-calib-{name}-layout{layout}.bin"), &dev);
    } }
    Ok(())
}

// ------------------------------------------------------------------ P2: alone = batch
/// P2 (the batch-equality premise, before any large kernel build). A small pool of token rows ("chats"), each routed
/// to two experts. Every word of every (token, expert, weight row) is computed (1) by the CPU twin, (2) ALONE: one
/// token a launch, in every form, with and without zero-skip, (3) IN A BATCH: the same token inside batches of
/// 2 .. the whole pool, at other places in the batch (forward, reversed, strided), in every form. All three must be
/// equal on every word. Four fixture kinds: real scales, every scale code, sparse codes, cancelled sums.
fn alone_batch(g: &mut Gpu, forms: &[(&str, F)], t: Twin) -> Result<()> {
    let (rows, cols, experts, pool) = (256usize, 1024usize, 6usize, 448usize);
    let (mut differ, mut alone_words, mut batch_words, mut launches) = (0usize, 0usize, 0usize, 0usize);
    for (kind, kname, seed) in [(Kind::Real, "real-scales", 500u64), (Kind::Random, "every-scale", 510), (Kind::Sparse, "sparse", 520), (Kind::Cancel, "cancelled-sums", 530)] {
        let started = Instant::now();
        let mut full = fixture(rows, cols, &vec![pool; experts], kind, seed);
        full.codes.truncate(pool * cols); full.scales.truncate(pool * cols / 16);
        for p in full.parts.iter_mut() { p.tokens = (0..pool).collect(); }
        let picks: Vec<[usize; 2]> = (0..pool).map(|k| { let a = k % experts; let b = (k * 5 + 1 + k / experts) % experts; [a, if b == a { (a + 1) % experts } else { b }] }).collect();
        let want: Vec<[Vec<u32>; 2]> = (0..pool).map(|tok| std::array::from_fn(|j| (0..rows).map(|row| twin(t, &full, picks[tok][j], row, tok).1).collect())).collect();
        let shared: Vec<(u64, u64)> = full.parts.iter().map(|p| Ok((g.upload(&column_codes(&p.w, rows, cols))?, g.upload(&column_scales(&p.s, rows, cols))?))).collect::<Result<_>>()?;
        // One launch of `order` (pool token ids) in form f; the number of words that differ from the twin.
        let run = |g: &mut Gpu, name: &str, f: F, order: &[usize], skip: u32, what: &str| -> Result<(usize, usize)> {
            let mut lists = vec![vec![]; experts];
            for &tok in order { for &e in &picks[tok] { lists[e].push(tok); } }
            let live: Vec<usize> = (0..experts).filter(|&e| !lists[e].is_empty()).collect();
            let mut codes = vec![]; let mut scales = vec![]; let mut local_of = vec![];
            // A compact pool for this launch: only the tokens of `order`, renumbered.
            for (i, &tok) in order.iter().enumerate() { codes.extend(&full.codes[tok * cols..][..cols]); scales.extend(&full.scales[tok * cols / 16..][..cols / 16]); local_of.push((tok, i)); }
            let renumber = |tok: usize| local_of.iter().find(|x| x.0 == tok).unwrap().1;
            let fx = Fixture { rows, cols, parts: live.iter().map(|&e| HostPart { w: vec![], s: vec![], tokens: lists[e].iter().map(|&tok| renumber(tok)).collect(), global: full.parts[e].global, activation: full.parts[e].activation }).collect(), codes, scales };
            let sh: Vec<(u64, u64)> = live.iter().map(|&e| shared[e]).collect();
            let d = upload(g, &fx, 1, Some(&sh))?;
            launch(g, f, &d, skip, 0)?; g.sync()?;
            let got = words(g, &d)?;
            let (mut bad, mut n) = (0usize, 0usize);
            for (pi, &e) in live.iter().enumerate() {
                let m = lists[e].len();
                for (local, &tok) in lists[e].iter().enumerate() {
                    let j = if picks[tok][0] == e { 0 } else { 1 };
                    for row in 0..rows {
                        n += 1;
                        let (have, expect) = (got[pi][row * m + local], want[tok][j][row]);
                        if have != expect {
                            if bad == 0 { println!("FP4COHORT P2 DIFF kind={kname} form={name} {what} batch={} skip={skip} token={tok} place={local} expert={e} row={row} device={have:08x} twin={expect:08x}", order.len()); }
                            bad += 1;
                        }
                    }
                }
            }
            Ok((bad, n))
        };
        // (2) alone.
        let (mut a_bad, mut a_n) = (0usize, 0usize);
        for &(name, f) in forms { for skip in [0u32, 1] { for tok in 0..pool {
            let (b, n) = run(g, name, f, &[tok], skip, "alone")?; a_bad += b; a_n += n; launches += 1;
        } } }
        // (3) in a batch.
        let (mut b_bad, mut b_n) = (0usize, 0usize);
        for n in [2usize, 3, 4, 8, 16, 64, 128, 129, 448] {
            let forward: Vec<usize> = (0..n).collect();
            let reversed: Vec<usize> = (pool - n..pool).rev().collect();
            let strided: Vec<usize> = (0..n).map(|i| (3 + i * 5) % pool).collect();
            for (order, what) in [(&forward, "forward"), (&reversed, "reversed-tail"), (&strided, "strided")] {
                for &(name, f) in forms { for skip in if n <= 8 { vec![0u32, 1] } else { vec![0] } {
                    let (b, w) = run(g, name, f, order, skip, what)?; b_bad += b; b_n += w; launches += 1;
                } }
            }
        }
        println!("FP4COHORT P2 kind={kname} forms={} alone_words={a_n} alone_differ={a_bad} batch_words={b_n} batch_differ={b_bad} elapsed_s={:.1}", forms.len(), started.elapsed().as_secs_f64());
        differ += a_bad + b_bad; alone_words += a_n; batch_words += b_n;
        g.clear()?;
    }
    let names: Vec<&str> = forms.iter().map(|f| f.0).collect();
    if differ > 0 { return Err(format!("FP4COHORT P2 FAIL differ={differ} of alone_words={alone_words} batch_words={batch_words} forms={names:?}")); }
    println!("FP4COHORT P2 PASS alone = batch = CPU twin on every word: alone_words={alone_words} batch_words={batch_words} launches={launches} differ=0 forms={names:?} batches=2,3,4,8,16,64,128,129,448 x (forward, reversed-tail, strided)");
    Ok(())
}

// ------------------------------------------------------------------ peak: the instruction with operands in registers
fn peak(g: &mut Gpu, f: H) -> Result<()> {
    #[repr(C)] #[derive(Clone, Copy)]
    struct Peak { output: u64, steps: u32, chains: u32 }
    let max_blocks = 8192usize;
    let out = g.upload(&vec![0u8; max_blocks * 256 * 4])?;
    let mut best = 0f64;
    for chains in [16u32, 8, 4, 2, 1] {
        for blocks in [1usize, 6, 12, 24, 48, 96, 144, 192, 288, 384, 576, 768, 1536, 8192] {
            let steps = if blocks >= 1536 { 2048u32 } else { 16384 };
            let mut a = Peak { output: out, steps, chains };
            g.launch(f, [blocks as u32, 1, 1], 256, 0, &mut [ptr(&mut a)])?; g.sync()?;
            let mut times = vec![];
            for _ in 0..5 { let t0 = Instant::now(); g.launch(f, [blocks as u32, 1, 1], 256, 0, &mut [ptr(&mut a)])?; g.sync()?; times.push(t0.elapsed().as_secs_f64()); }
            let t = median(times);
            let mmas = blocks as f64 * 8.0 * chains as f64 * steps as f64;
            let tflops = mmas * 16384.0 / t / 1e12;
            best = best.max(tflops);
            println!("FP4COHORT PEAK chains={chains} blocks={blocks} warps={} steps={steps} ms={:.3} mma_per_s={:.3e} TFLOPS={tflops:.1}", blocks * 8, t * 1e3, mmas / t);
        }
    }
    println!("FP4COHORT PEAK BEST TFLOPS={best:.1} (m16n8k64 mxf4nvf4 4X, operands in registers, nothing read from memory)");
    g.clear()
}

// ------------------------------------------------------------------ nv_cols_b: the decode kernel on contract (b)
/// Every word of `nv_cols_b` (the nv_cols launch: one job a (part, token), FP32 activation rows of 17 words a block)
/// against the CPU twin, on edge fixtures. Jobs go 16 a launch, as the host admits them.
fn cols_b_gate(g: &mut Gpu, kernel: H) -> Result<()> {
    let edge: Vec<(usize, usize, Vec<usize>, Kind)> = vec![
        (128, 4096, vec![24], Kind::Random), (128, 1024, vec![20, 1, 9], Kind::Small), (256, 512, vec![33, 7], Kind::Sparse),
        (128, 2048, vec![30], Kind::Cancel), (128, 1024, vec![24], Kind::Spread), (128, 64, vec![1, 2, 8, 9, 31], Kind::Random),
        (384, 128, vec![17, 28], Kind::Random), (256, 4096, vec![5, 2], Kind::Real), (256, 1088, vec![6, 3], Kind::Random),
    ];
    let (mut checked, mut differ) = (0usize, 0usize);
    for (i, (rows, cols, counts, kind)) in edge.iter().enumerate() {
        if !admits(*cols) { println!("FP4COHORT NVCOLSB edge{i} cols={cols} not admitted by chains={} (skipped)", chains()); continue; }
        let (rows, cols) = (*rows, *cols);
        let fx = fixture(rows, cols, counts, *kind, 100 + i as u64);
        let shape_of = |p: &HostPart| Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 128, block_cols: 16, tokens: p.tokens.len() as u32, global: p.global, activation: p.activation, bound: 0.0, role: if chains() == 4 { 1 } else { 0 } };
        // One input row a pool token; one output a part.
        let inputs: Vec<u64> = (0..fx.codes.len() / cols).map(|t| {
            let mut v = vec![];
            for b in 0..cols / 16 { for e in 0..16 { v.extend(e2m1_value(fx.codes[t * cols + b * 16 + e]).to_le_bytes()); } v.extend(ue4m3_f32(fx.scales[t * cols / 16 + b]).to_le_bytes()); }
            g.upload(&v)
        }).collect::<Result<_>>()?;
        let mut jobs = vec![]; let mut outs = vec![];
        for p in &fx.parts {
            let (w, s) = (g.upload(&column_codes(&p.w, rows, cols))?, g.upload(&column_scales(&p.s, rows, cols))?);
            let m = p.tokens.len();
            let out = g.upload(&vec![0xffu8; rows * m * 4])?;
            outs.push(out);
            for (local, &t) in p.tokens.iter().enumerate() { jobs.push(Arguments { input: [inputs[t], 4, 4, 4], weight: w, scales: s, output: out + (local * 4) as u64, shape: shape_of(p) }); }
        }
        let shared_bytes = nv_shared(cols);
        for chunk in jobs.chunks(16) {
            let table = g.upload(&chunk.iter().flat_map(bytes_of).collect::<Vec<u8>>())?;
            let mut l = Launch { single: chunk[0], batch: table, count: chunk.len() as u32 };
            g.launch(kernel, [(rows as u32 / 128) * chunk.len() as u32, 1, 1], 128, shared_bytes, &mut [ptr(&mut l)])?;
        }
        g.sync()?;
        let mut bad = 0usize;
        for (pi, p) in fx.parts.iter().enumerate() {
            let m = p.tokens.len();
            let got: Vec<u32> = g.read(outs[pi], rows * m * 4)?.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
            for row in 0..rows { for local in 0..m {
                let want = twin(DEVICE_MODEL, &fx, pi, row, local).1;
                checked += 1;
                if got[row * m + local] != want {
                    if bad == 0 { println!("FP4COHORT NVCOLSB DIFF edge{i} part={pi} row={row} token={local} device={:08x} twin={want:08x}", got[row * m + local]); }
                    bad += 1;
                }
            } }
        }
        println!("FP4COHORT NVCOLSB edge{i} rows={rows} cols={cols} parts={:?} differ={bad}", counts);
        differ += bad;
        g.clear()?;
    }
    if differ > 0 { return Err(format!("FP4COHORT NVCOLSB G1 FAIL differ={differ} of {checked}")); }
    println!("FP4COHORT NVCOLSB G1 PASS every word = CPU twin: words={checked} chains={} (nv_cols launch, FP32 activation rows, column tiles)", chains());
    Ok(())
}

// ------------------------------------------------------------------ timing
/// The memory bus of a GB10 as the lab measures it (kernels reach 227 to 231 of it): bytes a second.
const BUS_BYTES_PER_S: f64 = 235e9;
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn prompt_time(g: &mut Gpu, f: F) -> Result<()> {
    for (rows, cols, label) in [(1024usize, 4096usize, "gate_up"), (2048, 2048, "down")] {
        let mut r = Rng(7);
        let shared: Vec<(u64, u64)> = (0..288).map(|_| {
            let w: Vec<u8> = (0..rows * cols / 2).map(|_| r.next_u64() as u8).collect();
            let s: Vec<u8> = (0..rows * cols / 16).map(|_| 0x30 + r.below(20) as u8).collect();
            let w = column_codes(&w.iter().flat_map(|b| [b & 15, b >> 4]).collect::<Vec<_>>(), rows, cols);
            Ok((g.upload(&w)?, g.upload(&column_scales(&s, rows, cols))?))
        }).collect::<Result<_>>()?;
        for group in [256usize, 1024, 4096, 8192] {
            let counts = route(group, 0.5, 99 + group as u64);
            let routed: usize = counts.iter().sum();
            let pool = routed;
            let codes: Vec<u8> = (0..pool * cols).map(|_| r.below(16) as u8).collect();
            let scales: Vec<u8> = (0..pool * cols / 16).map(|_| 0x30 + r.below(20) as u8).collect();
            let mut first = 0; let mut parts = vec![];
            for &n in &counts { parts.push(HostPart { w: vec![], s: vec![], tokens: (first..first + n).collect(), global: 0.0078125, activation: 0.1173 }); first += n; }
            let fx = Fixture { rows, cols, parts, codes, scales };
            let d = upload(g, &fx, 1, Some(&shared))?;
            launch(g, f, &d, 0, 0)?; g.sync()?;
            let mut times = vec![];
            for _ in 0..7 { let t0 = Instant::now(); for _ in 0..3 { launch(g, f, &d, 0, 0)?; } g.sync()?; times.push(t0.elapsed().as_secs_f64() / 3.0); }
            let t = median(times);
            let tflops = 2.0 * routed as f64 * rows as f64 * cols as f64 / t / 1e12;
            // Bytes one launch must move: each live part's weights and scales once, the gathered activations, the output words.
            let bytes = counts.len() * (rows * cols / 2 + rows * cols / 16) + routed * (cols / 2 + cols / 16) + routed * rows * 4;
            let floor = bytes as f64 / BUS_BYTES_PER_S;
            println!("FP4COHORT L1 TIME {label} cohort={group} routed_rows={routed} parts={} most={} ms={:.3} TFLOPS={tflops:.1} bytes_mb={:.1} bus_floor_ms={:.3} time_over_bus_floor={:.2} ms_per_prompt_token_x42x(gate,up,down)={:.4}{}",
                counts.len(), counts.iter().max().unwrap(), t * 1e3, bytes as f64 / 1e6, floor * 1e3, t / floor, t * 1e3 * 42.0 * if label == "gate_up" { 2.0 } else { 1.0 } / group as f64,
                if group == 8192 { if tflops >= 60.0 { " LINE MET (>= 60)" } else { " LINE NOT MET (< 60)" } } else { "" });
            g.sync()?;
        }
        g.clear()?;
    }
    Ok(())
}
// nv_cols, mirrored from expert_ops repr(C): Shape 40 bytes (the tenth word is the matrix role), Arguments 96, Launch 112.
#[repr(C)] #[derive(Clone, Copy)]
struct Shape { op: u32, rows: u32, cols: u32, block_rows: u32, block_cols: u32, tokens: u32, global: f32, activation: f32, bound: f32, role: u32 }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 4], weight: u64, scales: u64, output: u64, shape: Shape }
#[repr(C)] #[derive(Clone, Copy)]
struct Launch { single: Arguments, batch: u64, count: u32 }
fn bytes_of(a: &Arguments) -> Vec<u8> {
    let mut v = vec![];
    for x in a.input.iter().chain([&a.weight, &a.scales, &a.output]) { v.extend(x.to_le_bytes()); }
    let s = a.shape; v.extend(s.op.to_le_bytes()); for x in [s.rows, s.cols, s.block_rows, s.block_cols, s.tokens] { v.extend(x.to_le_bytes()); }
    for x in [s.global, s.activation, s.bound] { v.extend(x.to_le_bytes()); } v.extend(s.role.to_le_bytes()); v
}
/// Dynamic shared bytes of a block: `expert_ops::nv_cols_shared` (the contract-b body keeps the nv_cols layout).
fn nv_shared(cols: usize) -> u32 { ((cols / 16 * 17 * 4).div_ceil(16) * 16 + 3 * 4 * (1024 + 128)) as u32 }
/// The third timing arm of `decode_time`.
#[derive(Clone, Copy)]
enum Third { RowMajor, Same, NvAbi { kernel: H, name: &'static str, shared: fn(usize) -> u32, check: bool } }
/// E2M1 code to the FP32 value the quantizer writes (code 8 = -0.0).
fn e2m1_value(code: u8) -> f32 { let m = HALF[(code & 7) as usize] as f32 * 0.5; if code & 8 != 0 { -m } else { m } }
fn decode_time(g: &mut Gpu, cols_fn: H, f: F, third: Third, skip: u32) -> Result<()> {
    let row_major = matches!(third, Third::RowMajor);
    for (rows, cols, count, label) in [(2048usize, 2048usize, 8usize, "down"), (1024, 4096, 16, "gate_up")] {
        let mut r = Rng(77 + rows as u64);
        // Eight rotating weight sets, as the nv_cols bench, so each call reads its weights from memory.
        let mut sets = vec![];
        for _ in 0..8 {
            let mut jobs = vec![];
            for _ in 0..count {
                let w: Vec<u8> = (0..rows * cols).map(|_| r.below(16) as u8).collect();
                let s: Vec<u8> = (0..rows * cols / 16).map(|_| 0x30 + r.below(20) as u8).collect();
                jobs.push((g.upload(&column_codes(&w, rows, cols))?, g.upload(&column_scales(&s, rows, cols))?, g.upload(&pack(&w))?, g.upload(&s)?));
            }
            sets.push(jobs);
        }
        for share in [0.0f64, 0.0685, 0.2298] {
            // One token a job: codes with `share` of zero codes, their scales; the same values as nv_cols's FP32 rows.
            let (mut codes, mut scales, mut rows17) = (vec![], vec![], vec![]);
            for _ in 0..count {
                let c: Vec<u8> = (0..cols).map(|_| if ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < share { if r.below(2) == 0 { 0 } else { 8 } } else { 1 + r.below(7) as u8 | (r.below(2) as u8) << 3 }).collect();
                let s: Vec<u8> = (0..cols / 16).map(|_| 0x30 + r.below(20) as u8).collect();
                let mut v = vec![];
                for b in 0..cols / 16 { for e in 0..16 { v.extend(e2m1_value(c[b * 16 + e]).to_le_bytes()); } v.extend(ue4m3_f32(s[b]).to_le_bytes()); }
                codes.extend(pack(&c)); scales.extend(&s); rows17.push(g.upload(&v)?);
            }
            codes.extend([0u8; 64]); scales.extend([0u8; 64]);
            let (codes, scales) = (g.upload(&codes)?, g.upload(&scales)?);
            let out = g.upload(&vec![0u8; rows * count * 4])?;
            // `role` names the word function (expert_ops::NV_SECOND = 1); the rc.8 baseline ignores it.
            let shape = Shape { op: 1, rows: rows as u32, cols: cols as u32, block_rows: 128, block_cols: 16, tokens: 1, global: 0.75, activation: 0.125, bound: 0.0, role: if chains() == 4 { 1 } else { 0 } };
            let mut nv_tables = vec![]; let mut fp4 = vec![]; let mut fp4_rows = vec![];
            for jobs in &sets {
                let mut t = vec![]; let mut parts = vec![]; let mut parts_rows = vec![];
                for (i, &(w, s, wr, sr)) in jobs.iter().enumerate() {
                    t.extend(bytes_of(&Arguments { input: [rows17[i], 4, 4, 4], weight: w, scales: s, output: out + (i * rows * 4) as u64, shape }));
                    parts.extend(part_bytes(w, s, out + (i * rows * 4) as u64, i as u32, 1, 0.75, 0.125));
                    parts_rows.extend(part_bytes(wr, sr, out + (i * rows * 4) as u64, i as u32, 1, 0.75, 0.125));
                }
                fp4_rows.push(Args { table: g.upload(&parts_rows)?, codes, scales, count: count as u32, rows: rows as u32, cols: cols as u32, layout: 0, skip: 0, raw: 0, chains: chains() });
                nv_tables.push(g.upload(&t)?);
                fp4.push(Args { table: g.upload(&parts)?, codes, scales, count: count as u32, rows: rows as u32, cols: cols as u32, layout: 1, skip, raw: 0, chains: chains() });
            }
            // nv_cols gets its production size; the contract-b body gets its own (smaller from 1,984 columns up).
            let shared_bytes = nv_shared(cols);
            let nv_with = |kernel: H, i: usize| -> Result<()> {
                let (w, s, _, _) = sets[i][0];
                let mut l = Launch { single: Arguments { input: [rows17[0], 4, 4, 4], weight: w, scales: s, output: out, shape }, batch: nv_tables[i], count: count as u32 };
                let bytes = match third { Third::NvAbi { kernel: k, shared, .. } if k == kernel => shared(cols), _ => shared_bytes };
                g.launch(kernel, [(rows as u32).div_ceil(128) * count as u32, 1, 1], 128, bytes, &mut [ptr(&mut l)])
            };
            let nv = |i: usize| nv_with(cols_fn, i);
            let fp = |i: usize, arm: usize| -> Result<()> { let mut a = if arm == 1 || !row_major { fp4[i] } else { fp4_rows[i] }; g.launch(f.0, [1, rows as u32 / 128, count as u32], 256, 0, &mut [ptr(&mut a)]) };
            let mut times = [vec![], vec![], vec![]];
            for repeat in 0..9 { for k in 0..3 {
                let arm = (k + repeat) % 3;
                let one = |i: usize| -> Result<()> { match (arm, third) { (0, _) => nv(i), (2, Third::NvAbi { kernel, .. }) => nv_with(kernel, i), _ => fp(i, arm) } };
                one(0)?; g.sync()?;
                let calls = 32; let t0 = Instant::now();
                for i in 0..calls { one(i % 8)?; }
                g.sync()?; times[arm].push(t0.elapsed().as_secs_f64() * 1e6 / calls as f64);
            } }
            let (a, b, c) = (median(times[0].clone()), median(times[1].clone()), median(times[2].clone()));
            let bytes = count * (rows * cols / 2 + rows * cols / 16);
            let floor_us = bytes as f64 / BUS_BYTES_PER_S * 1e6;
            println!("FP4COHORT L2 TIME {label} zero_share={share} skip={skip} nv_cols_us={a:.1} fp4_one_token_us={b:.1} ratio={:.3}{} | bytes_mb={:.1} bus_floor_us={floor_us:.1} nv_cols_over_floor={:.2} fp4_over_floor={:.2} | {} us={c:.1} ratio={:.3}", b / a,
                if b <= a * 1.01 { " LINE MET (<= 1.01)" } else { " LINE NOT MET (> 1.01)" }, bytes as f64 / 1e6, a / floor_us, b / floor_us,
                match third { Third::RowMajor => "row-major weights (no transpose, no skip)", Third::Same => "same form again (noise check)", Third::NvAbi { name, .. } => name }, c / a);
            if let Third::NvAbi { kernel: k, name, check, .. } = third {
                // Same jobs, same values: every word of the contract-b decode kernel = the tensor-core form's word.
                println!("FP4COHORT L2 LINE {name} {label} zero_share={share} nv_cols_us={a:.1} {name}_us={c:.1} ratio={:.3} {}", c / a, if !check { "DIAGNOSTIC FORM (wrong words on purpose; no line)" } else if c <= a * 1.01 { "LINE MET (<= 1.01)" } else { "LINE NOT MET (> 1.01)" });
                let mut differ = 0usize; let mut n = 0usize;
                for i in if check { vec![0usize, 3, 7] } else { vec![] } {
                    g.upload_at(out, &vec![0xffu8; rows * count * 4])?; nv_with(k, i)?; g.sync()?; let x = g.read(out, rows * count * 4)?;
                    g.upload_at(out, &vec![0xffu8; rows * count * 4])?; fp(i, 1)?; g.sync()?; let y = g.read(out, rows * count * 4)?;
                    n += rows * count; differ += x.chunks_exact(4).zip(y.chunks_exact(4)).filter(|(p, q)| p != q).count();
                }
                if check { println!("FP4COHORT L2 WORDS {name} vs tensor-core form {label} zero_share={share} words={n} differ={differ}"); }
                if differ > 0 { return Err(format!("FP4COHORT FAIL {name} differs from the tensor-core form on {differ} of {n} words")); }
            }
        }
        g.clear()?;
    }
    Ok(())
}

/// The FP4 PTX with `.maxnreg` for one entry (Rust has no launch bounds; the production stamp does the same).
fn capped(ptx: &[u8], entry: &str, registers: u32) -> Result<Vec<u8>> {
    let mut text = String::from_utf8(ptx.to_vec()).map_err(|_| "PTX is not UTF-8")?;
    let at = text.find(&format!(".entry {entry}(")).ok_or("missing entry")?;
    let body = at + text[at..].find("\n{").ok_or("no body")?;
    text.insert_str(body + 1, &format!(".maxnreg {registers}\n"));
    Ok(text.into_bytes())
}
pub(crate) fn run(production: &str, fp4: &str, mode: &str) -> Result<()> {
    // "<mode>:q" runs the mode on the quarter-chain word function (second matrices).
    let (mode, quarter) = match mode.strip_suffix(":q") { Some(m) => (m, true), None => (mode, false) };
    if quarter { CHAINS.store(4, std::sync::atomic::Ordering::Relaxed); }
    println!("FP4COHORT WORD FUNCTION chains={} ({})", chains(), if quarter { "second matrices: four quarter chains, joined fl(fl(a0+a1)+fl(a2+a3))" } else { "first matrices: one chain" });
    let (prod, image) = (std::fs::read(production).map_err(|e| e.to_string())?, std::fs::read(fp4).map_err(|e| e.to_string())?);
    println!("FP4COHORT production_sha256={} fp4_sha256={} mode={mode}", sha256::digest(&prod), sha256::digest(&image));
    let mut g = Gpu::new()?;
    let (mp, mf, mc) = (g.module(&prod)?, g.module(&image)?, g.module(&capped(&image, "fp4_cohort", 128)?)?);
    // The decode baseline is the rc.8 chain: `nv_cols_rc8` on an image after the tensor-core rebase, `nv_cols` before it.
    let rebased = g.function(mp, "nv_cols_rc8").is_ok();
    let (cols_fn, free, cap) = (g.function(mp, if rebased { "nv_cols_rc8" } else { "nv_cols" })?, (g.function(mf, "fp4_cohort")?, 128), (g.function(mc, "fp4_cohort")?, 128));
    println!("FP4COHORT DECODE BASELINE {}", if rebased { "nv_cols_rc8 (image after the rebase: nv_cols itself runs the contract)" } else { "nv_cols" });
    // v3 forms of the same contract: 64-token blocks (3 stages) and 8-token blocks (4 stages, the decode step).
    let (t64, t8) = ((g.function(mf, "fp4_cohort_t64")?, 64), (g.function(mf, "fp4_cohort_t8")?, 8));
    // The 64-token and 8-token forms carry the quarter chain; the 128-token form and cc take one chain only.
    let others: Vec<(&str, F)> = if quarter { vec![("t8", t8)] } else { vec![("uncapped", free), ("t64", t64), ("t8", t8)] };
    if quarter && matches!(mode, "cc" | "decode" | "decode-cc" | "diag" | "peak") { return Err(format!("mode {mode} has no quarter-chain form")); }
    // The CUDA-core form of the same word function (one token a block, column tiles only).
    let cc = (g.function(mf, "fp4_cohort_cc")?, 1);
    if mode == "diag" {
        diag(&mut g, &[("maxnreg128", cap), ("uncapped", free), ("t64", t64), ("t8", t8)])?;
        if let Err(e) = calibrate(&mut g, free) { println!("FP4DIAG calibration on the uncapped form: {e}"); }
    }
    if mode == "gates" || mode == "all" {
        let t = gates(&mut g, if quarter { t64 } else { cap }, &others)?;
        if quarter { println!("FP4COHORT GATES PASS twin={t:?} chains=4 (gates on the 64-token form; the 8-token form equal on every edge word)"); println!("FP4COHORT DONE"); return Ok(()); }
        let fx = fixture(128, 4096, &[256, 3], Kind::Random, 5);
        let d = upload(&mut g, &fx, 1, None)?;
        for o in &d.outputs { g.upload_at(o.0, &vec![0xffu8; o.1 * 4])?; }
        launch(&g, free, &d, 0, 0)?; g.sync()?; let x = words(&g, &d)?;
        launch(&g, cap, &d, 0, 0)?; g.sync()?; let y = words(&g, &d)?;
        if x != y { return Err("FP4COHORT FAIL capped and uncapped forms differ".into()); }
        g.clear()?;
        println!("FP4COHORT GATES PASS twin={t:?} (gates on the 128-register form; uncapped form equal on every word of a 2-part fixture)");
    }
    if mode == "decode" {
        for (label, f) in [("t8", t8), ("t64", t64)] { println!("FP4COHORT FORM {label}"); decode_time(&mut g, cols_fn, f, Third::RowMajor, 1)?; }
    }
    if mode == "cc" {
        // The CUDA-core form against the twin, every word (column tiles), on the edge fixtures and the zero fixture.
        let mut failed = 0usize;
        let edge: Vec<(usize, usize, Vec<usize>, Kind)> = vec![
            (128, 4096, vec![40], Kind::Random), (128, 1024, vec![20, 1, 9], Kind::Small), (256, 512, vec![33, 7], Kind::Sparse),
            (128, 2048, vec![30], Kind::Cancel), (128, 1024, vec![24], Kind::Spread), (128, 64, vec![1, 2, 8, 9, 31], Kind::Random),
            (384, 128, vec![17, 28], Kind::Random), (128, 8192, vec![5], Kind::Random), (256, 1088, vec![6, 3], Kind::Random),
        ];
        let mut n = 0usize;
        for (i, (rows, cols, counts, kind)) in edge.iter().enumerate() {
            let fx = fixture(*rows, *cols, counts, *kind, 100 + i as u64);
            let d1 = upload(&mut g, &fx, 1, None)?;
            match compare(&mut g, cc, &fx, &d1, DEVICE_MODEL, true, &format!("cc edge{i}")) { Ok((x, _)) => n += x, Err(e) => { println!("{e}"); failed += 1; } }
            if let Err(e) = forms_equal(&mut g, t8, cc, &d1, 0, &format!("cc edge{i} vs t8")) { println!("{e}"); failed += 1; }
            g.clear()?;
        }
        println!("FP4COHORT CC G1 {} fixtures={} words_x(raw,final)={}", if failed == 0 { "PASS every word = twin and = t8" } else { "FAIL (lines above)" }, edge.len(), n * 2);
        if failed > 0 { return Err(format!("FP4COHORT CC FAIL: {failed} checks failed")); }
    }
    if mode == "p2" {
        if quarter { alone_batch(&mut g, &[("t64", t64), ("t8", t8)], DEVICE_MODEL)?; }
        else { alone_batch(&mut g, &[("t128-capped", cap), ("t128", free), ("t64", t64), ("t8", t8), ("cc", cc)], DEVICE_MODEL)?; }
    }
    if mode == "p2tc" {
        alone_batch(&mut g, &[("t128-capped", cap), ("t128", free), ("t64", t64), ("t8", t8)], DEVICE_MODEL)?;
    }
    if mode == "colsb" {
        // After the rebase the serving entry `nv_cols` picks the word function from the shape (role = NV_SECOND).
        let name = if rebased { "nv_cols" } else if quarter { "nv_cols_bq" } else { "nv_cols_b" };
        let kernel = g.function(mp, name)?;
        cols_b_gate(&mut g, kernel)?;
        println!("FP4COHORT FORM t8 + {name}");
        decode_time(&mut g, cols_fn, t8, Third::NvAbi { kernel, name, shared: nv_shared, check: true }, 1)?;
    }
    if mode == "peak" { let f = g.function(mf, "fp4_peak")?; peak(&mut g, f)?; }
    if mode == "decode-cc" {
        println!("FP4COHORT FORM cc (no zero-skip)"); decode_time(&mut g, cols_fn, cc, Third::Same, 0)?;
        println!("FP4COHORT FORM t8"); decode_time(&mut g, cols_fn, t8, Third::RowMajor, 1)?;
    }
    if mode == "time" || mode == "all" {
        let forms: Vec<(&str, F)> = if quarter { vec![("t8", t8), ("t64", t64)] } else { vec![("t8", t8), ("maxnreg128", cap), ("t64", t64), ("uncapped", free)] };
        for (label, f) in forms {
            println!("FP4COHORT FORM {label}");
            prompt_time(&mut g, f)?; decode_time(&mut g, cols_fn, f, Third::RowMajor, 1)?;
        }
    }
    println!("FP4COHORT DONE");
    Ok(())
}
