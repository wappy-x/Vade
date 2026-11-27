//! Same-bytes sweep and timing for `mx_rows_small2/4/8/8w` (2 to 8 tokens) against the production MXFP8 prompt-row path
//! (`apply_rows`: groups of 16 tokens, kernel chosen by token count), and against the kernels these entries replaced when the
//! baseline image is older than ROWS-MX (its `mx_rows_small2/4` at 2 to 4 tokens, its `mx_rows_tile` at 5 to 8), in one process
//! on one device.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, sha256};
use std::time::Instant;
use super::envelope as e;
const GUARD: [u8; 16] = [0x5a; 16];
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
struct Buf { base: u64, address: u64, bytes: usize }
impl Buf {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// Decoded activation values. 0 plain, 1 wide range, 2 denormal and +-0, 3 large, 4 sparse NaN and inf.
fn activations(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| { let (u, s) = (r.unit(), r.next()); match kind {
        0 => u * 4.0, 1 => u * f32::from_bits(((s % 40 + 107) as u32) << 23),
        2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
        3 => u.signum() * (1.0 + u.abs()) * 2f32.powi(40),
        _ => if i % 7919 == 5 { f32::NAN } else if i % 7919 == 21 { f32::INFINITY } else { u * 8.0 } } }).collect()
}
#[derive(Clone, Copy)]
struct Proj { weights: u64, scales: u64, input: u64, output: u64, rows: u32, cols: u32, tokens: u32 }
fn proj_bytes(p: &Proj) -> Vec<u8> {
    let mut v = vec![]; for x in [p.weights, p.scales, p.input, p.output] { v.extend(x.to_le_bytes()); }
    for x in [p.rows, p.cols, p.tokens, 0, 0] { v.extend(x.to_le_bytes()); } v.extend([0u8; 4]); v
}
/// Mirror of MxProjection::tile and kernel (round 0).
fn tile(p: &Proj) -> u32 {
    if p.tokens > 1 && p.tokens <= 8 { 8 } else if p.tokens == 1 || p.rows < 128 { 1 } else if p.rows <= 640 || p.cols >= 4096 { if p.tokens <= 2 { 2 } else { 4 } } else { p.tokens }
}
fn kernel(p: &Proj) -> &'static str { if p.tokens == 1 { "flash_next_mxfp8_m1" } else if p.tokens <= 8 { "flash_next_mxfp8_pair" } else { "flash_next_mxfp8_bits" } }
/// The production loop: groups of 16 tokens, one launch each.
fn launch_reference(g: &Gpu, fns: &[(&str, H)], p: &Proj) -> Result<()> {
    let mut token = 0u32;
    while token < p.tokens {
        let count = (p.tokens - token).min(16);
        let q = Proj { input: p.input + (token as u64) * (p.cols as u64) * 4, output: p.output + (token as u64) * (p.rows as u64) * 4, tokens: count, ..*p };
        let blocks = q.rows.div_ceil(4) * q.tokens.div_ceil(tile(&q));
        let mut layer = proj_bytes(&q); layer.resize(9 * 56, 0);
        let mut ends = [0u32; 10]; ends[1] = blocks; for e in ends { layer.extend(e.to_le_bytes()); } layer.extend(1u32.to_le_bytes()); layer.extend([0u8; 4]);
        let f = fns.iter().find(|f| f.0 == kernel(&q)).ok_or("kernel")?.1;
        g.launch(f, [blocks, 1, 1], 128, 0, &mut [layer.as_mut_ptr().cast()])?;
        token += count;
    }
    Ok(())
}
fn launch_tile(g: &Gpu, f: H, p: &Proj) -> Result<()> {
    let mut a = proj_bytes(p);
    g.launch(f, [p.rows.div_ceil(48), p.tokens.div_ceil(16), 1], 384, 0, &mut [a.as_mut_ptr().cast()])
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Fixture { p: Proj }
/// mode 1: row r holds code (r + column) % 256 and scale byte r / 256.
fn build(g: &mut Gpu, rows: usize, cols: usize, tokens: usize, mode: u32, kind: u32, seed: u64, out_words: usize) -> Result<(Fixture, Buf, Buf)> {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..rows * cols).map(|i| if mode == 1 { ((i / cols + i % cols) % 256) as u8 } else { let x = r.next(); let code = (x >> 8) as u8; if code & 127 == 127 && !x.is_multiple_of(4096) { code & 0xf0 | 5 } else { code } }).collect();
    let s: Vec<u8> = (0..rows * cols / 32).map(|i| if mode == 1 { ((i / (cols / 32)) / 256) as u8 } else { let x = r.next(); if x.is_multiple_of(32) { (x >> 8) as u8 } else { 110 + ((x >> 8) % 36) as u8 } }).collect();
    let wb = Buf::new(g, &w)?; let sb = Buf::new(g, &s)?; let xb = Buf::new(g, &le(&activations(kind, tokens * cols, seed + 5)))?;
    let oa = Buf::new(g, &vec![0xff; out_words * 4])?; let ob = Buf::new(g, &vec![0xff; out_words * 4])?;
    let p = Proj { weights: wb.address, scales: sb.address, input: xb.address, output: oa.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
    Ok((Fixture { p }, oa, ob))
}
/// Shared weights and scales for every token count and activation kind of one shape (random codes, plain scale bytes).
fn weights_of(g: &mut Gpu, rows: usize, cols: usize, seed: u64) -> Result<(u64, u64)> {
    let mut r = Rng(seed | 1);
    let w: Vec<u8> = (0..rows * cols).map(|_| { let x = r.next(); let code = (x >> 8) as u8; if code & 127 == 127 && !x.is_multiple_of(4096) { code & 0xf0 | 5 } else { code } }).collect();
    let sc: Vec<u8> = (0..rows * cols / 32).map(|_| { let x = r.next(); if x.is_multiple_of(32) { (x >> 8) as u8 } else { 110 + ((x >> 8) % 36) as u8 } }).collect();
    Ok((Buf::new(g, &w)?.address, Buf::new(g, &sc)?.address))
}
fn build_shared(g: &mut Gpu, rows: usize, cols: usize, tokens: usize, kind: u32, seed: u64, wa: (u64, u64)) -> Result<(Fixture, Buf, Buf)> {
    let xb = Buf::new(g, &le(&activations(kind, tokens * cols, seed + 5)))?;
    let oa = Buf::new(g, &vec![0xff; rows * tokens * 4])?; let ob = Buf::new(g, &vec![0xff; rows * tokens * 4])?;
    let p = Proj { weights: wa.0, scales: wa.1, input: xb.address, output: oa.address, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 };
    Ok((Fixture { p }, oa, ob))
}
/// `mx_rows_small*`: mirror of `mx_small::entry` and `grid`: (entry, tokens a block, weight rows a warp); grid (ceil(rows / (4 x rows a warp)), ceil(tokens / T)).
fn small_entry(tokens: u32, cols: u32) -> (&'static str, u32, u32) {
    match tokens { 2 => ("mx_rows_small2", 2, 2), 3 | 4 => ("mx_rows_small4", 4, 4), _ => if cols > 4096 { ("mx_rows_small8w", 8, 4) } else { ("mx_rows_small8", 8, 2) } }
}
fn launch_small(g: &Gpu, fs: &[(&str, H)], p: &Proj) -> Result<()> {
    let (name, per, warp_rows) = small_entry(p.tokens, p.cols); let f = fs.iter().find(|f| f.0 == name).ok_or("small entry")?.1;
    let mut a = proj_bytes(p);
    g.launch(f, [p.rows.div_ceil(4 * warp_rows), p.tokens.div_ceil(per), 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
/// The kernel a call of 2 to 8 tokens ran before ROWS-MX, from a baseline image older than it: `mx_rows_small2/4` with 16 rows a
/// block at 2 to 4 tokens, `mx_rows_tile` at 5 to 8.
fn launch_replaced(g: &Gpu, olds: &[(&str, H)], tile: H, p: &Proj) -> Result<()> {
    if p.tokens > 4 { return launch_tile(g, tile, p); }
    let (name, per) = if p.tokens == 2 { ("mx_rows_small2", 2) } else { ("mx_rows_small4", 4) };
    let f = olds.iter().find(|f| f.0 == name).ok_or("replaced small entry")?.1;
    let mut a = proj_bytes(p);
    g.launch(f, [p.rows.div_ceil(16), p.tokens.div_ceil(per), 1], 128, 0, &mut [a.as_mut_ptr().cast()])
}
/// `head`: the output head of one rank (77,440 x 4,096; `mx_rows::HEAD_ROWS`). Bytes: the candidate image's `mx_rows_small4` at 3 and 4
/// tokens and its tile at 5 to 8 (the host rule of `small_choice::mx`) against the production M kernels, five activation kinds, with a
/// shifted-rows control. Time: the production kernels, the tile, the small entries and ONE token, at 2 to 8 tokens.
fn head(g: &mut Gpu, fns: &[(&str, H)], tile: H, smalls: &[(&str, H)], rows:usize) -> Result<()> {
    let cols = 4096usize; let started = Instant::now();
    let wa = weights_of(g, rows, cols, 4242)?;
    let (mut comparisons, mut controls) = (0u64, 0u64);
    for tokens in 3..=8usize { for kind in 0..5u32 {
        let (fx, oa, ob) = build_shared(g, rows, cols, tokens, kind, 2000 + tokens as u64 * 11 + kind as u64, wa)?;
        oa.poison(g)?; ob.poison(g)?;
        // The host rule: `mx_rows_small4` at 3 and 4 tokens, the tile at 5 to 8.
        launch_reference(g, fns, &fx.p)?;
        if tokens <= 4 { launch_small(g, smalls, &Proj { output: ob.address, ..fx.p })?; } else { launch_tile(g, tile, &Proj { output: ob.address, ..fx.p })?; }
        g.sync()?;
        let (x, y) = (oa.read(g)?, ob.read(g)?);
        if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXSMALL HEAD BYTE FAIL tokens={tokens} act_kind={kind} first_word_and_byte={at:?}")); }
        if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err("MXSMALL head guard changed".into()); }
        comparisons += 1;
        if kind == 0 {
            ob.poison(g)?;
            let q = Proj { output: ob.address, input: fx.p.input + (cols * 4) as u64, tokens: tokens as u32 - 1, ..fx.p };
            if tokens <= 4 { launch_small(g, smalls, &q)?; } else { launch_tile(g, tile, &q)?; }
            g.sync()?;
            let shifted = ob.read(g)?; let n = 16 + (tokens - 1) * rows * 4;
            if shifted[16..n] == x[16..n] { return Err(format!("MXSMALL HEAD CONTROL did not fail tokens={tokens}")); }
            controls += 1;
        }
    } }
    println!("MXSMALL HEAD BYTES PASS rows={rows} cols={cols} mx_rows_small4 at 3 and 4 tokens and the tile at 5 to 8 equal the production kernels comparisons={comparisons} controls_failed_as_they_must={controls} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    g.clear()?;
    // Time: weights and scales at the start of their allocation, as the engine's are; two copies, so a launch reads cold memory.
    let k = 2usize; let mut ws = vec![];
    for i in 0..k {
        let mut r = Rng(77 + i as u64 * 13 + rows as u64);
        let w: Vec<u8> = (0..rows * cols).map(|_| { let c = (r.next() >> 24) as u8; if c & 127 == 127 { c & 0xfe } else { c } }).collect();
        let sc: Vec<u8> = (0..rows * cols / 32).map(|_| 120 + (r.next() % 14) as u8).collect();
        ws.push((g.upload(&w)?, g.upload(&sc)?));
    }
    let bytes = (rows * cols) as f64 * (1.0 + 1.0 / 32.0);
    for tokens in 2..=8usize {
        let mut ps = vec![];
        for (i, w) in ws.iter().enumerate() {
            let x = g.upload(&le(&activations(0, tokens * cols, 36 + i as u64)))?; let out = g.upload(&vec![0u8; rows * tokens * 4])?;
            ps.push(Proj { weights: w.0, scales: w.1, input: x, output: out, rows: rows as u32, cols: cols as u32, tokens: tokens as u32 });
        }
        let mut t: [Vec<f64>; 4] = Default::default();
        for repeat in 0..5 { for step in 0..4 { let arm = (step + repeat) % 4;
            let run = |g: &Gpu, i: usize| -> Result<()> { let p = &ps[i % k]; match arm {
                0 => launch_reference(g, fns, p), 1 => launch_tile(g, tile, p), 2 => launch_small(g, smalls, p), _ => launch_reference(g, fns, &Proj { tokens: 1, ..*p }) } };
            for i in 0..k { run(g, i)?; } g.sync()?;
            let t0 = Instant::now(); for i in 0..4 * k { run(g, i)?; } g.sync()?;
            t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / (4 * k) as f64);
        } }
        let us = t.map(median);
        println!("MXSMALL HEAD TIME tokens={tokens} production_us={:.0} ({:.0} GB/s) mx_rows_tile_us={:.0} ({:.0} GB/s) mx_rows_small_us={:.0} ({:.0} GB/s) one_token_us={:.0} ({:.0} GB/s)",
            us[0], bytes / us[0] / 1e3, us[1], bytes / us[1] / 1e3, us[2], bytes / us[2] / 1e3, us[3], bytes / us[3] / 1e3);
    }
    Ok(())
}
pub(crate) fn run(base_path: &str, candidate_path: &str, mode: &str) -> Result<()> {
    let time = mode == "time";
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXSMALL baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let names = ["flash_next_mxfp8_m1", "flash_next_mxfp8_pair", "flash_next_mxfp8_bits"];
    let fns: Vec<(&str, H)> = names.iter().map(|n| g.function(ma, n).map(|h| (*n, h))).collect::<Result<_>>()?;
    let tile_fn = g.function(ma, "mx_rows_tile")?;
    let smalls: Vec<(&str, H)> = ["mx_rows_small2", "mx_rows_small4", "mx_rows_small8", "mx_rows_small8w"].iter().map(|n| g.function(mb, n).map(|h| (*n, h))).collect::<Result<_>>()?;
    // The replaced kernels exist only in a baseline image older than ROWS-MX (one without `mx_rows_small8`).
    let olds: Vec<(&str, H)> = if g.function(ma, "mx_rows_small8").is_ok() { vec![] } else { ["mx_rows_small2", "mx_rows_small4"].iter().map(|n| g.function(ma, n).map(|h| (*n, h))).collect::<Result<_>>()? };
    println!("MXSMALL replaced_kernel_arm={}", if olds.is_empty() { "absent (the baseline image has the new entries)" } else { "present" });
    if matches!(mode,"head"|"head-quarter") {
        let tile_b=g.function(mb,"mx_rows_tile")?;
        return head(&mut g,&fns,tile_b,&smalls,if mode=="head-quarter" {38_720} else {77_440});
    }
    let started = Instant::now();
    if !time {
        let (comparisons, compared, replaced) = (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
        let check = |g: &mut Gpu, rows: usize, cols: usize, tokens: usize, mode: u32, kind: u32, seed: u64| -> Result<()> {
            let (fx, oa, ob) = build(g, rows, cols, tokens, mode, kind, seed, rows * tokens)?;
            oa.poison(g)?; ob.poison(g)?;
            launch_reference(g, &fns, &fx.p)?; launch_small(g, &smalls, &Proj { output: ob.address, ..fx.p })?; g.sync()?;
            let (x, y) = (oa.read(g)?, ob.read(g)?);
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXSMALL BYTE FAIL rows={rows} cols={cols} tokens={tokens} mode={mode} act_kind={kind} first_word_and_byte={at:?}")); }
            if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXSMALL guard changed".into()); }
            if !olds.is_empty() {
                oa.poison(g)?; launch_replaced(g, &olds, tile_fn, &fx.p)?; g.sync()?;
                if oa.read(g)? != y { return Err(format!("MXSMALL REPLACED-KERNEL BYTE FAIL rows={rows} cols={cols} tokens={tokens} mode={mode} act_kind={kind}")); }
                replaced.set(replaced.get() + 1);
            }
            comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * tokens * 4) as u64); g.clear()
        };
        // All 256 weight codes x all 256 scale bytes (cols 256: every code appears in every row).
        for (kind, tokens) in [(0u32, 2usize), (1, 3), (2, 4), (3, 2), (4, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 8)] { check(&mut g, 65536, 256, tokens, 1, kind, 11 + kind as u64 + tokens as u64 * 7)?; }
        println!("MXSMALL all 256x256 code pairs ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        // Every listed (rows, cols) of `small_sweep` (weights at most MX_MAX_BYTES) at every token count 2 to 8 and five activation kinds.
        let mut seed = 500u64;
        for &rows in e::MX_ROWS { for &cols in e::MX_COLS {
            if rows * cols > e::MX_MAX_BYTES { continue; }
            let wa = weights_of(&mut g, rows, cols, 7 + rows as u64 * 131 + cols as u64)?;
            for tokens in e::MX_MIN_TOKENS..=e::MX_MAX_TOKENS { for kind in 0..5u32 {
                seed += 1;
                let (fx, oa, ob) = build_shared(&mut g, rows, cols, tokens, kind, seed, wa)?;
                oa.poison(&g)?; ob.poison(&g)?;
                launch_reference(&g, &fns, &fx.p)?; launch_small(&g, &smalls, &Proj { output: ob.address, ..fx.p })?; g.sync()?;
                let (x, y) = (oa.read(&g)?, ob.read(&g)?);
                if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXSMALL BYTE FAIL rows={rows} cols={cols} tokens={tokens} act_kind={kind} first_word_and_byte={at:?}")); }
                if x[..16] != GUARD || x[x.len() - 16..] != GUARD { return Err("MXSMALL guard changed".into()); }
                if !olds.is_empty() {
                    oa.poison(&g)?; launch_replaced(&g, &olds, tile_fn, &fx.p)?; g.sync()?;
                    if oa.read(&g)? != y { return Err(format!("MXSMALL REPLACED-KERNEL BYTE FAIL rows={rows} cols={cols} tokens={tokens} act_kind={kind}")); }
                    replaced.set(replaced.get() + 1);
                }
                comparisons.set(comparisons.get() + 1); compared.set(compared.get() + (rows * tokens * 4) as u64);
            } }
            g.clear()?;
        } }
        println!("MXSMALL sweep ok comparisons={} elapsed_s={:.1}", comparisons.get(), started.elapsed().as_secs_f64());
        for (rows, cols) in [(4096usize, 4096usize), (2048, 8192), (6144, 4096), (2048, 12288)] { for tokens in 2..=8usize { for kind in 0..5u32 { check(&mut g, rows, cols, tokens, 0, kind, 900 + rows as u64 + kind as u64 + tokens as u64)?; } } }
        // The host rule (`small_choice::mx`) sends 5 to 8 tokens of the 4,096-column shapes to `mx_rows_tile`: the candidate image's tile against the
        // small entries (which the sweep above holds to the production M kernels), on the four served shapes, five activation kinds. Control: the tile
        // on the rows shifted by one token must NOT give the same bytes.
        let tile_b = g.function(mb, "mx_rows_tile")?; let (mut tiles, mut controls) = (0u64, 0u64);
        for (rows, cols) in [(4096usize, 4096usize), (6144, 4096), (2048, 8192), (2048, 12288)] { for tokens in 5..=8usize { for kind in 0..5u32 {
            let (fx, oa, ob) = build(&mut g, rows, cols, tokens, 0, kind, 1300 + rows as u64 + kind as u64 * 3 + tokens as u64 * 17, rows * tokens)?;
            oa.poison(&g)?; ob.poison(&g)?;
            launch_small(&g, &smalls, &fx.p)?; launch_tile(&g, tile_b, &Proj { output: ob.address, ..fx.p })?; g.sync()?;
            let (x, y) = (oa.read(&g)?, ob.read(&g)?);
            if x != y { let at = x.iter().zip(&y).position(|(p, q)| p != q).map(|i| (i.saturating_sub(16) / 4, i)); return Err(format!("MXSMALL TILE BYTE FAIL rows={rows} cols={cols} tokens={tokens} act_kind={kind} first_word_and_byte={at:?}")); }
            if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err("MXSMALL tile guard changed".into()); }
            tiles += 1;
            if kind == 0 {
                ob.poison(&g)?; launch_tile(&g, tile_b, &Proj { output: ob.address, input: fx.p.input + (cols * 4) as u64, tokens: tokens as u32 - 1, ..fx.p })?; g.sync()?;
                let shifted = ob.read(&g)?; let n = 16 + (tokens - 1) * rows * 4;
                if shifted[16..n] == x[16..n] { return Err(format!("MXSMALL TILE CONTROL did not fail rows={rows} cols={cols} tokens={tokens}")); }
                controls += 1;
            }
            g.clear()?;
        } } }
        println!("MXSMALL TILE at 5 to 8 tokens equals the small entries comparisons={tiles} controls_failed_as_they_must={controls} elapsed_s={:.1}", started.elapsed().as_secs_f64());
        // The recorded wrapper of the tile (`resident_launch_mx_rows_tile`, Kind 68: the launch a decode step makes for 5 to 8 rows) against the
        // direct entry: same bytes, and the launch itself is accepted (384 threads a block: registers and shared memory).
        let wrapper = g.function(mb, "resident_launch_mx_rows_tile")?; let mut wrapped = 0u64;
        for (rows, cols) in [(4096usize, 4096usize), (6144, 4096)] { for tokens in 5..=8usize {
            let (fx, oa, ob) = build(&mut g, rows, cols, tokens, 0, (tokens % 5) as u32, 1700 + rows as u64 + tokens as u64 * 29, rows * tokens)?;
            oa.poison(&g)?; ob.poison(&g)?;
            launch_tile(&g, tile_b, &fx.p)?;
            super::fp8small::launch_wrapped(&mut g, wrapper, 68, &proj_bytes(&Proj { output: ob.address, ..fx.p }), [rows.div_ceil(48) as u32, tokens.div_ceil(16) as u32, 1], 384, 0).map_err(|e| format!("MXSMALL WRAPPER LAUNCH FAIL rows={rows} tokens={tokens}: {e}"))?;
            g.sync().map_err(|e| format!("MXSMALL WRAPPER RUN FAIL rows={rows} tokens={tokens}: {e}"))?;
            if oa.read(&g)? != ob.read(&g)? { return Err(format!("MXSMALL WRAPPER BYTE FAIL rows={rows} cols={cols} tokens={tokens}")); }
            wrapped += 1; g.clear()?;
        } }
        println!("MXSMALL WRAPPER PASS resident_launch_mx_rows_tile equals mx_rows_tile comparisons={wrapped}");
        println!("MXSMALL BYTES PASS comparisons={} replaced_kernel_comparisons={} output_bytes={} digest={} elapsed_s={:.1}", comparisons.get(), replaced.get(), compared.get(), sha256::digest(format!("{}{}", comparisons.get(), compared.get()).as_bytes()), started.elapsed().as_secs_f64());
        return Ok(());
    }
    // Rank-local MXFP8 shapes of the model and their calls in a pass: q, k, v 34 layers each (4096 x 4096), o 34 (2048 x 8192), dense gate and up 3 each (6144 x 4096), down 3 (2048 x 12288).
    // Weights and scales start at their allocation, as the engine's do (finite codes, plain scales, as --mx-wave). The guarded sweep buffers
    // sit 16 bytes past a line start, where EVERY MXFP8 kernel reads about 15 percent slower (one Spark, 2026-11-12: the one-token kernel 12.7 against 14.8 ms a pass).
    let shapes: [(usize, usize, f64, &str); 4] = [(4096, 4096, 102.0, "qkv"), (2048, 8192, 34.0, "o"), (6144, 4096, 6.0, "gate_up"), (2048, 12288, 3.0, "down")];
    let names = ["production", "mx_rows_tile", "replaced", "one_token_m1", "mx_rows_small", "mx_rows_small_wrapped", "mx_rows_tile_wrapped"];
    println!("MXSMALL TIME arms: production (the M kernels in 16-token groups) | mx_rows_tile | replaced (the baseline's kernel for the count) | one_token_m1 (ONE token, the rate to reach) | mx_rows_small (candidate) | the candidate's small entry and its tile as a decode step records them (the resident launch wrappers); us a launch and GB/s");
    let small_wrappers: Vec<(&str, H, u32)> = [("mx_rows_small2", 42u32), ("mx_rows_small4", 43), ("mx_rows_small8", 44), ("mx_rows_small8w", 45)].iter().map(|(n, k)| g.function(mb, &format!("resident_launch_{n}")).map(|h| (*n, h, *k))).collect::<Result<_>>()?;
    let tile_wrapper = g.function(mb, "resident_launch_mx_rows_tile")?;
    // The activations: `input_offset=0` at the start of their allocation, as an engine vector is (a fresh device buffer, or a token row of one);
    // `input_offset=16` in the guarded sweep buffer, 16 bytes past a line start (the row entries read the activations straight from the buffer).
    let mut totals: Vec<(usize, usize, [f64; 7])> = vec![];
    for &input_offset in &[0usize, 16] { for &(rows, cols, calls, label) in &shapes {
        let bytes = (rows * cols) as f64 * (1.0 + 1.0 / 32.0);
        for &tokens in &[2usize, 3, 4, 5, 8] {
            let k = ((150e6 / bytes).ceil() as usize).clamp(2, 32);
            let mut fxs = vec![]; for i in 0..k {
                let mut fx = build(&mut g, rows, cols, tokens, 0, 0, 31 + i as u64, rows * tokens)?.0;
                let mut r = Rng(77 + i as u64 * 13 + rows as u64);
                let w: Vec<u8> = (0..rows * cols).map(|_| { let c = (r.next() >> 24) as u8; if c & 127 == 127 { c & 0xfe } else { c } }).collect();
                let sc: Vec<u8> = (0..rows * cols / 32).map(|_| 120 + (r.next() % 14) as u8).collect();
                fx.p.weights = g.upload(&w)?; fx.p.scales = g.upload(&sc)?;
                if input_offset == 0 { fx.p.input = g.upload(&le(&activations(0, tokens * cols, 36 + i as u64)))?; }
                fxs.push(fx);
            }
            let commands: Vec<(u64, u64)> = fxs.iter().map(|fx| { let a = proj_bytes(&fx.p); Ok((super::fp8small::command_of(&mut g, small_wrappers.iter().find(|w| w.0 == small_entry(fx.p.tokens, fx.p.cols).0).ok_or("small wrapper")?.2, &a)?, super::fp8small::command_of(&mut g, 68, &a)?)) }).collect::<Result<_>>()?;
            let per = 4 * k;
            let mut t: [Vec<f64>; 7] = Default::default();
            for repeat in 0..5 { for step in 0..7 { let arm = (step + repeat) % 7;
                if arm == 2 && olds.is_empty() { continue; }
                let run = |g: &Gpu, i: usize| -> Result<()> { let p = &fxs[i % k].p; match arm {
                    0 => launch_reference(g, &fns, p), 1 => launch_tile(g, tile_fn, p), 2 => launch_replaced(g, &olds, tile_fn, p),
                    3 => launch_reference(g, &fns, &Proj { tokens: 1, ..*p }), 4 => launch_small(g, &smalls, p),
                    5 => { let (name, per_block, warp_rows) = small_entry(p.tokens, p.cols); let w = small_wrappers.iter().find(|w| w.0 == name).ok_or("small wrapper")?.1;
                        super::fp8small::launch_command(g, w, commands[i % k].0, [p.rows.div_ceil(4 * warp_rows), p.tokens.div_ceil(per_block), 1], 128, 0) }
                    _ => super::fp8small::launch_command(g, tile_wrapper, commands[i % k].1, [p.rows.div_ceil(48), p.tokens.div_ceil(16), 1], 384, 0) } };
                for i in 0..k { run(&g, i)?; } g.sync()?;
                let t0 = Instant::now(); for i in 0..per { run(&g, i)?; } g.sync()?;
                t[arm].push(t0.elapsed().as_secs_f64() * 1e6 / per as f64);
            } }
            let us: [f64; 7] = t.map(|v| if v.is_empty() { f64::NAN } else { median(v) });
            println!("MXSMALL TIME input_offset={input_offset} {label} rows={rows} cols={cols} tokens={tokens} {} floor_us_at_230GBs={:.0}", (0..7).map(|a| format!("{}_us={:.1} ({:.0} GB/s)", names[a], us[a], bytes / us[a] / 1e3)).collect::<Vec<_>>().join(" "), bytes / 230e3);
            totals.push((input_offset, tokens, us.map(|u| u * calls / 1e3)));
            g.clear()?;
        }
    } }
    for &input_offset in &[0usize, 16] { for &tokens in &[2usize, 3, 4, 5, 8] {
        let mut sum = [0.0f64; 7];
        for t in totals.iter().filter(|t| t.0 == input_offset && t.1 == tokens) { for a in 0..7 { sum[a] += t.2[a]; } }
        println!("MXSMALL PASS input_offset={input_offset} tokens={tokens} ms_a_pass: {}", (0..7).map(|a| format!("{}={:.2}", names[a], sum[a])).collect::<Vec<_>>().join(" "));
    } }
    println!("MXSMALL TIME DONE elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
