//! kind 75: the FP8 ring's BF16 decode projections (flash_bf16_mma1_* row kernels) at the
//! world-2 rank shapes, with the decode entry, grid and threads of blocks.rs project_rows
//! (tokens <= BF16_ROWS_TOKENS, FP32 input, input_bf16 0). Weights rotate as finite BF16.
use super::*;

/// One rank of the FP8 pair: (tag, rows of matrix 1, rows of matrix 2, K, launches per layer
/// pass): GDN qkv+z pair, a+b pair, out (36 layers); QSA q/gate+index pair, k+v pair, out (12);
/// the drafter's MTP fc (H x H) and the QSA index matrix alone.
pub(super) const SHAPES: [(&str, usize, usize, usize, usize); 8] = [
    ("gdn-qkvz", 5120, 3072, 2560, 36), ("gdn-ab", 24, 24, 2560, 36), ("gdn-out", 2560, 0, 3072, 36),
    ("qsa-qgi", 6144, 640, 2560, 12), ("qsa-kv", 256, 256, 2560, 12), ("qsa-out", 2560, 0, 3072, 12),
    ("mtp-fc", 2560, 0, 2560, 0), ("qsa-idx", 640, 0, 2560, 0)];

/// Attention matrices alone, as LBF12 launches would run them: (tag, rows, K, launches per pass).
pub(super) const LBF: [(&str, usize, usize, usize); 6] = [("gdn-qkv", 5120, 2560, 36), ("gdn-z", 3072, 2560, 36),
    ("gdn-out", 2560, 3072, 36), ("qsa-qg", 6144, 2560, 12), ("qsa-out", 2560, 3072, 12), ("mtp-fc", 2560, 2560, 0)];
/// The rest of the FP8 pair's BF16 decode rows (own-fp8-kv): (tag, rows 1, rows 2, K, launches
/// per pass, entry, BF16 packed input, HC SiLU epilogue). HC down + injection and HC up run 96
/// times (attention and MLP HC of 48 layers, replicated on both ranks); the shared down and
/// the router + shared gate | shared gate + up pairs run 48 times; PLE key/value once.
pub(super) const REST: [(&str, usize, usize, usize, usize, &str, bool, bool); 5] = [
    ("hc-down", 4, 320, 10240, 96, "flash_bf16_mma1_10240", true, true),
    ("hc-up", 10240, 0, 320, 96, "flash_bf16_mma1_320_bf16", true, false),
    ("shared-down", 2560, 0, 320, 48, "flash_bf16_mma1_320", false, false),
    ("ple-key", 10240, 0, 2560, 1, "flash_bf16_mma1_2560", false, false),
    ("ple-value", 2560, 0, 2560, 1, "flash_bf16_mma1_2560_native", false, false)];
/// flash_bf16_mma1_2560_pairs: pair 0 = router 512 + shared-expert gate 1, pair 1 = this
/// rank's shared gate 320 + up 320 (blocks.rs invariant_pairs, grid Y 2).
pub(super) const PAIRS: [usize; 4] = [512, 1, 320, 320];

pub(crate) fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for m in [1, 3] {
        for (tag, r1, r2, k, calls, kernel, bf16, silu) in REST {
            if m != 1 && silu { continue; }
            out.push(Case { name: format!("fp8k-{tag}-m{m}"), kernel, m, n: r1 + r2, k, kind: 75,
                mats: [r1, r2, usize::from(bf16), usize::from(silu)], calls: if m == 1 { calls } else { 0 }, ..Default::default() });
        }
        // X reads HC up through its LBF12 twin (hc_silu bit 1); A reads the BF16 matrix.
        out.push(Case { name: format!("fp8k-hc-up-lbf-m{m}"), kernel: "flash_bf16_mma1_320_bf16", m, n: 10240, k: 320, kind: 75,
            mats: [10240, 0, 1, 2], calls: if m == 1 { 96 } else { 0 }, ..Default::default() });
        // The same with the FP32 input (the FP8 session's HC up at 2..8 verify rows).
        out.push(Case { name: format!("fp8k-hc-up-lbf32-m{m}"), kernel: "flash_bf16_mma1_320", m, n: 10240, k: 320, kind: 75,
            mats: [10240, 0, 0, 2], ..Default::default() });
        out.push(Case { name: format!("fp8k-route-shared-m{m}"), kernel: "flash_bf16_mma1_2560_pairs", m,
            n: PAIRS.iter().sum(), k: 2560, kind: 76, mats: PAIRS, calls: if m == 1 { 48 } else { 0 }, ..Default::default() });
    }
    for m in [1, 2, 3, 4, 8] { for (tag, r1, r2, k, calls) in SHAPES {
        let kernel = entry(m, r1, r2, k);
        out.push(Case { name: format!("fp8a-{tag}-m{m}"), kernel, m, n: r1 + r2, k, kind: 75, mats: [r1, r2, 0, 0],
            calls: if m == 1 { calls } else { 0 }, ..Default::default() });
    } }
    // One attention matrix per launch (the LBF12 twin takes weight2): A reads BF16, X its twin.
    for m in [1, 3] { for (tag, rows, k, calls) in LBF {
        out.push(Case { name: format!("fp8a-{tag}-lbf-m{m}"), kernel: entry(m, rows, 0, k), m, n: rows, k, kind: 75,
            mats: [rows, 0, 0, 2], calls: if m == 1 { calls } else { 0 }, ..Default::default() });
    } }
    // FP8 HC injection + down at 2..8 FP32 rows (mats[3] bit 4): A = flash_bf16_mma1_10240 (main's
    // entry), X = the engine's latency pack + one flash_bf16_mma_latency_group launch. Bytes A == X.
    for m in [2, 3, 4, 5, 6, 7, 8] {
        out.push(Case { name: format!("fp8k-hcgroup-m{m}"), kernel: "flash_bf16_mma1_10240", m, n: 324, k: 10240, kind: 75,
            mats: [4, 320, 0, 4], ..Default::default() });
    }
    // Wave probe: K2560 twin reads from 2560 to 3072 rows (320 to 384 split CTAs), M1 only.
    for rows in [2560, 2624, 2688, 2752, 2816, 2944, 3072] {
        out.push(Case { name: format!("fp8a-wave-{rows}-lbf-m1"), kernel: entry(1, rows, 0, 2560), m: 1, n: rows, k: 2560, kind: 75,
            mats: [rows, 0, 0, 2], ..Default::default() });
    }
    out
}

/// blocks.rs project_rows' decode entry at tokens <= 8 with an FP32 input.
fn entry(m: usize, r1: usize, r2: usize, k: usize) -> &'static str {
    let rows = r1 + r2;
    if m == 1 && k == 2560 && rows < 64 { return "flash_bf16_mma1_2560_small"; }
    match k {
        2560 if rows < 1024 => "flash_bf16_mma1_2560_narrow",
        2560 if rows < 16384 && r1 % 64 == 0 && r2 % 64 == 0 => "flash_bf16_mma1_2560_native",
        2560 => "flash_bf16_mma1_2560",
        3072 if rows >= 1024 && rows < 16384 && r1 % 64 == 0 && r2 % 64 == 0 => "flash_bf16_mma1_3072_native",
        _ => "flash_bf16_mma1" }
}
/// Its grid and block: one CTA per output (small), else per eight rows of each matrix,
/// halved at 64 threads where flash_bf16_decode_split is false.
#[cfg(target_os = "linux")]
fn shape(c: &Case) -> (usize, u32) {
    let (rows, blocks) = (c.mats[0] + c.mats[1], c.mats[0].div_ceil(8) + c.mats[1].div_ceil(8));
    if c.kind == 76 { return ((c.mats[0].div_ceil(8) + c.mats[1].div_ceil(8)).max(c.mats[2].div_ceil(8) + c.mats[3].div_ceil(8)), 128); }
    if c.kernel == "flash_bf16_mma1_2560_small" { (rows, 128) }
    else if params::flash_bf16_decode_split(rows, c.k) { (blocks, 128) } else { (blocks.div_ceil(2), 64) }
}

#[cfg(target_os = "linux")]
pub(super) fn fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, k) = (c.m, c.k);
    let (r1, r2) = if c.kind == 76 { (c.mats[0] + c.mats[1], c.mats[2] + c.mats[3]) } else { (c.mats[0], c.mats[1]) };
    let (bf16, silu) = if c.kind == 76 { (0, 0) } else { (c.mats[2] as u32, c.mats[3] as u32 & 1) };
    let weight_bytes = (r1 + r2) * k * 2;
    let copies = rotate_bytes().div_ceil(weight_bytes).max(2);
    // LBF12 cases: kind 0 bytes are BF16 with exponent 0x76/0x77 and varied mantissas, so every slot packs.
    let slab = g.fill(weight_bytes * copies, if c.mats[3] & 2 != 0 { 0 } else { 3 })?;
    let ring = (0..copies).map(|i| slab + (i * weight_bytes) as u64).collect::<Vec<_>>();
    // A packed BF16 input holds its planes (8 plane strides is more than every TERMS count).
    let plane_words = m.div_ceil(16) * k.div_ceil(16) * 128;
    let input = if bf16 != 0 { g.fill(plane_words * 8 * 4, 2)? } else { g.fill(m * k * 4, 2)? };
    let out_bytes = m * (r1 + r2) * 4;
    let out = g.alloc(out_bytes)?;
    // Group cases: arm X's packed hi/lo planes of the FP32 input (flash_bf16_input).
    let aux = if c.kind == 75 && c.mats[3] & 4 != 0 { g.fill(plane_words * 2 * 4, 0)? } else { 0 };
    let rows = params::Bf16Rows { weight: ring[0], output: out, rows: r1 as u32, stride: r1 as u32,
        weight2: if r2 == 0 { 0 } else { ring[0] + (r1 * k * 2) as u64 }, output2: if r2 == 0 { 0 } else { out + (m * r1 * 4) as u64 },
        rows2: r2 as u32, stride2: r2 as u32, input, cols: k as u32, tokens: m as u32, input_bf16: bf16, hc_silu: silu,
        input_plane_words: plane_words as u64 };
    let (blocks, threads) = shape(c);
    Ok(Fixture { rows, group: None, ring, out, out_bytes, aux, aux_bytes: 0, grid: [blocks as u32, 1, 1], threads,
        shared: 0, stored: weight_bytes + m * k * 4 + out_bytes, flops: 2 * m * (r1 + r2) * k, chunks: 1, down_slots: 0,
        raw: 0, pack: null_mut(), norms: [null_mut(); 2], weight_bytes, written: Vec::new(), qsa: None, gather: None,
        ring_x: vec![], scales_x: vec![] })
}

#[cfg(target_os = "linux")]
/// One launch on ring copy i; KBENCH_ROWS_X_THREADS / _X_BLOCKS reshape arm X only.
pub(super) fn launch(g: &Gpu, f: H, c: &Case, v: &mut Fixture, i: usize, arm: usize) -> Result<()> {
    let env = |n: &str| std::env::var(n).ok().and_then(|s| s.parse::<u32>().ok()).filter(|_| arm == 1);
    if c.kind == 76 {
        // Four matrices back to back in the ring copy; outputs back to back in `out`.
        let mut at = (v.ring[i], v.out);
        let mut m = [(0, 0, 0); 4];
        for (j, &rows) in c.mats.iter().enumerate() { m[j] = (at.0, at.1, rows); at = (at.0 + (rows * c.k * 2) as u64, at.1 + (c.m * rows * 4) as u64); }
        let pair = |a: (u64, u64, usize), b: (u64, u64, usize)| params::Bf16Rows { weight: a.0, output: a.1, rows: a.2 as u32,
            stride: a.2 as u32, weight2: b.0, output2: b.1, rows2: b.2 as u32, stride2: b.2 as u32, ..v.rows };
        let mut p = params::Bf16Pairs { pairs: [pair(m[0], m[1]), pair(m[2], m[3])] };
        let grid = [env("KBENCH_ROWS_X_BLOCKS").unwrap_or(v.grid[0]), 2, 1];
        return g.launch(f, grid, v.threads, 0, &mut [ptr(&mut p)]);
    }
    if c.kind == 75 && c.mats[3] & 4 != 0 && arm == 1 {
        // As blocks.rs next_rows_group_at: pack_flash_latency_input, then the group launch.
        let (lanes, words) = (params::flash_latency_lanes(c.m), c.m.div_ceil(16) * c.k.div_ceil(16) * 128);
        let mut q = params::FlashBf16Input { input: v.rows.input, output: v.aux, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: lanes as u32 };
        g.launch(v.pack, [(words / 128 * lanes).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        let (r1, r2) = (c.mats[0], c.mats[1]);
        let a = params::Bf16Rows { weight: v.ring[i], output: v.out, rows: r1 as u32, stride: r1 as u32,
            weight2: v.ring[i] + (r1 * c.k * 2) as u64, output2: v.out + (c.m * r1 * 4) as u64, rows2: r2 as u32, stride2: r2 as u32,
            input: v.aux, cols: c.k as u32, tokens: c.m as u32, input_bf16: 1, hc_silu: 0, input_plane_words: words as u64 };
        let none = params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0, ..a };
        let (mut p, mut partials) = (params::Bf16Pairs { pairs: [a, none] }, 0u64);
        return g.launch(f, [192, 1, 1], 128, (4 * params::LATENCY_RING) as u32, &mut [ptr(&mut p), ptr(&mut partials)]);
    }
    v.rows.weight = v.ring[i];
    if c.kind == 75 && c.mats[3] & 2 != 0 {
        (v.rows.weight2, v.rows.hc_silu) = if arm == 1 { (v.ring_x[i], 2) } else { (0, 0) };
    }
    if c.mats[1] != 0 { v.rows.weight2 = v.ring[i] + (c.mats[0] * c.k * 2) as u64; }
    // A native K2560/K3072 twin read (hc_silu bit 1) runs the split: one 128-thread CTA per 8 rows.
    let lbf = arm == 1 && c.kind == 75 && c.mats[3] & 2 != 0 && c.k != 320;
    let (blocks, threads) = if lbf { (c.mats[0].div_ceil(8) as u32, 128) } else { (v.grid[0], v.threads) };
    let grid = [env("KBENCH_ROWS_X_BLOCKS").unwrap_or(blocks), 1, 1];
    g.launch(f, grid, env("KBENCH_ROWS_X_THREADS").unwrap_or(threads), 0, &mut [ptr(&mut v.rows)])
}

#[cfg(target_os = "linux")]
/// One LBF12 twin per weight ring copy (flash_bf16_lbf12_twin from X's module, zeroed flags first).
pub(super) fn twins(g: &mut Gpu, m: H, c: &Case, v: &mut Fixture) -> Result<()> {
    let (rows, k) = (c.mats[0], c.k);
    let slots = rows / 8 * (k / 32);
    let bytes = slots * 392 + rows / 8 * (k / 32).div_ceil(32) * 4;
    let f = g.function(m, "flash_bf16_lbf12_twin")?;
    for i in 0..v.ring.len() {
        let twin = g.fill(bytes, 9)?;
        let mut pack = params::FlashBf16Pack { source: v.ring[i], weight: twin, rows: rows as u32, cols: k as u32 };
        g.launch(f, [(slots * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut pack)])?;
        v.ring_x.push(twin);
    }
    g.sync()
}
