//! Vocabulary fixtures and production M16/experimental M32 slice launches; the HC up
//! LBF12 twin fixture.
use super::{Case, Fixture, Gpu, H, Result, check, params, ptr};
pub(super) fn prepare(g: &mut Gpu, module: H, c: &Case, v: &mut Fixture) -> Result<()> {
if c.kind == 77 && c.name.starts_with("hc_act_pack") {
    let m = g.module(super::prompt::copy_ptx().as_bytes()).map_err(|e| format!("copy PTX: {e}"))?;
    v.norms[1] = g.function(m, "kb_copy")?;
    return Ok(());
}
if c.kind == 2 && c.real {
    // BF16 middles (exponent span 9, so every slot fits LBF12) in each copy, the same bytes
    // in every copy; -lbf12raw- puts a 2^16 element in one slot in seven (raw: BF16 loads).
    let (n, k) = (c.n, c.k);
    let mut w = super::noise(0x4c42_4631_32, n * k * 2, 3);
    let raw = c.name.contains("lbf12raw");
    let (pairs, slots) = (k / 32, n / 8 * (k / 32));
    let mut marked = 0;
    for slot in 0..slots {
        if raw && slot % 7 == 3 {
            let at = params::flash_bf16_element(slot / pairs * 8, slot % pairs * 32, k, n) * 2;
            w[at..at + 2].copy_from_slice(&0x4780u16.to_le_bytes());
            marked += 1;
        }
    }
    g.upload_at(v.aux, &super::noise(0x4843_5550, c.m * n * 4, 2))?;
    let builder = g.function(module, "flash_bf16_lbf12_twin")?;
    for &weight in &v.ring {
        g.upload_at(weight, &w)?;
        let twin = g.upload(&vec![0u8; slots * 392 + n / 8 * 4])?;
        let mut p = params::FlashBf16Pack { source: weight, weight: twin, rows: n as u32, cols: k as u32 };
        g.launch(builder, [(slots * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut p)])?;
        v.ring_x.push(twin);
    }
    g.sync()?;
    for &twin in &v.ring_x {
        let flags = g.read(twin + (slots * 392) as u64, n / 8 * 4)?;
        let packed: usize = flags.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()).count_ones() as usize).sum();
        if packed != slots - marked { return Err(format!("{} packed {packed} slots, expected {}", c.name, slots - marked)); }
    }
    eprintln!("hc_up LBF12 coverage case={} packed_slots={} raw_slots={marked}", c.name, slots - marked);
    return Ok(());
}
if c.kind == 1 && c.real {
    // Encode the same lossless fixture in place for both arms, outside timing.
    let encoder = g.function(module, "flash_next_head_lbf12_encode")?;
    for &weight in &v.ring {
        let mut flags = g.alloc(c.n / 8 * 16)?;
        unsafe { check((g.d.memset)(flags, 0, c.n / 8 * 16))?; }
        let mut bases = g.alloc(c.n * c.k / 32)?;
        let mut p = params::FlashBf16Pack { source: 0, weight, rows: c.n as u32, cols: c.k as u32 };
        g.launch(encoder, [(c.n * c.k / 8).div_ceil(128) as u32, 1, 1], 128, 0,
            &mut [ptr(&mut p), ptr(&mut flags), ptr(&mut bases)])?;
        v.ring_x.push(flags);
        v.scales_x.push(bases);
    }
    g.sync()?;
    // One warp encodes 8 rows x K32, hence N*K/8 threads. The fixture's
    // exponent span fits every slot, so verify all expected bits were produced.
    for &flags in &v.ring_x {
        let raw = g.read(flags, c.n / 8 * 16)?;
        let encoded: usize = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()).count_ones() as usize).sum();
        if encoded != c.n * c.k / 256 { return Err(format!("{} encoded {encoded} slots, expected {}", c.name, c.n * c.k / 256)); }
        eprintln!("head LBF12 coverage case={} encoded_slots={encoded} expected_slots={}", c.name, c.n * c.k / 256);
    }
}
    Ok(())
}
pub(super) fn launch(g: &Gpu, f: H, c: &Case, v: &mut Fixture, i: usize, arm: usize) -> Result<()> {
    // Production main uses M16 slices. HEAD32 shares each fragment
    // across all of a wider slice, preserving each row's K graph.
    let wide = arm == 1 && std::env::var("KBENCH_XKERNEL").is_ok_and(|n| n.starts_with("flash_next_head32"));
    for start in (0..c.m).step_by(if wide { 32 } else { 16 }) {
        let mut p = v.rows;
        p.weight = v.ring[i];
        p.tokens = (c.m - start).min(if wide { 32 } else { 16 }) as u32;
        p.input += (start / 8 * (c.k / 16) * 32 * 16) as u64;
        p.output += (start * c.n * 4) as u64;
        let mut partial = v.aux + (start * 192 * 3 * 4) as u64;
        let mut args = vec![ptr(&mut p), ptr(&mut partial)];
        let (mut flags, mut bases) = if c.real { (v.ring_x[i], v.scales_x[i]) } else { (0, 0) };
        if c.real { args.extend([ptr(&mut flags), ptr(&mut bases)]); }
        g.launch(f, v.grid, v.threads, v.shared, &mut args)?;
    }
    Ok(())
}
/// kind 77: the decode HC norm (flash_hc_norm_f32, packed 0) of M rows [M][4H] FP32, F32
/// weights [4H] rotating across 64 MiB (cold, as one layer's norm per token); input warm.
/// hc_norm_pack-mM: then HC down's operand too: A = the norm + main's latency pack of u (the
/// decode chain), X = the norm writing the packed planes; A==X on u and on the staged words
/// (staged(): lanes 0..L-1, L = flash_latency_lanes(M), the words HC down stages).
pub(super) fn norm_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    if c.name.starts_with("hc_act_pack") { return act_fixture(g, c); }
    let (m, k) = (c.m, c.k);
    let input: Vec<u8> = (0..m * k).flat_map(|i| ((((i / k) * 37 + (i % k) * 11) % 263) as f32 / 32. - 4.).to_le_bytes()).collect();
    let norm: Vec<u8> = (0..k).flat_map(|i| (((i * 7) % 61) as f32 / 256. - 0.1).to_le_bytes()).collect();
    let copies = (64usize << 20).div_ceil(k * 4);
    let slab = g.upload(&norm.repeat(copies))?;
    let plane = m.div_ceil(16) * (k / 16) * 128;
    let (pack, lanes) = (c.name.starts_with("hc_norm_pack"), params::flash_latency_lanes(m));
    let out_bytes = if pack { m * k * 4 + 2 * plane * 4 } else { m * k * 4 };
    let written = if pack { staged(m, k, lanes) } else { Vec::new() };
    let (raw, out) = (g.upload(&input)?, g.alloc(out_bytes)?);
    let rows = params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0,
        input: 0, cols: 0, tokens: 0, input_bf16: 0, hc_silu: 0, input_plane_words: 0 };
    Ok(Fixture { rows, group: None, ring: (0..copies).map(|i| slab + (i * k * 4) as u64).collect(), out, out_bytes,
        aux: 0, aux_bytes: 0, grid: [8, m as u32, 1], threads: 128, shared: 0, stored: 2 * m * k * 4 + k * 4 + if pack { m * k * 4 } else { 0 }, flops: 0,
        chunks: 1, down_slots: 0, raw, pack: std::ptr::null_mut(), norms: [std::ptr::null_mut(); 2], weight_bytes: k * 4,
        written, qsa: None, gather: None, ring_x: vec![], scales_x: vec![] })
}
/// hc_act_pack-mM (kind 77): the decode HC activation of low [M][320] FP32 and HC up's
/// operand. Both arms first restore low from the raw rows (kb_copy); A = main's
/// flash_next_hc_activate + main's dense pack (32 lanes, K 320), X = the activation writing
/// the packed planes; A==X on low and on the staged words (staged(), L = 32: HC up's pack).
fn act_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, k) = (c.m, c.k);
    let input: Vec<u8> = (0..m * k).flat_map(|i| ((((i / k) * 37 + (i % k) * 11) % 263) as f32 / 16. - 8.).to_le_bytes()).collect();
    let plane = m.div_ceil(16) * (k / 16) * 128;
    let out_bytes = m * k * 4 + 2 * plane * 4;
    let written = staged(m, k, 32);
    let (raw, out) = (g.upload(&input)?, g.alloc(out_bytes)?);
    let rows = params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0,
        input: 0, cols: 0, tokens: 0, input_bf16: 0, hc_silu: 0, input_plane_words: 0 };
    Ok(Fixture { rows, group: None, ring: vec![0; 256], out, out_bytes, aux: 0, aux_bytes: 0, grid: [(m * k).div_ceil(128) as u32, 1, 1],
        threads: 128, shared: 0, stored: 3 * m * k * 4, flops: 0, chunks: 1, down_slots: 0, raw, pack: std::ptr::null_mut(),
        norms: [std::ptr::null_mut(); 2], weight_bytes: 0, written, qsa: None, gather: None, ring_x: vec![], scales_x: vec![] })
}
/// Byte ranges A==X compares for M rows [M][K] FP32 followed by the two packed planes: every
/// FP32 row, then in lanes 0..lanes-1 of each K16 tile the words of rows < M and of padding
/// row t + 8 of a row t < 8 (its 0 word); other padding rows stay unwritten by the fused
/// writers (their outputs are masked), so they are not compared.
fn staged(m: usize, k: usize, lanes: usize) -> Vec<(usize, usize)> {
    let tiles = 2 * m.div_ceil(16) * (k / 16);
    let mut out = vec![(0, m * k * 4)];
    for tile in 0..tiles {
        for lane in 0..lanes {
            for j in 0..4 {
                // Word j of a fragment: row (tile row) lane / 4 + 8 * (j % 2).
                let row = tile % (tiles / 2) / (k / 16) * 16 + lane / 4 + 8 * (j % 2);
                if row < m || (j % 2 == 1 && row - 8 < m) {
                    out.push((m * k * 4 + ((tile * 32 + lane) * 4 + j) * 4, 4));
                }
            }
        }
    }
    out
}
/// HC up at M1-16 from the LBF12 twin: grid H/8, 128 threads, (rows, FP32 streams, twin);
/// kind 77: hc_norm-mM = the HC norm (input, weight copy, output, packed 0); hc_norm_pack-mM =
/// that norm + main's latency pack (A) vs the norm writing the planes (X); hc_act_pack-mM =
/// kb_copy of low + main's activation + main's dense pack (A) vs kb_copy + the activation
/// writing the planes (X).
pub(super) fn launch_hc(g: &Gpu, f: H, c: &Case, v: &mut Fixture, i: usize, arm: usize) -> Result<()> {
    if c.kind == 77 && c.name.starts_with("hc_act_pack") {
        let (planes, words) = (v.out + (c.m * c.k * 4) as u64, c.m.div_ceil(16) * (c.k / 16) * 128);
        let (mut dst, mut src, mut n) = (v.out, v.raw, (c.m * c.k / 4) as u32);
        g.launch(v.norms[1], [n.div_ceil(128), 1, 1], 128, 0, &mut [ptr(&mut dst), ptr(&mut src), ptr(&mut n)])?;
        let (mut x, mut e, mut p) = (v.out, (c.m * c.k) as u32, if arm == 1 { planes } else { 0 });
        g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut x), ptr(&mut e), ptr(&mut p)])?;
        if arm == 0 {
            // dense.rs pack_flash_dense_input: 32 lanes, one thread per fragment.
            let mut q = params::FlashBf16Input { input: v.out, output: planes, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: 32 };
            g.launch(v.pack, [(words / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        }
        return Ok(());
    }
    if c.kind == 77 {
        let (pack, planes) = (c.name.starts_with("hc_norm_pack"), v.out + (c.m * c.k * 4) as u64);
        let mut p = params::FlashHcNorm { input: v.raw, weight: v.ring[i], output: v.out, packed: if pack && arm == 1 { planes } else { 0 } };
        g.launch(f, v.grid, v.threads, v.shared, &mut [ptr(&mut p)])?;
        if pack && arm == 0 {
            // dense.rs pack_flash_latency_input: one thread per written fragment.
            let lanes = params::flash_latency_lanes(c.m);
            let words = c.m.div_ceil(16) * (c.k / 16) * 128;
            let mut q = params::FlashBf16Input { input: v.out, output: planes, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: lanes as u32 };
            g.launch(v.pack, [(words / 128 * lanes).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        }
        return Ok(());
    }
    v.rows.weight = v.ring[i];
    let (mut normalized, mut twin) = (v.aux, v.ring_x[i]);
    g.launch(f, v.grid, v.threads, v.shared, &mut [ptr(&mut v.rows), ptr(&mut normalized), ptr(&mut twin)])
}
