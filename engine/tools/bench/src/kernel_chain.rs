//! One GDN attention sublayer as a decode step runs it, in launch order: the case times the
//! sublayer span, not a kernel. M1 holds all 11 launches. M2-4 skip the GDN step (launches 6
//! and 7): the verify step's own kernels stay plain, and fixed rows feed the output pack. A = main's entries in plain launches (the chunk
//! slot stays programmatic, as on main). X = the entries that `mask` names as `_chain`
//! twins in programmatic launches. A chain copy's kernels run after the previous copy's,
//! so the copies form one long chain, as the layers of a step do. Every output is hashed:
//! each kernel reads the previous kernel's output, so a read before the wait changes bytes.
use super::super::params::{self, Bf16Pairs, Bf16Rows, FlashBf16Input, FlashBf16Pack, FlashHcAdd, FlashHcNorm, GdnPrepare};
use super::{blob, floats, function, gdn, Arm, Fixture, Gpu, Launch, Result, Rng, D, H, HEADS, KEYS, PDL, STATE};
use super::super::ptr;

const WIDE: usize = 10240;
const LOW: usize = 320;
const HIDDEN: usize = 2560;
const OUT: usize = 6144;
const GROUP: [usize; 4] = [WIDE, OUT, 48, 48];
/// Launch order of one sublayer; bit i of the mask selects launch i.
const ENTRIES: [&str; 11] = [
    "flash_hc_norm_f32",
    "flash_bf16_mma_latency_group",
    "flash_next_hc_activate",
    "flash_next_hc_up_latency4_lbf12",
    "flash_bf16_input",
    "flash_mx8_mma_latency_group",
    "flash_next_gdn_prepare_hist_slot",
    "flash_next_gdn_chunk_slot",
    "flash_bf16_input",
    "flash_mx8_mma_latency",
    "flash_next_hc_add",
];

/// An MXFP8 image [n][k]: E4M3 codes at their packed elements, then UE8M0 scales. The small
/// scales keep the sums of the chain finite through the GDN exponentials.
fn mx_image(n: usize, k: usize, salt: usize) -> Vec<u8> {
    let code = |r: usize, col: usize| {
        let r = r + salt * 7919;
        let h = (r.wrapping_mul(2654435761) ^ col.wrapping_mul(40503)).wrapping_mul(2246822519) >> 9;
        let h = h % 254;
        (if h >= 0x7f { h + 1 } else { h }) as u8
    };
    let scale = |r: usize, block: usize| (104 + (r * 7 + block * 3) % 5) as u8;
    let mut img = vec![0u8; n * k];
    for r in 0..n {
        for col in 0..k {
            img[params::flash_bf16_element(r, col, k, n)] = code(r, col);
        }
    }
    let mut sc = vec![127u8; n.div_ceil(8) * (k / 32) * 8];
    for r in 0..n {
        for block in 0..k / 32 {
            sc[(r / 8 * (k / 32) + block) * 8 + r % 8] = scale(r, block);
        }
    }
    img.resize(img.len().div_ceil(256) * 256, 0);
    img.extend_from_slice(&sc);
    img.resize(img.len().div_ceil(256) * 256, 127);
    img
}
/// BF16 weights of one magnitude band, so the LBF12 twin packs every slot.
fn bf16_words(r: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| {
        let v = r.range(0.002, 0.02) * if r.unit() < 0.5 { -1.0 } else { 1.0 };
        ((v.to_bits() >> 16) as u16).to_le_bytes()
    }).collect()
}
fn rows(x: u64, cols: usize, m: usize, a: (u64, u64, usize, usize), b: (u64, u64, usize, usize)) -> Bf16Rows {
    Bf16Rows { weight: a.0, output: a.1, rows: a.2 as u32, stride: a.3 as u32, weight2: b.0, output2: b.1, rows2: b.2 as u32, stride2: b.3 as u32,
        input: x, cols: cols as u32, tokens: m as u32, input_bf16: 1, hc_silu: 0, input_plane_words: (m.div_ceil(16) * cols.div_ceil(16) * 128) as u64 }
}
fn entry(i: usize, m: usize) -> &'static str {
    match (i, m) {
        (3, 2..) => "flash_next_hc_up_latency16_lbf12",
        (5, 2..) => "flash_verify_mx4_part_group",
        (9, 2..) => "flash_verify_mx_solo97",
        _ => ENTRIES[i],
    }
}

pub(super) fn fixture(g: &mut Gpu, modules: [H; 2], mask: usize, m: usize) -> Result<Fixture> {
    const WIDTH: usize = (2 * KEYS + HEADS) * D;
    let copies = 4;
    let mut r = Rng(0x5eed_8a11);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let residual: Vec<f32> = (0..m * WIDE).map(|_| r.range(-2., 2.)).collect();
    let fixed_gated: Vec<f32> = (0..m * OUT).map(|_| r.range(-1., 1.)).collect();
    let norm: Vec<f32> = (0..WIDE).map(|_| r.range(0.5, 1.5)).collect();
    let inject = bf16_words(&mut r, 4 * WIDE);
    let down = bf16_words(&mut r, LOW * WIDE);
    let up = bf16_words(&mut r, WIDE * LOW);
    let group: Vec<Vec<u8>> = GROUP.iter().enumerate().map(|(j, &n)| mx_image(n, HIDDEN, j)).collect();
    let out_image = mx_image(HIDDEN, OUT, 9);
    let conv: Vec<u16> = (0..WIDTH * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let small: Vec<u16> = (0..2 * HEADS).map(|_| bf(r.range(-1., 1.))).collect();
    let hist: Vec<f32> = (0..WIDTH * 4).map(|_| r.range(-2., 2.)).collect();
    let gate_norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    let mut state: Vec<f32> = (0..gdn::chunk_state_floats(HEADS, KEYS)).map(|_| r.range(-0.1, 0.1)).collect();
    // An open chunk of 4 rows: unit key rows, v' in [-0.5, 0.5], descending gamma (as cstep).
    let (tk, tw, tg) = (STATE, STATE + KEYS * super::C * D, STATE + KEYS * super::C * D + HEADS * super::C * D);
    let open = 4;
    for kh in 0..KEYS {
        for j in 0..open {
            let raw: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
            let n = 1. / (raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            for d in 0..D { state[tk + (kh * super::C + j) * D + d] = raw[d] * n; }
        }
    }
    for h in 0..HEADS {
        let mut acc = 0.0f32;
        for j in 0..open {
            for d in 0..D { state[tw + (h * super::C + j) * D + d] = r.range(-0.5, 0.5); }
            acc -= r.range(0.001, 1.0);
            state[tg + h * super::C + j] = acc;
        }
    }
    let twin_entry = function(g, modules[0], "flash_bf16_lbf12_twin", 0)?;
    let (rd, nd) = (floats(g, &residual)?, floats(g, &gate_norm)?);
    let (hd, sd, gd) = (floats(g, &hist)?, floats(g, &state)?, floats(g, &fixed_gated)?);
    let (cd, md) = (g.upload(&halves(&conv))?, g.upload(&halves(&small))?);
    let mut arms = [vec![], vec![]];
    // One copy of the weights per chain copy, shared by both arms: the graph reads them cold.
    let mut weights = vec![];
    for _ in 0..copies {
        let norm_w = floats(g, &norm)?;
        let (iw, dw, uw) = (g.upload(&inject)?, g.upload(&down)?, g.upload(&up)?);
        let gw: Vec<u64> = group.iter().map(|img| g.upload(img)).collect::<Result<_>>()?;
        let ow = g.upload(&out_image)?;
        let slots = WIDE / 8 * (LOW / 32);
        let twin = g.fill(slots * 392 + WIDE / 8 * (LOW / 32).div_ceil(32) * 4, 9)?;
        let mut pack = FlashBf16Pack { source: uw, weight: twin, rows: WIDE as u32, cols: LOW as u32 };
        g.launch(twin_entry, [(slots * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut pack)])?;
        weights.push((norm_w, iw, dw, uw, gw, ow, twin));
    }
    g.sync()?;
    for arm in 0..2 {
        let bits = if arm == 1 { mask } else { 0 };
        let f: Vec<(H, u32)> = (0..ENTRIES.len()).map(|i| {
            let chain = bits >> i & 1 != 0;
            let name = if chain { format!("{}_chain", entry(i, m)) } else { entry(i, m).to_string() };
            let shared = match i {
                1 => 4 * params::LATENCY_RING,
                5 if m > 1 => params::VERIFY_MX4_PART_SHARED,
                5 => params::latency_mx8_shared_rows(1, HIDDEN, 4, GROUP.iter().sum(), true),
                9 if m > 1 => params::verify_solo97_shared(OUT),
                9 => params::latency_mx8_shared_rows(1, OUT, 4, HIDDEN, false),
                _ => 0,
            } as u32;
            Ok((function(g, modules[arm], &name, shared)?, shared | if chain || i == 7 { PDL } else { 0 }))
        }).collect::<Result<_>>()?;
        for (norm_w, iw, dw, uw, gw, ow, twin) in weights.clone() {
            let live_r = g.alloc(m * WIDE * 4)?;
            let (u, injection, low, up_out) = (g.alloc(m * WIDE * 4)?, g.alloc(m * 4 * 4)?, g.alloc(m * LOW * 4)?, g.alloc(m * HIDDEN * 4)?);
            let planes = |g: &mut Gpu, cols: usize| g.alloc(m.div_ceil(16) * cols.div_ceil(16) * 128 * 2 * 4);
            let (p_wide, p_low, p_hid, p_out) = (planes(g, WIDE)?, planes(g, LOW)?, planes(g, HIDDEN)?, planes(g, OUT)?);
            let out: Vec<u64> = GROUP.iter().map(|&n| g.alloc(m * n * 4)).collect::<Result<_>>()?;
            let feat = HEADS * D * 4;
            let (q, k, v, core, gated) = (g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?);
            let (beta, gg) = (g.alloc(HEADS * 4)?, g.alloc(HEADS * 4)?);
            let (block, live_h, live_s) = (g.alloc(m * HIDDEN * 4)?, g.alloc(WIDTH * 16)?, g.alloc(state.len() * 4)?);
            let mut t = vec![0u8; 2 * 1168];
            t[1168..1176].copy_from_slice(&live_s.to_le_bytes());
            t[1176..1184].copy_from_slice(&live_h.to_le_bytes());
            t[1168 + 1160..1168 + 1164].copy_from_slice(&((10 * super::C + open) as u32).to_le_bytes());
            let table = g.upload(&t)?;
            let slots = g.words(&[1])?;
            let step: [u64; 3] = [slots, table, 1u64 << 32];
            let none = (0, 0, 0, 0);
            let hc = Bf16Pairs { pairs: [rows(p_wide, WIDE, m, (iw, injection, 4, 4), (dw, low, LOW, LOW)), rows(p_wide, WIDE, m, none, none)] };
            let up_p = rows(p_low, LOW, m, (uw, up_out, WIDE, HIDDEN), none);
            let flagged = |j: usize, n: usize| (gw[j] | 1, out[j], n, n);
            let attn = Bf16Pairs { pairs: [rows(p_hid, HIDDEN, m, flagged(0, WIDE), flagged(1, OUT)), rows(p_hid, HIDDEN, m, flagged(2, 48), flagged(3, 48))] };
            let out_p = rows(p_out, OUT, m, (ow | 1, block, HIDDEN, HIDDEN), none);
            let prep = GdnPrepare {
                x: out[0] as _, w: cd as _, history: std::ptr::null(), a: out[2] as _, b: out[3] as _,
                alog: md as _, dt: (md + (HEADS * 2) as u64) as _, q: q as _, k: k as _, v: v as _, beta: beta as _, g: gg as _,
                key_heads: KEYS as u32, value_heads: HEADS as u32, key_start: 0, value_start: 0, tokens: 1, precision: 0, scale: 0,
            };
            let chunk = gdn::GdnChunkNext {
                q, k, v, beta, g: gg, rows: 0, out: core, a: 0, p: 0, gamma: 0, k0: 0,
                heads: HEADS as u32, keys: KEYS as u32, count: 1, base: 0,
                z: out[1], norm: nd, gated, epsilon: 1e-6, spare: 0, x: 0, w: 0, history: 0, channels: 0, x_start: 0, ga: 0, gb: 0, alog: 0, dt: 0, pk: 0, pk_tokens: 0, pk_row0: 0,
            };
            let lanes = params::flash_latency_lanes(m);
            let pack = |input: u64, output: u64, cols: usize| FlashBf16Input { input, output, tokens: m as u32, cols: cols as u32, hc_silu: 0, lanes: lanes as u32 };
            let pack_grid = |cols: usize| ((m.div_ceil(16) * (cols / 16) * 128 / 128 * lanes).div_ceil(128)) as u32;
            let add = FlashHcAdd { residual: live_r, block, peer: 0, gates: injection, hidden: HIDDEN as u32, rank: 0 };
            let norm_p = FlashHcNorm { input: live_r, weight: norm_w, output: u, packed: p_wide };
            let (y, active) = (m as u32, (m * LOW) as u32);
            let mut launches = vec![
                Launch { f: f[0].0, grid: [8, y, 1], block: 128, shared: f[0].1, params: vec![blob(&norm_p)] },
                Launch { f: f[1].0, grid: [192, 1, 1], block: 128, shared: f[1].1, params: vec![blob(&hc), blob(&0u64)] },
                Launch { f: f[2].0, grid: [active.div_ceil(128), 1, 1], block: 128, shared: f[2].1, params: vec![blob(&low), blob(&active), blob(&p_low)] },
                Launch { f: f[3].0, grid: [(HIDDEN / 8) as u32, 1, 1], block: 128, shared: f[3].1, params: vec![blob(&up_p), blob(&u), blob(&twin), blob(&0u64)] },
                Launch { f: f[4].0, grid: [pack_grid(HIDDEN), 1, 1], block: 128, shared: f[4].1, params: vec![blob(&pack(up_out, p_hid, HIDDEN))] },
                Launch { f: f[5].0, grid: [192, 1, 1], block: 128, shared: f[5].1, params: vec![blob(&attn), blob(&0u64)] },
            ];
            if m == 1 {
                launches.push(Launch { f: f[6].0, grid: [20, 1, 1], block: 128, shared: f[6].1, params: vec![blob(&prep), blob(&step)] });
                launches.push(Launch { f: f[7].0, grid: [HEADS as u32, 1, 1], block: 128, shared: f[7].1, params: vec![blob(&chunk), blob(&step)] });
            }
            launches.extend([
                Launch { f: f[8].0, grid: [pack_grid(OUT), 1, 1], block: 128, shared: f[8].1, params: vec![blob(&pack(if m == 1 { gated } else { gd }, p_out, OUT))] },
                Launch { f: f[9].0, grid: [192, 1, 1], block: 128, shared: f[9].1, params: vec![blob(&out_p), blob(&0u64), blob(&0u64)] },
                Launch { f: f[10].0, grid: [20, y, 1], block: 128, shared: f[10].1, params: vec![blob(&add)] },
            ]);
            let mut hashed = vec![(u, m * WIDE * 4), (injection, m * 16), (low, m * LOW * 4), (up_out, m * HIDDEN * 4)];
            hashed.extend(GROUP.iter().zip(&out).map(|(&n, &p)| (p, m * n * 4)));
            hashed.extend([(block, m * HIDDEN * 4), (live_r, m * WIDE * 4)]);
            let mut resets = vec![(live_r, rd, m * WIDE * 4)];
            if m == 1 {
                hashed.extend([(q, feat), (k, feat), (v, feat), (beta, HEADS * 4), (gg, HEADS * 4), (core, feat), (gated, feat), (live_h, WIDTH * 16), (live_s, state.len() * 4)]);
                resets.extend([(live_h, hd, WIDTH * 16), (live_s, sd, state.len() * 4)]);
            }
            arms[arm].push(Arm { resets, launches, hashed, layout: 0 });
        }
    }
    let group_bytes: usize = group.iter().map(Vec::len).sum();
    let stored = 324 * WIDE * 2 + WIDE * LOW * 3 / 2 + group_bytes + out_image.len();
    Ok(Fixture { arms, stored, flops: 0, copies })
}
