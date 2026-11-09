//! QSA prompt-attention fixtures at production shape: 24 query heads and 2 KV
//! heads per Spark (q_proj 12288 = 24 x 512, k_proj 512 = 2 x 256), head dimension 256,
//! 2048-row prompt chunks, 512 selected four-token groups plus the open tail,
//! score stride min(history, 2051). Synthetic finite FP32 data with full
//! mantissas (the hi/lo split must see every bit); IDs are 512 distinct
//! ascending groups per row, uniform over the causal groups (worst locality).
use super::{Gpu, Result, check, params, ptr};

pub(super) const HEADS: usize = 24;
pub(super) const KHEADS: usize = 2;
pub(super) const DIM: usize = 256;
pub(super) const MAXPOS: usize = 4 * 512 + 3;
const GROUPS: usize = 2; // (query_start % 12 + 24).div_ceil(12)

pub(super) struct Host {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub ids: Vec<u32>,
    pub tokens: usize,
    pub past: usize,
    pub blocks: usize,
}
/// Host copy of a values fixture (FP64 reference and alone == chunked).
pub(super) struct VHost {
    pub v: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub denominators: Vec<f32>,
    pub gates: Vec<f32>,
    pub ids: Vec<u32>,
    pub tokens: usize,
    pub past: usize,
    pub blocks: usize,
}
/// Host copy of an index-score fixture (FP64 reference, top-512 agreement).
pub(super) struct IHost {
    pub q: Vec<f32>,
    pub raw: Vec<f32>,
    pub tokens: usize,
    pub past: usize,
    pub blocks: usize,
}
pub(super) struct Fix {
    /// Kind 12 arm A: the X-on chain (entry, grid, threads), launched in order.
    pub chain: Vec<(super::H, [u32; 3], u32)>,
    pub ihost: Option<IHost>,
    pub vhost: Option<VHost>,
    pub attn: params::QsaAttention,
    /// Arm X override (raw-score values: raw scores + (denominator, highest) pairs).
    pub attn_x: Option<params::QsaAttention>,
    pub index: params::QsaIndex,
    pub cursor: u64,
    /// Per arm (A, X): grid, threads, dynamic shared bytes.
    pub launch: [([u32; 3], u32, u32); 2],
    pub host: Option<Host>,
    pub out: u64,
    pub out_bytes: usize,
    pub stored: usize,
    pub kind: u8,
    /// Kind 15 (pooled decode stages): batch parameters, stage, session pasts.
    pub batch: Option<Batch>,
    /// Split values (KBENCH_QSA_VALUES_SPLIT): the k-step tile buffer of arm X's two kernels.
    pub extra: u64,
}
pub(super) struct Batch {
    pub p: params::QsaBatch,
    pub stage: usize,
    pub pasts: Vec<usize>,
    pub blocks: usize,
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Roughly N(0,1) with full 24-bit mantissas.
    fn normal(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732_050_8
    }
}
/// Q/K rows like RMSNorm x (1 + w) then RoPE: unit scale with per-feature spread.
fn rows(seed: u64, count: usize) -> Vec<f32> {
    let mut r = Rng(seed);
    let scale: Vec<f32> = (0..DIM).map(|_| 0.35 + 1.6 * r.unit()).collect();
    (0..count * DIM).map(|i| r.normal() * scale[i % DIM]).collect()
}
pub(super) fn selected_length(length: usize) -> usize {
    if length / 4 <= 512 { length } else { 4 * 512 + length % 4 }
}
pub(super) fn selected_position(ids: &[u32], complete: usize, token: usize, blocks: usize, slot: usize) -> usize {
    if complete <= 512 {
        slot
    } else if slot < 4 * 512 {
        4 * ids[token * blocks + slot / 4] as usize + slot % 4
    } else {
        4 * complete + slot - 4 * 512
    }
}
fn ids(seed: u64, tokens: usize, past: usize, blocks: usize) -> Vec<u32> {
    // KBENCH_QSA_IDS=file: a real selection dumped from a prefill (header u32
    // tokens, past, blocks; then 512 ascending groups per token); used when the
    // case's tokens and past match the file, uniform random IDs otherwise.
    if let Some(bytes) = std::env::var("KBENCH_QSA_IDS").ok().and_then(|f| std::fs::read(f).ok()) {
        let w: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if w.len() >= 3 && w[0] as usize == tokens && w[1] as usize == past && w.len() == 3 + tokens * 512 {
            let stride = blocks.max(512);
            let mut out = vec![0u32; tokens * stride];
            for t in 0..tokens { out[t * stride..t * stride + 512].copy_from_slice(&w[3 + t * 512..3 + (t + 1) * 512]); }
            eprintln!("ids from KBENCH_QSA_IDS (tokens {tokens}, past {past})");
            return out;
        }
    }
    let mut r = Rng(seed);
    let mut out = vec![0u32; tokens * blocks.max(512)];
    let stride = blocks.max(512);
    let mut seen = vec![0u64; blocks.div_ceil(64).max(1)];
    for t in 0..tokens {
        let complete = (past + t + 1) / 4;
        if complete <= 512 { continue; }
        seen.iter_mut().for_each(|w| *w = 0);
        let mut picked = 0;
        while picked < 512 {
            let g = (r.next() % complete as u64) as usize;
            if seen[g / 64] >> (g % 64) & 1 == 0 {
                seen[g / 64] |= 1 << (g % 64);
                picked += 1;
            }
        }
        let row = &mut out[t * stride..t * stride + 512];
        let mut n = 0;
        for g in 0..complete {
            if seen[g / 64] >> (g % 64) & 1 != 0 { row[n] = g as u32; n += 1; }
        }
    }
    out
}
fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// Score keys per warp of the X arm's FP32 scores entry (QSA_SCORE_KEYS of the
/// X build; KBENCH_QSA_KEYS_X, default 4 = main). The A arm is main (4).
pub(super) fn keys_x() -> usize {
    std::env::var("KBENCH_QSA_KEYS_X").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}
/// Feature slices per (token, KV head) of the X arm's values entry (main = 4;
/// one 256-feature block per (token, KV head) = 1). KBENCH_QSA_VALUES_SLICES_X.
/// Order-mode prompt values grid multiplier: CTAs per (token, KV head) of the
/// X entry (1 for _tc; 2 for a half-feature benchmark entry). KBENCH_QSA_VALUES_HALVES_X.
pub(super) fn value_halves_x() -> usize {
    std::env::var("KBENCH_QSA_VALUES_HALVES_X").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}
/// Threads of the X arm's sliced values entry (main 768 = 12 heads x 64
/// features; 192 for 16-feature slices with KBENCH_QSA_VALUES_SLICES_X=16).
pub(super) fn value_threads_x() -> u32 {
    std::env::var("KBENCH_QSA_VALUES_THREADS_X").ok().and_then(|v| v.parse().ok()).unwrap_or(768)
}
pub(super) fn value_slices_x() -> usize {
    std::env::var("KBENCH_QSA_VALUES_SLICES_X").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}
/// Order-mode X entry for values launches under 64 tokens (the host's small
/// grid: KV heads x 4 feature slices); KBENCH_XKERNEL_SMALL.
pub(super) fn small_x() -> Option<String> { std::env::var("KBENCH_XKERNEL_SMALL").ok() }
/// Grid-y slot ranges for the tensor-core scores entry; the range size only
/// distributes work (each score's arithmetic is fixed by head dimension 256 alone).
pub(super) fn tc_planes(tokens: usize, capacity: usize) -> u32 {
    if tokens >= 64 { 8 } else { capacity.min(MAXPOS).div_ceil(16) as u32 }
}

/// Distinct cache positions any token of the launch reads (the as-stored K/V bytes).
fn touched(ids: &[u32], tokens: usize, past: usize, blocks: usize) -> usize {
    let mut seen = vec![false; past + tokens];
    for t in 0..tokens {
        let length = past + t + 1;
        for slot in 0..selected_length(length) { seen[selected_position(ids, length / 4, t, blocks, slot)] = true; }
    }
    seen.iter().filter(|&&b| b).count()
}
/// kind 7 attention scores, 8 attention values, 9 index scores.
pub(super) fn fixture(g: &mut Gpu, kind: u8, tokens: usize, past: usize, order: bool) -> Result<Fix> {
    let capacity = past + tokens;
    let blocks = capacity / 4;
    // Production stride = history.min(2051) with history >= 32832 (context).
    let stride = MAXPOS;
    let seed = (kind as u64) << 48 ^ (tokens as u64) << 24 ^ past as u64;
    let rows_q = tokens * HEADS;
    let mut index = params::QsaIndex { query: 0, pooled: 0, scores: 0, tokens: tokens as u32, past: past as u32, blocks: blocks as u32, padding: 0 };
    let mut attn = params::QsaAttention {
        query: 0, query_gate: 0, keys: 0, values: 0, ids: 0, output: 0, scores: 0,
        query_heads: HEADS as u32, key_heads: KHEADS as u32, query_start: 0, key_start: 0,
        tokens: tokens as u32, past: past as u32, blocks: blocks as u32, queries_per_key: 12,
        stride: stride as u32, padding: 0, denominators: 0,
    };
    match kind {
        7 => {
            let q = rows(seed ^ 1, rows_q);
            let k = rows(seed ^ 2, capacity * KHEADS);
            let id = ids(seed ^ 3, tokens, past, blocks);
            attn.query = g.upload(&f32_bytes(&q))?;
            attn.keys = g.upload(&f32_bytes(&k))?;
            attn.values = attn.keys;
            attn.ids = g.upload(&u32_bytes(&id))?;
            attn.blocks = blocks.max(512) as u32;
            let out_bytes = rows_q * stride * 4;
            attn.scores = g.alloc(out_bytes)?;
            let visible: usize = (0..tokens).map(|t| selected_length(past + t + 1)).sum();
            // As stored (DRAM compulsory): Q once, every K row any token selects
            // once (both KV heads), the visible scores once.
            let stored = rows_q * DIM * 4 + touched(&id, tokens, past, blocks.max(512)) * KHEADS * DIM * 4 + visible * HEADS * 4;
            let a = ([(GROUPS * tokens) as u32, capacity.min(MAXPOS).div_ceil(4).div_ceil(4) as u32, 1], 128, 0);
            let x = if order { ([(GROUPS * tokens) as u32, tc_planes(tokens, capacity), 1], 128, 0) }
                else if std::env::var("KBENCH_SCORES_T_X").is_ok() { ([(GROUPS * tokens) as u32, 8, 1], 128, 0) } // flash_qsa_attention_scores_t
                else if std::env::var("KBENCH_SCORES_T6_X").is_ok() { ([(GROUPS * tokens) as u32, 8, 1], 64, 0) } // _t6
                else { ([(GROUPS * tokens) as u32, capacity.min(MAXPOS).div_ceil(keys_x()).div_ceil(4) as u32, 1], 128, 0) };
            Ok(Fix { chain: Vec::new(), batch: None, extra: 0, ihost: None, vhost: None, attn, attn_x: None, index, cursor: 0, launch: [a, x], out: attn.scores, out_bytes, stored, kind,
                host: Some(Host { q, k, ids: id, tokens, past, blocks: blocks.max(512) }) })
        }
        8 => {
            let mut r = Rng(seed ^ 4);
            let v = rows(seed ^ 5, capacity * KHEADS);
            let mut probabilities = vec![0f32; rows_q * stride];
            let mut denominators = vec![0f32; rows_q];
            // KBENCH_QSA_VALUES_RAW: raw scores; P = ex2.approx.ftz((s - highest) * log2 e)
            // on the GPU (the instructions of the softmax), so a raw-score X arm
            // (KBENCH_QSA_VALUES_XRAW) must give the bytes of a P arm.
            let raw_mode = std::env::var("KBENCH_QSA_VALUES_RAW").is_ok();
            let mut raw_scores = vec![0f32; if raw_mode { rows_q * stride } else { 0 }];
            let mut highest = vec![0f32; if raw_mode { rows_q } else { 0 }];
            if raw_mode {
                let mut x = vec![f32::NEG_INFINITY; rows_q * stride];
                for row in 0..rows_q {
                    let visible = selected_length(past + row / HEADS + 1);
                    let mut h = f32::NEG_INFINITY;
                    for s in 0..visible { let v = r.normal() * 2.0; raw_scores[row * stride + s] = v; h = h.max(v); }
                    highest[row] = h;
                    for s in 0..visible { x[row * stride + s] = (raw_scores[row * stride + s] - h) * std::f32::consts::LOG2_E; }
                }
                let mut xd = g.upload(&f32_bytes(&x))?;
                let mut pd = g.alloc(rows_q * stride * 4)?;
                let mut n = (rows_q * stride) as u32;
                let m = g.module(ex2_ptx().as_bytes())?;
                let f = g.function(m, "kb_ex2")?;
                g.launch(f, [n.div_ceil(256), 1, 1], 256, 0, &mut [ptr(&mut xd), ptr(&mut pd), ptr(&mut n)])?;
                g.sync()?;
                let got = g.read(pd, rows_q * stride * 4)?;
                for (i, c) in got.chunks_exact(4).enumerate() { probabilities[i] = f32::from_le_bytes(c.try_into().unwrap()); }
                for row in 0..rows_q {
                    let visible = selected_length(past + row / HEADS + 1);
                    let mut d = 0.0f32;
                    for s in 0..visible { d += probabilities[row * stride + s]; }
                    denominators[row] = d;
                }
            } else {
                for row in 0..rows_q {
                    let visible = selected_length(past + row / HEADS + 1);
                    let mut d = 0.0f32;
                    for s in 0..visible {
                        let p = (r.normal() * 1.5 - 1.5).exp().min(1.0);
                        probabilities[row * stride + s] = p;
                        d += p;
                    }
                    denominators[row] = d;
                }
            }
            let gates: Vec<f32> = (0..rows_q * 512).map(|_| r.normal()).collect();
            attn.values = g.upload(&f32_bytes(&v))?;
            attn.keys = attn.values;
            let id = ids(seed ^ 3, tokens, past, blocks);
            attn.ids = g.upload(&u32_bytes(&id))?;
            attn.blocks = blocks.max(512) as u32;
            attn.scores = g.upload(&f32_bytes(&probabilities))?;
            attn.denominators = g.upload(&f32_bytes(&denominators))?;
            attn.query_gate = g.upload(&f32_bytes(&gates))?;
            let out_bytes = rows_q * DIM * 4;
            attn.output = g.alloc(out_bytes)?;
            let attn_x = if raw_mode && std::env::var("KBENCH_QSA_VALUES_XRAW").is_ok() {
                let pairs: Vec<f32> = (0..rows_q).flat_map(|r| [denominators[r], highest[r]]).collect();
                let mut x = attn;
                x.scores = g.upload(&f32_bytes(&raw_scores))?;
                x.denominators = g.upload(&f32_bytes(&pairs))?;
                Some(x)
            } else { None };
            let visible: usize = (0..tokens).map(|t| selected_length(past + t + 1)).sum();
            // As stored: every selected V row once, probabilities, gates, denominators, output.
            let stored = touched(&id, tokens, past, blocks.max(512)) * KHEADS * DIM * 4 + visible * HEADS * 4 + rows_q * (DIM * 4 + 256 * 4 + 4);
            let a = ([(KHEADS * tokens * 4) as u32, 1, 1], 768, 0);
            let x = if order && tokens < 64 && small_x().is_some() { ([(KHEADS * tokens * 4) as u32, 1, 1], 128, 0) }
                else if order { ([(KHEADS * tokens * value_halves_x()) as u32, 1, 1], 128, 0) }
                else if std::env::var("KBENCH_VALUES_W2_X").is_ok() { ([(KHEADS * tokens) as u32, 1, 1], 384, 0) } // values_wide2
                else if std::env::var("KBENCH_VALUES_W4_X").is_ok() { ([(KHEADS * tokens) as u32, 1, 1], 192, 0) } // values_wide4
                else if std::env::var("KBENCH_VALUES_U_X").is_ok() { ([(tokens.div_ceil(8) * KHEADS) as u32, 1, 1], 768, 0) } // values_u
                else { ([(KHEADS * tokens * value_slices_x()) as u32, 1, 1], value_threads_x(), 0) };
            let vhost = Some(VHost { v, probabilities, denominators, gates, ids: id, tokens, past, blocks: blocks.max(512) });
            // KBENCH_QSA_VALUES_SPLIT: X = flash_qsa_attention_values_tc_k then _tc_kr over a [token][KV head][slice][k-step][16][64] tile buffer.
            let extra = if std::env::var("KBENCH_QSA_VALUES_SPLIT").is_ok() { g.alloc(tokens * KHEADS * 4 * MAXPOS.div_ceil(16) * 16 * 64 * 4)? } else { 0 };
            Ok(Fix { chain: Vec::new(), batch: None, extra, ihost: None, vhost, attn, attn_x, index, cursor: 0, launch: [a, x], out: attn.output, out_bytes, stored, kind, host: None })
        }
        9 => {
            let iq = rows(seed ^ 6, tokens * 4 / 2); // [tokens, 4, 128] = tokens * 2 * 256
            let raw_host = rows(seed ^ 7, capacity / 2); // [capacity, 128]
            index.query = g.upload(&f32_bytes(&iq))?;
            let raw = g.upload(&f32_bytes(&raw_host))?;
            index.pooled = raw + 3 * 128 * 4;
            let out_bytes = tokens * blocks * 4;
            index.scores = g.alloc(out_bytes)?;
            let stored = tokens * 512 * 4 + blocks * 128 * 4 + out_bytes;
            let a = ([((tokens.div_ceil(4) * blocks).div_ceil(4)) as u32, 1, 1], 128, 0);
            // KBENCH_INDEX_TILE_X: the exact FP32 scorer's 4-query x 4-group warp
            // tiles (d2ccb29 index_score_grid) for 4+ tokens; same bytes as A.
            let tile_x = std::env::var("KBENCH_INDEX_TILE_X").is_ok() && tokens >= 4;
            // KBENCH_INDEX_THREAD_X: flash_qsa_scores_t (16 queries x 64 groups per CTA, no shuffles; same bytes).
            let thread_x = std::env::var("KBENCH_INDEX_THREAD_X").is_ok();
            let thread44_x = std::env::var("KBENCH_INDEX_T44_X").is_ok();
            // KBENCH_INDEX_T4_X: flash_qsa_scores_t4 (one warp per 4 queries x 32 groups; same bytes).
            let thread4_x = std::env::var("KBENCH_INDEX_T4_X").is_ok();
            let x = if order && tokens < 64 && small_x().is_some() { ([blocks.div_ceil(32) as u32, tokens.div_ceil(16) as u32, 1], 128, 0) }
                else if order { ([blocks.div_ceil(32) as u32, tokens.div_ceil(64) as u32, 1], 128, 0) }
                else if thread4_x { ([(tokens.div_ceil(4) * blocks.div_ceil(32)) as u32, 1, 1], 32, 0) }
                else if thread44_x { ([(tokens.div_ceil(32) * blocks.div_ceil(64)) as u32, 1, 1], 128, 0) }
                else if thread_x { ([(tokens.div_ceil(16) * blocks.div_ceil(64)) as u32, 1, 1], 128, 0) }
                else if tile_x { ([((tokens.div_ceil(4) * blocks.div_ceil(4)).div_ceil(4)) as u32, 1, 1], 128, 0) } else { a };
            let out = index.scores;
            let ihost = Some(IHost { q: iq, raw: raw_host, tokens, past, blocks });
            Ok(Fix { chain: Vec::new(), batch: None, extra: 0, ihost, vhost: None, attn, attn_x: None, index, cursor: 0, launch: [a, x], out, out_bytes, stored, kind, host: None })
        }
        14 => {
            // Index selection (kind 37): random scores quantized to 1/64 so the
            // 512th key has ties; the row's first `visible` groups are read.
            let mut r = Rng(seed ^ 9);
            let sc: Vec<f32> = (0..tokens * blocks).map(|_| (r.normal() * 64.0).round() / 64.0).collect();
            index.scores = g.upload(&f32_bytes(&sc))?;
            let out_bytes = tokens * blocks * 4;
            let a = ([tokens as u32, 1, 1], 256, 0);
            let out = index.scores;
            Ok(Fix { chain: Vec::new(), batch: None, extra: 0, ihost: None, vhost: None, attn, attn_x: None, index, cursor: 0, launch: [a, a], out, out_bytes, stored: out_bytes, kind, host: None })
        }
        11 => {
            // Softmax timing (kind 11): A = main's in-place softmax (probabilities
            // + denominators; replays keep the traffic, not the values), X = the
            // pairs-only softmax_md of the raw-score TC values (no P store).
            let mut r = Rng(seed ^ 8);
            let mut sc = vec![0f32; rows_q * stride];
            let mut visible_total = 0;
            for row in 0..rows_q {
                let visible = selected_length(past + row / HEADS + 1);
                visible_total += visible;
                for s in 0..visible { sc[row * stride + s] = r.normal() * 2.0; }
            }
            attn.scores = g.upload(&f32_bytes(&sc))?;
            let out_bytes = rows_q * 2 * 4;
            attn.denominators = g.alloc(out_bytes)?;
            let stored = visible_total * 4 + rows_q * 8;
            let a = ([rows_q.div_ceil(4) as u32, 1, 1], 128, 0);
            // X runs on its own copy of the raw scores: the first launch of each arm sees S (A == X on
            // the denominators); later in-place replays are timing only.
            let mut x = attn;
            x.scores = g.upload(&f32_bytes(&sc))?;
            Ok(Fix { chain: Vec::new(), batch: None, extra: 0, ihost: None, vhost: None, attn, attn_x: Some(x), index, cursor: 0, launch: [a, a], out: attn.denominators, out_bytes, stored, kind, host: None })
        }
        12 => {
            // Fused prompt attention: Q, K, V, ids and gates; the scores buffer is
            // scratch (A leaves P there, X leaves S); output = the values rows.
            let mut r = Rng(seed ^ 9);
            let q = rows(seed ^ 1, rows_q);
            let k = rows(seed ^ 2, capacity * KHEADS);
            let v = rows(seed ^ 5, capacity * KHEADS);
            let id = ids(seed ^ 3, tokens, past, blocks);
            let gates: Vec<f32> = (0..rows_q * 512).map(|_| r.normal()).collect();
            attn.query = g.upload(&f32_bytes(&q))?;
            attn.keys = g.upload(&f32_bytes(&k))?;
            attn.values = g.upload(&f32_bytes(&v))?;
            attn.ids = g.upload(&u32_bytes(&id))?;
            attn.blocks = blocks.max(512) as u32;
            attn.query_gate = g.upload(&f32_bytes(&gates))?;
            attn.scores = g.alloc(rows_q * stride * 4)?;
            attn.denominators = g.alloc(rows_q * 4)?;
            let out_bytes = rows_q * DIM * 4;
            attn.output = g.alloc(out_bytes)?;
            // As stored: Q, every selected K and V row once, gates, output.
            let stored = rows_q * DIM * 4 + 2 * touched(&id, tokens, past, blocks.max(512)) * KHEADS * DIM * 4 + rows_q * (DIM * 4 + 256 * 4);
            let x = ([(GROUPS * tokens) as u32, 1, 1], 128, 0);
            // KBENCH_QSA_KV_BF16 (benchmark screen only): arm X reads K / V rounded to BF16 (RN even).
            let attn_x = if std::env::var("KBENCH_QSA_KV_BF16").is_ok() {
                let b16 = |s: &[f32]| -> Vec<u8> { s.iter().flat_map(|f| { let b = f.to_bits(); (((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16).to_le_bytes() }).collect() };
                let mut x = attn; x.keys = g.upload(&b16(&k))?; x.values = g.upload(&b16(&v))?; Some(x)
            } else { None };
            Ok(Fix { chain: Vec::new(), batch: None, extra: 0, ihost: None, vhost: None, attn, attn_x, index, cursor: 0, launch: [x, x], out: attn.output, out_bytes, stored, kind, host: None })
        }
        13 => {
            let q = rows(seed ^ 1, rows_q);
            let k = rows(seed ^ 2, capacity * KHEADS);
            let id = ids(seed ^ 3, tokens, past, blocks);
            attn.query = g.upload(&f32_bytes(&q))?;
            attn.keys = g.upload(&f32_bytes(&k))?;
            attn.values = attn.keys;
            attn.ids = g.upload(&u32_bytes(&id))?;
            attn.blocks = blocks.max(512) as u32;
            // One buffer: probabilities [rows][stride] then denominators [rows].
            let out_bytes = rows_q * stride * 4 + rows_q * 4;
            attn.scores = g.alloc(out_bytes)?;
            attn.denominators = attn.scores + (rows_q * stride * 4) as u64;
            let visible: usize = (0..tokens).map(|t| selected_length(past + t + 1)).sum();
            let stored = rows_q * DIM * 4 + touched(&id, tokens, past, blocks.max(512)) * KHEADS * DIM * 4 + visible * HEADS * 4;
            let x = ([(GROUPS * tokens) as u32, 1, 1], 128, 0);
            Ok(Fix { chain: Vec::new(), batch: None, extra: 0, ihost: None, vhost: None, attn, attn_x: None, index, cursor: 0, launch: [x, x], out: attn.scores, out_bytes, stored, kind, host: None })
        }
        _ => Err("unknown QSA kind".into()),
    }
}
/// Pooled C8 decode stage (kind 38 = 15): `count` one-token sessions (session s
/// at past - s, each with its own caches and IDs from the one-token fixtures),
/// one launch per stage as the engine's attend_pool. Stage 0 = attention scores
/// (out [session][24][2051]), 1 = values (out [session][24][256]), 2 = index
/// scores (out [session][blocks]), 3 = top-512 selection (in place, as kind 14),
/// 4 = softmax (in place on each session's scores; out = denominators [session][24]; timing only).
pub(super) fn batch_fixture(g: &mut Gpu, stage: usize, count: usize, past: usize) -> Result<Fix> {
    if stage == 5 { return prepare_fixture(g, count, past); }
    let blocks = (past + 1) / 4;
    let (width, base) = match stage { 0 => (HEADS * MAXPOS, 7), 1 => (HEADS * DIM, 8), 2 | 3 => (blocks, 9), 4 => (HEADS, 11), _ => return Err("unknown QSA batch stage".into()) };
    let out_bytes = count * width * 4;
    let out = if stage == 3 {
        let mut r = Rng(past as u64 ^ 9);
        let sc: Vec<f32> = (0..count * width).map(|_| (r.normal() * 64.0).round() / 64.0).collect();
        g.upload(&f32_bytes(&sc))?
    } else { g.alloc(out_bytes)? };
    let (mut rows, mut gates, mut pasts, mut stored) = (Vec::new(), Vec::new(), Vec::new(), 0);
    for s in 0..count {
        let p = past - s;
        let at = out + (s * width * 4) as u64;
        let mut r = params::QsaBatchRow { keys: out, values: out, raw_index: 0, query: 0, index_query: 0, pooled: 0,
            ids: at, scores: 0, denominators: 0, norm: 0, past: p as u32, stride: MAXPOS as u32 };
        if stage == 3 { stored += (p + 1) / 4 * 4; } else {
            let f = fixture(g, base, 1, p, false)?;
            stored += f.stored;
            match stage {
                0 => { r.keys = f.attn.keys; r.values = f.attn.keys; r.query = f.attn.query; r.ids = f.attn.ids; r.scores = at; }
                1 => { r.keys = f.attn.values; r.values = f.attn.values; r.ids = f.attn.ids; r.scores = f.attn.scores;
                    r.denominators = f.attn.denominators; gates.extend_from_slice(&f.vhost.as_ref().unwrap().gates); }
                4 => { r.scores = f.attn.scores; r.denominators = at; }
                _ => { r.index_query = f.index.query; r.pooled = f.index.pooled; r.raw_index = f.index.pooled - 3 * 128 * 4; }
            }
        }
        rows.push(r);
        pasts.push(p);
    }
    let table = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(&rows[..])) };
    let p = params::QsaBatch { rows: g.upload(table)?, query_gate: if gates.is_empty() { 0 } else { g.upload(&f32_bytes(&gates))? },
        key: 0, value: 0, index: 0, output: if stage == 1 { out } else { 0 }, query_heads: HEADS as u32, key_heads: KHEADS as u32,
        query_start: 0, key_start: 0, count: count as u32, padding: 0 };
    let index = params::QsaIndex { query: 0, pooled: 0, scores: 0, tokens: 1, past: past as u32, blocks: blocks as u32, padding: 0 };
    let attn = params::QsaAttention { query: 0, query_gate: 0, keys: 0, values: 0, ids: 0, output: 0, scores: 0,
        query_heads: HEADS as u32, key_heads: KHEADS as u32, query_start: 0, key_start: 0, tokens: 1, past: past as u32,
        blocks: blocks as u32, queries_per_key: 12, stride: MAXPOS as u32, padding: 0, denominators: 0 };
    let none = ([1, 1, 1], 128, 0);
    Ok(Fix { chain: Vec::new(), batch: Some(Batch { p, stage, pasts, blocks: width }), extra: 0, ihost: None, vhost: None, attn, attn_x: None,
        index, cursor: 0, launch: [none, none], out, out_bytes, stored, kind: 15, host: None })
}
/// Pooled decode prepare (stage 5): out [session][24 Q + 2 K + 2 V][256]; the K / V cache bases are
/// offset so each session's row at its past lands in out.
fn prepare_fixture(g: &mut Gpu, count: usize, past: usize) -> Result<Fix> {
    let width = (HEADS + 2 * KHEADS) * DIM;
    let (out_bytes, mut r) = (count * width * 4, Rng(past as u64 ^ 13));
    let out = g.alloc(out_bytes)?;
    let qg: Vec<f32> = (0..count * HEADS * 512).map(|_| r.normal()).collect();
    let kv: Vec<f32> = (0..2 * count * KHEADS * DIM).map(|_| r.normal()).collect();
    let norm: Vec<f32> = (0..2 * DIM).map(|_| r.normal() * 0.1).collect();
    let (qg, kv, norm) = (g.upload(&f32_bytes(&qg))?, g.upload(&f32_bytes(&kv))?, g.upload(&f32_bytes(&norm))?);
    let (mut rows, mut pasts) = (Vec::new(), Vec::new());
    for s in 0..count {
        let (p, at) = (past - s, out + (s * width * 4) as u64);
        let row = (p * KHEADS * DIM * 4) as u64;
        rows.push(params::QsaBatchRow { keys: (at + (HEADS * DIM * 4) as u64).wrapping_sub(row),
            values: (at + ((HEADS + KHEADS) * DIM * 4) as u64).wrapping_sub(row), raw_index: 0, query: at, index_query: 0,
            pooled: 0, ids: 0, scores: 0, denominators: 0, norm, past: p as u32, stride: MAXPOS as u32 });
        pasts.push(p);
    }
    let table = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(&rows[..])) };
    let p = params::QsaBatch { rows: g.upload(table)?, query_gate: qg, key: kv, value: kv + (count * KHEADS * DIM * 4) as u64, index: 0,
        output: 0, query_heads: HEADS as u32, key_heads: KHEADS as u32, query_start: 0, key_start: 0, count: count as u32, padding: 0 };
    let index = params::QsaIndex { query: 0, pooled: 0, scores: 0, tokens: 1, past: past as u32, blocks: 0, padding: 0 };
    let attn = params::QsaAttention { query: 0, query_gate: 0, keys: 0, values: 0, ids: 0, output: 0, scores: 0,
        query_heads: HEADS as u32, key_heads: KHEADS as u32, query_start: 0, key_start: 0, tokens: 1, past: past as u32,
        blocks: 0, queries_per_key: 12, stride: MAXPOS as u32, padding: 0, denominators: 0 };
    let none = ([1, 1, 1], 128, 0);
    Ok(Fix { chain: Vec::new(), batch: Some(Batch { p, stage: 5, pasts, blocks: width }), extra: 0, ihost: None, vhost: None, attn, attn_x: None,
        index, cursor: 0, launch: [none, none], out, out_bytes, stored: count * (HEADS + 2 * KHEADS) * DIM * 4 + out_bytes, kind: 15, host: None })
}
/// attend_pool's grid and block for a batch entry: sessions padded to a power
/// of two; score planes and index blocks from the pool capacity
/// (KBENCH_QSA_POOL, default 32832 = the 32K serve context).
fn batch_launch(entry: &str, count: usize) -> Result<([u32; 3], u32, u32)> {
    let pool = std::env::var("KBENCH_QSA_POOL").map_or(Ok(32832), |s| s.parse::<usize>()).map_err(|_| "invalid KBENCH_QSA_POOL")?;
    let (padded, planes, blocks) = (count.next_power_of_two() as u32, pool.min(MAXPOS).div_ceil(4).div_ceil(4) as u32, (pool / 4).next_power_of_two());
    Ok(match entry {
        "flash_qsa_attention_scores_batch" | "flash_qsa_attention_scores_tc_batch" => ([GROUPS as u32, planes, padded], 128, 0),
        "flash_qsa_attention_values_batch" => ([KHEADS as u32 * 4, padded, 1], 768, 0),
        "flash_qsa_attention_values_tc_m_batch" => ([KHEADS as u32 * 4, padded, 1], 128, 0),
        "flash_qsa_attention_values_s16_batch" => ([KHEADS as u32 * 16, padded, 1], 192, 0),
        "flash_qsa_scores_batch" => ([blocks.div_ceil(4) as u32, padded, 1], 128, 0),
        "flash_qsa_scores_tc_batch" => ([blocks.div_ceil(32) as u32, padded, 1], 128, 0),
        "flash_qsa_select_batch" => ([1, padded, 1], 256, 0),
        "flash_qsa_attention_softmax_batch" | "flash_qsa_attention_softmax_v_batch" => ([HEADS.div_ceil(4) as u32, padded, 1], 128, 0),
        "flash_qsa_select_w_batch" => ([1, padded, 1], 1024, 0),
        "flash_qsa_prepare_batch" => ([(HEADS + KHEADS).div_ceil(4) as u32, padded, 1], 128, 0),
        _ => return Err(format!("no pooled launch shape for {entry}")),
    })
}
/// Production grids: qsa_prod_* (same entry in both arms) and qsa_batch_* (per arm).
pub(super) fn set_launch(c: &super::Case, order: bool, q: &mut Fix) -> Result<()> {
    if let Some(l) = prod_launch(c) { q.launch = [l, l]; }
    // Selection X = flash_qsa_select_w (KBENCH_XKERNEL): 1024 threads per token.
    if c.kind == 37 && std::env::var("KBENCH_XKERNEL").is_ok_and(|n| n.ends_with("_w")) { q.launch[1].1 = 1024; }
    if c.kind == 38 {
        let x = std::env::var("KBENCH_XKERNEL").ok();
        q.launch = [batch_launch(&a_name(c), c.m)?, batch_launch(x.as_deref().unwrap_or(if order { c.xkernel } else { c.kernel }), c.m)?];
    }
    Ok(())
}
/// Arm A's entry: KBENCH_AKERNEL overrides it in qsa_batch_* cases (a TC entry
/// against a same-bytes TC candidate), the case's kernel otherwise.
pub(super) fn a_name(c: &super::Case) -> String {
    match std::env::var("KBENCH_AKERNEL") { Ok(n) if c.kind == 38 => n, _ => c.kernel.to_string() }
}
/// Launch shape of a qsa_prod_* case's production entry (the host's grid and
/// block for >= 64-token prompt launches, engine/crates/engine/src/flash/qsa.rs).
pub(super) fn prod_launch(c: &super::Case) -> Option<([u32; 3], u32, u32)> {
    if !c.name.starts_with("qsa_prod_") { return None; }
    let tokens = c.m;
    Some(match c.kernel {
        "flash_qsa_attention_scores_t" => ([(GROUPS * tokens) as u32, 8, 1], 128, 0),
        "flash_qsa_attention_values_wide" => ([(KHEADS * tokens) as u32, 1, 1], 768, 0),
        "flash_qsa_attention_values_wide4t" => ([(KHEADS * tokens) as u32, 1, 1], 192, 0),
        "flash_qsa_attention_fused_tc" => ([(GROUPS * tokens) as u32, 1, 1], 128, 0),
        _ => return None,
    })
}
pub(super) fn launch(g: &Gpu, f: super::H, v: &mut Fix, arm: usize) -> Result<()> {
    let (grid, threads, shared) = v.launch[arm];
    if (v.kind == 12 || v.kind == 13) && arm == 0 && !v.chain.is_empty() {
        for i in 0..v.chain.len() {
            let (h, cg, ct) = v.chain[i];
            g.launch(h, cg, ct, 0, &mut [ptr(&mut v.attn), ptr(&mut v.cursor)])?;
        }
        Ok(())
    } else if v.extra != 0 && arm == 1 && !v.chain.is_empty() {
        let tokens = v.attn.tokens as usize;
        g.launch(f, [(KHEADS * tokens * 4) as u32, MAXPOS.div_ceil(64) as u32, 1], 128, 0, &mut [ptr(&mut v.attn), ptr(&mut v.cursor), ptr(&mut v.extra)])?;
        let (h, grid, threads) = v.chain[0];
        g.launch(h, grid, threads, 0, &mut [ptr(&mut v.attn), ptr(&mut v.cursor), ptr(&mut v.extra)])
    } else if let Some(b) = v.batch.as_mut() {
        g.launch(f, grid, threads, shared, &mut [ptr(&mut b.p)])
    } else if v.kind == 9 || v.kind == 14 {
        g.launch(f, grid, threads, shared, &mut [ptr(&mut v.index), ptr(&mut v.cursor)])
    } else {
        match v.attn_x.as_mut() {
            Some(x) if arm == 1 => g.launch(f, grid, threads, shared, &mut [ptr(x), ptr(&mut v.cursor)]),
            _ => g.launch(f, grid, threads, shared, &mut [ptr(&mut v.attn), ptr(&mut v.cursor)]),
        }
    }
}
/// kb_ex2: out[i] = ex2.approx.ftz.f32(x[i]), the softmax's exp instruction.
fn ex2_ptx() -> String {
    String::from(".version 8.8\n.target sm_121\n.address_size 64\n.visible .entry kb_ex2(.param .u64 x, .param .u64 out, .param .u32 n)\n{\n\
.reg .pred p1;\n.reg .b32 r<8>;\n.reg .b64 d<8>;\n\
ld.param.u64 d1,[x]; ld.param.u64 d2,[out]; ld.param.u32 r1,[n];\n\
mov.u32 r2,%ctaid.x; mov.u32 r3,%ntid.x; mov.u32 r4,%tid.x; mad.lo.u32 r5,r2,r3,r4;\n\
setp.ge.u32 p1,r5,r1; @p1 bra DONE;\n\
mul.wide.u32 d3,r5,4; add.s64 d4,d1,d3; add.s64 d5,d2,d3;\n\
ld.global.f32 r6,[d4]; ex2.approx.ftz.f32 r7,r6; st.global.f32 [d5],r7;\nDONE:\nret;\n}\n")
}
/// Every written element must be finite; unwritten ones keep the 0xff poison.
pub(super) fn finite(v: &Fix, bytes: &[u8]) -> bool {
    let at = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    match v.kind {
        7 => {
            let (t, p, s) = (v.attn.tokens as usize, v.attn.past as usize, v.attn.stride as usize);
            (0..t * HEADS).all(|row| (0..selected_length(p + row / HEADS + 1)).all(|i| at(row * s + i).is_finite()))
        }
        9 => {
            let (t, p, b) = (v.index.tokens as usize, v.index.past as usize, v.index.blocks as usize);
            (0..t).all(|q| {
                let visible = (p + q + 1) / 4;
                visible <= 512 || (0..visible.min(b)).all(|i| at(q * b + i).is_finite())
            })
        }
        // Softmax: main writes [rows] denominators, softmax_md [rows][2] pairs;
        // selection writes IDs over the scores.
        11 | 13 | 14 => true,
        15 => {
            let b = v.batch.as_ref().unwrap();
            b.pasts.iter().enumerate().all(|(s, &p)| match b.stage {
                0 => (0..HEADS).all(|h| (0..selected_length(p + 1)).all(|i| at((s * HEADS + h) * MAXPOS + i).is_finite())),
                1 => (0..HEADS * DIM).all(|i| at(s * HEADS * DIM + i).is_finite()),
                2 => (p + 1) / 4 <= 512 || (0..(p + 1) / 4).all(|i| at(s * b.blocks + i).is_finite()),
                _ => true,
            })
        }
        _ => bytes.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
    }
}
/// FP64 reference on sampled rows: max/mean |s - ref| / (sum|q_i k_i| / 16)
/// per arm, plus how many sampled scores are bitwise equal between arms.
pub(super) fn score_errors(h: &Host, stride: usize, a: &[u8], x: &[u8]) -> String {
    let at = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as f64;
    let mut tokens: Vec<usize> = [0, 1, 2, 3, h.tokens / 3, h.tokens / 2, h.tokens.saturating_sub(2), h.tokens - 1]
        .into_iter().filter(|&t| t < h.tokens).collect();
    tokens.dedup();
    let (mut max, mut sum, mut equal, mut n) = ([0f64; 2], [0f64; 2], 0usize, 0usize);
    for &t in &tokens {
        for head in [0, 7, 11, 12, 19, 23] {
            let row = t * HEADS + head;
            let length = h.past + t + 1;
            let q = &h.q[row * DIM..row * DIM + DIM];
            for slot in 0..selected_length(length) {
                let pos = selected_position(&h.ids, length / 4, t, h.blocks, slot);
                let k = &h.k[(pos * KHEADS + head / 12) * DIM..(pos * KHEADS + head / 12) * DIM + DIM];
                let (mut exact, mut scale) = (0f64, 0f64);
                for i in 0..DIM {
                    exact += q[i] as f64 * k[i] as f64;
                    scale += (q[i] as f64 * k[i] as f64).abs();
                }
                let (exact, scale) = (exact / 16.0, (scale / 16.0).max(1e-30));
                for (arm, b) in [a, x].into_iter().enumerate() {
                    let e = (at(b, row * stride + slot) - exact).abs() / scale;
                    max[arm] = max[arm].max(e);
                    sum[arm] += e;
                }
                equal += usize::from(a[(row * stride + slot) * 4..(row * stride + slot) * 4 + 4] == x[(row * stride + slot) * 4..(row * stride + slot) * 4 + 4]);
                n += 1;
            }
        }
    }
    format!(
        "\"ref_scores\":{},\"relerr_max_a\":{:.3e},\"relerr_max_x\":{:.3e},\"relerr_mean_a\":{:.3e},\"relerr_mean_x\":{:.3e},\"bitwise_equal_ax\":{}",
        n, max[0], max[1], sum[0] / n.max(1) as f64, sum[1] / n.max(1) as f64, equal
    )
}
/// Alone == chunked at kernel level: rescore token rows as a 1-token launch and
/// as a 16-token sub-chunk (decode/verify launch shapes) and compare their
/// visible slots with the full launch's bytes. Returns (checked rows, mismatches).
pub(super) fn chunk_check(g: &mut Gpu, f: super::H, v: &mut Fix, arm: usize, order: bool, full: &[u8]) -> Result<(usize, usize)> {
    let Some(h) = v.host.as_ref() else { return Ok((0, 0)) };
    let (tokens, past, blocks, stride) = (h.tokens, h.past, h.blocks, v.attn.stride as usize);
    let base = v.attn;
    let mut checked = 0;
    let mut bad = 0;
    for (t0, count) in [(tokens - 1, 1usize), (tokens / 2, 16usize.min(tokens - tokens / 2)), (0, 1)] {
        let bytes = count * HEADS * stride * 4;
        let out = g.alloc(bytes)?;
        unsafe { check((g.d.memset)(out, 0xff, bytes))?; }
        let mut p = base;
        p.query = base.query + (t0 * HEADS * DIM * 4) as u64;
        p.ids = base.ids + (t0 * blocks * 4) as u64;
        p.scores = out;
        p.tokens = count as u32;
        p.past = (past + t0) as u32;
        let capacity = past + t0 + count;
        let grid = if arm == 1 && order {
            [(GROUPS * count) as u32, tc_planes(count, capacity), 1]
        } else if arm == 1 {
            [(GROUPS * count) as u32, capacity.min(MAXPOS).div_ceil(keys_x()).div_ceil(4) as u32, 1]
        } else {
            [(GROUPS * count) as u32, capacity.min(MAXPOS).div_ceil(4).div_ceil(4) as u32, 1]
        };
        let mut cursor = 0u64;
        g.launch(f, grid, 128, v.launch[arm].2, &mut [ptr(&mut p), ptr(&mut cursor)])?;
        g.sync()?;
        let got = g.read(out, bytes)?;
        for r in 0..count * HEADS {
            let row = t0 * HEADS + r;
            let visible = selected_length(past + row / HEADS + 1) * 4;
            checked += 1;
            if got[r * stride * 4..r * stride * 4 + visible] != full[row * stride * 4..row * stride * 4 + visible] { bad += 1; }
        }
    }
    Ok((checked, bad))
}

/// Values FP64 reference on sampled rows (all 256 features): max/mean
/// |out - ref| / (sum|p v| / d * |gate|) per arm, and bitwise-equal count.
pub(super) fn value_errors(h: &VHost, stride: usize, a: &[u8], x: &[u8]) -> String {
    let at = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as f64;
    let mut tokens: Vec<usize> = [0, 1, 2, 3, h.tokens / 3, h.tokens / 2, h.tokens.saturating_sub(2), h.tokens - 1]
        .into_iter().filter(|&t| t < h.tokens).collect();
    tokens.dedup();
    let (mut max, mut sum, mut equal, mut n) = ([0f64; 2], [0f64; 2], 0usize, 0usize);
    for &t in &tokens {
        for head in [0, 7, 11, 12, 19, 23] {
            let row = t * HEADS + head;
            let length = h.past + t + 1;
            let d = h.denominators[row] as f64;
            let positions: Vec<usize> = (0..selected_length(length))
                .map(|slot| selected_position(&h.ids, length / 4, t, h.blocks, slot)).collect();
            for f in 0..DIM {
                let (mut exact, mut scale) = (0f64, 0f64);
                for (slot, &pos) in positions.iter().enumerate() {
                    let term = h.probabilities[row * stride + slot] as f64 * h.v[(pos * KHEADS + head / 12) * DIM + f] as f64;
                    exact += term;
                    scale += term.abs();
                }
                let gate = 1.0 / (1.0 + (-(h.gates[row * 512 + 256 + f] as f64)).exp());
                let (exact, scale) = (exact / d * gate, (scale / d * gate.abs()).max(1e-30));
                let i = row * DIM + f;
                for (arm, b) in [a, x].into_iter().enumerate() {
                    let e = (at(b, i) - exact).abs() / scale;
                    max[arm] = max[arm].max(e);
                    sum[arm] += e;
                }
                equal += usize::from(a[i * 4..i * 4 + 4] == x[i * 4..i * 4 + 4]);
                n += 1;
            }
        }
    }
    format!(
        "\"ref_values\":{},\"relerr_max_a\":{:.3e},\"relerr_max_x\":{:.3e},\"relerr_mean_a\":{:.3e},\"relerr_mean_x\":{:.3e},\"bitwise_equal_ax\":{}",
        n, max[0], max[1], sum[0] / n.max(1) as f64, sum[1] / n.max(1) as f64, equal
    )
}
/// Index scores vs an FP64 reference on sampled tokens (all visible groups):
/// max/mean |s - ref| / (sum_h sum_i |q k| / sqrt 128) per arm, and the
/// top-512 selection (descending score, then ascending group, as
/// flash_qsa_select) of each arm and of the reference: groups that differ.
pub(super) fn index_errors(h: &IHost, a: &[u8], x: &[u8]) -> String {
    let at = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let mut tokens: Vec<usize> = [0, 1, h.tokens / 3, h.tokens / 2, h.tokens.saturating_sub(2), h.tokens - 1]
        .into_iter().filter(|&t| t < h.tokens).collect();
    tokens.dedup();
    let (mut max, mut sum, mut n) = ([0f64; 2], [0f64; 2], 0usize);
    let (mut d_ar, mut d_xr, mut d_ax, mut checked) = (0usize, 0usize, 0usize, 0usize);
    let pick = |v: &[(f64, usize)]| -> Vec<usize> {
        let mut s = v.to_vec();
        s.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap().then(p.1.cmp(&q.1)));
        let mut out: Vec<usize> = s[..512].iter().map(|p| p.1).collect();
        out.sort();
        out
    };
    let diff = |p: &[usize], q: &[usize]| p.iter().filter(|g| q.binary_search(g).is_err()).count();
    for &t in &tokens {
        let visible = (h.past + t + 1) / 4;
        if visible <= 512 { continue; }
        let (mut ra, mut rx, mut rr) = (Vec::new(), Vec::new(), Vec::new());
        for b in 0..visible {
            let key = &h.raw[(4 * b + 3) * 128..(4 * b + 3) * 128 + 128];
            let (mut exact, mut scale) = (0f64, 0f64);
            for head in 0..4 {
                let q = &h.q[(t * 4 + head) * 128..(t * 4 + head) * 128 + 128];
                let (mut dot, mut abs) = (0f64, 0f64);
                for i in 0..128 { dot += q[i] as f64 * key[i] as f64; abs += (q[i] as f64 * key[i] as f64).abs(); }
                exact += dot.max(0.0);
                scale += abs;
            }
            let root = 128f64.sqrt();
            let (exact, scale) = (exact / root, (scale / root).max(1e-30));
            let i = t * h.blocks + b;
            for (arm, buf) in [a, x].into_iter().enumerate() {
                let e = (at(buf, i) as f64 - exact).abs() / scale;
                max[arm] = max[arm].max(e);
                sum[arm] += e;
            }
            n += 1;
            ra.push((at(a, i) as f64, b));
            rx.push((at(x, i) as f64, b));
            rr.push((exact, b));
        }
        let (sa, sx, sr) = (pick(&ra), pick(&rx), pick(&rr));
        d_ar += diff(&sa, &sr);
        d_xr += diff(&sx, &sr);
        d_ax += diff(&sa, &sx);
        checked += 1;
    }
    format!(
        "\"ref_index\":{},\"relerr_max_a\":{:.3e},\"relerr_max_x\":{:.3e},\"relerr_mean_a\":{:.3e},\"relerr_mean_x\":{:.3e},\"select_tokens\":{},\"select_diff_a_ref\":{},\"select_diff_x_ref\":{},\"select_diff_a_x\":{}",
        n, max[0], max[1], sum[0] / n.max(1) as f64, sum[1] / n.max(1) as f64, checked, d_ar, d_xr, d_ax
    )
}
/// Index alone == chunked: rescore token rows as 1- and 16-token launches.
pub(super) fn index_chunk_check(g: &mut Gpu, f: super::H, small: Option<super::H>, v: &mut Fix, arm: usize, order: bool, full: &[u8]) -> Result<(usize, usize)> {
    let Some(h) = v.ihost.as_ref() else { return Ok((0, 0)) };
    let (tokens, past, blocks) = (h.tokens, h.past, h.blocks);
    let (query, pooled) = (v.index.query, v.index.pooled);
    let (mut checked, mut bad) = (0, 0);
    for (t0, count) in [(tokens - 1, 1usize), (tokens / 2, 16usize.min(tokens - tokens / 2)), (0, 1)] {
        let bytes = count * blocks * 4;
        let out = g.alloc(bytes)?;
        unsafe { check((g.d.memset)(out, 0xff, bytes))?; }
        let mut p = params::QsaIndex { query: query + (t0 * 4 * 128 * 4) as u64, pooled, scores: out,
            tokens: count as u32, past: (past + t0) as u32, blocks: blocks as u32, padding: 0 };
        // Production picks the decode / verify entry below 64 tokens (KBENCH_XKERNEL_SMALL).
        let (f, grid, threads) = match small {
            Some(s) if arm == 1 && order && count < 64 => (s, [blocks.div_ceil(32) as u32, count.div_ceil(16) as u32, 1], 128),
            _ if arm == 1 && order => (f, [blocks.div_ceil(32) as u32, count.div_ceil(64) as u32, 1], 128),
            _ if arm == 1 && v.launch[1].1 == 32 => (f, [(count.div_ceil(4) * blocks.div_ceil(32)) as u32, 1, 1], 32),
            _ => (f, [((count.div_ceil(4) * blocks).div_ceil(4)) as u32, 1, 1], 128),
        };
        let mut cursor = 0u64;
        g.launch(f, grid, threads, 0, &mut [ptr(&mut p), ptr(&mut cursor)])?;
        g.sync()?;
        let got = g.read(out, bytes)?;
        for r in 0..count {
            let row = t0 + r;
            let visible = (past + row + 1) / 4;
            if visible <= 512 { continue; }
            checked += 1;
            if got[r * blocks * 4..(r * blocks + visible) * 4] != full[row * blocks * 4..(row * blocks + visible) * 4] { bad += 1; }
        }
    }
    Ok((checked, bad))
}
/// Values alone == chunked: rerun token rows as 1- and 16-token launches.
pub(super) fn values_chunk_check(g: &mut Gpu, f: super::H, small: Option<super::H>, v: &mut Fix, arm: usize, order: bool, full: &[u8]) -> Result<(usize, usize)> {
    let Some(h) = v.vhost.as_ref() else { return Ok((0, 0)) };
    if v.extra != 0 && arm == 1 { return Ok((0, 0)); } // split values: two kernels, checked by bytes instead
    let (tokens, past, blocks, stride) = (h.tokens, h.past, h.blocks, v.attn.stride as usize);
    let (base, pairs) = match v.attn_x { Some(x) if arm == 1 => (x, 2), _ => (v.attn, 1) };
    let (mut checked, mut bad) = (0, 0);
    for (t0, count) in [(tokens - 1, 1usize), (tokens / 2, 16usize.min(tokens - tokens / 2)), (0, 1)] {
        let bytes = count * HEADS * DIM * 4;
        let out = g.alloc(bytes)?;
        unsafe { check((g.d.memset)(out, 0xff, bytes))?; }
        let mut p = base;
        p.scores = base.scores + (t0 * HEADS * stride * 4) as u64;
        p.ids = base.ids + (t0 * blocks * 4) as u64;
        p.denominators = base.denominators + (t0 * HEADS * pairs * 4) as u64;
        p.query_gate = base.query_gate + (t0 * HEADS * 512 * 4) as u64;
        p.output = out;
        p.tokens = count as u32;
        p.past = (past + t0) as u32;
        let small = small.filter(|_| arm == 1 && order && count < 64);
        let f = small.unwrap_or(f);
        let (grid, threads) = if small.is_some() { ([(KHEADS * count * 4) as u32, 1, 1], 128) }
            else if arm == 1 && order { ([(KHEADS * count) as u32, 1, 1], 128) }
            else if arm == 1 { ([(KHEADS * count * value_slices_x()) as u32, 1, 1], v.launch[1].1) }
            else { ([(KHEADS * count * 4) as u32, 1, 1], 768) };
        let mut cursor = 0u64;
        g.launch(f, grid, threads, 0, &mut [ptr(&mut p), ptr(&mut cursor)])?;
        g.sync()?;
        let got = g.read(out, bytes)?;
        for r in 0..count * HEADS {
            let row = t0 * HEADS + r;
            checked += 1;
            if got[r * DIM * 4..(r + 1) * DIM * 4] != full[row * DIM * 4..(row + 1) * DIM * 4] { bad += 1; }
        }
    }
    Ok((checked, bad))
}

/// L2 random-gather microbench (kind 10): one CTA per (token, KV head) reads
/// the 2048 selected 1 KB FP32 K rows of its token with 128-bit L2-only loads
/// (ld.global.cg, the cp.async.cg path), four groups in flight per warp, and
/// XOR-folds them so nothing is dead. k = 0: KV head interleaved (token-major,
/// as today's grids); k = 1: KV head is the slow grid index.
pub(super) fn gather_ptx() -> String {
    let mut s = String::from(".version 8.8\n.target sm_121\n.address_size 64\n");
    s += ".visible .entry kb_gather(.param .u64 ids, .param .u64 base, .param .u64 out, .param .u32 tokens, .param .u32 stride, .param .u32 slow)\n{\n";
    s += ".reg .pred p<4>;\n.reg .b32 r<128>;\n.reg .b64 d<64>;\n";
    s += "ld.param.u64 d1,[ids]; ld.param.u64 d2,[base]; ld.param.u64 d3,[out]; ld.param.u32 r1,[tokens]; ld.param.u32 r2,[stride]; ld.param.u32 r3,[slow];\n";
    s += "mov.u32 r4,%ctaid.x; mov.u32 r5,%tid.x; shr.u32 r6,r5,5; and.b32 r7,r5,31;\n";
    s += "setp.ne.u32 p1,r3,0; div.u32 r8,r4,r1; rem.u32 r9,r4,r1; and.b32 r10,r4,1; shr.u32 r11,r4,1; selp.b32 r12,r8,r10,p1; selp.b32 r13,r9,r11,p1;\n";
    s += "mul.wide.u32 d4,r13,r2; shl.b64 d4,d4,2; add.s64 d5,d1,d4;\n";
    s += "mul.wide.u32 d6,r7,16; mul.wide.u32 d7,r12,1024; add.s64 d8,d2,d7; add.s64 d8,d8,d6;\n";
    s += "mov.u32 r20,0; mov.u32 r14,0;\nLOOP:\n";
    for k in 0..4 {
        s += &format!("add.u32 r{a},r14,{k}; mul.wide.u32 d{b},r{a},4; add.s64 d{b},d5,d{b}; ld.global.nc.u32 r{c},[d{b}];\n", a = 30 + k, b = 10 + k, c = 40 + k);
    }
    for k in 0..4 {
        let v = 50 + 8 * k;
        s += &format!("shl.b32 r{a},r{g},2; add.u32 r{a},r{a},r6; mul.wide.u32 d{d},r{a},2048; add.s64 d{d},d8,d{d};\n", a = 44 + k, g = 40 + k, d = 20 + k);
        s += &format!("ld.global.cg.v4.u32 {{r{},r{},r{},r{}}},[d{}]; ld.global.cg.v4.u32 {{r{},r{},r{},r{}}},[d{}+512];\n",
            v, v + 1, v + 2, v + 3, 20 + k, v + 4, v + 5, v + 6, v + 7, 20 + k);
    }
    for r in 50..82 { s += &format!("xor.b32 r20,r20,r{r};\n"); }
    s += "add.u32 r14,r14,4; setp.lt.u32 p2,r14,512; @p2 bra LOOP;\n";
    s += "mul.wide.u32 d30,r4,128; cvt.u64.u32 d31,r5; add.s64 d30,d30,d31; shl.b64 d30,d30,2; add.s64 d30,d3,d30; st.global.u32 [d30],r20;\nret;\n}\n";
    s
}
pub(super) struct Gather { pub ids: u64, pub base: u64, pub out: u64, pub tokens: u32, pub stride: u32, pub slow: u32 }
pub(super) fn gather_fixture(g: &mut Gpu, tokens: usize, past: usize, slow: bool) -> Result<(Gather, usize, usize)> {
    let capacity = past + tokens;
    let blocks = capacity / 4;
    if past < 2048 { return Err("gather microbench needs sparse rows (past >= 2048)".into()); }
    let k = rows(11, capacity * KHEADS);
    let id = ids(12, tokens, past, blocks);
    let stride = blocks.max(512);
    let out_bytes = tokens * KHEADS * 128 * 4;
    let fix = Gather { ids: g.upload(&u32_bytes(&id))?, base: g.upload(&f32_bytes(&k))?, out: g.alloc(out_bytes)?,
        tokens: tokens as u32, stride: stride as u32, slow: u32::from(slow) };
    Ok((fix, out_bytes, tokens * KHEADS * 2048 * 1024))
}

/// Kind 12 arm A: scores _tc_w (grid y = tc_planes), main's softmax (a warp
/// per row), values _tc (a CTA per token x KV head), all from the X module.
/// Arm-X helper kernels set up after the modules load: the fused chain (kind 35) or split values' phase B.
pub(super) fn chains(g: &Gpu, c: &super::Case, x: super::H, v: &mut Fix) -> Result<()> {
    if c.kind == 35 && prod_launch(c).is_none() { return fused_chain(g, x, v); }
    if v.extra != 0 { v.chain = vec![(g.function(x, "flash_qsa_attention_values_tc_kr")?, [(KHEADS * v.attn.tokens as usize * 4) as u32, 1, 1], 256)]; }
    Ok(())
}
pub(super) fn fused_chain(g: &Gpu, x: super::H, v: &mut Fix) -> Result<()> {
    let (tokens, capacity) = (v.attn.tokens as usize, (v.attn.past + v.attn.tokens) as usize);
    v.chain = vec![
        (g.function(x, "flash_qsa_attention_scores_tc_w")?, [(GROUPS * tokens) as u32, tc_planes(tokens, capacity), 1], 128),
        (g.function(x, "flash_qsa_attention_softmax")?, [(tokens * HEADS).div_ceil(4) as u32, 1, 1], 128),
        (g.function(x, "flash_qsa_attention_values_tc")?, [(KHEADS * tokens) as u32, 1, 1], 128),
    ];
    Ok(())
}

/// Kind 13 arm A: main's scores (grid y = ceil(slots / 16)) then main's softmax.
pub(super) fn ssm_chain(g: &Gpu, x: super::H, v: &mut Fix) -> Result<()> {
    let (tokens, capacity) = (v.attn.tokens as usize, (v.attn.past + v.attn.tokens) as usize);
    v.chain = vec![
        (g.function(x, "flash_qsa_attention_scores")?, [(GROUPS * tokens) as u32, capacity.min(MAXPOS).div_ceil(4).div_ceil(4) as u32, 1], 128),
        (g.function(x, "flash_qsa_attention_softmax")?, [(tokens * HEADS).div_ceil(4) as u32, 1, 1], 128),
    ];
    Ok(())
}

/// The same query rows must agree when rerun with their original absolute past
/// as one- and sixteen-token launches. Only row-view addresses change; selected
/// IDs, K/V, score stride, head geometry, and the FP32 output contract stay fixed.
pub(super) fn fused_chunk_check(g: &mut Gpu, f: super::H, v: &mut Fix, full: &[u8]) -> Result<(usize, usize)> {
    let base = v.attn;
    let tokens = base.tokens as usize;
    let mut checked = 0;
    let mut bad = 0;
    for (t0, count) in [(tokens - 1, 1), (tokens / 2, 16usize.min(tokens - tokens / 2)), (0, 1)] {
        let bytes = count * HEADS * DIM * 4;
        let out = g.alloc(bytes)?;
        unsafe { check((g.d.memset)(out, 0xff, bytes))?; }
        let mut p = base;
        p.query += (t0 * HEADS * DIM * 4) as u64;
        p.query_gate += (t0 * HEADS * 512 * 4) as u64;
        p.scores += (t0 * HEADS * base.stride as usize * 4) as u64;
        p.ids += (t0 * base.blocks as usize * 4) as u64;
        p.tokens = count as u32;
        p.past += t0 as u32;
        p.output = out;
        let mut cursor = 0u64;
        g.launch(f, [(GROUPS * count) as u32, 1, 1], 128, 0, &mut [ptr(&mut p), ptr(&mut cursor)])?;
        g.sync()?;
        let got = g.read(out, bytes)?;
        for row in 0..count * HEADS {
            let first = (t0 * HEADS + row) * DIM * 4;
            checked += 1;
            if got[row * DIM * 4..(row + 1) * DIM * 4] != full[first..first + DIM * 4] { bad += 1; }
        }
    }
    Ok((checked, bad))
}
