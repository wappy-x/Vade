//! GDN scan cases (owner gdn_scan). A always runs main's production pipeline;
//! X runs either the same pipeline (bytes A==X required) or an order-changed
//! chunk pipeline (bytes reported, never required equal; error vs A printed).
//! Shapes: Flash-Next 48 value heads, 16 key heads, D=128, 2,048-token pieces.
use super::{Case, Gpu, Graph, H, Options, Result, check, grade, median, sample, sha256};
use std::{ffi::c_void, ptr::null_mut, time::Instant};
#[allow(dead_code, clippy::module_inception)]
#[path = "../../../crates/gpu/src/gdn_params.rs"]
mod gdn;
#[path = "kernel_chain.rs"]
mod chain;
const HEADS: usize = 48;
const KEYS: usize = 16;
const D: usize = 128;
const C: usize = 64;
const STATE: usize = HEADS * D * D;

pub(super) struct Launch {
    f: H,
    grid: [u32; 3],
    block: u32,
    shared: u32,
    params: Vec<Vec<u64>>,
}
fn blob<T>(v: &T) -> Vec<u64> {
    let n = std::mem::size_of::<T>();
    let mut out = vec![0u64; n.div_ceil(8).max(1)];
    unsafe { std::ptr::copy_nonoverlapping((v as *const T).cast::<u8>(), out.as_mut_ptr().cast::<u8>(), n) };
    out
}
fn run_launch(g: &Gpu, l: &mut Launch) -> Result<()> {
    if l.block == 0 {
        // A device copy [dst, src, bytes] between launches (verify rollback in
        // the split checks; never part of a timed arm).
        let (dst, src, bytes) = (l.params[0][0], l.params[0][1], l.params[0][2] as usize);
        g.sync()?;
        unsafe { check((g.d.copy)(dst, src, bytes))? };
        return g.sync();
    }
    let mut args: Vec<H> = l.params.iter_mut().map(|b| b.as_mut_ptr().cast::<c_void>()).collect();
    if l.shared & PDL != 0 { return g.launch_pdl(l.f, l.grid, l.block, l.shared & !PDL, &mut args); }
    unsafe { check((g.d.launch)(l.f, l.grid[0], l.grid[1], l.grid[2], l.block, 1, 1, l.shared, g.s, args.as_mut_ptr(), null_mut())) }
}
/// Flag bit in Launch::shared: launch with programmatic stream serialization.
const PDL: u32 = 1 << 31;
/// VADE_KBENCH_GDN_PDL=1: X-module chunk scans start in prep's tail (programmatic
/// launch; the module's scan must griddepcontrol.wait, as the engine's does).
fn scan_pdl() -> u32 { if std::env::var("VADE_KBENCH_GDN_PDL").is_ok_and(|v| v == "1") { PDL } else { 0 } }
fn function(g: &Gpu, m: H, name: &str, shared: u32) -> Result<H> {
    let f = g.function(m, name).map_err(|e| format!("{name}: {e}"))?;
    if shared > 48 * 1024 {
        unsafe { check((g.d.attribute)(f, 8, shared as i32)).map_err(|e| format!("{name} shared {shared}: {e}"))? };
    }
    Ok(f)
}
/// flash_next_gdn_chunk_rows and its dynamic shared bytes. A module whose rows entry has no static
/// shared takes gdn::CHUNK_ROWS_BYTES. An older module with a static layout either refuses that
/// attribute (static + dynamic over the per-block cap) and launches with 0, or ignores the bytes.
fn rows_function(g: &Gpu, m: H) -> Result<(H, u32)> {
    const ROWS: &str = "flash_next_gdn_chunk_rows";
    match function(g, m, ROWS, gdn::CHUNK_ROWS_BYTES) { Ok(f) => Ok((f, gdn::CHUNK_ROWS_BYTES)), Err(_) => Ok((function(g, m, ROWS, 0)?, 0)) }
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.unit() }
}
/// Token-major Flash-Next inputs exactly as conv_prepare_tokens lays them out:
/// q/k expanded per value head (identical within a key group), L2-normalised
/// k, q scaled by 1/sqrt(128); raw v; beta = sigmoid; g = -decay (log space).
struct Inputs { q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, beta: Vec<f32>, g: Vec<f32>, state: Vec<f32> }
fn inputs(tokens: usize, rows: usize) -> Inputs {
    let n = tokens * rows;
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut q, mut k) = (vec![0f32; n * HEADS * D], vec![0f32; n * HEADS * D]);
    for t in 0..n {
        for key in 0..KEYS {
            let raw_q: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
            let raw_k: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
            let sq = 1. / (raw_q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            let sk = 1. / (raw_k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            for h in key * 3..key * 3 + 3 {
                for d in 0..D {
                    q[(t * HEADS + h) * D + d] = raw_q[d] * sq * (1. / 128f32.sqrt());
                    k[(t * HEADS + h) * D + d] = raw_k[d] * sk;
                }
            }
        }
    }
    let v = (0..n * HEADS * D).map(|_| r.range(-1., 1.)).collect();
    let beta = (0..n * HEADS).map(|_| 1. / (1. + (-r.range(-3., 3.)).exp())).collect();
    let g = (0..n * HEADS).map(|_| -(r.range(0.001f32.ln(), 0.0)).exp()).collect();
    let state = (0..rows * STATE).map(|_| r.range(-0.1, 0.1)).collect();
    Inputs { q, k, v, beta, g, state }
}
fn floats(g: &mut Gpu, v: &[f32]) -> Result<u64> {
    g.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}

/// One arm's work for one ring copy: optional resets (device copies before a
/// check run), the launches, and the canonical output regions to hash.
struct Arm {
    resets: Vec<(u64, u64, usize)>,
    launches: Vec<Launch>,
    hashed: Vec<(u64, usize)>,
    /// 0: token-major out + [h][k][v] state (canonical); 1: head-major out (fla0);
    /// 2: q/k regions hashed on each key group's first value head only
    layout: u8,
}
struct Fixture { arms: [Vec<Arm>; 2], stored: usize, flops: usize, copies: usize }

/// main's prompt path: flash_next_gdn_scan per 2,048-token piece, 192x32,
/// state carried piece to piece through output_state (input state untouched).
fn sequential(g: &mut Gpu, f: H, dev: [u64; 5], tokens: usize, pieces: usize, state_in: u64, out: u64) -> Result<Arm> {
    let mut launches = Vec::new();
    let mut prior = state_in;
    let width = HEADS * D;
    let mut last = 0;
    for p in 0..pieces {
        let next = g.alloc(STATE * 4)?;
        let desc = g.upload(&[prior.to_le_bytes(), 0u64.to_le_bytes(), next.to_le_bytes()].concat())?;
        let at = (p * tokens * width * 4) as u64;
        let gate = (p * tokens * HEADS * 4) as u64;
        let params = gdn::GdnStepBatch {
            q: dev[0] + at, k: dev[1] + at, v: dev[2] + at, beta: dev[3] + gate, g: dev[4] + gate,
            rows: desc, out: out + at, sessions: 0, heads: HEADS as u32, count: tokens as u32,
        };
        launches.push(Launch { f, grid: [(HEADS * 4) as u32, 1, 1], block: 32, shared: 0, params: vec![blob(&params)] });
        prior = next;
        last = next;
    }
    Ok(Arm { resets: vec![], launches, hashed: vec![(out, tokens * pieces * width * 4), (last, STATE * 4)], layout: 0 })
}

/// Existing FP32 chunk kernels (flash_gdn_chunk.rs, hybrid path) at Flash-Next
/// shapes: head-major Q/K per key head (key_repeat 3), beta-scaled V, in-place state.
fn fla0(g: &mut Gpu, m: H, x: &Inputs, tokens: usize, state_in: u64) -> Result<Arm> {
    let chunks = tokens.div_ceil(C);
    let (mut qh, mut kh) = (vec![0f32; KEYS * tokens * D], vec![0f32; KEYS * tokens * D]);
    let mut vh = vec![0f32; HEADS * tokens * D];
    let (mut bh, mut gh) = (vec![0f32; HEADS * tokens], vec![0f32; HEADS * tokens]);
    for t in 0..tokens {
        for h in 0..HEADS {
            let beta = x.beta[t * HEADS + h];
            bh[h * tokens + t] = beta;
            gh[h * tokens + t] = x.g[t * HEADS + h];
            for d in 0..D {
                vh[(h * tokens + t) * D + d] = x.v[(t * HEADS + h) * D + d] * beta;
                if h % 3 == 0 {
                    qh[(h / 3 * tokens + t) * D + d] = x.q[(t * HEADS + h) * D + d];
                    kh[(h / 3 * tokens + t) * D + d] = x.k[(t * HEADS + h) * D + d];
                }
            }
        }
    }
    let (q, k, v, beta, gg) = (floats(g, &qh)?, floats(g, &kh)?, floats(g, &vh)?, floats(g, &bh)?, floats(g, &gh)?);
    let cumulative = g.alloc(chunks * HEADS * C * 4)?;
    let chunk_decay = g.alloc(chunks * HEADS * 4)?;
    let system = g.alloc(chunks * HEADS * C * C * 4)?;
    let attention = g.alloc(chunks * HEADS * C * C * 4)?;
    let values = g.alloc(chunks * HEADS * C * D * 4)?;
    let decayed = g.alloc(chunks * HEADS * C * D * 4)?;
    let state = g.alloc(STATE * 4)?;
    let out = g.alloc(HEADS * tokens * D * 4)?;
    let (h, t) = (HEADS as u32, tokens as u32);
    let system_shared = 4 * (128 * 33 + 2 * 32 * 128);
    let scan_shared = 4 * (128 * 64 + 2 * 64 * 128);
    let mut launches = vec![Launch {
        f: function(g, m, "flash_gdn_cumulative", 0)?,
        grid: [(chunks * HEADS).div_ceil(128) as u32, 1, 1],
        block: 128,
        shared: 0,
        params: vec![blob(&gg), blob(&cumulative), blob(&chunk_decay), blob(&h), blob(&t)],
    }];
    let sys = super::params::GdnSystem {
        q: q as _, k: k as _, beta: beta as _, cumulative: cumulative as _, system: system as _, attention: attention as _,
        heads: h, tokens: t, key_repeat: 3,
    };
    launches.push(Launch { f: function(g, m, "flash_gdn_system", system_shared)?, grid: [(chunks * HEADS * 4) as u32, 1, 1], block: 256, shared: system_shared, params: vec![blob(&sys)] });
    let solve = super::params::GdnSolve {
        qkv: std::ptr::null(), conv: std::ptr::null(), history: std::ptr::null(), keys: KEYS as u32, precision: 0,
        k: k as _, v: v as _, beta: beta as _, cumulative: cumulative as _, system: system as _,
        new_values: values as _, decayed_keys: decayed as _, heads: h, tokens: t, key_repeat: 3,
    };
    launches.push(Launch { f: function(g, m, "flash_gdn_solve", 0)?, grid: [(chunks * HEADS * 2) as u32, 1, 1], block: 512, shared: 0, params: vec![blob(&solve)] });
    let scan = super::params::GdnScan {
        q: q as _, k: k as _, attention: attention as _, new_values: values as _, decayed_keys: decayed as _,
        chunk_decay: chunk_decay as _, cumulative: cumulative as _, state: state as _, output: out as _,
        heads: h, tokens: t, start: 0, key_repeat: 3,
    };
    launches.push(Launch { f: function(g, m, "flash_gdn_chunk_scan", scan_shared)?, grid: [(HEADS * 2) as u32, 1, 1], block: 512, shared: scan_shared, params: vec![blob(&scan)] });
    Ok(Arm { resets: vec![(state, state_in, STATE * 4)], launches, hashed: vec![(out, HEADS * tokens * D * 4), (state, STATE * 4)], layout: 1 })
}

/// main's decode path: flash_next_gdn_scan_slot over `rows` chats (96 x rows CTAs
/// of 64 threads), each chat's StateSlot pointing at its own input/output state.
fn slot(g: &mut Gpu, f: H, rows: usize, dev: [u64; 5], state: &[f32]) -> Result<Arm> {
    let mut table = vec![0u8; 17 * 1168];
    let mut outputs = Vec::new();
    for r in 0..rows {
        let input = floats(g, &state[r * STATE..(r + 1) * STATE])?;
        let output = g.alloc(STATE * 4)?;
        let at = (r + 1) * 1168;
        table[at..at + 8].copy_from_slice(&input.to_le_bytes());
        table[at + 16..at + 24].copy_from_slice(&output.to_le_bytes());
        outputs.push(output);
    }
    let table = g.upload(&table)?;
    let slots = g.words(&(1..=rows as u32).collect::<Vec<_>>())?;
    let out = g.alloc(rows * HEADS * D * 4)?;
    let params = gdn::GdnStepBatch {
        q: dev[0], k: dev[1], v: dev[2], beta: dev[3], g: dev[4], rows: 0, out, sessions: 0, heads: HEADS as u32, count: 1,
    };
    let step: [u64; 3] = [slots, table, (rows as u64) << 32];
    let mut hashed = vec![(out, rows * HEADS * D * 4)];
    hashed.extend(outputs.iter().map(|&o| (o, STATE * 4)));
    Ok(Arm { resets: vec![], launches: vec![Launch { f, grid: [96, rows as u32, 1], block: 64, shared: 0, params: vec![blob(&params), blob(&step)] }], hashed, layout: 0 })
}

/// Order-changed chunk pipeline (X arm only): per call, flash_next_gdn_chunk_prep
/// (call chunks x key heads CTAs) then flash_next_gdn_chunk_scan (one CTA per
/// head), in place on one chunk-form state that each check run resets from
/// `pristine`. Calls are (first row, rows); base = first row (absolute position).
/// Hashed: all outputs, S, and the open chunk's live tail rows only.
pub(super) const SCAN_BYTES: u32 = gdn::CHUNK_SCAN_BYTES;
const CHUNK_STATE: usize = gdn::chunk_state_floats(HEADS, KEYS);
struct Work { a: u64, p: u64, gamma: u64, k0: u64 }
fn work(g: &mut Gpu, chunks: usize) -> Result<Work> {
    Ok(Work {
        a: g.alloc(chunks * HEADS * C * C * 4)?, p: g.alloc(chunks * HEADS * C * C * 4)?,
        gamma: g.alloc(chunks * HEADS * C * 4)?, k0: g.alloc(KEYS * C * D * 4)?,
    })
}
fn pristine(g: &mut Gpu, s: &[f32]) -> Result<u64> {
    let mut host = s[..STATE].to_vec();
    host.resize(CHUNK_STATE, 0.0);
    floats(g, &host)
}
/// A StateSlot table (17 records of 1,168 bytes) whose slot r+1 holds states[r]
/// as layer 0's state at positions[r]; returns (table, slot ids).
fn table(g: &mut Gpu, states: &[u64], positions: &[usize]) -> Result<(u64, u64)> {
    let mut t = vec![0u8; 17 * 1168];
    for (r, (&st, &pos)) in states.iter().zip(positions).enumerate() {
        let at = (r + 1) * 1168;
        t[at..at + 8].copy_from_slice(&st.to_le_bytes());
        t[at + 1160..at + 1164].copy_from_slice(&(pos as u32).to_le_bytes());
    }
    Ok((g.upload(&t)?, g.words(&(1..=states.len() as u32).collect::<Vec<_>>())?))
}
/// True when some prompt call is small enough for the row kernel (unless
/// VADE_KBENCH_GDN_NO_ROWS is set: then every call takes prep + scan).
fn count_small(calls: &[(usize, usize, u8)]) -> bool {
    std::env::var("VADE_KBENCH_GDN_NO_ROWS").is_err() && calls.iter().any(|c| c.2 != DECODE && c.1 <= gdn::CHUNK_ROWS_LIMIT)
}
/// Call kinds for `chunk`: a prompt call (prep + scan), one decode row through
/// the slot kernel, or a verify window that is rejected: the state is saved,
/// the window runs, and the state is restored (its output rows are rewritten
/// by the calls that follow, as after an MTP rollback).
const PROMPT: u8 = 0;
const DECODE: u8 = 1;
const REJECT: u8 = 2;
/// Output gate for `chunk_gated` (kind 16): z and norm inputs, the gated
/// output, and whether the scan applies it (fused) or a flash_gdn_gate launch
/// follows the calls (the unfused chunk path, the A arm).
#[derive(Clone, Copy)]
/// conv = [x, w, history] of the conv call (0 = the scan reads v): the scan
/// convolves its value rows itself (kind 17, X arm).
struct Gate { z: u64, norm: u64, gated: u64, fused: bool, conv: [u64; 3], gates: [u64; 4], pk: (u64, usize) }
fn chunk(g: &mut Gpu, m: H, dev: [u64; 5], calls: &[(usize, usize, u8)], pristine: u64, out: u64, w: &Work, x_module: bool) -> Result<Arm> {
    chunk_gated(g, m, dev, calls, pristine, out, w, None, x_module)
}
#[allow(clippy::too_many_arguments)]
fn chunk_gated(g: &mut Gpu, m: H, dev: [u64; 5], calls: &[(usize, usize, u8)], pristine: u64, out: u64, w: &Work, gate: Option<Gate>, x_module: bool) -> Result<Arm> {
    let prep = function(g, m, "flash_next_gdn_chunk_prep", 0)?;
    let scan = function(g, m, if gate.is_some_and(|f| f.conv[0] != 0) { "flash_next_gdn_chunk_scan_cv" } else { "flash_next_gdn_chunk_scan" }, SCAN_BYTES)?;
    let slot = function(g, m, "flash_next_gdn_chunk_slot", 0)?;
    // Small prompt calls take the row kernel exactly as the engine dispatches
    // them (X modules without it, e.g. older branches, keep prep + scan).
    let rows = if count_small(calls) { rows_function(g, m).ok() } else { None };
    let state = g.alloc(CHUNK_STATE * 4)?;
    let desc = g.upload(&[state.to_le_bytes(), 0u64.to_le_bytes(), 0u64.to_le_bytes()].concat())?;
    let width = HEADS * D;
    let mut launches = Vec::new();
    let snap = if calls.iter().any(|c| c.2 == REJECT) { g.alloc(CHUNK_STATE * 4)? } else { 0 };
    let copy = |dst: u64, src: u64| Launch { f: null_mut(), grid: [0; 3], block: 0, shared: 0, params: vec![vec![dst, src, (CHUNK_STATE * 4) as u64]] };
    for &(start, count, kind) in calls {
        if kind == DECODE {
            // One decode row through the slot kernel (rows = 1 chat at position `start`).
            let (t, ids) = table(g, &[state], &[start])?;
            let at = (start * width * 4) as u64;
            let gate = (start * HEADS * 4) as u64;
            let params = gdn::GdnChunkNext {
                q: dev[0] + at, k: dev[1] + at, v: dev[2] + at, beta: dev[3] + gate, g: dev[4] + gate,
                rows: 0, out: out + at, a: 0, p: 0, gamma: 0, k0: 0,
                heads: HEADS as u32, keys: KEYS as u32, count: 1, base: start as u32,
                z: 0, norm: 0, gated: 0, epsilon: 0.0, spare: 0, x: 0, w: 0, history: 0, channels: 0, x_start: 0, ga: 0, gb: 0, alog: 0, dt: 0, pk: 0, pk_tokens: 0, pk_row0: 0,
            };
            let step: [u64; 3] = [ids, t, 1u64 << 32];
            launches.push(Launch { f: slot, grid: [HEADS as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&params), blob(&step)] });
            continue;
        }
        let chunks = (start % C + count).div_ceil(C);
        let at = (start * width * 4) as u64;
        let gate_at = (start * HEADS * 4) as u64;
        let fuse = gate.filter(|f| f.fused && count > gdn::CHUNK_ROWS_LIMIT);
        let params = gdn::GdnChunkNext {
            q: dev[0] + at, k: dev[1] + at, v: dev[2] + at, beta: dev[3] + gate_at, g: dev[4] + gate_at,
            rows: desc, out: out + at, a: w.a, p: w.p, gamma: w.gamma, k0: w.k0,
            heads: HEADS as u32, keys: KEYS as u32, count: count as u32, base: start as u32,
            z: fuse.map_or(0, |f| f.z + at), norm: fuse.map_or(0, |f| f.norm), gated: fuse.map_or(0, |f| f.gated + at),
            epsilon: 1e-6, spare: 0,
            x: gate.map_or(0, |f| f.conv[0]), w: gate.map_or(0, |f| f.conv[1]), history: gate.map_or(0, |f| f.conv[2]),
            channels: if gate.is_some_and(|f| f.conv[0] != 0) { ((2 * KEYS + HEADS) * D) as u32 } else { 0 }, x_start: start as u32,
            // Gate projections per call row (like q/k/beta/g); A_log/dt_bias per head.
            ga: gate.filter(|f| f.gates[0] != 0).map_or(0, |f| f.gates[0] + gate_at), gb: gate.filter(|f| f.gates[1] != 0).map_or(0, |f| f.gates[1] + gate_at),
            alog: gate.map_or(0, |f| f.gates[2]), dt: gate.map_or(0, |f| f.gates[3]),
            // Packed out-projection input (kind 51 X arm): rows of all calls, this call at row `start`.
            pk: gate.map_or(0, |f| f.pk.0), pk_tokens: gate.map_or(0, |f| f.pk.1 as u32), pk_row0: start as u32,
        };
        if kind == REJECT { launches.push(copy(snap, state)); }
        match rows {
            Some((f, bytes)) if count <= gdn::CHUNK_ROWS_LIMIT => {
                let params = gdn::GdnChunkNext { k0: state, ..params };
                launches.push(Launch { f, grid: [HEADS as u32, 1, 1], block: 128, shared: bytes, params: vec![blob(&params)] });
            }
            _ => {
                launches.push(Launch { f: prep, grid: [chunks as u32, KEYS as u32, 1], block: 256, shared: 0, params: vec![blob(&params)] });
                if std::env::var("VADE_KBENCH_GDN_ONLY_PREP").is_err() {
                    launches.push(Launch { f: scan, grid: [HEADS as u32, 1, 1], block: 512, shared: SCAN_BYTES | if x_module { scan_pdl() } else { 0 }, params: vec![blob(&params)] });
                }
            }
        }
        if kind == REJECT { launches.push(copy(state, snap)); }
    }
    let total = calls.iter().filter(|c| c.2 != REJECT).map(|&(s, n, _)| s + n).max().unwrap_or(0);
    let end = total % C;
    let word = |x: usize| state + (x * 4) as u64;
    if let Some(f) = gate.filter(|f| !f.fused) {
        let p = super::params::GdnGate { x: out as _, z: f.z as _, norm: f.norm as _, output: f.gated as _,
            heads: (HEADS * total) as u32, tokens: 1, precision: 0, packed: 0, epsilon: 1e-6, sessions: 0 };
        launches.push(Launch { f: function(g, m, "flash_gdn_gate", 0)?, grid: [(HEADS * total * 32).div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&p)] });
    }
    let mut hashed = vec![(gate.map_or(out, |f| f.gated), total * width * 4), (state, STATE * 4)];
    if end > 0 {
        let (tk, tw, tg) = (STATE, STATE + KEYS * C * D, STATE + KEYS * C * D + HEADS * C * D);
        for kh in 0..KEYS { hashed.push((word(tk + kh * C * D), end * D * 4)); }
        for h in 0..HEADS { hashed.push((word(tw + h * C * D), end * D * 4)); }
        for h in 0..HEADS { hashed.push((word(tg + h * C), end * 4)); }
    }
    Ok(Arm { resets: vec![(state, pristine, CHUNK_STATE * 4)], launches, hashed, layout: 0 })
}

/// Decode in chunk form (kind 12, X arm): flash_next_gdn_chunk_slot over `rows`
/// chats whose chunk-form states hold an open chunk of 4 + 8r rows (host-built:
/// unit key rows, v' in [-0.5, 0.5], descending gamma), so the step reads S plus
/// half a tail on average and never closes a chunk (row 63 cost is separate).
/// `gate` = (z, norm, gated): the slot applies the output gate itself.
fn cslot(g: &mut Gpu, m: H, rows: usize, dev: [u64; 5], state: &[f32], seed: u64, open_all: Option<usize>, gate: Option<(u64, u64, u64)>) -> Result<Arm> {
    let f = function(g, m, "flash_next_gdn_chunk_slot", 0)?;
    let mut r = Rng(seed);
    let (mut states, mut positions, mut resets) = (Vec::new(), Vec::new(), Vec::new());
    for chat in 0..rows {
        let open = open_all.unwrap_or((4 + 8 * chat) % C);
        let mut host = state[chat * STATE..(chat + 1) * STATE].to_vec();
        host.resize(CHUNK_STATE, 0.0);
        let (tk, tw, tg) = (STATE, STATE + KEYS * C * D, STATE + KEYS * C * D + HEADS * C * D);
        for kh in 0..KEYS {
            for j in 0..open {
                let raw: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
                let n = 1. / (raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                for d in 0..D { host[tk + (kh * C + j) * D + d] = raw[d] * n; }
            }
        }
        for h in 0..HEADS {
            let mut acc = 0.0f32;
            for j in 0..open {
                for d in 0..D { host[tw + (h * C + j) * D + d] = r.range(-0.5, 0.5); }
                acc -= r.range(0.001, 1.0);
                host[tg + h * C + j] = acc;
            }
        }
        let pristine = floats(g, &host)?;
        let live = g.alloc(CHUNK_STATE * 4)?;
        resets.push((live, pristine, CHUNK_STATE * 4));
        states.push(live);
        positions.push(10 * C + open);
    }
    let (t, ids) = table(g, &states, &positions)?;
    let out = g.alloc(rows * HEADS * D * 4)?;
    let params = gdn::GdnChunkNext {
        q: dev[0], k: dev[1], v: dev[2], beta: dev[3], g: dev[4], rows: 0, out, a: 0, p: 0, gamma: 0, k0: 0,
        heads: HEADS as u32, keys: KEYS as u32, count: 1, base: 0,
                z: gate.map_or(0, |x| x.0), norm: gate.map_or(0, |x| x.1), gated: gate.map_or(0, |x| x.2), epsilon: if gate.is_some() { 1e-6 } else { 0.0 },
                spare: 0, x: 0, w: 0, history: 0, channels: 0, x_start: 0, ga: 0, gb: 0, alog: 0, dt: 0, pk: 0, pk_tokens: 0, pk_row0: 0,
    };
    let step: [u64; 3] = [ids, t, (rows as u64) << 32];
    let mut hashed = vec![(out, rows * HEADS * D * 4)];
    hashed.extend(states.iter().map(|&s| (s, CHUNK_STATE * 4)));
    Ok(Arm { resets, launches: vec![Launch { f, grid: [HEADS as u32, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&params), blob(&step)] }], hashed, layout: 0 })
}

/// Decode step with its output gate (kind 19): A = flash_next_gdn_chunk_slot then
/// flash_next_gdn_gate_slot (two launches), X = the slot applying the gate itself;
/// each arm from its own module. Non-order: the gated words, the core
/// output and every chat's state must be byte-equal.
fn cslotg(g: &mut Gpu, modules: [H; 2], rows: usize) -> Result<Fixture> {
    use super::params::GdnGate;
    let copies = 8;
    let x = inputs(1, rows);
    let dev = [floats(g, &x.q)?, floats(g, &x.k)?, floats(g, &x.v)?, floats(g, &x.beta)?, floats(g, &x.g)?];
    let mut r = Rng(0x6a7e_1519);
    let z: Vec<f32> = (0..rows * HEADS * D).map(|_| r.range(-4., 4.)).collect();
    let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    let (zd, nd) = (floats(g, &z)?, floats(g, &norm)?);
    let mut arms = [vec![], vec![]];
    for arm in 0..2 {
        for _ in 0..copies {
            let gated = g.alloc(rows * HEADS * D * 4)?;
            let mut a = cslot(g, modules[arm], rows, dev, &x.state, 77, None, if arm == 1 { Some((zd, nd, gated)) } else { None })?;
            if arm == 0 {
                let p = GdnGate { x: a.hashed[0].0 as _, z: zd as _, norm: nd as _, output: gated as _, heads: HEADS as u32, tokens: 1,
                    precision: 0, packed: 0, epsilon: 1e-6, sessions: 0 };
                let step = a.launches[0].params[1].clone();
                a.launches.push(Launch { f: function(g, modules[0], "flash_next_gdn_gate_slot", 0)?, grid: [12, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&p), step] });
            }
            a.hashed.push((gated, rows * HEADS * D * 4));
            arms[arm].push(a);
        }
    }
    let stored = rows * (2 * STATE * 4 + 6 * HEADS * D * 4 + 2 * HEADS * 4) + D * 4;
    Ok(Fixture { arms, stored, flops: rows * HEADS * D * D * 8, copies })
}

/// The whole chunk-form decode step (kind 20): flash_next_gdn_prepare_hist_slot
/// (q/k/v/beta/g + conv history from the projection rows) then
/// flash_next_gdn_chunk_slot applying the output gate, on states holding an open
/// chunk of 4 + 8r rows (as kind 12). Each arm from its own module; X's slot is a
/// programmatic launch when VADE_KBENCH_GDN_PDL=1 (its module must wait). Non-order:
/// q/k/v/beta/g, the core and gated outputs, every state and history byte-equal.
fn cstep(g: &mut Gpu, modules: [H; 2], rows: usize) -> Result<Fixture> {
    use super::params::GdnPrepare;
    const WIDTH: usize = (2 * KEYS + HEADS) * D;
    let copies = 8;
    let mut r = Rng(0x5eed_1996);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let x: Vec<f32> = (0..rows * WIDTH).map(|_| r.range(-2., 2.)).collect();
    let w: Vec<u16> = (0..WIDTH * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let ab: Vec<f32> = (0..2 * rows * HEADS).map(|_| r.range(-3., 3.)).collect();
    let small: Vec<u16> = (0..2 * HEADS).map(|_| bf(r.range(-1., 1.))).collect();
    let hist: Vec<f32> = (0..rows * WIDTH * 4).map(|_| r.range(-2., 2.)).collect();
    let z: Vec<f32> = (0..rows * HEADS * D).map(|_| r.range(-4., 4.)).collect();
    let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let base = inputs(1, rows);
    // Chunk-form states with an open chunk of 4 + 8r rows (unit key rows, v' in
    // [-0.5, 0.5], descending gamma), as cslot builds them.
    let mut hosts = Vec::new();
    for chat in 0..rows {
        let open = (4 + 8 * chat) % C;
        let mut host = base.state[chat * STATE..(chat + 1) * STATE].to_vec();
        host.resize(CHUNK_STATE, 0.0);
        let (tk, tw, tg) = (STATE, STATE + KEYS * C * D, STATE + KEYS * C * D + HEADS * C * D);
        for kh in 0..KEYS {
            for j in 0..open {
                let raw: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
                let n = 1. / (raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                for d in 0..D { host[tk + (kh * C + j) * D + d] = raw[d] * n; }
            }
        }
        for h in 0..HEADS {
            let mut acc = 0.0f32;
            for j in 0..open {
                for d in 0..D { host[tw + (h * C + j) * D + d] = r.range(-0.5, 0.5); }
                acc -= r.range(0.001, 1.0);
                host[tg + h * C + j] = acc;
            }
        }
        hosts.push((host, 10 * C + open));
    }
    let (xd, wd, ad, sd) = (floats(g, &x)?, g.upload(&halves(&w))?, floats(g, &ab)?, g.upload(&halves(&small))?);
    let (zd, nd, hp) = (floats(g, &z)?, floats(g, &norm)?, floats(g, &hist)?);
    let mut arms = [vec![], vec![]];
    for arm in 0..2 {
        let pdl = if arm == 1 { scan_pdl() } else { 0 };
        let prepare = function(g, modules[arm], "flash_next_gdn_prepare_hist_slot", 0)?;
        let slot = function(g, modules[arm], "flash_next_gdn_chunk_slot", 0)?;
        for _ in 0..copies {
            let mut t = vec![0u8; 17 * 1168];
            let (mut resets, mut hashed, mut states) = (vec![], vec![], vec![]);
            let hd = floats(g, &hist)?;
            resets.push((hd, hp, rows * WIDTH * 16));
            for (chat, (host, pos)) in hosts.iter().enumerate() {
                let pristine = floats(g, host)?;
                let live = g.alloc(CHUNK_STATE * 4)?;
                resets.push((live, pristine, CHUNK_STATE * 4));
                states.push(live);
                let at = (chat + 1) * 1168;
                t[at..at + 8].copy_from_slice(&live.to_le_bytes());
                t[at + 8..at + 16].copy_from_slice(&(hd + (chat * WIDTH * 16) as u64).to_le_bytes());
                t[at + 1160..at + 1164].copy_from_slice(&(*pos as u32).to_le_bytes());
            }
            let table = g.upload(&t)?;
            let slots = g.words(&(1..=rows as u32).collect::<Vec<_>>())?;
            let step: [u64; 3] = [slots, table, (rows as u64) << 32];
            let feat = rows * HEADS * D * 4;
            let (q, k, v) = (g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?);
            let (beta, gg) = (g.alloc(rows * HEADS * 4)?, g.alloc(rows * HEADS * 4)?);
            let (out, gated) = (g.alloc(feat)?, g.alloc(feat)?);
            let pp = GdnPrepare {
                x: xd as _, w: wd as _, history: std::ptr::null(), a: ad as _, b: (ad + (rows * HEADS * 4) as u64) as _,
                alog: sd as _, dt: (sd + (HEADS * 2) as u64) as _, q: q as _, k: k as _, v: v as _, beta: beta as _, g: gg as _,
                key_heads: KEYS as u32, value_heads: HEADS as u32, key_start: 0, value_start: 0, tokens: 1, precision: 0, scale: 0,
            };
            let sp = gdn::GdnChunkNext {
                q, k, v, beta, g: gg, rows: 0, out, a: 0, p: 0, gamma: 0, k0: 0,
                heads: HEADS as u32, keys: KEYS as u32, count: 1, base: 0,
                z: zd, norm: nd, gated, epsilon: 1e-6, spare: 0, x: 0, w: 0, history: 0, channels: 0, x_start: 0, ga: 0, gb: 0, alog: 0, dt: 0, pk: 0, pk_tokens: 0, pk_row0: 0,
            };
            let launches = vec![
                Launch { f: prepare, grid: [20, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&pp), blob(&step)] },
                Launch { f: slot, grid: [HEADS as u32, rows as u32, 1], block: 128, shared: pdl, params: vec![blob(&sp), blob(&step)] },
            ];
            hashed.extend([(q, feat), (k, feat), (v, feat), (beta, rows * HEADS * 4), (gg, rows * HEADS * 4), (out, feat), (gated, feat), (hd, rows * WIDTH * 16)]);
            hashed.extend(states.iter().map(|&s| (s, CHUNK_STATE * 4)));
            arms[arm].push(Arm { resets, launches, hashed, layout: 0 });
        }
    }
    let stored = rows * (WIDTH * 4 + 2 * WIDTH * 16 + 2 * STATE * 4 + 6 * HEADS * D * 4 + 4 * HEADS * 4) + WIDTH * 8 + D * 4;
    Ok(Fixture { arms, stored, flops: rows * HEADS * D * D * 8, copies })
}

/// One GDN layer step on one chat (kind 21) as gdn.rs runs it for `rows` rows at chunk
/// offset `open` (a state whose open chunk holds `open` rows; a window past row 63 is
/// cut at the chunk end), `keys` key heads and 3 value heads per key head (8 = the
/// tensor split's rank, 16 = one Spark). A = main's flash_gdn_conv_prepare_tokens_khs,
/// flash_next_gdn_chunk_rows per chunk piece, flash_gdn_history and flash_gdn_gate;
/// X = flash_next_gdn_chunk_rows per chunk piece with its fused prepare and gate, then
/// flash_gdn_history. Non-order: the core and gated rows, the
/// state and the conv history byte-equal.
fn fstep(g: &mut Gpu, modules: [H; 2], rows: usize, open: usize, keys: usize) -> Result<Fixture> {
    use super::params::{GdnGate, GdnHistory, GdnPrepare};
    let (heads, copies) = (3 * keys, 8);
    let (width, hd, floats_state) = ((2 * keys + 3 * keys) * D, 3 * keys * D, gdn::chunk_state_floats(3 * keys, keys));
    let mut r = Rng(0x5eed_2026 + (rows * 64 + open) as u64);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let x: Vec<f32> = (0..rows * width).map(|_| r.range(-2., 2.)).collect();
    let w: Vec<u16> = (0..width * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let ab: Vec<f32> = (0..2 * rows * heads).map(|_| r.range(-3., 3.)).collect();
    let small: Vec<u16> = (0..2 * heads).map(|_| bf(r.range(-1., 1.))).collect();
    let hist: Vec<f32> = (0..width * 4).map(|_| r.range(-2., 2.)).collect();
    let z: Vec<f32> = (0..rows * hd).map(|_| r.range(-4., 4.)).collect();
    let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    // A chunk-form state with an open chunk of `open` rows (unit key rows, v' in
    // [-0.5, 0.5], descending gamma).
    let mut host: Vec<f32> = (0..floats_state).map(|_| r.range(-0.1, 0.1)).collect();
    let (tk, tw, tg) = (heads * D * D, heads * D * D + keys * C * D, heads * D * D + keys * C * D + heads * C * D);
    for kh in 0..keys {
        for j in 0..open {
            let raw: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
            let n = 1. / (raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            for d in 0..D { host[tk + (kh * C + j) * D + d] = raw[d] * n; }
        }
    }
    for h in 0..heads {
        let mut acc = 0.0f32;
        for j in 0..open {
            for d in 0..D { host[tw + (h * C + j) * D + d] = r.range(-0.5, 0.5); }
            acc -= r.range(0.001, 0.2);
            host[tg + h * C + j] = acc;
        }
    }
    let base = 10 * C + open;
    let (xd, wd, ad, sd) = (floats(g, &x)?, g.upload(&halves(&w))?, floats(g, &ab)?, g.upload(&halves(&small))?);
    let (zd, nd, hp, sp) = (floats(g, &z)?, floats(g, &norm)?, floats(g, &hist)?, floats(g, &host)?);
    let bd = ad + (rows * heads * 4) as u64;
    let mut pieces = vec![];
    let mut done = 0;
    while done < rows { let n = (rows - done).min(C - (base + done) % C); pieces.push((done, n)); done += n; }
    let mut arms = [vec![], vec![]];
    #[allow(clippy::needless_range_loop)]
    for arm in 0..2 {
        for _ in 0..copies {
            let (state, history) = (g.alloc(floats_state * 4)?, g.alloc(width * 16)?);
            let (core, gated) = (g.alloc(rows * hd * 4)?, g.alloc(rows * hd * 4)?);
            let mut launches = vec![];
            let mut next_history = 0u64;
            if arm == 0 {
                let (q, k, v) = (g.alloc(rows * hd * 4)?, g.alloc(rows * hd * 4)?, g.alloc(rows * hd * 4)?);
                let (beta, gg) = (g.alloc(rows * heads * 4)?, g.alloc(rows * heads * 4)?);
                let desc = g.upload(&[state.to_le_bytes(), history.to_le_bytes(), 0u64.to_le_bytes()].concat())?;
                let pp = GdnPrepare { x: xd as _, w: wd as _, history: history as _, a: ad as _, b: bd as _, alog: sd as _,
                    dt: (sd + (heads * 2) as u64) as _, q: q as _, k: k as _, v: v as _, beta: beta as _, g: gg as _,
                    key_heads: keys as u32, value_heads: heads as u32, key_start: 0, value_start: 0, tokens: rows as u32, precision: 0, scale: 0 };
                launches.push(Launch { f: function(g, modules[0], "flash_gdn_conv_prepare_tokens_khs", 0)?,
                    grid: [(keys * rows * 3 * 32).div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&pp)] });
                for &(at, n) in &pieces {
                    let (f4, h4) = ((at * hd * 4) as u64, (at * heads * 4) as u64);
                    let c = gdn::GdnChunkNext { q: q + f4, k: k + f4, v: v + f4, beta: beta + h4, g: gg + h4, rows: desc, out: core + f4,
                        a: 0, p: 0, gamma: 0, k0: state, heads: heads as u32, keys: keys as u32, count: n as u32, base: (base + at) as u32,
                        z: 0, norm: 0, gated: 0, epsilon: 1e-6, spare: 0, x: 0, w: 0, history: 0, channels: 0, x_start: 0,
                        ga: 0, gb: 0, alog: 0, dt: 0, pk: 0, pk_tokens: 0, pk_row0: 0 };
                    let (f, shared) = rows_function(g, modules[0])?;
                    launches.push(Launch { f, grid: [heads as u32, 1, 1], block: 128, shared, params: vec![blob(&c)] });
                }
                let hp = GdnHistory { x: xd as _, history: history as _, rows: 0, channels: width as u32, tokens: rows as u32, precision: 0 };
                launches.push(Launch { f: function(g, modules[0], "flash_gdn_history", 0)?, grid: [width.div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&hp)] });
                let gp = GdnGate { x: core as _, z: zd as _, norm: nd as _, output: gated as _, heads: (heads * rows) as u32, tokens: 1,
                    precision: 0, packed: 0, epsilon: 1e-6, sessions: 0 };
                launches.push(Launch { f: function(g, modules[0], "flash_gdn_gate", 0)?, grid: [(heads * rows * 32).div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&gp)] });
            } else {
                // X: flash_next_gdn_chunk_rows per chunk piece with the fused prepare (ga) and
                // gate (gated), then flash_gdn_history once, as gdn.rs launches small windows.
                let (q, k, v) = (g.alloc(rows * hd * 4)?, g.alloc(rows * hd * 4)?, g.alloc(rows * hd * 4)?);
                let (beta, gg) = (g.alloc(rows * heads * 4)?, g.alloc(rows * heads * 4)?);
                let desc = g.upload(&[state.to_le_bytes(), history.to_le_bytes(), 0u64.to_le_bytes()].concat())?;
                for &(at, n) in &pieces {
                    let (f4, h4) = ((at * hd * 4) as u64, (at * heads * 4) as u64);
                    let c = gdn::GdnChunkNext { q: q + f4, k: k + f4, v: v + f4, beta: beta + h4, g: gg + h4, rows: desc, out: core + f4,
                        a: 0, p: 0, gamma: 0, k0: state, heads: heads as u32, keys: keys as u32, count: n as u32, base: (base + at) as u32,
                        z: zd + f4, norm: nd, gated: gated + f4, epsilon: 1e-6, spare: 0, x: xd, w: wd, history, channels: width as u32, x_start: at as u32,
                        ga: ad + h4, gb: bd + h4, alog: sd, dt: sd + (heads * 2) as u64, pk: 0, pk_tokens: 0, pk_row0: 0 };
                    let (f, shared) = rows_function(g, modules[1])?;
                    launches.push(Launch { f, grid: [heads as u32, 1, 1], block: 128, shared, params: vec![blob(&c)] });
                }
                // The X history out of place, as a one-chat shallow verify runs it: the old
                // words stay in `history` (bytes A == X on the new words at `next`).
                let (to, target) = match g.function(modules[1], "flash_gdn_history_to") {
                    Ok(_) => ("flash_gdn_history_to", g.alloc(width * 16)?),
                    Err(_) => ("flash_gdn_history", 0),
                };
                next_history = target;
                let hp = GdnHistory { x: xd as _, history: history as _, rows: target, channels: width as u32, tokens: rows as u32, precision: 0 };
                launches.push(Launch { f: function(g, modules[1], to, 0)?, grid: [width.div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&hp)] });
            }
            let resets = vec![(state, sp, floats_state * 4), (history, hp, width * 16)];
            let written = if next_history != 0 { next_history } else { history };
            let hashed = vec![(core, rows * hd * 4), (gated, rows * hd * 4), (state, floats_state * 4), (written, width * 16)];
            arms[arm].push(Arm { resets, launches, hashed, layout: 0 });
        }
    }
    // As stored: x, a/b, z rows; the state's S read (plus the closing step's write) and its
    // open tail rows; history read + write; core and gated rows written.
    let stored = rows * (width * 4 + 2 * heads * 4 + 3 * hd * 4) + heads * D * D * 4 + (keys + heads) * open * D * 4 + width * 32 + D * 4;
    Ok(Fixture { arms, stored, flops: rows * heads * D * D * 4, copies })
}

/// One cohort verify GDN layer step (kind 22) as gdn_cohort.rs launches it: `chats` chats of
/// `rows` rows each at chunk offset `open` (a window past row 63 is cut at the chunk end),
/// 16 key heads and 48 value heads (one Spark), each launch with the chat in grid-Y and its
/// parameters in a table. A = main's flash_gdn_conv_prepare_tokens_kh_cohort,
/// flash_next_gdn_chunk_rows_cohort per chunk piece, flash_gdn_history_cohort; X =
/// flash_next_gdn_chunk_rows_cohort per chunk piece with its fused prepare (ga, gate deferred),
/// flash_gdn_history_cohort. Both then gate every row in one flash_gdn_gate (Gdn::gate_rows).
/// Non-order: the core and gated rows, every chat's state and conv history byte-equal.
fn cfstep(g: &mut Gpu, modules: [H; 2], rows: usize, open: usize, chats: usize) -> Result<Fixture> {
    use super::params::{GdnGate, GdnHistory, GdnPrepare};
    let (keys, copies) = (16usize, 4);
    let heads = 3 * keys;
    let (width, hd, floats_state) = ((2 * keys + heads) * D, heads * D, gdn::chunk_state_floats(heads, keys));
    let all = chats * rows;
    let mut r = Rng(0x5eed_2027 + (all * 64 + open) as u64);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let table = |g: &mut Gpu, records: Vec<Vec<u64>>| g.upload(&records.concat().iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
    let x: Vec<f32> = (0..all * width).map(|_| r.range(-2., 2.)).collect();
    let w: Vec<u16> = (0..width * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let ab: Vec<f32> = (0..2 * all * heads).map(|_| r.range(-3., 3.)).collect();
    let small: Vec<u16> = (0..2 * heads).map(|_| bf(r.range(-1., 1.))).collect();
    let z: Vec<f32> = (0..all * hd).map(|_| r.range(-4., 4.)).collect();
    let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    // Per chat: a conv history and a chunk-form state with an open chunk of `open` rows (unit
    // key rows, v' in [-0.5, 0.5], descending gamma), as fstep.
    let (tk, tw, tg) = (heads * D * D, heads * D * D + keys * C * D, heads * D * D + keys * C * D + heads * C * D);
    let mut pristine = vec![];
    for _ in 0..chats {
        let hist: Vec<f32> = (0..width * 4).map(|_| r.range(-2., 2.)).collect();
        let mut host: Vec<f32> = (0..floats_state).map(|_| r.range(-0.1, 0.1)).collect();
        for kh in 0..keys {
            for j in 0..open {
                let raw: Vec<f32> = (0..D).map(|_| r.range(-1., 1.)).collect();
                let n = 1. / (raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                for d in 0..D { host[tk + (kh * C + j) * D + d] = raw[d] * n; }
            }
        }
        for h in 0..heads {
            let mut acc = 0.0f32;
            for j in 0..open {
                for d in 0..D { host[tw + (h * C + j) * D + d] = r.range(-0.5, 0.5); }
                acc -= r.range(0.001, 0.2);
                host[tg + h * C + j] = acc;
            }
        }
        pristine.push((floats(g, &host)?, floats(g, &hist)?));
    }
    let base = 10 * C + open;
    let (xd, wd, ad, sd) = (floats(g, &x)?, g.upload(&halves(&w))?, floats(g, &ab)?, g.upload(&halves(&small))?);
    let (zd, nd) = (floats(g, &z)?, floats(g, &norm)?);
    let bd = ad + (all * heads * 4) as u64;
    let mut pieces = vec![];
    let mut done = 0;
    while done < rows { let n = (rows - done).min(C - (base + done) % C); pieces.push((done, n)); done += n; }
    let mut arms = [vec![], vec![]];
    for arm in 0..2 {
        let m = modules[arm];
        for _ in 0..copies {
            let chat: Vec<(u64, u64)> = (0..chats).map(|_| Ok((g.alloc(floats_state * 4)?, g.alloc(width * 16)?))).collect::<Result<_>>()?;
            let (core, gated) = (g.alloc(all * hd * 4)?, g.alloc(all * hd * 4)?);
            let (q, k, v) = (g.alloc(all * hd * 4)?, g.alloc(all * hd * 4)?, g.alloc(all * hd * 4)?);
            let (beta, gg) = (g.alloc(all * heads * 4)?, g.alloc(all * heads * 4)?);
            let mut descs = vec![];
            for &(state, history) in &chat { descs.push(g.upload(&[state.to_le_bytes(), history.to_le_bytes(), 0u64.to_le_bytes()].concat())?); }
            let mut launches = vec![];
            if arm == 0 {
                let records = (0..chats).map(|c| {
                    let (f4, h4, x4) = ((c * rows * hd * 4) as u64, (c * rows * heads * 4) as u64, (c * rows * width * 4) as u64);
                    blob(&GdnPrepare { x: (xd + x4) as _, w: wd as _, history: chat[c].1 as _, a: (ad + h4) as _, b: (bd + h4) as _, alog: sd as _,
                        dt: (sd + (heads * 2) as u64) as _, q: (q + f4) as _, k: (k + f4) as _, v: (v + f4) as _, beta: (beta + h4) as _, g: (gg + h4) as _,
                        key_heads: keys as u32, value_heads: heads as u32, key_start: 0, value_start: 0, tokens: rows as u32, precision: 0, scale: 0 })
                }).collect();
                let t = table(g, records)?;
                launches.push(Launch { f: function(g, m, "flash_gdn_conv_prepare_tokens_kh_cohort", 0)?,
                    grid: [(keys * rows * 32).div_ceil(128) as u32, chats as u32, 1], block: 128, shared: 0, params: vec![vec![t]] });
            }
            for &(at, n) in &pieces {
                let records = (0..chats).map(|c| {
                    let (f4, h4) = (((c * rows + at) * hd * 4) as u64, ((c * rows + at) * heads * 4) as u64);
                    let fused = arm == 1;
                    blob(&gdn::GdnChunkNext { q: q + f4, k: k + f4, v: v + f4, beta: beta + h4, g: gg + h4, rows: descs[c], out: core + f4,
                        a: 0, p: 0, gamma: 0, k0: chat[c].0, heads: heads as u32, keys: keys as u32, count: n as u32, base: (base + at) as u32,
                        z: 0, norm: 0, gated: 0, epsilon: 1e-6, spare: 0,
                        x: if fused { xd + (c * rows * width * 4) as u64 } else { 0 }, w: if fused { wd } else { 0 }, history: if fused { chat[c].1 } else { 0 },
                        channels: if fused { width as u32 } else { 0 }, x_start: if fused { at as u32 } else { 0 },
                        ga: if fused { ad + h4 } else { 0 }, gb: if fused { bd + h4 } else { 0 }, alog: if fused { sd } else { 0 },
                        dt: if fused { sd + (heads * 2) as u64 } else { 0 }, pk: 0, pk_tokens: 0, pk_row0: 0 })
                }).collect();
                let t = table(g, records)?;
                launches.push(Launch { f: function(g, m, "flash_next_gdn_chunk_rows_cohort", 0)?, grid: [heads as u32, chats as u32, 1], block: 128, shared: 0, params: vec![vec![t]] });
            }
            let records = (0..chats).map(|c| blob(&GdnHistory { x: (xd + (c * rows * width * 4) as u64) as _, history: chat[c].1 as _, rows: 0,
                channels: width as u32, tokens: rows as u32, precision: 0 })).collect();
            let t = table(g, records)?;
            launches.push(Launch { f: function(g, m, "flash_gdn_history_cohort", 0)?, grid: [width.div_ceil(128) as u32, chats as u32, 1], block: 128, shared: 0, params: vec![vec![t]] });
            let gp = GdnGate { x: core as _, z: zd as _, norm: nd as _, output: gated as _, heads: (heads * all) as u32, tokens: 1,
                precision: 0, packed: 0, epsilon: 1e-6, sessions: 0 };
            launches.push(Launch { f: function(g, m, "flash_gdn_gate", 0)?, grid: [(heads * all * 32).div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&gp)] });
            let mut resets = vec![];
            let mut hashed = vec![(core, all * hd * 4), (gated, all * hd * 4)];
            for (&(state, history), &(sp, hp)) in chat.iter().zip(&pristine) {
                resets.extend([(state, sp, floats_state * 4), (history, hp, width * 16)]);
                hashed.extend([(state, floats_state * 4), (history, width * 16)]);
            }
            arms[arm].push(Arm { resets, launches, hashed, layout: 0 });
        }
    }
    // As fstep, per chat.
    let stored = chats * (rows * (width * 4 + 2 * heads * 4 + 3 * hd * 4) + heads * D * D * 4 + (keys + heads) * open * D * 4 + width * 32) + D * 4;
    Ok(Fixture { arms, stored, flops: all * heads * D * D * 4, copies })
}

/// EXACTNESS LAW boundary check (kind 11): the same rows cut into calls at
/// 63/64/65, 127/128/129, 191/192/193, uneven pieces and one-row (decode) calls
/// must give the same output and state bytes as one call. A = main over all rows.
fn splits(g: &mut Gpu, modules: [H; 2], c: &Case, started: Instant) -> Result<()> {
    let total = c.m;
    let x = inputs(total, 1);
    let width = HEADS * D;
    let dev = [floats(g, &x.q)?, floats(g, &x.k)?, floats(g, &x.v)?, floats(g, &x.beta)?, floats(g, &x.g)?];
    let base = pristine(g, &x.state)?;
    let w = work(g, total / C + 2)?;
    // (name, cuts, decode rows): rows in `decode` run one at a time through the
    // decode slot kernel, everything else through prompt calls cut at `cuts`.
    let mut cuts: Vec<(String, Vec<usize>, std::ops::Range<usize>)> = vec![("one call".into(), vec![], 0..0)];
    for list in [vec![63], vec![64], vec![65], vec![127, 128, 129], vec![191, 192, 193], vec![1, 2, 3],
                 vec![5, 75, 76, 136, 166, 199], (63..=70).collect::<Vec<_>>(), (1..=70).collect::<Vec<_>>()] {
        let name = if list.len() > 8 { format!("rows 1..{} one at a time", list.len()) } else { format!("cuts {list:?}") };
        cuts.push((name, list, 0..0));
    }
    cuts.push(("prompt 0..63 + decode 63..70 + prompt".into(), vec![], 63..70));
    cuts.push(("decode 0..70 + prompt".into(), vec![], 0..70));
    cuts.push(("prompt 0..127 + decode 127..131 + prompt".into(), vec![], 127..131));
    cuts.push(("prompt 0..200 + decode 200..263".into(), vec![], 200..263));
    let mut plans: Vec<(String, Vec<(usize, usize, u8)>)> = Vec::new();
    for (name, list, decode) in &cuts {
        let mut calls = Vec::new();
        let mut at = 0;
        let mut ends: Vec<usize> = list.clone();
        ends.extend(decode.clone());
        ends.push(decode.end);
        ends.push(total);
        ends.sort();
        for cut in ends {
            if cut > at && cut <= total { calls.push((at, cut - at, if decode.contains(&at) { DECODE } else { PROMPT })); at = cut; }
        }
        plans.push((name.clone(), calls));
    }
    // MTP verify windows of k + 1 = 4 rows across the 64-row boundary: accepted windows are prompt calls; a rejected window runs, the
    // state is restored, and the accepted rows are replayed one row per call
    // (the engine's replay), then decoding continues. All must equal one call.
    let (p, d, r) = (PROMPT, DECODE, REJECT);
    let rest = |at: usize| (at, total - at, p);
    let mut verify = vec![
        ("verify 62..66 accepted", vec![(0, 62, p), (62, 4, p), rest(66)]),
        ("verify 63..67 accepted", vec![(0, 63, p), (63, 4, p), rest(67)]),
        ("verify 64..68 accepted", vec![(0, 64, p), (64, 4, p), rest(68)]),
        ("single steps 62..68 (decode slot)", vec![(0, 62, p), (62, 1, d), (63, 1, d), (64, 1, d), (65, 1, d), (66, 1, d), (67, 1, d), rest(68)]),
        ("verify 62..66 rejected, keep 1 (replay 62), verify 63..67 accepted", vec![(0, 62, p), (62, 4, r), (62, 1, p), (63, 4, p), rest(67)]),
        ("verify 63..67 rejected, keep 2 (replay 63, 64), decode 65", vec![(0, 63, p), (63, 4, r), (63, 1, p), (64, 1, p), (65, 1, d), rest(66)]),
        ("verify 64..68 rejected, keep 0, decode 64, 65", vec![(0, 64, p), (64, 4, r), (64, 1, d), (65, 1, d), rest(66)]),
        ("verify 62..66 rejected, keep 3 (replay 62..64), verify 65..69 accepted", vec![(0, 62, p), (62, 4, r), (62, 1, p), (63, 1, p), (64, 1, p), (65, 4, p), rest(69)]),
        ("verify 61..65 keep 3, verify 64..68 keep 1, decode 65", vec![(0, 61, p), (61, 4, r), (61, 1, p), (62, 1, p), (63, 1, p), (64, 4, r), (64, 1, p), (65, 1, d), rest(66)]),
        ("verify windows 120..136 accepted (crosses 128)", vec![(0, 120, p), (120, 4, p), (124, 4, p), (128, 4, p), (132, 4, p), rest(136)]),
    ];
    for (name, calls) in verify.drain(..) {
        if calls.iter().all(|c| c.0 + c.1 <= total) { plans.push((name.into(), calls)); }
    }
    let mut shas = Vec::new();
    let mut first = Vec::new();
    for (name, calls) in &plans {
        let out = g.alloc(total * width * 4)?;
        let mut arm = chunk(g, modules[1], dev, calls, base, out, &w, true)?;
        let bytes = run_arm(g, &mut arm)?;
        let sha = sha256::digest(&bytes);
        eprintln!("gdn split case={} {name}: calls={} sha={}", c.name, calls.len(), &sha[..16]);
        if first.is_empty() { first = bytes; }
        shas.push(sha);
    }
    let equal = shas.iter().filter(|s| **s == shas[0]).count();
    let f = function(g, modules[0], "flash_next_gdn_scan", 0)?;
    let state_in = floats(g, &x.state[..STATE])?;
    let out = g.alloc(total * width * 4)?;
    let mut a = sequential(g, f, dev, total, 1, state_in, out)?;
    let abytes = run_arm(g, &mut a)?;
    let to = |b: &[u8]| -> Vec<f32> { b.as_chunks::<4>().0.iter().map(|&w| f32::from_le_bytes(w)).collect() };
    let n = total * width;
    let (om, orel, od) = errors(&to(&abytes[..n * 4]), &to(&first[..n * 4]));
    let (sm, srel, _) = errors(&to(&abytes[n * 4..]), &to(&first[n * 4..(n + STATE) * 4]));
    eprintln!("gdn split case={} equal {equal}/{} out rel_l2={orel:.3e} state rel_l2={srel:.3e} elapsed_s={:.1}", c.name, shas.len(), started.elapsed().as_secs_f64());
    println!("{{\"case\":{:?},\"arm\":\"A\",\"sha256\":{:?},\"splits\":0}}", c.name, sha256::digest(&abytes));
    println!("{{\"case\":{:?},\"arm\":\"X\",\"sha256\":{:?},\"splits\":{},\"splits_equal\":{equal},\"bytes_equal\":false,\"order_change\":true,\"out_max_abs_err\":{om:.3e},\"out_rel_l2\":{orel:.3e},\"out_words_differ\":{od},\"state_max_abs_err\":{sm:.3e},\"state_rel_l2\":{srel:.3e}}}", c.name, shas[0], shas.len());
    g.clear()?;
    if equal != shas.len() { return Err(format!("{} splits differ: {equal}/{} equal", c.name, shas.len())); }
    Ok(())
}

/// Decode prepare (kind 13): A = main's two launches,
/// flash_next_gdn_prepare_slot (4 x rows CTAs) then flash_next_gdn_history_slot
/// (80 x rows); X = the fused flash_next_gdn_prepare_hist_slot (20 x rows) when
/// the X module has it, else the same two launches. Non-order: the bytes of q,
/// k, v, beta, g and every chat's updated history must equal A's. Each ring copy
/// has its own inputs, weights and histories (cold in L2 like production).
fn prep(g: &mut Gpu, modules: [H; 2], rows: usize) -> Result<Fixture> {
    use super::params::{GdnHistory, GdnPrepare};
    const WIDTH: usize = (2 * KEYS + HEADS) * D;
    let copies = if rows == 1 { 96 } else { 16 };
    let mut r = Rng(0x5eed_1419);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let x: Vec<f32> = (0..rows * WIDTH).map(|_| r.range(-2., 2.)).collect();
    let w: Vec<u16> = (0..WIDTH * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let ab: Vec<f32> = (0..2 * rows * HEADS).map(|_| r.range(-3., 3.)).collect();
    let small: Vec<u16> = (0..2 * HEADS).map(|_| bf(r.range(-1., 1.))).collect();
    let hist: Vec<f32> = (0..rows * WIDTH * 4).map(|_| r.range(-2., 2.)).collect();
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let pristine = floats(g, &hist)?;
    let fused = g.function(modules[1], "flash_next_gdn_prepare_hist_slot").ok();
    let mut arms = [vec![], vec![]];
    for _ in 0..copies {
        let (xd, wd, ad) = (floats(g, &x)?, g.upload(&halves(&w))?, floats(g, &ab)?);
        let sd = g.upload(&halves(&small))?;
        for arm in 0..2 {
            let hd = floats(g, &hist)?;
            let mut t = vec![0u8; 17 * 1168];
            for row in 0..rows {
                let at = (row + 1) * 1168 + 8;
                t[at..at + 8].copy_from_slice(&(hd + (row * WIDTH * 16) as u64).to_le_bytes());
            }
            let table = g.upload(&t)?;
            let slots = g.words(&(1..=rows as u32).collect::<Vec<_>>())?;
            let feat = rows * HEADS * D * 4;
            let (q, k, v) = (g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?);
            let (beta, gg) = (g.alloc(rows * HEADS * 4)?, g.alloc(rows * HEADS * 4)?);
            let p = GdnPrepare {
                x: xd as _, w: wd as _, history: std::ptr::null(), a: ad as _, b: (ad + (rows * HEADS * 4) as u64) as _,
                alog: sd as _, dt: (sd + (HEADS * 2) as u64) as _, q: q as _, k: k as _, v: v as _, beta: beta as _, g: gg as _,
                key_heads: KEYS as u32, value_heads: HEADS as u32, key_start: 0, value_start: 0, tokens: 1, precision: 0, scale: 0,
            };
            let h = GdnHistory { x: xd as _, history: null_mut(), rows: 0, channels: WIDTH as u32, tokens: 1, precision: 0 };
            let step: [u64; 3] = [slots, table, (rows as u64) << 32];
            let launches = match (arm, fused) {
                (1, Some(f)) => vec![Launch { f, grid: [(WIDTH / 128 / 4) as u32, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&p), blob(&step)] }],
                _ => vec![
                    Launch { f: function(g, modules[arm], "flash_next_gdn_prepare_slot", 0)?, grid: [4, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&p), blob(&step)] },
                    Launch { f: function(g, modules[arm], "flash_next_gdn_history_slot", 0)?, grid: [80, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&h), blob(&step)] },
                ],
            };
            let hashed = vec![(q, feat), (k, feat), (v, feat), (beta, rows * HEADS * 4), (gg, rows * HEADS * 4), (hd, rows * WIDTH * 16)];
            arms[arm].push(Arm { resets: vec![(hd, pristine, rows * WIDTH * 16)], launches, hashed, layout: 0 });
        }
    }
    let stored = rows * (WIDTH * 4 + 2 * WIDTH * 16 + 3 * HEADS * D * 4 + 4 * HEADS * 4) + WIDTH * 8 + HEADS * 4;
    Ok(Fixture { arms, stored, flops: rows * WIDTH * 12, copies })
}

/// Decode gate (kind 14): flash_next_gdn_gate_slot (12 x rows CTAs) from each
/// module: the RMS norm of the core output times sigmoid(z). Non-order: the
/// gated output bytes must equal A's.
fn gate(g: &mut Gpu, modules: [H; 2], rows: usize) -> Result<Fixture> {
    use super::params::GdnGate;
    let copies = 8;
    let width = HEADS * D;
    let mut r = Rng(0x6a7e_1505);
    let core: Vec<f32> = (0..rows * width).map(|_| r.range(-1., 1.)).collect();
    let z: Vec<f32> = (0..rows * width).map(|_| r.range(-4., 4.)).collect();
    let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    let mut arms = [vec![], vec![]];
    for _ in 0..copies {
        let (cd, zd, nd) = (floats(g, &core)?, floats(g, &z)?, floats(g, &norm)?);
        let slots = g.words(&(1..=rows as u32).collect::<Vec<_>>())?;
        for arm in 0..2 {
            let out = g.alloc(rows * width * 4)?;
            let p = GdnGate { x: cd as _, z: zd as _, norm: nd as _, output: out as _, heads: HEADS as u32, tokens: 1,
                precision: 0, packed: 0, epsilon: 1e-6, sessions: 0 };
            let step: [u64; 3] = [slots, slots, (rows as u64) << 32];
            let f = function(g, modules[arm], "flash_next_gdn_gate_slot", 0)?;
            arms[arm].push(Arm { resets: vec![], launches: vec![Launch { f, grid: [12, rows as u32, 1], block: 128, shared: 0, params: vec![blob(&p), blob(&step)] }],
                hashed: vec![(out, rows * width * 4)], layout: 0 });
        }
    }
    Ok(Fixture { arms, stored: rows * width * 12 + D * 4, flops: rows * width * 8, copies })
}

/// Prompt prepare (kind 15): main's flash_gdn_conv_prepare_tokens over `tokens`
/// rows (A) vs flash_gdn_conv_prepare_tokens_kh (X, when the module has it,
/// else the same kernel): q/k are compared on each key group's first value
/// head only (layout 2; the chunk kernels read no other q/k row), v, beta and
/// g in full. Non-order: the bytes must be equal. Floor = main's stored bytes.
fn cprep(g: &mut Gpu, modules: [H; 2], tokens: usize) -> Result<Fixture> {
    use super::params::GdnPrepare;
    const WIDTH: usize = (2 * KEYS + HEADS) * D;
    let copies = 2;
    let mut r = Rng(0xc0de_2048);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let x: Vec<f32> = (0..tokens * WIDTH).map(|_| r.range(-2., 2.)).collect();
    let w: Vec<u16> = (0..WIDTH * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let ab: Vec<f32> = (0..2 * tokens * HEADS).map(|_| r.range(-3., 3.)).collect();
    let small: Vec<u16> = (0..2 * HEADS).map(|_| bf(r.range(-1., 1.))).collect();
    let hist: Vec<f32> = (0..WIDTH * 4).map(|_| r.range(-2., 2.)).collect();
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let kh = g.function(modules[1], "flash_gdn_conv_prepare_tokens_kh").ok();
    let mut arms = [vec![], vec![]];
    for _ in 0..copies {
        let (xd, wd, ad, hd) = (floats(g, &x)?, g.upload(&halves(&w))?, floats(g, &ab)?, floats(g, &hist)?);
        let sd = g.upload(&halves(&small))?;
        for arm in 0..2 {
            let feat = tokens * HEADS * D * 4;
            let (q, k, v) = (g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?);
            let (beta, gg) = (g.alloc(tokens * HEADS * 4)?, g.alloc(tokens * HEADS * 4)?);
            let p = GdnPrepare {
                x: xd as _, w: wd as _, history: hd as _, a: ad as _, b: (ad + (tokens * HEADS * 4) as u64) as _,
                alog: sd as _, dt: (sd + (HEADS * 2) as u64) as _, q: q as _, k: k as _, v: v as _, beta: beta as _, g: gg as _,
                key_heads: KEYS as u32, value_heads: HEADS as u32, key_start: 0, value_start: 0, tokens: tokens as u32, precision: 0, scale: 0,
            };
            let f = match (arm, kh) { (1, Some(f)) => f, _ => function(g, modules[arm], "flash_gdn_conv_prepare_tokens", 0)? };
            let blocks = (KEYS * tokens * 32).div_ceil(128) as u32;
            arms[arm].push(Arm { resets: vec![], launches: vec![Launch { f, grid: [blocks, 1, 1], block: 128, shared: 0, params: vec![blob(&p)] }],
                hashed: vec![(q, feat), (k, feat), (v, feat), (beta, tokens * HEADS * 4), (gg, tokens * HEADS * 4)], layout: 2 });
        }
    }
    let stored = tokens * (WIDTH * 4 + 3 * HEADS * D * 4 + 4 * HEADS * 4) + WIDTH * 24 + HEADS * 4;
    Ok(Fixture { arms, stored, flops: tokens * WIDTH * 12, copies })
}

/// Kind 17: the chunk-form prompt pipeline on real conv inputs, X module both
/// arms: A = `flash_gdn_conv_prepare_tokens_kh` (writes v) plus per piece chunk prep
/// plus scan reading v (gate fused); X = the prepare without v plus the scan
/// convolving its value rows from the projection rows. Non-order: bytes equal.
fn cpipe(g: &mut Gpu, modules: [H; 2], tokens: usize, pieces: usize, packed: bool) -> Result<Fixture> {
    use super::params::GdnPrepare;
    const WIDTH: usize = (2 * KEYS + HEADS) * D;
    let n = tokens * pieces;
    let copies = 2;
    let mut r = Rng(0xc0de_0017);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let x: Vec<f32> = (0..n * WIDTH).map(|_| r.range(-2., 2.)).collect();
    let w: Vec<u16> = (0..WIDTH * 4).map(|_| bf(r.range(-0.6, 0.6))).collect();
    let ab: Vec<f32> = (0..2 * n * HEADS).map(|_| r.range(-3., 3.)).collect();
    let small: Vec<u16> = (0..2 * HEADS).map(|_| bf(r.range(-1., 1.))).collect();
    let hist: Vec<f32> = (0..WIDTH * 4).map(|_| r.range(-2., 2.)).collect();
    let z: Vec<f32> = (0..n * HEADS * D).map(|_| r.range(-4., 4.)).collect();
    let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
    let s0: Vec<f32> = (0..STATE).map(|_| r.range(-0.1, 0.1)).collect();
    let halves = |v: &[u16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let kh = function(g, modules[1], "flash_gdn_conv_prepare_tokens_kh", 0)?;
    let mut arms = [vec![], vec![]];
    for _ in 0..copies {
        let (xd, wd, ad, hd) = (floats(g, &x)?, g.upload(&halves(&w))?, floats(g, &ab)?, floats(g, &hist)?);
        let sd = g.upload(&halves(&small))?;
        let (zd, nd) = (floats(g, &z)?, floats(g, &norm)?);
        let base = pristine(g, &s0)?;
        #[allow(clippy::needless_range_loop)]
        for arm in 0..2 {
            let feat = n * HEADS * D * 4;
            let (q, k, v) = (g.alloc(feat)?, g.alloc(feat)?, g.alloc(feat)?);
            let (beta, gg) = (g.alloc(n * HEADS * 4)?, g.alloc(n * HEADS * 4)?);
            let p = GdnPrepare {
                x: xd as _, w: wd as _, history: hd as _, a: ad as _, b: (ad + (n * HEADS * 4) as u64) as _,
                alog: sd as _, dt: (sd + (HEADS * 2) as u64) as _, q: q as _, k: k as _,
                v: if arm == 0 { v as _ } else { core::ptr::null_mut() }, beta: beta as _, g: gg as _,
                key_heads: KEYS as u32, value_heads: HEADS as u32, key_start: 0, value_start: 0, tokens: n as u32, precision: 0, scale: 0,
            };
            let blocks = (KEYS * n * 32).div_ceil(128) as u32;
            let calls: Vec<(usize, usize, u8)> = (0..pieces).map(|c| (c * tokens, tokens, PROMPT)).collect();
            let wk = work(g, tokens / C + 2)?;
            let out = g.alloc(feat)?;
            // X: prep convolves q/k and computes the gates itself (no prepare launch) when
            // VADE_KBENCH_GDN_PREP_FUSED is not 0 (the module must include the fused preparation entry).
            // Kind 18 (packed): both arms run the fused pipeline; A packs the FP32
            // gated output with flash_bf16_input, X writes the packed words itself.
            let fused_prep = (arm == 1 || packed) && std::env::var("VADE_KBENCH_GDN_PREP_FUSED").map_or(true, |v| v != "0");
            let gates = if fused_prep { [ad, ad + (n * HEADS * 4) as u64, sd, sd + (HEADS * 2) as u64] } else { [0; 4] };
            let words = n.div_ceil(16) * (HEADS * D / 16) * 128;
            let pkbuf = if packed { g.alloc(words * super::params::TERMS * 4)? } else { 0 };
            let gated = g.alloc(feat)?;
            let gate = Gate { z: zd, norm: nd, gated, fused: true, conv: if arm == 1 || packed { [xd, wd, hd] } else { [0; 3] }, gates,
                pk: if packed && arm == 1 { (pkbuf, n) } else { (0, 0) } };
            let mut a = chunk_gated(g, modules[1], [q, k, v, beta, gg], &calls, base, out, &wk, Some(gate), true)?;
            if !fused_prep { a.launches.insert(0, Launch { f: kh, grid: [blocks, 1, 1], block: 128, shared: 0, params: vec![blob(&p)] }); }
            if packed {
                if arm == 0 {
                    let pp = super::params::FlashBf16Input { input: gated, output: pkbuf, tokens: n as u32, cols: (HEADS * D) as u32, hc_silu: 0, lanes: 32 };
                    a.launches.push(Launch { f: function(g, modules[1], "flash_bf16_input", 0)?, grid: [(words / 128 * 32).div_ceil(128) as u32, 1, 1], block: 128, shared: 0, params: vec![blob(&pp)] });
                }
                a.hashed[0] = (pkbuf, words * super::params::TERMS * 4);
            }
            a.hashed.push((beta, n * HEADS * 4));
            a.hashed.push((gg, n * HEADS * 4));
            arms[arm].push(a);
        }
    }
    let stored = n * (WIDTH * 4 + 2 * HEADS * D * 4 + 2 * HEADS * 4) + pieces * 2 * STATE * 4;
    Ok(Fixture { arms, stored, flops: n * HEADS * D * D * 8, copies })
}

fn fixture(g: &mut Gpu, modules: [H; 2], c: &Case) -> Result<Fixture> {
    if c.kind == 23 { return chain::fixture(g, modules, c.k, c.m); }
    if c.kind == 13 { return prep(g, modules, c.k); }
    if c.kind == 17 { return cpipe(g, modules, c.m, c.n, false); }
    if c.kind == 18 { return cpipe(g, modules, c.m, c.n, true); }
    if c.kind == 14 { return gate(g, modules, c.k); }
    if c.kind == 15 { return cprep(g, modules, c.m); }
    if c.kind == 19 { return cslotg(g, modules, c.k); }
    if c.kind == 20 { return cstep(g, modules, c.k); }
    if c.kind == 21 { return fstep(g, modules, c.m, c.n, c.k); }
    if c.kind == 22 { return cfstep(g, modules, c.m, c.n, c.k); }
    let (tokens, pieces, rows) = (c.m, c.n, c.k);
    let x = inputs(tokens * pieces, rows);
    let width = HEADS * D;
    if c.kind == 9 || c.kind == 12 {
        let copies = 8;
        let dev = [floats(g, &x.q)?, floats(g, &x.k)?, floats(g, &x.v)?, floats(g, &x.beta)?, floats(g, &x.g)?];
        let mut arms = [vec![], vec![]];
        for arm in 0..2 {
            let f = function(g, modules[arm], "flash_next_gdn_scan_slot", 0)?;
            for _ in 0..copies {
                let work = if arm == 1 && c.kind == 12 { cslot(g, modules[arm], rows, dev, &x.state, 77, if c.n > 1 { Some(c.n - 2) } else { None }, None)? } else { slot(g, f, rows, dev, &x.state)? };
                arms[arm].push(work);
            }
        }
        let stored = rows * (2 * STATE * 4 + 4 * width * 4 + 2 * HEADS * 4);
        return Ok(Fixture { arms, stored, flops: rows * HEADS * D * D * 8, copies });
    }
    let copies = 2;
    let mut arms = [vec![], vec![]];
    for _ in 0..copies {
        let dev = [floats(g, &x.q)?, floats(g, &x.k)?, floats(g, &x.v)?, floats(g, &x.beta)?, floats(g, &x.g)?];
        let state_in = floats(g, &x.state[..STATE])?;
        for arm in 0..2 {
            let out = g.alloc(tokens * pieces * width * 4)?;
            let work = if (arm == 0 && c.kind != 16) || c.kind == 7 {
                let f = function(g, modules[arm], "flash_next_gdn_scan", 0)?;
                sequential(g, f, dev, tokens, pieces, state_in, out)?
            } else if c.kind == 10 {
                fla0(g, modules[arm], &x, tokens, state_in)?
            } else if c.kind == 16 {
                // A = the unfused chunk path (scan writes o, then flash_gdn_gate),
                // X = the scan with the fused gate; both from the X module.
                let calls: Vec<(usize, usize, u8)> = (0..pieces).map(|p| (p * tokens, tokens, PROMPT)).collect();
                let base = pristine(g, &x.state)?;
                let w = work(g, tokens / C + 2)?;
                let n = tokens * pieces;
                let mut r = Rng(0x2a7e_0016);
                let z: Vec<f32> = (0..n * width).map(|_| r.range(-4., 4.)).collect();
                let norm: Vec<f32> = (0..D).map(|_| r.range(0.5, 1.5)).collect();
                let gate = Gate { z: floats(g, &z)?, norm: floats(g, &norm)?, gated: g.alloc(n * width * 4)?, fused: arm == 1, conv: [0; 3], gates: [0; 4], pk: (0, 0) };
                chunk_gated(g, modules[1], dev, &calls, base, out, &w, Some(gate), true)?
            } else {
                let calls: Vec<(usize, usize, u8)> = (0..pieces).map(|p| (p * tokens, tokens, PROMPT)).collect();
                let base = pristine(g, &x.state)?;
                let w = work(g, tokens / C + 2)?;
                chunk(g, modules[arm], dev, &calls, base, out, &w, arm == 1)?
            };
            arms[arm].push(work);
        }
    }
    let n = tokens * pieces;
    // As stored: q, k, v, out [n][48][128] FP32, beta/g [n][48], state read+write per piece.
    let stored = 4 * n * width * 4 + 2 * n * HEADS * 4 + pieces * (2 * STATE * 4 + 24);
    Ok(Fixture { arms, stored, flops: n * HEADS * D * D * 8, copies })
}

fn run_arm(g: &Gpu, arm: &mut Arm) -> Result<Vec<u8>> {
    for &(p, bytes) in &arm.hashed {
        unsafe { check((g.d.memset)(p, 0xff, bytes))? };
    }
    for &(dst, src, bytes) in &arm.resets {
        unsafe { check((g.d.copy)(dst, src, bytes))? };
    }
    g.sync()?;
    for l in arm.launches.iter_mut() { run_launch(g, l)?; }
    g.sync()?;
    let mut bytes = Vec::new();
    for (idx, &(p, n)) in arm.hashed.iter().enumerate() {
        let read = g.read(p, n)?;
        if arm.layout == 2 && idx < 2 {
            // q/k: only each key group's first value head (head = row % 48).
            for (row, cells) in read.as_chunks::<{ D * 4 }>().0.iter().enumerate() {
                if (row % HEADS).is_multiple_of(HEADS / KEYS) { bytes.extend_from_slice(cells); }
            }
        } else {
            bytes.extend(read);
        }
    }
    if !bytes.as_chunks::<4>().0.iter().all(|&b| f32::from_le_bytes(b).is_finite()) {
        return Err("GDN nonfinite or unwritten output".into());
    }
    Ok(bytes)
}
fn canonical(bytes: &[u8], layout: u8, tokens: usize) -> Vec<f32> {
    let f: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
    if layout == 0 { return f; }
    let mut out = f.clone();
    for h in 0..HEADS {
        for t in 0..tokens {
            for d in 0..D { out[(t * HEADS + h) * D + d] = f[(h * tokens + t) * D + d]; }
        }
    }
    out
}
fn errors(a: &[f32], x: &[f32]) -> (f64, f64, usize) {
    let (mut max, mut num, mut den, mut differ) = (0f64, 0f64, 0f64, 0usize);
    for (&p, &q) in a.iter().zip(x) {
        let d = (p as f64 - q as f64).abs();
        max = max.max(d);
        num += d * d;
        den += (p as f64) * (p as f64);
        differ += usize::from(p.to_bits() != q.to_bits());
    }
    (max, (num / den.max(1e-30)).sqrt(), differ)
}

fn capture<'a>(g: &'a Gpu, timer: H, times: u64, arms: &mut [Arm]) -> Result<Graph<'a>> {
    let (mut raw, mut exec, mut start, mut end) = (null_mut(), null_mut(), null_mut(), null_mut());
    unsafe { check((g.d.capture)(g.s, 0))? };
    let mut t = times;
    g.launch(timer, [1, 1, 1], 1, 0, &mut [super::ptr(&mut t)])?;
    for arm in arms.iter_mut() {
        for l in arm.launches.iter_mut() { run_launch(g, l)?; }
    }
    t = times + 8;
    g.launch(timer, [1, 1, 1], 1, 0, &mut [super::ptr(&mut t)])?;
    unsafe {
        check((g.d.capture_end)(g.s, &mut raw))?;
        check((g.d.instantiate)(&mut exec, raw, 0))?;
        check((g.d.event)(&mut start, 0))?;
        check((g.d.event)(&mut end, 0))?;
    }
    Ok(Graph { g, raw, exec, start, end })
}

pub(super) fn run_case(g: &mut Gpu, modules: [H; 2], timer: H, c: &Case, o: &Options, hashes: &[String; 2], source: &str, started: Instant) -> Result<()> {
    if c.kind == 11 { return splits(g, modules, c, started); }
    let order = c.kind == 8 || c.kind == 10 || c.kind == 12;
    let mut v = fixture(g, modules, c)?;
    let times = g.alloc(16)?;
    eprintln!("gdn case={} copies={} launches A={} X={} elapsed_s={:.1}", c.name, v.copies, v.arms[0][0].launches.len(), v.arms[1][0].launches.len(), started.elapsed().as_secs_f64());
    let mut digest = Vec::new();
    let mut first = Vec::new();
    for arm in 0..2 {
        let mut all = Vec::new();
        for copy in 0..v.copies {
            let wall = Instant::now();
            let bytes = run_arm(g, &mut v.arms[arm][copy])?;
            eprintln!("gdn check case={} arm={} copy={} wall_ms={:.3}", c.name, arm, copy, wall.elapsed().as_secs_f64() * 1e3);
            all.push(sha256::digest(&bytes));
            if copy == 0 { first.push(bytes); }
        }
        if all.iter().any(|h| h != &all[0]) {
            return Err(format!("{} arm {} repeats differ", c.name, arm));
        }
        digest.push(all[0].clone());
    }
    let equal = digest[0] == digest[1];
    if !order && !equal {
        // The first differing FP32 word of the hashed regions, in hashed order.
        let word = first[0].as_chunks::<4>().0.iter().zip(first[1].as_chunks::<4>().0).position(|(a, x)| a != x);
        return Err(format!("{} bytes A==X: no A={} X={} first differing word {word:?} of regions {:?}", c.name, digest[0], digest[1],
            v.arms[0][0].hashed.iter().map(|h| h.1 / 4).collect::<Vec<_>>()));
    }
    let tokens = c.m * c.n;
    let out_len = tokens * c.k.max(1) * HEADS * D;
    let mut error = String::new();
    // kind 12 runs X on different (chunk-form) states: no error vs A.
    if order && c.kind != 12 {
        let a = canonical(&first[0][..out_len * 4], v.arms[0][0].layout, tokens);
        let x = canonical(&first[1][..out_len * 4], v.arms[1][0].layout, tokens);
        let (om, orel, od) = errors(&a, &x);
        let sa: Vec<f32> = first[0][out_len * 4..].as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
        let sx: Vec<f32> = first[1][out_len * 4..].as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
        let (sm, srel, sd) = errors(&sa, &sx);
        error = format!(",\"out_max_abs_err\":{om:.3e},\"out_rel_l2\":{orel:.3e},\"out_words_differ\":{od},\"state_max_abs_err\":{sm:.3e},\"state_rel_l2\":{srel:.3e},\"state_words_differ\":{sd}");
        eprintln!("gdn order case={} out max_abs={om:.3e} rel_l2={orel:.3e} differ={od}/{} state max_abs={sm:.3e} rel_l2={srel:.3e}", c.name, a.len());
    }
    if std::env::var("VADE_KBENCH_GDN_CHECK_ONLY").is_ok_and(|v| v == "1") {
        for (arm, d) in digest.iter().enumerate() { println!("{{\"case\":{:?},\"arm\":{:?},\"sha256\":{:?},\"bytes_equal\":{},\"order_change\":{}{}}}", c.name, if arm == 0 { "A" } else { "X" }, d, equal, order, if arm == 1 { error.as_str() } else { "" }); }
        g.clear()?;
        return Ok(());
    }
    let [a_arms, x_arms] = &mut v.arms;
    let ga = capture(g, timer, times, a_arms)?;
    let gx = capture(g, timer, times, x_arms)?;
    let warm = Instant::now();
    while warm.elapsed().as_millis() < 250 {
        sample(&ga, times)?;
        sample(&gx, times)?;
    }
    let mut events = [vec![], vec![]];
    let mut global = [vec![], vec![]];
    let count = v.copies as f64;
    while events[0].len() < 9 || events.iter().any(|e| e.iter().sum::<f64>() < o.ms as f64 * 1000.) {
        let order = if events[0].len() % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let (event, gt) = sample(if arm == 0 { &ga } else { &gx }, times)?;
            events[arm].push(event);
            global[arm].push(gt);
        }
    }
    for arm in 0..2 {
        let e = events[arm].iter().map(|x| x / count).collect::<Vec<_>>();
        let t = global[arm].iter().map(|x| x / count).collect::<Vec<_>>();
        let mid = median(&e);
        let avg = e.iter().sum::<f64>() / e.len() as f64;
        let cv = (e.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / e.len() as f64).sqrt() / avg * 100.;
        let floor = v.stored as f64 / 230e3;
        let pct = floor / mid * 100.;
        let launches = v.arms[arm][0].launches.len();
        println!(
            "{{\"case\":{:?},\"kernel\":{:?},\"arm\":{:?},\"m\":{},\"n\":{},\"k\":{},\"samples\":{},\"launches_per_problem\":{},\"ring_copies\":{},\"stored_bytes\":{},\"floor_bytes_basis\":{:?},\"flops\":{},\"median_us\":{:.6},\"min_us\":{:.6},\"max_us\":{:.6},\"cv_pct\":{:.6},\"globaltimer_us\":{:.6},\"gbps\":{:.6},\"floor_us\":{:.6},\"pct_floor\":{:.3},\"grade\":{:?},\"sha256\":{:?},\"bytes_equal\":{},\"order_change\":{}{},\"ptx_sha256\":{:?},\"bench_source_sha256\":{:?}}}",
            c.name, if arm == 0 && order { "flash_next_gdn_scan" } else { c.kernel }, if arm == 0 { "A" } else { "X" },
            c.m, c.n, c.k, e.len(), launches, v.copies, v.stored, "FP32 q/k/v/output + beta/g + state read/write per piece + metadata", v.flops, mid,
            e.iter().copied().fold(f64::INFINITY, f64::min), e.iter().copied().fold(0., f64::max), cv, median(&t),
            v.stored as f64 / mid / 1000., floor, pct, grade(pct), digest[arm], equal, order,
            if arm == 1 { error.as_str() } else { "" }, hashes[arm], source
        );
    }
    drop(ga);
    drop(gx);
    g.clear()?;
    Ok(())
}
