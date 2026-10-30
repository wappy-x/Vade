//! Host bounds proof for flash_next_gdn_chunk_rows with its fused prepare and gate
//! (gdn.rs small windows: at most CHUNK_ROWS_LIMIT rows, cut at the chunk end). It
//! walks the kernel's loops for every CTA (value head h) and thread (value column
//! tid), forms each global word index and shared byte offset as the kernel does, and
//! checks it against the buffer sizes gdn.rs and kernel-bench give the launch:
//!   x `[rows][channels]`, w BF16 `[channels][4]`, history `[channels][4]`;
//!   ga/gb, beta/g `[rows][heads]`; A_log/dt_bias `[heads]`; norm `[128]`;
//!   q/k/v, out, z, gated `[rows][heads][128]`; the chunk-form state (chunk_state_floats);
//!   static shared CHUNK_ROW_BYTES.
//! Row indices are absolute in the conv call: piece p starts at row x_start.
//! A write that two CTAs make to one word, or that one CTA makes to a word another
//! CTA reads in the same launch, is a failure too (q/k slots, tail rows).
use std::collections::HashMap;

const D: usize = 128;
const C: usize = 64;
/// Static shared bytes of the rows kernel: kd `[64][128]` (stride 65 as `[128][65]`),
/// q_i, k_i `[128]`, A_ij, P_ij, f_j `[64]` (gdn_params::CHUNK_ROW_BYTES).
const SLOT_KD: usize = 0;
const SLOT_Q: usize = D * (C + 1) * 4;
const SLOT_K: usize = SLOT_Q + D * 4;
const SLOT_A: usize = SLOT_K + D * 4;
const SLOT_P: usize = SLOT_A + C * 4;
const SLOT_F: usize = SLOT_P + C * 4;
const SLOT_BYTES: usize = SLOT_F + C * 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Buf { X, W, History, Ga, Gb, Alog, Dt, Q, K, V, Beta, G, Out, Z, Gated, Norm, State }
/// Bounds, plus the CTAs (bit masks) that write and read each word a launch writes.
/// Inputs the launch never writes (x, w, history, ga/gb, A_log/dt, z, norm) and the
/// state words of one head (S, v' tail, gamma) get only the bounds check; the tail
/// key rows (one block per key group, range `tk`) are tracked.
struct Walk { lens: [usize; 17], tk: (usize, usize), errors: Vec<String>, writes: HashMap<(Buf, usize), u64>, reads: HashMap<(Buf, usize), u64> }
impl Walk {
    fn word(&mut self, b: Buf, index: isize, cta: usize, write: bool) {
        let len = self.lens[b as usize];
        if index < 0 || index as usize >= len {
            if self.errors.len() < 8 { self.errors.push(format!("{b:?}[{index}] outside {len} words (CTA {cta}, {})", if write { "write" } else { "read" })); }
            return;
        }
        let input = matches!(b, Buf::X | Buf::W | Buf::History | Buf::Ga | Buf::Gb | Buf::Alog | Buf::Dt | Buf::Z | Buf::Norm);
        if input || b == Buf::State && !(self.tk.0..self.tk.1).contains(&(index as usize)) { return; }
        let map = if write { &mut self.writes } else { &mut self.reads };
        *map.entry((b, index as usize)).or_insert(0) |= 1u64 << cta;
    }
    fn shared(&mut self, byte: usize) {
        if byte + 4 > SLOT_BYTES && self.errors.len() < 8 { self.errors.push(format!("shared byte {byte} outside {SLOT_BYTES}")); }
    }
}

/// One launch: `n` rows at absolute position `base`, rows x_start.. of the conv call;
/// `heads` value heads in `keys` groups. Mirrors chunk_rows' loops.
fn launch(w: &mut Walk, heads: usize, keys: usize, n: usize, base: usize, x_start: usize, gated: bool) {
    let (repeat, hd, channels) = (heads / keys, heads * D, (2 * keys + heads) * D);
    let (tk_at, tw_at, tg_at) = (heads * D * D, heads * D * D + keys * C * D, heads * D * D + keys * C * D + heads * C * D);
    let state = |w: &mut Walk, at: usize, cta: usize, write: bool| w.word(Buf::State, at as isize, cta, write);
    // Piece-relative row r of this launch is conv-call row x_start + r.
    let row = |r: usize| x_start + r;
    for h in 0..heads {
        let (kh, h0) = (h / repeat, h / repeat * repeat);
        let (tk, tw, tg, s) = (tk_at + kh * C * D, tw_at + h * C * D, tg_at + h * C, h * D * D);
        for tid in 0..D {
            for d in 0..D { state(w, s + d * D + tid, h, false); }
            let mut idx = tid;
            while idx < base % C * 32 {
                let (j, d4) = (idx / 32, idx % 32 * 4);
                for x in 0..4 { state(w, tk + j * D + d4 + x, h, false); w.shared(SLOT_KD + ((d4 + x) * (C + 1) + j) * 4); }
                idx += 128;
            }
        }
        // Fused prepare: warp wp takes rows wp, wp + 4; lane l features l + 32 f.
        for wp in 0..4 {
            let mut r = wp;
            while r < n {
                let token = row(r);
                for lane in 0..32 {
                    for f in 0..4 {
                        for at in [kh * D, (keys + kh) * D, (2 * keys + h) * D] {
                            let c = at + lane + 32 * f;
                            for tap in 0..4 {
                                let source = token as isize + tap as isize - 3;
                                if source >= 0 { w.word(Buf::X, source * channels as isize + c as isize, h, false); }
                                else { w.word(Buf::History, (c * 4) as isize + source + 4, h, false); }
                                w.word(Buf::W, (c * 4 + tap) as isize, h, false);
                            }
                        }
                        let o = (token * hd + h * D + lane + 32 * f) as isize;
                        w.word(Buf::Q, o, h, true); w.word(Buf::K, o, h, true); w.word(Buf::V, o, h, true);
                    }
                }
                let gate = (token * heads + h) as isize;
                w.word(Buf::Ga, gate, h, false); w.word(Buf::Gb, gate, h, false);
                w.word(Buf::Alog, h as isize, h, false); w.word(Buf::Dt, h as isize, h, false);
                w.word(Buf::Beta, gate, h, true); w.word(Buf::G, gate, h, true);
                r += 4;
            }
        }
        // The rows: the fused launch reads q/k from its own slot h, not h0.
        let qh = h;
        for r in 0..n {
            let (pos, at) = (base + r, row(r) * hd);
            let (i, start) = (pos % C, pos - pos % C);
            let gate = (row(r) * heads + h) as isize;
            for tid in 0..D {
                w.word(Buf::Beta, gate, h, false); w.word(Buf::G, gate, h, false);
                if i > 0 { state(w, tg + i - 1, h, false); }
                w.word(Buf::Q, (at + qh * D + tid) as isize, h, false); w.word(Buf::K, (at + qh * D + tid) as isize, h, false);
                w.shared(SLOT_Q + tid * 4); w.shared(SLOT_K + tid * 4);
                if tid < i {
                    for d in 0..D { w.shared(SLOT_KD + (d * (C + 1) + tid) * 4); w.shared(SLOT_K + d * 4); }
                    state(w, tg + tid, h, false);
                    w.shared(SLOT_A + tid * 4); w.shared(SLOT_P + tid * 4);
                } else if tid == i { w.shared(SLOT_P + tid * 4); }
                w.word(Buf::V, (at + h * D + tid) as isize, h, false);
                for j in 0..i { state(w, tw + j * D + tid, h, false); w.shared(SLOT_A + j * 4); w.shared(SLOT_P + j * 4); }
                w.shared(SLOT_P + i * 4);
                w.word(Buf::Out, (at + h * D + tid) as isize, h, true);
                if i + 1 < C {
                    state(w, tw + i * D + tid, h, true);
                    if tid == 0 { state(w, tg + i, h, true); }
                    if h == h0 { state(w, tk + i * D + tid, h, true); }
                    w.shared(SLOT_KD + (tid * (C + 1) + i) * 4);
                    continue;
                }
                for d in 0..D { state(w, s + d * D + tid, h, true); }
                if tid < C { if tid + 1 < C { state(w, tg + tid, h, false); } w.shared(SLOT_F + tid * 4); }
                let mut idx = tid;
                while idx < C * D {
                    let (j, d) = (idx / D, idx % D);
                    if j + 1 < C {
                        if start + j >= base { w.word(Buf::K, (row(start + j - base) * hd + qh * D + d) as isize, h, false); }
                        else { state(w, tk + j * D + d, h, false); }
                    }
                    w.shared(SLOT_KD + idx * 4); w.shared(SLOT_F + j * 4);
                    idx += 128;
                }
                for j in 0..C - 1 { state(w, tw + j * D + tid, h, false); }
                for d in 0..D { state(w, s + d * D + tid, h, false); state(w, s + d * D + tid, h, true); }
            }
        }
        if gated {
            for wp in 0..4 {
                let mut r = wp;
                while r < n {
                    for f in 0..D {
                        let o = (row(r) * hd + h * D + f) as isize;
                        w.word(Buf::Out, o, h, false); w.word(Buf::Z, o, h, false); w.word(Buf::Gated, o, h, true);
                        w.word(Buf::Norm, f as isize, h, false);
                    }
                    r += 4;
                }
            }
        }
    }
}

/// Buffer sizes in words, indexed by Buf.
fn sizes(list: [(Buf, usize); 17]) -> [usize; 17] {
    let mut out = [0; 17];
    for (b, len) in list { out[b as usize] = len; }
    out
}

/// gdn.rs' small-window call: `rows` rows from absolute position `base`, cut at the
/// chunk end into at most two launches. Returns the walk's failures.
pub fn call(heads: usize, keys: usize, rows: usize, base: usize, gated: bool) -> Vec<String> { call_sized(heads, keys, rows, base, gated, None) }
/// call with one buffer one word short (the walk must report it).
fn call_sized(heads: usize, keys: usize, rows: usize, base: usize, gated: bool, short: Option<Buf>) -> Vec<String> {
    let (hd, channels) = (heads * D, (2 * keys + heads) * D);
    let state = heads * D * D + keys * C * D + heads * C * D + heads * C;
    let lens = sizes([(Buf::X, rows * channels), (Buf::W, channels * 4), (Buf::History, channels * 4), (Buf::Ga, rows * heads),
        (Buf::Gb, rows * heads), (Buf::Alog, heads), (Buf::Dt, heads), (Buf::Q, rows * hd), (Buf::K, rows * hd), (Buf::V, rows * hd),
        (Buf::Beta, rows * heads), (Buf::G, rows * heads), (Buf::Out, rows * hd), (Buf::Z, rows * hd), (Buf::Gated, rows * hd),
        (Buf::Norm, D), (Buf::State, state)]);
    let mut lens = lens;
    if let Some(b) = short { lens[b as usize] -= 1; }
    let mut errors = Vec::new();
    let first = rows.min(C - base % C);
    for (x_start, n) in [(0, first), (first, rows - first)] {
        if n == 0 { continue; }
        let mut w = Walk { lens, tk: (heads * D * D, heads * D * D + keys * C * D), errors: vec![], writes: HashMap::new(), reads: HashMap::new() };
        launch(&mut w, heads, keys, n, base + x_start, x_start, gated);
        // Two CTAs that write one word, or a word one CTA writes and another reads in
        // the same launch, would race.
        let mut races: Vec<String> = w.writes.iter().filter_map(|(at, &writers)| {
            let readers = w.reads.get(at).copied().unwrap_or(0);
            (writers.count_ones() > 1 || readers & !writers != 0).then(|| format!("{:?}[{}] writers {writers:#x} readers {readers:#x}", at.0, at.1))
        }).collect();
        races.sort();
        w.errors.extend(races.into_iter().take(4));
        errors.extend(w.errors.into_iter().map(|e| format!("launch rows {x_start}..{}: {e}", x_start + n)));
    }
    errors
}

/// Every chunk offset with windows of 1..=8 rows for the tensor split's rank (8 key
/// heads) and one Spark (16), plus group sizes 1 and 2. Returns (cases, failures).
pub fn run() -> (usize, usize) {
    let (mut cases, mut failures) = (0, 0);
    for (heads, keys) in [(24, 8), (48, 16), (8, 8), (16, 8)] {
        for open in 0..C {
            for rows in 1..=8 {
                cases += 1;
                let errors = call(heads, keys, rows, 10 * C + open, true);
                if !errors.is_empty() { failures += 1; println!("gdn-rows-bounds heads {heads} keys {keys} open {open} rows {rows}: {}", errors.join("; ")); }
            }
        }
    }
    println!("gdn-rows-bounds: {cases} cases, {failures} failures");
    (cases, failures)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn screen_shapes_stay_in_bounds() {
        // kernel-bench gdn_fstep: (rows, open, keys).
        for (rows, open, keys) in [(1, 4, 8), (1, 31, 8), (1, 63, 8), (4, 30, 8), (4, 62, 8), (8, 60, 8), (1, 31, 16), (4, 62, 16),
            (1, 0, 8), (2, 31, 8), (3, 31, 8), (8, 56, 8)] {
            assert_eq!(call(3 * keys, keys, rows, 10 * C + open, true), Vec::<String>::new(), "rows {rows} open {open} keys {keys}");
        }
    }
    #[test]
    fn every_offset_stays_in_bounds() {
        // Decode rows at every offset; windows that end at, close or cross the chunk end.
        // (kernel-bench --gdn-rows-bounds walks every offset with 1..=8 rows.)
        for open in 0..C { assert_eq!(call(24, 8, 1, C + open, true), Vec::<String>::new(), "open {open}"); }
        for (rows, open) in [(8, 56), (8, 57), (2, 62), (2, 63), (3, 63)] {
            assert_eq!(call(24, 8, rows, C + open, rows != 2), Vec::<String>::new(), "open {open} rows {rows}");
        }
    }
    #[test]
    fn a_short_buffer_is_reported() {
        // (Not State: no launch touches gamma of tail row 63, the state's last word.)
        for b in [Buf::X, Buf::Q, Buf::Beta, Buf::Gated, Buf::Norm, Buf::History, Buf::Dt] {
            assert!(!call_sized(24, 8, 4, C + 62, true, Some(b)).is_empty(), "{b:?}");
        }
    }
}
