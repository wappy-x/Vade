//! Tensor-core QSA attention scores, values and index scores with a fixed
//! accumulation order. ORDER CHANGE behind the host switch VADE_QSA_TC (default on; setting it to 0 selects
//! the FP32 entries in flash_qsa.rs). Same parameters, launch roles and
//! [rows, stride] layouts as the flash_qsa entries they replace; the softmax
//! between scores and values is main's (flash_qsa_attention_softmax).
//!
//! Precision. Every FP32 operand x (query and selected key features) is split
//! into BF16 planes by round-to-nearest: h = RN(x), m = RN(x - h),
//! l = RN(x - h - m). The residuals are exact in FP32 and, for normal-range x
//! (|x| >= 2^-100), x = h + m + l exactly (3 x 8 significant bits). PLANES = 3
//! sums the six products of order >= 2^-16 (hh, hm, mh, hl, mm, lh); the
//! dropped ml, lm, ll terms are together below 2^-23 |q_i k_i|, the size of
//! one FP32 rounding of that product, so no operand bit is lost against
//! main's FP32 math. PLANES = 2 keeps only hh, hm, mh of the hi/lo split
//! (residual up to 2^-16 |x|): about 3 x 2^-16 per product, eight bits fewer
//! than FP32 (a benchmark screen, never the default).
//!
//! Order. One score's arithmetic depends only on D = 256: warp w of four owns
//! features [64w, 64w + 64) as four K16 steps; lane (g = lane / 4,
//! t = lane % 4) feeds features 4t .. 4t + 3 of a step as MMA k = 2t, 2t + 1,
//! 2t + 8, 2t + 9 (one fixed permutation shared by Q and K). Per warp: a big
//! chain of the four hh products and a small chain of the others in
//! (step, product) order, both from 0, then partial = big + small; the CTA
//! adds ((p0 + p1) + (p2 + p3)) and scales by 1/16. Every score is computed
//! this way at every length, launch shape, chunking, batch and cache state:
//! no atomics, no split over keys, padded rows never touch real rows.
//!
//! Layout. One CTA = one token x one group of twelve query heads (one KV
//! head) x a range of selected slots: range = 16 * ceil(ceil(2051 / grid_y)
//! / 16), so grid_y only distributes work. Heads fill MMA rows 0..11, rows
//! 12..15 are zero. Keys stream in 16-slot tiles of FP32 rows gathered with
//! cp.async (double buffered, 16-byte chunks XOR-swizzled by row parity);
//! each warp splits its 64-feature slice of a row in registers.
//!
//! REDUCTION-PLAN (rule 13:25; every VADE_QSA_TC entry; boundaries follow from
//! the head dims alone, never from length, batch, chunking, cache or load).
//! MMA: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (exact BF16 x BF16
//! products, FP32 accumulate); FP32 operands split into BF16 planes h, m, l;
//! six products hh, hm, mh, hl, mm, lh (PLANES = 3, the only switch value).
//! 1. Attention scores S = q . k / 16, D = 256 (_tc, _tc_w, _tc_batch).
//!    Pieces: feature blocks B_w = [64w, 64w + 64), w < 4, each four K16 steps
//!    [64w + 16s, + 16) (D only). Per piece: big = hh(s = 0..3) from 0; small =
//!    (s = 0..3: hm, mh, hl, mm, lh) from 0; p_w = big + small (RN).
//!    Merge tree TREE4: S = ((p0 + p1) + (p2 + p3)) * (1/16). No split over
//!    slots; CTAs, grid y and warp tiles only distribute slots.
//! 2. Attention values out = (sum_s P[s] V[s][f]) / d * gate (_tc, _tc_m and
//!    the _batch twins). Pieces: 16-slot k-steps
//!    [16k, 16k + 16) of a row's visible slots from slot 0 (last one zero
//!    padded; slot 4t + i = MMA k 2t, 2t + 1, 2t + 8, 2t + 9). Per piece: one
//!    fresh accumulator from 0: lh, mm, hl, mh, hm, hh. Merge SEQ: out =
//!    (((0 + step0) + step1) + ...) ascending k (RN), then / d, * gate. Feature
//!    halves / slices / tiles only distribute outputs. P and d are main's
//!    softmax outputs (flash_qsa_attention_softmax), read as stored.
//! 3. Index S[t][b] = sum_h relu(q_th . k_b) / sqrt(128), D = 128 (flash_qsa_scores_tc,
//!    _tc_m, _tc_batch). Pieces per (score, head): eight K16 steps [16s, + 16);
//!    big = hh(s = 0..7) from 0; small = (s: hm, mh, hl, mm, lh) from 0;
//!    dot = big + small. Merge HEADS4: ((((0 + r0) + r1) + r2) + r3), r_h =
//!    relu(dot_h), then / sqrt(128) (main's head order). Token / group tiles
//!    (64 or 16 tokens x 32 groups) only distribute scores.
//! 4. Fused prompt attention (flash_qsa_attention_fused_tc, >= 64-token prompt
//!    launches): no new pieces. Scores = plan 1 (TREE4) over [0, visible);
//!    softmax = main's flash_qsa_attention_softmax: highest = max over visible
//!    slots (exact in any order), P = exp(S - highest) (layer::exp, the same
//!    ex2.approx.ftz of (x * log2 e)), d = SEQ (((0 + P0) + P1) + ...) ascending
//!    slots, carried lane to lane (padded slots and rows add exp(-inf) = +0,
//!    exact); values = plan 2 with those P and d. Output bytes == scores _tc_w +
//!    softmax + values _tc (kbench kind 12 checks SHA equality per case).
//! Single-block case (one visible slot, one group, one token) runs the same
//! pieces and merges; no atomics; padded rows and slots never feed real ones.
//! Guards (review 13:58). (a) _tc / _tc_batch do not zero-fill tile slots at
//! or past `end`: an MMA output column depends only on its own B column and
//! columns >= end are never stored, so stale K cannot reach a stored score
//! (a softmax fused over the tile would need the zero fill first); _tc_w
//! loads zero K registers for those slots. (b) Every launched KV head has a
//! non-empty query group by shape: plan.kv_heads = grouped(query_heads, 12)
//! (model/src/flash/plan.rs) lists exactly the KV heads that own a local
//! query head, so group = end - first >= 1 at every world size; this is why
//! main's unguarded `group` line, copied by flash_qsa_attention_values_wide,
//! is unreachable. The TC entries return early anyway.
use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_y, _block_idx_z, _grid_dim_y, _thread_idx_x}};
use crate::flash_qsa::{batch_attention, batch_index, batch_row, output_gate, selected_length, selected_position, stage_ids, tile_position, ID_TILE};
use crate::kernel_params::{QsaAttention, QsaBatch, QsaIndex, QSA_MAX_POSITIONS, QSA_SCORE_HEADS};
use crate::ptx::{commit, copy4, copy16, load, load2, load4, shared, store, sync, wait};
use crate::{layer::exp, shuffle_xor};

const TILE: usize = 16;
const ROW: usize = 1024;
const STAGE: usize = TILE * ROW;

macro_rules! two { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body }
}; }
macro_rules! three { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body } { const $i: usize = 2; $body }
}; }
macro_rules! four { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body }
    { const $i: usize = 2; $body } { const $i: usize = 3; $body }
}; }
macro_rules! eight { ($i:ident, $body:block) => {
    four!($i, $body);
    { const $i: usize = 4; $body } { const $i: usize = 5; $body }
    { const $i: usize = 6; $body } { const $i: usize = 7; $body }
}; }

/// Two RN BF16 conversions, `lo` in the low half (MMA's lower k).
#[inline(always)]
unsafe fn pair(lo: f32, hi: f32) -> u32 {
    let d: u32;
    asm!("cvt.rn.bf16x2.f32 {d}, {hi}, {lo};", d = out(reg32) d, hi = in(reg32) hi, lo = in(reg32) lo,
        options(pure, nomem, nostack));
    d
}
/// BF16 planes [h, m, l] of two adjacent operands (low half = a).
#[inline(always)]
unsafe fn split(a: f32, b: f32) -> [u32; 3] {
    let h = pair(a, b);
    let (ra, rb) = (a - f32::from_bits(h << 16), b - f32::from_bits(h & 0xffff_0000));
    let m = pair(ra, rb);
    let (sa, sb) = (ra - f32::from_bits(m << 16), rb - f32::from_bits(m & 0xffff_0000));
    [h, m, pair(sa, sb)]
}
#[inline(always)]
unsafe fn mma(a: [u32; 4], b0: u32, b1: u32, c: [f32; 4]) -> [f32; 4] {
    let (w, x, y, z): (f32, f32, f32, f32);
    asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{w},{x},{y},{z}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
        w = out(reg32) w, x = out(reg32) x, y = out(reg32) y, z = out(reg32) z,
        a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
        b0 = in(reg32) b0, b1 = in(reg32) b1,
        c0 = in(reg32) c[0], c1 = in(reg32) c[1], c2 = in(reg32) c[2], c3 = in(reg32) c[3]);
    [w, x, y, z]
}
#[inline(always)]
unsafe fn global4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];", a = out(reg32) a, b = out(reg32) b,
        c = out(reg32) c, d = out(reg32) d, p = in(reg64) p);
    [a, b, c, d]
}
/// global4 without L1 allocation (L2 only): the fused kernel's V row pieces are
/// read once per warp, so they should not evict the S lines all warps re-read
/// (K stays on global4: L2-only K loads lost 16 % at 32k).
#[inline(always)]
unsafe fn global4_cg(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    asm!("ld.global.cg.v4.f32 {{{a},{b},{c},{d}}}, [{p}];", a = out(reg32) a, b = out(reg32) b,
        c = out(reg32) c, d = out(reg32) d, p = in(reg64) p);
    [a, b, c, d]
}
#[inline(always)]
unsafe fn store2(p: u32, a: f32, b: f32) {
    asm!("st.shared.v2.f32 [{p}], {{{a}, {b}}};", p = in(reg32) p, a = in(reg32) a, b = in(reg32) b);
}

#[inline(always)]
unsafe fn zero16(p: u32) {
    asm!("st.shared.v4.b32 [{p}], {{0, 0, 0, 0}};", p = in(reg32) p);
}

#[inline(always)]
unsafe fn scores<const PLANES: usize>(p: QsaAttention) {
    const HEADS: usize = QSA_SCORE_HEADS;
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let groups = (p.query_start as usize % HEADS + p.query_heads as usize).div_ceil(HEADS);
    let group = _block_idx_x() as usize;
    if group >= groups * p.tokens as usize { return; }
    let token = group / groups;
    let first = (p.query_start as usize / HEADS + group % groups) * HEADS;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let range = QSA_MAX_POSITIONS.div_ceil(_grid_dim_y() as usize).div_ceil(TILE) * TILE;
    let begin = _block_idx_y() as usize * range;
    if begin >= visible { return; }
    let end = (begin + range).min(visible);
    // HEADS divides queries_per_key, so even a partial local group shares K.
    let key_head = first / p.queries_per_key as usize - p.key_start as usize;
    let (qs, qe) = (p.query_start as usize, (p.query_start + p.query_heads) as usize);
    let row_of = |r: usize| if r < HEADS && first + r >= qs && first + r < qe {
        token * p.query_heads as usize + first + r - qs } else { usize::MAX };

    // A fragments of this warp's four K16 steps, three planes each.
    let (ra, rb) = (row_of(g), row_of(g + 8));
    let query = p.query as *const f32;
    let mut qf = [[[0u32; 4]; 3]; 4];
    four!(J, {
        let d = 64 * warp + 16 * J + 4 * t;
        let xa = if ra != usize::MAX { global4(query.add(ra * 256 + d)) } else { [0.0; 4] };
        let xb = if rb != usize::MAX { global4(query.add(rb * 256 + d)) } else { [0.0; 4] };
        let (a01, a23, b01, b23) = (split(xa[0], xa[1]), split(xa[2], xa[3]), split(xb[0], xb[1]), split(xb[2], xb[3]));
        three!(P, { qf[J][P] = [a01[P], b01[P], a23[P], b23[P]]; });
    });

    // Copy role: rows (tid / 64) + 2m of a tile, 16-byte chunk tid % 64.
    let ids = p.ids as *const u32;
    let keys = (p.keys as *const f32).add(key_head * 256);
    let stride = p.key_heads as usize * 256;
    // The current tile's buffer holds the partials once every warp has read
    // its keys: 2 x 16 KB static shared, three CTAs per SM.
    let base = shared::<{ 2 * STAGE }>();
    let (copy_row, chunk) = (tid / 64, tid % 64);
    let swizzled = ((chunk ^ ((copy_row & 1) << 2)) << 4) as u32;
    let issue = |tile: usize| {
        let (start, buffer) = (begin + tile * TILE, base + (tile % 2 * STAGE) as u32);
        eight!(M, {
            let row = copy_row + 2 * M;
            if start + row < end {
                let pos = selected_position(ids, complete, token, p.blocks, start + row);
                copy16(buffer + (row * ROW) as u32 + swizzled, keys.add(pos * stride + chunk * 4).cast());
            }
        });
        commit();
    };
    let tiles = (end - begin).div_ceil(TILE);
    issue(0);
    for tile in 0..tiles {
        wait();
        sync();
        if tile + 1 < tiles { issue(tile + 1); }
        let buffer = base + (tile % 2 * STAGE) as u32;
        // Both N8 key tiles advance together: four independent MMA chains.
        let mut big = [[0.0f32; 4]; 2];
        let mut small = [[0.0f32; 4]; 2];
        four!(J, {
            let c = 16 * warp + 4 * J + t;
            let mut k = [[[0u32; 3]; 2]; 2];
            two!(N, {
                let row = 8 * N + g;
                let x = load4(buffer + (row * ROW + ((c ^ ((row & 1) << 2)) << 4)) as u32).map(f32::from_bits);
                k[N] = [split(x[0], x[1]), split(x[2], x[3])];
            });
            two!(N, { big[N] = mma(qf[J][0], k[N][0][0], k[N][1][0], big[N]); });
            two!(N, { small[N] = mma(qf[J][0], k[N][0][1], k[N][1][1], small[N]); });
            two!(N, { small[N] = mma(qf[J][1], k[N][0][0], k[N][1][0], small[N]); });
            if PLANES == 3 {
                two!(N, { small[N] = mma(qf[J][0], k[N][0][2], k[N][1][2], small[N]); });
                two!(N, { small[N] = mma(qf[J][1], k[N][0][1], k[N][1][1], small[N]); });
                two!(N, { small[N] = mma(qf[J][2], k[N][0][0], k[N][1][0], small[N]); });
            }
        });
        // C fragment: rows g and g + 8, keys 2t and 2t + 1 of each N8 tile.
        sync();
        let partial = buffer;
        let at = |r: usize, key: usize| partial + (((warp * 16 + r) * 16 + key) * 4) as u32;
        two!(N, {
            store2(at(g, 8 * N + 2 * t), big[N][0] + small[N][0], big[N][1] + small[N][1]);
            store2(at(g + 8, 8 * N + 2 * t), big[N][2] + small[N][2], big[N][3] + small[N][3]);
        });
        sync();
        // Fixed fold of the four feature slices, then the exact 1/16 scale.
        two!(H, {
            let o = tid + 128 * H;
            if o < HEADS * 16 {
                let (r, key) = (o / 16, o % 16);
                let slot = begin + tile * TILE + key;
                let row = row_of(r);
                if slot < end && row != usize::MAX {
                    let v = |w: usize| load(partial + (((w * 16 + r) * 16 + key) * 4) as u32);
                    *(p.scores as *mut f32).add(row * p.stride as usize + slot) = ((v(0) + v(1)) + (v(2) + v(3))) * (1.0 / 16.0);
                }
            }
        });
    }
}

/// Scores with warp-owned slots (prompt rows): the same per-score arithmetic
/// as `scores` (feature block w = features 64w .. 64w + 63: a big hh chain
/// of its four K16 steps and a small chain of the other products from 0,
/// p_w = big + small; S = ((p0 + p1) + (p2 + p3)) * (1/16)), but one warp
/// computes all four blocks of an 8-slot N tile, so no partials cross warps
/// and the main loop has no barrier. K fragments come straight from L2 into
/// registers two 64-feature chunks ahead (slot positions one tile ahead);
/// the Q planes are split once per CTA into shared memory (24 KB, A-fragment
/// order). Warp w takes tiles w, w + 4, ... of the CTA's slot range.
const QPLANES: usize = 16 * 3 * 32 * 16;
#[inline(always)]
unsafe fn scores_w(p: QsaAttention) {
    const HEADS: usize = QSA_SCORE_HEADS;
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let groups = (p.query_start as usize % HEADS + p.query_heads as usize).div_ceil(HEADS);
    let group = _block_idx_x() as usize;
    if group >= groups * p.tokens as usize { return; }
    let token = group / groups;
    let first = (p.query_start as usize / HEADS + group % groups) * HEADS;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let range = QSA_MAX_POSITIONS.div_ceil(_grid_dim_y() as usize).div_ceil(TILE) * TILE;
    let begin = _block_idx_y() as usize * range;
    if begin >= visible { return; }
    let end = (begin + range).min(visible);
    let key_head = first / p.queries_per_key as usize - p.key_start as usize;
    let (qs, qe) = (p.query_start as usize, (p.query_start + p.query_heads) as usize);
    let row_of = |r: usize| if r < HEADS && first + r >= qs && first + r < qe {
        token * p.query_heads as usize + first + r - qs } else { usize::MAX };
    // Q planes: entry (J, plane, lane) = this lane's A fragment of K16 step J.
    let base = shared::<QPLANES>();
    let (ra, rb) = (row_of(g), row_of(g + 8));
    let query = p.query as *const f32;
    four!(JJ, {
        let j = 4 * warp + JJ;
        let d = 16 * j + 4 * t;
        let xa = if ra != usize::MAX { global4(query.add(ra * 256 + d)) } else { [0.0; 4] };
        let xb = if rb != usize::MAX { global4(query.add(rb * 256 + d)) } else { [0.0; 4] };
        let (a01, a23, b01, b23) = (split(xa[0], xa[1]), split(xa[2], xa[3]), split(xb[0], xb[1]), split(xb[2], xb[3]));
        three!(P, { store4u(base + (((j * 3 + P) * 32 + lane) * 16) as u32, [a01[P], b01[P], a23[P], b23[P]]); });
    });
    let ids = p.ids as *const u32;
    let keys = (p.keys as *const f32).add(key_head * 256 + 4 * t);
    let stride = p.key_heads as usize * 256;
    let scores = p.scores as *mut f32;
    let tiles = (end - begin).div_ceil(8);
    // Chunk c = (tile, block): tile = warp + 4 (c / 4), block = c % 4.
    let chunks = if warp < tiles { (tiles - warp).div_ceil(4) * 4 } else { 0 };
    let position = |c: usize| {
        let slot = begin + 8 * (warp + 4 * (c / 4)) + g;
        if c < chunks && slot < end { selected_position(ids, complete, token, p.blocks, slot) } else { usize::MAX }
    };
    let fetch = |pos: usize, block: usize| -> [[f32; 4]; 4] {
        let mut x = [[0.0f32; 4]; 4];
        if pos != usize::MAX { four!(JJ, { x[JJ] = global4(keys.add(pos * stride + 64 * block + 16 * JJ)); }); }
        x
    };
    sync();
    let mut pos = position(0);
    let mut pos_next = position(4);
    let mut k0 = fetch(pos, 0);
    let mut k1 = fetch(pos, 1);
    let (mut p01, mut p2) = ([0.0f32; 4], [0.0f32; 4]);
    let mut c = 0;
    while c < chunks {
        let block = c % 4;
        // Two chunks ahead: blocks 2, 3 of this tile, then 0, 1 of the next.
        let ahead = c + 2;
        let k2 = if ahead % 4 >= 2 { fetch(pos, ahead % 4) } else { fetch(pos_next, ahead % 4) };
        let mut big = [0.0f32; 4];
        let mut small = [0.0f32; 4];
        four!(JJ, {
            let j = 4 * block + JJ;
            let mut a = [[0u32; 4]; 3];
            three!(P, { a[P] = load4(base + (((j * 3 + P) * 32 + lane) * 16) as u32); });
            let x = k0[JJ];
            let (b0, b1) = (split(x[0], x[1]), split(x[2], x[3]));
            big = mma(a[0], b0[0], b1[0], big);
            small = mma(a[0], b0[1], b1[1], small);
            small = mma(a[1], b0[0], b1[0], small);
            small = mma(a[0], b0[2], b1[2], small);
            small = mma(a[1], b0[1], b1[1], small);
            small = mma(a[2], b0[0], b1[0], small);
        });
        let mut pw = [0.0f32; 4];
        four!(E, { pw[E] = big[E] + small[E]; });
        if block == 0 { p01 = pw; }
        else if block == 1 { four!(E, { p01[E] = p01[E] + pw[E]; }); }
        else if block == 2 { p2 = pw; }
        else {
            // C fragment: heads g and g + 8, slots 2t and 2t + 1 of the tile.
            let tile_start = begin + 8 * (warp + 4 * (c / 4));
            two!(HALF, {
                let row = row_of(g + 8 * HALF);
                if row != usize::MAX {
                    two!(C, {
                        let slot = tile_start + 2 * t + C;
                        let e = 2 * HALF + C;
                        if slot < end { *scores.add(row * p.stride as usize + slot) = (p01[e] + (p2[e] + pw[e])) * (1.0 / 16.0); }
                    });
                }
            });
            pos = pos_next;
            pos_next = position(c + 5);
        }
        k0 = k1;
        k1 = k2;
        c += 1;
    }
}

/// Prompt rows (>= 64 tokens), same bytes as flash_qsa_attention_scores_tc.
/// Block 128, grid (groups * tokens, slot ranges); static shared 24,576 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_tc_w(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    scores_w(p);
}

/// Block128, grid (groups * tokens, slot ranges); static shared 32,768 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_tc(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    scores::<3>(p);
}
/// Sessions in grid-z, as flash_qsa_attention_scores_batch.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_tc_batch(p: QsaBatch) {
    let row = _block_idx_z() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    scores::<3>(batch_attention(p, r, row));
}
/// Benchmark screen only: 2-plane hi/lo (hh + hm + mh), about 2^-14.4 per product.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_tc2_batch(p: QsaBatch) {
    let row = _block_idx_z() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    scores::<2>(batch_attention(p, r, row));
}
/// Benchmark screen only: 2-plane hi/lo (hh + hm + mh), about 2^-14.4 per product.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_tc2(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    scores::<2>(p);
}

/// Values stage: one V row (256 FP32 features) and the P tile [16 heads][16 slots].
const VROW: usize = 1024;
const VSTAGE: usize = TILE * VROW + 16 * 16 * 4;

/// Tensor-core values: out[row][f] = (sum_s P[row][s] V[pos(s)][f] / d[row]) * gate,
/// the probabilities and denominators of flash_qsa_attention_softmax (main's).
/// Order: 16-slot k-steps in ascending slot order (slot 4t + i of a step is MMA
/// k = 2t, 2t + 1, 2t + 8, 2t + 9); per k-step and 8-feature tile one fresh
/// accumulator sums the small products first and hh last (lh, mm, hl, mh, hm,
/// hh for PLANES = 3; mh, hm, hh for 2), then out += step (RN). Truncation of
/// each MMA is relative to that step's partial only. Warp w owns features
/// 64w + 32q + 4j + r (q < 2, j < 8, r < 4): MMA tile 4q + r, column j.
/// CTA = one token x one KV head (all its local query heads), 128 threads,
/// static shared 2 x VSTAGE = 34,816 B; slots past the row and heads outside
/// the group are zero in both operands.
#[inline(always)]
unsafe fn values<const PLANES: usize>(p: QsaAttention) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let tokens = p.tokens as usize;
    let block = _block_idx_x() as usize;
    if block >= tokens * p.key_heads as usize { return; }
    let (token, key_head) = (block % tokens, block / tokens);
    let qpk = p.queries_per_key as usize;
    let first = ((key_head + p.key_start as usize) * qpk).max(p.query_start as usize);
    let end = ((key_head + p.key_start as usize + 1) * qpk).min((p.query_start + p.query_heads) as usize);
    if first >= end { return; }
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let ids = p.ids as *const u32;
    let values = (p.values as *const f32).add(key_head * 256);
    let stride = p.key_heads as usize * 256;
    let probabilities = p.scores as *const f32;
    let base = shared::<{ 2 * VSTAGE }>();
    let (copy_row, chunk) = (tid / 64, tid % 64);
    let issue = |step: usize| {
        let (start, buffer) = (step * TILE, base + (step % 2 * VSTAGE) as u32);
        eight!(M, {
            let row = copy_row + 2 * M;
            let dst = buffer + (row * VROW + ((chunk ^ (((row >> 2) & 3) << 1)) << 4)) as u32;
            if start + row < visible {
                let pos = selected_position(ids, complete, token, p.blocks, start + row);
                copy16(dst, values.add(pos * stride + chunk * 4).cast());
            } else {
                zero16(dst);
            }
        });
        let weights = buffer + (TILE * VROW) as u32;
        two!(H, {
            let e = tid + 128 * H;
            let (r, i) = (e / 16, e % 16);
            if r < QSA_SCORE_HEADS {
                if r < group && start + i < visible {
                    copy4(weights + (e * 4) as u32, probabilities.add((first_row + r) * p.stride as usize + start + i).cast());
                } else {
                    store(weights + (e * 4) as u32, 0.0);
                }
            }
        });
        commit();
    };
    let steps = visible.div_ceil(TILE);
    let mut out = [[0.0f32; 4]; 8];
    issue(0);
    for step in 0..steps {
        wait();
        sync();
        if step + 1 < steps { issue(step + 1); }
        let buffer = base + (step % 2 * VSTAGE) as u32;
        let weights = buffer + (TILE * VROW) as u32;
        let xa = load4(weights + ((g * 16 + 4 * t) * 4) as u32).map(f32::from_bits);
        let xb = if g < 4 { load4(weights + (((g + 8) * 16 + 4 * t) * 4) as u32).map(f32::from_bits) } else { [0.0; 4] };
        let (a01, b01, a23, b23) = (split(xa[0], xa[1]), split(xb[0], xb[1]), split(xa[2], xa[3]), split(xb[2], xb[3]));
        let mut a = [[0u32; 4]; 3];
        three!(P, { a[P] = [a01[P], b01[P], a23[P], b23[P]]; });
        two!(Q, {
            let mut v = [[0.0f32; 4]; 4];
            four!(S, {
                let row = 4 * t + S;
                let c = 16 * warp + 8 * Q + g;
                v[S] = load4(buffer + (row * VROW + ((c ^ (((row >> 2) & 3) << 1)) << 4)) as u32).map(f32::from_bits);
            });
            four!(R, {
                let (b0, b1) = (split(v[0][R], v[1][R]), split(v[2][R], v[3][R]));
                let mut acc = [0.0f32; 4];
                if PLANES == 3 {
                    acc = mma(a[2], b0[0], b1[0], acc);
                    acc = mma(a[1], b0[1], b1[1], acc);
                    acc = mma(a[0], b0[2], b1[2], acc);
                }
                acc = mma(a[1], b0[0], b1[0], acc);
                acc = mma(a[0], b0[1], b1[1], acc);
                acc = mma(a[0], b0[0], b1[0], acc);
                four!(E, { out[4 * Q + R][E] += acc[E]; });
            });
        });
    }
    // C fragment: heads g and g + 8, columns 2t and 2t + 1 of each tile.
    let denominators = p.denominators as *const f32;
    let gates = p.query_gate as *const f32;
    let output = p.output as *mut f32;
    two!(HALF, {
        let r = g + 8 * HALF;
        if r < group {
            let row = first_row + r;
            let d = *denominators.add(row);
            eight!(I, {
                two!(C, {
                    let feature = 64 * warp + 32 * (I / 4) + 4 * (2 * t + C) + I % 4;
                    let gate = output_gate(*gates.add(row * 512 + 256 + feature), 0);
                    *output.add(row * 256 + feature) = (out[I][2 * HALF + C] / d) * gate;
                });
            });
        }
    });
}

/// Block 128, grid (tokens * local KV heads); static shared 34,816 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    values::<3>(p);
}
/// Sessions in grid-y, as flash_qsa_attention_values_batch (grid x = local KV heads).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc_batch(p: QsaBatch) {
    let row = _block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    values::<3>(batch_attention(p, r, row));
}
/// Benchmark screen only: 2-plane hi/lo values.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc2(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    values::<2>(p);
}
/// Benchmark screen only: 2-plane hi/lo values, sessions in grid-y.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc2_batch(p: QsaBatch) {
    let row = _block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    values::<2>(batch_attention(p, r, row));
}

/// Fused prompt attention: flash_qsa_attention_scores_tc_w, main's
/// flash_qsa_attention_softmax and flash_qsa_attention_values_tc in one CTA per
/// token x KV head, with the arithmetic of that chain for every output (same
/// bytes). Phase 1 = scores_w over all visible slots (S stored where the chain
/// stores it; each lane keeps the maximum of the scores it stored). Phase 2 =
/// values with P = exp(S - highest) formed in registers from the stored S
/// (re-read one k-step ahead, L2-hot) instead of main's stored probabilities;
/// the denominator is main's (((0 + P0) + P1) + ...) in ascending slot order,
/// carried across the four lanes of a row (padded slots add +0: exact).
/// Static shared 32,768 B, aliased: Q planes (phase 1), row maxima at
/// QPLANES (between phases), two 16-slot V stages (phase 2); plus the token's
/// 512 selected group ids (2 KB), copied once at the start so no K or
/// V gather waits on a global ids load (positions are selected_position's).
const FSTAGE: usize = TILE * VROW;
#[inline(always)]
unsafe fn lane_value(v: f32, source: usize) -> f32 {
    let d: u32;
    asm!("shfl.sync.idx.b32 {d}, {v}, {s}, 31, 0xffffffff;", d = out(reg32) d, v = in(reg32) v.to_bits(),
        s = in(reg32) source as u32);
    f32::from_bits(d)
}
#[inline(always)]
/// One FP32 load with L1 evict-last priority (the S row all four warps re-read).
#[inline(always)]
unsafe fn load1_keep(p: *const f32) -> f32 {
    let v: f32;
    asm!("ld.global.L1::evict_last.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) p);
    v
}
unsafe fn fused<const VREG: bool>(p: QsaAttention) {
    const HEADS: usize = QSA_SCORE_HEADS;
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let groups = (p.query_start as usize % HEADS + p.query_heads as usize).div_ceil(HEADS);
    let group_index = _block_idx_x() as usize;
    if group_index >= groups * p.tokens as usize { return; }
    // KV-head-major (one head's K and V at a time in L2); grid order only.
    let token = group_index % p.tokens as usize;
    let first = (p.query_start as usize / HEADS + group_index / p.tokens as usize) * HEADS;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let key_head = first / p.queries_per_key as usize - p.key_start as usize;
    let (qs, qe) = (p.query_start as usize, (p.query_start + p.query_heads) as usize);
    let qpk = p.queries_per_key as usize;
    let vfirst = ((key_head + p.key_start as usize) * qpk).max(qs);
    let vend = ((key_head + p.key_start as usize + 1) * qpk).min(qe);
    if vfirst >= vend { return; }
    let row_of = |r: usize| if r < HEADS && first + r >= qs && first + r < qe {
        token * p.query_heads as usize + first + r - qs } else { usize::MAX };
    let (base, id_tile) = if VREG {
        let b = shared::<{ QPLANES + 256 + ID_TILE }>();
        (b, b + (QPLANES + 256) as u32)
    } else {
        let b = shared::<{ 2 * FSTAGE + ID_TILE }>();
        (b, b + (2 * FSTAGE) as u32)
    };
    stage_ids(id_tile, p.ids as *const u32, complete, token, p.blocks, tid, 128);
    commit();
    let spos = |slot: usize| tile_position(id_tile, complete, slot);
    // Phase 1: scores_w with one slot range [0, visible).
    let (ra, rb) = (row_of(g), row_of(g + 8));
    let query = p.query as *const f32;
    four!(JJ, {
        let j = 4 * warp + JJ;
        let d = 16 * j + 4 * t;
        let xa = if ra != usize::MAX { global4(query.add(ra * 256 + d)) } else { [0.0; 4] };
        let xb = if rb != usize::MAX { global4(query.add(rb * 256 + d)) } else { [0.0; 4] };
        let (a01, a23, b01, b23) = (split(xa[0], xa[1]), split(xa[2], xa[3]), split(xb[0], xb[1]), split(xb[2], xb[3]));
        three!(P, { store4u(base + (((j * 3 + P) * 32 + lane) * 16) as u32, [a01[P], b01[P], a23[P], b23[P]]); });
    });
    let keys = (p.keys as *const f32).add(key_head * 256 + 4 * t);
    let stride = p.key_heads as usize * 256;
    let scores = p.scores as *mut f32;
    let end = visible;
    let tiles = end.div_ceil(8);
    let chunks = if warp < tiles { (tiles - warp).div_ceil(4) * 4 } else { 0 };
    let position = |c: usize| {
        let slot = 8 * (warp + 4 * (c / 4)) + g;
        if c < chunks && slot < end { spos(slot) } else { usize::MAX }
    };
    let fetch = |pos: usize, block: usize| -> [[f32; 4]; 4] {
        let mut x = [[0.0f32; 4]; 4];
        if pos != usize::MAX { four!(JJ, { x[JJ] = global4(keys.add(pos * stride + 64 * block + 16 * JJ)); }); }
        x
    };
    wait();
    sync();
    let mut high = [f32::NEG_INFINITY; 2];
    // Positions come from the ID tile at fetch time (the earlier variant kept three
    // tiles of positions in registers to hide the global ids load).
    let mut k0 = fetch(position(0), 0);
    let mut k1 = fetch(position(1), 1);
    let (mut p01, mut p2) = ([0.0f32; 4], [0.0f32; 4]);
    let mut c = 0;
    while c < chunks {
        let block = c % 4;
        let ahead = c + 2;
        let k2 = fetch(position(ahead), ahead % 4);
        let mut big = [0.0f32; 4];
        let mut small = [0.0f32; 4];
        four!(JJ, {
            let j = 4 * block + JJ;
            let mut a = [[0u32; 4]; 3];
            three!(P, { a[P] = load4(base + (((j * 3 + P) * 32 + lane) * 16) as u32); });
            let x = k0[JJ];
            let (b0, b1) = (split(x[0], x[1]), split(x[2], x[3]));
            big = mma(a[0], b0[0], b1[0], big);
            small = mma(a[0], b0[1], b1[1], small);
            small = mma(a[1], b0[0], b1[0], small);
            small = mma(a[0], b0[2], b1[2], small);
            small = mma(a[1], b0[1], b1[1], small);
            small = mma(a[2], b0[0], b1[0], small);
        });
        let mut pw = [0.0f32; 4];
        four!(E, { pw[E] = big[E] + small[E]; });
        if block == 0 { p01 = pw; }
        else if block == 1 { four!(E, { p01[E] = p01[E] + pw[E]; }); }
        else if block == 2 { p2 = pw; }
        else {
            let tile_start = 8 * (warp + 4 * (c / 4));
            two!(HALF, {
                let row = row_of(g + 8 * HALF);
                if row != usize::MAX {
                    two!(C, {
                        let slot = tile_start + 2 * t + C;
                        let e = 2 * HALF + C;
                        if slot < end {
                            let s = (p01[e] + (p2[e] + pw[e])) * (1.0 / 16.0);
                            *scores.add(row * p.stride as usize + slot) = s;
                            high[HALF] = high[HALF].max(s);
                        }
                    });
                }
            });
        }
        k0 = k1;
        k1 = k2;
        c += 1;
    }
    // Row maxima (exact in any order): lanes of a row, then warps via shared memory.
    two!(HALF, {
        high[HALF] = high[HALF].max(f32::from_bits(shuffle_xor(u32::MAX, high[HALF].to_bits(), 1, 31)));
        high[HALF] = high[HALF].max(f32::from_bits(shuffle_xor(u32::MAX, high[HALF].to_bits(), 2, 31)));
    });
    let maxima = base + QPLANES as u32;
    if t == 0 { two!(HALF, { store(maxima + ((warp * 16 + g + 8 * HALF) * 4) as u32, high[HALF]); }); }
    sync();
    // Phase 2: values (rows = the KV head's local query heads from vfirst).
    let group = vend - vfirst;
    let shift = vfirst - first;
    let first_row = token * p.query_heads as usize + vfirst - qs;
    // Rows outside the group: highest 0 and raw scores -inf, so P = exp(-inf) = +0.
    let mut highest = [0.0f32; 2];
    let mut srow = [usize::MAX; 2];
    two!(HALF, {
        let r = g + 8 * HALF;
        if r < group {
            let mut h = f32::NEG_INFINITY;
            four!(W, { h = h.max(load(maxima + ((W * 16 + r + shift) * 4) as u32)); });
            highest[HALF] = h;
            srow[HALF] = (first_row + r) * p.stride as usize;
        }
    });
    let values = (p.values as *const f32).add(key_head * 256);
    let (copy_row, chunk) = (tid / 64, tid % 64);
    let issue = |step: usize| {
        let (start, buffer) = (step * TILE, base + (step % 2 * FSTAGE) as u32);
        eight!(M, {
            let row = copy_row + 2 * M;
            let dst = buffer + (row * VROW + ((chunk ^ (((row >> 2) & 3) << 1)) << 4)) as u32;
            if start + row < visible {
                copy16(dst, values.add(spos(start + row) * stride + chunk * 4).cast());
            } else {
                zero16(dst);
            }
        });
        commit();
    };
    // This lane's A-fragment scores: rows g, g + 8, slots 4t .. 4t + 3 of a k-step;
    // branch-free (padding reads element 0 and becomes -inf) so the denominator
    // shuffles below stay convergent.
    // The -inf select happens at use (one step later), so the loads are not
    // consumed right after issue.
    let valid = |step: usize, half: usize, i: usize| srow[half] != usize::MAX && step * TILE + 4 * t + i < visible;
    let raw = |step: usize| -> [[f32; 4]; 2] {
        let mut x = [[0.0f32; 4]; 2];
        two!(HALF, {
            four!(I, {
                let slot = step * TILE + 4 * t + I;
                x[HALF][I] = load1_keep(scores.add(if valid(step, HALF, I) { srow[HALF] + slot } else { 0 }));
            });
        });
        x
    };
    // VREG: this lane's B fragments (slots 4t + S, features 4(16w + 8Q + g) ..)
    // straight from the cache into registers a k-step ahead; no stage, no barrier.
    // Slots 4t .. 4t + 3 of a k-step are one selected group (or consecutive
    // dense / tail positions), so their positions are p0 .. p0 + 3 (
    // one ID-tile read per k-step instead of four).
    let vfetch = |step: usize| -> [[[f32; 4]; 2]; 4] {
        let mut v = [[[0.0f32; 4]; 2]; 4];
        let first = step * TILE + 4 * t;
        let p0 = if first < visible { spos(first) } else { 0 };
        four!(S, {
            if first + S < visible {
                let row = values.add((p0 + S) * stride);
                two!(Q, { v[S][Q] = global4_cg(row.add(4 * (16 * warp + 8 * Q + g))); });
            }
        });
        v
    };
    let steps = visible.div_ceil(TILE);
    let mut out = [[0.0f32; 4]; 8];
    let mut den = [0.0f32; 2];
    let mut vnext = if VREG { vfetch(0) } else { [[[0.0f32; 4]; 2]; 4] };
    if !VREG { issue(0); }
    let mut next = raw(0);
    for step in 0..steps {
        let mut pr = [[0.0f32; 4]; 2];
        two!(HALF, { four!(I, {
            let x = if valid(step, HALF, I) { next[HALF][I] } else { f32::NEG_INFINITY };
            pr[HALF][I] = exp(x - highest[HALF]);
        }); });
        if step + 1 < steps { next = raw(step + 1); }
        four!(TT, {
            two!(HALF, {
                let mut d = den[HALF];
                four!(I, { d = d + pr[HALF][I]; });
                den[HALF] = lane_value(if t == TT { d } else { den[HALF] }, 4 * g + TT);
            });
        });
        let vcur = vnext;
        if VREG {
            if step + 1 < steps { vnext = vfetch(step + 1); }
        } else {
            wait();
            sync();
            if step + 1 < steps { issue(step + 1); }
        }
        let buffer = base + (step % 2 * FSTAGE) as u32;
        let (xa, xb) = (pr[0], pr[1]);
        let (a01, b01, a23, b23) = (split(xa[0], xa[1]), split(xb[0], xb[1]), split(xa[2], xa[3]), split(xb[2], xb[3]));
        let mut a = [[0u32; 4]; 3];
        three!(P, { a[P] = [a01[P], b01[P], a23[P], b23[P]]; });
        two!(Q, {
            let mut v = [[0.0f32; 4]; 4];
            four!(S, {
                if VREG { v[S] = vcur[S][Q]; } else {
                    let row = 4 * t + S;
                    let c = 16 * warp + 8 * Q + g;
                    v[S] = load4(buffer + (row * VROW + ((c ^ (((row >> 2) & 3) << 1)) << 4)) as u32).map(f32::from_bits);
                }
            });
            four!(R, {
                let (b0, b1) = (split(v[0][R], v[1][R]), split(v[2][R], v[3][R]));
                let mut acc = [0.0f32; 4];
                acc = mma(a[2], b0[0], b1[0], acc);
                acc = mma(a[1], b0[1], b1[1], acc);
                acc = mma(a[0], b0[2], b1[2], acc);
                acc = mma(a[1], b0[0], b1[0], acc);
                acc = mma(a[0], b0[1], b1[1], acc);
                acc = mma(a[0], b0[0], b1[0], acc);
                four!(E, { out[4 * Q + R][E] += acc[E]; });
            });
        });
    }
    let gates = p.query_gate as *const f32;
    let output = p.output as *mut f32;
    two!(HALF, {
        let r = g + 8 * HALF;
        if r < group {
            let row = first_row + r;
            let d = den[HALF];
            eight!(I, {
                two!(C, {
                    let feature = 64 * warp + 32 * (I / 4) + 4 * (2 * t + C) + I % 4;
                    let gate = output_gate(*gates.add(row * 512 + 256 + feature), 0);
                    *output.add(row * 256 + feature) = (out[I][2 * HALF + C] / d) * gate;
                });
            });
        }
    });
}

/// Prompt rows (>= 64 tokens) under VADE_QSA_TC: replaces scores _tc_w +
/// softmax + values _tc (same output bytes; the scores buffer keeps S, not P).
/// Block 128, grid (groups * tokens); static shared 26,880 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_fused_tc(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    // Phase 2 takes V fragments into registers a k-step ahead (the
    // staged-V form, fused::<false>, ran 20.4 ms at 32k vs 16.0).
    fused::<true>(p);
}

/// Small launches (decode, MTP verify, prompt tails under 64 tokens): stage =
/// 64 slots x one 64-feature slice (V) + the P tile [12 heads][64 slots,
/// rows padded to 80 floats].
const MROWS: usize = 64;
const MVROW: usize = 256;
const MPROW: usize = 80;
const MSTAGE: usize = MROWS * MVROW + QSA_SCORE_HEADS * MPROW * 4;

/// Same per-output arithmetic as `values` (16-slot k-steps ascending, slot
/// 4t + i = MMA k 2t, 2t + 1, 2t + 8, 2t + 9, one fresh accumulator per step
/// and 8-feature tile with the small products first and hh last, out += step,
/// only k-steps holding a visible slot) on four 64-feature slices per KV head
/// and four k-steps per barrier. Warp w of slice s owns features
/// 64s + 32(w >> 1) + 4j + 2(w & 1) + i (tile i < 2, column j): every feature sits in the MMA
/// column it has in `values` (column j of 64w' + 32q + 4j + r), so the two
/// entries agree by construction, not only by measured column invariance.
/// CTA = token x KV head x slice,
/// 128 threads, static shared 2 x MSTAGE = 40,448 B.
#[inline(always)]
unsafe fn values_m<const PLANES: usize>(p: QsaAttention) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let tokens = p.tokens as usize;
    let block = _block_idx_x() as usize;
    if block >= tokens * p.key_heads as usize * 4 { return; }
    let (token, rest) = (block % tokens, block / tokens);
    let (key_head, slice) = (rest / 4, rest % 4);
    let qpk = p.queries_per_key as usize;
    let first = ((key_head + p.key_start as usize) * qpk).max(p.query_start as usize);
    let end = ((key_head + p.key_start as usize + 1) * qpk).min((p.query_start + p.query_heads) as usize);
    if first >= end { return; }
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let ids = p.ids as *const u32;
    let values = (p.values as *const f32).add(key_head * 256 + slice * 64);
    let stride = p.key_heads as usize * 256;
    let probabilities = p.scores as *const f32;
    let base = shared::<{ 2 * MSTAGE }>();
    let (copy_row, chunk) = (tid / 16, tid % 16);
    let positions = |stage: usize| {
        let start = stage * MROWS;
        let mut pos = [0u32; 8];
        eight!(M, {
            let row = copy_row + 8 * M;
            if start + row < visible { pos[M] = selected_position(ids, complete, token, p.blocks, start + row) as u32; }
        });
        pos
    };
    let issue = |stage: usize, pos: [u32; 8]| {
        let (start, buffer) = (stage * MROWS, base + (stage % 2 * MSTAGE) as u32);
        eight!(M, {
            let row = copy_row + 8 * M;
            let dst = buffer + (row * MVROW + ((chunk ^ (((row >> 2) & 3) << 1)) << 4)) as u32;
            if start + row < visible {
                copy16(dst, values.add(pos[M] as usize * stride + chunk * 4).cast());
            } else {
                zero16(dst);
            }
        });
        let weights = buffer + (MROWS * MVROW) as u32;
        two!(H, {
            three!(K, {
                let e = tid + 128 * (3 * H + K);
                let (r, i) = (e / MROWS, e % MROWS);
                let at = weights + ((r * MPROW + i) * 4) as u32;
                if r < group && start + i < visible {
                    copy4(at, probabilities.add((first_row + r) * p.stride as usize + start + i).cast());
                } else {
                    store(at, 0.0);
                }
            });
        });
        commit();
    };
    let stages = visible.div_ceil(MROWS);
    let mut out = [[0.0f32; 4]; 2];
    issue(0, positions(0));
    let mut next = if stages > 1 { positions(1) } else { [0; 8] };
    for stage in 0..stages {
        wait();
        sync();
        if stage + 1 < stages { issue(stage + 1, next); }
        if stage + 2 < stages { next = positions(stage + 2); }
        let buffer = base + (stage % 2 * MSTAGE) as u32;
        let weights = buffer + (MROWS * MVROW) as u32;
        let ksteps = (visible - stage * MROWS).min(MROWS).div_ceil(TILE);
        four!(KK, {
            if KK < ksteps {
                let xa = load4(weights + ((g * MPROW + 16 * KK + 4 * t) * 4) as u32).map(f32::from_bits);
                let xb = if g < 4 { load4(weights + (((g + 8) * MPROW + 16 * KK + 4 * t) * 4) as u32).map(f32::from_bits) } else { [0.0; 4] };
                let (a01, b01, a23, b23) = (split(xa[0], xa[1]), split(xb[0], xb[1]), split(xa[2], xa[3]), split(xb[2], xb[3]));
                let mut a = [[0u32; 4]; 3];
                three!(P, { a[P] = [a01[P], b01[P], a23[P], b23[P]]; });
                let mut v = [[0.0f32; 2]; 4];
                four!(S, {
                    let row = 16 * KK + 4 * t + S;
                    let c = 8 * (warp >> 1) + g;
                    v[S] = load2(buffer + (row * MVROW + ((c ^ (((row >> 2) & 3) << 1)) << 4) + 8 * (warp & 1)) as u32).map(f32::from_bits);
                });
                two!(I, {
                    let (b0, b1) = (split(v[0][I], v[1][I]), split(v[2][I], v[3][I]));
                    let mut acc = [0.0f32; 4];
                    if PLANES == 3 {
                        acc = mma(a[2], b0[0], b1[0], acc);
                        acc = mma(a[1], b0[1], b1[1], acc);
                        acc = mma(a[0], b0[2], b1[2], acc);
                    }
                    acc = mma(a[1], b0[0], b1[0], acc);
                    acc = mma(a[0], b0[1], b1[1], acc);
                    acc = mma(a[0], b0[0], b1[0], acc);
                    four!(E, { out[I][E] += acc[E]; });
                });
            }
        });
    }
    let denominators = p.denominators as *const f32;
    let gates = p.query_gate as *const f32;
    let output = p.output as *mut f32;
    two!(HALF, {
        let r = g + 8 * HALF;
        if r < group {
            let row = first_row + r;
            let d = *denominators.add(row);
            two!(I, {
                two!(C, {
                    let feature = 64 * slice + 32 * (warp >> 1) + 4 * (2 * t + C) + 2 * (warp & 1) + I;
                    let gate = output_gate(*gates.add(row * 512 + 256 + feature), 0);
                    *output.add(row * 256 + feature) = (out[I][2 * HALF + C] / d) * gate;
                });
            });
        }
    });
}

/// Block 128, grid (tokens * local KV heads * 4 slices); static shared 40,448 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc_m(mut p: QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    values_m::<3>(p);
}
/// Sessions in grid-y, grid x = local KV heads x 4 slices.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc_m_batch(p: QsaBatch) {
    let row = _block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    values_m::<3>(batch_attention(p, r, row));
}

/// values_m split by k-step (one-token rows). Phase A: CTA = token x KV
/// head x 64-feature slice x 64-slot stage (grid-y), computing each k-step's
/// fresh tile exactly as values_m and storing it (tile [16 rows][64 features] per
/// k-step). Phase B: one thread per output adds its tiles from 0.0 in ascending
/// k-step order, then / d * gate. Same tiles, same additions, same bytes.
const KMAX: usize = QSA_MAX_POSITIONS.div_ceil(TILE);
#[inline(always)]
fn split_tile(p: &QsaAttention, token: usize, key_head: usize, slice: usize) -> usize {
    ((token * p.key_heads as usize + key_head) * 4 + slice) * KMAX
}
// Compact constant-trip loops and plain arithmetic (no core min / max / div_ceil) keep this code's
// codegen unit small, so no other entry's code moves.
#[inline(always)]
unsafe fn values_k(p: QsaAttention, partials: *mut f32) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let tokens = p.tokens as usize;
    let block = _block_idx_x() as usize;
    if block >= tokens * p.key_heads as usize * 4 { return; }
    let (token, rest) = (block % tokens, block / tokens);
    let (key_head, slice) = (rest / 4, rest % 4);
    let qpk = p.queries_per_key as usize;
    let (lo, hi) = ((key_head + p.key_start as usize) * qpk, (key_head + p.key_start as usize + 1) * qpk);
    let first = if lo > p.query_start as usize { lo } else { p.query_start as usize };
    let last = (p.query_start + p.query_heads) as usize;
    let end = if hi < last { hi } else { last };
    if first >= end { return; }
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let stage = _block_idx_y() as usize;
    if stage * MROWS >= visible { return; }
    let ids = p.ids as *const u32;
    let values = (p.values as *const f32).add(key_head * 256 + slice * 64);
    let stride = p.key_heads as usize * 256;
    let probabilities = p.scores as *const f32;
    let buffer = shared::<MSTAGE>();
    let (copy_row, chunk) = (tid / 16, tid % 16);
    let start = stage * MROWS;
    let mut m = 0;
    while m < 8 {
        let row = copy_row + 8 * m;
        let dst = buffer + (row * MVROW + ((chunk ^ (((row >> 2) & 3) << 1)) << 4)) as u32;
        if start + row < visible {
            let pos = selected_position(ids, complete, token, p.blocks, start + row);
            copy16(dst, values.add(pos * stride + chunk * 4).cast());
        } else {
            zero16(dst);
        }
        m += 1;
    }
    let weights = buffer + (MROWS * MVROW) as u32;
    let mut j = 0;
    while j < 6 {
        let e = tid + 128 * j;
        let (r, i) = (e / MROWS, e % MROWS);
        let at = weights + ((r * MPROW + i) * 4) as u32;
        if r < group && start + i < visible {
            copy4(at, probabilities.add((first_row + r) * p.stride as usize + start + i).cast());
        } else {
            store(at, 0.0);
        }
        j += 1;
    }
    commit();
    wait();
    sync();
    let rem = visible - start;
    let ksteps = ((if rem < MROWS { rem } else { MROWS }) + TILE - 1) / TILE;
    let tile = split_tile(&p, token, key_head, slice);
    let mut kk = 0;
    while kk < ksteps {
        let xa = load4(weights + ((g * MPROW + 16 * kk + 4 * t) * 4) as u32).map(f32::from_bits);
        let xb = if g < 4 { load4(weights + (((g + 8) * MPROW + 16 * kk + 4 * t) * 4) as u32).map(f32::from_bits) } else { [0.0; 4] };
        let (a01, b01, a23, b23) = (split(xa[0], xa[1]), split(xb[0], xb[1]), split(xa[2], xa[3]), split(xb[2], xb[3]));
        let a = [[a01[0], b01[0], a23[0], b23[0]], [a01[1], b01[1], a23[1], b23[1]], [a01[2], b01[2], a23[2], b23[2]]];
        let (mut v, mut s, c) = ([[0.0f32; 2]; 4], 0, 8 * (warp >> 1) + g);
        while s < 4 {
            let row = 16 * kk + 4 * t + s;
            v[s] = load2(buffer + (row * MVROW + ((c ^ (((row >> 2) & 3) << 1)) << 4) + 8 * (warp & 1)) as u32).map(f32::from_bits);
            s += 1;
        }
        let k = stage * 4 + kk;
        let mut i = 0;
        while i < 2 {
            let (b0, b1) = (split(v[0][i], v[1][i]), split(v[2][i], v[3][i]));
            let mut acc = [0.0f32; 4];
            acc = mma(a[2], b0[0], b1[0], acc);
            acc = mma(a[1], b0[1], b1[1], acc);
            acc = mma(a[0], b0[2], b1[2], acc);
            acc = mma(a[1], b0[0], b1[0], acc);
            acc = mma(a[0], b0[1], b1[1], acc);
            acc = mma(a[0], b0[0], b1[0], acc);
            let base = (tile + k) * 16 + g;
            let local = 32 * (warp >> 1) + 8 * t + 2 * (warp & 1) + i;
            *partials.add(base * 64 + local) = acc[0];
            *partials.add(base * 64 + local + 4) = acc[1];
            *partials.add((base + 8) * 64 + local) = acc[2];
            *partials.add((base + 8) * 64 + local + 4) = acc[3];
            i += 1;
        }
        kk += 1;
    }
}
#[inline(always)]
unsafe fn values_kr(p: QsaAttention, partials: *const f32) {
    let tid = _thread_idx_x() as usize;
    let tokens = p.tokens as usize;
    let block = _block_idx_x() as usize;
    if block >= tokens * p.key_heads as usize * 4 { return; }
    let (token, rest) = (block % tokens, block / tokens);
    let (key_head, slice) = (rest / 4, rest % 4);
    let qpk = p.queries_per_key as usize;
    let (lo, hi) = ((key_head + p.key_start as usize) * qpk, (key_head + p.key_start as usize + 1) * qpk);
    let first = if lo > p.query_start as usize { lo } else { p.query_start as usize };
    let last = (p.query_start + p.query_heads) as usize;
    let end = if hi < last { hi } else { last };
    if first >= end { return; }
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    let ksteps = (selected_length(p.past as usize + token + 1) + TILE - 1) / TILE;
    let tile = split_tile(&p, token, key_head, slice);
    let (denominators, gates, output) = (p.denominators as *const f32, p.query_gate as *const f32, p.output as *mut f32);
    let mut e = tid;
    while e < group * 64 {
        let (r, local) = (e / 64, e % 64);
        let mut out = 0.0f32;
        let mut k = 0;
        while k < ksteps { out += *partials.add(((tile + k) * 16 + r) * 64 + local); k += 1; }
        let (row, feature) = (first_row + r, 64 * slice + local);
        let gate = output_gate(*gates.add(row * 512 + 256 + feature), 0);
        *output.add(row * 256 + feature) = (out / *denominators.add(row)) * gate;
        e += 256;
    }
}
/// Phase A: block 128, grid (tokens * local KV heads * 4, ceil(2051 / 64)); static shared 20,224 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc_k(mut p: QsaAttention, cursor: *const u32, partials: *mut f32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    values_k(p, partials);
}
/// Phase B: block 256, grid (tokens * local KV heads * 4).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_tc_kr(mut p: QsaAttention, cursor: *const u32, partials: *const f32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    values_kr(p, partials);
}

/// Index scores on tensor cores (VADE_QSA_TC switch, flips with the attention
/// entries): S[t][b] = (sum over heads 0..3 of relu(q_th . k_b)) / sqrt(128)
/// for complete groups b < (past + t + 1) / 4 of rows with more than 512.
/// Three BF16 planes (six products, no operand bit lost). Order per score
/// and head: a big chain of the eight hh K16 steps and a small chain of the
/// other products (step, product) from 0, dot = big + small; relu and the
/// head sum ((((0 + r0) + r1) + r2) + r3) as main, then the division.
/// A CTA owns 32 groups (four N8 tiles), split once into BF16 planes in
/// shared memory (h/m 16 KB + l 8 KB, swizzled, conflict-free); queries are
/// read one K16 step ahead. Prompt shape (WIDE): 64 tokens, warp = 16
/// tokens, heads in a loop. Decode / verify / batch shape: 16 tokens, warp =
/// head, the four relu values summed in head order through shared memory.
/// Both shapes and every grid give the same bytes.
const IGROUPS: usize = 32;
const IPLANES: usize = IGROUPS * 512;
#[inline(always)]
unsafe fn store4u(p: u32, v: [u32; 4]) {
    asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) p, a = in(reg32) v[0], b = in(reg32) v[1],
        c = in(reg32) v[2], d = in(reg32) v[3]);
}
#[inline(always)]
unsafe fn store2u(p: u32, v: [u32; 2]) {
    asm!("st.shared.v2.b32 [{p}], {{{a}, {b}}};", p = in(reg32) p, a = in(reg32) v[0], b = in(reg32) v[1]);
}
/// A loop bound the compilers cannot see through (keeps the K16 loop rolled:
/// unrolled, ptxas hoists every key fragment and spills).
#[inline(always)]
unsafe fn opaque(n: u32) -> usize {
    let v: u32;
    asm!("mov.u32 {v}, {n};", v = out(reg32) v, n = in(reg32) n);
    v as usize
}
#[inline(always)]
unsafe fn index_tc<const WIDE: bool>(p: QsaIndex, bx: usize, by: usize) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let (tokens, past, blocks) = (p.tokens as usize, p.past as usize, p.blocks as usize);
    let rows = if WIDE { 64 } else { 16 };
    let (b0, t0) = (bx * IGROUPS, by * rows);
    if t0 >= tokens { return; }
    let reach = (past + (t0 + rows).min(tokens)) / 4;
    if reach <= 512 || b0 >= reach { return; }
    let hm = shared::<{ IPLANES + IPLANES / 2 }>();
    let lo = hm + IPLANES as u32;
    // Planes: (row, K16 step s, lane t) holds features 16s + 4t .. + 3 as
    // B fragments b0 = (f, f + 1), b1 = (f + 2, f + 3): h/m as one 16-byte
    // entry [h0, h1, m0, m1], l as an 8-byte entry [l0, l1].
    let pooled = p.pooled as *const f32;
    let mut x = [[0.0f32; 4]; 8];
    eight!(M, { if b0 + warp + 4 * M < reach { x[M] = global4(pooled.add((b0 + warp + 4 * M) * 512 + lane * 4)); } });
    eight!(M, {
        let row = warp + 4 * M;
        let (k0, k1) = (split(x[M][0], x[M][1]), split(x[M][2], x[M][3]));
        store4u(hm + (row * 512 + ((lane ^ ((row & 1) << 2)) << 4)) as u32, [k0[0], k1[0], k0[1], k1[1]]);
        store2u(lo + (row * 256 + ((lane ^ ((row & 3) << 2)) << 3)) as u32, [k0[2], k1[2]]);
    });
    let (ta, tb) = if WIDE { (t0 + 16 * warp + g, t0 + 16 * warp + g + 8) } else { (t0 + g, t0 + g + 8) };
    let (la, lb) = (ta < tokens, tb < tokens);
    let q = p.query as *const f32;
    let (qa, qb) = (q.wrapping_add(ta * 512 + 4 * t), q.wrapping_add(tb * 512 + 4 * t));
    let fetch = |at: usize| -> [[f32; 4]; 2] {
        [if la { global4(qa.add(at)) } else { [0.0; 4] }, if lb { global4(qb.add(at)) } else { [0.0; 4] }]
    };
    sync();
    let (mut h, heads) = if WIDE { (0, opaque(4)) } else { (warp, warp + 1) };
    let steps = opaque(8);
    let mut score = [[0.0f32; 4]; 4];
    while h < heads {
        let mut big = [[0.0f32; 4]; 4];
        let mut small = [[0.0f32; 4]; 4];
        let mut next = fetch(h * 128);
        let mut s = 0;
        while s < steps {
            let [xa, xb] = next;
            if s + 1 < steps { next = fetch(h * 128 + 16 * (s + 1)); }
            let (a01, a23, b01, b23) = (split(xa[0], xa[1]), split(xa[2], xa[3]), split(xb[0], xb[1]), split(xb[2], xb[3]));
            let mut a = [[0u32; 4]; 3];
            three!(P, { a[P] = [a01[P], b01[P], a23[P], b23[P]]; });
            let c = 4 * s + t;
            four!(N, {
                let row = 8 * N + g;
                let k = load4(hm + (row * 512 + ((c ^ ((row & 1) << 2)) << 4)) as u32);
                let l = load2(lo + (row * 256 + ((c ^ ((row & 3) << 2)) << 3)) as u32);
                big[N] = mma(a[0], k[0], k[1], big[N]);
                small[N] = mma(a[0], k[2], k[3], small[N]);
                small[N] = mma(a[1], k[0], k[1], small[N]);
                small[N] = mma(a[0], l[0], l[1], small[N]);
                small[N] = mma(a[1], k[2], k[3], small[N]);
                small[N] = mma(a[2], k[0], k[1], small[N]);
            });
            s += 1;
        }
        four!(N, { four!(E, {
            let r = (big[N][E] + small[N][E]).max(0.0);
            if WIDE { score[N][E] += r; } else { score[N][E] = r; }
        }); });
        h += 1;
    }
    if !WIDE {
        // Head sum in main's order: warp w holds relu of head w for the same fragment.
        sync();
        four!(N, { four!(E, { store(hm + (((warp * 16 + 4 * N + E) * 32 + lane) * 4) as u32, score[N][E]); }); });
        sync();
        if warp != 0 { return; }
        four!(N, { four!(E, {
            let mut sum = 0.0f32;
            four!(H, { sum += load(hm + (((H * 16 + 4 * N + E) * 32 + lane) * 4) as u32); });
            score[N][E] = sum;
        }); });
    }
    // C fragment: tokens ta (E 0, 1) and tb (E 2, 3), groups 2t and 2t + 1 of each N8 tile.
    let out = p.scores as *mut f32;
    two!(HALF, {
        let token = ta + 8 * HALF;
        let visible = (past + token + 1) / 4;
        if token < tokens && visible > 512 {
            four!(N, {
                two!(C, {
                    let b = b0 + 8 * N + 2 * t + C;
                    if b < visible { *out.add(token * blocks + b) = score[N][2 * HALF + C] / crate::layer::sqrt(128.0); }
                });
            });
        }
    });
}

/// Prompt rows (tokens >= 64). Block 128, grid (ceil(groups / 32),
/// ceil(tokens / 64)); static shared 24 KB.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_tc(mut p: QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    index_tc::<true>(p, _block_idx_x() as usize, _block_idx_y() as usize);
}
/// Decode / verify rows (tokens < 64), same bytes. Block 128, grid
/// (ceil(groups / 32), ceil(tokens / 16)).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_tc_m(mut p: QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    index_tc::<false>(p, _block_idx_x() as usize, _block_idx_y() as usize);
}
/// Sessions in grid-y (one token each), grid x = ceil(groups / 32).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_tc_batch(p: QsaBatch) {
    let row = _block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    index_tc::<false>(batch_index(r), _block_idx_x() as usize, 0);
}

/// flash_qsa_select with 1024 threads (decode, verify and pooled rows; same IDs).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_select_w(mut p: QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    crate::flash_qsa::index_select_n::<1024, 1168>(p);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_select_w_batch(p: QsaBatch) {
    let row = _block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    crate::flash_qsa::index_select_n::<1024, 1168>(batch_index(r));
}
