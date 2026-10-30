//! Order-changed Flash-Next GDN scan, chunk form (host switch
//! VADE_GDN_CHUNK, default ON, 0 = off). Exactness contract: chunks of 64 rows at
//! absolute positions (0, 64, 128 ...); the state S changes only when a chunk's
//! row 63 is accepted; the open chunk lives in a tail after S (key rows, v' rows,
//! gamma); a call may start and stop at any row and gives the same bytes as any
//! other cut of the same rows. FP32 throughout, every sum one FMA chain in
//! ascending order from +0, no atomics, no masked terms inside a sum.
//! REDUCTION-PLAN (rule 13:25):
//!   MMA shape: none; every product is an FP32 FMA on the CUDA cores.
//!   Pieces: chunks of 64 rows at absolute positions (row p is in chunk p / 64,
//!   at index p % 64); host pieces of <= 8,192 rows (gdn.rs CHUNK_PIECE) also start at absolute
//!   positions; a call of <= CHUNK_ROWS_LIMIT rows runs the rows kernel (the
//!   same row function). All of it depends only on positions and the call's
//!   row count: never on batch placement, occupancy, cache state or the Spark.
//!   Per-piece order: the sums below, each one FMA chain from +0 in ascending
//!   d or j; one CTA owns one value head, so no sum is split across CTAs.
//!   Merge tree (named "chunk carry"): the only carry between chunks is S,
//!   S_{c+1} = fma(S_c, e_63, sum_j ...) applied in ascending chunk order; the
//!   open chunk's tail rows carry between calls. No atomics, no arrival-order
//!   sums. The fused output gate uses flash_gdn_gate's fixed warp butterfly.
//! Canonical arithmetic for value head h, chunk rows i, j in 0..64 (form DS):
//!   gamma_i = gamma_{i-1} + g_i (gamma_{-1} = +0) ; e_i = exp(gamma_i)
//!   kk_ij = sum_d k_i[d] k_j[d] ; qk_ij = sum_d q_i[d] k_j[d]       (d ascending)
//!   A_ij = (kk_ij * exp(gamma_i - gamma_j)) * beta_i                (j < i)
//!   P_ij = qk_ij * exp(gamma_i - gamma_j)                           (j <= i)
//!   ks_i = sum_d k_i[d] S[d][v] ; qs_i = sum_d q_i[d] S[d][v]       (d ascending)
//!   r_i = fma(-e_i, ks_i, v_i) * beta_i
//!   v'_i = r_i, then v'_i = fma(-A_ij, v'_j, v'_i) for j = 0 .. i-1 ascending
//!   o_i = fma(e_i, qs_i, sum_{j<=i} P_ij v'_j)                     (j ascending)
//!   row 63: S' = fma(S, e_63, sum_{j<64} (k_j[d] * exp(gamma_63 - gamma_j)) v'_j)
//! State (gdn_params::chunk_state_floats): S [heads][128][128], then the open
//! chunk's key rows [keys][64][128], v' rows [heads][64][128], gamma [heads][64];
//! it holds base % 64 rows (base = absolute position of the call's first row).
use crate::gdn_params::GdnChunkNext;
use crate::gdn_params::GdnDecodeRow;
use crate::layer::{exp, sqrt, sum};
use crate::math::bf16_to_f32 as bf16;
use core::arch::asm;
use core::arch::nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x};
/// Expands `body` for X = 0, 1, 2, 3 as constants, so register arrays indexed
/// by X stay in registers even when LLVM would not unroll a large loop.
macro_rules! each4 {
    ($x:ident => $body:block) => {
        { const $x: usize = 0; $body } { const $x: usize = 1; $body }
        { const $x: usize = 2; $body } { const $x: usize = 3; $body }
    };
}
const C: usize = 64;
const D: usize = 128;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.shfl.sync.idx.i32"]
    fn shfl_idx(mask: u32, value: u32, lane: u32, clamp: u32) -> u32;
}

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 { core::f32::math::mul_add(a, b, c) }
/// flash_gdn's sigmoid (the conv prepare's SiLU), same ops.
#[inline(always)]
unsafe fn sigmoid(value: f32) -> f32 { 1.0 / (1.0 + exp(-value)) }
#[inline(always)]
unsafe fn lds(p: u32) -> f32 {
    let v: f32;
    asm!("ld.shared.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p, options(nostack, readonly, pure));
    v
}
#[inline(always)]
unsafe fn lds4(p: u32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    asm!("ld.shared.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];", a = out(reg32) a, b = out(reg32) b,
        c = out(reg32) c, d = out(reg32) d, p = in(reg32) p, options(nostack, readonly, pure));
    [a, b, c, d]
}
#[inline(always)]
unsafe fn sts(p: u32, v: f32) {
    asm!("st.shared.f32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v, options(nostack));
}
#[inline(always)]
unsafe fn sts4(p: u32, v: [f32; 4]) {
    asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) p, a = in(reg32) v[0],
        b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3], options(nostack));
}
/// Global 16-byte load; `p` must be 16-byte aligned.
#[inline(always)]
unsafe fn ldg4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];", a = out(reg32) a, b = out(reg32) b,
        c = out(reg32) c, d = out(reg32) d, p = in(reg64) p, options(nostack, readonly, pure));
    [a, b, c, d]
}
#[inline(always)]
unsafe fn stg4(p: *mut f32, v: [f32; 4]) {
    asm!("st.global.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg64) p, a = in(reg32) v[0],
        b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3], options(nostack));
}
#[inline(always)]
unsafe fn prefetch_l2(p: *const f32) {
    asm!("prefetch.global.L2 [{p}];", p = in(reg64) p, options(nostack, readonly));
}
/// Convergent barrier intrinsic (an asm barrier may be duplicated into divergent paths).
#[inline(always)]
unsafe fn sync() { core::arch::nvptx::_syncthreads(); }
/// Identity the optimizer cannot see through: a per-chunk copy of a thread's
/// shared base stops LICM from hoisting hundreds of constant-offset addresses
/// out of the chunk loop (they were spilled).
#[inline(always)]
fn opaque(x: u32) -> u32 {
    let y: u32;
    unsafe { asm!("mov.b32 {y}, {x};", y = out(reg32) y, x = in(reg32) x, options(nostack, nomem)); }
    y
}

/// Rows of call chunk c: [lo, hi) are the call's rows c*64 + i - n0; rows below
/// lo (first chunk only) are the open tail carried in the state.
#[inline(always)]
fn span(c: usize, n0: usize, count: usize) -> (usize, usize) {
    (if c == 0 { n0 } else { 0 }, core::cmp::min(C, n0 + count - c * C))
}
/// Packed P block (floats per chunk and value head): row j keeps the 4-column
/// groups g >= j / 4, the only ones the scan's o sums read.
const PK: usize = 16 * 136;
/// Float offset of P row j inside a packed block (its first column is j / 4 * 4).
#[inline(always)]
fn packed_row(j: usize) -> usize {
    let (b, r) = (j / 4, j % 4);
    16 * (b * (33 - b) / 2) + 4 * r * (16 - b)
}
/// Float offsets of the tail's key rows, v' rows and gamma inside the state.
#[inline(always)]
fn tail(heads: usize, keys: usize) -> (usize, usize, usize) {
    let k = heads * D * D;
    let w = k + keys * C * D;
    (k, w, w + heads * C * D)
}

// Prep shared layout (bytes): staging Kt/Qt [32][68] overlaps KK/QK [64][65];
// gamma and beta [3][64] after them.
const ST: usize = 68;
const SQ: usize = 65;
const PREP_KK: u32 = 0;
const PREP_QK: u32 = (C * SQ * 4) as u32;
const PREP_G: u32 = (2 * C * SQ * 4) as u32;
const PREP_B: u32 = PREP_G + (3 * C * 4) as u32;
const PREP_BYTES: usize = 2 * C * SQ * 4 + 6 * C * 4;

/// One CTA (256 threads) per (call chunk, key head): kk/qk once for the key
/// group, then per value head gamma, and once per key group qk [j][i] (packed, see PK;
/// the scan forms P_ij = qk_ij * exp(gamma_i - gamma_j)) and kk_mj
/// (j < m) as [j][m / 16][m % 4][m / 4 % 4] (workspace `a`; the scan turns it
/// into A_mj = (kk_mj * exp(gamma_m - gamma_j)) * beta_m per value head).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_chunk_prep(p: GdnChunkNext) {
    // The scan (a programmatic launch) may take SMs as this grid drains; it
    // waits (griddepcontrol) before its first read of these outputs.
    asm!("griddepcontrol.launch_dependents;", options(nostack));
    let sh = crate::ptx::shared::<PREP_BYTES>();
    let tid = _thread_idx_x() as usize;
    let c = _block_idx_x() as usize;
    let kh = _block_idx_y() as usize;
    let (heads, keys) = (p.heads as usize, p.keys as usize);
    if keys == 0 || heads < keys { return; }
    let repeat = heads / keys;
    if repeat == 0 || repeat > 3 { return; }
    let h0 = kh * repeat;
    let hd = heads * D;
    let count = p.count as usize;
    let n0 = p.base as usize % C;
    if c * C >= n0 + count { return; }
    let (lo, hi) = span(c, n0, count);
    let first = c * C;
    let (q, k) = (p.q as *const f32, p.k as *const f32);
    let state = (*(p.rows as *const GdnDecodeRow)).state as *const f32;
    let (tk_at, _, tg_at) = tail(heads, keys);
    let (tk, tg) = (state.add(tk_at), state.add(tg_at));
    if p.ga != 0 {
        // Fused prepare: this chunk's call rows of q and k (key head kh, on
        // value head h0) and beta/g of heads h0 .. h0 + repeat, with
        // flash_gdn_conv_prepare_tokens_kh's functions and lane layout (lane
        // holds features lane + 32 f; the same squares order and warp sum).
        let lane = tid % 32;
        let channels = p.channels as usize;
        let (x, w, history) = (p.x as *const f32, p.w as *const u16, p.history as *const f32);
        let mut task = tid / 32;
        while task < 2 * C {
            let (arr, i) = (task / C, task % C);
            if i >= lo && i < hi {
                let row = first + i - n0;
                let token = p.x_start as usize + row;
                let base = if arr == 0 { kh } else { keys + kh } * D;
                let conv = |f: usize| crate::flash_gdn::convolution(x, w, history, base + lane + 32 * f, token, channels, 0);
                let out = [conv(0), conv(1), conv(2), conv(3)];
                let mut squares = 0.0;
                for f in 0..4 { squares += out[f] * out[f]; }
                let scale = 1.0 / sqrt(sum(squares) + 1e-6);
                let dst = (if arr == 0 { p.q } else { p.k }) as *mut f32;
                for f in 0..4 {
                    *dst.add(row * hd + h0 * D + lane + 32 * f) =
                        if arr == 0 { (out[f] * scale) * (1.0 / sqrt(128.0)) } else { out[f] * scale };
                }
            }
            task += 8;
        }
        if tid < repeat * C {
            let (r, i) = (tid / C, tid % C);
            let (mut learning, mut decay) = (0.0f32, 0.0f32);
            if i >= lo && i < hi {
                let at = (first + i - n0) * heads + h0 + r;
                (learning, decay) = crate::flash_gdn::gate_values(*(p.ga as *const f32).add(at), *(p.gb as *const f32).add(at),
                    p.alog as *const u16, p.dt as *const u16, h0 + r, 0);
                *(p.beta as *mut f32).add(at) = learning;
                *(p.g as *mut f32).add(at) = decay;
            }
            // The same words into the beta/g staging (zero off [lo, hi)): no global round trip.
            sts(sh + PREP_B + (tid * 4) as u32, learning);
            sts(sh + PREP_G + (tid * 4) as u32, decay);
        }
        // Block-scope visibility of these global words before the reads below.
        sync();
    }
    if tid < repeat * C && p.ga == 0 {
        let (r, i) = (tid / C, tid % C);
        let live = i >= lo && i < hi;
        let at = if live { (first + i - n0) * heads + h0 + r } else { 0 };
        sts(sh + PREP_B + (tid * 4) as u32, if live { *(p.beta as *const f32).add(at) } else { 0.0 });
        sts(sh + PREP_G + (tid * 4) as u32, if live { *(p.g as *const f32).add(at) } else { 0.0 });
    }
    // The first chunk's tail key rows go to the k0 workspace: the scan's chunk
    // close reads them there while it rewrites the tail.
    if c == 0 {
        let mut idx = tid;
        while idx < lo * 32 {
            let at = (kh * C + idx / 32) * D + idx % 32 * 4;
            stg4((p.k0 as *mut f32).add(at), ldg4(tk.add(at)));
            idx += 256;
        }
    }
    let (a, b) = (tid / 16, tid % 16);
    let (i0, j0) = (a * 4, b * 4);
    let mut kk = [[0.0f32; 4]; 4];
    let mut qk = [[0.0f32; 4]; 4];
    let kt = sh;
    let qt = sh + (32 * ST * 4) as u32;
    // Slice s + 32's words are requested before slice s's products (double-buffered
    // through registers, like the scan's K/Q staging).
    let load = |s: usize| -> [[f32; 4]; 4] {
        core::array::from_fn(|u| {
            let idx = tid + u * 256;
            let (arr, rem) = (idx / 512, idx % 512);
            // d group fastest: eight threads read one row's 128 contiguous bytes.
            let (i, dg) = (rem / 8, rem % 8);
            if i < lo {
                if arr == 0 { ldg4(tk.add((kh * C + i) * D + s + dg * 4)) } else { [0.0; 4] }
            } else if i < hi {
                let src = if arr == 0 { k } else { q };
                ldg4(src.add((first + i - n0) * hd + h0 * D + s + dg * 4))
            } else { [0.0; 4] }
        })
    };
    let mut next = load(0);
    let mut s = 0;
    while s < D {
        let cur = next;
        for u in 0..4 {
            let idx = tid + u * 256;
            let (arr, rem) = (idx / 512, idx % 512);
            let (i, dg) = (rem / 8, rem % 8);
            let base = if arr == 0 { kt } else { qt };
            for x in 0..4 { sts(base + (((dg * 4 + x) * ST + i) * 4) as u32, cur[u][x]); }
        }
        sync();
        if s + 32 < D { next = load(s + 32); }
        if a >= b {
            for dd in 0..32 {
                let ki = lds4(kt + ((dd * ST + i0) * 4) as u32);
                let kj = lds4(kt + ((dd * ST + j0) * 4) as u32);
                let qi = lds4(qt + ((dd * ST + i0) * 4) as u32);
                for x in 0..4 {
                    for y in 0..4 {
                        kk[x][y] = fma(ki[x], kj[y], kk[x][y]);
                        qk[x][y] = fma(qi[x], kj[y], qk[x][y]);
                    }
                }
            }
        }
        sync();
        s += 32;
    }
    if a >= b {
        for x in 0..4 {
            for y in 0..4 {
                sts(sh + PREP_KK + (((i0 + x) * SQ + j0 + y) * 4) as u32, kk[x][y]);
                sts(sh + PREP_QK + (((i0 + x) * SQ + j0 + y) * 4) as u32, qk[x][y]);
            }
        }
    }
    // gamma: tail rows from the state, then one ascending sum per value head.
    if tid % 32 == 0 && tid / 32 < repeat {
        let r = tid / 32;
        let h = h0 + r;
        let mut acc = if lo > 0 { *tg.add(h * C + lo - 1) } else { 0.0 };
        // g read eight at a time ahead of the sums (the add chain no longer waits on each load).
        let mut i = 0;
        while i < C {
            let g8 = [lds4(sh + PREP_G + ((r * C + i) * 4) as u32), lds4(sh + PREP_G + ((r * C + i + 4) * 4) as u32)];
            for half in 0..2 {
                each4!(V => {
                    let x = i + 4 * half + V;
                    let value = if x < lo { *tg.add(h * C + x) } else { acc = acc + g8[half][V]; acc };
                    sts(sh + PREP_G + ((r * C + x) * 4) as u32, value);
                    *(p.gamma as *mut f32).add((c * heads + h) * C + x) = value;
                });
            }
            i += 8;
        }
    }
    sync();
    // qk once per key group, packed like P ([j][i], row j keeps the 4-column
    // groups >= j / 4): the scan forms P_ij = qk_ij * exp(gamma_i - gamma_j).
    let mut idx = tid;
    while idx < C * C {
        let (j, x) = (idx / C, idx % C);
        if x / 4 >= j / 4 { *(p.p as *mut f32).add((c * keys + kh) * PK + packed_row(j) + x - j / 4 * 4) = lds(sh + PREP_QK + ((x * SQ + j) * 4) as u32); }
        idx += 256;
    }
    let mut idx = tid;
    while idx < C * C {
        let (j, x) = (idx / C, idx % C);
        let m = x / 4 % 4 + 4 * (x / 16 * 4 + x % 4);
        let kv = if j < m { lds(sh + PREP_KK + ((m * SQ + j) * 4) as u32) } else { 0.0 };
        *(p.a as *mut f32).add((c * keys + kh) * C * C + idx) = kv;
        idx += 256;
    }
}

// Scan shared layout (bytes): S [128][128], R [64][128] (staging during the
// KS/QS product, then r and v'), then E, beta, F and gamma [64] each.
const SCAN_S: u32 = 0;
const SCAN_R: u32 = (D * D * 4) as u32;
const SCAN_E: u32 = SCAN_R + (C * D * 4) as u32;
const SCAN_BETA: u32 = SCAN_E + (C * 4) as u32;
const SCAN_F: u32 = SCAN_BETA + (C * 4) as u32;
const SCAN_G: u32 = SCAN_F + (C * 4) as u32;
/// The state pointer, kept in shared memory: a register copy lives across the
/// chunk loop and was spilled to local memory (an L2 round trip per chunk).
const SCAN_PTR: u32 = SCAN_G + (C * 4) as u32;
/// Dynamic shared bytes for flash_next_gdn_chunk_scan (host sets the attribute).
pub const SCAN_BYTES: u32 = SCAN_PTR + 16;
const _: () = assert!(SCAN_BYTES == crate::gdn_params::CHUNK_SCAN_BYTES && SCAN_BYTES <= 101376);
const SLICE: u32 = (2 * 16 * ST * 4) as u32;
/// Column swizzle of the scan's r / v' rows in R: row i's column v sits at
/// v ^ ((i % 4) * 8), so the column-wise substitution loads and stores (rows q4 + 4t
/// of one column per lane) hit four different banks; 4-aligned groups stay
/// contiguous, so the row-wise lds4 / sts4 reads keep their shape.
#[inline(always)]
const fn rsw(i: usize, v: usize) -> usize { v ^ ((i & 3) << 3) }

/// Substitution step J: every row m > J of this thread's column (rows q + 4t
/// inside [lo, hi)) takes v'_m = fma(-A_mJ, v'_J, v'_m); v'_J comes from the
/// lane that owns row J (lanes 4c .. 4c+3 share column c). A sits in shared
/// memory as [j][t / 4][q][t % 4] (rows m = q + 4t); `a` = A + q * 16 bytes.
#[inline(always)]
unsafe fn substitute<const J: usize>(rows: &mut [f32; 16], a: u32, lane: u32, q: usize, lo: usize, hi: usize) {
    let owner = (lane & !3) | (J as u32 & 3);
    // fma(a, -w, v) == fma(-a, w, v) exactly: one negation per step, not per term.
    let w = -f32::from_bits(shfl_idx(u32::MAX, rows[J / 4].to_bits(), owner, 31));
    for g in J / 16..4 {
        let a4 = lds4(a + ((J * C + g * 16) * 4) as u32);
        for u in 0..4 {
            let t = g * 4 + u;
            let m = q + 4 * t;
            if t >= J / 4 && m > J && m >= lo && m < hi { rows[t] = fma(a4[u], w, rows[t]); }
        }
    }
}
/// The same step for a whole chunk (lo = 0, hi = 64): rows past J/4 always
/// take the term, row J/4 only where m = q + J - J % 4 > J.
#[inline(always)]
unsafe fn substitute_full<const J: usize>(rows: &mut [f32; 16], a: u32, lane: u32, q: usize) {
    let owner = (lane & !3) | (J as u32 & 3);
    let w = -f32::from_bits(shfl_idx(u32::MAX, rows[J / 4].to_bits(), owner, 31));
    for g in J / 16..4 {
        let a4 = lds4(a + ((J * C + g * 16) * 4) as u32);
        for u in 0..4 {
            let t = g * 4 + u;
            if t > J / 4 || (t == J / 4 && q > J % 4) { rows[t] = fma(a4[u], w, rows[t]); }
        }
    }
}
macro_rules! substitute_rows {
    ($rows:ident, $a:ident, $lane:ident, $q:ident, $lo:ident, $hi:ident; $($j:literal)*) => {
        if $lo == 0 && $hi == C {
            $( substitute_full::<$j>(&mut $rows, $a, $lane, $q); )*
        } else {
            $( if $j + 1 < $hi { substitute::<$j>(&mut $rows, $a, $lane, $q, $lo, $hi); } )*
        }
    };
}

/// Close step j: u[z][y] += kd_j[d0 + z] * v'_j[v0 + y], kd_j from lane j % 32.
macro_rules! close_rows {
    ($kl:ident, $u:ident, $w_at:ident; $($j:literal)*) => {
        $( {
            let w = lds4($w_at[$j % 4] + ($j * D * 4) as u32);
            for z in 0..8 {
                let kd = f32::from_bits(shfl_idx(u32::MAX, $kl[$j / 32][z].to_bits(), ($j % 32) as u32, 31));
                for y in 0..4 { $u[z][y] = fma(kd, w[y], $u[z][y]); }
            }
        } )*
    };
}

/// KS/QS step dd (literal): ks/qs[x][y] += k/q[i0 + x][d] * S[d][v0 + y].
macro_rules! ks_qs_steps {
    ($kt:ident, $qt:ident, $s_at:ident, $ks:ident, $qs:ident; $($dd:literal)*) => {
        $( {
            let kv = lds4($kt + ($dd * ST * 4) as u32);
            let qv = lds4($qt + ($dd * ST * 4) as u32);
            let sv = lds4($s_at + ($dd * D * 4) as u32);
            for x in 0..4 {
                for y in 0..4 {
                    $ks[x][y] = fma(kv[x], sv[y], $ks[x][y]);
                    $qs[x][y] = fma(qv[x], sv[y], $qs[x][y]);
                }
            }
        } )*
    };
}

/// Chunk c's first global inputs for scan thread `tid` of value head h: its
/// four v rows and its word of the first K/Q staging slice (zeros off [lo, hi)).
#[inline(always)]
unsafe fn chunk_inputs<const CONV: bool>(p: &GdnChunkNext, c: usize, tid: usize, h: usize, repeat: usize, n0: usize, count: usize) -> ([[f32; 4]; 4], [f32; 4]) {
    (chunk_v::<CONV>(p, c, tid, h, n0, count), chunk_slice(p, c, tid, h, repeat, n0, count))
}
/// Chunk c's four v rows of thread `tid` (inputs of launches before prep).
#[inline(always)]
unsafe fn chunk_v<const CONV: bool>(p: &GdnChunkNext, c: usize, tid: usize, h: usize, n0: usize, count: usize) -> [[f32; 4]; 4] {
    let hd = p.heads as usize * D;
    let (lo, hi) = span(c, n0, count);
    let first = c * C;
    let (i0, v0) = (tid / 32 * 4, tid % 32 * 4);
    let mut vr = [[0.0f32; 4]; 4];
    if CONV {
        // v = SiLU of the causal 4-tap conv of the projection rows, the words
        // flash_gdn_conv_prepare_tokens writes at precision 0 (same products,
        // same sums from +0 in tap order, same sigmoid).
        let (x, w, history) = (p.x as *const f32, p.w as *const u16, p.history as *const f32);
        let channels = p.channels as usize;
        let ch = (2 * p.keys as usize + h) * D + v0;
        let mut wt = [[0.0f32; 4]; 4];
        for y in 0..4 { for tap in 0..4 { wt[y][tap] = bf16(*w.add((ch + y) * 4 + tap)); } }
        // Samples of x rows t0 - 3 .. t0 + 3, t0 = the x row of chunk row i0.
        let t0 = (p.x_start as usize + first + i0) as isize - n0 as isize;
        let end = (p.x_start as usize + count) as isize;
        let mut sm = [[0.0f32; 4]; 7];
        for r in 0..7 {
            let source = t0 + r as isize - 3;
            if source >= 0 && source < end {
                sm[r] = ldg4(x.add(source as usize * channels + ch));
            } else if (-3..0).contains(&source) {
                for y in 0..4 { sm[r][y] = *history.add((ch + y) * 4 + (source + 4) as usize); }
            }
        }
        for xr in 0..4 {
            let i = i0 + xr;
            if i >= lo && i < hi {
                for y in 0..4 {
                    let mut value = 0.0f32;
                    for tap in 0..4 { value += sm[xr + tap][y] * wt[y][tap]; }
                    vr[xr][y] = value * sigmoid(value);
                }
            }
        }
    } else {
        // The head's column base through `opaque`: ptxas 13 lowered the folded
        // p.v + h * 512 on its uniform path with the high address word dropped.
        let vh = (p.v as *const f32).add(opaque((h * D + v0) as u32) as usize);
        for x in 0..4 {
            let i = i0 + x;
            if i >= lo && i < hi { vr[x] = ldg4(vh.add((first + i - n0) * hd)); }
        }
    }
    vr
}
/// Thread `tid`'s word of chunk c's first K/Q staging slice (prep's q/k when fused).
#[inline(always)]
unsafe fn chunk_slice(p: &GdnChunkNext, c: usize, tid: usize, h: usize, repeat: usize, n0: usize, count: usize) -> [f32; 4] {
    let hd = p.heads as usize * D;
    let h0 = h / repeat * repeat;
    let (lo, hi) = span(c, n0, count);
    let first = c * C;
    let (arr, rem) = (tid / 256, tid % 256);
    let (i, dg) = (rem / 4, rem % 4);
    if i >= lo && i < hi {
        let src = (if arr == 0 { p.k } else { p.q }) as *const f32;
        ldg4(src.add((first + i - n0) * hd + h0 * D + dg * 4))
    } else { [0.0; 4] }
}

/// One CTA (512 threads) per value head; S stays in shared memory across the
/// call's chunks, which advance in ascending order. v rows from p.v.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_chunk_scan(p: GdnChunkNext) { chunk_scan::<false>(p) }
/// The same scan convolving its v rows from p.x (needs x, w, history, channels).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_chunk_scan_cv(p: GdnChunkNext) { chunk_scan::<true>(p) }
#[inline(always)]
unsafe fn chunk_scan<const CONV: bool>(p: GdnChunkNext) {
    if CONV && p.x == 0 { return; }
    let tid = _thread_idx_x() as usize;
    let h = _block_idx_x() as usize;
    let (heads, keys) = (p.heads as usize, p.keys as usize);
    if keys == 0 || heads < keys { return; }
    let repeat = heads / keys;
    if repeat == 0 { return; }
    let kh = h / repeat;
    let h0 = kh * repeat;
    let hd = heads * D;
    let count = p.count as usize;
    let n0 = p.base as usize % C;
    let state = (*(p.rows as *const GdnDecodeRow)).state as *mut f32;
    if tid == 0 {
        sts(SCAN_PTR, f32::from_bits(state as u64 as u32));
        sts(SCAN_PTR + 4, f32::from_bits((state as u64 >> 32) as u32));
    }
    // Read back per use from an opaque address (never hoisted, never spilled).
    let state_at = || -> *mut f32 {
        let at = opaque(SCAN_PTR);
        ((lds(at + 4).to_bits() as u64) << 32 | lds(at).to_bits() as u64) as *mut f32
    };
    let (tk_at, tw_at, tg_at) = tail(heads, keys);
    let (q, k, v) = (p.q as *const f32, p.k as *const f32, p.v as *const f32);
    for u in 0..8 {
        let idx = tid + u * 512;
        let (d, c4) = (idx / 32, idx % 32);
        sts4(SCAN_S + ((d * D + c4 * 4) * 4) as u32, ldg4(state.add(h * D * D + d * D + c4 * 4)));
    }
    // S and chunk 0's v rows come from launches before prep: read them while prep
    // drains, then wait (griddepcontrol; a no-op for a plain launch) for prep's
    // q/k/gates/gamma/kk/qk/k0 words. Nothing is written before the wait (prep
    // reads the tail).
    let v_rows = chunk_v::<CONV>(&p, 0, tid, h, n0, count);
    sync();
    asm!("griddepcontrol.wait;", options(nostack));
    let chunks = (n0 + count).div_ceil(C);
    // The next chunk's v rows and first staging slice are loaded before this
    // chunk's close, so their latency hides behind it.
    let mut carried = (v_rows, chunk_slice(&p, 0, tid, h, repeat, n0, count));
    // Chunk c's gamma_i, gamma_63 and beta_i for threads < 64, loaded one chunk
    // ahead (they gated the chunk start's first barrier).
    let gates_of = |c: usize| -> [f32; 3] {
        if tid >= C { return [0.0; 3]; }
        let (lo, hi) = span(c, n0, count);
        let gamma = (p.gamma as *const f32).add((c * heads + h) * C);
        let live = tid >= lo && tid < hi;
        [*gamma.add(tid), *gamma.add(C - 1), if live { *(p.beta as *const f32).add((c * C + tid - n0) * heads + h) } else { 0.0 }]
    };
    let mut gates = gates_of(0);
    for c in 0..chunks {
        // Per-chunk opaque thread and head ids: every global address below is
        // rebuilt from them each chunk, so LICM cannot hoist (and spill) dozens
        // of 64-bit addresses across the chunk loop.
        let tid = opaque(tid as u32) as usize;
        let h = opaque(h as u32) as usize;
        let (kh, h0) = (h / repeat, h / repeat * repeat);
        let (ty, tx) = (tid / 32, tid % 32);
        let (i0, v0, d0) = (ty * 4, tx * 4, ty * 8);
        let (col, q4, lane) = (tid / 4, tid % 4, (tid % 32) as u32);
        let (lo, hi) = span(c, n0, count);
        let first = c * C;
        // Per-chunk opaque shared bases (see `opaque`).
        let s_col = opaque(SCAN_S + (v0 * 4) as u32);
        let kq_row = opaque(SCAN_R + (i0 * 4) as u32);
        let r_tile = opaque(SCAN_R + ((i0 * D) * 4) as u32);
        let e_row = opaque(SCAN_E + (i0 * 4) as u32);
        let r_col = opaque(SCAN_R + ((q4 * D + rsw(q4, col)) * 4) as u32);
        let s_tile = opaque(SCAN_S + ((d0 * D + v0) * 4) as u32);
        let stage_at = {
            let (arr, rem) = (tid / 256, tid % 256);
            let (i, dg) = (rem / 4, rem % 4);
            opaque(SCAN_R + ((arr * 16 * ST + dg * 4 * ST + i) * 4) as u32)
        };
        if tid < C {
            let [gi, last, beta] = gates;
            sts(SCAN_E + (tid * 4) as u32, exp(gi));
            sts(SCAN_F + (tid * 4) as u32, exp(last - gi));
            sts(SCAN_G + (tid * 4) as u32, gi);
            sts(SCAN_BETA + (tid * 4) as u32, beta);
        }
        let (vr, first_slice) = carried;
        // KS and QS over d in 8 slices of 16, double-buffered through registers
        // (four threads per row slice: 64 contiguous bytes).
        let stage = |s: usize| -> [f32; 4] {
            let (arr, rem) = (tid / 256, tid % 256);
            let (i, dg) = (rem / 4, rem % 4);
            if i >= lo && i < hi {
                let src = if arr == 0 { k } else { q };
                ldg4(src.add((first + i - n0) * hd + h0 * D + s * 16 + dg * 4))
            } else { [0.0; 4] }
        };
        let mut ks = [[0.0f32; 4]; 4];
        let mut qs = [[0.0f32; 4]; 4];
        let mut next = first_slice;
        for s in 0..8 {
            let half = (s % 2) as u32 * SLICE;
            for x in 0..4 { sts(stage_at + half + (x * ST * 4) as u32, next[x]); }
            sync();
            if s < 7 { next = stage(s + 1); }
            let kt = kq_row + half;
            let qt = kt + (16 * ST * 4) as u32;
            // d steps fully unrolled (literal offsets, no loop counter).
            let s_at = s_col + (s * 16 * D * 4) as u32;
            ks_qs_steps!(kt, qt, s_at, ks, qs; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
        }
        // The key group's kk words for A, requested before the R rows (used after them).
        let ag = (p.a as *const f32).add((c * keys + kh) * C * C);
        let kkw = [ldg4(ag.add(tid * 4)), ldg4(ag.add((tid + 512) * 4))];
        sync();
        // R rows: the tail's v' below lo, r_i = fma(-e_i, ks_i, v_i) * beta_i on [lo, hi), zero above.
        for x in 0..4 {
            let i = i0 + x;
            let value = if i < lo {
                ldg4(state_at().add(tw_at + h * C * D + i * D + v0))
            } else if i < hi {
                let e = lds(e_row + (x * 4) as u32);
                let beta = lds(e_row + (C * 4 + x * 4) as u32);
                let mut r = [0.0f32; 4];
                for y in 0..4 { r[y] = fma(-e, ks[x][y], vr[x][y]) * beta; }
                r
            } else { [0.0; 4] };
            sts4(r_tile + ((x * D + rsw(x, v0)) * 4) as u32, value);
        }
        sync();
        // v' by forward substitution: 4 lanes per column, rows q4 + 4t each.
        let mut rows = [0.0f32; 16];
        for t in 0..16 { rows[t] = lds(r_col + (4 * t * D * 4) as u32); }
        sync();
        // A (16 KB) into the R region while the rows sit in registers: the key
        // group's kk_mj, times exp(gamma_m - gamma_j), times beta_m (prep's words).
        for u in 0..2 {
            let idx = tid + u * 512;
            let kk4 = kkw[u];
            let mut a4 = [0.0f32; 4];
            for e in 0..4 {
                let (j, x) = ((idx * 4 + e) / C, (idx * 4 + e) % C);
                let m = x / 4 % 4 + 4 * (x / 16 * 4 + x % 4);
                if m >= lo && m < hi && j < m {
                    let decay = exp(lds(SCAN_G + (m * 4) as u32) - lds(SCAN_G + (j * 4) as u32));
                    a4[e] = (kk4[e] * decay) * lds(SCAN_BETA + (m * 4) as u32);
                }
            }
            sts4(SCAN_R + (idx * 16) as u32, a4);
        }
        sync();
        // Warm L2 with the next chunk's rows (K, Q of the key group, V) and A, P
        // here, where the substitution leaves the memory queues idle (issued at
        // the chunk start they delayed that chunk's own first loads).
        if c + 1 < chunks {
            let next = (c + 1) * C + tid / 4 % C - n0;
            let line = (tid % 4) * 32;
            if next < count {
                let at = next * hd;
                if tid < 256 { prefetch_l2(k.add(at + h0 * D + line)); } else { prefetch_l2(q.add(at + h0 * D + line)); }
                // _cv reads the projection rows of its v section, not v.
                if tid < 256 {
                    if CONV { prefetch_l2((p.x as *const f32).add((p.x_start as usize + next) * p.channels as usize + (2 * keys + h) * D + line)); }
                    else { prefetch_l2(v.add(at + h * D + line)); }
                }
                if tid >= 256 && p.gated != 0 { prefetch_l2((p.z as *const f32).add(at + h * D + line)); }
            }
            if tid >= 256 {
                let lines = (tid - 256) * 32;
                let block = ((c + 1) * keys + kh) * PK;
                let kk = (c + 1) * keys * C * C + kh * C * C;
                if lines < C * C { prefetch_l2((p.a as *const f32).add(kk + lines)); }
                if lines < PK { prefetch_l2((p.p as *const f32).add(block + lines)); }
            }
        }
        // The o sums' qk rows (lane l: rows l and l + 32 at columns i0 .. i0 + 4),
        // requested before the substitution (used after it).
        let pt = (p.p as *const f32).add((c * keys + kh) * PK);
        let mut qkl = [[0.0f32; 4]; 2];
        for t in 0..2 {
            let j = lane as usize + 32 * t;
            if j < i0 + 4 { qkl[t] = ldg4(pt.add(packed_row(j) + i0 - j / 4 * 4)); }
        }
        let a = opaque(SCAN_R + (q4 * 16) as u32);
        substitute_rows!(rows, a, lane, q4, lo, hi;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62);
        sync();
        for t in 0..16 { sts(r_col + (4 * t * D * 4) as u32, rows[t]); }
        sync();
        // Fused gate inputs (gated != 0), loaded before the o sums so their
        // latency hides behind them: lane tx takes flash_gdn_gate's features
        // tx + 32 kq of its warp's rows i0..i0 + 3 and the norm weights.
        let gated = p.gated as *mut f32;
        let mut zg = [[0.0f32; 4]; 4];
        let mut wg = [0.0f32; 4];
        if !gated.is_null() {
            each4!(X => {
                let i = i0 + X;
                if i >= lo && i < hi {
                    let row = (first + i - n0) * hd + h * D;
                    each4!(K => { zg[X][K] = *(p.z as *const f32).add(row + tx + 32 * K); });
                }
            });
            each4!(K => { wg[K] = *(p.norm as *const f32).add(tx + 32 * K); });
        }
        // o_i = fma(e_i, qs_i, sum_{j<=i} P_ij v'_j). Every lane of a warp reads the
        // same words: lane l loads rows l and l + 32 (columns i0 .. i0 + 4) of the key
        // group's qk (stored [j][i], packed) and forms P_ij = qk_ij * exp(gamma_i -
        // gamma_j) on live rows i, j <= i (zero elsewhere); row j comes from lane j % 32.
        let mut pl = [[0.0f32; 4]; 2];
        for t in 0..2 {
            let j = lane as usize + 32 * t;
            if j < i0 + 4 {
                let qk4 = qkl[t];
                let gj = lds(SCAN_G + (j * 4) as u32);
                for x in 0..4 {
                    let i = i0 + x;
                    pl[t][x] = if i >= lo && i < hi && j <= i { qk4[x] * exp(lds(SCAN_G + (i * 4) as u32) - gj) } else { 0.0 };
                }
            }
        }
        let mut o = [[0.0f32; 4]; 4];
        for jb in (0..i0 + 4).step_by(4) {
            let p4: [[f32; 4]; 4] = core::array::from_fn(|z| {
                let src = if jb < 32 { pl[0] } else { pl[1] };
                let from = ((jb + z) % 32) as u32;
                core::array::from_fn(|x| f32::from_bits(shfl_idx(u32::MAX, src[x].to_bits(), from, 31)))
            });
            let w4: [[f32; 4]; 4] = core::array::from_fn(|z| lds4(SCAN_R + (((jb + z) * D + rsw(z, v0)) * 4) as u32));
            let diagonal = jb == i0;
            for z in 0..4 {
                for x in 0..4 {
                    if !diagonal || z <= x {
                        for y in 0..4 { o[x][y] = fma(p4[z][x], w4[z][y], o[x][y]); }
                    }
                }
            }
        }
        each4!(X => {
            let x = X;
            let i = i0 + x;
            if i >= lo && i < hi {
                let e = lds(e_row + (x * 4) as u32);
                let mut out = [0.0f32; 4];
                for y in 0..4 { out[y] = fma(e, qs[x][y], o[x][y]); }
                let row = (first + i - n0) * hd + h * D;
                if gated.is_null() {
                    stg4((p.out as *mut f32).add(row + v0), out);
                } else {
                    // flash_gdn_gate on this row (warp-uniform branch): lane tx
                    // gathers features tx + 32 kq from lane tx / 4 + 8 kq, then
                    // the same squares, warp sum, scale, weight and sigmoid.
                    let m = tx % 4;
                    let mut vals = [0.0f32; 4];
                    each4!(K => {
                        let src = (tx / 4 + 8 * K) as u32;
                        let t0 = f32::from_bits(shfl_idx(u32::MAX, out[0].to_bits(), src, 31));
                        let t1 = f32::from_bits(shfl_idx(u32::MAX, out[1].to_bits(), src, 31));
                        let t2 = f32::from_bits(shfl_idx(u32::MAX, out[2].to_bits(), src, 31));
                        let t3 = f32::from_bits(shfl_idx(u32::MAX, out[3].to_bits(), src, 31));
                        vals[K] = if m == 0 { t0 } else if m == 1 { t1 } else if m == 2 { t2 } else { t3 };
                    });
                    let mut squares = 0.0;
                    each4!(K => { squares += vals[K] * vals[K]; });
                    let scale = 1.0 / sqrt(sum(squares) / 128.0 + p.epsilon);
                    let mut gv = [0.0f32; 4];
                    each4!(K => {
                        let value = vals[K] * scale * wg[K];
                        gv[K] = value * (1.0 / (1.0 + exp(-zg[x][K]))) * 1.0;
                    });
                    if p.pk == 0 {
                        each4!(K => { *gated.add(row + tx + 32 * K) = gv[K]; });
                    } else {
                        // flash_bf16_input's words for this row: lanes tx & !1 and tx | 1
                        // both pack columns c, c + 1 (c = h * D + (tx & !1) + 32 K) at the
                        // A-fragment slot of (r, c); the even lane stores the hi plane, the
                        // odd lane the lo plane (one store per lane, all 32 lanes).
                        let r = p.pk_row0 as usize + first + i - n0;
                        let kt = hd / 16;
                        let plane = (p.pk_tokens as usize).div_ceil(16) * kt * 128;
                        let pk = p.pk as *mut u32;
                        let odd = tx % 2;
                        each4!(K => {
                            let other = f32::from_bits(shfl_idx(u32::MAX, gv[K].to_bits(), (tx ^ 1) as u32, 31));
                            let (a, b) = if odd == 0 { (gv[K], other) } else { (other, gv[K]) };
                            let c = h * D + (tx & !1) + 32 * K;
                            let (hi, lo) = crate::gemm::split2_pair(a, b);
                            let (rr, cc) = (r % 16, c % 16);
                            let at = (((r / 16) * kt + c / 16) * 32 + rr % 8 * 4 + cc % 8 / 2) * 4 + (rr / 8) + 2 * (cc / 8);
                            *pk.add(at + odd * plane) = if odd == 0 { hi } else { lo };
                        });
                    }
                }
            }
        });
        // Row 63 accepted: S' = fma(S, e_63, sum_j (k_j * f_j) v'_j); tail keys from k0.
        // The close's key cells are requested before the next chunk's inputs.
        let krow = |j: usize| -> *const f32 {
            if j < lo { (p.k0 as *const f32).add((kh * C + j) * D + d0) } else { k.add((first + j - n0) * hd + h0 * D + d0) }
        };
        let mut kraw = [[[0.0f32; 4]; 2]; 2];
        if hi == C {
            for t in 0..2 {
                let r = krow(lane as usize + 32 * t);
                kraw[t] = [ldg4(r), ldg4(r.add(4))];
            }
        }
        if c + 1 < chunks {
            carried = chunk_inputs::<CONV>(&p, c + 1, tid, h, repeat, n0, count);
            gates = gates_of(c + 1);
        }
        if hi == C {
            // Lane l holds the warp's cells d0 .. d0 + 8 of rows l and l + 32, scaled
            // by f_j; row j then comes from lane j % 32.
            let mut kl = [[0.0f32; 8]; 2];
            for t in 0..2 {
                let j = lane as usize + 32 * t;
                let [lo4, hi4] = kraw[t];
                let f = lds(SCAN_F + (j * 4) as u32);
                kl[t] = [lo4[0] * f, lo4[1] * f, lo4[2] * f, lo4[3] * f, hi4[0] * f, hi4[1] * f, hi4[2] * f, hi4[3] * f];
            }
            let mut u = [[0.0f32; 4]; 8];
            let w_at: [u32; 4] = core::array::from_fn(|k| opaque(SCAN_R + (rsw(k, v0) * 4) as u32));
            close_rows!(kl, u, w_at;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63);
            let decay = lds(SCAN_E + ((C - 1) * 4) as u32);
            for z in 0..8 {
                let at = s_tile + (z * D * 4) as u32;
                let old = lds4(at);
                let mut new = [0.0f32; 4];
                for y in 0..4 { new[y] = fma(old[y], decay, u[z][y]); }
                sts4(at, new);
            }
        }
        sync();
    }
    let state = state_at();
    let (tk, tw, tg) = (state.add(tk_at), state.add(tw_at + h * C * D), state.add(tg_at + h * C));
    if n0 + count >= C {
        for u in 0..8 {
            let idx = tid + u * 512;
            let (d, c4) = (idx / 32, idx % 32);
            stg4(state.add(h * D * D + d * D + c4 * 4), lds4(SCAN_S + ((d * D + c4 * 4) * 4) as u32));
        }
    }
    // The open chunk's new rows join the tail: v' and gamma per head, key rows
    // once per key group (by its first value head).
    let end = (n0 + count) % C;
    if end > 0 {
        let c = chunks - 1;
        let (lo, _) = span(c, n0, count);
        let first = c * C;
        let mut idx = tid;
        while idx < (end - lo) * 32 {
            let (i, x4) = (lo + idx / 32, idx % 32 * 4);
            stg4(tw.add(i * D + x4), lds4(SCAN_R + ((i * D + rsw(i, x4)) * 4) as u32));
            if h == h0 { stg4(tk.add((kh * C + i) * D + x4), ldg4(k.add((first + i - n0) * hd + h0 * D + x4))); }
            idx += 512;
        }
        if tid >= lo && tid < end { *tg.add(tid) = *(p.gamma as *const f32).add((c * heads + h) * C + tid); }
    }
}

/// Close step j for eight cells d .. d + 8 of one column: u[z] += kd[j][d + z] * v'_j.
macro_rules! close_cells {
    ($kd:ident, $wv:ident, $u:ident; $($j:literal)*) => {
        $( {
            let k0 = lds4($kd + ($j * D * 4) as u32);
            let k1 = lds4($kd + (($j * D + 4) * 4) as u32);
            let kd = [k0[0], k0[1], k0[2], k0[3], k1[0], k1[1], k1[2], k1[3]];
            for z in 0..8 { $u[z] = fma(kd[z], $wv[$j], $u[z]); }
        } )*
    };
}

// Decode slot shared layout (bytes): kd [64][128] (chunk close; the slot kernel
// also stages the tail keys there as [128][65] and streams S through it), q_i
// and k_i [128], A_ij and P_ij [64], f_j [64].
const SLOT_KD: u32 = 0;
const SLOT_Q: u32 = (D * (C + 1) * 4) as u32;
const SLOT_K: u32 = SLOT_Q + (D * 4) as u32;
const SLOT_A: u32 = SLOT_K + (D * 4) as u32;
const SLOT_P: u32 = SLOT_A + (C * 4) as u32;
const SLOT_F: u32 = SLOT_P + (C * 4) as u32;
const SLOT_BYTES: usize = SLOT_F as usize + C * 4;
const _: () = assert!(SLOT_BYTES as u32 == crate::gdn_params::CHUNK_ROW_BYTES);

/// `$body` once per d = 0 .. 127 with `$d` a literal (register arrays stay in
/// registers).
macro_rules! d128 {
    ($d:ident => $body:expr) => {
        d128!(@ $d => $body;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
            64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
            96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127)
    };
    (@ $d:ident => $body:expr; $($n:literal)*) => { $( { let $d: usize = $n; $body; } )* };
}

/// Small execute-path calls (MTP verify windows, one-row replays): the decode
/// row function of flash_next_gdn_chunk_slot over `count` consecutive rows of
/// one session, with this thread's S column held in registers across rows.
/// One CTA (128 threads, thread = value column) per value head. Key rows of
/// this call come from the input (the key group's first head writes the tail's
/// key rows while other CTAs still read); earlier rows come from the tail.
/// Same bytes as prep + scan over the same rows (kernel-bench split checks).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_chunk_rows(p: GdnChunkNext) { chunk_rows(p) }
/// Cohort verify form (VADE_GDN_COHORT_KERNELS): grid-Y = chat, whose GdnChunkNext is
/// entry y of the table at `table`. Each CTA runs flash_next_gdn_chunk_rows' code on its
/// chat's parameters, and no chat reads or writes another's state, tail or rows, so every
/// chat gets that kernel's words.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_chunk_rows_cohort(table: u64) {
    chunk_rows(core::ptr::read((table as *const GdnChunkNext).add(_block_idx_y() as usize)))
}
#[inline(always)]
unsafe fn chunk_rows(p: GdnChunkNext) {
    let sh = crate::ptx::shared::<SLOT_BYTES>();
    let tid = _thread_idx_x() as usize;
    let h = _block_idx_x() as usize;
    let (heads, keys) = (p.heads as usize, p.keys as usize);
    if keys == 0 || heads < keys || h >= heads { return; }
    let repeat = heads / keys;
    if repeat == 0 { return; }
    let kh = h / repeat;
    let h0 = kh * repeat;
    let hd = heads * D;
    let (base, count) = (p.base as usize, p.count as usize);
    let state = (*(p.rows as *const GdnDecodeRow)).state as *mut f32;
    let (tk_at, tw_at, tg_at) = tail(heads, keys);
    let (tk, tw, tg) = (state.add(tk_at + kh * C * D), state.add(tw_at + h * C * D), state.add(tg_at + h * C));
    let s = state.add(h * D * D);
    let (q, k, v) = (p.q as *const f32, p.k as *const f32, p.v as *const f32);
    let mut scol = [0.0f32; D];
    d128!(d => scol[d] = *s.add(d * D + tid));
    // Key rows of the open chunk, transposed into the kd region as [d][j]
    // (stride 65): the tail's rows now (whole-row loads), each new row's key
    // after its step; the chunk close rewrites the region, the next chunk
    // starts empty.
    let mut idx = tid;
    while idx < base % C * 32 {
        let (j, d4) = (idx / 32, idx % 32 * 4);
        let k4 = ldg4(tk.add(j * D + d4));
        for x in 0..4 { sts(sh + SLOT_KD + (((d4 + x) * (C + 1) + j) * 4) as u32, k4[x]); }
        idx += 128;
    }
    // Fused prepare (ga != 0): this CTA writes flash_gdn_conv_prepare_tokens_khs'
    // words for its own value head with prepare_values itself, one warp per row
    // (lane l holds features l + 32 f): the key group's q/k rows into this head's
    // slot (not h0's, so no two CTAs write one word), its v row and its beta/g.
    // The conv reads x rows x_start + r of the conv call (history before row 0).
    let qh = if p.ga != 0 { h } else { h0 };
    if p.ga != 0 {
        let (lane, channels) = (tid % 32, p.channels as usize);
        let conv = |channel: usize, token: usize| crate::flash_gdn::convolution(p.x as *const f32, p.w as *const u16,
            p.history as *const f32, channel, token, channels, 0);
        let mut r = tid / 32;
        while r < count {
            let token = p.x_start as usize + r;
            let load = |at: usize| [conv(at + lane, token), conv(at + lane + 32, token), conv(at + lane + 64, token), conv(at + lane + 96, token)];
            let (features, gates) = (r * hd, r * heads);
            crate::flash_gdn::prepare_values(load(kh * D), load((keys + kh) * D), load((2 * keys + h) * D),
                (p.ga as *const f32).add(gates), (p.gb as *const f32).add(gates), p.alog as *const u16, p.dt as *const u16,
                (p.q as *mut f32).add(features), (p.k as *mut f32).add(features), (p.v as *mut f32).add(features),
                (p.beta as *mut f32).add(gates), (p.g as *mut f32).add(gates), h, Some(h), lane, heads, 1, 0, false);
            r += 4;
        }
        sync();
    }
    for r in 0..count {
        let pos = base + r;
        let i = pos % C;
        let start = pos - i;
        let key = |j: usize, d: usize| -> f32 {
            if start + j >= base { *k.add((start + j - base) * hd + qh * D + d) } else { *tk.add(j * D + d) }
        };
        let at = r * hd;
        let beta = *(p.beta as *const f32).add(r * heads + h);
        let gamma = (if i > 0 { *tg.add(i - 1) } else { 0.0 }) + *(p.g as *const f32).add(r * heads + h);
        let e = exp(gamma);
        sts(sh + SLOT_Q + (tid * 4) as u32, *q.add(at + qh * D + tid));
        sts(sh + SLOT_K + (tid * 4) as u32, *k.add(at + qh * D + tid));
        sync();
        if tid < i {
            let (mut kk, mut qk) = (0.0f32, 0.0f32);
            for d in 0..D {
                let kj = lds(sh + SLOT_KD + ((d * (C + 1) + tid) * 4) as u32);
                kk = fma(lds(sh + SLOT_K + (d * 4) as u32), kj, kk);
                qk = fma(lds(sh + SLOT_Q + (d * 4) as u32), kj, qk);
            }
            let decay = exp(gamma - *tg.add(tid));
            sts(sh + SLOT_A + (tid * 4) as u32, (kk * decay) * beta);
            sts(sh + SLOT_P + (tid * 4) as u32, qk * decay);
        } else if tid == i {
            let mut qk = 0.0f32;
            for d in 0..D { qk = fma(lds(sh + SLOT_Q + (d * 4) as u32), lds(sh + SLOT_K + (d * 4) as u32), qk); }
            sts(sh + SLOT_P + (tid * 4) as u32, qk * exp(gamma - gamma));
        }
        sync();
        let (mut ks, mut qs) = (0.0f32, 0.0f32);
        d128!(d => {
            let sv = scol[d];
            ks = fma(lds(sh + SLOT_K + (d * 4) as u32), sv, ks);
            qs = fma(lds(sh + SLOT_Q + (d * 4) as u32), sv, qs);
        });
        let mut w = fma(-e, ks, *v.add(at + h * D + tid)) * beta;
        let mut o = 0.0f32;
        for j in 0..i {
            let t = *tw.add(j * D + tid);
            w = fma(-lds(sh + SLOT_A + (j * 4) as u32), t, w);
            o = fma(lds(sh + SLOT_P + (j * 4) as u32), t, o);
        }
        o = fma(lds(sh + SLOT_P + (i * 4) as u32), w, o);
        *(p.out as *mut f32).add(at + h * D + tid) = fma(e, qs, o);
        if i + 1 < C {
            *tw.add(i * D + tid) = w;
            if tid == 0 { *tg.add(i) = gamma; }
            if h == h0 { *tk.add(i * D + tid) = lds(sh + SLOT_K + (tid * 4) as u32); }
            sts(sh + SLOT_KD + ((tid * (C + 1) + i) * 4) as u32, lds(sh + SLOT_K + (tid * 4) as u32));
            sync();
            continue;
        }
        // Row 63 closes the chunk exactly as the slot kernel does (S cells in
        // global memory), then the column is reloaded for the next chunk.
        d128!(d => *s.add(d * D + tid) = scol[d]);
        if tid < C {
            let gj = if tid + 1 < C { *tg.add(tid) } else { gamma };
            sts(sh + SLOT_F + (tid * 4) as u32, exp(gamma - gj));
        }
        sync();
        let mut idx = tid;
        while idx < C * D {
            let (j, d) = (idx / D, idx % D);
            let kj = if j + 1 < C { key(j, d) } else { lds(sh + SLOT_K + (d * 4) as u32) };
            sts(sh + SLOT_KD + (idx * 4) as u32, kj * lds(sh + SLOT_F + (j * 4) as u32));
            idx += 128;
        }
        sync();
        let mut wv = [0.0f32; C];
        for j in 0..C - 1 { wv[j] = *tw.add(j * D + tid); }
        wv[C - 1] = w;
        let mut db = 0;
        while db < D {
            let mut old = [0.0f32; 8];
            for z in 0..8 { old[z] = *s.add((db + z) * D + tid); }
            let mut u = [0.0f32; 8];
            let kd_at = sh + SLOT_KD + (db * 4) as u32;
            close_cells!(kd_at, wv, u;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63);
            for z in 0..8 { *s.add((db + z) * D + tid) = fma(old[z], e, u[z]); }
            db += 8;
        }
        d128!(d => scol[d] = *s.add(d * D + tid));
        sync();
    }
    // Fused gate (gated != 0 with ga != 0): flash_gdn_gate's words (gate_plain itself)
    // on this head's out rows, which this CTA wrote before the loop's last barrier;
    // warp w gates rows w, w + 4.
    if p.gated != 0 && p.ga != 0 {
        let mut r = tid / 32;
        while r < count {
            let at = r * hd + h * D;
            crate::flash_gdn::gate_plain(crate::kernel_params::GdnGate { x: (p.out as *const f32).add(at), z: (p.z as *const f32).add(at),
                norm: p.norm as *const f32, output: (p.gated as *mut f32).add(at), heads: 1, tokens: 1, precision: 0, packed: 0,
                epsilon: p.epsilon, sessions: 0 }, tid % 32);
            r += 4;
        }
    }
}

/// Decode step in chunk form (switch on): one row per chat, the canonical row
/// function on the open chunk (row i = slot position % 64), the same FP32 ops as
/// the prompt kernels so any cut of a sequence into prompt calls and decode steps
/// gives the same bytes. One CTA (128 threads = value columns) per (value head,
/// chat). Rows below 63 join the tail; row 63 closes the chunk into S.
/// p: q/k/v/out [rows][heads][128] and beta/g [rows][heads] of this step (as the
/// main slot kernel); keys = key heads; rows/a/p/gamma/k0/count/base unused.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_chunk_slot(p: GdnChunkNext, step: crate::next_math::SlotStep) {
    let row = _block_idx_y() as usize;
    let Some(id) = step.selected(row) else {
        // Fused output gate: flash_next_gdn_gate_slot writes zeros for the rows a step skips.
        let h = _block_idx_x() as usize;
        if p.gated != 0 && h < p.heads as usize {
            *(p.gated as *mut f32).add(row * p.heads as usize * D + h * D + _thread_idx_x() as usize) = 0.0;
        }
        return;
    };
    let slot = &*(step.table as *const crate::next_math::StateSlot).add(id);
    let sh = crate::ptx::shared::<SLOT_BYTES>();
    let tid = _thread_idx_x() as usize;
    let h = _block_idx_x() as usize;
    let (heads, keys) = (p.heads as usize, p.keys as usize);
    if keys == 0 || heads < keys || h >= heads { return; }
    let repeat = heads / keys;
    if repeat == 0 { return; }
    let kh = h / repeat;
    let h0 = kh * repeat;
    let hd = heads * D;
    let i = slot.position as usize % C;
    let Some(layer) = slot.layers.get(step.layer as usize) else { return; };
    let state = layer[0] as *mut f32;
    let (tk_at, tw_at, tg_at) = tail(heads, keys);
    let (tk, tw, tg) = (state.add(tk_at + kh * C * D), state.add(tw_at + h * C * D), state.add(tg_at + h * C));
    let s = state.add(h * D * D);
    // The head's whole S (64 KB) requested into L2 at once; the piece loads
    // below then hit L2 while the kk/qk phase runs.
    for u in 0..4 { prefetch_l2(s.add((tid + u * 128) * 32)); }
    let at = row * hd;
    // Fused output gate (p.gated != 0): warp 0 takes flash_next_gdn_gate_slot's
    // inputs for this head's row up front (lane l: features l + 32 k).
    let gated = p.gated as *mut f32;
    let (mut zg, mut wg) = ([0.0f32; 4], [0.0f32; 4]);
    if !gated.is_null() && tid < 32 {
        each4!(K => {
            zg[K] = *(p.z as *const f32).add(at + h * D + tid + 32 * K);
            wg[K] = *(p.norm as *const f32).add(tid + 32 * K);
        });
    }
    // The tail's key rows, transposed into the kd region as [d][j] (stride 65)
    // with whole-row loads (32 threads per 512-B row). State only: read before the
    // wait for prepare_hist_slot (a no-op for a plain launch).
    let mut idx = tid;
    while idx < i * 32 {
        let (j, d4) = (idx / 32, idx % 32 * 4);
        let k4 = ldg4(tk.add(j * D + d4));
        for x in 0..4 { sts(sh + SLOT_KD + (((d4 + x) * (C + 1) + j) * 4) as u32, k4[x]); }
        idx += 128;
    }
    let gamma0 = if i > 0 { *tg.add(i - 1) } else { 0.0 };
    asm!("griddepcontrol.wait;", options(nostack));
    let beta = *(p.beta as *const f32).add(row * heads + h);
    let gamma = gamma0 + *(p.g as *const f32).add(row * heads + h);
    let e = exp(gamma);
    sts(sh + SLOT_Q + (tid * 4) as u32, *(p.q as *const f32).add(at + h0 * D + tid));
    sts(sh + SLOT_K + (tid * 4) as u32, *(p.k as *const f32).add(at + h0 * D + tid));
    sync();
    // kk_ij, qk_ij against the tail's key rows (one thread per j, d ascending).
    if tid < i {
        let (mut kk, mut qk) = (0.0f32, 0.0f32);
        for d in 0..D {
            let kj = lds(sh + SLOT_KD + ((d * (C + 1) + tid) * 4) as u32);
            kk = fma(lds(sh + SLOT_K + (d * 4) as u32), kj, kk);
            qk = fma(lds(sh + SLOT_Q + (d * 4) as u32), kj, qk);
        }
        let decay = exp(gamma - *tg.add(tid));
        sts(sh + SLOT_A + (tid * 4) as u32, (kk * decay) * beta);
        sts(sh + SLOT_P + (tid * 4) as u32, qk * decay);
    } else if tid == i {
        let mut qk = 0.0f32;
        for d in 0..D { qk = fma(lds(sh + SLOT_Q + (d * 4) as u32), lds(sh + SLOT_K + (d * 4) as u32), qk); }
        sts(sh + SLOT_P + (tid * 4) as u32, qk * exp(gamma - gamma));
    }
    sync();
    // S streams through the kd region (free until the chunk close) in four
    // pieces of 32 rows, double-buffered: 8 vector loads in flight per thread.
    let (mut ks, mut qs) = (0.0f32, 0.0f32);
    let mut next = [[0.0f32; 4]; 8];
    for u in 0..8 {
        let idx = tid + u * 128;
        next[u] = ldg4(s.add((idx / 32) * D + idx % 32 * 4));
    }
    for b in 0..4 {
        let half = sh + SLOT_KD + (b % 2) as u32 * (32 * D * 4) as u32;
        for u in 0..8 {
            let idx = tid + u * 128;
            sts4(half + (((idx / 32) * D + idx % 32 * 4) * 4) as u32, next[u]);
        }
        sync();
        if b < 3 {
            for u in 0..8 {
                let idx = tid + u * 128;
                next[u] = ldg4(s.add(((b + 1) * 32 + idx / 32) * D + idx % 32 * 4));
            }
        }
        for d in 0..32 {
            let sv = lds(half + ((d * D + tid) * 4) as u32);
            ks = fma(lds(sh + SLOT_K + ((b * 32 + d) * 4) as u32), sv, ks);
            qs = fma(lds(sh + SLOT_Q + ((b * 32 + d) * 4) as u32), sv, qs);
        }
    }
    let mut w = fma(-e, ks, *(p.v as *const f32).add(at + h * D + tid)) * beta;
    let mut o = 0.0f32;
    for j in 0..i {
        let t = *tw.add(j * D + tid);
        w = fma(-lds(sh + SLOT_A + (j * 4) as u32), t, w);
        o = fma(lds(sh + SLOT_P + (j * 4) as u32), t, o);
    }
    o = fma(lds(sh + SLOT_P + (i * 4) as u32), w, o);
    let out = fma(e, qs, o);
    *(p.out as *mut f32).add(at + h * D + tid) = out;
    if !gated.is_null() {
        // flash_next_gdn_gate_slot (gate_plain) on this head's row, the same words:
        // the head's 128 outputs through the q region (free after the KS/QS pass),
        // then warp 0 with lane l on features l + 32 k, the same squares, warp
        // sum, scale, weight and sigmoid.
        sync();
        sts(sh + SLOT_Q + (tid * 4) as u32, out);
        sync();
        if tid < 32 {
            let mut values = [0.0f32; 4];
            each4!(K => { values[K] = lds(sh + SLOT_Q + ((tid + 32 * K) * 4) as u32); });
            let mut squares = 0.0;
            each4!(K => { squares += values[K] * values[K]; });
            let scale = 1.0 / sqrt(sum(squares) / 128.0 + p.epsilon);
            each4!(K => {
                let value = values[K] * scale * wg[K];
                let value = value * sigmoid(zg[K]) * 1.0;
                *gated.add(at + h * D + tid + 32 * K) = value;
            });
        }
    }
    if i + 1 < C {
        *tw.add(i * D + tid) = w;
        if tid == 0 { *tg.add(i) = gamma; }
        if h == h0 { *tk.add(i * D + tid) = lds(sh + SLOT_K + (tid * 4) as u32); }
        return;
    }
    // Row 63: S' = fma(S, e_63, sum_j (k_j[d] * f_j) v'_j), j ascending.
    if tid < C {
        let gj = if tid + 1 < C { *tg.add(tid) } else { gamma };
        sts(sh + SLOT_F + (tid * 4) as u32, exp(gamma - gj));
    }
    sync();
    let mut idx = tid;
    while idx < C * D {
        let (j, d) = (idx / D, idx % D);
        let kj = if j + 1 < C { *tk.add(j * D + d) } else { lds(sh + SLOT_K + (d * 4) as u32) };
        sts(sh + SLOT_KD + (idx * 4) as u32, kj * lds(sh + SLOT_F + (j * 4) as u32));
        idx += 128;
    }
    sync();
    // This thread's column of v' sits in registers (literal indices below);
    // eight cells advance together, each with its own chain over j ascending.
    let mut wv = [0.0f32; C];
    for j in 0..C - 1 { wv[j] = *tw.add(j * D + tid); }
    wv[C - 1] = w;
    let mut db = 0;
    while db < D {
        let mut old = [0.0f32; 8];
        for z in 0..8 { old[z] = *s.add((db + z) * D + tid); }
        let mut u = [0.0f32; 8];
        let kd_at = sh + SLOT_KD + (db * 4) as u32;
        close_cells!(kd_at, wv, u;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63);
        for z in 0..8 { *s.add((db + z) * D + tid) = fma(old[z], e, u[z]); }
        db += 8;
    }
}
