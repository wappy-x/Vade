//! FP4-COHORT: routed-expert NVFP4 products on GB10's native block-scaled tensor-core instruction, for prompt groups.
//! Serving: `fp4_cohort` (first matrices, one chain) and `fp4_cohort_t64` (second matrices, quarter chains), launched
//! by `backend/tc_family.rs` from the second kernel image. Bench only: `fp4_cohort_t8`, `fp4_cohort_cc`, `fp4_peak`.
//!
//! Contract (b), per output word (part, row, token): acc = +0.0; for each 64-column step in ascending K,
//! acc = mma.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X(weight codes, activation codes, weight and activation
//! UE4M3 block scales, acc); word = bf16(fl(acc * fl(global * activation))). The instruction equals an integer model
//! on every word (lane multi-row probe: exact 16-column block sums, exact scale products, terms aligned on the largest
//! exponent and cut at 35 fraction bits, cut toward zero to FP32); the bench's CPU twin applies that model in this
//! order. A word does not depend on its tile, on the cohort size or on the token's place in the tile, so the decode
//! form is this kernel at one token.
//!
//! Weights, `layout` 1 = column tiles (128 rows a tile; column j of tile t is 64 bytes at (t*cols+j)*64, rows 2i and
//! 2i+1 in the low and high nibble of byte i; scale block b of tile t is 128 bytes at (t*cols/16+b)*128, one a row),
//! `layout` 0 = row-major (rows x cols/2 bytes, low nibble = even column; scales rows x cols/16). Activations:
//! token-major E2M1 codes (cols/2 bytes a token, low nibble = even column) and UE4M3 scale codes (cols/16 bytes a
//! token). Output: the part's `[row][tokens]` FP32 words (BF16 values), like the NV row kernels.
//!
//! Launch: entry `fp4_cohort`, grid (ceil(max_tokens/128), rows/128, count), 256 threads, no dynamic shared memory.
//! A block owns 128 weight rows (one column tile) and 128 tokens of one part. Eight warps: warp w takes rows
//! 64*(w%2)..+63 (four m16 tiles) and tokens 32*(w/2)..+31 (four n8 tiles). K passes through shared memory in 64-column
//! steps, two stages; column tiles are transposed to row-major K in shared memory (8 x 8 nibble transposes).
//! `chains` 4 (second matrices): the quarter-chain word of `tc_contract`, in the 64-token and 8-token forms. The
//! 128-token form takes one chain only: a second accumulator set there costs 255 registers and 1,460 spill bytes
//! (ptxas, a lane box, 2026-11-11).
//! `skip` 1 with one live token in the block: weight columns whose activation code is +-0 are not read (their shared
//! bytes are set to 0): a zero activation makes every product of that column zero whatever the weight code.
use crate::canonical::Acc;

pub use crate::tc_abi::{NvPart as Fp4CohortPart, NvRows as Fp4Cohort};

const THREADS: usize = 256;
const BR: usize = 128; // weight rows a block
const BT: usize = 128; // tokens a block
// Shared layout, in 32-bit words. AR: raw weight step (column tiles: column c at c*16 + (c/8)*4, so the eight columns a
// transposing warp reads at once fall in distinct banks; row-major: row r at r*8). AS: raw scale step (column tiles:
// block b at b*32, one byte a row; row-major: unused). B: activation codes, token n at n*12 (48 bytes: the eight fragment
// rows of a load hit distinct banks). BS: activation scale words, token n at n. AT: weight codes as row-major K, row r at
// r*12 + (r/8)*8 (the pad spreads the transpose stores). ATS: weight scale words, row r at r. SK: the one live token's
// codes for zero-skip.
const AR: usize = 1056;
const AS: usize = 128;
const B: usize = BT * 12;
const BS: usize = BT;
const STAGE: usize = AR + AS + B + BS;
const AT: usize = BR * 12 + BR / 8 * 8;
const ATS: usize = BR;
const SK: usize = 1024; // up to 8,192 columns
const WORDS: usize = 2 * STAGE + AT + ATS + SK;
const BYTES: usize = WORDS * 4;
const _: () = assert!(BYTES <= 48 * 1024);

// The helpers below are macros, not functions: they expand inside the one unsafe body of the kernel entry that uses
// them, under that entry's safety contract (shared addresses inside the static array; register operands only).
macro_rules! ld {($p:expr)=>{crate::ptx::load($p).to_bits()};}
macro_rules! st {($p:expr,$v:expr)=>{crate::ptx::store($p, f32::from_bits($v))};}
macro_rules! zero16 {($p:expr)=>{core::arch::asm!("st.shared.v4.b32 [{p}], {{{z}, {z}, {z}, {z}}};", p = in(reg32) $p, z = in(reg32) 0u32)};}
macro_rules! add_rn {($a:expr,$b:expr)=>{{
    let o: f32;
    core::arch::asm!("add.rn.f32 {o}, {a}, {b};", o = out(reg32) o, a = in(reg32) $a, b = in(reg32) $b);
    o
}};}
/// fl(x + y) word by word: the quarter join of four accumulator words.
macro_rules! add_acc {($x:expr,$y:expr)=>{{
    let (x, y): (Acc, Acc) = ($x, $y);
    Acc { a: add_rn!(x.a, y.a), b: add_rn!(x.b, y.b), c: add_rn!(x.c, y.c), d: add_rn!(x.d, y.d) }
}};}
macro_rules! mma {($a:expr,$b:expr,$sa:expr,$sb:expr,$c:expr)=>{{
    let (a, b, sa, sb, c): ([u32; 4], [u32; 2], u32, u32, Acc) = ($a, $b, $sa, $sb, $c);
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    core::arch::asm!(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \
         {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, \
         {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,0}};",
        d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
        a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
        b0 = in(reg32) b[0], b1 = in(reg32) b[1],
        c0 = in(reg32) c.a, c1 = in(reg32) c.b, c2 = in(reg32) c.c, c3 = in(reg32) c.d,
        sa = in(reg32) sa, sb = in(reg32) sb,
    );
    Acc { a: d0, b: d1, c: d2, d: d3 }
}};}
// Straight-line copies, so every array index below is a constant and the arrays stay in registers.
macro_rules! rep4 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}{const $v:usize=2;$body}{const $v:usize=3;$body}}};}
macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const $v:usize=4;$body}{const $v:usize=5;$body}{const $v:usize=6;$body}{const $v:usize=7;$body}}};}
#[inline(always)]
fn swap(a: &mut u32, b: &mut u32, shift: u32, mask: u32) {
    let t = ((*a >> shift) ^ *b) & mask;
    *b ^= t;
    *a ^= t << shift;
}

/// The 128-token form: prompt groups of FIRST matrices on the serving path (`Backend::nv_rows_tc`), and the bench.
/// SAFETY: the host checks before the launch: rows a multiple of 128; cols a multiple of 64 and at most 8,192; every
/// part's weight and scale span for the layout, 16-byte aligned; the packed activation codes and scale codes of every
/// token row of every part; each part's output span of rows x tokens words; `chains` 1. Every owner lives through
/// stream completion. Inputs are read only. Each output word has exactly one writing thread (the thread whose
/// accumulator fragment holds that row and token) and no reader; only words of tokens below the part's token count
/// are stored, and every such word is stored, so the output may hold anything before the launch. Shared reads and
/// writes stay inside this entry's static array; a token slot past the last live token holds zeros. All 256 threads
/// reach every barrier: loop bounds depend only on the block's part. Any other launch traps.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_cohort(p: Fp4Cohort) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x};
    if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
        || p.rows % BR as u32 != 0 || p.layout > 1 || p.chains != 1 {
        core::arch::asm!("trap;");
    }
    let part = _block_idx_z() as usize;
    if part >= p.count as usize { return; }
    let e = *(p.table as *const Fp4CohortPart).add(part);
    let (rows, cols, m) = (p.rows as usize, p.cols as usize, e.tokens as usize);
    let (token_tile, row_tile) = (_block_idx_x() as usize, _block_idx_y() as usize);
    if row_tile * BR >= rows || token_tile * BT >= m { return; }
    let live = (m - token_tile * BT).min(BT);
    let tid = _thread_idx_x() as usize;
    let smem = crate::ptx::shared::<BYTES>();
    let word = |w: usize| smem + (w * 4) as u32;
    let at = 2 * STAGE;
    let ats = at + AT;
    let sk = ats + ATS;
    let steps = cols / 64;
    let tiled = p.layout == 1;
    let skip = p.skip == 1 && live == 1;
    let codes = (p.codes as *const u8).add((e.first as usize + token_tile * BT) * (cols / 2));
    let ascales = (p.scales as *const u8).add((e.first as usize + token_tile * BT) * (cols / 16));
    if skip {
        // The one live token's codes, cols/2 bytes as cols/8 words.
        let mut i = tid;
        while i < cols / 8 { st!(word(sk + i), *(codes as *const u32).add(i)); i += THREADS; }
        crate::ptx::sync();
    }
    let weights = e.weight as *const u8;
    let wscales = e.scales as *const u8;
    let issue = |c: usize| {
        let s = (c & 1) * STAGE;
        // Weights: 4 KB, 256 copies of 16 bytes.
        if tiled {
            let (col, q) = (tid / 4, tid % 4);
            let column = c * 64 + col;
            let dst = word(s + col * 16 + (col / 8) * 4 + q * 4);
            let zero = skip && (ld!(word(sk + column / 8)) >> (4 * (column % 8)) & 7) == 0;
            if zero { zero16!(dst); } else {
                crate::ptx::copy16(dst, weights.add(((row_tile * cols + column) * 64) + q * 16).cast());
            }
            // Scales: four 128-byte blocks, 32 copies.
            if tid < 32 {
                crate::ptx::copy16(word(s + AR + tid * 4), wscales.add((row_tile * (cols / 16) + c * 4) * 128 + tid * 16).cast());
            }
        } else {
            let (row, q) = (tid / 2, tid % 2);
            let r = row_tile * BR + row;
            crate::ptx::copy16(word(s + row * 8 + q * 4), weights.add(r * (cols / 2) + c * 32 + q * 16).cast());
        }
        // Activations: 32 bytes a token as two 16-byte copies, and one scale word a token.
        let mut i = tid;
        while i < live * 2 {
            let (n, q) = (i / 2, i % 2);
            crate::ptx::copy16(word(s + AR + AS + n * 12 + q * 4), codes.add(n * (cols / 2) + c * 32 + q * 16).cast());
            i += THREADS;
        }
        if tid < live {
            crate::ptx::copy4(word(s + AR + AS + B + tid), ascales.add(tid * (cols / 16) + c * 4).cast());
        }
    };
    let (lane, warp) = (tid % 32, tid / 32);
    let (g, t) = (lane / 4, lane % 4);
    let (wr, wc) = (warp % 2, warp / 2);
    let mut acc = [[crate::canonical::ZERO; 4]; 4];
    let output = e.output as *mut f32;
    // A token slot past the last live token is never copied. Its codes and its scale word are set to zero in both
    // stages once, before the first step, so every operand of the instruction is defined. (The first read of a stage
    // follows a barrier.) Its words are dropped, never stored.
    {
        let mut i = live * 2 + tid;
        while i < BT * 2 {
            let at = AR + AS + (i / 2) * 12 + (i % 2) * 4;
            zero16!(word(at)); zero16!(word(STAGE + at));
            i += THREADS;
        }
        if tid >= live && tid < BT { st!(word(AR + AS + B + tid), 0u32); st!(word(STAGE + AR + AS + B + tid), 0u32); }
    }
    issue(0);
    crate::ptx::commit();
    let mut c = 0;
    while c < steps {
        if c + 1 < steps { issue(c + 1); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
        crate::ptx::sync();
        let s = (c & 1) * STAGE;
        // Weight codes to row-major K (AT) and scale words (ATS).
        if tiled {
            if tid < 128 {
                let (cb, rb) = (tid % 8, tid / 8);
                let mut x = [0u32; 8];
                rep8!(J, { let col = 8 * cb + J; x[J] = ld!(word(s + col * 16 + (col / 8) * 4 + rb)); });
                // x[j] nibble i = (row 8rb+i, column 8cb+j); after the transpose x[i] nibble j holds it.
                let [mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7] = x;
                swap(&mut x0, &mut x4, 16, 0x0000_ffff); swap(&mut x1, &mut x5, 16, 0x0000_ffff);
                swap(&mut x2, &mut x6, 16, 0x0000_ffff); swap(&mut x3, &mut x7, 16, 0x0000_ffff);
                swap(&mut x0, &mut x2, 8, 0x00ff_00ff); swap(&mut x1, &mut x3, 8, 0x00ff_00ff);
                swap(&mut x4, &mut x6, 8, 0x00ff_00ff); swap(&mut x5, &mut x7, 8, 0x00ff_00ff);
                swap(&mut x0, &mut x1, 4, 0x0f0f_0f0f); swap(&mut x2, &mut x3, 4, 0x0f0f_0f0f);
                swap(&mut x4, &mut x5, 4, 0x0f0f_0f0f); swap(&mut x6, &mut x7, 4, 0x0f0f_0f0f);
                let y = [x0, x1, x2, x3, x4, x5, x6, x7];
                rep8!(I, { let r = 8 * rb + I; st!(word(at + r * 12 + (r / 8) * 8 + cb), y[I]); });
            } else {
                let r = tid - 128;
                let b = |q: usize| crate::ptx::load_u8(word(s + AR) + (q * 128 + r) as u32);
                st!(word(ats + r), b(0) | b(1) << 8 | b(2) << 16 | b(3) << 24);
            }
        } else {
            let (row, q) = (tid / 2, tid % 2);
            let v = crate::ptx::load4(word(s + row * 8 + q * 4));
            let base = at + row * 12 + (row / 8) * 8 + q * 4;
            rep4!(K, { st!(word(base + K), v[K]); });
            if tid < BR {
                let r = row_tile * BR + tid;
                st!(word(ats + tid), *(wscales.add(r * (cols / 16) + c * 4) as *const u32));
            }
        }
        crate::ptx::sync();
        // Products: warp (wr, wc) = rows 64wr.. (four m16 tiles) x tokens 32wc.. (four n8 tiles).
        // Token fragments once a step (12 registers), then each row tile's fragment and its four products.
        let mut b = [[0u32; 3]; 4];
        rep4!(NJ, {
            let n0 = 32 * wc + 8 * NJ;
            if n0 < live {
                let tok = s + AR + AS + (n0 + g) * 12;
                b[NJ] = [ld!(word(tok + t)), ld!(word(tok + t + 4)), ld!(word(s + AR + AS + B + n0 + g))];
            }
        });
        rep4!(MI, {
            let r0 = 64 * wr + 16 * MI;
            let row = |r: usize, k: usize| ld!(word(at + r * 12 + (r / 8) * 8 + k));
            let a = [row(r0 + g, t), row(r0 + g + 8, t), row(r0 + g, t + 4), row(r0 + g + 8, t + 4)];
            let sa = ld!(word(ats + r0 + g + (t % 2) * 8));
            rep4!(NJ, {
                if 32 * wc + 8 * NJ < live { acc[MI][NJ] = mma!(a, [b[NJ][0], b[NJ][1]], sa, b[NJ][2], acc[MI][NJ]); }
            });
        });
        crate::ptx::sync();
        c += 1;
    }
    // word = bf16(fl(acc * fl(global * activation))).
    let scale: f32;
    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e.global, b = in(reg32) e.activation);
    rep4!(MI, {
        rep4!(NJ, {
            let r = row_tile * BR + 64 * wr + 16 * MI + g;
            let n = token_tile * BT + 32 * wc + 8 * NJ + 2 * t;
            let v = acc[MI][NJ];
            let put = |row: usize, token: usize, x: f32| {
                if token < m && p.raw == 1 { *output.add(row * m + token) = x; } else if token < m {
                    let y: f32;
                    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                    *output.add(row * m + token) = crate::expert_ops::round(y);
                }
            };
            put(r, n, v.a);
            put(r, n + 1, v.b);
            put(r + 8, n, v.c);
            put(r + 8, n + 1, v.d);
        });
    });
}

// Two more forms of the same contract (the same chain a word, so the same words), for the speed questions: a 64-token
// block with half the accumulators a warp and three stages, and an 8-token block for the decode step with four stages
// in flight. `$bt` tokens a block, `$wr` warps along the 128 rows (8 / $wr along the tokens), `$st` stages.
macro_rules! cohort_entry {
    ($name:ident, $bt:expr, $wr:expr, $st:expr) => {
        /// SAFETY: as `fp4_cohort`, for this block shape: grid (ceil(max_tokens/BT), rows/128, count), 256 threads;
        /// `chains` 1, or 4 with cols a multiple of 256. With `chains` 4 a thread stores fl(a0 + a1) of its OWN words in
        /// the part's output after the second quarter and loads its own words back at the join after the fourth. No
        /// other thread reads or writes those words (one writer a word), no barrier is needed for a thread's own store
        /// and load, and the output span must be exclusive to this launch until stream completion: nothing else may
        /// read it before, because it holds partial sums until the last step.
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(p: Fp4Cohort) {
            use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x};
            const BT: usize = $bt;
            const WR: usize = $wr;
            const WC: usize = 8 / WR;
            const MIC: usize = 8 / WR; // m16 tiles a warp
            const NJC: usize = BT / (8 * WC); // n8 tiles a warp
            const S: usize = $st;
            const WAITN: usize = S - 2;
            const BW: usize = BT * 12;
            const STAGE: usize = AR + AS + BW + BT;
            const AT0: usize = S * STAGE;
            const WORDS: usize = AT0 + AT + ATS + SK;
            const BYTES: usize = WORDS * 4;
            const _: () = assert!(BYTES <= 48 * 1024 && MIC <= 4 && NJC <= 4 && NJC >= 1 && S >= 2);
            if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
                || p.rows % BR as u32 != 0 || p.layout > 1 || !(p.chains == 1 || p.chains == 4 && p.cols % 256 == 0) {
                core::arch::asm!("trap;");
            }
            let part = _block_idx_z() as usize;
            if part >= p.count as usize { return; }
            let e = *(p.table as *const Fp4CohortPart).add(part);
            let (rows, cols, m) = (p.rows as usize, p.cols as usize, e.tokens as usize);
            let (token_tile, row_tile) = (_block_idx_x() as usize, _block_idx_y() as usize);
            if row_tile * BR >= rows || token_tile * BT >= m { return; }
            let live = (m - token_tile * BT).min(BT);
            let tid = _thread_idx_x() as usize;
            let smem = crate::ptx::shared::<BYTES>();
            let word = |w: usize| smem + (w * 4) as u32;
            let (at, ats, sk) = (AT0, AT0 + AT, AT0 + AT + ATS);
            let steps = cols / 64;
            let tiled = p.layout == 1;
            let skip = p.skip == 1 && live == 1;
            let codes = (p.codes as *const u8).add((e.first as usize + token_tile * BT) * (cols / 2));
            let ascales = (p.scales as *const u8).add((e.first as usize + token_tile * BT) * (cols / 16));
            if skip {
                let mut i = tid;
                while i < cols / 8 { st!(word(sk + i), *(codes as *const u32).add(i)); i += THREADS; }
                crate::ptx::sync();
            }
            let weights = e.weight as *const u8;
            let wscales = e.scales as *const u8;
            let issue = |c: usize| {
                let s = (c % S) * STAGE;
                if tiled {
                    let (col, q) = (tid / 4, tid % 4);
                    let column = c * 64 + col;
                    let dst = word(s + col * 16 + (col / 8) * 4 + q * 4);
                    let zero = skip && (ld!(word(sk + column / 8)) >> (4 * (column % 8)) & 7) == 0;
                    if zero { zero16!(dst); } else {
                        crate::ptx::copy16(dst, weights.add(((row_tile * cols + column) * 64) + q * 16).cast());
                    }
                    if tid < 32 {
                        crate::ptx::copy16(word(s + AR + tid * 4), wscales.add((row_tile * (cols / 16) + c * 4) * 128 + tid * 16).cast());
                    }
                } else {
                    let (row, q) = (tid / 2, tid % 2);
                    let r = row_tile * BR + row;
                    crate::ptx::copy16(word(s + row * 8 + q * 4), weights.add(r * (cols / 2) + c * 32 + q * 16).cast());
                }
                let mut i = tid;
                while i < live * 2 {
                    let (n, q) = (i / 2, i % 2);
                    crate::ptx::copy16(word(s + AR + AS + n * 12 + q * 4), codes.add(n * (cols / 2) + c * 32 + q * 16).cast());
                    i += THREADS;
                }
                if tid < live {
                    crate::ptx::copy4(word(s + AR + AS + BW + tid), ascales.add(tid * (cols / 16) + c * 4).cast());
                }
            };
            let (lane, warp) = (tid % 32, tid / 32);
            let (g, t) = (lane / 4, lane % 4);
            let (wr, wc) = (warp % WR, warp / WR);
            let mut acc = [[crate::canonical::ZERO; 4]; 4];
            // Quarter chains (second matrices): the finished chain of quarter 0 or 2 waits here.
            let mut hold = [[crate::canonical::ZERO; 4]; 4];
            let quarters = p.chains == 4;
            let per_chain = steps / p.chains as usize;
            let output = e.output as *mut f32;
            // A token slot past the last live token is never copied: zero codes and a zero scale word in every stage,
            // once, before the first step (the first read of a stage follows a barrier). Its words are never stored.
            {
                let mut i = live * 2 + tid;
                while i < BT * 2 {
                    let at = AR + AS + (i / 2) * 12 + (i % 2) * 4;
                    let mut stage = 0;
                    while stage < S { zero16!(word(stage * STAGE + at)); stage += 1; }
                    i += THREADS;
                }
                if tid >= live && tid < BT {
                    let mut stage = 0;
                    while stage < S { st!(word(stage * STAGE + AR + AS + BW + tid), 0u32); stage += 1; }
                }
            }
            let mut k = 0;
            while k < S - 1 { if k < steps { issue(k); } crate::ptx::commit(); k += 1; }
            let mut c = 0;
            while c < steps {
                core::arch::asm!("cp.async.wait_group {n};", n = const WAITN);
                crate::ptx::sync();
                let s = (c % S) * STAGE;
                if tiled {
                    if tid < 128 {
                        let (cb, rb) = (tid % 8, tid / 8);
                        let mut x = [0u32; 8];
                        rep8!(J, { let col = 8 * cb + J; x[J] = ld!(word(s + col * 16 + (col / 8) * 4 + rb)); });
                        let [mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7] = x;
                        swap(&mut x0, &mut x4, 16, 0x0000_ffff); swap(&mut x1, &mut x5, 16, 0x0000_ffff);
                        swap(&mut x2, &mut x6, 16, 0x0000_ffff); swap(&mut x3, &mut x7, 16, 0x0000_ffff);
                        swap(&mut x0, &mut x2, 8, 0x00ff_00ff); swap(&mut x1, &mut x3, 8, 0x00ff_00ff);
                        swap(&mut x4, &mut x6, 8, 0x00ff_00ff); swap(&mut x5, &mut x7, 8, 0x00ff_00ff);
                        swap(&mut x0, &mut x1, 4, 0x0f0f_0f0f); swap(&mut x2, &mut x3, 4, 0x0f0f_0f0f);
                        swap(&mut x4, &mut x5, 4, 0x0f0f_0f0f); swap(&mut x6, &mut x7, 4, 0x0f0f_0f0f);
                        let y = [x0, x1, x2, x3, x4, x5, x6, x7];
                        rep8!(I, { let r = 8 * rb + I; st!(word(at + r * 12 + (r / 8) * 8 + cb), y[I]); });
                    } else {
                        let r = tid - 128;
                        let b = |q: usize| crate::ptx::load_u8(word(s + AR) + (q * 128 + r) as u32);
                        st!(word(ats + r), b(0) | b(1) << 8 | b(2) << 16 | b(3) << 24);
                    }
                } else {
                    let (row, q) = (tid / 2, tid % 2);
                    let v = crate::ptx::load4(word(s + row * 8 + q * 4));
                    let base = at + row * 12 + (row / 8) * 8 + q * 4;
                    rep4!(K, { st!(word(base + K), v[K]); });
                    if tid < BR {
                        let r = row_tile * BR + tid;
                        st!(word(ats + tid), *(wscales.add(r * (cols / 16) + c * 4) as *const u32));
                    }
                }
                crate::ptx::sync();
                let mut b = [[0u32; 3]; 4];
                rep4!(NJ, {
                    let n0 = (BT / WC) * wc + 8 * NJ;
                    if NJ < NJC && n0 < live {
                        let tok = s + AR + AS + (n0 + g) * 12;
                        b[NJ] = [ld!(word(tok + t)), ld!(word(tok + t + 4)), ld!(word(s + AR + AS + BW + n0 + g))];
                    }
                });
                rep4!(MI, {
                    if MI < MIC {
                        let r0 = (128 / WR) * wr + 16 * MI;
                        let row = |r: usize, k: usize| ld!(word(at + r * 12 + (r / 8) * 8 + k));
                        let a = [row(r0 + g, t), row(r0 + g + 8, t), row(r0 + g, t + 4), row(r0 + g + 8, t + 4)];
                        let sa = ld!(word(ats + r0 + g + (t % 2) * 8));
                        rep4!(NJ, {
                            if NJ < NJC && (BT / WC) * wc + 8 * NJ < live { acc[MI][NJ] = mma!(a, [b[NJ][0], b[NJ][1]], sa, b[NJ][2], acc[MI][NJ]); }
                        });
                    }
                });
                crate::ptx::sync();
                if c + S - 1 < steps { issue(c + S - 1); }
                crate::ptx::commit();
                c += 1;
                // A chain ends every `per_chain` steps and the next starts from +0.0. Quarters 0 and 2 wait in `hold`.
                // After quarter 1 the pair fl(a0 + a1) is parked in the thread's own output words (private to this
                // launch, written again by the epilogue). After quarter 3: acc = fl(fl(a0 + a1) + fl(a2 + a3)).
                // Only register work and the thread's own words: no barrier.
                if quarters && c % per_chain == 0 {
                    let quarter = c / per_chain - 1;
                    rep4!(MI, { rep4!(NJ, { if MI < MIC && NJ < NJC {
                        if quarter == 0 || quarter == 2 {
                            hold[MI][NJ] = acc[MI][NJ];
                            acc[MI][NJ] = crate::canonical::ZERO;
                        } else {
                            let pair = add_acc!(hold[MI][NJ], acc[MI][NJ]);
                            let r = row_tile * BR + (128 / WR) * wr + 16 * MI + g;
                            let n = token_tile * BT + (BT / WC) * wc + 8 * NJ + 2 * t;
                            let at = |row: usize, token: usize| output.add(row * m + token);
                            if quarter == 1 {
                                if n < m { *at(r, n) = pair.a; *at(r + 8, n) = pair.c; }
                                if n + 1 < m { *at(r, n + 1) = pair.b; *at(r + 8, n + 1) = pair.d; }
                                acc[MI][NJ] = crate::canonical::ZERO;
                            } else {
                                let mut first = crate::canonical::ZERO;
                                if n < m { first.a = *at(r, n); first.c = *at(r + 8, n); }
                                if n + 1 < m { first.b = *at(r, n + 1); first.d = *at(r + 8, n + 1); }
                                acc[MI][NJ] = add_acc!(first, pair);
                            }
                        }
                    } }); });
                }
            }
            let scale: f32;
            core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e.global, b = in(reg32) e.activation);
            rep4!(MI, {
                rep4!(NJ, {
                    if MI < MIC && NJ < NJC {
                        let r = row_tile * BR + (128 / WR) * wr + 16 * MI + g;
                        let n = token_tile * BT + (BT / WC) * wc + 8 * NJ + 2 * t;
                        let v = acc[MI][NJ];
                        let put = |row: usize, token: usize, x: f32| {
                            if token < m && p.raw == 1 { *output.add(row * m + token) = x; } else if token < m {
                                let y: f32;
                                core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                                *output.add(row * m + token) = crate::expert_ops::round(y);
                            }
                        };
                        put(r, n, v.a);
                        put(r, n + 1, v.b);
                        put(r + 8, n, v.c);
                        put(r + 8, n + 1, v.d);
                    }
                });
            });
        }
    };
}
cohort_entry!(fp4_cohort_t64, 64, 4, 3);
cohort_entry!(fp4_cohort_t8, 8, 8, 4);

// CUDA-core form of the same contract for a few tokens a part (the decode step). A word's value is the instruction's
// integer model: exact 16-column block sums (integers, so any order), exact UE4M3 scale products, then per 64-column
// step in ascending K the aligned, cut sum of the four block terms and the old accumulator. Only that per-step chain is
// ordered; it runs once a row. Same words as the tensor-core forms (the bench checks every word against them).
// Launch: entry `fp4_cohort_cc`, grid (max_tokens, rows/128, count), 256 threads, column tiles only. One block = one
// token x 128 rows. K passes in chunks of 1,024 columns.
const CC_CH: usize = 1024; // columns a chunk
const CC_NB: usize = CC_CH / 16; // blocks a chunk
const CC_RS: usize = CC_NB + 1; // row stride of the block sums (padded)
const CC_S: usize = 0; // block sums: 2 * sum + live, [row][block]
const CC_SC: usize = CC_S + 128 * CC_RS; // weight scale bytes, [block][row]
const CC_ACT: usize = CC_SC + CC_NB * 32; // activation codes' signed half units, one word a column
const CC_ASC: usize = CC_ACT + CC_CH; // activation scale bytes
const CC_WORDS: usize = CC_ASC + CC_NB / 4;
const CC_BYTES: usize = CC_WORDS * 4;
const _: () = assert!(CC_BYTES <= 48 * 1024);
const E2M1_HALF: u32 = 0xc864_3210; // magnitudes in half units, one nibble a code 0..7
#[inline(always)]
fn half_units(code: u32) -> i32 {
    let mag = (E2M1_HALF >> ((code & 7) * 4)) & 15;
    if code & 8 != 0 { -(mag as i32) } else { mag as i32 }
}
#[inline(always)]
fn ue4m3_parts(byte: u32) -> (u64, i32) {
    let b = byte & 0x7f;
    let (e, m) = ((b >> 3) as i32, (b & 7) as u64);
    if e == 0 { (m, -6) } else { (8 + m, e - 7) }
}
/// sign * mag * 2^exp to FP32 bits, cut toward zero (the twin's `to_f32`).
#[inline(always)]
fn cut_f32(neg: bool, mag: u64, exp: i32) -> u32 {
    let sign = (neg as u32) << 31;
    if mag == 0 { return sign; }
    let top = 63 - mag.leading_zeros() as i32;
    let e = top + exp;
    let low = if e < -126 { -149 } else { e - 23 };
    let shift = low - exp;
    let kept = if shift <= 0 { mag << ((-shift) as u32) } else if shift >= 64 { 0 } else { mag >> (shift as u32) };
    if e < -126 { return sign | kept as u32; }
    if e > 127 { return sign | 0x7f80_0000; }
    sign | ((e + 127) as u32) << 23 | (kept as u32 & 0x7f_ffff)
}

/// Bench only: the CUDA-core twin form of the same word function. No host path launches it.
/// SAFETY: as `fp4_cohort` (layout 1 only, `chains` 1): the bench checks the same spans and alignments and keeps every
/// owner alive through the launch; grid (max_tokens, rows/128, count), 256 threads, no dynamic shared memory. One
/// writing thread a word; shared reads and writes stay inside this entry's static array; any other launch traps.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_cohort_cc(p: Fp4Cohort) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x};
    if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
        || p.rows % BR as u32 != 0 || p.layout != 1 || p.chains != 1 {
        core::arch::asm!("trap;");
    }
    let part = _block_idx_z() as usize;
    if part >= p.count as usize { return; }
    let e = *(p.table as *const Fp4CohortPart).add(part);
    let (rows, cols, m) = (p.rows as usize, p.cols as usize, e.tokens as usize);
    let (token, row_tile) = (_block_idx_x() as usize, _block_idx_y() as usize);
    if row_tile * BR >= rows || token >= m { return; }
    let tid = _thread_idx_x() as usize;
    let smem = crate::ptx::shared::<CC_BYTES>();
    let word = |w: usize| smem + (w * 4) as u32;
    let codes = (p.codes as *const u32).add((e.first as usize + token) * (cols / 8));
    let ascales = (p.scales as *const u8).add((e.first as usize + token) * (cols / 16));
    let weights = (e.weight as *const u32).add(row_tile * cols * 16); // column k = 16 words at k * 16
    let wscales = (e.scales as *const u32).add(row_tile * (cols / 16) * 32); // block b = 32 words at b * 32
    let (rg, bsel) = (tid % 16, tid / 16);
    let mut acc = 0u32; // +0.0; thread tid < 128 owns row tid
    let mut chunk = 0;
    while chunk < cols {
        let width = (cols - chunk).min(CC_CH);
        let nb = width / 16;
        let mut i = tid;
        while i < width / 8 {
            let w = *codes.add(chunk / 8 + i);
            rep8!(J, { st!(word(CC_ACT + i * 8 + J), half_units((w >> (4 * J)) & 15) as u32); });
            i += THREADS;
        }
        if tid < nb / 4 { st!(word(CC_ASC + tid), *(ascales.add(chunk / 16 + tid * 4) as *const u32)); }
        let mut i = tid;
        while i < nb * 32 { st!(word(CC_SC + i), *wscales.add((chunk / 16) * 32 + i)); i += THREADS; }
        crate::ptx::sync();
        // Exact block sums: thread (rg, bsel) owns rows 8rg..8rg+7 and blocks bsel, bsel + 16, ...
        let mut b = bsel;
        while b < nb {
            let mut w = [0u32; 16];
            rep8!(E, { w[E] = *weights.add((chunk + b * 16 + E) * 16 + rg); w[E + 8] = *weights.add((chunk + b * 16 + E + 8) * 16 + rg); });
            let mut sum = [0i32; 8];
            let mut live = 0u32;
            rep8!(E0, {
                rep4!(E1, { if E1 < 2 {
                    let col = E0 + 8 * E1;
                    let a = ld!(word(CC_ACT + b * 16 + col)) as i32;
                    let x = w[col];
                    if a != 0 { live |= x & 0x7777_7777; }
                    rep8!(I, { sum[I] += half_units((x >> (4 * I)) & 15) * a; });
                } });
            });
            rep8!(I, {
                let r = 8 * rg + I;
                st!(word(CC_S + r * CC_RS + b), (sum[I] * 2 + (((live >> (4 * I)) & 7) != 0) as i32) as u32);
            });
            b += 16;
        }
        crate::ptx::sync();
        // The ordered chain, one row a thread.
        if tid < BR {
            let r = tid;
            let mut st4 = 0;
            while st4 < nb / 4 {
                if acc & 0x7f80_0000 != 0x7f80_0000 {
                    let mut mag = [0u64; 4];
                    let mut neg = [false; 4];
                    let mut exp = [0i32; 4];
                    let mut top = i32::MIN / 4;
                    rep4!(Q, {
                        let blk = st4 * 4 + Q;
                        let v = ld!(word(CC_S + r * CC_RS + blk)) as i32;
                        let (s, live) = (v >> 1, v & 1);
                        let (mw, ew) = ue4m3_parts(crate::ptx::load_u8(word(CC_SC) + (blk * 128 + r) as u32));
                        let (ma, ea) = ue4m3_parts(crate::ptx::load_u8(word(CC_ASC) + blk as u32));
                        mag[Q] = s.unsigned_abs() as u64 * mw * ma;
                        neg[Q] = s < 0;
                        exp[Q] = ew + ea - 8;
                        if mag[Q] != 0 || live != 0 { top = top.max(ew + ea); }
                    });
                    let (ce, cm) = (((acc >> 23) & 255) as i32, (acc & 0x7f_ffff) as u64);
                    let (cmag, cexp, ctop) = if ce == 0 { (cm, -149, -126) } else { (cm | 1 << 23, ce - 150, ce - 127) };
                    top = top.max(ctop);
                    let unit = top - 35;
                    let mut total: i64 = 0;
                    let mut add = |n: bool, mg: u64, ex: i32| {
                        if mg != 0 {
                            let sh = ex - unit;
                            let kept = if sh >= 0 { (mg << (sh as u32)) as i64 } else if -sh >= 64 { 0 } else { (mg >> ((-sh) as u32)) as i64 };
                            total += if n { -kept } else { kept };
                        }
                    };
                    rep4!(Q, { add(neg[Q], mag[Q], exp[Q]); });
                    add(acc >> 31 != 0, cmag, cexp);
                    acc = cut_f32(total < 0, total.unsigned_abs(), unit);
                }
                st4 += 1;
            }
        }
        crate::ptx::sync();
        chunk += CC_CH;
    }
    if tid < BR {
        let row = row_tile * BR + tid;
        let output = e.output as *mut f32;
        let x = f32::from_bits(acc);
        if p.raw == 1 { *output.add(row * m + token) = x; } else {
            let (scale, y): (f32, f32);
            core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e.global, b = in(reg32) e.activation);
            core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
            *output.add(row * m + token) = crate::expert_ops::round(y);
        }
    }
}

/// Register-only peak probe (kit: the hardware limit of the instruction). Every warp runs `chains` (1 to 16)
/// independent accumulator chains for `steps` steps; no operand leaves a register and nothing is read from memory.
/// One FP32 word a thread is written at the end so the chains are live.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Fp4Peak {
    /// blocks * 256 FP32 words.
    pub output: u64,
    pub steps: u32,
    pub chains: u32,
}
/// Bench only. No host path launches it.
/// SAFETY: the bench gives `output` blocks * 256 words, alive through the launch, and launches 256 threads a block.
/// Each thread writes its own word once and reads no memory. No shared memory, no barrier. Any other launch traps.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_peak(p: Fp4Peak) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    if _block_dim_x() as usize != THREADS || p.chains == 0 || p.chains > 16 { core::arch::asm!("trap;"); }
    let tid = _thread_idx_x() as u32;
    // Codes: a fixed nonzero pattern that differs by thread; scales 2^-2 on both sides so the chain stays finite.
    let x = 0x1234_5678u32 ^ tid.wrapping_mul(0x9e37_79b9);
    let a = [x, x.rotate_left(7), x.rotate_left(13), x.rotate_left(21)];
    let b = [x.rotate_left(3), x.rotate_left(17)];
    let (sa, sb) = (0x2828_2828u32, 0x2828_2828u32);
    let mut acc = [[crate::canonical::ZERO; 4]; 4];
    let mut s = 0;
    while s < p.steps {
        rep4!(MI, { rep4!(NJ, {
            if ((MI * 4 + NJ) as u32) < p.chains { acc[MI][NJ] = mma!(a, b, sa, sb, acc[MI][NJ]); }
        }); });
        s += 1;
    }
    let mut sum = 0f32;
    rep4!(MI, { rep4!(NJ, { sum += acc[MI][NJ].a + acc[MI][NJ].d; }); });
    *(p.output as *mut f32).add(_block_idx_x() as usize * THREADS + tid as usize) = sum;
}
