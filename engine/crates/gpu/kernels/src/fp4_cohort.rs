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
//! PAIRED PARTS (`merge` S >= 2, work map only): a work record names the first of S consecutive parts of one expert
//! (the same weights, scales, global and activation; the kernel traps otherwise) and its token tiles run over the S
//! parts' tokens in part order. A slot table in shared memory gives each token slot of the block its own operand row
//! and its own output word base and stride, so every token reads its own operands and stores its own words exactly as
//! its own part would: the weight tile is copied through shared memory once for the S parts' tokens, not once a part.
//! The slot table is used in every form (S = 1: one part, the old addresses).
//! DIRECT PLACE (`rowmap` nonzero): a token's words go to the address its operand row names in the row map (token-major,
//! one row a word), so the down products land in the original pair order the fold reads, with no placement launch.
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
// SL: the slot table, four words a token slot: operand row, output word address (low, high), output stride in words.
const SL: usize = BT * 4;
const WORDS: usize = 2 * STAGE + AT + ATS + SK + SL;
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
    let (a, b, sa, sb, mut c): ([u32; 4], [u32; 2], u32, u32, Acc) = ($a, $b, $sa, $sb, $c);
    core::arch::asm!(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \
         {{{c0},{c1},{c2},{c3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, \
         {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,0}};",
        a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
        b0 = in(reg32) b[0], b1 = in(reg32) b[1],
        c0 = inout(reg32) c.a, c1 = inout(reg32) c.b, c2 = inout(reg32) c.c, c3 = inout(reg32) c.d,
        sa = in(reg32) sa, sb = in(reg32) sb,
    );
    c
}};}
// Straight-line copies, so every array index below is a constant and the arrays stay in registers.
macro_rules! rep2 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}}};}
macro_rules! rep4 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}{const $v:usize=2;$body}{const $v:usize=3;$body}}};}
macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const $v:usize=4;$body}{const $v:usize=5;$body}{const $v:usize=6;$body}{const $v:usize=7;$body}}};}
// Like the load macros, this expands inside the existing kernel safety boundary.
// A device-built map selects a part and token tile; it changes no output word.
macro_rules! routed_tile {($p:expr)=>{{
    if $p.work==0 { (core::arch::nvptx::_block_idx_z() as usize,core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize) }
    else {
        let at=core::arch::nvptx::_block_idx_y();if at>=$p.work_count {return;}
        let tile=*($p.work as *const crate::tc_abi::NvTile).add(at as usize);
        // The caller checks part<count before loading its descriptor. This
        // also returns on the empty marker without another live predicate.
        (tile.part as usize,tile.tile as usize,core::arch::nvptx::_block_idx_x() as usize)
    }
}};}
// PAIRED PARTS: the parts a block reads (1, or `merge` consecutive parts of one expert, work map only), their token
// count, and the slot table of the block's token tile at shared word `$sl` (threads below `$bt`, then a barrier that
// every thread reaches). Expands inside the kernel's safety boundary: the part records are inside the checked table
// (part + span <= count is checked here), the slot words inside the static array.
macro_rules! paired_parts {($p:expr,$part:expr)=>{{
    let span=if $p.merge>1 {$p.merge as usize} else {1};
    if span>1 && ($p.work==0 || $part%span!=0 || $part+span>$p.count as usize) {core::arch::asm!("trap;");}
    let head=*($p.table as *const Fp4CohortPart).add($part);
    let mut m=0usize;let mut s=0;while s<span {m+=(*($p.table as *const Fp4CohortPart).add($part+s)).tokens as usize;s+=1;}
    (head,m,span)
}};}
macro_rules! paired_slots {($p:expr,$part:expr,$span:expr,$head:expr,$tile:expr,$bt:expr,$sl:expr,$word:expr,$tid:expr)=>{{
    let (span,head)=($span,$head);
    if $tid<$bt {
        let n=$tile*$bt+$tid;let (mut row,mut out,mut stride)=(0u32,0u64,0u32);let mut base=0usize;let mut s=0;
        while s<span {
            let r=*($p.table as *const Fp4CohortPart).add($part+s);
            if r.weight!=head.weight || r.scales!=head.scales || r.global.to_bits()!=head.global.to_bits() || r.activation.to_bits()!=head.activation.to_bits() {core::arch::asm!("trap;");}
            let t=r.tokens as usize;
            if n>=base && n<base+t {
                let j=n-base;row=r.first+j as u32;
                // DIRECT PLACE: the operand row's own word (row 0) address, token-major (stride one word a row).
                if $p.rowmap!=0 {out=*($p.rowmap as *const u64).add(row as usize);stride=1;} else {out=r.output+4*j as u64;stride=r.tokens;}
            }
            base+=t;s+=1;
        }
        st!($word($sl+4*$tid),row);st!($word($sl+4*$tid+1),out as u32);st!($word($sl+4*$tid+2),(out>>32) as u32);st!($word($sl+4*$tid+3),stride);
    }
    crate::ptx::sync();
}};}
// The output word (row, slot) of a token slot below live: its own part's word.
macro_rules! slot_out {($sl:expr,$word:expr,$row:expr,$n:expr)=>{{
    let base=(ld!($word($sl+4*$n+1)) as u64)|((ld!($word($sl+4*$n+2)) as u64)<<32);
    (base as *mut f32).add($row*ld!($word($sl+4*$n+3)) as usize)
}};}
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
/// A nonzero work pointer names work_count immutable NvTile records made from
/// checked route metadata. Live records cover each (part, token tile) once;
/// unused records have part=u32::MAX. Its grid is (rows/128, work_count, 1).
/// PAIRED PARTS (`merge` S >= 2, work map only, checked here): a live record names part e*S and a token tile of the
/// S parts' tokens joined in part order; records cover each (expert, joined tile) once. The S parts are inside the
/// table (checked) and carry one expert's weights, scales, global and activation (checked, else trap). A slot's
/// operand row and output word are its own part's, so the one-writer rule and every span above hold per part.
/// The slot table is written by threads below BT before one barrier that every thread reaches (the block's part
/// decides the early return, uniformly, before it).
/// DIRECT PLACE (`rowmap` nonzero): the host passes a row map of one u64 a checked operand row, written by
/// `prompt_route_native` from the route's pairs (a permutation of each set's pairs, each checked below the set's pair
/// count) as the address of that pair's word (row 0) in the set's private output of pairs x rows words; a token's word
/// (row) is stored at that address + 4 * row. Distinct operand rows of a set name distinct pairs, so each word still
/// has one writer, and every address lies inside the set's output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_cohort(p: Fp4Cohort) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
        || p.rows % BR as u32 != 0 || p.layout > 1 || p.chains != 1 || p.merge > 8 {
        core::arch::asm!("trap;");
    }
    let (part,token_tile,row_tile) = routed_tile!(p);
    if part >= p.count as usize { return; }
    let (e, m, span) = paired_parts!(p, part);
    let (rows, cols) = (p.rows as usize, p.cols as usize);
    if row_tile * BR >= rows || token_tile * BT >= m { return; }
    let live = (m - token_tile * BT).min(BT);
    let tid = _thread_idx_x() as usize;
    let smem = crate::ptx::shared::<BYTES>();
    let word = |w: usize| smem + (w * 4) as u32;
    let at = 2 * STAGE;
    let ats = at + AT;
    let sk = ats + ATS;
    let sl = sk + SK;
    paired_slots!(p, part, span, e, token_tile, BT, sl, word, tid);
    let steps = cols / 64;
    let tiled = p.layout == 1;
    let skip = p.skip == 1 && live == 1;
    // This thread's activation copies at step 0: codes of token tid/2 (half tid%2), the scale word of token tid.
    let codes = if tid < live * 2 { (p.codes as *const u8).add(ld!(word(sl + 4 * (tid / 2))) as usize * (cols / 2) + (tid % 2) * 16) } else { p.codes as *const u8 };
    let ascales = if tid < live { (p.scales as *const u8).add(ld!(word(sl + 4 * tid)) as usize * (cols / 16)) } else { p.scales as *const u8 };
    if skip {
        // The one live token's codes, cols/2 bytes as cols/8 words.
        let one = (p.codes as *const u8).add(ld!(word(sl)) as usize * (cols / 2));
        let mut i = tid;
        while i < cols / 8 { st!(word(sk + i), *(one as *const u32).add(i)); i += THREADS; }
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
        // Activations: 32 bytes a token as two 16-byte copies (thread 2n + q), and one scale word a token (thread n).
        if tid < live * 2 {
            let (n, q) = (tid / 2, tid % 2);
            crate::ptx::copy16(word(s + AR + AS + n * 12 + q * 4), codes.add(c * 32).cast());
        }
        if tid < live {
            crate::ptx::copy4(word(s + AR + AS + B + tid), ascales.add(c * 4).cast());
        }
    };
    let (lane, warp) = (tid % 32, tid / 32);
    let (g, t) = (lane / 4, lane % 4);
    let (wr, wc) = (warp % 2, warp / 2);
    let mut acc = [[crate::canonical::ZERO; 4]; 4];
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
    // Re-read scale from part table after K loop and flush via per-MI shared park + runtime NJ loop so LLVM hoists
    // zero output pointers across the K loop (<= 128 regs -> 2 CTAs/SM, 0 spills). `smem[0..16384]` is already free
    // and synchronized by the final `crate::ptx::sync()` at the end of the K loop (`2*STAGE*4 = 24,832 B`).
    let e_end = *(p.table as *const Fp4CohortPart).add(part);
    let scale: f32;
    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e_end.global, b = in(reg32) e_end.activation);
    let ep = smem + (tid * 64) as u32;
    let put = |ptr: *mut f32, x: f32| {
        if p.raw == 1 { *ptr = x; } else {
            let y: f32;
            core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
            *ptr = crate::expert_ops::round(y);
        }
    };
    rep4!(MI, {
        rep4!(NJ, {
            let v = acc[MI][NJ];
            core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                p = in(reg32) ep + (NJ * 16) as u32,
                a = in(reg32) v.a, b = in(reg32) v.b, c = in(reg32) v.c, d = in(reg32) v.d, options(nostack));
        });
        let r = row_tile * BR + 64 * wr + 16 * MI + g;
        let mut nj = 0u32;
        while nj < 4 {
            core::arch::asm!("/* {0} */", inout(reg32) nj, options(nomem, nostack));
            let nju = nj as usize;
            let q = crate::ptx::load4(ep + (nju * 16) as u32);
            let n = 32 * wc + 8 * nju + 2 * t;
            if n < live {
                let base0 = ((ld!(word(sl + 4 * n + 1)) as u64) | ((ld!(word(sl + 4 * n + 2)) as u64) << 32)) as *mut f32;
                let str0 = ld!(word(sl + 4 * n + 3)) as usize;
                put(base0.add(r * str0), f32::from_bits(q[0]));
                put(base0.add((r + 8) * str0), f32::from_bits(q[2]));
            }
            if n + 1 < live {
                let base1 = ((ld!(word(sl + 4 * (n + 1) + 1)) as u64) | ((ld!(word(sl + 4 * (n + 1) + 2)) as u64) << 32)) as *mut f32;
                let str1 = ld!(word(sl + 4 * (n + 1) + 3)) as usize;
                put(base1.add(r * str1), f32::from_bits(q[1]));
                put(base1.add((r + 8) * str1), f32::from_bits(q[3]));
            }
            nj += 1;
        }
    });
}

// Two more forms of the same contract (the same chain a word, so the same words), for the speed questions: a 64-token
// block with half the accumulators a warp and three stages, and an 8-token block for the decode step with four stages
// in flight. `$bt` tokens a block, `$wr` warps along the 128 rows (8 / $wr along the tokens), `$st` stages.
macro_rules! cohort_body {
    ($p:expr, $bt:expr, $wr:expr, $st:expr) => {{
            let p: Fp4Cohort = $p;
            use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
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
            const WORDS: usize = AT0 + AT + ATS + SK + BT * 4;
            const BYTES: usize = WORDS * 4;
            const _: () = assert!(BYTES <= 48 * 1024 && MIC <= 4 && NJC <= 4 && NJC >= 1 && S >= 2);
            if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
                || p.rows % BR as u32 != 0 || p.layout > 1 || !(p.chains == 1 || p.chains == 4 && p.cols % 256 == 0) || p.merge > 8 {
                core::arch::asm!("trap;");
            }
            let (part,token_tile,row_tile) = if BT==64 || BT==8 {routed_tile!(p)} else {
                (core::arch::nvptx::_block_idx_z() as usize,core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize)
            };
            if part >= p.count as usize { return; }
            let pair_route = BT == 8 && p.rowmap != 0;
            let (e, m, span) = if pair_route {
                if p.work == 0 { core::arch::asm!("trap;"); }
                let head = *(p.table as *const Fp4CohortPart).add(part);
                (head, head.tokens as usize, 1usize)
            } else {
                paired_parts!(p, part)
            };
            let (rows, cols) = (p.rows as usize, p.cols as usize);
            if row_tile * BR >= rows || token_tile * BT >= m { return; }
            let live = (m - token_tile * BT).min(BT);
            let tid = _thread_idx_x() as usize;
            let smem = crate::ptx::shared::<BYTES>();
            let word = |w: usize| smem + (w * 4) as u32;
            let (at, ats, sk) = (AT0, AT0 + AT, AT0 + AT + ATS);
            let sl = sk + SK;
            if pair_route {
                if tid < BT {
                    let n = token_tile * BT + tid;
                    let (mut row, mut out, mut stride) = (0u32, 0u64, 0u32);
                    if n < m {
                        let pair = *(p.rowmap as *const u32).add(e.first as usize + n);
                        row = if p.merge != 0 { pair / p.merge } else { pair };
                        out = e.output + (pair as u64) * (rows as u64) * 4;
                        stride = 1;
                    }
                    st!(word(sl + 4 * tid), row);
                    st!(word(sl + 4 * tid + 1), out as u32);
                    st!(word(sl + 4 * tid + 2), (out >> 32) as u32);
                    st!(word(sl + 4 * tid + 3), stride);
                }
                crate::ptx::sync();
            } else {
                paired_slots!(p, part, span, e, token_tile, BT, sl, word, tid);
            }
            let steps = cols / 64;
            let tiled = p.layout == 1;
            let skip = p.skip == 1 && live == 1;
            let codes = if tid < live * 2 { (p.codes as *const u8).add(ld!(word(sl + 4 * (tid / 2))) as usize * (cols / 2) + (tid % 2) * 16) } else { p.codes as *const u8 };
            let ascales = if tid < live { (p.scales as *const u8).add(ld!(word(sl + 4 * tid)) as usize * (cols / 16)) } else { p.scales as *const u8 };
            if skip {
                let one = (p.codes as *const u8).add(ld!(word(sl)) as usize * (cols / 2));
                let mut i = tid;
                while i < cols / 8 { st!(word(sk + i), *(one as *const u32).add(i)); i += THREADS; }
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
                if tid < live * 2 {
                    let (n, q) = (tid / 2, tid % 2);
                    crate::ptx::copy16(word(s + AR + AS + n * 12 + q * 4), codes.add(c * 32).cast());
                }
                if tid < live {
                    crate::ptx::copy4(word(s + AR + AS + BW + tid), ascales.add(c * 4).cast());
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
                // For S >= 4 (BT == 8), stage (c + S - 1) % S = (c - 1) % 4 was last read before both syncs above,
                // so issuing the next stage before mma overlaps DRAM copies with tensor-core compute and drops 1 sync/step.
                if BT == 8 {
                    if c + S - 1 < steps { issue(c + S - 1); }
                    crate::ptx::commit();
                }
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
                if BT != 8 {
                    crate::ptx::sync();
                    if c + S - 1 < steps { issue(c + S - 1); }
                    crate::ptx::commit();
                }
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
                            let n = (BT / WC) * wc + 8 * NJ + 2 * t; // the token slot in the block
                            let at = |row: usize, slot: usize| slot_out!(sl, word, row, slot);
                            if quarter == 1 {
                                if n < live { *at(r, n) = pair.a; *at(r + 8, n) = pair.c; }
                                if n + 1 < live { *at(r, n + 1) = pair.b; *at(r + 8, n + 1) = pair.d; }
                                acc[MI][NJ] = crate::canonical::ZERO;
                            } else {
                                let mut first = crate::canonical::ZERO;
                                if n < live { first.a = *at(r, n); first.c = *at(r + 8, n); }
                                if n + 1 < live { first.b = *at(r, n + 1); first.d = *at(r + 8, n + 1); }
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
                        let n = (BT / WC) * wc + 8 * NJ + 2 * t; // the token slot in the block
                        let v = acc[MI][NJ];
                        let put = |row: usize, slot: usize, x: f32| {
                            if slot < live && p.raw == 1 { *slot_out!(sl, word, row, slot) = x; } else if slot < live {
                                let y: f32;
                                core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                                *slot_out!(sl, word, row, slot) = crate::expert_ops::round(y);
                            }
                        };
                        put(r, n, v.a);
                        put(r, n + 1, v.b);
                        put(r + 8, n, v.c);
                        put(r + 8, n + 1, v.d);
                    }
                });
            });
    }};
}
macro_rules! cohort_entry {
    ($name:ident, $bt:expr, $wr:expr, $st:expr) => {
        /// SAFETY: as `fp4_cohort` (PAIRED PARTS included), for this block shape: grid (ceil(max_tokens/BT), rows/128,
        /// count), or (rows/128, work_count, 1) with a work map in the 64-token and 8-token forms, 256 threads;
        /// `chains` 1, or 4 with cols a multiple of 256. With `chains` 4 a thread stores fl(a0 + a1) of its OWN words in
        /// the part's output after the second quarter and loads its own words back at the join after the fourth. No
        /// other thread reads or writes those words (one writer a word), no barrier is needed for a thread's own store
        /// and load, and the output span must be exclusive to this launch until stream completion: nothing else may
        /// read it before, because it holds partial sums until the last step. In the 8-token form with `rowmap != 0`,
        /// `work` is required and `rowmap` holds checked pair indices built by `device_route_tc_rows` (distinct pairs
        /// per phase; operand row = `pair / merge` if `merge != 0` else `pair`; output = `e.output + pair * rows * 4`).
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(p: Fp4Cohort) {
            if $bt == 8 { cohort_t8_body!(p); } else { cohort_body!(p, $bt, $wr, $st); }
        }
    };
}
/// Zero-AT-buffer, 1-barrier-per-step 8-token cohort body for `fp4_cohort_t8` and `resident_launch_fp4_cohort_t8`.
/// Eight warps (256 threads) each own one `m16n8k64` row tile (16 rows x up to 8 tokens). Column-tile weights stage
/// with 4-way XOR swizzle (`copy16_stream` + L2 prefetch) and transpose in-register via 3 butterfly shuffles + 1 idx
/// shuffle — zero `AT`/`ATS` SMEM buffers and only ONE `bar.sync 0` per 64-column step (24,320 B SMEM -> 4 blocks/SM).
macro_rules! cohort_t8_body {
    ($p:expr) => {{
        let p: Fp4Cohort = $p;
        use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
        const BT: usize = 8;
        const S: usize = 4;
        const T8_AR: usize = 1024;
        const T8_AS: usize = 128;
        const T8_BW: usize = BT * 12;
        const T8_STAGE: usize = T8_AR + T8_AS + T8_BW + BT;
        const T8_SK0: usize = S * T8_STAGE;
        const T8_SL0: usize = T8_SK0 + SK;
        const T8_WORDS: usize = T8_SL0 + BT * 4;
        const T8_BYTES: usize = T8_WORDS * 4;
        const _: () = assert!(T8_BYTES == 24_320);
        if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
            || p.rows % BR as u32 != 0 || p.layout > 1 || !(p.chains == 1 || p.chains == 4 && p.cols % 256 == 0) || p.merge > 8 {
            core::arch::asm!("trap;");
        }
        let (part, token_tile, row_tile) = routed_tile!(p);
        if part >= p.count as usize { return; }
        let pair_route = p.rowmap != 0;
        let (e, m, span) = if pair_route {
            if p.work == 0 { core::arch::asm!("trap;"); }
            let head = *(p.table as *const Fp4CohortPart).add(part);
            (head, head.tokens as usize, 1usize)
        } else {
            paired_parts!(p, part)
        };
        let (rows, cols) = (p.rows as usize, p.cols as usize);
        if row_tile * BR >= rows || token_tile * BT >= m { return; }
        let live = (m - token_tile * BT).min(BT);
        let tid = _thread_idx_x() as usize;
        let smem = crate::ptx::shared::<T8_BYTES>();
        let word = |w: usize| smem + (w * 4) as u32;
        let (sk, sl) = (T8_SK0, T8_SL0);
        if pair_route {
            if tid < BT {
                let n = token_tile * BT + tid;
                let (mut row, mut out, mut stride) = (0u32, 0u64, 0u32);
                if n < m {
                    let pair = *(p.rowmap as *const u32).add(e.first as usize + n);
                    row = if p.merge != 0 { pair / p.merge } else { pair };
                    out = e.output + (pair as u64) * (rows as u64) * 4;
                    stride = 1;
                }
                st!(word(sl + 4 * tid), row);
                st!(word(sl + 4 * tid + 1), out as u32);
                st!(word(sl + 4 * tid + 2), (out >> 32) as u32);
                st!(word(sl + 4 * tid + 3), stride);
            }
            crate::ptx::sync();
        } else {
            paired_slots!(p, part, span, e, token_tile, BT, sl, word, tid);
        }
        let steps = cols / 64;
        let tiled = p.layout == 1;
        let skip = p.skip == 1 && live <= 2;
        let codes = if tid < live * 2 { (p.codes as *const u8).add(ld!(word(sl + 4 * (tid / 2))) as usize * (cols / 2) + (tid % 2) * 16) } else { p.codes as *const u8 };
        let ascales = if tid < live { (p.scales as *const u8).add(ld!(word(sl + 4 * tid)) as usize * (cols / 16)) } else { p.scales as *const u8 };
        if skip {
            let mut tok = 0usize;
            while tok < live {
                let one = (p.codes as *const u8).add(ld!(word(sl + 4 * tok)) as usize * (cols / 2)) as *const u32;
                let mut i = tid;
                while i < cols / 8 {
                    let v = *one.add(i);
                    st!(word(sk + i), if tok == 0 { v } else { ld!(word(sk + i)) | v });
                    i += THREADS;
                }
                tok += 1;
            }
            crate::ptx::sync();
        }
        let weights = e.weight as *const u8;
        let wscales = e.scales as *const u8;
        let issue = |c: usize| {
            let s = (c & 3) * T8_STAGE;
            if tiled {
                let (col, q) = (tid / 4, tid % 4);
                let column = c * 64 + col;
                let swz_q = q ^ ((col >> 1) & 3);
                let dst = word(s + col * 16 + swz_q * 4);
                let zero = skip && (ld!(word(sk + column / 8)) >> (4 * (column & 7)) & 7) == 0;
                if zero { zero16!(dst); } else {
                    crate::ptx::copy16(dst, weights.add(((row_tile * cols + column) * 64) + q * 16).cast());
                }
                if tid < 32 {
                    crate::ptx::copy16(word(s + T8_AR + tid * 4), wscales.add((row_tile * (cols / 16) + c * 4) * 128 + tid * 16).cast());
                }
            } else {
                let (row, q) = (tid / 2, tid % 2);
                let r = row_tile * BR + row;
                crate::ptx::copy16(word(s + row * 8 + q * 4), weights.add(r * (cols / 2) + c * 32 + q * 16).cast());
            }
            if tid < live * 2 {
                let (n, q) = (tid / 2, tid % 2);
                crate::ptx::copy16(word(s + T8_AR + T8_AS + n * 12 + q * 4), codes.add(c * 32).cast());
            }
            if tid < live {
                crate::ptx::copy4(word(s + T8_AR + T8_AS + T8_BW + tid), ascales.add(c * 4).cast());
            }
        };
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let j = lane & 7;
        let remap_src = (t * 8 + g) as u32;
        let mut acc = crate::canonical::ZERO;
        let mut hold = crate::canonical::ZERO;
        let quarters = p.chains == 4;
        let per_chain = steps / p.chains as usize;
        macro_rules! shfl_bfly {($v:expr,$d:expr)=>{{let o:u32;core::arch::asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)o,v=in(reg32)$v,d=const {$d},options(pure,nomem,nostack));o}};}
        macro_rules! shfl_idx {($v:expr,$s:expr)=>{{let o:u32;core::arch::asm!("shfl.sync.idx.b32 {o}, {v}, {s}, 31, -1;",o=out(reg32)o,v=in(reg32)$v,s=in(reg32)$s,options(pure,nomem,nostack));o}};}
        macro_rules! load_8x32 {($s_code:expr,$rb:expr,$col_base:expr)=>{{
            let col = $col_base + lane;
            let q = ($rb) >> 2;
            let sub = ($rb) & 3;
            let swz_q = q ^ ((col >> 1) & 3);
            let mut v = ld!($s_code + (col * 64 + swz_q * 16 + sub * 4) as u32);
            let p4 = shfl_bfly!(v, 4);
            v = if (j & 4) == 0 { (v & 0x0000_ffff) | ((p4 & 0x0000_ffff) << 16) } else { (p4 >> 16) | (v & 0xffff_0000) };
            let p2 = shfl_bfly!(v, 2);
            v = if (j & 2) == 0 { (v & 0x00ff_00ff) | ((p2 & 0x00ff_00ff) << 8) } else { ((p2 >> 8) & 0x00ff_00ff) | (v & 0xff00_ff00) };
            let p1 = shfl_bfly!(v, 1);
            v = if (j & 1) == 0 { (v & 0x0f0f_0f0f) | ((p1 & 0x0f0f_0f0f) << 4) } else { ((p1 >> 4) & 0x0f0f_0f0f) | (v & 0xf0f0_f0f0) };
            shfl_idx!(v, remap_src)
        }};}
        {
            let mut i = live * 2 + tid;
            while i < BT * 2 {
                let at = T8_AR + T8_AS + (i / 2) * 12 + (i % 2) * 4;
                let mut stage = 0;
                while stage < S { zero16!(word(stage * T8_STAGE + at)); stage += 1; }
                i += THREADS;
            }
            if tid >= live && tid < BT {
                let mut stage = 0;
                while stage < S { st!(word(stage * T8_STAGE + T8_AR + T8_AS + T8_BW + tid), 0u32); stage += 1; }
            }
        }
        let mut k = 0;
        while k < S - 1 { if k < steps { issue(k); } crate::ptx::commit(); k += 1; }
        let mut c = 0;
        while c < steps {
            core::arch::asm!("cp.async.wait_group 2;");
            crate::ptx::sync();
            if c + S - 1 < steps { issue(c + S - 1); }
            crate::ptx::commit();
            let s = (c & 3) * T8_STAGE;
            let (b0, b1, sb) = if g < live {
                let tok = s + T8_AR + T8_AS + g * 12;
                (ld!(word(tok + t)), ld!(word(tok + t + 4)), ld!(word(s + T8_AR + T8_AS + T8_BW + g)))
            } else { (0u32, 0u32, 0u32) };
            let (a, sa) = if tiled {
                let s_code = word(s);
                let rb0 = warp * 2;
                let a = [load_8x32!(s_code, rb0, 0), load_8x32!(s_code, rb0 + 1, 0), load_8x32!(s_code, rb0, 32), load_8x32!(s_code, rb0 + 1, 32)];
                let rs = word(s + T8_AR) + (warp * 16 + g + (t & 1) * 8) as u32;
                let sa = crate::ptx::load_u8(rs) | (crate::ptx::load_u8(rs + 128) << 8)
                    | (crate::ptx::load_u8(rs + 256) << 16) | (crate::ptx::load_u8(rs + 384) << 24);
                (a, sa)
            } else {
                let r0 = warp * 16;
                let a = [ld!(word(s + (r0 + g) * 8 + t)), ld!(word(s + (r0 + g + 8) * 8 + t)),
                    ld!(word(s + (r0 + g) * 8 + t + 4)), ld!(word(s + (r0 + g + 8) * 8 + t + 4))];
                let sa = *(wscales.add((row_tile * BR + r0 + g + (t & 1) * 8) * (cols / 16) + c * 4) as *const u32);
                (a, sa)
            };
            acc = mma!(a, [b0, b1], sa, sb, acc);
            c += 1;
            if quarters && c % per_chain == 0 {
                let quarter = c / per_chain - 1;
                if quarter == 0 || quarter == 2 {
                    hold = acc;
                    acc = crate::canonical::ZERO;
                } else {
                    let pair = add_acc!(hold, acc);
                    let r = row_tile * BR + 16 * warp + g;
                    let n = 2 * t;
                    let at = |row: usize, slot: usize| slot_out!(sl, word, row, slot);
                    if quarter == 1 {
                        if n < live { *at(r, n) = pair.a; *at(r + 8, n) = pair.c; }
                        if n + 1 < live { *at(r, n + 1) = pair.b; *at(r + 8, n + 1) = pair.d; }
                        acc = crate::canonical::ZERO;
                    } else {
                        let mut first = crate::canonical::ZERO;
                        if n < live { first.a = *at(r, n); first.c = *at(r + 8, n); }
                        if n + 1 < live { first.b = *at(r, n + 1); first.d = *at(r + 8, n + 1); }
                        acc = add_acc!(first, pair);
                    }
                }
            }
        }
        let scale: f32;
        core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e.global, b = in(reg32) e.activation);
        let r = row_tile * BR + 16 * warp + g;
        let n = 2 * t;
        let put = |row: usize, slot: usize, x: f32| {
            if slot < live && p.raw == 1 { *slot_out!(sl, word, row, slot) = x; } else if slot < live {
                let y: f32;
                core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                *slot_out!(sl, word, row, slot) = crate::expert_ops::round(y);
            }
        };
        put(r, n, acc.a);
        put(r, n + 1, acc.b);
        put(r + 8, n, acc.c);
        put(r + 8, n + 1, acc.d);
    }};
}
cohort_entry!(fp4_cohort_t64, 64, 4, 3);
cohort_entry!(fp4_cohort_t8, 8, 8, 4);
/// Recorded replay of `fp4_cohort_t8`: reads the checked `NvRows` payload from the indirect command table.
/// SAFETY: As `fp4_cohort_t8` (grid `(rows/128, work_count, 1)`, 256 threads). The checked recorder serializes
/// `NvRows` into `table[index].payload` with kind `Kind::Fp4CohortT8` (75) and retains every buffer owner
/// through stream completion. Any kind mismatch traps before touching memory.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fp4_cohort_t8(table: *const crate::resident_launch_abi::Command, index: u64) {
    let c = &*table.add(index as usize);
    if c.kind != crate::resident_launch_abi::Kind::Fp4CohortT8 as u32 { core::arch::nvptx::trap(); }
    cohort_t8_body!(c.payload.as_ptr().cast::<Fp4Cohort>().read());
}

/// The 128-token quarter-chain form (`chains == 4`, second matrices): parks quarter 0 and `fl(a0 + a1)` in each
/// thread's own disjoint output words (`slot_out!`) and quarter 2 (`a2`) in dynamic shared memory (`65,536 B`),
/// keeping the entire 128x128 working set + `park2` in `97,536 B` dynamic shared memory (`<= 99 KiB`) with zero
/// `hold` registers (<= 176 regs, 0 spills) and 1.64x fewer CTAs/weight reads than `fp4_cohort_t64`.
/// SAFETY: As `fp4_cohort`, with `chains == 4`, `cols % 256 == 0`, and dynamic shared memory of at least `97,536 B`
/// (`32,000 B` staging + `65,536 B` quarter-2 park, checked here via `%dynamic_smem_size`, zero static smem).
/// Each thread parks quarter 0 and `fl(a0 + a1)` in its own disjoint output words (no other reader or writer until
/// stream completion) and quarter 2 in its own disjoint 256-byte dynamic shared slice `32000 + tid * 256`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_cohort_q128(p: Fp4Cohort) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    const Q_WORDS: usize = 2 * STAGE + AT + ATS + SL;
    const Q_BASE_BYTES: usize = Q_WORDS * 4;
    const Q_PARK_BYTES: usize = THREADS * 16 * 16;
    const Q_TOTAL_BYTES: u32 = (Q_BASE_BYTES + Q_PARK_BYTES) as u32;
    const _: () = assert!(Q_BASE_BYTES == 32_000 && Q_PARK_BYTES == 65_536 && Q_TOTAL_BYTES == 97_536);
    let dynamic: u32;
    core::arch::asm!("mov.u32 {d}, %dynamic_smem_size;", d = out(reg32) dynamic, options(nomem, nostack));
    if _block_dim_x() as usize != THREADS || dynamic < Q_TOTAL_BYTES || p.count == 0 || p.cols == 0 || p.cols % 256 != 0
        || p.cols > 8192 || p.rows % BR as u32 != 0 || p.layout > 1 || p.chains != 4 || p.merge > 8
    {
        core::arch::asm!("trap;");
    }
    let (part, token_tile, row_tile) = routed_tile!(p);
    if part >= p.count as usize { return; }
    let (e, m, span) = paired_parts!(p, part);
    let (rows, cols) = (p.rows as usize, p.cols as usize);
    if row_tile * BR >= rows || token_tile * BT >= m { return; }
    let live = (m - token_tile * BT).min(BT);
    let tid = _thread_idx_x() as usize;
    let smem = 0u32;
    let word = |w: usize| smem + (w * 4) as u32;
    let at = 2 * STAGE;
    let ats = at + AT;
    let sl = ats + ATS;
    let park2 = (Q_BASE_BYTES + tid * 16 * 16) as u32;
    paired_slots!(p, part, span, e, token_tile, BT, sl, word, tid);
    let steps = cols / 64;
    let per_chain = steps / 4;
    let tiled = p.layout == 1;
    let skip = p.skip == 1 && live == 1;
    let one_codes = if skip { (p.codes as *const u32).add(ld!(word(sl)) as usize * (cols / 8)) } else { p.codes as *const u32 };
    let codes = if tid < live * 2 { (p.codes as *const u8).add(ld!(word(sl + 4 * (tid / 2))) as usize * (cols / 2) + (tid % 2) * 16) } else { p.codes as *const u8 };
    let ascales = if tid < live { (p.scales as *const u8).add(ld!(word(sl + 4 * tid)) as usize * (cols / 16)) } else { p.scales as *const u8 };
    let weights = e.weight as *const u8;
    let wscales = e.scales as *const u8;
    let issue = |c: usize| {
        let s = (c & 1) * STAGE;
        if tiled {
            let (col, q) = (tid / 4, tid % 4);
            let column = c * 64 + col;
            let dst = word(s + col * 16 + (col / 8) * 4 + q * 4);
            let zero = skip && ((*one_codes.add(column / 8) >> (4 * (column % 8))) & 7) == 0;
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
        if tid < live * 2 {
            let (n, q) = (tid / 2, tid % 2);
            crate::ptx::copy16(word(s + AR + AS + n * 12 + q * 4), codes.add(c * 32).cast());
        }
        if tid < live {
            crate::ptx::copy4(word(s + AR + AS + B + tid), ascales.add(c * 4).cast());
        }
    };
    let (lane, warp) = (tid % 32, tid / 32);
    let (g, t) = (lane / 4, lane % 4);
    let (wr, wc) = (warp % 2, warp / 2);
    let mut acc = [[crate::canonical::ZERO; 4]; 4];
    {
        let mut i = live * 2 + tid;
        while i < BT * 2 {
            let at_w = AR + AS + (i / 2) * 12 + (i % 2) * 4;
            zero16!(word(at_w)); zero16!(word(STAGE + at_w));
            i += THREADS;
        }
        if tid >= live && tid < BT { st!(word(AR + AS + B + tid), 0u32); st!(word(STAGE + AR + AS + B + tid), 0u32); }
    }
    issue(0);
    crate::ptx::commit();
    let mut c = 0usize;
    let mut quarter = 0usize;
    while quarter < 4 {
        let q_end = (quarter + 1) * per_chain;
        while c < q_end {
            if c + 1 < steps { issue(c + 1); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
            crate::ptx::sync();
            let s = (c & 1) * STAGE;
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
        if quarter == 0 || quarter == 2 {
            // Park Q0 or Q2 in dynamic shared memory (`park2`), keeping Q1's and Q3's partner register-free.
            rep4!(MI, { rep4!(NJ, {
                let v = acc[MI][NJ];
                core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                    p = in(reg32) park2 + ((MI * 4 + NJ) * 16) as u32,
                    a = in(reg32) v.a, b = in(reg32) v.b, c = in(reg32) v.c, d = in(reg32) v.d, options(nostack));
                acc[MI][NJ] = crate::canonical::ZERO;
            }); });
        } else if quarter == 1 {
            // Join fl(a0 + a1) into `park2`, then flush to this thread's own private output words via a runtime loop
            // so LLVM hoists zero global output pointers across the K loop.
            rep4!(MI, { rep4!(NJ, {
                let q = crate::ptx::load4(park2 + ((MI * 4 + NJ) * 16) as u32);
                let a0 = Acc { a: f32::from_bits(q[0]), b: f32::from_bits(q[1]), c: f32::from_bits(q[2]), d: f32::from_bits(q[3]) };
                let pair = add_acc!(a0, acc[MI][NJ]);
                core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                    p = in(reg32) park2 + ((MI * 4 + NJ) * 16) as u32,
                    a = in(reg32) pair.a, b = in(reg32) pair.b, c = in(reg32) pair.c, d = in(reg32) pair.d, options(nostack));
                acc[MI][NJ] = crate::canonical::ZERO;
            }); });
            let mut nj = 0usize;
            while nj < 4 {
                let n = 32 * wc + 8 * nj + 2 * t;
                let live0 = n < live;
                let live1 = n + 1 < live;
                let (base0, str0) = if live0 {
                    (((ld!(word(sl + 4 * n + 1)) as u64) | ((ld!(word(sl + 4 * n + 2)) as u64) << 32)) as *mut f32,
                     ld!(word(sl + 4 * n + 3)) as usize)
                } else { (core::ptr::null_mut(), 0) };
                let (base1, str1) = if live1 {
                    (((ld!(word(sl + 4 * (n + 1) + 1)) as u64) | ((ld!(word(sl + 4 * (n + 1) + 2)) as u64) << 32)) as *mut f32,
                     ld!(word(sl + 4 * (n + 1) + 3)) as usize)
                } else { (core::ptr::null_mut(), 0) };
                let mut mi = 0usize;
                while mi < 4 {
                    let q = crate::ptx::load4(park2 + ((mi * 4 + nj) * 16) as u32);
                    let r = row_tile * BR + 64 * wr + 16 * mi + g;
                    if live0 { *base0.add(r * str0) = f32::from_bits(q[0]); *base0.add((r + 8) * str0) = f32::from_bits(q[2]); }
                    if live1 { *base1.add(r * str1) = f32::from_bits(q[1]); *base1.add((r + 8) * str1) = f32::from_bits(q[3]); }
                    mi += 1;
                }
                nj += 1;
            }
        }
        quarter += 1;
    }
    let scale: f32;
    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e.global, b = in(reg32) e.activation);
    rep4!(MI, { rep4!(NJ, {
        let q = crate::ptx::load4(park2 + ((MI * 4 + NJ) * 16) as u32);
        let a2 = Acc { a: f32::from_bits(q[0]), b: f32::from_bits(q[1]), c: f32::from_bits(q[2]), d: f32::from_bits(q[3]) };
        let second = add_acc!(a2, acc[MI][NJ]);
        core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
            p = in(reg32) park2 + ((MI * 4 + NJ) * 16) as u32,
            a = in(reg32) second.a, b = in(reg32) second.b, c = in(reg32) second.c, d = in(reg32) second.d, options(nostack));
    }); });
    let mut nj = 0usize;
    while nj < 4 {
        let n = 32 * wc + 8 * nj + 2 * t;
        let live0 = n < live;
        let live1 = n + 1 < live;
        let (base0, str0) = if live0 {
            (((ld!(word(sl + 4 * n + 1)) as u64) | ((ld!(word(sl + 4 * n + 2)) as u64) << 32)) as *mut f32,
             ld!(word(sl + 4 * n + 3)) as usize)
        } else { (core::ptr::null_mut(), 0) };
        let (base1, str1) = if live1 {
            (((ld!(word(sl + 4 * (n + 1) + 1)) as u64) | ((ld!(word(sl + 4 * (n + 1) + 2)) as u64) << 32)) as *mut f32,
             ld!(word(sl + 4 * (n + 1) + 3)) as usize)
        } else { (core::ptr::null_mut(), 0) };
        let mut mi = 0usize;
        while mi < 4 {
            let q = crate::ptx::load4(park2 + ((mi * 4 + nj) * 16) as u32);
            let second = Acc { a: f32::from_bits(q[0]), b: f32::from_bits(q[1]), c: f32::from_bits(q[2]), d: f32::from_bits(q[3]) };
            let r = row_tile * BR + 64 * wr + 16 * mi + g;
            let mut first = crate::canonical::ZERO;
            if live0 { first.a = *base0.add(r * str0); first.c = *base0.add((r + 8) * str0); }
            if live1 { first.b = *base1.add(r * str1); first.d = *base1.add((r + 8) * str1); }
            let v = add_acc!(first, second);
            let put = |ptr: *mut f32, x: f32| {
                if p.raw == 1 { *ptr = x; } else {
                    let y: f32;
                    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                    *ptr = crate::expert_ops::round(y);
                }
            };
            if live0 { put(base0.add(r * str0), v.a); put(base0.add((r + 8) * str0), v.c); }
            if live1 { put(base1.add(r * str1), v.b); put(base1.add((r + 8) * str1), v.d); }
            mi += 1;
        }
        nj += 1;
    }
}

// 64 weight rows x 128 tokens (256 threads, WR=2, WC=4, MIC=2, NJC=4): halves accumulator register pressure
// (32 regs vs 64 regs) so both single-chain (`fp4_cohort_r64`, 18,688 B static smem) and quarter-chain
// (`fp4_cohort_q64r128`, 49,408 B dynamic smem, ZERO global parking, coalesced rowmap epilogue) hit 2 CTAs/SM
// at <= 128 registers with 0 spills and bank-conflict-free XOR-swizzled shared memory.
const BR64: usize = 64;
const AR64: usize = 544; // 64 cols x 8 words + 8 groups x 4 pad words
const AS64: usize = 64; // 4 scale blocks x 16 words (64 rows)
const B64: usize = BT * 8; // 128 tokens x 8 words, XOR-swizzled on q ^ ((n >> 2) & 1)
const BS64: usize = BT;
const STAGE64: usize = AR64 + AS64 + B64 + BS64;
const AT64: usize = BR64 * 8 + (BR64 / 8) * 8; // 64 rows x 8 words + 8 groups x 8 pad, XOR-swizzled on k ^ (r & 4)
const ATS64: usize = BR64;
const SL64: usize = BT * 4;
const R64_STAGE_WORDS: usize = 2 * STAGE64 + AT64 + ATS64;
const R64_WORDS: usize = R64_STAGE_WORDS + SL64;
const R64_BYTES: usize = R64_WORDS * 4;
const _: () = assert!(R64_STAGE_WORDS * 4 == 16_640 && R64_BYTES == 18_688 && THREADS * 64 <= R64_STAGE_WORDS * 4);

/// High-occupancy 64x128 single-chain form (`chains == 1`, routed gate/up): 18,688 B static shared memory,
/// 32 accumulator registers (<= 128 regs -> 2+ CTAs/SM, 0 spills), exact same contract (b) words as `fp4_cohort`.
/// SAFETY: As `fp4_cohort`, with grid `(rows/64, work_count, 1)` (work map) or `(ceil(max_tokens/128), rows/64, count)`
/// (direct). `p.rows % 128 == 0` ensures every `tile128 = row_tile >> 1` and `half = row_tile & 1` indexes inside a
/// valid 128-row column tile. Shared reads/writes stay inside the `18,688 B` static array; `smem[0..16384]` used for
/// epilogue staging is disjoint from `sl` at `16,640..18,688` and synchronized by the K loop's final barrier.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_cohort_r64(p: Fp4Cohort) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    if _block_dim_x() as usize != THREADS || p.count == 0 || p.cols == 0 || p.cols % 64 != 0 || p.cols > 8192
        || p.rows % BR as u32 != 0 || p.layout > 1 || p.chains != 1 || p.merge > 8
    {
        core::arch::asm!("trap;");
    }
    let (part, token_tile, row_tile) = routed_tile!(p);
    if part >= p.count as usize { return; }
    let (e, m, span) = paired_parts!(p, part);
    let (rows, cols) = (p.rows as usize, p.cols as usize);
    if row_tile * BR64 >= rows || token_tile * BT >= m { return; }
    let live = (m - token_tile * BT).min(BT);
    let tid = _thread_idx_x() as usize;
    let tid32 = tid as u32;
    let smem = crate::ptx::shared::<R64_BYTES>();
    let word = |w: usize| smem + (w * 4) as u32;
    let sl = (2 * STAGE64 + AT64 + ATS64) as u32;
    paired_slots!(p, part, span, e, token_tile, BT, sl as usize, word, tid);
    let steps = cols / 64;
    let tiled = p.layout == 1;
    let skip = p.skip == 1 && live == 1;
    let one_codes = if skip { (p.codes as *const u32).add(ld!(word(sl as usize)) as usize * (cols / 8)) } else { p.codes as *const u32 };
    let codes = if tid < live * 2 { (p.codes as *const u8).add(ld!(word(sl as usize + 4 * (tid / 2))) as usize * (cols / 2) + (tid % 2) * 16) } else { p.codes as *const u8 };
    let ascales = if tid < live { (p.scales as *const u8).add(ld!(word(sl as usize + 4 * tid)) as usize * (cols / 16)) } else { p.scales as *const u8 };
    let weights = e.weight as *const u8;
    let wscales = e.scales as *const u8;
    let (tile128, half) = (row_tile >> 1, row_tile & 1);
    let (lane, warp) = (tid32 & 31, tid32 >> 5);
    let (g, t) = (lane >> 2, lane & 3);
    let (wr, wc) = (warp & 1, warp >> 1);
    let mut acc = [[crate::canonical::ZERO; 4]; 2];
    {
        let mut i = live * 2 + tid;
        while i < BT * 2 {
            let (n, q) = (i / 2, i % 2);
            let at_w = AR64 + AS64 + n * 8 + (q ^ ((n >> 2) & 1)) * 4;
            zero16!(word(at_w)); zero16!(word(STAGE64 + at_w));
            i += THREADS;
        }
        if tid >= live && tid < BT { st!(word(AR64 + AS64 + B64 + tid), 0u32); st!(word(STAGE64 + AR64 + AS64 + B64 + tid), 0u32); }
    }
    let issue = |c: usize, base_s: u32| {
        let s = base_s + (((c & 1) * STAGE64 * 4) as u32);
        if tiled {
            if tid32 < 128 {
                let (col, q) = (tid32 >> 1, tid32 & 1);
                let column = c * 64 + col as usize;
                let dst = s + ((col * 8 + (col >> 3) * 4 + q * 4) << 2);
                let zero = skip && ((*one_codes.add(column / 8) >> (4 * (column % 8))) & 7) == 0;
                if zero { zero16!(dst); } else {
                    crate::ptx::copy16(dst, weights.add(((tile128 * cols + column) * 64) + half * 32 + (q as usize) * 16).cast());
                }
            } else if tid32 < 144 {
                let idx = tid32 - 128;
                let (blk, sq) = (idx >> 2, idx & 3);
                crate::ptx::copy16(s + ((AR64 as u32 + blk * 16 + sq * 4) << 2), wscales.add((tile128 * (cols / 16) + c * 4 + blk as usize) * 128 + half * 64 + (sq as usize) * 16).cast());
            }
        } else if tid32 < 128 {
            let (row, q) = (tid32 >> 1, tid32 & 1);
            let r = row_tile * BR64 + row as usize;
            crate::ptx::copy16(s + ((row * 8 + q * 4) << 2), weights.add(r * (cols / 2) + c * 32 + (q as usize) * 16).cast());
        }
        if tid < live * 2 {
            let (n, q) = (tid32 >> 1, tid32 & 1);
            let sw = q ^ ((n >> 2) & 1);
            crate::ptx::copy16(s + (((AR64 + AS64) as u32 + n * 8 + sw * 4) << 2), codes.add(c * 32).cast());
        }
        if tid < live {
            crate::ptx::copy4(s + (((AR64 + AS64 + B64) as u32 + tid32) << 2), ascales.add(c * 4).cast());
        }
    };
    issue(0, smem);
    crate::ptx::commit();
    let mut c = 0usize;
    while c < steps {
        let mut smem_l = smem;
        // Prevent LLVM LICM from hoisting ~60 shared addresses across the K loop.
        core::arch::asm!("/* {0} */", inout(reg32) smem_l, options(nomem, nostack));
        if c + 1 < steps { issue(c + 1, smem_l); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
        crate::ptx::sync();
        let s = smem_l + (((c & 1) * STAGE64 * 4) as u32);
        let at_b = smem_l + ((2 * STAGE64 * 4) as u32);
        let ats_b = at_b + ((AT64 * 4) as u32);
        if tiled {
            if tid32 < 64 {
                let (cb, rb) = (tid32 & 7, tid32 >> 3);
                let mut x = [0u32; 8];
                rep8!(J, { let col = (cb << 3) + J as u32; x[J] = ld!(s + ((col * 8 + (col >> 3) * 4 + rb) << 2)); });
                let [mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7] = x;
                swap(&mut x0, &mut x4, 16, 0x0000_ffff); swap(&mut x1, &mut x5, 16, 0x0000_ffff);
                swap(&mut x2, &mut x6, 16, 0x0000_ffff); swap(&mut x3, &mut x7, 16, 0x0000_ffff);
                swap(&mut x0, &mut x2, 8, 0x00ff_00ff); swap(&mut x1, &mut x3, 8, 0x00ff_00ff);
                swap(&mut x4, &mut x6, 8, 0x00ff_00ff); swap(&mut x5, &mut x7, 8, 0x00ff_00ff);
                swap(&mut x0, &mut x1, 4, 0x0f0f_0f0f); swap(&mut x2, &mut x3, 4, 0x0f0f_0f0f);
                swap(&mut x4, &mut x5, 4, 0x0f0f_0f0f); swap(&mut x6, &mut x7, 4, 0x0f0f_0f0f);
                let y = [x0, x1, x2, x3, x4, x5, x6, x7];
                rep8!(I, { let r = (rb << 3) + I as u32; st!(at_b + ((r * 8 + (r >> 3) * 8 + (cb ^ (r & 4))) << 2), y[I]); });
            } else if tid32 < 128 {
                let r = tid32 - 64;
                let base_sc = s + ((AR64 * 4) as u32) + r;
                let b0 = crate::ptx::load_u8(base_sc);
                let b1 = crate::ptx::load_u8(base_sc + 64);
                let b2 = crate::ptx::load_u8(base_sc + 128);
                let b3 = crate::ptx::load_u8(base_sc + 192);
                st!(ats_b + (r << 2), b0 | b1 << 8 | b2 << 16 | b3 << 24);
            }
        } else {
            if tid32 < 128 {
                let (row, q) = (tid32 >> 1, tid32 & 1);
                let v = crate::ptx::load4(s + ((row * 8 + q * 4) << 2));
                let base = at_b + ((row * 8 + (row >> 3) * 8) << 2);
                let flip = row & 4;
                rep4!(K, { st!(base + (((q * 4 + K as u32) ^ flip) << 2), v[K]); });
            }
            if tid32 < BR64 as u32 {
                let r = row_tile * BR64 + tid;
                st!(ats_b + (tid32 << 2), *(wscales.add(r * (cols / 16) + c * 4) as *const u32));
            }
        }
        crate::ptx::sync();
        let mut smem_m = smem_l;
        core::arch::asm!("/* {0} */", inout(reg32) smem_m, options(nomem, nostack));
        let s_m = smem_m + (((c & 1) * STAGE64 * 4) as u32);
        let at_m = smem_m + ((2 * STAGE64 * 4) as u32);
        let ats_m = at_m + ((AT64 * 4) as u32);
        let load_a = |mi: u32| -> ([u32; 4], u32) {
            let r0 = 32 * wr + 16 * mi;
            let row = |r: u32, k: u32| ld!(at_m + ((r * 8 + (r >> 3) * 8 + (k ^ (r & 4))) << 2));
            ([row(r0 + g, t), row(r0 + g + 8, t), row(r0 + g, t + 4), row(r0 + g + 8, t + 4)], ld!(ats_m + ((r0 + g + (t & 1) * 8) << 2)))
        };
        let (a0, sa0) = load_a(0);
        let (a1, sa1) = load_a(1);
        rep4!(NJ, {
            let n = 32 * wc + 8 * NJ as u32 + g;
            let tok = s_m + (((AR64 + AS64) as u32 + n * 8) << 2);
            let sw = ((n >> 2) & 1) * 4;
            let b0 = ld!(tok + ((sw + t) << 2));
            let b1 = ld!(tok + ((4 - sw + t) << 2));
            let sb = ld!(s_m + (((AR64 + AS64 + B64) as u32 + n) << 2));
            acc[0][NJ] = mma!(a0, [b0, b1], sa0, sb, acc[0][NJ]);
            acc[1][NJ] = mma!(a1, [b0, b1], sa1, sb, acc[1][NJ]);
        });
        crate::ptx::sync();
        c += 1;
    }
    let mut smem_ep = smem;
    core::arch::asm!("/* {0} */", inout(reg32) smem_ep, options(nomem, nostack));
    let word_ep = |w: usize| smem_ep + (w * 4) as u32;
    let e_end = *(p.table as *const Fp4CohortPart).add(part);
    let scale: f32;
    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) scale, a = in(reg32) e_end.global, b = in(reg32) e_end.activation);
    let ep = smem_ep + (tid32 << 6);
    let put = |ptr: *mut f32, x: f32| {
        if p.raw == 1 { *ptr = x; } else {
            let y: f32;
            core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
            *ptr = crate::expert_ops::round(y);
        }
    };
    rep2!(MI, {
        rep4!(NJ, {
            let v = acc[MI][NJ];
            core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                p = in(reg32) ep + (NJ * 16) as u32,
                a = in(reg32) v.a, b = in(reg32) v.b, c = in(reg32) v.c, d = in(reg32) v.d, options(nostack));
        });
        let r = row_tile * BR64 + (32 * wr + 16 * MI as u32 + g) as usize;
        let mut nj = 0u32;
        while nj < 4 {
            core::arch::asm!("/* {0} */", inout(reg32) nj, options(nomem, nostack));
            let q = crate::ptx::load4(ep + (nj << 4));
            let n = (32 * wc + 8 * nj + 2 * t) as usize;
            if n < live {
                let base0 = ((ld!(word_ep(sl as usize + 4 * n + 1)) as u64) | ((ld!(word_ep(sl as usize + 4 * n + 2)) as u64) << 32)) as *mut f32;
                let str0 = ld!(word_ep(sl as usize + 4 * n + 3)) as usize;
                put(base0.add(r * str0), f32::from_bits(q[0]));
                put(base0.add((r + 8) * str0), f32::from_bits(q[2]));
            }
            if n + 1 < live {
                let base1 = ((ld!(word_ep(sl as usize + 4 * (n + 1) + 1)) as u64) | ((ld!(word_ep(sl as usize + 4 * (n + 1) + 2)) as u64) << 32)) as *mut f32;
                let str1 = ld!(word_ep(sl as usize + 4 * (n + 1) + 3)) as usize;
                put(base1.add(r * str1), f32::from_bits(q[1]));
                put(base1.add((r + 8) * str1), f32::from_bits(q[3]));
            }
            nj += 1;
        }
    });
}

/// High-occupancy 64x128 quarter-chain form (`chains == 4`, routed down): `50,688 B` dynamic shared memory
/// (`14,080 B` double-buffered staging + `2,048 B` zero-pad conflict-free `AT64` + `256 B` `ATS64` + `1,536 B`
/// persistent 3-word output slot table + `32,768 B` `park01` = `101,376 / 2` -> 2 CTAs/SM!), zero `hold` registers
/// (`<= 128` regs -> 2 CTAs/SM, 0 spills!): builds slot table ONCE at setup, runs one continuous K pipeline across all
/// 4 quarters, parks Q0/Q2 in `park01`, flushes `fl(Q0 + Q1)` coalesced via `park01`, and finishes `fl(first + fl(Q2 + Q3))`.
/// SAFETY: As `fp4_cohort_r64`, with `chains == 4`, `cols % 256 == 0`, and dynamic shared memory `>= 50,688 B`
/// (checked here via `%dynamic_smem_size`, zero static smem). Setup temporarily stages `paired_slots!` (2,048 B) at
/// `smem[0..2048]`, extracts input pointers and copies `(out_lo, out_hi, stride)` into persistent `SL64_OUT` (1,536 B),
/// then zeroes dead token slots and runs all 4 quarters without rebuilding slots.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_cohort_q64r128(p: Fp4Cohort) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    const AT64_SWIZ: usize = BR64 * 8; // 512 words (2,048 B): zero-pad conflict-free XOR-swizzled layout
    const SL64_OUT: usize = BT * 3; // 384 words (1,536 B): persistent (out_lo, out_hi, stride) per token slot
    const Q64_SL_OFF: usize = (2 * STAGE64 + AT64_SWIZ + ATS64) * 4;
    const Q64_BASE_BYTES: usize = Q64_SL_OFF + SL64_OUT * 4;
    const Q64_PARK_BYTES: usize = THREADS * 8 * 16;
    const Q64_TOTAL_BYTES: u32 = (Q64_BASE_BYTES + Q64_PARK_BYTES) as u32;
    const _: () = assert!(Q64_SL_OFF == 16_384 && Q64_BASE_BYTES == 17_920 && Q64_PARK_BYTES == 32_768 && Q64_TOTAL_BYTES == 50_688);
    let dynamic: u32;
    core::arch::asm!("mov.u32 {d}, %dynamic_smem_size;", d = out(reg32) dynamic, options(nomem, nostack));
    if _block_dim_x() as usize != THREADS || dynamic < Q64_TOTAL_BYTES || p.count == 0 || p.cols == 0 || p.cols % 256 != 0
        || p.cols > 8192 || p.rows % BR as u32 != 0 || p.layout != 1 || p.chains != 4 || p.merge > 8
    {
        core::arch::asm!("trap;");
    }
    let (part, token_tile, row_tile) = routed_tile!(p);
    if part >= p.count as usize { return; }
    let (e, m, span) = paired_parts!(p, part);
    let (rows, cols) = (p.rows as usize, p.cols as usize);
    if row_tile * BR64 >= rows || token_tile * BT >= m { return; }
    let live = (m - token_tile * BT).min(BT);
    let tid = _thread_idx_x() as usize;
    let tid32 = tid as u32;
    let smem = 0u32;
    let word = |w: usize| smem + (w * 4) as u32;
    // Build full 4-word slot table once at `smem[0..2048]`, extract input pointers, and copy 3-word output metadata into `SL64_OUT`.
    paired_slots!(p, part, span, e, token_tile, BT, 0, word, tid);
    let codes = if tid < live * 2 { (p.codes as *const u8).add(ld!(word(4 * (tid / 2))) as usize * (cols / 2) + (tid % 2) * 16) } else { p.codes as *const u8 };
    let ascales = if tid < live { (p.scales as *const u8).add(ld!(word(4 * tid)) as usize * (cols / 16)) } else { p.scales as *const u8 };
    if tid < live {
        let dst = (Q64_SL_OFF as u32) + (tid32 * 3) * 4;
        st!(dst, ld!(word(4 * tid + 1)));
        st!(dst + 4, ld!(word(4 * tid + 2)));
        st!(dst + 8, ld!(word(4 * tid + 3)));
    }
    crate::ptx::sync();
    let weights = e.weight as *const u8;
    let wscales = e.scales as *const u8;
    let per_chain = (cols / 64) / 4;
    {
        let mut i = live * 2 + tid;
        while i < BT * 2 {
            let (n, q) = (i / 2, i % 2);
            let at_w = AR64 + AS64 + n * 8 + (q ^ ((n >> 2) & 1)) * 4;
            zero16!(word(at_w)); zero16!(word(STAGE64 + at_w));
            i += THREADS;
        }
        if tid >= live && tid < BT { st!(word(AR64 + AS64 + B64 + tid), 0u32); st!(word(STAGE64 + AR64 + AS64 + B64 + tid), 0u32); }
    }
    let issue = |c: usize, base_s: u32| {
        let s = base_s + (((c & 1) * STAGE64 * 4) as u32);
        let (tile128, half) = (row_tile >> 1, row_tile & 1);
        if tid32 < 128 {
            let (col, q) = (tid32 >> 1, tid32 & 1);
            let column = c * 64 + col as usize;
            let dst = s + ((col * 8 + (col >> 3) * 4 + q * 4) << 2);
            crate::ptx::copy16(dst, weights.add(((tile128 * cols + column) * 64) + half * 32 + (q as usize) * 16).cast());
        } else if tid32 < 144 {
            let idx = tid32 - 128;
            let (blk, sq) = (idx >> 2, idx & 3);
            crate::ptx::copy16(s + ((AR64 as u32 + blk * 16 + sq * 4) << 2), wscales.add((tile128 * (cols / 16) + c * 4 + blk as usize) * 128 + half * 64 + (sq as usize) * 16).cast());
        }
        if tid < live * 2 {
            let (n, q) = (tid32 >> 1, tid32 & 1);
            let sw = q ^ ((n >> 2) & 1);
            crate::ptx::copy16(s + (((AR64 + AS64) as u32 + n * 8 + sw * 4) << 2), codes.add(c * 32).cast());
        }
        if tid < live {
            crate::ptx::copy4(s + (((AR64 + AS64 + B64) as u32 + tid32) << 2), ascales.add(c * 4).cast());
        }
    };
    issue(0, smem);
    crate::ptx::commit();
    let mut c = 0usize;
    let mut half_k = 0usize;
    while half_k < 2 {
        let mut acc = [[crate::canonical::ZERO; 4]; 2];
        let half_end = (half_k + 1) * (2 * per_chain);
        let q0_end = half_k * (2 * per_chain) + per_chain;
        while c < half_end {
            let mut smem_l = smem;
            core::arch::asm!("/* {0} */", inout(reg32) smem_l, options(nomem, nostack));
            if c + 1 < cols / 64 { issue(c + 1, smem_l); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
            crate::ptx::sync();
            let s = smem_l + (((c & 1) * STAGE64 * 4) as u32);
            let at_b = smem_l + ((2 * STAGE64 * 4) as u32);
            let ats_b = at_b + ((AT64_SWIZ * 4) as u32);
            if tid32 < 64 {
                let (cb, rb) = (tid32 & 7, tid32 >> 3);
                let mut x = [0u32; 8];
                rep8!(J, { let col = (cb << 3) + J as u32; x[J] = ld!(s + ((col * 8 + (col >> 3) * 4 + rb) << 2)); });
                let [mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7] = x;
                swap(&mut x0, &mut x4, 16, 0x0000_ffff); swap(&mut x1, &mut x5, 16, 0x0000_ffff);
                swap(&mut x2, &mut x6, 16, 0x0000_ffff); swap(&mut x3, &mut x7, 16, 0x0000_ffff);
                swap(&mut x0, &mut x2, 8, 0x00ff_00ff); swap(&mut x1, &mut x3, 8, 0x00ff_00ff);
                swap(&mut x4, &mut x6, 8, 0x00ff_00ff); swap(&mut x5, &mut x7, 8, 0x00ff_00ff);
                swap(&mut x0, &mut x1, 4, 0x0f0f_0f0f); swap(&mut x2, &mut x3, 4, 0x0f0f_0f0f);
                swap(&mut x4, &mut x5, 4, 0x0f0f_0f0f); swap(&mut x6, &mut x7, 4, 0x0f0f_0f0f);
                let y = [x0, x1, x2, x3, x4, x5, x6, x7];
                rep8!(I, {
                    let r = (rb << 3) + I as u32;
                    let idx = ((r & 7) << 6) | (((r >> 3) ^ (r & 7)) << 3) | (cb ^ (r & 4));
                    st!(at_b + (idx << 2), y[I]);
                });
            } else if tid32 < 128 {
                let r = tid32 - 64;
                let base_sc = s + ((AR64 * 4) as u32) + r;
                let b0 = crate::ptx::load_u8(base_sc);
                let b1 = crate::ptx::load_u8(base_sc + 64);
                let b2 = crate::ptx::load_u8(base_sc + 128);
                let b3 = crate::ptx::load_u8(base_sc + 192);
                st!(ats_b + (r << 2), b0 | b1 << 8 | b2 << 16 | b3 << 24);
            }
            crate::ptx::sync();
            let mut smem_m = smem_l;
            core::arch::asm!("/* {0} */", inout(reg32) smem_m, options(nomem, nostack));
            let mut s_m = smem_m + (((c & 1) * STAGE64 * 4) as u32);
            let at_m = smem_m + ((2 * STAGE64 * 4) as u32);
            let ats_m = at_m + ((AT64_SWIZ * 4) as u32);
            let (lane_m, warp_m) = (tid32 & 31, tid32 >> 5);
            let (g_m, t_m) = (lane_m >> 2, lane_m & 3);
            let (wr_m, wc_m) = (warp_m & 1, warp_m >> 1);
            let load_a = |mi: u32| -> ([u32; 4], u32) {
                let r0 = 32 * wr_m + 16 * mi;
                let row = |r: u32, k: u32| ld!(at_m + ((((r & 7) << 6) | (((r >> 3) ^ (r & 7)) << 3) | (k ^ (r & 4))) << 2));
                ([row(r0 + g_m, t_m), row(r0 + g_m + 8, t_m), row(r0 + g_m, t_m + 4), row(r0 + g_m + 8, t_m + 4)], ld!(ats_m + ((r0 + g_m + (t_m & 1) * 8) << 2)))
            };
            let (a0, sa0) = load_a(0);
            let (a1, sa1) = load_a(1);
            rep4!(NJ, {
                if NJ > 0 { core::arch::asm!("/* {0} */", inout(reg32) s_m, options(nomem, nostack)); }
                let n = 32 * wc_m + 8 * NJ as u32 + g_m;
                let tok = s_m + (((AR64 + AS64) as u32 + n * 8) << 2);
                let sw = ((n >> 2) & 1) * 4;
                let b0 = ld!(tok + ((sw + t_m) << 2));
                let b1 = ld!(tok + ((4 - sw + t_m) << 2));
                let sb = ld!(s_m + (((AR64 + AS64 + B64) as u32 + n) << 2));
                acc[0][NJ] = mma!(a0, [b0, b1], sa0, sb, acc[0][NJ]);
                acc[1][NJ] = mma!(a1, [b0, b1], sa1, sb, acc[1][NJ]);
            });
            crate::ptx::sync();
            c += 1;
            if c == q0_end {
                let mut park_t = smem_l + (Q64_BASE_BYTES as u32) + (tid32 << 7);
                core::arch::asm!("/* {0} */", inout(reg32) park_t, options(nomem, nostack));
                rep2!(MI, { rep4!(NJ, {
                    let v = acc[MI][NJ];
                    core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                        p = in(reg32) park_t + ((MI * 4 + NJ) * 16) as u32,
                        a = in(reg32) v.a, b = in(reg32) v.b, c = in(reg32) v.c, d = in(reg32) v.d, options(nostack));
                    acc[MI][NJ] = crate::canonical::ZERO;
                }); });
            }
        }
        let mut smem_f = smem;
        core::arch::asm!("/* {0} */", inout(reg32) smem_f, options(nomem, nostack));
        let sl_out = smem_f + Q64_SL_OFF as u32;
        let park01 = smem_f + Q64_BASE_BYTES as u32;
        let mut park_t = park01 + (tid32 << 7);
        let (lane, warp) = (tid32 & 31, tid32 >> 5);
        let (g, t) = (lane >> 2, lane & 3);
        let (wr, wc) = (warp & 1, warp >> 1);
        rep2!(MI, { rep4!(NJ, {
            core::arch::asm!("/* {0} */", inout(reg32) park_t, options(nomem, nostack));
            let q = crate::ptx::load4(park_t + ((MI * 4 + NJ) * 16) as u32);
            let a0 = Acc { a: f32::from_bits(q[0]), b: f32::from_bits(q[1]), c: f32::from_bits(q[2]), d: f32::from_bits(q[3]) };
            acc[MI][NJ] = add_acc!(a0, acc[MI][NJ]);
        }); });
        if p.rowmap != 0 {
            crate::ptx::sync();
            rep2!(MI, {
                let rl = 32 * wr + 16 * MI as u32 + g;
                rep4!(NJ, {
                    let n = 32 * wc + 8 * NJ as u32 + 2 * t;
                    let v = acc[MI][NJ];
                    st!(park01 + ((n * BR64 as u32 + rl) << 2), v.a.to_bits());
                    st!(park01 + (((n + 1) * BR64 as u32 + rl) << 2), v.b.to_bits());
                    st!(park01 + ((n * BR64 as u32 + rl + 8) << 2), v.c.to_bits());
                    st!(park01 + (((n + 1) * BR64 as u32 + rl + 8) << 2), v.d.to_bits());
                });
            });
            crate::ptx::sync();
            let scale = if half_k == 1 && p.raw != 1 {
                let e_cur = *(p.table as *const Fp4CohortPart).add(part);
                let sc: f32;
                core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) sc, a = in(reg32) e_cur.global, b = in(reg32) e_cur.activation);
                sc
            } else { 1.0f32 };
            let fin = |x: f32| -> f32 {
                if p.raw == 1 { x } else {
                    let y: f32;
                    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                    crate::expert_ops::round(y)
                }
            };
            let (tg, v4) = (tid32 >> 4, tid32 & 15);
            let mut pass = 0u32;
            while pass < 8 {
                let n = (pass * 16 + tg) as usize;
                if n < live {
                    let sp = sl_out + ((n as u32) * 12);
                    let base = ((ld!(sp) as u64) | ((ld!(sp + 4) as u64) << 32)) as *mut f32;
                    let dst = base.add(row_tile * BR64 + (v4 as usize) * 4);
                    let q = crate::ptx::load4(park01 + (((n as u32) * BR64 as u32 + v4 * 4) << 2));
                    if half_k == 0 {
                        if (dst as u64) & 15 == 0 {
                            core::arch::asm!("st.global.v4.b32 [{p}], {{{a},{b},{c},{d}}};",
                                p = in(reg64) dst, a = in(reg32) q[0], b = in(reg32) q[1], c = in(reg32) q[2], d = in(reg32) q[3], options(nostack));
                        } else {
                            *dst = f32::from_bits(q[0]); *dst.add(1) = f32::from_bits(q[1]);
                            *dst.add(2) = f32::from_bits(q[2]); *dst.add(3) = f32::from_bits(q[3]);
                        }
                    } else if (dst as u64) & 15 == 0 {
                        let (a0, a1, a2, a3): (f32, f32, f32, f32);
                        core::arch::asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                            a = out(reg32) a0, b = out(reg32) a1, c = out(reg32) a2, d = out(reg32) a3,
                            p = in(reg64) dst, options(nostack));
                        let w0 = fin(add_rn!(a0, f32::from_bits(q[0])));
                        let w1 = fin(add_rn!(a1, f32::from_bits(q[1])));
                        let w2 = fin(add_rn!(a2, f32::from_bits(q[2])));
                        let w3 = fin(add_rn!(a3, f32::from_bits(q[3])));
                        core::arch::asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                            p = in(reg64) dst, a = in(reg32) w0, b = in(reg32) w1, c = in(reg32) w2, d = in(reg32) w3, options(nostack));
                    } else {
                        *dst = fin(add_rn!(*dst, f32::from_bits(q[0])));
                        *dst.add(1) = fin(add_rn!(*dst.add(1), f32::from_bits(q[1])));
                        *dst.add(2) = fin(add_rn!(*dst.add(2), f32::from_bits(q[2])));
                        *dst.add(3) = fin(add_rn!(*dst.add(3), f32::from_bits(q[3])));
                    }
                }
                pass += 1;
            }
        } else {
            rep2!(MI, { rep4!(NJ, {
                let v = acc[MI][NJ];
                core::arch::asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                    p = in(reg32) park_t + ((MI * 4 + NJ) * 16) as u32,
                    a = in(reg32) v.a, b = in(reg32) v.b, c = in(reg32) v.c, d = in(reg32) v.d, options(nostack));
            }); });
            let scale = if half_k == 1 && p.raw != 1 {
                let e_cur = *(p.table as *const Fp4CohortPart).add(part);
                let sc: f32;
                core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) sc, a = in(reg32) e_cur.global, b = in(reg32) e_cur.activation);
                sc
            } else { 1.0f32 };
            let fin = |x: f32| -> f32 {
                if p.raw == 1 { x } else {
                    let y: f32;
                    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) y, a = in(reg32) x, b = in(reg32) scale);
                    crate::expert_ops::round(y)
                }
            };
            let mut nj = 0u32;
            while nj < 4 {
                core::arch::asm!("/* {0} */", inout(reg32) nj, options(nomem, nostack));
                let n = (32 * wc + 8 * nj + 2 * t) as usize;
                let mut mi = 0u32;
                while mi < 2 {
                    let q = crate::ptx::load4(park_t + ((mi * 4 + nj) << 4));
                    let r = row_tile * BR64 + (32 * wr + 16 * mi + g) as usize;
                    let mut e_slot = 0usize;
                    while e_slot < 2 {
                        let ns = n + e_slot;
                        if ns < live {
                            let sp = sl_out + ((ns as u32) * 12);
                            let base = ((ld!(sp) as u64) | ((ld!(sp + 4) as u64) << 32)) as *mut f32;
                            let stride = ld!(sp + 8) as usize;
                            let v_lo = f32::from_bits(if e_slot == 0 { q[0] } else { q[1] });
                            let v_hi = f32::from_bits(if e_slot == 0 { q[2] } else { q[3] });
                            if half_k == 0 {
                                *base.add(r * stride) = v_lo;
                                *base.add((r + 8) * stride) = v_hi;
                            } else {
                                *base.add(r * stride) = fin(add_rn!(*base.add(r * stride), v_lo));
                                *base.add((r + 8) * stride) = fin(add_rn!(*base.add((r + 8) * stride), v_hi));
                            }
                        }
                        e_slot += 1;
                    }
                    mi += 1;
                }
                nj += 1;
            }
        }
        if half_k == 0 { crate::ptx::sync(); }
        half_k += 1;
    }
}

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
        || p.rows % BR as u32 != 0 || p.layout != 1 || p.chains != 1 || p.merge > 1 {
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
