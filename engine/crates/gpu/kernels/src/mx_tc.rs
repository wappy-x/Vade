//! MXFP8 rows on the tensor cores: the word function of `engine/docs/tc-dense-contract.md`, section MXFP8. The CPU
//! twin is `gpu/src/tc_twin_mx.rs`. Every entry here gives the same word for the same weight row and token row.
//!
//! The word (one weight row, one token row), `groups = cols / 32`:
//! 1. Weight operand: the stored E4M3 codes as they are.
//! 2. Activation operand of a group of 32 columns: the largest magnitude of the group's 32 FP32 values gives the scale
//!    code `xs` (the activation quantizer's rule); each value times `2^(127 - xs)` goes through
//!    `cvt.rn.satfinite.e4m3x2.f32`. A group the MXFP8 activation quantizer wrote is on that grid, so this is exact.
//! 3. Range `p` = groups `[groups*p/4, groups*(p+1)/4)`, cut into sub-chains of `SUB` = 8 groups counted from the
//!    range's first group. A sub-chain, in ascending group order from +0.0: `acc = step(acc, weight codes, activation
//!    codes, ws, xs)`, the block-scaled step `mma.sync m16n8k32 kind::mxf8f6f4 block_scale scale_vec::1X` with the old
//!    sum as its accumulator (`ws`, `xs`: the UE8M0 scale codes of the group; `xs` = 255 when a value is not finite;
//!    no multiply and no add follows a step). `part p` = the sub-chain results added in ascending order from +0.0.
//! 4. `word = fl(fl(part0 + part1) + fl(part2 + part3))` (`add.rn.f32`, never fused), stored as FP32.
//!
//! The instruction's k index inside a group is free (the 32 products of a step enter one sum; the twin and the device
//! sweep hold that): a lane holds 8 consecutive columns of the group, `8 * (lane % 4) ..+8`, of weight rows
//! `lane / 4` and `lane / 4 + 8` (operand A) and of token `lane / 4` (operand B). So a lane reads its weight bytes with
//! one 8-byte load a row and a group, and the four lanes of a quad read a row's 32 bytes of the group in order.
//!
//! Entries: `mx_tc_rows` (1 to 8 tokens a block, a warp a range: the decode step, small row groups, the drafting
//! check rows), `mx_tc_tile` (64 tokens a block through shared memory: prompt groups), `mx_tc_pack` and
//! `mx_tc_tile_packed` (prompt groups: the activation operands of the token rows made once, for every matrix that reads
//! them), `mx_tc_quantize_pack` (the quantize pass and `mx_tc_pack` in one launch: the same bytes),
//! `mx_tc_probe` and `mx_tc_probe_cvt` (bench only: one raw step a warp; the activation conversion).
//!
//! The step needs the image's target `sm_121a` (lead's ruling of 2026-11-16 12:25, `engine/docs/tc-dense-contract.md`).
use crate::canonical::{Acc, ZERO};
use crate::next_params::MxProjection;

/// Warps of a `mx_tc_rows` block (one a range of the word) and the weight rows the block owns.
pub(crate) const ROWS_WARPS: usize = 4;
pub(crate) const ROWS_BLOCK: usize = 16;
/// The prefetch of the row body: groups ahead of the weight loads (2 = one 64-byte line), and groups between two
/// prefetches of a row (1 = every step; MEASURED run `onerow-full-1`: a pass of 145 calls at 1 / 2 / 4 / 8 rows
/// 11.64 / 11.71 / 11.71 / 12.01 ms, every second step 11.82 / 11.87 / 12.00 / 12.26).
pub(crate) const AHEAD: usize = 2;
pub(crate) const EVERY: usize = 1;
const _: () = assert!(ROWS_BLOCK as u32 == crate::next_params::MX_TC_ROWS && crate::next_params::MX_TC_TOKENS == 8);
/// Steps of a sub-chain: inside a range the tensor-core step carries its sum for `SUB` groups (256 columns) from +0.0,
/// then the sub-chain's result is added to the range's sum with one FP32 add (round to nearest). A step cuts toward
/// zero, so one chain over a whole range (32 to 96 steps) fails the word accuracy count (MEASURED 2026-11-16 on
/// 3,182,592 real words a chain: 213 / 226 / 244 words not correctly rounded against today's 144 / 146 / 159); with
/// sub-chains of 8 it is 139 / 167 / 183. The CPU twin has the same constant.
pub(crate) const SUB: usize = 8;
/// The join slots of a `mx_tc_rows` block: 16 bytes a thread.
const JOIN_BYTES: usize = ROWS_WARPS * 32 * 16;

// The helpers below are macros, not functions: they expand inside the one unsafe body of the entry that uses them,
// under that entry's safety contract (register operands only, except the loads, which name their pointer).
/// The block-scaled step: `D = (A * 2^(sa - 127)) * (B * 2^(sb - 127)) + C` on E4M3 codes, one UE8M0 scale code a weight
/// row and one a token for the step's 32 columns. `$sa`: this lane's weight scale code; the instruction reads it from
/// lanes 0 and 1 of a quad (selector {0,0}): rows `g` and `g + 8`. `$sb`: the scale code of token `g`, read from lane 0.
/// The code sits in the low byte of its register.
macro_rules! mma {($a:expr,$b:expr,$c:expr,$sa:expr,$sb:expr)=>{{
    let (a, b, c): ([u32; 4], [u32; 2], Acc) = ($a, $b, $c);
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    core::arch::asm!(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 \
         {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,0}};",
        d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
        a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
        b0 = in(reg32) b[0], b1 = in(reg32) b[1],
        c0 = in(reg32) c.a, c1 = in(reg32) c.b, c2 = in(reg32) c.c, c3 = in(reg32) c.d,
        sa = in(reg32) $sa, sb = in(reg32) $sb, options(nomem, nostack),
    );
    Acc { a: d0, b: d1, c: d2, d: d3 }
}};}
/// One word of shared memory at the byte offset `$p` (inside an unsafe block; the caller keeps the offset inside
/// the block's static array).
macro_rules! load_word {($p:expr)=>{{
    let word: u32;
    core::arch::asm!("ld.shared.u32 {a}, [{p}];", a = out(reg32) word, p = in(reg32) $p, options(nostack, readonly));
    word
}};}
macro_rules! mul_rn {($a:expr,$b:expr)=>{{
    let o: f32;
    core::arch::asm!("mul.rn.f32 {o}, {a}, {b};", o = out(reg32) o, a = in(reg32) $a, b = in(reg32) $b, options(nomem, nostack));
    o
}};}
macro_rules! add_rn {($a:expr,$b:expr)=>{{
    let o: f32;
    core::arch::asm!("add.rn.f32 {o}, {a}, {b};", o = out(reg32) o, a = in(reg32) $a, b = in(reg32) $b, options(nomem, nostack));
    o
}};}
macro_rules! add_acc {($x:expr,$y:expr)=>{{
    let (x, y): (Acc, Acc) = ($x, $y);
    Acc { a: add_rn!(x.a, y.a), b: add_rn!(x.b, y.b), c: add_rn!(x.c, y.c), d: add_rn!(x.d, y.d) }
}};}
/// Two FP32 values to two E4M3 codes, nearest even, saturating: the low byte is the code of `$low`.
macro_rules! cvt_pair {($low:expr,$high:expr)=>{{
    let d: u16;
    core::arch::asm!("cvt.rn.satfinite.e4m3x2.f32 {d}, {h}, {l};", d = out(reg16) d, h = in(reg32) $high, l = in(reg32) $low, options(nomem, nostack));
    d as u32
}};}
macro_rules! shfl_xor {($v:expr,$mask:expr)=>{{
    let o: u32;
    core::arch::asm!("shfl.sync.bfly.b32 {o}, {v}, {m}, 31, -1;", o = out(reg32) o, v = in(reg32) $v, m = in(reg32) $mask, options(nomem, nostack));
    o
}};}
macro_rules! shfl_idx {($v:expr,$lane:expr)=>{{
    let o: u32;
    core::arch::asm!("shfl.sync.idx.b32 {o}, {v}, {l}, 31, -1;", o = out(reg32) o, v = in(reg32) $v, l = in(reg32) $lane, options(nomem, nostack));
    o
}};}
macro_rules! rep2 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}}};}
macro_rules! rep4 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}{const $v:usize=2;$body}{const $v:usize=3;$body}}};}
macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const $v:usize=4;$body}{const $v:usize=5;$body}{const $v:usize=6;$body}{const $v:usize=7;$body}}};}

/// Eight FP32 patterns of a global row.
macro_rules! load8 {($pointer:expr)=>{{
    let pointer: *const u8 = $pointer;
    let (x0, x1, x2, x3, x4, x5, x6, x7): (u32, u32, u32, u32, u32, u32, u32, u32);
    core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x0, b = out(reg32) x1, c = out(reg32) x2, d = out(reg32) x3, pointer = in(reg64) pointer, options(nostack, readonly));
    core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x4, b = out(reg32) x5, c = out(reg32) x6, d = out(reg32) x7, pointer = in(reg64) pointer.add(16), options(nostack, readonly));
    [x0, x1, x2, x3, x4, x5, x6, x7]
}};}

/// The scale code that stands for "a value of this group is not finite": the step then writes NaN.
const NOT_FINITE: u32 = 255;
/// The activation operand of one lane: its 8 FP32 patterns and the group's largest magnitude pattern give the lane's
/// two code words and the group's UE8M0 scale code `xs` (`NOT_FINITE` when a value of the group is not finite).
macro_rules! encode {($x:expr,$maximum:expr)=>{{
    let (x, maximum): ([u32; 8], u32) = ($x, $maximum);
    let code = ((maximum >> 23) + (maximum & 0x007f_ffff > 0x0060_0000) as u32).saturating_sub(8);
    // 2^(127 - xs): the exponent field 254 - xs is 7 to 254 for a finite maximum.
    let factor = f32::from_bits((254u32.wrapping_sub(code)) << 23);
    let mut y = [0.0f32; 8];
    rep8!(I, { y[I] = mul_rn!(f32::from_bits(x[I]), factor); });
    let b = [cvt_pair!(y[0], y[1]) | (cvt_pair!(y[2], y[3]) << 16), cvt_pair!(y[4], y[5]) | (cvt_pair!(y[6], y[7]) << 16)];
    (b, if maximum >= 0x7f80_0000 { NOT_FINITE } else { code })
}};}

/// Grid (ceil(rows / 16), ceil(tokens / 8)), 128 threads, no dynamic shared memory. Output word (token, row) is
/// `output[token * rows + row]`, FP32, not rounded: the layout of the other MXFP8 row kernels.
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_rows(p: MxProjection) { block::<AHEAD, EVERY>(p, None, false) }

/// One block of four warps: the 16 weight rows `first .. first + 16` for the tokens `token0 .. token0 + 8`. `origin` is
/// `(first, token0)`, or None for the block's own place in the grid (ceil(rows / 16), ceil(tokens / 8)). Every thread of
/// the block calls it with the same arguments; a block past `rows` or `tokens` returns as a whole. Warp `w` runs range `w` of the
/// word (its own columns: the four warps read a row's four quarters at the same time), parks its range accumulators in
/// shared memory, and warp 0 joins them `fl(fl(p0 + p1) + fl(p2 + p3))` and stores `output[token * rows + row]`.
///
/// The load pattern (MEASURED 2026-11-16, runs `onerow-v1-1` and `onerow-v2b-1`: the one-row pass of the model's 145 calls
/// went from 21.0 ms to 11.8 ms, today's scalar kernel 12.6 ms). A warp waits for each weight load, and a warp of this
/// body owns 16 rows, so without help the memory system sees a quarter of the requests of a one-warp-a-row body.
/// - A lane pulls its rows' next line into L2 (`prefetch.global.L2`: no value, no order) `PF` groups ahead of the loads,
///   one prefetch a row every `PS` groups. Nearer is better: 2 groups ahead 11.83 ms, 4: 12.27, 8: 13.12, 12: 13.84.
/// - The activation operand is made once for `8 / span` groups (`span` = the block's token count rounded up to a power of
///   two): quad `q` encodes token `q % span` of group `q / span` of the pass, and a step takes its operand by a shuffle.
///   One token: one input load pass in 8 steps.
/// - The weight scale codes come 4 groups a load when the scale rows are 4-byte aligned.
#[inline(always)]
pub(crate) fn block<const PF: usize, const PS: usize>(p: MxProjection, origin: Option<(usize, usize)>, round: bool) {
    // SAFETY: The host checks the descriptor as `MxLayer::push` does (complete E4M3 rows and UE8M0 groups in the retained
    // arena, FP32 input rows for all tokens, a disjoint output of tokens x rows words) and keeps every owner alive through
    // stream completion. Weights, scales and input are immutable (the non-coherent loads need that); each (token, row) output
    // word has exactly one writer (the lane of warp 0 whose fragment holds it). The body traps unless round and input_bf16
    // are zero, cols is a multiple of 32 and weights and input are 16-byte aligned. A lane reads weight bytes only of its rows
    // below `rows`, at row * cols + group * 32 + 8 * (lane % 4) .. +8 with group below the warp's range end, so inside the
    // row. A prefetch names an address inside a live row of this lane at a group below the range end and reads nothing.
    // Scale codes: one byte at row * (cols / 32) + group, or, only when `p.scales`, `groups` and the range start are
    // multiples of 4, the four bytes at row * groups + s with s a multiple of 4 below groups, so inside the row's codes.
    // Input words of the encode pass: a token below `tokens` and a group below the range end, at
    // token * cols + group * 32 + 8 * (lane % 4) .. +8. Shared memory: thread i writes its 16 bytes at i * 16 before the
    // one barrier, and a lane of warp 0 reads the four slots of its lane index after it. All four warps reach the barrier
    // (the caller's contract: one call a block with equal arguments; the loop bounds and the pass position depend only on
    // the descriptor and the warp), and all 32 lanes of a warp reach every shuffle and `mma`.
    unsafe {
        use core::arch::asm;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let (n, k, m) = (p.rows as usize, p.cols as usize, p.tokens as usize);
        if p.round != 0 || p.input_bf16 != 0 || k % 32 != 0 || k == 0 || (p.weights | p.input) & 15 != 0 || core::arch::nvptx::_block_dim_x() as usize != ROWS_WARPS * 32 || PS == 0 { asm!("trap;"); }
        // Only indices are read before this return: a block past the rows or the tokens leaves as a whole.
        let (first, token0) = match origin {
            Some(origin) => origin,
            None => (core::arch::nvptx::_block_idx_x() as usize * ROWS_BLOCK, core::arch::nvptx::_block_idx_y() as usize * 8),
        };
        if first >= n || token0 >= m { return; }
        let groups = k / 32;
        let (start, end) = (groups * warp / 4, groups * (warp + 1) / 4);
        let live = if m - token0 < 8 { m - token0 } else { 8 };
        let span = if live <= 1 { 1 } else if live <= 2 { 2 } else if live <= 4 { 4 } else { 8 };
        let per = 8 / span;
        let rows = [first + g, first + g + 8];
        let row_live = [rows[0] < n, rows[1] < n];
        // This lane's two output tokens (accumulator columns 2t and 2t + 1).
        let out_live = [2 * t < live, 2 * t + 1 < live];
        let base = p.weights as *const u8;
        let weights = base.wrapping_add(8 * t);
        // The encode pass: this quad's token and its group inside the pass.
        let (pass_token, pass_group) = (g % span, g / span);
        let pass_live = pass_token < live;
        let input = (p.input as *const u8).wrapping_add(((token0 + pass_token) * k + 8 * t) * 4);
        let scales = p.scales as *const u8;
        let scale_words = (p.scales as usize | groups | start) & 3 == 0;
        // The weight row whose scale code this lane gives the step: row g (lanes 0 and 2 of a quad) or g + 8 (lanes 1, 3).
        let (scale_row, scale_live) = (rows[t % 2], row_live[t % 2]);
        // The row a lane prefetches: lanes 0 and 1 of a quad, one row each.
        let fetch_row = if t == 0 { rows[0] } else if t == 1 { rows[1] } else { n };
        macro_rules! prefetch {($group:expr)=>{
            asm!("prefetch.global.L2 [{pointer}];", pointer = in(reg64) base.add(fetch_row * k + $group * 32), options(nostack, readonly));
        };}
        if PF != 0 && fetch_row < n {
            let mut j = 0;
            while j < PF { if start + j < end { prefetch!(start + j); } j += PS; }
        }
        let (mut codes, mut code_scale) = ([0u32; 2], 0u32);
        let mut scale_word = 0u32;
        // `cur`: the open sub-chain. `total`: the range's sum of the closed sub-chains.
        let (mut cur, mut total) = (ZERO, ZERO);
        let mut s = start;
        while s < end {
            let at = s - start;
            let j = at % per;
            if j == 0 {
                // The largest magnitude of a group: the four lanes of the quad hold its 32 values.
                let group = s + pass_group;
                let on = pass_live && group < end;
                let mut x = [0u32; 8];
                let mut maximum = 0u32;
                if on {
                    x = load8!(input.add(group * 32 * 4));
                    rep8!(I, { let magnitude = x[I] & 0x7fff_ffff; if magnitude > maximum { maximum = magnitude; } });
                }
                let other = shfl_xor!(maximum, 1u32);
                if other > maximum { maximum = other; }
                let other = shfl_xor!(maximum, 2u32);
                if other > maximum { maximum = other; }
                if on { let (b, scale) = encode!(x, maximum); codes = b; code_scale = scale; } else { codes = [0, 0]; code_scale = 0; }
            }
            if PF != 0 && at % PS == 0 && fetch_row < n && s + PF < end { prefetch!(s + PF); }
            if scale_words && at % 4 == 0 && scale_live { scale_word = *(scales.add(scale_row * groups + s) as *const u32); }
            // Operand A: this lane's 8 columns of the group, of its two rows.
            let mut words = [[0u32; 2]; 2];
            let mut i = 0;
            while i < 2 {
                if row_live[i] {
                    let (low, high): (u32, u32);
                    asm!("ld.global.nc.L1::no_allocate.v2.u32 {{{low},{high}}}, [{pointer}];", low = out(reg32) low, high = out(reg32) high,
                        pointer = in(reg64) weights.add(rows[i] * k + s * 32), options(nostack, readonly));
                    words[i] = [low, high];
                }
                i += 1;
            }
            // Operand B: token g of the block, group s, from the quad of the pass that encoded it.
            let source = (((j * span + g) * 4 + t) & 31) as u32;
            let (b0, b1) = (shfl_idx!(codes[0], source), shfl_idx!(codes[1], source));
            let b = if g < span { [b0, b1] } else { [0, 0] };
            // The scale codes of the step: this lane's weight row, and token g of the block from the quad of the pass that
            // encoded it. A row or a token that is not there has zero codes: its products are zero and leave the sum as it is,
            // whatever the scale code below 255; 127 stands for it.
            let sa = if !scale_live { 127 } else if scale_words { (scale_word >> (8 * (at % 4))) & 255 } else { *scales.add(scale_row * groups + s) as u32 };
            let token_scale = shfl_idx!(code_scale, (((j * span + g) * 4) & 31) as u32);
            let sb = if g < span { token_scale } else { 127 };
            cur = mma!([words[0][0], words[1][0], words[0][1], words[1][1]], b, cur, sa, sb);
            if at % SUB == SUB - 1 { total = add_acc!(total, cur); cur = ZERO; }
            s += 1;
        }
        // The last sub-chain of the range (after a whole one `cur` is +0.0, and x + +0.0 is x: no sum here is -0.0).
        total = add_acc!(total, cur);
        // The join: every warp parks its range, warp 0 adds the four in the contract's order.
        let smem = crate::ptx::shared::<JOIN_BYTES>();
        asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg32) smem + (tid * 16) as u32,
            a = in(reg32) total.a, b = in(reg32) total.b, c = in(reg32) total.c, d = in(reg32) total.d, options(nostack));
        crate::ptx::sync();
        if warp != 0 { return; }
        let part = |w: usize| -> Acc {
            let v = crate::ptx::load4(smem + ((w * 32 + lane) * 16) as u32);
            Acc { a: f32::from_bits(v[0]), b: f32::from_bits(v[1]), c: f32::from_bits(v[2]), d: f32::from_bits(v[3]) }
        };
        let mut word = add_acc!(add_acc!(part(0), part(1)), add_acc!(part(2), part(3)));
        // The caller's BF16 epilogue, in place (the merged decode launches store rounded words).
        if round { word = Acc { a: crate::flash_gdn::rounded(word.a), b: crate::flash_gdn::rounded(word.b), c: crate::flash_gdn::rounded(word.c), d: crate::flash_gdn::rounded(word.d) }; }
        let output = p.output as *mut f32;
        if out_live[0] {
            if row_live[0] { *output.add((token0 + 2 * t) * n + rows[0]) = word.a; }
            if row_live[1] { *output.add((token0 + 2 * t) * n + rows[1]) = word.c; }
        }
        if out_live[1] {
            if row_live[0] { *output.add((token0 + 2 * t + 1) * n + rows[0]) = word.b; }
            if row_live[1] { *output.add((token0 + 2 * t + 1) * n + rows[1]) = word.d; }
        }
    }
}
/// Block `local` of a tensor-core descriptor's grid (`next_params::mx_tc_blocks`: row tile fastest, then token tile): the
/// body of every entry that takes a descriptor with `mode == MX_TC` (the one-row decode entries, the merged batches, the
/// row stage, the small row entries, and their recorded forms). `round`: the BF16 epilogue of the merged launches.
/// Every thread of the block calls it with the same arguments; a block past the grid returns as a whole.
#[inline(always)]
pub(crate) fn layer_block(p: MxProjection, local: u32, round: bool) {
    let tiles = p.rows.div_ceil(crate::next_params::MX_TC_ROWS);
    if tiles == 0 { return; }
    block::<AHEAD, EVERY>(p, Some(((local % tiles) as usize * ROWS_BLOCK, (local / tiles) as usize * 8)), round);
}
/// Warps, tokens and staged groups of a `mx_tc_tile` block: 128 weight rows, 64 tokens, 256 columns a stage.
pub(crate) const TILE_WARPS: usize = 8;
pub(crate) const TILE_TOKENS: usize = 64;
const STAGE_GROUPS: usize = 8;
const STAGE_CODES: usize = STAGE_GROUPS * TILE_TOKENS * 32;
const STAGE_BYTES: usize = STAGE_CODES + STAGE_GROUPS * TILE_TOKENS * 4;
const _: () = assert!(STAGE_BYTES <= 48 * 1024 && STAGE_GROUPS * TILE_TOKENS == 2 * TILE_WARPS * 32);

/// Prompt groups. Grid (ceil(tokens / 64), ceil(rows / 128)), 256 threads, no dynamic shared memory. The same word and
/// the same output layout as `mx_tc_rows`. A block owns 128 weight rows (16 a warp) and 64 tokens. The activation
/// operands of the block's tokens are made once a stage of 8 groups (256 columns) in shared memory (a thread encodes
/// two (group, token) items: all 32 values of the group, so no shuffle), then each warp runs its 16 rows against the
/// 8 token groups of 8: one weight fragment feeds 8 products.
/// The token tile is on grid x, so the blocks that run together read the SAME weight rows for different tokens, and a
/// block asks for the next stage's weight lines and input rows (L2 prefetch: no value, no order) before this stage's
/// products. MEASURED run `onerow-t2-2` (same words on every shape): 1,024 token rows, us a launch, without to with
/// both: 4096 x 4096 2819.8 to 1694.5; 6144 x 4096 5709.3 to 2484.3; 2048 x 8192 2373.6 to 1875.3; 2048 x 12288
/// 3523.2 to 2835.1.
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_tile(p: MxProjection) { tile(p) }
/// The body of `mx_tc_tile`.
#[inline(always)]
pub(crate) fn tile(p: MxProjection) {
    // SAFETY: The host checks the descriptor as `MxLayer::push` does (complete E4M3 rows and UE8M0 groups in the retained
    // arena, FP32 input rows for all tokens, a disjoint output of tokens x rows words) and keeps every owner alive through
    // stream completion. Weights, scales and input are immutable; each (token, row) output word has exactly one writer (the
    // lane whose fragment holds it). The kernel traps unless round and input_bf16 are zero, cols is a multiple of 32, weights
    // and input are 16-byte aligned and the block has 256 threads. A lane reads weight bytes only of its rows below `rows`, at
    // row * cols + group * 32 + 8 * (lane % 4) .. +8 with group below cols / 32; scale bytes at row * (cols / 32) + group;
    // input words only of tokens below `tokens` and groups below cols / 32, 32 words at token * cols + group * 32. Shared
    // reads and writes stay inside the one static stage: item (group, token) owns 32 code bytes at
    // (group * 64 + token) * 32 and one scale word at STAGE_CODES + (group * 64 + token) * 4, and every item of a stage is
    // written (zeros for a token or a group past the end) before the barrier that precedes its reads. All 256 threads that
    // pass the uniform early return reach every barrier: the loop bounds depend only on the descriptor. All 32 lanes of a
    // warp take the same branch around its products (the warp's first row is live or not). A prefetch names an address
    // inside a live weight row of this lane at a group below cols / 32, or inside the input row of a token below `tokens`
    // at a group below cols / 32 (its 128 bytes), and reads nothing.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _thread_idx_x};
        let tid = _thread_idx_x() as usize;
        let (n, k, m) = (p.rows as usize, p.cols as usize, p.tokens as usize);
        if _block_dim_x() as usize != TILE_WARPS * 32 || p.round != 0 || p.input_bf16 != 0 || k % 32 != 0 || k == 0 || (p.weights | p.input) & 15 != 0 { asm!("trap;"); }
        let (token_tile, row_tile) = (_block_idx_x() as usize, _block_idx_y() as usize);
        if row_tile * TILE_WARPS * 16 >= n || token_tile * TILE_TOKENS >= m { return; }
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let groups = k / 32;
        let token_first = token_tile * TILE_TOKENS;
        let live = if m - token_first < TILE_TOKENS { m - token_first } else { TILE_TOKENS };
        let smem = crate::ptx::shared::<STAGE_BYTES>();
        let first = row_tile * TILE_WARPS * 16 + warp * 16;
        let rows = [first + g, first + g + 8];
        let row_live = [rows[0] < n, rows[1] < n];
        let warp_live = first < n;
        let weights = (p.weights as *const u8).wrapping_add(8 * t);
        let input = p.input as *const u8;
        // `cur` = the open sub-chain; `sum` = the open range's closed sub-chains; `joined` = part 0, then fl(part0 + part1);
        // `third` = part 2. One set a token group. `begin` = the open range's first group.
        let (mut cur, mut sum, mut joined, mut third) = ([ZERO; 8], [ZERO; 8], [ZERO; 8], [ZERO; 8]);
        let ends = [groups / 4, groups * 2 / 4, groups * 3 / 4];
        let mut part = 0;
        let mut begin = 0;
        macro_rules! close {() => {{
            rep8!(J, {
                let value = add_acc!(sum[J], cur[J]);
                if part == 0 { joined[J] = value; } else if part == 1 { joined[J] = add_acc!(joined[J], value); } else { third[J] = value; }
                cur[J] = ZERO;
                sum[J] = ZERO;
            });
            begin = ends[part];
            part += 1;
        }};}
        let mut stage = 0;
        while stage < groups {
            // This lane's weight bytes of the stage's groups: asked for before the encode work, so they arrive during it.
            let mut wq = [[[0u32; 2]; 2]; STAGE_GROUPS];
            if warp_live {
                rep8!(S, {
                    if stage + S < groups {
                        let mut i = 0;
                        while i < 2 {
                            if row_live[i] {
                                let (low, high): (u32, u32);
                                asm!("ld.global.nc.L1::no_allocate.v2.u32 {{{low},{high}}}, [{pointer}];", low = out(reg32) low, high = out(reg32) high,
                                    pointer = in(reg64) weights.add(rows[i] * k + (stage + S) * 32), options(nostack, readonly));
                                wq[S][i] = [low, high];
                            }
                            i += 1;
                        }
                    }
                });
            }
            // The activation operands of the stage: this thread's two (group, token) items.
            let mut item = tid;
            while item < STAGE_GROUPS * TILE_TOKENS {
                let (token, group) = (item % TILE_TOKENS, item / TILE_TOKENS);
                let mut words = [0u32; 8];
                let mut scale = 0u32;
                if token < live && stage + group < groups {
                    let pointer = input.add(((token_first + token) * k + (stage + group) * 32) * 4);
                    let mut maximum = 0u32;
                    let mut q = 0;
                    while q < 4 {
                        let x = load8!(pointer.add(q * 32));
                        rep8!(I, { let magnitude = x[I] & 0x7fff_ffff; if magnitude > maximum { maximum = magnitude; } });
                        q += 1;
                    }
                    let mut q = 0;
                    while q < 4 {
                        let (b, activation_scale) = encode!(load8!(pointer.add(q * 32)), maximum);
                        words[2 * q] = b[0];
                        words[2 * q + 1] = b[1];
                        scale = activation_scale;
                        q += 1;
                    }
                }
                let codes = smem + ((group * TILE_TOKENS + token) * 32) as u32;
                asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg32) codes, a = in(reg32) words[0], b = in(reg32) words[1], c = in(reg32) words[2], d = in(reg32) words[3], options(nostack));
                asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg32) codes + 16, a = in(reg32) words[4], b = in(reg32) words[5], c = in(reg32) words[6], d = in(reg32) words[7], options(nostack));
                asm!("st.shared.b32 [{p}], {v};", p = in(reg32) smem + (STAGE_CODES + (group * TILE_TOKENS + token) * 4) as u32, v = in(reg32) scale, options(nostack));
                item += TILE_WARPS * 32;
            }
            crate::ptx::sync();
            // The next stage's bytes are asked for now, so they arrive during this stage's products: this lane's weight
            // lines (a 64-byte line holds two groups of a row; the lanes with t == 0 ask) and this thread's two input items.
            let next = stage + STAGE_GROUPS;
            if warp_live && t == 0 {
                rep4!(S2, {
                    let mut i = 0;
                    while i < 2 {
                        if row_live[i] && next + 2 * S2 < groups {
                            asm!("prefetch.global.L2 [{pointer}];", pointer = in(reg64) (p.weights as *const u8).add(rows[i] * k + (next + 2 * S2) * 32), options(nostack, readonly));
                        }
                        i += 1;
                    }
                });
            }
            let mut item = tid;
            while item < STAGE_GROUPS * TILE_TOKENS {
                let (token, group) = (item % TILE_TOKENS, item / TILE_TOKENS);
                if token < live && next + group < groups {
                    let pointer = input.add(((token_first + token) * k + (next + group) * 32) * 4);
                    asm!("prefetch.global.L2 [{pointer}];", pointer = in(reg64) pointer, options(nostack, readonly));
                    asm!("prefetch.global.L2 [{pointer}];", pointer = in(reg64) pointer.add(64), options(nostack, readonly));
                }
                item += TILE_WARPS * 32;
            }
            if warp_live {
                rep8!(S, {
                    let s = stage + S;
                    if s < groups {
                        // Close every range that ends here (an empty range closes at once, as +0.0).
                        while part < 3 && s == ends[part] { close!(); }
                        let a = [wq[S][0][0], wq[S][1][0], wq[S][0][1], wq[S][1][1]];
                        // This lane's weight scale code of the group: row g (lanes 0 and 2 of a quad) or g + 8 (lanes 1, 3).
                        // A row that is not there has zero codes; 127 stands for its scale code.
                        let sa = if row_live[t % 2] { *(p.scales as *const u8).add(rows[t % 2] * groups + s) as u32 } else { 127 };
                        rep8!(J, {
                            if 8 * J < live {
                                let b = crate::ptx::load2(smem + ((S * TILE_TOKENS + 8 * J + g) * 32 + 8 * t) as u32);
                                // The scale code of token 8 J + g (0 with zero codes for a token past the end).
                                let sb = load_word!(smem + (STAGE_CODES + (S * TILE_TOKENS + 8 * J + g) * 4) as u32);
                                cur[J] = mma!(a, b, cur[J], sa, sb);
                            }
                        });
                        // A sub-chain of `SUB` steps of the open range ends here.
                        if (s + 1 - begin) % SUB == 0 { rep8!(J, { sum[J] = add_acc!(sum[J], cur[J]); cur[J] = ZERO; }); }
                    }
                });
            }
            crate::ptx::sync();
            stage += STAGE_GROUPS;
        }
        if !warp_live { return; }
        while part < 3 { close!(); }
        let output = p.output as *mut f32;
        rep8!(J, {
            let word = add_acc!(joined[J], add_acc!(third[J], add_acc!(sum[J], cur[J])));
            let token = token_first + 8 * J + 2 * t;
            if 8 * J + 2 * t < live {
                if row_live[0] { *output.add(token * n + rows[0]) = word.a; }
                if row_live[1] { *output.add(token * n + rows[1]) = word.c; }
            }
            if 8 * J + 2 * t + 1 < live {
                if row_live[0] { *output.add((token + 1) * n + rows[0]) = word.b; }
                if row_live[1] { *output.add((token + 1) * n + rows[1]) = word.d; }
            }
        });
    }
}

/// Groups of a `mx_tc_tile_packed` stage (128 columns): 64 tokens times 4 groups = one 32-byte item a thread.
const PACKED_STAGE: usize = 4;
const PACKED_CODES: usize = PACKED_STAGE * TILE_TOKENS * 32;
const PACKED_BYTES: usize = PACKED_CODES + TILE_TOKENS * 4;
const _: () = assert!(PACKED_STAGE * TILE_TOKENS == TILE_WARPS * 32 && PACKED_BYTES <= 48 * 1024 && SUB % PACKED_STAGE == 0 && 8 % SUB == 0);

/// The prompt tile on PACKED token rows (`mx_tc_pack` made them): `input` = `tokens` lines of `cols + cols / 32` bytes,
/// a row's E4M3 codes and then one scale code a group. The same grid, the same word and the same output layout as
/// `mx_tc_tile`. `cols` is a multiple of 1,024, so a line is whole 32-byte items, a range is whole sub-chains of `SUB`
/// groups, a sub-chain is whole stages of 4 groups, and the scale codes of a stage are one aligned word (of a weight
/// row, and of a token).
/// A block owns 128 weight rows (16 a warp) and 64 tokens. A stage: every lane loads its 8 weight bytes of each of the
/// 4 groups and one word of weight scale codes; every thread copies one (group, token) item of 32 code bytes to shared
/// memory, and the first 64 threads one word of token scale codes; then each warp runs 4 groups against the 8 token
/// groups of 8. Nothing is encoded, and between two sub-chain ends nothing but the tensor-core step computes.
/// MEASURED (bench run `r3-proof-4`): 1,024 token rows in 546 us at 4096 x 4096, 569 at 2048 x 8192, 805 at
/// 6144 x 4096, 928 at 2048 x 12288. Tried and dropped in that run: two of the four sets of sums kept in memory between
/// range ends in place of registers (734 / 720 / 1,109 / 1,145 us).
/// `round` = 1: each word is stored BF16-rounded (the prompt path's round after the product, without its own launch
/// and its second pass over the output); 0: the raw FP32 word, as before.
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_tile_packed(p: MxProjection) { tile_packed(p) }
/// The body of `mx_tc_tile_packed`.
#[inline(always)]
pub(crate) fn tile_packed(p: MxProjection) {
    // SAFETY: The host checks the descriptor (complete E4M3 rows and UE8M0 groups in the retained arena, tokens *
    // (cols + cols / 32) input bytes written by `mx_tc_pack`, a disjoint output of tokens x rows words) and keeps every owner
    // alive through stream completion. Weights, scales and input are immutable during the launch; each (token, row) output
    // word has exactly one writer (the lane whose fragment holds it). The kernel traps unless round is 0 or 1, input_bf16 is zero,
    // cols is a multiple of 1,024, weights and input are 16-byte aligned, scales and output 4-byte aligned, and the block
    // has 256 threads. A lane reads weight bytes only of its rows below `rows`, at row * cols + group * 32 + 8 * (lane % 4)
    // .. +8 with group below cols / 32, and one scale word at row * (cols / 32) + stage (4 bytes, stage a multiple of 4
    // below cols / 32). A thread reads the 32 code bytes of an item at token * line + group * 32 and a token's scale word
    // at token * line + cols + stage, token below `tokens`: inside the token's line. Shared reads and writes stay inside
    // the one static array: item (group, token) owns 32 bytes at (group * 64 + token) * 32, token `i` one word at
    // PACKED_CODES + i * 4, and every item and word of a stage is written (zeros for a token past the end) before the
    // barrier that precedes its reads. All 256 threads that pass the uniform early return reach every barrier: the loop
    // bounds depend only on the descriptor. A row or a token past the end has zero codes and the scale code 0, so its
    // products are zero and its step leaves the sum; its output words are not written.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _thread_idx_x};
        let tid = _thread_idx_x() as usize;
        let (n, k, m) = (p.rows as usize, p.cols as usize, p.tokens as usize);
        if _block_dim_x() as usize != TILE_WARPS * 32 || p.round > 1 || p.input_bf16 != 0 || k == 0 || k % 1024 != 0 || (p.weights | p.input) & 15 != 0 || (p.scales | p.output) & 3 != 0 { asm!("trap;"); }
        // `round` = 1: every word is stored BF16-rounded (`bf16::rounded`), the word `bf16_lanes_round` gives in a launch of its own.
        let round = p.round == 1;
        let groups = k / 32;
        // A packed token line: the codes, then one scale code a group.
        let line = k + groups;
        let (token_tile, row_tile) = (_block_idx_x() as usize, _block_idx_y() as usize);
        if row_tile * TILE_WARPS * 16 >= n || token_tile * TILE_TOKENS >= m { return; }
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let token_first = token_tile * TILE_TOKENS;
        let live = if m - token_first < TILE_TOKENS { m - token_first } else { TILE_TOKENS };
        let smem = crate::ptx::shared::<PACKED_BYTES>();
        let first = row_tile * TILE_WARPS * 16 + warp * 16;
        let rows = [first + g, first + g + 8];
        let row_live = [rows[0] < n, rows[1] < n];
        let weights = (p.weights as *const u8).wrapping_add(8 * t);
        let input = p.input as *const u8;
        let output = p.output as *mut f32;
        // This lane's four output words of token group J: tokens 8 J + 2 t and + 1, rows g and g + 8.
        macro_rules! put {($J:expr,$value:expr)=>{{
            let (value, token): (Acc, usize) = ($value, token_first + 8 * $J + 2 * t);
            if 8 * $J + 2 * t < live {
                if row_live[0] { *output.add(token * n + rows[0]) = value.a; }
                if row_live[1] { *output.add(token * n + rows[1]) = value.c; }
            }
            if 8 * $J + 2 * t + 1 < live {
                if row_live[0] { *output.add((token + 1) * n + rows[0]) = value.b; }
                if row_live[1] { *output.add((token + 1) * n + rows[1]) = value.d; }
            }
        }};}
        // `cur` = the open sub-chain; `sum` = the open range's closed sub-chains; `joined` = part 0, then fl(part0 + part1);
        // `third` = part 2. One set a token group.
        // Every array below is indexed by constants only (the `rep` macros), so each element is a register.
        let (mut cur, mut sum, mut joined, mut third) = ([ZERO; 8], [ZERO; 8], [ZERO; 8], [ZERO; 8]);
        let mut stage = 0;
        while stage < groups {
            // This lane's weight bytes of the stage's 4 groups, and the scale codes of those groups of the row the step
            // reads from this lane: row g (lanes 0 and 2 of a quad) or g + 8 (lanes 1 and 3). Byte S is group stage + S.
            let mut wq = [[[0u32; 2]; 2]; PACKED_STAGE];
            let mut ws = 0u32;
            if row_live[t % 2] { ws = *(p.scales as *const u32).add((rows[t % 2] * groups + stage) / 4); }
            rep4!(S, {
                rep2!(I, {
                    if row_live[I] {
                        let (low, high): (u32, u32);
                        asm!("ld.global.nc.v2.u32 {{{low},{high}}}, [{pointer}];", low = out(reg32) low, high = out(reg32) high,
                            pointer = in(reg64) weights.add(rows[I] * k + (stage + S) * 32), options(nostack, readonly));
                        wq[S][I] = [low, high];
                    }
                });
            });
            // The stage's token operands into shared memory: thread `tid` owns item (group tid / 64, token tid % 64), and
            // the first 64 threads the scale codes of the stage's 4 groups of token `tid`.
            {
                let (token, s) = (tid % TILE_TOKENS, tid / TILE_TOKENS);
                let mut words = [0u32; 8];
                if token < live {
                    let pointer = input.add((token_first + token) * line + (stage + s) * 32);
                    let (x0, x1, x2, x3, x4, x5, x6, x7): (u32, u32, u32, u32, u32, u32, u32, u32);
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x0, b = out(reg32) x1, c = out(reg32) x2, d = out(reg32) x3, pointer = in(reg64) pointer, options(nostack, readonly));
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x4, b = out(reg32) x5, c = out(reg32) x6, d = out(reg32) x7, pointer = in(reg64) pointer.add(16), options(nostack, readonly));
                    words = [x0, x1, x2, x3, x4, x5, x6, x7];
                }
                let codes = smem + ((s * TILE_TOKENS + token) * 32) as u32;
                asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg32) codes, a = in(reg32) words[0], b = in(reg32) words[1], c = in(reg32) words[2], d = in(reg32) words[3], options(nostack));
                asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg32) codes + 16, a = in(reg32) words[4], b = in(reg32) words[5], c = in(reg32) words[6], d = in(reg32) words[7], options(nostack));
                if tid < TILE_TOKENS {
                    let word = if tid < live { *(input.add((token_first + tid) * line + k + stage) as *const u32) } else { 0 };
                    asm!("st.shared.b32 [{p}], {v};", p = in(reg32) smem + (PACKED_CODES + tid * 4) as u32, v = in(reg32) word, options(nostack));
                }
            }
            crate::ptx::sync();
            // The scale codes of token 8 J + g for the stage's 4 groups (the step reads a token's code from lane 0 of a quad).
            let mut xs = [0u32; 8];
            rep8!(J, { xs[J] = load_word!(smem + (PACKED_CODES + (8 * J + g) * 4) as u32); });
            rep4!(S, {
                let a = [wq[S][0][0], wq[S][1][0], wq[S][0][1], wq[S][1][1]];
                let sa = (ws >> (8 * S)) & 255;
                rep8!(J, {
                    let b = crate::ptx::load2(smem + ((S * TILE_TOKENS + 8 * J + g) * 32 + 8 * t) as u32);
                    cur[J] = mma!(a, b, cur[J], sa, (xs[J] >> (8 * S)) & 255);
                });
            });
            let done = stage + PACKED_STAGE;
            // A sub-chain ends: `groups` is a multiple of 32, so a range starts at a multiple of `SUB` and of the stage.
            if done % SUB == 0 { rep8!(J, { sum[J] = add_acc!(sum[J], cur[J]); cur[J] = ZERO; }); }
            // A range ends (at a sub-chain end).
            if done == groups / 4 {
                rep8!(J, { joined[J] = sum[J]; sum[J] = ZERO; });
            } else if done == groups / 2 {
                rep8!(J, { joined[J] = add_acc!(joined[J], sum[J]); sum[J] = ZERO; });
            } else if done == groups / 4 * 3 {
                rep8!(J, { third[J] = sum[J]; sum[J] = ZERO; });
            }
            crate::ptx::sync();
            stage += PACKED_STAGE;
        }
        // The last range ended on a sub-chain end, so `sum` holds it whole.
        rep8!(J, {
            let mut word = add_acc!(joined[J], add_acc!(third[J], sum[J]));
            if round { word = Acc { a: crate::bf16::rounded(word.a), b: crate::bf16::rounded(word.b), c: crate::bf16::rounded(word.c), d: crate::bf16::rounded(word.d) }; }
            put!(J, word);
        });
    }
}

/// The packed activation operands of prompt groups, made once for every tile that reads these token rows
/// (`mx_tc_tile_packed`). `input`: `tokens` rows of `cols` FP32 words, as `mx_tc_tile` reads them. `output`: `tokens`
/// lines of `cols + cols / 32` bytes: the row's E4M3 codes, then one byte a group of 32 columns, the group's UE8M0 scale
/// code (255: a value of the group is not finite). Codes and scale code are the ones `mx_tc_tile` makes in its stage:
/// the same macro on the same loads. `cols` is a multiple of 1,024, so a line is whole 32-byte items and every byte of
/// `output` is written. Grid (ceil(tokens * (cols / 32) / 256)), 256 threads: thread `i` of the launch owns item `i`,
/// group `i % (cols / 32)` of token `i / (cols / 32)`.
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_pack(p: MxProjection) {
    // SAFETY: The host gives tokens * cols FP32 input words and tokens * (cols + cols / 32) output bytes, both 16-byte
    // aligned, disjoint, alive through stream completion; the input is immutable. The kernel traps unless round and
    // input_bf16 are zero and cols is a multiple of 1,024. A thread past the last item returns before any read. Thread i
    // reads the 32 input words of its item, at (token * cols + group * 32) * 4, and writes its item's 32 code bytes at
    // token * line + group * 32 and its one scale byte at token * line + cols + group: each output byte has one writer.
    unsafe {
        use core::arch::asm;
        let k = p.cols as usize;
        if p.round != 0 || p.input_bf16 != 0 || k == 0 || k % 1024 != 0 || (p.input | p.output) & 15 != 0 { asm!("trap;"); }
        let groups = k / 32;
        let item = crate::index();
        let (token, group) = (item / groups, item % groups);
        if token >= p.tokens as usize { return; }
        let pointer = (p.input as *const u8).add((token * k + group * 32) * 4);
        let mut maximum = 0u32;
        let mut q = 0;
        while q < 4 {
            let x = load8!(pointer.add(q * 32));
            rep8!(I, { let magnitude = x[I] & 0x7fff_ffff; if magnitude > maximum { maximum = magnitude; } });
            q += 1;
        }
        let mut words = [0u32; 8];
        let mut scale = 0u32;
        let mut q = 0;
        while q < 4 {
            let (b, activation_scale) = encode!(load8!(pointer.add(q * 32)), maximum);
            words[2 * q] = b[0];
            words[2 * q + 1] = b[1];
            scale = activation_scale;
            q += 1;
        }
        let out = (p.output as *mut u8).add(token * (k + groups));
        asm!("st.global.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg64) out.add(group * 32), a = in(reg32) words[0], b = in(reg32) words[1], c = in(reg32) words[2], d = in(reg32) words[3], options(nostack));
        asm!("st.global.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg64) out.add(group * 32 + 16), a = in(reg32) words[4], b = in(reg32) words[5], c = in(reg32) words[6], d = in(reg32) words[7], options(nostack));
        *out.add(k + group) = scale as u8;
    }
}

/// The E4M3 code of 224 (1.75 x 2^7): the largest code of a group whose largest quantized value sits exactly on the top
/// value of the scale code below its own.
const CODE_224: u32 = 118;
/// The scale code of the largest finite BF16 values: a quantized value of such a group can pass the largest float.
const LAST_SCALE: u32 = 247;

/// The quantize pass of MXFP8 prompt rows (`Op::Mxfp8Quantize`, plane 0: `causal_small::mxfp8_quantize`) AND
/// `mx_tc_pack` in ONE launch, without the FP32 copy of the quantized rows between them. `input`: `tokens` rows of
/// `cols` FP32 words, the rows the quantizer reads. `output`: the bytes `mx_tc_pack` writes from the quantizer's
/// output: `tokens` lines of `cols + cols / 32` bytes, the row's E4M3 codes, then one UE8M0 scale code a group of 32
/// columns (255: a quantized value of the group is not finite).
///
/// Why these are `mx_tc_pack`'s bytes. The quantizer's source is the BF16-rounded input word (`bf16::rounded`).
/// A group whose sources are all finite and whose scale code `e` (the rule inside `encode!`, on the largest source
/// magnitude) is below `LAST_SCALE`:
/// - The quantizer writes `d = value(code) * 2^(e - 127)`, exact in FP32: `code` is the nearest E4M3 value of
///   `source * 2^(127 - e)`, ties to the even code, with the source's sign: what `cvt.rn.satfinite.e4m3x2.f32` gives.
///   So `encode!` on the sources with their own largest magnitude gives the quantizer's codes and `e`.
/// - `mx_tc_pack` takes its scale code from the largest `|d|`. That is `e` again, except when the largest value rounded
///   down onto 224 x 2^(e - 127), the top value of code `e - 1` (its code is then `CODE_224`): then the pack's scale
///   code is `e - 1` (when `e` is not 0) and it writes every value at that scale, which doubles it exactly: a code
///   magnitude below 8 doubles, one from 8 gains 8 (one binade); the sign bit stays.
/// Every other group (a source that is not finite: the quantizer writes NaN in all 32 lanes; scale code `LAST_SCALE`:
/// a quantized value can be an infinity) takes the two steps as they stand: each lane's quantized word
/// (`mxfp8_quantize::finish`, the quantizer's own function), then `mx_tc_pack`'s maximum and `encode!` on those words.
/// `cols` is a multiple of 1,024, so a line is whole 32-byte items and every byte of `output` is written. Grid
/// (ceil(tokens * (cols / 32) / 256)), 256 threads: thread `i` owns item `i`, group `i % (cols / 32)` of token
/// `i / (cols / 32)`, as in `mx_tc_pack`.
/// Proof: kernel-bench --mx-quant <base image> <image> pack: every byte against `causal_fast` then `mx_tc_pack`.
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_quantize_pack(p: MxProjection) {
    // SAFETY: The host gives tokens * cols FP32 input words and tokens * (cols + cols / 32) output bytes, both 16-byte
    // aligned, disjoint, alive through stream completion; the input is immutable. The kernel traps unless round and
    // input_bf16 are zero and cols is a multiple of 1,024. A thread past the last item returns before any read. Thread i
    // reads the 32 input words of its item, at (token * cols + group * 32) * 4 (`pointer.add(lane * 4)`, lane below 32),
    // and writes its item's 32 code bytes at token * line + group * 32 and its one scale byte at
    // token * line + cols + group: each output byte has one writer. No shared memory and no barrier.
    unsafe {
        use core::arch::asm;
        let k = p.cols as usize;
        if p.round != 0 || p.input_bf16 != 0 || k == 0 || k % 1024 != 0 || (p.input | p.output) & 15 != 0 { asm!("trap;"); }
        let groups = k / 32;
        let item = crate::index();
        let (token, group) = (item / groups, item % groups);
        if token >= p.tokens as usize { return; }
        let pointer = (p.input as *const u8).add((token * k + group * 32) * 4);
        // The largest magnitude of the group's sources.
        let mut maximum = 0u32;
        let mut q = 0;
        while q < 4 {
            let x = load8!(pointer.add(q * 32));
            rep8!(I, { let magnitude = crate::bf16::rounded(f32::from_bits(x[I])).to_bits() & 0x7fff_ffff; if magnitude > maximum { maximum = magnitude; } });
            q += 1;
        }
        let finite = maximum < 0x7f80_0000;
        let mut words = [0u32; 8];
        let mut scale = ((maximum >> 23) + (maximum & 0x007f_ffff > 0x0060_0000) as u32).saturating_sub(8);
        if finite && scale < LAST_SCALE {
            let mut q = 0;
            while q < 4 {
                let x = load8!(pointer.add(q * 32));
                let mut source = [0u32; 8];
                rep8!(I, { source[I] = crate::bf16::rounded(f32::from_bits(x[I])).to_bits(); });
                let (b, _) = encode!(source, maximum);
                words[2 * q] = b[0];
                words[2 * q + 1] = b[1];
                q += 1;
            }
            // The code of the group's largest value at the quantizer's scale.
            let top = cvt_pair!(mul_rn!(f32::from_bits(maximum), f32::from_bits((254u32.wrapping_sub(scale)) << 23)), 0.0f32) & 0x7f;
            if scale != 0 && top == CODE_224 {
                // The quantized group fits the scale code below: `mx_tc_pack` writes every value at twice its code value.
                scale -= 1;
                rep8!(W, {
                    let mut doubled = 0u32;
                    rep4!(B, {
                        let code = (words[W] >> (8 * B)) & 0xff;
                        let magnitude = code & 0x7f;
                        doubled |= ((code & 0x80) | if magnitude < 8 { 2 * magnitude } else { magnitude + 8 }) << (8 * B);
                    });
                    words[W] = doubled;
                });
            }
        } else {
            // The two steps as they stand: the quantizer's word of each lane, then `mx_tc_pack` on those words.
            let quantized = |lane: usize| -> u32 {
                if finite { crate::mxfp8_quantize::finish(f32::from_bits(maximum), *(pointer.add(lane * 4) as *const f32), 0, false).to_bits() } else { f32::NAN.to_bits() }
            };
            let mut largest = 0u32;
            let mut lane = 0;
            while lane < 32 {
                let magnitude = quantized(lane) & 0x7fff_ffff;
                if magnitude > largest { largest = magnitude; }
                lane += 1;
            }
            let mut q = 0;
            while q < 4 {
                let mut decoded = [0u32; 8];
                rep8!(I, { decoded[I] = quantized(q * 8 + I); });
                let (b, code) = encode!(decoded, largest);
                words[2 * q] = b[0];
                words[2 * q + 1] = b[1];
                scale = code;
                q += 1;
            }
        }
        let out = (p.output as *mut u8).add(token * (k + groups));
        asm!("st.global.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg64) out.add(group * 32), a = in(reg32) words[0], b = in(reg32) words[1], c = in(reg32) words[2], d = in(reg32) words[3], options(nostack));
        asm!("st.global.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg64) out.add(group * 32 + 16), a = in(reg32) words[4], b = in(reg32) words[5], c = in(reg32) words[6], d = in(reg32) words[7], options(nostack));
        *out.add(k + group) = scale as u8;
    }
}

/// Bench only: one raw step a warp. Case `c` (block `c`, 32 threads) reads 16 weight rows of 32 E4M3 codes at
/// `weights + c * 512`, 8 token rows of 32 E4M3 codes at `input + c * 256`, 16 weight scale codes and 8 token scale
/// codes at `scales + c * 32` (bytes 0..16 and 16..24), and the OLD SUMS `[16][8]` FP32 words at `output + c * 512`,
/// where it writes `D` in their place. `rows` = the case count.
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_probe(p: MxProjection) {
    // SAFETY: bench only. The host gives `rows` cases: rows * 512 weight bytes, rows * 256 input bytes, rows * 32 scale
    // bytes and rows * 128 output words, 4-byte aligned, disjoint, alive through stream completion. A block past `rows`
    // returns before any read. A lane reads its own two weight rows, its quad's token row, two scale bytes of its case and
    // its own four output words, and writes those four words.
    unsafe {
        use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
        if _block_dim_x() != 32 || (p.weights | p.input | p.output) & 3 != 0 { core::arch::asm!("trap;"); }
        let case = _block_idx_x() as usize;
        if case >= p.rows as usize { return; }
        let lane = _thread_idx_x() as usize;
        let (g, t) = (lane / 4, lane % 4);
        let w = (p.weights as *const u32).add(case * 128);
        let x = (p.input as *const u32).add(case * 64);
        let a = [*w.add(g * 8 + 2 * t), *w.add((g + 8) * 8 + 2 * t), *w.add(g * 8 + 2 * t + 1), *w.add((g + 8) * 8 + 2 * t + 1)];
        let b = [*x.add(g * 8 + 2 * t), *x.add(g * 8 + 2 * t + 1)];
        let codes = (p.scales as *const u8).add(case * 32);
        let (sa, sb) = (*codes.add(g + 8 * (t % 2)) as u32, *codes.add(16 + g) as u32);
        let old = (p.output as *const f32).add(case * 128);
        let c = Acc { a: *old.add(g * 8 + 2 * t), b: *old.add(g * 8 + 2 * t + 1), c: *old.add((g + 8) * 8 + 2 * t), d: *old.add((g + 8) * 8 + 2 * t + 1) };
        let d = mma!(a, b, c, sa, sb);
        let out = (p.output as *mut f32).add(case * 128);
        *out.add(g * 8 + 2 * t) = d.a;
        *out.add(g * 8 + 2 * t + 1) = d.b;
        *out.add((g + 8) * 8 + 2 * t) = d.c;
        *out.add((g + 8) * 8 + 2 * t + 1) = d.d;
    }
}

/// Bench only: the activation conversion. Thread `i` of `rows` reads the FP32 pair `input[2i], input[2i + 1]` and writes
/// the two E4M3 codes as one word `output[i]` (low byte = the first value's code).
#[no_mangle]
pub extern "ptx-kernel" fn mx_tc_probe_cvt(p: MxProjection) {
    // SAFETY: bench only. The host gives 2 * rows input words and rows output words, 4-byte aligned, disjoint, alive through
    // stream completion. A thread past `rows` returns before any read; thread i reads and writes only its own words.
    unsafe {
        let i = crate::index();
        if (p.input | p.output) & 3 != 0 { core::arch::asm!("trap;"); }
        if i >= p.rows as usize { return; }
        let x = (p.input as *const f32).add(2 * i);
        *(p.output as *mut u32).add(i) = cvt_pair!(*x, *x.add(1));
    }
}
