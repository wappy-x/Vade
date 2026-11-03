//! Flash projections, using the dense engine's canonical BF16 MMA.
//! Weights are permuted in place; one resident copy is retained. Four fixed K ranges
//! merge as (p0+p1)+(p2+p3), independent of token count and output tiling.
use crate::{canonical::{self, Acc, ZERO, add, madd},
    kernel_params::{Bf16Rows, FlashBf16Pack, FlashBf16Input, flash_bf16_packed, flash_bf16_element, flash_bf16_decode_split, TERMS}, ptx};
use core::arch::nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x};

/// Use the dense engine's operand conversion and packed A layout.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_input(p: FlashBf16Input) {
    let lanes = if matches!(p.lanes, 4 | 8 | 16) { p.lanes as usize } else { 32 };
    if lanes == 32 && p.tokens >= 64 && p.cols % 64 == 0 && core::arch::nvptx::_block_dim_x() == 128 {
        if p.hc_silu != 0 { pack_rows::<true>(p) } else { pack_rows::<false>(p) }
        return;
    }
    if p.hc_silu != 0 {
        crate::gemm::pack_f32_input::<true, true>(p.input as *const f32, p.output as *mut u32, p.tokens, p.cols, lanes);
    } else {
        crate::gemm::pack_f32_input::<true, false>(p.input as *const f32, p.output as *mut u32, p.tokens, p.cols, lanes);
    }
}

/// The prompt pack (32 lanes, K % 64 == 0) staged through shared memory: CTA b of the
/// one-thread-per-fragment grid packs K16 tiles 4b..4b+4, one 16-row group, after loading its
/// 16 x 64 FP32 block with 16-byte loads (256 contiguous bytes per row, not 32 per row per warp).
/// Every fragment word is the same split2 pair (or 0 past the last row) as pack_f32_input's.
#[inline(always)]
unsafe fn pack_rows<const HC_SILU: bool>(p: FlashBf16Input) {
    // Row stride 72 floats: the 8-byte fragment reads of a half warp hit 32 distinct banks.
    const S: usize = 72;
    let (t, k) = (p.tokens as usize, p.cols as usize);
    let kt = k / 16;
    let tid = _thread_idx_x() as usize;
    let first = _block_idx_x() as usize * 4;
    if first >= t.div_ceil(16) * kt {
        return;
    }
    let (row0, col0) = (first / kt * 16, first % kt * 16);
    let block = ptx::shared::<{ 16 * S * 4 }>();
    let src = p.input as *const f32;
    for j in 0..2 {
        let (r, c) = ((tid + j * 128) / 16, (tid + j * 128) % 16 * 4);
        let v: [f32; 4] = if row0 + r < t {
            let (a, b, x, y): (f32, f32, f32, f32);
            core::arch::asm!("ld.global.v4.f32 {{{a}, {b}, {x}, {y}}}, [{p}];", a = out(reg32) a, b = out(reg32) b,
                x = out(reg32) x, y = out(reg32) y, p = in(reg64) src.add((row0 + r) * k + col0 + c));
            [a, b, x, y]
        } else {
            [0.0; 4]
        };
        for (e, &x) in v.iter().enumerate() {
            ptx::store(block + ((r * S + c + e) * 4) as u32, x);
        }
    }
    ptx::sync();
    let (lane, warp) = (tid % 32, tid / 32);
    let (r, c) = (lane / 4, warp * 16 + lane % 4 * 2);
    let value = |r: usize, c: usize| {
        let x = ptx::load(block + ((r * S + c) * 4) as u32);
        if HC_SILU { crate::flash_ops::silu(x, 1.0, 4.0, 0) } else { x }
    };
    let load = |r: usize, c: usize| if row0 + r >= t { (0, 0) } else { crate::gemm::split2_pair(value(r, c), value(r, c + 1)) };
    let pairs = [load(r, c), load(r + 8, c), load(r, c + 8), load(r + 8, c + 8)];
    let plane_words = t.div_ceil(16) * kt * 128;
    let at = ((first + warp) * 32 + lane) * 4;
    let dst = p.output as *mut u32;
    core::arch::asm!("st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg64) dst.add(at),
        a = in(reg32) pairs[0].0, b = in(reg32) pairs[1].0, c = in(reg32) pairs[2].0, d = in(reg32) pairs[3].0);
    core::arch::asm!("st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg64) dst.add(plane_words + at),
        a = in(reg32) pairs[0].1, b = in(reg32) pairs[1].1, c = in(reg32) pairs[2].1, d = in(reg32) pairs[3].1);
}

/// Each CTA's four complete rows have a disjoint read/write footprint. All
/// original words are in shared memory before any resident word is replaced.
/// Block128, grid N/4, dynamic shared 8*K bytes. N%4=0, K%16=0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_pack(p: FlashBf16Pack) {
    if p.source != 0 {
        crate::gemm::pack_bf16_weights(p.source as *const u16, p.weight as *mut u32, p.rows, p.cols);
        return;
    }
    let tid = _thread_idx_x() as usize;
    let k = p.cols as usize;
    let w = (p.weight as *mut u16).add(_block_idx_x() as usize * 4 * k);
    let mut i = tid;
    while i < k / 2 {
        ptx::copy16((i * 16) as u32, w.add(i * 8).cast());
        i += 128;
    }
    ptx::commit();
    ptx::wait();
    ptx::sync();
    let mut fragment = tid;
    while fragment < k {
        let (tile, lane) = (fragment / 16, fragment % 16);
        let src = ((lane / 4 * k + tile * 16 + lane % 4 * 2) * 2) as u32;
        let (lo, hi) = (ptx::load(src).to_bits(), ptx::load(src + 16).to_bits());
        *w.cast::<u64>().add(fragment) = lo as u64 | ((hi as u64) << 32);
        fragment += 128;
    }
}

#[inline(always)]
unsafe fn pair(x: *const f32) -> (u32, u32) {
    let (lo, hi): (f32, f32);
    core::arch::asm!("ld.global.v2.f32 {{{a}, {b}}}, [{p}];",
        a = out(reg32) lo, b = out(reg32) hi, p = in(reg64) x);
    crate::gemm::split2_pair(lo, hi)
}

#[inline(always)]
fn target(p: Bf16Rows, row: usize) -> (*const u16, *mut f32, usize, usize) {
    if row < p.rows as usize {
        (p.weight as *const u16, p.output as *mut f32, row, p.stride as usize)
    } else {
        (p.weight2 as *const u16, p.output2 as *mut f32, row - p.rows as usize, p.stride2 as usize)
    }
}

#[inline(always)]
fn matrix_rows(p: Bf16Rows, row: usize) -> usize {
    if row < p.rows as usize { p.rows as usize } else { p.rows2 as usize }
}

#[inline(always)]
fn packed(p: Bf16Rows, row: usize) -> bool {
    flash_bf16_packed(if row < p.rows as usize { p.rows } else { p.rows2 } as usize, p.cols as usize)
}

#[inline(always)]
unsafe fn weight_fragment(p: Bf16Rows, row: usize, col: usize) -> [u32; 2] {
    if row >= (p.rows + p.rows2) as usize || col >= p.cols as usize { return [0; 2]; }
    let (w, _, local, _) = target(p, row);
    let k = p.cols as usize;
    if packed(p, row) {
        let address = w.add(flash_bf16_element(local, col, k, matrix_rows(p, row)));
        let value: u64;
        core::arch::asm!("ld.global.u64 {v}, [{p}];", v = out(reg64) value, p = in(reg64) address);
        [value as u32, (value >> 32) as u32]
    } else {
        [*w.add(local * k + col).cast::<u32>(),
            if col + 8 < k { *w.add(local * k + col + 8).cast::<u32>() } else { 0 }]
    }
}

#[inline(always)]
unsafe fn put(p: Bf16Rows, value: f32, token: usize, row: usize) {
    if token < p.tokens as usize && row < (p.rows + p.rows2) as usize {
        let (_, y, local, stride) = target(p, row);
        *y.add(token * stride + local) = value;
    }
}

#[inline(always)]
unsafe fn write(p: Bf16Rows, value: Acc, token: usize, row: usize) {
    put(p, value.a, token, row);
    put(p, value.b, token, row + 1);
    put(p, value.c, token + 8, row);
    put(p, value.d, token + 8, row + 1);
}

#[inline(always)]
unsafe fn dot_range<const DEEP: bool, const INPUT_BF16: bool, const RANGES: u32>(p: Bf16Rows, row: usize, lane: usize, part: usize) -> Acc {
    // Callers select one matrix first, so its layout is a constant of the loop.
    // HC is never a K-split shard (one at four ranks is at most 1536 wide).
    if p.cols == 10240 && p.tokens == 1 && (p.rows == 320 || p.rows == 4) {
        return dot_hc::<INPUT_BF16>(p, row, lane, part);
    }
    if p.rows % 64 == 0 && p.cols % 16 == 0 {
        return dot_inner::<true, true, DEEP, INPUT_BF16, false, false, RANGES>(p, row, lane, part);
    }
    if flash_bf16_packed(p.rows as usize, p.cols as usize) {
        dot_inner::<true, false, DEEP, INPUT_BF16, false, false, RANGES>(p, row, lane, part)
    } else { dot_inner::<false, false, DEEP, INPUT_BF16, false, false, RANGES>(p, row, lane, part) }
}

/// Solo HC: forty native eight-row tiles plus one compact four-row matrix.
/// Advance packed B addresses by two K64 blocks per fetch group. The eight
/// operands still reach the same ascending K16 MMAs and canonical range merge.
#[inline(always)]
unsafe fn dot_hc<const INPUT_BF16: bool>(p: Bf16Rows, row: usize, lane: usize, part: usize) -> Acc {
    let tail = p.rows == 4;
    let valid = !tail || lane < 16;
    let base = if tail { (lane % 16) * 4 }
        else { row / 64 * (64 * 10240) + row % 64 / 8 * 512 + lane * 4 };
    let (step, block) = if tail { (64, 256) } else { (128, 4096) };
    let (start, end) = canonical::range(640, part);
    let mut w = (p.weight as *const u16).add(base + start / 4 * block);
    let load = |tile: usize, at: *const u16| {
        let value: u64;
        core::arch::asm!("ld.global.nc.u64 {v}, [{p}];", v = out(reg64) value, p = in(reg64) at);
        let b = if valid { [value as u32, (value >> 32) as u32] } else { [0; 2] };
        let c = tile * 16 + lane % 4 * 2;
        let words = |c: usize| {
            if INPUT_BF16 {
                let x = (p.input as *const u32).add(c / 2);
                (*x, *x.add(10240 / 2))
            } else { pair((p.input as *const f32).add(c)) }
        };
        let (c0, c8) = (words(c), words(c + 8));
        ([c0.0, c0.1, c8.0, c8.1], b)
    };
    let mut acc = ZERO;
    let mut tile = start;
    while tile < end {
        // Hoist matrix/tile address work; issue eight reads before their MMAs.
        let v0 = load(tile, w);
        let v1 = load(tile + 1, w.add(step));
        let v2 = load(tile + 2, w.add(2 * step));
        let v3 = load(tile + 3, w.add(3 * step));
        let v4 = load(tile + 4, w.add(block));
        let v5 = load(tile + 5, w.add(block + step));
        let v6 = load(tile + 6, w.add(block + 2 * step));
        let v7 = load(tile + 7, w.add(block + 3 * step));
        acc = madd(v0.0, v0.1, acc);
        acc = madd(v1.0, v1.1, acc);
        acc = madd(v2.0, v2.1, acc);
        acc = madd(v3.0, v3.1, acc);
        acc = madd(v4.0, v4.1, acc);
        acc = madd(v5.0, v5.1, acc);
        acc = madd(v6.0, v6.1, acc);
        acc = madd(v7.0, v7.1, acc);
        tile += 8;
        // Do not form a pointer past the selected matrix after its last group.
        if tile < end { w = w.add(2 * block); }
    }
    planes(acc)
}

#[inline(always)]
unsafe fn dot_inner<const PACKED: bool, const NATIVE: bool, const DEEP: bool, const INPUT_BF16: bool, const SINGLE: bool, const LBF: bool, const RANGES: u32>(p: Bf16Rows, row: usize, lane: usize, part: usize) -> Acc {
    let k = p.cols as usize;
    let wr = row + if SINGLE { 0 } else { lane / 4 };
    let (start, end) = if RANGES == 0 { canonical::range(k.div_ceil(16), part) } else {
        match canonical::span(k.div_ceil(16), part, RANGES) { Some(span) => span, None => return canonical::EMPTY }
    };
    let valid = NATIVE || wr < (p.rows + p.rows2) as usize;
    let (w, _, local, _) = target(p, wr);
    // Decode has already selected one matrix (rows2 == 0). Native tiles contain
    // eight complete rows, so their base never needs the second-matrix selector.
    // Rows past the matrix read the matrix's first row and select zero, and rows
    // without a token read token 0: every fetch below issues without a
    // lane-divergent branch, so a deep step issues all of its loads before the
    // first conversion waits.
    let (w, local) = if NATIVE { (p.weight as *const u16, local) }
        else if valid { (w, local) } else { (p.weight as *const u16, 0) };
    let base = if PACKED { flash_bf16_element(local, lane % 4 * 2, k, p.rows as usize) }
        else { local * k + lane % 4 * 2 };
    let stride = if PACKED { 64 } else { 16 };
    let token = if p.tokens == 1 { 0 } else { lane / 4 };
    let live = token < p.tokens as usize;
    let x_row = if live { token * k } else { 0 };
    let mut acc = ZERO;
    // LBF12 twin in weight2 (flash_bf16_lbf12_twin of this native matrix, K <= 4096): one
    // 8-row x K32 slot per tile pair, 12 bytes per lane, row bases after the slots, then
    // ceil(K / 1024) flag words per 8-row group. A flagged slot decodes to the identical B
    // fragments.
    let (pairs, slots) = (k / 32, p.rows as usize / 8 * (k / 32));
    let twin = p.weight2 as *const u8;
    let words = pairs.div_ceil(32);
    let fl = twin.add(slots * 392 + row / 8 * words * 4).cast::<u32>();
    let flags = if LBF { [*fl, if words > 1 { *fl.add(1) } else { 0 }, if words > 2 { *fl.add(2) } else { 0 },
        if words > 3 { *fl.add(3) } else { 0 }] } else { [0; 4] };
    let load = |tile: usize| {
        let c = tile * 16 + lane % 4 * 2;
        // Packed layouts have K % 16 == 0, so only row-major shards end inside
        // a K16 tile; their columns past K read the matrix's first words.
        let (in_k, in_k8) = (PACKED || c < k, PACKED || c + 8 < k);
        let at = if !in_k { p.weight as *const u16 }
            else if NATIVE { w.add((canonical::weight_tile(row / 8, tile, k / 16) * 32 + lane) * 4) }
            else { w.add(base + tile * stride) };
        let (slot, q) = (row / 8 * pairs + tile / 2, tile / 2);
        let word = if q < 32 { flags[0] } else if q < 64 { flags[1] } else if q < 96 { flags[2] } else { flags[3] };
        let b = if LBF && word >> (q % 32) & 1 != 0 {
            let v = dense_latency::lbf12_decode(twin.add(slot * 384), twin.add(slots * 384 + slot * 8), lane);
            if tile % 2 == 0 { [v[0], v[1]] } else { [v[2], v[3]] }
        } else if PACKED {
            let value: u64;
            core::arch::asm!("ld.global.nc.u64 {v}, [{p}];", v = out(reg64) value, p = in(reg64) at);
            [value as u32, (value >> 32) as u32]
        } else { [*at.cast::<u32>(), *(if in_k8 { at.add(8) } else { at }).cast::<u32>()] };
        let b = [if valid && in_k { b[0] } else { 0 }, if valid && in_k8 { b[1] } else { 0 }];
        // Rows 0-7 of A hold the hi terms of up to eight verification/chat
        // tokens and rows 8-15 the lo terms of the same tokens. mma.sync forms
        // every output from its own A row, so one instruction per K16 tile
        // advances both chains: H in rows 0-7 (.a/.b), L in rows 8-15 (.c/.d).
        let column = |c: usize, inside: bool| if inside { c } else { 0 };
        let (c0, c8) = if INPUT_BF16 {
            let words = |c: usize| {
                let x = (p.input as *const u32).add(c / 2);
                (*x, *x.add(k / 2))
            };
            (words(column(c, in_k)), words(column(c + 8, in_k8)))
        } else {
            let x = |c: usize| pair((p.input as *const f32).add(x_row + c));
            (x(column(c, in_k)), x(column(c + 8, in_k8)))
        };
        let (at0, at8) = (live && in_k, live && in_k8);
        let (c0, c8) = (if at0 { c0 } else { (0, 0) }, if at8 { c8 } else { (0, 0) });
        ([c0.0, c0.1, c8.0, c8.1], b)
    };
    let mut tile = start;
    if k == 320 && RANGES == 0 {
        // Each canonical range is exactly five K16 tiles. Fetch the entire
        // range before its dependent MMAs, including the fifth operand.
        let (a0, b0) = load(tile);
        let (a1, b1) = load(tile + 1);
        let (a2, b2) = load(tile + 2);
        let (a3, b3) = load(tile + 3);
        let (a4, b4) = load(tile + 4);
        acc = madd(a0, b0, acc);
        acc = madd(a1, b1, acc);
        acc = madd(a2, b2, acc);
        acc = madd(a3, b3, acc);
        return planes(madd(a4, b4, acc));
    }
    if DEEP {
        macro_rules! steps {
            ($($offset:literal => $v:ident),*) => {{
                $(let $v = load(tile + $offset);)*
                $(acc = madd($v.0, $v.1, acc);)*
            }};
        }
        while tile + 16 <= end {
            steps!(0=>v0, 1=>v1, 2=>v2, 3=>v3, 4=>v4, 5=>v5, 6=>v6, 7=>v7,
                8=>v8, 9=>v9, 10=>v10, 11=>v11, 12=>v12, 13=>v13, 14=>v14, 15=>v15);
            tile += 16;
        }
    }
    while tile + 4 <= end {
        // Issue four independent weight/input reads before the dependent MMAs.
        let (a0, b0) = load(tile);
        let (a1, b1) = load(tile + 1);
        let (a2, b2) = load(tile + 2);
        let (a3, b3) = load(tile + 3);
        acc = madd(a0, b0, acc);
        acc = madd(a1, b1, acc);
        acc = madd(a2, b2, acc);
        acc = madd(a3, b3, acc);
        tile += 4;
    }
    while tile < end {
        let (a, b) = load(tile);
        acc = madd(a, b, acc);
        tile += 1;
    }
    planes(acc)
}

/// One range's P = H + L for the row-0-7 outputs of a two-plane decode tile.
/// Callers read only those outputs.
#[inline(always)]
fn planes(t: Acc) -> Acc { Acc { a: t.a + t.c, b: t.b + t.d, c: 0.0, d: 0.0 } }

/// Large projections give each warp eight outputs and merge K ranges in
/// registers. Narrow or short-K projections distribute the same ranges across
/// four warps to fill the GPU without duplicate weight reads or MMAs.
/// A rows 0-7 carry up to eight tokens' hi terms and rows 8-15 their lo terms,
/// so widths 1..8 share one weight pass and one MMA per K16 tile without input
/// packing or a prompt tile.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1(p: Bf16Rows) {
    // A four-rank K-split shard runs its own two-range copy; the four-range
    // copies are the fixed-shape entries' code.
    let (cols, ranges) = crate::kernel_params::bf16_cols_ranges(p.cols);
    if ranges == 2 { decode::<false, 2>(Bf16Rows { cols, ..p }); }
    else if wide_decode_lookahead(&p) { decode::<true, 0>(p); }
    else { decode::<false, 0>(p); }
}

/// Wide singleton/draft heads issue sixteen independent operand reads ahead
/// of the same ascending K16 MMAs. Range boundaries and merges stay fixed.
#[inline(always)]
fn wide_decode_lookahead(p: &Bf16Rows) -> bool {
    p.tokens <= 4 && p.rows >= 32768 && p.rows2 == 0
}

/// Block256, <=256 tokens, eight outputs, dynamic shared 16*K bytes.
/// Stage each B fragment once, then reuse it across 32 independent M8 groups.
/// Each group executes the decode K16 MMA sequence, H+L per range and the
/// same four-range balanced merge. No input packing or output rounding changes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_prompt(p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let row = _block_idx_x() as usize * 8;
    let (k, kt) = (p.cols as usize, p.cols as usize / 16);
    let mut i = tid;
    while i < kt * 32 {
        let (tile, l) = (i / 32, i % 32);
        let b = weight_fragment(p, row + l / 4, tile * 16 + l % 4 * 2);
        ptx::store((i * 8) as u32, f32::from_bits(b[0]));
        ptx::store((i * 8 + 4) as u32, f32::from_bits(b[1]));
        i += 256;
    }
    ptx::sync();
    let (mut left, mut right) = ([ZERO; 4], [ZERO; 4]);
    for part in 0..4 {
        let mut sums = [ZERO; 4];
        let (start, end) = canonical::range(kt, part);
        for tile in start..end {
            let b = ptx::load2(((tile * 32 + lane) * 8) as u32);
            let c = tile * 16 + lane % 4 * 2;
            for group in 0..4 {
                let token = warp * 8 + group * 64 + lane / 4;
                let live = token < p.tokens as usize;
                let x = (p.input as *const f32).add(if live { token * k + c } else { 0 });
                let (a0, a8) = if live { (pair(x), pair(x.add(8))) } else { ((0, 0), (0, 0)) };
                sums[group] = madd([a0.0, a0.1, a8.0, a8.1], b, sums[group]);
            }
        }
        for group in 0..4 {
            let value = planes(sums[group]);
            match part {
                0 => left[group] = value, 1 => left[group] = add(left[group], value),
                2 => right[group] = value, _ => right[group] = add(right[group], value),
            }
        }
    }
    for group in 0..4 {
        let value = add(left[group], right[group]);
        let token = warp * 8 + group * 64 + lane / 4;
        put(p, value.a, token, row + lane % 4 * 2);
        put(p, value.b, token, row + lane % 4 * 2 + 1);
    }
}

/// Independent FP32 row groups use the unchanged eight-token decode arithmetic.
/// Grid Y replaces host launches; each CTA sees exactly the old local descriptor.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_rows(mut p: Bf16Rows) {
    let deep = wide_decode_lookahead(&p);
    let first = _block_idx_y() as u32 * 8;
    if first >= p.tokens { return; }
    p.input += first as u64 * p.cols as u64 * 4;
    p.output += first as u64 * p.stride as u64 * 4;
    p.output2 += first as u64 * p.stride2 as u64 * 4;
    p.tokens = (p.tokens - first).min(8);
    if deep { decode::<true, 0>(p); } else { decode::<false, 0>(p); }
}

// Complete K=320 native tiles keep the generic edge/layout handling and FP32
// input conversion out of the BF16-input kernel's register allocation.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_320(mut p: Bf16Rows) {
    // hc_silu bit 1: weight2 holds the matrix's LBF12 twin (one matrix, rows2 == 0).
    p.cols = 320;
    if p.hc_silu & 2 != 0 && p.rows2 == 0 { decode_input::<false, false, true, true, 0>(p); }
    else { decode_input::<false, false, true, false, 0>(p); }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_320_bf16(mut p: Bf16Rows) {
    // hc_silu bit 1: weight2 holds the matrix's LBF12 twin (one matrix, rows2 == 0).
    p.cols = 320;
    if p.hc_silu & 2 != 0 && p.rows2 == 0 { decode_input::<false, true, true, true, 0>(p); }
    else { decode_input::<false, true, true, false, 0>(p); }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_3072_native(mut p: Bf16Rows) {
    // hc_silu bit 1: weight2 holds the matrix's LBF12 twin (one matrix, rows2 == 0).
    p.cols = 3072;
    if p.hc_silu & 2 != 0 && p.rows2 == 0 { decode_input::<false, false, true, true, 0>(p); }
    else { decode_input::<false, false, true, false, 0>(p); }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_2560_native(mut p: Bf16Rows) {
    // hc_silu bit 1: weight2 holds the matrix's LBF12 twin (one matrix, rows2 == 0).
    p.cols = 2560;
    if p.hc_silu & 2 != 0 && p.rows2 == 0 { decode_input::<false, false, true, true, 0>(p); }
    else { decode_input::<false, false, true, false, 0>(p); }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_2560(mut p: Bf16Rows) { p.cols = 2560; decode::<false, 0>(p); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_10240(mut p: Bf16Rows) { p.cols = 10240; decode::<true, 0>(p); }

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_2560_narrow(mut p: Bf16Rows) { p.cols = 2560; decode::<true, 0>(p); }

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_2560_pairs(p: crate::kernel_params::Bf16Pairs) {
    let mut pair = if core::arch::nvptx::_block_idx_y() == 0 { p.pairs[0] } else { p.pairs[1] };
    pair.cols = 2560;
    if pair.tokens == 1 { decode_rows::<true, false, false, false, false, 0>(pair, true); }
    else { decode_rows::<true, false, false, true, false, 0>(pair, true); }
}

/// Tiny gate matrices use one CTA per output instead of leaving most SMs
/// idle. Repeated MMA columns read the same weight row; only column zero is kept.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma1_2560_small(mut p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let lane = tid % 32;
    let part = tid / 32 & 3;
    let mut row = _block_idx_x() as usize;
    if row >= (p.rows + p.rows2) as usize { return; }
    if row >= p.rows as usize {
        row -= p.rows as usize;
        p.weight = p.weight2; p.output = p.output2; p.rows = p.rows2; p.stride = p.stride2;
    }
    p.rows2 = 0; p.cols = 2560; p.tokens = 1;
    let sum = if flash_bf16_packed(p.rows as usize, 2560) { dot_inner::<true, false, true, false, true, false, 0>(p, row, lane, part) }
        else { dot_inner::<false, false, true, false, true, false, 0>(p, row, lane, part) };
    let sh = ptx::shared::<16>();
    if lane == 0 { ptx::store(sh + (part * 4) as u32, sum.a); }
    ptx::sync();
    if tid == 0 {
        put(p, (ptx::load(sh) + ptx::load(sh + 4)) + (ptx::load(sh + 8) + ptx::load(sh + 12)), 0, row);
    }
}

#[inline(always)]
unsafe fn decode<const DEEP: bool, const RANGES: u32>(p: Bf16Rows) {
    if p.input_bf16 != 0 { decode_input::<DEEP, true, false, false, RANGES>(p); }
    else { decode_input::<DEEP, false, false, false, RANGES>(p); }
}

#[inline(always)]
unsafe fn decode_input<const DEEP: bool, const INPUT_BF16: bool, const NATIVE: bool, const LBF: bool, const RANGES: u32>(p: Bf16Rows) {
    let split = if NATIVE { p.cols == 320 || LBF }
        else { flash_bf16_decode_split((p.rows + p.rows2) as usize, p.cols as usize) };
    if p.tokens == 1 { decode_rows::<DEEP, INPUT_BF16, NATIVE, false, LBF, RANGES>(p, split); }
    else { decode_rows::<DEEP, INPUT_BF16, NATIVE, true, LBF, RANGES>(p, split); }
}

#[inline(always)]
unsafe fn decode_rows<const DEEP: bool, const INPUT_BF16: bool, const NATIVE: bool, const BATCH: bool, const LBF: bool, const RANGES: u32>(p: Bf16Rows, split: bool) {
    let tid = _thread_idx_x() as usize;
    let lane = tid % 32;
    // A two-range shard has no parts 1 and 3: a split block runs two row tiles,
    // warps 0-1 parts 0 and 2 of the first, warps 2-3 of the second.
    let pair = split && RANGES == 2;
    let first = (p.rows as usize).div_ceil(8);
    let tiles = first + (p.rows2 as usize).div_ceil(8);
    let tile = if pair { (_block_idx_x() as usize * 2 + tid / 64).min(tiles - 1) }
        else if split { _block_idx_x() as usize }
        else { _block_idx_x() as usize * 2 + tid / 32 };
    if tile >= tiles { return; }
    // Keep every warp aligned to its original matrix, including a four-row
    // matrix fused ahead of a large projection.
    let mut p = p;
    if !BATCH { p.tokens = 1; }
    let whole = p;
    let row = if tile < first { tile * 8 } else {
        p.weight = p.weight2; p.output = p.output2; p.rows = p.rows2; p.stride = p.stride2;
        (tile - first) * 8
    };
    p.rows2 = 0;
    let dot = |part| if NATIVE { dot_inner::<true, true, DEEP, INPUT_BF16, false, LBF, RANGES>(p, row, lane, part) }
        else { dot_range::<DEEP, INPUT_BF16, RANGES>(p, row, lane, part) };
    if pair {
        let sum = dot((tid / 32 & 1) * 2);
        let sh = if BATCH { ptx::shared::<1024>() } else { ptx::shared::<128>() };
        let part_rows = if BATCH { 64 } else { 8 };
        let token = if p.tokens == 1 { 0 } else { lane / 4 };
        if (p.tokens != 1 || lane < 4) && token < p.tokens as usize {
            let at = tid / 32 * part_rows + token * 8 + lane % 4 * 2;
            ptx::store(sh + (at * 4) as u32, sum.a);
            ptx::store(sh + ((at + 1) * 4) as u32, sum.b);
        }
        ptx::sync();
        let n = p.tokens as usize * 8;
        let (sub, i) = (tid / n, tid % n);
        let tile = _block_idx_x() as usize * 2 + sub;
        if sub < 2 && tile < tiles {
            let at = sh + ((sub * 2 * part_rows + i) * 4) as u32;
            // The four-part merge with -0.0 parts 1 and 3: (l0 + -0.0) + (l2 + -0.0).
            let value = ptx::load(at) + ptx::load(at + (part_rows * 4) as u32);
            let mut q = whole;
            let row = if tile < first { tile * 8 } else {
                q.weight = q.weight2; q.output = q.output2; q.rows = q.rows2; q.stride = q.stride2;
                (tile - first) * 8
            };
            q.rows2 = 0;
            if q.cols == 10240 && q.hc_silu != 0 && q.rows == 320 {
                let value = crate::flash_ops::silu(value, 1.0, 4.0, 0);
                let (hi, lo) = crate::gemm::split2(value);
                *(q.output as *mut u16).add(row + i) = hi;
                *(q.output as *mut u16).add(320 + row + i) = lo;
            } else { put(q, value, i / 8, row + i % 8); }
        }
        return;
    }
    if split {
        let part = tid / 32 & 3;
        let sum = dot(part);
        let sh = if BATCH { ptx::shared::<1024>() } else { ptx::shared::<128>() };
        let part_rows = if BATCH { 64 } else { 8 };
        let token = if p.tokens == 1 { 0 } else { lane / 4 };
        if (p.tokens != 1 || lane < 4) && token < p.tokens as usize {
            let at = part * part_rows + token * 8 + lane % 4 * 2;
            ptx::store(sh + (at * 4) as u32, sum.a);
            ptx::store(sh + ((at + 1) * 4) as u32, sum.b);
        }
        ptx::sync();
        if tid < p.tokens as usize * 8 {
            let at = sh + (tid * 4) as u32;
            let step = (part_rows * 4) as u32;
            let value = (ptx::load(at) + ptx::load(at + step)) + (ptx::load(at + 2 * step) + ptx::load(at + 3 * step));
            if p.cols == 10240 && p.hc_silu != 0 && p.rows == 320 {
                let value = crate::flash_ops::silu(value, 1.0, 4.0, 0);
                let (hi, lo) = crate::gemm::split2(value);
                *(p.output as *mut u16).add(row + tid) = hi;
                *(p.output as *mut u16).add(320 + row + tid) = lo;
            } else { put(p, value, tid / 8, row + tid % 8); }
        }
        return;
    }
    let a = dot(0);
    let b = dot(1);
    let c = dot(2);
    let d = dot(3);
    let sum = add(add(a, b), add(c, d));
    if p.tokens != 1 || lane < 4 {
        let token = if p.tokens == 1 { 0 } else { lane / 4 };
        put(p, sum.a, token, row + lane % 4 * 2);
        put(p, sum.b, token, row + lane % 4 * 2 + 1);
    }
}

/// Stage packed BF16 weights and A fragments asynchronously. Edge fragments
/// are zero, never read past K or the padded token extent.
#[inline(always)]
unsafe fn stage<const WIDE: bool>(p: Bf16Rows, sh: u32, row: usize, token: usize, col: usize, tid: usize) {
    let k = p.cols as usize;
    let threads = if WIDE { 256 } else { 128 };
    let x_offset = (if WIDE { 64 } else { 32 }) * 128;
    for step in 0..2 {
        let i = tid + step * threads;
        let (group, tile, lane) = (i / 32, i % 32 / 8, i % 8 * 2);
        let (r, c) = (group * 4 + lane / 4, tile * 16 + lane % 4 * 2);
        let dst = sh + (i * 16) as u32;
        if row + r < (p.rows + p.rows2) as usize && col + c < k {
            let (w, _, local, _) = target(p, row + r);
            if packed(p, row + r) {
                ptx::copy16(dst, w.add(flash_bf16_element(local, col + c, k, matrix_rows(p, row + r))).cast());
            } else {
                let a = weight_fragment(p, row + r, col + c);
                let b = weight_fragment(p, row + r, col + c + 2);
                ptx::store(dst, f32::from_bits(a[0]));
                ptx::store(dst + 4, f32::from_bits(a[1]));
                ptx::store(dst + 8, f32::from_bits(b[0]));
                ptx::store(dst + 12, f32::from_bits(b[1]));
            }
        } else {
            for j in 0..4 { ptx::store(dst + j * 4, 0.0); }
        }
    }
    for step in 0..if WIDE { 2 } else { 1 } {
        let i = tid + step * threads;
        let (t, c) = (token / 16 + i / 128, col / 16 + i % 128 / 32);
        for plane in 0..TERMS {
            let dst = sh + (x_offset + plane * (if WIDE { 64 } else { 16 }) * 128 + i * 16) as u32;
            if t * 16 < p.tokens as usize && c < k.div_ceil(16) {
                ptx::copy16(dst, (p.input as *const u32).add(plane * p.input_plane_words as usize
                    + ((t * k.div_ceil(16) + c) * 32 + i % 32) * 4));
            } else {
                for j in 0..4 { ptx::store(dst + j * 4, 0.0); }
            }
        }
    }
    ptx::commit();
}

#[derive(Clone, Copy)]
struct Tile { a: Acc, b: Acc, c: Acc, d: Acc }
const TILE_ZERO: Tile = Tile { a: ZERO, b: ZERO, c: ZERO, d: ZERO };
#[inline(always)]
fn plus(a: Tile, b: Tile) -> Tile {
    Tile { a: add(a.a, b.a), b: add(a.b, b.b), c: add(a.c, b.c), d: add(a.d, b.d) }
}
#[inline(always)]
unsafe fn a_fragment(at: u32) -> [u32; 4] { ptx::load4(at) }

#[inline(always)]
unsafe fn b_fragment(at: u32) -> [u32; 2] {
    ptx::load2(at)
}

/// Both tile shapes stage resident fragments directly; every output executes
/// the same K=16 instructions and balanced range merge as single-row decode.
#[inline(always)]
unsafe fn project<const WIDE: bool, const RANGES: u32>(p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (m, n) = if WIDE { (64, 64) } else { (16, 32) };
    let (wm, wn) = if WIDE { (warp / 4 * 32, warp % 4 * 16) } else { (0, warp * 8) };
    let x_offset = n * 128;
    let stage_bytes = x_offset + TERMS * m * 128;
    let tiles = (p.tokens as usize).div_ceil(m);
    let block = _block_idx_x() as usize;
    let (row, token) = (block / tiles * n, block % tiles * m);
    let sh = if WIDE { 0 } else { ptx::shared::<16384>() };
    // Uneven rank shards keep row-major weights and zero the final K tile.
    let kt = (p.cols as usize).div_ceil(16);
    let (mut left, mut right) = (TILE_ZERO, TILE_ZERO);
    let (mut sum, mut low, mut part) = (TILE_ZERO, TILE_ZERO, 0);
    stage::<WIDE>(p, sh, row, token, 0, tid);
    ptx::wait();
    ptx::sync();
    let mut tile = 0;
    while tile < kt {
        let buffer = tile / 4 % 2;
        let current = sh + (buffer * stage_bytes) as u32;
        if tile + 4 < kt {
            stage::<WIDE>(p, sh + ((1 - buffer) * stage_bytes) as u32, row, token, (tile + 4) * 16, tid);
        }
        for z in 0..4 {
            if tile + z < kt {
                let a = current + (x_offset + (wm / 16 * 4 + z) * 512 + lane * 16) as u32;
                let b = current + (((wn / 4 + lane / 16) * 4 + z) * 16 + lane % 16) as u32 * 8;
                let (a0, a0_lo, b0) = (a_fragment(a), a_fragment(a + (m * 128) as u32), b_fragment(b));
                sum.a = madd(a0, b0, sum.a);
                low.a = madd(a0_lo, b0, low.a);
                if WIDE {
                    let (a1, a1_lo, b1) = (a_fragment(a + 2048),
                        a_fragment(a + 2048 + (m * 128) as u32), b_fragment(b + 1024));
                    sum.b = madd(a0, b1, sum.b);
                    low.b = madd(a0_lo, b1, low.b);
                    sum.c = madd(a1, b0, sum.c);
                    low.c = madd(a1_lo, b0, low.c);
                    sum.d = madd(a1, b1, sum.d);
                    low.d = madd(a1_lo, b1, low.d);
                }
                if tile + z + 1 == canonical::end(kt, part, RANGES) {
                    let partial = plus(sum, low);
                    match part {
                        0 => left = partial,
                        1 => left = plus(left, partial),
                        2 => right = partial,
                        _ => right = plus(right, partial),
                    }
                    sum = TILE_ZERO;
                    low = TILE_ZERO;
                    // Parts outside the shard add -0.0, so skipping them is exact.
                    part = canonical::next(part, RANGES);
                }
            }
        }
        ptx::wait();
        ptx::sync();
        tile += 4;
    }
    let result = plus(left, right);
    let (t, r) = (token + wm + lane / 4, row + wn + lane % 4 * 2);
    write(p, result.a, t, r);
    if WIDE {
        write(p, result.b, t, r + 8);
        write(p, result.c, t + 16, r);
        write(p, result.d, t + 16, r + 8);
    }
}

/// Block128, grid ceil(N/32)*ceil(T/16), shared16,384B. A four-rank K-split
/// shard runs its own two-range copy; four ranges run the code without ranges.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma(p: Bf16Rows) {
    let (cols, ranges) = crate::kernel_params::bf16_cols_ranges(p.cols);
    if ranges == 2 { project::<false, 2>(Bf16Rows { cols, ..p }); } else { project::<false, 0>(p); }
}

/// Block256, grid ceil(N/64)*ceil(T/64), dynamic shared49,152B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma64(p: Bf16Rows) {
    let (cols, ranges) = crate::kernel_params::bf16_cols_ranges(p.cols);
    if ranges == 2 { project::<true, 2>(Bf16Rows { cols, ..p }); } else { project::<true, 0>(p); }
}

#[path = "flash_prefill.rs"]
mod prefill;

#[path = "dense_latency.rs"]
mod dense_latency;

/// HC option A: canonical K16 up projection and stream-ordered mix in one
/// epilogue for every row count. No gate matrix round trip or M-dependent sum.
/// Grid ceil(H/8),ceil(M/16), block32; normalized is FP32 [M,4*H].
/// # Safety
/// H=2560 and K=320. Both packed input planes cover complete padded M16 tiles;
/// weights use the canonical 4H/K layout. Normalized/output cover M*4H/M*H
/// finite FP32 values respectively. Each N8 tile is owned by exactly one warp.
#[path = "hc_up.rs"]
mod hc_lookahead;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_latency(p: Bf16Rows, normalized: u64) {
    let tid = _thread_idx_x() as usize;
    let lane = tid % 32;
    let (h, k) = (2560usize, 320usize);
    let feature = _block_idx_x() as usize * 8 + tid / 32 * 8;
    let mt = _block_idx_y() as usize * 16;
    let mut mixed = ZERO;
    for stream in 0..4 {
        let row=stream*h+feature;
        let (mut left,mut right)=(ZERO,ZERO);
        for part in 0..4 {
            let (mut hi, mut lo) = (ZERO, ZERO);
            let (start, _) = canonical::range(k / 16, part);
            let load = |kt| {
                let at = (mt / 16 * (k / 16) * 32 + kt * 32 + lane) * 4;
                let x = p.input as *const u32;
                let a = (*(x.add(at).cast::<HcFragment>())).0;
                let low = (*(x
                    .add(at + p.input_plane_words as usize)
                    .cast::<HcFragment>()))
                .0;
                let b = {
                    let at = flash_bf16_element(row + lane / 4, kt * 16 + lane % 4 * 2, k, 4 * h);
                    let value: u64;
                    core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) value,
                        p=in(reg64) (p.weight as *const u16).add(at));
                    [value as u32, (value >> 32) as u32]
                };
                (a, low, b)
            };
            let mut apply = |(a, low, b)| {
                hi = madd(a, b, hi);
                lo = madd(low, b, lo);
            };
            hc_lookahead::five(load, &mut apply, start);
            let z=add(hi,lo);
            match part { 0=>left=z,1=>left=add(left,z),2=>right=z,_=>right=add(right,z) }
        }
        let gate=add(left,right);
        let t=mt+lane/4; let f=feature+lane%4*2;
        let sig=|v: f32| crate::flash_next::hc_sigmoid(v);
        if t<p.tokens as usize {
            let x=(normalized as *const f32).add(t*4*h+stream*h+f);
            mixed.a=crate::next_math::hc_stream(mixed.a,*x,gate.a,sig); mixed.b=crate::next_math::hc_stream(mixed.b,*x.add(1),gate.b,sig);
        }
        if t+8<p.tokens as usize {
            let x=(normalized as *const f32).add((t+8)*4*h+stream*h+f);
            mixed.c=crate::next_math::hc_stream(mixed.c,*x,gate.c,sig); mixed.d=crate::next_math::hc_stream(mixed.d,*x.add(1),gate.d,sig);
        }
    }
    let t=mt+lane/4; let f=feature+lane%4*2;
    if t<p.tokens as usize { *(p.output as *mut f32).add(t*h+f)=mixed.a*0.25;
        *(p.output as *mut f32).add(t*h+f+1)=mixed.b*0.25; }
    if t+8<p.tokens as usize { *(p.output as *mut f32).add((t+8)*h+f)=mixed.c*0.25;
        *(p.output as *mut f32).add((t+8)*h+f+1)=mixed.d*0.25; }
}

/// One stream's HC up gate for eight features: the canonical K16 walk of
/// flash_next_hc_up_latency (four ranges of five tiles, hi/lo chains, (p0+p1)+(p2+p3)).
#[inline(always)]
unsafe fn hc_up_gate(p: Bf16Rows, stream: usize, feature: usize, mt: usize, lane: usize) -> Acc {
    let (h, k) = (2560usize, 320usize);
    let row = stream * h + feature;
    let (mut left, mut right) = (ZERO, ZERO);
    for part in 0..4 {
        let (mut hi, mut lo) = (ZERO, ZERO);
        let (start, _) = canonical::range(k / 16, part);
        let load = |kt| {
            let at = (mt / 16 * (k / 16) * 32 + kt * 32 + lane) * 4;
            let x = p.input as *const u32;
            let a = (*(x.add(at).cast::<HcFragment>())).0;
            let low = (*(x.add(at + p.input_plane_words as usize).cast::<HcFragment>())).0;
            let b = {
                let at = flash_bf16_element(row + lane / 4, kt * 16 + lane % 4 * 2, k, 4 * h);
                let value: u64;
                core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) value,
                    p=in(reg64) (p.weight as *const u16).add(at));
                [value as u32, (value >> 32) as u32]
            };
            (a, low, b)
        };
        let mut apply = |(a, low, b)| {
            hi = madd(a, b, hi);
            lo = madd(low, b, lo);
        };
        hc_lookahead::five(load, &mut apply, start);
        let z = add(hi, lo);
        match part { 0 => left = z, 1 => left = add(left, z), 2 => right = z, _ => right = add(right, z) }
    }
    add(left, right)
}

/// hc_up_gate at M1: all 20 B fragments and token 0's A words issued before the first
/// MMA (one DRAM round trip per warp instead of one per five-tile range), then each range's
/// five tiles in the transposed form (weights in A, token hi/lo in B columns 0/1; bit-exact,
/// see dense_latency::tmadd), so every chain, range sum and merge is hc_up_gate's.
#[inline(always)]
unsafe fn hc_up_gate_m1(p: Bf16Rows, stream: usize, feature: usize, lane: usize) -> Acc {
    let (h, k) = (2560usize, 320usize);
    let row = stream * h + feature;
    let x = p.input as *const u32;
    // Lanes 0-3: the hi plane's a0/a2 words of token 0; lanes 4-7: the lo plane's.
    let plane = if lane < 4 { 0 } else { p.input_plane_words as usize };
    let load = |kt: usize| {
        let at = flash_bf16_element(row + lane / 4, kt * 16 + lane % 4 * 2, k, 4 * h);
        let value: u64;
        core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) value,
            p=in(reg64) (p.weight as *const u16).add(at));
        let a = (kt * 32 + lane % 4) * 4 + plane;
        let t = if lane < 8 { [*x.add(a), 0, *x.add(a + 2), 0] } else { [0; 4] };
        ([value as u32, (value >> 32) as u32], t)
    };
    // Ranges of five tiles (canonical::range(20, part) = 5 * part), literal indices so the
    // twenty fragments stay in registers.
    macro_rules! tiles { ($($i:literal)*) => { [$(load($i)),*] } }
    let v = tiles!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19);
    let range = |s: usize| {
        let mut c = ZERO;
        for i in 0..5 {
            c = dense_latency::tmadd(v[s + i].0, v[s + i].1, c);
        }
        dense_latency::untranspose(c, lane)
    };
    let (z0, z1, z2, z3) = (range(0), range(5), range(10), range(15));
    add(add(z0, z1), add(z2, z3))
}

/// flash_next_hc_up_latency with one warp per stream: block128, grid ceil(H/8),ceil(M/16).
/// Warps 1..3 park their gates in shared memory and warp 0 mixes streams 0, 1, 2, 3 in
/// the solo kernel's order, so every output float is the solo kernel's.
/// # Safety
/// As flash_next_hc_up_latency.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_latency4(p: Bf16Rows, normalized: u64) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let h = 2560usize;
    let feature = _block_idx_x() as usize * 8;
    let mt = _block_idx_y() as usize * 16;
    let gate = if p.tokens == 1 { hc_up_gate_m1(p, warp, feature, lane) } else { hc_up_gate(p, warp, feature, mt, lane) };
    let sh = ptx::shared::<2048>();
    if warp > 0 {
        let at = sh + (((warp - 1) * 32 + lane) * 16) as u32;
        ptx::store(at, gate.a); ptx::store(at + 4, gate.b); ptx::store(at + 8, gate.c); ptx::store(at + 12, gate.d);
    }
    ptx::sync();
    if warp != 0 { return; }
    let t = mt + lane / 4; let f = feature + lane % 4 * 2;
    let sig = |v: f32| crate::flash_next::hc_sigmoid(v);
    let mut mixed = ZERO;
    for stream in 0..4 {
        let gate = if stream == 0 { gate } else {
            let at = sh + (((stream - 1) * 32 + lane) * 16) as u32;
            Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
        };
        if t < p.tokens as usize {
            let x = (normalized as *const f32).add(t * 4 * h + stream * h + f);
            mixed.a = crate::next_math::hc_stream(mixed.a, *x, gate.a, sig); mixed.b = crate::next_math::hc_stream(mixed.b, *x.add(1), gate.b, sig);
        }
        if t + 8 < p.tokens as usize {
            let x = (normalized as *const f32).add((t + 8) * 4 * h + stream * h + f);
            mixed.c = crate::next_math::hc_stream(mixed.c, *x, gate.c, sig); mixed.d = crate::next_math::hc_stream(mixed.d, *x.add(1), gate.d, sig);
        }
    }
    if t < p.tokens as usize { *(p.output as *mut f32).add(t * h + f) = mixed.a * 0.25;
        *(p.output as *mut f32).add(t * h + f + 1) = mixed.b * 0.25; }
    if t + 8 < p.tokens as usize { *(p.output as *mut f32).add((t + 8) * h + f) = mixed.c * 0.25;
        *(p.output as *mut f32).add((t + 8) * h + f + 1) = mixed.d * 0.25; }
}

#[repr(C, align(16))]
struct HcFragment([u32;4]);

#[path = "flash_next_dense.rs"]
mod next_dense;


/// flash_bf16_mma_latency for BF16 M1 solo launches with N <= 64 (kernel_params::latency_fold1) in one 128-thread launch: matrices up to 1 MiB are first
/// pulled toward L2 by the whole grid (one prefetch per 128-byte line), then fold1 (one CTA
/// per row group: a CTA input copy, K64 fragment rings, the CTA merge; dense_latency_fold.rs).
/// The entry lives here, not in dense_latency.rs: there its size reorders rustc's
/// codegen-unit merge and two unowned entries' code changes (no drift here).
/// Same K walk, chains and merge as flash_bf16_mma_latency + flash_bf16_mma_latency_merge.
/// # Safety
/// Launch 192 CTAs x 128 threads with latency_fold1(M, K, N) true, BF16 weight, rows2=0;
/// kernel_params::latency_fold_shared dynamic shared bytes; partials unused.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_latency_fold(p: Bf16Rows, _partials: u64) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};
    let bytes = p.rows as usize * p.cols as usize * 2;
    if bytes <= 1 << 20 {
        let mut at = (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize * 128;
        while at < bytes {
            core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) (p.weight as usize + at) as u64);
            at += (_grid_dim_x() * _block_dim_x()) as usize * 128;
        }
    }
    dense_latency::fold1(p);
}

// ---- LBF12 twins of the HC up weights : an M1-only copy of
// flash_next_hc_up_latency4 whose gate reads the twin, and the twin builder. Private hcu_
// helpers and plain loops, alongside flash_next_hc_up_latency4.
#[inline(always)]
unsafe fn hcu_shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX, v, d, 31) }
/// One lane's LBF12 slot words (lane * 12 of the slot) and its row base -> the two K16 BF16
/// fragments the raw slot holds (sign | base + offset | mantissa; offset 15 = signed zero).
#[inline(always)]
unsafe fn hcu_decode(slot: *const u8, base: u32) -> [u32; 4] {
    let w = slot as *const u32;
    let lo = *w as u64 | (*w.add(1) as u64) << 32;
    let hi = *w.add(2) as u64;
    let f = |j: u32| -> u32 { (if j < 5 { lo >> (12 * j) } else if j == 5 { lo >> 60 | hi << 4 } else { hi >> (12 * j - 64) }) as u32 & 0xFFF };
    let v = |x: u32| -> u32 { let off = (x >> 7) & 15; (x >> 11) << 15 | if off == 15 { 0 } else { (base + off) << 7 | (x & 0x7F) } };
    [v(f(0)) | v(f(1)) << 16, v(f(2)) | v(f(3)) << 16, v(f(4)) | v(f(5)) << 16, v(f(6)) | v(f(7)) << 16]
}
/// hc_up_gate_m1 reading the LBF12 twin of the [4H][K] up weight: a packed K32 slot decodes
/// to the two K16 fragments the BF16 loads give, a raw slot loads them from the BF16 matrix;
/// same tiles, ranges and merges, so the same bits.
#[inline(always)]
unsafe fn hcu_gate_m1(p: Bf16Rows, twin: u64, stream: usize, feature: usize, lane: usize) -> Acc {
    let (h, k) = (2560usize, 320usize);
    let row = stream * h + feature;
    let x = p.input as *const u32;
    let plane = if lane < 4 { 0 } else { p.input_plane_words as usize };
    let (pairs, slots) = (k / 32, 4 * h / 8 * (k / 32));
    let slot0 = row / 8 * pairs;
    let data = twin as *const u8;
    let flags = *(data.add(slots * 392 + row / 8 * 4) as *const u32);
    let input = |kt: usize| {
        let a = (kt * 32 + lane % 4) * 4 + plane;
        if lane < 8 { [*x.add(a), 0, *x.add(a + 2), 0] } else { [0; 4] }
    };
    let load = |j: usize| -> [u32; 4] {
        if flags >> j & 1 != 0 {
            hcu_decode(data.add((slot0 + j) * 384 + lane * 12), *data.add(slots * 384 + (slot0 + j) * 8 + lane / 4) as u32)
        } else {
            let element = |t: usize| (p.weight as *const u16).add(flash_bf16_element(row + lane / 4, t * 16 + lane % 4 * 2, k, 4 * h));
            let (a, b): (u64, u64);
            core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) a, p=in(reg64) element(2 * j));
            core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) b, p=in(reg64) element(2 * j + 1));
            [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]
        }
    };
    let v = [load(0), load(1), load(2), load(3), load(4), load(5), load(6), load(7), load(8), load(9)];
    let frag = |t: usize| -> [u32; 2] { let w = v[t / 2]; if t % 2 == 0 { [w[0], w[1]] } else { [w[2], w[3]] } };
    let range = |s: usize| {
        let mut c = ZERO;
        let mut i = 0;
        while i < 5 {
            c = dense_latency::tmadd(frag(s + i), input(s + i), c);
            i += 1;
        }
        dense_latency::untranspose(c, lane)
    };
    let (z0, z1, z2, z3) = (range(0), range(5), range(10), range(15));
    add(add(z0, z1), add(z2, z3))
}
/// HC up's or HC down's epilogue: output row t's elements f, f + 1 (`a`, `b`) also land in the
/// next projection's packed operand, the hi/lo words flash_bf16_input forms from those rows
/// (K, M rows, split2_pair, same layout); row t + 8 of a 16-row tile past M gets its 0 word,
/// as the pack's; other rows past M stay unwritten (MMA rows are independent, outputs masked).
/// A macro, so each expansion sits under its kernel's one safety contract: planes covers the two
/// dense planes of `rows` x K. Row t and elements f, f + 1 name one word pair, written by the one
/// lane that owns them.
macro_rules! hc_planes { ($k:literal, $planes:expr, $rows:expr, $t:expr, $f:expr, $a:expr, $b:expr) => {{
    let (rows, t): (usize, usize) = ($rows, $t);
    let (plane, word) = hc_plane_word::<$k>(rows, t, $f);
    let (hi, lo) = crate::gemm::split2_pair($a, $b);
    let p = $planes as *mut u32;
    *p.add(word) = hi;
    *p.add(plane + word) = lo;
    if t % 16 < 8 && t + 8 >= rows { *p.add(word + 1) = 0; *p.add(plane + word + 1) = 0; }
}} }
pub(crate) use hc_planes;
/// One plane's words and the word of row t's elements f, f + 1 in the packed operand of `rows` x K.
#[inline(always)]
fn hc_plane_word<const K: usize>(rows: usize, t: usize, f: usize) -> (usize, usize) {
    let kt = K / 16;
    (rows.div_ceil(16) * kt * 128, ((t / 16 * kt + f / 16) * 32 + t % 8 * 4 + f % 8 / 2) * 4 + t % 16 / 8 + f % 16 / 8 * 2)
}
/// flash_next_hc_up_latency4 at M1 with its gate read from the up weight's LBF12 twin
/// (flash_bf16_lbf12_twin): same gate bits, same stream mix in the same order. planes != 0:
/// hc_planes writes the next projections' operand.
/// # Safety
/// As flash_next_hc_up_latency4 with M = 1 (grid H/8 x 1), plus twin = the up weight's
/// flash_bf16_lbf12_twin image, and planes = 0 or the two dense planes of M x H.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_latency4_lbf12(p: Bf16Rows, normalized: u64, twin: u64, planes: u64) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let h = 2560usize;
    let feature = _block_idx_x() as usize * 8;
    let gate = hcu_gate_m1(p, twin, warp, feature, lane);
    let sh = ptx::shared::<2048>();
    if warp > 0 {
        let at = sh + (((warp - 1) * 32 + lane) * 16) as u32;
        ptx::store(at, gate.a); ptx::store(at + 4, gate.b); ptx::store(at + 8, gate.c); ptx::store(at + 12, gate.d);
    }
    ptx::sync();
    if warp != 0 { return; }
    let t = lane / 4; let f = feature + lane % 4 * 2;
    let sig = |v: f32| crate::flash_next::hc_sigmoid(v);
    let mut mixed = ZERO;
    let mut stream = 0;
    while stream < 4 {
        let gate = if stream == 0 { gate } else {
            let at = sh + (((stream - 1) * 32 + lane) * 16) as u32;
            Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
        };
        if t < p.tokens as usize {
            let x = (normalized as *const f32).add(t * 4 * h + stream * h + f);
            mixed.a = crate::next_math::hc_stream(mixed.a, *x, gate.a, sig); mixed.b = crate::next_math::hc_stream(mixed.b, *x.add(1), gate.b, sig);
        }
        stream += 1;
    }
    if t < p.tokens as usize {
        let (a, b) = (mixed.a * 0.25, mixed.b * 0.25);
        *(p.output as *mut f32).add(t * h + f) = a;
        *(p.output as *mut f32).add(t * h + f + 1) = b;
        if planes != 0 { hc_planes!(2560, planes, p.tokens as usize, t, f, a, b); }
    }
}
#[inline(always)]
unsafe fn twin_shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX, v, d, 31) }
/// LBF12 twin of a packed BF16 matrix [N][K] (flash_bf16_element order; N % 64 == 0,
/// K % 32 == 0, K <= 4096): one warp per 8-row x K32 slot reads the lanes' two K16 fragments
/// (as the latency kernels load them); if every row's nonzero exponents span <= 14 and none
/// is subnormal (a warp vote), it writes twin[slot * 384 + lane * 12], the row bases at
/// twin + slots * 384 and the slot's bit in ceil(K / 1024) u32 flag words per 8-row group at
/// twin + slots * 392 (zeroed by the host); other slots are read from the BF16 matrix.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_lbf12_twin(p: crate::kernel_params::FlashBf16Pack) {
    let i = core::arch::nvptx::_block_idx_x() as usize * core::arch::nvptx::_block_dim_x() as usize + core::arch::nvptx::_thread_idx_x() as usize;
    let (slot, lane) = (i / 32, i % 32);
    let (n, k) = (p.rows as usize, p.cols as usize);
    let (pairs, slots) = (k / 32, n / 8 * (k / 32));
    if slot >= slots { return; }
    let (row, kt) = (slot / pairs * 8 + lane / 4, slot % pairs * 2);
    let element = |t: usize| (p.source as *const u16).add(crate::kernel_params::flash_bf16_element(row, t * 16 + lane % 4 * 2, k, n)) as *const u64;
    let (f0, f1) = (*element(kt), *element(kt + 1));
    let v = [f0 as u32, (f0 >> 32) as u32, f1 as u32, (f1 >> 32) as u32];
    let h = |j: usize| (v[j / 2] >> (16 * (j % 2))) & 0xFFFF;
    let (mut lo, mut hi, mut bad) = (255u32, 0u32, 0u32);
    let mut j = 0;
    while j < 8 { let x = h(j); if x & 0x7FFF != 0 { let e = (x >> 7) & 0xFF; bad |= u32::from(e == 0);
        if e < lo { lo = e; } if e > hi { hi = e; } } j += 1; }
    let mut d = 1;
    while d <= 2 { let (l, u) = (twin_shuffle(lo, d), twin_shuffle(hi, d)); if l < lo { lo = l; } if u > hi { hi = u; }
        bad |= twin_shuffle(bad, d); d += 1; }
    let ok = u32::from(bad == 0 && (hi < lo || hi - lo <= 14));
    let all: u32;
    core::arch::asm!("{{ .reg .pred p, q; setp.ne.u32 p, {o}, 0; vote.sync.all.pred q, p, -1; selp.u32 {a}, 1, 0, q; }}", o = in(reg32) ok, a = out(reg32) all);
    if all == 0 { return; }
    let base = if hi < lo { 0 } else { lo };
    let f = |j: usize| -> u64 { let x = h(j); let off = if x & 0x7FFF == 0 { 15 } else { ((x >> 7) & 0xFF) - base }; ((x >> 15) << 11 | off << 7 | (x & 0x7F)) as u64 };
    let lo64 = f(0) | f(1) << 12 | f(2) << 24 | f(3) << 36 | f(4) << 48 | f(5) << 60;
    let hi64 = f(5) >> 4 | f(6) << 8 | f(7) << 20;
    let twin = p.weight as *mut u8;
    let w = twin.add(slot * 384 + lane * 12) as *mut u32;
    *w = lo64 as u32; *w.add(1) = (lo64 >> 32) as u32; *w.add(2) = hi64 as u32;
    if lane % 4 == 0 { *twin.add(slots * 384 + slot * 8 + lane / 4) = base as u8; }
    if lane == 0 {
        core::arch::asm!("red.global.or.b32 [{a}], {v};", a = in(reg64) twin.add(slots * 392 + (slot / pairs * pairs.div_ceil(32) + slot % pairs / 32) * 4) as u64, v = in(reg32) 1u32 << (slot % pairs % 32));
    }
}

// HC up at M2-16: flash_next_hc_up_latency4 with its gate read from the up weight's
// LBF12 twin (flash_bf16_lbf12_twin). Private hc16_ copies of every helper except the shared
// dense_latency::tmadd and dense_latency::untranspose, so no existing entry's codegen changes.
#[cfg(target_arch = "nvptx64")]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn hc16_mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32,
        c0: f32, c1: f32, c2: f32, c3: f32) -> crate::canonical::Acc;
}
#[cfg(target_arch = "nvptx64")]
#[repr(C, align(16))]
struct Hc16Fragment([u32; 4]);
/// canonical::madd and canonical::add (same intrinsic, same adds).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hc16_madd(a: [u32; 4], b: [u32; 2], c: crate::canonical::Acc) -> crate::canonical::Acc {
    hc16_mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d)
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn hc16_add(a: crate::canonical::Acc, b: crate::canonical::Acc) -> crate::canonical::Acc {
    crate::canonical::Acc { a: a.a + b.a, b: a.b + b.b, c: a.c + b.c, d: a.d + b.d }
}
/// flash_bf16.rs hcu_decode: one lane's LBF12 slot words and row base -> the two K16 BF16
/// fragments the raw slot holds (sign | base + offset | mantissa; offset 15 = signed zero).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hc16_decode(slot: *const u8, base: u32) -> [u32; 4] {
    let w = slot as *const u32;
    let lo = *w as u64 | (*w.add(1) as u64) << 32;
    let hi = *w.add(2) as u64;
    let f = |j: u32| -> u32 { (if j < 5 { lo >> (12 * j) } else if j == 5 { lo >> 60 | hi << 4 } else { hi >> (12 * j - 64) }) as u32 & 0xFFF };
    let v = |x: u32| -> u32 { let off = (x >> 7) & 15; (x >> 11) << 15 | if off == 15 { 0 } else { (base + off) << 7 | (x & 0x7F) } };
    [v(f(0)) | v(f(1)) << 16, v(f(2)) | v(f(3)) << 16, v(f(4)) | v(f(5)) << 16, v(f(6)) | v(f(7)) << 16]
}
/// hc_up_gate (flash_bf16.rs) for M2-16, one M tile, reading the LBF12 twin of the [4H][K]
/// up weight: the twenty K16 weight fragments come from the twin as in hcu_gate_m1 (a packed
/// slot decodes to the BF16 loads' fragments, a raw slot loads them), then hc_up_gate's hi/lo
/// chains per five-tile range (canonical::range(20, part) = 5 * part) and its (p0+p1)+(p2+p3)
/// merge, so the same bits as flash_next_hc_up_latency4.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hc16_gate(p: crate::kernel_params::Bf16Rows, twin: u64, stream: usize, feature: usize, lane: usize) -> crate::canonical::Acc {
    let (h, k) = (2560usize, 320usize);
    let row = stream * h + feature;
    let x = p.input as *const u32;
    let (pairs, slots) = (k / 32, 4 * h / 8 * (k / 32));
    let slot0 = row / 8 * pairs;
    let data = twin as *const u8;
    let flags = *(data.add(slots * 392 + row / 8 * 4) as *const u32);
    let load = |j: usize| -> [u32; 4] {
        if flags >> j & 1 != 0 {
            hc16_decode(data.add((slot0 + j) * 384 + lane * 12), *data.add(slots * 384 + (slot0 + j) * 8 + lane / 4) as u32)
        } else {
            let element = |t: usize| (p.weight as *const u16).add(crate::kernel_params::flash_bf16_element(row + lane / 4, t * 16 + lane % 4 * 2, k, 4 * h));
            let (a, b): (u64, u64);
            core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) a, p=in(reg64) element(2 * j));
            core::arch::asm!("ld.global.u64 {v}, [{p}];", v=out(reg64) b, p=in(reg64) element(2 * j + 1));
            [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]
        }
    };
    let v = [load(0), load(1), load(2), load(3), load(4), load(5), load(6), load(7), load(8), load(9)];
    let frag = |t: usize| -> [u32; 2] { let w = v[t / 2]; if t % 2 == 0 { [w[0], w[1]] } else { [w[2], w[3]] } };
    if p.tokens <= 4 {
        // One transposed MMA per tile carries both chains of tokens 0-3 (dense_latency::tmadd,
        // exact under the operand swap): lane 8t + 4p + q reads plane p of token t from
        // normal-layout lane 4t + q; same tiles, ranges and merges, so hc_up_gate's bits.
        let src = lane / 8 * 4 + lane % 4;
        let plane = if lane / 4 % 2 == 0 { 0 } else { p.input_plane_words as usize };
        let range = |s: usize| {
            let mut c = crate::canonical::ZERO;
            let mut i = 0;
            while i < 5 {
                let at = ((s + i) * 32 + src) * 4 + plane;
                let a = if src < 16 { (*(x.add(at).cast::<Hc16Fragment>())).0 } else { [0; 4] };
                c = dense_latency::tmadd(frag(s + i), a, c);
                i += 1;
            }
            dense_latency::untranspose(c, lane)
        };
        let (z0, z1, z2, z3) = (range(0), range(5), range(10), range(15));
        return hc16_add(hc16_add(z0, z1), hc16_add(z2, z3));
    }
    let (mut left, mut right) = (crate::canonical::ZERO, crate::canonical::ZERO);
    let mut part = 0;
    while part < 4 {
        let (mut hi, mut lo) = (crate::canonical::ZERO, crate::canonical::ZERO);
        let mut i = 0;
        while i < 5 {
            let kt = part * 5 + i;
            let at = (kt * 32 + lane) * 4;
            let a = (*(x.add(at).cast::<Hc16Fragment>())).0;
            let low = (*(x.add(at + p.input_plane_words as usize).cast::<Hc16Fragment>())).0;
            let b = frag(kt);
            hi = hc16_madd(a, b, hi);
            lo = hc16_madd(low, b, lo);
            i += 1;
        }
        let z = hc16_add(hi, lo);
        match part { 0 => left = z, 1 => left = hc16_add(left, z), 2 => right = z, _ => right = hc16_add(right, z) }
        part += 1;
    }
    hc16_add(left, right)
}
/// flash_next_hc_up_latency4 at M2-16 (one M tile) with its gate read from the up weight's
/// LBF12 twin (hc16_gate: hc_up_gate's bits) and latency4's stream mix in the same order.
/// planes != 0: hc_planes writes the next projections' operand.
/// # Safety
/// As flash_next_hc_up_latency4 with 2 <= M <= 16 (grid H/8 x 1, block 128), plus twin = the
/// up weight's flash_bf16_lbf12_twin image, and planes = 0 or the two dense planes of M x H.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_latency16_lbf12(p: crate::kernel_params::Bf16Rows, normalized: u64, twin: u64, planes: u64) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let h = 2560usize;
    let feature = _block_idx_x() as usize * 8;
    let gate = hc16_gate(p, twin, warp, feature, lane);
    let sh = crate::ptx::shared::<2048>();
    if warp > 0 {
        let at = sh + (((warp - 1) * 32 + lane) * 16) as u32;
        crate::ptx::store(at, gate.a); crate::ptx::store(at + 4, gate.b); crate::ptx::store(at + 8, gate.c); crate::ptx::store(at + 12, gate.d);
    }
    crate::ptx::sync();
    if warp != 0 { return; }
    let t = lane / 4; let f = feature + lane % 4 * 2;
    let sig = |v: f32| crate::flash_next::hc_sigmoid(v);
    let mut mixed = crate::canonical::ZERO;
    let mut stream = 0;
    while stream < 4 {
        let gate = if stream == 0 { gate } else {
            let at = sh + (((stream - 1) * 32 + lane) * 16) as u32;
            crate::canonical::Acc { a: crate::ptx::load(at), b: crate::ptx::load(at + 4), c: crate::ptx::load(at + 8), d: crate::ptx::load(at + 12) }
        };
        if t < p.tokens as usize {
            let x = (normalized as *const f32).add(t * 4 * h + stream * h + f);
            mixed.a = crate::next_math::hc_stream(mixed.a, *x, gate.a, sig); mixed.b = crate::next_math::hc_stream(mixed.b, *x.add(1), gate.b, sig);
        }
        if t + 8 < p.tokens as usize {
            let x = (normalized as *const f32).add((t + 8) * 4 * h + stream * h + f);
            mixed.c = crate::next_math::hc_stream(mixed.c, *x, gate.c, sig); mixed.d = crate::next_math::hc_stream(mixed.d, *x.add(1), gate.d, sig);
        }
        stream += 1;
    }
    let rows = p.tokens as usize;
    if t < rows {
        let (a, b) = (mixed.a * 0.25, mixed.b * 0.25);
        *(p.output as *mut f32).add(t * h + f) = a;
        *(p.output as *mut f32).add(t * h + f + 1) = b;
        if planes != 0 { hc_planes!(2560, planes, rows, t, f, a, b); }
    }
    if t + 8 < rows {
        let (a, b) = (mixed.c * 0.25, mixed.d * 0.25);
        *(p.output as *mut f32).add((t + 8) * h + f) = a;
        *(p.output as *mut f32).add((t + 8) * h + f + 1) = b;
        if planes != 0 { hc_planes!(2560, planes, rows, t + 8, f, a, b); }
    }
}
