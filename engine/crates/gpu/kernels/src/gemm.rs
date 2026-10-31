//! Tensor-core projections through LLVM intrinsics and small packing PTX blocks.
//! Build core and this crate for sm_121, PTX 8.8, with `tensor-cores` enabled.
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _grid_dim_x, _thread_idx_x};
use crate::ptx;
#[path = "moe_decode.rs"]
mod moe_decode;

use crate::canonical::{Acc, ZERO, madd, add, weight_tile};
#[inline(always)]
unsafe fn pair(p: *const u16, row: usize, col: usize, rows: usize, k: usize) -> u32 {
    if row < rows {
        *p.add(row * k + col).cast::<u32>()
    } else {
        0
    }
}
#[inline(always)]
unsafe fn store(y: *mut f32, c: Acc, m: usize, n: usize, t: usize, rows: usize) {
    if m < t {
        if n < rows {
            *y.add(m * rows + n) = c.a;
        }
        if n + 1 < rows {
            *y.add(m * rows + n + 1) = c.b;
        }
    }
    if m + 8 < t {
        if n < rows {
            *y.add((m + 8) * rows + n) = c.c;
        }
        if n + 1 < rows {
            *y.add((m + 8) * rows + n + 1) = c.d;
        }
    }
}
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct AFrag([u32; 4]);
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct BFrag([u32; 2]);

// Explicit expansion keeps large accumulator tiles in registers. LLVM's
// default loop-unroll budget otherwise materializes them in thread-local RAM.
macro_rules! unroll {
    ($n:expr, $i:ident, $body:block) => {
        const { assert!($n <= 8); }
        if $n > 0 {
            let $i = 0;
            $body
        }
        if $n > 1 {
            let $i = 1;
            $body
        }
        if $n > 2 {
            let $i = 2;
            $body
        }
        if $n > 3 {
            let $i = 3;
            $body
        }
        if $n > 4 {
            let $i = 4;
            $body
        }
        if $n > 5 {
            let $i = 5;
            $body
        }
        if $n > 6 {
            let $i = 6;
            $body
        }
        if $n > 7 {
            let $i = 7;
            $body
        }
    };
}

/// Pack static row-major BF16 weights once into coalesced MMA B fragments.
/// # Safety
/// K multiple of 16; source N*K BF16; output ceil(N/8)*(K/16)*32*2 u32.
/// Launch 1D with at least ceil(N/8)*(K/16)*32 threads; output 8-byte aligned.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn pack_bf16_mma_weights(
    src: *const u16,
    dst: *mut u32,
    rows: u32,
    width: u32,
) {
    pack_bf16_weights(src, dst, rows, width);
}

#[inline(always)]
pub(crate) unsafe fn pack_bf16_weights(src: *const u16, dst: *mut u32, rows: u32, width: u32) {
    let i = (_block_idx_x() as usize * _block_dim_x() as usize) + _thread_idx_x() as usize;
    let (n, k) = (rows as usize, width as usize);
    let tile_rows = if k % 64 == 0 {64} else {8};
    let tiles = n.div_ceil(tile_rows) * (tile_rows / 8) * (k / 16);
    if i >= tiles * 32 {
        return;
    }
    let (lane, tile) = (i % 32, i / 32);
    let row = tile / (k / 16) * 8 + lane / 4;
    let col = tile % (k / 16) * 16 + lane % 4 * 2;
    let at = weight_tile(tile / (k / 16), tile % (k / 16), k / 16) * 32 + lane;
    *dst.cast::<BFrag>().add(at) = BFrag([pair(src, row, col, n, k), pair(src, row, col + 8, n, k)]);
}

/// Pack row-major BF16 activations into coalesced MMA A fragments.
/// # Safety
/// K multiple of 16; source T*K BF16; output ceil(T/16)*(K/16)*32*4 u32.
/// Launch 1D with at least ceil(T/16)*(K/16)*32 threads; output 16-byte aligned.
#[inline(always)]
unsafe fn tile_ids<const TM: usize, const TN: usize>() -> (usize, usize) {
    let warp = _thread_idx_x() as usize / 32;
    let warps = _block_dim_x() as usize / 32;
    let (m, n) = (_block_idx_x() as usize, _block_idx_y() as usize);
    (m * TM, (n * warps + warp) * TN)
}

#[inline(always)]
fn round_bf16(x: f32) -> u16 {
    (crate::bf16::rounded(x).to_bits() >> 16) as u16
}
#[inline(always)]
unsafe fn rounded_pair(x: *const f32, r: usize, c: usize, t: usize, k: usize) -> u32 {
    if r >= t || c >= k { return 0; }
    bf16_pair(*x.add(r * k + c), *x.add(r * k + c + 1))
}
/// Store one element into the established MMA A-fragment layout. Epilogues
/// use the same BF16 rounding as the standalone operand packing kernel.
#[inline(always)]
pub(crate) unsafe fn store_bf16_input_element(dst: *mut u16, token: usize, column: usize, width: usize, value: f32) {
    let tile = token / 16 * (width / 16) + column / 16;
    let lane = token % 8 * 4 + column % 8 / 2;
    let component = token % 16 / 8 + column % 16 / 8 * 2;
    *dst.add((tile * 32 + lane) * 8 + component * 2 + column % 2) = bf16_pair(value, value) as u16;
}
#[inline(always)]
pub(crate) unsafe fn store_bf16_input_bits(dst: *mut u16, token: usize, column: usize, width: usize, value: u16) {
    let tile = token / 16 * (width / 16) + column / 16;
    let lane = token % 8 * 4 + column % 8 / 2;
    let component = token % 16 / 8 + column % 16 / 8 * 2;
    *dst.add((tile * 32 + lane) * 8 + component * 2 + column % 2) = value;
}
/// Fused producers retain the standalone packer's two rounding points.
#[inline(always)]
pub(crate) unsafe fn store_split_input(dst: *mut u16, token: usize, column: usize, width: usize, plane: usize, value: f32) {
    let (hi, lo) = split2(value);
    store_bf16_input_bits(dst, token, column, width, hi);
    store_bf16_input_bits(dst.add(plane), token, column, width, lo);
}

/// Round FP32 storage once to the checkpoint's BF16 operand precision.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn pack_f32_bf16_mma_input(
    src: *const f32, dst: *mut u32, tokens: u32, width: u32,
) {
    pack_f32_input::<false, false>(src, dst, tokens, width, 32);
}

#[inline(always)]
pub(crate) unsafe fn pack_f32_input<const SPLIT: bool, const HC_SILU: bool>(src: *const f32, dst: *mut u32, tokens: u32, width: u32, lanes: usize) {
    // One thread per written fragment: lanes (a power of two <= 32) per K16 tile.
    let i = crate::index();
    let (t, k) = (tokens as usize, width as usize);
    let kt = k.div_ceil(16);
    if i >= (t + 15) / 16 * kt * lanes { return; }
    let (lane, tile) = (i & (lanes - 1), i >> lanes.trailing_zeros());
    let r = tile / kt * 16 + lane / 4;
    let c = tile % kt * 16 + lane % 4 * 2;
    if SPLIT {
        let value = |r, c| {
            let x = *src.add(r * k + c);
            if HC_SILU { crate::flash_ops::silu(x, 1.0, 4.0, 0) } else { x }
        };
        let load = |r: usize, c: usize| if r >= t || c >= k { (0, 0) }
            else { split2_pair(value(r, c), value(r, c + 1)) };
        let pairs = [load(r, c), load(r + 8, c), load(r, c + 8), load(r + 8, c + 8)];
        let plane_words = t.div_ceil(16) * kt * 128;
        *dst.cast::<AFrag>().add(tile * 32 + lane) = AFrag(pairs.map(|p| p.0));
        *dst.add(plane_words).cast::<AFrag>().add(tile * 32 + lane) = AFrag(pairs.map(|p| p.1));
    } else {
        *dst.cast::<AFrag>().add(tile * 32 + lane) = AFrag([
            rounded_pair(src, r, c, t, k), rounded_pair(src, r + 8, c, t, k),
            rounded_pair(src, r, c + 8, t, k), rounded_pair(src, r + 8, c + 8, t, k),
        ]);
    }
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn rms_shuffle(mask: u32, value: u32, distance: u32, clamp: u32) -> u32;
    #[link_name = "llvm.sqrt.f32"]
    fn rms_sqrt(value: f32) -> f32;
}

#[inline(always)]
unsafe fn normalized_pair(
    x: *const f32, w: *const f32, scale: f32,
    row: usize, col: usize, t: usize, k: usize,
) -> u32 {
    if row >= t { return 0; }
    bf16_pair(*x.add(row * k + col) * scale * *w.add(col),
        *x.add(row * k + col + 1) * scale * *w.add(col + 1))
}

#[inline(always)]
unsafe fn rms_pack(
    x: *const f32, w: *const f32, partials: *const f32, xp: *mut u32,
    t: u32, k: u32, epsilon: f32,
) {
    let (t, k) = (t as usize, k as usize);
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let row_base = _block_idx_y() as usize * 16;
    let ktile = _block_idx_x() as usize * 8 + warp;
    let chunks = k / 128;
    let scales = crate::ptx::shared::<64>();
    unroll!(2, h, {
        let row = row_base + warp + h * 8;
        let mut squares = 0.0f32;
        if row < t {
            let mut j = lane;
            while j < chunks { squares += *partials.add(row * chunks + j); j += 32; }
        }
        let mut distance = 16;
        while distance != 0 {
            squares += f32::from_bits(rms_shuffle(u32::MAX, squares.to_bits(), distance, 31));
            distance /= 2;
        }
        if lane == 0 {
            let scale = 1.0 / rms_sqrt(squares / k as f32 + epsilon);
            crate::ptx::store(scales + ((warp + h * 8) * 4) as u32, scale);
        }
    });
    crate::ptx::sync();
    let row = row_base + lane / 4;
    let col = ktile * 16 + lane % 4 * 2;
    let lo_scale = crate::ptx::load(scales + (lane / 4 * 4) as u32);
    let hi_scale = crate::ptx::load(scales + ((lane / 4 + 8) * 4) as u32);
    let pairs = [
        normalized_pair(x, w, lo_scale, row, col, t, k),
        normalized_pair(x, w, hi_scale, row + 8, col, t, k),
        normalized_pair(x, w, lo_scale, row, col + 8, t, k),
        normalized_pair(x, w, hi_scale, row + 8, col + 8, t, k),
    ];
    let tile = (row_base / 16) * (k / 16) + ktile;
    *xp.cast::<AFrag>().add(tile * 32 + lane) = AFrag(pairs);
}

/// RMS apply -> BF16 MMA input. Grid(K/128,ceil(T/16)), block256, shared64B.
/// XP aligned16 with ceil(T/16)*16*K*2 bytes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rms_pack_bf16(
    x: *const f32, w: *const f32, partials: *const f32, xp: *mut u32,
    t: u32, k: u32, epsilon: f32,
) {
    rms_pack(x, w, partials, xp, t, k, epsilon);
}

#[inline(always)]
unsafe fn packed_mma<const TM: usize, const TN: usize, const BLOCKED: bool>(
    weights: *const u32, input: *const u32, output: *mut f32,
    rows: u32, width: u32, tokens: u32, weight_stride: u32,
) {
    let lane = _thread_idx_x() as usize % 32;
    let (mtile, ntile) = tile_ids::<TM, TN>();
    let (n, k, t) = (rows as usize, width as usize / 16, tokens as usize);
    let weight_tiles = weight_stride as usize / 16;
    let mut sums = [[ZERO; TN]; TM];
    let mut col = 0;
    while col < k {
        let mut bb = [[0u32; 2]; TN];
        unroll!(TN, j, {
            if (ntile + j) * 8 < n {
                bb[j] = (*weights.cast::<BFrag>().add((if BLOCKED {weight_tile(ntile + j, col, weight_tiles)} else {(ntile + j) * weight_tiles + col}) * 32 + lane)).0;
            }
        });
        unroll!(TM, i, {
            let mut a = [0u32; 4];
            if (mtile + i) * 16 < t {
                a = (*input.cast::<AFrag>().add(((mtile + i) * k + col) * 32 + lane)).0;
            }
            unroll!(TN, j, { sums[i][j] = madd(a, bb[j], sums[i][j]); });
        });
        col += 1;
    }
    unroll!(TM, i, {
        unroll!(TN, j, {
            store(output, sums[i][j], (mtile + i) * 16 + lane / 4,
                (ntile + j) * 8 + lane % 4 * 2, t, n);
        });
    });
}

// Expert activations round once to the checkpoint's BF16 operand precision.
#[inline(always)]
pub(super) unsafe fn bf16_pair(a: f32, b: f32) -> u32 {
    let pair: u32;
    core::arch::asm!("cvt.rn.bf16x2.f32 {pair}, {b}, {a};",
        pair=out(reg32)pair, a=in(reg32)a, b=in(reg32)b, options(nomem, nostack, pure));
    pair
}

/// Carry an FP32 Flash activation as two independently rounded BF16 terms.
#[inline(always)]
pub(crate) unsafe fn split2(x: f32) -> (u16, u16) {
    let hi: u16;
    core::arch::asm!("cvt.rn.bf16.f32 {hi}, {x};",
        hi=out(reg16)hi, x=in(reg32)x, options(nomem, nostack, pure));
    let residual = x - f32::from_bits((hi as u32) << 16);
    let lo: u16;
    core::arch::asm!("cvt.rn.bf16.f32 {lo}, {residual};",
        lo=out(reg16)lo, residual=in(reg32)residual, options(nomem, nostack, pure));
    (hi, lo)
}

/// split2 of two adjacent columns (a in the low half): the same hardware RN
/// conversion and FP32 residual per element, two elements per instruction.
#[inline(always)]
pub(crate) unsafe fn split2_pair(a: f32, b: f32) -> (u32, u32) {
    let hi = bf16_pair(a, b);
    (hi, bf16_pair(a - f32::from_bits(hi << 16), b - f32::from_bits(hi & 0xffff_0000)))
}

#[inline(always)]
unsafe fn expert_tile<const ROWS: u32>(block: u32, offsets: *const u32, experts: usize) -> (usize, usize) {
    let row = block * ROWS;
    if row >= *offsets.add(experts) { return (experts, 0); }
    let (mut low, mut high) = (0, experts);
    while low + 1 < high {
        let middle = (low + high) / 2;
        if *offsets.add(middle) <= row { low = middle; } else { high = middle; }
    }
    (low, (row - *offsets.add(low)) as usize)
}

// Visit all output columns while eight adjacent M tiles remain hot in L2.
#[inline(always)]
unsafe fn grouped_tiles<const WIDTH: u32>(n: u32) -> (u32, usize) {
    let mt = _grid_dim_x();
    let nt = (n + WIDTH - 1) / WIDTH;
    let linear = _block_idx_y() * mt + _block_idx_x();
    let first = linear / (nt * 8) * 8;
    let group = (mt - first).min(8);
    let local = linear % (nt * 8);
    (first + local % group, (local / group) as usize)
}

#[inline(always)]
unsafe fn gathered_pair(input: *const f32, rows: *const u32, row: usize,
    col: usize, count: usize, hidden: usize, selected: usize, packed: u32) -> u32 {
    if row >= count || col >= hidden { return 0; }
    let token = *rows.add(row) as usize / selected;
    if packed != 0 { return *input.cast::<u32>().add((token * hidden + col) / 2); }
    let x = input.add(token * hidden + col);
    bf16_pair(*x, *x.add(1))
}

/// Round each normalized row once before its selected experts gather it.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_pack_input(input: *const f32, packed: *mut u32, pairs: u32) {
    let i = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    if i < pairs as usize { *packed.add(i) = bf16_pair(*input.add(i * 2), *input.add(i * 2 + 1)); }
}

#[inline(always)]
unsafe fn pack_expert_tile(p: &crate::kernel_params::MoePack, expert: usize, row0: usize, column: usize) {
    let input = p.input as *const f32;
    let rows = p.rows as *const u32;
    let counts = p.counts as *const u32;
    let offsets = p.offsets as *const u32;
    let packed = p.packed as *mut u32;
    let (hidden, selected, tokens, experts, input_packed) =
        (p.hidden, p.selected, p.tokens, p.experts, p.input_packed);
    if expert == experts as usize { return; }
    let count = *counts.add(expert) as usize;
    let offset = *offsets.add(expert) as usize;
    let k = hidden as usize;
    let rows = rows.add(expert * tokens as usize);
    let mut i = _thread_idx_x() as usize;
    while i < p.tile_rows as usize / 16 * 8 * 32 {
        let (lane, tile) = (i % 32, i / 32);
        let row = row0 + tile / 8 * 16 + lane / 4;
        let ktile = column * 8 + tile % 8;
        let col = ktile * 16 + lane % 4 * 2;
        if ktile * 16 < k {
            let a = gathered_pair(input, rows, row, col, count, k, selected as usize, input_packed);
            let b = gathered_pair(input, rows, row + 8, col, count, k, selected as usize, input_packed);
            let c = gathered_pair(input, rows, row, col + 8, count, k, selected as usize, input_packed);
            let d = gathered_pair(input, rows, row + 8, col + 8, count, k, selected as usize, input_packed);
            let tile = ((offset + row0) / 16 + tile / 8) * (k / 16) + ktile;
            *packed.cast::<AFrag>().add(tile * 32 + lane) = AFrag([a, b, c, d]);
        }
        i += _block_dim_x() as usize;
    }
}

/// Decode consumes the same compact work list as the projections; prefill
/// retains its 32-row tiles. Packed operands contain one BF16 per element.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_pack_grouped(p: crate::kernel_params::MoePack) {
    let offsets = p.offsets as *const u32;
    if p.tile_experts != 0 {
        let columns = (p.hidden as usize + 127) / 128;
        let total = *offsets.add(p.experts as usize) as usize / 16 * columns;
        let mut work = _block_idx_x() as usize;
        while work < total {
            let tile = work / columns;
            let expert = *(p.tile_experts as *const u32).add(tile) as usize;
            let row = tile * 16 - *offsets.add(expert) as usize;
            pack_expert_tile(&p, expert, row, work % columns);
            work += _grid_dim_x() as usize;
        }
    } else {
        let (expert, row) = expert_tile::<32>(_block_idx_x(), offsets, p.experts as usize);
        pack_expert_tile(&p, expert, row, _block_idx_y() as usize);
    }
}

#[inline(always)]
unsafe fn activated_pair(output: *mut u32, row: usize, col: usize, width: usize,
    gate_a: f32, gate_b: f32, up_a: f32, up_b: f32) {
    let a = gate_a / (1.0 + crate::layer::exp(-gate_a)) * up_a;
    let b = gate_b / (1.0 + crate::layer::exp(-gate_b)) * up_b;
    let v = bf16_pair(a, b);
    let tile = row / 16 * (width / 16) + col / 16;
    let lane = row % 8 * 4 + col % 8 / 2;
    let fragment = row % 16 / 8 + col % 16 / 8 * 2;
    *output.add((tile * 32 + lane) * 4 + fragment) = v;
}

// Four K fragments per stage; eight warps share each down-weight tile.
#[inline(always)]
unsafe fn stage_expert<const COLS: u32>(
    w: *const BFrag, n: u32, k: u32, ntile: u32, col: u32,
) {
    let mut i = _thread_idx_x();
    while i < COLS * 8 {
        let at = weight_tile((ntile + i / 64) as usize, (col + i / 16 % 4) as usize, k as usize) * 32 + i as usize % 16 * 2;
        if (ntile + i / 64) * 8 < n {
            ptx::copy16(i * 16, w.add(at as usize).cast());
        } else {
            ptx::store(i * 16, 0.0); ptx::store(i * 16 + 4, 0.0);
            ptx::store(i * 16 + 8, 0.0); ptx::store(i * 16 + 12, 0.0);
        }
        i += 256;
    }
}

/// Grouped 32x128 tensor-core tile, fused gate/up and SiLU*up packing.
/// Grid=(ceil(padded_rows/32),ceil(intermediate/128),1), block128, shared0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_prefill_gate_up(
    tables: *const u64, counts: *const u32, offsets: *const u32,
    packed: *const u32, activated: *mut u32,
    intermediate: u32, hidden: u32, experts: u32,
) {
    let (mtile, ntile) = grouped_tiles::<128>(intermediate);
    let (expert, row0) = expert_tile::<32>(mtile, offsets, experts as usize);
    if expert == experts as usize { return; }
    let (count, offset) = (*counts.add(expert) as usize, *offsets.add(expert) as usize);
    if row0 >= count { return; }
    let (n, k) = (intermediate as usize, hidden as usize / 16);
    let lane = _thread_idx_x() as usize % 32;
    let ntile = (ntile * (_block_dim_x() as usize / 32) + _thread_idx_x() as usize / 32) * 4;
    let gate = *tables.add(expert * 3) as *const BFrag;
    let up = *tables.add(expert * 3 + 1) as *const BFrag;
    let input = packed.cast::<AFrag>().add(offset / 16 * k * 32);
    let mut gh = [[ZERO; 4]; 2];
    let mut uh = [[ZERO; 4]; 2];
    let mut col = 0;
    while col < k {
        let mut bg = [[0u32; 2]; 4];
        let mut bu = [[0u32; 2]; 4];
        unroll!(4, j, {
            if (ntile + j) * 8 < n {
                let at = weight_tile(ntile + j, col, k) * 32 + lane;
                bg[j] = (*gate.add(at)).0;
                bu[j] = (*up.add(at)).0;
            }
        });
        unroll!(2, i, {
            let at = ((row0 / 16 + i) * k + col) * 32 + lane;
            let a = (*input.add(at)).0;
            unroll!(4, j, {
                gh[i][j] = madd(a, bg[j], gh[i][j]);
                uh[i][j] = madd(a, bu[j], uh[i][j]);
            });
        });
        col += 1;
    }
    unroll!(2, i, {
        unroll!(4, j, {
            let row = row0 + i * 16 + lane / 4;
            let col = (ntile + j) * 8 + lane % 4 * 2;
            if col < n {
                let (g, u) = (gh[i][j], uh[i][j]);
                // Zero-padded rows are stored too, so down reads complete tiles.
                if row < count {
                    activated_pair(activated, offset + row, col, n, g.a, g.b, u.a, u.b);
                } else {
                    activated_pair(activated, offset + row, col, n, 0.0, 0.0, 0.0, 0.0);
                }
                if row + 8 < count {
                    activated_pair(activated, offset + row + 8, col, n, g.c, g.d, u.c, u.d);
                } else {
                    activated_pair(activated, offset + row + 8, col, n, 0.0, 0.0, 0.0, 0.0);
                }
            }
        });
    });
}

#[inline(always)]
unsafe fn expert_down<const ROWS: u32, const COLS: u32>(
    tables: *const u64, counts: *const u32, offsets: *const u32,
    packed: *const u32, projected: *mut f32, hidden: u32, intermediate: u32, experts: u32,
) {
    let (mtile, ntile) = grouped_tiles::<COLS>(hidden);
    let (expert, row0) = expert_tile::<ROWS>(mtile, offsets, experts as usize);
    if expert == experts as usize { return; }
    let (count, offset, row0) = (*counts.add(expert), *offsets.add(expert), row0 as u32);
    if row0 >= count { return; }
    let (n, k) = (hidden, intermediate / 16);
    let lane = _thread_idx_x() % 32;
    let warp = _thread_idx_x() / 32;
    let columns = COLS / 64;
    let tile = ntile as u32 * (COLS / 8);
    let ntile = tile + warp % columns * 8;
    let row0 = row0 + warp / columns * 32;
    let weights = *tables.add(expert * 3 + 2) as *const BFrag;
    let input = packed.cast::<AFrag>().add((offset / 16 * k * 32) as usize);
    let output = projected.add(offset as usize * n as usize);
    let mut hi = [[ZERO; 8]; 2];
    let mut col = 0;
    while col < k {
        stage_expert::<COLS>(weights, n, k, tile, col);
        ptx::wait(); ptx::sync();
        unroll!(4, z, {
            let mut bb = [[0u32; 2]; 8];
            unroll!(8, j, {
                let at = ((warp % columns * 8 + j as u32) * 4 + z as u32) * 256 + lane * 8;
                bb[j] = ptx::load2(at);
            });
            unroll!(2, i, {
                let at = (((row0 / 16 + i as u32) * k + col + z as u32) * 32 + lane) as usize;
                let a = (*input.add(at)).0;
                unroll!(8, j, {
                    hi[i][j] = madd(a, bb[j], hi[i][j]);
                });
            });
        });
        ptx::sync(); col += 4;
    }
    let (count, row0, lane, ntile, n) = (count as usize, row0 as usize,
        lane as usize, ntile as usize, n as usize);
    unroll!(2, i, {
        unroll!(8, j, {
            store(output, hi[i][j], row0 + i * 16 + lane / 4,
                (ntile + j) * 8 + lane % 4 * 2, count, n);
        });
    });
}

/// Large groups share each down-weight tile across 128 rows. Block256;
/// wide: grid(ceil(padded/128),ceil(hidden/128)), shared16384, otherwise 64x256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_prefill_down(
    tables: *const u64, counts: *const u32, offsets: *const u32,
    packed: *const u32, projected: *mut f32, hidden: u32, intermediate: u32, experts: u32, wide: u32,
) {
    if wide != 0 {
        expert_down::<128, 128>(tables, counts, offsets, packed, projected, hidden, intermediate, experts);
    } else {
        expert_down::<64, 256>(tables, counts, offsets, packed, projected, hidden, intermediate, experts);
    }
}

/// BF16 input, grid=(ceil(T/32),ceil(N/(48*block/32)),1), shared0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t32n48_strided(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, weight_stride: u32,
) {
    packed_mma::<2, 6, true>(w, x, y, n, k, t, weight_stride);
}

/// Packed8 short projections retain the hybrid cartridge's ascending K order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t32n48_native(p: crate::projection_params::DecodeProjection) {
    packed_mma::<2,6,false>(p.weights as *const u32, p.input as *const u32, p.output as *mut f32,
        p.rows, p.width, p.tokens, p.width);
}

/// Narrow native BF16 projections: bounded K partitions limit FP32 cancellation
/// before the BF16 output boundary. Grid(ceil(T/16),ceil(N/48)), block32.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t16n48_bf16(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, stride: u32,
) {
    let lane = _thread_idx_x() as usize;
    let mt = _block_idx_x() as usize;
    let nt = _block_idx_y() as usize * 6;
    let (kt, wt) = (k as usize / 16, stride as usize / 16);
    let total = narrow_total(w, x, mt, nt, kt, wt, lane, n as usize);
    unroll!(6, j, {
        let a = total[j];
        store(y, Acc { a:crate::flash_gdn::rounded(a.a), b:crate::flash_gdn::rounded(a.b),
            c:crate::flash_gdn::rounded(a.c), d:crate::flash_gdn::rounded(a.d) },
            mt * 16 + lane / 4, (nt + j) * 8 + lane % 4 * 2, t as usize, n as usize);
    });
}

#[inline(always)]
unsafe fn narrow_total(w: *const u32, x: *const u32, mt: usize, nt: usize,
    kt: usize, wt: usize, lane: usize, n: usize) -> [Acc; 6] {
    let mut total = [ZERO; 6];
    for part in 0..16 {
        let mut sums = [ZERO; 6];
        let mut col = kt * part / 16;
        while col < kt * (part + 1) / 16 {
            let a = (*x.cast::<AFrag>().add((mt * kt + col) * 32 + lane)).0;
            unroll!(6, j, {
                if (nt + j) * 8 < n {
                    let b = (*w.cast::<BFrag>().add(((nt + j) * wt + col) * 32 + lane)).0;
                    sums[j] = madd(a, b, sums[j]);
                }
            });
            col += 1;
        }
        unroll!(6, j, {
            total[j].a += sums[j].a; total[j].b += sums[j].b;
            total[j].c += sums[j].c; total[j].d += sums[j].d;
        });
    }
    total
}


#[path = "attention.rs"]
mod attention;
mod attention_fast {
//! Independent grouped-query prefill: register scores, absolute 64-key tiles.
//! Grouped-head staging idea studied in TensorFold; this Rust implementation is our own.
use super::*;
use crate::ptx;

macro_rules! each16 {
    ($i:ident, $body:block) => {{
        unroll!(8, j, { let $i = j; $body });
        unroll!(8, j, { let $i = j + 8; $body });
    }};
}

#[inline(always)]
fn swizzle(row: usize, col: usize) -> u32 {
    ((row * 128 + (col ^ ((row & 7) * 8))) * 2) as u32
}

// Keep address formation with the load: hoisting every unrolled swizzle
// across the key loop otherwise consumes more registers than Q and O.
#[inline(always)]
unsafe fn matrix<const TRANS: bool>(sh: u32, lane_offset: u32, xor: u32, offset: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    if TRANS {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; add.u32 p, p, {offset}; add.u32 p, p, {sh}; ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{{a}, {b}}}, [p]; }}",
            a=out(reg32)a, b=out(reg32)b, sh=in(reg32)sh,
            lane=in(reg32)lane_offset, xor=in(reg32)xor, offset=in(reg32)offset);
    } else {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; add.u32 p, p, {offset}; add.u32 p, p, {sh}; ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{a}, {b}}}, [p]; }}",
            a=out(reg32)a, b=out(reg32)b, sh=in(reg32)sh,
            lane=in(reg32)lane_offset, xor=in(reg32)xor, offset=in(reg32)offset);
    }
    [a, b]
}

/// Packed, swizzled operands are shared by every query tile in this layer call.
/// K and V each use one BF16 plane per 32 rows. The selected KV heads are
/// contiguous in packed storage; source rows retain their stride.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_fast_pack(
    k: *const f32, v: *const f32, kp: *mut u16, vp: *mut u16,
    kv: u32, start: u32, end: u32, packed_kv: u32, first_head: u32,
) {
    let i = (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize + start as usize * packed_kv as usize * 64;
    let padded = (end as usize + 63) / 64 * 64;
    if i >= padded * packed_kv as usize * 64 { return; }
    let col = i % 64 * 2;
    let head = i / 64 % packed_kv as usize;
    let row = i / (packed_kv as usize * 64);
    let tile = row / 32 * packed_kv as usize + head;
    let offset = swizzle(row % 32, col) as usize / 2;
    let at = (row * kv as usize + first_head as usize + head) * 128 + col;
    let (a, b, c, d) = if row < end as usize {
        (*k.add(at), *k.add(at + 1), *v.add(at), *v.add(at + 1))
    } else { (0.0, 0.0, 0.0, 0.0) };
    *kp.add(tile * 4096 + offset).cast::<u32>() = bf16_pair(a, b);
    *vp.add(tile * 4096 + offset).cast::<u32>() = bf16_pair(c, d);
}

#[inline(always)]
unsafe fn fetch(k: *const u16, v: *const u16,
    pos: usize, kv: usize, head: usize, half: usize, tid: usize,
) {
    let tile = pos / 32 * kv + head;
    // Two complete 64-key banks let the next tile copy during all QK/PV work.
    let bank = ((pos as u32 >> 6) & 1) * 32768;
    let key = bank + half as u32 * 8192;
    let value = bank + 16384;
    unroll!(2, z, {
        let i = tid + z * 256;
        ptx::copy16(key + (i * 16) as u32, k.add(tile * 4096 + i * 8).cast());
        ptx::copy16(value + (half * 8192 + i * 16) as u32,
            v.add(tile * 4096 + i * 8).cast());
    });
}

/// BF16 grouped-query attention with FP32 accumulation and softmax statistics.
/// FP32 Q/O[T,H,128], packed K/V for KV heads starting at first_head.
/// Grid(ceil(T/32),KV,1), block256, dynamic shared65536.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_fast_adaptive(
    q: *const f32, k: *const u16, v: *const u16, o: *mut f32,
    h: u32, kv: u32, t: u32, base: u32, capacity: u32, first_head: u32,
) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let kh = _block_idx_y() as usize;
    let head = (kh + first_head as usize) * 4 + warp % 4;
    let block_row = _block_idx_x() as usize * 32;
    let row = block_row + warp / 4 * 16 + lane / 4;
    let col = lane % 4 * 2;
    let end = (base as usize + (block_row + 32).min(t as usize)).min(capacity as usize);
    // No static shared allocations: the driver's dynamic region starts at 0.
    let key_lane = swizzle(lane % 8, (lane / 8 % 2) * 8);
    let value_lane = swizzle(lane % 16, 0);
    let mut query = [[0u32; 4]; 8];
    unroll!(8, d, {
        let at = d * 16 + col;
        let load = |r: usize, c: usize| {
            if r < t as usize { *q.add((r * h as usize + head) * 128 + c) } else { 0.0 }
        };
        query[d] = [
            bf16_pair(load(row, at), load(row, at + 1)),
            bf16_pair(load(row + 8, at), load(row + 8, at + 1)),
            bf16_pair(load(row, at + 8), load(row, at + 9)),
            bf16_pair(load(row + 8, at + 8), load(row + 8, at + 9)),
        ];
    });
    let mut out = [ZERO; 16];
    let mut maximum = [-f32::INFINITY; 2];
    let mut denom = [0.0f32; 2];
    fetch(k, v, 0, kv as usize, kh, 0, tid);
    fetch(k, v, 32, kv as usize, kh, 1, tid);
    let mut pos = 0;
    while pos < end {
        ptx::wait();
        ptx::sync();
        let bank = ((pos as u32 >> 6) & 1) * 32768;
        if pos + 64 < end {
            fetch(k, v, pos + 64, kv as usize, kh, 0, tid);
            fetch(k, v, pos + 96, kv as usize, kh, 1, tid);
        }
        let mut p = [[0.0f32; 16]; 2];
        unroll!(2, half, {
            let key = bank + half as u32 * 8192;
            let mut dot = [ZERO; 4];
            unroll!(8, d, {
              unroll!(4, n, {
                let b = matrix::<false>(key, key_lane, (d * 32) as u32, (n * 2048) as u32);
                dot[n] = madd(query[d], b, dot[n]);
              });
            });
            unroll!(4, n, {
                let at = half * 8 + n * 2;
                p[0][at] = dot[n].a * 0.08838834764831845;
                p[0][at + 1] = dot[n].b * 0.08838834764831845;
                p[1][at] = dot[n].c * 0.08838834764831845;
                p[1][at + 1] = dot[n].d * 0.08838834764831845;
            });
        });
        unroll!(2, r, {
            let qr = row + r * 8;
            let mut mx = -f32::INFINITY;
            each16!(n, {
                if pos + n / 2 * 8 + col + n % 2 > base as usize + qr || qr >= t as usize {
                    p[r][n] = -f32::INFINITY;
                }
                else if !p[r][n].is_finite() {
                    // Preserve an invalid numerical row for host-side rejection.
                    p[r][n] = f32::NAN;
                }
                mx = mx.max(p[r][n]);
            });
            mx = mx.max(f32::from_bits(crate::shuffle_xor(u32::MAX, mx.to_bits(), 1, 31)));
            mx = mx.max(f32::from_bits(crate::shuffle_xor(u32::MAX, mx.to_bits(), 2, 31)));
            let nm = if qr < t as usize { maximum[r].max(mx) } else { 0.0 };
            let mut sum = 0.0;
            each16!(n, {
                let prob = f32::from_bits((round_bf16(ptx::exp(p[r][n] - nm)) as u32) << 16);
                p[r][n] = prob;
                sum += prob;
            });
            sum += f32::from_bits(crate::shuffle_xor(u32::MAX, sum.to_bits(), 1, 31));
            sum += f32::from_bits(crate::shuffle_xor(u32::MAX, sum.to_bits(), 2, 31));
            if nm > maximum[r] {
                let scale = ptx::exp(maximum[r] - nm);
                denom[r] *= scale;
                each16!(n, {
                    if r == 0 { out[n].a *= scale; out[n].b *= scale; }
                    else { out[n].c *= scale; out[n].d *= scale; }
                });
                maximum[r] = nm;
            }
            denom[r] += sum;
        });
        let value = bank + 16384;
        unroll!(4, z, {
            let a = [
                bf16_pair(p[0][z * 4], p[0][z * 4 + 1]),
                bf16_pair(p[1][z * 4], p[1][z * 4 + 1]),
                bf16_pair(p[0][z * 4 + 2], p[0][z * 4 + 3]),
                bf16_pair(p[1][z * 4 + 2], p[1][z * 4 + 3]),
            ];
            each16!(n, {
                let b = matrix::<true>(value, value_lane, (n * 16) as u32, (z * 4096) as u32);
                out[n] = madd(a, b, out[n]);
            });
        });
        pos += 64;
    }
    each16!(n, {
        let a = out[n];
        if row < t as usize {
            let at = (row * h as usize + head) * 128 + n * 8 + col;
            *o.add(at) = a.a / denom[0];
            *o.add(at + 1) = a.b / denom[0];
        }
        if row + 8 < t as usize {
            let at = ((row + 8) * h as usize + head) * 128 + n * 8 + col;
            *o.add(at) = a.c / denom[1];
            *o.add(at + 1) = a.d / denom[1];
        }
    });
}
}

#[path = "gemm_shared.rs"]
mod shared;
#[path = "gemm_gate_packed.rs"]
mod gate_packed;
#[path = "gemm_prefill_wide.rs"]
mod prefill_wide;
