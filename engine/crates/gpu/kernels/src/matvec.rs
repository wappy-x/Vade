//! Single-row projections use the same native BF16 MMA order as verification.
use crate::{math::bf16_to_f32, shuffle_xor, canonical::{range, Acc, ZERO, madd, add, weight_tile}};
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _thread_idx_x};

#[inline(always)]
fn activation(x: f32) -> f32 {
    let bits = x.to_bits();
    if bits & 0x7f800000 == 0x7f800000 { return x; }
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Words([u32; 4]);
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct Pair([u32; 2]);

/// Optional row-major scalar bring-up path; production uses packed MMA weights.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_f32_vec8(
    weights: *const u16, input: *const f32, output: *mut f32, rows: u32, width: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (row, lane) = (tid / 32, tid % 32);
    if row >= rows as usize { return; }
    let mut col = lane * 8;
    let mut s = [0.0f32; 8];
    while col < width as usize {
        let v = *(weights.add(row * width as usize + col) as *const Words);
        for j in 0..4 {
            s[j * 2] += bf16_to_f32(v.0[j] as u16) * activation(*input.add(col + j * 2));
            s[j * 2 + 1] += bf16_to_f32((v.0[j] >> 16) as u16) * activation(*input.add(col + j * 2 + 1));
        }
        col += 256;
    }
    let mut sum = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for d in [16, 8, 4, 2, 1] {
        sum += f32::from_bits(shuffle_xor(u32::MAX, sum.to_bits(), d, 31));
    }
    if lane == 0 { *output.add(row) = sum; }
}

#[inline(always)]
unsafe fn bf16_pair(p: *const f32) -> u32 {
    let v = (*p.cast::<Pair>()).0;
    let packed: u32;
    core::arch::asm!("cvt.rn.bf16x2.f32 {p}, {b}, {a};", p = out(reg32) packed,
        a = in(reg32) f32::from_bits(v[0]), b = in(reg32) f32::from_bits(v[1]));
    packed
}

/// Eight output columns per warp, using the canonical BF16 MMA order.
#[inline(always)]
pub(crate) unsafe fn dot_range(weights: *const u32, input: *const f32, tile: usize,
    lane: usize, width: u32, stride: u32, column_offset: u32, part: usize) -> Acc {
    let kt = width as usize / 16;
    let (mut c, end) = range(kt, part);
    let mut sum = ZERO;
    while c < end {
        let b = (*weights.cast::<Pair>().add(weight_tile(tile, c + column_offset as usize / 16, stride as usize / 16) * 32 + lane)).0;
        let col = c * 16 + lane % 4 * 2;
        let lo = bf16_pair(input.add(col));
        let hi = bf16_pair(input.add(col + 8));
        sum = madd([lo, lo, hi, hi], b, sum);
        c += 1;
    }
    sum
}

#[inline(always)]
unsafe fn dot(weights: *const u32, input: *const f32, tile: usize,
    lane: usize, width: u32, stride: u32, column_offset: u32) -> Acc {
    let a = dot_range(weights, input, tile, lane, width, stride, column_offset, 0);
    let b = dot_range(weights, input, tile, lane, width, stride, column_offset, 1);
    let c = dot_range(weights, input, tile, lane, width, stride, column_offset, 2);
    let d = dot_range(weights, input, tile, lane, width, stride, column_offset, 3);
    add(add(a, b), add(c, d))
}

#[inline(always)]
unsafe fn store_pair(output: *mut f32, sum: Acc, tile: usize, lane: usize, rows: usize, offset: usize) {
    if lane < 4 {
        let row = tile * 8 + lane * 2;
        if row >= offset && row < offset + rows { *output.add(row - offset) = sum.a; }
        if row + 1 >= offset && row + 1 < offset + rows { *output.add(row + 1 - offset) = sum.b; }
    }
}

/// Packed W, FP32 storage rounded to BF16 operands, K%16=0. Block64,
/// grid ceil(N/16); no shared memory or duplicate weights.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_f32_packed8(
    weights: *const u32, input: *const f32, output: *mut f32, rows: u32, width: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (tile, lane) = (tid / 32, tid % 32);
    if tile * 8 >= rows as usize { return; }
    store_pair(output, dot(weights, input, tile, lane, width, width, 0), tile, lane, rows as usize, 0);
}

/// An original packed matrix view. Grid ceil((N+offset%8)/16); no weight copy.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_f32_packed8_strided(
    weights: *const u32, input: *const f32, output: *mut f32,
    rows: u32, width: u32, stride: u32, row_offset: u32, column_offset: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (tile, lane) = ((tid / 32) + row_offset as usize / 8, tid % 32);
    if tile * 8 >= row_offset as usize + rows as usize { return; }
    store_pair(output, dot(weights, input, tile, lane, width, stride, column_offset), tile, lane, rows as usize, row_offset as usize);
}

/// Narrow draft projections distribute the same four K ranges across CTAs.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_f32_split(
    weights: *const u32, input: *const f32, output: *mut f32, rows: u32, width: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (tile, lane) = (tid / 32, tid % 32);
    if tile * 8 >= rows as usize { return; }
    let part = core::arch::nvptx::_block_idx_y() as usize;
    let sum = dot_range(weights, input, tile, lane, width, width, 0, part);
    store_pair(output.add(part * rows as usize), sum, tile, lane, rows as usize, 0);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_split_merge(a: crate::kernel_params::ProjectionMerge) {
    let input = a.input as *const f32; let output = a.output as *mut f32;
    let n = a.elements as usize;
    let row = crate::index() * 4;
    if row + 4 <= n && n % 4 == 0 && (a.input | a.output) % 16 == 0 {
        let p = [(*input.add(row).cast::<Words>()).0,
            (*input.add(n + row).cast::<Words>()).0,
            (*input.add(2*n + row).cast::<Words>()).0,
            (*input.add(3*n + row).cast::<Words>()).0];
        let mut values = [0u32; 4];
        for j in 0..4 {
            let value = (f32::from_bits(p[0][j]) + f32::from_bits(p[1][j]))
                + (f32::from_bits(p[2][j]) + f32::from_bits(p[3][j]));
            values[j] = if a.round_bf16 != 0 {activation(value)} else {value}.to_bits();
        }
        *output.add(row).cast::<Words>() = Words(values);
    } else {
        for j in row..(row + 4).min(n) {
            let value = (*input.add(j) + *input.add(n + j))
                + (*input.add(2*n + j) + *input.add(3*n + j));
            *output.add(j) = if a.round_bf16 != 0 {activation(value)} else {value};
        }
    }
}

/// Native scalar projection retains its original packed reduction order.
#[inline(always)]
pub(crate) unsafe fn wide_row_dot(weights: *const u32, input: *const f32, row: usize, width: u32) -> f32 {
    let tid = _thread_idx_x() as usize;
    let tiles = width as usize / 16;
    let base = row / 8 * tiles * 32 + row % 8 * 4 + tid % 4;
    let mut tile = tid / 4;
    while tile < tiles {
        let v = packed_weight(weights, base + tile * 32);
        let col = tile * 16 + tid % 4 * 2;
        let a = (*input.add(col).cast::<Pair>()).0;
        let b = (*input.add(col + 8).cast::<Pair>()).0;
        crate::ptx::store((col * 4) as u32, bf16_to_f32(v[0] as u16) * activation(f32::from_bits(a[0])));
        crate::ptx::store((col * 4 + 4) as u32, bf16_to_f32((v[0] >> 16) as u16) * activation(f32::from_bits(a[1])));
        crate::ptx::store((col * 4 + 32) as u32, bf16_to_f32(v[1] as u16) * activation(f32::from_bits(b[0])));
        crate::ptx::store((col * 4 + 36) as u32, bf16_to_f32((v[1] >> 16) as u16) * activation(f32::from_bits(b[1])));
        tile += 32;
    }
    crate::ptx::sync();
    let mut s = [0.0f32; 8];
    if tid < 4 {
        let mut tile = 0;
        while tile + 1 < tiles {
            for part in 0..2 {
                let col = (tile + part) * 16 + tid * 2;
                let p = part * 4;
                s[p] += crate::ptx::load((col * 4) as u32);
                s[p + 1] += crate::ptx::load((col * 4 + 4) as u32);
                s[p + 2] += crate::ptx::load((col * 4 + 32) as u32);
                s[p + 3] += crate::ptx::load((col * 4 + 36) as u32);
            }
            tile += 2;
        }
        if tile < tiles {
            let col = tile * 16 + tid * 2;
            s[0] += crate::ptx::load((col * 4) as u32);
            s[1] += crate::ptx::load((col * 4 + 4) as u32);
            s[2] += crate::ptx::load((col * 4 + 32) as u32);
            s[3] += crate::ptx::load((col * 4 + 36) as u32);
        }
    }
    reduce_packed(s)
}

/// Q/K/V share one launch and the input without concatenating weights.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_qkv_packed(
    q_weight: *const u32, k_weight: *const u32, v_weight: *const u32,
    input: *const f32, query: *mut f32, key: *mut f32, value: *mut f32,
    q_rows: u32, kv_rows: u32, width: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (tile, lane) = (tid / 32, tid % 32);
    let qt = q_rows as usize / 8;
    let kt = kv_rows as usize / 8;
    let (w, y, tile, rows) = if tile < qt { (q_weight, query, tile, q_rows) }
        else if tile < qt + kt { (k_weight, key, tile - qt, kv_rows) }
        else { (v_weight, value, tile - qt - kt, kv_rows) };
    if tile * 8 >= rows as usize { return; }
    store_pair(y, dot(w, input, tile, lane, width, width, 0), tile, lane, rows as usize, 0);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn gate_up_swiglu_packed(
    gate: *const u32, up: *const u32, input: *const f32, output: *mut f32,
    rows: u32, width: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (tile, lane) = (tid / 32, tid % 32);
    if tile * 8 >= rows as usize { return; }
    let g = dot(gate, input, tile, lane, width, width, 0);
    let u = dot(up, input, tile, lane, width, width, 0);
    if lane < 4 {
        let row = tile * 8 + lane * 2;
        if row < rows as usize { *output.add(row) = g.a / (1.0 + crate::layer::exp(-g.a)) * u.a; }
        if row + 1 < rows as usize { *output.add(row + 1) = g.b / (1.0 + crate::layer::exp(-g.b)) * u.b; }
    }
}

#[cfg(feature = "tensor-cores")]
#[path = "verify_mma.rs"]
mod verify_mma;

macro_rules! rows_unroll {
    ($n:expr, $i:ident, $body:block) => {
        const { assert!($n <= 10); }
        if $n > 0 { let $i = 0; $body }
        if $n > 1 { let $i = 1; $body }
        if $n > 2 { let $i = 2; $body }
        if $n > 3 { let $i = 3; $body }
        if $n > 4 { let $i = 4; $body }
        if $n > 5 { let $i = 5; $body }
        if $n > 6 { let $i = 6; $body }
        if $n > 7 { let $i = 7; $body }
        if $n > 8 { let $i = 8; $body }
        if $n > 9 { let $i = 9; $body }
    };
}


use crate::projection_params::DecodeProjection;


#[inline(always)]
fn operand<const NATIVE: bool>(x: f32) -> f32 {
    if NATIVE { x } else { activation(x) }
}


#[inline(always)]
unsafe fn packed_weight(weights: *const u32, pair: usize) -> [u32; 2] {
    (*weights.cast::<Pair>().add(pair)).0
}


#[inline(always)]
unsafe fn reduce_packed(s: [f32; 8]) -> f32 {
    let mut sum = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    sum += f32::from_bits(shuffle_xor(u32::MAX, sum.to_bits(), 2, 31));
    sum += f32::from_bits(shuffle_xor(u32::MAX, sum.to_bits(), 1, 31));
    sum
}


#[inline(always)]
unsafe fn stage_decode_weight(sh: u32, weights: *const u32, row: usize, kt: u32, tile: u32, lane: u32) {
    let base = (row / 8 * kt as usize + tile as usize) * 64;
    for chunk in 0..4u32 {
        let i = lane + chunk * 32;
        crate::ptx::copy16(sh + i * 16, weights.add(base + i as usize * 4));
    }
}


#[inline(always)]
unsafe fn packed_dot_staged<const WIDTH: u32, const NATIVE: bool>(weights: *const u32, input: *const f32, row: usize, lane: usize, rows: usize) -> f32 {
    let sh = crate::ptx::shared::<8192>() + _thread_idx_x() as u32 / 32 * 4096;
    let kt = WIDTH / 16;
    let q = lane % 4 * 2;
    let valid = row < rows;
    let mut s = [0f32; 8];
    if valid { stage_decode_weight(sh, weights, row, kt, 0, lane as u32); }
    crate::ptx::wait();
    crate::ptx::warp_sync();
    let mut tile = 0u32;
    while tile < kt {
        let current = sh + tile / 8 % 2 * 2048;
        let next = sh + (tile / 8 + 1) % 2 * 2048;
        if valid && tile + 8 < kt { stage_decode_weight(next, weights, row, kt, tile + 8, lane as u32); }
        if valid {
            let mut j = 0u32;
            while j < 8 {
                for part in 0..2u32 {
                    let v = crate::ptx::load2(current + (j + part) * 256 + lane as u32 * 8);
                    let col = (tile + j + part) as usize * 16 + q;
                    let a = (*input.add(col).cast::<Pair>()).0;
                    let b = (*input.add(col + 8).cast::<Pair>()).0;
                    let p = part as usize * 4;
                    s[p] += bf16_to_f32(v[0] as u16) * operand::<NATIVE>(f32::from_bits(a[0]));
                    s[p+1] += bf16_to_f32((v[0] >> 16) as u16) * operand::<NATIVE>(f32::from_bits(a[1]));
                    s[p+2] += bf16_to_f32(v[1] as u16) * operand::<NATIVE>(f32::from_bits(b[0]));
                    s[p+3] += bf16_to_f32((v[1] >> 16) as u16) * operand::<NATIVE>(f32::from_bits(b[1]));
                }
                j += 2;
            }
        }
        crate::ptx::wait();
        crate::ptx::warp_sync();
        tile += 8;
    }
    reduce_packed(s)
}


#[inline(always)]
unsafe fn gate_up_staged<const WIDTH: u32, const NATIVE: bool>(gate: *const u32, up: *const u32, input: *const f32, output: *mut f32, rows: u32, precision: u32) {
    let tid = _block_idx_x() as usize * 64 + _thread_idx_x() as usize;
    let (row, lane) = (tid / 4, tid % 32);
    let sh = crate::ptx::shared::<16384>() + _thread_idx_x() as u32 / 32 * 8192;
    let kt = WIDTH / 16;
    let q = lane % 4 * 2;
    let valid = row < rows as usize;
    let mut g = [0f32; 8];
    let mut u = [0f32; 8];
    if valid {
        stage_decode_weight(sh, gate, row, kt, 0, lane as u32);
        stage_decode_weight(sh + 2048, up, row, kt, 0, lane as u32);
    }
    crate::ptx::wait();
    crate::ptx::warp_sync();
    let mut tile = 0u32;
    while tile < kt {
        let current = sh + tile / 8 % 2 * 4096;
        let next = sh + (tile / 8 + 1) % 2 * 4096;
        if valid && tile + 8 < kt {
            stage_decode_weight(next, gate, row, kt, tile + 8, lane as u32);
            stage_decode_weight(next + 2048, up, row, kt, tile + 8, lane as u32);
        }
        if valid {
            let mut j = 0u32;
            while j < 8 {
                for part in 0..2u32 {
                    let at = current + (j + part) * 256 + lane as u32 * 8;
                    let a = crate::ptx::load2(at);
                    let b = crate::ptx::load2(at + 2048);
                    let col = (tile + j + part) as usize * 16 + q;
                    let lo = (*input.add(col).cast::<Pair>()).0;
                    let hi = (*input.add(col + 8).cast::<Pair>()).0;
                    let x = [f32::from_bits(lo[0]), f32::from_bits(lo[1]), f32::from_bits(hi[0]), f32::from_bits(hi[1])];
                    let aa = [a[0] as u16, (a[0] >> 16) as u16, a[1] as u16, (a[1] >> 16) as u16];
                    let bb = [b[0] as u16, (b[0] >> 16) as u16, b[1] as u16, (b[1] >> 16) as u16];
                    for c in 0..4 {
                        g[part as usize * 4 + c] += bf16_to_f32(aa[c]) * operand::<NATIVE>(x[c]);
                        u[part as usize * 4 + c] += bf16_to_f32(bb[c]) * operand::<NATIVE>(x[c]);
                    }
                }
                j += 2;
            }
        }
        crate::ptx::wait();
        crate::ptx::warp_sync();
        tile += 8;
    }
    let g = reduce_packed(g);
    let u = reduce_packed(u);
    if lane % 4 == 0 && valid { *output.add(row) = swiglu(g, u, precision); }
}


#[inline(always)]
unsafe fn swiglu(g: f32, u: f32, precision: u32) -> f32 {
    if precision == 0 { g / (1.0 + crate::layer::exp(-g)) * u }
    else {
        let g = activation(g);
        activation(activation(g * (1.0 / (1.0 + crate::layer::exp(-g)))) * activation(u))
    }
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_native(
    weights: *const u32, input: *const f32, output: *mut f32, rows: u32, width: u32,
) {
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (row, lane) = (tid / 4, tid % 32);
    let sum = if width == 5120 && rows % 8 == 0 {
        packed_dot_staged::<5120, true>(weights, input, row, lane, rows as usize)
    } else if width == 6144 && rows % 8 == 0 {
        packed_dot_staged::<6144, true>(weights, input, row, lane, rows as usize)
    } else { packed_dot::<true>(weights, input, row, lane, rows as usize, width) };
    if lane % 4 == 0 && row < rows as usize { *output.add(row) = activation(sum); }
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_native_narrow(p: DecodeProjection) {
    let row = _block_idx_x() as usize;
    let token = _block_idx_y() as usize;
    if row >= p.rows as usize || token >= p.tokens as usize { return; }
    let input = (p.input as *const f32).add(token * p.width as usize);
    let output = (p.output as *mut f32).add(token * p.rows as usize);
    let sum = wide_row_dot(p.weights as *const u32, input, row, p.width);
    if _thread_idx_x() == 0 { *output.add(row) = activation(sum); }
}


#[inline(always)]
pub(crate) unsafe fn packed_dot<const NATIVE: bool>(
    weights: *const u32, input: *const f32, row: usize, lane: usize,
    rows: usize, width: u32,
) -> f32 {
    packed_dot_view::<NATIVE>(weights, input, row, lane, rows, width, width, 0)
}


#[inline(always)]
unsafe fn packed_dot_view<const NATIVE: bool>(
    weights: *const u32, input: *const f32, row: usize, lane: usize,
    rows: usize, width: u32, stride: u32, row_offset: usize,
) -> f32 {
    let ktiles = width / 16;
    let q = (lane % 4 * 2) as u32;
    let stored_row = row + row_offset;
    let base = stored_row / 8 * (stride / 16) as usize * 32 + stored_row % 8 * 4 + lane % 4;
    let mut s = [0.0f32; 8];
    let mut tile = 0u32;
    if row < rows {
        while tile + 1 < ktiles {
            for part in 0..2u32 {
                let v = packed_weight(weights, base + (tile + part) as usize * 32);
                let col = ((tile + part) * 16 + q) as usize;
                let a = (*input.add(col).cast::<Pair>()).0;
                let b = (*input.add(col + 8).cast::<Pair>()).0;
                let p = part as usize * 4;
                s[p] += bf16_to_f32(v[0] as u16) * operand::<NATIVE>(f32::from_bits(a[0]));
                s[p + 1] += bf16_to_f32((v[0] >> 16) as u16) * operand::<NATIVE>(f32::from_bits(a[1]));
                s[p + 2] += bf16_to_f32(v[1] as u16) * operand::<NATIVE>(f32::from_bits(b[0]));
                s[p + 3] += bf16_to_f32((v[1] >> 16) as u16) * operand::<NATIVE>(f32::from_bits(b[1]));
            }
            tile += 2;
        }
        if tile < ktiles {
            let v = packed_weight(weights, base + tile as usize * 32);
            let col = (tile * 16 + q) as usize;
            let a = (*input.add(col).cast::<Pair>()).0;
            let b = (*input.add(col + 8).cast::<Pair>()).0;
            s[0] += bf16_to_f32(v[0] as u16) * operand::<NATIVE>(f32::from_bits(a[0]));
            s[1] += bf16_to_f32((v[0] >> 16) as u16) * operand::<NATIVE>(f32::from_bits(a[1]));
            s[2] += bf16_to_f32(v[1] as u16) * operand::<NATIVE>(f32::from_bits(b[0]));
            s[3] += bf16_to_f32((v[1] >> 16) as u16) * operand::<NATIVE>(f32::from_bits(b[1]));
        }
    }
    reduce_packed(s)
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn gate_up_swiglu_bf16_packed(
    gate: *const u32, up: *const u32, input: *const f32, output: *mut f32,
    rows: u32, width: u32,
) { gate_up_packed::<true>(gate, up, input, output, rows, width); }


#[inline(always)]
unsafe fn gate_up_packed<const NATIVE: bool>(
    gate: *const u32, up: *const u32, input: *const f32, output: *mut f32,
    rows: u32, width: u32,
) {
    let precision = u32::from(NATIVE);
    if NATIVE && width == 5120 && rows % 8 == 0 && _block_dim_x() == 64 {
        gate_up_staged::<5120, true>(gate, up, input, output, rows, precision);
        return;
    }
    if (width == 1024 || width == 4096) && rows % 8 == 0 && _block_dim_x() == 64 {
        if width == 1024 { gate_up_staged::<1024, NATIVE>(gate, up, input, output, rows, precision); }
        else { gate_up_staged::<4096, NATIVE>(gate, up, input, output, rows, precision); }
        return;
    }
    let tid = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (row, lane) = (tid / 4, tid % 32);
    let g = packed_dot::<NATIVE>(gate, input, row, lane, rows as usize, width);
    let u = packed_dot::<NATIVE>(up, input, row, lane, rows as usize, width);
    if lane % 4 == 0 && row < rows as usize {
        *output.add(row) = swiglu(g, u, precision);
    }
}


#[inline(always)]
unsafe fn verify_stage<const RT: u32>(
    sh: u32,
    w: *const u32,
    row_tile: u32,
    kt: u32,
    col: u32,
    tid: u32,
    threads: u32,
) {
    let mut q = tid;
    while q < 512 * RT {
        crate::ptx::copy16(
            sh + q * 16,
            w.add(
                (((row_tile * RT + q / 512) * kt + col + q % 512 / 16) * 64 + q % 16 * 4) as usize,
            ),
        );
        q += threads;
    }
}


#[inline(always)]
unsafe fn verify_shared<const RT: u32, const BYTES: usize>(
    w: *const u32,
    x: *const f32,
    y: *mut f32,
    n: u32,
    k: u32,
    t: u32,
) {
    let tid = _thread_idx_x() as u32;
    let threads = _block_dim_x() as u32;
    let (lane, warp) = (tid % 32, tid / 32);
    let rt = _block_idx_x() as u32;
    let row = rt * 8 * RT + warp % RT * 8 + lane / 4;
    let token = core::arch::nvptx::_block_idx_y() as u32 * (threads / 32 / RT) + warp / RT;
    let kt = k / 16;
    let q = lane % 4 * 2;
    let sh = crate::ptx::shared::<BYTES>();
    let mut s = [0f32; 8];
    let mut c = 0;
    verify_stage::<RT>(sh, w, rt, kt, 0, tid, threads);
    crate::ptx::wait();
    crate::ptx::sync();
    while c < kt {
        let current = c / 32 % 2 * 8192 * RT + warp % RT * 8192;
        let next = (c / 32 + 1) % 2 * 8192 * RT;
        if c + 32 < kt {
            verify_stage::<RT>(sh + next, w, rt, kt, c + 32, tid, threads);
        }
        if token < t {
            let mut j = 0;
            while j < 32 {
                let a = crate::ptx::load2(sh + current + j * 256 + lane * 8);
                let b = crate::ptx::load2(sh + current + (j + 1) * 256 + lane * 8);
                let col = (c + j) * 16 + q;
                let xp = x.add((token * k + col) as usize);
                s[0] += bf16_to_f32(a[0] as u16) * activation(*xp);
                s[1] += bf16_to_f32((a[0] >> 16) as u16) * activation(*xp.add(1));
                s[2] += bf16_to_f32(a[1] as u16) * activation(*xp.add(8));
                s[3] += bf16_to_f32((a[1] >> 16) as u16) * activation(*xp.add(9));
                s[4] += bf16_to_f32(b[0] as u16) * activation(*xp.add(16));
                s[5] += bf16_to_f32((b[0] >> 16) as u16) * activation(*xp.add(17));
                s[6] += bf16_to_f32(b[1] as u16) * activation(*xp.add(24));
                s[7] += bf16_to_f32((b[1] >> 16) as u16) * activation(*xp.add(25));
                j += 2;
            }
        }
        crate::ptx::wait();
        crate::ptx::sync();
        c += 32;
    }
    let mut sum = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    sum += f32::from_bits(shuffle_xor(u32::MAX, sum.to_bits(), 2, 31));
    sum += f32::from_bits(shuffle_xor(u32::MAX, sum.to_bits(), 1, 31));
    if lane % 4 == 0 && row < n && token < t {
        *y.add((token * n + row) as usize) = sum;
    }
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn verify_bf16_f32_shared(
    w: *const u32,
    x: *const f32,
    y: *mut f32,
    n: u32,
    k: u32,
    t: u32,
) {
    verify_shared::<1, 16384>(w, x, y, n, k, t);
}


#[inline(always)]
unsafe fn stage_verify<const RT: usize, const CK: usize, const GROUPS: usize>(
    sh: u32,
    w: *const u32,
    x: *const f32,
    rt: usize,
    n: usize,
    token: usize,
    t: usize,
    k: usize,
    c: usize,
    tid: usize,
    threads: usize,
) {
    let mut i = tid;
    while i < RT * CK * 16 {
        if rt * RT * 8 + i / (CK * 16) * 8 < n { crate::ptx::copy16(
            sh + (i * 16) as u32,
            w.add(((rt * RT + i / (CK * 16)) * (k / 16) + c + i % (CK * 16) / 16) * 64 + i % 16 * 4),
        ); } else {
            for word in 0..4 { crate::ptx::store(sh + (i * 16 + word * 4) as u32, 0.0); }
        }
        i += threads;
    }
    i = tid;
    while i < GROUPS * (threads / 32) * CK * 4 {
        if token + i / (CK * 4) < t {
            crate::ptx::copy16(
                sh + (RT * CK * 256 + i * 16) as u32,
                x.add((token + i / (CK * 4)) * k + c * 16 + i % (CK * 4) * 4)
                    .cast(),
            );
        }
        i += threads;
    }
}


#[inline(always)]
pub(crate) unsafe fn staged_verify<const RT: usize, const CK: usize, const BYTES: usize, const NATIVE: bool, const GROUPS: usize>(
    w: *const u32,
    x: *const f32,
    y: *mut f32,
    n: u32,
    k: u32,
    t: u32,
    token0: usize,
) {
    let tid = _thread_idx_x() as usize;
    let threads = _block_dim_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let rt = core::arch::nvptx::_block_idx_y() as usize;
    let sh = if BYTES > 49152 { 0 } else { crate::ptx::shared::<BYTES>() };
    let stride = RT * CK * 256 + GROUPS * threads / 32 * CK * 64;
    let mut s = [[[0f32; 8]; RT]; GROUPS];
    let mut c = 0;
    stage_verify::<RT, CK, GROUPS>(sh, w, x, rt, n as usize, token0, t as usize, k as usize, 0, tid, threads);
    crate::ptx::wait();
    crate::ptx::sync();
    while c < k as usize / 16 {
        let cur = sh + ((c / CK % 2) * stride) as u32;
        if c + CK < k as usize / 16 {
            stage_verify::<RT, CK, GROUPS>(
                sh + (((c / CK + 1) % 2) * stride) as u32,
                w,
                x,
                rt,
                n as usize,
                token0,
                t as usize,
                k as usize,
                c + CK,
                tid,
                threads,
            );
        }
        if GROUPS == 1 { rows_unroll!(GROUPS, group, {
          let token = token0 + warp + group * (threads / 32);
          if token < t as usize {
            let mut part = 0;
            while part < CK / 2 {
                let j = part * 2;
                let xp = cur + (RT * CK * 256 + (warp + group * (threads / 32)) * CK * 64 + (j * 16 + lane % 4 * 2) * 4) as u32;
                let a = crate::ptx::load2(xp);
                let b = crate::ptx::load2(xp + 32);
                let cc = crate::ptx::load2(xp + 64);
                let dd = crate::ptx::load2(xp + 96);
                let xv = [a[0], a[1], b[0], b[1], cc[0], cc[1], dd[0], dd[1]];
                rows_unroll!(RT, r, {
                    let p = cur + ((r * CK + j) * 256 + lane * 8) as u32;
                    let a = crate::ptx::load2(p);
                    let b = crate::ptx::load2(p + 256);
                    let ww = [
                        a[0] as u16,
                        (a[0] >> 16) as u16,
                        a[1] as u16,
                        (a[1] >> 16) as u16,
                        b[0] as u16,
                        (b[0] >> 16) as u16,
                        b[1] as u16,
                        (b[1] >> 16) as u16,
                    ];
                    for z in 0..8 {
                        let weight = bf16_to_f32(ww[z]);
                        let input = operand::<NATIVE>(f32::from_bits(xv[z]));
                        // Promoted BF16 products are exact in FP32; fusion
                        // retains these accumulator boundaries and their order.
                        if NATIVE { s[group][r][z] = core::f32::math::mul_add(weight, input, s[group][r][z]); }
                        else { s[group][r][z] += weight * input; }
                    }
                });
                part += 1;
            }
          }
        }); } else {
            let mut part = 0;
            while part < CK / 2 {
                let j = part * 2;
                rows_unroll!(RT, r, {
                    let p = cur + ((r * CK + j) * 256 + lane * 8) as u32;
                    let a = crate::ptx::load2(p);
                    let b = crate::ptx::load2(p + 256);
                    let ww = [bf16_to_f32(a[0] as u16), bf16_to_f32((a[0] >> 16) as u16),
                        bf16_to_f32(a[1] as u16), bf16_to_f32((a[1] >> 16) as u16),
                        bf16_to_f32(b[0] as u16), bf16_to_f32((b[0] >> 16) as u16),
                        bf16_to_f32(b[1] as u16), bf16_to_f32((b[1] >> 16) as u16)];
                    // Keep each canonical accumulator's K order while sharing
                    // loaded weights across several independent input rows.
                    rows_unroll!(GROUPS, group, {
                        let token = token0 + warp + group * (threads / 32);
                        if token < t as usize {
                            let xp = cur + (RT * CK * 256 + (warp + group * (threads / 32)) * CK * 64 + (j * 16 + lane % 4 * 2) * 4) as u32;
                            let a = crate::ptx::load2(xp);
                            let b = crate::ptx::load2(xp + 32);
                            let cc = crate::ptx::load2(xp + 64);
                            let dd = crate::ptx::load2(xp + 96);
                            let xv = [a[0], a[1], b[0], b[1], cc[0], cc[1], dd[0], dd[1]];
                            for z in 0..8 {
                                let input = operand::<NATIVE>(f32::from_bits(xv[z]));
                                if NATIVE { s[group][r][z] = core::f32::math::mul_add(ww[z], input, s[group][r][z]); }
                                else { s[group][r][z] += ww[z] * input; }
                            }
                        }
                    });
                });
                part += 1;
            }
        }
        crate::ptx::wait();
        crate::ptx::sync();
        c += CK;
    }
    rows_unroll!(GROUPS, group, {
      let token = token0 + warp + group * (threads / 32);
      rows_unroll!(RT, r, {
        let a = s[group][r];
        let mut v = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        v += f32::from_bits(shuffle_xor(u32::MAX, v.to_bits(), 2, 31));
        v += f32::from_bits(shuffle_xor(u32::MAX, v.to_bits(), 1, 31));
        let row = rt * RT * 8 + r * 8 + lane / 4;
        if lane % 4 == 0 && row < n as usize && token < t as usize {
            *y.add(token * n as usize + row) = if NATIVE { activation(v) } else { v };
        }
      });
    });
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn verify_bf16_f32_staged64(
    w: *const u32,
    x: *const f32,
    y: *mut f32,
    n: u32,
    k: u32,
    t: u32,
) {
    staged_verify::<8, 8, 49152, false, 1>(w, x, y, n, k, t,
        _block_idx_x() as usize * (_block_dim_x() as usize / 32));
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_bf16_multi(p: DecodeProjection) {
    staged_verify::<2, 8, 24576, true, 1>(p.weights as *const u32,
        p.input as *const f32, p.output as *mut f32, p.rows, p.width, p.tokens, 0);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_bf16_multi32(p: DecodeProjection) {
    staged_verify::<3, 8, 49152, true, 1>(p.weights as *const u32,
        p.input as *const f32, p.output as *mut f32, p.rows, p.width, p.tokens, 0);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_bf16_multi80(p: DecodeProjection) {
    macro_rules! run { ($groups:expr, $bytes:expr) => {
        staged_verify::<2, 8, $bytes, true, $groups>(p.weights as *const u32,
            p.input as *const f32, p.output as *mut f32, p.rows, p.width, p.tokens, 0)
    }; }
    match p.tokens {
        0..=24 => run!(3, 98304),
        25..=32 => run!(4, 98304),
        33..=48 => run!(6, 98304),
        49..=64 => run!(8, 98304),
        _ => run!(10, 98304),
    }
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_bf16_multi64(p: DecodeProjection) {
    staged_verify::<8, 16, 98304, true, 1>(p.weights as *const u32,
        p.input as *const f32, p.output as *mut f32, p.rows, p.width, p.tokens, 0);
}
