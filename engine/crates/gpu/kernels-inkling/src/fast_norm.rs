//! Zero-stack bit-exact Inkling norm kernel (`FAST_NORM = 109`).
//!
//! Computes `PlainNorm` (32-lane tree `row_squares`, 1 warp per row = 4 rows/block, zero smem/barrier)
//! and `RoundedNorm` (chunked 128-element `norm_squares`: 1 warp/row for `dim == 128`, or 4 warps/row
//! for `dim % 128 == 0` with all 128 threads active during reduction and exact `norm_join` in smem).
//! Every output word is bit-for-bit identical to `block_stream::unit`.

use crate::block_ops::{LANES, NORM_CHUNK, root};
use crate::fast_ops::{
    KV_FINISH_HEAD_DIM, NORM_CHUNKS_MAX, NORM_MODE_KV_DIRECT, NORM_MODE_SHORT_CONV_DIRECT,
    head_clip_admitted, kv_finish_direct_admitted, norm_admitted, short_conv_direct_admitted,
};
use crate::fixed_math::rn;
use crate::tc_ptx::{ld_global_nc_v4, load_u32, shfl_xor_f32, store_u32};

const SMEM_BYTES: usize = (NORM_CHUNKS_MAX + 4) * 4; // 144 bytes
/// Chunks of one warp in the block of a row (4 warps share at most `NORM_CHUNKS_MAX` chunks).
const PER_WARP: usize = NORM_CHUNKS_MAX / 4;
/// Places of a lane whose loads go out together in the plain norm.
const GROUP: usize = 8;
/// The widest plain row that one block stages in shared words (`STAGE_WORDS * 4` bytes) when the launch has one row.
const STAGE_WORDS: usize = 4096;
const _: () = assert!(PER_WARP * 4 == NORM_CHUNKS_MAX);

/// Replay one recorded command of kind `FAST_NORM`.
/// SAFETY (caller contract, `resident_launch_fast_norm`): `table[index]` is a valid `FAST_NORM`
/// command recorded by `Backend::fast_norm_into` or `Backend::fast_head_clip_into` after operand checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - Host verified via `fast_ops::check_norm` that `input` and `output` hold `rows * dim` FP32 words,
    //   optional `weight` holds `dim` FP32 words, `output` is disjoint from inputs, and all are 4-byte aligned.
    // - Device re-checks `norm_admitted` and alignment before dereferencing.
    // - Head clip mode has `rows = tokens * heads` and optional weight and factor arrays of
    //   `heads * dim` and `tokens` words. The host checks their extents, alignment and disjoint output.
    // - The one-row plain form copies `dim / 4` 16-byte groups (`dim <= STAGE_WORDS`, a multiple of 32; the input
    //   16-byte aligned, checked here) into the stage, each group by one thread. All threads wait and reach a barrier
    //   before any read; it reads only words below `dim`, and writes output words `0..dim` once each (`rows == 1`).
    // - Shared memory writes (`0..chunks` and slot 32) lie strictly inside `SMEM_BYTES` (144 bytes);
    //   all 128 threads reach every `bar.sync 0` and all 32 lanes of each warp reach every `shfl.sync.bfly.b32`.
    // - Wide rounded prefill rows are a multiple of 128. The vector epilogue also checks
    //   that all non-null bases are 16-byte aligned before accessing four FP32 words.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FAST_NORM { core::arch::nvptx::trap(); }
        // The next launch may start now (griddepcontrol.launch_dependents): a launch that the host marks dependent
        // (`resident_launch::dependent_launch`, a decode panel product right after this norm) reads only its panels
        // before its griddepcontrol.wait, which waits for this whole grid. This launch is not itself dependent, so every
        // launch before it has ended.
        core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
        let w = c.payload.as_ptr().cast::<u64>();
        let mode = *w.add(5);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid >> 5, tid & 31);

        if mode == NORM_MODE_KV_DIRECT {
            // SAFETY: host checked via `fast_ops::check_kv_finish_direct` that all inputs, fixed/made taps,
            // per-chat rings, per-chat plane row slots, and `every` buffers are valid, word-aligned, and
            // pairwise disjoint. Each warp handles one `(is_val, row, head)` with 128 channels (4/lane).
            let rows = *w.add(3) as usize;
            let kv_heads = *w.add(4) as usize;
            let taps = *w.add(8) as usize;
            if !kv_finish_direct_admitted(rows, kv_heads, taps, threads) { core::arch::nvptx::trap(); }
            let total = 2 * rows * kv_heads;
            if core::arch::nvptx::_grid_dim_x() as usize != total.div_ceil(4) { core::arch::nvptx::trap(); }
            let unit = block * 4 + warp;
            if unit >= total { return; }
            let is_val = unit >= rows * kv_heads;
            let local = if is_val { unit - rows * kv_heads } else { unit };
            let (row, head) = (local / kv_heads, local % kv_heads);
            let inputs = *w.add(if is_val { 1 } else { 0 }) as *const f32;
            let fixed = *w.add(if is_val { 9 } else { 2 }) as *const f32;
            let made = *w.add(if is_val { 11 } else { 10 }) as *const f32;
            let every_out = *w.add(if is_val { 13 } else { 12 }) as *mut f32;
            let ring = *w.add(16 + 4 * row + usize::from(is_val)) as *const f32;
            let plane_row = *w.add(16 + 4 * row + 2 + usize::from(is_val)) as *mut f32;
            let own = ((*w.add(14) >> (4 * row)) & 0xf) as usize;
            if own >= taps || inputs.is_null() || fixed.is_null() || every_out.is_null() || ring.is_null() || plane_row.is_null()
                || ((inputs as usize) | (fixed as usize) | (made as usize) | (every_out as usize) | (ring as usize) | (plane_row as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let shift = if is_val { 32 } else { 0 };
            let param = f32::from_bits((*w.add(6) >> shift) as u32);
            let conv_scale = f32::from_bits((*w.add(7) >> shift) as u32);
            let width = kv_heads * KV_FINISH_HEAD_DIM;
            let tok_ptr = if made.is_null() { made } else { made.add((row * kv_heads + head) * taps) };
            let in_ptr = inputs.add(row * width);
            let ev_base = every_out.add(row * taps * width);
            let (y0, y1, y2, y3) = if taps == 4 && (fixed as usize) & 15 == 0 {
                let (tk0, tk1, tk2, tk3) = if tok_ptr.is_null() {
                    (0.0f32, 0.0f32, 0.0f32, 0.0f32)
                } else {
                    (rn(*tok_ptr), rn(*tok_ptr.add(1)), rn(*tok_ptr.add(2)), rn(*tok_ptr.add(3)))
                };
                let (s1, s2, s3) = ((own + 3) & 3, (own + 2) & 3, (own + 1) & 3);
                let (r1, r2, r3) = (ring.add(s1 * width), ring.add(s2 * width), ring.add(s3 * width));
                let (e0, e1, e2, e3) = (ev_base.add(own * width), ev_base.add(s1 * width), ev_base.add(s2 * width), ev_base.add(s3 * width));
                macro_rules! conv4 {($k:literal)=>{{
                    let ch = head * KV_FINISH_HEAD_DIM + $k * LANES + lane;
                    let ft = ld_global_nc_v4(fixed.add(ch * 4));
                    let x3 = *r3.add(ch);
                    let x2 = *r2.add(ch);
                    let x1 = *r1.add(ch);
                    let x0 = *in_ptr.add(ch);
                    *e3.add(ch) = x3;
                    *e2.add(ch) = x2;
                    *e1.add(ch) = x1;
                    *e0.add(ch) = x0;
                    let mut sum = 0.0f32;
                    sum += (if tok_ptr.is_null() { ft[3] } else { ft[3] + tk3 }) * rn(x3);
                    sum += (if tok_ptr.is_null() { ft[2] } else { ft[2] + tk2 }) * rn(x2);
                    sum += (if tok_ptr.is_null() { ft[1] } else { ft[1] + tk1 }) * rn(x1);
                    let cur = rn(x0);
                    sum += (if tok_ptr.is_null() { ft[0] } else { ft[0] + tk0 }) * cur;
                    rn((sum + cur) * conv_scale)
                }};}
                (conv4!(0), conv4!(1), conv4!(2), conv4!(3))
            } else {
                macro_rules! conv_g {($k:literal)=>{{
                    let ch = head * KV_FINISH_HEAD_DIM + $k * LANES + lane;
                    let mut sum = 0.0f32;
                    let mut lag = taps;
                    while lag > 0 {
                        lag -= 1;
                        let mut tap = *fixed.add(ch * taps + lag);
                        if !tok_ptr.is_null() { tap += rn(*tok_ptr.add(lag)); }
                        let s_idx = if own >= lag { own - lag } else { own + taps - lag };
                        let val = if lag == 0 { *in_ptr.add(ch) } else { *ring.add(s_idx * width + ch) };
                        *ev_base.add(s_idx * width + ch) = val;
                        sum += tap * rn(val);
                    }
                    let cur = rn(*in_ptr.add(ch));
                    rn((sum + cur) * conv_scale)
                }};}
                (conv_g!(0), conv_g!(1), conv_g!(2), conv_g!(3))
            };
            let mut sq = 0.0f32;
            sq += y0 * y0;
            sq += y1 * y1;
            sq += y2 * y2;
            sq += y3 * y3;
            sq += shfl_xor_f32(sq, 16);
            sq += shfl_xor_f32(sq, 8);
            sq += shfl_xor_f32(sq, 4);
            sq += shfl_xor_f32(sq, 2);
            sq += shfl_xor_f32(sq, 1);
            let scale = if is_val {
                1.0f32 / root(sq / 128.0f32 + param)
            } else {
                let bits: u32;
                core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",
                    got = out(reg32) bits, v = in(reg32) sq.to_bits());
                crate::block_attention::clip_factor(root(f32::from_bits(bits)), param)
            };
            let out_base = plane_row.add(head * 64);
            macro_rules! pack_ch {($k:literal, $yk:expr)=>{{
                let val = rn($yk * scale);
                let hi = shfl_xor_f32(val, 1);
                if (lane & 1) == 0 {
                    let lb = val.to_bits();
                    let hb = hi.to_bits();
                    let lo16 = lb.wrapping_add(0x7fff + ((lb >> 16) & 1)) >> 16;
                    let hi16 = hb.wrapping_add(0x7fff + ((hb >> 16) & 1)) >> 16;
                    *out_base.add($k * 16 + (lane >> 1)) = f32::from_bits(lo16 | (hi16 << 16));
                }
            }};}
            pack_ch!(0, y0);
            pack_ch!(1, y1);
            pack_ch!(2, y2);
            pack_ch!(3, y3);
            return;
        }

        if mode == NORM_MODE_SHORT_CONV_DIRECT {
            // SAFETY: host checked via `fast_ops::check_short_conv_direct` that inputs, fixed/token taps,
            // addend, per-chat rings, output, and `every` are valid, word-aligned, and disjoint.
            let inputs = *w as *const f32;
            let fixed = *w.add(1) as *const f32;
            let output = *w.add(2) as *mut f32;
            let rows = *w.add(3) as usize;
            let dim = *w.add(4) as usize;
            let bound = f32::from_bits(*w.add(6) as u32);
            let heads = *w.add(7) as usize;
            let taps = *w.add(8) as usize;
            let form = *w.add(9) as u32;
            let token_taps = *w.add(10) as *const f32;
            let addend = *w.add(11) as *const f32;
            let every_out = *w.add(12) as *mut f32;
            if !short_conv_direct_admitted(rows, heads, dim, taps, form, threads) { core::arch::nvptx::trap(); }
            let width = heads * dim;
            let row = core::arch::nvptx::_block_idx_y() as usize;
            if core::arch::nvptx::_grid_dim_x() as usize != width.div_ceil(threads)
                || core::arch::nvptx::_grid_dim_y() as usize != rows || row >= rows
            {
                core::arch::nvptx::trap();
            }
            let ch = block * threads + tid;
            if ch >= width { return; }
            let ring = *w.add(16 + 2 * row) as *const f32;
            let own = *w.add(16 + 2 * row + 1) as usize;
            if own >= taps || inputs.is_null() || fixed.is_null() || output.is_null() || every_out.is_null() || ring.is_null()
                || ((inputs as usize) | (fixed as usize) | (output as usize) | (every_out as usize) | (ring as usize) | (token_taps as usize) | (addend as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let has_tok = (form & 1) != 0;
            let has_add = (form & 2) != 0;
            let round_sum = (form & 4) != 0;
            let head = if heads == 1 { 0 } else { ch / dim };
            let tok_ptr = if has_tok { token_taps.add((row * heads + head) * taps) } else { core::ptr::null() };
            let cur_raw = *inputs.add(row * width + ch);
            let ev_base = every_out.add(row * taps * width + ch);
            let mut sum = 0.0f32;
            if taps == 4 && (fixed as usize) & 15 == 0 {
                let ft = ld_global_nc_v4(fixed.add(ch * 4));
                let (s1, s2, s3) = ((own + 3) & 3, (own + 2) & 3, (own + 1) & 3);
                let x3 = *ring.add(s3 * width + ch);
                let x2 = *ring.add(s2 * width + ch);
                let x1 = *ring.add(s1 * width + ch);
                *ev_base.add(s3 * width) = x3;
                *ev_base.add(s2 * width) = x2;
                *ev_base.add(s1 * width) = x1;
                *ev_base.add(own * width) = cur_raw;
                if has_tok {
                    sum += (ft[3] + rn(*tok_ptr.add(3))) * rn(x3);
                    sum += (ft[2] + rn(*tok_ptr.add(2))) * rn(x2);
                    sum += (ft[1] + rn(*tok_ptr.add(1))) * rn(x1);
                    sum += (ft[0] + rn(*tok_ptr)) * rn(cur_raw);
                } else {
                    sum += ft[3] * rn(x3);
                    sum += ft[2] * rn(x2);
                    sum += ft[1] * rn(x1);
                    sum += ft[0] * rn(cur_raw);
                }
            } else {
                let mut lag = taps;
                while lag > 0 {
                    lag -= 1;
                    let mut tap = *fixed.add(ch * taps + lag);
                    if has_tok { tap += rn(*tok_ptr.add(lag)); }
                    let s_idx = if own >= lag { own - lag } else { own + taps - lag };
                    let val = if lag == 0 { cur_raw } else { *ring.add(s_idx * width + ch) };
                    *ev_base.add(s_idx * width) = val;
                    sum += tap * rn(val);
                }
            }
            let y = rn((sum + rn(cur_raw)) * bound);
            let out_val = if !has_add {
                y
            } else if !round_sum {
                *addend.add(row * width + ch) + y
            } else {
                rn(*addend.add(row * width + ch) + y)
            };
            *output.add(row * width + ch) = out_val;
            return;
        }

        let input = *w as *const f32;
        let weight = *w.add(1) as *const f32;
        let output = *w.add(2) as *mut f32;
        let rows = *w.add(3) as usize;
        let dim = *w.add(4) as usize;
        let rounded = mode == 1;
        let eps = f32::from_bits(*w.add(6) as u32);
        let admitted = if mode == 2 { head_clip_admitted(rows, dim, threads) } else { norm_admitted(rows, dim, rounded, threads) };
        if !admitted { core::arch::nvptx::trap(); }
        if ((input as usize) | (weight as usize) | (output as usize)) & 3 != 0 { core::arch::nvptx::trap(); }

        if mode == 2 {
            let heads = *w.add(7) as usize;
            let factors = *w.add(8) as *const f32;
            if heads == 0 || !rows.is_multiple_of(heads) || (factors as usize) & 3 != 0 {
                core::arch::nvptx::trap();
            }
            let row = block * 4 + warp;
            if row >= rows { return; }
            let x_ptr = input.add(row * dim);
            let o_ptr = output.add(row * dim);
            let w_ptr = if weight.is_null() { weight } else { weight.add((row % heads) * dim) };
            let value = |j| {
                let v = rn(*x_ptr.add(j));
                if w_ptr.is_null() { v } else { v * *w_ptr.add(j) }
            };
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let v = value(j);
                sum += v * v;
                j += LANES;
            }
            sum += shfl_xor_f32(sum, 16);
            sum += shfl_xor_f32(sum, 8);
            sum += shfl_xor_f32(sum, 4);
            sum += shfl_xor_f32(sum, 2);
            sum += shfl_xor_f32(sum, 1);
            let bits: u32;
            core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",
                got = out(reg32) bits, v = in(reg32) sum.to_bits());
            let scale = crate::block_attention::clip_factor(root(f32::from_bits(bits)), eps);
            let position = if factors.is_null() { 0.0 } else { *factors.add(row / heads) };
            j = lane;
            while j < dim {
                let clipped = rn(value(j) * scale);
                *o_ptr.add(j) = if factors.is_null() { clipped } else { rn(clipped * position) };
                j += LANES;
            }
            return;
        }

        if !rounded && rows == 1 && dim <= STAGE_WORDS && (input as usize) & 15 == 0 {
            // PlainNorm of one row: the four warps copy the row into shared words (16 bytes a copy, every copy issued
            // before the one wait), warp 0 runs the lane chains from there (lane `l`: places l, l + 32, .. in rising
            // order, as below), and every thread writes its places. The same chains, join, scale and products.
            let stage = crate::ptx::shared::<{ STAGE_WORDS * 4 }>();
            let smem = crate::ptx::shared::<SMEM_BYTES>();
            let mut q = tid;
            while q < dim / 4 { crate::ptx::copy16(stage + (16 * q) as u32, input.add(4 * q).cast::<u32>()); q += threads; }
            crate::ptx::commit();
            crate::ptx::wait();
            core::arch::nvptx::_syncthreads();
            if warp == 0 {
                let mut sum = 0.0f32;
                let mut j = lane;
                while j < dim { let v = f32::from_bits(load_u32(stage + (4 * j) as u32)); sum += v * v; j += LANES; }
                sum += shfl_xor_f32(sum, 16);
                sum += shfl_xor_f32(sum, 8);
                sum += shfl_xor_f32(sum, 4);
                sum += shfl_xor_f32(sum, 2);
                sum += shfl_xor_f32(sum, 1);
                if lane == 0 { store_u32(smem + (NORM_CHUNKS_MAX * 4) as u32, (1.0f32 / root(sum / dim as f32 + eps)).to_bits()); }
            }
            core::arch::nvptx::_syncthreads();
            let scale = f32::from_bits(load_u32(smem + (NORM_CHUNKS_MAX * 4) as u32));
            let mut j = tid;
            while j < dim { *output.add(j) = f32::from_bits(load_u32(stage + (4 * j) as u32)) * scale; j += threads; }
            return;
        }
        if !rounded {
            // PlainNorm: 1 warp per row (4 rows per 128-thread block), exact 32-lane tree `row_squares`.
            let row = block * 4 + warp;
            if row >= rows { return; }
            let x_ptr = input.add(row * dim);
            let o_ptr = output.add(row * dim);
            // The lane's chain over places lane, lane + 32, ... in rising order. The loads of a group of `GROUP`
            // places are issued before its adds, so the chain does not wait a load at each place.
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let mut v = [0.0f32; GROUP];
                let mut g = 0;
                while g < GROUP { if j + g * LANES < dim { v[g] = *x_ptr.add(j + g * LANES); } g += 1; }
                g = 0;
                while g < GROUP { if j + g * LANES < dim { sum += v[g] * v[g]; } g += 1; }
                j += GROUP * LANES;
            }
            // Exact binary tree join: distances 16, 8, 4, 2, 1 (same order as `block_ops::join`).
            sum += shfl_xor_f32(sum, 16);
            sum += shfl_xor_f32(sum, 8);
            sum += shfl_xor_f32(sum, 4);
            sum += shfl_xor_f32(sum, 2);
            sum += shfl_xor_f32(sum, 1);
            let scale = 1.0f32 / root(sum / dim as f32 + eps);
            // The output is disjoint from the input (host check): a group's loads go before its stores.
            j = lane;
            while j < dim {
                let mut v = [0.0f32; GROUP];
                let mut g = 0;
                while g < GROUP { if j + g * LANES < dim { v[g] = *x_ptr.add(j + g * LANES); } g += 1; }
                g = 0;
                while g < GROUP { if j + g * LANES < dim { *o_ptr.add(j + g * LANES) = v[g] * scale; } g += 1; }
                j += GROUP * LANES;
            }
            return;
        }

        if dim == NORM_CHUNK {
            // RoundedNorm with dim == 128: 1 warp per row (4 rows per block), zero smem, zero barrier!
            let row = block * 4 + warp;
            if row >= rows { return; }
            let x_ptr = input.add(row * NORM_CHUNK);
            let o_ptr = output.add(row * NORM_CHUNK);
            let v0 = *x_ptr.add(lane);
            let v1 = *x_ptr.add(lane + 32);
            let v2 = *x_ptr.add(lane + 64);
            let v3 = *x_ptr.add(lane + 96);
            let mut sum = 0.0f32;
            sum += v0 * v0;
            sum += v1 * v1;
            sum += v2 * v2;
            sum += v3 * v3;
            sum += shfl_xor_f32(sum, 16);
            sum += shfl_xor_f32(sum, 8);
            sum += shfl_xor_f32(sum, 4);
            sum += shfl_xor_f32(sum, 2);
            sum += shfl_xor_f32(sum, 1);
            let scale = 1.0f32 / root(sum / 128.0f32 + eps);
            if weight.is_null() {
                *o_ptr.add(lane) = rn(v0 * scale);
                *o_ptr.add(lane + 32) = rn(v1 * scale);
                *o_ptr.add(lane + 64) = rn(v2 * scale);
                *o_ptr.add(lane + 96) = rn(v3 * scale);
            } else {
                *o_ptr.add(lane) = rn((v0 * scale) * *weight.add(lane));
                *o_ptr.add(lane + 32) = rn((v1 * scale) * *weight.add(lane + 32));
                *o_ptr.add(lane + 64) = rn((v2 * scale) * *weight.add(lane + 64));
                *o_ptr.add(lane + 96) = rn((v3 * scale) * *weight.add(lane + 96));
            }
            return;
        }

        // RoundedNorm with dim = chunks * 128 (e.g., 2304 = 18 chunks): 1 block (4 warps) per row. Warp w owns the
        // chunks w, w + 4, ... (at most `PER_WARP`); each lane keeps its four values of each of them, and their weights,
        // in registers from the sum to the store in decode. Prefill retains its aligned vector loads and stores
        // after the join. Both use the same chunk sums and `rn((x * scale) * w)` (or `rn(x * scale)`).
        if block >= rows { return; }
        let x_ptr = input.add(block * dim);
        let o_ptr = output.add(block * dim);
        let chunks = dim / NORM_CHUNK;
        let smem = crate::ptx::shared::<SMEM_BYTES>();
        let weighted = !weight.is_null();
        let vector_store = *w.add(9) != 0 && ((input as usize) | (weight as usize) | (output as usize)) & 15 == 0;
        let mut v = [0.0f32; 4 * PER_WARP];
        let mut wv = [0.0f32; 4 * PER_WARP];
        let mut i = 0;
        while i < PER_WARP {
            let ch = warp + 4 * i;
            if ch < chunks {
                let base = ch * NORM_CHUNK + lane;
                let mut g = 0;
                while g < 4 { v[4 * i + g] = *x_ptr.add(base + g * 32); g += 1; }
                if weighted && !vector_store { g = 0; while g < 4 { wv[4 * i + g] = *weight.add(base + g * 32); g += 1; } }
            }
            i += 1;
        }
        i = 0;
        while i < PER_WARP {
            // `ch < chunks` is the same on every lane of the warp: all 32 lanes reach each shuffle.
            let ch = warp + 4 * i;
            if ch < chunks {
                let mut sum = 0.0f32;
                sum += v[4 * i] * v[4 * i];
                sum += v[4 * i + 1] * v[4 * i + 1];
                sum += v[4 * i + 2] * v[4 * i + 2];
                sum += v[4 * i + 3] * v[4 * i + 3];
                sum += shfl_xor_f32(sum, 16);
                sum += shfl_xor_f32(sum, 8);
                sum += shfl_xor_f32(sum, 4);
                sum += shfl_xor_f32(sum, 2);
                sum += shfl_xor_f32(sum, 1);
                if lane == 0 {
                    store_u32(smem + (ch * 4) as u32, sum.to_bits());
                }
            }
            i += 1;
        }
        core::arch::nvptx::_syncthreads();
        if warp == 0 {
            let mut total = if lane < chunks {
                f32::from_bits(load_u32(smem + (lane * 4) as u32))
            } else {
                0.0f32
            };
            // Exact binary tree join: distances 16, 8, 4, 2, 1 (same order as `block_ops::norm_join` for chunks <= 32).
            total += shfl_xor_f32(total, 16);
            total += shfl_xor_f32(total, 8);
            total += shfl_xor_f32(total, 4);
            total += shfl_xor_f32(total, 2);
            total += shfl_xor_f32(total, 1);
            if lane == 0 {
                let scale = 1.0f32 / root(total / dim as f32 + eps);
                store_u32(smem + (NORM_CHUNKS_MAX * 4) as u32, scale.to_bits());
            }
        }
        core::arch::nvptx::_syncthreads();
        let scale = f32::from_bits(load_u32(smem + (NORM_CHUNKS_MAX * 4) as u32));
        if vector_store {
            let mut j = tid * 4;
            while j < dim {
                let [a, b, c, d] = ld_global_nc_v4(x_ptr.add(j));
                let [a, b, c, d] = if weight.is_null() {
                    [rn(a * scale), rn(b * scale), rn(c * scale), rn(d * scale)]
                } else {
                    let [wa, wb, wc, wd] = ld_global_nc_v4(weight.add(j));
                    [rn((a * scale) * wa), rn((b * scale) * wb), rn((c * scale) * wc), rn((d * scale) * wd)]
                };
                core::arch::asm!("st.global.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg64) o_ptr.add(j), a = in(reg32) a, b = in(reg32) b,
                    c = in(reg32) c, d = in(reg32) d);
                j += threads * 4;
            }
            return;
        }
        i = 0;
        while i < PER_WARP {
            let ch = warp + 4 * i;
            if ch < chunks {
                let base = ch * NORM_CHUNK + lane;
                let mut g = 0;
                while g < 4 {
                    let x = v[4 * i + g];
                    *o_ptr.add(base + g * 32) = if weighted { rn((x * scale) * wv[4 * i + g]) } else { rn(x * scale) };
                    g += 1;
                }
            }
            i += 1;
        }
    }
}
