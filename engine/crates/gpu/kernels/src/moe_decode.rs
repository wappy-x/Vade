//! Selected-expert decode over the resident MMA-packed BF16 weight arena.
use super::{ZERO, madd, rounded_pair, round_bf16};
use crate::canonical::{add, range, weight_tile};
use crate::ptx;
use core::arch::nvptx::{_block_idx_x, _block_idx_y, _grid_dim_x, _thread_idx_x};

#[inline(always)]
unsafe fn stage(shared: u32, first: *const u32, second: *const u32,
    output_tile: usize, col: usize, tiles: usize, tid: usize) {
    for copy in 0..2 {
        let i = tid + copy * 128;
        let part = i / 64;
        let (start, end) = range(tiles, part);
        let tile = start + col + i % 64 / 16;
        if tile < end {
            let at = weight_tile(output_tile, tile, tiles) * 64 + i % 16 * 4;
            ptx::copy16_stream(shared + i as u32 * 16, first.add(at));
            ptx::copy16_stream(shared + 8192 + i as u32 * 16, second.add(at));
        }
    }
    ptx::commit();
}

#[inline(always)]
unsafe fn balanced(at: u32, stride: u32) -> f32 {
    (ptx::load(at) + ptx::load(at + stride))
        + (ptx::load(at + 2 * stride) + ptx::load(at + 3 * stride))
}

// Four warps partition K. Each reads disjoint weight fragments for eight
// output columns; only the first MMA row is retained for single-token decode.
#[inline(always)]
unsafe fn project(
    first: *const u32, second: *const u32, input: *const f32,
    output: *mut u16, width: usize,
) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let tiles = width / 16;
    let output_tile = _block_idx_x() as usize;
    let (mut g, mut u) = (ZERO, ZERO);
    let shared = ptx::shared::<16640>();
    stage(shared, first, second, output_tile, 0, tiles, tid);
    if tiles / 4 > 4 { stage(shared + 4096, first, second, output_tile, 4, tiles, tid); }
    let mut block = 0;
    while block < tiles / 4 {
        if block + 4 < tiles / 4 { ptx::wait_one(); } else { ptx::wait(); }
        ptx::sync();
        let buffer = shared + ((block / 4 % 2) * 4096) as u32;
        for part in 0..4 {
            let local = warp * 4 + part;
            let tile = range(tiles, warp).0 + block + part;
            if tile < tiles {
                let col = tile * 16 + lane % 4 * 2;
                let lo = rounded_pair(input, 0, col, 1, width);
                let hi = rounded_pair(input, 0, col + 8, 1, width);
                let a = [lo, lo, hi, hi];
                let offset = (local * 32 + lane) as u32 * 8;
                g = madd(a, ptx::load2(buffer + offset), g);
                u = madd(a, ptx::load2(buffer + 8192 + offset), u);
            }
        }
        ptx::sync();
        if block + 8 < tiles / 4 { stage(buffer, first, second, output_tile, block + 8, tiles, tid); }
        block += 4;
    }
    let shared = shared + 16384;
    if lane < 4 {
        let offset = (warp * 8 + lane * 2) as u32 * 4;
        ptx::store(shared + offset, g.a);
        ptx::store(shared + offset + 4, g.b);
        ptx::store(shared + 128 + offset, u.a);
        ptx::store(shared + 128 + offset + 4, u.b);
    }
    ptx::sync();
    if tid < 8 {
        let mut value = balanced(shared + tid as u32 * 4, 32);
        let up = balanced(shared + 128 + tid as u32 * 4, 32);
        value = value / (1.0 + crate::layer::exp(-value)) * up;
        *output.add(_block_idx_x() as usize * 8 + tid) = round_bf16(value);
    }
}

/// Grid (intermediate/8, selected), block128. Each selected expert is read
/// directly through its table pointer; no gathered or duplicate weight arena.
/// Activations occupy the BF16 prefix of the scratch shared with scalar decode.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_gate_up(p: crate::kernel_params::MoeSingleProjection) {
    let (tables, indices, input, output) = (p.tables as *const u64, p.indices as *const u32,
        p.input as *const f32, p.output as *mut f32);
    let (intermediate, hidden) = (p.width, p.input_width);

    let slot = _block_idx_y() as usize;
    let expert = *indices.add(slot) as usize;
    project(*tables.add(expert * 3) as *const u32,
        *tables.add(expert * 3 + 1) as *const u32, input,
        output.cast::<u16>().add(slot * intermediate as usize), hidden as usize);
}

/// Fixed CTA grid consumes only compact selected tiles. The four K ranges,
/// BF16 operands, and balanced merge come from the canonical decode pipeline.
#[inline(always)]
unsafe fn grouped_decode<const FUSED: bool, const ROWS: usize>(p: crate::kernel_params::MoeDecode) {
    let (tables, counts, offsets, tile_experts) = (p.tables as *const u64,
        p.counts as *const u32, p.offsets as *const u32, p.tile_experts as *const u32);
    let columns = p.width as usize / 64;
    let total = *offsets.add(p.experts as usize) as usize / (16 * ROWS) * columns;
    let mut work = _block_idx_x() as usize;
    while work < total {
        let tile = work / columns * ROWS;
        let expert = *tile_experts.add(tile) as usize;
        let offset = *offsets.add(expert) as usize;
        let row = tile - offset / 16;
        let count = *counts.add(expert);
        let weights = *tables.add(expert * 3 + if FUSED { 0 } else { 2 }) as *const u32;
        let up = if FUSED { *tables.add(expert * 3 + 1) as *const u32 } else { weights };
        let output = if FUSED { (p.output as *mut u32).add(offset * p.width as usize / 2).cast() }
            else { (p.output as *mut f32).add(offset * p.width as usize) };
        super::shared::narrow_precision::<FUSED, false, false, ROWS>(weights, up,
            (p.packed as *const u32).add(offset * p.input_width as usize / 2), output,
            p.width, p.input_width, count, p.input_width, row, work % columns * 8, p.bf16 != 0);
        work += _grid_dim_x() as usize;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_decode_grouped_gate_up(p: crate::kernel_params::MoeDecode) {
    grouped_decode::<true, 1>(p);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_decode_grouped_down(p: crate::kernel_params::MoeDecode) {
    grouped_decode::<false, 1>(p);
}

/// Canonical prefill reuses each expert weight fragment across 32 gate/up rows and 64 down rows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_grouped_prefill_gate_up(p: crate::kernel_params::MoeDecode) {
    grouped_decode::<true, 2>(p);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_grouped_prefill_down(p: crate::kernel_params::MoeDecode) {
    grouped_decode::<false, 4>(p);
}

/// Grid (ceil(hidden/32), selected), block128. Each warp owns eight output
/// columns, avoiding the four-warp reduction for the shorter down projection.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_down(p: crate::kernel_params::MoeSingleProjection) {
    let (tables, indices, input, output) = (p.tables as *const u64, p.indices as *const u32,
        p.input as *const f32, p.output as *mut f32);
    let (hidden, intermediate) = (p.width, p.input_width);

    let slot = _block_idx_y() as usize;
    let expert = *indices.add(slot) as usize;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let row = _block_idx_x() as usize * 32 + warp * 8;
    if row >= hidden as usize { return; }
    let width = intermediate as usize;
    let tiles = width / 16;
    let weights = *tables.add(expert * 3 + 2) as *const u32;
    let input = input.cast::<u16>().add(slot * width);
    let (mut sum, mut left, mut right) = (ZERO, ZERO, ZERO);
    let (mut part, mut boundary) = (0, range(tiles, 0).1);
    let shared = ptx::shared::<16384>() + warp as u32 * 4096;
    stage_down(shared, weights, row/8, 0, tiles, lane);
    if tiles > 8 { stage_down(shared + 2048, weights, row/8, 8, tiles, lane); }
    let mut block = 0;
    while block < tiles {
        if block + 8 < tiles { ptx::wait_one(); } else { ptx::wait(); }
        ptx::warp_sync();
        let buffer = shared + ((block / 8 % 2) * 2048) as u32;
        for local in 0..8 {
            let tile = block + local;
            if tile < tiles {
                let col = tile * 16 + lane % 4 * 2;
                let lo = *input.add(col).cast::<u32>();
                let hi = *input.add(col + 8).cast::<u32>();
                sum = madd([lo, lo, hi, hi], ptx::load2(buffer + (local * 256 + lane * 8) as u32), sum);
                if tile + 1 == boundary {
                    match part {
                        0 => left = sum,
                        1 => left = add(left, sum),
                        2 => right = sum,
                        _ => right = add(right, sum),
                    }
                    sum = ZERO;
                    part += 1;
                    boundary = range(tiles, part).1;
                }
            }
        }
        ptx::warp_sync();
        if block + 16 < tiles { stage_down(buffer, weights, row/8, block + 16, tiles, lane); }
        block += 8;
    }
    let sum = add(left, right);
    if lane < 4 {
        let output = output.add(slot * hidden as usize + row + lane * 2);
        *output = sum.a;
        *output.add(1) = sum.b;
    }
}

#[inline(always)]
unsafe fn stage_down(shared: u32, weights: *const u32, row_tile: usize, tile: usize, tiles: usize, lane: usize) {
    for copy in 0..4 {
        let offset = lane * 4 + copy * 128;
        if offset < (tiles - tile).min(8) * 64 {
            ptx::copy16_stream(shared + offset as u32 * 4, weights.add(weight_tile(row_tile, tile + offset / 64, tiles) * 64 + offset % 64));
        }
    }
    ptx::commit();
}
