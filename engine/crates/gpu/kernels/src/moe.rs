//! Sparse expert decode: device routing and only selected BF16 projections.
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _thread_idx_x};

#[inline(always)]
unsafe fn bf16(value: f32) -> f32 {
    let bits: u16;
    core::arch::asm!("cvt.rn.bf16.f32 {out}, {input};", out = out(reg16) bits, input = in(reg32) value);
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
unsafe fn thread() -> usize {
    _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize
}

/// One thread computes softmax and stable top-k; ties prefer the lower expert.
/// Scores have `experts` entries, indices/weights have `selected` entries.
/// A row with a non-finite score is closed by finish_route instead.
#[inline(always)]
unsafe fn route_row(
    scores: *mut f32, indices: *mut u32, weights: *mut f32, status: *mut u32,
    experts: u32, selected: u32, renormalize: u32,
) {
    if renormalize & 2 != 0 {
        for expert in 0..experts as usize { *scores.add(expert) = bf16(*scores.add(expert)); }
    }
    let first = renormalize & crate::kernel_params::ROUTE_FIRST_LAYER != 0;
    let mut largest = f32::NEG_INFINITY;
    let mut finite = true;
    for expert in 0..experts as usize {
        let score = *scores.add(expert);
        largest = largest.max(score);
        finite &= score.is_finite();
    }
    if !finite { crate::kernel_params::finish_route(indices, weights, status, selected, false, first); return; }
    let mut total = 0.0f32;
    for expert in 0..experts as usize {
        let value = crate::layer::exp(*scores.add(expert) - largest);
        *scores.add(expert) = value;
        total += value;
    }
    // Finite scores always give a finite total and a candidate for every
    // slot; the checks reject the row rather than trust that.
    let mut valid = total.is_finite();
    let mut selected_total = 0.0f32;
    for slot in 0..selected as usize {
        let mut best = 0usize;
        let mut value = f32::NEG_INFINITY;
        for expert in 0..experts as usize {
            let candidate = *scores.add(expert);
            if candidate > value { value = candidate; best = expert; }
        }
        valid &= value != f32::NEG_INFINITY;
        let probability = value / total;
        *indices.add(slot) = best as u32;
        *weights.add(slot) = probability;
        selected_total += probability;
        *scores.add(best) = f32::NEG_INFINITY;
    }
    crate::kernel_params::finish_route(indices, weights, status, selected, valid, first);
    if !valid { return; }
    if renormalize & 1 != 0 {
        for slot in 0..selected as usize { *weights.add(slot) /= selected_total; }
    }
    if renormalize & 2 != 0 { for slot in 0..selected as usize { *weights.add(slot) = bf16(*weights.add(slot)); } }
    // HF's eager experts accumulate in expert-index order.
    for slot in 1..selected as usize {
        let id = *indices.add(slot);
        let weight = *weights.add(slot);
        let mut at = slot;
        while at > 0 && *indices.add(at - 1) > id {
            *indices.add(at) = *indices.add(at - 1);
            *weights.add(at) = *weights.add(at - 1);
            at -= 1;
        }
        *indices.add(at) = id;
        *weights.add(at) = weight;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_route(p: crate::kernel_params::MoeRoute) {
    let (scores, indices, weights) = (p.scores as *mut f32, p.indices as *mut u32, p.weights as *mut f32);
    let (experts, selected, renormalize) = (p.experts, p.selected, p.renormalize);

    route_warp(scores, indices, weights, p.status as *mut u32, experts, selected, renormalize, _thread_idx_x() as usize);
}

/// One warp per row; same selection, sum order and rounding as route_row.
/// A row with a non-finite score is closed by finish_route instead.
#[inline(always)]
unsafe fn route_warp(scores: *mut f32, indices: *mut u32, weights: *mut f32, status: *mut u32,
    experts: u32, selected: u32, renormalize: u32, lane: usize) {
    if renormalize & 2 != 0 {
        for expert in (lane..experts as usize).step_by(32) { *scores.add(expert) = bf16(*scores.add(expert)); }
        crate::ptx::warp_sync();
    }
    let first = renormalize & crate::kernel_params::ROUTE_FIRST_LAYER != 0;
    let mut largest = f32::NEG_INFINITY;
    let mut finite = true;
    for expert in (lane..experts as usize).step_by(32) {
        let score = *scores.add(expert);
        largest = largest.max(score);
        finite &= score.is_finite();
    }
    // One NaN or infinity leaves the row without a softmax; the whole warp
    // leaves before any selection could miss every candidate.
    if crate::ptx::ballot(!finite) != 0 {
        if lane == 0 { crate::kernel_params::finish_route(indices, weights, status, selected, false, first); }
        return;
    }
    largest = crate::layer::maximum(largest);
    for expert in (lane..experts as usize).step_by(32) {
        *scores.add(expert) = crate::layer::exp(*scores.add(expert) - largest);
    }
    crate::ptx::warp_sync();
    // Preserve the existing expert-order sum and weight rounding exactly.
    let mut total = 0.0f32;
    if lane == 0 { for expert in 0..experts as usize { total += *scores.add(expert); } }
    // Finite scores always give a finite total and a candidate for every
    // slot; the checks keep lane 0's writes in range rather than trust that.
    let mut valid = total.is_finite();
    let mut selected_total = 0.0f32;
    for slot in 0..selected as usize {
        let (mut value, mut best) = (f32::NEG_INFINITY, u32::MAX);
        for expert in (lane..experts as usize).step_by(32) {
            let candidate = *scores.add(expert);
            if candidate > value { value = candidate; best = expert as u32; }
        }
        for distance in [16, 8, 4, 2, 1] {
            let other = f32::from_bits(crate::shuffle_xor(u32::MAX, value.to_bits(), distance, 31));
            let id = crate::shuffle_xor(u32::MAX, best, distance, 31);
            if other > value || other == value && id < best { value = other; best = id; }
        }
        valid &= best < experts;
        if lane == 0 && valid {
            let probability = value / total;
            *indices.add(slot) = best;
            *weights.add(slot) = probability;
            selected_total += probability;
            *scores.add(best as usize) = f32::NEG_INFINITY;
        }
        crate::ptx::warp_sync();
    }
    if lane == 0 {
        crate::kernel_params::finish_route(indices, weights, status, selected, valid, first);
        if !valid { return; }
        if renormalize & 1 != 0 { for slot in 0..selected as usize { *weights.add(slot) /= selected_total; } }
        if renormalize & 2 != 0 { for slot in 0..selected as usize { *weights.add(slot) = bf16(*weights.add(slot)); } }
        for slot in 1..selected as usize {
            let (id, weight) = (*indices.add(slot), *weights.add(slot));
            let mut at = slot;
            while at > 0 && *indices.add(at - 1) > id {
                *indices.add(at) = *indices.add(at - 1);
                *weights.add(at) = *weights.add(at - 1);
                at -= 1;
            }
            *indices.add(at) = id;
            *weights.add(at) = weight;
        }
    }
}

/// Eight router scores share each packed MMA tile. Four warps retain the
/// canonical K ranges and balanced merge; the final CTA performs stable top-k.
/// The session counter is reset before this same stream can run another layer.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_router(p: crate::kernel_params::MoeRouter) {
    let (matrix, input, scores) = (p.matrix as *const u32, p.input as *const f32, p.scores as *mut f32);
    let (indices, weights, completed) = (p.indices as *mut u32, p.weights as *mut f32, p.completed as *mut u32);
    let status = p.status as *mut u32;
    let (experts, width, selected, renormalize) = (p.experts, p.width, p.selected, p.renormalize);
    let tid = _thread_idx_x() as usize;
    let tile = _block_idx_x() as usize;
    let (lane, part) = (tid % 32, tid / 32);
    let sum = crate::matvec::dot_range(matrix, input, tile, lane, width, width, 0, part);
    if lane < 4 {
        let at = (part * 8 + lane * 2) as u32 * 4;
        crate::ptx::store(at, sum.a);
        crate::ptx::store(at + 4, sum.b);
    }
    crate::ptx::sync();
    if tid < 8 {
        let at = tid as u32 * 4;
        let score = (crate::ptx::load(at) + crate::ptx::load(at + 32))
            + (crate::ptx::load(at + 64) + crate::ptx::load(at + 96));
        *scores.add(tile * 8 + tid) = score;
    }
    // Every score writer publishes before the CTA leader signals completion.
    core::arch::asm!("fence.acq_rel.gpu;");
    crate::ptx::sync();
    if tid == 0 {
        let previous: u32;
        core::arch::asm!("atom.acq_rel.gpu.global.add.u32 {old}, [{counter}], 1;",
            old=out(reg32)previous, counter=in(reg64)completed);
        crate::ptx::store(128, (previous + 1 == experts / 8) as u32 as f32);
    }
    crate::ptx::sync();
    if crate::ptx::load(128) != 0.0 && tid < 32 {
        route_warp(scores, indices, weights, status, experts, selected, renormalize, tid);
        if tid == 0 { *completed = 0; }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_route_batch(p: crate::kernel_params::MoeRouteBatch) {
    let (scores, indices, weights) = (p.scores as *mut f32, p.indices as *mut u32, p.weights as *mut f32);
    let (experts, selected, tokens, renormalize) = (p.experts, p.selected, p.tokens, p.renormalize);
    let row = if tokens >= 256 { thread() / 32 } else { thread() };
    if row < tokens as usize {
        let scores = scores.add(row * experts as usize);
        let indices = indices.add(row * selected as usize);
        let weights = weights.add(row * selected as usize);
        let status = (p.status as *mut u32).add(row);
        if tokens >= 256 {
            route_warp(scores, indices, weights, status, experts, selected, renormalize, _thread_idx_x() as usize % 32);
        } else {
            route_row(scores, indices, weights, status, experts, selected, renormalize);
        }
    }
}

/// Typed prompts use one warp per row with the canonical sum and tie order.
/// Emit the routed FFN's (token, expert, weight) triples without a host bridge.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_route_rows(p: crate::kernel_params::MoeRouteBatch, pairs: *mut f32) {
    let row = if p.tokens >= 256 { thread() / 32 } else { thread() };
    route_rows(p, pairs, row);
}

#[inline(always)]
pub(crate) unsafe fn route_rows(p: crate::kernel_params::MoeRouteBatch, pairs: *mut f32, row: usize) {
    let prompt = p.tokens >= 256;
    let lane = if prompt { _thread_idx_x() as usize % 32 } else { 0 };
    if row >= p.tokens as usize { return; }
    let selected = p.selected as usize;
    let indices = (p.indices as *mut u32).add(row * selected);
    let weights = (p.weights as *mut f32).add(row * selected);
    let scores = (p.scores as *mut f32).add(row * p.experts as usize);
    let status = (p.status as *mut u32).add(row);
    if prompt { route_warp(scores, indices, weights, status, p.experts, p.selected, p.renormalize, lane); }
    else { route_row(scores, indices, weights, status, p.experts, p.selected, p.renormalize); }
    if lane != 0 { return; }
    for slot in 0..selected {
        let pair = pairs.add((row * selected + slot) * 3);
        *pair = row as f32;
        *pair.add(1) = *indices.add(slot) as f32;
        *pair.add(2) = *weights.add(slot);
    }
}

/// Fused small cohorts route every row with one warp pass per row (warp 0);
/// route_warp keeps route_row's serial sum, tie order and rounding verbatim.
/// The syncs reconverge the warp after lane 0's early exits and pair writes.
#[inline(always)]
pub(crate) unsafe fn route_rows_small(p: crate::kernel_params::MoeRouteBatch, pairs: *mut f32, lane: usize) {
    let selected = p.selected as usize;
    for row in 0..p.tokens as usize {
        let indices = (p.indices as *mut u32).add(row * selected);
        let weights = (p.weights as *mut f32).add(row * selected);
        let scores = (p.scores as *mut f32).add(row * p.experts as usize);
        route_warp(scores, indices, weights, (p.status as *mut u32).add(row),
            p.experts, p.selected, p.renormalize, lane);
        crate::ptx::warp_sync();
        if lane == 0 {
            for slot in 0..selected {
                let pair = pairs.add((row * selected + slot) * 3);
                *pair = row as f32;
                *pair.add(1) = *indices.add(slot) as f32;
                *pair.add(2) = *weights.add(slot);
            }
        }
        crate::ptx::warp_sync();
    }
}

/// Decode routing and compact row lists in one CTA. Block1024; tokens <= 32.
/// Barriers separate score selection, expert compaction, and offset publication.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_route_group_decode(p: crate::kernel_params::MoeRouteGroup) {
    let (scores, indices, weights) = (p.scores as *mut f32, p.indices as *mut u32, p.weights as *mut f32);
    let (rows, counts, offsets, inverse) = (p.rows as *mut u32, p.counts as *mut u32,
        p.offsets as *mut u32, p.inverse as *mut u32);
    let (experts, selected, tokens, renormalize) = (p.experts, p.selected, p.tokens, p.renormalize);
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    if warp < tokens as usize {
        route_warp(scores.add(warp * experts as usize), indices.add(warp * selected as usize),
            weights.add(warp * selected as usize), (p.status as *mut u32).add(warp), experts, selected, renormalize, lane);
    }
    crate::ptx::sync();
    let assignments = tokens as usize * selected as usize;
    for expert in (warp..experts as usize).step_by(32) {
        let mut count = 0usize;
        for start in (0..assignments).step_by(32) {
            let slot = start + lane;
            let hit = slot < assignments && *indices.add(slot) as usize == expert;
            let mask: u32;
            core::arch::asm!(
                "{{ .reg .pred hit; setp.ne.u32 hit, {value}, 0; vote.ballot.sync.b32 {mask}, hit, 0xffffffff; }}",
                value = in(reg32) hit as u32, mask = out(reg32) mask);
            if hit {
                let before = (mask & ((1u32 << lane).wrapping_sub(1))).count_ones() as usize;
                *rows.add(expert * tokens as usize + count + before) = slot as u32;
            }
            count += mask.count_ones() as usize;
        }
        if lane == 0 { *counts.add(expert) = count as u32; }
    }
    crate::ptx::sync();
    if tid == 0 {
        let mut offset = 0u32;
        for expert in 0..experts as usize {
            *offsets.add(expert) = offset;
            offset += (*counts.add(expert) + 15) / 16 * 16;
        }
        *offsets.add(experts as usize) = offset;
    }
    crate::ptx::sync();
    for expert in (warp..experts as usize).step_by(32) {
        // Routing is finished: reuse its scores for the compact 16-row work list.
        let offset = *offsets.add(expert);
        for tile in (lane..(*counts.add(expert) as usize + 15) / 16).step_by(32) {
            *scores.cast::<u32>().add(offset as usize / 16 + tile) = expert as u32;
        }
        for row in (lane..*counts.add(expert) as usize).step_by(32) {
            let slot = *rows.add(expert * tokens as usize + row) as usize;
            *inverse.add(slot) = *offsets.add(expert) + row as u32;
        }
    }
}

/// One warp compacts each expert's assignments in ascending token/slot order.
/// Launch at least `experts * 32` threads; each token selects an expert once.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_group(
    indices: *const u32, rows: *mut u32, counts: *mut u32,
    experts: u32, selected: u32, tokens: u32,
) {
    let tid = thread();
    let (expert, lane) = (tid / 32, tid % 32);
    if expert >= experts as usize { return; }
    let mut count = 0usize;
    let mut start = 0usize;
    let assignments = tokens as usize * selected as usize;
    while start < assignments {
        let slot = start + lane;
        let hit = slot < assignments && *indices.add(slot) as usize == expert;
        let mask: u32;
        core::arch::asm!(
            "{{ .reg .pred hit; setp.ne.u32 hit, {value}, 0; vote.ballot.sync.b32 {mask}, hit, 0xffffffff; }}",
            value = in(reg32) hit as u32, mask = out(reg32) mask,
        );
        if hit {
            let preceding = (mask & ((1u32 << lane).wrapping_sub(1))).count_ones() as usize;
            *rows.add(expert * tokens as usize + count + preceding) = slot as u32;
        }
        count += mask.count_ones() as usize;
        start += 32;
    }
    if lane == 0 { *counts.add(expert) = count as u32; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_group_offsets(p: crate::kernel_params::MoeGroupOffsets) {
    let (rows, counts, offsets, inverse) = (p.rows as *const u32, p.counts as *const u32,
        p.offsets as *mut u32, p.inverse as *mut u32);
    let (experts, tokens, alignment) = (p.experts, p.tokens, p.alignment);
    let tid = _thread_idx_x() as usize;
    if tid == 0 {
        let mut offset = 0u32;
        for expert in 0..experts as usize {
            *offsets.add(expert) = offset;
            offset += (*counts.add(expert) + alignment - 1) / alignment * alignment;
        }
        *offsets.add(experts as usize) = offset;
    }
    crate::ptx::sync();
    let mut expert = tid / 32;
    while expert < experts as usize {
        let offset = *offsets.add(expert);
        if p.tile_experts != 0 {
            for tile in (tid % 32..((*counts.add(expert) + alignment - 1) / alignment * alignment) as usize / 16).step_by(32) {
                *(p.tile_experts as *mut u32).add(offset as usize / 16 + tile) = expert as u32;
            }
        }
        let mut row = tid % 32;
        while row < *counts.add(expert) as usize {
            let slot = *rows.add(expert * tokens as usize + row) as usize;
            *inverse.add(slot) = offset + row as u32;
            row += 32;
        }
        expert += _block_dim_x() as usize / 32;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_merge_batch(p: crate::kernel_params::MoeMergeBatch) {
    let (projected, weights, inverse, output) = (p.projected as *const f32, p.weights as *const f32,
        p.inverse as *const u32, p.output as *mut f32);
    let (hidden, selected, tokens) = (p.hidden, p.selected, p.tokens);
    let i = thread();
    if i >= tokens as usize * hidden as usize { return; }
    let (token, col) = (i / hidden as usize, i % hidden as usize);
    let mut sum = 0.0f32;
    for slot in token * selected as usize..(token + 1) * selected as usize {
        let row = *inverse.add(slot) as usize;
        let value = *projected.add(row * hidden as usize + col);
        if p.bf16 != 0 { sum = bf16(sum + bf16(bf16(value) * *weights.add(slot))); }
        else { sum += value * *weights.add(slot); }
    }
    *output.add(i) = sum;
}

/// Expert sum, residual add, and the next norm's exact 128-column partials.
/// One warp handles four columns per lane, matching layer_rms's reduction.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_merge_residual_rms(
    projected: *const f32, weights: *const f32, inverse: *const u32, output: *mut f32,
    partials: *mut f32, hidden: u32, selected: u32, tokens: u32,
) {
    let token = _block_idx_y() as usize;
    if token >= tokens as usize { return; }
    let tid = thread();
    let (lane, chunk) = (tid % 32, tid / 32);
    let h = hidden as usize;
    if chunk * 128 >= h { return; }
    let mut sums = [0.0; 4];
    for slot in token * selected as usize..(token + 1) * selected as usize {
        let row = *inverse.add(slot) as usize;
        let weight = *weights.add(slot);
        for j in 0..4 {
            let col = chunk * 128 + lane + j * 32;
            if col < h {
                sums[j] += *projected.add(row * h + col) * weight;
            }
        }
    }
    let mut squares = 0.0;
    for j in 0..4 {
        let col = chunk * 128 + lane + j * 32;
        if col < h {
            let at = token * h + col;
            let x = *output.add(at) + sums[j];
            *output.add(at) = x;
            squares += x * x;
        }
    }
    squares = crate::layer::sum(squares);
    if lane == 0 { *partials.add(token * ((h + 127) / 128) + chunk) = squares; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn moe_merge(p: crate::kernel_params::MoeMerge) {
    let (input, weights, output) = (p.input as *const f32, p.weights as *const f32, p.output as *mut f32);
    let (hidden, selected) = (p.hidden, p.selected);

    let row = thread();
    if row >= hidden as usize { return; }
    let mut sum = 0.0f32;
    for slot in 0..selected as usize { sum += *input.add(slot * hidden as usize + row) * *weights.add(slot); }
    *output.add(row) = sum;
}
