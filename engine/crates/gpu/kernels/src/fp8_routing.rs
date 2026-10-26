//! Device routing for the Flash Next grouped expert FFN.
use core::arch::{asm, nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _thread_idx_x}};

#[inline(always)]
unsafe fn index() -> usize {
    _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize
}

#[inline(always)]
unsafe fn increment(address: *mut u32) -> u32 {
    let previous: u32;
    asm!("atom.global.add.u32 {old}, [{address}], 1;",
        old=out(reg32)previous, address=in(reg64)address);
    previous
}

#[inline(always)]
unsafe fn inclusive(mut value: u32, lane: u32) -> u32 {
    let mut distance = 1;
    while distance < 32 {
        let other: u32;
        asm!("shfl.sync.up.b32 {other}, {value}, {distance}, 0, 0xffffffff;",
            other=out(reg32)other, value=in(reg32)value, distance=in(reg32)distance);
        if lane >= distance { value += other; }
        distance *= 2;
    }
    value
}

#[inline(always)]
unsafe fn routing_error(p: crate::fp8_abi::GroupedRouteParams, token: usize) {
    asm!("atom.global.or.b32 {old}, [{address}], 1;", old=out(reg32) _, address=in(reg64)(p.error + (token * 4) as u64));
}

/// Clear histograms and reject invalid rows before they enter expert storage.
/// # Safety
/// Counts has experts words, token_count has tokens words. Block 256, grid
/// ceil(max(experts,32*tokens)/256). One warp validates each complete row
/// before the next kernel can increment any expert histogram.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_clear(p: crate::fp8_abi::GroupedRouteParams) {
    clear(p, 10);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_clear(g: crate::fp8_abi::RoutedParams) {
    // Use the warp's validation result without rereading its global status write.
    if clear(g.grouped.route, g.layout.selected as usize) {
        let token = index() / 32;
        let h = g.grouped.cols as usize;
        for slot in 0..g.layout.selected as usize {
            for col in (index() % 32..h).step_by(32) {
                *(g.grouped.output as *mut f32).add(g.layout.output_index(token, slot) * h + col) = 0.0;
            }
        }
    }
}
#[inline(always)]
unsafe fn clear(p: crate::fp8_abi::GroupedRouteParams, selected: usize) -> bool {
    let counts = p.counts as *mut u32;
    let token_count = p.token_count as *mut u32;
    let experts = p.experts;
    let tokens = p.tokens;
    let i = index();
    if i < experts as usize { *counts.add(i) = 0; }
    let (token, lane) = (i / 32, i % 32);
    if token >= tokens as usize { return false; }
    if lane == 0 { *token_count.add(token) = 0; }
    if p.failed(token) { return true; }
    let mut invalid = false;
    if lane < selected {
        let pairs = (p.pairs as *const f32).add(token * (selected * 3) + lane * 3);
        invalid = *pairs != token as f32 || !crate::fp8_abi::valid_route(
            token as f32, *pairs.add(1), *pairs.add(2), tokens, experts);
    }
    // Every live row's lanes reach the vote, including the unused lanes.
    let failed = crate::ptx::ballot(invalid) != 0;
    if failed && lane == 0 { routing_error(p, token); }
    failed
}

/// Histogram assignments and retain a local expert row plus token membership.
/// # Safety
/// Pairs is token-major [tokens,10,3] numeric FP32 (token, expert, weight). Every token has
/// exactly ten assignments, all token/expert IDs are in range, and IDs are
/// exactly representable in FP32. Scratch lengths are E, P, P, and T words.
/// Block 256, grid ceil(P/256). Counts and token_count were cleared.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_count(p: crate::fp8_abi::GroupedRouteParams) {
    count::<false>(p, crate::fp8_abi::RoutingLayout::FLASH);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_count(g: crate::fp8_abi::RoutedParams) { count::<false>(g.grouped.route, g.layout); }
#[inline(always)]
unsafe fn count<const STABLE: bool>(p: crate::fp8_abi::GroupedRouteParams, layout: crate::fp8_abi::RoutingLayout) {
    let selected = layout.selected as usize;
    let pairs = p.pairs as *const f32;
    let counts = p.counts as *mut u32;
    let pair_to_row = p.pair_to_row as *mut u32;
    let token_pairs = p.token_pairs as *mut u32;
    let token_count = p.token_count as *mut u32;
    let pair_count = p.pair_count;
    let pair = index();
    if pair >= pair_count as usize { return; }
    let token = pair / selected;
    if p.failed(token) { return; }
    let expert = *pairs.add(pair * 3 + 1) as usize;
    let row = increment(counts.add(expert));
    *pair_to_row.add(pair) = if STABLE { layout.stable_row(p, pair) } else { row };
    let mut slot = increment(token_count.add(token)) as usize;
    if STABLE { slot = pair % selected; }
    if slot >= selected { core::arch::nvptx::trap(); }
    *token_pairs.add(token * selected + slot) = pair as u32;
}

/// Prefix expert counts into padded row offsets and a compact M128 work list.
/// # Safety
/// Launch one warp. Counts has E entries, starts E+1, sorted_row at least
/// P+15*min(P,E). Tasks has 1+2*(ceil(P/128)+E) words: count then (expert,m_start)
/// tuples. Prefix writes only padding sentinels; order fills every real row.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_prefix(p: crate::fp8_abi::GroupedRouteParams) {
    let counts = p.counts as *const u32;
    let starts = p.starts as *mut u32;
    let tasks = p.tasks as *mut u32;
    let sorted_row = p.sorted_row as *mut u32;
    let experts = p.experts;
    let lane = _thread_idx_x() as u32;
    if lane >= 32 { return; }
    let per_lane = (experts + 31) / 32;
    let first = lane * per_lane;
    let end = (first + per_lane).min(experts);
    let mut rows = 0;
    let mut tiles = 0;
    let mut expert = first;
    while expert < end {
        let count = *counts.add(expert as usize);
        rows += (count + 15) & !15;
        tiles += (count + 127) / 128;
        expert += 1;
    }
    let row_end = inclusive(rows, lane);
    let tile_end = inclusive(tiles, lane);
    let mut row = row_end - rows;
    let mut tile = tile_end - tiles;
    expert = first;
    while expert < end {
        let count = *counts.add(expert as usize);
        let padded = (count + 15) & !15;
        *starts.add(expert as usize) = row;
        let mut padding = count;
        while padding < padded {
            if row + padding >= p.row_capacity { core::arch::nvptx::trap(); }
            *sorted_row.add((row + padding) as usize) = u32::MAX;
            padding += 1;
        }
        let mut local = 0;
        while local < count {
            *tasks.add((1 + tile * 2) as usize) = expert;
            *tasks.add((2 + tile * 2) as usize) = local;
            tile += 1;
            local += 128;
        }
        row += padded;
        expert += 1;
    }
    if lane == 31 {
        *starts.add(experts as usize) = row_end;
        *tasks = tile_end;
    }
}

/// Sort the ten routes per token and materialize the expert-major permutation.
/// # Safety
/// One thread per token, block128. Prefix and count completed. Buffers follow
/// their contracts; output_row has the same capacity as sorted_row. Ties use
/// input pair index, so duplicate experts also have a fixed reduction order.
/// Grouped row placement need not itself be stable.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_order(p: crate::fp8_abi::GroupedRouteParams) {
    order::<10>(p, crate::fp8_abi::RoutingLayout::FLASH);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_order(g: crate::fp8_abi::RoutedParams) { order::<32>(g.grouped.route, g.layout); }
#[inline(always)]
unsafe fn order<const MAX: usize>(p: crate::fp8_abi::GroupedRouteParams, layout: crate::fp8_abi::RoutingLayout) {
    let pairs = p.pairs as *const f32;
    let token_pairs = p.token_pairs as *mut u32;
    let pair_to_row = p.pair_to_row as *mut u32;
    let starts = p.starts as *const u32;
    let sorted_row = p.sorted_row as *mut u32;
    let tokens = p.tokens;
    let token = index();
    if token >= tokens as usize || p.failed(token) { return; }
    let selected = layout.selected as usize;
    if *(p.token_count as *const u32).add(token) != selected as u32 { core::arch::nvptx::trap(); }
    let mut ids = [0u32; MAX];
    let mut experts = [0u32; MAX];
    let mut i = 0;
    while i < selected {
        let pair = *token_pairs.add(token * selected + i);
        if pair >= p.pair_count { core::arch::nvptx::trap(); }
        let selected = *pairs.add(pair as usize * 3 + 1);
        if !crate::fp8_abi::valid_route(*pairs.add(pair as usize * 3), selected,
            *pairs.add(pair as usize * 3 + 2), p.tokens, p.experts) { core::arch::nvptx::trap(); }
        let expert = selected as u32;
        let row = *starts.add(expert as usize) + *pair_to_row.add(pair as usize);
        if row >= p.row_capacity { core::arch::nvptx::trap(); }
        *pair_to_row.add(pair as usize) = row;
        *sorted_row.add(row as usize) = token as u32;
        let mut position = i;
        while position > 0 && (experts[position - 1] > expert
            || (experts[position - 1] == expert && ids[position - 1] > pair))
        {
            ids[position] = ids[position - 1];
            experts[position] = experts[position - 1];
            position -= 1;
        }
        ids[position] = pair;
        experts[position] = expert;
        i += 1;
    }
    i = 0;
    while i < selected {
        *token_pairs.add(token * selected + i) = ids[i];
        let row = *pair_to_row.add(ids[i] as usize) as usize;
        if row >= p.row_capacity as usize { core::arch::nvptx::trap(); }
        *(p.output_row as *mut u32).add(row) = layout.output_index(token, i) as u32;
        i += 1;
    }
}

/// Build all routing metadata in one CTA: for a single token, or for several tokens
/// (the host sends 2..=31) with at most SMALL_ROUTE_THREADS experts and pairs.
/// # Safety
/// One token: launch one block256; pairs contains ten distinct, valid expert IDs,
/// all for token zero; scratch has the ordinary grouped sizes, with 160 padded rows,
/// and every destination is independently derived from the ten input assignments.
/// Several tokens: launch one block of SMALL_ROUTE_THREADS; pairs is token-major
/// [tokens,10,3] like fp8_grouped_count's, and scratch has the ordinary grouped
/// capacities for that many tokens (row capacity checked, rows past it trap).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_single(p: crate::fp8_abi::GroupedRouteParams) {
    if p.tokens > 1 {
        // 2..=31 tokens, block512: clear, count, prefix and order in one CTA, phase by
        // phase. Starts, padding and tasks are fp8_grouped_prefix's, from a block scan with
        // one expert per thread; rows inside an expert follow pair order where
        // fp8_grouped_count places them by atomic arrival. Each row is computed alone and
        // consumers address it through pair_to_row and output_row, so outputs are unchanged.
        let layout = crate::fp8_abi::RoutingLayout::FLASH;
        let selected = layout.selected as usize;
        let (tokens, pair_count) = (p.tokens as usize, p.pair_count as usize);
        let threads = crate::fp8_abi::SMALL_ROUTE_THREADS;
        if _block_dim_x() != threads || p.experts > threads || pair_count > threads as usize || pair_count != tokens * selected {
            core::arch::nvptx::trap();
        }
        let pairs = p.pairs as *const f32;
        let shared = crate::ptx::shared::<384>();
        let tid = _thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        // fp8_grouped_clear's row validation; the CTA keeps each token's final status.
        let mut token = warp;
        while token < tokens {
            let mut failed = p.failed(token);
            if !failed {
                let mut invalid = false;
                if lane < selected {
                    let at = pairs.add(token * (selected * 3) + lane * 3);
                    invalid = *at != token as f32 || !crate::fp8_abi::valid_route(token as f32, *at.add(1), *at.add(2), p.tokens, p.experts);
                }
                failed = crate::ptx::ballot(invalid) != 0;
                if failed && lane == 0 { routing_error(p, token); }
            }
            if lane == 0 { crate::ptx::store(shared + (token * 4) as u32, failed as u32 as f32); }
            token += 16;
        }
        crate::ptx::sync();
        let live = |pair: usize| crate::ptx::load(shared + (pair / selected * 4) as u32) == 0.0;
        let expert_of = |pair: usize| *pairs.add(pair * 3 + 1) as u32;
        let mut count = 0;
        if tid < p.experts as usize {
            for pair in 0..pair_count { count += (live(pair) && expert_of(pair) == tid as u32) as u32; }
            *(p.counts as *mut u32).add(tid) = count;
        }
        if tid < tokens { *(p.token_count as *mut u32).add(tid) = if live(tid * selected) { selected as u32 } else { 0 }; }
        // fp8_grouped_prefix's offsets as a block scan in expert order (one expert per thread).
        let (rows, tiles) = ((count + 15) & !15, (count + 127) / 128);
        let (row_in, tile_in) = (inclusive(rows, lane as u32), inclusive(tiles, lane as u32));
        if lane == 31 {
            crate::ptx::store(shared + 256 + (warp * 8) as u32, f32::from_bits(row_in));
            crate::ptx::store(shared + 260 + (warp * 8) as u32, f32::from_bits(tile_in));
        }
        crate::ptx::sync();
        let (mut row, mut tile) = (row_in - rows, tile_in - tiles);
        for w in 0..warp {
            row += crate::ptx::load(shared + 256 + (w * 8) as u32).to_bits();
            tile += crate::ptx::load(shared + 260 + (w * 8) as u32).to_bits();
        }
        if tid < p.experts as usize {
            *(p.starts as *mut u32).add(tid) = row;
            let mut padding = count;
            while padding < rows {
                if row + padding >= p.row_capacity { core::arch::nvptx::trap(); }
                *(p.sorted_row as *mut u32).add((row + padding) as usize) = u32::MAX;
                padding += 1;
            }
            let mut local = 0;
            while local < count {
                *(p.tasks as *mut u32).add((1 + tile * 2) as usize) = tid as u32;
                *(p.tasks as *mut u32).add((2 + tile * 2) as usize) = local;
                tile += 1;
                local += 128;
            }
            if tid + 1 == p.experts as usize {
                *(p.starts as *mut u32).add(tid + 1) = row + rows;
                *(p.tasks as *mut u32) = tile;
            }
        }
        crate::ptx::sync();
        // fp8_grouped_order: expert-major rows, then ascending (expert, pair) slots per token.
        if tid < pair_count && live(tid) {
            let (pair, token) = (tid, tid / selected);
            let expert = expert_of(pair);
            let mut local = 0;
            for previous in 0..pair { local += (live(previous) && expert_of(previous) == expert) as u32; }
            let row = *(p.starts as *const u32).add(expert as usize) + local;
            if row >= p.row_capacity { core::arch::nvptx::trap(); }
            *(p.pair_to_row as *mut u32).add(pair) = row;
            *(p.sorted_row as *mut u32).add(row as usize) = token as u32;
            let mut rank = 0;
            for other in token * selected..(token + 1) * selected {
                let e = expert_of(other);
                rank += (e < expert || (e == expert && other < pair)) as usize;
            }
            *(p.token_pairs as *mut u32).add(token * selected + rank) = pair as u32;
            *(p.output_row as *mut u32).add(row as usize) = layout.output_index(token, rank) as u32;
        }
        return;
    }
    let pairs = p.pairs as *const f32;
    let counts = p.counts as *mut u32;
    let starts = p.starts as *mut u32;
    let pair_to_row = p.pair_to_row as *mut u32;
    let token_pairs = p.token_pairs as *mut u32;
    let token_count = p.token_count as *mut u32;
    let tasks = p.tasks as *mut u32;
    let sorted_row = p.sorted_row as *mut u32;
    let experts = p.experts;
    let tid = _thread_idx_x() as usize;
    let failed = p.failed(0);
    crate::ptx::sync();
    if failed { if tid == 0 { *tasks = 0; *starts.add(experts as usize) = 0; } return; }
    if tid < 10 {
        let selected = *pairs.add(tid * 3 + 1);
        if *pairs.add(tid * 3) != 0.0 || !crate::fp8_abi::valid_route(0.0, selected,
            *pairs.add(tid * 3 + 2), p.tokens, p.experts) || p.row_capacity < 160 {
            routing_error(p, 0);
        }
        for previous in 0..tid {
            if *pairs.add(previous * 3 + 1) == selected { routing_error(p, 0); }
        }
    }
    crate::ptx::sync();
    if p.failed(0) { if tid == 0 { *tasks = 0; *starts.add(experts as usize) = 0; } return; }
    let mut expert = tid;
    while expert < experts as usize {
        let mut count = 0;
        let mut previous = 0;
        let mut pair = 0;
        while pair < 10 {
            let selected = *pairs.add(pair * 3 + 1) as usize;
            count += (selected == expert) as u32;
            previous += (selected < expert) as u32;
            pair += 1;
        }
        *counts.add(expert) = count;
        *starts.add(expert) = previous * 16;
        expert += 256;
    }
    if tid < 10 {
        let selected = *pairs.add(tid * 3 + 1);
        let mut rank = 0;
        let mut pair = 0;
        while pair < 10 {
            rank += (*pairs.add(pair * 3 + 1) < selected) as usize;
            pair += 1;
        }
        *pair_to_row.add(tid) = (rank * 16) as u32;
        *token_pairs.add(rank) = tid as u32;
        *(p.output_row as *mut u32).add(rank * 16) = rank as u32;
        *tasks.add(1 + rank * 2) = selected as u32;
        *tasks.add(2 + rank * 2) = 0;
    }
    if tid < 160 {
        *sorted_row.add(tid) = if tid % 16 == 0 { 0 } else { u32::MAX };
    }
    if tid == 0 {
        *starts.add(experts as usize) = 160;
        *tasks = 10;
        *token_count = 10;
    }
}

#[inline(always)]
unsafe fn bf16_pair(input: *const f32, token: u32, hidden: usize, column: usize) -> u32 {
    if token == u32::MAX { return 0; }
    let source = input.add(token as usize * hidden + column);
    let values = (*source.cast::<FloatPair>()).0;
    let result: u32;
    asm!("cvt.rn.bf16x2.f32 {result}, {high}, {low};",
        result=out(reg32)result, high=in(reg32)values[1], low=in(reg32)values[0]);
    result
}

#[repr(C, align(8))]
struct FloatPair([f32; 2]);

#[repr(C, align(16))]
struct InputFragment([u32; 4]);

/// Round token activations to packed BF16 MMA operands; zero the padding.
/// # Safety
/// Hidden is a multiple of 128; packed has row_capacity*hidden BF16 values.
/// Starts[E] gives the real padded extent. Input is [tokens,hidden] FP32.
/// Legacy grid ceil(row_capacity*hidden/8/256), block256. With activation
/// quantization, grid(80,tasks,1) consumes gate/up and writes E4M3 intermediate groups.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_gather(p: crate::fp8_abi::GroupedParams) {
    if p.activation_quantization != 0 { quantize_middle::<false>(p, crate::fp8_abi::RoutingLayout::FLASH); return; }
    let (input, packed) = (p.input as *const f32, p.output as *mut u32);
    let (sorted_row, starts) = (p.route.sorted_row as *const u32, p.route.starts as *const u32);
    let (row_capacity, hidden, experts) = (p.route.row_capacity, p.cols, p.route.experts);

    let i = index();
    let hidden = hidden as usize;
    let rows = (*starts.add(experts as usize)).min(row_capacity) as usize;
    if i >= rows * hidden / 8 { return; }
    let lane = i % 32;
    let tile = i / 32;
    let row = tile / (hidden / 16) * 16 + lane / 4;
    let column = tile % (hidden / 16) * 16 + lane % 4 * 2;
    let first = *sorted_row.add(row);
    let second = *sorted_row.add(row + 8);
    let values = [
        bf16_pair(input, first, hidden, column),
        bf16_pair(input, second, hidden, column),
        bf16_pair(input, first, hidden, column + 8),
        bf16_pair(input, second, hidden, column + 8),
    ];
    *packed.cast::<InputFragment>().add(i) = InputFragment(values);
}

/// Fixed row pitch; grid(ceil(128*intermediate/128/8),tasks,1), block256.
/// Prefix/order precede this entry; all padded rows and unowned channels become zero.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_gather(g: crate::fp8_abi::RoutedParams) { quantize_middle::<true>(g.grouped, g.layout); }

/// Gather unweighted rows in ascending expert order for cross-rank reduction.
/// # Safety
/// Output is [tokens,11,hidden]; the final shared row is written separately.
/// Token_pairs is sorted and pair_to_row addresses [rows,hidden] FP32 storage.
/// Expert descriptors supply the local width; absent experts contribute zero
/// without reading the grouped output row, which their projections skip.
/// Grid (tokens,ceil(hidden/256),1), block256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_scatter(p: crate::fp8_abi::GroupedParams) {
    scatter::<false>(p);
}

#[inline(always)]
pub(crate) unsafe fn scatter<const FP4: bool>(p: crate::fp8_abi::GroupedParams) {
    scatter_rows::<FP4, false>(p, crate::fp8_abi::RoutingLayout::FLASH);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_scatter(g: crate::fp8_abi::RoutedParams) { scatter_rows::<false, true>(g.grouped, g.layout); }
#[inline(always)]
unsafe fn scatter_rows<const FP4: bool, const TYPED: bool>(p: crate::fp8_abi::GroupedParams, layout: crate::fp8_abi::RoutingLayout) {
    let (token_pairs, pair_to_row) = (p.route.token_pairs as *const u32, p.route.pair_to_row as *const u32);
    let (down, output) = (p.input as *const f32, p.output as *mut f32);
    let (tokens, hidden) = (p.route.tokens, p.cols);

    let token = _block_idx_x() as usize;
    let column = _block_idx_y() as usize * 256 + _thread_idx_x() as usize;
    let hidden = hidden as usize;
    if token >= tokens as usize || column >= hidden { return; }
    if p.route.failed(token) {
        for i in 0..layout.selected as usize { *output.add(layout.output_index(token, i) * hidden + column) = 0.0; }
        return;
    }
    for i in 0..layout.selected as usize {
        let pair = *token_pairs.add(token * layout.selected as usize + i) as usize;
        let expert = *(p.route.pairs as *const f32).add(pair * 3 + 1) as usize;
        let width = if TYPED { (*(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3 + 2)).cols as u64 }
            else if FP4 { (*(p.weights as *const crate::fp4_abi::Expert).add(expert)).width as u64 }
            else { *((p.weights as *const u64).add(expert * 7 + 6)) };
        let value = if width == 0 { 0.0 } else {
            let row = *pair_to_row.add(pair) as usize;
            *down.add(row * hidden + column)
        };
        *output.add(layout.output_index(token, i) * hidden + column) = value;
    }
}

/// Each warp quantizes one local intermediate K128 group after BF16 SwiGLU.
/// Grid (ceil(128*5/8), task_capacity, 1), block256; unused expert tails skip.
#[inline(always)]
unsafe fn quantize_middle<const TYPED: bool>(p: crate::fp8_abi::GroupedParams, layout: crate::fp8_abi::RoutingLayout) {
    use crate::fp8::{rounded, quantize4};
    // Written for eight warps per block.
    if _block_dim_x() != 256 { core::arch::nvptx::trap(); }
    let tasks = p.route.tasks as *const u32;
    let task = _block_idx_y() as usize;
    if task >= *tasks as usize { return; }
    let expert = *tasks.add(1 + task * 2) as usize;
    let width = if TYPED { (*(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3)).rows as usize }
        else { *((p.weights as *const u64).add(expert * 7 + 6)) as usize };
    if !TYPED && width == 0 { return; }
    let tid = _thread_idx_x() as usize;
    let group = _block_idx_x() as usize * 8 + tid / 32;
    let blocks = layout.intermediate as usize / 128;
    let row = *tasks.add(2 + task * 2) as usize + group / blocks;
    let block = group % blocks;
    let m = *(p.route.counts as *const u32).add(expert) as usize;
    if row >= if TYPED { (m + 15) & !15 } else { m } || (!TYPED && block >= width / 128) { return; }
    let start = *(p.route.starts as *const u32).add(expert) as usize;
    let col = block * 128 + tid % 32 * 4;
    let input = (p.input as *const f32).add((start + row) * 2 * layout.intermediate as usize + col);
    let mut values = [0.0; 4];
    if !TYPED || (row < m && block < width / 128) {
        for j in 0..4 {
            let gate = *input.add(j);
            let silu = rounded(gate / (1.0 + crate::flash_moe::exp(-gate)));
            values[j] = rounded(silu * *input.add(layout.intermediate as usize + j));
        }
    }
    let (word, scale) = quantize4(values);
    *((p.output as *mut u8).add(layout.code_index(start, row, width, col)).cast::<u32>()) = word;
    if tid % 32 == 0 { *(p.input_scales as *mut f32).add(layout.scale_index(start, row, width, block)) = scale; }
}

/// Compact actual routed rows in ascending-expert token order. Padding is only
/// the unused third K128 block of a two-block shard, never padded GEMM rows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_middle_pack(p: crate::fp8_abi::MiddleParams) {
    let word = index();
    let token = word / crate::fp8_abi::MIDDLE_PACKET_WORDS;
    if token >= p.route.tokens as usize { return; }
    if p.route.failed(token) { *((p.send as *mut u32).add(word)) = 0; return; }
    let at = word % crate::fp8_abi::MIDDLE_PACKET_WORDS;
    let value = if at < 990 {
        let slot = at / 99;
        let part = at % 99;
        let pair = *(p.route.token_pairs as *const u32).add(token * 10 + slot) as usize;
        let expert = *((p.route.pairs as *const f32).add(pair * 3 + 1)) as usize;
        let start = *(p.route.starts as *const u32).add(expert) as usize;
        let row = *(p.route.pair_to_row as *const u32).add(pair) as usize - start;
        let width = *((p.weights as *const u64).add(expert * 7 + 6)) as usize;
        if part < 96 {
            if part * 4 < width {
                *((p.local_codes as *const u8).add(start * 640 + row * width + part * 4).cast::<u32>())
            } else { 0 }
        } else if part - 96 < width / 128 {
            *((p.local_scales as *const u32).add(start * 5 + row * (width / 128) + part - 96))
        } else { 0 }
    } else {
        let col = (1 - p.rank as usize) * 1280 + at - 990;
        *((p.partials as *const u32).add((token * 11 + 10) * 2560 + col))
    };
    *((p.send as *mut u32).add(word)) = value;
}

/// Copy codes/scales unchanged into full-K grouped rows and fold only the
/// shared expert's original FP32 partials. Routed rank subtotals are folded by
/// the down projection before the existing expert rounding and finish.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_middle_unpack(p: crate::fp8_abi::MiddleParams) {
    let word = index();
    let token = word / 2930;
    if token >= p.route.tokens as usize { return; }
    if p.route.failed(token) { return; }
    let at = word % 2930;
    if at < 1650 {
        let slot = at / 165;
        let part = at % 165;
        let pair = *(p.route.token_pairs as *const u32).add(token * 10 + slot) as usize;
        let expert = *((p.route.pairs as *const f32).add(pair * 3 + 1)) as usize;
        let row = *(p.route.pair_to_row as *const u32).add(pair) as usize;
        let cut = (3 - expert % 2) * 128;
        let column = if part < 160 { part * 4 } else { (part - 160) * 128 };
        let owner = usize::from(column >= cut);
        let local = column - if owner == 0 { 0 } else { cut };
        let source = if owner == p.rank as usize { p.send } else { p.receive } as *const u32;
        let source = source.add(token * crate::fp8_abi::MIDDLE_PACKET_WORDS + slot * 99);
        if part < 160 {
            *((p.codes as *mut u32).add(row * 160 + part)) = *source.add(local / 4);
        } else {
            *((p.scales as *mut u32).add(row * 5 + part - 160)) = *source.add(96 + local / 128);
        }
    } else {
        let col = at - 1650;
        let output = (p.partials as *mut f32).add((token * 11 + 10) * 2560 + p.rank as usize * 1280 + col);
        let peer = *((p.receive as *const f32).add(token * crate::fp8_abi::MIDDLE_PACKET_WORDS + 990 + col));
        *output = if p.rank == 0 { *output + peer } else { peer + *output };
    }
}

/// General-world records pad only the maximum local K128 count, followed by
/// the complete shared partial. Code and scale words are never converted.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_middle_pack_general(g: crate::fp8_abi::GeneralMiddleParams) {
    let p = g.middle;
    let world = g.world as usize;
    let record = 33 * 5usize.div_ceil(world);
    let words = crate::fp8_abi::middle_packet_words(world);
    let word = index();
    let token = word / words;
    if token >= p.route.tokens as usize { return; }
    let at = word % words;
    let mut value = 0;
    if !p.route.failed(token) {
        if at < 10 * record {
            let slot = at / record;
            let part = at % record;
            let pair = *(p.route.token_pairs as *const u32).add(token * 10 + slot) as usize;
            let expert = *((p.route.pairs as *const f32).add(pair * 3 + 1)) as usize;
            let start = *(p.route.starts as *const u32).add(expert) as usize;
            let row = *(p.route.pair_to_row as *const u32).add(pair) as usize - start;
            let width = *((p.weights as *const u64).add(expert * 7 + 6)) as usize;
            let codes = record / 33 * 32;
            if part < codes && part * 4 < width {
                value = *((p.local_codes as *const u8).add(start * 640 + row * width + part * 4).cast::<u32>());
            } else if part >= codes && part - codes < width / 128 {
                value = *((p.local_scales as *const u32).add(start * 5 + row * (width / 128) + part - codes));
            }
        } else {
            value = *((p.partials as *const u32).add((token * 11 + 10) * 2560 + at - 10 * record));
        }
    }
    *((p.send as *mut u32).add(word)) = value;
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_middle_unpack_general(g: crate::fp8_abi::GeneralMiddleParams) {
    let p = g.middle;
    let world = g.world as usize;
    let record = 33 * 5usize.div_ceil(world);
    let words = crate::fp8_abi::middle_packet_words(world);
    let span = (p.route.tokens as usize * words * 4).div_ceil(256) * 64;
    let word = index();
    let token = word / 4210;
    if token >= p.route.tokens as usize || p.route.failed(token) { return; }
    let at = word % 4210;
    if at < 1650 {
        let slot = at / 165;
        let part = at % 165;
        let pair = *(p.route.token_pairs as *const u32).add(token * 10 + slot) as usize;
        let expert = *((p.route.pairs as *const f32).add(pair * 3 + 1)) as usize;
        let row = *(p.route.pair_to_row as *const u32).add(pair) as usize;
        let block = if part < 160 { part / 32 } else { part - 160 };
        let mut owner = 0;
        while block >= crate::fp8_abi::middle_expert_end(expert, owner, world) { owner += 1; }
        let start = if owner == 0 { 0 } else { crate::fp8_abi::middle_expert_end(expert, owner - 1, world) };
        let source = (p.receive as *const u32).add(owner * span + token * words + slot * record);
        if part < 160 {
            *((p.codes as *mut u32).add(row * 160 + part)) = *source.add(part - start * 32);
        } else {
            *((p.scales as *mut u32).add(row * 5 + part - 160)) = *source.add(record / 33 * 32 + block - start);
        }
    } else {
        let col = at - 1650;
        let source = (p.receive as *const f32).add(token * words + 10 * record + col);
        let mut value = *source;
        for rank in 1..world { value += *source.add(rank * span); }
        *((p.partials as *mut f32).add((token * 11 + 10) * 2560 + col)) = value;
    }
}

/// Restore token-major full rows before the existing HC injection.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_middle_columns(g: crate::fp8_abi::GeneralMiddleParams) {
    let p = g.middle;
    let word = index();
    if word >= p.route.tokens as usize * 2560 { return; }
    let token = word / 2560;
    let col = word % 2560;
    let world = g.world as usize;
    let span = (p.route.tokens as usize * crate::fp8_abi::middle_packet_words(world) * 4).div_ceil(256) * 64;
    let mut owner = 0;
    while col >= crate::fp8_abi::middle_columns(owner, world).end { owner += 1; }
    let columns = crate::fp8_abi::middle_columns(owner, world);
    *((g.output as *mut u32).add(word)) = *((p.receive as *const u32)
        .add(owner * span + token * columns.len() + col - columns.start));
}
