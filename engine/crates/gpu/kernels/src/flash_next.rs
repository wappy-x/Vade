//! Multi-row Flash-Next execution; independent state and ordered expert folds.
use crate::next_math as math;
#[path = "next_verify.rs"]
mod verify;

/// Same recurrence function as the host chunk/batch oracle; no triangular solve.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_scan(p: crate::gdn_params::GdnStepBatch) {
    gdn_scan(p);
}
#[inline(always)]
pub(crate) unsafe fn gdn_scan(p: crate::gdn_params::GdnStepBatch) {
    let output = crate::index();
    let width = p.heads as usize * 128;
    if output >= width {
        return;
    }
    let (head, feature) = (output / 128, output % 128);
    let state = ((&*(p.rows as *const crate::gdn_params::GdnDecodeRow)).state as *mut f32)
        .add(head * 16384 + feature);
    // Runtime-indexed transfers force a local-memory array around column().
    // Constant indices keep all 128 cells in registers. Each warp still accesses
    // 32 adjacent features of the original coordinate-major state (128 bytes).
    macro_rules! eight {
        ($i:ident, $body:block) => {{
            const $i: usize = 0;
            $body
        }
        {
            const $i: usize = 1;
            $body
        }
        {
            const $i: usize = 2;
            $body
        }
        {
            const $i: usize = 3;
            $body
        }
        {
            const $i: usize = 4;
            $body
        }
        {
            const $i: usize = 5;
            $body
        }
        {
            const $i: usize = 6;
            $body
        }
        {
            const $i: usize = 7;
            $body
        }};
    }
    let mut cells = [0.0; 128];
    eight!(G, {
        eight!(P, {
            const I: usize = G * 16 + P * 2;
            cells[I] = *state.add(I * 128);
            cells[I + 1] = *state.add((I + 1) * 128);
        });
    });
    for token in 0..p.count as usize {
        let at = token * width + head * 128;
        let q = core::slice::from_raw_parts((p.q as *const f32).add(at), 128);
        let k = core::slice::from_raw_parts((p.k as *const f32).add(at), 128);
        let value = math::column(
            &mut cells,
            q,
            k,
            *(p.v as *const f32).add(token * width + output),
            *(p.beta as *const f32).add(token * p.heads as usize + head),
            crate::layer::exp(*(p.g as *const f32).add(token * p.heads as usize + head)),
        );
        *(p.out as *mut f32).add(token * width + output) = value;
    }
    let next = (&*(p.rows as *const crate::gdn_params::GdnDecodeRow)).output_state;
    let destination = if next == 0 {
        state
    } else {
        (next as *mut f32).add(head * 16384 + feature)
    };
    eight!(G, {
        eight!(P, {
            const I: usize = G * 16 + P * 2;
            *destination.add(I * 128) = cells[I];
            *destination.add((I + 1) * 128) = cells[I + 1];
        });
    });
}

/// One CTA builds compact expert bounds and the inverse map. Integer-only
/// ownership preserves token/slot order without atomics or padded activation rows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route(p: crate::fp8_abi::GroupedRouteParams) {
    compact_route(p);
}
/// flash_moe_assignments' stores (token, expert, weight per pair) in the route's
/// CTA, then the same route: one launch fewer per decode layer, same tables.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_ids(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64) {
    route_ids::<false>(p, ids, weights, NO_STEP);
}
/// Decode (rows <= 16): CTA 0 is flash_next_compact_route_ids; CTAs 1.. run flash_moe_silu's
/// elements (gate = silu(gate) * up, 256 per CTA) on other SMs: one launch fewer per layer.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_silu(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64,
    gate: u64, up: u64, n: u32) {
    let cta=core::arch::nvptx::_block_idx_x() as usize;
    if cta==0 { return route_ids::<false>(p, ids, weights, NO_STEP); }
    crate::flash_moe::silu_split_at(gate as *mut f32, up as *const f32, n, (cta-1)*256+core::arch::nvptx::_thread_idx_x() as usize);
}
/// Pooled decode: flash_next_compact_route_silu with flash_next_route_mask's stores (pairs of
/// rows < step.rows without a selected slot: id u32::MAX, weight 0) in the route's CTA.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_silu_mask(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64,
    gate: u64, up: u64, n: u32, step: math::SlotStep) {
    route_silu_mask(p, ids, weights, gate, up, n, step);
}
#[inline(always)]
unsafe fn route_silu_mask(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64,
    gate: u64, up: u64, n: u32, step: math::SlotStep) {
    let cta=core::arch::nvptx::_block_idx_x() as usize;
    if cta==0 { return route_ids::<true>(p, ids, weights, step); }
    crate::flash_moe::silu_split_at(gate as *mut f32, up as *const f32, n, (cta-1)*256+core::arch::nvptx::_thread_idx_x() as usize);
}
/// One decode token (C1, no route mask): CTA 0 runs flash_moe_route's top-10 for token 0 in
/// its first 128 threads (route_token, same exps / sums / selection), a block sync, then
/// flash_next_compact_route_ids; CTAs 1.. run flash_moe_silu's elements: route, assignments,
/// compact route and the shared expert's silu of a C1 layer in one launch.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_route_compact1_silu(r: crate::kernel_params::FlashMoeRoute,
    p: crate::fp8_abi::GroupedRouteParams, gate: u64, up: u64, n: u32) {
    let (cta, tid)=(core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_thread_idx_x() as usize);
    if cta==0 {
        crate::flash_moe::route_token::<true, true>(r, 0, tid);
        crate::ptx::sync();
        return route_ids::<false>(p, r.ids, r.weights, NO_STEP);
    }
    crate::flash_moe::silu_split_at(gate as *mut f32, up as *const f32, n, (cta-1)*256+tid);
}
/// Pooled C1: flash_next_route_compact1_silu with flash_next_route_mask's stores after the route.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_route_compact1_silu_mask(r: crate::kernel_params::FlashMoeRoute,
    p: crate::fp8_abi::GroupedRouteParams, gate: u64, up: u64, n: u32, step: math::SlotStep) {
    route1_silu_mask(r, p, gate, up, n, step);
}
#[inline(always)]
unsafe fn route1_silu_mask(r: crate::kernel_params::FlashMoeRoute,
    p: crate::fp8_abi::GroupedRouteParams, gate: u64, up: u64, n: u32, step: math::SlotStep) {
    let (cta, tid)=(core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_thread_idx_x() as usize);
    if cta==0 {
        crate::flash_moe::route_token::<true, true>(r, 0, tid);
        crate::ptx::sync();
        return route_ids::<true>(p, r.ids, r.weights, step);
    }
    crate::flash_moe::silu_split_at(gate as *mut f32, up as *const f32, n, (cta-1)*256+tid);
}
/// Pooled decode: flash_next_compact_route_silu_mask whose CTAs 1.. also write flash_bf16_input's
/// hi/lo planes of the silu'd gate (silu_pack), offered to the shared expert down's pack.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_pack_mask(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64,
    gate: u64, up: u64, n: u32, step: math::SlotStep, planes: u64) {
    let (cta, tid)=(core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_thread_idx_x() as usize);
    if cta==0 { return route_ids::<true>(p, ids, weights, step); }
    silu_pack(gate as *mut f32, up as *const f32, n, planes as *mut u32, (cta-1)*256+tid);
}
/// Verify / draft rows (<= 16, no route mask): flash_next_compact_route_silu whose CTAs 1..
/// run silu_pack instead of silu_at.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_pack(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64,
    gate: u64, up: u64, n: u32, planes: u64) {
    let (cta, tid)=(core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_thread_idx_x() as usize);
    if cta==0 { return route_ids::<false>(p, ids, weights, NO_STEP); }
    silu_pack(gate as *mut f32, up as *const f32, n, planes as *mut u32, (cta-1)*256+tid);
}
/// Pooled C1: flash_next_route_compact1_silu_mask whose CTAs 1.. run silu_pack.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_route_compact1_pack_mask(r: crate::kernel_params::FlashMoeRoute,
    p: crate::fp8_abi::GroupedRouteParams, gate: u64, up: u64, n: u32, step: math::SlotStep, planes: u64) {
    let (cta, tid)=(core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_thread_idx_x() as usize);
    if cta==0 {
        crate::flash_moe::route_token::<true, true>(r, 0, tid);
        crate::ptx::sync();
        return route_ids::<true>(p, r.ids, r.weights, step);
    }
    silu_pack(gate as *mut f32, up as *const f32, n, planes as *mut u32, (cta-1)*256+tid);
}
/// Column pair slot j (row j / 320, columns 2 (j % 320), + 1) of the one 16-row tile row of
/// [n / 640 <= 16][640]: flash_moe_silu's two elements (same arithmetic, gate written), then
/// pack_f32_input's split2 words of that pair at lanes 32 (its fragment's word: K16 tile c / 16,
/// lane row % 8 * 4 + c % 8 / 2, pair row / 8 + 2 (c % 16 / 8)); rows >= n / 640 write zeros.
#[inline(always)]
unsafe fn silu_pack(gate: *mut f32, up: *const f32, n: u32, planes: *mut u32, j: usize) {
    if j >= 16 * 320 { return; }
    let (row, c)=(j / 320, j % 320 * 2);
    let silu=|i: usize| {
        let value=*gate.add(i);
        let y=(value / (1.0 + crate::flash_moe::exp_split(-value))) * *up.add(i);
        *gate.add(i)=y;
        y
    };
    let (hi, lo)=if row * 640 < n as usize { crate::gemm::split2_pair(silu(row * 640 + c), silu(row * 640 + c + 1)) } else { (0, 0) };
    let word=(c / 16 * 32 + row % 8 * 4 + c % 8 / 2) * 4 + row / 8 + 2 * (c % 16 / 8);
    *planes.add(word)=hi; *planes.add(40 * 128 + word)=lo;
}
const NO_STEP: math::SlotStep = math::SlotStep { slots: 0, table: 0, layer: 0, rows: 0 };
/// MASK: flash_next_route_mask's stores for the pair (same thread, before its loads below).
#[inline(always)]
unsafe fn route_ids<const MASK: bool>(p: crate::fp8_abi::GroupedRouteParams, ids: u64, weights: u64, step: math::SlotStep) {
    let tid=core::arch::nvptx::_thread_idx_x() as usize; let a=p.pairs as *mut f32;
    for i in (tid..p.pair_count as usize).step_by(256) {
        if MASK && i<step.rows as usize*10 && step.selected(i/10).is_none() {
            *(ids as *mut u32).add(i)=u32::MAX; *(weights as *mut f32).add(i)=0.0;
        }
        *a.add(i*3)=(i/10) as f32; *a.add(i*3+1)=*(ids as *const u32).add(i) as f32; *a.add(i*3+2)=*(weights as *const f32).add(i);
    }
    crate::ptx::sync();
    compact_route(p);
}
#[inline(always)]
unsafe fn compact_route(p: crate::fp8_abi::GroupedRouteParams) {
    let tid=core::arch::nvptx::_thread_idx_x() as usize;
    for token in (tid..p.tokens as usize).step_by(256) { crate::compact::validate(p,token); }
    crate::ptx::sync();
    if p.pair_count as usize<=STAGED { return staged_route(p); }
    for expert in (tid..p.experts as usize).step_by(256) { crate::compact::count(p,expert); }
    crate::ptx::sync();
    compact_prefix(p);
    crate::ptx::sync();
    for expert in (tid..p.experts as usize).step_by(256) { crate::compact::order(p,expert); }
}

/// Pairs staged in shared memory by compact_route (decode and verify batches).
const STAGED: usize = 1024;
/// count/order's tables from pair-owned work: each live pair's expert ID is
/// staged once (u32::MAX = failed token; validated IDs are exact integers, so
/// the f32 comparisons are unchanged), its row within the expert is the number
/// of earlier live pairs with that expert (count's token/slot order), and the
/// expert's last pair writes the count. Same counts, starts, tasks and maps.
#[inline(always)]
unsafe fn staged_route(p: crate::fp8_abi::GroupedRouteParams) {
    let tid=core::arch::nvptx::_thread_idx_x() as usize; let pairs=p.pair_count as usize;
    let (ids,rows,counts)=(crate::ptx::shared::<{STAGED*4}>(),crate::ptx::shared::<{STAGED*4}>(),crate::ptx::shared::<2048>());
    let id=|pair: usize| crate::ptx::load(ids+(pair*4) as u32).to_bits();
    for pair in (tid..pairs).step_by(256) {
        let e=if p.failed(pair/10) { u32::MAX } else { *(p.pairs as *const f32).add(pair*3+1) as u32 };
        crate::ptx::store(ids+(pair*4) as u32,f32::from_bits(e));
    }
    for expert in (tid..512).step_by(256) { crate::ptx::store(counts+(expert*4) as u32,0.0); }
    crate::ptx::sync();
    for pair in (tid..pairs).step_by(256) {
        let e=id(pair); if e==u32::MAX { continue; }
        let (mut before,mut after)=(0u32,0u32);
        for other in 0..pairs { if id(other)==e { if other<pair { before+=1; } else if other>pair { after+=1; } } }
        crate::ptx::store(rows+(pair*4) as u32,f32::from_bits(before));
        if after==0 { crate::ptx::store(counts+(e*4),f32::from_bits(before+1)); }
    }
    crate::ptx::sync();
    for expert in (tid..p.experts as usize).step_by(256) {
        *(p.counts as *mut u32).add(expert)=crate::ptx::load(counts+(expert*4) as u32).to_bits();
    }
    crate::ptx::sync();
    compact_prefix(p);
    crate::ptx::sync();
    for pair in (tid..pairs).step_by(256) {
        let e=id(pair); if e==u32::MAX { continue; }
        let row=*(p.starts as *const u32).add(e as usize)+crate::ptx::load(rows+(pair*4) as u32).to_bits();
        *(p.sorted_row as *mut u32).add(row as usize)=pair as u32;
        let rank=crate::fp8_abi::RoutingLayout::FLASH.ordered_slot(pair%10,|slot| id(pair/10*10+slot));
        *(p.token_pairs as *mut u32).add(pair/10*10+rank)=pair as u32;
        *(p.pair_to_row as *mut u32).add(pair)=(pair/10*11+rank) as u32;
    }
}

/// Stream-ordered validation/count/prefix/order phases. Per-expert CTAs scan
/// consecutive pairs; ballots assign stable ranks without global atomics.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_prefill(
    p: crate::fp8_abi::GroupedRouteParams,
    phase: u32,
) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let expert = core::arch::nvptx::_block_idx_x() as usize;
    if phase == 0 {
        let t = expert * 256 + tid;
        if t < p.tokens as usize {
            crate::compact::validate(p, t);
        }
        return;
    }
    if phase == 2 {
        compact_prefix(p);
        return;
    }
    let (lane, warp) = (tid % 32, tid / 32);
    let sh = crate::ptx::shared::<32>();
    let mut offset = if phase == 3 {
        *(p.starts as *const u32).add(expert)
    } else {
        0
    };
    for first in (0..p.pair_count as usize).step_by(256) {
        let pair = first + tid;
        let live = pair < p.pair_count as usize
            && !p.failed(pair / 10)
            && *(p.pairs as *const f32).add(pair * 3 + 1) == expert as f32;
        let bits = crate::ptx::ballot(live);
        if lane == 0 {
            crate::ptx::store(sh + (warp * 4) as u32, f32::from_bits(bits.count_ones()));
        }
        crate::ptx::sync();
        let mut before = 0;
        let mut total = 0;
        for w in 0..8 {
            let n = crate::ptx::load(sh + w * 4).to_bits();
            if (w as usize) < warp {
                before += n;
            }
            total += n;
        }
        if phase == 3 && live {
            let row = offset + before + (bits & ((1u32 << lane) - 1)).count_ones();
            *(p.sorted_row as *mut u32).add(row as usize) = pair as u32;
            let rank = crate::fp8_abi::RoutingLayout::FLASH.ordered_slot(pair % 10, |slot| {
                *(p.pairs as *const f32).add((pair / 10 * 10 + slot) * 3 + 1) as u32
            });
            *(p.token_pairs as *mut u32).add(pair / 10 * 10 + rank) = pair as u32;
            *(p.pair_to_row as *mut u32).add(pair) = (pair / 10 * 11 + rank) as u32;
        }
        offset += total;
        crate::ptx::sync();
    }
    if phase == 1 && tid == 0 {
        *(p.counts as *mut u32).add(expert) = offset;
    }
}

/// Ascending-ID FMA fold; shared remains last. Sticky status masks
/// rejected/padded rows before reading expert partials or route weights.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_k16_finish(p: crate::fp8_abi::MoeFinishParams) {
    let row=core::arch::nvptx::_block_idx_y() as usize; let tid=core::arch::nvptx::_thread_idx_x() as usize;
    let col=crate::index(); let out=(p.output as *mut f32).add(row*2560);
    let error=*(p.error as *const u32).add(row)!=0; let sh=crate::ptx::shared::<40>();
    if tid<10 && !error {
        // Slot = tid; ordered_slot breaks ties by slot, so the ten ranks are distinct.
        let ids=(p.ids as *const u32).add(row*10); let weights=(p.weights as *const f32).add(row*10);
        let rank=crate::fp8_abi::RoutingLayout::FLASH.ordered_slot(tid,|j| *ids.add(j));
        crate::ptx::store(sh+(rank*4) as u32,*weights.add(tid));
    }
    crate::ptx::sync(); if col>=2560 { return; }
    if error { *out.add(col)=0.0; return; }
    let values=(p.partials as *const f32).add(row*11*2560+col); let mut value=0.0;
    let routed=(p.partials as *const u16).add(row*22*2560+col);
    for rank in 0..10 { let down=f32::from_bits((*routed.add(rank*2*2560) as u32)<<16);
        value=math::routed_add(value,down,crate::ptx::load(sh+(rank*4) as u32)); }
    let gate=1.0/(1.0+crate::flash_moe::exp(-*(p.shared_gate as *const f32).add(row)));
    *out.add(col)=f32::from_bits(math::bf16(math::routed_add(value,*values.add(10*2560),gate))<<16);
}

/// Option A's common expf sigmoid, with separate exp product and denominator add.
#[inline(always)]
pub(crate) unsafe fn hc_sigmoid(x: f32) -> f32 {
    let (a,b)=crate::fp4_experts::exp_parts(-x); 1.0/(1.0+a*b)
}
/// HC low activation in place (x/4 * hc_sigmoid(x/4)) of `elements` = M x 320 FP32 values.
/// With planes != 0 (decode rows) one thread per value pair also writes the pair's split2
/// hi/lo words into flash_bf16_input's dense operand layout (K 320, M rows), as that pack of
/// the activated rows would, so HC up skips it; row t + 8 of a 16-row tile past M gets its 0
/// word, other rows past M stay unwritten (MMA rows are independent, those outputs masked).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_activate(x: u64,elements: u32,planes: u64) {
    let i=crate::index();
    let act=|v: f32| { let value=v*0.25; value*hc_sigmoid(value) };
    if planes==0 { if i<elements as usize { let out=(x as *mut f32).add(i); *out=act(*out); } return; }
    let (k, at)=(320usize, 2*i); if at>=elements as usize { return; }
    let out=(x as *mut f32).add(at);
    let (a, b)=(act(*out), act(*out.add(1)));
    *out=a; *out.add(1)=b;
    let (rows, t, c, kt)=(elements as usize/k, at/k, at%k, k/16);
    let plane=rows.div_ceil(16)*kt*128;
    let word=((t/16*kt+c/16)*32+t%8*4+c%8/2)*4+t%16/8+c%16/8*2;
    let (hi, lo)=crate::gemm::split2_pair(a, b);
    let p=planes as *mut u32;
    *p.add(word)=hi; *p.add(plane+word)=lo;
    if t%16<8 && t+8>=rows { *p.add(word+1)=0; *p.add(plane+word+1)=0; }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_add(p: crate::kernel_params::FlashHcAdd) {
    let col=crate::index(); let row=core::arch::nvptx::_block_idx_y() as usize;
    let h=p.hidden as usize; if col>=h { return; }
    let block=*(p.block as *const f32).add(row*h+col);
    for stream in 0..4 { let x=(p.residual as *mut f32).add(row*4*h+stream*h+col);
        let gate=*(p.gates as *const f32).add(row*4+stream);
        *x=math::hc_add(*x,gate,block,|v| hc_sigmoid(v)); }
}
/// Decode rows (M <= 16): flash_next_hc_add (peer 0) fused with the next mix's flash_hc_norm_f32
/// and its packed planes. Grid 4 x M, block 128: CTA (stream, token) alone owns that row's stream.
/// Warp 0 forms every value residual + 2 sigmoid(gate / 4) block (next_math::hc_add), adds its
/// square in the norm's lane order and forms the norm's scale. Warp w forms its own 640 values
/// again (lane l: pairs w * 640 + 2l + 64j, +1, j < 10) from the same words. The one barrier
/// keeps every residual store behind every residual read of the CTA. Same values, products,
/// square order and packed words as the two launches.
/// # Safety
/// `rows` = grid.y = M <= 16. residual covers M x 4H FP32 words, block M x H, gates M x 4, weight
/// 4H, output M x 4H, packed the two dense planes of M x 4H; no two of these overlap.
// SAFETY: Hc::mix_chunk (blocks.rs) and the kernel bench (kernel_head.rs launch_hc) launch this
// entry, with buffers of those sizes. CTAs touch disjoint (stream, row) words, and the barrier orders each CTA's reads and stores.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_add_norm_rows(p: crate::kernel_params::FlashHcAddNorm, rows: u32) {
    let (group, groups) = (2560usize, 4usize);
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (stream, token) = (core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_block_idx_y() as usize);
    let base = token * groups * group + stream * group;
    let (x, y) = ((p.residual as *mut f32).add(base), (p.output as *mut f32).add(base));
    let (block, w) = ((p.block as *const f32).add(token * group), (p.weight as *const f32).add(stream * group));
    let sig = hc_sigmoid(*(p.gates as *const f32).add(token * groups + stream) * 0.25);
    let add = |r: f32, b: f32| crate::next_math::hc_add(r, 0.0, b, |_| sig);
    let first = warp * 640 + lane * 2;
    // Loads before the math: no value waits behind another value's store or square add.
    let pair = |j: usize| {
        let i = first + 64 * j;
        ([*x.add(i), *x.add(i + 1)], [*block.add(i), *block.add(i + 1)], [*w.add(i), *w.add(i + 1)])
    };
    let raw = [pair(0), pair(1), pair(2), pair(3), pair(4), pair(5), pair(6), pair(7), pair(8), pair(9)];
    let sh = crate::ptx::shared::<4>();
    if warp == 0 {
        let (lane_x, lane_b) = (x.add(lane), block.add(lane));
        let mut squares = 0.0f32;
        // Twenty values per batch: forty words in registers, then their adds in lane order.
        macro_rules! batch { ($at:literal; $($j:literal)*) => {
            let r = [$(*lane_x.add(($at + $j) * 32)),*];
            let b = [$(*lane_b.add(($at + $j) * 32)),*];
            $(squares = crate::flash_ops::norm_square_add(squares, add(r[$j], b[$j]));)*
        } }
        batch!(0; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19);
        batch!(20; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19);
        batch!(40; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19);
        batch!(60; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19);
        let scale = 1.0 / crate::layer::sqrt(crate::layer::sum(squares) / group as f32 + 1e-6);
        if lane == 0 { crate::ptx::store(sh, scale); }
    }
    crate::ptx::sync();
    let scale = crate::ptx::load(sh);
    let rows = rows as usize;
    let kt = groups * group / 16;
    let plane = rows.div_ceil(16) * kt * 128;
    let packed = p.packed as *mut u32;
    let word = |c: usize| ((token / 16 * kt + c / 16) * 32 + token % 8 * 4 + c % 8 / 2) * 4 + token % 16 / 8 + c % 16 / 8 * 2;
    macro_rules! store { ($($j:literal)*) => { $(
        let i = first + 64 * $j;
        let (r, b, g) = raw[$j];
        let (v0, v1) = (add(r[0], b[0]), add(r[1], b[1]));
        let (a, c) = (v0 * scale * (1.0 + g[0]), v1 * scale * (1.0 + g[1]));
        *x.add(i) = v0; *x.add(i + 1) = v1;
        *y.add(i) = a; *y.add(i + 1) = c;
        let (hi, lo) = crate::gemm::split2_pair(a, c);
        let at = word(stream * group + i);
        *packed.add(at) = hi;
        *packed.add(plane + at) = lo;
        // Row token + 8 of this 16-row tile is padding: its word is 0, as the pack's.
        if token % 16 < 8 && token + 8 >= rows { *packed.add(at + 1) = 0; *packed.add(plane + at + 1) = 0; }
    )* } }
    store!(0 1 2 3 4 5 6 7 8 9);
}

/// flash_next_hc_add whose block value is formed here exactly as flash_next_k16_finish forms
/// it (decode rows: one launch fewer per deferred layer); the block row is still stored.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_finish_add(p: crate::kernel_params::FlashHcAdd,
    f: crate::fp8_abi::MoeFinishParams) {
    let row=core::arch::nvptx::_block_idx_y() as usize; let tid=core::arch::nvptx::_thread_idx_x() as usize;
    let col=crate::index(); let h=p.hidden as usize;
    let error=*(f.error as *const u32).add(row)!=0; let sh=crate::ptx::shared::<40>();
    if tid<10 && !error {
        let ids=(f.ids as *const u32).add(row*10); let weights=(f.weights as *const f32).add(row*10);
        let rank=crate::fp8_abi::RoutingLayout::FLASH.ordered_slot(tid,|j| *ids.add(j));
        crate::ptx::store(sh+(rank*4) as u32,*weights.add(tid));
    }
    crate::ptx::sync(); if col>=h { return; }
    let block=if error { 0.0 } else {
        let values=(f.partials as *const f32).add(row*11*2560+col); let mut value=0.0;
        let routed=(f.partials as *const u16).add(row*22*2560+col);
        for rank in 0..10 { let down=f32::from_bits((*routed.add(rank*2*2560) as u32)<<16);
            value=math::routed_add(value,down,crate::ptx::load(sh+(rank*4) as u32)); }
        let gate=1.0/(1.0+crate::flash_moe::exp_split(-*(f.shared_gate as *const f32).add(row)));
        f32::from_bits(math::bf16(math::routed_add(value,*values.add(10*2560),gate))<<16)
    };
    *(f.output as *mut f32).add(row*2560+col)=block;
    for stream in 0..4 { let x=(p.residual as *mut f32).add(row*4*h+stream*h+col);
        let gate=*(p.gates as *const f32).add(row*4+stream);
        *x=math::hc_add(*x,gate,block,|v| hc_sigmoid(v)); }
}

/// Two adjacent experts per thread, with independent exact integer scans for
/// compact rows and ceil(count/128) tasks. Task order remains (expert, first).
#[inline(always)]
unsafe fn compact_prefix(p: crate::fp8_abi::GroupedRouteParams) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (lane, warp, expert) = (tid % 32, tid / 32, tid * 2);
    let n = |e: usize| if e < p.experts as usize { *(p.counts as *const u32).add(e) } else { 0 };
    let (a, b) = (n(expert), n(expert + 1));
    let (ta, tb) = (a.div_ceil(128), b.div_ceil(128));
    let (mut rows, mut tasks) = (a + b, ta + tb);
    let mut distance = 1u32;
    while distance < 32 {
        let (r, t): (u32, u32);
        core::arch::asm!(
            "shfl.sync.up.b32 {r}, {rows}, {d}, 0, -1;",
            "shfl.sync.up.b32 {t}, {tasks}, {d}, 0, -1;",
            r = out(reg32) r, t = out(reg32) t, rows = in(reg32) rows,
            tasks = in(reg32) tasks, d = in(reg32) distance);
        // Inclusive warp scan; each lane reads the previous stage before adding.
        if lane >= distance as usize { rows += r; tasks += t; }
        distance *= 2;
    }
    let sh = crate::ptx::shared::<64>();
    if lane == 31 {
        crate::ptx::store(sh + (warp * 8) as u32, f32::from_bits(rows));
        crate::ptx::store(sh + (warp * 8 + 4) as u32, f32::from_bits(tasks));
    }
    crate::ptx::sync();
    let (mut row, mut task) = (rows - a - b, tasks - ta - tb);
    for w in 0..warp {
        row += crate::ptx::load(sh + (w * 8) as u32).to_bits();
        task += crate::ptx::load(sh + (w * 8 + 4) as u32).to_bits();
    }
    let mut pair = 0;
    while pair < 2 {
        let e = expert + pair;
        let count = if pair == 0 { a } else { b };
        if e < p.experts as usize {
            *(p.starts as *mut u32).add(e) = row;
            for first in (0..count).step_by(128) {
                *(p.tasks as *mut u32).add(1 + task as usize * 2) = e as u32;
                *(p.tasks as *mut u32).add(2 + task as usize * 2) = first;
                task += 1;
            }
            row += count;
        }
        pair += 1;
    }
    if tid == 255 {
        *(p.starts as *mut u32).add(p.experts as usize) = row;
        *(p.tasks as *mut u32) = task;
    }
}

/// Stable routing with fixed10240-pair segments, independent of prompt length.
/// Warp assignment is segment_index%8; offsets are in ascending segment order.
/// Scratch has512*ceil(pair_count/10240) u32 counts, separately allocated from outputs.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_compact_route_prefill_packed_ids(
    p: crate::fp8_abi::GroupedRouteParams, phase:u32, scratch:u64,
){
    const SPAN:usize=10240;
    let tid=core::arch::nvptx::_thread_idx_x() as usize;
    let expert=core::arch::nvptx::_block_idx_x() as usize;
    let segments=(p.pair_count as usize).div_ceil(SPAN);
    let ids=(scratch as *mut u32).add(512*segments);
    if phase==0 {
        let token=expert*256+tid;
        if token<p.tokens as usize {
            crate::compact::validate(p,token);
            for slot in 0..10 {
                let pair=token*10+slot;
                *ids.add(pair)=if p.failed(token) {u32::MAX}else{*(p.pairs as *const f32).add(pair*3+1) as u32};
            }
            if !p.failed(token) {
                // Validation publishes each token's independent maps before counting.
                // No count/prefix is needed for these destinations. Rejected rows stay untouched.
                for slot in 0..10 {
                    let pair=token*10+slot;
                    let rank=crate::fp8_abi::RoutingLayout::FLASH.ordered_slot(slot,|j|{
                        *(p.pairs as *const f32).add((token*10+j)*3+1) as u32
                    });
                    *(p.token_pairs as *mut u32).add(token*10+rank)=pair as u32;
                    *(p.pair_to_row as *mut u32).add(pair)=(token*11+rank) as u32;
                }
            }
        }
        return;
    }
    if phase==2 {compact_prefix(p);return;}
    let (lane,warp)=(tid%32,tid/32);
    let segments=(p.pair_count as usize).div_ceil(SPAN);
    let counts=(scratch as *mut u32).add(expert*segments);
    let sh=crate::ptx::shared::<32>();
    if phase==1 {
        let mut total=0u32;
        let mut segment=warp;
        while segment<segments {
            let mut count=0u32;
            let mut base=segment*SPAN;
            let end=((segment+1)*SPAN).min(p.pair_count as usize);
            while base<end {
                let pair=base+lane;
                let live=pair<end && *ids.add(pair)==expert as u32;
                let bits=crate::ptx::ballot(live);
                if lane==0 {count+=bits.count_ones();}
                base+=32;
            }
            if lane==0 {*counts.add(segment)=count;total+=count;}
            segment+=8;
        }
        if lane==0 {crate::ptx::store(sh+(warp*4) as u32,f32::from_bits(total));}
        crate::ptx::sync();
        if tid==0 {let mut sum=0u32;for w in 0..8 {sum+=crate::ptx::load(sh+w*4).to_bits();}*(p.counts as *mut u32).add(expert)=sum;}
        return;
    }
    let mut segment=warp;
    while segment<segments {
        let mut offset=*(p.starts as *const u32).add(expert);
        for j in 0..segment {offset+=*counts.add(j);}
        let mut base=segment*SPAN;
        let end=((segment+1)*SPAN).min(p.pair_count as usize);
        while base<end {
            let pair=base+lane;
            let live=pair<end && *ids.add(pair)==expert as u32;
            let bits=crate::ptx::ballot(live);
            if live {
                let row=offset+(bits&((1u32<<lane)-1)).count_ones();
                *(p.sorted_row as *mut u32).add(row as usize)=pair as u32;
                // Per-token rank maps were already published by phase0.
            }
            offset+=bits.count_ones();base+=32;
        }
        segment+=8;
    }
}
