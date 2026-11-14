//! Fused causal attention: BF16 Q/K/V and P, native FP32 tensor-core accumulation.
#![allow(non_upper_case_globals)] // Constant unroll indices.
use super::*;
use crate::ptx;
macro_rules! each_query {
    ($dim:expr,$n:ident,$body:block) => {{
        unroll!(8, j, { let $n = j; $body });
        if $dim == 256 { unroll!(8, j, { let $n = j + 8; $body }); }
    }};
}
macro_rules! each_output {
    ($dim:expr,$n:ident,$body:block) => {{
        each16!($n, $body);
        if $dim == 256 { each16!(j, { let $n = j + 16; $body }); }
    }};
}
macro_rules! each_score {
    ($dim:expr,$n:ident,$body:block) => {{
        unroll!(8, j, { let $n = j; $body });
        if $dim == 128 { unroll!(8, j, { let $n = j + 8; $body }); }
    }};
}


// Constant indices also make ldmatrix swizzles and offsets immediate operands.
macro_rules! unroll {
    ($count:expr, $n:ident, $body:block) => {{
        { const $n: usize = 0; if $n < $count $body }
        { const $n: usize = 1; if $n < $count $body }
        { const $n: usize = 2; if $n < $count $body }
        { const $n: usize = 3; if $n < $count $body }
        { const $n: usize = 4; if $n < $count $body }
        { const $n: usize = 5; if $n < $count $body }
        { const $n: usize = 6; if $n < $count $body }
        { const $n: usize = 7; if $n < $count $body }
    }};
}
macro_rules! each16 {
    ($n:ident,$body:block) => {{
        unroll!(8, each_index, { const $n: usize = each_index; $body });
        unroll!(8, each_index, { const $n: usize = each_index + 8; $body });
    }};
}
#[inline(always)]
fn pair(a: f32, b: f32) -> u32 {
    let packed: u32;
    unsafe { core::arch::asm!("cvt.rn.bf16x2.f32 {p}, {b}, {a};", p = out(reg32) packed,
        a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)); }
    packed
}
#[inline(always)]
unsafe fn attention_exp<const BASE2: bool>(x: f32) -> f32 {
    if BASE2 {
        let y: f32;
        core::arch::asm!("ex2.approx.ftz.f32 {y}, {x};", y=out(reg32)y, x=in(reg32)x);
        y
    } else { ptx::exp(x) }
}
#[inline(always)]
fn kv_shared<const DIM: usize>(row: usize, col: usize) -> usize {
    row * DIM + (col ^ ((row & 7) * 8))
}
// Keep address arithmetic next to ldmatrix instead of retaining each unrolled
// swizzle address in a register across the tile loop.
#[inline(always)]
unsafe fn matrix_pair<const TRANS: bool, const XOR: u32, const OFFSET: u32>(sh: u32, lane: u32) -> [[u32; 2]; 2] {
    let lane = sh + lane;
    let (a, b, c, d): (u32, u32, u32, u32);
    if TRANS {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{a}, {b}, {c}, {d}}}, [p+{offset}]; }}",
            a=out(reg32)a, b=out(reg32)b, c=out(reg32)c, d=out(reg32)d,
            lane=in(reg32)lane, xor=const XOR, offset=const OFFSET);
    } else {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a}, {b}, {c}, {d}}}, [p+{offset}]; }}",
            a=out(reg32)a, b=out(reg32)b, c=out(reg32)c, d=out(reg32)d,
            lane=in(reg32)lane, xor=const XOR, offset=const OFFSET);
    }
    [[a, b], [c, d]]
}
#[inline(always)]
unsafe fn copy_packed<const DST: u32, const SRC: u32>(sh: u32, k: *const u16, v: *const u16) {
    core::arch::asm!(
        "cp.async.cg.shared.global [{sh}+{kd}], [{k}+{src}], 16;",
        "cp.async.cg.shared.global [{sh}+{vd}], [{v}+{src}], 16;",
        sh=in(reg32)sh, k=in(reg64)k, v=in(reg64)v,
        kd=const DST, vd=const (16384 + DST), src=const SRC);
}
#[inline(always)]
unsafe fn stage_half<const DST: u32, const NEXT: u32>(src: *const u16, pos: u32, kv: u32, head: u32, tid: u32) {
    let tid = tid & 127;
    // Keep four transient global addresses out of the tile loop's live state.
    core::arch::asm!(
        "{{ .reg .u64 p; .reg .u32 at; add.u32 at, {pos}, {next}; shr.u32 at, at, 5;",
        "mad.lo.u32 at, at, {kv}, {head}; shl.b32 at, at, 12; mad.lo.u32 at, {tid}, 8, at;",
        "mad.wide.u32 p, at, 2, {src};",
        "cp.async.cg.shared.global [{sh}+{d0}], [p], 16;",
        "cp.async.cg.shared.global [{sh}+{d1}], [p+2048], 16;",
        "cp.async.cg.shared.global [{sh}+{d2}], [p+4096], 16;",
        "cp.async.cg.shared.global [{sh}+{d3}], [p+6144], 16; }}",
        sh=in(reg32)(tid * 16), src=in(reg64)src, tid=in(reg32)tid,
        pos=in(reg32)pos, kv=in(reg32)kv, head=in(reg32)head, next=const NEXT,
        d0=const DST, d1=const (DST + 2048), d2=const (DST + 4096), d3=const (DST + 6144));
}
#[inline(always)]
// Stage GQA4's first K/V tile with transient addresses outside the live fragments.
#[inline(always)]
#[allow(named_asm_labels)]
unsafe fn refill(sh: u32, k: *const u16, v: *const u16,
    pos: usize, end: usize, kv: usize, head: usize, tid: usize) {
    let at = (pos * kv + head) * 128;
    core::arch::asm!("{{
        .reg .u32 i, r, c, d, s, row;
        .reg .u64 offset, ks, vs;
        .reg .pred valid, more;
        mov.u32 i, {tid};
        refill_loop:
        shr.u32 r, i, 4;
        and.b32 c, i, 15;
        and.b32 s, r, 7;
        xor.b32 s, s, c;
        shl.b32 s, s, 4;
        shl.b32 d, r, 8;
        add.u32 d, d, s;
        add.u32 d, d, {sh};
        add.u32 row, r, {pos};
        setp.lt.u32 valid, row, {end};
        @!valid bra refill_zero;
        mul.wide.u32 offset, r, {stride};
        shl.b32 c, c, 4;
        cvt.u64.u32 ks, c;
        add.u64 offset, offset, ks;
        add.u64 ks, {k}, offset;
        add.u64 vs, {v}, offset;
        cp.async.cg.shared.global [d], [ks], 16;
        add.u32 d, d, 16384;
        cp.async.cg.shared.global [d], [vs], 16;
        bra refill_next;
        refill_zero:
        st.shared.v4.u32 [d], {{0, 0, 0, 0}};
        add.u32 d, d, 16384;
        st.shared.v4.u32 [d], {{0, 0, 0, 0}};
        refill_next:
        add.u32 i, i, 128;
        setp.lt.u32 more, i, 1024;
        @more bra refill_loop;
    }}", sh=in(reg32)sh, tid=in(reg32)tid as u32,
        pos=in(reg32)pos as u32, end=in(reg32)end as u32,
        stride=in(reg32)(kv * 256) as u32,
        k=in(reg64)k.add(at), v=in(reg64)v.add(at));
}
// Refill a retired K/V region directly from the resident token-major cache.
#[inline(always)]
#[allow(named_asm_labels)]
unsafe fn refill_half<const DST: u32, const NEXT: u32, const COUNT: u32>(src: *const u16,
    pos: u32, end: u32, kv: u32, head: u32, tid: u32, prefix: Option<(*const u16, usize)>) {
    let (src, pos, end) = if let Some((shared, base)) = prefix {
        let base = base as u32;
        if pos + NEXT >= base {
            (src, pos.wrapping_sub(base), end.saturating_sub(base))
        } else if pos + NEXT + COUNT / 16 <= base {
            (shared, pos, end.min(base))
        } else {
            // A prefix boundary may bisect a refill. Keep the same swizzle.
            let mut i = tid;
            while i < COUNT {
                let row = pos + NEXT + i / 16;
                let dst = DST + (kv_shared::<128>((i / 16) as usize, (i % 16 * 8) as usize) * 2) as u32;
                if row < end {
                    let (src, row) = if row < base { (shared, row) } else { (src, row - base) };
                    ptx::copy16(dst, src.add(((row * kv + head) * 128 + i % 16 * 8) as usize).cast());
                } else {
                    for j in 0..4 { ptx::store(dst + j * 4, 0.0); }
                }
                i += 128;
            }
            return;
        }
    } else { (src, pos, end) };
    core::arch::asm!("{{
        .reg .u32 i, r, c, d, s, row, bytes, stride;
        .reg .u64 offset, p, step;
        .reg .pred valid, more;
        shr.u32 r, {tid}, 4;
        and.b32 c, {tid}, 15;
        and.b32 s, r, 7;
        xor.b32 s, s, c;
        shl.b32 s, s, 4;
        shl.b32 d, r, 8;
        add.u32 d, d, s;
        add.u32 d, d, {dst};
        add.u32 row, r, {pos};
        add.u32 row, row, {next};
        mad.lo.u32 r, row, {kv}, {head};
        shl.b32 r, r, 8;
        shl.b32 c, c, 4;
        add.u32 r, r, c;
        cvt.u64.u32 offset, r;
        add.u64 p, {src}, offset;
        shl.b32 stride, {kv}, 11;
        cvt.u64.u32 step, stride;
        add.u32 r, row, {last};
        setp.lt.u32 valid, r, {end};
        @!valid bra half_masked;
        cp.async.cg.shared.global [d], [p], 16;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+2048], [p], 16;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+4096], [p], 16;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+6144], [p], 16;
        setp.eq.u32 more, {count}, 4;
        @more bra half_done;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+8192], [p], 16;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+10240], [p], 16;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+12288], [p], 16;
        add.u64 p, p, step;
        cp.async.cg.shared.global [d+14336], [p], 16;
        bra half_done;
        half_masked:
        mov.u32 i, 0;
        half_loop:
        setp.lt.u32 valid, row, {end};
        selp.u32 bytes, 16, 0, valid;
        cp.async.cg.shared.global [d], [p], 16, bytes;
        add.u64 p, p, step;
        add.u32 d, d, 2048;
        add.u32 row, row, 8;
        add.u32 i, i, 1;
        setp.lt.u32 more, i, {count};
        @more bra half_loop;
        half_done:
    }}", tid=in(reg32)tid, pos=in(reg32)pos, end=in(reg32)end,
        kv=in(reg32)kv, head=in(reg32)head, src=in(reg64)src,
        dst=const DST, next=const NEXT, count=const (COUNT / 128), last=const (COUNT / 16 - 8));
}
// One shared K/V tile serves every query head in the KV group.
#[inline(always)]
unsafe fn stage<const RAW: bool, const PACKED: bool, const DIM: usize>(sh: u32, k: *const u16, v: *const u16,
    pos: usize, end: usize, kv: usize, head: usize, tid: usize, threads: usize,
    prefix: Option<(*const u16, *const u16, usize)>) {
    if DIM == 256 {
        // Native heads retain the full-tile copy path; only the last tile masks.
        if pos + 8192 / DIM <= end {
            stage_bf16::<true, DIM>(sh,k,v,pos,end,kv,head,tid,threads);
        } else {
            stage_bf16::<false, DIM>(sh,k,v,pos,end,kv,head,tid,threads);
        }
        return;
    }
    if let Some((pk, pv, base)) = prefix {
        let mut i = tid;
        while i < 1024 {
            let row = pos + i / 16;
            let dst = sh + (kv_shared::<128>(i / 16, i % 16 * 8) * 2) as u32;
            if row < end {
                let (k, v, row) = if row < base { (pk, pv, row) } else { (k, v, row - base) };
                let at = (row * kv + head) * 128 + i % 16 * 8;
                ptx::copy16(dst, k.add(at).cast()); ptx::copy16(dst + 16384, v.add(at).cast());
            } else {
                for j in 0..4 { ptx::store(dst + j * 4, 0.0); ptx::store(dst + 16384 + j * 4, 0.0); }
            }
            i += threads;
        }
    } else if PACKED {
        let (pos, kv, head) = (pos as u32, kv as u32, head as u32);
        // The packed entry is launched only with four warps and GQA8.
        let tid = tid as u32 & 127;
        let at = (pos / 32 * kv + head) * 4096 + tid * 8;
        let (k0, v0) = (k.add(at as usize), v.add(at as usize));
        let (k1, v1) = (k0.add(kv as usize * 4096), v0.add(kv as usize * 4096));
        unroll!(8, j, {
            copy_packed::<{ (j * 2048) as u32 }, { (j % 4 * 2048) as u32 }>(sh + tid * 16,
                if j < 4 { k0 } else { k1 }, if j < 4 { v0 } else { v1 });
        });
    } else if RAW {
        let mut i = tid;
        while i < 4096 {
            let row = pos + i / 64;
            let col = i % 64 * 2;
            let at = (row * kv + head) * 128 + col;
            let dst = sh + (kv_shared::<128>(i / 64, col) * 2) as u32;
            let (a, b) = if row < end {
                (pair(*k.cast::<f32>().add(at), *k.cast::<f32>().add(at + 1)),
                 pair(*v.cast::<f32>().add(at), *v.cast::<f32>().add(at + 1)))
            } else { (0, 0) };
            ptx::store(dst, f32::from_bits(a));
            ptx::store(dst + 16384, f32::from_bits(b));
            i += threads;
        }
    } else if pos + 64 <= end {
        stage_bf16::<true, 128>(sh, k, v, pos, end, kv, head, tid, threads);
    } else {
        stage_bf16::<false, 128>(sh, k, v, pos, end, kv, head, tid, threads);
    }
}
/// GQA8: block128/grid(ceil(T/16),2*KV), shared32768.
/// Other groups dividing eight: block256/grid(ceil(T/(128/group)),KV).
/// Remaining groups: block128/grid(ceil(T/64),H), shared65536.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_bf16(a: crate::kernel_params::PrefillAttention) {
    let (q, k, v, o) = (a.query as *const f32, a.keys as *const u16, a.values as *const u16, a.output as *mut f32);
    if a.exp2 != 0 {
        prefill_attention::<false, false, false, false, true, 128>(q, k, v, o, a.heads, a.kv_heads, a.rows, a.position, a.capacity, None, None);
    } else {
        prefill_attention::<false, false, false, false, false, 128>(q, k, v, o, a.heads, a.kv_heads, a.rows, a.position, a.capacity, None, None);
    }
}
/// GQA4: four warps, longest-first query tiles and overlapped direct K/V refills.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_gqa4_bf16(
    q: *const f32, k: *const u16, v: *const u16, o: *mut f32,
    h: u32, kv: u32, t: u32, base: u32, capacity: u32,
) { prefill_attention::<false, false, false, true, false, 128>(q, k, v, o, h, kv, t, base, capacity, None, None); }
/// GQA4 output in the O projection's BF16 fragment layout.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_gqa4_fragments(a: crate::kernel_params::PrefillAttention) {
    prefill_attention::<false, false, true, true, false, 128>(a.query as *const f32, a.keys as *const u16,
        a.values as *const u16, a.output as *mut f32, a.heads, a.kv_heads, a.rows, a.position, a.capacity, None, None);
}
/// GQA8 swizzled head-major BF16 cache tiles, identical QK/PV arithmetic.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_swizzled_bf16(
    q: *const f32, k: *const u16, v: *const u16, o: *mut f32,
    h: u32, kv: u32, t: u32, base: u32, capacity: u32,
) { prefill_attention::<false, true, false, false, true, 128>(q, k, v, o, h, kv, t, base, capacity, None, None); }
/// GQA8 expert prefill writes the O projection's BF16 fragments directly.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_swizzled_packed_bf16(
    q: *const f32, k: *const u16, v: *const u16, o: *mut f32,
    h: u32, kv: u32, t: u32, base: u32, capacity: u32,
) { prefill_attention::<false, true, true, false, true, 128>(q, k, v, o, h, kv, t, base, capacity, None, None); }
/// Ragged prompt tiles retain the scalar kernel's head geometry and arithmetic.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_multi_bf16(a: crate::kernel_params::PrefillAttentionMulti) {
    let (q, metadata, tiles, o) = (a.query as *const f32, a.metadata as *const u32, a.tiles as *const u32, a.output as *mut f32);
    let (h, kv) = (a.heads, a.kv_heads);
    let tile = tiles.add(_block_idx_x() as usize * 2);
    let m = metadata.add(*tile as usize * 12);
    let key = (*m as u64 | ((*m.add(1) as u64) << 32)) as *const u16;
    let value = (*m.add(2) as u64 | ((*m.add(3) as u64) << 32)) as *const u16;
    let (base, rows, first) = (*m.add(4), *m.add(5), *m.add(6) as usize);
    let pk = (*m.add(7) as u64 | ((*m.add(8) as u64) << 32)) as *const u16;
    let pv = (*m.add(9) as u64 | ((*m.add(10) as u64) << 32)) as *const u16;
    let prefix = if *m.add(11) == 0 { None } else { Some((pk, pv, *m.add(11) as usize)) };
    let offset = first * h as usize * 128;
    let group = h / kv;
    let tile_rows = if group == 4 || group == 8 { 16 } else if group <= 8 && 8 % group == 0 { 128 / group } else { 64 };
    let block_row = if group == 8 { *tile.add(1) * tile_rows }
        else { ((rows + tile_rows - 1) / tile_rows - 1 - *tile.add(1)) * tile_rows };
    if group == 4 {
        prefill_attention::<false, false, false, true, false, 128>(q.add(offset), key, value, o.add(offset), h, kv, rows, base, base + rows, Some(block_row), prefix);
    } else if a.exp2 != 0 {
        prefill_attention::<false, false, false, false, true, 128>(q.add(offset), key, value, o.add(offset), h, kv, rows, base, base + rows, Some(block_row), prefix);
    } else {
        prefill_attention::<false, false, false, false, false, 128>(q.add(offset), key, value, o.add(offset), h, kv, rows, base, base + rows, Some(block_row), prefix);
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_pack_swizzled_bf16(
    k: *const u16, v: *const u16, kp: *mut u16, vp: *mut u16, kv: u32, end: u32,
) {
    let i = crate::index();
    if i >= (end as usize + 63) / 64 * 64 * kv as usize * 64 { return; }
    let (col, head, row) = (i % 64 * 2, i / 64 % kv as usize, i / (kv as usize * 64));
    let at = (row * kv as usize + head) * 128 + col;
    let dst = (row / 32 * kv as usize + head) * 4096 + kv_shared::<128>(row % 32, col);
    *kp.add(dst).cast::<u32>() = if row < end as usize { *k.add(at).cast::<u32>() } else { 0 };
    *vp.add(dst).cast::<u32>() = if row < end as usize { *v.add(at).cast::<u32>() } else { 0 };
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_pack_kv(
    k: *const f32, v: *const f32, kp: *mut u16, vp: *mut u16, n: u32, format: u32,
) {
    let i = crate::index();
    if i < n as usize {
        *kp.add(i) = round_bf16(crate::layer::batch::cache_load(k, i, format));
        *vp.add(i) = round_bf16(crate::layer::batch::cache_load(v, i, format));
    }
}
#[inline(always)]
unsafe fn prefill_attention<const RAW: bool, const PACKED: bool, const OUTPUT_PACKED: bool, const GQA4: bool, const EXP2: bool, const DIM: usize>(
    q: *const f32, k: *const u16, v: *const u16, o: *mut f32,
    h: u32, kv: u32, t: u32, base: u32, capacity: u32, tile_row: Option<u32>,
    prefix: Option<(*const u16, *const u16, usize)>,
) {
    if DIM == 256 {

    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let group = h as usize / kv as usize;
    let six = DIM == 256 && group == 6;
    let grouped = six || (group <= 8 && 8 % group == 0);
    let rows = if six { 16 } else if grouped { 128 / group } else { 64 };
    let threads = if six { 192 } else if grouped { 256 } else { 128 };
    let tile = 8192 / DIM;
    let head = if grouped { _block_idx_y() as usize * group + warp % group } else { _block_idx_y() as usize };
    let block_row = _block_idx_x() as usize * rows;
    let row = block_row + (if grouped { warp / group } else { warp }) * 16 + lane / 4;
    let col = lane % 4 * 2;
    let key_lane = (kv_shared::<DIM>(lane % 8, lane / 8 % 2 * 8) * 2) as u32;
    let value_lane = (kv_shared::<DIM>(lane % 16, 0) * 2) as u32;
    let end = (base as usize + (block_row + rows).min(t as usize)).min(capacity as usize);
    // No static shared declarations: the launch provides the dynamic region.
    let sh = 0u32;
    let mut query = [[0u32; 4]; 16];
    each_query!(DIM, d, {
        let at = d * 16 + col;
        let load = |r: usize, c: usize| if r < t as usize { *q.add((r * h as usize + head) * DIM + c) } else { 0.0 };
        query[d] = [pair(load(row, at), load(row, at + 1)), pair(load(row + 8, at), load(row + 8, at + 1)),
            pair(load(row, at + 8), load(row, at + 9)), pair(load(row + 8, at + 8), load(row + 8, at + 9))];
    });
    let mut out = [ZERO; 32];
    let mut maximum = [-f32::INFINITY; 2];
    let mut denom = [0f32; 2];
    stage::<RAW, false, DIM>(sh, k, v, 0, end, kv as usize, head / group, tid, threads, None);
    ptx::wait(); ptx::sync();
    let mut pos = 0;
    while pos < end {
        let buf = sh + ((pos / tile % 2) * 32768) as u32;
        if pos + tile < end {
            stage::<RAW, false, DIM>(sh + (((pos / tile + 1) % 2) * 32768) as u32,
                k, v, pos + tile, end, kv as usize, head / group, tid, threads, None);
        }
        let mut dot = [ZERO; 8];
        each_query!(DIM, d, {
            unroll!(8192 / DIM / 8, n, {
                let b = matrix::<false>(buf, key_lane, (d * 32) as u32, (n * DIM * 16) as u32);
                dot[n] = madd(query[d], b, dot[n]);
            });
        });
        let mut p = [[0f32; 16]; 2];
        let scale = if DIM == 256 { 0.0625 } else { 0.08838834764831845 };
        unroll!(8192 / DIM / 8, n, {
            p[0][n * 2] = dot[n].a * scale;
            p[0][n * 2 + 1] = dot[n].b * scale;
            p[1][n * 2] = dot[n].c * scale;
            p[1][n * 2 + 1] = dot[n].d * scale;
        });
        let mut packed_p = [[0u32; 16]; 2];
        let full = pos + tile <= base as usize + block_row && block_row + rows <= t as usize;
        unroll!(2, r, {
            let qr = row + r * 8;
            let mut mx = -f32::INFINITY;
            if !full {
                each_score!(DIM, n, {
                    let key = pos + n / 2 * 8 + col + n % 2;
                    if key >= end || key > base as usize + qr || qr >= t as usize { p[r][n] = -f32::INFINITY; }
                });
            }
            each_score!(DIM, n, { mx = mx.max(p[r][n]); });
            mx = mx.max(f32::from_bits(crate::shuffle_xor(u32::MAX, mx.to_bits(), 1, 31)));
            mx = mx.max(f32::from_bits(crate::shuffle_xor(u32::MAX, mx.to_bits(), 2, 31)));
            let nm = maximum[r].max(mx);
            // Masked scores exponentiate to zero. Padded query rows need a
            // finite subtraction operand because their maximum stays -inf.
            let shift = if nm == -f32::INFINITY { 0.0 } else { nm };
            let mut sum = 0.0;
            unroll!(8192 / DIM / 8, n, {
                let a = ptx::exp(p[r][n * 2] - shift);
                let b = ptx::exp(p[r][n * 2 + 1] - shift);
                // Build P once; PV and its divisor consume these same BF16 bits.
                let packed = pair(a, b);
                packed_p[r][n] = packed;
                sum += f32::from_bits(packed << 16);
                sum += f32::from_bits(packed & 0xffff0000);
            });
            sum += f32::from_bits(crate::shuffle_xor(u32::MAX, sum.to_bits(), 1, 31));
            sum += f32::from_bits(crate::shuffle_xor(u32::MAX, sum.to_bits(), 2, 31));
            if nm > maximum[r] {
                let scale = ptx::exp(maximum[r] - nm);
                denom[r] *= scale;
                each_output!(DIM, n, {
                    if r == 0 { out[n].a *= scale; out[n].b *= scale; }
                    else { out[n].c *= scale; out[n].d *= scale; }
                });
                maximum[r] = nm;
            }
            denom[r] += sum;
        });
        unroll!(8192 / DIM / 16, z, {
            let a = [packed_p[0][z * 2], packed_p[1][z * 2],
                packed_p[0][z * 2 + 1], packed_p[1][z * 2 + 1]];
            each_output!(DIM, n, {
                let b = matrix::<true>(buf + 16384, value_lane, (n * 16) as u32, (z * DIM * 32) as u32);
                out[n] = madd(a, b, out[n]);
            });
        });
        ptx::wait(); ptx::sync(); pos += tile;
    }
    // Validate complete FP32 outputs outside the repeated softmax tile loop.
    each_output!(DIM, n, {
        let a = out[n];
        if row < t as usize {
            if !a.a.is_finite() || !a.b.is_finite() || !denom[0].is_finite() || denom[0] <= 0.0 { core::arch::asm!("trap;"); }
            let at = (row * h as usize + head) * DIM + n * 8 + col;
            *o.add(at) = a.a / denom[0]; *o.add(at + 1) = a.b / denom[0];
        }
        if row + 8 < t as usize {
            if !a.c.is_finite() || !a.d.is_finite() || !denom[1].is_finite() || denom[1] <= 0.0 { core::arch::asm!("trap;"); }
            let at = ((row + 8) * h as usize + head) * DIM + n * 8 + col;
            *o.add(at) = a.c / denom[1]; *o.add(at + 1) = a.d / denom[1];
        }
    });

        return;
    }
    let tid = _thread_idx_x();
    let (lane, warp) = (tid % 32, tid / 32);
    let group = if PACKED { 8 } else if GQA4 { 4 } else { h / kv };
    let grouped = group <= 8 && 8 % group == 0;
    let single = group == 8 || GQA4;
    let warp_group = if group == 8 { 4 } else { group };
    let rows = if single { 16 } else if grouped { 128 / group } else { 64 };
    let threads = if grouped && !single { 256 } else { 128 };
    let head = if grouped { _block_idx_y() * warp_group + warp % warp_group } else { _block_idx_y() };
    // Longest-first query tiles (GQA4, swizzled GQA8) shorten the causal tail;
    // direct GQA8 keeps m32's paired-head order.
    let block_row = tile_row.unwrap_or_else(|| if group == 8 && !PACKED { _block_idx_x() * rows }
        else { ((t + rows - 1) / rows - 1 - _block_idx_x()) * rows });
    let row = block_row + (if grouped { warp / warp_group } else { warp }) * 16 + lane / 4;
    let col = lane % 4 * 2;
    let key_lane = (kv_shared::<128>((lane % 8) as usize, (lane / 8 % 2 * 8) as usize) * 2) as u32;
    let value_lane = (kv_shared::<128>((lane % 16) as usize, 0) * 2) as u32;
    let key_pair_lane = key_lane + lane / 16 * 2048;
    let value_pair_lane = value_lane ^ (lane / 16 * 16);
    let end = (base + (block_row + rows).min(t)).min(capacity);
    // No static shared declarations: the launch provides the dynamic region.
    let sh = 0u32;
    let mut query = [[0u32; 4]; 8];
    unroll!(8, d, {
        let at = d as u32 * 16 + col;
        let load = |r: u32, c: u32| if r < t { *q.add(((r * h + head) * 128 + c) as usize) } else { 0.0 };
        query[d] = [pair(load(row, at), load(row, at + 1)), pair(load(row + 8, at), load(row + 8, at + 1)),
            pair(load(row, at + 8), load(row, at + 9)), pair(load(row + 8, at + 8), load(row + 8, at + 9))];
    });
    let mut out = [ZERO; 16];
    let mut maximum = [-f32::INFINITY; 2];
    let mut denom = [0f32; 2];
    if GQA4 && prefix.is_none() { refill(sh, k, v, 0, end as usize, kv as usize, (head / group) as usize, tid as usize); }
    else { stage::<RAW, PACKED, 128>(sh, k, v, 0, end as usize, kv as usize, (head / group) as usize, tid as usize, threads, prefix); }
    ptx::wait(); ptx::sync();
    let mut pos = 0;
    while pos < end {
        let buf = sh + (if single { 0 } else { (pos / 64 % 2) * 32768 }) as u32;
        if !single && pos + 64 < end {
            stage::<RAW, PACKED, 128>(sh + (((pos / 64 + 1) % 2) * 32768) as u32,
                k, v, (pos + 64) as usize, end as usize, kv as usize, (head / group) as usize, tid as usize, threads, prefix);
        }
        let mut p = [[0f32; 16]; 2];
        // Four live score fragments; each dot product keeps the same K order.
        let scale = 0.08838834764831845 * if EXP2 { core::f32::consts::LOG2_E } else { 1.0 };
        unroll!(2, half, {
            let mut dot = [ZERO; 4];
            unroll!(8, d, {
                unroll!(2, n, {
                    let b = matrix_pair::<false, { (d * 32) as u32 }, { ((half * 4 + n * 2) * 2048) as u32 }>(buf, key_pair_lane);
                    dot[n * 2] = madd(query[d], b[0], dot[n * 2]);
                    dot[n * 2 + 1] = madd(query[d], b[1], dot[n * 2 + 1]);
                });
            });
            unroll!(4, n, {
                let at = half * 8 + n * 2;
                p[0][at] = dot[n].a * scale;
                p[0][at + 1] = dot[n].b * scale;
                p[1][at] = dot[n].c * scale;
                p[1][at + 1] = dot[n].d * scale;
            });
        });
        if GQA4 && pos + 64 < end {
            // All QK readers are done; hide K copies behind softmax and PV.
            ptx::sync();
            refill_half::<0, 64, 1024>(k, pos, end, kv, head / group, tid, prefix.map(|(k, _, base)| (k, base)));
        }
        let mut packed_p = [[0u32; 8]; 2];
        unroll!(2, r, {
            let qr = row + r as u32 * 8;
            // Only the diagonal tile needs per-element causal predicates.
            if pos + 64 > base + block_row || block_row + rows > t {
                each16!(n, {
                    let key = pos + n as u32 / 2 * 8 + col + n as u32 % 2;
                    if key >= end || key > base + qr || qr >= t {
                        p[r][n] = -f32::INFINITY;
                    }
                });
            }
            let mut mx = -f32::INFINITY;
            each16!(n, {
                mx = mx.max(p[r][n]);
            });
            mx = mx.max(f32::from_bits(crate::shuffle_xor(u32::MAX, mx.to_bits(), 1, 31)));
            mx = mx.max(f32::from_bits(crate::shuffle_xor(u32::MAX, mx.to_bits(), 2, 31)));
            let nm = maximum[r].max(mx);
            let shift = if nm == -f32::INFINITY { 0.0 } else { nm };
            let mut sum = 0.0;
            unroll!(8, n, {
                let a = attention_exp::<EXP2>(p[r][n * 2] - shift);
                let b = attention_exp::<EXP2>(p[r][n * 2 + 1] - shift);
                // Build P once; PV and its divisor consume these same BF16 bits.
                let packed = pair(a, b);
                packed_p[r][n] = packed;
                sum += f32::from_bits(packed << 16);
                sum += f32::from_bits(packed & 0xffff0000);
            });
            sum += f32::from_bits(crate::shuffle_xor(u32::MAX, sum.to_bits(), 1, 31));
            sum += f32::from_bits(crate::shuffle_xor(u32::MAX, sum.to_bits(), 2, 31));
            if nm > maximum[r] {
                let scale = attention_exp::<EXP2>(maximum[r] - nm);
                denom[r] *= scale;
                each16!(n, {
                    if r == 0 { out[n].a *= scale; out[n].b *= scale; }
                    else { out[n].c *= scale; out[n].d *= scale; }
                });
                maximum[r] = nm;
            }
            denom[r] += sum;
        });
        if PACKED && pos + 64 < end {
            ptx::sync();
            stage_half::<0, 64>(k, pos, kv, head / 8, tid);
            stage_half::<8192, 96>(k, pos, kv, head / 8, tid);
        }
        unroll!(4, z, {
            let a = [packed_p[0][z * 2], packed_p[1][z * 2],
                packed_p[0][z * 2 + 1], packed_p[1][z * 2 + 1]];
            unroll!(8, n, {
                let b = matrix_pair::<true, { (n * 32) as u32 }, { (16384 + z * 4096) as u32 }>(buf, value_pair_lane);
                out[n * 2] = madd(a, b[0], out[n * 2]);
                out[n * 2 + 1] = madd(a, b[1], out[n * 2 + 1]);
            });
            if PACKED && z % 2 == 1 && pos + 64 < end {
                ptx::sync();
                stage_half::<{ (16384 + z / 2 * 8192) as u32 }, { (64 + z / 2 * 32) as u32 }>(v, pos, kv, head / 8, tid);
            }
        });
        if GQA4 && pos + 64 < end {
            ptx::sync();
            refill_half::<16384, 64, 1024>(v, pos, end, kv, head / group, tid, prefix.map(|(_, v, base)| (v, base)));
        }
        if single && !PACKED && !GQA4 {
            ptx::sync();
            if pos + 64 < end {
                stage::<RAW, PACKED, 128>(sh, k, v, (pos + 64) as usize, end as usize, kv as usize, (head / group) as usize, tid as usize, threads, prefix);
            }
        }
        ptx::wait(); ptx::sync(); pos += 64;
    }
    // Nonfinite rows reach the ordinary logits validity flag on the host.
    if row < t && (!denom[0].is_finite() || denom[0] <= 0.0) { denom[0] = f32::NAN; }
    if row + 8 < t && (!denom[1].is_finite() || denom[1] <= 0.0) { denom[1] = f32::NAN; }
    if OUTPUT_PACKED {
        unroll!(8, n, {
            let (a, b) = (out[n * 2], out[n * 2 + 1]);
            let lo = |x: f32, y: f32| if row < t { pair(x / denom[0], y / denom[0]) } else { 0 };
            let hi = |x: f32, y: f32| if row + 8 < t { pair(x / denom[1], y / denom[1]) } else { 0 };
            let tile = row / 16 * (h * 8) + head * 8 + n as u32;
            *o.cast::<AFrag>().add((tile * 32 + lane) as usize) = AFrag([
                lo(a.a, a.b), hi(a.c, a.d), lo(b.a, b.b), hi(b.c, b.d),
            ]);
        });
        return;
    }
    each16!(n, {
        let a = out[n];
        if row < t {
            let at = ((row * h + head) * 128 + n as u32 * 8 + col) as usize;
            *o.add(at) = a.a / denom[0]; *o.add(at + 1) = a.b / denom[0];
        }
        if row + 8 < t {
            let at = (((row + 8) * h + head) * 128 + n as u32 * 8 + col) as usize;
            *o.add(at) = a.c / denom[1]; *o.add(at + 1) = a.d / denom[1];
        }
    });
}

#[inline(always)]
unsafe fn matrix<const TRANS: bool>(sh: u32, lane: u32, xor: u32, offset: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    if TRANS {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; add.u32 p, p, {offset}; add.u32 p, p, {sh}; ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{{a}, {b}}}, [p]; }}",
            a=out(reg32)a, b=out(reg32)b, sh=in(reg32)sh,
            lane=in(reg32)lane, xor=in(reg32)xor, offset=in(reg32)offset);
    } else {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; add.u32 p, p, {offset}; add.u32 p, p, {sh}; ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{a}, {b}}}, [p]; }}",
            a=out(reg32)a, b=out(reg32)b, sh=in(reg32)sh,
            lane=in(reg32)lane, xor=in(reg32)xor, offset=in(reg32)offset);
    }
    [a, b]
}

#[inline(always)]
unsafe fn stage_bf16_chunk<const FULL: bool, const DIM: usize>(sh: u32, k: *const u16, v: *const u16,
    pos: usize, end: usize, kv: usize, head: usize, i: usize) {
    let row = pos + i / (DIM / 8);
    let col = i % (DIM / 8) * 8;
    let dst = sh + (kv_shared::<DIM>(i / (DIM / 8), col) * 2) as u32;
    if FULL || row < end {
        let at = (row * kv + head) * DIM + col;
        ptx::copy16(dst, k.add(at).cast());
        ptx::copy16(dst + 16384, v.add(at).cast());
    } else {
        for j in 0..4 { ptx::store(dst + j * 4, 0.0); ptx::store(dst + 16384 + j * 4, 0.0); }
    }
}

#[inline(always)]
unsafe fn stage_bf16<const FULL: bool, const DIM: usize>(sh: u32, k: *const u16, v: *const u16,
    pos: usize, end: usize, kv: usize, head: usize, tid: usize, threads: usize) {
    if threads == 256 {
        unroll!(4, j, { stage_bf16_chunk::<FULL, DIM>(sh, k, v, pos, end, kv, head, tid + j * 256); });
    } else if threads == 192 {
        unroll!(5, j, { stage_bf16_chunk::<FULL, DIM>(sh, k, v, pos, end, kv, head, tid + j * 192); });
        if tid < 64 { stage_bf16_chunk::<FULL, DIM>(sh, k, v, pos, end, kv, head, tid + 960); }
    } else {
        unroll!(8, j, { stage_bf16_chunk::<FULL, DIM>(sh, k, v, pos, end, kv, head, tid + j * 128); });
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_256_bf16(a: crate::kernel_params::PrefillAttention) {
    prefill_attention::<false,false,false,false,false,256>(a.query as *const f32, a.keys as *const u16,
        a.values as *const u16, a.output as *mut f32, a.heads, a.kv_heads, a.rows, a.position, a.capacity, None, None);
}

// Wide heads share the flash kernel's ldmatrix/MMA layout. The score buffer
// separates QK from PV so normalization sees the entire row before BF16 rounding.
#[inline(always)]
unsafe fn wide_stage(src: *const u16, pos: usize, end: usize, head: usize,
    kv: usize, dim: usize, feature: usize, ring: usize) {
    let mut i = _thread_idx_x() as usize;
    while i < 1024 {
        let row = i / 16;
        let col = i % 16 * 8;
        let dst = (kv_shared::<128>(row, col) * 2) as u32;
        if pos + row < end {
            ptx::copy16(dst, src.add(((if ring == 0 {pos+row} else {((pos+row) as u32 % ring as u32) as usize})*kv+head)*dim+feature+col).cast());
        } else {
            for j in 0..4 { ptx::store(dst+j*4, 0.0); }
        }
        i += 128;
    }
    ptx::wait(); ptx::sync();
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_wide_scores(a: crate::kernel_params::BoundedAttention) {
    if a.dim == 256 { wide_scores::<256>(a); } else { wide_scores::<512>(a); }
}
#[inline(always)]
unsafe fn wide_scores<const D: usize>(a: crate::kernel_params::BoundedAttention) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid%32, tid/32);
    let (h,kv,_d) = (a.heads as usize,a.kv_heads as usize,D);
    let head = _block_idx_y() as usize;
    let block = _block_idx_x() as usize*64;
    let row = block+warp*16+lane/4;
    let col = lane%4*2;
    let end = (a.position as usize+(block+64).min(a.rows as usize)).min(a.position as usize+a.rows as usize);
    let begin = if a.window == 0 {0} else {(a.position as usize+block+1).saturating_sub(a.window as usize)};
    let key_lane = (kv_shared::<128>(lane%8,lane/8%2*8)*2+lane/16*2048) as u32;
    let mut query = [[0u32;4];32];
    unroll!(4,quarter,{
        if quarter < D/128 {
            unroll!(8,n,{
                const FEATURE: usize = quarter*128+n*16;
                let load = |r: usize, c: usize| if r < a.rows as usize {
                    *(a.query as *const f32).add((r*h+head)*D+c)
                } else {0.0};
                let c = FEATURE+col;
                query[quarter*8+n] = [pair(load(row,c),load(row,c+1)),pair(load(row+8,c),load(row+8,c+1)),
                    pair(load(row,c+8),load(row,c+9)),pair(load(row+8,c+8),load(row+8,c+9))];
            });
        }
    });
    let mut pos = begin;
    while pos < end {
        let mut dot = [ZERO;8];
        unroll!(4,quarter,{
            if quarter < D/128 {
                wide_stage(a.keys as *const u16,pos,end,head/(h/kv),kv,D,quarter*128,
                    if a.window == 0 {0} else {a.cache_capacity as usize});
                unroll!(8,feature,{
                    let q = query[quarter*8+feature];
                    unroll!(4,n,{
                        let k = matrix_pair::<false,0,0>((n*4096) as u32,key_lane ^ (feature*32) as u32);
                        dot[n*2] = madd(q,k[0],dot[n*2]);
                        dot[n*2+1] = madd(q,k[1],dot[n*2+1]);
                    });
                });
                ptx::sync();
            }
        });
        unroll!(8,n,{
            let value = dot[n];
            let key = pos+n*8+col;
            for r in 0..2 {
                let qr = row+r*8;
                if qr < a.rows as usize {
                    let absolute = a.position as usize+qr;
                    let start = if a.window == 0 {0} else {(absolute+1).saturating_sub(a.window as usize)};
                    let (x,y) = if r == 0 {(value.a,value.b)} else {(value.c,value.d)};
                    let out = (a.scores as *mut f32).add((qr*h+head)*a.capacity as usize);
                    if a.round_bf16 == 2 {
                        let out = (a.scores as *mut u16).add((qr*h+head)*a.capacity as usize);
                        if key >= start && key <= absolute { *out.add(key-start) = round_bf16(x*a.scale); }
                        if key+1 >= start && key+1 <= absolute { *out.add(key+1-start) = round_bf16(y*a.scale); }
                    } else {
                        if key >= start && key <= absolute { *out.add(key-start) = crate::layer::round_bf16(x*a.scale); }
                        if key+1 >= start && key+1 <= absolute { *out.add(key+1-start) = crate::layer::round_bf16(y*a.scale); }
                    }
                }
            }
        });
        ptx::sync(); pos += 64;
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_attention_wide_values(a: crate::kernel_params::BoundedAttention) {
    let tid = _thread_idx_x() as usize;
    let (lane,warp) = (tid%32,tid/32);
    let (h,kv,d) = (a.heads as usize,a.kv_heads as usize,a.dim as usize);
    let head = _block_idx_y() as usize;
    let block = _block_idx_x() as usize*64;
    let row = block+warp*16+lane/4;
    let col = lane%4*2;
    let feature = core::arch::nvptx::_block_idx_z() as usize*128;
    let begin = if a.window == 0 {0} else {(a.position as usize+block+1).saturating_sub(a.window as usize)};
    let end = crate::wide_weights::block_end(a.position as usize, a.rows as usize, block);
    let value_lane = (kv_shared::<128>(lane%16,0)*2) as u32 ^ (lane/16*16) as u32;
    let mut out = [ZERO;16];
    let mut pos = begin;
    while pos < end {
        let mut i = tid;
        while i < 1024 {
            let r = i/16; let c = i%16*8;
            let dst = (kv_shared::<128>(r,c)*2) as u32;
            if pos+r < end {
                ptx::copy16(dst,(a.values as *const u16).add(((if a.window == 0 {pos+r} else {((pos+r) as u32%a.cache_capacity) as usize})*kv+head/(h/kv))*d+feature+c).cast());
            } else { for j in 0..4 {ptx::store(dst+j*4,0.0);} }
            i += 128;
        }
        ptx::wait(); ptx::sync();
        unroll!(4,z,{
            let load = |r: usize, key: usize| {
                let absolute = a.position as usize+r;
                let start = if a.window == 0 {0} else {(absolute+1).saturating_sub(a.window as usize)};
                if r < a.rows as usize && key >= start && key <= absolute {
                    *(a.scores as *const f32).add((r*h+head)*a.capacity as usize+key-start)
                } else {0.0}
            };
            let key = pos+z*16+col;
            let p = if a.round_bf16 == 2 {
                (*(a.metadata as *const AFrag).add(((row/16*h+head)*(a.capacity as usize/16)+pos/16+z)*32+lane)).0
            } else {
                [pair(load(row,key),load(row,key+1)),pair(load(row+8,key),load(row+8,key+1)),
                 pair(load(row,key+8),load(row,key+9)),pair(load(row+8,key+8),load(row+8,key+9))]
            };
            unroll!(8,n,{
                let v = matrix_pair::<true,{(n*32) as u32},{(z*4096) as u32}>(0,value_lane);
                out[n*2] = madd(p,v[0],out[n*2]);
                out[n*2+1] = madd(p,v[1],out[n*2+1]);
            });
        });
        ptx::sync(); pos += 64;
    }
    each16!(n,{
        let value = out[n];
        if row < a.rows as usize {
            let at = ((row*h+head)*d+feature+n*8+col) as usize;
            *(a.output as *mut f32).add(at) = crate::layer::round_bf16(value.a);
            *(a.output as *mut f32).add(at+1) = crate::layer::round_bf16(value.b);
        }
        if row+8 < a.rows as usize {
            let at = (((row+8)*h+head)*d+feature+n*8+col) as usize;
            *(a.output as *mut f32).add(at) = crate::layer::round_bf16(value.c);
            *(a.output as *mut f32).add(at+1) = crate::layer::round_bf16(value.d);
        }
    });
}

/// Reject non-finite/overflowing operands before hd256 attention can trap.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_validate_256(p: crate::kernel_params::AttentionValidate) {
    let i = crate::index();
    let mut bad = false;
    if i < p.query_elements as usize {
        bad |= !crate::kernel_params::hd256_operand(*(p.query as *const f32).add(i));
    }
    if i < p.kv_elements as usize {
        for base in [p.keys, p.values] {
            bad |= !crate::kernel_params::hd256_operand(f32::from_bits(
                (*(base as *const u16).add(i) as u32) << 16));
        }
    }
    if bad { core::arch::asm!("atom.global.or.b32 {old}, [{address}], 1;",
        old=out(reg32) _, address=in(reg64)p.error); }
}
