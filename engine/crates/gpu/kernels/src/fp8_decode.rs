//! Decode directly from checkpoint bytes; split K only inside each CTA.
use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_z, _thread_idx_x}};
use crate::ptx;

#[inline(always)]
unsafe fn promote<const FAST: bool>(raw: u16, scale: u32) -> u32 {
    let result: u32;
    if FAST {
        // Shifting FP16 fields represents FP8*2^-112 exactly, including zero.
        // The caller offsets the block scale's exponent by +112 once.
        asm!("{{ .reg .b32 h, magnitude;
            cvt.rn.f16x2.e4m3x2 h, {raw};
            shr.u32 magnitude, h, 3;
            and.b32 magnitude, magnitude, 0x0fff0fff;
            lop3.b32 magnitude, magnitude, h, 0x80008000, 0xf8;
            mul.rn.bf16x2 {result}, magnitude, {scale};
        }}", raw=in(reg16)raw, scale=in(reg32)scale, result=out(reg32)result);
    } else {
        // Very small or large scales cannot absorb the exponent offset.
        asm!("{{ .reg .b32 h; .reg .b16 lo, hi; .reg .f32 a, b, s;
            cvt.rn.f16x2.e4m3x2 h, {raw};
            mov.b32 {{lo, hi}}, h;
            cvt.f32.f16 a, lo;
            cvt.f32.f16 b, hi;
            mov.b32 s, {scale};
            mul.f32 a, a, s;
            mul.f32 b, b, s;
            cvt.rn.bf16x2.f32 {result}, b, a;
        }}", raw=in(reg16)raw, scale=in(reg32)scale, result=out(reg32)result);
    }
    result
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Input([u32; 4]);

#[inline(always)]
unsafe fn resident_fragment(p: *const u32) -> u32 {
    let value: u32;
    // Keep streamed weights out of L1, retaining the reused input fragments.
    asm!("ld.global.L1::no_allocate.L2::256B.u32 {value}, [{p}];",
        value=out(reg32)value, p=in(reg64)p, options(readonly, nostack));
    value
}

#[inline(always)]
unsafe fn resident_input(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("ld.global.L1::evict_last.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        a=out(reg32)a, b=out(reg32)b, c=out(reg32)c, d=out(reg32)d,
        p=in(reg64)p, options(readonly, nostack));
    [a,b,c,d]
}

#[inline(always)]
fn address(shared: u32, row: usize, col: usize) -> u32 {
    // Rotate 16-byte sectors so the eight MMA row groups use distinct banks.
    shared + (row * 128 + (col ^ (row * 16))) as u32
}

#[inline(always)]
unsafe fn stage(shared: u32, w: *const u8, k: usize, kb: usize, lane: usize) {
    let row = lane / 8;
    let col = lane % 8 * 16;
    crate::fp8_bf16::copy_weight16(address(shared, row, col), w.add(row * k + kb + col));
    crate::fp8_bf16::copy_weight16(address(shared, row + 4, col), w.add((row + 4) * k + kb + col));
}

#[inline(always)]
unsafe fn fragment(shared: u32, row: usize, col: usize) -> u32 {
    let (lo, hi): (u16, u16);
    asm!("ld.shared.b16 {lo}, [{p}]; ld.shared.b16 {hi}, [{p}+8];",
        lo=out(reg16)lo, hi=out(reg16)hi, p=in(reg32)address(shared,row,col));
    lo as u32 | ((hi as u32) << 16)
}

#[inline(always)]
unsafe fn step<const FAST: bool>(raw: u32, a: [u32; 4], scale: u32, sum: [f32; 4]) -> [f32; 4] {
    let b0 = promote::<FAST>(raw as u16, scale);
    let b1 = promote::<FAST>((raw >> 16) as u16, scale);
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}},
        {{{b0}, {b1}}}, {{{c0}, {c1}, {c2}, {c3}}};",
        d0=out(reg32)d0, d1=out(reg32)d1, d2=out(reg32)d2, d3=out(reg32)d3,
        a0=in(reg32)a[0], a1=in(reg32)a[1], a2=in(reg32)a[2], a3=in(reg32)a[3],
        b0=in(reg32)b0, b1=in(reg32)b1,
        c0=in(reg32)sum[0], c1=in(reg32)sum[1], c2=in(reg32)sum[2], c3=in(reg32)sum[3]);
    [d0, d1, d2, d3]
}

/// BF16 math on resident FP8 weights, without a BF16 weight allocation.
/// # Safety
/// 1<=M<=16; N/K positive multiples of128. Finite weights [E,N,K], BF16 scales
/// [E,N/128,K/128], packed BF16 X [E,K/16,32,4] u32, FP32 Y [E,M,N].
/// Grid (N/8,1,E), block128, static shared18432. X pads missing M rows to zero.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_fp8_bf16_decode(
    p: crate::fp8_abi::DecodeParams,
) {
    project::<false>(p, ptx::shared::<18432>());
}

#[inline(always)]
unsafe fn project<const FRAGMENTS: bool>(p: crate::fp8_abi::DecodeParams, shared: u32) {
    let (m, n, k) = (p.tokens as usize, p.rows as usize, p.cols as usize);
    let (w, ws, x, y) = (p.weights as *const u8, p.scales as *const u16,
        p.input as *const u32, p.output as *mut f32);
    let expert = _block_idx_z() as usize;
    let tiles = if FRAGMENTS { 2 } else { 1 };
    let task = _block_idx_x() as usize * if FRAGMENTS { crate::fp8_abi::DENSE_TILE_GROUPS as usize } else { 1 }
        + if FRAGMENTS { _thread_idx_x() as usize / 128 } else { 0 };
    let shared = shared + if FRAGMENTS { (_thread_idx_x() as usize / 128 * 4096) as u32 } else { 0 };
    let chunks = k / 512;
    // Pair equal K rotations inside one 128-row scale block, so both tiles
    // retain the same scale load as the adjacent schedule.
    let (row_tile, distance) = if FRAGMENTS && k % 512 == 0 && chunks < 16 {
        let (base, slot) = (task / 8 * 16, task % 8);
        if chunks > 8 {
            let paired = 16 - chunks;
            if slot < paired { (base + slot, chunks) }
            else { (base + paired + (slot - paired) * 2, 1) }
        } else {
            let paired = 16 / (chunks * 2) * chunks;
            if slot < paired { (base + slot / chunks * chunks * 2 + slot % chunks, chunks) }
            else { (base + paired * 2 + (slot - paired) * 2, 1) }
        }
    } else { (task * tiles, 1) };
    let nb = row_tile * 8;
    let w = w.add(expert * n * k + nb * k);
    let ws = ws.add((expert * (n / 128) + nb / 128) * (k / 128));
    let x = x.add(expert * (k / 16) * 128);
    let y = y.add(expert * m * n);
    let mut sum = if p.packed_input != 0 { decode_tile::<true, FRAGMENTS>(shared,w,ws,x,k,m,row_tile,distance) }
        else { decode_tile::<false, FRAGMENTS>(shared,w,ws,x,k,m,row_tile,distance) };
    if p.round_output != 0 {
        for tile in &mut sum {
            for v in tile {
                *v = f32::from_bits(v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff0000);
            }
        }
    }
    let lane=_thread_idx_x() as usize % 128;
    if lane<32 {
        let row=lane/4;
        for tile in 0..tiles {
            let col=nb+tile*distance*8+lane%4*2;
            let v=sum[tile];
            if row<m { *y.add(row*n+col)=v[0]; *y.add(row*n+col+1)=v[1]; }
            if row+8<m { *y.add((row+8)*n+col)=v[2]; *y.add((row+8)*n+col+1)=v[3]; }
        }
    }
}

#[inline(always)]
unsafe fn prefetch(w: *const u8, k: usize, kb: usize, lane: usize, distance: usize) {
    for tile in 0..2 {
        for z in 0..8 {
            let p = w.cast::<u32>().add(tile * distance * k * 2 + (kb / 16 + z) * 32 + lane);
            asm!("prefetch.global.L2 [{p}];", p=in(reg64)p);
        }
    }
}

#[inline(always)]
unsafe fn decode_group<const PACKED: bool, const FRAGMENTS: bool, const FIRST: bool, const SECOND: bool>(
    shared: u32, w: *const u8, ws: *const u16, input: *const u32, k: usize, m: usize,
    kb: usize, lane: usize, distance: usize, mut sum: [[f32;4];2],
) -> [[f32;4];2] {
    let x = input.add(lane * 4);
    macro_rules! raw {
        ($z:expr, $tile:expr) => {
            if FRAGMENTS { resident_fragment(w.cast::<u32>().add($tile * distance * k * 2 + (kb / 16 + $z) * 32 + lane)) }
            else { fragment(shared, lane / 4, $z * 16 + lane % 4 * 2) }
        };
    }
    macro_rules! weights {
        ($tile:expr) => { [raw!(0,$tile),raw!(1,$tile),raw!(2,$tile),raw!(3,$tile),
            raw!(4,$tile),raw!(5,$tile),raw!(6,$tile),raw!(7,$tile)] };
    }
    macro_rules! input {
        ($z:expr) => {
            if PACKED && FRAGMENTS { resident_input(x.add((kb / 16 + $z) * 128)) }
            else if PACKED { (*x.add((kb / 16 + $z) * 128).cast::<Input>()).0 }
            else {
                let row = lane / 4;
                let col = kb + $z * 16 + lane % 4 * 2;
                let pair = |r: usize, c: usize| {
                    if r >= m { 0 } else {
                        let p = input.cast::<f32>().add(r * k + c);
                        let round = |v: f32| ((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) >> 16) as u16;
                        round(*p) as u32 | ((round(*p.add(1)) as u32) << 16)
                    }
                };
                [pair(row, col), pair(row + 8, col), pair(row, col + 8), pair(row + 8, col + 8)]
            }
        };
    }
    let raw = if FIRST { weights!(0) } else { [0;8] };
    let raw_second = if SECOND { weights!(1) } else { [0;8] };
    let a = [input!(0),input!(1),input!(2),input!(3),input!(4),input!(5),input!(6),input!(7)];
    macro_rules! paired_part {
        ($fast:expr, $scale:expr, $z:expr) => {{
            if FIRST { sum[0] = step::<$fast>(raw[$z], a[$z], $scale, sum[0]); }
            if SECOND { sum[1] = step::<$fast>(raw_second[$z], a[$z], $scale, sum[1]); }
        }};
    }
    macro_rules! paired_parts {
        ($fast:expr, $scale:expr) => {{
            paired_part!($fast,$scale,0); paired_part!($fast,$scale,1);
            paired_part!($fast,$scale,2); paired_part!($fast,$scale,3);
            paired_part!($fast,$scale,4); paired_part!($fast,$scale,5);
            paired_part!($fast,$scale,6); paired_part!($fast,$scale,7);
        }};
    }
    let bits = *ws.add(kb / 128) as u32;
    if bits >= 0x0080 && bits < 0x4780 {
        let biased = bits + 0x3800;
        let scale = biased | (biased << 16);
        paired_parts!(true,scale);
    } else { paired_parts!(false,bits<<16); }
    sum
}

#[inline(always)]
unsafe fn decode_tile<const PACKED: bool, const FRAGMENTS: bool>(shared: u32, w: *const u8, ws: *const u16, x: *const u32,
    k: usize, m: usize, row_tile: usize, distance: usize) -> [[f32;4];2] {
    let tid=_thread_idx_x() as usize % 128;
    let (lane,warp)=(tid%32,tid/32);
    let raw_shared = shared + (warp * 4096) as u32;
    let tiles = if FRAGMENTS { 2 } else { 1 };
    let mut sum = [[0.0f32; 4];2];
    // Four warp-private stages hide DRAM latency; rotate K to spread requests.
    let chunks = (k / 128 + 3 - warp) / 4;
    let offset = row_tile % chunks.max(1);
    let delayed = FRAGMENTS && (k % 512 != 0 || distance != chunks);
    let mut iteration = 0;
    let mut group = offset;
    let mut kb = offset * 512 + warp * 128;
    if !FRAGMENTS && kb < k {
        stage(raw_shared, w, k, kb, lane);
        ptx::commit();
        if chunks > 1 {
            let second = if offset + 1 == chunks { 0 } else { offset + 1 };
            stage(raw_shared + 1024, w, k, second * 512 + warp * 128, lane);
            ptx::commit();
            if chunks > 2 {
                let third = if second + 1 == chunks { 0 } else { second + 1 };
                stage(raw_shared + 2048, w, k, third * 512 + warp * 128, lane);
                ptx::commit();
                asm!("cp.async.wait_group 2;");
            } else { ptx::wait_one(); }
        } else { ptx::wait(); }
        ptx::warp_sync();
    }
    // Peel delayed tiles' edges; the interior always computes both complete
    // groups, in the same rotated K order as each tile's canonical schedule.
    if FRAGMENTS && delayed && kb < k {
        let next = if offset + 1 == chunks { 0 } else { offset + 1 };
        if chunks > 1 { prefetch(w,k,next * 512 + warp * 128,lane,distance); }
        sum = decode_group::<PACKED, FRAGMENTS, true, false>(raw_shared,w,ws,x,k,m,kb,lane,distance,sum);
        iteration = 1;
        group = next;
        kb = next * 512 + warp * 128;
    }
    while iteration < chunks {
        let current = raw_shared + ((iteration % 4) * 1024) as u32;
        let next_group = if group + 1 == chunks { 0 } else { group + 1 };
        let next_kb = next_group * 512 + warp * 128;
        if FRAGMENTS && iteration + 1 < chunks {
            prefetch(w,k,next_kb,lane,distance);
        }
        if !FRAGMENTS && iteration + 3 < chunks {
            let ahead = if next_group + 1 == chunks { 0 } else { next_group + 1 };
            let ahead = if ahead + 1 == chunks { 0 } else { ahead + 1 };
            stage(raw_shared + (((iteration + 3) % 4) * 1024) as u32, w, k, ahead * 512 + warp * 128, lane);
            ptx::commit();
        }
        sum = decode_group::<PACKED, FRAGMENTS, true, FRAGMENTS>(current,w,ws,x,k,m,kb,lane,distance,sum);
        if !FRAGMENTS {
            if iteration + 3 < chunks { asm!("cp.async.wait_group 2;"); }
            else if iteration + 2 < chunks { ptx::wait_one(); } else { ptx::wait(); }
            ptx::warp_sync();
        }
        iteration += 1;
        group = next_group;
        kb = next_kb;
    }
    if FRAGMENTS && delayed && kb < k {
        sum = decode_group::<PACKED, FRAGMENTS, false, true>(raw_shared,w,ws,x,k,m,kb,lane,distance,sum);
    }
    let shared = shared + if FRAGMENTS { 0 } else { 16384 };
    for tile in 0..tiles {
        for component in 0..4 {
            ptx::store(shared + ((tile * 128 * 4 + tid * 4 + component) * 4) as u32, sum[tile][component]);
        }
    }
    ptx::sync();
    if warp == 0 {
        for tile in 0..tiles {
            for component in 0..4 {
                let p = shared + ((tile * 128 * 4 + lane * 4 + component) * 4) as u32;
                sum[tile][component] = (ptx::load(p) + ptx::load(p + 512))
                    + (ptx::load(p + 1024) + ptx::load(p + 1536));
            }
        }
    }
    sum
}

/// Coalesced resident FP8 fragments with the same promotion and K reduction.
/// Activations may be packed MMA operands or dense FP32 rows rounded to BF16.
/// # Safety
/// 1<=M<=16; N/K multiples of128; G=DENSE_TILE_GROUPS; grid(N/(16*G),1,E),
/// block128*G, shared4096*G. Each group retains its own canonical four warps.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_fp8_bf16_dense_decode(
    p: crate::fp8_abi::DecodeParams,
) {
    project::<true>(p, ptx::shared::<{4096 * crate::fp8_abi::DENSE_TILE_GROUPS as usize}>());
}

/// Single-token routed down projection, one task per selected expert.
/// Grid(320,1,tasks), block128. Packed SiLU*up input, FP32 grouped output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_down_decode(p: crate::fp8_abi::GroupedParams) {
    let shared = ptx::shared::<18432>();
    let (w, x, y) = (p.weights as *const u8, p.input as *const u32, p.output as *mut f32);
    let (counts, starts) = (p.route.counts as *const u32, p.route.starts as *const u32);
    let task_count = p.route.tasks as *const u32;
    let tasks = task_count.add(1);
    let n = p.cols;

    let task=_block_idx_z() as usize;
    if task>=*task_count as usize { return; }
    let expert=*tasks.add(task*2) as usize;
    let m=*counts.add(expert) as usize;
    let start=*starts.add(expert) as usize;
    let n = n as usize;
    let table = w.cast::<u64>().add(expert * 7);
    let k = *table.add(6) as usize;
    if k == 0 { return; }
    let nb=_block_idx_x() as usize*8;
    let w=(*table.add(4) as *const u8).add(nb*k);
    let ws=(*table.add(5) as *const u16).add(nb/128*(k/128));
    let x=x.add(start*640/2);
    let sum=decode_tile::<true, false>(shared,w,ws,x,k,m,nb/8,1)[0];
    let tid=_thread_idx_x() as usize;
    if tid<32 {
        let row=tid/4;
        let col=nb+tid%4*2;
        let y=y.add(start*n);
        if row<m { *y.add(row*n+col)=sum[0]; *y.add(row*n+col+1)=sum[1]; }
        if row+8<m { *y.add((row+8)*n+col)=sum[2]; *y.add((row+8)*n+col+1)=sum[3]; }
    }
}
