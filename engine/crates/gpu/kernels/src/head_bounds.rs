//! Exact greedy target head: conservative MX8 bounds, canonical BF16 rescoring.
#[path = "../../src/head_bound.rs"]
mod bound_math;
use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_y, _block_dim_x, _thread_idx_x}};

#[inline(always)]
unsafe fn fma_up(a: f64, b: f64, c: f64) -> f64 {
    let out: f64;
    asm!("fma.rp.f64 {o}, {a}, {b}, {c};", o=out(reg64)out, a=in(reg64)a, b=in(reg64)b, c=in(reg64)c);
    out
}
#[inline(always)]
unsafe fn sqrt_up(x: f64) -> f32 {
    let root: f64; let out: f32;
    asm!("sqrt.rp.f64 {o}, {x};", o=out(reg64)root, x=in(reg64)x);
    asm!("cvt.rp.f32.f64 {o}, {x};", o=out(reg32)out, x=in(reg64)root);
    out
}
#[inline(always)]
unsafe fn add_up(a: f32, b: f32) -> f32 {
    let out: f32; asm!("add.rp.f32 {o}, {a}, {b};", o=out(reg32)out, a=in(reg32)a, b=in(reg32)b); out
}
#[inline(always)]
unsafe fn add_down(a: f32, b: f32) -> f32 {
    let out: f32; asm!("add.rm.f32 {o}, {a}, {b};", o=out(reg32)out, a=in(reg32)a, b=in(reg32)b); out
}
#[inline(always)]
unsafe fn mul_up(a: f32, b: f32) -> f32 {
    let out: f32; asm!("mul.rp.f32 {o}, {a}, {b};", o=out(reg32)out, a=in(reg32)a, b=in(reg32)b); out
}
#[inline(always)]
unsafe fn error_bound(norms: u64, limits: u64, token: usize, id: usize) -> f32 {
    let w = (norms as *const f32).add(id * 2);
    let x = (limits as *const f32).add(token * 3);
    bound_math::envelope(*w, *w.add(1), *x, *x.add(1),
        |a,b| mul_up(a,b), |a,b| add_up(a,b))
}
#[inline(always)]
unsafe fn eligible(scores: u64, norms: u64, limits: u64, n: usize, token: usize, id: usize) -> bool {
    let v = *(scores as *const f32).add(token * n + id);
    !v.is_finite() || add_up(v, error_bound(norms, limits, token, id)) >= *(limits as *const f32).add(token * 3 + 2)
}

/// Once at load, before LBF12 encoding. Norms use directed-up FP64 sums and roots.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_head_bound_weights(weight: u64, twin: u64, rows: u32, norms: u64) {
    let row = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let n = rows as usize; let k = 2560;
    if row >= n { return; }
    let mut e = 0.0f64; let mut w2 = 0.0f64; let mut q2 = 0.0f64;
    for col in 0..k {
        let w = f32::from_bits((*(weight as *const u16).add(crate::next_math::head_weight(row, col, k)) as u32) << 16);
        let raw = *(twin as *const u8).add(crate::kernel_params::flash_bf16_element(row, col, k, n));
        let s = *(twin as *const u8).add((n*k).div_ceil(256)*256 + (row/8*(k/32)+col/32)*8+row%8);
        let q = crate::fp8_bf16::decode(raw) * f32::from_bits((s as u32) << 23);
        let (qb, _) = crate::gemm::split2(q);
        let q = f32::from_bits((qb as u32) << 16);
        // A widely separated BF16 pair need not subtract exactly in FP64.
        // Form the nonnegative difference with directed-up rounding as well.
        let d = if w >= q { fma_up(1.0, w as f64, -(q as f64)) }
            else { fma_up(1.0, q as f64, -(w as f64)) };
        e = fma_up(d, d, e);
        w2 = fma_up(w as f64, w as f64, w2);
        q2 = fma_up(q as f64, q as f64, q2);
    }
    *(norms as *mut f32).add(row*2) = sqrt_up(e);
    *(norms as *mut f32).add(row*2+1) = add_up(sqrt_up(w2), sqrt_up(q2));
}

#[inline(always)]
unsafe fn norm_store(address: u32, value: f64) {
    asm!("st.shared.f64 [{p}], {v};", p=in(reg32)address, v=in(reg64)value);
}
#[inline(always)]
unsafe fn norm_load(address: u32) -> f64 {
    let value: f64;
    asm!("ld.shared.f64 {v}, [{p}];", v=out(reg64)value, p=in(reg32)address);
    value
}

/// One 128-thread CTA per target row. Every partial sum and merge rounds up,
/// preserving an upper bound regardless of this fixed reduction tree.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_head_bound_input(input: u64, rows: u32, limits: u64) {
    let row = _block_idx_x() as usize;
    let tid = _thread_idx_x() as usize;
    if row >= rows as usize { return; }
    let mut h2 = 0.0; let mut l2 = 0.0;
    let mut col = tid;
    while col < 2560 {
        let value = *(input as *const f32).add(row*2560+col);
        let (h, l) = crate::gemm::split2(value);
        let h = f32::from_bits((h as u32) << 16) as f64;
        let l = f32::from_bits((l as u32) << 16) as f64;
        h2 = fma_up(h, h, h2);
        l2 = fma_up(l, l, l2);
        col += 128;
    }
    let shared = crate::ptx::shared::<2048>();
    norm_store(shared+(tid*8) as u32, h2);
    norm_store(shared+((128+tid)*8) as u32, l2);
    crate::ptx::sync();
    let mut span = 64;
    while span > 0 {
        if tid < span {
            for plane in 0..2 {
                let address = shared+((plane*128+tid)*8) as u32;
                norm_store(address, fma_up(1.0, norm_load(address+(span*8) as u32), norm_load(address)));
            }
        }
        crate::ptx::sync();
        span /= 2;
    }
    if tid == 0 {
        // Triangle bound avoids a rounded signed hi+lo sum in the norm proof.
        let planes = add_up(sqrt_up(norm_load(shared)), sqrt_up(norm_load(shared+1024)));
        *(limits as *mut f32).add(row*3) = planes;
        *(limits as *mut f32).add(row*3+1) = planes;
    }
}

/// Grid (192, M), block128. A lower bound on the maximum target logit per row.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_head_bound_lower(scores: u64, norms: u64, limits: u64, rows: u32, partials: u64) {
    let tid = _thread_idx_x() as usize; let block = _block_idx_x() as usize;
    let token = _block_idx_y() as usize; let n = rows as usize;
    let mut best = f32::NEG_INFINITY; let mut id = block*128+tid;
    while id < n {
        let v = *(scores as *const f32).add(token*n+id);
        if v.is_finite() { best = best.max(add_down(v, -error_bound(norms, limits, token, id))); }
        id += 192*128;
    }
    let sh = crate::ptx::shared::<512>();
    crate::ptx::store(sh+(tid*4) as u32, best); crate::ptx::sync();
    if tid == 0 {
        let mut maximum = f32::NEG_INFINITY;
        for j in 0..128 { maximum = maximum.max(crate::ptx::load(sh+(j*4) as u32)); }
        *(partials as *mut f32).add(token*192+block) = maximum;
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_head_bound_lower_finish(partials: u64, rows: u32, limits: u64) {
    let row = _thread_idx_x() as usize;
    if row >= rows as usize { return; }
    let mut best = f32::NEG_INFINITY;
    for j in 0..192 { best = best.max(*(partials as *const f32).add(row*192+j)); }
    *(limits as *mut f32).add(row*3+2) = best;
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn hb_mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32, c0: f32, c1: f32, c2: f32, c3: f32) -> crate::canonical::Acc;
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_madd(a: [u32; 4], b: [u32; 2], c: crate::canonical::Acc) -> crate::canonical::Acc { hb_mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn hb_range(tiles: usize, part: usize) -> (usize, usize) { (tiles * part / 4, tiles * (part + 1) / 4) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn hb_better(a: (f32, u32), b: (f32, u32)) -> (f32, u32) { if b.0 > a.0 || b.0 == a.0 && b.1 < a.1 { b } else { a } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_shared() -> u32 { let p: u32; core::arch::asm!("{{ .shared .align 16 .b8 scratch[1024]; mov.u32 {p}, scratch; }}", p = out(reg32) p); p }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_sload(p: u32) -> f32 { let v: f32; core::arch::asm!("ld.shared.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p); v }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_store(p: u32, v: f32) { core::arch::asm!("st.shared.f32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v); }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_sync() { core::arch::asm!("bar.sync 0;"); }
#[cfg(target_arch = "nvptx64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct HbFragment([u32; 4]);
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_load(p: *const HbFragment) -> [u32; 4] {
    let (a,b,c,d): (u32,u32,u32,u32);
    core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a=out(reg32)a,b=out(reg32)b,
        c=out(reg32)c,d=out(reg32)d,p=in(reg64)p);
    [a,b,c,d]
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn hb_bits(x: f32) -> u32 { unsafe { core::mem::transmute(x) } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn hb_float(x: u32) -> f32 { unsafe { core::mem::transmute(x) } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX,v,d,31) }
/// LBF12: a packed 512-byte head slot (8 rows x K32) holds per lane 8 values x 12
/// bits (sign << 11 | exponent offset << 7 | mantissa; offset 15 = signed zero) in its first
/// 384 bytes (lane * 12); the 8 row base exponents sit in bases[slot][8], four slots per
/// 32-byte sector. The decode gives back the identical BF16 bits.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb12_decode(slot: *const u8, bases: *const u8, lane: usize) -> [u32; 4] {
    let w = slot.add(lane*12) as *const u32;
    let lo = *w as u64 | (*w.add(1) as u64) << 32;
    let hi = *w.add(2) as u64;
    let base = *bases.add(lane/4) as u32;
    let f = |j: u32| -> u32 { (if j < 5 { lo >> (12*j) } else if j == 5 { lo >> 60 | hi << 4 } else { hi >> (12*j-64) }) as u32 & 0xFFF };
    let v = |x: u32| -> u32 { let off = (x >> 7) & 15; (x >> 11) << 15 | if off == 15 { 0 } else { (base+off) << 7 | (x & 0x7F) } };
    [v(f(0)) | v(f(1)) << 16, v(f(2)) | v(f(3)) << 16, v(f(4)) | v(f(5)) << 16, v(f(6)) | v(f(7)) << 16]
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn hb_project<const GROUPS: usize>(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64, norms: u64, limits: u64) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp,lane) = (tid/32,tid%32);
    let (n,k,m) = (p.rows as usize,p.cols as usize,p.tokens as usize);
    let blocks = core::arch::nvptx::_grid_dim_x() as usize; let block = core::arch::nvptx::_block_idx_x() as usize;
    let span = (n/8 + blocks - 1)/blocks;
    let stop = if (block+1)*span < n/8 { (block+1)*span } else { n/8 };
    let mut best = [(f32::NEG_INFINITY,u32::MAX); GROUPS]; let mut valid = [1u32; GROUPS];
    let mut nt = block*span+warp;
    while nt < stop {
        let mut keep = 0u32;
        let t = lane / 4;
        if t < m {
            let id = nt * 8 + lane % 4 * 2;
            keep = u32::from(eligible(p.output, norms, limits, n, t, id)
                || eligible(p.output, norms, limits, n, t, id + 1));
        }
        for d in [16, 8, 4, 2, 1] { keep |= hb_shuffle(keep, d); }
        if keep == 0 {
            if t < m {
                let id = nt * 8 + lane % 4 * 2;
                *(p.output as *mut f32).add(t*n+id) = f32::NEG_INFINITY;
                *(p.output as *mut f32).add(t*n+id+1) = f32::NEG_INFINITY;
            }
            nt += 4;
            continue;
        }
        let mut left = [[0.0;2]; GROUPS]; let mut right = left;
        let packed = if flags == 0 { [0u32; 4] } else { hb_load((flags as *const HbFragment).add(nt)) };
        let mut part = 0;
        while part < 4 {
            let mut sums = [crate::canonical::ZERO; GROUPS];
            let (start,end) = hb_range(k/16,part);
            let mut pair = start/2;
            while pair < end/2 {
                let b = if (packed[pair/32] >> (pair%32)) & 1 != 0 {
                    hb12_decode((p.weight as *const u8).add((nt*(k/32)+pair)*512), (bases as *const u8).add((nt*(k/32)+pair)*8), lane) }
                    else { hb_load((p.weight as *const HbFragment).add((nt*(k/32)+pair)*32+lane)) };
                let mut g = 0;
                while g < GROUPS {
                    let a = p.input as *const HbFragment;
                    sums[g] = hb_madd((*a.add((g*(k/16)+pair*2)*32+lane)).0,[b[0],b[1]],sums[g]);
                    sums[g] = hb_madd((*a.add((g*(k/16)+pair*2+1)*32+lane)).0,[b[2],b[3]],sums[g]);
                    g += 1;
                }
                pair += 1;
            }
            let mut g = 0;
            while g < GROUPS {
                let z = [sums[g].a+sums[g].c,sums[g].b+sums[g].d];
                let mut j = 0;
                while j < 2 { match part { 0 => left[g][j]=z[j], 1 => left[g][j]+=z[j],
                    2 => right[g][j]=z[j], _ => right[g][j]+=z[j] } j += 1; }
                g += 1;
            }
            part += 1;
        }
        let mut g = 0;
        while g < GROUPS { let t=g*8+lane/4;
            if t<m { let mut j = 0; while j < 2 { let id=nt*8+lane%4*2+j; let value=left[g][j]+right[g][j];
                *(p.output as *mut f32).add(t*n+id)=value;
                valid[g] &= u32::from(hb_bits(value) & 0x7f80_0000 != 0x7f80_0000); best[g]=hb_better(best[g],(value,id as u32));
                j += 1; } }
            g += 1;
        }
        nt += 4;
    }
    let sh = hb_shared();
    let mut g = 0;
    while g < GROUPS {
        let mut d = 2;
        while d > 0 { best[g]=hb_better(best[g],(hb_float(hb_shuffle(hb_bits(best[g].0),d)),hb_shuffle(best[g].1,d)));
            valid[g] &= hb_shuffle(valid[g],d); d -= 1; }
        if lane%4==0 { let at=sh+((warp*GROUPS*8+g*8+lane/4)*12) as u32;
            hb_store(at,best[g].0); hb_store(at+4,best[g].1 as f32); hb_store(at+8,valid[g] as f32); }
        g += 1;
    }
    hb_sync();
    if tid<m { let (g,t)=(tid/8,tid%8); let mut best=(f32::NEG_INFINITY,u32::MAX); let mut valid=1;
        let mut w = 0;
        while w < 4 { let at=sh+((w*GROUPS*8+g*8+t)*12) as u32;
            best=hb_better(best,(hb_sload(at),hb_sload(at+4) as u32)); valid &= hb_sload(at+8) as u32; w += 1; }
        let out=(partial as *mut f32).add((tid*blocks+block)*3);
        *out=best.0; *out.add(1)=best.1 as f32; *out.add(2)=valid as f32;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_head_bound_exact(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64, norms: u64, limits: u64) {
    if p.tokens == 0 || p.tokens > 4 || p.cols != 2560 { return; }
    hb_project::<1>(p, partial, flags, bases, norms, limits);
}
