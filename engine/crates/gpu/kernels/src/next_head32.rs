//! One vocabulary pass for M17..32, with plain's exact row arithmetic.
//! REDUCTION-PLAN: mma.m16n8k16 BF16 hi/lo, absolute K16 ranges
//! [Ktiles*p/4, Ktiles*(p+1)/4), p=0..3. Each range keeps its H/L chains;
//! ((p0+p1)+(p2+p3)) is fixed at every M. A weight fragment feeds up to four
//! groups of eight independent rows. Max/lowest-ID argmax is unchanged.
//! The lossless LBF12 decoder and private helpers duplicate the existing
//! head walk to leave its entry and other kernel modules untouched.
#[cfg(target_arch = "nvptx64")]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn wide_mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32, c0: f32, c1: f32, c2: f32, c3: f32) -> crate::canonical::Acc;
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_madd(a: [u32; 4], b: [u32; 2], c: crate::canonical::Acc) -> crate::canonical::Acc { wide_mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn wide_range(tiles: usize, part: usize) -> (usize, usize) { (tiles * part / 4, tiles * (part + 1) / 4) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn wide_better(a: (f32, u32), b: (f32, u32)) -> (f32, u32) { if b.0 > a.0 || b.0 == a.0 && b.1 < a.1 { b } else { a } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_shared() -> u32 { let p: u32; core::arch::asm!("{{ .shared .align 16 .b8 scratch[1536]; mov.u32 {p}, scratch; }}", p = out(reg32) p); p }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_sload(p: u32) -> f32 { let v: f32; core::arch::asm!("ld.shared.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p); v }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_store(p: u32, v: f32) { core::arch::asm!("st.shared.f32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v); }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_sync() { core::arch::asm!("bar.sync 0;"); }
#[cfg(target_arch = "nvptx64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct WideFragment([u32; 4]);
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_load(p: *const WideFragment) -> [u32; 4] {
    let (a,b,c,d): (u32,u32,u32,u32);
    core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a=out(reg32)a,b=out(reg32)b,
        c=out(reg32)c,d=out(reg32)d,p=in(reg64)p);
    [a,b,c,d]
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn wide_bits(x: f32) -> u32 { x.to_bits() }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn wide_float(x: u32) -> f32 { f32::from_bits(x) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide_shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX,v,d,31) }
/// LBF12: a packed 512-byte head slot (8 rows x K32) holds per lane 8 values x 12
/// bits (sign << 11 | exponent offset << 7 | mantissa; offset 15 = signed zero) in its first
/// 384 bytes (lane * 12); the 8 row base exponents sit in bases[slot][8], four slots per
/// 32-byte sector. The decode gives back the identical BF16 bits.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf12_decode(slot: *const u8, bases: *const u8, lane: usize) -> [u32; 4] {
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
unsafe fn wide_project<const GROUPS: usize, const ENCODED: bool>(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp,lane) = (tid/32,tid%32);
    let (n,k,m) = (p.rows as usize,p.cols as usize,p.tokens as usize);
    let blocks = core::arch::nvptx::_grid_dim_x() as usize; let block = core::arch::nvptx::_block_idx_x() as usize;
    let span = (n/8 + blocks - 1)/blocks;
    let stop = if (block+1)*span < n/8 { (block+1)*span } else { n/8 };
    let mut best = [(f32::NEG_INFINITY,u32::MAX); GROUPS]; let mut valid = [1u32; GROUPS];
    let mut nt = block*span+warp;
    while nt < stop {
        let mut left = [[0.0;2]; GROUPS]; let mut right = left;
        let packed = if ENCODED { wide_load((flags as *const WideFragment).add(nt)) } else { [0; 4] };
        let mut part = 0;
        while part < 4 {
            let mut sums = [crate::canonical::ZERO; GROUPS];
            let (start,end) = wide_range(k/16,part);
            let mut pair = start/2;
            while pair < end/2 {
                let b = if ENCODED && (packed[pair/32] >> (pair%32)) & 1 != 0 {
                    lbf12_decode((p.weight as *const u8).add((nt*(k/32)+pair)*512), (bases as *const u8).add((nt*(k/32)+pair)*8), lane) }
                    else { wide_load((p.weight as *const WideFragment).add((nt*(k/32)+pair)*32+lane)) };
                let mut g = 0;
                while g < GROUPS {
                    let a = p.input as *const WideFragment;
                    sums[g] = wide_madd((*a.add((g*(k/16)+pair*2)*32+lane)).0,[b[0],b[1]],sums[g]);
                    sums[g] = wide_madd((*a.add((g*(k/16)+pair*2+1)*32+lane)).0,[b[2],b[3]],sums[g]);
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
                valid[g] &= u32::from(wide_bits(value) & 0x7f80_0000 != 0x7f80_0000); best[g]=wide_better(best[g],(value,id as u32));
                j += 1; } }
            g += 1;
        }
        nt += 4;
    }
    let sh = wide_shared();
    let mut g = 0;
    while g < GROUPS {
        let mut d = 2;
        while d > 0 { best[g]=wide_better(best[g],(wide_float(wide_shuffle(wide_bits(best[g].0),d)),wide_shuffle(best[g].1,d)));
            valid[g] &= wide_shuffle(valid[g],d); d -= 1; }
        if lane%4==0 { let at=sh+((warp*GROUPS*8+g*8+lane/4)*12) as u32;
            wide_store(at,best[g].0); wide_store(at+4,best[g].1 as f32); wide_store(at+8,valid[g] as f32); }
        g += 1;
    }
    wide_sync();
    if tid<m { let (g,t)=(tid/8,tid%8); let mut best=(f32::NEG_INFINITY,u32::MAX); let mut valid=1;
        let mut w = 0;
        while w < 4 { let at=sh+((w*GROUPS*8+g*8+t)*12) as u32;
            best=wide_better(best,(wide_sload(at),wide_sload(at+4) as u32)); valid &= wide_sload(at+8) as u32; w += 1; }
        let out=(partial as *mut f32).add((tid*blocks+block)*3);
        *out=best.0; *out.add(1)=best.1 as f32; *out.add(2)=valid as f32;
    }
}

#[inline(always)]
unsafe fn dispatch<const ENCODED: bool>(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64) {
    if p.tokens == 0 || p.tokens > 32 || p.cols as usize > 4096 { return; }
    if p.tokens <= 8 { wide_project::<1, ENCODED>(p, partial, flags, bases); }
    else if p.tokens <= 16 { wide_project::<2, ENCODED>(p, partial, flags, bases); }
    else if p.tokens <= 24 { wide_project::<3, ENCODED>(p, partial, flags, bases); }
    else { wide_project::<4, ENCODED>(p, partial, flags, bases); }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head32(p: crate::kernel_params::Bf16Rows, partial: u64) {
    dispatch::<false>(p, partial, 0, 0);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head32_lbf12(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64) {
    dispatch::<true>(p, partial, flags, bases);
}
