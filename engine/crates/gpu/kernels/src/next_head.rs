//! One-pass first-party head projection for M1..16.
//! Only the resident permutation changes; all four K ranges and H/L chains stay fixed.
use core::arch::{asm, nvptx::{_block_idx_x, _block_dim_x, _grid_dim_x, _thread_idx_x}};
use crate::{canonical::{madd, ZERO}, kernel_params::{Bf16Rows, FlashBf16Pack, FlashBf16Input}, next_math, ptx};

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_layout(p: FlashBf16Pack) {
    let i = _block_idx_x() as usize*_block_dim_x() as usize+_thread_idx_x() as usize;
    let (n, k) = (p.rows as usize, p.cols as usize);
    if i < n*k {
        let (r, c) = (i/k, i%k);
        *(p.weight as *mut u16).add(next_math::head_weight(r,c,k)) = *(p.source as *const u16)
            .add(crate::kernel_params::flash_bf16_element(r,c,k,n));
    }
}
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Fragment([u32; 4]);
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_input(p: FlashBf16Input) {
    let i = _block_idx_x() as usize*_block_dim_x() as usize+_thread_idx_x() as usize;
    let (k, m) = (p.cols as usize, p.tokens as usize);
    if i >= m.div_ceil(8)*(k/16)*32 { return; }
    let (lane, tile, group) = (i%32, i/32%(k/16), i/(32*(k/16)));
    let token = group*8+lane/4;
    let mut a = [0; 4];
    if token < m {
        let c = tile*16+lane%4*2;
        let x = (p.input as *const f32).add(token*k+c);
        (a[0], a[1]) = crate::gemm::split2_pair(*x, *x.add(1));
        (a[2], a[3]) = crate::gemm::split2_pair(*x.add(8), *x.add(9));
    }
    *(p.output as *mut Fragment).add(i) = Fragment(a);
}
#[inline(always)]
unsafe fn load(p: *const Fragment) -> [u32; 4] {
    let (a,b,c,d): (u32,u32,u32,u32);
    asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a=out(reg32)a,b=out(reg32)b,
        c=out(reg32)c,d=out(reg32)d,p=in(reg64)p);
    [a,b,c,d]
}
#[inline(always)]
unsafe fn shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX,v,d,31) }
#[inline(always)]
unsafe fn project<const GROUPS: usize>(p: Bf16Rows, partial: u64) {
    let tid = _thread_idx_x() as usize;
    let (warp,lane) = (tid/32,tid%32);
    let (n,k,m) = (p.rows as usize,p.cols as usize,p.tokens as usize);
    let blocks = _grid_dim_x() as usize; let block = _block_idx_x() as usize;
    let span = (n/8).div_ceil(blocks);
    let mut best = [(f32::NEG_INFINITY,u32::MAX); GROUPS]; let mut valid = [1u32; GROUPS];
    for nt in (block*span+warp..((block+1)*span).min(n/8)).step_by(4) {
        let mut left = [[0.0;2]; GROUPS]; let mut right = left;
        for part in 0..4 {
            let mut sums = [ZERO; GROUPS];
            let (start,end) = crate::canonical::range(k/16,part);
            for pair in start/2..end/2 {
                let b = load((p.weight as *const Fragment).add((nt*(k/32)+pair)*32+lane));
                for g in 0..GROUPS {
                    let a = p.input as *const Fragment;
                    sums[g] = madd((*a.add((g*(k/16)+pair*2)*32+lane)).0,[b[0],b[1]],sums[g]);
                    sums[g] = madd((*a.add((g*(k/16)+pair*2+1)*32+lane)).0,[b[2],b[3]],sums[g]);
                }
            }
            for g in 0..GROUPS {
                let z = [sums[g].a+sums[g].c,sums[g].b+sums[g].d];
                for j in 0..2 { match part { 0 => left[g][j]=z[j], 1 => left[g][j]+=z[j],
                    2 => right[g][j]=z[j], _ => right[g][j]+=z[j] } }
            }
        }
        for g in 0..GROUPS { let t=g*8+lane/4;
            if t<m { for j in 0..2 { let id=nt*8+lane%4*2+j; let value=left[g][j]+right[g][j];
                *(p.output as *mut f32).add(t*n+id)=value;
                valid[g] &= u32::from(value.is_finite()); best[g]=next_math::better(best[g],(value,id as u32));
            } }
        }
    }
    let sh = ptx::shared::<1024>();
    for g in 0..GROUPS {
        for d in [2,1] { best[g]=next_math::better(best[g],(f32::from_bits(shuffle(best[g].0.to_bits(),d)),shuffle(best[g].1,d)));
            valid[g] &= shuffle(valid[g],d); }
        if lane%4==0 { let at=sh+((warp*GROUPS*8+g*8+lane/4)*12) as u32;
            ptx::store(at,best[g].0); ptx::store(at+4,best[g].1 as f32); ptx::store(at+8,valid[g] as f32); }
    }
    ptx::sync();
    if tid<m { let (g,t)=(tid/8,tid%8); let mut best=(f32::NEG_INFINITY,u32::MAX); let mut valid=1;
        for w in 0..4 { let at=sh+((w*GROUPS*8+g*8+t)*12) as u32;
            best=next_math::better(best,(ptx::load(at),ptx::load(at+4) as u32)); valid &= ptx::load(at+8) as u32; }
        let out=(partial as *mut f32).add((tid*blocks+block)*3);
        *out=best.0; *out.add(1)=best.1 as f32; *out.add(2)=valid as f32;
    }
}
/// Grid 4*SMs, block128. N multiple of eight, K=2560, M1..16.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head(p: Bf16Rows, partial: u64) {
    if p.tokens == 0 || p.tokens > 16 { return; }
    if p.tokens<=8 { project::<1>(p,partial); } else { project::<2>(p,partial); }
}

/// Optional draft-only projection: an MXFP8 image of the resident BF16 head in the
/// latency kernel's twin layout (E4M3 bytes in flash_bf16_element order, then at the next
/// 256-byte boundary UE8M0 scales as [row / 8][K / 32][8]). Per 32-column block the scale
/// puts the block's largest |w| at or under 448 (E4M3 max), clamped to the decoder's exact
/// range 10..=134; codes round to nearest with saturation. Fewer bits than the BF16 head:
/// only the drafter reads it (verification keeps the BF16 head, so output bytes do not
/// depend on it). One thread per (row, 32-column block); N % 8 == 0, K % 32 == 0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_mx8(p: FlashBf16Pack) {
    let i = _block_idx_x() as usize*_block_dim_x() as usize+_thread_idx_x() as usize;
    let (n, k) = (p.rows as usize, p.cols as usize);
    if i >= n*(k/32) { return; }
    let (r, c) = (i/(k/32), i%(k/32));
    let src = p.source as *const u16;
    let value = |j: usize| f32::from_bits((*src.add(next_math::head_weight(r, c*32+j, k)) as u32) << 16);
    let mut amax = 0f32;
    for j in 0..32 { amax = amax.max(value(j).abs()); }
    // Smallest s with amax * 2^(127-s) <= 448: 2^(s-127) >= amax / 448.
    let t = (amax / 448.0).to_bits();
    let mut s = ((t >> 23) & 0xff) as i32 + i32::from(t & 0x7f_ffff != 0);
    if amax == 0.0 { s = 127; }
    let s = s.clamp(10, 134) as u32;
    let scale = f32::from_bits((254 - s) << 23);
    let out = p.weight as *mut u8;
    for j in (0..32).step_by(2) {
        let (a, b) = (value(j) * scale, value(j + 1) * scale);
        let pair: u32;
        asm!("{{ .reg .b16 h; cvt.rn.satfinite.e4m3x2.f32 h, {b}, {a}; cvt.u32.u16 {d}, h; }}", d = out(reg32) pair, a = in(reg32) a.to_bits(), b = in(reg32) b.to_bits());
        let col = c*32 + j;
        *out.add(crate::kernel_params::flash_bf16_element(r, col, k, n)) = pair as u8;
        *out.add(crate::kernel_params::flash_bf16_element(r, col + 1, k, n)) = (pair >> 8) as u8;
    }
    *out.add((n*k).div_ceil(256)*256 + (r/8*(k/32) + c)*8 + r%8) = s as u8;
}


/// Drafter-only NVFP4 image of the resident BF16 head (VADE_FLASHNEXT_DRAFT_FP4=1, default
/// off): row-major E2M1 pairs (low nibble = even column), then at the next 256-byte boundary
/// one E4M3 scale per 16 columns ([row][K / 16] bytes); a weight reads back as
/// e2m1 * scale * 2^-6. The scale is the block's largest |w| / 6 in units of 2^-6, rounded
/// to nearest with saturation; codes round to nearest with saturation. Only the drafter
/// reads it (verification keeps the head), so output bytes do not depend on it.
/// One thread per (row, 16-column block); K % 16 == 0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_fp4_pack(p: FlashBf16Pack) {
    let i = _block_idx_x() as usize*_block_dim_x() as usize+_thread_idx_x() as usize;
    let (n, k) = (p.rows as usize, p.cols as usize);
    if i >= n*(k/16) { return; }
    let (r, c) = (i/(k/16), i%(k/16));
    let src = p.source as *const u16;
    let value = |j: usize| f32::from_bits((*src.add(next_math::head_weight(r, c*16+j, k)) as u32) << 16);
    let mut amax = 0f32;
    for j in 0..16 { amax = amax.max(value(j).abs()); }
    let s8: u32;
    asm!("{{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, {z}, {v}; cvt.u32.u16 {d}, t; }}",
        d = out(reg32) s8, z = in(reg32) 0f32.to_bits(), v = in(reg32) (amax * (64.0 / 6.0)).to_bits());
    let s8 = s8 & 0xff;
    let scale = crate::fp8_bf16::decode(s8 as u8) * (1.0 / 64.0);
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let out = (p.weight as *mut u8).add(r*(k/2) + c*8);
    for j in 0..8 {
        *out.add(j) = (e2m1_code(value(2*j) * inv) | e2m1_code(value(2*j + 1) * inv) << 4) as u8;
    }
    *(p.weight as *mut u8).add((n*k/2).div_ceil(256)*256 + r*(k/16) + c) = s8 as u8;
}

/// Nearest E2M1 code (0, 0.5, 1, 1.5, 2, 3, 4, 6; ties to the even code), saturating at 6,
/// sign in bit 3. Software rounding: the production PTX target (sm_121) has no e2m1 cvt.
#[inline(always)]
fn e2m1_code(v: f32) -> u32 {
    let m = v.abs();
    let code = u32::from(m > 0.25) + u32::from(m >= 0.75) + u32::from(m > 1.25) + u32::from(m >= 1.75)
        + u32::from(m > 2.5) + u32::from(m >= 3.5) + u32::from(m > 5.0);
    code | u32::from(v < 0.0 && code != 0) << 3
}
#[inline(always)]
unsafe fn ld_stream_u32(address: u64) -> u32 {
    let v: u32;
    asm!("ld.global.nc.L1::no_allocate.u32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) address, options(readonly, nostack));
    v
}
#[inline(always)]
unsafe fn ld_u8(address: u64) -> u32 {
    let v: u32;
    asm!("ld.global.nc.u8 {v}, [{p}];", v = out(reg32) v, p = in(reg64) address, options(readonly, nostack));
    v
}
#[inline(always)]
unsafe fn sum_xor(value: f32) -> f32 {
    let mut v = value;
    macro_rules! step { ($d:literal) => {{
        let other: u32;
        asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) other, v = in(reg32) v.to_bits(), d = in(reg32) $d as u32);
        v += f32::from_bits(other);
    }}; }
    step!(16); step!(8); step!(4); step!(2); step!(1);
    v
}
#[inline(always)]
fn dot8(word: u32, x: &[f32; 8]) -> f32 {
    let e = |i: u32| crate::fp4_experts::e2m1(word >> (4 * i));
    // Two chains of four (fixed order), then their sum.
    let lo = ((e(0) * x[0] + e(1) * x[1]) + e(2) * x[2]) + e(3) * x[3];
    let hi = ((e(4) * x[4] + e(5) * x[5]) + e(6) * x[6]) + e(7) * x[7];
    lo + hi
}
macro_rules! ten { ($m:ident) => { $m!(0); $m!(1); $m!(2); $m!(3); $m!(4); $m!(5); $m!(6); $m!(7); $m!(8); $m!(9); }; }

/// Drafter-only M1 projection through flash_next_head_fp4_pack's image (K = 2560):
/// out[r] = sum over 16-column blocks of (sum of e2m1 * x) * scale, times 2^-6, r < rows.
/// FP32 throughout; not the head's arithmetic (drafter proposals only). Grid any x 256
/// threads: warp g takes row pairs (2g, 2g + 1) + 2 * warps * i; lane l keeps x for its
/// eight-column words l + 32 j (j < 10) in registers and streams one word per row each.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_fp4(weights: u64, x: u64, out: u64, rows: u32) {
    const K: usize = 2560;
    const J: usize = K / 8 / 32;
    let n = rows as usize;
    let per = _block_dim_x() as usize / 32;
    let lane = _thread_idx_x() as usize % 32;
    let warps = _grid_dim_x() as usize * per;
    let warp = _block_idx_x() as usize * per + _thread_idx_x() as usize / 32;
    let xs = x as *const [f32; 8];
    let mut xv = [[0f32; 8]; J];
    macro_rules! load_x { ($j:literal) => { xv[$j] = *xs.add(lane + 32 * $j); }; }
    ten!(load_x);
    let scales = weights + ((n * K / 2).div_ceil(256) * 256) as u64;
    let y = out as *mut f32;
    let mut r = warp * 2;
    while r < n {
        let two = r + 1 < n;
        let r1 = if two { r + 1 } else { r };
        let (wr0, wr1) = (weights + (r * K / 2 + lane * 4) as u64, weights + (r1 * K / 2 + lane * 4) as u64);
        let (sr0, sr1) = (scales + (r * (K / 16) + lane / 2) as u64, scales + (r1 * (K / 16) + lane / 2) as u64);
        let (mut wa, mut wb, mut sa, mut sb) = ([0u32; J], [0u32; J], [0u32; J], [0u32; J]);
        macro_rules! load_w { ($j:literal) => {
            wa[$j] = ld_stream_u32(wr0 + 128 * $j);
            wb[$j] = ld_stream_u32(wr1 + 128 * $j);
            sa[$j] = ld_u8(sr0 + 16 * $j);
            sb[$j] = ld_u8(sr1 + 16 * $j);
        }; }
        ten!(load_w);
        let (mut a, mut b) = (0f32, 0f32);
        macro_rules! acc { ($j:literal) => {
            a += dot8(wa[$j], &xv[$j]) * crate::fp8_bf16::decode(sa[$j] as u8);
            b += dot8(wb[$j], &xv[$j]) * crate::fp8_bf16::decode(sb[$j] as u8);
        }; }
        ten!(acc);
        let (a, b) = (sum_xor(a), sum_xor(b));
        if lane == 0 {
            *y.add(r) = a * (1.0 / 64.0);
            if two { *y.add(r + 1) = b * (1.0 / 64.0); }
        }
        r += warps * 2;
    }
}

/// FP4 drafter screen (VADE_FLASHNEXT_DRAFT_FP4=2): the four 1024-row chunks whose FP4
/// maxima (logits_argmax partials: value, id, valid) are highest, ascending chunk order,
/// into sel[0..4] (u32). One warp; ties pick the lower chunk.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_fp4_select(partials: u64, chunks: u32, sel: u64) {
    let lane = _thread_idx_x() as usize;
    if lane >= 32 { return; }
    let p = partials as *const f32;
    let n = chunks as usize;
    let mut taken = 0u32;
    let mut picks = [u32::MAX; 4];
    for pick in 0..4 {
        let (mut best, mut id) = (f32::NEG_INFINITY, u32::MAX);
        let mut c = lane;
        let mut bit = 0;
        while c < n {
            let v = *p.add(c * 3);
            let ok = *p.add(c * 3 + 2) == 1.0 && v == v;
            if ok && taken & (1 << bit) == 0 && (v > best || id == u32::MAX) { best = v; id = c as u32; }
            c += 32; bit += 1;
        }
        // Warp max (value, then lower chunk).
        for d in [16u32, 8, 4, 2, 1] {
            let (ov, oi): (u32, u32);
            asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) ov, v = in(reg32) best.to_bits(), d = in(reg32) d);
            asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) oi, v = in(reg32) id, d = in(reg32) d);
            let ov = f32::from_bits(ov);
            if oi != u32::MAX && (id == u32::MAX || ov > best || ov == best && oi < id) { best = ov; id = oi; }
        }
        picks[pick] = id;
        if id != u32::MAX && id as usize % 32 == lane { taken |= 1 << (id as usize / 32); }
    }
    if lane == 0 {
        // Ascending chunk order (unused picks = u32::MAX sort last).
        for i in 0..4 { for j in 0..3 - i { if picks[j] > picks[j + 1] { picks.swap(j, j + 1); } } }
        for i in 0..4 { *(sel as *mut u32).add(i) = picks[i]; }
    }
}

/// FP4 drafter screen: rows of the selected chunks scored from the MXFP8 drafter twin's
/// values (flash_next_head_mx8 layout, rows % 64 == 0, K = 2560) in FP32, into
/// cand[slot * 1024 + j] (-inf past the vocabulary or for an unused slot). Grid 4 * 16 CTAs
/// x 256 threads: CTA (slot, 64-row block); thread t holds 16 columns of row
/// (t / 32) * 8 + t % 8 per 64-column tile, reduced over lanes t ^ 8, t ^ 16.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_fp4_rescore(twin: u64, rows: u32, x: u64, sel: u64, cand: u64) {
    const K: usize = 2560;
    let (slot, block) = (_block_idx_x() as usize / 16, _block_idx_x() as usize % 16);
    let t = _thread_idx_x() as usize;
    let n = rows as usize;
    let chunk = *(sel as *const u32).add(slot);
    let local = (t / 32) * 8 + t % 8;
    let out = (cand as *mut f32).add(slot * 1024 + block * 64 + local);
    let row0 = chunk as usize * 1024 + block * 64;
    if chunk == u32::MAX || row0 >= n {
        if t & 24 == 0 { *out = f32::NEG_INFINITY; }
        return;
    }
    let row = row0 + local;
    let scales = twin + ((n * K).div_ceil(256) * 256) as u64;
    let xs = x as *const f32;
    let ct = (t / 8) % 4;
    let mut acc = 0f32;
    let base = twin + (row0 / 64 * (K / 64) * 4096) as u64;
    for tile in 0..K / 64 {
        let at = base + (tile * 4096 + t * 16) as u64;
        let (w0, w1, w2, w3): (u32, u32, u32, u32);
        asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];", a = out(reg32) w0, b = out(reg32) w1, c = out(reg32) w2, d = out(reg32) w3, p = in(reg64) at, options(readonly, nostack));
        let col0 = tile * 64 + ct * 16;
        let s = ld_u8(scales + ((row / 8 * (K / 32) + col0 / 32) * 8 + row % 8) as u64);
        let scale = f32::from_bits(s << 23);
        let mut part = 0f32;
        // Byte b of the 16: q % 4 = b / 4, s = b % 4 -> column 2 (q % 4) + s % 2 + (s / 2) * 8.
        macro_rules! word { ($w:expr, $qi:literal) => {
            part += e4m3($w & 0xff) * *xs.add(col0 + 2 * $qi);
            part += e4m3(($w >> 8) & 0xff) * *xs.add(col0 + 2 * $qi + 1);
            part += e4m3(($w >> 16) & 0xff) * *xs.add(col0 + 2 * $qi + 8);
            part += e4m3(($w >> 24) & 0xff) * *xs.add(col0 + 2 * $qi + 9);
        }; }
        word!(w0, 0); word!(w1, 1); word!(w2, 2); word!(w3, 3);
        acc += part * scale;
    }
    // Lanes t, t ^ 8, t ^ 16, t ^ 24 hold the same row's four 16-column groups.
    macro_rules! xor { ($d:literal) => {{
        let o: u32;
        asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) o, v = in(reg32) acc.to_bits(), d = in(reg32) $d as u32);
        acc += f32::from_bits(o);
    }}; }
    xor!(8); xor!(16);
    if t & 24 == 0 {
        // UE8M0 s scales by 2^(s - 127) (the pack stored w * 2^(127 - s)), applied per tile.
        *out = if row < n { acc } else { f32::NEG_INFINITY };
    }
}
#[inline(always)]
fn e4m3(raw: u32) -> f32 { crate::fp8_bf16::decode(raw as u8) }

/// FP4 drafter screen: argmax over the 4096 candidates (first maximum = lowest vocabulary
/// id, slots ascend), written as main's argmax result row (value, id, valid). One CTA x 256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_fp4_final(cand: u64, sel: u64, result: u64) {
    let t = _thread_idx_x() as usize;
    let c = cand as *const f32;
    let (mut best, mut id) = (f32::NEG_INFINITY, u32::MAX);
    let mut i = t;
    while i < 4096 {
        let v = *c.add(i);
        if v > best { best = v; id = i as u32; }
        i += 256;
    }
    let shared = ptx::shared::<2048>();
    ptx::store(shared + (t * 8) as u32, best);
    ptx::store(shared + (t * 8 + 4) as u32, f32::from_bits(id));
    ptx::sync();
    if t == 0 {
        let (mut b, mut d) = (f32::NEG_INFINITY, u32::MAX);
        for j in 0..256 {
            let v = ptx::load(shared + (j * 8) as u32);
            let k = ptx::load(shared + (j * 8 + 4) as u32).to_bits();
            if k != u32::MAX && (d == u32::MAX || v > b || v == b && k < d) { b = v; d = k; }
        }
        let r = result as *mut f32;
        if d == u32::MAX {
            *r.add(2) = 0.0;
        } else {
            let chunk = *(sel as *const u32).add(d as usize / 1024);
            *r = b;
            *r.add(1) = (chunk as usize * 1024 + d as usize % 1024) as f32;
            *r.add(2) = 1.0;
        }
    }
}

/// Drafter confidence (VADE_MTP_CONF, drafter policy only; verification never reads it):
/// CTA c sums exp(l_i - m) over rows i = c*256 + t (+ grid stride), m = the argmax result's
/// value, into partials[c]. Grid <= 128 x 256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_draft_conf_partial(logits: u64, rows: u32, result: u64, partials: u64) {
    let t = _thread_idx_x() as usize;
    let c = _block_idx_x() as usize;
    let stride = _grid_dim_x() as usize * 256;
    let m = *(result as *const f32);
    let l = logits as *const f32;
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = c * 256 + t;
    let n = rows as usize;
    while i + 3 * stride < n {
        let (v0, v1, v2, v3) = (*l.add(i), *l.add(i + stride), *l.add(i + 2 * stride), *l.add(i + 3 * stride));
        a0 += ptx::exp(v0 - m); a1 += ptx::exp(v1 - m); a2 += ptx::exp(v2 - m); a3 += ptx::exp(v3 - m);
        i += 4 * stride;
    }
    while i < n {
        a0 += ptx::exp(*l.add(i) - m);
        i += stride;
    }
    let s = sum_xor((a0 + a1) + (a2 + a3));
    let shared = ptx::shared::<32>();
    if t % 32 == 0 { ptx::store(shared + (t / 32 * 4) as u32, s); }
    ptx::sync();
    if t == 0 {
        let mut total = 0.0f32;
        for j in 0..8 { total += ptx::load(shared + (j * 4) as u32); }
        *(partials as *mut f32).add(c) = total;
    }
}

/// out = (value, id, valid, p): the argmax result row plus p = 1 / sum(partials), the draft
/// token's softmax probability over the scored rows. One warp.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_draft_conf_final(partials: u64, count: u32, result: u64, out: u64) {
    let t = _thread_idx_x() as usize;
    let p = partials as *const f32;
    let mut s = 0.0f32;
    let mut i = t;
    while i < count as usize { s += *p.add(i); i += 32; }
    let s = sum_xor(s);
    if t == 0 {
        let r = result as *const f32;
        let o = out as *mut f32;
        *o = *r; *o.add(1) = *r.add(1); *o.add(2) = *r.add(2);
        *o.add(3) = if s > 0.0 { 1.0 / s } else { 0.0 };
    }
}

/// VADE_FLASHNEXT_DRAFT_MX8_LAYER=1 (default off, drafter only): an MXFP8 twin of an official
/// BF16 [N][K] MTP-layer matrix, which the loader stores packed (flash_bf16_element order, as
/// every flash_bf16_packed matrix), in the latency kernel's twin layout, quantized as
/// flash_next_head_mx8 does (per 32-column block: UE8M0 scale putting the block's largest |w|
/// at or under 448, clamped to 10..=134; E4M3 codes round to nearest with saturation). Only
/// draft proposals read it; verification never does. One thread per (row, 32-column block);
/// N % 8 == 0, K % 32 == 0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mx8_twin(p: FlashBf16Pack) {
    let i = _block_idx_x() as usize*_block_dim_x() as usize+_thread_idx_x() as usize;
    let (n, k) = (p.rows as usize, p.cols as usize);
    if i >= n*(k/32) { return; }
    let (r, c) = (i/(k/32), i%(k/32));
    let src = p.source as *const u16;
    let value = |j: usize| f32::from_bits((*src.add(crate::kernel_params::flash_bf16_element(r, c*32 + j, k, n)) as u32) << 16);
    let mut amax = 0f32;
    for j in 0..32 { amax = amax.max(value(j).abs()); }
    let t = (amax / 448.0).to_bits();
    let mut s = ((t >> 23) & 0xff) as i32 + i32::from(t & 0x7f_ffff != 0);
    if amax == 0.0 { s = 127; }
    let s = s.clamp(10, 134) as u32;
    let scale = f32::from_bits((254 - s) << 23);
    let out = p.weight as *mut u8;
    for j in (0..32).step_by(2) {
        let (a, b) = (value(j) * scale, value(j + 1) * scale);
        let pair: u32;
        asm!("{{ .reg .b16 h; cvt.rn.satfinite.e4m3x2.f32 h, {b}, {a}; cvt.u32.u16 {d}, h; }}", d = out(reg32) pair, a = in(reg32) a.to_bits(), b = in(reg32) b.to_bits());
        let col = c*32 + j;
        *out.add(crate::kernel_params::flash_bf16_element(r, col, k, n)) = pair as u8;
        *out.add(crate::kernel_params::flash_bf16_element(r, col + 1, k, n)) = (pair >> 8) as u8;
    }
    *out.add((n*k).div_ceil(256)*256 + (r/8*(k/32) + c)*8 + r%8) = s as u8;
}
