//! Checkpoint E4M3 conversion and grouped BF16 tensor-core operands.
use core::arch::asm;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
use crate::ptx;

#[inline(always)]
pub(crate) fn decode(value: u8) -> f32 {
    let sign = (value as u32 & 0x80) << 24;
    let exponent = (value as u32 >> 3) & 15;
    let fraction = value as u32 & 7;
    if exponent == 0 {
        f32::from_bits(((fraction as f32) * (1.0 / 512.0)).to_bits() | sign)
    } else if exponent == 15 && fraction == 7 {
        f32::from_bits(sign | 0x7fc00000)
    } else {
        f32::from_bits(sign | ((exponent + 120) << 23) | (fraction << 20))
    }
}

#[inline(always)]
fn bf16(value: f32) -> u16 {
    ((value.to_bits().wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1))) >> 16) as u16
}


#[repr(C, align(8))]
struct WeightFragment([u32; 2]);

/// Dequantize straight into the existing BF16 MMA B layout, without a row-major copy.
/// # Safety
/// N and K are multiples of 128. W is [E,N,K] E4M3, scales [E,N/128,K/128] BF16.
/// weight_fragments declares row-major bytes (0) or the resident MMA layout (1).
/// Output is E*N*K*2 bytes aligned to 16. Launch ceil(E*N*K/1024) blocks of 256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_dequant_bf16_mma_weights(
    params: crate::kernel_params::Fp8PrefillWeights,
) {
    let crate::kernel_params::Fp8PrefillWeights { weights: w, scales, output, rows: n, cols: k, experts, weight_fragments } = params;
    let i = _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize;
    let (n, k) = (n as usize, k as usize);
    let fragments = n * k / 4;
    if i >= experts as usize * fragments { return; }
    let expert = i / fragments;
    let local = i % fragments;
    let (lane, tile) = (local % 32, local / 32);
    let row = tile / (k / 16) * 8 + lane / 4;
    let col = tile % (k / 16) * 16 + lane % 4 * 2;
    let w = w.add(expert * n * k);
    let scales = scales.add(expert * (n / 128) * (k / 128));
    let scale = *scales.add((row / 128) * (k / 128) + col / 128) as u32;
    let shifted = scale + 0x3800;
    let raw = if weight_fragments != 0 { *w.cast::<u32>().add(local) }
        else { core::ptr::read_unaligned(w.add(row * k + col).cast::<u16>()) as u32
            | ((core::ptr::read_unaligned(w.add(row * k + col + 8).cast::<u16>()) as u32) << 16) };
    let pair = |raw| {
        if fast_scale(scale) {
            scaled_pair::<true>(raw, shifted | (shifted << 16), 0.0)
        } else {
            scaled_pair::<false>(raw, 0, f32::from_bits(scale << 16))
        }
    };
    *output.cast::<WeightFragment>().add(i) = WeightFragment([pair(raw as u16), pair((raw >> 16) as u16)]);
}


// The checkpoint arena guarantees scalar alignment, not vector alignment.
#[inline(always)]
pub(crate) unsafe fn load_weight4(src: *const u8) -> u32 {
    if src as usize & 3 == 0 { *src.cast::<u32>() }
    else { core::ptr::read_unaligned(src.cast::<u32>()) }
}

#[inline(always)]
pub(crate) unsafe fn copy_weight16(dst: u32, src: *const u8) {
    if src as usize & 15 == 0 { crate::ptx::copy16(dst, src.cast()); }
    else {
        let a = load_weight4(src);
        let b = load_weight4(src.add(4));
        let c = load_weight4(src.add(8));
        let d = load_weight4(src.add(12));
        asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};", p=in(reg32)dst,
            a=in(reg32)a, b=in(reg32)b, c=in(reg32)c, d=in(reg32)d);
    }
}

/// BF16 block scales whose biased exponent still fits after the 2^112 shift
/// take scaled_pair's packed fast path; the rest use the ordinary scale.
#[inline(always)]
pub(crate) fn fast_scale(bits: u32) -> bool { (0x0080..0x4780).contains(&bits) }

// Convert via exact FP16, reinterpret at 2^-112, and move that exponent into
// the block scale. This avoids per-element FP32 promotion and handles FP8 zeros
// and subnormals. Packed BF16 multiply rounds exactly once, as the BF16 policy.
#[inline(always)]
pub(crate) unsafe fn scaled_pair<const FAST: bool>(raw: u16, scale: u32, ordinary_scale: f32) -> u32 {
    if FAST {
        let half: u32;
        asm!("cvt.rn.f16x2.e4m3x2 {h}, {r};", h=out(reg32)half, r=in(reg16)raw);
        let bits = (half & 0x80008000) | ((half >> 3) & 0x0fff0fff);
        let result: u32;
        asm!("mul.rn.bf16x2 {d}, {a}, {b};", d=out(reg32)result, a=in(reg32)bits, b=in(reg32)scale);
        result
    } else {
        bf16(decode(raw as u8) * ordinary_scale) as u32
            | ((bf16(decode((raw >> 8) as u8) * ordinary_scale) as u32) << 16)
    }
}

#[derive(Clone, Copy)]
struct FusedAcc([f32; 4]);
#[inline(always)]
unsafe fn fused_mma(a: [u32; 4], b: [u32; 2], c: FusedAcc) -> FusedAcc {
    let (d0,d1,d2,d3): (f32,f32,f32,f32);
    asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
        {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, \
        {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
        d0=out(reg32)d0,d1=out(reg32)d1,d2=out(reg32)d2,d3=out(reg32)d3,
        a0=in(reg32)a[0],a1=in(reg32)a[1],a2=in(reg32)a[2],a3=in(reg32)a[3],
        b0=in(reg32)b[0],b1=in(reg32)b[1],
        c0=in(reg32)c.0[0],c1=in(reg32)c.0[1],c2=in(reg32)c.0[2],c3=in(reg32)c.0[3]);
    FusedAcc([d0,d1,d2,d3])
}
macro_rules! fused_unroll {
    ($count:expr, $i:ident, $body:block) => {
        if $count > 0 { let $i=0; $body }
        if $count > 1 { let $i=1; $body }
        if $count > 2 { let $i=2; $body }
        if $count > 3 { let $i=3; $body }
        if $count > 4 { let $i=4; $body }
        if $count > 5 { let $i=5; $body }
        if $count > 6 { let $i=6; $body }
        if $count > 7 { let $i=7; $body }
    }
}
macro_rules! row_unroll {
    ($small:expr, $i:ident, $body:block) => {
        { let $i=0; $body }
        if !$small { let $i=1; $body }
    }
}

#[inline(always)]
unsafe fn fused_stage<const SMALL: bool, const FRAGMENTS: bool>(sh: u32, w: *const u8, x: *const u32, ws: *const u16,
    mb: usize, nb: usize, kb: usize, m: usize, k: usize, tid: usize) {
    use crate::ptx;
    // Packed activation fragments are already contiguous and 16-byte aligned.
    row_unroll!(SMALL, step, {
        let q = tid + step * 256;
        let row = q / 64;
        let kt = q / 32 % 2;
        let lane = q % 32;
        let dst = sh + (q * 16) as u32;
        if mb + row * 16 < m {
            ptx::copy16(dst, x.add((((mb / 16 + row) * (k / 16) + kb / 16 + kt) * 32 + lane) * 4));
        } else {
            fused_unroll!(4, j, { ptx::store(dst + j * 4, 0.0); });
        }
    });
    let mut raw = [0u32; 4];
    let scale_bits = *ws.add((nb / 128) * (k / 128) + kb / 128) as u32;
    let ordinary = f32::from_bits(scale_bits << 16);
    let scaled = scale_bits + 0x3800;
    let packed_scale = scaled | (scaled << 16);
    fused_unroll!(4, step, {
        let q = tid + step * 256;
        let row = nb + q / 8;
        raw[step] = if FRAGMENTS {
            let col = kb + q % 8 / 4 * 16;
            *w.cast::<u32>().add((row / 8 * (k / 16) + col / 16) * 32 + row % 8 * 4 + q % 4)
        } else { load_weight4(w.add(row * k + kb + q % 8 * 4)) };
    });
    macro_rules! store_weights {
        ($fast:literal) => {
            fused_unroll!(4, step, {
                let q = tid + step * 256;
                let lo = scaled_pair::<$fast>(raw[step] as u16, packed_scale, ordinary);
                let hi = scaled_pair::<$fast>((raw[step] >> 16) as u16, packed_scale, ordinary);
                // Padding separates all eight MMA row groups in shared memory.
                let col = if FRAGMENTS { q % 8 / 4 * 16 + q % 4 * 2 } else { q % 8 * 4 };
                let dst = sh + if SMALL { 4096 } else { 8192 } + (q / 8 * 80 + col * 2) as u32;
                if FRAGMENTS {
                    asm!("st.shared.b32 [{p}], {a}; st.shared.b32 [{p}+16], {b};",
                        p=in(reg32)dst, a=in(reg32)lo, b=in(reg32)hi);
                } else {
                    asm!("st.shared.v2.b32 [{p}], {{{a},{b}}};",p=in(reg32)dst,a=in(reg32)lo,b=in(reg32)hi);
                }
            });
        };
    }
    if fast_scale(scale_bits) { store_weights!(true); }
    else { store_weights!(false); }
}


#[inline(always)]
unsafe fn fused_tile<const SMALL: bool, const FRAGMENTS: bool>(w: *const u8, ws: *const u16,
    x: *const u32, y: *mut f32, m: usize, n: usize, k: usize, mb: usize, nb: usize, round_output: bool, shared: u32) {
    use crate::ptx;
    let tid=_thread_idx_x() as usize;
    let (lane,warp)=(tid%32,tid/32);
    let (mr,nr)=(warp/2*if SMALL { 16 } else { 32 },warp%2*64);
    let stride=if SMALL {14336} else {18432};
    let mut sums=[[FusedAcc([0.0;4]);8];2];
    fused_stage::<SMALL, FRAGMENTS>(shared,w,x,ws,mb,nb,0,m,k,tid);
    ptx::wait(); ptx::sync();
    let mut kb=0;
    while kb<k {
        let current=shared+((kb/32%2)*stride) as u32;
        if kb+32<k {
            fused_stage::<SMALL, FRAGMENTS>(shared+(((kb/32+1)%2)*stride) as u32,w,x,ws,mb,nb,kb+32,m,k,tid);
        }
        fused_unroll!(2, z, {
            let col=z*16+lane%4*2;
            let aa = [ptx::load4(current+(((mr/16*2+z)*32+lane)*16) as u32),
                ptx::load4(current+((((mr/16+1)*2+z)*32+lane)*16) as u32)];
            fused_unroll!(4, group, {
                let mut bb=[[0;2];2];
                fused_unroll!(2,jj,{
                    let row=nr+(group*2+jj)*8+lane/4;
                    let p=current+if SMALL {4096} else {8192}+(row*80+col*2) as u32;
                    let (b0,b1):(u32,u32);
                    asm!("ld.shared.b32 {b0}, [{p}]; ld.shared.b32 {b1}, [{p}+16];",
                        b0=out(reg32)b0,b1=out(reg32)b1,p=in(reg32)p);
                    bb[jj]=[b0,b1];
                });
                row_unroll!(SMALL,i,{
                    let a=aa[i];
                    fused_unroll!(2,jj,{
                        sums[i][group*2+jj]=fused_mma(a,bb[jj],sums[i][group*2+jj]);
                    });
                });
            });
        });
        ptx::wait(); ptx::sync(); kb+=32;
    }
    row_unroll!(SMALL,i,{
        let row=mb+mr+i*16+lane/4;
        fused_unroll!(8,j,{
            let col=nb+nr+j*8+lane%4*2;
            let mut v=sums[i][j].0;
            if round_output {
                for value in &mut v { *value = f32::from_bits((bf16(*value) as u32) << 16); }
            }
            if row<m { *y.add(row*n+col)=v[0]; *y.add(row*n+col+1)=v[1]; }
            if row+8<m { *y.add((row+8)*n+col)=v[2]; *y.add((row+8)*n+col+1)=v[3]; }
        });
    });
}

#[inline(always)]
pub(crate) unsafe fn store_activation(y: *mut u32, width: usize, row: usize, col: usize,
    g0: f32, g1: f32, u0: f32, u1: f32) {
    let exp = |v: f32| { let e: f32; asm!("ex2.approx.ftz.f32 {e}, {v};",
        e=out(reg32)e,v=in(reg32)(-v*core::f32::consts::LOG2_E)); e };
    let a=bf16(g0/(1.0+exp(g0))*u0) as u32;
    let b=bf16(g1/(1.0+exp(g1))*u1) as u32;
    store_middle_pair(y,width,row,col,a|(b<<16));
}

#[inline(always)]
pub(crate) unsafe fn activation_pair(g0: f32,g1: f32,u0: f32,u1: f32) -> u32 {
    let exp = |v: f32| { let e: f32; asm!("ex2.approx.ftz.f32 {e}, {v};",
        e=out(reg32)e,v=in(reg32)(-v*core::f32::consts::LOG2_E)); e };
    let a=bf16(g0/(1.0+exp(g0))*u0) as u32;
    let b=bf16(g1/(1.0+exp(g1))*u1) as u32;
    a|(b<<16)
}

#[inline(always)]
pub(crate) unsafe fn store_middle_pair(y: *mut u32, width: usize, row: usize, col: usize, pair: u32) {
    let offset=((row/16*(width/16)+col/16)*32+row%8*4+col%8/2)*4+row%16/8+col%16/8*2;
    *y.add(offset)=pair;
}

/// Compact variable-M tasks for the FP32 down projection.
/// Counts and starts describe the expert rows, padded to 16 in storage.
/// Grid (20, task_capacity*2), block256. Each task splits into
/// two M64 CTAs for occupancy and to reduce work on uneven expert tails.
/// W8A8 uses block128, grid(40*planes,tasks,1), sharing K128 arithmetic.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_down(p: crate::fp8_abi::GroupedParams) {
    let shared = ptx::shared::<{ crate::fp8::GROUPED_DECODE_SHARED }>();
    if p.activation_quantization != 0 {
        crate::fp8::grouped_project::<false>(p, true, shared);
        return;
    }
    let (w, x, y) = (p.weights as *const u8, p.input as *const u32, p.output as *mut f32);
    let (counts, starts) = (p.route.counts as *const u32, p.route.starts as *const u32);
    let task_count = p.route.tasks as *const u32;
    let tasks = task_count.add(1);
    let n = p.cols;

    use core::arch::nvptx::_block_idx_y;
    let task=_block_idx_y() as usize/2;
    if task>=*task_count as usize { return; }
    let expert=*tasks.add(task*2) as usize;
    let mb=*tasks.add(task*2+1) as usize+(_block_idx_y() as usize%2)*64;
    let m=*counts.add(expert) as usize;
    if mb>=m { return; }
    let start=*starts.add(expert) as usize;
    let n = n as usize;
    let table = w.cast::<u64>().add(expert * 7);
    let k = *table.add(6) as usize;
    if k == 0 { return; }
    let w = *table.add(4) as *const u8;
    let ws = *table.add(5) as *const u16;
    let x=x.add(start*640/2);
    fused_tile::<true, false>(w,ws,x,y.add(start*n),m,n,k,mb,_block_idx_x() as usize*128,false,shared);
}
