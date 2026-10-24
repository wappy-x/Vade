//! Resident NVFP4, register dequantization and canonical BF16 tensor cores.
//! Four disjoint K ranges read every packed weight byte once per M16 tile.
use crate::{
    canonical::{self, Acc, ZERO, add, madd},
    fp4_abi::{Expert, Matrix, ProjectParams},
    fp8_abi::GroupedParams,
    ptx,
};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x},
};

#[repr(C, align(16))]
struct Input([u32; 4]);

#[inline(always)]
pub(crate) fn e2m1(raw: u32) -> f32 {
    let magnitude = raw & 7;
    // F32 magnitudes in units of bit 22: 0,252,254,255,256,257,258,259.
    let bits = (magnitude + 252 - u32::from(magnitude == 1))
        & 0u32.wrapping_sub(u32::from(magnitude != 0));
    f32::from_bits((bits << 22) | ((raw & 8) << 28))
}

#[inline(always)]
pub(crate) unsafe fn next_pair(raw: u32, scale: f32, global: f32) -> u32 {
    // E2M1*E4M3 is exact in F32. Keep the global multiply second, then round
    // once to BF16; folding the scales first can change a rounding boundary.
    crate::gemm::bf16_pair(
        (e2m1(raw) * scale) * global,
        (e2m1(raw >> 4) * scale) * global,
    )
}

#[inline(always)]
unsafe fn shuffle(value: u32, source: usize) -> u32 {
    let out: u32;
    asm!("shfl.sync.idx.b32 {out}, {value}, {source}, 31, 0xffffffff;",
        out=out(reg32)out, value=in(reg32)value, source=in(reg32)source as u32);
    out
}

// Literal indices let LLVM retain every independent accumulator in registers.
macro_rules! each_row {
    ($f:ident) => {
        $f!(0);
        $f!(1);
        $f!(2);
        $f!(3);
        $f!(4);
        $f!(5);
        $f!(6);
        $f!(7);
        $f!(8);
        $f!(9);
        $f!(10);
        $f!(11);
        $f!(12);
        $f!(13);
        $f!(14);
        $f!(15);
        $f!(16);
        $f!(17);
        $f!(18);
        $f!(19);
        $f!(20);
        $f!(21);
        $f!(22);
        $f!(23);
        $f!(24);
        $f!(25);
        $f!(26);
        $f!(27);
        $f!(28);
        $f!(29);
        $f!(30);
        $f!(31);
    };
}
#[derive(Clone, Copy, Default)]
struct Rows {
    sorted: u64,
    start: usize,
    first: usize,
    count: usize,
    middle: bool,
}
#[inline(always)]
unsafe fn fragment<const COMPACT: bool>(
    x: *const u32,
    k: usize,
    kt: usize,
    lane: usize,
    r: Rows,
) -> [u32; 4] {
    if !COMPACT {
        return (*x.add((kt * 32 + lane) * 4).cast::<Input>()).0;
    }
    let column = kt * 16 + lane % 4 * 2;
    let pair = |local: usize, col: usize| {
        if let Some(row) =
            crate::compact::input_row(r.sorted, r.start, r.first, local, r.count, r.middle)
        {
            if r.middle {
                *x.add(row * k / 2 + col / 2)
            } else {
                let x = (x as *const f32).add(row * k + col);
                crate::gemm::bf16_pair(*x, *x.add(1))
            }
        } else {
            0
        }
    };
    [
        pair(lane / 4, column),
        pair(lane / 4 + 8, column),
        pair(lane / 4, column + 8),
        pair(lane / 4 + 8, column + 8),
    ]
}
#[inline(always)]
unsafe fn activation<const NEXT: bool, const COMPACT: bool>(
    y: *mut u32,
    width: usize,
    row: usize,
    col: usize,
    g0: f32,
    g1: f32,
    u0: f32,
    u1: f32,
) {
    let pair = if NEXT {
        next_activation_pair(g0, g1, u0, u1)
    } else {
        crate::fp8_bf16::activation_pair(g0, g1, u0, u1)
    };
    if COMPACT {
        *y.add(row * width / 2 + col / 2) = pair;
    } else {
        crate::fp8_bf16::store_middle_pair(y, width, row, col, pair);
    }
}
// Same expf reconstruction and BF16 GU/activation boundaries as card-int FAST_MOE.
#[inline(always)]
unsafe fn next_activation_pair(g0: f32, g1: f32, u0: f32, u1: f32) -> u32 {
    let one = |g: f32, u: f32| {
        let g = f32::from_bits(crate::next_math::bf16(g) << 16);
        let u = f32::from_bits(crate::next_math::bf16(u) << 16);
        let (a, b) = exp_parts(-g);
        crate::next_math::bf16((g / core::f32::math::mul_add(a, b, 1.0)) * u)
    };
    one(g0, u0) | (one(g1, u1) << 16)
}
#[inline(always)]
pub(crate) unsafe fn exp_parts(x: f32) -> (f32, f32) {
    let (a, b): (f32, f32);
    asm!("{{ .reg .f32 t,n,r; .reg .b32 bits; fma.rn.f32 t, {x}, 0f3BBB989D, 0f3F000000; cvt.sat.f32.f32 t, t;
        fma.rm.f32 n, t, 0f437C0000, 0f4B400001; add.f32 r, n, 0fCB40007F; neg.f32 r, r;
        fma.rn.f32 r, {x}, 0f3FB8AA3B, r; fma.rn.f32 r, {x}, 0f32A57060, r;
        mov.b32 bits, n; shl.b32 bits, bits, 23; mov.b32 {b}, bits; ex2.approx.ftz.f32 {a}, r; }}",
        x=in(reg32)x,a=out(reg32)a,b=out(reg32)b);
    (a, b)
}
/// Eight output columns, sixteen token rows, four warps splitting canonical K.
/// Each lane loads a whole K16 block; four lanes cover a contiguous 32-byte
/// row segment. Shuffles redistribute its bytes into MMA B fragments, so no
/// dequantized weight tile or second resident weight layout is needed.
#[inline(always)]
unsafe fn tile<const NEXT: bool, const COMPACT: bool, const T: usize>(
    matrix: Matrix,
    x: *const u32,
    n: usize,
    k: usize,
    shared: u32,
    rows: Rows,
) -> [Acc; T] {
    let tid = _thread_idx_x() as usize % 128;
    let (lane, part) = (tid % 32, tid / 32);
    let row = n + if NEXT { part * 8 } else { 0 } + lane / 4;
    let (start, end) = if NEXT {
        (0, k / 16)
    } else {
        canonical::range(k / 16, part)
    };
    let global = *(matrix.global_scale as *const f32);
    let w = (matrix.weights as *const u64).add(row * (k / 16));
    let scales = (matrix.scales as *const u8).add(row * (k / 16));
    let mut sums = [ZERO; T];
    let mut kt = start;
    while kt < end {
        let column = kt + lane % 4;
        let (raw, scale) = if column < end {
            (*w.add(column), crate::fp8_bf16::decode(*scales.add(column)))
        } else {
            (0, 0.0)
        };
        macro_rules! step {
            ($z:literal) => {
                if kt + $z < end {
                    let source = lane / 4 * 4 + $z;
                    let lo = shuffle(raw as u32, source) >> (lane % 4 * 8);
                    let hi = shuffle((raw >> 32) as u32, source) >> (lane % 4 * 8);
                    let s = f32::from_bits(shuffle(scale.to_bits(), source));
                    let b = [next_pair(lo, s, global), next_pair(hi, s, global)];
                    macro_rules! update {
                        ($i:literal) => {
                            if $i < T {
                                let r = Rows {
                                    first: rows.first + $i * 16,
                                    ..rows
                                };
                                let a = fragment::<COMPACT>(x, k, kt + $z, lane, r);
                                sums[$i] = madd(a, b, sums[$i]);
                            }
                        };
                    }
                    each_row!(update);
                }
            };
        }
        step!(0);
        step!(1);
        step!(2);
        step!(3);
        kt += 4;
    }
    if NEXT {
        return sums;
    }
    let mut sum = sums[0];
    let at = shared + (tid * 16) as u32;
    ptx::store(at, sum.a);
    ptx::store(at + 4, sum.b);
    ptx::store(at + 8, sum.c);
    ptx::store(at + 12, sum.d);
    ptx::sync();
    if part == 0 {
        let load = |part: u32| {
            let at = shared + lane as u32 * 16 + part * 512;
            Acc {
                a: ptx::load(at),
                b: ptx::load(at + 4),
                c: ptx::load(at + 8),
                d: ptx::load(at + 12),
            }
        };
        sum = add(add(load(0), load(1)), add(load(2), load(3)));
    }
    sums[0] = sum;
    sums
}

#[inline(always)]
unsafe fn write(y: *mut f32, sum: Acc, row: usize, col: usize, m: usize, n: usize) {
    if row < m {
        *y.add(row * n + col) = sum.a;
        *y.add(row * n + col + 1) = sum.b;
    }
    if row + 8 < m {
        *y.add((row + 8) * n + col) = sum.c;
        *y.add((row + 8) * n + col + 1) = sum.d;
    }
}

/// Grid (N/8,ceil(M/16)), block128, static shared2048. Every matrix pointer
/// must be eight-byte aligned for weights, byte aligned for scales, F32 for
/// global_scale; packed input starts on sixteen bytes. All operands are finite.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_bf16_project(p: ProjectParams) {
    let (n, k, m) = (p.rows as usize, p.cols as usize, p.tokens as usize);
    let (nb, mb) = (_block_idx_x() as usize * 8, _block_idx_y() as usize * 16);
    let x = (p.input as *const u32).add(mb * k / 2);
    let sum =
        tile::<false, false, 1>(p.matrix, x, nb, k, ptx::shared::<2048>(), Rows::default())[0];
    let lane = _thread_idx_x() as usize;
    if lane < 32 {
        write(
            p.output as *mut f32,
            sum,
            mb + lane / 4,
            nb + lane % 4 * 2,
            m,
            n,
        );
    }
}

#[inline(always)]
unsafe fn task(p: GroupedParams, include_empty: bool) -> Option<(Expert, usize, usize, usize)> {
    let tasks = p.route.tasks as *const u32;
    let task = _block_idx_y() as usize;
    if task >= *tasks as usize {
        return None;
    }
    let expert = *tasks.add(1 + task * 2) as usize;
    let mb = *tasks.add(2 + task * 2) as usize + _block_idx_z() as usize * 16;
    let m = *(p.route.counts as *const u32).add(expert) as usize;
    let start = *(p.route.starts as *const u32).add(expert) as usize;
    let descriptor = *(p.weights as *const Expert).add(expert);
    if mb >= m || (!include_empty && descriptor.width == 0) {
        return None;
    }
    Some((descriptor, start, mb, m))
}

// Compact tiles: K is 2560 or 640, hence rows and every eight-K16 batch are
// aligned to 16 bytes; no weight prepack/copy. Gate: default-cache vector reads
// with an L2 lookahead. STREAM (down): read-only reads that skip L1 and fetch
// 256-byte L2 blocks, so each short row streams in whole blocks.
#[rustfmt::skip]
#[inline(always)]
unsafe fn bw_load<const STREAM: bool>(matrix: Matrix, row: usize, k: usize, kt: usize, lane: usize) -> ([u32;4], u32) {
    let column=kt+lane%4*2;
    let address=matrix.weights+(row*k/2+column*8) as u64;
    let (a,b,c,d): (u32,u32,u32,u32);
    if STREAM {
        asm!("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {{{a}, {b}, {c}, {d}}}, [{address}];",
            a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,address=in(reg64)address);
    } else {
        asm!("ld.global.v4.u32 {{{a}, {b}, {c}, {d}}}, [{address}];",
            a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,address=in(reg64)address);
    }
    ([a,b,c,d], *((matrix.scales as *const u16).add((row*(k/16)+column)/2)) as u32)
}

#[inline(always)]
unsafe fn shfl(value: u32, source: usize) -> u32 {
    let out: u32;
    asm!("shfl.sync.idx.b32 {out}, {value}, {source}, 31, 0xffffffff;",
        out=out(reg32)out, value=in(reg32)value, source=in(reg32)source as u32, options(nomem, nostack));
    out
}

#[inline(always)]
unsafe fn prmt(a: u32, b: u32, select: u32) -> u32 {
    let out: u32;
    asm!("prmt.b32 {out}, {a}, {b}, {s};", out=out(reg32)out, a=in(reg32)a, b=in(reg32)b,
        s=in(reg32)select, options(pure, nomem, nostack));
    out
}

/// The low two E4M3 bytes as exact F16 (low byte -> low half).
#[inline(always)]
unsafe fn scale_pair(raw: u32) -> u32 {
    let out: u32;
    asm!("{{ .reg .b16 r; cvt.u16.u32 r, {raw}; cvt.rn.f16x2.e4m3x2 {out}, r; }}",
        out=out(reg32)out, raw=in(reg32)raw, options(pure, nomem, nostack));
    out
}

/// Bytes 0 and 2 of `z` hold one row's packed E2M1 pair for two consecutive K16
/// blocks; `s` holds those blocks' scales as F16x2. Returns the BF16 pairs
/// (lower nibble in the lower half) for the first and second block, bit-equal
/// to next_pair(byte, scale, global) with `g` = global * 2^14: e * 2^-14 is an
/// exact F16 value, the mixed FMA forms e * s * 2^-14 exactly in F32 (adding
/// -0 keeps signed zeros) and global * 2^14 is exact, so the one F32 rounding
/// sees the same real product (e * s) * global as main.
#[inline(always)]
unsafe fn dequant(z: u32, s: u32, g: f32) -> (u32, u32) {
    let (first, second): (u32, u32);
    asm!("{{ .reg .b32 m, n, lo, hi; .reg .b16 h0, h1, h2, h3, s0, s1; .reg .f32 f0, f1, f2, f3;
        and.b32 m, {z}, 0x00070007; and.b32 n, {z}, 0x00080008; shl.b32 m, m, 9; shl.b32 n, n, 12; or.b32 lo, m, n;
        and.b32 m, {z}, 0x00700070; and.b32 n, {z}, 0x00800080; shl.b32 m, m, 5; shl.b32 n, n, 8; or.b32 hi, m, n;
        mov.b32 {{h0, h1}}, lo; mov.b32 {{h2, h3}}, hi; mov.b32 {{s0, s1}}, {s};
        fma.rn.f32.f16 f0, h0, s0, 0f80000000; fma.rn.f32.f16 f1, h1, s1, 0f80000000;
        fma.rn.f32.f16 f2, h2, s0, 0f80000000; fma.rn.f32.f16 f3, h3, s1, 0f80000000;
        mul.rn.f32 f0, f0, {g}; mul.rn.f32 f1, f1, {g}; mul.rn.f32 f2, f2, {g}; mul.rn.f32 f3, f3, {g};
        cvt.rn.bf16x2.f32 {first}, f2, f0; cvt.rn.bf16x2.f32 {second}, f3, f1; }}",
        z=in(reg32)z, s=in(reg32)s, g=in(reg32)g, first=out(reg32)first, second=out(reg32)second,
        options(pure, nomem, nostack));
    (first, second)
}

/// One A word: F32 token pair rounded to BF16x2 (gate) or a packed BF16 pair (down).
#[inline(always)]
unsafe fn a_word<const F32: bool>(address: u64) -> u32 {
    if F32 {
        let (a, b): (f32, f32);
        asm!("ld.global.nc.v2.f32 {{{a}, {b}}}, [{p}];", a=out(reg32)a, b=out(reg32)b,
            p=in(reg64)address, options(pure, readonly, nostack));
        crate::gemm::bf16_pair(a, b)
    } else {
        let v: u32;
        asm!("ld.global.nc.u32 {v}, [{p}];", v=out(reg32)v, p=in(reg64)address,
            options(pure, readonly, nostack));
        v
    }
}

// Same m16n8k16 fragments and ZERO -> K16(0)..K16(K/16-1) recurrence as
// tile::<true,true,1>, with bit-equal operands. Gate/up share A, retaining
// independent MMA chains; two adjacent K16 loads per lane per batch. Each lane
// resolves its two A rows once; live output rows keep main's bytes.
#[rustfmt::skip]
#[inline(always)]
unsafe fn bw_tile<const GATE: bool,const LEAN: bool>(matrix: Matrix, up: Matrix, x: *const u32,
    n: usize, k: usize, rows: Rows) -> [Acc;2] {
    // Tiles with at most eight live rows (every decode tile up to C8) skip rows 8..15.
    if rows.first+8<rows.count { bw_rows::<GATE,true,LEAN>(matrix,up,x,n,k,rows) } else { bw_rows::<GATE,false,LEAN>(matrix,up,x,n,k,rows) }
}
#[rustfmt::skip]
#[inline(always)]
unsafe fn bw_rows<const GATE: bool,const HI: bool,const LEAN: bool>(matrix: Matrix, up: Matrix, x: *const u32,
    n: usize, k: usize, rows: Rows) -> [Acc;2] {
    let lane=_thread_idx_x() as usize%32;
    let (quad,t)=(lane/4,lane%4);
    let row=n+quad;
    let global=*(matrix.global_scale as *const f32)*16384.0;
    let ug=if GATE { *(up.global_scale as *const f32)*16384.0 } else { 0.0 };
    let (mut sum,mut usum)=(ZERO,ZERO);
    let select=(t|(4+t)<<8) as u32;
    // A padded row reads the tile's first live row: MMA output row i depends
    // only on A row i, and padded output rows are never written.
    let a_row=|local: usize| -> u64 {
        let local=if rows.first+local<rows.count { local } else { 0 };
        let r=crate::compact::input_row(rows.sorted,rows.start,rows.first,local,rows.count,rows.middle).unwrap_or(0);
        // Gate reads F32 rows (K*4 bytes); down reads BF16 middle rows (K*2 bytes).
        x as u64+(r*k*if GATE { 4 } else { 2 }) as u64
    };
    let lo_row=a_row(quad);
    let hi_row=a_row(quad+8);
    let fragment=|kt: usize| -> [u32;4] {
        let at=(kt*16+t*2)*if GATE { 4 } else { 2 };
        let eight=if GATE { 32 } else { 16 };
        let lo0=a_word::<GATE>(lo_row+at as u64);
        let lo8=a_word::<GATE>(lo_row+(at+eight) as u64);
        if HI {
            [lo0,a_word::<GATE>(hi_row+at as u64),lo8,a_word::<GATE>(hi_row+(at+eight) as u64)]
        } else {
            [lo0,0,lo8,0]
        }
    };
    let load=|kt| {
        if kt>=k/16 { return (([0;4],0),([0;4],0)); }
        if GATE { (bw_load::<false>(matrix,row,k,kt,lane),bw_load::<false>(up,row,k,kt,lane)) }
        else { (bw_load::<true>(matrix,row,k,kt,lane),([0;4],0)) }
    };
    // Scale rows (k/16 bytes) are not covered by the lookahead: fetch their L2
    // lines up front, the quad's lanes splitting them (last line by last byte).
    let lines=|base: u64,bytes: usize| {
        let mut l=t;
        while l*128<bytes+128 {
            asm!("prefetch.global.L2 [{p}];",p=in(reg64)base+(l*128).min(bytes-1) as u64,options(nostack));
            l+=4;
        }
    };
    lines(matrix.scales+(row*(k/16)) as u64,k/16);
    if GATE { lines(up.scales+(row*(k/16)) as u64,k/16); }
    // A rows: each batch reads 8*16 columns; bring the next batch's lines into L1.
    let width=if GATE { 4 } else { 2 };
    let (mut a,mut b)=(load(0),load(8));
    // fp4_bw_down_pdl reads the gate's output only after this point (programmatic launch).
    if !GATE && LEAN { pdl_wait(); }
    for kt in (0..k/16).step_by(8) {
        let ((w,s),(u,v))=a;
        // LEAN (fp4_bw_gate_c1): the >8-row path skips the A-line prefetch, the kernel's
        // register peak (104 -> 86 = 10 CTAs/SM, so C1's 400 tiles fit one wave).
        if (!GATE || !HI || !LEAN) && kt+8<k/16 && t*128<128*width {
            let at=((kt+8)*16*width+t*128) as u64;
            asm!("prefetch.global.L1 [{p}];",p=in(reg64)lo_row+at,options(nostack));
            if HI { asm!("prefetch.global.L1 [{p}];",p=in(reg64)hi_row+at,options(nostack)); }
        }
        // Credit: llama.cpp GB10 L2 lookahead idea; no copied code. Gate only.
        let ahead=if LEAN { 2 } else { 3 };
        let next=kt+ahead*8+lane%4*2;
        if GATE && next<k/16 {
            let address=matrix.weights+(row*k/2+next*8) as u64;
            asm!("prefetch.global.L2 [{p}];",p=in(reg64)address);
            if GATE {
                let address=up.weights+(row*k/2+next*8) as u64;
                asm!("prefetch.global.L2 [{p}];",p=in(reg64)address);
            }
        }
        // Both matrices' scale pairs travel in one word.
        let raw=s|v<<16;
        // Each accumulator keeps its K16 order; gate and up chains are disjoint.
        macro_rules! pair { ($p:literal) => {{
            let source=quad*4+$p;
            let scales=shfl(raw,source);
            let (z0,z1)=(prmt(shfl(w[0],source),shfl(w[2],source),select),prmt(shfl(w[1],source),shfl(w[3],source),select));
            let scale=scale_pair(scales);
            let (g0,g1)=dequant(z0,scale,global);
            let (h0,h1)=dequant(z1,scale,global);
            let a0=fragment(kt+2*$p);
            sum=madd(a0,[g0,h0],sum);
            let a1=fragment(kt+2*$p+1);
            sum=madd(a1,[g1,h1],sum);
            if GATE {
                let (z0,z1)=(prmt(shfl(u[0],source),shfl(u[2],source),select),prmt(shfl(u[1],source),shfl(u[3],source),select));
                let scale=scale_pair(scales>>16);
                let (e0,e1)=dequant(z0,scale,ug);
                let (f0,f1)=dequant(z1,scale,ug);
                usum=madd(a0,[e0,f0],usum);
                usum=madd(a1,[e1,f1],usum);
            }
        }}; }
        pair!(0); pair!(1); pair!(2); pair!(3);
        a=b;
        b=load(kt+16);
    }
    [sum,usum]
}

/// Grid (40,task_capacity,planes), block64; no shared memory or split-K.
/// # Safety
/// Compact route tasks/counts and sorted rows must describe live finite K2560
/// input rows. Gate/up matrices are N640/K2560 with 16-byte-aligned weights;
/// packed middle storage covers all routed rows. Launch exactly two warps.
/// One-token launches only (hosts pick these entries by token count): programmatic
/// dependent launch lets fp4_bw_down_pdl's CTAs start while fp4_bw_gate_c1 runs;
/// down's route, slot and weight prologue never reads gate output, and its first
/// activation read waits for the gate grid's completion and memory flush. Both are
/// no-ops for a plain launch, so bytes and order are unchanged. At 8 rows the same
/// overlap cost +9.5 us per layer in production (the gate is DRAM-bound there), so
/// fp4_bw_gate / fp4_bw_down keep main's code.
#[inline(always)]
unsafe fn pdl_wait() { asm!("griddepcontrol.wait;", options(nostack)); }
#[rustfmt::skip]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_bw_gate(p: GroupedParams) { bw_gate::<false>(p); }
/// fp4_bw_gate's bytes with a one-wave budget: no >8-row A prefetch (86 registers),
/// L2 lookahead 2, and the PDL trigger for fp4_bw_down_pdl.
#[rustfmt::skip]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_bw_gate_c1(p: GroupedParams) {
    asm!("griddepcontrol.launch_dependents;", options(nostack));
    bw_gate::<true>(p);
}
#[rustfmt::skip]
#[inline(always)]
unsafe fn bw_gate<const LEAN: bool>(p: GroupedParams) {
    let Some((expert,start,mb,m))=task(p, false) else { return; };
    let tid=_thread_idx_x() as usize; let lane=tid%32;
    let nb=_block_idx_x() as usize*16+tid/32*8;
    if nb>=expert.width as usize { return; }
    let rows=Rows { sorted:p.route.sorted_row,start,first:mb,count:m,middle:false };
    let [g,u]=bw_tile::<true,LEAN>(expert.gate,expert.up,p.input as *const u32,nb,2560,rows);
    let y=(p.output as *mut u32).add(start*640/2); let col=nb+lane%4*2;
    let row=mb+lane/4;
    if row<m { activation::<true,true>(y,640,row,col,g.a,g.b,u.a,u.b); }
    if row+8<m { activation::<true,true>(y,640,row+8,col,g.c,g.d,u.c,u.d); }
}

/// Grid (160,task_capacity,planes), block64; original down BF16 boundary.
/// # Safety
/// Route tasks must describe live packed K640 middle rows and disjoint N2560
/// output rows. Down weights are 16-byte aligned with complete scales/global;
/// launch exactly two warps and fund every routed output row.
#[rustfmt::skip]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_bw_down(p: GroupedParams) { bw_down::<false>(p); }
/// fp4_bw_down's bytes, waiting (griddepcontrol) before its first activation read.
#[rustfmt::skip]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_bw_down_pdl(p: GroupedParams) { bw_down::<true>(p); }
#[rustfmt::skip]
#[inline(always)]
unsafe fn bw_down<const PDL: bool>(p: GroupedParams) {
    let Some((expert,start,mb,m))=task(p, true) else { return; };
    let tid=_thread_idx_x() as usize; let lane=tid%32;
    let nb=_block_idx_x() as usize*16+tid/32*8;
    let rows=Rows { sorted:0,start:0,first:mb,count:m,middle:true };
    let x=(p.input as *const u32).add(start*640/2);
    // Output slots resolve before the K loop so their loads overlap the stream.
    let row=mb+lane/4;
    let slots=(ranked_slot(p,start,row,m),ranked_slot(p,start,row+8,m));
    let s=if expert.width==0 { if PDL { pdl_wait(); } ZERO } else {
        bw_tile::<false,PDL>(expert.down,Matrix::default(),x,nb,640,rows)[0]
    };
    write_slots(p,s,slots,row,nb+lane%4*2,m);
}

#[inline(always)]
unsafe fn ranked_slot(p: GroupedParams,start: usize,r: usize,m: usize) -> usize {
    if r>=m { return 0; }
    let pair=*(p.route.sorted_row as *const u32).add(start+r) as usize;
    *(p.route.pair_to_row as *const u32).add(pair) as usize
}

// write_ranked with the two output slots resolved up front (same stores).
#[inline(always)]
unsafe fn write_slots(p: GroupedParams,sum: Acc,slots: (usize,usize),row: usize,col: usize,m: usize) {
    let one=|r: usize,slot: usize,a: f32,b: f32| {
        if r<m {
            *(p.output as *mut u32).add(slot*2560+col/2)=crate::next_math::bf16(a)|(crate::next_math::bf16(b)<<16);
        }
    };
    one(row,slots.0,sum.a,sum.b);
    one(row+8,slots.1,sum.c,sum.d);
}

// Routed rows retain their BF16 boundary in u16 storage. Each slot keeps its
// existing FP32-sized stride so the shared expert's FP32 row stays unchanged.
#[inline(always)]
unsafe fn write_ranked(p: GroupedParams,start: usize,sum: Acc,row: usize,col: usize,m: usize) {
    let one=|r: usize,a: f32,b: f32| {
        if r<m {
            let pair=*(p.route.sorted_row as *const u32).add(start+r) as usize;
            let slot=*(p.route.pair_to_row as *const u32).add(pair) as usize;
            *(p.output as *mut u32).add(slot*2560+col/2)=crate::next_math::bf16(a)|(crate::next_math::bf16(b)<<16);
        }
    };
    one(row,sum.a,sum.b);
    one(row+8,sum.c,sum.d);
}

/// Same task list/gather as FP8; paired gate/up and fused BF16 SwiGLU output.
/// Grid (80,task_capacity,min(8,ceil(tokens/16))), block256, shared4096.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_grouped_gate(p: GroupedParams) {
    grouped_gate::<false, false, 1>(p);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_next_grouped_gate(p: GroupedParams) {
    grouped_gate::<true, false, 1>(p);
}
#[inline(always)]
unsafe fn grouped_gate<const NEXT: bool, const COMPACT: bool, const T: usize>(p: GroupedParams) {
    let Some((expert, start, mb, m)) = task(p, false) else {
        return;
    };
    let width = expert.width as usize;
    let nb = _block_idx_x() as usize * if NEXT { 32 } else { 8 };
    if nb >= width {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let group = tid / 128;
    let shared = ptx::shared::<4096>();
    let x = (p.input as *const u32).add(if COMPACT { 0 } else { (start + mb) * 2560 / 2 });
    let rows = Rows {
        sorted: p.route.sorted_row,
        start,
        first: mb,
        count: m,
        middle: false,
    };
    let sums = tile::<NEXT, COMPACT, T>(
        if group == 0 { expert.gate } else { expert.up },
        x,
        nb,
        2560,
        shared + (group * 2048) as u32,
        rows,
    );
    macro_rules! finish {
        ($i:literal) => {
            if $i < T {
                let sum = sums[$i];
                if tid < if NEXT { 128 } else { 32 } {
                    let at = shared + (tid * 16) as u32;
                    ptx::store(at, sum.a);
                    ptx::store(at + 4, sum.b);
                    ptx::store(at + 8, sum.c);
                    ptx::store(at + 12, sum.d);
                }
                ptx::sync();
                if (128..if NEXT { 256 } else { 160 }).contains(&tid) {
                    let lane = (tid - 128) % 32;
                    let warp = (tid - 128) / 32;
                    let row = mb + $i * 16 + lane / 4;
                    let col = nb + if NEXT { warp * 8 } else { 0 } + lane % 4 * 2;
                    let at = shared + ((tid - 128) * 16) as u32;
                    let y = (p.output as *mut u32).add(start * 640 / 2);
                    if row < m {
                        activation::<NEXT, COMPACT>(
                            y,
                            width,
                            row,
                            col,
                            ptx::load(at),
                            ptx::load(at + 4),
                            sum.a,
                            sum.b,
                        );
                    } else if !COMPACT {
                        crate::fp8_bf16::store_middle_pair(y, width, row, col, 0);
                    }
                    if row + 8 < m {
                        activation::<NEXT, COMPACT>(
                            y,
                            width,
                            row + 8,
                            col,
                            ptx::load(at + 8),
                            ptx::load(at + 12),
                            sum.c,
                            sum.d,
                        );
                    } else if !COMPACT {
                        crate::fp8_bf16::store_middle_pair(y, width, row + 8, col, 0);
                    }
                }
                ptx::sync(); // Reuse gate scratch only after the up warps consume it.
            }
        };
    }
    each_row!(finish);
}

/// Grid (320,task_capacity,min(8,ceil(tokens/16))), block128, shared2048.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_grouped_down(p: GroupedParams) {
    grouped_down::<false, false, 1>(p);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_next_grouped_down(p: GroupedParams) {
    grouped_down::<true, false, 1>(p);
}
#[inline(always)]
unsafe fn grouped_down<const NEXT: bool, const COMPACT: bool, const T: usize>(p: GroupedParams) {
    let Some((expert, start, mb, m)) = task(p, false) else {
        return;
    };
    let k = expert.width as usize;
    let nb = _block_idx_x() as usize * if NEXT { 32 } else { 8 };
    let x = (p.input as *const u32).add(start * 640 / 2 + if COMPACT { 0 } else { mb * k / 2 });
    let rows = Rows {
        sorted: 0,
        start: 0,
        first: mb,
        count: m,
        middle: true,
    };
    let sums = tile::<NEXT, COMPACT, T>(expert.down, x, nb, k, ptx::shared::<2048>(), rows);
    let tid = _thread_idx_x() as usize;
    let lane = tid % 32;
    macro_rules! finish {
        ($i:literal) => {
            if $i < T {
                let sum = sums[$i];
                let value = if NEXT {
                    let r = |x| f32::from_bits(crate::next_math::bf16(x) << 16);
                    Acc {
                        a: r(sum.a),
                        b: r(sum.b),
                        c: r(sum.c),
                        d: r(sum.d),
                    }
                } else {
                    sum
                };
                if tid < if NEXT { 128 } else { 32 } {
                    write(
                        (p.output as *mut f32).add(start * 2560),
                        value,
                        mb + $i * 16 + lane / 4,
                        nb + if NEXT { tid / 32 * 8 } else { 0 } + lane % 4 * 2,
                        m,
                        2560,
                    );
                }
            }
        };
    }
    each_row!(finish);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_grouped_scatter(p: GroupedParams) {
    crate::fp8_routing::scatter::<true>(p);
}

#[path = "fp4_prefill_tc.rs"]
mod prefill_tc;
