//! Packed GEMM: on-chip asynchronous staging; 8-row packed or blocked 64-row weights.
use super::*;
use crate::ptx;

#[inline(always)]
unsafe fn stage_decode<const FUSED: bool, const THIN: bool, const NATIVE: bool, const ROWS: usize>(sh: u32, w: *const u32, up: *const u32, x: *const u32,
    mb: usize, nb: usize, kt: usize, wt: usize, c: usize, tid: usize, t: usize) {
    if tid < 128 {
        unroll!(ROWS, r, {
            let dst = sh + (r*2048+tid*16) as u32;
            if ROWS == 1 || (mb+r)*16 < t {
                ptx::copy16_cached(dst, x.add(((mb+r)*kt+c+tid/32)*128+tid%32*4));
            } else { unroll!(4, j, { ptx::store(dst+j as u32*4, 0.0); }); }
        });
    }
    let base = (nb/8*(wt/4)+c/4)*2048+nb%8*256;
    if NATIVE {
        unroll!(4, i, {
            let q = tid+i*128;
            let tile = q/16;
            ptx::copy16(sh+(ROWS*2048) as u32+(q*16) as u32,w.add(((nb+tile/4)*wt+c+tile%4)*64+q%16*4));
        });
    } else {
        unroll!(2, i, {
            let q = tid+i*if THIN {128} else {256};
            ptx::copy16_stream(sh+(ROWS*2048) as u32+(q*16) as u32,w.add(base+q*4));
            if FUSED { ptx::copy16_stream(sh+(ROWS*2048) as u32+(if THIN {4096} else {8192})+(q*16) as u32,up.add(base+q*4)); }
        });
    }
}

/// A 16x64 decode tile: contiguous 64x64 weight reads, four fixed K ranges.
/// Block256/grid(ceil(T/16),N/64); N%64=0, K%256=0, aligned packed operands.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_decode(
    p: crate::projection_params::PackedDecodeProjection,
) {
    narrow_precision::<false, false, false, 1>(p.weights as *const u32, p.weights as *const u32,
        p.input as *const u32, p.output as *mut f32, p.rows, p.width, p.tokens, p.stride,
        _block_idx_x() as usize, _block_idx_y() as usize*8, false);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_decode_gate_up(
    p: crate::projection_params::DecodeGateUp,
) {
    narrow_precision::<true, true, false, 1>(p.gate as *const u32, p.up as *const u32,
        p.input as *const u32, p.output as *mut f32, p.rows, p.width, p.tokens, p.width,
        _block_idx_x() as usize, _block_idx_y() as usize*4, false);
}

/// Concatenated Q/K/V tiles reuse the canonical decode pipeline in one launch.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_decode_qkv(
    qw: *const u32, kw: *const u32, vw: *const u32, x: *const u32,
    q: *mut f32, key: *mut f32, value: *mut f32, nq: u32, nkv: u32, k: u32, t: u32,
) {
    let tile = _block_idx_y() as usize;
    let (qt, kt) = (nq as usize/64, nkv as usize/64);
    let (w, y, n, tile) = if tile < qt { (qw, q, nq, tile) }
        else if tile < qt+kt { (kw, key, nkv, tile-qt) }
        else { (vw, value, nkv, tile-qt-kt) };
    narrow_precision::<false, false, false, 1>(w, w, x, y, n, k, t, k, _block_idx_x() as usize, tile*8, false);
}

/// More output tiles distribute narrow O/down matrices evenly across the SMs.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_decode_n32(
    p: crate::projection_params::PackedDecodeProjection,
) {
    narrow_precision::<false, true, false, 1>(p.weights as *const u32, p.weights as *const u32,
        p.input as *const u32, p.output as *mut f32, p.rows, p.width, p.tokens, p.stride,
        _block_idx_x() as usize, _block_idx_y() as usize*4, false);
}

#[inline(always)]
pub(super) unsafe fn narrow_precision<const FUSED: bool, const THIN: bool, const NATIVE: bool, const ROWS: usize>(w: *const u32, up: *const u32, x: *const u32,
    y: *mut f32, n: u32, k: u32, t: u32, weight_stride: u32, mb: usize, nb: usize, bf16: bool) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid%32, tid/32);
    let (kt, wt) = (k as usize/16, weight_stride as usize/16);
    // The fused 16x32 tile fits three asynchronous stages in 30 KiB.
    // Three resident CTAs can overlap the gate/up weight reads on GB10.
    let sh = if ROWS == 2 { ptx::shared::<40960>() } else if ROWS > 1 { ptx::shared::<49152>() } else if FUSED && !THIN { ptx::shared::<36864>() } else if !FUSED && THIN {ptx::shared::<18432>()} else { ptx::shared::<30720>() };
    let stride = ROWS*2048 + if FUSED && !THIN { 16384 } else if !FUSED && THIN {4096} else {8192};
    let stages = if FUSED && !THIN {2} else {3};
    let (mut left, mut right, mut up_left, mut up_right) = ([ZERO; ROWS], [ZERO; ROWS], [ZERO; ROWS], [ZERO; ROWS]);
    let (mut sums, mut ups) = ([[ZERO;2]; ROWS], [[ZERO;2]; ROWS]);
    stage_decode::<FUSED, THIN, NATIVE, ROWS>(sh, w, up, x, mb, nb, kt, wt, 0, tid, t as usize);
    ptx::commit();
    if stages == 3 {
        stage_decode::<FUSED, THIN, NATIVE, ROWS>(sh+stride as u32, w, up, x, mb, nb, kt, wt, 4, tid, t as usize);
        ptx::commit(); ptx::wait_one();
    } else { ptx::wait(); }
    ptx::sync();
    let (mut c, mut buffer, mut part, mut boundary) = (0, 0, 0, kt/4);
    while c < kt {
        // Native verification keeps merge27's K-derived stage schedule.
        let current = sh + (if NATIVE { c/4%3*10240 } else { buffer*stride }) as u32;
        let ahead = c+4*(stages-1);
        if ahead < kt {
            let next = sh + (if NATIVE { (c/4+2)%3*10240 } else { (buffer+stages-1)%stages*stride }) as u32;
            stage_decode::<FUSED, THIN, NATIVE, ROWS>(next, w, up, x, mb, nb, kt, wt, ahead, tid, t as usize);
            ptx::commit();
        }
        unroll!(4, z, {
            unroll!(ROWS, r, {
                let a = ptx::load4(current+(r*2048+(z*32+lane)*16) as u32);
                unroll!(2, j, { if NATIVE || j == 0 {
                    let at = current+(ROWS*2048) as u32+(((warp*if NATIVE {2} else {1}+j)*4+z)*256+lane*8) as u32;
                    let b = ptx::load2(at);
                    sums[r][j] = madd(a,b,sums[r][j]);
                    if FUSED { let u = ptx::load2(at+(if THIN {4096} else {8192})); ups[r][j] = madd(a,u,ups[r][j]); }
                } });
            });
        });
        // Pipeline across range boundaries; only the accumulator resets.
        // The exact four ranges and balanced merge are unchanged.
        c += 4;
        if !NATIVE && c == boundary {
            unroll!(ROWS, r, {
                match part {
                    0 => { left[r] = sums[r][0]; up_left[r] = ups[r][0]; }
                    1 => { left[r] = add(left[r], sums[r][0]); up_left[r] = add(up_left[r], ups[r][0]); }
                    2 => { right[r] = sums[r][0]; up_right[r] = ups[r][0]; }
                    _ => { right[r] = add(right[r], sums[r][0]); up_right[r] = add(up_right[r], ups[r][0]); }
                }
            });
            sums = [[ZERO;2]; ROWS]; ups = [[ZERO;2]; ROWS];
            part += 1;
            boundary += kt/4;
        }
        if stages == 3 && ahead < kt { ptx::wait_one(); } else { ptx::wait(); }
        ptx::sync();
        buffer = (buffer+1)%stages;
    }
    unroll!(ROWS, r, {
        if !NATIVE { sums[r][0] = add(left[r],right[r]); ups[r][0] = add(up_left[r],up_right[r]); }
        if FUSED {
            let row = (mb+r)*16+lane/4;
            let col = (nb+warp)*8+lane%4*2;
            let (g,u) = (sums[r][0], ups[r][0]);
            let pair = |a: f32, b: f32, u: f32, v: f32, row: usize| {
                if row >= t as usize { 0 } else {
                    if bf16 {
                        let rounded = |x| f32::from_bits((round_bf16(x) as u32) << 16);
                        let (a, b, u, v) = (rounded(a), rounded(b), rounded(u), rounded(v));
                        bf16_pair(rounded(a/(1.0+ptx::exp(-a)))*u, rounded(b/(1.0+ptx::exp(-b)))*v)
                    } else { bf16_pair(a/(1.0+ptx::exp(-a))*u, b/(1.0+ptx::exp(-b))*v) }
                }
            };
            let put = |value: u32, row: usize| {
                let tile = row/16*(n as usize/16)+col/16;
                let slot = (row%8*4+col%8/2)*4+row%16/8+col%16/8*2;
                if ROWS == 1 || row/16*16 < t as usize { *y.cast::<u32>().add(tile*128+slot) = value; }
            };
            put(pair(g.a,g.b,u.a,u.b,row), row);
            put(pair(g.c,g.d,u.c,u.d,row+8), row+8);
        } else {
            unroll!(2, j, { if NATIVE || j == 0 {
                store_wide::<false>(y,sums[r][j],(mb+r)*16+lane/4,(nb+warp*if NATIVE {2} else {1}+j)*8+lane%4*2,t as usize,n as usize);
            } });
        }
    });
}

#[inline(always)]
fn weight_tile<const TILED: bool>(row: usize, col: usize, kt: usize) -> usize {
    if TILED { (row/8*(kt/4)+col/4)*32+row%8*4+col%4 } else { row*kt+col }
}
#[inline(always)]
unsafe fn store_wide<const ROUND: bool>(y: *mut f32, mut v: Acc, row: usize, col: usize, t: usize, n: usize) {
    if ROUND {
        (v.a, v.b) = rounded_pair(v.a, v.b);
        (v.c, v.d) = rounded_pair(v.c, v.d);
    }
    if row < t {
        *y.add(row * n + col).cast::<u64>() = v.a.to_bits() as u64 | ((v.b.to_bits() as u64) << 32);
    }
    if row + 8 < t {
        *y.add((row + 8) * n + col).cast::<u64>() =
            v.c.to_bits() as u64 | ((v.d.to_bits() as u64) << 32);
    }
}

#[inline(always)]
pub(super) unsafe fn store_residual_precision<const ROUND: bool, const TILED: bool>(sh: u32, y: *mut f32, v: Acc, row: usize, col: usize,
    local_row: usize, local_col: usize, t: usize, n: usize) {
    if ROUND && !TILED || row < t {
        let p = y.add(row * n + col);
        let (va,vb) = if ROUND { rounded_pair(v.a,v.b) } else { (v.a,v.b) };
        let (a,b) = (*p+va,*p.add(1)+vb);
        let (a,b) = if ROUND { rounded_pair(a,b) } else { (a,b) };
        *p.cast::<u64>() = a.to_bits() as u64 | ((b.to_bits() as u64) << 32);
        ptx::store(sh + ((local_row * 128 + local_col) * 4) as u32, a);
        ptx::store(sh + ((local_row * 128 + local_col + 1) * 4) as u32, b);
    }
    if ROUND && !TILED || row + 8 < t {
        let p = y.add((row + 8) * n + col);
        let (va,vb) = if ROUND { rounded_pair(v.c,v.d) } else { (v.c,v.d) };
        let (a,b) = (*p+va,*p.add(1)+vb);
        let (a,b) = if ROUND { rounded_pair(a,b) } else { (a,b) };
        *p.cast::<u64>() = a.to_bits() as u64 | ((b.to_bits() as u64) << 32);
        ptx::store(sh + (((local_row + 8) * 128 + local_col) * 4) as u32, a);
        ptx::store(sh + (((local_row + 8) * 128 + local_col + 1) * 4) as u32, b);
    }
}

#[inline(always)]
unsafe fn stage_rows<const FUSED: bool, const ROWS: usize, const TILED: bool, const WEIGHTS: bool>(
    sh: u32,
    w: *const u32,
    up: *const u32,
    x: *const u32,
    mb: usize,
    nb: usize,
    kt: usize,
    weight_tiles: usize,
    c: usize,
    tid: usize,
    t: usize,
) {
    macro_rules! copy_x {
        ($q:expr) => {{
            let q = $q;
            let a = q / 64;
            let z = q % 64;
            let ck = z / 32;
            let l = z % 32;
            let dst = sh + (q * 16) as u32;
            if (mb + a) * 16 < t {
                ptx::copy16(dst, x.add(((mb + a) * kt + c + ck) * 128 + l * 4));
            } else {
                for j in 0..4 {
                    ptx::store(dst + j * 4, 0.0);
                }
            }
        }};
    }
    // All wide projections use the gate/up kernel's 256-thread copy schedule.
    unroll!(ROWS / 64, step, { copy_x!(tid + step * 256); });
    if WEIGHTS { stage_weights::<FUSED, ROWS, TILED>(sh, w, up, nb, weight_tiles, c, tid); }
}

#[inline(always)]
unsafe fn stage_weights<const FUSED: bool, const ROWS: usize, const TILED: bool>(
    sh: u32, w: *const u32, up: *const u32, nb: usize, weight_tiles: usize, c: usize, tid: usize,
) {
    macro_rules! copy_w {
        ($q:expr) => {{
            let q = $q;
            let b = q / 32;
            let z = q % 32;
            let ck = z / 16;
            let l = z % 16;
            ptx::copy16(
                sh + (ROWS * 64 + q * 16) as u32,
                (if FUSED && b & 4 != 0 { up } else { w }).add(
                    weight_tile::<TILED>(if FUSED {
                        nb + b / 8 * 4 + b % 4
                    } else {
                        nb + b
                    }, c + ck, weight_tiles)
                        * 64
                        + l * 4,
                ),
            );
        }};
    }
    unroll!(2, step, { copy_w!(tid + step * 256); });
}
#[inline(always)]
unsafe fn wide<
    const FUSED: bool,
    const QKV: bool,
    const GROUP: usize,
    const ROWS: usize,
    const TILED: bool,
    const RESIDUAL: bool,
    const ROUND: bool,
>(
    weights: [*const u32; 3],
    up: *const u32,
    x: *const u32,
    outputs: [*mut f32; 3],
    widths: [u32; 3],
    k: u32,
    t: u32,
    weight_stride: u32,
    partials: *mut f32,
    gates: Option<Gates>,
) {
    let n = if QKV {
        widths[0] + widths[1] + widths[2]
    } else {
        widths[0]
    };
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let tile_m = ROWS;
    let tile_n = 128;
    let xbytes = ROWS * 64;
    let mt = (t as usize + tile_m - 1) / tile_m;
    let nt = n as usize / (if FUSED { 64 } else { tile_n });
    let group = if k <= 8192 { 8 } else { 4 };
    // Bound the active packed-X working set; large batches otherwise thrash L2.
    if t as usize * k as usize * 2 > 16 * 1024 * 1024 {
        let linear = ni * mt + mi;
        let first = linear / (nt * group) * group;
        let rows = group.min(mt - first);
        let within = linear - first * nt;
        // The last group may contain fewer row tiles on a rank-local span.
        mi = first + within % rows;
        ni = within / rows;
    }
    // A Q/K/V tile stays within one projection: select its base once per CTA.
    let mut part = 0;
    if QKV {
        let qtiles = widths[0] as usize / 128;
        let kvtiles = widths[1] as usize / 128;
        if ni >= qtiles + kvtiles {
            ni -= qtiles + kvtiles;
            part = 2;
        } else if ni >= qtiles {
            ni -= qtiles;
            part = 1;
        }
    }
    if part == 2 {
        if let Some(gates) = gates {
            gate_epilogue(weights[2], up, x, outputs[2], gates, mi, k as usize, t as usize, tid);
            return;
        }
    }
    let (w, y, n) = if !QKV || part == 0 {
        (weights[0], outputs[0], widths[0])
    } else if part == 1 {
        (weights[1], outputs[1], widths[1])
    } else {
        (weights[2], outputs[2], widths[2])
    };
    let (mb, nb) = (mi * (tile_m / 16), ni * if FUSED { 8 } else { tile_n / 8 });
    let warp_columns = 2;
    let (mw, nw) = (warp / warp_columns * (ROWS / 64), warp % warp_columns * 8);
    let kt = k as usize / 16;
    let weight_tiles = weight_stride as usize / 16;
    let sh = if ROWS > 128 { 0 } else { ptx::shared::<49152>() };
    let stride = xbytes + tile_n * 64;
    let stages = 3;
    // GROUP B fragments stay live per MMA group (eight for paired gate/up).
    macro_rules! each_row {
        ($i:ident, $body:block) => {
            unroll!(4, $i, { if $i < ROWS / 64 { $body } });
        };
    }
    let mut sums = [[ZERO; 8]; 4];
    // Native packed8 keeps the incoming 128-row pointer-advance schedule.
    // Other layouts retain their existing copy dispatch and accumulator order.
    let native128 = ROUND && !TILED && ROWS == 128;
    let source_x = x.wrapping_add((mb + tid / 64) * kt * 128 + tid % 64 * 4);
    let source_w = (if FUSED && tid / 32 & 4 != 0 { up } else { w })
        .add((nb + if FUSED { tid / 32 % 4 } else { tid / 32 }) * weight_tiles * 64 + tid % 32 * 4);
    let valid = [(mb + tid / 64) * 16 < t as usize, (mb + tid / 64 + 4) * 16 < t as usize];
    let copy_x = |sh: u32, source: *const u32| {
        unroll!(2, i, {
            let dst = sh + (tid * 16 + i * 4096) as u32;
            if valid[i] { ptx::copy16(dst, source.add(i * kt * 512)); }
            else { unroll!(4, j, { ptx::store(dst + (j * 4) as u32, 0.0); }); }
        });
    };
    let copy_w = |sh: u32, source: *const u32| {
        ptx::copy16(sh + (8192 + tid * 16) as u32, source);
        ptx::copy16(sh + (12288 + tid * 16) as u32, source.add(weight_tiles * if FUSED { 256 } else { 512 }));
    };
    if native128 { copy_x(sh, source_x); copy_w(sh, source_w); }
    else { stage_rows::<FUSED, ROWS, TILED, true>(sh, w, up, x, mb, nb, kt, weight_tiles, 0, tid, t as usize); }
    if stages == 3 {
        ptx::commit();
        if native128 {
            copy_x(sh + stride as u32, source_x.wrapping_add(256));
            copy_w(sh + stride as u32, source_w.add(128));
        } else {
            stage_rows::<FUSED, ROWS, TILED, true>(sh + stride as u32,
                w, up, x, mb, nb, kt, weight_tiles, 2, tid, t as usize);
        }
        ptx::commit();
        ptx::wait_one();
    } else {
        ptx::wait();
    }
    ptx::sync();
    let mut c = 0;
    let mut current = 0usize;
    let mut next = (stages - 1) * stride;
    let mut ahead_x = source_x.wrapping_add(512);
    let mut ahead_w = source_w.add(256);
    while c < kt {
        let ahead = c + 2 * (stages - 1);
        if ahead < kt {
            if native128 { copy_x(sh + next as u32, ahead_x); }
            else { stage_rows::<FUSED, ROWS, TILED, false>(sh + next as u32,
                w, up, x, mb, nb, kt, weight_tiles, ahead, tid, t as usize); }
            if ROWS > 128 {
                stage_weights::<FUSED, ROWS, TILED>(sh + next as u32, w, up, nb, weight_tiles, ahead, tid);
                ptx::commit();
            }
        }
        unroll!(2, z, {
            unroll!(8 / GROUP, group, {
                let mut bb = [[0u32; 2]; 8];
                unroll!(GROUP, jj, {
                    let j = group * GROUP + jj;
                    bb[j] = ptx::load2(
                        sh + (current + xbytes + ((nw + j) * 2 + z) * 256 + lane * 8) as u32,
                    );
                });
                // Padded input rows are zero; bound stores, not every MMA.
                each_row!(i, {
                    let p = sh + (current + ((mw + i) * 2 + z) * 512 + lane * 16) as u32;
                    let a = ptx::load4(p);
                    unroll!(GROUP, jj, {
                        let j = group * GROUP + jj;
                        sums[i][j] = madd(a, bb[j], sums[i][j]);
                    });
                });
                if ROWS <= 128 && z == 0 && group == 0 && ahead < kt {
                    if native128 { copy_w(sh + next as u32, ahead_w); }
                    else { stage_weights::<FUSED, ROWS, TILED>(sh + next as u32, w, up, nb, weight_tiles, ahead, tid); }
                    ptx::commit();
                }
            });
        });
        if stages == 3 && ahead < kt {
            ptx::wait_one();
        } else {
            ptx::wait();
        }
        ptx::sync();
        if native128 && ahead < kt { ahead_x = ahead_x.wrapping_add(256); ahead_w = ahead_w.add(128); }
        c += 2;
        current += stride;
        if current == stages * stride { current = 0; }
        next += stride;
        if next == stages * stride { next = 0; }
    }
    if FUSED {
        // Each warp owns matching gate/up columns; emit BF16 down fragments directly.
        let pair = |ga: f32, gb: f32, ua: f32, ub: f32, row: usize| {
            if row >= t as usize { return 0; }
            let silu = |a: f32, b: f32| {
                let (a,b) = if ROUND { rounded_pair(a,b) } else { (a,b) };
                let a = a / (1.0 + ptx::exp(-a));
                let a = if ROUND { rounded_pair(a,0.0).0 } else { a }; a*b
            };
            bf16_pair(silu(ga, ua), silu(gb, ub))
        };
        each_row!(i, {
            let row_tile = mb + mw + i;
            if row_tile * 16 < t as usize {
                let row = row_tile * 16 + lane / 4;
                unroll!(2, j, {
                    let (g0, g1) = (sums[i][j * 2], sums[i][j * 2 + 1]);
                    let (u0, u1) = (sums[i][j * 2 + 4], sums[i][j * 2 + 5]);
                    let p0 = pair(g0.a, g0.b, u0.a, u0.b, row);
                    let p1 = pair(g0.c, g0.d, u0.c, u0.d, row + 8);
                    let p2 = pair(g1.a, g1.b, u1.a, u1.b, row);
                    let p3 = pair(g1.c, g1.d, u1.c, u1.d, row + 8);
                    let tile = row_tile * (n as usize / 16) + nb / 2 + nw / 4 + j;
                    *y.cast::<AFrag>().add(tile * 32 + lane) = AFrag([p0, p1, p2, p3]);
                });
            }
        });
        return;
    }
    if RESIDUAL {
        // Reuse completed copy stages for one 64-row epilogue at a time.
        // Each reduction uses the existing RMS kernel's four loads and tree.
        each_row!(i, {
            unroll!(8, j, {
                store_residual_precision::<false, false>(sh, y, sums[i][j],
                    (mb + mw + i) * 16 + lane / 4, (nb + nw + j) * 8 + lane % 4 * 2,
                    warp / 2 * 16 + lane / 4, warp % 2 * 64 + j * 8 + lane % 4 * 2,
                    t as usize, n as usize);
            });
            ptx::sync();
            let mut r = warp;
            while r < 64 {
                let row = (mb + r / 16 * (ROWS / 64) + i) * 16 + r % 16;
                let mut squares = 0.0f32;
                if row < t as usize {
                    unroll!(4, j, {
                        let v = ptx::load(sh + ((r * 128 + lane + j * 32) * 4) as u32);
                        squares += v * v;
                    });
                }
                squares = crate::layer::sum(squares);
                if lane == 0 && row < t as usize {
                    *partials.add(row * (n as usize / 128) + nb / 16) = squares;
                }
                r += 8;
            }
            ptx::sync();
        });
        return;
    }
    each_row!(i, {
        unroll!(8, j, {
            store_wide::<ROUND>(
                y,
                sums[i][j],
                (mb + mw + i) * 16 + lane / 4,
                (nb + nw + j) * 8 + lane % 4 * 2,
                t as usize,
                n as usize,
            );
        });
    });
}

/// BF16 input, tile128x128; block256/grid(ceil(T/128),N/128), static shared49152.
/// W/XP aligned16, Y aligned8; N multiple128, K multiple64; arbitrary T. Output FP32[T,N].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_pipe3_strided(
    w: *const u32,
    x: *const u32,
    y: *mut f32,
    n: u32,
    k: u32,
    t: u32,
    weight_stride: u32,
) {
    wide::<false, false, 8, 128, false, false, false>([w; 3], w, x, [y; 3], [n; 3], k, t, weight_stride, core::ptr::null_mut(), None);
}

/// Long prefill: block256/grid(ceil(T/256),N/128), dynamic shared73728.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_m256_strided(
    w: *const u32, x: *const u32, y: *mut f32,
    n: u32, k: u32, t: u32, weight_stride: u32,
) {
    wide::<false, false, 8, 256, false, false, false>([w; 3], w, x, [y; 3], [n; 3], k, t, weight_stride, core::ptr::null_mut(), None);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_pipe3_tiled(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, stride: u32,
) {
    wide::<false, false, 8, 128, true, false, false>([w; 3], w, x, [y; 3], [n; 3], k, t, stride, core::ptr::null_mut(), None);
}

/// Long prefill on blocked 64-row weights; same launch as m256_strided.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_m256_tiled(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, stride: u32,
) {
    wide::<false, false, 8, 256, true, false, false>([w; 3], w, x, [y; 3], [n; 3], k, t, stride, core::ptr::null_mut(), None);
}

/// Projection, residual and canonical RMS partials; block256, static shared49152.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_rms_m128(
    w: *const u32, x: *const u32, state: *mut f32, partials: *mut f32,
    n: u32, k: u32, t: u32, weight_stride: u32,
) {
    wide::<false, false, 8, 128, false, true, false>([w; 3], w, x, [state; 3], [n; 3], k, t, weight_stride, partials, None);
}

/// Long-K projection with the same residual/RMS epilogue; dynamic shared73728.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_rms_m256(
    w: *const u32, x: *const u32, state: *mut f32, partials: *mut f32,
    n: u32, k: u32, t: u32, weight_stride: u32,
) {
    wide::<false, false, 4, 256, false, true, false>([w; 3], w, x, [state; 3], [n; 3], k, t, weight_stride, partials, None);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_rms_m128_tiled(
    w: *const u32, x: *const u32, state: *mut f32, partials: *mut f32,
    n: u32, k: u32, t: u32, weight_stride: u32,
) {
    wide::<false, false, 8, 128, true, true, false>([w; 3], w, x, [state; 3], [n; 3], k, t, weight_stride, partials, None);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_rms_m256_tiled(
    w: *const u32, x: *const u32, state: *mut f32, partials: *mut f32,
    n: u32, k: u32, t: u32, weight_stride: u32,
) {
    wide::<false, false, 4, 256, true, true, false>([w; 3], w, x, [state; 3], [n; 3], k, t, weight_stride, partials, None);
}

/// Grouped BF16 GEMM, packed W [E,N,K], packed X [E,ceil(T/16)*16,K], Y [E,T,N].
/// Same requirements as pipe3_strided; grid (ceil(T/128),N/128,E), block 256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_pipe3_grouped(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32,
) {
    let expert = core::arch::nvptx::_block_idx_z() as usize;
    let w = w.add(expert * n as usize * k as usize / 2);
    let x = x.add(expert * ((t as usize + 15) / 16) * 16 * k as usize / 2);
    let y = y.add(expert * t as usize * n as usize);
    wide::<false, false, 8, 128, false, false, false>([w; 3], w, x, [y; 3], [n; 3], k, t, k, core::ptr::null_mut(), None);
}
/// One packed input, disjoint Q/K/V weights and outputs; no concatenation copy.
/// Block256/grid(ceil(T/128),(NQ+2*NKV)/128), NQ/NKV multiples128, K multiple64.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_qkv(
    wq: *const u32,
    wk: *const u32,
    wv: *const u32,
    x: *const u32,
    yq: *mut f32,
    yk: *mut f32,
    yv: *mut f32,
    nq: u32,
    nkv: u32,
    k: u32,
    t: u32,
) {
    wide::<false, true, 8, 128, false, false, false>([wq, wk, wv], wq, x, [yq, yk, yv], [nq, nkv, nkv], k, t, k, core::ptr::null_mut(), None);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_qkv_tiled(
    qw: *const u32, kw: *const u32, vw: *const u32, x: *const u32,
    q: *mut f32, key: *mut f32, value: *mut f32, nq: u32, nkv: u32, k: u32, t: u32,
) {
    wide::<false, true, 8, 128, true, false, false>([qw, kw, vw], qw, x, [q, key, value], [nq, nkv, nkv], k, t, k, core::ptr::null_mut(), None);
}

/// Fused SwiGLU -> BF16 MMA A fragments, ceil(T/16)*16*N*2 output bytes.
/// Block256/grid(ceil(T/128),N/64), static shared49152. N/K multiples64.
/// Output must not alias XP.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gate_up_pair_packed(
    w: *const u32,
    u: *const u32,
    x: *const u32,
    y: *mut u32,
    n: u32,
    k: u32,
    t: u32,
) {
    wide::<true, false, 8, 128, false, false, false>([w; 3], u, x, [y.cast(); 3], [n; 3], k, t, k, core::ptr::null_mut(), None);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gate_up_pair_tiled(
    w: *const u32, u: *const u32, x: *const u32, y: *mut u32, n: u32, k: u32, t: u32,
) {
    wide::<true, false, 8, 128, true, false, false>([w; 3], u, x, [y.cast(); 3], [n; 3], k, t, k, core::ptr::null_mut(), None);
}

use crate::projection_params::GdnProjection;
struct Gates {
    heads: usize,
    beta: *mut f32,
    decay: *mut f32,
    alog: *const u16,
    dt: *const u16,
    precision: u32,
}



#[inline(always)]
unsafe fn rounded_pair(a: f32, b: f32) -> (f32, f32) {
    let pair = bf16_pair(a, b);
    (if a.to_bits() & 0x7f800000 == 0x7f800000 { a } else { f32::from_bits(pair << 16) },
     if b.to_bits() & 0x7f800000 == 0x7f800000 { b } else { f32::from_bits(pair & 0xffff0000) })
}








#[inline(always)]
unsafe fn gate_epilogue(w: *const u32, up: *const u32, x: *const u32, a: *mut f32,
    gates: Gates, mi: usize, k: usize, t: usize, tid: usize) {
    let (lane, mt) = (tid % 32, mi * 8 + tid / 32);
    let first = narrow_total(w, x, mt, 0, k / 16, k / 16, lane, gates.heads);
    unroll!(6, j, {
        if j * 8 < gates.heads {
            store_wide::<true>(a, first[j], mt * 16 + lane / 4,
                j * 8 + lane % 4 * 2, t, gates.heads);
        }
    });
    let second = narrow_total(up, x, mt, 0, k / 16, k / 16, lane, gates.heads);
    unroll!(6, j, {
        let head = j * 8 + lane % 4 * 2;
        if head < gates.heads {
            let value = second[j];
            unroll!(2, r, {
                let row = mt * 16 + lane / 4 + r * 8;
                if row < t {
                    let (b0, b1) = if r == 0 { rounded_pair(value.a, value.b) }
                        else { rounded_pair(value.c, value.d) };
                    let at = row * gates.heads + head;
                    let (beta0, decay0) = crate::flash_gdn::gate_values(*a.add(at), b0, gates.alog, gates.dt, head, gates.precision);
                    let (beta1, decay1) = crate::flash_gdn::gate_values(*a.add(at + 1), b1, gates.alog, gates.dt, head + 1, gates.precision);
                    *gates.beta.add(head * t + row) = beta0;
                    *gates.beta.add((head + 1) * t + row) = beta1;
                    *gates.decay.add(head * t + row) = decay0;
                    *gates.decay.add((head + 1) * t + row) = decay1;
                }
            });
        }
    });
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_pipe3_strided_bf16(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, stride: u32,
) {
    wide::<false, false, 8, 128, false, false, true>([w; 3], w, x, [y; 3], [n; 3], k, t, stride, core::ptr::null_mut(), None);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_qkv_bf16(
    qw: *const u32, kw: *const u32, vw: *const u32, x: *const u32,
    q: *mut f32, key: *mut f32, value: *mut f32, nq: u32, nkv: u32, k: u32, t: u32,
) {
    let tile = _block_idx_y() as usize;
    let qt = nq as usize/64;
    let kt = nkv as usize/64;
    let (w, y, n, tile) = if tile < qt { (qw,q,nq,tile) }
        else if tile < qt+kt { (kw,key,nkv,tile-qt) }
        else { (vw,value,nkv,tile-qt-kt) };
    narrow_precision::<false,false,true, 1>(w,w,x,y,n,k,t,k,_block_idx_x() as usize,tile*8,false);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_qkv_bf16(
    wq: *const u32, wk: *const u32, wv: *const u32, x: *const u32,
    yq: *mut f32, yk: *mut f32, yv: *mut f32, nq: u32, nkv: u32, k: u32, t: u32,
) {
    wide::<false, true, 8, 128, false, false, true>([wq, wk, wv], wq, x, [yq, yk, yv], [nq, nkv, nkv], k, t, k, core::ptr::null_mut(), None);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gdn_bf16(
    p: GdnProjection,
) {
    wide::<false, true, 8, 128, false, false, true>(
        [p.weights[0] as _, p.weights[1] as _, p.weights[2] as _], p.weights[3] as _,
        p.input as _, [p.outputs[0] as _, p.outputs[1] as _, p.outputs[2] as _],
        [p.qkv_rows, p.z_rows, 128], p.width, p.tokens, p.width,
        core::ptr::null_mut(), Some(Gates { heads: p.heads as usize, beta: p.outputs[3] as _, decay: p.outputs[4] as _,
            alog: p.alog as _, dt: p.dt as _, precision: p.parameter_precision }));
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gate_up_pair_fast(
    w: *const u32, u: *const u32, x: *const u32, y: *mut f32,
    n: u32, k: u32, t: u32,
) {
    wide::<true, false, 4, 128, false, false, false>([w; 3], u, x, [y; 3], [n; 3], k, t, k, core::ptr::null_mut(), None);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gate_up_pair_bf16(
    w: *const u32, u: *const u32, x: *const u32, y: *mut f32,
    n: u32, k: u32, t: u32,
) {
    wide::<true, false, 2, 128, false, false, true>([w; 3], u, x, [y; 3], [n; 3], k, t, k, core::ptr::null_mut(), None);
}

#[inline(always)]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t16n64_native(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, weight_stride: u32,
) {
    narrow_precision::<false,false,true, 1>(w,w,x,y,n,k,t,weight_stride,_block_idx_x() as usize,_block_idx_y() as usize*8,false);
}
