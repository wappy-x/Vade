//! Full-row prefill tiles: 256x128x64, two L2-only asynchronous stages.
use super::*;
use crate::ptx;
use core::arch::asm;

const STAGE: usize = 49152;

macro_rules! unroll {
    (2, $i:ident, $body:block) => {
        { const $i: usize = 0; $body } { const $i: usize = 1; $body }
    };
    (3, $i:ident, $body:block) => {
        unroll!(2, $i, $body); { const $i: usize = 2; $body }
    };
    (4, $i:ident, $body:block) => {
        unroll!(3, $i, $body); { const $i: usize = 3; $body }
    };
    (8, $i:ident, $body:block) => {
        unroll!(4, $i, $body);
        { const $i: usize = 4; $body } { const $i: usize = 5; $body }
        { const $i: usize = 6; $body } { const $i: usize = 7; $body }
    };
}

#[inline(always)]
unsafe fn stage(sh: u32, w: *const u32, x: *const u32, x_advance: u64, w_advance: u64, offset: u32) {
    asm!(r#"{{
        .reg .u32 d;
        .reg .u64 p;
        add.u32 d, {offset}, {sh};
        cp.async.cg.shared.global [d], [{x}], 16;
        add.u64 p, {x}, {xa};
        cp.async.cg.shared.global [d+8192], [p], 16;
        add.u64 p, p, {xa};
        cp.async.cg.shared.global [d+16384], [p], 16;
        add.u64 p, p, {xa};
        cp.async.cg.shared.global [d+24576], [p], 16;
        cp.async.cg.shared.global [d+32768], [{w}], 16;
        add.u64 p, {w}, {wa};
        cp.async.cg.shared.global [d+40960], [p], 16;
    }}"#, sh=in(reg32)sh, x=in(reg64)x, w=in(reg64)w,
        xa=in(reg64)x_advance, wa=in(reg64)w_advance, offset=in(reg32)offset);
}

// Exact tile halves the per-thread accumulator footprint, leaving room for
// both balanced pairs without local-memory spills.
#[inline(always)]
unsafe fn stage_exact(sh: u32, w: *const u32, x: *const u32, xa: u64, wa: u64, offset: u32) {
    asm!(r#"{{
        .reg .u32 d;
        .reg .u64 p;
        add.u32 d, {offset}, {sh};
        cp.async.cg.shared.global [d], [{x}], 16;
        add.u64 p, {x}, {xa};
        cp.async.cg.shared.global [d+4096], [p], 16;
        add.u64 p, p, {xa};
        cp.async.cg.shared.global [d+8192], [p], 16;
        add.u64 p, p, {xa};
        cp.async.cg.shared.global [d+12288], [p], 16;
        cp.async.cg.shared.global [d+16384], [{w}], 16;
        add.u64 p, {w}, {wa};
        cp.async.cg.shared.global [d+20480], [p], 16;
    }}"#, sh=in(reg32)sh, x=in(reg64)x, w=in(reg64)w,
        xa=in(reg64)xa, wa=in(reg64)wa, offset=in(reg32)offset);
}

#[inline(always)]
unsafe fn stage_o(sh: u32, w: *const u32, x: *const u32, x_advance: u64, w_advance: u64, offset: u32) {
    asm!(r#"{{
        .reg .u32 d;
        .reg .u64 p;
        add.u32 d, {offset}, {sh};
        cp.async.cg.shared.global [d], [{x}], 16;
        add.u64 p, {x}, {xa};
        cp.async.cg.shared.global [d+8192], [p], 16;
        cp.async.cg.shared.global [d+16384], [{w}], 16;
        add.u64 p, {w}, {wa};
        cp.async.cg.shared.global [d+24576], [p], 16;
        add.u64 p, p, {wa};
        cp.async.cg.shared.global [d+32768], [p], 16;
        add.u64 p, p, {wa};
        cp.async.cg.shared.global [d+40960], [p], 16;
    }}"#, sh=in(reg32)sh, x=in(reg64)x, w=in(reg64)w,
        xa=in(reg64)x_advance, wa=in(reg64)w_advance, offset=in(reg32)offset);
}

#[inline(always)]
unsafe fn load_a<const OFFSET: usize>(base: u32) -> [u32; 4] {
    let (a,b,c,d): (u32,u32,u32,u32);
    asm!("{{ .reg .u32 p; add.u32 p, {base}, {offset}; ld.shared.v4.b32 {{{a},{b},{c},{d}}}, [p]; }}",
        base=in(reg32)base, offset=const OFFSET, a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d);
    [a,b,c,d]
}

#[inline(always)]
unsafe fn load_b<const OFFSET: usize>(base: u32) -> [u32; 2] {
    let (a,b): (u32,u32);
    asm!("{{ .reg .u32 p; add.u32 p, {base}, {offset}; ld.shared.v2.b32 {{{a},{b}}}, [p]; }}",
        base=in(reg32)base, offset=const OFFSET, a=out(reg32)a,b=out(reg32)b);
    [a,b]
}

#[inline(always)]
#[allow(non_upper_case_globals)]
unsafe fn wide<const TILED: bool, const FUSED: bool, const QKV: bool, const RESIDUAL: bool, const O: bool, const ROUND: bool, const FP8: bool, const EXACT: bool>(weights: [*const u32; 3], up: *const u32, x: *const u32,
    outputs: [*mut f32; 3], widths: [u32; 3], k: u32, t: u32, stride: u32, partials: *mut f32, scales: [*const u16; 3], output_strides: [u32; 3]) {
    let n = if QKV { widths[0] + widths[1] + widths[2] } else { widths[0] };
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let tile_m = if O || EXACT {128} else {256};
    let fragments = if EXACT {2} else {4};
    let stage_bytes = if EXACT {24576} else {STAGE};
    let tile_n = if O {256} else if FUSED || EXACT {64} else {128};
    let warp_columns = tile_n / 32;
    let epilogue_rows = tile_m / 4;
    let mt = t as usize / tile_m;
    let nt = n as usize / tile_n;
    if t as usize * k as usize * 2 > 16 * 1024 * 1024 {
        let linear = ni * mt + mi;
        // Reuse resident weights across four row tiles when packed X fits L2.
        let group = if EXACT { if k <= 8192 {8} else {4} } else if FP8 { if k <= 5120 {8} else if k <= 8192 {4} else {2} } else if ROUND && !TILED {4} else if RESIDUAL { if k <= 8192 {8} else {4} } else if k <= 8192 {4} else {2};
        let first = linear / (nt * group) * group;
        let rows = group.min(mt - first);
        let within = linear - first * nt;
        mi = first + within % rows;
        ni = within / rows;
    }
    // Keep each CTA inside one projection while sharing the packed input.
    let mut part = 0;
    if QKV {
        let qtiles = widths[0] as usize / tile_n;
        let ktiles = widths[1] as usize / tile_n;
        if ni >= qtiles + ktiles {
            ni -= qtiles + ktiles;
            part = 2;
        } else if ni >= qtiles {
            ni -= qtiles;
            part = 1;
        }
    }
    let (w, y, n, ws) = if !QKV || part == 0 {
        (weights[0], outputs[0], widths[0], scales[0])
    } else if part == 1 {
        (weights[1], outputs[1], widths[1], scales[1])
    } else {
        (weights[2], outputs[2], widths[2], scales[2])
    };
    let output_stride = if output_strides[part] == 0 { n } else { output_strides[part] } as usize;
    let (mb, nb) = (mi * (tile_m / 16), ni * (tile_n / 8));
    let mw = if FUSED {warp / 4 * 4} else {warp / warp_columns * fragments};
    let nw = if FUSED { warp % 4 / 2 * 8 + warp % 2 * 2 } else { warp % warp_columns * 4 };
    let kt = k as usize / 16;
    let weight_tiles = stride as usize / 16;
    let sh = 0u32;
    let b = tid / 64;
    let row = nb + if FUSED { b % 4 } else { b };
    let weights = if FUSED && b & 4 != 0 { up } else { w };
    // Advance the K pointers; tile addressing is invariant across stages.
    let mut source_x = x.add((mb + tid / 128) * kt * 128 + tid % 128 * 4);
    // Both layouts stage the same fragments without repacking resident weights.
    let weight_row = if TILED { row / 8 * weight_tiles * 512 + row % 8 * 256 }
        else { row * weight_tiles * 64 };
    let mut source_w = weights.add(weight_row + tid % 64 * 4);
    let mut source_w8 = w.cast::<u8>().add((nb + tid / 32) * kt * 128 + tid % 32 * 16);
    let x_advance = kt as u64 * if EXACT {1024} else {2048};
    // Hybrid packed8 gate/up advances four row tiles, unlike blocked64.
    let w_advance = if EXACT || FUSED && TILED { 4096 } else { weight_tiles as u64 * if FUSED { 1024 } else { 2048 } };
    let offset = tid as u32 * 16;
    let mut sums = [[ZERO; 4]; 4];
    let mut left = [[ZERO;4];4];
    let mut right = [[ZERO;4];4];
    if EXACT { stage_exact(sh, source_w, source_x, x_advance, w_advance, offset); }
    else if FP8 { stage_fp8(sh, source_w8, source_x, x_advance, offset); }
    else if O { stage_o(sh, source_w, source_x, x_advance, w_advance, offset); }
    else { stage(sh, source_w, source_x, x_advance, w_advance, offset); }
    ptx::wait();
    ptx::sync();
    let (mut c, mut current, mut next) = (0usize, 0usize, stage_bytes);
    let mut range = 0;
    while c < kt {
        let ahead = c + 4;
        if ahead < kt {
            source_x = source_x.add(512);
            source_w = source_w.add(if TILED { 2048 } else { 256 });
            if EXACT { stage_exact(sh + next as u32, source_w, source_x, x_advance, w_advance, offset); }
            else if FP8 { source_w8 = source_w8.add(512); stage_fp8(sh + next as u32, source_w8, source_x, x_advance, offset); }
            else
            if O { stage_o(sh + next as u32, source_w, source_x, x_advance, w_advance, offset); }
            else { stage(sh + next as u32, source_w, source_x, x_advance, w_advance, offset); }
        }
        if FP8 {
        macro_rules! step {
            ($fast:literal, $packed:expr, $scale:expr) => {
                unroll!(4, z, {
                    let base_a = (current + mw * 2048 + z * 512 + lane * 16) as u32;
                    let mut b = [[0u32; 2]; 4];
                    if FP8 {
                        let base_b = (current + 32768 + nw * 512 + z * 128 + lane * 4) as u32;
                        unroll!(4, j, { b[j] = promoted_b::<$fast>(base_b + (j * 512) as u32, $packed, $scale); });
                    } else {
                        let base_b = (current + 32768 + nw * 1024 + z * 256 + lane * 8) as u32;
                        unroll!(4, j, {
                            b[j] = if FUSED && j >= 2 { load_b::<{(j + 2) * 1024}>(base_b) }
                                else { load_b::<{j * 1024}>(base_b) };
                        });
                    }
                    unroll!(4, i, {
                        let a = load_a::<{i * 2048}>(base_a);
                        unroll!(4, j, { sums[i][j] = madd(a, b[j], sums[i][j]); });
                    });
                });
            };
        }
        if FP8 {
            // Match fp8_dequant_bf16_mma_weights: one scale per 128x128 block.
            let bits = *ws.add(ni * (kt / 8) + c / 8) as u32;
            let shifted = bits + 0x3800;
            if crate::fp8_bf16::fast_scale(bits) { step!(true, shifted | (shifted << 16), 0.0); }
            else { step!(false, 0, f32::from_bits(bits << 16)); }
        } else { step!(true, 0, 0.0); }
        } else {
        unroll!(4, z, {
            let base_b = (current + tile_m * 128 + nw * 1024 + z * 256 + lane * 8) as u32;
            let base_a = (current + mw * 2048 + z * 512 + lane * 16) as u32;
            let a = [load_a::<0>(base_a), load_a::<2048>(base_a),
                if EXACT {[0;4]} else {load_a::<4096>(base_a)}, if EXACT {[0;4]} else {load_a::<6144>(base_a)}];
            unroll!(4, j, {
                let b = if FUSED && j >= 2 { load_b::<{(j + 2) * 1024}>(base_b) }
                    else { load_b::<{j * 1024}>(base_b) };
                unroll!(4, i, { if i < fragments {sums[i][j] = madd(a[i], b, sums[i][j]);} });
            });
            // K=320 has five K16 instructions per range, crossing K64 stages.
            if EXACT && c + z + 1 == crate::canonical::range(kt, range).1 {
                unroll!(4,i,{ if i < fragments {
                    unroll!(4,j,{
                        let v = sums[i][j];
                        if range == 0 {left[i][j] = v;}
                        else if range == 2 {right[i][j] = v;}
                        else {
                            let p = if range == 1 {left[i][j]} else {right[i][j]};
                            let merged = Acc {a:p.a+v.a,b:p.b+v.b,c:p.c+v.c,d:p.d+v.d};
                            if range == 1 {left[i][j] = merged;} else {right[i][j] = merged;}
                        }
                        sums[i][j] = ZERO;
                    });
                }});
                range += 1;
            }
        });
        }
        c += 4;
        current += stage_bytes;
        if current == 2 * stage_bytes { current = 0; }
        next += stage_bytes;
        if next == 2 * stage_bytes { next = 0; }
        ptx::wait();
        ptx::sync();
    }
    if RESIDUAL {
        // Two row fragments share one epilogue barrier and reduction pass.
        unroll!(2, half, {
            unroll!(2, r, {
                const i: usize = half * 2 + r;
                unroll!(4, j, {
                    super::shared::store_residual_precision::<ROUND, TILED>(sh, y, sums[i][j],
                        (mb + mw + i) * 16 + lane / 4, (nb + nw + j) * 8 + lane % 4 * 2,
                        r * epilogue_rows + warp / warp_columns * 16 + lane / 4 + if O {warp % 8 / 4 * 64} else {0},
                        warp % 4 * 32 + j * 8 + lane % 4 * 2,
                        t as usize, n as usize);
                });
            });
            ptx::sync();
            let mut r = warp;
            while r < 128 {
                let local = if O {r % 64} else {r};
                let row = (mb + local % epilogue_rows / 16 * 4 + half * 2 + local / epilogue_rows) * 16 + local % 16;
                let mut squares = 0.0f32;
                unroll!(4, j, {
                    let value = ptx::load(sh + ((r * 128 + lane + j * 32) * 4) as u32);
                    squares += value * value;
                });
                squares = crate::layer::sum(squares);
                if lane == 0 { *partials.add(row * (n as usize / 128) + nb / 16 + if O {r / 64} else {0}) = squares; }
                r += 16;
            }
            ptx::sync();
        });
    } else if FUSED {
        let pair = |g: f32, u: f32| {
            let (g,u) = if ROUND { let p = output_pair::<true>(g,u); (f32::from_bits(p as u32),f32::from_bits((p>>32) as u32)) } else { (g,u) };
            let g = g / (1.0 + ptx::exp(-g));
            (if ROUND { f32::from_bits(output_pair::<true>(g,0.0) as u32) } else { g }) * u
        };
        unroll!(4, i, {
            let (g0, g1) = (sums[i][0], sums[i][1]);
            let (u0, u1) = (sums[i][2], sums[i][3]);
            let values = [bf16_pair(pair(g0.a,u0.a),pair(g0.b,u0.b)),
                bf16_pair(pair(g0.c,u0.c),pair(g0.d,u0.d)),
                bf16_pair(pair(g1.a,u1.a),pair(g1.b,u1.b)),
                bf16_pair(pair(g1.c,u1.c),pair(g1.d,u1.d))];
            let tile = (mb + mw + i) * (n as usize / 16) + nb / 2 + warp % 4;
            *y.cast::<AFrag>().add(tile * 32 + lane) = AFrag(values);
        });
    } else {
        unroll!(4, i, {
            unroll!(4, j, {
                if i < fragments {
                let a = if EXACT {
                    let l = left[i][j]; let r = right[i][j];
                    Acc {a:l.a+r.a,b:l.b+r.b,c:l.c+r.c,d:l.d+r.d}
                } else {sums[i][j]};
                let row = (mb + mw + i) * 16 + lane / 4;
                let col = (nb + nw + j) * 8 + lane % 4 * 2;
                *y.add(row * output_stride + col).cast::<u64>() = output_pair::<ROUND>(a.a,a.b);
                *y.add((row + 8) * output_stride + col).cast::<u64>() = output_pair::<ROUND>(a.c,a.d);
                }
            });
        });
    }
}

/// Full 256-row tiles, block512, dynamic shared98304, K >= 64 and K%64 == 0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t256n128_tiled(w:*const u32,x:*const u32,y:*mut f32,n:u32,k:u32,t:u32,stride:u32) {
    wide::<true, false, false, false, false, false, false, false>([w;3],w,x,[y;3],[n;3],k,t,stride,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}

/// Paired gate/up tile256x64; fused SwiGLU writes the down operand directly.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gate_up_pair_t256n64_tiled(w:*const u32,up:*const u32,x:*const u32,y:*mut f32,n:u32,k:u32,t:u32) {
    wide::<true, true, false, false, false, false, false, false>([w;3],up,x,[y;3],[n;3],k,t,k,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}

/// Q/K/V tile256x128x64, block512, dynamic shared98304; each width is a multiple of 128.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_qkv_t256n128_tiled(wq:*const u32,wk:*const u32,wv:*const u32,x:*const u32,
    yq:*mut f32,yk:*mut f32,yv:*mut f32,nq:u32,nkv:u32,k:u32,t:u32) {
    wide::<true, false, true, false, false, false, false, false>([wq,wk,wv],wq,x,[yq,yk,yv],[nq,nkv,nkv],k,t,k,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}

/// Packed8 weights, with the same full tiles and shared-memory layout.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t256n128_packed(w:*const u32,x:*const u32,y:*mut f32,n:u32,k:u32,t:u32,stride:u32) {
    wide::<false, false, false, false, false, false, false, false>([w;3],w,x,[y;3],[n;3],k,t,stride,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}

/// Q/K/V use their existing packed8 arenas; no second weight copy.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_qkv_t256n128_packed(wq:*const u32,wk:*const u32,wv:*const u32,x:*const u32,
    yq:*mut f32,yk:*mut f32,yv:*mut f32,nq:u32,nkv:u32,k:u32,t:u32) {
    wide::<false, false, true, false, false, false, false, false>([wq,wk,wv],wq,x,[yq,yk,yv],[nq,nkv,nkv],k,t,k,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}

/// Full-row residual and canonical RMS partials, block512, shared98304.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_rms_t256n128_tiled(a: crate::kernel_params::PrefillResidual) {
    let w = a.weights as *const u32;
    wide::<true, false, false, true, false, false, false, false>([w;3],w,a.input as *const u32,[a.output as *mut f32;3],
        [a.width;3],a.inner,a.rows,a.stride,a.partials as *mut f32,[core::ptr::null();3],[0;3]);
}

/// 8B O projection, 128x256x64, block512, shared98304.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_rms_o_t128n256(a: crate::kernel_params::PrefillResidual) {
    let w = a.weights as *const u32;
    wide::<true, false, false, true, true, false, false, false>([w;3],w,a.input as *const u32,[a.output as *mut f32;3],
        [a.width;3],a.inner,a.rows,a.stride,a.partials as *mut f32,[core::ptr::null();3],[0;3]);
}


#[inline(always)]
unsafe fn stage_fp8(sh: u32, w: *const u8, x: *const u32, x_advance: u64, offset: u32) {
    asm!(r#"{{
        .reg .u32 d;
        .reg .u64 p;
        add.u32 d, {offset}, {sh};
        cp.async.cg.shared.global [d], [{x}], 16;
        add.u64 p, {x}, {xa};
        cp.async.cg.shared.global [d+8192], [p], 16;
        add.u64 p, p, {xa};
        cp.async.cg.shared.global [d+16384], [p], 16;
        add.u64 p, p, {xa};
        cp.async.cg.shared.global [d+24576], [p], 16;
        cp.async.cg.shared.global [d+32768], [{w}], 16;
    }}"#, sh=in(reg32)sh, x=in(reg64)x, w=in(reg64)w,
        xa=in(reg64)x_advance, offset=in(reg32)offset);
}


#[inline(always)]
pub(crate) unsafe fn promoted_b<const FAST: bool>(p: u32, packed_scale: u32, scale: f32) -> [u32; 2] {
    let raw: u32;
    asm!("ld.shared.b32 {r}, [{p}];", r=out(reg32)raw, p=in(reg32)p);
    [crate::fp8_bf16::scaled_pair::<FAST>(raw as u16, packed_scale, scale),
        crate::fp8_bf16::scaled_pair::<FAST>((raw >> 16) as u16, packed_scale, scale)]
}


#[inline(always)]
unsafe fn output_pair<const ROUND: bool>(a: f32, b: f32) -> u64 {
    let (mut a, mut b) = (a.to_bits(), b.to_bits());
    if ROUND {
        let pair = bf16_pair(f32::from_bits(a), f32::from_bits(b));
        // Match logits_bf16 for infinities and NaNs as well as finite values.
        if a & 0x7f800000 != 0x7f800000 { a = pair << 16; }
        if b & 0x7f800000 != 0x7f800000 { b = pair & 0xffff0000; }
    }
    a as u64 | ((b as u64) << 32)
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t256n128_bf16(w:*const u32,x:*const u32,y:*mut f32,n:u32,k:u32,t:u32,stride:u32) {
    wide::<false,false, false, false, false, true, false, false>([w;3],w,x,[y;3],[n;3],k,t,stride,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_residual_bf16_t256n128(w:*const u32,x:*const u32,y:*mut f32,partials:*mut f32,n:u32,k:u32,t:u32,stride:u32) {
    wide::<false,false, false, true, false, true, false, false>([w;3],w,x,[y;3],[n;3],k,t,stride,partials,[core::ptr::null();3],[0;3]);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_gate_up_pair_t256n64_bf16(w:*const u32,up:*const u32,x:*const u32,y:*mut f32,n:u32,k:u32,t:u32) {
    wide::<false,true, false, false, false, true, false, false>([w;3],up,x,[y;3],[n;3],k,t,k,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_fp8_t256n128(p: crate::kernel_params::Fp8PrefillGemm) {
    let w = p.weights.map(|w| w.cast::<u32>());
    if p.partials.is_null() {
        wide::<false,false, false, false, false, true, true, false>(w,w[0],p.input,p.outputs,p.widths,p.k,p.tokens,p.k,p.partials,p.scales,[0;3]);
    } else {
        wide::<false,false, false, true, false, true, true, false>(w,w[0],p.input,p.outputs,p.widths,p.k,p.tokens,p.k,p.partials,p.scales,[0;3]);
    }
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prefill_qkv_fp8_t256n128(p: crate::kernel_params::Fp8PrefillGemm) {
    let w = p.weights.map(|w| w.cast::<u32>());
    wide::<false,false, true, false, false, true, true, false>(w,w[0],p.input,p.outputs,p.widths,p.k,p.tokens,p.k,p.partials,p.scales,[0;3]);
}

/// Gemma projection: four fixed K ranges and a balanced register merge.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_bf16_mma_t256n128_exact(a: crate::kernel_params::PrefillProjection) {
    let w = a.weights as *const u32;
    if a.round_bf16 != 0 {
        wide::<true, false, false, false, false, true, false, true>([w;3],w,a.input as *const u32,[a.output as *mut f32;3],
            [a.width;3],a.inner,a.rows,a.stride,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
    } else {
        wide::<true, false, false, false, false, false, false, true>([w;3],w,a.input as *const u32,[a.output as *mut f32;3],
            [a.width;3],a.inner,a.rows,a.stride,core::ptr::null_mut(),[core::ptr::null();3],[0;3]);
    }
}
