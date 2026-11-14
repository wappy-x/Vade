//! Rust port of the first-party phased_selected / fused_core_opt prototypes.
//! Fixed 32 logical lanes and ascending coordinate updates, no FP atomics.
use crate::{
    flash_gdn::rounded as rb,
    next_params::{MxLayer, MxProjection},
};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _thread_idx_x},
};

macro_rules! eight {
    ($i:ident, $b:block) => {{
        const $i: usize = 0;
        $b
    }
    {
        const $i: usize = 1;
        $b
    }
    {
        const $i: usize = 2;
        $b
    }
    {
        const $i: usize = 3;
        $b
    }
    {
        const $i: usize = 4;
        $b
    }
    {
        const $i: usize = 5;
        $b
    }
    {
        const $i: usize = 6;
        $b
    }
    {
        const $i: usize = 7;
        $b
    }};
}
macro_rules! nine {
    ($i:ident, $b:block) => {
        eight!($i, $b);
        {
            const $i: usize = 8;
            $b
        }
    };
}
macro_rules! sixteen {
    ($i:ident, $b:block) => {
        eight!($i, $b);
        {
            const $i: usize = 8;
            $b
        }
        {
            const $i: usize = 9;
            $b
        }
        {
            const $i: usize = 10;
            $b
        }
        {
            const $i: usize = 11;
            $b
        }
        {
            const $i: usize = 12;
            $b
        }
        {
            const $i: usize = 13;
            $b
        }
        {
            const $i: usize = 14;
            $b
        }
        {
            const $i: usize = 15;
            $b
        }
    };
}
#[inline(always)]
unsafe fn load16(p: *const u8) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)p);
    [a, b, c, d]
}
#[inline(always)]
unsafe fn pair(raw: u16) -> [f32; 2] {
    let (h, a, b): (u32, f32, f32);
    asm!("{{ .reg .b16 lo,hi; cvt.rn.f16x2.e4m3x2 {h}, {raw}; mov.b32 {{lo,hi}}, {h}; cvt.f32.f16 {a}, lo; cvt.f32.f16 {b}, hi; }}",
        h=out(reg32)h,a=out(reg32)a,b=out(reg32)b,raw=in(reg16)raw);
    let _ = h;
    [a, b]
}
#[inline(always)]
fn weight(c: u8) -> f32 {
    let x = (c & 127) as u32;
    let v = if x == 127 {
        f32::from_bits(0x7fc00000)
    } else if x < 8 {
        x as f32 * 0.001953125
    } else {
        f32::from_bits((x << 20) + (120 << 23))
    };
    if c & 128 != 0 {
        -v
    } else {
        v
    }
}
#[inline(always)]
unsafe fn row<const MT: usize, const PREFETCH: bool, const PAIR: bool>(
    p: MxProjection,
    r: usize,
    mb: usize,
) {
    let lane = _thread_idx_x() as usize & 31;
    let (n, k, m) = (p.rows as usize, p.cols as usize, p.tokens as usize);
    let w = (p.weights as *const u8).add(r * k);
    let s = (p.scales as *const u8).add(r * k / 32);
    let mut acc = [0.0f32; MT];
    let mut tile = 0;
    while tile < k.div_ceil(256) {
        let at = tile * 256 + lane * 8;
        if at < k {
            let v = load16(w.add(at & !15));
            let words = if lane & 1 == 0 {
                [v[0], v[1]]
            } else {
                [v[2], v[3]]
            };
            let sc = *s.add(at / 32);
            let scale = f32::from_bits(if sc == 0 {
                0x00400000
            } else if sc == 255 {
                0x7fc00000
            } else {
                (sc as u32) << 23
            });
            let mut ww = [0.0; 8];
            eight!(J, {
                if PAIR {
                    if J % 2 == 0 {
                        let f = pair((words[J / 4] >> ((J & 3) * 8)) as u16);
                        ww[J] = f[0] * scale;
                        ww[J + 1] = f[1] * scale;
                    }
                } else {
                    ww[J] = weight((words[J / 4] >> ((J & 3) * 8)) as u8) * scale;
                }
            });
            if PREFETCH && r + 8 < n && lane & 3 == 0 {
                asm!("prefetch.global.L2 [{w}]; prefetch.global.L2 [{s}];",w=in(reg64)w.add(8*k+at),s=in(reg64)s.add(8*k/32+at/32));
            }
            sixteen!(B, {
                if B < MT && mb + B < m {
                    let base = (mb + B) * k + at;
                    let mut xx = [0.0; 8];
                    if p.input_bf16 != 0 {
                        let x = load16((p.input as *const u16).add(base).cast());
                        eight!(J, {
                            xx[J] = f32::from_bits((x[J / 2] >> ((J & 1) * 16)) << 16);
                        });
                    } else {
                        let x = load16((p.input as *const f32).add(base).cast());
                        let y = load16((p.input as *const f32).add(base + 4).cast());
                        eight!(J, {
                            xx[J] = p.f32_input(if J < 4 { x[J % 4] } else { y[J % 4] });
                        });
                    }
                    eight!(J, {
                        acc[B] = core::f32::math::mul_add(ww[J], xx[J], acc[B]);
                    });
                }
            });
        }
        tile += 1;
    }
    sixteen!(B, {
        if B < MT {
            let mut v = acc[B];
            // Prototype shfl_down tree. Only lane zero is stored.
            let mut d = 16u32;
            while d != 0 {
                let x: f32;
                asm!("shfl.sync.down.b32 {x}, {v}, {d}, 31, -1;",x=out(reg32)x,v=in(reg32)v,d=in(reg32)d,options(nomem,nostack));
                v += x;
                d /= 2;
            }
            if lane == 0 && mb + B < m {
                *(p.output as *mut f32).add((mb + B) * n + r) =
                    if p.round != 0 { rb(v) } else { v };
            }
        }
    });
}
#[inline(always)]
unsafe fn project<const SOLO: bool, const PAIR: bool>(layer: MxLayer) {
    let block = _block_idx_x() as u32;
    // Static offsets avoid copying the by-value CUDA parameter table to local memory.
    let mut p = layer.projections[0];
    let mut start = 0;
    eight!(I, {
        if I + 1 < layer.count as usize && block >= layer.ends[I + 1] {
            p = layer.projections[I + 1];
            start = layer.ends[I + 1];
        }
    });
    let local = block - start;
    let rows = p.rows.div_ceil(4);
    let r = (local % rows) as usize * 4 + _thread_idx_x() as usize / 32;
    let group = (local / rows) as usize;
    if r >= p.rows as usize {
        return;
    }
    if SOLO {
        row::<1, true, true>(p, r, 0);
    } else {
        match p.tile() {
            1 => row::<1, false, PAIR>(p, r, group),
            2 => row::<2, false, PAIR>(p, r, group * 2),
            4 => row::<4, false, PAIR>(p, r, group * 4),
            8 if p.round == 0 => row::<8, true, PAIR>(p, r, 0),
            8 => row::<8, false, PAIR>(p, r, 0),
            _ => row::<16, false, PAIR>(p, r, 0),
        }
    }
}
/// Grid: ends[count] blocks of 128 threads. Independent output rows, M=1.
/// # Safety
/// Descriptors must pass MxLayer::push; buffers cover their declared shapes,
/// remain live on the stream, and outputs alias no inputs or other outputs.
#[no_mangle]
pub extern "ptx-kernel" fn flash_next_mxfp8_m1(layer: MxLayer) {resident_flash_next_mxfp8_m1(layer)}
#[inline(always)]
pub(crate) fn resident_flash_next_mxfp8_m1(layer:MxLayer) {
    // SAFETY: Preserve the original kernel contract and launch geometry. The
    // direct entry and checked indirect command adapter retain the same operands
    // through ordered completion. This is the original kernel body verbatim.
    unsafe {
    project::<true, true>(layer)

    }
}

#[no_mangle]
pub extern "ptx-kernel" fn flash_resident_projection_batch(batch:crate::next_params::ResidentProjectionBatch) {resident_flash_resident_projection_batch::<false>(batch)}
#[no_mangle]
pub extern "ptx-kernel" fn flash_resident_projection_bf16(batch:crate::next_params::ResidentProjectionBatch) {resident_flash_resident_projection_bf16(batch)}
#[inline(always)]
pub(crate) fn resident_flash_resident_projection_bf16(batch:crate::next_params::ResidentProjectionBatch) {resident_flash_resident_projection_batch::<true>(batch)}
#[inline(always)]
pub(crate) fn resident_flash_resident_projection_batch<const BF16:bool>(batch:crate::next_params::ResidentProjectionBatch) {
    // SAFETY: The checked host batch funds immutable same-context input and
    // weight rows. Plain weights hold BF16 patterns exactly when BF16 is true.
    // Its prefix grid assigns each block one validated descriptor.
    // MXFP8 warps and plain threads own disjoint output rows, raw stores (the host skips the fill). Owners
    // remain live until stream completion. Plain descriptors never read scales.
    // The blocks of plain descriptors run first: the block index is mapped to
    // (descriptor, block within it) with all plain descriptors before all others,
    // each group in descriptor order. That is a bijection on the same grid, so
    // every row is still computed once by the same arithmetic; the long serial
    // chains of the plain rows now overlap the MXFP8 blocks instead of trailing them.
    unsafe {
        let layer=batch.layer;let block=_block_idx_x() as u32;
        let (job,rest)=crate::plain_admission::batch_block(layer.ends,batch.plain,layer.count as usize,block);
        let plain_first=batch.plain&(1<<job)!=0;
        let mut p=layer.projections[0];
        nine!(I,{if job==I {p=layer.projections[I];}});
        let thread=_thread_idx_x() as usize;
        if plain_first {
            let row=rest as usize*128+thread;if row>=p.rows as usize {return;}
            let cols=p.cols as usize;let weights=if BF16 {(p.weights as *const u16).add(row*cols).cast::<f32>()} else {(p.weights as *const f32).add(row*cols)};
            let sum=if BF16 {
                crate::plain_warp::row_deep::<true>(weights,p.input as *const f32,cols)
            } else if crate::plain_admission::deep_admitted(p.rows as usize,cols,p.weights as usize,p.input as usize) {
                crate::plain_warp::row_deep::<false>(weights,p.input as *const f32,cols)
            } else {
                crate::serial_projection::plain_row!(weights,p.input as *const f32,cols)
            };
            *(p.output as *mut f32).add(row)=if batch.round!=0 {rb(sum)} else {sum};
        } else {
            let index=rest as usize*4+thread/32;if index>=p.rows as usize {return;}
            row::<1,true,true>(p,index,0);
            if thread%32==0 {let output=(p.output as *mut f32).add(index);*output=rb(*output);}
        }
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn flash_resident_projection_plain_bf16(batch:crate::next_params::ResidentProjectionBatch) {resident_flash_resident_projection_plain_bf16(batch)}
#[inline(always)]
pub(crate) fn resident_flash_resident_projection_plain_bf16(batch:crate::next_params::ResidentProjectionBatch) {
    // SAFETY: The host admits only complete aligned BF16 plain matrices in the
    // fold envelope, with FP32 inputs and disjoint complete outputs retained
    // through stream completion. Every block has 160 threads and owns four
    // rows of one checked descriptor. All threads enter the fold's barriers.
    // The block's shared area covers its two product slots. Only live folding
    // lanes write, once per output, after the unchanged ascending add chain.
    unsafe {
        let layer=batch.layer;let block=_block_idx_x() as u32;
        let (job,rest)=crate::plain_admission::batch_block(layer.ends,batch.plain,layer.count as usize,block);
        let mut p=layer.projections[0];nine!(I,{if job==I {p=layer.projections[I];}});
        let first=rest as usize*crate::plain_admission::FOLD_ROWS;let thread=_thread_idx_x() as usize;
        let smem=crate::ptx::shared::<{crate::plain_warp::FOLD_BYTES}>();
        let sum=crate::plain_warp::rows_fold::<true>(p.weights as _,p.input as _,p.cols as usize,first,p.rows as usize,smem);
        if thread<crate::plain_admission::FOLD_ROWS && first+thread<p.rows as usize {
            *(p.output as *mut f32).add(first+thread)=if batch.round!=0 {rb(sum)} else {sum};
        }
    }
}
/// M2..16 rounded GDN or M2..8 general projections: packed pair conversion.
/// # Safety
/// Same buffers/grid contract as flash_next_mxfp8_m1; M is 2..16.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_mxfp8_pair(layer: MxLayer) {
    project::<false, true>(layer)
}
/// M9..16 general projections: prototype's linear bit decoder, identical order.
/// # Safety
/// Same buffers/grid contract as flash_next_mxfp8_m1; M is 9..16.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_mxfp8_bits(layer: MxLayer) {
    project::<false, false>(layer)
}
macro_rules! four {
    ($i:ident, $b:block) => {{
        const $i: usize = 0;
        $b
    }
    {
        const $i: usize = 1;
        $b
    }
    {
        const $i: usize = 2;
        $b
    }
    {
        const $i: usize = 3;
        $b
    }};
}
/// N-row twin of `flash_resident_projection_bf16` (lane multi-row, premise P1; bench only until a ruling). The same
/// descriptors, each with `tokens` = N source rows, N = 2..4: input `[token][cols]` FP32, output `[token][rows]`.
/// Every output keeps the one-row arithmetic and order:
/// - MXFP8 descriptors: `row::<2>` or `row::<4>` of this file, the body `row::<1>` runs for one row (lane L owns columns
///   `tile * 256 + L * 8 + J`, one `fma` chain a lane a token in tile order and J order, the same `shfl` tree, then
///   the same BF16 round of the stored word). One decoded weight is multiplied into N accumulators: one weight read.
/// - plain descriptors (BF16 weight patterns): `plain_warp::row_deep::<true>`, the one-row body, once a (weight row,
///   source row). Two schedules, both bench arms (the first P1 run used a third, a plain loop with no load schedule:
///   its 384 chains set the launch time at every row count):
///   `_rows`: SPREAD = true, one thread a (weight row, source row): a plain descriptor takes `tokens` times its
///   one-row blocks, source row major. The N chains of a weight row run at the same time; its 2 bytes a column are
///   read once a source row (the plain rows of a KDA batch are 6 percent of its weight bytes).
///   SPREAD = false (the one-row grid; a thread runs its weight row's N chains one after the other) was the second bench
///   arm of premise D1: 1.9 to 3.8 times the one-row time on the plain rows, so it has no entry.
/// Grid: `ends[count]` blocks of 128 threads; MXFP8 descriptors as in the one-row entry.
#[inline(always)]
fn resident_projection_bf16_rows<const SPREAD:bool>(batch:crate::next_params::ResidentProjectionBatch) {
    // SAFETY: The caller passes descriptors that satisfy the one-row entry's contract with `tokens` in 2..=4 on
    // every descriptor: weights (and scales) as there; `tokens * cols` initialized FP32 input words and
    // `tokens * rows` output words a descriptor, 16-byte aligned input and plain weights, cols a multiple of 32,
    // owners retained through stream completion, outputs aliasing no input and no other output. `ends` gives an
    // MXFP8 descriptor `rows.div_ceil(4)` blocks and a plain descriptor `rows.div_ceil(128)` blocks (times `tokens`
    // when SPREAD), so each block owns one descriptor's rows exactly once a source row (`batch_block` is a bijection
    // on the grid). A warp (MXFP8) owns one weight row and stores its `tokens` output words; a plain thread owns one
    // weight row and stores one output word (SPREAD) or its `tokens` words. `row_deep` and `row` read only inside
    // the declared extents, as in the one-row entry; prefetch addresses never fault. All 32 lanes of an MXFP8 warp
    // reach the `shfl` tree together.
    unsafe {
        let layer=batch.layer;let block=_block_idx_x() as u32;
        let (job,rest)=crate::plain_admission::batch_block(layer.ends,batch.plain,layer.count as usize,block);
        let plain_first=batch.plain&(1<<job)!=0;
        let mut p=layer.projections[0];
        nine!(I,{if job==I {p=layer.projections[I];}});
        let thread=_thread_idx_x() as usize;
        let (rows,cols,tokens)=(p.rows as usize,p.cols as usize,p.tokens as usize);
        if tokens<2 || tokens>4 {core::arch::nvptx::trap();}
        if plain_first {
            let per=rows.div_ceil(128);
            let (first,row)=if SPREAD {(rest as usize/per,(rest as usize%per)*128+thread)} else {(0,rest as usize*128+thread)};
            if row>=rows || first>=tokens {return;}
            let weights=(p.weights as *const u16).add(row*cols).cast::<f32>();
            four!(T,{if (SPREAD && T==0) || (!SPREAD && T<tokens) {
                let token=first+T;
                let sum=crate::plain_warp::row_deep::<true>(weights,(p.input as *const f32).add(token*cols),cols);
                *(p.output as *mut f32).add(token*rows+row)=if batch.round!=0 {rb(sum)} else {sum};
            }});
        } else {
            let index=rest as usize*4+thread/32;if index>=rows {return;}
            // ROWS-MX: the row's chains on the one-token load schedule (`mx_rows_small::wave`, one weight row a warp); the same words as `row`.
            if tokens==2 {crate::mx_rows_small::wave::<2,2,1>(p,index,0);} else {crate::mx_rows_small::wave::<4,2,1>(p,index,0);}
            if thread%32==0 {four!(T,{if T<tokens {let output=(p.output as *mut f32).add(T*rows+index);*output=rb(*output);}});}
        }
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn flash_resident_projection_bf16_rows(batch:crate::next_params::ResidentProjectionBatch) {
    resident_projection_bf16_rows::<true>(batch)
}
/// A lane's eight input operands of one token at element `base`: one 16-byte BF16 word set,
/// or two FP32 word sets (the second is unused for BF16 input).
#[inline(always)]
unsafe fn input8(p: MxProjection, base: usize) -> ([u32; 4], [u32; 4]) {
    if p.input_bf16 != 0 {
        (load16((p.input as *const u16).add(base).cast()), [0; 4])
    } else {
        let x = load16((p.input as *const f32).add(base).cast());
        (x, load16((p.input as *const f32).add(base + 4).cast()))
    }
}
/// row::<16, false, false> for the four output rows r0..r0+3 of one warp: each lane's input
/// words load once per token and feed the four rows. Every (row, token) keeps row()'s element
/// set (tile * 256 + lane * 8 + J), linear bit decoder, ascending FMA chain and shuffle tree.
#[inline(always)]
unsafe fn rows4(p: MxProjection, r0: usize) {
    let lane = _thread_idx_x() as usize & 31;
    let (n, k, m) = (p.rows as usize, p.cols as usize, p.tokens as usize);
    let mut acc = [[0.0f32; 16]; 4];
    let mut tile = 0;
    while tile < k.div_ceil(256) {
        let at = tile * 256 + lane * 8;
        if at < k {
            // All four rows' weight words and scales, and token 0's input words, issue before
            // the first decode; token B+1's input words load before token B's FMAs.
            let mut vw = [[0u32; 4]; 4];
            let mut scs = [0u8; 4];
            four!(Q, {
                let r = (r0 + Q).min(n - 1);
                vw[Q] = load16((p.weights as *const u8).add(r * k + (at & !15)));
                scs[Q] = *(p.scales as *const u8).add(r * k / 32 + at / 32);
            });
            let mut xc = if m > 0 { input8(p, at) } else { ([0u32; 4], [0u32; 4]) };
            let mut ww = [[0.0f32; 8]; 4];
            four!(Q, {
                let v = vw[Q];
                let words = if lane & 1 == 0 { [v[0], v[1]] } else { [v[2], v[3]] };
                let sc = scs[Q];
                let scale = f32::from_bits(if sc == 0 {
                    0x00400000
                } else if sc == 255 {
                    0x7fc00000
                } else {
                    (sc as u32) << 23
                });
                eight!(J, {
                    ww[Q][J] = weight((words[J / 4] >> ((J & 3) * 8)) as u8) * scale;
                });
            });
            sixteen!(B, {
                if B < m {
                    let xn = if B + 1 < m { input8(p, (B + 1) * k + at) } else { xc };
                    let (x, y) = xc;
                    let mut xx = [0.0; 8];
                    if p.input_bf16 != 0 {
                        eight!(J, {
                            xx[J] = f32::from_bits((x[J / 2] >> ((J & 1) * 16)) << 16);
                        });
                    } else {
                        eight!(J, {
                            xx[J] = p.f32_input(if J < 4 { x[J % 4] } else { y[J % 4] });
                        });
                    }
                    four!(Q, {
                        eight!(J, {
                            acc[Q][B] = core::f32::math::mul_add(ww[Q][J], xx[J], acc[Q][B]);
                        });
                    });
                    xc = xn;
                }
            });
        }
        tile += 1;
    }
    four!(Q, {
        sixteen!(B, {
            let mut v = acc[Q][B];
            // Prototype shfl_down tree. Only lane zero is stored.
            let mut d = 16u32;
            while d != 0 {
                let x: f32;
                asm!("shfl.sync.down.b32 {x}, {v}, {d}, 31, -1;",x=out(reg32)x,v=in(reg32)v,d=in(reg32)d,options(nomem,nostack));
                v += x;
                d /= 2;
            }
            if lane == 0 && B < m && r0 + Q < n {
                *(p.output as *mut f32).add(B * n + r0 + Q) = if p.round != 0 { rb(v) } else { v };
            }
        });
    });
}
/// M9..16 general projections with four output rows per warp (16 per CTA): the values of
/// flash_next_mxfp8_bits. Grid: per projection rows.div_ceil(16) blocks (ends) of 128 threads.
/// # Safety
/// Same buffers contract as flash_next_mxfp8_m1; M is 9..16; ends follow the 16-row grid.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_mxfp8_bits_r4(layer: MxLayer) {
    let block = _block_idx_x() as u32;
    let mut p = layer.projections[0];
    let mut start = 0;
    eight!(I, {
        if I + 1 < layer.count as usize && block >= layer.ends[I + 1] {
            p = layer.projections[I + 1];
            start = layer.ends[I + 1];
        }
    });
    let r0 = (block - start) as usize * 16 + _thread_idx_x() as usize / 32 * 4;
    if r0 < p.rows as usize {
        rows4(p, r0);
    }
}
#[cfg(feature = "tensor-cores")]
mod moe_hc {
//! First-party MoE reconcile K16 and compact HC prototype schedules.
//! No M-dependent reduction, split K, activation quantizer, or float atomics.
use core::arch::{asm, nvptx::{_thread_idx_x, _block_idx_x, _block_idx_y, _grid_dim_x}};
use crate::{canonical::{madd, ZERO}, kernel_params::*, ptx};
const H: usize = 2560;
const W: usize = 10240;
const I: usize = 640;
const L: usize = 320;
#[inline(always)]
unsafe fn bf(x: f32) -> u16 { let y: u16; asm!("cvt.rn.bf16.f32 {y}, {x};",y=out(reg16)y,x=in(reg32)x); y }
#[inline(always)]
fn f(x: u16) -> f32 { f32::from_bits((x as u32) << 16) }
#[inline(always)]
unsafe fn sh(x: u32, d: u32, mode: u32) -> u32 {
    let y: u32;
    if mode == 0 { asm!("shfl.sync.down.b32 {y}, {x}, {d}, 31, -1;",y=out(reg32)y,x=in(reg32)x,d=in(reg32)d); }
    else { asm!("shfl.sync.idx.b32 {y}, {x}, {d}, 31, -1;",y=out(reg32)y,x=in(reg32)x,d=in(reg32)d); }
    y
}
#[inline(always)]
unsafe fn sum(mut x: f32) -> f32 { for d in [16,8,4,2,1] { x += f32::from_bits(sh(x.to_bits(),d,0)); } f32::from_bits(sh(x.to_bits(),0,1)) }
// Preserve expf's exponent reconstruction and low correction, as emitted by
// the frozen prototype. A bare ex2(x*log2e) changes its output bytes.
#[inline(always)]
unsafe fn exp_parts(x: f32) -> (f32, f32) {
    let (a,b): (f32,f32);
    asm!("{{ .reg .f32 t,n,r; .reg .b32 bits;\n\
        fma.rn.f32 t, {x}, 0f3BBB989D, 0f3F000000;\n\
        cvt.sat.f32.f32 t, t;\n\
        fma.rm.f32 n, t, 0f437C0000, 0f4B400001;\n\
        add.f32 r, n, 0fCB40007F; neg.f32 r, r;\n\
        fma.rn.f32 r, {x}, 0f3FB8AA3B, r;\n\
        fma.rn.f32 r, {x}, 0f32A57060, r;\n\
        mov.b32 bits, n; shl.b32 bits, bits, 23; mov.b32 {b}, bits;\n\
        ex2.approx.ftz.f32 {a}, r; }}",x=in(reg32)x,a=out(reg32)a,b=out(reg32)b);
    (a,b)
}
#[inline(always)]
unsafe fn exp(x: f32) -> f32 { let (a,b)=exp_parts(x); a*b }
#[inline(always)]
unsafe fn sig(x: f32) -> f32 { 1.0/(1.0+exp(-x)) }
#[inline(always)]
unsafe fn swiglu(g: f32, u: f32) -> u16 { let (a,b)=exp_parts(-g); bf((g/core::f32::math::mul_add(a,b,1.0))*u) }
#[inline(always)]
unsafe fn weight(p: u64, r: usize, c: usize, n: usize, k: usize, packed: u32) -> u16 {
    *(p as *const u16).add(if packed != 0 && flash_bf16_packed(n,k) { flash_bf16_element(r,c,k,n) } else { r*k+c })
}
#[inline(always)]
unsafe fn wp(p: u64, r: usize, c: usize, n: usize, k: usize, packed: u32) -> u32 {
    weight(p,r,c,n,k,packed) as u32 | ((weight(p,r,c+1,n,k,packed) as u32)<<16)
}
/// Transposed register MMA: CN=8 router, CN=16 shared; one warp/CTA.
/// Rounded outputs are BF16 represented in FP32 storage.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_dense(p: NextDense) {
    let lane=_thread_idx_x() as usize; let (g,t)=(lane/4,lane%4);
    let (n,k,m)=(p.rows as usize,p.cols as usize,p.tokens as usize);
    let cn=if p.rounded==0 {8} else {16}; let mut task=_block_idx_x() as usize;
    // One-token router rows (CN 8, packed): the same ascending K16 MMA chain,
    // but a batch's weight words (columns c, c+1 share one aligned word) and
    // row-0 inputs load before its MMAs; lanes of absent rows keep B = 0.
    let router=p.rounded==0 && p.weight2==0 && m==1 && p.packed!=0 && flash_bf16_packed(n,k);
    while task < m.div_ceil(8)*(n/cn) {
        let (token,nc)=(task/(n/cn)*8,task%(n/cn)*cn); let mut d=ZERO;
        if router {let(wr,x0)=(p.weight as *const u16,p.input as *const f32);
            for kb in (0..k).step_by(128) {let(mut a,mut c,mut x)=([0u32;8],[0u32;8],[0.0f32;32]);let mut j=0;
                while j<8 {let kk=kb+j*16+t*2;a[j]=*(wr.add(flash_bf16_element(nc+g,kk,k,n)) as *const u32);c[j]=*(wr.add(flash_bf16_element(nc+g,kk+8,k,n)) as *const u32);j+=1;}
                j=0;while j<8 {let kk=kb+j*16+t*2;x[j*4]=*x0.add(kk);x[j*4+1]=*x0.add(kk+1);x[j*4+2]=*x0.add(kk+8);x[j*4+3]=*x0.add(kk+9);j+=1;}
                j=0;while j<8 {let(b0,b1)=if g==0 {(bf(x[j*4]) as u32|((bf(x[j*4+1]) as u32)<<16),bf(x[j*4+2]) as u32|((bf(x[j*4+3]) as u32)<<16))}else{(0,0)};
                    d=madd([a[j],0,c[j],0],[b0,b1],d);j+=1;}}
        } else {
        for kk in (0..k).step_by(16) {
            let pair=|r: usize,c: usize| { if r<m { let x=(p.input as *const f32).add(r*k+c); bf(*x) as u32|((bf(*x.add(1)) as u32)<<16) } else {0} };
            let w=|r: usize,c: usize| { if r<n { if p.weight2!=0 && r>=n/2 {wp(p.weight2,r-n/2,c,n/2,k,p.packed)} else {wp(p.weight,r,c,if p.weight2==0 {n}else{n/2},k,p.packed)} } else {0} };
            d=madd([w(nc+g,kk+t*2),if g+8<cn {w(nc+g+8,kk+t*2)}else{0},w(nc+g,kk+t*2+8),if g+8<cn {w(nc+g+8,kk+t*2+8)}else{0}],
                [pair(token+g,kk+t*2),pair(token+g,kk+t*2+8)],d);
        }}
        for (r,v0,v1) in [(token+t*2,d.a,d.c),(token+t*2+1,d.b,d.d)] {
            if r<m { let y=(p.output as *mut f32).add(r*n+nc+g); *y=if p.rounded==0 {v0}else{f(bf(v0))}; if cn==16 {*y.add(8)=f(bf(v1));} }
        }
        task+=_grid_dim_x() as usize;
    }
}
/// Router prototype's parallel full softmax and stable bitonic selection.
/// The preceding projection remains K16, including prompt rows. Rounded
/// probabilities and renormalization are NOT algebraically cancelled.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_route(p: NextMoe) {
    let tid=_thread_idx_x() as usize;let row=_block_idx_x() as usize;
    if row>=p.tokens as usize {return;}
    let s=ptx::shared::<4112>();let logits=(p.logits as *const f32).add(row*512);
    if tid<32 {let mut hi=f32::NEG_INFINITY;let mut bad=0.0f32;for j in 0..16 {let v=*logits.add(tid+j*32);hi=hi.max(v);if !v.is_finite(){bad=1.0;}}
        hi=crate::layer::maximum(hi);bad=crate::layer::maximum(bad);if tid==0 {ptx::store(s+4096,hi);ptx::store(s+4104,bad);}}
    ptx::sync();let hi=ptx::load(s+4096);
    for i in (tid..512).step_by(256) {ptx::store(s+i as u32*4,crate::flash_moe::exp(*logits.add(i)-hi));}
    ptx::sync();if tid<32 {let mut den=0.0;for j in 0..16 {den+=ptx::load(s+(tid+j*32) as u32*4);}den=crate::layer::sum(den);if tid==0 {ptx::store(s+4100,den);}}
    ptx::sync();let den=ptx::load(s+4100);
    if ptx::load(s+4104)!=0.0 || !den.is_finite() || den<=0.0 {
        if tid<10 {*(p.ids as *mut u32).add(row*10+tid)=u32::MAX;*(p.weights as *mut f32).add(row*10+tid)=f32::NAN;}return;
    }
    for i in (tid..512).step_by(256) {ptx::store(s+i as u32*4,ptx::load(s+i as u32*4)/den);ptx::store(s+2048+i as u32*4,f32::from_bits(i as u32));}
    ptx::sync();let mut k=2;while k<=512 {let mut delta=k/2;while delta!=0 {
        for i in (tid..512).step_by(256) {let other=i^delta;if other>i {
            let a=ptx::load(s+i as u32*4);let b=ptx::load(s+other as u32*4);let ai=ptx::load(s+2048+i as u32*4).to_bits();let bi=ptx::load(s+2048+other as u32*4).to_bits();
            if ((a>b)||(a==b&&ai<bi)) != (i&k==0) {ptx::store(s+i as u32*4,b);ptx::store(s+other as u32*4,a);ptx::store(s+2048+i as u32*4,f32::from_bits(bi));ptx::store(s+2048+other as u32*4,f32::from_bits(ai));}
        }}ptx::sync();delta/=2;
    }k*=2;}
    let mut total=0.0;for j in 0..10 {total+=ptx::load(s+j*4);}
    if tid<10 {*(p.ids as *mut u32).add(row*10+tid)=ptx::load(s+2048+tid as u32*4).to_bits();*(p.weights as *mut f32).add(row*10+tid)=ptx::load(s+tid as u32*4)/total;}
}
/// Pack selected token rows to BF16 for rank-local expert tiles.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gather(p: NextMoe) {
    let j=crate::index();if j>=p.tokens as usize*10*H {return;}
    let row=(*(p.rows as *const u32).add(j/H) as usize).min(p.tokens as usize-1);
    *(p.gathered as *mut u16).add(j)=bf(*(p.input as *const f32).add(row*H+j%H));
}
/// Rank prompt-chunk MoE fold (one thread per row and BF16 pair word): row r's owned expert
/// rows in its plan order (ascending expert id), each value f(bf16) * weight then added, both
/// rounded to nearest with no FMA, from 0: the host fold of next_chunk::moe_rows_chunk.
/// p.tiles = plan (per row 21 words: count, then count (slot, weight bits) pairs), p.down = the
/// fp4_bw_down output slots (H words each, the first H/2 used), p.combined = n x H F32 output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_bw_fold_rows(p: NextMoe) {
    let j=crate::index();if j>=p.tokens as usize*H/2 {return;}let(r,w)=(j/(H/2),j%(H/2));
    let plan=(p.tiles as *const u32).add(r*21);let count=(*plan).min(10) as usize;let(mut a,mut b)=(0.0f32,0.0f32);let mut q=0;
    while q<count {let slot=*plan.add(1+2*q) as usize;let wt=f32::from_bits(*plan.add(2+2*q));let word=*(p.down as *const u32).add(slot*H+w);
        let(x,y):(f32,f32);asm!("mul.rn.f32 {x}, {u}, {wt}; mul.rn.f32 {y}, {v}, {wt}; add.rn.f32 {a}, {a}, {x}; add.rn.f32 {b}, {b}, {y};",
            x=out(reg32)x,y=out(reg32)y,u=in(reg32)f(word as u16),v=in(reg32)f((word>>16) as u16),wt=in(reg32)wt,a=inout(reg32)a,b=inout(reg32)b,options(pure,nomem,nostack));
        let _=(x,y);q+=1;}
    let o=(p.combined as *mut f32).add(r*H+2*w);*o=a;*o.add(1)=b;
}
#[inline(always)]
fn ix(r: usize,c: usize)->usize {r*64+(c^((r&15)*4))}
#[inline(always)]
unsafe fn prefetch(p: NextRouted, s: u32, buf: usize, kk: usize, base: usize, count: usize, nc: usize, expert: NextExpert) {
    let tid=_thread_idx_x() as usize; let k=p.cols as usize;
    let ah=s+buf as u32*4096;let raw=s+8192+buf as u32*4096;let sc=s+16384+buf as u32*512;
    let(r,c)=(tid/16,tid%16*8);let dst=ah+(ix(r,c/2)*4) as u32;
    if r<count {ptx::copy16(dst,(p.input as *const u16).add((base+r)*k+kk+c).cast());}
    else {for q in 0..4 {ptx::store(dst+q*4,0.0);}}
    let(r,c)=(tid/4,tid%4*16);let row=nc+r;let matrix=expert.matrices[if p.down!=0 {2}else{usize::from(row>=I)}];let local=if p.down!=0 {row}else{row%I};ptx::copy16(raw+(r*64+c) as u32,(matrix.weight as *const u8).add(local*k/2+kk/2+c).cast());
    if tid<64 {let row=nc+tid;let matrix=expert.matrices[if p.down!=0 {2}else{usize::from(row>=I)}];let local=if p.down!=0 {row}else{row%I};
        asm!("cp.async.ca.shared.global [{d}], [{src}], 8;",d=in(reg32)(sc+tid as u32*8),src=in(reg64)(matrix.scales as *const u8).add(local*k/16+kk/16));}
    ptx::commit();
}
#[inline(always)]
unsafe fn decode(p: NextRouted,s: u32,buf: usize,nc: usize,expert: NextExpert) {
    let tid=_thread_idx_x() as usize;let raw=s+8192+buf as u32*4096;let sc=s+16384+buf as u32*512;let bh=s+17408;
    for i in (tid..1024).step_by(256) {let(r,c)=(i/16,i%16*8);let v=ptx::load(raw+(r*64+c/2) as u32).to_bits();let scale=ptx::load_u16((sc+(r*8+c/16) as u32)&!1);let code=((scale>>((c/16%2)*8))&255) as u8;
        let matrix=expert.matrices[if p.down!=0 {2}else{usize::from(nc+r>=I)}];
        for j in 0..4 {let q=v>>(j*8);let pair=crate::fp4_experts::next_pair(q,crate::fp8_bf16::decode(code),matrix.global);ptx::store(bh+(ix(r,c/2+j)*4) as u32,f32::from_bits(pair));}
    }
}
// Same-row lookahead, independent of routing or accumulation order. Packed
// weights already enter shared memory through aligned 16-byte cp.async loads.
#[inline(always)]
unsafe fn lookahead(p: NextRouted, kk: usize, nc: usize, expert: NextExpert) {
    let tid=_thread_idx_x() as usize;let k=p.cols as usize;
    if tid%4==0 {let row=nc+tid/4;let matrix=expert.matrices[if p.down!=0 {2}else{usize::from(row>=I)}];
        let local=if p.down!=0 {row}else{row%I};
        for ahead in [128,256] {if kk+ahead<k {
            asm!("prefetch.global.L2 [{w}]; prefetch.global.L2 [{s}];",
                w=in(reg64)(matrix.weight as *const u8).add(local*k/2+(kk+ahead)/2),
                s=in(reg64)(matrix.scales as *const u8).add(local*k/16+(kk+ahead)/16));
        }}
    }
}
#[inline(always)]
unsafe fn routed<const PREFETCH: bool, const SWAP: bool>(p: NextRouted) {
    let tid=_thread_idx_x() as usize;let(lane,warp)=(tid%32,tid/32);let(g,t)=(lane/4,lane%4);
    let s=ptx::shared::<33792>();let nt=*(p.tile_count as *const u32) as usize;let n=p.rows as usize;let k=p.cols as usize;
    let mut task=_block_idx_x() as usize;
    while task<nt*(n/64) {let tile=(p.tiles as *const u32).add(task/(n/64)*3);let expert=*(p.experts as *const NextExpert).add(*tile as usize);let base=*tile.add(1) as usize;let count=*tile.add(2) as usize;let nc=task%(n/64)*64;let mut d=ZERO;
        prefetch(p,s,0,0,base,count,nc,expert);ptx::wait();ptx::sync();decode(p,s,0,nc,expert);ptx::sync();
        for kk in (0..k).step_by(128) {let buf=kk/128%2;if PREFETCH {lookahead(p,kk,nc,expert);}if kk+128<k {prefetch(p,s,buf^1,kk+128,base,count,nc,expert);}
            for kt in (0..128).step_by(16) {
                // Four output tiles, two n8 token slices. The operand words
                // and ascending K16 instructions are unchanged per element.
                if SWAP && warp/4*8>=count {continue;}
                let (a0,a1,a2,a3,b0,b1):(u32,u32,u32,u32,u32,u32);
                let aa=if SWAP {s+17408+(ix(warp%4*16+lane%16,kt/2+lane/16*4)*4) as u32}
                    else {s+buf as u32*4096+(ix(lane%16,kt/2+lane/16*4)*4) as u32};
                let ba=if SWAP {s+buf as u32*4096+(ix(warp/4*8+lane%8,kt/2+(lane/8%2)*4)*4) as u32}
                    else {s+17408+(ix(warp*8+lane%8,kt/2+(lane/8%2)*4)*4) as u32};
                asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a0},{a1},{a2},{a3}}}, [{p}];",
                    a0=out(reg32)a0,a1=out(reg32)a1,a2=out(reg32)a2,a3=out(reg32)a3,p=in(reg32)aa);
                asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{b0},{b1}}}, [{p}];",b0=out(reg32)b0,b1=out(reg32)b1,p=in(reg32)ba);
                d=madd([a0,a1,a2,a3],[b0,b1],d);
            }
            ptx::sync();if kk+128<k {ptx::wait();ptx::sync();decode(p,s,buf^1,nc,expert);ptx::sync();}
        }
        for (r,a,b) in [(g,d.a,d.b),(g+8,d.c,d.d)] {
            if SWAP {let token=warp/4*8+t*2;let col=nc+warp%4*16+r;
                if token<count {*(p.output as *mut u16).add((base+token)*n+col)=bf(a);}
                if token+1<count {*(p.output as *mut u16).add((base+token+1)*n+col)=bf(b);}
            }else if r<count {let y=(p.output as *mut u16).add((base+r)*n+nc+warp*8+t*2);*y=bf(a);*y.add(1)=bf(b);}
        }
        ptx::sync();task+=_grid_dim_x() as usize;
    }
}
/// M16 N64 K128, eight warps, double-buffered 16-byte weight loads.
/// reserved selects 0 baseline, 1 prefetch (A), 2 swap-AB (B), 3 both (C).
/// Every schedule retains the base's ascending K16 chain and BF16 boundaries.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_routed(p: NextRouted) {
    match p.reserved {1=>routed::<true,false>(p),2=>routed::<false,true>(p),
        3=>routed::<true,true>(p),_=>routed::<false,false>(p)}
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_routed_activate(p: NextActivation) {
    let j=crate::index();if j>=p.rows as usize*I {return;}let(r,c)=(j/I,j%I);let x=(p.input as *const u16).add(r*2*I+c);
    *(p.output as *mut u16).add(j)=swiglu(f(*x),f(*x.add(I)));
}

#[inline(always)]
unsafe fn combined(p: NextHc,t: usize,i: usize,s: u32)->f32 {let x=*(p.input as *const f32).add(t*W+i);if p.block==0 {x}else{x+ptx::load(s+20480+(i/H*4) as u32)* *(p.block as *const f32).add(t*H+i%H)}}
/// Rank rows only: the p.block==0 norm phase of hc_pre, once per token. The same
/// per-lane ascending sums/butterfly, rsqrt, F32 norm and RN BF16 operands are kept.
/// p.block is an exclusive tokens*W BF16 scratch row here (not a pending residual).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_norm_once(p: NextHc) {
    let tid=_thread_idx_x() as usize;let(l,w)=(tid%32,tid/32);let t=_block_idx_y() as usize;let s=ptx::shared::<16>();
    let input=(p.input as *const f32).add(t*W);let nf32=p.norm_f32!=0;
    if w<4 {let mut z=0.0;let x0=input.add(w*H+l);
        for q in (0..H/32).step_by(16) {let mut x=[0.0f32;16];let mut j=0;
            while j<16 {x[j]=*x0.add((q+j)*32);j+=1;}
            j=0;while j<16 {z+=x[j]*x[j];j+=1;}}
        let v=sum(z)/H as f32+1e-6;let inv:f32;asm!("rsqrt.approx.ftz.f32 {y}, {x};",y=out(reg32)inv,x=in(reg32)v);if l==0 {ptx::store(s+w as u32*4,inv);}}ptx::sync();
    for q in (0..W/256).step_by(8) {let(mut x,mut v,mut raw)=([0.0f32;8],[0.0f32;8],[0u32;8]);let mut j=0;
        while j<8 {let i=tid+(q+j)*256;x[j]=*input.add(i);v[j]=ptx::load(s+(i/H*4) as u32);j+=1;}
        j=0;if nf32 {while j<8 {raw[j]=*(p.weight as *const u32).add(tid+(q+j)*256);j+=1;}}else{while j<8 {raw[j]=*(p.weight as *const u16).add(tid+(q+j)*256) as u32;j+=1;}}
        j=0;while j<8 {let i=tid+(q+j)*256;let nw=if nf32 {f32::from_bits(raw[j])}else{f(raw[j] as u16)};let n=(x[j]*v[j])*(1.0+nw);
            *(p.state as *mut f32).add(t*W+i)=x[j];*(p.normalized as *mut f32).add(t*W+i)=n;*(p.block as *mut u16).add(t*W+i)=bf(n);j+=1;}
    }
}
/// Rank rows only: the original hc_pre down/inject chain, with its RN BF16 row
/// from hc_norm_once in p.block. Grid and every dot lane's element order stay fixed.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_pre_normed(p: NextHc) {
    let tid=_thread_idx_x() as usize;let(l,w)=(tid%32,tid/32);let t=_block_idx_y() as usize;let b=_block_idx_x() as usize;let s=ptx::shared::<20480>();
    let input=(p.block as *const u32).add(t*W/2);for i in (tid..W/2).step_by(256) {let v=*input.add(i);asm!("st.shared.b32 [{a}], {v};",a=in(reg32)(s+i as u32*4),v=in(reg32)v);}ptx::sync();
    let r=b*8+w;if r<L+if p.injection==0 {0}else{4} {let(wt,row,n)=if r<L {(p.down,r,L)}else{(p.injection,r-L,4)};
        // Element k = l + 32j sits at (j even ? e0 : e1) + j/2 * step for the
        // tiled, 4-row and plain layouts alike (W = 10240): same addresses.
        let packed=p.packed!=0 && flash_bf16_packed(n,W);let at=|k: usize| if packed {flash_bf16_element(row,k,W,n)}else{row*W+k};
        let(e0,e1)=(at(l),at(l+32));let step=at(l+64)-e0;let(w0,w1)=((wt as *const u16).add(e0),(wt as *const u16).add(e1));let mut z=0.0;
        for q in (0..W/32).step_by(16) {let(mut a,mut c)=([0u16;16],[0u32;16]);let mut j=0;
            while j<16 {a[j]=*w0.add((q/2+j/2)*step);a[j+1]=*w1.add((q/2+j/2)*step);j+=2;}
            j=0;while j<16 {c[j]=ptx::load_u16(s+(l+(q+j)*32) as u32*2);j+=1;}
            j=0;while j<16 {z=core::f32::math::mul_add(f(a[j]),f(c[j] as u16),z);j+=1;}}
        let z=f(bf(sum(z)));if l==0 {if r<L {let x=z*0.25;*(p.low as *mut u16).add(t*L+r)=bf(x*sig(x));}else{*(p.gates as *mut f32).add(t*4+r-L)=z;}}}
}

/// Compact pre: pending residual, F32 norms, down/inject and SiLU.
/// input/state MUST differ: other CTAs read input while CTA 0 exports state.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_pre(p: NextHc) {
    let tid=_thread_idx_x() as usize;let t=_block_idx_y() as usize;let s=ptx::shared::<20512>();
    if tid<4 {ptx::store(s+20480+tid as u32*4,if p.block==0 {0.0}else{2.0*sig(*(p.previous as *const f32).add(t*4+tid)*0.25)});}ptx::sync();
    if p.down==0 {for i in (tid..W).step_by(256) {*(p.state as *mut f32).add(t*W+i)=combined(p,t,i,s);}return;}
    let(l,w)=(tid%32,tid/32);let b=_block_idx_x() as usize;let blk=p.block!=0;let nf32=p.norm_f32!=0;
    // Batched form of the prototype's three loops: every lane/thread keeps its
    // element set and ascending order (z += x*x, (state*inv)*(1+nw), the
    // K-strided FMA chain, the same butterfly), but a batch's global and shared
    // loads issue before its first use: no memory latency per step.
    let input=(p.input as *const f32).add(t*W);let block=(if blk {p.block}else{p.input}) as *const f32;let block=block.add(t*H);
    if w<4 {let mut z=0.0;let g=if blk {ptx::load(s+20480+w as u32*4)}else{0.0};let x0=input.add(w*H+l);let y0=block.add(l);
        for q in (0..H/32).step_by(16) {let(mut x,mut y)=([0.0f32;16],[0.0f32;16]);let mut j=0;
            while j<16 {x[j]=*x0.add((q+j)*32);j+=1;}
            if blk {j=0;while j<16 {y[j]=*y0.add((q+j)*32);j+=1;}}
            j=0;while j<16 {let v=if blk {x[j]+g*y[j]}else{x[j]};z+=v*v;j+=1;}}
        let v=sum(z)/H as f32+1e-6;let inv:f32;asm!("rsqrt.approx.ftz.f32 {y}, {x};",y=out(reg32)inv,x=in(reg32)v);if l==0 {ptx::store(s+20496+w as u32*4,inv);}}ptx::sync();
    for q in (0..W/256).step_by(8) {let(mut x,mut y,mut g,mut v,mut raw)=([0.0f32;8],[0.0f32;8],[0.0f32;8],[0.0f32;8],[0u32;8]);let mut j=0;
        while j<8 {let i=tid+(q+j)*256;x[j]=*input.add(i);v[j]=ptx::load(s+20496+(i/H*4) as u32);j+=1;}
        if blk {j=0;while j<8 {let i=tid+(q+j)*256;y[j]=*block.add(i%H);g[j]=ptx::load(s+20480+(i/H*4) as u32);j+=1;}}
        j=0;if nf32 {while j<8 {raw[j]=*(p.weight as *const u32).add(tid+(q+j)*256);j+=1;}}else{while j<8 {raw[j]=*(p.weight as *const u16).add(tid+(q+j)*256) as u32;j+=1;}}
        let(mut state,mut n)=([0.0f32;8],[0.0f32;8]);j=0;
        while j<8 {let i=tid+(q+j)*256;state[j]=if blk {x[j]+g[j]*y[j]}else{x[j]};let nw=if nf32 {core::mem::transmute::<u32,f32>(raw[j])}else{f(raw[j] as u16)};
            n[j]=(state[j]*v[j])*(1.0+nw);let bits=bf(n[j]);asm!("st.shared.b16 [{a}], {v};",a=in(reg32)(s+i as u32*2),v=in(reg16)bits);j+=1;}
        if b==0 {j=0;while j<8 {let i=tid+(q+j)*256;*(p.state as *mut f32).add(t*W+i)=state[j];*(p.normalized as *mut f32).add(t*W+i)=n[j];j+=1;}}
    }
    ptx::sync();
    let r=b*8+w;if r<L+if p.injection==0 {0}else{4} {let(wt,row,n)=if r<L {(p.down,r,L)}else{(p.injection,r-L,4)};
        // Element k = l + 32j sits at (j even ? e0 : e1) + j/2 * step for the
        // tiled, 4-row and plain layouts alike (W = 10240): same addresses.
        let packed=p.packed!=0 && flash_bf16_packed(n,W);let at=|k: usize| if packed {flash_bf16_element(row,k,W,n)}else{row*W+k};
        let(e0,e1)=(at(l),at(l+32));let step=at(l+64)-e0;let(w0,w1)=((wt as *const u16).add(e0),(wt as *const u16).add(e1));let mut z=0.0;
        for q in (0..W/32).step_by(16) {let(mut a,mut c)=([0u16;16],[0u32;16]);let mut j=0;
            while j<16 {a[j]=*w0.add((q/2+j/2)*step);a[j+1]=*w1.add((q/2+j/2)*step);j+=2;}
            j=0;while j<16 {c[j]=ptx::load_u16(s+(l+(q+j)*32) as u32*2);j+=1;}
            j=0;while j<16 {z=core::f32::math::mul_add(f(a[j]),f(c[j] as u16),z);j+=1;}}
        let z=f(bf(sum(z)));if l==0 {if r<L {let x=z*0.25;*(p.low as *mut u16).add(t*L+r)=bf(x*sig(x));}else{*(p.gates as *mut f32).add(t*4+r-L)=z;}}}
}
/// Up + four-stream sigmoid mean, same lane-strided reduction at every M.
/// Rank rows always carry packed BF16 weights. Specialize only their address path;
/// the lane-strided FMA chains, BF16 sum rounding and sigmoid-mean order stay hc_up's.
/// Pure storage copy for rank prompt rows: packed BF16 up weights -> row-major BF16.
/// No floating point operation. p.previous is the exclusive W*L BF16 scratch buffer.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_unpack_up(p: NextHc) {
    let j=crate::index();if j<W*L {*(p.previous as *mut u16).add(j)=weight(p.up,j/L,j%L,W,L,p.packed);}
}
/// Same hc_up FMA/sum/mean plan, reading the copied BF16 words at row-major addresses.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_rowmajor(p: NextHc) {
    let tid=_thread_idx_x() as usize;let(l,w)=(tid%32,tid/32);let t=_block_idx_y() as usize;let i=_block_idx_x() as usize*8+w;if i>=H {return;}
    let mut z=[0.0;4];for k in (l..L).step_by(32) {let x=f(*(p.low as *const u16).add(t*L+k));for g in 0..4 {z[g]=core::f32::math::mul_add(f(*(p.previous as *const u16).add((g*H+i)*L+k)),x,z[g]);}}
    for g in 0..4 {z[g]=f(bf(sum(z[g])));}if l==0 {let mut v=0.0;for g in 0..4 {v+=*(p.normalized as *const f32).add(t*W+g*H+i)*sig(z[g]);}*(p.output as *mut f32).add(t*H+i)=f(bf(v*0.25));}
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_packed(p: NextHc) {
    let tid=_thread_idx_x() as usize;let(l,w)=(tid%32,tid/32);let t=_block_idx_y() as usize;let i=_block_idx_x() as usize*8+w;if i>=H {return;}
    let mut z=[0.0;4];for k in (l..L).step_by(32) {let x=f(*(p.low as *const u16).add(t*L+k));for g in 0..4 {z[g]=core::f32::math::mul_add(f(weight(p.up,g*H+i,k,W,L,1)),x,z[g]);}}
    for g in 0..4 {z[g]=f(bf(sum(z[g])));}if l==0 {let mut v=0.0;for g in 0..4 {v+=*(p.normalized as *const f32).add(t*W+g*H+i)*sig(z[g]);}*(p.output as *mut f32).add(t*H+i)=f(bf(v*0.25));}
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up(p: NextHc) {
    let tid=_thread_idx_x() as usize;let(l,w)=(tid%32,tid/32);let t=_block_idx_y() as usize;let i=_block_idx_x() as usize*8+w;if i>=H {return;}
    let mut z=[0.0;4];for k in (l..L).step_by(32) {let x=f(*(p.low as *const u16).add(t*L+k));for g in 0..4 {z[g]=core::f32::math::mul_add(f(weight(p.up,g*H+i,k,W,L,p.packed)),x,z[g]);}}
    for g in 0..4 {z[g]=f(bf(sum(z[g])));}if l==0 {let mut v=0.0;for g in 0..4 {v+=*(p.normalized as *const f32).add(t*W+g*H+i)*sig(z[g]);}*(p.output as *mut f32).add(t*H+i)=f(bf(v*0.25));}
}
}
