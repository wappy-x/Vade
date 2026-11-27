//! FP8-only prompt Hyper-Connection & vectorized RDMA copy/add kernels:
//! - `flash_fp8_hc_add_norm_bf16`: 4 KiB shared-mem (6 CTAs/SM) BF16-norm + optional fused HC add
//!   (both single-buffer `attn.combine` and rank-split `mlp.combine_columns` via `peer` + `rank`).
//! - `flash_fp8_hc_down_fused`: on-read FP8 HC-down + HC-inject directly from FP32 residual + scales
//!   + BF16 norm weight, fusing exact `flash_ops::silu` + `split2_pair` into K=320 planes (eliminating
//!   168 MB packed write + 504 MB 3x read + `flash_bf16_input`).
//! - `flash_fp8_hc_up_tiled`: 4-slice fused HC-up GEMM + exact FP8 4-stream mix (overlapping next
//!   slice's 2 stages behind current slice's low-register epilogue).
//! - `flash_fp8_copy_v4` / `flash_fp8_add_v4`: 16-byte coalesced RDMA mailbox receive helpers.
use crate::canonical::{self, add, madd, Acc, ZERO};
use crate::kernel_params::{Bf16Rows, FlashHcAddNorm};
use crate::layer::{exp, sqrt, sum};
use crate::math::bf16_to_f32 as bf;
use crate::ptx;
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x},
};

const H: usize = 2560;
const W: usize = 10240;
const PLANE: usize = 8192;
const STAGE: usize = 32768;
const ROWS_A: usize = STAGE;
const ROWS_W: usize = STAGE + 16384;
const ROWS_STAGE: usize = ROWS_W + 256; // 49408 B x 2 = 98816 B == FLASH_BF16_PREFILL_ROWS_SHARED

macro_rules! unroll {
    (2, $i:ident, $body:block) => {
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 1; $body }
    };
    (4, $i:ident, $body:block) => {
        unroll!(2, $i, $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 2; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 3; $body }
    };
}

/// FP8 prompt HC norm (+ optional fused HC add when `p.block != 0`, supporting both full-width
/// `right_half == 0` and 2-half column-split `right_half != 0` with `p.block = left_half`).
/// Writes per-(token, stream) RMS scales into `p.scales` (`[rows][4]`) using only 4 KiB shared
/// memory (512 B per warp -> 6 CTAs/SM = 48 warps/SM) while preserving the exact `hc_norm_prompt`
/// per-lane square order `(((sq + a*a) + b*b) + c*c) + d*d` and exact `flash_hc_add` arithmetic.
/// Grid: `(1, ceil(rows/2), 1)`, block: `256` (8 warps = 2 tokens x 4 streams), dynamic shared: `4096`.
// SAFETY: Caller validates `p.residual` [rows][10240], `p.scales` [rows][4], optional `p.block`
// (`[rows][2560]` if `right_half == 0`, else `[rows][1280]` alongside `right_half` `[rows][1280]`)
// and `p.gates` `[rows][4]`, with grid `(1, ceil(rows/2), 1)`, block `256`, shared `4096`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_fp8_hc_add_norm_bf16(
    p: FlashHcAddNorm,
    rows: u32,
    right_half: u64,
) {
    let rows = rows as usize;
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let token = _block_idx_y() as usize * 2 + warp / 4;
    let stream = warp % 4;
    if token >= rows {
        return;
    }
    let residual = (p.residual as *mut f32).add(token * W + stream * H);
    let sh = (warp * 512) as u32;
    let mut squares = 0.0f32;

    if p.block != 0 {
        let gate = *(p.gates as *const f32).add(token * 4 + stream);
        let factor = 2.0 * (1.0 / (1.0 + exp(-(gate * 0.25))));
        let left = p.block as *const f32;
        let right = right_half as *const f32;
        let half_h = H / 2;
        let mut base = 0usize;
        while base < H {
            let idx = base + lane * 4;
            let (r0, r1, r2, r3): (f32, f32, f32, f32);
            let (b0, b1, b2, b3): (f32, f32, f32, f32);
            asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                a = out(reg32) r0, b = out(reg32) r1, c = out(reg32) r2, d = out(reg32) r3,
                p = in(reg64) residual.add(idx));
            let blk_ptr = if right_half == 0 {
                left.add(token * H + idx)
            } else if base < half_h {
                left.add(token * half_h + idx)
            } else {
                right.add(token * half_h + (idx - half_h))
            };
            asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                a = out(reg32) b0, b = out(reg32) b1, c = out(reg32) b2, d = out(reg32) b3,
                p = in(reg64) blk_ptr, options(readonly, nostack));
            let v0 = r0 + factor * b0;
            let v1 = r1 + factor * b1;
            let v2 = r2 + factor * b2;
            let v3 = r3 + factor * b3;
            asm!("st.global.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};",
                p = in(reg64) residual.add(idx),
                a = in(reg32) v0, b = in(reg32) v1, c = in(reg32) v2, d = in(reg32) v3);
            asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};",
                p = in(reg32) sh + (lane * 16) as u32,
                a = in(reg32) v0, b = in(reg32) v1, c = in(reg32) v2, d = in(reg32) v3);
            ptx::warp_sync();
            let a = ptx::load(sh + (lane * 4) as u32);
            let b_v = ptx::load(sh + ((lane + 32) * 4) as u32);
            let c = ptx::load(sh + ((lane + 64) * 4) as u32);
            let d = ptx::load(sh + ((lane + 96) * 4) as u32);
            squares = (((squares + a * a) + b_v * b_v) + c * c) + d * d;
            base += 128;
        }
    } else {
        let mut base = 0usize;
        while base < H {
            let idx = base + lane * 4;
            let (v0, v1, v2, v3): (f32, f32, f32, f32);
            asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                a = out(reg32) v0, b = out(reg32) v1, c = out(reg32) v2, d = out(reg32) v3,
                p = in(reg64) residual.add(idx), options(readonly, nostack));
            asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};",
                p = in(reg32) sh + (lane * 16) as u32,
                a = in(reg32) v0, b = in(reg32) v1, c = in(reg32) v2, d = in(reg32) v3);
            ptx::warp_sync();
            let a = ptx::load(sh + (lane * 4) as u32);
            let b_v = ptx::load(sh + ((lane + 32) * 4) as u32);
            let c = ptx::load(sh + ((lane + 64) * 4) as u32);
            let d = ptx::load(sh + ((lane + 96) * 4) as u32);
            squares = (((squares + a * a) + b_v * b_v) + c * c) + d * d;
            base += 128;
        }
    }
    let scale = 1.0 / sqrt(sum(squares) / H as f32 + 1e-6);
    if lane == 0 && p.scales != 0 {
        *(p.scales as *mut f32).add(token * 4 + stream) = scale;
    }
}

/// On-read FP8 prompt HC-down (`N = 320`) + optional HC-inject (`N2 = 4`, `K = 10240`) directly
/// from FP32 `residual` + `[tokens][4]` RMS `scales` + `[10240]` BF16 `norm_bf16`.
/// Applies exact FP8 `flash_ops::silu(x, 1.0, 4.0, 0)` + `split2_pair` directly into `p.output`
/// (packed K=320 hi/lo planes for `flash_fp8_hc_up_tiled`, zero-padded past `tokens`), and writes
/// FP32 `hc_inject` rows to `p.output2`. Supports arbitrary `tokens >= 64` (including non-multiples
/// of 16). Uses `group = 4` L2 swizzle so `residual` stays hot in 24 MiB L2 across the 3 N-tiles.
/// Grid: `(ceil(tokens/64), ceil((rows + rows2)/128), 1)`, block: `256`, shared: `98816`.
// SAFETY: Caller validates `p` (`input = residual`, `weight = down`, `weight2 = inject`,
// `output = K=320 packed planes`, `output2 = inject_out`), `scales` `[tokens][4]`, `norm_bf16`
// `[10240]` BF16, and dynamic shared >= 98816 B (`FLASH_BF16_PREFILL_ROWS_SHARED`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_fp8_hc_down_fused(
    p: Bf16Rows,
    scales: u64,
    norm_bf16: u64,
) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let mt_count = (p.tokens as usize).div_ceil(64);
    let nt_count = ((p.rows + p.rows2) as usize).div_ceil(128);
    if p.tokens as usize * W * 4 > 16 * 1024 * 1024 {
        // Schedule all 3 N-tiles of the same M-tile together so 64-token residual stays hot in 24 MiB L2.
        let linear = ni * mt_count + mi;
        mi = linear / nt_count;
        ni = linear % nt_count;
    }
    let (mt, nt) = (mi * 4, ni * 16);
    let kt = W / 16; // 640
    let tail = (nt + 16) * 8 > p.rows as usize;

    let issue = |sh: u32, k: usize| {
        // Stage FP32 residual [64 tokens][64 K] XOR-swizzled by r % 8.
        let mut step = 0usize;
        while step < 4 {
            let i = tid + step * 256;
            let (r, c) = (i / 16, i % 16);
            let token = mt * 16 + r;
            let valid = token < p.tokens as usize;
            let at = if valid { token * W + k * 16 + c * 4 } else { 0 };
            asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
                d = in(reg32) sh + (ROWS_A + (r * 16 + (c ^ (r % 8))) * 16) as u32,
                s = in(reg64) (p.input as *const u32).add(at),
                n = in(reg32) if valid { 16u32 } else { 0u32 });
            step += 1;
        }
        // Stage 64 BF16 norm weights (32 u32s = 128 B).
        if tid < 32 {
            let w_u32 = *((norm_bf16 as *const u32).add(k * 8 + tid));
            ptx::store(sh + (ROWS_W + tid * 4) as u32, f32::from_bits(w_u32));
        }
        // Stage B [128 rows][64 K] without holding a 12-register plan_b array live across the loop.
        let mut bstep = 0usize;
        while bstep < 4 {
            let i = tid + bstep * 256;
            let dst = sh + (2 * PLANE + i * 16) as u32;
            let (row_g, c) = (nt + i / 64, i % 64);
            if !tail || row_g * 8 < p.rows as usize {
                let at = canonical::weight_tile(row_g, k, kt) * 64 + c * 4;
                asm!("cp.async.cg.shared.global [{d}], [{s}], 16, 16;",
                    d = in(reg32) dst, s = in(reg64) (p.weight as *const u32).add(at));
            } else if p.rows2 >= 4 {
                let quad = (row_g * 8 - p.rows as usize) / 4 + c % 16 / 8;
                if quad * 4 < p.rows2 as usize {
                    let at = (quad * kt + k + c / 16) * 32 + c % 8 * 4;
                    asm!("cp.async.cg.shared.global [{d}], [{s}], 16, 16;",
                        d = in(reg32) dst, s = in(reg64) (p.weight2 as *const u32).add(at));
                } else {
                    asm!("cp.async.cg.shared.global [{d}], [{s}], 16, 0;",
                        d = in(reg32) dst, s = in(reg64) p.weight as *const u32);
                }
            } else {
                asm!("cp.async.cg.shared.global [{d}], [{s}], 16, 0;",
                    d = in(reg32) dst, s = in(reg64) p.weight as *const u32);
            }
            bstep += 1;
        }
        ptx::commit();
    };

    let convert = |sh: u32, k: usize| {
        let (a32, w32) = (sh + ROWS_A as u32, sh + ROWS_W as u32);
        let mut step = 0usize;
        while step < 2 {
            let i = tid + step * 256;
            let (m, kk, g, q) = (i / 128, i % 128 / 32, i % 32 / 4, i % 4);
            let r = m * 16 + g;
            let token = mt * 16 + r;
            let (mut hi, mut lo) = ([0u32; 4], [0u32; 4]);
            if token < p.tokens as usize {
                let (local, c) = (kk * 16 + q * 2, (k + kk) * 16 + q * 2);
                let stream = c / H;
                let w01 = ptx::load(w32 + (local * 2) as u32).to_bits();
                let w89 = ptx::load(w32 + ((local + 8) * 2) as u32).to_bits();
                let wf = [
                    bf(w01 as u16) + 1.0,
                    bf((w01 >> 16) as u16) + 1.0,
                    bf(w89 as u16) + 1.0,
                    bf((w89 >> 16) as u16) + 1.0,
                ];
                let load_half = |half: usize| -> (u32, u32, u32, u32) {
                    let tok = token + half * 8;
                    if tok >= p.tokens as usize {
                        return (0, 0, 0, 0);
                    }
                    let row = r + half * 8;
                    let scale = *(scales as *const f32).add(tok * 4 + stream);
                    let at0 = a32 + ((row * 16 + ((local / 4) ^ (row % 8))) * 16 + (local % 4) * 4) as u32;
                    let at8 = a32 + ((row * 16 + (((local + 8) / 4) ^ (row % 8))) * 16 + ((local + 8) % 4) * 4) as u32;
                    let v0 = ptx::load2(at0);
                    let v8 = ptx::load2(at8);
                    let a0 = f32::from_bits(v0[0]) * scale * wf[0];
                    let a1 = f32::from_bits(v0[1]) * scale * wf[1];
                    let a2 = f32::from_bits(v8[0]) * scale * wf[2];
                    let a3 = f32::from_bits(v8[1]) * scale * wf[3];
                    let (ha, la) = crate::gemm::split2_pair(a0, a1);
                    let (hb, lb) = crate::gemm::split2_pair(a2, a3);
                    (ha, la, hb, lb)
                };
                let (h0, l0, h2, l2) = load_half(0);
                let (h1, l1, h3, l3) = load_half(1);
                hi = [h0, h1, h2, h3];
                lo = [l0, l1, l2, l3];
            }
            let at = sh + (i * 16) as u32;
            asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) at,
                a = in(reg32) hi[0], b = in(reg32) hi[1], c = in(reg32) hi[2], d = in(reg32) hi[3]);
            asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) at + PLANE as u32,
                a = in(reg32) lo[0], b = in(reg32) lo[1], c = in(reg32) lo[2], d = in(reg32) lo[3]);
            step += 1;
        }
    };

    issue(0, 0);
    issue(ROWS_STAGE as u32, 4);

    let mut current = 0usize;
    macro_rules! chain {
        ($t0:expr, $t1:expr) => {{
            let (mut high, mut low) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
            let mut t = $t0;
            while t < $t1 {
                if t % 4 == 0 {
                    let s = t / 4;
                    if s == 0 {
                        asm!("cp.async.wait_group 1;");
                    } else {
                        ptx::wait();
                    }
                    core::arch::nvptx::_syncthreads();
                    if s > 0 && t + 4 < kt {
                        issue(((s + 1) % 2 * ROWS_STAGE) as u32, t + 4);
                    }
                    current = s % 2 * ROWS_STAGE;
                    convert(current as u32, t);
                    core::arch::nvptx::_syncthreads();
                }
                let a = current + wm * 2048 + t % 4 * 512 + lane * 16;
                let b = current + 2 * PLANE + wn * 1024 + t % 4 * 256 + lane * 8;
                let hi0 = ptx::load4(a as u32);
                let hi1 = ptx::load4((a + 2048) as u32);
                unroll!(4, j, {
                    let w = ptx::load2((b + j * 1024) as u32);
                    high[0][j] = madd(hi0, w, high[0][j]);
                    high[1][j] = madd(hi1, w, high[1][j]);
                });
                let lo0 = ptx::load4((a + PLANE) as u32);
                let lo1 = ptx::load4((a + PLANE + 2048) as u32);
                unroll!(4, j, {
                    let w = ptx::load2((b + j * 1024) as u32);
                    low[0][j] = madd(lo0, w, low[0][j]);
                    low[1][j] = madd(lo1, w, low[1][j]);
                });
                t += 1;
            }
            let mut value = [[ZERO; 4]; 2];
            unroll!(2, i, { unroll!(4, j, { value[i][j] = add(high[i][j], low[i][j]); }); });
            value
        }};
    }

    let (mut left, mut right) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
    let mut part = 0;
    while part < 4 {
        let v = chain!(kt * part / 4, kt * (part + 1) / 4);
        unroll!(2, i, {
            unroll!(4, j, {
                match part {
                    0 => left[i][j] = v[i][j],
                    1 => left[i][j] = add(left[i][j], v[i][j]),
                    2 => right[i][j] = v[i][j],
                    _ => right[i][j] = add(right[i][j], v[i][j]),
                }
            });
        });
        part += 1;
    }
    let mut out = [[ZERO; 4]; 2];
    unroll!(2, i, { unroll!(4, j, { out[i][j] = add(left[i][j], right[i][j]); }); });

    let kt_low = p.rows as usize / 16; // 20 for 320
    let padded_tokens = (p.tokens as usize).div_ceil(16) * 16;
    let plane_words = (padded_tokens / 16) * kt_low * 128;
    let out_planes = p.output as *mut u32;
    let out_inject = p.output2 as *mut f32;
    let put_low = |t: usize, col: usize, a: f32, b: f32| {
        if t < padded_tokens {
            let word = ((t / 16 * kt_low + col / 16) * 32 + t % 8 * 4 + col % 8 / 2) * 4
                + t % 16 / 8 + col % 16 / 8 * 2;
            let (hi, lo) = if t < p.tokens as usize {
                crate::gemm::split2_pair(
                    crate::flash_ops::silu(a, 1.0, 4.0, 0),
                    crate::flash_ops::silu(b, 1.0, 4.0, 0),
                )
            } else {
                (0u32, 0u32)
            };
            asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) out_planes.add(word), v = in(reg32) hi);
            asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) out_planes.add(plane_words + word), v = in(reg32) lo);
        }
    };
    let put_inj = |t: usize, local_col: usize, v: f32| {
        if t < p.tokens as usize && local_col < p.rows2 as usize {
            *out_inject.add(t * p.stride2 as usize + local_col) = v;
        }
    };
    unroll!(2, i, {
        unroll!(4, j, {
            let (token, col) = ((mt + wm + i) * 16 + lane / 4, (nt + wn + j) * 8 + lane % 4 * 2);
            let v: Acc = out[i][j];
            if col < p.rows as usize {
                put_low(token, col, v.a, v.b);
                put_low(token + 8, col, v.c, v.d);
            } else if p.rows2 != 0 {
                let lc = col - p.rows as usize;
                put_inj(token, lc, v.a);
                put_inj(token, lc + 1, v.b);
                put_inj(token + 8, lc, v.c);
                put_inj(token + 8, lc + 1, v.d);
            }
        });
    });
}

/// Fused 4-slice FP8 prompt HC-up GEMM (`K = 320`, `N = 10240`) + exact FP8 4-stream mix
/// (`layer::exp` sigmoid + BF16 norm weights) + optional dense hi/lo plane write.
/// Each CTA runs 4 consecutive 32-feature slices (128 features/CTA, `grid.y = 20`), overlapping
/// the next slice's first two stages behind the current slice's low-register streaming epilogue.
/// Grid: `(ceil(tokens/64), 20, 1)`, block: `256`, dynamic shared: `98352` (`FLASH_BF16_PREFILL_TILED_SHARED`).
// SAFETY: Caller validates `p` packed K=320 input planes, blocked64 `p.weight` [10240][320],
// `p.output` [tokens][2560], `scales` [tokens][4] (or normalized [tokens][10240] when residual==0),
// optional `planes` (2 dense planes of ceil(tokens/16)*16 x 2560), and `residual`/`weight_bf16`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_fp8_hc_up_tiled(
    p: Bf16Rows,
    scales: u64,
    planes: u64,
    residual: u64,
    weight_bf16: u64,
) {
    const SLICES: usize = 4;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    let nj = _block_idx_y() as usize;
    let kt = p.cols as usize / 16; // 20
    if kt <= 4 || (nj + 1) * SLICES > H / 32 {
        return;
    }

    let first_policy: u64;
    asm!("createpolicy.fractional.L2::evict_first.b64 {p}, 1.0;", p = out(reg64) first_policy);
    let source = if residual != 0 { residual } else { scales };
    let kt_h = H / 16;
    let padded_tokens = (p.tokens as usize).div_ceil(16) * 16;
    let plane_words = (padded_tokens / 16) * kt_h * 128;
    let gates_at = 2 * STAGE;

    let stage_slice = |sh: u32, mt: usize, ni: usize, k: usize| {
        unroll!(2, step, {
            let i = tid + step * 256;
            let row = mt + i / 128;
            let valid = row * 16 < p.tokens as usize;
            let at = ((row * kt + k + i % 128 / 32) * 32 + i % 32) * 4;
            unroll!(2, pl, {
                let x = (p.input as *const u32).add(if valid { at + pl * p.input_plane_words as usize } else { 0 });
                asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
                    d = in(reg32) sh + (pl * PLANE + i * 16) as u32,
                    s = in(reg64) x, n = in(reg32) if valid { 16u32 } else { 0u32 });
            });
        });
        unroll!(4, step, {
            let i = tid + step * 256;
            let g = i / 64;
            let at = canonical::weight_tile(g / 4 * (H / 8) + ni * 4 + g % 4, k, kt) * 64 + i % 64 * 4;
            asm!("cp.async.cg.shared.global [{d}], [{s}], 16, 16;",
                d = in(reg32) sh + (2 * PLANE + i * 16) as u32,
                s = in(reg64) (p.weight as *const u32).add(at));
        });
        ptx::commit();
    };

    let mt0 = _block_idx_x() as usize * 4;
    stage_slice(0, mt0, nj * SLICES, 0);
    stage_slice(STAGE as u32, mt0, nj * SLICES, 4);

    let mut slice_idx = 0usize;
    while slice_idx < SLICES {
        let (mt_u, ni_u): (u64, u64);
        asm!("mov.b64 {m}, {a}; mov.b64 {n}, {b};", m = out(reg64) mt_u, n = out(reg64) ni_u,
            a = in(reg64) (_block_idx_x() as usize * 4) as u64, b = in(reg64) (nj * SLICES + slice_idx) as u64);
        let (mt, ni) = (mt_u as usize, ni_u as usize);

        let mut current = 0usize;
        macro_rules! chain {
            ($t0:expr, $t1:expr) => {{
                let (mut high, mut low) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
                let mut t = $t0;
                while t < $t1 {
                    if t % 4 == 0 {
                        let s = t / 4;
                        if s == 0 {
                            asm!("cp.async.wait_group 1;");
                        } else {
                            ptx::wait();
                        }
                        core::arch::nvptx::_syncthreads();
                        if s > 0 && t + 4 < kt {
                            stage_slice(((s + 1) % 2 * STAGE) as u32, mt, ni, t + 4);
                        }
                        current = s % 2 * STAGE;
                    }
                    let a = current + wm * 2048 + t % 4 * 512 + lane * 16;
                    let b_addr = current + 2 * PLANE + wn * 1024 + t % 4 * 256 + lane * 8;
                    let hi = [ptx::load4(a as u32), ptx::load4((a + 2048) as u32)];
                    let lo = [ptx::load4((a + PLANE) as u32), ptx::load4((a + PLANE + 2048) as u32)];
                    unroll!(4, j, {
                        let w = ptx::load2((b_addr + j * 1024) as u32);
                        unroll!(2, i, {
                            high[i][j] = madd(hi[i], w, high[i][j]);
                            low[i][j] = madd(lo[i], w, low[i][j]);
                        });
                    });
                    t += 1;
                }
                unroll!(2, i, { unroll!(4, j, { high[i][j] = add(high[i][j], low[i][j]); }); });
                high
            }};
        }

        let (mut left, mut right) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
        let mut part = 0;
        while part < 4 {
            let v = chain!(kt * part / 4, kt * (part + 1) / 4);
            unroll!(2, i, {
                unroll!(4, j, {
                    match part {
                        0 => left[i][j] = v[i][j],
                        1 => left[i][j] = add(left[i][j], v[i][j]),
                        2 => right[i][j] = v[i][j],
                        _ => right[i][j] = add(right[i][j], v[i][j]),
                    }
                });
            });
            part += 1;
        }
        unroll!(2, i, { unroll!(4, j, { left[i][j] = add(left[i][j], right[i][j]); }); });

        core::arch::nvptx::_syncthreads();
        unroll!(2, i, {
            unroll!(4, j, {
                let (t, f) = ((wm + i) * 16 + lane / 4, j * 8 + lane % 4 * 2);
                let at0 = (gates_at + ((warp % 4 * 64 + t) * 32 + (f ^ t % 4 * 8)) * 4) as u32;
                let at1 = (gates_at + ((warp % 4 * 64 + (t + 8)) * 32 + (f ^ (t + 8) % 4 * 8)) * 4) as u32;
                asm!("st.shared.v2.f32 [{s}], {{{a}, {b}}};", s = in(reg32) at0, a = in(reg32) left[i][j].a, b = in(reg32) left[i][j].b);
                asm!("st.shared.v2.f32 [{s}], {{{a}, {b}}};", s = in(reg32) at1, a = in(reg32) left[i][j].c, b = in(reg32) left[i][j].d);
            });
        });
        if slice_idx + 1 < SLICES {
            stage_slice(0, mt, ni + 1, 0);
            stage_slice(STAGE as u32, mt, ni + 1, 4);
        }
        core::arch::nvptx::_syncthreads();

        let mut ws = [0.0f32; 8];
        if residual != 0 {
            unroll!(4, stream, {
                let packed_w: u32;
                asm!("ld.global.nc.u32 {v}, [{s}];", v = out(reg32) packed_w,
                    s = in(reg64) (weight_bf16 as *const u32).add((stream * H + ni * 32 + (tid % 16) * 2) / 2),
                    options(readonly, nostack));
                ws[stream * 2] = bf(packed_w as u16) + 1.0;
                ws[stream * 2 + 1] = bf((packed_w >> 16) as u16) + 1.0;
            });
        }
        let mut q = 0usize;
        while q < 4 {
            let e = tid + q * 256;
            let (t, f) = (e / 16, (e % 16) * 2);
            let (token, feature) = (mt * 16 + t, ni * 32 + f);
            if token < p.tokens as usize {
                let mut x = [0.0f32; 8];
                unroll!(4, stream, {
                    let at = (source as *const f32).add(token * W + stream * H + feature);
                    asm!("ld.global.nc.L2::cache_hint.v2.f32 {{{a}, {b}}}, [{s}], {p};",
                        a = out(reg32) x[stream * 2], b = out(reg32) x[stream * 2 + 1],
                        s = in(reg64) at, p = in(reg64) first_policy, options(readonly, nostack));
                });
                if residual != 0 {
                    let (s0, s1, s2, s3): (f32, f32, f32, f32);
                    asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{s}];",
                        a = out(reg32) s0, b = out(reg32) s1, c = out(reg32) s2, d = out(reg32) s3,
                        s = in(reg64) (scales as *const f32).add(token * 4), options(readonly, nostack));
                    let ss = [s0, s1, s2, s3];
                    unroll!(4, stream, {
                        x[stream * 2] = x[stream * 2] * ss[stream] * ws[stream * 2];
                        x[stream * 2 + 1] = x[stream * 2 + 1] * ss[stream] * ws[stream * 2 + 1];
                    });
                }
                let mut a = 0.0f32;
                let mut b = 0.0f32;
                let mut stream = 0;
                while stream < 4 {
                    let g = ptx::load2((gates_at + ((stream * 64 + t) * 32 + (f ^ t % 4 * 8)) * 4) as u32);
                    let sa = 1.0 / (1.0 + exp(-f32::from_bits(g[0])));
                    let sb = 1.0 / (1.0 + exp(-f32::from_bits(g[1])));
                    a += x[stream * 2] * sa;
                    b += x[stream * 2 + 1] * sb;
                    stream += 1;
                }
                a *= 0.25;
                b *= 0.25;
                asm!("st.global.cs.v2.f32 [{y}], {{{a}, {b}}};",
                    y = in(reg64) (p.output as *mut f32).add(token * H + feature), a = in(reg32) a, b = in(reg32) b);
                if planes != 0 {
                    let word = ((token / 16 * kt_h + feature / 16) * 32 + token % 8 * 4 + feature % 8 / 2) * 4
                        + token % 16 / 8 + feature % 16 / 8 * 2;
                    let (hi, lo) = crate::gemm::split2_pair(a, b);
                    asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) (planes as *mut u32).add(word), v = in(reg32) hi);
                    asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) (planes as *mut u32).add(plane_words + word), v = in(reg32) lo);
                }
            } else if planes != 0 && token < padded_tokens {
                let word = ((token / 16 * kt_h + feature / 16) * 32 + token % 8 * 4 + feature % 8 / 2) * 4
                    + token % 16 / 8 + feature % 16 / 8 * 2;
                asm!("st.global.cs.b32 [{y}], 0;", y = in(reg64) (planes as *mut u32).add(word));
                asm!("st.global.cs.b32 [{y}], 0;", y = in(reg64) (planes as *mut u32).add(plane_words + word));
            }
            q += 1;
        }
        slice_idx += 1;
    }
}

/// 16-byte coalesced word copy for aligned RDMA mailbox receive (`words4 = bytes / 16`).
// SAFETY: Caller validates 16-byte aligned `src` and `dst` covering at least `words4 * 16` bytes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_fp8_copy_v4(src: *const u32, dst: *mut u32, words4: u32) {
    let i = _block_idx_x() as usize * 128 + _thread_idx_x() as usize;
    if i < words4 as usize {
        let (a, b, c, d): (u32, u32, u32, u32);
        asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
            p = in(reg64) src.add(i * 4), options(readonly, nostack));
        asm!("st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};",
            p = in(reg64) dst.add(i * 4),
            a = in(reg32) a, b = in(reg32) b, c = in(reg32) c, d = in(reg32) d);
    }
}

/// 16-byte coalesced in-place FP32 add for aligned RDMA mailbox receive (`words4 = bytes / 16`).
// SAFETY: Caller validates 16-byte aligned `dst` and `src` covering at least `words4 * 16` bytes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_fp8_add_v4(dst: *mut f32, src: *const f32, words4: u32) {
    let i = _block_idx_x() as usize * 128 + _thread_idx_x() as usize;
    if i < words4 as usize {
        let (a0, a1, a2, a3): (f32, f32, f32, f32);
        let (b0, b1, b2, b3): (f32, f32, f32, f32);
        asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) a0, b = out(reg32) a1, c = out(reg32) a2, d = out(reg32) a3,
            p = in(reg64) dst.add(i * 4));
        asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) b0, b = out(reg32) b1, c = out(reg32) b2, d = out(reg32) b3,
            p = in(reg64) src.add(i * 4), options(readonly, nostack));
        asm!("st.global.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};",
            p = in(reg64) dst.add(i * 4),
            a = in(reg32) a0 + b0, b = in(reg32) a1 + b1, c = in(reg32) a2 + b2, d = in(reg32) a3 + b3);
    }
}
