//! Native E4M3 tensor-core projections for the checkpoint's 128x128 blocks.
//! Dynamic per-token E4M3 operands and one fixed 128-K accumulation order.
use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x}};
use crate::ptx;

#[derive(Clone, Copy)]
struct Acc([f32; 4]);
const ZERO: Acc = Acc([0.0; 4]);

macro_rules! unroll4 {
    ($i:ident, $body:block) => {
        { let $i = 0; $body }
        { let $i = 1; $body }
        { let $i = 2; $body }
        { let $i = 3; $body }
    };
}

#[inline(always)]
unsafe fn mma(a: [u32; 4], b: [u32; 2], c: Acc) -> Acc {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    asm!(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \
         {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, \
         {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
        d0=out(reg32)d0, d1=out(reg32)d1, d2=out(reg32)d2, d3=out(reg32)d3,
        a0=in(reg32)a[0], a1=in(reg32)a[1], a2=in(reg32)a[2], a3=in(reg32)a[3],
        b0=in(reg32)b[0], b1=in(reg32)b[1],
        c0=in(reg32)c.0[0], c1=in(reg32)c.0[1], c2=in(reg32)c.0[2], c3=in(reg32)c.0[3],
    );
    Acc([d0, d1, d2, d3])
}

#[inline(always)]
pub(crate) unsafe fn cvt_pair(low: f32, high: f32) -> u16 {
    let pair: u16;
    asm!("cvt.rn.satfinite.e4m3x2.f32 {d}, {h}, {l};",
        d=out(reg16)pair, h=in(reg32)high, l=in(reg32)low);
    pair
}

#[inline(always)]
pub(crate) unsafe fn warp_max(mut value: f32) -> f32 {
    let mut distance = 16;
    while distance != 0 {
        let other: f32;
        asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, 0xffffffff;",
            o=out(reg32)other, v=in(reg32)value, d=in(reg32)distance);
        value = value.max(other);
        distance /= 2;
    }
    value
}

/// The one dynamic E4M3 quantizer: a warp's 32 lanes hold a 128-value group,
/// four values each. Scale = group maxabs / 448 (stored as is, zero groups
/// included); values are divided by the scale floored at 1e-12 and rounded to
/// E4M3 with saturation. Returns this lane's packed word (values 0..3 in byte
/// order) and the group scale. Every lane of the warp must call it.
#[inline(always)]
pub(crate) unsafe fn quantize4(v: [f32; 4]) -> (u32, f32) {
    let scale = warp_max(v[0].abs().max(v[1].abs()).max(v[2].abs()).max(v[3].abs())) / 448.0;
    let denominator = scale.max(1e-12);
    let word = cvt_pair(v[0] / denominator, v[1] / denominator) as u32
        | ((cvt_pair(v[2] / denominator, v[3] / denominator) as u32) << 16);
    (word, scale)
}

/// FP32 activations -> E4M3, one positive FP32 scale per contiguous 128 values.
/// # Safety
/// K is a multiple of 128. X has rows*K elements; scales has rows*K/128.
/// Q has rows*K bytes, or 16*K bytes in packed mode (1<=rows<=16).
/// Flatten experts into rows. Grid ceil(rows*(K/128)/8), block 256, shared 0.
/// Inputs are finite. The stored scale is maxabs/448, including zero groups;
/// the quantization denominator alone is floored at 1e-12.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_quantize_128(
    p: crate::fp8_abi::QuantizeParams,
) {
    let (x, q, scales, rows, k) = (p.input as *const f32, p.output as *mut u8,
        p.scales as *mut f32, p.rows, p.cols);
    let tid = _thread_idx_x() as usize;
    let group = _block_idx_x() as usize * 8 + tid / 32;
    if group >= rows as usize * (k as usize / 128) { return; }
    let offset = group * 128 + (tid % 32) * 4;
    let mut values = [*x.add(offset), *x.add(offset + 1), *x.add(offset + 2), *x.add(offset + 3)];
    if p.round_input != 0 { for v in &mut values { *v = rounded(*v); } }
    let (word, scale) = quantize4(values);
    let output = if p.packed_output != 0 {
        let row = group / (k as usize / 128);
        let col = group % (k as usize / 128) * 128 + tid % 32 * 4;
        // Four contiguous thread words are the m16n8k32 A registers. Missing
        // rows are masked by the reader, so quantization needs no padding pass.
        (col / 32 * 128 + (row % 8 * 4 + col % 16 / 4) * 4
            + row / 8 + col % 32 / 16 * 2) * 4
    } else { offset };
    *q.add(output).cast::<u32>() = word;
    if tid % 32 == 0 { *scales.add(group) = scale; }
}

#[inline(always)]
fn address(base: u32, row: usize, col: usize) -> u32 {
    // XOR the eight 16-byte sectors: contiguous lane words hit distinct banks.
    base + (row * 128 + (col ^ ((row & 7) * 16))) as u32
}

#[inline(always)]
unsafe fn load_a(base: u32, row: usize, col: usize, lane: usize) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    let p = address(base, row + lane % 16, col + lane / 16 * 16);
    asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a},{b},{c},{d}}}, [{p}];",
        a=out(reg32)a, b=out(reg32)b, c=out(reg32)c, d=out(reg32)d, p=in(reg32)p);
    [a, b, c, d]
}

#[inline(always)]
unsafe fn load_b(base: u32, row: usize, col: usize, lane: usize) -> [u32; 2] {
    let (a, b): (u32, u32);
    let p = address(base, row + lane % 8, col + (lane / 8 % 2) * 16);
    asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{a},{b}}}, [{p}];",
        a=out(reg32)a, b=out(reg32)b, p=in(reg32)p);
    [a, b]
}

#[inline(always)]
pub(crate) fn rounded(v: f32) -> f32 {
    crate::bf16::rounded(v)
}

#[inline(always)]
unsafe fn stage<const TM: usize, const TN: usize, const THREADS: usize, const FULL: bool>(
    shared: u32, x: *const u8, w: *const u8, xs: *const f32,
    mb: usize, nb: usize, kb: usize, m: usize, k: usize, tid: usize,
    kind: u32, row_map: *const u32, input_stride: usize, up: *const u8,
) {
    let (tm, tn, threads) = (TM, TN, THREADS);
    if kind == 0 {
        let mut copy = tid;
        while copy < tm * 8 {
            let row = copy / 8;
            let col = copy % 8 * 16;
            let dst = address(shared, row, col);
            if FULL || mb + row < m {
                ptx::copy16(dst, x.add((if row_map.is_null() { mb + row } else { *row_map.add(mb + row) as usize }) * input_stride + kb + col).cast::<u32>());
            } else {
                unroll4!(j, { ptx::store(dst + (j * 4) as u32, 0.0); });
            }
            copy += threads;
        }
        if tid < tm {
            ptx::store(shared + (tm * 128 + tn * 128) as u32 + (tid * 4) as u32,
                if FULL || mb + tid < m { *xs.add((if row_map.is_null() { mb + tid } else { *row_map.add(mb + tid) as usize }) * (input_stride / 128) + kb / 128) } else { 0.0 });
        }
    } else {
        // A warp owns each token/group. Padding participates in all barriers.
        let mut row = tid / 32;
        let col = tid % 32 * 4;
        while row < tm {
            let mut v = [0.0; 4];
            if FULL || mb + row < m {
                unroll4!(j, { v[j] = rounded(*x.cast::<f32>().add((mb + row) * input_stride + kb + col + j)); });
            }
            let (raw, scale) = quantize4(v);
            let dst = address(shared, row, col);
            asm!("st.shared.b32 [{p}], {v};", p=in(reg32)dst, v=in(reg32)raw);
            if tid % 32 == 0 { ptx::store(shared + (tm * 128 + tn * 128) as u32 + (row * 4) as u32, scale); }
            row += threads / 32;
        }
    }
    let mut copy = tid;
    while copy < tn * 8 {
        let row = copy / 8;
        let col = copy % 8 * 16;
        let dst = address(shared + (tm * 128) as u32, row, col);
        let (weight, row) = if !up.is_null() && row >= tn / 2 { (up, row - tn / 2) } else { (w, row) };
        ptx::copy16(dst, weight.add((nb + row) * k + kb + col).cast::<u32>());
        copy += threads;
    }
}

/// Y[e,m,n] = sum_k decoded(W[e,n,k])*WS[e,n/128,k/128]
///                         * decoded(X[e,m,k])*XS[e,m,k/128].
/// The raw E4M3 dot product accumulates in FP32 for exactly 128 K values;
/// both block scales are applied before accumulating the next block.
/// # Safety
/// W/X are row-major E4M3; WS is row-major BF16, XS/Y row-major FP32.
/// N and K are positive multiples of 128. All pointers cover the full grid Z.
/// X is quantized. Grid(ceil(M/128),N/128,E), block512, shared66560. No static shared region.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_fp8_block128(
    p: crate::fp8_abi::BlockParams,
) {
    let mut p = p;
    p.input_kind = 0;
    project::<128, 128, 512>(p);
}

/// Narrow rows use the same arithmetic with their own register allocation.
/// # Safety
/// M<=16, N<=512 row-major E4M3: grid(ceil(M/8),N/8,1), block128, shared1024.
/// Packed input: 1<=M<=16, E=1, grid(1,N/8,1), block128, shared14336.
/// Row-major input: M<=16 uses M16/shared20608; 17..=32 uses M32/shared24832.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_fp8_block128_decode(p: crate::fp8_abi::BlockParams) {
    if crate::fp8_abi::register_decode(p.input_kind, p.rows, p.outputs) { decode_registers(p); }
    else if p.input_kind == 2 { decode_packed(p); }
    else if p.rows > 16 && p.rows <= 32 { project::<32, 64, 128>(p); }
    else { project::<16, 64, 128>(p); }
}

/// Four warps compute independent K128 dots in registers. Only the scaled
/// results cross warps; warp zero adds them in the original ascending K order.
/// Grid(ceil(M/8),N/8,1), block128, shared1024; row-major input, M<=16.
///
/// A row slab changes only activation/scales/output addresses. MMA weight
/// fragments still use the local lane group: matrix B has no token dimension.
/// All warps enter both barriers, including lanes past the final live row.
/// Each slab has its own shared K128 partials and owns disjoint output rows.
/// The four independent K128 chains, separate scale multiply, and ascending
/// cross-warp additions below are unchanged from the single-slab entry.
#[inline(always)]
unsafe fn decode_registers(p: crate::fp8_abi::BlockParams) {
    let (m, n, k) = (p.rows as usize, p.outputs as usize, p.cols as usize);
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let local_row = lane / 4;
    let row = _block_idx_x() as usize * crate::fp8_abi::REGISTER_DECODE_TILE[0] as usize + local_row;
    let nb = _block_idx_y() as usize * 8;
    let (w, ws, x, xs, y) = (p.weights as *const u8, p.weight_scales as *const u16,
        p.input as *const u8, p.input_scales as *const f32, p.output as *mut f32);
    let mut sum = [0.0f32; 2];
    let mut step = 0;
    while step < k {
        let kb = step + warp * 128;
        if kb < k {
            let mut block = ZERO;
            unroll4!(z, {
                let col = kb + z * 32 + lane % 4 * 4;
                let mut a = [0u32; 4];
                if row < m {
                    a[0] = *x.add(row * k + col).cast::<u32>();
                    a[2] = *x.add(row * k + col + 16).cast::<u32>();
                }
                let weight = w.add((nb + local_row) * k + col).cast::<u32>();
                // Adjacent groups of four lanes cover a complete K32 sector.
                let b = [*weight, *weight.add(4)];
                block = mma(a, b, block);
            });
            let weight_scale = f32::from_bits((*ws.add(nb / 128 * (k / 128) + kb / 128) as u32) << 16);
            let scale = if row < m { *xs.add(row * (k / 128) + kb / 128) * weight_scale } else { 0.0 };
            let dst = ((warp * 64 + lane * 2) * 4) as u32;
            asm!("st.shared.v2.f32 [{p}], {{{a}, {b}}};", p=in(reg32)dst,
                a=in(reg32)(block.0[0] * scale), b=in(reg32)(block.0[1] * scale));
        }
        ptx::sync();
        if warp == 0 {
            unroll4!(g, {
                if step + g * 128 < k {
                    let v = ptx::load2(((g * 64 + lane * 2) * 4) as u32);
                    sum[0] += f32::from_bits(v[0]);
                    sum[1] += f32::from_bits(v[1]);
                }
            });
        }
        ptx::sync();
        step += 512;
    }
    if warp == 0 && row < m {
        if p.round_output != 0 { sum[0] = rounded(sum[0]); sum[1] = rounded(sum[1]); }
        let col = nb + lane % 4 * 2;
        *y.add(row * n + col) = sum[0];
        *y.add(row * n + col + 1) = sum[1];
    }
}

#[inline(always)]
unsafe fn stage_decode(shared: u32, w: *const u8, k: usize, kb: usize, lane: usize) {
    let mut copy = lane;
    while copy < 64 {
        let row = copy / 8;
        let col = copy % 8 * 16;
        ptx::copy16(address(shared, row, col), w.add(row * k + kb + col).cast::<u32>());
        copy += 32;
    }
}

/// Four warps compute successive independent K128 dots. Warp zero consumes
/// them in K order, preserving the canonical scale multiply and FP32 sum.
#[inline(always)]
unsafe fn decode_packed(p: crate::fp8_abi::BlockParams) {
    let (m, n, k) = (p.rows as usize, p.outputs as usize, p.cols as usize);
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let nb = _block_idx_y() as usize * 8;
    let w = (p.weights as *const u8).add(nb * k);
    let ws = (p.weight_scales as *const u16).add(nb / 128 * (k / 128));
    let x = p.input as *const u32;
    let xs = p.input_scales as *const f32;
    let y = p.output as *mut f32;
    let raw = (warp * 3072) as u32;
    let row = lane / 4;
    let mut sum = ZERO;
    if warp * 128 < k { stage_decode(raw, w, k, warp * 128, lane); }
    ptx::commit();
    if warp * 128 + 512 < k {
        stage_decode(raw + 1024, w, k, warp * 128 + 512, lane);
        ptx::commit();
        ptx::wait_one();
    } else { ptx::wait(); }
    ptx::warp_sync();
    let mut step = 0;
    while step < k {
        let kb = step + warp * 128;
        let current = raw + ((step / 512 % 3) * 1024) as u32;
        if kb + 1024 < k { stage_decode(raw + (((step / 512 + 2) % 3) * 1024) as u32, w, k, kb + 1024, lane); ptx::commit(); }
        let mut block = ZERO;
        if kb < k {
            unroll4!(z, {
                let mut a = [0; 4];
                if row < m {
                    let src = x.add((kb / 32 + z) * 128 + lane * 4);
                    asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                        a=out(reg32)a[0], b=out(reg32)a[1], c=out(reg32)a[2], d=out(reg32)a[3], p=in(reg64)src);
                    if row + 8 >= m { a[1] = 0; a[3] = 0; }
                }
                block = mma(a, load_b(current, 0, z * 32, lane), block);
            });
            let scale = f32::from_bits((*ws.add(kb / 128) as u32) << 16);
            let s0 = if row < m { *xs.add(row * (k / 128) + kb / 128) * scale } else { 0.0 };
            let s1 = if row + 8 < m { *xs.add((row + 8) * (k / 128) + kb / 128) * scale } else { 0.0 };
            let values = [block.0[0] * s0, block.0[1] * s0, block.0[2] * s1, block.0[3] * s1];
            let dst = 12288 + ((warp * 128 + lane * 4) * 4) as u32;
            asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p=in(reg32)dst,
                a=in(reg32)values[0], b=in(reg32)values[1], c=in(reg32)values[2], d=in(reg32)values[3]);
        }
        ptx::sync();
        if warp == 0 {
            unroll4!(g, {
                if step + g * 128 < k {
                    let values = ptx::load4(12288 + ((g * 128 + lane * 4) * 4) as u32);
                    unroll4!(c, { sum.0[c] += f32::from_bits(values[c]); });
                }
            });
        }
        if kb + 1024 < k { ptx::wait_one(); } else { ptx::wait(); }
        ptx::sync();
        step += 512;
    }
    if warp == 0 {
        let col = nb + lane % 4 * 2;
        let mut v = sum.0;
        if p.round_output != 0 { for x in &mut v { *x = rounded(*x); } }
        if row < m { *y.add(row * n + col) = v[0]; *y.add(row * n + col + 1) = v[1]; }
        if row + 8 < m { *y.add((row + 8) * n + col) = v[2]; *y.add((row + 8) * n + col + 1) = v[3]; }
    }
}

#[inline(always)]
unsafe fn project<const TM: usize, const TN: usize, const THREADS: usize>(p: crate::fp8_abi::BlockParams) {
    let (m, n, k) = (p.rows as usize, p.outputs as usize, p.cols as usize);
    let (w, ws, x, xs, y) = (p.weights as *const u8, p.weight_scales as *const u16,
        p.input as *const u8, p.input_scales as *const f32, p.output as *mut f32);
    let expert = _block_idx_z() as usize;
    let w = w.add(expert * n * k);
    let ws = ws.add(expert * (n / 128) * (k / 128));
    let x = x.add(expert * m * k * if p.input_kind == 0 { 1 } else { 4 });
    let xs = if p.input_kind == 0 { xs.add(expert * m * (k / 128)) } else { xs };
    let y = y.add(expert * m * n);
    let (tm, tn) = (TM, TN);
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    if THREADS != 128 && m >= TM {
        // Keep eight input tiles in L2 while visiting output tiles, as in
        // the resident E4M3 prefill pipeline. Padding still has one CTA.
        let mt = (m + TM - 1) / TM;
        let nt = n / TN;
        let linear = ni * mt + mi;
        let group = if k <= 8192 { 8 } else { 4 };
        let first = linear / (nt * group) * group;
        let rows = group.min(mt - first);
        let within = linear - first * nt;
        mi = first + within % rows;
        ni = within / rows;
    }
    project_tile::<TM, TN, THREADS, false, false, 2>(crate::fp8_abi::BlockParams {
        weights: w as u64, weight_scales: ws as u64, input: x as u64,
        input_scales: xs as u64, output: y as u64, ..p
    }, mi * tm, ni * tn, n, core::ptr::null(), core::ptr::null(), 0, 0, k, [0; 2]);
}

/// Canonical K128 arithmetic shared by dense and rank-sharded routed experts.
#[inline(always)]
pub(crate) unsafe fn project_tile<const TM: usize, const TN: usize, const THREADS: usize, const FULL: bool, const GENERAL: bool, const STAGES: usize>(
    p: crate::fp8_abi::BlockParams, mb: usize, nb: usize, output_stride: usize,
    row_map: *const u32, output_map: *const u32, shared: u32, mut split_k: usize, input_stride: usize, paired: [u64; 2],
) {
    let (m, k) = (p.rows as usize, p.cols as usize);
    let (w, ws, x, xs, y) = (p.weights as *const u8, p.weight_scales as *const u16,
        p.input as *const u8, p.input_scales as *const f32, p.output as *mut f32);
    let tid = _thread_idx_x() as usize % THREADS;
    let (lane, warp) = (tid % 32, tid / 32);
    let (tm, tn) = (TM, TN);
    let warp_rows = if THREADS == 512 { TM / 4 } else if THREADS == 128 { TM } else { 16 };
    let (mr, nr) = (warp / 4 * warp_rows, warp % 4 * (tn / 4));
    // Dense callers supply dynamic shared memory; Flash reuses its static tile.
    let stride = tm * 128 + tn * 128 + tm * 4;
    let mut sums = [[ZERO; 4]; 4];
    let mut first = [[ZERO; 4]; 4];
    let (world, expert) = (split_k >> 16, split_k & 65535);
    let mut owner = 0;
    // An empty rank shard still writes exact zero partials for its routes.
    if k != 0 { stage::<TM, TN, THREADS, FULL>(shared, x, w, xs, mb, nb, 0, m, k, tid, p.input_kind, row_map, input_stride, paired[0] as *const u8); }
    // STAGES 3 keeps two K128 blocks in flight: one commit group per block, and each
    // block waits only for its own group. The per-output K128 order is unchanged.
    if STAGES == 3 {
        ptx::commit();
        if 128 < k { stage::<TM, TN, THREADS, FULL>(shared + stride as u32, x, w, xs, mb, nb, 128, m, k, tid, p.input_kind, row_map, input_stride, paired[0] as *const u8); }
        ptx::commit();
    } else {
        ptx::wait();
        ptx::sync();
    }
    let mut kb = 0;
    while kb < k {
        if STAGES == 3 {
            ptx::wait_one();
            ptx::sync();
        }
        if split_k != 0 && kb == split_k {
            first = sums;
            sums = [[ZERO; 4]; 4];
        }
        if GENERAL {
            while owner + 1 < world && kb == crate::fp8_abi::middle_expert_end(expert, owner, world) * 128 {
                if owner == 0 { first = sums; } else {
                    unroll4!(i, { unroll4!(j, { unroll4!(c, { first[i][j].0[c] += sums[i][j].0[c]; }); }); });
                }
                sums = [[ZERO; 4]; 4];
                owner += 1;
            }
        }
        let current = shared + ((kb / 128 % STAGES) * stride) as u32;
        // The refilled slot was last read by the previous block; every warp has passed its barrier.
        if kb + (STAGES - 1) * 128 < k {
            let next = shared + (((kb / 128 + STAGES - 1) % STAGES) * stride) as u32;
            stage::<TM, TN, THREADS, FULL>(next, x, w, xs, mb, nb, kb + (STAGES - 1) * 128, m, k, tid, p.input_kind, row_map, input_stride, paired[0] as *const u8);
        }
        if STAGES == 3 { ptx::commit(); }
        let mut block = [[ZERO; 4]; 4];
        unroll4!(z, {
            let col = z * 32;
            let mut bb = [[0; 2]; 4];
            unroll4!(j, {
                let row = nr + j * 8;
                if j < tn / 32 { bb[j] = load_b(current + (tm * 128) as u32, row, col, lane); }
            });
            unroll4!(i, {
                if i < warp_rows / 16 {
                let a = load_a(current, mr + i * 16, col, lane);
                unroll4!(j, { if j < tn / 32 { block[i][j] = mma(a, bb[j], block[i][j]); } });
                }
            });
        });
        let ws = if paired[0] != 0 && nr >= tn / 2 { paired[1] as *const u16 } else { ws };
        let weight_scale = f32::from_bits((*ws.add((nb / 128) * (k / 128) + kb / 128) as u32) << 16);
        unroll4!(i, {
            if i < warp_rows / 16 {
            let row = mb + mr + i * 16 + lane / 4;
            let s0 = if FULL || row < m { ptx::load(current + (tm * 128 + tn * 128) as u32 + ((row - mb) * 4) as u32) * weight_scale } else { 0.0 };
            let s1 = if FULL || row + 8 < m { ptx::load(current + (tm * 128 + tn * 128) as u32 + ((row + 8 - mb) * 4) as u32) * weight_scale } else { 0.0 };
            unroll4!(j, {
                sums[i][j].0[0] += block[i][j].0[0] * s0;
                sums[i][j].0[1] += block[i][j].0[1] * s0;
                sums[i][j].0[2] += block[i][j].0[2] * s1;
                sums[i][j].0[3] += block[i][j].0[3] * s1;
            });
            }
        });
        if STAGES != 3 {
            ptx::wait();
            ptx::sync();
        }
        kb += 128;
    }
    if STAGES == 3 { ptx::wait(); }
    if GENERAL {
        unroll4!(i, { unroll4!(j, { unroll4!(c, {
            if owner != 0 { sums[i][j].0[c] = first[i][j].0[c] + sums[i][j].0[c]; }
            // Trailing empty shards still participate in the rank fold.
            for _ in owner + 1..world { sums[i][j].0[c] += 0.0; }
        }); }); });
        split_k = 0;
    }
    unroll4!(i, {
        if i < warp_rows / 16 {
        let row = mb + mr + i * 16 + lane / 4;
        unroll4!(j, {
            if j < tn / 32 {
            let col = nb + nr + j * 8 + (lane % 4) * 2
                + if paired[0] != 0 && nr >= tn / 2 { output_stride / 2 - tn / 2 } else { 0 };
            let mut v = sums[i][j].0;
            if split_k != 0 {
                for c in 0..4 { v[c] = first[i][j].0[c] + v[c]; }
            }
            if p.round_output != 0 {
                for a in &mut v { *a = rounded(*a); }
            }
            if FULL || row < m {
                let output_row = if output_map.is_null() { row } else { *output_map.add(row) as usize };
                *y.add(output_row * output_stride + col) = v[0];
                *y.add(output_row * output_stride + col + 1) = v[1];
            }
            if FULL || row + 8 < m {
                let output_row = if output_map.is_null() { row + 8 } else { *output_map.add(row + 8) as usize };
                *y.add(output_row * output_stride + col) = v[2];
                *y.add(output_row * output_stride + col + 1) = v[3];
            }
            }
        });
        }
    });
}

/// Static shared bytes of the grouped decode entries: three stages of the decode W8A8
/// 16x64 tile (A rows, B rows, row scales per K128 block); it also covers their BF16
/// branch's 28672-byte tile.
pub(crate) const GROUPED_DECODE_SHARED: usize = 3 * (16 * 128 + 64 * 128 + 16 * 4);
const _: () = assert!(GROUPED_DECODE_SHARED >= 28672);

/// Flash tiles retain the dense K128 sum order at every row count.
/// Gate and up share token quantization; down receives expert-local groups.
#[inline(always)]
pub(crate) unsafe fn grouped_project<const PREFILL: bool>(p: crate::fp8_abi::GroupedParams, down: bool, shared: u32) {
    grouped_project_impl::<PREFILL, false>(p, down, shared);
}
#[inline(always)]
unsafe fn grouped_project_impl<const PREFILL: bool, const GENERAL: bool>(p: crate::fp8_abi::GroupedParams, down: bool, shared: u32) {
    // The tile is written for exactly four warps; another launch shape would
    // read past the shared A tile and write wrong rows.
    if core::arch::nvptx::_block_dim_x() != 128 { core::arch::nvptx::trap(); }
    let task = _block_idx_y() as usize;
    let tasks = p.route.tasks as *const u32;
    if task >= *tasks as usize { return; }
    let expert = *tasks.add(1 + task * 2) as usize;
    let m = *(p.route.counts as *const u32).add(expert) as usize;
    let mut tile_rows = if PREFILL && m >= 32 { 32 } else { 16 };
    // The host folds the old Z plane into X's low index: neighboring CTAs
    // visit the same expert/output weight tile. Row ownership stays unchanged.
    let planes = crate::fp8_abi::grouped_planes(p.route.tokens, PREFILL);
    let block = _block_idx_x();
    let tile = (block / planes) as usize;
    let plane_index = (block % planes) as usize;
    let plane = plane_index * tile_rows;
    if plane >= 128 { return; }
    let mb = *tasks.add(2 + task * 2) as usize + plane;
    if mb >= m { return; }
    if PREFILL && !down && tile_rows == 32 {
        // Pair only complete adjacent M32 planes; retain the original tail tile.
        if plane_index % 2 != 0 && mb + 32 <= m { return; }
        if plane_index % 2 == 0 && mb + 64 <= m { tile_rows = 64; }
    }
    let start = *(p.route.starts as *const u32).add(expert) as usize;
    let table = (p.weights as *const u64).add(expert * 7);
    let width = *table.add(6) as usize;
    let (projection, nb) = if down { (4, tile * 64) } else { ((tile / 10) * 2, tile % 10 * 64) };
    if !down && nb >= width { return; }
    let owned = down && p.output_shard != 0;
    let k = if owned { 640 } else if down { width } else { 2560 };
    let output_stride = if down { 2560 } else { 1280 };
    let output_offset = if down { 0 } else { start * output_stride + if projection == 2 { 640 } else { 0 } };
    let params = crate::fp8_abi::BlockParams {
        weights: *table.add(projection), weight_scales: *table.add(projection + 1),
        input: p.input + if down { (start * 640) as u64 } else { 0 },
        input_scales: p.input_scales + if down { (start * 5 * 4) as u64 } else { 0 },
        output: p.output + (output_offset * 4) as u64,
        rows: m as u32, outputs: if GENERAL { p.output_shard & 65535 } else if owned { 1280 } else if down { 2560 } else { width as u32 }, cols: k as u32,
        input_kind: 0, round_output: u32::from(!down),
    };
    let map = if down { core::ptr::null() } else { (p.route.sorted_row as *const u32).add(start) };
    let output_map = if down { (p.route.output_row as *const u32).add(start) } else { core::ptr::null() };
    // GENERAL folds the K128 blocks in the two-rank groups at every rank count,
    // so its sums equal the TP2 entry's split_k fold bit for bit.
    let split_k = if GENERAL { (crate::fp8_abi::EXPERT_FOLD_RANKS << 16) | expert } else if owned { (3 - expert % 2) * 128 } else { 0 };
    // Down partials stay FP32 until rank reduction, then round once to BF16.
    // Counts are device-resident. One uniform decision per CTA keeps the
    // complete tile's staging, scale loads and stores free of row guards.
    if PREFILL && !down && tile_rows == 64 {
        // Four canonical M16 fragments reuse one weight tile; all 64 rows are live.
        project_tile::<64, 64, 128, true, GENERAL, 2>(params, mb, nb, output_stride, map, output_map, shared, split_k, params.cols as usize, [0; 2]);
    } else if PREFILL && tile_rows == 32 {
        if mb + 32 <= m {
            project_tile::<32, 64, 128, true, GENERAL, 2>(params, mb, nb, output_stride, map, output_map, shared, split_k, params.cols as usize, [0; 2]);
        } else {
            project_tile::<32, 64, 128, false, GENERAL, 2>(params, mb, nb, output_stride, map, output_map, shared, split_k, params.cols as usize, [0; 2]);
        }
    } else if PREFILL && mb + 16 <= m {
        project_tile::<16, 64, 128, true, GENERAL, 2>(params, mb, nb, output_stride, map, output_map, shared, split_k, params.cols as usize, [0; 2]);
    } else if !PREFILL && !GENERAL {
        // Decode W8A8 tiles stream three K128 stages (the entries reserve 3 x 10304 bytes).
        project_tile::<16, 64, 128, false, false, 3>(params, mb, nb, output_stride, map, output_map, shared, split_k, params.cols as usize, [0; 2]);
    } else {
        project_tile::<16, 64, 128, false, GENERAL, 2>(params, mb, nb, output_stride, map, output_map, shared, split_k, params.cols as usize, [0; 2]);
    }
}

/// Separate prompt entry points keep the larger tile's registers out of decode.
/// Block128, grid(20*planes,tasks,1), planes=min(ceil(tokens/16),4).
/// Each task covers up to 128 routed rows; plane is the fastest block index.
/// Shared33280 supports complete M64 pairs without increasing padded row work.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_gate_prefill(p: crate::fp8_abi::GroupedParams) {
    grouped_project::<true>(p, false, ptx::shared::<33280>());
}

/// Block128, grid(40*planes,tasks,1); output-owned down uses 20*planes.
/// Identical K128 arithmetic and FP32 rank partials.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_down_prefill(p: crate::fp8_abi::GroupedParams) {
    grouped_project::<true>(p, true, ptx::shared::<24832>());
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_down_general(p: crate::fp8_abi::GroupedParams) {
    grouped_project_impl::<false, true>(p, true, ptx::shared::<20608>());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_down_general_prefill(p: crate::fp8_abi::GroupedParams) {
    grouped_project_impl::<true, true>(p, true, ptx::shared::<24832>());
}

/// Generic expert projections: M16 decode or M128 prompt routing tasks.
/// Selected weights/scales are already local; original block origins are provenance.
#[inline(always)]
unsafe fn routed_project(g: crate::fp8_abi::RoutedParams, projection: usize) {
    let tile = crate::fp8_abi::RoutedTile::new(g.grouped.route.tokens);
    if core::arch::nvptx::_block_dim_x() != tile.threads { core::arch::nvptx::trap(); }
    let p = g.grouped;
    let tasks = p.route.tasks as *const u32;
    let (mut task, mut block) = (_block_idx_y() as usize, _block_idx_x() as usize);
    if tile.rows == 128 {
        // Visit eight routed M tiles at one N tile before advancing columns.
        // Only CTA ownership changes; each output keeps its canonical K128 chain.
        let columns = core::arch::nvptx::_grid_dim_x() as usize;
        let linear = task * columns + block;
        let count = *tasks as usize;
        if linear >= count * columns { return; }
        let first = linear / (columns * 8) * 8;
        let rows = (count - first).min(8);
        let within = linear - first * columns;
        task = first + within % rows;
        block = within / rows;
    } else if task >= *tasks as usize { return; }
    let expert = *tasks.add(1 + task * 2) as usize;
    let m = *(p.route.counts as *const u32).add(expert) as usize;
    let paired = projection == 0 && tile.cols == 128;
    let mb = *tasks.add(2 + task * 2) as usize + block % tile.planes * tile.rows;
    let nb = block / tile.planes * if paired { tile.cols / 2 } else { tile.cols };
    let d = *(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3 + projection);
    if mb >= m || nb >= d.rows as usize { return; }
    let start = *(p.route.starts as *const u32).add(expert) as usize;
    let down = projection == 2;
    let (h, i) = (p.cols as usize, g.layout.intermediate as usize);
    let pitch = if down { h } else { 2 * i };
    let input_stride = if down { i } else { h };
    let params = crate::fp8_abi::BlockParams {
        weights: d.weights, weight_scales: d.scales,
        input: p.input + if down { (start * i) as u64 } else { 0 },
        input_scales: p.input_scales + if down { (start * (i / 128) * 4) as u64 } else { 0 },
        output: p.output + if down { 0 } else { ((start * pitch + projection * i) * 4) as u64 },
        rows: m as u32, outputs: d.rows, cols: d.cols, input_kind: 0, round_output: u32::from(!down),
    };
    let map = if down { core::ptr::null() } else { (p.route.sorted_row as *const u32).add(start) };
    let output_map = if down { (p.route.output_row as *const u32).add(start) } else { core::ptr::null() };
    if tile.rows == 128 {
        let up = *(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3 + 1);
        project_tile::<128, 128, 512, false, false, 2>(params, mb, nb, pitch, map, output_map,
            0, 0, input_stride, if paired { [up.weights, up.scales] } else { [0; 2] });
    } else if tile.cols == 128 {
        let up = *(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3 + 1);
        project_tile::<16, 128, 128, false, false, 2>(params, mb, nb, pitch, map, output_map,
            0, 0, input_stride, if paired { [up.weights, up.scales] } else { [0; 2] });
    } else if tile.rows == 32 {
        project_tile::<32, 64, 128, false, false, 2>(params, mb, nb, pitch, map, output_map,
            0, 0, input_stride, [0; 2]);
    } else {
        project_tile::<16, 64, 128, false, false, 2>(params, mb, nb, pitch, map, output_map,
            0, 0, input_stride, [0; 2]);
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_gate(g: crate::fp8_abi::RoutedParams) { routed_project(g, 0); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_up(g: crate::fp8_abi::RoutedParams) { routed_project(g, 1); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_down(g: crate::fp8_abi::RoutedParams) { routed_project(g, 2); }

/// Small cohorts need no expert histogram, alignment, or row permutation. Each
/// CTA owns one selected slot and 64 gate/up columns. Two independent
/// four-warp groups retain the canonical gate/up dot order. The epilogue
/// writes activated values over this CTA's own gate columns; quantization
/// moves to fp8_quantize_activated because an N64 tile is half a 128 scale
/// group and its other half belongs to a neighboring CTA.
/// Grid(intermediate/64, selected+shared, tokens), block256, shared41216.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_single_gate_up(
    g: crate::fp8_abi::RoutedParams, _middle: u64, _scales: u64, shared: u32,
) {
    let p = g.grouped;
    let token = _block_idx_z() as usize;
    if p.route.failed(token) { return; }
    let slot = _block_idx_y() as usize;
    let selected = g.layout.selected as usize;
    let row = token * (selected + shared as usize) + slot;
    let expert = if slot == selected { p.route.experts as usize }
        else { *(p.route.pairs as *const f32).add((token * selected + slot) * 3 + 1) as usize };
    let tid = _thread_idx_x() as usize;
    let projection = tid / 128;
    let d = *(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3 + projection);
    let nb = _block_idx_x() as usize * 64;
    if nb >= d.rows as usize { return; }
    let i = g.layout.intermediate as usize;
    let params = crate::fp8_abi::BlockParams {
        weights: d.weights, weight_scales: d.scales, input: p.input + (token * p.cols as usize) as u64,
        input_scales: p.input_scales + (token * (p.cols as usize / 128) * 4) as u64,
        output: p.output + ((row * 2 * i + projection * i) * 4) as u64,
        rows: 1, outputs: d.rows, cols: d.cols, input_kind: 0, round_output: 1,
    };
    project_tile::<16, 64, 128, false, false, 2>(params, 0, nb, 2 * i,
        core::ptr::null(), core::ptr::null(), (projection * 20608) as u32, 0, d.cols as usize, [0; 2]);
    ptx::sync();
    if tid < 16 {
        let col = nb + tid * 4;
        let input = (p.output as *const f32).add(row * 2 * i + col);
        let mut values = [0.0; 4];
        for j in 0..4 {
            let gate = *input.add(j);
            values[j] = if slot == selected { crate::flash_ops::silu(gate, *input.add(i + j), 1.0, 3) }
                else { rounded(rounded(gate / (1.0 + crate::flash_moe::exp(-gate))) * *input.add(i + j)) };
        }
        for j in 0..4 { *(p.output as *mut f32).add(row * 2 * i + col + j) = values[j]; }
    }
}

/// Quantize activated gate/up rows (stride 2*cols) into contiguous codes and
/// scales. One warp owns each whole 128 group, so the warp-max scale and
/// codes match the fused N128 epilogue exactly; values arrive rounded.
/// Grid ceil(rows*(cols/128)/8), block256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_quantize_activated(input: *const f32, output: *mut u8,
    scales: *mut f32, rows: u32, cols: u32) {
    let tid = _thread_idx_x() as usize;
    let group = _block_idx_x() as usize * 8 + tid / 32;
    let groups = cols as usize / 128;
    if group >= rows as usize * groups { return; }
    let (row, at) = (group / groups, group % groups * 128 + tid % 32 * 4);
    let x = input.add(row * 2 * cols as usize + at);
    let (word, scale) = quantize4([*x, *x.add(1), *x.add(2), *x.add(3)]);
    *output.add(row * cols as usize + at).cast::<u32>() = word;
    if tid % 32 == 0 { *scales.add(group) = scale; }
}

/// Direct selected-slot down projection. Each output retains its K128 sum
/// order and writes the same ascending-expert slot as the multi-row pipeline.
/// Grid(hidden/64, selected+shared, tokens), block128, shared20608.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_single_down(g: crate::fp8_abi::RoutedParams, shared: u32) {
    let p = g.grouped;
    let token = _block_idx_z() as usize;
    let slot = _block_idx_y() as usize;
    let selected = g.layout.selected as usize;
    let row = token * (selected + shared as usize) + slot;
    let nb = _block_idx_x() as usize * 64;
    if p.route.failed(token) {
        let output = p.output + (g.layout.output_index(token, slot) * p.cols as usize * 4) as u64;
        let tid = _thread_idx_x() as usize;
        if tid < 64 { *(output as *mut f32).add(nb + tid) = 0.0; }
        return;
    }
    let output_slot = g.layout.ordered_slot(slot, |lane|
        *(p.route.pairs as *const f32).add((token * selected + lane) * 3 + 1) as u32);
    let output = p.output + (g.layout.output_index(token, output_slot) * p.cols as usize * 4) as u64;
    let expert = if slot == selected { p.route.experts as usize }
        else { *(p.route.pairs as *const f32).add((token * selected + slot) * 3 + 1) as usize };
    let d = *(p.weights as *const crate::fp8_abi::ExpertProjection).add(expert * 3 + 2);
    let i = g.layout.intermediate as usize;
    let params = crate::fp8_abi::BlockParams {
        weights: d.weights, weight_scales: d.scales, input: p.input + (row * i) as u64,
        input_scales: p.input_scales + (row * (i / 128) * 4) as u64, output,
        rows: 1, outputs: d.rows, cols: d.cols, input_kind: 0, round_output: 0,
    };
    project_tile::<16, 64, 128, false, false, 2>(params, 0, nb, p.cols as usize,
        core::ptr::null(), core::ptr::null(), 0, 0, i, [0; 2]);
}

#[inline(always)]
unsafe fn quantize_small(q: crate::fp8_abi::QuantizeParams, group: usize, lane: usize) {
    if group >= q.rows as usize * (q.cols as usize / 128) { return; }
    let offset = group * 128 + lane * 4;
    let x = (q.input as *const f32).add(offset);
    let (word, scale) = quantize4([rounded(*x), rounded(*x.add(1)), rounded(*x.add(2)), rounded(*x.add(3))]);
    *(q.output as *mut u32).add(offset / 4) = word;
    if lane == 0 { *(q.scales as *mut f32).add(group) = scale; }
}

/// Warp-parallel row routing and quantization overlap in disjoint warps. The
/// next stream launch consumes both; no inter-CTA barrier or polling is needed.
/// Grid ceil(tokens*cols/128/7), block256; tokens<=4.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_route_quantize(p: crate::kernel_params::MoeRouteBatch,
    pairs: *mut f32, q: crate::fp8_abi::QuantizeParams) {
    let tid = _thread_idx_x() as usize;
    if tid < 32 {
        if _block_idx_x() == 0 { crate::moe::route_rows_small(p, pairs, tid); }
    } else { quantize_small(q, _block_idx_x() as usize * 7 + tid / 32 - 1, tid % 32); }
}

/// Raw assignment callers retain validation before selected descriptor reads.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_routed_small_quantize(g: crate::fp8_abi::RoutedParams, q: crate::fp8_abi::QuantizeParams) {
    let tid = _thread_idx_x() as usize;
    let group = _block_idx_x() as usize * 8 + tid / 32;
    let blocks = q.cols as usize / 128;
    if group >= q.rows as usize * blocks { return; }
    if group % blocks == 0 && tid % 32 == 0 {
        let token = group / blocks;
        for slot in 0..g.layout.selected as usize {
            let pair = (g.grouped.route.pairs as *const f32).add((token * g.layout.selected as usize + slot) * 3);
            if *pair != token as f32 || !crate::fp8_abi::valid_route(*pair, *pair.add(1), *pair.add(2), q.rows, g.grouped.route.experts) {
                asm!("atom.global.or.b32 {old}, [{p}], 1;", old=out(reg32)_, p=in(reg64)(g.grouped.route.error + token as u64 * 4));
            }
        }
    }
    quantize_small(q, group, tid % 32);
}
