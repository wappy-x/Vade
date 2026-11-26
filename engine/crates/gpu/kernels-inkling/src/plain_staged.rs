//! The staged BF16 row product (`staged_product`): the entry `resident_launch_plain_staged` of the Inkling image
//! (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_STAGED`. Its three unsafe sites stay counted
//! (`unsafe fn replay`, its block, `unsafe fn warp`).
//!
//! Computes `Y = X * W^T` for one FP32 input row `X: [1, K]` (rounded once to BF16 via `cvt.rn.bf16x2.f32`) and
//! `W: [rows, K]` (packed BF16, row-major) using SM121a Weights-as-A `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
//! in strictly ascending K order (`k = 0, 16, 32, ..., K - 16`) from `+0.0`, zero FP atomics, zero split-K.
//! Uses the same K order as the wide form of `PLAIN_STAGED_FEW` and `PLAIN_TC_ROWS`.
//!
//! One warp owns 32 consecutive weight rows (two 16-row `m16n8k16` tiles). A stage holds 32 rows x 64 BF16
//! (4,096 bytes, 128B full cache-line coalesced `cp.async`, XOR-swizzled for zero-bank-conflict `ldmatrix.x4`):
//! 8 stages in a one-warp block, 2 stages per warp in a four-warp block (32,768 bytes SMEM total, 3 blocks/SM).
use core::arch::asm;
use crate::tc_ptx::{cvt_bf16x2, ld_global_nc_v2_u32, ldmatrix_x4, mma_m16n8k16_bf16, st_shared_zero16};

/// Columns of a stage (4 x k16 substeps = 128 bytes per row, full cache line).
const BK: usize = crate::staged_product::STAGE;
/// Stages of a warp alone in its block, and of each warp of a four-warp block (the same shared bytes in both).
const NS_ALONE: usize = 8;
const NS_BLOCK: usize = 2;
/// Shared bytes of one weight row in a stage: 64 BF16 = 128 bytes (8 chunks of 16 bytes, XOR-swizzled).
const ROW_BYTES: usize = BK * 2;
const STAGE_BYTES: usize = 32 * ROW_BYTES;
const BLOCK_BYTES: usize = NS_ALONE * STAGE_BYTES;
const _: () = assert!(BK == 64 && BLOCK_BYTES == 4 * NS_BLOCK * STAGE_BYTES);
const _: () = assert!(crate::staged_product::TILE_THREADS == 4 * 32 && crate::staged_product::NARROW_THREADS == 32);
const _: () = assert!(BLOCK_BYTES == 32_768 && ROW_BYTES == 128 && STAGE_BYTES == 4_096);

/// Zero-bank-conflict XOR-swizzled byte offset within a 128-byte-stride BF16 tile (`row` in 0..32, `chunk` in 0..8).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 128 + ((chunk ^ (row & 7)) * 16)) as u32
}

/// The replay of one recorded command of this kind: `table[index]` holds five words, the addresses of the weights
/// (rows x cols BF16 patterns), of the input (cols FP32 values) and of the output (rows words), then rows and cols.
/// A launch that the rule `staged_product::admitted` does not take stops the block before any read.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_staged`): `table` and `index` are the arguments
/// the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::project_staged_into, after
    // staged_product::check passed for its five words. That check holds: the weights to exactly
    // rows x cols BF16 values, the input to exactly cols words, the output to exactly rows words that meet no operand,
    // the weights and the input to 16-byte addresses, cols to whole stages of 64 inside the envelope. All three live
    // through completion. The rule staged_product::admitted is asked here again before the first read (shape, the
    // geometry of the launch, alignment); a launch that fails it stops. Each warp owns 32 distinct rows; lanes with
    // `s == 0` store at most four distinct output words (`first + g`, `+ 8`, `+ 16`, `+ 24`), only when below `rows`.
    // Each warp runs on its own disjoint slice of the block's static `BLOCK_BYTES` shared memory.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_STAGED { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (weights, input, output) = (*words as *const u16, *words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, cols) = (*words.add(3) as usize, *words.add(4) as usize);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        if !crate::staged_product::admitted(rows, cols, weights as usize, input as usize, threads) { core::arch::nvptx::trap(); }
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        let first = core::arch::nvptx::_block_idx_x() as usize * threads + tid / 32 * 32;
        if first < rows {
            let (c_lo, c_hi) = if threads == crate::staged_product::NARROW_THREADS {
                warp::<NS_ALONE>(weights, input, cols, first, rows, smem)
            } else {
                warp::<NS_BLOCK>(weights, input, cols, first, rows, smem + (tid / 32 * NS_BLOCK * STAGE_BYTES) as u32)
            };
            let lane = tid & 31;
            if (lane & 3) == 0 {
                let g = lane >> 2;
                if first + g < rows { *output.add(first + g) = c_lo[0]; }
                if first + g + 8 < rows { *output.add(first + g + 8) = c_lo[2]; }
                if first + 16 + g < rows { *output.add(first + 16 + g) = c_hi[0]; }
                if first + 24 + g < rows { *output.add(first + 24 + g) = c_hi[2]; }
            }
        }
    }
}

/// Weights-as-A `m16n8k16` BF16 dot products of rows `first_row..first_row+32` with `input[0..cols]`.
/// `smem` is this warp's own `NS * STAGE_BYTES` slice (`BK = 64`, 128B coalesced cache-line transfers).
#[inline(always)]
unsafe fn warp<const NS: usize>(weights: *const u16, input: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> ([f32; 4], [f32; 4]) {
    // SAFETY: Private to this file; its only two call sites are in `replay`, which reach it with a WHOLE warp,
    // `staged_product::admitted` true (`cols` whole stages of 64 up to 8192, `rows * cols` inside the envelope,
    // 16-byte aligned weights and input, matching block geometry), `first_row < rows` uniform across the warp,
    // and `smem` the start of `NS * STAGE_BYTES` of shared memory owned exclusively by this warp (`NS = 8` alone or
    // `NS = 2` per warp in a 4-warp block, both totaling `BLOCK_BYTES = 32,768`). Every lane executes every `cp.async`,
    // commit, wait, `bar.warp.sync`, `ldmatrix_x4`, and `mma_m16n8k16_bf16`. Global reads stay strictly in-bounds and
    // aligned. Out-of-bounds weight rows zero-fill their shared slot. No global memory is written.
    let lane = (core::arch::nvptx::_thread_idx_x() as usize) & 31;
    let r_base = lane >> 3;
    let w_chunk = lane & 7;
    let off0 = swiz128(lane & 15, lane >> 4);
    let off1 = off0 + 2048;
    let (g, s) = (lane >> 2, lane & 3);
    let load_x = g == 0;

    macro_rules! issue {($chunk:expr,$stage:expr)=>{{
        let col = $chunk * BK + w_chunk * 8;
        let mut i = 0usize;
        while i < 8 {
            let r = i * 4 + r_base;
            let dst = $stage + swiz128(r, w_chunk);
            let gr = first_row + r;
            if gr < rows {
                crate::ptx::copy16(dst, weights.add(gr * cols + col).cast());
            } else {
                st_shared_zero16(dst);
            }
            i += 1;
        }
    }};}

    let chunks = cols / BK;
    let mut st = 0usize;
    while st + 1 < NS {
        if st < chunks { issue!(st, smem + (st * STAGE_BYTES) as u32); }
        crate::ptx::commit();
        st += 1;
    }

    let mut c_lo = [0.0f32; 4];
    let mut c_hi = [0.0f32; 4];
    let mut c = 0usize;
    while c < chunks {
        let ahead = c + NS - 1;
        if ahead < chunks { issue!(ahead, smem + ((ahead % NS) * STAGE_BYTES) as u32); }
        crate::ptx::commit();
        let pc = (c + NS) * BK;
        if pc < cols && first_row + lane < rows {
            asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights.add((first_row + lane) * cols + pc) as u64);
        }
        let k_base = c * BK;
        let bx = if load_x {
            let base = k_base + 2 * s;
            let p00 = ld_global_nc_v2_u32(input.add(base).cast());
            let p01 = ld_global_nc_v2_u32(input.add(base + 8).cast());
            let p10 = ld_global_nc_v2_u32(input.add(base + 16).cast());
            let p11 = ld_global_nc_v2_u32(input.add(base + 24).cast());
            let p20 = ld_global_nc_v2_u32(input.add(base + 32).cast());
            let p21 = ld_global_nc_v2_u32(input.add(base + 40).cast());
            let p30 = ld_global_nc_v2_u32(input.add(base + 48).cast());
            let p31 = ld_global_nc_v2_u32(input.add(base + 56).cast());
            [
                [cvt_bf16x2(f32::from_bits(p00[0]), f32::from_bits(p00[1])), cvt_bf16x2(f32::from_bits(p01[0]), f32::from_bits(p01[1]))],
                [cvt_bf16x2(f32::from_bits(p10[0]), f32::from_bits(p10[1])), cvt_bf16x2(f32::from_bits(p11[0]), f32::from_bits(p11[1]))],
                [cvt_bf16x2(f32::from_bits(p20[0]), f32::from_bits(p20[1])), cvt_bf16x2(f32::from_bits(p21[0]), f32::from_bits(p21[1]))],
                [cvt_bf16x2(f32::from_bits(p30[0]), f32::from_bits(p30[1])), cvt_bf16x2(f32::from_bits(p31[0]), f32::from_bits(p31[1]))],
            ]
        } else {
            [[0u32; 2]; 4]
        };
        asm!("cp.async.wait_group {n};", n = const (NS - 1));
        asm!("bar.warp.sync 0xffffffff;");
        let stage = smem + ((c % NS) * STAGE_BYTES) as u32;
        macro_rules! kstep {($ks:literal, $xmask:expr)=>{{
            let a0 = ldmatrix_x4(stage + (off0 ^ $xmask));
            let a1 = ldmatrix_x4(stage + (off1 ^ $xmask));
            c_lo = mma_m16n8k16_bf16(a0, bx[$ks], c_lo);
            c_hi = mma_m16n8k16_bf16(a1, bx[$ks], c_hi);
        }};}
        kstep!(0, 0u32);
        kstep!(1, 32u32);
        kstep!(2, 64u32);
        kstep!(3, 96u32);
        asm!("bar.warp.sync 0xffffffff;");
        c += 1;
    }
    (c_lo, c_hi)
}
