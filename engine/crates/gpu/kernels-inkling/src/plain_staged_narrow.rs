//! The narrow staged BF16 row product (`staged_narrow`): the entry `resident_launch_plain_staged_narrow` of the Inkling
//! image (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_STAGED_NARROW`. Three unsafe sites,
//! counted: `unsafe fn replay`, its block, `unsafe fn warp`.
//!
//! Computes `Y = X * W^T` for 1 to 4 narrow BF16 matrices (`rows <= 512`, `cols % 256 == 0`) with one FP32 input row
//! each, using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`A = X`, `B = W`) in ascending K order.
//! The narrow few-row and prefill products use the same order.
//!
//! One warp owns four consecutive rows. It streams the weights and the FP32 input through a six-stage `cp.async`
//! ring of 256-column stages, so five stages are in flight while it runs the MMAs of one.
use core::arch::asm;
use crate::tc_ptx::{cvt_bf16x2, load_u32, mma_m16n8k16_bf16};

/// One warp streams four weight rows and one FP32 input row over 256-column stages, `NS` stages in a ring.
const R: usize = crate::staged_narrow::ROWS_A_BLOCK;
const BK: usize = crate::staged_narrow::STAGE_COLS;
const NS: usize = 6;
const ROW_BYTES: usize = BK * 2 + 16;
const INPUT_AT: usize = R * ROW_BYTES;
const STAGE_BYTES: usize = INPUT_AT + BK * 4;
pub(crate) const BLOCK_BYTES: usize = NS * STAGE_BYTES;
const _: () = assert!(R == 4 && BK == 256 && crate::staged_narrow::THREADS == 32 && BLOCK_BYTES == 18_816);

/// The replay of one recorded command of this kind: `table[index]` holds the addresses of the weights (rows x cols
/// BF16 patterns), of the input (cols FP32 values) and of the output, then rows, cols and the count of matrices G of
/// the grouped form (`staged_narrow::group_word`: matrix g >= 1 in words 4 + 2g and 5 + 2g). The output is
/// `[G][rows]`. A launch that the rule `staged_narrow::admitted` does not take stops the block before any read.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_staged_narrow`): `table` and `index` are the
/// arguments the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::project_staged_narrow_group_into (the one-matrix
    // call `project_staged_narrow_into` is a group of one), after staged_narrow::check_group passed for its words: 1 to
    // GROUPS_MAX matrices; for each matrix g, 1 to 512 rows and whole stages of 256 columns up to 8192, its weights
    // exactly rows x cols BF16 values, its input exactly cols words, its part of the output (rows words from word
    // g x rows) meeting none of its operands, its weights and input at 16-byte addresses; the whole output of G x rows
    // words meets no weights and no input of any matrix. Every operand lives through completion. The count is read
    // before any other word of a group and a count outside 1..=GROUPS_MAX traps, so only the 6 + 2 (G - 1) written
    // words are read (all inside the 1,024-byte payload, host test). The rule staged_narrow::admitted is asked again
    // for the block's own matrix before the first read (shape, one-warp blocks, alignment); a launch that fails it
    // stops. `first < rows` is uniform across the warp. Only lanes 0 and 1 write the 4 distinct output words
    // `group * rows + first + 0..4` (guarded by `< rows`).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_STAGED_NARROW { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let output = *words.add(2) as *mut f32;
        let (rows, cols, groups) = (*words.add(3) as usize, *words.add(4) as usize, *words.add(5) as usize);
        if !crate::staged_narrow::groups_admitted(groups) || rows == 0 { core::arch::nvptx::trap(); }
        let per_matrix = crate::staged_narrow::blocks(rows) as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let group = block / per_matrix;
        if group >= groups { core::arch::nvptx::trap(); }
        let weights = *words.add(crate::staged_narrow::group_word(group, false)) as *const u16;
        let input = *words.add(crate::staged_narrow::group_word(group, true)) as *const f32;
        let flags = *words.add(crate::staged_narrow::NARROW_FLAGS_WORD);
        let round_out = (flags & crate::staged_narrow::NARROW_ROUND_OUT) != 0;
        let dual = (flags & crate::staged_narrow::NARROW_DUAL_INPUT) != 0;
        if (flags & !(crate::staged_narrow::NARROW_ROUND_OUT | crate::staged_narrow::NARROW_DUAL_INPUT)) != 0 {
            core::arch::nvptx::trap();
        }
        let input2 = if dual { *words.add(crate::staged_narrow::NARROW_INPUT2_WORD + group) as *const f32 } else { core::ptr::null() };
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        if !crate::staged_narrow::admitted(rows, cols, weights as usize, input as usize, threads)
            || (dual && (cols % (2 * BK) != 0 || input2.is_null() || (input2 as usize) & 15 != 0))
        {
            core::arch::nvptx::trap();
        }
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        let first = block % per_matrix * R;
        if first < rows {
            let acc = warp(weights, input, input2, cols, first, rows, smem);
            if tid < 2 {
                let (r0, r1) = (first + 2 * tid, first + 2 * tid + 1);
                if r0 < rows { *output.add(group * rows + r0) = if round_out { crate::fixed_math::rn(acc[0]) } else { acc[0] }; }
                if r1 < rows { *output.add(group * rows + r1) = if round_out { crate::fixed_math::rn(acc[1]) } else { acc[1] }; }
            }
        }
    }
}

/// One warp runs the ascending-K `m16n8k16` dot products of four rows with `input[0..cols]` (or `[input, input2]`
/// when `input2` is non-null). Prefetches static weight stages `0..NS-1` BEFORE `griddepcontrol.wait;` so PDL
/// overlaps static weight traffic with the preceding norm/mix/stream-op. `smem` is the block's `BLOCK_BYTES` buffer.
#[inline(always)]
pub(crate) unsafe fn warp(weights: *const u16, input: *const f32, input2: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> [f32; 4] {
    // SAFETY: both replay callers admit 32 threads, four rows per block,
    // 16-byte aligned inputs and weights, and whole 256-column stages (the
    // shape rule; dual input also checks `cols % 512 == 0` and 16-byte `input2`).
    // Each stage holds four padded BF16 weight rows and the 256 FP32 inputs of its columns.
    // Static weight stages `0..NS-1` are committed in group 0 before `griddepcontrol.wait`;
    // dynamic input stages `0..NS-1` follow in `NS-1` separate commits, so `cp.async.wait_group (NS - 1)`
    // after each loop step's commit guarantees both the weights and inputs of stage `s` are complete.
    // Every lane crosses the warp barrier before and after each stage's MMAs. No row past `rows` is read.
    // Input is rounded via `cvt.rn.bf16x2.f32`, and MMAs accumulate in strictly ascending K order from +0.0.
    let lane = core::arch::nvptx::_thread_idx_x() as usize;
    let stages = cols / BK;
    let half_stages = stages / 2;
    let dual = !input2.is_null();
    let issue_w = |s: usize| {
        let slot = smem + ((s % NS) * STAGE_BYTES) as u32;
        for row in 0..R {
            let source = weights.add((first_row + row).min(rows - 1) * cols + s * BK);
            crate::ptx::copy16(slot + (row * ROW_BYTES + lane * 16) as u32, source.add(lane * 8).cast());
        }
    };
    let issue_x = |s: usize| {
        let slot = smem + ((s % NS) * STAGE_BYTES) as u32;
        let src = if dual && s >= half_stages { input2.add((s - half_stages) * BK) } else { input.add(s * BK) };
        for part in [lane, lane + 32] {
            crate::ptx::copy16(slot + (INPUT_AT + part * 16) as u32, src.add(part * 4).cast());
        }
    };
    const _: () = assert!(BK == 32 * 8 && BK * 4 == 64 * 16 && NS == 6);
    let mut s = 0;
    while s + 1 < NS {
        if s < stages { issue_w(s); }
        s += 1;
    }
    crate::ptx::commit();
    asm!("griddepcontrol.wait;", options(nostack));
    s = 0;
    while s + 1 < NS {
        if s < stages { issue_x(s); }
        crate::ptx::commit();
        s += 1;
    }
    let mut acc = [0.0; 4];
    for s in 0..stages {
        if s + NS - 1 < stages {
            issue_w(s + NS - 1);
            issue_x(s + NS - 1);
        }
        crate::ptx::commit();
        asm!("cp.async.wait_group {n};", n = const (NS - 1));
        asm!("bar.warp.sync 0xffffffff;");
        let slot = smem + ((s % NS) * STAGE_BYTES) as u32;
        let x_at = slot + (INPUT_AT + (lane % 4) * 8) as u32;
        let w_at = slot + ((lane / 4 % R) * ROW_BYTES + (lane % 4) * 4) as u32;
        for k in (0..BK).step_by(16) {
            let lo = crate::ptx::load2(x_at + (k * 4) as u32);
            let hi = crate::ptx::load2(x_at + (k * 4 + 32) as u32);
            let w = w_at + (k * 2) as u32;
            acc = mma_m16n8k16_bf16([cvt_bf16x2(f32::from_bits(lo[0]), f32::from_bits(lo[1])), 0,
                                     cvt_bf16x2(f32::from_bits(hi[0]), f32::from_bits(hi[1])), 0],
                                    [load_u32(w), load_u32(w + 16)], acc);
        }
        asm!("bar.warp.sync 0xffffffff;");
    }
    acc
}
