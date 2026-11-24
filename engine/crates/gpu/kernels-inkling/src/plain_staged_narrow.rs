//! The narrow staged BF16 row product (`staged_narrow`): the entry `resident_launch_plain_staged_narrow` of the Inkling
//! image (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_STAGED_NARROW`. Three unsafe sites,
//! counted: `unsafe fn replay`, its block, `unsafe fn warp`.
//!
//! One launch gives the product of ONE matrix of few rows with one input row: each output is ONE ascending chain over
//! the columns from 0.0, every product rounded by its own `mul.rn.f32` before its `add.rn.f32`, no fused multiply-add:
//! the chain of the row product (`plain_staged.rs`), so the same words. A weight is read as its 16-bit BF16 pattern and
//! widened by a shift of the pattern (exact).
//!
//! The host checks every extent and every alignment before it records a launch (`staged_narrow::check`).
//!
//! A one-warp block owns 4 consecutive rows. A stage holds those 4 rows x 256 columns of BF16 (2,048 bytes of weights)
//! and the matching 256 FP32 inputs, copied by all 32 lanes with coalesced 16-byte `cp.async`, four stages deep
//! (12,544 bytes of shared memory). Lanes 0 to 3 run the chain of their own row from shared memory; each loads the
//! next 8 values before it multiplies the current 8, so the loads wait on no add.
use core::arch::asm;

/// Rows of a block, columns of a stage, stages in flight.
const R: usize = crate::staged_narrow::ROWS_A_BLOCK;
const CH: usize = crate::staged_narrow::STAGE_COLS;
const NS: usize = crate::staged_narrow::STAGES;
/// Shared bytes of one weight row in a stage: 256 BF16 (512 bytes) and a 16-byte pad, so the 4 rows that the 4 chain
/// lanes read at one step start in distinct banks.
const ROW_BYTES: usize = CH * 2 + 16;
const INPUT_AT: usize = R * ROW_BYTES;
const STAGE_BYTES: usize = INPUT_AT + CH * 4;
const BLOCK_BYTES: usize = NS * STAGE_BYTES;
/// Groups of 8 values (16 bytes) in a row of a stage: one a lane. Groups of 4 input words in a stage: two a lane.
const GROUPS: usize = CH / 8;
const _: () = assert!(GROUPS == 32 && CH / 4 == 64 && crate::staged_narrow::THREADS == 32 && R <= 32);
const _: () = assert!(BLOCK_BYTES == 12_544 && ROW_BYTES % 16 == 0 && STAGE_BYTES % 16 == 0 && INPUT_AT % 16 == 0);

/// The replay of one recorded command of this kind: `table[index]` holds five words, the addresses of the weights
/// (rows x cols BF16 patterns), of the input (cols FP32 values) and of the output (rows words), then rows and cols.
/// A launch that the rule `staged_narrow::admitted` does not take stops the block before any read.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_staged_narrow`): `table` and `index` are the
/// arguments the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::project_staged_narrow_into, after
    // staged_narrow::check passed for its five words. That check holds: 1 to 512 rows and whole stages of 256 columns
    // up to 8192, the weights to exactly rows x cols BF16 values, the input to exactly cols words, the output to
    // exactly rows words that meet no operand, the weights and the input to 16-byte addresses. All three live through
    // completion. The rule staged_narrow::admitted is asked here again before the first read (shape, one-warp
    // blocks, alignment); a launch that fails it stops. The grid has rows / 4 blocks rounded up, so `first` is below
    // rows in every block of a recorded launch; it is checked anyway. Lane r of a block stores at most one output
    // word, of row first + r, only when r < 4 and that row is below rows: distinct rows, one writer each.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_STAGED_NARROW { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (weights, input, output) = (*words as *const u16, *words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, cols) = (*words.add(3) as usize, *words.add(4) as usize);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let lane = core::arch::nvptx::_thread_idx_x() as usize;
        if !crate::staged_narrow::admitted(rows, cols, weights as usize, input as usize, threads) { core::arch::nvptx::trap(); }
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        let first = core::arch::nvptx::_block_idx_x() as usize * R;
        if first < rows {
            let sum = warp(weights, input, cols, first, rows, smem);
            if lane < R && first + lane < rows { *output.add(first + lane) = sum; }
        }
    }
}

/// The dot product of row `first_row + lane` for lanes 0 to 3 whose row is below `rows` (0.0 for the other lanes,
/// which the caller does not store). `smem` is the block's NS stages.
#[inline(always)]
unsafe fn warp(weights: *const u16, input: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> f32 {
    // SAFETY: Private to this file; its one call site is in `replay`, which reaches it with the WHOLE warp of a
    // one-warp block (every lane, no early exit: `first_row < rows` is the same on all lanes), staged_narrow::admitted
    // true (cols whole stages of 256 up to 8192, rows 1 to 512, 16-byte aligned weights and input), and `smem` the
    // start of the block's BLOCK_BYTES of shared memory, which this warp owns alone. Every lane executes every copy
    // issue, commit, wait and barrier. Global copies are made only for rows below `rows` and columns below `cols`
    // (cols is a multiple of 256, so a 16-byte group of 8 weight values or of 4 input words is whole when its first
    // column is below cols): lane l copies group l of each of the 4 rows (into the row's 512 bytes of the stage) and
    // input groups l and l + 32 (into the stage's 1,024 input bytes). A chain lane reads shared memory only of its own
    // row and of the stage's input, at a group below 32, after cp.async.wait_group and the warp barrier completed the
    // stage. A barrier after each stage keeps a stage from being refilled while any lane still reads it. No global
    // memory is written.
    let lane = (core::arch::nvptx::_thread_idx_x() as usize) & 31;
    let live = lane < R && first_row + lane < rows;
    macro_rules! issue {($chunk:expr,$stage:expr)=>{{
        let col = $chunk * CH + lane * 8;
        let mut r = 0;
        while r < R {
            let g = first_row + r;
            if g < rows && col < cols { crate::ptx::copy16($stage + (r * ROW_BYTES + lane * 16) as u32, weights.add(g * cols + col).cast()); }
            r += 1;
        }
        let mut half = 0;
        while half < 2 {
            let group = lane + 32 * half;
            if $chunk * CH + group * 4 < cols { crate::ptx::copy16($stage + (INPUT_AT + group * 16) as u32, input.add($chunk * CH + group * 4).cast()); }
            half += 1;
        }
    }};}
    let chunks = cols / CH;
    let mut s = 0;
    while s + 1 < NS {
        if s < chunks { issue!(s, smem + (s * STAGE_BYTES) as u32); }
        crate::ptx::commit();
        s += 1;
    }
    let mut sum = 0.0f32;
    let mut c = 0;
    while c < chunks {
        let ahead = c + NS - 1;
        if ahead < chunks { issue!(ahead, smem + ((ahead % NS) * STAGE_BYTES) as u32); }
        crate::ptx::commit();
        asm!("cp.async.wait_group {n};", n = const (NS - 1));
        asm!("bar.warp.sync 0xffffffff;");
        let stage = smem + ((c % NS) * STAGE_BYTES) as u32;
        if live {
            let row_at = stage + (lane * ROW_BYTES) as u32;
            let input_at = stage + INPUT_AT as u32;
            let mut w = crate::ptx::load4(row_at);
            let mut x0 = crate::ptx::load4(input_at);
            let mut x1 = crate::ptx::load4(input_at + 16);
            let mut q = 0;
            while q < GROUPS {
                // The next 8 values first (the last group loads itself again, inside the stage).
                let next = if q + 1 < GROUPS { q + 1 } else { q };
                let w_next = crate::ptx::load4(row_at + (next * 16) as u32);
                let x0_next = crate::ptx::load4(input_at + (next * 32) as u32);
                let x1_next = crate::ptx::load4(input_at + (next * 32 + 16) as u32);
                let mut e = 0;
                while e < 8 {
                    let word = w[e / 2];
                    let bits = if e & 1 == 0 { word << 16 } else { word & 0xffff_0000 };
                    let x = if e < 4 { x0[e] } else { x1[e - 4] };
                    let product: f32;
                    asm!("mul.rn.f32 {product}, {w}, {x};", product = out(reg32) product, w = in(reg32) f32::from_bits(bits), x = in(reg32) f32::from_bits(x), options(nomem, nostack));
                    asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) product, options(nomem, nostack));
                    e += 1;
                }
                (w, x0, x1) = (w_next, x0_next, x1_next);
                q += 1;
            }
        }
        asm!("bar.warp.sync 0xffffffff;");
        c += 1;
    }
    sum
}
