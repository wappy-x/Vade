//! The staged BF16 row product (`staged_product`): the entry `resident_launch_plain_staged` of the Inkling image
//! (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_STAGED`. The same body was measured in the
//! macro form of Ruling R6f on branch ink-prod 522398a59d; here it is written out, with its three unsafe sites counted (`unsafe fn replay`, its block, `unsafe fn warp`).
//!
//! One launch gives the product of ONE matrix of BF16 values with one input row: each output is ONE ascending chain
//! over the columns from 0.0, every product rounded by its own `mul.rn.f32` before its `add.rn.f32`, no fused
//! multiply-add. A weight is read as its 16-bit BF16 pattern and widened by a shift of the pattern (exact), so the
//! f32 the chain multiplies is bit for bit the f32 a 32-bit copy holds.
//!
//! The host checks every extent and every alignment before it records a launch (`staged_product::check`).
//!
//! One warp owns 32 consecutive rows. A stage holds 32 rows x 64 columns of BF16 (4,096 bytes of weights) and the
//! matching 64 FP32 inputs, copied with coalesced 16-byte `cp.async`: eight stages deep in a one-warp block, two
//! stages a warp in a four-warp block of 128 threads (the same 38,912 bytes of shared memory). Each lane runs the
//! chain of its own row from shared memory.
use core::arch::asm;

/// Columns of a stage.
const CH: usize = crate::staged_product::STAGE;
/// Stages of a warp alone in its block, and of each warp of a four-warp block (the same shared bytes in both).
const NS_ALONE: usize = 8;
const NS_BLOCK: usize = 2;
/// Shared bytes of one weight row in a stage: 64 BF16 (128 bytes) and a 16-byte pad so eight neighbouring rows read
/// 16 bytes each from distinct banks.
const ROW_BYTES: usize = CH * 2 + 16;
const INPUT_AT: usize = 32 * ROW_BYTES;
const STAGE_BYTES: usize = INPUT_AT + CH * 4;
const BLOCK_BYTES: usize = NS_ALONE * STAGE_BYTES;
const _: () = assert!(BLOCK_BYTES == 4 * NS_BLOCK * STAGE_BYTES && crate::staged_product::TILE_THREADS == 4 * 32 && crate::staged_product::NARROW_THREADS == 32);
const _: () = assert!(BLOCK_BYTES == 38_912 && ROW_BYTES % 16 == 0 && STAGE_BYTES % 16 == 0);

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
    // geometry of the launch, alignment); a launch that fails it stops. Each lane stores at most one output word, of
    // its own row, and only when that row is below rows. The warp tile runs on the warp's own slice of the block's
    // shared memory.
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
            let sum = if threads == crate::staged_product::NARROW_THREADS {
                warp::<NS_ALONE>(weights, input, cols, first, rows, smem)
            } else {
                warp::<NS_BLOCK>(weights, input, cols, first, rows, smem + (tid / 32 * NS_BLOCK * STAGE_BYTES) as u32)
            };
            if first + tid % 32 < rows { *output.add(first + tid % 32) = sum; }
        }
    }
}

/// The dot product of row `first_row + lane` for every lane whose row is below `rows` (0.0 for the others, which the
/// caller does not store). `smem` is this warp's own NS stages.
#[inline(always)]
unsafe fn warp<const NS: usize>(weights: *const u16, input: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> f32 {
    // SAFETY: Private to this file; its only two call sites are in `replay`, which reach it with a WHOLE warp (every
    // lane, no early exit), staged_product::admitted true (cols whole stages of 64 up to 8192, rows x cols inside the
    // envelope, 16-byte aligned weights and input, the geometry of the entry), first_row below rows and the same on
    // all lanes, and `smem` the start of NS * STAGE_BYTES of shared memory that belongs to this warp ALONE: NS = 8 in
    // a one-warp block (the warp owns the block's whole 38,912-byte array) and NS = 2 in a four-warp block (each warp
    // owns its own disjoint 9,728-byte slice at tid / 32 * NS_BLOCK * STAGE_BYTES). Every lane executes every copy
    // issue, commit, wait and barrier; global copies and prefetches are made only for rows below `rows` and columns
    // below `cols` (cols is a multiple of 64, so a 16-byte weight group of 8 values and a 16-byte input group of 4
    // words are whole when their first column is below cols), and a lane reads shared memory only of its own row and
    // of the shared input row, after cp.async.wait_group and the warp barrier completed the stage. A barrier after
    // each stage keeps a stage from being refilled while any lane still reads it. No global memory is written.
    let lane = (core::arch::nvptx::_thread_idx_x() as usize) & 31;
    let live = first_row + lane < rows;
    macro_rules! issue {($chunk:expr,$stage:expr)=>{{
        let col = $chunk * CH + (lane % 8) * 8;
        let mut i = 0;
        while i < 8 {
            let row = i * 4 + lane / 8;
            let g = first_row + row;
            if g < rows && col < cols {
                crate::ptx::copy16($stage + (row * ROW_BYTES + (lane % 8) * 16) as u32, weights.add(g * cols + col).cast());
            }
            i += 1;
        }
        if lane < 16 && $chunk * CH + lane * 4 < cols {
            crate::ptx::copy16($stage + (INPUT_AT + lane * 16) as u32, input.add($chunk * CH + lane * 4).cast());
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
        let pc = (c + 4 * NS) * CH;
        let g = first_row + lane;
        if pc < cols && g < rows { asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights.add(g * cols + pc) as u64); }
        asm!("cp.async.wait_group {n};", n = const (NS - 1));
        asm!("bar.warp.sync 0xffffffff;");
        let stage = smem + ((c % NS) * STAGE_BYTES) as u32;
        if live {
            let mut q = 0;
            while q < CH / 8 {
                let w = crate::ptx::load4(stage + (lane * ROW_BYTES + q * 16) as u32);
                let x0 = crate::ptx::load4(stage + (INPUT_AT + q * 32) as u32);
                let x1 = crate::ptx::load4(stage + (INPUT_AT + q * 32 + 16) as u32);
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
                q += 1;
            }
        }
        asm!("bar.warp.sync 0xffffffff;");
        c += 1;
    }
    sum
}
