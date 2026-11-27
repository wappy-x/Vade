//! Plain projection rows whose weights are stored as BF16 (the router gate 144 x 4096, kv_b 16384 x 512, indexer wq_b 2048 x 1536, f_b and g_b 4096 x 128 per rank).
//! The arithmetic is the one of `plain_warp::rows_warp` and `serial_projection::plain_row!`: each output is ONE ascending
//! chain over the columns from 0.0, every product rounded by its own `mul.rn.f32` before its `add.rn.f32`, no fused
//! multiply-add. The only change is the load: a weight is read as its 16-bit BF16 pattern and widened by the loader's
//! operation (`tp/src/resident_weights.rs`: `f32::from_bits(u32::from(u16) << 16)`), so the f32 the chain multiplies is
//! bit for bit the f32 the FP32 copy holds. The widening is exact (a shift of the pattern, no rounding).
//!
//! One warp owns 32 consecutive rows. A stage holds 32 rows x 64 columns of BF16 (4,096 bytes of weights, the same bytes in
//! flight as the FP32 tile of 32 columns) and the matching 64 FP32 inputs, copied with coalesced 16-byte `cp.async` and
//! eight stages deep in a one-warp block (two stages per warp in a four-warp block of 128 threads, the same 38,912 bytes of
//! shared memory); each lane runs the chain of its own row from shared memory.
use core::arch::asm;

const CH: usize = 64;
/// Stages of a warp alone in its block, and of each warp of a four-warp block (the same shared bytes in both).
const NS_ALONE: usize = 8;
const NS_BLOCK: usize = 2;
/// Shared bytes of one weight row in a stage: 64 BF16 (128 bytes) and a 16-byte pad so eight neighbouring rows read
/// 16 bytes each from distinct banks.
const ROW_BYTES: usize = CH * 2 + 16;
const INPUT_AT: usize = 32 * ROW_BYTES;
const STAGE_BYTES: usize = INPUT_AT + CH * 4;
const BLOCK_BYTES: usize = NS_ALONE * STAGE_BYTES;
const _: () = assert!(BLOCK_BYTES == 4 * NS_BLOCK * STAGE_BYTES);
const _: () = assert!(BLOCK_BYTES <= 48 * 1024);

/// BF16 pattern to the f32 the loader makes: the high half word of the f32 is the pattern, the low half is zero.
#[inline(always)]
fn widen(bits: u32) -> f32 { f32::from_bits(bits << 16) }

/// Rows of the plain projection with BF16 weights. `weights` points at rows x cols BF16 patterns.
#[no_mangle]
pub extern "ptx-kernel" fn project_bf16_ordered(weights: *const u16, input: *const f32, output: *mut f32, rows: u32, cols: u32) {
    resident_project_bf16_ordered(weights, input, output, rows, cols)
}
/// The body, shared by the direct entry and the recorded replay entry (`resident_launch_plain_bf16`).
#[inline(always)]
pub(crate) fn resident_project_bf16_ordered(weights: *const u16, input: *const f32, output: *mut f32, rows: u32, cols: u32) {
    // SAFETY: The caller checks contiguous BF16 weights of rows x cols, an input of cols FP32 values and a disjoint output
    // of rows words, all live through completion. Each lane owns at most one row. When plain_admission::bf16_admitted
    // holds (a block of one or four whole warps, cols a multiple of 8 in 32..=8192, rows * cols within the swept envelope,
    // weights and input 16-byte aligned) the warp tile runs on the warp's own slice of the block's shared memory;
    // otherwise every thread with a row below `rows` runs the scalar chain over its own row, reading 2 bytes at a time
    // inside the row. No global memory is written except the one output word of an owned row.
    unsafe {
        let (rows, cols) = (rows as usize, cols as usize);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        if crate::plain_admission::bf16_admitted(rows, cols, weights as usize, input as usize, threads) {
            let smem = crate::ptx::shared::<BLOCK_BYTES>();
            let first = core::arch::nvptx::_block_idx_x() as usize * threads + tid / 32 * 32;
            if first < rows {
                let sum = if threads == crate::plain_admission::NARROW_THREADS {
                    rows_warp_bf16::<NS_ALONE>(weights, input, cols, first, rows, smem)
                } else {
                    rows_warp_bf16::<NS_BLOCK>(weights, input, cols, first, rows, smem + (tid / 32 * NS_BLOCK * STAGE_BYTES) as u32)
                };
                if first + tid % 32 < rows { *output.add(first + tid % 32) = sum; }
            }
        } else {
            let row = crate::index();
            if row < rows {
                let mut sum = 0.0f32;
                let mut column = 0;
                while column < cols {
                    let bits = u32::from(*weights.add(row * cols + column));
                    let product: f32;
                    asm!("mul.rn.f32 {product}, {w}, {x};", product = out(reg32) product, w = in(reg32) widen(bits), x = in(reg32) *input.add(column), options(nomem, nostack));
                    asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) product, options(nomem, nostack));
                    column += 1;
                }
                *output.add(row) = sum;
            }
        }
    }
}

/// The dot product of row `first_row + lane` for every lane whose row is below `rows` (0.0 for the others, which the caller does not store).
/// `smem` is this warp's own NS stages.
#[inline(always)]
unsafe fn rows_warp_bf16<const NS: usize>(weights: *const u16, input: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> f32 {
    // SAFETY: Private to this file; its only two call sites are in `project_bf16_ordered`, which reach it with a WHOLE warp (every lane, no early exit),
    // plain_admission::bf16_admitted true (a listed step shape in its own geometry, 16-byte aligned weights and input, cols a multiple of 8 in 32..=8192),
    // first_row below rows and the same on all lanes, and `smem` the start of NS * STAGE_BYTES of shared memory that belongs to this warp ALONE:
    // NS = 8 in a one-warp block (NS_ALONE, the warp owns the block's whole 38,912-byte array) and NS = 2 in a four-warp block (NS_BLOCK, each warp
    // owns its own disjoint 9,728-byte slice at tid / 32 * NS_BLOCK * STAGE_BYTES). Every lane executes every copy issue, commit, wait and barrier;
    // global copies and prefetches are made only for rows below `rows` and columns below `cols` (cols is a multiple of 8, so a 16-byte weight group is
    // whole when its first column is below cols), and a lane reads shared memory only of its own row and the shared input row, after
    // cp.async.wait_group and the warp barrier completed the stage. A barrier after each chunk keeps a stage from being refilled while any lane
    // still reads it. No global memory is written.
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
    let chunks = cols.div_ceil(CH);
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
        macro_rules! octet {($q:expr)=>{{
            let w = crate::ptx::load4(stage + (lane * ROW_BYTES + $q * 16) as u32);
            let x0 = crate::ptx::load4(stage + (INPUT_AT + $q * 32) as u32);
            let x1 = crate::ptx::load4(stage + (INPUT_AT + $q * 32 + 16) as u32);
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
        }};}
        if !live {
            // Nothing was copied for this lane's row; read nothing.
        } else if (c + 1) * CH <= cols {
            let mut q = 0;
            while q < CH / 8 { octet!(q); q += 1; }
        } else {
            let mut q = 0;
            while q < (cols - c * CH) / 8 { octet!(q); q += 1; }
        }
        asm!("bar.warp.sync 0xffffffff;");
        c += 1;
    }
    sum
}
