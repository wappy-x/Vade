//! The rows tile of the staged BF16 product (`staged_product`, the part THE ROWS TILE): the entry
//! `resident_launch_plain_staged_rows` of the Inkling image (crate `kernels-inkling`, Ruling R6h), kind
//! `inkling_launch::PLAIN_STAGED_ROWS`. The same body was measured in the macro form of Ruling R6f on branch ink-prod
//! 522398a59d; here it is written out, with its two unsafe sites counted (`unsafe fn replay` and its block).
//!
//! One launch gives the product of ONE matrix of BF16 values with `inputs` input rows, output `[inputs][rows]`.
//! Every output word is ONE ascending FP32 chain over the columns, from 0.0, each product rounded before its add, no
//! fused multiply-add: the word of the staged row product (`plain_staged.rs`) on that input row.
//!
//! The host checks every extent and every alignment before it records a launch (`staged_product::check_rows`).
//!
//! A block takes 64 weight rows and 64 input rows; each of its 256 threads owns 4 weight rows x 4 input rows (16
//! independent chains), so each shared-memory load feeds several multiplies. Columns pass through shared memory in
//! chunks of 32 in ascending order, two stages deep. A thread reads four columns of a weight row as two words and
//! stores the four widened 32-bit values in the stage (the low half shifted up, the high half kept: the value that a
//! 32-bit copy holds). The input rows are copied with 16-byte `cp.async`. Edge tiles skip rows that do not exist;
//! they never pad a chain with extra terms.
//!
//! Index rows (the expert runs of a step): a launch may read input row `t` from row `in[t]` of a larger input and
//! write it to row `out[t]` of a larger output, through two tables of u32 row numbers (a null table is the identity).
//! The body reads each index once and traps on an index at or above the rows of its operand, so a bad table can
//! neither read nor write outside the operands. The words are the same: each one is still the chain of its own input row.
//!
//! Launch: grid = ceil(inputs/64) * ceil(rows/64) blocks (the tile of input rows fastest, so neighbouring blocks
//! share weight lines), 256 threads, no dynamic shared memory.

/// Weight rows and input rows of a block.
const BR: usize = crate::staged_product::ROWS_TILE;
const BT: usize = crate::staged_product::ROWS_TILE;
/// Columns of a chunk, and shared words of one row of a chunk (the columns and a pad of 4 words).
const KC: usize = 32;
const SK: usize = 36;
/// Shared words of one stage, and shared bytes of the two stages of a block.
const STAGE: usize = (BR + BT) * SK;
const BLOCK_BYTES: usize = 2 * STAGE * 4;
const _: () = assert!(crate::staged_product::ROWS_THREADS == BR / 4 * (BT / 4) && crate::staged_product::STAGE % KC == 0 && (BR + BT) * 8 % crate::staged_product::ROWS_THREADS == 0);
const _: () = assert!(BLOCK_BYTES == 36_864);
// A thread copies the input rows `tid / 8` and `tid / 8 + 32` of its tile (the items after the weight items).
const _: () = assert!(BR * 8 % crate::staged_product::ROWS_THREADS == 0 && BT * 8 == 2 * crate::staged_product::ROWS_THREADS);

/// The replay of one recorded command of this kind: `table[index]` holds eleven words, the addresses of the weights
/// (rows x cols BF16 patterns, two a word), of the input (input count x cols FP32 values) and of the output ((output
/// count - 1) x stride + rows words), then rows, cols, inputs (the input rows of the launch) and stride, then the input
/// table (null, or inputs u32 row numbers) and the input count, then the output table (null, or inputs u32 row numbers)
/// and the output count, then the run table (null for one matrix) and the runs, then the block table and the blocks.
/// Launch row `t` reads input row `in[t]` (`t` without a table) and writes the words `out[t] * stride + row`
/// (`t * stride + row` without a table). In the grouped form block `b` reads its run and its input and weight tiles
/// from the block table, and that run's matrix, first launch row and count from the run table; its launch rows are
/// `first .. first + count`. A launch that the rule `staged_product::rows_admitted`
/// does not take, a table that is not word aligned, and an index at or above its count stop the block.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_staged_rows`): `table` and `index` are the
/// arguments the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::project_staged_rows_at_into (the strided and the
    // plain calls go through it) and Backend::project_staged_rows_grouped_into, after staged_product::check_rows_at
    // or check_rows_grouped passed for its fifteen words. That check holds (for each run's matrix in the grouped form):
    // the weights to exactly rows x cols BF16 values, the input to exactly input count x cols words, the output to
    // exactly (output count - 1) x stride + rows words (checked arithmetic) that meet no operand, with stride at least
    // rows, the weights and the input to 16-byte addresses, cols to whole stages of 64. All three live through
    // completion. The row tables are not the caller's: the host checks its own host words (each table exactly inputs
    // words, every index below its count, the output rows distinct) and the recorder stages exactly those words with
    // this command in its own stage area (record_resident_tabled), uploaded with the command's piece and written by
    // nothing else before this replay. The grouped form's run and block tables are staged the same way from host
    // words the host check built and checked (every run's matrix exactly rows x cols BF16 values, 16-byte aligned,
    // meeting no output; first + count of every run at most the launch rows; count at most 1,024). Here the block
    // index is compared with the blocks and the run with the runs before either table is read at it, both tables'
    // alignment is checked, and first + count is compared with the launch rows, so the run's launch rows stay inside
    // the index tables. So a table is read only at a launch row below inputs, inside it; every index is
    // compared here with its count before any use and the block traps when it is not below it. The rule
    // staged_product::rows_admitted is asked here again before the first read; a launch that fails it stops. Weight
    // reads: two words (4 values) of a row below rows, from a column below cols that is a multiple of 4, so inside the
    // row. Input reads: one 16-byte copy (4 words) of an input row below the input count, from a column below cols that
    // is a multiple of 4. Each (launch row, weight row) word, at output row x stride + weight row, is stored only when
    // the launch row is below inputs and the weight row below rows; the output row is below the output count, and with
    // stride at least rows (checked again here) the largest offset is (output count - 1) x stride + rows - 1. The
    // output rows of one launch are distinct (the staged words are the checked ones), and stride at least rows keeps
    // two output rows apart, so each word has exactly one writer. Every shared read and write stays in the block's
    // own two-stage static array. All 256 threads reach every barrier; the loop bounds depend only on the shape.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_STAGED_ROWS { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (one_weights, input, output) = (*words as *const u32, *words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, k, launch_rows) = (*words.add(3) as usize, *words.add(4) as usize, *words.add(5) as usize);
        let stride = *words.add(6) as usize;
        if stride < rows { core::arch::nvptx::trap(); }
        let (in_table, in_count) = (*words.add(7) as *const u32, *words.add(8) as usize);
        let (out_table, out_count) = (*words.add(9) as *const u32, *words.add(10) as usize);
        if (in_table as usize | out_table as usize) & 3 != 0 { core::arch::nvptx::trap(); }
        // The operand row of launch row `t` (below `m`): the table word, or `t` without a table; never at or above `count`.
        let place = |table: *const u32, count: usize, t: usize| -> usize {
            let row = if table.is_null() { t } else { table.add(t).read() as usize };
            if row >= count { core::arch::nvptx::trap(); }
            row
        };
        // The grouped form (word 11 not null, the expert runs of a visit): the block reads its run and its two tiles
        // from the staged block table, and the run's matrix, first launch row and count from the staged run table.
        let (runs_table, runs) = (*words.add(11) as *const u32, *words.add(12) as usize);
        let (blocks_table, blocks) = (*words.add(13) as *const u32, *words.add(14) as usize);
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let (weights, base, m, tile_t, tile_r) = if runs_table.is_null() {
            let token_tiles = launch_rows.div_ceil(BT);
            (one_weights, 0, launch_rows, block % token_tiles, block / token_tiles)
        } else {
            if (runs_table as usize & 15) != 0 || (blocks_table as usize & 7) != 0 || block >= blocks { core::arch::nvptx::trap(); }
            let entry = blocks_table.add(2 * block);
            let (run, tiles) = (entry.read() as usize, entry.add(1).read() as usize);
            if run >= runs { core::arch::nvptx::trap(); }
            // 16 bytes a run: the matrix address (two words, low first), its first launch row, its count.
            let at = runs_table.add(4 * run);
            let matrix = (at.read() as usize | (at.add(1).read() as usize) << 32) as *const u32;
            let (first, count) = (at.add(2).read() as usize, at.add(3).read() as usize);
            if first + count > launch_rows { core::arch::nvptx::trap(); }
            (matrix, first, count, tiles & 0xffff, tiles >> 16)
        };
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        if !crate::staged_product::rows_admitted(rows, k, m, weights as usize, input as usize, threads) { core::arch::nvptx::trap(); }
        if tile_r * BR >= rows { return; }
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (ty, tx) = (tid / 16, tid % 16);
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        // The input rows of the two input items of this thread, read once (unused when the launch row is not below m).
        // Two plain values, not an array read at a computed place, so they stay in registers.
        let first_row = tile_t * BT + tid / 8;
        let source_low = if first_row < m { place(in_table, in_count, base + first_row) } else { 0 };
        let source_high = if first_row + 32 < m { place(in_table, in_count, base + first_row + 32) } else { 0 };
        // The columns are whole stages of 64, so every chunk of 32 is whole.
        let chunks = k / KC;
        // Copy chunk `c` of this block's 64 weight rows and 64 input rows into stage `c & 1`.
        let issue = |c: usize| {
            let stage = smem + ((c & 1) * STAGE * 4) as u32;
            let first = c * KC;
            let mut item = tid;
            while item < (BR + BT) * 8 {
                let (which, row, quad) = (item / (BR * 8), item % (BR * 8) / 8, item % 8);
                let column = first + quad * 4;
                let dst = stage + ((which * BR + row) * SK * 4 + quad * 16) as u32;
                if which == 0 {
                    let r = tile_r * BR + row;
                    if r < rows {
                        let pair = weights.add((r * k + column) / 2);
                        let (low, high) = (pair.read(), pair.add(1).read());
                        crate::ptx::store(dst, f32::from_bits(low << 16));
                        crate::ptx::store(dst + 4, f32::from_bits(low & 0xffff_0000));
                        crate::ptx::store(dst + 8, f32::from_bits(high << 16));
                        crate::ptx::store(dst + 12, f32::from_bits(high & 0xffff_0000));
                    }
                } else {
                    let t = tile_t * BT + row;
                    if t < m { crate::ptx::copy16(dst, input.add((if item < BR * 8 + 256 { source_low } else { source_high }) * k + column).cast()); }
                }
                item += 256;
            }
        };
        let mut acc = [[0.0f32; 4]; 4];
        issue(0);
        crate::ptx::commit();
        let mut c = 0;
        while c < chunks {
            if c + 1 < chunks { issue(c + 1); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
            crate::ptx::sync();
            let stage = smem + ((c & 1) * STAGE * 4) as u32;
            let mut q = 0;
            while q < KC / 4 {
                let mut w = [[0u32; 4]; 4];
                let mut x = [[0u32; 4]; 4];
                for i in 0..4 { w[i] = crate::ptx::load4(stage + (((ty * 4 + i) * SK + q * 4) * 4) as u32); }
                for j in 0..4 { x[j] = crate::ptx::load4(stage + (((BR + tx + 16 * j) * SK + q * 4) * 4) as u32); }
                for e in 0..4 {
                    for i in 0..4 { for j in 0..4 { acc[i][j] += f32::from_bits(x[j][e]) * f32::from_bits(w[i][e]); } }
                }
                q += 1;
            }
            crate::ptx::sync();
            c += 1;
        }
        let mut target = [0usize; 4];
        for (j, slot) in target.iter_mut().enumerate() {
            let token = tile_t * BT + tx + 16 * j;
            if token < m { *slot = place(out_table, out_count, base + token); }
        }
        for i in 0..4 {
            let row = tile_r * BR + ty * 4 + i;
            if row < rows {
                for j in 0..4 {
                    let token = tile_t * BT + tx + 16 * j;
                    if token < m { *output.add(target[j] * stride + row) = acc[i][j]; }
                }
            }
        }
    }
}
