//! Block FP8 on tensor cores: the ONE body of every block FP8 kernel entry. The word function is the contract of
//! `tc_f8` (`docs/tc-dense-contract.md`, section Block FP8), the same for every row count:
//! - a weight byte is its E4M3 value as an exact BF16 operand (`tc_f8::e4m3x2_bf16x2`, the host-tested function);
//! - an activation is rounded once to BF16 (`cvt.rn.bf16x2.f32`, round to nearest even);
//! - a scale block's sum is one ascending chain of `mma.sync m16n8k16 bf16` steps from +0.0, 16 columns a step;
//! - the scale enters once a block: one `mul.rn.f32` on the block's sum, then one `add.rn.f32` into the range sum;
//! - four ranges of whole scale blocks, cut by the block count only, joined fl(fl(R0 + R1) + fl(R2 + R3)) by three
//!   `add.rn.f32`; then the one BF16 round. No fused multiply-add anywhere. A raw job is one range and no round.
//!
//! Who does the work. A warp owns 16 consecutive weight rows (the A operand of the instruction: 16 rows x 16 columns
//! a step) and `NF` tiles of 8 token rows (the B operand: 16 columns x 8 tokens), so it holds 16 x 8*NF words. The
//! warps of a block own neighbouring row tiles and share the block's token rows. Columns pass through shared memory
//! 128 at a time in ascending order: asynchronous 16-byte copies, `D` stages deep, of the raw weight bytes and the raw
//! FP32 activations. A thread converts its operands when it loads them from the stage.
//!
//! The fragment of thread (g = lane / 4, t = lane % 4) in a step that starts at column k: weight rows g and g + 8,
//! columns k + 4t .. k + 4t + 4 of each (one 4-byte load a row); token g of each tile, the same four columns (one
//! 16-byte load a tile). A step sums its 16 products exactly after the cut, so the place of a column inside its step
//! does not change a bit (the twin sums the 16 terms as integers).
//!
//! A word (row, token) depends on its own weight row, its own token row and the column index only: a dead row or a
//! dead token is another word of the instruction, fed with +0 operands and never stored.
use crate::canonical::{madd, Acc, ZERO};
use crate::expert_ops::{GroupArguments, GroupEntry, Op};

/// Columns of one stage.
pub(crate) const CHUNK: usize = 128;
/// Weight rows of one warp.
pub(crate) const ROWS: usize = 16;
/// Token rows of one tensor-core tile.
pub(crate) const TILE: usize = 8;

/// What one block does: `rows` x `cols` weight bytes at `weights`, row-major; block scales at `scales`
/// (`[row / block_rows][column / block_cols]`); the block's first weight row `row0` (warp w owns rows
/// `row0 + 16 w ..`); `tokens` token rows, token n at `input + n * stride` words; the word of (row, token n) is
/// stored at `output + row * pitch + n`. `raw`: one range and no join (a quarter job). `round`: the BF16 round of
/// the word (false for a raw job and for the unrounded word of a check tool).
#[derive(Clone, Copy)]
pub(crate) struct Job {
    pub weights: *const u8,
    pub scales: *const f32,
    pub rows: usize,
    pub cols: usize,
    pub block_rows: usize,
    pub block_cols: usize,
    pub row0: usize,
    pub input: *const f32,
    pub stride: usize,
    pub tokens: usize,
    pub output: *mut f32,
    pub pitch: usize,
    pub raw: bool,
    pub round: bool,
    /// `row_body` only: 16-row tiles a block (1, 2 or 4): the block's warps are dealt to the tiles in equal runs, and
    /// tile i holds rows `row0 + 16 i ..` with its own shared bytes. The word does not depend on it.
    pub tiles: u32,
}
/// The two flags of a job from the role word of its shape.
#[inline(always)]
pub(crate) fn flags(role: u32) -> (bool, bool) {
    (role == crate::expert_ops::FP8_QUARTER, role != crate::expert_ops::FP8_QUARTER && role != crate::expert_ops::FP8_UNROUNDED)
}
/// Shared bytes of one stage: the weight bytes of `warps` warps, then the FP32 activations of `tokens` token rows.
#[inline(always)]
pub(crate) const fn stage_bytes(warps: usize, tokens: usize) -> usize { warps * ROWS * CHUNK + tokens * CHUNK * 4 }

/// The body. `smem` is the start of `D * stage_bytes(warps, job.tokens)` bytes of the block's shared memory.
#[inline(always)]
pub(crate) fn run<const NF: usize, const D: usize>(job: &Job, smem: u32) {
    // SAFETY: Private to the block FP8 entries of this crate (`fp8_tile::project`, `fp8_tc::group` behind
    // `fp8_rows_small`, `fp8_rows_few`, `fp8_rows_multi` and `fp8_rows_tile`), which reach it with every thread of the
    // block (no early exit after this call starts), the same `job` on every thread, a block of whole warps, `cols` a
    // multiple of 16, `block_cols` a multiple of 16, `block_rows` at least 1, `weights` and `input` 16-byte aligned,
    // `stride` a multiple of 4 words, `scales` 4-byte aligned, 1 to 8 * NF tokens, D at least 2, and `smem` the
    // start of D * stage_bytes(warps, tokens) bytes that belong to this block. Global reads: a weight copy is 16
    // whole columns of a row below `rows` at a column below `cols`; an activation copy is 4 whole columns of a token
    // below `tokens` at a column below `cols`; a scale load is the word of a row below `rows` and a block below
    // ceil(cols / block_cols). Shared reads: a thread reads a weight word only of a row below `rows` and an
    // activation word only of a token below `tokens`, both of a column below `cols`, in the stage of the chunk that
    // every thread of the block copied, after its own `cp.async.wait_group` and the block barrier; a dead row or
    // token loads nothing and feeds +0 operands. A stage is refilled only after the barrier that ends its chunk.
    // Every thread runs every commit, wait and barrier: the loop bounds depend only on `cols`. Global writes: the one
    // output word of each (row, token) the thread owns, row below `rows` and token below `tokens`; every such word
    // has exactly one owner (lane (row % 8) * 4 + token % 8 / 2 of the warp of the row tile, register
    // (row % 16 / 8) * 2 + token % 2 of tile token / 8). The host keeps every owner alive through completion.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($a:expr,$b:expr)=>{{let value:f32;asm!("add.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add4 {($x:expr,$y:expr)=>{{let (x,y):(Acc,Acc)=($x,$y);Acc {a:add!(x.a,y.a),b:add!(x.b,y.b),c:add!(x.c,y.c),d:add!(x.d,y.d)}}};}
        // Two FP32 values to two BF16 patterns, the first in the low half: one round to nearest even each.
        macro_rules! bf16x2 {($first:expr,$second:expr)=>{{
            let packed:u32;
            asm!("cvt.rn.bf16x2.f32 {packed}, {second}, {first};",packed=out(reg32)packed,first=in(reg32)f32::from_bits($first),second=in(reg32)f32::from_bits($second),options(nomem,nostack));
            packed
        }};}
        macro_rules! load_u32 {($p:expr)=>{{let v:u32;asm!("ld.shared.u32 {v}, [{p}];",v=out(reg32)v,p=in(reg32)$p);v}};}
        let threads = _block_dim_x() as usize;
        let warps = threads / 32;
        let tid = _thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let (rows, cols, tokens) = (job.rows, job.cols, job.tokens);
        let x_at = warps * ROWS * CHUNK;
        let stage_size = stage_bytes(warps, tokens);
        let chunks = cols.div_ceil(CHUNK);
        let tile_row = job.row0 + warp * ROWS;
        let (row_a, row_b) = (tile_row + g, tile_row + g + 8);
        let (live_a, live_b) = (row_a < rows, row_b < rows);
        let blocks = cols.div_ceil(job.block_cols);
        // The copies of chunk `c`: this warp's 128 weight pieces (16 bytes each, 4 a thread), and the block's
        // activation pieces (4 columns of one token each), dealt to the block's threads in turn.
        let issue = |c: usize| {
            if c < chunks {
                let stage = smem + ((c % D) * stage_size) as u32;
                let col0 = c * CHUNK;
                let mut q = 0;
                while q < 4 {
                    let piece = lane + 32 * q;
                    let (r, part) = (piece / 8, piece % 8);
                    let (row, col) = (tile_row + r, col0 + part * 16);
                    if row < rows && col < cols {
                        crate::ptx::copy16(stage + ((warp * ROWS + r) * CHUNK + part * 16) as u32, job.weights.add(row * cols + col).cast());
                    }
                    q += 1;
                }
                let mut piece = tid;
                while piece < tokens * 32 {
                    let (n, part) = (piece / 32, piece % 32);
                    let col = col0 + part * 4;
                    if col < cols {
                        crate::ptx::copy16(stage + (x_at + n * CHUNK * 4 + part * 16) as u32, job.input.add(n * job.stride + col).cast());
                    }
                    piece += threads;
                }
            }
            crate::ptx::commit();
        };
        // The sum of the open scale block, the open range, and the two held values of the join.
        let mut block_sum = [ZERO; NF];
        let mut range_sum = [ZERO; NF];
        let mut held = [ZERO; NF];
        let mut joined = [ZERO; NF];
        // Ranges closed so far, and scale blocks done so far. A range closes when its last block is done; an empty
        // range closes with +0.0. Ranges 0 and 2 are held; range 1 joins range 0; range 3 joins range 2, then the
        // two joins join: fl(fl(R0 + R1) + fl(R2 + R3)).
        let (mut part, mut done) = (0usize, 0usize);
        macro_rules! close_ranges {()=>{
            if !job.raw {
                while part < 4 && crate::tc_f8::range(blocks, part).1 == done {
                    let mut f = 0;
                    while f < NF {
                        if part == 0 || part == 2 { held[f] = range_sum[f]; range_sum[f] = ZERO; }
                        else if part == 1 { joined[f] = add4!(held[f], range_sum[f]); range_sum[f] = ZERO; }
                        else { let high = add4!(held[f], range_sum[f]); range_sum[f] = add4!(joined[f], high); }
                        f += 1;
                    }
                    part += 1;
                }
            }
        };}
        close_ranges!();
        let mut c = 0;
        while c + 1 < D { issue(c); c += 1; }
        c = 0;
        while c < chunks {
            issue(c + D - 1);
            asm!("cp.async.wait_group {n};", n = const (D - 1));
            crate::ptx::sync();
            let stage = smem + ((c % D) * stage_size) as u32;
            let col0 = c * CHUNK;
            let steps = (if col0 + CHUNK < cols { CHUNK } else { cols - col0 }) / 16;
            let mut s = 0;
            while s < steps {
                let at = (s * 16 + t * 4) as u32;
                // The A operand: four columns of weight rows g and g + 8.
                let wa = if live_a { load_u32!(stage + ((warp * ROWS + g) * CHUNK) as u32 + at) } else { 0 };
                let wb = if live_b { load_u32!(stage + ((warp * ROWS + g + 8) * CHUNK) as u32 + at) } else { 0 };
                let a = [crate::tc_f8::e4m3x2_bf16x2(wa & 0xffff), crate::tc_f8::e4m3x2_bf16x2(wb & 0xffff),
                    crate::tc_f8::e4m3x2_bf16x2(wa >> 16), crate::tc_f8::e4m3x2_bf16x2(wb >> 16)];
                // The B operand of each tile: the same four columns of token g of the tile.
                let mut f = 0;
                while f < NF {
                    let n = f * TILE + g;
                    let b = if n < tokens {
                        let x = crate::ptx::load4(stage + (x_at + n * CHUNK * 4) as u32 + at * 4);
                        [bf16x2!(x[0], x[1]), bf16x2!(x[2], x[3])]
                    } else { [0, 0] };
                    block_sum[f] = madd(a, b, block_sum[f]);
                    f += 1;
                }
                // The end of a scale block: the scale enters, the block joins its range.
                let end = col0 + s * 16 + 16;
                if end % job.block_cols == 0 || end == cols {
                    let block = (end - 1) / job.block_cols;
                    let scale_a = if live_a { *job.scales.add(row_a / job.block_rows * blocks + block) } else { 0.0 };
                    let scale_b = if live_b { *job.scales.add(row_b / job.block_rows * blocks + block) } else { 0.0 };
                    let mut f = 0;
                    while f < NF {
                        let (v, r) = (block_sum[f], range_sum[f]);
                        range_sum[f] = Acc { a: add!(r.a, mul!(v.a, scale_a)), b: add!(r.b, mul!(v.b, scale_a)), c: add!(r.c, mul!(v.c, scale_b)), d: add!(r.d, mul!(v.d, scale_b)) };
                        block_sum[f] = ZERO;
                        f += 1;
                    }
                    done += 1;
                    close_ranges!();
                }
                s += 1;
            }
            crate::ptx::sync();
            c += 1;
        }
        // Words: registers a, b = row g, tokens 2t and 2t + 1 of the tile; c, d = row g + 8.
        let put = |row: usize, n: usize, value: f32| {
            if n < tokens { *job.output.add(row * job.pitch + n) = if job.round { crate::expert_ops::round(value) } else { value }; }
        };
        let mut f = 0;
        while f < NF {
            let n = f * TILE + 2 * t;
            if live_a { put(row_a, n, range_sum[f].a); put(row_a, n + 1, range_sum[f].b); }
            if live_b { put(row_b, n, range_sum[f].c); put(row_b, n + 1, range_sum[f].d); }
            f += 1;
        }
    }
}

/// One block of a row-group launch (`GroupArguments` and the table of `expert_group`): part `part`, token tile `tile`
/// (tokens `tile * 8 * NF ..` of the part), row group `row_group` (rows `row_group * 16 * warps ..`). The entries map
/// their grid to these three numbers; the word of a (part, row, token) does not depend on that map.
#[inline(always)]
pub(crate) fn group<const NF: usize, const D: usize>(p: GroupArguments, part: usize, tile: usize, row_group: usize) {
    // SAFETY: The host checks every part (shape, raw owners, input span) and keeps all owners and the table alive
    // through stream completion, as for `expert_group`. Weights, scales and inputs are immutable. The output is
    // private; each (part, row, token) word has exactly one writing thread (`run`), at first + row * tokens + token
    // inside the checked units. The body traps unless the block is whole warps, one word is an output, the part is an
    // Fp8Rows part with the first part's rows and cols, cols is a multiple of 16, block_cols a power of two of at
    // least 16, block_rows at least 1, weight and input are 16-byte aligned, scales 4-byte aligned, the row stride
    // is cols (a multiple of 4 words), the part is not a raw quarter job, and the dynamic shared region holds the D
    // stages of this block. The dynamic region is the only shared allocation of the kernels that reach this body, so
    // it starts at 0. A block with no part, no row or no token returns before any barrier, all its threads alike.
    unsafe {
        use core::arch::asm;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let warps = threads / 32;
        let dynamic: u32;
        asm!("mov.u32 {d}, %dynamic_smem_size;", d = out(reg32) dynamic, options(nomem, nostack));
        if threads % 32 != 0 || warps == 0 || D < 2 || p.count == 0 || p.words != 1 { asm!("trap;"); }
        if part >= p.count as usize { return; }
        let table = core::slice::from_raw_parts(p.table as *const GroupEntry, p.count as usize);
        let (first, e) = (&table[0], &table[part]);
        let (rows, cols, m) = (e.shape.rows as usize, e.shape.cols as usize, e.shape.tokens as usize);
        let (block_rows, block_cols) = (e.shape.block_rows as usize, e.shape.block_cols as usize);
        let (raw, round) = flags(e.shape.role);
        if e.shape.op != Op::Fp8Rows || cols % 16 != 0 || !block_cols.is_power_of_two() || block_cols < 16 || block_rows == 0
            || rows != first.shape.rows as usize || cols != first.shape.cols as usize || (e.weight | p.input) & 15 != 0 || e.scales & 3 != 0
            || p.stride as usize != cols || raw { asm!("trap;"); }
        let (row0, token0) = (row_group * ROWS * warps, tile * TILE * NF);
        if row0 >= rows || token0 >= m { return; }
        let tokens = if m - token0 < TILE * NF { m - token0 } else { TILE * NF };
        if (dynamic as usize) < D * stage_bytes(warps, tokens) { asm!("trap;"); }
        let job = Job { weights: e.weight as *const u8, scales: e.scales as *const f32, rows, cols, block_rows, block_cols, row0,
            input: (p.input as *const f32).add((e.first_row as usize + token0) * cols), stride: cols, tokens,
            output: (p.output as *mut f32).add(e.first as usize + token0), pitch: m, raw, round, tiles: 1 };
        run::<NF, D>(&job, 0);
    }
}
/// The (part, token tile, row group) of block `block` of a ONE-dimension grid of `grid` blocks: `count` parts, each
/// with the same number of token tiles, each tile with `ceil(rows / (16 * warps))` row groups; the row groups of one
/// tile are neighbours (they read the same token rows). None when the grid is not such a product.
#[inline(always)]
pub(crate) fn linear(block: usize, grid: usize, count: usize, rows: usize, warps: usize) -> Option<(usize, usize, usize)> {
    if count == 0 || warps == 0 || rows == 0 { return None; }
    // 32-bit numbers: a grid has fewer than 2^31 blocks. One part needs no division for the part.
    let (block, grid, count) = (block as u32, grid as u32, count as u32);
    let row_groups = rows.div_ceil(ROWS * warps) as u32;
    let (part, rest, per_part) = if count == 1 { (0, block, grid) } else {
        let per_part = grid / count;
        if per_part == 0 || per_part * count != grid { return None; }
        (block / per_part, block % per_part, per_part)
    };
    let (tiles, tile) = (per_part / row_groups, rest / row_groups);
    if tiles == 0 || tiles * row_groups != per_part { return None; }
    Some((part as usize, tile as usize, (rest - tile * row_groups) as usize))
}
use crate::expert_ops::{fp8_row_shared as rows_shared, FP8_ROW_JOIN};

/// The body of the decode step and of small row groups: ONE tile of 16 weight rows (`job.row0 ..`) and 1 to 8 token
/// rows a block, on `W` warps (a power of two from 4 to 32). The word is the contract word (`tc_f8`), as `run` gives.
///
/// Who does the work. The contract cuts a row's scale blocks into four ranges and gives every scale block its own
/// chain from +0.0, so the blocks of a row do not wait for each other. The `W / 4` warps of a range (all `W` warps
/// of a raw job's one range) cut the range's blocks into consecutive spans, one a warp. A warp walks its span: 16
/// columns a step, the weight bytes and the activations straight from global memory (no stage, no barrier inside
/// the walk), one `mul.rn.f32` by the block's scale at the block's end. The first warp of a range adds its blocks
/// into the range sum as it goes; every other warp hands its scaled block sums through shared memory, and after one
/// barrier the first warp adds them in ascending block order. Then the four first warps park their range sums and
/// warp 0 joins them fl(fl(R0 + R1) + fl(R2 + R3)). So the adds of a word are the contract's adds in the contract's
/// order, whatever `W` is.
///
/// Why (MEASURED 2026-11-15, run `f8-takeover-full2`, body `run` with one warp for all columns of 16 rows): one row
/// took the same time at 288, 768 and 1,024 weight rows of 4,096 columns (80 us) and four times that at 16,384
/// columns: the serial walk of the columns set the time, not the bytes.
///
/// A lane (g = lane / 4, t = lane % 4) holds rows g and g + 8 of the tile and four columns `k + 4t ..` of a step:
/// bytes 0 and 2 of its weight word in one operand register, bytes 1 and 3 in the other (`tc_f8::e4m3x4_bf16x4`),
/// and the activations of token g paired the same way. The place of a column inside its step changes no bit.
/// A row past `rows` or a token past `tokens` is another word of the instruction: the lane loads a live row or
/// token in its place (no branch in the walk) and never stores the word.
/// `PF`: steps ahead of the walk that lanes 0 and 1 of a quad pull their rows' next 64-byte line into L2 (0: none).
/// `PS`: steps between two such prefetches of a row (a power of two; 4 is once a line).
#[inline(always)]
pub(crate) fn row_body<const PF: usize, const PS: usize>(job: &Job, smem: u32) {
    // SAFETY: Private to the block FP8 row entries of this crate (`fp8_tile::project`, `group_rows` behind the small
    // entries and the bench entries of `fp8_tc_rows`), which reach it with every thread of the block (no early exit
    // after this call starts), the same `job` on every thread, a block of 4, 8, 16 or 32 whole warps, `tiles` 1, 2
    // or 4 with at least 4 warps a tile, `row0` below `rows` (a tile past `rows` reads row `row0` and stores nothing),
    // `cols` a multiple of 16, `block_cols` a power of two of at least 16, `block_rows` at least 1,
    // `weights` and `input` 16-byte aligned, `stride` a multiple of 4 words (0 with one token), `scales` 4-byte
    // aligned, 1 to 8 tokens, and `smem` the start of `tiles * rows_shared(warps a tile, blocks, tokens, raw)` bytes that belong to
    // this block. Global reads: a weight load is 4 whole columns of a row below `rows` at a column below `cols` (a
    // lane whose row is past `rows` reads row `row0`); an activation load is 4 whole columns of a token below
    // `tokens` (a lane whose token is past `tokens` reads token 0); a scale load is the word of a row below `rows`
    // and a block below ceil(cols / block_cols); a prefetch names a column below `cols` of a row below `rows` and
    // reads nothing. Weights, scales and inputs are immutable (the non-coherent loads need that). Shared memory: a
    // lane writes its handed quad of block i of its span at its own slot before the first barrier, and the first
    // warp of the range reads the same slot after it; a first warp writes its join slot before the last barrier and
    // warp 0 reads the four slots of its lane index after it. Every thread reaches every barrier: their number
    // depends only on the warp count and `raw`. Global writes: the one output word of each (row, token) a lane of
    // warp 0 owns, row below `rows` and token below `tokens`; every such word has exactly one owner (lane
    // (row % 8) * 4 + token / 2, register (row % 16 / 8) * 2 + token % 2). The host keeps every owner alive.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($a:expr,$b:expr)=>{{let value:f32;asm!("add.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add4 {($x:expr,$y:expr)=>{{let (x,y):(Acc,Acc)=($x,$y);Acc {a:add!(x.a,y.a),b:add!(x.b,y.b),c:add!(x.c,y.c),d:add!(x.d,y.d)}}};}
        macro_rules! bf16x2 {($first:expr,$second:expr)=>{{
            let packed:u32;
            asm!("cvt.rn.bf16x2.f32 {packed}, {second}, {first};",packed=out(reg32)packed,first=in(reg32)f32::from_bits($first),second=in(reg32)f32::from_bits($second),options(nomem,nostack));
            packed
        }};}
        macro_rules! quad {($p:expr)=>{{let v=crate::ptx::load4($p);Acc {a:f32::from_bits(v[0]),b:f32::from_bits(v[1]),c:f32::from_bits(v[2]),d:f32::from_bits(v[3])}}};}
        macro_rules! park {($p:expr,$v:expr)=>{{let v:Acc=$v;asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};",p=in(reg32)$p,a=in(reg32)v.a,b=in(reg32)v.b,c=in(reg32)v.c,d=in(reg32)v.d,options(nostack));}};}
        // The warps of one tile, this thread's tile of the block and its warp inside the tile.
        let warps = (_block_dim_x() / 32) >> job.tiles.trailing_zeros();
        let tid = _thread_idx_x() as u32;
        let lane = tid % 32;
        let (tile, warp) = ((tid / 32) >> warps.trailing_zeros(), (tid / 32) & (warps - 1));
        let (g, t) = (lane / 4, lane % 4);
        let (rows, cols, tokens) = (job.rows, job.cols, job.tokens as u32);
        let shift = job.block_cols.trailing_zeros();
        let blocks = ((cols + job.block_cols - 1) >> shift) as u32;
        let smem = smem + tile * rows_shared(warps, blocks, tokens, job.raw);
        let row0 = job.row0 + tile as usize * ROWS;
        // Ranges, the warps of a range (a power of two), this warp's range and its place in the range.
        let range_shift = if job.raw { 0 } else { 2 };
        let per = warps >> range_shift;
        let per_shift = per.trailing_zeros();
        let (range, place) = (warp >> per_shift, warp & (per - 1));
        let (first, last) = if job.raw { (0, blocks) } else { (blocks * range / 4, blocks * (range + 1) / 4) };
        let span = |place: u32| -> (u32, u32) { let n = last - first; (first + ((n * place) >> per_shift), first + ((n * (place + 1)) >> per_shift)) };
        let (begin, end) = span(place);
        let carries = place == 0;
        // The handed quads: `most` slots a handing warp, `slot` bytes a block, this lane's quad at `own` inside a slot.
        let pairs = (tokens + 1) / 2;
        let slot = 8 * pairs * 16;
        let most = (((blocks + (1 << range_shift) - 1) >> range_shift) + per - 1) >> per_shift;
        let hands = t < pairs;
        let own = (g * pairs + t) * 16;
        let handed = |range: u32, place: u32, index: u32| -> u32 { smem + FP8_ROW_JOIN + ((range * (per - 1) + place - 1) * most + index) * slot + own };
        // Rows, tokens and pointers. A dead row or token reads a live one.
        let (row_a, row_b) = (row0 + g as usize, row0 + g as usize + 8);
        let (live_a, live_b) = (row_a < rows, row_b < rows);
        let (read_a, read_b) = (if live_a { row_a } else { job.row0 }, if live_b { row_b } else { job.row0 });
        let token = if g < tokens { g as usize } else { 0 };
        let weights_a = job.weights.add(read_a * cols + 4 * t as usize);
        let weights_b = job.weights.add(read_b * cols + 4 * t as usize);
        let input = (job.input as *const u8).add((token * job.stride + 4 * t as usize) * 4);
        // The scale row of a weight row: a shift when the scale blocks are a power of two of rows high (the model's 128).
        let scale_row = |row: usize| -> u32 { if job.block_rows.is_power_of_two() { row as u32 >> job.block_rows.trailing_zeros() } else { row as u32 / job.block_rows as u32 } };
        let scales_a = job.scales.add((scale_row(read_a) * blocks) as usize);
        let scales_b = job.scales.add((scale_row(read_b) * blocks) as usize);
        let fetches = t < 2;
        let fetch = job.weights.add(if t == 0 { read_a } else { read_b } * cols);
        let (mut col, stop) = ((begin as usize) << shift, { let e = (end as usize) << shift; if e < cols { e } else { cols } });
        macro_rules! prefetch {($col:expr)=>{asm!("prefetch.global.L2 [{pointer}];",pointer=in(reg64)fetch.add($col),options(nostack,readonly));};}
        if PF != 0 && fetches {
            let mut ahead = col;
            while ahead < col + PF * 16 && ahead < stop { prefetch!(ahead); ahead += 64; }
        }
        let mut range_sum = ZERO;
        let mut block = begin;
        while block < end {
            let block_stop = { let e = (block as usize + 1) << shift; if e < cols { e } else { cols } };
            let mut sum = ZERO;
            while col < block_stop {
                if PF != 0 && fetches && col & (PS * 16 - 1) == 0 && col + PF * 16 < stop { prefetch!(col + PF * 16); }
                let (wa, wb): (u32, u32);
                asm!("ld.global.nc.L1::no_allocate.u32 {v}, [{pointer}];", v = out(reg32) wa, pointer = in(reg64) weights_a.add(col), options(nostack, readonly));
                asm!("ld.global.nc.L1::no_allocate.u32 {v}, [{pointer}];", v = out(reg32) wb, pointer = in(reg64) weights_b.add(col), options(nostack, readonly));
                let (x0, x1, x2, x3): (u32, u32, u32, u32);
                asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x0, b = out(reg32) x1, c = out(reg32) x2, d = out(reg32) x3, pointer = in(reg64) input.add(col * 4), options(nostack, readonly));
                let (mut a, mut b) = (crate::tc_f8::e4m3x4_fast(wa), crate::tc_f8::e4m3x4_fast(wb));
                if crate::tc_f8::any_small(wa) | crate::tc_f8::any_small(wb) { a = crate::tc_f8::e4m3x4_slow(wa); b = crate::tc_f8::e4m3x4_slow(wb); }
                sum = madd([a[0], b[0], a[1], b[1]], [bf16x2!(x0, x2), bf16x2!(x1, x3)], sum);
                col += 16;
            }
            // The end of a scale block: the scale enters; the block joins its range, here or after the barrier.
            let (scale_a, scale_b) = (*scales_a.add(block as usize), *scales_b.add(block as usize));
            let scaled = Acc { a: mul!(sum.a, scale_a), b: mul!(sum.b, scale_a), c: mul!(sum.c, scale_b), d: mul!(sum.d, scale_b) };
            if carries { range_sum = add4!(range_sum, scaled); } else if hands { park!(handed(range, place, block - begin), scaled); }
            block += 1;
        }
        if per > 1 {
            crate::ptx::sync();
            if carries && hands {
                let mut other = 1;
                while other < per {
                    let (from, to) = span(other);
                    let mut index = 0;
                    while index < to - from { range_sum = add4!(range_sum, quad!(handed(range, other, index))); index += 1; }
                    other += 1;
                }
            }
        }
        if !job.raw {
            if carries { park!(smem + (range * 32 + lane) * 16, range_sum); }
            crate::ptx::sync();
            if warp == 0 {
                let part = |range: u32| -> Acc { quad!(smem + (range * 32 + lane) * 16) };
                range_sum = add4!(add4!(part(0), part(1)), add4!(part(2), part(3)));
            }
        }
        if warp != 0 { return; }
        // Words: registers a, b = row g, tokens 2t and 2t + 1; c, d = row g + 8.
        let put = |row: usize, n: u32, value: f32| {
            if n < tokens { *job.output.add(row * job.pitch + n as usize) = if job.round { crate::expert_ops::round(value) } else { value }; }
        };
        if live_a { put(row_a, 2 * t, range_sum.a); put(row_a, 2 * t + 1, range_sum.b); }
        if live_b { put(row_b, 2 * t, range_sum.c); put(row_b, 2 * t + 1, range_sum.d); }
    }
}

/// One block of a row-group launch on the body `row_body`: part `part`, token tile `tile` (tokens `tile * 8 ..` of the
/// part), row tile `row_tile` (rows `row_tile * 16 ..`). The checks of `group`, and the shared bytes of `row_body`.
#[inline(always)]
pub(crate) fn group_rows<const PF: usize, const PS: usize>(p: GroupArguments, part: usize, tile: usize, row_tile: usize) {
    // SAFETY: As `group`: the host checks every part and keeps all owners and the table alive through stream
    // completion; weights, scales and inputs are immutable; the output is private and each (part, row, token) word
    // has exactly one writing thread (`row_body`). The body traps unless the block is 4, 8, 16 or 32 whole warps, one
    // word is an output, the part is an Fp8Rows part with the first part's rows and cols, cols is a multiple of 16,
    // block_cols a power of two of at least 16, block_rows at least 1, weight and input are 16-byte aligned, scales
    // 4-byte aligned, the row stride is cols, the part is not a raw quarter job, and the dynamic shared region
    // (the only shared allocation of the kernels that reach this body, so it starts at 0) holds `rows_shared`. A
    // block with no part, no row or no token returns before any barrier, all its threads alike.
    unsafe {
        use core::arch::asm;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let warps = threads / 32;
        let dynamic: u32;
        asm!("mov.u32 {d}, %dynamic_smem_size;", d = out(reg32) dynamic, options(nomem, nostack));
        if threads % 32 != 0 || !warps.is_power_of_two() || warps < 4 || warps > 32 || p.count == 0 || p.words != 1 { asm!("trap;"); }
        if part >= p.count as usize { return; }
        let table = core::slice::from_raw_parts(p.table as *const GroupEntry, p.count as usize);
        let (first, e) = (&table[0], &table[part]);
        let (rows, cols, m) = (e.shape.rows as usize, e.shape.cols as usize, e.shape.tokens as usize);
        let (block_rows, block_cols) = (e.shape.block_rows as usize, e.shape.block_cols as usize);
        let (raw, round) = flags(e.shape.role);
        if e.shape.op != Op::Fp8Rows || cols % 16 != 0 || !block_cols.is_power_of_two() || block_cols < 16 || block_rows == 0
            || rows != first.shape.rows as usize || cols != first.shape.cols as usize || (e.weight | p.input) & 15 != 0 || e.scales & 3 != 0
            || p.stride as usize != cols || raw { asm!("trap;"); }
        let (row0, token0) = (row_tile * ROWS, tile * TILE);
        if row0 >= rows || token0 >= m { return; }
        let tokens = if m - token0 < TILE { m - token0 } else { TILE };
        if dynamic < rows_shared(warps as u32, ((cols + block_cols - 1) >> block_cols.trailing_zeros()) as u32, tokens as u32, raw) { asm!("trap;"); }
        let job = Job { weights: e.weight as *const u8, scales: e.scales as *const f32, rows, cols, block_rows, block_cols, row0,
            input: (p.input as *const f32).add((e.first_row as usize + token0) * cols), stride: cols, tokens,
            output: (p.output as *mut f32).add(e.first as usize + token0), pitch: m, raw, round, tiles: 1 };
        if !PS.is_power_of_two() || PS > 4 { asm!("trap;"); }
        row_body::<PF, PS>(&job, 0);
    }
}

/// Steps of one operand stage of `tile_body`: 128 columns.
pub(crate) const STEPS: usize = 8;
/// Columns of one operand stage of `tile_body`.
pub(crate) const SPAN: usize = STEPS * 16;
/// Shared bytes of a `tile_body` block: the BF16 operands of one stage, 8 bytes for each of 8 token places x STEPS
/// steps x 4 lanes x `nf` tiles.
pub(crate) const fn tile_shared(nf: usize) -> usize { 8 * STEPS * 4 * nf * 8 }

/// The body of the prompt tile, second form: the word is the contract word (`tc_f8`), as `run` and `row_body` give.
/// A warp owns 16 weight rows (`job.row0 + 16 * warp ..`) and `NF` tiles of 8 token rows (an even number, 2 to 8),
/// so a block of `W` warps makes 16 W x 8 NF words.
///
/// Why (MEASURED 2026-11-16, runs `f8row-all-3` and `f8row-all-4`): the first form (`run`) makes 6.1 to 6.9 TMAC/s
/// at 1,024 token rows, and the instruction alone makes 68 to 104 on this device (seat ONEROW's raw-rate probe, run
/// `onerow-tv-1`): the time is in the work around the instruction. In `run` every warp rounds the block's
/// activations again at every step (two conversions and a 16-byte load a tile), converts its weights for four tiles
/// only, divides a 64-bit number at every step, and waits at two barriers and a copy group for every 128 columns.
/// Here:
/// - the block rounds its activations ONCE a stage of 128 columns into shared memory, as BF16 operands in the
///   order a lane loads them (one 16-byte load for two tiles);
/// - a warp converts its weights once a step for all `NF` tiles (`tc_f8::e4m3x4_bf16x4`), straight from global
///   memory (no copy group), with the row body's prefetch;
/// - the walk is a loop nest over ranges, scale blocks and steps: no division;
/// - the join of ranges 0 and 1 waits in the word's own output slot while ranges 2 and 3 are summed (a lane owns
///   its words alone), so a tile keeps three accumulators in registers, not four.
///
/// A lane (g = lane / 4, t = lane % 4) holds rows g and g + 8 of its warp's tile and four columns `k + 4t ..` of a
/// step, paired as in `row_body`: bytes 0 and 2 of the weight word in one operand register, bytes 1 and 3 in the
/// other, and the activations of token g of each tile the same way. A row past `rows` or a token past `tokens`
/// loads a live one in its place and is never stored; a tile with no live token is skipped (all lanes alike).
/// `PF`: steps ahead of the walk that lanes 0 and 1 of a quad pull their rows' next 64-byte line into L2 (0: none).
#[inline(always)]
pub(crate) fn tile_body<const NF: usize, const PF: usize>(job: &Job, smem: u32) {
    // SAFETY: Private to `group_tile` behind the tile entries of this crate, which reach it with every thread of the
    // block (no early exit after this call starts), the same `job` on every thread, a block of whole warps, `NF`
    // even and from 2 to 8, `row0` below `rows`, `cols` a multiple of 16, `block_cols` a power of two of at least
    // 16, `block_rows` at least 1, `weights` and `input` 16-byte aligned, `stride` a multiple of 4 words, `scales`
    // 4-byte aligned, 1 to 8 * NF tokens, not a raw job's caller unless it wants one range, and `smem` the start of
    // `tile_shared(NF)` bytes that belong to this block. Global reads: a weight load is 4 whole columns of a row
    // below `rows` at a column below `cols` (a lane whose row is past `rows` reads row `row0`); an activation load
    // is 4 whole columns of a token below `tokens` (a place past `tokens` reads token 0) at a column below `cols`; a
    // scale load is the word of a row below `rows` and a block below ceil(cols / block_cols); a prefetch names a
    // column below `cols` of a live row or token and reads nothing. Weights, scales and inputs are immutable (the
    // non-coherent loads need that). Shared memory: between the two barriers of a stage every thread writes its own
    // operand slots (slot q belongs to thread q % threads), and after the second barrier a lane reads only slots of
    // live steps and live tiles, which were written in this stage. Every thread reaches every barrier: their number
    // depends only on `cols`. Global writes and their reads back: the one output word of each (row, token) a lane
    // owns, row below `rows` and token below `tokens` (lane (row % 8) * 4 + token % 8 / 2 of the warp of the row's
    // tile, register (row % 16 / 8) * 2 + token % 2 of tile token / 8): the lane parks a partial join there and
    // reads it back itself before it stores the word; the output is private until the stream completes. The host
    // keeps every owner alive.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($a:expr,$b:expr)=>{{let value:f32;asm!("add.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add4 {($x:expr,$y:expr)=>{{let (x,y):(Acc,Acc)=($x,$y);Acc {a:add!(x.a,y.a),b:add!(x.b,y.b),c:add!(x.c,y.c),d:add!(x.d,y.d)}}};}
        macro_rules! bf16x2 {($first:expr,$second:expr)=>{{
            let packed:u32;
            asm!("cvt.rn.bf16x2.f32 {packed}, {second}, {first};",packed=out(reg32)packed,first=in(reg32)f32::from_bits($first),second=in(reg32)f32::from_bits($second),options(nomem,nostack));
            packed
        }};}
        let threads = _block_dim_x() as u32;
        let tid = _thread_idx_x() as u32;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let (rows, cols, tokens) = (job.rows, job.cols, job.tokens as u32);
        let shift = job.block_cols.trailing_zeros();
        let blocks = (cols + job.block_cols - 1) >> shift;
        let (nf, steps_stage) = (NF as u32, STEPS as u32);
        // Tiles with a live token, and the operand slots of a stage.
        let live_tiles = (tokens + 7) / 8;
        let slots = 8 * steps_stage * 4 * nf;
        // Rows and pointers. A dead row reads the block's first row.
        let tile_row = job.row0 + warp as usize * ROWS;
        let (row_a, row_b) = (tile_row + g as usize, tile_row + g as usize + 8);
        let (live_a, live_b) = (row_a < rows, row_b < rows);
        let (read_a, read_b) = (if live_a { row_a } else { job.row0 }, if live_b { row_b } else { job.row0 });
        let weights_a = job.weights.add(read_a * cols + 4 * t as usize);
        let weights_b = job.weights.add(read_b * cols + 4 * t as usize);
        let scale_row = |row: usize| -> usize { if job.block_rows.is_power_of_two() { row >> job.block_rows.trailing_zeros() } else { row / job.block_rows } };
        let scales_a = job.scales.add(scale_row(read_a) * blocks);
        let scales_b = job.scales.add(scale_row(read_b) * blocks);
        let fetches = t < 2;
        let fetch = job.weights.add(if t == 0 { read_a } else { read_b } * cols);
        macro_rules! prefetch {($pointer:expr)=>{asm!("prefetch.global.L2 [{pointer}];",pointer=in(reg64)$pointer,options(nostack,readonly));};}
        if PF != 0 && fetches {
            let mut ahead = 0;
            while ahead < PF * 16 && ahead < cols { prefetch!(fetch.add(ahead)); ahead += 64; }
        }
        // This lane's operands of step s of the stage: `own + s * 32 * NF`, 8 bytes a tile.
        let own = smem + ((g * steps_stage * 4 + t) * nf) * 8;
        // One stage: the block rounds the activations of 128 columns from `c0` once. Slot q = ((place * STEPS +
        // step) * 4 + lane quarter) * NF + tile holds the two operand registers of token `tile * 8 + place`, columns
        // `c0 + 16 step + 4 quarter ..`; thread i fills the slots i, i + threads, ... .
        macro_rules! stage {($c0:expr)=>{{
            let c0: usize = $c0;
            crate::ptx::sync();
            let live_steps = ((if c0 + SPAN < cols { SPAN } else { cols - c0 }) / 16) as u32;
            let mut q = tid;
            while q < slots {
                let (f, rest) = (q % nf, q / nf);
                let (quarter, step, place) = (rest % 4, (rest / 4) % steps_stage, rest / (4 * steps_stage));
                if step < live_steps && f < live_tiles {
                    let n = f * 8 + place;
                    let n = if n < tokens { n } else { 0 };
                    let pointer = (job.input as *const u8).add((n as usize * job.stride + c0 + 16 * step as usize + 4 * quarter as usize) * 4);
                    // The next stage's line of this token (64 bytes: the four quarters of a step).
                    if PF != 0 && quarter == 0 && c0 + SPAN + 16 * (step as usize) < cols { prefetch!(pointer.add(SPAN * 4)); }
                    let (x0, x1, x2, x3): (u32, u32, u32, u32);
                    asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x0, b = out(reg32) x1, c = out(reg32) x2, d = out(reg32) x3, pointer = in(reg64) pointer, options(nostack, readonly));
                    let (low, high) = (bf16x2!(x0, x2), bf16x2!(x1, x3));
                    asm!("st.shared.v2.b32 [{p}], {{{low},{high}}};", p = in(reg32) smem + q * 8, low = in(reg32) low, high = in(reg32) high, options(nostack));
                }
                q += threads;
            }
            crate::ptx::sync();
        }};}
        // The output words of this lane in tile f: registers a, b = row g, tokens 2t and 2t + 1 of the tile; c, d = row g + 8.
        let slot = |row: usize, n: u32| -> *mut f32 { job.output.add(row * job.pitch + n as usize) };
        let parts = if job.raw { 1 } else { 4 };
        let mut held = [ZERO; NF];
        let mut word = [ZERO; NF];
        let mut col = 0usize;
        let mut part = 0;
        while part < parts {
            let (first, last) = if job.raw { (0, blocks) } else { crate::tc_f8::range(blocks, part) };
            let mut range_sum = [ZERO; NF];
            let mut block = first;
            while block < last {
                let block_stop = { let e = (block + 1) << shift; if e < cols { e } else { cols } };
                let mut sum = [ZERO; NF];
                while col < block_stop {
                    if col & (SPAN - 1) == 0 { stage!(col); }
                    if PF != 0 && fetches && col & 63 == 0 && col + PF * 16 < cols { prefetch!(fetch.add(col + PF * 16)); }
                    let (wa, wb): (u32, u32);
                    asm!("ld.global.nc.L1::no_allocate.u32 {v}, [{pointer}];", v = out(reg32) wa, pointer = in(reg64) weights_a.add(col), options(nostack, readonly));
                    asm!("ld.global.nc.L1::no_allocate.u32 {v}, [{pointer}];", v = out(reg32) wb, pointer = in(reg64) weights_b.add(col), options(nostack, readonly));
                    let (mut a, mut b) = (crate::tc_f8::e4m3x4_fast(wa), crate::tc_f8::e4m3x4_fast(wb));
                    if crate::tc_f8::any_small(wa) | crate::tc_f8::any_small(wb) { a = crate::tc_f8::e4m3x4_slow(wa); b = crate::tc_f8::e4m3x4_slow(wb); }
                    let operands = own + ((col & (SPAN - 1)) as u32 / 16) * (32 * nf);
                    let mut f = 0;
                    while f < NF {
                        if (f as u32) < live_tiles {
                            let o = crate::ptx::load4(operands + 8 * f as u32);
                            sum[f] = madd([a[0], b[0], a[1], b[1]], [o[0], o[1]], sum[f]);
                            if (f as u32) + 1 < live_tiles { sum[f + 1] = madd([a[0], b[0], a[1], b[1]], [o[2], o[3]], sum[f + 1]); }
                        }
                        f += 2;
                    }
                    col += 16;
                }
                // The end of a scale block: the scale enters, the block joins its range.
                let (scale_a, scale_b) = (*scales_a.add(block), *scales_b.add(block));
                let mut f = 0;
                while f < NF {
                    let (v, r) = (sum[f], range_sum[f]);
                    range_sum[f] = Acc { a: add!(r.a, mul!(v.a, scale_a)), b: add!(r.b, mul!(v.b, scale_a)), c: add!(r.c, mul!(v.c, scale_b)), d: add!(r.d, mul!(v.d, scale_b)) };
                    f += 1;
                }
                block += 1;
            }
            // The range is closed (an empty range closes with +0.0): fl(fl(R0 + R1) + fl(R2 + R3)).
            let mut f = 0;
            while f < NF {
                let n = f as u32 * 8 + 2 * t;
                if job.raw { word[f] = range_sum[f]; }
                else if part == 0 || part == 2 { held[f] = range_sum[f]; }
                else if part == 1 {
                    let low = add4!(held[f], range_sum[f]);
                    if live_a { if n < tokens { *slot(row_a, n) = low.a; } if n + 1 < tokens { *slot(row_a, n + 1) = low.b; } }
                    if live_b { if n < tokens { *slot(row_b, n) = low.c; } if n + 1 < tokens { *slot(row_b, n + 1) = low.d; } }
                } else {
                    let high = add4!(held[f], range_sum[f]);
                    let mut low = ZERO;
                    if live_a { if n < tokens { low.a = *slot(row_a, n); } if n + 1 < tokens { low.b = *slot(row_a, n + 1); } }
                    if live_b { if n < tokens { low.c = *slot(row_b, n); } if n + 1 < tokens { low.d = *slot(row_b, n + 1); } }
                    word[f] = add4!(low, high);
                }
                f += 1;
            }
            part += 1;
        }
        let put = |row: usize, n: u32, value: f32| {
            if n < tokens { *slot(row, n) = if job.round { crate::expert_ops::round(value) } else { value }; }
        };
        let mut f = 0;
        while f < NF {
            let n = f as u32 * 8 + 2 * t;
            if live_a { put(row_a, n, word[f].a); put(row_a, n + 1, word[f].b); }
            if live_b { put(row_b, n, word[f].c); put(row_b, n + 1, word[f].d); }
            f += 1;
        }
    }
}

/// One block of a row-group launch on the body `tile_body`: part `part`, token tile `tile` (tokens `tile * 8 * NF ..`
/// of the part), row group `row_group` (rows `row_group * 16 * warps ..`). The checks of `group`.
#[inline(always)]
pub(crate) fn group_tile<const NF: usize, const PF: usize>(p: GroupArguments, part: usize, tile: usize, row_group: usize) {
    // SAFETY: As `group`: the host checks every part and keeps all owners and the table alive through stream
    // completion; weights, scales and inputs are immutable; the output is private and each (part, row, token) word
    // has exactly one thread that writes and reads it (`tile_body`). The body traps unless the block is whole
    // warps, NF is even and from 2 to 8, one word is an output, the part is an Fp8Rows part with the first part's
    // rows and cols, cols is a multiple of 16, block_cols a power of two of at least 16, block_rows at least 1,
    // weight and input are 16-byte aligned, scales 4-byte aligned, the row stride is cols (a multiple of 4 words),
    // the part is not a raw quarter job, and the dynamic shared region (the only shared allocation of the kernels
    // that reach this body, so it starts at 0) holds `tile_shared(NF)`. A block with no part, no row or no token
    // returns before any barrier, all its threads alike.
    unsafe {
        use core::arch::asm;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let warps = threads / 32;
        let dynamic: u32;
        asm!("mov.u32 {d}, %dynamic_smem_size;", d = out(reg32) dynamic, options(nomem, nostack));
        if threads % 32 != 0 || warps == 0 || NF < 2 || NF > 8 || NF % 2 != 0 || p.count == 0 || p.words != 1 { asm!("trap;"); }
        if part >= p.count as usize { return; }
        let table = core::slice::from_raw_parts(p.table as *const GroupEntry, p.count as usize);
        let (first, e) = (&table[0], &table[part]);
        let (rows, cols, m) = (e.shape.rows as usize, e.shape.cols as usize, e.shape.tokens as usize);
        let (block_rows, block_cols) = (e.shape.block_rows as usize, e.shape.block_cols as usize);
        let (raw, round) = flags(e.shape.role);
        if e.shape.op != Op::Fp8Rows || cols % 16 != 0 || !block_cols.is_power_of_two() || block_cols < 16 || block_rows == 0
            || rows != first.shape.rows as usize || cols != first.shape.cols as usize || (e.weight | p.input) & 15 != 0 || e.scales & 3 != 0
            || p.stride as usize != cols || raw { asm!("trap;"); }
        let (row0, token0) = (row_group * ROWS * warps, tile * TILE * NF);
        if row0 >= rows || token0 >= m { return; }
        let tokens = if m - token0 < TILE * NF { m - token0 } else { TILE * NF };
        if (dynamic as usize) < tile_shared(NF) { asm!("trap;"); }
        let job = Job { weights: e.weight as *const u8, scales: e.scales as *const f32, rows, cols, block_rows, block_cols, row0,
            input: (p.input as *const f32).add((e.first_row as usize + token0) * cols), stride: cols, tokens,
            output: (p.output as *mut f32).add(e.first as usize + token0), pitch: m, raw, round, tiles: 1 };
        tile_body::<NF, PF>(&job, 0);
    }
}

/// Shared bytes of a `tile3_body` block of `warps` warps: 2 double-buffered stages of BF16 operands (`2 * tile_shared`),
/// then `16 * PARK` bytes for each lane and tile (`PARK` 1: the waiting range sum; 2: and the low join; 0: none).
pub(crate) const fn tile3_shared(nf: usize, warps: usize, park: usize) -> usize { 2 * tile_shared(nf) + warps * 32 * nf * if park == 1 { 16 } else if park == 2 { 32 } else { 0 } }

/// The body of the prompt tile, third form: the word is the contract word (`tc_f8`), as `run`, `row_body` and
/// `tile_body` give. A warp owns 16 weight rows (`job.row0 + 16 * warp ..`) and `NF` tiles of 8 token rows (an even
/// number, 2 to 8). The lane layout, the operand slots and the output words are those of `tile_body`.
///
/// Why (MEASURED 2026-11-16, run `yard-1` of the yardstick seat): a plain BF16 product with both operands packed
/// takes 209 microseconds on 2,048 x 2,048 at 1,024 token rows, and `tile_body` takes 534: the time is in the way a
/// warp gets its operands. Here:
/// - a STAGE is one scale block of 128 columns (the caller checks that the scale blocks are 128 columns wide): no
///   test of the column inside the step loop, and the scale enters at the end of the stage;
/// - a warp stages the activations of one token row at a time: its 32 threads read the 512 contiguous bytes of the
///   stage's 128 columns, round them and store the operand pairs;
/// - a weight load is asked for one step ahead of its use;
/// - `DEC` = 0: THE TENSOR CORE DECODES THE WEIGHTS. A lane's weight bytes are the A operand of two E4M3 steps
///   (`m16n8k32`, the other 16 columns +0) whose B operand selects one column a token place with the value 1.0
///   (code 0x38) and whose accumulator is +0.0. Word (row, place) of such a step has ONE term that is not zero, the
///   byte's value times 1.0, so it is that value exactly, as FP32; `cvt.rn.bf16x2.f32` then packs two of them, exact
///   again (an E4M3 value is a BF16 value). Two steps give the lane's eight weights. Against `tc_f8::e4m3_bf16`: the
///   same BF16 value for every code but 0x7f and 0xff (outside the contract; NaN here, 480 there) and 0x80 (+0 here,
///   -0 there: a zero operand, which brings no term to the BF16 step, so no bit of the word differs).
///   `DEC` = 1: the integer function `tc_f8::e4m3x2_bf16x2`, as the other bodies (a bench form: it proves on the
///   device that the two decodes give one word, and what the integer one costs).
/// - `PARK`: where a sum waits. The range sum that waits for its partner (R0, then R2): in registers (0) or in
///   shared memory (1, 2). The join of ranges 0 and 1 while ranges 2 and 3 are summed: in the word's own output slot
///   as `tile_body` (0, 1) or in shared memory (2).
/// - Every address inside the stage is a register plus a constant (the step, the tile): nothing is kept a step or a
///   tile but the sums.
/// - `PF`: while a stage is worked, the next stage's lines of the block's weight rows and token rows are pulled into
///   L2 (one request a 64-byte line: weights by the four lanes of a row pair, activations by lanes 0, 4, .. of the warp
///   that rounds the token row).
/// A row past `rows` loads row `row0` in its place and is never stored. A token place past `tokens` gets +0 operands
/// and is never stored: its tile is still multiplied (no test in the step loop).
#[inline(always)]
pub(crate) fn tile3_body<const NF: usize, const DEC: usize, const PARK: usize, const PF: bool>(job: &Job, smem: u32) {
    // SAFETY: Private to `group_tile3` behind the third-form tile entries of this crate, which reach it with every
    // thread of the block (no early exit after this call starts), the same `job` on every thread, a block of 1 to
    // 32 whole warps, `NF` even and from 2 to 8, `row0` below `rows` and a multiple of 16, `cols` a multiple of 128,
    // `block_cols` 128, `block_rows` a multiple of 16, `weights` and `input` 16-byte aligned, `stride` a multiple of 4
    // words, `scales` 4-byte aligned, 1 to 8 * NF tokens, not a raw job, and `smem` the start of
    // `tile3_shared(NF, warps, PARK)` bytes that belong to this block. Global reads: a weight load is 4 whole columns
    // of a row below `rows` at a column below `cols`
    // (a lane whose row is past `rows` reads row `row0`); an activation load is 4 whole columns of a token below
    // `tokens` at a column below `cols`; a scale load is the word of a row below `rows` and a block below
    // `cols / 128`; a prefetch names a column below `cols` of a live row or token and reads nothing. Weights, scales
    // and inputs are immutable (the non-coherent loads need that). Shared memory:
    // between the two barriers of a stage every operand slot is written by exactly one thread (slot (step, place,
    // quarter, tile) by lane `step * 4 + quarter` of the warp `place % warps`), and after the second barrier a lane
    // reads only operand slots, all written in this stage; a lane's parked sums (`PARK`) are its own 16 bytes a tile,
    // written before they are read. Every thread reaches every barrier: their number depends only on `cols`. Global
    // writes and their reads back: the one output word of each (row, token) a lane owns, row below `rows` and token
    // below `tokens`, as in `tile_body`; the output is private until the stream completes. The host keeps every owner
    // alive.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($a:expr,$b:expr)=>{{let value:f32;asm!("add.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add4 {($x:expr,$y:expr)=>{{let (x,y):(Acc,Acc)=($x,$y);Acc {a:add!(x.a,y.a),b:add!(x.b,y.b),c:add!(x.c,y.c),d:add!(x.d,y.d)}}};}
        // Two FP32 values to one register of two BF16 patterns, the first in the low half.
        macro_rules! pack {($first:expr,$second:expr)=>{{
            let packed:u32;
            asm!("cvt.rn.bf16x2.f32 {packed}, {second}, {first};",packed=out(reg32)packed,first=in(reg32)$first,second=in(reg32)$second,options(nomem,nostack));
            packed
        }};}
        // The E4M3 step: 16 rows x 8 places x 32 columns, `D = A * B + C`.
        macro_rules! mma8 {($a:expr,$b:expr)=>{{
            let (a, b): ([u32; 4], [u32; 2]) = ($a, $b);
            let (d0, d1, d2, d3): (f32, f32, f32, f32);
            asm!("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
                d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
                a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3], b0 = in(reg32) b[0], b1 = in(reg32) b[1],
                c0 = in(reg32) 0.0f32, c1 = in(reg32) 0.0f32, c2 = in(reg32) 0.0f32, c3 = in(reg32) 0.0f32, options(nomem, nostack));
            Acc { a: d0, b: d1, c: d2, d: d3 }
        }};}
        // Constant indices only, so every array element below is a register.
        macro_rules! rep4 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}{const $v:usize=2;$body}{const $v:usize=3;$body}}};}
        macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const $v:usize=4;$body}{const $v:usize=5;$body}{const $v:usize=6;$body}{const $v:usize=7;$body}}};}
        let threads = _block_dim_x() as u32;
        let warps = threads / 32;
        let tid = _thread_idx_x() as u32;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let (rows, cols, tokens) = (job.rows, job.cols, job.tokens as u32);
        let blocks = cols / SPAN;
        let nf = NF as u32;
        // Rows and pointers. A dead row reads the block's first row.
        let tile_row = job.row0 + warp as usize * ROWS;
        let (row_a, row_b) = (tile_row + g as usize, tile_row + g as usize + 8);
        let (live_a, live_b) = (row_a < rows, row_b < rows);
        let (read_a, read_b) = (if live_a { row_a } else { job.row0 }, if live_b { row_b } else { job.row0 });
        let weights_a = job.weights.add(read_a * cols + 4 * t as usize);
        let weights_b = job.weights.add(read_b * cols + 4 * t as usize);
        // The warp's 16 rows are inside one scale row (`block_rows` is a multiple of 16).
        let scales = job.scales.add((if tile_row < rows { tile_row } else { job.row0 }) / job.block_rows * blocks);
        // The B operand of the decode steps: place g of this lane selects column 4 (g / 2) + g % 2 of the step's low
        // 16 (the lane of quarter g / 2 holds it, byte g % 2), and two columns further in the second step.
        let select = if g >> 1 == t { 0x38u32 << (8 * (g & 1)) } else { 0 };
        // The weights of this lane for step `S` of the stage whose first columns are at the two pointers: 4 bytes of
        // row g, then 4 of row g + 8. The step is a constant offset in the address, so no pointer is kept a step.
        macro_rules! fetch {($a:expr,$b:expr,$step:expr)=>{{
            let (w0, w1): (u32, u32);
            asm!("ld.global.nc.u32 {v}, [{pointer}+{o}];", v = out(reg32) w0, pointer = in(reg64) $a, o = const 16 * $step, options(nostack, readonly));
            asm!("ld.global.nc.u32 {v}, [{pointer}+{o}];", v = out(reg32) w1, pointer = in(reg64) $b, o = const 16 * $step, options(nostack, readonly));
            [w0, w1]
        }};}
        macro_rules! prefetch {($pointer:expr)=>{asm!("prefetch.global.L2 [{pointer}];",pointer=in(reg64)$pointer,options(nostack,readonly));};}
        // The next stage's line this lane asks for, from its own weight pointer of the row (two lanes a row, one line
        // each): back to the row's column 0 of the stage, one stage on, then the lane's line.
        let lead = (SPAN - 4 * t as usize + 64 * (t as usize & 1)) as u32;
        // The A operand of the BF16 step from a fetch: rows g and g + 8, the lane's columns 0 and 1 in the first two
        // registers, columns 2 and 3 in the last two.
        macro_rules! operand {($w:expr)=>{{
            let w: [u32; 2] = $w;
            if DEC == 1 { [crate::tc_f8::e4m3x2_bf16x2(w[0]), crate::tc_f8::e4m3x2_bf16x2(w[1]), crate::tc_f8::e4m3x2_bf16x2(w[0] >> 16), crate::tc_f8::e4m3x2_bf16x2(w[1] >> 16)] }
            else {
                let low = mma8!([w[0], w[1], 0, 0], [select, 0]);
                let high = mma8!([w[0], w[1], 0, 0], [select << 16, 0]);
                [pack!(low.a, low.b), pack!(low.c, low.d), pack!(high.a, high.b), pack!(high.c, high.d)]
            }
        }};}
        // Four words of shared memory at a constant offset from a register (no address register a load).
        macro_rules! quad {($p:expr,$o:expr)=>{{
            let (a, b, c, d): (u32, u32, u32, u32);
            asm!("ld.shared.v4.b32 {{{a},{b},{c},{d}}}, [{p}+{o}];", a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg32) $p, o = const $o, options(nostack, readonly));
            [a, b, c, d]
        }};}
        macro_rules! sums {($p:expr,$o:expr)=>{{ let v = quad!($p, $o); Acc { a: f32::from_bits(v[0]), b: f32::from_bits(v[1]), c: f32::from_bits(v[2]), d: f32::from_bits(v[3]) } }};}
        macro_rules! park {($p:expr,$o:expr,$v:expr)=>{{
            let v: Acc = $v;
            asm!("st.shared.v4.f32 [{p}+{o}], {{{a},{b},{c},{d}}};", p = in(reg32) $p, o = const $o, a = in(reg32) v.a, b = in(reg32) v.b, c = in(reg32) v.c, d = in(reg32) v.d, options(nostack));
        }};}
        // This lane's operands of step s of the stage: `own_off + stage_base + s * 256 * NF`, 8 bytes a tile.
        let stage_bytes = tile_shared(NF) as u32;
        let own_off = ((g * 4 + t) * nf) * 8;
        // This lane's parked sums: `16 * PARK` bytes a tile (the waiting range sum, then with `PARK` 2 the low join).
        let parked = smem + 2 * stage_bytes + tid * nf * if PARK == 2 { 32 } else { 16 };
        // The output words of this lane in tile J: registers a, b = row g, tokens 8 J + 2 t and the next; c, d = row
        // g + 8. Word (J, 0) is live when 8 J < `live`, word (J, 1) when 8 J + 1 < `live`. Compute output pointers
        // only on part 1/3 so LLVM does not keep them live across the inner K loop (<= 128 regs -> 2 CTAs/SM, 0 spills).
        let live = tokens as i32 - 2 * t as i32;
        let mut held = [ZERO; 8];
        // The stage's weights of the two rows: the walk is ascending and whole, so the pointers only step on.
        let (mut wa, mut wb) = (weights_a, weights_b);
        macro_rules! stage_acts {($blk:expr)=>{
            let s_base = smem + (($blk & 1) as u32) * stage_bytes;
            let c0 = $blk * SPAN;
            let more = PF && c0 + SPAN < cols;
            let mut place = warp;
            while place < 8 * nf {
                let (mut low, mut high) = (0u32, 0u32);
                if place < tokens {
                    let pointer = (job.input as *const u8).add((place as usize * job.stride + c0 + 4 * lane as usize) * 4);
                    if more && lane % 4 == 0 { prefetch!(pointer.add(SPAN * 4)); }
                    let (x0, x1, x2, x3): (f32, f32, f32, f32);
                    asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];", a = out(reg32) x0, b = out(reg32) x1, c = out(reg32) x2, d = out(reg32) x3, pointer = in(reg64) pointer, options(nostack, readonly));
                    low = pack!(x0, x1);
                    high = pack!(x2, x3);
                }
                let q = (((lane / 4) * 8 + place % 8) * 4 + lane % 4) * nf + place / 8;
                asm!("st.shared.v2.b32 [{p}], {{{low},{high}}};", p = in(reg32) s_base + q * 8, low = in(reg32) low, high = in(reg32) high, options(nostack));
                place += warps;
            }
        };}
        // Double-buffered activation staging: stage block 0 before the loop, then overlap staging block + 1 with
        // only ONE block barrier per 128-column scale block.
        if blocks > 0 { stage_acts!(0); }
        crate::ptx::sync();
        let mut part = 0;
        while part < 4 {
            let (first, last) = crate::tc_f8::range(blocks, part);
            let mut range_sum = [ZERO; 8];
            let mut block = first;
            while block < last {
                let c0 = block * SPAN;
                let mut now = fetch!(wa, wb, 0);
                if PF && c0 + SPAN < cols { prefetch!((if t < 2 { wa } else { wb }).add(lead as usize)); }
                let own = smem + ((block & 1) as u32) * stage_bytes + own_off;
                let mut sum = [ZERO; 8];
                rep8!(S, {
                    let next = if S + 1 < STEPS { fetch!(wa, wb, (S + 1) % STEPS) } else { [0u32; 2] };
                    let a = operand!(now);
                    rep4!(P, {
                        if 2 * P < NF {
                            let o = quad!(own, S * 256 * NF + 16 * P);
                            sum[2 * P] = madd(a, [o[0], o[1]], sum[2 * P]);
                            sum[2 * P + 1] = madd(a, [o[2], o[3]], sum[2 * P + 1]);
                        }
                    });
                    now = next;
                });
                // The end of the scale block: the scale enters, the block joins its range.
                let scale = *scales.add(block);
                rep8!(J, {
                    if J < NF {
                        let (v, r) = (sum[J], range_sum[J]);
                        range_sum[J] = Acc { a: add!(r.a, mul!(v.a, scale)), b: add!(r.b, mul!(v.b, scale)), c: add!(r.c, mul!(v.c, scale)), d: add!(r.d, mul!(v.d, scale)) };
                    }
                });
                // Stage the next block into the opposite buffer (safe: no warp reads it during this block) and sync ONCE.
                if block + 1 < blocks { stage_acts!(block + 1); }
                crate::ptx::sync();
                (wa, wb) = (wa.add(SPAN), wb.add(SPAN));
                block += 1;
            }
            // The range is closed (an empty range closes with +0.0): fl(fl(R0 + R1) + fl(R2 + R3)).
            let out_a = job.output.wrapping_add(row_a * job.pitch + 2 * t as usize);
            let out_b = job.output.wrapping_add(row_b * job.pitch + 2 * t as usize);
            rep8!(J, {
                if J < NF {
                    let (one, two) = ((8 * J as i32) < live, (8 * J as i32 + 1) < live);
                    let r = range_sum[J];
                    if part == 0 || part == 2 {
                        if PARK != 0 { park!(parked, 16 * PARK * J, r); } else { held[J] = r; }
                    } else {
                        let waiting = if PARK != 0 { sums!(parked, 16 * PARK * J) } else { held[J] };
                        let joined = add4!(waiting, r);
                        if part == 1 {
                            if PARK == 2 { park!(parked, 32 * J + 16, joined); }
                            else {
                                if live_a { if one { *out_a.add(8 * J) = joined.a; } if two { *out_a.add(8 * J + 1) = joined.b; } }
                                if live_b { if one { *out_b.add(8 * J) = joined.c; } if two { *out_b.add(8 * J + 1) = joined.d; } }
                            }
                        } else {
                            let mut low = ZERO;
                            if PARK == 2 { low = sums!(parked, 32 * J + 16); }
                            else {
                                if live_a { if one { low.a = *out_a.add(8 * J); } if two { low.b = *out_a.add(8 * J + 1); } }
                                if live_b { if one { low.c = *out_b.add(8 * J); } if two { low.d = *out_b.add(8 * J + 1); } }
                            }
                            let word = add4!(low, joined);
                            let finish = |value: f32| if job.round { crate::expert_ops::round(value) } else { value };
                            if live_a { if one { *out_a.add(8 * J) = finish(word.a); } if two { *out_a.add(8 * J + 1) = finish(word.b); } }
                            if live_b { if one { *out_b.add(8 * J) = finish(word.c); } if two { *out_b.add(8 * J + 1) = finish(word.d); } }
                        }
                    }
                }
            });
            part += 1;
        }
    }
}

/// One block of a row-group launch on the body `tile3_body`: part `part`, token tile `tile` (tokens `tile * 8 * NF ..`
/// of the part), row group `row_group` (rows `row_group * 16 * warps ..`). The checks of `group_tile`, and the third
/// form's own: 1 to 32 warps, scale blocks 128 columns wide and a multiple of 16 rows high, whole stages.
#[inline(always)]
pub(crate) fn group_tile3<const NF: usize, const DEC: usize, const PARK: usize, const PF: bool>(p: GroupArguments, part: usize, tile: usize, row_group: usize) {
    // SAFETY: As `group_tile`: the host checks every part and keeps all owners and the table alive through stream
    // completion; weights, scales and inputs are immutable; the output is private and each (part, row, token) word
    // has exactly one thread that writes and reads it (`tile3_body`). The body traps unless the block is 1 to 32
    // warps, NF is even and from 2 to 8, one word is an output, the part is an Fp8Rows part with the first part's
    // rows and cols, cols is a multiple of 128, block_cols is 128, block_rows a multiple of 16 (not zero), weight and
    // input are 16-byte aligned, scales 4-byte aligned, the row stride is cols (a multiple of 4 words), the part is
    // not a raw quarter job, and the dynamic shared region (the only shared allocation of the kernels that reach this
    // body, so it starts at 0) holds `tile3_shared(NF, warps, PARK)`. A block with no part, no row or no token
    // returns before any barrier, all its threads alike.
    unsafe {
        use core::arch::asm;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let warps = threads / 32;
        let dynamic: u32;
        asm!("mov.u32 {d}, %dynamic_smem_size;", d = out(reg32) dynamic, options(nomem, nostack));
        if threads % 32 != 0 || warps == 0 || warps > 32 || NF < 2 || NF > 8 || NF % 2 != 0 || DEC > 1 || PARK > 2 || p.count == 0 || p.words != 1 { asm!("trap;"); }
        if part >= p.count as usize { return; }
        let table = core::slice::from_raw_parts(p.table as *const GroupEntry, p.count as usize);
        let (first, e) = (&table[0], &table[part]);
        let (rows, cols, m) = (e.shape.rows as usize, e.shape.cols as usize, e.shape.tokens as usize);
        let (block_rows, block_cols) = (e.shape.block_rows as usize, e.shape.block_cols as usize);
        let (raw, round) = flags(e.shape.role);
        if e.shape.op != Op::Fp8Rows || cols == 0 || cols % SPAN != 0 || block_cols != SPAN || block_rows == 0 || block_rows % ROWS != 0
            || rows != first.shape.rows as usize || cols != first.shape.cols as usize || (e.weight | p.input) & 15 != 0 || e.scales & 3 != 0
            || p.stride as usize != cols || raw { asm!("trap;"); }
        let (row0, token0) = (row_group * ROWS * warps, tile * TILE * NF);
        if row0 >= rows || token0 >= m { return; }
        let tokens = if m - token0 < TILE * NF { m - token0 } else { TILE * NF };
        if (dynamic as usize) < tile3_shared(NF, warps, PARK) { asm!("trap;"); }
        let job = Job { weights: e.weight as *const u8, scales: e.scales as *const f32, rows, cols, block_rows, block_cols, row0,
            input: (p.input as *const f32).add((e.first_row as usize + token0) * cols), stride: cols, tokens,
            output: (p.output as *mut f32).add(e.first as usize + token0), pitch: m, raw, round, tiles: 1 };
        tile3_body::<NF, DEC, PARK, PF>(&job, 0);
    }
}
