//! The Inkling 2 Mini kernel family: VADE's third kernel image (`cargo kernels --inkling`, Ruling R6h). It holds only
//! Inkling's own entries (the block arms and the packed product), so the general image (`spark-rust-kernels`) stays
//! byte for byte the image of private main
//! and no change here can move a register, an instruction or a build unit of another family's kernels.
//! The arm bodies are the safe shared files `block_*.rs` (one source with the host twins). The one unsafe block of
//! the image is in `run`, written out in full (no macro), with its contract above it; the staged BF16 products
//! have their own counted sites in `plain_staged.rs`, `plain_staged_rows.rs` and `plain_staged_narrow.rs`; the panic
//! handler traps. `run` and the
//! entries are `unsafe fn`, as the tensor-core image declares its entry (review REVIMAGE finding 4).
//!
//! How a seat adds an entry (see also `src/inkling_launch.rs`): a kind number in `inkling_launch.rs` and in the host's
//! `Kind`; one arm in the `match` of `run` below (its body as a safe function of its own file, called with the slices
//! that `run` makes from checked counts) and one `entry` line at the end; its row in `REGISTERS.txt`.
// Off the device target this crate is EMPTY: a host build of the whole workspace passes over it.
#![cfg(target_arch = "nvptx64")]
#![no_std]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]
#![feature(core_float_math)]
#![allow(dead_code)]

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _syncthreads, _thread_idx_x};
use core::panic::PanicInfo;
#[path = "../../kernels/src/ptx.rs"]
mod ptx;
#[path = "../../src/fixed_math.rs"]
mod fixed_math;
#[path = "../../src/block_ops.rs"]
mod block_ops;
#[path = "../../src/block_rows.rs"]
mod block_rows;
#[path = "../../src/block_scores.rs"]
mod block_scores;
#[path = "../../src/block_attention.rs"]
mod block_attention;
#[path = "../../src/block_stream.rs"]
mod block_stream;
#[path = "../../src/block_experts.rs"]
mod block_experts;
#[path = "../../src/inkling_launch.rs"]
mod inkling_launch;
#[path = "../../src/staged_product.rs"]
mod staged_product;
mod plain_staged;
mod plain_staged_rows;
#[path = "../../src/staged_narrow.rs"]
mod staged_narrow;
mod plain_staged_narrow;
#[path = "../../src/packed_product.rs"]
mod packed_product;
mod plain_packed;
/// The correctly rounded square root of the arms (`block_ops::root`): `sqrt.rn.f32`, safe core math.
mod inkling_math {
    #[inline(always)]
    pub fn root(x: f32) -> f32 { core::f32::math::sqrt(x) }
}
use block_ops::device::{Arguments, RowsArguments, FIRST, LAST, OP_AT};
use inkling_launch::{Command, BLOCK_ARMS, BLOCK_ROWS, BLOCK_SCORES, BLOCK_SOFTMAX, PLAIN_PACKED, PLAIN_PACKED_GROUP, PLAIN_PACKED_PICKED};

/// The replay of one recorded command of an Inkling kind: `table[index]`, launched in the geometry the host recorded.
/// SAFETY (contract of the caller, the four entries block_arms, block_rows, block_scores and block_softmax): `table` and `index` are the arguments the host passes to the
/// entry (`Backend::resident_causal_into` and the recorded replay); the block inside names what that host checks.
#[inline(always)]
unsafe fn run<const KIND: u32>(table: *const Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector: a recorded
    // table (Backend::record_resident_block) or the one-command table Backend::resident_causal_into uploads and checks
    // for alignment; no replay writes it. A command of another kind traps. The op word is read as a u32 at OP_AT and
    // the payload is taken as Arguments only when it names an arm (FIRST..=LAST), so no invalid Op value is read.
    // The host made the command only after block_check::operands passed: each input and the weight hold exactly the
    // words block_ops::checked_lengths (BLOCK_ARMS) or block_rows::lengths (BLOCK_ROWS) names for this shape, on this
    // context, contiguous and 4-byte aligned; the output holds exactly units*words, private, contiguous, and overlaps
    // no input and not the weight. So each slice made below lies inside its buffer, and the input slices are read only.
    // Writes: in the one-thread-a-unit form, thread `index < units` writes only its own unit (index*words..+words). In
    // the shared-sum form (block_ops::shared_sum, taken only when this launch has exactly `blocks` blocks of
    // SUM_THREADS threads, read here), blocks*span = units*words with span = block_ops::shared_span (a multiple of
    // SUM_THREADS that divides the width; host test), so thread t of block b writes the each = span/SUM_THREADS words
    // b*span + t*each ..+each, inside the output, each once and by no other thread (host test). The block's shared words: thread p < parts
    // (parts <= LANES, the 4*LANES bytes reserved) stores word p before the barrier, every thread of the block reaches
    // the barrier (no thread leaves before it), and after it the threads only read. The rows form keeps one thread a
    // (row, unit) and writes only its own unit (index < units of block_rows::lengths). The staged RowDots form
    // (block_ops::dot_stage, taken only when this launch has exactly `dots` blocks of SUM_THREADS threads, read here):
    // block b is dot b < dots = units (one word a unit), so thread 0 of block b writes output word b alone; its input
    // and weight rows are slices of the checked slices (a panic traps). Shared words: tile word j (and DOT_TILE + j)
    // has one writer, thread j mod threads, before a barrier, thread 0 reads after it, and the second barrier ends the
    // reads before the next tile's writes; every thread of the block runs every barrier (the loop bounds are uniform).
    // The staged SinkSoftmax form (block_attention::softmax_stage, taken only with heads blocks of SUM_THREADS
    // threads): block b is head b, and its words b*rows..(b+1)*rows (the units of the head) are written only by it,
    // word j by thread j mod threads. Stage words 2t and 2t+1 (t < SUM_THREADS) are written by thread t alone before
    // a barrier and read by thread 0 after it; stage word 2*SUM_THREADS + t (< 3*SUM_THREADS <= 2*DOT_TILE) by thread
    // t alone in each group, read by every thread after the next barrier and before the one after it; thread 0 alone
    // writes sum word APART (the top). The loop bounds (n, the live count, and the groups) are the same for the whole
    // block, so every thread reaches every barrier. The staged WindowValues form (block_attention::values_stage, taken only with units /
    // VALUES_CHANNELS blocks of SUM_THREADS threads): thread t of block b writes stage word t alone, between barriers
    // reached by every thread (the distance loop bounds are the same for the block), and only the thread of part 0
    // writes output word b*VALUES_CHANNELS + t mod VALUES_CHANNELS < units, its own.
    // The staged WindowScores form (KIND == BLOCK_SCORES; its commands are rows commands, checked by
    // block_rows::lengths like BLOCK_ROWS; the launch traps unless its grid is exactly block_scores::stage of the
    // command, which also asks for a shared key plane, own queries and step words, and t <= TAPS_MAX, d/2 <= HALF_MAX):
    // the query words item*d + c (item < ITEMS, c < d <= 2*HALF_MAX, so < QUERY_WORDS) are written by one thread each
    // before the first barrier and only read after it; each query row (row < rows) of block (group, chunk) is taken by
    // one warp (rows w, w + 4, ..), and lane p writes only the words of distance (slot - first - p) mod r of that row's
    // t heads group*t + head (out_index < rows*heads*r = units): each distance once (a stored row is one distance of a
    // slot); the tile words p*(half + 1) + j (p < TILE_KEYS, j < half <= HALF_MAX, so < TILE_WORDS) are written by one
    // thread each before a barrier and only read after it, and the tile loop bounds (r) are the same for the whole
    // block, so every thread reaches every barrier.
    // The staged SinkSoftmax at rows (KIND == BLOCK_SOFTMAX; rows commands, checked by block_rows::lengths like
    // BLOCK_ROWS; the launch traps unless its block is SUM_THREADS threads and its grid is exactly
    // block_attention::softmax_rows_stage of the command, rows * h blocks with the scores per row): block b runs its
    // copy of the staged SinkSoftmax body above (the same reads, barriers and shared words) for head b mod h on the
    // slices of row b / h (block_rows::row_inputs, inside the checked lengths: row < rows) and writes only the r words
    // of (row, head) at (row*h + head)*r < rows*h*r, the output count; its shared words are declared for this kind as
    // for BLOCK_ARMS.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != KIND { core::arch::nvptx::trap(); }
        let p = c.payload.as_ptr();
        let op = p.add(OP_AT).cast::<u32>().read();
        if !(FIRST..=LAST).contains(&op) { core::arch::nvptx::trap(); }
        let (a, rows, mask) = match KIND {
            BLOCK_ARMS => (p.cast::<Arguments>().read(), None, 0),
            BLOCK_ROWS | BLOCK_SCORES | BLOCK_SOFTMAX => { let r = p.cast::<RowsArguments>().read(); (r.arguments, Some(r.rows as usize), r.shared) }
            _ => core::arch::nvptx::trap(),
        };
        let counts = match rows { Some(n) => block_rows::lengths(a.shape, n, mask), None => block_ops::checked_lengths(a.shape) };
        let Some((lengths, weights, units, words)) = counts else { return };
        let input: [&[f32]; 4] = core::array::from_fn(|i| core::slice::from_raw_parts(a.input[i] as *const f32, lengths[i]));
        let weight = core::slice::from_raw_parts(a.weight as *const f32, weights);
        let output = a.output as *mut f32;
        let (block, thread, threads) = (_block_idx_x() as usize, _thread_idx_x() as usize, _block_dim_x() as usize);
        // The shared words of the arms entry and of the BLOCK_SOFTMAX entry, declared once (below, `tiles` and `sums`):
        // the stage (two RowDots tiles; or, for a staged SinkSoftmax head, the best (value, distance) of each thread in
        // words 0..2*SUM_THREADS and the block-sum parts in words 2*SUM_THREADS..3*SUM_THREADS; or the parts of a staged
        // WindowValues block) and the sum parts (the top of a staged SinkSoftmax in word APART). The rows entry has none;
        // the BLOCK_SCORES entry declares its own.
        // WindowScores at rows, staged (block_scores; the BLOCK_SCORES entry only): a block a key head and ITEMS / t
        // query rows. The block rounds the queries of its ITEMS items into shared words once; then for each tile it copies
        // TILE_KEYS stored key rows (rows of half + 1 words), warp w takes the query rows w, w + 4, .. of the block, and
        // lane p runs the t chains of that row's heads against tile row p and writes their words.
        if KIND == BLOCK_SCORES {
            let n_rows = match rows { Some(n) => n, None => core::arch::nvptx::trap() };
            if block_scores::stage(a.shape, n_rows, mask) != Some((_grid_dim_x() as usize, threads)) { core::arch::nvptx::trap(); }
            let tile = ptx::shared::<{ 4 * block_scores::TILE_WORDS }>();
            let queries = ptx::shared::<{ 4 * block_scores::QUERY_WORDS }>();
            let s = a.shape;
            let (half, r, width, d, t) = (s.dim as usize / 2, s.rows as usize, (s.heads * s.dim) as usize, s.dim as usize, s.taps as usize);
            let (group, first_row) = block_scores::block_of(s, n_rows, block);
            let (warp, lane, warps) = (thread / block_scores::TILE_KEYS, thread % block_scores::TILE_KEYS, threads / block_scores::TILE_KEYS);
            let mut i = thread;
            while i < block_scores::ITEMS * d {
                let (it, c) = (i / d, i % d);
                let value = match block_scores::item(s, n_rows, group, first_row, it) { Some((row, head)) => fixed_math::rn(input[0][row * width + head * d + c]), None => 0.0 };
                ptx::store(queries + (4 * block_scores::query_at(s, it, c)) as u32, value);
                i += threads;
            }
            let mut first = 0usize;
            while first < r {
                let count = if r - first > block_scores::TILE_KEYS { block_scores::TILE_KEYS } else { r - first };
                let mut i = thread;
                while i < count * half {
                    let (p, j) = (i / half, i % half);
                    ptx::store(tile + (4 * block_scores::tile_at(half, p, j)) as u32, input[1][block_scores::plane_word(s, group, first + p, j)]);
                    i += threads;
                }
                _syncthreads();
                if lane < count {
                    let mut local = warp;
                    while local < block_scores::block_rows_of(s) {
                        let row = first_row + local;
                        if row < n_rows {
                            match block_scores::live(s, row, input[3]) {
                                // A query row whose step words are no count: NaN at the distance numbered like the stored row.
                                None => { let mut head = 0; while head < t { *output.add(block_scores::out_index(s, row, group * t + head, first + lane)) = f32::NAN; head += 1; } }
                                Some((n, slot)) => {
                                    let dist = block_scores::distance(r, slot, first + lane);
                                    let sums = if dist < n {
                                        block_scores::chains_of(t, half, &|head, i| ptx::load(queries + (4 * block_scores::query_at(s, local * t + head, i)) as u32), &|j| ptx::load(tile + (4 * block_scores::tile_at(half, lane, j)) as u32).to_bits())
                                    } else { [0.0; block_scores::TAPS_MAX] };
                                    let bias = block_scores::bias_of(s, row, mask, input[2]);
                                    let mut head = 0;
                                    while head < t {
                                        let word = if dist < n { block_scores::score_word(s, group * t + head, dist, bias, sums[head]) } else { 0.0 };
                                        *output.add(block_scores::out_index(s, row, group * t + head, dist)) = word;
                                        head += 1;
                                    }
                                }
                            }
                        }
                        local += warps;
                    }
                }
                _syncthreads();
                first += count;
            }
            return;
        }
        let (tiles, sums) = if KIND == BLOCK_ARMS || KIND == BLOCK_SOFTMAX { (ptx::shared::<{ 8 * block_ops::DOT_TILE }>(), ptx::shared::<{ 4 * block_ops::LANES }>()) } else { (0, 0) };
        // A staged form runs only in the geometry block_ops::geometry gives it: blocks of SUM_THREADS, `count` blocks.
        let one_grid = |count: usize| rows.is_none() && threads == block_ops::SUM_THREADS && _grid_dim_x() as usize == count;
        // RowDots staged (block_ops::dot_stage): a block a dot. All threads copy a tile of the dot's input row and
        // weight row into shared memory, then thread 0 goes on with the dot's chain over that tile.
        let dots = if rows.is_none() { block_ops::dot_stage(a.shape).filter(|&(count, _)| threads == block_ops::SUM_THREADS && _grid_dim_x() as usize == count) } else { None };
        if let Some((_, k)) = dots {
            let (row, line) = block_ops::split(block, a.shape.heads as usize);
            let (values, weights) = (&input[0][line * k..(line + 1) * k], &weight[row * k..(row + 1) * k]);
            let base = tiles;
            let (mut sum, mut first) = (0.0f32, 0usize);
            while first < k {
                let n = (k - first).min(block_ops::DOT_TILE);
                let mut j = thread;
                while j < n { ptx::store(base + (4 * j) as u32, values[first + j]); ptx::store(base + (4 * (block_ops::DOT_TILE + j)) as u32, weights[first + j]); j += threads; }
                _syncthreads();
                if thread == 0 { sum = block_stream::dot_steps(sum, n, |j| ptx::load(base + (4 * j) as u32), |j| ptx::load(base + (4 * (block_ops::DOT_TILE + j)) as u32)); }
                _syncthreads();
                first += n;
            }
            if thread == 0 { *output.add(block) = sum; }
            return;
        }
        // SinkSoftmax staged (block_attention::softmax_stage): a block a head, any row count. Each thread finds the
        // best of its distances; thread 0 picks the top; SOFTMAX_GROUP blocks of distances at a time, 8 threads each
        // make the parts of the block sums; every thread chains them and writes the weights of its distances.
        // SinkSoftmax at rows (the BLOCK_SOFTMAX entry only, block_attention::softmax_rows_stage): block b runs the
        // staged head below (the same body as the arms entry's) for head b mod h on the inputs of row b / h
        // (block_rows::row_inputs) and writes the words of (row, head). Its own copy, so the arms entry is unchanged.
        if KIND == BLOCK_SOFTMAX {
            let n_rows = match rows { Some(n) => n, None => core::arch::nvptx::trap() };
            if threads != block_ops::SUM_THREADS || block_attention::softmax_rows_stage(a.shape, n_rows, mask) != Some(_grid_dim_x() as usize) { core::arch::nvptx::trap(); }
            let (h, r) = (a.shape.heads as usize, a.shape.rows as usize);
            let (row, head) = (block / h, block % h);
            let Some((one, _, _, _)) = block_ops::checked_lengths(a.shape) else { core::arch::nvptx::trap() };
            let input = block_rows::row_inputs(mask, row, one, input);
            let out = output.add((row * h + head) * r);
            let at = |i: usize| tiles + (4 * i) as u32;
            match block_attention::softmax_live(a.shape, input) {
                None => { let mut j = thread; while j < r { *out.add(j) = f32::NAN; j += threads; } }
                Some(n) => {
                    let scores = &input[0][head * r..(head + 1) * r];
                    let sink = block_attention::softmax_sink(head, input);
                    let (value, dist) = block_attention::softmax_best(n, thread, threads, &|d| scores[d]);
                    ptx::store(at(2 * thread), value);
                    ptx::store(at(2 * thread + 1), f32::from_bits(dist));
                    _syncthreads();
                    let top_at = sums + (4 * block_ops::APART) as u32;
                    if thread == 0 { ptx::store(top_at, block_attention::softmax_pick(sink, (0..threads).map(|t| (ptx::load(at(2 * t)), ptx::load(at(2 * t + 1)).to_bits())))); }
                    _syncthreads();
                    let top = ptx::load(top_at);
                    let e = |d: usize| block_attention::softmax_exp(scores[d], top);
                    let (blocks, parts_at) = (n.div_ceil(block_attention::BLOCK), 2 * block_ops::SUM_THREADS);
                    let (mut chain, mut group) = (0.0f32, 0usize);
                    while group < blocks {
                        let (q, part) = (group + thread / block_ops::APART, thread % block_ops::APART);
                        if q < blocks {
                            let first = q * block_attention::BLOCK;
                            let end = if n - first > block_attention::BLOCK { first + block_attention::BLOCK } else { n };
                            ptx::store(at(parts_at + thread), block_attention::softmax_part_sum(first, end, part, &e));
                        }
                        _syncthreads();
                        let mut k = 0;
                        while k < block_attention::SOFTMAX_GROUP && group + k < blocks {
                            chain = block_attention::softmax_add(chain, core::array::from_fn(|p| ptx::load(at(parts_at + k * block_ops::APART + p))));
                            k += 1;
                        }
                        _syncthreads();
                        group += block_attention::SOFTMAX_GROUP;
                    }
                    let total = block_attention::softmax_total(chain, sink, top);
                    let mut j = thread;
                    while j < r { let w = if j < n { e(j) } else { 0.0 }; *out.add(j) = block_attention::softmax_word(j, n, w, total); j += threads; }
                }
            }
            return;
        }
        if block_attention::softmax_stage(a.shape).is_some_and(one_grid) {
            let (head, r) = (block, a.shape.rows as usize);
            let out = output.add(head * r);
            let at = |i: usize| tiles + (4 * i) as u32;
            match block_attention::softmax_live(a.shape, input) {
                None => { let mut j = thread; while j < r { *out.add(j) = f32::NAN; j += threads; } }
                Some(n) => {
                    let scores = &input[0][head * r..(head + 1) * r];
                    let sink = block_attention::softmax_sink(head, input);
                    let (value, dist) = block_attention::softmax_best(n, thread, threads, &|d| scores[d]);
                    ptx::store(at(2 * thread), value);
                    ptx::store(at(2 * thread + 1), f32::from_bits(dist));
                    _syncthreads();
                    let top_at = sums + (4 * block_ops::APART) as u32;
                    if thread == 0 { ptx::store(top_at, block_attention::softmax_pick(sink, (0..threads).map(|t| (ptx::load(at(2 * t)), ptx::load(at(2 * t + 1)).to_bits())))); }
                    _syncthreads();
                    let top = ptx::load(top_at);
                    let e = |d: usize| block_attention::softmax_exp(scores[d], top);
                    let (blocks, parts_at) = (n.div_ceil(block_attention::BLOCK), 2 * block_ops::SUM_THREADS);
                    let (mut chain, mut group) = (0.0f32, 0usize);
                    while group < blocks {
                        let (q, part) = (group + thread / block_ops::APART, thread % block_ops::APART);
                        if q < blocks {
                            let first = q * block_attention::BLOCK;
                            let end = if n - first > block_attention::BLOCK { first + block_attention::BLOCK } else { n };
                            ptx::store(at(parts_at + thread), block_attention::softmax_part_sum(first, end, part, &e));
                        }
                        _syncthreads();
                        let mut k = 0;
                        while k < block_attention::SOFTMAX_GROUP && group + k < blocks {
                            chain = block_attention::softmax_add(chain, core::array::from_fn(|p| ptx::load(at(parts_at + k * block_ops::APART + p))));
                            k += 1;
                        }
                        _syncthreads();
                        group += block_attention::SOFTMAX_GROUP;
                    }
                    let total = block_attention::softmax_total(chain, sink, top);
                    let mut j = thread;
                    while j < r { let w = if j < n { e(j) } else { 0.0 }; *out.add(j) = block_attention::softmax_word(j, n, w, total); j += threads; }
                }
            }
            return;
        }
        // WindowValues staged (block_attention::values_stage): VALUES_CHANNELS words a block, 8 threads a word, one
        // for each part of its sum; the thread of part 0 joins the parts of each block of distances and writes.
        if block_attention::values_stage(a.shape).is_some_and(one_grid) {
            let (channel, part) = (thread % block_attention::VALUES_CHANNELS, thread / block_attention::VALUES_CHANNELS);
            let index = block * block_attention::VALUES_CHANNELS + channel;
            match block_attention::values_live(a.shape, input) {
                None => { if part == 0 { *output.add(index) = f32::NAN; } }
                Some((n, slot)) => {
                    let (mut total, mut first) = (0.0f32, 0usize);
                    while first < n {
                        let end = if n - first > block_attention::BLOCK { first + block_attention::BLOCK } else { n };
                        ptx::store(tiles + (4 * thread) as u32, block_attention::values_part(a.shape, index, slot, input, first, end, part));
                        _syncthreads();
                        if part == 0 {
                            let parts: [f32; block_ops::APART] = core::array::from_fn(|p| ptx::load(tiles + (4 * (p * block_attention::VALUES_CHANNELS + channel)) as u32));
                            total += block_ops::join_parts(parts);
                        }
                        _syncthreads();
                        first = end;
                    }
                    if part == 0 { *output.add(index) = block_attention::values_word(a.shape, index, input, total); }
                }
            }
            return;
        }
        // The norm and clip arms whose rows share one sum: a block for each `shared_span` words of a row.
        let shared = if rows.is_none() { block_ops::shared_sum(a.shape).filter(|&(blocks, _)| threads == block_ops::SUM_THREADS && _grid_dim_x() as usize == blocks) } else { None };
        if let Some((_, parts)) = shared {
            let span = block_ops::shared_span(a.shape);
            let (row, first) = block_ops::split(block * span, a.shape.dim as usize);
            let base = sums;
            if thread < parts { ptx::store(base + (thread * 4) as u32, block_ops::sum_part(a.shape, row, thread, input, weight)); }
            _syncthreads();
            let mut sums = [0.0f32; block_ops::LANES];
            for (part, sum) in sums.iter_mut().enumerate().take(parts) { *sum = ptx::load(base + (part * 4) as u32); }
            let each = span / block_ops::SUM_THREADS;
            let words = core::slice::from_raw_parts_mut(output.add(block * span + thread * each), each);
            block_ops::shared_words(a.shape, row, first + thread * each, input, weight, &sums, words);
        } else {
            let index = block * threads + thread;
            if index < units {
                let unit = core::slice::from_raw_parts_mut(output.add(index * words), words);
                match rows {
                    Some(n) => block_rows::unit(a.shape, n, mask, index, input, weight, unit),
                    None => block_ops::unit(a.shape, index, input, weight, unit),
                }
            }
        }
    }
}

/// The replay of one recorded packed product (`Backend::record_resident_packed`): the five words of `table[index]`
/// are the unit, the input, the output, the rows and the column word (`packed_product::cols_word`).
/// SAFETY (contract of the caller, the entry): `table` and `index` are the arguments of a recorded replay of kind
/// PLAIN_PACKED; the block inside names what the host checked.
#[inline(always)]
unsafe fn run_packed(table: *const Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (a recorded
    // table); no replay writes it. A command of another kind traps. Its first five 8-byte payload words were written
    // by Backend::record_resident_packed, which Backend::project_packed_into calls only after the host check
    // packed_product::check passed for exactly these words: the unit is exactly layout(rows, cols, wide).bytes long
    // and 16-byte aligned, the input is exactly cols words and 16-byte aligned, the output is exactly rows words,
    // private, contiguous, and overlaps neither. plain_packed::tile reads and writes only inside those three buffers
    // (its own SAFETY note says which check covers each access) and traps on words that `admitted` does not take.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != PLAIN_PACKED { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        plain_packed::tile(*words as *const u8, *words.add(1) as *const f32, *words.add(2) as *mut f32, *words.add(3) as u32, *words.add(4) as u32);
    }
}

/// The replay of one recorded group of packed products (`Backend::record_resident_packed_group`): block row `e` of the
/// grid runs the tile of matrix `e`: its unit (word `GROUP_UNITS_AT + e`), its input and its output rows.
/// SAFETY (contract of the caller, the entry): `table` and `index` are the arguments of a recorded replay of kind
/// PLAIN_PACKED_GROUP; the block inside names what the host checked.
#[inline(always)]
unsafe fn run_packed_group(table: *const Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (a recorded
    // table); no replay writes it. A command of another kind traps. Its words were written by
    // Backend::record_resident_packed_group, which Backend::project_packed_group_into calls only after the host check
    // packed_product::check_group passed for exactly these words: 1 to GROUP_MAX matrices (so word GROUP_UNITS_AT + e
    // is inside the 1024-byte payload), and for each matrix e the single check packed_product::check passed with unit
    // e, the input at `input + e * input_words` words (one input for all when input_words is 0) and the output at
    // `output + e * rows` words; the whole output meets no unit and not the input. A block row at or above the count
    // of matrices traps. So the call of plain_packed::tile below is a call with words the single check passed, and
    // its own SAFETY note holds for it.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != PLAIN_PACKED_GROUP { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (input, output, rows, word, picks, input_words) = (*words as *const f32, *words.add(1) as *mut f32, *words.add(2) as u32, *words.add(3) as u32, *words.add(4) as usize, *words.add(5) as usize);
        let e = core::arch::nvptx::_block_idx_y() as usize;
        if picks == 0 || picks > packed_product::GROUP_MAX || e >= picks { core::arch::nvptx::trap(); }
        let unit = *words.add(packed_product::GROUP_UNITS_AT + e) as *const u8;
        plain_packed::tile(unit, input.add(e * input_words), output.add(e * rows as usize), rows, word);
    }
}

/// The replay of one recorded group of packed products whose matrices the route picked on the device
/// (`Backend::record_resident_packed_picked`): block row `e` of the grid reads expert number `ids[e]` (an FP32 whole
/// number, as `Op::Route` writes it) and runs the tile of that expert's unit in the stack, its input and its output rows.
/// SAFETY (contract of the caller, the entry): `table` and `index` are the arguments of a recorded replay of kind
/// PLAIN_PACKED_PICKED; the block inside names what the host checked.
#[inline(always)]
unsafe fn run_packed_picked(table: *const Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (a recorded
    // table); no replay writes it. A command of another kind traps. Its words were written by
    // Backend::record_resident_packed_picked, which Backend::project_packed_picked_into calls only after the host check
    // packed_product::check_picked passed for exactly these words: 1 to GROUP_MAX picks; `ids` is exactly `picks` words,
    // word aligned; the stack is exactly `experts` units of `unit_bytes` each (layout(rows, cols, wide).bytes, a
    // multiple of 16) and 16-byte aligned, so unit `n` for any `n < experts` is a unit the single check
    // packed_product::check passed with the input at `input + e * input_words` words and the output at
    // `output + e * rows` words; the whole output meets no byte of the stack, of `ids` or of the input, so no block
    // writes a word that another block reads. A block row at or above the picks traps; an id word that is not a whole
    // number at least 0 and below `experts` traps (NaN fails the first compare). So the call of plain_packed::tile below
    // is a call with words the single check passed, and its own SAFETY note holds for it.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != PLAIN_PACKED_PICKED { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (input, output, rows, word, picks, input_words) = (*words as *const f32, *words.add(1) as *mut f32, *words.add(2) as u32, *words.add(3) as u32, *words.add(4) as usize, *words.add(5) as usize);
        let (stack, ids, experts, unit_bytes) = (*words.add(6) as *const u8, *words.add(7) as *const f32, *words.add(8) as u32, *words.add(9) as usize);
        let e = core::arch::nvptx::_block_idx_y() as usize;
        if picks == 0 || picks > packed_product::GROUP_MAX || e >= picks { core::arch::nvptx::trap(); }
        let id = *ids.add(e);
        if !(id >= 0.0 && id < experts as f32) || (id as u32) as f32 != id || id as u32 >= experts { core::arch::nvptx::trap(); }
        plain_packed::tile(stack.add(id as usize * unit_bytes), input.add(e * input_words), output.add(e * rows as usize), rows, word);
    }
}

/// SAFETY: launched only by the host with a table of commands of kind BLOCK_ARMS (`run`, whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_block_arms(table: *const Command, index: u64) { run::<BLOCK_ARMS>(table, index) }
/// SAFETY: launched only by the host with a table of commands of kind BLOCK_ROWS (`run`, whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_block_rows(table: *const Command, index: u64) { run::<BLOCK_ROWS>(table, index) }
/// SAFETY: launched only by the host with a table of commands of kind BLOCK_SCORES (`run`, whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_block_scores(table: *const Command, index: u64) { run::<BLOCK_SCORES>(table, index) }
/// SAFETY: launched only by the host with a table of commands of kind BLOCK_SOFTMAX (`run`, whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_block_softmax(table: *const Command, index: u64) { run::<BLOCK_SOFTMAX>(table, index) }
/// The staged BF16 row product (`plain_staged.rs`); its body reads and checks its own recorded command.
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_STAGED (`plain_staged::replay`, whose
/// contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_staged(table: *const Command, index: u64) { plain_staged::replay(table, index) }
/// The rows tile of the staged BF16 product (`plain_staged_rows.rs`); its body reads and checks its own recorded command.
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_STAGED_ROWS (`plain_staged_rows::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_staged_rows(table: *const Command, index: u64) { plain_staged_rows::replay(table, index) }
/// The narrow staged BF16 row product (`plain_staged_narrow.rs`); its body reads and checks its own recorded command.
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_STAGED_NARROW (`plain_staged_narrow::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_staged_narrow(table: *const Command, index: u64) { plain_staged_narrow::replay(table, index) }
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_PACKED (`run_packed`, whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_packed(table: *const Command, index: u64) { run_packed(table, index) }
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_PACKED_GROUP (`run_packed_group`, whose
/// contract holds), in a grid whose second dimension is the count of matrices of the command.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_packed_group(table: *const Command, index: u64) { run_packed_group(table, index) }
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_PACKED_PICKED (`run_packed_picked`, whose
/// contract holds), in a grid whose second dimension is the count of picks of the command.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_packed_picked(table: *const Command, index: u64) { run_packed_picked(table, index) }

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    // Any panic is a programming defect. Trap, so the host sees a failed launch.
    // SAFETY: `trap` takes no operand and touches no memory; it ends the thread's kernel, which is what a panic means here.
    unsafe { core::arch::nvptx::trap() }
}
