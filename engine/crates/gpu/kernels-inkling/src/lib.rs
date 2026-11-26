//! The Inkling 2 Mini kernel family: VADE's third kernel image (`cargo kernels --inkling`, Ruling R6h). It holds only
//! Inkling's own entries (the block arms and the packed product), so the general image (`spark-rust-kernels`) stays
//! byte for byte the image of private main
//! and no change here can move a register, an instruction or a build unit of another family's kernels.
//! The arm bodies are the safe shared files `block_*.rs` (one source with the host twins). The one unsafe block of
//! the image is in `run`, written out in full (no macro), with its contract above it; the staged BF16 products
//! have their own counted sites in `plain_staged.rs`, `plain_staged_rows.rs` , `plain_staged_narrow.rs`, `plain_staged_few.rs` and `plain_staged_one.rs`; the panic
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
#[path = "../../src/block_values.rs"]
mod block_values;
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
#[path = "../../src/few_product.rs"]
mod few_product;
mod plain_staged_few;
mod plain_staged_one;
#[path = "../../src/bf16_panels.rs"]
mod bf16_panels;
mod markov_fold;
mod draft_lanes;
mod draft_panel;
#[path = "../../src/packed_product.rs"]
mod packed_product;
mod plain_packed;
mod stream_pairs;
mod plain_packed_rows;
mod embed_rows;
#[path = "../../src/tc_rows.rs"]
mod tc_rows;
#[path = "../../src/prefill_moe.rs"]
mod prefill_moe;
#[path = "../../src/prefill_gate.rs"]
mod prefill_gate;
#[path = "../../src/flash_window.rs"]
mod flash_window_abi;
#[path = "../../src/fast_ops.rs"]
mod fast_ops;
mod tc_ptx;
#[macro_use]
mod tc_panels;
mod plain_tc_rows;
mod plain_packed_tc_rows;
mod prefill_packed_tc;
mod prefill_flash_window;
#[path = "prefill_gate.rs"]
mod prefill_gate_kern;
mod plain_tc_big;
mod moe_group_route;
mod flash_window;
mod flash_window_split;
mod fast_mix;
mod fast_norm;
mod fast_elem;
mod fast_arms;
/// The correctly rounded square root of the arms (`block_ops::root`): `sqrt.rn.f32`, safe core math.
mod inkling_math {
    #[inline(always)]
    pub fn root(x: f32) -> f32 { core::f32::math::sqrt(x) }
}
use block_ops::device::{Arguments, RowsArguments, FIRST, LAST, OP_AT};
use inkling_launch::{Command, BLOCK_ARMS, BLOCK_ROWS, BLOCK_SCORES, BLOCK_VALUES,BLOCK_SOFTMAX, PLAIN_PACKED, PLAIN_PACKED_GROUP, PLAIN_PACKED_PICKED};

/// The replay of one recorded command of an Inkling kind: `table[index]`, launched in the geometry the host recorded.
/// SAFETY (contract of the caller, the five entries block_arms, block_rows, block_scores, block_softmax and block_values): `table` and `index` are the arguments the host passes to the
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
    // and weight rows are slices of the checked slices (a panic traps). Product word j has one writer,
    // thread j mod threads, before a barrier. Thread 0 reads aligned groups of four below n, then remaining
    // single words. The tile is 16-byte aligned and the second barrier ends the
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
    // In the solo softmax, the four warp winners occupy words 0..8. When live scores fit, each
    // thread writes its own distances at 3*SUM_THREADS+j below 2*DOT_TILE; a uniform barrier
    // precedes all reads. Part sums end before that cache starts.
    // The warp HeadClip form (block_ops::clip_warps, taken only with exactly heads.div_ceil(SUM_THREADS / LANES) blocks
    // of SUM_THREADS threads; then units = heads and words = d <= LANES): lane l < d of the warp of head h writes output
    // word h*d + l alone, inside units*words, and initializes it before borrowing it; a warp past the last head
    // writes nothing; the shuffles run with the whole
    // warp (head is the same for its lanes).
    // The one-row WindowScores warp form (KIND == BLOCK_ARMS, dim a multiple of 16, SUM_THREADS threads): warp w writes
    // and reads only the tile words w*256..(w+1)*256 (4 warps, inside the 8*DOT_TILE bytes); in each group every word
    // of the quarter is written by one lane, a warp barrier ends the writes before the reads and the reads before the
    // next writes (the group loop bound is the same for the warp, the returns before it are the same for the warp);
    // key reads are at starts[q] + first + word < start + half of a live row (row < r), or of row 0 for a lane with
    // no live unit, inside the checked plane; thread `index < units` writes output word `index` alone.
    // The staged WindowScores form (KIND == BLOCK_SCORES; its commands are rows commands, checked by
    // block_rows::lengths like BLOCK_ROWS; the launch traps unless its grid is exactly block_scores::stage of the
    // command, which also asks for a shared key plane, own queries and step words, and t <= TAPS_MAX, d/2 <= HALF_MAX):
    // the query words item*d + c (item < ITEMS, c < d <= 2*HALF_MAX, so < QUERY_WORDS) are written by one thread each
    // before the first barrier and only read after it; each query row (row < rows) of block (part, group, chunk) is taken
    // by one warp (rows w, w + 4, ..), and lane p writes only the words of distance (slot - first - p) mod r of that row's
    // t heads group*t + head (out_index < rows*heads*r = units): each distance once (a stored row is one distance of a
    // slot, and it lies in one tile, which one part walks: part k takes the tiles k, k + parts, ..); the tile words p*(half + 1) + j (p < TILE_KEYS, j < half <= HALF_MAX, so < TILE_WORDS) are written by one
    // thread each before a barrier and only read after it, and the tile loop bounds (r) are the same for the whole
    // block, so every thread reaches every barrier.
    // The staged SinkSoftmax at rows (KIND == BLOCK_SOFTMAX; rows commands, checked by block_rows::lengths like
    // BLOCK_ROWS; the launch traps unless its block is SUM_THREADS threads and its grid is exactly
    // block_attention::softmax_rows_stage of the command, rows * h blocks with the scores per row): block b runs its
    // copy of the staged SinkSoftmax body above (the same reads, barriers and shared words) for head b mod h on the
    // slices of row b / h (block_rows::row_inputs, inside the checked lengths: row < rows) and writes only the r words
    // of (row, head) at (row*h + head)*r < rows*h*r, the output count; its shared words are declared for this kind as
    // for BLOCK_ARMS.
    // The staged WindowValues at rows (KIND == BLOCK_VALUES; rows commands, checked by block_rows::lengths like
    // BLOCK_ROWS; the launch traps unless its grid is exactly block_values::stage of the command, which also asks for
    // the mask SHARED (one value plane, each row its own weights, gates and step words), 0 < t <= TAPS_MAX, h a multiple
    // of t and d a multiple of CHANNELS): block (chunk, group, slice) reads the weights of its rows (weight_at < rows*h*r,
    // dist < n <= r), the plane words of its slice (plane_word < r*(h/t)*d/2: stored row < r, group < h/t, slice*WORDS
    // + w < d/2) and the gates of its words; it writes only the words (row, group*t + h, slice*CHANNELS + c) of its rows
    // first..end (out_index < rows*h*d = units), each once (host test), after its last barrier. Its tile words
    // tile_at(k, w) (k < tile_count <= TILE_ROWS, w < WORDS, so < TILE_WORDS) are written by one thread each between
    // barriers and only read between the next barriers; its totals (total_at < TOTAL_WORDS) are set by their threads
    // before the first barrier, then read and written only by lane 0 of the one warp that owns the row (rows
    // q = w mod 4), then read by all threads after the last barrier. The distance loop bound (most_live) is the same
    // for the whole block, so every thread reaches every barrier; the shuffles run with the whole warp (the row loop
    // bound is the same for a warp, and every lane reaches them after its distance loop).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != KIND { core::arch::nvptx::trap(); }
        let p = c.payload.as_ptr();
        let op = p.add(OP_AT).cast::<u32>().read();
        if !(FIRST..=LAST).contains(&op) { core::arch::nvptx::trap(); }
        let (a, rows, mask) = match KIND {
            BLOCK_ARMS => (p.cast::<Arguments>().read(), None, 0),
            BLOCK_ROWS | BLOCK_SCORES | BLOCK_SOFTMAX | BLOCK_VALUES => { let r = p.cast::<RowsArguments>().read(); (r.arguments, Some(r.rows as usize), r.shared) }
            _ => core::arch::nvptx::trap(),
        };
        let planes=(KIND==BLOCK_SCORES || KIND==BLOCK_VALUES) && core::arch::nvptx::_grid_dim_y()>1;
        let rows=if planes {
            // SAFETY: block_planes_into checked each owner against the single plane extent
            // and all other operands against the cohort extent. The private command contains
            // one address per chat. Distinct (chat,unit) pairs own disjoint output slices.
            let n=core::arch::nvptx::_grid_dim_y() as usize;
            let op=if KIND==BLOCK_SCORES {block_ops::Op::WindowScores} else {block_ops::Op::WindowValues};
            if rows!=Some(n) || !(2..=inkling_launch::PLANES_MAX).contains(&n) || mask!=2 || a.shape.op!=op {core::arch::nvptx::trap();}
            Some(1)
        } else {rows};
        let counts = match rows { Some(n) => block_rows::lengths(a.shape, n, mask), None => block_ops::checked_lengths(a.shape) };
        let Some((lengths, weights, units, words)) = counts else { return };
        // SAFETY: When `RING_DELTA_AT` word 0 is non-zero (only set by `Backend::block_ring_delta_into` for
        // `BLOCK_SCORES`/`BLOCK_VALUES` with `!planes`), host verified `input[1]` holds `ring_rows * row_stride`
        // words and `delta` holds `(r - prefix) * row_stride` words, contiguous, aligned, and disjoint from output.
        let (delta_addr, ring_rows, start_slot, prefix) = if (KIND == BLOCK_SCORES || KIND == BLOCK_VALUES) && !planes {
            let rd = p.add(inkling_launch::RING_DELTA_AT).cast::<u64>();
            (*rd, *rd.add(1) as usize, *rd.add(2) as usize, *rd.add(3) as usize)
        } else { (0, 0, 0, 0) };
        let row_stride = if delta_addr != 0 {
            let r = a.shape.rows as usize;
            if ring_rows == 0 || prefix == 0 || prefix > ring_rows || prefix >= r || start_slot >= ring_rows
                || delta_addr & 3 != 0 || a.shape.taps == 0 || !a.shape.heads.is_multiple_of(a.shape.taps) || a.shape.dim % 2 != 0
            {
                core::arch::nvptx::trap();
            }
            (a.shape.heads / a.shape.taps) as usize * (a.shape.dim as usize / 2)
        } else if KIND == BLOCK_SCORES || KIND == BLOCK_VALUES {
            (a.shape.heads / a.shape.taps.max(1)) as usize * (a.shape.dim as usize / 2)
        } else { 0 };
        let delta_ptr = delta_addr as *const f32;
        let chat=if planes {core::arch::nvptx::_block_idx_y() as usize} else {0};
        let input: [&[f32]; 4] = core::array::from_fn(|i| {
            let address=if planes && i==1 {p.add(inkling_launch::PLANES_AT+chat*8).cast::<u64>().read() as *const f32}
                else {(a.input[i] as *const f32).add(chat*lengths[i])};
            let len = if i == 1 && delta_addr != 0 { ring_rows * row_stride } else { lengths[i] };
            core::slice::from_raw_parts(address,len)
        });
        let weight = core::slice::from_raw_parts(a.weight as *const f32, weights);
        let output = (a.output as *mut f32).add(chat*units*words);
        let (block, thread, threads) = (_block_idx_x() as usize, _thread_idx_x() as usize, _block_dim_x() as usize);
        // The ring row the host joined into a one-row ShortConv (`resident_conv_ring::ring`, reserved[2] bit 4): word
        // `i` of the new input goes to word `i` of a whole row of the ring (input 0), by the thread of channel `i`, the
        // one thread that reads that channel of the ring below (the joined and the one-thread-a-unit forms, the only
        // forms of this arm). The source meets no word this launch writes (host check). The ring slice is made again
        // after the write, from the address, so no slice made before it is read again.
        let input = if KIND==BLOCK_ARMS && rows.is_none() && a.shape.op==block_ops::Op::ShortConv && c.reserved[2]&4!=0 {
            let (index, width) = (block*threads+thread, (a.shape.heads*a.shape.dim) as usize);
            if index < width {
                let source = *p.cast::<u64>().add(19) as *mut f32;
                let row = *p.cast::<u64>().add(20) as *mut f32;
                // A joined route fold (bit 8): word `index` of the fold, by the chain of the fast fold (fast_elem:
                // rn of each expert word times its weight, in pick order from 0.0, then rn), into the fold's output
                // (the source) first. Its weights and expert rows meet nothing this launch writes (host check).
                let value = if c.reserved[2]&8!=0 {
                    let (weights, downs, picks) = (*p.cast::<u64>().add(21) as *const f32, *p.cast::<u64>().add(22) as *const f32, *p.cast::<u64>().add(23) as usize);
                    if picks == 0 || picks > fast_ops::ROUTER_PICKS_MAX { core::arch::nvptx::trap(); }
                    let mut sum = 0.0f32;
                    let mut q = 0;
                    while q < picks { sum += fixed_math::rn(*downs.add(q*width+index)) * *weights.add(q); q += 1; }
                    let value = fixed_math::rn(sum);
                    source.add(index).write(value);
                    value
                } else { *source.add(index) };
                row.add(index).write(value);
            }
            [core::slice::from_raw_parts(a.input[0] as *const f32, lengths[0]), input[1], input[2], input[3]]
        } else { input };
        let row_dots=if KIND==BLOCK_ROWS && rows.is_some_and(|n|n<=16) && _grid_dim_x() as usize==units && threads==128 {
            block_ops::dot_stage(a.shape)
        } else {None};
        if let Some((dots,k))=row_dots {
            let chat=block/dots;
            let (query,line)=block_ops::split(block%dots,a.shape.heads as usize);
            let source=if mask & block_rows::LIST!=0 {line*rows.unwrap()+chat}
                else if block_rows::is_shared(mask,0) {line} else {chat*a.shape.heads as usize+line};
            let values=&input[0][source*k..(source+1)*k];
            let w=&weight[query*k..(query+1)*k];
            // SAFETY: checked_lengths gives one output word per dot. Each lane writes distinct
            // products in this block's tile. Both barriers have uniform bounds. Only lane zero
            // reads the tile and writes the sum, preserving the ascending addition chain.
            let tile=ptx::shared::<2048>();
            let mut sum=0.0f32;
            for first in (0..k).step_by(512) {
                let n=(k-first).min(512);
                let mut j=thread;
                while j<n {
                    ptx::store(tile+4*j as u32,values[first+j]*w[first+j]);
                    j+=threads;
                }
                _syncthreads();
                if thread==0 {
                    for j in 0..n {sum+=ptx::load(tile+4*j as u32);}
                }
                _syncthreads();
            }
            if thread==0 {*output.add(block)=sum;}
            return;
        }
        if KIND == BLOCK_ROWS && a.shape.op == block_ops::Op::RouterPick && mask == 0 && a.shape.rows <= 128
            && threads == 128 && rows == Some(_grid_dim_x() as usize) {
            let experts = a.shape.rows as usize;
            let picks = a.shape.cols as usize;
            // SAFETY: checked lengths give one logits row and 2*picks output words per block.
            // Each lane writes its own shared score. After the barrier only lane zero uses the tile.
            let tile = ptx::shared::<512>();
            let logits = &input[0][block * experts..(block + 1) * experts];
            if thread < experts {
                ptx::store(tile + 4 * thread as u32, fixed_math::sigmoid(logits[thread]) + weight[thread]);
            }
            core::arch::nvptx::_syncthreads();
            if thread == 0 {
                let ids = output.add(block * words);
                let weights = ids.add(picks);
                let mut sum = 0.0;
                for slot in 0..picks {
                    let (mut best, mut selected) = (f32::NEG_INFINITY, 0usize);
                    for expert in 0..experts {
                        let choice = ptx::load(tile + 4 * expert as u32);
                        if choice > best { best = choice; selected = expert; }
                    }
                    ptx::store(tile + 4 * selected as u32, f32::NEG_INFINITY);
                    let score = fixed_math::sigmoid(logits[selected]);
                    ids.add(slot).write(selected as f32);
                    weights.add(slot).write(score);
                    sum += score;
                }
                // The selection loop initialized all picks before this slice is created.
                for w in core::slice::from_raw_parts_mut(weights, picks) {
                    if a.shape.taps != 0 { *w /= sum + 1e-20; }
                    *w *= a.shape.bound;
                }
            }
            return;
        }
        // The shared words of the arms entry and of the BLOCK_SOFTMAX entry, declared once (below, `tiles` and `sums`):
        // the stage (two RowDots tiles; or, for a staged SinkSoftmax head, the best (value, distance) of each thread in
        // words 0..2*SUM_THREADS and the block-sum parts in words 2*SUM_THREADS..3*SUM_THREADS; or the parts of a staged
        // WindowValues block) and the sum parts (the top of a staged SinkSoftmax in word APART). The rows entry has none;
        // the BLOCK_SCORES entry and the BLOCK_VALUES entry declare their own (`tile`; `tile` and `totals`).
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
            let (group, first_row, part) = block_scores::block_of(s, n_rows, block);
            let parts = block_scores::parts(s, n_rows);
            let (warp, lane, warps) = (thread / block_scores::TILE_KEYS, thread % block_scores::TILE_KEYS, threads / block_scores::TILE_KEYS);
            let mut i = thread;
            while i < block_scores::ITEMS * d {
                let (it, c) = (i / d, i % d);
                let value = match block_scores::item(s, n_rows, group, first_row, it) { Some((row, head)) => fixed_math::rn(input[0][row * width + head * d + c]), None => 0.0 };
                ptx::store(queries + (4 * block_scores::query_at(s, it, c)) as u32, value);
                i += threads;
            }
            let in1_ptr = input[1].as_ptr();
            let group_off = group * half;
            let mut first = part * block_scores::TILE_KEYS;
            while first < r {
                let count = if r - first > block_scores::TILE_KEYS { block_scores::TILE_KEYS } else { r - first };
                let mut i = thread;
                while i < count * half {
                    let (p, j) = (i / half, i % half);
                    let vp = first + p;
                    let (src_ptr, r_idx) = if delta_addr == 0 {
                        (in1_ptr, vp)
                    } else if vp < prefix {
                        let rp = start_slot + vp;
                        (in1_ptr, if rp >= ring_rows { rp - ring_rows } else { rp })
                    } else {
                        (delta_ptr, vp - prefix)
                    };
                    ptx::store(tile + (4 * block_scores::tile_at(half, p, j)) as u32, *src_ptr.add(r_idx * row_stride + group_off + j));
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
                first += parts * block_scores::TILE_KEYS;
            }
            return;
        }
        // WindowValues at rows, staged (block_values; the BLOCK_VALUES entry only): a block a value head, a slice of
        // CHANNELS channels and QUERY_ROWS query rows. For each block of BLOCK distances the block copies the stored rows
        // of the slice into its tile; then warp w takes the query rows w, w + 4, .. of the block, lane l runs lane chain l
        // of the t * CHANNELS words of the row, the warp joins the 32 lane sums by shuffles (block_values::join_down)
        // and lane 0 adds the block sum to the row's chain in the block's shared totals.
        if KIND == BLOCK_VALUES {
            let n_rows = match rows { Some(n) => n, None => core::arch::nvptx::trap() };
            if block_values::stage(a.shape, n_rows, mask) != Some((_grid_dim_x() as usize, threads)) { core::arch::nvptx::trap(); }
            let tile = ptx::shared::<{ 4 * block_values::TILE_WORDS }>();
            let totals = ptx::shared::<{ 4 * block_values::TOTAL_WORDS }>();
            let s = a.shape;
            let (r, t) = (s.rows as usize, s.taps as usize);
            let (group, slice, first) = block_values::block_of(s, block);
            let end = block_values::end_of(n_rows, first);
            let anchor = block_values::anchor(s, end, input[3]);
            let most = block_values::most_live(s, first, end, input[3]);
            let (warp, lane, warps) = (thread / block_ops::LANES, thread % block_ops::LANES, threads / block_ops::LANES);
            let mut i = thread;
            while i < block_values::TOTAL_WORDS { ptx::store(totals + (4 * i) as u32, 0.0); i += threads; }
            // The value of lane `lane + distance` (its own past lane 31): every lane of the warp runs it together.
            let down = |v: f32, distance: u32| -> f32 {
                let got: u32;
                core::arch::asm!("shfl.sync.down.b32 {got}, {v}, {distance}, 31, 0xffffffff;", got = out(reg32) got, v = in(reg32) v.to_bits(), distance = in(reg32) distance);
                f32::from_bits(got)
            };
            let in1_ptr = input[1].as_ptr();
            let col_base = group * (s.dim as usize / 2) + slice * block_values::WORDS;
            let read_val = |vp: usize, w: usize| -> f32 {
                let (src_ptr, r_idx) = if delta_addr == 0 {
                    (in1_ptr, vp)
                } else if vp < prefix {
                    let rp = start_slot + vp;
                    (in1_ptr, if rp >= ring_rows { rp - ring_rows } else { rp })
                } else {
                    (delta_ptr, vp - prefix)
                };
                *src_ptr.add(r_idx * row_stride + col_base + w)
            };
            let mut base = 0usize;
            while base < most {
                let count = block_values::tile_count(base, most);
                let mut i = thread;
                while i < count * block_values::WORDS {
                    let (k, w) = (i / block_values::WORDS, i % block_values::WORDS);
                    ptx::store(tile + (4 * block_values::tile_at(k, w)) as u32, read_val(block_values::tile_row(r, anchor, base, k), w));
                    i += threads;
                }
                _syncthreads();
                let mut q = warp;
                while first + q < end {
                    let row = first + q;
                    if let Some((n, slot)) = block_values::live(s, row, input[3]).filter(|&(n, _)| base < n) {
                        let left = if n - base > block_attention::BLOCK { block_attention::BLOCK } else { n - base };
                        let at = block_values::offset(r, anchor, slot);
                        let mut chains = [[0.0f32; block_values::CHANNELS]; block_values::TAPS_MAX];
                        let mut j = lane;
                        if block_values::in_tile(at) {
                            while j < left {
                                let dist = base + j;
                                block_values::add_terms(&mut chains, t, &|h| input[0][block_values::weight_at(s, row, group * t + h, dist)], &|w| ptx::load(tile + (4 * block_values::tile_at(at + j, w)) as u32).to_bits());
                                j += block_ops::LANES;
                            }
                        } else {
                            while j < left {
                                let dist = base + j;
                                let stored = block_values::stored_row(r, slot, dist);
                                block_values::add_terms(&mut chains, t, &|h| input[0][block_values::weight_at(s, row, group * t + h, dist)], &|w| read_val(stored, w).to_bits());
                                j += block_ops::LANES;
                            }
                        }
                        block_values::join_chains(&chains, t, &down, &mut |h, c, sum| {
                            if lane == 0 { let p = totals + (4 * block_values::total_at(q, h, c)) as u32; ptx::store(p, ptx::load(p) + sum); }
                        });
                    }
                    q += warps;
                }
                _syncthreads();
                base += block_attention::BLOCK;
            }
            _syncthreads();
            let mut o = thread;
            while o < (end - first) * t * block_values::CHANNELS {
                let (q, h, c) = block_values::output_of(t, o);
                let (row, head, channel) = (first + q, group * t + h, slice * block_values::CHANNELS + c);
                let at = block_values::out_index(s, row, head, channel);
                *output.add(at) = match block_values::live(s, row, input[3]) { None => f32::NAN, Some(_) => block_values::word(ptx::load(totals + (4 * block_values::total_at(q, h, c)) as u32), input[2][at]) };
                o += threads;
            }
            return;
        }
        let (tiles, sums) =if KIND == BLOCK_ARMS || KIND == BLOCK_SOFTMAX { (ptx::shared::<{ 8 * block_ops::DOT_TILE }>(), ptx::shared::<{ 4 * block_ops::LANES }>()) } else { (0, 0) };
        // All callers execute the shuffle with a full warp. Lane zero gets the reference
        // lane tree; its five additions have the same operands as block_ops::join.
        let warp_sum=|mut sum:f32| {
            for distance in [16u32,8,4,2,1] {
                let bits:u32;
                core::arch::asm!("shfl.sync.down.b32 {got}, {v}, {d}, 31, 0xffffffff;",got=out(reg32) bits,v=in(reg32) sum.to_bits(),d=in(reg32) distance);
                sum+=f32::from_bits(bits);
            }
            sum
        };
        // SAFETY: the host joins only two checked calls of this exact width and head count.
        // The compact tail address passed the norm's private output check and is disjoint
        // from every convolution operand. One thread owns each word in both outputs.
        // The finish form (reserved[2] CONV_FINISH) comes only from conv_finish_into: it checked the product (one
        // word a channel, read only) and the plane row (width/2 packed words, private, in its plane) against every
        // operand of the command, and the step word as a ring row. A thread writes its own channel of the ring row
        // after its last ring read (no other thread reads that channel) and the first half of a block writes the
        // packed words of its own head only. A pair (reserved[2] CONV_FINISH_PAIR) comes only from the recorder's
        // conv_norm::pair: two such checked commands of one shape, the writes of each disjoint from every operand of
        // the other; each half of the grid runs one of them as above.
        if KIND==BLOCK_ARMS && c.reserved[0]!=0 {
            // The finish form (conv_finish_into checked it): the product of the token is the input of ring row
            // `slot`; the launch writes it there after its reads, and writes the finished words packed into one
            // plane row (two words a packed word, the bits of causal_ops PackBf16) instead of the compact tail.
            // A pair (two finish commands joined by the recorder, the key and then the value of one token): blocks
            // `heads..2*heads` run the second command, whose operands are the payload words from PAIR_AT.
            let payload=p.cast::<u64>();
            let pair=c.reserved[2]==inkling_launch::CONV_FINISH_PAIR;
            let finish=pair || c.reserved[2]==inkling_launch::CONV_FINISH;
            let heads=a.shape.heads as usize;
            let second=pair && block>=heads;
            let block=if second {block-heads} else {block};
            let at=|i:usize|*payload.add(inkling_launch::PAIR_AT+i);
            let (op,param)=if second {let w=at(6);(w as u32,(w>>32) as u32)} else {(c.reserved[0],c.reserved[1])};
            if a.shape.op!=block_ops::Op::ShortConv || threads!=block_ops::NORM_CHUNK ||
                a.shape.dim as usize!=threads || _grid_dim_x() as usize!=heads*(1+usize::from(pair)) ||
                !(op==block_ops::Op::HeadClip as u32 || op==block_ops::Op::RoundedNorm as u32) {core::arch::nvptx::trap();}
            let (input,weight,output,ring,product,plane)=if second {
                ([core::slice::from_raw_parts(at(0) as *const f32,lengths[0]),core::slice::from_raw_parts(at(1) as *const f32,lengths[1]),input[2],input[3]],
                    core::slice::from_raw_parts(at(2) as *const f32,weights),at(3) as *mut f32,at(0) as *mut f32,at(4),at(5))
            } else {(input,weight,output,a.input[0] as *mut f32,*payload.add(inkling_launch::FINISH_PRODUCT_WORD),*payload.add(inkling_launch::FINISH_PLANE_WORD))};
            let idx=block*threads+thread;
            let slot=input[3][0] as usize;
            if finish && (a.shape.cols&block_stream::CONV_ADDEND!=0 || slot>=a.shape.taps as usize || slot as f32!=input[3][0]) {core::arch::nvptx::trap();}
            let current=if finish {(product as *const f32).add(idx).read()} else {0.0};
            let word=if finish {stream_pairs::convolution_current(a.shape,idx,input,weight,slot,current)}
                else {stream_pairs::convolution(a.shape,idx,input,weight)};
            if !finish {*output.add(idx)=word;}
            let clip=op==block_ops::Op::HeadClip as u32;
            let value=if clip {fixed_math::rn(word)} else {word};
            let tile=tiles;
            ptx::store(tile+(4*thread) as u32,value);
            _syncthreads();
            if thread<32 {
                let mut sum=0.0;
                for i in 0..4 {let x=ptx::load(tile+(4*(thread+i*32)) as u32);sum+=x*x;}
                sum=warp_sum(sum);
                if thread==0 {
                    let param=f32::from_bits(param);
                    let factor=if clip {
                        let f=param*(1.0/(block_ops::root(sum)+block_attention::CLIP_EPSILON));
                        if f>1.0 {1.0} else {f}
                    } else {1.0/block_ops::root(sum/threads as f32+param)};
                    ptx::store(tile+(4*threads) as u32,factor);
                }
            }
            _syncthreads();
            let finished=fixed_math::rn(value*ptx::load(tile+(4*threads) as u32));
            if !finish {
                let dst=*payload.add(16) as *mut f32;
                *dst.add(idx)=finished;
                return;
            }
            *output.add(idx)=finished;
            // Word `thread` of the tile is read only before the factor barrier above: each thread now puts its finished
            // word there, and after the barrier below the first half of the block packs the pairs of its head.
            ptx::store(tile+(4*thread) as u32,finished);
            ring.add(slot*heads*threads+idx).write(current);
            _syncthreads();
            if thread<threads/2 {
                let bf16=|x:f32| {let bits=x.to_bits();bits.wrapping_add(0x7fff+((bits>>16)&1))>>16};
                let (low,high)=(ptx::load(tile+(8*thread) as u32),ptx::load(tile+(8*thread+4) as u32));
                (plane as *mut u32).add(block*(threads/2)+thread).write(bf16(low)|(bf16(high)<<16));
            }
            return;
        }
        // HeadClip with heads of at most LANES values, a warp a head (block_ops::clip_warps; taken only with exactly
        // that grid of SUM_THREADS threads). Head is the same for the 32 lanes of a warp, so a whole warp runs the
        // shuffles or none of it does.
        if KIND==BLOCK_ARMS && rows.is_none() && threads==block_ops::SUM_THREADS && block_ops::clip_warps(a.shape)==Some(_grid_dim_x() as usize) {
            let (lane,head)=(thread%block_ops::LANES,block*(threads/block_ops::LANES)+thread/block_ops::LANES);
            if head<a.shape.heads as usize {
                let total=warp_sum(block_attention::clip_lane(a.shape,head,lane,input,weight));
                let bits:u32;
                core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",got=out(reg32) bits,v=in(reg32) total.to_bits());
                let d=a.shape.dim as usize;
                if lane<d {
                    let dst=output.add(head*d+lane);
                    dst.write(0.0);
                    let word=core::slice::from_raw_parts_mut(dst,1);
                    block_attention::clipped(a.shape,head,lane,input,weight,f32::from_bits(bits),word);
                }
            }
            return;
        }
        // WindowScores of one row, a warp of 32 units at a time (the geometry stays one thread a unit in blocks of
        // SUM_THREADS). The warp copies the key rows of its 32 units into its quarter of the tiles, 8 packed words of
        // each row at a time (lanes of 8 words a row, so a read covers 4 rows of 32 bytes), and each lane runs the
        // chain of its own unit from there: the pairs in rising place, as block_attention::unit adds them. The words
        // of a row lie at row*8 + (word ^ (row/4 mod 8)) of the quarter, so neither the copy nor the chain reads
        // meet in a bank. The next 8 words of each row are read before the chain of the current ones runs.
        if KIND==BLOCK_ARMS && rows.is_none() && a.shape.op==block_ops::Op::WindowScores && threads==block_ops::SUM_THREADS
            && a.shape.dim%16==0 && a.shape.dim>0 {
            let s=a.shape;
            let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
            let index=block*threads+thread;
            let Some((n,slot))=block_attention::live_of(input[3],r) else {if index<units {*output.add(index)=f32::NAN;}return;};
            let (lane,warp)=(thread%block_ops::LANES,thread/block_ops::LANES);
            let (head,dist)=if index<units {block_ops::split(index,r)} else {(0,n)};
            let live=dist<n;
            let half=d/2;
            let start=if live {let row=if dist<=slot {slot-dist} else {slot+r-dist};(row*(h/t)+head/t)*half} else {0};
            if ptx::ballot(live)==0 {if index<units {*output.add(index)=0.0;}return;}
            let quarter=tiles+(4*warp*block_ops::LANES*8) as u32;
            let starts:[usize;8]=core::array::from_fn(|q| {
                let got:u32;
                core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, {l}, 31, 0xffffffff;",got=out(reg32) got,v=in(reg32) start as u32,l=in(reg32) (lane/8+4*q) as u32);
                got as usize
            });
            let word=lane%8;
            let mut next:[f32;8]=core::array::from_fn(|q|input[1][starts[q]+word]);
            let query=&input[0][head*d..(head+1)*d];
            let mut sum=0.0f32;let mut first=0;
            while first<half {
                // Row lane/8 + 4q of the warp (its swizzle q), word `word` of this group.
                for (q,v) in next.iter().enumerate() {ptx::store(quarter+(4*((lane/8+4*q)*8+(word^q))) as u32,*v);}
                ptx::warp_sync();
                if first+8<half {for (q,v) in next.iter_mut().enumerate() {*v=input[1][starts[q]+first+8+word];}}
                if live {
                    for m in 0..8 {
                        let bits=ptx::load(quarter+(4*(lane*8+(m^((lane>>2)&7)))) as u32).to_bits();
                        sum+=fixed_math::rn(query[2*(first+m)])*f32::from_bits(bits<<16);
                        sum+=fixed_math::rn(query[2*(first+m)+1])*f32::from_bits(bits&0xffff_0000);
                    }
                }
                ptx::warp_sync();
                first+=8;
            }
            if index<units {
                *output.add(index)=if live {let scaled=fixed_math::rn(fixed_math::rn(sum)*s.bound);fixed_math::rn(scaled+if dist<k {input[2][head*k+dist]} else {0.0})} else {0.0};
            }
            return;
        }
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
                while j < n {
                    ptx::store(base+(4*j) as u32,values[first+j]*weights[first+j]);
                    j+=threads;
                }
                _syncthreads();
                // The cooperative loads round the products once. Keep the serial additions
                // in column order, loading four adjacent products at a time.
                if thread==0 {
                    let mut k=0;
                    while k+4<=n {
                        let v=ptx::load4(base+(4*k) as u32);
                        for bits in v {sum+=f32::from_bits(bits);}
                        k+=4;
                    }
                    while k<n {sum+=ptx::load(base+(4*k) as u32);k+=1;}
                }
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
                    let (mut value, mut dist) = block_attention::softmax_best(n, thread, threads, &|d| scores[d]);
                    // Equal scores keep the earliest distance, including equal signed zeros.
                    for offset in [16u32,8,4,2,1] {
                        let (bits,peer_dist):(u32,u32);
                        core::arch::asm!("shfl.sync.down.b32 {got}, {v}, {d}, 31, 0xffffffff;",got=out(reg32) bits,v=in(reg32) value.to_bits(),d=in(reg32) offset);
                        core::arch::asm!("shfl.sync.down.b32 {got}, {v}, {d}, 31, 0xffffffff;",got=out(reg32) peer_dist,v=in(reg32) dist,d=in(reg32) offset);
                        let peer=f32::from_bits(bits);
                        if peer>value || peer==value && peer_dist<dist {(value,dist)=(peer,peer_dist);}
                    }
                    if thread%32==0 {
                        ptx::store(at(2*(thread/32)),value);
                        ptx::store(at(2*(thread/32)+1),f32::from_bits(dist));
                    }
                    _syncthreads();
                    let top_at = sums + (4 * block_ops::APART) as u32;
                    if thread == 0 { ptx::store(top_at, block_attention::softmax_pick(sink, (0..threads/32).map(|t| (ptx::load(at(2 * t)), ptx::load(at(2 * t + 1)).to_bits())))); }
                    _syncthreads();
                    let top = ptx::load(top_at);
                    // Cache live exp words after the partial sums to reuse them in the final division.
                    const CACHE:usize=3*block_ops::SUM_THREADS;
                    let cached=n<=2*block_ops::DOT_TILE-CACHE;
                    if cached {
                        let mut j=thread;
                        while j<n {ptx::store(at(CACHE+j),block_attention::softmax_exp(scores[j],top));j+=threads;}
                        _syncthreads();
                    }
                    let e=|d:usize| if cached {ptx::load(at(CACHE+d))} else {block_attention::softmax_exp(scores[d],top)};
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
        // The contiguous-distance RelBias grid writes one of the checked units*words per lane (single-row or unshared multi-row).
        if (KIND==BLOCK_ARMS || (KIND==BLOCK_ROWS && mask==0)) && a.shape.op==block_ops::Op::RelBias && words>1 &&
            threads==block_ops::SUM_THREADS && _grid_dim_x() as usize==(units*words).div_ceil(threads) {
            let size=units*words;
            let i=block*threads+thread;
            if i<size {
                let (width,extent)=(a.shape.dim as usize,a.shape.cols as usize);
                let (head,distance)=(i/extent,i%extent);
                *output.add(i)=fast_arms::rel_word(width,extent,head,distance,input[0],weight);
            }
            return;
        }
        // The norm and clip arms whose rows share one sum: a block for each `shared_span` words of a row.
        let shared = if rows.is_none() { block_ops::shared_sum(a.shape).filter(|&(blocks, _)| threads == block_ops::SUM_THREADS && _grid_dim_x() as usize == blocks) } else { None };
        if let Some((_, parts)) = shared {
            let span = block_ops::shared_span(a.shape);
            let (row, first) = block_ops::split(block * span, a.shape.dim as usize);
            let base = sums;
            if a.shape.op == block_ops::Op::RoundedNorm {
                // The host admits full 128-word chunks. Each warp computes the four squares
                // per lane and the same five joins as norm_chunk. Lane zero owns the chunk.
                let width = a.shape.dim as usize;
                let lane = thread % block_ops::LANES;
                let mut chunk = thread / block_ops::LANES;
                while chunk < parts {
                    let start = row * width + chunk * block_ops::NORM_CHUNK + lane;
                    let mut sum = 0.0f32;
                    for group in 0..4 {
                        let value = input[0][start + group * block_ops::LANES];
                        sum += value * value;
                    }
                    sum=warp_sum(sum);
                    if lane == 0 { ptx::store(base + (chunk * 4) as u32, sum); }
                    chunk += threads / block_ops::LANES;
                }
            } else if thread < parts { ptx::store(base + (thread * 4) as u32, block_ops::sum_part(a.shape, row, thread, input, weight)); }
            _syncthreads();
            // Each warp reads the parts once and replays the reference lane tree.
            let lane=thread%block_ops::LANES;
            let mut total=if lane<parts {ptx::load(base+(4*lane) as u32)} else {0.0};
            total=warp_sum(total);
            let bits:u32;
            core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",got=out(reg32) bits,v=in(reg32) total.to_bits());
            let total=f32::from_bits(bits);
            let each = span / block_ops::SUM_THREADS;
            let dst = output.add(block * span + thread * each);
            for i in 0..each { dst.add(i).write(0.0); }
            let words = core::slice::from_raw_parts_mut(dst, each);
            if a.shape.op==block_ops::Op::HeadClip {
                block_attention::clipped(a.shape,row,first+thread*each,input,weight,total,words);
            } else {
                block_stream::normed(a.shape,row,first+thread*each,input,weight,total,words);
            }
        } else {
            let index = block * threads + thread;
            if index < units {
                let dst = output.add(index * words);
                match rows {
                    None if a.shape.op==block_ops::Op::ShortConv => dst.write(stream_pairs::convolution(a.shape,index,input,weight)),
                    None if a.shape.op==block_ops::Op::GateBlend => {
                        let forget=stream_pairs::gate_tanh(input[0][index]);
                        let gain=stream_pairs::gate_tanh(input[0][a.shape.dim as usize+index]);
                        dst.write((1.0-forget)*input[1][index]+gain*input[2][index]);
                    }
                    Some(n) => {
                        for i in 0..words { dst.add(i).write(0.0); }
                        block_rows::unit(a.shape, n, mask, index, input, weight, core::slice::from_raw_parts_mut(dst, words));
                    }
                    None => {
                        for i in 0..words { dst.add(i).write(0.0); }
                        block_ops::unit(a.shape, index, input, weight, core::slice::from_raw_parts_mut(dst, words));
                    }
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
        plain_packed::tile::<2>(ptx::shared::<{plain_packed::BLOCK_BYTES}>(),*words as *const u8, *words.add(1) as *const f32, *words.add(2) as *mut f32, *words.add(3) as u32, *words.add(4) as u32,core::ptr::null(),None);
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
        plain_packed::tile::<2>(ptx::shared::<{plain_packed::BLOCK_BYTES}>(),unit, input.add(e * input_words), output.add(e * rows as usize), rows, word,core::ptr::null(),None);
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
    // packed_product::check_picked passed for exactly these words: 1 to PICKS_MAX picks; `ids` is exactly `picks` words,
    // word aligned; the stack is exactly `experts` units of `unit_bytes` each (layout(rows, cols, wide).bytes, a
    // multiple of 16) and 16-byte aligned, so unit `n` for any `n < experts` is a unit the single check
    // packed_product::check passed with the input at `input + e * input_words` words and the output at
    // `output + e * rows` words; the whole output meets no byte of the stack, of `ids` or of the input, so no block
    // writes a word that another block reads. A block row at or above the picks traps; an id word that is not a whole
    // number at least 0 and below `experts` traps (NaN fails the first compare). For the paired gate, words
    // 10..14 hold caps initialized by the host. Each pick owns rows/2 compact words in the checked output;
    // its even matrix row alone writes pick*(rows/2)+row/2, including when lanes share a tile across picks.
    // For strided routes the host checks the last indexed ID and the extent of each repeated input row.
    // So the call of plain_packed::tile below
    // is a call with words the single check passed, and its own SAFETY note holds for it.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != PLAIN_PACKED_PICKED { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (input, output, rows, word, picks, input_words) = (*words as *const f32, *words.add(1) as *mut f32, *words.add(2) as u32, *words.add(3) as u32, *words.add(4) as usize, *words.add(5) as usize);
        let (stack, ids, experts, unit_bytes) = (*words.add(6) as *const u8, *words.add(7) as *const f32, *words.add(8) as u32, *words.add(9) as usize);
        let e = core::arch::nvptx::_block_idx_y() as usize;
        if picks == 0 || picks > packed_product::PICKS_MAX || e >= picks { core::arch::nvptx::trap(); }
        let repeat = { let r = *words.add(14) as usize; if r == 0 { 1 } else { r } };
        let caps=if *words.add(10)==0 {core::ptr::null()} else {
            if word & packed_product::PAIRS == 0 { core::arch::nvptx::trap(); }
            words.add(10)
        };
        let smem=ptx::shared::<{plain_packed::PICKED_BLOCK_BYTES}>();
        let one_row = *words.add(packed_product::PICKED_ONE_ROW_WORD) != 0;
        // Edge 3A (`FastRouterPick -> GateUp`): when not `Down` (`caps.is_null() && repeat == 1 && input_words != 0`),
        // `ids`/`compact` may still be written by `FastRouterPick`. For single-row `GateUp` (`input_words == 0`,
        // `cols <= 5120`), `X` is already ready before `FastRouterPick` and fits inside `PANEL_X_BYTES` (`10,240 B`),
        // so pre-stage `X` into `smem` BEFORE `griddepcontrol.wait`.
        let mut preloaded_x = false;
        let is_down_pdl = caps.is_null() && repeat == 1 && input_words != 0;
        if !is_down_pdl {
            let cols = (word & packed_product::COLS) as usize;
            if one_row && input_words == 0 && !caps.is_null()
                && rows % 64 == 0 && cols > 0 && cols % 64 == 0 && cols <= 5120
                && (input as usize) & 15 == 0
                && core::arch::nvptx::_block_dim_x() as usize == packed_product::PICKED_THREADS
            {
                let tid = core::arch::nvptx::_thread_idx_x() as u32;
                let pairs_in_slab = (cols / 64) as u32 * 16;
                let mut pair = tid;
                while pair < pairs_in_slab {
                    let c_i = pair >> 4;
                    let ks = pair & 15;
                    let xp = input.add((c_i * 64 + (ks >> 2) * 16 + (ks & 3) * 2) as usize);
                    let lo = tc_ptx::ld_global_nc_v2_u32(xp.cast());
                    let hi = tc_ptx::ld_global_nc_v2_u32(xp.add(8).cast());
                    let dst = smem + c_i * 128 + ((ks >> 2) << 5) + ((ks & 3) << 3);
                    tc_ptx::store_u32(dst, tc_ptx::cvt_bf16x2(f32::from_bits(lo[0]), f32::from_bits(lo[1])));
                    tc_ptx::store_u32(dst + 4, tc_ptx::cvt_bf16x2(f32::from_bits(hi[0]), f32::from_bits(hi[1])));
                    pair += packed_product::PICKED_THREADS as u32;
                }
                preloaded_x = true;
            }
            core::arch::asm!("griddepcontrol.wait;", options(nostack));
        }
        // SAFETY: word PICKED_ONE_ROW_WORD + 2 is zero (standard route scan) or a host-checked
        // 16-byte aligned compacted dispatch table of `picks` 16-byte entries built by `fast_router_pick`.
        let compact = *words.add(packed_product::PICKED_ONE_ROW_WORD + 2) as *const u32;
        let (id_u32, indices, count, ordinal) = if !compact.is_null() {
            if (compact as usize) & 15 != 0 { core::arch::nvptx::trap(); }
            let ent = tc_ptx::ld_global_nc_v4_u32(compact.add(e * 4));
            let indices = (ent[0] as u64) | ((ent[1] as u64) << 32);
            let id_u32 = ent[2];
            let count = (ent[3] & 0xffff) as usize;
            let ordinal = (ent[3] >> 16) as usize;
            if id_u32 >= experts || count == 0 || count > 8 || ordinal >= count { core::arch::nvptx::trap(); }
            if (core::arch::nvptx::_block_idx_x() as usize * count + ordinal) * 64 >= rows as usize { return; }
            (id_u32, indices, count, ordinal)
        } else {
            let ppr = *words.add(15) as usize;
            let id_stride = *words.add(16) as usize;
            let id_at = |j: usize| -> f32 {
                if ppr == 0 { *ids.add(j) } else { *ids.add((j / ppr) * id_stride + (j % ppr)) }
            };
            let id = id_at(e);
            if !(id >= 0.0 && id < experts as f32) || (id as u32) as f32 != id || id as u32 >= experts { core::arch::nvptx::trap(); }
            let (mut indices,mut count,mut ordinal)=(0u64,0,0);
            if picks >= 32 {
                // All lanes vote on consecutive route words. Consume the set bits in pick
                // order, so each group of eight and its block ordinals match the scalar scan.
                let lane=core::arch::nvptx::_thread_idx_x() as usize & 31;
                'route: for first in (0..picks).step_by(32) {
                    let at=first+lane;
                    let mut matches=ptx::ballot(at<picks && id_at(at)==id);
                    while matches!=0 {
                        let j=first+matches.trailing_zeros() as usize;
                        matches&=matches-1;
                        if j==e {ordinal=count;}
                        indices|=(j as u64)<<(8*count);
                        count+=1;
                        if count==8 {
                            if j>=e {break 'route;}
                            indices=0;
                            count=0;
                        }
                    }
                }
            } else {
                for j in 0..picks {
                    if id_at(j)==id {
                        if j==e {ordinal=count;}
                        indices|=(j as u64)<<(8*count);
                        count+=1;
                        if count==8 {
                            if j>=e {break;}
                            indices=0;
                            count=0;
                        }
                    }
                }
            }
            (id as u32, indices, count, ordinal)
        };
        let unit=stack.add(id_u32 as usize*unit_bytes);
        // SAFETY: word PICKED_ONE_ROW_WORD + 1 is zero or the checked disjoint packed stack of the same
        // length as the raw stack. Its expert stride and index use the validated unit.
        let panels = *words.add(packed_product::PICKED_ONE_ROW_WORD + 1) as *const u8;
        let panels = if panels.is_null() { panels } else { panels.add(id_u32 as usize*unit_bytes) };
        if preloaded_x && count > 1 { core::arch::nvptx::_syncthreads(); }
        let group=plain_packed::SharedPicks {indices,count,stride:input_words,repeat,ordinal,one_row,panels,x_preloaded: preloaded_x && count == 1};
        if word&packed_product::WIDE==0 {
            plain_packed::tile::<0>(smem,unit,input,output,rows,word,caps,Some(group));
        } else {
            plain_packed::tile::<1>(smem,unit,input,output,rows,word,caps,Some(group));
        }
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
/// SAFETY: launched only by the host with a table of commands of kind BLOCK_VALUES (`run`, whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_block_values(table: *const Command, index: u64) { run::<BLOCK_VALUES>(table, index) }
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
/// The packed rows tile (`plain_packed_rows.rs`): the expert runs of a visit read from their packed units; its body
/// reads and checks its own recorded command.
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_PACKED_ROWS (`plain_packed_rows::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_packed_rows(table: *const Command, index: u64) { plain_packed_rows::replay(table, index) }
/// The embedding rows of a step (`embed_rows.rs`); its body reads and checks its own recorded command.
/// SAFETY: launched only by the host with a table of commands of kind EMBED_ROWS (`embed_rows::replay`, whose contract
/// holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_embed_rows(table: *const Command, index: u64) { embed_rows::replay(table, index) }
/// The small-batch BF16 product (`plain_staged_few.rs`, 2 to 16 input rows); its body reads and checks its own
/// recorded command.
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_STAGED_FEW (`plain_staged_few::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_staged_few(table: *const Command, index: u64) { plain_staged_few::replay(table, index) }
/// The one-row BF16 product (`plain_staged_one.rs`, one input row, with the lossless panel form); its body reads and
/// checks its own command.
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_STAGED_ONE (`plain_staged_one::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_staged_one(table: *const Command, index: u64) { plain_staged_one::replay(table, index) }
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
/// SM121a BF16 tensor-core MMA multi-row product (`plain_tc_rows.rs`).
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_TC_ROWS (`plain_tc_rows::replay`, whose
/// contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_tc_rows(table: *const Command, index: u64) { plain_tc_rows::replay(table, index) }
/// Deterministic GPU MoE router grouping (`moe_group_route.rs`).
/// SAFETY: launched only by the host with a table of commands of kind MOE_GROUP_ROUTE (`moe_group_route::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_moe_group_route(table: *const Command, index: u64) { moe_group_route::replay(table, index) }
/// SM121a BF16 tensor-core fused causal/sliding-window attention (`flash_window.rs`).
/// SAFETY: launched only by the host with a table of commands of kind FLASH_WINDOW_ATTN (`flash_window::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_flash_window(table: *const Command, index: u64) { flash_window::replay(table, index) }
/// GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention, `<= 16` queries/KV-head (`flash_window_split.rs`).
/// SAFETY: launched only by `Backend::flash_window_split_into` with kind FLASH_WINDOW_SPLIT16 (`flash_window_split::replay_split::<1, 26496>`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_flash_window_split16(table: *const Command, index: u64) { flash_window_split::replay_split::<1, 26496>(table, index) }
/// GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention, `<= 32` queries/KV-head (`flash_window_split.rs`).
/// SAFETY: launched only by `Backend::flash_window_split_into` with kind FLASH_WINDOW_SPLIT32 (`flash_window_split::replay_split::<2, 33536>`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_flash_window_split32(table: *const Command, index: u64) { flash_window_split::replay_split::<2, 33536>(table, index) }
/// Deterministic Split-KV reduction with exact sink and output gate (`flash_window_split.rs`).
/// SAFETY: launched only by `Backend::flash_window_split_into` with kind FLASH_WINDOW_REDUCE (`flash_window_split::replay_reduce`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_flash_window_reduce(table: *const Command, index: u64) { flash_window_split::replay_reduce(table, index) }
/// SM121a BF16 tensor-core MMA packed MoE product (`plain_packed_tc_rows.rs`).
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_PACKED_TC_ROWS (`plain_packed_tc_rows::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_packed_tc_rows(table: *const Command, index: u64) { plain_packed_tc_rows::replay(table, index) }
/// Bit-exact W4A16/W8A16 BF16 tensor-core prefill MoE GEMM, packer, and fold (`prefill_packed_tc.rs`).
/// SAFETY: launched only by the host with a table of commands of kind PREFILL_PACKED_TC (`prefill_packed_tc::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_prefill_packed_tc(table: *const Command, index: u64) { prefill_packed_tc::replay(table, index) }
/// High-occupancy 2-stage prefill causal/sliding-window attention (`prefill_flash_window.rs`).
/// SAFETY: launched only by the host with a table of commands of kind PREFILL_FLASH_WINDOW (`prefill_flash_window::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_prefill_flash_window(table: *const Command, index: u64) { prefill_flash_window::replay(table, index) }
/// Fused prefill lazy message-meta and GateDown+GateOuter+GateBlend+RoundedNorm (`prefill_gate.rs`).
/// SAFETY: launched only by the host with a table of commands of kind PREFILL_GATE_FUSE (`prefill_gate_kern::replay`,
/// whose contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_prefill_gate_fuse(table: *const Command, index: u64) { prefill_gate_kern::replay(table, index) }
/// SM121a BF16 tensor-core MMA big-tile dense product for prefill (`plain_tc_big.rs`).
/// SAFETY: launched only by the host with a table of commands of kind PLAIN_TC_BIG (`plain_tc_big::replay`, whose
/// contract holds).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_plain_tc_big(table: *const Command, index: u64) { plain_tc_big::replay(table, index) }
/// Zero-stack fast Inkling mix blend (`fast_mix.rs`).
/// SAFETY: launched only by the host with a table of commands of kind FAST_MIX_BLEND (`fast_mix::replay_mix_blend`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fast_mix_blend(table: *const Command, index: u64) { fast_mix::replay_mix_blend(table, index) }
/// Zero-stack fast Inkling list-form row dots (`fast_mix.rs`).
/// SAFETY: launched only by the host with a table of commands of kind FAST_ROW_DOTS (`fast_mix::replay_row_dots`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fast_row_dots(table: *const Command, index: u64) { fast_mix::replay_row_dots(table, index) }
/// Zero-stack bit-exact Inkling norm (`fast_norm.rs`).
/// SAFETY: launched only by the host with a table of commands of kind FAST_NORM (`fast_norm::replay`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fast_norm(table: *const Command, index: u64) { fast_norm::replay(table, index) }
/// Zero-stack bit-exact Inkling capped gate (`fast_elem.rs`).
/// SAFETY: launched only by the host with a table of commands of kind FAST_CAPPED_GATE (`fast_elem::replay_capped_gate`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fast_capped_gate(table: *const Command, index: u64) { fast_elem::replay_capped_gate(table, index) }
/// Zero-stack warp-parallel bit-exact Inkling router pick (`fast_elem.rs`).
/// SAFETY: launched only by the host with a table of commands of kind FAST_ROUTER_PICK (`fast_elem::replay_router_pick`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fast_router_pick(table: *const Command, index: u64) { fast_elem::replay_router_pick(table, index) }
/// Zero-stack bit-exact Inkling stream ops (`fast_elem.rs`: window ShortConv, RouteFold, GateBlend).
/// SAFETY: launched only by the host with a table of commands of kind FAST_STREAM_OPS (`fast_elem::replay_stream_ops`).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn resident_launch_fast_stream_ops(table: *const Command, index: u64) { fast_elem::replay_stream_ops(table, index) }

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    // Any panic is a programming defect. Trap, so the host sees a failed launch.
    // SAFETY: `trap` takes no operand and touches no memory; it ends the thread's kernel, which is what a panic means here.
    unsafe { core::arch::nvptx::trap() }
}
