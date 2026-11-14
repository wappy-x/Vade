//! Shape rules that pick the schedule of the plain FP32 row kernels. Pure functions, shared by the
//! kernels and the host tests, so one predicate decides both. Every schedule computes each output as
//! one ascending chain of separately rounded products and adds from 0.0; the rules only choose how
//! the operands travel. A shape outside a rule takes the next simpler schedule, down to the original
//! one-thread-per-row loop.

/// Fewest columns for a deeper schedule: one batch of the register schedule is 8 vectors of 4 columns.
pub const DEEP_MIN_COLS: usize = 32;
/// The proof envelope of the deeper schedules: the device sweeps (kernel-bench --plain-probe) cover every width 1..=600, every
/// multiple of four from 32 to 8192 (mode `cols`: three row counts, five patterns each), the listed widths up to 8192, rows
/// from 1 to 1,048,576 at width 32 and the corners of the element cap below. Larger matrices, or wider ones, keep the original
/// one-thread-per-row loop.
pub const COLS_MAX: usize = 8192;
pub const ELEMENTS_MAX: usize = 1 << 25;
/// Threads per block the warp tile takes: four warps of 32 rows, or one warp alone. The lone warp owns the whole block
/// of shared memory, so its pipeline is four times deeper; the host uses it for matrices with few rows.
pub const TILE_THREADS: usize = 128;
pub const NARROW_THREADS: usize = 32;
/// Most rows that get one-warp blocks: one warp per 32 rows spreads a small matrix over more SMs.
pub const NARROW_MAX_ROWS: usize = 512;
/// Fold schedule: blocks of one folding warp and four producer warps own FOLD_ROWS rows each. The host uses it for matrices with
/// few rows and long rows, where one chain per row is the floor and the one-warp tile ran copies and chain one after the other.
pub const FOLD_THREADS: usize = 160;
pub const FOLD_ROWS: usize = 4;
/// Fewest columns that take the fold schedule (shorter rows keep the one-warp tile).
pub const FOLD_MIN_COLS: usize = 1024;
/// Launch geometry (blocks, threads) of `project_f32_ordered` for a matrix of `rows` x `cols`. Every schedule of the kernel is
/// correct under any block size; the geometry only decides speed.
pub fn plain_geometry(rows: usize, cols: usize) -> (u32, u32) {
    if rows <= NARROW_MAX_ROWS && cols >= FOLD_MIN_COLS { (rows.div_ceil(FOLD_ROWS) as u32, FOLD_THREADS as u32) }
    else if rows <= NARROW_MAX_ROWS { (rows.div_ceil(32) as u32, NARROW_THREADS as u32) } else { (rows.div_ceil(128) as u32, TILE_THREADS as u32) }
}

/// The original fast path: whole 16-byte vectors on both operands.
pub const fn vectors(cols: usize, weights: usize, input: usize) -> bool {
    cols % 4 == 0 && (weights | input) & 15 == 0
}
/// Inside the swept envelope: 32..=COLS_MAX columns and at most ELEMENTS_MAX weights.
pub const fn in_envelope(rows: usize, cols: usize) -> bool {
    cols >= DEEP_MIN_COLS && cols <= COLS_MAX && rows >= 1 && rows <= ELEMENTS_MAX / cols
}
/// Register schedule: loads run ahead of the chain, one thread per row.
pub const fn deep_admitted(rows: usize, cols: usize, weights: usize, input: usize) -> bool {
    in_envelope(rows, cols) && vectors(cols, weights, input)
}
/// ROW STAGE (`flash_resident_projection_bf16_rows`): a plain matrix by its shape alone, so every rank of a split matrix makes
/// the same choice. The entry also needs 16-byte aligned weights and inputs (`deep_admitted`); the host makes them so.
pub const fn rows_admitted(rows: usize, cols: usize) -> bool {
    in_envelope(rows, cols) && cols % 32 == 0
}
/// Warp tile: 32 rows per warp staged through shared memory, one thread per row.
pub const fn tile_admitted(rows: usize, cols: usize, weights: usize, input: usize, threads: usize) -> bool {
    (threads == TILE_THREADS || threads == NARROW_THREADS) && deep_admitted(rows, cols, weights, input)
}

/// Fold schedule (`plain_warp::rows_fold`): a FOLD_THREADS-thread block inside the register schedule's envelope.
pub const fn fold_admitted(rows: usize, cols: usize, weights: usize, input: usize, threads: usize) -> bool {
    threads == FOLD_THREADS && deep_admitted(rows, cols, weights, input)
}

/// Launch geometry of `project_bf16_ordered`: one-warp blocks to NARROW_MAX_ROWS rows, four-warp blocks above (the rule the BF16 entry was swept in).
pub const fn bf16_geometry(rows: usize) -> (u32, u32) {
    if rows <= NARROW_MAX_ROWS { (rows.div_ceil(32) as u32, NARROW_THREADS as u32) } else { (rows.div_ceil(128) as u32, TILE_THREADS as u32) }
}
/// BF16 weights (`project_bf16_ordered`): admitted ONLY for the matrix shapes the plain single-token step reads on its own, in the launch geometry
/// `bf16_geometry` gives them (the one-warp / four-warp rule; the FP32 entry's fold and short-row rules do not apply to this entry), with 16-byte aligned weights and input. Rank-local (two-way row split) and logical (unsplit) forms:
/// f_a, g_a, wk, compress gate 64 / 128 x 4096; b_proj 32 / 64 x 4096 (also 64 x 4096 rank-local f_a); weights_proj 16 / 32 x 4096;
/// f_b, g_b 4096 / 8192 x 128; indexer wq_b 2048 / 4096 x 1536; kv_b 16384 / 32768 x 512. Any other shape takes the scalar chain of the same entry. The router gate
/// (144 / 288 x 4096) is NOT listed (ruling of 2026-11-06, router fold): BF16 gives it nothing (35 us a call in either dtype, chain-bound),
/// and it reads FP32 through the fold schedule (`plain_geometry`, `fold_admitted`).
/// The device sweep (kernel-bench --plain-probe ... bf16) walks exactly this list (`BF16_STEP_SHAPES` in kernel_plainp.rs; a host test compares the two) from poisoned
/// outputs, with the neighbours of each shape outside it.
pub const BF16_SHAPES: [(usize, usize); 10] = [(16, 4096), (32, 4096), (64, 4096), (128, 4096),
    (4096, 128), (8192, 128), (2048, 1536), (4096, 1536), (16384, 512), (32768, 512)];
pub const fn bf16_listed(rows: usize, cols: usize) -> bool {
    let mut i = 0;
    while i < BF16_SHAPES.len() { if BF16_SHAPES[i].0 == rows && BF16_SHAPES[i].1 == cols { return true; } i += 1; }
    false
}
pub const fn bf16_admitted(rows: usize, cols: usize, weights: usize, input: usize, threads: usize) -> bool {
    threads == bf16_geometry(rows).1 as usize && bf16_listed(rows, cols) && (weights | input) & 15 == 0
}

/// Mixed projection batch: which descriptor and which block of it a grid block runs, with the blocks of
/// plain descriptors first (each group in descriptor order). `ends` are the prefix block counts of the
/// batch, `plain` the bit mask of plain descriptors, `count` at most 9. A bijection on `0..ends[count]`.
pub const fn batch_block(ends: [u32; 10], plain: u32, count: usize, block: u32) -> (usize, u32) {
    let mut plain_blocks = 0u32;
    let mut i = 0;
    while i < 9 {
        if i < count && plain & (1 << i) != 0 { plain_blocks += ends[i + 1] - ends[i]; }
        i += 1;
    }
    let plain_first = block < plain_blocks;
    let mut rest = if plain_first { block } else { block - plain_blocks };
    let mut i = 0;
    while i < 9 {
        if i < count && (plain & (1 << i) != 0) == plain_first {
            let n = ends[i + 1] - ends[i];
            if rest < n { return (i, rest); }
            rest -= n;
        }
        i += 1;
    }
    (0, block)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_row_stage_takes_every_bf16_copied_shape_and_decides_by_shape_alone() {
        for (rows, cols) in BF16_SHAPES { assert!(rows_admitted(rows, cols), "{rows} {cols}"); }
        for (rows, cols) in [(64, 4096), (4096, 128), (1, 32), (4096, 8192)] { assert!(rows_admitted(rows, cols)); assert!(deep_admitted(rows, cols, 16, 32)); }
        for (rows, cols) in [(0, 4096), (64, 0), (64, 48), (64, 8224), (8192, 8192)] { assert!(!rows_admitted(rows, cols), "{rows} {cols}"); }
    }
    #[test]
    fn short_misaligned_ragged_or_oversized_shapes_keep_the_original_loop() {
        for (cols, w, x) in [(0, 16, 16), (4, 16, 16), (28, 16, 16), (31, 16, 16), (36, 20, 16), (64, 16, 20), (66, 16, 16), (4098, 16, 16), (4097, 16, 16)] {
            assert!(!deep_admitted(4, cols, w, x) || (cols >= 32 && cols % 4 == 0 && (w | x) & 15 == 0), "{cols} {w} {x}");
        }
        assert!(!deep_admitted(4, 28, 16, 16) && !deep_admitted(4, 31, 16, 16) && !deep_admitted(4, 0, 16, 16));
        assert!(!deep_admitted(4, 36, 20, 16) && !deep_admitted(4, 64, 16, 20) && !deep_admitted(4, 66, 16, 16) && !deep_admitted(4, 4097, 16, 16));
        assert!(vectors(28, 16, 16) && !vectors(31, 16, 16));
    }
    #[test]
    fn the_wide_counterexample_and_everything_beyond_the_sweep_takes_the_old_path_in_both_entries() {
        // One 1 x 16384 matrix (the review's example): standalone entry and the mixed batch use the same rule.
        for threads in [32usize, 128] {
            assert!(!tile_admitted(1, 16384, 256, 512, threads), "standalone tile, {threads} threads");
        }
        assert!(!deep_admitted(1, 16384, 256, 512), "mixed batch register schedule");
        assert!(!deep_admitted(1, 8196, 256, 512) && !deep_admitted(3, 32768, 256, 512));
        // The widest swept width, and the element cap corners, are inside.
        assert!(deep_admitted(1, 8192, 256, 512) && tile_admitted(1, 8192, 256, 512, 128));
        assert!(deep_admitted(4096, 8192, 256, 512) && !deep_admitted(4097, 8192, 256, 512));
        assert!(deep_admitted(65536, 512, 256, 512) && !deep_admitted(65537, 512, 256, 512));
        assert!(deep_admitted(1 << 20, 32, 256, 512) && !deep_admitted((1 << 20) + 1, 32, 256, 512));
        assert!(!deep_admitted(0, 64, 256, 512));
    }
    #[test]
    fn real_decode_shapes_take_the_tile_only_with_the_launch_the_kernel_expects() {
        for (rows, cols) in [(144usize, 4096usize), (16384, 512), (2048, 2048), (4096, 128), (384, 4096), (1, 32)] {
            assert!(deep_admitted(rows, cols, 256, 512));
            assert!(tile_admitted(rows, cols, 256, 512, 128) && tile_admitted(rows, cols, 256, 512, 32));
            assert!(!tile_admitted(rows, cols, 256, 512, 256) && !tile_admitted(rows, cols, 256, 512, 64) && !tile_admitted(rows, cols, 256, 512, 1));
        }
        assert!(!tile_admitted(144, 4096, 4, 512, 128) && !tile_admitted(144, 4096, 256, 8, 128));
        assert!(!tile_admitted(144, 28, 256, 512, 128));
    }
    #[test]
    fn plain_blocks_run_first_and_every_block_maps_once() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for round in 0..4000 {
            let count = 1 + (next() % 9) as usize;
            let mut ends = [0u32; 10];
            let mut plain = 0u32;
            let sizes: Vec<u32> = (0..count).map(|_| 1 + (next() % 40) as u32).collect();
            for i in 0..count { ends[i + 1] = ends[i] + sizes[i]; if next() % 2 == 0 { plain |= 1 << i; } }
            let total = ends[count];
            let plain_total: u32 = (0..count).filter(|i| plain & (1 << i) != 0).map(|i| sizes[i]).sum();
            let mut seen = vec![0u32; total as usize];
            for block in 0..total {
                let (job, local) = batch_block(ends, plain, count, block);
                assert!(job < count && local < sizes[job], "round {round} block {block}");
                assert_eq!(block < plain_total, plain & (1 << job) != 0, "plain descriptors first, round {round} block {block}");
                seen[(ends[job] + local) as usize] += 1;
            }
            assert!(seen.iter().all(|&n| n == 1), "bijection, round {round}");
            // Inside a group the descriptors keep their order.
            let order: Vec<usize> = (0..total).map(|b| batch_block(ends, plain, count, b).0).collect();
            let (first, second) = order.split_at(plain_total as usize);
            assert!(first.windows(2).all(|w| w[0] <= w[1]) && second.windows(2).all(|w| w[0] <= w[1]));
        }
        assert_eq!(batch_block([0, 3, 4, 7, 0, 0, 0, 0, 0, 0], 0b010, 3, 0), (1, 0));
        assert_eq!(batch_block([0, 3, 4, 7, 0, 0, 0, 0, 0, 0], 0b010, 3, 1), (0, 0));
    }
    #[test]
    fn bf16_rows_are_admitted_only_for_the_listed_step_shapes_in_their_own_geometry() {
        // Every listed shape is admitted in the geometry bf16_geometry gives it, and in no other.
        for &(rows, cols) in &BF16_SHAPES {
            let (_, threads) = bf16_geometry(rows);
            assert!(bf16_admitted(rows, cols, 256, 512, threads as usize), "{rows}x{cols}");
            for other in [0usize, 1, 32, 64, 96, 128, 256] { if other as u32 != threads { assert!(!bf16_admitted(rows, cols, 256, 512, other), "{rows}x{cols} threads {other}"); } }
            assert!(!bf16_admitted(rows, cols, 8, 512, threads as usize) && !bf16_admitted(rows, cols, 256, 4, threads as usize), "alignment {rows}x{cols}");
        }
        // Nothing else: every row count to 33,000 at each listed width, every width to 8,200 at each listed row count.
        let widths: Vec<usize> = { let mut w: Vec<usize> = BF16_SHAPES.iter().map(|s| s.1).collect(); w.sort(); w.dedup(); w };
        let heights: Vec<usize> = { let mut h: Vec<usize> = BF16_SHAPES.iter().map(|s| s.0).collect(); h.sort(); h.dedup(); h };
        for &cols in &widths { for rows in 0..=33_000usize { let want = BF16_SHAPES.contains(&(rows, cols)); assert_eq!(bf16_listed(rows, cols), want, "{rows}x{cols}"); } }
        for &rows in &heights { for cols in 0..=8_200usize { let want = BF16_SHAPES.contains(&(rows, cols)); assert_eq!(bf16_listed(rows, cols), want, "{rows}x{cols}"); } }
        // The shapes the earlier envelope admitted and the review named are refused now.
        for (rows, cols) in [(1_048_576usize, 32usize), (65_536, 512), (4_096, 8_192), (512, 8_192), (513, 8_192), (4_096, 1_024), (144, 4_088)] {
            assert!(!bf16_listed(rows, cols), "{rows}x{cols}");
        }
    }
    #[test]
    fn the_four_bf16_standalone_matrices_have_a_listed_shape_at_one_and_two_ranks_and_none_at_four() {
        // Logical shapes (rows, cols) of the matrices the measuring form copies: kv_b, indexer wq_b, f_b and g_b (the router gate is not one, below).
        for (rows, cols) in [(32768usize, 512usize), (4096, 1536), (8192, 128)] {
            assert!(bf16_listed(rows, cols), "{rows}x{cols} at one rank");
            assert!(bf16_listed(rows / 2, cols), "{} x {cols} at two ranks", rows / 2);
            assert!(!bf16_listed(rows / 4, cols), "{} x {cols} at four ranks must keep the FP32 reads", rows / 4);
        }
    }
    #[test]
    fn the_router_gate_reads_fp32_through_the_fold_at_one_two_and_four_ranks() {
        // Ruling of 2026-11-06 (router fold on the BF16 stack): the router gate (288 x 4096 logical) is on the BF16 list at no
        // rank count, so the loader keeps no BF16 copy of it (resident_weights::bf16_copy_wanted is bf16_listed) and its projection is the FP32
        // entry, whose geometry is the fold for every row split; the fold admits it, and the BF16 entry refuses it in any geometry.
        for parts in [1usize, 2, 4] {
            let rows = 288 / parts;
            assert!(!bf16_listed(rows, 4096), "router {rows} x 4096 must not be on the BF16 list");
            let (blocks, threads) = plain_geometry(rows, 4096);
            assert_eq!((blocks as usize, threads as usize), (rows.div_ceil(FOLD_ROWS), FOLD_THREADS), "router {rows} x 4096 geometry");
            assert!(fold_admitted(rows, 4096, 256, 512, threads as usize), "router {rows} x 4096 fold");
            for t in [32usize, 128, 160] { assert!(!bf16_admitted(rows, 4096, 256, 512, t), "router {rows} x 4096 BF16 at {t} threads"); }
        }
        // The BF16 shapes that stay listed keep the geometry they were swept in on the BF16 entry, whatever the FP32 entry would take.
        for &(rows, _) in &BF16_SHAPES {
            let swept = if rows <= NARROW_MAX_ROWS { (rows.div_ceil(32) as u32, 32) } else { (rows.div_ceil(128) as u32, 128) };
            assert_eq!(bf16_geometry(rows), swept, "{rows} rows");
        }
        assert_eq!((plain_geometry(64, 4096), bf16_geometry(64)), ((16, 160), (2, 32)));
    }
    #[test]
    fn the_bf16_sweep_walks_exactly_the_admitted_list() {
        // The bench cannot link this crate, so it carries the list as one literal; the two must be equal, in order.
        let bench = include_str!("../../../tools/bench/src/kernel_plainp.rs");
        let at = bench.find("const BF16_STEP_SHAPES: [(usize, usize); 10] = [").expect("the bench list") + "const BF16_STEP_SHAPES: [(usize, usize); 10] = [".len();
        let literal = &bench[at..at + bench[at..].find("];").expect("end of list")];
        let numbers: Vec<usize> = literal.split(|c: char| !c.is_ascii_digit()).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
        let pairs: Vec<(usize, usize)> = numbers.chunks(2).map(|p| (p[0], p[1])).collect();
        assert_eq!(pairs, BF16_SHAPES.to_vec());
    }
    #[test]
    fn small_matrices_get_one_warp_blocks_and_big_ones_four() {
        assert_eq!(plain_geometry(144, 512), (5, 32));
        assert_eq!(plain_geometry(144, 1020), (5, 32));
        assert_eq!(plain_geometry(1, 128), (1, 32));
        assert_eq!(plain_geometry(512, 512), (16, 32));
        assert_eq!(plain_geometry(513, 4096), (5, 128));
        assert_eq!(plain_geometry(16384, 512), (128, 128));
        assert_eq!(plain_geometry(2048, 1536), (16, 128));
        assert_eq!(plain_geometry(4096, 128), (32, 128));
        for rows in [1usize, 31, 32, 33, 127, 128, 129, 144, 511, 512, 513, 2048, 16384] {
            for cols in [32usize, 128, 512, 1020, 1024, 1536, 4096, 8192] {
                let (blocks, threads) = plain_geometry(rows, cols);
                assert!(blocks as usize * threads as usize >= rows, "{rows} {cols}");
                // A fold block owns FOLD_ROWS rows, whatever its thread count; the other schedules one row a thread.
                if threads as usize == FOLD_THREADS { assert!(blocks as usize * FOLD_ROWS >= rows && (blocks as usize - 1) * FOLD_ROWS < rows, "{rows} {cols}"); }
            }
        }
    }
    #[test]
    fn the_router_gate_takes_the_fold_and_nothing_else_does() {
        // The step's FP32 standalone shapes: only the 144 x 4096 router gate (and its unsplit 288 x 4096) has few rows and long rows.
        assert_eq!(plain_geometry(144, 4096), (36, 160));
        assert_eq!(plain_geometry(288, 4096), (72, 160));
        assert_eq!(plain_geometry(145, 4096), (37, 160));
        assert_eq!(plain_geometry(512, 1024), (128, 160));
        assert!(fold_admitted(144, 4096, 256, 512, FOLD_THREADS) && fold_admitted(1, 8192, 256, 512, FOLD_THREADS));
        assert!(!fold_admitted(144, 4096, 256, 512, 128) && !fold_admitted(144, 4096, 256, 512, 32));
        assert!(!fold_admitted(144, 4096, 4, 512, FOLD_THREADS) && !fold_admitted(144, 4096, 256, 8, FOLD_THREADS));
        assert!(!fold_admitted(144, 4098, 256, 512, FOLD_THREADS) && !fold_admitted(1, 16384, 256, 512, FOLD_THREADS));
        // The tile's admission does not take the fold's block size, so a fold block never runs the warp tile.
        assert!(!tile_admitted(144, 4096, 256, 512, FOLD_THREADS));
    }
}
