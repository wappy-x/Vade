use super::*;

/// The checkpoint's shape on four ranks (widths as the session's state vectors hold them).
fn shape(slot: usize) -> Shape {
    let state = (0..48).map(|l| if l % 4 == 3 { LayerState::Rows([640, 640, 128]) } else { LayerState::Whole([180_000, 460_000, if l == 0 { 92_160 } else { 0 }]) }).collect();
    Shape { slot, slots: 8, input: 377, state, draft: Some([640, 640, 128]) }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Which ranks take message `at` of each origin.
fn takers(ops: &[Vec<Op>]) -> HashMap<(usize, u64), Vec<usize>> {
    let mut map: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (rank, stream) in ops.iter().enumerate() {
        for op in stream {
            if let Op::Take { origin, at, .. } | Op::Drop { origin, at } = *op { map.entry((origin, at)).or_default().push(rank); }
        }
    }
    map
}

/// Run every rank's stream under random timing: a rank stops at a take whose message is not out.
/// A put of message s needs message s - slots taken by its takers and the path's far ends (the
/// relays of tp_route read the slot until then); a stall of every rank is a deadlock.
fn simulate(ops: &[Vec<Op>], slots: u64, seed: u64) {
    let world = ops.len();
    let (mut rng, who) = (Rng(seed), takers(ops));
    let (mut pc, mut puts, mut taken) = (vec![0; world], vec![0u64; world], std::collections::HashSet::new());
    loop {
        let open: Vec<usize> = (0..world).filter(|&r| pc[r] < ops[r].len() && match ops[r][pc[r]] {
            Op::Take { origin, at, .. } | Op::Drop { origin, at } => puts[origin] >= at,
            _ => true,
        }).collect();
        if open.is_empty() {
            assert!((0..world).all(|r| pc[r] == ops[r].len()), "deadlock at {:?}: {:?}", pc, (0..world).map(|r| ops[r].get(pc[r])).collect::<Vec<_>>());
            return;
        }
        let r = open[rng.next() as usize % open.len()];
        match ops[r][pc[r]] {
            Op::Put(_) => {
                puts[r] += 1;
                if let Some(old) = puts[r].checked_sub(slots).filter(|&s| s > 0) {
                    for &t in who.get(&(r, old)).map_or(&[][..], Vec::as_slice) {
                        assert!(taken.contains(&(t, r, old)), "rank {r} put {} over message {old} that rank {t} has not taken", puts[r]);
                    }
                }
            }
            Op::Take { origin, at, .. } | Op::Drop { origin, at } => {
                assert!(at + slots > puts[origin], "rank {r} takes message {at} of {origin} after it was overwritten");
                taken.insert((r, origin, at));
            }
            _ => (),
        }
        pc[r] += 1;
    }
}

#[test]
fn the_wire_has_no_deadlock_or_overwrite_under_any_timing() {
    for s in [shape(8 << 20), Shape { draft: None, ..shape(8 << 20) }] {
        for (rows, world) in [(6144, 4), (32_064, 4), (32_768, 4), (16_320, 4), (48_000, 4), (24_000, 3)] {
            let plan = Plan::new(rows, world, 4096, &s).unwrap();
            let ops = plan.ops(&s);
            for seed in 1..=300 { simulate(&ops, 8, seed * 7919); }
        }
    }
}

/// Both path ends take every message that travels toward them.
#[test]
fn the_ends_take_every_message() {
    for s in [shape(8 << 20), Shape { draft: None, ..shape(8 << 20) }] {
        let plan = Plan::new(32_064, 4, 4096, &s).unwrap();
        let ops = plan.ops(&s);
        let (who, puts) = (takers(&ops), Plan::puts(&ops));
        for origin in 0..4 {
            for at in 1..=puts[origin] {
                let t = &who[&(origin, at)];
                if origin != 0 { assert!(t.contains(&0), "rank 0 misses {origin} {at}"); }
                if origin != 3 { assert!(t.contains(&3), "rank 3 misses {origin} {at}"); }
            }
        }
    }
}

/// Before layer l a rank takes that layer's rows of every earlier part, once, on row edges, and
/// the GDN state only of the part before it; its stages follow.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn each_rank_reads_every_earlier_row_of_each_layer_before_its_stages() {
    let s = shape(8 << 20);
    let plan = Plan::new(32_064, 4, 4096, &s).unwrap();
    let ops = plan.ops(&s);
    for rank in 0..4 {
        for l in 0..48 {
            let first = ops[rank].iter().position(|op| matches!(op, Op::Stage { layer, tile: 0 } if *layer == l)).unwrap();
            let mut got = [Vec::new(), Vec::new(), Vec::new()];
            let mut gdn = Vec::new();
            for op in &ops[rank][..first] {
                if let Op::Take { origin, msg: Msg::State { layer, k, from, words }, .. } = *op && layer == l {
                    if matches!(s.state[l], LayerState::Rows(_)) { got[k].push((from, words)); } else { gdn.push(origin); }
                }
            }
            let start = plan.part(rank).start;
            if let LayerState::Rows(width) = s.state[l] {
                for k in 0..3 {
                    got[k].sort_unstable();
                    let mut next = 0;
                    for &(from, words) in &got[k] { assert_eq!(from, next); next += words; }
                    assert_eq!(next, start * width[k], "rank {rank} layer {l} vector {k}");
                }
            } else {
                gdn.dedup();
                let before = if rank == 0 { vec![3] } else if rank == 1 { vec![] } else { vec![rank - 1] };
                assert_eq!(gdn, before, "rank {rank} layer {l}");
            }
        }
    }
}

/// Rank 0 hashes the rows in token order; each back has its tiles' input before its first stage,
/// and each part has the PLE history of the part before it before layer 0.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn every_rank_starts_from_its_input_and_the_ple_history() {
    let s = shape(8 << 20);
    let plan = Plan::new(32_064, 4, 4096, &s).unwrap();
    let ops = plan.ops(&s);
    let hashed: Vec<_> = ops[0].iter().filter_map(|op| if let Op::Pack { owner, tile } = *op { Some(plan.span(owner, tile)) } else { None }).collect();
    let mut at = 0;
    for span in hashed { assert_eq!(span.start, at); at = span.end; }
    assert_eq!(at, 32_064);
    for rank in 0..4 {
        let first = ops[rank].iter().position(|op| matches!(op, Op::Stage { .. })).unwrap();
        for tile in 0..plan.tiles(rank) {
            let packed = ops[rank][..first].iter().filter(|op| match **op {
                Op::Pack { owner, tile: k } => (owner, k) == (rank, tile),
                Op::Take { msg: Msg::Seed { owner, tile: k, words }, .. } => (owner, k) == (rank, tile) && words == plan.input(rank, tile, &s).len(),
                _ => false,
            }).count();
            assert_eq!(packed, 1, "rank {rank} tile {tile}");
        }
        // The input vector holds every tile's words and the token after the part.
        assert_eq!(plan.input_words(rank, &s), plan.input(rank, plan.tiles(rank) - 1, &s).start + 4096 * 377 + 1);
        let ple: Vec<_> = ops[rank][..first].iter().filter_map(|op| match *op { Op::Take { origin, msg: Msg::State { layer: 0, k: 2, .. }, .. } => Some(origin), _ => None }).collect();
        assert_eq!(ple, if rank == 1 { vec![] } else if rank == 0 { vec![3] } else { vec![rank - 1] }, "rank {rank}");
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn parts_cover_the_rows_in_wave_order_on_the_grid() {
    let s = shape(8 << 20);
    let plan = Plan::new(32_064, 4, 4096, &s).unwrap();
    let mut next = 0;
    for rank in [1, 2, 3, 0] {
        let part = plan.part(rank);
        assert_eq!((part.start, part.start % 64), (next, 0));
        let tiles: Vec<_> = (0..plan.tiles(rank)).map(|k| plan.span(rank, k)).collect();
        assert_eq!((tiles[0].start, tiles.last().unwrap().end), (part.start, part.end));
        assert!(tiles.iter().all(|t| t.start % 64 == 0 && t.len() % 64 == 0 && t.len() <= 4096));
        next = part.end;
    }
    assert_eq!(next, 32_064);
    assert_eq!((plan.part(1).len(), plan.part(2).len(), plan.part(3).len(), plan.part(0).len()), (8000, 8000, 8000, 8064));
    // Every part stays within world x 32 rows of the even share, on the 64-row grid.
    for rows in [6144, 8128, 14_464, 16_320, 24_000, 32_064, 37_440, 48_000] {
        let part = part_rows(rows, 4);
        assert!(part.is_multiple_of(64) && (rows - 3 * part).abs_diff(rows / 4) < 4 * 32, "{rows}");
    }
    // Every rank holds the first part's rows, so the push skips them.
    assert_eq!(shared_rows(32_064, 4), plan.part(1).end);
    // Each part's layers run in order, each tile once.
    let ops = plan.ops(&s);
    for rank in 0..4 {
        let stages: Vec<(usize, usize)> = ops[rank].iter().filter_map(|op| if let Op::Stage { layer, tile } = *op { Some((layer, tile)) } else { None }).collect();
        let want: Vec<_> = (0..48).flat_map(|l| (0..plan.tiles(rank)).map(move |k| (l, k))).collect();
        assert_eq!(stages, want);
    }
    // Each rank observes its own tiles after the last row before them, and rank 0 takes every
    // other part's draft rows.
    for rank in 0..4 {
        let before = ops[rank].iter().position(|op| matches!(op, Op::Observe { .. })).unwrap();
        let observed: Vec<usize> = ops[rank].iter().filter_map(|op| if let Op::Observe { tile } = *op { Some(tile) } else { None }).collect();
        assert_eq!(observed, (0..plan.tiles(rank)).collect::<Vec<_>>());
        let next = ops[rank][..before].iter().filter(|op| matches!(op, Op::Take { msg: Msg::Next { owner }, .. } if *owner == rank)).count();
        assert_eq!(next, usize::from(rank != 0), "rank {rank}");
    }
    for k in 0..3 {
        let mut at = 0;
        let width = s.draft.unwrap()[k];
        for op in &ops[0] {
            if let Op::Take { msg: Msg::Draft { k: vector, from, words, .. }, .. } = *op && vector == k {
                assert_eq!(from, at * width);
                at += words / width;
            }
        }
        assert_eq!(at + plan.part(0).len(), 32_064);
    }
}

/// Every position but the first is observed once, in the wave's order. A rank reads final rows
/// from its part's first through the row after it, and draft rows of its own part stay on the
/// index pool's 4-row groups.
#[test]
fn the_drafter_observes_each_position_once_on_its_rank() {
    let s = shape(8 << 20);
    for rows in [6144, 8128, 14_464, 32_064, 37_440, 48_000] {
        let plan = Plan::new(rows, 4, 4096, &s).unwrap();
        let mut next = 1;
        for rank in [1, 2, 3, 0] {
            let part = plan.part(rank);
            assert!(part.start.is_multiple_of(4));
            for tile in 0..plan.tiles(rank) {
                let seen = plan.observed(rank, tile);
                assert_eq!(seen.start, next, "{rows} rows, rank {rank} tile {tile}");
                assert!(seen.start > part.start && seen.end <= part.end + 1);
                next = seen.end;
            }
            if rank != 0 { assert_eq!(next, part.end + 1, "a part's draft rows end on its last row"); }
        }
        assert_eq!(next, rows);
    }
}

/// The push sends the rows after the shared ones, tail rows included, and both ends cut the same
/// list from the same words.
#[test]
fn the_push_skips_only_the_rows_every_rank_holds() {
    let (rows, tail) = (32_064, 35);
    let shared = shared_rows(rows, 4);
    for width in [640, 128] {
        let words = (rows + tail) * width;
        let first = push_first_word(words, rows + tail, shared);
        assert_eq!(first, shared * width);
        assert_eq!(words - first, (rows + tail - shared) * width);
        let mut at = first;
        for (from, n) in slices(words - first, 1 << 20) { assert_eq!(first + from, at); at += n; }
        assert_eq!(at, words);
    }
    assert_eq!(push_first_word(0, 0, shared), 0);
    assert_eq!(push_first_word(640, 1, shared), 640);
    // A second push holds every attention row: it sends no attention word.
    assert_eq!(push_first_word(640, 40, 40), 640);
}

/// A prompt too short for MIN_PART rows per rank, or a slot that a tile's input, three layers or a
/// part's draft rows do not fit, is refused.
#[test]
fn the_plan_refuses_what_it_cannot_carry() {
    let s = shape(8 << 20);
    assert!(Plan::new(16_320, 4, 4096, &s).is_some());
    assert!(Plan::new(2_048, 4, 4096, &s).is_none());
    assert!(Plan::new(32_065, 4, 4096, &s).is_none());
    assert!(Plan::new(32_064, 4, 4096, &shape(1 << 20)).is_none());
    // A tile's input is one message.
    let wide = Shape { input: (8 << 20) / 4096 + 1, ..shape(8 << 20) };
    assert!(Plan::new(32_064, 4, 4096, &wide).is_none());
    // A part's draft rows must leave a slot free: 3 + 3 + 1 messages pass, 3 + 3 + 2 do not.
    assert!(Plan::new(32_064, 4, 4096, &Shape { draft: Some([2200, 2200, 640]), ..shape(8 << 20) }).is_some());
    assert!(Plan::new(32_064, 4, 4096, &Shape { draft: Some([2200, 2200, 1100]), ..shape(8 << 20) }).is_none());
    let full = Shape { draft: Some([2097 * 2, 2097 * 2, 2097 * 2]), ..shape(8 << 20) };
    assert!(Plan::new(32_064, 4, 4096, &full).is_none());
    assert!(Plan::new(32_064, 4, 4096, &Shape { draft: None, ..full }).is_some());
}
