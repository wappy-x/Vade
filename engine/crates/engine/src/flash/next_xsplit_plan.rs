//! Host plan of a cold prompt's sequence split over more than two ranks. The rows split in one
//! contiguous part per rank, in wave order: rank 1 holds the first part, then rank 2 and so on,
//! and rank 0 the last part (the head and the first token). Rank 0 hashes every row
//! in token order, reads its compact frontend input and sends each part its input tiles. Each
//! rank then runs the frontend on its tiles and sweeps its part layer by layer. Before layer l it
//! takes layer l's rows of every earlier part (keys, values, raw index) and the GDN state at the
//! end of the part before it, so its rows keep world one's arithmetic. The PLE history crosses
//! with layer 0's state, as the PLE update ends layer 0's stage. The drafter is one more stage:
//! its row i reads final row i and token i + 1 and no other row, so each rank writes the draft
//! rows of its own part, on the 4-row groups the index pools. Rank 0 takes every other part's
//! draft rows.
//!
//! Every message of a world-wide mailbox reaches every rank, and a relay reads each origin's
//! messages in strict order (spark_gpu_probe::tp_route). So no origin may reuse a slot before the
//! ranks that read its message and both path ends took it: the ends take every message (one
//! word where they do not need it), and a sender waits for a credit that follows those takes.
//! Rank 0's pad after layer l follows its takes of every part's layer l, and each of those follows
//! that rank's own takes of layer l, so one pad credits every sender. `Plan::ops` fixes every
//! put and take of every rank from the shape of the work, with no run-time count.

use std::collections::HashMap;
use std::ops::Range;

/// A rank's export of layer l waits for rank 0's pad of the layers below l + 1 - PAD_LAG. The
/// wave's first rank runs world - 2 layers ahead of rank 0's takes, so three layers keep four
/// ranks from waiting.
pub(crate) const PAD_LAG: usize = 3;
/// The fewest rows of a part.
pub(crate) const MIN_PART: usize = 1024;

/// The words of one layer's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerState {
    /// Attention: words per row of each state vector (keys, values, raw index).
    Rows([usize; 3]),
    /// GDN: the whole length of each state vector (state, conv history, and at layer 0 the PLE
    /// history); only the next part reads it.
    Whole([usize; 3]),
}

/// The shape of the work: what every rank knows before the first message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    /// Words of one mailbox message and slots per channel.
    pub slot: usize,
    pub slots: usize,
    /// Most words of one row's compact frontend input.
    pub input: usize,
    /// Each layer's state.
    pub state: Vec<LayerState>,
    /// Words per row of the drafter's state vectors; None: the chat has no drafter.
    pub draft: Option<[usize; 3]>,
}

/// One mailbox message, named so that a message of an origin has one name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Msg {
    /// Rank 0's credit: its takes of the layers below `layer` are queued (0: its seeds are out).
    LayerPad { layer: usize },
    /// A credit for tile `tile` of rank `owner`'s part: its input taken.
    SeedPad { owner: usize, tile: usize },
    /// The compact frontend input of a tile, one message.
    Seed { owner: usize, tile: usize, words: usize },
    /// Words [from, from + words) of state vector `k` of layer `layer`.
    State { layer: usize, k: usize, from: usize, words: usize },
    /// The token after rank `owner`'s part (its last draft row reads it), one word.
    Next { owner: usize },
    /// Words [from, from + words) of the drafter's state vector `k`, in rank `owner`'s rows.
    Draft { owner: usize, k: usize, from: usize, words: usize },
}

/// One step of a rank, in the order the rank runs it. `at` counts the origin's messages from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    /// Rank 0: hash the rows of tile `tile` of rank `owner`'s part and read their compact
    /// frontend input, for the seed message that follows or for rank 0's own stage.
    Pack { owner: usize, tile: usize },
    /// Set layer `layer`'s position at this rank's part (attention rows, GDN cached flag).
    Enter { layer: usize },
    /// Layer `layer` of this rank's tile `tile`; layer 0 starts with the tile's frontend.
    Stage { layer: usize, tile: usize },
    Put(Msg),
    /// Take message `at` of `origin`, which is `msg`.
    Take { origin: usize, at: u64, msg: Msg },
    /// Take one word of message `at` of `origin`.
    Drop { origin: usize, at: u64 },
    /// The drafter observes this rank's tile `tile`.
    Observe { tile: usize },
}

/// Marks a take of no words it needs, until `Plan::place`.
const DROPPED: u64 = u64::MAX;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub rows: usize,
    pub world: usize,
    pub tile: usize,
    /// Row edges in wave order: rank r >= 1 holds [edges[r - 1], edges[r]), rank 0 the last.
    edges: Vec<usize>,
}

impl Plan {
    /// The plan of `rows` rows over `world` ranks in `tile`-row tiles, or None when the rows do
    /// not give every rank MIN_PART rows, an edge leaves the 64-row grid, a tile's input passes
    /// one message, or three layers' messages or a part's draft rows pass the mailbox's slots.
    pub(crate) fn new(rows: usize, world: usize, tile: usize, shape: &Shape) -> Option<Self> {
        if tile == 0 || !tile.is_multiple_of(64) || !rows.is_multiple_of(64) { return None; }
        let part = part_rows(rows, world);
        let mut edges: Vec<usize> = (0..world).map(|i| i * part).collect();
        edges.push(rows);
        let plan = Self { rows, world, tile, edges };
        let fits = tile * shape.input <= shape.slot
            && (0..world).all(|r| plan.part(r).len() >= MIN_PART)
            && (1..world).all(|r| plan.draft_rows(r, shape).len() < shape.slots)
            && (1..world).all(|r| (0..shape.state.len()).all(|l| {
                (l.saturating_sub(PAD_LAG - 1)..=l).map(|l| plan.state(r, l, shape).len()).sum::<usize>() <= shape.slots
            }));
        fits.then_some(plan)
    }
    /// Rank `rank`'s rows.
    pub(crate) fn part(&self, rank: usize) -> Range<usize> {
        let i = if rank == 0 { self.world - 1 } else { rank - 1 };
        self.edges[i]..self.edges[i + 1]
    }
    /// Rank `rank`'s tiles; the last one may be short.
    pub(crate) fn tiles(&self, rank: usize) -> usize { self.part(rank).len().div_ceil(self.tile) }
    /// Rows of tile `k` of rank `rank`'s part.
    pub(crate) fn span(&self, rank: usize, k: usize) -> Range<usize> {
        let part = self.part(rank);
        let start = part.start + k * self.tile;
        start..(start + self.tile).min(part.end)
    }
    /// The positions the drafter observes in tile `k` of rank `rank`. Draft row i reads final row
    /// i and token i + 1, and position 0 only starts the history, so a tile observes the positions
    /// one past its rows; its last row's draft row needs the next tile's first token, and a part's
    /// last one the token after the part (Msg::Next).
    pub(crate) fn observed(&self, rank: usize, k: usize) -> Range<usize> {
        let span = self.span(rank, k);
        span.start + 1..(span.end + 1).min(self.rows)
    }
    /// Words of rank `rank`'s input vector: its tiles' input and the token after its part.
    pub(crate) fn input_words(&self, rank: usize, shape: &Shape) -> usize { self.tiles(rank) * self.tile * shape.input + 1 }
    /// Words of tile `k`'s input in rank `rank`'s input vector.
    pub(crate) fn input(&self, rank: usize, k: usize, shape: &Shape) -> Range<usize> {
        let at = k * self.tile * shape.input;
        at..at + self.span(rank, k).len() * shape.input
    }
    /// The state rank `rank` exports after layer `layer`, in order.
    fn state(&self, rank: usize, layer: usize, shape: &Shape) -> Vec<Msg> {
        let part = self.part(rank);
        let mut out = Vec::new();
        for k in 0..3 {
            let (start, words) = match shape.state[layer] {
                LayerState::Rows(width) => (part.start * width[k], part.len() * width[k]),
                LayerState::Whole(len) => (0, len[k]),
            };
            out.extend(slices(words, shape.slot).map(|(from, n)| Msg::State { layer, k, from: start + from, words: n }));
        }
        out
    }
    /// The draft rows rank `owner` sends rank 0, in order.
    fn draft_rows(&self, owner: usize, shape: &Shape) -> Vec<Msg> {
        let part = self.part(owner);
        let Some(width) = shape.draft else { return Vec::new() };
        (0..3).flat_map(|k| slices(part.len() * width[k], shape.slot)
            .map(move |(from, n)| Msg::Draft { owner, k, from: part.start * width[k] + from, words: n })).collect()
    }
    /// Every rank's steps. The ranks' streams are built side by side in one walk of the work;
    /// then each take finds its message's place in the origin's stream.
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    pub(crate) fn ops(&self, shape: &Shape) -> Vec<Vec<Op>> {
        let (world, end) = (self.world, self.world - 1);
        let layers = shape.state.len();
        let mut ops: Vec<Vec<Op>> = vec![Vec::new(); world];
        let take = |origin, msg| Op::Take { origin, at: 0, msg };
        let dropped = |origin, msg| Op::Take { origin, at: DROPPED, msg };
        // Seeds, in wave order and one message per tile. Rank 0 keeps one tile unconfirmed. The
        // far end takes every seed.
        let mut last = None;
        for owner in 1..world {
            for k in 0..self.tiles(owner) {
                let seed = Msg::Seed { owner, tile: k, words: self.input(owner, k, shape).len() };
                let pad = Msg::SeedPad { owner, tile: k };
                ops[0].push(Op::Pack { owner, tile: k });
                if let Some(paid) = last.replace((owner, pad)) { seed_credits(&mut ops[0], paid, end); }
                ops[0].push(Op::Put(seed));
                ops[owner].push(take(0, seed));
                ops[owner].push(Op::Put(pad));
                if owner != end {
                    ops[end].push(dropped(0, seed));
                    ops[end].push(dropped(owner, pad));
                    ops[end].push(Op::Put(pad));
                }
            }
            if shape.draft.is_some() {
                let next = Msg::Next { owner };
                ops[0].push(Op::Put(next));
                ops[owner].push(take(0, next));
                if owner != end { ops[end].push(dropped(0, next)); }
            }
        }
        if let Some(paid) = last { seed_credits(&mut ops[0], paid, end); }
        ops[0].push(Op::Put(Msg::LayerPad { layer: 0 }));
        ops[0].extend((0..self.tiles(0)).map(|tile| Op::Pack { owner: 0, tile }));
        // Layers, in wave order within a layer.
        for l in 0..layers {
            for rank in (1..world).chain(std::iter::once(0)) {
                let earlier = if rank == 0 { world } else { rank };
                if earlier > 1 { ops[rank].push(Op::Enter { layer: l }); }
                let whole = matches!(shape.state[l], LayerState::Whole(_));
                for o in 1..earlier {
                    for m in self.state(o, l, shape) {
                        if !whole || o + 1 == earlier { ops[rank].push(take(o, m)); } else if rank == 0 || rank == end { ops[rank].push(dropped(o, m)); }
                    }
                }
                if rank == 0 { ops[0].push(Op::Put(Msg::LayerPad { layer: l + 1 })); }
                ops[rank].extend((0..self.tiles(rank)).map(|tile| Op::Stage { layer: l, tile }));
                if rank != 0 {
                    if l + 1 >= PAD_LAG { ops[rank].push(take(0, Msg::LayerPad { layer: l + 1 - PAD_LAG })); }
                    ops[rank].extend(self.state(rank, l, shape).into_iter().map(Op::Put));
                }
            }
        }
        // The last exports are read before a rank's rows start.
        for rank in 1..world { ops[rank].extend((layers + 1 - PAD_LAG..=layers).map(|layer| take(0, Msg::LayerPad { layer }))); }
        // The drafter, one more stage: each rank observes its rows, and rank 0 takes the draft
        // rows of every other part. The takes above confirm every earlier message of these origins.
        if shape.draft.is_some() {
            for (rank, stream) in ops.iter_mut().enumerate() { stream.extend((0..self.tiles(rank)).map(|tile| Op::Observe { tile })); }
            for rank in 1..world {
                for m in self.draft_rows(rank, shape) {
                    ops[rank].push(Op::Put(m));
                    ops[0].push(take(rank, m));
                    if rank != end { ops[end].push(dropped(rank, m)); }
                }
            }
        }
        place(ops)
    }
    /// Messages each rank put, for the ranks' counts after the transaction.
    pub(crate) fn puts(ops: &[Vec<Op>]) -> Vec<u64> {
        ops.iter().map(|stream| stream.iter().filter(|op| matches!(op, Op::Put(_))).count() as u64).collect()
    }
}

/// Rows of each part but rank 0's, which takes the rest. The ranks wait for each other layer by
/// layer, so the wave moves at the pace of its largest part, and a smaller part only waits. The
/// parts are equal on the 64-row grid, so rank 0's differs from them by at most `world` x 32 rows.
fn part_rows(rows: usize, world: usize) -> usize {
    (rows / world + 32) / 64 * 64
}
/// The first rows every rank holds after the split: rank r holds the rows up to its part's end
/// (it takes each earlier part's rows), and the first part ends here. Rank 0's push skips them.
pub(crate) fn shared_rows(rows: usize, world: usize) -> usize { part_rows(rows, world) }
/// The first word of a state vector of `words` words at `rows` rows that a push sends, when every
/// rank holds its first `shared` rows. Both ends of the push compute their piece lists from it.
#[allow(clippy::manual_checked_ops)] // size arithmetic stays as written.
pub(crate) fn push_first_word(words: usize, rows: usize, shared: usize) -> usize {
    if rows == 0 { 0 } else { (words / rows * shared).min(words) }
}
/// `words` words as (first word, words) of at most `slot` words each.
fn slices(words: usize, slot: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..words).step_by(slot.max(1)).map(move |from| (from, slot.min(words - from)))
}
fn seed_credits(front: &mut Vec<Op>, (owner, pad): (usize, Msg), end: usize) {
    front.push(Op::Take { origin: owner, at: DROPPED, msg: pad });
    if owner != end { front.push(Op::Take { origin: end, at: DROPPED, msg: pad }); }
}
/// Give each take its message's place in the origin's stream; a take of no words it needs
/// becomes a drop.
fn place(mut ops: Vec<Vec<Op>>) -> Vec<Vec<Op>> {
    let mut at: HashMap<(usize, Msg), u64> = HashMap::new();
    for (origin, stream) in ops.iter().enumerate() {
        let mut n = 0;
        for op in stream {
            if let Op::Put(msg) = *op {
                n += 1;
                at.insert((origin, msg), n);
            }
        }
    }
    for stream in &mut ops {
        for op in stream.iter_mut() {
            if let Op::Take { origin, at: mark, msg } = *op {
                let place = at[&(origin, msg)];
                *op = if mark == DROPPED { Op::Drop { origin, at: place } } else { Op::Take { origin, at: place, msg } };
            }
        }
    }
    ops
}

#[cfg(test)]
#[path = "next_xsplit_plan_tests.rs"]
mod tests;
