//! Host plan of a two-rank layer-major prompt transaction (the long token split). A cold prompt
//! of `rows` rows splits at `cut`: the worker (rank 1) holds the prefix [0, cut), the front
//! (rank 0) the suffix [cut, rows), the head and the drafter. Both ranks visit their rows in
//! tiles of `tile` rows. The front runs the frontend and layers [0, split) for every row in token
//! order and sends each prefix tile's residual (a seed tile) at once; the worker runs layer
//! `split` on each seed tile, then sweeps the later layers, each over its tiles in token order,
//! and exports a layer's causal state at `cut` after its last tile. The front runs the suffix
//! layer by layer: it imports layer l's state, then runs all suffix tiles of l in order.
//!
//! A tile stage of layer l ends after layer l + 1's attention mix (flash/next_stage.rs stage_mix),
//! while its expert outputs are live, so the MoE combine stays fused. Each rank keeps one bank:
//! per row the residual, the mixed rows and the HC injection words (`bank_row_bytes`). The
//! bank is funded on both ranks before either starts; if it does not fit, the prompt keeps the
//! pipeline tiles. Every tile edge is a multiple of 64 rows from row 0 (a GDN chunk edge, as the
//! token split's cut), and inner tiles keep main's kernels, so each row runs the same operations.
//! The cut is a share of the rows (`cut_at`); the front also runs layers [0, split) of every row
//! and its drafter observes every row.

/// One step of a rank's schedule, in the order the rank runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Front: frontend and layers [0, split) of prefix tile `k`, then send its residual rows.
    Seed(usize),
    /// Front: frontend and layers [0, split) of suffix tile `k`, then layer `split`'s mix.
    Head(usize),
    /// Either rank: layer `l` of tile `k` (the worker's prefix tiles, the front's suffix tiles);
    /// it ends after layer l + 1's attention mix, or with the whole residual at the last layer.
    Stage(usize, usize),
    /// Worker: send layer `l`'s causal state at the cut.
    Export(usize),
    /// Front: take layer `l`'s causal state at the cut.
    Import(usize),
    /// Prefix tile `k`'s final rows: the worker sends them, the front takes them (its drafter
    /// observes the prefix rows before the suffix rows).
    Rows(usize),
}

/// Layers the worker runs past a layer before it exports that layer's state (`SplitPlan::worker`).
pub(crate) const EXPORT_LAG: usize = 2;
/// The worker's first layers that run tile by tile as the seed tiles arrive (`SplitPlan::worker`).
pub(crate) const SEED_WAVE: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SplitPlan {
    pub rows: usize,
    pub cut: usize,
    pub tile: usize,
}

impl SplitPlan {
    /// The plan of a cold prompt of `rows` rows in `tile`-row tiles cut at the tile edge nearest
    /// rows / 2 (`with_cut`).
    #[cfg(test)]
    pub(crate) fn new(rows: usize, tile: usize) -> Option<Self> {
        Self::with_cut(rows, tile, (rows + tile) / (2 * tile) * tile)
    }
    /// The plan cut at `cut`, or None when the prompt has fewer than four tiles (the bounded token
    /// split and the pipeline cover those) or the cut is off a 64-row edge or leaves either
    /// side less than one tile.
    pub(crate) fn with_cut(rows: usize, tile: usize, cut: usize) -> Option<Self> {
        if tile == 0 || !tile.is_multiple_of(64) || rows < 4 * tile { return None; }
        (cut.is_multiple_of(64) && cut >= tile && cut + tile <= rows).then_some(Self { rows, cut, tile })
    }
    /// Prefix tiles [0, cut); the last one may be short.
    pub(crate) fn prefix_tiles(&self) -> usize { self.cut.div_ceil(self.tile) }
    /// Suffix tiles [cut, rows); the last one may be short.
    pub(crate) fn suffix_tiles(&self) -> usize { (self.rows - self.cut).div_ceil(self.tile) }
    /// Rows of prefix tile `k` (worker) or suffix tile `k` (front).
    pub(crate) fn prefix_span(&self, k: usize) -> std::ops::Range<usize> { k * self.tile..((k + 1) * self.tile).min(self.cut) }
    pub(crate) fn suffix_span(&self, k: usize) -> std::ops::Range<usize> {
        let start = self.cut + k * self.tile;
        start..(start + self.tile).min(self.rows)
    }
    /// Bank rows: [worker, front].
    pub(crate) fn bank_rows(&self) -> [usize; 2] { [self.cut, self.rows - self.cut] }
    /// KV rows of each rank's attention and draft caches that the transaction writes: [worker,
    /// front]. The worker writes the prefix in layers [split, layers); the front writes every
    /// row (layers [0, split), the imported prefix state, the suffix and its drafter's rows).
    pub(crate) fn kv_rows(&self) -> [usize; 2] { [self.cut, self.rows] }
    /// The worker's schedule over layers [split, layers). Layer l's state at the cut is final
    /// after its last tile, so its export waits until layer l + EXPORT_LAG ran: the worker can
    /// run that many layers ahead of the front (which first runs layers [0, split) of every row)
    /// before a full mailbox stops it. The lag tapers at the end: from the layer whose lagged
    /// export would come after the last layer, each layer goes out right after its tiles, so
    /// the front runs one layer after the worker's last, not EXPORT_LAG + 1. The first
    /// SEED_WAVE layers run tile by tile (each seed tile through them as it arrives, while the
    /// front still sends later seeds); the rest layer by layer.
    pub(crate) fn worker(&self, split: usize, layers: usize) -> Vec<Step> {
        let tiles = self.prefix_tiles();
        let wave = (split + SEED_WAVE).min(layers);
        let stages = (0..tiles).flat_map(|k| (split..wave).map(move |l| (l, k)))
            .chain((wave..layers).flat_map(|l| (0..tiles).map(move |k| (l, k))));
        let mut steps = Vec::with_capacity(tiles * (layers - split + 1) + layers - split);
        let mut next = split;
        for (l, k) in stages {
            steps.push(Step::Stage(l, k));
            if k + 1 < tiles { continue; }
            // Layer l is complete (layers complete in order).
            let upto = if l + EXPORT_LAG + 1 >= layers { l + 1 } else { (l + 1).saturating_sub(EXPORT_LAG) };
            while next < upto { steps.push(Step::Export(next)); next += 1; }
        }
        steps.extend((0..tiles).map(Step::Rows));
        steps
    }
    /// The front's schedule: every row through layers [0, split) in token order, then the
    /// suffix layer by layer from `split`, each layer after its import. The worker sends the
    /// prefix's final rows after its last export, so the front takes them between its last
    /// layer's tiles (each after a tile), not after them.
    pub(crate) fn front(&self, split: usize, layers: usize) -> Vec<Step> {
        let (tiles, prefix) = (self.suffix_tiles(), self.prefix_tiles());
        let mut steps: Vec<Step> = (0..prefix).map(Step::Seed).collect();
        steps.extend((0..tiles).map(Step::Head));
        for l in split..layers {
            steps.push(Step::Import(l));
            for k in 0..tiles {
                steps.push(Step::Stage(l, k));
                if l + 1 == layers && k < prefix { steps.push(Step::Rows(k)); }
            }
        }
        steps.extend((tiles.min(prefix)..prefix).map(Step::Rows));
        steps
    }
}

/// What a rank funds before a transaction (flash/next_split.rs).
pub(crate) trait Funds {
    type Bank;
    type Work;
    /// Map the KV pages of rows 0..rows of every attention and draft cache of the chat
    /// (NextSession::fit_kv); false when they do not fit.
    fn pages(&mut self, rows: usize) -> bool;
    fn bank(&mut self, rows: usize) -> Option<Self::Bank>;
    fn work(&mut self) -> Option<Self::Work>;
}

/// A rank's funding (`worker`: rank 1), all before its first stage: the KV pages of every row it
/// writes (a pass maps its rows the same way before it queues a kernel), then its bank, then
/// its workspace. None at the first that fails: the rank refuses and nothing ran, so the prompt
/// keeps the usual path, which maps its rows pass by pass (pages mapped so far stay with the
/// session, as a pass's do).
pub(crate) fn fund<F: Funds>(f: &mut F, plan: &SplitPlan, worker: bool) -> Option<(F::Bank, F::Work)> {
    let rank = usize::from(!worker);
    if !f.pages(plan.kv_rows()[rank]) { return None; }
    let bank = f.bank(plan.bank_rows()[rank])?;
    Some((bank, f.work()?))
}

// The host model of flash/next_split.rs's mailbox order and bank size, for the tests below.

/// One mailbox message of the transaction: the worker's funding answer (one word, first), a
/// seed tile's piece (front to worker), a layer's state piece or a prefix tile's final rows
/// piece (worker to front), or a one-word credit pad.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wire {
    Answer,
    Seed(usize, usize),
    State(usize, usize),
    Rows(usize, usize),
    Pad,
}

/// A rank's mailbox operation, in its stream order.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Put(Wire),
    Take(Wire),
}

/// A rank's mailbox operations for its `steps` (SplitPlan::front or ::worker), `slots` mailbox
/// slots per direction. Pieces per message group: `seed` per seed tile, `rows` per prefix tile,
/// `state(l)` per layer. Credits as the token split's state export: the receiver puts a pad for
/// each data message it takes; the sender takes the pad of message m - slots before it puts
/// message m, so no put reuses a slot whose message is still in flight. The front takes its
/// last seed credits just before its first import (its suffix's first layers run meanwhile),
/// then puts one more pad: the worker takes it before its first export, which reuses the slot
/// of a seed credit (a pad earns no credit, so only this pad shows the front took them). The
/// worker takes its last credits after its last rows. The worker first answers whether it funded
/// its bank; the front takes the answer before its first seed (a refusal ends both lists there,
/// so the answer's slot is free again once the front took it).
#[cfg(test)]
pub(crate) fn wire(steps: &[Step], front: bool, slots: usize, seed: usize, rows: usize, state: impl Fn(usize) -> usize) -> Vec<Op> {
    let mut ops = Vec::new();
    let mut sent = 0;
    let mut owed = false;
    let put = |ops: &mut Vec<Op>, sent: &mut usize, w: Wire| {
        if *sent >= slots { ops.push(Op::Take(Wire::Pad)); }
        ops.push(Op::Put(w));
        *sent += 1;
    };
    let take = |ops: &mut Vec<Op>, w: Wire| ops.extend([Op::Take(w), Op::Put(Wire::Pad)]);
    ops.push(if front { Op::Take(Wire::Answer) } else { Op::Put(Wire::Answer) });
    for &step in steps {
        match (front, step) {
            (true, Step::Seed(k)) => { for p in 0..seed { put(&mut ops, &mut sent, Wire::Seed(k, p)); } owed = true; }
            (true, Step::Import(l)) => {
                if std::mem::take(&mut owed) {
                    ops.extend((0..sent.min(slots)).map(|_| Op::Take(Wire::Pad)));
                    ops.push(Op::Put(Wire::Pad));
                }
                for p in 0..state(l) { take(&mut ops, Wire::State(l, p)); }
            }
            (true, Step::Rows(k)) => for p in 0..rows { take(&mut ops, Wire::Rows(k, p)); },
            (false, Step::Stage(l, k)) if first_stage(steps, l) => {
                for p in 0..seed { take(&mut ops, Wire::Seed(k, p)); }
            }
            (false, Step::Export(l)) => {
                if sent == 0 { ops.push(Op::Take(Wire::Pad)); }
                for p in 0..state(l) { put(&mut ops, &mut sent, Wire::State(l, p)); }
            }
            (false, Step::Rows(k)) => for p in 0..rows { put(&mut ops, &mut sent, Wire::Rows(k, p)); },
            _ => {}
        }
    }
    if !front { ops.extend((0..sent.min(slots)).map(|_| Op::Take(Wire::Pad))); }
    ops
}

/// True when layer `l` is the first layer in `steps` (the worker's seed tiles arrive for it).
#[cfg(test)]
fn first_stage(steps: &[Step], l: usize) -> bool {
    steps.iter().find_map(|s| if let Step::Stage(first, _) = s { Some(*first) } else { None }) == Some(l)
}

/// The cut at `permille` of the rows, on the nearest 64-row edge. The cut that balances the ranks'
/// row-layers (rows (layers + drafter) / (2 (layers - split)), 17,088 of 32,064 rows) measured
/// slower than the half cut: the worker waits for its seed tiles and its credits.
pub(crate) fn cut_at(rows: usize, permille: usize) -> usize { (rows * permille / 1000 + 32) / 64 * 64 }

/// Bank bytes per row: the residual (`w` FP32 words), the mixed rows (`h`) and the HC injection
/// words (`streams`), all kept from a tile stage's end to its next stage.
#[cfg(test)]
pub(crate) fn bank_row_bytes(w: usize, h: usize, streams: usize) -> usize { 4 * (w + h + streams) }

/// The bank fits when it and `reserve` bytes fit in `available` bytes (checked, no overflow).
#[cfg(test)]
pub(crate) fn bank_fits(rows: usize, row_bytes: usize, reserve: usize, available: usize) -> bool {
    rows.checked_mul(row_bytes).and_then(|b| b.checked_add(reserve)).is_some_and(|need| need <= available)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_the_prompt_once_on_tile_edges() {
        for tile in [64, 1024, 2048] {
            for rows in [4 * tile - 1, 4 * tile, 32_099, 32_768, 65_536, 131_072] {
                let Some(p) = SplitPlan::new(rows, tile) else { assert!(rows < 4 * tile); continue };
                assert!(p.cut % 64 == 0 && p.cut >= tile && p.cut + tile <= rows, "{rows} {tile}");
                assert!(p.cut.abs_diff(rows - p.cut) <= tile, "balanced cut {rows} {tile}");
                let spans: Vec<_> = (0..p.prefix_tiles()).map(|k| p.prefix_span(k))
                    .chain((0..p.suffix_tiles()).map(|k| p.suffix_span(k))).collect();
                assert_eq!(spans[0].start, 0);
                assert_eq!(spans.last().unwrap().end, rows);
                assert!(spans.windows(2).all(|w| w[0].end == w[1].start && w[1].start % 64 == 0));
                assert!(spans.iter().all(|s| !s.is_empty() && s.len() <= tile));
            }
        }
        assert_eq!(SplitPlan::new(32_768, 2048).map(|p| p.cut), Some(16_384));
        // A balanced cut off the tile edges: the last prefix tile is short, the suffix starts there.
        let cut = cut_at(32_064, 533);
        assert_eq!(cut, 17_088);
        let p = SplitPlan::with_cut(32_064, 2048, cut).unwrap();
        assert_eq!((p.prefix_tiles(), p.prefix_span(8), p.suffix_span(0)), (9, 16_384..17_088, 17_088..19_136));
        assert_eq!(p.bank_rows(), [17_088, 14_976]);
        assert_eq!(cut_at(16_320, 500), 8_192);
        assert_eq!(cut_at(32_064, 560), 17_984);
        assert!(SplitPlan::with_cut(32_064, 2048, 17_000).is_none());
        assert!(SplitPlan::with_cut(32_064, 2048, 31_000 / 64 * 64).is_none());
        assert!(SplitPlan::with_cut(32_064, 2048, 1_024).is_none());
    }

    #[test]
    fn each_layer_exports_after_its_last_tile_and_imports_before_its_first() {
        let (split, layers) = (2, 48);
        let p = SplitPlan::new(32_099, 2048).unwrap();
        let worker = p.worker(split, layers);
        let at = |steps: &[Step], s: Step| steps.iter().position(|&x| x == s).unwrap();
        for l in split..layers {
            let export = at(&worker, Step::Export(l));
            assert!((0..p.prefix_tiles()).all(|k| at(&worker, Step::Stage(l, k)) < export));
            // The export follows layer l + EXPORT_LAG's last tile, in layer order; the last layers'
            // exports taper: each comes right after its own tiles, before the next layer.
            let last = if l + 2 * EXPORT_LAG + 1 >= layers { l.max(layers - EXPORT_LAG - 1) } else { l + EXPORT_LAG };
            assert!(at(&worker, Step::Stage(last, p.prefix_tiles() - 1)) < export, "{l}");
            if last + 1 < layers { assert!(export < at(&worker, Step::Stage(last + 1, 0)), "{l}"); }
            if l > split { assert!(at(&worker, Step::Export(l - 1)) < export); }
            // Each layer's tiles run in token order, each tile after its earlier layer.
            assert!((1..p.prefix_tiles()).all(|k| at(&worker, Step::Stage(l, k - 1)) < at(&worker, Step::Stage(l, k))));
            if l > split { assert!((0..p.prefix_tiles()).all(|k| at(&worker, Step::Stage(l - 1, k)) < at(&worker, Step::Stage(l, k)))); }
            // From split + SEED_WAVE on, a layer starts after the earlier layer's last tile.
            if l >= split + SEED_WAVE { assert!(at(&worker, Step::Stage(l - 1, p.prefix_tiles() - 1)) < at(&worker, Step::Stage(l, 0))); }
        }
        assert_eq!(worker.len(), (layers - split) * (p.prefix_tiles() + 1) + p.prefix_tiles());
        // The seed wave: tile 0 runs its first SEED_WAVE layers before tile 1 starts.
        assert!(at(&worker, Step::Stage(split + SEED_WAVE - 1, 0)) < at(&worker, Step::Stage(split, 1)));
        // After the worker's last tile only the last layer's export and the rows remain.
        assert_eq!(at(&worker, Step::Export(layers - 1)), at(&worker, Step::Stage(layers - 1, p.prefix_tiles() - 1)) + 1);
        assert!(at(&worker, Step::Export(layers - 2)) < at(&worker, Step::Stage(layers - 1, 0)));
        let front = p.front(split, layers);
        // Every row runs layers [0, split) in token order: the seed tiles, then the suffix's.
        let heads: Vec<_> = front.iter().take_while(|s| matches!(s, Step::Seed(_) | Step::Head(_))).collect();
        assert_eq!(heads.len(), p.prefix_tiles() + p.suffix_tiles());
        for l in split..layers {
            let import = at(&front, Step::Import(l));
            assert!((0..p.suffix_tiles()).all(|k| import < at(&front, Step::Stage(l, k))));
            if l > split { assert!(at(&front, Step::Stage(l - 1, p.suffix_tiles() - 1)) < import); }
        }
        // The prefix's final rows come in order, during the front's last layer (after its import).
        let rows: Vec<_> = (0..p.prefix_tiles()).map(|k| at(&front, Step::Rows(k))).collect();
        assert!(rows.windows(2).all(|w| w[0] < w[1]) && at(&front, Step::Import(layers - 1)) < rows[0]);
        assert!(rows[0] < at(&front, Step::Stage(layers - 1, p.suffix_tiles() - 1)));
    }

    /// Runs both ranks' mailbox operations against `slots` slots per direction, picking the
    /// next rank by `pick` (true: the front) among those that can move. A put must find its slot
    /// free (the message `slots` earlier taken); a take needs its message and must match it.
    fn simulate(front: &[Op], worker: &[Op], slots: usize, pick: impl Fn(usize) -> bool) {
        let ranks = [front, worker];
        let (mut at, mut sent, mut taken): ([usize; 2], [Vec<Wire>; 2], [usize; 2]) = ([0; 2], [vec![], vec![]], [0; 2]);
        for turn in 0.. {
            let ready = |r: usize| at[r] < ranks[r].len() && match ranks[r][at[r]] {
                Op::Put(_) => true,
                Op::Take(_) => sent[1 - r].len() > taken[r],
            };
            let r = match (ready(0), ready(1)) {
                (false, false) => break,
                (true, true) => usize::from(!pick(turn)),
                (f, _) => usize::from(!f),
            };
            match ranks[r][at[r]] {
                Op::Put(w) => {
                    // Message m (1-based) reuses the slot of m - slots: the peer took that one.
                    assert!(sent[r].len() < slots + taken[1 - r], "rank {r} overwrites an untaken slot at {:?}", w);
                    sent[r].push(w);
                }
                Op::Take(w) => { assert_eq!(sent[1 - r][taken[r]], w, "rank {r} takes the wrong message"); taken[r] += 1; }
            }
            at[r] += 1;
        }
        assert_eq!(at, [front.len(), worker.len()], "deadlock");
        assert_eq!((taken[0], taken[1]), (sent[1].len(), sent[0].len()), "every message taken");
    }

    #[test]
    fn the_wire_has_no_deadlock_overwrite_or_misread() {
        let (split, layers, slots) = (2, 48, 4);
        for (rows, tile) in [(16_384, 2048), (32_099, 2048), (8_192, 1024)] {
            let p = SplitPlan::new(rows, tile).unwrap();
            // Pieces: 8 per 2,048-row tile at 10 MiB slots; QSA layers 9, GDN layers 2 (32K shapes).
            let (seed, back) = (tile / 256, tile / 256);
            let state = |l: usize| if l % 4 == 3 { 9 } else { 2 };
            let front = wire(&p.front(split, layers), true, slots, seed, back, state);
            let worker = wire(&p.worker(split, layers), false, slots, seed, back, state);
            for pick in [&(|_| true) as &dyn Fn(usize) -> bool, &|_| false, &|t| t % 2 == 0, &|t| t % 7 < 3] {
                simulate(&front, &worker, slots, pick);
            }
            let data = |ops: &[Op]| ops.iter().filter(|o| matches!(o, Op::Put(w) if !matches!(w, Wire::Pad | Wire::Answer))).count();
            assert_eq!(data(&front), p.prefix_tiles() * seed);
            assert_eq!(data(&worker), (split..layers).map(state).sum::<usize>() + p.prefix_tiles() * back);
            // A refusal: the answer alone, taken.
            simulate(&front[..1], &worker[..1], slots, |_| true);
        }
    }

    #[test]
    fn bank_bytes_and_funding() {
        // W 10,240 and H 2,560 FP32 words, 4 HC streams: 51,216 B per row.
        let row = bank_row_bytes(10_240, 2_560, 4);
        assert_eq!(row, 51_216);
        let p = SplitPlan::new(32_768, 2048).unwrap();
        assert_eq!(p.bank_rows().map(|r| r * row), [839_122_944; 2]);
        assert!(bank_fits(16_384, row, 1 << 30, 839_122_944 + (1 << 30)));
        assert!(!bank_fits(16_384, row, 1 << 30, 839_122_944 + (1 << 30) - 1));
        assert!(!bank_fits(usize::MAX, row, 0, usize::MAX));
    }

    /// A chat's KV pages as QsaState maps them: `mapped` rows in whole pages of `page` rows, at
    /// most `room` rows; `log` keeps the funding calls in order.
    struct Pages { mapped: usize, page: usize, room: usize, bank: bool, log: Vec<&'static str> }
    impl Funds for Pages {
        type Bank = usize;
        type Work = ();
        fn pages(&mut self, rows: usize) -> bool {
            self.log.push("pages");
            let want = rows.div_ceil(self.page) * self.page;
            if want > self.room { return false; }
            self.mapped = self.mapped.max(want);
            true
        }
        fn bank(&mut self, rows: usize) -> Option<usize> { self.log.push("bank"); self.bank.then_some(rows) }
        fn work(&mut self) -> Option<()> { self.log.push("work"); Some(()) }
    }

    /// The KV rows a step of either rank writes (at most): its tile's rows, or the prefix for a
    /// layer's state at the cut; the front's prefix rows also enter its drafter.
    fn writes(p: &SplitPlan, step: Step, front: bool) -> std::ops::Range<usize> {
        match step {
            Step::Seed(k) | Step::Rows(k) => p.prefix_span(k),
            Step::Head(k) => p.suffix_span(k),
            Step::Stage(_, k) => if front { p.suffix_span(k) } else { p.prefix_span(k) },
            Step::Import(_) | Step::Export(_) => 0..p.cut,
        }
    }

    #[test]
    fn each_rank_maps_every_kv_row_before_its_first_stage_or_refuses() {
        let (split, layers, page) = (2, 48, 64);
        for rows in [8_256, 12_100, 20_416, 32_099] {
            let p = SplitPlan::with_cut(rows, 2048, cut_at(rows, 500)).unwrap();
            assert_eq!(p.kv_rows(), [p.cut, rows]);
            for worker in [true, false] {
                let steps = if worker { p.worker(split, layers) } else { p.front(split, layers) };
                let end = steps.iter().map(|&s| writes(&p, s, !worker).end).max().unwrap();
                assert_eq!(end, p.kv_rows()[usize::from(!worker)], "{rows} {worker}");
                // A fresh session maps one page (no row hint); a session with a smaller hint
                // grows; one with more keeps its pages. Each covers every row any step writes.
                for mapped in [page, 4_096, 32_768] {
                    let mut f = Pages { mapped, page, room: 32_768, bank: true, log: vec![] };
                    assert!(fund(&mut f, &p, worker).is_some());
                    assert_eq!(f.log, ["pages", "bank", "work"]);
                    assert!(f.mapped >= end && f.mapped == mapped.max(end.div_ceil(page) * page), "{rows} {worker} {mapped}");
                    if mapped < end { assert!(steps.iter().any(|&s| writes(&p, s, !worker).end > mapped)); }
                }
                // Pages that do not fit: the rank refuses before its bank and workspace, so no
                // stage is queued and the session keeps the pages it had.
                let mut f = Pages { mapped: page, page, room: end.div_ceil(page) * page - page, bank: true, log: vec![] };
                assert!(fund(&mut f, &p, worker).is_none());
                assert_eq!((f.log.as_slice(), f.mapped), (&["pages"][..], page));
                // A bank that does not fit after the pages: a refusal before the workspace.
                let mut f = Pages { mapped: page, page, room: 32_768, bank: false, log: vec![] };
                assert!(fund(&mut f, &p, worker).is_none());
                assert_eq!(f.log, ["pages", "bank"]);
            }
        }
    }
}
