//! The two-rank tensor split (flash::next_batched::tp): its selection, wire kinds and split
//! geometry, shared with the host-only pipeline protocol build. It is the only 2-Spark path.

/// A verify pass posted to rank 1: it records its GDN tape (pipeline KIND_VERIFY) and acks
/// without rows, like an asynchronous prompt tile (pipeline KIND_ROWS_ASYNC).
pub const KIND_VERIFY: u32 = 7;
/// Request kind bit: the pass's residual rows arrive in rank 1's in-graph mailbox; the
/// request's rows are zeros of the same length.
pub const MAIL: u32 = 1 << 30;
/// A handoff request: rank 1 swaps the named chats' session state with rank 0 (raw words over
/// the split's TCP link) after their prompt ran as pipeline tiles, and acks without rows.
pub const KIND_HANDOFF: u32 = 8;
/// Request kind bit: a prompt tile that runs as a pipeline tile (rank 1: layers 24..48).
pub const TILE: u32 = 1 << 29;
/// Request kind bit: rank 0 runs this one-chat pass's GDN layers by heads, so rank 1 does too;
/// without it rank 1 keeps the chat's GDN states whole.
pub const HEADS: u32 = 1 << 28;
/// Request kind bit: rank 0 splits this plain pass's greedy head by vocabulary halves, so rank 1
/// projects its half of each chat's last row and sends the argmax rows (execution::head_split).
pub const HEAD: u32 = 1 << 27;
/// The first layer rank 1 runs: rank 0 keeps the frontend and layer 1's PLE (pp::Stage).
pub const SPLIT: usize = 2;

/// Request kind bit (with TILE): a token-split prompt tile (next_tp_seq.rs). Rank 1 runs it from
/// layer SPLIT and puts each layer's sequence state in the mailbox for rank 0's next pass.
pub const SEQ: u32 = 1 << 26;

/// The two-rank layer pipeline's prompt tile (rows): prompts that do not take the token split run
/// as tiles of this many rows (tp_part), as before the token split. An operator figure measured on
/// two Sparks (cold 8K: 4 x 2048 rows 2.78 s, 8 x 1024 2.89 s, 2 x 4096 3.16 s), not a checkpoint one.
pub const PIPELINE_TILE: usize = 2048;
/// The smallest first part of a token-split prompt (rows).
pub const SEQ_MIN_ROWS: usize = 1024;

/// Rows of the next prompt slice of a chat at `pos` whose prompt ends at `end`, on a two-rank
/// split front with a `chunk`-row prompt workspace. A cold prompt of 2 SEQ_MIN_ROWS..=2 chunk rows
/// comes whole in one slice, and the engine runs its two parts back to back in that one call
/// (seq_first), so no other pass, cancellation or slot reuse falls between the first part and
/// its import; every other prompt keeps PIPELINE_TILE slices.
pub fn seq_slice(pos: usize, end: usize, chunk: usize) -> usize {
    if seq_first(pos, end, chunk).is_some() { return end - pos; }
    PIPELINE_TILE.min(chunk)
}
/// The token split's first part of a prompt at `pos` ending at `end`, if it takes one: P = N / 2
/// rounded up to 64 (the measured best cut: 2.18 s at P = N / 2, 2.21 s at 0.52 N, 2.28 s at
/// 0.48 N; rank 1's later start and rank 0's longer attention cancel). None if that P passes a
/// workspace that is not a multiple of 64 rows (the prompt then keeps the pipeline tiles).
pub fn seq_first(pos: usize, end: usize, chunk: usize) -> Option<usize> {
    (pos == 0 && (2 * SEQ_MIN_ROWS..=2 * chunk).contains(&end)).then(|| end.div_ceil(2).div_ceil(64) * 64).filter(|&p| p <= chunk)
}

/// A token-split chat's stage on rank 0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SeqStage {
    /// The first part ran: rank 1 holds layers SPLIT..L of rows 0..rows and owes rank 0 their
    /// state in the mailbox (the pending export); rank 0 holds layers 0..SPLIT.
    Tiled,
    /// Rank 0 took the whole export: it holds every layer; rank 1 still needs rows..pos.
    Owed,
    /// The export was discarded before its import (another pass came first): the chat's state on
    /// rank 0 is incomplete, so its later passes fail; a new chat on the slot starts clean.
    Lost,
}
/// The current pass on this rank.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SeqMode {
    #[default]
    Off,
    /// Rank 1, first part: put each layer's state after it ran.
    Export,
    /// Rank 0, second part: take each layer's state at row `rows` before it runs.
    Import(usize),
    /// Rank 0: a prompt tail of an owed chat runs here alone (no post, no state).
    Alone,
}
/// Rank 0's decision for one pass.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SeqPlan {
    /// Neither part. First, `discard`: take the remaining messages of the pending export of that
    /// slot (a pass other than its import came first) and return their credits; then `push`
    /// (index, rank 1's position): send rank 1 the rest of these owed chats' state.
    Normal { discard: Option<u32>, push: Vec<(usize, usize)> },
    /// The first part: run layers 0..SPLIT here and post a SEQ tile.
    Tile,
    /// The second part (Import) or a prompt tail of an owed chat (Alone).
    Here,
}
/// Rank 0's token-split chats and the current pass's mode.
#[derive(Default, Debug)]
pub struct SeqChats {
    chats: Vec<(u32, usize, SeqStage)>,
    /// First parts the engine cut (slot, rows): only such a pass runs as the first part, so a long
    /// prompt's first pipeline tile never does.
    expected: Vec<(u32, usize)>,
    pub mode: SeqMode,
}
impl SeqChats {
    /// Before a pass of these chats (slot, position, span rows, prompt: no logits yet). Err: the
    /// pass names a chat whose export was discarded (its state is incomplete).
    pub fn plan(&mut self, chats: &[(u32, usize, usize, bool)], verify: bool, output: bool) -> Result<SeqPlan, String> {
        self.mode = SeqMode::Off;
        // An expectation lasts until its slot's next pass; other chats' passes keep it.
        let expected: Vec<_> = self.expected.iter().filter(|e| chats.iter().any(|c| c.0 == e.0)).copied().collect();
        self.expected.retain(|e| !chats.iter().any(|c| c.0 == e.0));
        // A new chat on a slot (position 0) drops the old chat's Owed or Lost record before the
        // tail rule reads it; a pending export (Tiled) stays, so the pass discards it below.
        self.chats.retain(|&(slot, _, stage)| stage == SeqStage::Tiled || !chats.iter().any(|&(s, pos, ..)| s == slot && pos == 0));
        if let [(slot, pos, rows, prompt)] = *chats {
            if !verify && prompt {
                match self.chats.iter().find(|c| c.0 == slot) {
                    // The stage moves to Owed only after the import (imported): a pass that
                    // fails before it leaves Tiled, and its retry imports.
                    Some(c) if c.2 == SeqStage::Tiled && pos == c.1 => { self.mode = SeqMode::Import(c.1); return Ok(SeqPlan::Here); }
                    // A tail after the second part's last 64-row boundary: rank 0 alone, so the
                    // first token comes before the push; a longer pass pushes first.
                    Some(c) if c.2 == SeqStage::Owed && rows < 64 => { self.mode = SeqMode::Alone; return Ok(SeqPlan::Here); }
                    Some(_) => (),
                    // The record follows the tile's post (posted): a pass that fails before it
                    // leaves nothing pending.
                    None if pos == 0 && !output && rows >= SEQ_MIN_ROWS && rows % 64 == 0 && expected.contains(&(slot, rows))
                        && self.pending().is_none() => return Ok(SeqPlan::Tile),
                    None => (),
                }
            }
        }
        // A pending export that this pass does not import is discarded first.
        let discard = self.lose();
        // A slot that a new chat reuses from position 0 owes and keeps nothing.
        self.chats.retain(|&(slot, ..)| !chats.iter().any(|&(s, pos, ..)| s == slot && pos == 0));
        if let Some(&(slot, ..)) = self.chats.iter().find(|c| c.2 == SeqStage::Lost && chats.iter().any(|x| x.0 == c.0)) {
            return Err(format!("token split: slot {slot} lost its first part's state before the import; reset the chat"));
        }
        let push = chats.iter().enumerate()
            .filter_map(|(i, c)| self.chats.iter().find(|r| r.0 == c.0 && r.2 == SeqStage::Owed).map(|r| (i, r.1))).collect();
        Ok(SeqPlan::Normal { discard, push })
    }
    /// The engine cuts a first part of `rows` rows for this slot next (seq_first).
    pub fn expect(&mut self, slot: u32, rows: usize) {
        self.expected.retain(|e| e.0 != slot);
        self.expected.push((slot, rows));
    }
    /// Rank 0 posted this slot's first part of `rows` rows: rank 1 now runs it and owes the export.
    pub fn posted(&mut self, slot: u32, rows: usize) {
        self.chats.retain(|c| c.0 != slot);
        self.chats.push((slot, rows, SeqStage::Tiled));
    }
    /// The slot whose export is pending (at most one: a first part starts only without one).
    pub fn pending(&self) -> Option<u32> {
        self.chats.iter().find(|c| c.2 == SeqStage::Tiled).map(|c| c.0)
    }
    /// Drop the pending export (the engine takes the rest of it): its chat's later passes fail.
    pub fn lose(&mut self) -> Option<u32> {
        let slot = self.pending()?;
        for c in self.chats.iter_mut().filter(|c| c.0 == slot) { c.2 = SeqStage::Lost; }
        Some(slot)
    }
    /// Rank 0 queued the takes of this slot's whole export: it now owes rank 1 the push.
    pub fn imported(&mut self, slot: u32) {
        for c in self.chats.iter_mut().filter(|c| c.0 == slot && c.2 == SeqStage::Tiled) { c.2 = SeqStage::Owed; }
    }
    /// These slots' state is whole on both ranks again.
    pub fn settled(&mut self, slots: &[u32]) {
        self.chats.retain(|c| !slots.contains(&c.0));
    }
}

/// The multi-Spark path flash-serve chose at startup and why, for the receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Path {
    pub name: &'static str,
    pub reason: &'static str,
}
impl Path {
    pub const TENSOR_SPLIT: Self = Self { name: "tensor_split", reason: "FP4-Local on two Sparks" };
    pub const LAYER_PIPELINE: Self = Self { name: "layer_pipeline", reason: "FP4-Local on three or more Sparks" };
    pub const RANK_PATH: Self = Self { name: "rank_path", reason: "FP4-Local with --tensor-map" };
    pub const RING: Self = Self { name: "ring", reason: "FP8: the tensor split does not serve FP8" };
}
static PATH: std::sync::OnceLock<Option<Path>> = std::sync::OnceLock::new();
/// flash-serve selects the path once at startup, before anything reads it (None on one Spark).
/// The tensor split is on for the FP4-Local service at world 2 only. A second selection is refused.
pub fn select(path: Option<Path>) -> Result<(), String> {
    PATH.set(path).map_err(|_| "the multi-Spark path is already selected".to_string())
}
/// The selected path (None until flash-serve selects one, and on one Spark).
pub fn path() -> Option<Path> {
    PATH.get().copied().flatten()
}
/// Whether this process serves the tensor split.
pub fn on() -> bool {
    path() == Some(Path::TENSOR_SPLIT)
}
/// A plain pass of the split whose chats each decode one token runs as captured segments, as a
/// verify pass does (flash::next_execution): the same arithmetic and mailbox order as the eager
/// pass (a capture may add an empty QSA pool launch, which does no tensor work), the host
/// capturing each segment while the GPU runs the one before. A first-seen shape, an unfunded
/// segment or a pass whose rows cross with the request stays eager, and so does a pass a decode
/// profile reads back from (`readback`: its probes synchronize, which an open capture refuses).
/// `lens`: each chat's rows.
pub fn plain_segments(split: bool, verify: bool, readback: bool, lens: impl IntoIterator<Item = usize>) -> bool {
    split && !verify && !readback && lens.into_iter().all(|n| n == 1)
}

/// The split's geometry, derived from the checkpoint: residual width (HC branches x hidden),
/// hidden width, routed-expert width (gate/up rows, down K), routed experts per token and the
/// expert count. Rank r owns half of the expert width and half of the hidden width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpLayout {
    pub hidden: usize,
    pub residual: usize,
    pub expert_width: usize,
    pub top_k: usize,
    pub experts: usize,
}
impl TpLayout {
    /// Rows a verify pass may have for its experts to split: the served cohort tick's widest pass
    /// (VADE_VERIFY_ROWS, 32 rows) stays split. A mailbox slot holds this many residual rows.
    pub const ROWS: usize = 32;
    /// Rows any other pass (decode, short prompt tiles) may have to split: above 16 rows a prompt
    /// pass runs the prompt expert kernels, which compute whole columns (Workspace::moe_at).
    pub const PROMPT_ROWS: usize = 16;
    /// The split bound of a pass of this kind.
    pub fn split_rows(verify: bool) -> usize { if verify { Self::ROWS } else { Self::PROMPT_ROWS } }
    pub fn new(hidden: u64, branches: u64, expert_width: u64, top_k: u64, experts: u64) -> Result<Self, String> {
        let n = |v: u64| usize::try_from(v).map_err(|_| "tensor split geometry overflow".to_string());
        let layout = Self { hidden: n(hidden)?, residual: n(hidden.checked_mul(branches).ok_or("tensor split geometry overflow")?)?,
            expert_width: n(expert_width)?, top_k: n(top_k)?, experts: n(experts)? };
        // Halves of whole K16 tiles and BF16 pairs; the finish covers 256 columns per CTA.
        if layout.hidden == 0 || !layout.hidden.is_multiple_of(32) || layout.expert_width == 0 || !layout.expert_width.is_multiple_of(32)
            || layout.top_k == 0 || layout.experts == 0 || branches == 0 {
            return Err(format!("tensor split needs even 16-column halves: hidden {hidden}, expert width {expert_width}, top-k {top_k}"));
        }
        Ok(layout)
    }
    /// Gate/up output columns and down output columns per rank.
    pub fn mid_cols(&self) -> usize { self.expert_width / 2 }
    pub fn out_cols(&self) -> usize { self.hidden / 2 }
    /// One BF16 middle row in 32-bit words (BF16 pairs).
    pub fn mid_words(&self) -> usize { self.expert_width / 2 }
    /// Gate output (BF16 pairs): rank r writes middle columns from mid_cols * r.
    pub fn mid(&self, rank: usize) -> u64 { (rank * self.mid_cols() * 2) as u64 }
    /// Down output and the finish's partials: rank r's columns from out_cols * r of each BF16
    /// slot row (a slot row is `hidden` FP32-sized words, its BF16 half the first hidden / 2).
    /// Rank r also writes the FP32 shared row this many bytes lower, so the shifted finish
    /// reads shared column c + out_cols for its column c (inside the shared slot's stride and
    /// the previous slot's unused upper half).
    pub fn routed(&self, rank: usize) -> u64 { (rank * self.out_cols() * 2) as u64 }
    /// Finish output: rank r's columns from out_cols * r of each FP32 MoE output row.
    pub fn out(&self, rank: usize) -> u64 { (rank * self.out_cols() * 4) as u64 }
    /// A world-one launch of `whole` CTAs over `full` output columns, scaled to `own` columns.
    pub fn share(&self, whole: u32, own: usize, full: usize) -> u32 { (whole as usize * own).div_ceil(full).max(1) as u32 }
    /// Finish CTAs (256 columns each) over one rank's columns.
    pub fn finish_ctas(&self) -> u32 { self.out_cols().div_ceil(256) as u32 }
    /// Rank 1's descriptor shift: gate/up start at row mid_cols (K = hidden), down at row
    /// out_cols (K = expert_width); FP4 nibbles take k / 2 bytes per row, E4M3 scales k / 16.
    pub fn gate_shift(&self) -> (usize, usize) { (self.mid_cols() * self.hidden / 2, self.mid_cols() * (self.hidden / 16)) }
    pub fn down_shift(&self) -> (usize, usize) { (self.out_cols() * self.expert_width / 2, self.out_cols() * (self.expert_width / 16)) }
    /// The largest mailbox message: ROWS rows of the residual (the layer-2 post).
    pub fn slot_bytes(&self) -> usize { Self::ROWS * self.residual * 4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// flash_next_k16_finish's addressing (slot rows of `hidden` FP32-sized words, routed BF16
    /// at u16 index row*2*(top_k+1)*hidden + rank*2*hidden + col, shared FP32 at word
    /// (row*(top_k+1)+top_k)*hidden + col, output at row*hidden + col) and fp4_bw_down's
    /// (u32 word slot*hidden + col/2) with rank r's offsets: the finish reads exactly the
    /// columns rank r wrote, and rank 1's shifted shared row stays clear of routed columns.
    /// Run for the shipped checkpoint geometry and a different one, so nothing is fixed.
    #[test]
    fn rank_offsets_address_each_ranks_columns_for_any_geometry() {
        for layout in [TpLayout::new(2560, 4, 640, 10, 512).unwrap(), TpLayout::new(4096, 2, 1536, 8, 128).unwrap()] {
            let (h, k) = (layout.hidden, layout.top_k);
            let slots = k + 1;
            for rank in 0..2 {
                let (routed_u16, routed_words, out_words) = (layout.routed(rank) as usize / 2, layout.routed(rank) as usize / 4, layout.out(rank) as usize / 4);
                assert_eq!(layout.routed(rank) % 4, 0);
                for row in [0usize, 3, TpLayout::ROWS - 1] {
                    for slot in 0..k {
                        for c in [0usize, 1, 255, layout.out_cols() - 2] {
                            let pair = ((row * slots + slot) * h + c / 2) * 2 + c % 2 + routed_u16;
                            assert_eq!(pair, (row * slots + slot) * 2 * h + rank * layout.out_cols() + c);
                            assert_eq!(routed_u16 + row * 2 * slots * h + slot * 2 * h + c, pair);
                            assert!(rank * layout.out_cols() + c < h, "inside the slot's BF16 half");
                        }
                    }
                    for c in [0usize, 7, layout.out_cols() - 1] {
                        let read = routed_words + row * slots * h + c + k * h;
                        assert_eq!(read + routed_words - (row * slots + k) * h, rank * layout.out_cols() + c);
                        assert_eq!(out_words + row * h + c, row * h + rank * layout.out_cols() + c);
                    }
                    let first = (row * slots + k) * h - routed_words;
                    assert!(rank == 0 || first >= (row * slots + k - 1) * h + h / 2);
                }
                assert_eq!(layout.mid(rank) as usize / 4, rank * layout.mid_cols() / 2);
            }
            assert!(layout.finish_ctas() as usize * 256 >= layout.out_cols());
            assert_eq!(layout.share(40, layout.mid_cols(), layout.expert_width), 20);
            assert_eq!(layout.share(160, layout.out_cols(), layout.hidden), 80);
            assert_eq!(layout.share(5, 1, 3), 2);
            assert_eq!(layout.gate_shift().0, layout.mid_cols() * h / 2);
            assert_eq!(layout.down_shift().1, layout.out_cols() * layout.expert_width / 16);
            assert_eq!(layout.slot_bytes(), TpLayout::ROWS * layout.residual * 4);
        }
        assert!(TpLayout::new(2560, 4, 650, 10, 512).is_err());
        // Prompt passes split only below the prompt expert kernels' threshold; verify passes up to the mailbox.
        assert_eq!((TpLayout::split_rows(true), TpLayout::split_rows(false)), (TpLayout::ROWS, TpLayout::PROMPT_ROWS));
        assert!(TpLayout::PROMPT_ROWS <= 16 && TpLayout::PROMPT_ROWS <= TpLayout::ROWS);
        assert!(TpLayout::new(2570, 4, 640, 10, 512).is_err());
        assert!(TpLayout::new(2560, 0, 640, 10, 512).is_err());
    }
    /// Each request kind bit is its own bit, above the kind numbers and below the pipeline's
    /// release bit (1 << 31).
    #[test]
    fn request_kind_bits_do_not_overlap() {
        let bits = [MAIL, TILE, HEADS, HEAD, SEQ];
        for (i, a) in bits.iter().enumerate() {
            assert!(a.is_power_of_two() && *a > 0xff && *a < 1 << 31);
            for b in &bits[i + 1..] { assert_eq!(a & b, 0); }
        }
    }
    /// A cold 2K..2 x workspace prompt comes whole (one call runs both parts; the first a multiple
    /// of 64, at least half), every other prompt in pipeline tiles; for other workspaces too.
    #[test]
    fn token_split_slices_follow_the_prompt_and_the_workspace() {
        assert_eq!(seq_slice(0, 8192, 4096), 8192);
        assert_eq!(seq_slice(0, 8009, 4096), 8009);
        assert_eq!(seq_first(0, 8009, 4096), Some(4032));
        assert_eq!(seq_slice(0, 2048, 4096), 2048);
        assert_eq!(seq_first(0, 2048, 4096), Some(1024));
        assert_eq!(seq_slice(0, 2047, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice(0, 8193, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice(0, 32768, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice(4096, 12288, 4096), PIPELINE_TILE);
        assert_eq!(seq_first(0, 16384, 8192), Some(8192));
        assert_eq!(seq_slice(0, 3000, 1024), 1024);
        // A workspace that is not a multiple of 64: a cut past it keeps the pipeline tiles.
        assert_eq!(seq_first(0, 2200, 1100), None);
        assert_eq!(seq_slice(0, 2200, 1100), 1100);
        assert_eq!(seq_first(0, 2176, 1100), Some(1088));
        for end in (2048..=8192).step_by(37) {
            let p = seq_first(0, end, 4096).unwrap();
            assert!(p % 64 == 0 && 2 * p >= end && p < end && p <= 4096 && end - p <= 4096, "{end}");
        }
        for chunk in [1030, 1100, 2047, 3000, 4095] {
            for end in 2 * SEQ_MIN_ROWS..=2 * chunk {
                if let Some(p) = seq_first(0, end, chunk) { assert!(p % 64 == 0 && p <= chunk && end - p <= chunk, "{chunk} {end}"); }
            }
        }
    }
    /// First part, second part, a tail alone, then the push before decode; only an expected first
    /// part splits; a slot reused from position 0 drops its state; short or final parts stay normal.
    #[test]
    fn token_split_plans_each_pass() {
        let mut seq = SeqChats::default();
        let normal = |discard: Option<u32>, push: Vec<(usize, usize)>| Ok(SeqPlan::Normal { discard, push });
        assert_eq!(seq.plan(&[(1, 0, 4096, true)], false, true), normal(None, vec![]));
        assert_eq!(seq.plan(&[(1, 0, 4032, true)], false, false), normal(None, vec![]));
        seq.expect(1, seq_first(0, 8009, 4096).unwrap());
        assert_eq!(seq.plan(&[(1, 0, 4032, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(1, 4032);
        assert_eq!(seq.pending(), Some(1));
        assert_eq!(seq.plan(&[(1, 4032, 3968, true)], false, false), Ok(SeqPlan::Here));
        assert_eq!(seq.mode, SeqMode::Import(4032));
        seq.imported(1);
        assert_eq!(seq.pending(), None);
        assert_eq!(seq.plan(&[(1, 8000, 9, true)], false, true), Ok(SeqPlan::Here));
        assert_eq!(seq.mode, SeqMode::Alone);
        assert_eq!(seq.plan(&[(2, 7, 1, false), (1, 8009, 1, false)], true, true), normal(None, vec![(1, 4032)]));
        assert_eq!(seq.mode, SeqMode::Off);
        seq.settled(&[1]);
        assert_eq!(seq.plan(&[(1, 8010, 1, false)], true, true), normal(None, vec![]));
        // A longer later pass of an owed chat pushes first; a new chat on an owed slot owes nothing.
        seq.expect(6, 2048);
        assert_eq!(seq.plan(&[(6, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(6, 2048);
        assert_eq!(seq.plan(&[(6, 2048, 2048, true)], false, false), Ok(SeqPlan::Here));
        seq.imported(6);
        assert_eq!(seq.plan(&[(6, 4096, 2048, true)], false, false), normal(None, vec![(0, 2048)]));
        seq.expect(7, 2048);
        assert_eq!(seq.plan(&[(7, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(7, 2048);
        assert_eq!(seq.plan(&[(7, 2048, 2048, true)], false, false), Ok(SeqPlan::Here));
        seq.imported(7);
        assert_eq!(seq.plan(&[(7, 0, 100, true)], false, true), normal(None, vec![]));
        assert_eq!(seq.plan(&[(7, 100, 1, false)], true, true), normal(None, vec![]));
    }
    /// An admission failure after the plan leaves the first part pending: a retry of the same
    /// pass imports again (no push, no tail alone without the import).
    #[test]
    fn token_split_retry_after_a_failed_pass_imports_again() {
        let mut seq = SeqChats::default();
        // A first part that fails before its post leaves nothing pending; the retry cuts again.
        seq.expect(1, 2048);
        assert_eq!(seq.plan(&[(1, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        assert_eq!(seq.pending(), None);
        seq.expect(1, 2048);
        assert_eq!(seq.plan(&[(1, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(1, 2048);
        for _ in 0..3 {
            assert_eq!(seq.plan(&[(1, 2048, 1984, true)], false, false), Ok(SeqPlan::Here));
            assert_eq!(seq.mode, SeqMode::Import(2048));
            assert_eq!(seq.pending(), Some(1));
        }
        seq.imported(1);
        assert_eq!(seq.plan(&[(1, 4032, 16, true)], false, true), Ok(SeqPlan::Here));
        assert_eq!(seq.mode, SeqMode::Alone);
        // The engine drops the export of a failed second part at once (tp_seq_abandon): the
        // retry fails instead of running without it, and a new chat on the slot starts clean.
        seq.expect(2, 2048);
        assert_eq!(seq.plan(&[(2, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(2, 2048);
        assert_eq!(seq.plan(&[(2, 2048, 1984, true)], false, false), Ok(SeqPlan::Here));
        assert_eq!(seq.lose(), Some(2));
        assert_eq!(seq.lose(), None);
        assert!(seq.plan(&[(2, 2048, 1984, true)], false, false).is_err());
        assert!(seq.plan(&[(2, 2048, 16, true)], false, true).is_err());
        assert_eq!(seq.plan(&[(2, 0, 2048, true)], false, false), normal_discard(None));
    }
    /// A pass other than the import while a first part is pending (another chat, a second first
    /// part, a verify pass, a new chat on the slot) discards the export first; the chat whose
    /// export was discarded fails on its next pass; a new chat on its slot starts clean.
    #[test]
    fn token_split_discards_a_pending_export_before_other_work() {
        let mut seq = SeqChats::default();
        seq.expect(1, 4096);
        assert_eq!(seq.plan(&[(1, 0, 4096, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(1, 4096);
        // Two queued first parts: the second does not split while one is pending.
        seq.expect(2, 4096);
        assert_eq!(seq.plan(&[(2, 0, 4096, true)], false, false), normal_discard(Some(1)));
        assert_eq!(seq.pending(), None);
        assert!(seq.plan(&[(1, 4096, 4096, true)], false, false).is_err());
        assert!(seq.plan(&[(3, 9, 1, false), (1, 4096, 1, false)], true, true).is_err());
        assert_eq!(seq.plan(&[(1, 0, 64, true)], false, true), normal_discard(None));
        assert!(seq.plan(&[(1, 64, 1, false)], true, true).is_ok());
        // Slot reuse from position 0 while pending: discard, then the new chat runs clean.
        seq.expect(4, 2048);
        assert_eq!(seq.plan(&[(4, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(4, 2048);
        assert_eq!(seq.plan(&[(4, 0, 100, true)], false, true), normal_discard(Some(4)));
        assert_eq!(seq.plan(&[(4, 100, 1, false)], true, true), normal_discard(None));
        // A verify pass of another chat while pending: discard.
        seq.expect(5, 2048);
        assert_eq!(seq.plan(&[(5, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(5, 2048);
        assert_eq!(seq.plan(&[(8, 50, 4, false)], true, true), normal_discard(Some(5)));
        assert_eq!(seq.pending(), None);
    }
    /// A token-split chat that ends owed (one output token, or a cancel after its prompt) leaves its
    /// slot: a new chat of 1 to 63 prompt rows on that slot runs normal, owes nothing and pushes
    /// nothing at decode; a long new chat on the slot takes the token split again.
    #[test]
    fn token_split_short_prompt_on_a_reused_owed_slot_starts_clean() {
        let mut seq = SeqChats::default();
        for rows in 1..64 {
            seq.expect(1, 4096);
            assert_eq!(seq.plan(&[(1, 0, 4096, true)], false, false), Ok(SeqPlan::Tile));
            seq.posted(1, 4096);
            assert_eq!(seq.plan(&[(1, 4096, 4096, true)], false, true), Ok(SeqPlan::Here));
            seq.imported(1);
            assert_eq!(seq.plan(&[(1, 0, rows, true)], false, true), normal_discard(None), "{rows}");
            assert_eq!(seq.mode, SeqMode::Off);
            assert_eq!(seq.plan(&[(1, rows, 1, false)], false, true), normal_discard(None), "{rows}");
            assert_eq!(seq.plan(&[(2, 9, 1, false), (1, rows + 1, 4, false)], true, true), normal_discard(None), "{rows}");
        }
        // A lost chat's slot too: the new short chat runs clean instead of failing.
        seq.expect(3, 2048);
        assert_eq!(seq.plan(&[(3, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(3, 2048);
        assert_eq!(seq.lose(), Some(3));
        assert_eq!(seq.plan(&[(3, 0, 16, true)], false, true), normal_discard(None));
        assert_eq!(seq.plan(&[(3, 16, 1, false)], false, true), normal_discard(None));
        // A long new chat on an owed slot splits again.
        seq.expect(4, 2048);
        assert_eq!(seq.plan(&[(4, 0, 2048, true)], false, false), Ok(SeqPlan::Tile));
        seq.posted(4, 2048);
        assert_eq!(seq.plan(&[(4, 2048, 2048, true)], false, true), Ok(SeqPlan::Here));
        seq.imported(4);
        seq.expect(4, 1024);
        assert_eq!(seq.plan(&[(4, 0, 1024, true)], false, false), Ok(SeqPlan::Tile));
    }
    fn normal_discard(discard: Option<u32>) -> Result<SeqPlan, String> { Ok(SeqPlan::Normal { discard, push: vec![] }) }
    /// Only the split's plain passes whose chats each decode one token run as segments.
    #[test]
    fn plain_segments_take_one_token_split_passes_only() {
        assert!(plain_segments(true, false, false, [1]));
        assert!(plain_segments(true, false, false, [1, 1, 1]));
        assert!(!plain_segments(false, false, false, [1]));
        assert!(!plain_segments(true, true, false, [1]));
        assert!(!plain_segments(true, false, true, [1]), "a decode profile reading back keeps the pass eager");
        assert!(!plain_segments(true, false, false, [1, 2]));
        assert!(!plain_segments(true, false, false, [TpLayout::PROMPT_ROWS]));
    }
}
