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
/// Request kind bit: rank 0 splits this plain pass's greedy head by vocabulary parts, so ranks 1..
/// project their part of each chat's last row and send the argmax rows (execution::head_split).
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
/// The shortest prompt, in prompt workspaces, whose middle slices take the whole workspace
/// (seq_slice). On the two-rank pipeline the first token comes after one rank's time for every row
/// plus its time for the largest tile (the fill and the drain), so a wide tile adds a PIPELINE_TILE
/// of bubble that the experts' saving pays only on a long prompt (two Sparks, every middle slice
/// wide: 12K +0.36 s, 32K -0.15 s, 64K -1.06 s, 128K -3.0 s; the crossing is near 26K rows, so
/// 7 workspaces of 4096 rows take a 32K prompt of 32099 rows). An operator figure.
pub const WIDE_MIN_CHUNKS: usize = 7;

/// Rows of the next prompt slice of a chat at `pos` whose prompt ends at `end`, on a two-rank
/// split front with a `chunk`-row prompt workspace. A cold prompt of 2 SEQ_MIN_ROWS..=2 chunk rows
/// comes whole in one slice, and the engine runs its two parts back to back in that one call
/// (seq_first), so no other pass, cancellation or slot reuse falls between the first part and
/// its import. Other prompts run as pipeline tiles: in a prompt of WIDE_MIN_CHUNKS workspaces or
/// more, a slice after a chat's first takes the whole workspace while more than chunk +
/// PIPELINE_TILE rows remain (the routed experts see twice the rows per expert in a 4096-row
/// tile, at about 0.6 of the cost per row); the first slice keeps the pipeline's fill short and
/// the last rows its drain. Every other slice, and every slice of a workspace of at most
/// PIPELINE_TILE rows, stays PIPELINE_TILE.min(chunk) rows. A prompt the layer-major transaction
/// takes (split_rows) comes whole too.
pub fn seq_slice(pos: usize, end: usize, chunk: usize) -> usize { seq_slice_if(long_split(), pos, end, chunk) }
fn seq_slice_if(on: bool, pos: usize, end: usize, chunk: usize) -> usize {
    if seq_first(pos, end, chunk).is_some() || split_rows_if(on, pos, end, chunk).is_some() { return end - pos; }
    if pos > 0 && end >= WIDE_MIN_CHUNKS * chunk && end - pos > chunk + PIPELINE_TILE { return chunk; }
    PIPELINE_TILE.min(chunk)
}
/// The token split's first part of a prompt at `pos` ending at `end`, if it takes one: P = N / 2
/// rounded up to 64 (the measured best cut: 2.18 s at P = N / 2, 2.21 s at 0.52 N, 2.28 s at
/// 0.48 N; rank 1's later start and rank 0's longer attention cancel). None if that P passes a
/// workspace that is not a multiple of 64 rows (the prompt then keeps the pipeline tiles).
pub fn seq_first(pos: usize, end: usize, chunk: usize) -> Option<usize> {
    (pos == 0 && (2 * SEQ_MIN_ROWS..=2 * chunk).contains(&end)).then(|| end.div_ceil(2).div_ceil(64) * 64).filter(|&p| p <= chunk)
}

/// KIND_HANDOFF length field of SPLIT_PROMPT plus a row count (below SPLIT_PROMPT): on the pair
/// rank 1 runs the layer-major prompt transaction's worker for that many rows (flash/next_split.rs);
/// over more ranks each back runs its steps of the sequence split (flash/next_xsplit.rs).
pub const SPLIT_PROMPT: u32 = 1 << 25;
/// VADE_LONG_SPLIT (on unless 0): a cold prompt whose transaction has up to LONG_SPLIT_MAX rows
/// takes the layer-major transaction (B8 round 2).
pub fn long_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_LONG_SPLIT").map_or(true, |v| v != "0"))
}
/// The most rows the transaction takes (prompts up to LONG_SPLIT_MAX + 64 tokens): the longest
/// measured transaction whose first token still wins at least 1 % (20,416 rows of a 20,417-token
/// prompt; 8K to 20K win 1 to 5 %, 24K and 32K lose 1 to 4 %). Longer prompts keep the pipeline
/// path unchanged.
pub const LONG_SPLIT_MAX: usize = 20_416;
const _: () = assert!(LONG_SPLIT_MAX % 64 == 0 && LONG_SPLIT_MAX < SPLIT_PROMPT as usize);
/// The transaction's cut as a share of its rows, in permille (the measured best).
pub const LONG_SPLIT_CUT: usize = 500;
/// `VADE_LONG_SPLIT_CUT=<permille>`: a screen setting that places the cut at that share instead.
pub fn long_split_cut() -> usize {
    static CUT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CUT.get_or_init(|| std::env::var("VADE_LONG_SPLIT_CUT").ok().and_then(|v| v.parse().ok()).filter(|&p| (1..1000).contains(&p)).unwrap_or(LONG_SPLIT_CUT))
}
/// Whether a session's prompt call may take the layer-major transaction: the call gives output,
/// the session is valid at position 0 and owns its state, and no debug dump or layer check is on.
/// A failed session stays refused until reset: it keeps the usual path, which refuses it.
pub fn split_session(output: bool, valid: bool, pos: usize, diagnostics: bool, owns_state: bool) -> bool {
    output && valid && pos == 0 && !diagnostics && owns_state
}
/// Rows of a cold prompt at `pos` ending at `end` that the layer-major transaction runs (tiles
/// of PIPELINE_TILE rows on both ranks), if it takes the prompt: past the bounded token split
/// (more than 2 chunk rows), at least four tiles and at most LONG_SPLIT_MAX rows; it stops at
/// the last 64-row edge before `end`. A tail of 1 to 63 rows then gives the first token on rank 0
/// alone (as after the token split) after the prefix-cache checkpoint; a prompt that ends on a
/// 64-row edge leaves a 64-row tail, which pushes the owed state first.
pub fn split_rows(pos: usize, end: usize, chunk: usize) -> Option<usize> { split_rows_if(long_split(), pos, end, chunk) }
fn split_rows_if(on: bool, pos: usize, end: usize, chunk: usize) -> Option<usize> {
    let rows = end.checked_sub(1)? / 64 * 64;
    (on && pos == 0 && end > 2 * chunk && rows >= 4 * PIPELINE_TILE && rows <= LONG_SPLIT_MAX).then_some(rows)
}

/// The most rows of a prompt's sequence split over more than two ranks: a part of 13,000 rows
/// still sends each attention vector in one message of the mailbox's slots.
pub const XSPLIT_MAX: usize = 48_000;
/// The fewest rows of that split: below it the parts are short and the routed experts see few rows.
pub const XSPLIT_MIN: usize = 3 * PIPELINE_TILE;
/// Rows of a cold prompt at `pos` ending at `end` that the sequence split over more than two ranks
/// runs (flash/next_xsplit_plan.rs), if it takes the prompt: XSPLIT_MIN to XSPLIT_MAX rows; it
/// stops at the last 64-row edge before `end`, as split_rows.
pub fn xsplit_rows(pos: usize, end: usize, chunk: usize) -> Option<usize> {
    let rows = end.checked_sub(1)? / 64 * 64;
    (pos == 0 && chunk >= PIPELINE_TILE && (XSPLIT_MIN..=XSPLIT_MAX).contains(&rows)).then_some(rows)
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
    /// These slots' state is whole on every rank again.
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
    /// The split over more than two ranks (TpLayout::for_config cuts every width, RDMA links).
    pub const TENSOR_PARTS: Self = Self { name: "tensor_split", reason: "FP4-Local on more than two Sparks whose widths cut into whole parts" };
    pub const LAYER_PIPELINE: Self = Self { name: "layer_pipeline", reason: "FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take)" };
    pub const RANK_PATH: Self = Self { name: "rank_path", reason: "FP4-Local with --tensor-map" };
    pub const RING: Self = Self { name: "ring", reason: "FP8: the tensor split does not serve FP8" };
}
static PATH: std::sync::OnceLock<Option<Path>> = std::sync::OnceLock::new();
/// flash-serve selects the path once at startup, before anything reads it (None on one Spark).
/// The tensor split is on for the FP4-Local service on the pair, and over more ranks over RDMA
/// when TpLayout::for_config cuts every width. A second selection is refused.
pub fn select(path: Option<Path>) -> Result<(), String> {
    PATH.set(path).map_err(|_| "the multi-Spark path is already selected".to_string())
}
/// The ranks of the pair: every split level runs there (routed experts, projections, GDN heads,
/// the head and token splits, prompt tiles and their handoffs, the prefix cache). On more ranks
/// the split cuts the routed and shared experts, the projections and the greedy head into parts;
/// long prompts run as chain tiles and a state gather (next_tp_gather), or cold ones of XSPLIT_MIN
/// to XSPLIT_MAX rows, when no other chat decodes, as a sequence split (next_xsplit_plan), and
/// every rank runs the GDN heads whole, with no token split. Each rank keeps its own prefix cache copies.
pub const PAIR: usize = 2;
static WORLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// flash-serve sets the split's ranks once at startup, with the path (unset: the pair).
pub fn select_world(ranks: usize) -> Result<(), String> {
    WORLD.set(ranks).map_err(|_| "the tensor split's ranks are already selected".to_string())
}
/// The split's ranks.
pub fn world() -> usize {
    WORLD.get().copied().unwrap_or(PAIR)
}
static RANK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// flash-serve sets this process's rank of the split once at startup, with the ranks (the head
/// split's vocabulary part, flash_head_options::bounds_rows).
pub fn select_rank(rank: usize) -> Result<(), String> {
    RANK.set(rank).map_err(|_| "the tensor split's rank is already selected".to_string())
}
/// This process's rank of the split (None: not selected).
pub fn rank() -> Option<usize> {
    RANK.get().copied()
}
/// This process serves the tensor split on the pair (every level), not on more ranks.
pub fn pair() -> bool {
    on() && world() == PAIR
}
/// The selected path (None until flash-serve selects one, and on one Spark).
pub fn path() -> Option<Path> {
    PATH.get().copied().flatten()
}
/// Whether this process serves the tensor split.
pub fn on() -> bool {
    matches!(path(), Some(p) if p == Path::TENSOR_SPLIT || p == Path::TENSOR_PARTS)
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
/// expert count, over `world` ranks. Rank r owns part r of `world` equal parts of the expert
/// width and of the hidden width (`part`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpLayout {
    pub hidden: usize,
    pub residual: usize,
    pub expert_width: usize,
    pub top_k: usize,
    pub experts: usize,
    pub world: usize,
}
impl TpLayout {
    /// Rows a verify pass may have for its experts to split: the served cohort tick's widest pass
    /// (VADE_VERIFY_ROWS, 32 rows) stays split. A mailbox slot holds this many residual rows.
    pub const ROWS: usize = 32;
    /// Rows any other pass (decode, short prompt tiles) may have to split: above 16 rows a prompt
    /// pass runs the prompt expert kernels, which compute whole columns (Workspace::moe_at).
    pub const PROMPT_ROWS: usize = 16;
    /// The packed BF16 weights' row tile: a rank's output rows start and end on it.
    pub const ROW_TILE: usize = 64;
    /// The split bound of a pass of this kind.
    pub fn split_rows(verify: bool) -> usize { if verify { Self::ROWS } else { Self::PROMPT_ROWS } }
    /// The expert and hidden geometry over `world` ranks, refused unless the hidden and expert
    /// widths cut into whole 16-column K16 tiles and BF16 pairs and each hidden part is whole
    /// ROW_TILE rows. The other matrices the split cuts are checked by for_config.
    pub fn new(hidden: u64, branches: u64, expert_width: u64, top_k: u64, experts: u64, world: usize) -> Result<Self, String> {
        let n = |v: u64| usize::try_from(v).map_err(|_| "tensor split geometry overflow".to_string());
        let layout = Self { hidden: n(hidden)?, residual: n(hidden.checked_mul(branches).ok_or("tensor split geometry overflow")?)?,
            expert_width: n(expert_width)?, top_k: n(top_k)?, experts: n(experts)?, world };
        // Parts of whole K16 tiles and BF16 pairs; each rank's output rows are whole 64-row tiles
        // of the packed BF16 shared weights (shared_rows) and whole finish CTAs.
        let edge = 16usize.checked_mul(world).filter(|&e| e > 0).ok_or("tensor split needs at least one rank")?;
        if layout.hidden == 0 || !layout.hidden.is_multiple_of(edge) || layout.expert_width == 0 || !layout.expert_width.is_multiple_of(edge)
            || layout.top_k == 0 || layout.experts == 0 || branches == 0 {
            return Err(format!("tensor split needs {world} whole 16-column parts: hidden {hidden}, expert width {expert_width}, top-k {top_k}"));
        }
        if !layout.out_cols().is_multiple_of(Self::ROW_TILE) {
            return Err(format!("tensor split needs {world} whole {}-row parts of hidden {hidden}", Self::ROW_TILE));
        }
        Ok(layout)
    }
    /// The layout of this checkpoint over `world` ranks: TpLayout::new, and every other matrix the
    /// split cuts by rows has whole ROW_TILE-row parts: the GDN qkv and z in-projections and the
    /// QSA q/gate rows, which also hold whole query heads per part (next_tp::tp_half_twins,
    /// Qsa::project_batch_part). A world that does not cut every one of them has no layout.
    /// Startup (settings::split_accepts) and NextModel::set_tp both use it.
    pub fn for_config(c: &engine_model::flash::Config, world: usize) -> Result<Self, String> {
        let layout = Self::new(c.hidden, c.branches, c.expert_width, c.top_k, c.experts, world)?;
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(|| "tensor split geometry overflow".to_string());
        let value = mul(c.linear_value_heads, c.linear_value_dim)?;
        let qkv = mul(2, mul(c.linear_key_heads, c.linear_key_dim)?)?.checked_add(value).ok_or("tensor split geometry overflow")?;
        Self::row_parts(world, [("GDN qkv", qkv), ("GDN z", value), ("QSA q/gate", mul(2, mul(c.query_heads, c.head_dim)?)?)], c.query_heads)?;
        Ok(layout)
    }
    /// Each named row count cuts into `world` whole ROW_TILE-row parts, and the query heads into
    /// `world` whole parts.
    fn row_parts<const N: usize>(world: usize, rows: [(&str, u64); N], query_heads: u64) -> Result<(), String> {
        let tile = u64::try_from(Self::ROW_TILE * world).map_err(|_| "tensor split geometry overflow".to_string())?;
        for (name, n) in rows {
            if n == 0 || !n.is_multiple_of(tile) {
                return Err(format!("tensor split needs {world} whole {}-row parts of the {name} rows ({n})", Self::ROW_TILE));
            }
        }
        if query_heads == 0 || !query_heads.is_multiple_of(world as u64) {
            return Err(format!("tensor split needs {world} whole parts of {query_heads} query heads"));
        }
        Ok(())
    }
    /// Rank r's part of `n` columns (or rows): [n r / world, n (r + 1) / world).
    pub fn part(&self, n: usize, rank: usize) -> std::ops::Range<usize> { n * rank / self.world..n * (rank + 1) / self.world }
    /// Gate/up output columns and down output columns per rank.
    pub fn mid_cols(&self) -> usize { self.expert_width / self.world }
    pub fn out_cols(&self) -> usize { self.hidden / self.world }
    /// One whole BF16 middle row in 32-bit words (BF16 pairs).
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
    /// Finish threads per CTA: the widest of 256, 128 and 64 that divides the rank's columns, so
    /// the CTAs cover them exactly (the finish kernel bounds only the whole row).
    pub fn finish_threads(&self) -> u32 { [256, 128, Self::ROW_TILE].into_iter().find(|&t| self.out_cols().is_multiple_of(t)).unwrap_or(Self::ROW_TILE) as u32 }
    /// Finish CTAs over one rank's columns.
    pub fn finish_ctas(&self) -> u32 { (self.out_cols() / self.finish_threads() as usize) as u32 }
    /// Rank r's descriptor shift: gate/up start at row r mid_cols (K = hidden), down at row
    /// r out_cols (K = expert_width); FP4 nibbles take k / 2 bytes per row, E4M3 scales k / 16.
    pub fn gate_shift(&self, rank: usize) -> (usize, usize) { (rank * self.mid_cols() * self.hidden / 2, rank * self.mid_cols() * (self.hidden / 16)) }
    pub fn down_shift(&self, rank: usize) -> (usize, usize) { (rank * self.out_cols() * self.expert_width / 2, rank * self.out_cols() * (self.expert_width / 16)) }
    /// The largest mailbox message: ROWS rows of the residual (the layer-2 post).
    pub fn slot_bytes(&self) -> usize { Self::ROWS * self.residual * 4 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sequence_split_takes_long_cold_prompts_at_a_row_edge() {
        assert_eq!(xsplit_rows(0, 32_099, 4096), Some(32_064));
        assert_eq!(xsplit_rows(0, 32_064, 4096), Some(32_000));
        assert_eq!(xsplit_rows(0, 8192, 4096), Some(8128));
        assert_eq!(xsplit_rows(0, 6145, 4096), Some(6144));
        assert_eq!(xsplit_rows(0, 6144, 4096), None);
        assert_eq!(xsplit_rows(0, 48_064, 4096), Some(48_000));
        assert_eq!(xsplit_rows(0, 48_065, 4096), None);
        assert_eq!(xsplit_rows(64, 32_099, 4096), None);
        assert_eq!(xsplit_rows(0, 32_099, 1024), None);
        assert_eq!(xsplit_rows(0, 0, 4096), None);
    }

    /// One Spark selects no path, world or rank (flash-serve selects them only under the split), so
    /// world() stays the pair's: the head's result arena keeps main's three floats per pair rank
    /// (flash::head::Head::arena_lengths), and no part, rank or layout is read.
    #[test]
    fn one_spark_selects_no_split_world_or_rank() {
        assert_eq!((on(), pair(), world(), rank()), (false, false, PAIR, None));
    }

    #[test]
    fn long_split_cuts_cold_long_prompts_before_a_short_tail() {
        // Off: never. On: cold prompts past 2 chunk rows and four tiles, to the last 64-row edge
        // before the end (at most LONG_SPLIT_MAX rows), so the tail of 1 to 64 rows gives the
        // first token.
        assert_eq!(split_rows_if(false, 0, 12_100, 4096), None);
        assert_eq!(split_rows_if(true, 0, 12_100, 4096), Some(12_096));
        assert_eq!(split_rows_if(true, 0, 20_417, 4096), Some(20_416));
        assert_eq!(split_rows_if(true, 0, LONG_SPLIT_MAX + 64, 4096), Some(LONG_SPLIT_MAX));
        assert_eq!(split_rows_if(true, 0, LONG_SPLIT_MAX + 65, 4096), None);
        assert_eq!(split_rows_if(true, 0, 32_100, 4096), None);
        assert_eq!(split_rows_if(true, 64, 12_100, 4096), None);
        assert_eq!(split_rows_if(true, 0, 8_192, 4096), None);
        assert_eq!(split_rows_if(true, 0, 8_193, 2048), Some(8_192));
        assert_eq!(split_rows_if(true, 0, 8_192, 2048), None);
        assert_eq!(split_rows_if(true, 0, 0, 2048), None);
        for end in [8_193, 12_100, LONG_SPLIT_MAX + 64] {
            let rows = split_rows_if(true, 0, end, 2048).unwrap();
            assert!(rows % 64 == 0 && (1..=64).contains(&(end - rows)));
        }
    }
    /// Astra 5e48233, the trigger in its order: check_layers on a directory that is then removed;
    /// prefill_advance of 64 valid tokens (no output, too short for the transaction) fails at the
    /// first layer-output open and leaves the session invalid at position 0 with the link usable;
    /// stop_layer_check; then an eligible long prompt. The transaction must not be entered until
    /// reset, so the prompt keeps the usual path, which refuses the failed session.
    #[test]
    fn a_failed_session_never_takes_the_transaction_until_reset() {
        let (mut valid, pos, mut layer_check, owns) = (true, 0, true, true);
        // prefill_advance(64 tokens) with the layer check on: not a transaction call.
        assert_eq!(split_rows_if(true, pos, 64, 4096), None);
        assert!(!split_session(false, valid, pos, layer_check, owns));
        valid = false;
        // stop_layer_check, then a long prompt whose rows the transaction would take.
        layer_check = false;
        assert_eq!(split_rows_if(true, pos, 12_100, 4096), Some(12_096));
        assert!(!split_session(true, valid, pos, layer_check, owns));
        // reset: valid again at position 0, so the same prompt may take it.
        valid = true;
        assert!(split_session(true, valid, pos, layer_check, owns));
        // A session that does not own its state, one past position 0, or a call without output never does.
        assert!(!split_session(true, true, 0, false, false));
        assert!(!split_session(true, true, 64, false, true));
        assert!(!split_session(false, true, 0, false, true));
    }
    /// flash_next_k16_finish's addressing (slot rows of `hidden` FP32-sized words, routed BF16
    /// at u16 index row*2*(top_k+1)*hidden + rank*2*hidden + col, shared FP32 at word
    /// (row*(top_k+1)+top_k)*hidden + col, output at row*hidden + col) and fp4_bw_down's
    /// (u32 word slot*hidden + col/2) with rank r's offsets: the finish reads exactly the
    /// columns rank r wrote, and every later rank's shifted shared row stays clear of routed
    /// columns. Run for the shipped checkpoint geometry and a different one, on two and four
    /// ranks, so nothing is fixed.
    #[test]
    fn rank_offsets_address_each_ranks_columns_for_any_geometry() {
        let layouts = [2, 4].map(|world| [TpLayout::new(2560, 4, 640, 10, 512, world).unwrap(), TpLayout::new(4096, 2, 1536, 8, 128, world).unwrap()]);
        for layout in layouts.into_iter().flatten() {
            let (h, k, world) = (layout.hidden, layout.top_k, layout.world);
            let slots = k + 1;
            for rank in 0..world {
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
                assert_eq!((layout.part(h, rank), layout.part(layout.expert_width, rank).len()), (rank * layout.out_cols()..(rank + 1) * layout.out_cols(), layout.mid_cols()));
                assert_eq!(layout.gate_shift(rank), (rank * layout.mid_cols() * h / 2, rank * layout.mid_cols() * h / 16));
                assert_eq!(layout.down_shift(rank).1, rank * layout.out_cols() * layout.expert_width / 16);
            }
            // Parts of whole K16 tiles that tile each width exactly.
            assert!(layout.mid_cols().is_multiple_of(16) && layout.out_cols().is_multiple_of(16));
            assert_eq!((layout.part(h, world - 1).end, layout.mid_cols() * world), (h, layout.expert_width));
            assert_eq!(layout.finish_ctas() as usize * layout.finish_threads() as usize, layout.out_cols());
            assert_eq!(layout.share(40, layout.mid_cols(), layout.expert_width) as usize, 40 / world);
            assert_eq!(layout.share(160, layout.out_cols(), layout.hidden) as usize, 160 / world);
            assert_eq!(layout.share(5, 1, 3), 2);
            assert_eq!(layout.slot_bytes(), TpLayout::ROWS * layout.residual * 4);
        }
        // The two-rank layout is the halves it always was.
        let pair = TpLayout::new(2560, 4, 640, 10, 512, 2).unwrap();
        assert_eq!((pair.mid_cols(), pair.out_cols(), pair.mid_words(), pair.gate_shift(1), pair.down_shift(1)),
            (320, 1280, 320, (320 * 2560 / 2, 320 * 2560 / 16), (1280 * 640 / 2, 1280 * 640 / 16)));
        // A world that does not cut every width into whole 16-column parts has no layout.
        for world in [0, 3, 6, 7] { assert!(TpLayout::new(2560, 4, 640, 10, 512, world).is_err(), "{world}"); }
        assert!(TpLayout::new(2560, 4, 640, 10, 512, 8).is_ok());
        // Whole 16-column parts that cut a 64-row tile of the shared weights have no layout.
        assert!(TpLayout::new(2624, 4, 640, 10, 512, 4).is_err() && TpLayout::new(2624, 4, 640, 10, 512, 2).is_err());
        // The finish CTAs cover each rank's columns exactly: 5 x 256 on the pair, 5 x 128 on quarters.
        let quarters = TpLayout::new(2560, 4, 640, 10, 512, 4).unwrap();
        assert_eq!([(pair.finish_ctas(), pair.finish_threads()), (quarters.finish_ctas(), quarters.finish_threads())], [(5, 256), (5, 128)]);
        assert!(TpLayout::new(2560, 4, 650, 10, 512, 2).is_err());
        // Prompt passes split only below the prompt expert kernels' threshold; verify passes up to the mailbox.
        assert_eq!((TpLayout::split_rows(true), TpLayout::split_rows(false)), (TpLayout::ROWS, TpLayout::PROMPT_ROWS));
        assert!(TpLayout::PROMPT_ROWS <= 16 && TpLayout::PROMPT_ROWS <= TpLayout::ROWS);
        assert!(TpLayout::new(2570, 4, 640, 10, 512, 2).is_err());
        assert!(TpLayout::new(2560, 0, 640, 10, 512, 2).is_err());
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
        assert_eq!(seq_slice_if(false, 0, 8192, 4096), 8192);
        assert_eq!(seq_slice_if(false, 0, 8009, 4096), 8009);
        assert_eq!(seq_first(0, 8009, 4096), Some(4032));
        assert_eq!(seq_slice_if(false, 0, 2048, 4096), 2048);
        assert_eq!(seq_first(0, 2048, 4096), Some(1024));
        assert_eq!(seq_slice_if(false, 0, 2047, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice_if(false, 0, 8193, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice_if(false, 0, 32768, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice_if(false, 4096, 12288, 4096), PIPELINE_TILE);
        assert_eq!(seq_first(0, 16384, 8192), Some(8192));
        assert_eq!(seq_slice_if(false, 0, 3000, 1024), 1024);
        // With the transaction on, a prompt it takes comes whole; longer ones keep the tiles.
        assert_eq!(seq_slice_if(true, 0, 8193, 4096), 8193);
        assert_eq!(seq_slice_if(true, 0, LONG_SPLIT_MAX + 64, 4096), LONG_SPLIT_MAX + 64);
        assert_eq!(seq_slice_if(true, 0, 32768, 4096), PIPELINE_TILE);
        assert_eq!(seq_slice_if(true, 4096, 12288, 4096), PIPELINE_TILE);
        // A workspace that is not a multiple of 64: a cut past it keeps the pipeline tiles.
        assert_eq!(seq_first(0, 2200, 1100), None);
        assert_eq!(seq_slice_if(false, 0, 2200, 1100), 1100);
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
    /// Prompts of WIDE_MIN_CHUNKS workspaces or more: the first slice and the last rows keep
    /// PIPELINE_TILE, the middle slices the whole workspace; shorter prompts keep PIPELINE_TILE;
    /// every cut stays a multiple of 64 and the token split does not change.
    #[test]
    fn long_prompts_take_the_workspace_in_the_middle() {
        let walk_if = |on: bool, end: usize, chunk: usize| {
            let (mut pos, mut cuts) = (0, vec![]);
            while pos < end { let n = seq_slice_if(on, pos, end, chunk).min(end - pos); cuts.push(n); pos += n; }
            cuts
        };
        let walk = |end: usize, chunk: usize| walk_if(false, end, chunk);
        // With the transaction on: prompts it takes come whole, longer ones walk as with it off.
        assert_eq!(walk_if(true, 12000, 4096), [12000]);
        assert_eq!(walk_if(true, 32768, 4096), walk(32768, 4096));
        assert_eq!(walk(32768, 4096), [2048, 4096, 4096, 4096, 4096, 4096, 4096, 2048, 2048, 2048]);
        assert_eq!(walk(40000, 4096), [[2048].as_slice(), &[4096; 8], &[2048, 2048, 1088]].concat());
        assert_eq!(walk(32099, 4096), [[2048].as_slice(), &[4096; 6], &[2048, 2048, 1379]].concat());
        assert_eq!(walk(28671, 4096), [[2048; 13].as_slice(), &[2047]].concat());
        assert_eq!(walk(12000, 4096), [2048, 2048, 2048, 2048, 2048, 1760]);
        assert_eq!(walk(8192, 4096), [8192]);
        assert_eq!(walk(10000, 2048), [2048, 2048, 2048, 2048, 1808]);
        for (end, chunk) in [(129220, 4096), (64217, 4096), (32769, 4096), (70000, 8192), (9000, 1024)] {
            let cuts = walk(end, chunk);
            assert_eq!((cuts[0], cuts.iter().sum::<usize>()), (PIPELINE_TILE.min(chunk), end));
            assert!(cuts.iter().all(|&n| n <= chunk) && cuts[..cuts.len() - 1].iter().all(|n| n % 64 == 0), "{end} {chunk}");
            assert!(cuts.iter().rev().take_while(|&&n| n < chunk).sum::<usize>() <= chunk + PIPELINE_TILE, "{end} {chunk}");
        }
        assert_eq!((seq_slice_if(false, 4096, 28672, 4096), seq_slice_if(false, 4096, 28671, 4096)), (4096, PIPELINE_TILE));
        assert_eq!(seq_slice_if(false, 26624, 32768, 4096), PIPELINE_TILE);
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
