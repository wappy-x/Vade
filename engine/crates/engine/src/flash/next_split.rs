//! The layer-major prompt transaction's tile stage (flash/next_split_plan.rs): one bank tile
//! through one layer. The stage runs the layer's attention and MoE, then layer l + 1's attention
//! mix, which takes over the MoE combine while the expert outputs are live (main's fused
//! boundary), so the tile leaves the residual, mixed rows and injection words that its next
//! stage starts from. The calls are the ones a pass over those layers makes, in the same order.
use super::*;
use crate::flash_next_split_plan::{cut_at, fund, Funds, SplitPlan, Step};
use crate::flash_tp_switch::SPLIT;
use split_bank::Bank;
#[path = "next_xsplit.rs"]
mod xsplit;

impl<'a, 'w> NextSession<'a, 'w> {
    /// Layer `li` of `layers` for bank rows `rows`, which are the chat's prompt rows from
    /// position `at`. `mix`: the tile holds only a residual (a seed tile at its first layer), so
    /// the layer's attention mix runs first. The chat's position is `at` during the stage.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tile_stage(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        bank: &Bank<'a>,
        chat: &mut Self,
        rows: std::ops::Range<usize>,
        at: usize,
        li: usize,
        layers: usize,
        mix: bool,
    ) -> Result<()> {
        if li >= layers { return Err(format!("prompt bank stage at layer {li} of {layers}")); }
        let n = rows.len();
        let unobserved = chat.layer_bytes.is_none() && chat.dump.is_none();
        let entered = bank.enter(work, rows)?;
        let pos = std::mem::replace(&mut chat.pos, at);
        let run = (|| -> Result<()> {
            work.block.resize_within_capacity(n * H)?;
            let sessions = &mut [&mut *chat];
            let spans = [0..n];
            let mut pending = None;
            if mix { Self::stage_mix(b, work, sessions, &spans, false, li, &mut pending)?; }
            // A pass that ends one layer later: the combine is left to layer li + 1's mix.
            Self::stage_body(b, work, sessions, &spans, false, li, (li + 2).min(layers), unobserved, &mut pending)?;
            if li + 1 < layers { Self::stage_mix(b, work, sessions, &spans, false, li + 1, &mut pending)?; }
            match pending { None => Ok(()), Some(l) => Err(format!("prompt bank stage left layer {l}'s combine")) }
        })();
        chat.pos = pos;
        Bank::leave(b, work, entered);
        run
    }
}

/// The transaction's tile on both ranks (rank 1 builds the same plan from the row count).
const TILE: usize = crate::flash_tp_switch::PIPELINE_TILE;

/// A rank's funding for `fund`: its chat's KV pages, a bank and a workspace.
struct Funding<'s, 'a, 'w> {
    b: &'s Backend<'a>,
    chat: &'s mut NextSession<'a, 'w>,
}
impl<'a> Funds for Funding<'_, 'a, '_> {
    type Bank = Bank<'a>;
    type Work = Workspace<'a>;
    fn pages(&mut self, rows: usize) -> bool { self.chat.fit_kv(self.b, rows).is_ok() }
    fn bank(&mut self, rows: usize) -> Option<Bank<'a>> { Bank::new(self.b, rows).ok() }
    fn work(&mut self) -> Option<Workspace<'a>> { self.chat.model.take_work(self.b, self.chat.capacity, TILE).ok() }
}

impl<'a, 'w> NextSession<'a, 'w> {
    /// Rank 0, a prompt call (prefill_inner, its `chunk`): a cold long prompt's rows up to its
    /// last 64-row edge run as the sequence split over more than two ranks, or on the pair, with
    /// VADE_LONG_SPLIT on, as the layer-major transaction with rank 1. The rows it ran (0: none);
    /// the call's tail runs as usual after them.
    pub(in crate::flash) fn split_prompt(&mut self, b: &Backend<'a>, tokens: &[u32], output: bool, chunk: usize) -> Result<usize> {
        if let Some(n) = self.model.tp_xsplit_cut(self, output, tokens.len(), chunk) {
            return Ok(if self.xsplit_front(b, &tokens[..n], chunk)? { n } else { 0 });
        }
        let Some(n) = self.model.tp_split_cut(self, output, tokens.len(), chunk) else { return Ok(0) };
        Ok(if self.split_front(b, &tokens[..n])? { n } else { 0 })
    }
    /// Rank 0: the layer-major transaction of a cold prompt's first `tokens.len()` rows with
    /// rank 1 (flash/next_split_plan.rs). The front funds its KV pages for every row, its suffix
    /// bank and its workspace before it asks; Ok(false): either rank could not fund them, nothing
    /// ran and the prompt takes the usual path.
    /// After it the chat is at the cut's end with no logits, rank 1 holds layers SPLIT..L of the
    /// prefix and the chat owes rank 1 the rest (as after the token split's import).
    fn split_front(&mut self, b: &Backend<'a>, tokens: &[u32]) -> Result<bool> {
        let model = self.model;
        let started = std::time::Instant::now();
        let cut = cut_at(tokens.len(), crate::flash_tp_switch::long_split_cut());
        let Some(plan) = SplitPlan::with_cut(tokens.len(), TILE, cut) else { return Ok(false) };
        let Some((bank, mut work)) = fund(&mut Funding { b, chat: &mut *self }, &plan, false) else {
            eprintln!("flashnext tp long split refused rows={} by=front", tokens.len());
            return Ok(false);
        };
        model.tp_split_start(self.slot);
        let mut answer = b.vector(1)?;
        let asked = model.tp_split_post(self.slot, self.capacity, tokens.len(), plan.cut)
            .and_then(|()| model.tp_split_begin(b))
            .and_then(|()| model.tp_split_word(b, &mut answer, false))
            .and_then(|()| Ok(b.read_vector_range(&answer, 0, 1)?[0] == 1.0));
        match asked {
            Ok(true) => (),
            Ok(false) => {
                eprintln!("flashnext tp long split refused rows={} by=worker", tokens.len());
                model.return_work(work);
                return Ok(false);
            }
            Err(error) => { model.return_work(work); return Err(model.tp_split_fail(error)); }
        }
        self.valid = false;
        self.logits_ready = false;
        if let Some(d) = self.draft.as_ref() { d.bound_at.set(None); }
        self.head.take_split();
        let result = work.status.clear(b, TILE, 0)
            .and_then(|()| model.tp_gdn_frame(b, &mut [&mut *self], TILE))
            .and_then(|()| self.front_steps(b, &mut work, &bank, &plan, tokens))
            .and_then(|()| {
                b.synchronize()?;
                if work.status.failed_after_readback()? || model.tp_failed() {
                    return Err(if model.tp_failed() { "tensor split mailbox timed out or its proxy failed" } else { "Flash-Next device routing rejected a row; reset cohort" }.into());
                }
                Ok(())
            });
        model.return_work(work);
        result.map_err(|e| model.tp_split_fail(e))?;
        self.pos = tokens.len();
        self.sync_pool_position(b)?;
        self.valid = true;
        model.tp_split_owed(self.slot, plan.cut);
        eprintln!("flashnext tp long split rows={} cut={} tile={TILE} ms={:.1}", tokens.len(), plan.cut, started.elapsed().as_secs_f64() * 1e3);
        Ok(true)
    }
    fn front_steps(&mut self, b: &Backend<'a>, work: &mut Workspace<'a>, bank: &Bank<'a>, plan: &SplitPlan, tokens: &[u32]) -> Result<()> {
        let model = self.model;
        let mut owed = false;
        for step in plan.front(SPLIT, LAYERS) {
            match step {
                Step::Seed(k) => {
                    let span = plan.prefix_span(k);
                    self.lead(b, work, None, span.clone(), &tokens[span])?;
                    model.tp_split_put(b, &work.residual)?;
                    owed = true;
                }
                Step::Head(k) => {
                    let span = plan.suffix_span(k);
                    self.lead(b, work, Some((bank, plan.cut)), span.clone(), &tokens[span])?;
                }
                Step::Import(l) => {
                    if std::mem::take(&mut owed) {
                        model.tp_seq_credits(b)?;
                        model.tp_split_notice(b, true)?;
                    }
                    model.tp_seq_state(b, &mut [&mut *self], l, plan.cut, false)?;
                }
                Step::Stage(l, k) => {
                    let span = plan.suffix_span(k);
                    let rows = span.start - plan.cut..span.end - plan.cut;
                    Self::tile_stage(b, work, bank, self, rows, span.start, l, LAYERS, false)?;
                }
                Step::Rows(k) => {
                    let span = plan.prefix_span(k);
                    work.residual.resize_within_capacity(span.len() * W)?;
                    model.tp_split_take(b, &mut work.residual)?;
                    let hidden = work.residual.slice(0..span.len() * W)?;
                    self.split_observe(b, work, &hidden, span.start, &tokens[span])?;
                }
                Step::Export(_) => return Err("prompt transaction: the front has no export".into()),
            }
        }
        // The drafter observes the suffix rows after the prefix rows, from the bank.
        for k in 0..plan.suffix_tiles() {
            let span = plan.suffix_span(k);
            let hidden = bank.residual(span.start - plan.cut..span.end - plan.cut)?;
            self.split_observe(b, work, &hidden, span.start, &tokens[span])?;
        }
        Ok(())
    }
    /// Rank 0: the frontend and layers [0, SPLIT) of prompt rows `span` (`tokens`). Without a
    /// bank (a seed tile) in the workspace, with the whole residual at the end; with a bank
    /// (the suffix, whose bank row 0 is position `cut`) in its bank rows, ending after layer
    /// SPLIT's attention mix.
    fn lead(&mut self, b: &Backend<'a>, work: &mut Workspace<'a>, bank: Option<(&Bank<'a>, usize)>, span: std::ops::Range<usize>, tokens: &[u32]) -> Result<()> {
        let n = span.len();
        let entered = match bank {
            Some((bank, cut)) => Some(bank.enter(work, span.start - cut..span.end - cut)?),
            None => {
                work.residual.resize_within_capacity(n * W)?;
                work.mixed.resize_within_capacity(n * H)?;
                None
            }
        };
        let pos = std::mem::replace(&mut self.pos, span.start);
        let run = (|| -> Result<()> {
            work.block.resize_within_capacity(n * H)?;
            let sessions = &mut [&mut *self];
            let spans = [0..n];
            Self::frontend(b, work, sessions, &[tokens], &spans, false)?;
            let end = if entered.is_some() { LAYERS } else { SPLIT };
            let mut pending = None;
            for li in 0..SPLIT { Self::layer_stage(b, work, sessions, &spans, false, li, end, true, &mut pending)?; }
            if entered.is_some() { Self::stage_mix(b, work, sessions, &spans, false, SPLIT, &mut pending)?; }
            match pending { None => Ok(()), Some(l) => Err(format!("prompt lead left layer {l}'s combine")) }
        })();
        self.pos = pos;
        if let Some(entered) = entered { Bank::leave(b, work, entered); }
        run
    }
    /// Rank 0: final rows `hidden` of positions `at..` enter the draft history, as a pass's
    /// rows do (execute).
    fn split_observe(&mut self, b: &Backend<'a>, work: &mut Workspace<'a>, hidden: &DeviceVector<'a>, at: usize, tokens: &[u32]) -> Result<()> {
        let model = self.model;
        let tiled = self.batch_drafts || (tokens.len() > 1 && super::mtp::tile_all());
        let Some(d) = &mut self.draft else { return Ok(()) };
        Self::flush_into(b, model, d, work)?;
        if model.pp_mtp() { Self::observe_kept(b, model, d, work, self.slot, at, self.batch_drafts, None)?; }
        if tiled {
            d.observe_rows(b, model, &work.mixed16, &mut work.hc, &mut work.qsa, at, tokens, hidden, 0)
        } else {
            d.observe(b, model, at, tokens, hidden, 0)
        }
    }
    /// Rank 1, KIND_HANDOFF with a SPLIT_PROMPT length: the worker of a `rows`-row transaction cut
    /// at `cut` for this chat at position 0. It answers whether it funded its KV pages for the
    /// prefix, its bank and its workspace first (a refusal ends it there); after it the chat is at
    /// the cut with layers SPLIT..L of the prefix.
    pub(in crate::flash) fn split_worker(b: &Backend<'a>, chat: &mut Self, rows: usize, cut: usize) -> Result<()> {
        let model = chat.model;
        if chat.pos != 0 || rows > chat.capacity { return Err(format!("prompt transaction of {rows} rows at position {}", chat.pos)); }
        model.tp_split_begin(b)?;
        // Funded: the plan, the KV pages, the bank and the workspace, all before the answer.
        let funded = SplitPlan::with_cut(rows, TILE, cut)
            .and_then(|p| fund(&mut Funding { b, chat: &mut *chat }, &p, true).map(|(bank, work)| (p, bank, work)));
        let mut answer = b.vector(1)?;
        b.write_vector(&mut answer, &[if funded.is_some() { 1.0 } else { 0.0 }])?;
        model.tp_split_word(b, &mut answer, true)?;
        // A refusal: the answer word lives until its put ran.
        let Some((plan, bank, mut work)) = funded else { return b.synchronize() };
        chat.valid = false;
        let result = work.status.clear(b, TILE, 0)
            .and_then(|()| model.tp_gdn_frame(b, &mut [&mut *chat], TILE))
            .and_then(|()| Self::worker_steps(b, &mut work, &bank, chat, &plan))
            .and_then(|()| {
                b.synchronize()?;
                if work.status.failed_after_readback()? || model.tp_failed() {
                    return Err(if model.tp_failed() { "tensor split mailbox timed out or its proxy failed" } else { "Flash-Next device routing rejected a row; reset cohort" }.into());
                }
                Ok(())
            });
        model.return_work(work);
        result?;
        chat.pos = plan.cut;
        chat.sync_pool_position(b)?;
        chat.valid = true;
        chat.logits_ready = false;
        Ok(())
    }
    fn worker_steps(b: &Backend<'a>, work: &mut Workspace<'a>, bank: &Bank<'a>, chat: &mut Self, plan: &SplitPlan) -> Result<()> {
        let model = chat.model;
        let mut exported = false;
        for step in plan.worker(SPLIT, LAYERS) {
            match step {
                Step::Stage(l, k) => {
                    let span = plan.prefix_span(k);
                    if l == SPLIT { model.tp_split_take(b, &mut bank.residual(span.clone())?)?; }
                    Self::tile_stage(b, work, bank, chat, span.clone(), span.start, l, LAYERS, l == SPLIT)?;
                }
                Step::Export(l) => {
                    if !std::mem::replace(&mut exported, true) { model.tp_split_notice(b, false)?; }
                    model.tp_seq_state(b, &mut [&mut *chat], l, plan.cut, true)?;
                }
                Step::Rows(k) => model.tp_split_put(b, &bank.residual(plan.prefix_span(k))?)?,
                Step::Seed(_) | Step::Head(_) | Step::Import(_) => return Err("prompt transaction: the worker has no front step".into()),
            }
        }
        model.tp_seq_credits(b)
    }
}
