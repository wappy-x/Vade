//! Flash-Next adapter for the serving scheduler's existing Session/Engine APIs.
//! The caller owns model loading, HTTP setup, global budget and metadata.
use crate::{BatchError, BatchOutput, BatchRows, Engine, Predictions, Session};
use spark_engine::flash::next_batched::{NextModel, NextPrefix, NextSession, NextStorage};
use spark_gpu_probe::driver::Backend;
use std::{cell::RefCell, rc::{Rc, Weak}, sync::atomic::{AtomicUsize, Ordering}};
use crate::prefix_switch::{adopt, cache_yield, prefix_cache};
use crate::kv_pages::Geometry;

/// The KV page: one GB10 virtual-memory granule (segments.rs maps divisors of it).
const KV_PAGE: u64 = 2 << 20;
/// Page geometry of one chat on this rank: the paged keys, values and index rows of every
/// attention layer it keeps and of the draft layer; fixed bytes = `full` (the full-window
/// session charge) less those rows at the window.
pub fn kv_geometry(model: &NextModel<'_>, context: usize, full: usize) -> Result<Geometry, String> {
    let widths: Vec<u64> = model.kv_page_widths().into_iter().map(|w| w as u64).collect();
    let window = widths.iter().sum::<u64>().checked_mul(context as u64).ok_or("KV page geometry overflows")?;
    Geometry::new(KV_PAGE, widths, (full as u64).saturating_sub(window), context as u64)
}
/// Rows a chat charges past its scheduler tokens: the draft layer's verify rows
/// (NextSession::fit_kv) and 64 rows of margin.
pub fn kv_ahead(model: &NextModel<'_>) -> u64 { 64 + model.mtp_depth() as u64 + 1 }
/// Bytes a chat of `tokens` charges: whole pages of its rows (window at most) and its
/// fixed state. The session maps at most these pages (NextSession::fit_kv).
pub fn kv_charge(pages: &Geometry, ahead: u64, tokens: usize) -> Option<usize> {
    usize::try_from(pages.charge(tokens as u64, ahead)?).ok()
}

pub struct FlashNext<'b, 'a, 'w> {
    backend: &'b Backend<'a>,
    model: &'w NextModel<'a>,
    context: usize,
    eos: Vec<u32>,
    reservation: usize,
    /// Admission charges each chat its own KV pages (kv_charge), not the window.
    pages: Geometry,
    ahead: u64,
    chunk: usize,
    /// VADE_BATCH_PREFILL=0 disables guarded independent prompt spans.
    batch_prefill: bool,
    /// VADE_BATCH_PREFILL_MAX_TOKENS: longer prompts retain serial prefill.
    batch_prefill_max_tokens: usize,
    /// Env modes are fixed; AUTO resolves against the current active cohort.
    batch_on: crate::mtp_batch::Mode,
    cohort_cost_control: bool,
    /// VADE_C8_GUARD_ROBUST, fixed at construction. False is the live 2% latch.
    guard_robust: bool,
    graph_fallback: bool,
    cohort_plain: std::cell::Cell<bool>,
    cohort_draft: std::cell::Cell<bool>,
    one_row_pooled: crate::mtp_batch::Mode,
    /// VADE_VERIFY_ROWS: verify rows per tick and per pass for VADE_MTP_BATCH cohorts.
    verify_rows: usize,
    /// VADE_MTP_COHORT=wide: four chats or fewer keep 16 rows (tick_width); unset 32 above.
    wide_tick: bool,
    /// VADE_COHORT_EVEN_ROWS (default on; 0 = the sequential split): equal verify-row caps
    /// for a drafting cohort's decoders, so no lone chat buys one more drafter step.
    even_rows: bool,
    cohort: std::cell::Cell<usize>,
    /// VADE_MTP_ADAPT, read once at construction (default on).
    adapt: bool,
    /// VADE_MTP_ADAPT=full: the whole resident depth, no chooser.
    adapt_full: bool,
    /// VADE_MTP_SAMPLED: sampled requests keep MTP drafting (strict, seed-keyed verify).
    sampled_draft: bool,
    /// VADE_MTP_SAMPLED=2: sampled rows join the cohort's batched verify pass.
    sampled_cohort: bool,
    /// VADE_MTP_SAMPLED_TOPK=1: compact exact verifier readback (default off).
    sampled_top_k: bool,
    /// VADE_FLASHNEXT_PREFIX_CACHE (unset = on where supported; 0 = off, the no-reuse path): exact prompt
    /// prefixes at 64-aligned checkpoints for the serving cache.
    prefix_cache: bool,
    /// VADE_FLASHNEXT_ADOPT (default off).
    adopt: bool,
    /// VADE_FLASHNEXT_CACHE_YIELD (default on): retention yields to host memory pressure.
    cache_yield: bool,
    /// The retained cache's own room (prefix_switch::cache_room): set at startup, outside
    /// admission and host pressure.
    cache_room: usize,
}
/// Device bytes a live request holds above its admitted bound besides the chunk-form GDN
/// tails, measured at CONTEXT 262144 with the KV reserve off: session state steps up once a
/// prompt reaches 993 tokens and stays flat to 65535 (+49.4 MB over the plain bound after its
/// tails; +44.1..47.3 MB over the retaining bound), and publishing a long prompt's checkpoint
/// peaks 53.6 MB above the settled footprint. The retained cache leaves this much free per
/// live request; admission is unchanged.
const CACHE_REQUEST_MARGIN_BYTES: usize = 128 << 20;
/// Host bytes the yielding cache leaves free beyond the KV reserve's room when it grows:
/// growth between two cache balances (graph captures, KV pages a live request maps) stays
/// clear of the reserve's pressure rule. Yielding starts only past the room itself, so an
/// admission that maps pages inside this slack evicts nothing (hysteresis).
const CACHE_YIELD_SLACK_BYTES: usize = 1 << 30;
/// Adoption: the arenas an in-place prefix's rows live in, filled when its request ends.
/// `id` names them in the cache's shared-allocation accounting (charged once).
struct Arenas<'a> {
    id: usize,
    bytes: usize,
    storage: RefCell<Option<NextStorage<'a>>>,
}
fn arena_id() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(1 << 40);
    NEXT.fetch_add(1, Ordering::Relaxed)
}
pub struct Chat<'s, 'a> {
    backend: &'s Backend<'a>,
    model: &'s NextModel<'a>,
    capacity: usize,
    chunk: usize,
    /// None while this chat is a cached prefix; configure_request restores it.
    live: Option<NextSession<'a, 's>>,
    /// A cached chat's prefix. Only the cache owns prefixes: its byte charge is the memory.
    prefix: RefCell<Option<Rc<NextPrefix<'a>>>>,
    /// The prefix this live chat published, while the cache still holds it.
    published: RefCell<Weak<NextPrefix<'a>>>,
    prefix_cache: bool,
    /// This request asked for caching (`cache:true` or a `prompt_cache_key`): only such
    /// requests checkpoint and publish, so anonymous cold traffic pays nothing.
    retain_requested: bool,
    /// Effective per-request drafting policy, separate from retained MTP history.
    draft_disabled: bool,
    lookup: spark_engine::flash_next_lookup::Lookup,
    lookup_max: usize,
    depth: usize,
    /// VADE_MTP_BATCH: leave proposals to FlashNext::propose_batch (one cohort pass).
    batch_on: crate::mtp_batch::Mode,
    cohort: &'s std::cell::Cell<usize>,
    cohort_plain: &'s std::cell::Cell<bool>,
    /// Per-cycle adaptive depth, independently default-off.
    adapt: bool,
    /// VADE_MTP_ADAPT=full: the chooser never caps the depth.
    adapt_full: bool,
    /// VADE_MTP_SAMPLED, read once at construction (default off).
    sampled_draft: bool,
    depth_cap: usize,
    /// The request's own depth (min resident): what the chat drafts in a cohort.
    base_depth: usize,
    /// VADE_FLASHNEXT_ADOPT (default off): `reservation` charges an adopted arena; `arena`
    /// names this chat's arenas; `arenas` is where they go when its request ends (the prefix
    /// it published, else the one it adopted from) or, for a cached chat, where its prefix's
    /// rows are; `adopted` holds arenas a fork took until configure_request builds on them.
    adopt: bool,
    reservation: usize,
    arena: usize,
    arenas: RefCell<Option<Rc<Arenas<'a>>>>,
    adopted: RefCell<Option<NextStorage<'a>>>,
}
fn report_arenas(phase: &str, before: spark_gpu_probe::KvArenaStats, after: spark_gpu_probe::KvArenaStats) {
    eprintln!("flashnext kv_arenas phase={phase} allocated_count={} allocated_bytes={} reused_count={} reused_bytes={} returned_count={} returned_bytes={} direct_freed_count={} direct_freed_bytes={} evicted_count={} evicted_bytes={}",
        after.allocated_count - before.allocated_count, after.allocated_bytes - before.allocated_bytes,
        after.reused_count - before.reused_count, after.reused_bytes - before.reused_bytes,
        after.returned_count - before.returned_count, after.returned_bytes - before.returned_bytes,
        after.direct_freed_count - before.direct_freed_count, after.direct_freed_bytes - before.direct_freed_bytes,
        after.evicted_count - before.evicted_count, after.evicted_bytes - before.evicted_bytes);
}
impl Drop for Chat<'_, '_> {
    fn drop(&mut self) {
        if let Some(session) = self.live.take() {
            let before = self.backend.kv_reserve_stats();
            // Preserve field drop order: live state precedes cached prefix ownership.
            drop(session);
            report_arenas("retire", before, self.backend.kv_reserve_stats());
        }
    }
}
impl<'b, 'a, 'w> FlashNext<'b, 'a, 'w> {
    /// Rows one cohort tick may spend: VADE_MTP_COHORT=wide keeps four chats on 16.
    fn tick_width(&self, chats: usize) -> usize {
        if self.wide_tick { crate::mtp_batch::tick_width(chats, self.verify_rows) } else { self.verify_rows }
    }
    pub fn new(
        backend: &'b Backend<'a>,
        model: &'w NextModel<'a>,
        context: usize,
        eos: Vec<u32>,
    ) -> Result<Self, String> {
        Self::with_chunk(backend, model, context, eos, spark_engine::flash_next_rows::prompt_tile())
    }
    /// Explicit admitted chunk for embedded callers and integration qualification.
    pub fn with_chunk(
        backend: &'b Backend<'a>,
        model: &'w NextModel<'a>,
        context: usize,
        eos: Vec<u32>,
        chunk: usize,
    ) -> Result<Self, String> {
        Self::with_chunk_experiments(backend,model,context,eos,chunk,crate::mtp_batch::Experiments::default())
    }
    pub fn with_experiments(backend: &'b Backend<'a>, model: &'w NextModel<'a>, context: usize,
        eos: Vec<u32>, experiments: crate::mtp_batch::Experiments) -> Result<Self,String> {
        Self::with_chunk_experiments(backend,model,context,eos,spark_engine::flash_next_rows::prompt_tile(),experiments)
    }
    pub fn with_chunk_experiments(backend: &'b Backend<'a>, model: &'w NextModel<'a>, context: usize,
        eos: Vec<u32>, chunk: usize, experiments: crate::mtp_batch::Experiments) -> Result<Self,String> {
        let prefix_cache = prefix_cache(!model.pp_front() || model.tp_prefix_supported())?;
        // The prompt checkpoint borrows the MTP rollback storage's funding (next_prefix.rs);
        // a model without MTP funds it here.
        let reservation = model.session_reservation_bytes_for_chunk(context, chunk)?
            + if prefix_cache && model.mtp_depth() == 0 { NextSession::checkpoint_bytes() } else { 0 };
        let pages = kv_geometry(model, context, reservation)?;
        let ahead = kv_ahead(model);
        let batch_on = match experiments.batch_mode { Some(mode) => mode, None => crate::mtp_batch::switch()? };
        let cohort_cost_control = experiments.cohort_cost;
        let guard_robust = experiments.guard_robust;
        if cohort_cost_control && batch_on != crate::mtp_batch::Mode::Auto8 {
            return Err("cohort controller requires Auto8 batch mode".into());
        }
        let one_row_pooled = crate::mtp_batch::one_row_pooled()?;
        // VADE_MTP_COHORT=wide (Experiments::wide_tick): unset VADE_VERIFY_ROWS is 32, else 16.
        let verify_rows = spark_engine::flash_next_rows::verify_rows_setting()?
            .unwrap_or(if experiments.wide_tick { 32 } else { crate::mtp_batch::VERIFY_ROWS });
        let even_rows = match std::env::var("VADE_COHORT_EVEN_ROWS").as_deref() {
            Ok("0") => false,
            Err(std::env::VarError::NotPresent) | Ok("1") => true,
            _ => return Err("VADE_COHORT_EVEN_ROWS must be 0 or 1".into()),
        };
        let adapt = crate::mtp_batch::adapt_depth()?;
        let adapt_full = crate::mtp_batch::adapt_full()?;
        let sampled_draft = crate::mtp_batch::sampled_draft()?;
        let sampled_cohort = crate::mtp_batch::sampled_cohort()?;
        let sampled_top_k = crate::mtp_batch::sampled_top_k()?;
        let batch_prefill = match std::env::var("VADE_BATCH_PREFILL").as_deref() {
            Ok("0") => false,
            Err(std::env::VarError::NotPresent) | Ok("1") => true,
            _ => return Err("VADE_BATCH_PREFILL must be 0 or 1".into()),
        };
        let batch_prefill_max_tokens = match std::env::var("VADE_BATCH_PREFILL_MAX_TOKENS") {
            Err(std::env::VarError::NotPresent) => 512,
            Ok(value) => value.parse::<usize>().ok().filter(|&n| n > 0)
                .ok_or("VADE_BATCH_PREFILL_MAX_TOKENS must be a positive integer")?,
            Err(_) => return Err("VADE_BATCH_PREFILL_MAX_TOKENS must be a positive integer".into()),
        };
        // In place on one Spark; the tensor split's front copies like its back rank.
        let adopt = prefix_cache && adopt(!model.pp_front())?;
        let cache_yield = prefix_cache && cache_yield()?;
        Ok(Self {
            backend,
            model,
            context,
            eos,
            reservation,
            pages,
            ahead,
            chunk,
            batch_prefill,
            batch_prefill_max_tokens,
            cohort: std::cell::Cell::new(1),
            batch_on,
            cohort_cost_control,
            guard_robust,
            graph_fallback: experiments.graph_fallback,
            cohort_plain: std::cell::Cell::new(false),
            cohort_draft: std::cell::Cell::new(false),
            one_row_pooled,
            verify_rows,
            wide_tick: experiments.wide_tick,
            even_rows,
            adapt,
            adapt_full,
            sampled_draft,
            sampled_cohort,
            sampled_top_k,
            prefix_cache,
            adopt,
            cache_yield,
            cache_room: 0,
        })
    }
    /// The room the startup plan took from the session pool for retained prefixes.
    pub fn with_cache_room(mut self, bytes: usize) -> Self {
        self.cache_room = if self.prefix_cache { bytes } else { 0 };
        self
    }
    /// Whether the serving cache may retain and restore Flash-Next prompt prefixes.
    pub fn prefix_cache(&self) -> bool {
        self.prefix_cache
    }
    /// VADE_FLASHNEXT_ADOPT: retained entries may hold whole session arenas.
    pub fn adopt(&self) -> bool {
        self.adopt
    }
}
impl<'s, 'a> Chat<'s, 'a> {
    fn sync_batch(&mut self) {
        let chats = self.cohort.get();
        let (on, defer) = self.batch_on.feed(chats, self.cohort_plain.get());
        if let Some(s) = &mut self.live {
            s.batch_drafts(on);
            s.defer_draft_feed(defer);
        }
    }
    /// VADE_MTP_ADAPT: a chat alone drafts the chooser's depth (either draft path);
    /// in a cohort, or with the switch off, it drafts its request depth.
    fn model_draft_enabled(&self) -> bool {
        !self.draft_disabled && self.live.as_ref().is_some_and(|s| s.mtp_depth() > 0)
    }
    fn mtp_depth_now(&self) -> usize {
        if self.draft_disabled { return 0; }
        if self.adapt && self.cohort.get() <= 1 { self.depth.min(self.depth_cap) } else { self.base_depth }
    }
    fn s(&self) -> Result<&NextSession<'a, 's>, String> {
        self.live.as_ref().ok_or_else(|| "Flash-Next cached prefix was not restored".into())
    }
    fn s_mut(&mut self) -> Result<&mut NextSession<'a, 's>, String> {
        self.live.as_mut().ok_or_else(|| "Flash-Next cached prefix was not restored".into())
    }
    /// Restorable 64-aligned prefix length: the cached or published prefix, else the live
    /// session's checkpoint (publishing moves it into the shared prefix).
    fn prefix_position(&self) -> Option<usize> {
        if !self.prefix_cache { return None; }
        if let Some(p) = self.prefix.borrow().as_ref() { return Some(p.position()); }
        if let Some(p) = self.published.borrow().upgrade() { return Some(p.position()); }
        self.live.as_ref().and_then(|s| s.checkpoint_position())
    }
    /// A prefix that already exists (cached, or published and still cached): sharing it
    /// allocates nothing. New prefixes come only from the cache's own charged publish.
    fn existing_prefix(&self) -> Option<Rc<NextPrefix<'a>>> {
        self.prefix.borrow().clone().or_else(|| self.published.borrow().upgrade())
    }
    /// Adoption: this chat's prefix keeps its rows in arenas (a live chat's own, or the
    /// arenas a cached chat hands over).
    fn in_place(&self) -> bool {
        self.adopt && self.existing_prefix().is_some_and(|p| p.in_place())
    }
    /// Adoption: a cached chat's arenas are free to take (its request ended, no fork took them).
    fn adoptable(&self) -> bool {
        self.live.is_none() && self.arenas.borrow().as_ref().is_some_and(|a| a.storage.borrow().is_some())
    }
    /// Adoption only where a copy is costly: its rows are at least a quarter of the session
    /// reservation (long prompts); short prefixes copy, so each pins megabytes, not an arena.
    fn publish_in_place(&self) -> bool {
        self.adopt && self.live.as_ref().is_some_and(|s| s.prefix_bytes().saturating_mul(4) >= self.reservation)
    }
    /// Adoption: the arenas a cached snapshot of this chat charges (counted once by id).
    fn arena_allocation(&self) -> Option<(usize, usize)> {
        if !self.adopt { return None; }
        if let Some(a) = self.arenas.borrow().as_ref() { return Some((a.id, a.bytes)); }
        (self.retain_requested && self.existing_prefix().is_none() && self.publish_in_place())
            .then_some((self.arena, self.reservation))
    }
    /// A cached chat sharing this chat's prefix; a live chat moves its checkpoint out once.
    fn snapshot(&self) -> Result<Self, String> {
        self.prefix_position().ok_or("Flash-Next chat has no restorable prefix")?;
        let prefix = match self.existing_prefix() {
            Some(p) => p,
            None => {
                let s = self.s()?;
                let in_place = self.publish_in_place();
                let p = Rc::new(if in_place { s.save_prefix_in_place()? } else { s.save_prefix(self.backend)? });
                *self.published.borrow_mut() = Rc::downgrade(&p);
                if in_place {
                    *self.arenas.borrow_mut() = Some(Rc::new(Arenas { id: self.arena, bytes: self.reservation,
                        storage: RefCell::new(None) }));
                }
                p
            }
        };
        // Adoption: a fork of a cached chat takes its arenas; only one fork can.
        let adopted = if prefix.in_place() && self.live.is_none() {
            Some(self.arenas.borrow().as_ref().and_then(|a| a.storage.borrow_mut().take())
                .ok_or("Flash-Next prefix arenas were adopted")?)
        } else { None };
        let arenas = if prefix.in_place() { self.arenas.borrow().clone() } else { None };
        Ok(Self {
            backend: self.backend,
            model: self.model,
            capacity: self.capacity,
            chunk: self.chunk,
            live: None,
            prefix: RefCell::new(Some(prefix)),
            published: RefCell::new(Weak::new()),
            prefix_cache: self.prefix_cache,
            retain_requested: false,
            draft_disabled: false,
            lookup: Default::default(),
            lookup_max: 0,
            depth: self.depth,
            batch_on: self.batch_on,
            cohort: self.cohort,
            cohort_plain: self.cohort_plain,
            adapt: self.adapt,
            adapt_full: self.adapt_full,
            sampled_draft: self.sampled_draft,
            depth_cap: usize::MAX,
            base_depth: self.base_depth,
            adopt: self.adopt,
            reservation: self.reservation,
            arena: arenas.as_ref().map_or(0, |a| a.id),
            arenas: RefCell::new(arenas),
            adopted: RefCell::new(adopted),
        })
    }
}
impl Session for Chat<'_, '_> {
    fn configure_request(&mut self, request: &crate::protocol::Request) -> Result<(), String> {
        self.draft_disabled = false;
        if self.live.is_none() {
            // A cached prefix becomes a live session for this request (fresh pool slot).
            let prefix = self.prefix.borrow_mut().take().ok_or("empty Flash-Next chat")?;
            self.live = Some(match self.adopted.get_mut().take() {
                Some(storage) => self.model.session_from_storage(self.backend, self.chunk, &prefix, storage)?,
                None => {
                    if self.adopt { self.arena = arena_id(); *self.arenas.get_mut() = None; }
                    self.model.session_from_prefix(self.backend, self.capacity, self.chunk, &prefix)?
                }
            });
        }
        self.retain_requested = request.cache || request.prompt_cache_key.is_some();
        self.lookup = Default::default();
        self.lookup_max = if request.prompt_lookup && spark_engine::flash_next_lookup::enabled() && self.s()?.mtp_depth() > 0 {
            request.prompt_lookup_max_draft
        } else {
            0
        };
        // The server fills its default depth into requests that name none. VADE_MTP_ADAPT:
        // a depth at or above that default lets the chooser draft up to the resident depth
        // (VADE_FLASHNEXT_MTP_DEPTH); explicit shallower depths are kept.
        let default = spark_engine::flash_next_lookup::DEFAULT_MTP_DEPTH;
        let requested = request.mtp_depth.unwrap_or(default);
        self.depth = if self.adapt && requested >= default { usize::MAX } else { requested }
            .min(self.s()?.mtp_depth());
        self.base_depth = requested.min(self.s()?.mtp_depth());
        let (backend, max) = (self.backend, self.lookup_max);
        self.s_mut()?.enable_lookup(backend, max)?;
        Ok(())
    }
    fn set_cache_retention(&mut self, retain: bool) {
        let on = retain && self.prefix_cache && self.retain_requested;
        // Remember the scheduler's effective opt-in, including cache:false overrides.
        self.retain_requested = on;
        if let Some(s) = &mut self.live { s.set_prefix_checkpoint(on); }
    }
    fn disable_model_draft(&mut self) {
        self.draft_disabled = true;
        self.lookup_max = 0;
        // Plain generation makes no proposals. A retained prefix must still carry the
        // resident draft history: checkpoint_position rejects incomplete MTP prefixes.
        // The full session reservation already funds it; non-retaining traffic frees it.
        if !self.retain_requested {
            if let Some(s) = &mut self.live { s.disable_mtp(); }
        }
    }
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> {
        // A live speculator may own the pending token when policy switches to plain.
        // Keep its target for the one-token fallback; widths and verifier guards disable MTP.
        if self.live.is_some() { Some(self) } else { None }
    }
    fn truncate(&mut self, p: usize) -> Result<(), String> {
        let backend = self.backend;
        self.s_mut()?.commit_mtp(backend, p)
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.sync_batch();
        let backend = self.backend;
        let s = self.s_mut()?;
        s.step(backend, token)?;
        s.logits(backend)
    }
    fn decode_logits(&mut self, rows: usize) -> Result<Option<Vec<Vec<f32>>>, String> {
        self.s()?.decode_logits(self.backend, rows).map(Some)
    }
    fn advance(&mut self, token: u32) -> Result<(), String> {
        self.sync_batch();
        let backend = self.backend;
        self.s_mut()?.prefill_advance(backend, &[token])
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.sync_batch();
        let backend = self.backend;
        self.s_mut()?.prefill(backend, tokens)
    }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<(), String> {
        self.sync_batch();
        let backend = self.backend;
        self.s_mut()?.prefill_advance(backend, tokens)
    }
    fn read_ahead(&self, tokens: &[u32]) {
        if let Some(s) = &self.live { s.read_ahead(tokens) }
    }
    fn prefill_chunk_size(&self, prompt_end: usize) -> usize {
        let chunk = self.live.as_ref().map_or(self.chunk, |s| s.chunk_size());
        // Two Sparks: a cold 2K..8K prompt whole (the engine runs the token split's two parts),
        // others in pipeline tiles; elsewhere main's automatic prompt tile for live and lazily
        // restored chats.
        self.model.prompt_slice(self.live.as_ref(), prompt_end, chunk)
            .unwrap_or_else(|| spark_engine::flash_next_rows::prompt_tile_for(prompt_end, chunk))
    }
    /// Prefix cache: only the 64-aligned checkpoint restores exactly (recurrent GDN state).
    fn can_fork(&self) -> bool {
        self.prefix_position().is_some()
    }
    /// A live source (another request's prefix reuse) forks only an existing prefix.
    fn can_fork_prefix(&self, position: usize) -> bool {
        self.prefix_position() == Some(position) && (self.live.is_none() || self.existing_prefix().is_some())
            && (!self.in_place() || self.adoptable())
    }
    fn checkpoint_reusable_tokens(&self, tokens: usize) -> usize {
        self.prefix_position().filter(|&p| p <= tokens).unwrap_or(0)
    }
    fn fork(&self) -> Result<Self, String> {
        self.snapshot()
    }
    fn fork_prefix(&self, position: usize) -> Result<Self, String> {
        if !self.can_fork_prefix(position) {
            return Err("Flash-Next restores only its 64-aligned prompt checkpoint".into());
        }
        self.snapshot()
    }
    fn retained_bytes(&self) -> Option<usize> {
        if !self.prefix_cache {
            return None;
        }
        Some(4096 + match (self.existing_prefix(), &self.live) {
            (Some(p), _) => p.device_bytes(),
            (None, Some(s)) if self.publish_in_place() => s.checkpoint_prefix_bytes(),
            (None, Some(s)) => s.prefix_bytes(),
            (None, None) => 0,
        })
    }
    /// Adoption: an in-place snapshot charges its arenas as one shared allocation; the
    /// publishing request's reservation funds them while it runs, the next turn's while it
    /// adopts them.
    fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> {
        self.arena_allocation().map(|a| vec![a])
    }
    fn retained_private_bytes(&self) -> usize {
        if self.adopt { self.retained_bytes().unwrap_or(0) } else { 0 }
    }
    fn funded_snapshot_allocations(&self, tokens: Option<usize>) -> Vec<(usize, usize)> {
        // A retiring chat still funds its arenas until its terminal socket drains.
        let funded = if tokens.is_some() { self.adoptable() } else { self.live.is_some()
            || self.arenas.borrow().as_ref().is_some_and(|a| a.storage.borrow().is_some()) };
        self.arena_allocation().filter(|_| funded).into_iter().collect()
    }
    /// Adoption: a finished request's arenas go where `arenas` points; the session's pool
    /// slot and recurrent state are released now.
    fn retire(&mut self) {
        if !self.adopt { return; }
        let Some(arenas) = self.arenas.get_mut().as_ref() else { return };
        if let Some(storage) = self.live.take().and_then(|s| s.into_storage()) {
            *arenas.storage.borrow_mut() = Some(storage);
        }
    }
    fn finish(&mut self) -> Result<(), String> {
        self.backend.synchronize()
    }
}
impl spark_engine::draft::Target for Chat<'_, '_> {
    fn checkpoint(&self) -> usize {
        self.live.as_ref().map_or(0, |s| s.position())
    }
    fn truncate(&mut self, p: usize) -> Result<(), String> {
        Session::truncate(self, p)
    }
    fn step(&mut self, t: u32) -> Result<Vec<f32>, String> {
        Session::step(self, t)
    }
    fn verify_batch(&mut self, t: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        self.sync_batch();
        let backend = self.backend;
        self.s_mut()?.verify_mtp(backend, t)
    }
    fn max_verify_rows(&self) -> usize {
        self.live.as_ref().map_or(1, |s| s.max_verify_rows())
    }
    fn model_draft_width(&self) -> usize {
        if self.model_draft_enabled() { self.depth } else { 0 }
    }
    fn native_draft_policy(&self) -> bool {
        true
    }
    fn sampled_model_draft(&self) -> bool {
        self.sampled_draft
    }
    fn adaptive_model_depth(&self) -> bool {
        self.adapt && !self.adapt_full && self.cohort.get() <= 1
    }
    fn limit_model_depth(&mut self, depth: usize) {
        self.depth_cap = depth;
    }
    fn model_draft_costs(&self) -> Option<spark_engine::draft::CostCurve> {
        // VADE_MTP_ADAPT prior: verify pass ms by rows 1/2/4/8/16 (one chat, short context,
        // main df70005f: 41.8 + 4.76 per row); the speculator learns the scale online.
        self.adapt.then_some(spark_engine::draft::CostCurve([46.6, 51.3, 60.8, 79.9, 118.0]))
    }
    fn defer_model_draft(&self, limit: usize) -> Option<usize> {
        // The limit propose_model would apply to either source; lookup chats keep theirs.
        self.batch_on.enabled(self.cohort.get())
            .then(|| limit.min(self.mtp_depth_now().max(self.lookup_max)))
    }
    fn propose_model(
        &mut self,
        history: &[u32],
        next: u32,
        limit: usize,
    ) -> Result<Option<spark_engine::draft::ModelProposal>, String> {
        self.sync_batch();
        let now = std::time::Instant::now();
        let copied = if self.lookup_max == 0 {
            Vec::new()
        } else {
            self.lookup
                .propose(history, next, limit.min(self.lookup_max))
        };
        let tokens = if copied.is_empty() {
            let (backend, depth) = (self.backend, limit.min(self.mtp_depth_now()));
            self.s_mut()?.propose_mtp(backend, next, depth)?
        } else {
            copied
        };
        Ok(Some(spark_engine::draft::ModelProposal {
            tokens,
            proposal_ms: now.elapsed().as_secs_f64() * 1000.,
            catchup_ms: 0.,
        }))
    }
}
impl<'b, 'a, 'w> Engine for FlashNext<'b, 'a, 'w> {
    type Session<'s>
        = Chat<'s, 'a>
    where
        Self: 's;
    fn session_for_request(&self, request: &crate::protocol::Request) -> Result<Self::Session<'_>, String> {
        self.backend.select_arm(request.arm_x)?;
        self.session()
    }
    /// The session maps the KV pages its admission charge covers at once (kv_charge's extent),
    /// so it reuses a kept range of that size without trimming and mapping it page by page.
    fn session_for_admission(&self, request: &crate::protocol::Request, tokens: usize) -> Result<Self::Session<'_>, String> {
        self.model.set_kv_rows_hint(self.pages.extent((tokens as u64).saturating_add(self.ahead)) as usize);
        let session = self.session_for_request(request);
        self.model.set_kv_rows_hint(0);
        session
    }
    fn session(&self) -> Result<Self::Session<'_>, String> {
        let before = self.backend.kv_reserve_stats();
        let mut session = self.model.session_with_chunk(self.backend, self.context, self.chunk)?;
        report_arenas("admit", before, self.backend.kv_reserve_stats());
        session.batch_drafts(self.batch_on.enabled(self.cohort.get()));
        Ok(Chat {
            backend: self.backend,
            model: self.model,
            capacity: self.context,
            chunk: self.chunk,
            live: Some(session),
            prefix: RefCell::new(None),
            published: RefCell::new(Weak::new()),
            prefix_cache: self.prefix_cache,
            retain_requested: false,
            draft_disabled: false,
            lookup: Default::default(),
            lookup_max: 0,
            depth: self.model.mtp_depth(),
            batch_on: self.batch_on,
            cohort: &self.cohort,
            cohort_plain: &self.cohort_plain,
            adapt: self.adapt,
            adapt_full: self.adapt_full,
            sampled_draft: self.sampled_draft,
            depth_cap: usize::MAX,
            base_depth: self.model.mtp_depth().min(spark_engine::flash_next_lookup::DEFAULT_MTP_DEPTH),
            adopt: self.adopt,
            reservation: self.reservation,
            arena: if self.adopt { arena_id() } else { 0 },
            arenas: RefCell::new(None),
            adopted: RefCell::new(None),
        })
    }
    fn context_limit(&self) -> usize {
        self.context
    }
    fn vocab_size(&self) -> usize {
        self.model.vocabulary()
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    fn active_cohort(&self, chats: usize) {
        self.cohort.set(chats);
        self.backend.kv_reserve_cohort(chats);
        if self.cohort_cost_control {
            self.cohort_plain.set(false);
            self.cohort_draft.set(false);
        }
    }
    fn cohort_cost_control(&self) -> bool { self.cohort_cost_control }
    fn cohort_guard_robust(&self) -> bool { self.guard_robust }
    fn cohort_draft_head_limit(&self) -> Option<usize> {
        let rows = self.model.draft_head_vocabulary();
        (rows > 0 && rows < self.model.vocabulary()).then_some(rows)
    }
    fn set_cohort_choice(&self, draft: Option<bool>) {
        self.cohort_plain.set(draft == Some(false));
        self.cohort_draft.set(draft == Some(true));
    }
    fn prepare_cohort_path(&self, draft: bool) -> Result<(), BatchError> {
        if draft && self.cohort_draft.get() {
            self.model.release_pooled_decode(self.backend)
                .map_err(|message| BatchError { request: None, message })?;
        }
        Ok(())
    }
    /// Requests that do not ask for caching keep main's no-cache path (scheduling included).
    fn prefix_cache_opt_in(&self) -> bool {
        self.prefix_cache
    }
    fn shared_batch(&self) -> bool {
        true
    }
    fn shared_prefill_limit(&self) -> usize {
        if self.batch_prefill { self.chunk.min(256) } else { 64 }
    }
    fn shared_prefill_rows(&self) -> usize {
        if self.batch_prefill { self.chunk } else { 1024 }
    }
    fn shared_prefill_max_tokens(&self) -> usize {
        if self.batch_prefill { self.batch_prefill_max_tokens } else { usize::MAX }
    }
    fn shared_prefill_chunk_rows(&self) -> usize {
        if self.batch_prefill { self.chunk.min(256) } else { 0 }
    }
    fn prefill_batch<'s>(
        &'s self,
        rows: &mut [BatchRows<'_, Self::Session<'s>>],
    ) -> Result<Vec<Predictions>, BatchError> {
        if !self.batch_prefill || rows.iter().any(|r| matches!(r.output, BatchOutput::All)) {
            return self.step_batch(rows);
        }
        for row in rows.iter_mut() { row.session.sync_batch(); }
        let tokens: Vec<_> = rows.iter().map(|r| r.tokens).collect();
        let mut sessions: Vec<_> = rows.iter_mut().map(|r| r.session.s_mut())
            .collect::<Result<_, _>>()
            .map_err(|message| BatchError { request: None, message })?;
        NextSession::prefill_batch(self.backend, &mut sessions, &tokens)
            .map(|output| output.into_iter().zip(rows).map(|(logits, row)| {
                Predictions::Logits(if matches!(row.output, BatchOutput::None) {
                    Vec::new()
                } else { vec![logits] })
            }).collect())
            .map_err(|message| BatchError { request: None, message })
    }
    fn single_chat_draft(&self) -> bool {
        true
    }
    fn draft_chat_limit(&self) -> usize {
        self.batch_on.policy(self.cohort.get()).draft_chats
    }
    fn automatic_draft_limit(&self) -> usize {
        self.batch_on.policy(self.cohort.get()).automatic_chats
    }
    fn shared_model_drafts(&self) -> bool {
        true
    }
    fn shared_draft(&self) -> bool {
        true
    }
    /// VADE_MTP_BATCH: each chat's propose_model, with every MTP chain of the tick drafted
    /// together (NextSession::propose_mtp_batch): one MTP layer and drafter head pass per
    /// depth step instead of one per chat and step. Prompt lookup still goes first.
    fn propose_batch<'s>(
        &'s self,
        rows: &mut [crate::DraftRows<'_, Self::Session<'s>>],
    ) -> Result<Vec<spark_engine::draft::ModelProposal>, BatchError> {
        let now = std::time::Instant::now();
        // Complete cohort selection reclaimed unused execution storage before
        // planning. Partial fairness ticks retain their path and never release it.
        let mut out = Vec::with_capacity(rows.len());
        let (mut sessions, mut next, mut depths, mut at) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, r) in rows.iter_mut().enumerate() {
            let chat = &mut *r.session;
            chat.sync_batch();
            let copied = if chat.lookup_max == 0 {
                Vec::new()
            } else {
                chat.lookup
                    .propose(r.history, r.next, r.limit.min(chat.lookup_max))
            };
            if copied.is_empty() {
                at.push(i);
                next.push(r.next);
                depths.push(r.limit.min(chat.mtp_depth_now()));
                sessions.push(chat.s_mut().map_err(|message| BatchError { request: None, message })?);
            }
            out.push(copied);
        }
        let drafted = NextSession::propose_mtp_batch(self.backend, &mut sessions, &next, &depths)
            .map_err(|message| BatchError { request: None, message })?;
        for (i, tokens) in at.into_iter().zip(drafted) {
            out[i] = tokens;
        }
        // The cohort pass is shared; each chat is charged its even share of it.
        let ms = now.elapsed().as_secs_f64() * 1000. / rows.len().max(1) as f64;
        Ok(out
            .into_iter()
            .map(|tokens| spark_engine::draft::ModelProposal {
                tokens,
                proposal_ms: ms,
                catchup_ms: 0.,
            })
            .collect())
    }
    fn greedy_draft_only(&self) -> bool {
        true
    }
    fn sampled_draft(&self) -> bool {
        // VADE_MTP_SAMPLED: sampled requests keep their draft mode (strict, seed-keyed verify).
        self.sampled_draft
    }
    fn sampled_cohort_draft(&self) -> bool {
        self.sampled_cohort
    }
    fn decode_row_limit(&self, _: usize) -> usize {
        64
    }
    fn tick_row_limit(&self) -> Option<usize> {
        let chats = self.cohort.get();
        self.batch_on.policy(chats).tick_rows.map(|_| self.tick_width(chats))
    }
    fn even_cohort_rows(&self) -> bool {
        self.even_rows && self.batch_on.enabled(self.cohort.get())
    }
    fn reserve_foreground_slot(&self) -> bool {
        false
    }
    /// The smallest chat: one token.
    fn session_reservation_bytes(&self) -> Option<usize> {
        kv_charge(&self.pages, self.ahead, 1)
    }
    fn request_reservation_bytes(&self, tokens: usize) -> Option<usize> {
        kv_charge(&self.pages, self.ahead, tokens)
    }
    fn request_reservation_bytes_with_retention(&self, tokens: usize, _plain: bool,
        _cached: Option<usize>, retain: bool) -> Option<usize> {
        let extra = if retain && self.prefix_cache {
            spark_engine::flash_next_rows::prefix_retention_extra_bytes(self.model.mtp_depth() > 0)
        } else { 0 };
        self.request_reservation_bytes(tokens)?.checked_add(extra)
    }
    /// Verify-segment captures (VADE_MTP_GRAPH) take free device bytes above the decode
    /// graph allowance the shared reservation funds; retained cache yields those bytes.
    fn cache_hold_bytes(&self) -> usize {
        if !self.prefix_cache { return 0; }
        // An allowance error leaves no funded graph term: hold all graph storage.
        self.backend.graph_storage_bytes()
            .saturating_sub(self.model.decode_graph_allowance_bytes().unwrap_or(0))
    }
    /// A request admitted without retention keeps main's bound, which charges its live and
    /// rollback GDN states in the legacy form; the chunk-form tails are held from the cache.
    /// Every live request also keeps CACHE_REQUEST_MARGIN_BYTES free.
    fn unfunded_request_bytes(&self, retain: bool) -> usize {
        if !self.prefix_cache { return 0; }
        let tails = if retain { 0 } else {
            spark_engine::flash_next_rows::prefix_retention_extra_bytes(self.model.mtp_depth() > 0)
        };
        tails + CACHE_REQUEST_MARGIN_BYTES
    }
    fn cache_room_bytes(&self) -> usize { self.cache_room }
    /// Current physical room bounds retention. The scheduler also reclaims retained entries
    /// before new request allocations. Unused spare-buffer capacity needs no extra funding.
    fn cache_free_physical_bytes(&self, _live_reserved: usize) -> Result<Option<(i128, usize)>, String> {
        if !self.cache_yield { return Ok(None); }
        let local = crate::prefix_switch::cache_physical_room(self.backend.kv_reserve_room()?, CACHE_YIELD_SLACK_BYTES);
        Ok(crate::prefix_switch::cache_rank_room(local, self.model.pp_cache_room()?, CACHE_YIELD_SLACK_BYTES))
    }
    fn request_option_bytes(&self, request: &crate::protocol::Request) -> usize {
        if !request.prompt_lookup {
            return 0;
        }
        spark_engine::flash_next_lookup::reservation_bytes(
            self.context,
            self.model.vocabulary(),
            self.model.mtp_depth(),
            request.prompt_lookup_max_draft,
        )
    }
    fn memory_bytes(&self) -> Option<(usize, usize, usize)> {
        Some((
            self.backend.allocated_bytes(),
            self.backend.peak_allocated_bytes(),
            self.backend.pinned_host_bytes(),
        ))
    }
    fn fatal_error(&self) -> Option<String> {
        self.backend.fatal_error()
    }
    /// VADE_FLASHNEXT_KV_RESERVE: kept session arenas are released on demand by any allocation,
    /// not after 30 s idle alone: a conversation's next turn often follows a human pause.
    /// Under host memory pressure, spares inside the funded session plan stay until the server
    /// has been idle for 30 s (Backend::fund_session_reserves); other spares are released.
    /// The cap covers at least four sessions and grows to the largest observed cohort. It is
    /// restored after a release.
    /// VADE_FLASHNEXT_KV_RESERVE=0: nothing runs, without a KV reserve (unless the vector
    /// pool is configured: the same calls maintain it).
    fn maintain_memory(&self, idle: bool) -> Result<(), String> {
        if self.prefix_cache { self.model.pp_cache_maintain(idle)?; }
        if !spark_engine::flash::qsa::kv_reserve() && !self.backend.vector_pool_configured() { return Ok(()); }
        self.backend.maintain_kv_reserve(idle)?;
        self.backend.kv_conversation_ended(self.reservation);
        Ok(())
    }
    /// An idle period started before this request arrived must not release funded spares
    /// during its admission.
    fn admission_started(&self) { self.backend.kv_reserve_busy(); }
    fn refresh_memory_pressure(&self) -> Result<(), String> {
        self.backend.refresh_kv_pressure()?;
        // Pressure may clear the reserve and its cap. Restore the existing
        // conversation bound before the next session releases its arenas.
        self.backend.kv_conversation_ended(self.reservation);
        Ok(())
    }
    fn validate_request(
        &self,
        _: &[u32],
        request: &crate::protocol::Request,
    ) -> Result<(), String> {
        if request.prompt_lookup
            && (self.model.mtp_depth() == 0
                || request.mtp_depth == Some(0)
                || request.draft == Some(crate::protocol::DraftMode::Plain)
                || !spark_engine::draft::plain_greedy(request.sampling))
        {
            return Err("prompt lookup requires strict greedy decode and resident MTP".into());
        }
        if request
            .mtp_depth
            .is_some_and(|depth| depth > self.model.mtp_depth())
        {
            return Err("requested mtp_depth exceeds resident model depth".into());
        }
        if request.draft.is_some_and(|mode| {
            !matches!(
                mode,
                crate::protocol::DraftMode::Auto
                    | crate::protocol::DraftMode::Plain
                    | crate::protocol::DraftMode::Strict
            )
        }) {
            return Err("Flash-Next rows support plain or strict greedy decode".into());
        }
        Ok(())
    }
    fn step_batch<'s>(
        &'s self,
        rows: &mut [BatchRows<'_, Self::Session<'s>>],
    ) -> Result<Vec<Predictions>, BatchError> {
        let error = |message| BatchError {
            request: None,
            message,
        };
        if rows.is_empty() || rows.len() > 16 {
            return Err(error("Flash-Next batch must have 1..=16 chats".into()));
        }
        // Every scheduled chat was restored by configure_request; the unwraps below hold.
        if rows.iter().any(|r| r.session.live.is_none()) {
            return Err(error("Flash-Next cached prefix was not restored".into()));
        }
        // VADE_ONE_ROW_POOLED: an all-one-row cohort (nothing drafted) skips the verify rows for
        // the pooled greedy step below; that step also feeds each chat's drafter.
        let chats = self.cohort.get().max(rows.len());
        let (batch_on, defer) = self.batch_on.feed(chats, self.cohort_plain.get());
        for r in rows.iter_mut() {
            let s = r.session.live.as_mut().unwrap();
            s.batch_drafts(batch_on);
            s.defer_draft_feed(defer);
            if self.cohort_cost_control || self.graph_fallback {
                s.configure_pooled_decode(!self.cohort_draft.get(), self.graph_fallback);
            }
        }
        let pooled = !self.cohort_draft.get() && self.one_row_pooled.pooled(chats, rows.len() > 1 && rows.iter().all(|r| r.tokens.len() == 1));
        // VADE_MTP_SAMPLED=2: a cohort with sampled drafting rows verifies every chat in the
        // shared pass; greedy rows take their ids, sampled rows their retained per-row logits
        // (the bytes verify_mtp reads alone), which generation samples at each row's position.
        if !pooled && self.sampled_cohort && rows.iter().any(|r| !r.greedy) && rows.iter().all(|r| {
            (r.greedy || r.session.sampled_draft) && matches!(r.output, BatchOutput::All) && r.session.model_draft_enabled()
        }) {
            let greedy: Vec<_> = rows.iter().map(|r| r.greedy).collect();
            let top_k: Vec<_> = rows.iter().map(|r| if self.sampled_top_k && r.session.sampled_draft { r.top_k } else { 0 }).collect();
            let tokens: Vec<_> = rows.iter().map(|r| r.tokens).collect();
            let mut sessions: Vec<_> = rows.iter_mut().map(|r| r.session.live.as_mut().unwrap()).collect();
            let mut ids = Vec::with_capacity(tokens.len());
            if batch_on {
                let widths: Vec<_> = tokens.iter().map(|t| t.len()).collect();
                for g in crate::mtp_batch::verify_groups(&widths, crate::mtp_batch::VERIFY_ROWS) {
                    ids.extend(NextSession::verify_batch(self.backend, &mut sessions[g.clone()], &tokens[g]).map_err(error)?);
                }
            } else {
                ids = NextSession::verify_batch(self.backend, &mut sessions, &tokens).map_err(error)?;
            }
            return ids.into_iter().zip(greedy).zip(sessions).zip(top_k).map(|(((ids, greedy), s), k)| {
                if greedy { Ok(Predictions::Greedy(ids)) }
                else if k > 0 { s.decode_top_k(self.backend, ids.len(), k).map(Predictions::TopK).map_err(error) }
                else { s.decode_logits(self.backend, ids.len()).map(Predictions::Logits).map_err(error) }
            }).collect();
        }
        if !pooled && rows.iter().all(|r| {
            r.greedy && matches!(r.output, BatchOutput::All) && r.session.model_draft_enabled()
        }) {
            let tokens: Vec<_> = rows.iter().map(|r| r.tokens).collect();
            let mut sessions: Vec<_> = rows.iter_mut().map(|r| r.session.live.as_mut().unwrap()).collect();
            if batch_on {
                // The row budget already holds a tick to tick_width (the verify rows, or 16 for
                // four chats under VADE_MTP_COHORT=wide). Groups use that same width.
                let widths: Vec<_> = tokens.iter().map(|t| t.len()).collect();
                let mut out = Vec::with_capacity(tokens.len());
                for g in crate::mtp_batch::verify_groups(&widths, self.tick_width(chats)) {
                    let ids = NextSession::verify_batch_greedy(
                        self.backend,
                        &mut sessions[g.clone()],
                        &tokens[g],
                    )
                    .map_err(error)?;
                    out.extend(ids.into_iter().map(Predictions::Greedy));
                }
                return Ok(out);
            }
            return NextSession::verify_batch_greedy(self.backend, &mut sessions, &tokens)
                .map(|v| v.into_iter().map(Predictions::Greedy).collect())
                .map_err(error);
        }
        if rows.iter().any(|r| r.tokens.len() != 1) {
            return rows
                .iter_mut()
                .map(|r| {
                    if self.sampled_top_k && r.session.sampled_draft && r.top_k > 0
                        && matches!(r.output, BatchOutput::All) && r.session.model_draft_enabled() {
                        return r.session.live.as_mut().unwrap().verify_mtp_top_k(self.backend, r.tokens, r.top_k)
                            .map(Predictions::TopK).map_err(error);
                    }
                    let result = match r.output {
                        // VADE_MTP_SAMPLED: sampled rows take the same causal verify pass;
                        // generation samples each returned row at its own position.
                        BatchOutput::All if (r.greedy || r.session.sampled_draft) && r.session.model_draft_enabled() => {
                            r.session.live.as_mut().unwrap().verify_mtp(self.backend, r.tokens)
                        }
                        BatchOutput::All => r
                            .tokens
                            .iter()
                            .map(|&t| Session::step(r.session, t))
                            .collect(),
                        BatchOutput::Last => r.session.prefill(r.tokens).map(|v| vec![v]),
                        BatchOutput::None => {
                            r.session.prefill_advance(r.tokens).map(|_| Vec::new())
                        }
                    };
                    result.map(Predictions::Logits).map_err(error)
                })
                .collect();
        }
        let compact = rows
            .iter()
            .all(|r| r.greedy && !matches!(r.output, BatchOutput::None));
        let tokens: Vec<_> = rows.iter().map(|r| r.tokens[0]).collect();
        let mut sessions: Vec<_> = rows.iter_mut().map(|r| r.session.live.as_mut().unwrap()).collect();
        if compact {
            let ids = NextSession::step_batch_greedy(self.backend, &mut sessions, &tokens)
                .map_err(error)?;
            return Ok(ids
                .into_iter()
                .map(|id| Predictions::Greedy(vec![id]))
                .collect());
        }
        let output =
            NextSession::step_batch(self.backend, &mut sessions, &tokens).map_err(error)?;
        Ok(output
            .into_iter()
            .zip(rows)
            .map(|(logits, r)| {
                Predictions::Logits(if matches!(r.output, BatchOutput::None) {
                    Vec::new()
                } else {
                    vec![logits]
                })
            })
            .collect())
    }
}
