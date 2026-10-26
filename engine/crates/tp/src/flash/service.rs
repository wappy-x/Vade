//! Flash rank control with independent sessions, scoped checkpoints and shared weights.
use spark_engine::collective::{Bytes, write_profile};
use spark_engine::flash::session::{DecodeBatch, Exchange, Model, SampleRow, Session, Snapshot, SHARED_PREFIX_TOKENS};
use spark_gpu_probe::driver::{Backend, HostRegistration};
use spark_serve::{BatchError, BatchOutput, BatchRows, Predictions};
use spark_tp::transport::Ring as Commands;
use std::{cell::{Cell, RefCell}, collections::BTreeMap, rc::{Rc, Weak}, sync::{Arc, Mutex}};

use super::settings::CommandState;

type Result<T> = std::result::Result<T, String>;
const OPEN: u64 = 1;
const PREFILL: u64 = 2;
const STEP: u64 = 3;
const CLOSE: u64 = 4;
const STOP: u64 = 5;
const BATCH: u64 = 6;
const SAVE: u64 = 7;
const RESTORE: u64 = 8;
const RELEASE: u64 = 9;
const CANONICALIZE: u64 = 10;
const RESERVE: u64 = 11;
const VERIFY: u64 = 12;
const COMMIT: u64 = 13;
const DISABLE_MTP: u64 = 14;
const DRAFT: u64 = 15;
const CLAIM_MTP: u64 = 16;
const DRAFT_SAMPLED: u64 = 17;
const VERIFY_SAMPLED: u64 = 18;
const DECODE_LOGITS: u64 = 19;
const ENABLE_MTP: u64 = 20;
const AB_ARM: u64 = 21;
pub const TILE: usize = 64;
// The archived client first requests share 5362 tokens across runs. Keep the
// preceding complete GDN chunk, before their generated environment paths.
fn request_capacity(context: usize, tokens: usize) -> usize {
    tokens.max(512).div_ceil(512).saturating_mul(512).min(context)
}

// Sample at allocation milestones and before releasing live sessions. Graph
// bytes are conservative charges; unified availability is a runtime snapshot.
fn memory_receipt(backend: &Backend<'_>, rank: usize, phase: &str, live_sessions: usize,
    sampled_peak: &Cell<usize>, minimum_available: &Cell<usize>, force: bool) {
    let peak = backend.peak_accounted_bytes();
    if !force && peak.saturating_sub(sampled_peak.get()) < 64 << 20 { return; }
    sampled_peak.set(peak);
    let remaining = match backend.available_memory_bytes() {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("flash_serve memory rank={rank} phase={phase} remaining_unified_bytes=unavailable error={error}");
            return;
        }
    };
    minimum_available.set(minimum_available.get().min(remaining));
    eprintln!("flash_serve memory rank={rank} phase={phase} live_sessions={live_sessions} device_bytes={} device_peak_bytes={} graph_bound_bytes={} graph_bound_peak_bytes={} accounted_bytes={} accounted_peak_bytes={peak} remaining_unified_bytes={remaining} sampled_min_unified_bytes={} graph_captures={}",
        backend.allocated_bytes(), backend.peak_allocated_bytes(), backend.graph_storage_bytes(),
        backend.peak_graph_storage_bytes(), backend.accounted_bytes(), minimum_available.get(), backend.stats().graph_captures);
}

fn command_phase(op: u64) -> &'static str {
    match op {
        OPEN => "open", PREFILL => "prefill", STEP => "decode", CLOSE => "close", STOP => "stop",
        BATCH => "decode_batch", SAVE => "save", RESTORE => "restore", RELEASE => "release",
        CANONICALIZE => "canonicalize", RESERVE => "reserve", VERIFY => "verify", COMMIT => "commit",
        DISABLE_MTP => "disable_mtp", DRAFT => "draft", CLAIM_MTP => "claim_mtp",
        DRAFT_SAMPLED => "draft_sampled", VERIFY_SAMPLED => "verify_sampled", DECODE_LOGITS => "decode_logits",
        ENABLE_MTP => "enable_mtp",
        _ => "command",
    }
}
fn command_info(header: &[u8]) -> (&'static str, usize) {
    let word = |i: usize| u64::from_le_bytes(header[i * 8..i * 8 + 8].try_into().unwrap());
    let (op, arg, count) = (word(0), word(2), word(4) as usize);
    let rows = match op {
        BATCH => count / 4, PREFILL | STEP | VERIFY | VERIFY_SAMPLED => count,
        DRAFT | DRAFT_SAMPLED => arg as usize, _ => 0,
    };
    (command_phase(op), rows)
}
// Header and payload use the path selected by TP for their own size. Worker
// header timing includes the idle wait for rank zero and forwarding on a chain.
fn command_part(commands: &mut Commands, data: &Data, bytes: &mut [u8], header: bool) -> Result<()> {
    if !data.1 {
        return if header { commands.command(bytes) } else { commands.command_payload(bytes) };
    }
    let path = commands.command_path(bytes.len());
    let start = std::time::Instant::now();
    let result = if header { commands.command(bytes) } else { commands.command_payload(bytes) };
    let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let (phase, rows) = if header && result.is_ok() { command_info(bytes) } else { ("command", 0) };
    if let Ok(mut ring) = data.0.lock() {
        ring.profile_command(phase, rows, path, ns, Bytes {
            control_sent: if commands.rank() + 1 < commands.world() { bytes.len() as u64 } else { 0 },
            control_received: if commands.rank() != 0 { bytes.len() as u64 } else { 0 },
            ..Bytes::default()
        }, result.is_ok());
    }
    result
}

// One sample for the complete header + payload broadcast, alongside the
// per-part path/byte counters. Keep encoding/receiving inside the same span.
fn command_message<T>(data: &Data, work: impl FnOnce() -> Result<T>) -> Result<T> {
    if !data.1 { return work(); }
    let start = std::time::Instant::now();
    let result = work();
    let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    if let Ok(mut ring) = data.0.lock() { ring.profile_point(2, ns, result.is_ok()); }
    result
}

// The session owns an Exchange while the controller can still abort its DMA.
// There is one compute thread and one transport, not a second weight copy.
#[derive(Clone)]
pub struct Data(Arc<Mutex<spark_seqpar::Ring>>, bool);
impl Data {
    pub fn new(ring: spark_seqpar::Ring) -> Self {
        let profiling = ring.collective_profiling();
        Self(Arc::new(Mutex::new(ring)), profiling)
    }
    fn register_mailbox<'a>(&self, backend: &Backend<'a>) -> Result<Option<Rc<HostRegistration<'a>>>> {
        let mut ring = self.0.lock().map_err(|_| "Flash transport poisoned")?;
        // The controller retains Data until its registration and all sessions drop.
        ring.mailbox_memory().map(|(base, bytes)| unsafe { backend.register_host(base, bytes) }
            .map(Rc::new)).transpose()
    }
    fn abort(&self) { if let Ok(mut ring) = self.0.lock() { ring.abort(); } }
}
impl Exchange for Data {
    fn collective_profiling(&self) -> bool { self.1 }
    fn profile_flash(&mut self, ns: u64) {
        if let Ok(mut ring) = self.0.lock() { ring.profile_flash(ns); }
    }
    fn profile_point(&mut self, kind: usize, ns: u64, ok: bool) {
        if let Ok(mut ring) = self.0.lock() { ring.profile_point(kind, ns, ok); }
    }
    fn profile_begin_step(&mut self, phase: &'static str, rows: usize) {
        if let Ok(mut ring) = self.0.lock() { ring.profile_begin_step(phase, rows); }
    }
    fn profile_end_step(&mut self, ok: bool, device_bytes: usize) {
        if let Ok(mut ring) = self.0.lock() { ring.profile_end_step(ok, device_bytes); }
    }
    fn profile_stats(&self) -> Option<spark_engine::flash::session::ExchangeStats> {
        let stats = self.0.lock().ok()?.stats;
        Some(spark_engine::flash::session::ExchangeStats { elapsed_ns: stats.elapsed_ns,
            sent: stats.sent, received: stats.received, exchanges: stats.exchanges })
    }
    fn gather_bytes(&mut self, values: &mut [u8], offsets: &[usize]) -> Result<()> {
        self.0.lock().map_err(|_| "Flash transport poisoned")?.all_gather_bytes(values, offsets)
    }
    fn gather_f32(&mut self, values: &mut [f32], offsets: &[usize]) -> Result<()> {
        self.0.lock().map_err(|_| "Flash transport poisoned")?.all_gather_f32(values, offsets)
    }
    fn reduce(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        self.0.lock().map_err(|_| "Flash transport poisoned")?.all_reduce_f32(values, scratch)
    }
    fn exchange_slots_pipeline<T>(&mut self, sizes: &[(usize, usize)],
        fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.0.lock().map_err(|_| "Flash transport poisoned")?.exchange_slots_pipeline(sizes, fill, receive, wait)
    }
    fn gather_slots_pipeline<T>(&mut self, offsets: &[Vec<usize>],
        fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.0.lock().map_err(|_| "Flash transport poisoned")?.gather_slots_pipeline(offsets, fill, receive, wait)
    }
}

// Every rank protects admitted growth, including chats not opened on device yet.
fn save_snapshot<'a>(backend: &Backend<'a>, commands: &mut Commands, local: &Session<'a, '_, Data>,
    reserved: usize, allocated: usize, metadata: usize) -> Result<Option<Snapshot<'a>>> {
    let free = backend.device_memory_info().map_or(0, |(free, _)| free)
        .min(backend.available_memory_bytes().unwrap_or(0))
        .min(backend.allocation_budget().saturating_sub(backend.accounted_bytes()));
    let headroom = spark_tp::flash_pool::snapshot_headroom(free, reserved, allocated);
    spark_tp::flash_pool::save_collectively(local.snapshot_bytes().saturating_add(metadata) <= headroom,
        |ok| {
            let mut votes = vec![0; commands.world()]; votes[commands.rank()] = u8::from(ok);
            commands.all_gather(&mut votes, 1)?;
            Ok(votes.iter().all(|&v| v == 1))
        }, || local.snapshot(backend))
}

pub struct Engine<'m, 'b, 'a> {
    // Unregister before dropping the last transport owner.
    mailbox: Option<Rc<HostRegistration<'a>>>,
    model: &'m Model<'a>,
    backend: &'b Backend<'a>,
    commands: RefCell<Commands>,
    data: Data,
    context: usize,
    vocab: usize,
    eos: Vec<u32>,
    next: Cell<u64>,
    active: Cell<usize>,
    slots: usize,
    batch: RefCell<DecodeBatch<'a, 'm>>,
    failure: RefCell<Option<String>>,
    state: RefCell<CommandState>,
    previous_state: RefCell<Option<CommandState>>,
    cache_budget: usize,
    retained: Cell<usize>,
    admitted: RefCell<BTreeMap<u64, (usize, Weak<RefCell<Option<Session<'a, 'm, Data>>>>)>>,
    reservation: usize,
    workspace_slots: usize,
    drafter: RefCell<Option<(u64, Weak<RefCell<Option<Session<'a, 'm, Data>>>>)>>,
    draft_ready: Cell<bool>,
    sampled_peak: Cell<usize>,
    minimum_available: Cell<usize>,
    memory_phase: Cell<&'static str>,
    // Some: one chat drafts the chooser's depth each cycle (settings::rank_mtp).
    draft_costs: Option<spark_engine::draft::CostCurve>,
    // The last eight measured ms of one verify pass (and how many were seen) by rows - 1,
    // shared by every request.
    verify_ms: RefCell<[(usize, [f64; 8]); 16]>,
    // Model-draft survival by depth over recent verified cycles of every request (trials,
    // hits, each decayed by 0.99 per observation), and the verification awaiting its commit.
    survival: RefCell<[(f64, f64); 16]>,
    verifying: Cell<Option<(u64, usize, usize)>>,
}
/// Verify milliseconds for 1/2/4/8/16 rows from the rank's byte ledger at GB10's 273 GB/s;
/// the draft policy learns the measured scale of this shape online.
fn verify_costs(model: &Model<'_>) -> spark_engine::draft::CostCurve {
    let (fixed, row) = model.verify_byte_ledger();
    spark_engine::draft::CostCurve([1, 2, 4, 8, 16].map(|rows| (fixed + rows * row) as f64 / 273e6))
}
impl<'m, 'b, 'a> Engine<'m, 'b, 'a> {
    pub fn new(model: &'m Model<'a>, backend: &'b Backend<'a>, commands: Commands,
        data: Data, context: usize, vocab: usize, eos: Vec<u32>, slots: usize, cache_budget: usize, reservation: usize, workspace_slots: usize, adaptive: bool) -> Result<Self> {
        if commands.rank() != 0 { return Err("Flash controller must be rank zero".into()); }
        if slots == 0 { return Err("invalid Flash session limit".into()); }
        let mailbox = data.register_mailbox(backend)?;
        if workspace_slots < model.workspace_slots(mailbox.is_some()) { return Err("Flash workspace reservation omits this rank".into()); }
        let local = model.request_reservation_bytes(context, TILE, workspace_slots, cache_budget != 0)?;
        if reservation < local { return Err("negotiated Flash reservation does not cover this rank".into()); }
        let batch = RefCell::new(DecodeBatch::new(backend, model)?);
        let draft_costs = (adaptive && model.mtp_depth() != 0).then(|| verify_costs(model));
        if let Some(costs) = draft_costs {
            let (fixed, row) = model.verify_byte_ledger();
            eprintln!("flash_serve adaptive_draft resident_depth={} ledger_fixed_bytes={fixed} ledger_row_bytes={row} prior_ms={:?}",
                model.mtp_depth(), costs.0);
        }
        Ok(Self { mailbox, model, backend, commands: RefCell::new(commands), data, context, vocab, eos,
            next: Cell::new(0), active: Cell::new(0), slots, batch,
            failure: RefCell::new(None), state: RefCell::new(CommandState::default()),
            previous_state: RefCell::new(None), cache_budget, retained: Cell::new(0), admitted: RefCell::new(BTreeMap::new()), reservation, workspace_slots, drafter: RefCell::new(None),
            draft_ready: Cell::new(false),
            sampled_peak: Cell::new(backend.peak_accounted_bytes()), minimum_available: Cell::new(usize::MAX),
            memory_phase: Cell::new("setup"), draft_costs, verify_ms: Default::default(),
            survival: Default::default(), verifying: Cell::new(None) })
    }
    // Only this owner may retain the globally funded draft/verify workspace.
    // Hold it for the active turn; other sessions use canonical single rows.
    // Weak ownership does not keep closed sessions or their target caches alive.
    fn drafter_available(&self, id: u64) -> bool {
        self.drafter.borrow().as_ref().is_none_or(|(current, session)| *current == id
            || session.upgrade().is_none_or(|session| session.borrow().is_none()))
    }
    fn claim_drafter(&self, id: u64, local: &Rc<RefCell<Option<Session<'a, 'm, Data>>>>) -> Result<bool> {
        if !self.drafter_available(id) { return Ok(false); }
        if self.draft_ready.get() && self.drafter.borrow().as_ref().is_some_and(|(current, _)| *current == id) { return Ok(true); }
        self.command(CLAIM_MTP, id, 0, false, &[])?;
        let ready = self.result(local.borrow_mut().as_mut().ok_or("missing Flash drafter")?
            .prepare_mtp_workspace(self.backend))?;
        *self.drafter.borrow_mut() = ready.then(|| (id, Rc::downgrade(local)));
        self.draft_ready.set(ready);
        Ok(ready)
    }
    fn release_drafter(&self, id: u64) {
        let mut owner = self.drafter.borrow_mut();
        if owner.as_ref().is_some_and(|(current, _)| *current == id) {
            *owner = None;
            self.draft_ready.set(false);
        }
    }
    fn result<T>(&self, result: Result<T>) -> Result<T> {
        if let Err(error) = &result {
            if error.starts_with("Flash MoE routing rejected") {
                if let Some(previous) = self.previous_state.borrow_mut().take() { *self.state.borrow_mut() = previous; }
                return result;
            }
            spark_gpu_probe::teardown::arm();
            if self.failure.borrow().is_none() {
                *self.failure.borrow_mut() = Some(error.clone());
                self.commands.borrow_mut().abort();
                self.data.abort();
            }
        }
        if result.is_ok() {
            memory_receipt(self.backend, 0, self.memory_phase.get(), self.active.get(), &self.sampled_peak, &self.minimum_available, false);
        }
        result
    }
    fn command(&self, op: u64, id: u64, arg: u64, logits: bool, tokens: &[u32]) -> Result<()> {
        if let Some(error) = self.failure.borrow().as_ref() { return Err(error.clone()); }
        self.state.borrow().validate(op, id, arg, logits, tokens, self.context, self.vocab, self.slots, self.model.mtp_depth())?;
        let result = command_message(&self.data, || {
            let mut header = [0u8; 40];
            for (dst, n) in header.chunks_exact_mut(8).zip([op, id, arg, u64::from(logits), tokens.len() as u64]) {
                dst.copy_from_slice(&n.to_le_bytes());
            }
            let mut commands = self.commands.borrow_mut();
            command_part(&mut commands, &self.data, &mut header, true)?;
            if !tokens.is_empty() {
                let mut bytes = tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>();
                command_part(&mut commands, &self.data, &mut bytes, false)?;
            }
            Ok(())
        });
        if result.is_ok() {
            *self.previous_state.borrow_mut() = Some(self.state.borrow().clone());
            self.state.borrow_mut().apply(op, id, arg, logits, tokens);
            self.memory_phase.set(command_phase(op));
        }
        self.result(result)
    }
    fn next_id(&self) -> Result<u64> {
        let id = self.next.get().checked_add(1).ok_or("Flash session id overflow")?;
        self.next.set(id);
        Ok(id)
    }
    pub fn shutdown(&self) -> Result<()> { self.command(STOP, 0, 0, false, &[]) }
}

struct Saved<'e, 'm, 'b, 'a> {
    engine: &'e Engine<'m, 'b, 'a>,
    id: u64,
    state: Snapshot<'a>,
    tokens: Vec<u32>,
    shared: bool,
}
impl Saved<'_, '_, '_, '_> {
    fn bytes(&self) -> usize { self.state.bytes() + self.tokens.capacity() * 4 + std::mem::size_of::<Self>() }
}
impl Drop for Saved<'_, '_, '_, '_> {
    fn drop(&mut self) {
        self.engine.retained.set(self.engine.retained.get() - self.bytes());
        let _ = self.engine.command(RELEASE, self.id, 0, false, &[]);
    }
}

/// Prompt snapshots kept beside a new response: the shared prefix for other
/// prompts, and the longest longer prompt snapshot for an exact repeat.
fn kept_prompts(saved: &[(usize, bool)], end: usize) -> Vec<usize> {
    let shared = saved.iter().position(|&(_, shared)| shared);
    let floor = shared.map_or(0, |i| saved[i].0);
    let prompt = saved.iter().enumerate().filter(|&(_, &(n, shared))| !shared && n > floor && n <= end)
        .max_by_key(|&(_, &(n, _))| n).map(|(i, _)| i);
    shared.into_iter().chain(prompt).collect()
}

pub struct RemoteSession<'e, 'm, 'b, 'a> {
    engine: &'e Engine<'m, 'b, 'a>,
    local: Rc<RefCell<Option<Session<'a, 'm, Data>>>>,
    id: u64,
    capacity: usize,
    tokens: Vec<u32>,
    draft_disabled: bool,
    sampled: bool,
    cache_retention: bool,
    saved: Vec<Rc<Saved<'e, 'm, 'b, 'a>>>,
    // Weak aliases deduplicate publication without retaining cache-owned copies.
    published: RefCell<Vec<(usize, Weak<Saved<'e, 'm, 'b, 'a>>)>>,
    revision: Cell<usize>,
    cached_prompt_tokens: Option<usize>,
    // The adaptive chooser's model depth for the next proposal (limit_model_depth).
    depth_cap: usize,
}
/// A retained candidate, a new prompt, and its completion replacement may
/// coexist during publication. Fund their private metadata outside the active
/// request reservation; all retained snapshot identities are cache-funded.
pub fn publication_metadata_bytes(tokens: usize, vocab: usize, header_bytes: usize) -> Result<usize> {
    // Live token vectors can grow to less than twice their logical length.
    // kept_prompts keeps at most two snapshots; appending a response makes
    // three identities in a saved vector of capacity at most four.
    let private = tokens.checked_mul(8)
        .and_then(|n| n.checked_add(std::mem::size_of::<RemoteSession<'_, '_, '_, '_>>()))
        .and_then(|n| n.checked_add(4 * std::mem::size_of::<Rc<Saved<'_, '_, '_, '_>>>()))
        .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<RefCell<Option<Session<'_, '_, Data>>>>()));
    // Authorization is bounded by HTTP headers; cache salt is at most 256
    // bytes. Leave room for namespace length framing and Bearer normalization.
    private.and_then(|private| header_bytes.checked_add(512).and_then(|scope|
        spark_serve::cache::publication_metadata_bytes(tokens, vocab, private, 3, scope)))
        .and_then(|n| n.checked_mul(3)).ok_or_else(|| "Flash publication metadata overflow".into())
}
impl<'e, 'm, 'b, 'a> RemoteSession<'e, 'm, 'b, 'a> {
    fn reserve(&mut self) -> Result<()> {
        self.engine.admitted.borrow_mut().insert(self.id, (self.capacity, Rc::downgrade(&self.local)));
        let mut local = self.local.borrow_mut();
        if let Some(local) = local.as_mut() {
            if self.capacity > local.capacity() {
                self.engine.command(RESERVE, self.id, self.capacity as u64, false, &[])?;
                self.engine.result(local.reserve(self.engine.backend, self.capacity))?;
                // Growth releases verification scratch, but committed history
                // still belongs to this owner until disable or close.
                if self.engine.drafter.borrow().as_ref().is_some_and(|(id, _)| *id == self.id) {
                    self.engine.draft_ready.set(false);
                }
            }
        }
        Ok(())
    }
    fn save(&self, tokens: &[u32], shared: bool) -> Result<Option<Rc<Saved<'e, 'm, 'b, 'a>>>> {
        let revision = self.revision.get();
        let mut published = self.published.borrow_mut();
        published.retain(|(_, saved)| saved.strong_count() != 0);
        // Checkpoint and automatic-cache publication can fork the same state.
        // Share its immutable copy rather than exceed the two-turn bound.
        if let Some(saved) = published.iter().filter(|(epoch, _)| *epoch == revision)
            .filter_map(|(_, saved)| saved.upgrade()).find(|saved| saved.shared == shared && saved.tokens == tokens) {
            return Ok(Some(saved));
        }
        // Cache capacity is global; physical funding is voted on every rank.
        let bytes = self.local.borrow().as_ref().ok_or("missing Flash session")?.snapshot_bytes()
            + tokens.len() * 4 + std::mem::size_of::<Saved<'_, '_, '_, '_>>();
        if bytes > self.engine.cache_budget.saturating_sub(self.engine.retained.get()) { return Ok(None); }
        let reserved = self.engine.admitted.borrow().values().try_fold(0usize, |sum, (capacity, _)|
            self.engine.model.request_reservation_bytes(*capacity, TILE, self.engine.workspace_slots, self.engine.cache_budget != 0)
                .and_then(|bytes| sum.checked_add(bytes).ok_or_else(|| "Flash admitted bytes overflow".into())))?;
        if self.local.borrow().as_ref().is_some_and(Session::has_mtp_history) {
            let mut local = self.local.borrow_mut();
            let local = local.as_mut().ok_or("missing Flash session")?;
            self.engine.command(COMMIT, self.id, local.position() as u64, false, &[])?;
            self.engine.result(local.commit_verification(self.engine.backend, local.position()))?;
        }
        let id = self.engine.next_id()?;
        self.engine.command(SAVE, id, self.id, false, &[reserved as u32, (reserved as u64 >> 32) as u32])?;
        // Only private target KV/history is credited on either side. This
        // deliberately over-reserves scratch, graphs and allocation overlap.
        let allocated = self.engine.admitted.borrow().values().filter_map(|(_, local)| local.upgrade())
            .map(|local| local.borrow().as_ref().map_or(0, Session::private_cache_bytes)).sum();
        let attempt = save_snapshot(self.engine.backend, &mut self.engine.commands.borrow_mut(),
            self.local.borrow().as_ref().ok_or("missing Flash session")?, reserved, allocated,
            tokens.len() * 4 + std::mem::size_of::<Saved<'_, '_, '_, '_>>());
        let Some(state) = self.engine.result(attempt)? else {
            self.engine.state.borrow_mut().saved.remove(&id);
            return Ok(None);
        };
        self.engine.state.borrow_mut().saved.get_mut(&id).unwrap().1 = state.prefix_len();
        let barrier = self.engine.commands.borrow_mut().barrier();
        self.engine.result(barrier)?;
        memory_receipt(self.engine.backend, 0, "save", self.engine.active.get(), &self.engine.sampled_peak, &self.engine.minimum_available, true);
        let saved = Rc::new(Saved { engine: self.engine, id, state, tokens: tokens.to_vec(), shared });
        self.engine.retained.set(self.engine.retained.get() + saved.bytes());
        published.push((revision, Rc::downgrade(&saved)));
        Ok(Some(saved))
    }
    fn run(&mut self, tokens: &[u32], prefill: bool, logits: bool) -> Result<spark_engine::flash::session::Output> {
        if tokens.is_empty() || (!prefill && tokens.len() != 1) || tokens.len() > self.engine.context.saturating_sub(self.tokens.len())
            || tokens.iter().any(|&t| t as usize >= self.engine.vocab) {
            return Err("Flash input outside session, vocabulary or context".into());
        }
        let mut input = tokens.to_vec();
        let mut position = self.tokens.len();
        self.capacity = self.capacity.max(request_capacity(self.engine.context, position + tokens.len()));
        self.reserve()?;
        if self.local.borrow().is_none() {
            if self.engine.active.get() >= self.engine.slots { return Err("Flash session limit reached".into()); }
            let mut prompt = self.tokens.clone();
            if prefill { prompt.extend_from_slice(tokens); }
            self.saved.retain(|saved| {
                let n = if prefill { saved.state.prefix_len() } else { saved.tokens.len() };
                saved.tokens.len() <= self.capacity && n <= prompt.len() && prompt.starts_with(&saved.tokens[..n])
            });
            // Only a complete canonical boundary may precede new prompt work.
            // A decode restart instead restores the exact prompt-end state.
            let best = self.saved.iter().filter_map(|saved| {
                let n = if prefill { saved.state.prefix_len() } else { saved.tokens.len() };
                (n > 0 && n <= prompt.len() && (!prefill || n < prompt.len())
                    && prompt.starts_with(&saved.tokens[..n])).then_some((n, saved))
            }).max_by_key(|(n, _)| *n);
            // Every admission starts with ordinary target execution. Only the
            // single-chat owner may allocate and maintain MTP history.
            self.engine.command(OPEN, self.id, self.capacity as u64, false, &[])?;
            let mut local = self.engine.result(self.engine.model.session_plain_with_mailbox(self.engine.backend,
                vec![self.engine.data.clone()], self.capacity, TILE, self.engine.mailbox.clone()))?;
            if self.engine.cache_budget != 0 { local.enable_cache(); }
            if !self.draft_disabled && self.engine.active.get() == 0
                && self.engine.model.mtp_depth() != 0 && self.engine.drafter_available(self.id) {
                self.engine.command(ENABLE_MTP, self.id, 0, false, &[])?;
                if self.engine.result(local.enable_mtp_history(self.engine.backend))? {
                    *self.engine.drafter.borrow_mut() = Some((self.id, Rc::downgrade(&self.local)));
                    self.engine.draft_ready.set(false);
                }
            }
            let reused = if let Some((n, saved)) = best {
                self.engine.command(RESTORE, self.id, saved.id, prefill, &[])?;
                self.engine.result(local.restore(self.engine.backend, &saved.state, prefill))?;
                n
            } else { 0 };
            if !local.has_mtp_history() { self.engine.release_drafter(self.id); }
            self.cached_prompt_tokens = Some(reused);
            self.engine.active.set(self.engine.active.get() + 1);
            *self.local.borrow_mut() = Some(local);
            // RESTORE copied the selected state into private execution. Drop
            // inherited turn snapshots so the new prompt/response copies fit
            // the same two-turn bound; their cache owners retain their aliases.
            self.saved.retain(|saved| saved.shared);
            if prefill { input = prompt[reused..].to_vec(); }
            else if reused != prompt.len() {
                self.forward(&prompt[reused..], true, false)?;
            }
            if prefill { position = reused; }
            eprintln!("flash_cache session={} matched_tokens={} prompt_tokens={} retained_snapshots={} device_bytes={} mtp_depth={}",
                self.id, reused, prompt.len(), self.saved.len(), self.engine.backend.allocated_bytes(),
                self.local.borrow().as_ref().unwrap().mtp_depth());
        }
        // Retention only controls this optional snapshot. Keep local/worker
        // canonical replay enabled so partial prefill chunks stay identical.
        let output = if prefill && self.cache_retention && self.engine.cache_budget != 0
            && position < SHARED_PREFIX_TOKENS && position + input.len() >= SHARED_PREFIX_TOKENS
            && !self.saved.iter().any(|s| s.shared) {
            let split = SHARED_PREFIX_TOKENS - position;
            let output = self.forward(&input[..split], true, logits && split == input.len())?;
            if spark_serve::Session::retained_bytes(self).unwrap_or(0) <= self.engine.cache_budget {
                let mut prefix = self.tokens.clone();
                prefix.extend_from_slice(tokens);
                if let Some(saved) = self.save(&prefix[..SHARED_PREFIX_TOKENS], true)? {
                    eprintln!("flash_shared_prefix session={} tokens={} retained_bytes={} cache_budget_bytes={}",
                        self.id, saved.tokens.len(), saved.bytes(), self.engine.cache_budget);
                    self.saved.push(saved);
                }
            }
            if split == input.len() { output } else { self.forward(&input[split..], true, logits)? }
        } else { self.forward(&input, prefill, logits)? };
        self.tokens.extend_from_slice(tokens);
        Ok(output)
    }
    fn forward(&self, tokens: &[u32], prefill: bool, logits: bool) -> Result<spark_engine::flash::session::Output> {
        self.engine.command(if prefill { PREFILL } else { STEP }, self.id, 0, logits, tokens)?;
        let mut local = self.local.borrow_mut();
        let local = local.as_mut().ok_or("missing Flash session")?;
        let output = self.engine.result(if prefill { local.forward_prefill(self.engine.backend, tokens, logits) }
            else { local.forward(self.engine.backend, tokens, logits) })?;
        self.revision.set(self.revision.get().wrapping_add(1));
        if self.engine.model.layer_profiling() {
            write_profile(&format!("flash_serve forward session={} input_tokens={} elapsed_ms={:.3} device_bytes={} device_peak_bytes={}\n",
                self.id, tokens.len(), output.elapsed_ms,
                self.engine.backend.allocated_bytes(), self.engine.backend.peak_allocated_bytes()));
        }
        Ok(output)
    }
    fn close(&mut self) -> Result<()> {
        self.engine.admitted.borrow_mut().remove(&self.id);
        if self.local.borrow().is_none() { return Ok(()); }
        memory_receipt(self.engine.backend, 0, "close", self.engine.active.get(), &self.engine.sampled_peak, &self.engine.minimum_available, true);
        let result = self.engine.command(CLOSE, self.id, 0, false, &[]);
        if let Some(local) = self.local.borrow_mut().take() { local.report(); drop(local); }
        self.engine.release_drafter(self.id);
        self.engine.active.set(self.engine.active.get() - 1);
        result?;
        let result = self.engine.commands.borrow_mut().barrier();
        self.engine.result(result)?;
        eprintln!("flash_serve session={} closed device_bytes={} device_peak_bytes={}",
            self.id, self.engine.backend.allocated_bytes(), self.engine.backend.peak_allocated_bytes());
        Ok(())
    }
}
impl Drop for RemoteSession<'_, '_, '_, '_> {
    fn drop(&mut self) { let _ = self.close(); }
}
impl RemoteSession<'_, '_, '_, '_> {
    fn chosen_depth(&self) -> usize {
        if spark_engine::draft::Target::adaptive_model_depth(self) { self.depth_cap } else { usize::MAX }
    }
    // The adaptive chooser's row costs: one verify pass's wall time on this rank (command,
    // every rank's pass and the result).
    fn observe_verify(&self, rows: usize, start: std::time::Instant) {
        if self.engine.draft_costs.is_none() { return; }
        if let Some((seen, ms)) = self.engine.verify_ms.borrow_mut().get_mut(rows - 1) {
            ms[*seen % ms.len()] = start.elapsed().as_secs_f64() * 1000.0;
            *seen += 1;
        }
        self.engine.verifying.set(Some((self.id, self.tokens.len(), rows)));
    }
    // The commit after a verification: its first row is the target's, the rest proposals.
    fn observe_commit(&self, position: usize) {
        let Some((id, entry, rows)) = self.engine.verifying.take() else { return; };
        if id != self.id || position <= entry || position > entry + rows { return; }
        let accepted = position - entry - 1;
        for (depth, (trials, hits)) in self.engine.survival.borrow_mut().iter_mut().take(rows - 1).enumerate() {
            *trials = 0.99 * *trials + 1.0;
            *hits = 0.99 * *hits + f64::from(u8::from(depth < accepted));
            if depth >= accepted { break; }
        }
    }
    fn verify_sampled(&mut self, tokens: &[u32], k: usize) -> Result<Predictions> {
        if tokens.is_empty() || tokens.len() > self.engine.model.mtp_depth() + 1
            || tokens.len() > self.engine.context - self.tokens.len() || k > 64
            || tokens.iter().any(|&t| t as usize >= self.engine.vocab) {
            return Err("invalid Flash sampled verification input".into());
        }
        self.sampled = false;
        if tokens.len() == 1 {
            return self.run(tokens, false, true).and_then(|v|
                v.logits.ok_or_else(|| "missing Flash sampled logits".into()))
                .map(|v| Predictions::Logits(vec![v]));
        }
        if self.local.borrow().is_none() { return Err("Flash verification requires an open session".into()); }
        self.capacity = self.capacity.max(request_capacity(self.engine.context, self.tokens.len() + tokens.len()));
        self.reserve()?;
        if self.engine.drafter.borrow().as_ref().is_none_or(|(owner, _)| *owner != self.id) {
            return Err("Flash verifier has no draft workspace".into());
        }
        let start = std::time::Instant::now();
        self.engine.command(VERIFY_SAMPLED, self.id, k as u64, false, tokens)?;
        let output = self.engine.result(self.local.borrow_mut().as_mut().ok_or("missing Flash sampled verifier")?
            .verify_mtp_sampled(self.engine.backend, tokens, k))?;
        self.observe_verify(tokens.len(), start);
        self.revision.set(self.revision.get().wrapping_add(1));
        self.tokens.extend_from_slice(tokens);
        if k != 0 { output.into_iter().map(|row| match row {
            SampleRow::TopK(values) => Ok(values), _ => Err("missing Flash compact verifier row".into()),
        }).collect::<Result<Vec<_>>>().map(Predictions::TopK) }
        else { output.into_iter().map(|row| match row {
            SampleRow::Logits(values) => Ok(values), _ => Err("missing Flash full verifier row".into()),
        }).collect::<Result<Vec<_>>>().map(Predictions::Logits) }
    }
    fn verify(&mut self, tokens: &[u32], logits: bool) -> Result<Vec<spark_engine::flash::session::Output>> {
        if tokens.is_empty() || tokens.len() > self.engine.model.mtp_depth() + 1
            || tokens.len() > self.engine.context - self.tokens.len()
            || tokens.iter().any(|&t| t as usize >= self.engine.vocab) { return Err("invalid Flash verification input".into()); }
        if tokens.len() == 1 { return self.run(tokens, false, logits).map(|v| vec![v]); }
        if self.local.borrow().is_none() { return Err("Flash verification requires an open session".into()); }
        self.capacity = self.capacity.max(request_capacity(self.engine.context, self.tokens.len() + tokens.len()));
        self.reserve()?;
        // Draft planning already claimed and funded every verification row.
        // Never steal another request's pending rollback or allocate here.
        if self.engine.drafter.borrow().as_ref().is_none_or(|(owner, _)| *owner != self.id) {
            return Err("Flash verifier has no draft workspace".into());
        }
        let start = std::time::Instant::now();
        self.engine.command(VERIFY, self.id, 0, logits, tokens)?;
        let output = self.engine.result(self.local.borrow_mut().as_mut().ok_or("missing Flash verifier")?
            .verify_mtp(self.engine.backend, tokens, logits))?;
        self.observe_verify(tokens.len(), start);
        self.revision.set(self.revision.get().wrapping_add(1));
        self.tokens.extend_from_slice(tokens);
        Ok(output)
    }
}
impl spark_engine::draft::Target for RemoteSession<'_, '_, '_, '_> {
    fn checkpoint(&self) -> usize { self.tokens.len() }
    fn truncate(&mut self, position: usize) -> Result<()> {
        self.engine.command(COMMIT, self.id, position as u64, false, &[])?;
        self.engine.result(self.local.borrow_mut().as_mut().ok_or("missing Flash verifier")?
            .commit_verification(self.engine.backend, position))?;
        self.observe_commit(position);
        self.revision.set(self.revision.get().wrapping_add(1));
        self.tokens.truncate(position);
        if self.draft_disabled { spark_serve::Session::disable_model_draft(self); }
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> { spark_serve::Session::step(self, token) }
    fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.verify(tokens, true)?.into_iter().map(|o| o.logits.ok_or_else(|| "missing Flash verify logits".into())).collect()
    }
    fn max_verify_rows(&self) -> usize { self.model_draft_width() + 1 }
    fn model_draft_width(&self) -> usize {
        if self.draft_disabled || !self.engine.drafter_available(self.id) { 0 }
        else { self.local.borrow().as_ref().map_or(0, Session::mtp_depth) }
    }
    fn native_draft_policy(&self) -> bool { true }
    // One chat alone (the only drafter) picks its depth per cycle, up to the resident depth.
    fn adaptive_model_depth(&self) -> bool { self.engine.draft_costs.is_some() && self.engine.active.get() <= 1 }
    fn limit_model_depth(&mut self, depth: usize) { self.depth_cap = depth; }
    fn model_draft_costs(&self) -> Option<spark_engine::draft::CostCurve> { self.engine.draft_costs }
    // The median of the last (up to eight) passes, once three exist: a first pass (setup) or
    // one stalled pass never decides a width's cost.
    fn model_verify_ms(&self, rows: usize) -> Option<f64> {
        self.engine.draft_costs?;
        let (seen, ms) = *self.engine.verify_ms.borrow().get(rows.checked_sub(1)?)?;
        let mut recent = ms[..seen.min(ms.len())].to_vec();
        recent.sort_by(f64::total_cmp);
        (seen >= 3).then(|| recent[recent.len() / 2])
    }
    // Shrunk toward 0.8 like a request's own counts (4 pseudo-trials), so a depth few recent
    // cycles reached (one smoke miss) can never pin every later request below it.
    fn model_survival_prior(&self, depth: usize) -> Option<f64> {
        self.engine.draft_costs?;
        let (trials, hits) = *self.engine.survival.borrow().get(depth)?;
        Some((hits + 3.2) / (trials + 4.0))
    }
    fn sampled_model_draft(&self) -> bool { true }
    fn speculative_model_draft(&self) -> bool { true }
    fn propose_sampled_model(&mut self, history: &[u32], prompt_len: usize, next: u32,
        limit: usize, sampling: spark_engine::sample::Sampling, allowed: &dyn Fn(u32) -> bool)
        -> Result<(spark_engine::draft::ModelProposal, Vec<spark_engine::sample::Distribution>)> {
        use spark_engine::sample::{Sampler, speculative_uniform};
        if history != self.tokens || prompt_len > history.len() { return Err("invalid sampled MTP history".into()); }
        let depth = limit.min(self.model_draft_width()).min(self.chosen_depth())
            .min(self.engine.context.saturating_sub(self.tokens.len() + 1));
        let start = std::time::Instant::now();
        let k = sampling.compact_top_k();
        if k == 0 { return Err("sampled MTP requires top_k 1..64 without history penalties".into()); }
        self.sampled = true;
        let mut distributions = Vec::with_capacity(depth);
        let mut prefix = history.to_vec();
        prefix.push(next);
        let mut sampler = Sampler::new(sampling)?;
        let mut sampling_error = None;
        let tokens = if depth == 0 || !self.engine.claim_drafter(self.id, &self.local)? { Vec::new() }
        else {
            self.engine.command(DRAFT_SAMPLED, self.id, depth as u64, false, &[next, k as u32])?;
            self.engine.result(self.local.borrow_mut().as_mut().ok_or("missing Flash sampled drafter")?
                .propose_mtp_sampled(self.engine.backend, next, depth, k, |row, _| {
                    // Finish the collective chain on a host sampling error.
                    // Its placeholders never reach verification or output.
                    if sampling_error.is_some() { return Ok(Some(0)); }
                    let choice = (|| -> Result<Option<_>> {
                        let q = match row {
                            SampleRow::TopK(values) => {
                                let Some(q) = sampler.distribution_top_k(&values, allowed) else { return Ok(None); };
                                q
                            }
                            SampleRow::Logits(values) => sampler.distribution(&values, &prefix, prompt_len, allowed)?,
                        };
                        let token = q.draw(speculative_uniform(sampling.seed, prefix.len(), 0))?;
                        Ok(Some((token, q)))
                    })();
                    match choice {
                        Ok(Some((token, q))) => { prefix.push(token); distributions.push(q); Ok(Some(token)) }
                        Ok(None) => Ok(None),
                        Err(error) => { sampling_error = Some(error); Ok(Some(0)) }
                    }
                }))?
        };
        if let Some(error) = sampling_error { return Err(error); }
        Ok((spark_engine::draft::ModelProposal { tokens, proposal_ms: start.elapsed().as_secs_f64() * 1000.0,
            catchup_ms: 0.0 }, distributions))
    }
    fn model_draft_budget(&mut self, remaining: usize) -> Result<()> {
        let depth = self.model_draft_width().min(remaining.saturating_sub(1))
            .min(self.engine.context.saturating_sub(self.tokens.len() + 1));
        if depth != 0 {
            self.capacity = self.capacity.max(request_capacity(self.engine.context, self.tokens.len() + depth + 1));
            self.reserve()?;
            self.engine.claim_drafter(self.id, &self.local)?;
        }
        Ok(())
    }
    fn propose_model(&mut self, _: &[u32], next: u32, limit: usize) -> Result<Option<spark_engine::draft::ModelProposal>> {
        self.sampled = false;
        let depth = limit.min(self.model_draft_width()).min(self.chosen_depth())
            .min(self.engine.context.saturating_sub(self.tokens.len() + 1));
        let start = std::time::Instant::now();
        let tokens = if depth == 0 { Vec::new() } else {
            self.capacity = self.capacity.max(request_capacity(self.engine.context, self.tokens.len() + depth + 1));
            self.reserve()?;
            if !self.engine.claim_drafter(self.id, &self.local)? {
                return Ok(Some(spark_engine::draft::ModelProposal { tokens: Vec::new(),
                    proposal_ms: start.elapsed().as_secs_f64() * 1000.0, catchup_ms: 0.0 }));
            }
            self.engine.command(DRAFT, self.id, depth as u64, false, &[next])?;
            self.engine.result(self.local.borrow_mut().as_mut().ok_or("missing Flash drafter")?
                .propose_mtp(self.engine.backend, next, depth))?
        };
        Ok(Some(spark_engine::draft::ModelProposal { tokens, proposal_ms: start.elapsed().as_secs_f64() * 1000.0, catchup_ms: 0.0 }))
    }
}
impl spark_serve::Session for RemoteSession<'_, '_, '_, '_> {
    fn set_cache_retention(&mut self, retain: bool) { self.cache_retention = retain; }
    fn cached_prompt_tokens(&self, prompt_tokens: usize) -> Option<usize> {
        // A full hit can serve cached logits before opening execution scratch.
        // A partial match is credited only once RESTORE has actually succeeded.
        Some(self.cached_prompt_tokens.unwrap_or_else(||
            if self.tokens.len() == prompt_tokens { prompt_tokens } else { 0 }))
    }
    // Admission funds target execution; one complete drafter is funded globally.
    // The scheduler calls this hook for plain requests and output-window growth too.
    fn reserve_model_draft(&mut self, tokens: usize) {
        self.capacity = self.capacity.max(request_capacity(self.engine.context, tokens));
        self.engine.admitted.borrow_mut().insert(self.id, (self.capacity, Rc::downgrade(&self.local)));
    }
    fn disable_model_draft(&mut self) {
        self.draft_disabled = true;
        if let Some(local) = self.local.borrow_mut().as_mut() {
            if !local.has_mtp_history() {
                self.engine.release_drafter(self.id);
                return;
            }
            if self.engine.command(DISABLE_MTP, self.id, 0, false, &[]).is_ok() {
                if self.engine.result(local.disable_mtp_history(self.engine.backend)) == Ok(true) {
                    self.engine.release_drafter(self.id);
                }
            }
        }
    }
    // A retained prompt can carry a newer response snapshot. That response
    // cannot restore this entry's shorter logical prefix, so reclaim it before
    // the prompt checkpoint that makes the cached tokens/logits reusable.
    fn release_replay_snapshot(&mut self) {
        if self.local.borrow().is_some() || !self.saved.iter().any(|s| s.shared) { return; }
        let keys = self.saved.iter().map(|s| (s.tokens.len(), s.shared)).collect::<Vec<_>>();
        self.saved = kept_prompts(&keys, self.tokens.len()).into_iter().map(|i| self.saved[i].clone()).collect();
    }
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> {
        (self.engine.model.mtp_depth() != 0 && self.local.borrow().is_some()).then_some(self)
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.run(&[token], false, true)?.logits.ok_or_else(|| "missing Flash logits".into())
    }
    fn decode_logits(&mut self, rows: usize) -> Result<Option<Vec<Vec<f32>>>> {
        if rows == 0 || rows > self.engine.model.mtp_depth() + 1 {
            return Err("invalid Flash decode row count".into());
        }
        self.engine.command(DECODE_LOGITS, self.id, rows as u64, true, &[])?;
        self.engine.result(self.local.borrow_mut().as_mut().ok_or("missing Flash decoder")?
            .decode_logits(self.engine.backend, rows)).map(Some)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.run(tokens, true, true)?.logits.ok_or_else(|| "missing Flash logits".into())
    }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<()> { self.run(tokens, true, false).map(|_| ()) }
    fn prefill_chunk_size(&self, prompt_end: usize) -> usize {
        let remainder = self.tokens.len() % 512;
        if remainder != 0 { return 512 - remainder; }
        self.engine.model.prefill_chunk_size(request_capacity(self.engine.context, prompt_end),
            TILE, self.engine.mailbox.is_some())
    }
    fn can_fork(&self) -> bool { self.engine.cache_budget != 0 && self.engine.failure.borrow().is_none() }
    fn fork(&self) -> Result<Self> {
        let mut saved = self.saved.clone();
        let capacity = request_capacity(self.engine.context, self.tokens.len());
        if self.local.borrow().is_some() {
            let Some(new) = self.save(&self.tokens, false)? else { return Err("Flash snapshot cache is unfunded".into()); };
            // Rendered history may differ after a checkpoint's canonical prefix.
            saved.retain(|s| s.shared);
            saved.push(new);
        }
        // An exact replay can carry a response longer than its logical prompt.
        // The incoming prompt reservation funds only checkpoints at its bound.
        saved.retain(|s| s.tokens.len() <= capacity && s.state.prefix_len() <= self.tokens.len());
        Ok(Self { engine: self.engine, local: Rc::new(RefCell::new(None)), id: self.engine.next_id()?,
            capacity,
            tokens: self.tokens.clone(), draft_disabled: false, sampled: false, cache_retention: self.cache_retention, saved,
            published: RefCell::new(Vec::new()), revision: Cell::new(0), cached_prompt_tokens: None, depth_cap: usize::MAX })
    }
    fn fork_prefix(&self, position: usize) -> Result<Self> {
        if position > self.tokens.len() { return Err("Flash prefix beyond committed tokens".into()); }
        // A live fork saves the session and keeps only shared snapshots. Refuse
        // before that device copy when the fork could restore none of them.
        let live = self.local.borrow().as_ref().map(Session::prefix_len);
        let restorable = |n: usize, tokens: &[u32]| n != 0 && n <= position && self.tokens[..position].starts_with(&tokens[..n]);
        if !live.is_some_and(|n| restorable(n, &self.tokens)) && !self.saved.iter()
            .any(|s| (live.is_none() || s.shared) && restorable(s.state.prefix_len(), &s.tokens)) {
            return Err("Flash prefix precedes retained recurrent checkpoints".into());
        }
        // A shorter live branch must not copy its source's larger KV before
        // admission can fund it. Reuse only already retained shared state.
        let mut fork = if position < self.tokens.len() {
            Self { engine: self.engine, local: Rc::new(RefCell::new(None)), id: self.engine.next_id()?,
                capacity: request_capacity(self.engine.context, position), tokens: self.tokens[..position].to_vec(),
                draft_disabled: false, sampled: false, cache_retention: self.cache_retention,
                saved: self.saved.iter().filter(|s| live.is_none() || s.shared).cloned().collect(),
                published: RefCell::new(Vec::new()), revision: Cell::new(0), cached_prompt_tokens: None, depth_cap: usize::MAX }
        } else { self.fork()? };
        fork.tokens.truncate(position);
        fork.capacity = request_capacity(self.engine.context, position);
        fork.saved.retain(|s| s.tokens.len() <= fork.capacity && restorable(s.state.prefix_len(), &s.tokens));
        if fork.saved.is_empty() { return Err("Flash prefix has no bounded retained checkpoint".into()); }
        Ok(fork)
    }
    fn retained_bytes(&self) -> Option<usize> {
        Some(self.saved.iter().map(|s| s.bytes()).sum::<usize>() + self.tokens.capacity() * 4
            + self.local.borrow().as_ref().map_or(0, Session::snapshot_bytes))
    }
    fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> {
        Some(self.saved.iter().map(|s| (s.id as usize, s.bytes())).collect())
    }
    fn retained_private_bytes(&self) -> usize {
        self.tokens.capacity() * 4 + self.saved.capacity() * std::mem::size_of::<Rc<Saved<'_, '_, '_, '_>>>()
            + std::mem::size_of::<Self>()
    }
    // Retained snapshots no longer receive active-reservation funding credit.
    fn funded_snapshot_allocations(&self, _tokens: Option<usize>) -> Vec<(usize, usize)> { Vec::new() }
    // The scheduler's existing successful-completion hook publishes a response
    // checkpoint alongside the prompt. Scope and LRU ownership stay in its cache.
    fn retain_draft_prefix(&mut self, source: &Self) -> Result<bool> {
        if source.local.borrow().is_none() || !source.tokens.starts_with(&self.tokens)
            || self.retained_bytes().unwrap_or(0) + source.local.borrow().as_ref().unwrap().snapshot_bytes()
                > self.engine.cache_budget { return Ok(false); }
        self.engine.command(CANONICALIZE, source.id, 0, false, &[])?;
        self.engine.result(source.local.borrow_mut().as_mut().unwrap().canonicalize(self.engine.backend))?;
        source.revision.set(source.revision.get().wrapping_add(1));
        // The cache charges each retained allocation; this keeps none it did not already hold.
        let keys = self.saved.iter().map(|s| (s.tokens.len(), s.shared)).collect::<Vec<_>>();
        let kept = kept_prompts(&keys, self.tokens.len()).into_iter().map(|i| self.saved[i].clone()).collect();
        self.saved = kept;
        let Some(saved) = source.save(&source.tokens, false)? else { return Ok(false); };
        self.saved.push(saved);
        Ok(true)
    }
    fn has_draft_prefix(&self) -> bool { self.local.borrow().is_some() }
    fn retire(&mut self) { self.disable_model_draft(); }
    fn finish(&mut self) -> Result<()> { self.close() }
}
impl<'m, 'b, 'a> spark_serve::Engine for Engine<'m, 'b, 'a> {
    type Session<'s> = RemoteSession<'s, 'm, 'b, 'a> where Self: 's;
    fn session_for_request(&self, request: &spark_serve::protocol::Request) -> Result<Self::Session<'_>> {
        if self.backend.ab_enabled() {
            self.command(AB_ARM, 0, u64::from(request.arm_x), false, &[])?;
            self.backend.select_arm(request.arm_x)?;
        }
        self.session()
    }
    fn session(&self) -> Result<Self::Session<'_>> {
        Ok(RemoteSession { engine: self, local: Rc::new(RefCell::new(None)), id: self.next_id()?,
            capacity: request_capacity(self.context, 0), tokens: Vec::new(), draft_disabled: false, sampled: false,
            cache_retention: true, saved: Vec::new(),
            published: RefCell::new(Vec::new()), revision: Cell::new(0), cached_prompt_tokens: None, depth_cap: usize::MAX })
    }
    fn shared_batch(&self) -> bool { true }
    fn profile_tick_begin(&self, tick: u64) {
        if self.data.1 {
            if let Ok(mut ring) = self.data.0.lock() { ring.profile_begin_tick(tick); }
        }
    }
    fn profile_tick_end(&self) {
        if self.data.1 {
            if let Ok(mut ring) = self.data.0.lock() { ring.profile_end_tick(self.backend.allocated_bytes()); }
        }
    }
    // Preserve the existing 512-token prefill chunk boundaries, including short prompts.
    fn shared_prompt_limit(&self) -> usize { 0 }
    fn context_limit(&self) -> usize { self.context }
    fn vocab_size(&self) -> usize { self.vocab }
    fn eos_token_ids(&self) -> &[u32] { &self.eos }
    fn session_reservation_bytes(&self) -> Option<usize> {
        // Use the negotiated bound for all ranks, including an in-flight
        // compact response checkpoint before LRU insertion.
        Some(self.reservation)
    }
    fn request_reservation_bytes(&self, tokens: usize) -> Option<usize> {
        self.model.request_reservation_bytes(request_capacity(self.context, tokens), TILE,
            self.workspace_slots, self.cache_budget != 0).ok()
    }
    fn automatic_draft_allowed(&self, chats: usize) -> bool { self.model.mtp_depth() != 0 && chats == 1 }
    fn single_chat_draft(&self) -> bool { true }
    fn discard_contended_draft(&self) -> bool { true }
    fn memory_bytes(&self) -> Option<(usize, usize, usize)> {
        Some((self.backend.allocated_bytes(), self.backend.peak_allocated_bytes(), self.backend.pinned_host_bytes()))
    }
    fn validate_request(&self, prompt: &[u32], request: &spark_serve::protocol::Request) -> Result<()> {
        request.sampling.validate()?;
        if prompt.is_empty() || prompt.iter().any(|&t| t as usize >= self.vocab)
            || prompt.len().checked_add(request.max_tokens).is_none_or(|end| end > self.context)
            || self.context < TILE {
            return Err("Flash request outside vocabulary or context".into());
        }
        Ok(())
    }
    fn fatal_error(&self) -> Option<String> { self.failure.borrow().clone().or_else(|| self.backend.fatal_error()) }
    fn step_batch<'s>(&'s self, rows: &mut [BatchRows<'_, Self::Session<'s>>]) -> std::result::Result<Vec<Predictions>, BatchError> {
        let mut seen = std::collections::BTreeSet::new();
        let rejected = rows.iter().map(|row| {
            !std::ptr::eq(row.session.engine, self) || !seen.insert(row.session.id)
                || row.tokens.is_empty() || row.tokens.iter().any(|&t| t as usize >= self.vocab)
                || row.session.tokens.len().checked_add(row.tokens.len()).is_none_or(|end| end > self.context)
                || matches!(row.output, BatchOutput::All) && row.tokens.len() > self.model.mtp_depth() + 1
        }).collect::<Vec<_>>();
        if rejected.iter().any(|&bad| bad) {
            // Invalid request input never joins the distributed command. Keep
            // the healthy subset batched and put each verdict back in row order.
            let mut valid = rows.iter_mut().zip(&rejected).filter_map(|(row, &bad)| (!bad).then(||
                BatchRows { session: &mut *row.session, tokens: row.tokens, output: row.output,
                    greedy: row.greedy, top_k: row.top_k })).collect::<Vec<_>>();
            let mut output = if valid.is_empty() { Vec::new() } else { self.step_batch(&mut valid)? }.into_iter();
            return Ok(rejected.into_iter().map(|bad| if bad {
                Predictions::Rejected("invalid Flash batch request".into())
            } else { output.next().unwrap() }).collect());
        }
        // Plain decode rows (one token, every output, an open session) share
        // fixed 16-row groups whatever else the step holds. Draft verification,
        // prompt rows and an identical-prompt cache hit (no execution scratch
        // yet: it opens/restores and prepares scratch outside graphs) run singly
        // below, as does a lone plain row, in the same call.
        let plain = rows.iter().map(|r| matches!(r.output, BatchOutput::All) && r.tokens.len() == 1
            && r.session.local.borrow().is_some()).collect::<Vec<_>>();
        let mut predictions = rows.iter().map(|_| None).collect::<Vec<Option<Predictions>>>();
        let mut decode = rows.iter_mut().enumerate().filter(|(i, _)| plain[*i]).collect::<Vec<_>>();
        if decode.len() > 1 {
            for (i, row) in decode.iter() {
                if decode.len() > self.slots || !std::ptr::eq(row.session.engine, self)
                    || row.session.tokens.is_empty() || row.session.tokens.len() >= self.context
                    || row.tokens[0] as usize >= self.vocab {
                    return Err(BatchError { request: Some(*i), message: "invalid Flash decode batch row".into() });
                }
            }
            for (i, row) in decode.iter_mut() {
                row.session.capacity = row.session.capacity.max(request_capacity(self.context, row.session.tokens.len() + 1));
                row.session.reserve().map_err(|message| BatchError { request: Some(*i), message })?;
            }
            // Keep each row's layer/reduction order and the fixed 16-row GPU
            // workspace. The command order is also the worker's execution order.
            // A 17/33-row tail is left to the existing solo decode below.
            for rows in decode.chunks_mut(DecodeBatch::MAX_ROWS).filter(|rows| rows.len() > 1) {
                let tokens = rows.iter().map(|(_, r)| r.tokens[0]).collect::<Vec<_>>();
                let logits = rows.iter().map(|(_, r)| !r.greedy).collect::<Vec<_>>();
                let wire = rows.iter().flat_map(|(_, row)| [row.session.id as u32,
                    (row.session.id >> 32) as u32, row.tokens[0], u32::from(!row.greedy)]).collect::<Vec<_>>();
                self.command(BATCH, 0, 0, false, &wire).map_err(|message| BatchError { request: None, message })?;
                let result = {
                    let mut locals = rows.iter_mut().map(|(_, r)| r.session.local.borrow_mut()).collect::<Vec<_>>();
                    let mut sessions = locals.iter_mut().map(|local| local.as_mut().unwrap()).collect::<Vec<_>>();
                    Session::forward_batch(self.backend, &mut self.batch.borrow_mut(), &mut sessions, &tokens, &logits)
                };
                let outputs = self.result(result).map_err(|message| BatchError { request: None, message })?;
                if self.data.1 || self.model.layer_profiling() {
                    write_profile(&format!("flash_serve batch rows={} elapsed_ms={:.3} device_bytes={} device_peak_bytes={}\n",
                        rows.len(), outputs.iter().find_map(|row| row.as_ref().ok()).map_or(0.0, |row| row.elapsed_ms), self.backend.allocated_bytes(), self.backend.peak_allocated_bytes()));
                }
                for ((i, row), output) in rows.iter_mut().zip(outputs) {
                    let output = match output {
                        Ok(output) => output,
                        Err(message) => {
                            self.state.borrow_mut().sessions.get_mut(&row.session.id).unwrap().0 -= 1;
                            predictions[*i] = Some(Predictions::Rejected(message));
                            continue;
                        }
                    };
                    row.session.tokens.push(row.tokens[0]);
                    row.session.revision.set(row.session.revision.get().wrapping_add(1));
                    predictions[*i] = Some(if row.greedy { Predictions::Greedy(vec![output.token]) }
                        else { Predictions::Logits(vec![output.logits.ok_or_else(|| BatchError {
                            request: Some(*i), message: "missing Flash logits".into() })?]) });
                }
            }
        }
        drop(decode);
        rows.iter_mut().zip(predictions).enumerate().map(|(i, (row, batched))| {
            if let Some(prediction) = batched { return Ok(prediction); }
            let prefill = !matches!(row.output, BatchOutput::All);
            if !prefill && !row.greedy && row.session.sampled {
                return row.session.verify_sampled(row.tokens, row.top_k)
                    .map_err(|message| BatchError { request: Some(i), message });
            }
            if !prefill && row.tokens.len() > 1 {
                return row.session.verify(row.tokens, !row.greedy).and_then(|outputs| {
                    if row.greedy { Ok(Predictions::Greedy(outputs.iter().map(|o| o.token).collect())) }
                    else { outputs.into_iter().map(|o| o.logits.ok_or_else(|| "missing Flash verify logits".into()))
                        .collect::<Result<Vec<_>>>().map(Predictions::Logits) }
                }).map_err(|message| BatchError { request: Some(i), message });
            }
            let logits = (prefill || !row.greedy) && !matches!(row.output, BatchOutput::None);
            row.session.run(row.tokens, prefill, logits).and_then(|output| {
                Ok(if matches!(row.output, BatchOutput::None) { Predictions::Logits(Vec::new()) }
                    else if row.greedy && !prefill { Predictions::Greedy(vec![output.token]) }
                    else { Predictions::Logits(vec![output.logits.ok_or("missing Flash logits")?]) })
            }).map_err(|message| BatchError { request: Some(i), message })
        }).map(|result| match result {
            Err(error) if error.message.starts_with("Flash MoE routing rejected") =>
                Ok(Predictions::Rejected(error.message)),
            other => other,
        }).collect()
    }
}

pub fn worker<'a>(model: &Model<'a>, backend: &Backend<'a>, mut commands: Commands,
    data: Data, context: usize, vocab: usize, slots: usize, caching: bool, startup: std::time::Instant) -> Result<()> {
    let start = std::time::Instant::now();
    let mailbox = data.register_mailbox(backend)?;
    let mut sessions: BTreeMap<u64, (Session<'a, '_, Data>, usize)> = BTreeMap::new();
    let mut batch = DecodeBatch::new(backend, model)?;
    let mut saved: BTreeMap<u64, (usize, Snapshot<'a>)> = BTreeMap::new();
    let mut state = CommandState::default();
    let sampled_peak = Cell::new(backend.peak_accounted_bytes());
    let minimum_available = Cell::new(usize::MAX);
    backend.synchronize()?;
    eprintln!("flash_serve worker_ready rank={} startup_ms={:.3} worker_setup_ms={:.3} model_bytes={} device_bytes={} device_peak_bytes={} peak_pinned_staging_bytes={} graph_captures={}",
        commands.rank(), startup.elapsed().as_secs_f64() * 1000.0, start.elapsed().as_secs_f64() * 1000.0,
        model.resident_bytes(), backend.allocated_bytes(), backend.peak_allocated_bytes(), backend.peak_pinned_staging_bytes(), backend.stats().graph_captures);
    let result = (|| loop {
        if data.1 {
            if let Ok(mut ring) = data.0.lock() { ring.profile_begin_command(); }
        }
        let step: Result<bool> = (|| {
            let (words, tokens) = command_message(&data, || {
                let mut header = [0u8; 40];
                command_part(&mut commands, &data, &mut header, true)?;
                let words = header.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                let (logits, count) = (words[3], words[4]);
                if count > context.max(slots * 4) as u64 || logits > 1 {
                    return Err("Flash rank command header inconsistency".into());
                }
                let mut bytes = vec![0; count as usize * 4];
                if count != 0 { command_part(&mut commands, &data, &mut bytes, false)?; }
                let tokens = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                Ok((words, tokens))
            })?;
            let (op, requested, arg, logits, count) = (words[0], words[1], words[2], words[3], words[4]);
            let valid = state.validate(op, requested, arg, logits != 0, &tokens, context, vocab, slots, model.mtp_depth());
            let consistent = state.sessions.len() == sessions.len() && state.saved.len() == saved.len()
                && state.sessions.iter().all(|(id, &(position, capacity, _))|
                    sessions.get(id).is_some_and(|(local, actual)| *actual == position && local.capacity() == capacity))
                && state.saved.iter().all(|(id, &(end, prefix))|
                    saved.get(id).is_some_and(|(actual, snapshot)| *actual == end && snapshot.prefix_len() == prefix));
            // Rank zero validates before dispatch. A worker mismatch is fatal:
            // closing both transports wakes the peer's current/next collective,
            // including idle control polling, without per-command round trips.
            if valid.is_err() || !consistent { return Err("Flash rank command/session inconsistency".into()); }
            let executed: Result<()> = (|| { match op {
                AB_ARM => backend.select_arm(arg == 1)?,
                OPEN if sessions.len() < slots && count == 0 && requested != 0
                    && logits == 0 && arg >= TILE as u64 && arg <= context as u64
                    && !sessions.contains_key(&requested) && !saved.contains_key(&requested) => {
                    let mut local = model.session_plain_with_mailbox(backend, vec![data.clone()], arg as usize, TILE, mailbox.clone())?;
                    if caching { local.enable_cache(); }
                    sessions.insert(requested, (local, 0usize));
                }
                RESERVE if sessions.contains_key(&requested) && count == 0 && arg <= context as u64 => {
                    sessions.get_mut(&requested).unwrap().0.reserve(backend, arg as usize)?;
                }
                PREFILL | STEP if sessions.contains_key(&requested) => {
                    let (local, position) = sessions.get_mut(&requested).unwrap();
                    if tokens.is_empty() || tokens.len() > context - *position || tokens.iter().any(|&t| t as usize >= vocab)
                        || op == STEP && tokens.len() != 1 { return Err("invalid Flash forward command".into()); }
                    if op == PREFILL { local.forward_prefill(backend, &tokens, logits != 0)?; }
                    else { local.forward(backend, &tokens, logits != 0)?; }
                    *position += tokens.len();
                }
                BATCH if requested == 0 && logits == 0 && count >= 8
                    && count <= (DecodeBatch::MAX_ROWS.min(slots) * 4) as u64 && count % 4 == 0 => {
                    let rows = tokens.chunks_exact(4).map(|r| ((r[0] as u64) | ((r[1] as u64) << 32), r[2], r[3])).collect::<Vec<_>>();
                    let mut seen = std::collections::BTreeSet::new();
                    for &(id, token, logits) in &rows {
                        if !seen.insert(id) || token as usize >= vocab || logits > 1
                            || sessions.get(&id).is_none_or(|(_, position)| *position == 0 || *position >= context) {
                            return Err("invalid Flash decode batch command".into());
                        }
                    }
                    // Detach in wire order so every rank issues collectives in the
                    // scheduler's row order, independent of session-map ordering.
                    let mut locals = rows.iter().map(|&(id, _, _)| sessions.remove(&id).unwrap()).collect::<Vec<_>>();
                    let tokens = rows.iter().map(|r| r.1).collect::<Vec<_>>();
                    let logits = rows.iter().map(|r| r.2 != 0).collect::<Vec<_>>();
                    let output = Session::forward_batch(backend, &mut batch,
                        &mut locals.iter_mut().map(|(local, _)| local).collect::<Vec<_>>(), &tokens, &logits);
                    for (i, ((id, _, _), (local, position))) in rows.into_iter().zip(locals).enumerate() {
                        let advanced = output.as_ref().is_ok_and(|rows| rows[i].is_ok());
                        sessions.insert(id, (local, position + usize::from(advanced)));
                    }
                    output?;
                }
                DRAFT if sessions.contains_key(&requested) && count == 1 && arg > 0 && arg <= model.mtp_depth() as u64 => {
                    sessions.get_mut(&requested).unwrap().0.propose_mtp(backend, tokens[0], arg as usize)?;
                }
                DRAFT_SAMPLED if sessions.contains_key(&requested) && count == 2 && tokens[1] <= 64
                    && arg > 0 && arg <= model.mtp_depth() as u64 => {
                    sessions.get_mut(&requested).unwrap().0.propose_mtp_sampled(backend, tokens[0],
                        arg as usize, tokens[1] as usize, |_, _| Ok(None))?;
                }
                CLAIM_MTP if sessions.contains_key(&requested) && count == 0 => {
                    sessions.get_mut(&requested).unwrap().0.prepare_mtp_workspace(backend)?;
                }
                ENABLE_MTP if sessions.contains_key(&requested) && count == 0 => {
                    sessions.get_mut(&requested).unwrap().0.enable_mtp_history(backend)?;
                }
                VERIFY if sessions.contains_key(&requested) && count > 1 && count <= (model.mtp_depth() + 1) as u64 => {
                    let (local, position) = sessions.get_mut(&requested).unwrap();
                    local.verify_mtp(backend, &tokens, logits != 0)?;
                    *position += tokens.len();
                }
                VERIFY_SAMPLED if sessions.contains_key(&requested) && count >= 1
                    && count <= (model.mtp_depth() + 1) as u64 && arg <= 64 => {
                    let (local, position) = sessions.get_mut(&requested).unwrap();
                    local.verify_mtp_sampled(backend, &tokens, arg as usize)?;
                    *position += tokens.len();
                }
                DECODE_LOGITS if sessions.contains_key(&requested) && count == 0 && arg >= 1
                    && arg <= (model.mtp_depth() + 1) as u64 => {
                    sessions.get_mut(&requested).unwrap().0.decode_logits(backend, arg as usize)?;
                }
                COMMIT if sessions.contains_key(&requested) && count == 0 => {
                    let (local, position) = sessions.get_mut(&requested).unwrap();
                    local.commit_verification(backend, arg as usize)?;
                    *position = arg as usize;
                }
                DISABLE_MTP if sessions.contains_key(&requested) && count == 0 => {
                    sessions.get_mut(&requested).unwrap().0.disable_mtp_history(backend)?;
                }
                CLOSE if sessions.contains_key(&requested) && count == 0 => {
                    memory_receipt(backend, commands.rank(), "close", sessions.len(), &sampled_peak, &minimum_available, true);
                    let (local, _) = sessions.remove(&requested).unwrap();
                    local.report();
                    drop(local);
                    commands.barrier()?;
                    eprintln!("flash_serve worker session={requested} closed device_bytes={} device_peak_bytes={}",
                        backend.allocated_bytes(), backend.peak_allocated_bytes());
                }
                SAVE if sessions.contains_key(&arg) && count == 2 && requested != 0
                    && !saved.contains_key(&requested) && !sessions.contains_key(&requested) => {
                    let (local, position) = sessions.get(&arg).unwrap();
                    let reserved = (u64::from(tokens[0]) | (u64::from(tokens[1]) << 32)) as usize;
                    let allocated = sessions.values().map(|(s, _)| s.private_cache_bytes()).sum();
                    if let Some(snapshot) = save_snapshot(backend, &mut commands, local, reserved, allocated, 0)? {
                        saved.insert(requested, (*position, snapshot));
                        commands.barrier()?;
                    }
                    memory_receipt(backend, commands.rank(), "save", sessions.len(), &sampled_peak, &minimum_available, true);
                }
                RESTORE if sessions.contains_key(&requested) && count == 0 => {
                    let (end, snapshot) = saved.get(&arg).ok_or("unknown Flash snapshot")?;
                    let (local, position) = sessions.get_mut(&requested).unwrap();
                    local.restore(backend, snapshot, logits != 0)?;
                    *position = if logits != 0 { snapshot.prefix_len() } else { *end };
                }
                RELEASE if count == 0 => { saved.remove(&requested).ok_or("unknown Flash snapshot release")?; }
                CANONICALIZE if sessions.contains_key(&requested) && count == 0 => {
                    sessions.get_mut(&requested).unwrap().0.canonicalize(backend)?;
                }
                STOP if sessions.is_empty() && count == 0 => return Ok(()),
                _ => return Err("invalid Flash command or session state".into()),
            }
            Ok(()) })();
            if let Err(error) = executed {
                if error.starts_with("Flash MoE routing rejected") {
                    eprintln!("flash_serve worker rank={} session={requested} op={op} request_failed: {error}", commands.rank());
                }
                return Err(error);
            }
            memory_receipt(backend, commands.rank(), command_phase(op), sessions.len(), &sampled_peak, &minimum_available, false);
            if op != SAVE || saved.contains_key(&requested) { state.apply(op, requested, arg, logits != 0, &tokens); }
            if op == BATCH {
                for row in tokens.chunks_exact(4) {
                    let id = row[0] as u64 | ((row[1] as u64) << 32);
                    state.sessions.get_mut(&id).unwrap().0 = sessions[&id].1;
                }
            }
            // Snapshot prefix length follows the engine's canonical recurrent boundary.
            if op == SAVE && saved.contains_key(&requested) { state.saved.get_mut(&requested).unwrap().1 = saved[&requested].1.prefix_len(); }
            Ok(op == STOP)
        })();
        if data.1 {
            if let Ok(mut ring) = data.0.lock() { ring.profile_end_step(step.is_ok(), backend.allocated_bytes()); }
        }
        match step {
            Ok(true) => return Ok(()),
            Ok(false) => (),
            Err(error) if error.starts_with("Flash MoE routing rejected") => continue,
            Err(error) => return Err(error),
        }
    })();
    if result.is_err() { spark_gpu_probe::teardown::arm(); commands.abort(); data.abort(); }
    drop(sessions);
    result
}

#[cfg(test)]
mod tests {
    use super::kept_prompts;
    #[test]
    fn exact_repeat_keeps_prompt_end_beside_shared_prefix() {
        assert_eq!(kept_prompts(&[(5312, true), (8192, false), (8300, false)], 8192), [0, 1]);
        assert_eq!(kept_prompts(&[(5312, false), (5312, true)], 5312), [1]);
        assert_eq!(kept_prompts(&[(100, false), (300, false), (900, false)], 300), [1]);
    }
}
