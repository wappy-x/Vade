//! Bounded, parent-owned native children. This module owns scheduling, not a
//! second agent loop: every child runs `agent::run` with independent history.
//! Host model/tool adapters are trusted Rust code, not sandboxed plugins.
//! Cancellation is cooperative; terminal states are published only after join.
mod native;
mod native_tools;
mod parent;
mod policy;
mod prefix;
pub mod ui;

pub use native::{HttpChildModel, read_only_http_factory};
pub use native_tools::{
    ChildPreview, NativeToolHost, native_http_factory, native_http_factory_with_generation,
};
pub use parent::{
    ParentRun, definitions, run_parent, run_parent_observed, run_parent_observed_with_budget,
};
pub use policy::{Access, Scope, ToolPolicy};
pub use prefix::StablePrefix;

use crate::agent::{
    self, Conversation, Limits, Model, ModelRequest, RunReport, StopReason, ToolError,
    ToolExecutor, ToolInvocation,
};
use crate::agent_wire::{AssistantTurn, FinishReason};
use crate::contract::Usage;
use std::collections::BTreeMap;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MONITOR_INTERVAL: Duration = Duration::from_millis(10);
const MAX_GOAL_BYTES: usize = 16 * 1024;
const MAX_SUMMARY_BYTES: usize = 4096;
static NEXT_SCHEDULER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskId {
    scheduler: u64,
    sequence: usize,
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "child-{}-{}", self.scheduler, self.sequence)
    }
}

/// Hard upper bounds are validated independently of caller-supplied defaults.
#[derive(Clone, Copy, Debug)]
pub struct Quota {
    pub model_turns: usize,
    pub tool_calls: usize,
    /// Sum of exact prompt tokens and the full reserved output on every request.
    /// Reservations are never refunded, including on error or cancellation.
    pub tokens: u64,
    pub output_tokens_per_turn: u64,
    pub wall_time: Duration,
    pub history_bytes: usize,
    pub tool_output_bytes: usize,
}

impl Default for Quota {
    fn default() -> Self {
        Self {
            model_turns: 8,
            tool_calls: 16,
            tokens: 32_768,
            output_tokens_per_turn: 1024,
            wall_time: Duration::from_secs(120),
            history_bytes: 512 * 1024,
            tool_output_bytes: 128 * 1024,
        }
    }
}

impl Quota {
    fn validate(&self) -> bool {
        (1..=64).contains(&self.model_turns)
            && self.tool_calls <= 128
            && (1..=1_048_576).contains(&self.tokens)
            && (1..=16_384).contains(&self.output_tokens_per_turn)
            && self.output_tokens_per_turn <= self.tokens
            && !self.wall_time.is_zero()
            && self.wall_time <= Duration::from_secs(600)
            && (1024..=2 * 1024 * 1024).contains(&self.history_bytes)
            && (128..=1024 * 1024).contains(&self.tool_output_bytes)
    }

    fn limits(self) -> Limits {
        Limits {
            max_model_turns: self.model_turns,
            max_tool_turns: self.model_turns,
            max_tool_calls: self.tool_calls,
            max_tool_output_bytes: self.tool_output_bytes.min(64 * 1024),
            max_total_output_bytes: self.tool_output_bytes,
            max_history_bytes: self.history_bytes,
            ..Limits::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Task {
    pub goal: String,
    /// Name of a host-declared scope. None always means read-only.
    pub grant: Option<String>,
    pub quota: Quota,
}

impl Task {
    pub fn read_only(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            grant: None,
            quota: Quota::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub max_concurrent: usize,
    pub max_tasks: usize,
    pub total_model_turns: usize,
    pub total_tool_calls: usize,
    pub total_tokens: u64,
    pub total_history_bytes: usize,
    pub total_tool_output_bytes: usize,
    pub wall_time: Duration,
    pub default_quota: Quota,
    pub policy: ToolPolicy,
    /// Host authorization only: model requests can select, never create, grants.
    pub grants: BTreeMap<String, Scope>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            max_tasks: 16,
            total_model_turns: 64,
            total_tool_calls: 128,
            total_tokens: 262_144,
            total_history_bytes: 8 * 1024 * 1024,
            total_tool_output_bytes: 1024 * 1024,
            wall_time: Duration::from_secs(300),
            default_quota: Quota::default(),
            policy: ToolPolicy::default(),
            grants: BTreeMap::new(),
        }
    }
}

/// An adapter must reserve exact prompt + bounded output tokens once, before
/// generation dispatch. Unknown estimates are not accepted as token quotas.
/// `HttpChildModel` implements this using the negotiated admission contract.
pub trait ChildModel: Send {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        meter: &mut TokenMeter,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn>;

    fn usage(&self) -> Option<Usage> {
        None
    }
}

pub struct ChildWorker {
    pub model: Box<dyn ChildModel>,
    pub tools: Box<dyn ToolExecutor + Send>,
    /// Native hosts supply a separate child journal; mock/custom hosts can omit it.
    pub observer: Option<Box<dyn agent::RunObserver + Send>>,
    pub journal_name: Option<String>,
}

pub struct ChildContext {
    pub id: TaskId,
    pub scope: Scope,
    pub quota: Quota,
}

pub type Factory = dyn Fn(&ChildContext, &AtomicBool) -> io::Result<ChildWorker> + Send + Sync;

pub struct TokenMeter {
    limit: u64,
    output: u64,
    charged: u64,
    calls: usize,
    exhausted: bool,
    protocol_error: bool,
    turn_charged: bool,
    progress: Option<(Arc<Shared>, TaskId)>,
}

impl TokenMeter {
    /// Charge the full exact request and output reservation. Do not call this
    /// for cached tokens separately: they are already included in prompt_tokens.
    pub fn reserve(&mut self, prompt_tokens: u64, output_tokens: u64) -> io::Result<()> {
        if self.turn_charged || output_tokens != self.output {
            self.protocol_error = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid child token reservation",
            ));
        }
        let Some(total) = prompt_tokens.checked_add(output_tokens) else {
            self.exhausted = true;
            return Err(quota_error());
        };
        if total > self.limit - self.charged {
            self.exhausted = true;
            return Err(quota_error());
        }
        self.charged += total;
        self.calls += 1;
        self.turn_charged = true;
        if let Some((shared, id)) = &self.progress {
            let mut state = lock(&shared.state);
            if let Some(entry) = state.entries.get_mut(id) {
                entry.snapshot.tokens_reserved = self.charged;
                entry.snapshot.model_turns = self.calls;
                notify(shared, &mut state);
            }
        }
        Ok(())
    }

    pub fn remaining(&self) -> u64 {
        self.limit - self.charged
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopRequest {
    ParentCancelled,
    ChildCancelled,
    WallTime,
    ParentFinished,
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Cancelled,
    QuotaTokens,
    /// Opaque provider finish text stays in the parent-owned RunReport, never
    /// in lifecycle snapshots or automatically forwarded result summaries.
    UnknownFinish,
    Runtime(StopReason),
    SetupFailed(io::ErrorKind),
    Panicked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskState {
    Running,
    Stopping(StopRequest),
    Finished(Outcome),
    Collected(Outcome),
}

#[derive(Clone, Debug)]
pub struct TaskSnapshot {
    pub id: TaskId,
    pub state: TaskState,
    pub stop_requested: Option<StopRequest>,
    pub writable_files: usize,
    pub workspace_write_lease: bool,
    pub elapsed: Duration,
    pub tokens_reserved: u64,
    pub model_turns: usize,
    /// Completed, validated responses only. Reservations are separate; absent
    /// telemetry never implies zero cached tokens.
    pub usage: ChildUsage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChildUsage {
    pub last: Option<Usage>,
    pub completed_responses: u64,
    pub responses_with_cached_usage: u64,
    pub reported_cached_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    Started,
    Completed,
    /// A receipt is present, but a partial patch or unsuccessful command needs
    /// workspace reconciliation. Never confuse this with a denied operation.
    Uncertain,
    Denied,
    Cancelled,
    Failed,
    OutputLimit,
}

#[derive(Clone, Debug)]
pub struct ToolReceipt {
    pub call_id: String,
    pub tool: String,
    pub writes: bool,
    pub status: ToolStatus,
}

pub struct TaskResult {
    pub id: TaskId,
    pub outcome: Outcome,
    /// A request is not an acknowledgment. Completion may race with a request.
    pub stop_requested: Option<StopRequest>,
    pub report: Option<RunReport>,
    pub conversation: Option<Conversation>,
    pub summary: String,
    pub summary_truncated: bool,
    pub tools: Vec<ToolReceipt>,
    pub tokens_reserved: u64,
    /// Last completed model response telemetry, never inferred from reservations.
    pub last_usage: Option<Usage>,
    /// Durable child transcript, when supplied by the native host. Never replayed.
    pub journal_name: Option<String>,
}

impl TaskResult {
    /// A tool's failure/cancellation is not proof that its side effect did not
    /// happen. Edit and journal adapters must supply authoritative per-file receipts.
    pub fn uncertain_writes(&self) -> bool {
        self.tools
            .iter()
            .any(|r| r.writes && !matches!(r.status, ToolStatus::Completed | ToolStatus::Denied))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    Closed,
    Capacity,
    Quota,
    InvalidTask,
    UnknownGrant,
    WriteConflict,
    UnknownTask,
    NotFinished,
    AlreadyCollected,
    SpawnFailed,
}

#[derive(Default)]
struct Allocation {
    turns: usize,
    calls: usize,
    tokens: u64,
    history: usize,
    output: usize,
}

struct Entry {
    snapshot: TaskSnapshot,
    started: Instant,
    deadline: Instant,
    scope: Scope,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<TaskResult>>,
    result: Option<TaskResult>,
}

struct State {
    entries: BTreeMap<TaskId, Entry>,
    allocation: Allocation,
    closed: Option<StopRequest>,
    shutdown: bool,
    revision: u64,
    wake: Option<thread::Thread>,
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    parent_cancel: Arc<AtomicBool>,
    deadline: Instant,
}

/// Read-only lifecycle view: UI gets no result-consumption or tool authority.
#[derive(Clone)]
pub struct ChildView {
    shared: Arc<Shared>,
}

impl ChildView {
    /// Register one bounded UI wake target. Coalesced revisions still supply
    /// metadata; no per-token/event queue or watcher thread is introduced.
    pub fn wake_current_thread(&self) {
        lock(&self.shared.state).wake = Some(thread::current());
    }
    pub fn snapshots(&self) -> Vec<TaskSnapshot> {
        let state = lock(&self.shared.state);
        state
            .entries
            .values()
            .map(|entry| {
                let mut snapshot = entry.snapshot.clone();
                if entry.handle.is_some() {
                    snapshot.elapsed = entry.started.elapsed();
                }
                snapshot
            })
            .collect()
    }

    pub fn revision(&self) -> u64 {
        lock(&self.shared.state).revision
    }

    /// Bounded wait, suitable for a UI watcher. No provisional text queue grows.
    pub fn wait_for_change(&self, revision: u64, timeout: Duration) -> u64 {
        let state = lock(&self.shared.state);
        let (state, _) = self
            .shared
            .changed
            .wait_timeout_while(state, timeout.min(Duration::from_secs(1)), |s| {
                s.revision == revision
            })
            .unwrap_or_else(|e| e.into_inner());
        state.revision
    }
}

/// UI authority is limited to stopping existing work. It cannot spawn tasks,
/// execute tools, consume results, or mutate the parent's conversation.
#[derive(Clone)]
pub struct ChildControl {
    shared: Arc<Shared>,
}

impl ChildControl {
    /// Returns whether this call newly requested cancellation. A finished task
    /// keeps its actual outcome; an unknown or stale ID is rejected.
    pub fn cancel(&self, id: &str) -> Result<bool, ScheduleError> {
        let mut state = lock(&self.shared.state);
        let entry = state
            .entries
            .values_mut()
            .find(|entry| entry.snapshot.id.to_string() == id)
            .ok_or(ScheduleError::UnknownTask)?;
        let requested = entry.handle.is_some() && entry.snapshot.stop_requested.is_none();
        stop(entry, StopRequest::ChildCancelled);
        if requested {
            notify(&self.shared, &mut state);
        }
        Ok(requested)
    }

    /// Closes admission for this activation as well as cancelling existing
    /// children. The parent continues and can collect already completed results.
    pub fn cancel_all(&self) -> usize {
        let mut state = lock(&self.shared.state);
        state.closed.get_or_insert(StopRequest::ChildCancelled);
        let mut requested = 0;
        for entry in state.entries.values_mut() {
            requested +=
                usize::from(entry.handle.is_some() && entry.snapshot.stop_requested.is_none());
            stop(entry, StopRequest::ChildCancelled);
        }
        notify(&self.shared, &mut state);
        requested
    }
}

pub struct Scheduler {
    id: u64,
    config: Config,
    prefix: StablePrefix,
    factory: Arc<Factory>,
    shared: Arc<Shared>,
    monitor: Option<JoinHandle<()>>,
}

impl Scheduler {
    pub fn new(
        prefix: StablePrefix,
        config: Config,
        parent_cancel: Arc<AtomicBool>,
        factory: Arc<Factory>,
    ) -> io::Result<Self> {
        if !(1..=8).contains(&config.max_concurrent)
            || !(1..=32).contains(&config.max_tasks)
            || config.max_concurrent > config.max_tasks
            || !config.default_quota.validate()
            || config.total_model_turns > 256
            || config.total_tool_calls > 1024
            || config.total_tokens > 4_194_304
            || config.total_history_bytes > 32 * 1024 * 1024
            || config.total_tool_output_bytes > 8 * 1024 * 1024
            || config.wall_time.is_zero()
            || config.wall_time > Duration::from_secs(1800)
            || config.grants.len() > 32
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid child scheduler limits",
            ));
        }
        for (name, scope) in &config.grants {
            if !policy::identifier(name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid child grant name",
                ));
            }
            config.policy.validate(scope, &prefix.advertised)?;
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                entries: BTreeMap::new(),
                allocation: Allocation::default(),
                closed: None,
                shutdown: false,
                revision: 0,
                wake: None,
            }),
            changed: Condvar::new(),
            parent_cancel,
            deadline: Instant::now() + config.wall_time,
        });
        let monitor_shared = Arc::clone(&shared);
        let monitor = thread::Builder::new()
            .name("vade-child-watch".into())
            .spawn(move || monitor(monitor_shared))?;
        Ok(Self {
            id: NEXT_SCHEDULER.fetch_add(1, Ordering::Relaxed),
            config,
            prefix,
            factory,
            shared,
            monitor: Some(monitor),
        })
    }

    pub fn view(&self) -> ChildView {
        ChildView {
            shared: Arc::clone(&self.shared),
        }
    }
    pub fn control(&self) -> ChildControl {
        ChildControl {
            shared: Arc::clone(&self.shared),
        }
    }
    pub fn default_quota(&self) -> Quota {
        self.config.default_quota
    }

    /// No hidden queue: a full concurrent set refuses spawn with Capacity.
    /// Quotas are allocated for the entire parent turn and never replenished.
    pub fn spawn(&mut self, task: Task) -> Result<TaskId, ScheduleError> {
        if task.goal.trim().is_empty() || task.goal.len() > MAX_GOAL_BYTES || !task.quota.validate()
        {
            return Err(ScheduleError::InvalidTask);
        }
        let scope = match &task.grant {
            None => self.config.policy.read_scope(&self.prefix.advertised),
            Some(name) => self
                .config
                .grants
                .get(name)
                .cloned()
                .ok_or(ScheduleError::UnknownGrant)?,
        };
        let limits = task.quota.limits();
        let conversation = self
            .prefix
            .conversation(&task, &scope, limits)
            .map_err(|_| ScheduleError::InvalidTask)?;
        let mut state = lock(&self.shared.state);
        if state.closed.is_some()
            || self.shared.parent_cancel.load(Ordering::Acquire)
            || Instant::now() >= self.shared.deadline
        {
            return Err(ScheduleError::Closed);
        }
        if state
            .entries
            .values()
            .filter(|e| e.handle.is_some())
            .count()
            >= self.config.max_concurrent
        {
            return Err(ScheduleError::Capacity);
        }
        let a = &state.allocation;
        if state.entries.len() >= self.config.max_tasks
            || a.turns + task.quota.model_turns > self.config.total_model_turns
            || a.calls + task.quota.tool_calls > self.config.total_tool_calls
            || a.tokens + task.quota.tokens > self.config.total_tokens
            || a.history + task.quota.history_bytes > self.config.total_history_bytes
            || a.output + task.quota.tool_output_bytes > self.config.total_tool_output_bytes
        {
            return Err(ScheduleError::Quota);
        }
        if state.entries.values().any(|e| {
            !matches!(e.snapshot.state, TaskState::Collected(_))
                && self.config.policy.conflicts(&e.scope, &scope)
        }) {
            return Err(ScheduleError::WriteConflict);
        }
        let id = TaskId {
            scheduler: self.id,
            sequence: state.entries.len() + 1,
        };
        let context = ChildContext {
            id,
            scope: scope.clone(),
            quota: task.quota,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let child_cancel = Arc::clone(&cancel);
        let factory = Arc::clone(&self.factory);
        let policy = self.config.policy.clone();
        let shared = Arc::clone(&self.shared);
        let handle = thread::Builder::new()
            .name(id.to_string())
            .spawn(move || {
                let result = execute_child(
                    context,
                    conversation,
                    policy,
                    factory,
                    &child_cancel,
                    Arc::clone(&shared),
                );
                shared.changed.notify_all();
                result
            })
            .map_err(|_| ScheduleError::SpawnFailed)?;
        state.allocation.turns += task.quota.model_turns;
        state.allocation.calls += task.quota.tool_calls;
        state.allocation.tokens += task.quota.tokens;
        state.allocation.history += task.quota.history_bytes;
        state.allocation.output += task.quota.tool_output_bytes;
        let started = Instant::now();
        state.entries.insert(
            id,
            Entry {
                snapshot: TaskSnapshot {
                    id,
                    state: TaskState::Running,
                    stop_requested: None,
                    writable_files: scope.write_paths.len(),
                    workspace_write_lease: self.config.policy.shell(&scope),
                    elapsed: Duration::ZERO,
                    tokens_reserved: 0,
                    model_turns: 0,
                    usage: ChildUsage::default(),
                },
                started,
                deadline: (started + task.quota.wall_time).min(self.shared.deadline),
                scope,
                cancel,
                handle: Some(handle),
                result: None,
            },
        );
        notify(&self.shared, &mut state);
        Ok(id)
    }

    pub fn cancel(&mut self, id: TaskId) -> Result<(), ScheduleError> {
        let mut state = lock(&self.shared.state);
        let entry = state
            .entries
            .get_mut(&id)
            .ok_or(ScheduleError::UnknownTask)?;
        stop(entry, StopRequest::ChildCancelled);
        notify(&self.shared, &mut state);
        Ok(())
    }

    pub fn stop_all(&mut self, reason: StopRequest) {
        let mut state = lock(&self.shared.state);
        state.closed.get_or_insert(reason);
        for entry in state.entries.values_mut() {
            stop(entry, reason);
        }
        notify(&self.shared, &mut state);
    }

    pub fn wait(&self, id: TaskId, timeout: Duration) -> Result<TaskSnapshot, ScheduleError> {
        let state = lock(&self.shared.state);
        if !state.entries.contains_key(&id) {
            return Err(ScheduleError::UnknownTask);
        }
        let (state, _) = self
            .shared
            .changed
            .wait_timeout_while(state, timeout.min(Duration::from_secs(1)), |s| {
                s.entries.get(&id).is_some_and(|e| e.handle.is_some())
                    && !self.shared.parent_cancel.load(Ordering::Acquire)
            })
            .unwrap_or_else(|e| e.into_inner());
        Ok(state.entries.get(&id).unwrap().snapshot.clone())
    }

    /// Transfers authoritative results once, to the parent only. Ownership of
    /// declared files remains reserved until this explicit collection decision.
    pub fn take_result(&mut self, id: TaskId) -> Result<TaskResult, ScheduleError> {
        let mut state = lock(&self.shared.state);
        let entry = state
            .entries
            .get_mut(&id)
            .ok_or(ScheduleError::UnknownTask)?;
        if matches!(entry.snapshot.state, TaskState::Collected(_)) {
            return Err(ScheduleError::AlreadyCollected);
        }
        let result = entry.result.take().ok_or(ScheduleError::NotFinished)?;
        entry.snapshot.state = TaskState::Collected(result.outcome.clone());
        notify(&self.shared, &mut state);
        Ok(result)
    }

    /// Parent tool guard; the enclosing `run_parent` calls this for every
    /// non-scheduler tool. No write may overlap an uncollected child assignment.
    fn check_parent(&self, call: &ToolInvocation<'_>) -> Result<(), ToolError> {
        let targets = self.config.policy.targets(call)?;
        if matches!(targets, policy::Targets::Read) {
            return Ok(());
        }
        let state = lock(&self.shared.state);
        let leases: Vec<_> = state
            .entries
            .values()
            .filter(|entry| {
                !matches!(entry.snapshot.state, TaskState::Collected(_))
                    && (!entry.scope.write_paths.is_empty()
                        || self.config.policy.shell(&entry.scope))
            })
            .collect();
        if leases.is_empty() {
            return Ok(());
        }
        match targets {
            policy::Targets::Read => (),
            policy::Targets::Workspace => return Err(ToolError::Denied),
            policy::Targets::Paths(paths) => {
                for path in paths {
                    // Fail closed on aliases while any child lease is live.
                    policy::write_path(path)?;
                    if leases.iter().any(|entry| {
                        self.config.policy.shell(&entry.scope)
                            || entry
                                .scope
                                .write_paths
                                .iter()
                                .any(|p| policy::overlaps(path, p))
                    }) {
                        return Err(ToolError::Denied);
                    }
                }
            }
        }
        Ok(())
    }

    fn find_id(&self, value: &str) -> Result<TaskId, ToolError> {
        lock(&self.shared.state)
            .entries
            .keys()
            .copied()
            .find(|id| id.to_string() == value)
            .ok_or(ToolError::InvalidArguments("task_id must name a child in this session"))
    }

    /// Joins all owned workers; never detach a tool thread on timeout. Built-in
    /// transports poll cancellation. A hostile blocking host callback cannot be
    /// preempted by std threads and keeps cleanup visibly pending.
    pub fn shutdown(&mut self) {
        if self.monitor.is_none() {
            return;
        }
        self.stop_all(StopRequest::Shutdown);
        {
            let mut state = lock(&self.shared.state);
            state.shutdown = true;
            notify(&self.shared, &mut state);
        }
        if let Some(handle) = self.monitor.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}
fn notify(shared: &Shared, state: &mut State) {
    state.revision = state.revision.wrapping_add(1);
    shared.changed.notify_all();
    if let Some(wake) = &state.wake {
        wake.unpark();
    }
}
fn quota_error() -> io::Error {
    io::Error::new(io::ErrorKind::QuotaExceeded, "child token quota exhausted")
}

fn stop(entry: &mut Entry, reason: StopRequest) {
    if entry.handle.is_some() && entry.snapshot.stop_requested.is_none() {
        entry.snapshot.stop_requested = Some(reason);
        entry.snapshot.state = TaskState::Stopping(reason);
        entry.cancel.store(true, Ordering::Release);
    }
}

fn monitor(shared: Arc<Shared>) {
    loop {
        let mut state = lock(&shared.state);
        let now = Instant::now();
        if shared.parent_cancel.load(Ordering::Acquire) {
            state.closed.get_or_insert(StopRequest::ParentCancelled);
        }
        if now >= shared.deadline {
            state.closed.get_or_insert(StopRequest::WallTime);
        }
        let closed = state.closed;
        let mut changed = false;
        for entry in state.entries.values_mut() {
            if entry.handle.is_none() {
                continue;
            }
            if let Some(reason) =
                closed.or_else(|| (now >= entry.deadline).then_some(StopRequest::WallTime))
            {
                if entry.snapshot.stop_requested.is_none() {
                    changed = true;
                }
                stop(entry, reason);
            }
            if entry.handle.as_ref().is_some_and(JoinHandle::is_finished) {
                // is_finished means the thread has returned; joining cannot
                // wait on model/tool I/O while holding the scheduler lock.
                let joined = entry.handle.take().unwrap().join();
                let mut result =
                    joined.unwrap_or_else(|_| empty_result(entry.snapshot.id, Outcome::Panicked));
                result.stop_requested = entry.snapshot.stop_requested;
                entry.snapshot.elapsed = entry.started.elapsed();
                entry.snapshot.tokens_reserved = result.tokens_reserved;
                entry.snapshot.model_turns = result.report.as_ref().map_or(0, |r| r.model_turns);
                entry.snapshot.state = TaskState::Finished(result.outcome.clone());
                entry.result = Some(result);
                changed = true;
            }
        }
        if changed {
            notify(&shared, &mut state);
        }
        if state.shutdown && state.entries.values().all(|e| e.handle.is_none()) {
            break;
        }
        let _ = shared
            .changed
            .wait_timeout(state, MONITOR_INTERVAL)
            .unwrap_or_else(|e| e.into_inner());
    }
}

fn empty_result(id: TaskId, outcome: Outcome) -> TaskResult {
    TaskResult {
        id,
        outcome,
        stop_requested: None,
        report: None,
        conversation: None,
        summary: String::new(),
        summary_truncated: false,
        tools: Vec::new(),
        tokens_reserved: 0,
        last_usage: None,
        journal_name: None,
    }
}

struct MeteredModel<'a> {
    model: &'a mut dyn ChildModel,
    meter: &'a mut TokenMeter,
}

impl Model for MeteredModel<'_> {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        self.meter.turn_charged = false;
        let result = self.model.complete(request, cancel, self.meter, on_text);
        if result.is_ok()
            && (!self.meter.turn_charged || self.meter.protocol_error || self.meter.exhausted)
        {
            self.meter.protocol_error = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "child adapter omitted token reservation",
            ));
        }
        if result.is_ok() {
            // Trusted adapters can still block. Never call one while holding
            // the metadata/control lock used by the UI.
            let last = self.model.usage();
            if let Some((shared, id)) = &self.meter.progress {
                let mut state = lock(&shared.state);
                if let Some(entry) = state.entries.get_mut(id) {
                    let usage = &mut entry.snapshot.usage;
                    usage.last = last;
                    usage.completed_responses += 1;
                    if let Some(cached) = usage.last.and_then(|last| last.cached_tokens) {
                        usage.responses_with_cached_usage += 1;
                        usage.reported_cached_tokens =
                            usage.reported_cached_tokens.saturating_add(cached);
                    }
                    notify(shared, &mut state);
                }
            }
        }
        result
    }
}

struct ScopedTools<'a> {
    inner: &'a mut dyn ToolExecutor,
    policy: &'a ToolPolicy,
    scope: &'a Scope,
    receipts: &'a mut Vec<ToolReceipt>,
}

impl ToolExecutor for ScopedTools<'_> {
    fn last_side_effect_uncertain(&self) -> bool {
        self.receipts
            .last()
            .is_some_and(|receipt| receipt.status == ToolStatus::Uncertain)
    }
    fn execute(
        &mut self,
        call: ToolInvocation<'_>,
        cancel: &AtomicBool,
    ) -> Result<String, ToolError> {
        if cancel.load(Ordering::Acquire) {
            return Err(ToolError::Cancelled);
        }
        let writes = self.policy.check(self.scope, &call)?;
        self.receipts.push(ToolReceipt {
            call_id: call.call.id.clone(),
            tool: call.call.name.clone(),
            writes,
            status: ToolStatus::Started,
        });
        let result = self.inner.execute(call, cancel);
        self.receipts.last_mut().unwrap().status = match &result {
            _ if writes && self.inner.last_side_effect_uncertain() => ToolStatus::Uncertain,
            Ok(_) => ToolStatus::Completed,
            Err(ToolError::Denied | ToolError::InvalidArguments(_) | ToolError::UnknownTool) => {
                ToolStatus::Denied
            }
            Err(ToolError::Cancelled) => ToolStatus::Cancelled,
            Err(ToolError::OutputLimit) => ToolStatus::OutputLimit,
            Err(ToolError::Failed) => ToolStatus::Failed,
        };
        // Do not replace a completed operation with "cancelled" merely because
        // its cancellation flag raced with return. agent::run retains the result.
        result
    }
}

fn execute_child(
    context: ChildContext,
    mut conversation: Conversation,
    policy: ToolPolicy,
    factory: Arc<Factory>,
    cancel: &AtomicBool,
    shared: Arc<Shared>,
) -> TaskResult {
    let mut result = empty_result(context.id, Outcome::Cancelled);
    let mut meter = TokenMeter {
        limit: context.quota.tokens,
        output: context.quota.output_tokens_per_turn,
        charged: 0,
        calls: 0,
        exhausted: false,
        protocol_error: false,
        turn_charged: false,
        progress: Some((shared, context.id)),
    };
    let run = catch_unwind(AssertUnwindSafe(|| {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        let mut worker = match factory(&context, cancel) {
            Ok(worker) => worker,
            Err(error) => {
                result.outcome = Outcome::SetupFailed(error.kind());
                return;
            }
        };
        if let Some(observer) = worker.observer.as_mut() {
            if observer.checkpoint(&conversation, &[]).is_err() {
                result.outcome = Outcome::Runtime(StopReason::JournalError);
                return;
            }
        }
        result.journal_name = worker.journal_name.take();
        let mut tools = ScopedTools {
            inner: worker.tools.as_mut(),
            policy: &policy,
            scope: &context.scope,
            receipts: &mut result.tools,
        };
        let mut model = MeteredModel {
            model: worker.model.as_mut(),
            meter: &mut meter,
        };
        let report = if let Some(observer) = worker.observer.as_mut() {
            agent::run_observed(
                &mut conversation,
                &mut model,
                &mut tools,
                context.quota.limits(),
                cancel,
                &mut |_| {},
                observer.as_mut(),
            )
        } else {
            agent::run(
                &mut conversation,
                &mut model,
                &mut tools,
                context.quota.limits(),
                cancel,
                &mut |_| {},
            )
        };
        result.outcome = if meter.exhausted {
            Outcome::QuotaTokens
        } else {
            match &report.reason {
                StopReason::Finished(FinishReason::Stop) => Outcome::Completed,
                StopReason::Finished(FinishReason::Other(_)) => Outcome::UnknownFinish,
                StopReason::Cancelled => Outcome::Cancelled,
                other => Outcome::Runtime(other.clone()),
            }
        };
        result.last_usage = worker.model.usage();
        result.report = Some(report);
    }));
    if run.is_err() {
        result.outcome = Outcome::Panicked;
    }
    if result.outcome == Outcome::Completed {
        if let Some(text) = conversation
            .messages()
            .last()
            .and_then(|m| m.value().get("content"))
            .and_then(crate::json::Value::as_str)
        {
            let mut end = text.len().min(MAX_SUMMARY_BYTES);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            result.summary = text[..end].to_owned();
            result.summary_truncated = end < text.len();
        }
    }
    result.tokens_reserved = meter.charged;
    result.conversation = Some(conversation);
    result
}

#[cfg(test)]
mod tests;
