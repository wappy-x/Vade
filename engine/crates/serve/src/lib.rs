//! Bounded HTTP JSON/SSE responses and cooperative single-engine-thread scheduling.
//! Mutable KV belongs to one request; retained conversation snapshots fork into
//! independently writable sessions within an explicit aggregate memory budget.
pub mod budget;
pub mod kv_pages;
pub mod contract;
pub mod metadata;
#[cfg(test)] #[path = "../build.rs"] #[allow(dead_code)] mod build_metadata;
pub mod usage;
pub mod warm_estimate;
pub mod cache;
pub mod chat_output;
pub mod generation;
pub mod http;
pub mod auth;
pub mod protocol;
mod responses;
mod anthropic;
#[cfg(target_os = "linux")]
pub mod qwen;
#[cfg(target_os = "linux")]
pub mod hybrid;
pub mod scheduler;
pub mod route_receipt;
pub mod tokenizer;
pub mod text_program;
pub mod text_checkpoint;

pub mod mtp_batch;
mod cohort_cost;
pub mod prefix_switch;
#[cfg(target_os = "linux")]
pub mod flashnext;

use std::time::Duration;

/// A backend session appends one token's KV on each successful step. On any
/// error the caller drops the complete session; partially changed KV is not reused.
pub trait Session {
    /// Configure request-specific state only after its reservation is admitted.
    fn configure_request(&mut self, _request: &protocol::Request) -> Result<(), String> {
        Ok(())
    }
    /// Actual prompt prefix restored by a replaying backend, excluding rebuilt rows.
    /// None means ordinary KV forks preserve every retained token.
    fn cached_prompt_tokens(&self, _prompt_tokens: usize) -> Option<usize> { None }
    /// Set before prompt work on both fresh sessions and cached forks. False
    /// skips optional retention snapshots, not canonical replay or restored KV.
    fn set_cache_retention(&mut self, _retain: bool) {}
    fn disable_model_draft(&mut self) {}
    /// Drop retained state kept only to replay one exact prompt. Under pressure the
    /// cache calls this before evicting any whole entry.
    fn release_replay_snapshot(&mut self) {}
    /// Proposal capacity funded by this request's current admission reservation.
    fn reserve_model_draft(&mut self, _tokens: usize) {}
    /// The tokens admission charges this request, before `configure_request` restores a cached
    /// prefix: a restored session maps the pages a new session of that charge maps.
    fn admission_tokens(&mut self, _tokens: usize) {}
    fn automatic_draft(&mut self, _prompt: usize, _remaining: usize) {}
    /// Local exact verification; other backends retain their ordinary step path.
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> { None }
    /// Allocated KV and peak device bytes observed by this request.
    fn memory_bytes(&self) -> Option<(usize, usize)> { None }
    fn draft_model_memory_bytes(&self) -> Option<(usize, usize)> { None }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String>;
    /// Commit prompt KV without computing vocabulary logits.
    fn advance(&mut self, token: u32) -> Result<(), String> {
        self.step(token).map(|_| ())
    }
    /// Append a prompt chunk, returning logits only for its final position.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let (&last, prefix) = tokens.split_last().ok_or("empty prefill")?;
        for &token in prefix { self.advance(token)?; }
        self.step(last)
    }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens { self.advance(token)?; }
        Ok(())
    }
    /// Page-cache read-ahead of the prompt continuing at the current position (hints
    /// only; no state or bytes change). Backends without host-side tables ignore it.
    fn read_ahead(&self, _tokens: &[u32]) {}
    /// Preferred idle prompt chunk; layer-yielding backends may use larger rows.
    fn prefill_chunk_size(&self, _prompt_end: usize) -> usize { 512 }
    /// Smallest scheduling chunk that avoids replaying an unfinished canonical tile.
    fn prefill_chunk_min(&self) -> usize { 1 }
    /// The session's own group size applies whenever no other chat waits on it,
    /// whatever the server's small default is. Sessions opt in.
    fn prefill_owns_chunks(&self) -> bool { false }
    /// Resume the same prompt chunk until committed. None retains partial work;
    /// backends without layer-level yielding complete their ordinary chunk.
    /// No budget completes the chunk without synchronization between layers.
    fn prefill_slice(&mut self, tokens: &[u32], final_chunk: bool, _budget: Option<Duration>, _prompt_end: usize)
        -> Result<Option<Vec<f32>>, String> {
        if final_chunk { self.prefill(tokens).map(Some) }
        else { self.prefill_advance(tokens).map(|()| Some(Vec::new())) }
    }
    /// Whether a snapshot can be forked before the caller evicts cache entries.
    fn can_fork(&self) -> bool { true }
    /// Whether this exact truncation has retained recurrent state, without allocating.
    fn can_fork_prefix(&self, _position: usize) -> bool { self.can_fork() }
    /// Refresh only a proposal model's prefix, preserving canonical target KV.
    fn retain_draft_prefix(&mut self, _source: &Self) -> Result<bool, String> where Self: Sized { Ok(false) }
    /// Whether this finished session holds proposal state `retain_draft_prefix`
    /// could keep. The cache forks a snapshot for it only when this is true.
    fn has_draft_prefix(&self) -> bool { false }
    /// Independent branch. Admitted by the caller before allocation.
    fn fork(&self) -> Result<Self, String>
    where
        Self: Sized,
    {
        Err("backend does not support session forks".into())
    }
    /// Fork for prompt appends, which may require canonical prefill replay.
    /// Decode rollback keeps its separate accumulation and commit semantics.
    fn fork_prefix(&self, position: usize) -> Result<Self, String> where Self: Sized {
        let mut session = self.fork()?;
        session.truncate(position)?;
        Ok(session)
    }
    /// Bound for allocations retained while idle, including spare KV capacity.
    /// None charges the full active reservation to snapshots as well.
    fn retained_bytes(&self) -> Option<usize> {
        None
    }
    /// Shared physical allocation identities used to charge snapshots once.
    /// None retains the conservative per-session byte bound.
    fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { None }
    /// Release turn-exclusive resources after completion snapshots are published.
    fn retire(&mut self) {}
    /// Private scratch/state bound in addition to enumerated shared allocations.
    fn retained_private_bytes(&self) -> usize { 0 }
    /// Storage a fork adds that `retained_allocations` does not list yet: a snapshot copy the
    /// backend funds from its own retained budget only when it forks. The cache makes room
    /// for these bytes before it forks.
    fn fork_storage_bytes(&self) -> usize { 0 }
    /// Settle state a prompt's checkpoint still owes other ranks. The scheduler calls it at the
    /// prompt's end, before a snapshot or a KV checkpoint copies that state. Before a snapshot,
    /// the step's first token, if it has one, is written first.
    fn settle_prefix(&mut self) -> Result<(), String> { Ok(()) }
    /// Prefix guaranteed reusable on append; recurrent backends replay a tail.
    fn checkpoint_reusable_tokens(&self, tokens: usize) -> usize { tokens }
    /// Actual immutable allocation bound after completion, including scratch.
    /// None keeps the active reservation until the session is destroyed.
    fn retirement_bytes(&self) -> Option<usize> { None }
    /// Immutable complete BF16 prefix pages; admission funds their aliases once.
    fn prefix_allocations(&self, _tokens: usize) -> Option<Vec<(usize, usize)>> { None }
    /// Immutable snapshots already included in a full request reservation.
    /// Funding these identities never reduces that reservation. The adapter
    /// must keep publication and fork lifetimes within its snapshot bound.
    /// Some(tokens) restricts funding to a fork of that incoming prompt size;
    /// None describes actual allocations owned or published by a live request.
    fn funded_snapshot_allocations(&self, _tokens: Option<usize>) -> Vec<(usize, usize)> { Vec::new() }
    /// Retain exactly the requested committed prefix. Unsupported backends
    /// safely fall back to a fresh session when a conversation prompt changes.
    fn truncate(&mut self, _position: usize) -> Result<(), String> {
        Err("backend does not support prefix truncation".into())
    }
    /// Expand the last compact decode result without advancing the session.
    /// None requests the ordinary truncate-and-replay fallback.
    fn decode_logits(&mut self, _rows: usize) -> Result<Option<Vec<Vec<f32>>>, String> { Ok(None) }
    /// Finish a successful non-retained request before its terminal response.
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Pending GPU work stays on the engine's creating thread.
pub trait LongPrefill<S> {
    /// Authentication failures must fail the request instead of retrying locally.
    fn allows_fallback(&self, _error: &str) -> bool { true }
    /// None retains partial GPU/peer work for the next scheduler turn.
    /// On error, leave the session ready for ordinary local prefill.
    fn advance(&mut self, session: &mut S, budget: Duration) -> Result<Option<Vec<f32>>, String>;
}

/// Neither engine nor session requires Send/Sync.
pub trait Engine {
    type Session<'a>: Session
    where
        Self: 'a;
    /// Select test kernels before allocating a fresh request session.
    fn session_for_request(&self, _request: &protocol::Request) -> Result<Self::Session<'_>, String> { self.session() }
    /// Admission: a fresh session for a request whose reservation covers `tokens` tokens.
    fn session_for_admission(&self, request: &protocol::Request, _tokens: usize) -> Result<Self::Session<'_>, String> { self.session_for_request(request) }
    fn session(&self) -> Result<Self::Session<'_>, String>;
    /// Canonical prompt snapshots, independent forks and accounted shared KV.
    fn kv_branch_handles(&self) -> bool { false }
    /// Start optional coordinated prefill. None leaves the session untouched
    /// for ordinary sliced prefill; dropping the job cancels its peer work.
    fn prefill_long<'s>(&'s self, _session: &Self::Session<'s>, _tokens: &[u32])
        -> Result<Option<Box<dyn LongPrefill<Self::Session<'s>> + 's>>, String> { Ok(None) }
    /// Select cohort-dependent host policy before admission and each scheduler tick.
    fn active_cohort(&self, _chats: usize) {}
    /// Enable the default-off depth-one cost chooser for complete C6..8 cohorts.
    fn cohort_cost_control(&self) -> bool { false }
    /// VADE_C8_GUARD_ROBUST. Default false keeps embedded engines on the 2% latch.
    fn cohort_guard_robust(&self) -> bool { false }
    /// A prefix-limited drafter can offer this conservative coverage hint.
    fn cohort_draft_head_limit(&self) -> Option<usize> { None }
    /// None preserves the ordinary policy; false temporarily suspends proposals,
    /// retaining history; true prepares the chooser's speculative path.
    fn set_cohort_choice(&self, _draft: Option<bool>) {}
    /// Only complete measured cohorts may reclaim unused execution storage.
    /// A partial fairness tick keeps its path without releasing pooled storage.
    fn prepare_cohort_path(&self, _draft: bool) -> Result<(), BatchError> { Ok(()) }
    fn shared_batch(&self) -> bool { false }
    /// Independent prompts can share one prefill call without sharing decode scheduling.
    /// The engine must fund its execution of the prefill limits below.
    fn shared_prefill_batch(&self) -> bool { self.shared_batch() }
    /// Observation only; these hooks must not change scheduling or session state.
    fn profile_tick_begin(&self, _tick: u64) {}
    fn profile_tick_end(&self) {}
    /// Diagnostic observation only. Normal engines do not take these timestamps.
    fn profile_serving(&self) -> bool { false }
    fn profile_stream_output(&self, _request: u64, _tokens: usize, _ns: u128) {}
    fn greedy_draft_only(&self) -> bool { false }
    /// A greedy-draft-only engine that still drafts sampled requests in strict mode
    /// (seed- and position-keyed verification; Flash-Next VADE_MTP_SAMPLED).
    fn sampled_draft(&self) -> bool { false }
    /// sampled_draft engines whose cohorts verify sampled rows in the shared pass.
    fn sampled_cohort_draft(&self) -> bool { false }
    fn native_draft_policy(&self) -> bool { false }
    /// The prefix cache serves only requests with `cache:true` or a `prompt_cache_key`; the
    /// scheduler treats every other request as `cache:false` (no automatic reuse or gating).
    fn prefix_cache_opt_in(&self) -> bool { false }
    fn shared_prompt_limit(&self) -> usize { if self.shared_batch() { 64 } else { 8 } }
    fn shared_draft(&self) -> bool { false }
    /// Keep draft work off concurrent chats until this engine wins that gate.
    fn single_chat_draft(&self) -> bool { false }
    /// Largest active cohort that may draft; larger cohorts decode plain.
    fn draft_chat_limit(&self) -> usize { if self.single_chat_draft() { 1 } else { usize::MAX } }
    /// Discard optional history when requests overlap; that turn then remains
    /// plain because omitted target residuals cannot be recovered from KV.
    fn discard_contended_draft(&self) -> bool { false }
    fn automatic_draft_allowed(&self, _chats: usize) -> bool { true }
    fn shared_draft_costs(&self, _chats: usize) -> Option<spark_engine::draft::CostCurve> { None }
    fn propose_batch<'s>(&'s self, _rows: &mut [DraftRows<'_, Self::Session<'s>>])
        -> Result<Vec<spark_engine::draft::ModelProposal>, BatchError> {
        Err(BatchError { request: None, message: "shared draft unavailable".into() })
    }
    fn shared_model_drafts(&self) -> bool { false }
    /// Reattach the proposal model to a cached fork whose snapshot dropped it.
    fn restore_model_draft<'s>(&'s self, _session: &mut Self::Session<'s>, _prompt: usize) -> Result<(), String> { Ok(()) }
    fn automatic_draft_limit(&self) -> usize { 4 }
    fn automatic_draft_min_output(&self) -> usize { 0 }
    fn decode_row_limit(&self, chats: usize) -> usize { chats.div_ceil(16) * 16 }
    /// Rows one tick gives a chat that runs alone on an engine without shared batches. More
    /// than one lets its draft target verify proposals. A plain chat still uses one row.
    fn lone_chat_rows(&self) -> usize { 1 }
    /// Rows one tick gives several chats on an engine without shared batches: one decode row
    /// for each of up to this many chats, all in one `step_batch` call. One row keeps the turns.
    fn shared_decode_rows(&self) -> usize { 1 }
    /// The fewest chats in decode for which a tick gives `shared_decode_rows`. A tick with fewer keeps one row.
    fn shared_decode_min(&self) -> usize { 1 }
    /// Rows one tick may verify across all chats; None keeps the configured budget.
    fn tick_row_limit(&self) -> Option<usize> { None }
    /// Give a drafting cohort's decoders equal row caps (the tick remainder stays spare);
    /// false keeps the sequential split, which hands the remainder to the last chat.
    fn even_cohort_rows(&self) -> bool { false }
    fn prepare_drafts<'s>(&'s self, rows: &mut [BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<Option<Vec<u32>>>, BatchError> { Ok(rows.iter().map(|_| None).collect()) }
    fn shared_prefill_limit(&self) -> usize { 64 }
    fn shared_prefill_rows(&self) -> usize { 1024 }
    /// Total prompt length admitted to shared prefill, including cached tokens.
    fn shared_prefill_max_tokens(&self) -> usize { usize::MAX }
    /// Canonical independent prompt tiles; zero keeps the serial long path.
    fn shared_prefill_chunk_rows(&self) -> usize { 0 }
    /// Keep one admission slot free of background requests for interactive work.
    fn reserve_foreground_slot(&self) -> bool { true }
    /// Fund the complete output allowance before admitting a chat.
    fn reserve_full_output(&self) -> bool { false }
    /// Current device allocation, lifetime peak, and retained pinned host staging.
    fn memory_bytes(&self) -> Option<(usize, usize, usize)> { None }
    /// A sticky device fault requires a failed health response and process exit.
    fn fatal_error(&self) -> Option<String> { None }
    /// Return idle KV pages or reclaim them when system memory is under pressure.
    fn maintain_memory(&self, _idle: bool) -> Result<(), String> { Ok(()) }
    /// Startup, before the first request: backends with reserves build and drop one
    /// context-size admission session, so the first request reuses its memory as later ones do.
    /// An error is logged and serving starts anyway.
    fn warm_admission(&self) -> Result<(), String> { Ok(()) }
    /// Admission of a queued request starts. It runs before the cache balance, cache forks
    /// and session construction, which can allocate or poll memory pressure; it ends an idle
    /// period that maintain_memory(true) started.
    fn admission_started(&self) {}
    /// Refresh cached memory pressure after a retired session releases ownership,
    /// before the next retirement or admission. Backends without reserves do nothing.
    fn refresh_memory_pressure(&self) -> Result<(), String> { Ok(()) }
    /// Prompt-only rows may use the model's tensor-core prefill arithmetic.
    fn prefill_batch<'s>(&'s self, rows: &mut [BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<Predictions>, BatchError> { self.step_batch(rows) }
    fn step_batch<'s>(&'s self, rows: &mut [BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<Predictions>, BatchError> {
        rows.iter_mut().map(|row| {
            let result = match row.output {
                BatchOutput::All => row.tokens.iter().map(|&t| row.session.step(t)).collect(),
                BatchOutput::Last => row.session.prefill(row.tokens).map(|v| vec![v]),
                BatchOutput::None => row.session.prefill_advance(row.tokens).map(|_| Vec::new()),
            };
            result.map(Predictions::Logits).map_err(|message| BatchError { request: None, message })
        }).collect()
    }
    /// Validate engine-specific input before admission mutates sessions or transport.
    fn validate_request(&self, _prompt: &[u32], _request: &protocol::Request) -> Result<(), String> { Ok(()) }
    fn context_limit(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn eos_token_ids(&self) -> &[u32];
    /// Admission for this request's prompt and maximum generated suffix.
    fn request_reservation_bytes(&self, _tokens: usize) -> Option<usize> {
        self.session_reservation_bytes()
    }
    fn request_option_bytes(&self, _request: &protocol::Request) -> usize {
        0
    }
    /// Admission when this request will not allocate draft-model KV.
    fn request_reservation_bytes_plain(&self, tokens: usize) -> Option<usize> { self.request_reservation_bytes(tokens) }
    /// Cached plain rows fund only the remaining prefill activation rows.
    fn request_reservation_bytes_plain_cached(&self, tokens: usize, _cached: usize) -> Option<usize> {
        self.request_reservation_bytes_plain(tokens)
    }
    /// Live retention policy; other cartridges keep their existing ownership bounds.
    fn request_reservation_bytes_with_retention(&self, tokens: usize, plain: bool, cached: Option<usize>, _retain: bool) -> Option<usize> {
        if plain { match cached { Some(n) => self.request_reservation_bytes_plain_cached(tokens,n), None => self.request_reservation_bytes_plain(tokens) } }
        else { self.request_reservation_bytes(tokens) }
    }
    /// Engines that retain prompt-only intermediates can separate them from the suffix.
    /// The default preserves the existing reservation for every other engine.
    fn request_reservation_bytes_for_prompt(&self, _prompt: usize, tokens: usize, plain: bool, cached: Option<usize>, retain: bool) -> Option<usize> {
        self.request_reservation_bytes_with_retention(tokens, plain, cached, retain)
    }
    /// Conservative session allocation bound, including spare KV capacity and transient work.
    /// None disables reusable sessions for backends without an audited bound.
    fn session_reservation_bytes(&self) -> Option<usize> {
        None
    }
    /// Device bytes held now that no reservation funds (elastic graph captures). The
    /// prefix cache leaves them free: unfunded retention never takes their pages.
    fn cache_hold_bytes(&self) -> usize { 0 }
    /// Device bytes one live request may hold beyond its admitted reservation (`retain`:
    /// its retention flag). The prefix cache leaves them free for the request to grow into.
    fn unfunded_request_bytes(&self, _retain: bool) -> usize { 0 }
    /// Physical bytes the engine can spare beyond its own needs while `live_reserved` bytes are
    /// reserved (negative: short), and the slack retention leaves free when it grows. The prefix
    /// cache yields only past what it holds plus the first, and grows only while the slack also
    /// stays free, so admission never waits on memory a full cache took from the engine's spare
    /// session pages. None: the session pool bounds the cache alone.
    fn cache_free_physical_bytes(&self, _live_reserved: usize) -> Result<Option<(i128, usize)>, String> { Ok(None) }
    /// Bytes the startup plan sets aside for retained prefixes alone: admission does not
    /// use them, so neither admission nor host memory pressure reclaims them.
    fn cache_room_bytes(&self) -> usize { 0 }
}

pub struct DraftRows<'s, S> {
    pub session: &'s mut S,
    pub history: &'s [u32],
    pub next: u32,
    pub limit: usize,
}

#[derive(Clone, Copy)]
pub enum BatchOutput { All, Last, None }
pub struct BatchRows<'s, S> {
    pub session: &'s mut S, pub tokens: &'s [u32], pub output: BatchOutput, pub greedy: bool,
    /// Zero requests full logits; nonzero is a penalty-free sampling shortlist.
    pub top_k: usize,
}
/// `Rejected` fails only its own request: the engine kept that request's
/// session uncommitted while the batch's other rows committed exactly.
pub enum Predictions { Logits(Vec<Vec<f32>>), Greedy(Vec<u32>), TopK(Vec<Vec<(u32, f32)>>), Rejected(String) }
/// A failed batch invalidates every participant: earlier rows may have advanced
/// before a later failure. `request` identifies the cause, never a safe survivor.
pub struct BatchError { pub request: Option<usize>, pub message: String }

/// Incremental text decoding must retain incomplete UTF-8 until a later token.
pub trait Decoder {
    fn push(&mut self, token: u32) -> Result<String, String>;
    fn finish(&mut self) -> Result<String, String>;
}

pub trait Tokenizer: Sync {
    type Decoder<'a>: Decoder
    where
        Self: 'a;
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
    fn decoder(&self) -> Self::Decoder<'_>;
    fn chat_decoder(&self) -> Self::Decoder<'_> { self.decoder() }
    fn gemma_chat(&self) -> bool { false }
    fn glm_chat(&self) -> bool { false }
    fn chat_thinking_open(&self, _prompt: &[u32]) -> bool { false }
    fn has_token(&self, token: u32) -> bool;
    /// A checkpoint stop set that the backend must match before generation.
    /// None keeps the backend's existing stop contract.
    fn checkpoint_eos_token_ids(&self) -> Result<Option<&[u32]>, String> { Ok(None) }
    fn encode_chat(&self, input: &protocol::ChatInput, max_bytes: usize) -> Result<Vec<u32>, String>;
    /// Qwen3.5-family chat: XML tool calls and a prompt that opens the think block.
    fn hybrid_chat(&self) -> bool { false }
}

/// Defaults intentionally fit a small, correctness-first service. CLI flags
/// may change context/output bounds; application hard limits still apply.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Test-only Flash two-module service, frozen at startup.
    pub ab_ptx: bool,
    /// VADE_AB_COHORTS=1 with ab_ptx: concurrent same-arm cohorts instead of one request at a time.
    pub ab_cohorts: bool,
    /// Optional startup credential, enforced by the shared HTTP intake.
    pub api_key: Option<auth::ApiKey>,
    /// Loaded immutable artifacts; absent for in-process engines without a checkpoint.
    pub identity: Option<contract::Identity>,
    /// Cached build, loaded PTX, checkpoint and execution receipt.
    pub metadata: Option<metadata::Metadata>,
    /// Additional exact HTTP authorities and browser origins allowed by the operator.
    pub allowed_hosts: Vec<String>,
    pub allowed_origins: Vec<String>,
    pub header_bytes: usize,
    pub body_bytes: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub context_tokens: usize,
    pub output_bytes: usize,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    /// Fatal request deadline: default 3,600 seconds, maximum 86,400 seconds. Queue wait
    /// has the same bound; admission starts a fresh generation deadline.
    pub generation_timeout: Duration,
    /// Thinking for chat and Responses requests that do not choose it themselves.
    pub default_thinking: bool,
    /// The chat template reads `/think` and `/no_think` in the last user message (Qwen).
    /// False for a template with no such switch (GLM): there the words are plain text
    /// and never change thinking.
    pub thinking_soft_switch: bool,
    /// Checkpoint defaults; request fields override these independently.
    pub default_sampling: Option<spark_engine::sample::Sampling>,
    /// FlashNext direct requests with no named sampling fields use greedy.
    /// Requests naming any field retain checkpoint defaults for the rest.
    pub greedy_if_unsampled: bool,
    /// Only FlashNext accepts per-request depth; other engines ignore the field.
    pub next_mtp_depth: Option<usize>,
    /// Burst intake window (see `intake_burst`); zero keeps the first prompt
    /// pass of an idle batch unheld. VADE_INTAKE_BURST_MS overrides it.
    pub intake_burst: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            ab_ptx: false,
            ab_cohorts: false,
            api_key: None,
            identity: None,
            metadata: None,
            allowed_hosts: Vec::new(),
            allowed_origins: Vec::new(),
            header_bytes: 16 * 1024,
            body_bytes: 256 * 1024,
            input_tokens: 1536,
            output_tokens: 512,
            context_tokens: 2048,
            output_bytes: 1024 * 1024,
            read_timeout: Duration::from_secs(15),
            write_timeout: Duration::from_secs(15),
            generation_timeout: Duration::from_secs(3600),
            default_thinking: true,
            thinking_soft_switch: true,
            default_sampling: None,
            greedy_if_unsampled: false,
            next_mtp_depth: None,
            intake_burst: intake_burst::DEFAULT_WINDOW,
        }
    }
}
impl Limits {
    /// Long diagnostic runs can use this bound. Ordinary services keep the default.
    pub const MAX_GENERATION_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);
    /// Preserve endpoint-specific admission errors while raising byte intake.
    /// The tokenizer's decode allowance is four times its input byte limit.
    pub fn tokenizer_limits(&self) -> engine_tokenizer::Limits {
        engine_tokenizer::Limits {
            max_input_bytes: self.body_bytes.max(self.output_bytes.div_ceil(4)),
            max_tokens: engine_tokenizer::Limits::default().max_tokens,
        }
    }
    /// Largest reply for a nonempty prompt on the loaded engine. Each rendered
    /// request can have less room and is clamped before admission.
    pub fn output_limit(&self, context: usize) -> usize {
        self.output_tokens.min(self.context_tokens.min(context).saturating_sub(1))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.header_bytes == 0
            || self.header_bytes > 64 * 1024
            || self.body_bytes == 0
            || self.body_bytes > 4 * 1024 * 1024
            || self.input_tokens == 0
            || self.input_tokens > 262144
            || self.output_tokens == 0
            || self.output_tokens > 262144
            || self.context_tokens == 0
            || self.context_tokens > 262144
            || self.output_bytes < 1024
            || self.output_bytes > 16 * 1024 * 1024
            || [
                self.read_timeout,
                self.write_timeout,
            ]
            .iter()
            .any(|d| d.is_zero() || *d > Duration::from_secs(3600))
            || self.generation_timeout.is_zero()
            || self.generation_timeout > Self::MAX_GENERATION_TIMEOUT
            || self.intake_burst > intake_burst::MAX_WINDOW
        {
            return Err("invalid service limits".into());
        }
        Ok(())
    }
}

/// Safe identifier accepted as request `model`. This is an identifier, not an
/// assertion that an unverified checkpoint matches a published model.
pub const MODEL_ID: &str = "qwen3-8b";

#[cfg(target_os = "linux")]
pub mod gemma4;

/// Strict startup switch for collecting an initial burst of short prompts.
pub mod intake_grace;
/// Burst intake: an idle batch waits a bounded window for the rest of a chat burst.
pub mod intake_burst;
