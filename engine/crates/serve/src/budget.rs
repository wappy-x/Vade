//! Serving admission uses each request's prompt plus output limit. The CUDA
//! allocator independently enforces the aggregate physical memory ceiling.
use engine_model::qwen3::Config;
pub use spark_gpu_probe::admission::{bytes, refusal, CUDA, SERVE};

pub fn session_refusal(required: u128, available: usize, terms: &str, source: &str) -> String {
    refusal(required, available, &format!("{terms}; weights/rank already funded or not supplied; OUTPUT included in CONTEXT"), &format!("{source}; GPU-derived pools use {CUDA}, capped by the configured GPU ceiling and reserve"), SERVE)
}
pub fn token_refusal(required: u128, available: usize, input: usize, output: usize, source: &str) -> String {
    // Both request parsing and in-process validation reject zero output here.
    // It is invalid input, rather than a request exceeding a token ceiling.
    if required == 0 && output == 0 { return "max_tokens must be at least 1".into(); }
    spark_gpu_probe::admission::token_refusal(required, available, &format!("input={input}; OUTPUT/max_tokens={output}; configured token ceiling={available}; counts must be positive"), source, "request prompt length or max_tokens (vade --context, or server flags: --input-tokens / --context-tokens / --output-tokens)")
}

const MIB: usize = 1024 * 1024;
/// Ceiling bytes admission never hands to request/KV pools: per-tick transients
/// (draft-cohort switches allocated 218,898,432 bytes mid-generation) stay inside
/// the allocator ceiling, so an admitted request is not stopped for room.
pub const TRANSIENT_RESERVE: usize = 512 * MIB;
/// Bytes admission may hand to sessions out of `available` after the transient reserve.
pub fn after_transient_reserve(available: usize) -> usize { available.saturating_sub(TRANSIENT_RESERVE) }
const SCRATCH_ALLOWANCE: usize = 64 * MIB;
const KV_PAGE: usize = 2 * MIB;

/// One physical pool: resident/shared owners are funded before request/cache owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridBudget { pub sessions: usize }
/// Disjoint rank owners. Shared includes packing/gathers; request includes draft
/// and rollback. Retained/in-flight are private charges after shared-page credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridRankLedger {
    pub ceiling: usize, pub resident: usize, pub shared: usize, pub transport: usize,
    pub request: usize, pub retained: usize, pub in_flight: usize, pub measured_fixed: usize,
}
impl HybridRankLedger {
    pub fn values(self) -> [usize; 8] { [self.ceiling,self.resident,self.shared,self.transport,
        self.request,self.retained,self.in_flight,self.measured_fixed] }
    pub fn from_values(v: [usize; 8]) -> Self { Self { ceiling:v[0],resident:v[1],shared:v[2],
        transport:v[3],request:v[4],retained:v[5],in_flight:v[6],measured_fixed:v[7] } }
    pub fn fixed(self) -> Option<usize> { self.resident.checked_add(self.shared)?.checked_add(self.transport) }
    pub fn owners(self, active: usize) -> Option<usize> {
        self.request.checked_mul(active)?.checked_add(self.retained)?.checked_add(self.in_flight)
    }
}
impl HybridBudget {
    /// Each ceiling comes from that rank's measured capacity. Fund shared owners
    /// once, then use the smallest pool and largest private-owner bound everywhere.
    pub fn for_ranks(ledgers: &[HybridRankLedger], active: usize, requested: Option<usize>) -> Result<Self, String> {
        if ledgers.is_empty() || !(1..=32).contains(&active) { return Err("invalid hybrid rank admission".into()); }
        let (mut available,mut constrained,mut required)=(usize::MAX,0,0);
        for (rank,l) in ledgers.iter().copied().enumerate() {
            let fixed=l.fixed().ok_or("hybrid fixed owner overflow")?;
            if l.measured_fixed>fixed || l.request==0 { return Err(format!("hybrid rank {rank} invalid measured/owner ledger")); }
            let pool=l.ceiling.checked_sub(fixed).ok_or_else(||format!("hybrid rank {rank} fixed owners {fixed} exceed ceiling {}",l.ceiling))?;
            if pool<available { available=pool; constrained=rank; }
            required=required.max(l.owners(active).ok_or("hybrid private owner overflow")?);
        }
        let pool=Self::new(available,0,0,requested)?;
        if !pool.fits(Some(required)) { return Err(session_refusal(required as u128,pool.sessions,
            "worst rank's active + retained cache + in-flight owners; shared/transport funded once", &format!("minimum measured rank pool (rank={constrained})"))); }
        Ok(pool)
    }
    pub fn new(total: usize, resident: usize, shared: usize, requested: Option<usize>) -> Result<Self, String> {
        let fixed = resident.checked_add(shared).ok_or("hybrid resident/shared overflow")?;
        let available = total.checked_sub(fixed).ok_or_else(|| session_refusal(fixed as u128, total,
            "header-derived resident + shared workspace", "hybrid device ceiling"))?;
        let sessions = requested.unwrap_or(available);
        if sessions == 0 || sessions > available { return Err(session_refusal(sessions.max(1) as u128, available,
            "active + retained cache + in-flight owners", "hybrid session pool")); }
        Ok(Self { sessions })
    }
    pub fn active_limit(self, requested: Option<usize>, minimum: usize) -> Result<usize, String> {
        if minimum == 0 { return Err("zero hybrid request reservation".into()); }
        let funded = (self.sessions / minimum).min(32);
        let active = requested.unwrap_or(funded.min(16));
        if active == 0 || active > funded { return Err(session_refusal(active.max(1) as u128 * minimum as u128,
            self.sessions, "minimum request reservation x active slots", "hybrid session pool")); }
        Ok(active)
    }
    pub fn fits(self, reservation: Option<usize>) -> bool {
        reservation.is_some_and(|bytes| bytes != 0 && bytes <= self.sessions)
    }
}

/// CLI values stay optional until device memory and checkpoint geometry are known.
#[derive(Default)]
pub struct Sizing {
    pub context_tokens: Option<usize>,
    pub max_active: Option<usize>,
    pub kv_bytes: Option<usize>,
    pub session_bytes: Option<usize>,
    pub gpu_bytes: Option<usize>,
}

pub struct Sized {
    pub context_tokens: usize,
    pub draft_context_tokens: usize,
    pub max_active: usize,
    pub kv_bytes: usize,
    pub session_bytes: usize,
    pub gpu_bytes: usize,
}

/// Provisional minimum for weights and two slots; workspace is measured at startup.
pub fn minimum_free_bytes(config: &Config) -> Result<usize, String> {
    let weights = config.expected_shapes().values().try_fold(0u64, |total, shape| {
        shape.iter().try_fold(2u64, |bytes, &dimension| bytes.checked_mul(dimension))
            .and_then(|bytes| total.checked_add(bytes))
    }).and_then(|n| usize::try_from(n).ok()).ok_or("model weight size overflow")?;
    let sessions = Admission::for_config(config, 1, usize::MAX, usize::MAX)?
        .concurrent(2, usize::MAX)?.minimum_session_budget_bytes;
    let gpu = weights.checked_add(config.head_dim as usize * 4)
        .and_then(|n| n.checked_add(64 * MIB))
        .and_then(|n| n.checked_add(sessions)).ok_or("model memory size overflow")?;
    // Same OS/tokenizer reserve as Sizing::resolve: 10%, at least 1 GiB.
    let reserve = gpu.checked_mul(10).map(|n| n.div_ceil(9)).ok_or("memory reserve overflow")?;
    Ok(reserve.max(gpu.checked_add(1024 * MIB).ok_or("memory reserve overflow")?))
}

impl Sizing {
    pub fn resolve(&self, config: &Config, available_bytes: usize, draft: Option<&Config>) -> Result<Sized, String> {
        // Unified memory also serves the OS, tokenizer and CUDA driver. Leave
        // 10% (at least 1 GiB) free; this ceiling does not allocate that memory.
        let gpu_bytes = self.gpu_bytes.unwrap_or_else(||
            available_bytes.saturating_sub((available_bytes / 10).max(1024 * MIB)));
        let weights = |config: &Config| config.expected_shapes().values().try_fold(0u64, |total, shape| {
            shape.iter().try_fold(2u64, |bytes, &dimension| bytes.checked_mul(dimension))
                .and_then(|bytes| total.checked_add(bytes))
        }).and_then(|bytes| usize::try_from(bytes).ok()).ok_or("model weight size overflow");
        let weight_bytes = weights(config)?.checked_add(draft.map(weights).transpose()?.unwrap_or(0))
            .ok_or("combined model weight size overflow")?;
        // The model's context comes first. Slots share the active token pool;
        // short requests still use all slots when that scratch fits alongside
        // a full-context request.
        if self.max_active.is_some_and(|slots| slots < 2) {
            return Err("--max-active must be at least 2".into());
        }
        let plan = |tokens: usize, slots: Option<usize>| -> Result<Sized, String> {
            let draft_tokens = tokens.min(draft.map_or(0, |c| c.max_positions as usize));
            let rope = tokens.checked_mul(config.head_dim as usize)
                .and_then(|n| draft_tokens.checked_mul(draft.map_or(0, |c| c.head_dim) as usize)
                    .and_then(|d| n.checked_add(d)))
                .and_then(|n| n.checked_mul(4)).ok_or("RoPE size overflow")?;
            let fixed = weight_bytes.checked_add(rope).and_then(|n| n.checked_add(64 * MIB * (1 + usize::from(draft.is_some()))))
                .ok_or("model memory size overflow")?;
            let diagnostic = |required: u128| {
                let charge = Admission::for_config(config, tokens, usize::MAX, usize::MAX).and_then(|a| if let Some(draft) = draft { a.reserve_draft(Admission::for_config(draft, draft_tokens, usize::MAX, usize::MAX)?) } else { Ok(a) }).ok();
                let requested = charge.and_then(|a| a.concurrent(slots.unwrap_or(2), usize::MAX).ok()).map(|a| a.minimum_session_budget_bytes);
                refusal(required.max(fixed as u128 + self.session_bytes.or(requested).unwrap_or(0) as u128), gpu_bytes,
                    &format!("weights/rank {}; fixed weights/RoPE/loader {}; full reservation/session {} + scratch/extra-session {} x {} extra sessions; requested session pool {}; CONTEXT={tokens}; OUTPUT included in CONTEXT", bytes(weight_bytes as u128), bytes(fixed as u128), charge.map(|a| bytes(a.active_bytes as u128)).unwrap_or_else(|| "unavailable".into()), charge.map(|a| bytes(a.scratch_bytes as u128)).unwrap_or_else(|| "unavailable".into()), slots.unwrap_or(2).saturating_sub(1), self.session_bytes.or(requested).map(|n| bytes(n as u128)).unwrap_or_else(|| "automatic".into())),
                    &format!("{CUDA}; available input {}; capped by {}", bytes(available_bytes as u128), if self.gpu_bytes.is_some() { "--gpu-budget-mib" } else { "10% reserve (at least 1 GiB)" }), SERVE)
            };
            let remaining = gpu_bytes.checked_sub(fixed).ok_or_else(|| diagnostic(fixed as u128))?;
            let session_bytes = self.session_bytes.unwrap_or(remaining);
            if session_bytes > remaining { return Err(diagnostic(fixed as u128 + session_bytes as u128)); }
            // Provisional ceiling only. Startup allocates the workspace, then
            // finishes context/slot sizing against its measured resident bytes.
            let available = session_bytes;
            let inputs = |error| format!("{error}\nSizing inputs: weights/rank={weight_bytes} bytes; fixed={fixed} bytes; CONTEXT={tokens}; slots={slots:?}; unified available={available_bytes} bytes; GPU ceiling={gpu_bytes} bytes; session pool={available} bytes.");
            let mut admission = Admission::for_config(config, tokens,
                self.kv_bytes.unwrap_or(usize::MAX), available).map_err(inputs)?;
            if let Some(draft) = draft {
                admission = admission.reserve_draft(Admission::for_config(draft, draft_tokens,
                    self.kv_bytes.unwrap_or(usize::MAX), available).map_err(inputs)?)?;
            }
            let slots = slots.unwrap_or_else(|| 1 + available.saturating_sub(admission.active_bytes) / admission.scratch_bytes);
            admission.concurrent(slots, available).map_err(inputs)?;
            // Slots share a token pool: one long request must fit, but every
            // slot need not reserve the checkpoint's entire context ceiling.
            Ok(Sized { context_tokens: tokens, draft_context_tokens: draft_tokens, max_active: slots,
                kv_bytes: self.kv_bytes.unwrap_or(admission.kv_payload_bytes), session_bytes, gpu_bytes })
        };
        let slots = self.max_active.unwrap_or(2);
        let ceiling = usize::try_from(config.max_positions)
            .map_err(|_| "position ceiling overflow")?.min(262144);
        let tokens = if let Some(tokens) = self.context_tokens { tokens } else {
            let (mut low, mut high) = (1, ceiling);
            plan(low, Some(slots))?;
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if plan(middle, Some(slots)).is_ok() { low = middle; } else { high = middle - 1; }
            }
            low
        };
        plan(tokens, self.max_active)
    }
}

/// Finish the provisional plan after weights, RoPE and shared workspace have
/// been allocated. No fixed workspace allowance enters this decision.
impl Sizing {
    pub fn finish(&self, config: &Config, loaded_context: usize, ceiling: usize,
        draft: Option<(&Config, usize)>, slots: usize, kv_bytes: usize, available: usize)
        -> Result<(usize, usize, Admission), String> {
        let plan = |tokens, slots| -> Result<Admission, String> {
            let mut a = Admission::for_loaded_config(config, tokens, loaded_context, kv_bytes, available)?;
            if let Some((draft, window)) = draft {
                a = a.reserve_draft(Admission::for_loaded_config(draft, tokens.min(window), window, kv_bytes, available)?)?;
            }
            a.concurrent(slots, available)
        };
        let tokens = if self.context_tokens.is_some() { ceiling } else {
            let (mut low, mut high) = (1, ceiling);
            plan(low, 2)?;
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if plan(middle, 2).is_ok() { low = middle; } else { high = middle - 1; }
            }
            low
        };
        let slots = if self.max_active.is_some() { slots } else {
            let minimum = plan(tokens, 2)?;
            slots.min(2 + (available - minimum.minimum_session_budget_bytes) / minimum.scratch_bytes)
        };
        Ok((tokens, slots, plan(tokens, slots)?))
    }
}

pub fn parse_mib(text: &str, flag: &str) -> Result<usize, String> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{flag} requires a positive MiB integer"));
    }
    let mib: usize = text
        .parse()
        .map_err(|_| format!("{flag} integer overflow"))?;
    mib.checked_mul(MIB)
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| format!("{flag} must be positive and fit the host address space"))
}

// Per-session decode state, including generic attention's full-capacity scores
// and partials. Shared Multi prefill workspace is covered by the device budget.
fn scratch_allowance(config: &Config, tokens: usize, request_tokens: usize, prefill_tokens: usize) -> Option<usize> {
    let size = |n| usize::try_from(n).ok();
    let (hidden, intermediate, heads, kv_heads, dim, vocabulary) = (
        size(config.hidden_size)?, size(config.intermediate_size)?, size(config.attention_heads)?,
        size(config.kv_heads)?, size(config.head_dim)?, size(config.vocab_size)?,
    );
    let query = heads.checked_mul(dim)?;
    let kv_width = kv_heads.checked_mul(dim)?;
    let streaming = dim == 128 && (heads == kv_heads.checked_mul(4)? || heads == kv_heads.checked_mul(8)?);
    let (scores, parts) = if streaming {
        let count = if tokens > 16384 { 64.max(tokens.div_ceil(1024)) } else { tokens.div_ceil(256) };
        (1, heads.checked_mul(dim.checked_add(4)?)?.checked_mul(count.max(1))?)
    } else {
        (heads.checked_mul(tokens)?, query.checked_mul(tokens.div_ceil(64))?)
    };
    let lengths = [hidden, hidden, hidden.div_ceil(128), query, kv_width, kv_width,
        query, hidden, intermediate, intermediate, scores, parts, vocabulary,
        vocabulary.div_ceil(1024).checked_mul(3)?, 3];
    // Every arena view starts at a 64-byte boundary. Include both FFN vectors
    // even when packed projection kernels eliminate the second allocation.
    let bytes = lengths.into_iter().try_fold(0usize, |bytes, len|
        len.div_ceil(16).checked_mul(64).and_then(|n| bytes.checked_add(n)))?;
    // Long prefill keeps one session arena, including packed prefix K/V.
    // Bound both packed and unfused routes, and the <=16-row verifier logits.
    let rows = prefill_tokens.min(if config.num_experts != 0 { 16384 }
        else if request_tokens >= 6144 { 4096 } else { 512 }).max(1);
    let capacity = request_tokens.checked_add(KV_PAGE.div_ceil(kv_width.checked_mul(2)?))?
        .min(tokens).div_ceil(64).checked_mul(64)?;
    let batch_parts = heads.checked_mul(dim.checked_add(4)?)?
        .checked_mul(64.max(tokens.div_ceil(1024)))?.checked_mul(rows.min(16))?;
    let batch_scores = if streaming {1} else { heads.checked_mul(rows.min(64))?.checked_mul(capacity)? };
    let batch = [rows, rows.checked_mul(hidden)?, rows.checked_mul(hidden)?,
        rows.checked_mul(hidden.div_ceil(128))?, rows.checked_mul(query)?,
        rows.checked_mul(kv_width)?, rows.checked_mul(kv_width)?, rows.checked_mul(query)?,
        rows.checked_mul(hidden)?, rows.checked_mul(intermediate)?, rows.checked_mul(intermediate)?,
        batch_scores, batch_parts, 1,
        rows.div_ceil(16).checked_mul(16)?.checked_mul(intermediate.max(query).max(hidden))? / 2,
        capacity.checked_mul(kv_width)? / 2, capacity.checked_mul(kv_width)? / 2,
        rows.min(16).checked_mul(vocabulary)?];
    let arena = batch.into_iter().try_fold(0usize, |bytes, len|
        len.max(1).div_ceil(16).checked_mul(64).and_then(|n| bytes.checked_add(n)))?;
    let experts = if config.num_experts != 0 {
        let e = size(config.num_experts)?;
        let selected = size(config.num_experts_per_tok)?;
        let m = size(config.moe_intermediate_size)?;
        let routed = rows.checked_mul(selected)?;
        let alignment: usize = if routed >= 64usize.checked_mul(e)? {128} else {64};
        let padded = routed.checked_add((alignment - 1).checked_mul(e)?)?
            .div_ceil(alignment).checked_mul(alignment)?;
        let workspace = padded.checked_mul(hidden.checked_add(m / 2)?)?.checked_mul(4)?;
        let metadata = [rows.checked_mul(e)?, routed, routed, e.checked_mul(rows)?, e, e.checked_add(1)?, routed];
        metadata.into_iter().try_fold(workspace, |bytes, len|
            len.div_ceil(16).checked_mul(64).and_then(|n| bytes.checked_add(n)))?
    } else {0};
    // Covers scalar packing/graphs, overlapping attention-part growth, and
    // small scalar helper allocations beyond the explicit arena lengths.
    bytes.checked_add(arena)?.checked_add(experts)?.checked_add(SCRATCH_ALLOWANCE)
}

/// Keep the loader's logical FP32 ledger, but charge physical BF16 storage.
/// Paged vectors append in place; two extra pages per vector fund compact
/// growth/transition and a fork's unshareable tail, without doubling long KV.
pub fn backend_reservation(config: &Config, kv_payload_bytes: usize, context_tokens: usize) -> Option<usize> {
    backend_reservation_cached(config, kv_payload_bytes, context_tokens, 0)
}

/// A committed prefix needs no private prefill activations. Keep the complete
/// KV/packing bound, including materialization if its compact suffix grows.
pub fn backend_reservation_cached(config: &Config, kv_payload_bytes: usize, context_tokens: usize, cached_tokens: usize) -> Option<usize> {
    let vectors = usize::try_from(config.layers).ok()?.checked_mul(2)?;
    let stride = usize::try_from(config.kv_heads).ok()?
        .checked_mul(usize::try_from(config.head_dim).ok()?)?;
    let logical_token = vectors.checked_mul(stride)?.checked_mul(4)?;
    if logical_token == 0 || !kv_payload_bytes.is_multiple_of(logical_token) { return None; }
    let tokens = kv_payload_bytes / logical_token;
    if tokens == 0 || tokens > context_tokens || cached_tokens > tokens { return None; }
    let vector = tokens.checked_mul(stride)?.checked_mul(2)?;
    let paged = vector.div_ceil(KV_PAGE).checked_mul(KV_PAGE)?
        .checked_add(2 * KV_PAGE)?;
    let compact = tokens.checked_next_power_of_two()?.min(context_tokens)
        .checked_mul(stride)?.checked_mul(2)?.div_ceil(64).checked_mul(64)?.checked_mul(3)?;
    // Only the compact bound below a page avoids charging pages never mapped.
    let physical = if vector < KV_PAGE { compact.min(paged) } else { paged };
    physical.checked_mul(vectors)?.checked_add(scratch_allowance(config, context_tokens, tokens, tokens - cached_tokens)?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    /// Logical FP32 capacity for the loader only; never an admission charge.
    pub kv_payload_bytes: usize,
    pub backend_reservation_bytes: usize,
    /// Aggregate runnable pool; each request reserves only its own token limit.
    pub active_bytes: usize,
    /// Maximum accounted cache charge for one full-capacity retained entry.
    pub cached_entry_bytes: usize,
    /// Cache ceiling within the same pool; active reservations take precedence.
    pub retained_bytes: usize,
    pub minimum_session_budget_bytes: usize,
    scratch_bytes: usize,
}
impl Admission {
    /// Draft weights are outside this pool. Cached forks retain both models'
    /// immutable pages; each model funds its writable tail on admission.
    pub fn reserve_draft(mut self, draft: Self) -> Result<Self, String> {
        self.kv_payload_bytes = self.kv_payload_bytes.max(draft.kv_payload_bytes);
        self.active_bytes = self.active_bytes.checked_add(draft.backend_reservation_bytes).ok_or("draft active budget overflow")?;
        self.cached_entry_bytes = self.cached_entry_bytes.checked_add(draft.backend_reservation_bytes).ok_or("draft cache budget overflow")?;
        self.scratch_bytes = self.scratch_bytes.checked_add(draft.scratch_bytes).ok_or("draft scratch budget overflow")?;
        Ok(self)
    }
    /// Fund one full request and small concurrent slots in an elastic pool.
    pub fn concurrent(mut self, sessions: usize, budget: usize) -> Result<Self, String> {
        if sessions < 2 {
            return Err("--max-active must be at least 2".into());
        }
        let active = self.active_bytes
            .checked_add((sessions - 1).checked_mul(self.scratch_bytes).ok_or("active session budget overflow")?)
            .ok_or("active session budget overflow")?;
        self.minimum_session_budget_bytes = active;
        if budget < self.minimum_session_budget_bytes {
            return Err(session_refusal(active as u128, budget, &format!("one full-context reservation {} + {} scratch/session x {} extra sessions; CONTEXT not supplied to concurrent sizing", bytes(self.active_bytes as u128), bytes(self.scratch_bytes as u128), sessions - 1), "--session-budget-mib or the remaining GPU session pool after weights and shared workspace"));
        }
        // These are overlapping ceilings, not independent partitions. The
        // scheduler lends unreserved bytes to cache and reclaims them for work.
        self.active_bytes = budget;
        self.retained_bytes = budget;
        Ok(self)
    }
    pub fn for_config(
        config: &Config,
        tokens: usize,
        kv_budget_bytes: usize,
        session_budget_bytes: usize,
    ) -> Result<Self, String> {
        Self::for_loaded_config(config, tokens, tokens, kv_budget_bytes, session_budget_bytes)
    }
    pub fn for_loaded_config(config: &Config, tokens: usize, scratch_context: usize,
        kv_budget_bytes: usize, session_budget_bytes: usize) -> Result<Self, String> {
        if scratch_context < tokens { return Err("scratch context is below request capacity".into()); }
        if tokens == 0
            || u64::try_from(tokens).map_err(|_| "context token count overflow")?
                > config.max_positions
        {
            return Err(spark_gpu_probe::admission::token_refusal(tokens as u128, config.max_positions as usize, &format!("CONTEXT={tokens}; OUTPUT included in CONTEXT; context must be positive"), "the configured position ceiling", SERVE));
        }
        if kv_budget_bytes == 0 || session_budget_bytes == 0 {
            return Err("--kv-budget-mib and --session-budget-mib must be positive".into());
        }
        let layers = usize::try_from(config.layers).map_err(|_| "layer count overflow")?;
        let heads = usize::try_from(config.kv_heads).map_err(|_| "KV head count overflow")?;
        let dimension =
            usize::try_from(config.head_dim).map_err(|_| "KV head dimension overflow")?;
        let vocabulary =
            usize::try_from(config.vocab_size).map_err(|_| "vocabulary count overflow")?;
        if layers == 0 || heads == 0 || dimension == 0 || vocabulary == 0 {
            return Err("model session dimensions must be positive".into());
        }
        let vector_bytes = tokens
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(dimension))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .filter(|&v| v <= isize::MAX as usize)
            .ok_or("KV vector size exceeds host allocation bounds")?;
        let kv_payload_bytes = vector_bytes
            .checked_mul(2)
            .and_then(|v| v.checked_mul(layers))
            .ok_or("KV payload size overflow")?;
        if kv_payload_bytes > kv_budget_bytes {
            return Err(session_refusal(kv_payload_bytes as u128, kv_budget_bytes, &format!("KV payload/session {} x 1 session; CONTEXT={tokens}", bytes(kv_payload_bytes as u128)), "--kv-budget-mib (logical KV payload ceiling)"));
        }
        // Physical BF16 bytes plus bounded arena/page slack and prefill scratch.
        let backend_reservation_bytes =
            backend_reservation(config, kv_payload_bytes, scratch_context).ok_or("session backend reservation overflow")?;
        // History starts at a small Vec capacity and grows geometrically.
        // Prompt vectors are also live while history grows; a further history-
        // sized allowance covers the input IDs (including token-ID request copy).
        // Two logits vectors can overlap while replacing the previous result.
        // Cached entries retain only one; the cache also charges their actual
        // token/logit capacities rather than ignoring spare allocation.
        let history_bytes = tokens
            .max(4)
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or("session history capacity overflow")?;
        let logits_bytes = vocabulary
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|&v| v <= isize::MAX as usize)
            .ok_or("session logits capacity overflow")?;
        let cached_entry_bytes = backend_reservation_bytes
            .checked_add(history_bytes)
            .and_then(|v| v.checked_add(logits_bytes))
            .ok_or("cached session charge overflow")?;
        let active_bytes = cached_entry_bytes
            .checked_add(history_bytes)
            .and_then(|v| v.checked_add(logits_bytes))
            .ok_or("active session charge overflow")?;
        let minimum_session_budget_bytes = active_bytes;
        if session_budget_bytes < minimum_session_budget_bytes {
            return Err(session_refusal(minimum_session_budget_bytes as u128, session_budget_bytes, &format!("reservation/session {} x 1 session; CONTEXT={tokens}", bytes(minimum_session_budget_bytes as u128)), "--session-budget-mib or the remaining GPU session pool after weights and shared workspace"));
        }
        let retained_bytes = session_budget_bytes
            .checked_sub(active_bytes)
            .ok_or("aggregate session subtraction overflow")?;
        Ok(Self {
            kv_payload_bytes,
            backend_reservation_bytes,
            active_bytes,
            cached_entry_bytes,
            retained_bytes,
            minimum_session_budget_bytes,
            scratch_bytes: scratch_allowance(config, scratch_context, 1, 1).ok_or("session scratch size overflow")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        Config {
            qk_norm: true, fused_projections: false, llama_rope: None,
            hidden_size: 4096,
            intermediate_size: 12288,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            layers: 36,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_positions: 40960,
            bos_token: 151643,
            eos_token: 151645,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            yarn: false,
            tie_word_embeddings: false,
        }
    }
    #[test]
    fn transient_reserve_holds_back_half_a_gib_from_the_session_pool() {
        assert_eq!(TRANSIENT_RESERVE, 536_870_912);
        const { assert!(TRANSIENT_RESERVE >= 2 * 218_898_432) };
        assert_eq!(after_transient_reserve(TRANSIENT_RESERVE - 1), 0);
        assert_eq!(after_transient_reserve(TRANSIENT_RESERVE + 7), 7);
        // Release dry run (128 GB Spark, CONTEXT 32768): ceiling 110402312602,
        // resident 84324392132, shared 7305861892, charge 2315449668 per chat.
        // Without the reserve 8 chats left 248 MB, below the 271 MB growth seen.
        let pool = |reserve: usize| 110_402_312_602usize - 84_324_392_132 - 7_305_861_892 - reserve;
        assert_eq!(pool(0) / 2_315_449_668, 8);
        assert_eq!(after_transient_reserve(pool(0)), pool(TRANSIENT_RESERVE));
        let chats = after_transient_reserve(pool(0)) / 2_315_449_668;
        assert_eq!(chats, 7);
        assert!(pool(0) - chats * 2_315_449_668 >= TRANSIENT_RESERVE);
    }
    #[test]
    fn default_context_retains_existing_budgets_with_explicit_active_accounting() {
        let a = Admission::for_config(
            &config(),
            2048,
            1 << 30,
            parse_mib("4096", "--session-budget-mib").unwrap(),
        )
        .unwrap();
        assert_eq!(a.kv_payload_bytes, 603_979_776);
        assert_eq!((a.backend_reservation_bytes, a.active_bytes, a.cached_entry_bytes, a.minimum_session_budget_bytes, a.retained_bytes),
            (816_555_648, 817_803_904, 817_179_776, 817_803_904, 3_477_163_392));
        assert_eq!(a.active_bytes + a.retained_bytes, 4usize << 30);
        assert!(a.retained_bytes >= a.cached_entry_bytes);
    }
    #[test]
    fn admission_boundaries_fail_before_upload_with_the_relevant_flag() {
        let cfg = config();
        let good = Admission::for_config(&cfg, 2048, 1 << 30, 4usize << 30).unwrap();
        for (required, available, error) in [(603_979_776, 603_979_775, Admission::for_config(&cfg, 2048, good.kv_payload_bytes - 1, 4usize << 30).unwrap_err()), (817_803_904, 817_803_903, Admission::for_config(&cfg, 2048, good.kv_payload_bytes, good.active_bytes - 1).unwrap_err()), (good.active_bytes + good.scratch_bytes, good.active_bytes, good.concurrent(2, good.active_bytes).unwrap_err())] {
            for text in ["Requested size does not fit.", "CONTEXT", "OUTPUT", "Available:", "Gap:", "--gpu-budget-mib", &bytes(required as u128), &bytes(available as u128), &bytes((required-available) as u128)] { assert!(error.contains(text), "{error}"); }
        }
        for gpu in [1, 16usize << 30] { let e = Sizing { context_tokens: Some(2048), gpu_bytes: Some(gpu), session_bytes: Some(usize::MAX), ..Default::default() }.resolve(&cfg, 121usize << 30, None).err().unwrap(); assert!(e.contains("weights/rank") && e.contains("Linux MemAvailable (free + reclaimable page cache)") && e.contains(&bytes(gpu as u128)), "{e}"); }
        assert!(
            Admission::for_config(&cfg, 2048, good.kv_payload_bytes - 1, 4usize << 30)
                .unwrap_err()
                .contains("--kv-budget-mib")
        );
        assert!(
            Admission::for_config(
                &cfg,
                2048,
                good.kv_payload_bytes,
                good.minimum_session_budget_bytes
            )
            .is_ok()
        );
        assert!(
            Admission::for_config(
                &cfg,
                2048,
                good.kv_payload_bytes,
                good.minimum_session_budget_bytes - 1
            )
            .unwrap_err()
            .contains("--session-budget-mib")
        );
        assert!(Admission::for_config(&cfg, 3640, 1 << 30, usize::MAX).is_ok());
        assert!(Admission::for_config(&cfg, 3641, 1 << 30, usize::MAX).is_err());
        let full = Admission::for_config(&cfg, 40960, 12_079_595_520, usize::MAX).unwrap();
        assert_eq!(full.kv_payload_bytes, 12_079_595_520);
        let concurrent = full.concurrent(2, 72usize << 30).unwrap();
        assert_eq!(
            (concurrent.active_bytes, concurrent.retained_bytes),
            (72usize << 30, 72usize << 30)
        );
        assert!(full.concurrent(32, 72usize << 30).is_ok());
        assert_eq!(Sizing::default().resolve(&cfg, 121usize << 30, None).unwrap().context_tokens, 40960);
        let mut yarn = cfg.clone();
        yarn.enable_yarn().unwrap();
        let sized = Sizing::default().resolve(&yarn, 121usize << 30, Some(&cfg)).unwrap();
        assert_eq!((sized.context_tokens, sized.draft_context_tokens), (131072, 40960));
        assert!(full.concurrent(32, full.active_bytes + 31 * full.scratch_bytes).is_ok());
        assert!(Admission::for_config(&cfg, 40961, usize::MAX, usize::MAX).is_err());
        assert!(Admission::for_config(&cfg, 0, usize::MAX, usize::MAX).is_err());
        let mut overflow = cfg.clone();
        overflow.layers = u64::MAX;
        assert!(Admission::for_config(&overflow, 2048, usize::MAX, usize::MAX).is_err());
        assert!(backend_reservation(&config(), usize::MAX, 2048).is_none());
    }
    #[test]
    fn mib_flags_reject_zero_negative_fractional_and_overflowing_values() {
        assert_eq!(parse_mib("1024", "--kv-budget-mib").unwrap(), 1 << 30);
        for value in ["", "0", "-1", "+1", "1.5", "18446744073709551615"] {
            assert!(
                parse_mib(value, "--kv-budget-mib")
                    .unwrap_err()
                    .contains("--kv-budget-mib")
            );
        }
    }
}
