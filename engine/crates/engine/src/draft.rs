//! Interruptible speculative decoding. Every returned token already has KV.
use crate::sample::{Distribution, Sampler, Sampling, uniform};
use std::collections::HashMap;
use std::time::Instant;
#[path = "draft_steps.rs"] mod steps; pub use steps::StepRule;
type Result<T> = std::result::Result<T, String>;
/// VADE_MTP_SAMPLED=1 (default off): verified sampled rows draw in parallel (sample_rows).
fn parallel_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("VADE_MTP_SAMPLED").as_deref(), Ok("1" | "2")))
}

/// Implement this on a local or distributed session. Batches append one KV row
/// per input; strict verification must match `step` exactly from that prefix.
/// After an error, checkpoint remains at or beyond the entry position and
/// truncate(entry) restores it. Distributed truncation must be collective.
/// Greedy ties choose the lowest token ID.
pub trait Target {
    fn checkpoint(&self) -> usize;
    fn truncate(&mut self, position: usize) -> Result<()>;
    fn step(&mut self, token: u32) -> Result<Vec<f32>>;
    fn step_greedy(&mut self, token: u32) -> Result<u32> {
        crate::greedy_token(&self.step(token)?)
    }
    fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        tokens.iter().map(|&token| self.step(token)).collect()
    }
    fn forward_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> { self.verify_batch(tokens) }
    fn kv_bf16(&self) -> bool { false }
    fn max_verify_rows(&self) -> usize { 16 }
    /// This target's own ms for passes of 1, 2, 4, 8, 16 rows at this context. None keeps the caller's prior.
    fn verify_costs(&self, _context: usize) -> Option<CostCurve> { None }
    fn model_draft_width(&self) -> usize { 0 }
    /// Explicit target capability for sampled model proposals. Strict verification
    /// still samples canonical target rows with the request's seed and position.
    fn sampled_model_draft(&self) -> bool { false }
    fn speculative_model_draft(&self) -> bool { false }
    fn propose_sampled_model(&mut self, _history: &[u32], _prompt_len: usize, _next: u32,
        _limit: usize, _sampling: Sampling, _allowed: &dyn Fn(u32) -> bool)
        -> Result<(ModelProposal, Vec<Distribution>)> { Err("target has no sampled proposal distribution".into()) }
    fn native_draft_policy(&self) -> bool { false }
    fn model_draft_budget(&mut self, _remaining: usize) -> Result<()> { Ok(()) }
    fn plan_model_draft(&mut self, _history: &[u32], _limit: usize) -> Result<Option<usize>> { Ok(None) }
    /// Native policy, shared drafting: Some(n) reserves up to n proposals now and leaves
    /// them to the scheduler's one Engine::propose_batch call per tick (complete_model).
    fn defer_model_draft(&self, _limit: usize) -> Option<usize> { None }
    fn model_draft_exhausted(&self) -> bool { false }
    fn model_draft_catchup(&self) -> (usize, f64) { (0, 0.0) }
    fn model_draft_costs(&self) -> Option<CostCurve> { None }
    /// Native policies: a greedy chat alone drafts up to the resident depth when the target
    /// uses the step rule (draft_steps.rs). The rule can stop the chain before each step.
    /// With VADE_MTP_CONF=off, or when the rule does not apply, the chooser uses this request's
    /// measured depth survival and costs to choose the depth.
    fn adaptive_model_depth(&self) -> bool { false }
    /// Native policies: the model (not lookup) depth for the next proposal.
    fn limit_model_depth(&mut self, _depth: usize) {}
    /// This cycle's stop for a target that stops drafts step by step (draft_steps.rs): it then drafts the full width. Each finished cycle takes the confidence and ends the stop.
    fn draft_step_rule(&mut self, _rule: StepRule) -> bool { false } fn take_draft_confidence(&mut self) -> Vec<f32> { Vec::new() }
    /// Adaptive model depth: this target's measured ms for one verify pass of `rows` rows,
    /// shared by every request it serves. When known, the chooser uses it as is instead of
    /// the cost prior times this request's learned scale.
    fn model_verify_ms(&self, _rows: usize) -> Option<f64> { None }
    /// Adaptive model depth: this target's recent model-draft survival at `depth` across
    /// the requests it serves. A new request starts from it instead of the 0.8 prior.
    fn model_survival_prior(&self, _depth: usize) -> Option<f64> { None }
    fn propose_model(&mut self, _history: &[u32], _next: u32, _limit: usize) -> Result<Option<ModelProposal>> { Ok(None) }
}
pub struct ModelProposal {
    pub tokens: Vec<u32>,
    /// Wall time for proposals, including any previously queued short KV catchup.
    pub proposal_ms: f64,
    /// Host time for prefix catchup; long prefill catchup also waits for its GPU work.
    pub catchup_ms: f64,
}
#[cfg(target_os = "linux")]
impl Target for crate::qwen::Session<'_, '_, '_> {
    fn checkpoint(&self) -> usize { self.checkpoint() }
    fn truncate(&mut self, p: usize) -> Result<()> { self.truncate(p) }
    fn step(&mut self, t: u32) -> Result<Vec<f32>> { self.step(t) }
    fn step_greedy(&mut self, t: u32) -> Result<u32> { self.step_greedy(t) }
    fn verify_batch(&mut self, t: &[u32]) -> Result<Vec<Vec<f32>>> { self.verify_batch(t) }
    fn forward_batch(&mut self, t: &[u32]) -> Result<Vec<Vec<f32>>> { self.forward_batch(t) }
    fn kv_bf16(&self) -> bool { self.kv_bf16() }
    fn model_draft_width(&self) -> usize { self.draft_model_width() }
    fn sampled_model_draft(&self) -> bool { self.config().num_experts == 0 }
    fn model_draft_exhausted(&self) -> bool { self.draft_model_exhausted() }
    fn model_draft_catchup(&self) -> (usize, f64) { self.draft_model_catchup() }
    fn model_draft_costs(&self) -> Option<CostCurve> {
        // GB10 canonical 30B-A3B verification at widths 1/2/4/8/16.
        // Request-local observed timings scale this prior after every cycle.
        (self.config().num_experts != 0).then_some(CostCurve([31.919, 46.979, 67.484, 101.722, 179.450]))
    }
    fn propose_model(&mut self, history: &[u32], next: u32, limit: usize) -> Result<Option<ModelProposal>> {
        self.propose_model(history, next, limit)
    }
}
/// Six suffix lengths, updated once per appended token. Expected O(1) work per
/// token, independent of context length. Hash collisions are checked against IDs.
/// Retain the whole request so long-context drafts can quote its beginning.
#[derive(Default)]
pub struct Lookup {
    index: HashMap<(u64, u8), usize>,
    seen: usize,
}
impl Lookup {
    pub fn propose(&mut self, history: &[u32], limit: usize, output: &mut Vec<u32>) -> usize {
        if self.seen > history.len() { self.clear(); }
        for end in self.seen..history.len() {
            let mut hash = 0u64;
            for n in 1..=32.min(end) {
                hash = hash.wrapping_mul(0x100000001b3).wrapping_add(history[end-n] as u64 + 1);
                if n.is_power_of_two() { self.index.insert((hash, n as u8), end); }
            }
        }
        self.seen = history.len();
        let mut hash = 0u64;
        let mut found = None;
        for n in 1..=32.min(history.len()) {
            hash = hash.wrapping_mul(0x100000001b3).wrapping_add(history[history.len()-n] as u64 + 1);
            if n.is_power_of_two()
                && let Some(&end) = self.index.get(&(hash, n as u8))
                && end < history.len() && history[end-n..end] == history[history.len()-n..] {
                        found = Some((end, n));
                    }
        }
        if let Some((end, n)) = found {
            output.extend_from_slice(&history[end..history.len().min(end + limit)]);
            n
        } else { 0 }
    }
    fn clear(&mut self) { self.index.clear(); self.seen = 0; }
    pub fn memory_bytes(&self) -> usize {
        if self.index.capacity() == 0 { return 0; }
        self.index.capacity().next_power_of_two() * (std::mem::size_of::<((u64, u8), usize)>() + 1) + 16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Strict, Fast, Sampled }

/// Measured milliseconds for 1, 2, 4, 8, 16 rows, scaled by useful decode cycles.
#[derive(Clone, Copy, Debug)]
pub struct CostCurve(pub [f64; 5]);
impl CostCurve {
    pub fn gb10(mode: Mode) -> Self {
        // Original GB10 measurements at 1/4/8/16 rows; interpolate the 2-row prior.
        Self(match mode {
            Mode::Strict | Mode::Sampled => [67.776, 67.314333, 66.391, 69.706, 79.633],
            Mode::Fast => [67.1, 72.966667, 84.7, 86.7, 89.8],
        })
    }
    /// Setup-only calibration; the caller must reserve sixteen temporary rows.
    /// Reuse the result for the same target configuration and similar context.
    /// Rollback restores the prefix; temporary KV/scratch capacity may remain.
    pub fn measure(target: &mut (impl Target + ?Sized), tokens: &[u32], mode: Mode) -> Result<Self> {
        if tokens.len() != 16 { return Err("cost calibration needs sixteen input tokens".into()); }
        let entry = target.checkpoint();
        let mut costs = [0.0; 5];
        // Warm the largest batch once; ongoing decode refines these seed costs.
        for (probe, rows) in [16,1,2,4,8,16].into_iter().enumerate() {
            let start = Instant::now();
            let result = if rows == 1 { target.step(tokens[0]).map(|row| vec![row]) } else { match mode {
                Mode::Strict | Mode::Sampled => target.verify_batch(&tokens[..rows]),
                Mode::Fast => target.forward_batch(&tokens[..rows]),
            } };
            let ms = start.elapsed().as_secs_f64()*1000.0;
            let end = target.checkpoint();
            let rollback = target.truncate(entry);
            let output = result.map_err(|error| match &rollback {
                Ok(()) => error,
                Err(rollback) => format!("{error}; calibration rollback to {entry} also failed: {rollback}"),
            })?;
            rollback?;
            if output.len() != rows || end != entry + rows || target.checkpoint() != entry {
                return Err("invalid target calibration contract".into());
            }
            if probe > 0 { costs[probe-1] = ms.max(f64::EPSILON); }
        }
        Ok(Self(costs))
    }
    /// Measured short/2000-token STRICT slope; ongoing timings refine the scale.
    pub fn at_context(mode: Mode, tokens: usize) -> Self {
        let mut curve = Self::gb10(mode);
        if mode != Mode::Fast {
            for (time, slope) in curve.0.iter_mut().zip([0.0011335, 0.001704167, 0.0028455, 0.0045495, 0.008549]) {
                *time += tokens as f64 * slope;
            }
        }
        curve
    }
    fn ms(self, rows: usize) -> f64 {
        let knots = [1,2,4,8,16];
        for i in 1..5 {
            if rows <= knots[i] {
                let f = (rows-knots[i-1]) as f64/(knots[i]-knots[i-1]) as f64;
                return self.0[i-1]+f*(self.0[i]-self.0[i-1]);
            }
        }
        self.0[4]
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub max_new: usize,
    /// Total target rows per cycle, including its first, already selected token.
    pub max_rows: usize,
    pub mode: Mode,
    pub adaptive: bool,
    /// Request-local catchup gate; explicit draft modes leave this disabled.
    pub automatic: bool,
    pub costs: CostCurve,
    /// Per-logit FAST error envelope. Empirical guard, not an exactness proof.
    pub guard_error: f64,
}
impl Config {
    pub fn strict(max_new: usize) -> Self {
        Self { max_new, max_rows: 16, mode: Mode::Strict,
            adaptive: true, automatic: false, costs: CostCurve::gb10(Mode::Strict), guard_error: 0.01 }
    }
}

/// Prior ms of one proposed model token at the cost prior's own scale (cost_scale 1.0).
const TOKEN_PRIOR_MS: f64 = 4.0;
/// The native chooser's draft price reads this many verify scales and draft token times.
const PRICE_WINDOW: usize = 16;
struct Control {
    rates: [[f64; 15]; 3],
    guard_rate: [f64; 3],
    cost_scale: f64,
    draft_ms: f64,
    model_token_ms: Option<f64>,
    learn_ms: f64,
    /// This request's first verify scales and draft ms per proposed token, then their price.
    price_scales: Vec<f64>, price_tokens: Vec<f64>, price: Option<f64>,
}
impl Control {
    fn new() -> Self {
        Self { rates: [[0.25; 15], [0.90; 15], [0.50; 15]], guard_rate: [0.0; 3],
            cost_scale: 1.0, draft_ms: 0.0, model_token_ms: None, learn_ms: 0.0,
            price_scales: Vec::new(), price_tokens: Vec::new(), price: None }
    }
    /// The native chooser's ms for one proposed token, in units of the cost prior. It is the
    /// prior's TOKEN_PRIOR_MS until the request has PRICE_WINDOW samples of each kind; then
    /// it is the ratio of their medians as the nearest power of two, fixed for the request.
    /// Run-to-run noise (a few percent) moves it only at a level boundary, so the width
    /// follows the request's verified ids.
    fn draft_price(&self) -> f64 { self.price.unwrap_or(TOKEN_PRIOR_MS) }
    fn observe_price(&mut self, scale: Option<f64>, token_ms: Option<f64>) {
        if self.price.is_some() { return; }
        for (window, value) in [(&mut self.price_scales, scale), (&mut self.price_tokens, token_ms)] {
            if let Some(value) = value.filter(|v| v.is_finite() && *v > 0.0 && window.len() < PRICE_WINDOW) { window.push(value); }
        }
        if self.price_scales.len() < PRICE_WINDOW || self.price_tokens.len() < PRICE_WINDOW { return; }
        let median = |values: &[f64]| { let mut v = values.to_vec(); v.sort_by(f64::total_cmp); v[v.len() / 2] };
        let ratio = median(&self.price_tokens) / median(&self.price_scales);
        self.price = Some(if ratio.is_finite() && ratio > 0.0 { 2f64.powi(ratio.log2().round() as i32) } else { TOKEN_PRIOR_MS });
    }
    fn cost(&self, rows: usize, prior: CostCurve) -> f64 {
        self.cost_scale * prior.ms(rows)
    }
    fn score(&self, class: usize, rows: usize, costs: CostCurve) -> f64 {
        let (mut survival, mut expected) = (1.0, 1.0);
        for &rate in &self.rates[class][..rows-1] {
            survival *= rate * (1.0-self.guard_rate[class]);
            expected += survival;
        }
        let guard_probability = 1.0-(1.0-self.guard_rate[class]).powi(rows as i32);
        let draft_ms = self.model_token_ms.filter(|_| class == 0).map_or(self.draft_ms, |ms| ms * (rows - 1) as f64);
        expected / (self.cost(rows, costs) + draft_ms + self.learn_ms * rows as f64
            + guard_probability*self.cost(1, costs))
    }
    fn rows(&self, class: usize, available: usize, costs: CostCurve) -> usize {
        let (mut best, mut score) = (1, 1.02*self.score(class, 1, costs));
        for rows in 2..=available {
            let rate = self.score(class, rows, costs);
            if rate > score { best = rows; score = rate; }
        }
        best
    }
    fn observe(&mut self, class: usize, hits: &[bool]) {
        let alpha = if class == 0 && self.model_token_ms.is_some() { 0.20 } else { 0.35 };
        for (depth, &hit) in hits.iter().take(15).enumerate() {
            self.rates[class][depth] = self.rates[class][depth] * (1.0-alpha) + f64::from(hit) * alpha;
            if !hit { break; }
        }
    }
    fn observe_cost(&mut self, rows: usize, target_ms: f64, learn_ms: f64, prior: CostCurve) {
        self.observe_price(Some(target_ms/prior.ms(rows)), None);
        self.cost_scale = 0.75*self.cost_scale + 0.25*target_ms.max(f64::EPSILON)/prior.ms(rows);
        self.learn_ms = 0.75*self.learn_ms + 0.25*learn_ms / rows as f64;
    }
    fn observe_guard(&mut self, class: usize, checks: usize, fires: usize) {
        if checks > 0 {
            self.guard_rate[class] = 0.75*self.guard_rate[class] + 0.25*fires as f64/checks as f64;
        }
    }
}

/// Raw top-one successors, including rows beyond the accepted prefix. A
/// single vocabulary-sized table holds IDs only; logits are never retained.
struct Recycling { successors: Vec<u32>, contexts: HashMap<(u64, u8), u32> }
impl Recycling {
    fn new(vocab: usize) -> Self { Self { successors: vec![u32::MAX; vocab], contexts: HashMap::new() } }
    /// Retain short conditional successors from every verified row, including
    /// rejected prefixes. They only propose tokens; STRICT still verifies all.
    fn observe_contexts(&mut self, history: &[u32], input: &[u32], predictions: &[u32]) {
        for row in (0..input.len()).rev() {
            if predictions[row] == u32::MAX { continue; }
            let end = history.len()+row+1;
            let mut hash = 0u64;
            for n in 1..=8.min(end) {
                let pos = end-n;
                let token = if pos < history.len() { history[pos] } else { input[pos-history.len()] };
                hash = hash.wrapping_mul(0x100000001b3).wrapping_add(token as u64+1);
                if n >= 2 && n.is_power_of_two() { self.contexts.insert((hash, n as u8), predictions[row]); }
            }
        }
    }
    fn observe(&mut self, token: u32, logits: &[f32]) -> u32 {
        if self.successors.is_empty() || logits.len() > u32::MAX as usize { return u32::MAX; }
        let (mut next, mut best) = (u32::MAX, f32::NEG_INFINITY);
        for (id, &value) in logits.iter().enumerate() {
            if !value.is_finite() { return u32::MAX; }
            if value > best { next = id as u32; best = value; }
        }
        if let Some(slot) = self.successors.get_mut(token as usize) { *slot = next; }
        next
    }
    fn extend(&self, history: &[u32], limit: usize, output: &mut Vec<u32>, allowed: &impl Fn(u32) -> bool) {
        while output.len() < limit {
            let end = history.len()+output.len();
            let token_at = |pos: usize| if pos < history.len() { history[pos] } else { output[pos-history.len()] };
            let mut next = self.successors.get(token_at(end-1) as usize).copied().unwrap_or(u32::MAX);
            let mut hash = 0u64;
            for n in 1..=8.min(end) {
                hash = hash.wrapping_mul(0x100000001b3).wrapping_add(token_at(end-n) as u64+1);
                if n >= 2 && n.is_power_of_two()
                    && let Some(&id) = self.contexts.get(&(hash, n as u8)) { next = id; }
            }
            if next == u32::MAX || !allowed(next) { break; }
            output.push(next);
        }
    }
    fn memory_bytes(&self) -> usize {
        let contexts = if self.contexts.capacity() == 0 { 0 } else {
            self.contexts.capacity().next_power_of_two() * (std::mem::size_of::<((u64, u8), u32)>()+1)+16
        };
        self.successors.capacity()*4+contexts
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub cycles: usize,
    pub rows: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub guarded: usize,
    pub guard_fires: usize,
    pub rollbacks: usize,
    pub max_rows: usize,
    pub draft_ms: f64,
    pub model_proposal_ms: f64,
    pub model_catchup_ms: f64,
    pub model_proposals: usize,
    /// Useful target calls by row count; index zero is plain decoding.
    pub widths: [usize; 16],
    /// A hash (FNV-1a steps from zero) of the row count of each of those calls, in order:
    /// equal plans hash equal.
    pub plan_hash: u64,
    /// The native chooser's draft price at the end (Control::draft_price).
    pub draft_price: f64,
    pub recycle_ms: f64,
    pub target_ms: f64,
    /// Current target cost estimates at widths 1/2/4/8/16, including interpolation.
    pub costs_ms: [f64; 5],
    /// Sources: recycling, suffix lookup, rejected tail.
    /// Empty-draft serial cycles are not attributed to a source.
    pub source_cycles: [usize; 3],
    pub source_accepted: [usize; 3],
    /// Conditional survival, including free serial shadow comparisons.
    pub depth_trials: [[usize; 15]; 3],
    pub depth_hits: [[usize; 15]; 3],
    /// Model proposal survival by context: <4K, 4K..16K, >=16K.
    pub model_band_trials: [[usize; 15]; 3],
    pub model_band_hits: [[usize; 15]; 3],
}

/// Request-owned decoding state. The target can be cached or preempted after
/// every call. No returned token is pending; `history()` equals its KV prefix.
/// Prepared exact rows can share one weight pass with unrelated sessions.
pub struct Proposal {
    pub input: Vec<u32>,
    entry: usize, drafts: [Vec<u32>; 3], class: usize, draft_ms: f64,
    started: Instant, target_ms: Option<f64>,
    model_limit: usize,
    model_draft_used: bool,
    reserved_rows: Option<usize>,
    distributions: Vec<Distribution>,
}
impl Proposal {
    pub fn model_limit(&self) -> usize { self.model_limit }
    /// A model produced proposals, even if filtering or verification rejects them.
    /// The remaining proposal budget is independent of this completed work.
    pub fn model_draft_used(&self) -> bool { self.model_draft_used }
    pub fn row_budget(&self) -> usize { self.reserved_rows.unwrap_or(self.input.len() + self.model_limit) }
    pub fn row_count(&self) -> usize { self.row_budget() }
    pub fn replace_model_draft(&mut self, mut input: Vec<u32>, ms: f64, stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<()> {
        if input.first() != self.input.first() || input.len() > self.row_count() { return Err("invalid batched model proposal".into()); }
        self.model_draft_used |= input.len() > 1;
        if let Some(n) = input.iter().position(|&t| !allowed(t)) { input.truncate(n.max(1)); }
        if let Some(n) = input.iter().position(|&t| stop(t)) { input.truncate(n + 1); }
        self.drafts[0] = input[1..].to_vec(); self.input = input; self.draft_ms = ms; self.reserved_rows = None; Ok(())
    }
    /// Time only the shared target call, excluding queueing and other preparation.
    pub fn record_target_ms(&mut self, ms: f64) { self.target_ms = Some(ms); }
    fn elapsed_ms(&self) -> f64 {
        self.target_ms.unwrap_or_else(|| self.started.elapsed().as_secs_f64()*1000.0)
    }
}

pub struct Speculator {
    config: Config,
    options: Sampling,
    sampler: Sampler,
    /// VADE_MTP_SAMPLED: one scratch sampler per verified row (parallel draws, sample_rows).
    row_samplers: Vec<Sampler>,
    /// VADE_MTP_SAMPLED=2: this cycle's row draws taken ahead (presample), keyed by history length.
    presampled: Option<(usize, Vec<Result<u32>>)>,
    history: Vec<u32>,
    prompt_len: usize,
    next: u32,
    done: bool,
    skip_model: bool,
    lookup: Lookup,
    recycling: Recycling,
    tail: Vec<u32>,
    seed: Vec<u32>,
    control: Control,
    steps: steps::Steps,
    guard: Guard,
    pub stats: Stats,
}
impl Speculator {
    pub fn new(prompt: &[u32], logits: &[f32], config: Config, options: Sampling) -> Result<Self> {
        Self::new_filtered(prompt, logits, config, options, |_| true)
    }
    pub fn new_filtered(prompt: &[u32], logits: &[f32], mut config: Config, options: Sampling, allowed: impl Fn(u32) -> bool) -> Result<Self> {
        if options.compact_top_k() == 0 && config.mode == Mode::Sampled { config.mode = Mode::Strict; }
        if prompt.is_empty() || !(1..=16).contains(&config.max_rows)
            || !config.guard_error.is_finite() || config.guard_error < 0.0
            || !(config.costs.0 == [0.0; 5] || config.costs.0.iter().all(|t| t.is_finite() && *t > 0.0)) {
            return Err("invalid draft configuration or empty prompt".into());
        }
        if config.costs.0 == [0.0; 5] { config.costs = CostCurve::at_context(config.mode, prompt.len()); }
        let mut sampler = Sampler::new(options)?;
        let next = sampler.sample(logits, prompt, prompt.len(), prompt.len(), allowed)?;
        let mut recycling = Recycling::new(if config.max_rows > 1 && config.max_new > 1 { logits.len() } else { 0 });
        recycling.observe(*prompt.last().unwrap(), logits);
        Ok(Self { config, options, sampler, row_samplers: Vec::new(), presampled: None, history: prompt.to_vec(), prompt_len: prompt.len(),
            next, done: config.max_new == 0, skip_model: false, lookup: Lookup::default(),
            recycling, tail: Vec::new(), seed: Vec::new(),
            control: Control::new(), steps: Default::default(), guard: Guard::default(), stats: Stats::default() })
    }
    pub fn history(&self) -> &[u32] { &self.history }
    /// Resume at the committed KV prefix with the already sampled pending token.
    /// This lets serving enter STRICT after earlier plain rows without replay.
    pub fn from_selected(history: &[u32], prompt_len: usize, next: u32, vocab: usize,
        mut config: Config, options: Sampling) -> Result<Self> {
        if options.compact_top_k() == 0 && config.mode == Mode::Sampled { config.mode = Mode::Strict; }
        if prompt_len == 0 || prompt_len > history.len() || history.len()-prompt_len >= config.max_new
            || !(1..=16).contains(&config.max_rows) || next as usize >= vocab {
            return Err("invalid pending draft state".into());
        }
        let mut recycling = Recycling::new(vocab);
        if let Some(slot) = recycling.successors.get_mut(*history.last().unwrap() as usize) { *slot = next; }
        Ok(Self { config, options, sampler: Sampler::new(options)?, row_samplers: Vec::new(), presampled: None, history: history.to_vec(),
            prompt_len, next, done: false, skip_model: false, lookup: Lookup::default(), recycling,
            tail: Vec::new(), seed: Vec::new(), control: Control::new(), steps: Default::default(), guard: Guard::default(), stats: Stats::default() })
    }
    /// A previously verified continuation is only a proposal. The target must
    /// verify it again, including when sampling options or the prefix change.
    pub fn seed(&mut self, tokens: Vec<u32>) { self.seed = tokens; }
    pub fn tokens(&self) -> &[u32] { &self.history[self.prompt_len..] }
    pub fn is_done(&self) -> bool { self.done }
    /// Already-selected canonical target continuation; reading it never draws.
    pub fn canonical_next(&self) -> Option<u32> { (!self.done).then_some(self.next) }
    pub fn sampled_verifier(&self) -> bool { self.config.mode == Mode::Sampled }
    pub fn memory_bytes(&self) -> usize {
        self.history.capacity()*4 + self.lookup.memory_bytes() + self.recycling.memory_bytes()
            + self.tail.capacity()*4 + self.seed.capacity()*4 + self.guard.memory_bytes()
    }
    pub fn prepare_exact(&mut self, row_cap: usize, stop: impl Fn(u32) -> bool) -> Result<Proposal> {
        self.prepare_exact_filtered(row_cap, stop, |_| true)
    }
    /// Small-model guesses are untrusted. Acceptance uses only canonical target rows.
    pub fn prepare_with_target(&mut self, target: &mut (impl Target + ?Sized), row_cap: usize,
        stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<Proposal> {
        let mut plan = self.prepare_model(target, row_cap, None, &stop, &allowed)?;
        if plan.model_limit > 0 {
            let (proposal, distributions) = if self.sampled_verifier() {
                target.propose_sampled_model(&self.history, self.prompt_len, self.next,
                    plan.model_limit, self.options, &allowed)?
            } else { (target.propose_model(&self.history, self.next, plan.model_limit)?
                .ok_or("missing draft model")?, Vec::new()) };
            self.complete_model(&mut plan, proposal, stop, allowed)?;
            plan.distributions = distributions;
            plan.distributions.truncate(plan.input.len() - 1);
        }
        Ok(plan)
    }
    /// VADE_MTP_ADAPT's width: expected committed tokens per ms, with each depth's survival
    /// from this request's cumulative model-draft trials and hits (Beta prior: 4 trials at
    /// the target's recent survival, else 0.8), the target's row costs and the draft ms per
    /// proposed token. Cumulative counts keep one rejection from collapsing the depth.
    /// `optimistic` takes that depth's survival at its upper bound (two posterior standard
    /// deviations above the mean). With measured row costs (the target's model_verify_ms),
    /// rows cost those (else prior x the measured scale) plus the measured draft ms. Without
    /// them, rows cost the prior plus Control::draft_price: the top two widths differ by
    /// about 1 % or less, so a few percent of timing noise would pick a different width and
    /// shift every later cycle; the ids, not the timing, now decide the plan.
    fn native_rows(&self, available: usize, measured: &[Option<f64>], prior: &[Option<f64>], optimistic: Option<usize>) -> usize {
        let (trials, hits) = (&self.stats.depth_trials[0], &self.stats.depth_hits[0]);
        let token_ms = self.control.model_token_ms.unwrap_or(TOKEN_PRIOR_MS);
        let priced = measured.iter().any(Option::is_some);
        let (mut best, mut score, mut survival, mut expected) = (1, 0.0, 1.0, 1.0);
        for rows in 1..=available.min(16) {
            if rows > 1 {
                let d = rows - 2;
                let mean = (hits[d] as f64 + 4.0 * prior.get(d).copied().flatten().unwrap_or(0.8)) / (trials[d] as f64 + 4.0);
                survival *= if optimistic == Some(d) {
                    (mean + 2.0 * (mean * (1.0 - mean) / (trials[d] as f64 + 5.0)).sqrt()).min(1.0)
                } else { mean };
                expected += survival;
            }
            let rate = if priced {
                let cost = measured.get(rows).copied().flatten().unwrap_or_else(|| self.control.cost(rows, self.config.costs));
                expected / (cost + token_ms * (rows - 1) as f64)
            } else { expected / (self.config.costs.ms(rows) + self.control.draft_price() * (rows - 1) as f64) };
            if rate > score { best = rows; score = rate; }
        }
        best
    }
    /// Select the adaptive width without running a private model chain. The
    /// scheduler can share each draft step across all ready conversations.
    pub fn prepare_model(&mut self, target: &mut (impl Target + ?Sized), row_cap: usize, costs: Option<CostCurve>,
        stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<Proposal> {
        if self.sampled_verifier() && !target.speculative_model_draft() {
            return self.prepare_exact_filtered(row_cap.min(1), stop, allowed);
        }
        if target.native_draft_policy() {
            target.model_draft_budget(self.config.max_new - self.tokens().len())?;
            let row_cap = row_cap.min(target.max_verify_rows());
            if target.model_draft_width() == 0
                || (!plain_greedy(self.options) && !target.sampled_model_draft()) || row_cap <= 1 {
                return self.prepare_exact_filtered(row_cap, stop, allowed);
            }
            if self.done { return Err("draft has no available rows".into()); }
            let cap = row_cap.min(self.config.max_rows).min(self.config.max_new-self.tokens().len());
            if target.adaptive_model_depth() && cap > 1 {
                // A greedy chat alone drafts up to the resident depth when the target takes
                // the step rule (draft_steps.rs). The rule can stop the chain before each step.
                // With VADE_MTP_CONF=off, or when the rule does not apply, use the chooser's
                // depth from this request's survival rates and row costs (native_rows).
                if let Some(costs) = target.model_draft_costs() { self.config.costs = costs; }
                if self.control.model_token_ms.is_none() {
                    self.control.rates[0] = [0.8; 15];
                    self.control.model_token_ms = Some(TOKEN_PRIOR_MS);
                }
                let available = cap.min(target.model_draft_width() + 1);
                let measured: Vec<_> = (0..=available).map(|rows| target.model_verify_ms(rows)).collect();
                let prior: Vec<_> = (0..available).map(|depth| target.model_survival_prior(depth)).collect();
                let mut rows = self.native_rows(available, &measured, &prior, None);
                // With measured row costs a probe only refines survival, so probe one row deeper
                // only while that depth could still pay at its upper survival bound.
                if self.stats.cycles.is_multiple_of(16) && (measured.iter().all(Option::is_none)
                    || rows < available && self.native_rows(available, &measured, &prior, Some(rows - 1)) > rows) {
                    rows = (rows + 1).min(available);
                }
                if !self.sampled_verifier() && target.draft_step_rule(self.steps.rule(self.config.costs, self.control.draft_price(), self.stats.cycles)) { rows = available; }
                target.limit_model_depth(rows - 1);
            }
            let started = Instant::now();
            let mut plan = Proposal { input: vec![self.next], entry: self.history.len(), drafts: [Vec::new(),Vec::new(),Vec::new()],
                class: 0, draft_ms: 0.0, started, target_ms: None, model_limit: 0, model_draft_used: false, reserved_rows: None, distributions: Vec::new() };
            if !stop(self.next) && cap > 1 {
                if let Some(depth) = target.plan_model_draft(&self.history, cap-1)? {
                    plan.reserved_rows = Some(depth+1); return Ok(plan);
                }
                if let Some(limit) = target.defer_model_draft(cap-1).filter(|&n| n > 0) {
                    plan.model_limit = limit.min(cap-1); return Ok(plan);
                }
                let (proposal, distributions) = if self.sampled_verifier() {
                    target.propose_sampled_model(&self.history, self.prompt_len, self.next,
                        cap-1, self.options, &allowed)?
                } else { (target.propose_model(&self.history,self.next,cap-1)?.ok_or("missing draft model")?, Vec::new()) };
                plan.model_limit = cap-1;
                self.complete_model(&mut plan,proposal,&stop,&allowed)?;
                plan.distributions = distributions;
                plan.distributions.truncate(plan.input.len() - 1);
                plan.draft_ms = started.elapsed().as_secs_f64()*1000.0;
            }
            return Ok(plan);
        }
        // Exhausting a bounded model draft must not switch automatic decoding
        // to wider lookup verification; continue with canonical target rows.
        if self.config.automatic && target.model_draft_exhausted() {
            return self.prepare_exact_filtered(1, stop, allowed);
        }
        // Once catchup loses, keep lookup drafting for the rest of this request.
        if self.skip_model || target.model_draft_width() == 0 || !plain_greedy(self.options) && !target.sampled_model_draft()
            || row_cap <= 1 && !self.config.automatic {
            return self.prepare_exact_filtered(row_cap, stop, allowed);
        }
        if self.done { return Err("draft has no available rows".into()); }
        let available = self.config.max_rows.min(self.config.max_new - self.tokens().len())
            .min(target.model_draft_width() + 1);
        let mut cap = row_cap.min(available);
        // Cohort priors follow the ready population; retain measured rates.
        let costs = costs.or_else(|| target.model_draft_costs());
        if let Some(costs) = costs { self.config.costs = costs; }
        if self.control.model_token_ms.is_none() {
            if costs.is_some() {
                self.control.rates[0] = [0.8; 15];
            } else {
                self.control.rates[0] = [0.65; 15];
            }
            // GB10 0.6B: 8 ms short plus BF16 KV's bandwidth floor.
            // 28 layers * 8 KV heads * 128 * K/V * 2 bytes / 273 GB/s.
            // Explicit modes retain their original short-context prior.
            self.control.model_token_ms = Some(8.0 + if self.config.automatic && self.history.len() > 4096 {
                self.history.len() as f64 * (114688.0 / 273000000.0)
            } else { 0.0 });
        }
        if self.config.automatic {
            let (missing, ms_per_token) = target.model_draft_catchup();
            let plain_ms = self.control.cost(1, self.config.costs);
            // Best predicted saving over every width the policy can choose,
            // using its acceptance estimate and measured GB10 cost priors.
            let best_saving = (2..=available).map(|rows| plain_ms -
                1.0 / self.control.score(0, rows, self.config.costs))
                .fold(0.0, f64::max) * (self.config.max_new-self.tokens().len()) as f64;
            if missing > 64 && missing as f64 * ms_per_token > best_saving {
                self.skip_model = true;
                self.control = Control::new();
                return self.prepare_exact_filtered(row_cap, stop, allowed);
            }
        }
        if row_cap <= 1 { return self.prepare_exact_filtered(row_cap, stop, allowed); }
        // A single extra proposal occasionally probes the next depth. Do not
        // force a full-width chain when a request repeatedly rejects proposals.
        if self.config.adaptive {
            let available = cap;
            cap = self.control.rows(0, cap, self.config.costs);
            if self.stats.cycles.is_multiple_of(16) { cap = (cap+1).min(available); }
        }
        Ok(Proposal { input: vec![self.next], entry: self.history.len(), drafts: [Vec::new(), Vec::new(), Vec::new()],
            class: 0, draft_ms: 0.0, started: Instant::now(), target_ms: None,
            reserved_rows: None, distributions: Vec::new(), model_draft_used: false, model_limit: if stop(self.next) { 0 } else { cap-1 } })
    }
    /// Spare shared rows can follow depths another request already pays for.
    /// With no profitable cohort draft, the ordinary cost policy still wins.
    pub fn share_model_depth(&self, plan: &mut Proposal, width: usize, row_cap: usize,
        depth: usize, sampled_model: bool, stop: impl Fn(u32) -> bool) -> usize {
        if self.sampled_verifier() || self.skip_model || width == 0 || !plain_greedy(self.options) && !sampled_model || self.done || stop(self.next)
            || plan.entry != self.history.len() || plan.input.len() != 1 { return 0; }
        let limit = depth.min(width).min(row_cap.saturating_sub(1))
            .min(self.config.max_rows-1).min((self.config.max_new-self.tokens().len()).saturating_sub(1));
        let extra = limit.saturating_sub(plan.model_limit);
        plan.model_limit += extra;
        extra
    }
    pub fn complete_model(&mut self, plan: &mut Proposal, proposal: ModelProposal,
        stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<()> {
        if proposal.tokens.len() > plan.model_limit || plan.entry != self.history.len() {
            return Err("invalid shared draft proposal".into());
        }
        self.stats.model_proposal_ms += proposal.proposal_ms;
        self.stats.model_catchup_ms += proposal.catchup_ms;
        self.stats.model_proposals += proposal.tokens.len();
        plan.model_draft_used |= !proposal.tokens.is_empty();
        // Catchup is paid once and gated separately; it must not inflate the
        // steady-state proposal cost used to choose every subsequent width.
        if let Some(ms) = self.control.model_token_ms.as_mut().filter(|_| !proposal.tokens.is_empty()) {
            *ms = 0.75 * *ms + 0.25 * proposal.proposal_ms / proposal.tokens.len() as f64;
            self.control.observe_price(None, Some(proposal.proposal_ms / proposal.tokens.len() as f64));
        }
        let mut guesses = proposal.tokens;
        if let Some(n) = guesses.iter().position(|&t| !allowed(t)) { guesses.truncate(n); }
        if let Some(n) = guesses.iter().position(|&t| stop(t)) { guesses.truncate(n+1); }
        plan.input.extend_from_slice(&guesses);
        plan.drafts[0] = guesses;
        plan.draft_ms = proposal.proposal_ms + proposal.catchup_ms;
        plan.model_limit = 0;
        plan.started = Instant::now();
        Ok(())
    }
    pub fn prepare_exact_filtered(&mut self, row_cap: usize, stop: impl Fn(u32) -> bool,
        allowed: impl Fn(u32) -> bool) -> Result<Proposal> {
        if row_cap == 0 || self.done { return Err("draft has no available rows".into()); }
        let row_cap = if self.sampled_verifier() { row_cap.min(1) } else { row_cap };
        let entry = self.history.len();
        let cap = row_cap.min(self.config.max_rows).min(self.config.max_new - self.tokens().len());
        let draft_start = Instant::now();
        self.history.push(self.next);
        let limit = (self.config.max_rows - 1).min(self.config.max_new - self.tokens().len());
        let mut drafts = [Vec::new(), Vec::new(), std::mem::take(&mut self.tail)];
        self.recycling.extend(&self.history, limit, &mut drafts[0], &allowed);
        if limit > 0 { self.lookup.propose(&self.history, limit, &mut drafts[1]); }
        let generated = &self.history[self.prompt_len..];
        if self.seed.starts_with(generated) {
            drafts[1].clear();
            drafts[1].extend_from_slice(&self.seed[generated.len()..self.seed.len().min(generated.len()+limit)]);
        } else { self.seed.clear(); }
        for draft in &mut drafts {
            draft.truncate(limit);
            if let Some(n) = draft.iter().position(|&t| !allowed(t)) { draft.truncate(n); }
            if !draft.is_empty() { self.recycling.extend(&self.history, limit, draft, &allowed); }
            if let Some(n) = draft.iter().position(|&t| stop(t)) { draft.truncate(n+1); }
        }
        self.control.draft_ms = draft_start.elapsed().as_secs_f64()*1000.0;
        // Suffix matches start with the stronger prior and win ties, but a
        // stale continuation must yield once another source predicts better.
        let mut class = 0;
        let mut score = 0.0;
        for source in [1, 0, 2] {
            if drafts[source].is_empty() { continue; }
            let available = cap.min(drafts[source].len()+1);
            let rows = if self.config.adaptive { self.control.rows(source, available, self.config.costs) } else { available };
            let rate = self.control.score(source, rows, self.config.costs);
            if rate > score { class = source; score = rate; }
        }
        let guesses = &drafts[class];
        let available = cap.min(guesses.len() + 1);
        let rows = if self.config.adaptive { self.control.rows(class, available, self.config.costs) } else { available };
        let mut input = Vec::with_capacity(rows);
        input.push(self.next);
        if !stop(self.next) {
            for &token in guesses.iter().take(rows-1) {
                input.push(token);
                if stop(token) { break; }
            }
        }
        self.history.pop();
        Ok(Proposal { input, entry, drafts, class, draft_ms: self.control.draft_ms,
            started: Instant::now(), target_ms: None, reserved_rows: None, distributions: Vec::new(), model_limit: 0, model_draft_used: false })
    }
    /// Accept exact shared logits and recycle every row, including rejected rows.
    pub fn finish_exact(&mut self, target: &mut (impl Target + ?Sized), plan: Proposal,
        output: &[Vec<f32>], stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<&[u32]> {
        if output.len() != plan.input.len() { return Err("invalid prepared draft row count".into()); }
        let target_ms = plan.elapsed_ms();
        let learn = Instant::now();
        let mut predictions = vec![u32::MAX; output.len()];
        for i in (0..output.len()).rev() {
            predictions[i] = self.recycling.observe(plan.input[i], &output[i]);
        }
        let learn_ms = learn.elapsed().as_secs_f64()*1000.0;
        if self.sampled_verifier() {
            return self.finish_speculative(target, plan, &predictions, target_ms, learn_ms, stop,
                |s, i| s.sampler.distribution(&output[i], &s.history, s.prompt_len, &allowed));
        }
        let entry = self.history.len();
        let rows = self.presampled.take().filter(|(at, rows)| *at == entry && rows.len() == output.len()).map(|(_, rows)| rows)
            .or_else(|| (!plain_greedy(self.options) && output.len() > 1 && parallel_rows())
            .then(|| self.sample_rows(&plan.input, output, &allowed)));
        self.finish_choices(target, plan, &predictions, target_ms, learn_ms, stop, |s, i| {
            if plain_greedy(s.options) && predictions[i] != u32::MAX && allowed(predictions[i]) {
                Ok(predictions[i])
            } else if let Some(rows) = &rows { rows[i].clone() }
            else { s.sampler.sample(&output[i], &s.history, s.prompt_len, s.history.len(), &allowed) }
        })
    }
    /// Every verified row's draw at once, one thread per row. finish_choices reaches row i only
    /// when rows < i matched the drafts, with history = history + input[..=i] at position
    /// history.len() + i + 1; sample() is a pure function of those inputs (its scratch is
    /// refilled per call), so each draw equals the sequential one. Later rows are discarded.
    fn sample_rows(&mut self, input: &[u32], output: &[Vec<f32>], allowed: &impl Fn(u32) -> bool) -> Vec<Result<u32>> {
        let mask: Vec<bool> = (0..output[0].len() as u32).map(allowed).collect();
        self.sample_rows_masked(input, output, &mask)
    }
    /// VADE_MTP_SAMPLED=2: take this cycle's verified-row draws now (a cohort draws every chat's
    /// rows at once); finish_exact uses them only for the same history length and row count.
    /// `mask` must be the allowed set finish_exact receives.
    pub fn presample(&mut self, input: &[u32], output: &[Vec<f32>], mask: &[bool]) {
        if !plain_greedy(self.options) && output.len() > 1 && parallel_rows() {
            let rows = self.sample_rows_masked(input, output, mask);
            self.presampled = Some((self.history.len(), rows));
        }
    }
    fn sample_rows_masked(&mut self, input: &[u32], output: &[Vec<f32>], mask: &[bool]) -> Vec<Result<u32>> {
        while self.row_samplers.len() < output.len() {
            match Sampler::new(self.options) {
                Ok(sampler) => self.row_samplers.push(sampler),
                Err(error) => return vec![Err(error); output.len()],
            }
        }
        let (history, prompt_len) = (&self.history, self.prompt_len);
        std::thread::scope(|scope| {
            let rows: Vec<_> = self.row_samplers.iter_mut().zip(output).enumerate().map(|(i, (sampler, logits))| {
                scope.spawn(move || {
                    let mut seen = Vec::with_capacity(history.len() + i + 1);
                    seen.extend_from_slice(history);
                    seen.extend_from_slice(&input[..=i]);
                    sampler.sample(logits, &seen, prompt_len, seen.len(), |id| mask.get(id as usize) == Some(&true))
                })
            }).collect();
            rows.into_iter().map(|row| row.join().unwrap_or_else(|_| Err("row sampler panicked".into()))).collect()
        })
    }
    /// ID-only shared decode learns the observed best successor without logits.
    pub fn finish_greedy(&mut self, target: &mut (impl Target + ?Sized), plan: Proposal,
        output: &[u32], stop: impl Fn(u32) -> bool) -> Result<&[u32]> {
        if !plain_greedy(self.options) { return Err("greedy result for sampled draft".into()); }
        if output.len() != plan.input.len() { return Err("invalid prepared draft row count".into()); }
        let target_ms = plan.elapsed_ms();
        let learn = Instant::now();
        for i in (0..output.len()).rev() {
            if let Some(next) = self.recycling.successors.get_mut(plan.input[i] as usize) {
                *next = output[i];
            }
        }
        let learn_ms = learn.elapsed().as_secs_f64()*1000.0;
        self.finish_choices(target, plan, output, target_ms, learn_ms, stop, |_, i| Ok(output[i]))
    }
    /// Target samples are keyed at each proposed absolute position. Only the
    /// matching prefix commits; later choices and their hypothetical histories
    /// cannot affect the next draw. Greedy top-k heads still train recycling.
    pub fn finish_sampled(&mut self, target: &mut (impl Target + ?Sized), plan: Proposal,
        output: &[u32], best: &[u32], stop: impl Fn(u32) -> bool) -> Result<&[u32]> {
        if output.len() != plan.input.len() || best.len() != output.len() {
            return Err("invalid sampled draft row count".into());
        }
        let target_ms = plan.elapsed_ms();
        for i in (0..best.len()).rev() {
            if let Some(next) = self.recycling.successors.get_mut(plan.input[i] as usize) { *next = best[i]; }
        }
        self.finish_choices(target, plan, best, target_ms, 0.0, stop, |_, i| Ok(output[i]))
    }
    pub fn finish_speculative_top_k(&mut self, target: &mut (impl Target + ?Sized), plan: Proposal,
        output: &[Vec<(u32, f32)>], stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<&[u32]> {
        if output.len() != plan.input.len() { return Err("invalid sampled verifier rows".into()); }
        let target_ms = plan.elapsed_ms();
        let best = output.iter().map(|r| r.first().map_or(u32::MAX, |v| v.0)).collect::<Vec<_>>();
        self.finish_speculative(target, plan, &best, target_ms, 0.0, stop,
            |s, i| s.sampler.distribution_top_k(&output[i], &allowed).ok_or_else(|| "compact sampling needs full logits".into()))
    }
    /// Mode::Sampled proposals come from the draft head's filtered rows. Each row still commits
    /// only the canonical draw at its position: the ids equal drafting off, whatever was proposed.
    fn finish_speculative(&mut self, target: &mut (impl Target + ?Sized), plan: Proposal,
        best: &[u32], target_ms: f64, learn_ms: f64, stop: impl Fn(u32) -> bool,
        mut distribution: impl FnMut(&mut Self, usize) -> Result<Distribution>) -> Result<&[u32]> {
        if plan.distributions.len() + 1 != plan.input.len() { return Err("missing filtered draft distributions".into()); }
        self.finish_choices(target, plan, best, target_ms, learn_ms, stop, |s, i| {
            let position = s.history.len();
            distribution(s, i)?.draw(uniform(s.options.seed, position))
        })
    }
    fn finish_choices(&mut self, target: &mut (impl Target + ?Sized), plan: Proposal,
        predictions: &[u32], target_ms: f64, learn_ms: f64, stop: impl Fn(u32) -> bool,
        mut choose: impl FnMut(&mut Self, usize) -> Result<u32>) -> Result<&[u32]> {
        let entry = plan.entry;
        if entry != self.history.len() || target.checkpoint() != entry+plan.input.len() {
            return Err("invalid prepared draft result".into());
        }
        self.recycling.observe_contexts(&self.history, &plan.input, predictions);
        let result = (|| -> Result<()> {
            let mut choices = Vec::new();
            let mut next = self.next;
            for i in 0..plan.input.len() {
                self.history.push(plan.input[i]);
                next = choose(self, i)?;
                choices.push(next);
                if i+1 == plan.input.len() || plan.input[i+1] != next || stop(plan.input[i]) { break; }
            }
            target.truncate(self.history.len())?;
            let committed = self.history.len()-entry;
            if self.control.model_token_ms.is_none() {
                self.tail.extend(predictions[committed..].iter().copied().take_while(|&t| t != u32::MAX));
            }
            self.observe_drafts(&plan.drafts, plan.class, &choices);
            self.steps.observe(&target.take_draft_confidence(), &plan.drafts[0], &choices, plan.input.len(), self.config.costs, self.control.draft_price());
            self.control.observe_cost(plan.input.len(), target_ms, learn_ms, self.config.costs);
            self.next = next;
            self.stats.cycles += 1;
            self.stats.widths[plan.input.len()-1] += 1;
            self.stats.plan_hash = (self.stats.plan_hash ^ plan.input.len() as u64).wrapping_mul(0x100000001b3);
            self.stats.draft_price = self.control.draft_price();
            self.stats.rows += plan.input.len();
            self.stats.proposed += plan.input.len()-1;
            self.stats.accepted += committed-1;
            self.stats.rollbacks += usize::from(committed < plan.input.len());
            self.stats.max_rows = self.stats.max_rows.max(plan.input.len());
            self.stats.draft_ms += plan.draft_ms;
            self.stats.recycle_ms += learn_ms;
            self.stats.target_ms += target_ms;
            self.stats.costs_ms = [1,2,4,8,16].map(|n| self.control.cost(n, self.config.costs));
            if !plan.drafts[plan.class].is_empty() {
                self.stats.source_cycles[plan.class] += 1;
                self.stats.source_accepted[plan.class] += committed-1;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.history.truncate(entry);
            self.lookup.clear();
            self.tail.clear();
            return Err(match target.truncate(entry) {
                Ok(()) => error,
                Err(rollback) => format!("{error}; rollback to {entry} also failed: {rollback}"),
            });
        }
        self.done = self.tokens().len() == self.config.max_new || self.history.last().is_some_and(|&t| stop(t));
        Ok(&self.history[entry..])
    }
    pub fn step(&mut self, target: &mut (impl Target + ?Sized), row_cap: usize, stop: impl Fn(u32) -> bool) -> Result<&[u32]> {
        self.step_filtered(target, row_cap, stop, |_| true)
    }
    pub fn step_filtered(&mut self, target: &mut (impl Target + ?Sized), row_cap: usize, stop: impl Fn(u32) -> bool, allowed: impl Fn(u32) -> bool) -> Result<&[u32]> {
        let entry = self.history.len();
        if target.checkpoint() != entry { return Err("target/history checkpoint mismatch".into()); }
        if row_cap == 0 || self.done { return Ok(&self.history[entry..]); }
        let result = self.cycle(target, row_cap, &stop, &allowed);
        if let Err(error) = result {
            self.history.truncate(entry);
            self.lookup.clear();
            self.tail.clear();
            return Err(match target.truncate(entry) {
                Ok(()) => error,
                Err(rollback) => format!("{error}; rollback to {entry} also failed: {rollback}"),
            });
        }
        self.done = self.tokens().len() == self.config.max_new || self.history.last().is_some_and(|&t| stop(t));
        debug_assert_eq!(target.checkpoint(), self.history.len());
        Ok(&self.history[entry..])
    }
    fn observe_drafts(&mut self, drafts: &[Vec<u32>; 3], class: usize, choices: &[u32]) {
        let position = self.history.len().saturating_sub(choices.len());
        let band = if position < 4096 { 0 } else if position < 16384 { 1 } else { 2 };
        for (source, draft) in drafts.iter().enumerate() {
            let hits: Vec<_> = draft.iter().zip(choices).map(|(a,b)| a == b).collect();
            self.control.observe(source, &hits);
            if source == class {
                for (depth, &hit) in hits.iter().enumerate() {
                    self.stats.depth_trials[source][depth] += 1;
                    self.stats.depth_hits[source][depth] += usize::from(hit);
                    if source == 0 && self.control.model_token_ms.is_some() {
                        self.stats.model_band_trials[band][depth] += 1;
                        self.stats.model_band_hits[band][depth] += usize::from(hit);
                    }
                    if !hit { break; }
                }
            }
        }
    }
    fn cycle(&mut self, target: &mut (impl Target + ?Sized), row_cap: usize, stop: &impl Fn(u32) -> bool, allowed: &impl Fn(u32) -> bool) -> Result<()> {
        let plan = if target.native_draft_policy() { self.prepare_with_target(target, row_cap, stop, allowed)? }
            else { self.prepare_exact_filtered(row_cap, stop, allowed)? };
        if self.sampled_verifier() {
            let start = Instant::now();
            let output = if plan.input.len() == 1 { vec![target.step(plan.input[0])?] }
                else { target.verify_batch(&plan.input)? };
            let mut plan = plan;
            plan.record_target_ms(start.elapsed().as_secs_f64() * 1000.0);
            return self.finish_exact(target, plan, &output, stop, allowed).map(|_| ());
        }
        let Proposal { entry, input, drafts, class, draft_ms, .. } = plan;
        let guesses = &drafts[class];
        self.history.push(self.next);
        let mut choices = Vec::new();
        let mut guarded = 0;
        let mut fires = 0;
        let mut repair_ms = 0.0;
        let start = Instant::now();
        // Even serial fallback exposes logits: this is how recycling warms up
        // on original prose without spending an additional target pass.
        let output = if input.len() == 1 { vec![target.step(input[0])?] }
            else if self.config.mode == Mode::Strict { target.verify_batch(&input)? }
            else { target.forward_batch(&input)? };
        let target_ms = start.elapsed().as_secs_f64()*1000.0;
        if output.len() != input.len() || target.checkpoint() != entry + input.len() {
            return Err("invalid target batch contract".into());
        }
        let learn_start = Instant::now();
        let mut predictions = vec![u32::MAX; input.len()];
        // Earlier, more reliable prefixes win when a token occurs twice.
        for i in (0..input.len()).rev() {
            predictions[i] = self.recycling.observe(input[i], &output[i]);
        }
        self.recycling.observe_contexts(&self.history[..entry], &input, &predictions);
        let mut learn_ms = learn_start.elapsed().as_secs_f64()*1000.0;
        let mut next = self.next;
        for i in 0..input.len() {
            let row = &output[i];
            let mut chosen = if plain_greedy(self.options) && predictions[i] != u32::MAX && allowed(predictions[i]) {
                predictions[i]
            } else { self.sampler.sample(row, &self.history, self.prompt_len, self.history.len(), allowed)? };
            if self.config.mode == Mode::Fast && input.len() > 1 {
                guarded += 1;
                let error = if target.kv_bf16() { self.config.guard_error.max(0.2) } else { self.config.guard_error };
                if !self.guard.safe_filtered(row, &self.history, self.prompt_len, self.options, error, chosen, allowed) {
                    fires += 1;
                    target.truncate(entry+i)?;
                    let repair_start = Instant::now();
                    let exact = target.step(input[i])?;
                    repair_ms += repair_start.elapsed().as_secs_f64()*1000.0;
                    let learn_start = Instant::now();
                    self.recycling.observe(input[i], &exact);
                    learn_ms += learn_start.elapsed().as_secs_f64()*1000.0;
                    chosen = self.sampler.sample(&exact, &self.history, self.prompt_len, self.history.len(), allowed)?;
                    next = chosen;
                    choices.push(chosen);
                    break;
                }
            }
            next = chosen;
            choices.push(chosen);
            if i + 1 == input.len() || input[i+1] != chosen || stop(input[i]) { break; }
            self.history.push(chosen);
        }
        let committed = self.history.len() - entry;
        if target.checkpoint() < self.history.len() { return Err("target failed to commit input".into()); }
        target.truncate(self.history.len())?;
        // Shift wrong-prefix predictions to their next absolute positions.
        // They are guesses only; the next cycle verifies them from its new root.
        self.tail.extend(predictions[committed..].iter().copied().take_while(|&t| t != u32::MAX));
        self.observe_drafts(&drafts, class, &choices);
        self.control.observe_cost(input.len(), target_ms, learn_ms * input.len() as f64 / (input.len()+fires) as f64, self.config.costs);
        if self.config.mode == Mode::Fast {
            // Exact serial decisions are also free evidence about whether a
            // future FAST row could be guarded. Decay prior fallback penalties
            // when these shadow drafts begin hitting again.
            self.control.observe_guard(class, guarded.max(usize::from(guesses.first()==choices.first() && !choices.is_empty())), fires);
        }
        self.next = next;
        self.stats.cycles += 1;
        self.stats.rows += input.len() + fires;
        self.stats.proposed += input.len()-1;
        self.stats.accepted += committed-1;
        self.stats.guarded += guarded;
        self.stats.guard_fires += fires;
        self.stats.rollbacks += usize::from(committed < input.len());
        self.stats.max_rows = self.stats.max_rows.max(input.len());
        self.stats.draft_ms += draft_ms;
        self.stats.recycle_ms += learn_ms;
        self.stats.target_ms += target_ms + repair_ms;
        self.stats.costs_ms = [1,2,4,8,16].map(|n| self.control.cost(n, self.config.costs));
        if !guesses.is_empty() {
            self.stats.source_cycles[class] += 1;
            self.stats.source_accepted[class] += committed-1;
        }
        Ok(())
    }
}

pub fn plain_greedy(p: Sampling) -> bool {
    p.temperature == 0.0 && p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0
}

/// Separate analysis leaves the final shared sampler unchanged. Conservatively
/// rejects unstable ranking, top-k/nucleus support, and CDF decision boundaries.
#[derive(Default)]
struct Guard { ranked: Vec<(u32, f64)>, counts: Vec<(bool, u32)> }
impl Guard {
    fn memory_bytes(&self) -> usize { self.ranked.capacity()*16 + self.counts.capacity()*8 }
    #[cfg(test)]
    fn safe(&mut self, logits: &[f32], history: &[u32], prompt: usize, p: Sampling, error: f64, chosen: u32) -> bool {
        self.safe_filtered(logits, history, prompt, p, error, chosen, &|_| true)
    }
    fn safe_filtered(&mut self, logits: &[f32], history: &[u32], prompt: usize, p: Sampling, error: f64, chosen: u32, allowed: &impl Fn(u32) -> bool) -> bool {
        let penalize = p.repetition_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0;
        if penalize {
            self.counts.resize(logits.len(), (false,0)); self.counts.fill((false,0));
            for (i, &t) in history.iter().enumerate() {
                let Some(c) = self.counts.get_mut(t as usize) else { return false; };
                c.0 = true; c.1 += u32::from(i >= prompt);
            }
        }
        let delta = error * p.repetition_penalty.max(1.0 / p.repetition_penalty).max(1.0);
        self.ranked.clear();
        if p.temperature > 0.0 { self.ranked.reserve_exact(logits.len()); }
        let (mut best, mut second) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (id, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() { return false; }
            if !allowed(id as u32) { continue; }
            let mut v = logit as f64;
            if penalize {
                let (seen,count) = self.counts[id];
                if seen { v = if v < 0.0 { v*p.repetition_penalty } else { v/p.repetition_penalty }; }
                if count > 0 { v -= p.presence_penalty + p.frequency_penalty*count as f64; }
            }
            if v > best { second = best; best = v; } else { second = second.max(v); }
            if p.temperature > 0.0 { self.ranked.push((id as u32,v)); }
        }
        if p.temperature == 0.0 { return best - second > 2.0*delta; }
        // The shared sampler's top_k=0 partition order is not stable under
        // perturbation. It requires exact recomputation, as do unbounded errors and min_p.
        if p.top_k == 0 || p.min_p > 0.0 { return false; }
        let count = p.top_k.min(self.ranked.len());
        let cmp = |a: &(u32,f64), b: &(u32,f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if count < self.ranked.len() {
            self.ranked.select_nth_unstable_by(count, cmp);
            self.ranked.truncate(count+1);
        }
        self.ranked.sort_unstable_by(cmp);
        let Some(rank) = self.ranked.iter().position(|r|r.0==chosen) else { return false; };
        if (rank > 0 && self.ranked[rank-1].1-self.ranked[rank].1 <= 2.0*delta)
            || (rank+1 < self.ranked.len() && self.ranked[rank].1-self.ranked[rank+1].1 <= 2.0*delta) { return false; }
        // Only the selected ID needs a stable rank. Sums of the largest k
        // exp(logits) have the same multiplicative error bound even when other
        // IDs exchange ranks or cross the top-k boundary.
        self.ranked.truncate(count);
        for (_,v) in &mut self.ranked { *v = ((*v-best)/p.temperature).exp(); }
        let total: f64 = self.ranked.iter().map(|r|r.1).sum();
        let bound = (2.0*delta/p.temperature).exp_m1();
        if !bound.is_finite() || total <= 0.0 { return false; }
        let mut mass = 0.0;
        let mut keep = count;
        if p.top_p < 1.0 {
            for (i,r) in self.ranked.iter().enumerate() {
                let before = mass/total; mass += r.1;
                if mass/total >= p.top_p {
                    if p.top_p-before <= bound || mass/total-p.top_p <= bound { return false; }
                    keep = i+1; break;
                }
            }
        }
        self.ranked.truncate(keep);
        let total: f64 = self.ranked.iter().map(|r|r.1).sum();
        let mut low = 0.0;
        for &(token, mass) in &self.ranked {
            let high = low + mass/total;
            if token == chosen {
                let u = crate::sample::uniform(p.seed, history.len());
                return (low == 0.0 || u-low > bound) && (high >= 1.0-1e-14 || high-u > bound);
            }
            low = high;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Toy { history: Vec<u32>, fail_at: usize, fail_rollback: bool, fast: f32, pending: bool }
    impl Toy {
        fn logits(&self) -> Vec<f32> {
            let next = (self.history.last().copied().unwrap_or(0)+1)%7;
            (0..7).map(|id| if id==next { 2.0 } else { id as f32/11.0 }).collect()
        }
    }
    impl Target for Toy {
        fn checkpoint(&self) -> usize { self.history.len() }
        fn truncate(&mut self, p: usize) -> Result<()> {
            if self.fail_rollback { return Err("injected rollback failure".into()); }
            self.history.truncate(p); self.pending=false; Ok(())
        }
        fn step(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.history.len() == self.fail_at { return Err("injected forward failure".into()); }
            self.history.push(token); Ok(self.logits())
        }
        fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> { let rows=tokens.iter().map(|&t|self.step(t)).collect(); self.pending=true; rows }
        fn forward_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
            let mut rows = self.verify_batch(tokens)?;
            for row in &mut rows { row[0] += self.fast; }
            Ok(rows)
        }
    }
    fn toy(prompt: &[u32]) -> Toy { Toy { history: prompt.to_vec(), fail_at: usize::MAX, fail_rollback: false, fast: 0.01, pending: false } }
    #[test]
    fn cycles_equal_serial_with_sampling_penalties_eos_caps_and_resume() {
        let prompt: Vec<u32> = (0..35).map(|i|i%7).collect();
        for sampled in [false,true] {
            let options = Sampling { seed:42, temperature:if sampled {0.8} else {0.0}, top_k:4, top_p:0.85, min_p:0.0,
                repetition_penalty:1.1, presence_penalty:0.2, frequency_penalty:0.15 };
            let mut reference = toy(&prompt);
            let mut sample = Sampler::new(options).unwrap();
            let mut next = sample.sample(&reference.logits(),&prompt,prompt.len(),prompt.len(), |_|true).unwrap();
            let mut expected = Vec::new();
            for _ in 0..48 {
                expected.push(next);
                let logits = reference.step(next).unwrap();
                next = sample.sample(&logits,&reference.history,prompt.len(),reference.history.len(), |_|true).unwrap();
            }
            for (mode, sticky_plain) in [(Mode::Strict, false), (Mode::Fast, false), (Mode::Strict, true)] {
                for adaptive in [false,true] {
                    for max_new in [0,1,2,9,48] {
                        for eos in [None,Some(expected[5])] {
                            let mut target = toy(&prompt);
                            let mut config = Config::strict(max_new);
                            config.mode = mode; config.adaptive = adaptive;
                            config.guard_error = 100.0; // force exact fallback on every FAST row
                            let mut spec = Speculator::new(&prompt,&target.logits(),config,options).unwrap();
                            assert!(spec.step(&mut target,0, |_|false).unwrap().is_empty());
                            let mut calls = 0;
                            while !spec.is_done() {
                                let cap = if sticky_plain && calls >= 2 { 1 } else { [1,4,2,16][calls%4] };
                                let committed = spec.step(&mut target,cap, |t|eos==Some(t)).unwrap();
                                assert!(!committed.is_empty() && committed.len()<=cap);
                                assert_eq!(spec.history(),target.history); assert!(!target.pending, "verification must commit even on full acceptance");
                                calls += 1;
                            }
                            let len = expected[..max_new].iter().position(|t|Some(*t)==eos).map_or(max_new, |i|i+1);
                            assert_eq!(spec.tokens(),&expected[..len]);
                            assert_eq!(target.history.len(),prompt.len()+len);
                            assert_eq!(spec.stats.cycles+spec.stats.accepted,len);
                        }
                    }
                }
            }
        }
    }
    /// Native-policy chat for the shared-draft hook: proposals are a function of history.
    struct Native { history: Vec<u32>, defer: Option<usize>, proposed: usize }
    impl Target for Native {
        fn checkpoint(&self) -> usize { self.history.len() }
        fn truncate(&mut self, p: usize) -> Result<()> { self.history.truncate(p); Ok(()) }
        fn step(&mut self, t: u32) -> Result<Vec<f32>> { self.history.push(t); Ok(vec![0.0; 7]) }
        fn native_draft_policy(&self) -> bool { true }
        fn model_draft_width(&self) -> usize { 3 }
        fn defer_model_draft(&self, limit: usize) -> Option<usize> { self.defer.map(|n| n.min(limit)) }
        fn propose_model(&mut self, history: &[u32], next: u32, limit: usize) -> Result<Option<ModelProposal>> {
            self.proposed += 1;
            Ok(Some(ModelProposal { tokens: (1..=limit.min(3) as u32).map(|i| (next + i + history.len() as u32) % 7).collect(),
                proposal_ms: 0.0, catchup_ms: 0.0 }))
        }
    }
    #[test]
    fn deferred_model_draft_completes_to_the_inline_proposal() {
        let prompt: Vec<u32> = (0..9).map(|i| i % 7).collect();
        for cap in [1, 2, 3, 4, 16] {
            for defer in [1, 3, 15] {
                let speculator = || Speculator::from_selected(&prompt, prompt.len(), 5, 7, Config::strict(32), Sampling::greedy()).unwrap();
                let (mut inline, mut deferred) = (speculator(), speculator());
                let mut a = Native { history: prompt.clone(), defer: None, proposed: 0 };
                let mut b = Native { history: prompt.clone(), defer: Some(defer), proposed: 0 };
                let plain = inline.prepare_model(&mut a, cap, None, |_| false, |_| true).unwrap();
                let mut plan = deferred.prepare_model(&mut b, cap, None, |_| false, |_| true).unwrap();
                assert_eq!(b.proposed, 0, "cap {cap}: deferral must not draft inline");
                let limit = plan.model_limit();
                assert_eq!(limit, if cap > 1 { defer.min(cap - 1) } else { 0 });
                assert_eq!(plan.input, [5]);
                if limit > 0 {
                    // The scheduler's shared pass asks the target with the reserved limit.
                    let proposal = b.propose_model(&prompt, 5, limit).unwrap().unwrap();
                    deferred.complete_model(&mut plan, proposal, |_| false, |_| true).unwrap();
                }
                if defer >= 3 || cap <= 1 {
                    assert_eq!(plan.input, plain.input, "cap {cap} defer {defer}");
                } else {
                    assert_eq!(plan.input[..], plain.input[..plan.input.len()], "cap {cap} defer {defer}");
                }
                assert_eq!(plan.model_limit(), 0);
                assert!(plan.row_budget() <= cap.max(1));
            }
        }
    }
    #[test]
    fn measured_row_costs_choose_the_width_and_bound_probes() {
        // FP8 two-Spark rank path, 10-25 gate: verify ms by rows (2, 3, 4) and a prose request's
        // survival (0.63, 0.59, 0.53); 4.02 ms per proposed token.
        let prompt: Vec<u32> = (0..9).map(|i| i % 7).collect();
        let mut spec = Speculator::from_selected(&prompt, prompt.len(), 5, 7, Config::strict(1024), Sampling::greedy()).unwrap();
        spec.control.model_token_ms = Some(4.02);
        spec.stats.depth_trials[0][..3].copy_from_slice(&[479, 163, 15]);
        spec.stats.depth_hits[0][..3].copy_from_slice(&[301, 95, 8]);
        // Before any measured pass, the linear byte-ledger prior and the prior draft price pick 2.
        spec.config.costs = CostCurve([22.598, 26.919, 35.562, 52.849, 87.421]);
        spec.control.cost_scale = 1.7;
        assert_eq!(spec.native_rows(4, &[], &[], None), 2);
        let measured = [None, None, Some(46.50), Some(50.75), Some(58.29)];
        assert_eq!(spec.native_rows(4, &measured, &[], None), 3);
        // Even at its upper survival bound the fourth row cannot pay: no probe.
        assert_eq!(spec.native_rows(4, &measured, &[], Some(2)), 3);
        // An untried depth that loses at its prior mean but can pay at its bound is probed.
        spec.stats.depth_trials[0][..3].copy_from_slice(&[479, 359, 0]);
        spec.stats.depth_hits[0][..3].copy_from_slice(&[359, 251, 0]);
        assert_eq!(spec.native_rows(4, &measured, &[], None), 3);
        assert_eq!(spec.native_rows(4, &measured, &[], Some(2)), 4);
        // A recent prose-like survival prior starts a new request at 3 rows, not the 0.8 prior's 4.
        spec.stats.depth_trials[0] = [0; 15]; spec.stats.depth_hits[0] = [0; 15];
        assert_eq!(spec.native_rows(4, &measured, &[], None), 4);
        assert_eq!(spec.native_rows(4, &measured, &[Some(0.63), Some(0.59), Some(0.53)], None), 3);
        // Code-like survival keeps the full width.
        spec.stats.depth_trials[0][..3].copy_from_slice(&[435, 413, 382]);
        spec.stats.depth_hits[0][..3].copy_from_slice(&[413, 386, 357]);
        assert_eq!(spec.native_rows(4, &measured, &[], None), 4);
    }
    /// One chat on a native adaptive target without measured row costs (FP4-Local's shape).
    /// The drafter guesses the reference, wrong at fixed positions, so survival depends on ids.
    struct Adaptive { history: Vec<u32>, reference: Vec<u32>, depth: usize, draft_ms: f64 }
    impl Target for Adaptive {
        fn checkpoint(&self) -> usize { self.history.len() }
        fn truncate(&mut self, p: usize) -> Result<()> { self.history.truncate(p); Ok(()) }
        fn step(&mut self, t: u32) -> Result<Vec<f32>> { self.history.push(t); Ok(vec![0.0; 64]) }
        fn native_draft_policy(&self) -> bool { true } fn adaptive_model_depth(&self) -> bool { true }
        fn model_draft_width(&self) -> usize { 7 } fn limit_model_depth(&mut self, depth: usize) { self.depth = depth; }
        fn model_draft_costs(&self) -> Option<CostCurve> { Some(CostCurve([46.6, 51.3, 60.8, 79.9, 118.0])) }
        fn propose_model(&mut self, history: &[u32], _next: u32, limit: usize) -> Result<Option<ModelProposal>> {
            // Misses are denser in the first half, so the chooser moves between widths.
            let tokens: Vec<u32> = (history.len() + 1..history.len() + 1 + limit.min(self.depth)).map(|p| {
                let miss = (p * 7919) % if p < 160 { 5 } else { 13 } == 0;
                self.reference[p] ^ u32::from(miss)
            }).collect();
            Ok(Some(ModelProposal { proposal_ms: self.draft_ms * tokens.len() as f64, tokens, catchup_ms: 0.0 }))
        }
    }
    /// Runs one 256-token request the way the scheduler does and returns the rows of each
    /// cycle; `draft_ms` and `verify_ms` give each cycle's measured times.
    fn adaptive_plan(draft_ms: impl Fn(usize) -> f64, verify_ms: impl Fn(usize, usize) -> f64) -> (Vec<usize>, Stats) {
        let (reference, prompt): (Vec<u32>, _) = ((0..400u32).map(|i| (i * 13 + i / 7) % 31 * 2).collect(), 24);
        let mut target = Adaptive { history: reference[..prompt].to_vec(), reference: reference.clone(), depth: 7, draft_ms: 0.0 };
        let mut spec = Speculator::from_selected(&reference[..prompt], prompt, reference[prompt], 64, Config::strict(256), Sampling::greedy()).unwrap();
        let mut rows = Vec::new();
        while !spec.is_done() {
            target.draft_ms = draft_ms(rows.len());
            let mut plan = spec.prepare_model(&mut target, 16, None, |_| false, |_| true).unwrap();
            let entry = target.history.len(); target.history.extend_from_slice(&plan.input);
            plan.record_target_ms(verify_ms(rows.len(), plan.input.len())); rows.push(plan.input.len());
            let output: Vec<u32> = (1..=plan.input.len()).map(|i| reference[entry + i]).collect();
            spec.finish_greedy(&mut target, plan, &output, |_| false).unwrap();
        }
        assert_eq!(spec.tokens(), &reference[prompt..prompt + 256]);
        (rows, std::mem::take(&mut spec.stats))
    }
    #[test]
    fn native_width_follows_the_ids_not_one_pass_timing() {
        // Two-Spark C1 shape (10-27 cards): verify passes at 0.6 x the prior, 1.55 ms per
        // proposed token; the measured price 2.6 locks at level 2 after the window.
        let prior = CostCurve([46.6, 51.3, 60.8, 79.9, 118.0]);
        let (calm, stats) = adaptive_plan(|_| 1.55, |_, rows| 0.6 * prior.ms(rows));
        assert_eq!(stats.draft_price, 2.0);
        assert!(calm.iter().any(|&r| r != calm[0]), "the chooser must still move with survival: {calm:?}");
        // The same request, same ids: the plan sequence is the same every time.
        assert_eq!(adaptive_plan(|_| 1.55, |_, rows| 0.6 * prior.ms(rows)).0, calm);
        // A single stalled pass or draft early in the chat does not change any later width.
        assert_eq!(adaptive_plan(|c| if c == 2 { 40.0 } else { 1.55 }, |_, rows| 0.6 * prior.ms(rows)).0, calm);
        assert_eq!(adaptive_plan(|_| 1.55, |c, rows| if c == 1 { 600.0 } else { 0.6 * prior.ms(rows) }).0, calm);
        // Run-to-run jitter of a few percent on every sample (the 10-27 cards: draft ms per
        // token 1.41 to 1.56, verify scale 0.59 to 0.64) leaves the plan unchanged too.
        let (jitter, other) = adaptive_plan(|c| [1.41, 1.56, 1.49][c % 3], |c, rows| [0.59, 0.64, 0.61, 0.6][c % 4] * prior.ms(rows));
        assert_eq!((jitter, other.plan_hash, other.widths), (calm, stats.plan_hash, stats.widths));
    }
    #[test]
    fn draft_price_locks_one_level_after_its_window() {
        let fill = |token_ms: &dyn Fn(usize) -> f64| { let mut control = Control::new(); for i in 0..PRICE_WINDOW {
            assert_eq!(control.draft_price(), TOKEN_PRIOR_MS);
            control.observe_price(Some(0.6), None); control.observe_price(None, Some(token_ms(i)));
        } control };
        // Measured 2.6 and 2.3 per prior ms (C1, structured) share level 2; 3.3 takes level 4.
        assert_eq!(fill(&|_| 1.55).draft_price(), 2.0);
        assert_eq!(fill(&|i| if i % 2 == 0 { 1.4 } else { 1.38 }).draft_price(), 2.0);
        assert_eq!(fill(&|i| if i == 0 { 1.55 } else { 2.0 }).draft_price(), 4.0);
        // One outlier among the window's samples does not reach the price.
        assert_eq!(fill(&|i| if i == 3 { 500.0 } else { 1.55 }).draft_price(), 2.0);
        // Far slower or faster drafters take their own power of two; the level then holds.
        let mut slow = fill(&|_| 150.0); assert_eq!(slow.draft_price(), 256.0);
        slow.observe_price(Some(0.01), Some(0.01)); assert_eq!(slow.draft_price(), 256.0);
        assert_eq!(fill(&|_| 0.3).draft_price(), 0.5);
        // Pure plain cycles (no proposals) never leave the prior.
        let mut plain = Control::new();
        for _ in 0..4 * PRICE_WINDOW { plain.observe_price(Some(0.6), None); }
        assert_eq!(plain.draft_price(), TOKEN_PRIOR_MS);
    }
    #[test]
    fn partial_failure_rolls_back_cycle_and_preserves_both_errors() {
        let prompt: Vec<u32> = (0..21).map(|i|i%7).collect();
        let mut target = toy(&prompt);
        target.fail_at = prompt.len()+2;
        let mut config = Config::strict(16); config.adaptive = false;
        let mut spec = Speculator::new(&prompt,&target.logits(),config,Sampling::greedy()).unwrap();
        let error = spec.step(&mut target,8, |_|false).unwrap_err();
        assert!(error.contains("forward failure"));
        assert_eq!(spec.history(),prompt); assert_eq!(target.history,prompt);
        target.fail_at = usize::MAX;
        spec.step(&mut target,8, |_|false).unwrap();
        assert_eq!(spec.history(),target.history);
        let mut target = toy(&prompt); target.fail_at=prompt.len()+2; target.fail_rollback=true;
        let mut spec = Speculator::new(&prompt,&target.logits(),config,Sampling::greedy()).unwrap();
        let error = spec.step(&mut target,8, |_|false).unwrap_err();
        assert!(error.contains("forward failure") && error.contains("rollback failure"));
    }
    #[test]
    fn lookup_has_no_periodic_extension_and_policy_recovers_from_serial_shadows() {
        let mut lookup = Lookup::default();
        let mut guesses = Vec::new();
        assert_eq!(lookup.propose(&[1,2,3,4,1,2],16,&mut guesses),2);
        assert_eq!(guesses,[3,4,1,2]);
        let mut policy = Control::new();
        let costs = CostCurve([70.0, 73.0, 80.0, 90.0, 110.0]);
        for _ in 0..30 { policy.observe(1,&[false]); }
        assert_eq!(policy.rows(1,16,costs),1);
        policy.observe(1,&[true]); policy.observe(1,&[true]);
        assert!(policy.rows(1,16,costs)>1);
        // A scheduler delay is not an input to this decision.
        for _ in 0..20 { policy.observe(1,&[true;15]); }
        assert_eq!(policy.rows(1,16,costs),16);
        for _ in 0..12 { policy.observe(1, &[true, false]); }
        assert!(policy.rates[1][1] < 0.1);
    }
    #[test]
    fn parallel_row_draws_equal_serial_draws() {
        let vocab = 2048;
        for options in [Sampling { temperature: 0.7, top_p: 0.9, seed: 7, ..Sampling::greedy() },
            Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, repetition_penalty: 1.1, presence_penalty: 0.5,
                frequency_penalty: 0.2, seed: 11, ..Sampling::greedy() }] {
            let prompt: Vec<u32> = (0..40).map(|i| (i * 37 % vocab) as u32).collect();
            let logits = |row: usize| -> Vec<f32> { (0..vocab).map(|id| (((id * 131 + row * 17) % 509) as f32 - 250.0) / 40.0).collect() };
            let mut spec = Speculator::new(&prompt, &logits(99), Config::strict(64), options).unwrap();
            let input = [spec.next, 5, 9, 1500];
            let output: Vec<_> = (0..input.len()).map(logits).collect();
            let allowed = |id: u32| !id.is_multiple_of(3);
            let rows = spec.sample_rows(&input, &output, &allowed);
            for i in 0..input.len() {
                let mut history = prompt.clone();
                history.extend_from_slice(&input[..=i]);
                let serial = Sampler::new(options).unwrap().sample(&output[i], &history, prompt.len(), history.len(), allowed);
                assert_eq!(rows[i], serial, "row {i}");
            }
            // Reused scratch (second cycle) keeps the draws.
            assert_eq!(spec.sample_rows(&input, &output, &allowed), rows);
            // A cohort's ahead-of-time draws (presample) are the same draws, keyed by history length.
            if parallel_rows() {
                let mask: Vec<bool> = (0..vocab as u32).map(allowed).collect();
                spec.presample(&input, &output, &mask);
                assert_eq!(spec.presampled.as_ref().map(|(at, r)| (*at, r.clone())), Some((prompt.len(), rows.clone())));
            }
        }
    }
    #[test]
    fn filtered_vocabulary_matches_serial_for_both_modes() {
        let prompt: Vec<u32> = (0..35).map(|i| i % 7).collect();
        let allowed = |id| id != 0 && id != 4;
        for options in [Sampling::greedy(), Sampling { temperature: 0.7, top_k: 4, seed: 42, ..Sampling::greedy() }] {
            let mut plain = toy(&prompt);
            let mut sampler = Sampler::new(options).unwrap();
            let mut row = plain.logits();
            let mut expected = Vec::new();
            for _ in 0..24 {
                let token = sampler.sample(&row, &plain.history, prompt.len(), plain.history.len(), allowed).unwrap();
                expected.push(token); row = plain.step(token).unwrap();
            }
            for mode in [Mode::Strict, Mode::Fast] {
                let mut target = toy(&prompt);
                let mut config = Config::strict(24); config.mode = mode; config.guard_error = 100.0;
                let mut spec = Speculator::new_filtered(&prompt, &target.logits(), config, options, allowed).unwrap();
                while !spec.is_done() { spec.step_filtered(&mut target, 16, |_| false, allowed).unwrap(); }
                assert_eq!(spec.tokens(), expected);
            }
        }
    }
    #[test]
    fn fast_guard_rejects_ties_support_changes_and_cdf_edges() {
        let mut guard = Guard::default();
        assert!(!guard.safe(&[1.,1.],&[],0,Sampling::greedy(),0.01,0));
        assert!(guard.safe(&[1.,0.],&[],0,Sampling::greedy(),0.01,0));
        for seed in 0..100 {
            let p = Sampling { seed,temperature:0.7,top_k:3,top_p:0.8,..Sampling::greedy() };
            let logits = [2.0,1.1,0.3,-2.0];
            let mut sampler = Sampler::new(p).unwrap();
            let selected = sampler.sample(&logits,&[0,1],2,2, |_|true).unwrap();
            if guard.safe(&logits,&[0,1],2,p,0.01,selected) {
                for mask in 0..16 {
                    let changed: Vec<f32> = logits.iter().enumerate().map(|(i,&v)|v+if mask&(1<<i)==0 {-0.01} else {0.01}).collect();
                    assert_eq!(selected,sampler.sample(&changed,&[0,1],2,2, |_|true).unwrap());
                }
                // The guard has no min_p bound, so min_p always takes the exact path.
                assert!(!guard.safe(&logits,&[0,1],2,Sampling { min_p:0.01,..p },0.01,selected));
            }
        }
        let mut certified = 0;
        for seed in 0..300 {
            let logits: Vec<f32> = (0..6).map(|i|((seed*31+i*17)%43) as f32/17.0).collect();
            let p = Sampling { seed,temperature:0.4+0.3*(seed%3) as f64,top_k:2+(seed%5) as usize,
                top_p:0.65+0.1*(seed%4) as f64,repetition_penalty:1.1,presence_penalty:0.1,..Sampling::greedy() };
            let mut sampler = Sampler::new(p).unwrap();
            let history = [0,1,1,2];
            let token = sampler.sample(&logits,&history,2,4, |_|true).unwrap();
            if guard.safe(&logits,&history,2,p,0.02,token) {
                certified += 1;
                for mask in 0..64 {
                    let changed: Vec<f32> = logits.iter().enumerate().map(|(i,&v)|v+if mask&(1<<i)==0 {-0.019} else {0.019}).collect();
                    assert_eq!(token,sampler.sample(&changed,&history,2,4, |_|true).unwrap());
                }
            }
        }
        assert!(certified > 20);
    }
}
