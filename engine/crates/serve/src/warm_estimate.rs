//! Warm-cache estimate: before prefill runs, how much of the prompt is cached
//! and about how long the first token takes, with the cache ("warm") and
//! without it ("cold"). The estimate is a hint. It reads times the scheduler
//! already measures and makes no scheduling, cache or sampling decision, so
//! token IDs never depend on it.
//!
//! Model, in ms from admission to the first logits, with U = P - C rows to prefill:
//! `setup + [U > 0] * (fixed + max(0, U - TAIL_ROWS) * row * (1 + (P + C) / 2 / CONTEXT_SCALE))`.
//! `setup` is the measured time from admission to the estimate (cache lookup
//! and restore). A warm turn's short tail costs about the same as a short cold
//! prompt, so `fixed` covers the first `TAIL_ROWS` rows.
use std::sync::Mutex;
use std::time::Duration;

/// Rows covered by the fixed cost.
const TAIL_ROWS: usize = 256;
/// A prompt with at least this many rows teaches the per-row rate.
const BULK_ROWS: usize = 2048;
/// Mean context, in tokens, that doubles the per-row cost. It shapes only
/// this hint and comes from the one-Spark cold first-token times for 8K to
/// 128K prompts (docs/benchmarks.md).
const CONTEXT_SCALE: f64 = 300_000.0;
/// The weight of each new time after the first three.
const WEIGHT: f64 = 0.25;

/// The running prefill rate of one served model. One engine process serves one
/// model, so the process estimator is the per-model estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimator {
    fixed_ms: f64,
    row_ms: f64,
    fixed_seen: u32,
    row_seen: u32,
}

/// One request's estimate, fixed at admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    /// Ms from admission to the first logits with the cached prefix.
    pub warm_ms: f64,
    /// Ms for the same prompt with nothing cached.
    pub cold_ms: f64,
    setup: Duration,
}

impl Estimator {
    /// The start values come from the one-Spark FP4-Local card (main 380d8e5,
    /// docs/benchmarks.md): a cold 8,192-token prompt reached its first token
    /// in 3.761 s, and a short prefill costs about 350 ms. Each start value
    /// counts as one measured time.
    pub const SEED: Self = Self { fixed_ms: 350.0,
        row_ms: (3761.0 - 350.0) / ((8192 - TAIL_ROWS) as f64 * (1.0 + 8192.0 / 2.0 / CONTEXT_SCALE)),
        fixed_seen: 0, row_seen: 0 };

    fn context_factor(prompt: usize, cached: usize) -> f64 { 1.0 + (prompt + cached) as f64 / 2.0 / CONTEXT_SCALE }
    fn prefill_ms(&self, prompt: usize, cached: usize) -> f64 {
        let rows = prompt.saturating_sub(cached);
        if rows == 0 { return 0.0; }
        self.fixed_ms + rows.saturating_sub(TAIL_ROWS) as f64 * self.row_ms * Self::context_factor(prompt, cached)
    }
    /// `setup` is the time already spent since admission.
    pub fn estimate(&self, prompt: usize, cached: usize, setup: Duration) -> Estimate {
        let cached = cached.min(prompt);
        let warm_ms = ms(setup) + self.prefill_ms(prompt, cached);
        Estimate { prompt_tokens: prompt, cached_tokens: cached, warm_ms,
            cold_ms: if cached == 0 { warm_ms } else { self.prefill_ms(prompt, 0) }, setup }
    }
    /// Learns from one finished prefill: `first_logits` runs from admission.
    pub fn observe(&mut self, estimate: &Estimate, first_logits: Duration) {
        let (prompt, cached) = (estimate.prompt_tokens, estimate.cached_tokens);
        let rows = prompt - cached;
        let spent = ms(first_logits.saturating_sub(estimate.setup));
        if rows == 0 { return; }
        if rows <= TAIL_ROWS {
            blend(&mut self.fixed_ms, &mut self.fixed_seen, spent);
        } else if rows >= BULK_ROWS {
            let scaled_rows = (rows - TAIL_ROWS) as f64 * Self::context_factor(prompt, cached);
            blend(&mut self.row_ms, &mut self.row_seen, ((spent - self.fixed_ms) / scaled_rows).max(0.0));
        }
    }
}

/// A running mean of the start value and the first three times, then a
/// moving average.
fn blend(value: &mut f64, seen: &mut u32, sample: f64) {
    *seen = seen.saturating_add(1);
    *value += (1.0 / (*seen as f64 + 1.0)).max(WEIGHT) * (sample - *value);
}

fn ms(time: Duration) -> f64 { time.as_secs_f64() * 1000.0 }

impl Estimate {
    /// Stream response headers, sent before prefill starts.
    pub fn headers(&self) -> String {
        format!("X-Vade-Prompt-Tokens: {}\r\nX-Vade-Cached-Tokens: {}\r\nX-Vade-Estimated-First-Token-Ms: {:.0}\r\nX-Vade-Estimated-Cold-Ms: {:.0}\r\n",
            self.prompt_tokens, self.cached_tokens, self.warm_ms, self.cold_ms)
    }
    /// Fields of the top-level `vade` object, next to the measured prefill times.
    pub fn json(&self) -> String {
        format!("\"prefill_estimated_ms\":{:.1},\"prefill_cold_estimated_ms\":{:.1}", self.warm_ms, self.cold_ms)
    }
}

static PROCESS: Mutex<Estimator> = Mutex::new(Estimator::SEED);

/// The process estimate for an admitted request.
pub(crate) fn current(prompt: usize, cached: usize, setup: Duration) -> Estimate {
    PROCESS.lock().unwrap_or_else(|e| e.into_inner()).estimate(prompt, cached, setup)
}
/// Teaches the process estimator one finished prefill.
pub(crate) fn learn(estimate: &Estimate, first_logits: Duration) {
    PROCESS.lock().unwrap_or_else(|e| e.into_inner()).observe(estimate, first_logits);
}

#[cfg(test)]
#[path = "warm_estimate_tests.rs"]
mod tests;
