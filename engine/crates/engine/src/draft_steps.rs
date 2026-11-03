//! One chat's draft stop, step by step. Before each draft step the rule asks: can this step
//! still pay? It reads the chat's own verified counts, the target's row cost prior and the
//! request's locked draft price. So a plan follows the ids and the drafter's probabilities, not
//! one pass's timing; near a price level's boundary two runs can lock different levels and
//! differ (Control::draft_price). Verification picks every token, so a stop changes the rows of
//! a cycle and never an output id. Cycles of a chat in a batch do not count here.
use super::CostCurve;

/// Bands of the drafter probability p. Its upper values are a ranking score that can pass 1.
const BANDS: [f32; 2] = [0.6, 1.0];
/// A chat starts as the width chooser's survival prior does: four trials at 0.8.
const PRIOR_KEEP: f64 = 0.8;
const PRIOR_TRIALS: f64 = 4.0;
/// A band's own rate replaces the chat's mean rate after about this many trials.
const BAND_TRIALS: f64 = 8.0;
/// The value of a token starts as this many cycles of three rows that commit two tokens.
const START_CYCLES: f64 = 8.0;
/// Old cycles fade, so a chat that moves from prose to code follows the new text.
const FADE: f64 = 0.995;
/// The first step of every 16th cycle always runs, so a chat that stopped drafting can recover.
const PROBE_CYCLES: usize = 16;

fn band(p: f32) -> usize { BANDS.iter().position(|&edge| p < edge).unwrap_or(BANDS.len()) }

/// One cycle's stop. `keep` is the chat's acceptance per band of p. A step at depth d runs while
/// the chance that all earlier drafts pass is at least `need[d]`.
#[derive(Clone, Copy, Debug)]
pub struct StepRule { keep: [f64; 3], need: [f64; 15] }
impl StepRule {
    pub fn halts(&self, chain: f64, depth: usize) -> bool { self.need.get(depth).is_none_or(|&need| chain < need) }
    pub fn keep(&self, p: f32) -> f64 { self.keep[band(p)] }
}

/// One chat's counts, faded per cycle: verified drafts per band, draft steps run and kept, and
/// the plan's cost and committed tokens in the units of the cost prior.
#[derive(Default)]
pub(super) struct Steps { hits: [f64; 3], trials: [f64; 3], run: f64, kept: f64, cost: f64, tokens: f64, armed: bool }
impl Steps {
    /// A step pays when its expected tokens are worth more than what it costs: one draft step
    /// always, one verify row when the draft passes the drafter's stop. The tokens' value is
    /// this chat's cost per committed token.
    pub(super) fn rule(&mut self, costs: CostCurve, price: f64, cycle: usize) -> StepRule {
        self.armed = true;
        if self.tokens == 0.0 {
            self.cost = START_CYCLES * (costs.ms(3) + 2.0 * price);
            self.tokens = START_CYCLES * 2.0;
        }
        let mean = (self.hits.iter().sum::<f64>() + PRIOR_TRIALS * PRIOR_KEEP) / (self.trials.iter().sum::<f64>() + PRIOR_TRIALS);
        let keep = std::array::from_fn(|b| (self.hits[b] + BAND_TRIALS * mean) / (self.trials[b] + BAND_TRIALS));
        let passes = (self.kept + PRIOR_TRIALS * PRIOR_KEEP) / (self.run + PRIOR_TRIALS);
        let value = self.cost / self.tokens * mean;
        let mut need: [f64; 15] = std::array::from_fn(|d| (costs.ms(d + 2) - costs.ms(d + 1) + price / passes) / value);
        if cycle % PROBE_CYCLES == 0 { need[0] = 0.0; }
        StepRule { keep, need }
    }
    /// `confidence` holds p of each step the drafter ran; it is empty when it ran none. Only a
    /// cycle that asked for a rule counts.
    pub(super) fn observe(&mut self, confidence: &[f32], draft: &[u32], choices: &[u32], rows: usize, costs: CostCurve, price: f64) {
        if !std::mem::take(&mut self.armed) { return; }
        for ((&p, a), b) in confidence.iter().zip(draft).zip(choices) {
            self.trials[band(p)] += 1.0;
            if a != b { break; }
            self.hits[band(p)] += 1.0;
        }
        self.run += confidence.len() as f64;
        if !confidence.is_empty() { self.kept += draft.len() as f64; }
        self.cost += costs.ms(rows) + price * confidence.len() as f64;
        self.tokens += choices.len() as f64;
        for count in self.hits.iter_mut().chain(&mut self.trials).chain([&mut self.run, &mut self.kept, &mut self.cost, &mut self.tokens]) { *count *= FADE; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::{Config, ModelProposal, Speculator, Stats, Target};
    use crate::sample::Sampling;
    type Result<T> = std::result::Result<T, String>;
    const PRIOR: CostCurve = CostCurve([46.6, 51.3, 60.8, 79.9, 118.0]);
    /// One chat on a target that stops drafts step by step, like FP4-Local on one or two
    /// Sparks. `quality` gives the acceptance of a draft at a position and depth by its p.
    struct Stepped { history: Vec<u32>, reference: Vec<u32>, depth: usize, rule: Option<StepRule>, confidence: Vec<f32>, steps: usize,
        quality: fn(usize, usize) -> (f32, bool), alone: bool, lookup: bool }
    impl Target for Stepped {
        fn checkpoint(&self) -> usize { self.history.len() }
        fn truncate(&mut self, p: usize) -> Result<()> { self.history.truncate(p); Ok(()) }
        fn step(&mut self, t: u32) -> Result<Vec<f32>> { self.history.push(t); Ok(vec![0.0; 64]) }
        fn native_draft_policy(&self) -> bool { true } fn adaptive_model_depth(&self) -> bool { self.alone }
        fn model_draft_width(&self) -> usize { 7 } fn limit_model_depth(&mut self, depth: usize) { self.depth = depth; }
        fn model_draft_costs(&self) -> Option<CostCurve> { Some(PRIOR) }
        fn draft_step_rule(&mut self, rule: StepRule) -> bool { self.rule = Some(rule); true }
        fn take_draft_confidence(&mut self) -> Vec<f32> { self.rule = None; std::mem::take(&mut self.confidence) }
        fn propose_model(&mut self, history: &[u32], _next: u32, limit: usize) -> Result<Option<ModelProposal>> {
            if self.lookup { return Ok(Some(ModelProposal { proposal_ms: 0.1, tokens: self.reference[history.len() + 1..history.len() + 3].to_vec(), catchup_ms: 0.0 })); }
            let (rule, mut chain, mut tokens) = (self.rule.take(), 1.0, Vec::new());
            for depth in 0..limit.min(self.depth) {
                if rule.is_some_and(|r| r.halts(chain, depth)) { break; }
                let position = history.len() + 1 + depth;
                let (p, pass) = (self.quality)(position, depth);
                self.confidence.push(p); self.steps += 1;
                if p < 0.4 { break; }
                tokens.push(self.reference[position] ^ u32::from(!pass));
                if let Some(r) = rule { chain *= r.keep(p); }
            }
            Ok(Some(ModelProposal { proposal_ms: 1.5, tokens, catchup_ms: 0.0 }))
        }
    }
    fn run(quality: fn(usize, usize) -> (f32, bool), verify_ms: impl Fn(usize, usize) -> f64) -> (Stats, usize) {
        let (reference, prompt): (Vec<u32>, _) = ((0..1200u32).map(|i| (i * 13 + i / 7) % 31 * 2).collect(), 24);
        let mut target = Stepped { history: reference[..prompt].to_vec(), reference: reference.clone(), depth: 7, rule: None, confidence: Vec::new(), steps: 0, quality, alone: true, lookup: false };
        let mut spec = Speculator::from_selected(&reference[..prompt], prompt, reference[prompt], 64, Config::strict(800), Sampling::greedy()).unwrap();
        let mut cycle = 0;
        while !spec.is_done() {
            let mut plan = spec.prepare_model(&mut target, 16, None, |_| false, |_| true).unwrap();
            let entry = target.history.len(); target.history.extend_from_slice(&plan.input);
            plan.record_target_ms(verify_ms(cycle, plan.input.len())); cycle += 1;
            let output: Vec<u32> = (1..=plan.input.len()).map(|i| reference[entry + i]).collect();
            spec.finish_greedy(&mut target, plan, &output, |_| false).unwrap();
        }
        // Every output id is the reference id, at any stop.
        assert_eq!(spec.tokens(), &reference[prompt..prompt + 800]);
        (std::mem::take(&mut spec.stats), target.steps)
    }
    fn rows_per_cycle(stats: &Stats) -> f64 { stats.rows as f64 / stats.cycles as f64 }
    /// Code's shape: almost every draft passes at a high p.
    fn sure(position: usize, _: usize) -> (f32, bool) { (1.2, position * 7919 % 97 != 0) }
    /// Prose's shape: most drafts in the low band fail, most in the high band pass.
    fn mixed(position: usize, _: usize) -> (f32, bool) {
        let roll = position * 7919 % 100;
        if roll < 45 { (0.45, roll < 12) } else { (1.1, roll < 85) }
    }
    #[test]
    fn a_sure_chat_drafts_the_whole_width() {
        let (stats, _) = run(sure, |_, rows| PRIOR.ms(rows));
        assert!(rows_per_cycle(&stats) > 7.0, "{} rows per cycle: {:?}", rows_per_cycle(&stats), stats.widths);
    }
    #[test]
    fn a_low_band_draft_ends_the_chain_a_high_band_draft_does_not() {
        let (stats, _) = run(mixed, |_, rows| PRIOR.ms(rows));
        let (rows, wide) = (rows_per_cycle(&stats), stats.widths[3..].iter().sum::<usize>());
        assert!(rows < 3.6, "{rows} rows per cycle: {:?}", stats.widths);
        assert!(wide > 0, "high band chains go past three rows: {:?}", stats.widths);
    }
    #[test]
    fn the_plan_follows_the_ids_not_the_timing() {
        let (calm, _) = run(mixed, |_, rows| 0.6 * PRIOR.ms(rows));
        let (jitter, _) = run(mixed, |cycle, rows| [0.59, 0.64, 0.61, 0.6][cycle % 4] * PRIOR.ms(rows));
        assert_eq!((calm.plan_hash, calm.widths), (jitter.plan_hash, jitter.widths));
    }
    #[test]
    fn a_failing_chat_runs_fewer_steps_and_still_probes() {
        fn bad(position: usize, _: usize) -> (f32, bool) { (0.45, position * 7919 % 100 < 2) }
        let (stats, steps) = run(bad, |_, rows| PRIOR.ms(rows));
        assert!(steps * 4 < stats.cycles * 3, "{steps} steps in {} cycles", stats.cycles);
        assert!(steps >= stats.cycles / PROBE_CYCLES, "{steps} steps in {} cycles", stats.cycles);
    }
    #[test]
    fn a_chat_that_moves_to_sure_text_goes_deep_again() {
        fn moved(position: usize, depth: usize) -> (f32, bool) { if position < 424 { mixed(position, depth) } else { sure(position, depth) } }
        let (stats, _) = run(moved, |_, rows| PRIOR.ms(rows));
        assert!(stats.widths[7] > 30, "{:?}", stats.widths);
    }
    #[test]
    fn a_stop_ends_with_its_cycle_and_a_batch_cycle_does_not_count() {
        let reference: Vec<u32> = (0..80u32).map(|i| i % 31 * 2).collect();
        let mut target = Stepped { history: reference[..24].to_vec(), reference: reference.clone(), depth: 7, rule: None, confidence: Vec::new(), steps: 0, quality: mixed, alone: true, lookup: true };
        let mut spec = Speculator::from_selected(&reference[..24], 24, reference[24], 64, Config::strict(32), Sampling::greedy()).unwrap();
        let cycle = |target: &mut Stepped, spec: &mut Speculator| {
            let mut plan = spec.prepare_model(target, 16, None, |_| false, |_| true).unwrap();
            let entry = target.history.len(); target.history.extend_from_slice(&plan.input);
            plan.record_target_ms(50.0);
            let output: Vec<u32> = (1..=plan.input.len()).map(|i| reference[entry + i]).collect();
            spec.finish_greedy(target, plan, &output, |_| false).unwrap();
        };
        // A lookup cycle alone: no draft reads the stop, and the finished cycle ends it.
        cycle(&mut target, &mut spec);
        assert!(target.rule.is_none(), "a stop outlived its cycle");
        let counted = (spec.steps.tokens, spec.steps.cost);
        // The chat joins a batch: it gets no stop, it drafts its width, and its cycle does not count.
        (target.alone, target.lookup, target.depth) = (false, false, 3);
        cycle(&mut target, &mut spec);
        assert_eq!(target.steps, 3, "a chat in a batch drafts its depth");
        assert_eq!((spec.steps.tokens, spec.steps.cost), counted);
    }
    #[test]
    fn a_chat_that_starts_in_a_batch_starts_from_the_prior() {
        let mut steps = Steps::default();
        steps.observe(&[1.2], &[1], &[1, 2], 2, PRIOR, 2.0);
        assert_eq!(steps.tokens, 0.0);
        steps.rule(PRIOR, 2.0, 1);
        assert_eq!(steps.tokens, START_CYCLES * 2.0);
    }
    fn counted(cost: f64) -> Steps { Steps { hits: [8.0; 3], trials: [10.0; 3], run: 40.0, kept: 30.0, cost, tokens: 100.0, armed: false } }
    #[test]
    fn the_need_rises_with_the_price_and_falls_with_the_value_of_a_token() {
        let need = |cost: f64, price: f64| counted(cost).rule(PRIOR, price, 1).need[1];
        assert!(need(3000.0, 4.0) > need(3000.0, 2.0));
        assert!(need(6000.0, 2.0) < need(3000.0, 2.0));
    }
    #[test]
    fn keep_reads_the_band_of_p() {
        let mut steps = counted(3000.0);
        steps.hits = [1.0, 5.0, 9.0];
        let rule = steps.rule(PRIOR, 2.0, 1);
        assert!(rule.keep(0.45) < rule.keep(0.8) && rule.keep(0.8) < rule.keep(1.2));
    }
}
