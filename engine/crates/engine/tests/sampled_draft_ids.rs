//! Draft "sampled" commits the plain sampled ids: the draft head only proposes.
use spark_engine::draft::{Config, Mode, ModelProposal, Speculator, Target};
use spark_engine::sample::{Distribution, Sampler, Sampling, uniform};
type Result<T> = std::result::Result<T, String>;

/// A chat whose draft head samples its proposals (draft "sampled"). Rows are functions of
/// history; `busy` is a drafter that another chat holds, `skew` a head unlike the target.
struct SampledDraft { history: Vec<u32>, busy: bool, skew: f32 }
fn sampled_row(history: &[u32], skew: f32) -> Vec<f32> {
    let h = history.iter().fold(7u32, |h, &t| h.wrapping_mul(31).wrapping_add(t));
    (0..8u32).map(|id| (h.wrapping_add(id * 5) % 11) as f32 / 4.0 + if id == h % 8 { skew } else { 0.0 }).collect()
}
impl Target for SampledDraft {
    fn checkpoint(&self) -> usize { self.history.len() }
    fn truncate(&mut self, p: usize) -> Result<()> { self.history.truncate(p); Ok(()) }
    fn step(&mut self, t: u32) -> Result<Vec<f32>> { self.history.push(t); Ok(sampled_row(&self.history, 0.0)) }
    fn native_draft_policy(&self) -> bool { true }
    fn model_draft_width(&self) -> usize { 3 }
    fn sampled_model_draft(&self) -> bool { true }
    fn speculative_model_draft(&self) -> bool { true }
    fn propose_sampled_model(&mut self, history: &[u32], prompt_len: usize, next: u32, limit: usize,
        sampling: Sampling, allowed: &dyn Fn(u32) -> bool) -> Result<(ModelProposal, Vec<Distribution>)> {
        let (mut prefix, mut tokens, mut q) = (history.to_vec(), Vec::new(), Vec::new());
        prefix.push(next);
        let mut sampler = Sampler::new(sampling)?;
        for _ in 0..if self.busy { 0 } else { limit.min(3) } {
            let d = sampler.distribution(&sampled_row(&prefix, self.skew), &prefix, prompt_len, allowed)?;
            let token = d.draw(uniform(sampling.seed, prefix.len()))?;
            prefix.push(token); tokens.push(token); q.push(d);
        }
        Ok((ModelProposal { tokens, proposal_ms: 0.0, catchup_ms: 0.0 }, q))
    }
}
/// Drafting on or off, a busy drafter or a head unlike the target: the ids stay the plain ids.
#[test]
fn sampled_drafts_commit_the_plain_ids() {
    let prompt: Vec<u32> = (0..12).map(|i| i % 8).collect();
    let options = Sampling { temperature: 0.9, top_k: 4, top_p: 0.9, seed: 42, ..Sampling::greedy() };
    let mut plain = Sampler::new(options).unwrap();
    let mut history = prompt.clone();
    let mut expected = Vec::new();
    while expected.len() < 40 {
        let id = plain.sample(&sampled_row(&history, 0.0), &history, prompt.len(), history.len(), |_| true).unwrap();
        expected.push(id);
        history.push(id);
    }
    let (mut differ, mut skew_rejected) = (Vec::new(), 0);
    for (busy, skew) in [(false, 0.0), (false, 0.7), (true, 0.0)] {
        for cap in [1, 2, 4, 16] {
            let mut target = SampledDraft { history: prompt.clone(), busy, skew };
            let mut config = Config::strict(40);
            config.mode = Mode::Sampled;
            let mut spec = Speculator::new(&prompt, &sampled_row(&prompt, 0.0), config, options).unwrap();
            while !spec.is_done() { spec.step(&mut target, cap, |_| false).unwrap(); }
            if spec.tokens() != expected { differ.push((busy, skew, cap)); }
            // A head equal to the target proposes the target's own draws: every one survives.
            if !busy && skew == 0.0 && cap > 1 { assert!(spec.stats.proposed >= 19 && spec.stats.accepted == spec.stats.proposed, "cap {cap}: {} of {} accepted", spec.stats.accepted, spec.stats.proposed); }
            // A skewed head must lose a proposal, and a busy drafter must propose nothing.
            if skew != 0.0 { skew_rejected += spec.stats.proposed - spec.stats.accepted; }
            if busy { assert_eq!(spec.stats.proposed, 0, "cap {cap}"); }
        }
    }
    assert!(differ.is_empty(), "sampled drafting changed the ids for (busy, skew, cap) {differ:?}");
    assert!(skew_rejected > 0, "the skewed head lost no proposal");
}
