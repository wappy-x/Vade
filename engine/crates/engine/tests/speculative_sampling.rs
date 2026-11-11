//! Sampled drafting (draft::Mode::Sampled): the draft head only proposes. Each row commits the
//! plain draw at its position, so the committed token follows the target distribution p and
//! equals the drafting-off id whatever the draft q is; a proposal stays only when it is that id.
use spark_engine::{draft::{Config, Mode, ModelProposal, Speculator, Target}, sample::{Distribution, Sampler, Sampling, uniform}};

/// Target rows do not depend on history: p = 0.1, 0.2, 0.3, 0.4 at temperature 1.
/// `q` None is a head equal to the target: it proposes from the target's own filtered row.
struct Model { history: Vec<u32>, q: Option<Vec<(u32, f64)>>, proposal: Option<u32> }
fn target() -> Vec<f32> { [0.1f64, 0.2, 0.3, 0.4].iter().map(|p| p.ln() as f32).collect() }
impl Target for Model {
    fn checkpoint(&self) -> usize { self.history.len() }
    fn truncate(&mut self, p: usize) -> Result<(), String> { self.history.truncate(p); Ok(()) }
    fn step(&mut self, t: u32) -> Result<Vec<f32>, String> { self.history.push(t); Ok(target()) }
    fn native_draft_policy(&self) -> bool { true }
    fn sampled_model_draft(&self) -> bool { true }
    fn speculative_model_draft(&self) -> bool { true }
    fn model_draft_width(&self) -> usize { 1 }
    /// As the service: draw the proposal from q with the request's own draw at its position.
    fn propose_sampled_model(&mut self, history: &[u32], prompt_len: usize, next: u32, limit: usize, sampling: Sampling,
        allowed: &dyn Fn(u32) -> bool) -> Result<(ModelProposal, Vec<Distribution>), String> {
        assert_eq!(limit, 1);
        let mut prefix = history.to_vec();
        prefix.push(next);
        let q = match &self.q {
            Some(q) => Distribution(q.clone()),
            None => Sampler::new(sampling)?.distribution(&target(), &prefix, prompt_len, allowed)?,
        };
        let token = match self.proposal { Some(t) => t, None => q.draw(uniform(sampling.seed, prefix.len()))? };
        Ok((ModelProposal { tokens: vec![token], proposal_ms: 0.0, catchup_ms: 0.0 }, vec![q]))
    }
}
fn options(seed: u64, top_k: usize) -> Sampling { Sampling { temperature: 1.0, top_k, top_p: 1.0, seed, ..Sampling::greedy() } }
/// The second token with drafting off: two plain draws.
fn plain_second(seed: u64, top_k: usize) -> u32 {
    let mut sampler = Sampler::new(options(seed, top_k)).unwrap();
    let mut history = vec![3, 1];
    for _ in 0..2 {
        let id = sampler.sample(&target(), &history, 2, history.len(), |_| true).unwrap();
        history.push(id);
    }
    history[3]
}
/// One request of two tokens: returns the second token and whether its proposal was accepted.
fn second_token(seed: u64, q: Option<&[(u32, f64)]>, top_k: usize, proposal: Option<u32>) -> Result<(u32, bool), String> {
    let prompt = vec![3, 1];
    let mut model = Model { history: prompt.clone(), q: q.map(|q| q.to_vec()), proposal };
    let config = Config { mode: Mode::Sampled, adaptive: false, ..Config::strict(2) };
    let mut spec = Speculator::new(&prompt, &target(), config, options(seed, top_k))?;
    assert!(spec.sampled_verifier());
    let mut proposed = None;
    while !spec.is_done() {
        let plan = spec.prepare_with_target(&mut model, 4, |_| false, |_| true)?;
        if plan.input.len() == 2 { proposed = Some(plan.input[1]); }
        let output = model.verify_batch(&plan.input)?;
        spec.finish_exact(&mut model, plan, &output, |_| false, |_| true)?;
        assert_eq!(spec.history(), model.history);
    }
    let token = spec.tokens()[1];
    Ok((token, proposed.is_some_and(|d| spec.stats.accepted > 0 && d == token)))
}
#[test]
fn sampled_verification_keeps_the_target_distribution() {
    let p = [0.1, 0.2, 0.3, 0.4];
    // A head equal to the target proposes the plain draw itself, so every proposal stays.
    for seed in 0..64 { assert_eq!(second_token(seed, None, 4, None).unwrap(), (plain_second(seed, 4), true), "seed {seed}"); }
    // A draft far from the target: every committed id is the plain id, so the ids follow p.
    let far = [(0, 0.7), (1, 0.1), (2, 0.1), (3, 0.1)];
    let runs = 4000;
    let (mut counts, mut accepted) = ([0usize; 4], 0);
    for seed in 0..runs {
        let (token, kept) = second_token(seed, Some(&far), 4, None).unwrap();
        assert_eq!(token, plain_second(seed, 4), "seed {seed}");
        counts[token as usize] += 1;
        accepted += usize::from(kept);
    }
    // The far head still lands some proposals; bounds are about four standard errors at 4000 runs.
    let share = |n: usize| n as f64 / runs as f64;
    assert!(accepted > 0 && accepted < runs as usize, "acceptance {accepted}/{runs}");
    for (id, &want) in p.iter().enumerate() {
        assert!((share(counts[id]) - want).abs() < 0.03, "token {id}: {}/{runs} for p {want} (counts {counts:?})", counts[id]);
    }
    // The run is keyed by seed and position only: a replay gives the same ids.
    let again: Vec<_> = (0..64).map(|seed| second_token(seed, Some(&far), 4, None).unwrap()).collect();
    assert_eq!(again, (0..64).map(|seed| second_token(seed, Some(&far), 4, None).unwrap()).collect::<Vec<_>>());
}
#[test]
fn proposal_outside_the_target_support_is_never_kept() {
    // top_k 3 drops token 0 (p 0.1) from the target; a draft certain of token 0 is always
    // rejected, and the committed id is the plain draw over tokens 1..3.
    let certain = [(0, 1.0)];
    let mut seen = [0usize; 4];
    for seed in 0..256 {
        let (token, kept) = second_token(seed, Some(&certain), 3, None).unwrap();
        assert!(!kept && token != 0 && token == plain_second(seed, 3), "seed {seed}: token {token}");
        seen[token as usize] += 1;
    }
    assert!(seen[1..].iter().all(|&n| n > 0), "{seen:?}");
    // A proposal the draft gives no mass changes nothing: the row still commits the plain id.
    for seed in 0..64 { assert_eq!(second_token(seed, Some(&certain), 4, Some(2)).unwrap().0, plain_second(seed, 4), "seed {seed}"); }
}
