//! Compact canonical rows must preserve strict MTP's position-keyed token stream.
use spark_engine::{draft::{Config, ModelProposal, Speculator, Target}, sample::{Sampler, Sampling}};

struct Model { history: Vec<u32>, corrupt: bool }
impl Model {
    fn logits(&self) -> Vec<f32> {
        let state = self.history.iter().fold(7usize, |s, &t| s.wrapping_mul(31) ^ t as usize);
        (0..128).map(|id| ((id * 37 + state % 997) % 131) as f32 / 13.0).collect()
    }
}
impl Target for Model {
    fn checkpoint(&self) -> usize { self.history.len() }
    fn truncate(&mut self, p: usize) -> Result<(), String> { self.history.truncate(p); Ok(()) }
    fn step(&mut self, t: u32) -> Result<Vec<f32>, String> { self.history.push(t); Ok(self.logits()) }
    fn native_draft_policy(&self) -> bool { true }
    fn sampled_model_draft(&self) -> bool { true }
    fn model_draft_width(&self) -> usize { 3 }
    fn propose_model(&mut self, _: &[u32], next: u32, limit: usize) -> Result<Option<ModelProposal>, String> {
        let mut model = Self { history: self.history.clone(), corrupt: self.corrupt };
        let mut tokens = Vec::new();
        let mut next = next;
        for _ in 0..limit {
            next = spark_engine::greedy_token(&model.step(next)?)?;
            tokens.push(next);
        }
        if self.corrupt && let Some(t) = tokens.first_mut() { *t = (*t + 1) % 128; }
        Ok(Some(ModelProposal { tokens, proposal_ms: 0.0, catchup_ms: 0.0 }))
    }
}
fn shortlist(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut top: Vec<_> = row.iter().enumerate().map(|(id, &v)| (id as u32, v)).collect();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    top.truncate(k);
    top
}
fn run(options: Sampling, compact: bool, corrupt: bool) -> (Vec<u32>, usize, usize) {
    let prompt = vec![1, 4, 7];
    let mut model = Model { history: prompt.clone(), corrupt };
    let mut spec = Speculator::new(&prompt, &model.logits(), Config::strict(48), options).unwrap();
    let mut sampler = Sampler::new(options).unwrap();
    let mut compact_cycles = 0;
    while !spec.is_done() {
        let plan = spec.prepare_with_target(&mut model, 4, |_| false, |_| true).unwrap();
        let output = model.verify_batch(&plan.input).unwrap();
        let choices = compact.then(|| output.iter().enumerate().map(|(i, row)| {
            sampler.sample_top_k(&shortlist(row, options.top_k), spec.history().len() + i + 1, |_| true)
        }).collect::<Option<Vec<_>>>()).flatten();
        if let Some(choices) = choices {
            let best: Vec<_> = output.iter().map(|r| spark_engine::greedy_token(r).unwrap()).collect();
            spec.finish_sampled(&mut model, plan, &choices, &best, |_| false).unwrap();
            compact_cycles += 1;
        } else {
            spec.finish_exact(&mut model, plan, &output, |_| false, |_| true).unwrap();
        }
        assert_eq!(spec.history(), model.history);
    }
    (spec.tokens().to_vec(), compact_cycles, spec.stats.rollbacks)
}
#[test]
fn compact_verified_rows_match_full_logits_with_acceptance_and_rejection() {
    for seed in 0..16 {
        for k in [1, 20, 64] {
            for top_p in [0.8, 1.0] {
                let options = Sampling { temperature: 0.7, top_k: k, top_p, seed, ..Sampling::greedy() };
                for corrupt in [false, true] {
                    let full = run(options, false, corrupt);
                    let compact = run(options, true, corrupt);
                    assert_eq!(compact.0, full.0, "seed={seed} k={k} p={top_p} corrupt={corrupt}");
                    assert!(compact.1 > 0);
                    if corrupt { assert!(compact.2 > 0); }
                }
            }
        }
    }
}
#[test]
fn uncertain_nucleus_boundary_requests_full_canonical_logits() {
    let options = Sampling { temperature: 1.0, top_k: 2, top_p: 0.5, ..Sampling::greedy() };
    let mut sampler = Sampler::new(options).unwrap();
    assert_eq!(sampler.sample_top_k(&[(0, 0.0), (1, 0.0)], 4, |_| true), None);
    assert!(sampler.sample(&[0.0, 0.0], &[0, 1], 2, 4, |_| true).is_ok());
}
