use super::*;
use spark_engine::{
    draft::{Config, ModelProposal, Speculator, Target},
    flash_next_lookup::Lookup,
};

struct Oracle {
    history: Vec<u32>,
    states: Vec<u64>,
    lookup: Lookup,
}
impl Target for Oracle {
    fn checkpoint(&self) -> usize {
        self.history.len()
    }
    fn truncate(&mut self, p: usize) -> Result<(), String> {
        self.history.truncate(p);
        self.states.truncate(p);
        Ok(())
    }
    fn step(&mut self, t: u32) -> Result<Vec<f32>, String> {
        let state = self.states.last().copied().unwrap_or(0).wrapping_mul(31) ^ u64::from(t);
        self.states.push(state);
        self.history.push(t);
        let mut logits = vec![-1.; 256];
        logits[CODE[self.history.len() % CODE.len()] as usize] = 1.;
        Ok(logits)
    }
    fn native_draft_policy(&self) -> bool {
        true
    }
    fn model_draft_width(&self) -> usize {
        2
    }
    fn propose_model(
        &mut self,
        h: &[u32],
        next: u32,
        n: usize,
    ) -> Result<Option<ModelProposal>, String> {
        let mut tokens = self.lookup.propose(h, next, n);
        if tokens.is_empty() {
            tokens = vec![0; n.min(2)];
        }
        Ok(Some(ModelProposal {
            tokens,
            proposal_ms: 0.,
            catchup_ms: 0.,
        }))
    }
}
const CODE: &[u8] = b"fn edit(input: &[u32]) -> Vec<u32> { input.to_vec() }\nSTOP\n";

#[test]
fn lookup_code_edit_stop_string_equals_plain_and_state_replay() {
    let prompt: Vec<u32> = CODE
        .iter()
        .cycle()
        .take(CODE.len() * 2)
        .map(|&b| u32::from(b))
        .collect();
    let mut target = Oracle {
        history: Vec::new(),
        states: Vec::new(),
        lookup: Lookup::default(),
    };
    for &t in &prompt {
        target.step(t).unwrap();
    }
    let mut spec = Speculator::from_selected(
        &prompt,
        prompt.len(),
        u32::from(CODE[0]),
        256,
        Config::strict(256),
        spark_engine::sample::Sampling::greedy(),
    )
    .unwrap();
    let stops = vec!["\nSTOP".into()];
    let (mut plain, mut lookup) = (StopBuffer::default(), StopBuffer::default());
    let mut expected = String::new();
    for &b in CODE.iter().cycle().take(256) {
        expected.push_str(&plain.push(&(b as char).to_string(), &stops));
    }
    let mut got = String::new();
    while !spec.is_done() {
        let ids = spec.step(&mut target, 16, |_| false).unwrap();
        let text: String = ids.iter().map(|&t| char::from(t as u8)).collect();
        got.push_str(&lookup.push(&text, &stops));
        let replay = target
            .history
            .iter()
            .fold(0u64, |h, &t| h.wrapping_mul(31) ^ u64::from(t));
        assert_eq!(target.states.last(), Some(&replay));
    }
    assert_eq!(got.as_bytes(), expected.as_bytes());
    assert!(lookup.matched && plain.matched);
    assert_eq!(spec.stats.max_rows, 16);
}

#[test]
fn lookup_is_derived_without_a_new_request_switch() {
    let limits = crate::Limits::default();
    assert!(
        !parse(br#"{"prompt":"hi"}"#, false, &limits)
            .unwrap()
            .prompt_lookup
    );
    let limits = crate::Limits {
        next_mtp_depth: Some(3),
        ..limits
    };
    for extra in [
        "",
        ",\"prompt_lookup\":false",
        ",\"prompt_lookup_max_draft\":0",
        ",\"prompt_lookup_max_draft\":999",
    ] {
        let request = parse(
            format!("{{\"prompt\":\"hi\"{extra}}}").as_bytes(),
            false,
            &limits,
        )
        .unwrap();
        assert!(request.prompt_lookup);
        assert_eq!(request.prompt_lookup_max_draft, 15);
    }
    for extra in [
        ",\"mtp_depth\":0",
        ",\"draft\":\"plain\"",
        ",\"temperature\":1",
    ] {
        assert!(
            !parse(
                format!("{{\"prompt\":\"hi\"{extra}}}").as_bytes(),
                false,
                &limits
            )
            .unwrap()
            .prompt_lookup
        );
    }
}
