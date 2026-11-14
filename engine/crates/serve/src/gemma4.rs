//! Text serving adapter over the shared BF16 Gemma cartridge.
use spark_engine::gemma4::{Model, Session, PREFILL_CHUNK};
pub struct Gemma<'m, 'a, 'b>(pub &'m Model<'a, 'b>);
impl crate::Session for Session<'_, '_, '_> {
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> {
        if self.has_assistant() {Some(self)} else {None}
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> { self.prefill(&[token], true) }
    fn advance(&mut self, token: u32) -> Result<(), String> { self.prefill(&[token], false).map(|_| ()) }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> { Session::prefill(self, tokens, true) }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<(), String> { Session::prefill(self, tokens, false).map(|_| ()) }
    fn prefill_chunk_size(&self, _: usize) -> usize { PREFILL_CHUNK }
    fn fork(&self) -> Result<Self, String> { Session::fork(self) }
    fn truncate(&mut self, position: usize) -> Result<(), String> { Session::truncate(self, position) }
    fn retained_bytes(&self) -> Option<usize> { Some(self.bytes()) }
    fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(self.allocations()) }
    fn retirement_bytes(&self) -> Option<usize> { Some(self.bytes()) }
    fn memory_bytes(&self) -> Option<(usize, usize)> { Some((self.bytes(), self.peak_bytes())) }
}
impl<'m, 'a, 'b> crate::Engine for Gemma<'m, 'a, 'b> {
    type Session<'s> = Session<'s, 'a, 'b> where Self: 's;
    fn session(&self) -> Result<Self::Session<'_>, String> { Ok(self.0.session()) }
    fn shared_batch(&self) -> bool { true }
    fn shared_prompt_limit(&self) -> usize { 256 }
    fn single_chat_draft(&self) -> bool { true }
    fn step_batch<'s>(&'s self, rows: &mut [crate::BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<crate::Predictions>, crate::BatchError> {
        if rows.len() == 1 || rows.iter().any(|r| r.tokens.len() != 1) {
            return rows.iter_mut().map(|r| {
                let result = match r.output {
                    crate::BatchOutput::All => if self.0.has_assistant() { r.session.verify_batch(r.tokens) }
                        else { r.tokens.iter().map(|&t| r.session.prefill(&[t], true)).collect() },
                    crate::BatchOutput::Last => r.session.prefill(r.tokens, true).map(|v| vec![v]),
                    crate::BatchOutput::None => r.session.prefill(r.tokens, false).map(|_| Vec::new()),
                };
                result.map(crate::Predictions::Logits).map_err(|message| crate::BatchError { request: None, message })
            }).collect();
        }
        let mut requests: Vec<_> = rows.iter_mut().map(|r| (&mut *r.session, r.tokens[0])).collect();
        let output = self.0.step_batch(&mut requests).map_err(|message| crate::BatchError { request: None, message })?;
        Ok(output.into_iter().zip(rows).map(|(logits, r)| crate::Predictions::Logits(
            if matches!(r.output, crate::BatchOutput::None) {Vec::new()} else {vec![logits]})).collect())
    }
    fn context_limit(&self) -> usize { self.0.max_tokens }
    fn vocab_size(&self) -> usize { self.0.config.vocab_size as usize }
    fn eos_token_ids(&self) -> &[u32] { self.0.eos_token_ids() }
    fn request_reservation_bytes(&self, tokens: usize) -> Option<usize> { Model::session_bytes(&self.0.config, self.0.max_tokens, tokens).ok()
        .and_then(|n| n.checked_add(self.0.assistant_reservation_bytes())) }
    fn session_reservation_bytes(&self) -> Option<usize> { self.request_reservation_bytes(self.0.max_tokens) }
    fn memory_bytes(&self) -> Option<(usize, usize, usize)> {
        Some((self.0.backend.allocated_bytes(), self.0.backend.peak_allocated_bytes(), self.0.backend.pinned_host_bytes()))
    }
    fn fatal_error(&self) -> Option<String> { self.0.backend.fatal_error() }
    fn maintain_memory(&self, idle: bool) -> Result<(), String> { self.0.backend.maintain_kv_reserve(idle) }
}
