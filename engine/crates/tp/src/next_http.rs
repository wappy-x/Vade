//! Opt-in rank HTTP adapter. Command order owns every rank's independent rows.
use crate::{
    Result,
    next::{NextSession, RankKernels, RankTransport},
};
use spark_serve::{BatchError, BatchOutput, BatchRows, Predictions};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
};
pub const CONTEXT: usize = 32768;
pub const SLOTS: usize = 8;
const OPEN: u64 = 1;
const STEP: u64 = 2;
const CLOSE: u64 = 3;
const STOP: u64 = 4;
const ROWS: u64 = 5;
const AB_ARM: u64 = 6;
/// A prompt row without a prediction (scheduler output None): no head.
const ADVANCE: u64 = 7;
/// Up to CHUNK_MAX consecutive prompt tokens of one session (u32 payload after the header);
/// arg = count | head << 16 (head: the last row's prediction is returned).
const CHUNK_OP: u64 = 8;
pub const CHUNK: usize = 128;
/// Longest chunk a rank can hold: the CHUNK command's payload bound.
pub const CHUNK_MAX: usize = 256;
/// Prompt rows per CHUNK command.
pub fn prompt_chunk() -> usize { CHUNK }
pub trait Causal {
    fn token(&mut self, token: u32) -> Result<u32>;
    fn advance(&mut self, token: u32) -> Result<()> {
        self.token(token).map(|_| ())
    }
    fn rows(sessions: &mut [&mut Self], tokens: &[u32]) -> Result<Vec<u32>>
    where
        Self: Sized,
    {
        sessions
            .iter_mut()
            .zip(tokens)
            .map(|(s, &t)| s.token(t))
            .collect()
    }
    /// Consecutive prompt tokens; `head`: return the last row's prediction (else 0).
    fn chunk(&mut self, tokens: &[u32], head: bool) -> Result<u32> {
        let (&last, prefix) = tokens.split_last().ok_or("empty Next chunk")?;
        for &t in prefix {
            self.advance(t)?;
        }
        if head { self.token(last) } else { self.advance(last).map(|_| 0) }
    }
}
impl<S: Causal> Causal for Box<S> {
    fn token(&mut self, t: u32) -> Result<u32> {
        (**self).token(t)
    }
    fn advance(&mut self, t: u32) -> Result<()> {
        (**self).advance(t)
    }
    fn rows(s: &mut [&mut Self], t: &[u32]) -> Result<Vec<u32>> {
        S::rows(&mut s.iter_mut().map(|s| &mut ***s).collect::<Vec<_>>(), t)
    }
    fn chunk(&mut self, t: &[u32], head: bool) -> Result<u32> {
        (**self).chunk(t, head)
    }
}
impl<T: RankTransport, K: RankKernels> Causal for NextSession<'_, T, K> {
    fn token(&mut self, token: u32) -> Result<u32> {
        self.step(token)
    }
    fn advance(&mut self, token: u32) -> Result<()> {
        NextSession::advance(self, token)
    }
    fn rows(s: &mut [&mut Self], t: &[u32]) -> Result<Vec<u32>> {
        Self::step_rows(s, t)
    }
    fn chunk(&mut self, t: &[u32], head: bool) -> Result<u32> {
        NextSession::chunk(self, t, head)
    }
}
type Factory<'a, S> = Box<dyn Fn(u64) -> Result<S> + 'a>;
pub struct Runtime<'a, S> {
    arm: Option<Box<dyn Fn(bool) -> Result<()> + 'a>>,
    make: Factory<'a, S>,
    sessions: BTreeMap<u64, (S, usize)>,
    pub vocab: usize,
}
impl<'a, S: Causal> Runtime<'a, S> {
    pub fn new(vocab: usize, make: Factory<'a, S>) -> Self {
        Self {
            make,
            arm: None,
            sessions: BTreeMap::new(),
            vocab,
        }
    }
    pub fn with_arm_selector(mut self, arm: Box<dyn Fn(bool) -> Result<()> + 'a>) -> Self {
        self.arm = Some(arm); self
    }
    fn batch(&mut self, items: &[(u64, u32)]) -> Result<Vec<u32>> {
        let mut seen = BTreeSet::new();
        if items.is_empty()
            || items.len() > SLOTS
            || items
                .iter()
                .any(|&(id, t)| !seen.insert(id) || self.validate(STEP, id, t as u64).is_err())
        {
            return Err("invalid Next rows command".into());
        }
        let mut live = items
            .iter()
            .map(|&(id, _)| (id, self.sessions.remove(&id).unwrap()))
            .collect::<Vec<_>>();
        let result = S::rows(
            &mut live.iter_mut().map(|(_, s)| &mut s.0).collect::<Vec<_>>(),
            &items.iter().map(|&(_, t)| t).collect::<Vec<_>>(),
        );
        let result = result.and_then(|ids| {
            if ids.len() == items.len() && ids.iter().all(|&t| (t as usize) < self.vocab) {
                Ok(ids)
            } else {
                Err("invalid Next row predictions".into())
            }
        });
        for (id, (s, pos)) in live {
            self.sessions
                .insert(id, (s, pos + usize::from(result.is_ok())));
        }
        result
    }
    fn chunk(&mut self, id: u64, tokens: &[u32], head: bool) -> Result<u32> {
        if tokens.is_empty()
            || tokens.len() > CHUNK_MAX
            || tokens.iter().any(|&t| t as usize >= self.vocab)
            || self
                .sessions
                .get(&id)
                .is_none_or(|s| s.1 + tokens.len() > CONTEXT)
        {
            return Err("invalid_request_error: Next chunk outside admission limits".into());
        }
        let s = self.sessions.get_mut(&id).unwrap();
        let t = s.0.chunk(tokens, head)?;
        if t as usize >= self.vocab {
            return Err("Next returned an invalid token".into());
        }
        s.1 += tokens.len();
        Ok(t)
    }
    fn validate(&self, op: u64, id: u64, arg: u64) -> Result<()> {
        let valid = match op {
            OPEN => {
                id != 0
                    && arg == 0
                    && !self.sessions.contains_key(&id)
                    && self.sessions.len() < SLOTS
            }
            STEP | ADVANCE => {
                arg < self.vocab as u64 && self.sessions.get(&id).is_some_and(|s| s.1 < CONTEXT)
            }
            CLOSE => arg == 0 && self.sessions.contains_key(&id),
            AB_ARM => self.arm.is_some() && id == 0 && arg <= 1 && self.sessions.is_empty(),
            STOP => id == 0 && arg == 0 && self.sessions.is_empty(),
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err("invalid_request_error: Next command outside admission limits".into())
        }
    }
    pub fn execute(&mut self, op: u64, id: u64, arg: u64) -> Result<u32> {
        self.validate(op, id, arg)?;
        match op {
            AB_ARM => { self.arm.as_ref().ok_or("A/B module unavailable")?(arg == 1)?; Ok(0) }
            OPEN => {
                let s = (self.make)(id)?;
                self.sessions.insert(id, (s, 0));
                Ok(0)
            }
            STEP => {
                let s = self.sessions.get_mut(&id).unwrap();
                let t = s.0.token(arg as u32)?;
                if t as usize >= self.vocab {
                    return Err("Next returned an invalid token".into());
                }
                s.1 += 1;
                Ok(t)
            }
            ADVANCE => {
                let s = self.sessions.get_mut(&id).unwrap();
                s.0.advance(arg as u32)?;
                s.1 += 1;
                Ok(0)
            }
            CLOSE => {
                self.sessions.remove(&id);
                Ok(0)
            }
            _ => Ok(0),
        }
    }
}
/// Fixed three-word headers, no untrusted allocation lengths or token payloads.
pub trait Commands {
    fn broadcast(&mut self, header: &mut [u8]) -> Result<()>;
    fn verdict(&mut self, ok: bool) -> Result<()>;
}
impl Commands for crate::transport::Ring {
    fn broadcast(&mut self, h: &mut [u8]) -> Result<()> {
        self.command(h)
    }
    fn verdict(&mut self, ok: bool) -> Result<()> {
        let mut v = vec![0; self.world()];
        v[self.rank()] = u8::from(ok);
        self.all_gather(&mut v, 1)?;
        if v.iter().all(|&x| x == 1) {
            Ok(())
        } else {
            self.abort();
            Err("Next rank command failed".into())
        }
    }
}
/// The verdict carries one bit per rank; keep this rank's own reason in its log.
fn local_error<T>(op: u64, result: &Result<T>) {
    if let Err(error) = result {
        eprintln!("next_rank_local_error op={op} error={error}");
    }
}
fn header(op: u64, id: u64, arg: u64) -> [u8; 24] {
    let mut h = [0; 24];
    for (b, w) in h.as_chunks_mut::<8>().0.iter_mut().zip([op, id, arg]) {
        b.copy_from_slice(&w.to_le_bytes());
    }
    h
}
fn decode_chunk(payload: &[u8; CHUNK_MAX * 4], arg: u64) -> Result<(Vec<u32>, bool)> {
    let (n, head) = ((arg & 0xffff) as usize, arg >> 16);
    if n == 0 || n > CHUNK_MAX || head > 1 {
        return Err("invalid Next chunk length".into());
    }
    let tokens = payload.as_chunks::<4>().0.iter().take(n).map(|b| u32::from_le_bytes(*b)).collect();
    Ok((tokens, head == 1))
}
fn decode_rows(payload: &[u8; SLOTS * 16], n: u64) -> Result<Vec<(u64, u32)>> {
    if n == 0 || n > SLOTS as u64 {
        return Err("invalid Next rows length".into());
    }
    payload
        .as_chunks::<16>()
        .0
        .iter()
        .take(n as usize)
        .map(|b| {
            Ok((
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u32::try_from(u64::from_le_bytes(b[8..].try_into().unwrap()))
                    .map_err(|_| "invalid Next rows token")?,
            ))
        })
        .collect()
}
pub fn worker<S: Causal>(mut commands: impl Commands, mut runtime: Runtime<'_, S>) -> Result<()> {
    loop {
        let mut h = [0; 24];
        commands.broadcast(&mut h)?;
        let w = h
            .as_chunks::<8>().0.iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let result = if w[0] == ROWS {
            let mut payload = [0; SLOTS * 16];
            commands.broadcast(&mut payload)?;
            decode_rows(&payload, w[2])
                .and_then(|items| runtime.batch(&items))
                .map(|_| 0)
        } else if w[0] == CHUNK_OP {
            let mut payload = [0; CHUNK_MAX * 4];
            commands.broadcast(&mut payload)?;
            decode_chunk(&payload, w[2]).and_then(|(tokens, head)| runtime.chunk(w[1], &tokens, head))
        } else {
            runtime.execute(w[0], w[1], w[2])
        };
        local_error(w[0], &result);
        // The broadcast wait above includes idle time between requests.
        let timer = crate::next_timing::start();
        commands.verdict(result.is_ok())?;
        crate::next_timing::command(timer);
        result?;
        if w[0] == STOP {
            return Ok(());
        }
    }
}
pub struct Engine<'a, C, S> {
    commands: RefCell<C>,
    runtime: RefCell<Runtime<'a, S>>,
    next: Cell<u64>,
    failure: RefCell<Option<String>>,
    eos: Vec<u32>,
    reservation: usize,
}
impl<'a, C: Commands, S: Causal> Engine<'a, C, S> {
    pub fn new(commands: C, runtime: Runtime<'a, S>, eos: Vec<u32>, reservation: usize) -> Self {
        Self {
            commands: RefCell::new(commands),
            runtime: RefCell::new(runtime),
            next: Cell::new(1),
            failure: RefCell::new(None),
            eos,
            reservation,
        }
    }
    fn issue(&self, op: u64, id: u64, arg: u64) -> Result<u32> {
        if let Some(e) = self.failure.borrow().clone() {
            return Err(e);
        }
        self.runtime.borrow().validate(op, id, arg)?;
        let result = (|| {
            let mut c = self.commands.borrow_mut();
            let timer = crate::next_timing::start();
            c.broadcast(&mut header(op, id, arg))?;
            crate::next_timing::command(timer);
            let local = self.runtime.borrow_mut().execute(op, id, arg);
            local_error(op, &local);
            let timer = crate::next_timing::start();
            c.verdict(local.is_ok())?;
            crate::next_timing::command(timer);
            local
        })();
        if let Err(e) = &result {
            *self.failure.borrow_mut() = Some(e.clone());
        }
        result
    }
    fn issue_rows(&self, items: &[(u64, u32)]) -> Result<Vec<u32>> {
        if let Some(e) = self.failure.borrow().clone() {
            return Err(e);
        }
        let result = (|| {
            let mut c = self.commands.borrow_mut();
            let timer = crate::next_timing::start();
            c.broadcast(&mut header(ROWS, 0, items.len() as u64))?;
            let mut payload = [0; SLOTS * 16];
            for (b, &(id, t)) in payload.as_chunks_mut::<16>().0.iter_mut().zip(items) {
                b[..8].copy_from_slice(&id.to_le_bytes());
                b[8..].copy_from_slice(&(t as u64).to_le_bytes());
            }
            c.broadcast(&mut payload)?;
            crate::next_timing::command(timer);
            let local = self.runtime.borrow_mut().batch(items);
            local_error(ROWS, &local);
            let timer = crate::next_timing::start();
            c.verdict(local.is_ok())?;
            crate::next_timing::command(timer);
            local
        })();
        if let Err(e) = &result {
            *self.failure.borrow_mut() = Some(e.clone());
        }
        result
    }
    fn issue_chunk(&self, id: u64, tokens: &[u32], head: bool) -> Result<u32> {
        if let Some(e) = self.failure.borrow().clone() {
            return Err(e);
        }
        if tokens.is_empty() || tokens.len() > CHUNK_MAX {
            return Err("invalid Next chunk".into());
        }
        let result = (|| {
            let mut c = self.commands.borrow_mut();
            let timer = crate::next_timing::start();
            c.broadcast(&mut header(CHUNK_OP, id, tokens.len() as u64 | u64::from(head) << 16))?;
            let mut payload = [0; CHUNK_MAX * 4];
            for (b, &t) in payload.as_chunks_mut::<4>().0.iter_mut().zip(tokens) {
                b.copy_from_slice(&t.to_le_bytes());
            }
            c.broadcast(&mut payload)?;
            crate::next_timing::command(timer);
            let local = self.runtime.borrow_mut().chunk(id, tokens, head);
            local_error(CHUNK_OP, &local);
            let timer = crate::next_timing::start();
            c.verdict(local.is_ok())?;
            crate::next_timing::command(timer);
            local
        })();
        if let Err(e) = &result {
            *self.failure.borrow_mut() = Some(e.clone());
        }
        result
    }
    pub fn shutdown(&self) -> Result<()> {
        self.issue(STOP, 0, 0).map(|_| ())
    }
}
pub struct Remote<'e, 'a, C: Commands, S: Causal> {
    engine: &'e Engine<'a, C, S>,
    id: u64,
    position: usize,
}
impl<C: Commands, S: Causal> Drop for Remote<'_, '_, C, S> {
    fn drop(&mut self) {
        let _ = self.engine.issue(CLOSE, self.id, 0);
    }
}
impl<C: Commands, S: Causal> spark_serve::Session for Remote<'_, '_, C, S> {
    fn step(&mut self, t: u32) -> Result<Vec<f32>> {
        let id = self.engine.issue(STEP, self.id, t as u64)?;
        self.position += 1;
        // Selection-only row: request validation admits penalty-free greedy only.
        let mut logits = vec![f32::MIN; self.engine.runtime.borrow().vocab];
        logits[id as usize] = 0.;
        Ok(logits)
    }
    fn advance(&mut self, t: u32) -> Result<()> {
        self.engine.issue(ADVANCE, self.id, t as u64)?;
        self.position += 1;
        Ok(())
    }
    /// Prompt rows travel as CHUNK commands; only the last row of the final one has a head.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let parts = tokens.chunks(prompt_chunk()).count();
        let mut id = 0;
        for (i, part) in tokens.chunks(prompt_chunk()).enumerate() {
            id = self.engine.issue_chunk(self.id, part, i + 1 == parts)?;
            self.position += part.len();
        }
        if parts == 0 {
            return Err("empty prefill".into());
        }
        let mut logits = vec![f32::MIN; self.engine.runtime.borrow().vocab];
        logits[id as usize] = 0.;
        Ok(logits)
    }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<()> {
        for part in tokens.chunks(prompt_chunk()) {
            self.engine.issue_chunk(self.id, part, false)?;
            self.position += part.len();
        }
        Ok(())
    }
    fn prefill_chunk_size(&self, _: usize) -> usize {
        prompt_chunk()
    }
    fn can_fork(&self) -> bool {
        false
    }
}
impl<'a, C: Commands, S: Causal> spark_serve::Engine for Engine<'a, C, S> {
    type Session<'s>
        = Remote<'s, 'a, C, S>
    where
        Self: 's;
    fn session_for_request(&self, request: &spark_serve::protocol::Request) -> Result<Self::Session<'_>> {
        if self.runtime.borrow().arm.is_some() { self.issue(AB_ARM, 0, u64::from(request.arm_x))?; }
        self.session()
    }
    fn session(&self) -> Result<Self::Session<'_>> {
        let id = self.next.get();
        let next = id.checked_add(1).ok_or("Next session ID exhausted")?;
        self.issue(OPEN, id, 0)?;
        self.next.set(next);
        Ok(Remote {
            engine: self,
            id,
            position: 0,
        })
    }
    fn context_limit(&self) -> usize {
        CONTEXT
    }
    fn vocab_size(&self) -> usize {
        self.runtime.borrow().vocab
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    fn session_reservation_bytes(&self) -> Option<usize> {
        Some(self.reservation)
    }
    fn fatal_error(&self) -> Option<String> {
        self.failure.borrow().clone()
    }
    fn shared_batch(&self) -> bool {
        true
    }
    fn shared_prompt_limit(&self) -> usize {
        0
    }
    fn validate_request(
        &self,
        prompt: &[u32],
        request: &spark_serve::protocol::Request,
    ) -> Result<()> {
        if request.mtp_depth.is_some_and(|depth| depth > 0) {
            return Err("invalid_request_error: FP4-Local world 2/4 does not support mtp_depth > 0 yet; use mtp_depth: 0".into());
        }
        let s = request.sampling;
        s.validate()?;
        if prompt.is_empty()
            || prompt.iter().any(|&t| t as usize >= self.vocab_size())
            || request.max_tokens > 16384
            || prompt
                .len()
                .checked_add(request.max_tokens)
                .is_none_or(|n| n > CONTEXT)
            || request
                .draft
                .is_some_and(|d| d != spark_serve::protocol::DraftMode::Plain)
            || s.temperature != 0.
            || s.repetition_penalty != 1.
            || s.presence_penalty != 0.
            || s.frequency_penalty != 0.
        {
            return Err(
                "invalid_request_error: Next supports bounded penalty-free greedy requests only"
                    .into(),
            );
        }
        Ok(())
    }
    fn step_batch<'s>(
        &'s self,
        rows: &mut [BatchRows<'_, Self::Session<'s>>],
    ) -> std::result::Result<Vec<Predictions>, BatchError> {
        // Rows-lane seam: validate the WHOLE cohort before advancing causal state.
        let mut seen = BTreeSet::new();
        if rows.is_empty()
            || rows.len() > SLOTS
            || rows.iter().any(|r| {
                !std::ptr::eq(r.session.engine, self)
                    || !seen.insert(r.session.id)
                    || !r.greedy
                    || r.tokens.len() != 1
                    || r.session
                        .position
                        .checked_add(r.tokens.len())
                        .is_none_or(|n| n > CONTEXT)
                    || r.tokens.iter().any(|&t| t as usize >= self.vocab_size())
            })
        {
            return Err(BatchError {
                request: None,
                message: "invalid Next M1 cohort".into(),
            });
        }
        if rows.len() >= 2 {
            let items = rows
                .iter()
                .map(|r| (r.session.id, r.tokens[0]))
                .collect::<Vec<_>>();
            let ids = self.issue_rows(&items).map_err(|message| BatchError {
                request: None,
                message,
            })?;
            return Ok(rows
                .iter_mut()
                .zip(ids)
                .map(|(r, t)| {
                    r.session.position += 1;
                    Predictions::Greedy(if matches!(r.output, BatchOutput::None) {
                        vec![]
                    } else {
                        vec![t]
                    })
                })
                .collect());
        }
        rows.iter_mut()
            .map(|r| {
                let mut ids = Vec::new();
                // Output None: a prompt row nobody reads, so skip the head.
                let op = if matches!(r.output, BatchOutput::None) { ADVANCE } else { STEP };
                for &t in r.tokens {
                    let id = self
                        .issue(op, r.session.id, t as u64)
                        .map_err(|message| BatchError {
                            request: None,
                            message,
                        })?;
                    r.session.position += 1;
                    ids.push(id);
                }
                Ok(Predictions::Greedy(match r.output {
                    BatchOutput::All => ids,
                    BatchOutput::Last => vec![*ids.last().unwrap()],
                    BatchOutput::None => vec![],
                }))
            })
            .collect()
    }
}
pub fn schedule(reservation: usize) -> spark_serve::scheduler::Config {
    spark_serve::scheduler::Config {
        active: SLOTS,
        connections: 64,
        prefill_chunk: prompt_chunk(),
        active_bytes: reservation * SLOTS,
        session_budget_bytes: reservation * SLOTS,
        draft: spark_serve::protocol::DraftMode::Plain,
        mtp_depth: 0,
        draft_global_bank: false,
        prefix_cache: false,
        sharing: Default::default(),
        foreground_rows: SLOTS,
        background_rows: SLOTS,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use spark_serve::{Engine as _, Session as _};
    struct Fake;
    impl Causal for Fake {
        fn token(&mut self, t: u32) -> Result<u32> {
            Ok((t + 1) % 248320)
        }
    }
    #[test]
    fn ab_selection_requires_no_live_session() {
        let selected = Cell::new(false);
        let mut runtime = Runtime::new(248320, Box::new(|_| Ok(Fake)))
            .with_arm_selector(Box::new(|x| { selected.set(x); Ok(()) }));
        runtime.execute(AB_ARM, 0, 1).unwrap();
        assert!(selected.get());
        runtime.execute(OPEN, 1, 0).unwrap();
        assert!(runtime.execute(AB_ARM, 0, 0).is_err());
        assert!(selected.get());
        runtime.execute(CLOSE, 1, 0).unwrap();
        runtime.execute(AB_ARM, 0, 0).unwrap();
        assert!(!selected.get());
    }
    #[test]
    fn advance_moves_the_session_like_step_without_a_prediction() {
        let mut runtime = Runtime::new(248320, Box::new(|_| Ok(Fake)));
        runtime.execute(OPEN, 1, 0).unwrap();
        assert_eq!(runtime.execute(ADVANCE, 1, 7).unwrap(), 0);
        assert_eq!(runtime.execute(STEP, 1, 7).unwrap(), 8);
        assert_eq!(runtime.sessions[&1].1, 2);
        assert!(runtime.execute(ADVANCE, 2, 7).is_err());
        assert!(runtime.execute(ADVANCE, 1, 248320).is_err());
    }
    #[test]
    fn chunk_commands_move_the_session_by_their_rows() {
        let mut runtime = Runtime::new(248320, Box::new(|_| Ok(Fake)));
        runtime.execute(OPEN, 1, 0).unwrap();
        assert_eq!(runtime.chunk(1, &[5, 6], false).unwrap(), 0);
        assert_eq!(runtime.chunk(1, &[7; CHUNK], true).unwrap(), 8);
        assert_eq!(runtime.sessions[&1].1, 2 + CHUNK);
        assert_eq!(runtime.chunk(1, &[7; CHUNK_MAX], false).unwrap(), 0);
        assert_eq!(runtime.sessions[&1].1, 2 + CHUNK + CHUNK_MAX);
        assert!(runtime.chunk(1, &[7; CHUNK_MAX + 1], false).is_err());
        assert!(runtime.chunk(1, &[], false).is_err());
        assert!(runtime.chunk(2, &[7], false).is_err());
        assert!(runtime.chunk(1, &[248320], false).is_err());
        let mut payload = [0; CHUNK_MAX * 4];
        for (b, t) in payload.as_chunks_mut::<4>().0.iter_mut().zip([9u32, 10, 11]) {
            b.copy_from_slice(&t.to_le_bytes());
        }
        assert_eq!(decode_chunk(&payload, 3 | 1 << 16).unwrap(), (vec![9, 10, 11], true));
        assert_eq!(decode_chunk(&payload, 2).unwrap(), (vec![9, 10], false));
        assert!(decode_chunk(&payload, 0).is_err());
        assert_eq!(decode_chunk(&payload, CHUNK_MAX as u64).unwrap().0.len(), CHUNK_MAX);
        assert!(decode_chunk(&payload, (CHUNK_MAX + 1) as u64).is_err());
        assert!(decode_chunk(&payload, 3 | 2 << 16).is_err());
    }
    #[test]
    fn real_shape_admission_cancel_reuse() {
        for (world, path) in [(2, false), (4, false), (4, true)] {
            for width in [2, SLOTS] {
                let d = engine_model::flash::refcpu::Dims::real();
                assert_eq!(
                    (d.hidden, d.vocab, d.qh / world, d.vh / world),
                    (2560, 248320, 24 / world, 48 / world)
                );
                let listeners = (0..world)
                    .map(|_| {
                        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap()
                    })
                    .collect::<Vec<_>>();
                let addresses = listeners
                    .iter()
                    .map(|l| l.local_addr().unwrap())
                    .collect::<Vec<_>>();
                drop(listeners);
                std::thread::scope(|scope| {
                    let jobs = (1..world)
                        .map(|rank| {
                            let addresses = &addresses;
                            scope.spawn(move || {
                                worker(
                                    crate::transport::Ring::connect_options(
                                        rank,
                                        world,
                                        addresses[rank],
                                        addresses[(rank + 1) % world],
                                        path,
                                    )
                                    .unwrap(),
                                    Runtime::new(248320, Box::new(|_| Ok(Box::new(Fake)))),
                                )
                                .unwrap()
                            })
                        })
                        .collect::<Vec<_>>();
                    let e = Engine::new(
                        crate::transport::Ring::connect_options(
                            0,
                            world,
                            addresses[0],
                            addresses[1],
                            path,
                        )
                        .unwrap(),
                        Runtime::new(248320, Box::new(|_| Ok(Box::new(Fake)))),
                        vec![248046],
                        1 << 30,
                    );
                    let mut live = (0..SLOTS).map(|_| e.session().unwrap()).collect::<Vec<_>>();
                    assert!(e.session().is_err());
                    let mut cohort = live
                        .iter_mut()
                        .take(width)
                        .map(|session| BatchRows {
                            session,
                            tokens: &[3],
                            output: BatchOutput::Last,
                            greedy: true,
                            top_k: 0,
                        })
                        .collect::<Vec<_>>();
                    assert!(
                        e.step_batch(&mut cohort)
                            .unwrap_or_else(|e| panic!("{}", e.message))
                            .into_iter()
                            .all(|p| matches!(p,Predictions::Greedy(t) if t==vec![4]))
                    );
                    let selection = live[0].step(7).unwrap();
                    assert_eq!(selection[8], 0.);
                    assert!(selection.iter().all(|v| v.is_finite()));
                    drop(live.remove(3));
                    let mut reused = e.session().unwrap();
                    assert_eq!(reused.position, 0);
                    assert_eq!(reused.step(9).unwrap()[10], 0.);
                    let r = spark_serve::protocol::parse(
                        br#"{"prompt":"x","max_tokens":16384,"temperature":0}"#,
                        false,
                        &spark_serve::Limits {
                            output_tokens: 16384,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert!(e.validate_request(&[0; 16384], &r).is_ok());
                    assert!(e.validate_request(&[0; 16385], &r).is_err());
                    let mut unsupported = r.clone();
                    for depth in 1..=3 {
                        unsupported.mtp_depth = Some(depth);
                        assert!(
                            e.validate_request(&[0], &unsupported)
                                .unwrap_err()
                                .contains("mtp_depth > 0")
                        );
                    }
                    unsupported.mtp_depth = Some(0);
                    assert!(e.validate_request(&[0], &unsupported).is_ok());
                    drop(reused);
                    drop(live);
                    assert!(e.runtime.borrow().sessions.is_empty());
                    e.shutdown().unwrap();
                    for j in jobs {
                        j.join().unwrap();
                    }
                });
            }
        }
    }
}
