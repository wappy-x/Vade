//! Grouped prompts use the same ordered command exchange as a shared decode.
use super::*;
use crate::bounded_decoder::{PROMPT_GROUP, PROMPT_TRAIL, PromptChat, PromptResult, prompt_cohort_fits};
use spark_serve::{BatchOutput, BatchRows, MIN_WHOLE_PREFILL_COHORT, Predictions};

// The header carries chat count and total rows. Each payload entry is a u64
// session ID followed by a u32 row count. The tokens follow in that order.
pub(super) const PROMPT_COHORT: u32 = 34;
const ENTRY_BYTES: usize = 12;

impl<C: Compute, E: Exchange> Factory<'_, C, E> {
    /// The whole-group charge at 255 and 256 rows and 4 T, in bytes, for the start line `TEXT PROMPT COHORT BOUND`.
    pub fn prompt_cohort_growth(&self)->Result<(usize,usize,usize)> {
        crate::bounded_decoder::prompt_cohort_growth(self.checkpoint,self.context,self.latent_rows)
    }
    /// Cap the chats in one whole-prompt cohort; 0 turns whole cohorts off. Both ranks agree at startup.
    pub fn with_prompt_cohort_chats(mut self, chats: usize) -> Result<Self> {
        if chats != 0 && !(spark_serve::MIN_WHOLE_PREFILL_COHORT..=LIVE_SESSIONS).contains(&chats) {
            return Err("prompt cohort chat limit is outside the supported range".into());
        }
        self.group_chats = chats;
        Ok(self)
    }

    pub(super) fn can_prefill_cohort(&self, rows: &[BatchRows<'_, Session<'_, C, E>>]) -> Result<bool> {
        if !(MIN_WHOLE_PREFILL_COHORT..=self.group_chats).contains(&rows.len()) || rows.iter().any(|row| {
            !std::ptr::eq(row.session.factory, self) || !matches!(row.output, BatchOutput::Last)
                || row.session.open.is_some() || row.session.state.as_ref().is_none_or(|state| state.position != 0)
                || !(1..=PROMPT_GROUP).contains(&row.tokens.len())
        }) { return Ok(false); }
        for row in rows { row.session.validate_row(row.tokens, row.output)?; }
        let sizes = rows.iter().map(|row| row.tokens.len()).collect::<Vec<_>>();
        prompt_cohort_fits(self.checkpoint, self.context, self.group, self.latent_rows, &sizes)
    }

    pub(super) fn prefill_cohort(&self, rows: &mut [BatchRows<'_, Session<'_, C, E>>]) -> Result<Vec<Predictions>> {
        if let Some(error) = self.failure_reason() { return Err(error); }
        let total = rows.iter().map(|row| row.tokens.len()).sum::<usize>();
        let ids = rows.iter().map(|row| row.session.id).collect::<Vec<_>>();
        let mut bytes = Vec::with_capacity(rows.len() * ENTRY_BYTES + total * 4);
        for row in rows.iter() {
            bytes.extend_from_slice(&row.session.id.to_le_bytes());
            bytes.extend_from_slice(&(row.tokens.len() as u32).to_le_bytes());
        }
        for row in rows.iter() {
            bytes.extend(row.tokens.iter().flat_map(|token| token.to_le_bytes()));
        }
        self.finish(self.split.work(|| {
            let started = std::time::Instant::now();
            self.failed.set(true);
            self.split.command_for(PROMPT_COHORT, rows.len() as u32, total as u64)?;
            self.split.exchange.borrow_mut().command(&mut bytes)?;
            let (results, peak) = {
                let chats = rows.iter().map(|row| Ok(PromptChat {
                    decoder: &row.session.decoder,
                    state: row.session.state.as_ref().ok_or("prompt cohort has no state")?,
                    tokens: row.tokens,
                })).collect::<Result<Vec<_>>>()?;
                self.run_prompt_cohort(&chats, self.active.get())?
            };
            self.split.exchange.borrow_mut().complete()?;
            let seconds = started.elapsed().as_secs_f64();
            let mut output = Vec::with_capacity(rows.len());
            let rank = self.split.topology.rank;
            for (row, result) in rows.iter_mut().zip(results) {
                let id = row.session.id;
                row.session.state = Some(result.state);
                self.retain_cohort_tail(id, row.tokens, result.tail);
                output.push(Predictions::Logits(vec![result.logits]));
                // Apportion elapsed time so summing member prefill records
                // counts the shared interval once, as with shared decode.
                let share = seconds * row.tokens.len() as f64 / total as f64;
                eprintln!("SERVED GROUP rank={rank} session={id} phase=prefill position=0 rows={} seconds={share:.9}", row.tokens.len());
                crate::mxfp8_projection::served_counts(rank, id, "prompt");
            }
            eprintln!("SERVED PROMPT COHORT rank={rank} chats={} rows={total} sessions={ids:?} peak_bytes={peak} seconds={seconds:.9}", ids.len());
            eprintln!("SERVED PROMPT COUNTS rank={rank} session={} {}", ids[0], spark_gpu_probe::prompt_launches::line());
            self.split.cost_report("prefill");
            self.split.prompt_end()?;
            self.failed.set(false);
            Ok(output)
        }))
    }

    pub(super) fn worker_prompt_cohort(&self, sessions: &mut WorkerSessions<'_, '_, C, E>, count: usize, total: u64) -> Result<()> {
        if !(MIN_WHOLE_PREFILL_COHORT..=self.group_chats).contains(&count) || !(count as u64..=self.group as u64).contains(&total) {
            return Err("worker prompt cohort exceeds its bound".into());
        }
        let total = total as usize;
        self.split.work(|| {
            let mut bytes = vec![0; count * ENTRY_BYTES + total * 4];
            self.split.exchange.borrow_mut().command_payload(&mut bytes)?;
            let (entries, tokens) = bytes.split_at(count * ENTRY_BYTES);
            let tokens = tokens.as_chunks::<4>().0.iter().map(|bytes| u32::from_le_bytes(*bytes)).collect::<Vec<_>>();
            let mut rest = tokens.as_slice();
            let mut ids = Vec::with_capacity(count);
            let mut chats = Vec::with_capacity(count);
            for entry in entries.as_chunks::<ENTRY_BYTES>().0 {
                let id = u64::from_le_bytes(entry[..8].try_into().unwrap());
                let rows = u32::from_le_bytes(entry[8..].try_into().unwrap()) as usize;
                if ids.contains(&id) || !(1..=PROMPT_GROUP).contains(&rows) || rows > rest.len() {
                    return Err("worker prompt cohort entry differs".into());
                }
                let (decoder, state, open) = sessions.get(&id).ok_or("worker prompt cohort has no session")?;
                let state = state.as_ref().filter(|state| state.position == 0 && open.is_none())
                    .ok_or("worker prompt cohort has no cold state")?;
                let (tokens, remaining) = rest.split_at(rows);
                rest = remaining;
                ids.push(id);
                chats.push(PromptChat { decoder, state, tokens });
            }
            if !rest.is_empty() { return Err("worker prompt cohort token count differs".into()); }
            let sizes = chats.iter().map(|chat| chat.tokens.len()).collect::<Vec<_>>();
            if !prompt_cohort_fits(self.checkpoint, self.context, self.group, self.latent_rows, &sizes)? {
                return Err("worker prompt cohort exceeds its workspace".into());
            }
            let (results, peak) = self.run_prompt_cohort(&chats, sessions.len())?;
            self.split.exchange.borrow_mut().complete()?;
            let mut at = 0;
            let rank = self.split.topology.rank;
            for ((id, n), result) in ids.iter().copied().zip(sizes).zip(results) {
                sessions.get_mut(&id).ok_or("worker prompt cohort lost its session")?.1 = Some(result.state);
                self.retain_cohort_tail(id, &tokens[at..at + n], result.tail);
                at += n;
                crate::mxfp8_projection::served_counts(rank, id, "prompt");
            }
            eprintln!("SERVED PROMPT COHORT rank={rank} chats={count} rows={total} sessions={ids:?} peak_bytes={peak}");
            self.split.prompt_end()
        })
    }

    #[allow(clippy::type_complexity)]
    fn run_prompt_cohort(&self, chats: &[PromptChat<'_, '_, &Split<C, E>>], live: usize) -> Result<(Vec<PromptResult<C::Value>>, usize)> {
        let kept = if self.keep { self.kept_bytes() } else { 0 };
        let charge = self.reservation.saturating_mul(live).saturating_add(self.resident).saturating_add(self.workspace).saturating_add(kept);
        self.split.rows_charge(charge);
        self.split.rows_follow(false);
        let whole = self.split.rows_peak_begin();
        let seed = self.split.prompt_seeds();
        for chat in chats { chat.decoder.set_memo_seed(seed); }
        let result = Decoder::prefill_cohort(chats);
        for chat in chats { chat.decoder.set_memo_seed(false); }
        let peak = self.split.rows_peak_end(whole);
        eprintln!("PROMPT COHORT PEAK rank={} live={live} peak_bytes={peak} charge_bytes={charge} workspace_bytes={} ok={}",
            self.split.topology.rank, self.workspace, result.is_ok());
        Ok((result?, peak))
    }

    fn retain_cohort_tail(&self, id: u64, tokens: &[u32], tail: Option<crate::bounded_decoder::OpenRows<C::Value>>) {
        if let Some(tail) = tail {
            let first = tokens.len().saturating_sub(PROMPT_TRAIL);
            self.tails.borrow_mut().insert(id, vec![(tokens[first..].to_vec(), tail)]);
        }
    }
}
