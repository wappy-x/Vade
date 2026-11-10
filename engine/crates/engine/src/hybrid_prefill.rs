//! Shared dense projections for independent prompts. Recurrence, convolution
//! and causal attention keep each session's boundaries and arithmetic.
use super::*;

pub struct Row<'s, 'm, 'a, 'b> {
    pub session: &'s mut Session<'m, 'a, 'b>,
    pub tokens: &'s [u32],
    pub logits: bool,
}
pub struct Prefill<'m, 'a, 'b> {
    scratch: Session<'m, 'a, 'b>,
    final_hidden: DeviceVector<'a>,
    final_normalized: DeviceVector<'a>,
    logits: DeviceVector<'a>,
    argmax: DeviceVector<'a>, selected: DeviceVector<'a>,
    gdn_rows: DeviceVector<'a>,
    rejected: Vec<usize>,
}
impl<'m, 'a, 'b> Prefill<'m, 'a, 'b> {
    pub fn new(model: &'m Model<'a, 'b>) -> Result<Self> {
        let c = &model.config;
        let mut v = model.backend.vector_arena(&[decode::MAX_ROWS * c.hidden as usize,
            decode::MAX_ROWS * c.hidden as usize, decode::MAX_ROWS * c.vocab as usize,
            decode::MAX_ROWS * (c.vocab as usize).div_ceil(1024) * 3, decode::MAX_ROWS * 3])?.into_iter();
        Ok(Self { scratch: Session::new(model, false)?, final_hidden: v.next().unwrap(),
            final_normalized: v.next().unwrap(), logits: v.next().unwrap(),
            argmax: v.next().unwrap(), selected: v.next().unwrap(),
            gdn_rows: model.backend.vector_arena(&[c.layers.len() * decode::MAX_ROWS * 30])?.pop().unwrap(), rejected: Vec::new() })
    }
    /// Prompts of the last run whose router rows were rejected. They failed
    /// alone: no output (greedy: u32::MAX), uncommitted and invalid.
    pub fn rejected(&self) -> &[usize] { &self.rejected }
    // Continuations replay the canonical per-session tail. Pooling their
    // suffixes can select different projection reductions than a cold prompt.
    pub fn run(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<Vec<Vec<f32>>> {
        if self.scratch.model.bound.is_some() || rows.iter().any(|r| r.session.position != 0 || !r.session.prefill_tail.is_empty() || !r.session.verify_tokens.is_empty()
            || r.tokens.len() >= 256 || r.tokens.len() == 1) {
            return self.serial(rows);
        }
        self.forward(rows)?;
        let (b, v) = (self.scratch.model.backend, self.scratch.model.config.vocab as usize);
        let values = if rows.iter().any(|r| r.logits) { b.read_vector_values(&self.logits)? } else { b.synchronize()?; Vec::new() };
        let rejected = self.routing(rows)?;
        let output = rows.iter().enumerate().map(|(i, row)| Ok(if rejected[i] || values.is_empty() { Vec::new() } else {
            let logits = decode::Decode::checked(&values, i, v)?;
            if row.logits { logits.to_vec() } else { Vec::new() }
        })).collect::<Result<_>>()?;
        Self::commit(rows, &rejected);
        Ok(output)
    }
    /// Sessions run one at a time: a router rejection fails only its own.
    fn serial(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<Vec<Vec<f32>>> {
        self.validate(rows)?;
        self.rejected.clear();
        rows.iter_mut().enumerate().map(|(i, r)| {
            let result = if r.logits { r.session.prefill(r.tokens) } else { r.session.prefill_advance(r.tokens).map(|_| Vec::new()) };
            match result {
                Err(error) if error == MOE_ROUTING_ERROR => { self.rejected.push(i); Ok(Vec::new()) }
                result => result,
            }
        }).collect()
    }
    pub fn run_greedy(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<Vec<u32>> {
        if rows.iter().any(|r| !r.logits) { return Err("greedy prefill requires an output for each row".into()); }
        if self.scratch.model.bound.is_some() || rows.iter().any(|r| r.session.position != 0 || !r.session.prefill_tail.is_empty() || !r.session.verify_tokens.is_empty()
            || r.tokens.len() >= 256 || r.tokens.len() == 1) {
            let output = self.serial(rows)?;
            return output.iter().enumerate().map(|(i, v)| if self.rejected.contains(&i) { Ok(u32::MAX) } else { crate::greedy_token(v) }).collect();
        }
        self.forward(rows)?;
        self.argmax.resize_within_capacity(rows.len() * (self.scratch.model.config.vocab as usize).div_ceil(1024) * 3)?;
        self.selected.resize_within_capacity(rows.len() * 3)?;
        let ids = self.scratch.model.backend.argmax_batch_device(&self.logits, &mut self.argmax, &mut self.selected, rows.len())?;
        let rejected = self.routing(rows)?;
        let output = ids.into_iter().zip(&rejected).map(|(id, &rejected)| if rejected { Ok(u32::MAX) } else { id.map(|(id, _)| id) })
            .collect::<Result<_>>()?;
        Self::commit(rows, &rejected);
        Ok(output)
    }
    fn routing(&mut self, rows: &[Row<'_, '_, 'a, 'b>]) -> Result<Vec<bool>> {
        let ends: Vec<usize> = rows.iter().scan(0, |end, r| { *end += r.tokens.len(); Some(*end) }).collect();
        let rejected = match &self.scratch.moe {
            Some(moe) => moe.rejected(self.scratch.model.backend, *ends.last().unwrap(), rows.len(), |row| ends.partition_point(|&end| end <= row))?,
            None => vec![false; rows.len()],
        };
        self.rejected = (0..rows.len()).filter(|&i| rejected[i]).collect();
        Ok(rejected)
    }
    fn validate(&self, rows: &[Row<'_, '_, 'a, 'b>]) -> Result<usize> {
        let model = self.scratch.model;
        let c = &model.config;
        let count: usize = rows.iter().map(|r| r.tokens.len()).sum();
        if rows.is_empty() || rows.len() > decode::MAX_ROWS || count > model.prefill_chunk_size()
            || rows.iter().any(|r| !std::ptr::eq(r.session.model, model) || !r.session.valid
                || r.tokens.is_empty() || r.session.position.checked_add(r.tokens.len()).is_none_or(|n| n > model.capacity)
                || r.tokens.iter().any(|&t| t as u64 >= c.vocab)) {
            return Err("invalid shared hybrid prompts or capacity".into());
        }
        Ok(count)
    }
    fn forward(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<()> {
        let b = self.scratch.model.backend;
        b.generic_profile_begin(true, || (0, rows.iter().map(|r| r.tokens.len()).sum(), self.scratch.model.profile_plan.as_ref().unwrap().profile_classes(), 0))?;
        let result = self.forward_profiled(rows); b.generic_profile_end(true, result.is_ok())?; result
    }
    fn forward_profiled(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<()> {
        let count = self.validate(rows)?;
        let model = self.scratch.model;
        let (b, c) = (model.backend, &model.config);
        self.scratch.reserve_workspace_kind(count, true)?;
        for row in rows.iter_mut() {
            row.session.reserve_prompt_workspace(row.tokens.len())?;
            row.session.reserve_kv(row.session.position + row.tokens.len())?;
            row.session.reserve_state()?;
            if row.tokens.len() < model.prefill_chunk_size() { row.session.save_prefill_entry()?; }
            row.session.valid = false;
        }
        let mut descriptors = vec![0.0; self.gdn_rows.len()];
        for (layer, kind) in c.layers.iter().enumerate() {
            if *kind != LayerKind::DeltaNet { continue; }
            let mut offset = 0;
            for (i, row) in rows.iter().enumerate() {
                let at = (layer * decode::MAX_ROWS + i) * 30;
                descriptors[at..at + 30].copy_from_slice(&row.session.gdn_cache[layer].as_ref().unwrap()
                    .prompt_descriptor(&row.session.gdn, offset, row.tokens.len()));
                offset += row.tokens.len();
            }
        }
        b.write_vector(&mut self.gdn_rows, &descriptors)?;
        let h = c.hidden as usize;
        for v in [&mut self.scratch.hidden, &mut self.scratch.normalized, &mut self.scratch.block] {
            v.resize_within_capacity(count * h)?;
        }
        self.scratch.gate.resize_within_capacity(if c.fp8 { count * c.intermediate as usize }
            else { count.div_ceil(16) * 16 * c.intermediate as usize / 2 })?;
        self.scratch.ids.resize_within_capacity(count)?;
        b.write_vector(&mut self.scratch.ids, &rows.iter().flat_map(|r| r.tokens.iter().map(|&t| t as f32)).collect::<Vec<_>>())?;
        self.scratch.enqueue_forward_rows(count, true, false, Some(rows), Some(&self.gdn_rows))?;
        let mut offset = 0;
        for row in rows.iter_mut() {
            if row.session.mtp.is_some() {
                row.session.mtp_observe_hidden(row.tokens, &self.scratch.hidden.slice(offset * h..(offset + row.tokens.len()) * h)?)?;
            }
            offset += row.tokens.len();
        }
        self.final_hidden.resize_within_capacity(rows.len() * h)?;
        let mut offset = 0;
        for (i, row) in rows.iter().enumerate() {
            offset += row.tokens.len();
            b.copy_range(&self.scratch.hidden, &mut self.final_hidden, (offset - 1) * h, i * h, h)?;
        }
        if rows.iter().any(|r| r.logits) {
            model.profile_reads(None, Vec::new);
            self.final_normalized.resize_within_capacity(rows.len() * h)?;
            self.logits.resize_within_capacity(rows.len() * c.vocab as usize)?;
            Session::normalize_order(b, model, &self.final_hidden, model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(), &self.final_normalized, rows.len(), Some(0), false)?;
            b.profile_output_head(|| b.matmul_decode_bf16_device(model.matrix("lm_head.weight")?,
                &self.final_normalized, &mut self.logits, rows.len()))?;
        }
        Ok(())
    }
    fn commit(rows: &mut [Row<'_, '_, 'a, 'b>], rejected: &[bool]) {
        for row in rows.iter_mut().zip(rejected).filter(|r| !r.1).map(|r| r.0) {
            for state in row.session.gdn_cache.iter_mut().flatten() { state.commit_prompt(); }
            row.session.position += row.tokens.len();
            if row.tokens.len() < row.session.model.prefill_chunk_size() {
                row.session.prefill_tail.extend_from_slice(row.tokens);
            }
            row.session.valid = true;
        }
    }
}
