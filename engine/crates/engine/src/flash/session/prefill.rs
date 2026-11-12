//! Shared prompt scratch and staged prefill orchestration.
use super::*;

impl<'a> PromptScratch<'a> {
    pub(super) fn new(b: &Backend<'a>, weights: &[Weights<'_>], group: usize, slots: usize, activation: bool, mailbox: bool) -> Result<Self> {
        let local = weights.len();
        let mut scratch = Self { front: Vec::new(), gdn: Vec::new(), hc: Vec::new(), moe: Vec::new(),
            input: Vec::new(), hidden: Vec::new(), exchange: Vec::new(), staging: Vec::new(), group, slots, mailbox,
            routing: (0..local).map(|_| Fp8RoutingFlag::new(b, group)).collect::<Result<Vec<_>>>()? };
        for w in weights {
            scratch.front.push(FrontScratch::with_tokens(b, group)?);
            scratch.gdn.push(GdnScratch::group(b, w, group.min(GDN_SPAN))?);
        }
        for _ in 0..slots * local {
            scratch.hc.push(HcScratch::with_tokens(b, group)?);
            scratch.moe.push(MoeScratch::with_capacity(b, group)?);
            let plan = &weights[(scratch.moe.len() - 1) % local].plan;
            scratch.moe.last_mut().unwrap().reserve_grouped(b, group, activation,
                plan.expert_output_rows.as_ref().filter(|_| !plan.is_pair()).map(|_| plan.ranks as usize))?;
            let (input, hidden) = execution_rows(b, group)?;
            scratch.input.push(input);
            scratch.hidden.push(hidden);
            // Duplex prompt reductions use the registered mailbox directly.
            // Keep only a single-row slot for the common workspace interface.
            scratch.exchange.push(b.exchange_slot(if mailbox { PARTIALS } else { group * PARTIALS })?);
        }
        for _ in 0..2 { scratch.staging.push(b.prompt_buffer(group * slots * local * (EMBEDDING_BYTES + LEXICAL_BYTES))?); }
        Ok(scratch)
    }
    pub(super) fn device_bytes(&self) -> usize {
        self.routing.iter().map(Fp8RoutingFlag::bytes).sum::<usize>()
            + self.front.iter().map(FrontScratch::device_bytes).sum::<usize>()
            + self.gdn.iter().map(GdnScratch::device_bytes).sum::<usize>()
            + self.hc.iter().map(HcScratch::device_bytes).sum::<usize>()
            + self.moe.iter().map(MoeScratch::device_bytes).sum::<usize>()
            + self.input.len() * self.group * (H + W) * 4
    }
}

impl<'a, 'w, E: Exchange> Session<'a, 'w, E> {
    /// Keep chunk arithmetic for every prefill tile, including a final single
    /// token. Requested logits are gathered only for the final prompt position.
    /// Output-owned expert gathers use one group; other resident host reductions
    /// can carry GROUPS_IN_FLIGHT groups through each layer loop.
    pub fn forward_prefill(&mut self, b: &Backend<'a>, tokens: &[u32], save_logits: bool) -> Result<Output> {
        self.commit_verification(b, self.position)?;
        // Resolve a missing canonical draft prefix before borrowing/swapping
        // shared prompt scratch; disabling resizes only private decode buffers.
        if self.mtp_enabled && !self.prefill_tail.is_empty()
            && self.canonical.as_ref().is_some_and(|saved| saved.mtp.is_empty()) {
            self.disable_mtp_history(b)?;
        }
        self.model.prepare_prompt(b, self.capacity, self.tile, self.mailbox.is_some())?;
        let model = self.model;
        let mut prompt = model.prompt.try_borrow_mut().map_err(|_| "Flash prompt workspace is in use")?;
        if let Some(scratch) = prompt.as_mut() {
            self.drop_graphs();
            self.swap_prompt(scratch);
        }
        b.set_spin_waits(true);
        let previous_profile = self.model.profile_layers.then(|| {
            let previous = b.is_profiling();
            b.set_profiling(true);
            previous
        });
        let timing = b.is_profiling().then(|| (Instant::now(), b.stats()));
        let output = spark_gpu_probe::host_prep::with_worker(||
            self.prefill(b, tokens, save_logits, &mut prompt.as_mut().unwrap().staging)).and_then(|result| result);
        if let Some((start, before)) = timing {
            println!("flash_prompt_staging rows={} wall_ns={} host_pack_ns={} pinned_bytes={} pinned_host_bytes={}",
                tokens.len(), start.elapsed().as_nanos(), b.stats().since(before).host_pack_ns,
                prompt.as_ref().unwrap().staging.iter().map(PromptBuffer::bytes).sum::<usize>(), b.pinned_host_bytes());
        }
        if let Some(previous) = previous_profile { b.set_profiling(previous); }
        b.set_spin_waits(false);
        if let Some(scratch) = prompt.as_mut() { self.swap_prompt(scratch); }
        output
    }
    fn swap_prompt(&mut self, scratch: &mut PromptScratch<'a>) {
        std::mem::swap(&mut self.front, &mut scratch.front);
        std::mem::swap(&mut self.gdn_group, &mut scratch.gdn);
        std::mem::swap(&mut self.hc, &mut scratch.hc);
        std::mem::swap(&mut self.moe, &mut scratch.moe);
        std::mem::swap(&mut self.input, &mut scratch.input);
        std::mem::swap(&mut self.hidden, &mut scratch.hidden);
        std::mem::swap(&mut self.exchange, &mut scratch.exchange);
        std::mem::swap(&mut self.routing, &mut scratch.routing);
        std::mem::swap(&mut self.group, &mut scratch.group);
        std::mem::swap(&mut self.slots, &mut scratch.slots);
    }
    fn prefill(&mut self, b: &Backend<'a>, tokens: &[u32], save_logits: bool, staging: &mut [PromptBuffer<'a>]) -> Result<Output> {
        if !self.valid || tokens.is_empty() || self.position.checked_add(tokens.len()).is_none_or(|n| n > self.capacity) {
            return Err("invalid Flash session state or context capacity; reset after an error".into());
        }
        if tokens.len() > self.tile && self.tile != 64 {
            return Err("multi-tile Flash prefill requires tile=64 to preserve GDN chunk arithmetic".into());
        }
        self.drop_graphs();
        let start = Instant::now();
        let mut input = Vec::new();
        if self.caching && !self.prefill_tail.is_empty() {
            input = std::mem::take(&mut self.prefill_tail);
            let saved = self.canonical.clone().ok_or("missing Flash canonical checkpoint")?;
            self.restore_recurrent(b, &saved)?;
            self.restore_mtp(b, &saved.mtp)?;
            input.extend_from_slice(tokens);
        }
        let tokens = if input.is_empty() { tokens } else { &input };
        let mut load_ms = 0.0;
        let mut output = None;
        let allocated_group = self.group;
        self.group = self.prefill_group.unwrap_or(allocated_group);
        let stride = self.group * self.slots;
        // forward_tile's checkpoint and MTP offsets must use this same stride.
        // Keep the allocated workspace bound for subsequent decode and swaps,
        // including when a tile fails.
        let result = (|| {
            // A previous prefill reported any slot failure and drained CPU work.
            for slot in staging.iter_mut() { slot.reset_failed()?; }
            let mut capture = self.rows.clone();
            let mut windows = tokens.chunks(stride).peekable();
            let (send, mut ready) = std::sync::mpsc::channel();
            self.stage_prompt(b, &mut capture, windows.next().unwrap(), &mut staging[0])?.start_pack(b, &mut staging[0], send)?;
            let mut index = 0;
            loop {
                let slot = index % 2;
                b.record_host_pack(staging[slot].finish_pack()?);
                let prepared = ready.recv().map_err(|_| "prompt rows were not prepared")?;
                let next = if let Some(&window) = windows.peek() {
                    let (send, ready) = std::sync::mpsc::channel();
                    Some((self.stage_prompt(b, &mut capture, window, &mut staging[1 - slot])?, send, ready))
                } else { None };
                let last = (index + 1) * stride >= tokens.len();
                let window = &tokens[index * stride..tokens.len().min((index + 1) * stride)];
                let groups = window.chunks(self.group).collect::<Vec<_>>();
                let (left, right) = staging.split_at_mut(1);
                let (current_slot, next_slot) = if slot == 0 { (&mut left[0], &mut right[0]) } else { (&mut right[0], &mut left[0]) };
                let (lookahead, next_ready) = match next {
                    Some((pending, send, ready)) => (Some((pending, next_slot, send)), Some(ready)),
                    None => (None, None),
                };
                b.prefill_profile_begin(self.position, window.len(), self.model.persistent[0].plan.rank as usize, &PREFILL_PROFILE_CLASSES);
                let current = self.forward_tile(b, &groups, save_logits && last, true,
                    Some(prepared), Some(current_slot), lookahead);
                b.prefill_profile_end(current.is_ok())?;
                let current = current?;
                load_ms += current.load_ms;
                output = Some(current);
                let Some(next) = next_ready else { break; };
                ready = next;
                windows.next();
                index += 1;
            }
            Ok::<_, String>(())
        })();
        // Lookahead is drained on every exit before scratch can be swapped/freed.
        let mut drained = Ok(());
        for slot in staging {
            match slot.finish_pack() {
                Ok(ns) => b.record_host_pack(ns), Err(error) => { drained = Err(error); }
            }
        }
        self.group = allocated_group;
        if result.is_err() || drained.is_err() { self.valid = false; }
        result?; drained?;
        let mut output = output.unwrap();
        output.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        output.load_ms = load_ms;
        Ok(output)
    }
    /// Only immutable row copies move ahead of the current layer loop. Rank
    /// gathers remain in forward_tile, in their original chunk/owner order.
    fn stage_prompt(&self, b: &Backend<'a>, rows: &mut [Rows], window: &[u32], slot: &mut PromptBuffer<'a>)
        -> Result<PendingRows> {
        let local = rows.len();
        let groups = window.chunks(self.group).collect::<Vec<_>>();
        let boundary = window.len() / self.tile * self.tile;
        let mut saved = None;
        let mut batches: Vec<Vec<Vec<RowBatch>>> = (0..groups.len()).map(|_| (0..local).map(|_| Vec::new()).collect()).collect();
        let mut copies = (0..local).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut offsets = Vec::new();
        let mut at = 0;
        b.profile_host_pack(|| -> Result<PendingRows> {
            for (g, tokens) in groups.iter().enumerate() {
                for (index, chunk) in tokens.chunks(self.tile).enumerate() {
                    for r in 0..local {
                        let mut batch = rows[r].capture_prompt(&self.model.persistent[r], chunk, at)?;
                        copies[r].extend(batch.take_prompt_copies());
                        offsets.push((g, r, index, at));
                        at += chunk.len() * (EMBEDDING_BYTES + LEXICAL_BYTES);
                        batches[g][r].push(batch);
                    }
                    if g * self.group + (index + 1) * self.tile == boundary {
                        saved = Some(rows.iter().map(Rows::history).collect());
                    }
                }
            }
            for (weights, copies) in self.model.persistent.iter().zip(copies) { weights.stage_prompt_rows(b, slot, &copies)?; }
            let after = rows.iter().map(Rows::history).collect();
            let ranks = self.model.persistent.iter().map(|w| w.plan.rank as usize).collect::<Vec<_>>();
            Ok(PendingRows { prepared: PreparedRows { batches, after, boundary: saved }, offsets, ranks,
                group: self.group, tile: self.tile })
        })
    }
}
