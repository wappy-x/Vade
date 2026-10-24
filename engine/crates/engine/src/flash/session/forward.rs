//! Ordered tile execution, layer collectives and output projection.
use super::*;

pub(super) fn decode_graph_slots(model: &Model<'_>, group: usize) -> usize {
    (48 * (if model.persistent[0].plan.expert_output_rows.is_some() { 3 } else { 2 }) + 24) * group
}

impl<'a, 'w, E: Exchange> Session<'a, 'w, E> {
    /// Greedy output is reduced from global-ID local maxima; only requested
    /// logit rows are gathered. Cached state stays attached between tiles.
    /// Each group of up to `group` tokens runs attention in `tile`-token chunks.
    /// Groups pass through every layer in prompt order; a group's collectives
    /// are queued behind the next group's segments, so the host sums one
    /// group's pieces while the GPU still computes the other's. Per group,
    /// kernels, pieces and additions keep their order.
    pub(super) fn forward_tile(&mut self, b: &Backend<'a>, groups: &[&[u32]], save_logits: bool, prefill: bool,
        prepared: Option<PreparedRows>, staging: Option<&mut PromptBuffer<'a>>,
        lookahead: Option<(PendingRows, &mut PromptBuffer<'a>, std::sync::mpsc::Sender<PreparedRows>)>) -> Result<Output> {
        if !self.ranks.first().is_some_and(Exchange::collective_profiling) {
            return self.forward_tile_staged(b, groups, save_logits, prefill, prepared, staging, lookahead);
        }
        let count = groups.iter().map(|g| g.len()).sum();
        for rank in &mut self.ranks { rank.profile_begin_step(if self.verifying { "verify" } else if prefill { "prefill" } else { "decode" }, count); }
        let result = self.forward_tile_staged(b, groups, save_logits, prefill, prepared, staging, lookahead);
        for rank in &mut self.ranks { rank.profile_end_step(result.is_ok(), b.allocated_bytes()); }
        result
    }
    fn forward_tile_staged(&mut self, b: &Backend<'a>, groups: &[&[u32]], save_logits: bool, prefill: bool,
        prepared: Option<PreparedRows>, staging: Option<&mut PromptBuffer<'a>>,
        lookahead: Option<(PendingRows, &mut PromptBuffer<'a>, std::sync::mpsc::Sender<PreparedRows>)>) -> Result<Output> {
        let count = groups.iter().map(|g| g.len()).sum::<usize>();
        if !self.valid || groups.is_empty() || groups.len() > self.slots
            || groups.iter().any(|g| g.is_empty() || g.len() > self.group)
            || self.position.checked_add(count).is_none_or(|n| n > self.capacity) {
            return Err("invalid Flash session state, tile size or context capacity; reset after an error".into());
        }
        debug_assert!(self.input.iter().chain(&self.hidden).all(|v| v.slice(0..v.len()).is_ok()), "Flash execution rows require arena storage before graph capture");
        self.valid = false;
        let start = Instant::now();
        let mut load_ms = 0.0;
        let local = self.ranks.len();
        let boundary = count / self.tile * self.tile;
        let mut checkpoint = if self.caching && prefill && !self.verifying {
            Some(if boundary == 0 { self.recurrent(b)? } else {
                Recurrent { position: self.position + boundary, rows: Vec::new(), ple: Vec::new(), mtp: Vec::new(),
                    gdn: (0..local).map(|_| (0..48).map(|_| None).collect()).collect() }
            })
        } else { None };
        if self.verifying {
            let v = self.verification.as_mut().ok_or("missing Flash verification state")?;
            for rows in &mut self.rows {
                let before = rows.history();
                for (i, &token) in groups[0].iter().enumerate() {
                    rows.advance(token)?;
                    if i < v.saved.len() { v.saved[i].rows.push(rows.history()); }
                }
                rows.restore_history(before);
            }
        }
        let model = self.model;
        let tile = self.tile;
        let n = groups.len();
        let ws = |g: usize, r: usize| g * local + r;
        // Verification reuses each retained prefix allocation before capture.
        // Every width owns separate graphs; all dynamic token inputs live in
        // fixed workspaces, and recurrence is still one ordered step per row.
        let verified_ready = self.verifying && self.verification.as_ref().is_some_and(|v|
            v.saved.iter().take(count - 1).all(|p| p.ple.len() == local && p.gdn.iter().all(|layers|
                model.checkpoint.config.layers.iter().zip(layers).all(|(kind, s)| *kind != LayerKind::DeltaNet || s.is_some()))));
        // A wider verification can replace the private expert allocation.
        // Discard narrower graphs before that replacement, not just its capture.
        if self.moe.iter().take(local).any(|s| s.prepared_tokens() < count) { self.drop_graphs(); }
        let graphs = ((!prefill && count == 1) || verified_ready) && local == 1 && model.resident
            && self.gdn.iter().all(Option::is_some)
            && self.moe.iter().take(local).all(|s| s.prepared_tokens() >= count);
        for (r, flag) in self.routing.iter().enumerate() {
            flag.clear(b, groups.iter().map(|g| g.len()).max().unwrap(), model.persistent[r].plan.rank as usize)?;
        }
        for (g, tokens) in groups.iter().enumerate() {
            for r in 0..local {
                self.moe[ws(g, r)].reserve_grouped(b, tokens.len(), model.main_activation_quantization(),
                    model.persistent[r].plan.expert_output_rows.as_ref().map(|_| model.persistent[r].plan.ranks as usize))?;
                self.moe[ws(g, r)].set_routing_flag(&self.routing[r])?;
            }
        }
        let compact = model.persistent[0].plan.expert_output_rows.is_some();
        let segments = if compact { 3 } else { 2 };
        let graph_base = if self.verifying { (count - 1) * (48 * segments + 24) } else { 0 };
        if graphs {
            let mut shape = super::super::qsa::decode_shape(self.position, count, self.capacity);
            shape[0] = 0; // Keep both pooling variants; their other launch dimensions agree.
            if self.qsa_shapes[count - 1] != shape {
                b.synchronize()?;
                for graph in &mut self.graphs[graph_base + 48 * segments..graph_base + 48 * segments + 24] { *graph = None; }
                self.qsa_shapes[count - 1] = shape;
            }
            self.qsa[0].upload_cursor(b, self.position, count)?;
        }
        let duplex = prefill && !graphs && self.mailbox.is_some() && model.persistent[0].plan.is_pair();
        let mut batches = if let Some(mut prepared) = prepared {
            for g in &mut prepared.batches {
                for _ in 0..g[0].len() {
                    let captured = g.iter_mut().map(|rank| rank.remove(0)).collect::<Vec<_>>();
                    b.prefill_read_bytes(9, || captured.iter().map(|v| *v.offsets().last().unwrap()).sum());
                    let gathered = b.prefill_collective(|| parallel(&mut self.ranks, captured, |rank, mut batch| {
                        batch.gather(|v,o| rank.gather_bytes(v,o))?; Ok(batch)
                    }))?;
                    for (rank, batch) in g.iter_mut().zip(gathered) { rank.push(batch); }
                }
            }
            for (rows, history) in self.rows.iter_mut().zip(prepared.after) { rows.restore_history(history); }
            if let Some(saved) = &mut checkpoint { if let Some(history) = prepared.boundary { saved.rows = history; } }
            prepared.batches
        } else {
            let mut batches: Vec<Vec<Vec<RowBatch>>> = (0..n).map(|_| (0..local).map(|_| Vec::new()).collect()).collect();
            for (g, tokens) in groups.iter().enumerate() {
                for (index, chunk) in tokens.chunks(tile).enumerate() {
                    let captured = b.profile_host_pack(|| (0..local).map(|r| {
                        if !self.verifying { return self.rows[r].capture(&model.persistent[r], chunk); }
                        let (wire, table) = self.verify_staging[r].split_at_mut(1);
                        let mut batch = self.rows[r].capture_prompt(&model.persistent[r], chunk, wire[0].device_address() as usize)?;
                        RowBatch::gather_device(b, std::slice::from_mut(&mut batch), model.persistent[r].plan.rank as usize,
                            &mut wire[0], &mut table[0], |_, _| Ok(()))?;
                        Ok(batch)
                    }).collect::<Result<Vec<RowBatch>>>())?;
                    b.prefill_read_bytes(9, || captured.iter().map(|v| *v.offsets().last().unwrap()).sum());
                    let gathered = b.prefill_collective(|| parallel(&mut self.ranks, captured, |rank, mut batch| {
                        batch.gather(|v,o| rank.gather_bytes(v,o))?; Ok(batch)
                    }))?;
                    for (rows, batch) in batches[g].iter_mut().zip(gathered) { rows.push(batch); }
                    if g * self.group + (index + 1) * tile == boundary {
                        if let Some(saved) = &mut checkpoint { saved.rows = self.rows.iter().map(Rows::history).collect(); }
                    }
                }
            }
            batches
        };
        // Patch remote rows on the preparation worker after the ordered gather.
        // Uploads use this slot directly; local rows were packed in lookahead.
        let staging = if let Some(slot) = staging {
            let group = self.group;
            let tile = self.tile;
            let ranks = self.model.persistent.iter().map(|w| w.plan.rank as usize).collect::<Vec<_>>();
            let (send, ready) = std::sync::mpsc::channel();
            b.profile_host_pack(|| slot.start_pack(b.is_profiling(), move |bytes| {
                for g in 0..batches.len() {
                    for r in 0..local {
                        let base = (g * local + r) * group * (EMBEDDING_BYTES + LEXICAL_BYTES);
                        for (index, batch) in batches[g][r].iter().enumerate() {
                            batch.pack_prompt(bytes, base, index * tile, group, ranks[r], false);
                        }
                    }
                }
                let _ = send.send(batches);
                Ok(())
            }))?;
            b.record_host_pack(slot.finish_pack()?);
            batches = ready.recv().map_err(|_| "prompt rows were not packed")?;
            Some(&*slot)
        } else { None };
        let token = |g: usize, r: usize, t: usize| batches[g][r][t / tile].token(t % tile);
        for g in 0..n {
            for r in 0..local {
                let count = groups[g].len();
                if prefill {
                    if let Some(slot) = staging {
                        self.front[r].expand_staged(b, slot, ws(g, r) * self.group * (EMBEDDING_BYTES + LEXICAL_BYTES),
                            &self.hidden[ws(g, r)], count)?;
                    } else {
                        // One upload of the group's embedding rows, expanded straight into its residual rows.
                        let mut rows = Vec::with_capacity(count * EMBEDDING_BYTES);
                        for t in 0..count { rows.extend_from_slice(&token(g, r, t)?.embedding); }
                        self.front[r].expand_rows(b, &rows, &self.hidden[ws(g, r)], count)?;
                    }
                } else {
                    for t in 0..count {
                        self.front[r].expand(b, &token(g, r, t)?)?;
                        b.copy_range(&self.front[r].residual, &mut self.hidden[ws(g, r)], 0, t*W, W)?;
                    }
                }
                self.input[ws(g, r)].resize_within_capacity(count * H)?;
                self.moe[ws(g, r)].resize(count)?;
            }
        }
        // Current embedding kernels are queued. The next slot's download event
        // precedes them, so CPU packing now overlaps this chunk's GPU work.
        if let Some((pending, slot, send)) = lookahead { pending.start_pack(b, slot, send)?; }
        let layers = self.model.checkpoint.config.layers.len();
        for layer in 0..layers {
            if prefill { b.prefill_profile_layer(|| model.checkpoint.config.layers[layer].profile_name()); }
            let timing = if prefill && b.is_profiling() {
                b.synchronize()?;
                Some((Instant::now(), b.stats(), self.ranks[0].profile_stats()))
            } else { None };
            let prefix = format!("model.language_model.layers.{layer}.");
            let streamed = if model.resident { Vec::new() } else {
                let loading = Instant::now();
                let weights = (0..local).map(|r| Weights::load_selected(b, &model.root,
                    &model.checkpoint, model.persistent[r].plan.clone(),
                    |name| name.starts_with(&prefix) && !name.starts_with(PLE)))
                    .collect::<Result<Vec<_>>>()?;
                load_ms += loading.elapsed().as_secs_f64()*1000.0;
                weights
            };
            let weights = if model.resident { &model.persistent } else { &streamed };
            let mut active = [0;2];
            if !model.resident { for r in 0..local { active[r] = weights[r].resident_bytes(); } }
            let connections = weights.iter().map(|w| Hc::load(w, &format!("{prefix}attn_hyper_connection")))
                .collect::<Result<Vec<_>>>()?;
            let ple = self.model.checkpoint.config.ple_layers.contains(&(layer as u64));
            let deltanet = matches!(self.model.checkpoint.config.layers[layer], LayerKind::DeltaNet);
            b.prefill_collective(|| self.prepare_attention(b, weights, layer))?;
            for g in 0..n {
                let count = groups[g].len();
                // A later group's previous-layer MLP finishes here, behind the
                // attention just queued for the group before it.
                if g > 0 && layer > 0 && !compact { self.finish_mlp(b, g, layer - 1, count)?; }
                if ple && prefill {
                    for r in 0..local {
                        if let Some(slot) = staging {
                            self.front[r].upload_staged_lexical(b, slot,
                                ws(g, r) * self.group * (EMBEDDING_BYTES + LEXICAL_BYTES) + self.group * EMBEDDING_BYTES, count)?;
                        } else {
                            let mut rows = Vec::with_capacity(count * LEXICAL_BYTES);
                            for t in 0..count { rows.extend_from_slice(&token(g, r, t)?.ple); }
                            self.front[r].upload_lexical_rows(b, &rows, count)?;
                        }
                    }
                }
                // Segment A: the previous layer's MLP finish (decode graphs), the
                // lexical block, the attention HC mix and a whole GDN attention.
                self.segment(b, graphs.then_some(graph_base + segments * layer), |this| {
                    if graphs && layer > 0 {
                        this.resident_moe[0][layer - 1].finish_batch(b, &mut this.moe[0], &mut this.hidden[0], count)?;
                    }
                    for r in 0..local {
                        let w = ws(g, r);
                        // The lexical convolution is causal over tokens (in order inside
                        // project_rows); key and value projections cover 64 tokens per pass.
                        // The HC mix is per token.
                        if ple {
                            let scale = token(g, r, 0)?.scale;
                            for start in (0..count).step_by(if this.verifying && count != 3 { 1 } else { PLE_TOKENS }) {
                                let n = if this.verifying && count != 3 { 1 } else { PLE_TOKENS.min(count - start) };
                                for t in start..start + n {
                                    if prefill { this.ple[r].decode_lexical(b, this.front[r].lexical_row(t), scale, t - start)?; }
                                    // A decode token's lexical row was expanded at the start of the call.
                                    else { this.ple[r].set_lexical(b, &this.front[r].lexical, t - start)?; }
                                }
                                let residual = this.hidden[w].device_address() + (start * W * 4) as u64;
                                if this.verifying {
                                    let (keys, values) = this.ple[r].project_lexical(b, n)?;
                                    this.ple[r].gate_rows(b, residual, keys, values, n)?;
                                    for t in 0..n {
                                        this.ple[r].convolve_rows(b, residual + (t * W * 4) as u64,
                                            keys + (t * W * 4) as u64, t, 1)?;
                                        if start + t + 1 < count {
                                            let saved = &mut this.verification.as_mut().unwrap().saved[start + t].ple;
                                            if saved.len() <= r { saved.push(this.ple[r].save_history(b)?); }
                                            else { this.ple[r].copy_history(b, &mut saved[r])?; }
                                        }
                                    }
                                } else {
                                    this.ple[r].project_rows(b, residual, n)?;
                                }
                                if g * this.group + start + n == boundary {
                                    if let Some(saved) = &mut checkpoint { saved.ple.push(this.ple[r].save_history(b)?); }
                                }
                            }
                        }
                        connections[r].mix_tiled(b, &mut this.hc[w], &this.hidden[w], &this.input[w], count)?;
                    }
                    if !deltanet { return Ok(()); }
                    for r in 0..local {
                        let w = ws(g, r);
                        let scratch = this.gdn[r].as_mut().unwrap();
                        let fresh = this.gdn_cache[r][layer].is_none();
                        let mut attention = Gdn::load_cached(b, &weights[r], layer, this.gdn_cache[r][layer].take())?;
                        if fresh { this.cache_bytes[r] += attention.state_bytes(); }
                        let projected = if prefill && !this.verifying && count > tile {
                            let batch = GdnBatch::workspace(&weights[r], count, this.moe[w].partials())?;
                            attention.project_batch(b, &batch, &this.input[w], count, local == 1)?;
                            Some(batch)
                        } else { None };
                        let spans = if projected.is_some() {
                            this.gdn_group.get_mut(r).ok_or("missing Flash prompt GDN workspace")?
                                .copy_norm(b, this.gdn_norms[r][layer].as_ref().unwrap())?;
                            recurrence_spans(count, boundary.saturating_sub(g * this.group))
                        } else {
                            scratch.copy_norm(b, this.gdn_norms[r][layer].as_ref().unwrap())?;
                            vec![(0, count)]
                        };
                        for &(start, n) in &spans {
                            if this.verifying {
                                let mut saved = this.verification.as_mut().unwrap().saved.iter_mut().take(count - 1)
                                    .map(|v| &mut v.gdn[r][layer]).collect::<Vec<_>>();
                                attention.verify_rows(b, scratch, this.gdn_verify[r].as_mut().unwrap(), &this.input[w], count, &mut saved)?;
                                b.copy_range(scratch.output(), &mut this.input[w], 0, 0, count * H)?;
                                continue;
                            }
                            if let Some(batch) = &projected { attention.prefill_batch_rows(b, batch, start, &this.gdn_group[r], n)?; }
                            else if prefill { attention.prefill(b, scratch, &this.input[w], n)?; }
                            else { attention.run(b, scratch, &this.input[w], n)?; }
                            if g * this.group + start + n == boundary {
                                if let Some(saved) = &mut checkpoint { saved.gdn[r][layer] = Some(attention.snapshot(b)?); }
                            }
                            if projected.is_none() { b.copy_range(scratch.output(), &mut this.input[w], 0, start * H, n * H)?; }
                        }
                        if let Some(batch) = &projected { attention.finish_batch(b, batch, &mut this.input[w], count)?; }
                        this.gdn_cache[r][layer] = Some(attention.into_state());
                    }
                    Ok(())
                })?;
                if !deltanet {
                    let mut attention = Vec::new();
                    for r in 0..local {
                        self.qsa[r].use_norm(self.qsa_norms[r][layer].as_ref().unwrap())?;
                        let fresh = self.qsa_cache[r][layer].is_none();
                        let q = Qsa::load_cached(b,&weights[r],layer,self.capacity,self.qsa_cache[r][layer].take())?;
                        if fresh { self.cache_bytes[r] += q.kv_bytes() + q.index_bytes(); }
                        attention.push(q);
                    }
                    // Previous-layer MLP finish precedes this group on the stream;
                    // current-layer MLP starts after the final attention O launch.
                    let mut projected = Vec::new();
                    if prefill && !self.verifying && count > tile {
                        for (r, q) in attention.iter_mut().enumerate() {
                            let w = ws(g, r);
                            self.moe[w].partials_mut().resize_within_capacity(count * PARTIALS)?;
                            let mut batch = QsaBatch::workspace(&weights[r].plan, count, self.moe[w].partials())?;
                            q.project_prompt(b, &self.qsa[r], &mut batch, &self.input[w], count, local == 1)?;
                            projected.push(batch);
                        }
                    }
                    if graphs {
                        self.qsa[0].cursor = self.qsa[0].cursor_address();
                        self.segment(b, Some(graph_base + 48 * segments + layer / 4 * 2
                            + usize::from((self.position + count) / 4 - self.position / 4 > count / 4)), |this| {
                            attention[0].project(b, &mut this.qsa[0], &this.input[0], count)?;
                            attention[0].attend(b, &mut this.qsa[0])?;
                            b.copy_range(this.qsa[0].output(), &mut this.input[0], 0, 0, count * H)
                        })?;
                        // Capture records host cursor changes once; each replay commits them here.
                        attention[0].rewind(self.position + count);
                    } else { for start in (0..count).step_by(tile) {
                        let n = tile.min(count - start);
                        for (r, q) in attention.iter_mut().enumerate() {
                            let scratch = &mut self.qsa[r];
                            if let Some(batch) = projected.get(r) { q.attend_batch_row(b, batch, start, scratch, n)?; }
                            else {
                                let input = self.input[ws(g, r)].slice(start * H..(start + n) * H)?;
                                q.project(b,scratch,&input,n)?;
                                q.attend(b,scratch)?;
                                b.copy_range(scratch.output(), &mut self.input[ws(g, r)], 0, start * H, n * H)?;
                            }
                        }
                    }
                    }
                    for (r, q) in attention.into_iter().enumerate() {
                        if let Some(batch) = projected.get(r) { q.finish_batch(b, batch, &mut self.input[ws(g, r)], count)?; }
                        self.qsa_cache[r][layer] = Some(q.into_state());
                    }
                }
                if duplex {
                    reduce_device(b, &mut self.ranks, &mut self.communication, self.mailbox.as_deref(),
                        &mut self.input[..local].iter_mut().collect::<Vec<_>>())?;
                } else { self.begin_reduce(b, g, false)?; }
            }
            let streamed_moe = if model.resident { Vec::new() } else {
                weights.iter().map(|w| Moe::load(w,layer,model.activation_quantization)).collect::<Result<Vec<_>>>()?
            };
            for g in 0..n {
                let count = groups[g].len();
                if !duplex { self.finish_reduce(b, g, false)?; }
                // Segment B: attention injection and the routed/shared MLP partials.
                self.segment(b, graphs.then_some(graph_base + segments * layer + 1), |this| {
                    for r in 0..local {
                        let w = ws(g, r);
                        // Injection reads each token's normalized streams, still held from the tile's mix.
                        connections[r].combine_tiled(b, &mut this.hc[w], &mut this.hidden[w], &this.input[w], count)?;
                        let mlp = if model.resident { &this.resident_moe[r][layer] } else { &streamed_moe[r] };
                        mlp.project_batch(b, &mut this.moe[w], &this.hidden[w], count, this.moe_descriptors[r].get(layer))?;
                    }
                    Ok(())
                })?;
                if compact {
                    let mlp = if model.resident { &self.resident_moe[0][layer] } else { &streamed_moe[0] };
                    mlp.prepare_exchange(&mut self.moe[ws(g, 0)], count, self.moe_descriptors[0].get(layer))?;
                    exchange_middle(b, &mut self.ranks[0], self.mailbox.as_deref(), &self.moe[ws(g, 0)],
                        model.persistent[0].plan.rank as usize, count, false, &mut self.communication[0])?;
                    self.segment(b, graphs.then_some(graph_base + segments * layer + 2), |this| {
                        let mlp = if model.resident { &this.resident_moe[0][layer] } else { &streamed_moe[0] };
                        mlp.complete_middle(b, &this.moe[ws(g, 0)], count)
                    })?;
                    exchange_middle(b, &mut self.ranks[0], self.mailbox.as_deref(), &self.moe[ws(g, 0)],
                        model.persistent[0].plan.rank as usize, count, true, &mut self.communication[0])?;
                    if !graphs {
                        let mlp = if model.resident { &self.resident_moe[0][layer] } else { &streamed_moe[0] };
                        mlp.finish_batch(b, &mut self.moe[ws(g, 0)], &mut self.hidden[ws(g, 0)], count)?;
                    }
                } else if duplex {
                    let mlp = if model.resident { &self.resident_moe[0][layer] } else { &streamed_moe[0] };
                    reduce_finish_split(b, &mut self.ranks[0], self.mailbox.as_deref().unwrap(),
                        &mut self.moe[0], mlp, model.persistent[0].plan.rank == 0, count)?;
                    mlp.inject_batch(b, &mut self.moe[0], &mut self.hidden[0], count)?;
                } else { self.begin_reduce(b, g, true)?; }
            }
            // The first group's MLP finishes now; with decode graphs the next
            // segment A (or the final mixer) finishes it. Later groups finish
            // once the next layer's first attention is queued.
            if !duplex && !compact { self.finish_reduce(b, 0, true)?; }
            for r in (0..local).filter(|_| !graphs && !duplex && !compact) {
                let mlp = if model.resident { &self.resident_moe[r][layer] } else { &streamed_moe[r] };
                mlp.finish_batch(b, &mut self.moe[r], &mut self.hidden[r], groups[0].len())?;
            }
            self.account(b,active,0);
            if let Some((started, before, transport)) = timing {
                b.synchronize()?;
                let stats = b.stats().since(before);
                let after = self.ranks[0].profile_stats();
                let (transport_ns, sent, received, exchanges) = match (transport, after) {
                    (Some(before), Some(after)) => (after.elapsed_ns - before.elapsed_ns,
                        after.sent - before.sent, after.received - before.received, after.exchanges - before.exchanges),
                    _ => (0, 0, 0, 0),
                };
                println!("flash_layer_profile rank={} layer={layer} position={} rows={} groups={n} wall_ms={:.3} transport_ms={:.3} sent_bytes={sent} received_bytes={received} exchanges={exchanges} device_bytes={} kernel_ns={:?}",
                    model.persistent[0].plan.rank, self.position, count, started.elapsed().as_secs_f64() * 1000.0,
                    transport_ns as f64 / 1e6, b.allocated_bytes(), stats.device_kernel_ns);
            }
            if layer % 8 == 7 { println!("flash_session progress position={} tokens={count} layer={layer} elapsed_ms={:.3} load_ms={load_ms:.3} local_peak_device_bytes={}", self.position,start.elapsed().as_secs_f64()*1000.0,self.total_peak); }
        }
        if !compact {
            for g in 1..n { self.finish_mlp(b, g, layers - 1, groups[g].len())?; }
        }
        if graphs {
            self.resident_moe[0][layers - 1].finish_batch(b, &mut self.moe[0], &mut self.hidden[0], count)?;
        }
        if !self.mtp.is_empty() || self.verifying {
            if prefill { b.prefill_profile_layer(|| "draft"); }
            for g in 0..n {
                let count = groups[g].len();
                let mut hidden = Vec::new();
                let mut embeddings = Vec::new();
                for r in 0..local {
                    hidden.push(self.hidden[ws(g, r)].slice(0..count * W)?);
                    let mut rows = Vec::with_capacity(count * EMBEDDING_BYTES);
                    for t in 0..count { rows.extend_from_slice(&token(g, r, t)?.embedding); }
                    embeddings.push(rows);
                }
                if self.verifying {
                    let v = self.verification.as_mut().unwrap();
                    v.hidden = hidden;
                    v.embeddings = embeddings;
                } else {
                    let entry = self.position + g * self.group;
                    let split = boundary.saturating_sub(g * self.group).min(count);
                    self.observe_mtp(b, entry, split, &hidden, &embeddings, 0)?;
                    if split != 0 && g * self.group + split == boundary {
                        if let Some(saved) = &mut checkpoint { saved.mtp = self.save_mtp(b, entry + split)?; }
                    }
                    self.observe_mtp(b, entry + split, count - split, &hidden, &embeddings, split)?;
                }
            }
        }
        if prefill { b.prefill_profile_layer(|| "output"); }
        let (token, logits) = if self.verifying { (0, None) }
            else { self.output(b, n - 1, groups[n - 1].len() - 1, save_logits)? };
        if let Some(saved) = checkpoint {
            self.canonical = Some(Rc::new(saved));
            self.prefill_tail = groups.iter().flat_map(|g| g.iter().copied()).skip(boundary).collect();
        }
        self.position += count;
        self.valid = true;
        self.account(b, [0; 2], 0);
        Ok(Output {token,logits,elapsed_ms:start.elapsed().as_secs_f64()*1000.0,load_ms})
    }
    /// Queue group `g`'s attention output or MoE partials for the collective
    /// and return at once; launches queued afterwards keep the GPU busy.
    fn begin_reduce(&mut self, b: &Backend<'a>, g: usize, partials: bool) -> Result<()> {
        b.prefill_collective(|| exchange_point(&mut self.ranks, 0, |ranks| {
            let local = ranks.len();
            for r in 0..local {
                let w = g * local + r;
                let vector = if partials { self.moe[w].partials() } else { &self.input[w] };
                b.exchange_begin(&mut self.exchange[w], vector)?;
            }
            Ok(())
        }))
    }
    /// Wait for that download alone, sum it on the host in rank order (the same
    /// pieces and additions as a synchronous exchange) and queue it back.
    fn finish_reduce(&mut self, b: &Backend<'a>, g: usize, partials: bool) -> Result<()> {
        b.prefill_collective(|| exchange_point(&mut self.ranks, 0, |ranks| {
            let local = ranks.len();
            let hosts = self.exchange[g * local..(g + 1) * local].iter_mut()
                .map(|slot| b.exchange_wait(slot)).collect::<Result<Vec<_>>>()?;
            if local == 1 {
                reduce_pieces(&mut ranks[0], hosts.into_iter().next().unwrap(), &mut self.communication[0])?;
            } else {
                // Two local ranks on one GPU must both have staged before either blocks.
                std::thread::scope(|s| {
                    let jobs = ranks.iter_mut().zip(hosts).zip(&mut self.communication)
                        .map(|((rank, values), scratch)| s.spawn(move || reduce_pieces(rank, values, scratch)))
                        .collect::<Vec<_>>();
                    for job in jobs { job.join().map_err(|_| "Flash reduction worker panicked")??; }
                    Ok::<(), String>(())
                })?;
            }
            for r in 0..local {
                let w = g * local + r;
                let vector = if partials { self.moe[w].partials_mut() } else { &mut self.input[w] };
                b.exchange_upload(&self.exchange[w], vector)?;
            }
            Ok(())
        }))
    }
    /// Finish group `g`'s MLP of `layer` from its reduced partials. Only groups
    /// after the first defer their finish, and only resident sessions carry them.
    fn finish_mlp(&mut self, b: &Backend<'a>, g: usize, layer: usize, count: usize) -> Result<()> {
        self.finish_reduce(b, g, true)?;
        let local = self.ranks.len();
        for r in 0..local {
            self.resident_moe[r][layer].finish_batch(b, &mut self.moe[g * local + r], &mut self.hidden[g * local + r], count)?;
        }
        Ok(())
    }
    fn output(&mut self, b: &Backend<'a>, group: usize, row: usize, save_logits: bool) -> Result<(u32, Option<Vec<f32>>)> {
        for r in 0..self.ranks.len() {
            b.copy_range(&self.hidden[group * self.ranks.len() + r], &mut self.front[r].residual, row*W, 0, W)?;
            Hc::mixer(&self.model.persistent[r],MIXER)?.mix(b,&mut self.hc[r],&self.front[r].residual)?;
            self.head[r].project(b,self.hc[r].mixed())?;
        }
        b.prefill_collective(|| self.collect_output(b, save_logits))
    }
}
