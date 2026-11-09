//! Shared-embedding, one-layer MTP at checkpoint precision. Draft logits never choose an output:
//! only the canonical target's verified greedy prefix can commit.
use super::*;
include!("hybrid/mtp_proposal.rs");

pub fn validate_tokenizer(target: &Path, draft: &Path) -> Result<()> {
    use engine_model::json::{self, field};
    let tokenizer = |p: &Path| -> Result<json::Value> {
        let bytes = std::fs::read(p.join("tokenizer.json")).map_err(|e| e.to_string())?;
        json::parse(&bytes, json::Limits { bytes: 32 * 1024 * 1024, nodes: 2_000_000, ..Default::default() }).map_err(|e| e.to_string())
    };
    let (a, b) = (tokenizer(target)?, tokenizer(draft)?);
    let (a, b) = (a.object().map_err(|e| e.to_string())?, b.object().map_err(|e| e.to_string())?);
    for name in ["model", "normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        if a.get(name) != b.get(name) { return Err(format!("hybrid draft tokenizer {name} differs from target")); }
    }
    // Only the target tokenizes and renders chats. Extra target audio markers
    // may be absent from the text draft; every shared added ID must agree.
    let target_added = field(a, "added_tokens").and_then(|v| v.array()).map_err(|e| e.to_string())?;
    for token in field(b, "added_tokens").and_then(|v| v.array()).map_err(|e| e.to_string())? {
        let id = field(token.object().map_err(|e| e.to_string())?, "id").map_err(|e| e.to_string())?;
        if !target_added.iter().any(|t| t == token && t.object().is_ok_and(|o| o.get("id") == Some(id))) {
            return Err("hybrid draft added token differs from target".into());
        }
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct Policy {
    pub(super) options: crate::hybrid_draft::Options,
    pub(super) copy: crate::hybrid_draft::CopyIndex,
    copy_cycle: bool,
    pub(super) plain_ms: f64,
    automatic: bool,
    skip_model: bool,
    remaining: usize,
    target_ms: [f64; 9],
    draft_per_token_ms: f64,
    repair_ms: f64,
    successes: [usize; 8], trials: [usize; 8],
    cycles: usize,
    prepared_ms: f64,
    verified_ms: f64,
    pub(super) depths: [usize; 9],
}
impl Policy {
    pub(super) fn configured(options: crate::hybrid_draft::Options) -> Self { Self { options, ..Default::default() } }
    fn more(&self, cumulative: f64, level: usize) -> bool {
        if !self.options.gated_depth || self.cycles < 3 { return true; }
        let depth = self.depths.iter().enumerate().rev().find(|(d,n)| *d>0 && **n>0).map_or(1, |(d,_)|d);
        let (mut expected, mut survival) = (1.0, 1.0);
        for i in 0..depth { survival *= (self.successes[i] as f64+1.0)/(self.trials[i] as f64+2.0); expected += survival; }
        let cost = self.target_ms[depth]+depth as f64*self.draft_per_token_ms+self.repair_ms;
        let row = if self.target_ms[level+1]>0.0 && self.target_ms[level]>0.0 {
            self.target_ms[level+1]-self.target_ms[level]
        } else { self.plain_ms*0.04 };
        crate::hybrid_draft::more(cumulative, if cost>0.0 {expected/cost} else {0.0}, row, self.draft_per_token_ms)
    }
    fn depth(&self, limit: usize) -> usize {
        if self.plain_ms == 0.0 { return 0; }
        if self.cycles < 3 { return limit.min(2); }
        let (mut best, mut rate, mut expected, mut survival) = (0, 1.05 / self.plain_ms, 1.0, 1.0);
        for depth in 1..=limit.min(4) {
            let i = depth - 1;
            let prior = if i == 0 { 0.8 } else { (self.successes[i - 1] as f64 + 1.6) / (self.trials[i - 1] as f64 + 2.0) };
            survival *= (self.successes[i] as f64 + 2.0 * prior) / (self.trials[i] as f64 + 2.0);
            expected += survival;
            let measured = self.target_ms.iter().enumerate().filter(|(_, ms)| **ms > 0.0)
                .min_by_key(|(d, _)| d.abs_diff(depth));
            let target = if let Some((d, ms)) = measured { (ms + (depth as f64 - d as f64) * self.plain_ms * 0.04).max(self.plain_ms) }
                else { self.plain_ms * (1.0 + depth as f64 * 0.04) };
            let score = expected / (target + depth as f64 * self.draft_per_token_ms + self.repair_ms);
            // A wider chain must buy enough speed to justify its extra state.
            if score > rate * if best == 0 { 1.0 } else { 1.03 } { best = depth; rate = score; }
        }
        best
    }
    fn saving_ms(&self, limit: usize) -> f64 {
        if self.cycles < 3 { return self.plain_ms * limit as f64 / (limit + 1) as f64 * self.remaining as f64; }
        let (mut best, mut expected, mut survival) = (0.0_f64, 1.0, 1.0);
        for depth in 1..=limit.min(8) {
            let i = depth - 1;
            survival *= (self.successes[i] as f64 + 1.0) / (self.trials[i] as f64 + 2.0);
            expected += survival;
            let target = self.target_ms[depth].max(self.plain_ms * (1.0 + depth as f64 * 0.04));
            best = best.max(self.plain_ms - (target + depth as f64 * self.draft_per_token_ms + self.repair_ms) / expected);
        }
        best * self.remaining as f64
    }
    fn observe(&mut self, proposed: usize, accepted: usize, repair_ms: f64) {
        if std::mem::take(&mut self.copy_cycle) || proposed == 0 { return; }
        self.cycles += 1;
        self.depths[proposed] += 1;
        let warming = self.cycles <= 2;
        let update = |old: &mut f64, value: f64| { *old = if *old == 0.0 || warming { value } else { *old * 0.75 + value * 0.25 }; };
        update(&mut self.target_ms[proposed], self.verified_ms);
        update(&mut self.draft_per_token_ms, self.prepared_ms / proposed as f64);
        update(&mut self.repair_ms, repair_ms);
        for i in 0..proposed {
            if i <= accepted { self.trials[i] += 1; }
            if i < accepted { self.successes[i] += 1; }
        }
    }
}

pub fn reservation_bytes(c: &Config, capacity: usize, options: crate::hybrid_draft::Options) -> usize {
    // Public verification accepts MAX_ROWS, retaining all earlier recurrent rows.
    let typed = c.fp8 && c.experts > 0;
    ((decode::MAX_ROWS - 1) * c.state_bytes() as usize
        + capacity * (c.kv_heads * c.head_dim * 4) as usize
        + Draft::workspace_bytes(c, 1, typed)
        + capacity * c.query_heads as usize * 4
        + decode::MAX_ROWS * c.hidden as usize * 4 + 64
        + crate::hybrid_draft::confidence_bytes(c.vocab as usize, options)
        // Typed sessions already fund the executor in Model::session_bytes.
        + if typed { 0 } else { decode::Decode::reservation_bytes(c) })
        .saturating_add(crate::hybrid_draft::CopyIndex::reservation_bytes(capacity, options).unwrap_or(usize::MAX))
}

/// Typed draft owns one row; KV replacement and score growth can overlap their predecessors.
pub fn fixed_reservation_bytes(c: &Config, capacity: usize, options: crate::hybrid_draft::Options) -> Option<usize> {
    Model::fixed_decode_bytes(c,decode::MAX_ROWS)?;
    let (h,m,v)=(c.hidden as usize,c.intermediate as usize,c.vocab as usize);
    let lengths=[h,h,2*h,h,h,h,m,m,1,v,v.div_ceil(1024)*3,3,
        16*(2*h).max((c.query_heads*c.head_dim) as usize),6];
    let scratch=lengths.into_iter().map(|n| (n*4).div_ceil(64)*64).sum::<usize>()
        + DenseScratch::reservation_bytes(c,1)+forward::Moe::reservation_bytes(c,1);
    let kv=capacity.checked_mul(c.kv_heads as usize)?.checked_mul(c.head_dim as usize)?.checked_mul(8)?;
    let scores=capacity.checked_mul(c.query_heads as usize)?.checked_mul(8)?;
    scratch.checked_add(kv)?.checked_add(scores)?.checked_add(h.checked_mul(4)?)?.checked_add(4096)?
        .checked_add(crate::hybrid_draft::confidence_bytes(v, options))?
        .checked_add(crate::hybrid_draft::CopyIndex::reservation_bytes(capacity, options)?)
}

pub(super) struct Draft<'a> {
    confidence_scratch: Option<spark_gpu_probe::driver::TopKScratch<'a>>,
    graphs: graph::Graphs<'a>,
    proposal_root: Option<(usize, u32)>,
    last: DeviceVector<'a>,
    embedding: DeviceVector<'a>, fusion: DeviceVector<'a>,
    hidden: DeviceVector<'a>, normalized: DeviceVector<'a>, block: DeviceVector<'a>,
    gate: DeviceVector<'a>, up: DeviceVector<'a>, ids: DeviceVector<'a>, logits: DeviceVector<'a>,
    argmax: DeviceVector<'a>, selected: DeviceVector<'a>,
    attention: DenseScratch<'a>, cache: DenseState<'a>, past: usize,
    scratch: DeviceVector<'a>, descriptors: DeviceVector<'a>, moe: Option<forward::Moe<'a>>,
}
impl<'a> Draft<'a> {
    fn state_bytes(&self, model: &Model<'_, '_>) -> Result<Vec<u8>> {
        let mut bytes=(self.past as u64).to_le_bytes().to_vec();
        bytes.extend(model.backend.read_vector_values(&self.last)?.iter().flat_map(|v|v.to_bits().to_le_bytes()));
        let d=self.cache.descriptor(self.past);
        for at in [0,2] { bytes.extend(state_span(model,&d,at,self.past*(model.config.kv_heads*model.config.head_dim) as usize/2)?); }
        Ok(bytes)
    }
    pub(super) fn new(model: &Model<'a, '_>) -> Result<Self> {
        Self::workspace(model, 1)
    }
    fn lengths(c: &Config, rows: usize, typed: bool) -> [usize; 14] {
        let (h, m) = (c.hidden as usize, c.intermediate as usize);
        [h, rows*h, rows*2*h, rows*h, rows*h, rows*h, rows*m, rows*m,
            rows, rows*c.vocab as usize, rows*(c.vocab as usize).div_ceil(1024)*3, rows*3,
            if typed { rows.max(16)*(2*h).max((c.query_heads*c.head_dim) as usize) } else { 1 }, 6]
    }
    fn workspace_bytes(c: &Config, rows: usize, typed: bool) -> usize {
        Self::lengths(c, rows, typed).iter().map(|n| (n*4).div_ceil(64)*64).sum::<usize>()
            + DenseScratch::reservation_bytes(c, rows)
            + if typed { forward::Moe::reservation_bytes(c, rows) } else { 0 }
    }
    fn workspace(model: &Model<'a, '_>, rows: usize) -> Result<Self> {
        let (b, c) = (model.backend, &model.config);
        let mut v = b.vector_arena(&Self::lengths(c, rows, model.bound.is_some()))?.into_iter();
        Ok(Self { confidence_scratch: None, graphs: Vec::new(), proposal_root: None, last: v.next().unwrap(), embedding: v.next().unwrap(), fusion: v.next().unwrap(),
            hidden: v.next().unwrap(), normalized: v.next().unwrap(), block: v.next().unwrap(),
            gate: v.next().unwrap(), up: v.next().unwrap(), ids: v.next().unwrap(), logits: v.next().unwrap(),
            argmax: v.next().unwrap(), selected: v.next().unwrap(),
            scratch:v.next().unwrap(), descriptors:v.next().unwrap(), moe:forward::Moe::new(model,rows)?,
            attention: DenseScratch::new(b, c, rows)?, cache: DenseState::new(b, c, if rows == 1 { INITIAL_TILE.min(model.capacity) } else { 1 })?, past: 0 })
    }
    fn reserve(&mut self, model: &Model<'a, '_>) -> Result<()> {
        self.reserve_to(model, self.past + 1)
    }
    fn confidence(&mut self, model: &Model<'a, '_>) -> Result<f64> {
        let scores = model.backend.top_k_device(&self.logits, &mut self.confidence_scratch, 1, 16)?
            .pop().ok_or("missing draft confidence")??;
        Ok(crate::hybrid_draft::confidence(&scores.iter().map(|v|v.1).collect::<Vec<_>>()))
    }
    fn reserve_to(&mut self, model: &Model<'a, '_>, required: usize) -> Result<()> {
        let c = &model.config;
        if required > model.capacity { return Err("MTP history exceeds context".into()); }
        if required <= self.cache.capacity(c) { return Ok(()); }
        let mut cache = DenseState::new(model.backend, c, required.next_power_of_two().min(model.capacity))?;
        self.cache.copy_prefix(model.backend, c, &mut cache, self.past)?;
        self.graphs.clear(); self.cache = cache;
        Ok(())
    }
    fn forward(&mut self, model: &Model<'a, '_>, token: u32, hidden: &DeviceVector<'_>, logits: bool) -> Result<Option<u32>> {
        self.proposal_root = None;
        let (b, c) = (model.backend, &model.config);
        if token as u64 >= c.vocab || self.past >= model.capacity || hidden.len() != c.hidden as usize {
            return Err("MTP input outside vocabulary, hidden shape or context".into());
        }
        self.reserve(model)?;
        b.write_vector(&mut self.ids, &[token as f32])?;
        if logits {
            if self.attention.reserve_decode_scores(b, c.query_heads as usize, 1, self.past+1, model.capacity)? { self.graphs.clear(); }
        }
        if model.bound.is_some() {
            b.write_vector(&mut self.descriptors,&self.cache.descriptor(self.past))?;
            let mut graphs = std::mem::take(&mut self.graphs);
            let result = graph::replay(b, &mut graphs, vec![hidden.device_address(), u64::from(logits)], Vec::new(), 1, 1, || {
                self.enqueue_forward(model, hidden, logits)?;
                if logits { b.enqueue_argmax_device(&self.logits, &mut self.argmax, &mut self.selected, 0)?; }
                Ok(())
            });
            self.graphs = graphs; result?;
            self.past += 1;
            if !logits { return Ok(None); }
            if let Some(moe)=&self.moe {
                let (mut ids,rejected)=moe.selections(b,&self.selected,1)?;
                if !rejected.is_empty() { return Err(MOE_ROUTING_ERROR.into()); }
                return ids.remove(0).map(|(id,_)| Some(id));
            }
            return b.read_argmax_results(&self.selected, 1).map(|ids| Some(ids[0].0));
        }
        self.enqueue_forward(model, hidden, logits)?;
        self.past += 1;
        if !logits { return Ok(None); }
        if let Some(moe)=&self.moe { if !moe.rejected(b,1)?.is_empty() { return Err(MOE_ROUTING_ERROR.into()); } }
        b.argmax_device(&self.logits, &mut self.argmax, &mut self.selected).map(|(id, _)| Some(id))
    }
    fn enqueue_forward(&mut self, model: &Model<'a, '_>, hidden: &DeviceVector<'_>, logits: bool) -> Result<()> {
        let (b,c,h)=(model.backend,&model.config,model.config.hidden as usize);
        model.embedding_rows(&self.ids, &mut self.embedding)?;
        // Qwen RMSNorm weights are zero-centered, including all MTP norms.
        Session::normalize(b, model, &self.embedding, model.matrix("mtp.pre_fc_norm_embedding.weight")?.device_address(), &self.fusion.slice(0..h)?, 1)?;
        Session::normalize(b, model, hidden, model.matrix("mtp.pre_fc_norm_hidden.weight")?.device_address(), &self.fusion.slice(h..2 * h)?, 1)?;
        model.mtp_project("mtp.fc.weight", &self.fusion, &mut self.hidden, &mut self.scratch)?;
        let p = "mtp.layers.0";
        Session::normalize(b, model, &self.hidden, model.matrix(&format!("{p}.input_layernorm.weight"))?.device_address(), &self.normalized, 1)?;
        let mut reuse = false;
        for (name, out) in [("q_proj", &mut self.attention.qg), ("k_proj", &mut self.attention.k), ("v_proj", &mut self.attention.v)] {
            if !logits && name == "q_proj" { continue; }
            let name=format!("{p}.self_attn.{name}.weight");
            model.mtp_project_reusing(&name, &self.normalized, out, &mut self.scratch, reuse)?;
            reuse=model.bound.is_some() && model.matrix(&name).is_ok_and(Matrix::is_fp8);
        }
        // With one MTP layer only its projected KV feeds future predictions.
        if !logits {
            self.cache.append_replaying(b, c, &self.attention.k, &self.attention.v,
                model.address(&format!("{p}.self_attn.k_norm.weight"))?,
                model.frequencies.device_address(), 1, self.past,
                if model.bound.is_some() { self.descriptors.device_address()+16 } else { 0 })?;
            return Ok(());
        }
        // Long MTP histories take the staged dense decode path, as target layers do.
        let norms = [model.address(&format!("{p}.self_attn.q_norm.weight"))?, model.address(&format!("{p}.self_attn.k_norm.weight"))?];
        if model.bound.is_some() {
            self.attention.attend_batch(b,c,norms,model.frequencies.device_address(),&self.descriptors,1,0)?;
        } else { self.attention.attend(b, c, &mut self.cache,
            [model.address(&format!("{p}.self_attn.q_norm.weight"))?, model.address(&format!("{p}.self_attn.k_norm.weight"))?],
            model.frequencies.device_address(), 1, self.past, 0)?; }
        model.mtp_project(&format!("{p}.self_attn.o_proj.weight"), &self.attention.gated, &mut self.block, &mut self.scratch)?;
        model.reduce_partial(tp::Step::Mixer(model.layer_place(c.layers.len())),1,&mut self.block)?;
        launch_norm(b, &self.hidden, Some(&self.block), model.matrix(&format!("{p}.post_attention_layernorm.weight"))?.device_address(),
            &self.normalized, model, 1, Some(2), false)?;
        if let Some(moe)=&mut self.moe {
            moe.run(model,c.layers.len(),&self.normalized,&mut self.block,&mut self.scratch,1)?;
        } else {
        let gate = model.matrix(&format!("{p}.mlp.gate_proj.weight"))?;
        let up = model.matrix(&format!("{p}.mlp.up_proj.weight"))?;
        if gate.is_fp8() || up.is_fp8() {
            b.matvec_bf16_device(gate, &self.normalized, &mut self.gate)?;
            b.matvec_bf16_device(up, &self.normalized, &mut self.up)?;
            launch_silu(b, &self.gate, &self.up)?;
        } else {
            b.gate_up_swiglu_bf16_device(gate, up, &self.normalized, &mut self.gate, &mut self.up)?;
        }
        b.matvec_bf16_device(model.matrix(&format!("{p}.mlp.down_proj.weight"))?, &self.gate, &mut self.block)?;
        }
        launch(b, "flash_residual_bf16", h, &mut [Ptr(self.hidden.device_address()), Ptr(self.block.device_address()), U32(h as u32)])?;
        Session::normalize(b, model, &self.hidden, model.matrix("mtp.norm.weight")?.device_address(), &self.normalized, 1)?;
        if model.bound.is_some() { model.head_rows(&self.normalized,&mut self.logits,&mut self.scratch,1)?; }
        else { model.mtp_project("lm_head.weight", &self.normalized, &mut self.logits, &mut self.scratch)?; }
        Ok(())
    }
    fn observe_prompt(&mut self, model: &Model<'a, '_>, position: usize,
        tokens: &[u32], hidden: &DeviceVector<'_>) -> Result<()> {
        self.proposal_root = None;
        let (b, c) = (model.backend, &model.config);
        let (h, kv) = (c.hidden as usize, (c.kv_heads * c.head_dim) as usize);
        if model.bound.is_some() {
            for (i,&token) in tokens.iter().enumerate() {
                let pos=position+i;
                if pos>0 { self.past=pos-1; self.forward(model,token,&self.last.slice(0..h)?,false)?; }
                Session::normalize(b,model,&hidden.slice(i*h..(i+1)*h)?,model.address(&format!("{PREFIX}.norm.weight"))?,&self.last,1)?;
            }
            return Ok(());
        }
        // Row zero has no preceding target state. Every later row combines
        // token[i] with norm(target_hidden[i-1]), including across prefill tiles.
        let first = usize::from(position == 0);
        self.past = position.saturating_sub(1);
        self.reserve_to(model, position + tokens.len() - 1)?;
        let tile = (tokens.len() - first).min(64);
        if tile > 0 {
            let mut vectors = b.vector_arena(&[tile, tile * h, tile * h, tile * 2 * h,
                tile * h, tile * kv, tile * kv])?.into_iter();
            let (mut ids, mut embedding, mut previous, mut fusion, mut projected, mut k, mut v) =
                (vectors.next().unwrap(), vectors.next().unwrap(), vectors.next().unwrap(),
                 vectors.next().unwrap(), vectors.next().unwrap(), vectors.next().unwrap(), vectors.next().unwrap());
            for start in (first..tokens.len()).step_by(tile) {
                let count = tile.min(tokens.len() - start);
                ids.resize_within_capacity(count)?;
                for x in [&mut embedding, &mut previous, &mut projected] { x.resize_within_capacity(count * h)?; }
                fusion.resize_within_capacity(count * 2 * h)?;
                for x in [&mut k, &mut v] { x.resize_within_capacity(count * kv)?; }
                b.write_vector(&mut ids, &tokens[start..start + count].iter().map(|&t| t as f32).collect::<Vec<_>>())?;
                model.embedding_rows(&ids, &mut embedding)?;
                let offset = usize::from(start == 0);
                if offset == 1 { b.copy_vector_range(&self.last, &mut previous, 0, 0, h)?; }
                if count > offset {
                    Session::normalize(b, model, &hidden.slice((start + offset - 1) * h..(start + count - 1) * h)?,
                        model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(),
                        &previous.slice(offset * h..count * h)?, count - offset)?;
                }
                for (x, name) in [(&embedding, "embedding"), (&previous, "hidden")] {
                    Session::normalize(b, model, x, model.matrix(&format!("mtp.pre_fc_norm_{name}.weight"))?.device_address(), x, count)?;
                }
                b.copy_strided(&embedding, &mut fusion, 0, 0, h, 2 * h, h, count)?;
                b.copy_strided(&previous, &mut fusion, 0, h, h, 2 * h, h, count)?;
                // These shared projections retain the scalar packed8 sum order.
                b.matmul_decode_bf16_device(model.matrix("mtp.fc.weight")?, &fusion, &mut projected, count)?;
                let p = "mtp.layers.0";
                Session::normalize(b, model, &projected, model.matrix(&format!("{p}.input_layernorm.weight"))?.device_address(), &projected, count)?;
                for (name, out) in [("k_proj", &mut k), ("v_proj", &mut v)] {
                    b.matmul_decode_bf16_device(model.matrix(&format!("{p}.self_attn.{name}.weight"))?, &projected, out, count)?;
                }
                self.cache.append(b, c, &k, &v, model.address(&format!("{p}.self_attn.k_norm.weight"))?,
                    model.frequencies.device_address(), count, position + start - 1)?;
            }
        }
        self.past = position + tokens.len() - 1;
        Session::normalize(b, model, &hidden.slice((tokens.len() - 1) * h..tokens.len() * h)?,
            model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(), &self.last, 1)
    }
    /// History KV plus attention scratch, whose score rows grow with the history.
    fn bytes(&self) -> usize {
        let mut owners=Vec::new();
        if let Some(s) = &self.confidence_scratch { s.allocations(&mut owners); }
        for v in [&self.last,&self.embedding,&self.fusion,&self.hidden,&self.normalized,&self.block,&self.gate,&self.up,&self.ids,&self.logits,&self.argmax,&self.selected,&self.scratch,&self.descriptors] { v.allocations(&mut owners); }
        if let Some(moe)=&self.moe { moe.allocations(&mut owners); }
        owners.sort_unstable(); owners.dedup();
        self.cache.device_bytes()+self.attention.device_bytes()+owners.iter().map(|v|v.1).sum::<usize>()
    }
    /// Prefix-cache forks keep the proposal history, so a cached turn drafts like a cold one.
    /// None when the history is not in step with the target's `position`.
    pub(super) fn fork(&self, model: &Model<'a, '_>, position: usize) -> Result<Option<Self>> {
        if self.past + 1 != position.max(1) { return Ok(None); }
        let mut next = Self::new(model)?;
        next.reserve_to(model, self.past + 1)?;
        self.cache.copy_prefix(model.backend, &model.config, &mut next.cache, self.past)?;
        model.backend.copy_vector_range(&self.last, &mut next.last, 0, 0, self.last.len())?;
        next.past = self.past;
        Ok(Some(next))
    }
    /// Bound for a one-row fork: power-of-two history KV plus its workspace.
    pub(super) fn fork_bytes(&self, model: &Model<'_, '_>) -> usize {
        let c = &model.config;
        (self.past + 1).next_power_of_two().max(INITIAL_TILE).min(model.capacity) * (c.kv_heads * c.head_dim) as usize * 4
            + Self::workspace_bytes(c, 1, model.bound.is_some()) + crate::hybrid_draft::confidence_bytes(c.vocab as usize, model.draft_options)
    }
}

impl Model<'_, '_> {
    fn mtp_project(&self, name:&str, input:&DeviceVector<'_>, output:&mut DeviceVector<'_>, scratch:&mut DeviceVector<'_>) -> Result<()> {
        self.mtp_project_reusing(name,input,output,scratch,false)
    }
    fn mtp_project_reusing(&self, name:&str, input:&DeviceVector<'_>, output:&mut DeviceVector<'_>, scratch:&mut DeviceVector<'_>, reuse:bool) -> Result<()> {
        if self.bound.is_some() { self.fixed_project_reusing(name,input,output,scratch,1,reuse) }
        else { self.backend.matvec_bf16_device(self.matrix(name)?,input,output) }
    }
}

pub struct DraftRow<'s, 'm, 'a, 'b> {
    pub session: &'s mut Session<'m, 'a, 'b>,
    pub token: u32,
}
/// Reuse the shared embedding/head weights once at each draft depth.
pub struct BatchDraft<'m, 'a, 'b> {
    model: &'m Model<'a, 'b>,
    work: Option<(Draft<'a>, DeviceVector<'a>)>,
}
impl<'m, 'a, 'b> BatchDraft<'m, 'a, 'b> {
    pub fn new(model: &'m Model<'a, 'b>) -> Result<Self> {
        Ok(Self { model, work: if model.bound.is_some() { None } else {
            Some((Draft::workspace(model, decode::MAX_ROWS)?, model.backend.vector(decode::MAX_ROWS * 6)?))
        } })
    }
    pub fn propose(&mut self, rows: &mut [DraftRow<'_, '_, 'a, 'b>]) -> Result<Vec<Option<Vec<u32>>>> {
        let model = self.model;
        let (b, c) = (model.backend, &model.config);
        let h = c.hidden as usize;
        if rows.len() > decode::MAX_ROWS || rows.iter().any(|r| !std::ptr::eq(r.session.model, model)
            || r.token as u64 >= c.vocab || !r.session.valid) { return Err("invalid shared MTP draft cohort".into()); }
        if model.bound.is_some() {
            return rows.iter_mut().map(|r| match r.session.pending_draft.take() {
                None=>Ok(None), Some(0)=>Ok(Some(vec![r.token])), Some(depth)=> {
                    let mut tokens=vec![r.token]; tokens.extend(r.session.propose_mtp_tokens(r.token,depth)?); Ok(Some(tokens))
                }
            }).collect();
        }
        let cursors: Vec<_> = rows.iter().map(|r| r.session.mtp.as_ref().map(|m| m.past)).collect();
        let result = (|| {
        let (w, device_rows) = self.work.as_mut().ok_or("shared MTP workspace absent")?;
        let start = std::time::Instant::now();
        let depths: Vec<_> = rows.iter_mut().map(|r| r.session.pending_draft.take()).collect();
        let mut output: Vec<_> = rows.iter().zip(&depths).map(|(r, depth)| depth.map(|_| vec![r.token])).collect();
        let mut previous = vec![0; rows.len()];
        for step in 0..depths.iter().flatten().copied().max().unwrap_or(0) {
            let active: Vec<_> = depths.iter().enumerate().filter_map(|(i, depth)| depth.filter(|&n| step < n).map(|_| i)).collect();
            let count = active.len();
            for v in [&mut w.embedding, &mut w.hidden, &mut w.block] { v.resize_within_capacity(count * h)?; }
            w.fusion.resize_within_capacity(count * 2 * h)?;
            w.ids.resize_within_capacity(count)?;
            w.gate.resize_within_capacity(count * c.intermediate as usize)?;
            w.up.resize_within_capacity(count * c.intermediate as usize)?;
            w.logits.resize_within_capacity(count * c.vocab as usize)?;
            w.attention.resize(c, count)?;
            let mut descriptors = Vec::with_capacity(count * 6);
            let mut longest = 0;
            for (slot, &chat) in active.iter().enumerate() {
                let s = &mut rows[chat].session;
                let mtp = s.mtp.as_mut().ok_or("shared MTP requires loaded draft weights")?;
                if step == 0 && mtp.past + 1 != s.position { return Err("MTP cache does not match target prefix".into()); }
                if s.position + depths[chat].unwrap() + 1 > model.capacity { return Err("MTP proposal exceeds target context".into()); }
                mtp.proposal_root = None;
                mtp.reserve(model)?;
                if step == 0 { b.copy_vector_range(&mtp.last, &mut w.hidden, 0, slot * h, h)?; }
                else { b.copy_vector_range(&w.normalized, &mut w.hidden, previous[chat] * h, slot * h, h)?; }
                descriptors.extend_from_slice(&mtp.cache.descriptor(mtp.past));
                longest = longest.max(mtp.past + 1);
            }
            w.attention.reserve_decode_scores(b, c.query_heads as usize, count, longest, model.capacity)?;
            w.normalized.resize_within_capacity(count * h)?;
            device_rows.resize_within_capacity(count * 6)?;
            b.write_vector(device_rows, &descriptors)?;
            b.write_vector(&mut w.ids, &active.iter().map(|&i| *output[i].as_ref().unwrap().last().unwrap() as f32).collect::<Vec<_>>())?;
            model.embedding_rows(&w.ids, &mut w.embedding)?;
            Session::normalize(b, model, &w.embedding, model.matrix("mtp.pre_fc_norm_embedding.weight")?.device_address(), &w.embedding, count)?;
            Session::normalize(b, model, &w.hidden, model.matrix("mtp.pre_fc_norm_hidden.weight")?.device_address(), &w.hidden, count)?;
            for slot in 0..count {
                b.copy_vector_range(&w.embedding, &mut w.fusion, slot * h, slot * 2 * h, h)?;
                b.copy_vector_range(&w.hidden, &mut w.fusion, slot * h, slot * 2 * h + h, h)?;
            }
            b.matmul_decode_bf16_device(model.matrix("mtp.fc.weight")?, &w.fusion, &mut w.hidden, count)?;
            let p = "mtp.layers.0";
            Session::normalize(b, model, &w.hidden, model.matrix(&format!("{p}.input_layernorm.weight"))?.device_address(), &w.normalized, count)?;
            for (name, out) in [("q_proj", &mut w.attention.qg), ("k_proj", &mut w.attention.k), ("v_proj", &mut w.attention.v)] {
                b.matmul_decode_bf16_device(model.matrix(&format!("{p}.self_attn.{name}.weight"))?, &w.normalized, out, count)?;
            }
            w.attention.attend_batch(b, c, [model.address(&format!("{p}.self_attn.q_norm.weight"))?,
                model.address(&format!("{p}.self_attn.k_norm.weight"))?], model.frequencies.device_address(), device_rows, count, 0)?;
            b.matmul_decode_bf16_device(model.matrix(&format!("{p}.self_attn.o_proj.weight"))?, &w.attention.gated, &mut w.block, count)?;
            launch_norm(b, &w.hidden, Some(&w.block), model.matrix(&format!("{p}.post_attention_layernorm.weight"))?.device_address(),
                &w.normalized, model, count, Some(2), false)?;
            for (name, out) in [("gate_proj", &mut w.gate), ("up_proj", &mut w.up)] {
                b.matmul_decode_bf16_device(model.matrix(&format!("{p}.mlp.{name}.weight"))?, &w.normalized, out, count)?;
            }
            launch_silu(b, &w.gate, &w.up)?;
            b.matmul_decode_bf16_device(model.matrix(&format!("{p}.mlp.down_proj.weight"))?, &w.gate, &mut w.block, count)?;
            launch(b, "flash_residual_bf16", count * h, &mut [Ptr(w.hidden.device_address()), Ptr(w.block.device_address()), U32((count * h) as u32)])?;
            Session::normalize(b, model, &w.hidden, model.matrix("mtp.norm.weight")?.device_address(), &w.normalized, count)?;
            b.matmul_decode_bf16_device(model.matrix("lm_head.weight")?, &w.normalized, &mut w.logits, count)?;
            w.argmax.resize_within_capacity(count * (c.vocab as usize).div_ceil(1024) * 3)?;
            w.selected.resize_within_capacity(count * 3)?;
            let guesses = b.argmax_batch_device(&w.logits, &mut w.argmax, &mut w.selected, count)?;
            for ((slot, &chat), guess) in active.iter().enumerate().zip(guesses) {
                output[chat].as_mut().unwrap().push(guess?.0);
                rows[chat].session.mtp.as_mut().unwrap().past += 1;
                previous[chat] = slot;
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        for (r, depth) in rows.iter_mut().zip(depths) { if depth.is_some() { r.session.draft_policy.prepared_ms = ms; } }
        Ok(output)
        })();
        for (row, past) in rows.iter_mut().zip(cursors) {
            if let (Some(mtp), Some(past)) = (&mut row.session.mtp, past) { mtp.past = past; }
        }
        result
    }
}

impl<'m, 'a, 'b> Session<'m, 'a, 'b> {
    /// Once a request joins a plain cohort, release draft state rather than
    /// rebuilding its entire prefix if the other chats finish first.
    pub fn disable_model_draft(&mut self) {
        self.prefill_tail.extend_from_slice(&self.verify_tokens);
        self.mtp = None; self.external = None; self.batch_drafts = false;
        self.pending_draft = None; self.verify_tokens.clear();
        self.rollback_state.clear(); self.verify_hidden = None; self.verifier = None;
    }
    /// Gate eager MTP history before prefill, then use request-local decode costs.
    pub fn automatic_draft(&mut self, prompt: usize, remaining: usize) {
        if self.model.bound.is_some() { self.draft_policy.remaining=remaining; return; }
        self.draft_policy.automatic = true;
        self.draft_policy.remaining = remaining;
        // GB10 seeds: 27B plain ~224 ms short, ~244 ms at 28K (staged dense
        // attention); eager MTP history ~0.04 ms/token, 0.8B prefill ~0.08 ms/token.
        self.draft_policy.plain_ms = 224.0 + prompt as f64 * 0.0007;
        let limit = if self.draft_depth == 0 { 4 } else { self.draft_depth };
        let missing = self.external.as_ref().map_or(prompt.saturating_sub(self.position),
            |d| prompt.saturating_sub(d.session.position));
        let per_token = self.external.as_ref().map_or(0.04, |d| d.catchup_per_token_ms);
        if missing > 64 && missing as f64 * per_token > self.draft_policy.saving_ms(limit) {
            self.disable_model_draft();
        }
    }
    pub fn enable_batch_drafts(&mut self) { self.batch_drafts = self.mtp.is_some() && self.external.is_none(); }
    pub(super) fn record_batch_time(&mut self, rows: usize, ms: f64, plain: bool) {
        if rows == 1 && plain { self.draft_policy.plain_ms = ms; }
        else { self.draft_policy.verified_ms = ms; }
    }
    pub fn draft_depth_cycles(&self) -> [usize; 9] { self.draft_policy.depths }
    pub fn with_draft_model(mut self, model: &'m Model<'a, 'b>, depth: usize) -> Result<Self> {
        self.restore_model_draft(model, depth)?;
        Ok(self)
    }
    /// Cache snapshots may release proposal state. Catchup is still governed
    /// by the same request-local policy as a cold session.
    pub fn restore_model_draft(&mut self, model: &'m Model<'a, 'b>, depth: usize) -> Result<()> {
        if self.external.is_some() { return Ok(()); }
        let (a, b) = (&self.model.config, &model.config);
        if a.vocab != b.vocab || a.eos_token != b.eos_token || a.bos_token != b.bos_token
            || !std::ptr::eq(self.model.backend, model.backend) {
            return Err("hybrid draft requires the same backend, vocabulary and special tokens".into());
        }
        if depth > 8 { return Err("external draft depth must be 0 (adaptive) or 1..=8".into()); }
        self.draft_depth = depth;
        self.external = Some(Box::new(External { session: model.session()?, saved: Vec::new(), entry: 0, catchup_per_token_ms: 0.08 }));
        Ok(())
    }
    pub fn mtp_cache_bytes(&self) -> usize { self.mtp.as_ref().map_or(0, Draft::bytes) }
    pub fn propose_mtp(&mut self, token: u32) -> Result<u32> {
        self.propose_mtp_tokens(token, 1).map(|v| v[0])
    }
    pub(super) fn mtp_observe(&mut self, tokens: &[u32]) -> Result<()> {
        let hidden = self.hidden.slice(0..tokens.len() * self.model.config.hidden as usize)?;
        self.mtp_observe_hidden(tokens, &hidden)
    }
    pub(super) fn save_prefill_mtp(&mut self) -> Result<()> {
        if let Some(mtp) = &self.mtp {
            let b = self.model.backend;
            let mut last = b.vector(self.model.config.hidden as usize)?;
            b.copy_vector_range(&mtp.last, &mut last, 0, 0, mtp.last.len())?;
            self.prefill_mtp = Some(last);
        }
        Ok(())
    }
    pub(super) fn restore_prefill_mtp(&mut self) -> Result<()> {
        if let Some(mtp) = &mut self.mtp {
            if let Some(last) = &self.prefill_mtp {
                self.model.backend.copy_vector_range(last, &mut mtp.last, 0, 0, last.len())?;
            }
            mtp.proposal_root = None;
            mtp.past = self.position.saturating_sub(1);
        }
        Ok(())
    }
    pub(super) fn mtp_observe_hidden(&mut self, tokens: &[u32], hidden: &DeviceVector<'_>) -> Result<()> {
        let model = self.model;
        let (b, c) = (model.backend, &model.config);
        let mtp = self.mtp.as_mut().unwrap();
        if tokens.len() > 1 { return mtp.observe_prompt(model, self.position, tokens, hidden); }
        let h = c.hidden as usize;
        for (i, &token) in tokens.iter().enumerate() {
            let pos = self.position + i;
            if pos > 0 {
                let last = mtp.last.slice(0..h)?;
                // A proposal may already have written this exact MTP row.
                mtp.past = pos - 1;
                mtp.forward(model, token, &last, false)?;
            }
            Session::normalize(b, model, &hidden.slice(i * h..(i + 1) * h)?,
                model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(), &mtp.last, 1)?;
        }
        Ok(())
    }
    pub fn verify_mtp(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.finish_mtp_verification()?;
        self.begin_decode()?;
        if tokens.len() == 1 { return self.step(tokens[0]).map(|v| vec![v]); }
        let mut verifier = match self.verifier.take() {
            Some(v) => v, None => Box::new(decode::Decode::with_rows(self.model,if self.model.fixed_rows() { self.decode_rows() } else { decode::MAX_ROWS })?),
        };
        let start = std::time::Instant::now();
        let result = verifier.verify(self, tokens);
        self.draft_policy.verified_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.verifier = Some(verifier);
        result
    }
    /// Qualification receipt for committed target, in-step draft and prompt replay owners.
    /// Same-segmentation comparisons keep these potentially large bytes on the test machine.
    pub fn committed_state_bytes(&mut self) -> Result<Vec<u8>> {
        if self.external.is_some() { return Err("external draft state receipt unavailable".into()); }
        let mut bytes=self.target_state_bytes()?;
        let entry=self.replay_entry()?;
        if self.mtp.as_ref().is_some_and(|d|self.position==0 || d.past.checked_add(1)!=Some(self.position)) {
            return Err("draft state is not at the committed prefix".into());
        }
        if self.mtp.is_some() && entry!=0 && !self.prefill_tail.is_empty() && self.prefill_mtp.is_none() {
            return Err("MTP replay hidden row absent".into());
        }
        let model=self.model;let c=&model.config;
        bytes.extend((self.prefill_tail.len() as u64).to_le_bytes());
        for &token in &self.prefill_tail { bytes.extend(token.to_le_bytes()); }
        bytes.push(u8::from(self.mtp.is_some()));
        if let Some(draft)=&self.mtp { bytes.extend(draft.state_bytes(model)?); }
        bytes.extend((self.prefill_saved.len() as u64).to_le_bytes());
        for saved in &self.prefill_saved {
            bytes.push(u8::from(saved.is_some()));
            if let Some(saved)=saved {let d=saved.descriptor();
                bytes.extend(state_span(model,&d,0,(c.linear_value_heads*c.linear_key_dim*c.linear_value_dim) as usize)?);
                bytes.extend(state_span(model,&d,2,((2*c.linear_key_heads*c.linear_key_dim+c.linear_value_heads*c.linear_value_dim)*c.linear_conv/2) as usize)?);
            }
        }
        bytes.push(u8::from(self.prefill_mtp.is_some()));
        if let Some(last)=&self.prefill_mtp { bytes.extend(model.backend.read_vector_values(last)?.iter().flat_map(|v|v.to_bits().to_le_bytes())); }
        Ok(bytes)
    }
    pub(super) fn finish_mtp_verification(&mut self) -> Result<()> {
        if self.verify_tokens.is_empty() { return Ok(()); }
        self.commit_mtp_prefix(self.position)
    }
    fn commit_mtp_prefix(&mut self, position: usize) -> Result<()> {
        let start = std::time::Instant::now();
        let (model, entry) = (self.model, self.verify_entry);
        if !self.valid {
            return Err("MTP can only restore a verified consecutive prefix; drop session after an error".into());
        }
        let (b, c) = (model.backend, &model.config);
        let kept = mtp_contract::verified_prefix(entry,self.verify_tokens.len(),position)?;
        if kept < self.verify_tokens.len() {
            let saved=self.rollback_state.get(kept-1).ok_or("missing verified target snapshot")?;
            if saved.len()!=self.gdn_cache.len() || saved.iter().zip(&self.gdn_cache).any(|(a,b)| a.is_some()!=b.is_some()) {
                return Err("incomplete verified target snapshot".into());
            }
            self.valid=false;
            for (saved, live) in saved.iter().zip(&mut self.gdn_cache) {
                if let (Some(saved), Some(live)) = (saved, live) { saved.copy_into(b, live)?; }
            }
        }
        self.valid=false;
        if let Some(mtp) = self.mtp.as_mut() {
            let h = c.hidden as usize;
            let normalized = self.verify_hidden.as_ref().ok_or("missing verified target hidden rows")?;
            // Repair shifted history using only verified target hidden rows.
            let root = mtp.proposal_root.take();
            if mtp.past < entry && root != Some((entry - 1, self.verify_tokens[0])) {
                let last = mtp.last.slice(0..h)?;
                mtp.past = entry - 1;
                mtp.forward(model, self.verify_tokens[0], &last, false)?;
            }
            for i in 1..kept {
                mtp.past = entry + i - 1;
                mtp.forward(model, self.verify_tokens[i], &normalized.slice((i - 1) * h..i * h)?, false)?;
            }
            mtp.past = position - 1;
            b.copy_vector_range(normalized, &mut mtp.last, (kept - 1) * h, 0, h)?;
        }
        self.draft_policy.copy.truncate(position);
        self.position = position;
        self.prefill_tail.extend_from_slice(&self.verify_tokens[..kept]);
        if let Some(draft) = self.external.as_mut() { draft.commit(position)?; }
        self.draft_policy.observe(self.verify_tokens.len() - 1, kept - 1, start.elapsed().as_secs_f64() * 1000.0);
        self.verify_tokens.clear();
        self.valid=true;
        Ok(())
    }
}

fn state_span(model:&Model<'_, '_>, descriptor:&[f32], at:usize, words:usize)->Result<Vec<u8>> {
    if words==0 {return Ok(Vec::new());}
    let b=model.backend;let copy=b.vector(words)?;
    let source=u64::from(descriptor[at].to_bits())|u64::from(descriptor[at+1].to_bits())<<32;
    let mut p=spark_gpu_probe::kernel_params::HybridCopy {source:source as _,destination:copy.device_address() as _,
        words:u32::try_from(words).map_err(|_|"state receipt span exceeds copy ABI")?,padding:0,rows:0};
    unsafe {b.launch_flash_rows("flash_hybrid_copy",words.div_ceil(128) as u32,1,128,0,&mut[(&mut p as *mut spark_gpu_probe::kernel_params::HybridCopy).cast()])?;}
    Ok(b.read_vector_values(&copy)?.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect())
}

impl crate::draft::Target for Session<'_, '_, '_> {
    fn checkpoint(&self) -> usize { self.position }
    fn truncate(&mut self, position: usize) -> Result<()> {
        if !self.verify_tokens.is_empty() { self.commit_mtp_prefix(position) }
        else if self.valid && position == self.position { Ok(()) }
        else { Err("hybrid rollback requires a retained MTP verification prefix".into()) }
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> { Session::step(self, token) }
    fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> { self.verify_mtp(tokens) }
    fn kv_bf16(&self) -> bool { true }
    fn max_verify_rows(&self) -> usize { if self.external.is_some() { 9 } else { 5 } }
    fn native_draft_policy(&self) -> bool { true }
    fn model_draft_budget(&mut self, remaining: usize) -> Result<()> {
        self.draft_policy.remaining = remaining;
        if self.model.bound.is_some() { return Ok(()); }
        if !self.draft_policy.automatic || self.draft_policy.skip_model { return Ok(()); }
        self.finish_mtp_verification()?;
        let limit = if self.draft_depth == 0 { 4 } else { self.draft_depth };
        let missing = self.external.as_ref().map_or(0, |d| self.position.saturating_sub(d.session.position));
        let catchup_ms = self.external.as_ref().map_or(0.0, |d| missing as f64 * d.catchup_per_token_ms);
        let saving = self.draft_policy.saving_ms(limit.min(remaining.saturating_sub(1)));
        if saving == 0.0 || missing > 64 && catchup_ms > saving {
            self.disable_model_draft();
            // Keep the active speculator on its canonical single-row path.
            self.draft_policy.skip_model = true;
        }
        Ok(())
    }
    fn model_draft_width(&self) -> usize { if self.draft_policy.skip_model { 1 } else if self.mtp.is_some() || self.external.is_some() { if self.draft_depth == 0 { 5 } else { self.draft_depth + 1 } } else { 0 } }
    fn plan_model_draft(&mut self, history: &[u32], limit: usize) -> Result<Option<usize>> {
        if self.external.is_none() && (self.draft_policy.options.copy_first || self.draft_policy.options.gated_depth) { return Ok(None); }
        if !self.batch_drafts { return Ok(None); }
        if history.len() != self.position { return Err("MTP history does not match target position".into()); }
        self.finish_mtp_verification()?;
        let limit = if self.draft_depth == 0 { limit } else { limit.min(self.draft_depth) };
        let depth = if self.model.bound.is_some() { mtp_contract::depth(self.draft_depth,limit) } else if self.draft_depth == 0 || self.draft_policy.automatic && self.draft_policy.cycles >= 3 { self.draft_policy.depth(limit) } else { limit };
        self.pending_draft = Some(depth);
        if depth == 0 { self.draft_policy.depths[0] += 1; }
        Ok(Some(depth))
    }
    fn propose_model(&mut self, history: &[u32], next: u32, limit: usize) -> Result<Option<crate::draft::ModelProposal>> {
        if history.len() != self.position { return Err("MTP history does not match target position".into()); }
        if limit == 0 || self.draft_policy.skip_model { return Ok(Some(crate::draft::ModelProposal { tokens: Vec::new(), proposal_ms: 0.0, catchup_ms: 0.0 })); }
        self.finish_mtp_verification()?;
        self.draft_policy.copy_cycle = false;
        let limit = if self.draft_depth == 0 { limit } else { limit.min(self.draft_depth) };
        if self.mtp.is_some() && self.external.is_none() && self.draft_policy.options.copy_first {
            let start = Instant::now();
            let tokens = self.draft_policy.copy.propose(history, next, limit.min(4));
            if !tokens.is_empty() {
                self.draft_policy.copy_cycle = true;
                self.draft_policy.prepared_ms = start.elapsed().as_secs_f64()*1000.0;
                return Ok(Some(crate::draft::ModelProposal { tokens, proposal_ms: self.draft_policy.prepared_ms, catchup_ms: 0.0 }));
            }
        }
        let depth = if self.model.bound.is_some() { mtp_contract::depth(self.draft_depth,limit) } else if self.draft_depth == 0 || self.draft_policy.automatic && self.draft_policy.cycles >= 3 { self.draft_policy.depth(limit) } else { limit };
        if depth == 0 { self.draft_policy.depths[0] += 1; return Ok(Some(crate::draft::ModelProposal { tokens: Vec::new(), proposal_ms: 0.0, catchup_ms: 0.0 })); }
        let start = std::time::Instant::now();
        let seed = if self.external.is_some() && self.mtp.is_some() { self.propose_mtp_tokens(next, 1)? } else { Vec::new() };
        let result = if let Some(draft) = self.external.as_mut() { draft.propose(history, next, depth, &seed) }
            else { self.propose_mtp_tokens(next, depth) };
        self.draft_policy.prepared_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.map(|tokens| Some(crate::draft::ModelProposal { tokens, proposal_ms: self.draft_policy.prepared_ms, catchup_ms: 0.0 }))
    }
}

/// The small hybrid model keeps only recurrent prefix copies and its own BF16 KV.
pub(super) struct External<'m, 'a, 'b> {
    session: Session<'m, 'a, 'b>,
    saved: Vec<Vec<Option<GdnState<'a>>>>,
    entry: usize,
    catchup_per_token_ms: f64,
}
impl<'m, 'a, 'b> External<'m, 'a, 'b> {
    pub(super) fn bytes(&self) -> usize {
        self.session.cache_bytes() + self.saved.iter().flatten().flatten().map(GdnState::device_bytes).sum::<usize>()
    }
    fn propose(&mut self, history: &[u32], mut next: u32, depth: usize, seed: &[u32]) -> Result<Vec<u32>> {
        if self.session.position > history.len() { return Err("external draft prefix was not committed".into()); }
        if self.session.position < history.len() {
            let missing = history.len() - self.session.position;
            let start = std::time::Instant::now();
            if missing > 64 || self.session.position == 0 {
                self.session.prefill_advance(&history[self.session.position..])?;
            } else {
                for &token in &history[self.session.position..] { self.session.advance(token)?; }
            }
            if missing > 64 { self.catchup_per_token_ms = start.elapsed().as_secs_f64() * 1000.0 / missing as f64; }
        }
        let (b, c) = (self.session.model.backend, &self.session.model.config);
        while self.saved.len() < depth { self.saved.push(GdnState::dense_layers(b, c)?); }
        self.entry = history.len();
        let mut guesses = Vec::with_capacity(depth);
        for i in 0..depth {
            let logits = self.session.step(next)?;
            next = if let Some(&token) = seed.get(i) { token } else { crate::greedy_token(&logits)? };
            for (live, saved) in self.session.gdn_cache.iter().zip(&mut self.saved[i]) {
                if let (Some(live), Some(saved)) = (live, saved) { live.copy_into(b, saved)?; }
            }
            guesses.push(next);
        }
        Ok(guesses)
    }
    fn commit(&mut self, position: usize) -> Result<()> {
        if self.session.position <= position { return Ok(()); }
        let kept = position.checked_sub(self.entry).filter(|&n| n > 0).ok_or("invalid external draft rollback")?;
        let saved = self.saved.get(kept - 1).ok_or("missing external draft prefix")?;
        let tail = self.session.prefill_tail.len().checked_sub(self.session.position - position)
            .ok_or("external draft rollback exceeds retained prefix")?;
        for (saved, live) in saved.iter().zip(&mut self.session.gdn_cache) {
            if let (Some(saved), Some(live)) = (saved, live) { saved.copy_into(self.session.model.backend, live)?; }
        }
        // Rejected proposal tokens must not enter a later catchup replay.
        self.session.prefill_tail.truncate(tail);
        self.session.position = position;
        Ok(())
    }
}
