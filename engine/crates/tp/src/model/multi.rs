//! Shared projection workspaces and execution across independent sessions.
use super::{candidate, finite, gathered_greedy, project_rows, DeviceVector, Model, Multi, Result, Ring, Session};
use spark_gpu_probe::driver::CacheRows;

pub(super) struct MultiState<'a> {
    rows: usize, outputs: usize, prefill: bool, hidden: DeviceVector<'a>, input: DeviceVector<'a>, sums: DeviceVector<'a>,
    q: DeviceVector<'a>, k: DeviceVector<'a>, v: DeviceVector<'a>, attention: DeviceVector<'a>,
    projected: DeviceVector<'a>, partial: DeviceVector<'a>, gathered: DeviceVector<'a>,
    gate: DeviceVector<'a>, up: DeviceVector<'a>, metadata: DeviceVector<'a>,
    weight_pack: DeviceVector<'a>, input_pack: DeviceVector<'a>,
    exchange: Option<DeviceVector<'a>>,
    head_input: DeviceVector<'a>, logits: DeviceVector<'a>, arg_parts: DeviceVector<'a>, arg_result: DeviceVector<'a>,
}
impl<'a> MultiState<'a> {
    fn new(m: &Model<'a, '_>, rows: usize, outputs: usize, prefill: bool) -> Result<Self> {
        let c = &m.config;
        let h = c.hidden_size as usize;
        let k = m.local_kv_heads * c.head_dim as usize;
        let q = k * c.attention_heads as usize / c.kv_heads as usize;
        let mid = crate::rows(c.intermediate_size as usize / 64, m.rank, m.world).len() * 64;
        let vocab = m.head.count();
        let fast_ffn = prefill && rows >= 64 && m.packed_ffn();
        let width=(c.intermediate_size as usize).max((c.attention_heads*c.head_dim) as usize).max(h);
        let packed = rows.div_ceil(16)*16*width/2;
        let gather_len=if fast_ffn {(rows*(c.attention_heads*c.head_dim) as usize).max(rows.div_ceil(16)*16*c.intermediate_size as usize/2)}else{rows*width};
        let mut a = m.backend.vector_arena(&[rows*h, rows*h, rows*h.div_ceil(128),
            (rows*q).max(1), (rows*k).max(1), (rows*k).max(1), if m.world>1{1}else{(rows*q).max(1)},
            if m.world>1{1}else{(if fast_ffn { rows.div_ceil(16)*16*mid/2 } else { rows*mid }).max(1)},
            if fast_ffn { 1 } else { (rows*mid).max(1) }, rows*16, 1, packed,
            if prefill { outputs*h } else { 1 },
            (outputs*vocab).max(1), (outputs*vocab.div_ceil(1024)*3).max(1), outputs*3,
            if m.world > 1 { gather_len } else { 1 }])?.into_iter();
        let projected=m.backend.vector(rows*h)?;
        let partial=m.backend.vector(if m.world>1 {(rows*crate::rows(h/128,m.rank,m.world).len()*128).max(1)}else{1})?;
        let mut state=Self { rows, outputs, prefill, exchange:if m.world>1 {Some(m.backend.mapped_vector(rows.div_ceil(16)*16*width)?)}else{None}, hidden: a.next().unwrap(), input: a.next().unwrap(), sums: a.next().unwrap(),
            q: a.next().unwrap(), k: a.next().unwrap(), v: a.next().unwrap(), attention: a.next().unwrap(),
            gate: a.next().unwrap(), up: a.next().unwrap(), metadata: a.next().unwrap(),
            weight_pack: a.next().unwrap(), input_pack: a.next().unwrap(), head_input: a.next().unwrap(), logits: a.next().unwrap(),
            arg_parts: a.next().unwrap(), arg_result: a.next().unwrap(), gathered: a.next().unwrap(), projected, partial };
        if let Some(exchange)=&state.exchange {
            if q>0 {
                let start=crate::rows(c.kv_heads as usize,m.rank,m.world).start*c.attention_heads as usize/c.kv_heads as usize*c.head_dim as usize*rows;
                state.attention=exchange.slice(start..start+rows*q)?;
            }
            if mid>0 {
                let factor=if fast_ffn{rows.div_ceil(16)*8}else{rows};
                let start=crate::rows(c.intermediate_size as usize/64,m.rank,m.world).start*64*factor;
                state.gate=exchange.slice(start..start+mid*factor)?;
            }
        }
        Ok(state)
    }
}
pub(super) struct MultiAttention<'a> { scores: DeviceVector<'a>, parts: DeviceVector<'a>, tiles: DeviceVector<'a> }

impl<'m, 'a, 'b> Multi<'m, 'a, 'b> {
    pub fn new(model: &'m Model<'a, 'b>) -> Result<Self> {
        model.backend.require_multi_kernels()?;
        Ok(Self { model, state: None, attention: None, peak_bytes: model.backend.allocated_bytes() })
    }
    pub fn device_peak_bytes(&self) -> usize { self.peak_bytes }
    /// Results keep session order. Greedy rows contain a single token ID;
    /// workers that disable logit collection return empty non-greedy rows.
    pub fn step(&mut self, sessions: &mut [&mut Session<'m, '_, 'a, 'b>], tokens: &[u32], greedy: bool) -> Result<Vec<Vec<f32>>> {
        if sessions.len() != tokens.len() || sessions.len() > 2048 { return Err("invalid TP decode batch size".into()); }
        if sessions.is_empty() { return Ok(Vec::new()); }
        for (session, &token) in sessions.iter().zip(tokens) {
            if !std::ptr::eq(session.model, self.model) || !std::ptr::eq(session.ring, sessions[0].ring) {
                return Err("TP decode batch must share model and transport".into());
            }
            if session.invalid || session.len() >= self.model.max_tokens || token as u64 >= self.model.config.vocab_size {
                return Err("TP decode batch outside vocabulary/capacity or invalid session".into());
            }
        }
        let result = if sessions.len() == 1 || self.model.full.is_some() {
            // Hybrid sessions retain their sequence-prefill hand-off.
            sessions.iter_mut().zip(tokens).map(|(s, &token)| if greedy {
                s.step_greedy(token).map(|id| vec![id as f32])
            } else { s.step(token) }).collect()
        } else { self.run(sessions, tokens, &vec![1; tokens.len()], greedy, false) };
        // Release transport references before the workspace can be dropped.
        if sessions.len()>1&&self.model.full.is_none() {sessions[0].ring.borrow_mut().clear_registered();}
        let peak = self.model.backend.allocated_bytes();
        self.peak_bytes = self.peak_bytes.max(peak);
        for session in sessions.iter_mut() {
            session.peak_bytes = session.peak_bytes.max(self.peak_bytes);
            session.pinned_peak = session.pinned_peak.max(self.model.backend.peak_pinned_staging_bytes());
            if result.is_err() { session.invalid = true; }
        }
        result
    }
    /// Concatenate prompt chunks for one shard-weight pass. Only each prompt's
    /// last row reaches the LM head; O/down gather operands and complete rows.
    pub fn prefill(&mut self, sessions: &mut [&mut Session<'m, '_, 'a, 'b>], tokens: &[&[u32]], greedy: bool) -> Result<Vec<Vec<f32>>> {
        if sessions.len() != tokens.len() || tokens.iter().map(|t| t.len()).sum::<usize>() > 2048 {
            return Err("invalid TP prefill batch size (maximum 2048 rows)".into());
        }
        if sessions.is_empty() { return Ok(Vec::new()); }
        for (session, ids) in sessions.iter().zip(tokens) {
            if !std::ptr::eq(session.model, self.model) || !std::ptr::eq(session.ring, sessions[0].ring) {
                return Err("TP prefill batch must share model and transport".into());
            }
            if session.invalid || ids.is_empty() || ids.len() > self.model.max_tokens.saturating_sub(session.len())
                || ids.iter().any(|&id| id as u64 >= self.model.config.vocab_size) {
                return Err("TP prefill batch outside vocabulary/capacity or invalid session".into());
            }
        }
        let result = if sessions.len() == 1 || self.model.full.is_some() {
            sessions.iter_mut().zip(tokens).map(|(session, ids)| {
                let collect = session.collect_logits;
                if greedy { session.collect_logits = true; }
                let result = session.prefill(ids);
                session.collect_logits = collect;
                result.and_then(|row| if greedy { Ok(vec![spark_engine::greedy_token(&row)? as f32]) } else { Ok(row) })
            }).collect()
        } else {
            let lengths = tokens.iter().map(|t| t.len()).collect::<Vec<_>>();
            let ids = tokens.iter().flat_map(|t| t.iter().copied()).collect::<Vec<_>>();
            self.run(sessions, &ids, &lengths, greedy, true)
        };
        let mut ring = sessions[0].ring.borrow_mut();
        ring.clear_registered();
        self.peak_bytes = self.peak_bytes.max(self.model.backend.allocated_bytes());
        for session in sessions.iter_mut() {
            session.peak_bytes = session.peak_bytes.max(self.peak_bytes);
            session.pinned_peak = session.pinned_peak.max(self.model.backend.peak_pinned_staging_bytes());
            if result.is_err() { session.invalid = true; }
        }
        self.state = None;
        self.attention = None;
        ring.release_scratch();
        result
    }
    fn run(&mut self, sessions: &mut [&mut Session<'m, '_, 'a, 'b>], tokens: &[u32], lengths: &[usize], greedy: bool, prefill: bool) -> Result<Vec<Vec<f32>>> {
        let m = self.model;
        let b = m.backend;
        // Prompt rows keep one Spark's prefill order, also in passes of 64 rows or fewer.
        let _order = b.prompt_order(prefill);
        let c = &m.config;
        let t = tokens.len();
        let outputs = sessions.len();
        let h = c.hidden_size as usize;
        let kh = m.local_kv_heads;
        let heads = kh*c.attention_heads as usize/c.kv_heads as usize;
        let d = c.head_dim as usize;
        let eps = c.rms_norm_eps as f32;
        for (session, &length) in sessions.iter_mut().zip(lengths) {
            session.handoff_sequence()?;
            if !session.buffered.is_empty() { return Err("sequence prompt must finish before decode".into()); }
            session.reserve(session.len + length)?;
            self.peak_bytes = self.peak_bytes.max(session.peak_bytes);
        }
        let mut ring = sessions[0].ring.borrow_mut();
        if ring.rank() != m.rank || ring.world() != m.world { return Err("model/transport rank mismatch".into()); }
        if self.state.as_ref().is_none_or(|s| s.rows != t || s.outputs != outputs || s.prefill != prefill) {
            ring.clear_registered();
            self.state = None;
            self.state = Some(MultiState::new(m, t, outputs, prefill)?);
        }
        let direct_prefill=prefill && d==128 && sessions.iter().all(|s|s.bf16);
        let needed = if direct_prefill { [1,1,1] } else { b.multi_attention_scratch(sessions.iter().zip(lengths)
            .flat_map(|(s, &length)| (s.len+1..=s.len+length).map(|n| (n, s.bf16.into()))), heads, kh, d) };
        let staging = if m.world > 2 { 2*t*h*m.world+t*16+needed[2] }
            else { t*h+(t*16+needed[2])*if m.world == 1 { m.layers.len() } else { 1 } };
        // Compare upload capacity, never the total that includes other pinned slots.
        if staging*4 > b.pinned_upload_bytes() {
            ring.clear_registered();
            b.reserve_exchange(staging)?;
            for session in sessions.iter_mut() { session.pinned_peak = session.pinned_peak.max(b.peak_pinned_staging_bytes()); }
        }
        if self.attention.as_ref().is_none_or(|a| a.scores.len() < needed[0] || a.parts.len() < needed[1] || a.tiles.len() < needed[2]) {
            self.attention = None;
            let mut a = b.vector_arena(&needed.map(usize::next_power_of_two))?.into_iter();
            self.attention = Some(MultiAttention { scores: a.next().unwrap(), parts: a.next().unwrap(), tiles: a.next().unwrap() });
        }
        let s = self.state.as_mut().unwrap();
        let a = self.attention.as_mut().unwrap();
        self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
        b.write_vector(&mut s.hidden, &m.embeddings(tokens)?)?;
        let rope = m.rope.as_ref().ok_or("missing rope")?;
        let mut sums_ready = false;
        let mut layers=|ring:&mut Ring|->Result<()> {
        for (i, layer) in m.layers.iter().enumerate() {
            let n = layer.norms.as_ref().ok_or("missing layer norms")?;
            if kh > 0 {
                let (q, k, v) = (layer.q.matrix.as_ref().unwrap(), layer.k.matrix.as_ref().unwrap(), layer.v.matrix.as_ref().unwrap());
                if prefill && t >= 64 && h % 128 == 0 && q.rows() % 128 == 0 && k.rows() % 128 == 0 {
                    if !sums_ready { b.rms_partials_device(&s.hidden, &mut s.sums, h)?; }
                    b.rms_pack_apply_device(&s.hidden, &n.input_norm, &mut s.input_pack, &s.sums, eps, false)?;
                    b.prefill_qkv_device(q, k, v, &s.input_pack, &mut s.q, &mut s.k, &mut s.v, t, false)?;
                } else {
                    if sums_ready { b.rms_batch_apply_device(&s.hidden, &n.input_norm, &mut s.input, &s.sums, eps)?; }
                    else { b.rms_batch_device(&s.hidden, &n.input_norm, &mut s.input, &mut s.sums, eps)?; }
                    b.matmul_device(q, &s.input, &mut s.q, &mut s.weight_pack, &mut s.input_pack, t, !prefill)?;
                    b.matmul_reusing_input(k, &s.input, &mut s.k, &mut s.weight_pack, &mut s.input_pack, t, !prefill, true)?;
                    b.matmul_reusing_input(v, &s.input, &mut s.v, &mut s.weight_pack, &mut s.input_pack, t, !prefill, true)?;
                }
                let mut caches = sessions.iter_mut().zip(lengths).map(|(session, &length)| {
                    let kv = &mut session.kv[i];
                    CacheRows { keys: &mut kv.keys, values: &mut kv.values, prefix: None, position: session.len, rows: length, format: session.bf16.into() }
                }).collect::<Vec<_>>();
                if direct_prefill {
                    let mut row=0;
                    for cache in &mut caches {
                        let end=row+cache.rows;
                        let mut query=s.q.slice(row*heads*d..end*heads*d)?;
                        b.qk_batch_device(&mut query,&s.k.slice(row*kh*d..end*kh*d)?,&s.v.slice(row*kh*d..end*kh*d)?,
                            &n.q_norm,&n.k_norm,rope,cache.keys,cache.values,cache.position,eps,cache.rows,cache.format)?;
                        b.prefill_attention_device(&query,cache.keys,cache.values,&mut s.attention.slice(row*heads*d..end*heads*d)?,
                            &mut s.weight_pack,&mut s.input_pack,heads,kh,cache.position,cache.rows,cache.format,
                            cache.position+cache.rows,false)?;
                        row=end;
                    }
                } else {
                    b.attention_multi_device(&mut s.q, &s.k, &s.v, &n.q_norm, &n.k_norm, rope, &mut caches,
                        &mut s.metadata, &mut a.tiles, &mut a.scores, &mut a.parts, &mut s.attention, heads, kh, d, eps)?;
                }
            }
            project_rows(b,ring,&layer.o,&s.attention,&mut s.gathered,&mut s.partial,&mut s.projected,
                &mut s.weight_pack,&mut s.input_pack,t,c.kv_heads as usize,c.attention_heads as usize/c.kv_heads as usize*d,h,false,!prefill,s.exchange.as_mut())?;
            b.residual_device(&mut s.hidden,&s.projected)?;
            b.rms_partials_device(&s.hidden,&mut s.sums,h)?;
            let packed=prefill && t>=64 && m.packed_ffn();
            if let (Some(gate), Some(up)) = (&layer.gate.matrix, &layer.up.matrix) {
                if packed {
                    b.rms_pack_apply_device(&s.hidden, &n.post_norm, &mut s.input_pack, &s.sums, eps, false)?;
                    b.prefill_gate_up_packed_device(gate, up, &s.input_pack, &mut s.gate, t)?;
                } else {
                    b.rms_batch_apply_device(&s.hidden, &n.post_norm, &mut s.input, &s.sums, eps)?;
                    b.matmul_device(gate, &s.input, &mut s.gate, &mut s.weight_pack, &mut s.input_pack, t, !prefill)?;
                    b.matmul_reusing_input(up, &s.input, &mut s.up, &mut s.weight_pack, &mut s.input_pack, t, !prefill, true)?;
                    b.swiglu_device(&mut s.gate, &s.up)?;
                }
            }
            project_rows(b,ring,&layer.down,&s.gate,&mut s.gathered,&mut s.partial,&mut s.projected,
                &mut s.weight_pack,&mut s.input_pack,t,c.intermediate_size as usize/64,64,h,packed,!prefill,s.exchange.as_mut())?;
            b.residual_device(&mut s.hidden,&s.projected)?;
            b.rms_partials_device(&s.hidden,&mut s.sums,h)?;
            sums_ready=true;
        }
            Ok(())
        };
        if m.world>1 {ring.with_transfer_regions(layers)?;}else{layers(&mut ring)?;}
        let norm = m.norm.as_ref().ok_or("missing output norm")?;
        if sums_ready { b.rms_batch_apply_device(&s.hidden, norm, &mut s.input, &s.sums, eps)?; }
        else { b.rms_batch_device(&s.hidden, norm, &mut s.input, &mut s.sums, eps)?; }
        if prefill {
            let mut source = 0;
            for (destination, &length) in lengths.iter().enumerate() {
                source += length;
                b.copy_strided(&s.input, &mut s.head_input, (source-1)*h, destination*h, h, h, h, 1)?;
            }
        }
        if let Some(head) = &m.head.matrix {
            b.matmul_device(head, if prefill { &s.head_input } else { &s.input }, &mut s.logits, &mut s.weight_pack, &mut s.input_pack, outputs, true)?;
            b.round_logits_bf16(&mut s.logits)?;
        }
        let output = if greedy {
            let local = if m.head.count() > 0 {
                b.argmax_batch_device(&s.logits, &mut s.arg_parts, &mut s.arg_result, outputs)?
            } else { vec![Ok((0, f32::NEG_INFINITY)); outputs] };
            let mut candidates = vec![0u8; m.world*outputs*8];
            for (dst, local) in candidates[m.rank*outputs*8..(m.rank+1)*outputs*8].chunks_exact_mut(8).zip(local) {
                let (id,value)=candidate(local,m.head.first);
                dst[..4].copy_from_slice(&id.to_le_bytes());
                dst[4..].copy_from_slice(&value.to_le_bytes());
            }
            ring.all_gather(&mut candidates, outputs*8)?;
            (0..outputs).map(|row| {
                gathered_greedy((0..m.world).map(|rank| {
                    let at = (rank*outputs+row)*8;
                    let id = u32::from_le_bytes(candidates[at..at+4].try_into().unwrap());
                    let score = f32::from_le_bytes(candidates[at+4..at+8].try_into().unwrap());
                    (id,score)
                }),c.vocab_size as usize).map(|id|vec![id as f32])
            }).collect::<Result<Vec<_>>>()?
        } else {
            let local = if m.head.count() == 0 { Vec::new() } else { b.read_vector_values(&s.logits)? };
            let all = ring.gather_batch(&local, c.vocab_size as usize, 1, outputs)?;
            finite(all)?;
            all.chunks_exact(c.vocab_size as usize).zip(sessions.iter()).map(|(row, session)| {
                if session.collect_logits { row.to_vec() } else { Vec::new() }
            }).collect()
        };
        b.synchronize()?;
        for (session, &length) in sessions.iter_mut().zip(lengths) { session.len += length; }
        Ok(output)
    }
}
