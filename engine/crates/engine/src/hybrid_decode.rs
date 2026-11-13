//! Independent chats share one BF16 projection pass while retaining their own
//! delta-net history/matrices and full-attention KV. Prefill stays in Session.
use super::*;
#[path = "hybrid/gdn.rs"]
pub mod gdn;

pub const MAX_ROWS: usize = 16;
/// Independent plain-decode chats; prompt and verification budgets stay fixed.
pub const MAX_CHAT_ROWS: usize = 32;
pub const MAX_VERIFY_ROWS: usize = MAX_ROWS * 5;
pub type PromptWorkspace<'m, 'a, 'b> = std::rc::Rc<std::cell::RefCell<Option<Box<Decode<'m, 'a, 'b>>>>>;

pub struct Row<'s, 'm, 'a, 'b> {
    pub session: &'s mut Session<'m, 'a, 'b>,
    pub token: u32,
    pub logits: bool,
}
pub struct VerifyRow<'s, 'm, 'a, 'b> {
    pub session: &'s mut Session<'m, 'a, 'b>,
    pub tokens: &'s [u32],
}
pub enum VerifyOutput { Logits(Vec<Vec<Vec<f32>>>), Greedy(Vec<Vec<u32>>) }

pub struct Decode<'m, 'a, 'b> {
    typed_graphs: graph::Graphs<'a>,
    graph: Option<DecodeGraph<'a>>,
    graph_rows: Vec<(std::rc::Weak<()>, usize)>,
    graph_position: Option<DeviceVector<'a>>,
    graph_logits: bool,
    model: &'m Model<'a, 'b>,
    capacity: usize,
    gdn: GdnScratch<'a>,
    attention: DenseScratch<'a>,
    moe: Option<moe::Workspace<'a>>,
    typed_moe: Option<forward::Moe<'a>>,
    token_step: Option<gdn::TokenStep<'a>>,
    weight_pack: DeviceVector<'a>,
    cache_rows: DeviceVector<'a>,
    // The immutable descriptors uploaded to cache_rows; history addresses never change on-device.
    cache_host: Vec<f32>,
    // Set only while prompt() enqueues ungraphed rows: kernel grids then follow the live pasts.
    prompting: bool,
    prompt_tiles: Vec<usize>,
    host_ids: Vec<u32>,
    hidden: DeviceVector<'a>, normalized: DeviceVector<'a>, block: DeviceVector<'a>,
    gate: DeviceVector<'a>, up: DeviceVector<'a>, ids: DeviceVector<'a>,
    input_pack: DeviceVector<'a>,
    logits: DeviceVector<'a>,
    argmax: DeviceVector<'a>, selected: DeviceVector<'a>,
    rejected: Vec<usize>,
}
impl<'m, 'a, 'b> Decode<'m, 'a, 'b> {
    pub fn reservation_bytes(c: &Config) -> usize {
        Self::reservation_bytes_for_rows(c, MAX_CHAT_ROWS)
    }
    pub fn reservation_bytes_for_rows(c: &Config, rows: usize) -> usize {
        let (h, m, v) = (c.hidden as usize, c.intermediate as usize, c.vocab as usize);
        let arms=arms::Table::for_config(c,arms::Limits::LOCAL_WORLD,false);
        let head = rows.min(schedule::PROMPT_ROWS);
        [rows * h, rows * h, rows * h, rows * m, rows * m, rows,
            head * v, head * v.div_ceil(1024) * 3, head * 3, c.layers.len() * rows * 6, 4,
            if c.fp8 { rows * h.max(m).max((c.linear_value_heads * c.linear_value_dim) as usize) / 2 } else { 1 }]
            .iter().map(|n| (n * 4).div_ceil(64) * 64).sum::<usize>()
            + arms.arm(arms::Kind::Gdn).reserve(|| Some(GdnScratch::decode_reservation_bytes(h, c.linear_value_heads as usize, c.linear_key_heads as usize, rows))).unwrap_or(usize::MAX)
            + arms.arm(arms::Kind::Attention).reserve(|| Some(DenseScratch::reservation_bytes(c, head))).unwrap_or(usize::MAX)
            + moe::Workspace::reservation_bytes(c, rows) + arms.arm(arms::Kind::MoeFp8).reserve(|| Some(forward::Moe::reservation_bytes(c, rows))).unwrap_or(usize::MAX)
            + head * c.query_heads as usize * c.max_positions as usize * 4 + 64
    }
    fn project(model: &Model<'_, '_>, name: &str, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        packed: &mut DeviceVector<'_>, count: usize, reuse: bool) -> Result<()> {
        if model.bound.is_some() { return model.fixed_project_reusing(name, input, output, packed, count, reuse); }
        let (b, matrix) = (model.backend, model.matrix(name)?);
        // Sharing weights must preserve the single-chat packed8 sum order.
        b.profile_phase(14, || b.matmul_decode_bf16_reusing_input(matrix, input, output, packed, count, reuse))
    }
    pub fn new(model: &'m Model<'a, 'b>) -> Result<Self> {
        Self::with_rows(model, MAX_CHAT_ROWS)
    }
    pub fn with_rows(model: &'m Model<'a, 'b>, capacity: usize) -> Result<Self> {
        if !(1..=if model.bound.is_some() { schedule::ROUTED_GROUP } else { MAX_VERIFY_ROWS }).contains(&capacity) { return Err("invalid hybrid decode row capacity".into()); }
        let (b, c) = (model.backend, &model.config);
        let (h, m, v) = (c.hidden as usize, c.intermediate as usize, c.vocab as usize);
        // Attention, scores and logits never take more than one canonical tile.
        let head = capacity.min(schedule::PROMPT_ROWS);
        let mut a = b.vector_arena(&[capacity * h, capacity * h, capacity * h,
            capacity * m, capacity * m, capacity,
            head * v, head * v.div_ceil(1024) * 3, head * 3,
            if c.fp8 { capacity.max(16) * h.max(m).max((c.linear_value_heads * c.linear_value_dim) as usize) / 2 } else { 1 }])?.into_iter();
        Ok(Self { typed_graphs: Vec::new(), graph: None, graph_rows: Vec::new(), graph_position: Some(b.vector(4)?), graph_logits: false, model, capacity,
            gdn: GdnScratch::decode_workspace(b, h, c.linear_value_heads as usize,
                c.linear_key_heads as usize, capacity)?,
            attention: DenseScratch::new(b, c, head)?,
            typed_moe: forward::Moe::new(model, capacity)?, token_step: if model.bound.is_some() { Some(gdn::TokenStep::new(b, c.linear_value_heads as usize)?) } else { None },
            moe: moe::Workspace::new(model, capacity)?, weight_pack: b.vector(1)?,
            cache_rows: b.vector_arena(&[c.layers.len() * capacity * 6])?.pop().unwrap(),
            cache_host: Vec::new(), prompting: false, prompt_tiles: Vec::new(), host_ids: Vec::new(),
            hidden: a.next().unwrap(), normalized: a.next().unwrap(), block: a.next().unwrap(),
            gate: a.next().unwrap(), up: a.next().unwrap(), ids: a.next().unwrap(),
            logits: a.next().unwrap(),
            argmax: a.next().unwrap(), selected: a.next().unwrap(), input_pack: a.next().unwrap(), rejected: Vec::new() })
    }
    fn attention_rows(&self) -> usize { self.capacity.min(schedule::PROMPT_ROWS) }
    /// Chats of the last step or verification whose router rows were rejected.
    /// They failed alone: no output (greedy: u32::MAX), uncommitted and left
    /// invalid, since their recurrent state advanced in place. The rest committed.
    pub fn rejected(&self) -> &[usize] { &self.rejected }
    pub(super) fn read_logits(&self, rows: usize) -> Result<Vec<Vec<f32>>> {
        let width = self.model.config.vocab as usize;
        let values = self.model.backend.read_vector_values(&self.logits.slice(0..rows * width)?)?;
        (0..rows).map(|row| Self::checked(&values, row, width).map(<[f32]>::to_vec)).collect()
    }
    fn routing(&mut self, rows: usize, chats: usize, owner: impl Fn(usize) -> usize) -> Result<Vec<bool>> {
        let rejected = if let Some(moe) = &self.typed_moe { let mut rejected=vec![false;chats]; for row in moe.rejected(self.model.backend,rows)? { rejected[owner(row)]=true; } rejected } else { match &self.moe { Some(moe) => moe.rejected(self.model.backend, rows, chats, owner)?, None => vec![false; chats] } };
        self.rejected = (0..chats).filter(|&i| rejected[i]).collect();
        Ok(rejected)
    }
    /// Raw values: a rejected row's output never fails its batch mates.
    pub(super) fn checked(values: &[f32], row: usize, width: usize) -> Result<&[f32]> {
        let values = values.get(row * width..(row + 1) * width).ok_or("missing hybrid batch output row")?;
        if values.iter().any(|v| !v.is_finite()) { return Err("nonfinite device output".into()); }
        Ok(values)
    }
    pub fn step(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<Vec<Vec<f32>>> {
        self.forward(rows, false)?;
        let (b, v) = (self.model.backend, self.model.config.vocab as usize);
        let (values, rejected) = if let Some(moe) = self.typed_moe.as_ref().filter(|_| self.model.bound.as_ref().is_some_and(|bound| bound.group.distributed())) {
            // Sticky device flags and logits share one completion boundary.
            // Read flags even when no row requests logits; commit stays below it.
            let mut vectors = vec![moe.status(rows.len())?];
            if rows.iter().any(|r| r.logits) { vectors.push(self.logits.slice(0..rows.len()*v)?); }
            let mut output = b.read_vectors_values(&vectors)?;
            self.rejected = spark_gpu_probe::routing::rejected_rows(&output[0]);
            let rejected = (0..rows.len()).map(|row| self.rejected.contains(&row)).collect::<Vec<_>>();
            (if output.len()>1 { output.remove(1) } else { Vec::new() }, rejected)
        } else {
            let values = if rows.iter().any(|r| r.logits) { b.read_vector_values(&self.logits)? } else { b.synchronize()?; Vec::new() };
            (values, self.routing(rows.len(), rows.len(), |row| row)?)
        };
        let result = rows.iter().enumerate().map(|(i, row)| Ok(if rejected[i] || values.is_empty() { Vec::new() } else {
            let logits = Self::checked(&values, i, v)?;
            if row.logits { logits.to_vec() } else { Vec::new() }
        })).collect::<Result<_>>()?;
        Self::commit(rows, &rejected);
        Ok(result)
    }
    pub fn step_greedy(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>]) -> Result<Vec<u32>> {
        if rows.iter().any(|r| !r.logits) { return Err("greedy decode requires an output for each row".into()); }
        self.forward(rows, true)?;
        self.argmax.resize_within_capacity(rows.len() * (self.model.config.vocab as usize).div_ceil(1024) * 3)?;
        self.selected.resize_within_capacity(rows.len() * 3)?;
        let (ids, rejected) = if let Some(moe) = &self.typed_moe {
            let (ids, rejected) = moe.selections(self.model.backend, &self.selected, rows.len())?;
            self.rejected = rejected;
            let rejected = (0..rows.len()).map(|row| self.rejected.contains(&row)).collect::<Vec<_>>();
            (ids, rejected)
        } else {
            let ids = if self.model.bound.is_some() { self.model.backend.read_argmax_batch_device(&self.selected)? }
                else { self.model.backend.argmax_batch_device(&self.logits, &mut self.argmax, &mut self.selected, rows.len())? };
            (ids, self.routing(rows.len(), rows.len(), |row| row)?)
        };
        let result = ids.into_iter().zip(&rejected).map(|(id, &rejected)| if rejected { Ok(u32::MAX) } else { id.map(|(id, _)| id) })
            .collect::<Result<_>>()?;
        Self::commit(rows, &rejected);
        Ok(result)
    }
    pub(super) fn verify(&mut self, session: &mut Session<'m, 'a, 'b>, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let output = self.verify_batch(&mut [VerifyRow { session, tokens }], false)?;
        if !self.rejected.is_empty() { return Err(MOE_ROUTING_ERROR.into()); }
        match output {
            VerifyOutput::Logits(mut rows) => Ok(rows.remove(0)),
            VerifyOutput::Greedy(_) => Err("missing canonical verification logits".into()),
        }
    }
    /// Consecutive prompt rows share the fixed layer loop. GDN descriptors all
    /// name the live state/history: TokenStep publishes them in token order.
    pub(super) fn prompt(&mut self, session: &mut Session<'m, 'a, 'b>, tokens: &[u32], logits: bool) -> Result<Vec<f32>> {
        let (model,count)=(self.model,tokens.len()); let (b,c)=(model.backend,&model.config);
        if model.bound.is_none() || !std::ptr::eq(session.model,model) || !session.valid || count==0 || count>self.capacity
            || session.position.checked_add(count).is_none_or(|n| n>model.capacity) || tokens.iter().any(|&t| t as u64>=c.vocab) {
            return Err("invalid typed prompt rows or context".into());
        }
        // Prefill owns its tile-entry checkpoint; begin_decode would snapshot
        // an interior tile and leave a stale entry after a full prompt chunk.
        session.reserve_state()?; session.reserve_kv(session.position+count)?;
        let entry=session.position; session.valid=false; self.graph=None; self.typed_graphs.clear();
        self.attention.reserve_decode_scores(b,c.query_heads as usize,self.attention_rows(),entry+count,model.capacity)?;
        let h=c.hidden as usize;
        for v in [&mut self.hidden,&mut self.normalized,&mut self.block] { v.resize_within_capacity(count*h)?; }
        self.gate.resize_within_capacity(count*c.intermediate as usize)?; self.up.resize_within_capacity(count*c.intermediate as usize)?;
        self.ids.resize_within_capacity(count)?;
        self.host_ids = tokens.to_vec();
        b.write_vector(&mut self.ids,&tokens.iter().map(|&t| t as f32).collect::<Vec<_>>())?;
        let mut descriptors=vec![0.0;self.cache_rows.len()];
        for (layer,kind) in c.layers.iter().enumerate() { for step in 0..count {
            let row=match kind {
                LayerKind::DeltaNet=>session.gdn_cache[layer].as_ref().unwrap().descriptor(),
                LayerKind::SparseAttention=>session.full_cache[layer].as_ref().unwrap().descriptor(entry+step),
            };
            let offset=(layer*self.capacity+step)*6; descriptors[offset..offset+6].copy_from_slice(&row);
        } }
        b.write_vector(&mut self.cache_rows,&descriptors)?;
        self.cache_host=descriptors;
        self.prompt_tiles.clear();
        self.prompting=true; let enqueued=self.enqueue_rows(count,false,entry,None,&[]); self.prompting=false; enqueued?;
        if self.routing(count,1,|_| 0)?[0] { return Err(MOE_ROUTING_ERROR.into()); }
        if logits {
            model.profile_reads(None, Vec::new);
            self.logits.resize_within_capacity(c.vocab as usize)?;
            let hidden = self.hidden.slice((count-1)*h..count*h)?;
            let normalized = self.normalized.slice(0..h)?;
            Session::normalize(b,model,&hidden,model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(),&normalized,1)?;
            b.profile_output_head(|| model.head_rows(&normalized,&mut self.logits,&mut self.input_pack,1))?;
        }
        let output=if logits { b.read_vector_values(&self.logits)? }
            else { b.synchronize()?; Vec::new() };
        // Keep the draft's one-token observation arithmetic as well.
        if session.mtp.is_some() { for (step,&token) in tokens.iter().enumerate() {
            session.position=entry+step;
            let observed=self.hidden.slice(step*h..(step+1)*h).and_then(|hidden| session.mtp_observe_hidden(&[token],&hidden));
            session.position=entry; observed?;
        } }
        session.position=entry+count; session.valid=true;
        Ok(output)
    }
    /// Independent canonical prompt tiles share projections, never recurrent state.
    pub fn prompt_batch(&mut self, rows: &mut [prefill::Row<'_, 'm, 'a, 'b>]) -> Result<Vec<Vec<f32>>> {
        let (model, count) = (self.model, rows.iter().map(|r| r.tokens.len()).sum::<usize>());
        let (b, c) = (model.backend, &model.config);
        let tile = model.prefill_chunk_size();
        if model.bound.as_ref().is_none_or(|v| v.group.distributed()) || rows.is_empty() || count > self.capacity
            || rows.iter().any(|r| !std::ptr::eq(r.session.model, model) || !r.session.valid
                || r.session.position % tile != 0 || !r.session.prefill_tail.is_empty()
                || r.session.pending_prefill.is_some() || !r.session.verify_tokens.is_empty()
                || r.tokens.is_empty() || r.tokens.len() > tile
                || r.session.position.checked_add(r.tokens.len()).is_none_or(|n| n > model.capacity)
                || r.tokens.iter().any(|&t| t as u64 >= c.vocab)) { return Err("invalid independent canonical prompt tiles".into()); }
        for row in rows.iter_mut() {
            let s = &mut row.session;
            s.finish_mtp_verification()?;
            s.reserve_state()?; s.reserve_kv(s.position + row.tokens.len())?;
            s.prefill_saved.clear(); s.prefill_mtp = None;
            if row.tokens.len() < tile { s.save_prefill_entry()?; }
            s.greedy_logits = None; s.valid = false;
        }
        self.graph = None; self.typed_graphs.clear();
        let past = rows.iter().map(|r| r.session.position + r.tokens.len()).max().unwrap();
        self.attention.reserve_decode_scores(b, c.query_heads as usize, self.attention_rows(), past, model.capacity)?;
        let h = c.hidden as usize;
        for v in [&mut self.hidden, &mut self.normalized, &mut self.block] { v.resize_within_capacity(count*h)?; }
        self.gate.resize_within_capacity(count*c.intermediate as usize)?;
        self.up.resize_within_capacity(count*c.intermediate as usize)?; self.ids.resize_within_capacity(count)?;
        self.host_ids = rows.iter().flat_map(|r| r.tokens.iter().copied()).collect();
        b.write_vector(&mut self.ids, &self.host_ids.iter().map(|&t| t as f32).collect::<Vec<_>>())?;
        let mut descriptors = vec![0.0; self.cache_rows.len()];
        for (layer, kind) in c.layers.iter().enumerate() {
            let mut at = 0;
            for row in rows.iter() { for step in 0..row.tokens.len() {
                let descriptor = match kind {
                    LayerKind::DeltaNet => row.session.gdn_cache[layer].as_ref().unwrap().descriptor(),
                    LayerKind::SparseAttention => row.session.full_cache[layer].as_ref().unwrap().descriptor(row.session.position+step),
                };
                let offset = (layer*self.capacity+at)*6;
                descriptors[offset..offset+6].copy_from_slice(&descriptor); at += 1;
            } }
        }
        b.write_vector(&mut self.cache_rows, &descriptors)?; self.cache_host = descriptors;
        // Tile-aligned request starts keep attention's canonical 256-row spans.
        self.prompt_tiles = rows.iter().map(|r| r.tokens.len()).collect();
        self.prompting = true; let result = self.enqueue_rows(count, false, 0, None, &[]); self.prompting = false; result?;
        let ends: Vec<_> = rows.iter().scan(0, |n,r| { *n += r.tokens.len(); Some(*n) }).collect();
        let rejected = self.routing(count, rows.len(), |i| ends.partition_point(|&end| end <= i))?;
        let mut output = Vec::with_capacity(rows.len()); let mut at = 0;
        for (i, row) in rows.iter_mut().enumerate() {
            let n = row.tokens.len();
            if row.logits && !rejected[i] {
                self.logits.resize_within_capacity(c.vocab as usize)?;
                let hidden = self.hidden.slice((at+n-1)*h..(at+n)*h)?;
                let normalized = self.normalized.slice(0..h)?;
                Session::normalize(b, model, &hidden, model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(), &normalized, 1)?;
                b.profile_output_head(|| model.head_rows(&normalized, &mut self.logits, &mut self.input_pack, 1))?;
                output.push(b.read_vector_values(&self.logits)?);
            } else { output.push(Vec::new()); }
            if !rejected[i] {
                let s = &mut row.session;
                if s.mtp.is_some() { for (step, &token) in row.tokens.iter().enumerate() {
                    let entry = s.position; s.position += step;
                    let result = self.hidden.slice((at+step)*h..(at+step+1)*h).and_then(|hidden| s.mtp_observe_hidden(&[token], &hidden));
                    s.position = entry; result?;
                } }
                s.position += n; s.valid = true;
                if n == tile { s.save_prefix_boundary()?; } else { s.prefill_tail.extend_from_slice(row.tokens); }
            }
            at += n;
        }
        b.synchronize()?;
        Ok(output)
    }
    pub fn verify_batch(&mut self, rows: &mut [VerifyRow<'_, '_, 'a, 'b>], greedy: bool) -> Result<VerifyOutput> {
        let started = std::time::Instant::now();
        let layout = self.forward_verify(rows, greedy)?;
        let (b, v) = (self.model.backend, self.model.config.vocab as usize);
        let mut route_rows = None;
        let ids = if greedy {
            self.argmax.resize_within_capacity(layout.len() * v.div_ceil(1024) * 3)?;
            self.selected.resize_within_capacity(layout.len() * 3)?;
            if let Some(moe)=&self.typed_moe { let (ids,rejected)=moe.selections(b,&self.selected,layout.len())?; route_rows=Some(rejected); ids }
            else { b.argmax_batch_device(&self.logits, &mut self.argmax, &mut self.selected, layout.len())? }
        } else { Vec::new() };
        let values = if greedy { Vec::new() } else { b.read_vector_values(&self.logits)? };
        let rejected = if let Some(route_rows)=route_rows {
            let mut rejected=vec![false;rows.len()]; for row in route_rows { rejected[layout[row].0]=true; }
            self.rejected=(0..rows.len()).filter(|&i|rejected[i]).collect(); rejected
        } else { self.routing(layout.len(), rows.len(), |row| layout[row].0)? };
        let result = if greedy {
            let mut output: Vec<Vec<u32>> = rows.iter().map(|_| Vec::new()).collect();
            for (&(chat, _), id) in layout.iter().zip(ids) { if !rejected[chat] { output[chat].push(id?.0); } }
            VerifyOutput::Greedy(output)
        } else {
            let mut output: Vec<Vec<Vec<f32>>> = rows.iter().map(|_| Vec::new()).collect();
            for (i, &(chat, _)) in layout.iter().enumerate() {
                if !rejected[chat] { output[chat].push(Self::checked(&values, i, v)?.to_vec()); }
            }
            VerifyOutput::Logits(output)
        };
        let h = self.model.config.hidden as usize;
        let plain = layout.len() == rows.len();
        let direct_hidden = self.model.bound.is_some() && rows.len()==1;
        for (i, (chat, step)) in layout.into_iter().enumerate() {
            if rejected[chat] || direct_hidden { continue; }
            let session = &mut rows[chat].session;
            self.model.backend.copy_vector_range(&self.normalized, session.verify_hidden.as_mut().unwrap(), i * h, step * h, h)?;
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        for row in rows.iter_mut().zip(&rejected).filter(|r| !r.1).map(|r| r.0) {
            let s = &mut row.session;
            s.record_batch_time(row.tokens.len(), ms, plain);
            s.verify_entry = s.position;
            s.verify_tokens = row.tokens.to_vec();
            s.position += row.tokens.len();
            s.valid = true;
        }
        Ok(result)
    }
    fn forward_verify(&mut self, rows: &mut [VerifyRow<'_, '_, 'a, 'b>], greedy: bool) -> Result<Vec<(usize, usize)>> {
        let b = self.model.backend;
        b.generic_profile_begin(false, || (0, rows.iter().map(|r| r.tokens.len()).sum(), self.model.profile_plan.as_ref().unwrap().profile_classes(), 0))?;
        let result = self.forward_verify_profiled(rows, greedy); b.generic_profile_end(false, result.is_ok())?; result
    }
    fn forward_verify_profiled(&mut self, rows: &mut [VerifyRow<'_, '_, 'a, 'b>], greedy: bool) -> Result<Vec<(usize, usize)>> {
        let (b, c) = (self.model.backend, &self.model.config);
        let count: usize = rows.iter().map(|r| r.tokens.len()).sum();
        if rows.is_empty() || count > self.capacity || rows.iter().any(|r| !std::ptr::eq(r.session.model, self.model)
            || !r.session.valid || r.session.position == 0 || r.tokens.is_empty() || r.tokens.len() > MAX_ROWS
            || r.session.position + r.tokens.len() > self.model.capacity || r.tokens.iter().any(|&t| t as u64 >= c.vocab)) {
            return Err("canonical verification requires valid prefilled chats within the row/context budget".into());
        }
        let plain = self.model.bound.is_none() && count == rows.len() && rows.len() <= MAX_ROWS;
        for row in rows.iter_mut() {
            let s = &mut row.session;
            s.greedy_logits = None;
            s.finish_mtp_verification()?;
            s.begin_decode()?;
            s.reserve_kv(s.position + row.tokens.len())?;
            while s.rollback_state.len() < row.tokens.len() - 1 { s.rollback_state.push(GdnState::dense_layers(b, c)?); }
            if s.verify_hidden.is_none() { s.verify_hidden = b.vector_arena(&[MAX_ROWS * c.hidden as usize])?.pop(); }
            if !plain { s.valid = false; }
        }
        // Depth-major packing lets each recurrence launch advance all chats once.
        let layout = schedule::tokens(&rows.iter().map(|r| r.tokens.len()).collect::<Vec<_>>());
        if plain {
            let mut plain: Vec<_> = rows.iter_mut().map(|r| Row { session: &mut *r.session, token: r.tokens[0], logits: true }).collect();
            self.forward_profiled(&mut plain, greedy)?;
            return Ok(layout);
        }
        let length = rows.iter().map(|r| r.session.position + r.tokens.len()).max().unwrap();
        if self.attention.reserve_decode_scores(b, c.query_heads as usize, self.attention_rows(), length, self.model.capacity)? {
            self.graph = None; self.typed_graphs.clear();
        }
        let h = c.hidden as usize;
        for v in [&mut self.hidden, &mut self.normalized, &mut self.block] { v.resize_within_capacity(count * h)?; }
        self.gate.resize_within_capacity(count * c.intermediate as usize)?;
        self.up.resize_within_capacity(count * c.intermediate as usize)?;
        self.ids.resize_within_capacity(count)?;
        self.host_ids = layout.iter().map(|&(chat, step)| rows[chat].tokens[step]).collect();
        b.write_vector(&mut self.ids, &layout.iter().map(|&(chat, step)| rows[chat].tokens[step] as f32).collect::<Vec<_>>())?;
        let mut descriptors = vec![0.0; self.cache_rows.len()];
        for (layer, kind) in c.layers.iter().enumerate() {
            for (i, &(chat, step)) in layout.iter().enumerate() {
                let s = &rows[chat].session;
                let descriptor = match kind {
                    LayerKind::DeltaNet => {
                        let live = s.gdn_cache[layer].as_ref().unwrap();
                        let input = if step == 0 { live } else { s.rollback_state[step - 1][layer].as_ref().unwrap() };
                        let output = if step + 1 == rows[chat].tokens.len() { live } else { s.rollback_state[step][layer].as_ref().unwrap() };
                        input.decode_descriptor(live, Some(output))
                    },
                    LayerKind::SparseAttention => s.full_cache[layer].as_ref().unwrap().descriptor(s.position + step),
                };
                let offset = (layer * self.capacity + i) * 6;
                descriptors[offset..offset + 6].copy_from_slice(&descriptor);
            }
        }
        b.write_vector(&mut self.cache_rows, &descriptors)?;
        self.cache_host = descriptors;
        self.graph = None;
        if self.model.bound.is_some() {
            let lives = rows.iter().map(|r| std::rc::Rc::downgrade(&r.session.decode_lifetime)).collect();
            self.typed_forward(count, true, rows[0].session.position, Some(rows), &layout, greedy, lives)?;
        } else { self.enqueue_rows(count, true, rows[0].session.position, Some(rows), &layout)?; }
        Ok(layout)
    }
    fn commit(rows: &mut [Row<'_, '_, 'a, 'b>], rejected: &[bool]) {
        for row in rows.iter_mut().zip(rejected).filter(|r| !r.1).map(|r| r.0) {
            row.session.greedy_logits = None;
            row.session.position += 1;
            row.session.prefill_tail.push(row.token);
            row.session.valid = true;
        }
    }
    fn forward(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>], greedy: bool) -> Result<()> {
        let b = self.model.backend;
        b.generic_profile_begin(false, || (0, rows.len(), self.model.profile_plan.as_ref().unwrap().profile_classes(), 0))?;
        let result = self.forward_profiled(rows, greedy); b.generic_profile_end(false, result.is_ok())?; result
    }
    fn forward_profiled(&mut self, rows: &mut [Row<'_, '_, 'a, 'b>], greedy: bool) -> Result<()> {
        let model = self.model;
        let (b, c, count) = (model.backend, &model.config, rows.len());
        if count == 0 || count > if model.bound.is_some() { self.capacity } else { self.capacity.min(64) } { return Err("hybrid decode exceeds its row capacity".into()); }
        for row in rows.iter() {
            let s = &row.session;
            if !std::ptr::eq(s.model, model) || !s.valid || (s.position == 0 && model.bound.is_none())
                || s.position >= model.capacity || row.token as u64 >= c.vocab
                || c.layers.iter().enumerate().any(|(i, kind)| match kind {
                    LayerKind::DeltaNet => s.gdn_cache[i].is_none() && model.bound.is_none(),
                    LayerKind::SparseAttention => s.full_cache[i].is_none() && model.bound.is_none(),
                }) { return Err("hybrid decode needs valid prefilled sessions with available context".into()); }
        }
        for row in rows.iter_mut() {
            row.session.reserve_state()?;
            row.session.reserve_kv(row.session.position + 1)?;
            row.session.begin_decode()?;
            row.session.valid = false;
        }
        let length = rows.iter().map(|r| r.session.position + 1).max().unwrap();
        if self.attention.reserve_decode_scores(b, c.query_heads as usize, self.attention_rows(), length, model.capacity)? {
            self.graph = None; self.typed_graphs.clear();
        }
        let h = c.hidden as usize;
        for v in [&mut self.hidden, &mut self.normalized, &mut self.block] { v.resize_within_capacity(count * h)?; }
        self.gate.resize_within_capacity(count * c.intermediate as usize)?;
        self.up.resize_within_capacity(count * c.intermediate as usize)?;
        self.ids.resize_within_capacity(count)?;
        self.host_ids = rows.iter().map(|r| r.token).collect();
        b.write_vector(&mut self.ids, &rows.iter().map(|r| r.token as f32).collect::<Vec<_>>())?;
        let base = rows.iter().map(|r| r.session.position).min().unwrap();
        if model.bound.is_some() {
            self.write_cache_rows(rows, 0)?;
            let lives = rows.iter().map(|r| std::rc::Rc::downgrade(&r.session.decode_lifetime)).collect();
            return self.typed_forward(count, rows.iter().any(|r| r.logits), base, None, &[], greedy, lives);
        }
        if b.is_profiling() {
            self.graph = None;
            self.write_cache_rows(rows, 0)?;
            return self.enqueue_rows(count, rows.iter().any(|r| r.logits), base, None, &[]);
        }
        let logits = rows.iter().any(|r| r.logits);
        let same = self.graph.is_some() && self.graph_logits == logits && self.graph_rows.len() == count
            && self.graph_rows.iter().zip(rows.iter()).all(|((life, offset), row)|
                std::rc::Weak::ptr_eq(life, &std::rc::Rc::downgrade(&row.session.decode_lifetime))
                    && *offset == row.session.position - base);
        b.write_vector(self.graph_position.as_mut().unwrap(), &[0.0, 0.0, 0.0, f32::from_bits(base as u32)])?;
        if !same {
            self.graph = None;
            self.write_cache_rows(rows, base)?;
            let position = self.graph_position.take().unwrap();
            // Weak identities prevent replay after a session was dropped or
            // replaced, including allocator reuse of the same device address.
            let graph = unsafe { b.capture_decode(&position, base, c.layers.len(), count,
                || self.enqueue_rows(count, logits, base, None, &[])) };
            self.graph_position = Some(position);
            self.graph = Some(graph?);
            self.graph_rows = rows.iter().map(|r| (std::rc::Rc::downgrade(&r.session.decode_lifetime), r.session.position - base)).collect();
            self.graph_logits = logits;
        }
        // Lifetime identities also change when a request grows its KV cache.
        unsafe { b.replay_decode(self.graph.as_ref().unwrap()) }
    }
    fn write_cache_rows(&mut self, rows: &[Row<'_, '_, 'a, 'b>], base: usize) -> Result<()> {
        let mut descriptors = vec![0.0; self.cache_rows.len()];
        for (layer, kind) in self.model.config.layers.iter().enumerate() {
            for (i, row) in rows.iter().enumerate() {
                let offset = (layer * self.capacity + i) * 6;
                let descriptor = match kind {
                    LayerKind::DeltaNet => row.session.gdn_cache[layer].as_ref().unwrap().descriptor(),
                    LayerKind::SparseAttention => row.session.full_cache[layer].as_ref().unwrap()
                        .descriptor(row.session.position - base),
                };
                descriptors[offset..offset + 6].copy_from_slice(&descriptor);
            }
        }
        self.model.backend.write_vector(&mut self.cache_rows, &descriptors)?;
        self.cache_host = descriptors;
        Ok(())
    }
    fn typed_forward(&mut self, count: usize, logits: bool, position: usize,
        mut sequence: Option<&mut [VerifyRow<'_, '_, 'a, 'b>]>, layout: &[(usize, usize)], greedy: bool,
        lives: Vec<std::rc::Weak<()>>) -> Result<()> {
        if logits { self.logits.resize_within_capacity(count*self.model.config.vocab as usize)?; }
        if greedy {
            self.argmax.resize_within_capacity(count*(self.model.config.vocab as usize).div_ceil(1024)*3)?;
            self.selected.resize_within_capacity(count*3)?;
        }
        let mut key = vec![count as u64, u64::from(logits), u64::from(greedy)];
        key.extend(layout.iter().flat_map(|&(chat, step)| [chat as u64, step as u64]));
        for (layer, kind) in self.model.config.layers.iter().enumerate() {
            let offset = layer*self.capacity*6;
            key.extend(self.cache_host[offset..offset+count*6].iter().enumerate()
                .filter(|(word,_)| !matches!(kind, LayerKind::SparseAttention) || word%6 != 4)
                .map(|(_,word)| u64::from(word.to_bits())));
        }
        if let Some(rows)=sequence.as_deref() && rows.len()==1 { key.push(rows[0].session.verify_hidden.as_ref().unwrap().device_address()); }
        let (b, layers) = (self.model.backend, self.model.config.layers.len());
        let mut graphs = std::mem::take(&mut self.typed_graphs);
        let result = graph::replay(b, &mut graphs, key, lives, layers, count, || {
            self.enqueue_rows(count, logits, position, sequence.as_deref_mut(), layout)?;
            if greedy {
                b.enqueue_argmax_batch_device(&self.logits, &mut self.argmax, &mut self.selected, count)?;
            }
            Ok(())
        });
        self.typed_graphs = graphs; result
    }
    fn enqueue_rows(&mut self, count: usize, logits: bool, position: usize,
        mut sequence: Option<&mut [VerifyRow<'_, '_, 'a, 'b>]>, layout: &[(usize, usize)]) -> Result<()> {
        let model = self.model;
        let (b, c) = (model.backend, &model.config);
        if model.bound.as_ref().is_some_and(|bound| bound.group.distributed()) {
            model.embedding_tokens(&self.host_ids, &mut self.hidden)?;
        } else { model.embedding_rows(&self.ids, &mut self.hidden)?; }
        for (layer, kind) in c.layers.iter().enumerate() {
            let p = format!("{PREFIX}.layers.{layer}");
            model.profile_reads(Some(layer), || vec![(0, 1); count]);
            if layer == 0 {
                Session::normalize(b, model, &self.hidden, model.matrix(&format!("{p}.input_layernorm.weight"))?.device_address(), &self.normalized, count)?;
            }
            match kind {
                LayerKind::DeltaNet => {
                    let a = format!("{p}.linear_attn");
                    let binding = model.step_binding(layer)?; let widths=binding.projection_rows();
                    let pair = model.bound.is_some() && widths[2]<512 && widths[2]%4==0 && c.hidden>512;
                    for (name, out, reuse) in [("in_proj_qkv", &mut self.gdn.qkv, false), ("in_proj_z", &mut self.gdn.z, true),
                        ("in_proj_a", &mut self.gdn.a, true), ("in_proj_b", &mut self.gdn.b, true)] {
                        let width = match name { "in_proj_qkv"=>widths[0], "in_proj_z"=>widths[1], _=>widths[2] };
                        out.resize_within_capacity(count * width)?;
                        if pair && matches!(name,"in_proj_a"|"in_proj_b") { continue; }
                        Self::project(model, &format!("{a}.{name}.weight"), &self.normalized, out, &mut self.input_pack, count, reuse)?;
                    }
                    if let Some(bound)=model.bound.as_ref().filter(|_|pair) {
                        let first=bound.canonical.get(&format!("{a}.in_proj_a.weight")).ok_or("GDN a receipt absent")?;
                        let second=bound.canonical.get(&format!("{a}.in_proj_b.weight")).ok_or("GDN b receipt absent")?;
                        first.project_pair(second,&self.normalized,&mut self.gdn.a,&mut self.gdn.b,count)?;
                    }
                    self.gdn.gated.resize_within_capacity(count * (c.linear_value_heads * c.linear_value_dim) as usize)?;
                    let weights = [model.matrix(&format!("{a}.conv1d.weight"))?.device_address(),
                        model.address(&format!("{a}.A_log"))?, model.address(&format!("{a}.dt_bias"))?,
                        model.address(&format!("{a}.norm.weight"))?];
                    if binding.geometry()[4] != model.scalar_precision(&a) { return Err("GDN resident precision differs from plan".into()); }
                    let offset = layer * self.capacity * 6;
                    if let (Some(step),Some(chats)) = (&mut self.token_step,sequence.as_deref_mut()) {
                        let history_outputs: Vec<_> = layout.iter().map(|&(chat,step)| {
                            let s=&mut chats[chat].session;
                            if step+1<chats[chat].tokens.len() { s.gdn_cache[layer].as_ref().unwrap().history_destination(s.rollback_state[step][layer].as_mut().unwrap()) } else { 0 }
                        }).collect();
                        let descriptors=self.cache_rows.slice(offset..offset+count*6)?;
                        step.run(&binding,b,&mut self.gdn,weights,&descriptors,&self.cache_host[offset..offset+count*6],0,count,&history_outputs)?;
                    } else if let Some(chats) = sequence.as_deref_mut() {
                        let mut begin = 0;
                        while begin < count {
                            let step = layout[begin].1;
                            let end = layout[begin..].iter().position(|r| r.1 != step).map_or(count, |n| begin + n);
                            let descriptors=self.cache_rows.slice(offset + begin*6..offset + end*6)?;
                            if let Some(step)=&mut self.token_step { step.run(&binding,b,&mut self.gdn,weights,&descriptors,&self.cache_host[offset+begin*6..offset+end*6],begin,end-begin,&[])?; }
                            else { unsafe { binding.run(b,&mut self.gdn,weights,&descriptors,begin,end-begin)?; } }
                            for &(chat, step) in &layout[begin..end] {
                                let retained = step + 1 < chats[chat].tokens.len();
                                let s = &mut chats[chat].session;
                                if retained {
                                    s.gdn_cache[layer].as_ref().unwrap().copy_history(b, s.rollback_state[step][layer].as_mut().unwrap())?;
                                }
                            }
                            begin = end;
                        }
                    } else {
                        let descriptors=self.cache_rows.slice(offset..offset+count*6)?;
                        if let Some(step)=&mut self.token_step { step.run(&binding,b,&mut self.gdn,weights,&descriptors,&self.cache_host[offset..offset+count*6],0,count,&[])?; }
                        else { unsafe { binding.run(b,&mut self.gdn,weights,&descriptors,0,count)?; } }
                    }
                    Self::project(model, &format!("{a}.out_proj.weight"), &self.gdn.gated,
                        &mut self.block, &mut self.input_pack, count, false)?;
                }
                LayerKind::SparseAttention => {
                    let a = format!("{p}.self_attn");
                    let norms = [model.address(&format!("{a}.q_norm.weight"))?,
                        model.address(&format!("{a}.k_norm.weight"))?];
                    let offset = layer * self.capacity * 6;
                    let (_, replay_position) = b.decode_position(position);
                    let h = c.hidden as usize;
                    // A prompt group visits its canonical tiles in order; each writes
                    // its KV rows before the next tile attends to them.
                    let tiles: Vec<_> = if self.prompting && !self.prompt_tiles.is_empty() {
                        self.prompt_tiles.iter().scan(0, |at, &n| { let first = *at; *at += n; Some((first, n)) }).collect()
                    } else { (0..count).step_by(schedule::PROMPT_ROWS).map(|first| (first, schedule::PROMPT_ROWS.min(count-first))).collect() };
                    for (first, rows) in tiles {
                        self.attention.resize(c, rows)?;
                        let input = self.normalized.slice(first * h..(first + rows) * h)?;
                        for (name, out, reuse) in [("q_proj", &mut self.attention.qg, false), ("k_proj", &mut self.attention.k, true), ("v_proj", &mut self.attention.v, true)] {
                            Self::project(model, &format!("{a}.{name}.weight"), &input,
                                out, &mut self.input_pack, rows, reuse)?;
                        }
                        let descriptors = &self.cache_rows.slice(offset + first * 6..offset + (first + rows) * 6)?;
                        match crate::flash::qsa::prompt_past(&self.cache_host[offset + first * 6..offset + (first + rows) * 6]) {
                            Some(past) if self.prompting && replay_position == 0 =>
                                self.attention.attend_prompt(b, c, norms, model.frequencies.device_address(), descriptors, rows, past)?,
                            _ => self.attention.attend_batch(b, c, norms, model.frequencies.device_address(), descriptors, rows, replay_position)?,
                        }
                        Self::project(model, &format!("{a}.o_proj.weight"), &self.attention.gated,
                            &mut self.block.slice(first * h..(first + rows) * h)?, &mut self.input_pack, rows, false)?;
                    }
                }
            }
            model.reduce_partial(tp::Step::Mixer(model.layer_place(layer)),count,&mut self.block)?;
            self.residual_norm(model.matrix(&format!("{p}.post_attention_layernorm.weight"))?.device_address(), count)?;
            if let Some(moe) = &mut self.typed_moe { moe.run(model,layer,&self.normalized,&mut self.block,&mut self.input_pack,count)?; }
            else if let Some(moe) = &mut self.moe {
                moe.run(model, layer, &self.normalized, &mut self.block, &mut self.weight_pack,
                    &mut self.input_pack, count, true, &[])?;
            } else {
            for (name, out, reuse) in [("gate_proj", &mut self.gate, false), ("up_proj", &mut self.up, true)] {
                b.profile_phase(16, || b.matmul_decode_bf16_reusing_input(model.matrix(&format!("{p}.mlp.{name}.weight"))?,
                    &self.normalized, out, &mut self.input_pack, count, reuse))?;
            }
            launch(b, "flash_silu", self.gate.len(), &mut [Ptr(self.gate.device_address()), Ptr(self.up.device_address()), U32(self.gate.len() as u32), F32(1.0), U32(1)])?;
            b.profile_phase(17, || b.matmul_decode_bf16_reusing_input(model.matrix(&format!("{p}.mlp.down_proj.weight"))?,
                &self.gate, &mut self.block, &mut self.input_pack, count, false))?;
            }
            if layer + 1 < c.layers.len() {
                self.residual_norm(model.matrix(&format!("{PREFIX}.layers.{}.input_layernorm.weight", layer + 1))?.device_address(), count)?;
            } else {
                launch(b, "flash_residual_bf16", self.hidden.len(), &mut [Ptr(self.hidden.device_address()), Ptr(self.block.device_address()), U32(self.hidden.len() as u32)])?;
            }
        }
        if logits {
            model.profile_reads(None, Vec::new);
            self.logits.resize_within_capacity(count * c.vocab as usize)?;
            // A singleton verifier can publish directly into its owned retained rows.
            let retained = if model.bound.is_some() { sequence.as_deref().filter(|rows|rows.len()==1)
                .map(|rows|rows[0].session.verify_hidden.as_ref().unwrap().slice(0..count*c.hidden as usize)).transpose()? } else { None };
            let normalized=retained.as_ref().unwrap_or(&self.normalized);
            Session::normalize(b, model, &self.hidden, model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(), normalized, count)?;
            b.profile_output_head(|| if model.bound.is_some() { model.head_rows(normalized,&mut self.logits,&mut self.input_pack,count) }
                else { b.matmul_decode_bf16_device(model.matrix("lm_head.weight")?,normalized,&mut self.logits,count) })?;
        }
        Ok(())
    }
    fn residual_norm(&self, weight: u64, count: usize) -> Result<()> {
        launch_norm(self.model.backend, &self.hidden, Some(&self.block), weight,
            &self.normalized, self.model, count, Some(2), false)
    }
}
