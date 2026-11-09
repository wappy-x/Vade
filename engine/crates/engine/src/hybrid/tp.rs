//! Shape-driven collectives; Exchange::reduce binds the existing rank-ordered ring.
//! Transport preserves FP32 partials. Mixer outputs round once. After the rank sum,
//! MoE finish rounds the routed down projection and adds the shared expert last.
use engine_fabric::exchange::Exchange;
use engine_model::{plan::ModelPlan, safetensors::Dtype, shapes::*, split::{Reduction, Selection, SplitPlan}};
use std::ops::Range;
type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step { Embedding, Mixer(Place), FeedForward(Place), Head }
/// Retain the exact plan/split used by the typed loader. No transport ownership.
pub struct Binding<'p> { plan: &'p ModelPlan, split: &'p SplitPlan, rank: usize }
fn distributed(split:&SplitPlan)->bool { split.ranks>1 }
fn size(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| "TP extent overflow".into()) }
fn product(a: usize, b: usize) -> Result<usize> { a.checked_mul(b).ok_or("TP extent overflow".into()) }
impl<'p> Binding<'p> {
    /// Every rank sees the same command epoch and rollback prefix before execution.
    /// Router rejection is an OR vote per row; unrelated rows keep their own vote.
    pub fn agree<E: Exchange>(&self, e: &mut E, command: &[u64], rejected: &[bool]) -> Result<Vec<bool>> {
        let mut lengths=vec![0u8; self.split.ranks as usize*16];
        lengths[self.rank*16..self.rank*16+8].copy_from_slice(&(command.len() as u64).to_le_bytes());
        lengths[self.rank*16+8..self.rank*16+16].copy_from_slice(&(rejected.len() as u64).to_le_bytes());
        e.gather_bytes(&mut lengths, &(0..=self.split.ranks as usize).map(|r|r*16).collect::<Vec<_>>())?;
        if lengths.as_chunks::<16>().0.iter().any(|v| v!=&lengths[..16]) || command.is_empty() { return Err("TP command extent disagreement".into()); }
        let bytes = product(command.len(), 8)?.checked_add(rejected.len()).ok_or("TP vote extent")?;
        let offsets: Vec<_> = (0..=self.split.ranks as usize).map(|r| product(r, bytes)).collect::<Result<_>>()?;
        let mut votes = vec![0u8; *offsets.last().unwrap()];
        let local = &mut votes[offsets[self.rank]..offsets[self.rank+1]];
        for (out, value) in local.as_chunks_mut::<8>().0.iter_mut().take(command.len()).zip(command) { out.copy_from_slice(&value.to_le_bytes()); }
        for (out, value) in local[command.len()*8..].iter_mut().zip(rejected) { *out=u8::from(*value); }
        e.gather_bytes(&mut votes, &offsets)?;
        let words=command.len()*8;
        if votes.chunks_exact(bytes).any(|v| v[..words] != votes[..words]) { return Err("TP command epoch disagreement".into()); }
        Ok((0..rejected.len()).map(|i| votes.chunks_exact(bytes).any(|v| v[words+i]!=0)).collect())
    }
    pub fn new(plan: &'p ModelPlan, split: &'p SplitPlan, rank: usize) -> Result<Self> {
        if split.owners.len() != size(u64::from(split.ranks))? || rank >= split.owners.len()
            || split.owners.is_empty() || split.owners.iter().any(|o| o.output_rows.is_some())
            || plan.shape.residual != Residual::Single || plan.shape.lexical.is_some()
            || plan.shape.hidden == 0 || plan.shape.vocab == 0 {
            return Err("TP requires single residual and column-owned down projections".into());
        }
        let b = Self { plan, split, rank }; b.vocab_offsets(1)?;
        for (place, layer) in b.layers(false).chain(b.layers(true)) {
            let part = match layer.mixer { Mixer::Attention(a) if a.index.is_none() => Part::Output,
                Mixer::Linear(_) => Part::LinearOut, _ => return Err("TP indexed attention unsupported".into()) };
            b.segments(place, part)?;
            match layer.ffn {
                Ffn::Dense { .. } => { b.segments(place, Part::Dense(Proj::Down))?; }
                Ffn::Experts(e) => {
                    for id in 0..u32::try_from(e.count).map_err(|_| "TP expert count")? {
                        b.segments(place, Part::Expert { id: Some(id), proj: Proj::Down })?;
                    }
                    if e.shared_width != 0 { b.segments(place, Part::Shared(Proj::Down))?; }
                }
            }
        }
        Ok(b)
    }
    fn layers(&self, draft: bool) -> impl Iterator<Item = (Place, &LayerShape)> {
        let layers = if draft { &self.plan.shape.draft } else { &self.plan.shape.layers };
        layers.iter().enumerate().map(move |(i, l)|
            (if draft { Place::Draft { layer: Some(i as u32) } } else { Place::Layer(i as u32) }, l))
    }
    fn layer(&self, place: Place) -> Result<&LayerShape> {
        let (layers, i) = match place { Place::Layer(i) => (&self.plan.shape.layers, i),
            Place::Draft { layer: Some(i) } => (&self.plan.shape.draft, i), _ => return Err("TP layer place".into()) };
        layers.get(size(u64::from(i))?).ok_or("TP layer index".into())
    }
    /// Norms gather once during setup, in inventory order, before any head norm.
    pub fn norms(&self) -> impl Iterator<Item = &TensorEntry> {
        self.plan.shape.tensors.iter().filter(|t| t.role.store == Store::Weight
            && matches!(t.role.part, Part::QueryNorm | Part::KeyNorm | Part::LinearNorm)
            && matches!(t.role.place, Place::Layer(_) | Place::Draft { layer: Some(_) })
            && self.split.selection(self.rank as u32, t).is_ok_and(|s| s != Selection::Absent))
    }
    /// Same steps for every row/phase: global embedding, O sum, down sum, head.
    /// Draft uses the shared embedding/head; replicated concat-fc/norms need no sum.
    pub fn schedule(&self, draft: bool) -> Result<Vec<Step>> {
        if draft && !self.plan.shape.tensors.iter().any(|t| matches!(t.role.place, Place::Draft { .. })
            && self.split.selection(self.rank as u32, t).is_ok_and(|s| s != Selection::Absent)) {
            return Err("TP draft weights not selected".into());
        }
        let mut steps = vec![Step::Embedding];
        for (place, _) in self.layers(draft) { steps.extend([Step::Mixer(place), Step::FeedForward(place)]); }
        steps.push(Step::Head); Ok(steps)
    }
    /// Original K128 coordinates; local matrices must come from these loader slices.
    pub fn segments(&self, place: Place, part: Part) -> Result<Vec<Range<u64>>> {
        let l = self.layer(place)?;
        let (reduction, width) = match (part, l.mixer, l.ffn) {
            (Part::Output, Mixer::Attention(a), _) => (Reduction::Attention, a.query_heads.checked_mul(a.head_dim)),
            (Part::LinearOut, Mixer::Linear(l), _) => (Reduction::Linear, l.value_heads.checked_mul(l.value_dim)),
            (Part::Dense(Proj::Down), _, Ffn::Dense { width }) => (Reduction::Dense, Some(width)),
            (Part::Shared(Proj::Down), _, Ffn::Experts(e)) => (Reduction::Shared, Some(e.shared_width)),
            (Part::Expert { id: Some(id), proj: Proj::Down }, _, Ffn::Experts(e)) if u64::from(id) < e.count => (Reduction::Expert(id), Some(e.width)),
            _ => return Err("TP partial requires row-parallel O/down role".into()),
        };
        let ranges = self.split.segments(reduction).map_err(|e| e.to_string())?;
        let mut end = 0;
        for r in &ranges {
            if r.start != end || r.end <= r.start || r.start % FP8_BLOCK != 0 || r.end % FP8_BLOCK != 0 {
                return Err("TP partial requires complete contiguous K128 owners".into());
            }
            end = r.end;
        }
        if Some(end) != width { return Err("TP split disagrees with layer geometry".into()); }
        Ok(ranges)
    }
    fn elements(&self, step: Step, rows: usize) -> Result<(usize, bool)> {
        if rows == 0 { return Err("TP empty row cohort".into()); }
        let (slots, round) = match step {
            Step::Mixer(place) => { self.layer(place)?; (1, true) }
            Step::FeedForward(place) => match self.layer(place)?.ffn {
                Ffn::Dense { .. } => (1, true),
                Ffn::Experts(e) => (size(e.selected.checked_add(1).ok_or("TP slots overflow")?)?, false),
            },
            _ => return Err("TP sum step required".into()),
        };
        Ok((product(product(rows, size(self.plan.shape.hidden)?)?, slots)?, round))
    }
    /// Exchange must implement P0 then P1.. in FP32, as Ring::all_reduce_f32.
    /// MoE packets are [row, selected+1, H], ascending expert IDs, shared last.
    pub fn reduce<E: Exchange>(&self, exchange: &mut E, step: Step, rows: usize,
        values: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        let (n, round) = self.elements(step, rows)?;
        if values.len() != n || scratch.len() < product(n, self.split.owners.len())? {
            return Err("TP partial/scratch extent".into());
        }
        exchange.reduce(values, scratch)?;
        if round { for value in values { *value = spark_gpu_probe::bf16::rounded(*value); } }
        Ok(())
    }
    /// Preserve BF16/F32 storage bits; the full vector exists before normalization.
    pub fn gather_norm<E: Exchange>(&self, exchange: &mut E, tensor: &TensorEntry,
        local: &[u8], full: &mut [u8]) -> Result<()> {
        let layer = self.layer(tensor.role.place)?;
        let width = match (tensor.role.part, layer.mixer) {
            (Part::QueryNorm | Part::KeyNorm, Mixer::Attention(a)) => a.head_dim,
            (Part::LinearNorm, Mixer::Linear(l)) => l.value_dim, _ => return Err("TP scoped norm role".into()),
        };
        if tensor.role.store != Store::Weight || tensor.shape != [width]
            || !matches!(tensor.dtype, Dtype::Bf16 | Dtype::F32) { return Err("TP norm storage".into()); }
        let mut offsets = vec![0];
        for rank in 0..self.split.ranks {
            let Selection::Rows(rs) = self.split.selection(rank, tensor).map_err(|e| e.to_string())? else { return Err("TP norm selection".into()); };
            if rs.len() != 1 || rs[0].start != (*offsets.last().unwrap() as u64 / tensor.dtype.bytes()) || rs[0].end < rs[0].start { return Err("TP norm partition".into()); }
            offsets.push(product(size(rs[0].end)?, size(tensor.dtype.bytes())?)?);
        }
        if *offsets.last().unwrap() != product(size(width)?, size(tensor.dtype.bytes())?)? { return Err("TP norm width".into()); }
        self.gather_bytes(exchange, local, full, &offsets)
    }
    fn gather_bytes<E: Exchange>(&self, e: &mut E, local: &[u8], full: &mut [u8], offsets: &[usize]) -> Result<()> {
        let r = offsets[self.rank]..offsets[self.rank+1];
        if full.len() != *offsets.last().unwrap() || local.len() != r.len() { return Err("TP norm byte extent".into()); }
        full[r].copy_from_slice(local); e.gather_bytes(full, offsets)
    }
    /// Only the global token's vocabulary owner supplies H values. Other ranks
    /// contribute an empty range: a gather preserves signed zeros, unlike a sum.
    pub fn local_token(&self, token: u32) -> Result<Option<usize>> {
        if u64::from(token) >= self.plan.shape.vocab { return Err("TP global token ID".into()); }
        let owner = &self.split.owners[self.rank].vocab;
        if owner.contains(&u64::from(token)) { Ok(Some(size(u64::from(token)-owner.start)?)) } else { Ok(None) }
    }
    pub fn embedding<E: Exchange>(&self, e: &mut E, token: u32, local: &[f32], full: &mut [f32]) -> Result<()> {
        if u64::from(token) >= self.plan.shape.vocab { return Err("TP global token ID".into()); }
        let h = size(self.plan.shape.hidden)?; let mut offsets = vec![0];
        for o in &self.split.owners { offsets.push(offsets.last().unwrap() + if o.vocab.contains(&u64::from(token)) { h } else { 0 }); }
        if full.len() != h || local.len() != offsets[self.rank+1]-offsets[self.rank] { return Err("TP embedding extent".into()); }
        full[offsets[self.rank]..offsets[self.rank+1]].copy_from_slice(local); e.gather_f32(full, &offsets)
    }
    fn vocab_offsets(&self, rows: usize) -> Result<Vec<usize>> {
        if rows == 0 { return Err("TP empty head cohort".into()); }
        let mut offsets = vec![0]; let mut end = 0;
        for o in &self.split.owners {
            if o.vocab.start != end || o.vocab.end < end { return Err("TP vocabulary partition".into()); }
            end = o.vocab.end; offsets.push(product(size(end)?, rows)?);
        }
        if end != self.plan.shape.vocab { return Err("TP vocabulary extent".into()); } Ok(offsets)
    }
    /// Wire is rank-major [rank,row,local_vocab]; returned logits are [row,vocab].
    pub fn heads<E: Exchange>(&self, e: &mut E, rows: usize, local: &[f32], wire: &mut [f32], full: &mut [f32]) -> Result<()> {
        let offsets = self.vocab_offsets(rows)?; let n = *offsets.last().unwrap();
        if wire.len() != n || full.len() != n || local.len() != offsets[self.rank+1]-offsets[self.rank] { return Err("TP head extent".into()); }
        wire[offsets[self.rank]..offsets[self.rank+1]].copy_from_slice(local); e.gather_f32(wire, &offsets)?;
        let vocab = size(self.plan.shape.vocab)?;
        for (rank, o) in self.split.owners.iter().enumerate() {
            let (start, end) = (size(o.vocab.start)?, size(o.vocab.end)?); let width = end-start;
            for row in 0..rows { full[row*vocab+start..row*vocab+end].copy_from_slice(&wire[offsets[rank]+row*width..offsets[rank]+(row+1)*width]); }
        }
        Ok(())
    }
    /// Caller retains the correctly selected local matrix/context through completion.
    #[cfg(target_os = "linux")]
    pub fn project_partial(&self, b: &spark_gpu_probe::driver::Backend<'_>, place: Place, part: Part,
        weights: &spark_gpu_probe::driver::Matrix<'_>, input: &spark_gpu_probe::driver::DeviceVector<'_>,
        output: &mut spark_gpu_probe::driver::DeviceVector<'_>, scratch: &mut spark_gpu_probe::driver::DeviceVector<'_>, rows: usize) -> Result<()> {
        let r = self.segments(place, part)?.remove(self.rank);
        if weights.rows() != size(self.plan.shape.hidden)? || weights.cols() != size(r.end-r.start)? {
            return Err("TP local partial weight shape".into());
        }
        b.fixed_fp8_projection(weights, input, output, scratch, rows, false)
    }
    /// Readback waits on the compute stream; the reduced result returns on that stream.
    #[cfg(target_os = "linux")]
    pub fn reduce_device<E: Exchange>(&self, b: &spark_gpu_probe::driver::Backend<'_>, e: &mut E,
        step: Step, rows: usize, values: &mut spark_gpu_probe::driver::DeviceVector<'_>, scratch: &mut [f32]) -> Result<()> {
        if values.len() != self.elements(step, rows)?.0 { return Err("TP device partial extent".into()); }
        b.exchange_vector(values, |v| self.reduce(e, step, rows, v, scratch))
    }
}

#[path="tp_group.rs"] mod group;
pub use group::{Group,LocalExchange};

/// One bounded host/wire envelope for setup, target, draft and verification.
/// CPU packing/gather lifetimes are funded separately from registered mailboxes.
#[derive(Debug, Clone, Copy)]
pub struct TransportPlan { ranks: usize, wire: usize, host: usize }
impl Binding<'_> {
    /// Header-derived local arena plus existing fixed-row owner bounds. Never
    /// divide a world1 session: only selected heads/intermediate widths shrink.
    #[cfg(target_os="linux")]
    pub fn rank_reservations(&self,index:&engine_model::safetensors::Index,
        headers:&std::collections::BTreeMap<String,engine_model::safetensors::Header>,context:usize,draft:bool)->Result<[usize;3]> {
        use super::{Model,weights::LoadPlan,plan,decode,mtp};
        if draft!=self.plan.shape.tensors.iter().any(|t|matches!(t.role.place,Place::Draft{..})
            && self.split.selection(self.rank as u32,t).is_ok_and(|s|s!=Selection::Absent)) { return Err("TP draft ledger disagrees with selected weights".into()); }
        let whole=SplitPlan::new(&self.plan.shape,&engine_model::split::SplitRequest { ranks:1,memory:vec![u64::MAX],reserve:vec![0],
            context:0,sequences:0,cache_bytes:2,state_bytes:4,draft,expert_rows:false }).map_err(|e|e.to_string())?;
        let whole_arena=LoadPlan::new(self.plan,index,headers,&whole)?.bytes();
        let local=LoadPlan::for_rank(self.plan,index,headers,self.split,self.rank as u32)?;
        let norm_bytes=self.norms().try_fold(0usize,|n,t|n.checked_add(size(t.shape.first()?.checked_mul(t.dtype.bytes())?).ok()?)).ok_or("TP norm owner overflow")?;
        let resident=Model::resident_reservation_bytes(self.plan,index,headers,draft)?.checked_sub(whole_arena)
            .and_then(|n|n.checked_add(local.bytes())).and_then(|n|n.checked_add(norm_bytes)).ok_or("TP resident owner overflow")?;
        let mut c=plan::config(self.plan)?; let o=&self.split.owners[self.rank];
        c.query_heads=o.query_heads.end-o.query_heads.start; c.kv_heads=o.kv_heads.end-o.kv_heads.start;
        c.linear_key_heads=o.key_heads.end-o.key_heads.start; c.linear_value_heads=o.value_heads.end-o.value_heads.start;
        c.shared_intermediate=o.shared.end-o.shared.start;
        c.intermediate=(0..u32::try_from(c.experts).map_err(|_|"TP expert count")?).map(|id|self.split.expert_blocks(self.rank as u32,id)
            .map_err(|e|e.to_string()).and_then(|r|(r.end-r.start).checked_mul(FP8_BLOCK).ok_or("TP expert extent overflow".into())))
            .collect::<Result<Vec<_>>>()?.into_iter().max().ok_or("TP expert owners absent")?;
        let shared=Model::fixed_decode_bytes(&c,if draft {decode::MAX_VERIFY_ROWS}else{decode::MAX_CHAT_ROWS})
            .and_then(|n|n.checked_add(if draft {mtp::fixed_reservation_bytes(&c,64.min(context),crate::hybrid_draft::Options::startup())?}else{0})).ok_or("TP shared owner overflow")?;
        let request=Model::fixed_request_bytes(&c,context,context,draft).ok_or("TP request owner overflow")?;
        Ok([resident,shared,request])
    }
    /// command_words includes the exact plan/split/member setup representation
    /// and the largest route/epoch command; over-envelope work is refused.
    pub fn transport_plan(&self, rows: usize, command_words: usize) -> Result<TransportPlan> {
        if rows==0 || command_words==0 { return Err("invalid TP transport envelope".into()); }
        let ranks=self.split.owners.len();
        // Prompt replay includes the retained tail row, beyond the decode/verifier cohort.
        let rows=if distributed(self.split) { rows.max(super::schedule::PROMPT_ROWS+1) } else { rows };
        let mut wire=product(ranks,product(command_words,8)?.checked_add(rows).ok_or("TP vote extent")?)?.max(product(ranks,112)?);
        wire=wire.max(product(product(rows,size(self.plan.shape.vocab.max(self.plan.shape.hidden))?)?,4)?);
        let group_rows=if distributed(self.split) { rows.max(super::schedule::PROMPT_GROUP) } else { rows };
        for (place,_) in self.layers(false).chain(self.layers(true)) {
            for step in [Step::Mixer(place),Step::FeedForward(place)] {
                wire=wire.max(product(product(self.elements(step,group_rows)?.0,ranks)?,4)?);
            }
        }
        for t in self.norms() { wire=wire.max(size(t.shape.first().ok_or("TP norm shape")?.checked_mul(t.dtype.bytes()).ok_or("TP norm extent")?)?); }
        // Full heads + rank-major wire + reduction/readback + command packing.
        let host=product(wire,6)?;
        if host>isize::MAX as usize { return Err("TP host envelope exceeds allocation bounds".into()); }
        Ok(TransportPlan { ranks,wire,host })
    }
}
pub struct PreparedTransport { plan: TransportPlan, mailbox: usize }
impl TransportPlan {
    pub fn wire_bytes(self) -> usize { self.wire }
    pub fn host_bytes(self) -> usize { self.host }
    pub fn ranks(self) -> usize { self.ranks }
    /// No weight callback is exposed until registration and sealing succeed.
    pub fn prepare<T>(self, transport: &mut T,
        register: impl FnOnce(&mut T,usize)->Result<usize>, seal: impl FnOnce(&mut T)->Result<()>) -> Result<PreparedTransport> {
        let mailbox=register(transport,self.wire)?; seal(transport)?;
        Ok(PreparedTransport { plan:self,mailbox })
    }
}
impl PreparedTransport {
    pub fn mailbox_bytes(&self) -> usize { self.mailbox }
    pub fn host_bytes(&self) -> usize { self.plan.host }
    pub fn before_weights<E:Exchange,A,T>(self, exchange:E,
        admit:impl FnOnce(&mut BoundedExchange<E>,usize,usize)->Result<A>,
        load:impl FnOnce(BoundedExchange<E>,A)->Result<T>)->Result<T> {
        let mut e=BoundedExchange { inner:exchange,plan:self.plan };
        let admitted=admit(&mut e,self.mailbox,self.plan.host)?;
        load(e,admitted)
    }
}
/// Oversized messages refuse before calling the sealed transport. Arithmetic
/// remains the existing Exchange implementation, including the fixed rank fold.
pub struct BoundedExchange<E> { inner:E, plan:TransportPlan }
impl<E:Exchange> Exchange for BoundedExchange<E> {
    fn gather_bytes(&mut self,v:&mut [u8],o:&[usize])->Result<()> {
        if v.len()>self.plan.wire || o.len()!=self.plan.ranks+1 { return Err("TP gather exceeds sealed envelope".into()); }
        self.inner.gather_bytes(v,o)
    }
    fn gather_f32(&mut self,v:&mut [f32],o:&[usize])->Result<()> {
        if product(v.len(),4)?>self.plan.wire || o.len()!=self.plan.ranks+1 { return Err("TP head exceeds sealed envelope".into()); }
        self.inner.gather_f32(v,o)
    }
    fn reduce(&mut self,v:&mut [f32],s:&mut [f32])->Result<()> {
        let n=product(v.len(),self.plan.ranks)?;
        if product(n,4)?>self.plan.wire || s.len()<n { return Err("TP reduction exceeds sealed envelope".into()); }
        self.inner.reduce(v,s)
    }
}
