//! Immutable shape terms for request admission. Build once at service startup.
#![forbid(unsafe_code)]
use engine_model::plan::{TextDescription,admission::CheckpointAdmission,description::TokenMixer,layer::Topology,state::{PlaneKind,StatePlan}};
use crate::{Result,bounded_decoder::{expanded_bytes,RESERVED_ROWS}};

/// Two cache banks can coexist. Each bank can keep HELD_ROOM future rows;
/// round each layer's spare bytes up to the device allocator's page boundary.
pub fn cache_append_reservation_bytes(description:&TextDescription)->Result<usize> {
    description.layers.iter().try_fold(0usize,|sum,layer| {
        let TokenMixer::LatentAttention(a)=layer.mixer else {return Some(sum);};
        let room=crate::bounded_decoder::HELD_ROOM.checked_mul(usize::try_from(a.kv_rank).ok()?)?.checked_mul(4)?;
        sum.checked_add(room.checked_add(4095)?.checked_div(4096)?.checked_mul(4096)?.checked_mul(2)?)
    }).ok_or_else(||"latent cache reservation overflow".into())
}

/// The indexer key and gate caches of each latent layer also append one row
/// (indexer.dim words) a decode step. Same two banks and HELD_ROOM rows, two caches.
pub fn index_append_reservation_bytes(description:&TextDescription)->Result<usize> {
    description.layers.iter().try_fold(0usize,|sum,layer| {
        let TokenMixer::LatentAttention(a)=layer.mixer else {return Some(sum);};
        let room=crate::bounded_decoder::HELD_ROOM.checked_mul(usize::try_from(a.indexer.dim).ok()?)?.checked_mul(4)?;
        sum.checked_add(room.checked_add(4095)?.checked_div(4096)?.checked_mul(4096)?.checked_mul(2)?.checked_mul(2)?)
    }).ok_or_else(||"index cache reservation overflow".into())
}

/// Every append reservation a session funds once, outside the request maxima.
pub fn append_reservation_bytes(description:&TextDescription)->Result<usize> {
    cache_append_reservation_bytes(description)?.checked_add(index_append_reservation_bytes(description)?).ok_or_else(||"append reservation overflow".into())
}

/// State storage is a sum of fixed planes, token rows, and completed pool rows.
/// All planes contain four-byte elements and use four-byte alignment, so their
/// lengths add without padding. No layer program is needed for a new extent.
pub struct StateChargePlan {capacity:usize,fixed:usize,row:usize,pools:Vec<(usize,usize)>,cache_slack:usize,index_slack:usize}
impl StateChargePlan {
    pub fn new(description:&TextDescription,capacity:usize)->Result<Self> {
        let plan=StatePlan::new(description,Topology {rank:0,parts:1},capacity as u64,isize::MAX as usize).map_err(|e|e.to_string())?;
        let (cache_slack,index_slack)=(cache_append_reservation_bytes(description)?,index_append_reservation_bytes(description)?);
        cache_slack.checked_add(index_slack).ok_or("append reservation overflow")?;
        let mut out=Self {capacity,fixed:0,row:0,pools:Vec::new(),cache_slack,index_slack};
        for plane in plan.planes() {
            let bytes=plane.allocation.shape.iter().skip(1).try_fold(4usize,|n,&v|n.checked_mul(usize::try_from(v).ok()?)).ok_or("state row charge overflow")?;
            match plane.kind {
                PlaneKind::KdaState|PlaneKind::KdaHistory=>out.fixed=out.fixed.checked_add(plane.allocation.bytes.len()).ok_or("fixed state charge overflow")?,
                PlaneKind::Pools=>{let TokenMixer::LatentAttention(a)=description.layers[plane.layer].mixer else {return Err("pool charge has no indexer".into());};out.pools.push((usize::try_from(a.indexer.pool).map_err(|_|"pool divisor exceeds address space")?,bytes));},
                PlaneKind::Latent|PlaneKind::IndexKeys|PlaneKind::IndexGates=>out.row=out.row.checked_add(bytes).ok_or("state row sum overflow")?,
            }
        }
        if out.bytes(capacity)?!=plan.state_layout().bytes() {return Err("cached state charge differs from storage plan".into());}Ok(out)
    }
    pub fn bytes(&self,tokens:usize)->Result<usize> {
        if tokens==0 || tokens>self.capacity {return Err("state reservation exceeds context".into());}
        let base=self.row.checked_mul(tokens).and_then(|n|n.checked_add(self.fixed)).ok_or("state charge overflow")?;
        self.pools.iter().try_fold(base,|n,&(divisor,row)|n.checked_add((tokens/divisor).checked_mul(row)?)).ok_or_else(||"pool state charge overflow".into())
    }
    pub fn cache_append_reservation_bytes(&self)->usize {self.cache_slack}
    pub fn index_append_reservation_bytes(&self)->usize {self.index_slack}
    /// Checked at construction.
    pub fn append_reservation_bytes(&self)->usize {self.cache_slack+self.index_slack}
}

/// Request-dependent arithmetic uses only these retained shape coefficients.
pub struct SessionChargePlan {state:StateChargePlan,expanded_row:usize,expanded_layouts:Option<Vec<crate::sparse_heads::Layout>>,tail_row:usize,vocab:usize}
impl SessionChargePlan {
    pub fn new(checkpoint:&CheckpointAdmission,capacity:usize)->Result<Self> {
        Self::from_description(checkpoint.description(),capacity,expanded_bytes(checkpoint,1))
    }
    /// Use the compute owner's topology. Its expanded caches must own these local heads.
    pub fn for_sparse_heads(description:&TextDescription,topology:Topology,capacity:usize)->Result<Self> {
        let layouts=crate::sparse_heads::layouts(description,topology)?.into_iter().map(|(_,layout)|layout).collect::<Vec<_>>();
        for layout in &layouts {layout.allocation_bytes(capacity.min(RESERVED_ROWS))?;}
        let expanded_row=layouts.iter().try_fold(0usize,|sum,layout|sum.checked_add(layout.row_words().checked_mul(4)?)).ok_or("sparse expanded row charge overflow")?;
        let mut plan=Self::from_description(description,capacity,expanded_row)?;plan.expanded_layouts=Some(layouts);Ok(plan)
    }
    fn from_description(description:&TextDescription,capacity:usize,expanded_row:usize)->Result<Self> {
        let tail_row=description.layers.iter().try_fold(0usize,|n,l|match l.mixer {TokenMixer::Kda(k)=>(k.heads as usize).checked_mul(k.head_dim as usize).and_then(|v|v.checked_mul(8)).and_then(|v|n.checked_add(v)),_=>Some(n)}).ok_or("state tail charge overflow")?;
        let vocab=usize::try_from(description.vocab).ok().and_then(|v|v.checked_mul(16)).ok_or("vocabulary charge overflow")?;
        Ok(Self {state:StateChargePlan::new(description,capacity)?,expanded_row,expanded_layouts:None,tail_row,vocab})
    }
    pub fn state_bytes(&self,tokens:usize)->Result<usize> {self.state.bytes(tokens)}
    pub fn expanded_bytes(&self,tokens:usize)->usize {
        let rows=tokens.min(RESERVED_ROWS);
        match &self.expanded_layouts {Some(layouts)=>layouts.iter().try_fold(0usize,|sum,layout|sum.checked_add(layout.allocation_bytes(rows).ok()?)).unwrap_or(usize::MAX),
            None=>self.expanded_row.saturating_mul(rows)}
    }
    fn base_session(&self,prompt:usize,tokens:usize)->Result<usize> {
        if prompt>tokens {return Err("invalid service reservation extent".into());}
        let held=self.expanded_bytes(prompt);
        let early=self.state.bytes(prompt.min(RESERVED_ROWS).max(1))?.checked_mul(2).and_then(|n|held.checked_mul(2).and_then(|h|n.checked_add(h))).ok_or("early session charge overflow")?;
        let late=self.state.bytes(tokens)?.checked_mul(2).and_then(|n|n.checked_add(held)).ok_or("late session charge overflow")?;
        early.max(late).checked_add(self.tail_row.checked_mul(prompt.clamp(1,256)).ok_or("state tail charge overflow")?).and_then(|n|n.checked_add(self.vocab)).ok_or_else(||"session charge overflow".into())
    }
    pub fn session(&self,prompt:usize,tokens:usize)->Result<usize> {
        self.base_session(prompt,tokens)?.checked_add(self.append_reservation_bytes()).ok_or_else(||"cache session charge overflow".into())
    }
    pub fn cache_append_reservation_bytes(&self)->usize {self.state.cache_append_reservation_bytes()}
    pub fn index_append_reservation_bytes(&self)->usize {self.state.index_append_reservation_bytes()}
    pub fn append_reservation_bytes(&self)->usize {self.state.append_reservation_bytes()}
    /// Preserve the default service's older, conservative retained-row charge.
    pub fn request(&self,prompt:usize,tokens:usize)->Result<usize> {
        let legacy=self.state.bytes(tokens)?.checked_mul(2).and_then(|n|self.expanded_bytes(prompt).checked_mul(2).and_then(|h|n.checked_add(h))).and_then(|n|n.checked_add(self.vocab)).ok_or("request state reservation overflow")?;
        legacy.max(self.base_session(prompt,tokens)?).checked_add(self.append_reservation_bytes()).ok_or_else(||"cache request charge overflow".into())
    }
}
