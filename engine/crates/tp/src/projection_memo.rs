//! One startup charge and capacity for the optional projection cache.
use std::{cell::OnceCell,rc::Rc};
pub const FULL:usize=3<<30;
pub const RECORD_WORDS:usize=13;
pub fn record(summary:Option<Summary>)->[f32;RECORD_WORDS] {
    let mut out=[0.0;RECORD_WORDS];if let Some(s)=summary {out[0]=1.0;
        for (at,value) in [s.unabsorbed,s.selection,s.chunk].into_iter().enumerate() {for i in 0..4 {out[1+4*at+i]=((value as u64>>(16*i))&65535) as f32;}}
    }out
}
pub fn agree(records:&[f32],chats:usize)->Result<Plan,String> {
    if records.is_empty() || !records.len().is_multiple_of(RECORD_WORDS) {return Err("memo rank record count differs".into());}
    let mut summaries=Vec::new();let mut unknown=false;
    for r in records.as_chunks::<RECORD_WORDS>().0 {
        if r.iter().any(|v|!v.is_finite() || *v<0.0 || *v>65535.0 || v.fract()!=0.0) || r[0]>1.0 {return Err("invalid memo rank record".into());}
        if r[0]==0.0 {unknown=true;continue;}
        let mut values=[0;3];for (at,value) in values.iter_mut().enumerate() {let mut n=0u64;for i in 0..4 {n|=(r[1+4*at+i] as u64)<<(16*i);}*value=usize::try_from(n).map_err(|_|"memo rank extent exceeds address space")?;}
        summaries.push(Summary {unabsorbed:values[0],selection:values[1],chunk:values[2]});
    }
    let Some(first)=summaries.first().copied() else {return Ok(Plan::full());};
    if summaries.iter().any(|s|s.unabsorbed!=first.unabsorbed) {return Err("ranks disagree on loaded unabsorbed projections".into());}
    Ok(if unknown || summaries.iter().any(|s|*s!=first) {Plan::full()} else {Plan::new(Some(first),chats)})
}
pub fn slab_chunk_bytes(rows:usize,width:usize)->Option<usize> {
    rows.checked_mul(width)?.checked_mul(4)?.checked_add(4095).map(|n|n/4096*4096)
}
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct Summary {pub unabsorbed:usize,pub selection:usize,pub chunk:usize}
impl Summary {
    /// One unabsorbed loaded matrix, with complete local attention heads.
    pub fn add(&mut self,rows:usize,latent:usize,slots:usize,name:usize)->Option<()> {
        let packed=rows.checked_mul(2)?;
        let chunk=slab_chunk_bytes(crate::bounded_decoder::MEMO_SLAB_ROWS,rows/2)?;
        if rows==0 || !rows.is_multiple_of(2) || chunk==0 {return None;}
        // A latent row has one origin span. Two stored keys and one shared slot handle.
        let row=packed.checked_add(latent.checked_mul(4)?)?.checked_add(24+2*72+8)?.checked_add(name.checked_mul(2)?)?;
        let bytes=row.checked_mul(slots)?;
        let selection=bytes.checked_add(chunk-1)?.checked_div(chunk)?.checked_mul(chunk)?;
        self.unabsorbed=self.unabsorbed.checked_add(1)?;self.selection=self.selection.max(selection);self.chunk=self.chunk.max(chunk);Some(())
    }
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Plan {summary:Option<Summary>,limit:usize}
impl Plan {
    pub const fn full()->Self {Self {summary:None,limit:FULL}}
    pub fn new(summary:Option<Summary>,chats:usize)->Self {
        let Some(s)=summary else {return Self::full();};
        if s.unabsorbed==0 {return Self {summary:Some(s),limit:0};}
        let bytes=(|| {
            if chats==0 || s.chunk==0 {return None;}
            let bytes=s.unabsorbed.checked_mul(chats)?.checked_mul(s.selection)?.checked_mul(3)?.checked_add(1)?/2;
            bytes.checked_add(s.chunk-1)?.checked_div(s.chunk)?.checked_mul(s.chunk)
        })();
        match bytes {Some(limit)=>Self {summary:Some(s),limit:limit.min(FULL)},None=>Self::full()}
    }
    pub const fn bytes(self)->usize {self.limit}
    pub const fn summary(self)->Option<Summary> {self.summary}
    pub fn replace_charge(self,workspace:usize)->Result<usize,String> {
        workspace.checked_sub(FULL).and_then(|n|n.checked_add(self.limit)).ok_or_else(||"projection memo workspace overflow".into())
    }
}
/// Shared with clones. Clearing entries cannot clear this startup decision.
#[derive(Default)]
pub struct Bound(OnceCell<Plan>);
impl Bound {
    /// Older device clones have separate memos. None may keep a pre-start limit.
    pub fn freeze_device(bound:&Rc<Self>,plan:Plan)->Result<Plan,String> {
        if bound.plan().is_none() && Rc::strong_count(bound)!=1 {return Err("memo startup bound has an older device clone".into());}
        bound.freeze(plan)
    }
    pub fn freeze(&self,plan:Plan)->Result<Plan,String> {
        if let Some(prior)=self.0.get() {return if *prior==plan {Ok(*prior)} else {Err("projection memo startup bound changed".into())};}
        self.0.set(plan).map_err(|_|"projection memo bound was already set".to_string())?;Ok(plan)
    }
    pub fn plan(&self)->Option<Plan> {self.0.get().copied()}
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn loaded_geometry_sets_one_charge_and_limit() {
        let mut s=Summary::default();s.add(16384,512,2051,57).unwrap();
        assert_eq!(s,Summary {unabsorbed:1,selection:73_400_320,chunk:2_097_152});
        let p=Plan::new(Some(s),8);assert_eq!(p.bytes(),880_803_840);
        let bound=Bound::default();assert_eq!(bound.freeze(p).unwrap().bytes(),p.replace_charge(FULL).unwrap());
        for _ in 0..3 {assert_eq!(bound.plan(),Some(p));assert_eq!(bound.freeze(p),Ok(p));}
        assert!(bound.freeze(Plan::full()).is_err());
    }
    #[test] fn no_projection_unknown_mixed_and_overflow() {
        assert_eq!(Plan::new(Some(Summary::default()),8).bytes(),0);
        assert_eq!(Plan::new(None,8),Plan::full());
        let mut s=Summary::default();s.add(16384,512,2051,57).unwrap();s.add(16384,512,2051,57).unwrap();
        assert_eq!(Plan::new(Some(s),8).bytes(),1_761_607_680);
        assert_eq!(Plan::new(Some(s),usize::MAX),Plan::full());
        assert!(Summary::default().add(usize::MAX,512,2051,57).is_none());
        assert!(Summary::default().add(16384,usize::MAX,2051,57).is_none());
        assert!(Plan::new(None,8).replace_charge(0).is_err());
        assert!(Plan::full().replace_charge(usize::MAX).is_ok());
    }
    #[test] fn ranks_must_agree_on_loaded_projections() {
        let mut s=Summary::default();s.add(16384,512,2051,57).unwrap();let known=record(Some(s));
        assert_eq!(agree(&[known,known].concat(),8).unwrap(),Plan::new(Some(s),8));
        assert_eq!(agree(&[known,record(None)].concat(),8).unwrap(),Plan::full());
        assert!(agree(&[known,record(Some(Summary::default()))].concat(),8).is_err());
        s.selection+=s.chunk;assert_eq!(agree(&[known,record(Some(s))].concat(),8).unwrap(),Plan::full());
        let mut bad=known;bad[1]=f32::NAN;assert!(agree(&bad,8).is_err());
    }
    #[test] fn a_device_clone_before_start_cannot_keep_an_old_limit() {
        let bound=Rc::new(Bound::default());let older=bound.clone();let plan=Plan::new(Some(Summary::default()),8);
        assert!(Bound::freeze_device(&bound,plan).is_err());assert_eq!(bound.plan(),None);
        drop(older);assert_eq!(Bound::freeze_device(&bound,plan),Ok(plan));
        let later=bound.clone();assert_eq!(Bound::freeze_device(&later,plan),Ok(plan));
        assert!(Bound::freeze_device(&bound,Plan::full()).is_err());
    }
}
