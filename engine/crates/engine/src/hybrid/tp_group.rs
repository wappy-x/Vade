//! One immutable rank domain and transport for target and draft collectives.
use super::*;
use std::sync::{Mutex, atomic::{AtomicBool,Ordering}};
trait Wire: Send {
    fn bytes(&mut self, v:&mut [u8], o:&[usize])->Result<()>;
    fn floats(&mut self, v:&mut [f32], o:&[usize])->Result<()>;
    fn sum(&mut self, v:&mut [f32], s:&mut [f32])->Result<()>;
}
impl<E:Exchange> Wire for E {
    fn bytes(&mut self,v:&mut [u8],o:&[usize])->Result<()> { self.gather_bytes(v,o) }
    fn floats(&mut self,v:&mut [f32],o:&[usize])->Result<()> { self.gather_f32(v,o) }
    fn sum(&mut self,v:&mut [f32],s:&mut [f32])->Result<()> { self.reduce(v,s) }
}
struct Port(Box<dyn Wire>);
impl Exchange for Port {
    fn gather_bytes(&mut self,v:&mut [u8],o:&[usize])->Result<()> { self.0.bytes(v,o) }
    fn gather_f32(&mut self,v:&mut [f32],o:&[usize])->Result<()> { self.0.floats(v,o) }
    fn reduce(&mut self,v:&mut [f32],s:&mut [f32])->Result<()> { self.0.sum(v,s) }
}
pub struct LocalExchange;
impl Exchange for LocalExchange {
    fn gather_bytes(&mut self,_:&mut [u8],_:&[usize])->Result<()> { Ok(()) }
    fn gather_f32(&mut self,_:&mut [f32],_:&[usize])->Result<()> { Ok(()) }
    fn reduce(&mut self,_:&mut [f32],_:&mut [f32])->Result<()> { Ok(()) }
}
pub struct Group { plan:ModelPlan, split:SplitPlan, rank:usize, members:Vec<u64>, incarnation:u64,
    port:Mutex<Port>, reduce_scratch:Mutex<Vec<f32>>, failed:AtomicBool }
impl Group {
    pub fn local(plan:&ModelPlan,split:SplitPlan)->Result<Self> { Self::new(plan,split,0,&[1],1,LocalExchange) }
    pub fn new<E:Exchange+'static>(plan:&ModelPlan,split:SplitPlan,rank:usize,members:&[u64],incarnation:u64,e:E)->Result<Self> {
        Binding::new(plan,&split,rank)?;
        if incarnation==0 || members.len()!=split.ranks as usize || members.contains(&0)
            || members.iter().collect::<std::collections::BTreeSet<_>>().len()!=members.len() { return Err("invalid hybrid rank group".into()); }
        let g=Self {plan:plan.clone(),split,rank,members:members.to_vec(),incarnation,port:Mutex::new(Port(Box::new(e))),reduce_scratch:Mutex::new(Vec::new()),failed:AtomicBool::new(false)};
        if g.distributed() { g.agree(&Self::setup_words(plan,&g.split,&g.members,incarnation),&[])?; } Ok(g)
    }
    /// The startup envelope and setup vote include the same immutable MTP flags.
    pub fn setup_words(plan:&ModelPlan,split:&SplitPlan,members:&[u64],incarnation:u64)->Vec<u64> {
        let metadata=format!("{plan:?}{split:?}{members:?}{incarnation}");
        std::iter::once(metadata.len() as u64).chain(metadata.as_bytes().chunks(8).map(|s|{let mut b=[0;8];b[..s.len()].copy_from_slice(s);u64::from_le_bytes(b)})).chain(crate::hybrid_draft::Options::startup().words()).collect()
    }
    pub fn valid(&self)->bool { !self.failed.load(Ordering::Relaxed) }
    pub fn invalidate(&self) { self.failed.store(true,Ordering::Relaxed); }
    pub fn split(&self)->&SplitPlan { &self.split }
    pub fn rank(&self)->usize { self.rank }
    pub fn members(&self)->&[u64] { &self.members }
    pub fn incarnation(&self)->u64 { self.incarnation }
    pub fn distributed(&self)->bool { distributed(&self.split) }
    fn transport<T>(&self,f:impl FnOnce(&mut Port)->Result<T>)->Result<T> {
        if self.failed.load(Ordering::Relaxed) { return Err("hybrid rank group invalid".into()); }
        let result=f(&mut *self.port.lock().map_err(|_|{self.invalidate();"hybrid transport lock invalid"})?);
        if result.is_err() { self.failed.store(true,Ordering::Relaxed); } result
    }
    // The owned plan/split are immutable after the complete checked constructor.
    fn binding(&self)->Binding<'_> { Binding {plan:&self.plan,split:&self.split,rank:self.rank} }
    fn with<T>(&self,f:impl FnOnce(&Binding<'_>,&mut Port)->Result<T>)->Result<T> { self.transport(|e|f(&self.binding(),e)) }
    pub fn agree(&self,words:&[u64],votes:&[bool])->Result<Vec<bool>> {
        self.with(|b,e| b.agree(e,words,votes))
    }
    pub fn routes(&self,layer:usize,rows:usize,ids:&[u32],weights:&[u32],rejects:&[bool])->Result<Vec<bool>> {
        let votes=self.agree(&[layer as u64,rows as u64,ids.len() as u64,weights.len() as u64],rejects)?;
        let place=if layer<self.plan.shape.layers.len(){Place::Layer(layer as u32)}else{Place::Draft{layer:Some(0)}};
        let Ffn::Experts(shape)=self.binding().layer(place)?.ffn else {return Err("TP expert route role".into())};
        let k=size(shape.selected)?; if votes.len()!=rows || ids.len()!=product(rows,k)? || weights.len()!=ids.len(){self.invalidate();return Err("TP route extent".into())}
        let words=std::iter::once(layer as u64).chain((0..ids.len()).filter(|i|!votes[i/k]).flat_map(|i|[u64::from(ids[i]),u64::from(weights[i])])).collect::<Vec<_>>();
        self.agree(&words,&[])?; Ok(votes)
    }
    pub fn embedding(&self,ids:&[u32],mut row:impl FnMut(usize)->Result<Vec<f32>>)->Result<Vec<f32>> {
        if self.distributed() && ids.len()>1 { return self.embedding_cohort(ids,&mut row); }
        self.with(|b,e| { b.agree(e,&std::iter::once(ids.len() as u64).chain(ids.iter().map(|&id|u64::from(id))).collect::<Vec<_>>(),&[])?;
            let mut output=Vec::new(); for &id in ids { let owner=b.local_token(id)?; let local=owner.map(&mut row).transpose().map(|v|v.unwrap_or_default());
                if b.agree(e,&[u64::from(id)],&[!local.as_ref().is_ok_and(|v|v.len()==if owner.is_some(){self.plan.shape.hidden as usize}else{0})])?[0] { return Err("rank embedding rejected".into()); } let local=local?;
                let mut full=vec![0.0;size(self.plan.shape.hidden)?]; b.embedding(e,id,&local,&mut full)?; output.extend(full); } Ok(output) })
    }
    // Gather owned rows once in rank-major order, then restore token order by
    // copying bits. No arithmetic, including addition to zero, touches a row.
    fn embedding_cohort(&self,ids:&[u32],row:&mut impl FnMut(usize)->Result<Vec<f32>>)->Result<Vec<f32>> {
        self.with(|b,e| {
            let command=std::iter::once(ids.len() as u64).chain(ids.iter().map(|&id|u64::from(id))).collect::<Vec<_>>();
            b.agree(e,&command,&[])?;
            let h=size(self.plan.shape.hidden)?;
            let owners=ids.iter().map(|&id|b.local_token(id)).collect::<Result<Vec<_>>>()?;
            let local=owners.iter().map(|&id|id.map(&mut *row).transpose().map(|v|v.unwrap_or_default())).collect::<Vec<_>>();
            let rejected=local.iter().zip(&owners).map(|(v,owner)|!v.as_ref().is_ok_and(|v|v.len()==if owner.is_some(){h}else{0})).collect::<Vec<_>>();
            if b.agree(e,&[ids.len() as u64],&rejected)?.iter().any(|&v|v) { return Err("rank embedding rejected".into()); }
            let mut offsets=vec![0usize];
            for owner in &self.split.owners {
                let words=product(ids.iter().filter(|&&id|owner.vocab.contains(&u64::from(id))).count(),h)?;
                offsets.push(offsets.last().unwrap().checked_add(words).ok_or("TP embedding extent")?);
            }
            let n=product(ids.len(),h)?;
            if offsets.last()!=Some(&n) { return Err("TP embedding extent".into()); }
            let mut wire=vec![0.0;n];let mut at=offsets[self.rank];
            for values in local { let values=values?;wire[at..at+values.len()].copy_from_slice(&values);at+=values.len(); }
            if at!=offsets[self.rank+1] { return Err("TP embedding extent".into()); }
            e.gather_f32(&mut wire,&offsets)?;
            let mut output=vec![0.0;n];let mut cursors=offsets;
            for (i,&id) in ids.iter().enumerate() {
                let rank=self.split.owners.iter().position(|o|o.vocab.contains(&u64::from(id))).ok_or("TP global token ID")?;
                let start=cursors[rank];output[i*h..(i+1)*h].copy_from_slice(&wire[start..start+h]);cursors[rank]+=h;
            }
            Ok(output)
        })
    }
    pub fn norm(&self,t:&TensorEntry,local:&[u8])->Result<Vec<u8>> {
        self.with(|b,e| { let mut full=vec![0;size(t.shape.first().ok_or("TP norm shape")?.checked_mul(t.dtype.bytes()).ok_or("TP norm extent")?)?]; b.gather_norm(e,t,local,&mut full)?; Ok(full) })
    }
    pub fn reduce(&self,step:Step,rows:usize,v:&mut [f32])->Result<()> {
        self.with(|b,e| {
            let n=product(v.len(),self.split.ranks as usize)?;
            let mut scratch=self.reduce_scratch.lock().map_err(|_|"hybrid reduction scratch invalid")?;
            if scratch.len()<n { scratch.resize(n,0.0); }
            b.reduce(e,step,rows,v,&mut scratch[..n])
        })
    }
    pub fn heads(&self,rows:usize,local:&[f32])->Result<Vec<f32>> {
        self.with(|b,e| { let n=product(rows,size(self.plan.shape.vocab)?)?; let mut full=vec![0.0;n];
            b.heads(e,rows,local,&mut vec![0.0;n],&mut full)?; Ok(full) })
    }
}
// Command votes borrow the same transport only for each exchange, leaving it
// available to the target loop while HybridProtocol executes the command.
impl Exchange for &Group {
    fn gather_bytes(&mut self,v:&mut [u8],o:&[usize])->Result<()> { self.transport(|e|e.gather_bytes(v,o)) }
    fn gather_f32(&mut self,v:&mut [f32],o:&[usize])->Result<()> { self.transport(|e|e.gather_f32(v,o)) }
    fn reduce(&mut self,v:&mut [f32],s:&mut [f32])->Result<()> { self.transport(|e|e.reduce(v,s)) }
}
