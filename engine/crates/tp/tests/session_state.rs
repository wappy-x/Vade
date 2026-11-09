use spark_tp::session_state::*;
use engine_model::plan::{ConfigPlan, TextDescription, description::*, layer::{Topology,cast::Dtype}, state::{StatePlan, PlaneKind}};
use engine_model::shapes::Ffn;
use std::{cell::{Cell, RefCell}, rc::{Rc, Weak}};

type Result<T> = std::result::Result<T,String>;
type Bytes = Rc<RefCell<Vec<u8>>>;
enum Op { Zero(Weak<RefCell<Vec<u8>>>), Write(Weak<RefCell<Vec<u8>>>,Vec<u8>), Copy(Weak<RefCell<Vec<u8>>>,Weak<RefCell<Vec<u8>>>) }
#[derive(Default)]
struct Fake { queue: RefCell<Vec<Op>>, log: RefCell<Vec<String>>, live: Cell<usize>, fault: RefCell<Option<(&'static str,usize)>>,
    panic_on: RefCell<Option<&'static str>> }
impl Fake {
    fn event(&self,name: &'static str) -> Result<()> {
        self.log.borrow_mut().push(name.into());
        let panics=*self.panic_on.borrow()==Some(name);
        if panics {self.panic_on.borrow_mut().take(); panic!("injected {name} panic");}
        let mut fault=self.fault.borrow_mut();
        if let Some((target,n))=fault.as_mut() { if *target==name {
            *n-=1; if *n==0 { *fault=None; return Err(format!("injected {name} failure")); }
        } } Ok(())
    }
    fn fail(&self,name: &'static str,n: usize) { *self.fault.borrow_mut()=Some((name,n)); }
}
#[derive(Clone,Default)]
struct Memory { fake: Rc<Fake> }
struct Buffer { bytes: Bytes, fake: Rc<Fake> }
impl Drop for Buffer {
    fn drop(&mut self) {
        assert!(self.fake.queue.borrow().is_empty(),"allocation released before completion");
        self.fake.live.set(self.fake.live.get()-1); self.fake.log.borrow_mut().push("free".into());
    }
}
impl Store for Memory {
    type Buffer=Buffer;
    fn allocate(&self,n: usize) -> Result<Buffer> {
        self.fake.event("allocate")?; self.fake.live.set(self.fake.live.get()+1);
        // Nonzero recycled bytes force the owner to initialize every allocation.
        Ok(Buffer {bytes:Rc::new(RefCell::new(vec![0xa5;n])),fake:self.fake.clone()})
    }
    fn size(&self,b: &Buffer) -> usize { b.bytes.borrow().len() }
    fn zero(&self,b: &mut Buffer) -> Result<()> {
        self.fake.queue.borrow_mut().push(Op::Zero(Rc::downgrade(&b.bytes))); self.fake.event("zero")
    }
    fn copy(&self,a: &Buffer,b: &mut Buffer) -> Result<()> {
        assert!(!Rc::ptr_eq(&a.bytes,&b.bytes));
        self.fake.queue.borrow_mut().push(Op::Copy(Rc::downgrade(&a.bytes),Rc::downgrade(&b.bytes))); self.fake.event("copy")
    }
    fn write(&self,b: &mut Buffer,bytes: &[u8]) -> Result<()> {
        self.fake.queue.borrow_mut().push(Op::Write(Rc::downgrade(&b.bytes),bytes.to_vec())); self.fake.event("write")
    }
    fn read(&self,b: &Buffer) -> Result<Vec<u8>> {
        assert!(self.fake.queue.borrow().is_empty()); self.fake.event("read")?; Ok(b.bytes.borrow().clone())
    }
    fn wait(&self) -> Result<()> {
        self.fake.event("compute_wait")?;
        for op in self.fake.queue.borrow_mut().drain(..) { match op {
            Op::Zero(b)=>b.upgrade().expect("zero owner was freed").borrow_mut().fill(0),
            Op::Write(b,v)=>b.upgrade().expect("write owner was freed").borrow_mut()[..v.len()].copy_from_slice(&v),
            Op::Copy(a,b)=>b.upgrade().expect("copy destination was freed").borrow_mut().copy_from_slice(&a.upgrade().expect("copy source was freed").borrow()),
        } } Ok(())
    }
}
struct Transport { fake: Rc<Fake>, fail: bool }
impl Completion for Transport {
    fn wait(&mut self) -> Result<()> { self.fake.event("transport_wait")?; if self.fail {Err("transport completion failed".into())} else {Ok(())} }
}
fn layout() -> Layout {
    use PlaneKind::*;
    Layout::new(7,vec![
        Plane {layer:0,kind:KdaState,length:Length::Fixed(32)},
        Plane {layer:0,kind:KdaHistory,length:Length::Fixed(18)},
        Plane {layer:1,kind:Latent,length:Length::Rows {row_bytes:6,tokens_per_row:1}},
        Plane {layer:1,kind:IndexKeys,length:Length::Rows {row_bytes:4,tokens_per_row:1}},
        Plane {layer:1,kind:IndexGates,length:Length::Rows {row_bytes:4,tokens_per_row:1}},
        Plane {layer:1,kind:Pools,length:Length::Rows {row_bytes:4,tokens_per_row:2}},
    ],82).unwrap()
}
fn budget() -> Budget { Budget {weights:4096,runtime:512,reserve:1024,lanes:3,saved:2,limit:1<<20} }
fn pool(memory: &Memory) -> Pool<Memory> { Pool::new(memory.clone(),layout(),budget()).unwrap() }
fn contents(s: &Session<Memory>) -> Vec<Vec<u8>> { (0..s.layout().planes().len()).map(|i| s.read(i).unwrap()).collect() }
fn advance(s: &mut Session<Memory>,token: u8) {
    let before=contents(s); let planes=s.layout().planes().to_vec(); let mut a=s.begin_append().unwrap();
    for (i,p) in planes.iter().enumerate() {
        let n=p.used_bytes(a.context()).unwrap();
        let bytes=match p.length {
            Length::Fixed(_)=>before[i].iter().map(|v| v.wrapping_mul(3).wrapping_add(token)).collect(),
            Length::Rows {..}=>{let mut b=before[i].clone(); b.resize(n,token); b},
        };
        a.write(i,&bytes).unwrap();
    }
    a.commit().unwrap();
}

#[test]
fn checked_rank_budget_and_physical_padding() {
    let p=layout(); let mut b=budget();
    assert_eq!(p.bank_bytes(),384); assert_eq!(p.scratch_bytes(),128); assert_eq!(p.lane_bytes().unwrap(),896);
    let total=b.required(&p).unwrap(); assert_eq!(total,9088);
    println!("SESSION_BUDGET weights={} runtime={} reserve={} lanes={} saved={} bank={} scratch={} required={total}",
        b.weights,b.runtime,b.reserve,b.lanes,b.saved,p.bank_bytes(),p.scratch_bytes());
    b.limit=total; assert_eq!(b.required(&p).unwrap(),total); b.limit-=1; assert!(b.required(&p).is_err());
    for bad in [Budget {lanes:0,..budget()},Budget {weights:usize::MAX,..budget()},Budget {lanes:usize::MAX,..budget()}] {
        assert!(bad.required(&p).is_err());
    }
    let one=Plane {layer:0,kind:PlaneKind::KdaHistory,length:Length::Fixed(0)};
    assert_eq!(Layout::new(1,vec![one.clone()],0).unwrap().bank_bytes(),64);
    assert!(Layout::new(1,vec![one.clone(),one],0).is_err());
    for length in [Length::Fixed(usize::MAX),Length::Rows {row_bytes:0,tokens_per_row:1},
        Length::Rows {row_bytes:4,tokens_per_row:0},Length::Rows {row_bytes:isize::MAX as usize,tokens_per_row:1}] {
        assert!(Layout::new(7,vec![Plane {layer:0,kind:PlaneKind::Latent,length}],1).is_err());
    }
    assert!(Layout::new(0,p.planes().to_vec(),1).is_err());
    assert!(Layout::new(u64::MAX,p.planes().to_vec(),1).is_err());
    assert!(Layout::new(7,p.planes().to_vec(),usize::MAX).is_err());
}

#[test]
fn logical_state_adapter_preserves_shapes_and_checkpoint_refusal() {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    d.hidden=8; d.max_positions=7; d.residual=SinkhornDescription {streams:2,iterations:4,epsilon:1e-5};
    d.layers=vec![DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:2,head_dim:4,conv:4,gate_lower_bound:-5.0}),ffn:Ffn::Dense {width:6}},
        DecoderDescription {mixer:TokenMixer::LatentAttention(LatentDescription {heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false}}),ffn:Ffn::Dense {width:6}}];
    d.storage.layers.clear(); d.storage.excluded.clear();
    check_logical(&d);
    let ConfigPlan::Deferred(full)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    check_logical(&full);
}
fn check_logical(d: &TextDescription) {
    for parts in [1,2,4] {
        // The small geometry has fewer head units than the widest fixture case.
        let Ok(p)=StatePlan::new(d,Topology {parts,rank:parts-1},7,isize::MAX as usize) else {continue};
        let l=Layout::logical_reference(&p).unwrap();
        for (a,b) in l.planes().iter().zip(p.planes()) { for n in 0..=7 { assert_eq!(a.used_bytes(n).unwrap(),b.used_bytes(n).unwrap()); } }
        assert!(l.scratch_bytes()>=p.scratch_bytes());
        assert!(Layout::checkpoint(&p).unwrap_err().contains("conditional"));
        let types=p.planes().iter().map(|p|CacheType {layer:p.layer,kind:p.kind,
            dtype:if p.kind==PlaneKind::KdaState {Dtype::F32} else {Dtype::Bf16}}).collect::<Vec<_>>();
        let physical=Layout::physical_storage(&p,&types,256).unwrap();
        for (a,b) in physical.planes().iter().zip(p.planes()) {
            match a.kind {
                PlaneKind::KdaState=>assert_eq!(a.used_bytes(7).unwrap(),b.used_bytes(7).unwrap()),
                PlaneKind::KdaHistory=>{
                    let TokenMixer::Kda(k)=d.layers[a.layer].mixer else {panic!()};
                    assert_eq!(a.used_bytes(7).unwrap() as u64,3*k.heads/u64::from(parts)*k.head_dim*k.conv*2);
                },
                _=>for n in 0..=7 {assert_eq!(a.used_bytes(n).unwrap()*2,b.used_bytes(n).unwrap());},
            }
        }
        assert!(physical.scratch_bytes()>=p.scratch_bytes()+256);
        assert!(Layout::physical_storage(&p,&types[..types.len()-1],0).is_err());
        let mut bad=types.clone(); bad[0].dtype=Dtype::Bf16; assert!(Layout::physical_storage(&p,&bad,0).is_err());
        bad[0].dtype=Dtype::Unresolved; assert!(Layout::physical_storage(&p,&bad,0).is_err());
        bad[0].dtype=Dtype::U32; assert!(Layout::physical_storage(&p,&bad,0).is_err());
        bad[0]=bad[1]; assert!(Layout::physical_storage(&p,&bad,0).is_err());
        assert!(Layout::physical_storage(&p,&types,usize::MAX).is_err());
    }
}

#[test]
fn slots_and_partial_allocation_cleanup() {
    let m=Memory::default(); let p=pool(&m);
    m.fake.fail("allocate",3); assert!(p.session().is_err());
    assert_eq!(p.used_slots(),(0,0)); assert_eq!(m.fake.live.get(),0);
    m.fake.fail("zero",2); assert!(p.session().is_err());
    assert_eq!(p.used_slots(),(0,0)); assert_eq!(m.fake.live.get(),0);
    let s=p.session().unwrap(); let t=p.session().unwrap(); let u=p.session().unwrap(); assert!(p.session().is_err());
    let saved=s.save().unwrap(); let saved2=s.save().unwrap(); assert!(s.save().is_err());
    assert_eq!(p.used_slots(),(3,2)); drop(saved2); assert_eq!(p.used_slots(),(3,1));
    m.fake.fail("copy",2); assert!(s.save().is_err()); assert_eq!(p.used_slots(),(3,1));
    drop((s,t,u)); assert_eq!(p.used_slots(),(0,1)); drop(saved); assert_eq!(p.used_slots(),(0,0));
    assert_eq!(m.fake.live.get(),0);
}

#[test]
fn append_rollback_restore_and_pool_identity() {
    let m=Memory::default(); let p=pool(&m); let other=pool(&m);
    let mut s=p.session().unwrap(); assert!(contents(&s).iter().all(|b| b.iter().all(|&v| v==0)));
    let zero=s.save().unwrap(); advance(&mut s,7); let saved=s.save().unwrap(); let at_one=contents(&s);
    { let mut a=s.begin_append().unwrap(); a.write(0,&[9;32]).unwrap(); assert!(a.commit().is_err()); }
    assert_eq!(s.context(),1); assert_eq!(contents(&s),at_one);
    { let mut a=s.begin_append().unwrap(); a.write(0,&[9;32]).unwrap(); a.rollback(); }
    { let mut a=s.begin_append().unwrap(); assert!(a.write(0,&[9;31]).is_err()); assert!(a.write(99,&[]).is_err()); }
    assert_eq!(contents(&s),at_one); advance(&mut s,9); s.truncate(&saved).unwrap(); assert_eq!(contents(&s),at_one);
    s.restore(&zero).unwrap(); assert!(s.truncate(&saved).is_err()); s.restore(&saved).unwrap();
    let foreign=other.session().unwrap().save().unwrap(); assert!(s.restore(&foreign).is_err()); assert_eq!(contents(&s),at_one);
    drop(s); let mut replacement=p.session().unwrap(); replacement.restore(&saved).unwrap(); assert_eq!(contents(&replacement),at_one);
    for token in 2..=7 { advance(&mut replacement,token); } assert!(replacement.begin_append().is_err());
}

#[test]
fn continuation_matches_interleaving_batch_chunks_cache_and_rejected_append() {
    let m=Memory::default(); let p=pool(&m);
    let tokens=[1,3,2,7,5,9,11]; let mut alone=p.session().unwrap(); let mut split=p.session().unwrap(); let mut interleaved=p.session().unwrap();
    for &token in &tokens { advance(&mut alone,token); }
    for chunk in tokens.chunks(2) { for &token in chunk { advance(&mut split,token); advance(&mut interleaved,token); } }
    assert_eq!(contents(&alone),contents(&split)); assert_eq!(contents(&alone),contents(&interleaved)); drop(split); drop(interleaved);
    let mut restored=p.session().unwrap();
    for &token in &tokens[..3] {advance(&mut restored,token);}
    let boundary=restored.save().unwrap();
    advance(&mut restored,99); advance(&mut restored,98); restored.truncate(&boundary).unwrap();
    for &token in &tokens[3..] {advance(&mut restored,token);}
    assert_eq!(contents(&restored),contents(&alone));
    restored.restore(&boundary).unwrap();
    { let mut a=restored.begin_append().unwrap(); a.write(0,&[0xff;32]).unwrap(); }
    for &token in &tokens[3..] {advance(&mut restored,token);}
    assert_eq!(contents(&restored),contents(&alone));
    println!("SESSION_CONTINUATION opaque_bytes tokens={} chunks=[2,2,2,1] cache_boundary={} state=identical",tokens.len(),boundary.context());
}

#[test]
fn failed_writes_and_restore_leave_committed_bank_intact() {
    let m=Memory::default(); let p=pool(&m); let mut s=p.session().unwrap(); advance(&mut s,4); let saved=s.save().unwrap();
    advance(&mut s,5); let committed=contents(&s);
    m.fake.fail("copy",2); assert!(s.restore(&saved).is_err()); assert_eq!(s.context(),2); assert_eq!(contents(&s),committed);
    m.fake.fail("copy",2); assert!(s.begin_append().is_err()); assert_eq!(contents(&s),committed);
    { let mut a=s.begin_append().unwrap(); m.fake.fail("write",1); assert!(a.write(0,&[1;32]).is_err()); assert!(a.commit().is_err()); }
    assert!(p.failure().is_none()); assert_eq!(contents(&s),committed); s.restore(&saved).unwrap(); assert_eq!(s.context(),1);
}

#[test]
fn a_replacement_upload_panic_cannot_commit_an_earlier_complete_append() {
    let m=Memory::default(); let p=pool(&m); let mut s=p.session().unwrap(); advance(&mut s,4); let before=contents(&s);
    let mut a=s.begin_append().unwrap();
    let sizes=a.layout().planes().iter().map(|p|p.used_bytes(a.context()).unwrap()).collect::<Vec<_>>();
    for (i,&n) in sizes.iter().enumerate() {a.write(i,&vec![8;n]).unwrap();}
    *m.fake.panic_on.borrow_mut()=Some("write");
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.write(0,&[9;32])));
    assert!(result.is_err()); assert!(m.fake.queue.borrow().is_empty()); assert!(a.commit().is_err());
    assert_eq!(s.context(),1); assert_eq!(contents(&s),before); assert!(p.failure().is_none());
    let saved=s.save().unwrap(); advance(&mut s,7); let later=contents(&s);
    *m.fake.panic_on.borrow_mut()=Some("copy");
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.restore(&saved))).is_err());
    assert!(m.fake.queue.borrow().is_empty()); assert_eq!(s.context(),2); assert_eq!(contents(&s),later);
    let mut a=s.begin_append().unwrap();
    let sizes=a.layout().planes().iter().map(|p|p.used_bytes(a.context()).unwrap()).collect::<Vec<_>>();
    for (i,&n) in sizes.iter().enumerate() {a.write(i,&vec![8;n]).unwrap();}
    assert!(a.write(0,&[9;31]).is_err()); assert!(a.commit().is_err());
    assert_eq!(s.context(),2); assert_eq!(contents(&s),later);
    let mut a=s.begin_append().unwrap();
    for (i,&n) in sizes.iter().enumerate() {a.write(i,&vec![8;n]).unwrap();}
    assert!(a.write(sizes.len(),&[]).is_err()); assert!(a.commit().is_err());
    assert_eq!(s.context(),2); assert_eq!(contents(&s),later);
    advance(&mut s,9); assert_eq!(s.context(),3); assert!(p.failure().is_none());
}

#[test]
fn launch_errors_and_panics_drain_compute_and_transport() {
    let m=Memory::default(); let p=pool(&m); let mut s=p.session().unwrap(); advance(&mut s,4); let before=contents(&s);
    let mut transport=Transport {fake:m.fake.clone(),fail:false};
    {
        let mut a=s.begin_append().unwrap();
        let error=a.run(&mut transport,|store,work| {store.write(&mut work.staged[0],&[7;32])?; Err("launch failed after enqueue".into())});
        assert!(error.is_err()); assert!(a.commit().is_err());
    }
    assert_eq!(contents(&s),before);
    let panic=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut a=s.begin_append().unwrap();
        let _=a.run(&mut transport,|store,work| {store.write(&mut work.staged[0],&[9;32])?; panic!("producer panic");});
    }));
    assert!(panic.is_err()); assert!(m.fake.queue.borrow().is_empty()); assert_eq!(contents(&s),before);
    let log=m.fake.log.borrow(); assert!(log.windows(2).any(|p| p==["transport_wait","compute_wait"]));
}

#[test]
fn completion_failure_retires_pool_and_quarantines_owners() {
    for compute_failure in [false,true] {
        let m=Memory::default(); let p=pool(&m); let mut s=p.session().unwrap();
        let live=m.fake.live.get(); let mut transport=Transport {fake:m.fake.clone(),fail:!compute_failure};
        {
            let mut a=s.begin_append().unwrap();
            if compute_failure {m.fake.fail("compute_wait",1);}
            assert!(a.run(&mut transport,|store,work| {store.write(&mut work.staged[0],&[9;32])?; Ok(vec![0])}).is_err());
        }
        assert!(p.failure().is_some()); assert_eq!(s.context(),0);
        assert!(s.save().is_err()); assert!(s.begin_append().is_err()); assert!(s.read(0).is_err()); assert!(p.session().is_err());
        drop(s); assert_eq!(m.fake.live.get(),live); assert_eq!(p.used_slots(),(1,0));
        assert!(!m.fake.log.borrow().iter().any(|s| s=="free"));
        // Simulate context retirement. The test allocation owners remain quarantined.
        m.fake.queue.borrow_mut().clear();
    }
}

#[test]
fn panicking_completion_also_retires_pool() {
    struct Panics;
    impl Completion for Panics {fn wait(&mut self)->Result<()> {panic!("transport wait panic")}}
    let m=Memory::default(); let p=pool(&m); let mut s=p.session().unwrap();
    let mut a=s.begin_append().unwrap();
    assert!(a.run(&mut Panics,|store,work| {store.write(&mut work.staged[0],&[1;32])?; Ok(vec![0])}).is_err());
    a.rollback(); assert!(m.fake.queue.borrow().is_empty()); assert!(p.failure().unwrap().contains("panicked"));
    let live=m.fake.live.get(); drop(s); assert_eq!(m.fake.live.get(),live);
}

#[path="../src/bin/session-state-bytes/fixture.rs"]
mod fixture;
#[test]
fn device_fixture_has_an_independent_host_trace() {
    let m=Memory::default(); let p=Pool::new(m,fixture::layout().unwrap(),budget()).unwrap(); let mut s=p.session().unwrap();
    let mut out=Vec::new(); let mut prefix=None;
    for (i,&token) in [1,3,2,7,5,9,11].iter().enumerate() {
        let bytes=fixture::next(s.layout(),&contents(&s),(i+1) as u64,token).unwrap();
        let mut a=s.begin_append().unwrap();
        for (plane,b) in bytes.iter().enumerate() {a.write(plane,b).unwrap();} a.commit().unwrap();
        fixture::record(s.context(),&contents(&s),&mut out);
        if i==2 {prefix=Some(s.save().unwrap());}
    }
    fixture::record(s.context(),&contents(&s),&mut out);
    s.restore(&prefix.unwrap()).unwrap(); fixture::record(s.context(),&contents(&s),&mut out);
    let expected=fixture::oracle().unwrap(); assert_eq!(out,expected); assert!(!expected.is_empty());
    println!("SESSION_DEVICE_ORACLE bytes={}",expected.len());
}

#[test]
fn resident_producer_keeps_buffers_disjoint_and_clears_stream_scratch() {
    let m=Memory::default(); let p=pool(&m); let mut s=p.session().unwrap();
    for token in 1..=2 {
        let mut a=s.begin_append().unwrap(); let lengths=a.layout().planes().iter().map(|p| p.used_bytes(a.context()).unwrap()).collect::<Vec<_>>();
        a.run(&mut ComputeOnly,|store,work| {
            assert!(store.read(work.scratch)?.iter().all(|&v| v==0));
            store.write(work.scratch,&[0xff;82])?;
            for (i,(a,b)) in work.committed.iter().zip(work.staged).enumerate() {
                assert!(!Rc::ptr_eq(&a.bytes,&b.bytes));
                store.zero(b)?; store.write(b,&vec![token;lengths[i]])?;
            }
            Ok((0..lengths.len()).collect())
        }).unwrap(); a.commit().unwrap();
        assert!(contents(&s).iter().flatten().all(|&v| v==token));
    }
}

#[path="../src/bin/session-plan-check/fixture.rs"]
mod plan_fixture;
const CACHE_TYPES: &str="kda_state f32\nkda_history bf16\nlatent bf16\nindex_keys bf16\nindex_gates bf16\npools bf16\nabi_scratch_bytes 258\n";
#[test]
fn physical_plan_continuation_matches_independent_bytes_at_pool_boundaries() {
    let saved=include_bytes!("../../model/tests/fixtures/glm/config.json");
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(saved).unwrap() else {panic!()};
    d.hidden=8;d.vocab=16;d.max_positions=7;d.residual=SinkhornDescription {streams:2,iterations:4,epsilon:1e-5};
    d.layers=vec![DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:2,head_dim:4,conv:4,gate_lower_bound:-5.0}),ffn:Ffn::Dense {width:6}},
        DecoderDescription {mixer:TokenMixer::LatentAttention(LatentDescription {heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false}}),ffn:Ffn::Dense {width:6}}];
    d.storage.layers.clear();d.storage.excluded.clear();
    for parts in [1,2] {for rank in 0..parts {
        let p=StatePlan::new(&d,Topology {parts,rank},7,isize::MAX as usize).unwrap();
        let types=p.planes().iter().map(|p|CacheType {layer:p.layer,kind:p.kind,dtype:if p.kind==PlaneKind::KdaState {Dtype::F32} else {Dtype::Bf16}}).collect::<Vec<_>>();
        let l=Layout::physical_storage(&p,&types,258).unwrap();
        let m=Memory::default();let b=Budget {weights:0,runtime:0,reserve:0,lanes:3,saved:1,limit:isize::MAX as usize};
        let pool=Pool::new(m.clone(),l.clone(),b).unwrap();
        for boundary in [1,2,3] {
            assert_eq!(plan_fixture::exercise(&pool,boundary).unwrap(),plan_fixture::oracle(&l,boundary).unwrap());
            assert_eq!(m.fake.live.get(),0);
        }
    }}
}
#[test]
fn saved_checkpoint_dimensions_require_explicit_cache_types_before_allocation() {
    let config=include_bytes!("../../model/tests/fixtures/glm/config.json");
    for parts in [1,2,4] {for rank in 0..parts {
        let t=Topology {parts,rank};let l=plan_fixture::layout(config,CACHE_TYPES,9,t).unwrap();
        let id=plan_fixture::identity(config,CACHE_TYPES,&l,t,3);
        assert!(id.contains("producer_types_qualified=false") && id.contains(&format!("parts={parts} rank={rank}")));
        assert!(l.planes().iter().any(|p|p.kind==PlaneKind::Pools && p.used_bytes(3).unwrap()==0 && p.used_bytes(4).unwrap()>0));
        for bad in [CACHE_TYPES.replace("kda_history bf16\n",""),CACHE_TYPES.replace("kda_state f32","kda_state bf16"),
            CACHE_TYPES.replace("latent bf16","latent unresolved"),format!("{CACHE_TYPES}pools bf16\n"),
            CACHE_TYPES.replace("abi_scratch_bytes 258\n",""),format!("{CACHE_TYPES}abi_scratch_bytes 258\n")] {
            assert!(plan_fixture::layout(config,&bad,9,t).is_err());
        }
    }}
}
