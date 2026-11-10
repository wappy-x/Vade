#[cfg(not(target_os = "linux"))] use spark_engine::hybrid_draft;
#[cfg(not(target_os = "linux"))] #[allow(dead_code)] #[path = "../src/hybrid/schedule.rs"] mod schedule;
#[cfg(not(target_os = "linux"))] #[allow(dead_code)] #[path = "../src/hybrid/tp.rs"] mod tp;
#[cfg(target_os = "linux")] use spark_engine::hybrid::{tp,schedule};
use engine_model::{plan::ModelPlan, shapes::*, split::{SplitPlan, SplitRequest}};
struct Anchor { peer: Vec<f32> }
impl engine_fabric::exchange::Exchange for Anchor {
    fn gather_bytes(&mut self, _: &mut [u8], _: &[usize]) -> Result<(), String> { unreachable!() }
    fn gather_f32(&mut self, _: &mut [f32], _: &[usize]) -> Result<(), String> { unreachable!() }
    fn reduce(&mut self, v: &mut [f32], _: &mut [f32]) -> Result<(), String> { for (v,p) in v.iter_mut().zip(&self.peer) { *v += *p; } Ok(()) }
}
struct FlagMismatch(usize); impl engine_fabric::exchange::Exchange for FlagMismatch {
    fn gather_bytes(&mut self,v:&mut [u8],o:&[usize])->Result<(),String> { let local=v[..o[1]].to_vec(); for chunk in v.chunks_exact_mut(o[1]) { chunk.copy_from_slice(&local); } if o[1]>16 { v[2*o[1]-16+8*self.0]^=1; } Ok(()) }
    fn gather_f32(&mut self,_:&mut [f32],_:&[usize])->Result<(),String> { unreachable!() } fn reduce(&mut self,_:&mut [f32],_:&mut [f32])->Result<(),String> { unreachable!() } }
#[allow(clippy::excessive_precision)] // literal constants stay as written.
#[test] fn owners_schedule_and_rounding() {
    for bytes in [include_bytes!("../../model/tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice(), include_bytes!("../../model/tests/fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice()] { let mut p=ModelPlan::from_config(bytes).unwrap(); for l in p.shape.layers.iter_mut().chain(&mut p.shape.draft) { if let Ffn::Experts(e)=&mut l.ffn { e.storage=ExpertStorage::PerExpert; } } p.shape.tensors.push(TensorEntry{name:"draft-fc".into(),dtype:engine_model::safetensors::Dtype::Bf16,shape:vec![1],role:Role{place:Place::Draft{layer:None},part:Part::Whole,store:Store::Weight}}); let q=SplitRequest{ranks:2,memory:vec![u64::MAX;2],reserve:vec![0;2],context:0,sequences:0,cache_bytes:2,state_bytes:4,draft:true,expert_rows:false}; let s=SplitPlan::new(&p.shape,&q).unwrap();
        for flag in 0..2 { assert!(tp::Group::new(&p,s.clone(),0,&[71,72],93,FlagMismatch(flag)).is_err()); }
        // Bind K128-aligned shared owners to isolate the envelope from the BF16-group splitter.
        for ranks in 2..=4 { let mut request=q.clone(); request.ranks=ranks; request.memory=vec![u64::MAX;ranks as usize]; request.reserve=vec![0;ranks as usize]; let mut split=SplitPlan::new(&p.shape,&request).unwrap(); let width=split.owners.last().unwrap().shared.end; for (rank,owner) in split.owners.iter_mut().enumerate() { owner.shared=(width/128*rank as u64/ranks as u64)*128..(width/128*(rank+1) as u64/ranks as u64)*128; } let envelope=tp::Binding::new(&p,&split,0).unwrap().transport_plan(16,64).unwrap();
            let tile=schedule::PROMPT_ROWS; let group=schedule::PROMPT_GROUP/tile;
            assert!(group*tile<=schedule::PROMPT_GROUP);
            for layer in p.shape.layers.iter().chain(&p.shape.draft) {
                let slots=match layer.ffn { Ffn::Experts(e)=>e.selected+1,_=>1 };
                let reduction=|rows| rows*slots as usize*p.shape.hidden as usize*4*ranks as usize;
                assert!(envelope.wire_bytes()>=reduction(group*tile));
                if p.shape.hidden==2048 && slots==9 && ranks==2 { assert!(envelope.wire_bytes()<reduction(1792)); assert!(envelope.wire_bytes()<reduction(schedule::ROUTED_GROUP)); }
            } }
        for rank in 0..2 { let b=tp::Binding::new(&p,&s,rank).unwrap(); assert_eq!(b.schedule(false).unwrap().len(),2*p.shape.layers.len()+2); assert_eq!(b.schedule(true).unwrap(),vec![tp::Step::Embedding,tp::Step::Mixer(Place::Draft{layer:Some(0)}),tp::Step::FeedForward(Place::Draft{layer:Some(0)}),tp::Step::Head]); let e=b.segments(Place::Layer(0),Part::Expert{id:Some(17),proj:Proj::Down}).unwrap(); assert_eq!(e,s.segments(engine_model::split::Reduction::Expert(17)).unwrap()); assert!(b.segments(Place::Layer(0),Part::Output).is_err());
            let envelope=b.transport_plan(16,64).unwrap(); assert!(envelope.wire_bytes()>=(257)*p.shape.vocab as usize*4); assert!(envelope.wire_bytes()>=16*p.shape.vocab as usize*4); assert_eq!(envelope.host_bytes(),6*envelope.wire_bytes()); assert!(b.transport_plan(usize::MAX,1).is_err()); assert!(b.transport_plan(0,1).is_err()); let trace=std::cell::RefCell::new(Vec::new());
            let prepared=envelope.prepare(&mut (),|_,n|{trace.borrow_mut().push("register");Ok(n*3)},|_|{trace.borrow_mut().push("seal");Ok(())}).unwrap(); assert_eq!(prepared.mailbox_bytes(),3*envelope.wire_bytes()); assert_eq!(prepared.host_bytes(),envelope.host_bytes());
            prepared.before_weights(Anchor{peer:vec![1.0]},|_,_,_|{trace.borrow_mut().push("admit");Ok(())},|mut e,()|{trace.borrow_mut().push("weights");use engine_fabric::exchange::Exchange; let mut v=[1.0]; e.reduce(&mut v,&mut [0.0;2])?; assert_eq!(v,[2.0]); assert!(e.gather_bytes(&mut vec![0;envelope.wire_bytes()+1],&[0,0,0]).is_err()); Ok(())}).unwrap(); assert_eq!(*trace.borrow(),["register","seal","admit","weights"]);
            let prepared=envelope.prepare(&mut (),|_,_|Ok(0),|_|Ok(())).unwrap(); assert!(prepared.before_weights(Anchor{peer:vec![]},|_,_,_|Err::<(),_>("low rank".into()),|_,()|->Result<(),String>{panic!("weights after refusal")}).is_err()); assert!(envelope.prepare(&mut (),|_,_|Err("registration".into()),|_|panic!("seal after failed registration")).is_err());
            let h=p.shape.hidden as usize; let mut v=vec![1.00390625;h]; let mut scratch=vec![0.0;2*h]; let mut a=Anchor{peer:vec![0.00390625;h]}; b.reduce(&mut a,tp::Step::Mixer(Place::Layer(0)),1,&mut v,&mut scratch).unwrap(); assert!(v.iter().all(|v|v.to_bits()==1.0078125f32.to_bits())); assert!(b.reduce(&mut a,tp::Step::Head,1,&mut v,&mut scratch).is_err()); assert!(b.reduce(&mut a,tp::Step::Mixer(Place::Layer(0)),0,&mut v,&mut scratch).is_err());
            let mut v=vec![1.00390625;9*h]; a.peer=vec![0.001;v.len()]; scratch.resize(2*v.len(),0.0); b.reduce(&mut a,tp::Step::FeedForward(Place::Layer(0)),1,&mut v,&mut scratch).unwrap(); assert!(v.iter().all(|v|v.to_bits()==(1.00390625f32+0.001).to_bits()));
        } let mut single=q.clone(); single.ranks=1; single.memory.truncate(1); single.reserve.truncate(1); let one=SplitPlan::new(&p.shape,&single).unwrap(); let expected=one.clone(); let mut bad=p.clone(); if let Ffn::Experts(e)=&mut bad.shape.layers[0].ffn { e.width+=128; } assert!(tp::Group::new(&bad,one.clone(),0,&[71],93,tp::LocalExchange).is_err()); let g=tp::Group::new(&p,one,0,&[71],93,tp::LocalExchange).unwrap(); assert_eq!(g.split(),&expected); assert_eq!(g.members(),[71]); assert_eq!(g.incarnation(),93); assert!(!tp::Group::setup_words(&p,g.split(),g.members(),g.incarnation()).is_empty()); let embedded=g.embedding(&[0],|_|Ok(vec![-0.0;p.shape.hidden as usize])).unwrap(); assert!(embedded.iter().all(|v|v.to_bits()==(-0.0f32).to_bits())); for rows in [1,4,2] { let mut v=vec![1.00390625;rows*p.shape.hidden as usize]; g.reduce(tp::Step::Mixer(Place::Layer(0)),rows,&mut v).unwrap(); assert!(v.iter().all(|v|v.to_bits()==spark_gpu_probe::bf16::rounded(1.00390625).to_bits())); }
        assert_eq!(g.rank(),0); assert_eq!(g.members(),[71]); assert!(tp::Group::local(&p,s.clone()).is_err()); g.invalidate(); assert!(g.agree(&[1],&[]).is_err()); let mut off=q.clone(); off.draft=false; let no=SplitPlan::new(&p.shape,&off).unwrap(); assert!(tp::Binding::new(&p,&no,0).unwrap().schedule(true).is_err()); assert!(tp::Binding::new(&p,&s,2).is_err());
    }
}
