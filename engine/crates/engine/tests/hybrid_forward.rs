#[path="../src/hybrid/schedule.rs"] mod schedule;
#[path="../src/hybrid/coverage.rs"] mod coverage;
#[path="../src/hybrid/cache.rs"] mod cache;
#[allow(dead_code)] #[path="../src/hybrid/mtp_contract.rs"] mod mtp_contract;
#[allow(dead_code)] #[path="../../gpu/src/projection_params.rs"] mod projections;
#[path="../../gpu/src/routing.rs"] mod routing;
#[test]
fn row_scheduling_and_role_refusals() {
    if let Ok(row)=std::env::var("VADE_TYPED_POISON_CHILD") { let row:usize=row.parse().unwrap(); let mut status=[0.0;2]; routing::ROUTING_DEFERRED.store(true,std::sync::atomic::Ordering::Relaxed); routing::clear(2,None,|r| { if let Some(r)=r { status[r]=f32::from_bits(0x01010101); } else { status.fill(0.0); } Ok(()) }).unwrap(); assert!(routing::rejected_rows(&status).is_empty()); routing::ROUTING_DEFERRED.store(false,std::sync::atomic::Ordering::Relaxed); assert_eq!(routing::forced_routing_rejection(1,None),None); for expected in [vec![row],vec![]] { routing::clear(2,None,|r| { if let Some(r)=r { status[r]=f32::from_bits(0x01010101); } else { status.fill(0.0); } Ok(()) }).unwrap(); assert_eq!(routing::rejected_rows(&status),expected); } return; }
    for row in 0..2 { assert!(std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact","row_scheduling_and_role_refusals"]).env("VADE_TYPED_POISON_CHILD",row.to_string()).env("VADE_TEST_ROUTING_REJECT",format!("1:{row}:2")).status().unwrap().success()); }
    use engine_model::{shapes::{TensorEntry,Role,Place,Part,Store},safetensors::Dtype};
    assert_eq!(schedule::ROUTED_GROUP, 4096);
    assert!(schedule::fused_router_boundary(true, 16, 16));
    assert!(!schedule::fused_router_boundary(false, 16, 16));
    for count in [1,2,16,17,32,33,255,256,257,1023,1024,1025] { let ranges:Vec<_>=schedule::ranges(count).collect(); assert_eq!(ranges.iter().flat_map(|r|r.clone()).collect::<Vec<_>>(),(0..count).collect::<Vec<_>>()); assert!(ranges.iter().all(|r|r.len()<=schedule::PROMPT_GROUP)); assert_eq!(ranges.len(),count.div_ceil(schedule::PROMPT_GROUP)); }
    assert_eq!(schedule::tokens(&[3,1,2]),[(0,0),(1,0),(2,0),(0,1),(2,1),(0,2)]); assert!(schedule::tokens(&[]).is_empty());
    let mut t=TensorEntry{name:"synthetic".into(),dtype:Dtype::Bf16,shape:vec![128,128],role:Role{place:Place::Layer(0),part:Part::Query,store:Store::Weight}};
    assert_eq!(coverage::use_of(&t).unwrap(),"fixed projection"); t.role.part=Part::Index; assert!(coverage::use_of(&t).is_err()); t.role.place=Place::Draft{layer:Some(0)}; assert!(coverage::use_of(&t).is_err()); t.role.part=Part::Query; assert_eq!(coverage::use_of(&t).unwrap(),"fixed projection"); t.role.place=Place::Unused; assert_eq!(coverage::use_of(&t).unwrap(),"vision excluded");
    t.role.place=Place::Layer(0); t.role.part=Part::Expert{id:None,proj:engine_model::shapes::Proj::Gate}; assert!(coverage::use_of(&t).is_err()); t.role.part=Part::Expert{id:Some(7),proj:engine_model::shapes::Proj::Gate}; assert_eq!(coverage::use_of(&t).unwrap(),"routed descriptor"); t.role.store=Store::BlockScale; assert_eq!(coverage::use_of(&t).unwrap(),"paired BF16 multiplier");
    let mut p=engine_model::plan::ModelPlan::from_config(include_bytes!("../../model/tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json")).unwrap(); for l in p.shape.layers.iter_mut().chain(&mut p.shape.draft) { if let engine_model::shapes::Ffn::Experts(e)=&mut l.ffn { e.storage=engine_model::shapes::ExpertStorage::PerExpert; } } let q=engine_model::split::SplitRequest{ranks:2,memory:vec![u64::MAX;2],reserve:vec![0;2],context:0,sequences:0,cache_bytes:2,state_bytes:4,draft:false,expert_rows:false}; let split=engine_model::split::SplitPlan::new(&p.shape,&q).unwrap();
    let key=cache::Identity::new(1,&p,"fixed",&split,0).unwrap(); assert_eq!(key,key.clone()); for other in [cache::Identity::new(2,&p,"fixed",&split,0).unwrap(),cache::Identity::new(1,&p,"changed",&split,0).unwrap(),cache::Identity::new(1,&p,"fixed",&split,1).unwrap(),cache::Identity::new(1,&p.clone(),"fixed",&split,0).unwrap()] { assert_ne!(key,other); }
    let group=key.clone().with_group(&[7,9],1).unwrap(); assert_ne!(group,key.clone().with_group(&[9,7],1).unwrap()); assert_ne!(group,key.clone().with_group(&[7,9],2).unwrap()); assert!(key.clone().with_group(&[7,7],1).is_err()); assert!(key.clone().with_group(&[7,9],0).is_err()); let mut other=split.clone(); other.owners[1].vocab.start+=1; assert_ne!(key,cache::Identity::new(1,&p,"fixed",&other,0).unwrap());
    assert!(cache::Identity::new(0,&p,"fixed",&split,0).is_err()); assert!(cache::Identity::new(1,&p,"fixed",&split,2).is_err());
    for n in [1,2,16,17,32,33,255,256,257] { assert_eq!(cache::replay_entry(n,n,false).unwrap(),0); assert_eq!(cache::replay_entry(n+512,n,true).unwrap(),512); assert!(cache::replay_entry(n+512,n,false).is_err()); assert!(cache::replay_entry(n,n+1,true).is_err()); }
    p.shape.tensors=vec![TensorEntry{name:"mtp.fc.weight".into(),dtype:Dtype::Bf16,shape:vec![p.shape.hidden,2*p.shape.hidden],role:Role{place:Place::Draft{layer:None},part:Part::Whole,store:Store::Weight}}]; assert!(mtp_contract::validate(&p).is_ok()); p.shape.tensors[0].dtype=Dtype::F8E4M3; assert!(mtp_contract::validate(&p).is_err()); p.shape.tensors[0].dtype=Dtype::Bf16; p.shape.tensors[0].role.part=Part::Embedding; assert!(mtp_contract::validate(&p).is_err());
    for depth in [0,2,3,4] { for limit in 0..=4 { assert_eq!(mtp_contract::depth(depth,limit),limit.min(if depth==0 {3} else {depth})); } } assert_eq!(mtp_contract::block_prefix(40,40),"mtp.layers.0"); assert_eq!(mtp_contract::block_prefix(48,48),"mtp.layers.0");
    for entry in [0,63,64,255,256,511,512,usize::MAX-4] { for kept in 1..=4 { assert_eq!(mtp_contract::verified_prefix(entry,4,entry+kept).unwrap(),kept); } assert!(mtp_contract::verified_prefix(entry,4,entry).is_err()); } assert!(mtp_contract::verified_prefix(8,3,7).is_err()); assert!(mtp_contract::verified_prefix(8,3,12).is_err());
    for h in [2048,3072] { let fc=projections::FixedLinear::new(h,2*h,false).unwrap(); assert_eq!(fc.k_ranges().iter().flat_map(|r|r.clone()).collect::<Vec<_>>(),(0..2*h as u32).collect::<Vec<_>>()); for n in [1,2,8,16,256] { assert_eq!(fc.elements(n).unwrap(),(n*2*h,n*h)); } }
}
#[test] fn prompt_chunks_preserve_token_order() {
    for n in [1,7,64,256,257,513,8192] { for chunk in [1,7,64,256,257] {
        let placed:Vec<_>=(0..n).collect::<Vec<_>>().chunks(chunk).flat_map(|rows| schedule::ranges(rows.len()).flat_map(move |r| rows[r].iter().copied())).collect();
        assert_eq!(placed,(0..n).collect::<Vec<_>>()); assert_eq!(schedule::tokens(&[n]),(0..n).map(|i| (0,i)).collect::<Vec<_>>());
    } } }

#[test] fn declared_arms_and_off_costs() {
    use spark_engine::arms::{Table,Kind,Limits}; use engine_model::{plan::ModelPlan,shapes::Mixer};
    for bytes in [include_bytes!("../../model/tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice(),include_bytes!("../../model/tests/fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice()] {
        let mut p=ModelPlan::from_config(bytes).unwrap(); let t=Table::new(&p,1,false); t.validate(&p).unwrap(); assert_eq!(t.arms.len(),5);
        let mut setup=Vec::new(); t.setup(|a| { setup.push(a.kind); assert!(!a.kernels().is_empty()); Ok(()) }).unwrap(); assert_eq!(setup,vec![Kind::Gdn,Kind::Attention,Kind::MoeFp8]);
        for k in [Kind::Mtp,Kind::Transport] { let a=t.arm(k); assert!(!a.on && a.kernels().is_empty()); assert_eq!(a.reserve(|| panic!("off arm evaluated reserve")),Some(0)); assert_eq!(a.weight_bytes(&p),Some(0)); }
        assert_eq!(t.transport_bytes(&p,usize::MAX),Some(0)); let all=Table::new(&p,2,true); assert!(all.arms.iter().all(|a|a.on)); assert_eq!(all.transport_bytes(&p,80),Some(80*p.shape.vocab as usize*4*3));
        let mut absent=p.clone(); absent.shape.layers.retain(|l| matches!(l.mixer,Mixer::Attention(_))); for l in &mut absent.shape.layers { l.ffn=engine_model::shapes::Ffn::Dense{width:128}; } let off=Table::new(&absent,1,false); for k in [Kind::Gdn,Kind::MoeFp8,Kind::Mtp,Kind::Transport] { assert_eq!(off.arm(k).reserve(|| panic!("unused arm")),Some(0)); assert!(off.arm(k).kernels().is_empty()); }
        absent.shape.layers.clear(); let zero=Table::new(&absent,1,false); zero.setup(|_| panic!("empty model set up an arm")).unwrap(); for a in &zero.arms { assert_eq!(a.reserve(|| panic!("empty arm reserve")),Some(0)); assert_eq!(a.weight_bytes(&p),Some(0)); }
        if let Mixer::Linear(ref mut l)=p.shape.layers[0].mixer { l.key_dim=129; } let error=t.validate(&p).unwrap_err(); assert!(error.contains("linear_key_head_dim=129"));
        assert!(Limits::world(2).unwrap_err().contains("world=2"));
    }
}
