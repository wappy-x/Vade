use engine_model::{plan::{ConfigPlan, TextDescription, description::*, layer::{*, reference::*}, state::*}, shapes::Ffn};
use std::{collections::BTreeMap, sync::Arc};

fn description() -> TextDescription {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else { panic!() };
    d.hidden=8; d.max_positions=16; d.residual=SinkhornDescription { streams:2,iterations:4,epsilon:1e-5 };
    d.layers=vec![DecoderDescription { mixer:TokenMixer::Kda(KdaDescription {heads:2,head_dim:4,conv:4,gate_lower_bound:-5.0}),ffn:Ffn::Dense {width:6} },
        DecoderDescription { mixer:TokenMixer::LatentAttention(LatentDescription {heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false}}),ffn:Ffn::Dense {width:6} }];
    d.storage.layers.clear(); d.storage.excluded.clear(); d
}
fn plan(d: &TextDescription,capacity: u64,parts: u32,rank: u32) -> StatePlan {
    StatePlan::new(d,Topology {parts,rank},capacity,isize::MAX as usize).unwrap()
}
fn staged(session: &mut SessionState,value: u8) {
    let planes=session.plan().planes().to_vec(); let mut append=session.begin_append().unwrap();
    for p in planes { append.write(p.layer,p.kind,&vec![value;p.used_bytes(append.context()).unwrap()]).unwrap(); }
    append.commit().unwrap();
}
fn contents(session: &SessionState) -> Vec<Vec<u8>> {
    session.plan().planes().iter().map(|p| session.read(p.layer,p.kind).unwrap().to_vec()).collect()
}

#[test]
fn checkpoint_description_plans_capacity_without_payload_allocation() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else { panic!() };
    for parts in [1,2,4] {
        let p=plan(&d,d.max_positions,parts,parts-1);
        assert_eq!(p.layers().len(),d.layers.len());
        for plane in p.planes() { assert_eq!(plane.used_bytes(d.max_positions).unwrap(),plane.allocation.bytes.len()); }
        println!("CHECKPOINT_STATE capacity={} parts={parts} rank={} bank={} scratch={} working={} layers={} declared_draft={} draft_planned=false",
            p.capacity(),p.topology().rank,p.state_layout().bytes(),p.scratch_bytes(),p.working_bytes(),p.layers().len(),d.declared_draft_layers);
    }
    assert!(ConfigPlan::Deferred(d).into_model_plan().is_err());
}

#[test]
fn byte_capacities_follow_program_shapes_and_local_heads() {
    let d=description(); let full=plan(&d,7,1,0);
    for rank in 0..2 {
        let p=plan(&d,7,2,rank);
        for kind in [PlaneKind::KdaState,PlaneKind::KdaHistory] {
            assert_eq!(full.plane_at(0,kind).unwrap().allocation.bytes.len(),2*p.plane_at(0,kind).unwrap().allocation.bytes.len());
        }
        for kind in [PlaneKind::Latent,PlaneKind::IndexKeys,PlaneKind::IndexGates,PlaneKind::Pools] {
            assert_eq!(full.plane_at(1,kind).unwrap().allocation.bytes.len(),p.plane_at(1,kind).unwrap().allocation.bytes.len());
        }
        let scratch=&p.layers()[1].scratch;
        let size=|name| scratch.entries().iter().find(|a| a.name==name).unwrap().bytes.len();
        assert_eq!(size("keys"),5*3*4); assert_eq!(size("values"),5*2*4);
        // The program gives no split for this step scratch. Retain its full extent.
        assert_eq!(size("expanded_kv"),5*2*(3+2)*4);
        assert_eq!(size("selected_latent"),5*4*4);
        assert_eq!(p.working_bytes(),2*p.state_layout().bytes()+p.scratch_bytes());
        let t=Topology {parts:2,rank};
        assert!(StatePlan::new(&d,t,7,p.working_bytes()-1).is_err());
        assert_eq!(StatePlan::new(&d,t,7,p.working_bytes()).unwrap(),p);
        let total=2*p.working_bytes()+3*p.state_layout().bytes();
        assert_eq!(p.check_owners(2,3,total).unwrap(),total);
        assert!(p.check_owners(2,3,total-1).is_err());
        assert!(p.check_owners(0,0,total).is_err());
        assert!(p.check_owners(usize::MAX,0,isize::MAX as usize).is_err());
    }
    let short=plan(&d,1,1,0); let long=plan(&d,16,1,0);
    let extent=|p: &StatePlan,name: &str| p.layers()[1].scratch.entries().iter().find(|a| a.name==name).unwrap().bytes.len();
    for name in ["keys","values","selected_latent","expanded_kv"] { assert_eq!(extent(&short,name),extent(&long,name)); }
    assert!(long.state_layout().bytes()>short.state_layout().bytes());
    println!("STATE_BYTES capacity=7 parts=1 bank={} scratch={} working={}",full.state_layout().bytes(),full.scratch_bytes(),full.working_bytes());
}

#[test]
fn invalid_capacity_shape_overflow_and_live_aliases_fail() {
    let d=description(); let t=Topology {parts:1,rank:0};
    for capacity in [0,17,u64::MAX] { assert!(StatePlan::new(&d,t,capacity,isize::MAX as usize).is_err()); }
    for bytes in [0,1,usize::MAX] { assert!(StatePlan::new(&d,t,7,bytes).is_err()); }
    for topology in [Topology {parts:0,rank:0},Topology {parts:2,rank:2},Topology {parts:3,rank:0}] {
        assert!(StatePlan::new(&d,topology,7,isize::MAX as usize).is_err());
    }
    let mut bad=d.clone(); bad.layers.clear(); assert!(StatePlan::new(&bad,t,7,4096).is_err());
    let mut bad=LayerPlan::new(&d,0,7,t).unwrap(); bad.inputs[0].shape=vec![u64::MAX]; assert!(ByteLayout::for_program(&bad).is_err());
    let mut huge=LayerPlan::new(&d,0,7,t).unwrap();
    huge.inputs.push(Buffer::new("huge_a",&[(isize::MAX as u64)/4]));
    huge.inputs.push(Buffer::new("huge_b",&[(isize::MAX as u64)/4]));
    assert!(ByteLayout::for_program(&huge).is_err());
    let p=plan(&d,7,1,0); let state=p.state_layout();
    let offsets=state.entries().iter().map(|a| a.bytes.start).collect::<Vec<_>>();
    state.check_offsets(&offsets,state.bytes()).unwrap();
    let mut alias=offsets.clone(); alias[1]=alias[0]; assert!(state.check_offsets(&alias,state.bytes()).is_err());
    let mut unaligned=offsets.clone(); unaligned[0]=1; assert!(state.check_offsets(&unaligned,state.bytes()).is_err());
    let mut overflow=offsets.clone(); overflow[0]=usize::MAX-3; assert!(state.check_offsets(&overflow,state.bytes()).is_err());
    assert!(state.check_offsets(&offsets,state.bytes()-1).is_err());
    assert!(state.check_offsets(&offsets[..1],state.bytes()).is_err());
    let working=p.working_layout(); let mut offsets=working.entries().iter().map(|a| a.bytes.start).collect::<Vec<_>>();
    working.check_offsets(&offsets,working.bytes()).unwrap();
    offsets[2]=offsets[1]; assert!(working.check_offsets(&offsets,working.bytes()).is_err());
    offsets[1]=offsets[0]; assert!(working.check_offsets(&offsets,working.bytes()).is_err());
    for plane in p.planes() { assert!(plane.used_bytes(8).is_err()); }
    for layer in p.layers() {
        let scratch=&layer.scratch; let mut alias=scratch.entries().iter().map(|a| a.bytes.start).collect::<Vec<_>>();
        let a=scratch.entries().iter().position(|a| a.name=="residual").unwrap();
        let b=scratch.entries().iter().position(|a| a.name=="attn.coefficients").unwrap();
        alias[b]=alias[a]; assert!(scratch.check_offsets(&alias,scratch.bytes()).is_err());
        let life=layer.program.lifetimes();
        for allocation in scratch.entries() { if let Some(&(start,end))=life.get(&allocation.name) {
            let end=if allocation.name=="pool_keys" {layer.program.steps.len()} else {end};
            assert_eq!(allocation.live,(start.map_or(0,|s| s+1),end+1));
        } }
    }
}

#[test]
fn saved_pool_output_stays_live_until_state_copy_out_ends() {
    let p=plan(&description(),7,1,0); let layer=&p.layers()[1];
    let generic=ByteLayout::for_program(&layer.program).unwrap();
    let pool=generic.entries().iter().position(|a| a.name=="pool_keys").unwrap();
    let temp=generic.entries().iter().position(|a| a.name=="ffn.normalized").unwrap();
    assert!(generic.entries()[pool].live.1<generic.entries()[temp].live.0);
    let mut offsets=generic.entries().iter().map(|a| a.bytes.start).collect::<Vec<_>>();
    offsets[temp]=offsets[pool]; generic.check_offsets(&offsets,generic.bytes()).unwrap();
    assert_eq!(layer.scratch.entries()[pool].live.1,layer.program.steps.len()+1);
    assert!(layer.scratch.check_offsets(&offsets,layer.scratch.bytes()).is_err());
    assert!(!layer.program.returns.iter().any(|name| name=="pool_keys"));
}

#[test]
fn inactive_aliases_and_empty_histories_own_no_live_bytes() {
    let d=description(); let t=Topology {parts:1,rank:0}; let p=LayerPlan::new(&d,0,7,t).unwrap();
    let layout=ByteLayout::for_program(&p).unwrap();
    let first=layout.entries().iter().position(|a| a.name=="attn.normalized").unwrap();
    let last=layout.entries().iter().position(|a| a.name=="ffn.normalized").unwrap();
    assert!(layout.entries()[first].live.1<layout.entries()[last].live.0);
    let mut offsets=layout.entries().iter().map(|a| a.bytes.start).collect::<Vec<_>>();
    offsets[last]=offsets[first]; layout.check_offsets(&offsets,layout.bytes()).unwrap();
    let mut d=d; let TokenMixer::Kda(k)=&mut d.layers[0].mixer else { panic!() }; k.conv=1;
    let p=plan(&d,1,1,0); let mut offsets=p.state_layout().entries().iter().map(|a| a.bytes.start).collect::<Vec<_>>();
    assert_eq!(p.plane_at(0,PlaneKind::KdaHistory).unwrap().allocation.bytes.len(),0);
    offsets[1]=0; p.state_layout().check_offsets(&offsets,p.state_layout().bytes()).unwrap();
}

#[test]
fn append_save_restore_truncate_and_rollback_are_atomic() {
    let p=Arc::new(plan(&description(),7,1,0)); let mut s=SessionState::new(p.clone()).unwrap();
    assert_eq!(s.context(),0); assert!(s.read(0,PlaneKind::KdaState).unwrap().iter().all(|&b| b==0));
    assert!(s.read(1,PlaneKind::Latent).unwrap().is_empty());
    let empty=s.save(p.state_layout().bytes()).unwrap();
    staged(&mut s,1); let saved=s.save(p.state_layout().bytes()).unwrap(); let one=contents(&s);
    assert!(s.save(p.state_layout().bytes()-1).is_err());
    {
        let mut a=s.begin_append().unwrap();
        let plane=p.plane_at(0,PlaneKind::KdaState).unwrap();
        a.write(0,PlaneKind::KdaState,&vec![99;plane.used_bytes(a.context()).unwrap()]).unwrap();
        assert!(a.write(1,PlaneKind::Latent,&[1]).is_err());
        a.rollback();
    }
    assert_eq!(contents(&s),one); assert_eq!(s.context(),1);
    assert!(s.begin_append().unwrap().commit().is_err()); assert_eq!(contents(&s),one);
    { let _cancel=s.begin_append().unwrap(); }
    assert_eq!(contents(&s),one);
    staged(&mut s,2); staged(&mut s,3); assert_eq!(saved.context(),1);
    s.truncate(&saved).unwrap(); assert_eq!(s.context(),1); assert_eq!(contents(&s),one);
    staged(&mut s,4); s.restore(&saved).unwrap(); assert_eq!(contents(&s),one);
    s.truncate(&empty).unwrap(); assert_eq!(s.context(),0);
    assert!(s.truncate(&saved).is_err()); s.restore(&saved).unwrap();
    for _ in 1..7 { staged(&mut s,8); }
    assert!(s.begin_append().is_err()); assert_eq!(s.context(),7);
    let mut other=SessionState::new(Arc::new(plan(&description(),8,1,0))).unwrap();
    assert!(other.restore(&saved).is_err()); assert_eq!(other.context(),0);
    let mut other=SessionState::new(Arc::new(plan(&description(),7,2,1))).unwrap();
    assert!(other.restore(&saved).is_err());
    let mut changed=description(); changed.epsilon*=2.0;
    let mut other=SessionState::new(Arc::new(plan(&changed,7,1,0))).unwrap();
    assert!(other.restore(&saved).is_err());
}

#[test]
fn complete_pools_and_tail_views_follow_each_saved_boundary() {
    for select_tail in [false,true] {
        let mut d=description(); let TokenMixer::LatentAttention(a)=&mut d.layers[1].mixer else { panic!() }; a.indexer.select_tail=select_tail;
        let p=Arc::new(plan(&d,7,1,0)); let mut s=SessionState::new(p.clone()).unwrap(); let mut saved=Vec::new();
        for context in 0..=7 {
            if context>0 { staged(&mut s,context as u8); }
            let tail=p.index_tail(1,context).unwrap();
            assert_eq!(tail.complete_pools,context/2); assert_eq!(tail.rows,context%2);
            assert_eq!(tail.keys.len(),context as usize%2*3*4); assert_eq!(tail.gates.len(),tail.keys.len());
            assert_eq!(s.read(1,PlaneKind::Pools).unwrap().len(),context as usize/2*3*4);
            assert!(tail.keys.end<=p.plane_at(1,PlaneKind::IndexKeys).unwrap().allocation.bytes.end);
            saved.push((s.save(p.state_layout().bytes()).unwrap(),contents(&s)));
        }
        for (boundary,bytes) in saved.iter().rev() { s.truncate(boundary).unwrap(); assert_eq!(&contents(&s),bytes); }
        assert!(p.index_tail(0,0).is_err()); assert!(p.index_tail(1,8).is_err()); assert!(p.index_tail(2,0).is_err());
    }
}

fn encoded(t: &Tensor) -> Vec<u8> {
    match &t.data { Data::F32(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),Data::U32(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect() }
}
fn decoded(bytes: &[u8],shape: &[u64]) -> Tensor {
    Tensor::f32(shape,bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()).unwrap()
}
fn token(d: &TextDescription,s: &mut SessionState,position: u64) -> Vec<u8> {
    let mut returned=BTreeMap::new(); let mut output=Vec::new();
    for layer in 0..d.layers.len() {
        let p=LayerPlan::new(d,layer,s.context()+1,Topology {parts:1,rank:0}).unwrap();
        let mut inputs=BTreeMap::new();
        for b in &p.inputs {
            let state=match b.name.as_str() { "kda.state" => Some(PlaneKind::KdaState),"kda.history" => Some(PlaneKind::KdaHistory),
                "latent_cache" => Some(PlaneKind::Latent),"index_keys" => Some(PlaneKind::IndexKeys),"index_gates" => Some(PlaneKind::IndexGates),_ => None };
            let t=if let Some(kind)=state { decoded(s.read(layer,kind).unwrap(),&b.shape) }
                else if b.element==Element::U32 { Tensor {shape:b.shape.clone(),data:Data::U32((0..b.elements().unwrap() as u32).collect())} }
                else { Tensor::f32(&b.shape,(0..b.elements().unwrap()).map(|i| ((i+position as usize)%9) as f32*0.03).collect()).unwrap() };
            inputs.insert(b.name.clone(),t);
        }
        let mut weights=BTreeMap::new();
        for w in p.steps.iter().flat_map(|s| &s.weights) {
            weights.entry(w.name.clone()).or_insert_with(|| Tensor::f32(&w.shape,(0..Buffer::new("w",&w.shape).elements().unwrap()).map(|i| ((i%7) as f32-3.0)*0.01).collect()).unwrap());
        }
        let f=forward(d,layer,s.context()+1,&weights,&inputs).unwrap();
        for plane in s.plan().planes().iter().filter(|v| v.layer==layer) {
            let name=match plane.kind { PlaneKind::KdaState => "next_state",PlaneKind::KdaHistory => "next_history",PlaneKind::Latent => "next_latent_cache",
                PlaneKind::IndexKeys => "next_index_keys",PlaneKind::IndexGates => "next_index_gates",PlaneKind::Pools => "pool_keys" };
            returned.insert((layer,plane.kind),encoded(&f.tensors[name]));
        }
        output.extend(encoded(&f.tensors["output"]));
    }
    let mut a=s.begin_append().unwrap(); for ((layer,kind),bytes) in returned { a.write(layer,kind,&bytes).unwrap(); } a.commit().unwrap(); output
}

#[test]
fn scalar_continuation_bytes_match_alone_interleaved_chunks_and_cache() {
    let d=description(); let p=Arc::new(plan(&d,7,1,0));
    let mut alone=SessionState::new(p.clone()).unwrap(); let mut outputs=Vec::new(); let mut boundaries=Vec::new();
    for position in 0..7 { outputs.push(token(&d,&mut alone,position)); boundaries.push(contents(&alone)); }
    let mut chunks=SessionState::new(p.clone()).unwrap(); let mut other=SessionState::new(p.clone()).unwrap();
    let mut index=0;
    for chunk in [2,1,4] {
        for _ in 0..chunk {
            assert_eq!(token(&d,&mut chunks,index as u64),outputs[index]); assert_eq!(contents(&chunks),boundaries[index]);
            token(&d,&mut other,30+index as u64); index+=1;
        }
        // Save and restore at the chunk boundary. A separate owner cannot change it.
        let saved=chunks.save(p.state_layout().bytes()).unwrap(); let mut resumed=SessionState::new(p.clone()).unwrap();
        resumed.restore(&saved).unwrap(); chunks=resumed;
    }
    let mut resumed=SessionState::new(p.clone()).unwrap();
    token(&d,&mut resumed,0); let saved=resumed.save(p.state_layout().bytes()).unwrap();
    token(&d,&mut resumed,99); resumed.truncate(&saved).unwrap();
    for index in 1..7 { assert_eq!(token(&d,&mut resumed,index as u64),outputs[index]); assert_eq!(contents(&resumed),boundaries[index]); }
    println!("STATE_LIFECYCLE scalar-only tokens=7 chunks=[2,1,4] boundaries=identical outputs=identical");
}
