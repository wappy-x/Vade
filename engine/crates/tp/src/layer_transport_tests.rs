use super::*;
use engine_model::plan::{ConfigPlan, layer::Topology};
use std::net::TcpListener;

fn program(rank: u32, layer: usize, context: u64) -> LayerPlan {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else { panic!() };
    LayerPlan::new(&d,layer,context,Topology {parts:PAIR_RANKS as u32,rank}).unwrap()
}
fn plan(rank: u32, layer: usize, context: u64, dtype: WireType) -> PairPlan {
    let p=program(rank,layer,context);
    let types=vec![dtype;p.steps.iter().map(|s| s.gather.len()).sum()];
    PairPlan::new(p,&types,usize::MAX).unwrap()
}
fn groups(work: impl Fn(usize,PairGroup)+Sync) {
    let listeners=(0..PAIR_RANKS).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect::<Vec<_>>();
    let addresses=listeners.iter().map(|l| l.local_addr().unwrap()).collect::<Vec<_>>(); drop(listeners);
    std::thread::scope(|scope| { for rank in 0..PAIR_RANKS { let work=&work; let addresses=&addresses;
        scope.spawn(move || {
            let ring=Ring::connect_options(rank,PAIR_RANKS,addresses[rank],addresses[(rank+1)%PAIR_RANKS],true).unwrap();
            work(rank,PairGroup::new(ring).unwrap());
        });
    } });
}

#[test]
fn every_saved_layer_gather_binds_in_declared_order_with_original_origins() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else { panic!() };
    for layer in 0..d.layers.len() { for context in [1,7,33] { for dtype in [WireType::Bf16,WireType::F32] {
        let a=plan(0,layer,context,dtype); let b=plan(1,layer,context,dtype);
        assert_eq!(a.manifest(),b.manifest());
        let expected=a.program.steps.iter().enumerate().flat_map(|(i,s)| s.gather.iter().map(move |n| (i,n.as_str()))).collect::<Vec<_>>();
        assert_eq!(a.gathers.iter().map(|g| (g.step,g.name())).collect::<Vec<_>>(),expected);
        for g in &a.gathers {
            let shape=&a.program.steps[g.step].outputs[g.output].shape;
            assert_eq!(g.layout.ownership().shape(),shape);
            let rows=g.rows(0).unwrap(); let other=g.rows(1).unwrap();
            assert_eq!(rows.part()+other.part(),rows.width()); assert_eq!(rows.own(),0); assert_eq!(other.own(),rows.part());
            assert_eq!(rows.packet_bytes()+other.packet_bytes(),g.layout.bytes());
            for word in 0..other.packet_bytes()/4 { assert_eq!(other.destination(word).unwrap(),word/other.part()*other.width()+other.own()+word%other.part()); }
        }
        println!("LAYER_TRANSPORT layer={layer} context={context} dtype={dtype:?} gathers={} slot_bytes={} rank_manifests_equal=true",a.gathers.len(),a.slot_bytes);
    } } }
}

#[test]
fn invalid_boundary_types_counts_duplicates_and_incomplete_inputs_are_refused() {
    let p=program(0,0,1); let n=p.steps.iter().map(|s| s.gather.len()).sum(); let types=vec![WireType::F32;n];
    assert!(PairPlan::new(p.clone(),&[],usize::MAX).is_err());
    assert!(PairPlan::new(p.clone(),&vec![WireType::U32;n],usize::MAX).is_err());
    assert!(PairPlan::new(p.clone(),&types,4).is_err());
    let mut duplicate=p.clone(); let s=duplicate.steps.iter_mut().find(|s| !s.gather.is_empty()).unwrap(); s.gather.push(s.gather[0].clone());
    assert!(PairPlan::new(duplicate,&vec![WireType::F32;n+1],usize::MAX).is_err());
    let mut missing=p.clone(); missing.steps.iter_mut().find(|s| !s.gather.is_empty()).unwrap().gather.clear();
    assert!(PairPlan::new(missing,&types,usize::MAX).is_err());
    let mut bad=p; bad.topology.parts=1; assert!(PairPlan::new(bad,&types,usize::MAX).is_err());
}

#[test]
fn step_order_rejects_skips_repeats_missing_outputs_and_early_completion() {
    let p=plan(0,3,7,WireType::F32); let mut order=Order::default();
    assert!(order.finish(&p).is_err()); assert!(order.check(&p,1,0).is_err());
    for step in 0..p.program.steps.len() {
        let outputs=p.step_gathers(step).count();
        assert!(order.check(&p,step,outputs+1).is_err());
        order.check(&p,step,outputs).unwrap(); order.next+=1;
        assert!(order.check(&p,step,outputs).is_err());
    }
    order.finish(&p).unwrap(); order.complete=true;
    assert!(order.check(&p,p.program.steps.len(),0).is_err());
}

#[test]
fn control_agrees_full_programs_and_keeps_counts_across_layers_tokens_and_requests() {
    groups(|rank,mut group| {
        for request in [11,23] { for token in [0,1,7] { for layer in [0,3] {
            let p=plan(rank as u32,layer,token+1,WireType::Bf16);
            group.agree(&p,request,token).unwrap();
            for _ in &p.gathers { group.next_sequence().unwrap(); }
            group.complete().unwrap();
        } } }
        assert_eq!(group.calls,12); assert!(group.sequence()>group.calls); assert!(!group.failed());
        println!("LAYER_CONTROL rank={rank} calls={} sequence={} ok=true",group.calls,group.sequence());
    });
}

#[test]
fn same_length_program_mismatch_fails_both_ranks_and_is_sticky() {
    groups(|rank,mut group| {
        let mut p=program(rank as u32,0,1);
        if rank==1 { p.steps[0].target.revision="0000000000000000000000000000000000000000"; }
        let n=p.steps.iter().map(|s|s.gather.len()).sum(); let p=PairPlan::new(p,&vec![WireType::F32;n],usize::MAX).unwrap();
        assert!(group.agree(&p,31,0).is_err()); assert!(group.failed());
        assert!(group.agree(&p,31,0).is_err()); assert!(group.complete().is_err());
    });
}

#[test]
fn request_sequence_type_and_length_mismatches_stop_before_device_work() {
    for case in 0..4 { groups(|rank,mut group| {
        let dtype=if case==2 && rank==1 { WireType::Bf16 } else { WireType::F32 };
        let context=if case==3 {rank as u64+1} else {1};
        let p=plan(rank as u32,0,context,dtype);
        if case==1 && rank==1 { group.next_sequence().unwrap(); }
        assert!(group.agree(&p,if case==0 {rank as u64} else {9},0).is_err()); assert!(group.failed());
    }); }
}

#[test]
fn sequence_does_not_wrap_or_reset() {
    groups(|_,mut group| {
        group.sequence=u32::MAX as u64-1;
        assert_eq!(group.next_sequence().unwrap(),u32::MAX as u64);
        assert!(group.next_sequence().is_err()); assert_eq!(group.sequence(),u32::MAX as u64);
    });
}

#[test]
fn canonical_manifest_keeps_scalar_bits_and_weight_scale_origins() {
    let p=plan(0,0,1,WireType::F32); let n=p.gathers.len();
    let mut changed=p.program.clone(); changed.settings.index_key_epsilon= -0.0;
    let a=PairPlan::new(changed.clone(),&vec![WireType::F32;n],usize::MAX).unwrap();
    changed.settings.index_key_epsilon=0.0;
    let b=PairPlan::new(changed,&vec![WireType::F32;n],usize::MAX).unwrap(); assert_ne!(a.manifest(),b.manifest());
    let first=p.program.steps.iter().flat_map(|s| &s.weights).find(|w| matches!(w.split,Split::Axis {..}) && w.shape.len()==2).unwrap();
    let Split::Axis {axis,unit}=first.split else {panic!()};
    let owner=Ownership::equal(&first.shape,axis,unit,PAIR_RANKS as u32).unwrap();
    let rows=ProjectionRows::new(&owner,1,first.storage).unwrap();
    assert_eq!(rows.shape(),first.shape[..]); assert_eq!(rows.input_features(),first.shape[1]);
    assert_eq!(rows.rows().start,first.shape[0]/PAIR_RANKS as u64);
}
