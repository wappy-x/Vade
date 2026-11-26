//! Host storage oracles. These sum the output ABI, not a device measurement.
use super::*;
use engine_model::plan::{ConfigPlan,TextDescription,description::TokenMixer,state::{PlaneKind,StatePlan}};
use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
#[allow(dead_code)]
#[path="../bin/resident-layer-check/fixture.rs"] pub(super) mod fixture;

pub(super) struct Local;
impl Exchange for Local {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,v:&[f32],_:usize)->Result<Vec<f32>> {Ok(v.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
fn page(bytes:usize)->usize {bytes.div_ceil(4096)*4096}
/// Sum the actual StepRows output extents. A short group's state range pins the
/// complete allocation. The state prefix is already in StateChargePlan.
fn outputs(d:&TextDescription,topology:Topology,tokens:usize)->usize {
    d.layers.iter().filter_map(|layer| {
        let TokenMixer::Kda(k)=layer.mixer else {return None;};
        let local=crate::kda_heads::layout(layer.mixer,topology).unwrap().local_heads();
        let mut shape=RowShape::new(Row::StepRows,tokens).unwrap();shape.heads=local as u32;shape.dim=k.head_dim as u32;
        let (_,_,units,words)=shape.lengths().unwrap();
        Some(page(units*words*4)-local*k.head_dim as usize*k.head_dim as usize*4)
    }).sum()
}
fn plane_storage(d:&TextDescription,plan:&StatePlan,topology:Topology,tokens:usize)->usize {
    // StatePlan has replicated recurrent planes. The decoder owns local KDA
    // heads; all latent and indexer planes stay replicated.
    plan.planes().iter().map(|plane| {
        let bytes=plane.used_bytes(tokens as u64).unwrap();
        let bytes=if matches!(plane.kind,PlaneKind::KdaState|PlaneKind::KdaHistory) {
            let layout=crate::kda_heads::layout(d.layers[plane.layer].mixer,topology).unwrap();bytes/layout.total_heads()*layout.local_heads()
        } else {bytes};page(bytes.max(4))
    }).sum::<usize>()
}
fn storage(d:&TextDescription,plan:&StatePlan,topology:Topology,tokens:usize)->usize {plane_storage(d,plan,topology,tokens)+outputs(d,topology,tokens)}

#[test]
fn kept_charge_covers_the_short_group_output_owner() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {fn drop(&mut self) {std::fs::remove_dir_all(&self.0).unwrap();}}
    let root=Scratch(std::env::temp_dir().join(format!("vade-kept-charge-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())));
    let mut d=fixture::description();d.max_positions=1024;
    let TokenMixer::Kda(k)=&mut d.layers[0].mixer else {panic!("expected recurrent fixture")};k.head_dim=32;
    let index=fixture::checkpoint(&root.0,&d).unwrap();let checkpoint=d.open_headers(&root.0,&index).unwrap();
    let engine=Factory::new(&checkpoint,Split::new(crate::bounded_decoder::Host,Local).unwrap(),1024,vec![0]).unwrap();
    let plan=crate::admission_plan::StateChargePlan::new(&d,1024).unwrap();
    for position in [1,7,128,153,255,256,257,512,1024] {
        let pinned=outputs(&d,Topology {rank:0,parts:1},position.min(256));
        let expected=plan.bytes(position).unwrap()+plan.append_reservation_bytes()+pinned;
        assert_eq!(engine.kept_charge(position as u64),expected,"position={position}: the state view pins its output owner");
        // The start line's planned charge is the same loose invoice, short-group output included.
        assert_eq!(engine.kept_prompt_bytes(position),expected,"position={position}: the start line plans the pinned output");
    }
    assert_eq!(engine.kept_prompt_bytes(1025),usize::MAX);
    assert_eq!(engine.kept_charge(1025),usize::MAX);
    assert_eq!(engine.kept_charge(u64::MAX),usize::MAX);
}

#[test]
fn real_config_short_group_output_sizes_and_pool_overrun() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!("expected text description")};
    let topology=Topology {rank:0,parts:2};
    // This is the same ownership decision Device::prompt uses. Sum the real
    // output ABI at each final-group size, then account for a prefix copy.
    for (rows,expected_output,expected_pinned) in [(1,557_056,557_056),(153,85_229_568,85_229_568),(256,142_606_336,142_606_336),(257,143_163_392,0)] {
        let output=outputs(&d,topology,rows);
        let pinned=if crate::bounded_decoder::copy_prompt_state(rows) {0} else {output};
        assert_eq!((output,pinned),(expected_output,expected_pinned),"final group={rows}");
        let charge=super::kept_charge::Tail::new(&d,topology).unwrap().bytes(rows);
        assert!(charge>=pinned);
        eprintln!("KEPT FINAL GROUP HOST rows={rows} output_bytes={output} pinned_bytes={pinned}");
    }
    let state=StatePlan::new(&d,Topology {rank:0,parts:1},16384,isize::MAX as usize).unwrap();
    for (tokens,group,expected) in [(1,1,77_053_952),(153,153,167_043_072),(256,256,227_934_208),(257,257,85_463_040),
        (512,512,94_339_072),(1024,1024,112_361_472),(1040,1024,121_860_096),(4158,1024,257_241_088),(8192,1024,364_675_072)] {
        let last=(tokens-1)%group+1;
        let allocations=state.planes().iter().enumerate().map(|(at,plane)| {
            let mut bytes=plane.used_bytes(tokens as u64).unwrap();
            if matches!(plane.kind,PlaneKind::KdaState|PlaneKind::KdaHistory) {let layout=crate::kda_heads::layout(d.layers[plane.layer].mixer,topology).unwrap();bytes=bytes/layout.total_heads()*layout.local_heads();}
            if plane.kind==PlaneKind::KdaState && last<=256 {let TokenMixer::Kda(k)=d.layers[plane.layer].mixer else {panic!()};let local=crate::kda_heads::layout(d.layers[plane.layer].mixer,topology).unwrap().local_heads();
                let mut shape=RowShape::new(Row::StepRows,last).unwrap();shape.heads=local as u32;shape.dim=k.head_dim as u32;let (_,_,units,words)=shape.lengths().unwrap();bytes=units*words*4;}
            (at+1,bytes.max(4))
        }).collect();
        let owners=crate::bounded_decoder::owners::Owners::new(allocations);
        assert_eq!(owners.bytes(),expected,"tokens={tokens}, group={group}");
        let mut ledger=crate::bounded_decoder::owners::Ledger::new();ledger.retain(&owners);ledger.retain(&owners);
        assert_eq!(ledger.bytes(),expected,"same physical owners must not be charged twice");
        if tokens==256 {
            let at=state.planes().iter().position(|p|p.kind==PlaneKind::Latent).unwrap();
            let TokenMixer::LatentAttention(a)=d.layers[state.planes()[at].layer].mixer else {panic!()};
            let room=page((257+128)*a.kv_rank as usize*4);assert_eq!(room,790_528);
            let longer=crate::bounded_decoder::owners::Owners::new(owners.allocations().iter().map(|&(id,bytes)|if id==at+1 {(1000,room)} else {(id,bytes)}).collect());
            let unrelated=crate::bounded_decoder::owners::Owners::new(owners.allocations().iter().map(|&(id,bytes)|(id+2000,bytes)).collect());
            ledger.retain(&longer);assert_eq!(ledger.bytes(),228_724_736,"111 shared owners and a new allocated tail");
            ledger.retain(&unrelated);assert_eq!(ledger.bytes(),456_658_944,"unrelated physical owners add in full");
            ledger.release(&longer);assert_eq!(ledger.bytes(),455_868_416);ledger.release(&unrelated);assert_eq!(ledger.bytes(),227_934_208);
        }
        ledger.release(&owners);assert_eq!(ledger.bytes(),expected);ledger.release(&owners);assert_eq!(ledger.bytes(),0);
    }
    let tail=outputs(&d,topology,256);let held=storage(&d,&state,topology,256);
    assert_eq!((tail,held),(142_606_336,227_934_208));
    let plain=crate::admission_plan::StateChargePlan::new(&d,16384).unwrap();
    let mut layered=d.with_draft_layer().unwrap().unwrap();layered.residual=d.residual;
    let predictor=crate::admission_plan::StateChargePlan::new(&layered,16384).unwrap();
    let old_plain=plain.bytes(256).unwrap()+plain.append_reservation_bytes();
    let old_predictor=predictor.bytes(256).unwrap()+predictor.append_reservation_bytes();
    assert_eq!((old_plain,old_predictor),(170_295_296,171_900_928));
    assert_eq!((held-old_plain,held-old_predictor),(57_638_912,56_033_280));
    let pool=6_400_000_000usize;let reservation=4_930_916_352usize;let cap=pool-reservation;
    let metadata=spark_serve::cache::publication_metadata_bytes(256,d.vocab as usize,0,0,0).unwrap();
    let old_entries=8*(old_predictor+metadata);
    assert_eq!((metadata,cap,old_entries),(621_344,1_469_083_648,1_380_178_176));
    assert!(old_entries<=cap && 8*held>cap && reservation+8*held>pool);
    assert_eq!(reservation+8*held,6_754_390_016);
    eprintln!("KEPT STORAGE HOST tail_bytes={tail} held_bytes={held} old_plain_bytes={old_plain} old_predictor_bytes={old_predictor} metadata_bytes={metadata} cap_bytes={cap} old_entries_bytes={old_entries}");
    let charged=old_predictor+super::kept_charge::Tail::new(&d,topology).unwrap().bytes(256);
    let kept=cap/(charged+metadata);
    assert_eq!((charged,kept),(314_507_264,4));
    assert!(reservation+kept*held<=pool);
    eprintln!("KEPT STORAGE HOST charged_bytes={charged} fitting_entries={kept} full_chat_and_kept_bytes={}",reservation+kept*held);
    for parts in [1,2,4] {for rank in 0..parts {for tokens in [1,7,128,255,256] {
        let topology=Topology {rank,parts};let tail=outputs(&d,topology,tokens);
        let held=storage(&d,&state,topology,tokens);
        assert_eq!(super::kept_charge::Tail::new(&d,topology).unwrap().bytes(tokens),tail);
        assert!(plain.bytes(tokens).unwrap()+plain.append_reservation_bytes()+tail>=held);
    }}}
}

#[test]
fn retained_capacity_table_from_real_config_charges() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let mut layered=d.with_draft_layer().unwrap().unwrap();layered.residual=d.residual;
    for (context,cap) in [(16384,1_469_083_648usize),(32768,261_410_138)] {
        let plan=crate::admission_plan::StateChargePlan::new(&layered,context).unwrap();
        let topology=Topology {rank:0,parts:2};let tail=super::kept_charge::Tail::new(&d,topology).unwrap();
        let state=StatePlan::new(&d,Topology {rank:0,parts:1},context as u64,isize::MAX as usize).unwrap();
        let entry=|tokens:usize,kind:usize| {
            // Exact estimates use whole prompts in groups of at most 1024.
            // Production reads owners instead; other group sizes can differ.
            let last=(tokens-1)%1024+1;
            let bytes=if kind==2 {plane_storage(&d,&state,topology,tokens)+if last<=256 {outputs(&d,topology,last)} else {0}}
                else {plan.bytes(tokens).unwrap()+plan.append_reservation_bytes()+if kind==1 {tail.bytes(tokens)} else {0}};
            spark_serve::cache::publication_metadata_bytes(tokens,d.vocab as usize,bytes,if kind==2 {state.planes().len()} else {0},0).unwrap()
        };
        for tokens in [256,1040,4158,8192] {
            let (main,loose,exact)=(entry(tokens,0),entry(tokens,1),entry(tokens,2));
            println!("KEPT CAPACITY ESTIMATE context={context} cap_bytes={cap} prompt_tokens={tokens} group=1024 main_entry_bytes={main} loose_entry_bytes={loose} exact_entry_bytes={exact} main_count={} loose_count={} exact_count={}",(cap/main).min(KEPT_PROMPTS),(cap/loose).min(KEPT_PROMPTS),(cap/exact).min(KEPT_PROMPTS));
        }
        let input_limit=super::limits(context,1).unwrap().input_tokens;
        let max=|kind| (1..=input_limit).rev().find(|&tokens|entry(tokens,kind)<=cap).unwrap_or(0);
        let (main,loose,exact)=(max(0),max(1),max(2));
        println!("KEPT MAXIMUM ESTIMATE context={context} cap_bytes={cap} group=1024 main_tokens={main} loose_tokens={loose} exact_tokens={exact} main_entry_bytes={} loose_entry_bytes={} exact_entry_bytes={}",entry(main,0),entry(loose,1),entry(exact,2));
        if context==32768 {assert_eq!((main,loose),(2570,165));assert!(entry(166,1)>cap);}
    }
}
