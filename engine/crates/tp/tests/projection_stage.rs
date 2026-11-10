#![allow(clippy::collapsible_if, clippy::manual_is_multiple_of, reason = "kept as on main while lane-persistent-step edits this file; fix after it merges")]
use engine_model::plan::{ConfigPlan,TextDescription,layer::{Kind,LayerPlan,Topology},storage::WeightStorage};
use spark_tp::projection_stage::{checked_messages,frontier};
use std::collections::BTreeSet;

fn description()->TextDescription {
    let ConfigPlan::Deferred(value)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};value
}
fn make_plan(layer:usize)->LayerPlan {LayerPlan::new(&description(),layer,1,Topology {parts:2,rank:0}).unwrap()}
fn names<'a>(items:&[&'a str])->BTreeSet<&'a str> {items.iter().copied().collect()}
#[test]
fn independent_frontiers_never_recompute_previously_ready_outputs() {
    for layer in 0..description().layers.len() {let plan=make_plan(layer);let mut ready=plan.inputs.iter().map(|input|input.name.as_str()).collect::<BTreeSet<_>>();let mut skipped=BTreeSet::new();let mut produced=BTreeSet::new();
        for (at,step) in plan.steps.iter().enumerate() {
            if skipped.remove(&at) {continue;}
            if matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown) {
                if let Some(stage)=frontier(at,&plan,&ready) {
                    assert!(stage.outputs.len()>=2);assert!(stage.outputs.contains(&at));
                    for &output in &stage.outputs {let name=plan.steps[output].outputs[0].name.as_str();assert!(!ready.contains(name));assert!(produced.insert(output));ready.insert(name);}
                    skipped.extend(stage.skipped);continue;
                }
            }
            assert!(produced.insert(at));for output in &step.outputs {ready.insert(output.name.as_str());}
        }
        assert!(skipped.is_empty());
    }
}
#[test]
fn kda_forget_and_gate_are_one_frontier_with_identical_rounding_boundaries() {
    let plan=make_plan(0);let at=plan.steps.iter().position(|step|step.id=="forget").unwrap();
    let ready=names(&["forget_down","output_gate_down"]);let stage=frontier(at,&plan,&ready).unwrap();
    assert_eq!(stage.groups.len(),2);assert_eq!(stage.outputs.iter().map(|&at|plan.steps[at].id.as_str()).collect::<Vec<_>>(),["forget","output_gate"]);
    assert!(stage.groups.iter().all(|group|group.middle.is_none() && group.indices.len()==1));
    let mut ready=ready;ready.insert("output_gate");assert!(frontier(at,&plan,&ready).is_none());
}
#[test]
fn router_stays_separate_and_block_fp8_shared_projections_keep_their_kernels() {
    let description=description();let layer=description.layers.iter().position(|layer|matches!(layer.ffn,engine_model::shapes::Ffn::Experts(_))).unwrap();let plan=make_plan(layer);
    let at=plan.steps.iter().position(|step|step.kind==Kind::Project && plan.steps.iter().any(|route|route.kind==Kind::Route && route.inputs.contains(&step.outputs[0].name))).unwrap();
    let ready=names(&[plan.steps[at].inputs[0].as_str()]);let stage=frontier(at,&plan,&ready).unwrap();
    assert_eq!(stage.groups[0].indices,vec![at]);assert_eq!(stage.groups[0].middle,None);
    // The router gate stays alone (its output feeds the Route step). The shared gate and up have one input, one storage and one shape: one two-job group.
    assert_eq!(stage.groups.len(),2);assert_eq!(stage.groups[1].indices.len(),2);
    let (gate,up)=(&plan.steps[stage.groups[1].indices[0]],&plan.steps[stage.groups[1].indices[1]]);
    assert!(matches!(gate.weights[0].storage,WeightStorage::Fp8Block {..}));assert_eq!(gate.weights[0].storage,up.weights[0].storage);assert_eq!(gate.weights[0].shape,up.weights[0].shape);assert_eq!(gate.inputs,up.inputs);
    // A different shape, a different storage, or a Route consumer on either output keeps the two apart.
    let mut other=make_plan(layer);let up=stage.groups[1].indices[1];other.steps[up].weights[0].shape[0]+=128;
    let split=frontier(at,&other,&ready).unwrap();assert_eq!(split.groups.len(),3);assert!(split.groups.iter().all(|group|group.indices.len()==1));
    let mut other=make_plan(layer);other.steps[up].weights[0].storage=WeightStorage::Fp8Block {block:[64,64]};
    let split=frontier(at,&other,&ready).unwrap();assert_eq!(split.groups.len(),3);assert!(split.groups.iter().all(|group|group.indices.len()==1));
    let mut other=make_plan(layer);let route=other.steps.iter().position(|step|step.kind==Kind::Route).unwrap();let output=other.steps[up].outputs[0].name.clone();other.steps[route].inputs.push(output);
    let split=frontier(at,&other,&ready).unwrap();assert!(split.groups.iter().all(|group|group.indices.len()==1));
}
#[test]
fn shared_output_moves_only_with_a_private_ready_pure_swiglu() {
    let description=description();let layer=description.layers.iter().position(|layer|matches!(layer.ffn,engine_model::shapes::Ffn::Experts(_))).unwrap();let mut plan=make_plan(layer);
    let at=plan.steps.iter().position(|step|step.kind==Kind::ExpertProject).unwrap();
    let middle=plan.steps.iter().position(|step|step.kind==Kind::SwiGlu && step.outputs[0].name.contains("shared")).unwrap();
    let inputs=plan.steps[at].inputs.iter().chain(&plan.steps[middle].inputs).map(String::as_str).collect::<BTreeSet<_>>();
    let stage=frontier(at,&plan,&inputs).unwrap();assert!(stage.groups.iter().any(|group|group.middle==Some(middle)));assert!(stage.skipped.contains(&middle));
    let inputs=inputs.iter().map(|name|name.to_string()).collect::<Vec<_>>();let ready=inputs.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let excludes_middle=|plan:&LayerPlan| {let stage=frontier(at,plan,&ready).unwrap();assert!(stage.groups.iter().all(|group|group.middle!=Some(middle)));assert!(!stage.skipped.contains(&middle));};
    let name=plan.steps[middle].outputs[0].name.clone();plan.returns.push(name.clone());excludes_middle(&plan);plan.returns.pop();
    let extra=plan.steps.iter().position(|step|step.kind==Kind::Combine).unwrap();plan.steps[extra].inputs.push(name);excludes_middle(&plan);plan.steps[extra].inputs.pop();
    plan.steps[middle].kind=Kind::Combine;excludes_middle(&plan);
}
#[test]
fn duplicate_writers_and_unsupported_batch_storage_refuse_grouping() {
    let mut plan=make_plan(0);let at=plan.steps.iter().position(|step|step.id=="forget").unwrap();
    let ready=names(&["forget_down","output_gate_down"]);let duplicate=plan.steps[at].clone();plan.steps.push(duplicate);assert!(frontier(at,&plan,&ready).is_none());
    let mut plan=make_plan(0);let first=plan.steps.iter().position(|step|step.kind==Kind::Project).unwrap();let source=plan.steps[first].inputs[0].clone();
    for step in &mut plan.steps {if step.kind==Kind::Project && step.inputs==[source.clone()] {step.weights[0].storage=WeightStorage::Fp8Block {block:[128,128]};}}
    // Block FP8 groups only matrices of one storage and one shape.
    let stage=frontier(first,&plan,&names(&[source.as_str()])).unwrap();
    for group in &stage.groups {let head=&plan.steps[group.indices[0]];assert!(group.indices.iter().all(|&at|plan.steps[at].weights[0].shape==head.weights[0].shape && plan.steps[at].weights[0].storage==head.weights[0].storage));}
}
#[test]
fn bounded_exchange_envelope_checks_jobs_ranks_slots_and_overflow() {
    for jobs in [2,3,16] {checked_messages(jobs,1024,4096).unwrap();}
    for (jobs,words,slot) in [(1,1,4),(17,1,4),(2,0,4),(2,1025,4096),(2,usize::MAX,usize::MAX),(2,usize::MAX/4+1,usize::MAX)] {assert!(checked_messages(jobs,words,slot).is_err());}
}
#[test]
fn split_shape_predicate_preserves_the_old_path_for_replicas_and_invalid_extents() {
    use spark_tp::{projection_stage::checked_rows,resident_weights::Rows};
    for parts in 2..=4 {for rank in 0..parts {let topology=Topology {parts,rank};
        for count in [1,3,127,128,129,2048] {let row=Rows {count,start:count*rank as usize,total:count*parts as usize,split:true};assert_eq!(checked_rows(&vec![row;32]).unwrap(),count*32);
            for bad in [Rows {split:false,..row},Rows {total:0,..row},Rows {start:row.total,..row},Rows {count:0,..row}] {assert!(checked_rows(&[bad]).is_err());}
            assert!(checked_rows(&vec![row;33]).is_err());
        }
        let replica=Rows::new(&[1,4096],WeightStorage::Plain,topology).unwrap();assert!(checked_rows(&[replica]).is_err());assert!(checked_rows(&[]).is_err());
    }}
    assert!(checked_rows(&[Rows {start:0,count:1,total:1,split:false}]).is_err());
}
fn schedule(plan:&LayerPlan,combined:bool)->(usize,usize,usize) {
    let mut ready=plan.inputs.iter().map(|input|input.name.as_str()).collect::<BTreeSet<_>>();let mut skipped=BTreeSet::new();let (mut messages,mut stages,mut old)=(0,0,0);
    for (at,step) in plan.steps.iter().enumerate() {
        if skipped.remove(&at) {continue;}
        if combined && matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown) {
            if let Some(stage)=frontier(at,plan,&ready) {
                messages+=1;stages+=1;old+=stage.groups.len();for &index in &stage.outputs {ready.insert(plan.steps[index].outputs[0].name.as_str());}skipped.extend(stage.skipped);continue;
            }
        }
        if step.kind==Kind::Project {
            let mut batch=Vec::new();
            for (index,next) in plan.steps.iter().enumerate().skip(at) {
                if next.outputs.iter().any(|output|step.inputs.contains(&output.name)) {break;}
                if next.kind!=Kind::Project || next.inputs!=step.inputs || ready.contains(next.outputs[0].name.as_str())
                    || plan.steps.iter().any(|route|route.kind==Kind::Route && route.inputs.contains(&next.outputs[0].name)) {continue;}
                batch.push(index);if batch.len()==9 {break;}
            }
            let mixed=batch.iter().all(|&index|matches!(plan.steps[index].weights[0].storage,WeightStorage::Mxfp8|WeightStorage::Plain) && plan.steps[index].weights[0].shape[1]%32==0);
            let packed=batch.iter().all(|&index|matches!(plan.steps[index].weights[0].storage,WeightStorage::Nvfp4 {..}) && plan.steps[index].weights[0].shape==step.weights[0].shape);
            if batch.len()>1 && batch[0]==at && (mixed || packed) {
                for index in batch {ready.insert(plan.steps[index].outputs[0].name.as_str());if index>at {skipped.insert(index);}}messages+=1;continue;
            }
            let row=spark_tp::resident_weights::Rows::new(&step.weights[0].shape,step.weights[0].storage,plan.topology).unwrap();if row.split {messages+=1;}
        } else if matches!(step.kind,Kind::ExpertProject|Kind::ExpertDown) {
            messages+=1;if step.kind==Kind::ExpertProject && plan.steps.get(at+1).is_some_and(|next|next.kind==Kind::ExpertProject && next.inputs==step.inputs) {
                skipped.insert(at+1);ready.insert(plan.steps[at+1].outputs[0].name.as_str());
            }
        } else if step.kind==Kind::ExpandKv {messages+=1;}
        for output in &step.outputs {ready.insert(output.name.as_str());}
    }(messages,stages,old)
}
#[test]
fn complete_layer_frontiers_match_the_dependency_message_estimate() {
    let (mut head,mut candidate,mut accepted,mut old)=(1,1,0,0);
    for layer in 0..description().layers.len() {let plan=make_plan(layer);let baseline=schedule(&plan,false);let next=schedule(&plan,true);head+=baseline.0;candidate+=next.0;accepted+=next.1;old+=next.2;}
    println!("EXCHANGE STATIC head={head} candidate={candidate} accepted={accepted} original_messages={old}");
    assert_eq!((head,candidate,accepted,old),(472,279,177,328));
}
#[test]
fn one_compatible_group_admits_several_independent_outputs() {
    let plan=make_plan(0);let at=plan.steps.iter().position(|step|step.kind==Kind::Project).unwrap();
    let source=plan.steps[at].inputs[0].as_str();let mut only=plan.clone();
    only.steps.retain(|step|step.kind==Kind::Project && step.inputs==[source]);
    let stage=frontier(0,&only,&names(&[source])).unwrap();assert_eq!(stage.groups.len(),1);assert!(stage.outputs.len()>1);
    let mut ready=names(&[source]);for &index in &stage.outputs {ready.insert(only.steps[index].outputs[0].name.as_str());}
    assert!(frontier(0,&only,&ready).is_none());
}
