#![allow(clippy::collapsible_if, clippy::manual_is_multiple_of, reason = "kept as on main while lane-persistent-step edits this file; fix after it merges")]
use engine_model::plan::{ConfigPlan,TextDescription,layer::{Kind,LayerPlan,Topology},storage::WeightStorage};
use spark_tp::projection_stage::{checked_messages,frontier,side_lanes};
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
/// GATES WHOLE: the recurrent mixer's gate inputs f_a and g_a are held whole on each rank (`Rows::held`). Then the
/// first projection stage of a recurrent layer (q, k, v, f_a, beta and g_a, all from the normed input) keeps every
/// output: the head-local ones as before, the two whole ones now. It sends nothing. With the shape rule it sent
/// exactly f_a and g_a. One exchange a recurrent layer less: 34 a decode step for the saved GLM config.
#[test]
fn whole_gate_inputs_leave_the_first_recurrent_stage_nothing_to_send() {
    use engine_model::plan::description::TokenMixer;use spark_tp::{kda_heads,resident_weights::Rows};
    let d=description();let (mut layers,mut before,mut after)=(0,0,0);
    for layer in 0..d.layers.len() {
        if !matches!(d.layers[layer].mixer,TokenMixer::Kda(_)) {continue;}
        let plan=make_plan(layer);let local=kda_heads::projections(&plan,kda_heads::layout(d.layers[layer].mixer,plan.topology).unwrap());
        // The walk to the layer's first projection step, as the decoder makes it.
        let mut ready=plan.inputs.iter().map(|input|input.name.as_str()).collect::<BTreeSet<_>>();
        let at=plan.steps.iter().position(|step| {let first=step.kind==Kind::Project;if !first {for output in &step.outputs {ready.insert(output.name.as_str());}}first}).unwrap();
        let stage=frontier(at,&plan,&ready).expect("the first recurrent stage");
        let ids=stage.outputs.iter().map(|&index|plan.steps[index].id.as_str()).collect::<BTreeSet<_>>();
        assert_eq!(ids,names(&["q","k","v","forget_down","beta_logits","output_gate_down"]),"layer {layer}");
        let sent=|held:bool|stage.outputs.iter().filter(|&&index| {let weight=&plan.steps[index].weights[0];
            let rows=if held {Rows::held(d.bind_tensor(&weight.name).unwrap().operation,&weight.shape,weight.storage,plan.topology)} else {Rows::new(&weight.shape,weight.storage,plan.topology)}.unwrap();
            rows.split && !local.contains(&weight.name)}).map(|&index|plan.steps[index].id.as_str()).collect::<Vec<_>>();
        assert_eq!(sent(false),["forget_down","output_gate_down"],"layer {layer}: the shape rule sends the two gate inputs");
        assert!(sent(true).is_empty(),"layer {layer}: whole gate inputs leave nothing to send");
        layers+=1;before+=usize::from(!sent(false).is_empty());after+=usize::from(!sent(true).is_empty());
    }
    assert_eq!((layers,before,after),(34,34,0));
}
/// GATES WHOLE widens one rule: a stage now keeps ANY whole (not split) output, where it refused a stage that held one.
/// For the saved GLM config the only whole projection outputs on two and on four Sparks are the gate inputs this
/// change holds whole: under the shape rule every projection matrix of every layer is split. So no other stage of
/// the model changes its path.
#[test]
fn the_gate_inputs_are_the_only_whole_projection_outputs_on_two_and_four_sparks() {
    use spark_tp::resident_weights::Rows;
    let d=description();
    for parts in [2u32,4] {let topology=Topology {parts,rank:0};let (mut whole_before,mut whole_now)=(Vec::new(),0);
        for layer in 0..d.layers.len() {let plan=LayerPlan::new(&d,layer,1,topology).unwrap();
            for step in plan.steps.iter().filter(|step|matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown)) {for weight in &step.weights {
                let shaped=Rows::new(&weight.shape,weight.storage,topology).unwrap();
                if !shaped.split {whole_before.push(weight.name.clone());}
                let held=Rows::held(d.bind_tensor(&weight.name).unwrap().operation,&weight.shape,weight.storage,topology).unwrap();
                if !held.split {whole_now+=1;assert!(weight.name.ends_with(".self_attn.f_a_proj.weight") || weight.name.ends_with(".self_attn.g_a_proj.weight"),"{}",weight.name);assert_eq!(held.count,128);}
            }}
        }
        assert!(whole_before.is_empty(),"parts {parts}: whole under the shape rule: {whole_before:?}");
        assert_eq!(whole_now,2*34,"parts {parts}");
    }
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
/// LOCAL FOLD (bounded_decoder expert_fold): when the walk reaches a routed layer's down step, no stage takes that step,
/// the fold is the only reader of the down outputs, and every other input of the fold is ready. So the down step and the
/// fold run as one stage in each of the model's routed layers.
#[test]
fn the_routed_fold_has_every_input_when_the_down_step_runs() {
    let (mut folds,mut staged)=(0,0);
    for layer in 0..description().layers.len() {
        let plan=make_plan(layer);
        let mut ready=plan.inputs.iter().map(|input|input.name.as_str()).collect::<BTreeSet<_>>();let mut skipped=BTreeSet::new();
        for (at,step) in plan.steps.iter().enumerate() {
            if skipped.remove(&at) {continue;}
            if matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown) && let Some(stage)=frontier(at,&plan,&ready) {
                if step.kind==Kind::ExpertDown {staged+=1;}
                for &index in &stage.outputs {ready.insert(plan.steps[index].outputs[0].name.as_str());}skipped.extend(stage.skipped);continue;
            }
            if step.kind==Kind::ExpertDown {
                let name=&step.outputs[0].name;let readers=plan.steps.iter().filter(|reader|reader.inputs.contains(name)).collect::<Vec<_>>();
                assert_eq!(readers.len(),1,"layer {layer}: the down outputs have another reader");let fold=readers[0];
                assert_eq!(fold.kind,Kind::ExpertSum,"layer {layer}");assert_eq!(&fold.inputs[2],name);assert_eq!(fold.inputs[0],step.inputs[1]);
                assert!(fold.inputs.iter().all(|input|input==name || ready.contains(input.as_str())),"layer {layer}: a fold input is not ready at the down step: {:?}",fold.inputs);
                assert!(!plan.returns.contains(name));folds+=1;
            }
            if step.kind==Kind::ExpertProject && plan.steps.get(at+1).is_some_and(|next|next.kind==Kind::ExpertProject && next.inputs==step.inputs) {
                skipped.insert(at+1);ready.insert(plan.steps[at+1].outputs[0].name.as_str());
            }
            for output in &step.outputs {ready.insert(output.name.as_str());}
        }
    }
    println!("ROUTED FOLD STATIC folds={folds} down_steps_taken_by_a_stage={staged}");
    assert_eq!((folds,staged),(42,0));
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
/// FOLD HONESTY: a refused earlier stage leaves the shared expert's down projection unrun when the walk reaches the routed
/// down step. A stage there must not take the routed down step: it would exchange the unfolded rows and skip LOCAL FOLD.
#[test]
fn a_routed_down_step_is_never_staged_after_a_refused_stage() {
    let mut checked=0;
    for layer in 0..description().layers.len() {
        let plan=make_plan(layer);let Some(down)=plan.steps.iter().position(|step|step.kind==Kind::ExpertDown) else {continue;};
        // Every step before the down step ran one at a time (no stage hoisted the shared expert), and the shared gate and up ran.
        let mut ready=plan.inputs.iter().map(|input|input.name.as_str()).collect::<BTreeSet<_>>();
        ready.extend(plan.steps[..down].iter().flat_map(|step|&step.outputs).map(|output|output.name.as_str()));
        let shared=plan.steps.iter().position(|step|step.kind==Kind::SwiGlu && step.outputs[0].name.contains("shared")).unwrap();
        ready.extend(plan.steps[shared].inputs.iter().map(String::as_str));
        for at in 0..plan.steps.len() {
            if let Some(stage)=frontier(at,&plan,&ready) {
                assert!(stage.outputs.iter().all(|&index|plan.steps[index].kind!=Kind::ExpertDown),"layer {layer}: a stage at step {at} took the routed down step");
            }
        }
        checked+=1;
    }
    assert_eq!(checked,42);
}
/// Device eligibility mirrored on the plan: a stage of plain projection groups forks a group of one matrix stored plain
/// or packed (FP8 block, NVFP4); an MXFP8 matrix or a batched group reads the shared flash input and stays on main.
/// SHARED LANE: a stage with routed experts forks its gated group (a SwiGLU and the down projection of its value).
fn eligible(plan:&LayerPlan,stage:&spark_tp::projection_stage::Frontier)->Vec<bool> {
    let plain=stage.groups.iter().all(|group|group.middle.is_none() && plan.steps[group.indices[0]].kind==Kind::Project);
    let routed=stage.groups.iter().any(|group|plan.steps[group.indices[0]].kind==Kind::ExpertProject);
    stage.groups.iter().map(|group| {let single=group.indices.len()==1 && plan.steps[group.indices[0]].weights[0].storage!=WeightStorage::Mxfp8;
        if group.middle.is_some() {routed && single} else {plain && single}}).collect()
}
#[derive(Clone,Copy,PartialEq)]
enum Lane {Main,Side,Join}
/// The commands apply_stage records for one stage: side groups, main groups, the join, then the gather on main.
fn record(plan:&LayerPlan,stage:&spark_tp::projection_stage::Frontier,side:&[bool],join:bool)->Vec<(Lane,Vec<String>,Vec<String>)> {
    // A gated group reads its SwiGLU's two inputs and writes the SwiGLU value and the down output; the value is its own.
    let group=|at:usize,lane| {let steps=&stage.groups[at].indices;
        if let Some(middle)=stage.groups[at].middle {let middle=&plan.steps[middle];
            return (lane,middle.inputs.clone(),middle.outputs.iter().map(|output|output.name.clone()).chain(steps.iter().map(|&i|plan.steps[i].outputs[0].name.clone())).collect());}
        (lane,steps.iter().flat_map(|&i|plan.steps[i].inputs.clone()).collect(),steps.iter().map(|&i|plan.steps[i].outputs[0].name.clone()).collect())};
    let mut events=(0..side.len()).filter(|&at|side[at]).map(|at|group(at,Lane::Side)).chain((0..side.len()).filter(|&at|!side[at]).map(|at|group(at,Lane::Main))).collect::<Vec<_>>();
    if join {events.push((Lane::Join,Vec::new(),Vec::new()));}
    events.push((Lane::Main,stage.outputs.iter().map(|&at|plan.steps[at].outputs[0].name.clone()).collect(),Vec::new()));events
}
/// Race checker: side commands read only values ready at the fork; write sets are disjoint and fresh; no main command
/// before the join reads or writes a side output.
fn race_free(events:&[(Lane,Vec<String>,Vec<String>)],ready:&BTreeSet<String>)->Result<(),String> {
    let (mut side,mut written,mut joined)=(BTreeSet::new(),BTreeSet::new(),false);
    for (lane,reads,writes) in events {
        match lane {
            Lane::Join=>joined=true,
            Lane::Side=>if let Some(name)=reads.iter().find(|name|!ready.contains(*name)) {return Err(format!("side reads {name} before it is ready"));},
            Lane::Main=>if let Some(name)=reads.iter().chain(writes).find(|name|!joined && side.contains(*name)) {return Err(format!("main touches side output {name} before the join"));},
        }
        for name in writes {if ready.contains(name) || !written.insert(name.clone()) {return Err(format!("{name} is written twice"));}
            if *lane==Lane::Side {side.insert(name.clone());}}
    }Ok(())
}
#[test]
fn forked_stages_are_race_free_and_a_dropped_join_is_caught() {
    let description=description();let mut forked=std::collections::BTreeMap::<String,usize>::new();let (mut total,mut caught)=(0,0);
    for (parts,rank) in [(1,0),(2,0),(2,1)] {
        for layer in 0..description.layers.len() {
            let plan=LayerPlan::new(&description,layer,1,Topology {parts,rank}).unwrap();
            let mut ready=plan.inputs.iter().map(|input|input.name.clone()).collect::<BTreeSet<_>>();let mut skipped=BTreeSet::new();
            for (at,step) in plan.steps.iter().enumerate() {
                if skipped.remove(&at) {continue;}
                if matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown) {
                    let stage=frontier(at,&plan,&ready.iter().map(String::as_str).collect());
                    if let Some(stage)=stage {
                        let side=side_lanes(&eligible(&plan,&stage));
                        if side.contains(&true) {
                            assert!(side.contains(&false),"a forked stage keeps one job on main");
                            race_free(&record(&plan,&stage,&side,true),&ready).unwrap();
                            if race_free(&record(&plan,&stage,&side,false),&ready).is_err() {caught+=1;}
                            let ids=stage.groups.iter().zip(&side).map(|(group,&lane)|format!("{}{}",plan.steps[group.indices[0]].id,if lane {"(side)"} else {""})).collect::<Vec<_>>();
                            total+=1;*forked.entry(format!("parts{parts} {}",ids.join("+"))).or_default()+=1;
                        }
                        for &output in &stage.outputs {ready.insert(plan.steps[output].outputs[0].name.clone());}
                        skipped.extend(stage.skipped);continue;
                    }
                }
                for output in &step.outputs {ready.insert(output.name.clone());}
            }
        }
    }
    for (stage,count) in &forked {println!("FORKED {count} {stage}");}
    println!("FORKED STAGES total={total} dropped_join_caught={caught}");
    assert!(total>0);assert_eq!(caught,total);
    // Each layer kind forks on both rank counts: KDA forget, MLA query_down + kv_down beside the index keys, the MoE router,
    // and (SHARED LANE) the shared expert's down beside the routed gate and up.
    for kind in ["forget(side)","query_down(side)+kv_down(side)","router_logits(side)","expert_gate+shared_output(side)"] {for parts in [1,2] {
        assert!(forked.keys().any(|stage|stage.starts_with(&format!("parts{parts} ")) && stage.contains(kind)),"{kind} forks with {parts} part(s)");}}
}
