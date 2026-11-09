use super::*;
pub(super) fn attention_tail(plan:&LayerPlan,index:usize)->Option<(&str,&str)> {
    let expand=plan.steps.get(index)?;let scores=plan.steps.get(index+1)?;
    let softmax=plan.steps.get(index+2)?;let values=plan.steps.get(index+3)?;
    if expand.kind!=Kind::ExpandKv || scores.kind!=Kind::Scores || softmax.kind!=Kind::Softmax || values.kind!=Kind::Values {return None;}
    if scores.inputs.get(1)!=expand.outputs.first().map(|v|&v.name) || values.inputs.get(1)!=expand.outputs.get(1).map(|v|&v.name)
        || softmax.inputs.first()!=scores.outputs.first().map(|v|&v.name) || values.inputs.first()!=softmax.outputs.first().map(|v|&v.name)
        || scores.inputs.get(2)!=expand.inputs.get(1) || softmax.inputs.get(1)!=expand.inputs.get(1) || values.inputs.get(2)!=expand.inputs.get(1) {return None;}
    let private=expand.outputs.iter().chain(&scores.outputs).chain(&softmax.outputs).map(|v|&v.name).collect::<Vec<_>>();
    if plan.returns.iter().any(|name|private.contains(&name)) || plan.steps[index+4..].iter().any(|step|step.inputs.iter().any(|name|private.contains(&name))) {return None;}
    Some((scores.inputs.first()?,values.outputs.first()?.name.as_str()))
}
#[cfg(test)]
#[allow(dead_code)]
#[path="../bin/resident-layer-check/fixture.rs"] mod fixture;
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn checkpoint_latent_plans_admit_direct_rows() {
        let engine_model::plan::ConfigPlan::Deferred(description)=engine_model::plan::ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
        for (layer,spec) in description.layers.iter().enumerate() {
            if !matches!(spec.mixer,TokenMixer::LatentAttention(_)) {continue;}
            for (rank,parts) in [(0,1),(0,2),(1,2)] {
                let mut schedule=LayerSchedule::new(&description,layer,engine_model::plan::layer::Topology {rank,parts}).unwrap();
                for context in [1041,4159,16384,32768] {
                    let plan=schedule.at(context).unwrap();
                    let index=plan.steps.iter().position(|step|step.kind==Kind::ExpandKv).unwrap();
                    assert!(attention_tail(&plan,index).is_some(),"layer {layer} rank {rank} context {context}: {:?}",plan.steps.iter().map(|step|(&step.id,&step.inputs,&step.outputs)).collect::<Vec<_>>());
                }
            }
        }
    }
    #[test] fn only_the_private_unmodified_attention_tail_can_be_replaced() {
        let description=fixture::description();
        let layer=description.layers.iter().position(|layer|matches!(layer.mixer,TokenMixer::LatentAttention(_))).unwrap();
        let plan=LayerPlan::new(&description,layer,3,engine_model::plan::layer::Topology {rank:0,parts:1}).unwrap();
        let index=plan.steps.iter().position(|step|step.kind==Kind::ExpandKv).unwrap();
        assert!(attention_tail(&plan,index).is_some());
        assert!(attention_tail(&plan,plan.steps.len()).is_none());
        for offset in 0..3 {for output in &plan.steps[index+offset].outputs {
            let mut visible=plan.clone();visible.returns.push(output.name.clone());assert!(attention_tail(&visible,index).is_none());
            let mut consumed=plan.clone();consumed.steps.last_mut().unwrap().inputs.push(output.name.clone());assert!(attention_tail(&consumed,index).is_none());
        }}
        for offset in 1..4 {let mut wrong=plan.clone();wrong.steps[index+offset].kind=Kind::Norm;assert!(attention_tail(&wrong,index).is_none());}
        for (offset,input) in [(1,1),(1,2),(2,0),(2,1),(3,0),(3,1),(3,2)] {
            let mut wrong=plan.clone();wrong.steps[index+offset].inputs[input]="unrelated".into();assert!(attention_tail(&wrong,index).is_none());
        }
    }
}
