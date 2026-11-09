use super::{need,Kind,LayerPlan,Result};
use std::collections::BTreeSet;

pub(super) fn check(plan:&LayerPlan)->Result<()> {
    let mut skipped=BTreeSet::new();
    for (index,step) in plan.steps.iter().enumerate() {
        if step.kind==Kind::MhcPrepare {
            need(plan.steps.get(index+1).is_some_and(|next|next.kind==Kind::Collapse)
                && plan.steps.get(index+2).is_some_and(|next|next.kind==Kind::Norm),
                "deferred mHC preparation requires its collapse/norm bundle")?;
            skipped.insert(index+1);skipped.insert(index+2);
        }
    }
    let mut names=vec!["residual"];
    for step in &plan.steps {if step.kind==Kind::Combine {names.push(&step.outputs[0].name);}}
    for name in names {
        let mut readers=Vec::new();
        for (index,step) in plan.steps.iter().enumerate() {
            if skipped.contains(&index) {continue;}
            let inputs=if step.kind==Kind::Combine {&step.inputs[2..]} else {&step.inputs[..]};
            for input in inputs {if input==name {readers.push(step.kind);}}
        }
        let returns=plan.returns.iter().filter(|output|*output==name).count();
        need(readers.len()+returns==1 && (returns==1 || readers[0]==Kind::MhcPrepare),
            "deferred mHC residual requires exactly one pre consumer or layer return")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_model::plan::{ConfigPlan,layer::Topology};

    fn plan()->LayerPlan {
        let ConfigPlan::Deferred(description)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
        LayerPlan::new(&description,0,1,Topology {rank:0,parts:1}).unwrap()
    }

    #[test]
    fn residual_protocol_holds_across_layers_contexts_and_topologies() {
        let ConfigPlan::Deferred(description)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
        for parts in [1,2,4] {for rank in 0..parts {for layer in 0..description.layers.len() {for context in [1,127,1040] {
            check(&LayerPlan::new(&description,layer,context,Topology {rank,parts}).unwrap()).unwrap();
        }}}}
    }

    #[test]
    fn a_second_runtime_reader_is_rejected_at_admission() {
        let mut plan=plan();let mut extra=plan.steps.iter().find(|step|step.kind==Kind::Norm).unwrap().clone();
        extra.id="extra.norm".into();extra.inputs=vec!["after_attention".into()];extra.outputs[0].name="extra.norm.output".into();
        plan.steps.push(extra);plan.validate().unwrap();
        assert!(check(&plan).unwrap_err().contains("exactly one"));
    }

    #[test]
    fn a_return_cannot_alias_the_next_pre_consumer() {
        let mut plan=plan();plan.returns.push("after_attention".into());plan.validate().unwrap();
        assert!(check(&plan).unwrap_err().contains("exactly one"));
    }

    #[test]
    fn missing_fused_bundle_is_rejected() {
        let mut plan=plan();let index=plan.steps.iter().position(|step|step.kind==Kind::MhcPrepare).unwrap();
        plan.steps[index+1].kind=Kind::Norm;
        assert!(check(&plan).unwrap_err().contains("collapse/norm"));
    }
}
