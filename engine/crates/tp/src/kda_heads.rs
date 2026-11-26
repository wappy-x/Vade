//! Complete recurrent heads owned by the same rank as their projection rows.
#![forbid(unsafe_code)]
use std::collections::{BTreeMap,BTreeSet};
use engine_model::plan::{description::TokenMixer,layer::{Kind,LayerPlan,Topology}};
use crate::{Result,sparse_heads::Layout};
pub fn layout(mixer:TokenMixer,topology:Topology)->Result<Layout> {
    let TokenMixer::Kda(k)=mixer else {return Err("recurrent ownership requires KDA".into());};
    Layout::new(k.heads,k.head_dim,k.head_dim,topology)
}
/// Direct consumers identify the head-local projections. The low-rank inputs stay global.
pub fn projections(plan:&LayerPlan,layout:Layout)->BTreeSet<String> {
    if layout.local_heads()==layout.total_heads() {return BTreeSet::new();}
    let inputs=plan.steps.iter().filter(|step|matches!(step.kind,Kind::Conv|Kind::KdaPrepare|Kind::GatedNorm))
        .flat_map(|step|step.inputs.iter()).collect::<BTreeSet<_>>();
    plan.steps.iter().filter(|step|step.kind==Kind::Project && step.outputs.iter().any(|output|inputs.contains(&output.name)))
        .flat_map(|step|step.weights.iter().map(|weight|weight.name.clone())).collect()
}
/// Each convolution parameter is [head, feature, tap]. Prepare has one head plane
/// and one [head, feature] plane. The norm parameter is shared across heads.
pub fn parameters(plan:&LayerPlan,layout:Layout,table:&mut BTreeMap<String,Vec<f32>>)->Result<()> {
    for step in plan.steps.iter().filter(|step|matches!(step.kind,Kind::Conv|Kind::KdaPrepare)) {
        for weight in &step.weights {
            let full=table.get(&weight.name).ok_or("missing recurrent parameter")?;
            if full.len()%layout.total_heads()!=0 {return Err("recurrent parameter head extent differs".into());}
            let width=full.len()/layout.total_heads();let heads=layout.heads();
            let local=full[heads.start*width..heads.end*width].to_vec();table.insert(weight.name.clone(),local);
        }
    }Ok(())
}
