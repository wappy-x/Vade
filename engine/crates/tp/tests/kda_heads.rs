use std::collections::BTreeMap;
use engine_model::plan::{ConfigPlan,layer::{Kind,LayerPlan,Topology}};
use spark_tp::kda_heads;
#[test]
fn local_parameters_reassemble_bit_for_bit_and_keep_norm_shared() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let plan=LayerPlan::new(&d,0,1,Topology {rank:0,parts:1}).unwrap();
    let mut full=BTreeMap::new();
    for step in plan.steps.iter().filter(|s|matches!(s.kind,Kind::Conv|Kind::KdaPrepare|Kind::GatedNorm)) {
        for w in &step.weights {let n=w.shape.iter().product::<u64>() as usize;full.insert(w.name.clone(),(0..n).map(|i|f32::from_bits(0x80000000|i as u32)).collect::<Vec<_>>());}
    }
    for parts in [1,2,4] {
        let local=(0..parts).map(|rank| {let mut table=full.clone();let layout=kda_heads::layout(d.layers[0].mixer,Topology {rank,parts}).unwrap();kda_heads::parameters(&plan,layout,&mut table).unwrap();table}).collect::<Vec<_>>();
        for step in plan.steps.iter().filter(|s|matches!(s.kind,Kind::Conv|Kind::KdaPrepare|Kind::GatedNorm)) {for w in &step.weights {
            if step.kind==Kind::GatedNorm {for table in &local {assert!(table[&w.name].iter().map(|v|v.to_bits()).eq(full[&w.name].iter().map(|v|v.to_bits())));}}
            else {assert!(local.iter().flat_map(|table|table[&w.name].iter().map(|v|v.to_bits())).eq(full[&w.name].iter().map(|v|v.to_bits())));}
        }}
    }
}
#[test]
fn ownership_follows_direct_consumers_and_keeps_low_rank_inputs_global() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let plan=LayerPlan::new(&d,0,1,Topology {rank:0,parts:1}).unwrap();let names=kda_heads::projections(&plan,kda_heads::layout(d.layers[0].mixer,Topology {rank:0,parts:2}).unwrap());
    assert!(kda_heads::projections(&plan,kda_heads::layout(d.layers[0].mixer,Topology {rank:0,parts:1}).unwrap()).is_empty());
    let local=plan.steps.iter().filter(|s|s.weights.first().is_some_and(|w|names.contains(&w.name))).map(|s|s.id.as_str()).collect::<Vec<_>>();
    assert_eq!(local,["q","k","v","forget","beta_logits","output_gate"]);
    assert_eq!(names.len(),6);
}
