//! A cached program must equal a newly validated program at every causal shape.
use engine_model::plan::{ConfigPlan,description::TokenMixer,layer::{LayerPlan,LayerSchedule,Topology}};
#[test]
fn cached_programs_equal_fresh_programs_across_context_and_topology_boundaries() {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else {panic!()};
    d.max_positions=1<<20;let mut feed_forward=Vec::new();for l in &d.layers {if !feed_forward.contains(&l.ffn) {feed_forward.push(l.ffn);}}
    let mut seen=Vec::new();
    for layer in 0..d.layers.len() {for &ffn in &feed_forward {d.layers[layer].ffn=ffn;
        if seen.contains(&d.layers[layer]) {continue;}seen.push(d.layers[layer]);
        for parts in [1,2,4] {for rank in 0..parts {let t=Topology {rank,parts};let mut cached=LayerSchedule::new(&d,layer,t).unwrap();
            let mut contexts=vec![1,2,3,7,8,15,16,31,32,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096,4097,d.max_positions];
            if let TokenMixer::LatentAttention(a)=d.layers[layer].mixer {for n in [a.indexer.pool,a.indexer.token_budget] {contexts.extend([n.saturating_sub(1).max(1),n,n+1]);}}
            contexts.retain(|&c|c>0 && c<=d.max_positions);contexts.sort_unstable();contexts.dedup();
            let descending=contexts.iter().rev().copied().collect::<Vec<_>>();contexts.extend(descending);
            for context in contexts {let expected=LayerPlan::new(&d,layer,context,t).unwrap();let actual=cached.at(context).unwrap();
                assert_eq!(actual,&expected,"layer={layer} context={context} topology={t:?}");actual.validate().unwrap();
            }
            let before=cached.at(1).unwrap().clone();
            for context in [0,d.max_positions+1,u64::MAX] {assert!(cached.at(context).is_err());assert_eq!(cached.at(1).unwrap(),&before);}
        }}
    }}
}
#[test]
fn cached_context_cannot_exceed_the_device_position_abi() {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else {panic!()};
    d.max_positions=u64::MAX;let mut cached=LayerSchedule::new(&d,0,Topology {rank:0,parts:1}).unwrap();
    assert!(cached.at(u32::MAX as u64+1).is_err());assert_eq!(cached.at(1).unwrap().context,1);
}
