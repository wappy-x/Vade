use engine_model::{plan::{ConfigPlan,TextDescription},safetensors::{Tensor,Dtype}};
use std::collections::BTreeMap;
fn description() -> TextDescription {
    let ConfigPlan::Deferred(d) = ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else {panic!()}; d
}
fn headers(d: &TextDescription) -> BTreeMap<String,Tensor> {
    [("model.language_model.embed_tokens.weight",Dtype::Bf16,vec![d.vocab,d.hidden]),
        ("model.language_model.norm.weight",Dtype::F32,vec![d.hidden]),
        ("lm_head.weight",Dtype::F8E4M3,vec![d.vocab,d.hidden]),
        ("lm_head.weight_scale",Dtype::U8,vec![d.vocab,d.hidden/32])].into_iter().map(|(name,dtype,shape)| {
            let end = shape.iter().product::<u64>()*dtype.bytes(); (name.into(),Tensor {dtype,shape,start:0,end})
        }).collect()
}
#[test]
fn retains_storage_and_complete_stop_metadata() {
    let d = description(); let h = headers(&d); let edges = d.text_edges(&h).unwrap();
    assert_eq!(edges.norm.tensor.dtype,Dtype::F32);
    assert_eq!(edges.embedding.tensor,h["model.language_model.embed_tokens.weight"]);
    assert_eq!(edges.head.tensor,h["lm_head.weight"]); assert_eq!(edges.head_scales.tensor,h["lm_head.weight_scale"]);
    assert_eq!(edges.eos_tokens,d.eos_tokens.iter().map(|&id| id as u32).collect::<Vec<_>>());
    assert_eq!((edges.hidden,edges.vocab,edges.streams),(d.hidden,d.vocab,d.residual.streams));
    assert!(ConfigPlan::Deferred(d).into_model_plan().is_err());
}
#[test]
fn rejects_missing_wrong_shapes_types_extents_and_recipes() {
    let d = description(); let h = headers(&d);
    for name in h.keys() {
        let mut bad = h.clone(); bad.remove(name); assert!(d.text_edges(&bad).is_err());
        let mut bad = h.clone(); bad.get_mut(name).unwrap().shape[0] += 1; assert!(d.text_edges(&bad).is_err());
        let mut bad = h.clone(); bad.get_mut(name).unwrap().dtype = Dtype::F16; assert!(d.text_edges(&bad).is_err());
        let mut bad = h.clone(); bad.get_mut(name).unwrap().end -= 1; assert!(d.text_edges(&bad).is_err());
        let mut bad = h.clone(); bad.get_mut(name).unwrap().start = u64::MAX; assert!(d.text_edges(&bad).is_err());
    }
    let mut d = d; d.storage.layers.remove("lm_head"); assert!(d.text_edges(&h).is_err());
}
#[test]
fn rejects_invalid_stops_and_overflow() {
    let d = description(); let h = headers(&d);
    for eos in [vec![],vec![d.vocab],vec![d.eos_tokens[0];2]] {let mut d = d.clone(); d.eos_tokens = eos; assert!(d.text_edges(&h).is_err());}
    let mut d = d; d.hidden = u64::MAX-31; assert!(d.text_edges(&h).is_err());
}
