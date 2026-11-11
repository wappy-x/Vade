use spark_tp::causal_layer::Layer;
use engine_model::safetensors::Dtype;
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn checkpoint_causal_dense_layer_owns_history_and_recurrent_boundaries() {
    let c=fixture::Checkpoint::new(Dtype::Bf16);assert!(!c.operands.is_empty());let layer=Layer::new(&c.admission,0).unwrap();
    let input=vec![0.25;c.admission.description().hidden as usize*c.admission.description().residual.streams as usize];
    let zero=layer.zero_state().unwrap();let (first,state)=layer.reference(&input,&zero).unwrap();
    assert_eq!(state.position,1);assert_eq!(zero.position,0);assert!(zero.recurrent.iter().all(|&v|v==0.0));
    assert_eq!(first,layer.reference(&input,&zero).unwrap().0);
    let (second,next)=layer.reference(&input,&state).unwrap();assert_eq!(next.position,2);
    assert_ne!(state.recurrent,next.recurrent);assert_ne!(first["convolved"],second["convolved"]);
    assert!(first.contains_key("output"));assert!(first.contains_key("ffn_output"));
    assert!(first["output"].iter().all(|&v|spark_gpu_probe::bf16::rounded(v).to_bits()==v.to_bits()));
    assert!(Layer::new(&c.admission,1).is_err());let mut bad=input;bad[0]=f32::NAN;
    assert!(layer.reference(&bad,&state).is_err());assert_eq!(state.position,1);
}
