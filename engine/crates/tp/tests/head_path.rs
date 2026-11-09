use spark_tp::head_path::{Head,mean};
use spark_gpu_probe::{bf16::rounded,text_edges::{Shape,greedy}};
use engine_model::{safetensors::Dtype,plan::storage::WeightStorage};
#[path="support/head_checkpoint.rs"] mod fixture;
#[test]
fn stored_bf16_head_rounds_before_greedy_and_reuses_selected_embedding() {
    let c=fixture::Checkpoint::new(Dtype::Bf16);let h=Head::new(&c.admission).unwrap();let s=h.shape();
    assert_eq!(h.storage(),WeightStorage::Plain);
    let residual=(0..s.residual()).map(|i|rounded((i%11) as f32/16.0-0.25)).collect::<Vec<_>>();
    let r=h.reference(&residual).unwrap();let again=h.reference(&residual).unwrap();
    assert_eq!(r.token,greedy(&r.logits).unwrap());assert_eq!(r.token,again.token);
    assert_eq!(r.logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),again.logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
    assert_eq!(r.next_embedding,h.embedding(r.token).unwrap());
    assert_eq!(r.logits,r.accumulator.iter().copied().map(rounded).collect::<Vec<_>>());
    assert_eq!(r.mean,mean(s,&residual).unwrap());
    assert_eq!(r.logits,h.reference_hidden(&r.mean).unwrap().logits);
    for row in r.next_embedding.chunks_exact(s.hidden()) {assert_eq!(row,&r.next_embedding[..s.hidden()]);}
    assert!(h.embedding(s.vocab() as u32).is_err());assert!(h.reference(&residual[..residual.len()-1]).is_err());
    let mut bad=residual.clone();bad[0]=f32::NAN;assert!(h.reference(&bad).is_err());bad[0]=1.0001;assert!(h.reference(&bad).is_err());
    assert!(h.reference_hidden(&[0.0]).is_err());
}
#[test]
fn stream_mean_has_fixed_order_and_greedy_uses_lowest_id() {
    let s=Shape::new(32,5,4).unwrap();let mut x=vec![0.0;s.residual()];
    x[0]=16_777_216.0;x[32]=1.0;x[64]=-16_777_216.0;x[96]=1.0;
    assert_eq!(mean(s,&x).unwrap()[0],0.25);
    assert_eq!(greedy(&[1.0,2.0,2.0,-1.0]).unwrap(),1);assert_eq!(greedy(&[-0.0,0.0]).unwrap(),0);
    assert_eq!(greedy(&[-5.0,-4.0,-4.0]).unwrap(),1);
    assert!(greedy(&[]).is_err());assert!(greedy(&[1.0,f32::NAN]).is_err());assert!(greedy(&[f32::INFINITY]).is_err());
}
#[test]
fn saved_mxfp8_head_uses_dynamic_activation_and_stored_norm_precision() {
    for dtype in [Dtype::Bf16,Dtype::F32] {
        let c=fixture::Checkpoint::with_head(dtype,true);let h=Head::new(&c.admission).unwrap();let s=h.shape();
        assert_eq!(h.storage(),WeightStorage::Mxfp8);
        let hidden=(0..s.hidden()).map(|i|rounded((i%17) as f32/8.0-1.0)).collect::<Vec<_>>();
        let output=h.reference_hidden(&hidden).unwrap();
        if dtype==Dtype::F32 {assert!(output.normalized.iter().any(|&v|rounded(v).to_bits()!=v.to_bits()));}
        let activation=spark_gpu_probe::dynamic_mxfp8::Activation::from_f32(&output.normalized).unwrap();
        assert!(activation.decoded().iter().zip(&output.normalized).any(|(a,b)|a.to_bits()!=b.to_bits()));
        let (w,scale)=&c.operands["lm_head.weight"];
        let expected=spark_gpu_probe::mxfp8_projection::Shape::new(s.vocab(),s.hidden(),spark_gpu_probe::norm_boundary::WeightType::F32).unwrap()
            .reference(w,scale,activation.decoded()).unwrap();
        assert_eq!(output.accumulator,expected.0);assert_eq!(output.logits,expected.1);
        assert_eq!(output.next_embedding,h.embedding(output.token).unwrap());
        assert_eq!(h.reference(&hidden.repeat(s.streams())).unwrap().logits,output.logits);
        let record=h.parameter_record(|b|format!("{}",b.len())).unwrap();assert!(record.contains("lm_head.weight_scale"));
    }
}
