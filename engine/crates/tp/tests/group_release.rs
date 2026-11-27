use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_tp::group_admission::PromptGroupAdmission;
#[allow(dead_code)]
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn last_release_returns_funding_to_resident_bytes() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,8192);
    let mut admission=PromptGroupAdmission::for_sparse_heads(&c.admission,Topology {rank:0,parts:2},8192,usize::MAX,123,true,1024).unwrap();
    let idle=admission.funded_bytes().unwrap();
    assert_eq!(idle,123);
    admission.reserve(1,4096,8192).unwrap();assert!(admission.funded_bytes().unwrap()>idle);
    let one=admission.funded_bytes().unwrap();admission.reserve(2,4096,8192).unwrap();
    admission.release(2).unwrap();assert_eq!(admission.funded_bytes().unwrap(),one);
    admission.release(1).unwrap();assert_eq!(admission.live(),0);
    assert_eq!(admission.funded_bytes().unwrap(),idle);
    admission.reserve(3,4096,8192).unwrap();assert_eq!(admission.funded_bytes().unwrap(),one);
    admission.release(3).unwrap();assert_eq!(admission.funded_bytes().unwrap(),idle);
}
