use engine_model::{plan::{admission::*,layer::{Topology,Kind},state::StatePlan,storage::WeightStorage},safetensors::Dtype};
use spark_tp::layer_driver::LayerBoundary;
use spark_gpu_probe::norm_boundary::WeightType;
use std::{fs,path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
#[path="../src/bin/resident-layer-check/fixture.rs"] mod fixture;
struct Temp(PathBuf);
impl Temp {fn new() -> Self {
    static NEXT:AtomicU64=AtomicU64::new(0);
    let p=std::env::temp_dir().join(format!("vade-layer-operands-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
    fs::create_dir(&p).unwrap();Self(p)
}}
impl Drop for Temp {fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();}}
fn read(c: &ComponentView) -> engine_model::Result<Vec<u8>> {
    let t=engine_model::safetensors::Tensor {dtype:c.dtype,shape:c.shape.clone(),start:c.start,end:c.start+c.bytes};
    Ok(fixture::payload(&c.name,&t))
}
#[test]
fn every_norm_and_plain_projection_binds_for_both_layer_forms_and_ranks() {
    let d=fixture::description();let a=d.admit_headers(&fixture::headers(&d)).unwrap();
    for rank in 0..2 {let t=Topology {parts:2,rank};let state=StatePlan::new(&d,t,4,isize::MAX as usize).unwrap();
        for layer in 0..d.layers.len() {let b=LayerBoundary::new(&d,&a,&state,layer,3,t,4096).unwrap();
            for (i,step) in b.program().steps.iter().enumerate() {
                if step.kind==Kind::Norm {
                    let n=b.prepare_stored_norm(i,read).unwrap();let c=&b.weights(i).unwrap()[0].view().components()[0];
                    assert_eq!(n.stored_dtype(),c.dtype);assert_eq!(n.shape().width(),step.outputs[0].elements().unwrap());
                    assert_eq!(n.shape().epsilon().to_bits(),(d.epsilon as f32).to_bits());
                    assert_eq!(n.shape().weight_type(),if c.dtype==Dtype::F32 {WeightType::F32} else {WeightType::Bf16});
                    assert_eq!(n.weight(),fixture::values(&c.name,n.shape().width(),c.dtype));
                }
                if step.kind==Kind::Project {
                    if step.id=="router_logits" {let mut calls=0;
                        assert!(b.prepare_bf16_projection(i,|_| {calls+=1;panic!()}).is_err());assert_eq!(calls,0);continue;}
                    let p=b.prepare_bf16_projection(i,read).unwrap();let w=&step.weights[0];
                    let (origin,rows)=t.interval(&w.shape,w.split).unwrap().map_or((0,w.shape[0]),|(_,a,n)| (a,n));
                    assert_eq!(p.shape(),[rows as usize,w.shape[1] as usize]);assert_eq!(p.row_origin(),origin);assert_eq!(p.topology(),t);
                    let c=&b.weights(i).unwrap()[0].view().components()[0];let all=read(c).unwrap();let stride=w.shape[1] as usize*2;
                    assert_eq!(p.weight(),&all[origin as usize*stride..(origin+rows) as usize*stride]);
                }
            }assert!(b.require_executable().is_err());
        }
    }
}
#[test]
fn wrong_operations_nonfinite_payloads_and_quantized_projections_refuse() {
    let mut d=fixture::description();
    for quantized in [false,true] {
        let mut h=fixture::headers(&d);
        if quantized {
            let module="model.language_model.layers.0.self_attn.q_proj";
            d.storage.layers.insert(module.into(),WeightStorage::Mxfp8);
            let name=format!("{module}.weight");let w=h.get_mut(&name).unwrap();w.dtype=Dtype::F8E4M3;w.end=w.start+w.shape.iter().product::<u64>();
            h.insert(format!("{module}.weight_scale"),engine_model::safetensors::Tensor {dtype:Dtype::U8,shape:vec![64,4],start:0,end:256});
        }
        let a=d.admit_headers(&h).unwrap();let t=Topology {parts:2,rank:0};let state=StatePlan::new(&d,t,4,isize::MAX as usize).unwrap();
        let b=LayerBoundary::new(&d,&a,&state,0,1,t,4096).unwrap();let q=b.program().steps.iter().position(|s| s.id=="q").unwrap();
        assert!(b.prepare_stored_norm(q,|_| panic!()).is_err());
        assert!(b.prepare_bf16_projection(2,|_| panic!()).is_err());
        assert!(b.prepare_stored_norm(usize::MAX,|_| panic!()).is_err());
        assert!(b.prepare_bf16_projection(usize::MAX,|_| panic!()).is_err());
        assert!(b.prepare_stored_norm(2,|c| Ok(vec![0xff;c.bytes as usize])).is_err());
        if quantized {assert!(b.prepare_bf16_projection(q,|_| panic!()).is_err());}
        else {assert!(b.prepare_bf16_projection(q,|c| Ok(vec![0xff;c.bytes as usize])).is_err());}
    }
}
#[test]
fn retained_checkpoint_owns_both_residual_sides_and_rejects_different_headers() {
    let d=fixture::description();let temp=Temp::new();let root=temp.0.join("checkpoint");let index=fixture::checkpoint(&root,&d).unwrap();
    let checkpoint=d.open_headers(&root,&index).unwrap();
    // A renamed path must not make the binder reopen the admitted file.
    fs::rename(root.join("fixture.safetensors"),root.join("retained.safetensors")).unwrap();
    let t=Topology {parts:2,rank:1};let state=StatePlan::new(&d,t,4,isize::MAX as usize).unwrap();
    for layer in 0..d.layers.len() {let b=LayerBoundary::new(&d,checkpoint.headers(),&state,layer,1,t,4096).unwrap();
        for (i,op) in b.program().steps.iter().enumerate().filter(|(_,s)| s.kind==Kind::MhcPrepare) {
            let block=b.prepare_residual_block_checkpoint(i,&checkpoint).unwrap();assert_eq!(block.mhc().shape().hidden as u64,d.hidden);
            assert_eq!(block.norm().stored_dtype(),if op.id=="ffn.prepare" {Dtype::F32} else {Dtype::Bf16});
        }
        assert!(b.prepare_residual_block_checkpoint(1,&checkpoint).is_err());
        let q=b.program().steps.iter().position(|s| s.kind==Kind::Project && s.id!="router_logits").unwrap();
        assert!(!b.prepare_bf16_projection_checkpoint(q,&checkpoint).unwrap().weight().is_empty());
    }
    let mut different=fixture::headers(&d);let name="model.language_model.layers.0.input_layernorm.weight";
    let weight=different.get_mut(name).unwrap();weight.dtype=Dtype::F32;weight.end=weight.start+d.hidden*4;
    let a=d.admit_headers(&different).unwrap();let b=LayerBoundary::new(&d,&a,&state,0,1,t,4096).unwrap();
    assert!(b.prepare_stored_norm_checkpoint(2,&checkpoint).err().unwrap().contains("headers differ"));
}
