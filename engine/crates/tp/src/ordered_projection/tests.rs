use super::*;
#[path="fixture.rs"] mod fixture;
use engine_model::{plan::layer::Split,safetensors::Limits};
use std::{path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
static NEXT: AtomicU64=AtomicU64::new(0);
fn path() -> PathBuf {std::env::temp_dir().join(format!("vade-ordered-projection-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)))}
fn file(p: &LayerPlan,id: &str,dtype: &str,cols: u64) -> TensorFile {
    let w=&p.steps.iter().find(|s| s.id==id).unwrap().weights[0];
    let bytes=vec![0;w.shape[0] as usize*cols as usize*if dtype=="F32" {4} else {2}];
    let f=fixture::write(&path(),&w.name,dtype,w.shape[0],cols,&bytes).unwrap();
    TensorFile::from_file(f,Limits::default()).unwrap()
}
#[test] fn retained_plain_rows_match_complete_scalar_dots() {
    let p=fixture::plan(8,1,0).unwrap();let step=p.steps.iter().position(|s| s.id=="q").unwrap();let w=&p.steps[step].weights[0];
    let (values,x)=fixture::values(w.shape[0] as usize,w.shape[1] as usize);
    let source=path();let renamed=path();
    let f=fixture::write(&source,&w.name,"F32",w.shape[0],w.shape[1],&fixture::bytes(&values)).unwrap();
    let f=TensorFile::from_file(f,Limits::default()).unwrap();
    std::fs::rename(&source,&renamed).unwrap();
    let expected=Shape::new(w.shape[0] as usize,x.len()).unwrap().reference(&values,&x).unwrap();
    for parts in [1,2,4] {let mut actual=Vec::new();for rank in 0..parts {
        let p=fixture::plan(8,parts,rank).unwrap();
        let bound=PlainF32Projection::bind(&p,step,&f,Profile::LogicalF32V1).unwrap();
        assert_eq!(bound.row_origin(),u64::from(rank)*w.shape[0]/u64::from(parts));
        assert_eq!(bound.shape().cols(),x.len());
        actual.extend(bound.shape().reference(&bound.read().unwrap(),&x).unwrap());
    }assert_eq!(fixture::bytes(&actual),fixture::bytes(&expected));}
    // The replicated low-rank boundary keeps its complete output on every rank.
    let step=p.steps.iter().position(|s| s.id=="forget_down").unwrap();let f=file(&p,"forget_down","F32",8);
    let a=PlainF32Projection::bind(&p,step,&f,Profile::LogicalF32V1).unwrap();
    let split=fixture::plan(8,4,3).unwrap();let b=PlainF32Projection::bind(&split,step,&f,Profile::LogicalF32V1).unwrap();
    assert_eq!((a.row_origin(),a.shape()),(b.row_origin(),b.shape()));
}
#[test] fn typed_refusals_leave_runtime_profiles_and_formats_closed() {
    let mut p=fixture::plan(8,1,0).unwrap();let step=p.steps.iter().position(|s| s.id=="q").unwrap();
    let f=file(&p,"q","F32",8);
    for profile in [Profile::ConditionalBf16,Profile::CheckpointRuntime] {
        assert!(matches!(PlainF32Projection::bind(&p,usize::MAX,&f,profile),Err(Error::Profile(_))));
    }
    assert!(matches!(PlainF32Projection::bind(&p,usize::MAX,&f,Profile::LogicalF32V1),Err(Error::Operation)));
    assert!(matches!(PlainF32Projection::bind(&p,0,&f,Profile::LogicalF32V1),Err(Error::Operation)));
    let bf=file(&p,"q","BF16",8);
    assert!(matches!(PlainF32Projection::bind(&p,step,&bf,Profile::LogicalF32V1),Err(Error::Storage)));
    let wrong=file(&p,"q","F32",7);
    assert!(matches!(PlainF32Projection::bind(&p,step,&wrong,Profile::LogicalF32V1),Err(Error::Header)));
    use engine_model::plan::storage::ScaleEncoding;
    for storage in [WeightStorage::Mxfp8,WeightStorage::Fp8Block {block:[128,128]},
        WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:false},
        WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:true}] {
        p.steps[step].weights[0].storage=storage;
        assert!(matches!(PlainF32Projection::bind(&p,step,&f,Profile::LogicalF32V1),Err(Error::Storage)));
    }
    p.steps[step].weights[0].storage=WeightStorage::Plain;
    p.steps[step].weights[0].split=Split::Axis {axis:1,unit:1};
    assert!(PlainF32Projection::bind(&p,step,&f,Profile::LogicalF32V1).is_err());
}
#[test] fn primitive_host_oracle_matches_independent_layer_projection_bytes() {
    use engine_model::plan::layer::reference::{Tensor,Tensors,Data,forward};
    let d=fixture::description(8).unwrap();let p=fixture::plan(8,1,0).unwrap();
    let weights=p.steps.iter().flat_map(|s| &s.weights).map(|w| {
        (w.name.clone(),Tensor::f32(&w.shape,vec![0.03125;w.shape.iter().product::<u64>() as usize]).unwrap())
    }).collect::<Tensors>();
    let inputs=p.inputs.iter().map(|b| {let n=b.elements().unwrap();let t=if b.element==engine_model::plan::layer::Element::F32 {
        Tensor::f32(&b.shape,(0..n).map(|j| (j as f32-3.0)*0.0625).collect()).unwrap()
    } else {Tensor {shape:b.shape.clone(),data:Data::U32((0..n as u32).collect())}};(b.name.clone(),t)}).collect();
    let result=forward(&d,0,1,&weights,&inputs).unwrap();
    for op in p.steps.iter().filter(|s| s.kind==Kind::Project) {
        let w=&weights[&op.weights[0].name];let input=&result.tensors[&op.inputs[0]];
        let expected=Shape::new(w.shape[0] as usize,w.shape[1] as usize).unwrap().reference(w.floats().unwrap(),input.floats().unwrap()).unwrap();
        assert_eq!(fixture::bytes(&expected),fixture::bytes(result.tensors[&op.outputs[0].name].floats().unwrap()));
    }
}
