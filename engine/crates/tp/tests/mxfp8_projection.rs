use engine_model::{plan::{admission::CheckpointAdmission,layer::{Topology,Kind},state::StatePlan,storage::WeightStorage},safetensors::{Tensor,Dtype}};
use spark_tp::layer_driver::LayerBoundary;
use spark_gpu_probe::norm_boundary::WeightType;
use std::{fs,path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
#[allow(dead_code)]
#[path="../src/bin/resident-layer-check/fixture.rs"] mod fixture;
const Q:&str="model.language_model.layers.0.self_attn.q_proj.weight";
const S:&str="model.language_model.layers.0.self_attn.q_proj.weight_scale";
const N:&str="model.language_model.layers.0.input_layernorm.weight";
struct Checkpoint {root:PathBuf,admission:CheckpointAdmission,weight:Vec<u8>,scale:Vec<u8>}
impl Checkpoint {
    fn new(input:Dtype,quantized:bool,bad_scale:bool)->Self {
        static NEXT:AtomicU64=AtomicU64::new(0);
        let root=std::env::temp_dir().join(format!("vade-mxfp8-projection-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&root).unwrap();let mut d=fixture::description();let mut h=fixture::headers(&d);
        h.get_mut(N).unwrap().dtype=input;
        let mut weight=Vec::new();let mut scale=Vec::new();
        if quantized {
            d.storage.layers.insert(Q.strip_suffix(".weight").unwrap().into(),WeightStorage::Mxfp8);
            let q=h.get_mut(Q).unwrap();q.dtype=Dtype::F8E4M3;let shape=q.shape.clone();
            h.insert(S.into(),Tensor {dtype:Dtype::U8,shape:vec![shape[0],shape[1]/32],start:0,end:0});
            weight=(0..shape[0]*shape[1]).map(|i|if i%7==0 {0xb8} else {0x38+(i%3) as u8}).collect();
            scale=(0..shape[0]*shape[1]/32).map(|i|if bad_scale {255} else {124+(i%4) as u8}).collect();
        }
        let mut body=Vec::new();let mut rows=Vec::new();let mut index=Vec::new();
        for (name,t) in h {
            let bytes=if name==Q && quantized {weight.clone()} else if name==S {scale.clone()} else {fixture::payload(&name,&t)};
            let start=body.len();body.extend(bytes);
            let dtype=match t.dtype {Dtype::Bf16=>"BF16",Dtype::F32=>"F32",Dtype::F8E4M3=>"F8_E4M3",Dtype::U8=>"U8",_=>panic!()};
            rows.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{:?},\"data_offsets\":[{start},{}]}}",t.shape,body.len()));
            index.push(format!("{name:?}:\"fixture.safetensors\""));
        }
        let mut header=format!("{{{}}}",rows.join(",")).into_bytes();while header.len()%8!=0 {header.push(b' ');}
        let total=body.len();let mut file=(header.len() as u64).to_le_bytes().to_vec();file.extend(header);file.extend(body);
        fs::write(root.join("fixture.safetensors"),file).unwrap();
        let index=format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{}}}}}",index.join(","));
        let admission=d.open_headers(&root,index.as_bytes()).unwrap();
        Self {root,admission,weight,scale}
    }
}
impl Drop for Checkpoint {fn drop(&mut self) {fs::remove_dir_all(&self.root).unwrap();}}
#[test]
fn retained_mxfp8_rows_keep_all_features_and_match_the_unsplit_projection() {
    for input in [Dtype::Bf16,Dtype::F32] {
        let c=Checkpoint::new(input,true,false);let d=c.admission.description();
        fs::rename(c.root.join("fixture.safetensors"),c.root.join("retained.safetensors")).unwrap();
        let x=vec![0.5;d.hidden as usize];let mut full=Vec::new();
        for parts in [1,2,4] {let mut joined=Vec::new();
            for rank in 0..parts {
                let t=Topology {rank,parts};let state=StatePlan::new(d,t,1,isize::MAX as usize).unwrap();
                let b=LayerBoundary::new(d,c.admission.headers(),&state,0,1,t,4096).unwrap();
                let step=b.program().steps.iter().position(|s|s.id=="q").unwrap();
                let p=b.prepare_mxfp8_projection_checkpoint(step,&c.admission).unwrap();
                assert_eq!(p.shape().cols(),d.hidden as usize);assert_eq!(p.topology(),t);
                assert_eq!(p.shape().input(),if input==Dtype::Bf16 {WeightType::Bf16} else {WeightType::F32});
                assert_eq!(p.component_names(),[Q,S]);assert_eq!(p.input_name(),"input_norm");assert_eq!(p.output_name(),"q");
                let start=p.row_origin() as usize*p.shape().cols();let len=p.shape().weight_bytes();
                let (w,s)=p.read().unwrap();assert_eq!(w,&c.weight[start..start+len]);assert_eq!(s,&c.scale[start/32..(start+len)/32]);
                assert_eq!(p.component_offsets(),[start as u64,start as u64/32]);
                joined.extend(p.shape().reference(&w,&s,&x).unwrap().1);
                assert!(b.require_executable().is_err());assert!(b.prepare_bf16_projection_checkpoint(step,&c.admission).is_err());
            }
            if full.is_empty() {full=joined;} else {assert_eq!(joined,full);}
        }
    }
}
#[test]
#[allow(clippy::if_same_then_else)] // the branches are separate test cases.
fn wrong_storage_producer_step_and_nonfinite_rows_refuse() {
    for (quantized,bad) in [(false,false),(true,false),(true,true)] {
        let c=Checkpoint::new(Dtype::Bf16,quantized,bad);let d=c.admission.description();let t=Topology {rank:0,parts:2};
        let state=StatePlan::new(d,t,1,isize::MAX as usize).unwrap();
        let b=LayerBoundary::new(d,c.admission.headers(),&state,0,1,t,4096).unwrap();
        assert!(b.prepare_mxfp8_projection_checkpoint(usize::MAX,&c.admission).is_err());
        for (step,op) in b.program().steps.iter().enumerate() {
            if op.kind!=Kind::Project || op.id!="q" {assert!(b.prepare_mxfp8_projection_checkpoint(step,&c.admission).is_err());}
            else if !quantized {assert!(b.prepare_mxfp8_projection_checkpoint(step,&c.admission).is_err());}
            else {assert_eq!(b.prepare_mxfp8_projection_checkpoint(step,&c.admission).unwrap().read().is_err(),bad);}
        }
    }
}
