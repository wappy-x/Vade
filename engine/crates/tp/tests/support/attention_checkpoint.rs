//! Complete attention matrix rows. These synthetic headers do not prove a decoder.
use engine_model::{plan::{admission::CheckpointAdmission,layer::{LayerPlan,Topology,Kind},storage::WeightStorage},safetensors::{Tensor,Dtype}};
use std::{collections::BTreeMap,fs,path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
#[allow(dead_code)]
#[path="../../src/bin/resident-layer-check/fixture.rs"] mod fixture;
pub struct Checkpoint {pub root:PathBuf,pub admission:CheckpointAdmission,pub operands:BTreeMap<String,(Vec<u8>,Vec<u8>)>}
impl Checkpoint {
    pub fn new(norm:Dtype)->Self {Self::new_with_head(norm,false)}
    pub fn new_with_head(norm:Dtype,head:bool)->Self {Self::new_with_capacity(norm,head,8)}
    pub fn new_with_capacity(norm:Dtype,head:bool,capacity:u64)->Self {
        static NEXT:AtomicU64=AtomicU64::new(0);
        let root=std::env::temp_dir().join(format!("vade-attention-projections-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&root).unwrap();let mut d=fixture::description();d.max_positions=capacity;d.hidden=256;let mut selected=Vec::new();
        if let engine_model::plan::description::TokenMixer::Kda(k)=&mut d.layers[0].mixer {k.head_dim=32;}
        for layer in 0..d.layers.len() {
            let p=LayerPlan::new(&d,layer,1,Topology {rank:0,parts:1}).unwrap();
            for op in &p.steps {
                if op.kind==Kind::Combine {break;}
                if op.kind==Kind::Project && op.weights[0].name.ends_with(".weight") {
                    let w=&op.weights[0];assert_eq!(w.shape[1]%32,0);
                    selected.push(w.name.clone());
                }
            }
        }
        if head {selected.push("lm_head.weight".into());}
        for name in &selected {d.storage.layers.insert(name.strip_suffix(".weight").unwrap_or(name).into(),WeightStorage::Mxfp8);}
        let mut h=fixture::headers(&d);let mut operands=BTreeMap::new();
        for name in selected {
            let t=h.get_mut(&name).unwrap();t.dtype=Dtype::F8E4M3;let shape=t.shape.clone();
            let scale=name.strip_suffix(".weight").map_or_else(||format!("{name}_scale"),|n|format!("{n}.weight_scale"));
            h.insert(scale,Tensor {dtype:Dtype::U8,shape:vec![shape[0],shape[1]/32],start:0,end:0});
            let w=(0..shape[0]*shape[1]).map(|i|if i%7==0 {0xb8} else {0x38+(i%3) as u8}).collect::<Vec<_>>();
            let s=(0..shape[0]*shape[1]/32).map(|i|124+(i%4) as u8).collect::<Vec<_>>();operands.insert(name,(w,s));
        }
        for (name,t) in &mut h {if name.ends_with("layernorm.weight") {t.dtype=norm;}}
        let mut body=Vec::new();let mut rows=Vec::new();let mut index=Vec::new();
        for (name,t) in h {
            let bytes=if let Some((w,_))=operands.get(&name) {w.clone()}
                else if let Some((_,s))=operands.get(&name.replace(".weight_scale",".weight")) {s.clone()}
                else if let Some((_,s))=name.strip_suffix("_scale").and_then(|n|operands.get(n)) {s.clone()}
                else {fixture::payload(&name,&t)};
            let start=body.len();body.extend(bytes);
            let dtype=match t.dtype {Dtype::Bf16=>"BF16",Dtype::F32=>"F32",Dtype::F8E4M3=>"F8_E4M3",Dtype::U8=>"U8",_=>panic!()};
            rows.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{:?},\"data_offsets\":[{start},{}]}}",t.shape,body.len()));
            index.push(format!("{name:?}:\"fixture.safetensors\""));
        }
        let mut header=format!("{{{}}}",rows.join(",")).into_bytes();while header.len()%8!=0 {header.push(b' ');}
        let total=body.len();let mut file=(header.len() as u64).to_le_bytes().to_vec();file.extend(header);file.extend(body);
        fs::write(root.join("fixture.safetensors"),file).unwrap();
        let index=format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{}}}}}",index.join(","));
        fs::write(root.join("config.json"),b"{}").unwrap();
        fs::write(root.join("model.safetensors.index.json"),index.as_bytes()).unwrap();
        let admission=d.open_headers(&root,index.as_bytes()).unwrap();Self {root,admission,operands}
    }
}
impl Drop for Checkpoint {fn drop(&mut self) {fs::remove_dir_all(&self.root).unwrap();}}
