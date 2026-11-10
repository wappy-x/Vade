use engine_model::{plan::{admission::CheckpointAdmission,storage::{WeightStorage,ScaleEncoding}},safetensors::Dtype};
use spark_tp::expert_path::ExpertPath;
use std::{fs,collections::BTreeMap,sync::atomic::{AtomicU64,Ordering}};
#[allow(dead_code)]
#[path="../src/bin/resident-layer-check/fixture.rs"] mod fixture;
struct Checkpoint {root:std::path::PathBuf,admission:CheckpointAdmission}
impl Drop for Checkpoint {fn drop(&mut self) {fs::remove_dir_all(&self.root).unwrap();}}
fn checkpoint()->Checkpoint {
    static NEXT:AtomicU64=AtomicU64::new(0);let mut d=fixture::description();
    d.storage.layers.insert("model.language_model.layers.1.mlp.experts".into(),WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:true});
    d.storage.layers.insert("model.language_model.layers.1.mlp.shared_experts".into(),WeightStorage::Fp8Block {block:[32,32]});
    let root=std::env::temp_dir().join(format!("vade-expert-path-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));fs::create_dir(&root).unwrap();
    let mut tensors=BTreeMap::new();
    for (name,w) in d.required_weights().unwrap() {
        let module=name.strip_suffix(".weight").unwrap_or(&name);let shape=w.shape;
        let mut put=|name:String,dtype:&str,shape:Vec<u64>,bytes:Vec<u8>| {assert!(tensors.insert(name,(dtype.to_owned(),shape,bytes)).is_none());};
        match w.storage {
            WeightStorage::Nvfp4 {..}=>{
                let id=module.split(".experts.").nth(1).unwrap().split('.').next().unwrap().parse::<u8>().unwrap();
                put(name.clone(),"U8",vec![shape[0],shape[1]/2],vec![if id%2==0 {0x12} else {0x23};(shape[0]*shape[1]/2) as usize]);
                put(format!("{module}.weight_scale"),"F8_E4M3",vec![shape[0],shape[1]/16],vec![0x30;(shape[0]*shape[1]/16) as usize]);
                put(format!("{module}.weight_scale_2"),"F32",vec![],0.25f32.to_le_bytes().to_vec());
                put(format!("{module}.input_scale"),"F32",vec![],0.5f32.to_le_bytes().to_vec());
            },
            WeightStorage::Fp8Block {block:[br,bc]}=>{
                put(name.clone(),"F8_E4M3",shape.clone(),vec![0x30;shape.iter().product::<u64>() as usize]);
                let grid=vec![shape[0].div_ceil(br),1,shape[1].div_ceil(bc),1];
                put(format!("{module}.weight_scale"),"F32",grid.clone(),(0..grid.iter().product::<u64>()).flat_map(|_|0.25f32.to_le_bytes()).collect());
            },
            WeightStorage::Plain=>{
                let mut physical=shape;if name.contains("_conv1d.weight") {physical.insert(1,1);}
                let dtype=if name.ends_with("_base") || name.ends_with("_scale") || name.ends_with("A_log") || name.ends_with("dt_bias") || name.ends_with("e_score_correction_bias") {Dtype::F32} else {Dtype::Bf16};
                let t=engine_model::safetensors::Tensor {dtype,shape:physical.clone(),start:0,end:0};
                put(name.clone(),if dtype==Dtype::F32 {"F32"} else {"BF16"},physical,fixture::payload(&name,&t));
            },_=>panic!(),
        }
    }
    let mut rows=Vec::new();let mut body=Vec::new();let mut index=Vec::new();
    for (name,(dtype,shape,bytes)) in tensors {let start=body.len();body.extend(bytes);
        rows.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",body.len()));index.push(format!("{name:?}:\"fixture.safetensors\""));}
    let mut header=format!("{{{}}}",rows.join(",")).into_bytes();while header.len()%8!=0 {header.push(b' ');}
    let total=body.len();let mut data=(header.len() as u64).to_le_bytes().to_vec();data.extend(header);data.extend(body);fs::write(root.join("fixture.safetensors"),data).unwrap();
    let index=format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{}}}}}",index.join(","));
    let admission=d.open_headers(&root,index.as_bytes()).unwrap();Checkpoint {root,admission}
}
#[test]
fn checkpoint_expert_path_keeps_calibration_planes_and_selected_slot_order() {
    let c=checkpoint();let p=ExpertPath::new(&c.admission,1,&[0,1,2,3]).unwrap();let x=vec![0.25;p.hidden()];
    let a=p.reference(&x,&[3,0],&[0.75,0.25]).unwrap();assert_eq!(a["output"].len(),p.hidden());
    assert_eq!(a["slot0.gate.quant"].len(),p.hidden()/16*17);
    assert_ne!(a["slot0.down"],a["slot1.down"]);
    assert!(a.contains_key("slot0.down.quant"));assert!(!a.contains_key("shared.gate.quant"));
    assert!(a["output"].iter().all(|v|spark_gpu_probe::bf16::rounded(*v).to_bits()==v.to_bits()));
    assert_eq!(a,p.reference(&x,&[3,0],&[0.75,0.25]).unwrap());
    assert_eq!(a["output"],p.reference(&x,&[0,3],&[0.25,0.75]).unwrap()["output"]);
    assert!(p.reference(&x,&[3,3],&[0.5,0.5]).is_err());assert!(p.reference(&x,&[4,0],&[0.5,0.5]).is_err());
    assert!(p.reference(&x[..127],&[3,0],&[0.5,0.5]).is_err());assert!(p.reference(&x,&[3,0],&[f32::NAN,0.5]).is_err());
    assert!(ExpertPath::new(&c.admission,0,&[0,1]).is_err());assert!(ExpertPath::new(&c.admission,1,&[4]).is_err());
}

#[test]
fn quantizer_matches_an_independent_nearest_even_enumeration() {
    use engine_model::flash::nvfp4::{e2m1,e4m3,f32_to_bf16,bf16_to_f32};
    use spark_gpu_probe::expert_ops::{self,Shape,Op};
    fn nearest(x:f32,last:u8,decode:fn(u8)->f32)->u8 {
        (0..=last).min_by(|&a,&b| {
            let da=(f64::from(x)-f64::from(decode(a))).abs();let db=(f64::from(x)-f64::from(decode(b))).abs();
            da.partial_cmp(&db).unwrap().then_with(||(a%2).cmp(&(b%2)))
        }).unwrap()
    }
    let mut random=0x12345678u32;
    for global in [1.0/128.0,0.03,0.5,1.0,17.3] {
        let s=Shape {op:Op::Quantize,tokens:1,rows:1,cols:16,block_rows:1,block_cols:16,global:1.0,activation:global,bound:10.0};
        for _ in 0..128 {
            let mut x=[0.0f32;16];for v in &mut x {random=random.wrapping_mul(1664525).wrapping_add(1013904223);
                *v=bf16_to_f32(f32_to_bf16((random as i32) as f32/16777216.0));}
            let maximum=x.iter().fold(0.0f32,|m,&v|m.max(v.abs()));let scale=nearest(maximum/global/6.0,126,e4m3);
            let mut expected=Vec::new();for pair in x.as_chunks::<2>().0 {
                let code=|v:f32|nearest(if scale==0 {0.0} else {v.abs()/global/e4m3(scale)},7,e2m1)|if v.is_sign_negative() {8} else {0};
                expected.push((code(pair[0])|code(pair[1])<<4) as f32);
            }expected.push(scale as f32);
            // Decode the original independently enumerated integer codes exactly.
            let decoded=expected[..8].iter().flat_map(|&v|[e2m1(v as u8&15),e2m1(v as u8>>4)])
                .chain([e4m3(expected[8] as u8)]).map(f32::to_bits).collect::<Vec<_>>();
            let actual=expert_ops::reference(s,[&x,&[],&[],&[]],&[],&[]).unwrap();
            assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),decoded);
        }
    }
}
