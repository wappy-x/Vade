//! Small saved-weight fixtures. These do not represent a public checkpoint.
use engine_model::{plan::{ConfigPlan,TextDescription,description::*},shapes::{Ffn,ExpertShape,ExpertStorage},safetensors::{Tensor,Dtype}};
use std::{collections::BTreeMap,fs::{self,OpenOptions},io::Write,path::Path};
type Result<T>=std::result::Result<T,String>;
pub fn description() -> TextDescription {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("../../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    d.hidden=128;d.vocab=16;d.max_positions=8;d.declared_draft_layers=0;
    d.residual=SinkhornDescription {streams:2,iterations:4,epsilon:1e-5};
    d.routing.groups=2;d.routing.selected_groups=1;
    d.layers=vec![
        DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:4,head_dim:16,conv:4,gate_lower_bound:-5.0}),ffn:Ffn::Dense {width:64}},
        DecoderDescription {mixer:TokenMixer::LatentAttention(LatentDescription {heads:4,query_rank:32,kv_rank:32,key_dim:16,value_dim:16,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:4,dim:16,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false}}),
            ffn:Ffn::Experts(ExpertShape {count:4,selected:2,width:64,shared_width:64,storage:ExpertStorage::PerExpert})},
    ];d.storage.layers.clear();d.storage.excluded.clear();d
}
pub fn headers(d: &TextDescription) -> BTreeMap<String,Tensor> {
    let mut offset=0;let mut out=BTreeMap::new();
    for (name,w) in d.required_weights().unwrap() {
        let dtype=if name.ends_with("_base") || name.ends_with("_scale") || name.ends_with("A_log") || name.ends_with("dt_bias")
            || name.ends_with("e_score_correction_bias") || name.ends_with("post_attention_layernorm.weight") {Dtype::F32} else {Dtype::Bf16};
        let mut shape=w.shape;if name.contains("_conv1d.weight") {shape.insert(1,1);}
        let start=offset;offset+=shape.iter().product::<u64>()*dtype.bytes();
        out.insert(name,Tensor {dtype,shape,start,end:offset});
    }out
}
pub fn data(words: usize,salt: usize) -> Vec<f32> {(0..words).map(|i| ((i*13+i/7+salt)%31) as f32/16.0-0.9375).collect()}
pub fn values(name: &str,words: usize,dtype: Dtype) -> Vec<f32> {
    if name.ends_with("_scale") {return vec![0.25,0.5,0.125];}
    if name.ends_with("layernorm.weight") {
        return (0..words).map(|i| 1.0+(i%5) as f32/128.0+if dtype==Dtype::F32 {1.0/1048576.0} else {0.0}).collect();
    }
    data(words,name.len()).into_iter().map(|v| if name.ends_with("_fn") {v/1024.0} else {v/32.0}).collect()
}
pub fn payload(name: &str,t: &Tensor) -> Vec<u8> {
    let v=if name.ends_with("self_attn.q_proj.weight") || name.ends_with("self_attn.q_a_proj.weight") {
        let cols=t.shape[1] as usize;
        (0..t.shape.iter().product::<u64>() as usize).map(|i| if i%cols==(i/cols)%cols {1.0} else {0.0}).collect()
    } else {values(name,t.shape.iter().product::<u64>() as usize,t.dtype)};
    match t.dtype {Dtype::Bf16=>v.into_iter().flat_map(|v| engine_model::flash::nvfp4::f32_to_bf16(v).to_le_bytes()).collect(),
        Dtype::F32=>v.into_iter().flat_map(f32::to_le_bytes).collect(),_=>panic!()}
}
fn write(path: &Path,bytes: &[u8]) -> Result<()> {
    OpenOptions::new().write(true).create_new(true).open(path).and_then(|mut f| f.write_all(bytes)).map_err(|e| e.to_string())
}
pub fn checkpoint(root: &Path,d: &TextDescription) -> Result<Vec<u8>> {
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let h=headers(d);let mut rows=Vec::new();let mut body=Vec::new();let mut index=Vec::new();
    for (name,t) in h {
        let start=body.len();body.extend(payload(&name,&t));
        let dtype=match t.dtype {Dtype::Bf16=>"BF16",Dtype::F32=>"F32",_=>panic!()};
        rows.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{:?},\"data_offsets\":[{start},{}]}}",t.shape,body.len()));
        index.push(format!("{name:?}:\"fixture.safetensors\""));
    }
    let mut header=format!("{{{}}}",rows.join(",")).into_bytes();while header.len()%8!=0 {header.push(b' ');}
    let mut stored=(header.len() as u64).to_le_bytes().to_vec();stored.extend(header);stored.extend(&body);
    write(&root.join("fixture.safetensors"),&stored)?;
    let index=format!("{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{{}}}}}",body.len(),index.join(",")).into_bytes();
    write(&root.join("model.safetensors.index.json"),&index)?;Ok(index)
}
