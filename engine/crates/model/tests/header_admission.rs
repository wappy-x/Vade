use engine_model::{Result,plan::{ConfigPlan,TextDescription,description::*,storage::*,admission::*},
    safetensors::{Header,Limits,Tensor,Dtype},shapes::*};
use std::{collections::BTreeMap,io::Cursor,path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
const FIXTURE: &[u8]=include_bytes!("fixtures/glm/bounded-headers.json");
fn description()->TextDescription {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else {panic!()};
    d.hidden=64; d.vocab=64; d.max_positions=64; d.residual.streams=2; d.declared_draft_layers=1;
    d.layers=vec![DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:2,head_dim:32,conv:3,gate_lower_bound:-5.0}),ffn:Ffn::Dense {width:64}},
        DecoderDescription {mixer:TokenMixer::LatentAttention(LatentDescription {heads:2,query_rank:32,kv_rank:32,key_dim:32,value_dim:32,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:2,dim:32,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false}}),
            ffn:Ffn::Experts(ExpertShape {count:2,selected:1,width:32,shared_width:32,storage:ExpertStorage::PerExpert})}];
    d.storage=StorageMap {excluded:vec![],layers:BTreeMap::from([
        ("lm_head".into(),WeightStorage::Mxfp8),
        ("model.language_model.layers.0.mlp".into(),WeightStorage::Mxfp8),
        ("model.language_model.layers.1.self_attn.q_a_proj".into(),WeightStorage::Fp8Block {block:[16,16]}),
        ("model.language_model.layers.1.mlp.experts".into(),WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:true}),
        ("model.language_model.layers.2.mlp.experts".into(),WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:false}),
    ])};d
}
fn fixture_bytes()->Vec<u8> {
    let mut b=(FIXTURE.len() as u64).to_le_bytes().to_vec(); b.extend(FIXTURE);
    // The literal fixture records its byte ranges. Payload is synthetic and bounded.
    let j=engine_model::json::parse(FIXTURE,engine_model::json::Limits::default()).unwrap();
    let end=j.object().unwrap().values().map(|v|v.object().unwrap()["data_offsets"].array().unwrap()[1].u64().unwrap()).max().unwrap();
    b.resize(b.len()+end as usize,0);b
}
fn tensors()->BTreeMap<String,Tensor> {Header::read(&mut Cursor::new(fixture_bytes()),Limits::default()).unwrap().tensors}
fn payload(c:&ComponentView)->Result<Vec<u8>> {
    let mut b=vec![0;c.bytes as usize];
    if c.dtype==Dtype::F32 && (c.name.contains(".weight_scale") || c.name.ends_with(".input_scale")) {
        for v in b.as_chunks_mut::<4>().0 {v.copy_from_slice(&1f32.to_le_bytes());}
    } Ok(b)
}
#[test]
fn complete_literal_inventory_is_admitted_without_execution() {
    let d=description(); let t=tensors(); let a=d.admit_headers(&t).unwrap();
    assert_eq!(a.weights().len(),d.required_weights().unwrap().len());
    assert_eq!(a.stored_bytes,t.values().map(Tensor::byte_len).sum::<u64>());
    for view in a.weights().values() {let u=view.upload(payload).unwrap();assert_eq!(u.shape,view.logical().shape);}
    assert!(d.require_resident_service().unwrap_err().to_string().contains("complete resident decoder is absent"));
    assert!(ConfigPlan::Deferred(d).into_model_plan().is_err());
}
#[test]
fn each_weight_and_scale_component_is_required() {
    let d=description();let t=tensors();
    for name in t.keys() {let mut bad=t.clone();bad.remove(name);assert!(d.admit_headers(&bad).is_err(),"missing {name}");}
}
#[test]
fn every_component_checks_its_dtype() {
    let d=description();let t=tensors();
    for name in t.keys() {let mut bad=t.clone();let v=bad.get_mut(name).unwrap();
        v.dtype=match v.dtype {Dtype::F32|Dtype::U32=>Dtype::I32,Dtype::Bf16|Dtype::U16=>Dtype::F16,_=>Dtype::I8};
        assert!(d.admit_headers(&bad).is_err(),"dtype {name}");}
}
#[test]
fn every_component_checks_its_shape_and_rank() {
    let d=description();let t=tensors();
    for name in t.keys() {let mut bad=t.clone();let v=bad.get_mut(name).unwrap();v.shape.push(1);
        assert!(d.admit_headers(&bad).is_err(),"rank {name}");}
    let conv="model.language_model.layers.0.self_attn.q_conv1d.weight";
    let mut bad=t.clone();bad.get_mut(conv).unwrap().shape=vec![64,3,1];assert!(d.admit_headers(&bad).is_err());
    let scale="model.language_model.layers.1.self_attn.q_a_proj.weight_scale";
    let mut bad=t.clone();bad.get_mut(scale).unwrap().shape=vec![2,4];assert!(d.admit_headers(&bad).is_err());
    // Stored NVFP4 global and activation scales are rank-zero, not one-element vectors.
    for suffix in ["weight_scale_2","input_scale"] {
        let name=format!("model.language_model.layers.1.mlp.experts.0.gate_proj.{suffix}");
        let mut bad=t.clone();bad.get_mut(&name).unwrap().shape=vec![1];assert!(d.admit_headers(&bad).is_err());
    }
}
#[test]
fn byte_extents_overflow_and_unexpected_text_are_refused() {
    let d=description();let t=tensors();let name="lm_head.weight";
    for mode in 0..3 {let mut bad=t.clone();let v=bad.get_mut(name).unwrap();match mode {
        0=>{v.start=1;v.end=0},1=>v.end+=1,_=>v.shape=vec![u64::MAX,2],
    };assert!(d.admit_headers(&bad).is_err());}
    for name in ["model.language_model.unknown.weight","lm_head.weight_scale_2","model.language_model.layers.1.mlp.experts.2.up_proj.weight"] {
        let mut bad=t.clone();bad.insert(name.into(),Tensor {dtype:Dtype::F32,shape:vec![],start:0,end:4});assert!(d.admit_headers(&bad).is_err());
    }
    let mut bad=t.clone();bad.insert("model.visual.rank".into(),Tensor {dtype:Dtype::U8,shape:vec![1;MAX_TENSOR_RANK+1],start:0,end:1});
    assert!(d.admit_headers(&bad).is_err());
    let mut bad=t.clone();for i in 0..2 {bad.insert(format!("model.visual.large{i}"),Tensor {dtype:Dtype::U8,shape:vec![u64::MAX/2],start:0,end:u64::MAX/2});}
    assert!(d.admit_headers(&bad).is_err());
    let mut bad=d;bad.hidden=u64::MAX;assert!(bad.required_weights().is_err());
}
#[test]
fn quantization_geometry_and_ambiguous_draft_data_are_refused() {
    let t=tensors();let d=description();
    for storage in [WeightStorage::Mxfp8,WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:true},WeightStorage::Fp8Block {block:[0,16]}] {
        let mut bad=d.clone();bad.storage.layers.insert("lm_head".into(),storage);bad.hidden=63;
        assert!(bad.admit_headers(&t).is_err());
    }
    let mut bad=d.clone();bad.declared_draft_layers=u64::MAX;assert!(bad.required_weights().is_err());
    let mut bad=d.clone();bad.storage.layers.insert("model.language_model.layers.0".into(),WeightStorage::Mxfp8);
    assert!(bad.required_weights().is_err());
    let mut bad=d.clone();let mut layer=bad.layers[1];let TokenMixer::LatentAttention(ref mut a)=layer.mixer else {panic!()};a.query_rank+=1;
    bad.layers.push(layer);assert!(bad.required_weights().is_err());
    let mut bad=d;bad.layers.pop();assert!(bad.required_weights().is_err());
}
#[test]
fn upload_checks_payload_extent_scalar_values_and_compressed_scales() {
    let a=description().admit_headers(&tensors()).unwrap();
    let v=&a.weights()["model.language_model.layers.1.mlp.experts.0.gate_proj.weight"];
    let u=v.upload(payload).unwrap();assert_eq!(u.scales,vec![0;128]);assert_eq!(u.global_scale,Some(1.0));
    assert!(v.upload(|c| {let mut b=payload(c)?;b.push(0);Ok(b)}).is_err());
    for suffix in ["weight_scale_2","input_scale","csf32_bases","csf32_offsets","csf32_page_bases"] {
        assert!(v.upload(|c| {let mut b=payload(c)?;if c.name.ends_with(suffix) {if c.dtype==Dtype::F32 {b.copy_from_slice(&f32::NAN.to_le_bytes());} else {b[0]=255;}}Ok(b)}).is_err(),"{suffix}");
    }
    for scalar in [0f32,-1.0,f32::INFINITY,f32::NAN] {
        assert!(v.upload(|c| {let mut b=payload(c)?;if c.name.ends_with(".weight_scale_2") {b.copy_from_slice(&scalar.to_le_bytes());}Ok(b)}).is_err());
    }
    let v=&a.weights()["model.language_model.layers.1.self_attn.q_a_proj.weight"];
    assert!(v.upload(|c| {let mut b=payload(c)?;if c.name.ends_with(".weight_scale") {b[..4].copy_from_slice(&f32::NAN.to_le_bytes());}Ok(b)}).is_err());
    let v=&a.weights()["model.language_model.layers.2.mlp.experts.0.gate_proj.weight"];
    assert!(v.upload(|c| {let mut b=payload(c)?;if c.name.ends_with(".weight_scale") {b[0]=127;}Ok(b)}).is_err());
    let v=&a.weights()["lm_head.weight"];
    assert!(v.upload(|c| {let mut b=payload(c)?;if c.name.ends_with(".weight_scale") {b[0]=255;}Ok(b)}).is_err());
}
#[test]
fn byte_views_keep_global_block_origins_and_storage_strides() {
    let a=description().admit_headers(&tensors()).unwrap();
    for name in ["lm_head.weight","model.language_model.layers.1.self_attn.q_a_proj.weight",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight","model.language_model.embed_tokens.weight"] {
        let u=a.weights()[name].upload(payload).unwrap();let v=u.matrix_view([1,18],[2,16]).unwrap();
        assert_eq!(v.origin,[1,18]);assert_eq!(v.shape,[2,16]);assert_eq!(v.scales,u.scales);
        assert_eq!(v.scale_origin,[1/v.scale_block[0],18/v.scale_block[1]]);
        let stride=if matches!(u.storage,WeightStorage::Nvfp4 {..}) {u.shape[1]/2} else {u.shape[1]*u.dtype.bytes()};
        assert_eq!(v.row_stride,stride);
        assert!(u.matrix_view([u.shape[0],0],[1,2]).is_err());assert!(u.matrix_view([u64::MAX,0],[2,2]).is_err());
        assert!(u.matrix_view([0,0],[0,2]).is_err());
        if matches!(u.storage,WeightStorage::Nvfp4 {..}) {assert!(u.matrix_view([0,1],[1,2]).is_err());}
    }
}
static NEXT: AtomicU64=AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {fn new()->Self {let p=std::env::temp_dir().join(format!("vade-header-admission-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));std::fs::create_dir(&p).unwrap();Self(p)}}
impl Drop for Temp {fn drop(&mut self) {let _=std::fs::remove_dir_all(&self.0);}}
fn index(t:&BTreeMap<String,Tensor>,size:u64)->Vec<u8> {
    format!("{{\"metadata\":{{\"total_size\":{size}}},\"weight_map\":{{{}}}}}",t.keys().map(|n|format!("{n:?}:\"weights.safetensors\"")).collect::<Vec<_>>().join(",")).into_bytes()
}
#[test]
fn file_admission_reconciles_index_shards_and_uses_retained_descriptors() {
    let dir=Temp::new();let d=description();let t=tensors();let size=t.values().map(Tensor::byte_len).sum();
    std::fs::write(dir.0.join("weights.safetensors"),fixture_bytes()).unwrap();let good=index(&t,size);
    let a=d.open_headers(&dir.0,&good).unwrap();assert_eq!(a.headers().stored_bytes,size);
    let name="model.language_model.embed_tokens.weight";
    let bytes=a.upload(name).unwrap();assert_eq!(bytes.dtype,Dtype::Bf16);assert!(bytes.weight.iter().all(|&v|v==0));
    assert!(a.upload("unknown").is_err());assert!(d.open_headers(&dir.0,&index(&t,size+1)).is_err());
    let mut missing=t.clone();missing.remove(name);assert!(d.open_headers(&dir.0,&index(&missing,size)).is_err());
    let mut extra=t;extra.insert("model.visual.absent".into(),Tensor {dtype:Dtype::U8,shape:vec![1],start:0,end:1});
    assert!(d.open_headers(&dir.0,&index(&extra,size)).is_err());
    // Replacing a pathname cannot change the file that supplied admission.
    std::fs::rename(dir.0.join("weights.safetensors"),dir.0.join("original.safetensors")).unwrap();
    std::fs::write(dir.0.join("weights.safetensors"),b"bad").unwrap();assert_eq!(a.upload(name).unwrap().weight,bytes.weight);
}
#[test]
fn unused_vision_bytes_are_counted_without_text_upload_views() {
    let d=description();let mut t=tensors();let before=d.admit_headers(&t).unwrap();
    t.insert("model.visual.unused".into(),Tensor {dtype:Dtype::U8,shape:vec![7],start:0,end:7});
    let after=d.admit_headers(&t).unwrap();assert_eq!(after.stored_bytes,before.stored_bytes+7);
    assert_eq!(after.weights().keys().collect::<Vec<_>>(),before.weights().keys().collect::<Vec<_>>());
}
#[cfg(unix)]
#[test]
fn shard_paths_cannot_escape_the_checkpoint_root() {
    let dir=Temp::new();let outside=Temp::new();let d=description();let t=tensors();let size=t.values().map(Tensor::byte_len).sum();
    std::fs::write(outside.0.join("weights.safetensors"),fixture_bytes()).unwrap();
    std::os::unix::fs::symlink(outside.0.join("weights.safetensors"),dir.0.join("weights.safetensors")).unwrap();
    assert!(d.open_headers(&dir.0,&index(&t,size)).is_err());
    let bad=String::from_utf8(index(&t,size)).unwrap().replace("weights.safetensors","../weights.safetensors");
    assert!(d.open_headers(&dir.0,bad.as_bytes()).is_err());
}
