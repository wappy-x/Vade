//! File-backed host checks. All payloads are synthetic. No model forward runs.
use engine_model::{plan::{ConfigPlan,TextDescription,description::*,storage::*,admission::CheckpointAdmission,
    glm_index::{TensorRole,KdaTensor}},safetensors::{Dtype,Header,Limits},shapes::Ffn};
use spark_serve::text_checkpoint::CheckpointTextEdges;
use std::{collections::BTreeMap,path::PathBuf,sync::atomic::{AtomicU64,Ordering}};

static NEXT:AtomicU64 = AtomicU64::new(0);
struct Fixture {root:PathBuf,description:TextDescription,index:Vec<u8>,payloads:BTreeMap<String,Vec<u8>>}
impl Drop for Fixture {fn drop(&mut self) {let _ = std::fs::remove_dir_all(&self.root);}}
fn fixture(norm:Dtype) -> Fixture {
    let ConfigPlan::Deferred(mut description) = ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    description.hidden=32; description.vocab=17; description.max_positions=64; description.eos_tokens=vec![3,5,6];
    description.residual.streams=3; description.declared_draft_layers=0;
    description.layers=vec![DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:1,head_dim:32,conv:3,gate_lower_bound:-5.0}),
        ffn:Ffn::Dense {width:32}}];
    description.storage=StorageMap {excluded:vec![],layers:BTreeMap::from([("lm_head".into(),WeightStorage::Mxfp8)])};
    let root=std::env::temp_dir().join(format!("vade-text-checkpoint-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
    std::fs::create_dir(&root).unwrap();
    let mut tensors=BTreeMap::new();
    for (name,weight) in description.required_weights().unwrap() {
        let role=description.bind_tensor(&name).unwrap().operation;
        let mut shape=weight.shape;
        let dtype=match role {TensorRole::Embedding=>Dtype::Bf16,TensorRole::Head=>Dtype::F8E4M3,TensorRole::FinalNorm=>norm,_=>Dtype::F32};
        if matches!(role,TensorRole::Kda(KdaTensor::QueryConv|KdaTensor::KeyConv|KdaTensor::ValueConv)) {shape.insert(1,1);}
        if role==TensorRole::Head {tensors.insert("lm_head.weight_scale".to_string(),(Dtype::U8,vec![17,1]));}
        tensors.insert(name,(dtype,shape));
    }
    let mut payloads=BTreeMap::new(); let mut parts:[Vec<u8>;2]=[Vec::new(),Vec::new()]; let mut headers=[Vec::new(),Vec::new()]; let mut locations=Vec::new();
    // Alternating files and a reverse name order separate file offsets from
    // arena offsets. An odd scale length also forces norm alignment padding.
    for (i,(name,(dtype,shape))) in tensors.into_iter().rev().enumerate() {
        let count=shape.iter().product::<u64>() as usize; let mut data=Vec::new();
        for n in 0..count {match dtype {
            Dtype::Bf16=>data.extend((0x3f00u16+(n%127) as u16).to_le_bytes()),
            Dtype::F32=>data.extend((0x3f800001u32+(n%1024) as u32).to_le_bytes()),
            Dtype::F8E4M3=>data.push((n%127) as u8),Dtype::U8=>data.push(120+(n%8) as u8),_=>unreachable!(),
        }}
        let file=i%2; let start=parts[file].len(); parts[file].extend(&data);
        let dtype=match dtype {Dtype::Bf16=>"BF16",Dtype::F32=>"F32",Dtype::F8E4M3=>"F8_E4M3",Dtype::U8=>"U8",_=>unreachable!()};
        headers[file].push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",parts[file].len()));
        locations.push(format!("{name:?}:\"part-{file}.safetensors\"")); payloads.insert(name,data);
    }
    let total:usize=parts.iter().map(Vec::len).sum();
    for (i,(header,data)) in headers.into_iter().zip(parts).enumerate() {
        let header=format!("{{{}}}",header.join(","));
        let mut file=(header.len() as u64).to_le_bytes().to_vec(); file.extend(header.as_bytes()); file.extend(data);
        std::fs::write(root.join(format!("part-{i}.safetensors")),file).unwrap();
    }
    let index=format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{}}}}}",locations.join(",")).into_bytes();
    Fixture {root,description,index,payloads}
}
fn open(f:&Fixture) -> CheckpointAdmission {f.description.open_headers(&f.root,&f.index).unwrap()}
fn expected(f:&Fixture) -> Vec<u8> {
    let mut bytes=Vec::new();
    bytes.extend(&f.payloads["model.language_model.embed_tokens.weight"]);
    assert_eq!(bytes.len()%16,0);
    bytes.extend(&f.payloads["lm_head.weight"]); bytes.extend(&f.payloads["lm_head.weight_scale"]);
    let width=if f.payloads["model.language_model.norm.weight"].len()==32*2 {2} else {4};
    bytes.resize(bytes.len().div_ceil(width)*width,0);
    bytes.extend(&f.payloads["model.language_model.norm.weight"]); bytes
}
#[test]
fn retained_files_preserve_every_stored_byte_and_norm_dtype() {
    for dtype in [Dtype::Bf16,Dtype::F32] {
        let f=fixture(dtype); let checkpoint=open(&f); let bound=CheckpointTextEdges::new(&checkpoint).unwrap();
        bound.validate_head_scales().unwrap();
        assert_eq!(checkpoint.description(),&f.description); assert_eq!(bound.edges().norm.tensor.dtype,dtype);
        let expected=expected(&f); assert_eq!(bound.arena_bytes(),expected.len());
        let mut full=vec![255;expected.len()]; bound.read_range(0,&mut full).unwrap(); assert_eq!(full,expected);
        // Check every cut through tensors and padding, including unaligned reads.
        for chunk in [1,3,17,64,257] {
            let mut read=Vec::new(); let mut offset=0;
            while offset<expected.len() {let mut part=vec![255;chunk.min(expected.len()-offset)];
                bound.read_range(offset,&mut part).unwrap(); offset+=part.len(); read.extend(part);}
            assert_eq!(read,expected);
        }
        let mut by_files=vec![0;expected.len()];
        use std::os::unix::fs::FileExt;
        for range in bound.stored_copies() {
            range.file.read_exact_at(&mut by_files[range.destination..range.destination+range.bytes],range.file_offset).unwrap();
        }
        assert_eq!(by_files,expected);
        // Path replacement cannot replace the descriptors used for admission.
        for i in 0..2 {let path=f.root.join(format!("part-{i}.safetensors"));
            std::fs::rename(&path,f.root.join(format!("original-{i}.safetensors"))).unwrap();std::fs::write(path,b"invalid").unwrap();}
        bound.read_range(0,&mut full).unwrap(); assert_eq!(full,expected);
        let mut bad=[9;2]; assert!(bound.read_range(usize::MAX,&mut bad).is_err()); assert_eq!(bad,[0;2]);
        assert!(bound.read_range(expected.len()-1,&mut bad).is_err()); assert_eq!(bad,[0;2]);
        assert!(bound.read_range(expected.len(),&mut []).is_ok());
    }
}
#[test]
fn failed_payload_read_clears_partial_output() {
    let f=fixture(Dtype::F32); let checkpoint=open(&f); let bound=CheckpointTextEdges::new(&checkpoint).unwrap();
    // Violate the immutable-file contract deliberately. The host reader must
    // report the error and discard all bytes in the current caller buffer.
    for i in 0..2 {std::fs::OpenOptions::new().write(true).open(f.root.join(format!("part-{i}.safetensors"))).unwrap().set_len(0).unwrap();}
    let mut output=vec![99;bound.arena_bytes()]; assert!(bound.read_range(0,&mut output).is_err());
    assert!(output.iter().all(|&b|b==0));
}
#[test]
fn reserved_scale_byte_is_refused_before_device_allocation() {
    let f=fixture(Dtype::F32); let checkpoint=open(&f); let bound=CheckpointTextEdges::new(&checkpoint).unwrap();
    let name="lm_head.weight_scale";
    let retained=checkpoint.component_file(name).unwrap();
    let offset=retained.header().data_start+retained.header().tensors[name].start;
    use std::os::unix::fs::FileExt;
    for i in 0..2 {let path=f.root.join(format!("part-{i}.safetensors"));
        if Header::open(&path,Limits::default()).unwrap().tensors.contains_key(name) {
            std::fs::OpenOptions::new().write(true).open(path).unwrap().write_all_at(&[255],offset).unwrap();
        }
    }
    assert_eq!(bound.validate_head_scales().unwrap_err(),"invalid MXFP8 scale byte");
}
#[test]
fn edge_binding_uses_admitted_description_and_original_offsets() {
    let mut f=fixture(Dtype::F32); let checkpoint=open(&f);
    f.description.hidden=64; f.description.epsilon=1.0; f.description.eos_tokens=vec![0];
    let bound=CheckpointTextEdges::new(&checkpoint).unwrap();
    assert_eq!(bound.layout().shape.hidden(),32); assert_eq!(bound.edges().eos_tokens,[3,5,6]);
    assert_ne!(bound.edges().epsilon,f.description.epsilon);
    assert!(checkpoint.component_file("model.visual.weight").is_err()); assert!(checkpoint.component_file("unknown").is_err());
    // Header admission already refuses a scalar that underflows the GPU ABI.
    let mut f=fixture(Dtype::F32); f.description.epsilon=f64::MIN_POSITIVE;
    assert!(f.description.open_headers(&f.root,&f.index).is_err());
    // The retained header view is the exact source offset, not an arena offset.
    let f=fixture(Dtype::Bf16); let checkpoint=open(&f);
    let retained=checkpoint.component_file("lm_head.weight").unwrap();
    let header=Header::read(&mut retained.file().try_clone().unwrap(),Limits::default()).unwrap();
    assert_eq!(header.tensors["lm_head.weight"],retained.header().tensors["lm_head.weight"]);
}

// This factory checks the service boundary, then refuses the absent decoder.
// It cannot return a program or produce model logits.
struct RefusingFactory(TextDescription);
enum NoProgram {}
impl spark_serve::text_program::TokenProgram for NoProgram {
    type Residual=();type Hidden=();
    fn embed(&mut self,_:u32,_:usize)->Result<(),String>{match *self {}}
    fn layer(&mut self,_:usize,_:&DecoderDescription,_:())->Result<(),String>{match *self {}}
    fn mean(&mut self,_:())->Result<(),String>{match *self {}}
    fn norm(&mut self,_:())->Result<(),String>{match *self {}}
    fn head(&mut self,_:())->Result<Vec<f32>,String>{match *self {}}
    fn commit(&mut self)->Result<(),String>{match *self {}}
    fn abort(&mut self)->Result<(),String>{match *self {}}
}
impl spark_serve::text_program::Factory for RefusingFactory {
    type Program<'a>=NoProgram;
    fn create(&self,description:&TextDescription,edges:&engine_model::plan::text_edges::TextEdges,context:usize)->Result<NoProgram,String>{
        assert_eq!(description,&self.0);assert_eq!(context,3);assert_eq!(edges.norm.tensor.dtype,Dtype::F32);
        Err("fixture has no decoder".into())
    }
}
struct Codec(Vec<u32>);
struct Decoder;
impl spark_serve::Decoder for Decoder {
    fn push(&mut self,_:u32)->Result<String,String>{Ok(String::new())}
    fn finish(&mut self)->Result<String,String>{Ok(String::new())}
}
impl spark_serve::Tokenizer for Codec {
    type Decoder<'a>=Decoder;
    fn encode(&self,_:&str)->Result<Vec<u32>,String>{Ok(vec![0])}
    fn decoder(&self)->Decoder{Decoder}
    fn has_token(&self,id:u32)->bool{id<17}
    fn checkpoint_eos_token_ids(&self)->Result<Option<&[u32]>,String>{Ok(Some(&self.0))}
    fn encode_chat(&self,_:&spark_serve::protocol::ChatInput,_:usize)->Result<Vec<u32>,String>{Ok(vec![0])}
}
#[test]
fn service_uses_checkpoint_metadata_without_claiming_decoder_readiness() {
    use spark_serve::{Engine,text_program::TextEngine};
    let f=fixture(Dtype::F32);let checkpoint=open(&f);
    let engine=TextEngine::from_checkpoint(RefusingFactory(f.description.clone()),&checkpoint,&Codec(vec![5,6,3]),3).unwrap();
    assert_eq!(engine.eos_token_ids(),[3,5,6]);assert_eq!(engine.context_limit(),3);assert_eq!(engine.vocab_size(),17);
    assert_eq!(engine.session().err().unwrap(),"fixture has no decoder");
    assert!(TextEngine::from_checkpoint(RefusingFactory(f.description.clone()),&checkpoint,&Codec(vec![3,5]),3).is_err());
    assert!(TextEngine::from_checkpoint(RefusingFactory(f.description.clone()),&checkpoint,&Codec(vec![3,5,6]),65).is_err());
}
