//! Synthetic checkpoint for the host stack probe. It is not model evidence.
use super::{Result,invalid};
use engine_model::{json::{self,Value},plan::{ConfigPlan,layer::{LayerPlan,Topology,Buffer}}};
use std::{fs,path::Path,collections::BTreeMap};

// The saved config uses plain ASCII names. This fixture writer needs no general
// JSON serializer and refuses characters that would require an escape.
fn render(value: &Value) -> Result<String> {
    Ok(match value {
        Value::Null=>"null".into(),Value::Bool(v)=>v.to_string(),Value::Number(v)=>v.clone(),
        Value::String(s)=>{if s.chars().any(|c| c.is_control() || c=='"' || c=='\\') {return Err(invalid("unsupported fixture string"));} format!("\"{s}\"")},
        Value::Array(v)=>format!("[{}]",v.iter().map(render).collect::<Result<Vec<_>>>()?.join(",")),
        Value::Object(v)=>format!("{{{}}}",v.iter().map(|(k,v)| Ok(format!("{}:{}",render(&Value::String(k.clone()))?,render(v)?)))
            .collect::<Result<Vec<_>>>()?.join(",")),
    })
}
fn config() -> Result<Vec<u8>> {
    let source=json::parse(include_bytes!("../../tests/fixtures/glm/config.json"),Default::default())?;
    let mut text=source.object()?["text_config"].object()?.clone();
    for (key,value) in [("hidden_size",32),("vocab_size",5),("max_position_embeddings",12),("num_hidden_layers",4),
        ("hc_mult",2),("hc_sinkhorn_iters",4),("num_attention_heads",2),("q_lora_rank",4),("kv_lora_rank",4),
        ("qk_nope_head_dim",3),("qk_head_dim",3),("v_head_dim",2),("index_n_heads",2),("index_head_dim",3),
        ("index_topk",4),("index_kpool",2),("n_routed_experts",4),("num_experts_per_tok",2),("moe_intermediate_size",6),
        ("intermediate_size",6),("n_group",2),("topk_group",1),("num_nextn_predict_layers",0),("pad_token_id",0)] {
        text.insert(key.into(),Value::Number(value.to_string()));
    }
    text.insert("eos_token_id".into(),Value::Number("4".into()));
    text.insert("tie_word_embeddings".into(),Value::Bool(false));
    for (key,values) in [("layer_types",vec!["linear_attention","deepseek_sparse_attention","linear_attention","deepseek_sparse_attention"]),
        ("mlp_layer_types",vec!["dense","sparse","sparse","dense"]),("indexer_types",vec!["full";4])] {
        text.insert(key.into(),Value::Array(values.iter().map(|v| Value::String((*v).into())).collect()));
    }
    let mut linear=text["linear_attn_config"].object()?.clone();
    for (key,value) in [("num_heads",2),("head_dim",4),("short_conv_kernel_size",4)] {linear.insert(key.into(),Value::Number(value.to_string()));}
    for (key,values) in [("kda_layers",vec![0,2]),("full_attn_layers",vec![1,3])] {
        linear.insert(key.into(),Value::Array(values.iter().map(|v| Value::Number(v.to_string())).collect()));
    }
    text.insert("linear_attn_config".into(),Value::Object(linear));
    let quant=json::parse(br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"lm_head":{"quant_algo":"MXFP8","group_size":32}}}"#,Default::default())?;
    text.insert("quantization_config".into(),quant); Ok(render(&Value::Object(text))?.into_bytes())
}
pub fn create(root: &Path) -> Result<()> {
    fs::create_dir(root)?; let config=config()?;
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&config).map_err(|e| invalid(e.to_string()))? else {return Err(invalid("fixture description is not deferred"));};
    let mut tensors=BTreeMap::new(); let mut seed=29u64;
    for layer in 0..d.layers.len() {for weight in LayerPlan::new(&d,layer,1,Topology {parts:1,rank:0})?.steps.iter().flat_map(|s| &s.weights) {
        let count=Buffer::new(&weight.name,&weight.shape).elements()?;
        let bytes=(0..count).flat_map(|_| {seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value=((seed>>40) as i32-8388608) as f32*(0.125/8388608.0);
            ((value.to_bits()>>16) as u16).to_le_bytes()}).collect::<Vec<_>>();
        tensors.insert(weight.name.clone(),("BF16",weight.shape.clone(),bytes));
    }}
    tensors.insert("model.language_model.embed_tokens.weight".into(),("BF16",vec![5,32],(0..160)
        .flat_map(|j| (((j%17) as f32/32.0-0.25).to_bits().wrapping_shr(16) as u16).to_le_bytes()).collect()));
    tensors.insert("model.language_model.norm.weight".into(),("F32",vec![32],(0..32).flat_map(|_| 1.0f32.to_le_bytes()).collect()));
    tensors.insert("lm_head.weight".into(),("F8_E4M3",vec![5,32],(0..160).map(|j| (j%96) as u8).collect()));
    tensors.insert("lm_head.weight_scale".into(),("U8",vec![5,1],vec![127;5]));
    let mut header=Vec::new(); let mut data=Vec::new(); let mut index=Vec::new();
    for (name,(dtype,shape,bytes)) in tensors {
        let start=data.len(); data.extend(bytes);
        header.push(format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",data.len()));
        index.push(format!("\"{name}\":\"weights.safetensors\""));
    }
    let header=format!("{{{}}}",header.join(",")); let mut encoded=(header.len() as u64).to_le_bytes().to_vec(); encoded.extend(header.as_bytes()); encoded.extend(&data);
    fs::write(root.join("weights.safetensors"),encoded)?;
    fs::write(root.join("model.safetensors.index.json"),format!("{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{{}}}}}",data.len(),index.join(",")))?;
    fs::write(root.join("config.json"),config)?; Ok(())
}
