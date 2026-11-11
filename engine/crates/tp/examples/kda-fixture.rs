#![forbid(unsafe_code)]
use engine_model::{json::{self,Value},plan::ConfigPlan};
use std::{fs,path::Path};
#[allow(dead_code)]
#[path="../src/bin/resident-layer-check/fixture.rs"] mod fixture;
fn render(value:&Value)->String {
    match value {
        Value::Null=>"null".into(),Value::Bool(v)=>v.to_string(),Value::Number(v)=>v.clone(),
        Value::String(v)=>{assert!(!v.chars().any(|c|c.is_control() || c=='"' || c=='\\'));format!("{v:?}")},
        Value::Array(v)=>format!("[{}]",v.iter().map(render).collect::<Vec<_>>().join(",")),
        Value::Object(v)=>format!("{{{}}}",v.iter().map(|(k,v)|format!("{k:?}:{}",render(v))).collect::<Vec<_>>().join(",")),
    }
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();let model=Path::new(&args[1]);let out=Path::new(&args[2]);
    let source=json::parse(&fs::read(model.join("config.json"))?,Default::default())?;
    let object=source.object()?;let mut text=object.get("text_config").unwrap_or(&source).object()?.clone();
    for (key,value) in [("hidden_size",128),("max_position_embeddings",32768),("num_hidden_layers",2),
        ("first_k_dense_replace",2),("hc_mult",2),("hc_sinkhorn_iters",4),("num_attention_heads",64),("q_lora_rank",32),("kv_lora_rank",32),
        ("qk_nope_head_dim",256),("qk_head_dim",256),("v_head_dim",256),("index_n_heads",4),("index_head_dim",16),
        ("index_topk",2048),("index_kpool",4),("n_routed_experts",4),("num_experts_per_tok",2),
        ("moe_intermediate_size",64),("intermediate_size",128),("n_group",2),("topk_group",1),("num_nextn_predict_layers",0)] {
        text.insert(key.into(),Value::Number(value.to_string()));
    }
    text.insert("tie_word_embeddings".into(),Value::Bool(false));
    text.insert("quantization_config".into(),json::parse(br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"lm_head":{"quant_algo":"MXFP8","group_size":32}}}"#,Default::default())?);
    for (key,values) in [("layer_types",vec!["linear_attention","deepseek_sparse_attention"]),
        ("mlp_layer_types",vec!["dense","dense"]),("indexer_types",vec!["full","full"])] {
        text.insert(key.into(),Value::Array(values.into_iter().map(|s|Value::String(s.into())).collect()));
    }
    let mut linear=text["linear_attn_config"].object()?.clone();
    for (key,value) in [("num_heads",64),("head_dim",128),("short_conv_kernel_size",4)] {linear.insert(key.into(),Value::Number(value.to_string()));}
    for (key,value) in [("kda_layers",0),("full_attn_layers",1)] {linear.insert(key.into(),Value::Array(vec![Value::Number(value.to_string())]));}
    text.insert("linear_attn_config".into(),Value::Object(linear));
    let config=render(&Value::Object(text)).into_bytes();
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&config)? else {return Err("fixture must be deferred".into());};
    println!("SYNTHETIC_FIXTURE hidden={} vocab={} layers={} kda_heads=64 kda_dim=128 latent_heads=64 latent_dim=256 no_public_checkpoint_evidence=true",d.hidden,d.vocab,d.layers.len());
    fs::create_dir(out)?;
    let mut rows=Vec::new();let mut data=Vec::new();let mut index=Vec::new();
    let mut tensors=fixture::headers(&d);
    tensors.insert("lm_head.weight_scale".into(),engine_model::safetensors::Tensor {dtype:engine_model::safetensors::Dtype::U8,shape:vec![d.vocab,d.hidden/32],start:0,end:d.vocab*d.hidden/32});
    for (name,t) in tensors {

        let n=t.shape.iter().product::<u64>() as usize;
        let (dtype,payload)=if name=="lm_head.weight" {("F8_E4M3",(0..n).map(|i|16+((i*13+i/7)%32) as u8).collect())}
            else if name=="lm_head.weight_scale" {("U8",vec![127u8;n])}
            else {(match t.dtype {engine_model::safetensors::Dtype::Bf16=>"BF16",engine_model::safetensors::Dtype::F32=>"F32",_=>panic!()},fixture::payload(&name,&t))};
        let start=data.len();data.extend(payload);
        rows.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{:?},\"data_offsets\":[{start},{}]}}",t.shape,data.len()));
        index.push(format!("{name:?}:\"fixture.safetensors\""));
    }
    let mut header=format!("{{{}}}",rows.join(",")).into_bytes();while header.len()%8!=0 {header.push(b' ');}
    let total=data.len();let mut saved=(header.len() as u64).to_le_bytes().to_vec();saved.extend(header);saved.extend(data);
    fs::write(out.join("fixture.safetensors"),saved)?;
    fs::write(out.join("model.safetensors.index.json"),format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{}}}}}",index.join(",")))?;

    fs::write(out.join("config.json"),config)?;
    for file in ["tokenizer.json","tokenizer_config.json","chat_template.jinja","generation_config.json"] {fs::copy(model.join(file),out.join(file))?;}
    println!("FIXTURE_COMPLETE {}",out.display());Ok(())
}
