//! Synthetic MTP headers. GLM names test rejection, not a verified mapping.
use super::*;
use super::tests::{hybrid_headers, synthetic};
use crate::safetensors::Dtype;

fn hybrid_config() -> String {
    synthetic(128,2,4,2,4,2,4,2,256).replace("qwen3_5_moe","qwen3_5")
        .replace("Qwen3_5Moe","Qwen3_5").replace("moe_intermediate_size","intermediate_size")
        .replace("\"mtp_num_hidden_layers\":0","\"mtp_num_hidden_layers\":1")
        .replace("[\"lm_head\"]","[\"lm_head\",\"mtp.fc\"]")
}
fn flash_config() -> String {
    synthetic(128,2,4,2,4,2,4,2,256).replace("qwen3_5_moe","qwen4_exp")
        .replace("Qwen3_5Moe","Qwen4Exp").replace("swish","sigmoid")
        .replace("\"fmt\":\"e4m3\"","\"weight_per_tensor\":false,\"act_per_tensor\":false")
        .replace("\"mtp_num_hidden_layers\":0",r#""hc_count":2,"hc_lowrank":8,"indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":64,"indexer_budget":16,"indexer_compress_ratio":2,"ple_layer_ids":[2],"heads_per_ngram":2,"ngram_size":3,"ple_embed_dim":128,"ple_conv_kernel_size":4,"split_ngram_parts":2,"mtp_use_dedicated_embeddings":false,"mtp_num_hidden_layers":1,"mtp":{"num_hidden_layers":1,"hybrid":true,"mtp_use_hidden_state_from_layer":null,"rope_theta":10000000,"layer_types":["full_attention"]}"#)
}
// This table is independent of tensors::expected. All matrix weights in the
// Flash fixture use FP8. The hybrid fixture keeps its concat projection in BF16.
fn put(out: &mut BTreeMap<String, TensorEntry>, name: &str, shape: &[u64], layer: Option<u32>, part: Part, fp8: bool) {
    let name = format!("mtp.{name}");
    let role = Role { place: Place::Draft { layer }, part, store: Store::Weight };
    if fp8 {
        let scale = format!("{name}_scale_inv");
        out.insert(scale.clone(), TensorEntry { name: scale, dtype: Dtype::Bf16,
            shape: vec![shape[0].div_ceil(128),shape[1].div_ceil(128)], role: Role { store: Store::BlockScale, ..role } });
    }
    out.insert(name.clone(), TensorEntry { name, dtype: if fp8 { Dtype::F8E4M3 } else { Dtype::Bf16 }, shape: shape.to_vec(), role });
}
fn attention(out: &mut BTreeMap<String, TensorEntry>) {
    for (suffix,shape,part) in [
        ("q_proj",vec![2048,128],Part::Query), ("k_proj",vec![512,128],Part::Key),
        ("v_proj",vec![512,128],Part::Value), ("o_proj",vec![128,1024],Part::Output),
        ("q_norm",vec![256],Part::QueryNorm), ("k_norm",vec![256],Part::KeyNorm),
    ] { put(out,&format!("layers.0.self_attn.{suffix}.weight"),&shape,Some(0),part,shape.len()==2); }
}
fn projections(out: &mut BTreeMap<String, TensorEntry>, prefix: &str, part: impl Fn(Proj) -> Part) {
    for (suffix,shape,proj) in [("gate_proj",[256,128],Proj::Gate),("up_proj",[256,128],Proj::Up),("down_proj",[128,256],Proj::Down)] {
        put(out,&format!("layers.0.mlp.{prefix}{suffix}.weight"),&shape,Some(0),part(proj),true);
    }
}
fn hybrid_draft() -> BTreeMap<String, TensorEntry> {
    let mut out = BTreeMap::new();
    put(&mut out,"fc.weight",&[128,256],None,Part::Whole,false);
    for suffix in ["norm","pre_fc_norm_embedding","pre_fc_norm_hidden"] { put(&mut out,&format!("{suffix}.weight"),&[128],None,Part::Whole,false); }
    for suffix in ["input_layernorm","post_attention_layernorm"] { put(&mut out,&format!("layers.0.{suffix}.weight"),&[128],Some(0),Part::Whole,false); }
    attention(&mut out); projections(&mut out,"",Part::Dense); out
}
fn flash_draft() -> BTreeMap<String, TensorEntry> {
    let mut out = BTreeMap::new();
    for (prefix,layer) in [("hyper_connection_mixer",None),("layers.0.attn_hyper_connection",Some(0)),("layers.0.mlp_hyper_connection",Some(0))] {
        for (suffix,shape) in [("hc_norm",vec![256]),("input_mix_weight_down",vec![8,256]),("input_mix_weight_up",vec![256,8])] {
            put(&mut out,&format!("{prefix}.{suffix}.weight"),&shape,layer,Part::Whole,shape.len()==2);
        }
        if layer.is_some() { put(&mut out,&format!("{prefix}.block_inject_weight.weight"),&[2,256],layer,Part::Whole,true); }
    }
    for (suffix,shape) in [("pre_fc_norm_embedding",vec![128]),("pre_fc_norm_hidden",vec![256]),("fc_embedding",vec![128,128]),("fc_hidden",vec![128,128])] {
        put(&mut out,&format!("{suffix}.weight"),&shape,None,Part::Whole,shape.len()==2);
    }
    attention(&mut out);
    for (suffix,shape,part) in [("index_qk_proj",vec![192,128],Part::Index),("q_layernorm",vec![64],Part::IndexNorm),("k_layernorm",vec![64],Part::IndexNorm)] {
        put(&mut out,&format!("layers.0.self_attn.indexer.{suffix}.weight"),&shape,Some(0),part,shape.len()==2);
    }
    put(&mut out,"layers.0.mlp.gate.weight",&[4,128],Some(0),Part::Whole,true);
    put(&mut out,"layers.0.mlp.shared_expert_gate.weight",&[1,128],Some(0),Part::Whole,true);
    projections(&mut out,"shared_expert.",Part::Shared);
    for id in 0..4 { projections(&mut out,&format!("experts.{id}."),|proj| Part::Expert { id: Some(id),proj }); }
    out
}
fn headers(entries: BTreeMap<String, TensorEntry>) -> BTreeMap<String, Tensor> {
    let mut offset = 0;
    entries.into_iter().map(|(name,e)| {
        let start = offset; offset += e.shape.iter().product::<u64>()*e.dtype.bytes();
        (name,Tensor { dtype: e.dtype,shape: e.shape,start,end: offset })
    }).collect()
}
fn draft(plan: &ModelPlan) -> BTreeMap<String, TensorEntry> {
    plan.shape.tensors.iter().filter(|t| matches!(t.role.place,Place::Draft { .. })).map(|t| (t.name.clone(),t.clone())).collect()
}

#[test]
fn qwen_hybrid_mtp_header_keeps_full_draft_inventory() {
    let config = hybrid_config(); let input = hybrid_headers(&config);
    let plan = ModelPlan::from_config(config.as_bytes()).unwrap().with_headers(&input).unwrap();
    assert_eq!(draft(&plan),hybrid_draft());
    crate::qwen3_5::plan::validate_typed_mtp(&plan).unwrap();
    assert_eq!(plan.clone().with_headers(&input).unwrap(),plan);
}

#[test]
fn qwen_flash_mtp_header_keeps_full_draft_inventory() {
    let mut plan = ModelPlan::from_config(flash_config().as_bytes()).unwrap();
    plan.shape.lexical.as_mut().unwrap().part_rows = 7;
    // Only the unrelated decoder headers come from the planner. All draft
    // names, shapes, dtypes and roles come from the independent table above.
    let mut entries = tensors::expected(&plan).unwrap();
    entries.retain(|_,e| !matches!(e.role.place,Place::Draft { .. }));
    entries.extend(flash_draft());
    let input = headers(entries); let plan = plan.with_headers(&input).unwrap();
    assert_eq!(draft(&plan),flash_draft());
    assert_eq!(plan.clone().with_headers(&input).unwrap(),plan);
}

#[test]
fn appended_glm_names_do_not_become_qwen_mtp_aliases() {
    // Use a supported Qwen config. This tests rejection only. It does not
    // admit a GLM config or establish any GLM tensor shape.
    let config = hybrid_config(); let input = hybrid_headers(&config);
    let plan = ModelPlan::from_config(config.as_bytes()).unwrap();
    let appended = plan.shape.layers.len();
    for suffix in ["eh_proj.weight","enorm.weight","hnorm.weight","shared_head.norm.weight","input_layernorm.weight","post_attention_layernorm.weight"] {
        let name = format!("model.language_model.layers.{appended}.{suffix}");
        let mut extra = input.clone();
        // The shape is arbitrary. This is a name-rejection test, not a claim
        // about a GLM checkpoint shape or a positive mapping.
        extra.insert(name.clone(),Tensor { dtype: Dtype::Bf16,shape: vec![1],start: 0,end: 2 });
        assert_eq!(plan.clone().with_headers(&extra).unwrap_err(),PlanError::UnexpectedTensor { name });
    }
    let mut appended_body = input;
    let names: Vec<_> = appended_body.keys().filter(|n| n.starts_with("mtp.layers.0.")).cloned().collect();
    for name in names {
        let tensor = appended_body.remove(&name).unwrap();
        appended_body.insert(name.replacen("mtp.layers.0.",&format!("model.language_model.layers.{appended}."),1),tensor);
    }
    assert!(matches!(plan.with_headers(&appended_body),Err(PlanError::MissingTensor { name }) if name.starts_with("mtp.layers.0.")));
}
