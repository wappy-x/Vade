use engine_model::{json::{self, Value}, plan::{ConfigPlan, ModelPlan, TextDescription,
    description::TokenMixer, glm_index::*, storage::{StorageMap, WeightStorage, ScaleEncoding}},
    shapes::{Ffn, Part, Place, Proj, Store}};
const CONFIG: &[u8] = include_bytes!("fixtures/glm/config.json");
fn description() -> TextDescription { let ConfigPlan::Deferred(p) = ConfigPlan::from_config(CONFIG).unwrap() else { panic!() }; p }
fn encode(v: &Value) -> String {
    match v { Value::Null => "null".into(),Value::Bool(v) => v.to_string(),Value::Number(v) => v.clone(),
        Value::String(s) => format!("{s:?}"),Value::Array(a) => format!("[{}]",a.iter().map(encode).collect::<Vec<_>>().join(",")),
        Value::Object(o) => format!("{{{}}}",o.iter().map(|(k,v)| format!("{k:?}:{}",encode(v))).collect::<Vec<_>>().join(",")) }
}
fn changed(f: impl FnOnce(&mut std::collections::BTreeMap<String,Value>)) -> Vec<u8> {
    let mut doc = json::parse(CONFIG,json::Limits::default()).unwrap();
    let Value::Object(root) = &mut doc else { panic!() };
    let Value::Object(text) = root.get_mut("text_config").unwrap() else { panic!() }; f(text);
    encode(&doc).into_bytes()
}
#[test]
fn official_text_config_is_a_typed_deferred_description() {
    let p = description();
    assert_eq!((p.hidden,p.vocab,p.max_positions,p.layers.len()),(4096,154880,1048576,45));
    assert_eq!(p.eos_tokens,[154820,154827,154829]); assert_eq!(p.pad_token,154820); assert!(!p.tied_embeddings);
    assert_eq!(p.epsilon,1e-5); assert!(p.rope.is_none()); assert_eq!(p.declared_draft_layers,1);
    let kda = p.layers.iter().filter(|l| matches!(l.mixer,TokenMixer::Kda(_))).count(); assert_eq!(kda,34);
    assert_eq!(p.layers.iter().filter(|l| matches!(l.ffn,Ffn::Dense { .. })).count(),3);
    let TokenMixer::Kda(k) = p.layers[0].mixer else { panic!() };
    assert_eq!((k.heads,k.head_dim,k.conv,k.gate_lower_bound),(64,128,4,-5.0));
    let TokenMixer::LatentAttention(a) = p.layers[3].mixer else { panic!() };
    assert_eq!((a.heads,a.query_rank,a.kv_rank,a.key_dim,a.value_dim,a.rotary_dim),(64,1536,512,256,256,0));
    assert_eq!((a.indexer.heads,a.indexer.dim,a.indexer.pool,a.indexer.token_budget),(32,128,4,2048));
    assert!(a.indexer.learned_pool && a.indexer.select_tail && a.indexer.share_for_draft_iteration);
    let Ffn::Experts(e) = p.layers[3].ffn else { panic!() };
    assert_eq!((e.count,e.selected,e.width,e.shared_width),(288,8,2048,2048));
    assert_eq!((p.residual.streams,p.residual.iterations,p.residual.epsilon),(4,20,1e-6));
    assert_eq!(p.routing.kind,engine_model::host_ops::router::RouterKind::CorrectedSigmoid);
    assert!(p.routing.normalize_selected); assert!(!p.routing.shared_gate);
    assert_eq!((p.routing.groups,p.routing.selected_groups,p.routing.scale,p.swiglu_limit),(1,1,2.5,10.0));
    assert!(ConfigPlan::from_config(CONFIG).unwrap().into_model_plan().is_err());
    assert!(ModelPlan::from_config(CONFIG).is_err());
}
#[test]
fn direct_text_schema_and_ignored_vision_config() {
    let mut root = json::parse(CONFIG,json::Limits::default()).unwrap();
    let Value::Object(o) = &mut root else { panic!() };
    let q = o["quantization_config"].clone();
    let mut text = o["text_config"].clone();
    let Value::Object(t) = &mut text else { panic!() }; t.insert("quantization_config".into(),q);
    assert_eq!(ConfigPlan::from_config(encode(&text).as_bytes()).unwrap(),ConfigPlan::Deferred(description()));
    assert_eq!(ConfigPlan::from_config(encode(&text).as_bytes()).unwrap().into_model_plan().unwrap_err(),
        description().require_resident_service().unwrap_err());
    o.insert("vision_config".into(),Value::String("ignored".into()));
    assert_eq!(ConfigPlan::from_config(encode(&root).as_bytes()).unwrap(),ConfigPlan::Deferred(description()));
}
#[test]
fn qwen_still_uses_the_existing_model_plan() {
    for bytes in [include_bytes!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice(),
        include_bytes!("fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice()] {
        let old = ModelPlan::from_config(bytes).unwrap();
        assert_eq!(ConfigPlan::from_config(bytes).unwrap().into_model_plan().unwrap(),old);
    }
}

#[test]
fn description_and_header_adapter_keep_their_documented_boundaries() {
    use engine_model::plan::{adapter::ConfigPlan as HeaderPlan, PlanError};
    let ConfigPlan::Deferred(description) = ConfigPlan::from_config(CONFIG).unwrap() else { panic!() };
    let HeaderPlan::Deferred(headers) = HeaderPlan::from_config(CONFIG).unwrap() else { panic!() };
    assert_eq!(description.hidden, headers.hidden);
    assert_eq!(description.layers.len() as u64, headers.decoder_layers);
    assert_eq!(description.declared_draft_layers, headers.draft_layers);
    assert!(matches!(ConfigPlan::Deferred(description).into_model_plan(), Err(PlanError::Unsupported { field: "execution", .. })));
    let error = HeaderPlan::Deferred(headers).into_model_plan().unwrap_err();
    assert!(matches!(error, PlanError::MissingOps { .. }));
    assert_eq!(ModelPlan::from_config(CONFIG).unwrap_err(), error);
    // Partial header admission does not claim full operation-field validation.
    let bytes = changed(|o| { o.remove("hc_eps"); });
    assert!(ConfigPlan::from_config(&bytes).is_err());
    assert!(matches!(HeaderPlan::from_config(&bytes).unwrap(), HeaderPlan::Deferred(_)));
}

#[test]
fn description_uses_the_host_sinkhorn_iteration_limit() {
    use engine_model::shapes::MAX_SINKHORN_ITERS;
    for iterations in [1, MAX_SINKHORN_ITERS] {
        let bytes = changed(|o| { o.insert("hc_sinkhorn_iters".into(), Value::Number(iterations.to_string())); });
        let ConfigPlan::Deferred(p) = ConfigPlan::from_config(&bytes).unwrap() else { panic!() };
        assert_eq!(p.residual.iterations, iterations);
    }
    for iterations in [0, MAX_SINKHORN_ITERS + 1, u64::MAX] {
        let bytes = changed(|o| { o.insert("hc_sinkhorn_iters".into(), Value::Number(iterations.to_string())); });
        assert!(ConfigPlan::from_config(&bytes).unwrap_err().to_string().contains("hc_sinkhorn_iters"));
    }
}
#[test]
fn explicit_schedules_have_priority_and_dimensions_are_not_fixed() {
    let bytes = changed(|o| { o.insert("hidden_size".into(),Value::Number("3072".into()));
        let Value::Array(a) = o.get_mut("mlp_layer_types").unwrap() else { panic!() }; a[0] = Value::String("sparse".into()); });
    let ConfigPlan::Deferred(p) = ConfigPlan::from_config(&bytes).unwrap() else { panic!() };
    assert_eq!(p.hidden,3072); assert!(matches!(p.layers[0].ffn,Ffn::Experts(_)));
}
#[test]
fn config_refuses_missing_wrong_and_conflicting_fields() {
    for (field,value) in [("hidden_size","0"),("rms_norm_eps","0"),("hc_eps","-1"),("hc_mult","0"),
        ("hc_sinkhorn_iters","0"),("qk_rope_head_dim","8"),("num_hidden_layers","46"),
        ("num_experts_per_tok","289"),("n_group","5"),("topk_group","2"),("index_kpool","3"),
        ("eos_token_id","[]"),("eos_token_id","[154820,154820]"),("eos_token_id","[154880]"),
        ("swiglu_limit","0"),("routed_scaling_factor","0"),("attention_bias","true"),
        ("tie_word_embeddings","true"),("indexer_types","[]"),("scoring_func","\"softmax\"")] {
        let bytes = changed(|o| { o.insert(field.into(),json::parse(value.as_bytes(),json::Limits::default()).unwrap()); });
        assert!(ConfigPlan::from_config(&bytes).is_err(),"{field}={value}");
    }
    for field in ["hidden_size","linear_attn_config","mlp_layer_types","hc_eps","num_nextn_predict_layers"] {
        assert!(ConfigPlan::from_config(&changed(|o| { o.remove(field); })).is_err(),"{field}");
    }
    for key in ["kda_layers","full_attn_layers"] {
        let bytes = changed(|o| { let Value::Object(l) = o.get_mut("linear_attn_config").unwrap() else { panic!() };
            l.insert(key.into(),Value::Array(vec![Value::Number("0".into()),Value::Number("0".into())])); });
        assert!(ConfigPlan::from_config(&bytes).is_err());
    }
}
#[test]
fn mixed_recipes_are_explicit_and_use_component_boundaries() {
    let p = description(); let storage = |s: &str| p.storage.storage(s);
    assert_eq!(storage("lm_head"),WeightStorage::Mxfp8);
    assert_eq!(storage("model.language_model.layers.0.self_attn.q_proj"),WeightStorage::Mxfp8);
    assert_eq!(storage("model.language_model.layers.3.self_attn.q_a_proj"),WeightStorage::Fp8Block { block: [128,128] });
    assert_eq!(storage("model.language_model.layers.3.self_attn.kv_b_proj"),WeightStorage::Plain);
    assert_eq!(storage("model.language_model.layers.3.mlp.experts.287.down_proj"),
        WeightStorage::Nvfp4 { encoding: ScaleEncoding::Csf32,activation_scale: true });
    assert_eq!(storage("model.language_model.layers.45.mlp.experts.0.gate_proj"),
        WeightStorage::Nvfp4 { encoding: ScaleEncoding::Direct,activation_scale: false });
    assert_eq!(storage("model.language_model.layers.3.mlp.experts_extra"),WeightStorage::Plain);
    let mut q = p.storage.clone(); q.excluded.push("model.language_model.layers.*.mlp.experts.0.*".into());
    assert_eq!(q.storage("model.language_model.layers.3.mlp.experts.0.gate_proj"),WeightStorage::Plain);
    assert!(matches!(q.storage("model.language_model.layers.3.mlp.experts.1.gate_proj"),WeightStorage::Nvfp4 { .. }));
    q.layers.insert("model.language_model.layers.3.mlp.experts.1".into(),WeightStorage::Mxfp8);
    assert_eq!(q.storage("model.language_model.layers.3.mlp.experts.1.gate_proj"),WeightStorage::Mxfp8);
}
#[test]
fn sidecar_and_config_wrappers_use_the_same_recipe_reader() {
    let source = r#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","exclude_modules":[],"quantized_layers":{"a":{"quant_algo":"MXFP8","weight_block_size":[1,32]},"b":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf_32x2_v1"}}}"#;
    let q = StorageMap::from_json(source.as_bytes()).unwrap();
    for key in ["quantization","quantization_config"] {
        assert_eq!(StorageMap::from_json(format!("{{{key:?}:{source}}}").as_bytes()).unwrap(),q);
    }
    for bad in [source.replace("[1,32]","[32,32]"),source.replace("\"group_size\":16","\"group_size\":32"),
        source.replace("csf_32x2_v1","unknown"),source.replace("\"quant_algo\":\"MXFP8\"","\"quant_algo\":\"MXFP8\",\"weight_scale_encoding\":\"csf\"")] {
        assert!(StorageMap::from_json(bad.as_bytes()).is_err());
    }
}
#[test]
fn saved_text_index_pattern_counts_match_roles() {
    let p = description(); let mut patterns = 0;
    let experts = p.layers.iter().filter_map(|l| if let Ffn::Experts(e) = l.ffn { Some(e.count) } else { None }).max().unwrap();
    for line in include_str!("fixtures/glm/index-patterns.txt").lines() {
        let mut fields = line.split_whitespace();
        let expected = fields.next().unwrap().parse::<usize>().unwrap();
        let pattern = fields.next().unwrap(); patterns += 1;
        if pattern.starts_with("model.visual.") {
            // Vision is unused. The text config does not supply its layer count.
            assert!(p.bind_tensor(&pattern.replace(".N.",".0.")).is_ok()); continue;
        }
        let layers = if pattern.contains(".N.") { p.layers.len()+p.declared_draft_layers as usize } else { 1 };
        let experts = if pattern.contains(".E.") { experts } else { 1 };
        let mut count = 0;
        for layer in 0..layers { for expert in 0..experts {
            let name = pattern.replace(".N.",&format!(".{layer}.")).replace(".E.",&format!(".{expert}."));
            count += usize::from(p.bind_tensor(&name).is_ok());
        } }
        assert_eq!(count,expected,"wrong tensor count for {pattern}");
    }
    assert_eq!(patterns,137);
}
#[test]
fn draft_layers_refuse_dense_projections() {
    let p = description();
    for layer in p.layers.len()..p.layers.len()+p.declared_draft_layers as usize {
        for projection in ["gate_proj","up_proj","down_proj"] {
            let name = format!("model.language_model.layers.{layer}.mlp.{projection}.weight");
            assert!(p.bind_tensor(&name).is_err(),"{name}");
        }
    }
}
#[test]
fn legacy_csf_recipes_bind_only_their_scale_components() {
    let mut p = description(); let module = "model.language_model.layers.3.mlp.experts.0.gate_proj";
    for algo in ["NVFP4","W4A16_NVFP4"] { for encoding in ["csf","csf_32x2_v1",""] {
        let field = if encoding.is_empty() { String::new() } else { format!(",\"weight_scale_encoding\":{encoding:?}") };
        let source = format!(r#"{{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{{{module:?}:{{"quant_algo":{algo:?},"group_size":16{field}}}}}}}"#);
        p.storage = StorageMap::from_json(source.as_bytes()).unwrap();
        for suffix in ["nvfp4_csf_fixed","nvfp4_csf_exceptions"] {
            let name = format!("{module}.weight_scale.{suffix}");
            let result = p.bind_tensor(&name);
            assert_eq!(result.is_ok(),encoding == "csf","{algo} {encoding} {name}");
            if let Ok(b) = result {
                assert_eq!((b.role.place,b.role.part,b.role.store),(Place::Layer(3),Part::Expert { id: Some(0),proj: Proj::Gate },Store::BlockScale));
                assert_eq!(b.storage,WeightStorage::Nvfp4 { encoding: ScaleEncoding::Csf,activation_scale: algo == "NVFP4" });
            }
        }
        assert_eq!(p.bind_tensor(&format!("{module}.weight_scale.csf32_offsets")).is_ok(),encoding == "csf_32x2_v1");
        assert_eq!(p.bind_tensor(&format!("{module}.weight_scale")).is_ok(),encoding.is_empty());
        assert!(p.bind_tensor(&format!("{module}.weight_scale_2")).is_ok());
        assert_eq!(p.bind_tensor(&format!("{module}.input_scale")).is_ok(),algo == "NVFP4");
        assert!(p.bind_tensor(&format!("{module}.weight_scale.nvfp4_csf_unknown")).is_err());
    } }
}
#[test]
fn roles_keep_the_new_operations_and_scale_components_distinct() {
    let p = description();
    for (suffix,operation) in [("self_attn.q_proj.weight",TensorRole::Kda(KdaTensor::Query)),
        ("self_attn.q_conv1d.weight",TensorRole::Kda(KdaTensor::QueryConv)),
        ("self_attn.f_a_proj.weight",TensorRole::Kda(KdaTensor::ForgetDown)),
        ("hc_attn_fn",TensorRole::Mhc { branch: Branch::Attention,parameter: MhcTensor::Projection })] {
        assert_eq!(p.bind_tensor(&format!("model.language_model.layers.0.{suffix}")).unwrap().operation,operation);
    }
    for (suffix,operation) in [("self_attn.kv_b_proj.weight",TensorRole::Latent(LatentTensor::KvUp)),
        ("self_attn.indexer.k_norm.bias",TensorRole::Index(IndexTensor::KeyNormBias)),
        ("self_attn.indexer.index_kpool_compress_gate",TensorRole::Index(IndexTensor::PoolGate)),
        ("mlp.gate.e_score_correction_bias",TensorRole::RouterBias),
        ("mlp.shared_experts.down_proj.weight",TensorRole::Shared(Proj::Down))] {
        assert_eq!(p.bind_tensor(&format!("model.language_model.layers.3.{suffix}")).unwrap().operation,operation);
    }
    let b = p.bind_tensor("model.language_model.layers.3.mlp.experts.287.gate_proj.weight_scale.csf32_offsets").unwrap();
    assert_eq!(b.component,TensorStorage::Csf32(CsfComponent::Offsets));
    assert_eq!((b.role.place,b.role.part,b.role.store),(Place::Layer(3),Part::Expert { id: Some(287),proj: Proj::Gate },Store::BlockScale));
    let b = p.bind_tensor("model.language_model.layers.45.eh_proj.weight").unwrap();
    assert_eq!(b.role.place,Place::Draft { layer: Some(0) });
    assert_eq!(b.operation,TensorRole::Draft(DraftTensor::EmbeddingHiddenProjection));
    assert_eq!(p.bind_tensor("model.visual.anything").unwrap().role.place,Place::Unused);
}
#[test]
fn unused_vision_tensors_keep_scale_components() {
    let p = description();
    for (suffix,component,store) in [("weight",TensorStorage::Weight,Store::Weight),
        ("weight_scale",TensorStorage::BlockScale,Store::BlockScale),
        ("weight_scale_2",TensorStorage::GlobalScale,Store::Scalar),
        ("input_scale",TensorStorage::InputScale,Store::Scalar),
        ("weight_scale.csf32_bases",TensorStorage::Csf32(CsfComponent::Bases),Store::BlockScale),
        ("weight_scale.csf32_codes",TensorStorage::Csf32(CsfComponent::Codes),Store::BlockScale),
        ("weight_scale.csf32_page_bases",TensorStorage::Csf32(CsfComponent::PageBases),Store::BlockScale),
        ("weight_scale.csf32_offsets",TensorStorage::Csf32(CsfComponent::Offsets),Store::BlockScale),
        ("weight_scale.csf32_exceptions",TensorStorage::Csf32(CsfComponent::Exceptions),Store::BlockScale)] {
        let b = p.bind_tensor(&format!("model.visual.proj.{suffix}")).unwrap();
        assert_eq!((b.operation,b.role.place,b.role.part),(TensorRole::Vision,Place::Unused,Part::Whole));
        assert_eq!((b.component,b.role.store,b.storage),(component,store,WeightStorage::Plain));
    }
    assert!(p.bind_tensor("model.visual.proj.weight_scale.csf32_unknown").is_err());
}
#[test]
fn index_reader_is_payload_free_and_refuses_unknown_text_or_bad_paths() {
    let p = description();
    let index = br#"{"metadata":{"total_size":123},"weight_map":{"lm_head.weight":"absent.safetensors","model.language_model.layers.0.self_attn.q_proj.weight":"absent.safetensors","model.visual.x":"absent.safetensors"}}"#;
    let mapped = p.map_index(index).unwrap(); assert_eq!(mapped.index.total_size,123); assert_eq!(mapped.tensors.len(),3);
    assert!(p.map_index(&String::from_utf8(index.to_vec()).unwrap().replace("absent.safetensors","../absent.safetensors").into_bytes()).is_err());
    for suffix in ["layers.46.eh_proj.weight","layers.03.self_attn.q_a_proj.weight","layers.0.self_attn.kv_b_proj.weight",
        "layers.3.self_attn.q_proj.weight","layers.3.mlp.experts.288.gate_proj.weight","layers.0.mlp.experts.0.gate_proj.weight",
        "layers.3.mlp.gate_proj.weight","layers.45.hc_attn_fn","layers.3.mlp.shared_experts.gate_proj.input_scale",
        "layers.3.mlp.experts.0.gate_proj.weight_scale","layers.3.mlp.experts.0.gate_proj.weight_scale.csf32_unknown",
        "layers.45.mlp.experts.0.gate_proj.input_scale","layers.0.self_attn.q_proj","layers.0.typo.weight"] {
        assert!(p.bind_tensor(&format!("model.language_model.{suffix}")).is_err(),"{suffix}");
    }
}
