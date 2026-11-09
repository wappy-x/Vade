use engine_model::{
    flash::mixed::Storage, json::{self, Value},
    plan::{adapter::ConfigPlan, DeferredPlan, FfnKind, ModelPlan, PlanError},
    safetensors::{Dtype, Header, Limits, Tensor}, shapes::{Part, Place, Proj, Role, Store},
};
use std::{collections::BTreeMap, io::{self, Cursor, Read, Seek, SeekFrom}};

const CONFIG: &str = include_str!("fixtures/glm-plan/config.json");
const HEADER: &[u8] = include_bytes!("fixtures/glm-plan/header.json");
const SHARED: &str = "model.language_model.layers.1.mlp.shared_experts";
const EXPERT: &str = "model.language_model.layers.1.mlp.experts.0.gate_proj";

// Only the prefix and header exist. Any payload read fails.
struct HeaderOnly { metadata: Cursor<Vec<u8>>, file_bytes: u64, bytes_read: usize }
impl Read for HeaderOnly {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.metadata.position() >= self.metadata.get_ref().len() as u64 {
            return Err(io::Error::other("test attempted to read tensor payload"));
        }
        let read = self.metadata.read(out)?; self.bytes_read += read; Ok(read)
    }
}
impl Seek for HeaderOnly {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(n) => self.file_bytes as i128 + n as i128,
            SeekFrom::Current(n) => self.metadata.position() as i128 + n as i128,
        };
        let position = u64::try_from(position).map_err(|_| io::Error::other("invalid test seek"))?;
        self.metadata.set_position(position); Ok(position)
    }
}
fn read_header(bytes: &[u8]) -> engine_model::Result<Header> {
    let doc = json::parse(bytes, json::Limits::default())?;
    let payload = doc.object()?.values().map(|v| v.object().unwrap()["data_offsets"].array().unwrap()[1].u64().unwrap()).max().unwrap();
    let mut data = (bytes.len() as u64).to_le_bytes().to_vec(); data.extend_from_slice(bytes);
    let file_bytes = data.len() as u64 + payload;
    let mut reader = HeaderOnly { metadata: Cursor::new(data), file_bytes, bytes_read: 0 };
    let header = Header::read(&mut reader, Limits::default())?;
    assert_eq!(reader.bytes_read, bytes.len() + 8);
    Ok(header)
}
fn headers() -> BTreeMap<String, Tensor> { read_header(HEADER).unwrap().tensors }
fn deferred(plan: ConfigPlan) -> DeferredPlan {
    let ConfigPlan::Deferred(plan) = plan else { panic!("expected deferred metadata"); }; plan
}
fn parse(source: &str) -> DeferredPlan { deferred(ConfigPlan::from_config(source.as_bytes()).unwrap()) }
fn reconcile(source: &str, headers: &BTreeMap<String, Tensor>) -> Result<DeferredPlan, PlanError> {
    ConfigPlan::from_config(source.as_bytes())?.with_headers(headers).map(deferred)
}
// The fixture has ASCII strings with no escape characters.
fn render(v: &Value) -> String {
    match v {
        Value::Null => "null".into(), Value::Bool(b) => b.to_string(), Value::Number(n) => n.clone(),
        Value::String(s) => { assert!(!s.chars().any(|c| c.is_control() || c == '\\' || c == '"')); format!("\"{s}\"") },
        Value::Array(a) => format!("[{}]", a.iter().map(render).collect::<Vec<_>>().join(",")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k,v)| format!("{}:{}", render(&Value::String(k.clone())), render(v))).collect::<Vec<_>>().join(",")),
    }
}

#[test]
fn root_and_text_aliases_keep_metadata_and_name_all_execution_gaps() {
    let doc = json::parse(CONFIG.as_bytes(), json::Limits::default()).unwrap(); let root = doc.object().unwrap();
    let mut text = root["text_config"].object().unwrap().clone();
    for field in ["quantization_config", "vision_config"] { text.insert(field.into(), root[field].clone()); }
    let plan = parse(CONFIG);
    assert_eq!(plan, parse(&render(&Value::Object(text))));
    assert_eq!((plan.hidden, plan.decoder_layers, plan.draft_layers), (128, 2, 1));
    assert_eq!(plan.layer_types, ["linear_attention", "deepseek_sparse_attention"]);
    assert_eq!(plan.mlp_layer_types, ["dense", "sparse"]);
    assert_eq!(plan.ffn_layers, [FfnKind::Dense, FfnKind::Experts, FfnKind::Experts]);
    assert_eq!(plan.storage("lm_head").unwrap(), Storage::Mxfp8);
    assert_eq!(plan.storage(&format!("{SHARED}.gate_proj")).unwrap(), Storage::Fp8Block { block: [128, 128] });
    assert_eq!(plan.storage("model.language_model.layers.0.self_attn.q_proj").unwrap(), Storage::Plain);
    assert!(matches!(plan.storage(EXPERT), Err(PlanError::MissingOps { .. })));
    let error = ModelPlan::from_config(CONFIG.as_bytes()).unwrap_err();
    assert!(matches!(error, PlanError::MissingOps { .. }));
    let message = error.to_string();
    for gap in ["KDA", "MLA low-rank projections and latent cache", "DSA", "mHC", "sigmoid", "noaux_tc", "SwiGLU", "schedule", "MTP", "vision/video", "csf_32x2_v1", "FP8_PB_WO"] {
        assert!(message.contains(gap), "missing {gap}: {message}");
    }
    assert_eq!(ConfigPlan::from_config(CONFIG.as_bytes()).unwrap().into_model_plan().unwrap_err(), error);
}

#[test]
fn header_fixture_maps_shared_roles_and_preserves_storage_names() {
    let h = headers(); let plan = reconcile(CONFIG, &h).unwrap();
    assert_eq!(plan.tensors.len(), 6);
    for (projection, part) in [("gate_proj", Proj::Gate), ("up_proj", Proj::Up), ("down_proj", Proj::Down)] {
        for (suffix, store) in [("weight", Store::Weight), ("weight_scale", Store::BlockScale)] {
            let source = format!("{SHARED}.{projection}.{suffix}");
            let tensor = plan.tensors.iter().find(|t| t.stored.name == source).unwrap();
            assert_eq!(tensor.plan_name, source.replace(".shared_experts.", ".shared_expert."));
            assert_eq!(tensor.stored.role, Role { place: Place::Layer(1), part: Part::Shared(part), store });
            assert_eq!((tensor.stored.dtype, &tensor.stored.shape), (h[&source].dtype, &h[&source].shape));
        }
    }
    assert_eq!(plan.unmapped_tensors.len(), h.len() - plan.tensors.len());
    assert!(plan.tensors.iter().all(|t| !t.plan_name.starts_with("mtp.")));
    assert!(plan.unmapped_tensors.contains(&format!("{EXPERT}.weight")));
    for name in ["eh_proj", "enorm", "hnorm", "shared_head.norm"] {
        assert!(plan.unmapped_tensors.contains(&format!("model.language_model.layers.2.{name}.weight")));
    }
    assert_eq!(deferred(ConfigPlan::Deferred(plan.clone()).with_headers(&h).unwrap()), plan);
}

#[test]
fn deferred_plan_keeps_the_plural_shared_expert_alias() {
    let plan = reconcile(CONFIG, &headers()).unwrap();
    let stored = format!("{SHARED}.gate_proj.weight");
    let mapped = plan.tensors.iter().find(|t| t.stored.name == stored).unwrap();
    assert_eq!(mapped.plan_name, "model.language_model.layers.1.mlp.shared_expert.gate_proj.weight");
    // The adapter reads the plural public names only. Singular stored names stay unmapped.
    let singular = headers().into_iter().map(|(n, t)| (n.replace(".mlp.shared_experts.", ".mlp.shared_expert."), t)).collect();
    let plan = reconcile(CONFIG, &singular).unwrap();
    assert!(plan.tensors.is_empty());
    assert!(plan.unmapped_tensors.contains(&stored.replace(".shared_experts.", ".shared_expert.")));
}

#[test]
fn inferred_ffn_plan_checks_shared_headers_without_a_label_list() {
    let source = CONFIG.replace("\"mlp_layer_types\": [\"dense\", \"sparse\"],", "\"n_routed_experts\": 4,");
    let plan = reconcile(&source, &headers()).unwrap();
    assert!(plan.mlp_layer_types.is_empty());
    assert_eq!(plan.ffn_layers, [FfnKind::Dense, FfnKind::Experts, FfnKind::Experts]);
    assert_eq!(plan.tensors.len(), 6);
    let dense = source.replace("\"first_k_dense_replace\": 1", "\"first_k_dense_replace\": 2");
    assert_eq!(reconcile(&dense, &headers()).unwrap_err(), PlanError::Invalid {
        field: "first_k_dense_replace", reason: "shared expert tensor outside a sparse layer" });
    let listed = CONFIG.replace("[\"dense\", \"sparse\"]", "[\"dense\", \"dense\"]");
    assert_eq!(reconcile(&listed, &headers()).unwrap_err(), PlanError::Invalid {
        field: "mlp_layer_types", reason: "shared expert tensor outside a sparse layer" });
    let no_experts = source.replace("\"n_routed_experts\": 4,", "");
    assert_eq!(reconcile(&no_experts, &headers()).unwrap_err(), PlanError::Invalid {
        field: "first_k_dense_replace or expert count", reason: "shared expert tensor outside a sparse layer" });
}

#[test]
fn shared_projection_shapes_keep_output_and_input_axes() {
    let source = CONFIG.replace("\"moe_intermediate_size\": 64", "\"moe_intermediate_size\": 128");
    let mut h = BTreeMap::new();
    let mut offset = 0;
    for (projection, weight_shape, dtype, scale_shape) in [
        ("gate_proj", vec![256, 128], Dtype::F32, vec![2, 1]),
        ("up_proj", vec![256, 128], Dtype::U8, vec![256, 4]),
        ("down_proj", vec![128, 256], Dtype::F32, vec![1, 1, 2, 1]),
    ] {
        for (suffix, dtype, shape) in [("weight", Dtype::F8E4M3, weight_shape), ("weight_scale", dtype, scale_shape)] {
            let end = offset + shape.iter().product::<u64>() * dtype.bytes();
            h.insert(format!("{SHARED}.{projection}.{suffix}"), Tensor { dtype, shape, start: offset, end });
            offset = end;
        }
    }
    let plan = reconcile(&source, &h).unwrap();
    let down = plan.tensors.iter().find(|t| t.stored.name.ends_with("down_proj.weight")).unwrap();
    assert_eq!(down.stored.shape, [128, 256]);
    let mut transposed = h;
    transposed.get_mut(&format!("{SHARED}.down_proj.weight")).unwrap().shape = vec![256, 128];
    assert!(matches!(reconcile(&source, &transposed), Err(PlanError::TensorMismatch { .. })));
}

#[test]
fn shared_header_validation_rejects_wrong_shapes_dtypes_and_scale_names() {
    let h = headers(); let weight = format!("{SHARED}.gate_proj.weight"); let scale = format!("{SHARED}.gate_proj.weight_scale");
    let mut bad = h.clone(); bad.get_mut(&weight).unwrap().shape[0] += 1;
    assert!(matches!(reconcile(CONFIG, &bad), Err(PlanError::TensorMismatch { .. })));
    let mut bad = h.clone(); bad.get_mut(&weight).unwrap().dtype = Dtype::Bf16;
    assert!(matches!(reconcile(CONFIG, &bad), Err(PlanError::Unsupported { field: "tensor dtype", .. })));
    for (dtype, shape) in [(Dtype::Bf16, vec![1, 1]), (Dtype::F32, vec![2, 1]), (Dtype::F32, vec![1, 1, 1])] {
        let mut bad = h.clone(); let t = bad.get_mut(&scale).unwrap(); t.dtype = dtype; t.shape = shape;
        assert!(reconcile(CONFIG, &bad).unwrap_err().to_string().contains("weight_scale"));
    }
    let mut missing = h.clone(); missing.remove(&scale);
    assert!(reconcile(CONFIG, &missing).unwrap_err().to_string().contains(&scale));
    let mut old_name = h.clone(); let t = old_name.remove(&scale).unwrap(); old_name.insert(format!("{scale}_inv"), t);
    assert!(reconcile(CONFIG, &old_name).is_err());
    let mut extra = h.clone(); extra.insert(format!("{SHARED}.gate_proj.input_scale"), Tensor { dtype: Dtype::F32, shape: vec![], start: 0, end: 4 });
    assert!(matches!(reconcile(CONFIG, &extra), Err(PlanError::UnexpectedTensor { .. })));
    let mut bad_mx = h.clone(); bad_mx.get_mut(&format!("{SHARED}.up_proj.weight_scale")).unwrap().shape[1] += 1;
    assert!(matches!(reconcile(CONFIG, &bad_mx), Err(PlanError::TensorMismatch { .. })));
}

#[test]
fn config_bounds_and_layer_headers_fail_before_execution_admission() {
    for source in [
        CONFIG.replace("\"hidden_size\": 128", "\"hidden_size\": 0"),
        CONFIG.replace("\"num_hidden_layers\": 2", "\"num_hidden_layers\": 3"),
        CONFIG.replace("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 1025"),
        CONFIG.replace("\"n_shared_experts\": 2", "\"n_shared_experts\": 1025"),
        CONFIG.replace("\"model_type\": \"glm5_next_text\"", "\"model_type\": \"wrong_text\""),
        CONFIG.replace("\"exclude_modules\": [", "\"exclude_modules\": null, \"ignored_test_field\": ["),
    ] { assert!(ConfigPlan::from_config(source.as_bytes()).is_err()); }
    let h = headers();
    assert!(reconcile(&CONFIG.replace("\"n_shared_experts\": 2", "\"n_shared_experts\": 0"), &h).is_err());
    assert!(reconcile(&CONFIG.replace("[\"dense\", \"sparse\"]", "[\"dense\", \"dense\"]"), &h).is_err());
    let mut outside = h.clone(); outside.insert("model.language_model.layers.3.enorm.weight".into(), Tensor { dtype: Dtype::Bf16, shape: vec![128], start: 0, end: 256 });
    assert!(matches!(reconcile(CONFIG, &outside), Err(PlanError::Invalid { field: "tensor layer", .. })));
    let mut overflow = h; overflow.insert("opaque.weight".into(), Tensor { dtype: Dtype::Bf16, shape: vec![u64::MAX, 2], start: 0, end: 0 });
    assert!(matches!(reconcile(CONFIG, &overflow), Err(PlanError::Overflow { .. })));
}

#[test]
fn adapter_rejects_unknown_attention_labels_in_each_schema() {
    let root = json::parse(CONFIG.as_bytes(), json::Limits::default()).unwrap();
    let root = root.object().unwrap();
    let mut text = root["text_config"].object().unwrap().clone();
    text.insert("quantization_config".into(), root["quantization_config"].clone());
    for source in [CONFIG.to_owned(), render(&Value::Object(text))] {
        assert_eq!(parse(&source).layer_types, ["linear_attention", "deepseek_sparse_attention"]);
        for label in ["linear_attention", "deepseek_sparse_attention"] {
            let bad = source.replace(label, "bogus_attention");
            let error = ConfigPlan::from_config(bad.as_bytes()).unwrap_err();
            assert_eq!(error, PlanError::Invalid { field: "layer_types", reason: "unknown attention operation" });
            assert_eq!(ModelPlan::from_config(bad.as_bytes()).unwrap_err(), error);
        }
        let bad = source.replace("\"linear_attention\"", "null");
        assert!(matches!(ConfigPlan::from_config(bad.as_bytes()), Err(PlanError::WrongType { field: "layer_types", .. })));
    }
}

#[test]
fn appended_draft_count_does_not_guess_or_collapse_mtp_aliases() {
    let source = CONFIG.replace("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 2");
    let mut h = headers();
    for name in ["eh_proj", "enorm", "hnorm", "shared_head.norm"] {
        let tensor = h[&format!("model.language_model.layers.2.{name}.weight")].clone();
        h.insert(format!("model.language_model.layers.3.{name}.weight"), tensor);
    }
    let plan = reconcile(&source, &h).unwrap();
    assert_eq!(plan.draft_layers, 2);
    assert_eq!(plan.ffn_layers, [FfnKind::Dense, FfnKind::Experts, FfnKind::Experts, FfnKind::Experts]);
    assert_eq!(plan.tensors.len(), 6);
    assert_eq!(plan.unmapped_tensors.iter().filter(|n| n.starts_with("model.language_model.layers.2.") || n.starts_with("model.language_model.layers.3.")).count(), 8);
    assert!(plan.missing_ops.iter().any(|s| s.contains("local public semantics missing")));
}

#[test]
fn unknown_codec_respects_exclusions_and_more_specific_known_recipes() {
    let ignored = CONFIG.replace("\"exclude_modules\": [", &format!("\"exclude_modules\": [\"{EXPERT}\", "));
    assert_eq!(parse(&ignored).storage(EXPERT).unwrap(), Storage::Plain);
    let child = CONFIG.replace("\"quantized_layers\": {", &format!("\"quantized_layers\": {{\"{EXPERT}\": {{\"quant_algo\":\"MXFP8\",\"group_size\":32}},"));
    assert_eq!(parse(&child).storage(EXPERT).unwrap(), Storage::Mxfp8);
    assert!(parse(&child).storage("model.language_model.layers.1.mlp.experts.1.gate_proj").is_err());
    assert_eq!(parse(CONFIG).storage("model.language_model.layers.1.mlp.experts_suffix").unwrap(), Storage::Plain);
    let bad = CONFIG.replace("\"group_size\": 16, \"weight_scale_encoding\"", "\"group_size\": 32, \"weight_scale_encoding\"");
    assert!(ConfigPlan::from_config(bad.as_bytes()).is_err());
}

#[test]
fn header_reader_checks_byte_geometry_without_payload_reads() {
    let header = read_header(HEADER).unwrap();
    assert_eq!(header.file_bytes - header.data_start, 152840);
    let text = std::str::from_utf8(HEADER).unwrap();
    assert!(read_header(text.replace("[0, 16384]", "[0, 16383]").as_bytes()).is_err());
    assert!(read_header(text.replace("[16384, 16388]", "[16385, 16389]").as_bytes()).is_err());
}

#[test]
fn existing_configs_remain_supported_and_new_storage_needs_execution_binding() {
    for bytes in [
        include_bytes!("fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice(),
        include_bytes!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice(),
    ] {
        let plan = ConfigPlan::from_config(bytes).unwrap();
        assert!(matches!(plan, ConfigPlan::Supported(_)));
        assert_eq!(plan.into_model_plan().unwrap(), ModelPlan::from_config(bytes).unwrap());
        let mut root = json::parse(bytes, json::Limits::default()).unwrap().object().unwrap().clone();
        let q = json::parse(br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","exclude_modules":[],"quantized_layers":{"model.language_model.layers.0.mlp.shared_expert.gate_proj":{"quant_algo":"FP8_PB_WO","weight_block_size":[128,128]}}}"#, json::Limits::default()).unwrap();
        root.insert("quantization_config".into(), q);
        let error = ModelPlan::from_config(render(&Value::Object(root)).as_bytes()).unwrap_err();
        assert!(matches!(error, PlanError::MissingOps { .. }));
        assert!(error.to_string().contains("FP8_PB_WO"));
    }
}
