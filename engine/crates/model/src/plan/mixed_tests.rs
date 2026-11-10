//! Metadata fixtures. No tensor payloads or device calls.
use super::*;
use crate::{flash::mixed::{self, Class, Storage}, safetensors::Dtype};

const NVFP4: &str = r#""a":{"quant_algo":"NVFP4","group_size":16}"#;
const QUERY: &str = "model.language_model.layers.1.self_attn.q_proj";

fn quantization(text: &str, quant: &str) -> String {
    let (prefix, _) = text.rsplit_once("\"quantization_config\":").unwrap();
    format!("{prefix}\"quantization_config\":{quant}}}")
}
fn mixed_config(fields: &str, layers: &str) -> String {
    let fields = if fields.is_empty() { r#""ignore":[],"# } else { fields };
    let quant = format!(r#"{{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",{fields}"quantized_layers":{{{layers}}}}}"#);
    quantization(&tests::flash_fixture(), &quant)
}
fn headers(plan: &ModelPlan) -> BTreeMap<String, Tensor> {
    let mut filled = plan.clone();
    if let Some(l) = &mut filled.shape.lexical { l.part_rows = 7; }
    let mut entries = tensors::expected(&filled).unwrap();
    if let Format::Mixed(q) = &filled.format { mixed::inventory(q, &mut entries, &BTreeMap::new()).unwrap(); }
    entries.into_iter().map(|(name, e)| (name, Tensor { dtype: e.dtype, shape: e.shape, start: 0, end: 0 })).collect()
}
fn parse(fields: &str, layers: &str) -> ModelPlan {
    ModelPlan::from_config(mixed_config(fields, layers).as_bytes()).unwrap()
}
fn storage(plan: &ModelPlan, module: &str) -> Storage {
    let Format::Mixed(q) = &plan.format else { panic!("expected mixed format") };
    q.storage(module)
}
// Build logical inventory metadata without changing executable admission.
fn storage_metadata(fields: &str, layers: &str) -> crate::Result<mixed::Mixed> {
    let bytes = mixed_config(fields, layers);
    let doc = crate::json::parse(bytes.as_bytes(), crate::json::Limits::default())?;
    mixed::Mixed::parse(doc.object()?["quantization_config"].object()?)
}
fn block_metadata(fields: &str, layers: &str) -> ModelPlan {
    let bytes = mixed_config(fields, layers);
    assert!(matches!(ModelPlan::from_config(bytes.as_bytes()), Err(PlanError::MissingOps { .. })));
    let q = storage_metadata(fields, layers).unwrap();
    let mut plan = ModelPlan::from_config(tests::flash_fixture().as_bytes()).unwrap();
    plan.format = Format::Mixed(q); plan
}
#[test]
fn public_mixed_exclusions_and_optional_cache() {
    let old = parse(r#""kv_cache_quant_algo":null,"ignore":["a.keep*"],"#, NVFP4);
    for fields in [r#""exclude_modules":["a.keep*"],"#,
        r#""kv_cache_quant_algo":null,"exclude_modules":["a.keep*"],"#] {
        let plan = parse(fields, NVFP4); assert_eq!(plan, old);
        assert_eq!(storage(&plan, "a.keep.proj"), Storage::Plain);
        assert_eq!(storage(&plan, "a.other"), Storage::Nvfp4 { csf: false, activation_scale: true });
    }
    let both = parse(r#""ignore":["a.keep*"],"exclude_modules":["a.other*"],"#, NVFP4);
    for name in ["a.keep.proj", "a.other.proj"] { assert_eq!(storage(&both, name), Storage::Plain); }
    assert_eq!(storage(&parse("", NVFP4), "a"), Storage::Nvfp4 { csf: false, activation_scale: true });
    let missing = mixed_config("", NVFP4).replace(r#""ignore":[],"#, "");
    assert!(ModelPlan::from_config(missing.as_bytes()).is_err());
    for fields in [r#""kv_cache_quant_algo":"FP8","#, r#""kv_cache_quant_algo":false,"#,
        r#""exclude_modules":null,"#, r#""exclude_modules":[1],"#, r#""ignore":"a","#] {
        assert!(ModelPlan::from_config(mixed_config(fields, NVFP4).as_bytes()).is_err(), "{fields}");
    }
}
#[test]
fn public_mxfp8_block_maps_to_existing_group_storage() {
    let recipe = |layout: &str| format!(r#"{QUERY:?}:{{"quant_algo":"MXFP8",{layout}}}"#);
    let group = parse(r#""ignore":[],"kv_cache_quant_algo":null,"#, &recipe(r#""group_size":32"#));
    let before = group.clone().with_headers(&headers(&group)).unwrap();
    for layout in [r#""weight_block_size":[1,32]"#, r#""group_size":32,"weight_block_size":[1,32]"#] {
        let plan = parse(r#""exclude_modules":[],"#, &recipe(layout));
        assert_eq!(storage(&plan, QUERY), Storage::Mxfp8);
        let after = plan.with_headers(&headers(&group)).unwrap();
        assert_eq!(format!("{before:?}").as_bytes(), format!("{after:?}").as_bytes());
        let scale = after.shape.tensors.iter().find(|t| t.name == format!("{QUERY}.weight_scale")).unwrap();
        assert_eq!((scale.dtype, scale.shape.as_slice(), scale.role.store), (Dtype::U8, &[2048,4][..], Store::BlockScale));
    }
    for layout in [r#""weight_block_size":[1,64]"#, r#""weight_block_size":[2,32]"#,
        r#""weight_block_size":[32]"#, r#""weight_block_size":[1,32,1]"#,
        r#""weight_block_size":null"#, r#""weight_block_size":[1,"32"]"#,
        r#""group_size":16,"weight_block_size":[1,32]"#] {
        assert!(storage_metadata("", &recipe(layout)).is_err(), "{layout}");
    }
    for suffix in ["weight", "weight_scale"] {
        let mut bad = headers(&group); bad.get_mut(&format!("{QUERY}.{suffix}")).unwrap().shape.push(1);
        assert!(matches!(group.clone().with_headers(&bad), Err(PlanError::TensorMismatch { .. })));
    }
}
fn block_fp8_headers(plan: &ModelPlan, module: &str, dtype: Dtype) -> BTreeMap<String, Tensor> {
    let mut plain = plan.clone(); plain.format = Format::Bf16;
    let mut h = headers(&plain);
    let weight = h.get_mut(&format!("{module}.weight")).unwrap(); weight.dtype = Dtype::F8E4M3;
    let shape = weight.shape.iter().map(|n| n.div_ceil(FP8_BLOCK)).collect();
    h.insert(format!("{module}.weight_scale"), Tensor { dtype, shape, start: 0, end: 0 }); h
}
#[test]
fn public_fp8_pb_wo_plans_weight_scale_without_runtime_support() {
    let recipe = |layout: &str| format!(r#"{QUERY:?}:{{"quant_algo":"FP8_PB_WO",{layout}}}"#);
    let plan = block_metadata("", &recipe(r#""weight_block_size":[128,128]"#));
    assert_eq!(storage(&plan, QUERY), Storage::Fp8Block { block: [128,128] });
    let grouped = block_metadata("", &recipe(r#""group_size":128,"weight_block_size":[128,128]"#));
    assert_eq!(storage(&grouped, QUERY), storage(&plan, QUERY));
    for shape in [vec![16,1], vec![16,1,1,1]] {
        let dtype = Dtype::F32;
        let mut h = block_fp8_headers(&plan, QUERY, dtype);
        h.get_mut(&format!("{QUERY}.weight_scale")).unwrap().shape = shape.clone();
        let p = plan.clone().with_headers(&h).unwrap();
        let weight = p.shape.tensors.iter().find(|t| t.name == format!("{QUERY}.weight")).unwrap();
        let scale = p.shape.tensors.iter().find(|t| t.name == format!("{QUERY}.weight_scale")).unwrap();
        assert_eq!((weight.dtype, weight.shape.as_slice()), (Dtype::F8E4M3, &[2048,128][..]));
        assert_eq!((scale.dtype, &scale.shape, scale.role), (dtype, &shape, Role { store: Store::BlockScale, ..weight.role }));
        let files = h.keys().map(|n| (n.clone(), "weights.safetensors".into())).collect();
        let mixed = mixed::Plan::from_model(p, files).unwrap();
        assert_eq!(mixed.classes[&Class::BlockFp8].tensors, 2);
        assert_eq!(mixed.classes[&Class::BlockFp8].bytes, 2048*128 + 16*dtype.bytes());
        assert!(mixed.nvfp4(QUERY).is_err()); assert!(mixed.model.hybrid_projections().is_err());
    }
    for layout in [r#""weight_block_size":[64,128]"#, r#""weight_block_size":[128]"#,
        r#""weight_block_size":[128,128,1]"#, r#""group_size":128"#,
        r#""group_size":64,"weight_block_size":[128,128]"#] {
        assert!(storage_metadata("", &recipe(layout)).is_err(), "{layout}");
    }
    for suffix in ["weight", "weight_scale"] {
        let mut bad = block_fp8_headers(&plan, QUERY, Dtype::F32);
        bad.get_mut(&format!("{QUERY}.{suffix}")).unwrap().shape.push(1);
        assert!(plan.clone().with_headers(&bad).is_err());
    }
    let scale = format!("{QUERY}.weight_scale");
    let mut bad = block_fp8_headers(&plan, QUERY, Dtype::F32); bad.get_mut(&scale).unwrap().shape[0] += 1;
    assert!(plan.clone().with_headers(&bad).is_err());
    for dtype in [Dtype::F16, Dtype::Bf16] {
        assert!(plan.clone().with_headers(&block_fp8_headers(&plan, QUERY, dtype)).is_err());
    }
    bad = block_fp8_headers(&plan, QUERY, Dtype::U8); assert!(plan.clone().with_headers(&bad).is_err());
    bad.remove(&scale); assert!(plan.clone().with_headers(&bad).is_err());
    bad.insert(format!("{QUERY}.weight_scale_inv"), Tensor { dtype: Dtype::F32, shape: vec![16,1], start: 0, end: 0 });
    assert!(plan.with_headers(&bad).is_err());
}
#[test]
fn supported_plan_refuses_plural_shared_expert_names() {
    let plan = parse("", NVFP4); let original = headers(&plan);
    let plural = |h: &BTreeMap<String, Tensor>| h.iter()
        .map(|(n, t)| (n.replace(".mlp.shared_expert.", ".mlp.shared_experts."), t.clone())).collect::<BTreeMap<_, _>>();
    let name = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
    let stored = name.replace(".shared_expert.", ".shared_experts.");
    let refused = PlanError::Unsupported { field: "shared expert tensor name",
        value: format!("{stored}: the supported plan reads {name}") };
    assert!(plan.clone().with_headers(&original).is_ok());
    assert_eq!(plan.clone().with_headers(&plural(&original)).unwrap_err(), refused);
    let mut both = plural(&original); both.insert(name.into(), original[name].clone());
    assert_eq!(plan.with_headers(&both).unwrap_err(), refused);
    let qwen = ModelPlan::from_config(tests::flash_fixture().as_bytes()).unwrap();
    assert!(matches!(qwen.clone().with_headers(&plural(&headers(&qwen))),
        Err(PlanError::Unsupported { field: "shared expert tensor name", .. })));
    let module = "model.language_model.layers.0.mlp.shared_experts.gate_proj";
    let recipe = format!(r#"{module:?}:{{"quant_algo":"FP8_PB_WO","weight_block_size":[128,128]}}"#);
    let block = block_metadata(r#""exclude_modules":[],"#, &recipe);
    let singular = module.replace(".shared_experts.", ".shared_expert.");
    let h = plural(&block_fp8_headers(&block, &singular, Dtype::F32));
    assert!(matches!(block.with_headers(&h), Err(PlanError::Unsupported { field: "shared expert tensor name", .. })));
}
#[test]
fn mixed_block_inventory_refuses_nonmatrix_targets() {
    let norm = "model.language_model.layers.1.self_attn.q_norm";
    for recipe in [r#""quant_algo":"MXFP8","weight_block_size":[1,32]"#,
        r#""quant_algo":"FP8_PB_WO","weight_block_size":[128,128]"#] {
        let layers = format!(r#"{norm:?}:{{{recipe}}}"#);
        let plan = if recipe.contains("FP8_PB_WO") { block_metadata("", &layers) } else { parse("", &layers) };
        let mut plain = plan.clone(); plain.format = Format::Bf16;
        let err = plan.with_headers(&headers(&plain)).unwrap_err().to_string();
        assert!(err.contains("quantized target is not a matrix"), "{err}");
    }
}
fn legacy_fixture_plans() -> Vec<(&'static str, ModelPlan)> {
    let flash = ModelPlan::from_config(tests::flash_fixture().as_bytes()).unwrap();
    // Flash has no BF16 config adapter. Test its existing typed BF16 inventory.
    let mut bf16 = flash.clone(); bf16.format = Format::Bf16;
    let fp4 = mixed_config(r#""kv_cache_quant_algo":null,"ignore":["model.language_model.layers.0.mlp.gate"],"#,
        r#""model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16},"mtp.layers.0.mlp.experts":{"quant_algo":"W4A16_NVFP4","group_size":16},"model.language_model.layers.0.linear_attn.in_proj_qkv":{"quant_algo":"MXFP8","group_size":32},"model.language_model.layers.1.ple.ple_embedding.ngram_embedding":{"quant_algo":"W4A16_NVFP4","group_size":16}"#);
    let fp4 = ModelPlan::from_config(fp4.as_bytes()).unwrap();
    let mut plans = vec![("flash-bf16", bf16), ("flash-fp8", flash), ("flash-fp4-local", fp4)];
    for (_, plan) in &mut plans { *plan = plan.clone().with_headers(&headers(plan)).unwrap(); }
    for (name, bytes) in [
        ("qwen3.6-fp8", include_bytes!("../../tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice()),
        ("qwen3.5-fp8", include_bytes!("../../tests/fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice()),
    ] { plans.push((name, ModelPlan::from_config(bytes).unwrap())); }
    plans
}
#[test]
fn qwen_fixture_plans_stay_byte_identical() {
    let plans = legacy_fixture_plans();
    let bytes = plans.iter().map(|(name, plan)| format!("{name}\n{plan:?}\n")).collect::<String>();
    // Saved before the parser change, with production code at 99addd660f1a9d34d529a2690df7470b3d180e4b.
    assert!(bytes.as_bytes() == include_bytes!("../../tests/fixtures/qwen/mixed-plan-baseline.txt"),
        "Qwen fixture plan bytes differ from the base revision");
}
