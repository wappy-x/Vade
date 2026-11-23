//! Looped text config, storage sidecar and refusal checks on a small made-up fixture.
use engine_model::{plan::{ConfigPlan, PlanError, description::*, inkling::{FIXED_TAP_ORDER, TOKEN_TAP_INDEX, TOKEN_TAP_ORDER},
    storage::{ScaleEncoding, WeightStorage}, storage_groups::*}, host_ops::router::RouterKind,
    safetensors::{Dtype, Index, Limits, Tensor}, shapes::*};
use std::collections::{BTreeMap, BTreeSet};

const CONFIG: &str = include_str!("fixtures/loop-plan/config.json");
const SIDECAR: &str = include_str!("fixtures/loop-plan/hf_quant_config.json");
const NVFP4: WeightStorage = WeightStorage::Nvfp4 { encoding: ScaleEncoding::AmaxDenominator, activation_scale: false };

fn description() -> LoopTextDescription { LoopTextDescription::from_config(CONFIG.as_bytes()).unwrap() }
fn groups() -> StorageGroups { StorageGroups::from_json(SIDECAR.as_bytes()).unwrap() }
/// Change one place of a fixture. A fixture that lacks the text fails the test.
fn edit(text: &str, from: &str, to: &str) -> String {
    assert_eq!(text.matches(from).count(), 1, "the fixture must hold `{from}` one time");
    text.replace(from, to)
}
fn config_refusal(from: &str, to: &str) -> PlanError { LoopTextDescription::from_config(edit(CONFIG, from, to).as_bytes()).unwrap_err() }
fn sidecar_refusal(from: &str, to: &str) -> String { StorageGroups::from_json(edit(SIDECAR, from, to).as_bytes()).unwrap_err().to_string() }
fn order(d: &LoopTextDescription) -> Vec<(u64, u64, u64, bool)> { d.visits.iter().map(|v| (v.layer, v.pass, v.slot, v.messages)).collect() }

#[test]
fn config_gives_the_operation_data() {
    let d = description();
    assert_eq!((d.hidden, d.vocab, d.head_rows, d.max_positions, d.epsilon), (16, 32, 30, 64, 1e-6));
    assert_eq!((d.eos_tokens.as_slice(), d.pad_token, d.declares_vision), ([29].as_slice(), None, true));
    assert_eq!(d.layers.len(), 6);
    let conv = |taps| ShortConvDescription { kernel: 4, taps };
    let window = SinkAttentionDescription { query_heads: 2, kv_heads: 1, head_dim: 8, window: Some(5),
        relative: RelativeBiasDescription { dim: 2, extent: 5, clip: 2.0 }, query_key_clip: 5.5, position_scale: None,
        key_value_conv: conv(ConvTaps::Fixed { order: TapOrder::OldestFirst }), separate_inputs: false };
    let full = SinkAttentionDescription { query_heads: 4, kv_heads: 2, head_dim: 4, window: None,
        relative: RelativeBiasDescription { dim: 2, extent: 8, clip: 2.0 }, query_key_clip: 5.5,
        position_scale: Some(PositionScale { floor: 16, alpha: 0.25 }),
        key_value_conv: conv(ConvTaps::FixedAndToken { order: TapOrder::NewestFirst, index: TapIndex::HeadThenLag }), separate_inputs: true };
    let ffn = Ffn::Experts(ExpertShape { count: 4, selected: 2, width: 8, shared_width: 0, storage: ExpertStorage::Stacked });
    // (attention, first layer of a block, rows of the attention mix query, message gates)
    let want = [(window, true, 0, false), (full, false, 3, false), (window, true, 1, true), (full, false, 3, true),
        (window, true, 1, false), (full, false, 3, false)];
    for (layer, (attention, boundary, attention_queries, message_gates)) in d.layers.iter().zip(want) {
        assert_eq!(*layer, LoopLayerDescription { attention, ffn, boundary, attention_queries, message_gates });
    }
    assert_eq!((d.full_layers(), d.loop_layers(), d.blocks()), (vec![1, 3, 5], 2..4, 3));
    assert_eq!(d.residual, BlockMixDescription { block_layers: 2, normalize: true });
    assert_eq!(d.core, LoopRange { start: 2, end: 4, passes: 3, gate_rank: 3 });
    assert_eq!(d.routing, RoutingDescription { kind: RouterKind::CorrectedSigmoid, normalize_selected: true, scale: 2.0,
        groups: 1, selected_groups: 1, shared_gate: false });
    assert_eq!(d.caps, SoftCaps { gate: 4.0, up: 25.0, logits: 30.0 });
    assert_eq!(d.conv, conv(ConvTaps::Fixed { order: FIXED_TAP_ORDER }));
    assert_eq!((FIXED_TAP_ORDER, TOKEN_TAP_ORDER, TOKEN_TAP_INDEX), (TapOrder::OldestFirst, TapOrder::NewestFirst, TapIndex::HeadThenLag));
    assert_eq!(d.declared_storage, Some(DeclaredStorage { method: "modelopt".into(), algo: "NVFP4".into() }));
    assert_eq!(d.declared_draft, Some(DraftDeclaration { layers: 2, taps: vec![1, 5, 10], hidden: 16, query_heads: 4, kv_heads: 2,
        head_dim: 4, width: 32, markov_rank: 5 }));
}

#[test]
fn visits_follow_the_loop_order() {
    // Layers 0 and 1 once, layers 2 and 3 three times, layers 4 and 5 once.
    // A later pass has its own state slots after the slots of the 6 layers.
    assert_eq!(order(&description()), [(0, 0, 0, false), (1, 0, 1, false), (2, 1, 2, false), (3, 1, 3, false),
        (2, 2, 6, true), (3, 2, 7, true), (2, 3, 8, true), (3, 3, 9, true), (4, 0, 4, false), (5, 0, 5, false)]);
    // Every slot occurs one time, and the slots are 0 to visits - 1.
    let d = description();
    assert_eq!(d.visits.iter().map(|v| v.slot).collect::<BTreeSet<_>>(), (0..d.visits.len() as u64).collect());
    // One pass: every layer runs once, in order, and no visit reads a message.
    let once = LoopTextDescription::from_config(edit(CONFIG, "\"num_passes\": 3", "\"num_passes\": 1").as_bytes()).unwrap_err();
    assert_eq!(once, PlanError::Invalid { field: "target_layer_ids", reason: "not rising or outside its limit" });
    let once = edit(&edit(CONFIG, "\"num_passes\": 3", "\"num_passes\": 1"), "      10\n", "      6\n");
    let once = LoopTextDescription::from_config(once.as_bytes()).unwrap();
    assert_eq!(order(&once), (0..6).map(|l| (l, u64::from((2..4).contains(&l)), l, false)).collect::<Vec<_>>());
    assert!(once.layers[2].message_gates && once.layers[3].message_gates);
    // The text config alone gives the same decoder. The root labels are then absent.
    let text = CONFIG.split_once("\"text_config\": ").unwrap().1.split_once(",\n  \"vision_config\"").unwrap().0;
    let alone = LoopTextDescription::from_config(edit(text, "\"model_type\": \"inkling2_loop_text\",", "\"model_type\": \"inkling2_loop_text\", \"eos_token_id\": 29,").as_bytes()).unwrap();
    assert_eq!(alone, LoopTextDescription { declared_storage: None, declared_draft: None, declares_vision: false, ..description() });
}

#[test]
fn config_plan_has_a_separate_arm_and_refuses_execution() {
    let plan = ConfigPlan::from_config(CONFIG.as_bytes()).unwrap();
    assert_eq!(plan, ConfigPlan::Loop(description()));
    let refusal = plan.into_model_plan().unwrap_err();
    assert_eq!(refusal, description().require_resident_service().unwrap_err());
    let PlanError::Unsupported { field: "execution", value } = refusal else { panic!("expected an execution refusal") };
    for part in ["metadata only", "state for each layer visit", "Image input is not supported"] { assert!(value.contains(part), "{value}"); }
    // The older text description does not take this schema, and this type takes no other schema.
    assert!(matches!(ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap(), ConfigPlan::Deferred(_)));
    assert_eq!(LoopTextDescription::from_config(include_bytes!("fixtures/glm/config.json")).unwrap_err(),
        PlanError::Unsupported { field: "model_type", value: "not a looped text schema".into() });
}

#[test]
fn each_unknown_or_wrong_field_is_refused() {
    let unsupported = |field, value: &str| PlanError::Unsupported { field, value: value.into() };
    let invalid = |field, reason| PlanError::Invalid { field, reason };
    let limit = "dimension is outside its limit";
    for (from, to, want) in [
        ("\"d_rel\": 2,", "\"d_rel\": 2, \"new_switch\": true,", unsupported("text_config", "unknown field new_switch")),
        ("\"gate_rank\": 3,", "\"gate_rank\": 3, \"extra\": 1,", unsupported("loop_config", "unknown field extra")),
        ("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"NVFP4\", \"quantized_layers\": {}", unsupported("quantization_config", "unknown field quantized_layers")),
        ("\"model_type\": \"inkling2_loop_text\"", "\"model_type\": \"other_text\"", unsupported("model_type", "other_text")),
        ("\"torch_dtype\": \"bfloat16\"", "\"torch_dtype\": \"float32\"", unsupported("torch_dtype", "float32")),
        ("\"dtype\": \"bfloat16\",\n  \"arch", "\"dtype\": \"float16\",\n  \"arch", unsupported("dtype", "float16")),
        ("\"d_rel\": 2,", "\"d_rel\": 2, \"scoring_func\": \"softmax\",", unsupported("scoring_func", "softmax")),
        ("\"d_rel\": 2,", "\"d_rel\": 2, \"hidden_act\": \"gelu\",", unsupported("hidden_act", "gelu")),
        ("\"q_bias\": false", "\"q_bias\": true", unsupported("q_bias", "true")),
        ("\"o_bias\": false", "\"o_bias\": true", unsupported("o_bias", "true")),
        ("\"use_value_norm_no_gamma\": true", "\"use_value_norm_no_gamma\": false", unsupported("use_value_norm_no_gamma", "false")),
        ("\"attn_res_normalize_values\": true", "\"attn_res_normalize_values\": false", unsupported("attn_res_normalize_values", "false")),
        ("\"use_gate_bias\": true", "\"use_gate_bias\": false", unsupported("use_gate_bias", "false")),
        ("\"is_varloop\": false", "\"is_varloop\": true", unsupported("is_varloop", "true")),
        ("\"d_rel\": 2,", "\"d_rel\": 2, \"layer_types\": [\"hybrid_sliding\",\"hybrid\",\"hybrid_sliding\",\"hybrid\",\"hybrid_sliding\",\"dense\"],", unsupported("layer_types", "dense")),
        ("\"d_rel\": 2,", "\"d_rel\": 2, \"layer_types\": [\"hybrid_sliding\",\"hybrid\",\"hybrid_sliding\",\"hybrid\",\"hybrid\",\"hybrid\"],", invalid("layer_types", "differs from local_layer_ids")),
        ("\"d_rel\": 2,", "\"d_rel\": 2, \"layer_types\": [\"hybrid\"],", invalid("layer_types", "length differs from num_hidden_layers")),
        ("\"hidden_size\": 16,\n    \"num_hidden_layers\": 6", "\"num_hidden_layers\": 6", PlanError::Missing { field: "hidden_size" }),
        ("\"hidden_size\": 16,\n    \"num_hidden_layers\": 6", "\"hidden_size\": \"16\",\n    \"num_hidden_layers\": 6", PlanError::WrongType { field: "hidden_size", expected: "unsigned integer" }),
        ("\"eos_token_id\": 29,", "", PlanError::Missing { field: "eos_token_id" }),
        ("\"eos_token_id\": 29,", "\"eos_token_id\": 30,", invalid("eos_token_id", "empty, repeated or outside the head rows")),
        ("\"eos_token_id\": 29,", "\"eos_token_id\": [29, 29],", invalid("eos_token_id", "empty, repeated or outside the head rows")),
        ("\"unpadded_vocab_size\": 30", "\"unpadded_vocab_size\": 33", invalid("unpadded_vocab_size", limit)),
        ("\"num_key_value_heads\": 2,\n    \"head_dim\": 4,\n    \"d_rel\"", "\"num_key_value_heads\": 3,\n    \"head_dim\": 4,\n    \"d_rel\"", invalid("num_attention_heads", "key heads do not divide query heads")),
        ("\"swa_num_key_value_heads\": 1", "\"swa_num_key_value_heads\": 0", invalid("swa_num_key_value_heads", limit)),
        ("\"sliding_window_size\": 5", "\"sliding_window_size\": 0", invalid("sliding_window_size", limit)),
        ("\"qk_l2_clip_threshold\": 5.5", "\"qk_l2_clip_threshold\": 0.0", invalid("qk_l2_clip_threshold", "must be positive")),
        ("\"mlp_up_softcap\": 25.0", "\"mlp_up_softcap\": -1", invalid("mlp_up_softcap", "must be positive")),
        ("\"n_shared_experts\": 0", "\"n_shared_experts\": 1", invalid("n_shared_experts", "this schema has no shared expert")),
        ("\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": 5", invalid("num_experts_per_tok", limit)),
        ("      0,\n      2,\n      4\n", "      0,\n      2,\n      6\n", invalid("local_layer_ids", "not rising or outside its limit")),
        ("      0,\n      2,\n      4\n", "      2,\n      0,\n      4\n", invalid("local_layer_ids", "not rising or outside its limit")),
        ("\"n_layers_per_attn_res_block\": 2", "\"n_layers_per_attn_res_block\": 7", invalid("n_layers_per_attn_res_block", limit)),
        ("\"core_start\": 2", "\"core_start\": 3", invalid("loop_config", "the loop range does not start and end on a block")),
        ("\"core_end\": 4", "\"core_end\": 2", invalid("loop_config", "the loop range is empty")),
        ("\"core_start\": 2", "\"core_start\": 0", invalid("core_start", limit)),
        ("\"core_end\": 4", "\"core_end\": 8", invalid("core_end", limit)),
        ("\"num_passes\": 3", "\"num_passes\": 0", invalid("num_passes", limit)),
        ("\"hidden_size\": 16,\n    \"num_hidden_layers\": 2", "\"hidden_size\": 8,\n    \"num_hidden_layers\": 2", invalid("mtp_config", "hidden size differs from the text model")),
        ("      10\n", "      11\n", invalid("target_layer_ids", "not rising or outside its limit")),
    ] {
        assert_eq!(config_refusal(from, to), want, "{from} -> {to}");
    }
    // Each refusal names its field in the text that a person reads.
    assert_eq!(config_refusal("\"d_rel\": 2,", "\"d_rel\": 2, \"new_switch\": true,").to_string(), "unsupported text_config: unknown field new_switch");
    assert_eq!(config_refusal("\"q_bias\": false", "\"q_bias\": true").to_string(), "unsupported q_bias: true");
}

#[test]
fn sidecar_gives_groups_and_the_storage_map() {
    let (d, g) = (description(), groups());
    assert_eq!((g.method.as_str(), g.counts()), ("inkling2", vec![2, 3, 8, 4, 1]));
    assert_eq!(g.model.iter().map(|g| g.kind).collect::<Vec<_>>(), [GroupKind::Plain { dtype: Dtype::F32, forward: Dtype::F32 },
        GroupKind::Plain { dtype: Dtype::Bf16, forward: Dtype::Bf16 }, GroupKind::Nvfp4, GroupKind::Mxfp8,
        GroupKind::Plain { dtype: Dtype::Bf16, forward: Dtype::F32 }]);
    assert_eq!(g.model.iter().map(|g| g.kind.stored_dtype()).collect::<Vec<_>>(), [Dtype::F32, Dtype::Bf16, Dtype::U8, Dtype::F8E4M3, Dtype::Bf16]);
    assert_eq!(g.draft.len(), 1);
    assert_eq!((g.draft[0].kind.stored_dtype(), g.draft[0].tensors.as_slice()), (Dtype::Bf16, ["fuse_weight".to_owned(), "norm.weight".to_owned()].as_slice()));
    assert_eq!(g.group_of("model.llm.layers.5.mlp.experts.w2_weight").unwrap().kind, GroupKind::Mxfp8);
    assert_eq!(g.group_of("model.llm.unembed.weight").unwrap().tensors.len(), 1);
    // A scale tensor and a draft name are not listed in the main groups.
    assert!(g.group_of("model.llm.layers.5.mlp.experts.w2_weight_scale").is_none() && g.group_of("fuse_weight").is_none());

    assert_eq!(g.layer_storage(&d).unwrap(), [NVFP4, NVFP4, NVFP4, NVFP4, WeightStorage::Mxfp8, WeightStorage::Mxfp8]);
    let map = g.storage_map(&d).unwrap();
    assert_eq!((map.layers.len(), map.excluded.len()), (12, 0));
    assert_eq!(map.storage("model.llm.layers.0.mlp.experts.w13_weight"), NVFP4);
    assert_eq!(map.storage("model.llm.layers.4.mlp.experts.w2_weight"), WeightStorage::Mxfp8);
    for plain in ["model.llm.layers.0.mlp.experts.w13_weight_scale", "model.llm.layers.0.mlp.experts.w13_weight_amax",
        "model.llm.layers.0.mlp.gate.bias", "model.llm.unembed.weight", "model.visual.backbone.layers.0.ln_1.weight"] {
        assert_eq!(map.storage(plain), WeightStorage::Plain, "{plain}");
    }

    // 4 experts, 16 rows, 16 values a row stored as 8 bytes: one scale for the 16 values.
    let name = "model.llm.layers.0.mlp.experts.w13_weight";
    let companion = |suffix: &str, dtype, shape: &[u64]| Companion { name: format!("{name}{suffix}"), dtype, shape: shape.to_vec() };
    assert_eq!(GroupKind::Nvfp4.companions(name, &[4, 16, 8]).unwrap(), [companion("_scale", Dtype::F8E4M3, &[4, 16, 1]),
        companion("_amax", Dtype::F32, &[4]), companion("_outer_scale_denominator", Dtype::F32, &[1])]);
    // 64 values a row stored as 64 bytes: one scale for each 32 values of one row.
    assert_eq!(GroupKind::Mxfp8.companions(name, &[4, 16, 64]).unwrap(), [companion("_scale", Dtype::U8, &[4, 16, 2])]);
    assert_eq!(GroupKind::Plain { dtype: Dtype::Bf16, forward: Dtype::Bf16 }.companions(name, &[16, 16]).unwrap(), []);
    for (kind, shape) in [(GroupKind::Nvfp4, vec![16, 8]), (GroupKind::Nvfp4, vec![4, 16, 4]), (GroupKind::Nvfp4, vec![4, 16, 0]),
        (GroupKind::Mxfp8, vec![4, 16, 48]), (GroupKind::Mxfp8, vec![4, 16, 2, 32])] {
        assert!(kind.companions(name, &shape).is_err(), "{kind:?} {shape:?}");
    }
    assert_eq!((NVFP4_NIBBLES, GATE_UP_ROWS), (NibbleOrder::LowFirst, ExpertRows::Interleaved));
    // The row layout of the 8-bit block has two spellings in the field.
    assert_eq!(StorageGroups::from_json(edit(SIDECAR, "32,\n          32\n", "1,\n          32\n").as_bytes()).unwrap(), g);
}

#[test]
fn sidecar_refusals_name_the_field() {
    for (from, to, want) in [
        ("\"quant_method\": \"inkling2\"", "\"quant_method\": \"modelopt\"", "unsupported sidecar quant_method modelopt"),
        ("\"quant_method\": \"inkling2\"", "\"quant_method\": \"inkling2\", \"kv_cache\": null", "unknown sidecar field kv_cache"),
        ("\"compute_dtype\": \"mxfp8\"", "\"compute_dtype\": \"mxfp8\", \"new_rule\": 1", "unknown storage group field new_rule"),
        ("\"compute_dtype\": \"float32\"", "\"compute_dtype\": \"float16\"", "unsupported storage group: compute_dtype float16, output_dtype float32, weight_transform cast"),
        ("\"weight_transform\": \"nvfp4_nearest\"", "\"weight_transform\": \"nvfp4_floor\"", "unsupported storage group: compute_dtype nvfp4, output_dtype uint8, weight_transform nvfp4_floor"),
        ("\"fwd_output_dtype\": \"float32\"", "\"fwd_output_dtype\": \"float64\"", "unsupported fwd_output_dtype float64"),
        ("\"weight_scale_format\": \"amax_denominator\"", "\"weight_scale_format\": \"direct\"", "unsupported 4-bit block, scale format or forward type"),
        ("1,\n          16\n", "1,\n          32\n", "unsupported 4-bit block, scale format or forward type"),
        ("32,\n          32\n", "32,\n          16\n", "unsupported 8-bit block, scale format or forward type"),
        ("\"scale_selection_rule\": \"fouroversix_mae\"", "\"scale_selection_rule\": \"max\"", "unsupported activation scale_selection_rule max"),
        ("\"amax_type\": \"per_token\"", "\"amax_type\": \"per_batch\"", "unsupported activation amax_type per_batch"),
        ("\"scale_dtype\": \"ue8m0\"", "\"scale_dtype\": \"ue8m0\", \"calibration\": 1", "unknown activation field calibration"),
        ("\"block_size\": 32", "\"block_size\": 16", "unsupported activation block_size"),
        ("\"model.llm.unembed.weight\"", "\"model.llm.embed.weight\"", "tensor model.llm.embed.weight is in two storage groups"),
        ("\"model.llm.unembed.weight\"", "\"\"", "a storage group has no tensor or an empty name"),
    ] {
        assert_eq!(sidecar_refusal(from, to), want, "{from} -> {to}");
    }
    let d = description();
    let map_refusal = |g: StorageGroups, d: &LoopTextDescription| g.storage_map(d).unwrap_err().to_string();
    let changed = |from: &str, to: &str| StorageGroups::from_json(edit(SIDECAR, from, to).as_bytes()).unwrap();
    // The cross-check with config.json.
    assert_eq!(map_refusal(groups(), &LoopTextDescription { declared_storage: None, ..d.clone() }),
        "the sidecar has quantized groups and config.json declares no storage");
    assert_eq!(map_refusal(groups(), &LoopTextDescription { declared_storage: Some(DeclaredStorage { method: "modelopt".into(), algo: "FP8".into() }), ..d.clone() }),
        "unsupported declared storage: quant_method modelopt, quant_algo FP8");
    assert_eq!(map_refusal(changed("\"model.llm.layers.5.mlp.experts.w2_weight\"", "\"model.llm.layers.6.mlp.experts.w2_weight\""), &d),
        "quantized tensor model.llm.layers.6.mlp.experts.w2_weight names no layer of config.json");
    assert_eq!(map_refusal(changed("\"model.llm.layers.5.mlp.experts.w2_weight\"", "\"model.llm.head.w2_weight\""), &d),
        "quantized tensor model.llm.head.w2_weight names no layer of config.json");
    assert_eq!(map_refusal(changed("\"model.llm.layers.5.mlp.experts.w2_weight\"", "\"model.llm.layers.3.mlp.experts.w3_weight\""), &d),
        "layer 3 has tensors in two storage formats");
    assert_eq!(map_refusal(changed("\"model.llm.layers.5.mlp.experts.w2_weight\"", "\"model.llm.layers.4.mlp.experts.w3_weight\""), &d),
        "layers differ in their count of quantized tensors");
    // A sidecar with no 4-bit group does not fit a config that declares 4-bit storage.
    let mut plain = groups();
    plain.model.retain(|g| g.kind != GroupKind::Nvfp4);
    assert_eq!(map_refusal(plain.clone(), &d), "config.json declares NVFP4 storage and the sidecar has no 4-bit group");
    // With no quantized group and no declared storage the map is empty and every layer is plain.
    plain.model.retain(|g| g.kind != GroupKind::Mxfp8);
    let bare = LoopTextDescription { declared_storage: None, ..d };
    assert!(plain.storage_map(&bare).unwrap().layers.is_empty());
    assert_eq!(plain.layer_storage(&bare).unwrap(), [WeightStorage::Plain; 6]);
}

#[test]
fn folder_read_cross_checks_both_files() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/loop-plan");
    let (d, g, map) = read_folder(&root).unwrap();
    assert_eq!((d, &g, map.layers.len()), (description(), &groups(), 12));
    assert_eq!(map, g.storage_map(&description()).unwrap());
    let missing = read_folder(&root.join("absent")).unwrap_err().to_string();
    assert!(!missing.is_empty());
}

/// Gate of slice S1, parts (a) and (b), on the released checkpoint. No payload byte is read.
/// Pass the model folder as a test argument:
/// `cargo test -p engine-model --test loop_plan -- --ignored --nocapture real_folder_gate <folder>`
#[test]
#[ignore = "skipped: no checkpoint; pass the model folder as a test argument"]
fn real_folder_gate() {
    let root = std::env::args().skip(1).map(std::path::PathBuf::from).find(|p| p.join("config.json").is_file())
        .expect("pass the model folder as a test argument");
    let started = std::time::Instant::now();
    let (d, g, map) = read_folder(&root).unwrap();
    let parse_time = started.elapsed();

    // The visit list: 48 entries in the order of SPEC 3.2, with the state slots of SPEC 3.6.
    let mut want = (0..6).map(|l| (l, 0, l, false)).collect::<Vec<_>>();
    want.extend((6..18).map(|l| (l, 1, l, false)));
    want.extend((6..18).map(|l| (l, 2, l + 18, true)));
    want.extend((6..18).map(|l| (l, 3, l + 30, true)));
    want.extend((18..24).map(|l| (l, 0, l, false)));
    assert_eq!(order(&d), want);
    assert_eq!(d.visits.len(), 48);
    assert_eq!((d.full_layers(), d.loop_layers(), d.blocks()), (vec![5, 11, 17, 23], 6..18, 4));

    // The group counts of the sidecar.
    assert_eq!(g.counts(), [74, 993, 40, 8, 1]);
    let layer_storage = g.layer_storage(&d).unwrap();
    assert_eq!(layer_storage.iter().filter(|&&s| s == NVFP4).count() + layer_storage.iter().filter(|&&s| s == WeightStorage::Mxfp8).count(), layer_storage.len());

    // Every tensor of the file is a listed name or a scale tensor of a listed name.
    let limits = Limits { header_bytes: HEADER_BYTES, tensors: MAX_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS };
    let index = Index::open(&root.join("model.safetensors.index.json"), limits).unwrap();
    let shards = index.inspect_shards(&root, limits).unwrap();
    let tensors: BTreeMap<&String, &Tensor> = shards.values().flat_map(|h| h.tensors.iter()).collect();
    let (mut bound, mut companions, mut bytes) = (BTreeSet::new(), 0usize, vec![0u64; g.model.len()]);
    for (group, total) in g.model.iter().zip(&mut bytes) {
        for name in &group.tensors {
            let t = tensors.get(name).unwrap_or_else(|| panic!("listed tensor {name} is not in the file"));
            assert_eq!(t.dtype, group.kind.stored_dtype(), "{name}");
            assert_eq!(map.storage(name), group.kind.storage(), "{name}");
            assert!(bound.insert(name.clone()), "{name} is bound two times");
            *total += t.byte_len();
            for c in group.kind.companions(name, &t.shape).unwrap() {
                let s = tensors.get(&c.name).unwrap_or_else(|| panic!("scale tensor {} is not in the file", c.name));
                assert_eq!((s.dtype, &s.shape), (c.dtype, &c.shape), "{}", c.name);
                assert!(bound.insert(c.name.clone()), "{} is bound two times", c.name);
                *total += s.byte_len();
                companions += 1;
            }
        }
    }
    let left = tensors.keys().filter(|name| !bound.contains(**name)).count();
    let payload = tensors.values().map(|t| t.byte_len()).sum::<u64>();
    assert_eq!((left, bound.len(), tensors.len()), (0, 1244, 1244));
    assert_eq!((payload, bytes.iter().sum::<u64>(), index.total_size), (19_525_870_752, 19_525_870_752, 19_525_870_752));
    assert_eq!(g.draft.iter().map(|g| g.tensors.len()).collect::<Vec<_>>(), [54]);

    println!("GATE visits={} order=spec-3.2 full_layers={:?} loop_layers={:?} passes={}", d.visits.len(), d.full_layers(), d.loop_layers(), d.core.passes);
    println!("GATE group_counts={:?} listed={} companions={} bound={} left={} file_tensors={}", g.counts(),
        g.counts().iter().sum::<usize>(), companions, bound.len(), left, tensors.len());
    println!("GATE group_bytes={bytes:?} payload_sum={payload} index_total_size={}", index.total_size);
    println!("GATE layer_storage nvfp4={} mxfp8={} draft_listed={}", layer_storage.iter().filter(|&&s| s == NVFP4).count(),
        layer_storage.iter().filter(|&&s| s == WeightStorage::Mxfp8).count(), g.draft[0].tensors.len());
    println!("GATE first_parse config_and_sidecar_seconds={:.6} all_seconds={:.6}", parse_time.as_secs_f64(), started.elapsed().as_secs_f64());
}
