//! Tensor roles, shape and type checks and refusal texts for the looped text family, on a made-up fixture.
use engine_model::{
    cartridge::Cartridge, json, plan::{self, PlanError, description::LoopTextDescription, inkling_index::*, storage_groups::StorageGroups},
    safetensors::{Dtype, Header, Limits, Tensor}, shapes::{Ffn, Place, Proj},
};
use std::{collections::BTreeMap, io::{self, Cursor, Read, Seek, SeekFrom}, path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering}};

const CONFIG: &str = include_str!("fixtures/inkling-index/config.json");
const SIDECAR: &str = include_str!("fixtures/inkling-index/hf_quant_config.json");
const HEADER: &str = include_str!("fixtures/inkling-index/header.json");
const DRAFT: &str = include_str!("fixtures/inkling-index/draft-header.json");
const SHARD: &str = "model-00001-of-00001.safetensors";
const P: &str = "model.llm.";

// Only the length prefix and the header exist. A payload read fails the test.
struct HeaderOnly { metadata: Cursor<Vec<u8>>, file_bytes: u64, bytes_read: usize }
impl Read for HeaderOnly {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.metadata.position() >= self.metadata.get_ref().len() as u64 { return Err(io::Error::other("test attempted to read tensor payload")); }
        let read = self.metadata.read(out)?; self.bytes_read += read; Ok(read)
    }
}
impl Seek for HeaderOnly {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from { SeekFrom::Start(n) => n as i128, SeekFrom::End(n) => self.file_bytes as i128 + n as i128,
            SeekFrom::Current(n) => self.metadata.position() as i128 + n as i128 };
        let position = u64::try_from(position).map_err(|_| io::Error::other("invalid test seek"))?;
        self.metadata.set_position(position); Ok(position)
    }
}
fn payload(header: &str) -> u64 {
    let doc = json::parse(header.as_bytes(), json::Limits::default()).unwrap();
    doc.object().unwrap().values().map(|v| v.object().unwrap()["data_offsets"].array().unwrap()[1].u64().unwrap()).max().unwrap()
}
fn headers(header: &str) -> BTreeMap<String, Tensor> {
    let mut data = (header.len() as u64).to_le_bytes().to_vec(); data.extend_from_slice(header.as_bytes());
    let file_bytes = data.len() as u64 + payload(header);
    let mut reader = HeaderOnly { metadata: Cursor::new(data), file_bytes, bytes_read: 0 };
    let read = Header::read(&mut reader, Limits::default()).unwrap();
    assert_eq!(reader.bytes_read, header.len() + 8);
    read.tensors
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Folder(PathBuf);
impl Drop for Folder { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tensor_file(path: &Path, header: &str) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap(); file.write_all(header.as_bytes()).unwrap();
    // The payload is a hole in the file. No test reads it.
    file.set_len(8 + header.len() as u64 + payload(header)).unwrap();
}
/// A checkpoint folder with headers and no weights. `draft` adds the draft file.
fn folder(config: &str, sidecar: &str, header: &str, draft: Option<&str>) -> Folder {
    let root = std::env::temp_dir().join(format!("vade-inkling-index-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("config.json"), config).unwrap();
    std::fs::write(root.join("hf_quant_config.json"), sidecar).unwrap();
    let doc = json::parse(header.as_bytes(), json::Limits::default()).unwrap();
    let map = doc.object().unwrap().keys().map(|n| format!("{n:?}:{SHARD:?}")).collect::<Vec<_>>().join(",");
    std::fs::write(root.join("model.safetensors.index.json"), format!(r#"{{"metadata":{{"total_size":{}}},"weight_map":{{{map}}}}}"#, payload(header))).unwrap();
    tensor_file(&root.join(SHARD), header);
    if let Some(draft) = draft { tensor_file(&root.join(DRAFT_FILE), draft); }
    Folder(root)
}
fn fixture() -> Folder { folder(CONFIG, SIDECAR, HEADER, Some(DRAFT)) }
fn input() -> IndexInput { read_input(&fixture().0).unwrap() }
fn bind(main: &BTreeMap<String, Tensor>) -> Result<BoundIndex, PlanError> { input().bind(main, Some(&headers(DRAFT))) }
fn unsupported(error: PlanError, field: &str, part: &str) {
    let PlanError::Unsupported { field: f, value } = &error else { panic!("{error:?}") };
    assert!(*f == field && value.contains(part), "{error:?}");
}

#[test]
fn every_fixture_name_has_a_role_and_no_payload_is_read() {
    let bound = inspect(&fixture().0).unwrap();
    let c = bound.counts();
    assert_eq!((c.stored, c.bound, c.text, c.unused, c.draft_stored, c.draft_bound), (292, 292, 286, 6, 39, 39));
    assert_eq!((c.stored_bytes, c.draft_bytes), (1_776_088, 388_748));
    assert!(bound.draft_declared && bound.draft.values().all(|t| t.state == TensorState::NotRun && matches!(t.place, Place::Draft { .. })));
    // The same headers through a reader that fails on a payload read.
    assert_eq!(bind(&headers(HEADER)).unwrap(), bound);
    let t = |name: &str| bound.tensors[&format!("{P}{name}")].clone();
    let role = |name: &str| { let t = t(name); (t.role, t.place, t.dtype, t.shape) };
    // Layers 0, 2, 4 and 6 are window layers with their own head sizes and the window as relative extent.
    assert_eq!(role("layers.0.attn.wq_du.weight"), (StoredRole::Query, Place::Layer(0), Dtype::Bf16, vec![56, 96]));
    assert_eq!(role("layers.1.attn.wq_du.weight"), (StoredRole::Query, Place::Layer(1), Dtype::Bf16, vec![132, 96]));
    assert_eq!(role("layers.0.attn.rel_logits_proj.proj").3, vec![10, 36]);
    assert_eq!(role("layers.1.attn.rel_logits_proj.proj").3, vec![10, 52]);
    assert_eq!(role("layers.0.attn.k_sconv.weight"), (StoredRole::ConvFixed(Stream::Key), Place::Layer(0), Dtype::Bf16, vec![28, 1, 5]));
    assert_eq!(role("layers.1.attn.v_sconv.static_w"), (StoredRole::ConvStatic(Stream::Value), Place::Layer(1), Dtype::Bf16, vec![5, 66]));
    assert_eq!(role("layers.1.attn.v_sconv.weight_proj").3, vec![15, 96]);
    // `attn_queries.i` is the attention input of layer `i + 1`: three rows for a full layer, one for a window layer.
    assert_eq!(role("attn_res.attn_queries.0"), (StoredRole::MixAttention, Place::Layer(1), Dtype::F32, vec![3, 96]));
    assert_eq!(role("attn_res.attn_queries.1"), (StoredRole::MixAttention, Place::Layer(2), Dtype::F32, vec![96]));
    assert!(!bound.tensors.contains_key(&format!("{P}attn_res.attn_queries.7")));
    assert_eq!(role("attn_res.mlp_queries.7").1, Place::Layer(7));
    // Layers 2 to 5 loop: two message query rows a layer, and four gates of rank 9.
    assert_eq!(role("middle_cycle_cross_pass_kv_queries"), (StoredRole::MessageKvQueries, Place::Model, Dtype::F32, vec![8, 96]));
    assert_eq!(role("layers.2.controllers.mlp.down_weight"),
        (StoredRole::Gate { gate: MessageGate::Expert, part: GatePart::Down }, Place::Layer(2), Dtype::Bf16, vec![9, 192]));
    assert!(!bound.tensors.contains_key(&format!("{P}layers.1.controllers.mlp.down_weight")));
    // 4-bit stacks in layers 0 to 5: two codes a byte, one scale for 16 values, a factor for each expert, one divisor.
    let expert = |proj, part| StoredRole::Expert { proj, part };
    assert_eq!(role("layers.0.mlp.experts.w13_weight"), (expert(Proj::GateUp, ExpertPart::Codes), Place::Layer(0), Dtype::U8, vec![7, 128, 48]));
    assert_eq!(role("layers.0.mlp.experts.w13_weight_scale"), (expert(Proj::GateUp, ExpertPart::BlockScale), Place::Layer(0), Dtype::F8E4M3, vec![7, 128, 6]));
    assert_eq!(role("layers.0.mlp.experts.w2_weight_amax"), (expert(Proj::Down, ExpertPart::ExpertScale), Place::Layer(0), Dtype::F32, vec![7]));
    assert_eq!(role("layers.0.mlp.experts.w2_weight_outer_scale_denominator").3, vec![1]);
    // 8-bit stacks in layers 6 and 7: one value a weight and one exponent for 32 values of a row.
    assert_eq!(role("layers.7.mlp.experts.w2_weight"), (expert(Proj::Down, ExpertPart::Codes), Place::Layer(7), Dtype::F8E4M3, vec![7, 96, 64]));
    assert_eq!(role("layers.7.mlp.experts.w2_weight_scale"), (expert(Proj::Down, ExpertPart::BlockScale), Place::Layer(7), Dtype::U8, vec![7, 96, 2]));
    assert_eq!(role("layers.3.mlp.gate.bias"), (StoredRole::RouterBias, Place::Layer(3), Dtype::F32, vec![7]));
    assert_eq!(role("unembed.weight"), (StoredRole::Head, Place::Model, Dtype::Bf16, vec![320, 96]));
    let vision = &bound.tensors["model.visual.proj_out.weight"];
    assert_eq!((vision.role, vision.place, vision.state), (StoredRole::Vision, Place::Unused, TensorState::Unused));
    let draft = |name: &str| { let t = &bound.draft[name]; (t.role, t.place, t.shape.clone()) };
    assert_eq!(draft("fuse_weight"), (StoredRole::Draft(DraftRole::Fuse), Place::Draft { layer: None }, vec![3, 4]));
    assert_eq!(draft("markov_head.rnn_proj.weight").2, vec![42, 220]);
    assert_eq!(draft("layers.1.mlp.down_proj.weight"), (StoredRole::Draft(DraftRole::Down), Place::Draft { layer: Some(1) }, vec![96, 176]));
}

#[test]
fn expected_names_equal_the_fixture_headers() {
    let input = input();
    let (main, draft) = (headers(HEADER), headers(DRAFT));
    let expected = input.expected_main().unwrap();
    assert_eq!(expected.keys().collect::<Vec<_>>(), main.keys().filter(|n| !n.starts_with("model.visual.")).collect::<Vec<_>>());
    assert!(expected.iter().all(|(n, e)| (e.dtype, &e.shape) == (main[n].dtype, &main[n].shape)));
    let expected = input.expected_draft().unwrap();
    assert_eq!(expected.keys().collect::<Vec<_>>(), draft.keys().collect::<Vec<_>>());
    assert!(expected.iter().all(|(n, e)| (e.dtype, &e.shape) == (draft[n].dtype, &draft[n].shape)));
}

#[test]
fn refuses_an_unknown_tensor_name() {
    let mut main = headers(HEADER);
    main.insert(format!("{P}layers.0.attn.new_part.weight"), Tensor { dtype: Dtype::Bf16, shape: vec![1], start: 0, end: 2 });
    assert_eq!(bind(&main).unwrap_err(), PlanError::UnexpectedTensor { name: format!("{P}layers.0.attn.new_part.weight") });
    // A layer that the config does not have.
    let mut main = headers(HEADER);
    let layer = main[&format!("{P}layers.7.mlp_norm.weight")].clone();
    main.insert(format!("{P}layers.8.mlp_norm.weight"), layer);
    assert_eq!(bind(&main).unwrap_err(), PlanError::UnexpectedTensor { name: format!("{P}layers.8.mlp_norm.weight") });
    // A message gate outside the loop range, and a per-expert factor on an 8-bit stack.
    for (copy, name) in [("layers.2.controllers.k.gain_weight", "layers.6.controllers.k.gain_weight"),
        ("layers.0.mlp.experts.w2_weight_amax", "layers.7.mlp.experts.w2_weight_amax")] {
        let mut main = headers(HEADER);
        let t = main[&format!("{P}{copy}")].clone();
        main.insert(format!("{P}{name}"), t);
        assert_eq!(bind(&main).unwrap_err(), PlanError::UnexpectedTensor { name: format!("{P}{name}") });
    }
}

#[test]
fn refuses_a_missing_tensor() {
    for name in ["embed_norm.weight", "attn_res.attn_queries.6", "layers.5.mlp.experts.w13_weight_outer_scale_denominator", "layers.1.q_input_norm.weight"] {
        let mut main = headers(HEADER);
        main.remove(&format!("{P}{name}")).unwrap();
        assert_eq!(bind(&main).unwrap_err(), PlanError::MissingTensor { name: format!("{P}{name}") });
    }
    // The image tower is not needed. A text-only file binds.
    let mut main = headers(HEADER);
    main.retain(|name, _| !name.starts_with("model.visual."));
    let bound = bind(&main).unwrap();
    assert_eq!((bound.counts().unused, bound.counts().text), (0, 286));
}

#[test]
fn refuses_a_wrong_shape() {
    // Each case swaps in a size that is right for another tensor of the same kind.
    for (name, shape, expected) in [
        ("layers.0.attn.wq_du.weight", vec![132, 96], vec![56, 96]),
        ("layers.0.attn.rel_logits_proj.proj", vec![10, 52], vec![10, 36]),
        ("attn_res.attn_queries.1", vec![3, 96], vec![96]),
        ("layers.6.mlp.experts.w13_weight_scale", vec![7, 128, 6], vec![7, 128, 3]),
        ("layers.1.attn.k_sconv.static_w", vec![66, 5], vec![5, 66]),
    ] {
        let mut main = headers(HEADER);
        let t = main.get_mut(&format!("{P}{name}")).unwrap();
        t.shape = shape.clone(); t.end = t.start + shape.iter().product::<u64>() * t.dtype.bytes();
        assert_eq!(bind(&main).unwrap_err(), PlanError::TensorMismatch { name: format!("{P}{name}"), expected, actual: shape });
    }
}

#[test]
fn refuses_a_wrong_type() {
    for (name, dtype, text) in [("layers.0.mlp.gate.bias", Dtype::Bf16, "expected F32, actual Bf16"),
        ("layers.0.mlp.experts.w13_weight", Dtype::F8E4M3, "expected U8, actual F8E4M3"),
        ("layers.7.mlp.experts.w2_weight_scale", Dtype::F8E4M3, "expected U8, actual F8E4M3"), ("norm.weight", Dtype::F16, "expected Bf16, actual F16")] {
        let mut main = headers(HEADER);
        let t = main.get_mut(&format!("{P}{name}")).unwrap();
        t.end = t.start + (t.end - t.start) / t.dtype.bytes() * dtype.bytes(); t.dtype = dtype;
        unsupported(bind(&main).unwrap_err(), "tensor dtype", &format!("{P}{name}: {text}"));
    }
    // Offsets that differ from shape and type.
    let mut main = headers(HEADER);
    main.get_mut(&format!("{P}norm.weight")).unwrap().end += 2;
    unsupported(bind(&main).unwrap_err(), "tensor byte extent", "norm.weight");
}

#[test]
fn refuses_storage_that_differs_from_the_roles() {
    let stack = format!("{P}layers.0.mlp.experts.w13_weight");
    // An expert stack in a plain group, a plain tensor in a packed group, a tensor in no group, and a listed name with no role.
    let mut plain_stack = input(); plain_stack.stored.insert(stack.clone(), Stored::Bf16);
    unsupported(plain_stack.expected_main().unwrap_err(), "tensor storage", "Bf16 storage for an expert stack");
    let mut packed_norm = input(); packed_norm.stored.insert(format!("{P}norm.weight"), Stored::Nvfp4);
    unsupported(packed_norm.expected_main().unwrap_err(), "tensor storage", "Nvfp4 storage for a tensor that is not an expert stack");
    let mut unlisted = input(); unlisted.stored.remove(&stack);
    unsupported(unlisted.expected_main().unwrap_err(), "tensor storage", "the storage sidecar does not list this tensor");
    let mut stray = input(); stray.stored.insert(format!("{P}layers.9.mlp_norm.weight"), Stored::Bf16);
    unsupported(stray.expected_main().unwrap_err(), "storage sidecar", "layers.9.mlp_norm.weight: the config gives this listed tensor no role");
    // A stack whose columns do not fill complete scale blocks.
    let mut ragged = input(); ragged.layers[0].expert_width = 40;
    assert_eq!(ragged.expected_main().unwrap_err(), PlanError::Invalid { field: "expert columns", reason: "columns do not fill complete scale blocks" });
    // The 8-bit form with the 4-bit header is a type refusal at the scale tensor or at the codes.
    let mut swapped = input(); swapped.stored.insert(stack, Stored::Mxfp8);
    assert!(matches!(swapped.bind(&headers(HEADER), None).unwrap_err(), PlanError::UnexpectedTensor { .. } | PlanError::Unsupported { field: "tensor dtype", .. }));
}

#[test]
fn draft_names_are_bound_and_not_run() {
    let input = input();
    let main = headers(HEADER);
    // The draft file can be absent. The text names still bind.
    let bound = input.bind(&main, None).unwrap();
    assert!(bound.draft.is_empty() && bound.draft_declared && !bound.operations.contains(&Operation::Draft));
    assert!(bound.reason("x").contains("the draft file is absent"));
    let mut draft = headers(DRAFT);
    draft.insert("layers.0.new_part.weight".into(), Tensor { dtype: Dtype::Bf16, shape: vec![1], start: 0, end: 2 });
    assert_eq!(input.bind(&main, Some(&draft)).unwrap_err(), PlanError::UnexpectedTensor { name: format!("{DRAFT_FILE}: layers.0.new_part.weight") });
    let mut draft = headers(DRAFT);
    draft.remove("mask_embed").unwrap();
    assert_eq!(input.bind(&main, Some(&draft)).unwrap_err(), PlanError::MissingTensor { name: format!("{DRAFT_FILE}: mask_embed") });
    let mut draft = headers(DRAFT);
    let t = draft.get_mut("fuse_weight").unwrap(); t.shape = vec![4, 3];
    assert_eq!(input.bind(&main, Some(&draft)).unwrap_err(), PlanError::TensorMismatch { name: "fuse_weight".into(), expected: vec![3, 4], actual: vec![4, 3] });
    // A draft file with no `mtp_config` is refused.
    let mut none = input.clone(); none.draft = None;
    assert_eq!(none.bind(&main, Some(&headers(DRAFT))).unwrap_err(),
        PlanError::Invalid { field: "mtp_config", reason: "a draft file is stored but the config declares no draft model" });
    assert!(none.expected_draft().unwrap().is_empty());
}

#[test]
fn execution_is_refused_with_the_missing_operations() {
    let bound = inspect(&fixture().0).unwrap();
    let PlanError::MissingOps { ops } = bound.require_execution().unwrap_err() else { panic!() };
    // First the operations that the stored tensors need, then the parts of a service that no tensor shows.
    assert_eq!(ops, ["embedding norm", "window attention", "full attention", "relative position bias", "attention sink", "attention output gate",
        "short convolutions", "block mix", "loop messages and gates", "router", "4-bit expert stacks", "8-bit expert stacks",
        "final norm and padded head", "draft model", "weight load", "state for each layer visit", "tokenizer split rule", "chat template",
        "stop set", "token-ID proof"]);
    assert_eq!(bound.missing_operations(), ops);
    // The list follows the file: no loop tensors, no loop entry; no 8-bit stack, no 8-bit entry.
    let mut flat = input();
    for l in &mut flat.layers { l.looped = false; }
    flat.stored.retain(|name, _| !name.contains(".controllers.") && !name.contains("middle_cycle_cross_pass"));
    for stored in flat.stored.values_mut() { if *stored == Stored::Mxfp8 { *stored = Stored::Nvfp4; } }
    let main = flat.expected_main().unwrap().into_iter().map(|(name, e)| {
        let bytes = e.shape.iter().product::<u64>() * e.dtype.bytes();
        (name, Tensor { dtype: e.dtype, shape: e.shape, start: 0, end: bytes })
    }).collect();
    let ops = flat.bind(&main, None).unwrap().missing_operations();
    assert!(!ops.iter().any(|o| matches!(o.as_str(), "loop messages and gates" | "8-bit expert stacks" | "draft model")), "{ops:?}");
    assert!(ops.contains(&"4-bit expert stacks".to_owned()));
}

#[test]
fn image_input_is_refused() {
    let bound = inspect(&fixture().0).unwrap();
    assert_eq!(bound.require_image_input().unwrap_err().to_string(),
        "unsupported image input: this engine runs the text part only; the 6 image tower tensors are bound as not used");
    let mut main = headers(HEADER);
    main.retain(|name, _| !name.starts_with("model.visual."));
    assert_eq!(bind(&main).unwrap().require_image_input().unwrap_err().to_string(),
        "unsupported image input: this engine runs text only and this model stores no image tower");
}

#[test]
fn the_reason_says_what_runs_what_is_missing_and_the_counts() {
    let root = fixture();
    let reason = plan::unserved_reason(&root.0, CONFIG.as_bytes()).unwrap();
    assert_eq!(reason, concat!("This engine runs this model only as a preview, and only with the start option --diagnostic-unqualified-text: ",
        "text chat runs; not built yet: image input and the draft model. It has not been through qualification. Model type inkling2_loop: ",
        "292 of 292 stored tensors have a role (286 text, 6 image tower and not used); 39 of 39 draft tensors are bound and not run."));
    assert!(!reason.contains('\n') && !reason.contains("Not built yet: embedding norm"),"the refusal must not list parts that run: {reason}");
    // Server start asks the same question first and gets the same text.
    assert_eq!(Cartridge::resident_text(&root.0).err().unwrap().to_string(), reason);
    // Another family keeps the caller's own text.
    for other in [r#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#, r#"{"model_type":"glm5_next"}"#, r#"{"model_type":7}"#, "{}", "[]", "not json"] {
        assert_eq!(plan::unserved_reason(&root.0, other.as_bytes()), None);
    }
}

#[test]
fn an_unknown_value_in_a_field_is_refused_with_the_field_and_the_value() {
    let reason = |config: &str, sidecar: &str, header: &str| {
        let root = folder(config, sidecar, header, Some(DRAFT));
        plan::unserved_reason(&root.0, config.as_bytes()).unwrap()
    };
    let start = "This engine cannot load this model. Model type inkling2_loop: ";
    // The storage sidecar: an unknown method and an unknown storage form.
    assert_eq!(reason(CONFIG, &SIDECAR.replace("\"inkling2\"", "\"other_method\""), HEADER),
        format!("{start}hf_quant_config.json: unsupported sidecar quant_method other_method."));
    assert_eq!(reason(CONFIG, &SIDECAR.replace("\"compute_dtype\": \"mxfp8\"", "\"compute_dtype\": \"mxfp6\""), HEADER),
        format!("{start}hf_quant_config.json: unsupported storage group: compute_dtype mxfp6, output_dtype float8_e4m3fn, weight_transform mxfp8_nearest."));
    // The config: an unknown value, an unknown field, a missing field, and a storage label that the sidecar does not hold.
    assert_eq!(reason(&CONFIG.replace("inkling2_loop_text", "other_text"), SIDECAR, HEADER), format!("{start}unsupported model_type: other_text."));
    assert_eq!(reason(&CONFIG.replace("\"is_varloop\": false", "\"is_varloop\": true"), SIDECAR, HEADER), format!("{start}unsupported is_varloop: true."));
    assert_eq!(reason(&CONFIG.replace("\"hidden_size\": 96,\n  \"num_hidden_layers\": 8", "\"new_field\": 1,\n  \"hidden_size\": 96,\n  \"num_hidden_layers\": 8"), SIDECAR, HEADER),
        format!("{start}unsupported text_config: unknown field new_field."));
    assert_eq!(reason(&CONFIG.replace("\"hidden_size\": 96,\n  \"num_hidden_layers\": 8", "\"num_hidden_layers\": 8"), SIDECAR, HEADER), format!("{start}missing field hidden_size."));
    assert_eq!(reason(&CONFIG.replace("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"NVFP2\""), SIDECAR, HEADER),
        format!("{start}config.json and hf_quant_config.json: unsupported declared storage: quant_method modelopt, quant_algo NVFP2."));
    // A value that the header reader does not know, and a tensor with no role.
    let name = "model.llm.norm.weight\": {\n  \"dtype\": \"BF16\"";
    assert!(HEADER.contains(name));
    let unknown = reason(CONFIG, SIDECAR, &HEADER.replace(name, "model.llm.norm.weight\": {\n  \"dtype\": \"BF8\""));
    assert!(unknown.starts_with(&format!("{start}safetensors headers: ")) && unknown.contains("BF8"), "{unknown}");
    let renamed = reason(CONFIG, SIDECAR, &HEADER.replace("model.llm.norm.weight", "model.llm.norm.gain"));
    assert_eq!(renamed, format!("{start}unexpected tensor model.llm.norm.gain."));
    // A missing file is named.
    let root = fixture();
    std::fs::remove_file(root.0.join("hf_quant_config.json")).unwrap();
    assert!(plan::unserved_reason(&root.0, CONFIG.as_bytes()).unwrap().starts_with(&format!("{start}hf_quant_config.json: ")));
}

#[test]
fn a_draft_file_outside_the_folder_is_refused() {
    let root = folder(CONFIG, SIDECAR, HEADER, None);
    let outside = folder(CONFIG, SIDECAR, HEADER, Some(DRAFT));
    std::os::unix::fs::symlink(outside.0.join(DRAFT_FILE), root.0.join(DRAFT_FILE)).unwrap();
    assert_eq!(inspect(&root.0).unwrap_err(), PlanError::Read { file: DRAFT_FILE, message: "symlink escapes checkpoint directory".into() });
}

/// The index works out the scale tensors of an expert stack from the config. The sidecar reader works them out
/// from the stored shape of the stack. Both must give the same names, types and shapes, with no name left over.
fn scale_tensors_agree(root: &Path, main: &BTreeMap<String, Tensor>) -> usize {
    let expected = read_input(root).unwrap().expected_main().unwrap();
    let groups = StorageGroups::from_json(&std::fs::read(root.join("hf_quant_config.json")).unwrap()).unwrap();
    let mut companions = BTreeMap::new();
    for group in &groups.model {
        for name in &group.tensors {
            for c in group.kind.companions(name, &main[name].shape).unwrap() { companions.insert(c.name, (c.dtype, c.shape)); }
        }
    }
    let listed = groups.model.iter().flat_map(|g| &g.tensors).collect::<std::collections::BTreeSet<_>>();
    let unlisted = expected.iter().filter(|(name, _)| !listed.contains(name)).map(|(name, e)| (name.clone(), (e.dtype, e.shape.clone()))).collect::<BTreeMap<_, _>>();
    assert_eq!(unlisted, companions);
    assert_eq!(listed.len() + companions.len(), main.len());
    companions.len()
}

#[test]
fn the_description_fills_the_index_input() {
    assert_eq!(scale_tensors_agree(&fixture().0, &headers(HEADER)), 40);
    let description = LoopTextDescription::from_config(CONFIG.as_bytes()).unwrap();
    let groups = StorageGroups::from_json(SIDECAR.as_bytes()).unwrap();
    let input = IndexInput::from_description(&description, &groups).unwrap();
    assert_eq!(input, read_input(&fixture().0).unwrap());
    assert_eq!((input.hidden, input.stored_vocab, input.conv_taps, input.gate_rank, input.layers.len(), input.stored.len()), (96, 320, 5, 9, 8, 252));
    // A window layer and a full layer carry the sizes of their own kind. Layer 0 stores no block-mix query for attention.
    let window = LayerSizes { query_heads: 4, kv_heads: 2, head_dim: 14, windowed: true, separate_inputs: false, mix_rows: 0, token_taps: false,
        relative_dim: 10, relative_extent: 36, kv_taps: 5, experts: 7, expert_width: 64, looped: false };
    let full = LayerSizes { query_heads: 6, kv_heads: 3, head_dim: 22, windowed: false, separate_inputs: true, mix_rows: 3, token_taps: true,
        relative_extent: 52, ..window };
    assert_eq!((input.layers[0], input.layers[1], input.layers[2], input.layers[5]),
        (window, full, LayerSizes { mix_rows: 1, looped: true, ..window }, LayerSizes { looped: true, ..full }));
    let draft = input.draft.clone().unwrap();
    assert_eq!((draft.hidden, draft.layers, draft.query_heads, draft.kv_heads, draft.head_dim, draft.width, draft.markov_rank, draft.taps, draft.stored.len()),
        (96, 2, 6, 2, 20, 176, 14, 4, 39));
    // A layer with no expert stack has no stored names in this schema.
    let mut dense = description.clone();
    dense.layers[3].ffn = Ffn::Dense { width: 64 };
    unsupported(IndexInput::from_description(&dense, &groups).unwrap_err(), "feed-forward", "a dense layer has no stored names");
    // The first layer cannot store a block-mix query for attention: `attn_queries.i` belongs to layer `i + 1`.
    let mut first = input.clone();
    first.layers[0].mix_rows = 1;
    assert_eq!(first.expected_main().unwrap_err(),
        PlanError::Invalid { field: "attention_queries", reason: "the first layer cannot store a block-mix query for attention" });
}

/// The real folder. Prints the gate numbers of the slice. The public model folder, or a link to it, must be
/// `.local/inkling-checkpoint` at the repository root. That folder is ignored and is not in the tracked tree.
/// `cargo test -p engine-model --test inkling_index -- --ignored --nocapture real_folder_binds_every_name`.
#[test]
#[ignore = "needs the public model folder in .local/inkling-checkpoint"]
fn real_folder_binds_every_name() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.local/inkling-checkpoint");
    assert!(root.join("config.json").is_file(), "put the public model folder, or a link to it, in .local/inkling-checkpoint for the ignored host test");
    let root = root.as_path();
    let started = std::time::Instant::now();
    let bound = inspect(root).unwrap();
    let elapsed = started.elapsed();
    let c = bound.counts();
    let header = Header::open(&root.join(SHARD), Limits { header_bytes: 64 << 20, tensors: 320_000, ..Limits::default() }).unwrap();
    let unmapped = header.tensors.keys().filter(|name| !bound.tensors.contains_key(*name)).count();
    println!("INKLING_INDEX names={} bound={} unmapped={unmapped} text={} unused={} stored_bytes={} draft_names={} draft_bound={} draft_bytes={} inspect_ms={:.1}",
        header.tensors.len(), c.bound, c.text, c.unused, c.stored_bytes, c.draft_stored, c.draft_bound, c.draft_bytes, elapsed.as_secs_f64() * 1000.0);
    println!("INKLING_REASON {}", plan::unserved_reason(root, &std::fs::read(root.join("config.json")).unwrap()).unwrap());
    assert_eq!((header.tensors.len(), c.bound, unmapped), (c.stored, c.stored, 0));
    assert_eq!((c.draft_bound, bound.draft.values().all(|t| t.state == TensorState::NotRun)), (c.draft_stored, true));
    assert!(bound.tensors.iter().all(|(name, t)| (t.state == TensorState::Unused) == name.starts_with("model.visual.")));
    assert!(bound.require_execution().is_err() && bound.require_image_input().is_err());
    println!("INKLING_SCALES names_from_config_and_from_stored_shapes_equal={}", scale_tensors_agree(root, &header.tensors));
}
