use engine_model::layout::LayoutReport;
use std::{path::PathBuf, sync::atomic::{AtomicU64, Ordering}};

fn fixture() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm-fp4-local") }
struct Copy { root: PathBuf }
impl Copy {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target")
            .join(format!("layout-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::create_dir(&root).unwrap();
        for entry in std::fs::read_dir(fixture()).unwrap() {
            let entry = entry.unwrap();
            if entry.path().extension().is_some_and(|v| v == "json") {
                std::fs::copy(entry.path(), root.join(entry.file_name())).unwrap();
            }
        }
        Self { root }
    }
    fn replace(&self, file: &str, from: &str, to: &str) {
        let path = self.root.join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(from), "missing test edit in {file}");
        std::fs::write(path, text.replace(from, to)).unwrap();
    }
}
impl Drop for Copy { fn drop(&mut self) { std::fs::remove_dir_all(&self.root).unwrap(); } }

#[test]
fn public_names_have_a_written_map_and_storage_gaps() {
    let report = LayoutReport::open_fixture(&fixture()).unwrap().text;
    assert!(report.contains("execution=not-qualified"));
    assert!(report.contains("gaps=written-audit"));
    assert!(report.contains("gap=config-plan"));
    assert!(report.contains("gap=mixed-schema"));
    assert!(report.contains("unsupported block scale encoding"));
    // The plan adapter reads FP8_PB_WO and MXFP8 weight_block_size recipes. Their tensors stay not qualified.
    assert!(report.contains(r#"recipe="model.language_model.layers.3.self_attn.q_a_proj""#));
    assert!(!report.contains("unsupported quant_algo FP8_PB_WO"));
    assert!(!report.contains("missing field group_size"));
    assert!(report.contains("recipe-schema-accepted; tensors-not-qualified"));
    for suffix in ["csf32_bases", "csf32_codes", "csf32_page_bases", "csf32_offsets", "csf32_exceptions"] {
        assert!(report.contains(&format!("weight_scale.{suffix}")));
    }
    for op in ["KDA recurrence and gates", "Latent attention", "Sparse attention indexer",
        "Manifold-constrained hyper-connections", "Sigmoid expert routing", "Capped SwiGLU",
        "Dense and expert layer schedule", "MTP plan and tensor names"] {
        assert!(report.contains(&format!("op={op:?}")));
    }
    assert!(report.contains("recipe-schema-accepted; tensors-not-qualified"));
    assert!(report.contains("payload_bytes_read=0"));
    assert_eq!(report, LayoutReport::open_fixture(&fixture()).unwrap().text);
}

#[test]
fn existing_inspect_binary_prints_fixture_map_without_weights() {
    let copy = Copy::new();
    assert!(!copy.root.join("fixture.safetensors").exists());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_inspect-checkpoint"))
        .arg("--layout-fixture").arg(&copy.root).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), LayoutReport::open_fixture(&copy.root).unwrap().text);
}

#[test]
fn malformed_storage_metadata_keeps_the_rest_of_the_fixture_report() {
    for (from, to) in [
        ("\"quantized_layers\"", "\"missing_quantized_layers\""),
        ("\"quantized_layers\": {", "\"quantized_layers\": null, \"unused_recipes\": {"),
        ("\"quantized_layers\": {", "\"quantized_layers\": [], \"unused_recipes\": {"),
        ("\"quantization_config\": {", "\"quantization_config\": null, \"unused_quantization\": {"),
    ] {
        let copy = Copy::new();
        copy.replace("config.json", from, to);
        let report = LayoutReport::open_fixture(&copy.root).unwrap().text;
        assert!(report.contains("gap=config-plan"));
        assert!(report.contains("gap=mixed-schema"));
        assert!(report.contains("gap=mixed-recipes"));
        assert!(report.contains("op=\"KDA recurrence and gates\""));
        assert!(report.contains("tensor=\"lm_head.weight\""));
        assert!(report.contains("payload_bytes_read=0"));
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_inspect-checkpoint"))
            .arg("--layout-fixture").arg(&copy.root).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(String::from_utf8(output.stdout).unwrap(), report);
    }
}

#[test]
fn index_and_audit_must_cover_the_header() {
    let copy = Copy::new();
    let name = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight";
    copy.replace("model.safetensors.index.json", &format!("\"{name}\""), "\"unmapped.weight\"");
    assert!(LayoutReport::open_fixture(&copy.root).unwrap_err().to_string().contains("absent from index"));
    let copy = Copy::new();
    copy.replace("audit.json", &format!("\"{name}\""), "\"unmapped.weight\"");
    assert!(LayoutReport::open_fixture(&copy.root).unwrap_err().to_string().contains("audit differs"));
}

#[test]
fn paths_cannot_escape_the_fixture() {
    let copy = Copy::new();
    copy.replace("model.safetensors.index.json", "fixture.safetensors", "../fixture.safetensors");
    assert!(LayoutReport::open_fixture(&copy.root).is_err());
}

#[cfg(unix)]
#[test]
fn outside_symlink_content_cannot_reach_the_report() {
    let outside = Copy::new();
    let marker = "OUTSIDE_FIXTURE_CONTENT";
    for file in ["config.json", "model.safetensors.index.json", "audit.json", "fixture.safetensors.header.json"] {
        let copy = Copy::new();
        let target = outside.root.join(file);
        std::fs::write(&target, format!(r#"{{"outside_marker":"{marker}"}}"#)).unwrap();
        let link = copy.root.join(file);
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(!target.canonicalize().unwrap().starts_with(copy.root.canonicalize().unwrap()));
        let error = LayoutReport::open_fixture(&copy.root).unwrap_err().to_string();
        assert!(error.contains("fixture path escapes its directory"), "{file}: {error}");
        assert!(!error.contains(marker));
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_inspect-checkpoint"))
            .arg("--layout-fixture").arg(&copy.root).output().unwrap();
        assert!(!output.status.success(), "{file}");
        assert!(output.stdout.is_empty(), "{file}: partial report");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("fixture path escapes its directory"), "{file}: {error}");
        assert!(!error.contains(marker));
    }
}

#[test]
fn raw_scale_examples_follow_loader_geometry() {
    use engine_model::json::{self, Value, field};
    let bytes = std::fs::read(fixture().join("fixture.safetensors.header.json")).unwrap();
    let doc = json::parse(&bytes, json::Limits::default()).unwrap();
    let tensors = doc.object().unwrap();
    let shape = |name: &str| field(tensors[name].object().unwrap(), "shape").unwrap().array().unwrap()
        .iter().map(|v| v.u64().unwrap()).collect::<Vec<_>>();
    let dtype = |name: &str| field(tensors[name].object().unwrap(), "dtype").unwrap().string().unwrap();
    let prefix = "model.language_model.layers.45.mlp.experts.0.gate_proj";
    let weight = format!("{prefix}.weight");
    let scale = format!("{prefix}.weight_scale");
    let global = format!("{prefix}.weight_scale_2");
    let packed = shape(&weight);
    let cols = packed[1] * 2;
    assert_eq!(cols % engine_model::flash::nvfp4::BLOCK as u64, 0);
    assert_eq!(shape(&scale), [packed[0], cols / 16]);
    assert_eq!((dtype(&weight), dtype(&scale), dtype(&global)), ("U8", "F8_E4M3", "F32"));
    assert_eq!(shape(&global), Vec::<u64>::new());
    assert!(!tensors.contains_key(&format!("{prefix}.input_scale")));
    let config = json::parse(&std::fs::read(fixture().join("config.json")).unwrap(), json::Limits::default()).unwrap();
    let quant = field(config.object().unwrap(), "quantization_config").unwrap().object().unwrap();
    let layers = field(quant, "quantized_layers").unwrap().object().unwrap();
    let mtp = layers["model.language_model.layers.45.mlp.experts"].object().unwrap();
    assert_eq!(mtp["quant_algo"], Value::String("W4A16_NVFP4".into()));
    assert_eq!(mtp["group_size"].u64().unwrap(), 16);
    let weight = shape("lm_head.weight");
    assert_eq!(shape("lm_head.weight_scale"), [weight[0], weight[1] / 32]);
    assert_eq!((dtype("lm_head.weight"), dtype("lm_head.weight_scale")), ("F8_E4M3", "U8"));
}
