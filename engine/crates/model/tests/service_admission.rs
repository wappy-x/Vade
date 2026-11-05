//! Deferred metadata and primitive targets cannot admit a resident service.
use engine_model::{cartridge::{Cartridge,Kind}, plan::{ConfigPlan,PlanError,TextDescription,layer::{LayerPlan,Topology}}};
use std::{convert::Infallible,path::PathBuf};

fn description() -> TextDescription {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else {panic!()}; d
}
#[test]
fn text_service_requires_complete_headers_and_a_matching_storage_sidecar() {
    let root=std::env::temp_dir().join(format!("vade-text-admission-{}",std::process::id()));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("config.json"),include_bytes!("fixtures/glm/config.json")).unwrap();
    assert!(Cartridge::resident_text(&root).is_err());
    std::fs::write(root.join("hf_quant_config.json"),br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"other":{"quant_algo":"MXFP8","weight_block_size":[1,32]}}}"#).unwrap();
    assert!(Cartridge::resident_text(&root).err().unwrap().to_string().contains("sidecar differ"));
    std::fs::remove_dir_all(root).unwrap();
}
fn refusal(result: Result<Infallible,PlanError>) -> PlanError {
    match result {Err(error)=>error,Ok(never)=>match never {}}
}
fn check_holds(error: &str) {
    for required in ["complete resident decoder", "layer, text, causal state, and ordered transport producers",
        "service Factory", "precision, casts, quantizers, and reductions", "Tokenizer", "chat template",
        "stop set", "appended prediction", "token-ID proof"] {
        assert!(error.contains(required),"missing {required}: {error}");
    }
}

#[test]
fn cartridge_preflight_and_kind_refuse_before_absent_weights_or_tokenizer() {
    let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm");
    assert!(!root.join("tokenizer.json").exists());
    assert!(!root.join("model.safetensors").exists());
    assert!(!root.join("model.safetensors.index.json").exists());
    let kind=Kind::detect(&root).unwrap_err().to_string(); check_holds(&kind);
    let preflight=Cartridge::preflight(&root).err().expect("deferred checkpoint must refuse").to_string();
    check_holds(&preflight); assert!(preflight.contains(&kind));
    let listing=Cartridge::inspect(&root).unwrap();
    assert!(listing.config.is_none() && listing.gemma4.is_none());
    assert_eq!(listing.reason.as_deref(),Some(kind.as_str()));
    assert!(listing.require_supported().is_err());
}

#[test]
fn partial_audit_metadata_cannot_skip_operation_field_checks() {
    let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm-fp4-local");
    // This audit fixture has an index but lacks complete operation metadata.
    let expected=ConfigPlan::from_config(include_bytes!("fixtures/glm-fp4-local/config.json")).unwrap_err();
    assert!(matches!(expected,PlanError::Missing {field:"attention_bias"}));
    for result in [Kind::detect(&root).map(|_| ()),Cartridge::preflight(&root).map(|_| ()),Cartridge::inspect(&root).map(|_| ())] {
        assert_eq!(result.unwrap_err().to_string(),expected.to_string());
    }
}

#[test]
fn complete_logical_targets_do_not_supply_resident_producers() {
    let d=description();
    for parts in [1,2,4] {
        // KDA/dense, latent/experts, and KDA/experts all name their primitive targets.
        for layer in [0,3,4] {
            let plan=LayerPlan::new(&d,layer,1,Topology {parts,rank:0}).unwrap();
            assert!(plan.missing().is_empty());
            assert!(plan.steps.iter().all(|s| !s.target.entries.is_empty()));
            assert!(plan.bf16_cast_contract().unwrap().require_complete().is_err());
        }
        check_holds(&refusal(d.require_resident_service()).to_string());
    }
}

#[test]
fn description_changes_cannot_open_service_or_executable_shape() {
    let d=description();
    for draft in [0,d.declared_draft_layers] {
        for tied in [false,true] {
            let mut changed=d.clone(); changed.declared_draft_layers=draft; changed.tied_embeddings=tied;
            // These synthetic metadata changes grant no text or draft policy.
            changed.storage.layers.clear(); changed.storage.excluded.clear();
            let error=refusal(changed.require_resident_service()); check_holds(&error.to_string());
            assert_eq!(ConfigPlan::Deferred(changed).into_model_plan().unwrap_err(),error);
        }
    }
}
