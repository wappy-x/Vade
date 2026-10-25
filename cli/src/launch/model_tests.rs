use super::*;
pub(super) struct Fixture(pub(super) PathBuf);
impl Fixture {
    pub(super) fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "vade-launch-model-{}",
            crate::text::random_hex(8).unwrap()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(fs::canonicalize(path).unwrap())
    }
    pub(super) fn model(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("config.json"), test_config(8192)).unwrap();
        test_write_serving_metadata(&path);
        fs::write(path.join("model.safetensors"), [0; 16]).unwrap();
        fs::write(path.join("model.safetensors.index.json"), test_index()).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn environment_then_preferred_default_then_one_prompt_and_saved_path() {
    let f = Fixture::new();
    let env = f.model("elsewhere/checkpoint");
    let default = f.model("models/org/checkpoint");
    let no_prompt = |_: &str| panic!("unexpected prompt");
    let found = discover_with(&f.0, None, Some(env.as_os_str()), no_prompt).unwrap();
    assert_eq!(found.path, env);
    assert_eq!((found.bytes, found.context_limit), (16, 8192));
    let preferred = f.model("models/nvidia/Qwen3.8-Flash-Next-FP4-Local");
    assert_eq!(discover_with(&f.0, None, None, no_prompt).unwrap().path, preferred);
    fs::remove_dir_all(preferred).unwrap();
    assert_eq!(discover_with(&f.0, None, None, no_prompt).unwrap().path, default);
    assert_eq!(
        discover_with(&f.0, Some(&env), None, no_prompt)
            .unwrap()
            .path,
        env
    );
    fs::remove_dir_all(&default).unwrap();
    assert_eq!(
        discover_with(&f.0, None, Some(env.as_os_str()), no_prompt)
            .unwrap()
            .path,
        env
    );
    let mut prompts = 0;
    assert_eq!(
        discover_with(&f.0, None, None, |_| {
            prompts += 1;
            Ok(env.clone())
        })
        .unwrap()
        .path,
        env
    );
    assert_eq!(prompts, 1);
    let mut prompts = 0;
    assert!(
        discover_with(&f.0, None, None, |_| {
            prompts += 1;
            Ok(f.0.join("missing"))
        })
        .is_err()
    );
    assert_eq!(prompts, 1);
}
#[test]
fn ambiguity_and_saved_missing_require_an_explicit_selection() {
    let f = Fixture::new();
    let a = f.model("models/a");
    f.model("models/b");
    assert!(
        locate(&f.0.join("models"))
            .unwrap_err()
            .contains("More than one")
    );
    assert_eq!(
        discover_with(&f.0, None, None, |_| Ok(a.clone()))
            .unwrap()
            .path,
        a
    );
    let mut prompts = 0;
    let recovered = discover_with(&f.0, Some(&f.0.join("missing")), None, |reason| {
        assert!(reason.contains("unavailable"));
        prompts += 1;
        Ok(a.clone())
    })
    .unwrap();
    assert_eq!(prompts, 1);
    assert_eq!(recovered.path, a);
}
#[test]
fn shard_inventory_counts_once_and_rejects_missing_empty_or_escape() {
    let f = Fixture::new();
    let path = f.model("models/checkpoint");
    let index = path.join("model.safetensors.index.json");
    fs::write(
        &index,
        r#"{"metadata":{},"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(inspect(&path).unwrap().bytes, 16);
    for value in [
        r#"{"metadata":{},"weight_map":{"a":"missing.safetensors"}}"#,
        r#"{"metadata":{},"weight_map":{"a":"../model.safetensors"}}"#,
        r#"{"metadata":{},"weight_map":{}}"#,
        r#"{"metadata":{},"weight_map":{"a":42}}"#,
        r#"{"metadata":{},"weight_map":{"a":"model.safetensors"},}"#,
    ] {
        fs::write(&index, value).unwrap();
        assert!(inspect(&path).is_err(), "{value}");
    }
    fs::write(index, test_index()).unwrap();
    fs::write(path.join("model.safetensors"), "").unwrap();
    assert!(inspect(&path).unwrap_err().contains("empty"));
}
#[test]
fn payload_links_stay_inside_checkpoint_and_directory_loops_are_skipped() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let path = f.model("models/checkpoint");
    let blob = f.0.join("blob");
    fs::write(&blob, [0; 24]).unwrap();
    fs::remove_file(path.join("model.safetensors")).unwrap();
    symlink(blob, path.join("model.safetensors")).unwrap();
    symlink(f.0.join("models"), f.0.join("models/loop")).unwrap();
    assert!(
        locate(&f.0.join("models"))
            .unwrap_err()
            .contains("escapes the checkpoint")
    );
    fs::remove_file(path.join("model.safetensors")).unwrap();
    fs::write(path.join("inside.safetensors"), [0; 24]).unwrap();
    symlink("inside.safetensors", path.join("model.safetensors")).unwrap();
    assert_eq!(locate(&f.0.join("models")).unwrap().bytes, 24);
    fs::remove_dir_all(path).unwrap();
    assert!(
        locate(&f.0.join("models"))
            .unwrap_err()
            .contains("No complete")
    );
}
#[test]
fn checkpoint_symlink_aliases_resolve_once_and_distinct_targets_are_ambiguous() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let checkpoint = f.model("other-volume/checkpoint");
    let models = f.0.join("models");
    fs::create_dir_all(&models).unwrap();
    symlink(&checkpoint, models.join("primary")).unwrap();
    symlink(&checkpoint, models.join("alias")).unwrap();
    let found = discover_with(&f.0, None, None, |_| {
        panic!("linked checkpoint should be found")
    })
    .unwrap();
    assert_eq!(found.path, checkpoint);
    assert_eq!(found.bytes, 16);
    let second = f.model("other-volume/second");
    symlink(second, models.join("another")).unwrap();
    assert!(locate(&models).unwrap_err().contains("More than one model"));
}
#[test]
fn dangling_loop_and_container_symlinks_do_not_expand_discovery() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let checkpoint = f.model("other-volume/checkpoint");
    let models = f.0.join("models");
    fs::create_dir_all(&models).unwrap();
    symlink(models.join("cycle-b"), models.join("cycle-a")).unwrap();
    symlink(models.join("cycle-a"), models.join("cycle-b")).unwrap();
    symlink(&models, models.join("self")).unwrap();
    symlink(f.0.join("missing"), models.join("dangling")).unwrap();
    symlink(f.0.join("other-volume"), models.join("container")).unwrap();
    assert!(locate(&models).unwrap_err().contains("No complete model"));
    symlink(&checkpoint, models.join("direct")).unwrap();
    assert_eq!(locate(&models).unwrap().path, checkpoint);
}
#[test]
fn engine_metadata_filters_architecture_quantization_and_recovers_saved_choice() {
    let f = Fixture::new();
    let supported = f.model("elsewhere/fp4-local");
    fs::write(supported.join("config.json"), test_mixed_config(8192)).unwrap();
    let unsupported = f.model("models/qwen3-8b");
    fs::write(
        unsupported.join("config.json"),
        test_config(8192).replace("qwen4_exp", "qwen3"),
    )
    .unwrap();
    let error = inspect(&unsupported).unwrap_err();
    assert!(
        error.contains("cannot load") && error.contains("model_type"),
        "{error}"
    );
    assert!(!error.contains('\n'));
    assert_eq!(
        discover_with(&f.0, None, Some(supported.as_os_str()), |_| panic!())
            .unwrap()
            .path,
        supported
    );
    assert_eq!(
        discover_with(
            &f.0,
            Some(&unsupported),
            Some(supported.as_os_str()),
            |_| panic!()
        )
        .unwrap()
        .path,
        supported
    );
    assert!(
        discover_with(&f.0, Some(&unsupported), None, |reason| Err(reason.into()))
            .unwrap_err()
            .contains("model_type")
    );
    // A supported architecture is insufficient: reject incompatible quantization,
    // cache settings and mixed-layer group sizes using the real engine parser.
    let config = test_config(65536);
    for invalid in [
        config.replace("qwen4_exp_text", "qwen3_text"),
        config.replace("\"fp8\"", "\"awq\""),
        config.replace("[128,128]", "[64,128]"),
        config.replace("\"dynamic\"", "\"static\""),
        test_mixed_config(65536).replace("\"group_size\":16", "\"group_size\":32"),
        test_mixed_config(65536).replace(
            "\"kv_cache_quant_algo\":null",
            "\"kv_cache_quant_algo\":\"FP8\"",
        ),
    ] {
        fs::write(supported.join("config.json"), invalid).unwrap();
        assert!(inspect(&supported).unwrap_err().contains("cannot load"));
    }
    for supported_config in [config, test_mixed_config(65536)] {
        fs::write(supported.join("config.json"), supported_config).unwrap();
        assert_eq!(inspect(&supported).unwrap().context_limit, 65536);
    }
}

#[test]
fn metadata_context_is_read_from_supported_engine_geometry() {
    let f = Fixture::new();
    let path = f.model("model");
    fs::write(path.join("config.json"), test_config(65536)).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 65536);
    for value in ["-1", "\"8192\"", "null", "[]", "false"] {
        fs::write(
            path.join("config.json"),
            test_config(8192).replace(
                "\"max_position_embeddings\":8192",
                &format!("\"max_position_embeddings\":{value}"),
            ),
        )
        .unwrap();
        assert!(inspect(&path).is_err(), "{value}");
    }
}

#[test]
fn legacy_nvfp4_and_malformed_sidecars_refuse_before_admission() {
    let f = Fixture::new();
    let path = f.model("checkpoint");
    let config = legacy_config(8192);
    let parsed = engine_model::flash::Config::parse(config.as_bytes()).unwrap();
    assert_eq!(parsed.format, engine_model::flash::WeightFormat::Nvfp4);
    assert!(parsed.mixed.is_none());
    fs::write(path.join("config.json"), config).unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("legacy NVFP4 is unsupported here")
    );
    fs::write(path.join("config.json"), test_config(8192)).unwrap();
    fs::write(path.join("hf_quant_config.json"), "{}").unwrap();
    assert!(inspect(&path).unwrap_err().contains("quantization sidecar"));
    fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
    assert!(inspect(&path).unwrap_err().contains("quantization sidecar"));
    fs::remove_file(path.join("hf_quant_config.json")).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
}

fn sidecar() -> String {
    let mut layers = (0..48)
        .map(|layer| {
            format!(r#""model.language_model.layers.{layer}.mlp.experts":{{"quant_algo":"NVFP4","group_size":16}}"#)
        })
        .collect::<Vec<_>>();
    layers.push(
        r#""mtp.layers.0.mlp.experts":{"quant_algo":"FP8_BLOCK_SCALES","group_size":128}"#
            .into(),
    );
    layers.push(format!(
        r#""{}.ngram_embedding":{{"quant_algo":"FP8"}}"#,
        engine_model::flash::PLE_PREFIX
    ));
    format!(
        r#"{{"quantization":{{"quant_algo":"MIXED_PRECISION","group_size":16,"kv_cache_quant_algo":null,"quantized_layers":{{{}}}}}}}"#,
        layers.join(",")
    )
}

#[test]
fn mixed_sidecar_admission_matches_the_complete_startup_contract() {
    use engine_model::flash::{WeightFormat, manifest::ReceiptPlan};
    let f = Fixture::new();
    let path = f.model("mixed");
    let config = test_mixed_config(8192);
    let parsed = engine_model::flash::Config::parse(config.as_bytes()).unwrap();
    assert_eq!(parsed.format, WeightFormat::Nvfp4);
    assert!(parsed.mixed.is_some());
    fs::write(path.join("config.json"), config).unwrap();
    // Both an absent sidecar and a fully valid legacy sidecar are supported.
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
    assert_eq!(
        ReceiptPlan::open(&path).unwrap().format,
        WeightFormat::Nvfp4
    );
    let valid = sidecar();
    fs::write(path.join("hf_quant_config.json"), &valid).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
    assert_eq!(
        ReceiptPlan::open(&path).unwrap().format,
        WeightFormat::Nvfp4
    );
    let sidecar_path = path.join("hf_quant_config.json");
    let permissions = fs::metadata(&sidecar_path).unwrap().permissions();
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&sidecar_path, fs::Permissions::from_mode(0o000)).unwrap();
    }
    // DAC-override identities may read mode-000 files. Exercise denial
    // only when this runner actually experiences it, then continue every
    // schema assertion below for both privileged and ordinary runners.
    let unreadable = fs::read(&sidecar_path).is_err();
    let admission = inspect(&path);
    let startup = ReceiptPlan::open(&path);
    fs::set_permissions(&sidecar_path, permissions).unwrap();
    if unreadable {
        let admission = admission.unwrap_err();
        let startup = startup.unwrap_err().to_string();
        assert!(admission.contains(&startup), "{admission}; {startup}");
    } else {
        assert_eq!(admission.unwrap().context_limit, 8192);
        assert_eq!(startup.unwrap().format, WeightFormat::Nvfp4);
    }
    for invalid in [
        "{".into(),
        "{}".into(),
        valid.replace("MIXED_PRECISION", "NVFP4"),
        valid.replacen("\"group_size\":16", "\"group_size\":32", 1),
        valid.replace(
            "\"kv_cache_quant_algo\":null",
            "\"kv_cache_quant_algo\":\"FP8\"",
        ),
        valid.replace(
            "\"quantized_layers\":{",
            "\"quantized_layers\":{\"extra\":{},",
        ),
        valid.replace("layers.47.mlp.experts", "layers.48.mlp.experts"),
        valid.replace("\"NVFP4\"", "\"FP8\""),
        valid.replace("\"group_size\":16", "\"group_size\":32"),
        valid.replace("FP8_BLOCK_SCALES", "FP8_PB_WO"),
        valid.replace("\"group_size\":128", "\"group_size\":64"),
        valid.replace("\"FP8\"", "\"NVFP4\""),
    ] {
        fs::write(path.join("hf_quant_config.json"), &invalid).unwrap();
        let admission = inspect(&path).unwrap_err();
        let startup = ReceiptPlan::open(&path).unwrap_err().to_string();
        assert!(
            admission.contains(&startup),
            "admission: {admission}; startup: {startup}"
        );
        assert!(
            discover_with(&f.0, Some(&path), None, |reason| Err(reason.into())).is_err(),
            "invalid saved model must not be republished"
        );
    }
    let sidecar_path = path.join("hf_quant_config.json");
    let file = File::create(&sidecar_path).unwrap();
    file.set_len(1024 * 1024 + 1).unwrap();
    assert!(inspect(&path).unwrap_err().contains("byte limit"));
    assert!(
        ReceiptPlan::open(&path)
            .unwrap_err()
            .to_string()
            .contains("byte limit")
    );
    fs::remove_file(&sidecar_path).unwrap();
    fs::create_dir(&sidecar_path).unwrap();
    assert!(inspect(&path).unwrap_err().contains("regular file"));
    assert!(
        ReceiptPlan::open(&path)
            .unwrap_err()
            .to_string()
            .contains("regular file")
    );
    fs::remove_dir(&sidecar_path).unwrap();
    std::os::unix::fs::symlink(path.join("missing"), &sidecar_path).unwrap();
    assert!(inspect(&path).is_err());
    assert!(ReceiptPlan::open(&path).is_err());
}

#[test]
fn mixed_fifo_sidecar_refuses_in_discovery_and_startup_without_a_writer() {
    use engine_model::flash::manifest::ReceiptPlan;
    let _guard = crate::runner::process_test_lock();
    let f = Fixture::new();
    let path = f.model("mixed");
    fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
    let sidecar_path = path.join("hf_quant_config.json");
    let fifo = path.join("fifo");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    for linked in [false, true] {
        if linked {
            std::os::unix::fs::symlink(&fifo, &sidecar_path).unwrap();
        } else {
            fs::rename(&fifo, &sidecar_path).unwrap();
        }
        let root = path.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let admission = inspect(&root).unwrap_err();
            let startup = ReceiptPlan::open(&root).unwrap_err().to_string();
            sender.send((admission, startup)).unwrap();
        });
        let (admission, startup) = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("mixed sidecar reader blocked on FIFO");
        worker.join().unwrap();
        assert!(admission.contains("regular file"), "{admission}");
        assert!(startup.contains("regular file"), "{startup}");
        if linked {
            fs::remove_file(&sidecar_path).unwrap();
        } else {
            fs::rename(&sidecar_path, &fifo).unwrap();
        }
    }
}

fn legacy_config(context: u32) -> String {
    let experts = (0..48)
        .map(|layer| format!("model.language_model.layers.{layer}.mlp.experts"))
        .collect::<Vec<_>>();
    let mtp = "mtp.layers.0.mlp.experts";
    let ple = format!("{}.ngram_embedding", engine_model::flash::PLE_PREFIX);
    let mut layers = experts
        .iter()
        .map(|name| format!(r#"{name:?}:{{"quant_algo":"NVFP4","group_size":16}}"#))
        .collect::<Vec<_>>();
    layers.push(format!(
        r#"{mtp:?}:{{"quant_algo":"FP8_PB_WO","group_size":128}}"#
    ));
    layers.push(format!(r#"{ple:?}:{{"quant_algo":"FP8"}}"#));
    let mut groups = Vec::new();
    for (id, bits, group, dynamic, targets) in [
        (0, 4, Some(16), false, experts),
        (1, 8, Some(128), true, vec![mtp.into()]),
        (2, 8, None, false, vec![ple]),
    ] {
        let size = group
            .map(|size| format!(r#", "group_size":{size}"#))
            .unwrap_or_default();
        let operand = |dynamic| {
            format!(r#"{{"num_bits":{bits},"type":"float","dynamic":{dynamic}{size}}}"#)
        };
        let activation = if group.is_some() {
            format!(r#", "input_activations":{}"#, operand(dynamic))
        } else {
            String::new()
        };
        groups.push(format!(
            r#""group_{id}":{{"targets":{targets:?},"weights":{}{activation}}}"#,
            operand(false)
        ));
    }
    let quant = format!(
        r#"{{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{{{}}},"config_groups":{{{}}}}}"#,
        layers.join(","),
        groups.join(",")
    );
    let config = test_config(context);
    let (prefix, _) = config.rsplit_once("\"quantization_config\":").unwrap();
    format!(r#"{prefix}"quantization_config":{quant}}}"#)
}

#[test]
fn shard_index_uses_engine_format_and_filename_limits() {
    let f = Fixture::new();
    let path = f.model("checkpoint");
    let index = path.join("model.safetensors.index.json");
    fs::remove_file(&index).unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("model.safetensors.index.json")
    );
    fs::create_dir(path.join("nested")).unwrap();
    fs::write(path.join("nested/weight.safetensors"), [0; 16]).unwrap();
    let nested = r#"{"metadata":{},"weight_map":{"a":"nested/weight.safetensors"}}"#;
    fs::write(&index, nested).unwrap();
    assert!(inspect(&path).unwrap_err().contains("single relative"));
    fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
    assert_eq!(inspect(&path).unwrap().bytes, 16);
    for shard in [
        "nested//weight.safetensors",
        "nested/./weight.safetensors",
        "has space.safetensors",
    ] {
        fs::write(
            &index,
            format!(r#"{{"metadata":{{}},"weight_map":{{"a":{shard:?}}}}}"#),
        )
        .unwrap();
        assert!(
            inspect(&path).unwrap_err().contains("unsafe relative"),
            "{shard}"
        );
    }
    for malformed in [
        r#"{"weight_map":{"a":"model.safetensors"}}"#,
        r#"{"metadata":{},"weight_map":{"":"model.safetensors"}}"#,
    ] {
        fs::write(&index, malformed).unwrap();
        assert!(
            inspect(&path)
                .unwrap_err()
                .contains("cannot load the model shard index")
        );
    }
}

#[test]
fn shard_index_links_match_startup_checkpoint_containment() {
    use engine_model::flash::manifest::ReceiptPlan;
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let path = f.model("mixed");
    fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
    let index = path.join("model.safetensors.index.json");
    let external = f.0.join("external-index.json");
    fs::rename(&index, &external).unwrap();
    symlink(&external, &index).unwrap();
    let startup = ReceiptPlan::open(&path).unwrap_err().to_string();
    assert!(startup.contains("escapes checkpoint"), "{startup}");
    let admission = inspect(&path).unwrap_err();
    assert!(admission.contains("index") && admission.contains("escapes"), "{admission}");
    assert!(discover_with(&f.0, Some(&path), None, |reason| Err(reason.into())).is_err());

    // The same index bytes can be linked within the checkpoint, including
    // when the owner chooses an alias of the whole directory.
    fs::remove_file(&index).unwrap();
    let internal = path.join("actual-index.json");
    fs::rename(&external, &internal).unwrap();
    symlink("actual-index.json", &index).unwrap();
    assert_eq!(inspect(&path).unwrap().path, path);
    ReceiptPlan::open(&path).unwrap();
    let alias = f.0.join("alias");
    symlink(&path, &alias).unwrap();
    assert_eq!(inspect(&alias).unwrap().path, path);
    ReceiptPlan::open(&alias).unwrap();
}

#[test]
fn fifo_metadata_is_rejected_without_waiting_for_a_writer() {
    let _guard = crate::runner::process_test_lock();
    let f = Fixture::new();
    let path = f.model("model");
    for name in [
        "model.safetensors.index.json",
        "hf_quant_config.json",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ] {
        let target = path.join(name);
        let original = fs::read(&target).ok();
        if original.is_some() {
            fs::remove_file(&target).unwrap();
        }
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&target)
                .status()
                .unwrap()
                .success()
        );
        let target = path.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = sender.send(inspect(&target));
        });
        let error = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("metadata reader blocked on FIFO")
            .unwrap_err();
        worker.join().unwrap();
        assert!(
            error.contains(if name == "hf_quant_config.json" {
                "quantization sidecar"
            } else {
                "regular file"
            }),
            "{name}: {error}"
        );
        fs::remove_file(path.join(name)).unwrap();
        if let Some(original) = original {
            fs::write(path.join(name), original).unwrap();
        }
    }
    fs::remove_file(path.join("model.safetensors.index.json")).unwrap();
    fs::create_dir(path.join("model.safetensors.index.json")).unwrap();
    assert!(inspect(&path).unwrap_err().contains("regular file"));
}

#[test]
fn large_checkpoint_index_fits_the_bound_and_oversized_index_refuses() {
    let f = Fixture::new();
    let path = f.model("large-index");
    let index = path.join("model.safetensors.index.json");
    let mut file = File::create(&index).unwrap();
    file.write_all(br#"{"metadata":{},"weight_map":{"#).unwrap();
    // Whitespace exercises the byte bound without violating the engine's
    // independent 1 MiB limit for an individual JSON string.
    io::copy(&mut io::repeat(b' ').take(48 * 1024 * 1024), &mut file).unwrap();
    file.write_all(br#""a":"model.safetensors"}}"#).unwrap();
    drop(file);
    assert_eq!(inspect(&path).unwrap().bytes, 16);
    let file = File::create(&index).unwrap();
    file.set_len(64 * 1024 * 1024 + 1).unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("exceeds its size limit")
    );
}
