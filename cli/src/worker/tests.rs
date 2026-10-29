use super::*;
use super::model::{MAX_MODEL_INDEX_BYTES, indexed_shards};

#[test]
fn private_open_accepts_regular_files_and_refuses_nonregular_or_linked_files() {
    let root = fs::canonicalize(std::env::temp_dir())
        .unwrap()
        .join(format!(
            "vade-private-open-{}",
            crate::text::random_hex(12).unwrap()
        ));
    crate::text::private_dir(&root).unwrap();
    let target = root.join("state");
    private_open(&target, true, false).unwrap();
    private_open(&target, false, false).unwrap();
    private_open(&target, false, true).unwrap();
    assert!(private_open(&target, true, false).is_err());
    let error = private_open(&root, false, false).unwrap_err();
    assert_eq!(
        error,
        "Machine state file is not private. Restore mode 600 before continuing."
    );
    std::os::unix::fs::symlink(&target, root.join("link")).unwrap();
    assert!(private_open(&root.join("link"), false, false).is_err());
    std::os::unix::fs::symlink(&root, root.join("parent-link")).unwrap();
    assert!(private_open(&root.join("parent-link/state"), false, false).is_err());
    fs::hard_link(&target, root.join("alias")).unwrap();
    assert!(private_open(&target, false, false).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn stat_handles_parentheses_and_zombies() {
    let mut columns = vec!["0"; 20];
    columns[0] = "S";
    columns[19] = "12345";
    assert_eq!(
        stat_start(&format!("42 (engine (rank)) {}", columns.join(" "))).unwrap(),
        Some("12345".into())
    );
    columns[0] = "Z";
    assert_eq!(
        stat_start(&format!("42 (engine) {}", columns.join(" "))).unwrap(),
        None
    );
}
#[test]
fn arguments_are_numeric_contiguous_and_binary_identity_roundtrips() {
    assert!(arguments(&fields(&[("arg.1", "missing zero")])).is_err());
    let bytes = b"engine\0arg\xff";
    assert_eq!(unhex(&to_hex(bytes)).unwrap(), bytes);
    assert!(unhex("a").is_err());
    assert!(environment(&fields(&[("env.KEY", "one"), ("env_unset.KEY", "1")])).is_err());
}
#[test]
fn shard_index_and_mocked_identity_fail_closed() {
    let shards = indexed_shards(r#"{"metadata":{},"weight_map":{"a":"shard-1.safetensors","b":"shard-2\u002esafetensors"}}"#).unwrap();
    assert_eq!(shards.len(), 2);
    assert!(shards.contains("shard-2.safetensors"));
    assert!(indexed_shards(r#"{"weight_map":{"a":"../outside.safetensors"}}"#).is_err());
    assert!(indexed_shards(r#"{"weight_map":{"a":"missing.safetensors"}"#).is_err());
    let original = Identity {
        pid: 42,
        start: "100".into(),
        boot: "boot-a".into(),
        exe: PathBuf::from("/engine"),
        command: b"engine\0".to_vec(),
    };
    assert!(verify_snapshot(&original, "boot-a", Some(&original)).unwrap());
    assert!(!verify_snapshot(&original, "boot-a", None).unwrap());
    let reboot = verify_snapshot(&original, "boot-b", None).unwrap_err();
    assert!(reboot.contains("Recovery in cli/README.md") && !reboot.contains("Preserve"));
    let mut replacement = original.clone();
    replacement.start = "101".into();
    let changed = verify_snapshot(&original, "boot-a", Some(&replacement)).unwrap_err();
    assert!(changed.contains("Nothing was signaled") && changed.contains("Recovery in cli/README.md"));
}
#[test]
fn stopped_recovery_is_idempotent_and_foreign_owner_is_preserved() {
    let _process_guard = crate::runner::process_test_lock();
    let base = fs::canonicalize(std::env::temp_dir()).unwrap();
    let root = base.join(format!(
        "vade-worker-{}",
        crate::text::random_hex(12).unwrap()
    ));
    let token = "a".repeat(64);
    let mut request = fields(&[
        ("op", "inspect"),
        ("token", &token),
        ("state_dir", root.to_str().unwrap()),
    ]);
    let store = Store::open(&request).unwrap();
    crate::text::private_dir(&store.run).unwrap();
    store
        .save(&fields(&[("token", &token), ("phase", "stopped")]))
        .unwrap();
    crate::text::atomic_write(
        &store.active(),
        &crate::text::encode(&fields(&[("token", &token)])),
    )
    .unwrap();
    crate::text::atomic_write(
        &store.env(),
        &crate::text::encode(&fields(&[("env.API_KEY", "hidden")])),
    )
    .unwrap();
    private_open(&store.log(), true, false)
        .unwrap()
        .write_all(b"ready hidden\n")
        .unwrap();
    drop(store);
    assert_eq!(handle(&request).unwrap()["alive"], "false");
    request.insert("op".into(), "stop".into());
    assert_eq!(handle(&request).unwrap()["phase"], "stopped");
    request.insert("op".into(), "clean".into());
    assert_eq!(handle(&request).unwrap()["log"], "ready [redacted]\n");
    assert_eq!(handle(&request).unwrap()["phase"], "absent");
    let foreign = fields(&[("token", &"b".repeat(64))]);
    crate::text::atomic_write(&root.join("active.conf"), &crate::text::encode(&foreign))
        .unwrap();
    assert!(handle(&request).is_err());
    assert_eq!(
        read_fields(&root.join("active.conf")).unwrap(),
        Some(foreign.clone())
    );
    request.insert("op".into(), "start".into());
    let refused = handle(&request).unwrap();
    assert_eq!(refused["error_code"], "start_not_written");
    assert_eq!(refused["phase"], "absent");
    assert!(!root.join(&token).exists());
    assert_eq!(
        read_fields(&root.join("active.conf")).unwrap(),
        Some(foreign)
    );
    crate::text::private_dir(&root.join(&token)).unwrap();
    let interrupted = crate::text::encode(&fields(&[("token", &token), ("phase", "starting")]));
    crate::text::atomic_write(&root.join(&token).join("journal.conf"), &interrupted).unwrap();
    assert!(handle(&request).is_err());
    assert_eq!(
        fs::read_to_string(root.join(&token).join("journal.conf")).unwrap(),
        interrupted
    );
    // With our original owner restored, incomplete startup still refuses
    // both inspect and stop, and tells the operator where the steps are.
    crate::text::atomic_write(
        &root.join("active.conf"),
        &crate::text::encode(&fields(&[("token", &token)])),
    )
    .unwrap();
    for operation in ["inspect", "stop"] {
        request.insert("op".into(), operation.into());
        let error = handle(&request).unwrap_err();
        assert!(
            error.contains("Recovery in cli/README.md")
                && !error.contains("journal")
                && !error.contains("Preserve")
        );
    }
    let model = root.join("model");
    fs::create_dir(&model).unwrap();
    for name in ["config.json", "tokenizer.json", "shard-1.safetensors"] {
        fs::write(model.join(name), "{}").unwrap();
    }
    fs::write(
        model.join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}}"#,
    )
    .unwrap();
    let request = fields(&[("op", "model"), ("model_path", model.to_str().unwrap())]);
    assert!(handle(&request).is_err());
    fs::write(model.join("shard-2.safetensors"), "{}").unwrap();
    assert_eq!(handle(&request).unwrap()["weights"], "present");
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn large_fp4_index_is_bounded_and_still_checks_every_shard() {
    let root = fs::canonicalize(std::env::temp_dir()).unwrap()
        .join(format!("vade-large-index-{}", crate::text::random_hex(12).unwrap()));
    fs::create_dir(&root).unwrap();
    for name in ["config.json", "tokenizer.json", "one.safetensors"] {
        fs::write(root.join(name), "{}").unwrap();
    }
    let index = root.join("model.safetensors.index.json");
    let mut file = File::create(&index).unwrap();
    file.write_all(b"{").unwrap();
    let padding = vec![b' '; 1024 * 1024];
    for _ in 0..33 { file.write_all(&padding).unwrap(); }
    file.write_all(br#""weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#).unwrap();
    drop(file);
    assert!(check_model(&root).unwrap_err().contains("shard is missing"));
    fs::write(root.join("two.safetensors"), "{}").unwrap();
    check_model(&root).unwrap();
    File::options().write(true).open(&index).unwrap().set_len(MAX_MODEL_INDEX_BYTES + 1).unwrap();
    assert!(check_model(&root).unwrap_err().contains("exceeds 64 MiB"));
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn shared_machine_state_is_refused_without_any_writes() {
    let root = fs::canonicalize(std::env::temp_dir())
        .unwrap()
        .join(format!(
            "vade-shared-{}",
            crate::text::random_hex(12).unwrap()
        ));
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(root.join("keep"), "Other job.").unwrap();
    let request = fields(&[
        ("op", "start"),
        ("token", &"a".repeat(64)),
        ("state_dir", root.to_str().unwrap()),
    ]);
    assert!(handle(&request).unwrap_err().contains("shared"));
    assert_eq!(
        fs::metadata(&root).unwrap().permissions().mode() & 0o777,
        0o755
    );
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    assert_eq!(fs::read_to_string(root.join("keep")).unwrap(), "Other job.");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cuda_cache_survives_runs_and_is_keyed_by_ptx() {
    let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-cuda-cache-{}", crate::text::random_hex(8).unwrap()));
    let mut env = Fields::new();
    persistent_cuda_cache(&root, &"a".repeat(64), &mut env).unwrap();
    let first = env["env.CUDA_CACHE_PATH"].clone();
    fs::write(Path::new(&first).join("cached-cubin"), "cache").unwrap();
    persistent_cuda_cache(&root, &"a".repeat(64), &mut env).unwrap();
    assert_eq!(env["env.CUDA_CACHE_PATH"], first);
    assert!(Path::new(&first).join("cached-cubin").is_file());
    assert_eq!(env["env.CUDA_CACHE_DISABLE"], "0");
    persistent_cuda_cache(&root, &"b".repeat(64), &mut env).unwrap();
    assert_ne!(env["env.CUDA_CACHE_PATH"], first);
    fs::remove_dir_all(root).unwrap();
}
