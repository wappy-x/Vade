//! Checkpoint receipts and private manifest staging: what a launch copies,
//! keeps private and refuses before the engine starts.
use super::*;

fn fixture_manifest(model: &Path) -> String {
    ["model.safetensors.index.json", "model.safetensors"]
        .into_iter()
        .map(|name| {
            let hash = crate::sha256::digest_reader(fs::File::open(model.join(name)).unwrap()).unwrap();
            format!("{hash}  {name}\n")
        })
        .collect()
}

#[test]
fn fp8_receipts_use_checkpoint_paths_despite_nvfp4_override() {
    let f = Fixture::new();
    fs::write(f.model.join("config.json"), model::test_config(8192)).unwrap();
    // Inspection understands FP8; the local launcher now requires FP4.
    // The serving receipt contract remains checkpoint-local for FP8.
    let discovered = model::discover(&f.home, Some(&f.model)).unwrap();
    Config::new(&f.root, &discovered).unwrap();
    let private = f.root.join("state/model-manifest.sha256");
    text::private_dir(&f.root).unwrap();
    text::private_dir(private.parent().unwrap()).unwrap();
    fs::write(&private, "NVFP4 override must remain unused\n").unwrap();
    let cache = f.root.join("state/cache");
    run_cpu_receipt(&f.model, &private, &cache, "fp8");
    let manifest = fixture_manifest(&f.model);
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        manifest.as_bytes()
    );
    assert!(f.model.join("manifest.sha256.sizes").is_file());
    assert_eq!(
        fs::read(&private).unwrap(),
        b"NVFP4 override must remain unused\n"
    );
    assert!(!f.root.join("state/model-manifest.sha256.sizes").exists());

    // FP8 replaces a stale server-written cache in that same checkpoint folder.
    fs::write(f.model.join("manifest.sha256"), "stale checkpoint cache\n").unwrap();
    fs::write(f.model.join("manifest.sha256.sizes"), "stale sizes\n").unwrap();
    run_cpu_receipt(&f.model, &private, &cache, "fp8");
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        manifest.as_bytes()
    );
    assert_ne!(
        fs::read(f.model.join("manifest.sha256.sizes")).unwrap(),
        b"stale sizes\n"
    );
    assert_eq!(
        fs::read(&private).unwrap(),
        b"NVFP4 override must remain unused\n"
    );

    // A full supported FP8 checkpoint still cannot pass one-Spark admission.
    fs::OpenOptions::new()
        .write(true)
        .open(f.model.join("model.safetensors"))
        .unwrap()
        .set_len(185_500_000_000)
        .unwrap();
    let mut launch = f.launcher();
    let error = launch.execute(Command::Install, false).unwrap_err();
    assert!(error.contains("Local launch needs"), "{error}");
    assert!(!f.root.join("config").exists());
    assert!(launch.runner.calls.is_empty());
}

#[test]
fn cold_launch_stages_only_manifest_metadata_and_keeps_receipts_private() {
    let f = Fixture::new();
    f.installed();
    let manifest = fixture_manifest(&f.model);
    let owner_sizes = b"owner's existing size cache\n";
    fs::write(f.model.join("manifest.sha256"), &manifest).unwrap();
    fs::write(f.model.join("manifest.sha256.sizes"), owner_sizes).unwrap();
    let weights = fs::read(f.model.join("model.safetensors")).unwrap();
    let mut launch = f.launcher();
    launch.runner.write_manifest_receipts = true;
    launch.execute(Command::Launch(Default::default()), false).unwrap();

    let request = launch
        .runner
        .calls
        .iter()
        .find(|e| e.label == "local start")
        .unwrap();
    let fields = text::parse(std::str::from_utf8(&request.stdin).unwrap()).unwrap();
    assert_eq!(fields["model_path"], f.model.to_string_lossy());
    assert_eq!(fields["arg.5"], f.model.to_string_lossy());
    let staged = f.root.join("state/model-manifest.sha256");
    assert_eq!(fields["env.FLASH_NVFP4_MANIFEST"], staged.to_string_lossy());
    assert_eq!(fs::read(&staged).unwrap(), manifest.as_bytes());
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        manifest.as_bytes()
    );
    assert_eq!(
        fs::read(f.model.join("manifest.sha256.sizes")).unwrap(),
        owner_sizes
    );
    assert_eq!(
        fs::read(f.model.join("model.safetensors")).unwrap(),
        weights
    );
    let sizes = fs::read_to_string(f.root.join("state/model-manifest.sha256.sizes")).unwrap();
    let manifest_hash = crate::sha256::sha(manifest.as_bytes());
    assert_eq!(sizes.lines().next(), Some(manifest_hash.as_str()));
    assert!(sizes.lines().any(|line| line == "32  model.safetensors"));
    assert!(f.root.join("state/cache/vade-manifest-v1").is_dir());
    assert!(!f.home.join(".cache").exists());

    // No checkpoint payload or discovery metadata belongs in the install tree.
    let mut pending = vec![f.root.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                assert_ne!(entry.file_name(), "config.json");
                assert_ne!(entry.file_name(), "tokenizer.json");
                assert_ne!(
                    entry.path().extension(),
                    Some(std::ffi::OsStr::new("safetensors"))
                );
                assert_ne!(fs::read(entry.path()).unwrap(), weights);
            }
        }
    }

    // A resident attach must not reread either discovery or manifest metadata.
    fs::remove_file(f.model.join("config.json")).unwrap();
    let changed_manifest = vec![b'x'; 1024 * 1024 + 1];
    fs::write(f.model.join("manifest.sha256"), &changed_manifest).unwrap();
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(launch.detector.calls, 1);
    assert_eq!((launch.runner.starts, launch.runner.chats), (1, 2));
    assert_eq!(launch.runner.worker_ops(), ["inspect", "listener"]);
    assert_eq!(launch.runner.calls.len(), 4);
    assert_eq!(fs::read(staged).unwrap(), manifest.as_bytes());
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        changed_manifest
    );
    assert_eq!(
        fs::read(f.model.join("manifest.sha256.sizes")).unwrap(),
        owner_sizes
    );
    assert_eq!(
        fs::read(f.model.join("model.safetensors")).unwrap(),
        weights
    );
}

#[test]
fn generated_private_manifest_survives_restart_and_warm_attach() {
    use std::os::unix::fs::MetadataExt;

    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    launch.runner.write_manifest_receipts = true;
    assert!(!f.model.join("manifest.sha256").exists());
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    let staged = f.root.join("state/model-manifest.sha256");
    let generated = fs::read(&staged).unwrap();
    let inode = fs::metadata(&staged).unwrap().ino();
    let sizes = fs::read(f.root.join("state/model-manifest.sha256.sizes")).unwrap();
    assert!(!generated.is_empty());
    assert!(!f.model.join("manifest.sha256").exists());
    assert!(!f.model.join("manifest.sha256.sizes").exists());

    launch.execute(Command::Down, false).unwrap();
    assert_eq!(fs::read(&staged).unwrap(), generated);
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(
        launch.runner.worker_ops(),
        ["probe", "start", "inspect", "listener"]
    );
    assert_eq!((launch.detector.calls, launch.runner.starts), (2, 2));
    assert_eq!(fs::read(&staged).unwrap(), generated);
    assert_eq!(fs::metadata(&staged).unwrap().ino(), inode);
    assert_eq!(
        fs::read(f.root.join("state/model-manifest.sha256.sizes")).unwrap(),
        sizes
    );

    fs::remove_file(f.model.join("config.json")).unwrap();
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(launch.runner.worker_ops(), ["inspect", "listener"]);
    assert_eq!(launch.runner.calls.len(), 4);
    assert_eq!(
        (
            launch.detector.calls,
            launch.runner.starts,
            launch.runner.chats
        ),
        (2, 2, 3)
    );
    assert_eq!(fs::read(&staged).unwrap(), generated);
    assert_eq!(fs::metadata(&staged).unwrap().ino(), inode);
    assert!(!f.model.join("manifest.sha256").exists());
    assert!(!f.model.join("manifest.sha256.sizes").exists());
}

#[test]
fn oversized_manifest_refuses_before_engine_start_without_copying_it() {
    let f = Fixture::new();
    f.installed();
    let manifest = vec![b'x'; 1024 * 1024 + 1];
    fs::write(f.model.join("manifest.sha256"), &manifest).unwrap();
    let mut launch = f.launcher();
    let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
    assert!(error.contains("Model checksum manifest exceeds 1 MiB."));
    assert!(error.contains("fixture engine diagnostic\nFull log: vade logs"));
    assert_eq!(launch.runner.starts, 0);
    assert_eq!(launch.runner.chats, 0);
    assert_eq!(launch.runner.worker_ops(), ["probe", "logs", "stop", "clean"]);
    assert!(!f.root.join("state/model-manifest.sha256").exists());
    assert!(!f.root.join("state/run.conf").exists());
    assert_eq!(fs::read(f.model.join("manifest.sha256")).unwrap(), manifest);
}

#[test]
fn fetch_helper_list_beside_the_checkpoint_seeds_the_private_manifest() {
    let f = Fixture::new();
    text::private_dir(&f.root.join("state")).unwrap();
    let launch = f.launcher();
    let private = f.root.join("state/model-manifest.sha256");
    // Nothing published: the engine writes its own receipt there later.
    assert_eq!(launch.stage_manifest(&f.model).unwrap(), Some(private.clone()));
    assert!(!private.exists());
    let mut sibling = f.model.as_os_str().to_owned();
    sibling.push(".sha256");
    let hash = "a".repeat(64);
    fs::write(&sibling, format!("{hash}  ./tensors/model-1.safetensors\n{hash} *./config.json\n{hash}  README.md\n")).unwrap();
    launch.stage_manifest(&f.model).unwrap();
    assert_eq!(fs::read_to_string(&private).unwrap(),
        format!("{hash}  tensors/model-1.safetensors\n{hash} *config.json\n{hash}  README.md\n"));
    // A manifest inside the checkpoint still comes first, byte for byte.
    fs::write(f.model.join("manifest.sha256"), format!("{hash}  model.safetensors\n")).unwrap();
    launch.stage_manifest(&f.model).unwrap();
    assert_eq!(fs::read_to_string(&private).unwrap(), format!("{hash}  model.safetensors\n"));
    fs::remove_file(sibling).unwrap();
}

#[test]
fn controller_receipts_keep_supplied_hashes_in_both_modes() {
    let f = Fixture::new();
    let supplied = fixture_manifest(&f.model);
    fs::write(f.model.join("manifest.sha256"), &supplied).unwrap();
    let cache = f.root.join("state/cache");
    let start = |mode: &str| {
        let state = f.root.join(format!("spark-{mode}"));
        text::private_dir(&state).unwrap();
        let mut environment = Fields::from([("env_unset.FLASH_NVFP4_MANIFEST".into(), "1".into())]);
        crate::worker::receipt(&state, &f.model, &Fields::from([("model_receipt".into(), mode.into())]), &mut environment).unwrap();
        let receipt = PathBuf::from(&environment["env.FLASH_NVFP4_MANIFEST"]);
        // Every rank expects the publisher's hashes, so a chain compares them.
        assert_eq!(fs::read_to_string(&receipt).unwrap(), supplied);
        receipt
    };
    for mode in ["private", "pipeline"] {
        run_cpu_receipt(&f.model, &start(mode), &cache, "mixed");
    }
    // Same length and header, different payload: only the hashes can tell.
    let shard = f.model.join("model.safetensors");
    fs::write(&shard, [1; 32]).unwrap();
    for mode in ["private", "pipeline"] {
        cpu_receipt(&f.model, &start(mode), &cache, "mixed", Some("SHA-256 mismatch: model.safetensors"));
    }
}

#[test]
fn exported_cache_names_do_not_erase_private_preparation_paths() {
    let _guard = crate::runner::process_test_lock();
    let mut exec = source_install_test_process("launch::tests::receipt_tests::exported_cache_child");
    // The local launcher clears inherited names; it must still pass its own.
    for name in ["XDG_CACHE_HOME", "FLASH_NVFP4_MANIFEST"] {
        exec.env.insert(name.into(), "/exported/elsewhere".into());
    }
    exec.env.insert("VADE_TEST_EXPORTED_CACHE".into(), "1".into());
    let output = crate::runner::SystemRunner { dry: false }.run(&exec).unwrap();
    assert!(output.success, "{}", output.stdout);
    assert!(output.stdout.contains("exported cache fixture complete"), "{}", output.stdout);
}

#[test]
fn exported_cache_child() {
    if std::env::var("VADE_TEST_EXPORTED_CACHE").as_deref() != Ok("1") {
        return;
    }
    assert_eq!(std::env::var("XDG_CACHE_HOME").as_deref(), Ok("/exported/elsewhere"));
    let f = Fixture::new();
    f.installed();
    fs::write(f.model.join("manifest.sha256"), fixture_manifest(&f.model)).unwrap();
    let mut launch = f.launcher();
    // The mock asserts the effective environment of the preparation helper.
    launch.execute(Command::Install, false).unwrap();
    assert_eq!(launch.runner.worker_ops(), ["warm-ptx"]);
    println!("exported cache fixture complete");
}
