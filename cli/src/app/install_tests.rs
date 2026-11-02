use super::*;

fn with_bundle() -> Fixture {
    with_bundle_machines(1)
}

fn with_bundle_machines(count: usize) -> Fixture {
    let mut f = Fixture::fp4(count, false);
    let root = f.dir.join("prebuilt");
    let dir = root.join(format!("{}-{}", "a".repeat(40), "b".repeat(40)));
    fs::create_dir_all(&dir).unwrap();
    let mut manifest = format!("format=vade-prebuilt-v1\ntarget=aarch64-unknown-linux-gnu\nengine_sha={}\ncli_sha={}\n", "a".repeat(40), "b".repeat(40));
    for name in crate::prebuilt::FILES {
        let mut bytes = vec![0; 32]; bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183;
        fs::write(dir.join(name), &bytes).unwrap();
        manifest.push_str(&format!("{name}.sha256={}\n", crate::sha256::sha(&bytes)));
    }
    fs::write(dir.join("manifest.conf"), manifest).unwrap();
    f.app.config.prebuilt_dir = Some(root);
    f
}

#[test]
fn changed_prebuilt_destination_gets_a_fresh_copy() {
    for changed in ["stage", "target", "legacy"] {
        let mut f = with_bundle();
        f.app.execute(Command::Install).unwrap();
        let old = Manifest::load(&f.app.config.build_manifest).unwrap().artifacts["a"].clone();
        match changed {
            "stage" => f.app.config.machines[0].stage_dir = "~/fresh-stage".into(),
            "target" => f.app.config.machines[0].ssh = "replacement-ssh".into(),
            "legacy" => {
                let data = fs::read_to_string(&f.app.config.build_manifest).unwrap();
                let mut fields = text::parse(&data).unwrap();
                fields.retain(|key, _| !key.ends_with(".install_ssh") && !key.ends_with(".install_stage_dir"));
                text::atomic_write(&f.app.config.build_manifest, &text::encode(&fields)).unwrap();
            }
            _ => unreachable!(),
        }
        f.app.runner.calls.clear();
        // A stale-path reuse would fail. A correct fresh stage never reaches this probe.
        f.app.runner.fail = Some("a existing prebuilt hashes".into());
        f.app.execute(Command::Install).unwrap();
        let calls = &f.app.runner.calls;
        assert!(!calls.iter().any(|e| e.label == "a existing prebuilt hashes"), "{changed}");
        assert!(calls.iter().any(|e| e.label == "a prebuilt copy"), "{changed}");
        assert!(!calls.iter().any(|e| e.label.ends_with("helper build") || e.label.ends_with("build and stage")), "{changed}");
        let machine = &f.app.config.machines[0];
        for call in calls.iter().filter(|e| e.label == "a prebuilt copy") {
            assert_eq!(call.args.last(), Some(&format!("{}:./", machine.ssh)));
        }
        let fresh = Manifest::load(&f.app.config.build_manifest).unwrap();
        let artifact = &fresh.artifacts["a"];
        assert_ne!(artifact.worker, old.worker, "{changed}");
        assert!(artifact.worker.starts_with(&format!("{}/builds/", machine.stage_dir)), "{changed}");
        assert_eq!(f.app.runner.fail.as_deref(), Some("a existing prebuilt hashes"));
    }
}

#[test]
fn rejected_bundle_builds_from_source_and_preserves_activation_on_build_failure() {
    for changed in ["byte", "target", "identity", "failed-build"] {
        let mut f = with_bundle();
        let dir = f.app.config.prebuilt_dir.as_ref().unwrap().join(format!("{}-{}", "a".repeat(40), "b".repeat(40)));
        let before = fs::read(&f.app.config.activation).unwrap();
        match changed {
            "target" | "identity" => {
                let manifest = dir.join("manifest.conf");
                let data = fs::read_to_string(&manifest).unwrap();
                fs::write(manifest, if changed == "target" { data.replace("aarch64", "x86_64") }
                    else { data.replace(&format!("engine_sha={}", "a".repeat(40)), &format!("engine_sha={}", "d".repeat(40))) }).unwrap();
            }
            _ => {
                let path = dir.join("flash.ptx");
                let mut bytes = fs::read(&path).unwrap(); bytes[24] ^= 1;
                fs::write(path, bytes).unwrap();
            }
        }
        if changed == "failed-build" { f.app.runner.fail = Some("a build and stage".into()); }
        let result = f.app.execute(Command::Install);
        if changed == "failed-build" {
            assert!(result.is_err());
            assert_eq!(fs::read(&f.app.config.activation).unwrap(), before);
        } else { result.unwrap(); }
        let calls = &f.app.runner.calls;
        assert!(!calls.iter().any(|e| e.label.contains("prebuilt")), "{changed}");
        assert!(calls.iter().any(|e| e.label == "a helper build"), "{changed}");
        assert!(calls.iter().any(|e| e.label == "a build and stage"), "{changed}");
        assert!(calls.iter().find(|e| e.label == "a prerequisites").unwrap().args.last().unwrap().contains("cargo --version"));
    }
}

struct HashFailureRunner { inner: FakeRunner, reject: &'static str, rejected: bool, source_hashes: Option<[String; 3]> }
impl Runner for HashFailureRunner {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if !self.rejected && exec.label == self.reject {
            self.inner.calls.push(exec.clone()); self.rejected = true;
            Ok(Output { success: false, stdout: String::new() })
        } else {
            let mut output = self.inner.run(exec)?;
            if exec.label.ends_with("build and stage") && output.success {
                if let Some(hashes) = &self.source_hashes {
                    output.stdout = format!("{}  binary\n{}  ptx\n{}  worker\n", hashes[0], hashes[1], hashes[2]);
                }
            }
            Ok(output)
        }
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

#[test]
fn rejected_remote_hashes_compile_a_fresh_generation_before_activation() {
    for reject in ["a prebuilt helper hash", "a prebuilt hashes", "a existing prebuilt hashes"] {
        let mut f = with_bundle();
        if reject.contains("existing") { f.app.execute(Command::Install).unwrap(); }
        f.app.runner.calls.clear();
        let mut app = App { config: f.app.config.clone(), dry: false,
            runner: HashFailureRunner { inner: std::mem::take(&mut f.app.runner), reject, rejected: false, source_hashes: None } };
        app.execute(Command::Install).unwrap();
        assert!(app.runner.rejected);
        let calls = &app.runner.inner.calls;
        let failure = calls.iter().position(|e| e.label == reject).unwrap();
        let compiler_check = calls.iter().enumerate().find(|(i, e)| *i > failure && e.label == "a prerequisites").unwrap().1;
        assert!(compiler_check.args.last().unwrap().contains("cargo --version"));
        assert!(calls.iter().any(|e| e.label == "a helper build"));
        assert!(calls.iter().any(|e| e.label == "a build and stage"));
        if reject == "a prebuilt helper hash" {
            assert!(!calls.iter().any(|e| e.label.ends_with("complete model check") && e.args.last().unwrap().contains("-prebuilt/vade")));
        }
        let manifest = Manifest::load(&app.config.build_manifest).unwrap();
        assert!(!manifest.artifacts["a"].worker.contains("-prebuilt/"));
    }
}

#[test]
fn mixed_prebuilt_source_hashes_preserve_active_generation() {
    // Check both machine orders and each artifact independently: a source
    // build need not reproduce the accepted bundle's compiler output.
    for reject in ["a existing prebuilt hashes", "b existing prebuilt hashes"] {
        for changed_hash in 0..3 {
            let mut f = with_bundle_machines(2);
            f.app.execute(Command::Install).unwrap();
            let activation = fs::read(&f.app.config.activation).unwrap();
            let previous = Manifest::load(&f.app.config.build_manifest).unwrap();
            let artifact = &previous.artifacts["a"];
            let mut hashes = [artifact.binary_sha256.clone(), artifact.ptx_sha256.clone(), artifact.worker_sha256.clone()];
            hashes[changed_hash] = "d".repeat(64);
            f.app.runner.calls.clear();
            let mut app = App { config: f.app.config.clone(), dry: false,
                runner: HashFailureRunner { inner: std::mem::take(&mut f.app.runner), reject,
                    rejected: false, source_hashes: Some(hashes) } };
            let result = app.execute(Command::Install);
            let error = result.expect_err("a mixed source/prebuilt generation must not activate");
            assert!(error.contains("different artifact hashes") && error.contains("active generation is unchanged"), "{error}");
            assert_eq!(fs::read(&app.config.activation).unwrap(), activation, "{reject}, artifact {changed_hash}");
            let mut active = app.config.clone();
            active.resolve_generation().unwrap();
            assert_eq!(Manifest::load(&active.build_manifest).unwrap().encode(), previous.encode());
            assert!(app.runner.rejected);
            let calls = &app.runner.inner.calls;
            let rebuilt = reject.split_whitespace().next().unwrap();
            assert!(calls.iter().any(|e| e.label == format!("{rebuilt} build and stage")));
            assert_eq!(calls.iter().filter(|e| e.label.ends_with("existing prebuilt hashes")).count(), 2);
            assert!(!calls.iter().any(|e| e.label == "harness build"));
        }
    }
}

#[test]
fn matching_prebuilt_source_hashes_activate_generation() {
    let mut f = with_bundle_machines(2);
    f.app.execute(Command::Install).unwrap();
    let activation = fs::read(&f.app.config.activation).unwrap();
    let previous = Manifest::load(&f.app.config.build_manifest).unwrap();
    let artifact = &previous.artifacts["a"];
    let hashes = [artifact.binary_sha256.clone(), artifact.ptx_sha256.clone(), artifact.worker_sha256.clone()];
    f.app.runner.calls.clear();
    let mut app = App { config: f.app.config.clone(), dry: false,
        runner: HashFailureRunner { inner: std::mem::take(&mut f.app.runner), reject: "b existing prebuilt hashes",
            rejected: false, source_hashes: Some(hashes) } };
    app.execute(Command::Install).unwrap();
    assert!(app.runner.rejected);
    assert_ne!(fs::read(&app.config.activation).unwrap(), activation);
    let manifest = Manifest::load(&app.config.build_manifest).unwrap();
    assert!(manifest.artifacts["a"].worker.contains("-prebuilt/"));
    assert!(!manifest.artifacts["b"].worker.contains("-prebuilt/"));
    for artifact in manifest.artifacts.values() {
        assert_eq!(artifact.binary_sha256, manifest.artifacts["a"].binary_sha256);
        assert_eq!(artifact.ptx_sha256, manifest.artifacts["a"].ptx_sha256);
        assert_eq!(artifact.worker_sha256, manifest.artifacts["a"].worker_sha256);
    }
}

fn with_host_bundle() -> Fixture {
    let f = with_bundle();
    let dir = f.app.config.prebuilt_dir.as_ref().unwrap().join(format!("{}-{}", "a".repeat(40), "b".repeat(40)));
    let mut bytes = vec![0; 64];
    if cfg!(target_os = "macos") {
        bytes[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
        let cpu: u32 = if cfg!(target_arch = "aarch64") { 0x0100_000c } else { 0x0100_0007 };
        bytes[4..8].copy_from_slice(&cpu.to_le_bytes());
    } else {
        bytes[..6].copy_from_slice(b"\x7fELF\x02\x01");
        bytes[18] = if cfg!(target_arch = "aarch64") { 183 } else { 62 };
    }
    let path = dir.join("manifest.conf");
    let mut manifest = fs::read_to_string(&path).unwrap();
    manifest.push_str(&format!("host_target={}\nharness_sha={}\n", crate::prebuilt::host_target(), "c".repeat(40)));
    for name in ["host-vade", "host-vade-harness"] {
        fs::write(dir.join(name), &bytes).unwrap();
        manifest.push_str(&format!("{name}.sha256={}\n", crate::sha256::sha(&bytes)));
    }
    fs::write(path, manifest).unwrap();
    f
}

struct HostHashRunner { inner: FakeRunner, corrupt_staged: bool }
impl Runner for HostHashRunner {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if matches!(exec.label.as_str(), "controller binary hashes" | "harness hash") {
            self.inner.calls.push(exec.clone());
            if self.corrupt_staged && exec.label == "controller binary hashes" {
                fs::write(&exec.args[2], b"changed during staging").unwrap();
            }
            let stdout = exec.args[2..].iter().map(|path| {
                let digest = crate::sha256::digest_reader(fs::File::open(path).unwrap()).unwrap();
                format!("{digest}  {path}\n")
            }).collect();
            Ok(Output { success: true, stdout })
        } else { self.inner.run(exec) }
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

#[test]
fn verified_host_payload_avoids_compilers_and_rejects_changed_staged_bytes() {
    for corrupt_staged in [false, true] {
        let mut f = with_host_bundle();
        let before = fs::read(&f.app.config.activation).unwrap();
        let mut app = App { config: f.app.config.clone(), dry: false,
            runner: HostHashRunner { inner: std::mem::take(&mut f.app.runner), corrupt_staged } };
        let result = app.execute(Command::Install);
        if corrupt_staged {
            assert!(result.unwrap_err().contains("Staged host artifact SHA256 mismatch"));
            assert_eq!(fs::read(&app.config.activation).unwrap(), before);
        } else {
            result.unwrap();
            assert!(app.config.stage_dir.join("vade-harness").is_file());
            assert!(app.config.stage_dir.join("vade").is_file());
        }
        let calls = &app.runner.inner.calls;
        assert!(!calls.iter().any(|e| e.label == "harness build" || e.label == "controller CLI build"));
        assert!(!calls.iter().any(|e| e.program.ends_with("cargo") || e.program.ends_with("rustc")));
    }
}

/// A config with `source = installed-bundle`: a launcher bundle, and no checkout at all.
fn with_installed_bundle() -> Fixture {
    let mut f = Fixture::fp4(1, false);
    let root = f.dir.join("bundles/vade-0.1.0-aaaaaaa");
    fs::create_dir_all(&root).unwrap();
    let mut conf = format!("format=vade-launch-bundle-v1\ntarget=aarch64-unknown-linux-gnu\nengine_sha={0}\ncli_sha={1}\nharness_sha={1}\n", "d".repeat(40), "e".repeat(40));
    for (name, key) in [("flash-serve", "engine_sha256"), ("flash.ptx", "ptx_sha256"), ("vade", "vade_sha256"), ("vade-harness", "harness_sha256")] {
        let mut bytes = vec![0; 32]; bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183; bytes[31] = name.len() as u8;
        fs::write(root.join(name), &bytes).unwrap();
        conf.push_str(&format!("{key}={}\n", crate::sha256::sha(&bytes)));
    }
    fs::write(root.join("launcher-bundle.conf"), conf).unwrap();
    for source in [&f.app.config.engine_source, &f.app.config.harness_source, &f.app.config.cli_source] { fs::remove_dir_all(source).unwrap(); }
    f.app.config.prebuilt_dir = Some(root);
    f.app.config.installed_bundle = true;
    f
}

fn installed_bundle_app(f: &mut Fixture) -> App<HostHashRunner> {
    crate::prebuilt::LINUX_HOST.set(true);
    App { config: f.app.config.clone(), dry: false, runner: HostHashRunner { inner: std::mem::take(&mut f.app.runner), corrupt_staged: false } }
}

#[test]
fn an_installed_bundle_gives_the_revisions_and_every_program_with_no_checkout() {
    let mut f = with_installed_bundle();
    let mut app = installed_bundle_app(&mut f);
    app.execute(Command::Install).unwrap();
    crate::prebuilt::LINUX_HOST.set(false);
    let calls = &app.runner.inner.calls;
    assert!(!calls.iter().any(|e| e.program == "git"), "a checkout was read");
    assert!(!calls.iter().any(|e| e.program.ends_with("cargo") || e.program.ends_with("rustc")));
    assert!(!calls.iter().any(|e| e.label.ends_with("source copy") || e.label.ends_with("helper build") || e.label.ends_with("build and stage")));
    assert!(calls.iter().any(|e| e.label == "a prebuilt helper hash"));
    assert!(app.config.stage_dir.join("vade-harness").is_file() && app.config.stage_dir.join("vade").is_file());
    let manifest = text::parse(&fs::read_to_string(&app.config.build_manifest).unwrap()).unwrap();
    assert_eq!((manifest["engine_sha"].as_str(), manifest["cli_sha"].as_str(), manifest["harness_source_sha"].as_str()), (&*"d".repeat(40), &*"e".repeat(40), &*"e".repeat(40)));
}

#[test]
fn an_installed_bundle_fault_stops_the_install_and_never_builds_from_source() {
    // A changed local program: the bundle is refused here, before any Spark work.
    let mut f = with_installed_bundle();
    fs::write(f.app.config.prebuilt_dir.as_ref().unwrap().join("flash-serve"), b"changed").unwrap();
    let error = installed_bundle_app(&mut f).execute(Command::Install).unwrap_err();
    crate::prebuilt::LINUX_HOST.set(false);
    assert!(error.starts_with("Cannot use the installed Vade bundle:") && error.ends_with("Run the install command again."), "{error}");
    // A helper that does not arrive whole on a Spark: no source copy follows.
    let mut f = with_installed_bundle();
    crate::prebuilt::LINUX_HOST.set(true);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: HashFailureRunner { inner: std::mem::take(&mut f.app.runner),
        reject: "a prebuilt helper hash", rejected: false, source_hashes: None } };
    let error = app.execute(Command::Install).unwrap_err();
    crate::prebuilt::LINUX_HOST.set(false);
    assert!(app.runner.rejected && error.starts_with("Prebuilt artifact rejected:"), "{error}");
    assert!(!app.runner.inner.calls.iter().any(|e| e.label.ends_with("source copy") || e.label.ends_with("build directories")));
    // This vade has no bundle beside it.
    let mut f = with_installed_bundle();
    f.app.config.prebuilt_dir = None;
    assert_eq!(f.app.execute(Command::Install).unwrap_err(), crate::app::deployment::NO_INSTALLED_BUNDLE);
}

#[test]
fn an_installed_bundle_has_no_checkout_to_update_and_the_plan_says_so() {
    let mut f = with_installed_bundle();
    let error = f.app.execute(Command::Update).unwrap_err();
    assert!(error.contains("no checkout to update") && error.contains("curl -fsSL https://vade.lol/install.sh | sh"), "{error}");
    assert!(!f.app.runner.calls.iter().any(|e| e.program == "git"));
    let plan = f.app.install_plan();
    assert!(plan.contains("the installed Vade bundle in") && plan.contains("none; the bundle holds every program") && plan.contains("none; the bundle is prebuilt"), "{plan}");
    f.app.config.prebuilt_dir = None;
    assert!(f.app.install_plan().contains("is not in a bundle"));
}

#[test]
fn host_payload_checks_target_revision_digest_and_executable_before_copying() {
    for changed in ["target", "revision", "digest", "executable"] {
        let mut f = with_host_bundle();
        let dir = f.app.config.prebuilt_dir.as_ref().unwrap().join(format!("{}-{}", "a".repeat(40), "b".repeat(40)));
        let manifest = dir.join("manifest.conf");
        let mut fields = text::parse(&fs::read_to_string(&manifest).unwrap()).unwrap();
        match changed {
            "target" => { fields.insert("host_target".into(), "other-platform".into()); }
            "revision" => { fields.insert("harness_sha".into(), "d".repeat(40)); }
            _ => {
                let path = dir.join("host-vade-harness");
                let mut bytes = fs::read(&path).unwrap(); bytes[0] ^= 1;
                fs::write(path, &bytes).unwrap();
                if changed == "executable" {
                    fields.insert("host-vade-harness.sha256".into(), crate::sha256::sha(&bytes));
                }
            }
        }
        fs::write(manifest, text::encode(&fields)).unwrap();
        f.app.execute(Command::Install).unwrap();
        let calls = &f.app.runner.calls;
        assert!(!calls.iter().any(|e| e.label.contains("prebuilt")), "{changed}");
        assert!(calls.iter().any(|e| e.label == "harness build"), "{changed}");
        assert!(calls.iter().any(|e| e.label == "controller CLI build"), "{changed}");
    }
}

struct PrepareRunner { inner: FakeRunner, fail: bool, missing_model: bool, helper_preparation: &'static str, cause: &'static str }
impl Runner for PrepareRunner {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let output = self.inner.run(exec)?;
        if exec.label.ends_with("preparation support") {
            let mut response = text::parse(&output.stdout)?;
            if !self.helper_preparation.is_empty() {
                response.insert("install_prepare".into(), self.helper_preparation.into());
            }
            Ok(Output { success: true, stdout: text::encode(&response) })
        } else if exec.label.ends_with("install preparation") {
            if self.fail {
                let reply = if self.cause.is_empty() { String::new() } else { text::encode(&Fields::from([("error".into(), self.cause.into())])) };
                return Ok(Output { success: false, stdout: reply });
            }
            Ok(Output { success: true, stdout: "ptx_cache=warm\nelapsed_ms=10.0\nptx_warmup_ms=5.0\nmodel_verify_ms=9.0\nmodel_hashed_bytes=4096\nmodel_reused_files=0\n".into() })
        } else if self.missing_model && exec.label.ends_with("complete model check") {
            Ok(Output { success: false, stdout: String::new() })
        } else { Ok(output) }
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

#[test]
fn installation_negotiates_the_selected_helper_before_preparing() {
    for install in ["source", "prebuilt", "reused-prebuilt"] {
        for engine_supports in [false, true] {
            for helper_preparation in ["", "warm-ptx-v1", "warm-ptx-v2"] {
                let mut f = with_bundle();
                if install == "source" { f.app.config.prebuilt_dir = None; }
                if install == "reused-prebuilt" { f.app.execute(Command::Install).unwrap(); }
                fs::write(f.app.config.engine_source.join("src/flash-serve.rs"),
                    if engine_supports { "warm_ptx_arg prepare_model_for_install" } else { "" }).unwrap();
                let before = fs::read(&f.app.config.activation).unwrap();
                f.app.runner.calls.clear();
                let helper_supports = helper_preparation == "warm-ptx-v1";
                let mut app = App { config: f.app.config.clone(), dry: false,
                    runner: PrepareRunner { inner: std::mem::take(&mut f.app.runner),
                        // An old helper would reject warm-ptx; the install must succeed without sending it.
                        fail: !helper_supports, missing_model: false, helper_preparation, cause: "" } };
                let capture = crate::output::testing::Capture::default();
                let printed = capture.bytes.clone();
                crate::output::with_writer(capture, || app.execute(Command::Install)).unwrap();
                let printed = String::from_utf8(printed.borrow().clone()).unwrap();
                assert_eq!(printed.lines().filter(|line| line.contains("continuing ordinary installation")).count(),
                    usize::from(engine_supports && !helper_supports));
                let calls = &app.runner.inner.calls;
                assert_eq!(calls.iter().any(|e| e.label.ends_with("preparation support")), engine_supports,
                    "{install}, engine={engine_supports}, helper={helper_preparation}");
                assert_eq!(calls.iter().any(|e| e.label.ends_with("install preparation")), engine_supports && helper_supports,
                    "{install}, engine={engine_supports}, helper={helper_preparation}");
                assert_ne!(fs::read(&app.config.activation).unwrap(), before);
                Manifest::load(&app.config.build_manifest).unwrap();
            }
        }
    }
}

#[test]
fn failed_helper_capability_query_preserves_activation() {
    let mut f = with_bundle();
    fs::write(f.app.config.engine_source.join("src/flash-serve.rs"), "warm_ptx_arg").unwrap();
    let before = fs::read(&f.app.config.activation).unwrap();
    f.app.runner.fail = Some("a preparation support".into());
    assert!(f.app.execute(Command::Install).is_err());
    assert_eq!(fs::read(&f.app.config.activation).unwrap(), before);
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with("install preparation")));
}

#[test]
fn preparation_uses_verified_artifacts_and_only_supported_model_options() {
    for (source, prebuilt, missing_model) in [("both", true, false), ("ptx", true, false), ("old", true, false), ("both", false, false), ("both", true, true)] {
        let mut f = with_bundle();
        if !prebuilt { f.app.config.prebuilt_dir = None; }
        let marker = match source {
            "both" => "warm_ptx_arg prepare_model_for_install",
            "ptx" => "warm_ptx_arg",
            _ => "",
        };
        fs::write(f.app.config.engine_source.join("src/flash-serve.rs"), marker).unwrap();
        let mut app = App { config: f.app.config.clone(), dry: false,
            runner: PrepareRunner { inner: std::mem::take(&mut f.app.runner), fail: false, missing_model, helper_preparation: "warm-ptx-v1", cause: "" } };
        let result = app.execute(Command::Install);
        // A missing model still stages the build, then fails so scripts and setup stop there.
        if missing_model {
            assert!(result.unwrap_err().starts_with("The model is missing or incomplete on at least one Spark"));
            assert!(app.config.activation.is_file());
        } else { result.unwrap(); }
        let calls = &app.runner.inner.calls;
        let preparation = calls.iter().position(|e| e.label == "a install preparation");
        if source == "old" { assert!(preparation.is_none()); continue; }
        let preparation = preparation.unwrap();
        let verified = calls.iter().position(|e| e.label == if prebuilt { "a prebuilt hashes" } else { "a build and stage" }).unwrap();
        let support = calls.iter().position(|e| e.label == "a preparation support").unwrap();
        assert!(verified < support && support < preparation);
        let request = text::parse(std::str::from_utf8(&calls[preparation].stdin).unwrap()).unwrap();
        let artifact = &Manifest::load(&app.config.build_manifest).unwrap().artifacts["a"];
        assert_eq!(calls[support].args.last(), calls[preparation].args.last());
        let support_request = text::parse(std::str::from_utf8(&calls[support].stdin).unwrap()).unwrap();
        assert_eq!(support_request["op"], "paths");
        assert_eq!(support_request["ptx_path"], artifact.ptx);
        assert_eq!(request["op"], "warm-ptx");
        assert_eq!(request["engine"], artifact.binary);
        assert_eq!(request["binary_sha256"], artifact.binary_sha256);
        assert_eq!(request["ptx_path"], artifact.ptx);
        assert_eq!(request["ptx_sha256"], artifact.ptx_sha256);
        assert_eq!(request["state_dir"], app.config.machines[0].state_dir);
        assert_eq!(request.contains_key("model_path"), source == "both" && !missing_model);
        assert!(!request.contains_key("token"));
        if !prebuilt { assert!(calls[verified].args.last().unwrap().contains("SPARK_CUBIN")); }
    }
}

#[test]
fn preparation_keeps_activation_on_failure_and_leaves_resident_servers_alone() {
    for borrowed in [false, true] {
        let mut f = with_bundle();
        if borrowed { f.app.up(&fp4_options(&[])).unwrap(); }
        fs::write(f.app.config.engine_source.join("src/flash-serve.rs"), "warm_ptx_arg prepare_model_for_install").unwrap();
        let before = fs::read(&f.app.config.activation).unwrap();
        f.app.runner.calls.clear();
        let mut app = App { config: f.app.config.clone(), dry: false,
            runner: PrepareRunner { inner: std::mem::take(&mut f.app.runner), fail: true, missing_model: false, helper_preparation: "warm-ptx-v1", cause: "" } };
        let result = app.execute(Command::Install);
        if borrowed {
            result.unwrap();
            assert!(!app.runner.inner.calls.iter().any(|e| e.label.ends_with("install preparation")));
            assert!(app.runner.inner.started.contains("a"));
        } else {
            assert!(result.unwrap_err().contains("cache preparation failed"));
            assert_eq!(fs::read(&app.config.activation).unwrap(), before);
        }
        assert!(!app.runner.inner.calls.iter().any(|e| e.label.ends_with(" stop")));
    }
}

#[test]
fn preparation_failure_names_the_helper_cause_first() {
    let busy = "This machine's GPU is in use. Stop the compute application with its original launcher, then retry; run vade doctor to check this machine.";
    for cause in [busy, ""] {
        let mut f = with_bundle();
        fs::write(f.app.config.engine_source.join("src/flash-serve.rs"), "warm_ptx_arg prepare_model_for_install").unwrap();
        let before = fs::read(&f.app.config.activation).unwrap();
        let mut app = App { config: f.app.config.clone(), dry: false,
            runner: PrepareRunner { inner: std::mem::take(&mut f.app.runner), fail: true, missing_model: false, helper_preparation: "warm-ptx-v1", cause } };
        let error = app.execute(Command::Install).unwrap_err();
        if cause.is_empty() {
            assert!(error.starts_with("Install cache preparation failed;"), "{error}");
        } else {
            assert_eq!(error, format!("a: {busy} Install cache preparation failed; the previous activation was kept."));
        }
        assert_eq!(fs::read(&app.config.activation).unwrap(), before);
    }
}

/// Shares one scripted runner between install's per-machine worker threads.
#[derive(Clone)]
struct ThreadedRunner { inner: std::sync::Arc<std::sync::Mutex<FakeRunner>>, fail: Option<&'static str> }
impl Runner for ThreadedRunner {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let result = self.inner.lock().unwrap().run(exec);
        if exec.label.ends_with("helper build") {
            // Each helper build waits for the other machine's: a sequential install never gets past this.
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.inner.lock().unwrap().calls.iter().filter(|e| e.label.ends_with("helper build")).count() < 2 {
                assert!(Instant::now() < deadline, "machine builds ran one after another");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        if self.fail == Some(exec.label.as_str()) { return Ok(Output { success: false, stdout: String::new() }); }
        result
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.lock().unwrap().interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.lock().unwrap().port_free(port) }
    fn pause(&mut self, _: Duration) {}
    fn worker(&self) -> Option<Box<dyn Runner + Send>> { Some(Box::new(self.clone())) }
}

#[test]
fn machines_build_concurrently_and_report_in_machine_order() {
    let cases = [(None, ""), (Some("b build and stage"), "Machine build did not finish. Inspect retained logs and the pinned Rust/CUDA toolchain; run vade install again."),
        (Some("a helper build"), "Helper build failed; check the pinned Rust toolchain and retained logs.")];
    for (fail, message) in cases {
        let mut f = Fixture::new(2, false);
        let before = fs::read(&f.app.config.activation).unwrap();
        let mut app = App { config: f.app.config.clone(), dry: false,
            runner: ThreadedRunner { inner: std::sync::Arc::new(std::sync::Mutex::new(std::mem::take(&mut f.app.runner))), fail } };
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || app.execute(Command::Install));
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        let at = |needle: &str| printed.find(needle).unwrap_or_else(|| panic!("{needle} missing from {printed}"));
        assert!(at("a: building engine and helper.") < at("b: building engine and helper."));
        let calls = app.runner.inner.lock().unwrap().calls.clone();
        // Every source copy is staged before any machine compiles.
        let first_build = calls.iter().position(|e| e.label.ends_with("helper build")).unwrap();
        assert!(calls.iter().enumerate().filter(|(_, e)| e.label.ends_with("source copy")).all(|(i, _)| i < first_build));
        assert_eq!(calls.iter().filter(|e| e.label.ends_with("source copy")).count(), 6);
        if fail.is_none() {
            result.unwrap();
            assert!(at("b: building engine and helper.") < at("a: all model shards present.") && at("a: all model shards present.") < at("b: all model shards present."));
            let manifest = Manifest::load(&app.config.build_manifest).unwrap();
            assert_eq!(manifest.artifacts.keys().cloned().collect::<Vec<_>>(), ["a", "b"]);
            assert!(manifest.artifacts.values().all(|artifact| artifact.binary_sha256 == "1".repeat(64)));
        } else {
            assert_eq!(result.unwrap_err(), message);
            assert_eq!(fs::read(&app.config.activation).unwrap(), before);
            // Output stops at the first failing machine, as a one-by-one install would.
            if fail == Some("a helper build") { assert!(!printed.contains("b: all model shards present.")); }
            else { assert!(printed.contains("a: all model shards present.") && !printed.contains("b: all model shards present.")); }
        }
    }
}

/// Run the real install GPU check against recorded nvidia-smi outputs.
#[test]
fn build_gpu_check_judges_recorded_samples() {
    let _process_guard = crate::runner::process_test_lock();
    use std::os::unix::fs::PermissionsExt;
    let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-build-gpu-{}", text::random_hex(8).unwrap()));
    fs::create_dir_all(&root).unwrap();
    let stub = root.join("nvidia-smi");
    fs::write(&stub, "#!/bin/sh\ncase \"$*\" in\n*query-compute-apps*) printf '%s' \"$STUB_APPS\" ;;\n*power.draw*) printf '%s' \"$STUB_POWER\" ;;\n*utilization.gpu*) printf '%s' \"$STUB_UTIL\" ;;\nesac\n").unwrap();
    fs::set_permissions(&stub, fs::Permissions::from_mode(0o700)).unwrap();
    let run = |apps: &str, power: &str, util: &str| {
        let out = std::process::Command::new("sh").args(["-c", super::super::deployment::BUILD_GPU_CHECK])
            .env("PATH", format!("{}:/usr/bin:/bin", root.display()))
            .env("STUB_APPS", apps).env("STUB_POWER", power).env("STUB_UTIL", util).output().unwrap();
        (out.status.success(), String::from_utf8(out.stdout).unwrap())
    };
    // Idle, then a stale 96 % counter at 19 W with no apps: both install.
    assert_eq!(run("", "18.50\n", "0\n"), (true, String::new()));
    assert_eq!(run("", "19.00\n", "96\n"), (true, "stale_utilization=96\n".into()));
    // A listed compute app blocks, and so does power draw alone at 30 W or more.
    assert!(!run("1234\n", "19.00\n", "0\n").0);
    assert!(!run("", "45.20\n", "0\n").0);
    assert!(!run("", "30.00\n", "0\n").0);
    assert!(!run("", "N/A\n", "0\n").0);
    fs::remove_dir_all(root).unwrap();
}

/// `FakeRunner`, with an old driver on the second Spark.
struct OldDriver { inner: FakeRunner }
impl Runner for OldDriver {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let reply = self.inner.run(exec)?;
        if exec.label == "b bundle fit" { return Ok(Output { success: true, stdout: "arch=aarch64\nglibc=2.39\ndriver=575.51.03\n".into() }); }
        Ok(reply)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

#[test]
fn a_rank_below_the_floor_makes_every_rank_build_and_says_why_in_one_line() {
    let mut f = with_bundle_machines(2);
    let root = f.app.config.prebuilt_dir.clone().unwrap();
    fs::write(root.join(format!("{}-{}", "a".repeat(40), "b".repeat(40))).join("MANIFEST"), "min_driver=580\nglibc=2.39\n").unwrap();
    let mut app = App { config: f.app.config.clone(), dry: false, runner: OldDriver { inner: std::mem::take(&mut f.app.runner) } };
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || app.execute(Command::Install)).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert_eq!(printed.lines().filter(|line| line.contains("does not fit")).collect::<Vec<_>>(),
        ["b: the bundle does not fit: it needs NVIDIA driver 580 or newer, and this Spark has 575.51.03; every Spark builds from source, which needs Rust on each Spark."], "{printed}");
    let calls = &app.runner.inner.calls;
    assert!(!calls.iter().any(|e| e.label.ends_with("prebuilt copy")));
    assert!(["a build and stage", "b build and stage"].iter().all(|label| calls.iter().any(|e| e.label == *label)));
    // The second check of each rank asks for the toolchains that the build needs.
    assert!(calls.iter().any(|e| e.label == "b prerequisites" && e.args.last().unwrap().contains("for tool in cargo rustc")));
}

#[test]
fn a_prebuilt_dir_that_serves_nothing_is_named_before_the_source_build() {
    let mut f = with_bundle();
    let root = f.app.config.prebuilt_dir.clone().unwrap();
    fs::rename(root.join(format!("{}-{}", "a".repeat(40), "b".repeat(40))), root.join("another-revision")).unwrap();
    let install = |f: &mut Fixture| {
        f.app.runner.calls.clear();
        let capture = crate::output::testing::Capture::default();
        crate::output::with_writer(capture.clone(), || f.app.execute(Command::Install)).unwrap();
        assert!(f.app.runner.calls.iter().any(|e| e.label == "a build and stage"), "a source build follows");
        String::from_utf8(capture.bytes.borrow().clone()).unwrap()
    };
    let printed = install(&mut f);
    assert!(printed.contains(&format!("No prebuilt bundle in {} matches engine {} and CLI {}; building from source, which needs Rust on each Spark.\n", root.display(), "a".repeat(40), "b".repeat(40))), "{printed}");
    f.app.config.prebuilt_dir = Some(root.join("missing"));
    let printed = install(&mut f);
    assert!(printed.contains(&format!("prebuilt_dir {} is not a folder; building from source, which needs Rust on each Spark.\n", root.join("missing").display())), "{printed}");
    f.app.config.prebuilt_dir = None;
    let printed = install(&mut f);
    assert!(!printed.contains("prebuilt"), "no prebuilt_dir, no note: {printed}");
}
