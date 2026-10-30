//! `vade setup` and the `vade install --dry-run` plan on the real binary, with
//! no network and no Spark. The walk-through's answers and the config it
//! writes have unit tests in cli/src/setup/walk_tests.rs.
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const SPARK: bool = cfg!(all(target_os = "linux", target_arch = "aarch64"));

/// A private home outside /tmp (vade refuses symlinked homes) with a `bin/`
/// of stubs that run ahead of the host's PATH.
fn home(name: &str) -> PathBuf {
    let base = std::fs::canonicalize(std::env::temp_dir()).unwrap();
    let path = base.join(format!("vade-setup-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir(&path).unwrap();
    path
}

/// An executable `dir/name` that prints nothing and exits `code`.
fn stub(dir: &Path, name: &str, code: i32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).unwrap();
    let file = dir.join(name);
    std::fs::write(&file, format!("#!/bin/sh\nexit {code}\n")).unwrap();
    std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// The binary in a sealed environment, with no terminal on standard input.
fn vade(args: &[&str], home: &Path, env: &[(&str, &std::ffi::OsStr)]) -> Output {
    let mut path = vec![home.join("bin")];
    path.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
    Command::new(env!("CARGO_BIN_EXE_vade"))
        .args(args)
        .env_clear()
        .env("PATH", std::env::join_paths(path).unwrap())
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .envs(env.iter().copied())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String { String::from_utf8(bytes.to_vec()).unwrap() }

#[test]
fn dry_run_prints_the_config_and_steps_and_writes_nothing() {
    let home = home("dry");
    let output = vade(&["setup", "--dry-run", "--yes", "--hosts", "spark-a"], &home, &[]);
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.starts_with("Dry run: setup writes nothing and starts nothing.\n"), "{stdout}");
    for line in ["[machine spark-a]", "ssh = spark-a", "model_path = ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local", "Steps:", "  vade install", "  vade doctor", "  vade up"] {
        assert!(stdout.lines().any(|row| row == line), "{line}:\n{stdout}");
    }
    assert!(stdout.contains(&format!("state_dir = {}", home.join(".vade/controller/state").display())), "{stdout}");
    // The pinned revision comes from the checkout's example config.
    let example = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/fp4-local.conf")).unwrap();
    let revision = example.lines().find(|line| line.starts_with("model_revision = ")).unwrap();
    assert!(stdout.lines().any(|line| line == revision), "{stdout}");
    assert!(!home.join("config").exists() && !home.join(".vade").exists(), "dry run wrote files");
}

#[test]
fn without_a_terminal_setup_asks_for_yes() {
    let home = home("no-terminal");
    let output = vade(&["setup"], &home, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("Standard input is not a terminal") && stderr.contains("hint: Add --yes"), "{stderr}");
    assert!(!home.join("config").exists());
}

#[test]
fn yes_off_a_spark_names_the_missing_flag() {
    if SPARK { return; }
    let home = home("no-hosts");
    let output = vade(&["setup", "--yes"], &home, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(text(&output.stderr).contains("Add --hosts"), "{output:?}");
    let output = vade(&["setup", "--yes", "--hosts", "a,b"], &home, &[]);
    assert!(text(&output.stderr).contains("Add --link a:b=ADDRESS"), "{output:?}");
    let output = vade(&["setup", "--yes", "--hosts", "a,b,c", "--fp8"], &home, &[]);
    assert!(text(&output.stderr).contains("FP8 runs on 2, 4 or 6 Sparks, and this setup has 3"), "{output:?}");
    assert!(!home.join("config").exists());
}

#[test]
fn bad_flags_name_the_fix() {
    let home = home("flags");
    for (args, expected) in [
        (&["setup", "--hosts", "a,a"][..], "distinct SSH destinations"),
        (&["setup", "--link", "a-b=192.0.2.1"], "FROM:TO=ADDRESS"),
        (&["setup", "--link", "a:b=192.0.2.1:80"], "without a port"),
        (&["setup", "--link", "a:b=192.0.2.1", "--link", "a:b=192.0.2.9"], "Give --link a:b only once"),
        (&["setup", "--fp8", "--fp4-local"], "Choose --fp4-local or --fp8"),
        (&["setup", "--yes", "--yes"], "Use --yes only once"),
        (&["setup", "--hosts=a"], "Put a space between --hosts and its value"),
        (&["setup", "--context", "8192"], "Unknown option \"--context\" for vade setup"),
    ] {
        let output = vade(args, &home, &[]);
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        assert!(text(&output.stderr).contains(expected), "{args:?}: {}", text(&output.stderr));
    }
}

#[test]
fn setup_writes_a_private_config_then_stops_at_the_first_failed_step() {
    let home = home("write");
    // git fails, so vade install stops at its first check and nothing reaches a network.
    for tool in ["git", "ssh", "rsync", "curl"] { stub(&home.join("bin"), tool, 1); }
    let bin = home.join("bin").into_os_string();
    let output = vade(&["setup", "--yes", "--hosts", "spark-a"], &home, &[("PATH", &bin)]);
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{stdout}{stderr}");
    let config = home.join("config/vade/config");
    assert!(stdout.contains("Wrote the controller config at") && stdout.contains("Setup step 1 of 3: vade install\n"), "{stdout}");
    assert!(!stdout.contains("Setup step 2 of 3"), "{stdout}");
    assert!(stderr.contains("error: vade install failed, so setup stopped.") && stderr.contains("hint: Fix the cause it printed above, then run vade setup again."), "{stderr}");
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(&config).unwrap().permissions().mode() & 0o777, 0o600);
    let written = std::fs::read_to_string(&config).unwrap();
    assert!(written.starts_with("# Written by vade setup. Start: vade up\n") && written.contains("[machine spark-a]\nssh = spark-a\n"), "{written}");

    // A second run uses that config as it is, and refuses answers that would change it.
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[("PATH", &bin)]);
    assert!(output.status.success(), "{output:?}");
    assert!(text(&output.stdout).starts_with("Using the controller config at"), "{output:?}");
    let output = vade(&["setup", "--yes", "--hosts", "spark-b"], &home, &[("PATH", &bin)]);
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("A controller config already exists at"), "{output:?}");
    assert_eq!(std::fs::read_to_string(&config).unwrap(), written);
}

#[test]
fn a_hand_written_fp8_config_starts_its_sparks_and_an_install_is_not_repeated() {
    let home = home("existing");
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let config = home.join("config/vade/config");
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    // Written by hand: no setup line to say how it starts.
    std::fs::write(&config, format!(
        "[global]\nengine_source = {0}/engine\nharness_source = {0}/harness\ncli_source = {0}/cli\nstage_dir = {1}/stage\nstate_dir = {1}/state\n\
         model = qwen3.8-flash-next-fp8\nmodel_repo = Qwen/Qwen3.8-Flash-Next-FP8\nmodel_revision = {2}\n\n\
         [machine spark-a]\nssh = spark-a\nfabric.spark-b = 192.0.2.1\n\n[machine spark-b]\nssh = spark-b\nfabric.spark-a = 192.0.2.2\n",
        checkout.display(), home.display(), "b".repeat(40))).unwrap();
    let steps = |stdout: &str| stdout.split("Steps:\n").nth(1).unwrap_or_default().to_owned();
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[]);
    assert!(output.status.success(), "{output:?}");
    // Doctor runs last, on the engine that up started, so it checks both Sparks.
    assert_eq!(steps(&text(&output.stdout)), "  vade install\n  vade up --fp8 --hosts spark-a,spark-b\n  vade doctor\n", "{output:?}");
    // With an install in place (its activation record), setup goes straight to up and doctor.
    std::fs::create_dir_all(home.join("state")).unwrap();
    std::fs::write(home.join("state/build.conf"), "").unwrap();
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[]);
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("This config already has an install, so setup skips vade install."), "{stdout}");
    assert_eq!(steps(&stdout), "  vade up --fp8 --hosts spark-a,spark-b\n  vade doctor\n", "{stdout}");
}

/// Run vade `args` with a controller config while `tool` waits as a long
/// build or probe does (vade runs it in its own session), send SIGTERM to the
/// vade process alone, and return its output. Panics if the waiting tool
/// outlives vade.
fn stop_while_tool_waits(name: &str, args: &[&str], tool: &str, activation: bool) -> Output {
    let home = home(name);
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    std::fs::create_dir_all(home.join("config/vade")).unwrap();
    std::fs::write(home.join("config/vade/config"), format!(
        "[global]\nengine_source = {0}/engine\nharness_source = {0}/harness\ncli_source = {0}/cli\nstage_dir = {1}/stage\nstate_dir = {1}/state\nmodel_revision = {2}\n\n[machine spark-a]\nssh = spark-a\n",
        checkout.display(), home.display(), "a".repeat(40))).unwrap();
    if activation {
        std::fs::create_dir_all(home.join("state")).unwrap();
        std::fs::write(home.join("state/build.conf"), "").unwrap();
    }
    use std::os::unix::fs::PermissionsExt;
    let bin = home.join("bin");
    for other in ["git", "ssh", "rsync", "curl", "rustc", "cargo"] { if other != tool { stub(&bin, other, 0); } }
    std::fs::write(bin.join(tool), format!("#!/bin/sh\necho $$ > \"{}\"\nexec sleep 60\n", home.join("tool.pid").display())).unwrap();
    std::fs::set_permissions(bin.join(tool), std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut path = vec![bin];
    path.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
    let child = Command::new(env!("CARGO_BIN_EXE_vade"))
        .args(args)
        .env_clear()
        .env("PATH", std::env::join_paths(path).unwrap())
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    let started = std::time::Instant::now();
    let pid = loop {
        if let Ok(pid) = std::fs::read_to_string(home.join("tool.pid")) { if !pid.trim().is_empty() { break pid.trim().to_owned(); } }
        assert!(started.elapsed() < std::time::Duration::from_secs(30), "vade {args:?} never reached {tool}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let output = child.wait_with_output().unwrap();
    assert!(started.elapsed() < std::time::Duration::from_secs(30), "vade {args:?} waited for {tool}");
    let alive = Command::new("kill").args(["-0", &pid]).stderr(Stdio::null()).status().unwrap().success();
    if alive { let _ = Command::new("kill").args(["-9", &pid]).status(); }
    assert!(!alive, "{tool} kept running after vade {args:?} stopped: {output:?}");
    output
}

#[test]
fn a_stop_sent_to_setup_alone_stops_the_install_and_its_build() {
    // git stands in for a long build: install runs it first, in its own session.
    let output = stop_while_tool_waits("stop-install", &["setup", "--yes"], "git", false);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("Setup was interrupted during vade install"), "{stderr}");
}

#[test]
fn a_stop_sent_to_doctor_stops_its_probe() {
    // Setup's last step: doctor checks the controller tools before the Sparks.
    let output = stop_while_tool_waits("stop-doctor", &["doctor"], "rsync", true);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
}

#[test]
fn a_stop_sent_to_doctor_fabric_stops_its_ssh_probe() {
    let output = stop_while_tool_waits("stop-fabric", &["doctor", "--fabric"], "ssh", true);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
}

#[test]
fn install_dry_run_prints_the_plan_before_the_commands() {
    let home = home("plan");
    for tool in ["ssh", "rsync", "curl"] { stub(&home.join("bin"), tool, 0); }
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let config = home.join("cluster.conf");
    std::fs::write(&config, format!(
        "[global]\nengine_source = {0}/engine\nharness_source = {0}/harness\ncli_source = {0}/cli\nstage_dir = {1}/stage\nstate_dir = {1}/state\nmodel_revision = {2}\n\n[machine spark-a]\nssh = spark-a\n",
        checkout.display(), home.display(), "a".repeat(40))).unwrap();
    let output = vade(&["install", "--dry-run"], &home, &[("VADE_CONFIG", config.as_os_str())]);
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    let plan = stdout.split("Commands it would run:\n").next().unwrap();
    for row in ["Plan for vade install:", "  Where      1 Spark over SSH: spark-a",
        "  Files      a source build on each Spark (the config sets no prebuilt_dir)",
        "  Toolchain  Rust 1.99.0 and nightly-2026-10-01 on each Spark, for a source build",
        "  Model      spark-a ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"] {
        assert!(plan.lines().any(|line| line == row), "{row}:\n{stdout}");
    }
    assert!(stdout.contains("Commands it would run:\n$ git -C "), "{stdout}");
}

#[test]
fn install_off_a_spark_without_a_config_names_setup() {
    if SPARK { return; }
    let home = home("not-a-spark");
    for args in [&["install"][..], &["install", "--dry-run"]] {
        let output = vade(args, &home, &[]);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        let stderr = text(&output.stderr);
        assert!(stderr.contains("This computer is not a Spark, and it has no controller config yet.") && stderr.contains("hint: Run vade setup"), "{stderr}");
    }
}

#[test]
fn saved_local_defaults_win_over_a_default_controller_config_as_in_dispatch() {
    let home = home("order");
    std::fs::create_dir_all(home.join(".vade")).unwrap();
    std::fs::write(home.join(".vade/config"), "").unwrap();
    std::fs::create_dir_all(home.join("config/vade")).unwrap();
    std::fs::write(home.join("config/vade/config"), "[machine spark-a]\nssh = spark-a\n").unwrap();
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[]);
    assert!(output.status.success(), "{output:?}");
    let stdout = text(&output.stdout);
    assert!(stdout.starts_with(&format!("Using the local setup in {}.", home.join(".vade/config").display())), "{stdout}");
    assert!(stdout.contains("Where: this Spark\nModel: the model saved in the local setup\n"), "{stdout}");
    // An explicit VADE_CONFIG selects that controller config instead.
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[("VADE_CONFIG", home.join("config/vade/config").as_os_str())]);
    assert!(text(&output.stdout).starts_with("Using the controller config at"), "{output:?}");
}
