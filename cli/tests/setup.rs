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

/// An ssh that reaches every Spark and finds no cables on it.
fn reach(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("ssh"), "#!/bin/sh\nwhile [ \"${1-}\" = -o ]; do shift 2; done\nprintf 'hostname %s\\n' \"$1\"\necho vade-cables-end\n").unwrap();
    std::fs::set_permissions(dir.join("ssh"), std::fs::Permissions::from_mode(0o755)).unwrap();
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
    for line in ["[machine spark-1]", "ssh = spark-a", "model_path = ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local", "Steps:", "  vade install", "  vade doctor", "  vade up"] {
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
    reach(&home.join("bin"));
    let output = vade(&["setup", "--yes"], &home, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(text(&output.stderr).contains("Add --hosts"), "{output:?}");
    let output = vade(&["setup", "--yes", "--hosts", "a,b"], &home, &[]);
    assert!(text(&output.stderr).contains("a and b share no cable, so check that cable or add --link a:b=ADDRESS --link b:a=ADDRESS."), "{output:?}");
    let output = vade(&["setup", "--yes", "--hosts", "a,b,c", "--fp8"], &home, &[]);
    assert!(text(&output.stderr).contains("This setup has 3.") && text(&output.stderr).contains("Use 2 or 4 Sparks in --hosts"), "{output:?}");
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
    for tool in ["git", "rsync", "curl"] { stub(&home.join("bin"), tool, 1); }
    reach(&home.join("bin"));
    let bin = home.join("bin").into_os_string();
    let output = vade(&["setup", "--yes", "--hosts", "spark-a"], &home, &[("PATH", &bin)]);
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{stdout}{stderr}");
    let config = home.join("config/vade/config");
    assert!(stdout.contains(&format!("Writing the controller config to {:?}.\n", config.display().to_string())), "{stdout}");
    assert!(stdout.contains("Setup step 1 of 3: vade install\nThis step checks SSH, the GPU, the disk and the model on each Spark, then installs Vade there.\n"), "{stdout}");
    assert!(!stdout.contains("Setup step 2 of 3"), "{stdout}");
    assert!(stderr.contains("error: vade install failed, so setup stopped.") && stderr.contains("hint: Fix the cause it printed above, then run vade setup again."), "{stderr}");
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(&config).unwrap().permissions().mode() & 0o777, 0o600);
    let written = std::fs::read_to_string(&config).unwrap();
    assert!(written.starts_with("# Written by vade setup. Start: vade up\n") && written.contains("[machine spark-a]\nssh = spark-a\n"), "{written}");

    // A second run uses that config as it is, and refuses answers with no Sparks.
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[("PATH", &bin)]);
    assert!(output.status.success(), "{output:?}");
    assert!(text(&output.stdout).starts_with("Using the controller config at"), "{output:?}");
    let output = vade(&["setup", "--yes", "--model", "/data/model"], &home, &[("PATH", &bin)]);
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("A controller config already exists at") && text(&output.stderr).contains("add --hosts with the Sparks"), "{output:?}");
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
    assert_eq!(steps(&text(&output.stdout)), "  vade install\n    This step checks SSH, the GPU, the disk and the model on each Spark, then installs Vade there.\n  vade up --fp8 --hosts spark-a,spark-b\n    This step starts the engine.\n  vade doctor\n    This step checks the running engine.\n", "{output:?}");
    // With an install in place (its activation record), setup goes straight to up and doctor.
    std::fs::create_dir_all(home.join("state")).unwrap();
    std::fs::write(home.join("state/build.conf"), "").unwrap();
    let output = vade(&["setup", "--yes", "--dry-run"], &home, &[]);
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("This config already has an install, so setup skips vade install."), "{stdout}");
    assert_eq!(steps(&stdout), "  vade up --fp8 --hosts spark-a,spark-b\n    This step starts the engine.\n  vade doctor\n    This step checks the running engine.\n", "{stdout}");
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

/// A list of Sparks is the answer, so it needs no flag. --replace-local stays
/// an accepted word, so old scripts still run.
#[test]
fn hosts_over_a_local_setup_move_it_aside_with_no_flag() {
    use std::os::unix::fs::PermissionsExt;
    let home = home("replace-local");
    let root = home.join(".vade");
    std::fs::create_dir(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    let file = root.join("config");
    std::fs::write(&file, "format=vade-owner-v1\n").unwrap();
    let aside = root.join("config.local");
    let dry = vade(&["setup", "--yes", "--hosts", "spark-a", "--dry-run"], &home, &[]);
    assert!(dry.status.success(), "{dry:?}");
    assert!(text(&dry.stdout).contains(&format!("Replacing the one-Spark setup with one Spark. Setup would move the old setup to {:?}.", aside.display().to_string())), "{dry:?}");
    assert!(file.is_file() && !aside.exists() && !home.join("config/vade/config").exists());
    // Another answer with no Sparks changes nothing.
    let model = vade(&["setup", "--yes", "--model", "/data/model"], &home, &[]);
    assert!(text(&model.stderr).contains("This Spark already has local setup in") && text(&model.stderr).contains("add --hosts with the Sparks"), "{model:?}");
    assert!(file.is_file() && !aside.exists());
    // The steps stop at the failing stub, after the move and the new config.
    for tool in ["rsync", "git"] { stub(&home.join("bin"), tool, 1); }
    reach(&home.join("bin"));
    let moved = vade(&["setup", "--yes", "--replace-local", "--hosts", "spark-a"], &home, &[]);
    assert!(text(&moved.stdout).contains(&format!("Replacing the one-Spark setup with one Spark. The old setup is now in {:?}.", aside.display().to_string())), "{moved:?}");
    assert_eq!(std::fs::read_to_string(&aside).unwrap(), "format=vade-owner-v1\n");
    assert!(!file.exists() && home.join("config/vade/config").is_file(), "{moved:?}");
}

/// With no Spark list, the old word asks for nothing new.
#[test]
fn the_replace_local_word_alone_changes_nothing() {
    let home = home("replace-word");
    std::fs::create_dir_all(home.join("config/vade")).unwrap();
    std::fs::write(home.join("config/vade/config"), "[machine spark-a]\nssh = spark-a\n").unwrap();
    let output = vade(&["setup", "--yes", "--replace-local", "--dry-run"], &home, &[]);
    assert!(output.status.success() && text(&output.stdout).starts_with("Using the controller config at"), "{output:?}");
}

/// A running local engine stops first. Off a Spark that stop fails, and then nothing moves.
#[test]
fn a_local_engine_that_does_not_stop_keeps_its_setup() {
    if SPARK { return; }
    use std::os::unix::fs::PermissionsExt;
    let home = home("running");
    let root = home.join(".vade");
    std::fs::create_dir_all(root.join("state")).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(root.join("config"), "format=vade-owner-v1\n").unwrap();
    std::fs::write(root.join("state/run.conf"), "version=1\n").unwrap();
    reach(&home.join("bin"));
    let output = vade(&["setup", "--yes", "--hosts", "spark-a"], &home, &[]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(text(&output.stdout).contains("Stopping the one-Spark engine.\n"), "{output:?}");
    assert!(text(&output.stderr).contains("vade down did not finish, so the old setup stays in place."), "{output:?}");
    assert!(root.join("config").is_file() && !root.join("config.local").exists() && !home.join("config/vade/config").exists());
}

/// The fake tools first on PATH, and a home folder for each fake host.
fn fake_hosts(home: &Path, rust: bool) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let tools = home.join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    for entry in std::fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_hosts")).unwrap() {
        let path = entry.unwrap().path();
        if !rust && ["rustc", "cargo"].iter().any(|name| path.ends_with(name)) { continue; }
        std::fs::copy(&path, tools.join(path.file_name().unwrap())).unwrap();
        std::fs::set_permissions(tools.join(path.file_name().unwrap()), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    for host in ["host-a", "host-b"] { std::fs::create_dir_all(home.join("hosts").join(host)).unwrap(); }
    std::fs::write(home.join("hosts/host-a/known"), "").unwrap();
    tools
}

fn no_address(output: &Output) -> String {
    let printed = format!("{}{}", text(&output.stdout), text(&output.stderr));
    for private in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(private), "{printed}"); }
    printed
}

#[test]
fn address_targets_stay_private_in_up_setup_status_and_doctor() {
    let home = home("private-addresses");
    let tools = fake_hosts(&home, true);
    let path = std::ffi::OsString::from(format!("{}:/usr/bin:/bin", tools.display()));
    let config = home.join("controller-config");
    std::fs::write(&config, format!("[global]\nmodel_revision = {}\nstate_dir = {}/state\nstage_dir = {}/stage\n[machine 192-0-2-10]\nssh = user@192.0.2.10\n", "a".repeat(40), home.display(), home.display())).unwrap();
    let run = |args: &[&str]| vade(args, &home, &[("VADE_CONFIG", config.as_os_str()), ("FAKE_HOSTS", home.join("hosts").as_os_str()), ("PATH", path.as_os_str())]);
    for args in [&["up", "--dry-run"][..], &["setup", "--yes", "--dry-run"], &["status"], &["status", "--json"], &["doctor", "--dry-run"], &["doctor", "--json", "--dry-run"]] {
        let output = run(args);
        let printed = no_address(&output);
        assert!(output.status.success(), "{args:?}: {printed}");
        assert!(printed.contains("spark-1"), "{args:?}: {printed}");
    }
    // Real fake SSH failures must protect stderr and the returned error too.
    let output = run(&["doctor"]);
    assert!(!output.status.success());
    assert!(no_address(&output).contains("Host key verification failed."));
}

#[test]
fn address_lists_report_all_ssh_failures_before_any_cable_probe() {
    let home = home("all-ssh-failures");
    let tools = fake_hosts(&home, true);
    let path = std::ffi::OsString::from(format!("{}:/usr/bin:/bin", tools.display()));
    let hosts = ["user@192.0.2.10", "user@192.0.2.11", "user@192.0.2.12"];
    for host in hosts { std::fs::create_dir_all(home.join("hosts").join(host)).unwrap(); }
    std::fs::write(home.join("hosts").join(hosts[1]).join("known"), "").unwrap();
    // Linux checks local SSH config first. Exercise that lookup on macOS too.
    for host in hosts {
        let output = Command::new(tools.join("ssh")).args(["-G", host]).env_clear().env("FAKE_HOSTS", home.join("hosts")).output().unwrap();
        assert!(output.status.success() && output.stdout.is_empty() && output.stderr.is_empty(), "{output:?}");
    }
    assert!(!home.join("hosts/calls.log").exists());
    let config = home.join("controller-config");
    let list = hosts.join(",");
    let mut expected = String::new();
    for args in [&["up", "--sparks", list.as_str()][..], &["setup", "--yes", "--hosts", list.as_str()]] {
        let output = vade(args, &home, &[("VADE_CONFIG", config.as_os_str()), ("FAKE_HOSTS", home.join("hosts").as_os_str()), ("PATH", path.as_os_str())]);
        assert!(!output.status.success());
        let printed = no_address(&output);
        assert_eq!(printed.matches("SSH checks failed.").count(), 1, "{printed}");
        assert_eq!(printed.matches("Host key verification failed.").count(), 2, "{printed}");
        for label in ["spark-1", "spark-3"] { assert!(printed.contains(&format!("{label}:")), "{printed}"); }
        assert_eq!(printed.matches("Read docs/INSTALL.md for the SSH check command.").count(), 2, "{printed}");
        assert!(!printed.contains("Run ssh "), "{printed}");
        assert!(!config.exists());
        for host in hosts { expected.push_str(&format!("ssh {host} true\n")); }
        assert_eq!(std::fs::read_to_string(home.join("hosts/calls.log")).unwrap(), expected);
    }
    let calls = std::fs::read_to_string(home.join("hosts/calls.log")).unwrap();
    assert_eq!(calls.lines().count(), 6, "{calls}");
    for host in hosts {
        assert_eq!(calls.lines().filter(|line| *line == format!("ssh {host} true")).count(), 2, "{calls}");
        assert!(!home.join("hosts").join(host).join(".ssh").exists());
    }
}

#[test]
fn setup_reads_the_short_hostname_once_and_keeps_the_destination_in_config() {
    let home = home("short-hostname");
    let tools = fake_hosts(&home, true);
    let path = std::ffi::OsString::from(format!("{}:/usr/bin:/bin", tools.display()));
    let host = "user@192.0.2.10";
    let remote = home.join("hosts").join(host);
    std::fs::create_dir(&remote).unwrap();
    std::fs::write(remote.join("known"), "").unwrap();
    std::fs::write(remote.join("hostname"), "box-one").unwrap();
    let config = home.join("controller-config");
    let output = vade(&["setup", "--yes", "--hosts", host], &home,
        &[("VADE_CONFIG", config.as_os_str()), ("FAKE_HOSTS", home.join("hosts").as_os_str()), ("PATH", path.as_os_str())]);
    no_address(&output);
    let written = std::fs::read_to_string(&config).unwrap();
    assert!(written.contains("[machine box-one]\nssh = user@192.0.2.10\n"), "{written}");
    assert_eq!(written.matches(host).count(), 1, "{written}");
    let calls = std::fs::read_to_string(home.join("hosts/calls.log")).unwrap();
    assert_eq!(calls.matches("hostname -s").count(), 1, "{calls}");
}

/// Bundle-only setup and doctor on two fake hosts, from a clone's folder with no
/// Rust: host-b has no saved host key. The real binary stops here on a Mac, which
/// cannot take a Linux bundle; cli/src/app/tests/fake_hosts.rs runs the install.
#[test]
fn a_bundle_setup_needs_no_rust_and_doctor_shows_the_ssh_error_with_its_fix() {
    let home = home("fake-hosts");
    let tools = fake_hosts(&home, true);
    let (bundle, clone) = (home.join("bundle"), home.join("clone"));
    std::fs::create_dir(&bundle).unwrap();
    for folder in ["engine", "harness", "cli/examples"] { std::fs::create_dir_all(clone.join(folder)).unwrap(); }
    for file in ["engine/Cargo.toml", "harness/Cargo.toml"] { std::fs::write(clone.join(file), "").unwrap(); }
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/fp4-local.conf"), clone.join("cli/examples/fp4-local.conf")).unwrap();
    let program = home.join("vade");
    std::fs::copy(env!("CARGO_BIN_EXE_vade"), &program).unwrap();
    let config = home.join("config-file");
    let run = |args: &[&str], source: Option<&Path>| {
        let mut command = Command::new(&program);
        command.args(args).current_dir(&clone).env_clear().env("HOME", &home).env("VADE_BUNDLE", &bundle).env("VADE_CONFIG", &config)
            .env("FAKE_HOSTS", home.join("hosts")).env("SEQPAR_RDMA", "0").env("PATH", format!("{}:/usr/bin:/bin", tools.display()));
        if let Some(source) = source { command.env("VADE_SOURCE", source); }
        command.output().unwrap()
    };
    // With no engine, doctor checks the first Spark, so host-b goes first.
    let setup = ["setup", "--dry-run", "--yes", "--hosts", "host-b,host-a", "--link", "host-a:host-b=192.0.2.1", "--link", "host-b:host-a=192.0.2.2"];
    let output = run(&setup, None);
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.lines().any(|line| line == "source = installed-bundle") && !stdout.contains("engine_source"), "the bundle wins over this folder: {stdout}");
    // VADE_SOURCE still asks for that checkout.
    let output = run(&setup, Some(&clone));
    assert!(text(&output.stdout).contains(&format!("engine_source = {}", clone.join("engine").display())), "{output:?}");
    // The config from the dry run, then doctor on the two hosts.
    let written: String = stdout.lines().skip_while(|line| !line.starts_with("# Written by vade setup")).take_while(|line| *line != "Steps:").map(|line| format!("{line}\n")).collect();
    std::fs::write(&config, written).unwrap();
    let output = run(&["doctor"], None);
    let stdout = text(&output.stdout);
    let row = |name: &str| stdout.lines().find(|line| line.split_whitespace().skip(1).collect::<Vec<_>>().join(" ").starts_with(name)).unwrap_or_else(|| panic!("{name}:\n{stdout}"));
    assert!(row("controller tools").starts_with("ok") && row("controller tools").contains("rsync, curl and ssh found"), "{stdout}");
    assert!(row("spark-1 ssh").starts_with("FAIL") && row("spark-1 ssh").contains("ssh host-b failed: Host key verification failed. Vade never edits ~/.ssh."), "{stdout}");
    assert!(stdout.contains("next: Run ssh host-b true and accept the key, or run ssh-keyscan -H host-b >> ~/.ssh/known_hosts if host-b is the real host name, then run vade doctor."), "{stdout}");
    let calls = std::fs::read_to_string(home.join("hosts/calls.log")).unwrap();
    assert!(!calls.contains("was asked"), "{calls}");
    assert!(!home.join("hosts/host-b/.ssh").exists() && !home.join(".ssh").exists());
}

#[test]
fn a_source_config_with_no_rust_names_the_install_commands() {
    let home = home("no-rust");
    let tools = fake_hosts(&home, false);
    let config = home.join("config-file");
    std::fs::write(&config, format!("[global]\nengine_source = {0}\nharness_source = {0}\ncli_source = {0}\nstage_dir = {1}/stage\nstate_dir = {1}/state\n\n[machine a]\nssh = host-a\nmodel_path = ~/m\n", home.display(), home.display())).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vade")).arg("doctor").env_clear().env("HOME", &home).env("VADE_CONFIG", &config)
        .env("FAKE_HOSTS", home.join("hosts")).env("PATH", format!("{}:/usr/bin:/bin", tools.display())).stdin(Stdio::null()).output().unwrap();
    let stdout = text(&output.stdout);
    assert!(stdout.contains("This setup builds from source, so this computer needs Rust.") && stdout.contains("rustup toolchain install 1.99.0 --profile minimal --no-self-update"), "{stdout}");
    assert!(!stdout.contains("Cannot run local rustc"), "{stdout}");
}

/// `scripts/install.sh --source` runs this. A controller config must not take it.
#[test]
fn the_private_one_spark_install_never_takes_the_controller_route() {
    if SPARK { return; }
    let home = home("install-here");
    std::fs::create_dir_all(home.join("config/vade")).unwrap();
    std::fs::write(home.join("config/vade/config"), "[global]\n\n[machine a]\nssh = host-a\nmodel_path = ~/m\n").unwrap();
    // The controller route of vade install reads that config. This one does not.
    let controller = vade(&["install"], &home, &[]);
    assert!(!text(&controller.stderr).contains("This computer is not a Spark"), "{controller:?}");
    let here = vade(&["__install-here"], &home, &[]);
    assert_eq!(here.status.code(), Some(1), "{here:?}");
    assert!(text(&here.stderr).contains("This computer is not a Spark"), "{here:?}");
    let chosen = vade(&["__install-here"], &home, &[("VADE_CONFIG", std::ffi::OsStr::new("/no/such/config"))]);
    assert!(text(&chosen.stderr).contains("VADE_CONFIG or VADE_GROUP selects the controller installation"), "{chosen:?}");
}

/// A one-line install: a copy of vade outside any checkout, with a launcher
/// bundle named by VADE_BUNDLE, as the installed program has one next to it.
#[test]
fn hosts_with_no_checkout_write_a_config_that_needs_no_clone() {
    let home = home("no-clone");
    let bundle = home.join("bundle");
    std::fs::create_dir(&bundle).unwrap();
    let program = home.join("vade");
    std::fs::copy(env!("CARGO_BIN_EXE_vade"), &program).unwrap();
    let run = |bundle: Option<&Path>| {
        let mut command = Command::new(&program);
        command.args(["setup", "--dry-run", "--yes", "--hosts", "spark-a,spark-b", "--link", "spark-a:spark-b=192.0.2.1", "--link", "spark-b:spark-a=192.0.2.2"])
            .current_dir(&home).env_clear().env("HOME", &home).env("XDG_CONFIG_HOME", home.join("xdg")).env("PATH", "/usr/bin:/bin");
        if let Some(bundle) = bundle { command.env("VADE_BUNDLE", bundle); }
        command.output().unwrap()
    };
    let output = run(Some(&bundle));
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.lines().any(|line| line == "source = installed-bundle") && !stdout.contains("engine_source"), "{stdout}");
    // The pin comes from the copy built into the program.
    let example = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/fp4-local.conf")).unwrap();
    assert!(stdout.lines().any(|line| Some(line) == example.lines().find(|line| line.starts_with("model_revision = "))), "{stdout}");
    assert!(!home.join("xdg").exists(), "dry run wrote files");
    // No bundle and no checkout: the error from before stays.
    let output = run(None);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(text(&output.stderr).contains("More Sparks need a Vade checkout on this computer"), "{output:?}");
}
