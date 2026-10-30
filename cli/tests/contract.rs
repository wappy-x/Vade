//! The output contract in cli/docs/ux.md, checked on the real binary with no
//! network and no Spark: every help page has the same layout, every --json
//! command prints one versioned object, and every error says what failed and
//! what to run next.
use std::process::{Command, Output};

/// The binary in a sealed environment: no config, no saved setup, no color,
/// and stub ssh, rsync and curl that succeed ahead of the host's PATH ([`home`]).
/// On a fresh home Linux selects the local owner and macOS selects none, so
/// tests of the controller rows name its config explicitly ([`controller`]).
fn vade(args: &[&str], home: &std::path::Path) -> Output { vade_with(args, home, &[]) }

/// [`vade`] with extra environment variables.
fn vade_with(args: &[&str], home: &std::path::Path, env: &[(&str, &std::ffi::OsStr)]) -> Output {
    sealed(args, home, env).output().unwrap()
}

/// The command [`vade_with`] runs, for tests that choose its stdout.
fn sealed(args: &[&str], home: &std::path::Path, env: &[(&str, &std::ffi::OsStr)]) -> Command {
    let mut path = vec![home.join("bin")];
    path.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
    command
        .args(args)
        .env_clear()
        .env("PATH", std::env::join_paths(path).unwrap())
        .env("HOME", home)
        .env("VADE_HOME", home.join(".vade"))
        .env("XDG_CONFIG_HOME", home.join("config"))
        .envs(env.iter().copied());
    command
}

/// [`vade`] with an explicit controller config that does not exist yet: the
/// controller owner on every platform.
fn controller(args: &[&str], home: &std::path::Path) -> Output {
    vade_with(args, home, &[("VADE_CONFIG", home.join("cluster.conf").as_os_str())])
}

/// A private home outside /tmp (vade refuses symlinked homes), empty except
/// for `bin/`: the local tools doctor probes, as stubs that succeed.
fn home(name: &str) -> std::path::PathBuf {
    let base = std::fs::canonicalize(std::env::temp_dir()).unwrap();
    let path = base.join(format!("vade-contract-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir(&path).unwrap();
    for tool in ["ssh", "rsync", "curl"] { stub(&path.join("bin"), tool, 0); }
    path
}

/// An executable `dir/name` that prints nothing and exits `code`.
fn stub(dir: &std::path::Path, name: &str, code: i32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).unwrap();
    let file = dir.join(name);
    std::fs::write(&file, format!("#!/bin/sh\nexit {code}\n")).unwrap();
    std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn text(bytes: &[u8]) -> String { String::from_utf8(bytes.to_vec()).unwrap() }

const PAGES: &[&str] = &[
    "launch", "up", "down", "chat", "status", "config", "doctor", "detect", "logs", "verify", "bench", "test", "install",
    "update", "run", "version", "pull", "models",
];

#[test]
fn every_help_page_is_one_summary_line_then_examples_then_usage() {
    let home = home("help");
    for page in PAGES {
        let output = vade(&[page, "--help"], &home);
        assert_eq!(output.status.code(), Some(0), "{page}");
        let help = text(&output.stdout);
        assert!(output.stderr.is_empty(), "{page}");
        let lines: Vec<&str> = help.lines().collect();
        assert!(lines[0].ends_with('.') && lines[1].is_empty(), "{page}: the summary is one sentence on one line:\n{help}");
        assert_eq!(lines[2], "Examples:", "{page}: examples come first:\n{help}");
        let usage = help.find("\nUsage: vade").unwrap_or_else(|| panic!("{page}: no usage:\n{help}"));
        assert!(help.find("\nExamples:").unwrap() < usage, "{page}");
        assert!(lines.iter().all(|line| line.chars().count() <= 80), "{page}: wider than 80 columns");
        assert!(!help.chars().any(emoji), "{page}: no emoji");
        // `vade help <command>` is the same page.
        assert_eq!(text(&vade(&["help", page], &home).stdout), help, "{page}");
    }
    let top = text(&vade(&["--help"], &home).stdout);
    for page in PAGES.iter().filter(|page| **page != "version") {
        assert!(top.contains(&format!("\n  {page:<7}  ")), "vade --help lists {page}");
    }
    let _ = std::fs::remove_dir_all(home);
}

fn emoji(c: char) -> bool {
    matches!(c as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE0F)
}

#[test]
fn json_commands_print_one_versioned_object_and_nothing_else() {
    let home = home("json");
    for (args, schema, code, explicit) in [
        (&["--version", "--json"][..], "vade.version.v1", 0, false),
        (&["version", "--json"], "vade.version.v1", 0, false),
        (&["config", "--json"], "vade.config.v1", 0, false),
        (&["detect", "--json", "--dry-run"], "vade.detect.v1", 0, false),
        (&["doctor", "--json", "--dry-run"], "vade.doctor.v1", 1, true),
        (&["doctor", "--json"], "vade.doctor.v1", 1, true),
        // Failures that end before the command's own report still answer in
        // its schema: a missing explicit config, an unreadable key file.
        (&["config", "--json"], "vade.config.v1", 1, true),
        (&["doctor", "--json", "--api-key-file", "/nonexistent/vade-key"], "vade.doctor.v1", 1, true),
    ] {
        let output = if explicit { controller(args, &home) } else { vade(args, &home) };
        let stdout = text(&output.stdout);
        assert_eq!(output.status.code(), Some(code), "{args:?}: {stdout}{}", text(&output.stderr));
        assert!(output.stderr.is_empty(), "{args:?}: --json keeps stderr quiet: {}", text(&output.stderr));
        assert_eq!(stdout.lines().count(), 1, "{args:?}: one line");
        assert!(stdout.starts_with(&format!("{{\"schema\":\"{schema}\"")), "{args:?}: {stdout}");
        assert!(!stdout.contains('\x1b'), "{args:?}");
    }
    let config = text(&vade(&["config", "--json"], &home).stdout);
    assert_eq!(config, "{\"schema\":\"vade.config.v1\",\"setup\":\"none\",\"path\":null,\"settings\":null,\"error\":null,\"next\":null}\n");
    let missing = text(&controller(&["config", "--json"], &home).stdout);
    assert!(missing.starts_with("{\"schema\":\"vade.config.v1\",\"setup\":\"controller\",\"path\":") && missing.contains("\"error\":\"VADE_CONFIG names a missing file: "), "{missing}");
    // A controller with no config still answers in its schema: one failed config row.
    let doctor = text(&controller(&["doctor", "--json", "--dry-run"], &home).stdout);
    for field in ["\"mode\":\"controller\"", "\"dry_run\":true", "\"result\":\"fail\"", "\"failed\":1", "\"checks\":[{\"name\":\"config\",\"state\":\"fail\""] {
        assert!(doctor.contains(field), "{field} missing from {doctor}");
    }
    let _ = std::fs::remove_dir_all(home);
}

/// Doctor without a config probes the local tools: a missing or failing tool
/// is its own failed row with a next step, still in one JSON object.
#[test]
fn doctor_reports_missing_and_failing_local_tools_in_its_schema() {
    let home = home("tools");
    let only = home.join("only");
    stub(&only, "ssh", 1);
    stub(&only, "curl", 0);
    // Failed probes keep their logs; keep them inside this home.
    let tmp = home.join("tmp");
    std::fs::create_dir(&tmp).unwrap();
    let env = [("VADE_CONFIG", home.join("cluster.conf").into_os_string()), ("PATH", only.into_os_string()), ("TMPDIR", tmp.clone().into_os_string())];
    let output = vade_with(&["doctor", "--json"], &home, &env.iter().map(|(k, v)| (*k, v.as_os_str())).collect::<Vec<_>>());
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(stdout.starts_with("{\"schema\":\"vade.doctor.v1\""), "{stdout}");
    for field in [
        "{\"name\":\"ssh\",\"state\":\"fail\"", "Install ssh, then run vade doctor.",
        "{\"name\":\"rsync\",\"state\":\"fail\"", "Install rsync, then run vade doctor.",
        "{\"name\":\"curl\",\"state\":\"ok\"", "\"failed\":3",
    ] {
        assert!(stdout.contains(field), "{field} missing from {stdout}");
    }
    // The only stderr is where each failed probe's logs were kept.
    let stderr = text(&output.stderr);
    let kept = |tool: &str, line: &str| line.starts_with(&format!("local {tool} logs retained privately at {}/", tmp.display()));
    assert!(stderr.lines().all(|line| kept("ssh", line) || kept("rsync", line)), "{stderr}");
    let _ = std::fs::remove_dir_all(home);
}

/// The documented fresh-home selection: on Linux doctor and status use the
/// local owner (a Spark before its first launch); elsewhere there is no setup.
#[test]
fn a_fresh_home_selects_the_local_owner_only_on_linux() {
    let home = home("fresh");
    let doctor = vade(&["doctor", "--json", "--dry-run"], &home);
    let stdout = text(&doctor.stdout);
    if cfg!(target_os = "linux") {
        assert_eq!(doctor.status.code(), Some(0), "{stdout}");
        assert!(stdout.contains("\"mode\":\"local\"") && stdout.contains("\"result\":\"plan\"") && stdout.contains("\"failed\":0"), "{stdout}");
        let plain = vade(&["doctor", "--dry-run"], &home);
        assert_eq!(plain.status.code(), Some(0));
        assert!(text(&plain.stdout).starts_with("Dry run: "), "{}", text(&plain.stdout));
    } else {
        assert_eq!(doctor.status.code(), Some(1), "{stdout}");
        assert!(stdout.contains("\"mode\":\"none\"") && stdout.contains("\"checks\":[{\"name\":\"config\",\"state\":\"fail\""), "{stdout}");
    }
    let status = vade(&["status"], &home);
    if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        assert_eq!(status.status.code(), Some(0));
        assert!(text(&status.stdout).contains("stopped"), "{}", text(&status.stdout));
    } else {
        assert_eq!(status.status.code(), Some(1));
        assert!(status.stdout.is_empty() && text(&status.stderr).starts_with("error: "), "{}", text(&status.stderr));
    }
    assert!(!home.join(".vade").exists(), "read-only commands create nothing");
    let _ = std::fs::remove_dir_all(home);
}

/// An owned-engine record keeps the local owner even when its config is
/// lost: `vade config` names that setup and fails, as status does.
#[test]
fn config_keeps_the_local_owner_of_an_orphan_engine_record() {
    use std::os::unix::fs::PermissionsExt;
    let home = home("orphan");
    let state = home.join(".vade/state");
    std::fs::create_dir_all(&state).unwrap();
    for dir in [home.join(".vade"), state.clone()] { std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).unwrap(); }
    std::fs::write(state.join("run.conf"), "version=1\n").unwrap();
    let json = vade(&["config", "--json"], &home);
    assert_eq!(json.status.code(), Some(1));
    let stdout = text(&json.stdout);
    assert!(stdout.starts_with("{\"schema\":\"vade.config.v1\",\"setup\":\"local\",") && stdout.contains("\"settings\":null,\"error\":\"An owned engine has no saved config.\""), "{stdout}");
    assert!(json.stderr.is_empty());
    let plain = vade(&["config"], &home);
    assert_eq!(plain.status.code(), Some(1));
    assert!(text(&plain.stdout).starts_with("Setup  local "), "{}", text(&plain.stdout));
    assert_eq!(text(&plain.stderr), "error: An owned engine has no saved config.\n hint: Preserve the state folder and run vade down.\n");
    let _ = std::fs::remove_dir_all(home);
}

/// A config that cannot be read fails even when the reader closes stdout
/// first; only an otherwise successful command exits 0 on a closed pipe.
#[test]
fn a_config_error_keeps_its_status_when_stdout_is_closed() {
    use std::os::{fd::OwnedFd, unix::net::UnixStream};
    let home = home("config-closed");
    let config = home.join("missing.conf");
    for args in [&["config"][..], &["config", "--json"]] {
        let env = [("VADE_CONFIG", config.as_os_str())];
        assert_eq!(vade_with(args, &home, &env).status.code(), Some(1), "{args:?}");
        let (writer, reader) = UnixStream::pair().unwrap();
        drop(reader);
        let closed = sealed(args, &home, &env).stdout(std::process::Stdio::from(OwnedFd::from(writer))).output().unwrap();
        assert_eq!(closed.status.code(), Some(1), "{args:?}: {closed:?}");
        assert!(!text(&closed.stderr).contains("panicked"), "{args:?}: {closed:?}");
    }
    let _ = std::fs::remove_dir_all(home);
}

/// Terminal controls in the config path are escaped in the table; JSON keeps
/// the path through its own escaping.
#[test]
fn config_escapes_terminal_controls_in_its_file_path() {
    let home = home("config-controls");
    let config = home.join("red\u{1b}[31mpath\nnext.conf");
    let env = [("VADE_CONFIG", config.as_os_str())];
    let plain = vade_with(&["config"], &home, &env);
    assert_eq!(plain.status.code(), Some(1));
    let stdout = text(&plain.stdout);
    assert!(!stdout.contains('\x1b') && stdout.contains("red\\u{1b}[31mpath\\nnext.conf"), "{stdout:?}");
    assert_eq!(stdout.lines().filter(|line| line.starts_with("File ")).count(), 1, "{stdout:?}");
    let json = vade_with(&["config", "--json"], &home, &env);
    let stdout = text(&json.stdout);
    assert!(!stdout.contains('\x1b') && stdout.contains("red\\u001b[31mpath\\nnext.conf"), "{stdout:?}");
    let _ = std::fs::remove_dir_all(home);
}

#[test]
fn doctor_rows_keep_their_columns_and_end_with_pass_or_fail() {
    let home = home("doctor");
    let output = controller(&["doctor", "--dry-run"], &home);
    assert_eq!(output.status.code(), Some(1));
    let stdout = text(&output.stdout);
    let rows: Vec<&str> = stdout.lines().collect();
    assert!(rows[0].starts_with("FAIL  config                "), "{stdout}");
    assert!(rows[1].starts_with("      next: Run vade launch on a Spark"), "{stdout}");
    assert!(rows.last().unwrap().starts_with("FAIL  0 ok, 1 failed, 0 skipped in "), "{stdout}");
    let stderr = text(&output.stderr);
    assert_eq!(stderr, "error: 1 doctor check failed.\n hint: Run the next: step under each FAIL row, then run vade doctor again.\n");
    let _ = std::fs::remove_dir_all(home);
}

#[test]
fn errors_say_what_failed_and_name_the_next_command() {
    let home = home("errors");
    for args in [
        &["frobnicate"][..], &["up", "--bogus"], &["up", "--mtp", "9"], &["launch", "--context"], &["down", "--force"],
        &["status", "--bogus"], &["config", "--all"], &["doctor", "--bogus"], &["detect", "--watch"], &["down", "--json"],
        &["--version", "now"], &["help", "nonsense"], &["verify", "--bogus"], &["verify", "-n", "0"],
        &["up", "--context", "4096", "--output", "8192"], &["run"], &["status"],
    ] {
        // `status` is a controller error only with a controller selected; on
        // a fresh Linux home it is the local owner's "stopped".
        let output = if args == ["status"] { controller(args, &home) } else { vade(args, &home) };
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}: errors never go to stdout: {}", text(&output.stdout));
        let stderr = text(&output.stderr);
        let lines: Vec<&str> = stderr.lines().collect();
        assert!(lines[0].starts_with("error: ") && lines[0].len() > "error: ".len(), "{args:?}: {stderr}");
        // The next step: a hint line, or an error that is itself the fix.
        let hint = lines.iter().find(|line| line.starts_with(" hint: "));
        let fix = ["Use ", "Set ", "Give ", "Choose "].iter().any(|verb| lines[0]["error: ".len()..].starts_with(verb));
        assert!(hint.is_some_and(|hint| hint.contains("vade") || hint.contains("--")) || fix, "{args:?}: no next step:\n{stderr}");
        assert!(!stderr.contains('\x1b'), "{args:?}: plain on a pipe");
    }
    let _ = std::fs::remove_dir_all(home);
}

/// A config that is not UTF-8 or names an unknown key says what to fix, not
/// a permission problem, and points at the current example.
#[test]
fn config_errors_name_the_encoding_and_the_current_example() {
    let home = home("config-content");
    let config = home.join("cluster.conf");
    std::fs::write(&config, b"[global]\nmodel = \xff\xfe\n").unwrap();
    let stderr = text(&controller(&["status"], &home).stderr);
    assert!(stderr.contains("is not UTF-8 text") && !stderr.contains("permissions"), "{stderr}");
    std::fs::write(&config, "[global]\nmodle = qwen3.8-flash-next-fp4-local\n").unwrap();
    let stderr = text(&controller(&["status"], &home).stderr);
    assert!(stderr.contains("Unknown config key modle in global") && stderr.contains("cli/examples/fp4-local.conf"), "{stderr}");
    let _ = std::fs::remove_dir_all(home);
}

#[test]
fn models_json_failures_are_objects_and_the_table_survives_a_malformed_record() {
    let home = home("models");
    // A failure after the arguments parse is the vade.models.v1 object on stdout.
    let file = home.join("not-a-folder");
    std::fs::write(&file, "x").unwrap();
    let output = vade(&["models", file.to_str().unwrap(), "--json"], &home);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert!(output.stderr.is_empty(), "{}", text(&output.stderr));
    assert!(stdout.starts_with("{\"schema\":\"vade.models.v1\",\"error\":\"Cannot read the model folder") && stdout.ends_with("}\n") && stdout.lines().count() == 1, "{stdout}");
    // A usage error still goes to stderr.
    let output = vade(&["models", "--bogus", "--json"], &home);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty() && text(&output.stderr).contains("Unexpected argument"), "{}", text(&output.stderr));
    // A record with a malformed commit prints the table with no commit.
    let model = home.join("models/org/model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join(".vade-model.json"), r#"{"repo":"org/name","commit":"0123456789aé0123456789"}"#).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let output = vade(&["models"], &home);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(stdout.lines().any(|line| line.starts_with("  org/model ") && line.contains(" org/name@? ")), "{stdout}");
    let _ = std::fs::remove_dir_all(home);
}
