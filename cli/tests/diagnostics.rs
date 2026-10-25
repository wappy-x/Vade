//! Public executable checks: fail before I/O, preserve evidence, escape terminals.
use std::{fs, path::PathBuf, process::{Command, Output}, sync::atomic::{AtomicUsize, Ordering}};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!("vade-diagnostic-{}-{}-{label}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command.env("VADE_HOME", self.0.join("home")).env("VADE_CONFIG", self.0.join("missing-config"))
            .env_remove("VADE_API_KEY_FILE").env_remove("VADE_API_KEY")
            // CLICOLOR_FORCE colours even captured output; safe() rejects every ESC byte.
            .env_remove("CLICOLOR_FORCE");
        command
    }
    fn config(&self) -> PathBuf {
        let path = self.0.join("config");
        fs::write(&path, "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nstate_dir = state\nmodel = qwen3.8-flash-next-fp4-local\n[machine spark]\nssh = fixture-no-network\nmodel_path = /models/fixture\n").unwrap();
        path
    }
}
impl Drop for Fixture { fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); } }
fn safe(output: &[u8]) -> &str {
    let text = std::str::from_utf8(output).unwrap();
    assert!(!text.chars().any(|c| (c.is_control() && c != '\n' && c != '\t') || matches!(c, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' | '\u{2028}' | '\u{2029}')), "terminal control in {text:?}");
    text
}
fn failure(output: &Output, expected: &[&str]) {
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty(), "errors belong on stderr");
    let stderr = safe(&output.stderr);
    for text in expected { assert!(stderr.contains(text), "missing {text:?} in {stderr:?}"); }
}
#[test]
fn argument_errors_identify_the_problem_before_config_or_network_io() {
    let fixture = Fixture::new("options");
    for (args, expected) in [
        (vec!["statuz"], vec!["Unknown command", "statuz", "vade --help"]),
        (vec!["up", "--spark"], vec!["Unknown option", "--spark", "vade up --help"]),
        (vec!["run", "qwen3-8b", "--porrt"], vec!["Unknown option", "--porrt", "vade run --help"]),
        (vec!["run", "qwen3-8b", "--port"], vec!["Give --port a value", "vade run --help"]),
        (vec!["run", "qwen3-8b", "--port", "8080", "--port", "8081"], vec!["Use --port only once", "vade run --help"]),
        (vec!["up", "--context"], vec!["Give --context a value", "vade up --help"]),
        (vec!["status", "--wat"], vec!["Unexpected argument", "--wat", "vade status --help"]),
    ] { failure(&fixture.command().args(args).output().unwrap(), &expected); }
    for hostile in ["bad\x1b]52;c;payload\x07", "bad\rforged", "bad\nforged", "bad\u{009b}2J", "bad\u{202e}txt", "bad\u{2066}txt"] {
        for prefix in [&[][..], &["up"][..], &["run", "qwen3-8b"][..]] {
            let output = fixture.command().args(prefix).arg(hostile).output().unwrap();
            failure(&output, &["Unknown", "bad", "--help"]);
            // One error line and one hint line; a hostile argument adds none.
            assert_eq!(safe(&output.stderr).lines().count(), 2);
        }
    }
    let output = fixture.command().arg("界".repeat(2000)).output().unwrap();
    failure(&output, &["Unknown command", "…", "vade --help"]);
    assert!(output.stderr.len() < 600, "unbounded argument diagnostic");
    assert!(!fixture.0.join("home").exists());
}
#[test]
fn missing_config_and_setup_menu_escape_hostile_paths() {
    let fixture = Fixture::new("missing");
    let path = fixture.0.join("config\x1b[2J\r\n\u{202e}.conf");
    for args in [&["status"][..], &[][..]] {
        let output = fixture.command().args(args).env("VADE_CONFIG", &path).output().unwrap();
        if args.is_empty() { assert!(output.status.success()); }
        else { assert_eq!(output.status.code(), Some(1)); }
        let text = format!("{}{}", safe(&output.stdout), safe(&output.stderr));
        assert!(text.contains("Config is missing") && text.contains("INSTALL.md"));
        assert!(text.contains("\\u{1b}") && text.contains("\\r\\n") && text.contains("\\u{202e}"));
        assert!(text.contains(".conf"), "path must remain complete");
    }
    assert!(!path.exists());
}
#[test]
fn dry_run_and_saved_logs_escape_controls_without_rewriting_evidence() {
    let fixture = Fixture::new("preview\x1b[2J\u{202e}");
    let config = fixture.config();
    let output = fixture.command().args(["up", "--fp4-local", "--dry-run"]).env("VADE_CONFIG", &config).output().unwrap();
    assert!(output.status.success(), "{}", safe(&output.stderr));
    assert!(safe(&output.stdout).contains("\\u{1b}"));
    assert!(!fixture.0.join("state").exists(), "preview must not create state");
    let logs = fixture.0.join("state/logs");fs::create_dir_all(&logs).unwrap();
    let path = logs.join("hostile\x1b[2J.log");
    let bytes = "Café 界\n\tindented\n\x1b]52;c;payload\x07\rforged\u{009b}2J\u{202e}hidden\n";
    fs::write(&path, bytes).unwrap();
    let output = fixture.command().arg("logs").env("VADE_CONFIG", &config).output().unwrap();
    assert!(output.status.success(), "{}", safe(&output.stderr));
    let text = safe(&output.stdout);
    assert!(text.contains("Café 界\n\tindented\n") && text.contains("\\u{1b}"));
    assert_eq!(fs::read(&path).unwrap(), bytes.as_bytes());
    let home = fixture.0.join("home");fs::create_dir(&home).unwrap();
    fs::write(home.join("setup.conf"), "").unwrap();fs::write(home.join("last-server.log"), bytes).unwrap();
    let output = fixture.command().arg("logs").env_remove("VADE_CONFIG").output().unwrap();
    assert!(output.status.success(), "{}", safe(&output.stderr));
    assert!(safe(&output.stdout).contains("Café 界"));
    assert_eq!(fs::read(home.join("last-server.log")).unwrap(), bytes.as_bytes());
}
#[test]
fn credential_failures_name_the_cause_and_action_before_any_launch_effect() {
    let f = Fixture::new("auth-errors");
    let directory = f.0.join("directory");fs::create_dir(&directory).unwrap();
    let invalid = f.0.join("invalid-utf8");fs::write(&invalid, b"\xffPRIVATE-KEY-CONTENTS").unwrap();
    let short = f.0.join("short");fs::write(&short, b"PRIVATEKEY").unwrap();
    for (path, cause, action) in [
        (f.0.join("missing-key"), "path not found", "Check the path"),
        (directory, "wrong path type", "Select the expected file"),
        (invalid, "invalid UTF-8", "Replace its contents"),
        (short, "at least 16 bytes", "Replace its contents"),
    ] {
        for arguments in [&["chat"][..], &["doctor"][..], &["run", "qwen3-8b"][..]] {
            let mut command = f.command();
            command.args(arguments).env("VADE_API_KEY_FILE", &path);
            if arguments[0] == "run" {
                command.env_remove("VADE_CONFIG").env("XDG_CONFIG_HOME", f.0.join("xdg"));
            }
            let output = command.output().unwrap();
            failure(&output, &["--api-key-file", cause, action]);
            assert!(!safe(&output.stderr).contains("PRIVATE"), "credential contents must never be printed");
            assert!(!f.0.join("home").exists(), "credential validation must precede lifecycle side effects");
        }
    }
    for (args, expected) in [
        (vec!["chat", "--api-key-file"], "Give --api-key-file a readable key file path"),
        (vec!["chat", "--api-key-file", "a", "--api-key-file", "b"], "Use --api-key-file only once"),
    ] { failure(&f.command().args(args).output().unwrap(), &[expected]); }
}

#[test]
fn launch_preview_summarizes_requested_settings_before_any_connection() {
    let f = Fixture::new("launch-plan");
    let config = f.config();
    let mut source = fs::read_to_string(&config).unwrap();
    source.push_str("fabric.peer = 192.0.2.1\n[machine peer]\nssh = fixture-peer-no-network\nmodel_path = /models/fixture\nfabric.spark = 192.0.2.2\n");
    fs::write(&config, source).unwrap();
    for (flags, rdma, expected) in [
        (vec![], "1", vec!["  Machines          spark\n", "  Context           32768 tokens\n  Output limit      8192 tokens\n", "  Drafting request  engine default\n  Concurrent chats  engine-admitted\n", "local (one Spark; no fabric)"]),
        (vec!["--chats", "4"], "1", vec!["  Context           32768 tokens\n  Output limit      8192 tokens\n", "  Concurrent chats  4\n"]),
        (vec!["--context", "262144", "--output", "16384"], "1", vec!["  Context           262144 tokens\n  Output limit      16384 tokens\n", "  Concurrent chats  engine-admitted\n"]),
        (vec!["--chats", "4", "--context", "8192"], "1", vec!["  Context           8192 tokens\n  Output limit      8192 tokens\n"]),
        (vec!["--mtp", "0", "--context", "4096", "--output", "2048"], "1", vec!["  Context           4096 tokens\n  Output limit      2048 tokens\n", "  Drafting request  MTP 0\n"]),
        (vec!["--sparks", "2", "--chats", "4"], "0", vec!["  Machines          spark, peer\n", "  Transport         TCP\n", "  Concurrent chats  4\n"]),
        (vec!["--sparks", "2"], "1", vec!["  Machines          spark, peer\n", "  Transport         RDMA\n"]),
    ] {
        let output = f.command().args(["up", "--dry-run"]).args(flags)
            .env("VADE_CONFIG", &config).env("SEQPAR_RDMA", rdma).env_remove("MTP").output().unwrap();
        assert!(output.status.success(), "{}", safe(&output.stderr));
        let stdout = safe(&output.stdout);
        for text in expected { assert!(stdout.contains(text), "missing {text:?} in {stdout}"); }
        assert!(stdout.find("Launch plan:").unwrap() < stdout.find("$ ").unwrap());
        assert!(stdout.contains("  Endpoint          http://localhost:18080/v1\n"));
        assert!(!f.0.join("state").exists() && !f.0.join("home").exists());
    }
}

#[test]
fn setup_previews_and_saved_logs_exit_cleanly_with_closed_output_consumers() {
    use std::{os::{fd::OwnedFd, unix::net::UnixStream}, process::Stdio};
    let fixture = Fixture::new("closed-output");
    let config = fixture.config();
    let logs = fixture.0.join("state/logs"); fs::create_dir_all(&logs).unwrap();
    let saved = logs.join("server.log"); fs::write(&saved, "retained logs\n").unwrap();
    for args in [&[][..], &["up", "--fp4-local", "--dry-run"], &["logs"], &["status"], &["run", "qwen3-8b", "--dry-run"]] {
        for closed_stderr in [false, true] {
            let (writer, reader) = UnixStream::pair().unwrap(); drop(reader);
            let fd: OwnedFd = writer.into();
            let mut cmd = fixture.command(); cmd.args(args).stdout(Stdio::from(fd));
            if args.first() == Some(&"run") {
                cmd.env_remove("VADE_CONFIG").env("XDG_CONFIG_HOME", fixture.0.join("xdg"));
            } else if !args.is_empty() { cmd.env("VADE_CONFIG", &config); }
            if closed_stderr {
                let (writer, reader) = UnixStream::pair().unwrap(); drop(reader);
                let fd: OwnedFd = writer.into(); cmd.stderr(Stdio::from(fd));
            }
            let output = cmd.output().unwrap();
            assert!(output.status.success() && output.stderr.is_empty(), "{args:?}: {output:?}");
        }
    }
    assert_eq!(fs::read_to_string(saved).unwrap(), "retained logs\n");
    assert!(!fixture.0.join("state/session.conf").exists());
}

#[test]
fn unpinned_controller_configs_warn_and_allow_recovery_commands() {
    for revision in ["", "main"] {
        let fixture = Fixture::new("old-revision");
        let config = fixture.config();
        let text = fs::read_to_string(&config).unwrap().replace("model_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", &if revision.is_empty() { String::new() } else { format!("model_revision = {revision}\n") });
        fs::write(&config, text).unwrap();
        for command in ["down", "logs", "status"] {
            let output = fixture.command().arg(command).env("VADE_CONFIG", &config).output().unwrap();
            assert!(output.status.success(), "{command}: {}", safe(&output.stderr));
            assert!(safe(&output.stderr).contains("Warning: model_revision"));
        }
        for command in ["install", "up"] {
            let output = fixture.command().args([command, "--dry-run"]).env("VADE_CONFIG", &config).output().unwrap();
            assert_eq!(output.status.code(), Some(1));
            assert!(safe(&output.stderr).contains("verified immutable"));
        }
    }
}
