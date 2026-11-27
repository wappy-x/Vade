//! `harness/scripts/route-a-check.sh` against command mocks: a stand-in
//! `vade`, `cargo`, `curl`, `uname`, `nvidia-smi` and `rustup` in a temporary
//! checkout and home. No engine, GPU, network or real build; each scenario
//! checks which instance every step talks to and what cleanup stops.
#![cfg(unix)]
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/route-a-check.sh");

// Records its arguments and the selectors it inherited, then plays the local
// launcher: the engine record lives in VADE_HOME, as the real CLI keeps it.
const VADE: &str = r#"#!/bin/sh
printf 'vade %s|home=%s|api=%s|config=%s\n' "$*" "${VADE_HOME-}" "${VADE_API_URL-unset}" "${VADE_CONFIG-unset}" >> "$MOCK/calls"
case $1 in
detect) printf 'Machine: mock\n%s\n' "$MOCK_DETECT" ;;
launch)
    [ "${MOCK_LAUNCH:-ok}" = norecord ] && { echo 'The default controller has an ownership record.'; exit 1; }
    mkdir -p "$VADE_HOME/state"
    printf 'format=vade-owner-v1\nmodel_path=%s\nport=18123\n' "$MOCK_RECORD_MODEL" > "$VADE_HOME/config"
    printf 'version=1\nphase=running\nport=18123\nmodel_path=%s\n' "$MOCK_RECORD_MODEL" > "$VADE_HOME/state/run.conf"
    [ "${MOCK_LAUNCH:-ok}" = fail ] && exit 1
    echo 'Launch ready in 1 s.' ;;
verify) echo 'IDENTICAL: 20/20 runs, 1 machine, 9 tokens' ;;
down) [ "${MOCK_DOWN:-ok}" = fail ] && exit 1; rm -f "$VADE_HOME/state/run.conf" ;;
esac
"#;
const CARGO: &str = r#"#!/bin/sh
case $* in *--version*) exit 0 ;; esac
printf 'cargo %s|build_target=%s\n' "$*" "${CARGO_BUILD_TARGET-unset}" >> "$MOCK/calls"
dir= previous=
for arg; do [ "$previous" = --target-dir ] && dir=$arg; previous=$arg; done
printf '{"reason":"compiler-artifact","executable":"%s"}\n' "${MOCK_CARGO_EXE:-$dir/release/vade}"
"#;
const CURL: &str = r#"#!/bin/sh
for arg; do url=$arg; done
printf 'curl %s\n' "$url" >> "$MOCK/calls"
case $url in
*/models) echo '{"data":[{"id": "qwen3.8-flash-next-fp4-local"}]}' ;;
*) [ -z "${MOCK_SLEEP:-}" ] || sleep "$MOCK_SLEEP"; echo '{"choices":[{"message":{"content": "Hello there, nice to meet."}}]}' ;;
esac
"#;
const TOOLS: [(&str, &str); 7] = [
    ("uname", "#!/bin/sh\n[ \"$1\" = -m ] && echo aarch64 || echo Linux\n"),
    ("nvidia-smi", "#!/bin/sh\ncase $* in *query-gpu*) echo 'NVIDIA GB10' ;; esac\n"),
    ("rustup", "#!/bin/sh\ncase $* in *component*) echo rust-src ;; *target*) echo nvptx64-nvidia-cuda ;; esac\n"),
    ("cc", "#!/bin/sh\n"),
    ("git", "#!/bin/sh\n"),
    ("sha256sum", "#!/bin/sh\n"),
    ("curl", CURL),
];

struct Mock(PathBuf);
impl Mock {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "vade-route-a-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let dir = fs::canonicalize(dir).unwrap();
        for sub in ["co/harness/scripts", "co/cli/target/release", "bin", "home", "tmp", "models/ck"] {
            fs::create_dir_all(dir.join(sub)).unwrap();
        }
        fs::copy(SCRIPT, dir.join("co/harness/scripts/route-a-check.sh")).unwrap();
        fs::write(dir.join("co/cli/Cargo.toml"), "").unwrap();
        fs::write(dir.join("models/ck/config.json"), "{}").unwrap();
        executable(&dir.join("co/cli/target/release/vade"), VADE);
        executable(&dir.join("bin/cargo"), CARGO);
        for (name, body) in TOOLS {
            executable(&dir.join("bin").join(name), body);
        }
        fs::write(dir.join("calls"), "").unwrap();
        Self(dir)
    }
    fn model(&self) -> PathBuf {
        self.0.join("models/ck")
    }
    /// The script under these extra variables; defaults name the mock model
    /// the way `vade detect` and the engine record spell it.
    fn command(&self, extra: &[(&str, String)]) -> Command {
        let mut command = Command::new("sh");
        command
            .arg(self.0.join("co/harness/scripts/route-a-check.sh"))
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", self.0.join("bin").display()))
            .env("HOME", self.0.join("home"))
            .env("TMPDIR", self.0.join("tmp"))
            .env("MOCK", &self.0)
            .env("MOCK_DETECT", detect_line(&self.model()))
            .env("MOCK_RECORD_MODEL", record_value(&self.model()))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in extra {
            command.env(key, value);
        }
        command
    }
    fn run(&self, extra: &[(&str, String)]) -> (i32, String) {
        let output = self.command(extra).output().unwrap();
        let text = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
        (output.status.code().unwrap_or(-1), text)
    }
    fn calls(&self) -> String {
        fs::read_to_string(self.0.join("calls")).unwrap()
    }
    fn vade_calls(&self, command: &str) -> Vec<String> {
        self.calls().lines().filter(|line| line.starts_with(&format!("vade {command}"))).map(str::to_owned).collect()
    }
    /// The private homes the script left behind in HOME.
    fn private_homes(&self) -> Vec<PathBuf> {
        fs::read_dir(self.0.join("home")).unwrap().map(|entry| entry.unwrap().path())
            .filter(|path| path.file_name().unwrap().to_string_lossy().starts_with(".vade-route-a.")).collect()
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
/// `vade detect`'s model line: `Models: ` and the path in Rust Debug quotes.
fn detect_line(path: &Path) -> String {
    format!("Models: {:?}", path.to_str().unwrap())
}
/// A key=value record value as `vade` writes it (backslashes doubled).
fn record_value(path: &Path) -> String {
    path.to_str().unwrap().replace('\\', "\\\\")
}
fn home_of(call: &str) -> &str {
    call.split('|').find_map(|part| part.strip_prefix("home=")).unwrap()
}
fn step<'a>(output: &'a str, name: &str) -> &'a str {
    output.lines().find(|line| line.starts_with('[') && line[6..].starts_with(name)).unwrap_or_else(|| panic!("no {name} step in:\n{output}"))
}

#[test]
fn passes_with_every_step_bound_to_its_own_private_engine() {
    let mock = Mock::new();
    // The owner's own installation is running; the check must not touch it.
    let owner = mock.0.join("home/.vade/state");
    fs::create_dir_all(&owner).unwrap();
    fs::write(owner.join("run.conf"), "version=1\nphase=running\nport=18080\n").unwrap();
    let (code, output) = mock.run(&[
        ("VADE_HOME", mock.0.join("home/.vade").display().to_string()),
        ("VADE_CONFIG", mock.0.join("controller.conf").display().to_string()),
        ("VADE_API_URL", "http://192.0.2.9:18080".into()),
        ("CARGO_BUILD_TARGET", "x86_64-unknown-linux-gnu".into()),
        ("CARGO_TARGET_DIR", mock.0.join("elsewhere").display().to_string()),
    ]);
    assert_eq!(code, 0, "{output}");
    assert!(output.contains("route-a-check: PASS (7/7 steps"), "{output}");
    let calls = mock.calls();
    let launch = mock.vade_calls("launch");
    assert_eq!(launch.len(), 1, "{calls}");
    let home = home_of(&launch[0]);
    assert!(home.contains("/home/.vade-route-a."), "{calls}");
    for line in calls.lines().filter(|line| line.starts_with("vade ")) {
        assert_eq!(home_of(line), home, "{calls}");
        assert!(line.ends_with("|api=unset|config=unset"), "{calls}");
    }
    assert!(calls.contains(&format!("--target-dir {}/co/cli/target --message-format=json-render-diagnostics|build_target=unset", mock.0.display())), "{calls}");
    let urls: Vec<_> = calls.lines().filter_map(|line| line.strip_prefix("curl ")).collect();
    assert_eq!(urls, ["http://127.0.0.1:18123/v1/models", "http://127.0.0.1:18123/v1/chat/completions"], "{calls}");
    assert!(mock.vade_calls("verify")[0].starts_with("vade verify --url http://127.0.0.1:18123 --receipt "), "{calls}");
    assert_eq!(mock.vade_calls("down").len(), 1, "{calls}");
    assert!(owner.join("run.conf").exists(), "the owner's engine record was touched");
    assert!(mock.private_homes().is_empty(), "a PASS removes its private home");
}

#[test]
fn a_refused_launch_without_a_record_stops_nothing() {
    let mock = Mock::new();
    let (code, output) = mock.run(&[("MOCK_LAUNCH", "norecord".into())]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "launch").contains("FAIL"), "{output}");
    assert!(step(&output, "down").contains("vade launch left no engine record"), "{output}");
    assert!(mock.vade_calls("down").is_empty(), "{}", mock.calls());
    assert!(mock.private_homes().is_empty(), "an empty private home is removed");
}

#[test]
fn a_failed_launch_with_a_record_is_stopped_and_its_home_kept() {
    let mock = Mock::new();
    let (code, output) = mock.run(&[("MOCK_LAUNCH", "fail".into())]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "down").contains("PASS"), "{output}");
    assert_eq!(mock.vade_calls("down").len(), 1, "{}", mock.calls());
    assert_eq!(mock.private_homes().len(), 1);
    assert!(output.contains("kept this run's Vade home for inspection"), "{output}");
}

#[test]
fn a_failed_stop_fails_the_run_once_and_keeps_the_home() {
    let mock = Mock::new();
    let (code, output) = mock.run(&[("MOCK_DOWN", "fail".into())]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "down").contains("FAIL"), "{output}");
    assert!(!output.contains("route-a-check: PASS"), "{output}");
    assert_eq!(mock.vade_calls("down").len(), 1, "{}", mock.calls());
    let homes = mock.private_homes();
    assert_eq!(homes.len(), 1);
    assert!(homes[0].join("state/run.conf").exists());
}

#[test]
fn escaped_model_paths_compare_as_the_real_path() {
    let mock = Mock::new();
    let model = mock.0.join("models/we\"ird\\ck");
    fs::create_dir_all(&model).unwrap();
    let (code, output) = mock.run(&[
        ("VADE_MODELS", mock.0.join("models").display().to_string()),
        ("MOCK_DETECT", detect_line(&model)),
        ("MOCK_RECORD_MODEL", record_value(&model)),
    ]);
    assert_eq!(code, 0, "{output}");
    assert!(step(&output, "model").ends_with(model.to_str().unwrap()), "{output}");
}

#[test]
fn a_model_outside_vade_models_fails_even_when_escaped() {
    let mock = Mock::new();
    let outside = mock.0.join("other\\ck");
    fs::create_dir_all(&outside).unwrap();
    let (code, output) = mock.run(&[
        ("VADE_MODELS", mock.0.join("models").display().to_string()),
        ("MOCK_DETECT", detect_line(&outside)),
    ]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "model").contains("not one under VADE_MODELS"), "{output}");
    assert!(mock.vade_calls("launch").is_empty(), "{}", mock.calls());
}

#[test]
fn unreadable_or_missing_model_paths_fail_explicitly() {
    let mock = Mock::new();
    let (code, output) = mock.run(&[("MOCK_DETECT", format!("Models: {:?}", "/models/a\nb"))]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "model").contains("cannot read the model path"), "{output}");
    let (code, output) = mock.run(&[("MOCK_DETECT", detect_line(&mock.0.join("gone")))]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "model").contains("cannot be opened"), "{output}");
    let (code, output) = mock.run(&[("VADE_MODELS", mock.0.join("gone").display().to_string())]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "model").contains("VADE_MODELS folder"), "{output}");
    assert!(mock.vade_calls("launch").is_empty(), "{}", mock.calls());
}

#[test]
fn an_engine_that_loaded_another_model_fails_and_is_stopped() {
    let mock = Mock::new();
    let (code, output) = mock.run(&[("MOCK_RECORD_MODEL", "/models/other".into())]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "launch").contains("the engine loaded /models/other"), "{output}");
    assert_eq!(mock.vade_calls("down").len(), 1, "{}", mock.calls());
}

#[test]
fn the_build_must_report_the_cli_the_script_runs() {
    let mock = Mock::new();
    let (code, output) = mock.run(&[("MOCK_CARGO_EXE", "/elsewhere/aarch64-unknown-linux-gnu/release/vade".into())]);
    assert_eq!(code, 1, "{output}");
    assert!(step(&output, "build").contains("Cargo did not report"), "{output}");
    assert!(mock.calls().lines().all(|line| !line.starts_with("vade ")), "{}", mock.calls());
}

#[test]
fn concurrent_runs_each_stop_only_their_own_engine() {
    let mock = Mock::new();
    let slow = [("MOCK_SLEEP", "1".to_string())];
    let children: Vec<_> = (0..2).map(|_| mock.command(&slow).spawn().unwrap()).collect();
    for child in children {
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stdout));
    }
    let mut launched: Vec<_> = mock.vade_calls("launch").iter().map(|line| home_of(line).to_owned()).collect();
    let mut stopped: Vec<_> = mock.vade_calls("down").iter().map(|line| home_of(line).to_owned()).collect();
    launched.sort();
    stopped.sort();
    assert_eq!(launched.len(), 2);
    assert_ne!(launched[0], launched[1]);
    assert_eq!(launched, stopped, "{}", mock.calls());
}

/// Whether `sh` can trap `signal` here: a test runner started in the
/// background or under nohup passes INT or HUP down as ignored, and POSIX
/// shells cannot trap a signal ignored on entry.
fn trappable(signal: &str) -> bool {
    let probe = format!("trap 'exit 7' {signal}; kill -{signal} $$; sleep 1; exit 0");
    Command::new("sh").args(["-c", &probe]).status().unwrap().code() == Some(7)
}

#[test]
fn termination_stops_the_engine_and_keeps_the_signal_status() {
    for (signal, status) in [("TERM", 143), ("INT", 130), ("HUP", 129)] {
        if signal != "TERM" && !trappable(signal) {
            eprintln!("SIG{signal} is ignored in this test process; skipped");
            continue;
        }
        let mock = Mock::new();
        let child = mock.command(&[("MOCK_SLEEP", "2".into())]).spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while !mock.calls().contains("/v1/chat/completions") {
            assert!(Instant::now() < deadline, "the chat request never started: {}", mock.calls());
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(Command::new("kill").arg(format!("-{signal}")).arg(child.id().to_string()).status().unwrap().success());
        let output = child.wait_with_output().unwrap();
        let text = String::from_utf8_lossy(&output.stdout);
        assert_eq!(output.status.code(), Some(status), "SIG{signal}: {text}");
        assert!(text.contains("route-a-check: stopped the engine this run started"), "SIG{signal}: {text}");
        assert_eq!(mock.vade_calls("down").len(), 1, "SIG{signal}: {}", mock.calls());
        let homes = mock.private_homes();
        assert_eq!(homes.len(), 1, "SIG{signal}: an interrupted run keeps its home");
        assert!(!homes[0].join("state/run.conf").exists(), "SIG{signal}");
    }
}
