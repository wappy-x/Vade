//! Exercise receipt placement through a relocated CLI with no build tree or
//! output overrides. An invalid URL fails before connecting but still produces
//! the benchmark's normal failure receipt, so this test needs no running engine.
use std::{fs, path::{Path, PathBuf}, process::Command, sync::atomic::{AtomicUsize, Ordering}, time::{SystemTime, UNIX_EPOCH}};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "vade-bench-paths-{}-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("bundle")).unwrap();
        fs::create_dir(root.join("user")).unwrap();
        fs::create_dir(root.join("working")).unwrap();
        fs::copy(env!("CARGO_BIN_EXE_vade"), root.join("bundle/vade")).unwrap();
        fs::write(root.join("prompts"), "hello\n").unwrap();
        Self(root)
    }

    fn cli(&self) -> Command {
        // Set the child's umask without changing the test process or adding unsafe.
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "umask 022; exec \"$@\"", "bench-path-test"])
            .arg(self.0.join("bundle/vade"));
        command.current_dir(self.0.join("working"))
            .env_clear().env("HOME", self.0.join("user"))
            .env("VADE_HOME", self.0.join("install"));
        command
    }

    fn command(&self) -> Command {
        let mut command = self.cli();
        command.args(["bench", "--url", "invalid://offline", "--prompts"])
            .arg(self.0.join("prompts"))
            .args(["--concurrency", "1", "--max-tokens", "1", "--timeout", "0.01"]);
        command
    }

    fn receipt(&self, command: &mut Command, directory: &Path) -> PathBuf {
        let result = command.output().unwrap();
        assert_eq!(result.status.code(), Some(1), "{result:?}");
        assert!(result.stderr.is_empty(), "{result:?}");
        let stdout = String::from_utf8(result.stdout).unwrap();
        let path = PathBuf::from(stdout.lines().find_map(|line| line.strip_prefix("JSON: ")).unwrap());
        let path = if path.is_absolute() { path } else { self.0.join("working").join(path) };
        let path = path.canonicalize().unwrap();
        assert_eq!(path.parent().unwrap(), directory, "{stdout}");
        let receipt = fs::read_to_string(&path).unwrap();
        assert!(receipt.contains("invalid://offline"), "{receipt}");
        assert!(path.with_extension("txt").is_file());
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

#[test]
fn relocated_benchmark_writes_to_cwd_without_creating_installation_roots() {
    let fixture = Fixture::new();
    let working = fixture.0.join("working");
    let expected = working.join("vade-bench.json");
    assert_eq!(fixture.receipt(&mut fixture.command(), &working), expected);
    assert!(!fixture.0.join("install").exists());
    assert_eq!(fixture.receipt(fixture.command().env_remove("VADE_HOME"), &working), expected);
    assert!(!fixture.0.join("user/.vade").exists());
    assert_eq!(fixture.receipt(fixture.command().env("VADE_HOME", "invalid-relative")
        .env("VADE_ROOT", fixture.0.join("legacy"))
        .env("VADE_ORCH_ROOT", fixture.0.join("orchestration")), &working), expected);
    for path in ["install", "user/.vade", "legacy", "orchestration", "working/.local", "working/invalid-relative"] {
        assert!(!fixture.0.join(path).exists(), "unexpected directory: {path}");
    }
}

#[test]
fn relocated_benchmark_preserves_explicit_output_paths() {
    let fixture = Fixture::new();
    for flag in ["--out", "--output"] {
        let explicit = fixture.0.join(format!("explicit/{flag}.json"));
        let actual = fixture.receipt(fixture.command().env("VADE_HOME", "invalid-relative")
            .arg(flag).arg(&explicit), explicit.parent().unwrap());
        assert_eq!(actual, explicit);
    }
    assert!(!fixture.0.join("working/vade-bench.json").exists());
    assert!(!fixture.0.join("install").exists());
    assert!(!fixture.0.join("user/.vade").exists());
}

#[test]
fn invalid_receipt_settings_fail_without_falling_back_to_the_build_tree() {
    let fixture = Fixture::new();
    for flag in ["--out", "--output"] {
        for values in [vec![flag], vec![flag, ""]] {
            let result = fixture.command().args(values).output().unwrap();
            assert_eq!(result.status.code(), Some(1));
            assert!(String::from_utf8(result.stderr).unwrap().contains("missing value for --output"));
        }
    }
    assert!(!fixture.0.join("working/vade-bench.json").exists());
    assert!(!fixture.0.join("install").exists());
}

#[test]
fn benchmark_help_names_the_current_directory_receipt() {
    let fixture = Fixture::new();
    for args in [["bench", "--help"], ["help", "bench"]] {
        let result = fixture.cli().args(args).output().unwrap();
        assert!(result.status.success(), "{result:?}");
        let help = String::from_utf8(result.stdout).unwrap();
        assert!(help.contains("./vade-bench.json"), "{help}");
        assert!(help.contains("--output"), "{help}");
        assert!(!help.contains(".local/runtime/orchestration"), "{help}");
    }
    assert!(!fixture.0.join("install").exists());
    assert!(!fixture.0.join("working/vade-bench.json").exists());
}
