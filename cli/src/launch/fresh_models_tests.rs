//! Test fresh install words with fake hardware and a private child environment.
use super::*;
use std::process::{Command as Process, Stdio};

const BASE: &str = "VADE_TEST_FRESH_MODELS_BASE";

#[test]
fn install_child() {
    let Some(base) = std::env::var_os(BASE).map(PathBuf::from) else { return; };
    let home = base.join("home");
    let root = home.join(".vade");
    let mut launch = Launcher {
        runner: MockRunner::new(&root), detector: MockDetector { identity: identity(), calls: 0 },
        root, home,
    };
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || launch.execute(Command::Install, false));
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    output::text(format_args!("{printed}")).unwrap();
    if let Err(error) = result {
        output::stderr_line(format_args!("{error}"));
        std::process::exit(1);
    }
}

fn install(f: &Fixture, models: Option<&Path>) -> (Option<i32>, String, String) {
    let mut command = Process::new(std::env::current_exe().unwrap());
    command.args(["--exact", "launch::tests::fresh_models_tests::install_child", "--nocapture"])
        .env_clear().env("PATH", "/usr/bin:/bin").env("HOME", &f.home).env("NO_COLOR", "1")
        .env(BASE, &f.base).stdin(Stdio::null());
    if let Some(models) = models { command.env("VADE_MODELS", models); }
    let result = command.output().unwrap();
    (result.status.code(), String::from_utf8(result.stdout).unwrap(), String::from_utf8(result.stderr).unwrap())
}

#[test]
fn default_folder_absent_prints_one_plain_line_and_succeeds() {
    let f = Fixture::new();
    f.installed();
    fs::rename(f.home.join("models"), f.base.join("before-pull")).unwrap();
    let (code, stdout, stderr) = install(&f, None);
    assert_eq!(code, Some(0), "{stderr}\n{stdout}");
    assert_eq!(stdout.lines().filter(|line| *line == detect::NO_MODELS).count(), 1, "{stdout}");
    assert!(!stdout.contains("Models:") && !stdout.contains("is unavailable") && !stdout.contains("Model:"), "{stdout}");
    assert!(stderr.is_empty(), "{stderr}");
    assert!(!f.home.join("models").exists());
    assert!(!f.root.join("config").exists());
}

#[test]
fn user_folder_absent_keeps_the_error_and_exit_code() {
    let f = Fixture::new();
    f.installed();
    let missing = f.home.join("models");
    fs::rename(&missing, f.base.join("before-pull")).unwrap();
    // An explicit default path is still a user choice.
    let (code, stdout, stderr) = install(&f, Some(&missing));
    assert_eq!(code, Some(1), "{stderr}\n{stdout}");
    assert!(stderr.contains(&format!("Model folder {} is unavailable.", text::quoted(&missing.to_string_lossy()))), "{stderr}");
    assert!(!stdout.contains(detect::NO_MODELS) && !stderr.contains(detect::NO_MODELS), "{stdout}\n{stderr}");
    assert!(!f.root.join("config").exists());
}

#[test]
fn present_folder_has_no_first_run_line() {
    let f = Fixture::new();
    f.installed();
    let (code, stdout, stderr) = install(&f, None);
    assert_eq!(code, Some(0), "{stderr}\n{stdout}");
    assert!(stdout.contains(&format!("Models: {}", text::quoted(&f.model.to_string_lossy()))), "{stdout}");
    assert!(!stdout.contains(detect::NO_MODELS), "{stdout}");
    assert!(stderr.is_empty(), "{stderr}");
    assert!(f.root.join("config").is_file());
}

#[test]
fn empty_folder_has_no_first_run_line() {
    let f = Fixture::new();
    f.installed();
    fs::rename(&f.model, f.base.join("before-pull")).unwrap();
    let (code, stdout, stderr) = install(&f, None);
    assert_eq!(code, Some(0), "{stderr}\n{stdout}");
    assert!(stdout.contains("No complete model was found"), "{stdout}");
    assert!(!stdout.contains(detect::NO_MODELS), "{stdout}");
    assert!(stderr.is_empty(), "{stderr}");
}
