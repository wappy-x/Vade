//! The first run on a computer with nothing set up: every everyday command
//! says what is missing and what to run next, and none of them tries a
//! network or an engine. The home is a new private folder. A Mac drives Sparks
//! through a controller config; a Spark (Linux aarch64) uses the local launcher;
//! other Linux computers can only preview.
use std::process::Command;

struct Home(std::path::PathBuf);
impl Home {
    fn new() -> Self {
        // macOS puts the temporary folder behind a symlink, which the CLI refuses.
        let path = std::env::temp_dir().canonicalize().unwrap().join(format!("vade-first-run-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
        Self(path)
    }
    fn run(&self, args: &[&str]) -> (Option<i32>, String, String) {
        let output = Command::new(env!("CARGO_BIN_EXE_vade")).args(args).env_clear()
            .env("PATH", "/usr/bin:/bin").env("HOME", &self.0).env("NO_COLOR", "1").current_dir(&self.0)
            .stdin(std::process::Stdio::null()).output().unwrap();
        (output.status.code(), String::from_utf8(output.stdout).unwrap(), String::from_utf8(output.stderr).unwrap())
    }
}
impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The cause (no setup, no controller config) and both next steps.
#[cfg(not(target_os = "linux"))]
fn names_cause_and_next_step(text: &str) {
    assert!(text.contains("Vade is not set up yet: there is no local setup and no controller config at "), "{text}");
    assert!(text.contains("Run vade launch on a Spark") && text.contains("cli/examples/fp4-local.conf") && text.contains("VADE_CONFIG") && text.contains("INSTALL.md"), "{text}");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn first_run_commands_name_what_is_missing_and_what_to_run() {
    let home = Home::new();
    // Bare vade explains once and points at the preview, the check and help.
    let (code, stdout, stderr) = home.run(&[]);
    assert_eq!(code, Some(0), "{stderr}");
    names_cause_and_next_step(&stdout);
    assert!(stdout.contains("Preview: vade launch --dry-run") && stdout.contains("Check: vade doctor") && stdout.contains("Help: vade --help"), "{stdout}");
    // Doctor fails the config row and gives the next step on its own line.
    let (code, stdout, _) = home.run(&["doctor"]);
    assert_eq!(code, Some(1));
    let row = stdout.lines().position(|l| l.starts_with("FAIL  config ")).unwrap_or_else(|| panic!("{stdout}"));
    names_cause_and_next_step(&stdout);
    assert!(stdout.lines().nth(row + 1).is_some_and(|l| l.trim_start().starts_with("next: Run vade launch on a Spark")), "{stdout}");
    // Everyday commands stop with the cause and a hint, before any engine call.
    for command in ["chat", "status", "down", "logs", "sessions"] {
        let (code, stdout, stderr) = home.run(&[command]);
        assert_eq!(code, Some(1), "{command}: {stderr}");
        assert!(stdout.is_empty(), "{command}: {stdout}");
        assert!(stderr.starts_with("error: Vade is not set up yet") && stderr.contains("\n hint: Run vade launch on a Spark"), "{command}: {stderr}");
        names_cause_and_next_step(&stderr);
    }
    // Nothing was written into the new home.
    assert_eq!(std::fs::read_dir(&home.0).unwrap().count(), 0);
}

/// On a Spark the same commands use the local launcher. The lines are the
/// output of the Linux aarch64 build on a Spark with a new empty home.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
#[test]
fn first_run_on_a_spark_points_at_vade_launch() {
    let home = Home::new();
    for command in [&[][..], &["status"]] {
        let (code, stdout, stderr) = home.run(command);
        assert_eq!(code, Some(0), "{command:?}: {stderr}");
        assert!(stdout.starts_with("○ Vade is stopped.\n  Start: ") && stdout.ends_with("vade launch\n"), "{command:?}: {stdout}");
    }
    let (code, stdout, stderr) = home.run(&["doctor"]);
    assert_eq!(code, Some(1), "{stderr}");
    let row = stdout.lines().position(|l| l.starts_with("FAIL  installed binaries    Vade is not installed yet.")).unwrap_or_else(|| panic!("{stdout}"));
    assert_eq!(stdout.lines().nth(row + 1).map(str::trim), Some("next: Run vade launch to install and start Vade."), "{stdout}");
    assert!(stderr.contains("\n hint: Run the next: step under each FAIL row, then run vade doctor again."), "{stderr}");
    for (command, expected) in [("down", "Vade is already stopped.\n"), ("logs", "No engine log yet.\n")] {
        assert_eq!(home.run(&[command]), (Some(0), expected.to_owned(), String::new()), "{command}");
    }
    for (command, expected) in [
        ("chat", "error: Vade is not set up on this Spark yet.\n hint: Run vade launch to install it, start it and open chat.\n"),
        ("sessions", "error: The engine and chat are not installed yet.\n hint: Run vade launch to install them.\n"),
    ] {
        assert_eq!(home.run(&[command]), (Some(1), String::new(), expected.to_owned()), "{command}");
    }
    // Only the lock file of the Vade home was written.
    let written: Vec<_> = std::fs::read_dir(home.0.join(".vade")).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(written, [".lock"]);
    assert_eq!(std::fs::read_dir(&home.0).unwrap().count(), 1);
}

/// Other Linux computers cannot run the engine: each command names the Spark
/// requirement and the two other paths, a preview or a controller config.
#[cfg(all(target_os = "linux", not(target_arch = "aarch64")))]
#[test]
fn first_run_on_other_linux_names_the_spark_requirement() {
    let home = Home::new();
    for command in [&[][..], &["doctor"], &["chat"], &["status"], &["down"], &["logs"], &["sessions"]] {
        let (code, stdout, stderr) = home.run(command);
        assert_eq!(code, Some(1), "{command:?}: {stdout}");
        assert!(stderr.contains("This command runs only on a DGX Spark (Linux aarch64)") && stderr.contains("--dry-run") && stderr.contains("VADE_CONFIG"), "{command:?}: {stderr}");
    }
}
