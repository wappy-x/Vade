//! A fixture HOME for every first-party binary a test starts, so a developer's
//! ~/.vade (instruction file, saved defaults, chat state) never reaches a test.
// Each test file uses a different subset of these helpers.
#![allow(dead_code)]
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

/// One empty folder per test process under Cargo's integration-test scratch
/// folder. A test that needs a home file sets its own HOME after this.
pub fn path() -> &'static PathBuf {
    static HOME: OnceLock<PathBuf> = OnceLock::new();
    HOME.get_or_init(|| {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    })
}
pub fn command(program: &str) -> Command {
    let mut command = Command::new(program);
    command.env("HOME", path());
    command
}
pub fn harness() -> Command {
    command(env!("CARGO_BIN_EXE_vade-harness"))
}
