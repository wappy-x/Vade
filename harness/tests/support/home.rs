//! A fixture HOME for every first-party binary a test starts, so a developer's
//! ~/.vade (instruction file, saved defaults, chat state, MCP servers) never
//! reaches a test.
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
    // VADE_HOME can name the developer's own MCP list, so MCP stays off.
    command.env("HOME", path()).env("VADE_MCP", "off");
    command
}
pub fn harness() -> Command {
    let mut command = command(env!("CARGO_BIN_EXE_vade-harness"));
    // Conformance requests must not inherit notes or memory tools from chat.
    command.env("VADE_MEMORY", "0");
    command
}
