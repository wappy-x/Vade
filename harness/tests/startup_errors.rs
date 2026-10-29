//! Local initialization errors identify the setting and preserve saved sessions.
#[path = "support/home.rs"]
mod home;
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{fs, net::TcpListener, path::{Path, PathBuf}, process::Command};

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("vade-startup-errors-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn command(mode: &str, workspace: &Path, peer: &TcpListener) -> Command {
    let mut command = home::harness();
    command.args([mode, "--state-in-project", "--endpoint", &format!("http://{}/v1", peer.local_addr().unwrap()), "--model", "mock", "--prompt", "fixture", "--workspace"])
        .arg(workspace).env_remove("VADE_API_KEY_FILE").env_remove("VADE_HARNESS_EFFORT").env_remove("VADE_HARNESS_THINKING");
    command
}
fn refusal(command: &mut Command, peer: &TcpListener, option: &str, remedy: &str) {
    let output = cli::Cli::start(command).finish();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert_eq!(error.lines().count(), 1, "{error}");
    assert!(error.starts_with(&format!("vade: {option}:")), "{error}");
    assert!(error.contains(remedy), "{error}");
    assert!(!error.contains("PRIVATE") && !error.contains('\x1b'), "{error}");
    assert_eq!(peer.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}

#[test]
fn startup_errors_name_workspace_or_session_and_preserve_original_files() {
    let root = Root::new();
    let peer = TcpListener::bind("127.0.0.1:0").unwrap();
    peer.set_nonblocking(true).unwrap();
    let file = root.0.join("not-a-directory"); fs::write(&file, b"keep").unwrap();
    let mut cases = 0;
    for workspace in [root.0.join("missing-雪\x1b[2J"), file.clone()] {
        for mode in ["direct", "agent"] {
            refusal(&mut command(mode, &workspace, &peer), &peer, "--workspace", "Choose an existing readable directory");
            cases += 1;
        }
    }
    assert_eq!(fs::read(&file).unwrap(), b"keep");
    let workspace = root.0.join("workspace");
    let sessions = workspace.join(".vade/sessions"); fs::create_dir_all(&sessions).unwrap();
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        for dir in [workspace.join(".vade"), sessions.clone()] { fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).unwrap(); }
    }
    for name in ["broken.journal", "broken.state"] {
        let path = sessions.join(name); fs::write(&path, b"PRIVATE synthetic corrupt state").unwrap();
        #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap(); }
    }
    for (mode, option, name, remedy) in [
        ("direct", "--session", "broken", "Keep the saved metadata"),
        ("agent", "--resume", "missing", "Check --workspace and the saved --resume NAME"),
        ("agent", "--resume", "broken", "Check --workspace and the saved --resume NAME"),
        ("agent", "--session", "broken", "Use --resume NAME for an existing session"),
    ] {
        refusal(command(mode, &workspace, &peer).args([option, name]), &peer, option, remedy);
        cases += 1;
    }
    for name in ["broken.journal", "broken.state"] {
        assert_eq!(fs::read(sessions.join(name)).unwrap(), b"PRIVATE synthetic corrupt state");
    }
    assert!(!sessions.join("missing.journal").exists());
    let blocked = root.0.join("blocked"); fs::create_dir(&blocked).unwrap();
    fs::write(blocked.join(".vade"), b"keep existing file").unwrap();
    for (mode, remedy) in [("direct", "Keep the saved metadata"), ("agent", "Use --resume NAME for an existing session")] {
        refusal(command(mode, &blocked, &peer).args(["--session", "new"]), &peer, "--session", remedy);
        cases += 1;
    }
    assert_eq!(fs::read(blocked.join(".vade")).unwrap(), b"keep existing file");
    assert_eq!(cases, 10);
}
