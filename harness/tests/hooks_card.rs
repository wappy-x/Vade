//! Offline chat startup and card isolation for hooks.
#[path = "support/home.rs"]
mod home;

#[test]
fn card_prompt_output_ignores_hooks() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("hooks-card-{}", std::process::id()));
    std::fs::create_dir_all(root.join(".vade")).unwrap();
    std::fs::write(root.join("hook.sh"), "touch hook-ran; printf 'HOOK MUST NOT RUN'\n").unwrap();
    let config = r#"{"hooks":[{"point":"before-tool","command":"/bin/sh hook.sh"},{"point":"after-tool","command":"/bin/sh hook.sh"},{"point":"on-stop","command":"/bin/sh hook.sh"}]}"#;
    std::fs::write(root.join(".vade/hooks.conf"), config).unwrap();
    let state = vade_harness::tools::StateLocation::home(root.join("home")).unwrap().path(&root.canonicalize().unwrap());
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("hooks.conf"), config).unwrap();
    let output = home::harness().current_dir(&root).env("VADE_HOME", root.join("home"))
        .args(["quality-card", "--print-prompts"]).output().unwrap();
    assert!(output.status.success());
    assert!(!output.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&output.stdout).contains("HOOK MUST NOT RUN"));
    assert!(!root.join("hook-ran").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[cfg(unix)]
fn malformed_hooks_are_reported_once_and_user_help_lists_templates() {
    use std::{io::Write, os::unix::fs::DirBuilderExt, process::Stdio};
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("hooks-startup-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::DirBuilder::new().mode(0o700).create(root.join(".vade")).unwrap();
    std::fs::write(root.join(".vade/hooks.conf"), "not JSON").unwrap();
    std::fs::set_permissions(root.join(".vade/hooks.conf"), std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    std::fs::create_dir(root.join(".vade/commands")).unwrap();
    std::fs::write(root.join(".vade/commands/review.md"), "description: Review local files\nReview $ARGUMENTS").unwrap();
    let mut child = home::harness().env("VADE_HOME", root.join("home"))
        .args(["agent", "--plain", "--no-instructions", "--state-in-project", "--endpoint", "http://unused.invalid", "--model", "fixture", "--workspace"])
        .arg(&root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    // Commands are local. There is no prompt and no HTTP request.
    child.stdin.take().unwrap().write_all(b"/help\n/clear\n/help review\n/unknown-name\n/quit\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "{text}");
    assert_eq!(text.matches("hooks.conf ignored for this chat").count(), 1, "{text}");
    assert!(text.contains("Review local files"), "{text}");
    assert!(text.contains("Unknown command /unknown-name"), "{text}");
    assert!(text.contains("Available commands:") && text.contains("/review"), "{text}");
    std::fs::remove_dir_all(root).unwrap();
}
