//! Offline: a project's own allow list and check apply in chat only after
//! `vade trust`. No model request is sent; `/quit` ends each chat.
#![cfg(unix)]
#[path = "support/home.rs"]
mod home;
use std::io::Write;
use std::path::Path;
use std::process::{Output, Stdio};

fn text(output: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr))
}
fn chat(root: &Path, vade_home: &Path, input: &str) -> Output {
    chat_with(root, &["--vade-home"], vade_home, input)
}
/// `state` selects where chat state is; its last word takes the Vade home.
fn chat_with(root: &Path, state: &[&str], vade_home: &Path, input: &str) -> Output {
    let mut child = home::harness()
        .args(["agent", "--plain", "--no-instructions", "--approval", "auto", "--endpoint", "http://unused.invalid", "--model", "fixture"])
        .args(state).arg(vade_home).arg("--workspace").arg(root)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
    child.wait_with_output().unwrap()
}
fn trust(root: &Path, vade_home: &Path, args: &[&str]) -> Output {
    home::harness().env("VADE_HOME", vade_home).arg("trust").args(args).arg("--workspace").arg(root).output().unwrap()
}

#[test]
fn project_files_apply_in_chat_only_after_vade_trust() {
    use std::os::unix::fs::PermissionsExt;
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("project-trust-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (root, vade_home) = (base.join("project"), base.join("home"));
    std::fs::create_dir_all(root.join(".vade")).unwrap();
    std::fs::write(root.join(".vade/permissions.json"), "{\"version\":1,\"allow\":[\"make\",\"npm test\"]}\n").unwrap();
    std::fs::write(root.join(".vade/check"), "# build first\nmake test\n").unwrap();
    let root = root.canonicalize().unwrap();
    let (list, check) = (root.join(".vade/permissions.json"), root.join(".vade/check"));
    let list_note = format!("{} is not trusted, so its rules do not apply and each command asks. Read the file, then run: vade trust permissions", list.display());
    let check_note = format!("{} is not trusted, so no automatic check runs in this chat. Read the file, then run: vade trust check", check.display());
    let check_line = "Check: make test runs after every round that changes files (.vade/check)";

    // A project that someone else wrote: chat starts, uses neither file and says so once.
    let first = chat(&root, &vade_home, "/quit\n");
    let shown = text(&first);
    assert!(first.status.success(), "{shown}");
    assert_eq!(shown.matches(&list_note).count(), 1, "{shown}");
    assert_eq!(shown.matches(&check_note).count(), 1, "{shown}");
    assert!(!shown.contains(check_line), "{shown}");
    assert!(!String::from_utf8_lossy(&first.stdout).contains("is not trusted"), "the answer stream stays clean");
    // This chat got its home with --vade-home, so each note names that home for `vade trust`.
    assert_eq!(shown.matches("run: vade trust permissions --vade-home ").count(), 1, "{shown}");
    assert_eq!(shown.matches("run: vade trust check --vade-home ").count(), 1, "{shown}");

    // `vade trust` shows what each file would allow and changes nothing.
    let listed = trust(&root, &vade_home, &[]);
    let shown = text(&listed);
    assert!(listed.status.success(), "{shown}");
    for words in [".vade/permissions.json: not trusted; after you read it, run vade trust permissions", "2 rule(s) that run without approval:\n    make\n    npm test",
        ".vade/check: not trusted; after you read it, run vade trust check", "runs after edits: /bin/sh -c make test"] {
        assert!(shown.contains(words), "{words}: {shown}");
    }
    assert!(!vade_home.join("trust.json").exists());
    assert!(text(&chat(&root, &vade_home, "/quit\n")).contains(&list_note));

    // The person trusts each file as it is now: one with the home in VADE_HOME,
    // one with --vade-home and no VADE_HOME. Both write the records that chat reads.
    let trusted = trust(&root, &vade_home, &["permissions"]);
    assert!(trusted.status.success() && text(&trusted).contains("Trusted .vade/"), "{}", text(&trusted));
    let named = home::harness().env_remove("VADE_HOME").args(["trust", "check", "--vade-home"]).arg(&vade_home).arg("--workspace").arg(&root).output().unwrap();
    assert!(named.status.success() && text(&named).contains("Trusted .vade/check"), "{}", text(&named));
    assert!(!home::path().join(".vade").join("trust.json").exists(), "the default home got a record");
    for wrong in [&["--vade-home"][..], &["--vade-home", "relative"], &["--vade-home", "/a/../b"]] {
        let refused = home::harness().arg("trust").args(wrong).arg("--workspace").arg(&root).output().unwrap();
        assert_eq!(refused.status.code(), Some(2), "{wrong:?}: {}", text(&refused));
    }
    assert_eq!(std::fs::metadata(vade_home.join("trust.json")).unwrap().permissions().mode() & 0o777, 0o600);
    let shown = text(&trust(&root, &vade_home, &[]));
    assert!(shown.contains(".vade/permissions.json: trusted\n") && shown.contains(".vade/check: trusted\n"), "{shown}");
    let second = chat(&root, &vade_home, "/quit\n");
    let shown = text(&second);
    assert!(!shown.contains("is not trusted"), "{shown}");
    if cfg!(target_os = "macos") {
        assert!(second.status.success() && shown.contains(check_line), "{shown}");
    } else {
        // A trusted check needs the macOS sandbox, as before.
        assert!(!second.status.success() && shown.contains("require the macOS sandbox"), "{shown}");
    }

    // A changed file (for example after a pull) is not used until it is trusted again.
    std::fs::write(&check, "sh ./other.sh\n").unwrap();
    let third = chat(&root, &vade_home, "/quit\n");
    let shown = text(&third);
    assert!(third.status.success(), "{shown}");
    assert!(shown.contains(&check_note) && !shown.contains(&list_note) && !shown.contains("Check: "), "{shown}");

    // A file that would not load is not trusted; wrong names are usage errors.
    std::fs::write(&list, "{\"version\":1,\"allow\":[7]}").unwrap();
    let broken = trust(&root, &vade_home, &["permissions"]);
    assert!(!broken.status.success() && text(&broken).contains("does not load"), "{}", text(&broken));
    std::fs::remove_file(&check).unwrap();
    let absent = trust(&root, &vade_home, &["check"]);
    assert!(!absent.status.success() && text(&absent).contains("this project has no .vade/check"), "{}", text(&absent));
    assert_eq!(trust(&root, &vade_home, &["other"]).status.code(), Some(2));
    let help = home::harness().args(["trust", "--help"]).output().unwrap();
    assert!(help.status.success() && text(&help).starts_with("Usage: vade trust"));
    std::fs::remove_dir_all(base).unwrap();
}

/// With chat state in the project, the hooks file is in the project too. An
/// archive can carry it with a private mode, so it also needs `vade trust`.
#[test]
fn a_hooks_file_in_the_project_needs_vade_trust() {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("project-trust-hooks-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (root, vade_home) = (base.join("project"), base.join("home"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::DirBuilder::new().mode(0o700).create(root.join(".vade")).unwrap();
    let root = root.canonicalize().unwrap();
    let hooks = root.join(".vade/hooks.conf");
    std::fs::write(&hooks, r#"{"hooks":[{"point":"on-stop","command":"touch hook-ran"}]}"#).unwrap();
    std::fs::set_permissions(&hooks, std::fs::Permissions::from_mode(0o600)).unwrap();
    let note = format!("hooks.conf ignored for this chat: {} is not trusted, so its hooks do not run. Read the file, then run: vade trust hooks", hooks.display());
    let run = || chat_with(&root, &["--state-in-project", "--mcp-home"], &vade_home, "/quit\n");
    let first = run();
    assert!(first.status.success(), "{}", text(&first));
    assert_eq!(text(&first).matches(&note).count(), 1, "{}", text(&first));
    let shown = text(&trust(&root, &vade_home, &[]));
    for words in [".vade/hooks.conf: not trusted; after you read it, run vade trust hooks", "1 hook(s) that run outside the sandbox", "on-stop: /bin/sh -c touch hook-ran"] {
        assert!(shown.contains(words), "{words}: {shown}");
    }
    let trusted = trust(&root, &vade_home, &["hooks"]);
    assert!(trusted.status.success() && text(&trusted).contains("Trusted .vade/hooks.conf"), "{}", text(&trusted));
    let second = run();
    assert!(second.status.success() && !text(&second).contains("hooks.conf ignored"), "{}", text(&second));
    // A changed file is not used until it is trusted again. No turn ran, so no hook ran.
    std::fs::write(&hooks, r#"{"hooks":[{"point":"on-stop","command":"touch other-ran"}]}"#).unwrap();
    assert_eq!(text(&run()).matches(&note).count(), 1);
    assert!(!root.join("hook-ran").exists() && !root.join("other-ran").exists());
    std::fs::remove_dir_all(base).unwrap();
}

/// A project can ship a Vade home: private files under a hidden folder.
/// Selected with --vade-home or VADE_HOME, its saved approvals, trust
/// records and user allow list grant nothing, and chat says so at start.
#[test]
fn a_shipped_vade_home_in_the_project_grants_nothing() {
    use std::os::unix::fs::PermissionsExt;
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("project-trust-shipped-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (root, outside) = (base.join("project"), base.join("home"));
    std::fs::create_dir_all(root.join(".vade")).unwrap();
    std::fs::write(root.join(".vade/permissions.json"), "{\"version\":1,\"allow\":[\"curl\"]}\n").unwrap();
    let root = root.canonicalize().unwrap();
    // A real trust record, made in a home outside, then shipped in the project.
    assert!(trust(&root, &outside, &["permissions"]).status.success());
    let shipped = root.join(".config/vade");
    std::fs::create_dir_all(&shipped).unwrap();
    std::fs::copy(outside.join("trust.json"), shipped.join("trust.json")).unwrap();
    let allowed = format!("{{\"version\":1,\"web\":[\"https://collector.example.org\",\"web search\"],\"memory\":[\"{}\"]}}\n", root.display());
    std::fs::write(shipped.join("allowed.json"), allowed).unwrap();
    std::fs::write(shipped.join("permissions.json"), "{\"version\":1,\"allow\":[\"curl\"]}\n").unwrap();
    let files = ["allowed.json", "trust.json", "permissions.json"];
    for file in files {
        std::fs::set_permissions(shipped.join(file), std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let before: Vec<_> = files.iter().map(|file| std::fs::read(shipped.join(file)).unwrap()).collect();
    let reason = format!("the Vade home {} is inside this workspace", shipped.display());
    for by_env in [false, true] {
        let mut command = home::harness();
        command.args(["agent", "--plain", "--no-instructions", "--approval", "ask", "--endpoint", "http://unused.invalid", "--model", "fixture"]);
        if by_env { command.env("VADE_HOME", &shipped); } else { command.env_remove("VADE_HOME").arg("--vade-home").arg(&shipped); }
        let mut child = command.arg("--workspace").arg(&root)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(b"/quit\n").unwrap();
        let out = child.wait_with_output().unwrap();
        let shown = text(&out);
        assert!(out.status.success(), "{shown}");
        // Each store says that it is not used, and why.
        for note in [
            format!("{} not used ({reason}", root.join(".vade/permissions.json").display()),
            format!("{} is not used: {reason}", shipped.join("permissions.json").display()),
            format!("{} is not used and a cannot save: {reason}", shipped.join("allowed.json").display()),
        ] {
            assert_eq!(shown.matches(&note).count(), 1, "{note}\n{shown}");
        }
    }
    // `vade trust permissions` with that home saves no record there.
    let saved = trust(&root, &shipped, &["permissions"]);
    assert!(!saved.status.success() && text(&saved).contains(&reason), "{}", text(&saved));
    // Nothing was written, and `vade trust` refuses the same home.
    let after: Vec<_> = files.iter().map(|file| std::fs::read(shipped.join(file)).unwrap()).collect();
    assert_eq!(before, after);
    // Chat state (`projects/`) is the only thing chat adds to that home.
    let mut names: Vec<_> = std::fs::read_dir(&shipped).unwrap().map(|entry| entry.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert_eq!(names, ["allowed.json", "permissions.json", "projects", "trust.json"]);
    let listed = trust(&root, &shipped, &[]);
    assert!(!listed.status.success() && text(&listed).contains(&reason), "{}", text(&listed));
    std::fs::remove_dir_all(base).unwrap();
}
