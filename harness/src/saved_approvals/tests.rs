use super::*;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

struct Temp(PathBuf);
impl Temp {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("vade-saved-approvals-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn file(web: &[&str], memory: &[&str]) -> String {
    render(&Saved { web: web.iter().map(|s| s.to_string()).collect(), memory: memory.iter().map(|s| s.to_string()).collect() })
}

#[test]
fn the_file_holds_only_exact_site_keys_and_project_folders() {
    let text = file(&["https://docs.example.org", "http://docs.example.org:8080", "web search", "https://[2001:4860::8888]"], &["/work/project"]);
    let saved = parse(&text).unwrap();
    assert_eq!(saved.web, ["https://docs.example.org", "http://docs.example.org:8080", "web search", "https://[2001:4860::8888]"]);
    assert_eq!(saved.memory, ["/work/project"]);
    assert_eq!(render(&saved), text);
    assert_eq!(parse(&file(&[], &[])).unwrap(), Saved::default());
    // A hand-written entry that is looser than the site key the prompt saves
    // does not load: no pattern, path, user info, case, default port or bare name.
    for loose in [
        "https://*.docs.example.org", "*", "https://docs.example.org/", "https://docs.example.org/a", "https://Docs.example.org", "HTTPS://docs.example.org",
        "https://docs.example.org:443", "http://docs.example.org:80", "https://user@docs.example.org", "docs.example.org", "https://docs", "ftp://docs.example.org",
        "https://docs.example.org?x", "https://docs.example.org#x", "https://docs.example.org.", "https://docs.example.org:0443", "https://docs.example.org:08443",
        "https://[2001:4860:0:0:0:0:0:8888]", "https://[2001:4860::8888]:443", "https://127.0.0.1", "https://0x7f000001", "https://2130706433", "web  search", "Web search", "",
    ] {
        assert!(parse(&file(&[loose], &[])).unwrap_err().contains("is not an exact site"), "{loose:?}");
    }
    for bad in ["relative/folder", "/work/../other", ""] {
        assert!(parse(&file(&[], &[bad])).is_err(), "{bad:?}");
    }
    for bad in [
        "", "[]", r#"{"web":[]}"#, r#"{"version":2}"#, r#"{"version":1,"web":"https://docs.example.org"}"#, r#"{"version":1,"web":[7]}"#,
        r#"{"version":1,"allow":[]}"#, r#"{"version":1,"web":[],"web":["https://evil.example.org"]}"#,
    ] {
        assert!(parse(bad).is_err(), "{bad}");
    }
}

#[cfg(unix)]
#[test]
fn the_file_reads_only_as_a_private_regular_file_of_yours() {
    let temp = Temp::new("private");
    let home = temp.0.join("home");
    assert_eq!(read(&home).unwrap(), Saved::default());
    // An `a` answer makes the home (0700) and a file only you can read and write.
    assert_eq!(add(&home, "https://docs.example.org", false).unwrap(), home.join(FILE_NAME).display().to_string());
    assert_eq!(add(&home, "/work/project", true).unwrap(), home.join(FILE_NAME).display().to_string());
    assert_eq!(std::fs::metadata(home.join(FILE_NAME)).unwrap().permissions().mode() & 0o777, 0o600);
    assert_eq!(std::fs::metadata(&home).unwrap().permissions().mode() & 0o777, 0o700);
    assert_eq!(read(&home).unwrap(), Saved { web: vec!["https://docs.example.org".into()], memory: vec!["/work/project".into()] });
    // Saving the same entry again changes nothing; a loose or empty key is refused.
    add(&home, "https://docs.example.org", false).unwrap();
    assert_eq!(read(&home).unwrap().web.len(), 1);
    for loose in ["https://*.example", "https://docs.example.org/", "relative"] {
        assert!(add(&home, loose, loose == "relative").is_err(), "{loose}");
    }
    // A file that others can read or write, a hard link and a symbolic link do not load.
    let path = home.join(FILE_NAME);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(read(&home).unwrap_err().contains("chmod 600"));
    assert!(add(&home, "https://other.example.org", false).unwrap_err().contains("chmod 600"));
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::hard_link(&path, temp.0.join("second-name")).unwrap();
    assert!(read(&home).unwrap_err().contains("chmod 600"));
    std::fs::remove_file(temp.0.join("second-name")).unwrap();
    assert!(read(&home).is_ok());
    std::fs::rename(&path, temp.0.join("elsewhere.json")).unwrap();
    std::os::unix::fs::symlink(temp.0.join("elsewhere.json"), &path).unwrap();
    assert!(read(&home).unwrap_err().contains("chmod 600"));
    std::fs::remove_file(&path).unwrap();
    // A Vade home that is a link is refused.
    std::os::unix::fs::symlink(&home, temp.0.join("home-link")).unwrap();
    assert!(read(&temp.0.join("home-link")).unwrap_err().contains("not a link"));
}

#[cfg(unix)]
#[test]
fn vade_allowed_lists_and_removes_one_entry() {
    let temp = Temp::new("cli");
    let home = temp.0.join("home");
    let root = temp.0.join("project");
    std::fs::create_dir_all(&root).unwrap();
    let run = |words: &[&str]| {
        let mut args: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        args.extend(["--vade-home".into(), home.display().to_string(), "--workspace".into(), root.display().to_string()]);
        cli(args)
    };
    assert_eq!(run(&["list"]), ExitCode::SUCCESS);
    add(&home, "https://docs.example.org", false).unwrap();
    add(&home, "https://docs.example.org:8443", false).unwrap();
    add(&home, crate::web::SEARCH_TARGET, false).unwrap();
    add(&home, &root.display().to_string(), true).unwrap();
    assert_eq!(run(&["list"]), ExitCode::SUCCESS);
    assert_eq!(run(&[]), ExitCode::SUCCESS);
    // Only the exact entry goes: the site on another port stays.
    assert_eq!(run(&["remove", "https://docs.example.org"]), ExitCode::SUCCESS);
    assert_eq!(read(&home).unwrap().web, ["https://docs.example.org:8443", "web search"]);
    assert_eq!(run(&["remove", "https://docs.example.org"]), ExitCode::FAILURE);
    assert_eq!(run(&["remove", "https://DOCS.example.org:8443"]), ExitCode::FAILURE);
    // The search key is two words, as the list shows it.
    assert_eq!(run(&["remove", "web", "search"]), ExitCode::SUCCESS);
    // `memory` alone names the project of --workspace.
    assert_eq!(run(&["remove", "memory"]), ExitCode::SUCCESS);
    assert_eq!(read(&home).unwrap(), Saved { web: vec!["https://docs.example.org:8443".into()], memory: Vec::new() });
    assert_eq!(std::fs::metadata(home.join(FILE_NAME)).unwrap().permissions().mode() & 0o777, 0o600);
    for wrong in [&["remove"][..], &["list", "x"], &["add", "https://evil.example.org"], &["trust"]] {
        assert_eq!(run(wrong), ExitCode::from(2), "{wrong:?}");
    }
    assert_eq!(cli(["list".to_string(), "--vade-home".into(), "relative".into()]), ExitCode::from(2));
}

#[cfg(unix)]
#[test]
fn removal_advice_quotes_entries_and_names_the_selected_home() {
    let f = Temp::new("advice");
    let root = f.0.join("a;$(touch planted)'s");
    let selected = f.0.join("custom home");
    std::fs::create_dir_all(&root).unwrap();
    let approvals = crate::shell::permissions::Approvals::for_workspace(&root, Some(&selected), true).unwrap();
    let advice = approvals.save_always(None, true).unwrap();
    assert!(advice.contains("a;$(touch planted)'\\''s'"), "{advice}");
    assert!(advice.ends_with(&format!(" --vade-home '{}'", selected.display())), "{advice}");
    // Check the shell's argument parsing without executing any advice.
    let args = advice.strip_prefix("To take this back, run: vade allowed remove ").unwrap();
    let out = std::process::Command::new("/bin/sh").args(["-c", &format!("set -- {args}; printf '%s\\n' \"$@\"")]).current_dir(&f.0).output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap(), format!("memory\n{}\n--vade-home\n{}\n", root.display(), selected.display()));
    assert!(!f.0.join("planted").exists());
}

/// The terminal drops control and direction characters, so a command with
/// them would name another path and remove nothing. Such a path is left out
/// of the command; the command that is shown removes the entry.
#[cfg(unix)]
#[test]
fn removal_advice_shows_no_path_that_the_terminal_would_change() {
    let f = Temp::new("advice-hidden");
    let plain_home = f.0.join("home");
    // A project path with ESC: run in the project folder, the shown command
    // (no path) removes the exact entry.
    let root = f.0.join("esc\u{1b}[2Kroot");
    std::fs::create_dir_all(&root).unwrap();
    let approvals = crate::shell::permissions::Approvals::for_workspace(&root, Some(&plain_home), true).unwrap();
    let advice = approvals.save_always(None, true).unwrap();
    assert_eq!(advice, format!("To take this back, run in the project folder: vade allowed remove memory --vade-home '{}'", plain_home.display()));
    assert_eq!(read(&plain_home).unwrap().memory.len(), 1);
    let root_arg = root.canonicalize().unwrap().to_str().unwrap().to_owned();
    let removed = cli(["remove".to_string(), "memory".into(), "--vade-home".into(), plain_home.display().to_string(), "--workspace".into(), root_arg]);
    assert_eq!(removed, ExitCode::SUCCESS);
    assert!(read(&plain_home).unwrap().memory.is_empty());
    // A home path that cannot be shown: no home in the command.
    let plain_root = f.0.join("plain");
    std::fs::create_dir_all(&plain_root).unwrap();
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut homes = vec![f.0.join("home\u{202e}txt")];
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStringExt;
        homes.push(f.0.join(std::ffi::OsString::from_vec(vec![b'h', 0x80])));
    }
    for home in homes {
        let approvals = crate::shell::permissions::Approvals::for_workspace(&plain_root, Some(&home), true).unwrap();
        let advice = approvals.save_always(Some("https://good.example.org/a"), false).unwrap();
        assert_eq!(advice, "To take this back, set VADE_HOME to the Vade home of this chat (its path has characters that cannot be shown here), then run: vade allowed remove 'https://good.example.org'");
        assert_eq!(read(&home).unwrap().web.len(), 1);
    }
    // A plain root and home still get the full command.
    let approvals = crate::shell::permissions::Approvals::for_workspace(&plain_root, Some(&plain_home), true).unwrap();
    assert!(approvals.save_always(Some("https://good.example.org/a"), false).unwrap().starts_with("To take this back, run: vade allowed remove 'https://good.example.org' --vade-home "));
}
