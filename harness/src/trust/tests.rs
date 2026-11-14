use super::*;
use crate::test_workspace::Fixture;

#[test]
fn digest_matches_the_standard_vectors() {
    assert_eq!(digest(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(digest(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // 56 bytes: the length goes into a second block.
    assert_eq!(digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    assert_eq!(digest(&[b'a'; 1000]), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn a_record_names_one_project_one_file_and_its_exact_bytes() {
    use std::os::unix::fs::PermissionsExt;
    let (project, other, vade) = (Fixture::new(), Fixture::new(), Fixture::new());
    let home = vade.0.join("home");
    let (root, elsewhere) = (project.0.canonicalize().unwrap(), other.0.canonicalize().unwrap());
    // No home, no file and no record all mean not trusted.
    assert_eq!(holds(None, &root, "check", b"make\n"), Ok(false));
    assert_eq!(holds(Some(&home), &root, "check", b"make\n"), Ok(false));
    Store::open(&home, &root).unwrap().set(&root, "check", Some(&digest(b"make\n"))).unwrap();
    assert_eq!(holds(Some(&home), &root, "check", b"make\n"), Ok(true));
    // Other bytes, another file name and another project folder have no record.
    assert_eq!(holds(Some(&home), &root, "check", b"make all\n"), Ok(false));
    assert_eq!(holds(Some(&home), &root, "permissions.json", b"make\n"), Ok(false));
    assert_eq!(holds(Some(&home), &elsewhere, "check", b"make\n"), Ok(false));
    // The home and the file are private; a second record keeps the first.
    assert_eq!(std::fs::metadata(&home).unwrap().permissions().mode() & 0o777, 0o700);
    assert_eq!(std::fs::metadata(home.join(STORE)).unwrap().permissions().mode() & 0o777, 0o600);
    let mut store = Store::open(&home, &root).unwrap();
    store.set(&elsewhere, "check", Some(&digest(b"npm test\n"))).unwrap();
    assert_eq!(store.recorded(&root, "check"), Some(digest(b"make\n")));
    store.set(&root, "check", Some(&digest(b"make all\n"))).unwrap();
    drop(store);
    assert_eq!(holds(Some(&home), &root, "check", b"make\n"), Ok(false));
    assert_eq!(holds(Some(&home), &root, "check", b"make all\n"), Ok(true));
    assert_eq!(holds(Some(&home), &elsewhere, "check", b"npm test\n"), Ok(true));
    Store::open(&home, &root).unwrap().set(&root, "check", None).unwrap();
    assert_eq!(holds(Some(&home), &root, "check", b"make all\n"), Ok(false));
    // A record file that others can change, or that does not parse, trusts nothing and says why.
    let path = home.join(STORE);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(holds(Some(&home), &elsewhere, "check", b"npm test\n").unwrap_err().contains("chmod 600"));
    assert!(Store::open(&home, &root).is_err());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    for broken in ["", "[]", r#"{"version":2,"trusted":[]}"#, r#"{"version":1}"#, r#"{"version":1,"trusted":[{"project":"/p","file":"check"}]}"#] {
        std::fs::write(&path, broken).unwrap();
        assert!(holds(Some(&home), &elsewhere, "check", b"npm test\n").is_err(), "{broken}");
    }
    // A linked home is not a place for records.
    let linked = vade.0.join("linked");
    std::fs::remove_file(&path).unwrap();
    Store::open(&home, &root).unwrap().set(&root, "check", Some(&digest(b"make\n"))).unwrap();
    std::os::unix::fs::symlink(&home, &linked).unwrap();
    assert!(holds(Some(&linked), &root, "check", b"make\n").unwrap_err().contains("not a link"));
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn project_files_read_without_links_and_within_their_limit() {
    let (project, outside) = (Fixture::new(), Fixture::new());
    let root = project.0.canonicalize().unwrap();
    assert_eq!(project_file(&root, "check", 4096), Ok(None));
    std::fs::create_dir(root.join(".vade")).unwrap();
    assert_eq!(project_file(&root, "check", 4096), Ok(None));
    std::fs::write(root.join(".vade/check"), "make\n").unwrap();
    assert_eq!(project_file(&root, "check", 4096), Ok(Some("make\n".into())));
    std::fs::write(root.join(".vade/check"), "x".repeat(4097)).unwrap();
    assert!(project_file(&root, "check", 4096).unwrap_err().contains("larger"));
    std::fs::write(root.join(".vade/check"), [0xff, 0xfe]).unwrap();
    assert!(project_file(&root, "check", 4096).unwrap_err().contains("UTF-8"));
    // A linked file and a linked folder are refused; they could name a file outside the project.
    std::fs::write(outside.0.join("check"), "make\n").unwrap();
    std::fs::remove_file(root.join(".vade/check")).unwrap();
    std::os::unix::fs::symlink(outside.0.join("check"), root.join(".vade/check")).unwrap();
    assert!(project_file(&root, "check", 4096).is_err());
    std::fs::remove_dir_all(root.join(".vade")).unwrap();
    std::os::unix::fs::symlink(&outside.0, root.join(".vade")).unwrap();
    assert!(project_file(&root, "check", 4096).unwrap_err().contains("link"));
}

#[test]
fn the_note_names_the_file_the_effect_and_the_command() {
    let root = Path::new("/work/repo");
    assert_eq!(
        note("check", root, "no automatic check runs in this chat", None, None),
        "/work/repo/.vade/check is not trusted, so no automatic check runs in this chat. Read the file, then run: vade trust check"
    );
    // A chat with another home than the default one gets the command that writes to its home.
    assert_eq!(
        note("check", root, "no automatic check runs in this chat", None, Some(Path::new("/srv/vade-2"))),
        "/work/repo/.vade/check is not trusted, so no automatic check runs in this chat. Read the file, then run: vade trust check --vade-home /srv/vade-2"
    );
    assert!(note("check", root, "x", None, Some(Path::new("/srv/my vade/it's"))).ends_with("vade trust check --vade-home '/srv/my vade/it'\\''s'"));
    if let Ok(home) = default_home() {
        assert!(note("check", root, "x", None, Some(&home)).ends_with("run: vade trust check"));
    }
    assert_eq!(
        note("permissions", root, "its rules do not apply and each command asks", Some("/h/trust.json: is not valid JSON"), None),
        "/work/repo/.vade/permissions.json not used (/h/trust.json: is not valid JSON), so its rules do not apply and each command asks."
    );
    assert_eq!(describe("check", "# build\n\n  make test  \nignored\n").unwrap(), "  runs after edits: /bin/sh -c make test\n");
    assert_eq!(describe("check", "# only a comment\n").unwrap(), "  holds no command\n");
    // Chat refuses a check line with a control character, so `vade trust check` refuses it first.
    assert!(describe("check", "make\ttest\n").unwrap_err().contains("no control character"));
    assert!(describe("check", &format!("{}\n", "x".repeat(4097))).is_err());
    assert_eq!(
        describe("hooks", r#"{"hooks":[{"point":"before-tool","command":"sh a.sh"},{"point":"on-stop","command":"sh b.sh"}]}"#).unwrap(),
        "  2 hook(s) that run outside the sandbox, with --state-in-project:\n    before-tool: /bin/sh -c sh a.sh\n    on-stop: /bin/sh -c sh b.sh\n"
    );
    assert!(describe("hooks", "not JSON").is_err());
    assert_eq!(note("hooks", root, "its hooks do not run", None, None), "/work/repo/.vade/hooks.conf is not trusted, so its hooks do not run. Read the file, then run: vade trust hooks");
    assert_eq!(describe("permissions", "{\"version\":1,\"allow\":[\"make\",\"npm test\"]}").unwrap(), "  2 rule(s) that run without approval:\n    make\n    npm test\n");
    assert!(describe("permissions", "{\"version\":1,\"allow\":[7]}").is_err());
    // `vade trust --vade-home DIR` takes one absolute folder.
    let words = |words: &[&str]| words.iter().map(|word| word.to_string()).collect::<Vec<_>>();
    let mut args = words(&["check", "--vade-home", "/srv/vade-2", "--workspace", "/w"]);
    assert_eq!(home_argument(&mut args), Ok(Some(PathBuf::from("/srv/vade-2"))));
    assert_eq!(args, words(&["check", "--workspace", "/w"]));
    assert_eq!(home_argument(&mut words(&["check"])), Ok(None));
    for wrong in [&["--vade-home"][..], &["--vade-home", "relative"], &["--vade-home", "/a/../b"], &["--vade-home", "/a", "--vade-home", "/b"]] {
        assert!(home_argument(&mut words(wrong)).is_err(), "{wrong:?}");
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn a_home_that_model_tools_can_write_holds_no_trusted_record() {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let project = Fixture::new();
    let root = project.0.canonicalize().unwrap();
    std::fs::create_dir(root.join(".vade")).unwrap();
    std::fs::write(root.join(".vade/check"), "make\n").unwrap();
    // A sandboxed command or apply_patch can write a plain folder of the
    // workspace. A private record file planted there trusts nothing.
    let home = root.join("vade-home");
    std::fs::DirBuilder::new().mode(0o700).create(&home).unwrap();
    let planted = render(&[(root.to_string_lossy().into_owned(), "check".into(), digest(b"make\n"))]).unwrap();
    std::fs::write(home.join(STORE), planted).unwrap();
    std::fs::set_permissions(home.join(STORE), std::fs::Permissions::from_mode(0o600)).unwrap();
    let reason = holds(Some(&home), &root, "check", b"make\n").unwrap_err();
    assert!(reason.contains("inside this workspace"), "{reason}");
    // The person is told why in one line.
    let line = note("check", &root, "no automatic check runs in this chat", Some(&reason), Some(&home));
    assert!(!line.contains('\n') && line.contains("not used (the Vade home") && line.ends_with("so no automatic check runs in this chat."), "{line}");
    // Neither `vade trust` nor a `p` answer writes a record there.
    let before = std::fs::read(home.join(STORE)).unwrap();
    assert!(Store::open(&home, &root).err().unwrap().contains("nothing is written"));
    let args = |home: &Path| ["check".to_string(), "--workspace".into(), root.display().to_string(), "--vade-home".into(), home.display().to_string()];
    assert_eq!(cli(args(&home)), ExitCode::FAILURE);
    assert_eq!(std::fs::read(home.join(STORE)).unwrap(), before);
    let fresh = root.join("other-home");
    assert_eq!(cli(args(&fresh)), ExitCode::FAILURE);
    assert!(!fresh.join(STORE).exists());
    // A folder with a non-ASCII name: the file tools refuse it, a sandboxed command does not.
    let accented = root.join("caf\u{e9}/vade");
    assert!(Store::open(&accented, &root).err().unwrap().contains("nothing is written"));
    assert!(holds(Some(&accented), &root, "check", b"make\n").unwrap_err().contains("inside this workspace"));
    // The same home named through a link from outside the workspace.
    let outside = Fixture::new();
    std::os::unix::fs::symlink(&root, outside.0.join("link")).unwrap();
    assert!(holds(Some(&outside.0.join("link/vade-home")), &root, "check", b"make\n").unwrap_err().contains("inside this workspace"));
    // A project can ship a private record under a hidden folder too.
    let hidden = root.join(".config/vade");
    std::fs::create_dir_all(&hidden).unwrap();
    let planted = render(&[(root.to_string_lossy().into_owned(), "check".into(), digest(b"make\n"))]).unwrap();
    std::fs::write(hidden.join(STORE), planted).unwrap();
    std::fs::set_permissions(hidden.join(STORE), std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(holds(Some(&hidden), &root, "check", b"make\n").is_err());
    assert_eq!(cli(args(&hidden)), ExitCode::FAILURE);
}
