//! Follow-up installer checks. All paths and fake tools stay in the fixture.
use super::*;

fn mark(f: &Fixture) {
    fs::create_dir_all(f.root.join("home/.vade")).unwrap();
    fs::set_permissions(f.root.join("home/.vade"), fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(f.root.join("home/.vade/.vade-home"), "vade-home-v1\n").unwrap();
}

fn replace_tool(f: &Fixture, name: &str, script: &str) {
    fs::remove_file(f.root.join("bin").join(name)).unwrap();
    f.tool(name, script);
}



#[test]
fn a_parent_swap_cannot_redirect_uninstall_removal() {
    let f = Fixture::new();
    mark(&f);
    for dir in ["home/.vade/stage", "outside/stage"] { fs::create_dir_all(f.root.join(dir)).unwrap(); }
    fs::write(f.root.join("outside/stage/keep"), "mine\n").unwrap();
    replace_tool(&f, "rm", r#"
case "$*" in *stage*)
    if [ ! -e "$INSTALL_ROOT/swapped" ]; then
        : > "$INSTALL_ROOT/swapped"
        /bin/mv "$HOME/.vade" "$HOME/saved"
        /bin/ln -s "$INSTALL_ROOT/outside" "$HOME/.vade"
    fi ;;
esac
exec /bin/rm "$@"
"#);
    let _ = f.run_args(&[], &["--uninstall", "--all"]);
    assert!(f.root.join("swapped").exists());
    assert_eq!(fs::read_to_string(f.root.join("outside/stage/keep")).unwrap(), "mine\n");
}

#[test]
fn invalid_launcher_paths_are_refused_before_install_work() {
    for kind in ["bin-link", "bin-file", "script-folder", "script-link"] {
        let f = Fixture::new();
        mark(&f);
        let bin = f.root.join("home/.vade/bin");
        match kind {
            "bin-link" => std::os::unix::fs::symlink(f.root.join("tmp"), &bin).unwrap(),
            "bin-file" => fs::write(&bin, "mine\n").unwrap(),
            _ => {
                fs::create_dir(&bin).unwrap();
                if kind == "script-folder" { fs::create_dir(bin.join("vade")).unwrap(); }
                else { std::os::unix::fs::symlink(f.root.join("tmp"), bin.join("vade")).unwrap(); }
            }
        }
        let out = f.run(&[]);
        assert!(!out.status.success(), "{kind}");
        assert!(!f.calls().contains("cargo ") && !f.calls().contains("vade __install-here"), "{kind}: {}", f.calls());
    }
}

#[test]
fn a_bin_swap_cannot_redirect_the_launcher_write() {
    let f = Fixture::new();
    mark(&f);
    fs::create_dir_all(f.root.join("outside")).unwrap();
    fs::write(f.root.join("outside/vade"), "mine\n").unwrap();
    let mktemp = ["/usr/bin/mktemp", "/bin/mktemp"].into_iter().find(|path| Path::new(path).is_file()).unwrap();
    // The first launcher temporary file: swap its folder for a link to outside, then make it.
    replace_tool(&f, "mktemp", &format!(r#"
for last do :; done
case $last in *.vade.*)
    if [ ! -e "$INSTALL_ROOT/swapped" ]; then
        : > "$INSTALL_ROOT/swapped"
        dir=$(cd -P -- "$(/usr/bin/dirname -- "$last")" && pwd -P)
        /bin/mv "$dir" "$dir.saved"
        /bin/ln -s "$INSTALL_ROOT/outside" "$dir"
    fi ;;
esac
exec {mktemp} "$@"
"#));
    let _ = f.run(&[]);
    assert!(f.root.join("swapped").exists());
    assert_eq!(fs::read_to_string(f.root.join("outside/vade")).unwrap(), "mine\n");
    assert_eq!(fs::read_dir(f.root.join("outside")).unwrap().count(), 1, "a file was made outside");
}

/// Every file and folder under `dir` with its contents, to prove a folder unchanged.
fn listing(dir: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() { stack.push(path.clone()); found.push((path, None)); }
            else { found.push((path.clone(), Some(fs::read(&path).unwrap()))); }
        }
    }
    found.sort();
    found
}

/// Wrap the system `tool`: on the first call whose arguments match the case
/// pattern `when`, run the real tool, then put a link to `outside` at the folder
/// that the shell code `folder` prints (moving the real one away), then return
/// the real tool's output and status. So the swap falls between that step and
/// the next one, after every check the installer made on the folder.
fn swap_after(f: &Fixture, tool: &str, when: &str, folder: &str) {
    let real = system_tool(tool);
    replace_tool(f, tool, &format!(r#"
set +e
out=$('{real}' "$@")
status=$?
case "$*" in {when})
    if [ "$status" = 0 ] && [ ! -e "$INSTALL_ROOT/swapped" ]; then
        : > "$INSTALL_ROOT/swapped"
        folder=$({folder})
        /bin/mv "$folder" "$INSTALL_ROOT/moved"
        /bin/ln -s "$INSTALL_ROOT/outside" "$folder"
    fi ;;
esac
[ -z "$out" ] || printf '%s\n' "$out"
exit "$status"
"#, real = real.display()));
}

const DOWNLOAD: &str = "/bin/ls -d \"$HOME\"/.vade/bundles/.download.*";

#[test]
fn a_swapped_download_folder_never_leads_a_bundle_step_outside() {
    let file = format!("{}.tar.gz", stem());
    // tool, the call after which the swap comes, the folder swapped.
    let cases = [
        // After mktemp made .download.X: curl's error file, the unpack folder and tar.
        ("mktemp", "*.download.*", "case $out in (/*) printf '%s\\n' \"$out\" ;; (*) printf '%s\\n' \"$HOME/.vade/bundles/${out#./}\" ;; esac"),
        // After the archive's sha256 check: the archive move (main moved it into .download.X).
        ("sha256sum", "*bundle.part*", DOWNLOAD),
        // After the MANIFEST check of the unpacked bundle: the move into bundles.
        ("sha256sum", "*-c*", DOWNLOAD),
        // After the bundle moved into place: the cache moved into .download.X for removal.
        ("mv", "*unpack*|*../../*", DOWNLOAD),
    ];
    let mut changed = Vec::new();
    for (tool, when, folder) in cases {
        let f = Fixture::new();
        let env = f.prebuilt(&bundle(Bad::None), Bad::None);
        let outside = f.root.join("outside");
        fs::create_dir_all(outside.join("unpack").join(stem())).unwrap();
        fs::write(outside.join("unpack").join(stem()).join("keep"), "theirs\n").unwrap();
        fs::write(outside.join("curl.error"), "theirs\n").unwrap();
        fs::write(outside.join(&file), "theirs\n").unwrap();
        let before = listing(&outside);
        swap_after(&f, tool, when, folder);
        let output = f.run_prebuilt(&env, &[]);
        let said = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert!(f.root.join("swapped").exists(), "{tool} {when}: no swap\n{said}");
        if listing(&outside) != before { changed.push(format!("{tool} {when}: the folder outside changed\n{said}")); }
    }
    assert!(changed.is_empty(), "{}", changed.join("\n"));
}

#[test]
fn a_swapped_clone_folder_never_receives_the_clone() {
    let f = Fixture::new();
    f.public_main();
    let outside = f.root.join("outside");
    fs::create_dir_all(&outside).unwrap();
    swap_after(&f, "mktemp", "*source.*", "case $out in (/*) printf '%s\\n' \"$out\" ;; (*) printf '%s\\n' \"$HOME/.vade/${out#./}\" ;; esac");
    let output = f.piped(&[], &[]);
    let said = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(f.root.join("swapped").exists(), "no swap\n{said}");
    assert!(listing(&outside).is_empty(), "the clone went outside\n{said}");
    assert!(!output.status.success(), "{said}");
}
