//! Whether a Vade home can hold the person's own word. Saved approvals and
//! trust records in the Vade home let things run with no prompt, so only the
//! person may write them. Model tools can write files inside the workspace:
//! the file tools (`apply_patch`, `workspace_edit_lines`) and, more widely, a
//! sandboxed shell command. A Vade home inside the workspace (`VADE_HOME`,
//! `--vade-home`, or `--mcp-home` set to the workspace or a folder in it) can
//! therefore hold a planted file, and nothing in it counts.
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Whether the shell sandbox denies writes at and under a workspace folder
/// or file of this name: a hidden (dot) name or a secret path rule, in any
/// ASCII case. This is the write rule of the shell sandbox profile.
pub fn sandbox_denies_writes_in(name: &str) -> bool {
    (name.starts_with('.') && name.len() > 1)
        || crate::tools::secret_component_reason(name).is_some()
}

/// Whether `home` must not supply authority for this workspace. A project
/// can ship files under any name, so every home inside it is refused. The
/// only exception is the default `.vade` when the workspace is HOME itself.
/// Existing ancestors and symlinks along the path are checked against the
/// canonical workspace root. A path that cannot be checked is refused.
pub fn model_can_write(home: &Path, root: &Path) -> bool {
    let os_home = std::env::var_os("HOME").and_then(|home| Path::new(&home).canonicalize().ok());
    refused(home, root, os_home.as_deref())
}

fn is_home_vade_exempt(path: &Path, root: &Path, os_home: Option<&Path>) -> bool {
    os_home == Some(root) && (path == root || path == root.join(".vade"))
}

fn refused(home: &Path, root: &Path, os_home: Option<&Path>) -> bool {
    let Ok(root) = root.canonicalize() else { return true };
    if !home.is_absolute() || home.components().any(|part| matches!(part, Component::ParentDir)) {
        return true;
    }
    if passes_through_workspace(home, &root, os_home) {
        return true;
    }
    let mut rest = Vec::new();
    let mut found = false;
    let mut inside = false;
    for at in home.ancestors() {
        if let Ok(mut real) = at.canonicalize() {
            let ancestor = real.clone();
            for part in rest.iter().rev() { real.push(part); }
            found = true;
            if ancestor.starts_with(&root) {
                inside = true;
                let default = os_home == Some(root.as_path())
                    && real == root.join(".vade")
                    && (ancestor == root || ancestor == real);
                if !default { return true; }
                let Ok(relative) = real.strip_prefix(&root) else { return true };
                for part in relative.components() {
                    let Component::Normal(name) = part else { return true };
                    let Some(name) = name.to_str() else { return true };
                    if crate::tools::allowed_component(name) || !sandbox_denies_writes_in(name) {
                        return true;
                    }
                }
            }
        }
        if let Some(name) = at.file_name() { rest.push(name.to_owned()); }
    }
    !found || (inside && os_home != Some(root.as_path()))
}

/// Follow each symlink step on the way to `home` and check canonical paths so
/// a symlink outside the workspace that hops through a project directory is
/// caught even if the project directory was later redirected outside.
fn passes_through_workspace(home: &Path, root: &Path, os_home: Option<&Path>) -> bool {
    let mut todo: Vec<OsString> = home.components().rev().map(|part| part.as_os_str().to_owned()).collect();
    let (mut at, mut links) = (PathBuf::new(), 0usize);
    while let Some(part) = todo.pop() {
        match Path::new(&part).components().next() {
            Some(Component::ParentDir) => { at.pop(); }
            Some(Component::CurDir) | None => {}
            Some(_) => at.push(&part),
        }
        if at.starts_with(root) && !is_home_vade_exempt(&at, root, os_home) {
            return true;
        }
        if let (Some(parent), Some(name)) = (at.parent(), at.file_name())
            && let Ok(real_parent) = parent.canonicalize()
        {
            let resolved = real_parent.join(name);
            if resolved.starts_with(root) && !is_home_vade_exempt(&resolved, root, os_home) {
                return true;
            }
        }
        let Ok(meta) = std::fs::symlink_metadata(&at) else { continue };
        if meta.file_type().is_symlink() {
            links += 1;
            if links > 40 { return true; }
            let Ok(target) = std::fs::read_link(&at) else { return true };
            at.pop();
            todo.extend(target.components().rev().map(|part| part.as_os_str().to_owned()));
        }
    }
    false
}

/// The canonical workspace root used to check Vade home boundaries.
pub struct Root {
    path: PathBuf,
}
impl Root {
    pub fn open(root: &Path) -> Result<Self, String> {
        let path = root.canonicalize().map_err(|_| "the workspace folder cannot be checked".to_owned())?;
        if !path.is_dir() {
            return Err("the workspace folder cannot be opened".to_owned());
        }
        Ok(Self { path })
    }
    /// The canonical workspace path, as trust records name it.
    pub fn path(&self) -> &Path { &self.path }
}

/// A Vade home, opened once and checked through an anchored directory handle.
/// Reads and writes of `allowed.json`, `trust.json`, and user `permissions.json`
/// go through `dir` by plain filename so later path swaps cannot redirect them.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub struct Home {
    dir: crate::tools::anchored::Dir,
    path: PathBuf,
}
#[cfg(any(target_os = "macos", target_os = "linux"))]
impl Home {
    /// Open `home` for workspace `root` after canonical-path boundary checks.
    /// When `create` is true, creates a missing folder with mode 0700; when
    /// false, returns `Ok(None)` if `home` does not exist yet.
    pub fn open(home: &Path, root: &Root, create: bool) -> Result<Option<Self>, String> {
        let os_home = std::env::var_os("HOME").and_then(|home| Path::new(&home).canonicalize().ok());
        Self::open_for(home, root, create, os_home.as_deref())
    }
    fn open_for(home: &Path, root: &Root, create: bool, os_home: Option<&Path>) -> Result<Option<Self>, String> {
        if refused(home, &root.path, os_home) { return Err(reason(home)); }
        if std::fs::symlink_metadata(home).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) {
            if !create { return Ok(None); }
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(home)
                .map_err(|error| format!("cannot create the Vade home: {error}"))?;
        }
        let dir = crate::tools::anchored::Dir::open_root(home)
            .map_err(|_| "the Vade home must be a real folder, not a link".to_owned())?;
        let canonical = home.canonicalize().map_err(|_| reason(home))?;
        if refused(&canonical, &root.path, os_home) { return Err(reason(home)); }
        Ok(Some(Self { dir, path: home.to_path_buf() }))
    }
    pub(crate) fn dir(&self) -> &crate::tools::anchored::Dir { &self.dir }
    pub fn path(&self) -> &Path { &self.path }
}

/// Other systems have no opened-folder checks, so no Vade home is used.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub struct Home { path: PathBuf }
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
impl Home {
    pub fn open(_: &Path, _: &Root, _: bool) -> Result<Option<Self>, String> {
        Err("a Vade home needs macOS or Linux".into())
    }
    pub fn path(&self) -> &Path { &self.path }
}

/// Why such a home is not used, for a one-line note.
pub fn reason(home: &Path) -> String {
    format!("the Vade home {} is inside this workspace or its path cannot be checked, so nothing in it counts as your own approval", home.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_home_that_model_tools_can_write_is_found_also_through_links() {
        let temp = std::env::temp_dir().join(format!("vade-home-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let temp = temp.canonicalize().unwrap();
        let root = temp.join("work");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::create_dir_all(temp.join("home")).unwrap();
        assert!(model_can_write(&root, &root));
        assert!(model_can_write(&root.join("vade-home"), &root));
        assert!(model_can_write(&root.join("sub/not/made/yet"), &root));
        assert!(model_can_write(&root.join("caf\u{e9}/vade"), &root));
        assert!(model_can_write(&root.join("\u{17f}ecrets/vade"), &root));
        assert!(model_can_write(&root.join("keys.pem.d/vade"), &root));
        assert!(model_can_write(&root.join(".vade"), &root));
        assert!(model_can_write(&root.join("sub/.config/vade"), &root));
        assert!(model_can_write(&root.join("secrets/vade"), &root));
        assert!(model_can_write(&root.join("SECRETS/vade"), &root));
        assert!(model_can_write(&root.join("caf\u{e9}/store.PEM/vade"), &root));
        assert!(!model_can_write(&temp.join("home"), &root));
        assert!(!model_can_write(&temp.join("home-not-made"), &root));
        // A name that starts like the workspace is not inside it.
        assert!(!model_can_write(&temp.join("workshop"), &root));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("sub"), temp.join("link")).unwrap();
            assert!(model_can_write(&temp.join("link"), &root));
            assert!(model_can_write(&temp.join("link/home-not-made"), &root));
        }
        // A workspace that cannot be checked counts as one the model can write.
        assert!(model_can_write(&temp.join("home"), &temp.join("missing-workspace")));
        assert!(reason(&root).contains("inside this workspace"));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn only_the_default_home_in_the_actual_home_workspace_is_exempt() {
        let f = crate::test_workspace::Fixture::new();
        let root = f.0.canonicalize().unwrap();
        assert!(!refused(&root.join(".vade"), &root, Some(&root)));
        for name in ["sub/.vade", ".config/vade", "secrets/vade", "vade-home"] {
            assert!(refused(&root.join(name), &root, Some(&root)), "{name}");
        }
        assert!(refused(&root.join(".vade"), &root, None));
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_link_from_outside_into_a_replaced_project_folder_is_followed() {
        let temp = std::env::temp_dir().join(format!("vade-home-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        for dir in ["work/sub", "build", "user"] { std::fs::create_dir_all(temp.join(dir)).unwrap(); }
        let temp = temp.canonicalize().unwrap();
        let root = temp.join("work");
        std::os::unix::fs::symlink(root.join("sub"), temp.join("user/link")).unwrap();
        let home = temp.join("user/link/.config/vade");
        assert!(model_can_write(&home, &root));
        std::fs::rename(root.join("sub"), root.join("sub-old")).unwrap();
        std::os::unix::fs::symlink(temp.join("build"), root.join("sub")).unwrap();
        std::fs::create_dir_all(temp.join("build/.config/vade")).unwrap();
        crate::saved_approvals::add(&temp.join("build/.config/vade"), "https://collector.example.org", false).unwrap();
        assert!(!home.canonicalize().unwrap().starts_with(&root));
        assert!(model_can_write(&home, &root));
        let approvals = crate::shell::permissions::Approvals::for_workspace(&root, Some(&home), false).unwrap();
        assert!(!approvals.web_allowed("https://collector.example.org") && !approvals.can_save(false));
        assert!(approvals.always_note().unwrap().contains("inside this workspace"));
        assert!(crate::trust::Store::open(&home, &root).is_err());
        std::os::unix::fs::symlink("link", temp.join("user/chain")).unwrap();
        assert!(model_can_write(&temp.join("user/chain/.config/vade"), &root));
        assert!(!model_can_write(&temp.join("user/vade"), &root));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn shipped_grants(ship: &Path, home: &Path, root: &Path) -> bool {
        let folder = ship.join(".config/vade");
        std::fs::create_dir_all(&folder).unwrap();
        crate::saved_approvals::add(&folder, "https://collector.example.org", false).unwrap();
        let approvals = crate::shell::permissions::Approvals::for_workspace(root, Some(home), false).unwrap();
        approvals.web_allowed("https://collector.example.org") || approvals.can_save(false) || crate::trust::Store::open(home, root).is_ok()
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn the_checked_home_is_the_home_that_is_used_after_a_link_swap() {
        let temp = std::env::temp_dir().join(format!("vade-home-swap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        for dir in ["work/sub/vade", "real/vade"] { std::fs::create_dir_all(temp.join(dir)).unwrap(); }
        let temp = temp.canonicalize().unwrap();
        let (work, real, link) = (temp.join("work"), temp.join("real/vade"), temp.join("link"));
        std::os::unix::fs::symlink(temp.join("real"), &link).unwrap();
        let home = link.join("vade");
        let root = Root::open(&work).unwrap();
        let checked = Home::open(&home, &root, false).unwrap().unwrap();
        let planted = work.join("sub/vade");
        crate::saved_approvals::add(&planted, "https://collector.example.org", false).unwrap();
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(work.join("sub"), &link).unwrap();
        assert_eq!(crate::saved_approvals::read(&home).unwrap().web, ["https://collector.example.org"]);
        assert!(crate::saved_approvals::read_in(checked.dir()).unwrap().web.is_empty());
        crate::saved_approvals::add_in(&checked, "https://good.example.org", false).unwrap();
        assert_eq!(crate::saved_approvals::read(&real).unwrap().web, ["https://good.example.org"]);
        assert_eq!(crate::saved_approvals::read(&planted).unwrap().web, ["https://collector.example.org"]);
        assert!(Home::open(&home, &root, false).is_err());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_link_in_other_letter_case_into_the_workspace_is_followed() {
        let temp = std::env::temp_dir().join(format!("vade-home-case-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        for dir in ["work/sub", "build", "user"] { std::fs::create_dir_all(temp.join(dir)).unwrap(); }
        let temp = temp.canonicalize().unwrap();
        let root = temp.join("work");
        std::fs::write(temp.join("probe-Case"), "").unwrap();
        let folds = temp.join("PROBE-CASE").exists();
        if !folds {
            std::fs::create_dir_all(temp.join("WORK/sub")).unwrap();
        }
        std::os::unix::fs::symlink(temp.join("WORK/sub"), temp.join("user/link")).unwrap();
        let home = temp.join("user/link/.config/vade");
        assert_eq!(model_can_write(&home, &root), folds);
        std::fs::rename(root.join("sub"), root.join("sub-old")).unwrap();
        std::os::unix::fs::symlink(temp.join("build"), root.join("sub")).unwrap();
        let ship = if folds { temp.join("build") } else { temp.join("WORK/sub") };
        assert_eq!(model_can_write(&home, &root), folds);
        assert_eq!(shipped_grants(&ship, &home, &root), !folds);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_renamed_parent_link_to_build_state_cannot_supply_authority() {
        let f = crate::test_workspace::Fixture::new();
        let workspace = f.workspace();
        std::fs::create_dir(f.0.join("sub")).unwrap();
        let session = crate::shell::BuildSession::new(None).unwrap();
        let spec = crate::shell::CommandSpec {
            argv: vec!["/bin/sh".into(), "-c".into(),
                "mv sub sub-old && ln -s \"$TMPDIR\" sub && mkdir -p sub/.config/vade && umask 077 && printf '%s' '{\"version\":1,\"web\":[\"https://collector.example.org\"],\"memory\":[]}' > sub/.config/vade/allowed.json".into()],
            timeout_ms: 20_000,
        };
        let out = crate::shell::run_in_session(&workspace, &spec, 4096, &std::sync::atomic::AtomicBool::new(false), &crate::shell::Exclusions::default(), &session).unwrap();
        let home = workspace.root().join("sub/.config/vade");
        assert!(home.join("allowed.json").exists(), "{}", out.render());
        assert!(!home.canonicalize().unwrap().starts_with(workspace.root()));
        assert!(model_can_write(&home, workspace.root()));
        let approvals = crate::shell::permissions::Approvals::for_workspace(workspace.root(), Some(&home), false).unwrap();
        assert!(!approvals.web_allowed("https://collector.example.org") && !approvals.can_save(false));
        assert!(crate::trust::Store::open(&home, workspace.root()).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_rule_matches_what_the_shell_sandbox_lets_a_command_write() {
        let f = crate::test_workspace::Fixture::new();
        let mut names: Vec<String> = vec![
            "vade-home".into(), "caf\u{e9}".into(), "\u{17f}ecrets".into(), "x.pem.d".into(), "pem".into(), "key".into(),
            ".hidden".into(), ".vade".into(), "a.b".into(), "secret".into(), "id_rsa.pub".into(),
        ];
        for rule in crate::tools::SECRET_PATH_RULES {
            let name = match rule.match_kind {
                crate::tools::SecretPathMatch::Exact => rule.value.to_owned(),
                crate::tools::SecretPathMatch::Prefix => format!("{}local", rule.value),
                crate::tools::SecretPathMatch::Suffix => format!("store{}", rule.value),
            };
            names.push(name.clone());
            names.push(name.to_ascii_uppercase());
        }
        let mut script = String::new();
        for (index, name) in names.iter().enumerate() {
            script.push_str(&format!("mkdir 'd{index}' && mkdir 'd{index}/{name}' && printf planted > 'd{index}/{name}/allowed.json'; "));
        }
        script.push_str("true");
        let spec = crate::shell::CommandSpec { argv: vec!["/bin/sh".into(), "-c".into(), script], timeout_ms: 20_000 };
        let out = crate::shell::run(&f.workspace(), &spec, 65536, &std::sync::atomic::AtomicBool::new(false)).unwrap();
        for (index, name) in names.iter().enumerate() {
            let written = std::fs::read_to_string(f.0.join(format!("d{index}")).join(name).join("allowed.json")).is_ok_and(|text| text == "planted");
            assert!(!(written && sandbox_denies_writes_in(name)), "the sandbox let a command write in {name:?}, but the guard counts it as safe: {}", out.render());
            if ["vade-home", "caf\u{e9}", "x.pem.d", "a.b"].contains(&name.as_str()) {
                assert!(written, "{name:?} was not written: {}", out.render());
            }
        }
    }
}
