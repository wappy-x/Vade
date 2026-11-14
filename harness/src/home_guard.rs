//! Whether a Vade home can hold the person's own word. Saved approvals and
//! trust records in the Vade home let things run with no prompt, so only the
//! person may write them. Model tools can write files inside the workspace:
//! the file tools (apply_patch, workspace_edit_lines) and, more widely, a
//! sandboxed shell command. A Vade home there (`VADE_HOME`, `--vade-home` or
//! `--mcp-home` set to the workspace or a folder in it) can therefore hold a
//! planted file, and nothing in it counts.
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Whether the shell sandbox denies writes at and under a workspace folder
/// or file of this name: a hidden (dot) name, a credential name or a name
/// with a credential suffix, in any ASCII case. This is the write rule of
/// the sandbox profile (`shell/execution.rs`: the dot pattern and the
/// patterns built from `tools::CREDENTIAL_NAMES` and `CREDENTIAL_SUFFIXES`).
/// A macOS test runs real sandboxed writes for each such name and for others,
/// so this function and the profile cannot drift apart: each name it calls
/// denied must be denied by the sandbox. The sandbox may deny more (macOS
/// folds some non-ASCII letters, such as `ſ` to `s`); that only makes the
/// guard wider. Paths excluded with
/// `--shell-exclude` are left out: they only add denials, so leaving them
/// out makes the guard wider, never narrower.
pub fn sandbox_denies_writes_in(name: &str) -> bool {
    let bytes = name.as_bytes();
    let ends_with = |suffix: &str| bytes.len() >= suffix.len() && bytes[bytes.len() - suffix.len()..].eq_ignore_ascii_case(suffix.as_bytes());
    (name.starts_with('.') && name.len() > 1)
        || crate::tools::CREDENTIAL_NAMES.iter().any(|credential| credential.as_bytes().eq_ignore_ascii_case(bytes))
        || crate::tools::CREDENTIAL_SUFFIXES.iter().any(|suffix| ends_with(suffix))
}

/// Whether `home` must not supply authority for this workspace. A project
/// can ship files under any name, so every home inside it is refused. The
/// only exception is the default `.vade` when the workspace is HOME itself.
/// Check every existing ancestor too: a writable project parent can be
/// replaced by a link that points outside the workspace. A path that cannot
/// be checked is refused. Each link on the way is followed by hand too, so
/// a home outside reached through a link into the workspace is refused.
/// Shell and file-tool write rules remain a second check for the HOME
/// exception.
pub fn model_can_write(home: &Path, root: &Path) -> bool {
    let os_home = std::env::var_os("HOME").and_then(|home| Path::new(&home).canonicalize().ok());
    refused(home, root, os_home.as_deref())
}
fn refused(home: &Path, root: &Path, os_home: Option<&Path>) -> bool {
    let Ok(root) = root.canonicalize() else { return true };
    if !home.is_absolute() || home.components().any(|part| matches!(part, Component::ParentDir)) { return true; }
    if visits_workspace(home, &root, os_home) { return true; }
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
                let default = os_home == Some(root.as_path()) && real == root.join(".vade")
                    && (ancestor == root || ancestor == real);
                if !default { return true; }
                let Ok(relative) = real.strip_prefix(&root) else { return true };
                for part in relative.components() {
                    let Component::Normal(name) = part else { return true };
                    let Some(name) = name.to_str() else { return true };
                    if crate::tools::allowed_component(name) || !sandbox_denies_writes_in(name) { return true; }
                }
            }
        }
        if let Some(name) = at.file_name() { rest.push(name.to_owned()); }
    }
    // A normal outside home has no workspace ancestor. An unchecked path
    // cannot establish that boundary.
    !found || (inside && os_home != Some(root.as_path()))
}

/// Whether the way to `home` passes through the workspace: a link outside
/// it can name a project folder that a sandboxed command later replaced
/// with a link to its own build folder, where it can write hidden names.
/// `canonicalize` gives only the end of the way, so each link is followed
/// here one step at a time and every place on the way is checked. Inside
/// `root` only the default `.vade` of a HOME workspace may be passed.
/// Too many links, or a link that cannot be read, count as passing.
///
/// Path text is not enough to say "this place is the workspace": one
/// folder can have other names (the macOS `/System/Volumes/Data` name of
/// the data volume, or other letter case on a case-folding disk). So each
/// place on the way is also compared with the root by identity (device and
/// inode). A place that IS the root means the way goes through it.
///
/// A name can also enter a folder below the root without passing the root
/// on its way (macOS `/.vol/<device>/<inode>`). So the walk ends with the
/// physical parents of the deepest real folder on the way: `..` of that
/// folder, then `..` of that one, up to the top, each compared with the root
/// by identity. A hard-linked folder or a Linux bind mount of a folder
/// below the root has its own `..`, so it is not seen.
fn visits_workspace(home: &Path, root: &Path, os_home: Option<&Path>) -> bool {
    let exempt = |at: &Path| os_home == Some(root) && (at == root || at == root.join(".vade"));
    let Ok(root_meta) = std::fs::metadata(root) else { return true };
    let root_id = identity(&root_meta);
    // The names still to walk, the next one last.
    let mut todo: Vec<OsString> = home.components().rev().map(|part| part.as_os_str().to_owned()).collect();
    let (mut at, mut links, mut deepest) = (PathBuf::new(), 0, None);
    while let Some(part) = todo.pop() {
        match Path::new(&part).components().next() {
            Some(Component::ParentDir) => { at.pop(); }
            Some(Component::CurDir) | None => {}
            Some(_) => at.push(&part),
        }
        if at.starts_with(root) && !exempt(&at) { return true; }
        // Every place before `at` is a real folder here, so `at` names
        // the entry itself. A place that does not exist ends no way.
        let Ok(meta) = std::fs::symlink_metadata(&at) else { continue };
        // The root under any name. Only the HOME workspace may go on, and
        // only to its `.vade` as the last name.
        if root_id.is_some() && identity(&meta) == root_id
            && !(os_home == Some(root) && todo.len() == 1 && todo[0] == ".vade") {
            return true;
        }
        if meta.file_type().is_symlink() {
            links += 1;
            let Ok(target) = std::fs::read_link(&at) else { return true };
            if links > 40 { return true; }
            at.pop();
            todo.extend(target.components().rev().map(|part| part.as_os_str().to_owned()));
        } else if meta.is_dir() {
            deepest = Some(at.clone());
        }
    }
    let Some(deepest) = deepest else { return true };
    // The default `.vade` of a HOME workspace: its real folder (one level
    // below the root) or, before it is made, the root itself.
    let vade = (os_home == Some(root) && home == root.join(".vade")).then(|| std::fs::symlink_metadata(root.join(".vade")));
    let may_meet_root = |level: usize, start: &std::fs::Metadata| match (&vade, level) {
        (Some(Err(_)), 0) => true,
        (Some(Ok(meta)), 1) => meta.is_dir() && identity(meta) == identity(start),
        _ => false,
    };
    physical_parents_meet(&deepest, root_id, may_meet_root)
}

/// Whether `start` or one of its physical parents (`..` of the real folder,
/// not the text before its last name) is the entry `root_id`, other than
/// where `may_meet_root` allows it. Gives up, and counts as meeting, after
/// 256 levels or when a level cannot be read.
fn physical_parents_meet(start: &Path, root_id: Option<(u64, u64)>, may_meet_root: impl Fn(usize, &std::fs::Metadata) -> bool) -> bool {
    if root_id.is_none() { return false; }
    let Ok(first) = std::fs::metadata(start) else { return true };
    let (mut at, mut meta) = (start.to_path_buf(), first.clone());
    for level in 0..256 {
        if identity(&meta) == root_id { return !may_meet_root(level, &first); }
        at.push("..");
        let Ok(parent) = std::fs::metadata(&at) else { return true };
        // The top is its own parent.
        if identity(&parent) == identity(&meta) { return false; }
        meta = parent;
    }
    true
}

/// The workspace root, held open for the session: the identity that a
/// Vade home is compared with does not change when a name does.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub struct Root {
    dir: crate::tools::anchored::Dir,
    path: PathBuf,
}
#[cfg(any(target_os = "macos", target_os = "linux"))]
impl Root {
    pub fn open(root: &Path) -> Result<Self, String> {
        let path = root.canonicalize().map_err(|_| "the workspace folder cannot be checked".to_owned())?;
        let dir = crate::tools::anchored::Dir::open_root(&path).map_err(|_| "the workspace folder cannot be opened".to_owned())?;
        Ok(Self { dir, path })
    }
    /// The canonical workspace path, as trust records name it.
    pub fn path(&self) -> &Path { &self.path }
}

/// A Vade home, opened once and checked through that handle. Every read
/// and write of `allowed.json`, `trust.json` and the user `permissions.json`
/// goes through `dir` by a plain name, so the folder that was checked is
/// the folder that is used: a name changed after the check changes nothing.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub struct Home {
    dir: crate::tools::anchored::Dir,
    path: PathBuf,
}
#[cfg(any(target_os = "macos", target_os = "linux"))]
impl Home {
    /// Open `home` for the workspace `root` and check it. `create` makes the
    /// folder (mode 0700) when it is missing; without it a missing home is
    /// `None` (it holds nothing). The path text checks run first
    /// (`model_can_write`). Then the kernel's own result, the opened folder,
    /// is walked up by `..` relative to the handle and compared with the
    /// root by device and inode. Any error refuses.
    pub fn open(home: &Path, root: &Root, create: bool) -> Result<Option<Self>, String> {
        let os_home = std::env::var_os("HOME").and_then(|home| Path::new(&home).canonicalize().ok());
        Self::open_for(home, root, create, os_home.as_deref())
    }
    fn open_for(home: &Path, root: &Root, create: bool, os_home: Option<&Path>) -> Result<Option<Self>, String> {
        if refused(home, &root.path, os_home) { return Err(reason(home)); }
        if std::fs::symlink_metadata(home).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) {
            if !create { return Ok(None); }
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(home).map_err(|error| format!("cannot create the Vade home: {error}"))?;
        }
        let dir = crate::tools::anchored::Dir::open_root(home).map_err(|_| "the Vade home must be a real folder, not a link".to_owned())?;
        // Only the default `.vade` of a HOME workspace may sit right below it.
        let exempt = os_home == Some(root.path.as_path()) && home == root.path.join(".vade");
        if handle_meets_root(&dir, &root.dir, exempt) { return Err(reason(home)); }
        Ok(Some(Self { dir, path: home.to_path_buf() }))
    }
    pub(crate) fn dir(&self) -> &crate::tools::anchored::Dir { &self.dir }
    pub fn path(&self) -> &Path { &self.path }
}

/// Other systems have no opened-folder checks, so no Vade home is used.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub struct Root { path: PathBuf }
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
impl Root {
    pub fn open(root: &Path) -> Result<Self, String> { Ok(Self { path: root.to_path_buf() }) }
    pub fn path(&self) -> &Path { &self.path }
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub struct Home { path: PathBuf }
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
impl Home {
    pub fn open(_: &Path, _: &Root, _: bool) -> Result<Option<Self>, String> { Err("a Vade home needs macOS or Linux".into()) }
    pub fn path(&self) -> &Path { &self.path }
}

/// Whether the opened folder `dir`, or one of its physical parents, is the
/// workspace root (by device and inode), other than the one exempt level.
/// 256 levels, or any error, count as meeting it.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn handle_meets_root(dir: &crate::tools::anchored::Dir, root: &crate::tools::anchored::Dir, exempt: bool) -> bool {
    let (Ok(root_id), Ok(mut at)) = (root.identity(), dir.duplicate()) else { return true };
    let Ok(mut id) = at.identity() else { return true };
    for level in 0..256 {
        if id == root_id { return !(exempt && level == 1); }
        let Ok(parent) = at.parent() else { return true };
        let Ok(parent_id) = parent.identity() else { return true };
        // The top is its own parent.
        if parent_id == id { return false; }
        (at, id) = (parent, parent_id);
    }
    true
}

/// The device and inode of an entry, where the OS gives them.
fn identity(meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    #[cfg(unix)]
    { use std::os::unix::fs::MetadataExt; Some((meta.dev(), meta.ino())) }
    #[cfg(not(unix))]
    { let _ = meta; None }
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
        // A shell command writes where the file tools do not: a non-ASCII
        // name, or a name in another case than a credential name.
        assert!(model_can_write(&root.join("caf\u{e9}/vade"), &root));
        assert!(model_can_write(&root.join("\u{17f}ecrets/vade"), &root));
        assert!(model_can_write(&root.join("keys.pem.d/vade"), &root));
        // A project can ship hidden and credential-named folders too.
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

    /// A person's own link outside the workspace points to a project
    /// folder. A command renames that folder and links its name to a place
    /// it can write, so the home now resolves outside the workspace.
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
        // A relative link in a chain of links is followed the same way.
        std::os::unix::fs::symlink("link", temp.join("user/chain")).unwrap();
        assert!(model_can_write(&temp.join("user/chain/.config/vade"), &root));
        // A home outside with no way through the workspace is still used.
        assert!(!model_can_write(&temp.join("user/vade"), &root));
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// A private allowed.json shipped under `.config/vade` of the folder
    /// `ship` names; whether a chat at `root` with that home trusts it.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn shipped_grants(ship: &Path, home: &Path, root: &Path) -> bool {
        planted_grants(&ship.join(".config/vade"), home, root)
    }
    /// A private allowed.json planted in `folder`; whether a chat at `root`
    /// with the home `home` trusts it, or would write in it.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn planted_grants(folder: &Path, home: &Path, root: &Path) -> bool {
        std::fs::create_dir_all(folder).unwrap();
        crate::saved_approvals::add(folder, "https://collector.example.org", false).unwrap();
        let approvals = crate::shell::permissions::Approvals::for_workspace(root, Some(home), false).unwrap();
        approvals.web_allowed("https://collector.example.org") || approvals.can_save(false) || crate::trust::Store::open(home, root).is_ok()
    }

    /// macOS gives the data volume a second name. The same folder written
    /// through it is still the workspace, for the home and for the root.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_workspace_under_its_other_macos_name_is_still_the_workspace() {
        let f = crate::test_workspace::Fixture::new();
        let root = f.0.canonicalize().unwrap();
        let alias = Path::new("/System/Volumes/Data").join(root.strip_prefix("/").unwrap());
        if !alias.exists() {
            eprintln!("skipped: {} does not exist on this system", alias.display());
            return;
        }
        assert!(alias.canonicalize().unwrap() != root, "the other name stays as written");
        let home = alias.join(".config/vade");
        assert!(model_can_write(&home, &root));
        assert!(!shipped_grants(&root, &home, &root));
        // The reverse: the workspace given by the other name.
        assert!(model_can_write(&root.join(".config/vade"), &alias));
        assert!(!shipped_grants(&root, &root.join(".config/vade"), &alias));
    }

    /// macOS names any folder by device and inode under `/.vol`. Such a
    /// name enters a folder below the root without passing the root.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_home_named_by_device_and_inode_is_still_in_the_workspace() {
        use std::os::unix::fs::MetadataExt;
        let f = crate::test_workspace::Fixture::new();
        let root = f.0.canonicalize().unwrap();
        std::fs::create_dir_all(root.join(".config")).unwrap();
        let meta = std::fs::metadata(root.join(".config")).unwrap();
        let folder = PathBuf::from(format!("/.vol/{}/{}", meta.dev(), meta.ino()));
        if std::fs::metadata(&folder).map(|named| named.ino()).ok() != Some(meta.ino()) {
            eprintln!("skipped: {} does not name the folder on this system", folder.display());
            return;
        }
        let home = folder.join("vade");
        // Before the home folder exists, and with shipped files in it.
        assert!(model_can_write(&home, &root));
        assert!(!shipped_grants(&root, &home, &root));
        assert!(model_can_write(&home, &root));
        // The workspace named the same way, and a home by its plain name.
        let root_meta = std::fs::metadata(&root).unwrap();
        let named_root = PathBuf::from(format!("/.vol/{}/{}", root_meta.dev(), root_meta.ino()));
        assert!(model_can_write(&root.join(".config/vade"), &named_root));
        // A folder outside the workspace named this way is still used.
        let outside = crate::test_workspace::Fixture::new();
        let out_meta = std::fs::metadata(&outside.0).unwrap();
        assert!(!model_can_write(&PathBuf::from(format!("/.vol/{}/{}/vade", out_meta.dev(), out_meta.ino())), &root));
    }

    /// Review 121d: a `/.vol` name of a folder below the root, then a link
    /// with `..`. The kernel resolves `..` by the physical parent and lands
    /// in the workspace; a walk of the text lands outside. The check is on
    /// the folder the kernel opened.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_device_and_inode_name_with_a_dotdot_link_is_checked_where_the_kernel_lands() {
        use std::os::unix::fs::MetadataExt;
        let f = crate::test_workspace::Fixture::new();
        let root = f.0.canonicalize().unwrap();
        for dir in ["a/b/c", "out/vade"] { std::fs::create_dir_all(root.join(dir)).unwrap(); }
        std::os::unix::fs::symlink("../../../out", root.join("a/b/c/l")).unwrap();
        std::os::unix::fs::symlink("../../..", root.join("a/b/c/up")).unwrap();
        let meta = std::fs::metadata(root.join("a/b/c")).unwrap();
        let base = PathBuf::from(format!("/.vol/{}/{}", meta.dev(), meta.ino()));
        if std::fs::metadata(&base).map(|named| named.ino()).ok() != Some(meta.ino()) {
            eprintln!("skipped: {} does not name the folder on this system", base.display());
            return;
        }
        for (home, lands) in [(base.join("l/vade"), root.join("out/vade")), (base.join("up/.config/vade"), root.join(".config/vade")), (base.join("up/out/vade"), root.join("out/vade"))] {
            assert!(!planted_grants(&lands, &home, &root), "{}", home.display());
            // The kernel does land in the workspace for this name.
            assert_eq!(std::fs::metadata(&home).unwrap().ino(), std::fs::metadata(&lands).unwrap().ino());
        }
    }

    /// Check the folder that was opened, then use that folder: a link on
    /// the home's way that is swapped into the workspace after the check
    /// changes nothing for reads and writes through the checked handle. A
    /// second lookup by the path (as before this change) gets the planted file.
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
        // After the check: a planted file in the workspace, and the link swapped there.
        let planted = work.join("sub/vade");
        crate::saved_approvals::add(&planted, "https://collector.example.org", false).unwrap();
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(work.join("sub"), &link).unwrap();
        assert_eq!(crate::saved_approvals::read(&home).unwrap().web, ["https://collector.example.org"], "a lookup by path now gets the planted file");
        // Through the checked handle: the checked folder, for reads and writes.
        assert!(crate::saved_approvals::read_in(checked.dir()).unwrap().web.is_empty());
        crate::saved_approvals::add_in(&checked, "https://good.example.org", false).unwrap();
        assert_eq!(crate::saved_approvals::read(&real).unwrap().web, ["https://good.example.org"]);
        assert_eq!(crate::saved_approvals::read(&planted).unwrap().web, ["https://collector.example.org"]);
        // A new open finds the swap and refuses.
        assert!(Home::open(&home, &root, false).is_err());
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// A person's link outside names a workspace folder in other letter
    /// case. After a command swaps that folder for a link to its build
    /// folder, the case-folding disk still leads the way through the root.
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
            // On a case-sensitive disk the other case is another folder,
            // outside the workspace, and a home there is used.
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

    /// The rule above against the real sandbox: a command tries to make a
    /// folder of each name and write a file in it. Each name that the rule
    /// calls denied must stay unwritten; plain names must be written, so the
    /// command did run.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_rule_matches_what_the_shell_sandbox_lets_a_command_write() {
        let f = crate::test_workspace::Fixture::new();
        let mut names: Vec<String> = vec![
            "vade-home".into(), "caf\u{e9}".into(), "\u{17f}ecrets".into(), "x.pem.d".into(), "pem".into(), "key".into(),
            ".hidden".into(), ".vade".into(), "a.b".into(), "secret".into(), "id_rsa.pub".into(),
        ];
        for credential in crate::tools::CREDENTIAL_NAMES {
            names.push((*credential).into());
            names.push(credential.to_ascii_uppercase());
        }
        for suffix in crate::tools::CREDENTIAL_SUFFIXES {
            names.push(format!("store{suffix}"));
            names.push(format!("Store{}", suffix.to_ascii_uppercase()));
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
