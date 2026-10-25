//! Where private chat state lives: native session journals, session metadata
//! and input recall. The default keeps it out of the project tree, under the
//! Vade home (`VADE_HOME`, else `~/.vade`, the `vade` launcher's root) in
//! `projects/<name>-<hash>`, so git never sees a conversation. The explicit
//! `--state-in-project` opt-in keeps the legacy `<workspace>/.vade` folder.
use super::error;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Written once into every state directory, never over an existing file, so git
/// ignores chat state even when the folder sits inside a repository. `check`
/// stays visible: `.vade/check` is a user-written project setting, not state.
pub(crate) const GITIGNORE: &[u8] = b"# Private Vade chat state; git ignores it.\n*\n!check\n";

/// Where a workspace keeps its private chat state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateLocation {
    /// `<vade home>/projects/<key>`, outside the project tree (the default).
    Home(PathBuf),
    /// Legacy `<workspace>/.vade`, an explicit opt-in.
    Project,
}

impl StateLocation {
    /// `VADE_HOME` when set, otherwise `$HOME/.vade`: the launcher's root.
    pub fn from_env() -> io::Result<Self> {
        Self::from_vars(std::env::var_os("VADE_HOME"), std::env::var_os("HOME"))
    }
    pub fn from_vars(vade_home: Option<OsString>, home: Option<OsString>) -> io::Result<Self> {
        match (vade_home.filter(|v| !v.is_empty()), home.filter(|v| !v.is_empty())) {
            (Some(path), _) => Self::home(PathBuf::from(path)),
            (None, Some(home)) => Self::home(PathBuf::from(home).join(".vade")),
            (None, None) => Err(error(
                "cannot find the Vade home for chat state; set VADE_HOME or HOME, pass --vade-home DIR, or opt in with --state-in-project",
            )),
        }
    }
    pub fn home(path: PathBuf) -> io::Result<Self> {
        if !path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(error("the Vade home for chat state must be an absolute path without '..'"));
        }
        Ok(Self::Home(path))
    }
    /// The state directory for a canonical workspace root.
    pub fn path(&self, root: &Path) -> PathBuf {
        match self {
            Self::Home(home) => home.join("projects").join(project_key(root)),
            Self::Project => root.join(".vade"),
        }
    }
}

/// A readable folder-name prefix plus a stable 64-bit FNV-1a hash of the
/// canonical workspace path. A collision is refused by the `workspace` record.
pub fn project_key(root: &Path) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in root.as_os_str().as_encoded_bytes() {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let name: String = root
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default()
        .chars()
        .take(32)
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    format!("{}-{hash:016x}", if name.is_empty() { "root" } else { &name })
}

/// Opens (creating privately) the per-project state directory under a Vade home.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(super) fn open_home(home: &Path, root: &Path) -> io::Result<super::anchored::Dir> {
    use std::io::Read;
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(home)
        .map_err(|_| error("cannot create the Vade home for chat state"))?;
    let dir = super::anchored::Dir::open_root(home)
        .map_err(|_| error("cannot safely open the Vade home for chat state"))?
        .private_child("projects")?
        .private_child(&project_key(root))?;
    dir.ignore_in_git()?;
    let mut owner = root.as_os_str().as_encoded_bytes().to_vec();
    owner.push(b'\n');
    match dir.optional_file("workspace")? {
        Some(file) => {
            let mut recorded = Vec::new();
            file.take(owner.len() as u64 + 1).read_to_end(&mut recorded)?;
            if recorded != owner {
                return Err(error("chat state folder belongs to another project path"));
            }
        }
        None => dir.write_atomic("workspace", &owner)?,
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_follows_vade_home_then_home() {
        let some = |s: &str| Some(OsString::from(s));
        assert_eq!(
            StateLocation::from_vars(some("/private/vade"), some("/srv/a")).unwrap(),
            StateLocation::Home("/private/vade".into())
        );
        assert_eq!(
            StateLocation::from_vars(None, some("/srv/a")).unwrap(),
            StateLocation::Home("/srv/a/.vade".into())
        );
        assert_eq!(
            StateLocation::from_vars(some(""), some("/srv/a")).unwrap(),
            StateLocation::Home("/srv/a/.vade".into())
        );
        assert!(StateLocation::from_vars(None, None).is_err());
        assert!(StateLocation::from_vars(some("relative"), None).is_err());
        assert!(StateLocation::from_vars(some("/a/../b"), None).is_err());
    }

    #[test]
    fn project_keys_are_stable_readable_and_path_specific() {
        let key = project_key(Path::new("/srv/a/my project.rs"));
        assert!(key.starts_with("my_project_rs-"), "{key}");
        assert_eq!(key, project_key(Path::new("/srv/a/my project.rs")));
        assert_ne!(key, project_key(Path::new("/srv/b/my project.rs")));
        assert!(project_key(Path::new("/")).starts_with("root-"));
        // FNV-1a is fixed here so existing folders survive toolchain updates.
        assert_eq!(project_key(Path::new("")), "root-cbf29ce484222325");
        let home = StateLocation::Home("/v".into());
        assert_eq!(home.path(Path::new("/w/p")), Path::new("/v/projects").join(project_key(Path::new("/w/p"))));
        assert_eq!(StateLocation::Project.path(Path::new("/w/p")), Path::new("/w/p/.vade"));
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn home_state_stays_out_of_the_project_and_refuses_another_owner() {
        use std::fs;
        let (project, home) = (crate::test_workspace::Fixture::new(), crate::test_workspace::Fixture::new());
        let location = StateLocation::home(home.0.join("vade")).unwrap();
        let workspace = super::super::Workspace::open(&project.0, location.clone()).unwrap();
        assert!(!home.0.join("vade").exists(), "state is created on first use");
        let mut session = crate::session::Session::new(&workspace, Some("chat")).unwrap();
        session.complete("private prompt".into(), "private answer".into()).unwrap();
        drop(session);
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 0, "nothing lands in the project");
        let state = workspace.state_path();
        assert_eq!(state, home.0.join("vade/projects").join(project_key(workspace.root())));
        assert!(state.join("sessions/chat.state").is_file());
        assert_eq!(fs::read(state.join(".gitignore")).unwrap(), GITIGNORE);
        let mut owner = workspace.root().as_os_str().as_encoded_bytes().to_vec();
        owner.push(b'\n');
        assert_eq!(fs::read(state.join("workspace")).unwrap(), owner);
        // A hash collision (simulated by a foreign record) is refused, not shared.
        fs::write(state.join("workspace"), "/another/project\n").unwrap();
        assert!(crate::session::Session::new(&workspace, Some("other")).is_err());
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn project_opt_in_ignores_state_in_git_and_keeps_a_user_gitignore() {
        use std::fs;
        let project = crate::test_workspace::Fixture::new();
        let workspace = project.workspace();
        crate::session::Session::new(&workspace, Some("chat")).unwrap().complete("a".into(), "b".into()).unwrap();
        assert_eq!(workspace.state_path(), workspace.root().join(".vade"));
        assert!(project.0.join(".vade/sessions/chat.state").is_file());
        assert_eq!(fs::read(project.0.join(".vade/.gitignore")).unwrap(), GITIGNORE);
        fs::write(project.0.join(".vade/.gitignore"), "custom\n").unwrap();
        drop(crate::session::Session::new(&workspace, Some("again")).unwrap());
        assert_eq!(fs::read_to_string(project.0.join(".vade/.gitignore")).unwrap(), "custom\n");
    }
}
