//! Fresh build state, separate from uninspected workspace artifacts. Never adopt
//! an existing directory or reuse a path obtained from a transcript/model.
use std::io;
use std::path::Path;
#[cfg(target_os = "macos")]
use std::path::PathBuf;

/// Private per-session build storage (Cargo target, home and temporary
/// directories) shared by the commands one session runs.
#[derive(Default)]
pub struct BuildSession {
    #[cfg(target_os = "macos")]
    pub(super) toolchain: Option<PathBuf>,
    #[cfg(target_os = "macos")]
    directory: std::sync::Mutex<Option<Directory>>,
}

impl BuildSession {
    /// An owner-selected Rust distribution, not a rustup shim or Cargo home.
    /// Only its bin/lib trees become readable; nothing is writable there.
    pub fn new(toolchain: Option<&Path>) -> io::Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let toolchain = toolchain.map(validate_toolchain).transpose()?;
            Ok(Self {
                toolchain,
                directory: std::sync::Mutex::new(None),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            if toolchain.is_some() {
                Err(io::Error::other(
                    "command sandbox is supported only on macOS",
                ))
            } else {
                Ok(Self::default())
            }
        }
    }

    pub fn description(&self) -> &'static str {
        #[cfg(target_os = "macos")]
        if self.toolchain.is_some() {
            return "private session build/cache/temp area; owner-selected Rust bin/lib read-only; Cargo offline";
        }
        "private session build/cache/temp area; system toolchains only; Cargo offline"
    }

    /// Named project checks need the owner-selected distribution rather than a
    /// rustup shim that might fetch a compiler or read a private Cargo home.
    pub fn has_toolchain(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.toolchain.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    #[cfg(target_os = "macos")]
    pub(super) fn prepare(
        &self,
        workspace: &Path,
    ) -> io::Result<std::sync::MutexGuard<'_, Option<Directory>>> {
        if self
            .toolchain
            .as_ref()
            .is_some_and(|p| p.starts_with(workspace) || workspace.starts_with(p))
        {
            return Err(io::Error::other(
                "Rust toolchain must be separate from the workspace",
            ));
        }
        let mut directory = self
            .directory
            .lock()
            .map_err(|_| io::Error::other("session build area unavailable; start a new session"))?;
        if directory.is_none() {
            *directory = Some(Directory::create(workspace)?);
        }
        let area = directory.as_mut().unwrap();
        area.verify()?;
        if area.limits_exceeded()? {
            return Err(io::Error::other(
                "session build area exceeded 1 GiB or 100,000 entries; start a new session",
            ));
        }
        Ok(directory)
    }
}

#[cfg(target_os = "macos")]
fn validate_toolchain(path: &Path) -> io::Result<PathBuf> {
    let invalid = || {
        io::Error::other(
            "--shell-toolchain needs an installed Rust distribution with real bin/cargo, bin/rustc and lib directories",
        )
    };
    let root = path.canonicalize().map_err(|_| invalid())?;
    let text = root.to_str().ok_or_else(invalid)?;
    if text.contains(':') || text.chars().any(char::is_control) {
        return Err(invalid());
    }
    for part in ["bin", "lib", "bin/cargo", "bin/rustc", "bin/rustdoc"] {
        let meta = root.join(part).symlink_metadata().map_err(|_| invalid())?;
        if meta.is_symlink()
            || if part.contains('/') {
                !meta.is_file()
            } else {
                !meta.is_dir()
            }
        {
            return Err(invalid());
        }
    }
    Ok(root)
}

#[cfg(target_os = "macos")]
pub(super) struct Directory {
    pub path: PathBuf,
    identity: (u64, u64),
    exhausted: bool,
}

#[cfg(target_os = "macos")]
impl Directory {
    fn create(workspace: &Path) -> io::Result<Self> {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = std::env::temp_dir().canonicalize()?;
        if parent.starts_with(workspace) {
            return Err(io::Error::other(
                "build temp parent must be outside the workspace",
            ));
        }
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for _ in 0..32 {
            let path = parent.join(format!(
                "vade-build-{}-{time}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => (),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err(io::Error::other("cannot create private session build area")),
            }
            let meta = path.symlink_metadata()?;
            let area = Self {
                path,
                identity: (meta.dev(), meta.ino()),
                exhausted: false,
            };
            area.verify()?;
            for name in [
                "cargo-target",
                "cargo-home",
                "tmp",
                "home",
                "npm-cache",
                "config",
                "cache",
            ] {
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(area.path.join(name))?;
            }
            for name in ["npm-user.rc", "npm-global.rc"] {
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(area.path.join(name))?;
            }
            return Ok(area);
        }
        Err(io::Error::other(
            "cannot allocate a fresh session build area",
        ))
    }

    pub fn verify(&self) -> io::Result<()> {
        use std::os::unix::fs::MetadataExt;
        unsafe extern "C" {
            fn geteuid() -> u32;
        }
        let meta = self.path.symlink_metadata()?;
        if !meta.is_dir()
            || meta.is_symlink()
            || (meta.dev(), meta.ino()) != self.identity
            || meta.uid() != unsafe { geteuid() }
            || meta.mode() & 0o777 != 0o700
        {
            return Err(io::Error::other(
                "session build area identity or permissions changed",
            ));
        }
        Ok(())
    }

    pub fn limits_exceeded(&mut self) -> io::Result<bool> {
        // A retained-state bound, checked between commands, not a filesystem
        // quota during execution. OS per-file limits and deadlines also apply.
        self.exhausted = self.exhausted || !within_limits(&self.path, 1024 * 1024 * 1024, 100_000)?;
        Ok(self.exhausted)
    }

    pub fn environment(&self, command: &mut std::process::Command, toolchain: Option<&Path>) {
        // Incremental artifacts remain private and subject to the session's
        // retained-state limits; cap parallel compilers to bound CPU pressure.
        let jobs = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(2, 8);
        let path = toolchain.map_or_else(
            || "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin".into(),
            |root| {
                format!(
                    "{}:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
                    root.join("bin").display()
                )
            },
        );
        command
            .env("PATH", path)
            .env("HOME", self.path.join("home"))
            .env("TMPDIR", self.path.join("tmp"))
            .env("CARGO_TARGET_DIR", self.path.join("cargo-target"))
            .env("CARGO_HOME", self.path.join("cargo-home"))
            .env("CARGO_NET_OFFLINE", "true")
            .env("CARGO_BUILD_JOBS", jobs.to_string())
            .env("CARGO_INCREMENTAL", "1")
            .env("npm_config_cache", self.path.join("npm-cache"))
            .env("npm_config_userconfig", self.path.join("npm-user.rc"))
            .env("npm_config_globalconfig", self.path.join("npm-global.rc"))
            .env("npm_config_offline", "true")
            .env("XDG_CACHE_HOME", self.path.join("cache"))
            .env("XDG_CONFIG_HOME", self.path.join("config"));
        if let Some(root) = toolchain {
            command
                .env("RUSTC", root.join("bin/rustc"))
                .env("RUSTDOC", root.join("bin/rustdoc"));
        }
    }
}

#[cfg(target_os = "macos")]
fn within_limits(root: &Path, max_bytes: u64, max_entries: usize) -> io::Result<bool> {
    let mut dirs = vec![root.to_path_buf()];
    let mut bytes = 0u64;
    let mut entries = 0;
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            entries += 1;
            if entries > max_entries {
                return Ok(false);
            }
            let meta = entry.path().symlink_metadata()?;
            if meta.is_dir() {
                dirs.push(entry.path());
            } else if meta.is_file() {
                bytes = bytes.saturating_add(meta.len());
                if bytes > max_bytes {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[cfg(target_os = "macos")]
impl Drop for Directory {
    fn drop(&mut self) {
        // Never follow a replaced root or adopt another session's artifacts.
        // remove_dir_all does not traverse symlink entries inside this tree.
        if self.verify().is_ok() {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::test_workspace::Fixture;
    use std::os::unix::fs::{MetadataExt, symlink};

    #[test]
    fn areas_are_private_separate_reused_only_in_session_and_removed_on_drop() {
        let workspace = Fixture::new();
        let first = BuildSession::default();
        assert!(first.directory.lock().unwrap().is_none()); // no files before an approved run
        let path = first
            .prepare(&workspace.0)
            .unwrap()
            .as_ref()
            .unwrap()
            .path
            .clone();
        assert!(!path.starts_with(&workspace.0));
        assert_eq!(path.symlink_metadata().unwrap().mode() & 0o777, 0o700);
        std::fs::write(path.join("cargo-target/cached"), "retained").unwrap();
        let reused = first.prepare(&workspace.0).unwrap();
        assert_eq!(reused.as_ref().unwrap().path, path);
        drop(reused);
        let second = BuildSession::default();
        let other = second
            .prepare(&workspace.0)
            .unwrap()
            .as_ref()
            .unwrap()
            .path
            .clone();
        assert_ne!(path, other);
        drop(first);
        assert!(!path.exists());
        assert!(other.is_dir());
        drop(second);
        assert!(!other.exists());
    }

    #[test]
    fn replaced_build_area_is_refused_and_cleanup_never_follows_it() {
        let workspace = Fixture::new();
        let foreign = Fixture::new();
        foreign.write("keep", "untouched");
        let session = BuildSession::default();
        let path = session
            .prepare(&workspace.0)
            .unwrap()
            .as_ref()
            .unwrap()
            .path
            .clone();
        let moved = path.with_extension("moved");
        std::fs::rename(&path, &moved).unwrap();
        symlink(&foreign.0, &path).unwrap();
        assert!(session.prepare(&workspace.0).is_err());
        drop(session);
        assert_eq!(foreign.read("keep"), "untouched");
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir_all(moved).unwrap();
    }

    #[test]
    fn retained_state_limits_count_files_and_entries_without_following_links() {
        let f = Fixture::new();
        let outside = Fixture::new();
        outside.write("large", "x".repeat(64).as_str());
        f.write("file", "1234");
        symlink(&outside.0, f.0.join("link")).unwrap();
        assert!(within_limits(&f.0, 4, 2).unwrap());
        assert!(!within_limits(&f.0, 3, 2).unwrap());
        assert!(!within_limits(&f.0, 4, 1).unwrap());
    }

    #[test]
    fn toolchain_is_explicit_narrow_and_refuses_shims_and_workspace_overlap() {
        let f = Fixture::new();
        assert!(BuildSession::new(Some(&f.0)).is_err());
        std::fs::create_dir(f.0.join("bin")).unwrap();
        std::fs::create_dir(f.0.join("lib")).unwrap();
        for name in ["cargo", "rustc", "rustdoc"] {
            f.write(&format!("bin/{name}"), "fixture");
        }
        let session = BuildSession::new(Some(&f.0)).unwrap();
        assert!(session.prepare(&f.0.canonicalize().unwrap()).is_err());
        std::fs::remove_file(f.0.join("bin/cargo")).unwrap();
        symlink("rustc", f.0.join("bin/cargo")).unwrap();
        assert!(BuildSession::new(Some(&f.0)).is_err());
    }
}
