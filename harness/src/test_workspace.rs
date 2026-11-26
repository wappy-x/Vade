//! Temporary local fixtures only. No provider, engine, or external network.
use crate::tools::Workspace;
use std::path::PathBuf;
pub(crate) struct Fixture(pub PathBuf);
impl Fixture {
    pub fn new() -> Self {
        // macOS sandboxed Cargo must be able to resolve every parent of cwd.
        // A workspace under a developer's home fails that check. Use the fixed
        // scratch parent, not the process's configurable temporary directory.
        #[cfg(target_os = "macos")]
        let root = PathBuf::from("/private/tmp");
        #[cfg(not(target_os = "macos"))]
        let root = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"))
            .join("test-workspaces");
        Self::under(root)
    }
    /// A folder that no source checkout holds, for a test of "no parent holds
    /// `.git`". `new` cannot serve it on Linux: its folder is under `target`,
    /// inside the checkout, so a walk up from it finds the checkout's `.git`.
    pub fn outside_checkout() -> Self {
        #[cfg(target_os = "macos")]
        let root = PathBuf::from("/private/tmp");
        #[cfg(not(target_os = "macos"))]
        let root = std::env::temp_dir();
        Self::under(root)
    }
    fn under(root: PathBuf) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join(format!(
            "vade-coding-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    pub fn workspace(&self) -> Workspace {
        Workspace::new(&self.0).unwrap()
    }
    pub fn write(&self, path: &str, text: &str) {
        std::fs::write(self.0.join(path), text).unwrap();
    }
    /// A file that only the owner can read and write, as a private setting needs.
    #[cfg(unix)]
    pub fn write_private(&self, path: &str, text: &str) {
        use std::os::unix::fs::PermissionsExt;
        self.write(path, text);
        std::fs::set_permissions(self.0.join(path), std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    /// A Vade home for trust records, beside this folder: a home inside the
    /// workspace holds no trusted record (`home_guard`).
    pub fn home(&self) -> PathBuf {
        let mut home = self.0.clone().into_os_string();
        home.push("-home");
        PathBuf::from(home)
    }
    /// A private file in `.vade`, trusted as it is: what `vade trust` records.
    #[cfg(unix)]
    pub fn write_trusted(&self, name: &str, text: &str) {
        self.write_private(&format!(".vade/{name}"), text);
        let root = self.0.canonicalize().unwrap();
        crate::trust::Store::open(&self.home(), &root).unwrap().set(&root, name, Some(&crate::trust::digest(text.as_bytes()))).unwrap();
    }
    pub fn read(&self, path: &str) -> String {
        std::fs::read_to_string(self.0.join(path)).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.home());
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
