//! Temporary local fixtures only. No provider, engine, or external network.
use crate::tools::Workspace;
use std::path::PathBuf;
pub(crate) struct Fixture(pub PathBuf);
impl Fixture {
    pub fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
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
