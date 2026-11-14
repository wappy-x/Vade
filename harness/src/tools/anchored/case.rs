//! Whether two names that differ only by ASCII case are one entry, as the
//! disk decides. Nothing is written and no file content is read.
use super::{Dir, Entries, LOOP, NONBLOCK};
use crate::tools::Workspace;
use std::io;
use std::os::unix::fs::MetadataExt;

const DT_DIR: u8 = 4;
const DT_REG: u8 = 8;

/// What a name is in a folder now.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Missing,
    /// A link, never followed.
    Link,
    /// Device and inode.
    Found(u64, u64),
}

fn flip(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
        .collect()
}

impl Dir {
    fn lookup(&self, name: &str) -> io::Result<Lookup> {
        match self.open(name, NONBLOCK) {
            Ok(file) => {
                let meta = file.metadata()?;
                Ok(Lookup::Found(meta.dev(), meta.ino()))
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory) => {
                Ok(Lookup::Missing)
            }
            Err(e) if e.raw_os_error() == Some(LOOP) => Ok(Lookup::Link),
            Err(e) => Err(e),
        }
    }

    /// Whether lookups in this folder fold ASCII case: an existing plain file
    /// or folder here is opened with its case flipped. The same entry means
    /// the folder folds; no entry or another entry means it does not. `None`
    /// when there is nothing here to ask. Protected names are never opened.
    fn folds_case(&self) -> io::Result<Option<bool>> {
        let mut entries = Entries::open(self)?;
        for _ in 0..16384 {
            let Some((_, kind, bytes)) = entries.next()? else {
                return Ok(None);
            };
            let Ok(name) = std::str::from_utf8(bytes) else { continue };
            if !matches!(kind, DT_DIR | DT_REG)
                || !crate::tools::allowed_component(name)
                || !name.bytes().any(|b| b.is_ascii_alphabetic())
            {
                continue;
            }
            let name = name.to_owned();
            let Ok(found @ Lookup::Found(..)) = self.lookup(&name) else { continue };
            match self.lookup(&flip(&name)) {
                Ok(other @ Lookup::Found(..)) => return Ok(Some(other == found)),
                Ok(Lookup::Missing) => return Ok(Some(false)),
                _ => continue,
            }
        }
        Ok(None)
    }
}

impl Workspace {
    /// Whether components `a` and `b`, which differ only by ASCII case, name
    /// one entry inside the workspace folder `parent` (maybe not created yet).
    /// Existing names are compared by device and inode. Otherwise the deepest
    /// existing folder is asked whether it folds case, then the folders above
    /// it on the same device, then the workspace's own path (read-only).
    /// `None` when nothing can tell. Workspace components are ASCII
    /// (`allowed_component`), so Unicode forms never make two of them one name.
    pub(crate) fn same_entry(&self, parent: &[&str], a: &str, b: &str) -> io::Result<Option<bool>> {
        let mut chain = vec![self.dir.duplicate()?];
        let mut complete = true;
        for part in parent {
            match chain[chain.len() - 1].optional_workspace_child(part)? {
                Some(child) => chain.push(child),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        let deepest = &chain[chain.len() - 1];
        if complete {
            match (deepest.lookup(a)?, deepest.lookup(b)?) {
                (x @ Lookup::Found(..), y @ Lookup::Found(..)) => return Ok(Some(x == y)),
                // A folding folder resolves both spellings to the same entry,
                // so one present and one absent (or of another kind) is two.
                (x, y) if x != y => return Ok(Some(false)),
                _ => (),
            }
        }
        let device = deepest.0.metadata()?.dev();
        for dir in chain.iter().rev() {
            if dir.0.metadata()?.dev() != device {
                return Ok(None);
            }
            if let Some(folds) = dir.folds_case()? {
                return Ok(Some(folds));
            }
        }
        for path in self.root.ancestors() {
            let (Some(up), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str())) else {
                break;
            };
            let Ok(meta) = std::fs::symlink_metadata(path) else { break };
            if meta.dev() != device {
                break;
            }
            if !name.bytes().any(|b| b.is_ascii_alphabetic()) {
                continue;
            }
            return Ok(Some(match std::fs::symlink_metadata(up.join(flip(name))) {
                Ok(other) => (other.dev(), other.ino()) == (meta.dev(), meta.ino()),
                Err(e) if e.kind() == io::ErrorKind::NotFound => false,
                Err(_) => continue,
            }));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::tools::Workspace;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("vade-case-test-{}-{n}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn case_probe_agrees_with_the_disk() {
        let f = Fixture::new();
        let w = Workspace::new(&f.0).unwrap();
        assert_eq!(w.dir.folds_case().unwrap(), None, "an empty folder has nothing to probe");
        fs::write(f.0.join("Probe"), "x").unwrap();
        // Decided by the test's own probe, not by the target OS.
        let folds = fs::symlink_metadata(f.0.join("pROBE")).is_ok();
        assert_eq!(w.dir.folds_case().unwrap(), Some(folds));
        // A missing parent is decided by the deepest existing folder.
        assert_eq!(w.same_entry(&["new"], "Abc", "abc").unwrap(), Some(folds));
        assert_eq!(w.same_entry(&[], "Probe", "probe").unwrap(), Some(folds));
        // An empty workspace is decided by its own path, without writing.
        let empty = Fixture::new();
        assert_eq!(Workspace::new(&empty.0).unwrap().same_entry(&[], "Abc", "abc").unwrap(), Some(folds));
        assert_eq!(fs::read_dir(&empty.0).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&f.0).unwrap().count(), 1);
    }
}
