//! Descriptor-anchored opens without symlinks in any path component, and the
//! file lock type that every lock owner in the CLI holds.
use crate::unix_flags::{
    O_CLOEXEC, O_CREAT, O_DIRECTORY, O_EXCL, O_NOFOLLOW, O_NONBLOCK, O_RDONLY, O_RDWR,
};
use std::ffi::{CStr, CString, OsStr};
use std::fs::{File, Metadata, OpenOptions, TryLockError};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path};

unsafe extern "C" {
    fn openat(dirfd: i32, path: *const std::ffi::c_char, flags: i32, ...) -> i32;
}

/// An exclusive flock on a file, held until this value drops.
///
/// The drop releases the lock now, not at close. The kernel keeps a flock until
/// every copy of the open file closes. A child holds a copy from its fork to its
/// exec, even with close-on-exec, and an unlock also clears that copy. After an
/// abrupt exit the kernel still releases the lock at the final close.
pub struct HeldLock(File);

impl HeldLock {
    /// Take the lock on `file` without waiting. None means that another open
    /// file holds it.
    pub fn try_take(file: File) -> io::Result<Option<Self>> {
        match file.try_lock() {
            Ok(()) => Ok(Some(Self(file))),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(error)) => Err(error),
        }
    }
}

impl Drop for HeldLock {
    fn drop(&mut self) {
        while self.0.unlock().is_err_and(|error| error.kind() == io::ErrorKind::Interrupted) {}
    }
}

/// A child with a copy of a lock file as its stdin. This is the same state as a
/// child that another thread forked and that has not reached exec. The drop
/// stops the child.
#[cfg(test)]
pub struct ChildCopy(std::process::Child);

#[cfg(test)]
impl ChildCopy {
    pub fn of(lock: &HeldLock) -> Self {
        let copy = std::process::Stdio::from(lock.0.try_clone().unwrap());
        Self(std::process::Command::new("sleep").arg("300").stdin(copy).spawn().unwrap())
    }
    /// A child that ended holds no copy, so a test proves nothing without it.
    pub fn running(&mut self) -> bool {
        self.0.try_wait().unwrap().is_none()
    }
}

#[cfg(test)]
impl Drop for ChildCopy {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// How [`AnchoredPath::open`] opens the final component.
pub enum OpenMode {
    Read,
    Create,
    CreateNew,
}

/// A path whose parent directories were opened one component at a time without
/// following symlinks; the final name opens relative to that descriptor.
pub struct AnchoredPath {
    parent: File,
    name: CString,
}

impl AnchoredPath {
    /// Walk to `path`'s parent with `O_NOFOLLOW` on every component.
    pub fn new(path: &Path) -> io::Result<Self> {
        let name = path
            .file_name()
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let parent_path = path
            .parent()
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let flags = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
        let mut parent = OpenOptions::new()
            .read(true)
            .custom_flags(flags)
            .open(if path.is_absolute() { "/" } else { "." })?;
        // O_NOFOLLOW on a whole parent path only protects its last component.
        // Open each component relative to the previous descriptor instead.
        for part in parent_path.components() {
            let part = match part {
                Component::RootDir | Component::CurDir => continue,
                Component::Normal(name) => name,
                Component::ParentDir => OsStr::new(".."),
                Component::Prefix(_) => return Err(io::ErrorKind::InvalidInput.into()),
            };
            parent = open_relative(&parent, &cstring(part)?, flags)?;
        }
        Ok(Self {
            parent,
            name: cstring(name)?,
        })
    }

    /// Open the final component without following a symlink.
    pub fn open(&self, mode: OpenMode) -> io::Result<File> {
        let access = match mode {
            OpenMode::Read => O_RDONLY,
            OpenMode::Create => O_RDWR | O_CREAT,
            OpenMode::CreateNew => O_RDWR | O_CREAT | O_EXCL,
        };
        // Nonblocking prevents a substituted FIFO from hanging before fstat.
        // It has no effect on the regular files accepted by our callers.
        open_relative(
            &self.parent,
            &self.name,
            access | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
        )
    }
}

fn cstring(name: &OsStr) -> io::Result<CString> {
    CString::new(name.as_bytes()).map_err(|_| io::ErrorKind::InvalidInput.into())
}

fn open_relative(parent: &File, name: &CStr, flags: i32) -> io::Result<File> {
    // The variadic mode uses an int-width slot on supported targets. Always
    // pass mode 600; openat ignores it unless O_CREAT is set.
    let fd = unsafe { openat(parent.as_raw_fd(), name.as_ptr(), flags, 0o600u32) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // A successful openat transfers ownership of this new descriptor to File.
    Ok(unsafe { File::from_raw_fd(fd) })
}

/// Whether both metadata describe the same single-link regular file (device
/// and inode), so a path was not swapped between check and open.
pub fn same_regular_file(expected: &Metadata, opened: &Metadata) -> bool {
    expected.is_file()
        && opened.is_file()
        && expected.nlink() == 1
        && opened.nlink() == 1
        && expected.dev() == opened.dev()
        && expected.ino() == opened.ino()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;

    struct TestDir(PathBuf);
    impl TestDir {
        fn new() -> Self {
            let path = fs::canonicalize(std::env::temp_dir())
                .unwrap()
                .join(format!(
                    "vade-anchored-{}",
                    crate::text::random_hex(12).unwrap()
                ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn anchored_regular_files_open_and_final_symlinks_fail_with_eloop() {
        let root = TestDir::new();
        let target = root.0.join("file");
        let anchor = AnchoredPath::new(&target).unwrap();
        anchor.open(OpenMode::CreateNew).unwrap();
        let expected = fs::symlink_metadata(&target).unwrap();
        for mode in [OpenMode::Read, OpenMode::Create] {
            let file = anchor.open(mode).unwrap();
            assert!(same_regular_file(&expected, &file.metadata().unwrap()));
        }
        assert!(anchor.open(OpenMode::CreateNew).is_err());
        fs::create_dir(root.0.join("directory")).unwrap();
        for name in ["file", "directory"] {
            let link = root.0.join(format!("{name}-link"));
            symlink(root.0.join(name), &link).unwrap();
            File::open(&link).expect("valid control symlink");
            let link = AnchoredPath::new(&link).unwrap();
            for mode in [OpenMode::Read, OpenMode::Create] {
                let error = link.open(mode).expect_err("final symlink must be refused");
                #[cfg(target_os = "linux")]
                let eloop = 40;
                #[cfg(target_os = "macos")]
                let eloop = 62;
                assert_eq!(error.raw_os_error(), Some(eloop));
            }
        }
    }

    #[test]
    fn symlinked_parents_are_refused_at_every_depth() {
        let root = TestDir::new();
        fs::create_dir_all(root.0.join("real/nested")).unwrap();
        symlink(root.0.join("real"), root.0.join("linked")).unwrap();
        for suffix in ["linked/file", "linked/nested/file"] {
            assert!(AnchoredPath::new(&root.0.join(suffix)).is_err());
        }
        assert!(!root.0.join("real/file").exists());
        assert!(!root.0.join("real/nested/file").exists());
    }

    #[test]
    fn open_uses_the_pinned_parent_after_path_replacement() {
        let root = TestDir::new();
        fs::create_dir(root.0.join("lane")).unwrap();
        fs::write(root.0.join("lane/file"), "original").unwrap();
        let anchor = AnchoredPath::new(&root.0.join("lane/file")).unwrap();
        fs::rename(root.0.join("lane"), root.0.join("saved")).unwrap();
        fs::create_dir(root.0.join("foreign")).unwrap();
        fs::write(root.0.join("foreign/file"), "foreign").unwrap();
        symlink(root.0.join("foreign"), root.0.join("lane")).unwrap();
        use std::io::Read;
        let mut found = String::new();
        anchor
            .open(OpenMode::Read)
            .unwrap()
            .read_to_string(&mut found)
            .unwrap();
        assert_eq!(found, "original");
        assert_eq!(
            fs::read_to_string(root.0.join("foreign/file")).unwrap(),
            "foreign"
        );
    }

    #[test]
    fn after_open_check_refuses_nonregular_replaced_and_linked_files() {
        let root = TestDir::new();
        let target = root.0.join("file");
        fs::write(&target, "original").unwrap();
        let expected = fs::symlink_metadata(&target).unwrap();
        let directory = AnchoredPath::new(&root.0)
            .unwrap()
            .open(OpenMode::Read)
            .unwrap()
            .metadata()
            .unwrap();
        assert!(!same_regular_file(&directory, &directory));
        fs::rename(&target, root.0.join("saved")).unwrap();
        fs::write(&target, "replacement").unwrap();
        let opened = AnchoredPath::new(&target)
            .unwrap()
            .open(OpenMode::Read)
            .unwrap();
        assert!(!same_regular_file(&expected, &opened.metadata().unwrap()));
        fs::hard_link(&target, root.0.join("alias")).unwrap();
        let linked = opened.metadata().unwrap();
        assert!(!same_regular_file(&linked, &linked));
        fs::remove_file(&target).unwrap();
        fs::remove_file(root.0.join("alias")).unwrap();
        let unlinked = opened.metadata().unwrap();
        assert!(!same_regular_file(&unlinked, &unlinked));
    }
}
