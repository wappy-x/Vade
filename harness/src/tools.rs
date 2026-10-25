//! Explicit, bounded, read-only workspace operations.
//!
//! Unix directory descriptors anchor every lookup. Symlinks, parent traversal,
//! hidden files and common credential filenames are refused, including at leaf
//! components. Results are local UI output, never automatic model context.
use std::io;
use std::path::{Component, Path, PathBuf};
mod discovery;
mod state;
pub use state::{StateLocation, project_key};

pub const MAX_READ_BYTES: usize = 64 * 1024;
pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_LIST_ENTRIES: usize = 256;

/// Local revision guard, not a content hash or a security identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileStamp {
    pub directory: bool,
    pub bytes: u64,
    pub identity: String,
}
impl FileStamp {
    pub(crate) fn from_metadata(m: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                directory: m.is_dir(),
                bytes: m.len(),
                identity: format!(
                    "{}:{}:{}:{}:{}:{}:{}",
                    m.dev(),
                    m.ino(),
                    m.len(),
                    m.mtime(),
                    m.mtime_nsec(),
                    m.ctime(),
                    m.ctime_nsec()
                ),
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                directory: m.is_dir(),
                bytes: m.len(),
                identity: String::new(),
            }
        }
    }
}

/// The folder the agent may read and edit, opened once and anchored so paths
/// cannot escape it, plus where its private chat state lives.
pub struct Workspace {
    root: PathBuf,
    state: StateLocation,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    dir: anchored::Dir,
}

impl std::fmt::Debug for Workspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workspace").field("root", &self.root).field("state", &self.state).finish_non_exhaustive()
    }
}

impl Workspace {
    /// Keeps private state in `<root>/.vade`; the binary's default is
    /// [`StateLocation::Home`] through [`Workspace::open`].
    pub fn new(root: &Path) -> io::Result<Self> {
        Self::open(root, StateLocation::Project)
    }

    pub fn open(root: &Path, state: StateLocation) -> io::Result<Self> {
        let root = root
            .canonicalize()
            .map_err(|_| error("cannot open workspace"))?;
        if !root.is_dir() {
            return Err(error("workspace must be a directory"));
        }
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        let dir = anchored::Dir::open_root(&root)?;
        Ok(Self {
            root,
            state,
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            dir,
        })
    }

    /// Used for local storage and sandbox setup, never automatically sent upstream.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Private session journals and input recall; created on first use.
    pub fn state_path(&self) -> PathBuf {
        self.state.path(&self.root)
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn verify_root(&self) -> io::Result<()> {
        self.dir.verify_path(&self.root)
    }

    pub fn read(&self, path: &str) -> io::Result<String> {
        self.read_optional_bounded(path, MAX_READ_BYTES)?
            .ok_or_else(|| error("file not found"))
    }

    /// Byte ranges use the same anchored path checks as whole-file reads.
    /// Partial results state the actual half-open range and total source size.
    pub fn read_range(&self, path: &str, offset: usize, limit: usize) -> io::Result<String> {
        use std::io::{Read, Seek, SeekFrom};
        if !(1..=MAX_READ_BYTES).contains(&limit) {
            return Err(error(&format!(
                "requested read limit is {limit} bytes; range limit is 1..={MAX_READ_BYTES} bytes"
            )));
        }
        let (mut file, stamp) = self
            .open_snapshot(path)?
            .ok_or_else(|| error("file not found"))?;
        if stamp.directory {
            return Err(error("read requires a regular file"));
        }
        if stamp.bytes > MAX_TEXT_BYTES as u64 {
            return Err(error(&format!(
                "source file is {} bytes; ranged-read source limit is {MAX_TEXT_BYTES} bytes (4 MiB)",
                stamp.bytes
            )));
        }
        if offset as u64 > stamp.bytes {
            return Err(error(&format!(
                "read offset is {offset} bytes; source file is {} bytes",
                stamp.bytes
            )));
        }
        file.seek(SeekFrom::Start(offset as u64))?;
        let mut bytes = Vec::new();
        (&mut file).take(limit as u64).read_to_end(&mut bytes)?;
        if FileStamp::from_metadata(&file.metadata()?) != stamp {
            return Err(error("source file changed during ranged read"));
        }
        // A limit may cut a multibyte character. Leave that character for the
        // next range; a non-boundary offset or invalid text still fails.
        let text = match std::str::from_utf8(&bytes) {
            Ok(text) => text,
            Err(e)
                if e.error_len().is_none()
                    && e.valid_up_to() > 0
                    && offset as u64 + (bytes.len() as u64) < stamp.bytes =>
            {
                std::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap()
            }
            Err(_) => {
                return Err(error(
                    "range is not UTF-8 text; offset must be a UTF-8 byte boundary and limit must fit one character",
                ));
            }
        };
        let end = offset + text.len();
        if offset == 0 && end as u64 == stamp.bytes {
            return Ok(text.to_owned());
        }
        Ok(format!(
            "Read bytes {offset}..{end} of {} ({} bytes; range limit {limit}; next offset {end}).\n{text}",
            stamp.bytes,
            text.len()
        ))
    }

    /// Uses the same anchored boundary as tools; a missing leaf is distinct from
    /// an unsafe redirect. Callers cannot broaden the global read limit.
    pub(crate) fn read_optional_bounded(
        &self,
        path: &str,
        cap: usize,
    ) -> io::Result<Option<String>> {
        Ok(self.read_snapshot_bounded(path, cap)?.map(|(text, _)| text))
    }
    pub(crate) fn read_snapshot_bounded(
        &self,
        path: &str,
        cap: usize,
    ) -> io::Result<Option<(String, FileStamp)>> {
        let Some((file, stamp)) = self.open_snapshot(path)? else {
            return Ok(None);
        };
        let cap = cap.min(MAX_READ_BYTES);
        if stamp.directory {
            return Err(error("read requires a regular file"));
        }
        if stamp.bytes > cap as u64 {
            return Err(error(&format!(
                "source file is {} bytes; whole-file read limit is {cap} bytes; use workspace_read offset/limit byte ranges",
                stamp.bytes
            )));
        }
        use std::io::Read;
        let mut bytes = Vec::new();
        (&file).take((cap + 1) as u64).read_to_end(&mut bytes)?;
        if bytes.len() > cap {
            return Err(error(&format!(
                "source read is {} bytes; whole-file read limit is {cap} bytes",
                bytes.len()
            )));
        }
        if bytes.len() as u64 != stamp.bytes || FileStamp::from_metadata(&file.metadata()?) != stamp
        {
            return Err(error("text source changed during read"));
        }
        String::from_utf8(bytes)
            .map(|text| Some((text, stamp)))
            .map_err(|_| error("file is not UTF-8 text"))
    }

    pub(crate) fn stamp(&self, path: &str) -> io::Result<Option<FileStamp>> {
        Ok(self.open_snapshot(path)?.map(|(_, stamp)| stamp))
    }
    fn open_snapshot(&self, path: &str) -> io::Result<Option<(std::fs::File, FileStamp)>> {
        let parts = safe_parts(path)?;
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            use std::os::unix::fs::MetadataExt;
            let file = if parts.is_empty() {
                self.dir.file_copy()?
            } else {
                let (parent, leaf) = self.parent(&parts)?;
                let Some(file) = parent.workspace_file(leaf)? else {
                    return Ok(None);
                };
                file
            };
            let metadata = file.metadata()?;
            if !metadata.is_dir() && (!metadata.is_file() || metadata.nlink() != 1) {
                return Err(error(
                    "only directories or regular files with one hard link may be inspected",
                ));
            }
            Ok(Some((file, FileStamp::from_metadata(&metadata))))
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = parts;
            Err(error(
                "workspace tools require macOS or Linux directory-handle protection",
            ))
        }
    }

    /// The deepest existing parent directory, any missing directories below it
    /// (created only when a patch is applied), the leaf name and the normalized
    /// path. Existing components are verified exactly as `parent` does.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn edit_parent(
        &self,
        path: &str,
    ) -> io::Result<(anchored::Dir, Vec<String>, String, String)> {
        let parts = safe_parts(path)?;
        let Some((name, parents)) = parts.split_last() else {
            return Err(error("file path required"));
        };
        let mut dir = self.dir.duplicate()?;
        let mut missing = Vec::new();
        for part in parents {
            if missing.is_empty() {
                if let Some(child) = dir.optional_workspace_child(part)? {
                    dir = child;
                    continue;
                }
            }
            missing.push(part.to_string());
        }
        Ok((dir, missing, name.to_string(), parts.join("/")))
    }

    pub fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let parts = safe_parts(path)?;
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let mut dir = self.dir.duplicate()?;
            for part in parts {
                dir = dir.workspace_child(part)?;
            }
            let mut names = dir.names(MAX_LIST_ENTRIES)?;
            names.retain(|name| allowed_component(name));
            names.sort();
            // Enumeration itself is capped, so a very large directory has a
            // bounded cost even if most of its entries are hidden.
            Ok(names)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = parts;
            Err(error(
                "workspace tools require macOS or Linux directory-handle protection",
            ))
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn parent<'a>(&self, parts: &[&'a str]) -> io::Result<(anchored::Dir, &'a str)> {
        let mut dir = self.dir.duplicate()?;
        for part in &parts[..parts.len() - 1] {
            dir = dir.workspace_child(part)?;
        }
        Ok((dir, parts[parts.len() - 1]))
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn state_dir(&self) -> io::Result<anchored::Dir> {
        match &self.state {
            StateLocation::Home(home) => state::open_home(home, &self.root),
            StateLocation::Project => {
                let dir = self.dir.private_child(".vade")?;
                dir.ignore_in_git()?;
                Ok(dir)
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn session_dir(&self) -> io::Result<anchored::Dir> {
        self.state_dir()?.private_child("sessions")
    }
}

fn error(message: &str) -> io::Error {
    io::Error::other(message)
}

pub(crate) const CREDENTIAL_NAMES: &[&str] = &[
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
    "netrc",
    "service-account.json",
    "service_account.json",
];
pub(crate) const CREDENTIAL_SUFFIXES: &[&str] = &[
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".keystore",
    ".p8",
    ".jks",
    ".tfstate",
    ".tfstate.backup",
];

pub(crate) fn allowed_component(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    name.is_ascii()
        && !name.starts_with('.')
        && !name.chars().any(char::is_control)
        && !CREDENTIAL_NAMES.contains(&lower.as_str())
        && !CREDENTIAL_SUFFIXES
            .iter()
            .any(|suffix| lower.ends_with(suffix))
}

pub(crate) fn safe_parts(path: &str) -> io::Result<Vec<&str>> {
    if path.len() > 4096 || path.contains('\\') || path.chars().any(char::is_control) {
        return Err(error("invalid workspace-relative path"));
    }
    let mut parts = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => (),
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| error("path must be UTF-8"))?;
                if !allowed_component(value) {
                    return Err(error(
                        "non-ASCII, hidden or credential-related paths are blocked",
                    ));
                }
                parts.push(value);
            }
            _ => return Err(error("use a relative path without parent traversal")),
        }
    }
    Ok(parts)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) mod anchored {
    #[cfg(all(
        target_os = "linux",
        not(any(target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    compile_error!("workspace ABI supports Linux x86_64 and aarch64 only");

    use super::error;
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::fs::{File, OpenOptions};
    use std::io::{self, Write};
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[cfg(target_os = "macos")]
    type Mode = u16;
    #[cfg(target_os = "linux")]
    type Mode = u32;

    #[cfg(target_os = "macos")]
    const NOFOLLOW: c_int = 0x100;
    #[cfg(target_os = "macos")]
    const DIRECTORY: c_int = 0x100000;
    #[cfg(target_os = "macos")]
    const CLOEXEC: c_int = 0x1000000;
    #[cfg(target_os = "macos")]
    const CREATE: c_int = 0x200;
    #[cfg(target_os = "macos")]
    const CREATE_EXCLUSIVE: c_int = CREATE | 0x800;
    #[cfg(target_os = "macos")]
    const NONBLOCK: c_int = 0x4;
    // Linux arm64 overrides the asm-generic O_DIRECTORY/O_NOFOLLOW values;
    // x86-64 uses asm-generic. Keep these tied to the target, not the build host.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const NOFOLLOW: c_int = 0x20000;
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const DIRECTORY: c_int = 0x10000;
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    const NOFOLLOW: c_int = 0x8000;
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    const DIRECTORY: c_int = 0x4000;
    #[cfg(target_os = "linux")]
    const CLOEXEC: c_int = 0x80000;
    #[cfg(target_os = "linux")]
    const CREATE: c_int = 0x40;
    #[cfg(target_os = "linux")]
    const CREATE_EXCLUSIVE: c_int = CREATE | 0x80;
    #[cfg(target_os = "linux")]
    const NONBLOCK: c_int = 0x800;

    #[repr(C)]
    struct Dirent {
        ino: u64,
        offset: i64,
        record_len: u16,
        #[cfg(target_os = "macos")]
        name_len: u16,
        kind: u8,
        name: [c_char; 1],
    }

    // readdir returns a larger object; this prefix must put d_name at exactly
    // the offset declared by the platform's dirent.h on both 64-bit targets.
    const _: () = {
        assert!(std::mem::offset_of!(Dirent, ino) == 0);
        assert!(std::mem::offset_of!(Dirent, offset) == 8);
        assert!(std::mem::offset_of!(Dirent, record_len) == 16);
        #[cfg(target_os = "macos")]
        {
            assert!(std::mem::offset_of!(Dirent, kind) == 20);
            assert!(std::mem::offset_of!(Dirent, name) == 21);
        }
        #[cfg(target_os = "linux")]
        {
            assert!(std::mem::offset_of!(Dirent, kind) == 18);
            assert!(std::mem::offset_of!(Dirent, name) == 19);
        }
    };

    unsafe extern "C" {
        fn openat(fd: c_int, path: *const c_char, flags: c_int, ...) -> c_int;
        fn mkdirat(fd: c_int, path: *const c_char, mode: Mode) -> c_int;
        fn renameat(from: c_int, old: *const c_char, to: c_int, new: *const c_char) -> c_int;
        #[cfg(target_os = "macos")]
        fn renameatx_np(
            from: c_int,
            old: *const c_char,
            to: c_int,
            new: *const c_char,
            flags: u32,
        ) -> c_int;
        #[cfg(target_os = "linux")]
        fn renameat2(
            from: c_int,
            old: *const c_char,
            to: c_int,
            new: *const c_char,
            flags: u32,
        ) -> c_int;
        fn unlinkat(fd: c_int, path: *const c_char, flags: c_int) -> c_int;
        fn fdopendir(fd: c_int) -> *mut c_void;
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "readdir$INODE64"
        )]
        fn readdir(dir: *mut c_void) -> *mut Dirent;
        fn closedir(dir: *mut c_void) -> c_int;
        #[cfg(target_os = "macos")]
        fn __error() -> *mut c_int;
        #[cfg(target_os = "linux")]
        fn __errno_location() -> *mut c_int;
        fn geteuid() -> u32;
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    pub(crate) struct Dir(File);
    // Own a fresh open description: dup would share a directory cursor with
    // other lookups. No pathname lookup is used to enumerate an anchored parent.
    struct Entries(*mut c_void);
    impl Entries {
        fn open(dir: &Dir) -> io::Result<Self> {
            let fd = dir.open(".", DIRECTORY)?.into_raw_fd();
            // SAFETY: fdopendir takes ownership only on success.
            let stream = unsafe { fdopendir(fd) };
            if stream.is_null() {
                drop(unsafe { File::from_raw_fd(fd) });
                return Err(error("cannot list directory"));
            }
            Ok(Self(stream))
        }
        fn next(&mut self) -> io::Result<Option<(u64, &[u8])>> {
            // readdir uses errno to distinguish EOF from an enumeration error.
            // SAFETY: these functions return this thread's live errno pointer.
            let errno = unsafe {
                #[cfg(target_os = "macos")]
                {
                    __error()
                }
                #[cfg(target_os = "linux")]
                {
                    __errno_location()
                }
            };
            loop {
                // SAFETY: stream is owned; the returned name is borrowed until
                // the next mutable call, so no readdir can invalidate it early.
                let entry = unsafe {
                    *errno = 0;
                    readdir(self.0)
                };
                if entry.is_null() {
                    return if unsafe { *errno } == 0 {
                        Ok(None)
                    } else {
                        Err(error("cannot finish directory enumeration"))
                    };
                }
                let bytes = unsafe { CStr::from_ptr((*entry).name.as_ptr()) }.to_bytes();
                if bytes != b"." && bytes != b".." {
                    return Ok(Some((unsafe { (*entry).ino }, bytes)));
                }
            }
        }
    }
    impl Drop for Entries {
        fn drop(&mut self) {
            unsafe {
                closedir(self.0);
            }
        }
    }
    /// The sole owner of an acquired session lock. Raw descriptor duplicates
    /// (including pre-exec child inheritance) must not extend this guard's lock.
    pub(crate) struct Lock(File);
    impl Drop for Lock {
        fn drop(&mut self) {
            // flock belongs to the open file description, not this descriptor.
            // Closing alone can leave it locked while a concurrent spawn holds
            // an inherited CLOEXEC descriptor before exec. LOCK_UN releases the
            // owner's lock now, without unlinking/replacing the stable lock file.
            // The kernel still releases locks on final close after abrupt exit.
            // SAFETY: the owned File stays live until this destructor returns.
            while unsafe { flock(self.0.as_raw_fd(), 8) } != 0 {
                if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    break; // File::drop remains the fallback on an OS error.
                }
            }
        }
    }
    impl Lock {
        #[cfg(test)]
        pub(crate) fn duplicate_descriptor(&self) -> io::Result<File> {
            self.0.try_clone()
        }
    }
    impl Dir {
        pub fn open_root(path: &Path) -> io::Result<Self> {
            OpenOptions::new()
                .read(true)
                .custom_flags(DIRECTORY | NOFOLLOW | CLOEXEC)
                .open(path)
                .map(Self)
                .map_err(|_| error("cannot safely open workspace directory"))
        }
        pub(crate) fn verify_path(&self, path: &Path) -> io::Result<()> {
            let current = std::fs::symlink_metadata(path)?;
            let opened = self.0.metadata()?;
            if !current.is_dir() || current.dev() != opened.dev() || current.ino() != opened.ino() {
                return Err(error("workspace root moved or replaced"));
            }
            Ok(())
        }
        pub(crate) fn file_copy(&self) -> io::Result<File> {
            self.0.try_clone()
        }
        pub fn duplicate(&self) -> io::Result<Self> {
            self.0.try_clone().map(Self)
        }
        fn open(&self, name: &str, flags: c_int) -> io::Result<File> {
            let name = CString::new(name).map_err(|_| error("invalid filename"))?;
            // SAFETY: live directory descriptor, NUL-terminated component, and
            // an explicit mode for the variadic creation argument.
            let fd = unsafe {
                openat(
                    self.0.as_raw_fd(),
                    name.as_ptr(),
                    flags | NOFOLLOW | CLOEXEC,
                    0o600u32,
                )
            };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: openat returned a new owned descriptor.
            Ok(unsafe { File::from_raw_fd(fd) })
        }
        pub fn child(&self, name: &str) -> io::Result<Self> {
            self.open(name, DIRECTORY)
                .map(Self)
                .map_err(|_| error("directory is unavailable or is a symlink"))
        }
        fn verify_workspace_entry(&self, name: &str, file: &File) -> io::Result<()> {
            // Checking only the requested spelling is insufficient on folding
            // filesystems: ASCII input can resolve to a non-ASCII entry too.
            // Require the actual entry's exact bytes and the opened inode to
            // agree before reading content or traversing another component.
            if !super::allowed_component(name) {
                return Err(error(
                    "non-ASCII, hidden or credential-related paths are blocked",
                ));
            }
            let inode = file.metadata()?.ino();
            let mut entries = Entries::open(self)?;
            for _ in 0..16384 {
                let Some((entry_inode, bytes)) = entries.next()? else {
                    return Err(error("opened file has no matching directory entry"));
                };
                if entry_inode == inode {
                    let actual = std::str::from_utf8(bytes)
                        .map_err(|_| error("non-UTF-8 directory entry is blocked"))?;
                    if actual != name || !super::allowed_component(actual) {
                        return Err(error(
                            "filename alias or protected directory entry is blocked",
                        ));
                    }
                    return Ok(());
                }
            }
            Err(error("directory too large to verify filename safely"))
        }
        pub(crate) fn workspace_child(&self, name: &str) -> io::Result<Self> {
            let child = self.child(name)?;
            self.verify_workspace_entry(name, &child.0)?;
            Ok(child)
        }
        /// Like `workspace_child`, but only an absent entry is `None`; links,
        /// files and aliases remain errors.
        pub(crate) fn optional_workspace_child(&self, name: &str) -> io::Result<Option<Self>> {
            match self.open(name, DIRECTORY) {
                Ok(file) => {
                    self.verify_workspace_entry(name, &file)?;
                    Ok(Some(Self(file)))
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(_) => Err(error("directory is unavailable or is a symlink")),
            }
        }
        /// Creates a missing workspace directory, then opens it without
        /// following links. An existing link or alias at `name` is refused.
        pub(crate) fn create_workspace_child(&self, name: &str) -> io::Result<Self> {
            if !super::allowed_component(name) {
                return Err(error(
                    "non-ASCII, hidden or credential-related paths are blocked",
                ));
            }
            let name_c = CString::new(name).map_err(|_| error("invalid directory name"))?;
            // SAFETY: valid descriptor and component; mkdirat never follows a
            // final link, and the umask narrows the requested mode.
            if unsafe { mkdirat(self.0.as_raw_fd(), name_c.as_ptr(), 0o777) } == 0 {
                self.0.sync_all()?;
            } else if io::Error::last_os_error().kind() != io::ErrorKind::AlreadyExists {
                return Err(error("cannot create workspace directory"));
            }
            self.workspace_child(name)
        }
        pub(crate) fn workspace_file(&self, name: &str) -> io::Result<Option<File>> {
            if !super::allowed_component(name) {
                return Err(error(
                    "non-ASCII, hidden or credential-related paths are blocked",
                ));
            }
            match self.open(name, NONBLOCK) {
                Ok(file) => {
                    self.verify_workspace_entry(name, &file)?;
                    Ok(Some(file))
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(_) => Err(error("file unavailable or symlink")),
            }
        }
        pub(crate) fn journal_file(&self, name: &str) -> io::Result<File> {
            let file = self.open(name, 2 | CREATE | NONBLOCK)?;
            let m = file.metadata()?;
            if !m.is_file()
                || m.nlink() != 1
                || m.mode() & 0o077 != 0
                || m.uid() != unsafe { geteuid() }
            {
                return Err(error("journal must be a private owned regular file"));
            }
            self.0.sync_all()?;
            Ok(file)
        }
        pub(crate) fn remove(&self, name: &str) -> io::Result<()> {
            let name = CString::new(name).map_err(|_| error("invalid filename"))?;
            if unsafe { unlinkat(self.0.as_raw_fd(), name.as_ptr(), 0) } != 0 {
                return Err(error("cannot remove file"));
            }
            self.0.sync_all()
        }
        pub(crate) fn move_exclusive(&self, name: &str, to: &Self, target: &str) -> io::Result<()> {
            self.rename(name, to, target, true)?;
            self.0.sync_all()?;
            to.0.sync_all()
        }
        fn rename(&self, name: &str, to: &Self, target: &str, exclusive: bool) -> io::Result<()> {
            let name = CString::new(name).map_err(|_| error("invalid filename"))?;
            let target = CString::new(target).map_err(|_| error("invalid filename"))?;
            let result = unsafe {
                if exclusive {
                    #[cfg(target_os = "macos")]
                    {
                        renameatx_np(
                            self.0.as_raw_fd(),
                            name.as_ptr(),
                            to.0.as_raw_fd(),
                            target.as_ptr(),
                            4,
                        )
                    }
                    #[cfg(target_os = "linux")]
                    {
                        renameat2(
                            self.0.as_raw_fd(),
                            name.as_ptr(),
                            to.0.as_raw_fd(),
                            target.as_ptr(),
                            1,
                        )
                    }
                } else {
                    renameat(
                        self.0.as_raw_fd(),
                        name.as_ptr(),
                        to.0.as_raw_fd(),
                        target.as_ptr(),
                    )
                }
            };
            if result != 0 {
                return Err(error("atomic rename failed; destination may already exist"));
            }
            Ok(())
        }
        pub(crate) fn install(
            &self,
            name: &str,
            bytes: &[u8],
            mode: u32,
            exclusive: bool,
            check: impl FnOnce() -> io::Result<()>,
        ) -> io::Result<()> {
            use std::os::unix::fs::PermissionsExt;
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let temp = format!(
                ".vade-edit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let mut file = self.open(&temp, 1 | CREATE_EXCLUSIVE)?;
            let result = (|| {
                file.write_all(bytes)?;
                file.set_permissions(std::fs::Permissions::from_mode(mode))?;
                file.sync_all()?;
                check()?;
                self.rename(&temp, self, name, exclusive)?;
                self.0.sync_all()
            })();
            if result.is_err() {
                let _ = self.remove(&temp);
            }
            result
        }
        pub fn optional_file(&self, name: &str) -> io::Result<Option<File>> {
            match self.open(name, NONBLOCK) {
                Ok(file) => {
                    let meta = file.metadata()?;
                    if !meta.is_file()
                        || meta.nlink() != 1
                        || meta.mode() & 0o077 != 0
                        || meta.uid() != unsafe { geteuid() }
                    {
                        return Err(error(
                            "session metadata must be a private, owned regular file",
                        ));
                    }
                    Ok(Some(file))
                }
                Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(_) => Err(error("cannot safely read session metadata")),
            }
        }
        pub fn lock(&self, name: &str) -> io::Result<Lock> {
            let file = self.open(name, 2 | CREATE | NONBLOCK)?;
            let meta = file.metadata()?;
            if !meta.is_file()
                || meta.nlink() != 1
                || meta.mode() & 0o077 != 0
                || meta.uid() != unsafe { geteuid() }
            {
                return Err(error("session lock must be a private, owned regular file"));
            }
            // LOCK_EX | LOCK_NB. Only a successful acquisition gets a guard;
            // failed contenders must never unlock another owner's description.
            if unsafe { flock(file.as_raw_fd(), 2 | 4) } != 0 {
                return Err(error("session name is already in use; choose another name"));
            }
            Ok(Lock(file))
        }
        pub fn private_child(&self, name: &str) -> io::Result<Self> {
            let name_c = CString::new(name).map_err(|_| error("invalid directory name"))?;
            // SAFETY: valid descriptor and component; creation never follows a link.
            if unsafe { mkdirat(self.0.as_raw_fd(), name_c.as_ptr(), 0o700) } != 0
                && io::Error::last_os_error().kind() != io::ErrorKind::AlreadyExists
            {
                return Err(error("cannot create private session directory"));
            }
            let dir = self.child(name)?;
            let meta = dir.0.metadata()?;
            if meta.mode() & 0o077 != 0 || meta.uid() != unsafe { geteuid() } {
                return Err(error(
                    "session directories must be owned by you with permissions 0700",
                ));
            }
            Ok(dir)
        }
        /// Never replaces an existing file; a link or user file there is kept.
        pub(crate) fn ignore_in_git(&self) -> io::Result<()> {
            match self.open(".gitignore", 1 | CREATE_EXCLUSIVE) {
                Ok(mut file) => {
                    file.write_all(super::state::GITIGNORE)?;
                    file.sync_all()?;
                    self.0.sync_all()
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
                Err(_) => Err(error("cannot create the private state .gitignore")),
            }
        }
        pub fn names(&self, cap: usize) -> io::Result<Vec<String>> {
            let mut entries = Entries::open(self)?;
            let mut names = Vec::new();
            let mut seen = 0;
            while seen < cap {
                let Some((_, bytes)) = entries.next()? else {
                    break;
                };
                seen += 1;
                if let Ok(name) = std::str::from_utf8(bytes) {
                    if super::allowed_component(name) {
                        names.push(name.to_string());
                    }
                }
            }
            if seen == cap {
                names.push("[listing limit reached]".into());
            }
            Ok(names)
        }
        pub fn write_atomic(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
            static SEQUENCE: AtomicU64 = AtomicU64::new(0);
            let temp = format!(
                ".write-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            );
            let mut file = self.open(&temp, 1 | CREATE_EXCLUSIVE)?;
            let temp_c = CString::new(temp).unwrap();
            let name_c = CString::new(name).map_err(|_| error("invalid metadata name"))?;
            let result = (|| {
                file.write_all(bytes)?;
                file.sync_all()?;
                if unsafe {
                    renameat(
                        self.0.as_raw_fd(),
                        temp_c.as_ptr(),
                        self.0.as_raw_fd(),
                        name_c.as_ptr(),
                    )
                } != 0
                {
                    return Err(error("cannot replace session metadata"));
                }
                self.0.sync_all()
            })();
            if result.is_err() {
                unsafe {
                    unlinkat(self.0.as_raw_fd(), temp_c.as_ptr(), 0);
                }
            }
            result.map_err(|_| error("cannot save session metadata"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "vade-tools-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn workspace(&self) -> Workspace {
            Workspace::new(&self.0).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn rejects_unsafe_paths() {
        for path in [
            "/etc/passwd",
            "../a",
            "a/../b",
            ".git/config",
            ".env",
            "a/credentials.json",
            "id_rsa",
            "x.pem",
            "a\\b",
            "a\0b",
        ] {
            assert!(safe_parts(path).is_err(), "{path:?}");
        }
        assert_eq!(safe_parts("./src/main.rs").unwrap(), ["src", "main.rs"]);
    }
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn bounded_text_reads_and_repeatable_listings() {
        let f = Fixture::new();
        fs::write(f.0.join("ok.txt"), "hello\nworld").unwrap();
        fs::write(f.0.join("large.txt"), vec![b'x'; MAX_READ_BYTES + 1]).unwrap();
        fs::write(f.0.join("binary"), [255, 254]).unwrap();
        fs::write(f.0.join(".env"), "private").unwrap();
        let w = f.workspace();
        assert_eq!(w.read("ok.txt").unwrap(), "hello\nworld");
        assert!(w.read("large.txt").is_err());
        assert!(
            w.read("large.txt")
                .unwrap_err()
                .to_string()
                .contains("65537 bytes; whole-file read limit is 65536 bytes")
        );
        assert!(
            w.read_range("large.txt", 65530, 32)
                .unwrap()
                .ends_with("\nxxxxxxx")
        );
        assert!(
            w.read_range("large.txt", 65530, 32)
                .unwrap()
                .contains("65530..65537 of 65537")
        );
        fs::write(f.0.join("utf8"), "a🦀z").unwrap();
        assert!(w.read_range("utf8", 0, 3).unwrap().ends_with("\na"));
        assert!(w.read_range("utf8", 2, 3).is_err());
        assert!(w.read_range(".env", 0, 16).is_err());
        assert!(w.read("binary").is_err());
        assert!(w.read(".").is_err());
        let names = w.list(".").unwrap();
        assert!(!names.iter().any(|n| n == ".env"));
        assert_eq!(names, w.list(".").unwrap());
        let cancel = std::sync::atomic::AtomicBool::new(false);
        fs::write(f.0.join("search.txt"), "a.b 🦀\naXb\na.b again\n").unwrap();
        let page = crate::json::parse(
            &w.search_literals("search.txt", &["a.b", "absent"], 0, 1, 512, &cancel)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            page.get("next_offset"),
            Some(&crate::json::Value::Number("1".into()))
        );
        let page = crate::json::parse(
            &w.search_literal("search.txt", "a.b", 1, 1, 512, &cancel)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            page.get("matches").unwrap().as_array().unwrap()[0].get("line"),
            Some(&crate::json::Value::Number("3".into()))
        );
        let lines = w
            .read_line_ranges("search.txt", &[(2, 2), (3, 3)], 1024, &cancel)
            .unwrap();
        assert!(lines.contains("aXb") && lines.contains("a.b again"));
        for path in ["../search.txt", ".env", "target"] {
            assert!(w.search_literal(path, "a", 0, 32, 512, &cancel).is_err());
        }
        assert!(w.read_lines(".env", 1, 2, 1024, &cancel).is_err());
        assert!(w.read_lines("search.txt", 0, 2, 1024, &cancel).is_err());
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(w.search_literal(".", "a", 0, 32, 512, &cancel).is_err());
    }
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn an_open_workspace_cannot_be_redirected_by_replacing_its_path() {
        use std::os::unix::fs::symlink;
        let mut inside = Fixture::new();
        let outside = Fixture::new();
        fs::write(inside.0.join("file"), "inside").unwrap();
        fs::write(outside.0.join("file"), "outside").unwrap();
        let workspace = inside.workspace();
        let moved = inside.0.with_extension("moved");
        fs::rename(&inside.0, &moved).unwrap();
        symlink(&outside.0, &inside.0).unwrap();
        assert_eq!(workspace.read("file").unwrap(), "inside");
        fs::remove_file(&inside.0).unwrap();
        inside.0 = moved;
    }
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn rejects_symlinks_and_hardlinks_at_every_path_component() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let other = Fixture::new();
        fs::write(other.0.join("outside"), "secret").unwrap();
        symlink(&other.0, f.0.join("escape")).unwrap();
        symlink(other.0.join("outside"), f.0.join("link")).unwrap();
        fs::hard_link(other.0.join("outside"), f.0.join("hard")).unwrap();
        assert!(anchored::Dir::open_root(&f.0.join("escape")).is_err());
        assert!(anchored::Dir::open_root(&other.0.join("outside")).is_err());
        let w = f.workspace();
        assert!(w.read("escape/outside").is_err());
        assert!(w.read("link").is_err());
        assert!(w.read("hard").is_err());
        assert!(w.list("escape").is_err());
        fs::create_dir(f.0.join("inside")).unwrap();
        fs::write(f.0.join("inside/file"), "safe").unwrap();
        assert_eq!(w.read("inside/file").unwrap(), "safe");
        for (actual, alias) in [
            ("token.key", "token.Key"), ("picK.txt", "pick.txt"), ("ReadMe", "README")
        ] {
            fs::write(f.0.join(actual), "synthetic\n").unwrap();
            assert!(w.read(alias).is_err());
            assert!(crate::edit::prepare(&w, &format!("*** Begin Patch\n*** Update File: {alias}\n@@\n-synthetic\n+changed\n*** End Patch")).is_err());
        }
        fs::create_dir(f.0.join("FoldEr")).unwrap();
        assert!(w.list("folder").is_err());
        // Only a case-folding filesystem makes "folder" an alias; elsewhere a
        // patch may create it as a distinct new directory.
        if f.0.join("folder").exists() {
            assert!(crate::edit::prepare(&w, "*** Begin Patch\n*** Add File: folder/new\n+synthetic\n*** End Patch").is_err());
        }
    }
}
