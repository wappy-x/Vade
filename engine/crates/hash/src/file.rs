//! Positioned reads from a held inode. Production Linux hashing requires direct
//! I/O except for a bounded metadata file/tail; portable hosts report buffering.
use std::{fs::File, io::{self, Read}, os::unix::fs::FileExt};

const BLOCK: usize = 1 << 20;
const ALIGN: usize = 4096;
const BUFFERED_LIMIT: u64 = 65536;

trait Source {
    fn direct_read(&self, out: &mut [u8], offset: u64) -> io::Result<usize>;
    fn buffered_read(&self, out: &mut [u8], offset: u64) -> io::Result<usize>;
}

pub(super) struct FileSource<'a> { file: &'a File, direct: Option<File> }
impl Source for FileSource<'_> {
    fn direct_read(&self, out: &mut [u8], offset: u64) -> io::Result<usize> {
        self.direct.as_ref().expect("direct reader selected").read_at(out, offset)
    }
    fn buffered_read(&self, out: &mut [u8], offset: u64) -> io::Result<usize> {
        self.file.read_at(out, offset)
    }
}

pub(super) struct Reader<S> {
    source: S,
    direct: bool,
    buffered_limit: Option<u64>,
    size: u64,
    buffer: Vec<u8>,
    aligned: usize,
    at: usize,
    end: usize,
    offset: u64,
    direct_bytes: u64,
    buffered_bytes: u64,
}

impl<'a> Reader<FileSource<'a>> {
    pub(super) fn new(file: &'a File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "hash input is not a regular file"));
        }
        let direct = open_direct(file)?;
        let has_direct = direct.is_some();
        Self::from_source(FileSource { file, direct }, metadata.len(), has_direct,
            if cfg!(target_os = "linux") { Some(BUFFERED_LIMIT) } else { None })
    }
}

impl<S> Reader<S> {
    fn from_source(source: S, size: u64, direct: bool, buffered_limit: Option<u64>) -> io::Result<Self> {
        if !direct && buffered_limit.is_some_and(|limit| size > limit) { return Err(cache_policy_error()); }
        // Safe aligned subslice; caller descriptors and flags are untouched.
        let buffer = if direct { vec![0; BLOCK + ALIGN - 1] } else { Vec::new() };
        let aligned = if buffer.is_empty() { 0 } else { buffer.as_ptr().align_offset(ALIGN) };
        Ok(Self { source, direct, buffered_limit, size, buffer, aligned,
            at: 0, end: 0, offset: 0, direct_bytes: 0, buffered_bytes: 0 })
    }

    pub(super) fn finish(&self, sha256: String) -> super::FileDigest {
        let io_mode = match (self.direct_bytes > 0, self.buffered_bytes > 0, self.buffered_limit) {
            (false, false, _) => "empty",
            (true, false, _) => "direct",
            (true, true, _) => "direct-with-buffered-tail",
            (false, true, Some(_)) => "buffered-bounded",
            (false, true, None) => "buffered-portable",
        };
        super::FileDigest { sha256, direct_bytes: self.direct_bytes, buffered_bytes: self.buffered_bytes, io_mode }
    }
}

impl<S: Source> Read for Reader<S> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() { return Ok(0); }
        if self.at == self.end {
            // Read the captured length; model callers check the held inode and
            // pathname identity before/after to reject truncation or growth.
            if self.offset == self.size { return Ok(0); }
            if self.direct {
                match self.source.direct_read(&mut self.buffer[self.aligned..self.aligned + BLOCK], self.offset) {
                    Ok(n) => {
                        check_read(n, self.size - self.offset)?;
                        self.at = 0;
                        self.end = n;
                        self.offset += n as u64;
                        self.direct_bytes += n as u64;
                    }
                    Err(error) if unsupported(&error) => self.direct = false,
                    Err(error) => return Err(error),
                }
            }
            if !self.direct {
                let remaining = self.size - self.offset;
                if self.buffered_limit.is_some_and(|limit| remaining > limit.saturating_sub(self.buffered_bytes)) {
                    return Err(cache_policy_error());
                }
                let wanted = out.len().min(remaining.min(usize::MAX as u64) as usize);
                let n = self.source.buffered_read(&mut out[..wanted], self.offset)?;
                check_read(n, remaining)?;
                self.offset += n as u64;
                self.buffered_bytes += n as u64;
                return Ok(n);
            }
        }
        let n = out.len().min(self.end - self.at);
        out[..n].copy_from_slice(&self.buffer[self.aligned + self.at..self.aligned + self.at + n]);
        self.at += n;
        Ok(n)
    }
}

fn check_read(n: usize, remaining: u64) -> io::Result<()> {
    if n == 0 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "hash file truncated")); }
    if n as u64 > remaining { return Err(io::Error::new(io::ErrorKind::InvalidData, "hash file length changed")); }
    Ok(())
}
fn cache_policy_error() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "checkpoint hashing requires direct I/O on this filesystem; buffered remainder exceeds 65536 bytes")
}
fn unsupported(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(22 | 95)) // EINVAL / EOPNOTSUPP only.
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn open_direct(file: &File) -> io::Result<Option<File>> {
    use std::os::{fd::AsRawFd, unix::fs::{MetadataExt, OpenOptionsExt}};
    #[cfg(target_arch = "aarch64")]
    const O_DIRECT: i32 = 0x10000;
    #[cfg(target_arch = "x86_64")]
    const O_DIRECT: i32 = 0x4000;
    let path = format!("/proc/self/fd/{}", file.as_raw_fd());
    match std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path) {
        Ok(direct) => {
            let (original, reopened) = (file.metadata()?, direct.metadata()?);
            if (original.dev(), original.ino()) != (reopened.dev(), reopened.ino()) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "hash descriptor identity changed"));
            }
            Ok(Some(direct))
        }
        Err(error) if unsupported(&error) || matches!(error.raw_os_error(), Some(2 | 13 | 38)) => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn open_direct(_: &File) -> io::Result<Option<File>> { Ok(None) }

#[cfg(test)]
#[path = "file_tests.rs"]
mod tests;
