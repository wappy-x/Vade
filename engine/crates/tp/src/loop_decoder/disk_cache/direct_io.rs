//! Safe stdlib direct-I/O (`O_DIRECT`) streaming reader and sequential writer for NVMe prefix cache files.
//!
//! Design properties:
//! - 100% safe Rust (`0` `unsafe` blocks), zero third-party crates.
//! - Uses Linux `O_DIRECT` (`OpenOptionsExt::custom_flags`) with 4096-byte aligned memory buffers,
//!   file offsets, and transfer sizes so multi-gigabyte KV snapshots stream directly at full NVMe
//!   hardware bandwidth without bloating unified-RAM Linux page cache or triggering direct reclaim.
//! - Transparently falls back to buffered `FileExt::{read_exact_at, write_all_at}` on non-Linux hosts
//!   or filesystems that do not support `O_DIRECT` (`EINVAL` / `EOPNOTSUPP`, e.g. tmpfs).
use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

#[cfg(target_os = "linux")]
use std::{
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    sync::atomic::{AtomicBool, Ordering},
};

pub(super) const DIRECT_ALIGN: usize = 4096;
/// Aligned sequential direct-write staging block size (`4 MiB`), matching `CHUNK_BYTES`.
pub(super) const DIRECT_WRITE_BYTES: usize = 4 * 1024 * 1024;

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const O_DIRECT: i32 = 0x10000;
#[cfg(all(target_os = "linux", not(target_arch = "aarch64")))]
const O_DIRECT: i32 = 0x4000;

#[cfg(target_os = "linux")]
fn is_direct_unsupported(err: &io::Error) -> bool {
    matches!(err.raw_os_error(), Some(22 | 95))
}

/// Reusable `DIRECT_ALIGN`-byte-aligned byte buffer built in 100% safe Rust.
pub(super) struct AlignedBuf {
    raw: Vec<u8>,
    offset: usize,
    capacity: usize,
}

impl AlignedBuf {
    pub(super) fn new(min_capacity: usize) -> Self {
        let capacity = min_capacity.max(DIRECT_ALIGN).next_multiple_of(DIRECT_ALIGN);
        let raw = vec![0u8; capacity + DIRECT_ALIGN];
        let addr = raw.as_ptr() as usize;
        let offset = (DIRECT_ALIGN - (addr & (DIRECT_ALIGN - 1))) & (DIRECT_ALIGN - 1);
        debug_assert!((addr + offset).is_multiple_of(DIRECT_ALIGN));
        Self { raw, offset, capacity }
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn as_slice(&self, len: usize) -> &[u8] {
        &self.raw[self.offset..self.offset + len]
    }

    pub(super) fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        &mut self.raw[self.offset..self.offset + len]
    }
}

/// Thread-safe, stateless direct-I/O file reader supporting concurrent `read_at` calls across workers.
///
/// Opens an `O_DIRECT` file description against the exact same verified inode (`/proc/self/fd/<fd>`)
/// on Linux, falling back cleanly to `buffered.read_exact_at` if `O_DIRECT` is unsupported.
pub(super) struct DirectReader {
    buffered: File,
    #[cfg(target_os = "linux")]
    direct: Option<File>,
    #[cfg(target_os = "linux")]
    fallback: AtomicBool,
}

impl DirectReader {
    pub(super) fn from_file(buffered: File) -> Self {
        #[cfg(target_os = "linux")]
        let direct = OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(format!("/proc/self/fd/{}", buffered.as_raw_fd()))
            .ok();
        Self {
            buffered,
            #[cfg(target_os = "linux")]
            direct,
            #[cfg(target_os = "linux")]
            fallback: AtomicBool::new(false),
        }
    }

    /// Minimum `AlignedBuf` capacity guaranteed to hold any unaligned read of up to `max_chunk_bytes`.
    pub(super) fn buffer_capacity(max_chunk_bytes: usize) -> usize {
        max_chunk_bytes
            .next_multiple_of(DIRECT_ALIGN)
            .saturating_add(DIRECT_ALIGN)
    }

    /// Read exact `len` bytes at arbitrary byte offset `abs_off` into `buf` and return the slice.
    pub(super) fn read_exact_chunk<'a>(
        &self,
        abs_off: u64,
        len: usize,
        buf: &'a mut AlignedBuf,
    ) -> io::Result<&'a [u8]> {
        if len == 0 {
            return Ok(&[]);
        }
        #[cfg(target_os = "linux")]
        if !self.fallback.load(Ordering::Relaxed)
            && let Some(direct) = &self.direct
        {
            let align_mask = (DIRECT_ALIGN as u64) - 1;
            let aligned_off = abs_off & !align_mask;
            let skip = (abs_off - aligned_off) as usize;
            let need_end = skip + len;
            let aligned_len = need_end.next_multiple_of(DIRECT_ALIGN);
            if aligned_len <= buf.capacity() {
                let mut done = 0usize;
                let mut fell_back = false;
                while done < need_end {
                    // Maintain 4096-byte offset and memory alignment on every O_DIRECT retry.
                    if !done.is_multiple_of(DIRECT_ALIGN) {
                        fell_back = true;
                        break;
                    }
                    let slice = &mut buf.as_mut_slice(aligned_len)[done..aligned_len];
                    match direct.read_at(slice, aligned_off + done as u64) {
                        Ok(0) => {
                            fell_back = true;
                            break;
                        }
                        Ok(n) => done += n,
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(e) if is_direct_unsupported(&e) => {
                            self.fallback.store(true, Ordering::Relaxed);
                            fell_back = true;
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }
                if !fell_back && done >= need_end {
                    return Ok(&buf.as_slice(need_end)[skip..need_end]);
                }
            }
        }

        let dst = buf.as_mut_slice(len);
        self.buffered.read_exact_at(dst, abs_off)?;
        Ok(buf.as_slice(len))
    }
}

/// Sequential file writer that streams aligned `4 MiB` blocks via Linux `O_DIRECT`, writing at most
/// one final `< 4 KiB` unaligned tail block through a buffered handle before `fsync`.
pub(super) struct DirectSequentialWriter {
    path: PathBuf,
    file: File,
    direct: bool,
    buf: AlignedBuf,
    pending: usize,
    file_offset: u64,
}

impl DirectSequentialWriter {
    pub(super) fn create_new(path: &Path) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        let (file, direct) = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .custom_flags(O_DIRECT)
            .open(path)
        {
            Ok(f) => (f, true),
            Err(e) if is_direct_unsupported(&e) => (
                OpenOptions::new().create_new(true).write(true).open(path)?,
                false,
            ),
            Err(e) => return Err(e),
        };
        #[cfg(not(target_os = "linux"))]
        let (file, direct) = (
            OpenOptions::new().create_new(true).write(true).open(path)?,
            false,
        );
        Ok(Self {
            path: path.to_path_buf(),
            file,
            direct,
            buf: AlignedBuf::new(DIRECT_WRITE_BYTES),
            pending: 0,
            file_offset: 0,
        })
    }

    pub(super) fn write_all(&mut self, mut data: &[u8]) -> io::Result<()> {
        let cap = self.buf.capacity();
        while !data.is_empty() {
            let space = cap - self.pending;
            let take = data.len().min(space);
            self.buf.as_mut_slice(cap)[self.pending..self.pending + take]
                .copy_from_slice(&data[..take]);
            self.pending += take;
            data = &data[take..];
            if self.pending == cap {
                self.flush_aligned(cap)?;
                self.pending = 0;
            }
        }
        Ok(())
    }

    fn flush_aligned(&mut self, len: usize) -> io::Result<()> {
        debug_assert!(len.is_multiple_of(DIRECT_ALIGN));
        debug_assert!(self.file_offset.is_multiple_of(DIRECT_ALIGN as u64));
        if len == 0 {
            return Ok(());
        }
        if self.direct {
            let mut done = 0usize;
            let mut fell_back = false;
            while done < len {
                if !done.is_multiple_of(DIRECT_ALIGN) {
                    fell_back = true;
                    break;
                }
                let slice = &self.buf.as_slice(len)[done..len];
                match self.file.write_at(slice, self.file_offset + done as u64) {
                    Ok(0) => {
                        return Err(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "direct write returned 0 bytes",
                        ));
                    }
                    Ok(n) => done += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    #[cfg(target_os = "linux")]
                    Err(e) if is_direct_unsupported(&e) => {
                        fell_back = true;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            if !fell_back {
                self.file_offset += len as u64;
                return Ok(());
            }
            // Fallback to buffered handle for the remainder and all future writes.
            let buffered = OpenOptions::new().write(true).open(&self.path)?;
            buffered.write_all_at(&self.buf.as_slice(len)[done..len], self.file_offset + done as u64)?;
            self.file = buffered;
            self.direct = false;
            self.file_offset += len as u64;
            return Ok(());
        }

        self.file.write_all_at(self.buf.as_slice(len), self.file_offset)?;
        self.file_offset += len as u64;
        Ok(())
    }

    /// Flush all buffered bytes (direct aligned prefix + optional `< 4 KiB` buffered tail) and `fsync`.
    pub(super) fn finish(mut self) -> io::Result<()> {
        if self.pending > 0 {
            if self.direct {
                let aligned_part = self.pending & !(DIRECT_ALIGN - 1);
                if aligned_part > 0 {
                    self.flush_aligned(aligned_part)?;
                }
                let tail_len = self.pending - aligned_part;
                if tail_len > 0 {
                    // Close the O_DIRECT fd before opening the buffered fd for the final <4 KiB tail.
                    drop(self.file);
                    let buffered = OpenOptions::new().write(true).open(&self.path)?;
                    let tail = &self.buf.as_slice(self.pending)[aligned_part..self.pending];
                    buffered.write_all_at(tail, self.file_offset)?;
                    buffered.sync_all()?;
                    return Ok(());
                }
            } else {
                self.file
                    .write_all_at(self.buf.as_slice(self.pending), self.file_offset)?;
            }
        }
        self.file.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_file(tag: &str) -> PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "vade-direct-io-test-{tag}-{}-{seq}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn aligned_buf_always_page_aligned() {
        for size in [1usize, 17, 4095, 4096, 4097, 65_536, 4 * 1024 * 1024, 4 * 1024 * 1024 + 123] {
            let buf = AlignedBuf::new(size);
            assert!(buf.capacity() >= size);
            assert!(buf.capacity().is_multiple_of(DIRECT_ALIGN));
            assert!((buf.as_slice(size).as_ptr() as usize).is_multiple_of(DIRECT_ALIGN));
        }
    }

    #[test]
    fn direct_writer_and_reader_roundtrip_unaligned_and_multichunk() {
        // Test sizes: tiny (<4K), exact 4K, 4K+tail, and >2*DIRECT_WRITE_BYTES (9 MiB + 777 B).
        for total_len in [13usize, 4096, 4096 + 37, 9 * 1024 * 1024 + 777] {
            let path = temp_file(&format!("rt-{total_len}"));
            let _ = std::fs::remove_file(&path);
            let expected: Vec<u8> = (0..total_len)
                .map(|i| ((i.wrapping_mul(2_654_435_761) >> 11) ^ (i >> 3)) as u8)
                .collect();

            // Feed writer in awkward, unaligned pieces (mimicking prelude + header + chunks + footer).
            let mut writer = DirectSequentialWriter::create_new(&path).unwrap();
            let mut pos = 0usize;
            for step in [64usize, 137, 4093, 1_000_003, 4 * 1024 * 1024, 3_000_000] {
                if pos >= expected.len() {
                    break;
                }
                let take = step.min(expected.len() - pos);
                writer.write_all(&expected[pos..pos + take]).unwrap();
                pos += take;
            }
            if pos < expected.len() {
                writer.write_all(&expected[pos..]).unwrap();
            }
            writer.finish().unwrap();

            let actual = std::fs::read(&path).unwrap();
            assert_eq!(actual.len(), expected.len(), "file size mismatch for {total_len}");
            assert_eq!(actual, expected, "file contents mismatch for {total_len}");

            // Read back via concurrent DirectReader at unaligned offsets straddling 4 KiB pages and near EOF.
            let file = File::open(&path).unwrap();
            let reader = Arc::new(DirectReader::from_file(file));
            let chunk_size = 512 * 1024;
            let offsets: Vec<(usize, usize)> = if total_len <= 10_000 {
                vec![(0, total_len), (total_len / 3, total_len - total_len / 3)]
            } else {
                let base_off = 64 + 137; // intentionally unaligned payload start!
                let mut v = Vec::new();
                let mut off = base_off;
                while off < total_len - 44 {
                    let take = chunk_size.min(total_len - 44 - off);
                    v.push((off, take));
                    off += take;
                }
                v.push((total_len - 44, 44));
                v
            };

            std::thread::scope(|scope| {
                for (tid, (off, len)) in offsets.iter().copied().enumerate() {
                    let r = reader.clone();
                    let exp = &expected[off..off + len];
                    scope.spawn(move || {
                        let mut buf = AlignedBuf::new(DirectReader::buffer_capacity(len));
                        let got = r.read_exact_chunk(off as u64, len, &mut buf).unwrap();
                        assert_eq!(got, exp, "DirectReader mismatch on thread {tid} off={off} len={len}");
                    });
                }
            });

            let _ = std::fs::remove_file(&path);
        }
    }
}
