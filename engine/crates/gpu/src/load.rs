//! Batched shard loading. Offsets address the original, immutable file.
use std::{fs::File, io, ops::Range, os::unix::fs::FileExt,
    sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}, mpsc::{self, Receiver, SyncSender}},
    thread::{self, JoinHandle}};
#[cfg(target_os = "linux")]
use std::{fs::OpenOptions, os::{fd::AsRawFd, unix::fs::OpenOptionsExt}};

/// A contiguous row selection from a row-major BF16 or resident E4M3 tensor. Only these rows
/// occupy device memory; direct I/O may read boundary-sector padding.
pub struct WeightRows { pub file_offset: u64, pub rows: Range<usize>, pub columns: usize, pub packed: bool, pub fp8: bool }
impl WeightRows {
    pub(crate) fn range(&self) -> Result<(u64, usize), String> {
        if self.rows.is_empty() || self.columns == 0 || self.rows.len() > u32::MAX as usize
            || self.columns > u32::MAX as usize { return Err("invalid BF16 row range".into()); }
        if self.packed && (self.fp8 || self.columns % 16 != 0 || self.rows.len() % 8 != 0) {
            return Err("packed weights require K divisible by 16 and N divisible by the stored 8/64-row tile".into());
        }
        let stride = self.columns.checked_mul(if self.fp8 { 1 } else { 2 }).ok_or("row stride overflow")?;
        let start = self.rows.start.checked_mul(stride).ok_or("row offset overflow")?;
        let offset = self.file_offset.checked_add(start as u64).ok_or("file offset overflow")?;
        let bytes = self.rows.len().checked_mul(stride).ok_or("row size overflow")?;
        offset.checked_add(bytes as u64).ok_or("file range overflow")?;
        Ok((offset, bytes))
    }
}
pub struct WeightShard<'a> { pub file: &'a File, pub matrices: &'a [WeightRows] }
pub(crate) use crate::staging::{FILE_CHUNK as CHUNK, FILE_READERS as READERS};
/// Official bytes copied to a single arena, without packing or conversion.
/// For a column slice, `row_bytes` is the selected width and `row_stride`
/// is the original file width; `bytes` counts only selected bytes.
pub struct WeightRange {
    pub file_offset: u64, pub destination: usize, pub bytes: usize,
    pub row_bytes: usize, pub row_stride: usize,
}
pub struct WeightFile<'a> { pub file: &'a File, pub ranges: &'a [WeightRange] }
/// Physical checkpoint storage; the loader never widens these bytes.
#[derive(Clone, Copy)]
pub enum WeightDtype { Bf16, Fp8, F32 }
impl WeightDtype {
    pub fn bytes(self) -> usize { match self { Self::Bf16 => 2, Self::Fp8 => 1, Self::F32 => 4 } }
}
impl WeightRange {
    pub(crate) fn jobs(&self, source: &Arc<Source>) -> Result<Vec<(ReadJob, usize)>, String> {
        let strided = self.row_bytes != 0 && self.row_bytes != self.row_stride;
        if self.bytes == 0 || self.row_bytes != 0
            && (self.row_stride < self.row_bytes || self.bytes % self.row_bytes != 0) {
            return Err("invalid selected weight range".into());
        }
        let span = if strided {
            (self.bytes / self.row_bytes - 1).checked_mul(self.row_stride)
                .and_then(|n| n.checked_add(self.row_bytes)).ok_or("selected weight span overflow")?
        } else { self.bytes };
        if self.file_offset.checked_add(span as u64).is_none_or(|end| end > source.len) {
            return Err("selected weight range exceeds file".into());
        }
        let mut jobs = Vec::new();
        let mut done = 0;
        while done < self.bytes {
            let start = if strided { done / self.row_bytes * self.row_stride } else { done };
            let at = self.file_offset.checked_add(start as u64).ok_or("weight file offset overflow")?;
            let skip = (at % 4096) as usize;
            let (payload, span) = if strided {
                let rows = (CHUNK - skip).checked_sub(self.row_bytes)
                    .map(|n| n / self.row_stride + 1).unwrap_or(0)
                    .min((self.bytes - done) / self.row_bytes);
                if rows == 0 { return Err("selected weight row exceeds staging slot".into()); }
                (rows * self.row_bytes, (rows - 1) * self.row_stride + self.row_bytes)
            } else {
                let count = (CHUNK - skip).min(self.bytes - done);
                (count, count)
            };
            if at.checked_add(span as u64).is_none_or(|end| end > source.len) {
                return Err("selected weight range exceeds file".into());
            }
            let destination = self.destination.checked_add(done).ok_or("weight destination overflow")?;
            jobs.push((ReadJob { source: source.clone(), offset: at - skip as u64,
                count: (skip + span).div_ceil(4096) * 4096, skip, payload,
                columns: 0, fp8: false, packed8: false,
                row_bytes: if strided { self.row_bytes } else { 0 }, row_stride: self.row_stride }, destination));
            done += payload;
        }
        Ok(jobs)
    }
}
/// One slot read and the (slot offset, arena destination, bytes) copies it feeds.
pub(crate) type ReadGroup = (ReadJob, Vec<(usize, usize, usize)>);
/// Payload gap a merged read may skip over (unselected bytes are read, never copied).
pub(crate) const GAP: u64 = 256 * 1024;
/// Merge consecutive plain jobs of one file into reads of at most CHUNK bytes: tiny
/// tensors (4-byte global scales) and weights separated by unselected CSF bytes
/// cost one direct read per slot instead of one each. Each payload byte is copied
/// from the same file offset to the same destination; contiguous copies merge.
pub(crate) fn coalesce(jobs: Vec<(ReadJob, usize)>) -> Vec<ReadGroup> {
    let plain = |j: &ReadJob| j.row_bytes == 0 && j.columns == 0 && !j.fp8;
    let mut out: Vec<ReadGroup> = Vec::with_capacity(jobs.len());
    for (job, destination) in jobs {
        let start = job.offset + job.skip as u64;
        let end = start + job.payload as u64;
        if let Some((group, pieces)) = out.last_mut() {
            let &(at, to, len) = pieces.last().expect("a group holds a piece");
            let last = group.offset + (at + len) as u64;
            if plain(group) && plain(&job) && Arc::ptr_eq(&group.source, &job.source)
                && start >= last && start - last <= GAP && end - group.offset <= CHUNK as u64 {
                let at_new = (start - group.offset) as usize;
                if at + len == at_new && to + len == destination { pieces.last_mut().unwrap().2 += job.payload; }
                else { pieces.push((at_new, destination, job.payload)); }
                group.payload = (end - group.offset) as usize - group.skip;
                group.count = (group.skip + group.payload).div_ceil(4096) * 4096;
                continue;
            }
        }
        let piece = (job.skip, destination, job.payload);
        out.push((job, vec![piece]));
    }
    out
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const O_DIRECT: i32 = 0x10000;
#[cfg(all(target_os = "linux", not(target_arch = "aarch64")))]
const O_DIRECT: i32 = 0x4000;

pub(crate) struct Source {
    buffered: File, direct: Option<File>, fallback: AtomicBool, pub len: u64,
    pub read_bytes: AtomicU64, pub buffered_bytes: AtomicU64,
}
impl Source {
    /// Text-only mixed shards must not read alignment padding from vision/MTP.
    pub fn open_selected(file: &File, exact: bool) -> Result<Arc<Self>, String> {
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        #[cfg(target_os = "linux")]
        let direct = if exact { None } else { match OpenOptions::new().read(true).custom_flags(O_DIRECT)
            .open(format!("/proc/self/fd/{}", file.as_raw_fd())) {
            Ok(file) => Some(file),
            Err(e) if matches!(e.raw_os_error(), Some(22 | 95)) => None,
            Err(e) => return Err(e.to_string()),
        } };
        #[cfg(not(target_os = "linux"))]
        let direct = { let _ = exact; None };
        Ok(Arc::new(Self { buffered: file.try_clone().map_err(|e| e.to_string())?,
            direct, fallback: AtomicBool::new(false), len,
            read_bytes: AtomicU64::new(0), buffered_bytes: AtomicU64::new(0) }))
    }
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), String> {
        let remaining = self.len.checked_sub(offset).ok_or("weight read outside file")?;
        let needed = out.len().min(remaining.min(usize::MAX as u64) as usize);
        if needed == 0 { return Err("empty weight file read".into()); }
        if !self.fallback.load(Ordering::Relaxed) {
            if let Some(file) = &self.direct {
                loop {
                    match file.read_at(out, offset) {
                        Ok(n) => {
                            self.read_bytes.fetch_add(n as u64, Ordering::Relaxed);
                            if n >= needed { return Ok(()); }
                            break; // Complete short direct reads with read_exact_at.
                        }
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(e) if matches!(e.raw_os_error(), Some(22 | 95)) => { self.fallback.store(true, Ordering::Relaxed); break; }
                        Err(e) => return Err(e.to_string()),
                    }
                }
            }
        }
        self.buffered.read_exact_at(&mut out[..needed], offset).map_err(|e| e.to_string())?;
        self.read_bytes.fetch_add(needed as u64, Ordering::Relaxed);
        self.buffered_bytes.fetch_add(needed as u64, Ordering::Relaxed);
        Ok(())
    }
}
#[derive(Clone)]
pub(crate) struct ReadJob {
    pub source: Arc<Source>, pub offset: u64, pub count: usize,
    pub skip: usize, pub payload: usize, pub columns: usize, pub fp8: bool, pub packed8: bool,
    pub row_bytes: usize, pub row_stride: usize,
}
impl ReadJob {
    fn fill(&self, out: &mut [u8]) -> Result<(), String> {
        self.source.read(self.offset, out)?;
        if self.row_bytes != 0 {
            // Earlier destinations never overlap an unread later source row.
            // Only checkpoint bytes move; scales keep their global coordinates.
            for row in 1..self.payload / self.row_bytes {
                let start = self.skip + row * self.row_stride;
                out.copy_within(start..start + self.row_bytes, self.skip + row * self.row_bytes);
            }
        }
        let bytes = &mut out[self.skip..self.skip + self.payload];
        if self.fp8 { pack_fp8_rows(bytes, self.columns); }
        else if self.columns != 0 { pack_rows_layout(bytes, self.columns, self.packed8); }
        Ok(())
    }
}
// Packing permutes bounded 64x64 blocks. Readers transform their own
// pinned slot before publishing it; one row tile is the entire host scratch.
pub(crate) fn pack_rows_layout(bytes: &mut [u8], columns: usize, packed8: bool) {
    let rows = if !packed8 && columns % 64 == 0 {64} else {8};
    let mut tile = vec![0u8; columns * rows * 2];
    for block in bytes.chunks_exact_mut(tile.len()) {
        tile.copy_from_slice(block);
        for col in (0..columns).step_by(16) {
            for row in 0..rows {
                // Fixed-size BF16 pair shuffles let rustc vectorize the tile.
                let fragment = if rows == 64 { (col/64*32+row/8*4+col%64/16)*32+row%8*4 } else { col/16*32+row*4 };
                unsafe {
                    let src = tile.as_ptr().add((row * columns + col) * 2).cast::<[u32; 8]>();
                    let dst = block.as_mut_ptr().add(fragment * 8).cast::<[u32; 8]>();
                    let a = std::ptr::read_unaligned(src);
                    std::ptr::write_unaligned(dst, [a[0], a[4], a[1], a[5], a[2], a[6], a[3], a[7]]);
                }
            }
        }
    }
}
// Each lane loads both K pairs of one m16n8k16 weight fragment as one u32.
// This is a byte permutation, with one eight-row temporary per reader.
fn pack_fp8_rows(bytes: &mut [u8], columns: usize) {
    let mut tile = vec![0u8; columns * 8];
    for block in bytes.chunks_exact_mut(tile.len()) {
        tile.copy_from_slice(block);
        for col in (0..columns).step_by(16) {
            for row in 0..8 {
                unsafe {
                    let a = std::ptr::read_unaligned(tile.as_ptr().add(row * columns + col).cast::<[u16; 8]>());
                    let dst = block.as_mut_ptr().add(col / 16 * 128 + row * 16).cast::<[u32; 4]>();
                    std::ptr::write_unaligned(dst, [a[0] as u32 | ((a[4] as u32) << 16),
                        a[1] as u32 | ((a[5] as u32) << 16), a[2] as u32 | ((a[6] as u32) << 16),
                        a[3] as u32 | ((a[7] as u32) << 16)]);
                }
            }
        }
    }
}
/// Workers own access to their slot until ready; the caller returns it only
/// after its CUDA copy finishes. Drop joins before slots are freed.
pub(crate) struct Readers {
    send: Vec<SyncSender<(usize, ReadJob)>>, threads: Vec<JoinHandle<()>>,
    pub ready: Receiver<(usize, Result<(), String>)>,
}
impl Readers {
    /// Addresses must remain writable CHUNK-byte pinned slots until drop.
    pub unsafe fn new(addresses: impl Iterator<Item = usize>) -> Result<Self, String> {
        Self::with_buffers(addresses, 1)
    }
    /// Two slots per reader: one may be copied while the other is filled.
    pub unsafe fn double_buffered(addresses: impl Iterator<Item = usize>) -> Result<Self, String> {
        Self::with_buffers(addresses, 2)
    }
    unsafe fn with_buffers(addresses: impl Iterator<Item = usize>, buffers: usize) -> Result<Self, String> {
        let (ready, receiver) = mpsc::channel();
        let mut result = Self { send: Vec::new(), threads: Vec::new(), ready: receiver };
        let addresses = addresses.collect::<Vec<_>>();
        for (worker, addresses) in addresses.chunks(buffers).enumerate() {
            let addresses = addresses.to_vec();
            let (send, jobs) = mpsc::sync_channel::<(usize, ReadJob)>(buffers);
            let ready = ready.clone();
            result.send.extend((0..addresses.len()).map(|_| send.clone()));
            result.threads.push(thread::Builder::new().name(format!("weight-reader-{worker}")).spawn(move || {
                while let Ok((slot, job)) = jobs.recv() {
                    let address = addresses[slot - worker * buffers];
                    let out = std::slice::from_raw_parts_mut(address as *mut u8, job.count);
                    let result = job.fill(out);
                    if ready.send((slot, result)).is_err() { break; }
                }
            }).map_err(|e| format!("weight reader spawn: {e}"))?);
        }
        Ok(result)
    }
    pub fn send(&self, slot: usize, job: ReadJob) -> Result<(), String> {
        let span = if job.row_bytes == 0 { job.payload } else {
            if job.row_stride < job.row_bytes || job.payload % job.row_bytes != 0 {
                return Err("invalid strided weight read".into());
            }
            (job.payload / job.row_bytes).checked_sub(1)
                .and_then(|rows| rows.checked_mul(job.row_stride))
                .and_then(|bytes| bytes.checked_add(job.row_bytes)).ok_or("weight read size overflow")?
        };
        if job.count > CHUNK || job.payload == 0
            || job.skip.checked_add(span).is_none_or(|end| end > job.count)
            || job.offset.checked_add((job.skip + span) as u64).is_none_or(|end| end > job.source.len) {
            return Err("weight read outside staging slot or file".into());
        }
        self.send.get(slot).ok_or("invalid weight reader slot")?
            .send((slot, job)).map_err(|_| "weight reader stopped".into())
    }
}
impl Drop for Readers {
    fn drop(&mut self) {
        self.send.clear();
        for thread in self.threads.drain(..) { let _ = thread.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coalesced_reads_copy_the_same_bytes_to_the_same_destinations() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("vade-coalesce-{}", std::process::id()));
        let mut file = std::fs::File::create(&path).unwrap();
        let bytes = (0..(3 * CHUNK + 12345) as u64).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        drop(file);
        let file = std::fs::File::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let source = Source::open_selected(&file, false).unwrap();
        // Tiny scalars, a CSF-like gap, a range spanning slots, a large gap, the file end.
        let (mut ranges, mut at, mut destination) = (Vec::new(), 0usize, 0usize);
        for (len, gap) in [(4, 0), (4, 0), (4, 8), (819_200, 53_760), (819_200, 300_000), (CHUNK + 777, 1), (4, 0), (12, 4096 - 16)] {
            ranges.push(WeightRange { file_offset: at as u64, destination, bytes: len, row_bytes: 0, row_stride: 0 });
            at += len + gap; destination += (len + 255) & !255;
        }
        let tail = bytes.len() - 100;
        ranges.push(WeightRange { file_offset: tail as u64, destination, bytes: 100, row_bytes: 0, row_stride: 0 });
        let size = destination + 100;
        let jobs = ranges.iter().flat_map(|r| r.jobs(&source).unwrap()).collect::<Vec<_>>();
        let (mut expected, mut actual) = (vec![0u8; size], vec![0u8; size]);
        let mut slot = vec![0u8; CHUNK];
        for (job, to) in &jobs {
            job.fill(&mut slot[..job.count]).unwrap();
            expected[*to..*to + job.payload].copy_from_slice(&slot[job.skip..job.skip + job.payload]);
        }
        let groups = coalesce(jobs.clone());
        assert!(groups.len() < jobs.len());
        for (job, pieces) in &groups {
            assert!(job.count <= CHUNK && job.count % 4096 == 0 && job.offset % 4096 == 0);
            job.fill(&mut slot[..job.count]).unwrap();
            for &(at, to, len) in pieces { actual[to..to + len].copy_from_slice(&slot[at..at + len]); }
        }
        assert_eq!(actual, expected);
        for r in &ranges { assert_eq!(actual[r.destination..r.destination + r.bytes], bytes[r.file_offset as usize..r.file_offset as usize + r.bytes]); }
    }
    #[test]
    fn unaligned_direct_read_falls_back_without_changing_bytes() {
        use std::{fs::OpenOptions, io::Write};
        let path = std::env::temp_dir().join(format!("vade-direct-fallback-{}", std::process::id()));
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let bytes = (0..8192).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        let source = Source::open_selected(&file, false).unwrap();
        let mut out = [0u8; 7];
        source.read(13, &mut out).unwrap(); // EINVAL: offset/count violate direct-I/O alignment.
        assert_eq!(out, bytes[13..20]);
        source.read(4093, &mut out).unwrap();
        assert_eq!(out, bytes[4093..4100]);
    }
}
