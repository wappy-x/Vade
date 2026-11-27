use super::*;
use std::{cell::RefCell, collections::VecDeque, io::{Seek, SeekFrom, Write}, rc::Rc, sync::atomic::{AtomicUsize, Ordering}};

#[derive(Clone, Copy)]
enum Step { Bytes(usize), Error(i32) }
type Calls = Rc<RefCell<Vec<(bool, u64, usize)>>>;
struct Fake {
    data: Vec<u8>,
    steps: RefCell<VecDeque<Step>>,
    buffered_steps: RefCell<VecDeque<Step>>,
    calls: Calls,
}
impl Fake {
    fn read(&self, direct: bool, out: &mut [u8], offset: u64) -> io::Result<usize> {
        self.calls.borrow_mut().push((direct, offset, out.len()));
        let step = if direct { &self.steps } else { &self.buffered_steps }.borrow_mut().pop_front();
        let max = match step { Some(Step::Error(code)) => return Err(io::Error::from_raw_os_error(code)),
            Some(Step::Bytes(n)) => n, None => out.len() };
        let n = out.len().min(max).min(self.data.len().saturating_sub(offset as usize));
        out[..n].copy_from_slice(&self.data[offset as usize..offset as usize + n]);
        Ok(n)
    }
}
impl Source for Fake {
    fn direct_read(&self, out: &mut [u8], offset: u64) -> io::Result<usize> {
        assert_eq!(out.as_ptr().align_offset(ALIGN), 0);
        assert_eq!(out.len(), BLOCK);
        self.read(true, out, offset)
    }
    fn buffered_read(&self, out: &mut [u8], offset: u64) -> io::Result<usize> { self.read(false, out, offset) }
}
fn fake(size: usize, steps: &[Step], buffered: &[Step]) -> (Fake, Calls, String) {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let data = (0..size).map(|i| ((i * 13 + i / 17) % 256) as u8).collect::<Vec<_>>();
    let expected = super::super::digest(&data);
    (Fake { data, steps: RefCell::new(steps.iter().copied().collect()),
        buffered_steps: RefCell::new(buffered.iter().copied().collect()), calls: calls.clone() }, calls, expected)
}

#[test]
fn direct_prefix_unsupported_tail_resumes_exactly_and_reports_actual_bytes() {
    for code in [22, 95] {
        let size = BLOCK + 73;
        let (source, calls, expected) = fake(size, &[Step::Bytes(BLOCK), Step::Error(code)], &[]);
        let mut reader = Reader::from_source(source, size as u64, true, Some(BUFFERED_LIMIT)).unwrap();
        assert_eq!(super::super::digest_reader(&mut reader).unwrap(), expected);
        let stats = reader.finish(expected);
        assert_eq!(stats.io_mode, "direct-with-buffered-tail");
        assert_eq!((stats.direct_bytes, stats.buffered_bytes), (BLOCK as u64, 73));
        assert_eq!(calls.borrow().as_slice(), &[(true, 0, BLOCK), (true, BLOCK as u64, BLOCK), (false, BLOCK as u64, 73)]);
    }
}

#[test]
fn unavailable_direct_and_large_fallback_never_populate_a_whole_shard() {
    let size = BLOCK + BUFFERED_LIMIT as usize + 1;
    let (source, calls, _) = fake(size, &[], &[]);
    assert!(matches!(Reader::from_source(source, size as u64, false, Some(BUFFERED_LIMIT)), Err(e) if e.kind() == io::ErrorKind::Unsupported));
    assert!(calls.borrow().is_empty());
    for code in [22, 95] {
        let (source, calls, _) = fake(size, &[Step::Bytes(BLOCK), Step::Error(code)], &[]);
        let mut reader = Reader::from_source(source, size as u64, true, Some(BUFFERED_LIMIT)).unwrap();
        assert_eq!(super::super::digest_reader(&mut reader).unwrap_err().kind(), io::ErrorKind::Unsupported);
        assert!(calls.borrow().iter().all(|(direct, _, _)| *direct));
        assert_eq!(reader.buffered_bytes, 0);
    }
}

#[test]
fn interruptions_retry_same_direct_and_buffered_offsets() {
    let size = BLOCK + 73;
    let (source, calls, expected) = fake(size,
        &[Step::Error(4), Step::Bytes(BLOCK), Step::Error(4), Step::Error(22)],
        &[Step::Error(4), Step::Bytes(7), Step::Error(4)]);
    let mut reader = Reader::from_source(source, size as u64, true, Some(BUFFERED_LIMIT)).unwrap();
    assert_eq!(super::super::digest_reader(&mut reader).unwrap(), expected);
    let offsets = calls.borrow().iter().map(|(d, at, _)| (*d, *at)).collect::<Vec<_>>();
    assert_eq!(offsets, [(true, 0), (true, 0), (true, BLOCK as u64), (true, BLOCK as u64),
        (false, BLOCK as u64), (false, BLOCK as u64), (false, BLOCK as u64 + 7), (false, BLOCK as u64 + 7)]);
    assert_eq!(reader.buffered_bytes, 73);
}

#[test]
fn real_io_failures_propagate_without_fallback() {
    for prefix in [0, BLOCK] {
        let steps = if prefix == 0 { vec![Step::Error(5)] } else { vec![Step::Bytes(BLOCK), Step::Error(5)] };
        let (source, calls, _) = fake(BLOCK + 73, &steps, &[]);
        let mut reader = Reader::from_source(source, (BLOCK + 73) as u64, true, Some(BUFFERED_LIMIT)).unwrap();
        assert_eq!(super::super::digest_reader(&mut reader).unwrap_err().raw_os_error(), Some(5));
        assert!(calls.borrow().iter().all(|(direct, _, _)| *direct));
    }
    let (source, _, _) = fake(73, &[], &[Step::Error(5)]);
    let mut reader = Reader::from_source(source, 73, false, Some(BUFFERED_LIMIT)).unwrap();
    assert_eq!(super::super::digest_reader(&mut reader).unwrap_err().raw_os_error(), Some(5));
}

#[test]
fn buffered_budget_is_total_and_portable_buffering_is_explicit() {
    for (size, limit, mode) in [(65536, Some(BUFFERED_LIMIT), "buffered-bounded"),
        (BLOCK + 73, None, "buffered-portable")] {
        let (source, _, expected) = fake(size, &[], &[Step::Bytes(7), Step::Error(4)]);
        let mut reader = Reader::from_source(source, size as u64, false, limit).unwrap();
        assert_eq!(super::super::digest_reader(&mut reader).unwrap(), expected);
        assert_eq!(reader.finish(expected).io_mode, mode);
        assert_eq!(reader.buffered_bytes, size as u64);
    }
    // Growth cannot turn a bounded metadata read into an unbounded cache fill.
    let (source, calls, _) = fake(BLOCK, &[], &[]);
    let mut reader = Reader::from_source(source, BUFFERED_LIMIT, false, Some(BUFFERED_LIMIT)).unwrap();
    super::super::digest_reader(&mut reader).unwrap();
    assert_eq!(reader.buffered_bytes, BUFFERED_LIMIT);
    assert_eq!(calls.borrow().len(), 1);
}

#[test]
fn truncated_or_growing_direct_file_is_rejected() {
    let (source, _, _) = fake(73, &[Step::Bytes(7), Step::Bytes(0)], &[]);
    let mut reader = Reader::from_source(source, 73, true, Some(BUFFERED_LIMIT)).unwrap();
    assert_eq!(super::super::digest_reader(&mut reader).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    let (source, _, _) = fake(100, &[], &[]);
    let mut reader = Reader::from_source(source, 73, true, Some(BUFFERED_LIMIT)).unwrap();
    assert_eq!(super::super::digest_reader(&mut reader).unwrap_err().kind(), io::ErrorKind::InvalidData);
}

#[test]
fn real_file_keeps_offsets_and_open_inode_and_evidences_platform_io() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!("vade-file-hash-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let mut file = File::options().read(true).write(true).create_new(true).open(&path).unwrap();
    let data = (0..(BLOCK * 2 + 73)).map(|i| ((i * 13 + i / 17) % 256) as u8).collect::<Vec<_>>();
    for len in [0, 1, 63, 64, 4095, 4096, 4097, BLOCK - 1, BLOCK, BLOCK + 1, data.len()] {
        file.set_len(0).unwrap(); file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&data[..len]).unwrap(); file.sync_all().unwrap();
        file.seek(SeekFrom::Start(7)).unwrap();
        let stats = super::super::digest_file_with_stats(&file).unwrap();
        assert_eq!(stats.sha256, super::super::digest(&data[..len]));
        assert_eq!(stats.direct_bytes + stats.buffered_bytes, len as u64);
        assert_eq!(file.stream_position().unwrap(), 7);
        if cfg!(target_os = "linux") && len >= BLOCK {
            assert!(stats.direct_bytes >= BLOCK as u64, "real Linux direct reads are required");
            assert!(stats.buffered_bytes <= BUFFERED_LIMIT);
        } else if len > 0 && !cfg!(target_os = "linux") { assert_eq!(stats.io_mode, "buffered-portable"); }
        eprintln!("hash_file_io mode={} direct_bytes={} buffered_bytes={}", stats.io_mode, stats.direct_bytes, stats.buffered_bytes);
    }
    let moved = path.with_extension("moved");
    std::fs::rename(&path, &moved).unwrap(); std::fs::write(&path, b"replacement").unwrap();
    assert_eq!(super::super::digest_file(&file).unwrap(), super::super::digest(&data));
    assert_eq!(file.stream_position().unwrap(), 7);
    drop(file); std::fs::remove_file(path).unwrap(); std::fs::remove_file(moved).unwrap();
}
