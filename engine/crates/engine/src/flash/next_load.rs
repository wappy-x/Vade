//! Load-time host helpers: shard files opened once, an ordered parallel map and the load digest.
//! Workers only read checkpoint bytes and run pure expansions; every device
//! allocation, upload and launch stays on the loading thread in its own order.
use engine_model::flash::Checkpoint;
use std::{collections::BTreeMap, fs::File, os::unix::fs::FileExt, path::Path};
type Result<T> = std::result::Result<T, String>;

/// Every shard opened once, with `Checkpoint::read_range`'s checks: inside the
/// checkpoint root and still the inspected length. Reads are positioned (`pread`),
/// so one handle serves every worker.
pub(super) struct Shards<'c> {
    checkpoint: &'c Checkpoint,
    files: BTreeMap<&'c str, File>,
}
impl<'c> Shards<'c> {
    pub(super) fn open(checkpoint: &'c Checkpoint, root: &Path) -> Result<Self> {
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let mut files = BTreeMap::new();
        for (name, header) in &checkpoint.shards {
            let path = root.join(name).canonicalize().map_err(|e| e.to_string())?;
            if !path.starts_with(&root) {
                return Err("shard escapes checkpoint".into());
            }
            let file = File::open(path).map_err(|e| e.to_string())?;
            if file.metadata().map_err(|e| e.to_string())?.len() != header.file_bytes {
                return Err("shard changed after inspection".into());
            }
            files.insert(name.as_str(), file);
        }
        Ok(Self { checkpoint, files })
    }
    /// The whole stored tensor, as `read_all` returns it.
    pub(super) fn read(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.checkpoint.tensor(name).map_err(|e| e.to_string())?;
        let shard = &self.checkpoint.index.weight_map[name];
        let file = self.files.get(shard.as_str()).ok_or("tensor shard was not opened")?;
        let mut out = vec![0u8; tensor.byte_len() as usize];
        file.read_exact_at(&mut out, self.checkpoint.shards[shard].data_start + tensor.start)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
}

/// Worker count for load-time host work (reads queue in parallel on NVMe).
pub(super) fn threads() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get()).clamp(1, 16)
}

/// `items.iter().map(f)` on `threads()` workers; results keep item order.
pub(super) fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let next = AtomicUsize::new(0);
    let workers = threads().min(items.len());
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles = (0..workers).map(|_| scope.spawn(|| {
            let mut out = Vec::new();
            loop {
                let i = next.fetch_add(1, Relaxed);
                let Some(item) = items.get(i) else { break };
                out.push((i, f(item)));
            }
            out
        })).collect::<Vec<_>>();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    });
    done.sort_unstable_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

/// `f(i, chunk_i)` over `data.chunks_mut(chunk)` on the same workers, in chunk order.
pub(super) fn parallel_chunks<R: Send>(data: &mut [u8], chunk: usize, f: impl Fn(usize, &mut [u8]) -> R + Sync) -> Vec<R> {
    let pieces = data.chunks_mut(chunk).map(std::sync::Mutex::new).collect::<Vec<_>>();
    let index = (0..pieces.len()).collect::<Vec<_>>();
    parallel_map(&index, |&i| f(i, &mut pieces[i].lock().unwrap_or_else(|e| e.into_inner())))
}

/// Order-dependent 64-bit digest of loaded device bytes for VADE_LOAD_DIGEST
/// receipts (not cryptographic): 1 MiB blocks hashed on the workers, folded in order.
pub(super) struct Digest(pub(super) u64, pub(super) u64);
impl Digest {
    pub(super) fn new() -> Self { Self(0xcbf2_9ce4_8422_2325, 0) }
    pub(super) fn bytes(&mut self, data: &[u8]) {
        let blocks = data.chunks(1 << 20).collect::<Vec<_>>();
        for h in parallel_map(&blocks, |b| block(b)) {
            self.0 = (self.0 ^ h).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(23);
        }
        self.1 += data.len() as u64;
    }
    pub(super) fn words(&mut self, values: &[f32]) {
        self.bytes(&values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>());
    }
}
fn block(b: &[u8]) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15 ^ b.len() as u64;
    let (words, rest) = b.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    }
    for &x in rest { h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3); }
    h
}

#[cfg(test)]
mod tests {
    #[test]
    fn parallel_map_keeps_item_order() {
        let items = (0..1000u64).collect::<Vec<_>>();
        let out = super::parallel_map(&items, |&x| x * x + 7);
        assert_eq!(out, items.iter().map(|&x| x * x + 7).collect::<Vec<_>>());
        assert!(super::parallel_map(&[] as &[u8], |&x| x).is_empty());
        let mut data = vec![0u8; 1000];
        let lens = super::parallel_chunks(&mut data, 7, |i, c| { c.fill(i as u8); c.len() });
        assert_eq!(lens.iter().sum::<usize>(), 1000);
        assert!(data.iter().enumerate().all(|(k, &v)| v == (k / 7) as u8));
        let (mut a, mut b, mut c) = (super::Digest::new(), super::Digest::new(), super::Digest::new());
        let bytes = (0..3_000_000u64).map(|i| ((i * 7919) >> 3) as u8).collect::<Vec<_>>();
        a.bytes(&bytes); b.bytes(&bytes);
        let mut flipped = bytes.clone(); flipped[2_500_000] ^= 1; c.bytes(&flipped);
        assert_eq!((a.0, a.1), (b.0, b.1));
        assert_ne!(a.0, c.0);
    }
}
