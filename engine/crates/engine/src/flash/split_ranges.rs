//! Canonical K ranges of K-split BF16 shards, by device address. Two ranks keep
//! four ranges per shard and register nothing. Four ranks register a shard that
//! holds two of the two-rank split's ranges, so its kernels compute only those
//! and the rank fold rebuilds the two-rank sum tree.
//! Each entry is owned by a [`Registration`] held by the weights that own the
//! memory: drop removes the entry before the memory is freed. With no entry,
//! a launch reads one atomic and takes no lock.
use std::{collections::BTreeMap, sync::{RwLock, atomic::{AtomicUsize, Ordering}}};

static RANGES: RwLock<BTreeMap<u64, u32>> = RwLock::new(BTreeMap::new());
/// Entries in RANGES, written under its write lock.
static ENTRIES: AtomicUsize = AtomicUsize::new(0);

#[must_use = "dropping a Registration unregisters its shard at once"]
#[derive(Debug)]
pub struct Registration { weight: u64, ranges: u32 }

/// Register `ranges` (kernel_params::shard_ranges) for the shard at `weight`.
/// 0 keeps four ranges and needs no entry; 2 is a four-rank shard. The kernels
/// strip the ranges from the cols word only for 2, so other counts are refused.
pub fn register(weight: u64, ranges: u32) -> Result<Option<Registration>, String> {
    match ranges {
        0 => return Ok(None),
        2 => {}
        _ => return Err(format!("unsupported canonical K range count {ranges}")),
    }
    let mut map = RANGES.write().unwrap_or_else(|e| e.into_inner());
    map.insert(weight, ranges);
    ENTRIES.store(map.len(), Ordering::Relaxed);
    Ok(Some(Registration { weight, ranges }))
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut map = RANGES.write().unwrap_or_else(|e| e.into_inner());
        if map.get(&self.weight) == Some(&self.ranges) { map.remove(&self.weight); }
        ENTRIES.store(map.len(), Ordering::Relaxed);
    }
}

/// Whether any shard is registered; false below four ranks.
pub fn active() -> bool { ENTRIES.load(Ordering::Relaxed) != 0 }

/// The canonical ranges of the matrix at `weight`; 0 (four) when it is not a registered shard.
pub fn get(weight: u64) -> u32 {
    if weight == 0 || !active() { return 0; }
    RANGES.read().unwrap_or_else(|e| e.into_inner()).get(&weight).copied().unwrap_or(0)
}
/// The ranges of a BF16 launch over `first` (and `second`, when it has rows), with
/// `cols` made the launch's cols word. With no entry: 0 and `cols` unchanged.
pub fn launch(cols: &mut u32, first: u64, second: Option<u64>) -> Result<u32, String> {
    if !active() { return Ok(0); }
    let ranges = get(first);
    if second.is_some_and(|weight| get(weight) != ranges) { return Err("paired BF16 matrices need the same canonical K ranges".into()); }
    *cols = spark_gpu_probe::kernel_params::bf16_cols_word(*cols, ranges);
    Ok(ranges)
}

#[cfg(test)]
#[test]
fn registration_lives_with_its_owner() {
    // Fake addresses unique to this test: the registry is process-global.
    let base = 0x7b00_0000_0000;
    assert!(register(base, 0).unwrap().is_none());
    assert!(register(base, 1).is_err() && register(base, 4).is_err() && !active());
    let owned = [register(base, 2).unwrap(), register(base + 0x1000, 2).unwrap()];
    assert!(active());
    assert_eq!((get(base), get(base + 0x1000), get(base + 0x2000), get(0)), (2, 2, 0, 0));
    drop(owned);
    assert_eq!((get(base), get(base + 0x1000), active()), (0, 0, false));
}
