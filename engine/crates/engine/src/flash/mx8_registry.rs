//! MXFP8 twin registry: (BF16 matrix address, rows) -> (twin address, rows, cols). Decode launches
//! (blocks::mx8_weight) and the opt-in drafter paths read a registered twin instead of the
//! BF16 copy. Every entry is owned by a [`Registration`], held next to the twin's memory:
//! dropping the owner (model teardown, or unwinding a partial load) removes the entry
//! before the memory is freed, so a later allocation that reuses a BF16 address never
//! finds a stale twin.
use std::{collections::BTreeMap, sync::RwLock};

/// Keyed by address and rows: a twin of a matrix's leading rows (the tensor split's half on
/// rank 0) shares the matrix's address and must not replace the whole matrix's twin.
static MX8: RwLock<BTreeMap<(u64, usize), (u64, usize, usize)>> = RwLock::new(BTreeMap::new());

/// One registry entry. Drop removes it, but only while the entry still names this twin:
/// a newer registration of the same BF16 address (a reload) is left in place.
#[must_use = "dropping a Registration unregisters its twin at once"]
#[derive(Debug)]
pub struct Registration {
    bf16: u64,
    rows: usize,
    twin: u64,
}

/// Register `twin` as the MXFP8 twin of the BF16 matrix at `bf16`, shaped [rows][cols].
pub fn register(bf16: u64, twin: u64, rows: usize, cols: usize) -> Registration {
    MX8.write().unwrap_or_else(|e| e.into_inner()).insert((bf16, rows), (twin, rows, cols));
    Registration { bf16, rows, twin }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut map = MX8.write().unwrap_or_else(|e| e.into_inner());
        if map.get(&(self.bf16, self.rows)).is_some_and(|&(twin, _, _)| twin == self.twin) {
            map.remove(&(self.bf16, self.rows));
        }
    }
}

/// The registered twin of the first `rows` rows of `bf16`: (twin address, rows, cols).
pub fn get(bf16: u64, rows: usize) -> Option<(u64, usize, usize)> {
    MX8.read().unwrap_or_else(|e| e.into_inner()).get(&(bf16, rows)).copied()
}

/// Whether `bf16` has a registered twin of exactly [rows][cols].
pub fn has(bf16: u64, rows: usize, cols: usize) -> bool {
    get(bf16, rows).is_some_and(|(_, r, c)| (r, c) == (rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The addresses are fake and unique to this test (the registry is process-global).
    fn load(base: u64, count: u64, fail_at: Option<u64>) -> Result<Vec<Registration>, String> {
        // Mirrors the loaders: twins are built one by one into a Vec that the model keeps;
        // an error part-way returns early and drops the twins built so far.
        let mut owned = Vec::new();
        for i in 0..count {
            if fail_at == Some(i) {
                return Err(format!("twin {i} failed"));
            }
            owned.push(register(base + i * 0x1000, base + 0x10_0000 + i * 0x1000, 8, 64));
        }
        Ok(owned)
    }

    #[test]
    fn load_drop_reload_leaves_no_stale_twin() {
        let base = 0x7a00_0000_0000;
        let first = load(base, 7, None).unwrap();
        assert!((0..7).all(|i| has(base + i * 0x1000, 8, 64)));
        drop(first);
        assert!((0..7).all(|i| get(base + i * 0x1000, 8).is_none()), "teardown left an entry");
        let again = load(base, 7, None).unwrap();
        assert!((0..7).all(|i| get(base + i * 0x1000, 8) == Some((base + 0x10_0000 + i * 0x1000, 8, 64))));
        drop(again);
        assert!((0..7).all(|i| get(base + i * 0x1000, 8).is_none()), "reload teardown left an entry");
    }

    #[test]
    fn partial_load_error_unregisters_built_twins() {
        let base = 0x7b00_0000_0000;
        assert!(load(base, 7, Some(4)).is_err());
        assert!((0..7).all(|i| get(base + i * 0x1000, 8).is_none()), "partial load left an entry");
    }

    #[test]
    fn late_drop_keeps_a_newer_registration() {
        let (bf16, old, new) = (0x7c00_0000_0000, 0x7c00_0010_0000, 0x7c00_0020_0000);
        let first = register(bf16, old, 8, 64);
        let second = register(bf16, new, 8, 64);
        drop(first);
        assert_eq!(get(bf16, 8), Some((new, 8, 64)));
        assert!(has(bf16, 8, 64) && !has(bf16, 16, 64));
        drop(second);
        assert_eq!(get(bf16, 8), None);
    }

    /// A twin of a matrix's leading half (same address, fewer rows) lives next to the whole
    /// matrix's twin; dropping either leaves the other.
    #[test]
    fn a_leading_half_twin_keeps_the_whole_twin() {
        let (bf16, whole, half) = (0x7d00_0000_0000, 0x7d00_0010_0000, 0x7d00_0020_0000);
        let w = register(bf16, whole, 16, 64);
        let h = register(bf16, half, 8, 64);
        assert_eq!((get(bf16, 16), get(bf16, 8)), (Some((whole, 16, 64)), Some((half, 8, 64))));
        drop(h);
        assert!(has(bf16, 16, 64) && !has(bf16, 8, 64));
        drop(w);
        assert_eq!(get(bf16, 16), None);
    }
}
