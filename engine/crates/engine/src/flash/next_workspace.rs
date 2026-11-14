//! Next-only activation ownership and reservation, independent of CUDA.
//! HC normalization and GDN input projections have disjoint stream lifetimes.
use std::ops::Range;
type Result<T> = std::result::Result<T, String>;

pub(crate) fn bytes(lengths: &[usize]) -> usize {
    lengths.iter().map(|n| (n * 4).div_ceil(64) * 64).sum()
}
/// Next HC-up computes gates in registers. Only injection survives attention.
/// HC injection words per row: one gate per residual stream.
pub(crate) const HC_INJECT_WORDS: usize = 4;
pub(crate) fn hc_lengths(width: usize, low: usize, hidden: usize, rows: usize) -> [usize; 4] {
    [width * rows, low * rows, HC_INJECT_WORDS * rows, hidden]
}
/// qkv retains a slice of HC normalization; Next always projects to work.block.
/// Keep one-word sentinels for the generic scratch structure, never kernel data.
pub(crate) fn gdn_owned_lengths(mut lengths: [usize; 20]) -> [usize; 20] {
    lengths[0] = 1;
    lengths[12] = 1;
    lengths
}
/// Compare against the original admission formula, including its unaligned HC
/// injection term. Thus even a short row count cannot overstate freed funding.
pub(crate) fn saved_bytes(hidden: usize, heads: usize, keys: usize, rows: usize) -> usize {
    let width = 4 * hidden;
    let old_hc = rows * (2 * width + 320 + 4) * 4 + hidden * 4;
    let hc = bytes(&hc_lengths(width, 320, hidden, rows));
    let gdn = spark_gpu_probe::next_math::gdn_lengths(hidden, heads, keys, rows);
    old_hc + bytes(&gdn) - hc - bytes(&gdn_owned_lengths(gdn))
}
/// An owning view, never a borrowed device address. This preserves the backing
/// arena if either scratch object drops first, including graph error cleanup.
pub(crate) trait View: Sized {
    fn slice(&self, range: Range<usize>) -> Result<Self>;
}
#[cfg(target_os = "linux")]
impl View for spark_gpu_probe::driver::DeviceVector<'_> {
    fn slice(&self, range: Range<usize>) -> Result<Self> { self.slice(range) }
}
pub(crate) fn qkv_view<V: View>(normalized: &V, words: usize) -> Result<V> {
    normalized.slice(0..words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn reservation_covers_owned_vectors_and_does_not_spend_pool_savings_twice() {
        let head = 655_564_800 + 248_320 * 8 + 16 * 40_960;
        for rows in [1, 2, 3, 4, 8, 16, 31, 63, 64, 65, 128, 4096, 8192] {
            let original = spark_gpu_probe::next_math::gdn_lengths(2560, 48, 16, rows);
            let owned = gdn_owned_lengths(original);
            // Chunk recurrence owns additional gamma/A/P/tail arrays. They stay
            // funded by the same old/new terms and do not contribute savings.
            let mut chunk = original;
            for (index, words) in [(14, 48 * 64 * 18), (15, 48 * 64 * 64 * 18),
                (16, 48 * 64 * 64 * 18), (18, 16 * 64 * 128)] {
                chunk[index] = words;
            }
            assert_eq!(bytes(&chunk) - bytes(&gdn_owned_lengths(chunk)), bytes(&original) - bytes(&owned));
            for i in 0..20 {
                if i != 0 && i != 12 { assert_eq!(original[i], owned[i]); }
            }
            let hc = hc_lengths(10240, 320, 2560, rows);
            assert_eq!(hc[0], original[0]);
            let old = rows * (2 * 10240 + 320 + 4) * 4 + 2560 * 4 + bytes(&original);
            let new = bytes(&hc) + bytes(&owned);
            assert_eq!(old - new, saved_bytes(2560, 48, 16, rows));
            let extra = if rows == 8192 { head } else { 0 };
            assert!(new + extra <= old);
        }
        assert_eq!(saved_bytes(2560, 48, 16, 8192), 754_974_592);
        // Funding uses only the largest prompt workspace. All private and pooled
        // widths save additional storage and cannot introduce an admission cliff.
        for active in 1usize..=16 {
            let pooled: usize = [1, 2, 4, 8, 16].into_iter()
                .filter(|n| *n <= active.next_power_of_two())
                .map(|n| saved_bytes(2560, 48, 16, n)).sum();
            assert!(saved_bytes(2560, 48, 16, 8192) + pooled + active * saved_bytes(2560, 48, 16, 1) >= head);
        }
    }

    struct HostView { owner: Rc<RefCell<Vec<u32>>>, start: usize, len: usize }
    impl View for HostView {
        fn slice(&self, range: Range<usize>) -> Result<Self> {
            if range.start >= range.end || range.end > self.len { return Err("invalid device vector slice".into()); }
            Ok(Self { owner: self.owner.clone(), start: self.start + range.start, len: range.len() })
        }
    }
    #[test]
    fn qkv_alias_retains_its_owner_and_cannot_cross_the_hc_normalization_range() {
        // Canary injection words follow normalized in a shared arena. Reusing
        // normalized cannot overwrite the four gates that remain live in GDN.
        let owner = Rc::new(RefCell::new(vec![0; 20]));
        owner.borrow_mut()[16..].fill(0xfeed);
        let hc = HostView { owner: owner.clone(), start: 0, len: 16 };
        let qkv = qkv_view(&hc, 16).unwrap();
        assert!(qkv_view(&hc, 17).is_err());
        assert!(qkv_view(&hc, 0).is_err());
        drop(hc);
        drop(owner);
        qkv.owner.borrow_mut()[qkv.start..qkv.start + qkv.len].fill(0xcafe);
        assert_eq!(&qkv.owner.borrow()[16..], &[0xfeed; 4]);
        assert_eq!(&qkv.owner.borrow()[..16], &[0xcafe; 16]);
        // Graphs may retain this pointer while host view lengths change. A view
        // owns its original extent; no resize of another view changes it.
        let mut shortened = qkv_view(&qkv, 8).unwrap();
        shortened.len = 4;
        assert_eq!(qkv.len, 16);
    }
}
