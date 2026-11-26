//! The shape sets of the NVFP4 activation-quantization launches and the device sweep lists that cover them.
//! `single_admitted` is the width predicate of the one-lane-per-element launch (`nv_quantize`; the host adds tokens == 1), `batch_admitted` of the
//! one-launch batch of several equal-width quantizations. The device sweep (tools/bench kernel_nvbatch, which includes this file)
//! runs exactly `single_sweep` and `batch_sweep`, and a host test below enumerates the admitted sets and fails if a shape
//! is admitted that the sweep does not contain. Shape only; no model names.
pub const MAX_COLS: u32 = 16384;
pub const BATCH_MIN_JOBS: usize = 2;
pub const BATCH_MAX_JOBS: usize = 9;
/// The widths above 1024 that the device sweep covers.
const WIDE: [u32; 8] = [1536, 2048, 3072, 4096, 7168, 8192, 12288, MAX_COLS];
/// One launch of one activation: 16 to 1024 in steps of 16 and the eight larger widths of the sweep.
pub fn single_admitted(cols: u32) -> bool { (cols > 0 && cols <= 1024 && cols % 16 == 0) || WIDE.contains(&cols) }
/// One launch of `jobs` equal-width activations: the admitted single widths that are multiples of 128, 2 to 9 jobs.
pub fn batch_admitted(cols: u32, jobs: usize) -> bool { single_admitted(cols) && cols % 128 == 0 && (BATCH_MIN_JOBS..=BATCH_MAX_JOBS).contains(&jobs) }
/// The widths the device sweep runs one launch for.
#[cfg(not(target_arch = "nvptx64"))]
pub fn single_sweep() -> Vec<u32> { (1..=64).map(|n| n * 16).chain(WIDE).collect() }
/// The (width, jobs) pairs the device sweep runs one batch launch for.
#[cfg(not(target_arch = "nvptx64"))]
pub fn batch_sweep() -> Vec<(u32, usize)> {
    single_sweep().into_iter().filter(|c| c % 128 == 0).flat_map(|c| (BATCH_MIN_JOBS..=BATCH_MAX_JOBS).map(move |jobs| (c, jobs))).collect()
}
/// The launch choice for a layer's distinct inputs, widths given: one batch launch, or one launch each.
pub fn use_batch(cols: &[u32]) -> bool { cols.first().is_some_and(|&first| cols.iter().all(|&c| c == first) && batch_admitted(first, cols.len())) }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_sweep_lists_contain_every_admitted_shape() {
        let singles = single_sweep(); let batches = batch_sweep();
        for cols in 0..=MAX_COLS + 256 {
            if single_admitted(cols) { assert!(singles.contains(&cols), "single {cols} is admitted but not swept"); }
            for jobs in 0..=BATCH_MAX_JOBS + 2 {
                if batch_admitted(cols, jobs) { assert!(batches.contains(&(cols, jobs)), "batch {cols} x {jobs} is admitted but not swept"); }
            }
        }
        // And the lists hold nothing the predicates reject.
        assert!(singles.iter().all(|&c| single_admitted(c)) && batches.iter().all(|&(c, j)| batch_admitted(c, j)));
    }
    #[test]
    fn dispatch_takes_the_batch_only_for_equal_admitted_widths() {
        assert!(use_batch(&[2048; 8]) && use_batch(&[4096, 4096]) && use_batch(&[128; 9]));
        // One input, too many inputs, unequal widths, unadmitted widths: one launch each (the old path).
        assert!(!use_batch(&[2048]) && !use_batch(&[2048; 10]) && !use_batch(&[2048, 4096]) && !use_batch(&[]));
        assert!(!use_batch(&[2064; 3]) && !use_batch(&[16; 3]) && !use_batch(&[16512; 3]) && !use_batch(&[0; 3]) && !use_batch(&[5120; 3]));
    }
}
