//! Which prompt-row launches write every word of their output, so the host need not clear the output first.
//! Each listed kernel is checked on the device from an output poisoned with 0xff (tools/bench `--no-fill`, and the
//! `--rows-small` sweep for the small plain kernel, which poisons both outputs), with byte equality to the host
//! reference. A launch outside these lists keeps the clear. Shape only; a host test enumerates the admitted
//! Transpose shapes and fails if the device sweep list does not contain one.
/// A verify pass has 1 to 16 token rows; one dimension of its transposes is that count.
pub const SMALL: u32 = 16;
/// The large dimensions the device sweep runs: 1 to 64, multiples of 16 up to 1024, multiples of 256 up to 32768.
pub fn large_dim(d: u32) -> bool { (1..=64).contains(&d) || (d.is_multiple_of(16) && d <= 1024) || (d.is_multiple_of(256) && d <= 32768) }
/// A Transpose of `rows` x `cols` words: one side at most SMALL, the other a swept dimension.
pub fn transpose_fill_free(rows: u32, cols: u32) -> bool {
    let (lo, hi) = if rows <= cols { (rows, cols) } else { (cols, rows) };
    (1..=SMALL).contains(&lo) && large_dim(hi)
}
/// The shapes the device sweep runs, in both orders.
pub fn transpose_sweep() -> Vec<(u32, u32)> {
    let dims: Vec<u32> = (1..=32768).filter(|&d| large_dim(d)).collect();
    let mut v = Vec::new();
    for lo in 1..=SMALL { for &hi in &dims { if hi >= lo { v.push((lo, hi)); if lo != hi { v.push((hi, lo)); } } } }
    v
}
/// True when the prompt kernel chosen for this launch writes every output word whatever the output held before.
/// `kernel` is the tile or small kernel name the host chose, or None for the generic `prompt_rows`.
pub fn writes_every_word(is_transpose: bool, rows: u32, cols: u32, kernel: Option<&str>) -> bool {
    // `pool_scores_tile` stores every word: a score, or zero for a pool that is not complete (kernel-bench --pool-scores, poisoned outputs).
    // `step_rows_tile` stores every word: each token's output and the state (kernel-bench --step-tile, poisoned outputs).
    match kernel { Some("plain_rows_small") | Some("pool_scores_tile") | Some("step_rows_tile") => true, Some(_) => false, None => is_transpose && transpose_fill_free(rows, cols) }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_sweep_list_contains_every_fill_free_transpose() {
        let swept: std::collections::HashSet<(u32, u32)> = transpose_sweep().into_iter().collect();
        for a in 0..=20u32 { for b in 0..=33000u32 {
            if transpose_fill_free(a, b) { assert!(swept.contains(&(a, b)), "transpose {a} x {b} is fill free but not swept"); }
            if transpose_fill_free(b, a) { assert!(swept.contains(&(b, a)), "transpose {b} x {a} is fill free but not swept"); }
        } }
        assert!(swept.iter().all(|&(a, b)| transpose_fill_free(a, b)));
    }
    #[test]
    fn model_transposes_are_fill_free_and_others_keep_the_clear() {
        for (r, k) in [(3u32, 6144u32), (6144, 3), (2048, 4), (16, 8192), (1, 288), (2, 16384), (4, 32768)] { assert!(writes_every_word(true, r, k, None), "{r}x{k}"); }
        for (r, k) in [(17u32, 4096u32), (3, 6145), (64, 64), (3, 40000), (0, 4)] { assert!(!writes_every_word(true, r, k, None), "{r}x{k}"); }
        assert!(!writes_every_word(false, 3, 6144, None) && writes_every_word(false, 3, 6144, Some("plain_rows_small")) && !writes_every_word(true, 3, 6144, Some("prompt_rows_small")));
    }
}
