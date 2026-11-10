//! Host checks shared by the FP4 draft projection and its confidence reader.
pub(crate) const RESCORE_ROWS: usize = 16 * 256;
pub(crate) const MAX_SCREEN_ROWS: usize = 1024 * 256;

/// The select kernel stores 32 tile values and one 32-bit mask per warp lane.
/// Larger heads use the full FP4 projection, as heads without an MXFP8 twin do.
pub(crate) fn fp4_rescore_enabled(fp4: bool, mx8: bool, requested: bool, rows: usize) -> bool {
    fp4 && mx8 && requested && rows != 0 && rows.is_multiple_of(64) && rows <= MAX_SCREEN_ROWS
}

/// Read the rows the projection wrote. The full FP4 path does not write `scored`.
pub(crate) fn confidence_logits<T>(rescored: bool, head: T, rows: usize, scored: T) -> (T, usize) {
    if rescored { (scored, RESCORE_ROWS) } else { (head, rows) }
}

#[cfg(test)]
mod tests {
    use super::{confidence_logits, fp4_rescore_enabled, MAX_SCREEN_ROWS, RESCORE_ROWS};

    #[test]
    fn confidence_without_a_rescore_ignores_unwritten_scratch() {
        let scored = vec![f32::NAN; RESCORE_ROWS];
        for (fp4, mx8, requested, rows) in [(true, false, true, 704), (true, true, false, 704),
            (false, true, true, 704), (true, true, true, MAX_SCREEN_ROWS + 64)] {
            let head = vec![0.0f32; rows];
            let rescore = fp4_rescore_enabled(fp4, mx8, requested, rows);
            assert!(!rescore);
            let (source, count) = confidence_logits(rescore, head.as_slice(), rows, scored.as_slice());
            assert!(std::ptr::eq(source, head.as_slice()));
            assert_eq!(count, rows);
            assert!(source.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn confidence_after_a_rescore_reads_only_rescored_rows() {
        let head = vec![f32::NAN; 65536];
        // Tile ids follow the scored rows. They must not enter the softmax.
        let mut scored = vec![0.0f32; RESCORE_ROWS + 16];
        scored[RESCORE_ROWS..].fill(f32::NAN);
        let rescore = fp4_rescore_enabled(true, true, true, head.len());
        assert!(rescore);
        let (source, count) = confidence_logits(rescore, head.as_slice(), head.len(), scored.as_slice());
        assert!(std::ptr::eq(source, scored.as_slice()));
        assert_eq!(count, RESCORE_ROWS);
        let sum: f32 = source[..count].iter().map(|v| v.exp()).sum();
        assert_eq!(1.0 / sum, 1.0 / RESCORE_ROWS as f32);
    }

    #[test]
    fn select_tile_bound_uses_the_full_fp4_fallback() {
        for rows in [64, 704, 4096, MAX_SCREEN_ROWS - 64, MAX_SCREEN_ROWS] {
            assert!(fp4_rescore_enabled(true, true, true, rows), "rows {rows}");
        }
        for rows in [0, 648, MAX_SCREEN_ROWS + 8, MAX_SCREEN_ROWS + 64, usize::MAX] {
            assert!(!fp4_rescore_enabled(true, true, true, rows), "rows {rows}");
        }
        for rows in [64, 640, 648, 704, MAX_SCREEN_ROWS] {
            assert!(!fp4_rescore_enabled(true, false, true, rows), "no twin, rows {rows}");
        }
    }
}
