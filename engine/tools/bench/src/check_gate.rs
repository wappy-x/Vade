//! Verdicts for device checks. A check that prints a mismatch count and goes on, or swaps in a default after a
//! failed step, passes after a failure. These helpers turn both into a failed run. They hold no device code, so
//! the host tests below cover the decisions the GPU checks make.
use std::cell::Cell;

/// Zero mismatches passes. Anything else fails with one line naming the check and the count.
pub fn no_mismatch(what: &str, bad: usize, checked: usize) -> Result<(), String> {
    if bad == 0 { Ok(()) } else { Err(format!("{what}: {bad} of {checked} rows differ")) }
}

/// A step whose failure must fail the run, while later diagnostics still run on `fallback`. On error the failure
/// is counted in `failed` and its message returned for the caller to print. The caller fails when `failed` > 0.
pub fn or_count_failed<T>(result: Result<T, String>, fallback: T, failed: &Cell<usize>) -> (T, Option<String>) {
    match result {
        Ok(value) => (value, None),
        Err(error) => {
            failed.set(failed.get() + 1);
            (fallback, Some(error))
        }
    }
}

/// Bytes to fill an output buffer with before a device check reads it: every 32-bit word is a NaN, so no
/// correct answer equals it and a kernel that skips a word cannot inherit a stale right answer.
pub fn poison(bytes: usize) -> Vec<u8> {
    vec![0xff; bytes]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mismatch_count_above_zero_fails() {
        assert!(no_mismatch("CHUNK case=c arm=0", 0, 48).is_ok());
        assert!(no_mismatch("CHUNK case=c arm=0", 0, 0).is_ok());
        assert_eq!(no_mismatch("CHUNK case=c arm=1", 3, 48).unwrap_err(), "CHUNK case=c arm=1: 3 of 48 rows differ");
    }

    #[test]
    fn a_fallback_after_an_error_counts_as_a_failed_check() {
        let failed = Cell::new(0);
        assert_eq!(or_count_failed(Ok(7), 0, &failed), (7, None));
        assert_eq!(failed.get(), 0);
        assert_eq!(or_count_failed(Err("CALIBRATION FAIL: no model fits".into()), 0, &failed), (0, Some("CALIBRATION FAIL: no model fits".into())));
        assert_eq!(failed.get(), 1);
    }

    #[test]
    fn poison_words_equal_no_number() {
        let bytes = poison(16);
        assert_eq!(bytes.len(), 16);
        for word in bytes.as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*word);
            assert!(value.is_nan());
            assert_ne!(value.to_bits(), 0f32.to_bits());
        }
    }
}
