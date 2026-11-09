// Part of the tally crate. License: Apache-2.0.
//! Statistics over integer samples.

/// The mean of the values, rounded toward zero. `None` for an empty slice.
pub fn mean(values: &[i64]) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<i64>() / values.len() as i64)
}
