//! Parse short duration text such as `1h30m`. See SPEC.md.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationError {
    /// The text is empty.
    Empty,
    /// The text does not follow the grammar in SPEC.md.
    Invalid,
    /// The number of seconds does not fit in a u64.
    Overflow,
}

/// Returns the duration in seconds. See SPEC.md for the rules.
pub fn parse_duration(text: &str) -> Result<u64, DurationError> {
    let _ = text;
    todo!("implement parse_duration as SPEC.md describes")
}
