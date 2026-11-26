// Hidden tests for implement-from-spec. The runner copies this file in after
// the agent finishes; the agent never sees it.
use duration::{parse_duration, DurationError};

#[test]
fn examples_from_the_spec() {
    assert_eq!(parse_duration("45s"), Ok(45));
    assert_eq!(parse_duration("1h30m"), Ok(5400));
    assert_eq!(parse_duration("2d3s"), Ok(172_803));
    assert_eq!(parse_duration("90m"), Ok(5400));
    assert_eq!(parse_duration("1d1h1m1s"), Ok(90_061));
}

#[test]
fn zeros_and_leading_zeros() {
    assert_eq!(parse_duration("0s"), Ok(0));
    assert_eq!(parse_duration("0h0m"), Ok(0));
    assert_eq!(parse_duration("007m"), Ok(420));
}

#[test]
fn empty_text() {
    assert_eq!(parse_duration(""), Err(DurationError::Empty));
}

#[test]
fn grammar_errors() {
    for text in ["1m1h", "5", "s", " 5s", "5s ", "5 s", "1H", "-5s", "+5s", "1.5h", "1h1h", "1x", "1_000s", "h1"] {
        assert_eq!(parse_duration(text), Err(DurationError::Invalid), "{text:?}");
    }
}

#[test]
fn overflow() {
    assert_eq!(parse_duration("18446744073709551615s"), Ok(u64::MAX));
    assert_eq!(parse_duration("18446744073709551616s"), Err(DurationError::Overflow));
    assert_eq!(parse_duration("213503982334602d"), Err(DurationError::Overflow));
    // Each part fits, but the sum does not: 60 + 18446744073709551556 = 2^64.
    assert_eq!(parse_duration("1m18446744073709551556s"), Err(DurationError::Overflow));
    assert_eq!(parse_duration("1m18446744073709551555s"), Ok(u64::MAX));
    assert_eq!(parse_duration("213503982334601d7h"), Ok(18_446_744_073_709_551_600));
    assert_eq!(parse_duration("213503982334601d8h"), Err(DurationError::Overflow));
    // A repeated unit breaks the grammar even when the sum would overflow.
    assert_eq!(parse_duration("18446744073709551615s1s"), Err(DurationError::Invalid));
}
