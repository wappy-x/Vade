use levels::{parse_level, Level};

#[test]
fn names_map_to_levels() {
    assert_eq!(parse_level("debug"), Some(Level::Debug));
    assert_eq!(parse_level("INFO"), Some(Level::Info));
    assert_eq!(parse_level("warn"), Some(Level::Warn));
    assert_eq!(parse_level("Warning"), Some(Level::Warn));
    assert_eq!(parse_level("error"), Some(Level::Error));
    assert_eq!(parse_level("trace"), None);
}
