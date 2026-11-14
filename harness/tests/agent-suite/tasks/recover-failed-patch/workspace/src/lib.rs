#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

/// The level that a config file names, in any case.
pub fn parse_level(name: &str) -> Option<Level> {
    match name.to_ascii_lowercase().as_str() {
        "debug" => Some(Level::Debug),
        "info" | "warn" | "warning" => Some(Level::Info),
        "error" => Some(Level::Error),
        _ => None,
    }
}
