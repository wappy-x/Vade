//! Shared argument parsing for CLI unit fixtures.
use super::args::{Config, Parsed};

pub(crate) fn parse(args: &[&str]) -> Result<Parsed, &'static str> {
    Config::parse(args.iter().map(|s| s.to_string())).map_err(|error| error.message)
}
