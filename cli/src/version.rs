//! `vade --version`: the package version and the commit of this build.
//! `cli/build.rs` finds the commit. A build with no commit says so, and a build
//! with changes outside its commit says that too.
use crate::evals::json::{self, Value};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The 40-digit commit of this build, or `None` when the build had no commit.
fn sha() -> Option<&'static str> { option_env!("VADE_CLI_COMMIT").filter(|sha| !sha.is_empty()) }

/// Whether the source tree of this build had changes that its commit does not hold.
fn modified() -> bool { option_env!("VADE_CLI_MODIFIED") == Some("true") }

fn describe(version: &str, sha: Option<&str>, modified: bool) -> String {
    match sha {
        Some(sha) if modified => format!("vade {version} ({}, modified)\n", &sha[..7]),
        Some(sha) => format!("vade {version} ({})\n", &sha[..7]),
        None => format!("vade {version} (unknown commit)\n"),
    }
}

/// The text of `vade --version`, with its line end.
pub fn text() -> String { describe(VERSION, sha(), modified()) }

/// The `vade.version.v1` object.
pub fn object() -> Value {
    json::object(vec![("schema", "vade.version.v1".into()), ("version", VERSION.into()),
        ("sha", sha().map_or(Value::Null, Into::into)), ("modified", Value::Bool(modified()))])
}

#[cfg(test)]
mod tests {
    use super::describe;
    #[test]
    fn a_build_names_its_short_commit_or_says_that_it_has_none() {
        let sha = "f7946e5aaaabbbbccccddddeeeeffff000011112";
        assert_eq!(describe("0.1.0", Some(sha), false), "vade 0.1.0 (f7946e5)\n");
        assert_eq!(describe("0.1.0", Some(sha), true), "vade 0.1.0 (f7946e5, modified)\n");
        assert_eq!(describe("0.1.0", None, false), "vade 0.1.0 (unknown commit)\n");
    }
}
