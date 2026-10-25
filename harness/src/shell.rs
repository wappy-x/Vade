//! Bounded macOS command execution. argv is explicit; there is no implicit shell.
//! Seatbelt confines writes to the selected root and denies network and secrets.
mod execution;
pub use execution::{run, run_in_session, run_with_exclusions};
pub(crate) use execution::run_in_session_observing;
#[cfg(all(test, target_os = "macos"))]
use execution::{profile, verify_tree};
mod build;
use crate::tools::Workspace;
pub use build::BuildSession;
use std::io;
#[cfg(target_os = "macos")]
use std::io::Read;
#[cfg(target_os = "macos")]
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "macos")]
use std::sync::atomic::Ordering;
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

/// Owner-selected directory names, inaccessible at every depth. Model arguments
/// cannot widen this policy. Skipping a scan needs matching OS read/write denial,
/// including removal/rename of the excluded directory itself.
#[derive(Clone, Debug)]
pub struct Exclusions {
    names: Vec<String>,
}
impl Default for Exclusions {
    fn default() -> Self {
        Self {
            names: vec!["target".into(), "node_modules".into()],
        }
    }
}
impl Exclusions {
    pub fn add(&mut self, name: &str) -> Result<(), &'static str> {
        if name.is_empty()
            || name.len() > 64
            || name.starts_with('.')
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
        {
            return Err(
                "shell exclusion must be one visible ASCII directory name (letters, digits, _, - or .; at most 64 bytes)",
            );
        }
        if self.names.iter().any(|n| n.eq_ignore_ascii_case(name)) {
            return Ok(());
        }
        if self.names.len() >= 10 {
            return Err("at most eight additional shell exclusions");
        }
        self.names.push(name.into());
        Ok(())
    }
    pub fn describe(&self) -> String {
        self.names.join(", ")
    }
    #[cfg(target_os = "macos")]
    fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|n| n.eq_ignore_ascii_case(name))
    }
    #[cfg(target_os = "macos")]
    fn access_denial(&self) -> String {
        // A component rule also covers filesystem aliases of the workspace
        // (including the macOS Data volume). Dots are literal regex classes.
        let names = self
            .names
            .iter()
            .map(|name| ascii_case_pattern(&name.replace('.', "[.]")))
            .collect::<Vec<_>>()
            .join("|");
        // Unscanned hard links may alias private files outside the workspace,
        // so read-only is insufficient: their contents must be opaque too.
        format!("(deny file-read* file-write* (regex #\"/({names})(/|$)\"))")
    }
}

#[cfg(target_os = "macos")]
fn ascii_case_pattern(pattern: &str) -> String {
    // Seatbelt's regex dialect is not PCRE: (?i) does not enable case folding.
    // Expand only ASCII letters in these first-party patterns; no workspace
    // regex syntax is accepted. Character classes handle macOS case aliases.
    let mut result = String::new();
    for c in pattern.chars() {
        if c.is_ascii_alphabetic() {
            result.push_str(&format!(
                "[{}{}]",
                c.to_ascii_lowercase(),
                c.to_ascii_uppercase()
            ));
        } else {
            result.push(c);
        }
    }
    result
}

/// Which edits and commands need the user's approval: `ask` (all but the
/// read-only allowlist), `auto` (none, still sandboxed) or `never` (all denied).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalPolicy {
    Ask,
    Auto,
    Never,
}
impl ApprovalPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ask" => Some(Self::Ask),
            "auto" => Some(Self::Auto),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Ask => "ask (read-only allowlist runs; ask for edits and other commands)",
            Self::Auto => "auto (apply patches and run all commands inside the OS sandbox)",
            Self::Never => "never (deny edits and commands)",
        }
    }
}
/// One command to run: explicit argv, no implicit shell, and a timeout.
#[derive(Clone, Debug)]
pub struct CommandSpec {
    pub argv: Vec<String>,
    pub timeout_ms: u64,
}
impl CommandSpec {
    /// Host compilation only, with each Cargo workspace selected separately.
    /// These argv do not generate GPU kernels, run a server or qualify numerics.
    pub fn host_check(profile: &str, timeout_ms: u64) -> io::Result<Self> {
        let manifest = match profile {
            "engine-host" => "engine/Cargo.toml",
            "engine-bench-host" => "engine/tools/bench/Cargo.toml",
            "harness-host" => "harness/Cargo.toml",
            "cli-host" => "cli/Cargo.toml",
            _ => return Err(io::Error::other("unknown host check profile")),
        };
        let mut argv = vec![
            "cargo".into(),
            "check".into(),
            "--offline".into(),
            "--manifest-path".into(),
            manifest.into(),
        ];
        if profile == "engine-host" {
            argv.extend([
                "--workspace".into(),
                "--exclude".into(),
                "spark-rust-kernels".into(),
            ]);
        }
        let spec = Self { argv, timeout_ms };
        spec.validate()?;
        Ok(spec)
    }
    pub fn validate(&self) -> io::Result<()> {
        if self.argv.is_empty()
            || self.argv.len() > 128
            || self.argv[0].is_empty()
            || self.argv.iter().map(String::len).sum::<usize>() > 16 * 1024
            || self.argv.iter().any(|s| s.contains('\0'))
            || !(1..=120_000).contains(&self.timeout_ms)
        {
            return Err(io::Error::other(
                "command needs 1–128 arguments, at most 16 KiB, and timeout_ms 1–120000",
            ));
        }
        Ok(())
    }
    pub fn allowlisted(&self) -> bool {
        self.read_only_argv().is_some()
    }
    /// Fixed system executables and narrow argument grammars. Bare names are
    /// resolved here, never through PATH or a workspace-supplied executable.
    fn read_only_argv(&self) -> Option<Vec<String>> {
        let (program, args) = self.argv.split_first()?;
        let path = |s: &String| {
            !s.starts_with('-')
                && crate::tools::safe_parts(s).is_ok_and(|parts| !parts.is_empty() || s == ".")
        };
        let executable = match program.as_str() {
            "pwd" | "/bin/pwd" if args.is_empty() => "/bin/pwd",
            "true" | "/usr/bin/true" if args.is_empty() => "/usr/bin/true",
            "ls" | "/bin/ls"
                if args.is_empty()
                    || args.iter().all(path)
                    || args[0] == "-1" && args[1..].iter().all(path) =>
            {
                "/bin/ls"
            }
            "cat" | "/bin/cat" if !args.is_empty() && args.iter().all(path) => "/bin/cat",
            "git" | "/usr/bin/git" => return self.read_only_git_argv(),
            _ => return None,
        };
        Some(
            std::iter::once(executable.to_owned())
                .chain(args.iter().cloned())
                .collect(),
        )
    }
    fn read_only_git_argv(&self) -> Option<Vec<String>> {
        let args = &self.argv[1..];
        let command = args.first()?.as_str();
        let valid = match command {
            "status" => args.len() == 1 || args.len() == 2 && args[1] == "--short",
            "diff" => args.len() == 1 || args.len() == 2 && args[1] == "--stat",
            "log" => {
                args.len() == 1
                    || args.len() == 2 && args[1] == "--oneline"
                    || args.len() == 3
                        && args[1] == "-n"
                        && args[2]
                            .parse::<usize>()
                            .is_ok_and(|n| (1..=100).contains(&n))
            }
            _ => false,
        };
        if !valid {
            return None;
        }
        let mut argv: Vec<String> = [
            "/usr/bin/git",
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "status.submoduleSummary=false",
            "-c",
            "submodule.recurse=false",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        argv.extend(args.iter().cloned());
        if command == "log" {
            argv.push("--no-show-signature".into());
            if args.len() < 3 {
                argv.extend(["-n".into(), "100".into()]);
            }
        } else {
            argv.push("--ignore-submodules=all".into());
            if command == "diff" {
                argv.extend(["--no-ext-diff".into(), "--no-textconv".into()]);
            }
            // Git's object database must not expose hidden or credential-file
            // contents that ordinary workspace reads would refuse.
            argv.extend([
                "--".into(),
                ".".into(),
                ":(exclude,glob)**/.*".into(),
                ":(exclude,glob)**/.*/**".into(),
            ]);
            for name in crate::tools::CREDENTIAL_NAMES
                .iter()
                .map(|s| (*s).to_owned())
                .chain(
                    crate::tools::CREDENTIAL_SUFFIXES
                        .iter()
                        .map(|s| format!("*{s}")),
                )
            {
                argv.push(format!(":(exclude,icase,glob)**/{name}"));
                argv.push(format!(":(exclude,icase,glob)**/{name}/**"));
            }
        }
        Some(argv)
    }
    pub fn preview(&self) -> String {
        let json = crate::json::stringify(&crate::json::Value::Array(
            self.argv
                .iter()
                .cloned()
                .map(crate::json::Value::String)
                .collect(),
        ))
        .expect("bounded command arguments");
        let mut visible = String::new();
        for c in json.chars() {
            // JSON already escapes C0 and literal backslashes. Escape other
            // nonprinting Unicode as UTF-16 so the displayed JSON still decodes
            // to the exact argv, including supplementary variation selectors.
            if crate::terminal::preview_char_needs_escape(c) {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    visible.push_str(&format!("\\u{unit:04x}"));
                }
            } else {
                visible.push(c);
            }
        }
        format!(
            "command argv={} · timeout={}ms · display≤64KiB (head/tail; truncation does not stop command) · network denied",
            visible, self.timeout_ms
        )
    }
}
/// How a command ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Exited(Option<i32>),
    Timeout,
    Cancelled,
    BuildLimit,
}
/// A finished command: its outcome, bounded stdout and stderr, byte counts
/// (including omitted middles) and elapsed time.
#[derive(Debug)]
pub struct CommandOutput {
    pub outcome: Outcome,
    pub stdout: String,
    pub stderr: String,
    /// Counts are bytes read from the pipes, including discarded middle bytes.
    /// Timeout/cancellation may stop before the producer finishes writing.
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    pub stdout_omitted: u64,
    pub stderr_omitted: u64,
    pub elapsed_ms: u64,
    /// Present for a completed child even if retained build state then exceeds
    /// its quota. Cancellation/deadline outcomes are not completed check passes.
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
}
impl CommandOutput {
    pub fn render(&self) -> String {
        format!(
            "status: {:?}\nexit_code: {:?}\nsignal: {:?}\nelapsed_ms: {}\noutput: stdout_read={} stderr_read={} stdout_omitted={} stderr_omitted={} bytes; head/tail retained\nstdout:\n{}\nstderr:\n{}",
            self.outcome,
            self.exit_code,
            self.signal,
            self.elapsed_ms,
            self.stdout_bytes,
            self.stderr_bytes,
            self.stdout_omitted,
            self.stderr_omitted,
            self.stdout,
            self.stderr
        )
    }
}

#[cfg(test)]
mod tests;
