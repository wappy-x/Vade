//! The shell policy that Vade ships: every `workspace_shell` command needs
//! approval, unless an allow-list rule names its exact argv or a separately
//! reviewed fixed grammar proves that it is harmless. The checker in `safety`
//! still refuses the commands that it blocks. Its wider safe class counts only
//! when the experimental checker is on (`VADE_SHELL_CHECKER=1`).
//!
//! A rule that a person writes in a permissions file can also hold
//! `<path>` slots: each one matches one plain word that names a path inside
//! the workspace (see `workspace_path`). This needs no shell reading: the
//! argv is compared word for word.
use super::permissions::Rule;
use super::safety::{self, Class, Finding, Verdict};
use std::path::Path;

/// The environment switch for the experimental checker.
pub const CHECKER_ENV: &str = "VADE_SHELL_CHECKER";

/// The reason that commands outside the fixed harmless grammar get under
/// the default policy.
pub const DEFAULT_REASON: &str = "this command is not in the fixed harmless command set";

/// Longest argv that can become a rule.
const MAX_RULE_WORDS: usize = 128;

/// True when the value of `VADE_SHELL_CHECKER` turns the experimental
/// checker on: `1`, `on` or `experimental`.
pub fn checker_from(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "on" | "experimental"))
}

pub fn decide(argv: &[String], root: &Path, checker: bool) -> Verdict {
    decide_fixed(argv, root, checker, false)
}

/// The class of one argv with a separate proof from the fixed harmless
/// command grammar in [`crate::shell::CommandSpec`].
pub fn decide_fixed(argv: &[String], root: &Path, checker: bool, fixed_safe: bool) -> Verdict {
    let mut verdict = safety::classify(argv, root);
    if verdict.class == Class::Blocked || checker || fixed_safe && verdict.class == Class::Safe {
        return verdict;
    }
    verdict.class = Class::Ask;
    verdict.findings.insert(0, Finding { class: Class::Ask, reason: DEFAULT_REASON.into(), words: argv.to_vec(), listable: true });
    verdict
}

/// A plain literal word: 1 to 4096 ASCII letters, digits and `- _ . , / :
/// = + @`. No shell reads it as anything else: it has no variable,
/// pattern, brace, bracket, quote, `~`, space or operator.
pub fn literal(word: &str) -> bool {
    !word.is_empty() && word.len() <= 4096 && word.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.,/:=+@".contains(&b))
}

/// The slot word of a written rule. It matches one workspace path.
pub const PATH_SLOT: &str = "<path>";

/// A word that names a hidden or credential file, or a permissions file,
/// in any part. The parts are split at `/ = : @ ,`, so values attached to
/// options (`-tfid_rsa`, `HEAD:.env`, `@id_rsa`) count too. Such a word
/// never goes on an allow list.
pub fn private_word(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    lower.contains(super::permissions::FILE_NAME)
        || lower.split(['/', '=', ':', '@', ',']).any(|part| {
            (part.starts_with('.') && part != "." && part != "..")
                || crate::tools::SECRET_PATH_RULES.iter().any(|rule| rule.matches_attached(part))
        })
}

/// Whether `word` fills a `<path>` slot: a plain literal word that is not
/// an option, relative to the workspace, with no `..`, and no hidden,
/// credential or non-ASCII part (`tools::safe_parts`, as for workspace
/// reads). `.` is the workspace itself. No part that exists under `root`
/// may be a link, so the path cannot point outside the workspace.
pub fn workspace_path(word: &str, root: &Path) -> bool {
    if !literal(word) || word.starts_with('-') || private_word(word) {
        return false;
    }
    let Ok(parts) = crate::tools::safe_parts(word) else { return false };
    let mut path = root.to_path_buf();
    for part in parts {
        path.push(part);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => return false,
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return true,
            Err(_) => return false,
        }
    }
    true
}

/// Whether one rule approves `argv`: the same number of words, each equal
/// to the rule word, or a workspace path where the rule has `<path>`.
pub fn rule_matches(rule: &[String], argv: &[String], root: &Path) -> bool {
    rule.len() == argv.len()
        && rule.iter().zip(argv).all(|(want, word)| want == word || want == PATH_SLOT && workspace_path(word, root))
}

/// The rule that `s` or `p` saves for `argv`: the exact argv, when every
/// word is a plain literal that names no private file and no finding keeps
/// the command off a list (root, credentials, text sent into a shell as
/// code).
pub fn rule_for(argv: &[String], verdict: &Verdict) -> Option<Rule> {
    listable(argv, verdict).then(|| argv.to_vec())
}

/// Whether a rule may approve `argv` with this verdict.
pub fn listable(argv: &[String], verdict: &Verdict) -> bool {
    verdict.class == Class::Ask
        && !argv.is_empty()
        && argv.len() <= MAX_RULE_WORDS
        && argv.iter().all(|word| literal(word) && !private_word(word))
        && verdict.findings.iter().all(|finding| finding.listable)
}
