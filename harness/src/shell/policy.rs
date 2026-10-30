//! The shell policy that Vade ships: every `workspace_shell` command needs
//! approval, unless an allow-list rule names its exact argv and every word
//! of that argv is a plain literal. The checker in `safety` still refuses
//! the commands that it blocks, which only refuses more. Its safe class is
//! advisory: it counts only when the experimental checker is on
//! (`VADE_SHELL_CHECKER=1`).
use super::permissions::Rule;
use super::safety::{self, Class, Finding, Verdict};
use std::path::Path;

/// The environment switch for the experimental checker.
pub const CHECKER_ENV: &str = "VADE_SHELL_CHECKER";

/// The reason that every command gets under the default policy.
pub const DEFAULT_REASON: &str = "every command needs approval unless an allow list names it";

/// Longest argv that can become a rule.
const MAX_RULE_WORDS: usize = 128;

/// True when the value of `VADE_SHELL_CHECKER` turns the experimental
/// checker on: `1`, `on` or `experimental`.
pub fn checker_from(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "on" | "experimental"))
}

/// The class of one argv. Blocked when the checker blocks it. Otherwise
/// Ask, with the default reason first, unless the experimental checker is
/// on and reads the command as safe.
pub fn decide(argv: &[String], root: &Path, checker: bool) -> Verdict {
    let mut verdict = safety::classify(argv, root);
    if verdict.class == Class::Blocked || checker {
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

/// The rule that `s` or `p` saves for `argv`: the exact argv, when every
/// word is a plain literal and no finding keeps the command off a list
/// (root, credentials, text sent into a shell as code).
pub fn rule_for(argv: &[String], verdict: &Verdict) -> Option<Rule> {
    listable(argv, verdict).then(|| argv.to_vec())
}

/// Whether a rule may approve `argv` with this verdict.
pub fn listable(argv: &[String], verdict: &Verdict) -> bool {
    verdict.class == Class::Ask
        && !argv.is_empty()
        && argv.len() <= MAX_RULE_WORDS
        && argv.iter().all(|word| literal(word))
        && verdict.findings.iter().all(|finding| finding.listable)
}
