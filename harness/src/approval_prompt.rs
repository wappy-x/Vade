//! Approval words and layout only. Callers retain every policy decision.
use crate::shell::safety::{Class, Verdict};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Risk { Red, Amber }

/// The same single-character answers apply to raw input and submitted lines.
pub fn answer_key(key: &str) -> Option<u8> {
    match key {
        "1" | "y" => Some(1), "2" | "n" => Some(2),
        "3" => Some(3), "4" => Some(4), "5" => Some(5), _ => None,
    }
}

/// Structured, trusted approval content. Layout is computed at the terminal width.
#[derive(Clone, Debug)]
pub struct Prompt {
    pub what: String,
    pub object: String,
    pub risk: Risk,
    pub tag: String,
    pub scope: Option<String>,
    pub sentences: Vec<String>,
    pub choices: String,
    pub files: Vec<String>,
}

pub(crate) fn action(what: &str, object: &str, risk: Risk, tag: &str, scope: Option<&str>) -> Prompt {
    Prompt { what: what.into(), object: object.into(), risk, tag: tag.into(), scope: scope.map(str::to_owned), sentences: Vec::new(), choices: String::new(), files: Vec::new() }
}

pub(crate) fn choices(no: &str, session: Option<&str>, always: Option<&str>, details: Option<&str>) -> String {
    let mut text = format!("1  Yes, one time\n2  No, {no}");
    if let Some(scope) = session { text.push_str(&format!("\n3  Yes, until you close Vade ({scope})")); }
    if let Some(scope) = always { text.push_str(&format!("\n4  Yes, always {scope}")); }
    if let Some(details) = details { text.push_str(&format!("\n5  Show the exact {details}")); }
    text
}

pub(crate) fn finish(what: &Prompt, why: &str, choices: &str) -> Prompt {
    let mut prompt = what.clone();
    // Callers separate warning paragraphs with newlines. Each sentence gets a row.
    prompt.sentences = why.split('\n').flat_map(|line| line.split_inclusive(". "))
        .map(|sentence| sentence.trim().to_owned()).filter(|sentence| !sentence.is_empty()).collect();
    prompt.choices = choices.into();
    prompt
}

/// Display the executed words with unambiguous quoting and visible controls.
pub(crate) fn command(words: &[String]) -> String {
    words.iter().map(|word| command_word(word, false)).collect::<Vec<_>>().join(" ")
}

fn command_word(word: &str, quote_comma: bool) -> String {
    if word.is_empty() { return "''".into(); }
    if word.chars().all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c))
        && !(quote_comma && word.contains(',')) { return word.into(); }
    // Plain single quotes hold a word that needs no escape. Every other word uses $'...', with only
    // the escapes that bash (also the 3.2 that macOS ships) and zsh read the same way: \\ \' \n \t \r
    // and \xHH. Invisible and look-alike characters are written as their UTF-8 bytes. A run of spaces is
    // written as \x20 for each space, so a padded argument cannot lay out rows of its own, and the
    // shown text still means exactly the words that run.
    let hidden = |c: char| c.is_control() || crate::terminal::preview_char_needs_escape(c);
    if !word.contains("  ") && !word.chars().any(|c| matches!(c, '\\' | '\'') || hidden(c)) { return format!("'{word}'"); }
    let mut shown = String::from("$'");
    let mut chars = word.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ' ' => {
                let mut count = 1;
                while chars.peek() == Some(&' ') { chars.next(); count += 1; }
                shown.push_str(&if count > 1 { "\\x20".repeat(count) } else { " ".into() });
            }
            '\\' => shown.push_str("\\\\"), '\'' => shown.push_str("\\'"),
            '\n' => shown.push_str("\\n"), '\t' => shown.push_str("\\t"), '\r' => shown.push_str("\\r"),
            c if hidden(c) => {
                let mut bytes = [0u8; 4];
                for byte in c.encode_utf8(&mut bytes).bytes() { shown.push_str(&format!("\\x{byte:02x}")); }
            }
            c => shown.push(c),
        }
    }
    shown.push('\'');
    shown
}

impl Prompt {
    pub fn render(&self, columns: usize, color: bool) -> String {
        use crate::terminal::render::{clip, style, width, wrap, wrap_exact};
        let columns = columns.clamp(20, 64);
        let risk = if self.risk == Risk::Red { "1;31" } else { "1;33" };
        let petrol = "1;38;2;27;143;196";
        let mut rows = Vec::new();
        let right = |rows: &mut Vec<String>, left: &str, label: &str, left_style: &str, right_style: &str| {
            if width(left) + width(label) + 2 <= columns {
                rows.push(format!("{}{}{}", style(left, left_style, color), " ".repeat(columns - width(left) - width(label)), style(label, right_style, color)));
            } else {
                rows.extend(wrap(left, columns).into_iter().map(|row| style(row.trim_end(), left_style, color)));
                for row in wrap(label, columns) {
                    let row = row.trim_end();
                    rows.push(format!("{}{}", " ".repeat(columns.saturating_sub(width(row))), style(row, right_style, color)));
                }
            }
        };
        // Owner's colors (2026-11-14): the top line and the marks are Petrol; the object and the key numbers take the risk color.
        right(&mut rows, &self.what, &self.tag, petrol, "1;37");
        let indent = |n: usize| " ".repeat(n.min(columns.saturating_sub(1)));
        let target_width = columns - 3;
        // File names: rows break between names. A name is split only when it alone is longer than a row.
        let file_rows = |count: usize| -> Vec<String> {
            let mut pieces: Vec<String> = self.files[..count].iter().map(|name| command_word(name, true)).collect();
            if count < self.files.len() { pieces.push(format!("and {} more", self.files.len() - count)); }
            let last = pieces.len() - 1;
            let (mut packed, mut line) = (Vec::new(), String::new());
            for (index, piece) in pieces.into_iter().enumerate() {
                let piece = if index < last { format!("{piece},") } else { piece };
                if !line.is_empty() && width(&line) + 1 + width(&piece) > target_width { packed.push(std::mem::take(&mut line)); }
                if width(&piece) > target_width {
                    let mut parts = wrap_exact(&piece, target_width);
                    line = parts.pop().unwrap_or_default();
                    packed.extend(parts);
                } else {
                    if !line.is_empty() { line.push(' '); }
                    line.push_str(&piece);
                }
            }
            if !line.is_empty() { packed.push(line); }
            packed
        };
        // Objects keep every space, including a space at the right edge.
        let target = if self.files.is_empty() { wrap_exact(&self.object, target_width) } else {
            // As many whole names as fit in four rows, and always the first name.
            (1..=self.files.len()).rev().map(file_rows).find(|rows| rows.len() <= 4).unwrap_or_else(|| file_rows(1))
        };
        let mark = if self.risk == Risk::Red { "■" } else { "▲" };
        for (index, row) in target.iter().enumerate() {
            let row = row.as_str();
            let prefix = if index == 0 { format!("{mark}  ") } else { "│  ".into() };
            let mut line = format!("{}{}", style(&prefix, petrol, color), style(row, risk, color));
            if index == 0 && let Some(scope) = &self.scope
                && width(&prefix) + width(row) + width(scope) + 2 <= columns {
                line.push_str(&" ".repeat(columns - width(&prefix) - width(row) - width(scope)));
                line.push_str(scope);
            }
            rows.push(line);
        }
        if let Some(scope) = &self.scope
            && target.first().is_none_or(|row| 3 + width(row) + width(scope) + 2 > columns) {
            for row in wrap(scope, target_width) {
                let row = row.trim_end();
                rows.push(format!("{}{}{row}", style("│  ", petrol, color), " ".repeat(target_width.saturating_sub(width(row)))));
            }
        }
        for (index, sentence) in self.sentences.iter().enumerate() {
            let prefix = indent(6);
            for row in wrap(sentence, columns.saturating_sub(width(&prefix)).max(1)) {
                let line = format!("{prefix}{}", row.trim_end());
                rows.push(if index == 0 { style(&line, "2", color) } else { line });
            }
        }
        if !self.choices.is_empty() {
            rows.push("-".repeat(columns));
            for (index, choice) in self.choices.lines().enumerate() {
                let prefix = if index == 0 { "Your turn.  ".to_owned() } else { " ".repeat(12) };
                let prefix = clip(&prefix, columns.saturating_sub(1));
                let continuation = indent(15);
                let available = columns.saturating_sub(width(&continuation)).max(1);
                // The number occupies three cells before the choice text.
                let (key, text) = choice.split_once("  ").unwrap_or(("", choice));
                let chunks = wrap(text, available);
                for (part, row) in chunks.iter().enumerate() {
                    if part == 0 {
                        rows.push(format!("{}{}  {}", if index == 0 { style(&prefix, "1;33", color) } else { prefix.clone() }, style(key, risk, color), row.trim_end()));
                    } else {
                        rows.push(format!("{continuation}{}", row.trim_end()));
                    }
                }
            }
        }
        rows.join("\n")
    }
}

/// Warning color uses existing findings and rule eligibility, never argv
/// parsing or a new security verdict. Unknown one-time effects stay red.
pub(crate) fn command_risk(verdict: &Verdict, saved_yes: bool) -> Risk {
    const LOSS: &[&str] = &["deletes", "discards", "overwrites", "writes", "rewrites", "edits", "replaces", "formats", "partitions", "force-pushes", "publishes", "network", "credential", "copies", "moves files"]; // vade-secret-list-allow: risk reason words, not path names
    if !saved_yes || verdict.findings.iter().any(|finding| finding.class == Class::Ask
        && LOSS.iter().any(|word| finding.reason.contains(word))) { Risk::Red } else { Risk::Amber }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn choice_numbers_never_move() {
        let two = choices("do not run it", None, None, None);
        assert_eq!(two, "1  Yes, one time\n2  No, do not run it");
        let all = choices("do not run it", Some("this exact command"), Some("in this project"), None);
        assert!(all.starts_with(&two) && all.contains("\n3  Yes, until you close Vade") && all.contains("\n4  Yes, always"));
    }
    #[test]
    fn command_colors_follow_findings_without_changing_them() {
        let root = std::path::Path::new(".");
        for (argv, saved, expected) in [
            (vec!["rm", "output.txt"], true, Risk::Red),
            (vec!["cargo", "test"], true, Risk::Amber),
            (vec!["sh", "-c", "echo test"], false, Risk::Red),
        ] {
            let words = argv.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let verdict = crate::shell::policy::decide(&words, root, false);
            assert_eq!(command_risk(&verdict, saved), expected);
        }
    }
}

#[cfg(test)]
#[path = "approval_prompt_tests.rs"]
pub(crate) mod layout_tests;
