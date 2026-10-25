//! Human output tokens: colour, error blocks, key/value tables and help pages.
//!
//! Colour is decided per stream: on only when that stream is a terminal,
//! `NO_COLOR` is unset or empty and `TERM` is not `dumb`. `CLICOLOR_FORCE`
//! (non-empty, not `0`) turns it on for pipes; `NO_COLOR` still wins. Every
//! token carries its meaning in words too, so plain output reads the same.
//! Machine output (`--json`, `key=value` records, receipts) never uses this.
use std::{ffi::OsString, io::IsTerminal, sync::OnceLock};

/// Which standard stream a piece of text is written to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream { Out, Err }

/// What a piece of text means; [`paint`] maps it to one SGR style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    /// The product name and page titles: bold cyan.
    Title,
    /// Section headings such as `Usage:`: bold.
    Heading,
    /// A command the reader can type: cyan.
    Command,
    /// A healthy state or finished step: green.
    Ok,
    /// Needs attention but nothing failed: yellow.
    Warn,
    /// The `error:` label: bold red.
    Error,
    /// The `hint:` label: bold cyan.
    Hint,
    /// Secondary detail: dim.
    Dim,
}
impl Token {
    fn sgr(self) -> &'static str {
        match self {
            Token::Title | Token::Hint => "1;36",
            Token::Heading => "1",
            Token::Command => "36",
            Token::Ok => "32",
            Token::Warn => "33",
            Token::Error => "1;31",
            Token::Dim => "2",
        }
    }
}

/// The colour decision from the environment and whether the stream is a terminal.
pub fn decide(no_color: Option<OsString>, force: Option<OsString>, term: Option<OsString>, terminal: bool) -> bool {
    if no_color.is_some_and(|value| !value.is_empty()) { return false; }
    if force.is_some_and(|value| !value.is_empty() && value != "0") { return true; }
    terminal && term.is_none_or(|term| term != "dumb")
}

/// Whether text written to `stream` is coloured. Unit tests always print plain.
pub fn color(stream: Stream) -> bool {
    static OUT: OnceLock<bool> = OnceLock::new();
    static ERR: OnceLock<bool> = OnceLock::new();
    if cfg!(test) { return false; }
    let terminal = || match stream { Stream::Out => std::io::stdout().is_terminal(), Stream::Err => std::io::stderr().is_terminal() };
    let env = std::env::var_os;
    *match stream { Stream::Out => &OUT, Stream::Err => &ERR }
        .get_or_init(|| decide(env("NO_COLOR"), env("CLICOLOR_FORCE"), env("TERM"), terminal()))
}

/// `text` styled as `token` when `on`.
pub fn style(token: Token, text: &str, on: bool) -> String {
    if on && !text.is_empty() { format!("\x1b[{}m{text}\x1b[0m", token.sgr()) } else { text.to_owned() }
}

/// `text` styled as `token` for `stream`.
pub fn paint(token: Token, text: &str, stream: Stream) -> String { style(token, text, color(stream)) }

/// Imperative openings that make a sentence a next step rather than a fact.
const STEPS: &[&str] = &[
    "Add", "Check", "Choose", "Close", "Copy", "Create", "Delete", "Disable", "Edit", "Enable", "Export", "Fix",
    "Free", "Give", "Inspect", "Install", "Keep", "Move", "Omit", "Open", "Pass", "Preserve", "Rebuild", "Reinstall",
    "Remove", "Rerun", "Restart", "Retry", "Run", "See", "Select", "Set", "Start", "Stop", "Supply", "Try", "Unset",
    "Update", "Use", "Wait",
];

fn step(sentence: &str) -> bool {
    STEPS.iter().any(|verb| sentence.strip_prefix(verb).is_some_and(|rest| rest.starts_with([' ', ',', ':'])))
}

/// Split one line into sentences at `. `, `? ` or `! ` before an uppercase
/// letter, never inside double quotes (quoted paths keep their dots).
fn sentences(line: &str) -> Vec<&str> {
    let (mut parts, mut start, mut quoted, mut escaped) = (Vec::new(), 0, false, false);
    let bytes = line.as_bytes();
    for (index, &byte) in bytes.iter().enumerate() {
        if escaped { escaped = false; continue; }
        match byte {
            b'\\' if quoted => escaped = true,
            b'"' => quoted = !quoted,
            b'.' | b'?' | b'!' if !quoted && bytes.get(index + 1) == Some(&b' ')
                && bytes.get(index + 2).is_some_and(u8::is_ascii_uppercase) => {
                parts.push(&line[start..=index]);
                start = index + 2;
            }
            _ => {}
        }
    }
    parts.push(&line[start..]);
    parts
}

/// A trailing `; next step` clause outside quotes, as (fact, Next step).
fn clause(sentence: &str) -> Option<(String, String)> {
    let mut quoted = false;
    let mut found = None;
    for (index, byte) in sentence.bytes().enumerate() {
        match byte {
            b'"' => quoted = !quoted,
            b';' if !quoted && sentence[index + 1..].starts_with(' ') => found = Some(index),
            _ => {}
        }
    }
    let index = found?;
    let mut tail = sentence[index + 2..].chars();
    let next: String = tail.next()?.to_uppercase().chain(tail).collect();
    step(&next).then(|| (format!("{}.", &sentence[..index]), next))
}

/// An error as people read it: `error:` and what happened (and why), then the
/// trailing next steps under `hint:`. Every word is kept, in order; a final
/// "; run …" clause becomes its own capitalised sentence.
pub fn error_block(message: &str, on: bool) -> String {
    let mut lines: Vec<&str> = message.trim_end().lines().collect();
    let last = lines.pop().unwrap_or("");
    let mut parts: Vec<String> = sentences(last).into_iter().map(str::to_owned).collect();
    let mut split = parts.len();
    while split > 1 && step(&parts[split - 1]) { split -= 1; }
    // A lone imperative line after earlier lines is the next step; a lone
    // imperative message is itself what to fix.
    if split == 1 && !lines.is_empty() && step(&parts[0]) { split = 0; }
    if split == parts.len() && !step(&parts[split - 1]) {
        if let Some((fact, next)) = clause(&parts[split - 1]) {
            parts[split - 1] = fact;
            parts.push(next);
        }
    }
    let hints = parts.split_off(split);
    let body = parts.join(" ");
    if !body.is_empty() { lines.push(&body); }
    let mut out = format!("{} {}", style(Token::Error, "error:", on), lines.first().copied().unwrap_or(""));
    for line in lines.iter().skip(1) { out.push_str(&format!("\n       {line}")); }
    if !hints.is_empty() { out.push_str(&format!("\n{} {}", style(Token::Hint, " hint:", on), hints.join(" "))); }
    out
}

/// An error split the way [`error_block`] shows it, without labels or
/// colour: what happened (one line), and the next step if it names one.
pub fn split(message: &str) -> (String, Option<String>) {
    let block = error_block(message, false);
    let (fact, hint) = match block.split_once("\n hint: ") { Some((fact, hint)) => (fact, Some(hint.to_owned())), None => (block.as_str(), None) };
    let fact = fact.strip_prefix("error: ").unwrap_or(fact).replace("\n       ", " ");
    (fact, hint)
}

/// Rows of `label  value`, each row indented, labels padded to one column.
pub fn table(rows: &[(&str, String)], indent: usize) -> String { table_at(rows, indent, 0) }

/// [`table`] with the label column at least `width` wide, so several tables
/// can share one column.
pub fn table_at(rows: &[(&str, String)], indent: usize, width: usize) -> String {
    let width = rows.iter().map(|(label, _)| label.chars().count()).fold(width, usize::max);
    rows.iter().map(|(label, value)| {
        let pad = width.saturating_sub(label.chars().count());
        format!("{}{label}{}  {value}", " ".repeat(indent), " ".repeat(pad)).trim_end().to_owned() + "\n"
    }).collect()
}

/// Greedy word wrap at `width` columns for text that starts at column
/// `indent.len()`; continuation lines start with `indent`.
pub fn wrap(text: &str, width: usize, indent: &str) -> String {
    let mut out = String::new();
    let (start, mut column) = (indent.len(), indent.len());
    for word in text.split(' ').filter(|word| !word.is_empty()) {
        let length = word.chars().count();
        if column > start && column + 1 + length > width {
            out.push('\n');
            out.push_str(indent);
            column = start;
        } else if column > start {
            out.push(' ');
            column += 1;
        }
        out.push_str(word);
        column += length;
    }
    out
}

/// One help page: a one-line summary, examples, usage lines, an options table and notes.
pub struct Page<'a> {
    pub summary: &'a str,
    pub usage: &'a [&'a str],
    pub options: &'a [(&'a str, &'a str)],
    pub examples: &'a [&'a str],
    pub notes: &'a [&'a str],
}

/// Help text width; pages wrap here regardless of the terminal.
pub const WIDTH: usize = 80;

impl Page<'_> {
    /// The page as text, headings styled when `on`.
    pub fn render(&self, on: bool) -> String {
        let heading = |name: &str| style(Token::Heading, name, on);
        let mut out = format!("{}\n\n", wrap(self.summary, WIDTH, ""));
        // Examples come first: most readers want a command to paste.
        if !self.examples.is_empty() {
            out.push_str(&format!("{}\n", heading("Examples:")));
            for example in self.examples { out.push_str(&format!("  {example}\n")); }
            out.push('\n');
        }
        for (index, usage) in self.usage.iter().enumerate() {
            let label = if index == 0 { heading("Usage:") } else { "      ".into() };
            out.push_str(&format!("{label} {}\n", wrap(usage, WIDTH, "       ")));
        }
        if !self.options.is_empty() {
            out.push_str(&format!("\n{}\n", heading("Options:")));
            let width = self.options.iter().map(|(flag, _)| flag.chars().count()).filter(|&n| n <= 26).max().unwrap_or(0);
            let indent = " ".repeat(width + 4);
            for (flag, text) in self.options {
                let length = flag.chars().count();
                let lead = if length > width { format!("  {flag}\n{indent}") } else { format!("  {flag}{}  ", " ".repeat(width - length)) };
                out.push_str(&format!("{lead}{}\n", wrap(text, WIDTH, &indent)));
            }
        }
        if !self.notes.is_empty() {
            out.push_str(&format!("\n{}\n", heading("Notes:")));
            for note in self.notes { out.push_str(&format!("  - {}\n", wrap(note, WIDTH, "    "))); }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn colour_follows_no_color_force_term_and_terminal() {
        let s = |v: &str| Some(OsString::from(v));
        assert!(decide(None, None, s("xterm-256color"), true));
        assert!(decide(None, None, None, true));
        assert!(!decide(None, None, s("xterm"), false), "pipes are plain");
        assert!(!decide(None, None, s("dumb"), true));
        assert!(!decide(s("1"), None, s("xterm"), true));
        assert!(decide(s(""), None, s("xterm"), true), "an empty NO_COLOR is unset");
        assert!(decide(None, s("1"), s("dumb"), false));
        assert!(!decide(None, s("0"), None, false));
        assert!(!decide(s("1"), s("1"), None, true), "NO_COLOR wins over CLICOLOR_FORCE");
        assert_eq!(style(Token::Error, "error:", true), "\x1b[1;31merror:\x1b[0m");
        assert_eq!(style(Token::Error, "error:", false), "error:");
        assert!(!color(Stream::Out) && !color(Stream::Err));
    }
    #[test]
    fn errors_split_what_happened_from_the_next_step() {
        for (message, expected) in [
            ("Unknown command \"x\". Run vade --help to see the commands.",
                "error: Unknown command \"x\".\n hint: Run vade --help to see the commands."),
            ("Set --mtp to a whole number from 0 to 3.", "error: Set --mtp to a whole number from 0 to 3."),
            ("Config is missing: \"/a. B/config\". Copy x, edit it, and set VADE_CONFIG. See INSTALL.md.",
                "error: Config is missing: \"/a. B/config\".\n hint: Copy x, edit it, and set VADE_CONFIG. See INSTALL.md."),
            ("Port 18080 is in use; close the application using it and retry vade up.",
                "error: Port 18080 is in use.\n hint: Close the application using it and retry vade up."),
            ("No endpoint; run vade launch (or vade run x) or supply --url URL.",
                "error: No endpoint.\n hint: Run vade launch (or vade run x) or supply --url URL."),
            ("Bad \"a; run b\" value.", "error: Bad \"a; run b\" value."),
            ("Memory is low; the engine refused it.", "error: Memory is low; the engine refused it."),
            ("Use x; run y.", "error: Use x; run y."),
            ("The engine refused --context 65536 for memory:\nout of memory\nChoose a smaller --context. Run vade doctor.",
                "error: The engine refused --context 65536 for memory:\n       out of memory\n hint: Choose a smaller --context. Run vade doctor."),
            ("A is wrong. Run b. Then c happened.", "error: A is wrong. Run b. Then c happened."),
            ("Startup failed.\nRun vade logs.", "error: Startup failed.\n hint: Run vade logs."),
        ] { assert_eq!(error_block(message, false), expected, "{message}"); }
        assert_eq!(split("Port 1 is in use; close it and retry vade up."), ("Port 1 is in use.".into(), Some("Close it and retry vade up.".into())));
        assert_eq!(split("Set --mtp to 0..3."), ("Set --mtp to 0..3.".into(), None));
        assert_eq!(error_block("A. Run b.", true), "\x1b[1;31merror:\x1b[0m A.\n\x1b[1;36m hint:\x1b[0m Run b.");
    }
    #[test]
    fn tables_and_pages_align_and_wrap() {
        assert_eq!(table_at(&[("Model", "m".into()), ("Endpoint", "http://x".into()), ("Empty", String::new())], 2, 0),
            "  Model     m\n  Endpoint  http://x\n  Empty\n");
        assert_eq!(table_at(&[("a", "1".into())], 0, 3), "a    1\n");
        assert_eq!(wrap("aa bb cc", 5, ""), "aa bb\ncc");
        assert_eq!(wrap("aa bb cc", 7, "  "), "aa bb\n  cc");
        let page = Page { summary: "Do a thing.", usage: &["vade x [OPTIONS]"], options: &[("--a N", "First"), ("--bb", "Second option text")],
            examples: &["vade x --a 1"], notes: &["One note."] };
        assert_eq!(page.render(false), "Do a thing.\n\nExamples:\n  vade x --a 1\n\nUsage: vade x [OPTIONS]\n\nOptions:\n  --a N  First\n  --bb   Second option text\n\nNotes:\n  - One note.\n");
        let words = "word ".repeat(40);
        let long = Page { summary: &words, usage: &[&words], options: &[("--flag", &words)], examples: &[], notes: &[&words] };
        for line in long.render(false).lines() {
            assert!(line.chars().count() <= WIDTH, "{line}");
        }
    }
}
