//! Terminal-safe display text, approval escaping and fixed character cell widths.
use std::io;

/// Display only: keep indentation on the first visible line, but suppress
/// blank lines left before an answer by hidden reasoning. History stays exact.
pub fn answer_text(text: &str) -> &str {
    let leading = text.len() - text.trim_start_matches(char::is_whitespace).len();
    if leading == text.len() {
        return "";
    }
    let start = text[..leading].rfind(['\r', '\n']).map_or(0, |at| at + 1);
    &text[start..]
}

#[derive(Default)]
pub(crate) struct AnswerPrefix {
    pending: String,
    started: bool,
}
impl AnswerPrefix {
    pub(crate) fn visible<'a>(&mut self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.started {
            return std::borrow::Cow::Borrowed(text);
        }
        self.pending.push_str(text);
        if self.pending.trim().is_empty() {
            return std::borrow::Cow::Borrowed("");
        }
        self.started = true;
        let mut full = std::mem::take(&mut self.pending);
        let start = full.len() - answer_text(&full).len();
        full.drain(..start);
        std::borrow::Cow::Owned(full)
    }
}

/// Prevent terminal escape/control injection from model output, filenames and
/// pasted input. Keep only text, newline and tab; remove directional overrides.
pub fn sanitize(text: &str) -> String {
    text.chars().filter(|c| (!c.is_control() || *c == '\n' || *c == '\t')
        && !matches!(*c, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')).collect()
}

/// Approval text must represent every nonprinting character visibly. Rust's Unicode
/// printability tables also cover combining marks, format characters, fillers
/// and variation selectors that `is_control` and our cell-width table miss.
/// LF is the preview's structural line separator; tabs must be escaped.
pub fn preview_char_needs_escape(c: char) -> bool {
    if matches!(c, '\n' | '\\' | '\'' | '"') {
        return false;
    }
    c.is_control()
        || (c.is_whitespace() && c != ' ')
        || c.escape_debug().next() == Some('\\')
        || cell_width(c) == 0
        || c == '\u{2800}' // Braille blank is printable, but has no visible dots.
}

/// Escape raw preview content once, before trusted headings or styling are
/// added. Literal backslashes are doubled so `\\u{200b}` cannot impersonate a
/// rendered zero-width space. The limit applies to the expanded display bytes.
pub fn escape_preview(text: &str, max_bytes: usize) -> io::Result<String> {
    let mut shown = String::with_capacity(text.len().min(max_bytes));
    for c in text.chars() {
        let replacement = match c {
            '\\' => Some("\\\\"),
            '\r' => Some("\\r"),
            '\t' => Some("\\t"),
            _ => None,
        };
        let unicode = replacement.is_none() && preview_char_needs_escape(c);
        let bytes = replacement.map_or_else(
            || {
                if unicode {
                    c.escape_unicode().count()
                } else {
                    c.len_utf8()
                }
            },
            str::len,
        );
        if shown.len().saturating_add(bytes) > max_bytes {
            return Err(io::Error::other("escaped preview exceeds display limit"));
        }
        if let Some(replacement) = replacement {
            shown.push_str(replacement);
        } else if unicode {
            shown.extend(c.escape_unicode());
        } else {
            shown.push(c);
        }
    }
    Ok(shown)
}

/// The last three lines of a reply, sanitized for display and clipped to its
/// last 600 characters.
pub fn reply_tail(text: &str) -> String {
    let mut lines: Vec<_> = text.trim().lines().rev().take(3).collect();
    lines.reverse();
    let text = sanitize(&lines.join("\n"));
    let start = text.char_indices().rev().nth(599).map_or(0, |(i, _)| i);
    format!("{}{}", if start > 0 { "…" } else { "" }, &text[start..])
}

pub(super) fn cell_width(c: char) -> usize {
    match c {
        '\u{0300}'..='\u{036f}'
        | '\u{1ab0}'..='\u{1aff}'
        | '\u{1dc0}'..='\u{1dff}'
        | '\u{200b}'..='\u{200d}'
        | '\u{20d0}'..='\u{20ff}'
        | '\u{fe00}'..='\u{fe0f}'
        | '\u{fe20}'..='\u{fe2f}'
        | '\u{e0100}'..='\u{e01ef}' => 0,
        '\u{1100}'..='\u{115f}'
        | '\u{2329}'..='\u{232a}'
        | '\u{2e80}'..='\u{a4cf}'
        | '\u{ac00}'..='\u{d7a3}'
        | '\u{f900}'..='\u{faff}'
        | '\u{fe10}'..='\u{fe19}'
        | '\u{fe30}'..='\u{fe6f}'
        | '\u{ff01}'..='\u{ff60}'
        | '\u{ffe0}'..='\u{ffe6}'
        | '\u{1f000}'..='\u{1faff}'
        | '\u{20000}'..='\u{3ffff}' => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::{render, MAX_APPROVAL_BYTES};

    #[test]
    fn preview_escapes_unicode_controls_and_literal_lookalikes() {
        let invisible = [
            '\0',
            '\u{001b}',
            '\u{0085}',
            '\r',
            '\t',
            '\u{00ad}',
            '\u{00a0}',
            '\u{0301}',
            '\u{034f}',
            '\u{0483}',
            '\u{05b0}',
            '\u{064b}',
            '\u{0890}',
            '\u{115f}',
            '\u{1160}',
            '\u{17b4}',
            '\u{180f}',
            '\u{200b}',
            '\u{200c}',
            '\u{200d}',
            '\u{200e}',
            '\u{2028}',
            '\u{202e}',
            '\u{2060}',
            '\u{2065}',
            '\u{2800}',
            '\u{3164}',
            '\u{fe0f}',
            '\u{feff}',
            '\u{ffa0}',
            '\u{fff0}',
            '\u{fff9}',
            '\u{13440}',
            '\u{1d173}',
            '\u{1d185}',
            '\u{e0000}',
            '\u{e0001}',
            '\u{e007f}',
            '\u{e0100}',
            '\u{e0fff}',
        ];
        for c in invisible {
            assert!(preview_char_needs_escape(c), "U+{:04X}", c as u32);
            let raw = format!("before{c}after");
            let shown = escape_preview(&raw, 64).unwrap();
            assert!(!shown.contains(c), "raw U+{:04X}", c as u32);
            assert!(!shown.chars().any(preview_char_needs_escape));
            assert_eq!(sanitize(&shown), shown);
            assert_eq!(render::plain(&render::diff(&shown, true)), shown);
            assert_ne!(shown, escape_preview(&shown, 128).unwrap());
        }
        assert_eq!(
            escape_preview("é 中 🦀 '\"\n", 64).unwrap(),
            "é 中 🦀 '\"\n"
        );
        assert_eq!(
            escape_preview("literal\\u{200b}\nactual\u{200b}\n", 64).unwrap(),
            "literal\\\\u{200b}\nactual\\u{200b}\n"
        );
    }

    #[test]
    fn preview_escape_limit_counts_expanded_bytes_and_utf8_boundaries() {
        let raw = "é\u{200b}\t\\\0";
        let expected = "é\\u{200b}\\t\\\\\\u{0}";
        assert_eq!(escape_preview(raw, expected.len()).unwrap(), expected);
        assert!(escape_preview(raw, expected.len() - 1).is_err());
        assert!(escape_preview("é", 1).is_err());
        assert_eq!(escape_preview("", 0).unwrap(), "");
        assert!(escape_preview(&"\0".repeat(MAX_APPROVAL_BYTES), MAX_APPROVAL_BYTES).is_err());
    }

    #[test]
    fn untrusted_text_cannot_control_the_terminal() {
        assert_eq!(sanitize("x\x1b[2Jy\x07\r\u{202e}z\n\t"), "x[2Jyz\n\t");
    }
}
