//! The plain forms of shell text: the only text that the checker can call
//! safe. A script must be made of words, plain quotes, `$name`, simple
//! `${...}`, plain `$((...))` and `$(...)` with plain words inside, and the
//! operators `|`, `||`, `&&`, `;`, `&` and file redirections. bash, sh and
//! zsh read these forms the same way, so the checker reads them as the shell
//! does. Anything else (backquotes, `$'...'`, a comment, parentheses, a
//! here-document or here-string, a line continuation, `${...}` with quotes
//! or nested expansions) needs approval, whatever the rest of the check
//! found. This check does not decide what a command does; it only stops a
//! form that the checker may misread from being called safe.

/// The operators a plain script may use.
const OPERATORS: &[&str] = &["|", "||", "&&", ";", "&", ">", ">>", "<", ">&", "&>", "&>>", ">|", "|&", "<&"];

/// True when `text` uses only plain forms.
pub(super) fn plain(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut at = 0;
    // The unquoted text of the current word, and whether it has other parts.
    let mut bare = String::new();
    let mut other = false;
    // True where a command starts; the open `( ... )` subshells.
    let mut start = true;
    let mut depth = 0usize;
    while let Some(&c) = chars.get(at) {
        match c {
            ' ' | '\t' | '\n' => {
                if !ended(&mut bare, &mut other) {
                    return false;
                }
                start |= c == '\n';
                at += 1;
            }
            '|' | '&' | ';' | '<' | '>' => {
                if !ended(&mut bare, &mut other) {
                    return false;
                }
                let end = chars[at..].iter().position(|c| !"|&;<>".contains(*c)).map_or(chars.len(), |n| at + n);
                let op: String = chars[at..end].iter().collect();
                if !OPERATORS.contains(&op.as_str()) {
                    return false;
                }
                // `<(...)` and `>(...)`: the same plain text as `$(...)`.
                if matches!(op.as_str(), "<" | ">") && chars.get(end) == Some(&'(') {
                    let Some(close) = substitution(&chars, end + 1) else { return false };
                    at = close;
                    other = true;
                    start = false;
                    continue;
                }
                start = !op.contains(['<', '>']);
                at = end;
            }
            // A subshell where a command starts; `((` is arithmetic, and a
            // `(` after a word is a function, an array or a zsh glob flag.
            '(' if start && bare.is_empty() && !other && chars.get(at + 1) != Some(&'(') => {
                depth += 1;
                at += 1;
            }
            ')' if depth > 0 => {
                if !ended(&mut bare, &mut other) || chars.get(at + 1).is_some_and(|&n| !(n.is_whitespace() || "|&;<>)".contains(n))) {
                    return false;
                }
                depth -= 1;
                start = false;
                at += 1;
            }
            '\'' => {
                let Some(close) = chars[at + 1..].iter().position(|&c| c == '\'') else { return false };
                at += close + 2;
                other = true;
                start = false;
            }
            '"' => {
                let Some(end) = double(&chars, at + 1) else { return false };
                at = end;
                other = true;
                start = false;
            }
            '\\' => {
                if chars.get(at + 1).is_none_or(|&n| n == '\n') {
                    return false;
                }
                at += 2;
                other = true;
                start = false;
            }
            '$' => {
                let Some(end) = dollar(&chars, at, false) else { return false };
                at = end;
                other = true;
                start = false;
            }
            '#' if bare.is_empty() && !other => return false,
            '`' | '(' | ')' => return false,
            c if c.is_control() => return false,
            _ => {
                bare.push(c);
                start = false;
                at += 1;
            }
        }
    }
    depth == 0 && ended(&mut bare, &mut other)
}

/// The end of a word: false for a lone `{` or `}`, a group or a zsh loop body.
fn ended(bare: &mut String, other: &mut bool) -> bool {
    let group = !*other && matches!(bare.as_str(), "{" | "}");
    bare.clear();
    *other = false;
    !group
}

/// The end of a double-quoted string that starts at `at`, after its quote.
fn double(chars: &[char], mut at: usize) -> Option<usize> {
    while let Some(&c) = chars.get(at) {
        match c {
            '"' => return Some(at + 1),
            '\\' if chars.get(at + 1).is_some_and(|&n| n != '\n') => at += 2,
            '\\' | '`' => return None,
            '$' => at = dollar(chars, at, true)?,
            _ => at += 1,
        }
    }
    None
}

/// The end of the `$` form at `at`.
fn dollar(chars: &[char], at: usize, quoted: bool) -> Option<usize> {
    let name = |c: char| c.is_ascii_alphanumeric() || c == '_';
    match chars.get(at + 1).copied() {
        None => Some(at + 1),
        Some(c) if c.is_whitespace() || (quoted && c == '"') => Some(at + 1),
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {
            Some(chars[at + 1..].iter().position(|&c| !name(c)).map_or(chars.len(), |n| at + 1 + n))
        }
        Some(c) if c.is_ascii_digit() || "@*#?$!-".contains(c) => Some(at + 2),
        Some('{') => brace(chars, at + 2),
        Some('(') if chars.get(at + 2) == Some(&'(') => arithmetic(chars, at + 3),
        Some('(') => substitution(chars, at + 2),
        _ => None,
    }
}

/// `${name}`, `${#name}`, `${name[0]}`, `${@}` and the like, with an
/// optional operator and a plain word: `${x:-default}`, `${f%.rs}`.
fn brace(chars: &[char], mut at: usize) -> Option<usize> {
    let name = |c: char| c.is_ascii_alphanumeric() || c == '_';
    if chars.get(at) == Some(&'#') && chars.get(at + 1).is_some_and(|&c| c != '}') {
        at += 1;
    }
    let start = at;
    while chars.get(at).is_some_and(|&c| name(c)) {
        at += 1;
    }
    if at == start {
        at += usize::from(chars.get(at).is_some_and(|&c| "@*#?$!-".contains(c)));
        if at == start {
            return None;
        }
    }
    if chars.get(at) == Some(&'[') {
        let close = chars[at..].iter().position(|&c| c == ']')? + at;
        let index: String = chars[at + 1..close].iter().collect();
        if !(matches!(index.as_str(), "@" | "*") || (!index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))) {
            return None;
        }
        at = close + 1;
    }
    let close = chars[at..].iter().position(|&c| c == '}')? + at;
    let rest: String = chars[at..close].iter().collect();
    let operators = [":-", ":+", ":?", "-", "+", "?", "##", "#", "%%", "%", "//", "/", "^^", "^", ",,", ","];
    let word = if rest.is_empty() {
        ""
    } else if let Some(offset) = rest.strip_prefix(':').filter(|r| !r.starts_with(['-', '+', '?', '='])) {
        // `${x:2:3}`: numbers only.
        if !offset.chars().all(|c| c.is_ascii_digit() || matches!(c, ':' | ' ')) {
            return None;
        }
        ""
    } else {
        let op = operators.iter().find(|op| rest.starts_with(**op))?;
        &rest[op.len()..]
    };
    word.chars().all(|c| c.is_alphanumeric() || "._-/+,:=@%~*?^ ".contains(c)).then_some(close + 1)
}

/// `$((...))` with names, numbers and operators only, after `$((`.
fn arithmetic(chars: &[char], mut at: usize) -> Option<usize> {
    let mut depth = 0;
    while let Some(&c) = chars.get(at) {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => return (chars.get(at + 1) == Some(&')')).then_some(at + 2),
            ')' => depth -= 1,
            // `$n`, `$1` and `$#`; `#` also gives a base (`16#ff`).
            '$' if chars.get(at + 1).is_some_and(|&n| n.is_ascii_alphanumeric() || n == '_' || n == '#') => {}
            '#' if at > 0 && (chars[at - 1].is_ascii_digit() || chars[at - 1] == '$') => {}
            c if c.is_ascii_alphanumeric() || c == '_' || c == ' ' || c == '\t' || "+-*/%<>=!&|^~?:,".contains(c) => {}
            _ => return None,
        }
        at += 1;
    }
    None
}

/// `$(...)` after `$(`: plain words and operators, no quotes, no nested
/// expansion and no parentheses, so the first `)` ends it in every shell.
fn substitution(chars: &[char], at: usize) -> Option<usize> {
    let close = chars[at..].iter().position(|&c| c == ')')? + at;
    let inner: String = chars[at..close].iter().collect();
    if inner.contains(['\'', '"', '`', '$', '\\', '(', '{', '}', '#', '\n']) || !plain(&inner) {
        return None;
    }
    Some(close + 1)
}
