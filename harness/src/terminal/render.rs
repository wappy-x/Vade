//! First-party terminal presentation. Only this module creates SGR sequences;
//! untrusted Markdown, input and tool text are sanitized before styling.
use super::{cell_width, sanitize};

/// `text` wrapped in the SGR color code `sgr` when `color` is on.
pub fn style(text: &str, sgr: &str, color: bool) -> String {
    if color && !text.is_empty() {
        format!("\x1b[{sgr}m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

/// Numeric budget fields get a reverse badge near their limit. Sanitize first:
/// session/model names can share this row and must never supply terminal codes.
pub fn budget_footer(text: &str, color: bool) -> String {
    let seconds = |text: &str| -> Option<u64> {
        if let Some(minutes) = text.strip_suffix('m') {
            return minutes.parse::<u64>().ok()?.checked_mul(60);
        }
        let number = text.strip_suffix('s')?;
        if let Some((m, s)) = number.split_once('m') {
            m.parse::<u64>().ok()?.checked_mul(60)?.checked_add(s.parse().ok()?)
        } else {
            number.parse().ok()
        }
    };
    let footer = sanitize(text).split(" · ").map(|part| {
        let ratio = if let Some(round) = part.strip_prefix("round ") {
            round.split_once('/').and_then(|(a, b)| {
                a.parse::<u64>().ok().zip(b.parse::<u64>().ok())
            })
        } else if let Some(ctx) = part.strip_prefix("ctx ") {
            // `ctx 26.2k/32.8k` (counts with k/M suffixes) or `ctx 80%`.
            let tokens = |text: &str| -> Option<u64> {
                let (number, scale) = match text.strip_suffix('M') { Some(n) => (n, 1e6), None => match text.strip_suffix('k') { Some(n) => (n, 1e3), None => (text, 1.0) } };
                number.parse::<f64>().ok().map(|n| (n * scale) as u64)
            };
            ctx.strip_suffix('%').and_then(|p| p.parse::<u64>().ok()).map(|used| (used, 100))
                .or_else(|| ctx.split_once('/').and_then(|(a, b)| tokens(a).zip(tokens(b))))
        } else {
            part.split_once('/').and_then(|(a, b)| seconds(a).zip(seconds(b)))
        };
        let warning = ratio.is_some_and(|(used, limit)| {
            limit > 0 && used.saturating_mul(5) >= limit.saturating_mul(4)
        });
        if warning && color {
            format!("\x1b[7m{part}\x1b[27m")
        } else {
            part.to_owned()
        }
    }).collect::<Vec<_>>().join(" · ");
    style(&footer, "2", color)
}

// Internal strings may contain trusted SGR, never cursor movement or OSC.
fn units(text: &str) -> impl Iterator<Item = (&str, usize)> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let c = rest.chars().next()?;
        let len = if rest.starts_with("\x1b[") {
            rest.find('m').map_or(c.len_utf8(), |end| end + 1)
        } else {
            c.len_utf8()
        };
        let (unit, tail) = rest.split_at(len);
        rest = tail;
        Some((unit, if c == '\x1b' { 0 } else { cell_width(c) }))
    })
}

/// `text` without its escape sequences.
pub fn plain(text: &str) -> String {
    units(text)
        .filter(|(s, _)| !s.starts_with('\x1b'))
        .map(|(s, _)| s)
        .collect()
}

/// The terminal cells `text` occupies, ignoring escape sequences.
pub fn width(text: &str) -> usize {
    units(text).map(|(_, n)| n).sum()
}

/// The longest prefix of `text` that fits in `columns` cells.
pub fn clip(text: &str, columns: usize) -> String {
    let mut out = String::new();
    let mut cells = 0;
    for (s, n) in units(text) {
        if cells + n > columns {
            break;
        }
        out.push_str(s);
        cells += n;
    }
    if out.contains('\x1b') {
        out.push_str("\x1b[0m");
    }
    out
}

/// `text` split into rows of at most `columns` cells at word boundaries,
/// carrying active styles (and a leading speaker bar) across row breaks.
pub fn wrap(text: &str, columns: usize) -> Vec<String> {
    wrap_at(text, columns, true)
}

/// [`wrap`] at exact cell boundaries, never moving a word: for approval
/// previews, where a moved fragment could read as a diff line of its own.
pub fn wrap_exact(text: &str, columns: usize) -> Vec<String> {
    wrap_at(text, columns, false)
}

fn wrap_at(text: &str, columns: usize, words: bool) -> Vec<String> {
    let columns = columns.max(1);
    let mut rows = Vec::new();
    let mut active = String::new();
    let lines: Vec<&str> = text.split('\n').collect();
    for (index, line) in lines.iter().enumerate() {
        // A line that starts with a speaker bar keeps it on every continuation
        // row that has room for it. Exact previews never add characters.
        let hang = hanging_bar(line).filter(|(_, n)| words && *n < columns);
        // The bar cells at the start of the current row.
        let mut bar_cells = hang.as_ref().map_or(0, |(_, n)| *n);
        let mut row = active.clone();
        let mut cells = 0;
        // Where the row may break: just after its last space, with the style
        // active there.
        let mut space: Option<(usize, String)> = None;
        for (s, n) in units(line) {
            if s.starts_with('\x1b') {
                if s == "\x1b[0m" { active.clear(); } else { active.push_str(s); }
                row.push_str(s);
                continue;
            }
            if n > columns {
                continue;
            }
            // A space that would overflow ends the row itself.
            let ends_row = words && s == " " && cells + n > columns;
            if cells + n > columns {
                let (mut bar, mut next_bar) = hang.clone().unwrap_or_default();
                // Move the unfinished word down when it fits on a fresh row;
                // a word longer than a row breaks where it must.
                let split = space.take().filter(|(at, _)| !ends_row && next_bar + width(&row[*at..]) + n <= columns);
                // A glyph too wide to follow the bar starts its row bare.
                if split.is_none() && next_bar + n > columns { (bar, next_bar) = (String::new(), 0); }
                let (mut head, tail, style) = match split {
                    // The space stays at the end of the row, so joined rows
                    // keep every character (approval previews must be exact).
                    Some((at, style)) => (row[..at].to_owned(), row[at..].to_owned(), style),
                    None => (std::mem::take(&mut row), String::new(), active.clone()),
                };
                if !style.is_empty() { head.push_str("\x1b[0m"); }
                rows.push(head);
                bar_cells = next_bar;
                cells = bar_cells + width(&tail);
                row = format!("{bar}{style}{tail}");
                if ends_row { continue; }
            }
            row.push_str(s);
            cells += n;
            if words && s == " " && cells > bar_cells + 1 {
                space = Some((row.len(), active.clone()));
            }
        }
        if index + 1 < lines.len() && !active.is_empty() {
            row.push_str("\x1b[0m");
        }
        rows.push(row);
    }
    rows
}

/// The leading `▌ ` or `┆ ` of `text` with its styling, and its cells.
fn hanging_bar(text: &str) -> Option<(String, usize)> {
    let visible = plain(text);
    if !(visible.starts_with("▌ ") || visible.starts_with("┆ ")) { return None; }
    let (mut bar, mut cells) = (String::new(), 0);
    for (s, n) in units(text) {
        if cells == 2 && !s.starts_with('\x1b') { break; }
        bar.push_str(s);
        cells += n;
    }
    if bar.contains('\x1b') && !bar.ends_with("\x1b[0m") { bar.push_str("\x1b[0m"); }
    Some((bar, cells))
}

/// A reply rendered for the terminal: sanitized, fenced code kept verbatim
/// (dimmed), headings, rules, quotes and bullets drawn, and inline styling
/// when `color` is on.
pub fn markdown(text: &str, color: bool) -> String {
    let text = sanitize(text).replace('\t', "    ");
    let mut out = Vec::new();
    let mut fence: Option<char> = None;
    for line in text.trim_end().lines() {
        let trimmed = line.trim_start();
        let marker = if trimmed.starts_with("```") {
            Some('`')
        } else if trimmed.starts_with("~~~") {
            Some('~')
        } else {
            None
        };
        if let Some(mark) = marker {
            if fence == Some(mark) {
                fence = None;
                out.push(style(line, "2", color));
                continue;
            }
            if fence.is_none() {
                fence = Some(mark);
                out.push(style(line, "2", color));
                continue;
            }
        }
        if fence.is_some() {
            out.push(style(line, "2", color));
            continue;
        }
        let heading = trimmed.bytes().take_while(|b| *b == b'#').count();
        let rule = trimmed.len() >= 3
            && trimmed
                .chars()
                .all(|c| c == '-' || c == '*' || c == '_' || c == ' ')
            && trimmed.chars().filter(|c| !c.is_whitespace()).count() >= 3
            && trimmed
                .chars()
                .filter(|c| !c.is_whitespace())
                .all(|c| c == trimmed.trim().chars().next().unwrap());
        if (1..=6).contains(&heading) && trimmed.as_bytes().get(heading) == Some(&b' ') {
            // H1 is underlined; deeper levels are bold only.
            out.push(style(
                &inline(trimmed[heading + 1..].trim_end(), color),
                if heading == 1 { "1;4" } else { "1" },
                color,
            ));
        } else if rule {
            out.push(style(&"─".repeat(24), "2", color));
        } else if let Some(quote) = trimmed
            .strip_prefix("> ")
            .or_else(|| trimmed.strip_prefix('>'))
        {
            out.push(format!(
                "{}{} {}",
                " ".repeat(line.len() - trimmed.len()),
                style("│", "2", color),
                style(&inline(quote.trim_start(), color), "2", color)
            ));
        } else if let Some(bullet) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("+ "))
        {
            out.push(format!(
                "{}• {}",
                " ".repeat(line.len() - trimmed.len()),
                inline(bullet, color)
            ));
        } else {
            out.push(inline(line, color));
        }
    }
    out.join("\n")
}

/// The bar in front of every row you wrote. It carries speaker identity
/// without color too: the model's answer never has a bar.
pub const USER_BAR: &str = "▌";
/// The dim dotted bar in front of expanded thinking rows, so a thought never
/// reads as part of the answer.
pub const THOUGHT_BAR: &str = "┆";

/// Your message: a bold cyan bar and bold text on every line.
pub fn user(text: &str, color: bool) -> String {
    sanitize(text)
        .split('\n')
        .map(|line| format!("{} {}", style(USER_BAR, "1;36", color), style(line, "1", color)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A complete answer: the rendered reply, with no header. Your bar above it
/// already says whose turn ended.
pub fn reply(text: &str, color: bool) -> String {
    markdown(text, color)
}

/// A warning block: what happened, what Vade is doing, what you can do.
/// The `▲ Warning` label carries the meaning without color; color adds yellow.
pub fn warning(what: &str, doing: &str, next: &str, color: bool) -> String {
    let mut text = style("▲ Warning", "1;33", color);
    for part in [what, doing, next].into_iter().filter(|p| !p.is_empty()) {
        text.push('\n');
        text.push_str(&sanitize(part));
    }
    text
}

/// A thought on one dim italic line: start and end clocks, length and time.
/// Expanded, its text follows behind a dim dotted bar.
pub fn thinking(
    text: &str,
    tokens: Option<u64>,
    elapsed: Option<std::time::Duration>,
    time: (&str, Option<&str>),
    expanded: bool,
    limited: bool,
    columns: usize,
    color: bool,
) -> Vec<String> {
    let count = tokens.map_or_else(
        || format!("≈{} tokens", super::clock::count(text.len().div_ceil(3) as u64)),
        |n| format!("{} tokens", super::clock::count(n)),
    );
    let label = format!(
        "{} thinking {}{} · {count}{} · Ctrl+T {}{}",
        if expanded { "▾" } else { "▸" },
        time.0,
        time.1.map_or_else(String::new, |end| format!("–{end}")),
        elapsed.map_or_else(String::new, |d| format!(" · {}", super::clock::duration(d))),
        if expanded { "collapse" } else { "expand" },
        if limited { " · display limit" } else { "" }
    );
    // A narrow row drops the key hint whole rather than clipping a word.
    let label = if width(&label) > columns { label.split(" · Ctrl+T").next().unwrap_or(&label).to_owned() } else { label };
    let mut rows = vec![clip(&style(&label, "2;3", color), columns)];
    if expanded {
        for line in sanitize(text).replace('\t', "    ").lines() {
            rows.extend(wrap(&format!("{} {}", style(THOUGHT_BAR, "2", color), style(line, "2;3", color)), columns));
        }
    }
    rows
}

fn inline(mut text: &str, color: bool) -> String {
    let mut out = String::new();
    while !text.is_empty() {
        let delimiter = if text.starts_with("**") {
            Some(("**", "1"))
        } else if text.starts_with('`') {
            Some(("`", "36"))
        } else if text.starts_with('*') {
            Some(("*", "3"))
        } else {
            None
        };
        // [label](url) → label, dim url. Unfinished links stay literal.
        if text.starts_with('[') {
            if let Some(close) = text.find("](") {
                let after = &text[close + 2..];
                if let Some(end) = after.find(')') {
                    let (label, url) = (&text[1..close], &after[..end]);
                    if !label.is_empty()
                        && !url.is_empty()
                        && !url.contains(char::is_whitespace)
                        && !label.contains('[')
                    {
                        out.push_str(&inline(label, color));
                        out.push_str(&style(&format!(" ({url})"), "2", color));
                        text = &after[end + 1..];
                        continue;
                    }
                }
            }
        }
        if let Some((mark, sgr)) = delimiter {
            let rest = &text[mark.len()..];
            if let Some(end) = rest.find(mark) {
                let content = if mark == "`" {
                    rest[..end].to_owned()
                } else {
                    inline(&rest[..end], color)
                };
                out.push_str(&style(&content, sgr, color));
                text = &rest[end + mark.len()..];
                continue;
            }
            // A streaming delimiter may arrive before its closing half.
            // Suppress unfinished emphasis; literal code contents stay exact.
            if mark == "**" || rest.is_empty() {
                text = rest;
                continue;
            }
        }
        let c = text.chars().next().unwrap();
        out.push(c);
        text = &text[c.len_utf8()..];
    }
    out
}

/// The complete popup is bounded by the caller's available height.
pub fn menu(
    entries: &[(&str, &str)],
    selected: usize,
    columns: usize,
    rows: usize,
    color: bool,
) -> String {
    if entries.is_empty() || rows < 3 || columns < 4 {
        return String::new();
    }
    let selected = selected.min(entries.len() - 1);
    let count = rows.saturating_sub(2).min(8).min(entries.len());
    let start = selected
        .saturating_sub(count - 1)
        .min(entries.len() - count);
    let inner = columns - 2;
    let mut out = vec![style(&format!("┌{}┐", "─".repeat(inner)), "2", color)];
    for (i, (name, description)) in entries.iter().enumerate().skip(start).take(count) {
        let name = format!("/{name}");
        let label = if columns >= 40 {
            format!("{name:<18}")
        } else {
            name.clone()
        };
        let content = format!(
            "{} {} {}",
            if i == selected { "›" } else { " " },
            label,
            description
        );
        let content = clip(&content, inner);
        let padding = " ".repeat(inner.saturating_sub(width(&content)));
        let line = if i == selected {
            style(&format!("{content}{padding}"), "7", color)
        } else {
            // Default foreground remains readable on both light and dark themes.
            let name_end = content.find('/').unwrap_or(0) + name.len();
            let name_end = name_end.min(content.len());
            format!(
                "{}{}{padding}",
                style(&content[..name_end], "1;36", color),
                &content[name_end..]
            )
        };
        out.push(format!("│{line}│"));
    }
    let hint = clip(
        &format!(
            " ↑↓ · Enter run · Tab complete · Esc close  {}/{} ",
            selected + 1,
            entries.len()
        ),
        inner,
    );
    out.push(style(
        &format!("└{hint}{}┘", "─".repeat(inner - width(&hint))),
        "2",
        color,
    ));
    out.join("\n")
}

/// The input dock's rows and where its cursor sits among them.
pub struct EditorRows {
    pub lines: Vec<String>,
    pub cursor_row: usize,
    pub cursor_column: usize,
}

/// `hint` shows the keys dimly in the empty prompt; callers show it until
/// the first message is sent, not on every frame of the session.
pub fn editor(
    text: &str,
    cursor: usize,
    prefix: &str,
    columns: usize,
    max_rows: usize,
    hint: bool,
    color: bool,
) -> EditorRows {
    // No box: the prompt marker alone marks where you type, and the dim
    // status strip above it is the only chrome.
    // Too narrow for the marker: the input alone.
    let prefix = if columns >= 4 { prefix } else { "" };
    let inner = columns.saturating_sub(width(prefix)).max(1);
    let cursor = cursor.min(text.len());
    let cursor = (0..=cursor)
        .rev()
        .find(|i| text.is_char_boundary(*i))
        .unwrap_or(0);
    let before = sanitize(&text[..cursor]).replace('\t', "  ");
    let after = sanitize(&text[cursor..]).replace('\t', "  ");
    let mut rows = vec![String::new()];
    let mut cells = 0;
    let mut caret = (0, 0);
    let mut fresh_wrap = false;
    for (part, value) in [&before, &after].into_iter().enumerate() {
        if part == 1 {
            caret = (rows.len() - 1, cells);
        }
        for c in value.chars() {
            let size = cell_width(c);
            if size > inner {
                continue;
            }
            if c == '\n' {
                if !fresh_wrap {
                    rows.push(String::new());
                }
                cells = 0;
                fresh_wrap = false;
                continue;
            }
            if cells + size > inner {
                rows.push(String::new());
                cells = 0;
            }
            rows.last_mut().unwrap().push(c);
            cells += size;
            fresh_wrap = cells == inner;
            if fresh_wrap {
                rows.push(String::new());
                cells = 0;
            }
        }
    }
    let available = max_rows.max(1);
    let start = caret.0.saturating_sub(available - 1);
    let mut lines = Vec::new();
    for (index, row) in rows.iter().enumerate().skip(start).take(available) {
        let prompt = if index == 0 { prefix.to_owned() } else { " ".repeat(width(prefix)) };
        let mut line = format!("{}{row}", style(&prompt, "1;36", color));
        // Until the first message, an empty prompt shows the keys dimly.
        if hint && text.is_empty() {
            let keys = if inner >= 50 { "Enter sends · Shift+Enter or Ctrl+J newline · / commands" } else { "Enter sends · Ctrl+J newline" };
            line.push_str(&style(&clip(keys, inner), "2", color));
        }
        lines.push(line);
    }
    EditorRows {
        lines,
        cursor_row: caret.0 - start,
        cursor_column: caret.1 + width(prefix),
    }
}

/// The approval prompt line: approve with y, deny with n.
pub fn approval(color: bool) -> String {
    format!(
        "{}  {} approve · {} deny · or /approve /deny",
        style("Approval required", "7", color),
        style("y", "1;32", color),
        style("n", "1;31", color)
    )
}

/// Color a unified diff: additions green, removals red, hunk headers cyan,
/// file headers bold, truncation notices dim. Input is sanitized text.
pub fn diff(text: &str, color: bool) -> String {
    text.lines()
        .map(|line| {
            let sgr =
                if line.starts_with("+++ ") || line.starts_with("--- ") || line.starts_with("*** ")
                {
                    "1"
                } else if line.starts_with("@@") {
                    "36"
                } else if line.starts_with('+') {
                    "32"
                } else if line.starts_with('-') {
                    "31"
                } else if line.starts_with('[') {
                    "2"
                } else {
                    ""
                };
            if sgr.is_empty() {
                line.to_owned()
            } else {
                style(line, sgr, color)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super as render;

    #[test]
    fn rows_break_between_words_and_your_bar_continues() {
        assert_eq!(render::wrap("the sky is blue", 8), ["the sky ", "is blue"]);
        assert_eq!(render::wrap("abcdefghij", 4), ["abcd", "efgh", "ij"], "a word longer than a row still breaks");
        assert_eq!(render::wrap_exact("the sky is blue", 8), ["the sky ", "is blue"]);
        assert_eq!(render::wrap_exact("-a -b -c", 4), ["-a -", "b -c"], "approval rows never move words");
        assert_eq!(render::wrap(&render::user("why is the sky blue", false), 12), ["▌ why is the", "▌ sky blue"]);
        let rows = render::wrap(&render::user("why is the sky blue", true), 12);
        assert!(rows.iter().all(|row| render::plain(row).starts_with("▌ ") && render::width(row) <= 12), "{rows:?}");
        assert_eq!(render::wrap(&render::user("a\nb", false), 20), ["▌ a", "▌ b"], "one bar per row");
        // Exact previews keep a literal bar as text and add nothing.
        assert_eq!(render::wrap_exact("▌ abcdef", 5), ["▌ abc", "def"]);
        assert_eq!(render::wrap_exact("┆ abc def", 5).concat(), "┆ abc def");
        // A wide glyph that cannot follow the bar starts its row bare.
        for (text, columns) in [("▌ 你", 3), ("▌ 你好", 3), ("▌ 🦀🦀", 3), ("▌ ab 你好世界", 4), ("▌ 🦀 x", 2)] {
            let rows = render::wrap(text, columns);
            assert!(rows.iter().all(|row| render::width(row) <= columns), "{text:?} at {columns}: {rows:?}");
            assert_eq!(rows.concat().replace("▌ ", "").replace(' ', ""), text["▌ ".len()..].replace(' ', ""), "{text:?}: every glyph kept");
        }
        assert_eq!(render::wrap("▌ 你", 3), ["▌ ", "你"]);
        assert_eq!(render::wrap("▌ 你好", 4), ["▌ 你", "▌ 好"], "a bar that fits stays");
        let thought = render::thinking("step one\nstep two", Some(9), None, ("10:00:00", Some("10:00:02")), true, false, 80, false);
        assert_eq!(thought, ["▾ thinking 10:00:00–10:00:02 · 9 tokens · Ctrl+T collapse", "┆ step one", "┆ step two"]);
    }
    #[test]
    fn diff_lines_are_colored_by_kind_and_stay_literal_without_color() {
        let sample = "*** Update File: a.rs\n--- a.rs\n+++ a.rs\n@@ -1,2 +1,2 @@\n ctx\n-old\n+new\n[context line truncated: 9 bytes]";
        assert_eq!(render::diff(sample, false), sample);
        let styled = render::diff(sample, true);
        assert!(styled.contains("\x1b[32m+new\x1b[0m"));
        assert!(styled.contains("\x1b[31m-old\x1b[0m"));
        assert!(styled.contains("\x1b[36m@@ -1,2 +1,2 @@\x1b[0m"));
        assert!(styled.contains("\x1b[1m+++ a.rs\x1b[0m"));
        assert!(styled.contains("\n ctx\n"));
        assert_eq!(render::plain(&styled), sample);
    }

    #[test]
    fn cursor_view_survives_narrow_resize_without_changing_input() {
        let text = "start 🦀 café\nsecond line";
        let cursor = "start 🦀 ca".len();
        for width in [4, 8, 20, 80] {
            let view = render::editor(text, cursor, "› ", width, 8, true, true);
            assert!(view.lines.iter().all(|line| render::width(line) <= width));
            assert!(view.cursor_column < width);
            assert!(view.cursor_row < view.lines.len());
        }
        assert_eq!(text, "start 🦀 café\nsecond line");
    }
}
