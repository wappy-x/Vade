//! First-party terminal presentation. Only this module creates SGR sequences;
//! untrusted Markdown, input and tool text are sanitized before styling.
use super::{cell_width, sanitize};

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
        } else if let Some(ctx) = part
            .strip_prefix("ctx ")
            .and_then(|c| c.strip_suffix('%'))
        {
            ctx.parse::<u64>().ok().map(|used| (used, 100))
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

pub fn plain(text: &str) -> String {
    units(text)
        .filter(|(s, _)| !s.starts_with('\x1b'))
        .map(|(s, _)| s)
        .collect()
}

pub fn width(text: &str) -> usize {
    units(text).map(|(_, n)| n).sum()
}

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

pub fn wrap(text: &str, columns: usize) -> Vec<String> {
    let columns = columns.max(1);
    let mut rows = vec![String::new()];
    let mut cells = 0;
    let mut active = String::new();
    for (s, n) in units(text) {
        if s.starts_with('\x1b') {
            if s == "\x1b[0m" {
                active.clear();
            } else {
                active.push_str(s);
            }
            rows.last_mut().unwrap().push_str(s);
            continue;
        }
        if n > columns {
            continue;
        }
        if s == "\n" || cells + n > columns {
            if !active.is_empty() {
                rows.last_mut().unwrap().push_str("\x1b[0m");
            }
            rows.push(active.clone());
            cells = 0;
        }
        if s != "\n" {
            rows.last_mut().unwrap().push_str(s);
            cells += n;
        }
    }
    rows
}

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

/// Labels carry speaker identity even with NO_COLOR or a light terminal theme.
pub fn user(text: &str, color: bool) -> String {
    format!("{}\n{}", header("You", &super::clock::now().0, "1;36", color), sanitize(text))
}

pub fn header(label: &str, time: &str, sgr: &str, color: bool) -> String {
    style(&format!("── {label} · {time} ──"), sgr, color)
}

pub fn reply(text: &str, time: &str, color: bool) -> String {
    format!("{}\n{}", header("Answer", time, "1", color), markdown(text, color))
}

/// A warning block: what happened, what Vade is doing, what you can do.
/// The label carries the meaning without colour; colour adds yellow.
pub fn warning(what: &str, doing: &str, next: &str, time: &str, color: bool) -> String {
    let mut text = header("Warning", time, "1;33", color);
    for part in [what, doing, next].into_iter().filter(|p| !p.is_empty()) {
        text.push('\n');
        text.push_str(&sanitize(part));
    }
    text
}

/// Thoughts keep their start clock and a closing clock even when collapsed.
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
        "{} Thinking · {} · {count}{} · Ctrl+T {}{}",
        if expanded { "▾" } else { "▸" },
        time.0,
        elapsed.map_or_else(String::new, |d| format!(" · {}", super::clock::duration(d))),
        if expanded { "collapse" } else { "expand" },
        if limited { " · display limit" } else { "" }
    );
    let mut rows = vec![clip(&style(&label, "2;3", color), columns)];
    if expanded {
        for line in sanitize(text).replace('\t', "    ").lines() {
            rows.extend(wrap(&style(line, "2", color), columns));
        }
    }
    if let Some(end) = time.1 {
        rows.push(clip(&style(&format!("╰ {end}"), "2", color), columns));
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

pub struct EditorRows {
    pub lines: Vec<String>,
    pub cursor_row: usize,
    pub cursor_column: usize,
}

/// `hint` shows the newline keys in the top border; callers show it until
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
    let boxed = columns >= 6 && max_rows >= 3;
    let prefix = if boxed { prefix } else { "" };
    let inner = columns.saturating_sub(if boxed { 4 } else { 0 }).max(1);
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
    let available = max_rows.saturating_sub(if boxed { 2 } else { 0 }).max(1);
    let start = caret.0.saturating_sub(available - 1);
    let mut lines = Vec::new();
    if boxed {
        let hint = clip(
            if !hint {
                ""
            } else if columns >= 60 {
                " Enter sends · Shift+Enter / Option+Enter / Ctrl+J newline "
            } else {
                " Enter sends · Ctrl+J newline "
            },
            columns - 2,
        );
        lines.push(style(
            &format!("┌{hint}{}┐", "─".repeat(columns - 2 - width(&hint))),
            "2",
            color,
        ));
    }
    for (index, row) in rows.iter().enumerate().skip(start).take(available) {
        if boxed {
            let prompt = if index == 0 { prefix } else { "  " };
            lines.push(format!(
                "│{}{row}{}│",
                style(prompt, "1;36", color),
                " ".repeat(inner.saturating_sub(width(row)))
            ));
        } else {
            lines.push(row.clone());
        }
    }
    if boxed {
        lines.push(style(&format!("└{}┘", "─".repeat(columns - 2)), "2", color));
    }
    EditorRows {
        lines,
        cursor_row: caret.0 - start + usize::from(boxed),
        cursor_column: caret.1 + if boxed { 3 } else { 0 },
    }
}

pub fn approval(color: bool) -> String {
    format!(
        "{}  {} approve · {} deny · or /approve /deny",
        style("Approval required", "7", color),
        style("y", "1;32", color),
        style("n", "1;31", color)
    )
}

/// Colour a unified diff: additions green, removals red, hunk headers cyan,
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
