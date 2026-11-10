//! RTF to plain text. Control words are dropped, `\par` and `\line` end a
//! line and `\uN` decodes as Unicode (skipping the `\ucN` fallback
//! characters, control words included). A `\'hh` byte decodes in the code
//! page of the current font's charset (`\fcharsetN` in the font table), or
//! else in the document code page (`\ansicpgN`, `\mac`). A code page with no
//! table here gives U+FFFD and a warning. `\plain` goes back to the default
//! font. `\binN` data is skipped by its byte count. Font tables, styles,
//! pictures, field instructions and other destinations are skipped. Open
//! groups and fonts count against the work cap of the [`Budget`].
use super::{Budget, Caps, Doc, Format, ReadError, Reader, codepage, extension};

pub(super) struct RtfReader;

/// Destinations that hold no document text.
const SKIP: &[&str] = &[
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "fldinst", "themedata", "colorschememapping", "datastore",
    "latentstyles", "listtable", "listoverridetable", "rsidtbl", "generator", "xmlnstbl", "mmathPr", "filetbl", "revtbl",
    "pgdsctbl", "header", "headerl", "headerr", "headerf", "footer", "footerl", "footerr", "footerf", "bkmkstart", "bkmkend",
    "xe", "tc", "nonshppict", "blipuid", "objdata", "listtext", "pntext", "pntxta", "pntxtb", "protusertbl", "wgrffmtfilter",
    "passwordhash", "docvar", "userprops",
];

impl Reader for RtfReader {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format> {
        if bytes.starts_with(b"{\\rtf") {
            return Some(Format::Rtf);
        }
        (!name.is_empty() && extension(name) == "rtf").then_some(Format::Rtf)
    }

    fn to_markdown(&self, bytes: &[u8], _: Format, caps: &Caps) -> Result<Doc, ReadError> {
        if !bytes.starts_with(b"{\\rtf") {
            return Err(ReadError::Malformed(Format::Rtf, "the file does not start with {\\rtf".into()));
        }
        let mut budget = Budget::new(caps);
        let parsed = convert(bytes, &mut budget);
        let mut doc = Doc::new(parsed.text, 1);
        if parsed.pictures > 0 {
            doc.warnings.push(format!("{} pictures or objects", parsed.pictures));
        }
        if let (n @ 1.., page) = parsed.unknown {
            doc.warnings.push(format!("{n} bytes in code page {page} (not supported; shown as U+FFFD)"));
        }
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

#[derive(Clone, Copy)]
struct State {
    skip: bool,
    uc: usize,
    /// Inside the font table, where `\fN` and `\fcharsetN` describe fonts.
    fonttbl: bool,
    /// The font from `\fN`; its charset decides the code page of `\'hh`.
    font: Option<i64>,
}

const TOP: State = State { skip: false, uc: 1, fonttbl: false, font: None };
/// The storage of one font table entry, with the map's own share.
const FONT: usize = 2 * size_of::<(i64, u32)>();

struct Parsed {
    text: String,
    pictures: usize,
    /// Bytes shown as U+FFFD because this build has no table for their code
    /// page, and the first such code page.
    unknown: (usize, u32),
}

fn convert(b: &[u8], budget: &mut Budget) -> Parsed {
    let mut out = String::new();
    let mut stack = Vec::new();
    let mut state = TOP;
    // Fallback characters still to skip after a \uN. A control word or
    // symbol counts as one character; a brace ends the fallback.
    let mut pending = 0usize;
    let (mut pictures, mut unknown) = (0usize, (0usize, 0u32));
    // The document code page (\ansicpgN), the default font (\deffN) and the
    // code page of each font from its \fcharsetN. A font charset wins over
    // the document code page, as in Word and the macOS text system.
    let (mut page, mut deff, mut entry) = (1252u32, None, None);
    let mut fonts: std::collections::HashMap<i64, u32> = std::collections::HashMap::new();
    let page_of = |state: &State, fonts: &std::collections::HashMap<i64, u32>, page: u32, deff: Option<i64>| {
        state.font.or(deff).and_then(|f| fonts.get(&f).copied()).unwrap_or(page)
    };
    // The code page for the current font, set again when a brace or a control
    // word can change it.
    let mut now = page;
    // A high surrogate from \uN that waits for its low half. Other output
    // before the low half, or the end, makes it U+FFFD (see `lone`).
    let mut high: Option<u32> = None;
    let mut i = 0;
    // True right after "{" (and "{\*"): the next control word may name a destination.
    let mut group_start = false;
    let mut starred = false;
    while i < b.len() && !budget.full() {
        let c = b[i];
        match c {
            b'{' => {
                if !budget.reserve(size_of::<State>()) {
                    break;
                }
                stack.push(state);
                group_start = true;
                starred = false;
                pending = 0;
                i += 1;
                continue;
            }
            b'}' => {
                let font = state.font;
                state = stack.pop().inspect(|_| budget.release(size_of::<State>())).unwrap_or(TOP);
                if state.font != font {
                    now = page_of(&state, &fonts, page, deff);
                }
                pending = 0;
                group_start = false;
                i += 1;
                continue;
            }
            b'\r' | b'\n' => {
                i += 1;
                continue;
            }
            b'\\' => {}
            _ => {
                group_start = false;
                if pending > 0 {
                    pending -= 1;
                    i += 1;
                    continue;
                }
                // A run of plain bytes goes through the budget at once.
                // A skipped run needs no storage, so one scan passes it. A
                // kept run decodes in parts no larger than the room left; the
                // scan stops at the end of the part, so no byte is scanned twice.
                let end = if state.skip { b.len() } else { b.len().min(i.saturating_add(budget.room().max(64))) };
                let run = b[i..end].iter().position(|c| matches!(c, b'\\' | b'{' | b'}' | b'\r' | b'\n')).map_or(end, |n| i + n);
                super::scanned(run - i);
                if !state.skip {
                    let text: String = b[i..run].iter().map(|&c| byte(now, c, &mut unknown)).collect();
                    lone(&mut high, budget, &mut out);
                    budget.push(&mut out, &text);
                }
                i = run;
                continue;
            }
        }
        // A control word or symbol.
        i += 1;
        let Some(&n) = b.get(i) else { break };
        if !n.is_ascii_alphabetic() {
            i += 1;
            if n == b'*' {
                starred = group_start;
                continue;
            }
            let hex = (n == b'\'').then(|| b.get(i..i + 2).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok()));
            i += if n == b'\'' { 2 } else { 0 };
            group_start = false;
            if pending > 0 {
                pending -= 1;
                continue;
            }
            if state.skip {
                continue;
            }
            let text = match n {
                b'\'' => hex.flatten().map(|h| byte(now, h, &mut unknown)),
                b'~' => Some('\u{a0}'),
                b'_' => Some('-'),
                b'\r' | b'\n' => Some('\n'),
                b'\\' | b'{' | b'}' => Some(n as char),
                _ => None,
            };
            if let Some(t) = text {
                lone(&mut high, budget, &mut out);
                budget.push(&mut out, t.encode_utf8(&mut [0; 4]));
            }
            continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_alphabetic() {
            i += 1;
        }
        let word = std::str::from_utf8(&b[start..i]).unwrap_or("");
        let num_start = i;
        if b.get(i) == Some(&b'-') {
            i += 1;
        }
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        // A number too large for i64 saturates.
        let digits = std::str::from_utf8(&b[num_start..i]).unwrap_or("");
        let param: Option<i64> = (digits.len() > usize::from(digits.starts_with('-')))
            .then(|| digits.parse().unwrap_or(if digits.starts_with('-') { i64::MIN } else { i64::MAX }));
        if b.get(i) == Some(&b' ') {
            i += 1;
        }
        if word == "bin" {
            // N raw bytes follow; they are data, never RTF syntax.
            let len = usize::try_from(param.unwrap_or(0).max(0)).unwrap_or(usize::MAX);
            i = i.saturating_add(len).min(b.len());
            group_start = false;
            // The control word and its data are one fallback character.
            pending = pending.saturating_sub(1);
            continue;
        }
        let destination = group_start;
        group_start = false;
        if pending > 0 {
            pending -= 1;
            continue;
        }
        if destination && (starred || SKIP.contains(&word)) {
            if matches!(word, "pict" | "object" | "shppict") && !state.skip {
                pictures += 1;
            }
            state.skip = true;
            state.fonttbl = word == "fonttbl";
            continue;
        }
        match word {
            "ansicpg" => page = param.and_then(|p| u32::try_from(p).ok()).unwrap_or(1252),
            "deff" => deff = param,
            "f" if state.fonttbl => entry = param,
            "f" if !state.skip => state.font = param,
            "plain" if !state.skip => state.font = None,
            "fcharset" if state.fonttbl => {
                if let (Some(font), Some(code)) = (entry, param.and_then(codepage::for_charset))
                    && (fonts.contains_key(&font) || budget.reserve(FONT))
                {
                    fonts.insert(font, code);
                }
            }
            "ansi" => page = 1252,
            "mac" => page = 10000,
            "pc" => page = 437,
            "pca" => page = 850,
            _ => {}
        }
        if matches!(word, "ansicpg" | "deff" | "f" | "plain" | "fcharset" | "ansi" | "mac" | "pc" | "pca") {
            now = page_of(&state, &fonts, page, deff);
        }
        if state.skip {
            continue;
        }
        let text: &str = match word {
            "par" | "line" | "row" | "sect" | "page" => "\n",
            "tab" => "\t",
            "cell" => " | ",
            "emdash" => "\u{2014}",
            "endash" => "\u{2013}",
            "bullet" => "\u{2022}",
            "lquote" => "\u{2018}",
            "rquote" => "\u{2019}",
            "ldblquote" => "\u{201c}",
            "rdblquote" => "\u{201d}",
            "emspace" | "enspace" | "qmspace" => " ",
            "uc" => {
                state.uc = param.unwrap_or(1).clamp(0, 16) as usize;
                ""
            }
            "u" => {
                let v = param.unwrap_or(0).clamp(-32768, 65535);
                let v = if v < 0 { v + 65536 } else { v } as u32;
                let mut chars = String::new();
                match (high.take(), v) {
                    (h, 0xd800..=0xdbff) => {
                        chars.extend(h.map(|_| '\u{fffd}'));
                        high = Some(v);
                    }
                    (Some(h), 0xdc00..=0xdfff) => chars.push(char::from_u32(0x10000 + ((h - 0xd800) << 10) + (v - 0xdc00)).unwrap_or('\u{fffd}')),
                    (h, v) => {
                        chars.extend(h.map(|_| '\u{fffd}'));
                        chars.push(char::from_u32(v).unwrap_or('\u{fffd}'));
                    }
                }
                budget.push(&mut out, &chars);
                pending = state.uc;
                ""
            }
            _ => "",
        };
        if !text.is_empty() {
            lone(&mut high, budget, &mut out);
            budget.push(&mut out, text);
        }
    }
    lone(&mut high, budget, &mut out);
    Parsed { text: tidy(&out), pictures, unknown }
}
/// A high surrogate with no low half after it shows as U+FFFD.
fn lone(high: &mut Option<u32>, budget: &mut Budget, out: &mut String) {
    if high.take().is_some() {
        budget.push(out, "\u{fffd}");
    }
}

/// Byte `c` in code page `page`; U+FFFD, counted in `unknown` with the
/// first such page, when this build has no table for the page.
fn byte(page: u32, c: u8, unknown: &mut (usize, u32)) -> char {
    codepage::decode(page, c).unwrap_or_else(|| {
        if unknown.0 == 0 {
            unknown.1 = page;
        }
        unknown.0 += 1;
        '\u{fffd}'
    })
}

/// Trims line ends and keeps at most one blank line in a row.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.lines() {
        let line = line.trim_end();
        blank = if line.is_empty() { blank + 1 } else { 0 };
        if blank > 1 || (blank == 1 && out.is_empty()) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}
