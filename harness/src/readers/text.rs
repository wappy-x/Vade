//! Plain text, Markdown, data files and source code. Encoding detection:
//! a UTF-8 or UTF-16 byte order mark, then UTF-16 without a mark (from the
//! zero-byte pattern, only when the sample holds zero bytes), then UTF-8, then
//! Windows-1252 (a Latin-1 superset).
use super::{Budget, Caps, Doc, Format, ReadError, Reader, extension, fence};

pub(super) struct TextReader;

/// JSON at or below this size is pretty-printed when it is minified.
const PRETTY_JSON_BYTES: usize = 256 * 1024;

impl Reader for TextReader {
    fn sniff(&self, _: &[u8], name: &str) -> Option<Format> {
        // Text has no magic bytes; unnamed text comes from the registry fallback.
        if name.is_empty() { None } else { by_name(name) }
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let (text, note) = decode(bytes).ok_or(ReadError::Binary)?;
        let mut budget = Budget::new(caps);
        let pretty = if format == Format::Json { pretty_json(&text, &mut budget) } else { None };
        let text = match &pretty {
            Some(pretty) => pretty.as_str(),
            None => budget.text(&text),
        };
        let body = match format {
            Format::Json => fence("json", text),
            Format::Yaml => fence("yaml", text),
            Format::Toml => fence("toml", text),
            Format::Xml => fence("xml", text),
            Format::Source(lang) => fence(lang, text),
            _ => text.to_owned(),
        };
        let mut doc = Doc::new(body, 1);
        doc.warnings.extend(note.map(str::to_owned));
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

/// The format named by the file name alone.
pub(super) fn by_name(name: &str) -> Option<Format> {
    let leaf = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let named = match leaf {
        "Makefile" | "makefile" | "GNUmakefile" => Some("make"),
        "Dockerfile" | "Containerfile" => Some("dockerfile"),
        "CMakeLists.txt" => Some("cmake"),
        "Cargo.lock" => return Some(Format::Toml),
        "Gemfile" | "Rakefile" => Some("ruby"),
        _ => None,
    };
    if let Some(lang) = named {
        return Some(Format::Source(lang));
    }
    let format = match extension(name).as_str() {
        "txt" | "text" | "log" | "out" | "err" | "nfo" | "me" | "1st" | "readme" => Format::Text,
        "md" | "markdown" | "mdown" | "mkd" | "mdx" => Format::Markdown,
        "json" | "jsonl" | "ndjson" | "geojson" | "jsonc" | "json5" | "ipynb" => Format::Json,
        "yaml" | "yml" => Format::Yaml,
        "toml" => Format::Toml,
        "xml" | "xsd" | "xsl" | "xslt" | "svg" | "plist" | "rss" | "atom" | "opf" | "ncx" | "csproj" | "fsproj" => {
            Format::Xml
        }
        ext => Format::Source(language(ext)?),
    };
    Some(format)
}

/// The fence language for a source extension.
fn language(ext: &str) -> Option<&'static str> {
    Some(match ext {
        "rs" => "rust",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" | "ino" => "cpp",
        "cu" | "cuh" => "cuda",
        "ptx" => "ptx",
        "m" | "mm" => "objectivec",
        "swift" => "swift",
        "go" => "go",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "scala" | "sc" => "scala",
        "groovy" | "gradle" => "groovy",
        "cs" => "csharp",
        "fs" | "fsx" => "fsharp",
        "vb" => "vbnet",
        "py" | "pyi" | "pyw" => "python",
        "rb" => "ruby",
        "pl" | "pm" => "perl",
        "php" => "php",
        "lua" => "lua",
        "r" => "r",
        "jl" => "julia",
        "js" | "mjs" | "cjs" => "javascript",
        "jsx" => "jsx",
        "ts" | "mts" | "cts" => "typescript",
        "tsx" => "tsx",
        "vue" => "vue",
        "svelte" => "svelte",
        "css" => "css",
        "scss" => "scss",
        "sass" => "sass",
        "less" => "less",
        "sh" | "bash" | "zsh" | "ksh" => "sh",
        "fish" => "fish",
        "ps1" | "psm1" => "powershell",
        "bat" | "cmd" => "bat",
        "sql" => "sql",
        "graphql" | "gql" => "graphql",
        "proto" => "protobuf",
        "hs" => "haskell",
        "ml" | "mli" => "ocaml",
        "ex" | "exs" => "elixir",
        "erl" | "hrl" => "erlang",
        "clj" | "cljs" | "edn" => "clojure",
        "lisp" | "el" | "scm" | "rkt" => "lisp",
        "dart" => "dart",
        "zig" => "zig",
        "nim" => "nim",
        "v" | "sv" | "svh" => "verilog",
        "vhd" | "vhdl" => "vhdl",
        "asm" | "s" => "asm",
        "f" | "f90" | "f95" | "for" => "fortran",
        "tex" | "sty" | "cls" | "bib" => "latex",
        "ini" | "cfg" | "conf" | "properties" => "ini",
        "env" => "sh",
        "diff" | "patch" => "diff",
        "cmake" => "cmake",
        "mk" => "make",
        "nix" => "nix",
        "tf" | "hcl" => "hcl",
        "dockerfile" => "dockerfile",
        "rst" => "rst",
        "adoc" | "asciidoc" => "asciidoc",
        "org" => "org",
        "wgsl" => "wgsl",
        "glsl" | "vert" | "frag" | "comp" => "glsl",
        "hlsl" => "hlsl",
        "metal" => "metal",
        "sol" => "solidity",
        _ => return None,
    })
}

/// True when the first 64 KiB of `bytes` decode to text (see [`decode`]).
/// The reader decodes and checks the whole file again.
pub(super) fn looks_like_text(bytes: &[u8]) -> bool {
    decode_part(&bytes[..bytes.len().min(64 * 1024)], false).is_some()
}

/// Decodes a whole file and names the encoding when it is not UTF-8. `None`
/// for binary data: NUL bytes or many control characters after decoding. A
/// cut sequence at the end is damage, shown as U+FFFD.
pub(super) fn decode(bytes: &[u8]) -> Option<(String, Option<&'static str>)> {
    decode_part(bytes, true)
}

/// [`decode`] of a whole file (`complete`) or of a sample, which can end in
/// the middle of a sequence.
fn decode_part(bytes: &[u8], complete: bool) -> Option<(String, Option<&'static str>)> {
    let (text, note) = if let Some(rest) = bytes.strip_prefix(b"\xef\xbb\xbf") {
        match std::str::from_utf8(rest) {
            Ok(text) => (text.to_owned(), None),
            // A cut sequence at the very end of a sample is not bad bytes.
            Err(e) if !complete && e.error_len().is_none() => (String::from_utf8_lossy(&rest[..e.valid_up_to()]).into_owned(), None),
            Err(_) => (
                String::from_utf8_lossy(rest).into_owned(),
                Some("the file has a UTF-8 byte order mark, but some bytes are not UTF-8 (shown as U+FFFD)"),
            ),
        }
    } else if let Some(little) = utf16_order(bytes) {
        let marked = bytes.starts_with(b"\xff\xfe") || bytes.starts_with(b"\xfe\xff");
        let note = match (marked, little) {
            (true, true) => "decoded from UTF-16 LE",
            (true, false) => "decoded from UTF-16 BE",
            (false, true) => "decoded from UTF-16 LE (no byte order mark)",
            (false, false) => "decoded from UTF-16 BE (no byte order mark)",
        };
        let mut text = utf16(&bytes[if marked { 2 } else { 0 }..], little);
        // An odd last byte of a whole file is half a code unit.
        if complete && bytes.len() % 2 == 1 {
            text.push('\u{fffd}');
        }
        (text, Some(note))
    } else if let Ok(text) = std::str::from_utf8(bytes) {
        (text.to_owned(), None)
    } else {
        (bytes.iter().map(|&b| windows_1252(b)).collect(), Some("decoded as Windows-1252 (Latin-1); the file is not UTF-8"))
    };
    is_text(&text).then_some((text, note))
}

/// A cut multibyte sequence at the very end (a sample) still counts as text.
/// Every character is checked, so a NUL byte late in a file makes it binary.
fn is_text(text: &str) -> bool {
    let mut control = 0usize;
    let mut total = 0usize;
    for c in text.chars() {
        total += 1;
        match c {
            '\0' => return false,
            '\t' | '\n' | '\r' | '\x0c' | '\x1b' => {}
            c if c.is_control() && (c as u32) < 0x20 => control += 1,
            _ => {}
        }
    }
    control * 20 <= total
}

/// UTF-16 byte order from the first 4 KiB, `true` for little endian: a byte
/// order mark, or the zero-byte pattern when the sample holds zero bytes.
/// The UTF-8 byte order mark wins over this, so test it first.
pub(super) fn utf16_order(bytes: &[u8]) -> Option<bool> {
    if bytes.starts_with(b"\xff\xfe") {
        Some(true)
    } else if bytes.starts_with(b"\xfe\xff") {
        Some(false)
    } else {
        bytes[..bytes.len().min(4096)].contains(&0).then(|| utf16_guess(bytes)).flatten()
    }
}

/// UTF-16 without a mark: most ASCII text has a zero in every other byte.
fn utf16_guess(bytes: &[u8]) -> Option<bool> {
    let sample = &bytes[..bytes.len().min(4096) & !1];
    if sample.len() < 4 {
        return None;
    }
    let pairs = sample.len() / 2;
    let even = sample.iter().step_by(2).filter(|&&b| b == 0).count();
    let odd = sample.iter().skip(1).step_by(2).filter(|&&b| b == 0).count();
    if odd * 10 >= pairs * 4 && even * 20 <= pairs {
        Some(true)
    } else if even * 10 >= pairs * 4 && odd * 20 <= pairs {
        Some(false)
    } else {
        None
    }
}

/// Lossy: an unpaired surrogate (for example at the end of a sample) is
/// U+FFFD. An odd last byte is left out.
fn utf16(bytes: &[u8], little: bool) -> String {
    let units = bytes.chunks_exact(2).map(|p| if little { u16::from_le_bytes([p[0], p[1]]) } else { u16::from_be_bytes([p[0], p[1]]) });
    char::decode_utf16(units).map(|c| c.unwrap_or('\u{fffd}')).collect()
}

/// Windows-1252; the five undefined bytes map to the C1 controls.
pub(super) fn windows_1252(b: u8) -> char {
    const HIGH: [u16; 32] = [
        0x20ac, 0x81, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0x8d, 0x017d,
        0x8f, 0x90, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0x9d,
        0x017e, 0x0178,
    ];
    match b {
        0x80..=0x9f => char::from_u32(HIGH[(b - 0x80) as usize] as u32).unwrap_or('\u{fffd}'),
        _ => b as char,
    }
}

/// Deepest JSON nesting that is pretty-printed; deeper JSON is kept as written.
const MAX_JSON_DEPTH: usize = 64;

/// Minified JSON, indented by two spaces through `budget`; `None` when the
/// text is large, already on more than two lines, or not one JSON value. The
/// check builds no values, and the output grows only within the budget.
fn pretty_json(text: &str, budget: &mut Budget) -> Option<String> {
    let body = text.trim();
    if body.len() > PRETTY_JSON_BYTES || body.lines().count() > 2 || !valid_json(body.as_bytes()) {
        return None;
    }
    const INDENT: &str = match std::str::from_utf8(&[b' '; 2 * MAX_JSON_DEPTH]) {
        Ok(spaces) => spaces,
        Err(_) => "",
    };
    // Pieces go through the budget a few KiB at a time.
    let (mut out, mut piece) = (String::new(), String::new());
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    let mut chars = body.chars().peekable();
    let newline = |piece: &mut String, depth: usize| {
        piece.push('\n');
        piece.push_str(&INDENT[..2 * depth.min(MAX_JSON_DEPTH)]);
    };
    while let Some(c) = chars.next() {
        if piece.len() >= 4096 {
            if !budget.push(&mut out, &piece) {
                return Some(out);
            }
            piece.clear();
        }
        if in_string {
            piece.push(c);
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                piece.push(c);
            }
            '{' | '[' => {
                piece.push(c);
                while chars.peek().is_some_and(|c| c.is_whitespace()) {
                    chars.next();
                }
                if matches!(chars.peek(), Some('}' | ']')) {
                    piece.push(chars.next().unwrap());
                } else {
                    depth += 1;
                    newline(&mut piece, depth);
                }
            }
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                newline(&mut piece, depth);
                piece.push(c);
            }
            ',' => {
                piece.push(c);
                newline(&mut piece, depth);
            }
            ':' => piece.push_str(": "),
            c if c.is_whitespace() => {}
            c => piece.push(c),
        }
    }
    piece.push('\n');
    budget.push(&mut out, &piece);
    Some(out)
}

/// True when `s` is one JSON value (RFC 8259) at most [`MAX_JSON_DEPTH`]
/// deep. One pass; the only storage is a fixed stack of open brackets.
fn valid_json(s: &[u8]) -> bool {
    let mut open = [0u8; MAX_JSON_DEPTH];
    let (mut depth, mut i) = (0usize, 0usize);
    loop {
        space(s, &mut i);
        match s.get(i) {
            Some(&c @ (b'{' | b'[')) => {
                if depth == MAX_JSON_DEPTH {
                    return false;
                }
                open[depth] = c;
                depth += 1;
                i += 1;
                space(s, &mut i);
                // `}` and `]` are two bytes after `{` and `[` in ASCII.
                if s.get(i) == Some(&(c + 2)) {
                    depth -= 1;
                    i += 1;
                } else if c == b'{' {
                    if !key(s, &mut i) {
                        return false;
                    }
                    continue;
                } else {
                    continue;
                }
            }
            Some(b'"') if string(s, &mut i) => {}
            Some(b't') if word(s, &mut i, b"true") => {}
            Some(b'f') if word(s, &mut i, b"false") => {}
            Some(b'n') if word(s, &mut i, b"null") => {}
            Some(b'-' | b'0'..=b'9') if number(s, &mut i) => {}
            _ => return false,
        }
        // After a value: close brackets up to a comma or the end.
        loop {
            space(s, &mut i);
            if depth == 0 {
                return i == s.len();
            }
            let c = open[depth - 1];
            match s.get(i) {
                Some(b',') => {
                    i += 1;
                    if c == b'{' && !key(s, &mut i) {
                        return false;
                    }
                    break;
                }
                Some(&e) if e == c + 2 => {
                    depth -= 1;
                    i += 1;
                }
                _ => return false,
            }
        }
    }
}

fn space(s: &[u8], i: &mut usize) {
    while s.get(*i).is_some_and(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r')) {
        *i += 1;
    }
}

/// An object key and its colon.
fn key(s: &[u8], i: &mut usize) -> bool {
    space(s, i);
    if !string(s, i) {
        return false;
    }
    space(s, i);
    let colon = s.get(*i) == Some(&b':');
    *i += usize::from(colon);
    colon
}

fn string(s: &[u8], i: &mut usize) -> bool {
    if s.get(*i) != Some(&b'"') {
        return false;
    }
    *i += 1;
    loop {
        match s.get(*i) {
            Some(b'"') => {
                *i += 1;
                return true;
            }
            Some(b'\\') => match s.get(*i + 1) {
                Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => *i += 2,
                Some(b'u') if s.get(*i + 2..*i + 6).is_some_and(|h| h.iter().all(u8::is_ascii_hexdigit)) => *i += 6,
                _ => return false,
            },
            Some(&c) if c >= 0x20 => *i += 1,
            _ => return false,
        }
    }
}

fn word(s: &[u8], i: &mut usize, w: &[u8]) -> bool {
    let ok = s[*i..].starts_with(w);
    *i += if ok { w.len() } else { 0 };
    ok
}

fn number(s: &[u8], i: &mut usize) -> bool {
    let digits = |i: &mut usize| {
        let from = *i;
        while s.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i > from
    };
    *i += usize::from(s.get(*i) == Some(&b'-'));
    if s.get(*i) == Some(&b'0') {
        *i += 1;
    } else if !digits(i) {
        return false;
    }
    if s.get(*i) == Some(&b'.') {
        *i += 1;
        if !digits(i) {
            return false;
        }
    }
    if matches!(s.get(*i), Some(b'e' | b'E')) {
        *i += 1;
        *i += usize::from(matches!(s.get(*i), Some(b'+' | b'-')));
        if !digits(i) {
            return false;
        }
    }
    true
}
