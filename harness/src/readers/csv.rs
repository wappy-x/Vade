//! CSV and TSV to one GitHub Markdown table. Quoted fields follow RFC 4180
//! (doubled quotes, delimiters and line breaks inside quotes). The
//! delimiter is sniffed from the first records: comma, semicolon, tab or pipe.
//! A quoted field with no closing quote is a damaged file. Fields are slices
//! of the decoded text; a cell copies at most the room left in the budget,
//! after its leading white space. A cell that loses text that way stops the
//! reader with the memory cap warning.
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, table_line, text::decode};

pub(super) struct CsvReader;

impl Reader for CsvReader {
    fn sniff(&self, _: &[u8], name: &str) -> Option<Format> {
        match extension(name).as_str() {
            "csv" => Some(Format::Csv),
            "tsv" | "tab" => Some(Format::Tsv),
            _ => None,
        }
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let (text, note) = decode(bytes).ok_or(ReadError::Binary)?;
        let delimiter = if format == Format::Tsv { '\t' } else { sniff_delimiter(&text) };
        // The first pass checks the quotes and finds the record count and the
        // widest kept row. The second writes the table within the budget,
        // one cell at a time, so padding never builds past the caps.
        let keep = caps.max_rows.saturating_add(1);
        let (mut total, mut width, mut fields) = (0usize, 0usize, 0usize);
        let mut lexer = Fields::new(&text, delimiter);
        while let Some((_, last)) = lexer.next() {
            fields += 1;
            if last {
                total += 1;
                if total <= keep {
                    width = width.max(fields);
                }
                fields = 0;
            }
        }
        if let Some(n) = lexer.unterminated {
            return Err(ReadError::Malformed(format, format!("the quoted field in record {n} has no closing quote")));
        }
        let mut budget = Budget::new(caps);
        let mut out = String::new();
        let mut lexer = Fields::new(&text, delimiter);
        'rows: for row in 0..keep {
            let Some(mut next) = lexer.next() else { break };
            if !budget.push(&mut out, "|") {
                break;
            }
            let mut n = 0;
            loop {
                let (field, last) = next;
                let (text, lost) = field.text(budget.room());
                let value = cell(&text);
                n += 1;
                if !(budget.push(&mut out, " ") && budget.push(&mut out, &value) && budget.push(&mut out, " |")) {
                    break 'rows;
                }
                if lost {
                    // The cell kept what fits; the memory cap warning says the rest is out.
                    budget.cut();
                    break 'rows;
                }
                if last {
                    break;
                }
                let Some(field) = lexer.next() else { break };
                next = field;
            }
            for _ in n..width {
                if !budget.push(&mut out, "  |") {
                    break 'rows;
                }
            }
            if !(budget.push(&mut out, "\n") && (row > 0 || table_line(&mut out, &mut budget, width, None))) {
                break;
            }
        }
        let mut doc = Doc::new(out, 1);
        doc.warnings.extend(note.map(str::to_owned));
        if total > keep {
            doc.warnings.push(format!("rows after {} of {} (row cap)", caps.max_rows, total - 1));
        }
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

/// The candidate that splits the first records into the most equal, widest rows.
fn sniff_delimiter(text: &str) -> char {
    let mut best = (',', 0usize);
    for d in [',', ';', '\t', '|'] {
        let mut counts = Vec::new();
        let (mut lexer, mut n) = (Fields::new(text, d), 0usize);
        while counts.len() < 20 {
            let Some((_, last)) = lexer.next() else { break };
            n += 1;
            if last {
                counts.push(n);
                n = 0;
            }
        }
        let Some(&first) = counts.first() else { continue };
        if first < 2 {
            continue;
        }
        let same = counts.iter().filter(|&&c| c == first).count();
        let score = same * 1000 + first;
        if score > best.1 {
            best = (d, score);
        }
    }
    best.0
}

/// One field: the part in quotes, with `""` still doubled, then the text
/// after the closing quote (or the whole field when it has no quotes).
struct Field<'a> {
    quoted: &'a str,
    rest: &'a str,
}

impl Field<'_> {
    /// The field's text with no leading white space (a cell trims it), cut
    /// near `limit` bytes, and true when the cut left out more than white space.
    fn text(&self, limit: usize) -> (String, bool) {
        let mut out = String::new();
        let mut lost = false;
        let mut take = |out: &mut String, part: &str| {
            let part = if out.is_empty() { part.trim_start() } else { part };
            let mut end = part.len().min(limit.saturating_sub(out.len()));
            while !part.is_char_boundary(end) {
                end -= 1;
            }
            out.push_str(&part[..end]);
            lost |= part[end..].chars().any(|c| !c.is_whitespace());
        };
        for (n, part) in self.quoted.split("\"\"").enumerate() {
            if n > 0 {
                take(&mut out, "\"");
            }
            take(&mut out, part);
        }
        take(&mut out, self.rest);
        (out, lost)
    }
}

/// The fields of the text in order, each with true when it ends its record.
struct Fields<'a> {
    rest: &'a str,
    delimiter: char,
    in_record: bool,
    /// Records started so far, the header included.
    records: usize,
    /// The record whose quoted field reached the end of the text.
    unterminated: Option<usize>,
}

impl<'a> Fields<'a> {
    fn new(text: &'a str, delimiter: char) -> Self {
        let rest = text.strip_prefix('\u{feff}').unwrap_or(text);
        Self { rest, delimiter, in_record: false, records: 0, unterminated: None }
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = (Field<'a>, bool);

    fn next(&mut self) -> Option<(Field<'a>, bool)> {
        if !self.in_record {
            // Blank lines between records carry no data.
            self.rest = self.rest.trim_start_matches(['\r', '\n']);
            if self.rest.is_empty() {
                return None;
            }
            self.in_record = true;
            self.records += 1;
        }
        let s = self.rest;
        // Spaces before the field are dropped; a quote after them opens it.
        let mut at = s.len() - s.trim_start_matches(' ').len();
        let mut quoted = "";
        if s[at..].starts_with('"') {
            let mut from = at + 1;
            loop {
                match s[from..].find('"') {
                    Some(n) if s[from + n + 1..].starts_with('"') => from += n + 2,
                    Some(n) => {
                        quoted = &s[at + 1..from + n];
                        at = from + n + 1;
                        break;
                    }
                    None => {
                        quoted = &s[at + 1..];
                        at = s.len();
                        self.unterminated = Some(self.records);
                        break;
                    }
                }
            }
        }
        let end = s[at..].find(|c| c == self.delimiter || c == '\r' || c == '\n').map_or(s.len(), |n| at + n);
        let field = Field { quoted, rest: &s[at..end] };
        let after = &s[end..];
        let last = match after.chars().next() {
            Some(c) if c == self.delimiter => {
                self.rest = &after[c.len_utf8()..];
                false
            }
            Some('\r') => {
                self.rest = after[1..].strip_prefix('\n').unwrap_or(&after[1..]);
                true
            }
            Some(_) => {
                self.rest = &after[1..];
                true
            }
            None => {
                self.rest = after;
                true
            }
        };
        self.in_record = !last;
        Some((field, last))
    }
}
