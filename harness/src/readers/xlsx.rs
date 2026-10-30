//! XLSX (SpreadsheetML) to Markdown: one sheet marker and one table for
//! each worksheet, in workbook order. Cells show their saved values: shared
//! and inline strings, numbers, booleans, errors and the cached result of a
//! formula. A number in a date or time format shows as an ISO 8601 date or
//! time (the 1900 and 1904 date systems). The table spans the used columns;
//! empty rows are left out, and the first kept row is the header.
//!
//! Skipped: formulas (their cached values show), charts, images, comments
//! and formatting. Shared strings, styles and kept cells count against the
//! work cap; rows after `Caps::max_rows` in a sheet are counted, not kept.
use super::xml::{self, Event, Scanner};
use super::zip;
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, page_cap_warning, sheet_marker};
use std::collections::HashMap;

pub(super) struct XlsxReader;

/// Storage of one shared string, style or cell, for the work cap.
const SLOT: usize = 32;
/// The largest column (XFD) and row of a worksheet.
pub(super) const MAX_COL: u32 = 16_384;
pub(super) const MAX_ROW: u32 = 1_048_576;

impl Reader for XlsxReader {
    fn sniff(&self, _: &[u8], name: &str) -> Option<Format> {
        matches!(extension(name).as_str(), "xlsx" | "xlsm" | "xltx" | "xltm").then_some(Format::Xlsx)
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let mut budget = Budget::new(caps);
        let mut zip = super::office_zip(bytes, format)?;
        let main = zip::main_part(&mut zip, &mut budget)?.unwrap_or_else(|| "xl/workbook.xml".to_owned());
        let bad = |name: &str, why: String| ReadError::Malformed(format, format!("{name} {why}"));
        let rels = zip.rels(&main, &mut budget)?;
        let (text, cut) = zip.text(&main)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {main}")))?;
        let (sheets, date1904) = workbook(&text, cut, &mut budget).map_err(|why| bad(&main, why))?;
        let part = |kind: &str| rels.first(kind).map(str::to_owned);
        let (strings, strings_cut) = match part("sharedStrings") {
            Some(name) => match zip.text(&name)? {
                Some((text, cut)) => (shared_strings(&text, cut, &mut budget).map_err(|why| bad(&name, why))?, cut || budget.full()),
                None => (Vec::new(), false),
            },
            None => (Vec::new(), false),
        };
        let dates = match part("styles") {
            Some(name) => match zip.text(&name)? {
                Some((text, cut)) => date_styles(&text, cut, &mut budget).map_err(|why| bad(&name, why))?,
                None => Vec::new(),
            },
            None => Vec::new(),
        };
        let book = Book { strings, strings_cut, dates, date1904 };
        let mut doc = Doc::new(String::new(), sheets.len());
        if sheets.is_empty() {
            // A sheet list that the work cap stopped, or whose ids the
            // attribute cap left out, is not damage.
            if budget.full() || budget.long_attr {
                doc.warnings.extend(budget.warning());
                return Ok(doc);
            }
            return Err(ReadError::Malformed(format, format!("{main} lists no sheets")));
        }
        let total = sheets.len();
        for (n, sheet) in sheets.iter().enumerate() {
            if n == caps.max_pages {
                doc.warnings.push(page_cap_warning(caps.max_pages, total, "sheets"));
                break;
            }
            if budget.full() {
                break;
            }
            let target = sheet.rel.as_deref().and_then(|id| rels.get(id)).filter(|r| !r.external);
            // The work cap stopped the relationships before this sheet's.
            if target.is_none() && rels.stopped {
                break;
            }
            let label = if sheet.hidden { format!("{} (hidden)", sheet.name) } else { sheet.name.clone() };
            let marker = sheet_marker(n + 1, total, &label);
            let marker = if n == 0 { marker.trim_start() } else { &marker };
            if !budget.fits(marker.len()) {
                budget.cut();
                break;
            }
            doc.markdown.push_str(budget.work(marker));
            let body = match target {
                Some(r) if r.kind == "worksheet" => {
                    let name = r.target.clone();
                    let (text, cut) = zip.text(&name)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {name}")))?;
                    let (md, rows) = worksheet(&text, cut, &book, caps, &mut budget).map_err(|why| bad(&name, why))?;
                    // The table moves into the output: its bytes go back first.
                    budget.release(md.len());
                    if rows > caps.max_rows.saturating_add(1) {
                        doc.warnings.push(format!("rows after {} of {} in sheet {} (row cap)", caps.max_rows, rows - 1, sheet.name));
                    }
                    md
                }
                Some(r) if r.kind == "chartsheet" => "(chart sheet; charts are not read)".to_owned(),
                _ => "(sheet part not found)".to_owned(),
            };
            // A table was held before; a note shows only whole.
            match body.as_str() {
                "" => budget.whole(&mut doc.markdown, "(empty sheet)"),
                b if b.starts_with('(') => budget.whole(&mut doc.markdown, b),
                b => doc.markdown.push_str(budget.work(b)),
            }
            doc.markdown.push('\n');
        }
        doc.warnings.append(&mut zip.warnings);
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

struct Sheet {
    name: String,
    rel: Option<String>,
    hidden: bool,
}

/// The sheets of `workbook.xml` in order, and whether dates count from 1904.
fn workbook(text: &str, cut: bool, budget: &mut Budget) -> Result<(Vec<Sheet>, bool), String> {
    let mut scan = Scanner::new(text, cut);
    let (mut sheets, mut date1904) = (Vec::new(), false);
    while let Some(event) = scan.next()? {
        let Event::Open(tag) = event else { continue };
        match tag.local() {
            "workbookPr" => date1904 = tag.flag("date1904")?.unwrap_or(false),
            "sheet" => {
                let (name, rel) = (tag.attr("name")?.unwrap_or_default(), tag.attr("r:id")?);
                if !budget.reserve(SLOT + name.len() + rel.as_ref().map_or(0, |r| r.len())) {
                    break;
                }
                let hidden = tag.attr("state")?.is_some_and(|s| s != "visible");
                sheets.push(Sheet { name: name.into_owned(), rel: rel.map(|r| r.into_owned()), hidden });
            }
            _ => {}
        }
    }
    scan.finish(budget)?;
    Ok((sheets, date1904))
}

/// The shared string table: each `si` item's text, phonetic runs left out.
fn shared_strings(text: &str, cut: bool, budget: &mut Budget) -> Result<Vec<String>, String> {
    let mut scan = Scanner::new(text, cut);
    let mut out = Vec::new();
    let (mut item, mut in_text, mut phonetic) = (None::<String>, false, 0usize);
    while let Some(event) = scan.next()? {
        match event {
            Event::Open(tag) => match tag.local() {
                "si" => item = Some(String::new()),
                "t" => in_text = true,
                "rPh" => phonetic += 1,
                _ => {}
            },
            Event::Close(name) => match xml::local(name) {
                "t" => in_text = false,
                "rPh" => phonetic = phonetic.saturating_sub(1),
                "si" => {
                    let text = item.take().unwrap_or_default();
                    if !budget.reserve(SLOT + text.len()) {
                        break;
                    }
                    out.push(text);
                }
                _ => {}
            },
            Event::Text(raw) | Event::Cdata(raw) if in_text && phonetic == 0 => {
                if let Some(item) = item.as_mut() {
                    // A string longer than the room left keeps what fits; only that is decoded.
                    let text = if matches!(event, Event::Text(_)) { xml::unescape_prefix(raw, budget.room())? } else { raw.into() };
                    let room = budget.room().saturating_sub(item.len() + SLOT);
                    let mut end = text.len().min(room);
                    while !text.is_char_boundary(end) {
                        end -= 1;
                    }
                    if end < text.len() {
                        budget.cut();
                    }
                    item.push_str(&text[..end]);
                }
            }
            _ => {}
        }
    }
    scan.finish(budget)?;
    Ok(out)
}

/// Date kind of a number format: none, a date, a time, both, or elapsed
/// time (`[h]:mm:ss`, whose hours go past 24).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum When {
    No,
    Date,
    Time,
    DateTime,
    Elapsed,
}

/// For each cell format (`cellXfs`, in order), whether its number format
/// shows a date or time.
fn date_styles(text: &str, cut: bool, budget: &mut Budget) -> Result<Vec<When>, String> {
    let mut scan = Scanner::new(text, cut);
    // Custom formats by id; a later format with the same id wins.
    let mut custom: HashMap<u32, When> = HashMap::new();
    let mut out = Vec::new();
    let mut in_xfs = false;
    while let Some(event) = scan.next()? {
        match event {
            Event::Open(tag) => match tag.local() {
                "numFmt" => {
                    let id = tag.attr("numFmtId")?.and_then(|v| v.parse().ok());
                    let code = tag.attr("formatCode")?.unwrap_or_default();
                    if let Some(id) = id {
                        if !budget.reserve(SLOT) {
                            break;
                        }
                        custom.insert(id, format_kind(&code));
                    }
                }
                "cellXfs" => in_xfs = true,
                "xf" if in_xfs => {
                    let id: u32 = tag.attr("numFmtId")?.and_then(|v| v.parse().ok()).unwrap_or(0);
                    let kind = custom.get(&id).copied().unwrap_or_else(|| builtin_kind(id));
                    if !budget.reserve(SLOT) {
                        break;
                    }
                    out.push(kind);
                }
                _ => {}
            },
            Event::Close(name) if xml::local(name) == "cellXfs" => in_xfs = false,
            _ => {}
        }
    }
    scan.finish(budget)?;
    Ok(out)
}

/// The built-in number formats that show dates or times (ECMA-376, 18.8.30).
fn builtin_kind(id: u32) -> When {
    match id {
        14..=17 | 27..=31 | 34..=36 | 50..=58 => When::Date,
        46 => When::Elapsed,
        18..=21 | 32 | 33 | 45 | 47 => When::Time,
        22 => When::DateTime,
        _ => When::No,
    }
}

/// The date kind of a custom format code. Quoted text, escaped characters
/// and bracketed parts (colors, locales, conditions) do not count; `[h]`
/// elapsed hours do.
fn format_kind(code: &str) -> When {
    // Only the first section (positive numbers) decides.
    let (mut date, mut time, mut month, mut elapsed) = (false, false, false, false);
    let mut chars = code.chars();
    while let Some(c) = chars.next() {
        match c.to_ascii_lowercase() {
            ';' => break,
            '"' => {
                for q in chars.by_ref() {
                    if q == '"' {
                        break;
                    }
                }
            }
            '\\' | '_' | '*' => {
                chars.next();
            }
            '[' => {
                let inner: String = chars.by_ref().take_while(|&b| b != ']').collect();
                elapsed |= matches!(inner.to_ascii_lowercase().as_str(), "h" | "hh" | "m" | "mm" | "s" | "ss");
            }
            'd' | 'y' => date = true,
            'h' | 's' => time = true,
            'm' => month = true,
            _ => {}
        }
    }
    let time = time || elapsed;
    if elapsed && !date {
        return When::Elapsed;
    }
    match (date || (month && !time), time) {
        (true, true) => When::DateTime,
        (true, false) => When::Date,
        (false, true) => When::Time,
        (false, false) => When::No,
    }
}

struct Book {
    strings: Vec<String>,
    /// A cap cut the shared string table, so a later index shows as empty.
    strings_cut: bool,
    dates: Vec<When>,
    date1904: bool,
}

/// One worksheet as a Markdown table, and the count of rows with values.
fn worksheet(text: &str, cut: bool, book: &Book, caps: &Caps, budget: &mut Budget) -> Result<(String, usize), String> {
    let keep = caps.max_rows.saturating_add(1);
    let mut scan = Scanner::new(text, cut);
    let mut rows: Vec<Vec<(u32, String)>> = Vec::new();
    let mut row: Vec<(u32, String)> = Vec::new();
    let mut counted = 0usize;
    let (mut row_at, mut col_at) = (0u32, 0u32);
    // The open cell: column, type, style; and whether a value or inline text is open.
    let mut open: Option<(u32, String, usize)> = None;
    let (mut value, mut in_value, mut in_inline, mut in_text, mut phonetic) = (String::new(), false, false, false, 0usize);
    // The open cell has a value that is counted but not kept (past a cap),
    // or a value whose start is kept and whose end the work cap cut.
    let (mut seen, mut cut_cell) = (false, false);
    let (mut min_col, mut max_col) = (u32::MAX, 0u32);
    // Work bytes that the kept rows and cells hold.
    let mut held = 0usize;
    while let Some(event) = scan.next()? {
        match event {
            Event::Open(tag) => match tag.local() {
                "row" => {
                    row_at = match tag.attr("r")? {
                        Some(r) => r.parse().ok().filter(|r| (1..=MAX_ROW).contains(r)).ok_or_else(|| format!("has a row number that is not valid: {r}"))?,
                        None => row_at + 1,
                    };
                    col_at = 0;
                    row.clear();
                }
                "c" => {
                    col_at = match tag.attr("r")? {
                        Some(r) => column(&r).ok_or_else(|| format!("has a cell reference that is not valid: {r}"))?,
                        None => col_at + 1,
                    };
                    let kind = tag.attr("t")?.unwrap_or_default().into_owned();
                    let style = tag.attr("s")?.and_then(|s| s.parse().ok()).unwrap_or(0);
                    open = Some((col_at, kind, style));
                    value.clear();
                    (seen, cut_cell) = (false, false);
                }
                "v" => in_value = true,
                "is" => in_inline = true,
                "t" => in_text = true,
                "rPh" => phonetic += 1,
                _ => {}
            },
            Event::Close(name) => match xml::local(name) {
                "v" => in_value = false,
                "is" => in_inline = false,
                "t" => in_text = false,
                "rPh" => phonetic = phonetic.saturating_sub(1),
                "c" => {
                    let Some((col, kind, style)) = open.take() else { continue };
                    if value.is_empty() && !seen {
                        continue;
                    }
                    // Rows past the caps are counted, not kept.
                    if seen || counted >= keep || (budget.full() && !cut_cell) {
                        if row.is_empty() {
                            row.push((col, String::new()));
                        }
                        continue;
                    }
                    // The cell's slot is held first, then its text keeps the
                    // start that fits; a cut cell stops the reader after it.
                    if !budget.reserve(SLOT) {
                        continue;
                    }
                    let shown = show(&value, &kind, style, book)?;
                    let kept = budget.text(&shown).to_owned();
                    if cut_cell {
                        budget.cut();
                    }
                    held += SLOT + kept.len();
                    (min_col, max_col) = (min_col.min(col), max_col.max(col));
                    row.push((col, cell(&kept)));
                }
                "row" => {
                    if !row.is_empty() {
                        counted += 1;
                        // A row keeps the cells it read, also when a cap stopped it.
                        if counted <= keep && row.iter().any(|(_, v)| !v.is_empty()) {
                            if budget.reserve(SLOT) {
                                held += SLOT;
                            }
                            rows.push(std::mem::take(&mut row));
                        }
                    }
                    row.clear();
                }
                _ => {}
            },
            Event::Text(raw) | Event::Cdata(raw) if open.is_some() && (in_value || (in_inline && in_text)) && phonetic == 0 => {
                // Past the row cap or a stop, a value is counted, not read.
                if counted >= keep || budget.full() {
                    seen |= !raw.is_empty();
                    continue;
                }
                // A value larger than the room keeps the start that fits, and
                // only that start is decoded.
                let room = budget.room().saturating_sub(value.len());
                let text = if matches!(event, Event::Text(_)) { xml::unescape_prefix(raw, room)? } else { raw.into() };
                // A cut number or string index would show a wrong value: it
                // is counted, not kept. An unzip cut that ends inside it has
                // its own warning.
                let whole = keep_prefix(&mut value, &text, room);
                if in_value && (!whole || scan.text_cut()) {
                    value.clear();
                    seen = true;
                    if !whole {
                        budget.cut();
                    }
                } else if !whole {
                    cut_cell = true;
                }
            }
            _ => {}
        }
    }
    scan.finish(budget)?;
    Ok((sheet_table(rows, held, min_col, max_col, budget), counted))
}

/// Sparse rows of (1-based column, cell) as a Markdown table over the columns
/// `min_col..=max_col`; rows with no values are left out. The rows give back
/// the `held` work bytes first, as they move into the table. Each line is
/// written from its sparse row, so a wide sheet needs no dense copy; the
/// table stops at the cell where a cap is reached, and its bytes stay
/// counted when it returns.
pub(super) fn sheet_table(mut rows: Vec<Vec<(u32, String)>>, held: usize, min_col: u32, max_col: u32, budget: &mut Budget) -> String {
    budget.release(held);
    rows.retain(|r| r.iter().any(|(_, v)| !v.is_empty()));
    if rows.is_empty() || min_col > max_col {
        return String::new();
    }
    for r in &mut rows {
        r.sort_by_key(|(c, _)| *c);
    }
    let mut out = String::new();
    // The cells were counted as text once; the table counts as work.
    let counts = std::mem::replace(&mut budget.text_counts, false);
    let _ = sheet_line(&mut out, budget, min_col, max_col, Some(&rows[0]))
        && sheet_line(&mut out, budget, min_col, max_col, None)
        && rows[1..].iter().all(|r| sheet_line(&mut out, budget, min_col, max_col, Some(r)));
    budget.text_counts = counts;
    out.truncate(out.trim_end().len());
    out
}

/// One table line over `min_col..=max_col` from a sorted sparse row; `None`
/// is the separator line. False when the budget cut the line.
fn sheet_line(out: &mut String, budget: &mut Budget, min_col: u32, max_col: u32, row: Option<&[(u32, String)]>) -> bool {
    if !budget.push(out, "|") {
        return false;
    }
    let mut cells = row.unwrap_or(&[]).iter().peekable();
    for c in min_col..=max_col {
        let mut value = "";
        while let Some((col, v)) = cells.peek() {
            if *col > c {
                break;
            }
            if *col == c && value.is_empty() {
                value = v;
            }
            cells.next();
        }
        let value = if row.is_none() { "---" } else { value };
        if !(budget.push(out, " ") && budget.push(out, value) && budget.push(out, " |")) {
            return false;
        }
    }
    budget.push(out, "\n")
}

/// Appends the start of `text` that fits in `room` bytes to `value`; false
/// when it had to leave text out.
pub(super) fn keep_prefix(value: &mut String, text: &str, room: usize) -> bool {
    let mut end = text.len().min(room);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    value.push_str(&text[..end]);
    end == text.len()
}

/// The 1-based column of a cell reference such as `AB12`.
fn column(r: &str) -> Option<u32> {
    let letters = r.bytes().take_while(u8::is_ascii_alphabetic).count();
    let digits = &r[letters..];
    if letters == 0 || letters > 3 || digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let col = r[..letters].bytes().fold(0u32, |n, b| n * 26 + u32::from(b.to_ascii_uppercase() - b'A' + 1));
    (col <= MAX_COL).then_some(col)
}

/// The text a cell shows for its saved `value`.
fn show(value: &str, kind: &str, style: usize, book: &Book) -> Result<String, String> {
    Ok(match kind {
        "s" => {
            let i: usize = value.trim().parse().map_err(|_| format!("has a shared string index that is not valid: {value}"))?;
            match book.strings.get(i) {
                Some(s) => s.clone(),
                None if book.strings_cut => String::new(),
                None => return Err(format!("refers to shared string {i}; the table has {}", book.strings.len())),
            }
        }
        "b" => match value.trim() {
            "1" => "TRUE".to_owned(),
            "0" => "FALSE".to_owned(),
            v => v.to_owned(),
        },
        "str" | "inlineStr" | "e" | "d" => value.to_owned(),
        _ => number(value.trim(), book.dates.get(style).copied().unwrap_or(When::No), book.date1904),
    })
}

/// A number as written, in its shortest form, or as a date or time.
fn number(value: &str, when: When, date1904: bool) -> String {
    let Ok(n) = value.parse::<f64>() else { return value.to_owned() };
    if !n.is_finite() {
        return value.to_owned();
    }
    if when != When::No && (0.0..2_958_466.0).contains(&n) {
        return date_time(n, when, date1904);
    }
    format!("{n}")
}

/// An Excel date serial as ISO 8601. In the 1900 system, serial 60 is the
/// day Excel counts as 1900-02-29.
fn date_time(serial: f64, when: When, date1904: bool) -> String {
    let mut days = serial.floor() as i64;
    let mut secs = ((serial - serial.floor()) * 86_400.0).round() as i64;
    if secs == 86_400 {
        days += 1;
        secs = 0;
    }
    let time = format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
    if when == When::Time {
        return time;
    }
    if when == When::Elapsed {
        let all = days * 86_400 + secs;
        return format!("{:02}:{:02}:{:02}", all / 3600, all / 60 % 60, all % 60);
    }
    let date = if date1904 {
        civil(days - 24_107)
    } else if days == 60 {
        "1900-02-29".to_owned()
    } else {
        civil(days - if days < 60 { 25_568 } else { 25_569 })
    };
    if when == When::DateTime || secs != 0 { format!("{date} {time}") } else { date }
}

/// The date `days` after 1970-01-01 (H. Hinnant's civil-from-days).
fn civil(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
pub(super) fn test_date(serial: f64, date1904: bool) -> String {
    date_time(serial, When::Date, date1904)
}

/// The sheet part `text` (cut by a cap when `cut`) with the shared
/// `strings`, as a Markdown table.
#[cfg(test)]
pub(super) fn test_sheet(text: &str, cut: bool, strings: &[&str]) -> String {
    let book = Book { strings: strings.iter().map(|s| s.to_string()).collect(), strings_cut: false, dates: Vec::new(), date1904: false };
    worksheet(text, cut, &book, &Caps::default(), &mut Budget::new(&Caps::default())).unwrap().0
}

/// The number `serial` shown with the custom format `code`.
#[cfg(test)]
pub(super) fn test_show(serial: &str, code: &str) -> String {
    number(serial, format_kind(code), false)
}

#[cfg(test)]
pub(super) fn test_format(code: &str) -> &'static str {
    match format_kind(code) {
        When::No => "no",
        When::Date => "date",
        When::Time => "time",
        When::DateTime => "datetime",
        When::Elapsed => "elapsed",
    }
}
