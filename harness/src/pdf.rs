//! Plain-text reader for text-based PDF files, standard library only.
//!
//! It reads the cross-reference (tables, streams, incremental updates, or a
//! scan of a damaged file), object streams, the page tree and each page's
//! content streams, and decodes text-showing operators through the fonts'
//! ToUnicode CMaps, simple encodings (Standard, WinAnsi, MacRoman with
//! /Differences) and Identity-H codes. Text comes out page by page in
//! content-stream order.
//!
//! It does not read: encrypted files (refused), scanned images (no OCR; a
//! file with no text says so), image codecs (JBIG2, JPX, DCT, CCITT; never
//! needed for text), Type3 glyph shapes (their text comes only through
//! ToUnicode or glyph names), vertical writing order, predefined CJK CMaps
//! without ToUnicode, annotations and form fields. Right-to-left scripts come
//! out in drawing order, often as Arabic presentation forms.
//!
//! Every read has one work budget (`budget::Budget`), and every parser and
//! decoder charges it before each loop iteration and allocation. A crafted
//! file stops at a limit, and the reply names the limit and the page.
pub(crate) mod budget;
mod content;
mod document;
mod filters;
mod font;
pub(crate) mod inflate;
mod syntax;
#[cfg(test)]
mod fuzz;
#[cfg(test)]
pub(crate) mod tests;

use budget::{Budget, Exhausted};
use document::Document;
use syntax::Object;
use std::sync::atomic::AtomicBool;

/// Largest PDF file the reader opens.
pub const MAX_PDF_BYTES: u64 = 50 << 20;
/// Pages read from one file; the reply notes any pages after this.
pub const MAX_PAGES: usize = 2_000;
/// Largest extracted text; the reply notes a cut.
pub const MAX_TEXT_BYTES: usize = 4 << 20;

/// Work bounds for one read. The first group sets the whole-read budget: a
/// read that reaches one stops, and the reply names it. The second group
/// bounds one page or one kind of call: content past it is skipped, and
/// the reply names the pages it affected.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    /// File and stream bytes lexed or scanned; a re-read counts again.
    pub(crate) read_bytes: usize,
    /// Stream bytes decoded or copied. Each use of a stream counts, so
    /// repeated streams and compression bombs stop here.
    pub(crate) stream_bytes: usize,
    /// Objects the parser builds.
    pub(crate) objects: usize,
    /// Bytes kept in caches: parsed objects, object streams, fonts.
    pub(crate) retained_bytes: usize,
    /// Tables built: Huffman tables, cross-reference sections, CMaps.
    pub(crate) tables: usize,
    /// Font entries: widths, encoding differences, CMap mappings.
    pub(crate) font_entries: usize,
    /// Loop iterations of all other kinds.
    pub(crate) steps: usize,
    /// Nesting of objects, references, forms and CMaps.
    pub(crate) depth: usize,
    /// One page's content streams, joined, and the entries read from its
    /// /Contents array.
    pub(crate) page_bytes: usize,
    pub(crate) page_streams: usize,
    /// Form XObject calls on one page, and the content bytes they run.
    pub(crate) form_calls: usize,
    pub(crate) form_bytes: usize,
    /// Form XObject calls in the whole read.
    pub(crate) document_form_calls: usize,
    pub(crate) text_bytes: usize,
    /// Pages read, at most [`MAX_PAGES`].
    pub(crate) pages: usize,
}

impl Limits {
    pub(crate) const DEFAULT: Self = Self {
        read_bytes: 1 << 30,
        stream_bytes: 512 << 20,
        objects: 1 << 26,
        retained_bytes: 256 << 20,
        tables: 1 << 20,
        font_entries: 1 << 22,
        steps: 1 << 28,
        depth: 64,
        page_bytes: document::STREAM_CAP,
        page_streams: 4_096,
        form_calls: 2_000,
        form_bytes: 32 << 20,
        document_form_calls: 100_000,
        text_bytes: MAX_TEXT_BYTES,
        pages: MAX_PAGES,
    };
}

const CANCELLED: &str = "PDF read cancelled";

/// True when `head` (the first 1,024 bytes of a file) starts a PDF file:
/// "%PDF-" at byte 0, or later in `head` when `head` is not UTF-8 text. A
/// text file that only mentions the header, such as source code, stays text.
pub fn is_pdf(head: &[u8]) -> bool {
    match document::header(head) {
        Some(0) => true,
        Some(_) => std::str::from_utf8(head).is_err_and(|e| e.error_len().is_some()),
        None => false,
    }
}

/// The text of a PDF file: a summary line, then for each page a
/// "page N of M" line and its text, one blank line between pages, then notes.
/// Errors name the reason a file cannot be read (for example, encryption).
/// A set `cancel` flag stops the read between operators.
pub fn text(data: &[u8], cancel: &AtomicBool) -> Result<String, String> {
    read(data, &Limits::DEFAULT, cancel)
}

pub(crate) fn read(data: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<String, String> {
    let budget = Budget::new(limits, cancel);
    read_with(data, limits, &budget)
}

/// `read` with a budget the caller owns, so tests can inspect it after.
pub(crate) fn read_with(data: &[u8], limits: &Limits, budget: &Budget) -> Result<String, String> {
    let x = extract(data, limits, budget)?;
    if let Some(line) = x.no_text {
        return Ok(line);
    }
    let mut out = format!("PDF text layer extracted by Vade: {} pages. Layout is approximate; text follows content-stream order.\n", x.total);
    out.push_str(&x.body);
    out.push('\n');
    out.push_str(&x.notes);
    if x.unmapped > 0 {
        out.push_str(&format!("\n[{} characters were left out: their fonts have no Unicode mapping]\n", x.unmapped));
    }
    Ok(out)
}

/// The text of one read, before [`read_with`] formats it.
pub(crate) struct Extract {
    /// Pages in the file.
    pub(crate) total: usize,
    /// Each page read: its number, and where its text starts and ends in `body`.
    pub(crate) pages: Vec<(usize, usize, usize)>,
    /// A `page N of M` line and the text of each page read.
    pub(crate) body: String,
    /// The notes on limits, each in brackets on its own line.
    pub(crate) notes: String,
    /// Characters left out because their fonts have no Unicode mapping.
    pub(crate) unmapped: usize,
    /// When the pages read have no text: the line that says why, notes included.
    pub(crate) no_text: Option<String>,
}

/// Reads the pages of `data` within `limits` and `budget`.
pub(crate) fn extract(data: &[u8], limits: &Limits, budget: &Budget) -> Result<Extract, String> {
    let structure = |e: Exhausted| if e == Exhausted::Cancelled { CANCELLED.to_string() } else { format!("PDF work limit reached while reading the file structure: {} limit", e.what()) };
    let doc = match Document::open(data, budget) {
        Ok(doc) => doc,
        Err(e) => return Err(budget.stopped().map_or(e, structure)),
    };
    let tree = doc.pages(limits.pages.min(MAX_PAGES));
    let (pages, total) = (&tree.list, tree.total);
    // A budget that stopped in the walk leaves nothing to read the pages with.
    if let Some(e) = budget.stopped() {
        return Err(structure(e));
    }
    if total == 0 {
        return Err("PDF has no pages".into());
    }
    let mut fonts = content::Fonts::default();
    let mut body = String::new();
    let mut marks = Vec::new();
    let mut unmapped = 0;
    let mut chars = 0;
    // The last page read, and whether the text cap cut it.
    let mut last = 0;
    let mut cut = false;
    // Pages where a page limit skipped some content.
    let mut short = Vec::new();
    // The page where the whole-read budget stopped, and why.
    let mut stop = None;
    let mut form_calls = limits.document_form_calls;
    for (i, page) in pages.iter().enumerate() {
        if let Some(e) = budget.stopped() {
            if e == Exhausted::Cancelled {
                return Err(CANCELLED.into());
            }
            stop = Some((last, e));
            break;
        }
        last = i + 1;
        let room = limits.text_bytes.saturating_sub(body.len());
        let page_limits = Limits { form_calls: limits.form_calls.min(form_calls), ..*limits };
        let mut reader = content::PageText::new(&doc, &mut fonts, &page_limits, room);
        let Some(dict) = page.dict.as_dict() else { continue };
        // Concatenated streams form one content stream (7.8.2). Each use is
        // charged, and the joined content stops at the page limits.
        let mut data = Vec::new();
        let mut add = |s: &Object| {
            if budget.take(budget::Kind::Steps, 1).is_none() {
                return false;
            }
            let room = limits.page_bytes.saturating_sub(data.len());
            if let Some(d) = doc.stream_data_within(s, room) {
                data.extend_from_slice(&d);
                data.push(b'\n');
            }
            // False once no more content can join this page.
            data.len() < limits.page_bytes && budget.stopped().is_none()
        };
        let contents = doc.key(dict, b"Contents");
        // Set when entries of /Contents were left unread.
        let mut over = false;
        match &*contents {
            Object::Array(a) => {
                let mut more = true;
                for (k, c) in a.iter().enumerate() {
                    if k == limits.page_streams || !more {
                        over = true;
                        break;
                    }
                    more = add(&doc.resolve(c));
                }
            }
            s => { add(s); }
        }
        reader.run(&data, &page.resources);
        if let Some(e) = budget.stopped() {
            if e == Exhausted::Cancelled {
                return Err(CANCELLED.into());
            }
            stop = Some((i + 1, e));
        }
        if (doc.take_cut() | reader.short | over) && stop.is_none() {
            short.push(i + 1);
        }
        form_calls -= page_limits.form_calls - reader.form_calls;
        unmapped += reader.unmapped;
        cut = reader.full;
        let text = reader.out.trim_end();
        chars += text.chars().filter(|c| !c.is_whitespace()).count();
        if i > 0 {
            body.push_str("\n\n");
        }
        body.push_str(&format!("page {} of {total}\n", i + 1));
        let start = body.len();
        body.push_str(if text.is_empty() { "(no text on this page)" } else { text });
        marks.push((i + 1, start, body.len()));
        if cut || body.len() > limits.text_bytes || stop.is_some() {
            break;
        }
    }
    if body.len() > limits.text_bytes {
        let end = (0..=limits.text_bytes).rev().find(|&i| body.is_char_boundary(i)).unwrap_or(0);
        body.truncate(end);
        cut = true;
        marks.retain(|m| m.1 <= end);
        for m in &mut marks {
            m.2 = m.2.min(end);
        }
    }
    let mut notes = String::new();
    if cut {
        let limit = if limits.text_bytes % (1 << 20) == 0 { format!("{} MiB", limits.text_bytes >> 20) } else { format!("{} bytes", limits.text_bytes) };
        notes.push_str(&format!("\n[text stopped inside page {last} of {total}: extracted text limit {limit}]\n"));
    } else if let Some((page, e)) = stop {
        notes.push_str(&format!("\n[work limit reached on page {page} of {total} ({} limit): the read stopped there]\n", e.what()));
    } else if pages.len() < total {
        notes.push_str(&format!("\n[stopped after page {} of {total}: page limit {}]\n", pages.len(), limits.pages.min(MAX_PAGES)));
    }
    if tree.incomplete {
        notes.push_str("\n[page tree: a limit skipped some branches, so some pages were not found]\n");
    }
    if !short.is_empty() {
        let list: Vec<String> = short.iter().take(10).map(usize::to_string).collect();
        let more = if short.len() > 10 { format!(" and {} more", short.len() - 10) } else { String::new() };
        notes.push_str(&format!("\n[work limit reached on page {}{more}: some content there was not read]\n", list.join(", ")));
    }
    // A no-text verdict covers only the pages read.
    let (scope, what) = if last < total { (format!(" in the first {last} of {total} pages"), "these pages have") } else { (format!(", {total} pages"), "this PDF has") };
    let no_text = if chars == 0 && unmapped > 0 {
        Some(format!("no readable text{scope}: the PDF shows {unmapped} characters, but its fonts have no Unicode mapping (no ToUnicode CMap; common for CJK CID fonts), so Vade cannot decode them.\n{notes}"))
    } else if chars == 0 && (stop.is_some() || tree.incomplete || !short.is_empty()) {
        // Content was skipped: an unread text layer is not an absent one.
        Some(format!("no text found in the content read{scope}: a work limit skipped some content (see the notes below), so the PDF may still have a text layer.\n{notes}"))
    } else if chars == 0 {
        Some(format!("no text layer{scope}: {what} no extractable text (scanned images or text drawn as shapes); Vade does not run OCR.\n{notes}"))
    } else {
        None
    };
    Ok(Extract { total, pages: marks, body, notes, unmapped, no_text })
}
