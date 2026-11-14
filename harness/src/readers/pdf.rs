//! PDF to Markdown for `vade read`: the PDF reader's text layer
//! (`crate::pdf`), with a page marker before each page. The PDF reader keeps
//! its own work budget and limits; the caller's page cap and work cap limit
//! the pages it reads and the text it keeps, and its notes on limits become
//! warnings. The agent's `workspace_read` keeps its own PDF path
//! (`crate::pdf::text`).
use super::{Budget, Caps, Doc, Format, ReadError, Reader, extension, page_cap_warning, page_marker};
use crate::pdf::{self as text_layer, Limits};
use std::sync::atomic::AtomicBool;

pub(super) struct PdfReader;

/// The cancel flag of the PDF budget; a read is never cancelled.
static NEVER: AtomicBool = AtomicBool::new(false);

impl Reader for PdfReader {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format> {
        if text_layer::is_pdf(&bytes[..bytes.len().min(1024)]) {
            return Some(Format::Pdf);
        }
        (!name.is_empty() && extension(name) == "pdf").then_some(Format::Pdf)
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        if !text_layer::is_pdf(&bytes[..bytes.len().min(1024)]) {
            return Err(ReadError::Malformed(format, "the file has no PDF header (%PDF-) in its first 1024 bytes".into()));
        }
        let mut budget = Budget::new(caps);
        // The extraction reads no page past the page cap and keeps no more
        // text than the work cap.
        let limits = Limits { pages: caps.max_pages.max(1), text_bytes: Limits::DEFAULT.text_bytes.min(budget.max_bytes), ..Limits::DEFAULT };
        let work = text_layer::budget::Budget::new(&limits, &NEVER);
        let x = text_layer::extract(bytes, &limits, &work).map_err(|e| {
            if e.contains("encrypted") { ReadError::Encrypted(format) } else { ReadError::Malformed(format, e) }
        })?;
        // The page cap and work cap notes take the words of the other readers.
        let mut text_cut = false;
        let mut notes = Vec::new();
        for note in x.notes.lines().filter_map(|l| l.trim().strip_prefix('[')?.strip_suffix(']')) {
            if note.starts_with("stopped after page") && x.total > caps.max_pages {
                notes.push(page_cap_warning(caps.max_pages, x.total, "pages"));
            } else if note.starts_with("text stopped inside page") && limits.text_bytes < Limits::DEFAULT.text_bytes {
                text_cut = true;
            } else {
                notes.push(note.to_owned());
            }
        }
        if let Some(line) = x.no_text {
            // Only a file read to its last page has no text layer; after a
            // cap the line says which pages were read, and the cap note stays.
            if line.starts_with("no text layer") && x.pages.len() >= x.total {
                return Err(ReadError::NoText(format, x.total));
            }
            let mut doc = Doc::new(line.lines().next().unwrap_or("").to_owned(), x.total);
            doc.warnings = notes;
            return Ok(doc);
        }
        let mut doc = Doc::new(String::new(), x.total);
        for (i, &(page, start, end)) in x.pages.iter().enumerate() {
            if budget.full() {
                break;
            }
            // A marker shows only whole.
            let marker = page_marker(page, x.total);
            let marker = if i == 0 { marker.trim_start() } else { &marker };
            if !budget.fits(marker.len()) {
                budget.cut();
                break;
            }
            doc.markdown.push_str(budget.work(marker));
            doc.markdown.push_str(budget.text(&x.body[start..end]));
            doc.markdown.push('\n');
        }
        if text_cut {
            budget.cut();
        }
        doc.warnings.extend(notes);
        if x.unmapped > 0 {
            doc.warnings.push(format!("{} characters with no Unicode mapping in their fonts", x.unmapped));
        }
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}
