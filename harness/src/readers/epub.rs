//! EPUB 2 and 3 to Markdown. `META-INF/container.xml` names the package
//! document; its spine gives the reading order. Each spine item gets a
//! chapter marker, and its XHTML goes through the HTML reader with the same
//! budget. The title and authors from the package metadata come first.
//!
//! A book with encrypted content (DRM) is refused. Font obfuscation, which
//! hides only embedded fonts, is not encryption of the text. Skipped:
//! images (their alt text shows), styles, scripts and the navigation file
//! when it is not in the spine.
use super::xml::{self, Event, Scanner};
use super::zip::{self, Zip};
use super::{Budget, Caps, Doc, Format, ReadError, Reader, chapter_marker, extension, html, is_zip, page_cap_warning, text};
use std::collections::HashMap;

pub(super) struct EpubReader;

/// Storage of one manifest or spine entry, for the work cap.
const SLOT: usize = 48;
/// Encryption methods that hide fonts only (IDPF and Adobe obfuscation).
const FONT_ONLY: &[&str] = &["http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"];

impl Reader for EpubReader {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format> {
        if zip::mimetype(bytes) == Some("application/epub+zip") {
            return Some(Format::Epub);
        }
        (!name.is_empty() && extension(name) == "epub").then_some(Format::Epub)
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        if !is_zip(bytes) {
            return Err(ReadError::Malformed(format, "the file is not a ZIP archive".into()));
        }
        let mut budget = Budget::new(caps);
        let mut zip = Zip::open(bytes, format)?;
        let bad = |name: &str, why: String| ReadError::Malformed(format, format!("{name} {why}"));
        const CONTAINER: &str = "META-INF/container.xml";
        let (text, cut) = zip.text(CONTAINER)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {CONTAINER}")))?;
        let opf = rootfile(&text, cut, &mut budget).map_err(|why| bad(CONTAINER, why))?.ok_or_else(|| bad(CONTAINER, "names no package document".into()))?;
        if let Some((text, cut)) = zip.text("META-INF/encryption.xml")?
            && encrypted(&text, cut, &mut budget).map_err(|why| bad("META-INF/encryption.xml", why))?
        {
            return Err(ReadError::Encrypted(format));
        }
        let (text, cut) = zip.text(&opf)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {opf}")))?;
        let dir = opf.rsplit_once('/').map_or("", |(d, _)| d);
        let book = package(&text, cut, dir, &mut budget).map_err(|why| bad(&opf, why))?;
        let total = book.spine.len();
        // A spine that the work cap stopped, or whose ids the attribute cap
        // left out, is not damage: the title stays.
        if total == 0 && !budget.full() && !budget.long_attr {
            return Err(bad(&opf, "has an empty spine".into()));
        }
        let mut doc = Doc::new(String::new(), total);
        // The title and authors were counted as text when they were read:
        // they move into the output, also after a later stop.
        budget.release(book.held);
        if !book.title.is_empty() {
            doc.markdown.push_str(budget.work(&format!("# {}\n", book.title)));
            if !book.authors.is_empty() {
                let more = if book.more > 0 { format!(" and {} more", book.more) } else { String::new() };
                doc.markdown.push_str(budget.work(&format!("\nby {}{more}\n", book.authors.join(", "))));
            }
        }
        for (n, id) in book.spine.iter().enumerate() {
            if n == caps.max_pages {
                doc.warnings.push(page_cap_warning(caps.max_pages, total, "chapters"));
                break;
            }
            // The work cap stopped the manifest before this item, or the
            // attribute cap left out its id or path.
            if budget.full() || (!book.items.contains_key(id) && (book.stopped || budget.long_attr)) {
                break;
            }
            // A marker shows only whole.
            let marker = chapter_marker(n + 1, total);
            let marker = if doc.markdown.is_empty() { marker.trim_start() } else { &marker };
            if !budget.fits(marker.len()) {
                budget.cut();
                break;
            }
            doc.markdown.push_str(budget.work(marker));
            let body = match book.items.get(id) {
                Some((href, media)) if matches!(media.as_str(), "application/xhtml+xml" | "text/html" | "application/xml" | "text/xml") => {
                    let mut part = zip.part(href)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {href}")))?;
                    if part.cut {
                        zip::drop_cut_char(&mut part.data);
                    }
                    let (source, _) = text::decode(&part.data).ok_or_else(|| bad(href, "is not text".into()))?;
                    // The HTML reader counts its output in the budget.
                    html::to_markdown_in(&source, &mut budget)
                }
                Some(_) => String::new(),
                None => return Err(bad(&opf, format!("lists the spine item {id}, which the manifest does not have"))),
            };
            let body = body.trim();
            if body.is_empty() {
                budget.whole(&mut doc.markdown, "(no text in this chapter)");
            } else {
                doc.markdown.push_str(body);
            }
            doc.markdown.push('\n');
        }
        doc.warnings.append(&mut zip.warnings);
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

/// The package document that `container.xml` names first.
fn rootfile(text: &str, cut: bool, budget: &mut Budget) -> Result<Option<String>, String> {
    let mut scan = Scanner::new(text, cut);
    while let Some(event) = scan.next()? {
        if let Event::Open(tag) = event
            && tag.local() == "rootfile"
            && tag.attr("media-type")?.is_none_or(|m| m == "application/oebps-package+xml")
            && let Some(path) = tag.path("full-path")?
        {
            return Ok(Some(zip::resolve("", &path)));
        }
    }
    scan.finish(budget)?;
    Ok(None)
}

/// True when `encryption.xml` encrypts anything with a method that is not
/// font obfuscation.
fn encrypted(text: &str, cut: bool, budget: &mut Budget) -> Result<bool, String> {
    let mut scan = Scanner::new(text, cut);
    while let Some(event) = scan.next()? {
        if let Event::Open(tag) = event
            && tag.local() == "EncryptionMethod"
            && !tag.attr("Algorithm")?.is_some_and(|a| FONT_ONLY.contains(&&*a))
        {
            return Ok(true);
        }
    }
    scan.finish(budget)?;
    Ok(false)
}

struct Book {
    title: String,
    /// The first 20 authors, and the count of the others.
    authors: Vec<String>,
    more: usize,
    /// Work bytes that the title and authors hold.
    held: usize,
    /// The work cap stopped the manifest or spine.
    stopped: bool,
    /// Manifest id to (part name, media type).
    items: HashMap<String, (String, String)>,
    spine: Vec<String>,
}

/// The metadata, manifest and spine of the package document in `dir`.
fn package(text: &str, cut: bool, dir: &str, budget: &mut Budget) -> Result<Book, String> {
    let mut scan = Scanner::new(text, cut);
    let mut book = Book { title: String::new(), authors: Vec::new(), more: 0, held: 0, stopped: false, items: HashMap::new(), spine: Vec::new() };
    // The open metadata element whose text is wanted, and that text.
    let mut field: Option<(&str, String)> = None;
    while let Some(event) = scan.next()? {
        match event {
            Event::Open(tag) => match tag.local() {
                "title" if book.title.is_empty() => field = Some(("title", String::new())),
                "creator" => field = Some(("creator", String::new())),
                "item" => {
                    let (Some(id), Some(href)) = (tag.attr("id")?, tag.path("href")?) else { continue };
                    let media = tag.attr("media-type")?.unwrap_or_default().into_owned();
                    // A resolved name is never longer than the directory and the name.
                    if !budget.reserve(SLOT + id.len() + dir.len() + 1 + href.len() + media.len()) {
                        book.stopped = true;
                        break;
                    }
                    book.items.insert(id.into_owned(), (zip::resolve(dir, &href), media));
                }
                "itemref" => {
                    let Some(id) = tag.attr("idref")? else { continue };
                    if !budget.reserve(SLOT + id.len()) {
                        book.stopped = true;
                        break;
                    }
                    book.spine.push(id.into_owned());
                }
                _ => {}
            },
            Event::Text(raw) | Event::Cdata(raw) => {
                if let Some((_, value)) = field.as_mut()
                    && value.len() < 1000
                {
                    let text = if matches!(event, Event::Text(_)) { xml::unescape_prefix(raw, 1000)? } else { raw.into() };
                    super::xlsx::keep_prefix(value, &text, 1000 - value.len());
                }
            }
            Event::Close(name) => {
                let local = xml::local(name);
                if let Some((which, value)) = field.take_if(|(w, _)| *w == local) {
                    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
                    if value.is_empty() {
                        continue;
                    }
                    if which == "creator" && book.authors.len() == 20 {
                        book.more += 1;
                        continue;
                    }
                    // The metadata counts as text now and moves into the
                    // output later.
                    let kept = budget.text(&value).to_owned();
                    if kept.is_empty() || !budget.reserve(SLOT) {
                        budget.release(kept.len());
                        continue;
                    }
                    book.held += SLOT + kept.len();
                    if which == "title" {
                        book.title = kept;
                    } else {
                        book.authors.push(kept);
                    }
                }
            }
        }
    }
    scan.finish(budget)?;
    Ok(book)
}
