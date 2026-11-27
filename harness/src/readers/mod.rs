//! Document readers: file bytes in, Markdown out.
//!
//! The engine reads only text, so every file the agent or `vade read` opens
//! goes through this one layer. [`read`] sniffs the format (magic bytes first,
//! then the file extension), runs the [`Reader`] for it and returns a [`Doc`].
//! [`render`] adds the header comment line. The conventions are fixed here and
//! documented in `docs/reading-files.md`:
//!
//! - First line: `<!-- vade read: <name> | <format> | pages N | chars C | skipped: ... -->`.
//! - Pages: `<!-- page N of M -->` then `## Page N` ([`page_marker`]). Slides
//!   use the same form with `slide` and `Slide`. EPUB chapters use `chapter` and
//!   `Chapter` ([`chapter_marker`]). Sheets use
//!   `<!-- sheet N of M -->` then `## Sheet <name>` ([`sheet_marker`]).
//!   Archive members use `<!-- file N of M -->` then `## File <path>`
//!   ([`file_marker`]).
//! - Tables are GitHub Markdown tables ([`table`]). Code and data are fenced
//!   blocks with the language ([`fence`]).
//! - Caps ([`Caps`]): 50 MB file, 2,000 pages, 2,000,000 chars. Readers apply
//!   the char cap and a work cap while they convert, through a [`Budget`]. A
//!   reader that stops at a cap says so in a warning; the warning goes in
//!   `skipped:` and in a note at the end of the text.
//! - An unsupported, encrypted or damaged file gives one clear line
//!   ([`ReadError`]), never partial garbage.
//!
//! A new format adds one [`Reader`] and one line in [`READERS`].
use std::fmt;

mod archive;
mod codepage;
mod csv;
mod docx;
mod epub;
mod html;
mod image;
mod odf;
mod pdf;
mod pptx;
mod rtf;
mod text;
mod xlsx;
mod xml;
mod zip;
#[cfg(test)]
mod audit_tests;
#[cfg(test)]
mod corpus_tests;
#[cfg(test)]
mod office_tests;
#[cfg(test)]
mod tests;

pub use html::{to_markdown as html_to_markdown, to_markdown_in as html_to_markdown_in};

/// Largest file a reader opens.
pub const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;
/// Most pages, slides or sheets a reader converts.
pub const MAX_PAGES: usize = 2_000;
/// Most characters of Markdown body kept.
pub const MAX_CHARS: usize = 2_000_000;
/// Most table rows kept for one CSV file or one sheet.
pub const MAX_ROWS: usize = 10_000;

/// A file format. Variants without a registered reader still sniff, so the
/// reply names the format in its "not supported" line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Text,
    Markdown,
    Html,
    Csv,
    Tsv,
    Json,
    Yaml,
    Toml,
    Xml,
    /// Source code; the value is the fence language, for example `rust`.
    Source(&'static str),
    Rtf,
    Pdf,
    Docx,
    Xlsx,
    Pptx,
    Epub,
    Odt,
    Ods,
    Odp,
    /// An image; the value is the format, for example `PNG`.
    Image(&'static str),
    /// An archive that the archive reader lists and reads; the value is the
    /// label, for example `ZIP archive`.
    Archive(&'static str),
    /// A recognized format that no reader converts, for example `PNG image`.
    Other(&'static str),
}

impl Format {
    /// The name in the header line and in error lines.
    pub fn label(self) -> &'static str {
        match self {
            Format::Text => "text",
            Format::Markdown => "Markdown",
            Format::Html => "HTML",
            Format::Csv => "CSV",
            Format::Tsv => "TSV",
            Format::Json => "JSON",
            Format::Yaml => "YAML",
            Format::Toml => "TOML",
            Format::Xml => "XML",
            Format::Source(lang) => lang,
            Format::Rtf => "RTF",
            Format::Pdf => "PDF",
            Format::Docx => "DOCX",
            Format::Xlsx => "XLSX",
            Format::Pptx => "PPTX",
            Format::Epub => "EPUB",
            Format::Odt => "ODT",
            Format::Ods => "ODS",
            Format::Odp => "ODP",
            Format::Image("PNG") => "PNG image",
            Format::Image("JPEG") => "JPEG image",
            Format::Image("GIF") => "GIF image",
            Format::Image("WebP") => "WebP image",
            Format::Image(_) => "image",
            Format::Archive(name) | Format::Other(name) => name,
        }
    }

    /// True when the raw bytes are already the text a person edits. The agent
    /// read path keeps these files byte-identical and converts the others.
    pub fn is_plain_text(self) -> bool {
        matches!(
            self,
            Format::Text | Format::Markdown | Format::Html | Format::Csv | Format::Tsv | Format::Json
                | Format::Yaml | Format::Toml | Format::Xml | Format::Source(_)
        )
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Limits for one read. Every reader must respect `max_pages` and stop at
/// `max_chars` through a [`Budget`]; [`read`] enforces `max_bytes` and cuts
/// the result to `max_chars`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    pub max_bytes: usize,
    pub max_pages: usize,
    pub max_chars: usize,
    pub max_rows: usize,
}

impl Default for Caps {
    fn default() -> Self {
        Self { max_bytes: MAX_FILE_BYTES, max_pages: MAX_PAGES, max_chars: MAX_CHARS, max_rows: MAX_ROWS }
    }
}

/// One converted document.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Doc {
    /// The body, without the header line.
    pub markdown: String,
    /// Pages, slides or sheets in the source (1 for a file with no pages).
    pub pages: usize,
    /// Characters in `markdown`; [`read`] sets it after the char cap.
    pub chars: usize,
    /// What was skipped or approximated, one short phrase each, for example
    /// `3 images` or `pages after 2000 of 5000 (pages cap)`.
    pub warnings: Vec<String>,
}

impl Doc {
    pub fn new(markdown: String, pages: usize) -> Self {
        Self { markdown, pages, chars: 0, warnings: Vec::new() }
    }
}

/// Retained bytes a reader may build for each character of `Caps::max_chars`.
/// Table padding, list indents and quote prefixes count, and so does text
/// that moves from a nested block into its parent.
pub const WORK_BYTES_PER_CHAR: usize = 16;
/// The smallest work allowance, so a small char cap still shows a table frame.
pub const MIN_WORK_BYTES: usize = 64 * 1024;

/// The caps a reader checks while it converts. [`read`] cuts the result to
/// `max_chars` after the reader returns, but a reader must not build more
/// first: each reader makes one `Budget` from its [`Caps`], takes every piece
/// of output through it and stops when [`Budget::full`] is true.
///
/// Two limits apply. Text is counted in characters against `max_chars`.
/// All retained output, markup included, is counted in bytes against
/// `max_chars * WORK_BYTES_PER_CHAR` (at least [`MIN_WORK_BYTES`]). A nested
/// block that moves into its parent gives its bytes back first, so the count
/// is the output held at one time. Parser storage (open groups, fonts, table
/// rows, list levels) counts too, before it is allocated. A parser keeps no
/// copy of source text that is larger than the room left. Decoded source text
/// is not counted; it is at most three bytes for each source byte.
///
/// After a stop, [`Budget::text`] keeps nothing more, but [`Budget::work`]
/// still keeps markup and moved text within the byte cap, so the blocks that
/// are open when a cap stops the reader still close with their text.
#[derive(Clone, Debug)]
pub struct Budget {
    chars: usize,
    bytes: usize,
    max_chars: usize,
    max_bytes: usize,
    hit: Option<Stop>,
    /// False while text that was counted once moves into a parent block, and
    /// while the open blocks close after a stop.
    text_counts: bool,
    /// A parser left out an attribute value longer than 4096 bytes.
    long_attr: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Chars,
    Bytes,
}

impl Budget {
    pub fn new(caps: &Caps) -> Self {
        let max_bytes = caps.max_chars.saturating_mul(WORK_BYTES_PER_CHAR).max(MIN_WORK_BYTES);
        Self { chars: 0, bytes: 0, max_chars: caps.max_chars, max_bytes, hit: None, text_counts: true, long_attr: false }
    }

    /// True after a cap stopped a piece; the reader stops converting.
    pub fn full(&self) -> bool {
        self.hit.is_some()
    }

    /// Bytes of work still allowed.
    pub fn room(&self) -> usize {
        if self.full() { 0 } else { self.max_bytes - self.bytes }
    }

    /// The longest prefix of `text` within both caps, counted as output text.
    pub fn text<'a>(&mut self, text: &'a str) -> &'a str {
        self.take(text, true)
    }

    /// The longest prefix of `text` within the work cap only. Use it for
    /// markup and for text that moves from a nested block into its parent,
    /// which [`Budget::text`] already counted once.
    pub fn work<'a>(&mut self, text: &'a str) -> &'a str {
        self.take(text, false)
    }

    /// Appends the part of `text` that fits to `out`; false when a cap cut it.
    pub fn push(&mut self, out: &mut String, text: &str) -> bool {
        let kept = self.text(text);
        out.push_str(kept);
        kept.len() == text.len()
    }

    /// Counts `bytes` of parser storage (a group, a font, a table row) against
    /// the work cap before the reader allocates it. False, and a stop, when
    /// they do not fit.
    fn reserve(&mut self, bytes: usize) -> bool {
        if bytes > self.max_bytes - self.bytes {
            self.hit.get_or_insert(Stop::Bytes);
            return false;
        }
        self.bytes += bytes;
        true
    }

    /// True when `bytes` more of [`Budget::work`] fit whole, also after a
    /// stop, so a reader can leave out a marker or link it cannot close.
    fn fits(&self, bytes: usize) -> bool {
        bytes <= self.max_bytes - self.bytes
    }

    /// Appends the note or marker `text` to `out` as [`Budget::work`], only
    /// whole: when it does not fit, nothing, and a stop at the memory cap.
    fn whole(&mut self, out: &mut String, text: &str) {
        if self.fits(text.len()) {
            out.push_str(self.work(text));
        } else {
            self.cut();
        }
    }

    /// Stops the reader when it had to leave out source text to keep its
    /// storage within the work cap, so the memory cap warning says so.
    fn cut(&mut self) {
        self.hit.get_or_insert(Stop::Bytes);
    }

    /// Gives back the bytes of a nested block whose text moves into its
    /// parent, or of storage that [`Budget::reserve`] counted.
    fn release(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_sub(bytes);
    }

    fn take<'a>(&mut self, text: &'a str, counted: bool) -> &'a str {
        let counted = counted && self.text_counts;
        if text.is_empty() || (counted && self.hit.is_some()) {
            return &text[..0];
        }
        let mut end = text.len().min(self.max_bytes - self.bytes);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        if end < text.len() {
            // The first stop names the cap.
            self.hit.get_or_insert(Stop::Bytes);
        }
        let mut chars = 0;
        if counted {
            // A piece with no more bytes than the chars left cannot pass the
            // cap. Otherwise one pass counts up to the chars left and cuts there.
            let left = self.max_chars - self.chars;
            if end <= left {
                chars = text[..end].chars().count();
            } else {
                for (at, _) in text[..end].char_indices() {
                    if chars == left {
                        end = at;
                        self.hit = Some(Stop::Chars);
                        break;
                    }
                    chars += 1;
                }
            }
        }
        self.bytes += end;
        self.chars += chars;
        &text[..end]
    }

    /// The warnings for the cap that stopped the reader, if any, and for
    /// attribute values left out. The char cap warning is the same phrase
    /// [`read`] uses.
    pub fn warning(&self) -> Vec<String> {
        let stop = self.hit.map(|hit| match hit {
            Stop::Chars => char_cap_warning(self.max_chars),
            Stop::Bytes => format!("text after {} bytes of conversion work (memory cap)", self.max_bytes),
        });
        let attr = self.long_attr.then(|| format!("attribute values over {} bytes left out (attribute cap)", xml::MAX_ATTR));
        stop.into_iter().chain(attr).collect()
    }
}

#[cfg(test)]
thread_local! {
    /// Source bytes that the scanners on this thread looked at, so that a
    /// test can prove a reader linear without a clock.
    pub(super) static SCANNED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Counts `bytes` of source that a reader scanned (see [`SCANNED`]).
#[cfg(test)]
fn scanned(bytes: usize) {
    SCANNED.with(|n| n.set(n.get() + bytes));
}

#[cfg(not(test))]
fn scanned(_: usize) {}

fn char_cap_warning(max_chars: usize) -> String {
    format!("text after {max_chars} chars (char cap)")
}

/// Why a file gives no Markdown. Each variant prints as one line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// A format that no reader converts yet.
    Unsupported(Format),
    /// Bytes that are not text and match no known format.
    Binary,
    /// The file is encrypted or password protected.
    Encrypted(Format),
    /// The file is larger than `Caps::max_bytes`.
    TooLarge { bytes: u64, cap: usize },
    /// The file claims a format but its structure is broken; the text says what.
    Malformed(Format, String),
    /// A document with no text layer, for example a scanned PDF; the value is the page count.
    NoText(Format, usize),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Unsupported(Format::Other(name)) => write!(f, "Vade does not read {name} files; no text was read"),
            ReadError::Unsupported(format) => write!(f, "{} files are not supported yet; no text was read", format.label()),
            ReadError::Binary => f.write_str("binary file with no known format; no text was read"),
            ReadError::Encrypted(format) => write!(f, "this {} file is encrypted; Vade cannot read encrypted files", format.label()),
            ReadError::TooLarge { bytes, cap } => write!(f, "file is {bytes} bytes; the reader limit is {cap} bytes"),
            ReadError::Malformed(format, why) => write!(f, "damaged {} file: {why}; no text was read", format.label()),
            ReadError::NoText(format, pages) => write!(
                f,
                "this {} file has no text layer ({pages} pages; scanned images?); Vade does not read images yet",
                format.label()
            ),
        }
    }
}

impl std::error::Error for ReadError {}

/// One file format family. Readers are stateless and shared.
pub trait Reader: Sync {
    /// The format of `bytes`, or `None`. Check magic bytes first. [`read`]
    /// calls every reader with an empty `name` first and with the real name
    /// second, so content always wins over the extension.
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format>;
    /// Converts a file that `sniff` accepted as `format`. Respect
    /// `caps.max_pages` and report each cap you hit in `Doc::warnings`.
    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError>;
}

/// The registry, in sniff order. The text reader is last. Bytes that no
/// reader or known format claims are read as text when they decode as text.
pub static READERS: &[&dyn Reader] = &[
    &rtf::RtfReader, &pdf::PdfReader, &docx::DocxReader, &xlsx::XlsxReader, &pptx::PptxReader, &epub::EpubReader, &odf::OdfReader,
    &archive::ArchiveReader, &image::ImageReader, &html::HtmlReader, &csv::CsvReader, &text::TextReader,
];

/// The lowercase extension of `name`, or `""`.
pub fn extension(name: &str) -> String {
    let leaf = name.rsplit(['/', '\\']).next().unwrap_or(name);
    match leaf.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => ext.to_ascii_lowercase(),
        _ => String::new(),
    }
}

/// Recognized formats with no reader in this build: magic bytes first, then
/// the extension (`by_name`). DOCX, XLSX, EPUB and plain ZIP share magic
/// bytes, so [`sniff`] names a ZIP file by its extension among the ZIP-based
/// formats. Office and PDF entries move to their readers when they land.
fn known(bytes: &[u8], ext: &str, by_name: bool) -> Option<Format> {
    const MAGIC: &[(&[u8], Format)] = &[
        (b"%PDF-", Format::Pdf),
        (b"II*\0", Format::Other("TIFF image")),
        (b"MM\0*", Format::Other("TIFF image")),
        (b"\x1f\x8b", Format::Other("gzip archive")),
        (b"7z\xbc\xaf\x27\x1c", Format::Other("7z archive")),
        (b"Rar!\x1a\x07", Format::Other("RAR archive")),
        (b"\xfd7zXZ\0", Format::Other("xz archive")),
        (b"\x28\xb5\x2f\xfd", Format::Other("zstd archive")),
        (b"\x7fELF", Format::Other("ELF executable")),
        (b"\xcf\xfa\xed\xfe", Format::Other("Mach-O executable")),
        (b"\xca\xfe\xba\xbe", Format::Other("Mach-O executable")),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", Format::Other("legacy Office (OLE)")),
        (b"SQLite format 3\0", Format::Other("SQLite database")),
        (b"bplist00", Format::Other("binary property list")),
        (b"\0asm", Format::Other("WebAssembly module")),
    ];
    if let Some((_, format)) = MAGIC.iter().find(|(magic, _)| bytes.starts_with(magic)) {
        return Some(*format);
    }
    // bzip2: `BZh`, the block size digit, then the block magic (pi in BCD).
    if bytes.len() >= 10 && bytes.starts_with(b"BZh") && (b'1'..=b'9').contains(&bytes[3]) && &bytes[4..10] == b"1AY&SY" {
        return Some(Format::Other("bzip2 archive"));
    }
    if !by_name {
        return None;
    }
    let by_ext = match ext {
        "doc" | "xls" | "ppt" => Format::Other("legacy Office (OLE)"),
        _ => return None,
    };
    Some(by_ext)
}

/// The magic bytes of an OLE compound file (legacy Office, or an encrypted
/// Office Open XML package).
const OLE: &[u8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1";

fn is_ole(bytes: &[u8]) -> bool {
    bytes.starts_with(OLE)
}

/// The ZIP container of an Office Open XML file. Office saves a file with a
/// password as an OLE file with an `EncryptedPackage` stream; any other OLE
/// file is a legacy Office file.
fn office_zip<'a>(bytes: &'a [u8], format: Format) -> Result<zip::Zip<'a>, ReadError> {
    if !is_ole(bytes) && !is_zip(bytes) {
        return Err(ReadError::Malformed(format, "the file is not a ZIP archive".into()));
    }
    if is_ole(bytes) {
        let name: Vec<u8> = "EncryptedPackage".encode_utf16().flat_map(u16::to_le_bytes).collect();
        scanned(bytes.len());
        let encrypted = bytes.windows(name.len()).any(|w| w == name);
        return Err(if encrypted { ReadError::Encrypted(format) } else { ReadError::Unsupported(Format::Other("legacy Office (OLE)")) });
    }
    zip::Zip::open(bytes, format)
}

/// True for the ZIP container magic (a local file header or an empty archive).
fn is_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06")
}

/// The formats stored in a ZIP container.
fn zip_based(format: Format) -> bool {
    matches!(
        format,
        Format::Docx | Format::Xlsx | Format::Pptx | Format::Epub | Format::Odt | Format::Ods | Format::Odp
            | archive::ZIP
    )
}

/// The format of `bytes` and the reader for it (`None` when no reader
/// converts the format). Order: reader magic, known magic, reader
/// extension, known extension, then text by content. ZIP magic is content
/// too: the extension then only picks among the ZIP-based formats, so a ZIP
/// file named `notes.txt` is never read as text.
pub fn sniff(bytes: &[u8], name: &str) -> Option<(Format, Option<&'static dyn Reader>)> {
    let ext = extension(name);
    for r in READERS {
        if let Some(format) = r.sniff(bytes, "") {
            return Some((format, Some(*r)));
        }
    }
    // An encrypted Office Open XML file is an OLE file with the Office extension.
    if is_ole(bytes) {
        for r in READERS {
            if let Some(format) = r.sniff(bytes, name).filter(|f| zip_based(*f)) {
                return Some((format, Some(*r)));
            }
        }
    }
    if let Some(format) = known(bytes, "", false) {
        return Some((format, None));
    }
    if is_zip(bytes) {
        for r in READERS {
            if let Some(format) = r.sniff(bytes, name).filter(|f| zip_based(*f)) {
                return Some((format, Some(*r)));
            }
        }
        // Any other ZIP file is an archive.
        return Some((archive::ZIP, Some(&archive::ArchiveReader)));
    }
    for r in READERS {
        if let Some(format) = r.sniff(bytes, name) {
            return Some((format, Some(*r)));
        }
    }
    if let Some(format) = known(bytes, &ext, true) {
        return Some((format, None));
    }
    text::looks_like_text(bytes).then_some((Format::Text, Some(&text::TextReader as &dyn Reader)))
}

/// True when `sample` (the start of a file) is UTF-16 text: a UTF-16 byte
/// order mark, or the zero-byte pattern of UTF-16 with no mark. Such bytes
/// can also be valid UTF-8, so test this before `std::str::from_utf8`.
pub fn is_utf16(sample: &[u8]) -> bool {
    !sample.starts_with(b"\xef\xbb\xbf") && text::utf16_order(sample).is_some()
}

/// Converts `bytes` (the file `name`) to Markdown within `caps`. Enforces the
/// byte and char caps and sets `Doc::chars`.
pub fn read(bytes: &[u8], name: &str, caps: &Caps) -> Result<(Format, Doc), ReadError> {
    if bytes.len() > caps.max_bytes {
        return Err(ReadError::TooLarge { bytes: bytes.len() as u64, cap: caps.max_bytes });
    }
    let (format, reader) = sniff(bytes, name).ok_or(ReadError::Binary)?;
    let reader = reader.ok_or(ReadError::Unsupported(format))?;
    let mut doc = reader.to_markdown(bytes, format, caps)?;
    if let Format::Image(_) = format {
        image::with_path(name, &mut doc);
    }
    doc.pages = doc.pages.max(1);
    let mut chars = doc.markdown.chars().count();
    if chars > caps.max_chars {
        let cut = doc.markdown.char_indices().nth(caps.max_chars).map_or(doc.markdown.len(), |(i, _)| i);
        doc.markdown.truncate(cut);
        chars = caps.max_chars;
        let warning = char_cap_warning(caps.max_chars);
        if !doc.warnings.contains(&warning) {
            doc.warnings.push(warning);
        }
    }
    doc.chars = chars;
    Ok((format, doc))
}

/// The header line, the body and, when a cap or skip applies, a closing note.
pub fn render(name: &str, format: Format, doc: &Doc) -> String {
    let leaf = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let skipped = if doc.warnings.is_empty() { "none".to_owned() } else { doc.warnings.join("; ") };
    let mut out = format!(
        "<!-- vade read: {} | {} | pages {} | chars {} | skipped: {} -->\n",
        comment_safe(leaf),
        format.label(),
        doc.pages,
        doc.chars,
        comment_safe(&skipped)
    );
    out.push_str(&doc.markdown);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    if !doc.warnings.is_empty() {
        out.push_str(&format!("\n<!-- vade read: skipped: {} -->\n", comment_safe(&skipped)));
    }
    out
}

/// [`read`] then [`render`].
pub fn read_markdown(bytes: &[u8], name: &str, caps: &Caps) -> Result<String, ReadError> {
    read(bytes, name, caps).map(|(format, doc)| render(name, format, &doc))
}

/// Text that cannot end an HTML comment early: a space goes between every
/// two hyphens in a row, so no `--` (and no `-->` or `--!>`) remains.
fn comment_safe(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c == '-' && out.ends_with('-') {
            out.push(' ');
        }
        out.push(if matches!(c, '\n' | '\r') { ' ' } else { c });
    }
    out
}

/// `<!-- page N of M -->` then `## Page N`, with a blank line before and after.
pub fn page_marker(n: usize, of: usize) -> String {
    format!("\n<!-- page {n} of {of} -->\n## Page {n}\n\n")
}

/// `<!-- slide N of M -->` then `## Slide N`, with blank lines like [`page_marker`].
pub fn slide_marker(n: usize, of: usize) -> String {
    format!("\n<!-- slide {n} of {of} -->\n## Slide {n}\n\n")
}

/// `<!-- chapter N of M -->` then `## Chapter N`, with blank lines like [`page_marker`].
pub fn chapter_marker(n: usize, of: usize) -> String {
    format!("\n<!-- chapter {n} of {of} -->\n## Chapter {n}\n\n")
}

/// `<!-- sheet N of M -->` then `## Sheet <name>`, with blank lines like [`page_marker`].
pub fn sheet_marker(n: usize, of: usize, name: &str) -> String {
    format!("\n<!-- sheet {n} of {of} -->\n## Sheet {}\n\n", name.replace('\n', " "))
}

/// `<!-- file N of M -->` then `## File <path>`, for a member of an archive,
/// with blank lines like [`page_marker`].
pub fn file_marker(n: usize, of: usize, path: &str) -> String {
    format!("\n<!-- file {n} of {of} -->\n## File {}\n\n", path.replace(['\n', '\r'], " "))
}

/// The warning for a page cap: `pages after 2000 of 5000 (pages cap)`.
pub fn page_cap_warning(kept: usize, total: usize, unit: &str) -> String {
    format!("{unit} after {kept} of {total} ({unit} cap)")
}

/// A fenced block in `lang` (may be empty). The fence is longer than any
/// backtick run inside `body`.
pub fn fence(lang: &str, body: &str) -> String {
    let (mut run, mut longest) = (0usize, 0usize);
    for c in body.chars() {
        run = if c == '`' { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    let ticks = "`".repeat(longest.max(2) + 1);
    let nl = if body.ends_with('\n') { "" } else { "\n" };
    format!("{ticks}{lang}\n{body}{nl}{ticks}\n")
}

/// The length of `](<target>)` for a link, with spaces and `)` escaped as
/// the office readers write them.
fn link_close_len(target: &str) -> usize {
    3 + target.len() + 2 * target.bytes().filter(|&b| b == b' ' || b == b')').count()
}

/// One GitHub Markdown table cell: pipes escaped, line breaks as `<br>`.
pub fn cell(text: &str) -> String {
    let text = text.trim().replace('|', "\\|");
    text.replace("\r\n", "<br>").replace(['\n', '\r'], "<br>")
}

/// A GitHub Markdown table. The first row is the header; short rows are
/// padded. Cells must already be escaped with [`cell`]. Empty input gives "".
/// The result stays within the default caps; a reader uses [`table_in`] with
/// its own [`Budget`].
pub fn table(rows: &[Vec<String>]) -> String {
    table_in(rows, &mut Budget::new(&Caps::default()))
}

/// [`table`] within `budget`: the table stops at the cell where a cap is
/// reached, so padding cannot grow past the caps.
pub fn table_in(rows: &[Vec<String>], budget: &mut Budget) -> String {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = String::new();
    if width == 0 {
        return out;
    }
    let mut line = |row: Option<&Vec<String>>| table_line(&mut out, budget, width, row.map(Vec::as_slice));
    let _ = line(rows.first()) && line(None) && rows[1..].iter().all(|row| line(Some(row)));
    out
}

/// One table line of `width` cells, padded; `None` is the separator line.
/// False when the budget cut the line.
fn table_line(out: &mut String, budget: &mut Budget, width: usize, row: Option<&[String]>) -> bool {
    if !budget.push(out, "|") {
        return false;
    }
    for i in 0..width {
        let value = row.map_or("---", |row| row.get(i).map_or("", String::as_str));
        if !(budget.push(out, " ") && budget.push(out, value) && budget.push(out, " |")) {
            return false;
        }
    }
    budget.push(out, "\n")
}
