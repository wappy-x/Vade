//! Archives: ZIP, tar and gzip (`.tar.gz`, `.tgz` or one compressed file).
//! The reader lists the entries with their sizes as a Markdown table, then
//! reads each file with the reader for its format, after a file marker.
//!
//! The members share the caps of the read: one [`Budget`] for the text, the
//! pages cap for the files read from one archive, and `Caps::max_bytes` for
//! the member bytes and the decoded ZIP data of all levels together, the
//! parts of office members included ([`SharedCap`]), so a compression bomb
//! stops at the size of a file that Vade would read. An archive in an archive is read
//! to [`MAX_DEPTH`] levels; a deeper one is listed but not opened. Links are
//! listed and never followed. Encrypted members are listed and not read.
use super::zip::{SharedCap, Zip, crc32, shared_left, take_shared};
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, page_cap_warning, scanned, table_in};
use crate::pdf::budget::{Budget as Work, Exhausted};
use crate::pdf::inflate::{self, inflate_raw};
use std::borrow::Cow;
use std::sync::atomic::AtomicBool;

pub(super) struct ArchiveReader;

pub(super) const ZIP: Format = Format::Archive("ZIP archive");
pub(super) const TAR: Format = Format::Archive("tar archive");
pub(super) const GZIP: Format = Format::Archive("gzip archive");

/// Archive levels that are opened: the file and one archive inside it.
pub const MAX_DEPTH: usize = 2;
/// Most entries listed for one tar archive, as for ZIP.
const MAX_ENTRIES: usize = super::zip::MAX_ENTRIES;
/// Longest tar path kept from a GNU long name or a pax record, the ZIP name
/// limit; the rest is left out.
const MAX_NAME: usize = 65_535;

/// The cancel flag of the inflate budget; a read is never cancelled.
static NEVER: AtomicBool = AtomicBool::new(false);

impl Reader for ArchiveReader {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format> {
        if name.is_empty() {
            return if bytes.starts_with(b"\x1f\x8b\x08") {
                Some(GZIP)
            } else if is_tar(bytes) {
                Some(TAR)
            } else {
                None
            };
        }
        // An old tar file has no magic bytes; its name decides.
        (extension(name) == "tar").then_some(TAR)
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let _shared = SharedCap::new(caps.max_bytes);
        let mut read = State { caps: *caps, budget: Budget::new(caps), warnings: Vec::new() };
        let (markdown, pages) = read.archive(bytes, format, 1, "")?;
        let mut doc = Doc::new(markdown, pages);
        doc.warnings = read.warnings;
        doc.warnings.extend(read.budget.warning());
        Ok(doc)
    }
}

/// What a listing says an entry is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    File,
    Folder,
    Link,
    HardLink,
    Encrypted,
    Other,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::File => "file",
            Kind::Folder => "folder",
            Kind::Link => "symbolic link (not followed)",
            Kind::HardLink => "hard link (not followed)",
            Kind::Encrypted => "encrypted (not read)",
            Kind::Other => "special file (not read)",
        }
    }
}

/// One entry: its path, its size and kind, and for tar its bytes.
struct Entry<'a> {
    path: String,
    size: usize,
    kind: Kind,
    data: Option<&'a [u8]>,
}

/// The state of one read, shared by the archives at all levels. The bytes
/// left of `Caps::max_bytes` are in [`SharedCap`], which ZIP parts draw from.
struct State {
    caps: Caps,
    budget: Budget,
    warnings: Vec<String>,
}

impl State {
    /// The Markdown of the archive `bytes` at `depth` and its file count.
    /// `outer` is the path of the archive in its parent, for member names.
    fn archive(&mut self, bytes: &[u8], format: Format, depth: usize, outer: &str) -> Result<(String, usize), ReadError> {
        match format {
            ZIP => {
                // Its parts draw from the shared cap as they decode.
                let mut zip = Zip::open(bytes, ZIP)?;
                let entries: Vec<Entry> = zip
                    .list()
                    .map(|e| {
                        let kind = if e.encrypted {
                            Kind::Encrypted
                        } else if e.link {
                            Kind::Link
                        } else if e.name.ends_with('/') {
                            Kind::Folder
                        } else {
                            Kind::File
                        };
                        Entry { path: e.name.to_owned(), size: e.size, kind, data: None }
                    })
                    .collect();
                let mut out = self.listing(&entries, outer);
                let pages = self.members(&entries, depth, outer, &mut out, |name| match zip.part(name) {
                    Ok(Some(part)) if !part.cut => Ok(part.data),
                    Ok(Some(_)) => Err(None),
                    Ok(None) => Err(Some("the entry is not in the archive".to_owned())),
                    Err(e) => Err(Some(e.to_string())),
                });
                Ok((out, pages))
            }
            TAR => {
                let entries = tar(bytes, false).map_err(|why| ReadError::Malformed(TAR, why))?;
                let mut out = self.listing(&entries, outer);
                let pages = self.members(&entries, depth, outer, &mut out, in_memory);
                Ok((out, pages))
            }
            GZIP => {
                // The decoded data is at most the bytes left and counts
                // against them, also when its check fails.
                let (name, data, cut) = gunzip(bytes, shared_left()).map_err(|why| ReadError::Malformed(GZIP, why))?;
                if cut {
                    let named = path_of(outer, name.as_deref().unwrap_or("the gzip data"));
                    self.warnings.push(format!("{named} after {} bytes (archive bytes cap)", data.len()));
                }
                // A tar file inside (a good first header checksum, as an old
                // tar file has no magic) is the archive; any other file is
                // its one member.
                if checksum_ok(&data) || name.as_deref().is_some_and(|n| extension(n) == "tar") {
                    let entries = tar(&data, cut).map_err(|why| ReadError::Malformed(GZIP, why))?;
                    let mut out = self.listing(&entries, outer);
                    let pages = self.members(&entries, depth, outer, &mut out, in_memory);
                    return Ok((out, pages));
                }
                let entries = [Entry { path: name.unwrap_or_else(|| "(no name)".to_owned()), size: data.len(), kind: Kind::File, data: Some(&data) }];
                let mut out = self.listing(&entries, outer);
                let pages = self.members(&entries, depth, outer, &mut out, in_memory);
                Ok((out, pages))
            }
            other => Err(ReadError::Unsupported(other)),
        }
    }

    /// The summary line and the table of entries.
    fn listing(&mut self, entries: &[Entry], outer: &str) -> String {
        let total = entries.iter().fold(0usize, |sum, e| sum.saturating_add(e.size));
        let mut out = String::new();
        let n = entries.len();
        let line = format!("{n} {}, {total} bytes when unpacked.\n\n", if n == 1 { "entry" } else { "entries" });
        self.budget.push(&mut out, &line);
        if entries.is_empty() {
            return out;
        }
        let mut rows = vec![vec!["#".to_owned(), "Path".to_owned(), "Size (bytes)".to_owned(), "Type".to_owned()]];
        // Rows past the chars left never show, so they are not built. The
        // storage of the rows is given back after the table.
        let (left, mut shown, mut held) = (self.budget.max_chars - self.budget.chars, 0usize, 0);
        for (n, e) in entries.iter().enumerate() {
            let size = 4 * size_of::<String>() + e.path.len() + outer.len();
            if shown > left || !self.budget.reserve(size) {
                break;
            }
            held += size;
            let row = vec![(n + 1).to_string(), cell(&path_of(outer, &e.path)), e.size.to_string(), e.kind.label().to_owned()];
            shown += row.iter().map(|c| c.chars().count()).sum::<usize>();
            rows.push(row);
        }
        let table = table_in(&rows, &mut self.budget);
        self.budget.release(held);
        out.push_str(&table);
        out
    }

    /// Reads the file entries after the listing, each after a file marker,
    /// and gives the number of files read. `open` decodes a ZIP member (its
    /// part draws from the shared cap): its bytes, or `None` for the bytes
    /// cap, or the reason it failed. Tar members are slices of bytes in
    /// memory, which the file size cap or the decoder that made them counted.
    fn members(&mut self, entries: &[Entry], depth: usize, outer: &str, out: &mut String, mut open: impl FnMut(&str) -> Result<Vec<u8>, Option<String>>) -> usize {
        let files: Vec<&Entry> = entries.iter().filter(|e| e.kind == Kind::File).collect();
        let total = files.len();
        let mut read = 0;
        for (n, e) in files.iter().enumerate() {
            if n == self.caps.max_pages {
                self.warnings.push(page_cap_warning(self.caps.max_pages, total, "files"));
                break;
            }
            // A file that filled the char cap exactly stops the read too.
            if self.budget.text_counts && self.budget.chars == self.budget.max_chars {
                self.budget.hit.get_or_insert(super::Stop::Chars);
            }
            if self.budget.full() {
                break;
            }
            let path = path_of(outer, &e.path);
            let marker = super::file_marker(n + 1, total, &path);
            let marker = if out.is_empty() { marker.trim_start() } else { &marker };
            if !self.budget.fits(marker.len()) {
                self.budget.cut();
                break;
            }
            out.push_str(self.budget.work(marker));
            read += 1;
            let left = shared_left();
            let (body, counted) = if e.data.is_none() && e.size > left {
                self.warnings.push(format!("{path} ({} bytes; archive bytes cap)", e.size));
                (format!("(not read: the member has {} bytes, more than the {left} bytes left of the archive bytes cap)", e.size), false)
            } else {
                let bytes = match e.data {
                    Some(data) => Ok(Cow::Borrowed(data)),
                    None => open(&e.path).map(Cow::Owned),
                };
                match bytes {
                    Ok(bytes) => self.member(&bytes, &path, depth),
                    Err(None) => {
                        self.warnings.push(format!("{path} (archive bytes cap)"));
                        ("(not read: the member is larger than the archive bytes cap)".to_owned(), false)
                    }
                    Err(Some(why)) => (format!("(not read: {why})"), false),
                }
            };
            if counted {
                // A nested archive's text was counted once as it was built;
                // it moves into this one as work.
                self.budget.release(body.len());
                out.push_str(self.budget.work(&body));
            } else {
                self.budget.push(out, &body);
            }
            if !out.ends_with('\n') {
                self.budget.push(out, "\n");
            }
        }
        read
    }

    /// The Markdown of one member file, or one line on why it was not read,
    /// and true when the budget already counted the text (a nested archive).
    fn member(&mut self, bytes: &[u8], path: &str, depth: usize) -> (String, bool) {
        scanned(bytes.len());
        if let Some((format @ Format::Archive(_), _)) = super::sniff(bytes, path) {
            if depth >= MAX_DEPTH {
                self.warnings.push(format!("{path} (archive past {MAX_DEPTH} levels; depth cap)"));
                return (format!("(not opened: an archive inside an archive is read {MAX_DEPTH} levels deep)"), false);
            }
            return match self.archive(bytes, format, depth + 1, path) {
                Ok((markdown, _)) => (markdown, true),
                Err(e) => (format!("(not read: {e})"), false),
            };
        }
        let chars = self.budget.max_chars - self.budget.chars;
        let caps = Caps { max_chars: chars.max(1), ..self.caps };
        match super::read(bytes, path, &caps) {
            Ok((_, doc)) => {
                self.warnings.extend(doc.warnings.iter().map(|w| format!("{path}: {w}")));
                (if doc.markdown.trim().is_empty() { "(no text in this file)".to_owned() } else { doc.markdown }, false)
            }
            Err(e) => (format!("(not read: {e})"), false),
        }
    }
}

/// The `open` of [`State::members`] for tar entries, whose bytes are in memory.
fn in_memory(_: &str) -> Result<Vec<u8>, Option<String>> {
    Err(Some("the entry has no data".to_owned()))
}

/// The path of a member as shown: `outer/path` inside a nested archive.
fn path_of(outer: &str, path: &str) -> String {
    match (outer.is_empty(), path.is_empty()) {
        (true, _) => path.to_owned(),
        (false, true) => outer.to_owned(),
        (false, false) => format!("{outer}/{path}"),
    }
}

/// The entries of a tar archive (POSIX ustar, pax `path` records and GNU
/// long names). A file that a cap `cut` may end inside an entry; the entries
/// before it stay.
fn tar(data: &[u8], cut: bool) -> Result<Vec<Entry<'_>>, String> {
    let mut entries = Vec::new();
    let (mut at, mut long_name) = (0usize, None::<String>);
    // The file may end after an entry, also inside its padding.
    while at < data.len() {
        let Some(h) = data.get(at..at + 512) else {
            if cut {
                break;
            }
            return Err(format!("the file ends inside the header at byte {at}"));
        };
        scanned(512);
        if h.iter().all(|&b| b == 0) {
            break;
        }
        if !checksum_ok(h) {
            return Err(format!("the header at byte {at} fails its checksum"));
        }
        let size = number(&h[124..136]).and_then(|n| usize::try_from(n).ok()).ok_or_else(|| format!("the header at byte {at} has a bad size"))?;
        let start = at + 512;
        let Some(body) = start.checked_add(size).and_then(|end| data.get(start..end)) else {
            if cut {
                break;
            }
            return Err(format!("the entry at byte {at} ends past the end of the file"));
        };
        at = start + size.div_ceil(512) * 512;
        let kind = match h[156] {
            // A GNU long name, or a pax header with a `path` record, names the next entry.
            b'L' => {
                long_name = Some(field(&body[..body.len().min(MAX_NAME)]));
                continue;
            }
            b'x' => {
                long_name = pax_path(body).or(long_name);
                continue;
            }
            b'g' | b'K' => continue,
            b'0' | b'\0' | b'7' => Kind::File,
            b'1' => Kind::HardLink,
            b'2' => Kind::Link,
            b'5' => Kind::Folder,
            _ => Kind::Other,
        };
        let path = long_name.take().unwrap_or_else(|| {
            let (name, prefix) = (field(&h[..100]), if &h[257..262] == b"ustar" { field(&h[345..500]) } else { String::new() });
            if prefix.is_empty() { name } else { format!("{prefix}/{name}") }
        });
        if entries.len() == MAX_ENTRIES {
            return Err(format!("the archive has more than {MAX_ENTRIES} entries; the limit is {MAX_ENTRIES}"));
        }
        let data = (kind == Kind::File).then_some(body);
        entries.push(Entry { path, size, kind, data });
    }
    if long_name.is_some() && !cut {
        return Err("the file ends after a long name with no entry".into());
    }
    Ok(entries)
}

/// True for the start of a ustar archive: the magic and a good checksum, so
/// a text file with `ustar` at byte 257 is not a tar archive.
fn is_tar(bytes: &[u8]) -> bool {
    bytes.get(257..262) == Some(b"ustar") && checksum_ok(&bytes[..512.min(bytes.len())])
}

/// True when the header `h` (512 bytes) matches its checksum: the byte sum
/// with the checksum field read as spaces.
fn checksum_ok(h: &[u8]) -> bool {
    if h.len() < 512 {
        return false;
    }
    let sum: u32 = h[..512].iter().enumerate().map(|(i, &b)| if (148..156).contains(&i) { 32 } else { u32::from(b) }).sum();
    number(&h[148..156]) == Some(u64::from(sum))
}

/// A tar text field: the bytes up to the first NUL, as UTF-8 (lossy).
fn field(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// A tar number: octal digits between spaces or NULs, or base-256 when the
/// first bit is set (GNU, for sizes of 8 GiB and more).
fn number(bytes: &[u8]) -> Option<u64> {
    if bytes.first().is_some_and(|&b| b & 0x80 != 0) {
        return bytes[1..].iter().try_fold(u64::from(bytes[0] & 0x7f), |n, &b| n.checked_mul(256).map(|n| n | u64::from(b)));
    }
    let digits: Vec<u8> = bytes.iter().copied().skip_while(|&b| b == b' ').take_while(|&b| (b'0'..=b'7').contains(&b)).collect();
    if digits.is_empty() {
        return None;
    }
    digits.iter().try_fold(0u64, |n, &d| n.checked_mul(8).map(|n| n + u64::from(d - b'0')))
}

/// The `path` value of pax extended header records (`<len> path=<value>\n`).
fn pax_path(body: &[u8]) -> Option<String> {
    let mut rest = body;
    let mut found = None;
    while let Some(space) = rest.iter().position(|&b| b == b' ') {
        let len: usize = std::str::from_utf8(&rest[..space]).ok()?.parse().ok()?;
        let record = rest.get(space + 1..len)?;
        if let Some(value) = record.strip_prefix(b"path=") {
            let value = value.strip_suffix(b"\n").unwrap_or(value);
            found = Some(String::from_utf8_lossy(&value[..value.len().min(MAX_NAME)]).into_owned());
        }
        rest = &rest[len..];
    }
    found
}

/// The file name in the header, the data and whether `cap` cut it, from the
/// first member of gzip data (RFC 1952). Later members are not read. The
/// decoded bytes count against the shared cap ([`take_shared`]).
fn gunzip(data: &[u8], cap: usize) -> Result<(Option<String>, Vec<u8>, bool), String> {
    if data.len() < 18 || !data.starts_with(b"\x1f\x8b\x08") {
        return Err("the file is not gzip data".into());
    }
    let flags = data[3];
    if flags & 0xe0 != 0 {
        return Err("the header has reserved flags".into());
    }
    let mut at = 10;
    let ends = || "the header ends early".to_owned();
    if flags & 4 != 0 {
        let extra = data.get(at..at + 2).ok_or_else(ends)?;
        at += 2 + usize::from(u16::from_le_bytes([extra[0], extra[1]]));
    }
    let mut name = None;
    for (flag, keep) in [(8, true), (16, false)] {
        if flags & flag != 0 {
            let len = data.get(at..).and_then(|d| d.iter().position(|&b| b == 0)).ok_or_else(ends)?;
            if keep {
                // The name is Latin-1; only its last part is kept.
                let full: String = data[at..at + len].iter().map(|&b| char::from(b)).collect();
                name = full.rsplit(['/', '\\']).next().map(str::to_owned).filter(|n| !n.is_empty());
            }
            at += len + 1;
        }
    }
    if flags & 2 != 0 {
        at += 2;
    }
    let raw = data.get(at..).filter(|r| !r.is_empty()).ok_or_else(ends)?;
    scanned(raw.len());
    let limits = crate::pdf::Limits { stream_bytes: cap, ..crate::pdf::Limits::DEFAULT };
    let work = Work::new(&limits, &NEVER);
    let mut out = Vec::new();
    let result = inflate_raw(raw, cap, &mut out, &work);
    take_shared(out.len());
    match result {
        Ok(used) => {
            let tail = raw.get(used..used + 8).ok_or("the data ends before the gzip trailer")?;
            let (crc, size) = (u32::from_le_bytes(tail[..4].try_into().unwrap()), u32::from_le_bytes(tail[4..].try_into().unwrap()));
            if crc32(&out) != crc || out.len() as u32 != size {
                return Err("the data fails its CRC-32 or size check".into());
            }
            Ok((name, out, false))
        }
        Err(inflate::Error::Cap | inflate::Error::Budget(Exhausted::Limit(_))) => Ok((name, out, true)),
        Err(inflate::Error::Budget(_)) => Err("the data could not be decoded".into()),
        Err(inflate::Error::Data(why)) => Err(why.into()),
    }
}
