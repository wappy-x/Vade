//! The ZIP container of DOCX, XLSX, PPTX, EPUB and the OpenDocument formats
//! (PKWARE APPNOTE 6.3): the central directory, ZIP64 sizes, and stored or
//! deflated entries. Inflate is the PDF reader's (`crate::pdf::inflate`),
//! with its work budget, so a compression bomb stops at a fixed cap.
//!
//! Caps: at most [`MAX_ENTRIES`] entries, [`MAX_PART_BYTES`] decoded bytes
//! for one part and [`MAX_UNZIP_BYTES`] for all parts of one read. A part cut
//! at a cap keeps its first bytes and says so in a warning. Every entry that
//! decodes in full must match its size and its CRC-32. Encrypted entries,
//! split archives, other compression methods and broken records are errors.
//!
//! The package helpers below follow Open Packaging Conventions (OPC):
//! relationship parts and part names relative to a source part.
use super::xml::{self, Event};
use super::{Budget, Format, ReadError, scanned};
use crate::pdf::budget::{Budget as Work, Exhausted, Kind};
use crate::pdf::inflate::{self, inflate_raw};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

/// Most entries in one archive.
pub(super) const MAX_ENTRIES: usize = 65_535;
/// Most decoded bytes of one part.
pub(super) const MAX_PART_BYTES: usize = 64 << 20;
/// Most decoded bytes of all parts in one read.
pub(super) const MAX_UNZIP_BYTES: usize = 256 << 20;

/// Storage of one directory entry and its name index. The directory is
/// charged to the ZIP's own budget, not to the conversion work cap: it is
/// bounded by the file (each entry has a 46-byte record) and [`MAX_ENTRIES`].
const ENTRY: usize = 96;

/// The cancel flag of the inflate budget; a read is never cancelled.
static NEVER: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The decoded bytes that ZIP parts on this thread may still use while
    /// an archive is read ([`SharedCap`]); `None` outside an archive read.
    static SHARED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The bytes cap of one archive read, shared by its members at all levels,
/// the parts of office members included. Every ZIP part decoded on this
/// thread while it lives draws from it, also a part of a member that then
/// fails. The cap before it comes back when it drops.
pub(super) struct SharedCap(Option<usize>);

impl SharedCap {
    pub(super) fn new(bytes: usize) -> Self {
        SharedCap(SHARED.with(|s| s.replace(Some(s.get().map_or(bytes, |left| left.min(bytes))))))
    }
}

impl Drop for SharedCap {
    fn drop(&mut self) {
        SHARED.with(|s| s.set(self.0));
    }
}

/// The bytes left of the [`SharedCap`], or `usize::MAX` outside one.
pub(super) fn shared_left() -> usize {
    SHARED.with(|s| s.get().unwrap_or(usize::MAX))
}

/// Counts `n` bytes against the [`SharedCap`], if one is set.
pub(super) fn take_shared(n: usize) {
    SHARED.with(|s| s.set(s.get().map(|left| left.saturating_sub(n))));
}

struct Entry {
    name: String,
    flags: u16,
    method: u16,
    crc: u32,
    packed: usize,
    size: usize,
    offset: usize,
    /// A Unix host wrote the entry as a symbolic link.
    link: bool,
}

/// One entry of the directory, for an archive listing.
pub(super) struct Listed<'z> {
    pub(super) name: &'z str,
    /// The size the directory declares for the decoded data.
    pub(super) size: usize,
    pub(super) link: bool,
    /// Encrypted (PKWARE or WinZip AES, method 99).
    pub(super) encrypted: bool,
}

/// One decoded part. `cut` is true when a cap kept only its first bytes.
pub(super) struct Part {
    pub(super) data: Vec<u8>,
    pub(super) cut: bool,
}

pub(super) struct Zip<'a> {
    data: &'a [u8],
    format: Format,
    entries: Vec<Entry>,
    /// Part name to entry. OPC part names (DOCX, XLSX, PPTX) do not depend
    /// on case, so their keys are lowercase; EPUB and OpenDocument names do.
    index: HashMap<String, usize>,
    fold: bool,
    work: Work<'static>,
    part_cap: usize,
    total_cap: usize,
    /// One phrase for each part that a cap cut.
    pub(super) warnings: Vec<String>,
}

impl<'a> Zip<'a> {
    /// Reads the central directory of `data`.
    pub(super) fn open(data: &'a [u8], format: Format) -> Result<Self, ReadError> {
        Self::open_with(data, format, MAX_PART_BYTES, MAX_UNZIP_BYTES)
    }

    pub(super) fn open_with(data: &'a [u8], format: Format, part_cap: usize, total_cap: usize) -> Result<Self, ReadError> {
        let bad = |why: &str| ReadError::Malformed(format, why.to_owned());
        let (count, start, size) = directory(data).map_err(bad)?;
        if count > MAX_ENTRIES {
            return Err(bad(&format!("the archive has {count} entries; the limit is {MAX_ENTRIES}")));
        }
        let limits = crate::pdf::Limits { stream_bytes: total_cap, ..crate::pdf::Limits::DEFAULT };
        let fold = matches!(format, Format::Docx | Format::Xlsx | Format::Pptx);
        let mut zip = Self { data, format, entries: Vec::new(), index: HashMap::new(), fold, work: Work::new(&limits, &NEVER), part_cap, total_cap, warnings: Vec::new() };
        let end = start + size;
        let mut at = start;
        for _ in 0..count {
            let fixed = data.get(at..at + 46).filter(|_| at + 46 <= end).ok_or_else(|| bad("the central directory ends early"))?;
            if &fixed[..4] != b"PK\x01\x02" {
                return Err(bad("a central directory record has a bad signature"));
            }
            let (n, m, k) = (u16le(fixed, 28) as usize, u16le(fixed, 30) as usize, u16le(fixed, 32) as usize);
            let name = data.get(at + 46..at + 46 + n).ok_or_else(|| bad("the central directory ends early"))?;
            let extra = data.get(at + 46 + n..at + 46 + n + m).ok_or_else(|| bad("the central directory ends early"))?;
            if at + 46 + n + m + k > end {
                return Err(bad("the central directory ends early"));
            }
            scanned(46 + n + m);
            let (mut size, mut packed, mut offset) = (u32le(fixed, 24) as u64, u32le(fixed, 20) as u64, u32le(fixed, 42) as u64);
            zip64(extra, &mut size, &mut packed, &mut offset).map_err(bad)?;
            at += 46 + n + m + k;
            // A name that is not UTF-8 is not a part any reader asks for.
            let Ok(name) = std::str::from_utf8(name) else { continue };
            if zip.work.charge(Kind::Retained, ENTRY + 2 * name.len()).is_err() {
                return Err(bad("the archive directory is too large"));
            }
            let key = zip.key(name);
            if zip.index.contains_key(&key) {
                return Err(bad(&format!("two entries are named {name}")));
            }
            let fits = |v: u64| usize::try_from(v).ok().filter(|&v| v <= data.len());
            let entry = Entry {
                name: name.to_owned(),
                flags: u16le(fixed, 8),
                method: u16le(fixed, 10),
                crc: u32le(fixed, 16),
                packed: fits(packed).ok_or_else(|| bad("an entry is larger than the file"))?,
                size: usize::try_from(size).unwrap_or(usize::MAX),
                offset: fits(offset).ok_or_else(|| bad("an entry starts past the end of the file"))?,
                // A Unix host (3) keeps the file mode in the high half of the
                // external attributes; 0o120000 is a symbolic link.
                link: u16le(fixed, 4) >> 8 == 3 && (u32le(fixed, 38) >> 16) & 0o170000 == 0o120000,
            };
            zip.index.insert(key, zip.entries.len());
            zip.entries.push(entry);
        }
        Ok(zip)
    }

    /// The entries in directory order. Names that are not UTF-8 are left out.
    pub(super) fn list(&self) -> impl Iterator<Item = Listed<'_>> {
        self.entries.iter().map(|e| Listed { name: &e.name, size: e.size, link: e.link, encrypted: e.flags & 1 != 0 || e.method == 99 })
    }

    /// The index key of the part name `name`.
    fn key(&self, name: &str) -> String {
        if self.fold { name.to_ascii_lowercase() } else { name.to_owned() }
    }

    /// The decoded bytes of the entry `name`, or `None` when there is none.
    pub(super) fn part(&mut self, name: &str) -> Result<Option<Part>, ReadError> {
        let Some(&i) = self.index.get(&self.key(name)) else { return Ok(None) };
        let format = self.format;
        let e = &self.entries[i];
        let bad = |why: String| ReadError::Malformed(format, why);
        if e.flags & 1 != 0 {
            return Err(ReadError::Encrypted(format));
        }
        let head = self.data.get(e.offset..e.offset + 30).ok_or_else(|| bad(format!("the entry {} starts past the end of the file", e.name)))?;
        if &head[..4] != b"PK\x03\x04" {
            return Err(bad(format!("the entry {} has a bad local header", e.name)));
        }
        let start = e.offset + 30 + u16le(head, 26) as usize + u16le(head, 28) as usize;
        let raw = self.data.get(start..start + e.packed).ok_or_else(|| bad(format!("the entry {} ends past the end of the file", e.name)))?;
        scanned(raw.len());
        // A part with more bytes than it declares is damaged; one with more
        // than the caps keeps the bytes up to the cap.
        let room = self.part_cap.min(MAX_UNZIP_BYTES).min(shared_left());
        let cap = e.size.min(room);
        let mut data = Vec::new();
        let mut cut = false;
        match e.method {
            0 => {
                if raw.len() != e.size {
                    return Err(bad(format!("the stored entry {} has the wrong size", e.name)));
                }
                // A stored part keeps the bytes that the read's total cap has left.
                let left = self.total_cap.saturating_sub(self.work.used(Kind::Decoded)).min(shared_left());
                let n = cap.min(raw.len()).min(left);
                // After an earlier stop the charge fails and the part keeps nothing.
                if self.work.charge(Kind::Decoded, n).is_ok() {
                    data.extend_from_slice(&raw[..n]);
                }
                cut = data.len() < raw.len();
                take_shared(data.len());
            }
            8 => {
                let result = inflate_raw(raw, cap, &mut data, &self.work);
                // The bytes decoded count against a shared cap also when the part fails.
                take_shared(data.len());
                match result {
                    Ok(_) => {}
                    Err(inflate::Error::Cap) if e.size > cap => cut = true,
                    Err(inflate::Error::Cap) => return Err(bad(format!("the entry {} holds more data than its size", e.name))),
                    Err(inflate::Error::Budget(Exhausted::Limit(_))) => cut = true,
                    Err(inflate::Error::Budget(_)) => return Err(bad(format!("the entry {} could not be decoded", e.name))),
                    Err(inflate::Error::Data(why)) => return Err(bad(format!("the entry {}: {why}", e.name))),
                }
            }
            m => return Err(bad(format!("the entry {} uses compression method {m}, which Vade does not read", e.name))),
        }
        if cut {
            self.warnings.push(format!("{} after {} bytes (unzip cap)", e.name, data.len()));
        } else if data.len() != e.size {
            return Err(bad(format!("the entry {} has the wrong size", e.name)));
        } else if crc32(&data) != e.crc {
            return Err(bad(format!("the entry {} fails its CRC-32 check", e.name)));
        }
        Ok(Some(Part { data, cut }))
    }

    /// The part `name` as text, or `None` when there is none.
    pub(super) fn text(&mut self, name: &str) -> Result<Option<(String, bool)>, ReadError> {
        let format = self.format;
        let Some(part) = self.part(name)? else { return Ok(None) };
        xml_text(part.data, part.cut).map(|t| Some((t, part.cut))).map_err(|why| ReadError::Malformed(format, format!("{name} {why}")))
    }

    /// The relationships of the part `source` ([`rels`]).
    pub(super) fn rels(&mut self, source: &str, budget: &mut Budget) -> Result<Rels, ReadError> {
        let (dir, leaf) = source.rsplit_once('/').unwrap_or(("", source));
        let path = if dir.is_empty() { format!("_rels/{leaf}.rels") } else { format!("{dir}/_rels/{leaf}.rels") };
        let format = self.format;
        let Some((text, cut)) = self.text(&path)? else { return Ok(Rels::default()) };
        rels(&text, cut, dir, budget).map_err(|why| ReadError::Malformed(format, format!("{path}: {why}")))
    }
}

/// One OPC relationship: the last segment of the type (for example
/// `image` or `officeDocument`) and the target. An internal target is a
/// resolved part name; an external one is kept as written.
pub(super) struct Rel {
    pub(super) kind: String,
    pub(super) target: String,
    pub(super) external: bool,
}

/// The relationships of one part, with an index by id. The first of two
/// relationships with the same id wins.
#[derive(Default)]
pub(super) struct Rels {
    list: Vec<Rel>,
    by_id: HashMap<String, usize>,
    /// Work bytes that the relationships hold.
    pub(super) held: usize,
    /// The work cap stopped the list: a missing id may be after the stop.
    pub(super) stopped: bool,
}

impl Rels {
    /// The relationship with `id`.
    pub(super) fn get(&self, id: &str) -> Option<&Rel> {
        self.by_id.get(id).map(|&i| &self.list[i])
    }

    /// The target of the first internal relationship of `kind`.
    pub(super) fn first(&self, kind: &str) -> Option<&str> {
        self.list.iter().find(|r| r.kind == kind && !r.external).map(|r| r.target.as_str())
    }
}

/// The relationships in `text`, a `.rels` part of a source part in `dir`.
/// Each one counts against the work cap before it is stored.
fn rels(text: &str, cut: bool, dir: &str, budget: &mut Budget) -> Result<Rels, String> {
    let mut out = Rels::default();
    let mut scan = xml::Scanner::new(text, cut);
    while let Some(event) = scan.next()? {
        let Event::Open(tag) = event else { continue };
        if tag.local() != "Relationship" {
            continue;
        }
        // A part target may be as long as a ZIP entry name; a longer web
        // address, or a longer target of a part other than the main document,
        // is left out at the attribute cap.
        let external = tag.attr("TargetMode")?.is_some_and(|m| m == "External");
        let Some(kind) = tag.attr("Type")? else { continue };
        let kind = kind.rsplit('/').next().unwrap_or("");
        let target = match (external, kind) {
            (true, _) => tag.attr("Target")?,
            (false, "officeDocument") => tag.path("Target")?,
            (false, _) => tag.optional_path("Target")?,
        };
        let (Some(id), Some(target)) = (tag.attr("Id")?, target) else { continue };
        // A resolved target is never longer than the directory and the target.
        let size = 64 + id.len() + kind.len() + dir.len() + 1 + target.len();
        if !budget.reserve(size) {
            out.stopped = true;
            break;
        }
        out.held += size;
        let target = if external { target.into_owned() } else { resolve(dir, &target) };
        out.by_id.entry(id.into_owned()).or_insert(out.list.len());
        out.list.push(Rel { kind: kind.to_owned(), target, external });
    }
    scan.finish(budget)?;
    Ok(out)
}

/// The main part of an Office Open XML package, from its root relationships.
pub(super) fn main_part(zip: &mut Zip, budget: &mut Budget) -> Result<Option<String>, ReadError> {
    Ok(zip.rels("", budget)?.first("officeDocument").map(str::to_owned))
}

/// The part name that `target` names from a part in `dir`: `/` starts at
/// the package root, `.` and `..` segments resolve, and `%XX` escapes decode.
/// The result is never longer than `dir` and `target` with one `/`, and no
/// other storage grows with the number of segments.
pub(super) fn resolve(dir: &str, target: &str) -> String {
    let target = target.split(['#', '?']).next().unwrap_or("");
    let mut out = String::new();
    let base = if target.starts_with('/') { "" } else { dir };
    for seg in base.split('/').chain(target.split('/')) {
        match seg {
            "" | "." => {}
            ".." => out.truncate(out.rfind('/').unwrap_or(0)),
            s => {
                if !out.is_empty() {
                    out.push('/');
                }
                out.push_str(s);
            }
        }
    }
    percent_decode(&out)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_owned())
}

/// The text of an XML part: UTF-8 (a byte order mark is dropped) or UTF-16
/// with a byte order mark. A part cut at a cap drops a character it cuts.
pub(super) fn xml_text(data: Vec<u8>, cut: bool) -> Result<String, String> {
    let utf16 = |be: bool| {
        let units = data[2..].chunks(2).filter(|c| c.len() == 2).map(|c| if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) });
        let mut out = String::new();
        for c in char::decode_utf16(units) {
            match c {
                Ok(c) => out.push(c),
                Err(_) if cut => break,
                Err(_) => return Err("is not valid UTF-16 text".to_owned()),
            }
        }
        Ok(out)
    };
    if data.starts_with(b"\xff\xfe") {
        return utf16(false);
    }
    if data.starts_with(b"\xfe\xff") {
        return utf16(true);
    }
    let skip = if data.starts_with(b"\xef\xbb\xbf") { 3 } else { 0 };
    let mut data = data;
    data.drain(..skip);
    match String::from_utf8(data) {
        Ok(text) => Ok(text),
        Err(e) if cut && e.utf8_error().error_len().is_none() => {
            let valid = e.utf8_error().valid_up_to();
            let mut data = e.into_bytes();
            data.truncate(valid);
            Ok(String::from_utf8(data).unwrap_or_default())
        }
        Err(_) => Err("is not UTF-8 text".to_owned()),
    }
}

/// The entry count, start and size of the central directory, from the end
/// of central directory record (and the ZIP64 one when its fields overflow).
/// A comment may hold the record's signature. The record is the candidate
/// whose directory checks out, ranked first by a comment that ends at the
/// end of the file, then by a directory with entries; among equals, the
/// last in the file.
fn directory(data: &[u8]) -> Result<(usize, usize, usize), &'static str> {
    // The record is 22 bytes plus a comment of at most 65,535 bytes.
    let from = data.len().saturating_sub(22 + 65_535);
    let tail = &data[from..];
    scanned(tail.len());
    let mut first = None;
    let mut best: Option<((bool, bool), (usize, usize, usize))> = None;
    // Directory records that the checks of all candidates may walk; each
    // record position is walked once, so false records that name the same
    // directory, or a start inside it, cost nothing more.
    let mut steps = 4 * MAX_ENTRIES;
    let mut walked = HashMap::new();
    for i in (0..tail.len().saturating_sub(21)).rev() {
        let end = i + 22 + u16le(tail, i + 20) as usize;
        if &tail[i..i + 4] != b"PK\x05\x06" || end > tail.len() {
            continue;
        }
        match end_record(data, from + i, &mut steps, &mut walked) {
            Ok(found) => {
                let rank = (end == tail.len(), found.0 > 0);
                if best.is_none_or(|(r, _)| rank > r) {
                    best = Some((rank, found));
                }
                if rank == (true, true) {
                    break;
                }
            }
            Err(why) => {
                first.get_or_insert(why);
            }
        }
    }
    best.map(|(_, found)| found).ok_or(first.unwrap_or("no ZIP end of central directory record"))
}

/// The directory that the end record at `at` gives, when it checks out: its
/// records fill the directory exactly. `steps` limits the records walked, and
/// `walked` keeps the run found from each record position (see [`record_run`]).
fn end_record(data: &[u8], at: usize, steps: &mut usize, walked: &mut Walked) -> Result<(usize, usize, usize), &'static str> {
    let r = &data[at..at + 22];
    if u16le(r, 4) != 0 || u16le(r, 6) != 0 {
        return Err("split ZIP archives are not supported");
    }
    let (mut count, mut size, mut start) = (u16le(r, 10) as u64, u32le(r, 12) as u64, u32le(r, 16) as u64);
    // The central directory ends where the (ZIP64) end record starts.
    let mut end = at;
    if count == 0xffff || size == 0xffff_ffff || start == 0xffff_ffff {
        let loc = at.checked_sub(20).filter(|&l| &data[l..l + 4] == b"PK\x06\x07").ok_or("the ZIP64 locator is missing")?;
        let o = usize::try_from(u64le(data, loc + 8)).ok().filter(|&o| o <= loc).ok_or("the ZIP64 record is out of the file")?;
        let rec = data.get(o..o + 56).ok_or("the ZIP64 record is out of the file")?;
        if &rec[..4] != b"PK\x06\x06" {
            return Err("the ZIP64 record has a bad signature");
        }
        (count, size, start, end) = (u64le(rec, 32), u64le(rec, 40), u64le(rec, 48), o);
    }
    let count = usize::try_from(count).map_err(|_| "too many entries")?;
    let start = usize::try_from(start).ok().filter(|&s| s <= end).ok_or("the central directory starts past the end of the file")?;
    let size = usize::try_from(size).ok().filter(|&s| s <= end - start).ok_or("the central directory ends past the end of the file")?;
    // The directory ends where the (ZIP64) end record starts, and its records
    // fill it: the run of records stops at that record, which is not a
    // directory record. Past the entry limit it is not walked: the archive is
    // refused.
    let found = start + size == end
        && match count {
            0 => true,
            n if n > MAX_ENTRIES => true,
            n => record_run(data, start, steps, walked) == Some((end, n)),
        };
    if !found {
        return Err("the central directory is not where the end record says");
    }
    Ok((count, start, size))
}

/// For each central directory record position walked, the run of records
/// from it (see [`record_run`]).
type Walked = HashMap<usize, Option<(usize, usize)>>;

/// The run of central directory records that starts at `start`: where it
/// stops (the first position that holds no record) and how many records it
/// has. `None` for a run of more than [`MAX_ENTRIES`] records, a record that
/// ends past the end of the file, or a walk that `steps` stopped. Each new
/// record position takes one of `steps`; `walked` keeps the run of each, so
/// runs from different starts in one directory join the walk already stored.
fn record_run(data: &[u8], start: usize, steps: &mut usize, walked: &mut Walked) -> Option<(usize, usize)> {
    let mut path = Vec::new();
    let mut at = start;
    let mut run = loop {
        if let Some(&known) = walked.get(&at) {
            break known;
        }
        let Some(fixed) = data.get(at..at + 46).filter(|f| &f[..4] == b"PK\x01\x02") else { break Some((at, 0)) };
        let Some(left) = steps.checked_sub(1) else { break None };
        *steps = left;
        path.push(at);
        at += 46 + u16le(fixed, 28) as usize + u16le(fixed, 30) as usize + u16le(fixed, 32) as usize;
        if at > data.len() {
            break None;
        }
    };
    for &p in path.iter().rev() {
        run = run.map(|(stop, n)| (stop, n + 1)).filter(|&(_, n)| n <= MAX_ENTRIES);
        walked.insert(p, run);
    }
    run
}

/// Sizes and offset from a ZIP64 extra field, for the fields set to all ones.
fn zip64(extra: &[u8], size: &mut u64, packed: &mut u64, offset: &mut u64) -> Result<(), &'static str> {
    let mut at = 0;
    while at + 4 <= extra.len() {
        let (id, len) = (u16le(extra, at), u16le(extra, at + 2) as usize);
        let body = extra.get(at + 4..at + 4 + len).ok_or("an extra field ends early")?;
        if id == 1 {
            let mut field = body.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap()));
            for v in [&mut *size, &mut *packed, &mut *offset] {
                if *v == 0xffff_ffff {
                    *v = field.next().ok_or("a ZIP64 field is missing")?;
                }
            }
        }
        at += 4 + len;
    }
    Ok(())
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64le(b: &[u8], at: usize) -> u64 {
    b.get(at..at + 8).map_or(u64::MAX, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

pub(super) fn crc32(data: &[u8]) -> u32 {
    !data.iter().fold(!0u32, |c, &b| CRC_TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8))
}

/// Drops a UTF-8 character that a cap cut at the end of `data`, so that a
/// cut UTF-8 part still decodes as UTF-8.
pub(super) fn drop_cut_char(data: &mut Vec<u8>) {
    if let Err(e) = std::str::from_utf8(data)
        && e.error_len().is_none()
    {
        data.truncate(e.valid_up_to());
    }
}

/// The `mimetype` entry of an EPUB or OpenDocument file: the first entry,
/// stored, named `mimetype`, read from the local header at the start.
pub(super) fn mimetype(bytes: &[u8]) -> Option<&str> {
    if !bytes.starts_with(b"PK\x03\x04") || bytes.get(30..38)? != b"mimetype" || u16le(bytes, 8) != 0 {
        return None;
    }
    let size = (u32le(bytes, 18) as usize).min(100);
    let start = 38 + u16le(bytes, 28) as usize;
    std::str::from_utf8(bytes.get(start..start + size)?).ok().map(str::trim)
}
