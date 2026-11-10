//! DOCX (WordprocessingML) to Markdown: paragraphs, headings (from the
//! style or the outline level), bullet and numbered lists, tables, links,
//! line breaks and page markers. Images are listed by alt text and part
//! name; their pixels are not read.
//!
//! Pages: a DOCX file has no fixed pages. When Word saved the file, it
//! marked where each page began (`w:lastRenderedPageBreak`), and those
//! marks give the pages. A file with no marks (written by another program)
//! breaks pages at hard page breaks, `pageBreakBefore` and section breaks.
//! A break inside a table takes effect after the table.
//!
//! Skipped: tracked deletions, field codes, the fallback copy of
//! alternate content, headers, footers, footnotes, comments and formatting.
//! Styles, list levels and links count against the work cap.
use super::xml::{self, Event, Scanner, Tag};
use super::zip::{self, Rels};
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, page_cap_warning, page_marker, table_in};
use std::collections::HashMap;

pub(super) struct DocxReader;

impl Reader for DocxReader {
    fn sniff(&self, _: &[u8], name: &str) -> Option<Format> {
        matches!(extension(name).as_str(), "docx" | "docm" | "dotx" | "dotm").then_some(Format::Docx)
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let mut budget = Budget::new(caps);
        let mut zip = super::office_zip(bytes, format)?;
        let main = zip::main_part(&mut zip, &mut budget)?.unwrap_or_else(|| "word/document.xml".to_owned());
        let rels = zip.rels(&main, &mut budget)?;
        let part = |kind: &str| rels.first(kind).map(str::to_owned);
        let bad = |name: &str, why: String| ReadError::Malformed(format, format!("{name} {why}"));
        let mut styles = Styles::default();
        if let Some(name) = part("styles")
            && let Some((text, cut)) = zip.text(&name)?
        {
            styles.read(&text, cut, &mut budget).map_err(|why| bad(&name, why))?;
        }
        let mut lists = Lists::default();
        if let Some(name) = part("numbering")
            && let Some((text, cut)) = zip.text(&name)?
        {
            lists.read(&text, cut, &mut budget).map_err(|why| bad(&name, why))?;
        }
        let (text, cut) = zip.text(&main)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {main}")))?;
        super::scanned(text.len());
        let rendered = rendered(&text, cut);
        let mut out = Convert::new(&mut budget, caps, &styles, &lists, &rels, rendered);
        out.run(&text, cut).map_err(|why| bad(&main, why))?;
        let (markdown, pages, images) = out.finish();
        let mut doc = Doc::new(markdown, pages);
        if images > 0 {
            doc.warnings.push(format!("{images} images (listed, not read)"));
        }
        if pages > caps.max_pages {
            doc.warnings.push(page_cap_warning(caps.max_pages, pages, "pages"));
        }
        doc.warnings.append(&mut zip.warnings);
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

/// Elements whose content is not document text.
/// The `...Change` elements hold the properties before a tracked change.
const SKIP: &[&str] = &[
    "Fallback", "del", "moveFrom", "instrText", "delText", "rPr", "sdtPr", "footnotePr", "endnotePr", "tblPr", "tblGrid", "trPr", "pPrChange",
    "sectPrChange", "tcPrChange", "tblPrEx",
];
/// Deepest list level Word writes, from 0.
const MAX_LEVEL: u8 = 8;
/// Storage of one style, list level or counter entry, for the work cap.
const ENTRY: usize = 64;
/// Storage of one table row or cell, for the work cap.
const SLOT: usize = 32;

#[derive(Default)]
struct Style {
    /// The heading level; `Some(None)` is an explicit body text outline
    /// level, which ends the search through `basedOn`.
    heading: Option<Option<u8>>,
    num: Option<String>,
    /// The list level; `ilvl` may come before or after `numId`.
    level: u8,
    based: Option<String>,
}

#[derive(Default)]
struct Styles(HashMap<String, Style>);

impl Styles {
    fn read(&mut self, text: &str, cut: bool, budget: &mut Budget) -> Result<(), String> {
        let mut scan = Scanner::new(text, cut);
        let mut current: Option<(String, Style)> = None;
        // Depth inside a tracked change's old properties.
        let mut skip = 0usize;
        while let Some(event) = scan.next()? {
            match event {
                Event::Open(tag) if skip > 0 || tag.local().ends_with("Change") => skip += 1,
                Event::Close(_) if skip > 0 => skip -= 1,
                Event::Open(tag) => {
                    let val = || tag.attr("val");
                    match (tag.local(), current.as_mut()) {
                        ("style", _) => current = tag.attr("styleId")?.map(|id| (id.into_owned(), Style::default())),
                        ("name", Some((_, s))) => s.heading = val()?.and_then(|n| heading_name(&n)).map(Some).or(s.heading),
                        ("basedOn", Some((_, s))) => s.based = val()?.map(|v| v.into_owned()),
                        ("outlineLvl", Some((_, s))) => s.heading = val()?.and_then(|v| outline(&v)).or(s.heading),
                        ("numId", Some((_, s))) => s.num = val()?.map(|v| v.into_owned()),
                        ("ilvl", Some((_, s))) => s.level = val()?.and_then(|v| v.parse::<u8>().ok()).unwrap_or(0).min(MAX_LEVEL),
                        _ => {}
                    }
                }
                Event::Close(name) if xml::local(name) == "style" => {
                    if let Some((id, style)) = current.take() {
                        let refs = style.num.as_ref().map_or(0, String::len) + style.based.as_ref().map_or(0, String::len);
                        if !budget.reserve(ENTRY + id.len() + refs) {
                            break;
                        }
                        self.0.insert(id, style);
                    }
                }
                _ => {}
            }
        }
        scan.finish(budget)
    }

    /// The heading level and list (id and level) of the style `id`, through
    /// its `basedOn` chain.
    fn get(&self, id: &str) -> (Option<u8>, Option<(&str, u8)>) {
        let (mut heading, mut list) = (None, None);
        let mut at = self.0.get(id);
        for _ in 0..16 {
            let Some(s) = at else { break };
            heading = heading.or(s.heading);
            list = list.or(s.num.as_deref().map(|n| (n, s.level)));
            at = s.based.as_deref().and_then(|b| self.0.get(b));
        }
        (heading.flatten(), list)
    }
}

/// The heading level of the outline level `val`: 0 to 8 give heading 1 to 6
/// (deeper levels stay at 6), 9 is body text (`Some(None)`). Other values
/// give no setting.
fn outline(val: &str) -> Option<Option<u8>> {
    let level = val.parse::<u8>().ok().filter(|&l| l <= 9)?;
    Some((level < 9).then(|| (level + 1).min(6)))
}

/// The level of a built-in heading style name: `heading 2` is 2, `Title` is 1.
fn heading_name(name: &str) -> Option<u8> {
    let name = name.trim().to_ascii_lowercase();
    if name == "title" {
        return Some(1);
    }
    let n: u8 = name.strip_prefix("heading")?.trim().parse().ok()?;
    (1..=9).contains(&n).then_some(n.min(6))
}

/// List formats: (abstract id, level) to (numbered, start), list id to
/// abstract id, and the level overrides of one list: (list id, level) to
/// the format and the start that replace the abstract list's.
#[derive(Default)]
struct Lists {
    levels: HashMap<(String, u8), (bool, u32)>,
    nums: HashMap<String, String>,
    overrides: HashMap<(String, u8), (Option<bool>, Option<u32>)>,
}

/// The element of `numbering.xml` that list properties belong to.
enum Scope {
    None,
    Abstract(String),
    Num(String),
}

impl Lists {
    fn read(&mut self, text: &str, cut: bool, budget: &mut Budget) -> Result<(), String> {
        let mut scan = Scanner::new(text, cut);
        let mut scope = Scope::None;
        // The level of the open `lvl`, and of the open `lvlOverride`.
        let (mut level, mut over): (Option<u8>, Option<u8>) = (None, None);
        let ilvl = |tag: &Tag| -> Result<Option<u8>, String> { Ok(tag.attr("ilvl")?.and_then(|v| v.parse::<u8>().ok()).filter(|&l| l <= MAX_LEVEL)) };
        while let Some(event) = scan.next()? {
            let tag = match event {
                Event::Open(tag) => tag,
                Event::Close(name) => {
                    match xml::local(name) {
                        "abstractNum" | "num" => scope = Scope::None,
                        "lvlOverride" => over = None,
                        "lvl" => level = None,
                        _ => {}
                    }
                    continue;
                }
                _ => continue,
            };
            match (tag.local(), &scope) {
                ("abstractNum", _) => scope = tag.attr("abstractNumId")?.map_or(Scope::None, |v| Scope::Abstract(v.into_owned())),
                ("num", _) => scope = tag.attr("numId")?.map_or(Scope::None, |v| Scope::Num(v.into_owned())),
                ("lvl", _) => level = ilvl(&tag)?,
                ("lvlOverride", Scope::Num(_)) => over = ilvl(&tag)?,
                ("numFmt" | "start", Scope::Abstract(a)) => {
                    let (Some(l), Some(v)) = (level, tag.attr("val")?) else { continue };
                    if !self.levels.contains_key(&(a.clone(), l)) && !budget.reserve(ENTRY + a.len()) {
                        break;
                    }
                    let entry = self.levels.entry((a.clone(), l)).or_insert((false, 1));
                    if tag.local() == "numFmt" {
                        entry.0 = numbered(&v);
                    } else {
                        entry.1 = v.parse().unwrap_or(1);
                    }
                }
                ("numFmt" | "start" | "startOverride", Scope::Num(n)) => {
                    let (Some(l), Some(v)) = (over, tag.attr("val")?) else { continue };
                    // A level inside the override needs its own level number to match.
                    if tag.local() != "startOverride" && level.is_some_and(|lv| lv != l) {
                        continue;
                    }
                    if !self.overrides.contains_key(&(n.clone(), l)) && !budget.reserve(ENTRY + n.len()) {
                        break;
                    }
                    let entry = self.overrides.entry((n.clone(), l)).or_insert((None, None));
                    if tag.local() == "numFmt" {
                        entry.0 = Some(numbered(&v));
                    } else {
                        entry.1 = Some(v.parse().unwrap_or(1));
                    }
                }
                ("abstractNumId", Scope::Num(n)) => {
                    let Some(v) = tag.attr("val")? else { continue };
                    if !budget.reserve(ENTRY + n.len() + v.len()) {
                        break;
                    }
                    self.nums.insert(n.clone(), v.into_owned());
                }
                _ => {}
            }
        }
        scan.finish(budget)
    }

    /// Whether list `num` at `level` is numbered, and its first number.
    fn format(&self, num: &str, level: u8) -> (bool, u32) {
        let base = self.nums.get(num).and_then(|a| self.levels.get(&(a.clone(), level))).copied().unwrap_or((false, 1));
        match self.overrides.get(&(num.to_owned(), level)) {
            Some(&(fmt, start)) => (fmt.unwrap_or(base.0), start.unwrap_or(base.1)),
            None => base,
        }
    }
}

/// True for a number format that is not a bullet.
fn numbered(format: &str) -> bool {
    !matches!(format, "bullet" | "none")
}

/// True when Word marked where its pages began: a `lastRenderedPageBreak`
/// element in the document text. Marks in content that the conversion
/// skips, in text boxes, in comments and in text do not count.
fn rendered(text: &str, cut: bool) -> bool {
    if !text.contains("lastRenderedPageBreak") {
        return false;
    }
    let mut scan = Scanner::new(text, cut);
    let (mut skip, mut text_boxes) = (0usize, 0usize);
    // A scan error is the conversion's to report.
    while let Ok(Some(event)) = scan.next() {
        match event {
            Event::Open(tag) if skip > 0 || SKIP.contains(&tag.local()) => skip += 1,
            Event::Close(_) if skip > 0 => skip -= 1,
            Event::Open(tag) => match tag.local() {
                "lastRenderedPageBreak" if text_boxes == 0 => return true,
                "txbxContent" => text_boxes += 1,
                _ => {}
            },
            Event::Close(name) if xml::local(name) == "txbxContent" => text_boxes = text_boxes.saturating_sub(1),
            _ => {}
        }
    }
    false
}

#[derive(Default)]
struct Para {
    text: String,
    style: Option<String>,
    num: Option<String>,
    /// The list level on the paragraph, if it gives one.
    level: Option<u8>,
    /// The outline level on the paragraph: `Some(None)` is body text.
    outline: Option<Option<u8>>,
    /// A section break after the paragraph: true for a new page.
    section: Option<bool>,
    /// The rest of a paragraph that a page break split; it has no prefix.
    continued: bool,
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<String>>,
    cell: String,
    span: usize,
    /// Work bytes that the table's storage and rows hold, not its open cell.
    held: usize,
}

struct Image {
    alt: String,
    target: Option<String>,
}

struct Convert<'a> {
    budget: &'a mut Budget,
    max_pages: usize,
    styles: &'a Styles,
    lists: &'a Lists,
    rels: &'a Rels,
    /// Pages come from Word's rendered page marks, not from hard breaks.
    rendered: bool,
    out: String,
    /// Where each page after the first starts in `out`.
    breaks: Vec<usize>,
    /// Work bytes held for the page markers, so that they show whole.
    marks: usize,
    pages: usize,
    /// The previous block was a list item, so the next item needs no blank line.
    in_list: bool,
    /// The last number at each level of each list; `None` before the first
    /// item, as 0 is a valid first number.
    counters: HashMap<String, [Option<u32>; 9]>,
    para: Para,
    p_depth: usize,
    p_props: usize,
    in_section: bool,
    in_text: bool,
    skip: usize,
    text_box: usize,
    link: Option<(usize, Option<String>)>,
    image: Option<Image>,
    images: usize,
    table: Option<Table>,
    nested: usize,
    /// Page breaks inside the open table, applied after it.
    table_breaks: usize,
}

impl<'a> Convert<'a> {
    fn new(budget: &'a mut Budget, caps: &Caps, styles: &'a Styles, lists: &'a Lists, rels: &'a Rels, rendered: bool) -> Self {
        let marks = mark_room(1);
        // The work cap is at least MIN_WORK_BYTES, so the first marker fits.
        budget.reserve(marks);
        Self {
            budget,
            max_pages: caps.max_pages.max(1),
            styles,
            lists,
            rels,
            rendered,
            out: String::new(),
            breaks: Vec::new(),
            marks,
            pages: 1,
            in_list: false,
            counters: HashMap::new(),
            para: Para::default(),
            p_depth: 0,
            p_props: 0,
            in_section: false,
            in_text: false,
            skip: 0,
            text_box: 0,
            link: None,
            image: None,
            images: 0,
            table: None,
            nested: 0,
            table_breaks: 0,
        }
    }

    /// True after the page cap: the scan only counts pages.
    fn muted(&self) -> bool {
        self.pages > self.max_pages
    }

    /// True when no more document text can be kept: after the page cap or a
    /// stop. The scan goes on to count pages but does no other work.
    fn idle(&self) -> bool {
        self.muted() || self.budget.full()
    }

    fn run(&mut self, text: &str, cut: bool) -> Result<(), String> {
        let mut scan = Scanner::new(text, cut);
        while let Some(event) = scan.next()? {
            match event {
                Event::Open(tag) => {
                    if self.skip > 0 || SKIP.contains(&tag.local()) {
                        self.skip += 1;
                        continue;
                    }
                    self.open(&tag)?;
                }
                Event::Close(name) => {
                    if self.skip > 0 {
                        self.skip -= 1;
                        continue;
                    }
                    self.close(xml::local(name));
                }
                // Only the text that can be kept is decoded.
                Event::Text(raw) if self.in_text && self.skip == 0 && !self.idle() => self.push(&xml::unescape_prefix(raw, self.budget.room())?),
                Event::Cdata(raw) if self.in_text && self.skip == 0 && !self.idle() => self.push(raw),
                _ => {}
            }
        }
        scan.finish(self.budget)?;
        if self.p_depth > 0 {
            self.end_para();
        }
        if self.table.is_some() {
            self.nested = 0;
            self.end_table();
        }
        Ok(())
    }

    fn open(&mut self, tag: &Tag) -> Result<(), String> {
        let props = self.p_props > 0;
        match tag.local() {
            "p" => {
                if self.p_depth == 0 {
                    self.para = Para::default();
                } else {
                    self.push(" ");
                }
                self.p_depth += 1;
            }
            "pPr" => self.p_props += 1,
            "pStyle" if props && self.p_depth == 1 => self.para.style = tag.attr("val")?.map(|v| v.into_owned()),
            "numId" if props && self.p_depth == 1 => self.para.num = tag.attr("val")?.map(|v| v.into_owned()),
            "ilvl" if props && self.p_depth == 1 => self.para.level = Some(tag.attr("val")?.and_then(|v| v.parse().ok()).unwrap_or(0).min(MAX_LEVEL)),
            "outlineLvl" if props && self.p_depth == 1 => {
                self.para.outline = tag.attr("val")?.and_then(|v| outline(&v));
            }
            "pageBreakBefore" if props && !self.rendered && tag.flag("val")? != Some(false) => self.page_break(),
            "sectPr" if props => {
                self.in_section = true;
                self.para.section = Some(true);
            }
            "type" if self.in_section && tag.attr("val")?.is_some_and(|v| v == "continuous") => self.para.section = Some(false),
            "t" if !props => self.in_text = true,
            "tab" if !props && self.p_depth > 0 => self.push("\t"),
            "br" | "cr" if !props => match tag.attr("type")?.as_deref() {
                Some("page") if !self.rendered => self.page_break(),
                Some("page" | "column") => {}
                _ => self.push("\n"),
            },
            "noBreakHyphen" => self.push("-"),
            "lastRenderedPageBreak" if self.rendered && self.text_box == 0 => self.page_break(),
            "hyperlink" if self.p_depth > 0 && self.link.is_none() => {
                let url = match tag.attr("r:id")? {
                    Some(id) if !self.idle() => self.rels.get(&id).filter(|r| r.external).map(|r| r.target.clone()),
                    _ => None,
                };
                self.link = Some((self.para.text.len(), url));
            }
            "drawing" | "pict" if self.image.is_none() && !self.idle() => self.image = Some(Image { alt: String::new(), target: None }),
            "docPr" | "cNvPr" => {
                if let Some(image) = self.image.as_mut().filter(|i| i.alt.is_empty()) {
                    image.alt = tag.alt_text(&["descr", "name"])?;
                }
            }
            "blip" | "imagedata" => {
                let id = if tag.local() == "blip" { tag.attr("r:embed")? } else { tag.attr("r:id")? };
                if let Some(image) = self.image.as_mut().filter(|i| i.target.is_none()) {
                    image.target = id.and_then(|id| self.rels.get(&id)).map(|r| r.target.clone());
                }
            }
            "txbxContent" => self.text_box += 1,
            "tbl" => {
                if self.table.is_some() {
                    self.nested += 1;
                } else if self.muted() {
                    // The table only defers its page breaks.
                    self.table = Some(Table::default());
                } else if self.budget.reserve(SLOT) {
                    self.in_list = false;
                    self.table = Some(Table { held: SLOT, ..Table::default() });
                }
            }
            "tr" if self.nested == 0 && !self.idle() => {
                if let Some(t) = self.table.as_mut()
                    && self.budget.reserve(SLOT)
                {
                    t.held += SLOT;
                    t.rows.push(Vec::new());
                }
            }
            "tc" if self.nested == 0 => {
                if let Some(t) = self.table.as_mut() {
                    t.cell.clear();
                    t.span = 1;
                }
            }
            "tr" | "tc" => self.cell_text(" "),
            "gridSpan" if self.nested == 0 => {
                if let Some(t) = self.table.as_mut() {
                    t.span = tag.attr("val")?.and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 64);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn close(&mut self, name: &str) {
        match name {
            "p" if self.p_depth > 0 => {
                self.p_depth -= 1;
                if self.p_depth == 0 {
                    self.end_para();
                }
            }
            "pPr" => self.p_props = self.p_props.saturating_sub(1),
            "sectPr" => self.in_section = false,
            "t" => self.in_text = false,
            "hyperlink" => {
                if let Some((start, url)) = self.link.take()
                    && let Some(url) = url
                    && start <= self.para.text.len()
                    && self.para.text.is_char_boundary(start)
                    && !self.para.text[start..].trim().is_empty()
                {
                    // A link that cannot close whole stays plain text and
                    // stops the reader; the room is known before the
                    // escaped target is built.
                    if self.budget.fits(1 + super::link_close_len(&url)) {
                        let close = format!("]({})", url.replace(' ', "%20").replace(')', "%29"));
                        let label = self.para.text.split_off(start);
                        self.para.text.push_str(self.budget.work("["));
                        self.para.text.push_str(&label);
                        self.para.text.push_str(self.budget.work(&close));
                    } else {
                        self.budget.cut();
                    }
                }
            }
            "drawing" | "pict" => {
                if let Some(image) = self.image.take()
                    && let Some(target) = image.target
                {
                    self.images += 1;
                    let alt = image.alt.replace(['[', ']', '\n'], " ");
                    let line = if alt.trim().is_empty() { format!("[image: {target}]") } else { format!("[image: {} ({target})]", alt.trim()) };
                    self.push(&line);
                }
            }
            "txbxContent" => self.text_box = self.text_box.saturating_sub(1),
            "tc" if self.nested == 0 => {
                let muted = self.muted();
                let Some(t) = self.table.as_mut() else { return };
                // The cell's text moves into its row: its bytes go back first.
                self.budget.release(t.cell.len());
                let mut value = cell(&std::mem::take(&mut t.cell));
                while value.ends_with("<br>") {
                    value.truncate(value.len() - 4);
                }
                // The cell's slot is held first, then its moved text, then
                // the slots of the columns it spans, as many as fit.
                if t.rows.is_empty() || muted || !self.budget.reserve(SLOT) {
                    return;
                }
                let value = self.budget.work(&value).to_owned();
                let mut spans = 1;
                while spans < t.span && self.budget.reserve(SLOT) {
                    spans += 1;
                }
                t.held += SLOT * spans + value.len();
                if let Some(row) = t.rows.last_mut() {
                    row.push(value);
                    row.extend((1..spans).map(|_| String::new()));
                }
            }
            "tbl" if self.nested > 0 => self.nested -= 1,
            "tbl" => self.end_table(),
            _ => {}
        }
    }

    /// Appends document text to the open paragraph, within the budget.
    fn push(&mut self, text: &str) {
        if self.muted() || self.p_depth == 0 {
            return;
        }
        let kept = self.budget.text(text);
        self.para.text.push_str(kept);
    }

    /// Appends separator text to the open cell of a nested table.
    fn cell_text(&mut self, text: &str) {
        if self.muted() {
            return;
        }
        if let Some(t) = self.table.as_mut() {
            t.cell.push_str(self.budget.work(text));
        }
    }

    /// A page starts here. The open paragraph's text so far stays on the
    /// page before; the rest continues on the new page.
    fn page_break(&mut self) {
        if self.table.is_some() {
            self.table_breaks += 1;
            return;
        }
        if self.p_depth > 0 && !self.para.text.trim().is_empty() {
            self.emit_para();
            self.para.text.clear();
            self.para.continued = true;
            self.link = None;
        }
        self.new_page();
    }

    fn new_page(&mut self) {
        self.pages += 1;
        // A page that starts after a stop is counted, not shown.
        if self.idle() {
            return;
        }
        let room = 8 + mark_room(self.pages);
        if self.budget.reserve(room) {
            self.marks += room;
            self.breaks.push(self.out.len());
            self.in_list = false;
        }
    }

    fn end_para(&mut self) {
        if self.para.text.trim().is_empty() {
            self.budget.release(self.para.text.len());
        } else {
            self.emit_para();
        }
        let section = self.para.section.take();
        self.para = Para::default();
        self.link = None;
        if section == Some(true) && !self.rendered {
            self.page_break();
        }
    }

    /// Moves the open paragraph, with its heading or list prefix, into the
    /// open table cell or the output. Its bytes go back to the budget first.
    fn emit_para(&mut self) {
        self.budget.release(self.para.text.len());
        if self.muted() {
            return;
        }
        let (style_heading, style_list) = self.para.style.as_deref().map_or((None, None), |s| self.styles.get(s));
        // An outline level on the paragraph wins, also body text (level 9).
        let heading = self.para.outline.unwrap_or(style_heading);
        // Word numbers its heading styles through lists; a heading stays a heading.
        // A level on the paragraph wins over the style's level, also for a
        // list that the style gives.
        let level = self.para.level.or(style_list.map(|(_, l)| l)).unwrap_or(0);
        let list = match (&self.para.num, style_list) {
            _ if heading.is_some() && self.table.is_none() => None,
            (Some(n), _) if n == "0" => None,
            (Some(n), _) => Some((n.clone(), level)),
            (None, Some((n, _))) if n != "0" => Some((n.to_owned(), level)),
            _ => None,
        };
        let text = self.para.text.trim();
        let prefix = if self.para.continued {
            String::new()
        } else if let Some((num, level)) = &list {
            let (numbered, start) = self.lists.format(num, *level);
            let indent = "    ".repeat(usize::from(*level));
            if numbered {
                if !self.counters.contains_key(num) && !self.budget.reserve(ENTRY + num.len()) {
                    return;
                }
                let c = self.counters.entry(num.clone()).or_insert([None; 9]);
                let l = usize::from(*level);
                let n = c[l].map_or(start, |n| n.saturating_add(1));
                c[l] = Some(n);
                c[l + 1..].fill(None);
                format!("{indent}{}. ", n.min(999_999_999))
            } else {
                format!("{indent}- ")
            }
        } else if let Some(h) = heading.filter(|_| self.table.is_none()) {
            format!("{} ", "#".repeat(usize::from(h)))
        } else {
            String::new()
        };
        if let Some(t) = self.table.as_mut() {
            let sep = if t.cell.is_empty() { "" } else { "\n" };
            let piece = format!("{sep}{prefix}{text}");
            t.cell.push_str(self.budget.work(&piece));
            return;
        }
        let item = list.is_some() && !self.para.continued;
        let sep = if self.out.is_empty() || self.breaks.last() == Some(&self.out.len()) {
            ""
        } else if item && self.in_list {
            "\n"
        } else {
            "\n\n"
        };
        let piece = format!("{sep}{prefix}{text}");
        self.out.push_str(self.budget.work(&piece));
        self.in_list = item;
    }

    fn end_table(&mut self) {
        let Some(t) = self.table.take() else { return };
        // The rows move into the output: their bytes go back first.
        self.budget.release(t.held + t.cell.len());
        let rows: Vec<Vec<String>> = t.rows.into_iter().filter(|r| r.iter().any(|c| !c.is_empty())).collect();
        if !self.muted() && !rows.is_empty() {
            let sep = if self.out.is_empty() || self.breaks.last() == Some(&self.out.len()) { "" } else { "\n\n" };
            self.out.push_str(self.budget.work(sep));
            if rows.len() == 1 && rows[0].len() == 1 {
                let text = rows[0][0].replace("<br>", "\n");
                self.out.push_str(self.budget.work(&text));
            } else {
                // The cells were counted as text once; the table counts as work.
                let counts = std::mem::replace(&mut self.budget.text_counts, false);
                let md = table_in(&rows, self.budget);
                self.budget.text_counts = counts;
                self.out.push_str(md.trim_end());
            }
        }
        self.in_list = false;
        for _ in 0..std::mem::take(&mut self.table_breaks) {
            self.new_page();
        }
    }

    /// The Markdown with a marker before each page, the page count and the
    /// image count. The output and the room held for the markers go back to
    /// the budget, and each piece is counted again as it moves.
    fn finish(self) -> (String, usize, usize) {
        let mut md = String::new();
        let mut start = 0;
        self.budget.release(self.out.len() + self.marks);
        let ends = self.breaks.iter().copied().chain([self.out.len()]);
        for (i, end) in ends.enumerate() {
            let marker = page_marker(i + 1, self.pages);
            let marker = if i == 0 { marker.trim_start() } else { &marker };
            if !self.budget.fits(marker.len()) {
                self.budget.cut();
                break;
            }
            md.push_str(self.budget.work(marker));
            md.push_str(self.budget.work(self.out[start..end].trim_matches('\n')));
            md.push('\n');
            start = end;
        }
        (md, self.pages, self.images)
    }
}

/// Work bytes to hold for the marker of page `n`: room for a page count of
/// up to four more digits than `n`.
fn mark_room(n: usize) -> usize {
    page_marker(n, n).len() + 4
}
