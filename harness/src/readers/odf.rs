//! OpenDocument text, spreadsheets and presentations (ODT, ODS, ODP) to
//! Markdown, from `content.xml`.
//!
//! - ODT: headings by outline level, paragraphs, bullet lists with their
//!   levels, tables, links, line breaks and images by name. Pages come from
//!   the soft page breaks that LibreOffice writes where its last layout
//!   started a page; a file with none is one page.
//! - ODS: one sheet marker and one table for each sheet, with the text each
//!   cell shows. Repeated rows and columns expand only between values, so a
//!   sheet that repeats empty cells to the last column costs nothing.
//! - ODP: one slide marker for each page, the title as a heading, the text
//!   frames, tables, images and the speaker notes.
//!
//! Encrypted files (a password) are refused. Skipped: styles, annotations,
//! footnotes, tracked changes, forms, scripts and formatting.
use super::xml::{self, Event, Scanner, Tag};
use super::xlsx::{MAX_COL, MAX_ROW, keep_prefix, sheet_table};
use super::zip::{self, Zip};
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, is_zip, page_cap_warning, page_marker, sheet_marker, slide_marker, table_in};

pub(super) struct OdfReader;

/// Elements whose content is not document text.
const SKIP: &[&str] = &[
    "automatic-styles", "font-face-decls", "scripts", "annotation", "tracked-changes", "sequence-decls", "forms", "note", "desc",
    "title", "variable-decls", "user-field-decls", "event-listeners",
];
/// Presentation frame classes that hold slide furniture.
const FURNITURE: &[&str] = &["page-number", "footer", "date-time", "header", "page"];
/// Storage of one table row or cell, for the work cap.
const SLOT: usize = 32;
/// Most copies of a repeated cell in a text or slide table.
const MAX_REPEAT: usize = 64;

impl Reader for OdfReader {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format> {
        let by_type = zip::mimetype(bytes).and_then(|m| match m.strip_prefix("application/vnd.oasis.opendocument.") {
            Some("text" | "text-template") => Some(Format::Odt),
            Some("spreadsheet" | "spreadsheet-template") => Some(Format::Ods),
            Some("presentation" | "presentation-template") => Some(Format::Odp),
            _ => None,
        });
        by_type.or_else(|| self.by_name(name))
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        if !is_zip(bytes) {
            return Err(ReadError::Malformed(format, "the file is not a ZIP archive".into()));
        }
        let mut budget = Budget::new(caps);
        let mut zip = Zip::open(bytes, format)?;
        let bad = |why: String| ReadError::Malformed(format, format!("content.xml {why}"));
        if let Some((text, cut)) = zip.text("META-INF/manifest.xml")?
            && encrypted(&text, cut, &mut budget).map_err(|why| ReadError::Malformed(format, format!("META-INF/manifest.xml {why}")))?
        {
            return Err(ReadError::Encrypted(format));
        }
        let (text, cut) = zip.text("content.xml")?.ok_or_else(|| ReadError::Malformed(format, "the archive has no content.xml".into()))?;
        let mut doc = if format == Format::Ods {
            sheets(&text, cut, caps, &mut budget).map_err(bad)?
        } else {
            let mut flow = Flow::new(&mut budget, caps, format == Format::Odp);
            flow.run(&text, cut).map_err(bad)?;
            flow.finish()
        };
        doc.warnings.append(&mut zip.warnings);
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

impl OdfReader {
    fn by_name(&self, name: &str) -> Option<Format> {
        match extension(name).as_str() {
            "odt" | "ott" => Some(Format::Odt),
            "ods" | "ots" => Some(Format::Ods),
            "odp" | "otp" => Some(Format::Odp),
            _ => None,
        }
    }
}

/// True when the manifest has encryption data for any entry.
fn encrypted(text: &str, cut: bool, budget: &mut Budget) -> Result<bool, String> {
    let mut scan = Scanner::new(text, cut);
    while let Some(event) = scan.next()? {
        if let Event::Open(tag) = event
            && tag.local() == "encryption-data"
        {
            return Ok(true);
        }
    }
    scan.finish(budget)?;
    Ok(false)
}

/// The repeat count in `attr`, from 1 to `max`.
fn repeat(tag: &Tag, attr: &str, max: usize) -> Result<usize, String> {
    Ok(tag.attr(attr)?.and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).clamp(1, max))
}

/// Work bytes to hold for the marker of page or slide `n`: room for a
/// count of up to four more digits than `n`.
fn mark_room(slides: bool, n: usize) -> usize {
    (if slides { slide_marker(n, n) } else { page_marker(n, n) }).len() + 4
}

/// Where a finished paragraph goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sink {
    Out,
    Title,
    Notes,
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<String>>,
    cell: String,
    span: usize,
    /// Work bytes that the table's storage and rows hold, not its open cell.
    held: usize,
}

/// Text flow for ODT and ODP.
struct Flow<'a> {
    budget: &'a mut Budget,
    max_pages: usize,
    slides: bool,
    out: String,
    breaks: Vec<usize>,
    /// Work bytes held for the page or slide markers, so that they show whole.
    marks: usize,
    pages: usize,
    skip: usize,
    para: String,
    p_depth: usize,
    /// Heading level of the open paragraph, or 0.
    heading: usize,
    /// List depth, and whether the open list item has had its marker.
    lists: usize,
    item_marked: bool,
    /// The previous block in each sink was a list item.
    last_item: [bool; 3],
    link: Option<(usize, String)>,
    image: Option<(String, Option<String>)>,
    images: usize,
    table: Option<Table>,
    nested: usize,
    table_breaks: usize,
    /// Frame classes, innermost last; `title` and the furniture classes route text.
    /// Each with the work bytes it holds (0 when the cap left no room).
    frames: Vec<(String, usize)>,
    notes: usize,
    title: String,
    note_text: String,
}

impl<'a> Flow<'a> {
    fn new(budget: &'a mut Budget, caps: &Caps, slides: bool) -> Self {
        let marks = mark_room(slides, 1);
        // The work cap is at least MIN_WORK_BYTES, so the first marker fits.
        budget.reserve(marks);
        Self {
            budget,
            max_pages: caps.max_pages.max(1),
            slides,
            out: String::new(),
            breaks: Vec::new(),
            marks,
            pages: usize::from(!slides),
            skip: 0,
            para: String::new(),
            p_depth: 0,
            heading: 0,
            lists: 0,
            item_marked: false,
            last_item: [false; 3],
            link: None,
            image: None,
            images: 0,
            table: None,
            nested: 0,
            table_breaks: 0,
            frames: Vec::new(),
            notes: 0,
            title: String::new(),
            note_text: String::new(),
        }
    }

    fn muted(&self) -> bool {
        self.pages > self.max_pages || self.frames.iter().any(|(c, _)| FURNITURE.contains(&c.as_str()))
    }

    /// True when no more document text can be kept here; the scan goes on
    /// to count pages but decodes no text.
    fn idle(&self) -> bool {
        self.muted() || self.budget.full()
    }

    /// Closes the open link with no Markdown; its target's bytes go back.
    fn drop_link(&mut self) {
        if let Some((_, href)) = self.link.take() {
            self.budget.release(href.len());
        }
    }

    /// Holds the room for one more marker; true when it fits.
    fn hold_mark(&mut self) -> bool {
        let room = 8 + mark_room(self.slides, self.pages);
        let held = self.budget.reserve(room);
        if held {
            self.marks += room;
        }
        held
    }

    fn sink(&self) -> Sink {
        if self.notes > 0 {
            Sink::Notes
        } else if self.frames.iter().any(|(c, _)| c == "title") {
            Sink::Title
        } else {
            Sink::Out
        }
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
                Event::Text(raw) if self.skip == 0 && !self.idle() => self.push(&xml::unescape_prefix(raw, self.budget.room())?),
                Event::Cdata(raw) if self.skip == 0 && !self.idle() => self.push(raw),
                _ => {}
            }
        }
        scan.finish(self.budget)?;
        if self.p_depth > 0 {
            self.p_depth = 1;
            self.close("p");
        }
        if self.table.is_some() {
            self.nested = 0;
            self.end_table();
        }
        Ok(())
    }

    fn open(&mut self, tag: &Tag) -> Result<(), String> {
        match tag.local() {
            "p" | "h" => {
                if self.p_depth == 0 {
                    self.para.clear();
                    self.heading = if tag.local() == "h" { tag.attr("outline-level")?.and_then(|l| l.parse().ok()).unwrap_or(1).clamp(1, 6) } else { 0 };
                } else {
                    self.push(" ");
                }
                self.p_depth += 1;
            }
            "s" => {
                let n = repeat(tag, "c", 64)?;
                self.push(&" ".repeat(n));
            }
            "tab" => self.push("\t"),
            "line-break" => self.push("\n"),
            "list" => self.lists += 1,
            "list-item" => self.item_marked = false,
            "list-header" => self.item_marked = true,
            "a" if self.p_depth > 0 && self.link.is_none() => {
                // The target counts against the work cap while the link is open.
                let href = match tag.attr("xlink:href")? {
                    Some(h) if !self.idle() && self.budget.reserve(h.len()) => h.into_owned(),
                    _ => String::new(),
                };
                self.link = Some((self.para.len(), href));
            }
            "soft-page-break" if !self.slides => self.page_break(),
            "page" if self.slides => {
                self.pages += 1;
                // A slide that starts after a stop is counted, not shown.
                if !self.idle() && self.pages > 1 && self.hold_mark() {
                    self.breaks.push(self.out.len());
                }
                self.last_item = [false; 3];
            }
            "notes" if self.slides => self.notes += 1,
            "frame" | "custom-shape" => {
                let class = tag.attr("presentation:class")?.unwrap_or_default().into_owned();
                // A frame always opens, so its close pops it; with no room it
                // holds nothing and routes nothing.
                let held = SLOT + class.len();
                let frame = if self.budget.reserve(held) { (class, held) } else { (String::new(), 0) };
                self.frames.push(frame);
                // The innermost frame names the image it holds.
                if tag.local() == "frame" && self.image.as_ref().is_none_or(|i| i.1.is_none()) {
                    self.image = Some((tag.alt_text(&["draw:name"])?, None));
                }
            }
            "image" if !self.idle() => {
                if let (Some((_, target)), Some(href)) = (self.image.as_mut(), tag.attr("xlink:href")?) {
                    target.get_or_insert_with(|| if href.contains(':') { href.into_owned() } else { zip::resolve("", &href) });
                }
            }
            "table" => {
                if self.table.is_some() {
                    self.nested += 1;
                } else if self.muted() {
                    // The table only defers its page breaks.
                    self.table = Some(Table::default());
                } else if self.budget.reserve(SLOT) {
                    self.last_item[0] = false;
                    self.table = Some(Table { held: SLOT, ..Table::default() });
                }
            }
            "table-row" if self.nested == 0 && !self.idle() => {
                if let Some(t) = self.table.as_mut()
                    && self.budget.reserve(SLOT)
                {
                    t.held += SLOT;
                    t.rows.push(Vec::new());
                }
            }
            "table-cell" | "covered-table-cell" if self.nested == 0 => {
                let span = repeat(tag, "number-columns-repeated", MAX_REPEAT)?;
                if let Some(t) = self.table.as_mut() {
                    t.cell.clear();
                    t.span = span;
                }
            }
            "table-row" | "table-cell" if !self.muted() => {
                if let Some(t) = self.table.as_mut() {
                    t.cell.push_str(self.budget.work(" "));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn close(&mut self, name: &str) {
        match name {
            "p" | "h" if self.p_depth > 0 => {
                self.p_depth -= 1;
                if self.p_depth == 0 {
                    self.end_para();
                }
            }
            "list" => self.lists = self.lists.saturating_sub(1),
            "a" => {
                let Some((start, href)) = self.link.take() else { return };
                self.budget.release(href.len());
                if start <= self.para.len()
                    && self.para.is_char_boundary(start)
                    && !self.para[start..].trim().is_empty()
                    && !href.is_empty()
                    && !href.starts_with('#')
                {
                    // A link that cannot close whole stays plain text and
                    // stops the reader; the room is known before the
                    // escaped target is built.
                    if self.budget.fits(1 + super::link_close_len(&href)) {
                        let close = format!("]({})", href.replace(' ', "%20").replace(')', "%29"));
                        let label = self.para.split_off(start);
                        self.para.push_str(self.budget.work("["));
                        self.para.push_str(&label);
                        self.para.push_str(self.budget.work(&close));
                    } else {
                        self.budget.cut();
                    }
                }
            }
            "notes" => {
                self.notes = self.notes.saturating_sub(1);
                if self.notes == 0 {
                    let notes = std::mem::take(&mut self.note_text);
                    self.budget.release(notes.len());
                    if !notes.trim().is_empty() && !self.muted() {
                        let block = format!("#### Notes\n\n{}", notes.trim());
                        self.add(Sink::Out, &block, false);
                    }
                }
            }
            "frame" | "custom-shape" => {
                let image = if name == "frame" { self.image.take_if(|i| i.1.is_some()) } else { None };
                if let Some((alt, Some(target))) = image
                    && !self.muted()
                {
                    let alt = alt.replace(['[', ']', '\n'], " ");
                    let line = if alt.trim().is_empty() { format!("[image: {target}]") } else { format!("[image: {} ({target})]", alt.trim()) };
                    // An image in a paragraph stays where it is in the text,
                    // also in a table cell; one between paragraphs is a block.
                    if self.p_depth > 0 {
                        self.images += 1;
                        self.push(&line);
                    } else if self.table.is_none() {
                        self.images += 1;
                        let line = self.budget.text(&line).to_owned();
                        self.add(Sink::Out, &line, false);
                    }
                }
                let (class, held) = self.frames.pop().unwrap_or_default();
                self.budget.release(held);
                if class == "title" {
                    let title = std::mem::take(&mut self.title);
                    self.budget.release(title.len());
                    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !title.is_empty() && !self.muted() {
                        self.add(Sink::Out, &format!("### {title}"), false);
                    }
                }
            }
            "table-cell" | "covered-table-cell" if self.nested == 0 => {
                let muted = self.muted();
                let Some(t) = self.table.as_mut() else { return };
                // The cell's text moves into its row: its bytes go back first.
                self.budget.release(t.cell.len());
                let value = cell(&std::mem::take(&mut t.cell));
                let Some(row) = t.rows.last_mut().filter(|_| !muted) else { return };
                // The cell's slot is held first, then its moved text. A
                // repeated cell keeps its text once for each column, as many
                // columns as fit.
                if !self.budget.reserve(SLOT) {
                    return;
                }
                let value = self.budget.work(&value).to_owned();
                t.held += SLOT + value.len();
                row.push(value.clone());
                for _ in 1..t.span {
                    if !self.budget.reserve(SLOT + value.len()) {
                        break;
                    }
                    t.held += SLOT + value.len();
                    row.push(value.clone());
                }
            }
            "table" if self.nested > 0 => self.nested -= 1,
            "table" => self.end_table(),
            _ => {}
        }
    }

    fn push(&mut self, text: &str) {
        if self.p_depth == 0 || self.muted() {
            return;
        }
        let kept = self.budget.text(text);
        self.para.push_str(kept);
    }

    fn page_break(&mut self) {
        if self.table.is_some() {
            self.table_breaks += 1;
            return;
        }
        if self.p_depth > 0 && !self.para.trim().is_empty() {
            let text = std::mem::take(&mut self.para);
            self.emit(text);
            self.heading = 0;
            self.item_marked = true;
            self.drop_link();
        }
        self.new_page();
    }

    fn new_page(&mut self) {
        self.pages += 1;
        // A page that starts after a stop is counted, not shown.
        if !self.idle() && self.hold_mark() {
            self.breaks.push(self.out.len());
            self.last_item = [false; 3];
        }
    }

    fn end_para(&mut self) {
        let text = std::mem::take(&mut self.para);
        self.drop_link();
        if !text.trim().is_empty() {
            self.emit(text);
        } else {
            self.budget.release(text.len());
        }
        self.heading = 0;
    }

    /// Moves a finished paragraph, with its heading or list prefix, into
    /// the open cell, the title, the notes or the output.
    fn emit(&mut self, text: String) {
        self.budget.release(text.len());
        if self.muted() {
            return;
        }
        let body = text.trim();
        let item = self.lists > 0 && !self.item_marked;
        let prefix = if item {
            format!("{}- ", "    ".repeat(self.lists - 1))
        } else if self.lists > 0 {
            "    ".repeat(self.lists)
        } else if self.heading > 0 && self.table.is_none() {
            format!("{} ", "#".repeat(self.heading))
        } else {
            String::new()
        };
        if self.lists > 0 {
            self.item_marked = true;
        }
        if let Some(t) = self.table.as_mut() {
            let sep = if t.cell.is_empty() { "" } else { "\n" };
            t.cell.push_str(self.budget.work(&format!("{sep}{body}")));
            return;
        }
        let line = format!("{prefix}{body}");
        match self.sink() {
            Sink::Title => self.title.push_str(self.budget.work(&format!(" {body}"))),
            sink => self.add(sink, &line, self.lists > 0),
        }
    }

    /// Appends a block to `sink`: a list line after a list line takes one
    /// line break, any other block a blank line.
    fn add(&mut self, sink: Sink, block: &str, item: bool) {
        let i = sink as usize;
        let at_page = self.breaks.last() == Some(&self.out.len());
        let target = if sink == Sink::Notes { &mut self.note_text } else { &mut self.out };
        let sep = if target.is_empty() || (sink == Sink::Out && at_page) {
            ""
        } else if item && self.last_item[i] {
            "\n"
        } else {
            "\n\n"
        };
        target.push_str(self.budget.work(&format!("{sep}{block}")));
        self.last_item[i] = item;
    }

    fn end_table(&mut self) {
        let Some(t) = self.table.take() else { return };
        // The rows move into the output: their bytes go back first.
        self.budget.release(t.held + t.cell.len());
        let rows: Vec<Vec<String>> = t.rows.into_iter().filter(|r| r.iter().any(|c| !c.is_empty())).collect();
        if !self.muted() && !rows.is_empty() {
            let md = if rows.len() == 1 && rows[0].len() == 1 {
                rows[0][0].replace("<br>", "\n")
            } else {
                let counts = std::mem::replace(&mut self.budget.text_counts, false);
                let md = table_in(&rows, self.budget);
                self.budget.text_counts = counts;
                md.trim_end().to_owned()
            };
            // `add` counts the block again as it moves into the sink.
            if rows.len() > 1 || rows[0].len() > 1 {
                self.budget.release(md.len());
            }
            let sink = if self.notes > 0 { Sink::Notes } else { Sink::Out };
            self.add(sink, &md, false);
        }
        for _ in 0..std::mem::take(&mut self.table_breaks) {
            self.new_page();
        }
    }

    fn finish(self) -> Doc {
        let total = self.pages.max(1);
        let mut md = String::new();
        let mut start = 0;
        self.budget.release(self.out.len() + self.marks);
        let kept = total.min(self.max_pages);
        let ends = self.breaks.iter().copied().chain([self.out.len()]).take(kept);
        for (i, end) in ends.enumerate() {
            let marker = if self.slides { slide_marker(i + 1, total) } else { page_marker(i + 1, total) };
            let marker = if i == 0 { marker.trim_start() } else { &marker };
            if !self.budget.fits(marker.len()) {
                self.budget.cut();
                break;
            }
            md.push_str(self.budget.work(marker));
            let body = self.out[start..end].trim_matches('\n');
            let empty = if self.slides { "(no text on this slide)" } else { "" };
            if body.is_empty() {
                self.budget.whole(&mut md, empty);
            } else {
                md.push_str(self.budget.work(body));
            }
            md.push('\n');
            start = end;
        }
        let mut doc = Doc::new(md, total);
        if self.images > 0 {
            doc.warnings.push(format!("{} images (listed, not read)", self.images));
        }
        if total > self.max_pages {
            doc.warnings.push(page_cap_warning(self.max_pages, total, if self.slides { "slides" } else { "pages" }));
        }
        doc
    }
}

/// An ODS spreadsheet: a sheet marker and a table for each `table:table`.
fn sheets(text: &str, cut: bool, caps: &Caps, budget: &mut Budget) -> Result<Doc, String> {
    let keep = caps.max_rows.saturating_add(1);
    let mut scan = Scanner::new(text, cut);
    // Finished sheets: name, table, and the work bytes held for them.
    let mut done: Vec<(String, String, usize)> = Vec::new();
    let mut warnings = Vec::new();
    let mut total = 0usize;
    // The open sheet: name, kept rows, the open row, counted rows, row and column.
    let mut name = String::new();
    let mut rows: Vec<Vec<(u32, String)>> = Vec::new();
    let mut row: Vec<(u32, String)> = Vec::new();
    let (mut counted, mut col, mut row_repeat, mut cell_repeat) = (0usize, 0u32, 1usize, 1usize);
    let (mut min_col, mut max_col) = (u32::MAX, 0u32);
    let (mut value, mut fallback, mut in_cell, mut paras, mut skip, mut depth) = (String::new(), String::new(), false, 0usize, 0usize, 0usize);
    // Work bytes that the open sheet's kept rows hold; whether the open cell
    // has text that is counted but not read (past a cap); and whether the
    // work cap cut the end of its text.
    let (mut held, mut seen, mut cut_cell) = (0usize, false, false);
    // The last sheet shown: the sheet cap, or the last sheet that started
    // before a cap stopped the reader. Later sheets are counted, not read.
    let mut last = caps.max_pages;
    while let Some(event) = scan.next()? {
        match event {
            Event::Open(tag) => {
                if skip > 0 || matches!(tag.local(), "annotation" | "note" | "automatic-styles" | "font-face-decls" | "scripts") {
                    skip += 1;
                    continue;
                }
                match tag.local() {
                    "table" => {
                        depth += 1;
                        if depth == 1 {
                            total += 1;
                            if budget.full() && total <= last {
                                last = total - 1;
                            }
                            name = tag.attr("table:name")?.unwrap_or_default().into_owned();
                            rows.clear();
                            (counted, min_col, max_col, held) = (0, u32::MAX, 0, 0);
                        }
                    }
                    "table-row" if depth == 1 => {
                        row.clear();
                        col = 0;
                        row_repeat = repeat(&tag, "number-rows-repeated", MAX_ROW as usize)?;
                    }
                    "table-cell" | "covered-table-cell" if depth == 1 => {
                        cell_repeat = repeat(&tag, "number-columns-repeated", MAX_COL as usize)?;
                        value.clear();
                        paras = 0;
                        in_cell = true;
                        (seen, cut_cell) = (false, false);
                        // A value with no displayed text shows its stored form.
                        // Past a cap it is counted, not read.
                        fallback.clear();
                        let skip_value = counted >= keep || budget.full() || total > last;
                        for a in ["value", "date-value", "time-value", "boolean-value", "string-value"] {
                            if skip_value {
                                seen |= tag.attr_prefix(a, 0)?.is_some();
                            } else if let Some(v) = tag.attr(a)? {
                                fallback = v.into_owned();
                                break;
                            }
                        }
                    }
                    // Spaces, tabs and breaks grow the value only within the room left.
                    "p" if in_cell => {
                        if paras > 0 {
                            if value.len() < budget.room() {
                                value.push('\n');
                            }
                        }
                        paras += 1;
                    }
                    "s" if in_cell => {
                        let want = repeat(&tag, "c", 64)?;
                        let n = want.min(budget.room().saturating_sub(value.len()));
                        cut_cell |= n < want;
                        value.push_str(&" ".repeat(n));
                    }
                    "tab" if in_cell => {
                        if value.len() < budget.room() {
                            value.push('\t');
                        }
                    }
                    "line-break" if in_cell => {
                        if value.len() < budget.room() {
                            value.push('\n');
                        }
                    }
                    _ => {}
                }
            }
            Event::Close(close) => {
                if skip > 0 {
                    skip -= 1;
                    continue;
                }
                match xml::local(close) {
                    "table-cell" | "covered-table-cell" if depth == 1 => {
                        in_cell = false;
                        let shown = if value.trim().is_empty() { std::mem::take(&mut fallback) } else { std::mem::take(&mut value) };
                        let start = col + 1;
                        col = col.saturating_add(cell_repeat as u32).min(MAX_COL);
                        let has = seen || !shown.trim().is_empty();
                        if !has || seen || counted >= keep || (budget.full() && !cut_cell) || total > last {
                            if has && row.is_empty() {
                                row.push((start, String::new()));
                            }
                            continue;
                        }
                        // The cell's slot is held first, then its text keeps the
                        // start that fits; each more copy holds its own text.
                        if !budget.reserve(SLOT) {
                            continue;
                        }
                        let text = budget.text(&shown);
                        let charged = text.len();
                        let kept = cell(text);
                        // Escapes that grow the text count too, when they fit.
                        let grown = kept.len().saturating_sub(charged);
                        held += SLOT + charged + if budget.reserve(grown) { grown } else { 0 };
                        (min_col, max_col) = (min_col.min(start), max_col.max(start));
                        row.push((start, kept.clone()));
                        for c in start + 1..=col {
                            if !budget.reserve(SLOT + kept.len()) {
                                break;
                            }
                            held += SLOT + kept.len();
                            max_col = max_col.max(c);
                            row.push((c, kept.clone()));
                        }
                        // The cut cell keeps its start; the stop comes after it.
                        if cut_cell {
                            budget.cut();
                        }
                    }
                    "table-row" if depth == 1 => {
                        if !row.is_empty() {
                            // Repeated rows past the row cap are counted, not copied.
                            // The row itself is held already, also when a cap
                            // stopped it; each more copy holds its cells' text too.
                            let copies = keep.saturating_sub(counted).min(row_repeat);
                            if copies > 0 && row.iter().any(|(_, v)| !v.is_empty()) {
                                let size = SLOT + row.iter().map(|(_, v)| SLOT + v.len()).sum::<usize>();
                                for _ in 1..copies {
                                    if budget.full() || !budget.reserve(size) {
                                        break;
                                    }
                                    held += size;
                                    rows.push(row.clone());
                                }
                                rows.push(std::mem::take(&mut row));
                            }
                            counted = counted.saturating_add(row_repeat);
                        }
                        row.clear();
                    }
                    "table" => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            if total > last {
                                continue;
                            }
                            if counted > keep {
                                warnings.push(format!("rows after {} of {} in sheet {name} (row cap)", caps.max_rows, counted - 1));
                            }
                            let md = sheet_table(std::mem::take(&mut rows), held, min_col, max_col, budget);
                            held = 0;
                            // Room for the sheet's marker, which shows only whole.
                            let mark = sheet_marker(done.len() + 1, done.len() + 1, &name).len() + 4;
                            let mark = if budget.reserve(mark) { mark } else { 0 };
                            done.push((std::mem::take(&mut name), md, mark));
                        }
                    }
                    _ => {}
                }
            }
            Event::Text(raw) | Event::Cdata(raw) if in_cell && skip == 0 && paras > 0 => {
                // Past the row cap, a stop or the sheet cap, a value is counted, not read.
                if counted >= keep || budget.full() || total > last {
                    seen |= !raw.trim().is_empty();
                    continue;
                }
                // A value larger than the room keeps the start that fits, and
                // only that start is decoded.
                let room = budget.room().saturating_sub(value.len());
                let text = if matches!(event, Event::Text(_)) { xml::unescape_prefix(raw, room)? } else { raw.into() };
                cut_cell |= !keep_prefix(&mut value, &text, room);
            }
            _ => {}
        }
    }
    scan.finish(budget)?;
    if total == 0 {
        return Err("has no sheets".into());
    }
    let mut md = String::new();
    // The sheets move into the output: their bytes go back first.
    budget.release(done.iter().map(|(_, body, mark)| body.len() + mark).sum());
    for (n, (name, body, _)) in done.iter().enumerate() {
        let marker = sheet_marker(n + 1, total, name);
        let marker = if n == 0 { marker.trim_start() } else { &marker };
        if !budget.fits(marker.len()) {
            budget.cut();
            break;
        }
        md.push_str(budget.work(marker));
        if body.is_empty() {
            budget.whole(&mut md, "(empty sheet)");
        } else {
            md.push_str(budget.work(body));
        }
        md.push('\n');
    }
    if total > caps.max_pages {
        warnings.push(page_cap_warning(caps.max_pages, total, "sheets"));
    }
    let mut doc = Doc::new(md, total);
    doc.warnings = warnings;
    Ok(doc)
}
