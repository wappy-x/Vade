//! HTML and XHTML to Markdown: headings, paragraphs, lists, links, emphasis,
//! code, quotes, tables and images (alt text). Scripts, styles and the
//! document head are dropped. The parser is tolerant: unclosed paragraphs,
//! cells and list items close where a browser would close them. Output and
//! work stay within a [`Budget`]; quotes, tables and lists nest at most
//! [`MAX_NESTING`] levels deep.
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, fence, table_in, text::decode};

pub(super) struct HtmlReader;

/// Deepest nesting of quotes, code blocks, links, tables and cells, and of
/// list indents. Deeper elements read as their parent's text.
pub const MAX_NESTING: usize = 32;
/// The largest list number Markdown accepts (nine digits).
const MAX_LIST_NUMBER: u32 = 999_999_999;

impl Reader for HtmlReader {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format> {
        if name.is_empty() {
            let head = String::from_utf8_lossy(&bytes[..bytes.len().min(512)]).to_ascii_lowercase();
            let head = head.trim_start_matches('\u{feff}').trim_start();
            return (head.starts_with("<!doctype html") || head.starts_with("<html")).then_some(Format::Html);
        }
        matches!(extension(name).as_str(), "html" | "htm" | "xhtml" | "xht" | "shtml").then_some(Format::Html)
    }

    fn to_markdown(&self, bytes: &[u8], _: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let (text, note) = decode(bytes).ok_or(ReadError::Binary)?;
        let mut budget = Budget::new(caps);
        let (markdown, dropped) = convert(&text, &mut budget);
        let mut doc = Doc::new(markdown, 1);
        doc.warnings.extend(note.map(str::to_owned));
        if dropped > 0 {
            doc.warnings.push(format!("{dropped} script, style or embedded blocks"));
        }
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

/// HTML to Markdown, for EPUB chapters and other XHTML parts, within the
/// default caps.
pub fn to_markdown(html: &str) -> String {
    to_markdown_in(html, &mut Budget::new(&Caps::default()))
}

/// [`to_markdown`] within `budget`; it stops where a cap is reached.
pub fn to_markdown_in(html: &str, budget: &mut Budget) -> String {
    convert(html, budget).0
}

enum Frame {
    Root,
    Link(String),
    Quote,
    Pre(String),
    /// A table cell and the columns it spans.
    Cell(usize),
    Table(Vec<Vec<String>>),
}

/// The slot of a frame kind in `Out::unpushed`, from the element name.
fn slot(name: &str) -> Option<usize> {
    ["a", "blockquote", "pre", "td", "table"].iter().position(|&n| n == name || (n == "td" && name == "th"))
}

struct Out<'b> {
    frames: Vec<(Frame, String)>,
    /// Elements past [`MAX_NESTING`] that read as text, by [`slot`], so that
    /// their close tags do not close an outer element. They are all inside the
    /// top frame, so a pop clears them.
    unpushed: [usize; 5],
    lists: Vec<List>,
    /// The number of open lists and of lists past the cap when each frame
    /// above the root opened. A list left open in a frame ends with the frame.
    list_marks: Vec<(usize, usize)>,
    /// List elements past [`MAX_NESTING`]; their close tags end no kept list.
    lists_past: usize,
    /// The frame depth and text length just after a list item marker, so a
    /// paragraph that starts the item stays on the marker line.
    marker: Option<(usize, usize)>,
    dropped: usize,
    budget: &'b mut Budget,
}

/// One open list: the next number of an ordered list, the indent of its
/// items, and the column where the text of its last item starts, so a
/// nested list lines up under that text as CommonMark needs.
struct List {
    next: Option<u32>,
    indent: usize,
    content: usize,
}

/// Elements whose content is never text for a reader.
const DROP: &[&str] = &["script", "style", "noscript", "template", "head", "svg", "math", "iframe", "object", "canvas", "select"];
const BLOCKS: &[&str] = &[
    "p", "div", "section", "article", "header", "footer", "main", "nav", "aside", "figure", "figcaption", "address",
    "details", "summary", "dl", "dt", "dd", "form", "fieldset", "center",
];

fn convert(html: &str, budget: &mut Budget) -> (String, usize) {
    let mut out = Out {
        frames: vec![(Frame::Root, String::new())],
        unpushed: [0; 5],
        lists: Vec::new(),
        list_marks: Vec::new(),
        lists_past: 0,
        marker: None,
        dropped: 0,
        budget,
    };
    let b = html.as_bytes();
    let mut i = 0;
    // Set when no `</head` follows: later head tags then skip the search.
    let mut head_unclosed = false;
    while i < b.len() && !out.budget.full() {
        if b[i] != b'<' {
            // A long run converts in parts no larger than the room left, cut
            // before a reference that the part would split. The scan for `<`
            // stops at the end of the part, so no byte is scanned twice.
            let limit = part_end(html, i, b.len(), out.budget.room());
            let mut end = html[i..limit].find('<').map_or(limit, |n| i + n);
            super::scanned(end - i);
            if end < b.len() && b[end] != b'<' {
                let mut tail = end.saturating_sub(40).max(i);
                while !html.is_char_boundary(tail) {
                    tail += 1;
                }
                end = html[tail..end].rfind('&').map(|n| tail + n).filter(|&n| n > i).unwrap_or(end);
            }
            out.text(&entities(&html[i..end], false));
            i = end;
            continue;
        }
        let rest = &html[i..];
        if rest.starts_with("<!--") {
            i += rest.find("-->").map_or(rest.len(), |n| n + 3);
            continue;
        }
        if rest.starts_with("<![CDATA[") {
            let end = rest.find("]]>").unwrap_or(rest.len());
            super::scanned(end);
            let mut at = 9;
            while at < end && !out.budget.full() {
                let to = part_end(rest, at, end, out.budget.room());
                out.text(&rest[at..to]);
                at = to;
            }
            i += (end + 3).min(rest.len());
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            i += rest.find('>').map_or(rest.len(), |n| n + 1);
            continue;
        }
        let Some((tag, len)) = parse_tag(rest, out.budget.room() / 4) else {
            out.text("<");
            i += 1;
            continue;
        };
        i += len;
        if !tag.close && !tag.self_closing && DROP.contains(&tag.name.as_str()) {
            // Skip to the matching close tag (see `find_end`). A head with no
            // close tag is ignored, not dropped to the end.
            let head = tag.name == "head";
            let close = if tag.qname.is_empty() { &tag.name } else { &tag.qname };
            let found = if head && head_unclosed { None } else { find_end(&html[i..], &tag.name, close) };
            match found {
                // A quoted value in the close tag may hold `>`.
                Some(n) => i = tag_end(b, i + n + 2).map_or(b.len(), |gt| gt + 1),
                None if head => {
                    head_unclosed = true;
                    continue;
                }
                None => i = html.len(),
            }
            out.dropped += usize::from(tag.name != "head");
            continue;
        }
        out.tag(&tag);
        if tag.cut {
            out.budget.cut();
        }
    }
    // The text in the open blocks is counted; closing them adds markup only,
    // so a cap that stopped the scan does not drop them. [`super::read`] cuts
    // the result to the char cap.
    let counts = std::mem::replace(&mut out.budget.text_counts, false);
    while out.frames.len() > 1 {
        out.pop();
    }
    out.budget.text_counts = counts;
    let body = out.frames.pop().unwrap().1;
    (tidy(&body), out.dropped)
}

/// The end of a part of `text[from..end]` with at most `room` bytes (at
/// least one char), on a char boundary.
fn part_end(text: &str, from: usize, end: usize, room: usize) -> usize {
    let mut to = end.min(from.saturating_add(room.max(64)));
    while !text.is_char_boundary(to) {
        to += 1;
    }
    to
}

struct Tag {
    /// The element name, lowercase, without a namespace prefix.
    name: String,
    /// The name as written with its prefix (`svg:svg`), lowercase; empty
    /// when it is too long.
    qname: String,
    close: bool,
    self_closing: bool,
    /// The values of the attributes in [`ATTRS`] that the element uses, by
    /// position; the first one of a name wins.
    attrs: [Option<String>; 6],
    /// True when a value was cut to the limit and lost text.
    cut: bool,
}

/// The attributes a reader uses. A tag keeps no others.
const ATTRS: [&str; 6] = ["start", "href", "alt", "src", "class", "colspan"];

/// True when the element `name` uses the attribute `ATTRS[n]`. Other values
/// are not decoded, so they use no room.
fn uses(name: &str, n: usize) -> bool {
    match ATTRS[n] {
        "start" => name == "ol",
        "href" => name == "a",
        "alt" | "src" => name == "img",
        "class" => name == "pre",
        _ => matches!(name, "td" | "th"),
    }
}

/// The most columns one table cell spans, as in the DOCX reader.
const MAX_SPAN: usize = 64;
/// Longer element names are not names this reader knows.
const MAX_NAME: usize = 32;

impl Tag {
    fn attr(&self, name: &str) -> Option<&str> {
        ATTRS.iter().position(|&n| n == name).and_then(|n| self.attrs[n].as_deref())
    }
}

/// The tag at the start of `s` and its length. The values of the attributes
/// the element uses are decoded up to `limit` bytes (a quarter of the room
/// left), so a tag holds at most two values of that size and a link target
/// leaves room for its text. A value that does not fit sets `cut`, and the
/// reader stops there.
fn parse_tag(s: &str, limit: usize) -> Option<(Tag, usize)> {
    let b = s.as_bytes();
    let mut i = 1;
    let close = b.get(i) == Some(&b'/');
    i += usize::from(close);
    let start = i;
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'-' || b[i] == b':') {
        i += 1;
    }
    if i == start || !b[start].is_ascii_alphabetic() {
        return None;
    }
    let full = &s[start..i];
    let local = full.rsplit(':').next().unwrap_or("");
    let name = if local.len() > MAX_NAME { String::new() } else { local.to_ascii_lowercase() };
    // The name with its prefix, for the close tag of a dropped element.
    let qname = if full.len() > 2 * MAX_NAME { String::new() } else { full.to_ascii_lowercase() };
    let mut attrs: [Option<String>; 6] = Default::default();
    let mut cut = false;
    let mut self_closing = false;
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        match b.get(i) {
            None => return Some((Tag { name, qname, close, self_closing, attrs, cut }, b.len())),
            Some(b'>') => return Some((Tag { name, qname, close, self_closing, attrs, cut }, i + 1)),
            Some(b'/') => {
                self_closing = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        let k = i;
        while i < b.len() && !b[i].is_ascii_whitespace() && !matches!(b[i], b'=' | b'>' | b'/') {
            i += 1;
        }
        let key = ATTRS.iter().position(|n| n.eq_ignore_ascii_case(&s[k..i])).filter(|&n| attrs[n].is_none() && uses(&name, n));
        let mut raw = "";
        // Spaces may stand on both sides of `=`.
        let eq = i + b[i..].iter().take_while(|c| c.is_ascii_whitespace()).count();
        if b.get(eq) == Some(&b'=') {
            i = eq + 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            match b.get(i) {
                Some(&q @ (b'"' | b'\'')) => {
                    let end = s[i + 1..].find(q as char).map_or(b.len(), |n| i + 1 + n);
                    raw = &s[i + 1..end];
                    i = (end + 1).min(b.len());
                }
                _ => {
                    let v = i;
                    while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'>' {
                        i += 1;
                    }
                    raw = &s[v..i];
                }
            }
        }
        if i == k {
            i += 1;
        }
        if let Some(n) = key {
            let (value, lost) = decode_refs(raw, true, limit);
            cut |= lost;
            attrs[n] = Some(value);
        }
    }
}

impl Out<'_> {
    fn buf(&mut self) -> &mut String {
        &mut self.frames.last_mut().unwrap().1
    }

    /// Appends new output text to the current block, within the budget.
    fn emit(&mut self, text: &str) {
        let kept = self.budget.text(text);
        self.frames.last_mut().unwrap().1.push_str(kept);
    }

    /// Appends text that the budget already counted, or markup around it,
    /// within the work cap only.
    fn emit_work(&mut self, text: &str) {
        let kept = self.budget.work(text);
        self.frames.last_mut().unwrap().1.push_str(kept);
    }

    fn in_pre(&self) -> bool {
        matches!(self.frames.last(), Some((Frame::Pre(_), _)))
    }

    fn in_cell(&self) -> bool {
        matches!(self.frames.last(), Some((Frame::Cell(_), _)))
    }

    /// [`Out::block`] for a paragraph or other block element, except right
    /// after a list item marker: `<li><p>text` reads `- text`.
    fn item_block(&mut self) {
        let at = (self.frames.len(), self.buf().len());
        if self.marker != Some(at) {
            self.block();
        }
    }

    fn text(&mut self, text: &str) {
        if self.in_pre() {
            self.emit(text);
            return;
        }
        if matches!(self.frames.last(), Some((Frame::Table(_), _))) && text.trim().is_empty() {
            return;
        }
        // Runs of white space become one space; none starts a line or doubles one.
        let buf = self.buf();
        let mut quiet = buf.is_empty() || buf.ends_with([' ', '\n']);
        let mut piece = String::with_capacity(text.len());
        let mut space = false;
        for c in text.chars() {
            if c.is_whitespace() && c != '\u{a0}' {
                space = true;
                continue;
            }
            if space && !quiet {
                piece.push(' ');
            }
            let c = if c == '\u{a0}' { ' ' } else { c };
            piece.push(c);
            (space, quiet) = (false, c == ' ');
        }
        if space && !quiet {
            piece.push(' ');
        }
        self.emit(&piece);
    }

    /// Ends the current line. Inside a list the break is single; elsewhere a
    /// blank line separates blocks. Inside a cell a block is a `<br>`.
    fn block(&mut self) {
        self.marker = None;
        if self.in_cell() {
            let buf = self.buf();
            trim_end_spaces(buf);
            if !buf.is_empty() && !buf.ends_with("<br>") {
                self.emit("<br>");
            }
            return;
        }
        let blank = self.lists.is_empty();
        let buf = self.buf();
        trim_end_spaces(buf);
        if buf.is_empty() {
            return;
        }
        let breaks = if buf.ends_with("\n\n") || (!blank && buf.ends_with('\n')) {
            ""
        } else if buf.ends_with('\n') || !blank {
            "\n"
        } else {
            "\n\n"
        };
        self.emit(breaks);
    }

    /// Opens a nested frame; past [`MAX_NESTING`] the element reads as text.
    fn push(&mut self, frame: Frame) {
        if self.frames.len() <= MAX_NESTING {
            if self.budget.reserve(payload(&frame)) {
                self.frames.push((frame, String::new()));
                self.list_marks.push((self.lists.len(), self.lists_past));
            }
            return;
        }
        let name = match frame {
            Frame::Link(_) => "a",
            Frame::Quote => "blockquote",
            Frame::Pre(_) => "pre",
            Frame::Cell(_) => "td",
            Frame::Table(_) => "table",
            Frame::Root => return,
        };
        if let Some(n) = slot(name) {
            self.unpushed[n] = self.unpushed[n].saturating_add(1);
        }
    }

    fn pop(&mut self) {
        let Some((frame, text)) = self.frames.pop() else { return };
        self.unpushed = [0; 5];
        // A list that the frame left open ends with it (`<td><ul><li>x</td>`).
        if !matches!(frame, Frame::Root) {
            let (open, past) = self.list_marks.pop().unwrap_or((0, 0));
            self.lists.truncate(open);
            self.lists_past = past;
        }
        // The block's text moves into the parent; its bytes count there.
        let held = text.len() + payload(&frame);
        if !matches!(frame, Frame::Root) {
            self.budget.release(held);
        }
        match frame {
            Frame::Root => self.frames.push((Frame::Root, text)),
            Frame::Link(href) => {
                let label = text.trim();
                let link = if href.is_empty() || href.starts_with('#') || href.starts_with("javascript:") || label.is_empty() {
                    label.to_owned()
                } else {
                    format!("[{label}]({})", href.replace(' ', "%20").replace(')', "%29"))
                };
                self.emit_work(&link);
            }
            Frame::Quote => {
                self.block();
                let quoted: String = tidy(&text).lines().map(|l| if l.is_empty() { ">\n".to_owned() } else { format!("> {l}\n") }).collect();
                self.emit_work(&quoted);
                self.block();
            }
            Frame::Pre(lang) => {
                let body = text.trim_matches('\n');
                if !body.trim().is_empty() {
                    self.block();
                    self.emit_work(&fence(&lang, body));
                    self.block();
                }
            }
            Frame::Cell(span) => {
                let mut text = text.trim().to_owned();
                while text.ends_with("<br>") {
                    text.truncate(text.len() - 4);
                }
                let value = cell(&text);
                let kept = self.budget.work(&value).len();
                let value = value[..kept].to_owned();
                if let Some((Frame::Table(rows), _)) = self.frames.last_mut() {
                    let slots = CELL + if rows.is_empty() { ROW } else { 0 };
                    if !self.budget.reserve(slots) {
                        return;
                    }
                    if rows.is_empty() {
                        rows.push(Vec::new());
                    }
                    let row = rows.last_mut().unwrap();
                    row.push(value);
                    // A cell that spans columns keeps the slots of the columns
                    // it spans, as many as fit.
                    for _ in 1..span {
                        if !self.budget.reserve(CELL) {
                            break;
                        }
                        row.push(String::new());
                    }
                } else {
                    self.buf().push_str(&value);
                }
            }
            Frame::Table(rows) => {
                let caption = text.trim().to_owned();
                self.block();
                if !caption.is_empty() {
                    self.emit_work(&caption);
                    self.block();
                }
                let rows: Vec<Vec<String>> = rows.into_iter().filter(|r| !r.is_empty()).collect();
                if rows.len() == 1 && rows[0].len() == 1 {
                    // A one-cell layout table is just text.
                    self.emit_work(&rows[0][0]);
                } else {
                    // The cells were counted as text once; the table counts as work.
                    let counts = std::mem::replace(&mut self.budget.text_counts, false);
                    let md = table_in(&rows, self.budget);
                    self.budget.text_counts = counts;
                    self.buf().push_str(md.trim_end());
                }
                self.block();
            }
        }
    }

    /// Pops frames up to and including the innermost one `is` matches, if any.
    fn close(&mut self, is: fn(&Frame) -> bool, stop: fn(&Frame) -> bool) -> bool {
        let Some(at) = self.frames.iter().rposition(|(f, _)| is(f) || stop(f)) else { return false };
        if !is(&self.frames[at].0) {
            return false;
        }
        while self.frames.len() > at {
            self.pop();
        }
        true
    }

    fn tag(&mut self, tag: &Tag) {
        let name = tag.name.as_str();
        if self.in_pre() && !(tag.close && name == "pre") {
            if name == "br" {
                self.emit("\n");
            }
            return;
        }
        if tag.close {
            self.close_tag(name);
        } else if tag.self_closing && !matches!(name, "br" | "hr" | "img") {
            // An XHTML empty element such as <blockquote/> opens and closes at
            // once. Empty emphasis, list items and headings print nothing.
            match name {
                "blockquote" | "pre" | "a" | "table" | "tr" | "td" | "th" | "ul" | "ol" => {
                    self.open_tag(tag);
                    self.close_tag(name);
                }
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => self.block(),
                n if BLOCKS.contains(&n) => self.block(),
                _ => {}
            }
        } else {
            self.open_tag(tag);
        }
    }

    fn close_tag(&mut self, name: &str) {
        if let Some(n) = slot(name).filter(|&n| self.unpushed[n] > 0) {
            self.unpushed[n] -= 1;
            return;
        }
        let is_table = |f: &Frame| matches!(f, Frame::Table(_));
        match name {
            "a" => {
                self.close(|f| matches!(f, Frame::Link(_)), is_table);
            }
            "blockquote" => {
                self.close(|f| matches!(f, Frame::Quote), is_table);
            }
            "pre" => {
                self.close(|f| matches!(f, Frame::Pre(_)), |_| false);
            }
            "td" | "th" => {
                self.close(|f| matches!(f, Frame::Cell(_)), is_table);
            }
            "table" => {
                self.close(is_table, |_| false);
            }
            "ul" | "ol" => {
                // A close tag ends only a list that opened in this frame.
                let (open, past) = self.list_marks.last().copied().unwrap_or((0, 0));
                if self.lists_past > past {
                    self.lists_past -= 1;
                } else if self.lists.len() > open {
                    self.lists.pop();
                }
                self.block();
                if self.lists.is_empty() {
                    self.block();
                }
            }
            "strong" | "b" => self.emit("**"),
            "em" | "i" | "cite" | "dfn" => self.emit("*"),
            "del" | "s" | "strike" => self.emit("~~"),
            "code" | "kbd" | "samp" | "tt" => self.emit("`"),
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "li" | "tr" => self.block(),
            n if BLOCKS.contains(&n) => self.block(),
            _ => {}
        }
    }

    fn open_tag(&mut self, tag: &Tag) {
        let name = tag.name.as_str();
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.block();
                if !self.in_cell() {
                    let level = (name.as_bytes()[1] - b'0') as usize;
                    self.emit(&format!("{} ", "#".repeat(level)));
                }
            }
            "br" => {
                if self.in_cell() {
                    self.emit("<br>");
                } else {
                    trim_end_spaces(self.buf());
                    self.emit("\n");
                }
            }
            "hr" => {
                self.block();
                self.emit("---");
                self.block();
            }
            "ul" | "ol" => {
                self.block();
                let start = tag.attr("start").map_or(1, list_start);
                if self.lists.len() > MAX_NESTING {
                    self.lists_past += 1;
                } else {
                    let indent = self.lists.last().map_or(0, |l| l.content);
                    self.lists.push(List { next: (name == "ol").then_some(start), indent, content: indent + 2 });
                }
            }
            "li" => {
                self.block();
                let (indent, marker) = match self.lists.last_mut() {
                    Some(list) => {
                        let marker = match &mut list.next {
                            Some(n) => {
                                let shown = *n;
                                *n = n.saturating_add(1).min(MAX_LIST_NUMBER);
                                format!("{shown}. ")
                            }
                            None => "- ".to_owned(),
                        };
                        list.content = list.indent + marker.len();
                        (list.indent, marker)
                    }
                    None => (0, "- ".to_owned()),
                };
                self.emit(&format!("{}{marker}", " ".repeat(indent)));
                self.marker = Some((self.frames.len(), self.buf().len()));
            }
            // A new link or cell ends an open one. When that one is past the cap,
            // the new one is too: the count stays.
            "a" if self.unpushed[0] > 0 => {}
            "td" | "th" if self.unpushed[3] > 0 => {}
            "a" => {
                self.close(|f| matches!(f, Frame::Link(_)), |f| matches!(f, Frame::Table(_) | Frame::Cell(_)));
                let href = tag.attr("href").unwrap_or("").trim().to_owned();
                self.push(Frame::Link(href));
            }
            "img" => {
                let alt = tag.attr("alt").unwrap_or("").trim().replace(['[', ']'], "");
                if !alt.is_empty() {
                    let src = tag.attr("src").unwrap_or("");
                    let image = if src.is_empty() || src.starts_with("data:") { format!("[image: {alt}]") } else { format!("![{alt}]({src})") };
                    self.emit(&image);
                }
            }
            "blockquote" => {
                self.block();
                self.push(Frame::Quote);
            }
            "pre" => {
                self.block();
                let lang = language(tag);
                self.push(Frame::Pre(lang));
            }
            "table" => {
                self.block();
                self.push(Frame::Table(Vec::new()));
            }
            "tr" => {
                self.unpushed[3] = 0;
                self.close(|f| matches!(f, Frame::Cell(_)), |f| matches!(f, Frame::Table(_)));
                if let Some((Frame::Table(rows), _)) = self.frames.last_mut()
                    && self.budget.reserve(ROW)
                {
                    rows.push(Vec::new());
                }
            }
            "td" | "th" => {
                self.close(|f| matches!(f, Frame::Cell(_)), |f| matches!(f, Frame::Table(_)));
                if matches!(self.frames.last(), Some((Frame::Table(_), _))) {
                    let span = tag.attr("colspan").and_then(|v| v.trim().parse().ok()).unwrap_or(1).clamp(1, MAX_SPAN);
                    self.push(Frame::Cell(span));
                }
            }
            "strong" | "b" => self.emit("**"),
            "em" | "i" | "cite" | "dfn" => self.emit("*"),
            "del" | "s" | "strike" => self.emit("~~"),
            "code" | "kbd" | "samp" | "tt" => self.emit("`"),
            n if BLOCKS.contains(&n) => self.item_block(),
            _ => {}
        }
    }
}

/// The storage of one table row and of one cell, apart from the cell text.
const ROW: usize = size_of::<Vec<String>>();
const CELL: usize = size_of::<String>();

/// The bytes a frame holds apart from its text: a link target, a fence
/// language, or the rows of a table with their cells.
fn payload(frame: &Frame) -> usize {
    match frame {
        Frame::Link(s) | Frame::Pre(s) => s.len(),
        Frame::Table(rows) => rows.iter().map(|r| ROW + r.iter().map(|c| CELL + c.len()).sum::<usize>()).sum(),
        Frame::Root | Frame::Quote | Frame::Cell(_) => 0,
    }
}

/// An `<ol start>` value as a Markdown list number, 0 to 999,999,999 (the
/// nine-digit limit). Larger numbers saturate, negative numbers give 0 and
/// other text gives 1.
fn list_start(value: &str) -> u32 {
    let value = value.trim();
    let digits = value.strip_prefix(['-', '+']).unwrap_or(value);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return 1;
    }
    if value.starts_with('-') {
        return 0;
    }
    let digits = digits.trim_start_matches('0');
    if digits.len() > 9 { MAX_LIST_NUMBER } else { digits.parse().unwrap_or(0) }
}

/// The fence language from `class="language-x"` or `class="lang-x"`.
fn language(tag: &Tag) -> String {
    let class = tag.attr("class").unwrap_or("");
    class
        .split_whitespace()
        .find_map(|c| c.strip_prefix("language-").or_else(|| c.strip_prefix("lang-")))
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '#' | '-' | '_'))
        .collect()
}

fn trim_end_spaces(buf: &mut String) {
    while buf.ends_with([' ', '\t']) {
        buf.pop();
    }
}

/// Trims each line end, keeps at most one blank line in a row, and trims
/// the whole text. Fenced blocks keep their lines.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    let mut fence: Option<String> = None;
    for line in text.lines() {
        if let Some(open) = &fence {
            out.push_str(line);
            out.push('\n');
            if line.trim_end() == open {
                fence = None;
            }
            continue;
        }
        let line = line.trim_end();
        if line.starts_with("```") {
            fence = Some(line.chars().take_while(|&c| c == '`').collect());
        }
        if line.is_empty() {
            blank += 1;
            if blank > 1 || out.is_empty() {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    let trimmed = out.trim_end().len();
    out.truncate(trimmed);
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Decodes character references: named (the common set and all of
/// Latin-1), decimal and hexadecimal. Unknown names stay as written. As in
/// the HTML standard, a name with no `;` decodes only when it is a legacy
/// name (Latin-1, `amp`, `lt`, `gt`, `quot`), and in an attribute (`attr`)
/// not when `=` follows it, so `?a=1&reg=2` stays a query string.
pub(super) fn entities(text: &str, attr: bool) -> String {
    decode_refs(text, attr, usize::MAX).0
}

/// [`entities`] that stops at `max` bytes of decoded text; true when text
/// was left out.
fn decode_refs(text: &str, attr: bool, max: usize) -> (String, bool) {
    let mut out = String::with_capacity(text.len().min(max));
    // Appends what fits of `piece`; false when some did not fit.
    let fits = |out: &mut String, piece: &str| {
        let mut end = piece.len().min(max - out.len());
        while !piece.is_char_boundary(end) {
            end -= 1;
        }
        out.push_str(&piece[..end]);
        end == piece.len()
    };
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        if !fits(&mut out, &rest[..at]) {
            return (out, true);
        }
        rest = &rest[at..];
        let end = rest[1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '#')).map_or(rest.len(), |n| n + 1);
        let name = &rest[1..end];
        let decoded = if let Some(num) = name.strip_prefix('#') {
            let value = match num.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => num.parse().ok(),
            };
            value.map(|v| match v {
                0 => '\u{fffd}',
                0x80..=0x9f => super::text::windows_1252(v as u8),
                v => char::from_u32(v).unwrap_or('\u{fffd}'),
            })
        } else if rest[end..].starts_with(';') || (legacy(name) && !(attr && rest[end..].starts_with('='))) {
            named(name)
        } else {
            None
        };
        let (piece, after) = match decoded {
            Some(c) if !name.is_empty() => (c, rest[end..].strip_prefix(';').unwrap_or(&rest[end..])),
            _ => ('&', &rest[1..]),
        };
        if !fits(&mut out, piece.encode_utf8(&mut [0; 4])) {
            return (out, true);
        }
        rest = after;
    }
    let all = fits(&mut out, rest);
    (out, !all)
}

/// Names that decode with no `;`.
fn legacy(name: &str) -> bool {
    LATIN1.contains(&name) || matches!(name, "amp" | "lt" | "gt" | "quot")
}

/// Latin-1, U+00A0 to U+00FF, in code point order.
const LATIN1: [&str; 96] = [
    "nbsp", "iexcl", "cent", "pound", "curren", "yen", "brvbar", "sect", "uml", "copy", "ordf", "laquo", "not", "shy", "reg",
    "macr", "deg", "plusmn", "sup2", "sup3", "acute", "micro", "para", "middot", "cedil", "sup1", "ordm", "raquo", "frac14",
    "frac12", "frac34", "iquest", "Agrave", "Aacute", "Acirc", "Atilde", "Auml", "Aring", "AElig", "Ccedil", "Egrave",
    "Eacute", "Ecirc", "Euml", "Igrave", "Iacute", "Icirc", "Iuml", "ETH", "Ntilde", "Ograve", "Oacute", "Ocirc", "Otilde",
    "Ouml", "times", "Oslash", "Ugrave", "Uacute", "Ucirc", "Uuml", "Yacute", "THORN", "szlig", "agrave", "aacute", "acirc",
    "atilde", "auml", "aring", "aelig", "ccedil", "egrave", "eacute", "ecirc", "euml", "igrave", "iacute", "icirc", "iuml",
    "eth", "ntilde", "ograve", "oacute", "ocirc", "otilde", "ouml", "divide", "oslash", "ugrave", "uacute", "ucirc", "uuml",
    "yacute", "thorn", "yuml",
];

fn named(name: &str) -> Option<char> {
    if let Some(i) = LATIN1.iter().position(|&n| n == name) {
        return char::from_u32(0xa0 + i as u32);
    }
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "ndash" => '\u{2013}',
        "mdash" => '\u{2014}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "sbquo" => '\u{201a}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "bdquo" => '\u{201e}',
        "dagger" => '\u{2020}',
        "Dagger" => '\u{2021}',
        "bull" => '\u{2022}',
        "hellip" => '\u{2026}',
        "permil" => '\u{2030}',
        "prime" => '\u{2032}',
        "lsaquo" => '\u{2039}',
        "rsaquo" => '\u{203a}',
        "euro" => '\u{20ac}',
        "trade" => '\u{2122}',
        "larr" => '\u{2190}',
        "uarr" => '\u{2191}',
        "rarr" => '\u{2192}',
        "darr" => '\u{2193}',
        "harr" => '\u{2194}',
        "rArr" => '\u{21d2}',
        "hArr" => '\u{21d4}',
        "minus" => '\u{2212}',
        "le" => '\u{2264}',
        "ge" => '\u{2265}',
        "ne" => '\u{2260}',
        "asymp" => '\u{2248}',
        "infin" => '\u{221e}',
        "sum" => '\u{2211}',
        "radic" => '\u{221a}',
        "alpha" => '\u{3b1}',
        "beta" => '\u{3b2}',
        "gamma" => '\u{3b3}',
        "delta" => '\u{3b4}',
        "pi" => '\u{3c0}',
        "sigma" => '\u{3c3}',
        "mu" => '\u{3bc}',
        "lambda" => '\u{3bb}',
        "theta" => '\u{3b8}',
        "OElig" => '\u{152}',
        "oelig" => '\u{153}',
        "Scaron" => '\u{160}',
        "scaron" => '\u{161}',
        "Yuml" => '\u{178}',
        "fnof" => '\u{192}',
        "circ" => '\u{2c6}',
        "tilde" => '\u{2dc}',
        "ensp" => '\u{2002}',
        "emsp" => '\u{2003}',
        "thinsp" => '\u{2009}',
        "zwnj" => '\u{200c}',
        "zwj" => '\u{200d}',
        "lrm" => '\u{200e}',
        "rlm" => '\u{200f}',
        "check" => '\u{2713}',
        _ => return None,
    })
}

/// Dropped elements that can hold an element of the same name. Script,
/// style and the other dropped elements are raw text: the first close tag
/// ends them.
const NESTED: &[&str] = &["template", "svg", "math", "object", "canvas", "select"];

/// The byte offset of the close tag `</qname` that ends a dropped element
/// whose content starts `hay`; `local` is its name without a prefix. In a
/// [`NESTED`] element each inner `<qname>` needs its own close tag; an empty
/// `<qname/>` needs none. Comments, CDATA sections, quoted attribute values
/// (in open and close tags) and the text of [`RAW`] elements hold no tags;
/// an empty `<title/>` or `<script/>` has no text.
/// One pass over `hay`.
fn find_end(hay: &str, local: &str, qname: &str) -> Option<usize> {
    if !NESTED.contains(&local) {
        return find_close(hay, qname);
    }
    let (h, n) = (hay.as_bytes(), qname.as_bytes());
    let (mut depth, mut at) = (0usize, 0);
    let found = loop {
        let Some(lt) = h[at..].iter().position(|&c| c == b'<').map(|p| at + p) else { break None };
        at = lt + 1;
        let rest = &hay[lt..];
        let skip = if rest.starts_with("<!--") {
            rest[4..].find("-->").map(|e| 4 + e + 3)
        } else if rest.starts_with("<![CDATA[") {
            rest[9..].find("]]>").map(|e| 9 + e + 3)
        } else if rest.starts_with("<!") || rest.starts_with("<?") {
            rest.find('>').map(|e| e + 1)
        } else {
            Some(0)
        };
        match skip {
            None => break None,
            Some(0) => {}
            Some(len) => {
                at = lt + len;
                continue;
            }
        }
        let close = h.get(lt + 1) == Some(&b'/');
        let from = lt + 1 + usize::from(close);
        if !h.get(from).is_some_and(u8::is_ascii_alphabetic) {
            continue;
        }
        let Some(gt) = tag_end(h, from) else { break None };
        at = gt + 1;
        let same = names_at(h, from, n);
        if close {
            if same {
                if depth == 0 {
                    break Some(lt);
                }
                depth -= 1;
            }
        } else if same {
            // An inner open tag counts unless it is empty.
            depth += usize::from(h[gt - 1] != b'/');
        } else if h[gt - 1] != b'/'
            && let Some(raw) = RAW.iter().find(|r| names_at(h, from, r.as_bytes()))
        {
            // An empty `<title/>` has no text; a quoted value in the close tag
            // may hold `>`.
            let Some(end) = close_at(&h[at..], raw.as_bytes()) else { break None };
            let Some(gt) = tag_end(h, at + end + 2) else { break None };
            at = gt + 1;
        }
    };
    super::scanned(found.unwrap_or(h.len()));
    found
}

/// Elements whose text holds no tags up to their close tag.
const RAW: &[&str] = &["script", "style", "textarea", "title", "xmp"];

/// The offset of the `>` that ends the tag whose name starts at `h[from]`.
/// A quoted attribute value may hold `>`.
fn tag_end(h: &[u8], from: usize) -> Option<usize> {
    let (mut i, mut value) = (from, false);
    while i < h.len() {
        match h[i] {
            b'>' => return Some(i),
            b'=' => value = true,
            q @ (b'"' | b'\'') if value => {
                i += 1 + h[i + 1..].iter().position(|&c| c == q)?;
                value = false;
            }
            c if c.is_ascii_whitespace() => {}
            _ => value = false,
        }
        i += 1;
    }
    None
}

/// True when the element name `n` (ASCII, lowercase) stands at `h[at..]`,
/// ignoring ASCII case, and ends there.
fn names_at(h: &[u8], at: usize, n: &[u8]) -> bool {
    h.get(at..at + n.len()).is_some_and(|s| s.eq_ignore_ascii_case(n))
        && h.get(at + n.len()).is_none_or(|&c| c == b'>' || c == b'/' || c.is_ascii_whitespace())
}

/// The byte offset of the close tag `</name` (ASCII, lowercase) in `hay`,
/// ignoring ASCII case. The name must end there: `</head` does not match
/// `</header>`.
fn find_close(hay: &str, name: &str) -> Option<usize> {
    let found = close_at(hay.as_bytes(), name.as_bytes());
    super::scanned(found.unwrap_or(hay.len()));
    found
}

/// [`find_close`] on bytes, not counted.
fn close_at(h: &[u8], n: &[u8]) -> Option<usize> {
    (0..h.len().saturating_sub(n.len() + 1)).find(|&i| h[i] == b'<' && h[i + 1] == b'/' && names_at(h, i + 2, n))
}
