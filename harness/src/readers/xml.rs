//! A bounded XML scanner for the parts of ZIP-based documents. One forward
//! pass gives open tags, close tags and text; each source byte is looked at
//! a fixed number of times. Comments, processing instructions and the
//! document type are skipped. Entities declared in a document type are not
//! expanded: only the five XML entities and character references decode, so
//! an entity cannot grow the text. Close tags must match their open tags,
//! and nesting stops at [`MAX_DEPTH`]. A part that a cap cut may end inside
//! a construct; any other early end is an error.
use super::{Budget, scanned};
use std::borrow::Cow;
use std::cell::Cell;
use std::rc::Rc;

/// Deepest element nesting in one part.
pub(super) const MAX_DEPTH: usize = 512;
/// Longest attribute value, in source bytes, that [`Tag::attr`] gives. Names,
/// numbers and part paths are much shorter; a reader that keeps a long value
/// as text takes its start with [`Tag::attr_prefix`].
pub(super) const MAX_ATTR: usize = 4096;
/// Longest part path, in bytes: the longest name a ZIP entry can have.
pub(super) const MAX_PATH: usize = 65_535;
/// Longest image alt text, in chars.
pub(super) const MAX_ALT_CHARS: usize = 200;

pub(super) enum Event<'a> {
    Open(Tag<'a>),
    /// The qualified name of a close tag, also given after an empty tag.
    Close(&'a str),
    /// Character data as written; decode it with [`unescape`].
    Text(&'a str),
    /// A CDATA section, which needs no decoding.
    Cdata(&'a str),
}

pub(super) struct Tag<'a> {
    /// The qualified name, for example `w:p`.
    pub(super) name: &'a str,
    attrs: &'a str,
    /// Set when [`Tag::attr`] left out a value longer than [`MAX_ATTR`].
    long: Rc<Cell<bool>>,
}

impl<'a> Tag<'a> {
    /// The name without its prefix.
    pub(super) fn local(&self) -> &'a str {
        local(self.name)
    }

    /// The value of the attribute `name`, decoded. A name with no prefix
    /// matches that name exactly first, then any prefixed attribute with
    /// that local name (Office files write `w:val`; others write `val`).
    /// A value longer than [`MAX_ATTR`] source bytes is not decoded or
    /// copied: it reads as absent, and [`Scanner::finish`] reports the
    /// attribute cap.
    pub(super) fn attr(&self, name: &str) -> Result<Option<Cow<'a, str>>, String> {
        match self.raw(name)? {
            Some(raw) if raw.len() > MAX_ATTR => {
                self.long.set(true);
                Ok(None)
            }
            raw => raw.map(unescape).transpose(),
        }
    }

    /// The start of the attribute `name` that decodes to at most `max`
    /// bytes, and true when that start is the whole value. Only that start
    /// is decoded.
    pub(super) fn attr_prefix(&self, name: &str, max: usize) -> Result<Option<(Cow<'a, str>, bool)>, String> {
        self.raw(name)?.map(|raw| decode_max(raw, max)).transpose()
    }

    /// The attribute `name` as a part path, decoded, up to [`MAX_PATH`]
    /// bytes. A longer path names no part of a ZIP archive: an error.
    pub(super) fn path(&self, name: &str) -> Result<Option<Cow<'a, str>>, String> {
        match self.attr_prefix(name, MAX_PATH)? {
            Some((_, false)) => Err(format!("has a part path longer than {MAX_PATH} bytes, the longest ZIP entry name")),
            value => Ok(value.map(|(v, _)| v)),
        }
    }

    /// The alt text of an image: the first of the attributes `names` that is
    /// not blank. Text after [`MAX_ALT_CHARS`] chars is left out and the
    /// alt text ends in `...`; only its start is decoded.
    pub(super) fn alt_text(&self, names: &[&str]) -> Result<String, String> {
        for name in names {
            if let Some((value, _)) = self.attr_prefix(name, (MAX_ALT_CHARS + 1) * 4)?
                && !value.trim().is_empty()
            {
                let mut alt: String = value.chars().take(MAX_ALT_CHARS).collect();
                if value.chars().nth(MAX_ALT_CHARS).is_some() {
                    alt.push_str("...");
                }
                return Ok(alt);
            }
        }
        Ok(String::new())
    }

    /// The raw value of the attribute `name` (see [`Tag::attr`]).
    fn raw(&self, name: &str) -> Result<Option<&'a str>, String> {
        let mut by_local = None;
        let mut rest = self.attrs;
        while let Some((key, value, next)) = attribute(rest)? {
            if key == name {
                return Ok(Some(value));
            }
            if by_local.is_none() && !name.contains(':') && local(key) == name && !key.starts_with("xmlns") {
                by_local = Some(value);
            }
            rest = next;
        }
        Ok(by_local)
    }

    /// [`Tag::attr`] as an `OOXML` on/off value: absent is `None`; present
    /// with no value, `1`, `true` or `on` is true.
    pub(super) fn flag(&self, name: &str) -> Result<Option<bool>, String> {
        Ok(self.attr(name)?.map(|v| !matches!(&*v, "0" | "false" | "off" | "none")))
    }
}

/// The part of a qualified name after its prefix.
pub(super) fn local(name: &str) -> &str {
    name.rsplit_once(':').map_or(name, |(_, l)| l)
}

/// The first attribute in `s`: its name, raw value and the rest.
fn attribute(s: &str) -> Result<Option<(&str, &str, &str)>, String> {
    let s = s.trim_start_matches(is_space);
    if s.is_empty() {
        return Ok(None);
    }
    let eq = s.find('=').ok_or("an attribute has no value")?;
    let name = s[..eq].trim_end_matches(is_space);
    if name.is_empty() || name.contains(is_space) {
        return Err("an attribute name is not valid".into());
    }
    let v = s[eq + 1..].trim_start_matches(is_space);
    let quote = v.chars().next().filter(|&c| c == '"' || c == '\'').ok_or("an attribute value has no quotes")?;
    let end = v[1..].find(quote).ok_or("an attribute value has no closing quote")?;
    Ok(Some((name, &v[1..1 + end], &v[end + 2..])))
}

fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

pub(super) struct Scanner<'a> {
    src: &'a str,
    at: usize,
    /// The names of the open elements.
    open: Vec<&'a str>,
    /// The close event owed for an empty tag.
    pending: Option<&'a str>,
    /// The part was cut at a cap, so it may end inside a construct.
    cut: bool,
    /// Shared with each [`Tag`]: an attribute value was left out.
    long: Rc<Cell<bool>>,
    /// The last text or CDATA event ran to the end of a cut part.
    text_cut: bool,
}

impl<'a> Scanner<'a> {
    pub(super) fn new(src: &'a str, cut: bool) -> Self {
        Self { src, at: 0, open: Vec::new(), pending: None, cut, long: Rc::default(), text_cut: false }
    }

    /// The next event, `None` at the end, or why the XML is not well formed.
    pub(super) fn next(&mut self) -> Result<Option<Event<'a>>, String> {
        if let Some(name) = self.pending.take() {
            return Ok(Some(Event::Close(name)));
        }
        loop {
            let rest = &self.src[self.at..];
            if rest.is_empty() {
                return Ok(self.close_cut());
            }
            if !rest.starts_with('<') {
                let n = rest.find('<').unwrap_or(rest.len());
                self.advance(n);
                let mut text = &rest[..n];
                // A cut part may end inside a reference; the text before it stays.
                self.text_cut = self.cut && n == rest.len();
                if self.text_cut
                    && let Some(amp) = text.rfind('&')
                    && !text[amp..].contains(';')
                {
                    text = &text[..amp];
                }
                return Ok(Some(Event::Text(text)));
            }
            let (skip, end) = if rest.starts_with("<!--") {
                (4, "-->")
            } else if rest.starts_with("<?") {
                (2, "?>")
            } else if rest.starts_with("<![CDATA[") {
                let Some(n) = self.through(9, "]]>") else {
                    // A cut part keeps the start of its last CDATA section;
                    // a `]` at the end may start the cut `]]>`.
                    if !self.cut {
                        return self.early();
                    }
                    self.advance(rest.len());
                    self.text_cut = true;
                    let text = &rest[9..];
                    return Ok(Some(Event::Cdata(text.strip_suffix("]]").or_else(|| text.strip_suffix(']')).unwrap_or(text))));
                };
                return Ok(Some(Event::Cdata(&rest[9..n - 3])));
            } else if rest.starts_with("<!") {
                let Some(n) = doctype(rest) else { return self.early() };
                self.advance(n);
                continue;
            } else {
                return self.tag();
            };
            if self.through(skip, end).is_none() {
                return self.early();
            }
        }
    }

    /// Moves past the first `end` after `skip` bytes; the bytes moved.
    fn through(&mut self, skip: usize, end: &str) -> Option<usize> {
        let rest = &self.src[self.at..];
        let n = rest.get(skip..)?.find(end)? + skip + end.len();
        self.advance(n);
        Some(n)
    }

    fn advance(&mut self, n: usize) {
        scanned(n);
        self.at += n;
    }

    /// The end of the part inside a construct: the end of a cut part, or an error.
    fn early(&mut self) -> Result<Option<Event<'a>>, String> {
        if self.cut {
            self.at = self.src.len();
            return Ok(self.close_cut());
        }
        Err(format!("ends inside markup at byte {}", self.at))
    }

    /// True when the last text or CDATA event ran to the end of a cut part:
    /// its end is missing, so a number or an index in it is not whole.
    pub(super) fn text_cut(&self) -> bool {
        self.text_cut
    }

    /// At the end of a cut part: a close event for each element that is
    /// still open, innermost first, so a reader keeps the blocks that the cut
    /// left open. At the end of a whole part: nothing.
    fn close_cut(&mut self) -> Option<Event<'a>> {
        if self.cut { self.open.pop().map(Event::Close) } else { None }
    }

    fn tag(&mut self) -> Result<Option<Event<'a>>, String> {
        let rest = &self.src[self.at..];
        // The tag ends at the first `>` outside a quoted value.
        let mut quote = None;
        let Some(end) = rest.bytes().position(|b| match quote {
            Some(q) => {
                if b == q {
                    quote = None;
                }
                false
            }
            None => {
                if b == b'"' || b == b'\'' {
                    quote = Some(b);
                }
                b == b'>'
            }
        }) else {
            return self.early();
        };
        let at = self.at;
        self.advance(end + 1);
        let bad = |why: &str| Err(format!("{why} at byte {at}"));
        let body = &rest[1..end];
        if let Some(name) = body.strip_prefix('/') {
            let name = name.trim_end_matches(is_space);
            return match self.open.pop() {
                Some(open) if open == name => Ok(Some(Event::Close(name))),
                Some(open) => bad(&format!("the close tag </{name}> does not match <{open}>")),
                None => bad(&format!("the close tag </{name}> has no open tag")),
            };
        }
        let (body, empty) = match body.strip_suffix('/') {
            Some(b) => (b, true),
            None => (body, false),
        };
        let n = body.find(is_space).unwrap_or(body.len());
        let (name, attrs) = body.split_at(n);
        if name.is_empty() || name.contains(['<', '&', '"', '\'', '=']) {
            return bad("a tag name is not valid");
        }
        // Check every attribute once, so later lookups cannot fail.
        let mut check = attrs;
        while let Some((_, value, next)) = attribute(check).map_err(|e| format!("{e} at byte {at}"))? {
            check_references(value).map_err(|e| format!("{e} at byte {at}"))?;
            check = next;
        }
        if empty {
            self.pending = Some(name);
        } else {
            if self.open.len() == MAX_DEPTH {
                return bad(&format!("elements nest deeper than {MAX_DEPTH} levels"));
            }
            self.open.push(name);
        }
        Ok(Some(Event::Open(Tag { name, attrs, long: Rc::clone(&self.long) })))
    }

    /// Checks the end of the part: every element closed, unless it was cut.
    /// A caller that stopped before the end (at a cap) is not checked; the
    /// cap warning says the rest was not read. An attribute value that
    /// [`Tag::attr`] left out goes on `budget` as the attribute cap.
    pub(super) fn finish(&self, budget: &mut Budget) -> Result<(), String> {
        if self.long.get() {
            budget.long_attr = true;
        }
        match self.open.last() {
            Some(name) if !self.cut && self.at == self.src.len() => Err(format!("ends inside <{name}>")),
            _ => Ok(()),
        }
    }
}

/// The length of a `<!...>` declaration, internal subset and quotes included.
fn doctype(s: &str) -> Option<usize> {
    let (mut depth, mut quote) = (0usize, None);
    for (i, b) in s.bytes().enumerate().skip(2) {
        match (quote, b) {
            (Some(q), _) if b == q => quote = None,
            (Some(_), _) => {}
            (None, b'"' | b'\'') => quote = Some(b),
            (None, b'[') => depth += 1,
            (None, b']') => depth = depth.saturating_sub(1),
            (None, b'>') if depth == 0 => return Some(i + 1),
            _ => {}
        }
    }
    None
}

/// The longest reference, `&` to `;`, that [`unescape`] accepts.
pub(super) const MAX_REFERENCE: usize = 13;

/// [`unescape`] of only the start of `raw`: at most `max` bytes and one more
/// character, so a reader that keeps `max` bytes sees that the text goes on.
pub(super) fn unescape_prefix(raw: &str, max: usize) -> Result<Cow<'_, str>, String> {
    decode_max(raw, max.saturating_add(4)).map(|(text, _)| text)
}

/// [`unescape`] of the start of `raw` that gives at most `max` bytes, and
/// true when that start is all of `raw`. Storage is at most `max` bytes; the
/// references after the end are not decoded.
fn decode_max(raw: &str, max: usize) -> Result<(Cow<'_, str>, bool), String> {
    let first = raw.find('&').unwrap_or(raw.len());
    if first >= max.min(raw.len()) {
        let end = floor_char(raw, max);
        return Ok((Cow::Borrowed(&raw[..end]), end == raw.len()));
    }
    let mut out = String::with_capacity(raw.len().min(max));
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        if at > max - out.len() {
            break;
        }
        out.push_str(&rest[..at]);
        let semi = rest[at..].find(';').filter(|&n| n < MAX_REFERENCE).ok_or("an & starts no entity")?;
        let c = reference(&rest[at + 1..at + semi])?;
        if c.len_utf8() > max - out.len() {
            return Ok((Cow::Owned(out), false));
        }
        out.push(c);
        rest = &rest[at + semi + 1..];
    }
    let end = floor_char(rest, max - out.len());
    out.push_str(&rest[..end]);
    Ok((Cow::Owned(out), end == rest.len()))
}

/// The largest char boundary of `s` at or below `max`.
fn floor_char(s: &str, max: usize) -> usize {
    let mut end = s.len().min(max);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Checks the references in `raw` as [`unescape`] does, with no copy of the
/// decoded text: the scanner checks each attribute that no reader asks for.
fn check_references(raw: &str) -> Result<(), String> {
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        let semi = rest[at..].find(';').filter(|&n| n < MAX_REFERENCE).ok_or("an & starts no entity")?;
        reference(&rest[at + 1..at + semi])?;
        rest = &rest[at + semi + 1..];
    }
    Ok(())
}

/// The character that the reference `&name;` stands for.
fn reference(name: &str) -> Result<char, String> {
    Ok(match name {
        "lt" => '<',
        "gt" => '>',
        "amp" => '&',
        "quot" => '"',
        "apos" => '\'',
        _ => {
            let code = match name.strip_prefix('#') {
                Some(hex) if hex.starts_with(['x', 'X']) => u32::from_str_radix(&hex[1..], 16).ok(),
                Some(dec) => dec.parse().ok(),
                None => None,
            };
            code.and_then(char::from_u32).filter(|&c| c != '\0').ok_or_else(|| format!("the entity &{name}; is not defined"))?
        }
    })
}

/// Decodes the five XML entities and character references. Any other
/// entity is an error: no entity can expand into more text.
pub(super) fn unescape(raw: &str) -> Result<Cow<'_, str>, String> {
    if !raw.contains('&') {
        return Ok(Cow::Borrowed(raw));
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let semi = rest[at..].find(';').filter(|&n| n < MAX_REFERENCE).ok_or("an & starts no entity")?;
        out.push(reference(&rest[at + 1..at + semi])?);
        rest = &rest[at + semi + 1..];
    }
    out.push_str(rest);
    Ok(Cow::Owned(out))
}
