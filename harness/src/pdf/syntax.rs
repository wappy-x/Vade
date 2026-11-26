//! PDF lexical syntax (ISO 32000-1, 7.2 and 7.3): objects in file bodies,
//! object streams and content streams. Malformed input gives `None`, never a
//! panic. Each byte the lexer looks at and each object it builds is charged
//! to the read's budget first; a stopped budget also gives `None`.
use super::budget::{Budget, Kind};
use std::collections::BTreeMap;

/// Objects one top-level object may hold, nested ones included.
pub(crate) const FILE_ITEMS: usize = 1 << 20;
/// The same bound for one content-stream operand: text arrays are short.
pub(crate) const CONTENT_ITEMS: usize = 1 << 16;
/// Bytes charged per object as cache weight, before its payload bytes.
const OBJECT_WEIGHT: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Object {
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    Str(Vec<u8>),
    Name(Vec<u8>),
    Array(Vec<Object>),
    Dict(Dict),
    Ref(u32, u16),
    /// A stream dictionary and the start and end of its raw bytes in the
    /// file. The bytes stay in the file until a charged decode reads them.
    Stream(Dict, usize, usize),
    /// A bare keyword: a content-stream operator, or `R`/`obj`/`endobj` text.
    Op(Vec<u8>),
}

pub(crate) type Dict = BTreeMap<Vec<u8>, Object>;

impl Object {
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match *self {
            Object::Int(i) => Some(i as f64),
            Object::Real(r) => Some(r),
            _ => None,
        }
    }
    pub(crate) fn as_int(&self) -> Option<i64> {
        match *self {
            Object::Int(i) => Some(i),
            Object::Real(r) if r.is_finite() => Some(r as i64),
            _ => None,
        }
    }
    pub(crate) fn as_name(&self) -> Option<&[u8]> {
        match self {
            Object::Name(n) => Some(n),
            _ => None,
        }
    }
    pub(crate) fn as_dict(&self) -> Option<&Dict> {
        match self {
            Object::Dict(d) | Object::Stream(d, ..) => Some(d),
            _ => None,
        }
    }
    pub(crate) fn as_array(&self) -> Option<&[Object]> {
        match self {
            Object::Array(a) => Some(a),
            _ => None,
        }
    }
}

pub(crate) fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0c' | b'\0')
}
pub(crate) fn is_delim(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}
fn is_regular(b: u8) -> bool {
    !is_space(b) && !is_delim(b)
}

pub(crate) struct Lexer<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) pos: usize,
    pub(crate) budget: &'a Budget<'a>,
    /// Objects the current top-level object may still hold.
    room: usize,
    items: usize,
    level: usize,
    /// Bytes the last top-level object allocated, for cache charges.
    pub(crate) weight: usize,
    /// Set when an object passed `items` and was dropped.
    pub(crate) over: bool,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize, budget: &'a Budget<'a>) -> Self {
        Self::with_items(data, pos, budget, FILE_ITEMS)
    }
    pub(crate) fn with_items(data: &'a [u8], pos: usize, budget: &'a Budget<'a>, items: usize) -> Self {
        Self { data, pos, budget, room: items, items, level: 0, weight: 0, over: false }
    }
    /// Move past one byte the caller has looked at, charging it.
    #[inline]
    fn step(&mut self) -> Option<()> {
        self.budget.take(Kind::Read, 1)?;
        self.pos += 1;
        Some(())
    }
    pub(crate) fn skip_space(&mut self) -> Option<()> {
        while let Some(&b) = self.data.get(self.pos) {
            if is_space(b) {
                self.step()?;
            } else if b == b'%' {
                while self.data.get(self.pos).is_some_and(|&b| b != b'\r' && b != b'\n') {
                    self.step()?;
                }
            } else {
                break;
            }
        }
        Some(())
    }
    fn regular(&mut self) -> Option<&'a [u8]> {
        let start = self.pos.min(self.data.len());
        self.pos = start;
        while self.data.get(self.pos).is_some_and(|&b| is_regular(b)) {
            self.step()?;
        }
        Some(&self.data[start..self.pos])
    }
    /// The next unsigned integer token, if the next token is one.
    pub(crate) fn uint(&mut self) -> Option<u64> {
        self.skip_space()?;
        let start = self.pos;
        let word = self.regular()?;
        match std::str::from_utf8(word).ok().and_then(|s| s.parse().ok()) {
            Some(n) if word.iter().all(u8::is_ascii_digit) => Some(n),
            _ => { self.pos = start; None }
        }
    }
    /// Consume `word` if it is the next token.
    pub(crate) fn keyword(&mut self, word: &[u8]) -> bool {
        if self.skip_space().is_none() {
            return false;
        }
        let start = self.pos;
        if self.regular() == Some(word) {
            return true;
        }
        self.pos = start;
        false
    }

    /// One object. In a file body, `n g R` becomes `Ref`; in a content stream
    /// (`refs` false) the operator words come back as `Op`.
    pub(crate) fn object(&mut self, refs: bool) -> Option<Object> {
        if self.level == 0 {
            self.room = self.items;
            self.weight = 0;
        }
        if self.room == 0 {
            self.over = true;
            return None;
        }
        self.room -= 1;
        self.budget.take(Kind::Objects, 1)?;
        let _nest = self.budget.nest().ok()?;
        self.level += 1;
        let o = self.parse(refs);
        self.level -= 1;
        self.weight += OBJECT_WEIGHT + match &o {
            Some(Object::Str(s) | Object::Name(s) | Object::Op(s)) => s.len(),
            _ => 0,
        };
        o
    }

    fn parse(&mut self, refs: bool) -> Option<Object> {
        self.skip_space()?;
        let b = *self.data.get(self.pos)?;
        match b {
            b'/' => {
                self.step()?;
                Some(Object::Name(name_bytes(self.regular()?)))
            }
            b'(' => self.literal().map(Object::Str),
            b'<' if self.data.get(self.pos + 1) == Some(&b'<') => {
                self.step()?;
                self.step()?;
                let mut dict = Dict::new();
                loop {
                    self.skip_space()?;
                    match self.data.get(self.pos)? {
                        b'>' => {
                            self.step()?;
                            if self.data.get(self.pos) == Some(&b'>') {
                                self.step()?;
                            }
                            break;
                        }
                        b'/' => {
                            self.step()?;
                            let key = name_bytes(self.regular()?);
                            self.weight += key.len();
                            let value = self.object(refs)?;
                            if !matches!(value, Object::Null) {
                                dict.insert(key, value);
                            }
                        }
                        // A stray token in a dictionary: skip it.
                        _ => { self.object(refs)?; }
                    }
                }
                Some(Object::Dict(dict))
            }
            b'<' => {
                self.step()?;
                let mut out = Vec::new();
                let mut half = None;
                loop {
                    let b = *self.data.get(self.pos)?;
                    self.step()?;
                    if b == b'>' {
                        break;
                    }
                    if let Some(v) = hex_value(b) {
                        match half.take() {
                            None => half = Some(v),
                            Some(h) => out.push(h << 4 | v),
                        }
                    }
                }
                if let Some(h) = half {
                    out.push(h << 4);
                }
                Some(Object::Str(out))
            }
            b'[' => {
                self.step()?;
                let mut items = Vec::new();
                loop {
                    self.skip_space()?;
                    if *self.data.get(self.pos)? == b']' {
                        self.step()?;
                        return Some(Object::Array(items));
                    }
                    items.push(self.object(refs)?);
                }
            }
            b']' | b'>' | b')' | b'{' | b'}' => {
                self.step()?;
                Some(Object::Op(vec![b]))
            }
            _ => {
                let word = self.regular()?;
                if word.is_empty() {
                    self.step()?;
                    return Some(Object::Op(vec![b]));
                }
                if let Some(number) = number(word) {
                    if refs && let Object::Int(n) = number && n >= 0 && n <= i64::from(u32::MAX) {
                        let save = self.pos;
                        if let Some(g) = self.uint() && g <= 65535 && self.keyword(b"R") {
                            return Some(Object::Ref(n as u32, g as u16));
                        }
                        self.pos = save;
                    }
                    return Some(number);
                }
                Some(match word {
                    b"true" => Object::Bool(true),
                    b"false" => Object::Bool(false),
                    b"null" => Object::Null,
                    _ => Object::Op(word.to_vec()),
                })
            }
        }
    }

    fn literal(&mut self) -> Option<Vec<u8>> {
        self.step()?;
        let mut out = Vec::new();
        let mut nest = 0usize;
        loop {
            let b = *self.data.get(self.pos)?;
            self.step()?;
            match b {
                b'(' => { nest += 1; out.push(b); }
                b')' if nest == 0 => return Some(out),
                b')' => { nest -= 1; out.push(b); }
                b'\\' => {
                    let e = *self.data.get(self.pos)?;
                    self.step()?;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'0'..=b'7' => {
                            let mut v = u32::from(e - b'0');
                            for _ in 0..2 {
                                match self.data.get(self.pos) {
                                    Some(&d @ b'0'..=b'7') => { v = v * 8 + u32::from(d - b'0'); self.step()?; }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        b'\r' => {
                            if self.data.get(self.pos) == Some(&b'\n') {
                                self.step()?;
                            }
                        }
                        b'\n' => {}
                        _ => out.push(e),
                    }
                }
                b'\r' => {
                    // An end of line in a literal string reads as one newline.
                    if self.data.get(self.pos) == Some(&b'\n') {
                        self.step()?;
                    }
                    out.push(b'\n');
                }
                _ => out.push(b),
            }
        }
    }
}
pub(crate) fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn name_bytes(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'#' && let (Some(h), Some(l)) = (raw.get(i + 1).and_then(|&b| hex_value(b)), raw.get(i + 2).and_then(|&b| hex_value(b))) {
            out.push(h << 4 | l);
            i += 3;
        } else {
            out.push(raw[i]);
            i += 1;
        }
    }
    out
}

fn number(word: &[u8]) -> Option<Object> {
    let first = *word.first()?;
    if !(first.is_ascii_digit() || matches!(first, b'+' | b'-' | b'.')) {
        return None;
    }
    let text = std::str::from_utf8(word).ok()?;
    if let Ok(i) = text.parse::<i64>() {
        return Some(Object::Int(i));
    }
    if word.iter().all(|&b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.')) {
        // Writers emit forms such as "-.5", "4." and "--2"; read the sign run once.
        let body = text.trim_start_matches(['+', '-']);
        let negative = text[..text.len() - body.len()].contains('-');
        let v: f64 = if body.is_empty() || body == "." { 0.0 } else { body.parse().ok()? };
        return Some(Object::Real(if negative { -v } else { v }));
    }
    None
}
