//! Content-stream text operators (ISO 32000-1, 9.3 and 9.4). Text comes out
//! in content-stream order. A baseline move starts a new line; a horizontal
//! gap wider than a fraction of the font size becomes one space. Each
//! operator and glyph is charged to the read's budget first.
use super::Limits;
use super::budget::{self, Kind};
use super::document::Document;
use super::filters::{ascii85, ascii_hex, run_length};
use super::font::{Font, Glyph};
use super::inflate::inflate;
use super::syntax::{CONTENT_ITEMS, Dict, Lexer, Object, is_space};
use std::collections::HashMap;
use std::rc::Rc;
type Matrix = [f64; 6];
const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
const MAX_FORM_DEPTH: usize = 8;
const MAX_STACK: usize = 64;

/// `a` then `b` (row-vector convention, as in the PDF specification).
fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

#[derive(Clone)]
struct State {
    ctm: Matrix,
    font: Option<Rc<Font>>,
    size: f64,
    char_space: f64,
    word_space: f64,
    scale: f64,
    leading: f64,
    rise: f64,
}

/// Where the previous glyph run ended, in device space.
struct Pen {
    x: f64,
    y: f64,
    size: f64,
}

/// Shared across the pages of one document: loaded fonts by object number.
#[derive(Default)]
pub(crate) struct Fonts(HashMap<(u32, u16), Rc<Font>>);

pub(crate) struct PageText<'d, 'a> {
    doc: &'d Document<'a>,
    fonts: &'d mut Fonts,
    pub(crate) out: String,
    pub(crate) unmapped: usize,
    pen: Option<Pen>,
    state: State,
    tm: Matrix,
    tlm: Matrix,
    /// Form XObjects being run, innermost last (a cycle guard).
    forms: Vec<(u32, u16)>,
    /// Bytes `out` may hold.
    limit: usize,
    /// Form calls and form content bytes left for this page.
    pub(crate) form_calls: usize,
    form_bytes: usize,
    /// `out` reached `limit`; the page stops there.
    pub(crate) full: bool,
    /// A limit skipped part of this page.
    pub(crate) short: bool,
}

impl<'d, 'a> PageText<'d, 'a> {
    /// A page reader whose text may take `limit` bytes. The document's
    /// budget, and the cancel flag in it, stop it between operators.
    pub(crate) fn new(doc: &'d Document<'a>, fonts: &'d mut Fonts, limits: &Limits, limit: usize) -> Self {
        Self {
            doc,
            fonts,
            out: String::new(),
            unmapped: 0,
            pen: None,
            state: State { ctm: IDENTITY, font: None, size: 0.0, char_space: 0.0, word_space: 0.0, scale: 1.0, leading: 0.0, rise: 0.0 },
            tm: IDENTITY,
            tlm: IDENTITY,
            forms: Vec::new(),
            limit,
            form_calls: limits.form_calls,
            form_bytes: limits.form_bytes,
            full: false,
            short: false,
        }
    }

    pub(crate) fn run(&mut self, content: &[u8], resources: &Object) {
        let empty = Dict::new();
        self.exec(content, resources.as_dict().unwrap_or(&empty));
    }

    fn exec(&mut self, content: &[u8], resources: &Dict) {
        let budget = self.doc.budget;
        let mut lex = Lexer::with_items(content, 0, budget, CONTENT_ITEMS);
        let mut args: Vec<Object> = Vec::new();
        let mut stack: Vec<State> = Vec::new();
        while let Some(o) = lex.object(false) {
            if self.full || budget.take(Kind::Steps, 1).is_none() {
                return;
            }
            let Object::Op(op) = o else {
                if args.len() < 64 {
                    args.push(o);
                }
                continue;
            };
            let num = |i: usize| args.get(i).and_then(Object::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0);
            let nums = |n: usize| -> Option<Matrix> {
                let mut m = [0.0; 6];
                for (i, v) in m.iter_mut().enumerate().take(n) {
                    *v = args.get(i)?.as_f64().filter(|v| v.is_finite())?;
                }
                Some(m)
            };
            match op.as_slice() {
                b"q" if stack.len() < MAX_STACK => stack.push(self.state.clone()),
                b"Q" => if let Some(s) = stack.pop() { self.state = s },
                b"cm" => if let Some(m) = nums(6) { self.state.ctm = mul(&m, &self.state.ctm) },
                b"BT" => { self.tm = IDENTITY; self.tlm = IDENTITY; }
                b"Tf" => {
                    self.state.size = num(1);
                    self.state.font = args.first().and_then(Object::as_name).and_then(|n| self.font(resources, n));
                }
                b"gs" => if let Some(n) = args.first().and_then(Object::as_name) {
                    self.graphics_state(resources, n);
                },
                b"Tc" => self.state.char_space = num(0),
                b"Tw" => self.state.word_space = num(0),
                b"Tz" => self.state.scale = num(0) / 100.0,
                b"TL" => self.state.leading = num(0),
                b"Ts" => self.state.rise = num(0),
                b"Td" => self.line(num(0), num(1)),
                b"TD" => { self.state.leading = -num(1); self.line(num(0), num(1)); }
                b"Tm" => if let Some(m) = nums(6) { self.tm = m; self.tlm = m; },
                b"T*" => self.line(0.0, -self.state.leading),
                b"Tj" => if let Some(Object::Str(s)) = args.first() { self.show(s) },
                b"'" => {
                    self.line(0.0, -self.state.leading);
                    if let Some(Object::Str(s)) = args.first() { self.show(s) }
                }
                b"\"" => {
                    self.state.word_space = num(0);
                    self.state.char_space = num(1);
                    self.line(0.0, -self.state.leading);
                    if let Some(Object::Str(s)) = args.get(2) { self.show(s) }
                }
                b"TJ" => if let Some(items) = args.first().and_then(Object::as_array) {
                    for item in items {
                        if self.full {
                            break;
                        }
                        match item {
                            Object::Str(s) => self.show(s),
                            n => if let Some(v) = n.as_f64().filter(|v| v.is_finite()) {
                                let tx = -v / 1000.0 * self.state.size * self.state.scale;
                                self.tm = mul(&[1.0, 0.0, 0.0, 1.0, tx, 0.0], &self.tm);
                            },
                        }
                    }
                },
                b"Do" => if let Some(n) = args.first().and_then(Object::as_name) {
                    self.form(resources, n);
                },
                b"BI" => if !self.inline_image(&mut lex, resources) {
                    // The image's end is unknown: its bytes must not be read
                    // as operators, so the rest of this stream is skipped.
                    self.short = true;
                    return;
                },
                _ => {}
            }
            args.clear();
        }
        // An operand past the item limit stopped the stream early.
        self.short |= lex.over;
    }
    fn line(&mut self, tx: f64, ty: f64) {
        self.tlm = mul(&[1.0, 0.0, 0.0, 1.0, tx, ty], &self.tlm);
        self.tm = self.tlm;
    }

    fn font(&mut self, resources: &Dict, name: &[u8]) -> Option<Rc<Font>> {
        let fonts = self.doc.key(resources, b"Font");
        self.load_font(fonts.as_dict()?.get(name)?)
    }

    /// "/Name gs": an ExtGState /Font entry is [font size] (8.4.5).
    fn graphics_state(&mut self, resources: &Dict, name: &[u8]) {
        let states = self.doc.key(resources, b"ExtGState");
        let Some(entry) = states.as_dict().and_then(|d| d.get(name)) else { return };
        let gs = self.doc.resolve(entry);
        let Some(gs) = gs.as_dict() else { return };
        if let Some([font, size]) = self.doc.key(gs, b"Font").as_array()
            && let Some(size) = self.doc.resolve(size).as_f64().filter(|v| v.is_finite())
        {
            self.state.font = self.load_font(font);
            self.state.size = size;
        }
    }

    fn load_font(&mut self, entry: &Object) -> Option<Rc<Font>> {
        if let Object::Ref(n, g) = entry && let Some(f) = self.fonts.0.get(&(*n, *g)) {
            return Some(f.clone());
        }
        let obj = self.doc.resolve(entry);
        let font = Rc::new(Font::load(self.doc, obj.as_dict()?)?);
        if let Object::Ref(n, g) = entry {
            self.fonts.0.insert((*n, *g), font.clone());
        }
        Some(font)
    }

    /// "/Name Do" for a form XObject. Depth and cycles are bounded, and so is
    /// the total work: each page has a budget of form calls and of form
    /// content bytes, so an acyclic chain of forms that call each other many
    /// times stops at the budget. A form past the depth limit marks the page
    /// incomplete, so its text is never reported as absent.
    fn form(&mut self, resources: &Dict, name: &[u8]) {
        let xobjects = self.doc.key(resources, b"XObject");
        let Some(entry) = xobjects.as_dict().and_then(|d| d.get(name)).cloned() else { return };
        let id = if let Object::Ref(n, g) = entry { (n, g) } else { (u32::MAX, 0) };
        if self.forms.contains(&id) {
            return;
        }
        let obj = self.doc.resolve(&entry);
        let Some(dict) = obj.as_dict() else { return };
        if dict.get(b"Subtype".as_slice()).and_then(Object::as_name) != Some(b"Form") {
            return;
        }
        if self.forms.len() >= MAX_FORM_DEPTH || self.form_calls == 0 || self.form_bytes == 0 {
            self.short = true;
            return;
        }
        let Ok(_nest) = self.doc.budget.nest() else {
            self.short = true;
            return;
        };
        self.form_calls -= 1;
        let Some(data) = self.doc.stream_data_within(&obj, self.form_bytes) else { return };
        self.form_bytes -= data.len();
        let own = self.doc.key(dict, b"Resources");
        let saved = (self.state.clone(), self.tm, self.tlm);
        if let Some(m) = self.doc.key(dict, b"Matrix").as_array() && m.len() == 6 {
            let m: Vec<f64> = m.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
            self.state.ctm = mul(&[m[0], m[1], m[2], m[3], m[4], m[5]], &self.state.ctm);
        }
        self.forms.push(id);
        self.exec(&data, own.as_dict().unwrap_or(resources));
        self.forms.pop();
        (self.state, self.tm, self.tlm) = saved;
    }

    fn show(&mut self, bytes: &[u8]) {
        let Some(font) = self.state.font.clone().filter(|_| !self.full) else { return };
        let s = self.state.clone();
        let trm = mul(&self.tm, &s.ctm);
        let (x, y) = (trm[4] + trm[2] * s.rise, trm[5] + trm[3] * s.rise);
        let size = (s.size * trm[2].hypot(trm[3])).abs().max(1e-6);
        if let Some(pen) = &self.pen {
            // Distances along and across the current baseline direction.
            let len = trm[0].hypot(trm[1]);
            let (ux, uy) = if len > 0.0 { (trm[0] / len, trm[1] / len) } else { (1.0, 0.0) };
            let (dx, dy) = (x - pen.x, y - pen.y);
            let along = dx * ux + dy * uy;
            let across = -dx * uy + dy * ux;
            let unit = size.max(pen.size);
            let gap = if font.real_widths { 0.15 } else { 0.3 };
            if across.abs() > 0.5 * unit {
                self.newline();
            } else if along > gap * unit || along < -unit {
                self.space();
            }
        }
        let mut advance = 0.0;
        let (fs, cs, ws, hs) = (s.size, s.char_space, s.word_space, s.scale);
        // Glyphs go straight to `out`, and decoding stops at the text limit.
        font.decode(bytes, self.doc.budget, |g: Glyph| {
            match g.text {
                Some(t) => self.push_text(&t),
                None => self.unmapped += 1,
            }
            advance += (g.width * fs + cs + if g.word_space { ws } else { 0.0 }) * hs;
            !self.full
        });
        self.tm = mul(&[1.0, 0.0, 0.0, 1.0, advance, 0.0], &self.tm);
        let end = mul(&self.tm, &self.state.ctm);
        self.pen = Some(Pen { x: end[4] + end[2] * self.state.rise, y: end[5] + end[3] * self.state.rise, size });
    }

    fn push_text(&mut self, text: &str) {
        for c in text.chars() {
            match c {
                '\u{fb00}' => self.put("ff"),
                '\u{fb01}' => self.put("fi"),
                '\u{fb02}' => self.put("fl"),
                '\u{fb03}' => self.put("ffi"),
                '\u{fb04}' => self.put("ffl"),
                '\u{a0}' | '\t' => self.space(),
                ' ' => self.space(),
                '\r' | '\n' => self.newline(),
                c if c.is_control() => {}
                c => self.put(c.encode_utf8(&mut [0; 4])),
            }
            if self.full {
                return;
            }
        }
    }

    /// Append `s` unless it would pass the limit; then the page is full.
    fn put(&mut self, s: &str) {
        if self.out.len() + s.len() > self.limit {
            self.full = true;
        } else {
            self.out.push_str(s);
        }
    }

    fn space(&mut self) {
        if !self.out.is_empty() && !self.out.ends_with([' ', '\n']) {
            self.put(" ");
        }
    }

    fn newline(&mut self) {
        let trimmed = self.out.trim_end_matches(' ').len();
        self.out.truncate(trimmed);
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.put("\n");
        }
    }

    /// Skip "BI <dict> ID <data> EI" so image bytes are not read as
    /// operators. The data ends where its own format says: the size from
    /// the image dictionary for unfiltered data, or the end of the first
    /// filter's encoding. False when that end cannot be found.
    fn inline_image(&mut self, lex: &mut Lexer, resources: &Dict) -> bool {
        let mut dict = Dict::new();
        loop {
            match lex.object(false) {
                Some(Object::Op(op)) if op == b"ID" => break,
                Some(Object::Name(k)) => {
                    let Some(v) = lex.object(false) else { return false };
                    dict.insert(k, v);
                }
                Some(_) => {}
                None => return false,
            }
        }
        let data = lex.data;
        // One white-space byte follows ID.
        let start = (lex.pos + 1).min(data.len());
        let Some(len) = self.image_bytes(&dict, &data[start..], resources) else { return false };
        let end = start.saturating_add(len).min(data.len());
        let mut after = Lexer::new(data, end, self.doc.budget);
        if after.keyword(b"EI") {
            lex.pos = after.pos;
            return true;
        }
        // The data ends where its size says, but EI does not follow at once:
        // look for EI after the data only, never inside it.
        let mut i = end;
        while let Ok(Some(at)) = budget::find(self.doc.budget, data, i, b"EI") {
            if at > 0 && is_space(data[at - 1]) && data.get(at + 2).is_none_or(|&b| is_space(b)) {
                lex.pos = at + 2;
                return true;
            }
            i = at + 1;
        }
        false
    }

    /// The length of inline image data that starts `data`, if it is known.
    fn image_bytes(&self, dict: &Dict, data: &[u8], resources: &Dict) -> Option<usize> {
        let budget = self.doc.budget;
        let get = |short: &[u8], long: &[u8]| dict.get(short).or_else(|| dict.get(long));
        let filter = match get(b"F", b"Filter") {
            Some(Object::Name(n)) => Some(n.as_slice()),
            Some(Object::Array(a)) => a.first().and_then(Object::as_name),
            _ => None,
        };
        match filter {
            None => {
                let int = |short: &[u8], long: &[u8]| get(short, long).and_then(Object::as_int).and_then(|v| usize::try_from(v).ok());
                let mask = matches!(get(b"IM", b"ImageMask"), Some(Object::Bool(true)));
                let (bpc, colors) = if mask { (1, 1) } else { (int(b"BPC", b"BitsPerComponent")?, self.colors(get(b"CS", b"ColorSpace")?, resources)?) };
                let row = int(b"W", b"Width")?.checked_mul(colors)?.checked_mul(bpc)?.div_ceil(8);
                row.checked_mul(int(b"H", b"Height")?)
            }
            Some(b"AHx" | b"ASCIIHexDecode") => Some(ascii_hex(data, usize::MAX, budget).ok()?.1),
            Some(b"A85" | b"ASCII85Decode") => Some(ascii85(data, usize::MAX, budget).ok()?.1),
            Some(b"RL" | b"RunLengthDecode") => Some(run_length(data, usize::MAX, budget).ok()?.1),
            Some(b"Fl" | b"FlateDecode") => inflate(data, super::document::STREAM_CAP, &mut Vec::new(), budget).ok(),
            // A JPEG ends at its EOI marker; entropy-coded data never holds FF D9.
            Some(b"DCT" | b"DCTDecode") => budget::find(budget, data, 0, b"\xff\xd9").ok()?.map(|at| at + 2),
            _ => None,
        }
    }

    /// Color components of an inline image color space: a device name, a
    /// name in the page's /ColorSpace resources, or a color-space array.
    fn colors(&self, cs: &Object, resources: &Dict) -> Option<usize> {
        let from_array = |a: &[Object]| -> Option<usize> {
            match a.first()?.as_name()? {
                b"I" | b"Indexed" | b"Separation" | b"CalGray" => Some(1),
                b"CalRGB" | b"Lab" => Some(3),
                b"DeviceN" => Some(self.doc.resolve(a.get(1)?).as_array()?.len()),
                b"ICCBased" => usize::try_from(self.doc.resolve(a.get(1)?).as_dict()?.get(b"N".as_slice())?.as_int()?).ok(),
                _ => None,
            }
        };
        match cs {
            Object::Name(n) => match n.as_slice() {
                b"G" | b"DeviceGray" | b"CalGray" | b"I" | b"Indexed" => Some(1),
                b"RGB" | b"DeviceRGB" | b"CalRGB" => Some(3),
                b"CMYK" | b"DeviceCMYK" => Some(4),
                _ => match &*self.doc.resolve(self.doc.key(resources, b"ColorSpace").as_dict()?.get(n.as_slice())?) {
                    Object::Name(d) if d.as_slice() != n.as_slice() => self.colors(&Object::Name(d.clone()), &Dict::new()),
                    Object::Array(a) => from_array(a),
                    _ => None,
                },
            },
            Object::Array(a) => from_array(a),
            _ => None,
        }
    }
}
