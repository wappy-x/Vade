//! Fonts as text decoders (ISO 32000-1, 9.6 to 9.10): character codes to
//! Unicode through ToUnicode CMaps, simple-font encodings with /Differences,
//! and glyph widths for word gaps. No glyph outlines are read.
//! Every table, entry and glyph is charged to the read's budget first.
use super::budget::{Budget, Kind};
use super::document::Document;
use super::syntax::{Dict, Lexer, Object};
use std::collections::HashMap;
use std::rc::Rc;

/// Names for codes 0xA1..=0xFF in ISO Latin-1 order; the character is the code.
const LATIN1: [&str; 95] = [
    "exclamdown", "cent", "sterling", "currency", "yen", "brokenbar", "section", "dieresis", "copyright", "ordfeminine", "guillemotleft", "logicalnot", "sfthyphen", "registered", "macron",
    "degree", "plusminus", "twosuperior", "threesuperior", "acute", "mu", "paragraph", "periodcentered", "cedilla", "onesuperior", "ordmasculine", "guillemotright", "onequarter", "onehalf", "threequarters", "questiondown",
    "Agrave", "Aacute", "Acircumflex", "Atilde", "Adieresis", "Aring", "AE", "Ccedilla", "Egrave", "Eacute", "Ecircumflex", "Edieresis", "Igrave", "Iacute", "Icircumflex", "Idieresis",
    "Eth", "Ntilde", "Ograve", "Oacute", "Ocircumflex", "Otilde", "Odieresis", "multiply", "Oslash", "Ugrave", "Uacute", "Ucircumflex", "Udieresis", "Yacute", "Thorn", "germandbls",
    "agrave", "aacute", "acircumflex", "atilde", "adieresis", "aring", "ae", "ccedilla", "egrave", "eacute", "ecircumflex", "edieresis", "igrave", "iacute", "icircumflex", "idieresis",
    "eth", "ntilde", "ograve", "oacute", "ocircumflex", "otilde", "odieresis", "divide", "oslash", "ugrave", "uacute", "ucircumflex", "udieresis", "yacute", "thorn", "ydieresis",
];

/// Names for printable ASCII 0x20..=0x7E that are not one letter.
const ASCII: [&str; 95] = [
    "space", "exclam", "quotedbl", "numbersign", "dollar", "percent", "ampersand", "quotesingle", "parenleft", "parenright", "asterisk", "plus", "comma", "hyphen", "period", "slash",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "colon", "semicolon", "less", "equal", "greater", "question",
    "at", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft", "backslash", "bracketright", "asciicircum", "underscore",
    "grave", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "braceleft", "bar", "braceright", "asciitilde",
];

/// Other glyph names from the Adobe Glyph List: the Standard, WinAnsi and
/// MacRoman sets, ligatures, Greek and common math symbols.
const OTHER: &[(&str, &str)] = &[
    ("quoteright", "\u{2019}"), ("quoteleft", "\u{2018}"), ("quotedblleft", "\u{201c}"), ("quotedblright", "\u{201d}"), ("quotesinglbase", "\u{201a}"), ("quotedblbase", "\u{201e}"),
    ("guilsinglleft", "\u{2039}"), ("guilsinglright", "\u{203a}"), ("bullet", "\u{2022}"), ("endash", "\u{2013}"), ("emdash", "\u{2014}"), ("dagger", "\u{2020}"), ("daggerdbl", "\u{2021}"),
    ("ellipsis", "\u{2026}"), ("perthousand", "\u{2030}"), ("trademark", "\u{2122}"), ("Euro", "\u{20ac}"), ("florin", "\u{192}"), ("fraction", "\u{2044}"), ("circumflex", "\u{2c6}"),
    ("tilde", "\u{2dc}"), ("breve", "\u{2d8}"), ("dotaccent", "\u{2d9}"), ("ring", "\u{2da}"), ("hungarumlaut", "\u{2dd}"), ("ogonek", "\u{2db}"), ("caron", "\u{2c7}"),
    ("dotlessi", "\u{131}"), ("Lslash", "\u{141}"), ("lslash", "\u{142}"), ("OE", "\u{152}"), ("oe", "\u{153}"), ("Scaron", "\u{160}"), ("scaron", "\u{161}"), ("Zcaron", "\u{17d}"),
    ("zcaron", "\u{17e}"), ("Ydieresis", "\u{178}"), ("fi", "fi"), ("fl", "fl"), ("ff", "ff"), ("ffi", "ffi"), ("ffl", "ffl"), ("minus", "\u{2212}"), ("notequal", "\u{2260}"),
    ("lessequal", "\u{2264}"), ("greaterequal", "\u{2265}"), ("infinity", "\u{221e}"), ("partialdiff", "\u{2202}"), ("summation", "\u{2211}"), ("product", "\u{220f}"), ("integral", "\u{222b}"),
    ("radical", "\u{221a}"), ("approxequal", "\u{2248}"), ("lozenge", "\u{25ca}"), ("nbspace", " "), ("space", " "), ("arrowleft", "\u{2190}"), ("arrowup", "\u{2191}"), ("arrowright", "\u{2192}"),
    ("arrowdown", "\u{2193}"), ("arrowboth", "\u{2194}"), ("element", "\u{2208}"), ("prime", "\u{2032}"), ("minute", "\u{2032}"), ("second", "\u{2033}"), ("similar", "\u{223c}"),
    ("proportional", "\u{221d}"), ("equivalence", "\u{2261}"), ("logicaland", "\u{2227}"), ("logicalor", "\u{2228}"), ("intersection", "\u{2229}"), ("union", "\u{222a}"),
    ("propersubset", "\u{2282}"), ("propersuperset", "\u{2283}"), ("reflexsubset", "\u{2286}"), ("reflexsuperset", "\u{2287}"), ("gradient", "\u{2207}"), ("nabla", "\u{2207}"),
    ("universal", "\u{2200}"), ("existential", "\u{2203}"), ("emptyset", "\u{2205}"), ("therefore", "\u{2234}"), ("perpendicular", "\u{22a5}"), ("dotmath", "\u{22c5}"),
    ("angle", "\u{2220}"), ("circleplus", "\u{2295}"), ("circlemultiply", "\u{2297}"), ("Delta", "\u{394}"), ("Omega", "\u{3a9}"), ("mu", "\u{3bc}"), ("hyphen", "-"),
    ("sfthyphen", "-"), ("brokenbar", "\u{a6}"), ("arrowdblright", "\u{21d2}"), ("arrowdblleft", "\u{21d0}"), ("arrowdblboth", "\u{21d4}"), ("degree", "\u{b0}"), ("copyrightsans", "\u{a9}"),
];

const GREEK_UPPER: [&str; 24] = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"];
const GREEK_LOWER: [&str; 25] = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma1", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"];

/// Unicode text for one glyph name, per the Adobe Glyph List rules: suffixes
/// after '.' drop, '_' joins components, and uniXXXX / uXXXX[XX] are parsed.
pub(crate) fn glyph_text(name: &str) -> Option<String> {
    let base = name.split('.').next().unwrap_or("");
    if base.contains('_') {
        let parts: Option<String> = base.split('_').map(glyph_text).collect();
        return parts.filter(|s| !s.is_empty());
    }
    if let Some(&(_, s)) = OTHER.iter().find(|(n, _)| *n == base) {
        return Some(s.into());
    }
    if let Some(i) = ASCII.iter().position(|n| *n == base) {
        return Some(char::from(0x20 + i as u8).to_string());
    }
    if let Some(i) = LATIN1.iter().position(|n| *n == base) {
        return Some(char::from(0xa1 + i as u8).to_string());
    }
    if let Some(i) = GREEK_UPPER.iter().position(|n| *n == base) {
        // U+03A2 is unassigned; Sigma and the letters after it shift by one.
        return char::from_u32(0x391 + i as u32 + u32::from(i >= 17)).map(String::from);
    }
    if let Some(i) = GREEK_LOWER.iter().position(|n| *n == base) {
        return char::from_u32(0x3b1 + i as u32).map(String::from);
    }
    let hex = |s: &str| u32::from_str_radix(s, 16).ok().filter(|_| s.bytes().all(|b| b.is_ascii_hexdigit()));
    if let Some(digits) = base.strip_prefix("uni") && !digits.is_empty() && digits.len() % 4 == 0 {
        let units: Option<Vec<u16>> = digits.as_bytes().chunks(4).map(|c| hex(std::str::from_utf8(c).ok()?).map(|v| v as u16)).collect();
        return units.map(|u| String::from_utf16_lossy(&u));
    }
    if let Some(digits) = base.strip_prefix('u') && (4..=6).contains(&digits.len()) {
        return hex(digits).and_then(char::from_u32).map(String::from);
    }
    None
}

/// The high half of StandardEncoding (0xA1..=0xFB), as (code, character).
const STANDARD_HIGH: &[(u8, char)] = &[
    (0xa1, '\u{a1}'), (0xa2, '\u{a2}'), (0xa3, '\u{a3}'), (0xa4, '\u{2044}'), (0xa5, '\u{a5}'), (0xa6, '\u{192}'), (0xa7, '\u{a7}'), (0xa8, '\u{a4}'), (0xa9, '\''), (0xaa, '\u{201c}'),
    (0xab, '\u{ab}'), (0xac, '\u{2039}'), (0xad, '\u{203a}'), (0xae, '\u{fb01}'), (0xaf, '\u{fb02}'), (0xb1, '\u{2013}'), (0xb2, '\u{2020}'), (0xb3, '\u{2021}'), (0xb4, '\u{b7}'),
    (0xb6, '\u{b6}'), (0xb7, '\u{2022}'), (0xb8, '\u{201a}'), (0xb9, '\u{201e}'), (0xba, '\u{201d}'), (0xbb, '\u{bb}'), (0xbc, '\u{2026}'), (0xbd, '\u{2030}'), (0xbf, '\u{bf}'),
    (0xc1, '`'), (0xc2, '\u{b4}'), (0xc3, '\u{2c6}'), (0xc4, '\u{2dc}'), (0xc5, '\u{af}'), (0xc6, '\u{2d8}'), (0xc7, '\u{2d9}'), (0xc8, '\u{a8}'), (0xca, '\u{2da}'), (0xcb, '\u{b8}'),
    (0xcd, '\u{2dd}'), (0xce, '\u{2db}'), (0xcf, '\u{2c7}'), (0xd0, '\u{2014}'), (0xe1, '\u{c6}'), (0xe3, '\u{aa}'), (0xe8, '\u{141}'), (0xe9, '\u{d8}'), (0xea, '\u{152}'), (0xeb, '\u{ba}'),
    (0xf1, '\u{e6}'), (0xf5, '\u{131}'), (0xf8, '\u{142}'), (0xf9, '\u{f8}'), (0xfa, '\u{153}'), (0xfb, '\u{df}'),
];

/// WinAnsiEncoding 0x80..=0x9F (Windows code page 1252); 0x00 marks an unused code.
const WIN_80: [char; 32] = [
    '\u{20ac}', '\0', '\u{201a}', '\u{192}', '\u{201e}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{2c6}', '\u{2030}', '\u{160}', '\u{2039}', '\u{152}', '\0', '\u{17d}', '\0',
    '\0', '\u{2018}', '\u{2019}', '\u{201c}', '\u{201d}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{2dc}', '\u{2122}', '\u{161}', '\u{203a}', '\u{153}', '\0', '\u{17e}', '\u{178}',
];

/// MacRomanEncoding 0x80..=0xFF as given in ISO 32000-1, Annex D.
pub(crate) const MAC_80: &str = "ÄÅÇÉÑÖÜáàâäãåçéèêëíìîïñóòôöõúùûü†°¢£§•¶ß®©™´¨≠ÆØ∞±≤≥¥µ∂∑∏π∫ªºΩæø¿¡¬√ƒ≈∆«»… ÀÃÕŒœ–—“”‘’÷◊ÿŸ⁄¤‹›ﬁﬂ‡·‚„‰ÂÊÁËÈÍÎÏÌÓÔ\0ÒÚÛÙıˆ˜¯˘˙˚¸˝˛ˇ";

#[derive(Clone, Copy, PartialEq)]
enum Base {
    Standard,
    WinAnsi,
    MacRoman,
    None,
}

fn base_char(base: Base, code: u8) -> Option<char> {
    let c = match (base, code) {
        (Base::None, _) => return None,
        (_, 0x20..=0x7e) if base != Base::Standard || !matches!(code, 0x27 | 0x60) => char::from(code),
        (Base::Standard, 0x27) => '\u{2019}',
        (Base::Standard, 0x60) => '\u{2018}',
        (Base::Standard, _) => STANDARD_HIGH.iter().find(|(c, _)| *c == code)?.1,
        (Base::WinAnsi, 0x80..=0x9f) => WIN_80[usize::from(code - 0x80)],
        (Base::WinAnsi, 0xa0) => ' ',
        (Base::WinAnsi, 0xad) => '-',
        (Base::WinAnsi, 0xa1..=0xff) => char::from(code),
        (Base::MacRoman, 0x80..=0xff) => MAC_80.chars().nth(usize::from(code - 0x80))?,
        _ => return None,
    };
    (c != '\0').then_some(c)
}

/// Code space and mappings from a CMap stream: codespacerange, bfchar,
/// bfrange (both forms) and, for an encoding CMap, cidchar and cidrange.
#[derive(Default)]
pub(crate) struct CMap {
    /// Code-space ranges as (length, low, high), sorted and merged so that
    /// one binary search finds a code.
    space: Vec<(usize, u32, u32)>,
    chars: HashMap<(usize, u32), String>,
    ranges: Vec<(usize, u32, u32, Vec<u16>)>,
    cids: Vec<(u32, u32, u32)>,
}

fn code_of(s: &[u8]) -> u32 {
    s.iter().take(4).fold(0, |a, &b| a << 8 | u32::from(b))
}

/// A CMap destination string; the CMap format allows at most 512 bytes.
fn utf16(s: &[u8]) -> Vec<u16> {
    s[..s.len().min(512)].chunks(2).map(|c| u16::from(c[0]) << 8 | u16::from(*c.get(1).unwrap_or(&0))).collect()
}

/// Charge one font entry: a count, and its bytes as cached data, since a
/// loaded font stays cached for the rest of the read.
fn entry(budget: &Budget, text: usize) -> Option<()> {
    budget.take(Kind::FontEntries, 1)?;
    budget.take(Kind::Retained, 64 + text)
}

/// Bases a CMap may inherit through /UseCMap.
const MAX_CMAP_BASES: usize = 4;
/// Section operands kept at once; a multiple of 2 and 3, so pairs and
/// triples stay whole when a long section is read in parts.
const SECTION_ARGS: usize = 3 << 12;

/// The CMap in stream `obj`, after the base CMap its /UseCMap entry names,
/// if any: the stream's own mappings replace the base mappings (9.7.5).
pub(crate) fn cmap(doc: &Document, obj: &Object) -> Option<CMap> {
    cmap_at(doc, obj, 0)
}

fn cmap_at(doc: &Document, obj: &Object, bases: usize) -> Option<CMap> {
    let Object::Stream(dict, ..) = obj else { return None };
    let _nest = doc.budget.nest().ok()?;
    let mut map = match dict.get(b"UseCMap".as_slice()) {
        Some(base) if bases < MAX_CMAP_BASES => cmap_at(doc, &doc.resolve(base), bases + 1).unwrap_or_default(),
        _ => CMap::default(),
    };
    let data = doc.stream_data(obj)?;
    map.read(&data, doc.budget)?;
    map.finish(doc.budget)?;
    Some(map)
}

impl CMap {
    #[cfg(test)]
    pub(crate) fn parse(data: &[u8], budget: &Budget) -> Option<Self> {
        let mut map = Self::default();
        map.read(data, budget)?;
        map.finish(budget)?;
        Some(map)
    }

    /// Add the mappings in `data`; later mappings replace earlier ones.
    fn read(&mut self, data: &[u8], budget: &Budget) -> Option<()> {
        budget.take(Kind::Tables, 1)?;
        let mut lex = Lexer::new(data, 0, budget);
        let mut args: Vec<Object> = Vec::new();
        let mut mode: &[u8] = b"";
        while let Some(o) = lex.object(false) {
            budget.take(Kind::Steps, 1)?;
            let Object::Op(op) = o else {
                if !mode.is_empty() {
                    args.push(o);
                    if args.len() == SECTION_ARGS {
                        self.section(mode, &args, budget)?;
                        args.clear();
                    }
                }
                continue;
            };
            match op.as_slice() {
                b"begincodespacerange" => mode = b"space",
                b"beginbfchar" => mode = b"bfchar",
                b"beginbfrange" => mode = b"bfrange",
                b"begincidchar" => mode = b"cidchar",
                b"begincidrange" => mode = b"cidrange",
                w if w.starts_with(b"end") => {
                    self.section(mode, &args, budget)?;
                    mode = b"";
                    args.clear();
                }
                _ => {}
            }
            if self.chars.len() + self.ranges.len() > 1 << 20 {
                break;
            }
        }
        budget.stopped().is_none().then_some(())
    }

    /// Sort for binary search: CJK maps often hold thousands of ranges.
    /// Code-space ranges of one length are merged where they overlap.
    fn finish(&mut self, budget: &Budget) -> Option<()> {
        budget.take(Kind::Steps, self.ranges.len() + self.cids.len() + self.space.len())?;
        self.ranges.sort_by_key(|&(len, lo, _, _)| (len, lo));
        self.cids.sort_by_key(|&(lo, _, _)| lo);
        self.space.sort_unstable();
        let mut merged: Vec<(usize, u32, u32)> = Vec::with_capacity(self.space.len());
        for &(len, lo, hi) in &self.space {
            match merged.last_mut() {
                Some(last) if last.0 == len && lo <= last.2.saturating_add(1) => last.2 = last.2.max(hi),
                _ => merged.push((len, lo, hi)),
            }
        }
        self.space = merged;
        Some(())
    }

    fn section(&mut self, mode: &[u8], args: &[Object], budget: &Budget) -> Option<()> {
        fn s(o: &Object) -> Option<&[u8]> {
            if let Object::Str(s) = o { Some(s) } else { None }
        }
        budget.take(Kind::Steps, args.len())?;
        let entry = |text: usize| entry(budget, text);
        match mode {
            b"space" => for p in args.chunks_exact(2) {
                if let (Some(lo), Some(hi)) = (s(&p[0]), s(&p[1])) && (1..=4).contains(&lo.len()) {
                    entry(0)?;
                    let (len, lo, hi) = (lo.len(), code_of(lo), code_of(hi));
                    if lo <= hi {
                        self.space.push((len, lo, hi));
                    }
                }
            },
            b"bfchar" => for p in args.chunks_exact(2) {
                if let (Some(src), Some(dst)) = (s(&p[0]), s(&p[1])) && !src.is_empty() {
                    entry(dst.len())?;
                    self.chars.insert((src.len(), code_of(src)), String::from_utf16_lossy(&utf16(dst)));
                }
            },
            b"bfrange" => for p in args.chunks_exact(3) {
                let (Some(lo), Some(hi)) = (s(&p[0]), s(&p[1])) else { continue };
                let (len, lo, hi) = (lo.len(), code_of(lo), code_of(hi));
                if lo > hi || len == 0 {
                    continue;
                }
                match &p[2] {
                    Object::Str(dst) => {
                        entry(dst.len())?;
                        self.ranges.push((len, lo, hi, utf16(dst)));
                    }
                    Object::Array(items) => for (i, d) in items.iter().enumerate().take((hi - lo) as usize + 1) {
                        if let Some(d) = s(d) {
                            entry(d.len())?;
                            self.chars.insert((len, lo + i as u32), String::from_utf16_lossy(&utf16(d)));
                        }
                    },
                    _ => {}
                }
            },
            b"cidchar" => for p in args.chunks_exact(2) {
                if let (Some(src), Some(cid)) = (s(&p[0]), p[1].as_int()) {
                    entry(0)?;
                    let c = code_of(src);
                    self.cids.push((c, c, cid as u32));
                }
            },
            b"cidrange" => for p in args.chunks_exact(3) {
                if let (Some(lo), Some(hi), Some(cid)) = (s(&p[0]), s(&p[1]), p[2].as_int()) {
                    entry(0)?;
                    self.cids.push((code_of(lo), code_of(hi), cid as u32));
                }
            },
            _ => {}
        }
        Some(())
    }

    /// True when `code` of `len` bytes is in the code space.
    fn in_space(&self, len: usize, code: u32) -> bool {
        let end = self.space.partition_point(|&(l, lo, _)| (l, lo) <= (len, code));
        end > 0 && matches!(self.space[end - 1], (l, _, hi) if l == len && code <= hi)
    }

    fn lookup(&self, len: usize, code: u32) -> Option<String> {
        if let Some(s) = self.chars.get(&(len, code)) {
            return Some(s.clone());
        }
        // The range with the largest start at or below `code`; a few earlier
        // ones are checked too, in case ranges overlap.
        let end = self.ranges.partition_point(|&(l, lo, _, _)| (l, lo) <= (len, code));
        let (_, lo, _, base) = self.ranges[..end].iter().rev().take(8).find(|(l, lo, hi, _)| *l == len && (*lo..=*hi).contains(&code))?;
        let mut units = base.clone();
        let last = units.last_mut()?;
        *last = last.wrapping_add((code - lo) as u16);
        Some(String::from_utf16_lossy(&units))
    }

    fn cid(&self, code: u32) -> u32 {
        let end = self.cids.partition_point(|&(lo, _, _)| lo <= code);
        self.cids[..end].iter().rev().take(8).find(|(lo, hi, _)| (*lo..=*hi).contains(&code)).map_or(code, |(lo, _, cid)| cid.wrapping_add(code - lo))
    }
}

/// One loaded font: how to split a string into codes and what each code means.
pub(crate) struct Font {
    /// Type0 (composite) fonts use multi-byte codes; simple fonts use one byte.
    composite: bool,
    /// UCS-2 predefined CMaps: the code is the UTF-16 unit.
    unicode_codes: bool,
    encoding: CMap,
    to_unicode: Option<CMap>,
    simple: Vec<Option<String>>,
    /// Glyph widths as (first code, last code, width), sorted by first code.
    /// A range stays one entry: it is never expanded code by code.
    widths: Vec<(u32, u32, f64)>,
    default_width: f64,
    /// Glyph space to text space; 1/1000 except for Type3 fonts.
    scale: f64,
    pub(crate) real_widths: bool,
}

/// One decoded glyph: its text (may be empty when unmapped), its advance in
/// text space per unit of font size, and whether it is the one-byte code 32.
pub(crate) struct Glyph {
    pub(crate) text: Option<String>,
    pub(crate) width: f64,
    pub(crate) word_space: bool,
}

impl Font {
    /// The font in `dict`; `None` when the budget stopped while it loaded.
    pub(crate) fn load(doc: &Document, dict: &Dict) -> Option<Self> {
        let budget = doc.budget;
        budget.take(Kind::Tables, 1)?;
        let name = |k: &[u8]| doc.key(dict, k).as_name().map(<[u8]>::to_vec).unwrap_or_default();
        let subtype = name(b"Subtype");
        let composite = subtype == b"Type0";
        let to_unicode = cmap(doc, &doc.key(dict, b"ToUnicode"));
        let mut font = Self {
            composite,
            unicode_codes: false,
            encoding: CMap::default(),
            to_unicode,
            simple: Vec::new(),
            widths: Vec::new(),
            default_width: 0.5,
            scale: 0.001,
            real_widths: false,
        };
        if composite {
            match &*doc.key(dict, b"Encoding") {
                Object::Name(n) => {
                    let n = String::from_utf8_lossy(n);
                    font.unicode_codes = n.starts_with("Uni") && (n.contains("UCS2") || n.contains("UTF16"));
                }
                s @ Object::Stream(..) => font.encoding = cmap(doc, s).unwrap_or_default(),
                _ => {}
            }
            let descendant = doc.key(dict, b"DescendantFonts");
            let d = descendant.as_array().and_then(|a| a.first()).map(|f| doc.resolve(f));
            if let Some(d) = d.as_deref().and_then(Object::as_dict) {
                font.default_width = doc.key(d, b"DW").as_f64().unwrap_or(1000.0) * 0.001;
                font.cid_widths(doc, &doc.key(d, b"W"))?;
            }
            return budget.stopped().is_none().then_some(font);
        }
        if subtype == b"Type3" && let Some(m) = doc.key(dict, b"FontMatrix").as_array() && let Some(a) = m.first().and_then(Object::as_f64) {
            font.scale = a.abs();
        }
        let base_font = String::from_utf8_lossy(&name(b"BaseFont")).into_owned();
        let symbolic = base_font.contains("Symbol") || base_font.contains("Dingbats");
        let mut base = if symbolic { Base::None } else if subtype == b"TrueType" { Base::WinAnsi } else { Base::Standard };
        let encoding = doc.key(dict, b"Encoding");
        let pick = |n: &[u8], fallback| match n {
            b"WinAnsiEncoding" => Base::WinAnsi,
            b"MacRomanEncoding" => Base::MacRoman,
            b"StandardEncoding" => Base::Standard,
            _ => fallback,
        };
        let mut differences = Rc::new(Object::Null);
        match &*encoding {
            Object::Name(n) => base = pick(n, base),
            Object::Dict(d) => {
                if let Some(n) = doc.key(d, b"BaseEncoding").as_name() {
                    base = pick(n, base);
                }
                differences = doc.key(d, b"Differences");
            }
            _ => {}
        }
        for _ in 0..256 {
            entry(budget, 0)?;
        }
        font.simple = (0..=255u8).map(|c| base_char(base, c).map(String::from)).collect();
        let mut code = 0usize;
        for item in differences.as_array().unwrap_or_default() {
            entry(budget, 0)?;
            match item {
                Object::Int(n) => code = (*n).clamp(0, 256) as usize,
                Object::Name(n) => {
                    if code < 256 {
                        font.simple[code] = glyph_text(&String::from_utf8_lossy(n));
                    }
                    code += 1;
                }
                _ => {}
            }
        }
        let first = doc.key(dict, b"FirstChar").as_int().unwrap_or(0);
        if let Some(w) = doc.key(dict, b"Widths").as_array() {
            font.real_widths = true;
            for (i, v) in w.iter().enumerate().take(1 << 16) {
                entry(budget, 0)?;
                if let Some(v) = doc.resolve(v).as_f64() {
                    let c = first.saturating_add(i as i64).clamp(0, 1 << 16) as u32;
                    font.widths.push((c, c, v * font.scale));
                }
            }
        }
        if let Some(d) = doc.key(dict, b"FontDescriptor").as_dict() && let Some(m) = doc.key(d, b"MissingWidth").as_f64() && m > 0.0 {
            font.default_width = m * font.scale;
        }
        font.sort_widths(budget)?;
        budget.stopped().is_none().then_some(font)
    }

    /// The CID font /W array (9.7.4.3): `c [w1 w2 ...]` gives one entry per
    /// width, and `c_first c_last w` gives one entry for the whole range.
    /// Each entry is charged; members are resolved one at a time, not copied.
    fn cid_widths(&mut self, doc: &Document, w: &Object) -> Option<()> {
        let budget = doc.budget;
        let Some(items) = w.as_array() else { return Some(()) };
        self.real_widths = true;
        let item = |i: usize| items.get(i).map(|o| doc.resolve(o));
        let mut i = 0;
        while i + 1 < items.len() {
            entry(budget, 0)?;
            let Some(first) = item(i).and_then(|o| o.as_int()) else { break };
            let next = item(i + 1)?;
            if let Some(list) = next.as_array() {
                for (k, v) in list.iter().enumerate() {
                    entry(budget, 0)?;
                    let c = first.saturating_add(k as i64).clamp(0, u32::MAX.into()) as u32;
                    self.widths.push((c, c, doc.resolve(v).as_f64().unwrap_or(0.0) * 0.001));
                }
                i += 2;
            } else if let (Some(last), Some(v)) = (next.as_int(), item(i + 2).and_then(|o| o.as_f64())) {
                let lo = first.clamp(0, u32::MAX.into()) as u32;
                let hi = last.clamp(0, u32::MAX.into()) as u32;
                if lo <= hi {
                    self.widths.push((lo, hi, v * 0.001));
                }
                i += 3;
            } else {
                break;
            }
        }
        self.sort_widths(budget)
    }

    /// Sort widths by first code. The sort is stable, so of two entries with
    /// the same first code the later one is found first.
    fn sort_widths(&mut self, budget: &Budget) -> Option<()> {
        budget.take(Kind::Steps, self.widths.len())?;
        self.widths.sort_by_key(|&(lo, _, _)| lo);
        Some(())
    }

    /// The width of `code`: the entry with the largest first code at or
    /// below it that covers it; a few earlier ones are checked too, in case
    /// ranges overlap.
    fn width(&self, code: u32) -> f64 {
        let end = self.widths.partition_point(|&(lo, _, _)| lo <= code);
        self.widths[..end].iter().rev().take(8).find(|&&(_, hi, _)| code <= hi).map_or(self.default_width, |&(_, _, w)| w)
    }

    /// Split `bytes` into codes using the code space (7.5 of the CMap spec:
    /// the shortest matching length wins; no match takes one byte, or two
    /// bytes in a composite font without a code space).
    fn code_len(&self, bytes: &[u8]) -> usize {
        if !self.composite {
            return 1;
        }
        let map = if self.encoding.space.is_empty() { self.to_unicode.as_ref() } else { Some(&self.encoding) };
        let space = map.map_or(true, |m| m.space.is_empty());
        if let Some(map) = map {
            for len in 1..=bytes.len().min(4) {
                if map.in_space(len, code_of(&bytes[..len])) {
                    return len;
                }
            }
        }
        if space || self.unicode_codes { 2.min(bytes.len()) } else { 1 }
    }

    /// Call `each` for each code in `bytes` until it returns false or the
    /// budget stops; each code is charged as one step.
    pub(crate) fn decode(&self, bytes: &[u8], budget: &Budget, mut each: impl FnMut(Glyph) -> bool) {
        let mut i = 0;
        while i < bytes.len() {
            if budget.take(Kind::Steps, 1).is_none() {
                return;
            }
            let len = self.code_len(&bytes[i..]).max(1);
            let code = code_of(&bytes[i..i + len]);
            i += len;
            let text = self.to_unicode.as_ref().and_then(|t| t.lookup(len, code)).or_else(|| {
                if self.unicode_codes {
                    char::from_u32(code).map(String::from)
                } else if !self.composite {
                    self.simple.get(code as usize).cloned().flatten()
                } else {
                    None
                }
            });
            let key = if self.composite { self.encoding.cid(code) } else { code };
            if !each(Glyph { text, width: self.width(key), word_space: len == 1 && code == 32 }) {
                return;
            }
        }
    }
}
