//! PDF file structure (ISO 32000-1, 7.5): cross-reference tables and streams,
//! incremental updates, object streams, stream filters and the page tree.
//! A damaged cross-reference falls back to a scan for "N G obj" headers.
//! All work is charged to the read's budget; stream bytes stay in the file
//! until a charged decode reads them, and every cache entry is charged.
use super::budget::{self, Budget, Kind};
use super::filters::{ascii85, ascii_hex, lzw, predict, run_length};
use super::inflate::{self, inflate};
use super::syntax::{Dict, Lexer, Object, is_space};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Largest single stream after its filters.
pub(crate) const STREAM_CAP: usize = 128 << 20;
const MAX_PAGE_NODES: usize = 200_000;
const MAX_TREE_DEPTH: usize = 64;
/// Cache weight of one cross-reference entry.
const ENTRY_WEIGHT: usize = 32;

/// A cross-reference entry. `At` keeps the generation of the object it
/// indexes; objects in object streams always have generation 0.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Free,
    At(usize, u16),
    InStream(u32, u32),
}

struct ObjStm {
    data: Vec<u8>,
    first: usize,
    offsets: Vec<(u32, usize)>,
}

pub(crate) struct Page {
    pub(crate) dict: Rc<Object>,
    pub(crate) resources: Rc<Object>,
}

/// The pages a walk of the page tree found.
pub(crate) struct Pages {
    /// The first pages, in document order, up to the caller's cap.
    pub(crate) list: Vec<Page>,
    pub(crate) total: usize,
    /// A limit skipped part of the tree; `total` may be low.
    pub(crate) incomplete: bool,
}

pub(crate) struct Document<'a> {
    data: &'a [u8],
    pub(crate) budget: &'a Budget<'a>,
    xref: HashMap<u32, Entry>,
    scan: OnceCell<HashMap<u32, Entry>>,
    /// Set while the scan table is built: lookups then use only the cross-reference.
    scanning: Cell<bool>,
    pub(crate) trailer: Dict,
    cache: RefCell<HashMap<(u32, u16), Rc<Object>>>,
    streams: RefCell<HashMap<u32, Rc<ObjStm>>>,
    /// Set when a caller cap cut a stream; `take_cut` clears it.
    cut: Cell<bool>,
}

/// The offset of the first "%PDF-" within the first 1,024 bytes, if any.
pub(crate) fn header(data: &[u8]) -> Option<usize> {
    data[..data.len().min(1024)].windows(5).position(|w| w == b"%PDF-")
}

impl<'a> Document<'a> {
    /// Open `data`; all work is charged to `budget`. When the budget stops,
    /// the error is the caller's to name: see `Budget::stopped`.
    pub(crate) fn open(data: &'a [u8], budget: &'a Budget<'a>) -> Result<Self, String> {
        let base = header(data).ok_or("file has no %PDF- header")?;
        let mut doc = Self {
            data,
            budget,
            xref: HashMap::new(),
            scan: OnceCell::new(),
            scanning: Cell::new(false),
            trailer: Dict::new(),
            cache: RefCell::new(HashMap::new()),
            streams: RefCell::new(HashMap::new()),
            cut: Cell::new(false),
        };
        let tail = data.len().saturating_sub(64 << 10);
        if let Ok(Some(at)) = budget::rfind(budget, &data[tail..], data.len() - tail, b"startxref") {
            let mut lex = Lexer::new(data, tail + at + 9, budget);
            if let Some(offset) = lex.uint() {
                // Some writers count offsets from the header, not the file start.
                let offset = offset as usize;
                if !doc.load_xref(offset) && base > 0 {
                    doc.load_xref(offset + base);
                }
            }
        }
        if doc.trailer.get(b"Root".as_slice()).is_none() || doc.catalog().is_none() {
            doc.xref.clear();
            doc.cache.borrow_mut().clear();
            doc.streams.borrow_mut().clear();
            doc.trailer = doc.scan_trailer();
        }
        if doc.trailer.contains_key(b"Encrypt".as_slice()) {
            return Err("PDF is encrypted; Vade does not decrypt PDF files".into());
        }
        if doc.catalog().is_none() {
            return Err("PDF has no readable document catalog".into());
        }
        Ok(doc)
    }

    fn catalog(&self) -> Option<Rc<Object>> {
        let root = self.resolve(self.trailer.get(b"Root".as_slice())?);
        root.as_dict()?;
        Some(root)
    }

    /// Read every cross-reference section from `start` back through /Prev.
    /// Newer revisions are read first, so their entries win. Within one
    /// revision of a hybrid file (7.5.8.4), the /XRefStm entries take
    /// precedence over the classic table for every object they define,
    /// free or in use.
    fn load_xref(&mut self, start: usize) -> bool {
        let mut todo = vec![start];
        let mut seen = HashSet::new();
        let mut any = false;
        while let Some(at) = todo.pop() {
            if at >= self.data.len() || !seen.insert(at) || self.budget.take(Kind::Tables, 1).is_none() {
                continue;
            }
            let mut revision = HashMap::new();
            let Some(trailer) = self.xref_section(at, &mut revision) else { continue };
            any = true;
            if let Some(prev) = trailer.get(b"Prev".as_slice()).and_then(Object::as_int) {
                todo.push(prev as usize);
            }
            if let Some(stm) = trailer.get(b"XRefStm".as_slice()).and_then(Object::as_int).and_then(|s| usize::try_from(s).ok())
                && stm < self.data.len()
                && seen.insert(stm)
                && let Some(d) = self.xref_section(stm, &mut revision)
                && let Some(prev) = d.get(b"Prev".as_slice()).and_then(Object::as_int)
            {
                todo.push(prev as usize);
            }
            for (num, entry) in revision {
                if self.budget.take(Kind::Retained, ENTRY_WEIGHT).is_none() {
                    break;
                }
                self.xref.entry(num).or_insert(entry);
            }
            for (k, v) in trailer {
                if !matches!(k.as_slice(), b"Prev" | b"XRefStm" | b"Index" | b"W" | b"Length" | b"Filter" | b"DecodeParms") {
                    self.trailer.entry(k).or_insert(v);
                }
            }
        }
        any
    }

    /// One section at `at`, into `revision`. A classic table keeps the first
    /// entry for each object; a stream section replaces any entry before it.
    fn xref_section(&self, at: usize, revision: &mut HashMap<u32, Entry>) -> Option<Dict> {
        let mut lex = Lexer::new(self.data, at, self.budget);
        if lex.keyword(b"xref") {
            loop {
                let save = lex.pos;
                let (Some(first), Some(count)) = (lex.uint(), lex.uint()) else {
                    lex.pos = save;
                    break;
                };
                for i in 0..count {
                    self.budget.take(Kind::Steps, 1)?;
                    let (Some(offset), Some(generation)) = (lex.uint(), lex.uint()) else { return None };
                    let generation = u16::try_from(generation).unwrap_or(u16::MAX);
                    let entry = if lex.keyword(b"n") { Entry::At(offset as usize, generation) } else if lex.keyword(b"f") { Entry::Free } else { return None };
                    let num = u32::try_from(first.saturating_add(i)).ok()?;
                    // Object 0 is the head of the free list, never a real object.
                    if num != 0 {
                        revision.entry(num).or_insert(entry);
                    }
                }
            }
            if !lex.keyword(b"trailer") {
                return None;
            }
            return lex.object(true).and_then(|o| if let Object::Dict(d) = o { Some(d) } else { None });
        }
        let (_, _, obj, _) = self.object_at(at)?;
        let Object::Stream(dict, ..) = &obj else { return None };
        if dict.get(b"Type".as_slice()).and_then(Object::as_name) != Some(b"XRef") {
            return None;
        }
        let data = self.stream_data(&obj)?;
        let widths: Vec<usize> = dict.get(b"W".as_slice())?.as_array()?.iter().take(3).map(|w| w.as_int().unwrap_or(0).clamp(0, 8) as usize).collect();
        if widths.len() < 3 {
            return None;
        }
        let row: usize = widths.iter().sum();
        let size = dict.get(b"Size".as_slice()).and_then(Object::as_int).unwrap_or(0);
        let index: Vec<i64> = match dict.get(b"Index".as_slice()).and_then(Object::as_array) {
            Some(a) => {
                self.budget.take(Kind::Steps, a.len())?;
                a.iter().filter_map(Object::as_int).collect()
            }
            None => vec![0, size],
        };
        let mut rows = data.chunks_exact(row.max(1));
        for pair in index.chunks_exact(2) {
            for num in pair[0]..pair[0].saturating_add(pair[1].max(0)) {
                let Some(r) = rows.next() else { break };
                self.budget.take(Kind::Steps, 1)?;
                let mut fields = [0u64; 3];
                let mut p = 0;
                for (f, &w) in fields.iter_mut().zip(&widths) {
                    *f = r[p..p + w].iter().fold(0, |a, &b| a << 8 | u64::from(b));
                    p += w;
                }
                let kind = if widths[0] == 0 { 1 } else { fields[0] };
                let entry = match kind {
                    0 => Entry::Free,
                    1 => Entry::At(fields[1] as usize, u16::try_from(fields[2]).unwrap_or(u16::MAX)),
                    2 => Entry::InStream(fields[1] as u32, fields[2] as u32),
                    _ => continue,
                };
                if let Ok(num) = u32::try_from(num) && num != 0 {
                    revision.insert(num, entry);
                }
            }
        }
        Some(dict.clone())
    }

    /// Parse "N G obj <object> [stream ... endstream]" at `at`: the number,
    /// the generation, the object and its cache weight in bytes.
    fn object_at(&self, at: usize) -> Option<(u32, u16, Object, usize)> {
        let mut lex = Lexer::new(self.data, at, self.budget);
        let num = u32::try_from(lex.uint()?).ok()?;
        let generation = u16::try_from(lex.uint()?).ok()?;
        if !lex.keyword(b"obj") {
            return None;
        }
        let obj = lex.object(true)?;
        let weight = lex.weight;
        let Object::Dict(dict) = obj else { return Some((num, generation, obj, weight)) };
        if !lex.keyword(b"stream") {
            return Some((num, generation, Object::Dict(dict), weight));
        }
        let mut start = lex.pos;
        if self.data.get(start) == Some(&b'\r') {
            start += 1;
        }
        if self.data.get(start) == Some(&b'\n') {
            start += 1;
        }
        let declared = match dict.get(b"Length".as_slice()) {
            Some(Object::Int(n)) => usize::try_from(*n).ok(),
            Some(r @ Object::Ref(..)) => self.resolve(r).as_int().and_then(|n| usize::try_from(n).ok()),
            _ => None,
        };
        let fits = |len: usize| {
            let end = start.checked_add(len)?;
            let mut l = Lexer::new(self.data, end, self.budget);
            (end <= self.data.len() && l.keyword(b"endstream")).then_some(end)
        };
        let end = match declared.and_then(fits) {
            Some(end) => end,
            None => {
                let mut end = budget::find(self.budget, self.data, start, b"endstream").ok()??;
                if end > start && self.data[end - 1] == b'\n' { end -= 1; }
                if end > start && self.data[end - 1] == b'\r' { end -= 1; }
                end
            }
        };
        Some((num, generation, Object::Stream(dict, start, end), weight))
    }

    /// Indirect object `num` of generation `generation`, through the
    /// cross-reference or, when that has no usable entry, through a scan of
    /// the whole file. A current free entry means the object was deleted,
    /// and an entry of another generation means the reference is stale:
    /// both read as absent, and the scan does not bring back an old copy.
    pub(crate) fn get(&self, num: u32, generation: u16) -> Option<Rc<Object>> {
        self.budget.take(Kind::Steps, 1)?;
        if let Some(o) = self.cache.borrow().get(&(num, generation)) {
            return Some(o.clone());
        }
        let _nest = self.budget.nest().ok()?;
        let entry = self.xref.get(&num).copied();
        if entry.is_some_and(|e| !current(e, generation)) {
            return None;
        }
        let (found, weight) = entry.and_then(|e| self.load(num, generation, e)).or_else(|| {
            if self.scanning.get() {
                return None;
            }
            let e = *self.scan_table().get(&num)?;
            if !current(e, generation) {
                return None;
            }
            self.load(num, generation, e)
        })?;
        self.budget.take(Kind::Retained, weight)?;
        let found = Rc::new(found);
        self.cache.borrow_mut().insert((num, generation), found.clone());
        Some(found)
    }

    fn load(&self, num: u32, generation: u16, entry: Entry) -> Option<(Object, usize)> {
        match entry {
            Entry::Free => None,
            Entry::At(at, _) => self.object_at(at).filter(|&(n, g, ..)| n == num && g == generation).map(|(.., o, w)| (o, w)),
            Entry::InStream(stm, index) => {
                let s = self.object_stream(stm)?;
                let at = match s.offsets.get(index as usize) {
                    Some(&(n, at)) if n == num => at,
                    _ => {
                        self.budget.take(Kind::Steps, s.offsets.len())?;
                        s.offsets.iter().find(|(n, _)| *n == num)?.1
                    }
                };
                let mut lex = Lexer::new(&s.data, s.first.checked_add(at)?, self.budget);
                let o = lex.object(true)?;
                Some((o, lex.weight))
            }
        }
    }

    fn object_stream(&self, stm: u32) -> Option<Rc<ObjStm>> {
        if let Some(s) = self.streams.borrow().get(&stm) {
            return Some(s.clone());
        }
        let obj = self.get(stm, 0)?;
        self.index_object_stream(stm, &obj)
    }

    /// Decode object stream `stm` and index its objects. The decoded bytes
    /// and the index stay cached, so both are charged.
    fn index_object_stream(&self, stm: u32, obj: &Object) -> Option<Rc<ObjStm>> {
        let dict = obj.as_dict()?;
        let n = dict.get(b"N".as_slice())?.as_int()?;
        let first = usize::try_from(dict.get(b"First".as_slice())?.as_int()?).ok()?;
        self.budget.take(Kind::Tables, 1)?;
        let data = self.stream_data(obj)?;
        self.budget.take(Kind::Retained, data.len())?;
        let mut offsets = Vec::new();
        {
            let mut lex = Lexer::new(&data, 0, self.budget);
            for _ in 0..n.clamp(0, 1 << 20) {
                self.budget.take(Kind::Retained, 16)?;
                let (Some(num), Some(at)) = (lex.uint(), lex.uint()) else { break };
                offsets.push((num as u32, at as usize));
            }
        }
        let s = Rc::new(ObjStm { data, first, offsets });
        self.streams.borrow_mut().insert(stm, s.clone());
        Some(s)
    }

    fn scan_table(&self) -> &HashMap<u32, Entry> {
        self.scan.get_or_init(|| {
            self.scanning.set(true);
            // Each entry with the file offset of its definition.
            let mut table = HashMap::new();
            let mut stms = Vec::new();
            for (num, generation, at) in self.scan_objects() {
                if self.budget.take(Kind::Retained, ENTRY_WEIGHT).is_none() {
                    break;
                }
                table.insert(num, (Entry::At(at, generation), at));
                stms.push((num, at));
            }
            // Objects inside object streams, unless a later plain body object
            // exists: the definition nearest the end of the file wins.
            for (num, at) in stms {
                let Some((_, _, obj, _)) = self.object_at(at) else { continue };
                let Some(d) = obj.as_dict() else { continue };
                if d.get(b"Type".as_slice()).and_then(Object::as_name) != Some(b"ObjStm") {
                    continue;
                }
                let Some(s) = self.index_object_stream(num, &obj) else { continue };
                for (i, &(n, _)) in s.offsets.iter().enumerate() {
                    if self.budget.take(Kind::Retained, ENTRY_WEIGHT).is_none() {
                        break;
                    }
                    if n != num && table.get(&n).is_none_or(|&(_, pos)| pos < at) {
                        table.insert(n, (Entry::InStream(num, i as u32), at));
                    }
                }
            }
            self.scanning.set(false);
            table.into_iter().map(|(n, (e, _))| (n, e)).collect()
        })
    }

    /// Every "N G obj" header in file order, as (number, generation,
    /// offset); a later definition of a number replaces an earlier one.
    fn scan_objects(&self) -> Vec<(u32, u16, usize)> {
        let d = self.data;
        let mut found = HashMap::new();
        let mut i = 0;
        while let Ok(Some(at)) = budget::find(self.budget, d, i, b"obj") {
            i = at + 3;
            if d.get(at + 3).is_some_and(|&b| !is_space(b) && !b"<[(/%".contains(&b)) {
                continue;
            }
            // Walk back over "N G " to the start of N. The walks of two
            // headers never overlap, and each byte walked is charged.
            let mut j = at;
            let mut parts = 0;
            while parts < 2 {
                while j > 0 && is_space(d[j - 1]) && self.budget.take(Kind::Read, 1).is_some() { j -= 1; }
                let end = j;
                while j > 0 && d[j - 1].is_ascii_digit() && self.budget.take(Kind::Read, 1).is_some() { j -= 1; }
                if j == end { break; }
                parts += 1;
            }
            if parts == 2 && (j == 0 || is_space(d[j - 1]) || !d[j - 1].is_ascii_alphanumeric()) {
                let mut lex = Lexer::new(d, j, self.budget);
                if let Some(num) = lex.uint().and_then(|n| u32::try_from(n).ok())
                    && let Some(generation) = lex.uint().and_then(|g| u16::try_from(g).ok())
                    && self.budget.take(Kind::Retained, ENTRY_WEIGHT).is_some()
                {
                    found.insert(num, (generation, j));
                }
            }
        }
        let mut list: Vec<_> = found.into_iter().map(|(n, (g, at))| (n, g, at)).collect();
        list.sort_by_key(|&(_, _, at)| at);
        list
    }

    /// A trailer for a file whose cross-reference failed: the last "trailer"
    /// dictionary with /Root, else the last catalog object found by scanning.
    fn scan_trailer(&self) -> Dict {
        let mut at = self.data.len();
        while let Ok(Some(p)) = budget::rfind(self.budget, self.data, at, b"trailer") {
            if let Some(Object::Dict(d)) = Lexer::new(self.data, p + 7, self.budget).object(true)
                && d.contains_key(b"Root".as_slice())
            {
                return d;
            }
            at = p;
        }
        let mut trailer = Dict::new();
        let mut list: Vec<(u32, Entry)> = self.scan_table().iter().map(|(&n, &e)| (n, e)).collect();
        list.sort_by_key(|&(n, _)| n);
        for (num, e) in list {
            let generation = if let Entry::At(_, g) = e { g } else { 0 };
            let Some(o) = self.get(num, generation) else { continue };
            let Some(d) = o.as_dict() else { continue };
            if d.get(b"Type".as_slice()).and_then(Object::as_name) == Some(b"XRef") && let Some(enc) = d.get(b"Encrypt".as_slice()) {
                trailer.insert(b"Encrypt".to_vec(), enc.clone());
            }
            if d.get(b"Type".as_slice()).and_then(Object::as_name) == Some(b"Catalog") && d.contains_key(b"Pages".as_slice()) {
                trailer.insert(b"Root".to_vec(), Object::Ref(num, generation));
            }
        }
        trailer
    }

    /// Follow indirect references (bounded); a dangling one reads as null.
    pub(crate) fn resolve(&self, o: &Object) -> Rc<Object> {
        let mut cur = match o {
            Object::Ref(n, g) => self.get(*n, *g),
            _ => return Rc::new(o.clone()),
        };
        for _ in 0..8 {
            match cur.as_deref() {
                Some(Object::Ref(n, g)) => cur = self.get(*n, *g),
                Some(_) => return cur.unwrap_or_else(|| Rc::new(Object::Null)),
                None => break,
            }
        }
        Rc::new(Object::Null)
    }

    pub(crate) fn key(&self, d: &Dict, key: &[u8]) -> Rc<Object> {
        d.get(key).map_or_else(|| Rc::new(Object::Null), |o| self.resolve(o))
    }

    /// The decoded bytes of a stream object, at most `STREAM_CAP`.
    /// Unsupported filters (image codecs) give `None`.
    pub(crate) fn stream_data(&self, obj: &Object) -> Option<Vec<u8>> {
        self.stream_data_within(obj, STREAM_CAP)
    }

    /// The decoded bytes of a stream object, at most `cap`. Every use is
    /// charged before its bytes are copied or decoded, unfiltered streams
    /// too; output past `cap` is cut and marks the cut.
    pub(crate) fn stream_data_within(&self, obj: &Object, cap: usize) -> Option<Vec<u8>> {
        let Object::Stream(dict, start, end) = obj else { return None };
        let raw = self.data.get(*start..*end)?;
        let filters: Vec<Vec<u8>> = match &*self.key(dict, b"Filter") {
            Object::Name(n) => vec![n.clone()],
            Object::Array(a) => {
                self.budget.take(Kind::Steps, a.len())?;
                a.iter().map(|f| self.resolve(f).as_name().unwrap_or_default().to_vec()).collect()
            }
            _ => Vec::new(),
        };
        let parms = self.key(dict, if dict.contains_key(b"DecodeParms".as_slice()) { b"DecodeParms".as_slice() } else { b"DP" });
        let cap = cap.min(STREAM_CAP);
        // `None` until the first filter runs: the input is then the raw bytes.
        let mut data: Option<Vec<u8>> = None;
        for (i, f) in filters.iter().enumerate() {
            let p = match &*parms {
                Object::Array(a) => a.get(i).map(|p| self.resolve(p)),
                Object::Dict(_) if i == 0 => Some(parms.clone()),
                _ => None,
            };
            let p = p.as_deref().and_then(Object::as_dict);
            let input = data.as_deref().unwrap_or(raw);
            let out = match f.as_slice() {
                b"FlateDecode" | b"Fl" => {
                    let mut out = Vec::new();
                    match inflate(input, cap, &mut out, self.budget) {
                        Ok(_) => {}
                        Err(inflate::Error::Budget(_)) => return None,
                        Err(inflate::Error::Data(_)) if out.is_empty() => return None,
                        Err(_) => {}
                    }
                    predict(out, p, self.budget).ok()??
                }
                b"LZWDecode" | b"LZW" => {
                    let early = p.and_then(|d| d.get(b"EarlyChange".as_slice())).and_then(Object::as_int).unwrap_or(1);
                    predict(lzw(input, early != 0, cap, self.budget).ok()?, p, self.budget).ok()??
                }
                b"ASCIIHexDecode" | b"AHx" => ascii_hex(input, cap, self.budget).ok()?.0,
                b"ASCII85Decode" | b"A85" => ascii85(input, cap, self.budget).ok()?.0,
                b"RunLengthDecode" | b"RL" => run_length(input, cap, self.budget).ok()?.0,
                _ => return None,
            };
            // Predictors never add bytes, so `out` stays within `cap`.
            if out.len() >= cap {
                self.cut.set(true);
            }
            data = Some(out);
        }
        if let Some(data) = data {
            return Some(data);
        }
        let n = raw.len().min(cap);
        if n < raw.len() {
            self.cut.set(true);
            if n == 0 {
                return None;
            }
        }
        self.budget.take(Kind::Decoded, n)?;
        Some(raw[..n].to_vec())
    }

    /// True when a caller cap cut a stream since the last call.
    pub(crate) fn take_cut(&self) -> bool {
        self.cut.replace(false)
    }

    /// Pages in document order, with inherited /Resources: the first `cap`
    /// pages, the total count, and whether a limit skipped part of the tree.
    /// A branch past the depth limit is skipped alone; the walk goes on with
    /// its siblings. Only dictionaries are tree nodes, never streams.
    pub(crate) fn pages(&self, cap: usize) -> Pages {
        let mut out = Pages { list: Vec::new(), total: 0, incomplete: false };
        let mut seen = HashSet::new();
        let Some(root) = self.catalog() else { return out };
        let Some(top) = root.as_dict().and_then(|d| d.get(b"Pages".as_slice())).cloned() else { return out };
        let mut stack = vec![(top, Rc::new(Object::Null), 0usize)];
        let mut nodes = 0;
        while let Some((node, inherited, depth)) = stack.pop() {
            nodes += 1;
            if nodes > MAX_PAGE_NODES || self.budget.take(Kind::Steps, 1).is_none() {
                out.incomplete = true;
                break;
            }
            if depth > MAX_TREE_DEPTH {
                out.incomplete = true;
                continue;
            }
            if let Object::Ref(n, g) = node && !seen.insert((n, g)) {
                continue;
            }
            let obj = self.resolve(&node);
            let Object::Dict(d) = &*obj else { continue };
            let resources = match d.get(b"Resources".as_slice()) {
                Some(r) => self.resolve(r),
                None => inherited,
            };
            let kids = self.key(d, b"Kids");
            let is_tree = d.get(b"Type".as_slice()).and_then(Object::as_name) == Some(b"Pages") || kids.as_array().is_some();
            if is_tree {
                // Nodes may share one /Kids array: the stack never holds more
                // entries than the node limit lets the walk visit. Kids are
                // references (7.7.3.2), so a push copies no object.
                let kids = kids.as_array().unwrap_or_default();
                let room = MAX_PAGE_NODES.saturating_sub(nodes + stack.len());
                if kids.len() > room {
                    out.incomplete = true;
                }
                for kid in kids[..kids.len().min(room)].iter().rev() {
                    if self.budget.take(Kind::Steps, 1).is_none() {
                        break;
                    }
                    if let Object::Ref(..) = kid {
                        stack.push((kid.clone(), resources.clone(), depth + 1));
                    }
                }
            } else {
                out.total += 1;
                if out.list.len() < cap {
                    out.list.push(Page { dict: obj.clone(), resources });
                }
            }
        }
        out
    }
}

/// True when `entry` can hold generation `generation` of its object.
fn current(entry: Entry, generation: u16) -> bool {
    match entry {
        Entry::Free => false,
        Entry::At(_, g) => g == generation,
        Entry::InStream(..) => generation == 0,
    }
}
