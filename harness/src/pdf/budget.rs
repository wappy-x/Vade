//! One work budget per PDF read. Every parser and decoder takes it, and every
//! loop iteration and allocation charges it before the work. When a counter
//! runs out, the budget stops: each later charge fails too, so every loop
//! ends within a few iterations. A set cancel flag stops it the same way.
use super::Limits;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

/// The counters of a budget. Per-byte work charges `Read` (input scanned) or
/// `Decoded` (output made); per-item work charges `Steps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// File or stream bytes the lexer and the scanners look at; a re-read counts again.
    Read,
    /// Stream bytes decoded or copied; each use of a stream counts.
    Decoded,
    /// Objects the parser builds: each number, name, string, array and dictionary entry.
    Objects,
    /// Bytes kept in a cache for the rest of the read.
    Retained,
    /// Tables built: Huffman tables, cross-reference sections, object-stream indexes, CMaps.
    Tables,
    /// Font entries: widths, encoding differences and CMap mappings.
    FontEntries,
    /// Loop iterations: operators, glyphs, codes, rows, page nodes.
    Steps,
}

const KINDS: usize = 7;

/// Why a budget stopped. `Depth` is the one error that does not stop the
/// budget: only the nested object that passed the limit is skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Exhausted {
    Cancelled,
    Limit(Kind),
    Depth,
}

impl Exhausted {
    /// The words the reply uses for this stop.
    pub(crate) fn what(self) -> &'static str {
        match self {
            Exhausted::Cancelled => "cancelled",
            Exhausted::Depth => "nesting depth",
            Exhausted::Limit(Kind::Read) => "bytes read",
            Exhausted::Limit(Kind::Decoded) => "bytes decoded",
            Exhausted::Limit(Kind::Objects) => "objects parsed",
            Exhausted::Limit(Kind::Retained) => "bytes cached",
            Exhausted::Limit(Kind::Tables) => "tables built",
            Exhausted::Limit(Kind::FontEntries) => "font entries",
            Exhausted::Limit(Kind::Steps) => "work steps",
        }
    }
}

pub(crate) struct Budget<'c> {
    left: [Cell<usize>; KINDS],
    used: [Cell<usize>; KINDS],
    depth: Cell<usize>,
    max_depth: usize,
    cancel: &'c AtomicBool,
    stop: Cell<Option<Exhausted>>,
    /// Charges refused after the stop. Tests bound it to prove that every
    /// loop ends once the budget stops.
    late: Cell<usize>,
}

/// One level of nesting; the level is released on drop.
pub(crate) struct Nest<'b>(&'b Cell<usize>);

impl Drop for Nest<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

impl<'c> Budget<'c> {
    pub(crate) fn new(limits: &Limits, cancel: &'c AtomicBool) -> Self {
        let l = [limits.read_bytes, limits.stream_bytes, limits.objects, limits.retained_bytes, limits.tables, limits.font_entries, limits.steps];
        Self {
            left: l.map(Cell::new),
            used: [0; KINDS].map(Cell::new),
            depth: Cell::new(0),
            max_depth: limits.depth,
            cancel,
            stop: Cell::new(None),
            late: Cell::new(0),
        }
    }

    /// Take `n` units of `kind` before the work they pay for.
    #[inline]
    pub(crate) fn charge(&self, kind: Kind, n: usize) -> Result<(), Exhausted> {
        if let Some(e) = self.stop.get() {
            self.late.set(self.late.get() + 1);
            return Err(e);
        }
        let i = kind as usize;
        let before = self.used[i].get();
        let used = before.saturating_add(n);
        // The cancel flag is read about once per 4,096 units of any counter.
        if (used ^ before) >> 12 != 0 && self.cancel.load(Ordering::Relaxed) {
            return self.halt(Exhausted::Cancelled);
        }
        match self.left[i].get().checked_sub(n) {
            Some(rest) => {
                self.left[i].set(rest);
                self.used[i].set(used);
                Ok(())
            }
            None => {
                self.left[i].set(0);
                self.halt(Exhausted::Limit(kind))
            }
        }
    }

    /// `charge`, as an `Option` for parsers that return `None` on failure.
    #[inline]
    pub(crate) fn take(&self, kind: Kind, n: usize) -> Option<()> {
        self.charge(kind, n).ok()
    }

    fn halt(&self, e: Exhausted) -> Result<(), Exhausted> {
        self.stop.set(Some(e));
        Err(e)
    }

    /// Enter one nesting level: objects in objects, references through
    /// references, forms in forms, CMaps over base CMaps.
    pub(crate) fn nest(&self) -> Result<Nest<'_>, Exhausted> {
        if let Some(e) = self.stop.get() {
            self.late.set(self.late.get() + 1);
            return Err(e);
        }
        if self.depth.get() >= self.max_depth {
            return Err(Exhausted::Depth);
        }
        self.depth.set(self.depth.get() + 1);
        Ok(Nest(&self.depth))
    }

    /// Why the budget stopped, if it did. A set cancel flag counts at once.
    pub(crate) fn stopped(&self) -> Option<Exhausted> {
        if self.stop.get().is_none() && self.cancel.load(Ordering::Relaxed) {
            self.stop.set(Some(Exhausted::Cancelled));
        }
        self.stop.get()
    }

    /// Units of `kind` charged so far.
    pub(crate) fn used(&self, kind: Kind) -> usize {
        self.used[kind as usize].get()
    }

    #[cfg(test)]
    pub(crate) fn late(&self) -> usize {
        self.late.get()
    }
}

/// The first `needle` at or after `from` in `data`, scanned in 64 KiB
/// chunks; each chunk is charged as bytes read before it is searched.
pub(crate) fn find(budget: &Budget, data: &[u8], from: usize, needle: &[u8]) -> Result<Option<usize>, Exhausted> {
    const CHUNK: usize = 64 << 10;
    let mut at = from;
    while at + needle.len() <= data.len() {
        let end = (at + CHUNK + needle.len() - 1).min(data.len());
        budget.charge(Kind::Read, end - at)?;
        if let Some(p) = data[at..end].windows(needle.len()).position(|w| w == needle) {
            return Ok(Some(at + p));
        }
        at = end + 1 - needle.len();
    }
    Ok(None)
}

/// The last `needle` that ends at or before `end`, scanned back in charged chunks.
pub(crate) fn rfind(budget: &Budget, data: &[u8], end: usize, needle: &[u8]) -> Result<Option<usize>, Exhausted> {
    const CHUNK: usize = 64 << 10;
    let mut hi = end.min(data.len());
    while hi >= needle.len() {
        let lo = hi.saturating_sub(CHUNK + needle.len() - 1);
        budget.charge(Kind::Read, hi - lo)?;
        if let Some(p) = data[lo..hi].windows(needle.len()).rposition(|w| w == needle) {
            return Ok(Some(lo + p));
        }
        if lo == 0 {
            break;
        }
        hi = lo + needle.len() - 1;
    }
    Ok(None)
}
