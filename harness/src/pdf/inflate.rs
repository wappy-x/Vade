//! RFC 1951 inflate (stored, fixed and dynamic Huffman blocks) with the
//! RFC 1950 zlib wrapper, for PDF FlateDecode streams. Standard library only.
//! Output stops at a caller cap, and every block, symbol, table and output
//! byte is charged to the read's budget before the work, so neither a
//! compression bomb nor a stream of empty blocks runs without bound.
use super::budget::{Budget, Exhausted, Kind};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Error {
    /// The data is not valid deflate data.
    Data(&'static str),
    /// The output reached the caller's cap.
    Cap,
    /// The read's budget stopped.
    Budget(Exhausted),
}

impl From<Exhausted> for Error {
    fn from(e: Exhausted) -> Self {
        Error::Budget(e)
    }
}

/// Inflate `input` into `out`; `Ok` gives the input bytes used. A zlib
/// header is used when present; raw deflate data is accepted too, as some
/// PDF writers omit the header. On error, `out` keeps the bytes decoded so
/// far: damaged PDF streams often still hold text.
pub(crate) fn inflate(input: &[u8], cap: usize, out: &mut Vec<u8>, budget: &Budget) -> Result<usize, Error> {
    let zlib = input.len() >= 2 && input[0] & 0x0f == 8 && input[0] >> 4 <= 7 && (u16::from(input[0]) << 8 | u16::from(input[1])) % 31 == 0;
    if zlib && input[1] & 0x20 != 0 {
        return Err(Error::Data("zlib preset dictionary is not supported"));
    }
    run(input, zlib, cap, out, budget)
}

/// [`inflate`] for raw deflate data with no zlib header, as in a ZIP entry.
/// Raw data can start with bytes that look like a zlib header, so a caller
/// that knows the data is raw must not guess.
pub(crate) fn inflate_raw(input: &[u8], cap: usize, out: &mut Vec<u8>, budget: &Budget) -> Result<usize, Error> {
    run(input, false, cap, out, budget)
}

fn run(input: &[u8], zlib: bool, cap: usize, out: &mut Vec<u8>, budget: &Budget) -> Result<usize, Error> {
    let mut bits = Bits { data: input, pos: if zlib { 2 } else { 0 }, buf: 0, count: 0, over: 0 };
    loop {
        // Each block is charged, so empty blocks cost work too.
        budget.charge(Kind::Steps, 1)?;
        let last = bits.take(1)? == 1;
        match bits.take(2)? {
            0 => stored(&mut bits, cap, out, budget)?,
            1 => {
                let (lit, dist) = fixed();
                codes(&mut bits, lit, dist, cap, out, budget)?
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits, budget)?;
                codes(&mut bits, &lit, &dist, cap, out, budget)?
            }
            _ => return Err(Error::Data("invalid deflate block type")),
        }
        if last {
            // Bytes used: the read position less the whole bytes still buffered.
            let end = bits.pos - ((bits.count / 8) as usize).saturating_sub(bits.over as usize);
            return Ok(if zlib { (end + 4).min(input.len()) } else { end });
        }
    }
}

/// The fixed literal and distance tables (RFC 1951, 3.2.6), built once.
fn fixed() -> (&'static Table, &'static Table) {
    static FIXED: OnceLock<(Table, Table)> = OnceLock::new();
    let (lit, dist) = FIXED.get_or_init(|| {
        let mut l = [8u8; 288];
        l[144..256].fill(9);
        l[256..280].fill(7);
        (Table::build(&l).expect("fixed literal code"), Table::build(&[5u8; 30]).expect("fixed distance code"))
    });
    (lit, dist)
}
/// Least-significant-bit-first reader. Reads past the end yield zero bits and
/// count as overrun; an overrun beyond the buffered bits is an error.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u32,
    over: u32,
}

impl Bits<'_> {
    fn fill(&mut self, need: u32) {
        while self.count < need {
            let byte = match self.data.get(self.pos) {
                Some(&b) => { self.pos += 1; b }
                None => { self.over += 1; 0 }
            };
            self.buf |= u64::from(byte) << self.count;
            self.count += 8;
        }
    }
    fn peek(&mut self, n: u32) -> u32 {
        self.fill(n);
        (self.buf & ((1u64 << n) - 1)) as u32
    }
    fn drop_bits(&mut self, n: u32) -> Result<(), Error> {
        self.buf >>= n;
        self.count -= n;
        // Zero padding fed in by `fill` may sit in the buffer unread; only
        // consuming it is an error.
        if self.over * 8 > self.count { Err(Error::Data("deflate data ends early")) } else { Ok(()) }
    }
    fn take(&mut self, n: u32) -> Result<u32, Error> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.peek(n);
        self.drop_bits(n)?;
        Ok(v)
    }
    fn align(&mut self) {
        let skip = self.count % 8;
        self.buf >>= skip;
        self.count -= skip;
    }
}

fn stored(bits: &mut Bits, cap: usize, out: &mut Vec<u8>, budget: &Budget) -> Result<(), Error> {
    bits.align();
    let len = bits.take(16)? as usize;
    let nlen = bits.take(16)? as usize;
    if len != !nlen & 0xffff {
        return Err(Error::Data("stored block length check failed"));
    }
    let n = len.min(cap.saturating_sub(out.len()));
    budget.charge(Kind::Decoded, n)?;
    for _ in 0..n {
        out.push(bits.take(8)? as u8);
    }
    if n < len { Err(Error::Cap) } else { Ok(()) }
}

/// Canonical Huffman decode table indexed by the next `width` input bits
/// (bit-reversed codes). Each entry packs `symbol << 4 | code length`.
struct Table {
    width: u32,
    entries: Vec<u16>,
}

impl Table {
    /// A table built for one dynamic block: charged as one table, plus
    /// one step per code length and per 8 entries filled.
    fn new(lengths: &[u8], budget: &Budget) -> Result<Self, Error> {
        let width = lengths.iter().copied().max().unwrap_or(0).max(1);
        budget.charge(Kind::Tables, 1)?;
        budget.charge(Kind::Steps, lengths.len() + (1usize << width) / 8)?;
        Self::build(lengths)
    }
    fn build(lengths: &[u8]) -> Result<Self, Error> {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let width = (1..16).rev().find(|&l| count[l] != 0).unwrap_or(1) as u32;
        // Over-subscribed code sets are invalid; incomplete sets are allowed
        // (RFC 1951 permits one distance code) and decode unused slots as errors.
        let mut left = 1i32;
        for &c in &count[1..] {
            left = left * 2 - i32::from(c);
            if left < 0 {
                return Err(Error::Data("over-subscribed Huffman code"));
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for l in 1..16 {
            code = (code + u32::from(count[l - 1])) << 1;
            next[l] = code;
        }
        let mut entries = vec![0u16; 1 << width];
        for (symbol, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = u32::from(l);
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = c.reverse_bits() >> (32 - l);
            let mut i = rev as usize;
            while i < entries.len() {
                entries[i] = (symbol as u16) << 4 | l as u16;
                i += 1 << l;
            }
        }
        Ok(Self { width, entries })
    }
    fn decode(&self, bits: &mut Bits) -> Result<usize, Error> {
        let e = self.entries[bits.peek(self.width) as usize];
        if e == 0 {
            return Err(Error::Data("invalid Huffman code"));
        }
        bits.drop_bits(u32::from(e & 15))?;
        Ok(usize::from(e >> 4))
    }
}

fn dynamic(bits: &mut Bits, budget: &Budget) -> Result<(Table, Table), Error> {
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let hlit = bits.take(5)? as usize + 257;
    let hdist = bits.take(5)? as usize + 1;
    let hclen = bits.take(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(Error::Data("too many Huffman codes"));
    }
    let mut cl = [0u8; 19];
    for &i in &ORDER[..hclen] {
        cl[i] = bits.take(3)? as u8;
    }
    let cl = Table::new(&cl, budget)?;
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        budget.charge(Kind::Steps, 1)?;
        let (value, repeat) = match cl.decode(bits)? {
            s @ 0..=15 => (s as u8, 1),
            16 => (*lengths.get(i.wrapping_sub(1)).ok_or(Error::Data("repeat with no previous length"))?, 3 + bits.take(2)? as usize),
            17 => (0, 3 + bits.take(3)? as usize),
            _ => (0, 11 + bits.take(7)? as usize),
        };
        if i + repeat > lengths.len() {
            return Err(Error::Data("code lengths overflow"));
        }
        lengths[i..i + repeat].fill(value);
        i += repeat;
    }
    if lengths[256] == 0 {
        return Err(Error::Data("no end-of-block code"));
    }
    Ok((Table::new(&lengths[..hlit], budget)?, Table::new(&lengths[hlit..], budget)?))
}

fn codes(bits: &mut Bits, lit: &Table, dist: &Table, cap: usize, out: &mut Vec<u8>, budget: &Budget) -> Result<(), Error> {
    const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
    const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
    const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
    loop {
        let symbol = lit.decode(bits)?;
        if symbol < 256 {
            if out.len() >= cap {
                return Err(Error::Cap);
            }
            budget.charge(Kind::Decoded, 1)?;
            out.push(symbol as u8);
            continue;
        }
        if symbol == 256 {
            return Ok(());
        }
        let s = symbol - 257;
        if s >= 29 {
            return Err(Error::Data("invalid length code"));
        }
        let len = usize::from(LEN_BASE[s]) + bits.take(u32::from(LEN_EXTRA[s]))? as usize;
        let d = dist.decode(bits)?;
        if d >= 30 {
            return Err(Error::Data("invalid distance code"));
        }
        let distance = usize::from(DIST_BASE[d]) + bits.take(u32::from(DIST_EXTRA[d]))? as usize;
        if distance > out.len() {
            return Err(Error::Data("distance reaches before the start of the output"));
        }
        let start = out.len() - distance;
        let n = len.min(cap.saturating_sub(out.len()));
        budget.charge(Kind::Decoded, n.max(1))?;
        for k in 0..n {
            out.push(out[start + k]);
        }
        if n < len {
            return Err(Error::Cap);
        }
    }
}
/// Stored-block deflate (no compression) in a zlib wrapper. Tests use it to
/// build FlateDecode streams; the Adler-32 trailer is real.
#[cfg(test)]
pub(crate) fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut chunks = data.chunks(65535).peekable();
    if data.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(chunk) = chunks.next() {
        out.push(u8::from(chunks.peek().is_none()));
        let n = chunk.len() as u16;
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&(!n).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&(b << 16 | a).to_be_bytes());
    out
}
