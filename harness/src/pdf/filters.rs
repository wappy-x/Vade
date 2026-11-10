//! Stream filters other than FlateDecode (ISO 32000-1, 7.4) and the PNG and
//! TIFF predictors. Each filter charges the read's budget before it reads an
//! input byte or makes an output byte, and stops at the caller's cap; no
//! append crosses the cap.
use super::budget::{Budget, Exhausted, Kind};
use super::syntax::{Dict, Object, hex_value};

/// PNG (10..15) and TIFF (2, 8-bit) predictors from /DecodeParms. The
/// output is never longer than the input: a short last row stays short.
/// `None` for a predictor this reader does not support.
pub(crate) fn predict(data: Vec<u8>, parms: Option<&Dict>, budget: &Budget) -> Result<Option<Vec<u8>>, Exhausted> {
    let int = |k: &[u8], d: i64| parms.and_then(|p| p.get(k)).and_then(Object::as_int).unwrap_or(d);
    let predictor = int(b"Predictor", 1);
    if predictor < 2 {
        return Ok(Some(data));
    }
    let colors = int(b"Colors", 1).clamp(1, 32) as usize;
    let bpc = int(b"BitsPerComponent", 8).clamp(1, 16) as usize;
    let columns = int(b"Columns", 1).clamp(1, 1 << 20) as usize;
    let bpp = (colors * bpc).div_ceil(8);
    let stride = (colors * bpc * columns).div_ceil(8);
    if predictor == 2 {
        if bpc != 8 {
            return Ok(None);
        }
        let mut data = data;
        for row in data.chunks_mut(stride) {
            budget.charge(Kind::Decoded, row.len())?;
            for i in bpp..row.len() {
                row[i] = row[i].wrapping_add(row[i - bpp]);
            }
        }
        return Ok(Some(data));
    }
    let mut out = Vec::with_capacity(data.len());
    let mut prev = vec![0u8; stride.min(data.len())];
    for row in data.chunks(stride + 1) {
        budget.charge(Kind::Decoded, row.len())?;
        let Some((&kind, row)) = row.split_first() else { break };
        let mut cur = row.to_vec();
        for i in 0..cur.len() {
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = prev[i];
            let c = if i >= bpp { prev[i - bpp] } else { 0 };
            cur[i] = cur[i].wrapping_add(match kind {
                1 => a,
                2 => b,
                3 => ((u16::from(a) + u16::from(b)) / 2) as u8,
                4 => {
                    let p = i16::from(a) + i16::from(b) - i16::from(c);
                    let (pa, pb, pc) = ((p - i16::from(a)).abs(), (p - i16::from(b)).abs(), (p - i16::from(c)).abs());
                    if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
                }
                _ => 0,
            });
        }
        out.extend_from_slice(&cur);
        prev = cur;
    }
    Ok(Some(out))
}

/// ASCIIHexDecode; `Ok` gives the output and the input bytes used.
pub(crate) fn ascii_hex(data: &[u8], cap: usize, budget: &Budget) -> Result<(Vec<u8>, usize), Exhausted> {
    let mut out = Vec::new();
    let mut half = None;
    let mut used = data.len();
    for (i, &b) in data.iter().enumerate() {
        if out.len() >= cap {
            return Ok((out, i));
        }
        budget.charge(Kind::Read, 1)?;
        if b == b'>' {
            used = i + 1;
            break;
        }
        if let Some(v) = hex_value(b) {
            match half.take() {
                None => half = Some(v),
                Some(h) => {
                    budget.charge(Kind::Decoded, 1)?;
                    out.push(h << 4 | v);
                }
            }
        }
    }
    if let Some(h) = half && out.len() < cap {
        budget.charge(Kind::Decoded, 1)?;
        out.push(h << 4);
    }
    Ok((out, used))
}

/// ASCII85Decode; `Ok` gives the output and the input bytes used.
pub(crate) fn ascii85(data: &[u8], cap: usize, budget: &Budget) -> Result<(Vec<u8>, usize), Exhausted> {
    let skip = if data.starts_with(b"<~") { 2 } else { 0 };
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut n = 0;
    let mut used = data.len();
    for (i, &b) in data.iter().enumerate().skip(skip) {
        if out.len() >= cap {
            out.truncate(cap);
            return Ok((out, i));
        }
        budget.charge(Kind::Read, 1)?;
        match b {
            b'~' => {
                used = (i + 2).min(data.len());
                break;
            }
            b'z' if n == 0 => {
                budget.charge(Kind::Decoded, 4)?;
                out.extend_from_slice(&[0; 4]);
            }
            b'!'..=b'u' => {
                group[n] = b - b'!';
                n += 1;
                if n == 5 {
                    let v = group.iter().fold(0u64, |a, &d| a * 85 + u64::from(d));
                    let Ok(v) = u32::try_from(v) else { return Ok((out, i)) };
                    budget.charge(Kind::Decoded, 4)?;
                    out.extend_from_slice(&v.to_be_bytes());
                    n = 0;
                }
            }
            _ => {}
        }
    }
    if n > 1 {
        group[n..].fill(84);
        let v = group.iter().fold(0u64, |a, &d| a * 85 + u64::from(d));
        if let Ok(v) = u32::try_from(v) {
            budget.charge(Kind::Decoded, n - 1)?;
            out.extend_from_slice(&v.to_be_bytes()[..n - 1]);
        }
    }
    out.truncate(cap);
    Ok((out, used))
}

/// RunLengthDecode; `Ok` gives the output and the input bytes used.
pub(crate) fn run_length(data: &[u8], cap: usize, budget: &Budget) -> Result<(Vec<u8>, usize), Exhausted> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(&len) = data.get(i) {
        if len == 128 {
            return Ok((out, i + 1));
        }
        if out.len() >= cap {
            break;
        }
        budget.charge(Kind::Steps, 1)?;
        let room = cap - out.len();
        if len < 128 {
            let end = (i + 2 + usize::from(len)).min(data.len());
            let literal = &data[i + 1..end];
            let n = literal.len().min(room);
            budget.charge(Kind::Decoded, n)?;
            out.extend_from_slice(&literal[..n]);
            i = end;
        } else {
            let Some(&b) = data.get(i + 1) else { break };
            let n = (257 - usize::from(len)).min(room);
            budget.charge(Kind::Decoded, n)?;
            out.extend(std::iter::repeat_n(b, n));
            i += 2;
        }
    }
    Ok((out, i.min(data.len())))
}

/// LZW with 9- to 12-bit codes, most significant bit first (7.4.4).
pub(crate) fn lzw(data: &[u8], early: bool, cap: usize, budget: &Budget) -> Result<Vec<u8>, Exhausted> {
    let mut out = Vec::new();
    let mut table: Vec<Vec<u8>> = Vec::new();
    let reset = |t: &mut Vec<Vec<u8>>| -> Result<(), Exhausted> {
        budget.charge(Kind::Tables, 1)?;
        budget.charge(Kind::Steps, 258)?;
        t.clear();
        t.extend((0..=255u8).map(|b| vec![b]));
        t.push(Vec::new());
        t.push(Vec::new());
        Ok(())
    };
    if cap == 0 {
        return Ok(out);
    }
    reset(&mut table)?;
    let (mut buf, mut count, mut width) = (0u32, 0u32, 9u32);
    let mut prev: Option<Vec<u8>> = None;
    for &b in data {
        budget.charge(Kind::Read, 1)?;
        buf = buf << 8 | u32::from(b);
        count += 8;
        while count >= width {
            let code = (buf >> (count - width)) as usize & ((1 << width) - 1);
            count -= width;
            if code == 256 {
                reset(&mut table)?;
                width = 9;
                prev = None;
                continue;
            }
            if code == 257 {
                return Ok(out);
            }
            // The entry is copied for the output and for the next table entry.
            let len = table.get(code).map_or_else(|| prev.as_ref().map_or(0, |p| p.len() + 1), Vec::len);
            budget.charge(Kind::Steps, 1 + len)?;
            let entry = match (table.get(code), &prev) {
                (Some(e), _) if code < 256 || !e.is_empty() => e.clone(),
                (_, Some(p)) if code == table.len() => {
                    let mut e = p.clone();
                    e.push(p[0]);
                    e
                }
                _ => return Ok(out),
            };
            if let Some(p) = prev.take() && table.len() < 4096 {
                let mut e = p;
                e.push(entry[0]);
                table.push(e);
            }
            // Stop at the cap; no append crosses it.
            let n = entry.len().min(cap - out.len());
            budget.charge(Kind::Decoded, n)?;
            out.extend_from_slice(&entry[..n]);
            if out.len() >= cap {
                return Ok(out);
            }
            prev = Some(entry);
            let limit = table.len() + usize::from(early);
            width = if limit >= 2048 { 12 } else if limit >= 1024 { 11 } else if limit >= 512 { 10 } else { 9 };
        }
    }
    Ok(out)
}
