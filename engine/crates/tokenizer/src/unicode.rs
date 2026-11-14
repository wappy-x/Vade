//! Independently authored Unicode normalization and category lookup.
//! Tables use Unicode16 categories and Unicode12.1 normalization data; see UNICODE-NOTICE.txt.
mod data {
    include!("unicode_data.rs");
}
pub fn letter(c: char) -> bool {
    #[cfg(test)] crate::work_counts::add();
    if c.is_ascii() { return c.is_ascii_alphabetic(); }
    in_ranges(c as u32, data::LETTERS)
}
pub fn number(c: char) -> bool {
    #[cfg(test)] crate::work_counts::add();
    if c.is_ascii() { return c.is_ascii_digit(); }
    in_ranges(c as u32, data::NUMBERS)
}
pub fn mark(c: char) -> bool { #[cfg(test)] crate::work_counts::add(); in_ranges(c as u32, data::MARKS) }
fn in_ranges(c: u32, ranges: &[(u32, u32)]) -> bool {
    let i = ranges.partition_point(|&(lo, _)| lo <= c);
    i > 0 && c <= ranges[i - 1].1
}
fn ccc(c: u32) -> u8 {
    #[cfg(test)] crate::work_counts::add();
    data::CCC
        .binary_search_by_key(&c, |x| x.0)
        .map(|i| data::CCC[i].1)
        .unwrap_or(0)
}
fn decompose(c: u32, out: &mut Vec<u32>) {
    #[cfg(test)] crate::work_counts::add();
    if (0xac00..0xd7a4).contains(&c) {
        let s = c - 0xac00;
        out.push(0x1100 + s / 588);
        out.push(0x1161 + (s % 588) / 28);
        if !s.is_multiple_of(28) {
            out.push(0x11a7 + s % 28)
        }
    } else if let Ok(i) = data::DECOMP.binary_search_by_key(&c, |x| x.0) {
        for &d in data::DECOMP[i].1 {
            decompose(d, out)
        }
    } else {
        out.push(c)
    }
}
fn compose(a: u32, b: u32) -> Option<u32> {
    #[cfg(test)] crate::work_counts::add();
    if (0x1100..0x1113).contains(&a) && (0x1161..0x1176).contains(&b) {
        return Some(0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28);
    }
    if (0xac00..0xd7a4).contains(&a) && (a - 0xac00).is_multiple_of(28) && (0x11a8..0x11c3).contains(&b) {
        return Some(a + b - 0x11a7);
    }
    data::COMPOSE
        .binary_search_by_key(&(a, b), |x| (x.0, x.1))
        .ok()
        .map(|i| data::COMPOSE[i].2)
}
pub fn nfc(s: &str, limit: usize) -> crate::Result<String> {
    if s.len() > limit { return Err(crate::err("normalized byte limit exceeded")); }
    if s.is_ascii() { return Ok(s.to_owned()); }
    let mut d = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose(c as u32, &mut d);
        if d.len() > limit { return Err(crate::err("normalization work limit exceeded")); }
    }
    let mut start = 0;
    for i in 0..=d.len() {
        if i == d.len() || ccc(d[i]) == 0 {
            d[start..i].sort_by_key(|&c| ccc(c));
            start = i + 1
        }
    }
    let mut out = Vec::with_capacity(d.len());
    let mut starter = None;
    let mut previous = 0;
    for c in d {
        let class = ccc(c);
        if let Some(i) = starter
            && (previous == 0 || previous < class)
                && let Some(combined) = compose(out[i], c)
            {
                out[i] = combined;
                continue;
            }
        if class == 0 {
            starter = Some(out.len())
        }
        previous = class;
        out.push(c);
    }
    if out.iter().map(|&c| char::from_u32(c).unwrap().len_utf8()).sum::<usize>() > limit { return Err(crate::err("normalized byte limit exceeded")); }
    Ok(out.into_iter()
        .map(|c| char::from_u32(c).unwrap())
        .collect())
}
