//! Split rule for the case-aware pattern (`CASED_PATTERN` in lib.rs).
//!
//! The source of the rule is the `Regex` string of the `Split` pre-tokenizer
//! in a checkpoint's `tokenizer.json`. The loader selects this rule when the
//! string equals `CASED_PATTERN`. No code of a regular expression library is
//! used here.
//!
//! The pattern is a regular expression with seven alternatives. A
//! backtracking engine tries them in order at each position, and the first
//! one that matches gives the next piece:
//!
//! 1. `P? U* L+ S?`: a word that ends in the lower class.
//! 2. `P? U+ L* S?`: a word that ends in the upper class.
//! 3. `\p{N}{1,3}`: one to three numbers.
//! 4. ` ?Q+[\r\n/]*`: symbols, then line breaks and slashes.
//! 5. `\s*[\r\n]+`: spaces up to the last line break.
//! 6. `\s+(?!\S)`: spaces, less the last one when text follows.
//! 7. `\s+`: the spaces that are left.
//!
//! - `P` is `[^\r\n\p{L}\p{N}]`, one optional character before a word.
//! - `U` is `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and `L` is `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
//! - `S` is `(?i:'s|'t|'re|'ve|'m|'ll|'d)`.
//! - `Q` is `[^\s\p{L}\p{N}]`.
//!
//! Each function below is one alternative. It makes the same choices, in the
//! same order, as the backtracking engine. The work is linear in the text:
//! a scan that fails is followed by a piece that covers the scanned span.
use crate::unicode;

const UPPER_SIDE: u8 = 1;
const LOWER_SIDE: u8 = 2;
const NUMBER: u8 = 4;
const SPACE: u8 = 8;
const PREFIX: u8 = 16;
const SYMBOL: u8 = 32;
struct Unit { at: usize, c: char, class: u8 }
fn class(c: char) -> u8 {
    if unicode::letter(c) {
        // Lm and Lo are in both word classes. A letter is not P and not Q.
        if unicode::lowercase(c) { LOWER_SIDE } else if unicode::uppercase_or_titlecase(c) { UPPER_SIDE } else { UPPER_SIDE | LOWER_SIDE }
    } else if unicode::number(c) { NUMBER }
    else if unicode::mark(c) { UPPER_SIDE | LOWER_SIDE | PREFIX | SYMBOL }
    else if crate::space(c) { if matches!(c, '\r' | '\n') { SPACE } else { SPACE | PREFIX } }
    else { PREFIX | SYMBOL }
}
fn is(u: &[Unit], at: usize, mask: u8) -> bool { u.get(at).is_some_and(|x| x.class & mask != 0) }
fn count(u: &[Unit], from: usize, mask: u8) -> usize {
    let mut n = 0;
    while is(u, from + n, mask) { #[cfg(test)] crate::work_counts::add(); n += 1; }
    n
}
/// Where a word can start: after the optional `P` character first, then at `i`.
fn word_starts(u: &[Unit], i: usize) -> impl Iterator<Item = usize> {
    [is(u, i, PREFIX).then_some(i + 1), Some(i)].into_iter().flatten()
}
/// `S?`: the length of a contraction at `at`, or 0.
fn suffix(u: &[Unit], at: usize) -> usize {
    if u.get(at).is_none_or(|x| x.c != '\'') { return 0; }
    for word in ["s", "t", "re", "ve", "m", "ll", "d"] {
        // Case folding also matches the long s (U+017F) with `s`.
        if word.chars().enumerate().all(|(j, a)| u.get(at + 1 + j).is_some_and(|x| x.c.eq_ignore_ascii_case(&a) || (a == 's' && x.c == 'ſ'))) {
            return 1 + word.len();
        }
    }
    0
}
/// Alternative 1.
fn word_lower(u: &[Unit], i: usize) -> Option<usize> {
    for start in word_starts(u, i) {
        // `U*` takes the longest run. Then it gives characters back, one at
        // a time, until `L+` can start.
        let mut k = start + count(u, start, UPPER_SIDE);
        loop {
            #[cfg(test)] crate::work_counts::add();
            if is(u, k, LOWER_SIDE) {
                let end = k + count(u, k, LOWER_SIDE);
                return Some(end + suffix(u, end));
            }
            if k == start { break; }
            k -= 1;
        }
    }
    None
}
/// Alternative 2.
fn word_upper(u: &[Unit], i: usize) -> Option<usize> {
    for start in word_starts(u, i) {
        let run = count(u, start, UPPER_SIDE);
        if run > 0 {
            let end = start + run + count(u, start + run, LOWER_SIDE);
            return Some(end + suffix(u, end));
        }
    }
    None
}
/// Alternative 3.
fn numbers(u: &[Unit], i: usize) -> Option<usize> {
    let run = (0..3).take_while(|&n| is(u, i + n, NUMBER)).count();
    (run > 0).then_some(i + run)
}
/// Alternative 4. A space that no symbol follows is not a symbol itself, so
/// the match without the optional space fails too.
fn symbols(u: &[Unit], i: usize) -> Option<usize> {
    let start = if u[i].c == ' ' { i + 1 } else { i };
    let run = count(u, start, SYMBOL);
    if run == 0 { return None; }
    let mut end = start + run;
    while u.get(end).is_some_and(|x| matches!(x.c, '\r' | '\n' | '/')) { end += 1; }
    Some(end)
}
/// Alternatives 5, 6 and 7.
fn spaces(u: &[Unit], i: usize) -> Option<usize> {
    let run = count(u, i, SPACE);
    if run == 0 { return None; }
    let end = i + run;
    if let Some(last) = (i..end).rev().find(|&k| matches!(u[k].c, '\r' | '\n')) { return Some(last + 1); }
    Some(if end == u.len() || run == 1 { end } else { end - 1 })
}
pub(crate) fn split(text: &str) -> Vec<&str> {
    let units: Vec<Unit> = text.char_indices().map(|(at, c)| Unit { at, c, class: class(c) }).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < units.len() {
        #[cfg(test)] crate::work_counts::add();
        // Every character is in one of the classes, so one alternative matches.
        let end = word_lower(&units, i).or_else(|| word_upper(&units, i)).or_else(|| numbers(&units, i))
            .or_else(|| symbols(&units, i)).or_else(|| spaces(&units, i)).unwrap_or(i + 1);
        // A piece is never empty. The floor keeps the loop finite if a rule is wrong.
        debug_assert!(end > i);
        let end = end.max(i + 1);
        out.push(&text[units[i].at..units.get(end).map_or(text.len(), |x| x.at)]);
        i = end;
    }
    out
}
#[cfg(test)]
mod tests {
    use super::split;
    use crate::unicode;
    use engine_model::json::{self, field};
    /// One JSON object a line: `text`, and `ends`, the end of each piece in
    /// characters. The `tokenizers` library, version 0.23.3, cut the pieces
    /// with the same pattern.
    const VECTORS: &str = include_str!("../tests/fixtures/cased_split/pieces.jsonl");
    #[test]
    fn pieces_equal_the_reference_vectors() {
        let (mut lines, mut pieces) = (0, 0);
        for line in VECTORS.lines() {
            let value = json::parse(line.as_bytes(), json::Limits::default()).unwrap();
            let object = value.object().unwrap();
            let text = field(object, "text").unwrap().string().unwrap();
            let expected: Vec<usize> = field(object, "ends").unwrap().array().unwrap().iter().map(|end| end.u64().unwrap() as usize).collect();
            let actual = split(text);
            assert_eq!(actual.concat(), text);
            let mut end = 0;
            let ends: Vec<usize> = actual.iter().map(|piece| { end += piece.chars().count(); end }).collect();
            assert_eq!(ends, expected, "{text:?} gave {actual:?}");
            lines += 1; pieces += expected.len();
        }
        // The counts of the fixture file: `wc -l`, and the sum of its `ends` lengths.
        assert_eq!((lines, pieces), (1135, 2967));
    }
    // The pieces in the next two tests are also from the reference library.
    #[test]
    fn words_split_where_the_case_rises() {
        assert_eq!(split("HelloWorld HTTPServer iOS"), ["Hello", "World", " HTTPServer", " i", "OS"]);
        // An uncased letter is in both word classes. A mark is too.
        assert_eq!(split("中文ABC ABC中文 ǅa"), ["中文", "ABC", " ABC中文", " ǅa"]);
        assert_eq!(split("\u{301}ABC A\u{301}B !\u{301}a"), ["\u{301}", "ABC", " A\u{301}", "B", " !\u{301}", "a"]);
        assert_eq!(split("it's IT'S a'ſ x'y"), ["it's", " IT'S", " a'ſ", " x", "'y"]);
    }
    #[test]
    fn numbers_spaces_and_symbols() {
        assert_eq!(split("1234567 ١٢٣٤ a12"), ["123", "456", "7", " ", "١٢٣", "٤", " a", "12"]);
        assert_eq!(split("a  !  \n  b\n"), ["a", " ", " !", "  \n", " ", " b", "\n"]);
        assert_eq!(split("x = y;\n//z\n/"), ["x", " =", " y", ";\n//", "z", "\n", "/"]);
    }
    /// The tables agree with UnicodeData 16.0.0 for every code point. Each
    /// pair is the count of a class and the FNV-1a 64 hash of its code
    /// points, as 4 little-endian bytes each, in rising order.
    #[test]
    fn class_tables_cover_every_code_point() {
        let classes: [(&str, fn(char) -> bool, usize, u64); 5] = [
            ("L", unicode::letter, 141_028, 0xf41b_f722_3945_5b4f),
            ("N", unicode::number, 1_911, 0xd155_86ee_8a14_5a41),
            ("M", unicode::mark, 2_501, 0xd996_d5f3_b939_0f58),
            ("Ll", unicode::lowercase, 2_258, 0x927b_ffdf_46d7_c43d),
            ("Lu and Lt", unicode::uppercase_or_titlecase, 1_889, 0xb0bd_1411_e4c8_2315),
        ];
        for (name, member, count, hash) in classes {
            let (mut n, mut h) = (0, 0xcbf2_9ce4_8422_2325u64);
            for c in ('\0'..=char::MAX).filter(|&c| member(c)) {
                n += 1;
                for b in (c as u32).to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3); }
            }
            assert_eq!((n, h), (count, hash), "{name}");
        }
        // The cased classes are inside L, and no code point is in two of them.
        for c in '\0'..=char::MAX {
            let (lower, upper) = (unicode::lowercase(c), unicode::uppercase_or_titlecase(c));
            assert!(!(lower && upper) && (unicode::letter(c) || !(lower || upper)), "U+{:04X}", c as u32);
        }
    }
}
