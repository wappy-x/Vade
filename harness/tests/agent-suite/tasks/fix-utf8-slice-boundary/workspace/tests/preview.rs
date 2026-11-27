use preview::truncate_utf8_prefix;

#[test]
fn ascii_exact_slice() {
    assert_eq!(truncate_utf8_prefix("abcdef", 4), "abcd");
}

#[test]
fn multibyte_steps_back_to_char_boundary() {
    // 'a' is 1 byte (0..1), 'é' is 2 bytes (1..3), '€' is 3 bytes (3..6), 'z' is 1 byte (6..7).
    let s = "a\u{00e9}\u{20ac}z";
    assert_eq!(truncate_utf8_prefix(s, 0), "");
    assert_eq!(truncate_utf8_prefix(s, 1), "a");
    assert_eq!(truncate_utf8_prefix(s, 2), "a");
    assert_eq!(truncate_utf8_prefix(s, 3), "a\u{00e9}");
    assert_eq!(truncate_utf8_prefix(s, 4), "a\u{00e9}");
    assert_eq!(truncate_utf8_prefix(s, 5), "a\u{00e9}");
    assert_eq!(truncate_utf8_prefix(s, 6), "a\u{00e9}\u{20ac}");
    assert_eq!(truncate_utf8_prefix(s, 10), s);
}
