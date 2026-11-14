/// The longest prefix of `input` that is at most `max_bytes` bytes long.
pub fn truncate_utf8_prefix(input: &str, max_bytes: usize) -> &str {
    if input.len() <= max_bytes {
        return input;
    }
    &input[..max_bytes]
}
