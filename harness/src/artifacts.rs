//! Session-scoped, bounded output snapshots. IDs are local journal record numbers,
//! never filenames, credentials, cache handles, or content-security hashes.
use crate::json::{self, Value};
use std::io;

pub const MAX_ARTIFACT_BYTES: usize = 64 * 1024;
pub const MAX_ARTIFACTS: usize = 128;
pub const MAX_TOTAL_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_CHUNK_BYTES: usize = 4096;
pub const INLINE_BYTES: usize = 4096;
pub const LIFETIME_SECONDS: u64 = 24 * 60 * 60;

pub trait Access {
    /// Called before a tool may execute. Single-writer ownership makes this a
    /// reservation check; no other worker may consume capacity in between.
    fn check_capacity(&mut self) -> io::Result<()>;
    fn retain(&mut self, text: &str) -> io::Result<String>;
    fn read(&mut self, id: &str, offset: usize, length: usize, cap: usize) -> io::Result<String>;
}
#[derive(Default)]
pub(crate) struct Archive {
    entries: Vec<Entry>,
    bytes: usize,
}
struct Entry {
    id: String,
    created: u64,
    text: String,
}
fn error(s: &str) -> io::Error {
    io::Error::other(s)
}
pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
impl Archive {
    pub fn check_capacity(&self) -> io::Result<()> {
        if self.entries.len() >= MAX_ARTIFACTS || self.bytes + MAX_ARTIFACT_BYTES > MAX_TOTAL_BYTES
        {
            return Err(error("artifact quota exhausted; start a new session"));
        }
        Ok(())
    }
    pub fn insert(&mut self, id: String, created: u64, text: String) -> io::Result<()> {
        if text.len() > MAX_ARTIFACT_BYTES
            || self.entries.len() >= MAX_ARTIFACTS
            || self.bytes + text.len() > MAX_TOTAL_BYTES
            || self.entries.iter().any(|e| e.id == id)
        {
            return Err(error("invalid artifact or artifact quota exceeded"));
        }
        self.bytes += text.len();
        self.entries.push(Entry { id, created, text });
        Ok(())
    }
    pub fn read(&self, id: &str, offset: usize, length: usize, cap: usize) -> io::Result<String> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| error("artifact not found in this session"))?;
        if now() >= entry.created.saturating_add(LIFETIME_SECONDS) {
            return Err(error(
                "artifact retrieval expired; raw journal retains the captured output",
            ));
        }
        chunk(&entry.text, id, offset, length, cap)
    }
}
/// Offsets are UTF-8 byte offsets; callers get the next exact boundary back.
/// An invalid boundary is refused rather than silently skipping source bytes.
pub fn chunk(text: &str, id: &str, offset: usize, length: usize, cap: usize) -> io::Result<String> {
    if length == 0
        || length > MAX_CHUNK_BYTES
        || offset > text.len()
        || !text.is_char_boundary(offset)
    {
        return Err(error("range needs a UTF-8 byte boundary and length 1–4096"));
    }
    let mut end = (offset + length).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    loop {
        let result = json::stringify(&Value::Object(vec![
            ("artifact".into(), Value::String(id.into())),
            ("total_bytes".into(), Value::Number(text.len().to_string())),
            ("offset".into(), Value::Number(offset.to_string())),
            ("next_offset".into(), Value::Number(end.to_string())),
            ("eof".into(), Value::Bool(end == text.len())),
            ("text".into(), Value::String(text[offset..end].into())),
        ]))?;
        if result.len() <= cap && (end > offset || offset == text.len()) {
            return Ok(result);
        }
        if end == offset {
            return Err(error("output budget too small for a chunk"));
        }
        end -= 1;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
    }
}
pub fn excerpt(text: &str, id: &str, cap: usize) -> io::Result<String> {
    let range = chunk(text, id, 0, INLINE_BYTES, cap.saturating_sub(360))?;
    Ok(format!(
        "Captured tool result excerpt; omitted bytes remain in a session artifact for 24 hours. For source code, use workspace_search and workspace_read_lines to retrieve only relevant ranges. Use tool_output_read only for needed captured bytes; avoid whole-artifact paging. Capture limits still apply.\n{range}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utf8_and_escaped_chunks_are_bounded_and_reassemble_exactly() {
        let text = "a\"\\\n🦀λ".repeat(1000);
        let mut out = String::new();
        let mut offset = 0;
        while offset < text.len() {
            let result = chunk(&text, "a-1", offset, 333, 512).unwrap();
            assert!(result.len() <= 512);
            let v = json::parse(&result).unwrap();
            out.push_str(v.get("text").unwrap().as_str().unwrap());
            let Value::Number(next) = v.get("next_offset").unwrap() else {
                panic!()
            };
            offset = next.parse().unwrap();
        }
        assert_eq!(out, text);
        assert!(chunk("🦀", "a", 1, 10, 512).is_err());
        assert!(chunk("🦀", "a", 0, 1, 512).is_err());
        assert!(chunk("x", "a", usize::MAX, 1, 512).is_err());
        assert!(chunk("x", "a", 0, MAX_CHUNK_BYTES + 1, 512).is_err());
    }
    #[test]
    fn quota_expiry_and_session_scope_fail_closed() {
        let mut a = Archive::default();
        a.insert("a-0".into(), 0, "expired".into()).unwrap();
        assert!(a.read("a-0", 0, 100, 512).is_err());
        assert!(a.read("other-session", 0, 100, 512).is_err());
        assert!(a.insert("a-0".into(), now(), "duplicate".into()).is_err());
        for i in 1..MAX_ARTIFACTS {
            a.insert(format!("a-{i}"), now(), "x".into()).unwrap();
        }
        assert!(a.check_capacity().is_err());
        assert!(a.insert("too-many".into(), now(), "x".into()).is_err());
        let mut a = Archive::default();
        for i in 0..MAX_TOTAL_BYTES / MAX_ARTIFACT_BYTES {
            a.insert(format!("a-{i}"), now(), "x".repeat(MAX_ARTIFACT_BYTES))
                .unwrap();
        }
        assert!(a.check_capacity().is_err());
    }
}
