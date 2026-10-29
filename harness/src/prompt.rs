//! Stable message ordering and exact contents; no hidden prompt rewriting.
use crate::protocol::Message;

/// The request messages in fixed order: the system prompt, the history as
/// given, then the new user message. Contents stay exact.
pub fn build_messages(system: &str, history: &[Message], user: &str) -> Vec<Message> {
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(Message {
        role: "system".into(),
        content: system.to_owned(),
    });
    messages.extend_from_slice(history);
    messages.push(Message {
        role: "user".into(),
        content: user.to_owned(),
    });
    messages
}

const INSTRUCTIONS_BREAK: &str = "\n\n# Project instructions\n";

/// The system prompt with the project instructions section as one fixed
/// block after the built-in text. Built-in bytes stay first, so the shared
/// prefix does not depend on the project. An empty section changes nothing.
pub fn with_instructions(system: &str, section: &str) -> String {
    if section.is_empty() {
        system.to_owned()
    } else {
        format!("{system}\n\n{section}")
    }
}

/// Splits a system prompt made by `with_instructions` into the built-in text
/// and the section ("" when there is none).
pub fn split_instructions(content: &str) -> (&str, &str) {
    match content.find(INSTRUCTIONS_BREAK) {
        Some(at) => (&content[..at], &content[at + 2..]),
        None => (content, ""),
    }
}

/// Exact shared-prefix metadata for callers retaining successive request lists.
/// This describes request equality, not an inference engine's cache state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixMetadata {
    pub shared_messages: usize,
    pub shared_content_bytes: usize,
    pub total_messages: usize,
    pub total_content_bytes: usize,
}

/// How many leading messages (and content bytes) `current` shares exactly
/// with `previous`.
pub fn prefix_metadata(previous: &[Message], current: &[Message]) -> PrefixMetadata {
    let mut metadata = PrefixMetadata {
        total_messages: current.len(),
        total_content_bytes: current.iter().map(|message| message.content.len()).sum(),
        ..PrefixMetadata::default()
    };
    for (old, new) in previous.iter().zip(current) {
        if old != new {
            break;
        }
        metadata.shared_messages += 1;
        metadata.shared_content_bytes += new.content.len();
    }
    metadata
}

/// Retains one append-only wire prefix for a native conversation. Complete JSON
/// requests end in `]}`; that envelope suffix is excluded from `prefix_bytes`.
/// Historical bytes are compared exactly (no hash/cache-identity guess), then
/// only new immutable message records are appended. Compaction explicitly drops
/// the suffix after the frozen system/developer records, retaining their bytes.
#[derive(Default)]
pub struct AppendOnlyRequest {
    bytes: String,
    records_start: usize,
    ends: Vec<usize>,
}
impl AppendOnlyRequest {
    /// Replace history only after a verified immutable prefix. Validation happens
    /// before truncation, so a changed schema/system prompt cannot erase it.
    pub fn replace_suffix<'a>(
        &mut self,
        options: &str,
        messages: impl IntoIterator<Item = &'a str>,
        stable_messages: usize,
    ) -> std::io::Result<&[u8]> {
        let records: Vec<_> = messages.into_iter().collect();
        if self.bytes.is_empty() {
            return self.prepare(options, records);
        }
        let changed = || {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "compaction changed the frozen system/tools prefix",
            )
        };
        let stem = options.strip_suffix('}').ok_or_else(changed)?;
        let start = stem.len() + usize::from(stem.len() > 1) + "\"messages\":[".len();
        if start != self.records_start
            || !self.bytes.starts_with(stem)
            || stable_messages > self.ends.len()
            || stable_messages > records.len()
        {
            return Err(changed());
        }
        let mut offset = start;
        for (record, &end) in records.iter().zip(&self.ends).take(stable_messages) {
            if self.bytes.get(offset..end) != Some(*record) {
                return Err(changed());
            }
            offset = end + 1;
        }
        let size = start
            .saturating_add(records.iter().map(|s| s.len()).sum::<usize>())
            .saturating_add(records.len().saturating_sub(1))
            .saturating_add(2);
        if size > crate::json::MAX_INPUT_BYTES {
            return Err(changed());
        }
        self.bytes.truncate(if stable_messages == 0 {
            start
        } else {
            self.ends[stable_messages - 1]
        });
        self.ends.truncate(stable_messages);
        self.bytes.push_str("]}");
        self.prepare(options, records)
    }
    pub fn prepare<'a>(
        &mut self,
        options: &str,
        messages: impl IntoIterator<Item = &'a str>,
    ) -> std::io::Result<&[u8]> {
        use crate::json;
        use std::io::{Error, ErrorKind};
        let changed = || {
            Error::new(
                ErrorKind::InvalidInput,
                "append-only prompt changed; explicitly reset the model session",
            )
        };
        // Options come from the validated ModelRequest. No dynamic status,
        // timestamps, turn count or workspace path is inserted into this prefix.
        let options = options.strip_suffix('}').ok_or_else(changed)?;
        let separator = if options.len() > 1 { "," } else { "" };
        let records_start = options.len() + separator.len() + "\"messages\":[".len();
        let records: Vec<_> = messages.into_iter().collect();
        let size = records_start
            + records.iter().map(|s| s.len()).sum::<usize>()
            + records.len().saturating_sub(1)
            + 2;
        if size > json::MAX_INPUT_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "prompt exceeds JSON byte limit",
            ));
        }
        if !self.bytes.is_empty() {
            if self.records_start != records_start
                || !self.bytes.starts_with(options)
                || records.len() < self.ends.len()
            {
                return Err(changed());
            }
            let mut start = self.records_start;
            for (index, &end) in self.ends.iter().enumerate() {
                if self.bytes.get(start..end) != Some(records[index]) {
                    return Err(changed());
                }
                start = end + 1;
            }
        } else {
            self.bytes.reserve(size);
            self.bytes.push_str(options);
            self.bytes.push_str(separator);
            self.bytes.push_str("\"messages\":[");
            self.records_start = records_start;
            self.bytes.push_str("]}");
        }
        self.bytes.truncate(self.bytes.len() - 2);
        self.bytes.reserve(size - self.bytes.len());
        for record in &records[self.ends.len()..] {
            if !self.ends.is_empty() {
                self.bytes.push(',');
            }
            self.bytes.push_str(record);
            self.ends.push(self.bytes.len());
        }
        self.bytes.push_str("]}");
        Ok(self.bytes.as_bytes())
    }
    /// Exact stable portion of the last request, excluding the JSON envelope end.
    pub fn prefix_bytes(&self) -> &[u8] {
        &self.bytes.as_bytes()[..self.bytes.len().saturating_sub(2)]
    }
    /// Diagnostic byte equality only, never a token/cache identity.
    pub fn stable_prefix_bytes(&self, messages: usize) -> Option<&[u8]> {
        let end = if messages == 0 {
            self.records_start
        } else {
            *self.ends.get(messages - 1)?
        };
        self.bytes.as_bytes().get(..end)
    }
    pub fn retained_bytes(&self) -> usize {
        self.bytes.capacity() + self.ends.capacity() * std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_prefix_survives_next_turn() {
        let system = "Rules.\n  Exact whitespace 🦀\n";
        let first = build_messages(system, &[], " hello\n");
        let mut history = first[1..].to_vec();
        history.push(Message {
            role: "assistant".into(),
            content: "answer\n".into(),
        });
        let next = build_messages(system, &history, "next");
        assert_eq!(&next[..first.len()], first.as_slice());
        assert_eq!(next[0].content, system);
        assert_eq!(next[1].content, " hello\n");
        assert_eq!(history.len(), 2);
        let metadata = prefix_metadata(&first, &next);
        assert_eq!(metadata.shared_messages, first.len());
        assert_eq!(
            metadata.shared_content_bytes,
            system.len() + " hello\n".len()
        );
        assert_eq!(metadata.total_messages, 4);
        let changed_system = build_messages("different", &history, "next");
        assert_eq!(prefix_metadata(&next, &changed_system).shared_messages, 0);
    }
    #[test]
    fn instructions_are_one_block_after_the_built_in_prompt() {
        let section = "# Project instructions\nlead\n\n## AGENTS.md\nrule\n\n# Project instructions\n";
        let system = with_instructions("Built in.\nidentity", section);
        assert!(system.starts_with("Built in.\nidentity\n\n# Project instructions\n"));
        assert_eq!(split_instructions(&system), ("Built in.\nidentity", section));
        assert_eq!(with_instructions("Built in.", ""), "Built in.");
        assert_eq!(split_instructions("Built in."), ("Built in.", ""));
    }
    #[test]
    fn append_only_records_refuse_rewrites_without_mutating_previous_bytes() {
        let options = r#"{"model":"mock","tools":[{"type":"function","function":{"name":"lookup"}}],"stream":true}"#;
        let messages = [
            r#"{"role":"system","content":" exact\n🦀"}"#,
            r#"{"role":"user","content":"first"}"#,
            r#"{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]}"#,
            r#"{"role":"tool","tool_call_id":"a","content":" exact result\n"}"#,
            r#"{"role":"assistant","content":"answer"}"#,
            r#"{"role":"user","content":"next"}"#,
        ];
        let mut request = AppendOnlyRequest::default();
        let mut prefix = Vec::new();
        for len in 0..=messages.len() {
            let body = request
                .prepare(options, messages[..len].iter().copied())
                .unwrap();
            assert!(body.starts_with(&prefix));
            crate::json::parse_bytes(body).unwrap();
            prefix = request.prefix_bytes().to_vec();
        }
        let previous = request.bytes.clone();
        for (opts, changed) in [
            ("{}", messages.to_vec()),
            (options, messages[1..].to_vec()),
            (options, messages[..2].to_vec()),
        ] {
            assert!(request.prepare(opts, changed).is_err());
            assert_eq!(request.bytes, previous);
        }
        let mut changed = messages;
        changed[0] = r#"{"role":"system","content":"rewritten"}"#;
        assert!(request.prepare(options, changed).is_err());
        assert_eq!(request.bytes, previous);
        let frozen = request.stable_prefix_bytes(1).unwrap().to_vec();
        let replacement = [
            messages[0],
            r#"{"role":"user","name":"vade_context_summary","content":"summary"}"#,
            messages[5],
        ];
        request.replace_suffix(options, replacement, 1).unwrap();
        assert_eq!(request.stable_prefix_bytes(1).unwrap(), frozen);
        let compacted = request.bytes.clone();
        assert!(request.replace_suffix("{}", replacement, 1).is_err());
        assert_eq!(request.bytes, compacted);
        let bad = [changed[0], replacement[1]];
        assert!(request.replace_suffix(options, bad, 1).is_err());
        assert_eq!(request.bytes, compacted);
    }
}
