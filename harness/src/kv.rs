//! Optional Engine-owned KV hints. No tokenization, raw KV or cache identity is
//! implemented here. The transcript remains authoritative on every request.
use crate::{
    json::{self, Value},
    transport::invalid,
};
use std::{
    fmt, io,
    time::{SystemTime, UNIX_EPOCH},
};

pub(crate) const MAX_HINTS: usize = 2;
pub(crate) const MAX_HANDLES: usize = 16;

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
fn text(v: json::View<'_>, key: &str) -> io::Result<String> {
    let s = v
        .get(key)
        .and_then(json::View::as_str)
        .ok_or_else(|| invalid("invalid KV metadata"))?;
    if s.is_empty() || s.len() > 512 || s.chars().any(char::is_control) {
        return Err(invalid("invalid KV metadata"));
    }
    Ok(s.into_owned())
}
fn count(v: json::View<'_>, key: &str) -> io::Result<u64> {
    let raw = v.get(key).ok_or_else(|| invalid("missing KV count"))?.raw();
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid("invalid KV count"));
    }
    raw.parse().map_err(|_| invalid("invalid KV count"))
}

/// The engine's advertised checkpoint support: its cache domain, per-scope
/// handle and byte limits, and handle lifetime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Authorization-specific; may differ from the ordinary shared cache domain.
    pub cache_domain: String,
    pub max_handles_per_scope: u64,
    pub max_bytes_per_scope: u64,
    pub ttl_ms: u64,
}
impl Capabilities {
    pub(crate) fn parse(root: json::View<'_>) -> io::Result<Option<Self>> {
        if root
            .get("kv_branch_handles")
            .is_none_or(|v| v.raw() != "true")
        {
            return Ok(None);
        }
        let v = root
            .get("kv_checkpoints")
            .ok_or_else(|| invalid("missing KV capabilities"))?;
        // Never follow a remote-supplied URL or release on a different origin.
        if text(v, "scope")? != "authorization" || text(v, "release")? != "/v1/kv/release" {
            return Err(invalid("unsupported KV scope or release route"));
        }
        let caps = Self {
            cache_domain: text(v, "cache_domain")?,
            max_handles_per_scope: count(v, "max_handles_per_scope")?,
            max_bytes_per_scope: count(v, "max_bytes_per_scope")?,
            ttl_ms: count(v, "ttl_ms")?,
        };
        if caps.max_handles_per_scope == 0 || caps.max_bytes_per_scope == 0 || caps.ttl_ms == 0 {
            return Err(invalid("invalid KV quotas"));
        }
        Ok(Some(caps))
    }
    pub(crate) fn limit(&self) -> usize {
        self.max_handles_per_scope.min(MAX_HANDLES as u64) as usize
    }
}

/// A private capability, never displayed by Debug or context diagnostics.
/// Journal serialization is deliberately separate from model-visible messages.
#[derive(Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub(crate) handle: String,
    pub cache_domain: String,
    pub prompt_tokens: u64,
    pub reusable_prefix_tokens: u64,
    pub expires_at_ms: u64,
    pub retained_bytes: u64,
    /// Engine's deduplicated scope total at issuance; never sum per-handle bytes.
    pub scope_retained_bytes: u64,
}
impl fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Checkpoint")
            .field("handle", &"[redacted]")
            .field("prompt_tokens", &self.prompt_tokens)
            .field("reusable_prefix_tokens", &self.reusable_prefix_tokens)
            .field("expires_at_ms", &self.expires_at_ms)
            .field("retained_bytes", &self.retained_bytes)
            .finish_non_exhaustive()
    }
}
impl Checkpoint {
    fn parse(v: json::View<'_>, cache_domain: String) -> io::Result<Self> {
        let handle = text(v, "handle")?;
        if handle.len() != 64
            || !handle
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(invalid("invalid KV handle"));
        }
        let result = Self {
            handle,
            cache_domain,
            prompt_tokens: count(v, "prompt_tokens")?,
            reusable_prefix_tokens: count(v, "reusable_prefix_tokens")?,
            expires_at_ms: count(v, "expires_at_ms")?,
            retained_bytes: count(v, "retained_bytes")?,
            scope_retained_bytes: count(v, "scope_retained_bytes")?,
        };
        if result.prompt_tokens == 0
            || result.reusable_prefix_tokens > result.prompt_tokens
            || result.retained_bytes == 0
            || result.scope_retained_bytes < result.retained_bytes
        {
            return Err(invalid("inconsistent KV checkpoint"));
        }
        Ok(result)
    }
    pub fn eligible(&self, caps: &Capabilities, at_ms: u64) -> bool {
        self.cache_domain == caps.cache_domain
            && self.expires_at_ms > at_ms
            && self.retained_bytes <= caps.max_bytes_per_scope
            && self.scope_retained_bytes <= caps.max_bytes_per_scope
    }
    pub(crate) fn value(&self) -> Value {
        let mut fields = vec![
            ("handle".into(), Value::String(self.handle.clone())),
            (
                "cache_domain".into(),
                Value::String(self.cache_domain.clone()),
            ),
        ];
        for (name, n) in [
            ("prompt_tokens", self.prompt_tokens),
            ("reusable_prefix_tokens", self.reusable_prefix_tokens),
            ("expires_at_ms", self.expires_at_ms),
            ("retained_bytes", self.retained_bytes),
            ("scope_retained_bytes", self.scope_retained_bytes),
        ] {
            fields.push((name.into(), Value::Number(n.to_string())));
        }
        Value::Object(fields)
    }
}

/// Binds the returned canonical **input** checkpoint to an unchanged transcript
/// prefix. Generated answer tokens are not included in this boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hint {
    pub checkpoint: Checkpoint,
    pub messages: usize,
}
impl Hint {
    pub(crate) fn value(&self) -> Value {
        Value::Object(vec![
            ("checkpoint".into(), self.checkpoint.value()),
            ("messages".into(), Value::Number(self.messages.to_string())),
        ])
    }
    fn parse(v: json::View<'_>) -> io::Result<Self> {
        let checkpoint = v
            .get("checkpoint")
            .ok_or_else(|| invalid("missing KV hint"))?;
        Ok(Self {
            checkpoint: Checkpoint::parse(checkpoint, text(checkpoint, "cache_domain")?)?,
            messages: usize::try_from(count(v, "messages")?)
                .map_err(|_| invalid("invalid KV boundary"))?,
        })
    }
}
pub(crate) fn hints_value(hints: &[Hint]) -> Value {
    Value::Array(hints.iter().map(Hint::value).collect())
}
/// Invalid optional metadata cannot make the authoritative transcript unusable.
/// Parse only a bounded number; stale hints are harmless and never sent.
pub(crate) fn parse_hints(value: &Value, messages: usize) -> Vec<Hint> {
    let Ok(raw) = json::stringify(value) else {
        return Vec::new();
    };
    if raw.len() > 4096 {
        return Vec::new();
    }
    let Ok(v) = json::view(&raw) else {
        return Vec::new();
    };
    let Some(items) = v.array() else {
        return Vec::new();
    };
    let mut hints = Vec::new();
    for item in items.take(MAX_HINTS + 1) {
        let Ok(hint) = Hint::parse(item) else {
            return Vec::new();
        };
        if hint.messages > messages
            || hint.messages == 0
            || hints
                .iter()
                .any(|h: &Hint| h.checkpoint.handle == hint.checkpoint.handle)
        {
            return Vec::new();
        }
        hints.push(hint);
    }
    if hints.len() > MAX_HINTS {
        Vec::new()
    } else {
        hints
    }
}

/// Whether a request reused a checkpoint handle, as the engine reported it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    Hit,
    Replayed,
    PromptMismatch,
    InvalidHandle,
    Unavailable,
    Unsupported,
    CacheDisabled,
    NotAttempted,
    NotRequested,
    Unknown,
}
impl Reuse {
    pub fn label(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Replayed => "replayed",
            Self::PromptMismatch => "prompt_mismatch",
            Self::InvalidHandle => "invalid_handle",
            Self::Unavailable => "unavailable",
            Self::Unsupported => "unsupported",
            Self::CacheDisabled => "cache_disabled",
            Self::NotAttempted => "not_attempted",
            Self::NotRequested => "not_requested",
            Self::Unknown => "unknown",
        }
    }
    fn parse(s: &str) -> Self {
        match s {
            "hit" => Self::Hit,
            "replayed" => Self::Replayed,
            "prompt_mismatch" => Self::PromptMismatch,
            "invalid_handle" => Self::InvalidHandle,
            "unavailable" => Self::Unavailable,
            "unsupported" => Self::Unsupported,
            "cache_disabled" => Self::CacheDisabled,
            "not_attempted" => Self::NotAttempted,
            "not_requested" => Self::NotRequested,
            _ => Self::Unknown,
        }
    }
    pub(crate) fn rejected(self) -> bool {
        matches!(
            self,
            Self::PromptMismatch | Self::InvalidHandle | Self::Unavailable | Self::Unsupported
        )
    }
}
/// Whether the engine kept a new checkpoint after a request, and if not, why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retention {
    Retained,
    Capacity,
    Unavailable,
    RandomnessUnavailable,
    Unsupported,
    CacheDisabled,
    PromptIncomplete,
    NotRequested,
    Unknown,
}
impl Retention {
    pub fn label(self) -> &'static str {
        match self {
            Self::Retained => "retained",
            Self::Capacity => "capacity",
            Self::Unavailable => "unavailable",
            Self::RandomnessUnavailable => "randomness_unavailable",
            Self::Unsupported => "unsupported",
            Self::CacheDisabled => "cache_disabled",
            Self::PromptIncomplete => "prompt_incomplete",
            Self::NotRequested => "not_requested",
            Self::Unknown => "unknown",
        }
    }
    fn parse(s: &str) -> Self {
        match s {
            "retained" => Self::Retained,
            "capacity" => Self::Capacity,
            "unavailable" => Self::Unavailable,
            "randomness_unavailable" => Self::RandomnessUnavailable,
            "unsupported" => Self::Unsupported,
            "cache_disabled" => Self::CacheDisabled,
            "prompt_incomplete" => Self::PromptIncomplete,
            "not_requested" => Self::NotRequested,
            _ => Self::Unknown,
        }
    }
}
/// The engine's checkpoint report for one response: reuse, reused tokens,
/// retention and any new checkpoint handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub cache_domain: String,
    pub reuse: Reuse,
    pub reused_tokens: u64,
    pub retention: Retention,
    pub checkpoint: Option<Checkpoint>,
}
impl Report {
    fn parse(v: json::View<'_>) -> io::Result<Self> {
        let cache_domain = text(v, "cache_domain")?;
        let checkpoint = v
            .get("checkpoint")
            .filter(|v| v.raw() != "null")
            .map(|v| Checkpoint::parse(v, cache_domain.clone()))
            .transpose()?;
        let result = Self {
            cache_domain,
            reuse: Reuse::parse(&text(v, "reuse")?),
            reused_tokens: count(v, "reused_tokens")?,
            retention: Retention::parse(&text(v, "retention")?),
            checkpoint,
        };
        if (result.retention == Retention::Retained) != result.checkpoint.is_some()
            || (result.reuse != Reuse::Hit && result.reused_tokens != 0)
        {
            return Err(invalid("inconsistent KV report"));
        }
        Ok(result)
    }
    pub fn describe(&self) -> String {
        let mut s = format!(
            "KV handle: {} · reused tokens {} · checkpoint {}",
            self.reuse.label(),
            self.reused_tokens,
            self.retention.label()
        );
        if let Some(c) = &self.checkpoint {
            s.push_str(&format!(" (input {} tokens, reusable {}, expires {} ms since epoch, {} bytes; scope {} bytes at issuance)",
                c.prompt_tokens, c.reusable_prefix_tokens, c.expires_at_ms, c.retained_bytes, c.scope_retained_bytes));
        }
        s
    }
}
#[derive(Default)]
pub(crate) struct Collector {
    report: Option<Report>,
    invalid: bool,
}
impl Collector {
    pub(crate) fn observe(&mut self, event: json::View<'_>) {
        let Some(usage) = event.get("usage") else {
            return;
        };
        let Some(kv) = usage
            .get("vade")
            .and_then(|v| v.get("kv"))
            .filter(|v| v.raw() != "null")
        else {
            return;
        };
        let parsed = Report::parse(kv).and_then(|r| {
            let p = usage
                .get("prompt_tokens")
                .map(|_| count(usage, "prompt_tokens"))
                .transpose()?;
            if p.is_some_and(|p| {
                r.reused_tokens > p || r.checkpoint.as_ref().is_some_and(|c| c.prompt_tokens != p)
            }) {
                return Err(invalid("inconsistent KV usage"));
            }
            if let Some(details) = usage
                .get("prompt_tokens_details")
                .filter(|v| v.get("cached_tokens").is_some())
            {
                if r.reused_tokens > count(details, "cached_tokens")? {
                    return Err(invalid("inconsistent KV cached usage"));
                }
            }
            Ok(r)
        });
        match parsed {
            Ok(r) if self.report.as_ref().is_none_or(|old| old == &r) => self.report = Some(r),
            _ => self.invalid = true,
        }
    }
    pub(crate) fn finish(self) -> (Option<Report>, bool) {
        (if self.invalid { None } else { self.report }, self.invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn checkpoint() -> Checkpoint {
        Checkpoint {
            handle: "a".repeat(64),
            cache_domain: "scope".into(),
            prompt_tokens: 100,
            reusable_prefix_tokens: 96,
            expires_at_ms: 2000,
            retained_bytes: 400,
            scope_retained_bytes: 500,
        }
    }
    fn caps() -> Capabilities {
        Capabilities {
            cache_domain: "scope".into(),
            max_handles_per_scope: 2,
            max_bytes_per_scope: 1000,
            ttl_ms: 1000,
        }
    }
    fn event(reuse: &str, checkpoint: &str) -> String {
        format!(
            r#"{{"usage":{{"prompt_tokens":100,"vade":{{"kv":{{"cache_domain":"scope","reuse":"{reuse}","reused_tokens":0,"retention":"retained","checkpoint":{checkpoint}}}}}}}}}"#
        )
    }
    #[test]
    fn metadata_requires_explicit_bounded_scoped_capabilities_and_local_leases_expire() {
        let meta = r#"{"kv_branch_handles":true,"kv_checkpoints":{"cache_domain":"scope","scope":"authorization","max_handles_per_scope":2,"max_bytes_per_scope":1000,"ttl_ms":1000,"release":"/v1/kv/release"}}"#;
        assert_eq!(
            Capabilities::parse(json::view(meta).unwrap()).unwrap(),
            Some(caps())
        );
        for bad in [
            meta.replace("/v1/kv/release", "http://elsewhere.invalid/release"),
            meta.replace("authorization", "shared"),
            meta.replace("\"ttl_ms\":1000", "\"ttl_ms\":0"),
            meta.replace(
                "\"max_handles_per_scope\":2",
                "\"max_handles_per_scope\":-1",
            ),
        ] {
            assert!(Capabilities::parse(json::view(&bad).unwrap()).is_err());
        }
        assert_eq!(
            Capabilities::parse(
                json::view(r#"{"kv_branch_handles":false,"kv_checkpoints":"ignored"}"#).unwrap()
            )
            .unwrap(),
            None
        );
        let c = checkpoint();
        assert!(c.eligible(&caps(), 1999));
        assert!(!c.eligible(&caps(), 2000));
        let mut other = caps();
        other.cache_domain = "another authorization namespace".into();
        assert!(!c.eligible(&other, 1));
        other = caps();
        other.max_bytes_per_scope = 499;
        assert!(!c.eligible(&other, 1));
    }
    #[test]
    fn conflicting_or_invalid_telemetry_is_unknown_and_never_leaks_handles() {
        let saved = json::stringify(&checkpoint().value()).unwrap();
        let raw = event("not_requested", &saved);
        let mut collector = Collector::default();
        collector.observe(json::view(&raw).unwrap());
        collector.observe(json::view(&raw).unwrap());
        let (report, invalid) = collector.finish();
        assert!(!invalid);
        assert!(!format!("{:?}", report).contains(&"a".repeat(64)));
        assert!(!report.unwrap().describe().contains(&"a".repeat(64)));
        for second in [
            event("unavailable", &saved),
            raw.replace("\"prompt_tokens\":100", "\"prompt_tokens\":99"),
            raw.replace("\"reused_tokens\":0", "\"reused_tokens\":-1"),
            raw.replace(
                "\"reusable_prefix_tokens\":96",
                "\"reusable_prefix_tokens\":101",
            ),
        ] {
            let mut collector = Collector::default();
            collector.observe(json::view(&raw).unwrap());
            collector.observe(json::view(&second).unwrap());
            assert_eq!(collector.finish(), (None, true));
        }
    }
    #[test]
    fn optional_hints_cannot_claim_unseen_history_or_unbounded_retention() {
        let hint = Hint {
            checkpoint: checkpoint(),
            messages: 2,
        };
        let raw = hints_value(std::slice::from_ref(&hint));
        assert_eq!(parse_hints(&raw, 2), vec![hint.clone()]);
        assert!(parse_hints(&raw, 1).is_empty());
        assert!(parse_hints(&hints_value(&[hint.clone(), hint]), 2).is_empty());
        assert!(parse_hints(&Value::String("invalid optional metadata".into()), 2).is_empty());
    }
}
