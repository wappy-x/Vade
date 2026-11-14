//! Parse the recognized engine capability contract and bounded identity metadata.
use super::{Capabilities, Identity, MAX_METADATA, VERSION, count, equals, number};
use crate::{json, transport::invalid};
use std::io;

fn string(value: json::View<'_>, name: &str) -> io::Result<String> {
    let text = value
        .get(name)
        .and_then(json::View::as_str)
        .ok_or_else(|| invalid("contract string missing or invalid"))?;
    if text.is_empty() || text.len() > 512 || text.chars().any(char::is_control) {
        return Err(invalid("contract string exceeds bounds"));
    }
    Ok(text.into_owned())
}
fn yes(value: json::View<'_>, name: &str) -> bool {
    value.get(name).is_some_and(|v| v.raw() == "true")
}
impl Capabilities {
    /// `None` is a foreign/unknown-version endpoint. A malformed recognized
    /// version is an error, so its promised admission cannot be silently skipped.
    pub fn parse(bytes: &str) -> io::Result<Option<Self>> {
        if bytes.len() > MAX_METADATA {
            return Err(invalid("capabilities exceed limit"));
        }
        let value = json::view(bytes)?;
        if !equals(value, "object", "engine.capabilities")
            || !equals(value, "contract_version", VERSION)
        {
            return Ok(None);
        }
        let identity = match value.get("identity") {
            None => None,
            Some(v) if v.raw() == "null" => None,
            Some(v) => Some(Identity {
                weights_fingerprint: string(
                    v.get("weights")
                        .ok_or_else(|| invalid("missing weights identity"))?,
                    "fingerprint",
                )?,
                weights_fingerprint_kind: v
                    .get("weights")
                    .filter(|w| w.get("fingerprint_kind").is_some())
                    .map(|w| string(w, "fingerprint_kind"))
                    .transpose()?,
                tokenizer_fingerprint: string(v, "tokenizer_fingerprint")?,
                template_fingerprint: string(v, "template_fingerprint")?,
            }),
        };
        let optional_string = |name| match value.get(name) {
            None => Ok(None),
            Some(v) if v.raw() == "null" => Ok(None),
            Some(_) => string(value, name).map(Some),
        };
        let result = Self {
            build_hash: optional_string("build_hash")?,
            kv_checkpoints: crate::kv::Capabilities::parse(value)?,
            model: string(value, "model")?,
            identity,
            context_limit: count(value, "context_limit")?,
            input_limit: count(value, "input_limit")?,
            output_limit: count(value, "output_limit")?,
            tools: yes(value, "tools"),
            thinking: yes(value, "thinking"),
            default_thinking: yes(value, "default_thinking"),
            cache_domain: optional_string("cache_domain")?,
            cache_scope: optional_string("cache_scope")?,
            prefix_cache: yes(value, "prefix_cache"),
            cache_granularity_tokens: value
                .get("cache_granularity_tokens")
                .filter(|v| v.raw() != "null")
                .map(number)
                .transpose()?,
            prompt_cache_key: equals(value, "prompt_cache_key", "scheduling_hint"),
            exact_admission: equals(value, "admission", "/v1/admission")
                && equals(value, "admission_kind", "context_fit"),
            chat_completions: value
                .get("endpoints")
                .and_then(json::View::array)
                .is_some_and(|mut a| {
                    a.any(|v| v.as_str().is_some_and(|s| s == "/v1/chat/completions"))
                }),
        };
        if result.context_limit == 0 || result.input_limit == 0 || result.output_limit == 0 {
            return Err(invalid("contract limits must be positive"));
        }
        if result.cache_granularity_tokens == Some(0) {
            return Err(invalid("cache granularity must be positive"));
        }
        Ok(Some(result))
    }
}

impl super::Context {
    /// Flags bound the server ceiling; otherwise capabilities select it outright.
    pub fn token_limit(&self, local: Option<u64>) -> u64 {
        let remote = self.capabilities.as_ref().map(|caps| caps.context_limit);
        match (local, remote) {
            (Some(local), Some(remote)) => local.min(remote),
            (Some(limit), None) | (None, Some(limit)) => limit,
            (None, None) => crate::context::DEFAULT_CONTEXT_TOKENS as u64,
        }
    }
}
