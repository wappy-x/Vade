//! Validated response usage, including engine prefill cache receipts.
use super::*;

/// A response may repeat its usage snapshot. Identical duplicates count once;
/// conflicting or malformed telemetry becomes unknown without inventing sums.
#[derive(Default)]
pub(crate) struct UsageCollector {
    usage: Option<Usage>,
    pub(super) prefill_elapsed: Option<std::time::Duration>,
    pub(super) server_decode_elapsed: Option<std::time::Duration>,
    pub(super) kv: crate::kv::Collector,
    invalid: bool,
}
fn may_have_usage(data: &str) -> bool {
    data.contains("\"usage\"") || data.contains('\\')
}
impl UsageCollector {
    /// Native assembly already validated this event. Most text deltas contain
    /// neither a usage key nor escapes; avoid another full validation pass.
    /// Any escape could encode a key, so it deliberately takes the general path.
    pub(crate) fn observe_data(&mut self, data: &str) -> io::Result<()> {
        if may_have_usage(data) {
            self.observe(json::view(data)?);
        }
        Ok(())
    }

    pub(crate) fn observe(&mut self, event: json::View<'_>) {
        if !may_have_usage(event.raw()) {
            return;
        }
        self.kv.observe(event);
        let Some(value) = event.get("usage").filter(|v| v.raw() != "null") else {
            return;
        };
        let parse = || -> io::Result<(Usage, Option<std::time::Duration>, Option<std::time::Duration>)> {
            if value.fields().is_none() {
                return Err(invalid("invalid usage"));
            }
            let optional =
                |v: Option<json::View<'_>>| v.filter(|v| v.raw() != "null").map(number).transpose();
            let details = value
                .get("prompt_tokens_details")
                .filter(|v| v.raw() != "null");
            if details.is_some_and(|v| v.fields().is_none()) {
                return Err(invalid("invalid usage details"));
            }
            let vade = event.get("vade").or_else(|| value.get("vade"));
            let restored = optional(vade.and_then(|v| v.get("prefill_cache_restored_tokens")))?;
            let cached = optional(details.and_then(|v| v.get("cached_tokens")))?;
            if cached.zip(restored).is_some_and(|(a, b)| a != b) { return Err(invalid("conflicting cache usage")); }
            let timing = |name| -> io::Result<Option<std::time::Duration>> {
                value.get("vade").and_then(|v| v.get(name)).or_else(|| vade.and_then(|v| v.get(name))).filter(|v| v.raw() != "null")
                    .map(|v| v.raw().parse::<f64>().ok().filter(|n| n.is_finite() && *n >= 0.0)
                        .and_then(|ms| std::time::Duration::try_from_secs_f64(ms / 1000.0).ok())
                        .ok_or_else(|| invalid("invalid response timing"))).transpose()
            };
            let usage = Usage {
                prompt_tokens: optional(value.get("prompt_tokens"))?,
                completion_tokens: optional(value.get("completion_tokens"))?,
                cached_tokens: cached.or(restored),
            };
            if usage
                .prompt_tokens
                .zip(usage.cached_tokens)
                .is_some_and(|(p, c)| c > p)
            {
                return Err(invalid("cached tokens exceed prompt tokens"));
            }
            Ok((usage, timing("prefill_ms")?, timing("decode_ms")?))
        };
        match parse() {
            Ok((usage, prefill, decode)) if self.usage.is_none() || (self.usage == Some(usage) && self.prefill_elapsed == prefill && self.server_decode_elapsed == decode) => {
                self.usage = Some(usage);
                self.prefill_elapsed = prefill;
                self.server_decode_elapsed = decode;
            }
            _ => self.invalid = true,
        }
    }
    pub(crate) fn finish(self) -> (Usage, bool) {
        (
            if self.invalid {
                Usage::default()
            } else {
                self.usage.unwrap_or_default()
            },
            self.invalid,
        )
    }
}

