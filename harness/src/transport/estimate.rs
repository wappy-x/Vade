//! The engine's warm-cache estimate, read from a stream response head before
//! the first token. It is a display hint and never changes a request.
use super::Head;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillEstimate {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    /// Ms from engine admission to the first token, with the cache.
    pub first_token_ms: u64,
    /// Ms for the same prompt with nothing cached.
    pub cold_ms: u64,
}

pub(super) type EstimateObserver = Arc<dyn Fn(PrefillEstimate) + Send + Sync>;

impl PrefillEstimate {
    /// All four `X-Vade-*` estimate headers, or none.
    pub(crate) fn from_head(head: &Head) -> Option<Self> {
        let number = |name| head.get(name)?.trim().parse::<u64>().ok();
        let estimate = Self {
            prompt_tokens: number("x-vade-prompt-tokens")?,
            cached_tokens: number("x-vade-cached-tokens")?,
            first_token_ms: number("x-vade-estimated-first-token-ms")?,
            cold_ms: number("x-vade-estimated-cold-ms")?,
        };
        (estimate.cached_tokens <= estimate.prompt_tokens).then_some(estimate)
    }
    /// One short line, for example
    /// `cached 7,812 of 8,000 tokens; about 0.4 s to first token (3.7 s without the cache)`.
    pub fn line(&self) -> String {
        let first = if self.first_token_ms < 50 { "under 0.1 s".into() } else { format!("about {}", seconds(self.first_token_ms)) };
        let mut line = format!("cached {} of {} tokens; {first} to first token",
            grouped(self.cached_tokens), grouped(self.prompt_tokens));
        if self.cached_tokens > 0 {
            line.push_str(&format!(" ({} without the cache)", seconds(self.cold_ms)));
        }
        line
    }
}

fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}

fn seconds(ms: u64) -> String { format!("{:.1} s", ms as f64 / 1000.0) }

#[cfg(test)]
mod tests {
    use super::*;

    fn head(fields: &[(&str, &str)]) -> Head {
        Head { first: "HTTP/1.1 200 OK".into(), fields: fields.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() }
    }
    const FIELDS: [(&str, &str); 4] = [("x-vade-prompt-tokens", "8000"), ("x-vade-cached-tokens", "7812"),
        ("x-vade-estimated-first-token-ms", "812"), ("x-vade-estimated-cold-ms", "3770")];

    #[test]
    fn reads_all_four_headers_or_none() {
        let estimate = PrefillEstimate::from_head(&head(&FIELDS)).unwrap();
        assert_eq!(estimate, PrefillEstimate { prompt_tokens: 8000, cached_tokens: 7812, first_token_ms: 812, cold_ms: 3770 });
        assert_eq!(PrefillEstimate::from_head(&head(&FIELDS[..3])), None);
        assert_eq!(PrefillEstimate::from_head(&head(&[FIELDS[0], ("x-vade-cached-tokens", "9000"), FIELDS[2], FIELDS[3]])), None);
        assert_eq!(PrefillEstimate::from_head(&head(&[FIELDS[0], FIELDS[1], ("x-vade-estimated-first-token-ms", "0.8"), FIELDS[3]])), None);
        assert_eq!(PrefillEstimate::from_head(&head(&[])), None);
    }

    #[test]
    fn formats_one_line() {
        let estimate = PrefillEstimate::from_head(&head(&FIELDS)).unwrap();
        assert_eq!(estimate.line(), "cached 7,812 of 8,000 tokens; about 0.8 s to first token (3.8 s without the cache)");
        let cold = PrefillEstimate { prompt_tokens: 129_220, cached_tokens: 0, first_token_ms: 66_811, cold_ms: 66_811 };
        assert_eq!(cold.line(), "cached 0 of 129,220 tokens; about 66.8 s to first token");
        let replay = PrefillEstimate { prompt_tokens: 64, cached_tokens: 64, first_token_ms: 2, cold_ms: 360 };
        assert_eq!(replay.line(), "cached 64 of 64 tokens; under 0.1 s to first token (0.4 s without the cache)");
        assert_eq!((grouped(0), grouped(999), grouped(1000), grouped(1_234_567)), ("0".into(), "999".into(), "1,000".into(), "1,234,567".into()));
    }
}
