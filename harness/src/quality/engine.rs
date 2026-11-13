//! The engine card (`quality-card --set engine-1`): five checks that the
//! served engine behaves, graded against a live server.
//!
//! 1. Long-context recall: a planted fact at 8K and 32K prompt tokens.
//! 2. JSON mode: 50 `response_format` json_schema requests, parsed and checked
//!    against their schema.
//! 3. Tool calls: 20 prompts that must call one named tool with exact
//!    arguments.
//! 4. Cache invariance: 20 prompts read cold (`"cache": false`, no lookup and
//!    no store), then primed and read warm; the token ids must be identical.
//! 5. Concurrency invariance: 8 simultaneous copies of a prompt against one
//!    alone, for 3 prompts; the token ids must be identical.
//!
//! Every request is greedy with thinking off. Exact means identical token ids
//! with complete token evidence (see `reply_evidence_failure`).
use super::engine_cases::*;
use super::*;
use std::sync::{Arc, Barrier};

/// The facts the server reports about itself in each reply's `vade` object.
#[derive(Clone, Debug, Default)]
pub struct ServerFacts {
    pub git_sha: Option<String>,
    pub dirty: Option<bool>,
    pub nvfp4: Option<bool>,
    pub mtp_depth: Option<u64>,
    pub world_size: Option<u64>,
    pub topology: Option<String>,
    pub draft: Option<String>,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_sha256: Option<String>,
    pub served_model: Option<String>,
}
impl ServerFacts {
    /// Read the facts from one reply. Absent fields stay None.
    pub fn from_reply(reply: &Reply) -> Self {
        let vade = reply.vade_receipt.as_deref().and_then(|text| json::parse(text).ok());
        let at = |path: &[&str]| {
            let mut value = vade.as_ref()?;
            for key in path {
                value = value.get(key)?;
            }
            Some(value.clone())
        };
        let text = |path: &[&str]| at(path).and_then(|v| v.as_str().map(str::to_owned));
        let flag = |path: &[&str]| match at(path) {
            Some(Value::Bool(flag)) => Some(flag),
            _ => None,
        };
        let count = |path: &[&str]| match at(path) {
            Some(Value::Number(text)) => text.parse().ok(),
            _ => None,
        };
        Self {
            git_sha: text(&["build", "git_sha"]),
            dirty: flag(&["build", "dirty"]),
            nvfp4: flag(&["numerics", "nvfp4"]),
            mtp_depth: count(&["mtp_depth"]),
            world_size: count(&["numerics", "world_size"]),
            topology: text(&["numerics", "topology"]),
            draft: text(&["draft"]),
            checkpoint_dir: text(&["checkpoint", "directory"]),
            checkpoint_sha256: text(&["checkpoint", "sha256"]),
            served_model: reply.served_model.clone(),
        }
    }
}

/// The calibration from two probe prompts: server tokens per filler sentence
/// and the tokens of the frame around the filler.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    pub probe_tokens: [u64; 2],
    pub tokens_per_sentence: f64,
    pub frame_tokens: f64,
}
impl Calibration {
    pub fn from_probes(probe_tokens: [u64; 2]) -> Result<Self, String> {
        let [low, high] = probe_tokens;
        let sentences = (PROBE_SENTENCES[1] - PROBE_SENTENCES[0]) as f64;
        if high <= low {
            return Err(format!("probe prompt tokens did not grow ({low} then {high})"));
        }
        let tokens_per_sentence = (high - low) as f64 / sentences;
        let frame_tokens = low as f64 - tokens_per_sentence * PROBE_SENTENCES[0] as f64;
        Ok(Self { probe_tokens, tokens_per_sentence, frame_tokens })
    }
    /// Filler sentences for a prompt of at least `target` tokens, with a
    /// 3 percent margin plus 8 sentences for sentence-length spread.
    pub fn sentences_for(&self, target: u64) -> usize {
        let filler = (target as f64 - self.frame_tokens).max(0.0);
        (filler * 1.03 / self.tokens_per_sentence).ceil() as usize + 8
    }
}

/// Answer tokens a needle request asks for (`max_tokens`). Its prompt must
/// leave this much room in the server context.
pub const NEEDLE_ANSWER_TOKENS: u64 = 32;
/// Admission checks one needle may use to find a filler size that fits.
pub const NEEDLE_SIZE_CHECKS: usize = 6;

/// The server's own context-fit answer for one exact request body
/// (`POST /v1/admission`): the prompt ids after the chat template, and the
/// limits it checked them against. It runs no model and reserves nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub prompt_tokens: u64,
    pub context_limit: u64,
    pub input_limit: u64,
    pub admitted: bool,
    pub reason: Option<String>,
}
impl Admission {
    pub fn parse(text: &str) -> Result<Self, String> {
        let value = json::parse(text).map_err(|error| format!("admission reply is not JSON: {error}"))?;
        let count = |key: &str| match value.get(key) {
            Some(Value::Number(text)) => text.parse::<u64>().map_err(|_| format!("admission {key} is not a count")),
            _ => Err(format!("admission reply has no {key}")),
        };
        let admitted = match value.get("admitted") {
            Some(Value::Bool(flag)) => *flag,
            _ => return Err("admission reply has no admitted flag".into()),
        };
        Ok(Self {
            prompt_tokens: count("prompt_tokens")?,
            context_limit: count("context_limit")?,
            input_limit: count("input_limit")?,
            admitted,
            reason: value.get("reason").and_then(Value::as_str).map(str::to_owned),
        })
    }
    /// The largest prompt the server admits with `answer` tokens of room.
    pub fn ceiling(&self, answer: u64) -> u64 {
        self.input_limit.min(self.context_limit.saturating_sub(answer))
    }
}

/// How one needle was sized against the server before it was sent.
#[derive(Clone, Debug, Default)]
pub struct NeedleSizing {
    /// The last admission answer, when the server has the check.
    pub admission: Option<Admission>,
    /// Set when the needle was not sent: why, with the server's limits.
    pub not_run: Option<String>,
    /// Set when the needle was sent without an admission check.
    pub note: Option<String>,
}

/// The decision for one needle: the filler size to send, or why not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeedleSize {
    Send { sentences: usize, admission: Option<Admission>, note: Option<String> },
    NotRun { sentences: usize, admission: Admission, reason: String },
}

/// Size one needle with the server's own token count before any chat
/// request. Start at the calibrated size; when the server would refuse it,
/// step the filler with the exact counts to a prompt from `target` up to the
/// largest one the server admits with answer room. A server whose limits
/// leave less than `target` gets NOT RUN with its numbers and the context to
/// start it with. The pass rule does not change: the sent prompt must still
/// hold `target` tokens. A server without the check is sent the calibrated
/// size unchecked, as before.
pub fn size_needle(target: u64, cal: &Calibration, mut admit: impl FnMut(usize) -> Result<Admission, String>) -> NeedleSize {
    let answer = NEEDLE_ANSWER_TOKENS;
    let mut sentences = cal.sentences_for(target);
    let mut last: Option<Admission> = None;
    for _ in 0..NEEDLE_SIZE_CHECKS {
        let admission = match (admit(sentences), last.take()) {
            (Ok(admission), _) => admission,
            (Err(error), None) => {
                let note = Some(format!("the server admission check failed ({error}); sent unchecked"));
                return NeedleSize::Send { sentences, admission: None, note };
            }
            (Err(error), Some(admission)) => {
                let reason = format!("the server admission check failed while sizing: {error}");
                return NeedleSize::NotRun { sentences, admission, reason };
            }
        };
        let (tokens, ceiling) = (admission.prompt_tokens, admission.ceiling(answer));
        if ceiling < target {
            // Answer room plus a 1/32 filler margin, rounded up to 8,192.
            let suggest = (target + answer + target / 32).div_ceil(8_192) * 8_192;
            let reason = format!(
                "the server admits at most {ceiling} prompt tokens (context {} less {answer} answer tokens, input limit {}); this needle needs at least {target}; start the server with a context of at least {} tokens (vade up --context {suggest})",
                admission.context_limit,
                admission.input_limit,
                target + answer
            );
            return NeedleSize::NotRun { sentences, admission, reason };
        }
        if (target..=ceiling).contains(&tokens) {
            if admission.admitted {
                return NeedleSize::Send { sentences, admission: Some(admission), note: None };
            }
            let reason = format!(
                "the server refused a {tokens}-token prompt inside its limits: {}",
                admission.reason.as_deref().unwrap_or("no reason given")
            );
            return NeedleSize::NotRun { sentences, admission, reason };
        }
        // Aim a little above the target, never past the ceiling.
        let aim = target + ((ceiling - target) / 2).min(target / 64);
        let step = ((aim.abs_diff(tokens) as f64) / cal.tokens_per_sentence).ceil().max(1.0) as usize;
        sentences = if tokens < target { sentences + step } else { sentences.saturating_sub(step) };
        last = Some(admission);
    }
    let admission = last.expect("at least one admission check ran");
    let reason = format!(
        "no filler size gave a prompt of {target} to {} tokens in {NEEDLE_SIZE_CHECKS} admission checks (last {})",
        admission.ceiling(answer),
        admission.prompt_tokens
    );
    NeedleSize::NotRun { sentences, admission, reason }
}

/// One needle at one depth.
#[derive(Clone, Debug)]
pub struct NeedleItem {
    pub needle: &'static Needle,
    pub sentences: usize,
    pub prompt_chars: usize,
    pub reply: Option<Reply>,
    pub error: Option<String>,
    pub sizing: NeedleSizing,
}
impl NeedleItem {
    pub fn failure_reason(&self) -> Option<String> {
        if let Some(reason) = &self.sizing.not_run {
            return Some(format!("NOT RUN: {reason}"));
        }
        if let Some(error) = &self.error {
            return Some(error.clone());
        }
        let Some(reply) = &self.reply else { return Some("request failed".into()) };
        let tokens = reply.prompt_tokens.unwrap_or(0);
        if tokens < self.needle.target_tokens {
            return Some(format!(
                "prompt held {tokens} tokens, under the {} target",
                self.needle.target_tokens
            ));
        }
        if !reply.text.contains(self.needle.answer) {
            return Some(format!("answer {} does not hold {}", shown_text(&reply.text), self.needle.answer));
        }
        None
    }
    pub fn passed(&self) -> bool {
        self.failure_reason().is_none()
    }
}

/// One JSON-mode request.
#[derive(Clone, Debug)]
pub struct JsonItem {
    pub case: &'static JsonSchemaCase,
    pub source: usize,
    pub reply: Option<Reply>,
    pub error: Option<String>,
}
impl JsonItem {
    pub fn id(&self) -> String {
        format!("{}-{}", self.case.name, self.source + 1)
    }
    /// True when the prompt names no key (see `SCHEMA_ONLY_SOURCE`).
    pub fn schema_only(&self) -> bool {
        self.source == SCHEMA_ONLY_SOURCE
    }
    pub fn failure_reason(&self) -> Option<String> {
        if let Some(error) = &self.error {
            return Some(error.clone());
        }
        let Some(reply) = &self.reply else { return Some("request failed".into()) };
        json_answer_failure(self.case.schema, &reply.text)
    }
    pub fn passed(&self) -> bool {
        self.failure_reason().is_none()
    }
}

/// Why a JSON-mode answer fails: it must parse as one JSON value and the
/// schema must accept it.
pub fn json_answer_failure(schema: &str, answer: &str) -> Option<String> {
    let schema = json::parse(schema).expect("fixed schema parses");
    let Ok(value) = json::parse(answer.trim()) else {
        return Some(format!("not one JSON value: {}", shown_text(answer)));
    };
    schema_failure(&schema, &value, "$").err()
}

/// Check `value` against the schema subset the card uses: type (object,
/// array, string, integer, number, boolean), properties, required,
/// additionalProperties false, items, enum, minimum and maximum. An object
/// that repeats a key fails. An integer must be spelled as one.
pub fn schema_failure(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    if let Some(choices) = schema.get("enum").and_then(Value::as_array) {
        if !choices.iter().any(|choice| values_equal(choice, value)) {
            return Err(format!("{path} {} is not in the enum", short(value)));
        }
    }
    let kind = schema.get("type").and_then(Value::as_str).unwrap_or("");
    let number = |text: &str| -> Result<(), String> {
        let got: f64 = text.parse().map_err(|_| format!("{path} {text} is not a number"))?;
        let bound = |key| schema.get(key).and_then(|v| match v {
            Value::Number(t) => t.parse::<f64>().ok(),
            _ => None,
        });
        if bound("minimum").is_some_and(|low| got < low) || bound("maximum").is_some_and(|high| got > high) {
            return Err(format!("{path} {text} is out of range"));
        }
        Ok(())
    };
    match (kind, value) {
        ("object", Value::Object(fields)) => {
            let properties = match schema.get("properties") {
                Some(Value::Object(properties)) => properties.as_slice(),
                _ => &[],
            };
            for (at, (key, _)) in fields.iter().enumerate() {
                if fields[..at].iter().any(|(other, _)| other == key) {
                    return Err(format!("{path} repeats key {key}"));
                }
            }
            for (key, field) in fields {
                match properties.iter().find(|(name, _)| name == key) {
                    Some((_, sub)) => schema_failure(sub, field, &format!("{path}.{key}"))?,
                    None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                        return Err(format!("{path} has extra key {key}"));
                    }
                    None => {}
                }
            }
            for key in schema.get("required").and_then(Value::as_array).unwrap_or(&[]) {
                let key = key.as_str().unwrap_or("");
                if !fields.iter().any(|(name, _)| name == key) {
                    return Err(format!("{path} misses key {key}"));
                }
            }
            Ok(())
        }
        ("array", Value::Array(items)) => {
            if let Some(sub) = schema.get("items") {
                for (index, item) in items.iter().enumerate() {
                    schema_failure(sub, item, &format!("{path}[{index}]"))?;
                }
            }
            Ok(())
        }
        ("string", Value::String(_)) | ("boolean", Value::Bool(_)) => Ok(()),
        ("integer", Value::Number(text)) => {
            if text.parse::<i64>().is_err() {
                return Err(format!("{path} {text} is not an integer"));
            }
            number(text)
        }
        ("number", Value::Number(text)) => number(text),
        ("", _) => Ok(()),
        _ => Err(format!("{path} {} is not of type {kind}", short(value))),
    }
}

/// One tool-call request.
#[derive(Clone, Debug)]
pub struct ToolItem {
    pub case: &'static ToolCase,
    pub reply: Option<Reply>,
    pub error: Option<String>,
}
impl ToolItem {
    pub fn failure_reason(&self) -> Option<String> {
        if let Some(error) = &self.error {
            return Some(error.clone());
        }
        let Some(reply) = &self.reply else { return Some("request failed".into()) };
        tool_call_failure(self.case, reply)
    }
    pub fn passed(&self) -> bool {
        self.failure_reason().is_none()
    }
}

/// Why a reply is not exactly the one expected call: one call, the exact
/// tool name, and arguments that parse to exactly the expected object
/// (the same keys and values; numbers compare by value, key order is free).
pub fn tool_call_failure(case: &ToolCase, reply: &Reply) -> Option<String> {
    match reply.tool_calls.as_slice() {
        [] => Some(format!("no tool call; content {}", shown_text(&reply.text))),
        [call] => {
            if call.name != case.name {
                return Some(format!("called {} (want {})", call.name, case.name));
            }
            let want = json::parse(case.arguments).expect("fixed arguments parse");
            match json::parse(&call.arguments) {
                Ok(got) if values_equal(&got, &want) => None,
                Ok(got) => Some(format!("arguments {} (want {})", short(&got), case.arguments)),
                Err(_) => Some(format!("arguments do not parse: {}", shown_text(&call.arguments))),
            }
        }
        calls => Some(format!("{} tool calls (want 1)", calls.len())),
    }
}

/// Why two replies to the same request are not exactly the same answer.
/// Both must carry complete token evidence and agree on the finish reason
/// and completion count; only then are ids and text compared. The draft
/// modes may differ: with the draft field omitted the server drafts strict
/// for one chat and plain above its chat limit, and drafting ON and OFF must
/// give the same ids. The receipt keeps the mode of every reply.
pub fn pair_failure(base: &Reply, other: &Reply) -> Option<String> {
    if let Some(reason) = reply_evidence_failure(base) {
        return Some(format!("first run: {reason}"));
    }
    if let Some(reason) = reply_evidence_failure(other) {
        return Some(format!("second run: {reason}"));
    }
    if base.finish_reason != other.finish_reason || base.completion_tokens != other.completion_tokens {
        return Some(format!(
            "finish or count differs: {} {} then {} {}",
            shown_mode(&base.finish_reason),
            base.completion_tokens.unwrap_or(0),
            shown_mode(&other.finish_reason),
            other.completion_tokens.unwrap_or(0)
        ));
    }
    if let Some(at) = first_id_diff(&base.token_ids, &other.token_ids) {
        return Some(format!(
            "ids differ at token {at}: {} then {}",
            base.token_ids.get(at).map_or_else(|| "end".into(), u32::to_string),
            other.token_ids.get(at).map_or_else(|| "end".into(), u32::to_string)
        ));
    }
    if base.text != other.text {
        return Some("same ids but the texts differ".into());
    }
    None
}

/// The first position where two id lists differ, counting a length change.
pub fn first_id_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// One cache prompt: cold, then primed, then warm.
#[derive(Clone, Debug)]
pub struct CacheItem {
    pub index: usize,
    pub cold: Option<Reply>,
    pub prime: Option<Reply>,
    pub warm: Option<Reply>,
    pub error: Option<String>,
}
impl CacheItem {
    /// A pass needs a cold run that read no cached tokens, a warm run that
    /// reused some, and identical ids with complete evidence on both.
    pub fn failure_reason(&self) -> Option<String> {
        if let Some(error) = &self.error {
            return Some(error.clone());
        }
        let (Some(cold), Some(warm)) = (&self.cold, &self.warm) else {
            return Some("request failed".into());
        };
        match cold.cached_tokens {
            Some(0) => {}
            Some(n) => return Some(format!("cold run read {n} cached tokens")),
            None => return Some("cold run reported no cached token count".into()),
        }
        match warm.cached_tokens {
            Some(n) if n > 0 => {}
            _ => {
                return Some(format!(
                    "warm run reused no cached tokens (prompt {} tokens)",
                    warm.prompt_tokens.unwrap_or(0)
                ));
            }
        }
        pair_failure(cold, warm)
    }
    pub fn passed(&self) -> bool {
        self.failure_reason().is_none()
    }
}

/// One copy of a concurrent request, with its send and finish times in
/// seconds from the start of the group.
#[derive(Clone, Debug)]
pub struct CopyRun {
    pub reply: Result<Reply, String>,
    pub start_s: f64,
    pub end_s: f64,
}
impl CopyRun {
    /// When the first token of this copy arrived, from the start of the group.
    pub fn first_token_s(&self) -> Option<f64> {
        Some(self.start_s + self.reply.as_ref().ok()?.ttft_s?)
    }
}

/// One concurrency prompt: one run alone, then the copies at once.
#[derive(Clone, Debug)]
pub struct ConcurrencyItem {
    pub id: &'static str,
    pub alone: Result<Reply, String>,
    pub copies: Vec<CopyRun>,
}
impl ConcurrencyItem {
    /// True when every copy was streaming at one shared moment: the last
    /// first token arrived before the first copy finished. Open sockets are
    /// not enough; a server that queues requests accepts them all but
    /// streams one at a time. A failed copy fails on its own and is left out
    /// of the window; a reply without a first token fails the window.
    pub fn overlapped(&self) -> bool {
        let answered: Vec<&CopyRun> = self.copies.iter().filter(|c| c.reply.is_ok()).collect();
        let firsts: Option<Vec<f64>> = answered.iter().map(|c| c.first_token_s()).collect();
        let Some(firsts) = firsts else { return false };
        let latest_first = firsts.into_iter().fold(f64::MIN, f64::max);
        let earliest_end = answered.iter().map(|c| c.end_s).fold(f64::MAX, f64::min);
        answered.len() > 1 && latest_first < earliest_end
    }
    /// Why copy `index` does not match the run alone, or None.
    pub fn copy_failure(&self, index: usize) -> Option<String> {
        let alone = match &self.alone {
            Ok(reply) => reply,
            Err(error) => return Some(format!("run alone failed: {error}")),
        };
        match self.copies.get(index).map(|c| &c.reply) {
            Some(Ok(reply)) => pair_failure(alone, reply),
            Some(Err(error)) => Some(error.clone()),
            None => Some("copy missing".into()),
        }
    }
    /// Copies identical to the run alone. Copies that did not overlap prove
    /// nothing about concurrency, so then none count.
    pub fn identical(&self) -> usize {
        if !self.overlapped() {
            return 0;
        }
        (0..CONCURRENT_COPIES).filter(|i| self.copy_failure(*i).is_none()).count()
    }
}

/// The card grade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    APlus,
    A,
    B,
    C,
}
impl Grade {
    pub fn label(self) -> &'static str {
        match self {
            Self::APlus => "A+",
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
        }
    }
}

/// Grade rules. A missed or not-run needle, or any cache or concurrency item that is
/// not identical, is a hard miss: grade C. A soft miss is exactly one item
/// short in JSON mode or in tool calls; two or more short in one check is C.
/// A+ has no miss, A one soft miss, B two soft misses.
pub fn grade(needles: (usize, usize), json: (usize, usize), tools: (usize, usize), cache: (usize, usize), concurrency: (usize, usize)) -> Grade {
    let full = |(passed, total): (usize, usize)| total > 0 && passed == total;
    if !full(needles) || !full(cache) || !full(concurrency) {
        return Grade::C;
    }
    let short = |(passed, total): (usize, usize)| total.saturating_sub(passed);
    let (json_short, tool_short) = (short(json), short(tools));
    if json.1 == 0 || tools.1 == 0 || json_short > 1 || tool_short > 1 {
        return Grade::C;
    }
    match json_short + tool_short {
        0 => Grade::APlus,
        1 => Grade::A,
        _ => Grade::B,
    }
}

/// A finished engine card run.
#[derive(Clone, Debug)]
pub struct EngineReport {
    pub config: Config,
    pub server: ServerFacts,
    pub calibration: Result<Calibration, String>,
    pub needles: Vec<NeedleItem>,
    pub json: Vec<JsonItem>,
    pub tools: Vec<ToolItem>,
    pub cache: Vec<CacheItem>,
    pub concurrency: Vec<ConcurrencyItem>,
    pub elapsed_s: f64,
}
impl EngineReport {
    pub fn needle_score(&self) -> (usize, usize) {
        (self.needles.iter().filter(|i| i.passed()).count(), NEEDLES.len())
    }
    pub fn json_score(&self) -> (usize, usize) {
        (self.json.iter().filter(|i| i.passed()).count(), JSON_CASES.len() * 5)
    }
    pub fn tool_score(&self) -> (usize, usize) {
        (self.tools.iter().filter(|i| i.passed()).count(), TOOL_CASES.len())
    }
    pub fn cache_score(&self) -> (usize, usize) {
        (self.cache.iter().filter(|i| i.passed()).count(), CACHE_PROMPTS)
    }
    pub fn concurrency_score(&self) -> (usize, usize) {
        (
            self.concurrency.iter().map(ConcurrencyItem::identical).sum(),
            CONCURRENCY_PROMPTS.len() * CONCURRENT_COPIES,
        )
    }
    pub fn grade(&self) -> Grade {
        grade(
            self.needle_score(),
            self.json_score(),
            self.tool_score(),
            self.cache_score(),
            self.concurrency_score(),
        )
    }
}

fn shown_text(text: &str) -> String {
    let flat: String = text.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let flat = flat.trim();
    if flat.chars().count() > 60 {
        format!("{:?}...", flat.chars().take(57).collect::<String>())
    } else {
        format!("{flat:?}")
    }
}

/// One engine-card request body. Greedy, thinking off, streamed with usage.
/// `cache` None omits the field and the server default applies, which for
/// the served FlashNext prefix cache is uncached; Some(false) runs the
/// request fresh and keeps nothing; Some(true) looks up and stores.
pub fn engine_body(
    config: &Config,
    system: Option<&str>,
    user: &str,
    max_tokens: u64,
    cache: Option<bool>,
    extra: Vec<(&'static str, Value)>,
) -> io::Result<String> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(obj([("role", s("system")), ("content", s(system))]));
    }
    messages.push(obj([("role", s("user")), ("content", s(user))]));
    let mut fields = vec![
        ("model", s(&config.model)),
        ("messages", Value::Array(messages)),
        ("temperature", n(0)),
        ("top_p", n(1)),
        ("top_k", n(0)),
        ("seed", n(config.seed)),
        ("max_tokens", n(max_tokens)),
        // GLM cannot turn thinking off and refuses the field: ask low effort there (see quality.rs).
        if crate::contract::thinking_off_supported(&config.model) {
            ("chat_template_kwargs", obj([("enable_thinking", Value::Bool(false))]))
        } else {
            ("reasoning_effort", s("low"))
        },
        ("include_usage_timing", Value::Bool(true)),
        ("stream", Value::Bool(true)),
        ("stream_options", obj([("include_usage", Value::Bool(true))])),
    ];
    if let Some(cache) = cache {
        fields.push(("cache", Value::Bool(cache)));
    }
    fields.extend(extra);
    json::stringify(&Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect()))
}

/// The `response_format` field for one JSON case.
pub fn response_format(case: &JsonSchemaCase) -> Value {
    obj([
        ("type", s("json_schema")),
        (
            "json_schema",
            obj([
                ("name", s(case.name)),
                ("strict", Value::Bool(true)),
                ("schema", json::parse(case.schema).expect("fixed schema parses")),
            ]),
        ),
    ])
}

/// The `tools` array offered on every tool request.
pub fn tools_value() -> Value {
    Value::Array(
        TOOLS
            .iter()
            .map(|tool| {
                obj([
                    ("type", s("function")),
                    (
                        "function",
                        obj([
                            ("name", s(tool.name)),
                            ("description", s(tool.description)),
                            ("parameters", json::parse(tool.parameters).expect("fixed tool parses")),
                        ]),
                    ),
                ])
            })
            .collect(),
    )
}

const NEEDLE_SYSTEM: &str = "You are a careful reader. Answer from the document only.";
const PLAIN_SYSTEM: &str = "Answer clearly and directly.";

struct Client {
    endpoint: Endpoint,
    auth: Option<String>,
    deadline: Duration,
}
impl Client {
    fn send(&self, body: io::Result<String>) -> Result<Reply, String> {
        body.and_then(|body| send(&self.endpoint, &body, self.auth.as_deref(), self.deadline))
            .map_err(|error| error.to_string())
    }
    /// Ask the server's context-fit check (`/v1/admission`, next to the chat
    /// route) about one exact chat body. The server renders it as inference
    /// would, counts its ids and checks them against its own limits.
    fn admission(&self, chat_body: &str) -> Result<Admission, String> {
        const REPLY_LIMIT: usize = 64 * 1024;
        let base = self.endpoint.path.strip_suffix("/chat/completions").unwrap_or("/v1");
        let url = format!("http://{}:{}{base}/admission", self.endpoint.host, self.endpoint.port);
        let endpoint = crate::transport::Endpoint::parse(&url).map_err(|error| error.to_string())?;
        let body = format!(
            "{{\"endpoint\":\"/v1/chat/completions\",\"output_tokens\":{NEEDLE_ANSWER_TOKENS},\"request\":{chat_body}}}"
        );
        let pool = crate::transport::ConnectionPool::default();
        let deadline = Some(Instant::now() + self.deadline);
        let mut response =
            crate::transport::request_method_pooled_deadline("POST", &endpoint, body.as_bytes(), None, None, &pool, deadline)
                .map_err(|error| error.to_string())?;
        if response.status != 200 {
            return Err(format!("HTTP status {}", response.status));
        }
        let mut bytes = Vec::new();
        response.body.by_ref().take(REPLY_LIMIT as u64 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        if bytes.len() > REPLY_LIMIT {
            return Err("admission reply exceeds its size limit".into());
        }
        Admission::parse(&String::from_utf8(bytes).map_err(|_| "admission reply is not UTF-8".to_owned())?)
    }
}

fn note(text: std::fmt::Arguments<'_>) {
    crate::diagnostics::stderr_line(format_args!("quality-card engine-1: {text}"));
}

/// Run the five checks in order: needles, JSON mode, tool calls, cache,
/// concurrency. A failed request fails its item and the run goes on.
pub fn run_engine(config: Config) -> io::Result<EngineReport> {
    config.validate().map_err(io::Error::other)?;
    let client = Arc::new(Client {
        endpoint: parse_endpoint(&config.endpoint)?,
        auth: configured_auth_header()?,
        deadline: Duration::from_secs(config.request_seconds),
    });
    // Every fixed body is built before the first request, so a body error
    // cannot end the run after requests have been made.
    let concurrency_bodies = CONCURRENCY_PROMPTS
        .iter()
        .map(|(_, prompt)| engine_body(&config, Some(PLAIN_SYSTEM), prompt, 160, Some(false), Vec::new()))
        .collect::<io::Result<Vec<String>>>()?;
    let started = Instant::now();
    let mut facts: Option<ServerFacts> = None;
    let mut learn = |reply: &Reply| {
        if facts.is_none() && reply.vade_receipt.is_some() {
            facts = Some(ServerFacts::from_reply(reply));
        }
    };

    note(format_args!("calibrating filler tokens with {} probes", PROBE_SENTENCES.len()));
    let mut probe_tokens = [0u64; 2];
    let mut calibration = Ok(());
    for (slot, sentences) in PROBE_SENTENCES.iter().enumerate() {
        let prompt = needle_prompt(&NEEDLES[0], *sentences, false);
        match client.send(engine_body(&config, Some(NEEDLE_SYSTEM), &prompt, 1, Some(false), Vec::new())) {
            Ok(reply) => {
                learn(&reply);
                match reply.prompt_tokens {
                    Some(tokens) => probe_tokens[slot] = tokens,
                    None => calibration = Err("probe reply had no prompt token count".to_owned()),
                }
            }
            Err(error) => calibration = Err(format!("probe request failed: {error}")),
        }
    }
    let calibration = calibration.and_then(|()| Calibration::from_probes(probe_tokens));

    let mut needles = Vec::new();
    for needle in &NEEDLES {
        note(format_args!("needle at {}", needle.label));
        let item = match &calibration {
            Ok(cal) => {
                let body = |sentences: usize| {
                    let prompt = needle_prompt(needle, sentences, true);
                    let body = engine_body(&config, Some(NEEDLE_SYSTEM), &prompt, NEEDLE_ANSWER_TOKENS, Some(false), Vec::new());
                    (prompt.len(), body)
                };
                let size = size_needle(needle.target_tokens, cal, |sentences| match body(sentences).1 {
                    Ok(body) => client.admission(&body),
                    Err(error) => Err(error.to_string()),
                });
                match size {
                    NeedleSize::Send { sentences, admission, note } => {
                        let (prompt_chars, body) = body(sentences);
                        let reply = client.send(body);
                        if let Ok(reply) = &reply {
                            learn(reply);
                        }
                        let error = reply.as_ref().err().cloned();
                        let sizing = NeedleSizing { admission, not_run: None, note };
                        NeedleItem { needle, sentences, prompt_chars, reply: reply.ok(), error, sizing }
                    }
                    NeedleSize::NotRun { sentences, admission, reason } => {
                        note(format_args!("needle at {} NOT RUN: {reason}", needle.label));
                        let sizing = NeedleSizing { admission: Some(admission), not_run: Some(reason), note: None };
                        NeedleItem { needle, sentences, prompt_chars: body(sentences).0, reply: None, error: None, sizing }
                    }
                }
            }
            Err(error) => NeedleItem {
                needle,
                sentences: 0,
                prompt_chars: 0,
                reply: None,
                error: Some(format!("not run: calibration failed: {error}")),
                sizing: NeedleSizing::default(),
            },
        };
        needles.push(item);
    }

    note(format_args!("JSON mode, {} requests", JSON_CASES.len() * 5));
    let mut json_items = Vec::new();
    for case in &JSON_CASES {
        for source in 0..case.sources.len() {
            let body = engine_body(
                &config,
                None,
                &json_prompt(case, source),
                256,
                Some(false),
                vec![("response_format", response_format(case))],
            );
            let reply = client.send(body);
            if let Ok(reply) = &reply {
                learn(reply);
            }
            let error = reply.as_ref().err().cloned();
            json_items.push(JsonItem { case, source, reply: reply.ok(), error });
        }
    }

    note(format_args!("tool calls, {} requests", TOOL_CASES.len()));
    let mut tool_items = Vec::new();
    for case in &TOOL_CASES {
        let body = engine_body(
            &config,
            Some(TOOL_SYSTEM),
            case.prompt,
            256,
            Some(false),
            vec![
                ("tools", tools_value()),
                ("tool_choice", s("auto")),
                ("parallel_tool_calls", Value::Bool(false)),
            ],
        );
        let reply = client.send(body);
        if let Ok(reply) = &reply {
            learn(reply);
        }
        let error = reply.as_ref().err().cloned();
        tool_items.push(ToolItem { case, reply: reply.ok(), error });
    }

    note(format_args!("cache invariance, {CACHE_PROMPTS} prompts cold, then primed and warm"));
    let cache_request = |index: usize, cache: Option<bool>| {
        client.send(engine_body(&config, Some(PLAIN_SYSTEM), &cache_prompt(index), 64, cache, Vec::new()))
    };
    let mut cache = Vec::new();
    for index in 0..CACHE_PROMPTS {
        let cold = cache_request(index, Some(false));
        if let Ok(reply) = &cold {
            learn(reply);
        }
        let error = cold.as_ref().err().map(|e| format!("cold run failed: {e}"));
        cache.push(CacheItem { index, cold: cold.ok(), prime: None, warm: None, error });
    }
    for item in &mut cache {
        let prime = cache_request(item.index, Some(true));
        let warm = cache_request(item.index, Some(true));
        if item.error.is_none() {
            item.error = match (&prime, &warm) {
                (Err(e), _) => Some(format!("prime run failed: {e}")),
                (_, Err(e)) => Some(format!("warm run failed: {e}")),
                _ => None,
            };
        }
        item.prime = prime.ok();
        item.warm = warm.ok();
    }

    let mut concurrency = Vec::new();
    for ((id, _), body) in CONCURRENCY_PROMPTS.iter().zip(concurrency_bodies) {
        note(format_args!("concurrency {id}: one alone, then {CONCURRENT_COPIES} at once"));
        let alone = client.send(Ok(body.clone()));
        if let Ok(reply) = &alone {
            learn(reply);
        }
        let barrier = Arc::new(Barrier::new(CONCURRENT_COPIES));
        let group = Instant::now();
        let handles: Vec<_> = (0..CONCURRENT_COPIES)
            .map(|_| {
                let (client, body, barrier) = (client.clone(), body.clone(), barrier.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    let start_s = group.elapsed().as_secs_f64();
                    let reply = client.send(Ok(body));
                    CopyRun { reply, start_s, end_s: group.elapsed().as_secs_f64() }
                })
            })
            .collect();
        let copies = handles
            .into_iter()
            .map(|handle| {
                handle.join().unwrap_or(CopyRun { reply: Err("copy thread panicked".into()), start_s: 0.0, end_s: 0.0 })
            })
            .collect();
        concurrency.push(ConcurrencyItem { id, alone, copies });
    }

    Ok(EngineReport {
        config,
        server: facts.unwrap_or_default(),
        calibration,
        needles,
        json: json_items,
        tools: tool_items,
        cache,
        concurrency,
        elapsed_s: started.elapsed().as_secs_f64(),
    })
}
