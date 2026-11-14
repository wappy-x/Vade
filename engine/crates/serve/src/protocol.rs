//! Chat-completions requests and JSON encoding. Unknown client fields are ignored.
use crate::{Limits, MODEL_ID};
use engine_model::json::{self, Value};

#[derive(Clone, Debug)]
pub struct ChatInput {
    pub messages: Vec<String>,
    pub tools: Vec<String>,
    pub enable_thinking: bool,
    /// The client explicitly asked for thinking off with a request field (enable_thinking
    /// false, or reasoning_effort none/minimal). Distinct from a server-side default and
    /// from the `/no_think` soft switch, which is message text: renderers with no
    /// thinking-off form refuse only explicit asks.
    pub thinking_off_requested: bool,
    pub reasoning_effort: ReasoningEffort,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice { #[default] Auto, None, Required, Named(String) }
/// Keep the requested high effort for templates that distinguish high from max.
/// Qwen retains its existing high-to-xhigh alias.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningEffort { Low, Medium, High, #[default] XHigh }
impl ReasoningEffort {
    pub fn qwen(self) -> engine_tokenizer::chat::ReasoningEffort {
        use engine_tokenizer::chat::ReasoningEffort as Qwen;
        match self { Self::Low => Qwen::Low, Self::Medium => Qwen::Medium, Self::High | Self::XHigh => Qwen::XHigh }
    }
    pub fn name(&self) -> &'static str { self.qwen().name() }
    pub(crate) fn glm(self) -> &'static str {
        match self { Self::Low => "low", Self::High => "high", Self::Medium | Self::XHigh => "max" }
    }
}
impl ChatInput {
    /// Only forced choices open a call or suppress the normal thought channel.
    pub fn forced_tool_choice(&self) -> bool {
        matches!(self.tool_choice, ToolChoice::Required | ToolChoice::Named(_))
    }
    /// Keep history intact while hiding callable definitions for a disabled turn.
    pub fn callable_tools(&self) -> &[String] {
        if self.tool_choice == ToolChoice::None { &[] } else { &self.tools }
    }
    /// The same prefix is rendered into the prompt and consumed by the output parser.
    pub fn generation_prefix(&self, hybrid: bool) -> String {
        match (&self.tool_choice, hybrid) {
            (ToolChoice::Auto | ToolChoice::None, _) => String::new(),
            (ToolChoice::Required, false) => "<tool_call>\n{\"name\": \"".into(),
            (ToolChoice::Required, true) => "<tool_call>\n<function=".into(),
            (ToolChoice::Named(name), false) => {
                format!("<tool_call>\n{{\"name\": {}, \"arguments\": ", quote(name))
            }
            (ToolChoice::Named(name), true) => format!("<tool_call>\n<function={name}>\n"),
        }
    }
}
#[derive(Clone, Debug)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Chat(ChatInput),
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DraftMode { #[default] Auto, Plain, Strict, Fast, Sampled }
impl DraftMode {
    pub fn require_typed_mtp(self) -> Result<(), String> {
        if matches!(self, Self::Fast | Self::Sampled) {
            return Err("typed hybrid MoE MTP requires draft auto, plain or strict for byte-identical answers; seeded sampling is supported".into());
        }
        Ok(())
    }
    pub(crate) fn name(self) -> &'static str {
        match self { Self::Auto => "auto", Self::Plain => "plain", Self::Strict => "strict",
            Self::Fast => "fast", Self::Sampled => "sampled" }
    }
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "plain" => Ok(Self::Plain), "strict" => Ok(Self::Strict), "fast" => Ok(Self::Fast),
            "sampled" => Ok(Self::Sampled),
            _ => Err("draft must be auto, plain, strict, fast or sampled".into()),
        }
    }
}
/// Typed MoE MTP currently has only a single-chat, built-in draft contract.
pub fn typed_mtp_admission(active: Option<usize>, external: bool, mode: DraftMode) -> Result<(), String> {
    if active != Some(1) { return Err("typed hybrid MoE MTP requires --max-active 1; multi-chat MTP is not wired".into()); }
    if external { return Err("typed hybrid MoE MTP does not support --draft-model".into()); }
    mode.require_typed_mtp()
}
#[derive(Clone, Debug)]
pub struct Request {
    /// Test-only kernel arm; false is A.
    pub arm_x: bool,
    pub prompt: Prompt,
    pub max_tokens: usize,
    pub(crate) max_tokens_explicit: bool,
    pub chat: bool,
    pub stream: bool,
    /// Chat usage framing; None retains the legacy finish-chunk receipt.
    pub stream_usage: Option<bool>,
    pub stop: Vec<String>,
    pub cache: bool,
    /// Exact cache namespace; never changes the model's prompt or sampling.
    pub cache_salt: Option<String>,
    /// Continue through model EOS tokens until the requested generation limit.
    pub ignore_eos: bool,
    /// Bounded affinity hint, never a cache identity or authorization scope.
    pub prompt_cache_key: Option<String>,
    pub include_usage_timing: bool,
    /// An explicit cache:false also opts this request out of automatic reuse.
    pub prefix_cache: bool,
    pub session: Option<crate::cache::SessionKey>,
    /// Retain this complete canonical prompt, independently of session history.
    pub kv_checkpoint: bool,
    /// Malformed hints report a miss and fall back without invalidating a prompt.
    pub kv_handle: Option<Result<crate::cache::SessionKey, ()>>,
    /// Optional fresh client secret, so cancellation also works while queued.
    pub cancel: Option<crate::cache::SessionKey>,
    pub sampling: spark_engine::sample::Sampling,
    /// Larger priorities run before background (zero) requests.
    pub priority: u8,
    pub fork: bool,
    pub draft: Option<DraftMode>,
    /// Requested model proposals, zero selects plain; bounded by resident depth.
    pub mtp_depth: Option<usize>,
    /// Derived Flash-Next policy, never an HTTP or environment selector.
    pub prompt_lookup: bool,
    pub prompt_lookup_max_draft: usize,
}

/// Request metadata only; never fed back into the sampler or draft policy.
#[derive(Clone, Debug)]
pub(crate) struct SamplingEcho {
    pub sampling: spark_engine::sample::Sampling,
    pub draft: DraftMode,
    pub mtp_depth: usize,
    pub thinking: bool,
    pub reasoning_effort: Option<&'static str>,
    pub draft_used: bool,
    pub model_draft_used: bool,
    pub prefill: Option<crate::usage::Prefill>,
    pub metadata: Option<crate::metadata::Metadata>,
    /// The process's best-effort event counts, read once when the scheduler builds the reply; none in a bare echo.
    pub events: Option<String>,
}
impl SamplingEcho {
    pub fn new(request: &Request, draft: DraftMode, mtp_depth: usize) -> Self {
        Self { sampling: request.sampling, draft, mtp_depth,
            thinking: matches!(&request.prompt, Prompt::Chat(input) if input.enable_thinking),
            reasoning_effort: match &request.prompt {
                Prompt::Chat(input) => Some(if input.enable_thinking { input.reasoning_effort.name() } else { "none" }),
                _ => None,
            },
            draft_used: false, model_draft_used: false, prefill: None, metadata: None, events: None }
    }
    pub fn with_metadata(mut self, limits: &Limits) -> Self {
        self.metadata = limits.metadata.clone();
        self
    }
    /// GLM's template always renders its think channel and an effort word (low, high,
    /// else max). Report what was rendered, not the unresolved request fields.
    pub fn with_glm_thinking(mut self, input: &ChatInput) -> Self {
        self.thinking = true;
        self.reasoning_effort = Some(input.reasoning_effort.glm());
        self
    }
    /// Owns a snapshot of the counts: every serialization of this echo prints the same events.
    pub fn with_events(mut self) -> Self {
        self.events = Some(engine_model::events::counts_json());
        self
    }
    fn sampling_json(&self, seed: &str) -> String {
        let s = self.sampling;
        // min_p appears only when set, so replies without it keep their bytes.
        let min_p = if s.min_p > 0.0 { format!("\"min_p\":{},", s.min_p) } else { String::new() };
        format!("{{\"temperature\":{},\"top_k\":{},\"top_p\":{},{min_p}\"repetition_penalty\":{},\"presence_penalty\":{},\"frequency_penalty\":{},\"seed\":{seed}}}",
            s.temperature, s.top_k, s.top_p, s.repetition_penalty, s.presence_penalty, s.frequency_penalty)
    }
    pub fn json(&self) -> String {
        let mtp_depth = if self.draft == DraftMode::Plain { 0 } else { self.mtp_depth };
        let prefill = self.prefill.as_ref().map(|p| format!(",{}", p.json())).unwrap_or_default();
        let metadata = self.metadata.as_ref().map(|m| format!(",{}", m.fields(self.draft, mtp_depth))).unwrap_or_default();
        let effort = self.reasoning_effort.map(|e| format!(",\"reasoning_effort\":{}", quote(e))).unwrap_or_default();
        // A card shows a fallback the same day.
        let events = self.events.as_ref().map(|e| format!(",\"events\":{e}")).unwrap_or_default();
        format!("{{\"sampling\":{},\"draft\":\"{}\",\"mtp_depth\":{},\"thinking\":{}{effort},\"draft_used\":{},\"model_draft_used\":{}{prefill}{metadata}{events}}}",
            self.sampling_json(&self.sampling.seed.to_string()), self.draft.name(), mtp_depth, self.thinking,
            self.draft_used, self.model_draft_used)
    }
    pub fn log(&self, id: u64) { eprintln!("request_sampling id={id} vade={}", self.json()); }
    /// All callers supply a complete encoded JSON object, including errors.
    pub fn attach(&self, mut body: String) -> String {
        debug_assert!(body.ends_with('}'));
        body.pop();
        body.push_str(&format!(",\"vade\":{}}}", self.json()));
        body
    }
    /// Adapter finish/error batches may contain several semantic SSE events.
    pub fn attach_events(&self, events: String) -> String {
        let mut result = String::with_capacity(events.len() + 512);
        for line in events.split_inclusive('\n') {
            if let Some(body) = line.strip_prefix("data: ").map(str::trim_end).filter(|s| s.starts_with('{')) {
                result.push_str("data: ");
                result.push_str(&self.attach(body.to_owned()));
                result.push('\n');
            } else { result.push_str(line); }
        }
        result
    }
    pub fn log_defaults(limits: &Limits, draft: DraftMode, mtp_depth: usize) {
        // Qwen fallback top-p depends on thinking. Log both branches and the
        // completion branch; omitted seeds are freshly randomized per request.
        let defaults = |thinking| {
            let echo = Self { sampling: default_sampling(limits, thinking, false), draft, mtp_depth, thinking,
                reasoning_effort: None,
                draft_used: false, model_draft_used: false, prefill: None, metadata: None, events: None };
            format!("{{\"sampling\":{},\"draft\":\"{}\",\"mtp_depth\":{mtp_depth},\"thinking\":{thinking}}}",
                echo.sampling_json("null"), draft.name())
        };
        eprintln!("sampling_defaults chat={} chat_thinking_on={} chat_thinking_off={} completions={} seed_policy=random_per_request",
            defaults(limits.default_thinking), defaults(true), defaults(false), defaults(false));
    }
}

pub(crate) const SUPPORTED_SAMPLING_FIELDS: &[&str] = &["temperature", "top_p", "top_k", "min_p", "seed",
    "repetition_penalty", "presence_penalty", "frequency_penalty"];

fn default_sampling(limits: &Limits, thinking: bool, named: bool) -> spark_engine::sample::Sampling {
    let mut sampling = spark_engine::sample::Sampling::qwen(thinking);
    sampling.temperature = 0.0;
    let defaults = if limits.greedy_if_unsampled && !named {
        Some(spark_engine::sample::Sampling::greedy())
    } else { limits.default_sampling };
    if let Some(defaults) = defaults {
        sampling = spark_engine::sample::Sampling { seed: sampling.seed, ..defaults };
    }
    sampling
}

/// An optional field set to null means the same as a missing one.
pub(crate) fn present<'a>(object: &'a std::collections::BTreeMap<String, Value>, key: &str) -> Option<&'a Value> {
    object.get(key).filter(|v| **v != Value::Null)
}

pub(crate) const IMAGE_INPUT_UNSUPPORTED: &str = "image input is unsupported by this text-only engine";
/// Text-only OpenAI content parts concatenate in their original order.
pub(crate) fn text_content(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let fields = part.object().map_err(|e| e.to_string())?;
                if fields.get("type").and_then(|v| v.string().ok()) == Some("image_url") { return Err(IMAGE_INPUT_UNSUPPORTED.into()); }
                if fields.get("type").and_then(|v| v.string().ok()) != Some("text") {
                    return Err("only text content parts are supported".into());
                }
                text.push_str(json::field(fields, "text").and_then(Value::string)
                    .map_err(|e| e.to_string())?);
            }
            Ok(text)
        }
        _ => Err("message content must be a string or text parts".into()),
    }
}

pub fn parse(body: &[u8], chat: bool, limits: &Limits) -> Result<Request, String> {
    parse_for_model(body, chat, limits, MODEL_ID)
}
pub(crate) fn parse_for_model(body: &[u8], chat: bool, limits: &Limits, model_id: &str) -> Result<Request, String> {
    parse_with_reservation(body, chat, limits, model_id, None)
}
pub(crate) fn parse_with_reservation(body: &[u8], chat: bool, limits: &Limits, model_id: &str,
    reservation: Option<usize>) -> Result<Request, String> {
    let value = json::parse(
        body,
        json::Limits {
            bytes: limits.body_bytes,
            depth: 24,
            nodes: limits.input_tokens.saturating_add(65536),
            string_bytes: limits.body_bytes,
        },
    )
    .map_err(|e| e.to_string())?;
    let object = value.object().map_err(|e| e.to_string())?;
    let get = |key| present(object, key);
    let arm_x = match object.get("arm") {
        Some(_) if !limits.ab_ptx => return Err("arm requires test-only VADE_AB_PTX on flash-serve".into()),
        Some(Value::String(arm)) if arm == "A" => false,
        Some(Value::String(arm)) if arm == "X" => true,
        None => false,
        _ => return Err("arm must be \"A\" or \"X\"".into()),
    };
    if limits.ab_ptx && ["session", "fork", "kv_handle", "kv_checkpoint"].iter().any(|k| get(k).is_some()) {
        return Err("request cache handles are disabled under VADE_AB_PTX".into());
    }
    let stream = get("stream").map(Value::boolean).transpose()
        .map_err(|e| e.to_string())?.unwrap_or(false);
    if get("n").is_some_and(|v| v.u64().ok() != Some(1)) {
        return Err("only n=1 is supported".into());
    }
    let stop = match get("stop") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.len() <= 4 => a.iter()
            .map(|v| {
                v.string().map(str::to_owned).map_err(|_| "stop entries must be strings".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("stop must be a string or an array of up to 4 strings".into()),
    };
    if stop.iter().any(String::is_empty) { return Err("stop strings must not be empty".into()); }
    if stop.iter().any(|s| s.len() > 1024) { return Err("stop strings must be at most 1024 bytes".into()); }
    if let Some(model) = get("model")
        && model.string().map_err(|e| e.to_string())? != model_id {
            return Err("unknown model".into());
        }
    let prompt = if chat {
        let messages = json::field(object, "messages")
            .and_then(Value::array)
            .map_err(|e| e.to_string())?;
        if messages.is_empty() || messages.len() > 4096 {
            return Err("messages count outside 1..4096".into());
        }
        let tools = match get("tools") {
            None => Vec::new(),
            Some(v) => {
                let list = v.array().map_err(|e| e.to_string())?;
                if list.len() > 64 {
                    return Err("too many tools".into());
                }
                raw_array(
                    raw_member(
                        std::str::from_utf8(body).map_err(|_| "invalid UTF-8")?,
                        "tools",
                    )?
                    .ok_or("missing tools")?,
                )?
            }
        };
        let tool_choice = match get("tool_choice") {
            None => ToolChoice::Auto,
            Some(Value::String(s)) => match s.as_str() {
                "auto" => ToolChoice::Auto,
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                _ => {
                    return Err("tool_choice must be auto, none, required or a named function".into());
                }
            },
            Some(value) => {
                let choice = value.object().map_err(|e| e.to_string())?;
                if json::field(choice, "type").and_then(Value::string).map_err(|e| e.to_string())? != "function" {
                    return Err("tool_choice type must be function".into());
                }
                let function = json::field(choice, "function").and_then(Value::object).map_err(|e| e.to_string())?;
                ToolChoice::Named(json::field(function, "name").and_then(Value::string).map_err(|e| e.to_string())?.into())
            }
        };
        let forced_tool_choice = matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_));
        if forced_tool_choice {
            if tools.is_empty() { return Err("forced tool_choice requires tools".into()); }
            if let ToolChoice::Named(name) = &tool_choice {
                if name.is_empty() || name.len() > 128 || name.chars().any(|c| c.is_control() || matches!(c, '<' | '>')) {
                    return Err("invalid forced tool name".into());
                }
                let declared = get("tools").and_then(|v| v.array().ok()).is_some_and(|tools| {
                        tools.iter().any(|tool| {
                            tool.object().ok().and_then(|o| o.get("function")).and_then(|v| v.object().ok())
                        .and_then(|o| o.get("name")).and_then(|v| v.string().ok()) == Some(name.as_str())
                        })
                    });
                if !declared { return Err("forced tool name is not declared".into()); }
            }
        }
        let parallel_tool_calls = get("parallel_tool_calls").map(Value::boolean).transpose()
            .map_err(|e| e.to_string())?.unwrap_or(true);
        let kwargs = get("chat_template_kwargs")
            .map(Value::object).transpose().map_err(|e| e.to_string())?;
        let effort = reasoning_effort(object)?;
        let effort = kwargs.map(reasoning_effort).transpose()?.flatten().or(effort);
        let thinking_field = kwargs.and_then(|o| present(o, "enable_thinking"))
            .or_else(|| get("enable_thinking"))
            .map(Value::boolean).transpose().map_err(|e| e.to_string())?;
        // Provenance of an off resolution, by the same precedence as enable_thinking:
        // an explicit enable_thinking field, else an explicit effort. The server
        // default is never a client ask.
        let thinking_off_requested = match thinking_field {
            Some(on) => !on,
            None => effort.is_some_and(|e| !e.0),
        };
        let mut enable_thinking = thinking_field
            .unwrap_or(effort.map(|e| e.0).unwrap_or(limits.default_thinking));
        // enable_thinking, then reasoning_effort, then the server default. A hard
        // disable wins; otherwise, on a template that has the soft switch, the final
        // switch in the last user turn wins. The switch is message text: it changes
        // the resolution, never the provenance above, and a template without it
        // (Limits::thinking_soft_switch false) reads no message text at all.
        if enable_thinking && limits.thinking_soft_switch
            && let Some(content) = messages.iter().rev().filter_map(|m| m.object().ok())
                .find(|m| m.get("role").and_then(|v| v.string().ok()) == Some("user"))
                .and_then(|m| m.get("content")).map(text_content).transpose()? {
                enable_thinking = match (content.rfind("/think"), content.rfind("/no_think")) {
                    (Some(on), Some(off)) => on > off,
                    (None, Some(_)) => false,
                    _ => true,
                };
            }
        Prompt::Chat(ChatInput {
            messages: raw_array(
                raw_member(
                    std::str::from_utf8(body).map_err(|_| "invalid UTF-8")?,
                    "messages",
                )?
                .ok_or("missing messages")?,
            )?,
            tools,
            // Forced calls begin in the model's tool syntax rather than its thought channel.
            enable_thinking: enable_thinking && !forced_tool_choice,
            thinking_off_requested,
            reasoning_effort: effort.map(|e| e.1).unwrap_or_default(),
            tool_choice,
            parallel_tool_calls,
        })
    } else {
        match (get("prompt"), get("prompt_token_ids")) {
            (Some(text @ Value::String(_)), None) => {
                Prompt::Text(text.string().map_err(|e| e.to_string())?.to_owned())
            }
            (Some(Value::Array(items)), None) if items.iter().any(|v| matches!(v, Value::String(_))) => {
                match items.as_slice() {
                    [Value::String(text)] => Prompt::Text(text.clone()),
                    _ => return Err("only one string prompt is supported".into()),
                }
            }
            (Some(ids @ Value::Array(_)), None) | (None, Some(ids)) => {
                let ids = ids.array().map_err(|e| e.to_string())?;
                if reservation.is_none() && ids.len() > limits.input_tokens {
                    return Err(crate::budget::token_refusal(ids.len() as u128, limits.input_tokens, ids.len(), 0, "--input-tokens"));
                }
                Prompt::Tokens(
                    ids.iter()
                        .map(|v| {
                            u32::try_from(v.u64().map_err(|e| e.to_string())?)
                                .map_err(|_| "token ID overflow".into())
                        })
                        .collect::<Result<_, String>>()?,
                )
            }
            _ => return Err("specify exactly one of prompt or prompt_token_ids".into()),
        }
    };
    let thinking = matches!(&prompt, Prompt::Chat(input) if input.enable_thinking);
    let requested_max_tokens = get("max_completion_tokens").or_else(|| get("max_tokens"));
    let max_tokens = match reservation {
        Some(n) => n,
        None => match requested_max_tokens {
            Some(v) => usize::try_from(v.u64().map_err(|e| e.to_string())?)
                .map_err(|_| "max_tokens overflow")?,
            None => limits.output_tokens,
        },
    };
    if reservation.is_none() && max_tokens == 0 {
        return Err(crate::budget::token_refusal(0, limits.output_tokens, 0, 0, "--output-tokens (max_tokens must be positive)"));
    }
    // Admission probes reserve the exact requested suffix; inference requests
    // use the configured ceiling, then the rendered prompt's remaining room.
    let max_tokens = if reservation.is_none() { max_tokens.min(limits.output_tokens) } else { max_tokens };
    // Select the base before applying non-null field values. Even a named
    // null field retains the checkpoint base; only a fully absent sampling
    // policy opts into the direct FlashNext greedy default.
    let named_sampling = object.keys().any(|key| engine_model::sampling_policy::is_field(key));
    let mut sampling = default_sampling(limits, thinking, named_sampling);
    for (key, target) in [
        ("temperature", &mut sampling.temperature),
        ("top_p", &mut sampling.top_p),
        ("min_p", &mut sampling.min_p),
        ("repetition_penalty", &mut sampling.repetition_penalty),
        ("presence_penalty", &mut sampling.presence_penalty),
        ("frequency_penalty", &mut sampling.frequency_penalty),
    ] {
        if let Some(value) = get(key) {
            *target = value.f64().map_err(|e| e.to_string())?;
        }
    }
    if let Some(value) = get("top_k") {
        sampling.top_k = usize::try_from(value.u64().map_err(|e| e.to_string())?)
            .map_err(|_| "top_k overflow")?;
    }
    if let Some(value) = get("seed") {
        sampling.seed = value.u64().map_err(|e| e.to_string())?;
    }
    sampling.validate()?;
    let cache = get("cache")
        .map(Value::boolean)
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(false) && !limits.ab_ptx;
    let cache_salt = get("cache_salt").map(|value| {
        let salt = value.string().map_err(|_| "cache_salt must be a string")?;
        if salt.is_empty() || salt.len() > 256 {
            return Err("cache_salt must contain 1..256 UTF-8 bytes".to_owned());
        }
        Ok(salt.to_owned())
    }).transpose()?;
    let ignore_eos = get("ignore_eos").map(Value::boolean).transpose()
        .map_err(|e| e.to_string())?.unwrap_or(false);
    // Qwen's plain-text encoding has no automatic BOS/EOS insertion. Token-ID
    // prompts are already encoded and are always passed through unchanged.
    get("add_special_tokens").map(Value::boolean).transpose().map_err(|e| e.to_string())?;
    let session = get("session")
        .map(|v| crate::cache::SessionKey::parse(v.string().map_err(|e| e.to_string())?))
        .transpose()?;
    if session.is_some() && !cache {
        return Err("session requires cache:true".into());
    }
    let priority = get("priority")
        .map(|v| {
            u8::try_from(v.u64().map_err(|e| e.to_string())?)
                .map_err(|_| "priority must be an integer in 0..255".to_owned())
        })
        .transpose()?
        .unwrap_or(0);
    let fork = get("fork")
        .map(Value::boolean)
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(false);
    if fork && session.is_none() {
        return Err("fork requires a session capability".into());
    }
    let mtp_depth = get("mtp_depth")
        .filter(|_| limits.next_mtp_depth.is_some())
        .map(|v| {
            let depth = v
                .u64()
                .map_err(|_| "mtp_depth must be an integer in 0..=3")?;
            if depth > 3 {
                return Err("mtp_depth must be an integer in 0..=3");
            }
            if limits
                .next_mtp_depth
                .is_some_and(|resident| depth as usize > resident)
            {
                return Err(if limits.next_mtp_depth == Some(0) {
                    "mtp_depth > 0 is unavailable on this Next service; use mtp_depth: 0"
                } else {
                    "requested mtp_depth exceeds resident model depth"
                });
            }
            Ok(depth as usize)
        })
        .transpose()?;
    Ok(Request {
        arm_x,
        prompt_lookup: limits.next_mtp_depth.is_some_and(|n| n > 0)
            && mtp_depth != Some(0)
            && spark_engine::draft::plain_greedy(sampling)
            && get("draft")
                .and_then(|v| v.string().ok())
                .is_none_or(|m| m != "plain"),
        prompt_lookup_max_draft: 15,
        mtp_depth,
        draft: get("draft").map(|v| DraftMode::parse(v.string().map_err(|e| e.to_string())?)).transpose()?
            .or(mtp_depth.map(|k| {
                if k == 0 {
                    DraftMode::Plain
                } else {
                    DraftMode::Strict
                }
            })),
        prompt,
        max_tokens,
        max_tokens_explicit: reservation.is_some() || requested_max_tokens.is_some(),
        chat,
        stream,
        stop,
        cache,
        cache_salt,
        ignore_eos,
        // Optional scheduling hints must not invalidate otherwise valid requests.
        prompt_cache_key: get("prompt_cache_key").and_then(|v| v.string().ok())
            .filter(|key| !key.is_empty() && key.len() <= 256).map(str::to_owned),
        include_usage_timing: get("include_usage_timing").map(Value::boolean).transpose()
            .map_err(|e| e.to_string())?.unwrap_or(false),
        stream_usage: get("stream_options").and_then(|v| v.object().ok())
            .and_then(|o| o.get("include_usage")).and_then(|v| v.boolean().ok()),
        prefix_cache: !limits.ab_ptx && (get("cache").is_none() || cache),
        session,
        kv_checkpoint: get("kv_checkpoint").map(Value::boolean).transpose()
            .map_err(|e| e.to_string())?.unwrap_or(false),
        kv_handle: get("kv_handle").map(|v| {
            v.string().map_err(|_| ())
            .and_then(|text| crate::cache::SessionKey::parse(text).map_err(|_| ()))
        }),
        cancel: get("cancel").map(cancel_key).transpose()?,
        sampling,
        priority,
        fork,
    })
}

fn cancel_key(value: &Value) -> Result<crate::cache::SessionKey, String> {
    crate::cache::SessionKey::parse(value.string().map_err(|e| e.to_string())?)
        .map_err(|_| "invalid cancellation capability".into())
}

/// Qwen3.8's official effort names are low, medium and xhigh. OpenAI high
/// aliases xhigh; none/minimal preserve the existing explicit thinking disable.
/// Validate effort even when enable_thinking decides the thought channel.
fn reasoning_effort(object: &std::collections::BTreeMap<String, Value>) -> Result<Option<(bool, ReasoningEffort)>, String> {
    use ReasoningEffort::{Low, Medium, High, XHigh};
    let Some(effort) = present(object, "reasoning_effort") else { return Ok(None);
    };
    match effort.string() {
        Ok("none" | "minimal") => Ok(Some((false, XHigh))),
        Ok("low") => Ok(Some((true, Low))),
        Ok("medium") => Ok(Some((true, Medium))),
        Ok("high") => Ok(Some((true, High))),
        Ok("xhigh" | "max" | "ultra") => Ok(Some((true, XHigh))),
        _ => Err("reasoning effort must be none, minimal, low, medium, high, xhigh, max or ultra".into()),
    }
}

/// Retain only a suffix that could complete a stop string in the next chunk.
#[derive(Default)]
pub(crate) struct StopBuffer {
    pub pending: String,
    pub matched: bool,
    pub matched_sequence: Option<String>,
}
impl StopBuffer {
    pub fn push(&mut self, text: &str, stops: &[String]) -> String {
        if self.matched { return String::new(); }
        self.pending.push_str(text);
        let end = stops.iter().filter_map(|s| self.pending.find(s).map(|at| (at, s)))
            .min_by_key(|(at, sequence)| (*at, sequence.len()));
        let emit = if let Some((end, sequence)) = end {
            self.matched = true;
            self.matched_sequence = Some(sequence.clone());
            end
        } else {
            let keep = stops.iter().flat_map(|s| {
                    s.char_indices().skip(1)
                .take_while(|(i, _)| *i <= self.pending.len()).map(|(i, _)| &s[..i])
                })
                .filter(|prefix| self.pending.ends_with(prefix)).map(str::len).max().unwrap_or(0);
            self.pending.len() - keep
        };
        let out = self.pending[..emit].to_owned();
        if self.matched { self.pending.clear(); } else { self.pending.drain(..emit); }
        out
    }
}

/// Only the bearer of a request's capability can cancel it. Repeated requests
/// have the same result even after that generation has left the scheduler.
pub(crate) fn parse_cancel(body: &[u8]) -> Result<crate::cache::SessionKey, String> {
    let value = json::parse(body, json::Limits {
        bytes: 256, depth: 2, nodes: 4, string_bytes: 64,
    }).map_err(|e| e.to_string())?;
    let object = value.object().map_err(|e| e.to_string())?;
    if object.len() != 1 {
        return Err("cancellation requires only a cancel capability".into());
    }
    cancel_key(object.get("cancel").ok_or("missing cancel capability")?)
}

pub(crate) fn parse_checkpoint_release(body: &[u8]) -> Result<Option<crate::cache::SessionKey>, String> {
    let value = json::parse(body, json::Limits { bytes: 256, depth: 2, nodes: 4, string_bytes: 128 })
        .map_err(|_| "invalid checkpoint release".to_owned())?;
    let object = value.object().map_err(|_| "invalid checkpoint release")?;
    if object.len() != 1 || !object.contains_key("kv_handle") { return Err("release requires only kv_handle".into()); }
    Ok(object["kv_handle"].string().ok().and_then(|s| crate::cache::SessionKey::parse(s).ok()))
}

pub use crate::chat_output::quote;
pub fn stringify(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.clone(),
        Value::String(v) => quote(v),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(stringify).collect::<Vec<_>>().join(",")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}:{}", quote(k), stringify(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}
pub fn error_json(code: &str, message: &str) -> String {
    let detail = match (code, message) {
        ("invalid_request_error", IMAGE_INPUT_UNSUPPORTED) => Some("image_input_unsupported"),
        ("invalid_request_error", m) if m.starts_with("Requested size does not fit.\nAsked: input=") || matches!(m, "input token limit exceeded" | "context token limit exceeded" | "context length overflow" | "chat template byte limit exceeded" | "input byte limit exceeded" | "token count limit exceeded") => Some("context_length_exceeded"),
        _ => None,
    }.map(quote).unwrap_or_else(|| "null".into());
    format!(
        "{{\"error\":{{\"type\":{},\"message\":{},\"param\":null,\"code\":{detail}}}}}",
        quote(code),
        quote(message)
    )
}

// Raw slices retain object insertion order for the checkpoint's chat template.
// All public callers first validate the complete request with the bounded JSON
// parser; these helpers additionally reject malformed scans rather than panic.
fn whitespace(bytes: &[u8], pos: &mut usize) {
    while bytes.get(*pos).is_some_and(u8::is_ascii_whitespace) {
        *pos += 1;
    }
}
fn value_end(bytes: &[u8], pos: &mut usize) -> Result<(), String> {
    whitespace(bytes, pos);
    let start = *pos;
    let mut stack = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    while let Some(&b) = bytes.get(*pos) {
        if quoted {
            *pos += 1;
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                quoted = false;
                if stack.is_empty() {
                    return Ok(());
                }
            }
            continue;
        }
        match b {
            b'"' => {
                quoted = true;
                *pos += 1;
            }
            b'{' | b'[' => {
                stack.push(b);
                *pos += 1;
                if stack.len() > 24 {
                    return Err("JSON depth exceeded".into());
                }
            }
            b'}' | b']' if !stack.is_empty() => {
                let opening = stack.pop().unwrap();
                if (opening == b'{') != (b == b'}') {
                    return Err("invalid JSON nesting".into());
                }
                *pos += 1;
                if stack.is_empty() {
                    return Ok(());
                }
            }
            b',' | b'}' | b']' if stack.is_empty() => break,
            b if b.is_ascii_whitespace() && stack.is_empty() => break,
            _ => *pos += 1,
        }
    }
    if *pos == start || quoted || !stack.is_empty() {
        Err("invalid JSON value slice".into())
    } else {
        Ok(())
    }
}
pub(crate) fn raw_member<'a>(source: &'a str, name: &str) -> Result<Option<&'a str>, String> {
    let bytes = source.as_bytes();
    let mut pos = 0;
    whitespace(bytes, &mut pos);
    if bytes.get(pos) != Some(&b'{') {
        return Err("expected JSON object".into());
    }
    pos += 1;
    loop {
        whitespace(bytes, &mut pos);
        if bytes.get(pos) == Some(&b'}') {
            return Ok(None);
        }
        let start = pos;
        value_end(bytes, &mut pos)?;
        let key =
            json::parse(&bytes[start..pos], json::Limits::default()).map_err(|e| e.to_string())?;
        let key = key.string().map_err(|e| e.to_string())?;
        whitespace(bytes, &mut pos);
        if bytes.get(pos) != Some(&b':') {
            return Err("missing object colon".into());
        }
        pos += 1;
        whitespace(bytes, &mut pos);
        let start = pos;
        value_end(bytes, &mut pos)?;
        if key == name {
            return Ok(Some(&source[start..pos]));
        }
        whitespace(bytes, &mut pos);
        match bytes.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => return Ok(None),
            _ => return Err("missing object separator".into()),
        }
    }
}
pub(crate) fn raw_array(source: &str) -> Result<Vec<String>, String> {
    let bytes = source.as_bytes();
    let mut pos = 0;
    whitespace(bytes, &mut pos);
    if bytes.get(pos) != Some(&b'[') {
        return Err("expected JSON array".into());
    }
    pos += 1;
    let mut out = Vec::new();
    loop {
        whitespace(bytes, &mut pos);
        if bytes.get(pos) == Some(&b']') {
            return Ok(out);
        }
        let start = pos;
        value_end(bytes, &mut pos)?;
        out.push(source[start..pos].to_owned());
        whitespace(bytes, &mut pos);
        match bytes.get(pos) {
            Some(b',') => pos += 1,
            Some(b']') => return Ok(out),
            _ => return Err("missing array separator".into()),
        }
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn explicit_stops_match_at_each_unicode_chunk_boundary() {
        for stops in [vec!["雪🚀".to_owned(), "END".to_owned()], vec!["END".to_owned(), "雪🚀".to_owned()]] {
            let text = "é start 雪🚀 tail END";
            let boundaries = text.char_indices().map(|(i,_)| i).chain(std::iter::once(text.len())).collect::<Vec<_>>();
            for &first in &boundaries { for &second in boundaries.iter().filter(|&&i| i >= first) {
                let mut buffer = StopBuffer::default();
                let mut out = String::new();
                for part in [&text[..first], &text[first..second], &text[second..]] { out.push_str(&buffer.push(part, &stops)); }
                assert_eq!(out, "é start ");
                assert!(buffer.matched && buffer.pending.is_empty());
                assert_eq!(buffer.matched_sequence.as_deref(), Some("雪🚀"));
                assert!(buffer.push("later", &stops).is_empty());
            } }
        }
        let mut buffer = StopBuffer::default();
        assert_eq!(buffer.push("keep 雪", &["雪🚀".into()]), "keep ");
        assert!(!buffer.matched);
        assert_eq!(buffer.pending, "雪", "a length or EOS stop must flush this unmatched suffix");
        for raw in [r#""""#, r#"[""]"#] {
            let body = format!(r#"{{"prompt":"go","stop":{raw}}}"#);
            assert!(parse(body.as_bytes(), false, &Limits::default()).is_err());
        }
    }
    #[test]
    fn equal_position_stops_report_the_shortest_match() {
        for stops in [vec!["END".to_owned(), "END NOW".to_owned()], vec!["END NOW".to_owned(), "END".to_owned()]] {
            let mut buffer = StopBuffer::default();
            assert_eq!(buffer.push("END NOW", &stops), "");
            assert_eq!(buffer.matched_sequence.as_deref(), Some("END"));
        }
    }
    #[test]
    fn an_echo_prints_the_event_snapshot_it_owns() {
        let request = parse(br#"{"prompt_token_ids":[1]}"#, false, &Limits::default()).unwrap();
        let bare = SamplingEcho::new(&request, DraftMode::Plain, 0);
        assert!(!bare.json().contains("\"events\""), "a bare echo reads no process counts");
        engine_model::events::note("echo_snapshot_test", format_args!("n=1"));
        let echo = bare.clone().with_events();
        let first = echo.json();
        assert!(first.contains("\"events\":{") && first.contains("\"echo_snapshot_test\":"), "{first}");
        engine_model::events::note("echo_snapshot_test", format_args!("n=2"));
        assert_eq!(echo.json(), first, "a later event does not change this reply");
        assert_eq!(bare.json(), SamplingEcho::new(&request, DraftMode::Plain, 0).json());
    }
    #[test]
    fn thinking_off_provenance_and_glm_echo_truth() {
        let limits = Limits::default();
        let off = |body: &[u8], limits: &Limits| { let Prompt::Chat(input) = parse(body, true, limits).unwrap().prompt else { panic!() }; input.thinking_off_requested };
        assert!(off(br#"{"messages":[{"role":"user","content":"hi"}],"enable_thinking":false}"#, &limits));
        assert!(off(br#"{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}"#, &limits));
        assert!(off(br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}"#, &limits));
        assert!(off(br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"minimal"}"#, &limits));
        // The soft switch is message text, never a client ask, whatever it resolves to.
        assert!(!off(br#"{"messages":[{"role":"user","content":"hi /no_think"}]}"#, &limits));
        assert!(!off(br#"{"messages":[{"role":"user","content":"hi /no_think then /think"}]}"#, &limits));
        assert!(!off(br#"{"messages":[{"role":"user","content":"hi /no_think"}],"enable_thinking":true}"#, &limits));
        assert!(!off(br#"{"messages":[{"role":"user","content":"hi"}],"enable_thinking":true,"reasoning_effort":"none"}"#, &limits));
        let off_default = Limits { default_thinking: false, ..Default::default() };
        assert!(!off(br#"{"messages":[{"role":"user","content":"hi"}]}"#, &off_default), "server default is not a client ask");
        // GLM echo truth: thinking on, effort in GLM's rendered words.
        for (fields, word) in [("", "max"), (r#","reasoning_effort":"low""#, "low"), (r#","reasoning_effort":"high""#, "high"), (r#","reasoning_effort":"medium""#, "max")] {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{fields}}}"#);
            let request = parse(body.as_bytes(), true, &limits).unwrap();
            let Prompt::Chat(input) = &request.prompt else { panic!() };
            let json = SamplingEcho::new(&request, DraftMode::Plain, 0).with_glm_thinking(input).json();
            assert!(json.contains(r#""thinking":true"#) && json.contains(&format!(r#""reasoning_effort":"{word}""#)), "{fields}: {json}");
        }
    }
    #[test]
    fn the_soft_switch_needs_its_template() {
        let thinking = |body: &str, limits: &Limits| { let Prompt::Chat(input) = parse(body.as_bytes(), true, limits).unwrap().prompt else { panic!() };
            (input.enable_thinking, input.thinking_off_requested) };
        let body = |text: &str, fields: &str| format!(r#"{{"messages":[{{"role":"user","content":"{text}"}}]{fields}}}"#);
        // A template with the switch (Qwen): while thinking is on, the last switch of the last user turn decides.
        let qwen = Limits::default();
        assert!(qwen.thinking_soft_switch);
        for (text, fields, on) in [("hi /no_think", "", false), ("hi /no_think then /think", "", true), ("hi /think", "", true),
            ("hi /no_think", r#","enable_thinking":true"#, false), ("hi /no_think", r#","reasoning_effort":"high""#, false),
            // A hard disable wins over the switch.
            ("hi /think", r#","enable_thinking":false"#, false), ("hi /think", r#","reasoning_effort":"none""#, false)] {
            assert_eq!(thinking(&body(text, fields), &qwen).0, on, "{text} {fields}");
        }
        // A template without the switch (GLM, served with thinking default off): the same request with and
        // without the words resolves the same way, for every field form. A pasted path or a quoted tag is text.
        let glm = Limits { thinking_soft_switch: false, default_thinking: false, ..Limits::default() };
        for fields in ["", r#","enable_thinking":true"#, r#","reasoning_effort":"low""#, r#","reasoning_effort":"max""#,
            r#","chat_template_kwargs":{"enable_thinking":true}"#, r#","enable_thinking":false"#, r#","reasoning_effort":"none""#] {
            let plain = thinking(&body("explain this", fields), &glm);
            for text in ["explain /no_think", "see https://example.com/no_think_guide", "first /think then /no_think", "/no_think"] {
                assert_eq!(thinking(&body(text, fields), &glm), plain, "{text} {fields}");
            }
        }
        // An explicit field keeps its meaning there: on stays on, off stays an explicit ask.
        assert_eq!(thinking(&body("a /no_think", r#","enable_thinking":true"#), &glm), (true, false));
        assert_eq!(thinking(&body("a /no_think", r#","reasoning_effort":"low""#), &glm), (true, false));
        assert_eq!(thinking(&body("a /think", r#","enable_thinking":false"#), &glm), (false, true));
    }
    #[test]
    fn defaults_overrides_and_rejections() {
        let limits = Limits::default();
        let render = |choice| { let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object"}}}}}}]{choice}}}"#); let request = parse(body.as_bytes(), true, &limits).unwrap(); let Prompt::Chat(input) = &request.prompt else { panic!() };
            if input.tool_choice == ToolChoice::None { assert!(input.callable_tools().is_empty() && input.enable_thinking); for hybrid in [false, true] { let mut parser = crate::http::output_parser(&request, &limits, hybrid).unwrap().unwrap(); let mut events = parser.push(b"</think><tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>").unwrap(); events.extend(parser.finish().unwrap()); assert!(events.iter().all(|e| matches!(e, crate::chat_output::Event::Content(_) | crate::chat_output::Event::Reasoning(_)))); } } crate::tokenizer::format_chat(input, limits.body_bytes).unwrap().into_bytes() };
        assert_eq!(render(r#", "tool_choice":"auto""#), render("")); assert!(!render(r#", "tool_choice":"none""#).windows(8).any(|s| s == b"# Tools\n"));
        for (effort, expected, on) in [("low", "low", true), ("medium", "medium", true), ("high", "xhigh", true), ("xhigh", "xhigh", true), ("max", "xhigh", true), ("ultra", "xhigh", true), ("none", "xhigh", false), ("minimal", "xhigh", false)] {
            for field in [format!(r#""reasoning_effort":"{effort}""#), format!(r#""reasoning_effort":"high","chat_template_kwargs":{{"reasoning_effort":"{effort}"}}"#)] {
                let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{field}}}"#); let Prompt::Chat(input) = parse(body.as_bytes(), true, &limits).unwrap().prompt else { panic!() }; assert_eq!((input.reasoning_effort.name(), input.enable_thinking), (expected, on));
            }
        }
        let thinking = parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            true,
            &limits,
        )
        .unwrap();
        assert_eq!(
            thinking.sampling,
            spark_engine::sample::Sampling {
                seed: thinking.sampling.seed,
                temperature: 0.0,
                ..spark_engine::sample::Sampling::qwen(true)
            }
        );
        assert_eq!(thinking.max_tokens, limits.output_tokens);
        let plain = parse(br#"{"messages":[{"role":"user","content":"hi"}],"enable_thinking":false,"stream":true}"#, true, &limits).unwrap();
        assert_eq!(
            plain.sampling,
            spark_engine::sample::Sampling {
                seed: plain.sampling.seed,
                temperature: 0.0,
                ..spark_engine::sample::Sampling::qwen(false)
            }
        );
        assert_ne!(plain.sampling.seed, thinking.sampling.seed);
        assert_eq!(plain.max_tokens, limits.output_tokens);
        let custom = parse(br#"{"prompt":"hi","stream":true,"temperature":0.9,"top_k":7,"top_p":0.75,"repetition_penalty":1.1,"presence_penalty":1.5,"frequency_penalty":0.25,"seed":18446744073709551615}"#, false, &limits).unwrap();
        assert_eq!(
            custom.sampling,
            spark_engine::sample::Sampling {
                temperature: 0.9,
                top_k: 7,
                top_p: 0.75,
                min_p: 0.0,
                repetition_penalty: 1.1,
                presence_penalty: 1.5,
                frequency_penalty: 0.25,
                seed: u64::MAX
            }
        );
        assert!(
            parse(
                br#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0}"#,
                true,
                &limits
            )
            .is_ok()
        );
        for field in [
            "\"temperature\":-1",
            "\"top_p\":0",
            "\"top_k\":-1",
            "\"top_k\":1.5",
            "\"seed\":-1",
            "\"repetition_penalty\":0",
            "\"presence_penalty\":3",
        ] {
            let body = format!("{{\"prompt\":\"hi\",\"stream\":true,{field}}}");
            assert!(parse(body.as_bytes(), false, &limits).is_err(), "{field}");
        }
    }
}

#[cfg(test)]
#[test]
fn request_mtp_depth_is_bounded_and_selects_plain_or_strict() {
    let limits = crate::Limits {
        next_mtp_depth: Some(3),
        ..crate::Limits::default()
    };
    for depth in 0..=3 {
        let body = format!(r#"{{"prompt":"hi","max_tokens":8,"mtp_depth":{depth}}}"#);
        let request = parse(body.as_bytes(), false, &limits).unwrap();
        assert_eq!(request.mtp_depth, Some(depth));
        assert_eq!(request.draft, Some(if depth == 0 { DraftMode::Plain } else { DraftMode::Strict }));
    }
    for depth in ["4", "-1", "1.5", "true", "\"2\""] {
        let body = format!(r#"{{"prompt":"hi","mtp_depth":{depth}}}"#);
        assert!(parse(body.as_bytes(), false, &limits).is_err());
    }
    assert_eq!(parse(br#"{"prompt":"hi"}"#, false, &limits).unwrap().mtp_depth, None);
}

#[cfg(test)]
#[path = "lookup_test.rs"]
mod lookup_tests;

#[test]
fn ab_arm_parsing_and_flag_off_rejection() {
    let mut limits = Limits::default();
    for arm in ["\"A\"", "\"X\"", "null", "1", "\"x\""] {
        let body = format!("{{\"prompt\":\"hello\",\"arm\":{arm}}}");
        assert!(parse(body.as_bytes(), false, &limits).unwrap_err().contains("VADE_AB_PTX"));
        limits.ab_ptx = true;
        let result = parse(body.as_bytes(), false, &limits);
        if arm == "\"A\"" || arm == "\"X\"" { assert_eq!(result.unwrap().arm_x, arm == "\"X\""); }
        else { assert!(result.unwrap_err().contains("arm must be")); }
        limits.ab_ptx = false;
    }
    for ab_ptx in [false, true] {
        limits.ab_ptx = ab_ptx;
        let responses = crate::responses::parse(br#"{"input":"hello","arm":"X"}"#, &limits, MODEL_ID, None);
        let messages = crate::anthropic::parse(br#"{"model":"test","max_tokens":1,"messages":[{"role":"user","content":"hello"}],"arm":"X"}"#, &limits, MODEL_ID, None);
        assert_eq!(responses.map(|r| r.0.arm_x).unwrap_or(false), ab_ptx);
        assert_eq!(messages.map(|r| r.0.arm_x).unwrap_or(false), ab_ptx);
    }
    limits.ab_ptx = true;
    let request = parse(br#"{"prompt":"hello","cache":true}"#, false, &limits).unwrap();
    assert!(!request.arm_x && !request.cache && !request.prefix_cache);
    assert!(parse(br#"{"prompt":"hello","kv_checkpoint":true}"#, false, &limits).is_err());
}

#[cfg(test)]
#[test]
fn greedy_default_applies_only_when_no_sampling_field_is_named_on_every_surface() {
    use spark_engine::sample::{Sampler, Sampling};
    let checkpoint = Sampling { temperature: 1.7, top_k: 20, top_p: 0.95,
        repetition_penalty: 1.2, presence_penalty: 0.3, frequency_penalty: 0.4,
        ..Sampling::greedy() };
    let limits = Limits { default_sampling: Some(checkpoint), greedy_if_unsampled: true,
        ..Default::default() };
    let parse_surface = |surface, fields: &str, limits: &Limits| -> Result<Request, String> {
        let input = match surface {
            0 => r#""prompt_token_ids":[1],"max_tokens":4"#,
            1 | 3 => r#""messages":[{"role":"user","content":"hello"}],"max_tokens":4"#,
            _ => r#""input":"hello","max_output_tokens":4"#,
        };
        let body = format!("{{{input}{fields}}}");
        match surface {
            0 | 1 => parse(body.as_bytes(), surface == 1, limits),
            2 => crate::responses::parse(body.as_bytes(), limits, crate::MODEL_ID, None).map(|p| p.0),
            _ => crate::anthropic::parse(body.as_bytes(), limits, crate::MODEL_ID, None).map(|p| p.0),
        }
    };
    let token_ids = |sampling| {
        let mut sampler = Sampler::new(sampling).unwrap();
        let mut ids = vec![1];
        for row in [[0., 1., 2., 1.], [0., 2., 2., 1.], [3., 1., 2., 0.], [0., 1., 2., 3.]] {
            ids.push(sampler.sample(&row, &ids, 1, ids.len(), |_| true).unwrap());
        }
        ids[1..].to_vec()
    };
    let greedy = token_ids(Sampling::greedy());
    assert_eq!(greedy, [2, 1, 0, 3]);
    for surface in 0..4 {
        for _ in 0..8 {
            let request = parse_surface(surface, "", &limits).unwrap();
            assert_eq!(request.sampling.temperature, 0.);
            assert_eq!(request.sampling.top_k, 0);
            assert_eq!(request.sampling.top_p, 1.);
            assert_eq!(request.sampling.repetition_penalty, 1.);
            assert_eq!(token_ids(request.sampling), greedy, "surface={surface}");
        }
        for field in SUPPORTED_SAMPLING_FIELDS {
            let value = if *field == "seed" { "7" } else if *field == "top_k" { "3" } else { "0.7" };
            let request = parse_surface(surface, &format!(",\"{field}\":{value}"), &limits).unwrap();
            match *field {
                "temperature" => assert_eq!(request.sampling.temperature, 0.7),
                "top_k" => assert_eq!(request.sampling.top_k, 3),
                "top_p" => assert_eq!(request.sampling.top_p, 0.7),
                "min_p" => assert_eq!(request.sampling.min_p, 0.7, "surface={surface}"),
                "repetition_penalty" => assert_eq!(request.sampling.repetition_penalty, 0.7),
                "presence_penalty" => assert_eq!(request.sampling.presence_penalty, 0.7),
                "frequency_penalty" => assert_eq!(request.sampling.frequency_penalty, 0.7),
                _ => assert_eq!(request.sampling.seed, 7),
            }
            if *field != "temperature" { assert_eq!(request.sampling.temperature, checkpoint.temperature, "{field}"); }
            if *field != "top_k" { assert_eq!(request.sampling.top_k, checkpoint.top_k); }
            if *field != "top_p" { assert_eq!(request.sampling.top_p, checkpoint.top_p); }
            if *field != "min_p" { assert_eq!(request.sampling.min_p, checkpoint.min_p); }
            if *field != "repetition_penalty" { assert_eq!(request.sampling.repetition_penalty, checkpoint.repetition_penalty); }
            if *field != "presence_penalty" { assert_eq!(request.sampling.presence_penalty, checkpoint.presence_penalty); }
            if *field != "frequency_penalty" { assert_eq!(request.sampling.frequency_penalty, checkpoint.frequency_penalty); }
            if *field == "seed" { assert_eq!(request.sampling.seed, 7); }
            let null = parse_surface(surface, &format!(",\"{field}\":null"), &limits).unwrap();
            assert_eq!(null.sampling.temperature, checkpoint.temperature, "named null {field}");
        }
        let zero = parse_surface(surface, ",\"temperature\":0", &limits).unwrap();
        assert_eq!(zero.sampling.temperature, 0.);
        assert_eq!(zero.sampling.top_k, checkpoint.top_k);
        assert_eq!(zero.sampling.top_p, checkpoint.top_p);
        assert_eq!(zero.sampling.repetition_penalty, checkpoint.repetition_penalty);
        let mut ordinary = limits.clone();
        ordinary.greedy_if_unsampled = false;
        assert_eq!(parse_surface(surface, "", &ordinary).unwrap().sampling.temperature, checkpoint.temperature);
        // Parse the exact fragment emitted by the real gateway. A checkpoint
        // with non-neutral penalties must still yield the direct greedy IDs.
        let defaults = engine_model::sampling_policy::GREEDY_DEFAULTS_JSON;
        let injected = parse_surface(surface, &format!(",{}", &defaults[1..defaults.len()-1]), &limits).unwrap();
        assert_eq!(injected.sampling.temperature, 0.);
        assert_eq!(injected.sampling.top_k, 0);
        assert_eq!(injected.sampling.top_p, 1.);
        assert_eq!(injected.sampling.repetition_penalty, 1.);
        assert_eq!(injected.sampling.presence_penalty, 0.);
        assert_eq!(injected.sampling.frequency_penalty, 0.);
        assert_eq!(injected.sampling.seed, 20260905);
        assert_eq!(token_ids(injected.sampling), greedy);
        for key in engine_model::sampling_policy::FIELDS.split(',')
            .filter(|key| !SUPPORTED_SAMPLING_FIELDS.contains(key)) {
            let values: &[&str] = if key == "n" { &["null", "1"] }
                else { &["null", "0", "1.20e0", "{}"] };
            for value in values {
                let request = parse_surface(surface, &format!(",\"{key}\":{value}"), &limits).unwrap();
                assert_eq!(request.sampling.temperature, checkpoint.temperature, "{key} {value}");
                assert_eq!(request.sampling.top_k, checkpoint.top_k);
                assert_eq!(request.sampling.top_p, checkpoint.top_p);
                assert_eq!(request.sampling.repetition_penalty, checkpoint.repetition_penalty);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn min_p_parses_and_echoes_only_when_set_and_greedy_keeps_its_bytes() {
    use spark_engine::sample::Sampling;
    let checkpoint = Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, ..Sampling::greedy() };
    let limits = Limits { default_sampling: Some(checkpoint), greedy_if_unsampled: true, ..Default::default() };
    let body = |fields: &str| format!(r#"{{"prompt_token_ids":[1],"max_tokens":4{fields}}}"#);
    // No sampling field: the greedy settings and today's echo bytes, with no min_p key.
    let unset = parse(body("").as_bytes(), false, &limits).unwrap();
    assert_eq!(unset.sampling, Sampling { seed: unset.sampling.seed, ..Sampling::greedy() });
    assert_eq!(SamplingEcho::new(&unset, DraftMode::Plain, 0).sampling_json("7"),
        r#"{"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":7}"#);
    let sampled = parse(body(r#","seed":3"#).as_bytes(), false, &limits).unwrap();
    assert_eq!(SamplingEcho::new(&sampled, DraftMode::Plain, 0).sampling_json("3"),
        r#"{"temperature":1,"top_k":20,"top_p":0.95,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":3}"#);
    let set = parse(body(r#","min_p":0.05,"seed":3"#).as_bytes(), false, &limits).unwrap();
    assert_eq!(set.sampling, Sampling { min_p: 0.05, seed: 3, ..checkpoint });
    assert_eq!(SamplingEcho::new(&set, DraftMode::Plain, 0).sampling_json("3"),
        r#"{"temperature":1,"top_k":20,"top_p":0.95,"min_p":0.05,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":3}"#);
    for bad in ["-0.1", "1.5", "\"x\""] {
        assert!(parse(body(&format!(r#","min_p":{bad}"#)).as_bytes(), false, &limits).is_err(), "{bad}");
    }
}
