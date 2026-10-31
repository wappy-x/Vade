//! HTTP/1.1, one request per connection, Content-Length only. No persistent
//! connection, multiplexing, authentication, or remote bind by
//! default. A complete request may half-close its sending side.
use crate::{
    Limits, Tokenizer, chat_output,
    generation::{self, FinishReason},
    protocol::{self, Prompt},
};
pub(crate) fn output_parser(
    request: &protocol::Request,
    limits: &Limits,
    hybrid: bool,
) -> Result<Option<chat_output::Parser>, String> {
    use engine_model::json::{self, Value};
    let Prompt::Chat(input) = &request.prompt else {
        return Ok(None);
    };
    let mut names = Vec::new();
    let mut json_parameters = std::collections::BTreeMap::new();
    for tool in input.callable_tools() {
        let value = json::parse(tool.as_bytes(), json::Limits { bytes: limits.body_bytes, string_bytes: limits.body_bytes, ..Default::default() }).map_err(|e| e.to_string())?;
        let object = value.object().map_err(|e| e.to_string())?;
        let function = json::field(object, "function")
            .and_then(|v| v.object())
            .map_err(|e| e.to_string())?;
        let name = json::field(function, "name").and_then(|v| v.string()).map_err(|e| e.to_string())?;
        if !matches!(&input.tool_choice, protocol::ToolChoice::Named(selected) if selected != name) {
            names.push(name.to_owned());
        }
        // Schema types decide which XML parameter values are JSON rather than strings.
        let properties = function.get("parameters").and_then(|p| p.object().ok())
            .and_then(|p| p.get("properties")).and_then(|p| p.object().ok());
        json_parameters.insert(name.to_owned(), properties.into_iter().flatten().filter(|(_, schema)| {
            match schema.object().ok().and_then(|o| o.get("type")) {
                Some(Value::String(kind)) => kind != "string",
                Some(Value::Array(kinds)) => !kinds.iter().any(|k| matches!(k, Value::String(s) if s == "string")),
                _ => false,
            }
        }).map(|(k, _)| k.clone()).collect());
    }
    let bounds = chat_output::Limits {
        total_bytes: limits.output_bytes,
        tool_call_bytes: (64 * 1024).min(limits.output_bytes),
        ..chat_output::Limits::default()
    };
    let parser = chat_output::Parser::new(bounds, names)?;
    let parser = if hybrid { parser.hybrid(input.enable_thinking, json_parameters) } else { parser };
    Ok(Some(parser.tool_policy(&input.generation_prefix(hybrid),
        input.forced_tool_choice(), !input.parallel_tool_calls)?))
}

/// FNV-1a over the original body keeps tool IDs independent of admission order.
pub(crate) fn request_hash(body: &[u8]) -> u64 {
    body.iter().fold(0xcbf29ce484222325, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3))
}
pub(crate) fn chat_delta(
    event: chat_output::Event,
    request_id: u64,
    tool_index: &mut usize,
) -> String {
    match event {
        chat_output::Event::Content(text) => format!("{{\"content\":{}}}", protocol::quote(&text)),
        chat_output::Event::Reasoning(text) => {
            format!("{{\"reasoning_content\":{}}}", protocol::quote(&text))
        }
        chat_output::Event::ToolStart { name } => {
            let index = *tool_index;
            format!(
                "{{\"tool_calls\":[{{\"index\":{index},\"id\":\"call-{request_id}-{index}\",\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":\"\"}}}}]}}",
                protocol::quote(&name)
            )
        }
        chat_output::Event::ToolArguments(arguments) => format!(
            "{{\"tool_calls\":[{{\"index\":{},\"function\":{{\"arguments\":{}}}}}]}}",
            *tool_index, protocol::quote(&arguments)
        ),
        chat_output::Event::ToolCall { .. } => {
            *tool_index += 1;
            "{}".into()
        }
    }
}

#[derive(Default)]
pub(crate) struct Completion {
    pub content: String,
    reasoning: String,
    tools: Vec<String>,
    bytes: usize,
}
impl Completion {
    pub fn push(&mut self, event: chat_output::Event, id: u64, index: &mut usize,
        limits: &Limits) -> Result<(), String> {
        match event {
            chat_output::Event::Content(s) => { self.bytes += s.len(); self.content.push_str(&s); }
            chat_output::Event::Reasoning(s) => { self.bytes += s.len(); self.reasoning.push_str(&s); }
            chat_output::Event::ToolStart { .. } | chat_output::Event::ToolArguments(_) => {}
            chat_output::Event::ToolCall { name, arguments } => {
                let tool = format!("{{\"id\":\"call-{id}-{index}\",\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}",
                    protocol::quote(&name), protocol::quote(&arguments));
                *index += 1;
                self.bytes += tool.len();
                self.tools.push(tool);
            }
        }
        if self.bytes > limits.output_bytes.saturating_sub(1024) {
            return Err("output byte limit exceeded".into());
        }
        Ok(())
    }
    pub fn json(&self, model: &str, chat: bool, id: u64, created: u64,
        reason: &str, usage: generation::Usage, echo: &protocol::SamplingEcho) -> String {
        let payload = if chat {
            let reasoning = if self.reasoning.is_empty() { String::new() }
                else { format!(",\"reasoning_content\":{}", protocol::quote(&self.reasoning)) };
            let tools = if self.tools.is_empty() { String::new() }
                else { format!(",\"tool_calls\":[{}]", self.tools.join(",")) };
            format!("\"message\":{{\"role\":\"assistant\",\"content\":{},\"refusal\":null{reasoning}{tools}}}", protocol::quote(&self.content))
        } else { format!("\"text\":{}", protocol::quote(&self.content)) };
        echo.attach(format!("{{\"id\":\"completion-{id}\",\"object\":{},\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,{payload},\"finish_reason\":{}}}],\"usage\":{}}}",
            protocol::quote(if chat { "chat.completion" } else { "text_completion" }),
            protocol::quote(model), protocol::quote(reason), usage.json(false)))
    }
}

pub struct HttpRequest {
    pub path: String,
    pub body: Vec<u8>,
    pub authorization: Option<String>,
}

pub(crate) fn route(path: &str) -> &str { path.split('?').next().unwrap_or(path) }
pub(crate) fn anthropic_route(path: &str) -> bool {
    matches!(route(path), "/v1/messages" | "/v1/messages/count_tokens")
}

#[derive(Clone, Copy)]
enum PromptLimit { Input, Context }
impl PromptLimit {
    fn exceeded(input: usize, context: usize, limits: &Limits, generating: bool) -> Option<Self> {
        if input > limits.input_tokens { Some(Self::Input) }
        else if input > limits.context_tokens.min(context)
            || (generating && input == limits.context_tokens.min(context)) { Some(Self::Context) }
        else { None }
    }
    fn message(self, path: &str, detail: String) -> String {
        match (anthropic_route(path), self) {
            (true, Self::Input | Self::Context) => "prompt is too long".into(),
            _ => detail,
        }
    }
}

/// Wire format alone varies; all formats consume the same chat parser events.
pub(crate) enum Output {
    Responses(crate::responses::Output),
    Anthropic(crate::anthropic::Output),
}
impl Output {
    pub fn start(&mut self, model: &str, id: u64, created: u64, usage: generation::Usage) -> String {
        match self {
            Self::Responses(o) => o.start(model, id, created),
            Self::Anthropic(o) => o.start(model, id, usage),
        }
    }
    pub fn push(&mut self, event: chat_output::Event, id: u64, call: &mut usize, limits: &Limits) -> Result<String, String> {
        match self {
            Self::Responses(o) => o.push(event, id, call, limits),
            Self::Anthropic(o) => o.push(event, id, call, limits),
        }
    }
    pub fn json(&self, model: &str, id: u64, created: u64, reason: &str, usage: generation::Usage,
        stop: Option<&str>, echo: &protocol::SamplingEcho) -> String {
        echo.attach(match self {
            Self::Responses(o) => o.json(model, id, created, if reason == "length" { "incomplete" } else { "completed" }, Some(usage), None),
            Self::Anthropic(o) => o.json(model, id, reason, usage, stop),
        })
    }
    pub fn finish(&mut self, model: &str, id: u64, created: u64, reason: &str, usage: generation::Usage,
        stop: Option<&str>, echo: &protocol::SamplingEcho) -> String {
        echo.attach_events(match self {
            Self::Responses(o) => o.finish(model, id, created, if reason == "length" { "incomplete" } else { "completed" }, usage),
            Self::Anthropic(o) => o.finish(model, id, reason, usage, stop),
        })
    }
    pub fn failed(&mut self, model: &str, id: u64, created: u64, usage: generation::Usage,
        status: u16, code: &str, message: &str, stream: bool, echo: &protocol::SamplingEcho) -> String {
        let body = match self {
            Self::Responses(o) if stream => o.failed(model, id, created, usage, code, message),
            Self::Responses(o) => o.json(model, id, created, "failed", Some(usage), Some((code, message))),
            Self::Anthropic(_) if stream && matches!(code, "client_cancelled" | "generation_deadline") =>
                format!("event: error\ndata: {}\n\n", anthropic_error(status, &protocol::error_json(code, message)).1),
            Self::Anthropic(o) if stream => o.failed(if code == "invalid_model_output" { "api_error" } else { anthropic_error_type(status) }, message),
            Self::Anthropic(_) => protocol::error_json(code, message),
        };
        if stream { echo.attach_events(body) } else { echo.attach(body) }
    }
}

pub(crate) fn anthropic_error_type(status: u16) -> &'static str {
    match status {
        401 => "authentication_error", 403 => "permission_error", 404 => "not_found_error",
        413 => "request_too_large", 429 => "rate_limit_error", 503 | 529 => "overloaded_error",
        500..=599 => "api_error", _ => "invalid_request_error",
    }
}
/// Convert failures even when parsing never reached a translator (headers, body,
/// loading or capacity). Successful responses remain byte-for-byte unchanged.
pub(crate) fn anthropic_error(status: u16, body: &str) -> (u16, String) {
    use engine_model::json;
    let value = json::parse(body.as_bytes(), json::Limits::default()).ok();
    let error = value.as_ref().and_then(|v| v.object().ok()).and_then(|o| o.get("error"))
        .and_then(|v| v.object().ok());
    let message = error.and_then(|o| o.get("message")).and_then(|v| v.string().ok()).unwrap_or("request failed");
    let code = error.and_then(|o| o.get("code").or_else(|| o.get("type"))).and_then(|v| v.string().ok());
    let model_output = code == Some("invalid_model_output");
    let status = if status == 400 && message.contains("byte limit exceeded") { 413 }
        else if status == 503 { 529 } else { status };
    let message = match message {
        "input token limit exceeded" | "context token limit exceeded" | "context length overflow" => "prompt is too long",
        other => other,
    };
    let mut body = crate::anthropic::error_json(if model_output { "api_error" } else { anthropic_error_type(status) }, message);
    if let Some(code @ ("client_cancelled" | "generation_deadline" | "image_input_unsupported")) = code {
        // Keep Anthropic's error type while making interruption distinguishable
        // from malformed input or a retryable generic server failure.
        body.truncate(body.len() - 2);
        body.push_str(&format!(",\"code\":{}}}}}", protocol::quote(code)));
    }
    if let Some(vade) = value.as_ref().and_then(|v| v.object().ok()).and_then(|o| o.get("vade")) {
        body.pop();
        body.push_str(&format!(",\"vade\":{}}}", protocol::stringify(vade)));
    }
    (status, body)
}

/// Exact Host names and origins prevent rebound DNS names from reaching the
/// local service. Loopback Host ports may differ when using an SSH tunnel.
/// Wildcard bind addresses are never accepted as Host names.
pub(crate) struct AccessPolicy {
    credential: Option<crate::auth::ApiKey>,
    hosts: Vec<String>,
    origins: Vec<String>,
}
impl AccessPolicy {
    pub fn new(address: std::net::SocketAddr, limits: &Limits) -> Result<Self, String> {
        let mut names = vec!["localhost".to_owned(), "127.0.0.1".to_owned(), "[::1]".to_owned()];
        if !address.ip().is_unspecified() {
            names.push(match address.ip() {
                std::net::IpAddr::V4(ip) => ip.to_string(),
                std::net::IpAddr::V6(ip) => format!("[{ip}]"),
            });
        }
        let mut hosts = Vec::new();
        let mut origins = Vec::new();
        for name in names {
            hosts.push(name.clone());
            let authority = format!("{name}:{}", address.port());
            hosts.push(authority.clone());
            origins.push(format!("http://{authority}"));
            if address.port() == 80 { origins.push(format!("http://{name}")); }
        }
        for host in &limits.allowed_hosts { hosts.push(Self::authority(host)?); }
        for origin in &limits.allowed_origins { origins.push(Self::origin(origin)?); }
        Ok(Self { hosts, origins, credential: limits.api_key.clone() })
    }
    fn authority(text: &str) -> Result<String, String> {
        if text.is_empty() || text.len() > 255
            || !text.bytes().all(|b| b.is_ascii_alphanumeric() || b".-:[]".contains(&b)) {
            return Err("invalid allowed Host authority".into());
        }
        let port = if let Some(bracketed) = text.strip_prefix('[') {
            let (ip, tail) = bracketed.split_once(']').ok_or("invalid IPv6 Host")?;
            ip.parse::<std::net::Ipv6Addr>().map_err(|_| "invalid IPv6 Host")?;
            if tail.is_empty() { None } else { Some(tail.strip_prefix(':').ok_or("invalid Host port")?) }
        } else {
            let (name, port) = text.split_once(':').map_or((text, None), |(name, port)| (name, Some(port)));
            if name.is_empty() || name.bytes().any(|b| matches!(b, b'[' | b']')) {
                return Err("invalid Host name".into());
            }
            port
        };
        if let Some(port) = port {
            if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit())
                || port.parse::<u16>().is_err() {
                return Err("invalid Host port".into());
            }
        }
        Ok(text.to_ascii_lowercase())
    }
    fn origin(text: &str) -> Result<String, String> {
        let (scheme, authority) = text.split_once("://").ok_or("invalid Origin")?;
        if !matches!(scheme, "http" | "https") { return Err("invalid Origin scheme".into()); }
        Ok(format!("{scheme}://{}", Self::authority(authority)?))
    }
    fn validate(&self, host: &str, origin: Option<&str>) -> Result<(), String> {
        let authority = Self::authority(host)?;
        // Separate the validated port without splitting IPv6's own colons.
        let name = match authority.strip_prefix('[') {
            Some(ipv6) => {
                let (name, _) = ipv6
                    .split_once(']')
                    .ok_or("invalid IPv6 Host")?;
                name
            }
            None => {
                authority
                    .split_once(':')
                    .map(|(name, _)| name)
                    .unwrap_or(&authority)
            }
        };
        let loopback = matches!(name, "localhost" | "127.0.0.1" | "::1");
        if !loopback && !self.hosts.contains(&authority) {
            return Err("Host is not allowed".into());
        }
        if let Some(origin) = origin {
            if !self.origins.contains(&Self::origin(origin)?) {
                return Err("Origin is not allowed".into());
            }
        }
        Ok(())
    }
}

/// Validate and tokenize a scheduled request before model admission.
pub(crate) fn prepare_request<T: Tokenizer>(
    http: &HttpRequest,
    context: usize,
    vocab: usize,
    eos: &[u32],
    tokenizer: Option<&T>,
    limits: &Limits,
    model: &str,
) -> Result<(protocol::Request, Vec<u32>, Option<chat_output::Parser>, Option<Output>), String> {
    let result = render_request(http, context, tokenizer, limits, model, None)?;
    limits.validate()?;
    // Rendering clamps output to the remaining context. Classify only failed
    // prompt bounds here; generation and its successful inputs stay unchanged.
    let ceiling = PromptLimit::exceeded(result.1.len(), context, limits, true);
    generation::validate_bounds(context, vocab, eos, &result.1, result.0.max_tokens, limits)
        .map_err(|detail| match ceiling { Some(kind) => kind.message(&http.path, detail), None => detail })?;
    validate_token_mapping(&result.1, tokenizer)?;
    Ok(result)
}
pub(crate) fn validate_token_mapping<T: Tokenizer>(prompt: &[u32], tokenizer: Option<&T>) -> Result<(), String> {
    if tokenizer.is_some_and(|t| prompt.iter().any(|&id| !t.has_token(id))) {
        return Err("prompt token has no tokenizer mapping".into());
    }
    Ok(())
}
pub(crate) fn render_request<T: Tokenizer>(http: &HttpRequest, context: usize, tokenizer: Option<&T>,
    limits: &Limits, model: &str, reservation: Option<usize>)
    -> Result<(protocol::Request, Vec<u32>, Option<chat_output::Parser>, Option<Output>), String> {
    let profile = std::env::var("SPARK_PROFILE").as_deref() == Ok("1");
    let started = std::time::Instant::now();
    let path = route(&http.path);
    let (mut request, output) = if path == "/v1/responses" {
        let (request, output) = crate::responses::parse(&http.body, limits, model, reservation)?;
        (request, Some(Output::Responses(output)))
    } else if anthropic_route(path) {
        let (request, output) = crate::anthropic::parse(&http.body, limits, model, reservation)?;
        (request, Some(Output::Anthropic(output)))
    } else {
        (protocol::parse_with_reservation(&http.body, path == "/v1/chat/completions", limits, model, reservation)?, None)
    };
    let parsed = started.elapsed();
    let encoding = std::time::Instant::now();
    let prompt = match &request.prompt {
        Prompt::Tokens(v) => v.clone(),
        Prompt::Text(s) => tokenizer
            .ok_or("text input requires a tokenizer")?
            .encode(s)?,
        Prompt::Chat(input) => {
            let codec = tokenizer.ok_or("chat input requires a tokenizer")?;
            codec.encode_chat(input, limits.body_bytes)?
        }
    };
    let encoded = encoding.elapsed();
    if reservation.is_none() {
        request.max_tokens = request.max_tokens.min(limits.context_tokens.min(context)
            .saturating_sub(prompt.len()));
    }
    let mut parser = output_parser(&request, limits, tokenizer.is_some_and(|t| t.hybrid_chat()))?;
    if let Some(codec) = tokenizer.filter(|t| t.gemma_chat()) {
        parser = parser.map(|p| p.gemma4(codec.chat_thinking_open(&prompt)));
    }
    if profile {
        eprintln!("request_prepare_profile prompt_tokens={} parse_ms={:.3} tokenize_ms={:.3} validate_ms={:.3}",
            prompt.len(), parsed.as_secs_f64()*1000.0, encoded.as_secs_f64()*1000.0,
            started.elapsed().saturating_sub(parsed + encoded).as_secs_f64()*1000.0);
    }
    Ok((request, prompt, parser, output))
}

pub(crate) fn count_tokens<T: Tokenizer>(http: &HttpRequest, context: usize, vocab: usize,
    tokenizer: Option<&T>, limits: &Limits, model: &str) -> Result<String, String> {
    let (_, prompt, _, _) = render_request(http, context, tokenizer, limits, model, Some(0))?;
    validate_count_prompt_route(&prompt, context, vocab, tokenizer, limits, &http.path)?;
    Ok(format!("{{\"input_tokens\":{}}}", prompt.len()))
}
/// The vLLM-compatible tokenizer extension used by exact-length clients.
pub(crate) fn tokenize<T: Tokenizer>(http: &HttpRequest, context: usize, vocab: usize,
    tokenizer: Option<&T>, limits: &Limits, model: &str) -> Result<ControlBody, String> {
    let request = protocol::parse_with_reservation(&http.body, false, limits, model, Some(0))?;
    let Prompt::Text(text) = request.prompt else { return Err("tokenize prompt must be a string".into()); };
    let codec = tokenizer.ok_or("tokenization requires a tokenizer")?;
    let tokens = codec.encode(&text)?;
    validate_token_mapping(&tokens, tokenizer)?;
    if tokens.len() > limits.input_tokens || tokens.len() > limits.context_tokens.min(context) {
        return Err(crate::budget::token_refusal(tokens.len() as u128, limits.input_tokens.min(limits.context_tokens.min(context)), tokens.len(), 0, "the lower of --input-tokens and engine/configured context"));
    }
    if tokens.iter().any(|&token| token as usize >= vocab) {
        return Err("prompt token outside vocabulary".into());
    }
    Ok(ControlBody::tokens(tokens, limits.context_tokens.min(context)))
}

/// Control output has its own input/context bounds, independent of the answer
/// content budget. Keep token IDs once and serialize them as the socket drains.
pub(crate) enum ControlBody {
    Json(String),
    Tokens(TokenBody),
}
pub(crate) struct TokenBody {
    tokens: Vec<u32>,
    head: String,
    length: usize,
}
impl ControlBody {
    // Small enough to keep control writes cooperative with active generations.
    pub const CHUNK_BYTES: usize = 64 * 1024;

    pub fn tokens(tokens: Vec<u32>, context: usize) -> Self {
        let count = tokens.len();
        let head = format!("{{\"count\":{count},\"max_model_len\":{context},\"tokens\":[");
        // Compute this in the preparation worker, not on the engine thread.
        // Validated input has at most 262144 u32 IDs, so arithmetic is bounded.
        let length = head.len() + 2 + count.saturating_sub(1)
            + tokens.iter().map(|&token| token.max(1).ilog10() as usize + 1).sum::<usize>();
        Self::Tokens(TokenBody { tokens, head, length })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Json(body) => body.len(),
            Self::Tokens(body) => body.length,
        }
    }

    /// JSON cursors count bytes; token cursors count the prefix, IDs and suffix.
    /// Returns true only after every body byte has entered the pending queue.
    pub fn fill(&self, cursor: &mut usize, bytes: &mut Vec<u8>) -> bool {
        debug_assert!(bytes.is_empty());
        match self {
            Self::Json(body) => {
                let end = cursor.saturating_add(Self::CHUNK_BYTES).min(body.len());
                bytes.extend_from_slice(&body.as_bytes()[*cursor..end]);
                *cursor = end;
                end == body.len()
            }
            Self::Tokens(body) => {
                if *cursor == 0 {
                    bytes.extend_from_slice(body.head.as_bytes());
                    *cursor = 1;
                }
                while *cursor <= body.tokens.len() {
                    // At most ten decimal digits plus the separator per ID.
                    if bytes.len() + 13 > Self::CHUNK_BYTES {
                        return false;
                    }
                    let index = *cursor - 1;
                    if index != 0 {
                        bytes.push(b',');
                    }
                    append_token(bytes, body.tokens[index]);
                    *cursor += 1;
                }
                bytes.extend_from_slice(b"]}");
                true
            }
        }
    }
}

fn append_token(bytes: &mut Vec<u8>, mut token: u32) {
    // No allocation per ID, including the full u32 range and zero.
    let mut digits = [0; 10];
    let mut first = digits.len();
    loop {
        first -= 1;
        digits[first] = b'0' + (token % 10) as u8;
        token /= 10;
        if token == 0 {
            break;
        }
    }
    bytes.extend_from_slice(&digits[first..]);
}
pub(crate) fn validate_count_prompt<T: Tokenizer>(prompt: &[u32], context: usize, vocab: usize,
    tokenizer: Option<&T>, limits: &Limits) -> Result<(), String> {
    validate_count_prompt_route(prompt, context, vocab, tokenizer, limits, "")
}
fn validate_count_prompt_route<T: Tokenizer>(prompt: &[u32], context: usize, vocab: usize,
    tokenizer: Option<&T>, limits: &Limits, path: &str) -> Result<(), String> {
    limits.validate()?;
    validate_token_mapping(prompt, tokenizer)?;
    if prompt.is_empty() { return Err("prompt must contain at least one token".into()); }
    if let Some(kind) = PromptLimit::exceeded(prompt.len(), context, limits, false) {
        return Err(kind.message(path, crate::budget::token_refusal(prompt.len() as u128, limits.input_tokens.min(limits.context_tokens.min(context)), prompt.len(), 0, "the lower of --input-tokens and engine/configured context")));
    }
    if vocab == 0 || prompt.iter().any(|&t| t as usize >= vocab) { return Err("prompt token outside vocabulary".into()); }
    Ok(())
}

pub(crate) struct Head {
    pub path: String,
    /// Admission uses the original target, before preflights normalize to /cors.
    pub control_target: bool,
    pub length: usize,
    pub authorization: Option<String>,
    pub cors: String,
    pub continue_body: bool,
}
pub(crate) fn parse_head(head: &[u8], access: &AccessPolicy, admin: bool)
    -> Result<Head, String> {
    if head
        .iter()
        .any(|&b| b > 126 || (b < 32 && !matches!(b, b'\r' | b'\n' | b'\t')))
    {
        return Err("invalid HTTP header bytes".into());
    }
    let text = std::str::from_utf8(head).map_err(|_| "invalid HTTP header encoding")?;
    let mut lines = text.split("\r\n");
    let first = lines.next().ok_or("missing request line")?;
    let parts: Vec<_> = first.split(' ').collect();
    if parts.len() != 3 || parts[2] != "HTTP/1.1" {
        return Err("expected HTTP/1.1 request line".into());
    }
    let path = route(parts[1]);
    let valid_route = if admin { parts[0] == "POST" && path == "/v1/admin/swap" }
        else { (parts[0] == "POST"
            && matches!(path, "/tokenize" | "/v1/completions" | "/v1/chat/completions" | "/v1/responses" | "/v1/messages" | "/v1/messages/count_tokens" | "/v1/cancel" | "/v1/kv/release" | "/v1/admission"))
            || (parts[0] == "GET" && matches!(path, "/health" | "/v1/models" | "/v1/capabilities" | "/v1/events"))
            || (parts[0] == "OPTIONS" && matches!(path, "/tokenize" | "/v1/completions" | "/v1/chat/completions" | "/v1/responses" | "/v1/messages" | "/v1/messages/count_tokens" | "/v1/cancel" | "/v1/kv/release" | "/v1/admission" | "/v1/models" | "/v1/capabilities" | "/v1/events")) };
    let mut headers = std::collections::BTreeMap::new();
    for line in lines {
        let (name, value) = line.split_once(':').ok_or("invalid header line")?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
            || value.bytes().any(|b| matches!(b, b'\r' | b'\n'))
        {
            return Err("invalid header name or value".into());
        }
        if headers
            .insert(name.to_ascii_lowercase(), value.trim_matches([' ', '\t']))
            .is_some()
        {
            return Err("duplicate HTTP header".into());
        }
    }
    // Private Unix administration uses its existing socket authorization.
    let authenticated = !admin && !(parts[0] == "GET" && path == "/health") && access.credential.is_some();
    let owner = if authenticated {
        access.credential.as_ref().unwrap().authorize(
            headers.get("authorization").copied(), headers.get("x-api-key").copied())
            .ok_or(crate::auth::UNAUTHORIZED)?
    } else { false };
    if !valid_route { return Err("unsupported method or path".into()); }
    access.validate(headers.get("host").ok_or("Host header is required")?, headers.get("origin").copied())?;
    if headers.contains_key("transfer-encoding") {
        return Err("Transfer-Encoding is unsupported".into());
    }
    let continue_body = match headers.get("expect") {
        None => false,
        Some(value) if parts[0] == "POST" && value.eq_ignore_ascii_case("100-continue") => true,
        _ => return Err("unsupported HTTP expectation".into()),
    };
    if headers.contains_key("content-encoding") {
        return Err("Content-Encoding is unsupported".into());
    }
    let len = match headers.get("content-length") {
        Some(v) if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) => {
            v.parse::<usize>().map_err(|_| "Content-Length overflow")?
        }
        Some(_) => return Err("invalid Content-Length".into()),
        None if matches!(parts[0], "GET" | "OPTIONS") => 0,
        None => return Err("Content-Length is required".into()),
    };
    if parts[0] != "POST" && len != 0 {
        return Err("request method does not accept a body".into());
    }
    if parts[0] == "POST" && !headers.contains_key("content-type") {
        return Err("Content-Type must be application/json".into());
    }
    if let Some(kind) = headers.get("content-type") {
        if !kind
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/json")
        {
            return Err("Content-Type must be application/json".into());
        }
    }
    let mut cors = String::new();
    if let Some(origin) = headers.get("origin") {
        cors = format!("Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\nAccess-Control-Allow-Credentials: true\r\nAccess-Control-Expose-Headers: X-Session-Capability, X-Cancel-Capability, X-Prefix-Reused-Tokens, X-Sampling-Seed, X-Vade-Prompt-Tokens, X-Vade-Cached-Tokens, X-Vade-Estimated-First-Token-Ms, X-Vade-Estimated-Cold-Ms\r\n");
    }
    let preflight = parts[0] == "OPTIONS";
    if preflight {
        let method = headers.get("access-control-request-method").copied().unwrap_or("POST");
        if method != if matches!(path, "/v1/models" | "/v1/capabilities" | "/v1/events") { "GET" } else { "POST" } {
            return Err("unsupported CORS method".into());
        }
        let requested = headers.get("access-control-request-headers").copied().unwrap_or("content-type, authorization");
        if requested.split(',').any(|name| {
            let name = name.trim();
            name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
        }) { return Err("invalid CORS header names".into()); }
        cors.push_str(&format!("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: {requested}\r\nAccess-Control-Max-Age: 600\r\n"));
    }
    // Both credential headers name the same scope. Matching credentials are
    // accepted together; conflicting credentials are never resolved by priority.
    let authorization = if authenticated {
        // Credentials authenticate one scope; never retain the password in jobs/cache.
        headers.remove("authorization"); headers.remove("x-api-key");
        Some(if owner { "authenticated-owner" } else { "authenticated" }.to_owned())
    } else { headers.remove("authorization").map(str::to_owned) };
    let authorization = match headers.remove("x-api-key") {
        Some(key) => {
            if key.is_empty() {
                return Err("X-Api-Key must not be empty".into());
            }
            let scope = format!("Bearer {key}");
            if let Some(value) = authorization.as_deref() {
                let (scheme, bearer) = value.split_once(' ').unwrap_or(("", ""));
                if !scheme.eq_ignore_ascii_case("Bearer") || bearer != key {
                    return Err("credential headers must identify the same API key".into());
                }
            }
            Some(scope)
        }
        None => authorization,
    };
    Ok(Head { path: if preflight { "/cors".into() } else { path.into() }, length: len,
        control_target: matches!(path, "/health" | "/v1/cancel"),
        authorization, cors,
        continue_body: continue_body && len != 0 })
}

pub(crate) fn finish_reason(reason: FinishReason, tools: usize) -> &'static str {
    match reason {
        FinishReason::Stop if tools > 0 => "tool_calls",
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}
pub(crate) fn response_head(code: u16, length: usize, cors: &str) -> String {
    if code == 204 {
        return format!("HTTP/1.1 204 No Content\r\nConnection: close\r\nCache-Control: no-store\r\n{cors}\r\n");
    }
    let reason = match code {
        200 => "OK",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        529 => "Overloaded",
        _ => "Bad Request",
    };
    let retry = if matches!(code, 409 | 504) { "x-should-retry: false\r\n" } else { "" };
    format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {length}\r\nConnection: close\r\nCache-Control: no-store\r\n{retry}{cors}\r\n"
    )
}
pub(crate) fn sse_chunk(
    model: &str,
    chat: bool,
    id: u64,
    created: u64,
    text: &str,
    token: Option<u32>,
    finish: Option<&str>,
    usage: Option<generation::Usage>,
    echo: Option<&protocol::SamplingEcho>,
) -> String {
    let payload = if chat {
        format!("\"delta\":{text}")
    } else {
        format!("\"text\":{}", protocol::quote(text))
    };
    let token = token
        .map(|v| format!(",\"token_id\":{v}"))
        .unwrap_or_default();
    let usage = usage.map(|v| format!(",\"usage\":{}", v.json(false))).unwrap_or_default();
    let vade = echo.map(|v| format!(",\"vade\":{}", v.json())).unwrap_or_default();
    format!(
        "data: {{\"id\":\"completion-{id}\",\"object\":{},\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,{payload},\"finish_reason\":{}{token}}}]{usage}{vade}}}\n\n",
        protocol::quote(if chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        }),
        protocol::quote(model),
        finish.map(protocol::quote).unwrap_or_else(|| "null".into())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_call_mock_repeats_identical_bytes_for_each_api() {
        assert_eq!(request_hash(b""), 0xcbf29ce484222325);
        assert_eq!(request_hash(b"hello"), 0xa430d84680aabd0b);
        let limits = Limits::default();
        for (api, body) in [
            ("chat", r#"{"messages":[{"role":"user","content":"go"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"temperature":0,"seed":42}"#),
            ("responses", r#"{"input":"go","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"temperature":0,"seed":42}"#),
            ("messages", r#"{"messages":[{"role":"user","content":"go"}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"max_tokens":4,"temperature":0,"seed":42}"#),
        ] {
            let mock = |server_id| {
                let (request, mut output) = match api {
                    "responses" => { let (r, o) = crate::responses::parse(body.as_bytes(), &limits, "model", None).unwrap(); (r, Some(Output::Responses(o))) }
                    "messages" => { let (r, o) = crate::anthropic::parse(body.as_bytes(), &limits, "model", None).unwrap(); (r, Some(Output::Anthropic(o))) }
                    _ => (protocol::parse(body.as_bytes(), true, &limits).unwrap(), None),
                };
                let echo = protocol::SamplingEcho::new(&request, protocol::DraftMode::Plain, 0);
                let mut parser = output_parser(&request, &limits, false).unwrap().unwrap();
                let mut events = parser.push(br#"</think><tool_call>{"name":"f","arguments":{"x":1}}</tool_call><tool_call>{"name":"f","arguments":{"x":2}}</tool_call>"#).unwrap();
                events.extend(parser.finish().unwrap());
                let mut completion = Completion::default();
                let (mut call, mut stream) = (0, String::new());
                for event in events {
                    if let Some(output) = &mut output {
                        stream.push_str(&output.push(event, server_id, &mut call, &limits).unwrap());
                    } else {
                        let mut index = call;
                        stream.push_str(&chat_delta(event.clone(), request_hash(body.as_bytes()), &mut index));
                        completion.push(event, request_hash(body.as_bytes()), &mut call, &limits).unwrap();
                    }
                }
                assert_eq!(call, 2);
                let json = if let Some(output) = &output { output.json("model", server_id, 123, "stop", Default::default(), None, &echo) }
                    else { completion.json("model", true, server_id, 123, "tool_calls", Default::default(), &echo) };
                for index in 0..2 {
                    let id = format!("call-{}-{index}", request_hash(body.as_bytes()));
                    assert!(json.contains(&id) && stream.contains(&id), "{api}: {json} {stream}");
                }
                (json, stream)
            };
            // Fix only envelope metadata; compare full JSON and tool-event bytes.
            assert_eq!(mock(7), mock(7), "{api}");
            mock(99); // A different admission counter must keep the same call IDs.
            assert_ne!(request_hash(body.as_bytes()), request_hash(format!("{body} ").as_bytes()));
        }
    }
    #[test]
    fn greedy_text_mock_bytes() {
        let limits = Limits::default();
        let fields = r#""temperature":0,"seed":42,"enable_thinking":false"#;
        let chat = format!(r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"hello"}}]}}],{fields}}}"#);
        let responses = format!(r#"{{"input":[{{"role":"user","content":[{{"type":"input_text","text":"hello"}}]}}],{fields}}}"#);
        let messages = format!(r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"hello"}}]}}],"thinking":{{"type":"disabled"}},"max_tokens":4,{fields}}}"#);
        let request = protocol::parse(chat.as_bytes(), true, &limits).unwrap();
        let echo = protocol::SamplingEcho::new(&request, protocol::DraftMode::Plain, 0);
        let usage = generation::Usage { prompt_tokens: 1, completion_tokens: 2, ..Default::default() };
        let content = "answer\né😀";
        let mut completion = Completion::default();
        completion.push(chat_output::Event::Content(content.into()), 7, &mut 0, &limits).unwrap();
        let mut bodies = vec![completion.json("model", true, 7, 123, "stop", usage, &echo),
            sse_chunk("model", true, 7, 123, &chat_delta(chat_output::Event::Content(content.into()), 7, &mut 0), None, None, None, None)
                + &sse_chunk("model", true, 7, 123, "{}", None, Some("stop"), Some(usage), Some(&echo)) + "data: [DONE]\n\n"];
        let (resp_request, resp_output) = crate::responses::parse(responses.as_bytes(), &limits, "model", None).unwrap();
        let (msg_request, msg_output) = crate::anthropic::parse(messages.as_bytes(), &limits, "model", None).unwrap();
        for (request, mut output) in [(resp_request, Output::Responses(resp_output)), (msg_request, Output::Anthropic(msg_output))] {
            assert_eq!(request.sampling, protocol::parse(chat.as_bytes(), true, &limits).unwrap().sampling);
            let mut stream = output.start("model", 7, 123, usage);
            stream.push_str(&output.push(chat_output::Event::Content(content.into()), 7, &mut 0, &limits).unwrap());
            stream.push_str(&output.finish("model", 7, 123, "stop", usage, None, &echo));
            bodies.push(output.json("model", 7, 123, "stop", usage, None, &echo));
            bodies.push(stream);
        }
        for body in bodies {
            assert!(body.contains(&protocol::quote(content)), "{body}");
            println!("GREEDY_WIRE {}", protocol::quote(&body));
        }
    }
    #[test]
    fn client_conformance_errors_and_effort_aliases() {
        let limits = Limits::default();
        for (fields, wording) in [
            (r#""messages":[{"role":"system","content":"retry"}]"#, "Unexpected role \"system\""),
            (r#""messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]"#, "Input tag 'web_search_20250305'"),
        ] {
            let body = format!("{{{fields},\"max_tokens\":1}}");
            let error = crate::anthropic::parse(body.as_bytes(), &limits, "model", None).err().unwrap();
            assert!(error.contains(wording), "{error}");
            assert!(anthropic_error(400, &protocol::error_json("invalid_request_error", &error)).1.contains("invalid_request_error"));
        }
        for effort in ["xhigh", "max", "ultra"] {
            let body = format!(r#"{{"input":"hi","reasoning":{{"effort":"{effort}"}},"temperature":0,"seed":42}}"#);
            let (request, _) = crate::responses::parse(body.as_bytes(), &limits, "model", None).unwrap();
            let Prompt::Chat(input) = &request.prompt else { panic!() };
            assert!(input.enable_thinking);
            assert_eq!(input.reasoning_effort.name(), "xhigh");
            assert_eq!(request.sampling.temperature, 0.0);
        }
        for error in ["input token limit exceeded", "context token limit exceeded", "context length overflow",
            &crate::budget::token_refusal(9, 8, 9, 1, "--input-tokens")] {
            assert!(protocol::error_json("invalid_request_error", error).contains(r#""code":"context_length_exceeded""#));
        }
        assert_eq!(protocol::error_json("invalid_request_error", "max_tokens must be at least 1"),
            r#"{"error":{"type":"invalid_request_error","message":"max_tokens must be at least 1"}}"#);
        assert!(!protocol::error_json("generation_error", "context length overflow").contains("\"code\""));
    }
    #[test] fn sampling_echo_response_shapes() {
        let limits = crate::Limits::default(); assert_eq!(crate::generation::validate_bounds(8, 8, &[], &[1], 0, &limits).unwrap_err(), "max_tokens must be at least 1"); let overflow = crate::Limits { context_tokens: 4, input_tokens: 8, output_tokens: 8, ..crate::Limits::default() }; assert!(crate::generation::validate_bounds(8, 8, &[], &[1, 2], 3, &overflow).unwrap_err().contains("vade --context")); assert!(crate::generation::validate_bounds(8, 8, &[], &[1; 5], 1, &overflow).unwrap_err().contains("and --context;")); for key in ["max_tokens", "max_completion_tokens"] { assert_eq!(crate::protocol::parse(format!(r#"{{"prompt_token_ids":[1],"{key}":0}}"#).as_bytes(), false, &limits).err().unwrap(), "max_tokens must be at least 1"); }
        struct Codec; impl crate::Decoder for Codec { fn push(&mut self, _: u32) -> Result<String, String> { Ok(String::new()) } fn finish(&mut self) -> Result<String, String> { Ok(String::new()) } } impl Tokenizer for Codec { type Decoder<'a> = Self; fn encode(&self, _: &str) -> Result<Vec<u32>, String> { Ok(vec![1; 5]) } fn decoder(&self) -> Self { Self } fn has_token(&self, _: u32) -> bool { true } fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { self.encode("") } }
        for (input_tokens, context_tokens) in [(4, 8), (8, 4)] { for path in ["/v1/messages", "/v1/messages/count_tokens", "/v1/completions", "/v1/chat/completions", "/v1/responses"] { let limits = Limits { input_tokens, context_tokens, output_tokens: 1, ..Limits::default() }; let http = HttpRequest { path: path.into(), body: br#"{"model":"model","max_tokens":1,"messages":[{"role":"user","content":"hi"}],"input":"hi","prompt":"hi"}"#.to_vec(), authorization: None };
        let error = if path.ends_with("count_tokens") { count_tokens(&http, 8, 8, Some(&Codec), &limits, "model").err().unwrap() } else { prepare_request(&http, 8, 8, &[], Some(&Codec), &limits, "model").err().unwrap() }; if anthropic_route(path) { assert_eq!(anthropic_error(400, &protocol::error_json("invalid_request_error", &error)), (400, r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}"#.into())); }
        else { assert_eq!(error, format!("Requested size does not fit.\nAsked: input=5; OUTPUT/max_tokens={}; configured token ceiling=4; counts must be positive; total {} tokens; weights/rank and per-session bytes are not part of this token check.\nAvailable: 4 tokens from {}.\nGap: {} tokens; GiB gap is not applicable to a token ceiling.\nNext: Lower request prompt length or max_tokens (vade --context, or server flags: --input-tokens / --context-tokens / --output-tokens). A GPU budget override or more Sparks cannot raise this token ceiling.", usize::from(input_tokens == 4), if input_tokens == 4 { 5 } else { 6 }, if input_tokens == 4 { "--input-tokens" } else { "the lower of engine context and --context; one output token needs room" }, if input_tokens == 4 { 1 } else { 2 })); assert!(input_tokens == 4 || error.contains("vade --context"), "{error}"); } } }
        let limits = Limits { default_sampling: Some(spark_engine::sample::Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, ..spark_engine::sample::Sampling::greedy() }), ..Limits::default() };
        for chat in [false, true] { for fields in ["\"seed\":42", "\"seed\":42,\"temperature\":0,\"top_k\":0,\"top_p\":1,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0"] {
            let input = if chat { "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]" } else { "\"prompt_token_ids\":[1]" };
            let request = protocol::parse(format!("{{{input},{fields}}}").as_bytes(), chat, &limits).unwrap(); let echo = protocol::SamplingEcho::new(&request, protocol::DraftMode::Plain, 0); let usage = generation::Usage::default();
            let expected = engine_model::json::parse(echo.json().as_bytes(), Default::default()).unwrap(); assert_eq!(request.sampling.temperature, if fields.contains("temperature") { 0.0 } else { 1.0 });
            let mut bodies = vec![Completion::default().json("model", chat, 1, 2, "length", usage, &echo), sse_chunk("model", chat, 1, 2, "{}", None, Some("length"), Some(usage), Some(&echo))];
            for mut output in [Output::Anthropic(Default::default()), Output::Responses(Default::default())] { bodies.push(output.json("model", 1, 2, "length", usage, None, &echo)); bodies.push(output.finish("model", 1, 2, "length", usage, None, &echo)); }
            for body in bodies { for data in body.lines().filter_map(|s| s.strip_prefix("data: ").or_else(|| s.starts_with('{').then_some(s))) { if data == "[DONE]" { continue; } let value = engine_model::json::parse(data.as_bytes(), Default::default()).unwrap(); assert_eq!(value.object().unwrap().get("vade"), Some(&expected)); } }
        } }
    }
}
