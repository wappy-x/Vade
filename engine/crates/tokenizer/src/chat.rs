//! Qwen3 chat formatting authored from the official tokenizer configuration.
//! Strings are text-only; multimodal content is not silently discarded.
pub mod gemma4;
pub(crate) mod llama;
pub(crate) mod glm;
use crate::{Result, err};
use engine_model::json;
#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;
/// Semantics of one Qwen3.5/3.6/3.8 hybrid template, read from its clauses.
/// The template is data: model names and tensor dimensions do not select it.
/// Unknown bytes fail closed, including edits that retain familiar markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HybridTemplate {
    /// Reasoning-effort instructions lead the system turn while thinking.
    effort: bool,
    /// Assistant content without reasoning_content yields its inline thought.
    split_inline: bool,
    history: History,
    /// Scalar tool arguments render as Python strings rather than JSON.
    python_scalars: bool,
    /// The generation prompt when the request leaves enable_thinking unset.
    pub(crate) default_thinking: bool,
}
/// Which assistant turns before the last user query keep their thought.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum History { None, OptIn, Always }
/// Each family variation point lists its official alternatives. A template is
/// one alternative per point plus a skeleton whose SHA-256 is pinned below.
const CLAUSES: [&[&str]; 7] = [
    &["provided.') }}\n{%- endif %}\n{%- if tools and tools is iterable and tools is not mapping %}\n    {{- '<|im_start|>system\\n' }}\n    {{- \"# Tools", "provided.') }}\n{%- endif %}\n{%- set reasoning_instructions = '' %}\n{%- if enable_thinking is undefined or enable_thinking is true %}\n    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\n    {%- endif %}\n    {%- if resolved_reasoning_effort == 'xhigh' %}\n        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}\n    {%- elif resolved_reasoning_effort == 'low' %}\n        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}\n    {%- endif %}\n{%- endif %}\n{%- if tools and tools is iterable and tools is not mapping %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if reasoning_instructions %}\n        {{- reasoning_instructions + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools"],
    &["        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n    {%- endif %}\n", "        {%- if content %}\n            {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n' }}\n        {%- elif reasoning_instructions %}\n            {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- elif reasoning_instructions %}\n        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n    {%- endif %}\n"],
    &["{%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n", "{%- set reasoning_content = message.reasoning_content %}\n        {%- endif %}\n"],
    &["{%- if loop.index0 > ns.last_query_index %}", "{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}", "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}"],
    &["{%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}", "{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}"],
    // Empty argument strings render no parameters under either alternative.
    &["{%- if tool_call.arguments is defined %}", "{%- if tool_call.arguments is defined and tool_call.arguments != '' %}"],
    &["enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n        {{- '<think>\\n' }}", "enable_thinking is defined and enable_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}"],
];
const SKELETON: &str = "9912d664004eecfa6a594d735c49173488ea9b87907a4cf8d0627a1d6e8df738";
impl HybridTemplate {
    /// Qwen3.8 semantics, also used before a directory template is read.
    pub(crate) const EFFORT: Self = Self { effort: true, split_inline: false, history: History::Always, python_scalars: false, default_thinking: true };
    pub(crate) const PRESERVE: Self = Self { effort: false, split_inline: true, history: History::OptIn, python_scalars: false, default_thinking: true };
    fn recognize(template: &str) -> Result<Self> {
        let (mut rest, mut choice) = (template.to_owned(), [0; CLAUSES.len()]);
        for (k, alternatives) in CLAUSES.iter().enumerate() {
            let hits: Vec<usize> = (0..alternatives.len()).filter(|&i| rest.matches(alternatives[i]).count() == 1).collect();
            let [hit] = hits[..] else { return Err(err("unsupported chat template configuration")) };
            rest = rest.replacen(alternatives[hit], &format!("\0{k}"), 1);
            choice[k] = hit;
        }
        // Effort instructions enter both system-turn forms or neither.
        if sha256::digest(rest.as_bytes()) != SKELETON || choice[0] != choice[1] {
            return Err(err("unsupported chat template configuration"));
        }
        Ok(Self { effort: choice[0] == 1, split_inline: choice[2] == 0, history: [History::None, History::OptIn, History::Always][choice[3]],
            python_scalars: choice[4] == 0, default_thinking: choice[6] == 0 })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}
impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}
impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    #[default]
    XHigh,
}
impl ReasoningEffort {
    pub fn name(&self) -> &'static str {
        match self { Self::Low => "low", Self::Medium => "medium", Self::XHigh => "xhigh" }
    }
    fn instructions(&self) -> &'static str {
        match self {
            Self::Low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            Self::Medium => "",
            Self::XHigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        }
    }
}
#[derive(Debug, Clone)]
pub struct Options {
    pub enable_thinking: bool,
    pub reasoning_effort: ReasoningEffort,
    pub preserve_thinking: bool,
    pub add_generation_prompt: bool,
    pub tools: Vec<String>,
    pub max_bytes: usize,
    pub generation_prefix: String,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            reasoning_effort: ReasoningEffort::default(),
            preserve_thinking: false,
            add_generation_prompt: true,
            tools: Vec::new(),
            max_bytes: 1024 * 1024,
            generation_prefix: String::new(),
        }
    }
}
struct Output {
    text: String,
    limit: usize,
    ordinary: Vec<std::ops::Range<usize>>,
}
impl Output {
    fn data(&mut self, s: &str) -> Result<()> {
        let start = self.text.len();
        self.push(s)?;
        self.ordinary.push(start..self.text.len());
        Ok(())
    }
    fn push(&mut self, s: &str) -> Result<()> {
        if self
            .text
            .len()
            .checked_add(s.len())
            .is_none_or(|n| n > self.limit)
        {
            return Err(err("chat template byte limit exceeded"));
        }
        self.text.push_str(s);
        Ok(())
    }
}
/// Verify that chat configuration matches the supported official template.
/// The configuration's model_max_length is deliberately not an engine capacity.
pub fn validate_config_file(path: impl AsRef<std::path::Path>) -> Result<()> {
    validate_config(path, false).map(|_| ())
}
// Return the Phi family and independently recognized hybrid template semantics.
pub(crate) fn validate_config(path: impl AsRef<std::path::Path>, hybrid: bool) -> Result<(bool, HybridTemplate)> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(2 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    let v = json::parse(
        &bytes,
        json::Limits {
            bytes: 2 * 1024 * 1024,
            ..Default::default()
        },
    )?;
    let o = v.object()?;
    let template = json::field(o, "chat_template")?.string()?;
    if template == include_str!("../templates/phi4.jinja") {
        for (key, expected) in [("bos_token", "<|endoftext|>"), ("eos_token", "<|im_end|>"),
            ("pad_token", "<|dummy_85|>"), ("tokenizer_class", "GPT2Tokenizer")] {
            if json::field(o, key)?.string()? != expected { return Err(err("Phi chat configuration mismatch")); }
        }
        if json::field(o, "add_prefix_space")?.boolean()? || json::field(o, "clean_up_tokenization_spaces")?.boolean()? {
            return Err(err("unsupported Phi tokenizer flags"));
        }
        return Ok((true, HybridTemplate::PRESERVE));
    }
    let semantics = if hybrid {
        HybridTemplate::recognize(template)?
    } else {
        if template != include_str!("../templates/qwen3.jinja") {
            return Err(err("unsupported chat template configuration"));
        }
        HybridTemplate::PRESERVE
    };
    for key in [
        "add_bos_token",
        "add_prefix_space",
        "clean_up_tokenization_spaces",
        "split_special_tokens",
    ] {
        if json::field(o, key)?.boolean()? {
            return Err(err("unsupported chat tokenizer flags"));
        }
    }
    if json::field(o, "eos_token")?.string()? != "<|im_end|>"
        || json::field(o, "pad_token")?.string()? != "<|endoftext|>"
    {
        return Err(err("unsupported chat special-token configuration"));
    }
    Ok((false, semantics))
}
pub fn format(messages: &[Message], options: &Options) -> Result<String> {
    Ok(render(messages, options)?.text)
}
pub(crate) fn format_hybrid(messages: &[Message], options: &Options, template: HybridTemplate) -> Result<String> {
    Ok(render_hybrid(messages, options, template)?.text)
}
/// Encode template controls while treating every caller-supplied field as text.
/// Keeping byte ranges preserves BPE merges across ordinary template boundaries.
pub fn encode(
    tokenizer: &crate::Tokenizer,
    messages: &[Message],
    options: &Options,
) -> Result<Vec<u32>> {
    tokenizer.check_chat_template()?;
    if !options.generation_prefix.is_empty() && (tokenizer.glm || tokenizer.llama || tokenizer.phi || tokenizer.is_gemma4()) {
        return Err(err("forced tool_choice requires a Qwen tool template"));
    }
    if tokenizer.glm {
        let output = glm::render(messages, options)?;
        return tokenizer.encode_protected(&output.text, &output.ordinary);
    }
    if tokenizer.llama {
        let output = llama::render(messages, options, tokenizer.llama_current_date)?;
        return tokenizer.encode_protected(&output.text, &output.ordinary);
    }
    if tokenizer.phi { return tokenizer.encode(&format_phi(messages, options)?); }
    if tokenizer.is_gemma4() {
        return tokenizer.encode(&gemma4::format(messages, options)?);
    }
    let output = if tokenizer.is_hybrid_text() { render_hybrid(messages, options, tokenizer.hybrid_template)? } else { render(messages, options)? };
    tokenizer.encode_protected(&output.text, &output.ordinary)
}
// One renderer for the recognized hybrid template semantics, including XML tools.
fn render_hybrid(messages: &[Message], options: &Options, template: HybridTemplate) -> Result<Output> {
    let reasoning = template.effort;
    if messages.is_empty() || messages.len() > 4096 || options.tools.len() > 1024 {
        return Err(err("chat item count limit exceeded"));
    }
    let last_query = messages.iter().rposition(|m| {
        let c = py_trim(&m.content);
        m.role == Role::User && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
    }).ok_or_else(|| err("hybrid chat requires a user query"))?;
    let mut out = Output { text: String::new(), limit: options.max_bytes, ordinary: Vec::new() };
    let instructions = if reasoning && options.enable_thinking { options.reasoning_effort.instructions() } else { "" };
    // Developer is a system alias. Keep the leading block/cache prefix unchanged;
    // later instructions render as individual system turns in their original positions.
    let lead = messages.iter().take_while(|m| matches!(m.role, Role::System | Role::Developer)).count();
    let joined = messages[..lead].iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n\n");
    let system = (lead > 0).then_some(py_trim(&joined));
    if !options.tools.is_empty() {
        out.push("<|im_start|>system\n")?;
        if !instructions.is_empty() { out.push(instructions)?; out.push("\n\n")?; }
        out.push("# Tools\n\nYou have access to the following functions:\n\n<tools>")?;
        for tool in &options.tools { out.push("\n")?; out.data(&json_for_template_limit(tool, options.max_bytes)?)?; }
        out.push("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")?;
        if let Some(system) = system.filter(|s| !s.is_empty()) { out.push("\n\n")?; out.data(system)?; }
        out.push("<|im_end|>\n")?;
    } else if let Some(system) = system.filter(|s| !s.is_empty() || !reasoning) {
        out.push("<|im_start|>system\n")?;
        if !instructions.is_empty() { out.push(instructions)?; out.push("\n\n")?; }
        out.data(system)?; out.push("<|im_end|>\n")?;
    } else if !instructions.is_empty() {
        out.push("<|im_start|>system\n")?; out.push(instructions)?; out.push("<|im_end|>\n")?;
    }
    for (i, message) in messages.iter().enumerate() {
        let mut content = py_trim(&message.content);
        match message.role {
            Role::System | Role::Developer if i < lead => {}
            Role::System | Role::Developer => {
                out.push("<|im_start|>system\n")?; out.data(content)?; out.push("<|im_end|>\n")?;
            }
            Role::User => { out.push("<|im_start|>user\n")?; out.data(content)?; out.push("<|im_end|>\n")?; }
            Role::Assistant => {
                let mut thought = message.reasoning_content.as_deref().unwrap_or("");
                if template.split_inline && message.reasoning_content.is_none() && content.contains("</think>") {
                    thought = content.split("</think>").next().unwrap().trim_end_matches('\n').rsplit("<think>").next().unwrap().trim_start_matches('\n');
                    content = content.rsplit("</think>").next().unwrap().trim_start_matches('\n');
                }
                out.push("<|im_start|>assistant\n")?;
                if template.history == History::Always || (template.history == History::OptIn && options.preserve_thinking) || i > last_query {
                    out.push("<think>\n")?; out.data(py_trim(thought))?; out.push("\n</think>\n\n")?;
                }
                out.data(content)?;
                for (j, call) in message.tool_calls.iter().enumerate() {
                    out.push(if j > 0 { "\n<tool_call>\n<function=" } else if py_trim(content).is_empty() { "<tool_call>\n<function=" } else { "\n\n<tool_call>\n<function=" })?;
                    out.data(&call.name)?; out.push(">\n")?;
                    if !call.arguments.is_empty() {
                        for (name, value) in template_arguments(&call.arguments, options.max_bytes, template.python_scalars)? {
                            out.push("<parameter=")?; out.data(&name)?; out.push(">\n")?;
                            out.data(&value)?; out.push("\n</parameter>\n")?;
                        }
                    }
                    out.push("</function>\n</tool_call>")?;
                }
                out.push("<|im_end|>\n")?;
            }
            Role::Tool => {
                if i > 0 && messages[i - 1].role != Role::Tool { out.push("<|im_start|>user")?; }
                out.push("\n<tool_response>\n")?; out.data(content)?; out.push("\n</tool_response>")?;
                if messages.get(i + 1).is_none_or(|m| m.role != Role::Tool) { out.push("<|im_end|>\n")?; }
            }
        }
    }
    if options.add_generation_prompt {
        out.push("<|im_start|>assistant\n")?;
        out.push(if options.enable_thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" })?;
        out.push(&options.generation_prefix)?;
    }
    Ok(out)
}
/// Jinja `trim` is Python `str.strip()`, which also strips U+001C..U+001F.
fn py_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}
/// Ordered `arguments|items`: raw strings, JSON containers, and variant-specific scalars.
fn template_arguments(arguments: &str, max_bytes: usize, scalar_strings: bool) -> Result<Vec<(String, String)>> {
    let text = json_for_template_limit(arguments, max_bytes)?;
    let b = text.as_bytes();
    if b.first() != Some(&b'{') { return Err(err("tool arguments must be a JSON object")); }
    let (mut members, mut i) = (Vec::new(), 1);
    while b[i] != b'}' {
        let key = i;
        i += 1;
        while b[i] != b'"' { i += if b[i] == b'\\' { 2 } else { 1 }; }
        let name = json::parse(&b[key..=i], json::Limits { string_bytes: text.len(), ..Default::default() })?.string()?.to_owned();
        i += 3;
        let start = i;
        let (mut depth, mut quoted) = (0usize, false);
        while quoted || depth > 0 || !matches!(b[i], b',' | b'}') {
            match b[i] {
                b'\\' if quoted => i += 1,
                b'"' => quoted = !quoted,
                b'{' | b'[' if !quoted => depth += 1,
                b'}' | b']' if !quoted => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        let raw = &text[start..i];
        let value = if raw.starts_with('"') {
            json::parse(raw.as_bytes(), json::Limits { string_bytes: text.len(), ..Default::default() })?.string()?.to_owned()
        } else if scalar_strings {
            match raw { "true" => "True", "false" => "False", "null" => "None", _ => raw }.to_owned()
        } else { raw.to_owned() };
        members.push((name, value));
        if b[i] == b',' { i += 2; }
    }
    Ok(members)
}
/// Direct rendering of the artifact's Phi chat template, without BOS or whitespace insertion.
pub fn format_phi(messages: &[Message], options: &Options) -> Result<String> {
    if messages.len() > 4096 { return Err(err("chat item count limit exceeded")); }
    let mut out = Output { text: String::new(), limit: options.max_bytes, ordinary: Vec::new() };
    for m in messages {
        if m.role == Role::Tool { continue; }
        out.push("<|im_start|>")?;
        out.push(m.role.name())?;
        out.push("<|im_sep|>")?;
        out.push(&m.content)?;
        out.push("<|im_end|>")?;
    }
    if options.add_generation_prompt { out.push("<|im_start|>assistant<|im_sep|>")?; }
    Ok(out.text)
}
fn render(messages: &[Message], options: &Options) -> Result<Output> {
    if messages.is_empty() {
        return Err(err("chat requires at least one message"));
    }
    if messages.len() > 4096 || options.tools.len() > 1024 {
        return Err(err("chat item count limit exceeded"));
    }
    let mut out = Output {
        text: String::new(),
        limit: options.max_bytes,
        ordinary: Vec::new(),
    };
    // Bound all input before intermediate normalization or reasoning extraction.
    let mut total = 0usize;
    for m in messages {
        for s in std::iter::once(m.content.as_str())
            .chain(m.reasoning_content.as_deref())
            .chain(
                m.tool_calls
                    .iter()
                    .flat_map(|t| [t.name.as_str(), t.arguments.as_str()]),
            )
        {
            total = total
                .checked_add(s.len())
                .ok_or_else(|| err("chat input overflow"))?
        }
        if m.tool_calls.len() > 1024 {
            return Err(err("tool-call count limit exceeded"));
        }
        if m.role != Role::Assistant && (!m.tool_calls.is_empty() || m.reasoning_content.is_some())
        {
            return Err(err("reasoning and tool calls require assistant role"));
        }
    }
    for t in &options.tools {
        total = total
            .checked_add(t.len())
            .ok_or_else(|| err("chat input overflow"))?
    }
    if total > options.max_bytes {
        return Err(err("chat input byte limit exceeded"));
    }
    if !options.tools.is_empty() {
        out.push("<|im_start|>system\n")?;
        if matches!(messages[0].role, Role::System | Role::Developer) {
            out.data(&messages[0].content)?;
            out.push("\n\n")?
        }
        out.push("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")?;
        for tool in &options.tools {
            out.push("\n")?;
            out.data(&json_for_template(tool)?)?
        }
        out.push("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")?;
    } else if matches!(messages[0].role, Role::System | Role::Developer) {
        out.push("<|im_start|>system\n")?;
        out.data(&messages[0].content)?;
        out.push("<|im_end|>\n")?
    }
    let last_query = messages
        .iter()
        .rposition(|m| {
            m.role == Role::User
                && !(m.content.starts_with("<tool_response>")
                    && m.content.ends_with("</tool_response>"))
        })
        .unwrap_or(messages.len() - 1);
    for (i, m) in messages.iter().enumerate() {
        match m.role {
            Role::System | Role::Developer if i == 0 => {}
            Role::System | Role::Developer | Role::User => {
                out.push("<|im_start|>")?;
                out.push(if m.role == Role::Developer { "system" } else { m.role.name() })?;
                out.push("\n")?;
                out.data(&m.content)?;
                out.push("<|im_end|>\n")?
            }
            Role::Assistant => {
                let mut content = m.content.as_str();
                let mut reasoning = m.reasoning_content.as_deref().unwrap_or("");
                if m.reasoning_content.is_none() && content.contains("</think>") {
                    reasoning = content
                        .split("</think>")
                        .next()
                        .unwrap()
                        .trim_end_matches('\n')
                        .rsplit("<think>")
                        .next()
                        .unwrap()
                        .trim_start_matches('\n');
                    content = content
                        .rsplit("</think>")
                        .next()
                        .unwrap()
                        .trim_start_matches('\n')
                }
                out.push("<|im_start|>assistant\n")?;
                if i > last_query && (i + 1 == messages.len() || !reasoning.is_empty()) {
                    out.push("<think>\n")?;
                    out.data(reasoning.trim_matches('\n'))?;
                    out.push("\n</think>\n\n")?;
                    out.data(content.trim_start_matches('\n'))?
                } else {
                    out.data(content)?
                }
                for (j, call) in m.tool_calls.iter().enumerate() {
                    if call.name.is_empty()
                        || call
                            .name
                            .chars()
                            .any(|c| c == '"' || c == '\\' || c.is_control())
                    {
                        return Err(err("tool name contains unsupported JSON characters"));
                    }
                    json::parse(
                        call.arguments.as_bytes(),
                        json::Limits {
                            bytes: options.max_bytes,
                            string_bytes: options.max_bytes,
                            ..Default::default()
                        },
                    )?;
                    if j > 0 || !content.is_empty() {
                        out.push("\n")?
                    }
                    out.push("<tool_call>\n{\"name\": \"")?;
                    out.data(&call.name)?;
                    out.push("\", \"arguments\": ")?;
                    out.data(&call.arguments)?;
                    out.push("}\n</tool_call>")?;
                }
                out.push("<|im_end|>\n")?;
            }
            Role::Tool => {
                if i == 0 || messages[i - 1].role != Role::Tool {
                    out.push("<|im_start|>user")?
                }
                out.push("\n<tool_response>\n")?;
                out.data(&m.content)?;
                out.push("\n</tool_response>")?;
                if i + 1 == messages.len() || messages[i + 1].role != Role::Tool {
                    out.push("<|im_end|>\n")?
                }
            }
        }
    }
    if options.add_generation_prompt {
        out.push("<|im_start|>assistant\n")?;
        if !options.enable_thinking {
            out.push("<think>\n\n</think>\n\n")?
        }
        out.push(&options.generation_prefix)?;
    }
    Ok(out)
}
/// Render JSON in the official template's `tojson` style, retaining key order.
/// Duplicate keys are rejected by the owned JSON parser. No HTML escaping occurs.
pub fn json_for_template(input: &str) -> Result<String> {
    json_for_template_limit(input, input.len())
}
fn json_for_template_limit(input: &str, max_bytes: usize) -> Result<String> {
    json::parse(input.as_bytes(), json::Limits { string_bytes: max_bytes.max(json::Limits::default().string_bytes), ..Default::default() })?;
    let b = input.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b' ' | b'\n' | b'\r' | b'\t' => i += 1,
            b',' => {
                out.push_str(", ");
                i += 1
            }
            b':' => {
                out.push_str(": ");
                i += 1
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' {
                        i += 2
                    } else if b[i] == b'"' {
                        i += 1;
                        break;
                    } else {
                        i += 1
                    }
                }
                let v = json::parse(&b[start..i], json::Limits { string_bytes: input.len(), ..Default::default() })?;
                out.push_str(&quote(v.string()?));
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                while i < b.len()
                    && (b[i].is_ascii_digit() || matches!(b[i], b'-' | b'+' | b'.' | b'e' | b'E'))
                {
                    i += 1
                }
                let raw = &input[start..i];
                if raw.contains(['.', 'e', 'E']) {
                    let n: f64 = raw
                        .parse()
                        .map_err(|_| err("invalid template JSON number"))?;
                    if !n.is_finite() {
                        return Err(err("nonfinite template JSON number"));
                    }
                    let s = format!("{n:?}");
                    if let Some((mantissa, exponent)) = s.split_once('e') {
                        let exponent: i32 =
                            exponent.parse().map_err(|_| err("invalid exponent"))?;
                        out.push_str(mantissa);
                        out.push('e');
                        out.push(if exponent < 0 { '-' } else { '+' });
                        out.push_str(&format!("{:02}", exponent.unsigned_abs()));
                    } else {
                        out.push_str(&s)
                    }
                } else {
                    out.push_str(if raw == "-0" { "0" } else { raw })
                }
            }
            _ => {
                out.push(b[i] as char);
                i += 1
            }
        }
    }
    Ok(out)
}
pub fn quote(input: &str) -> String {
    let mut out = String::from("\"");
    for c in input.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 32 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn hybrid_official_goldens() {
        let mut reply = Message::new(Role::Assistant, " done "); reply.reasoning_content = Some(" now ".into()); reply.tool_calls.push(ToolCall { id: None, name: "f".into(), arguments: r#"{"yes":true,"no":false,"nil":null,"int":-7,"float":1e-7,"text":"é <&>\nline","list":[true,null,2],"map":{"z":false,"a":"雪"}}"#.into() }); let mut old = Message::new(Role::Assistant, "past"); old.reasoning_content = Some(" old ".into()); let messages = [Message::new(Role::User, "first"), old, Message::new(Role::User, " next "), reply]; let goldens = json::parse(include_bytes!("../tests/hybrid-goldens.json"), Default::default()).unwrap();
        for (family, template) in official_templates().into_iter().enumerate() { for bits in 0..8 { let tools = bits & 1 != 0; let options = Options { enable_thinking: bits & 4 != 0, preserve_thinking: bits & 2 != 0, add_generation_prompt: !tools, generation_prefix: if tools { "" } else { "<tool_call>\n<function=f" }.into(), tools: if tools { vec![r#"{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}"#.into()] } else { vec![] }, ..Default::default() }; assert_eq!(render_hybrid(&messages, &options, template).unwrap().text, goldens.array().unwrap()[family].array().unwrap()[bits].string().unwrap(), "family={family} bits={bits}"); } } }
    /// Official Qwen3.8, 3.6, 3.5 and small-3.5 templates, rebuilt from the
    /// Qwen3.6 file by clause choice and checked against their published hashes.
    fn official_texts() -> [String; 4] {
        let swap = |choices: [(usize, usize); 7]| choices.iter().enumerate().fold(include_str!("../tests/qwen3.6.jinja").to_owned(), |t, (k, &(from, to))| t.replacen(CLAUSES[k][from], CLAUSES[k][to], 1));
        let texts = [swap([(0, 1), (0, 1), (0, 1), (1, 2), (1, 1), (0, 1), (0, 0)]), swap([(0, 0); 7]),
            swap([(0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0)]), swap([(0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 1)])];
        let hashes = ["c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"];
        for (t, h) in texts.iter().zip(hashes) { assert_eq!(sha256::digest(t.as_bytes()), h); }
        texts
    }
    fn official_templates() -> [HybridTemplate; 4] { official_texts().map(|t| HybridTemplate::recognize(&t).unwrap()) }
    #[test] fn hybrid_family_clauses() {
        let [flash, q36, q35, small] = official_templates();
        assert_eq!((flash, q36), (HybridTemplate::EFFORT, HybridTemplate::PRESERVE));
        assert_eq!(q35, HybridTemplate { history: History::None, python_scalars: true, ..HybridTemplate::PRESERVE });
        assert_eq!(small, HybridTemplate { default_thinking: false, ..q35 });
        let official = &official_texts()[3];
        for edited in [official.replacen("trim", "trim ", 1), official.replacen("{%- if loop.index0", "{%- if loop.index0 > 0 or loop.index0", 1),
            official.replacen(CLAUSES[0][0], CLAUSES[0][1], 1), format!("{official}{}", CLAUSES[5][0])] {
            assert!(HybridTemplate::recognize(&edited).is_err());
        }
    }
    #[test] fn hybrid_official_turn_goldens() {
        let call = |name: &str, arguments: &str| ToolCall { id: None, name: name.into(), arguments: arguments.into() };
        let mut first = Message::new(Role::Assistant, "<think>\nlook first\n</think>\n\nchecking");
        first.tool_calls = vec![call("ls", r#"{"path": "."}"#), call("cat", r#"{"path": "a.txt", "n": 3, "all": false}"#)];
        let mut summary = Message::new(Role::Assistant, "Two files."); summary.reasoning_content = Some("got it".into());
        let mut step = Message::new(Role::Assistant, ""); step.reasoning_content = Some("plan".into()); step.tool_calls = vec![call("ls", r#"{"path": "src"}"#)];
        let conversations = [vec![Message::new(Role::System, "  Be brief.\n"), Message::new(Role::User, "list files"), first, Message::new(Role::Tool, "a.txt\nb.txt"),
            Message::new(Role::Tool, "hello"), summary, Message::new(Role::User, "thanks")],
            vec![Message::new(Role::System, "You are a coder."), Message::new(Role::User, "run it"), step, Message::new(Role::Tool, "main.rs")]];
        let tools = vec![r#"{"type": "function", "function": {"name": "ls", "description": "List a folder", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#.to_owned(),
            r#"{"type": "function", "function": {"name": "cat", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}, "all": {"type": "boolean"}}}}}"#.to_owned()];
        let goldens = json::parse(include_bytes!("../tests/hybrid-turn-goldens.json"), Default::default()).unwrap();
        for (c, messages) in conversations.iter().enumerate() { for (family, template) in official_templates().into_iter().enumerate() { for bits in 0..8 {
            let options = Options { enable_thinking: bits & 4 != 0, preserve_thinking: bits & 2 != 0, tools: if bits & 1 != 0 { tools.clone() } else { vec![] }, ..Default::default() };
            let expected = goldens.array().unwrap()[c].array().unwrap()[family].array().unwrap()[bits].string().unwrap();
            assert_eq!(render_hybrid(messages, &options, template).unwrap().text, expected, "conversation={c} family={family} bits={bits}");
        } } }
    }
    #[test]
    fn old_reasoning_hidden() {
        let mut old = Message::new(Role::Assistant, "<think>\nsecret\n</think>\n\nanswer");
        old.tool_calls.clear();
        assert!(
            format(
                &[old, Message::new(Role::User, "next")],
                &Options::default()
            )
            .unwrap()
            .starts_with("<|im_start|>assistant\nanswer<|im_end|>\n")
        );
    }
    #[test]
    fn ordered_json() {
        assert_eq!(
            json_for_template(r#"{ "z": [1, "\u00e9"], "a":true }"#).unwrap(),
            "{\"z\": [1, \"é\"], \"a\": true}"
        );
    }
    #[test]
    fn bounded() {
        assert!(
            format(
                &[Message::new(Role::User, "abc")],
                &Options {
                    max_bytes: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    #[test]
    fn gemma_official_thinking_and_tool_fixture() {
        let mut reply = Message::new(Role::Assistant, "");
        reply.reasoning_content = Some("plan".into());
        reply.tool_calls.push(ToolCall { name: "f".into(), id: None, arguments: r##"{"a": {"z": 1, "Z": 2}, "A": 2, "query": "é", "URL": "x", "_id": true}"##.into() });
        let messages = [Message::new(Role::User, "go"), reply];
        let mut options = Options { enable_thinking: false, tools: vec![r##"{"function": {"name": "f", "description": "find", "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "A": {"type": "string"}, "URL": {"type": "string"}, "query": {"type": "string"}, "_id": {"type": "boolean"}}}}}"##.into()], ..Default::default() };
        let expected = "<bos><|turn>system\n<|tool>declaration:f{description:<|\"|>find<|\"|>,parameters:{properties:{_id:{type:<|\"|>BOOLEAN<|\"|>},a:{type:<|\"|>STRING<|\"|>},A:{type:<|\"|>STRING<|\"|>},query:{type:<|\"|>STRING<|\"|>},URL:{type:<|\"|>STRING<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\ngo<turn|>\n<|turn>model\n<|channel>thought\nplan\n<channel|><|tool_call>call:f{_id:true,a:{z:1,Z:2},A:2,query:<|\"|>é<|\"|>,URL:<|\"|>x<|\"|>}<tool_call|><|tool_response>";
        assert_eq!(gemma4::format(&messages, &options).unwrap(), expected);
        options.enable_thinking = true;
        assert_eq!(gemma4::format(&messages, &options).unwrap(), expected.replacen("<|turn>system\n", "<|turn>system\n<|think|>\n", 1));
    }
}
