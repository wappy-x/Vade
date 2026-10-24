//! Text branch of Google's canonical Gemma 4 template, authored in Rust.
//! The pinned Jinja artifact validates the model folder; no Jinja runtime is linked.
use super::{Message, Options, Output, Role};
use crate::{Result, err};
use engine_model::{
    json::{self, field},
    safetensors,
};
use std::path::Path;
use self::ordered::{Object, Value};
mod ordered;

pub fn validate_directory(path: &Path) -> Result<()> {
    let bytes = safetensors::read_bounded(&path.join("tokenizer_config.json"), 2 * 1024 * 1024)?;
    let doc = json::parse(
        &bytes,
        json::Limits {
            bytes: 2 * 1024 * 1024,
            ..Default::default()
        },
    )?;
    let o = doc.object()?;
    for (key, value) in [
        ("tokenizer_class", "GemmaTokenizer"),
        ("bos_token", "<bos>"),
        ("eos_token", "<eos>"),
        ("pad_token", "<pad>"),
        ("sot_token", "<|turn>"),
        ("eot_token", "<turn|>"),
    ] {
        if field(o, key)?.string()? != value {
            return Err(err("unsupported Gemma chat configuration"));
        }
    }
    for key in [
        "add_bos_token",
        "add_eos_token",
        "add_prefix_space",
        "clean_up_tokenization_spaces",
        "split_special_tokens",
    ] {
        if o.get(key).is_some_and(|v| v != &json::Value::Bool(false)) {
            return Err(err("unsupported Gemma tokenizer flags"));
        }
    }
    let template = safetensors::read_bounded(&path.join("chat_template.jinja"), 128 * 1024)?;
    if template != include_bytes!("../../templates/gemma4.jinja") {
        return Err(err("unsupported Gemma 4 chat template"));
    }
    Ok(())
}
fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(s) => s.parse::<f64>().is_ok_and(|n| n != 0.0),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
fn get<'a>(o: &'a Object, key: &str) -> &'a Value {
    o.get(key).unwrap_or(&Value::Null)
}
fn text(o: &Object, key: &str) -> Result<String> {
    match get(o, key) {
        Value::Null => Ok(String::new()),
        v => Ok(v.string()?.into()),
    }
}
fn argument(v: &Value, escape_keys: bool, out: &mut Output) -> Result<()> {
    match v {
        Value::Null => out.push("null"),
        Value::Bool(b) => out.push(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push(&super::json_for_template(n)?),
        Value::String(s) => {
            out.push("<|\"|>")?;
            out.push(s)?;
            out.push("<|\"|>")
        }
        Value::Array(a) => {
            out.push("[")?;
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(",")?;
                }
                argument(v, escape_keys, out)?;
            }
            out.push("]")
        }
        Value::Object(o) => {
            out.push("{")?;
            for (i, (k, v)) in o.sorted().into_iter().enumerate() {
                if i > 0 {
                    out.push(",")?;
                }
                if escape_keys {
                    argument(&Value::String(k.clone()), true, out)?;
                } else {
                    out.push(k)?;
                }
                out.push(":")?;
                argument(v, escape_keys, out)?;
            }
            out.push("}")
        }
    }
}
fn required(v: &Value, out: &mut Output) -> Result<()> {
    argument(v, true, out)
}
fn upper(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(s.to_uppercase()),
        Value::Array(a) => Value::Array(a.iter().map(upper).collect()),
        _ => v.clone(),
    }
}
fn parameters(properties: &Object, filter: bool, out: &mut Output) -> Result<()> {
    let mut first = true;
    for (key, value) in properties.sorted() {
        if filter
            && ["description", "type", "properties", "required", "nullable"].contains(&key.as_str())
        {
            continue;
        }
        if !first {
            out.push(",")?;
        }
        first = false;
        out.push(key)?;
        out.push(":{")?;
        let value = value.object()?;
        let mut comma = false;
        if truth(get(value, "description")) {
            out.push("description:")?;
            argument(get(value, "description"), true, out)?;
            comma = true;
        }
        let kind = text(value, "type")?.to_uppercase();
        if kind == "STRING" && truth(get(value, "enum")) {
            if comma {
                out.push(",")?;
            }
            comma = true;
            out.push("enum:")?;
            argument(get(value, "enum"), true, out)?;
        } else if kind == "ARRAY" {
            if let Value::Object(items) = get(value, "items") {
                if !items.is_empty() {
                    if comma {
                        out.push(",")?;
                    }
                    comma = true;
                    out.push("items:{")?;
                    let mut item_first = true;
                    for (k, v) in items.sorted() {
                        if v == &Value::Null {
                            continue;
                        }
                        if !item_first {
                            out.push(",")?;
                        }
                        item_first = false;
                        out.push(k)?;
                        out.push(":")?;
                        match k.as_str() {
                            "properties" => {
                                out.push("{")?;
                                if let Value::Object(p) = v {
                                    parameters(p, false, out)?;
                                }
                                out.push("}")?;
                            }
                            "required" => required(v, out)?,
                            "type" => argument(&upper(v), true, out)?,
                            _ => argument(v, true, out)?,
                        }
                    }
                    out.push("}")?;
                }
            }
        }
        if truth(get(value, "nullable")) {
            if comma {
                out.push(",")?;
            }
            comma = true;
            out.push("nullable:true")?;
        }
        if kind == "OBJECT" {
            if comma {
                out.push(",")?;
            }
            comma = true;
            out.push("properties:{")?;
            if let Value::Object(p) = get(value, "properties") {
                parameters(p, false, out)?;
            } else {
                parameters(value, true, out)?;
            }
            out.push("}")?;
            if truth(get(value, "required")) {
                out.push(",required:")?;
                required(get(value, "required"), out)?;
            }
        }
        if comma {
            out.push(",")?;
        }
        out.push("type:")?;
        argument(&Value::String(kind), true, out)?;
        out.push("}")?;
    }
    Ok(())
}
fn declaration(tool: &Value, out: &mut Output) -> Result<()> {
    let f = get(tool.object()?, "function").object()?;
    out.push("declaration:")?;
    out.push(&text(f, "name")?)?;
    out.push("{description:")?;
    argument(&Value::String(text(f, "description")?), true, out)?;
    if truth(get(f, "parameters")) {
        let p = get(f, "parameters").object()?;
        out.push(",parameters:{")?;
        if truth(get(p, "properties")) {
            out.push("properties:{")?;
            parameters(get(p, "properties").object()?, false, out)?;
            out.push("},")?;
        }
        if truth(get(p, "required")) {
            out.push("required:")?;
            required(get(p, "required"), out)?;
            out.push(",")?;
        }
        if truth(get(p, "type")) {
            out.push("type:")?;
            argument(&upper(get(p, "type")), true, out)?;
            out.push("}")?;
        }
    }
    if let Some(response) = f.get("response") {
        let r = response.object()?;
        out.push(",response:{")?;
        if truth(get(r, "description")) {
            out.push("description:")?;
            argument(get(r, "description"), true, out)?;
            out.push(",")?;
        }
        if text(r, "type")?.to_uppercase() == "OBJECT" {
            out.push("type:<|\"|>OBJECT<|\"|>}")?;
        }
    }
    out.push("}")
}
fn strip_thinking(text: &str) -> String {
    text.split("<channel|>")
        .map(|part| part.split("<|channel>").next().unwrap_or(""))
        .collect::<String>()
        .trim()
        .into()
}
/// Exact string-content branch, including thinking, consecutive model turns,
/// function declarations, calls and OpenAI tool responses. Multimodal content
/// and legacy embedded tool_responses are outside the text Message contract.
pub fn format(messages: &[Message], options: &Options) -> Result<String> {
    if messages.len() > 4096 || options.tools.len() > 1024 {
        return Err(err("chat item count limit exceeded"));
    }
    let input_bytes = messages
        .iter()
        .try_fold(0usize, |sum, m| {
            std::iter::once(m.content.as_str())
                .chain(m.reasoning_content.as_deref())
                .chain(
                    m.tool_calls
                        .iter()
                        .flat_map(|t| [t.name.as_str(), t.arguments.as_str()]),
                )
                .try_fold(sum, |sum, s| sum.checked_add(s.len()))
        })
        .and_then(|sum| {
            options
                .tools
                .iter()
                .try_fold(sum, |sum, s| sum.checked_add(s.len()))
        })
        .ok_or_else(|| err("chat input overflow"))?;
    if input_bytes > options.max_bytes {
        return Err(err("chat input byte limit exceeded"));
    }
    let mut out = Output {
        text: String::new(),
        limit: options.max_bytes,
        ordinary: Vec::new(),
    };
    out.push("<bos>")?;
    let mut start = 0;
    if options.enable_thinking
        || !options.tools.is_empty()
        || messages
            .first()
            .is_some_and(|m| matches!(m.role, Role::System | Role::Developer))
    {
        out.push("<|turn>system\n")?;
        if options.enable_thinking {
            out.push("<|think|>\n")?;
        }
        if let Some(m) = messages
            .first()
            .filter(|m| matches!(m.role, Role::System | Role::Developer))
        {
            out.push(m.content.trim())?;
            start = 1;
        }
        for tool in &options.tools {
            out.push("<|tool>")?;
            declaration(
                &ordered::parse(
                    tool.as_bytes(),
                    json::Limits {
                        bytes: options.max_bytes,
                        string_bytes: options.max_bytes,
                        ..Default::default()
                    },
                )?,
                &mut out,
            )?;
            out.push("<tool|>")?;
        }
        out.push("<turn|>\n")?;
    }
    let last_user = messages.iter().rposition(|m| m.role == Role::User);
    let mut previous = None;
    let mut last_type = None;
    for (i, m) in messages.iter().enumerate().skip(start) {
        if m.role == Role::Tool {
            continue;
        }
        last_type = None;
        if !(m.role == Role::Assistant && previous == Some(Role::Assistant)) {
            out.push("<|turn>")?;
            out.push(if m.role == Role::Assistant {
                "model"
            } else {
                m.role.name()
            })?;
            out.push("\n")?;
        }
        if let Some(thinking) = m.reasoning_content.as_deref().filter(|s| !s.is_empty()) {
            if last_user.is_none_or(|last| i > last)
                || (options.preserve_thinking && !m.tool_calls.is_empty())
            {
                out.push("<|channel>thought\n")?;
                out.push(thinking)?;
                out.push("\n<channel|>")?;
            }
        }
        for call in &m.tool_calls {
            out.push("<|tool_call>call:")?;
            out.push(&call.name)?;
            out.push("{")?;
            let args = ordered::parse(
                call.arguments.as_bytes(),
                json::Limits {
                    bytes: options.max_bytes,
                    string_bytes: options.max_bytes,
                    ..Default::default()
                },
            )?;
            if args != Value::Null {
                for (j, (key, value)) in args.object()?.sorted().into_iter().enumerate() {
                    if j > 0 {
                        out.push(",")?;
                    }
                    out.push(key)?;
                    out.push(":")?;
                    argument(value, false, &mut out)?;
                }
            }
            out.push("}<tool_call|>")?;
            last_type = Some("tool_call");
        }
        let mut responses = false;
        if !m.tool_calls.is_empty() {
            for follow in messages[i + 1..]
                .iter()
                .take_while(|m| m.role == Role::Tool)
            {
                let mut name = follow
                    .tool_name
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown");
                for call in &m.tool_calls {
                    if call.id == follow.tool_call_id {
                        name = &call.name;
                    }
                }
                out.push("<|tool_response>response:")?;
                out.push(name)?;
                out.push("{value:")?;
                argument(&Value::String(follow.content.clone()), false, &mut out)?;
                out.push("}<tool_response|>")?;
                responses = true;
                last_type = Some("tool_response");
            }
        }
        let content = if m.role == Role::Assistant {
            strip_thinking(&m.content)
        } else {
            m.content.trim().into()
        };
        out.push(&content)?;
        let next = messages[i + 1..].iter().find(|m| m.role != Role::Tool);
        let continues = m.role == Role::Assistant
            && next.is_some_and(|m| m.role == Role::Assistant)
            && (m.tool_calls.is_empty() || responses);
        if last_type == Some("tool_call") && !responses {
            out.push("<|tool_response>")?;
        } else if !continues && !(responses && content.trim().is_empty() && next.is_none()) {
            out.push("<turn|>\n")?;
        }
        previous = Some(m.role);
    }
    if options.add_generation_prompt {
        match last_type {
            Some("tool_call") => {}
            Some("tool_response") => {
                if options.enable_thinking {
                    out.push("<|channel>thought\n")?;
                }
            }
            _ => {
                out.push("<|turn>model\n")?;
                if !options.enable_thinking {
                    out.push("<|channel>thought\n<channel|>")?;
                }
            }
        }
    }
    Ok(out.text)
}
